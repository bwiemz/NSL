//! Emits the `calibration_main()` entrypoint and links it into a
//! standalone binary the harness driver spawns as a subprocess.
//!
//! **Strategy:**
//!
//! The emitted `calibration_main(argc, argv)`:
//!   1. Validates `argc >= 4`; returns 2 (Infrastructure) otherwise.
//!   2. Reads `argv[1]` = data_path, `argv[2]` = sidecar_path,
//!      `argv[3]` = weights_path (NUL-terminated C-strings from the OS).
//!   3. Loads calibration batches via `nsl_calibration_load`.
//!   4. Creates a runtime model from `argv[3]` and retrieves its weight-pointer array.
//!   5. Loops over batches calling `nsl_calib_model_forward`; each call populates
//!      the retention arena (spliced in by Task 4).
//!   6. For AWQ-style hooks with an `observe_plan`, emits a plan-driven
//!      2D max-abs loop per projection that reads the retention arena and
//!      accumulates into per-projection running-buffer globals.
//!   7. After the batch loop, builds a stack-allocated descriptor array
//!      and calls `nsl_awq_write_sidecar(sidecar_path, descriptors)`.
//!      Propagates non-zero return code.
//!   8. Calls `nsl_calibration_free`; returns 0.
//!
//! For hooks whose `requires()` returns `ObservationSet::Empty` or similar
//! (i.e. `needs_forward_pass() == false`), we still run through the same
//! path but skip the forward-loop and go directly to writing the pre-baked
//! sidecar JSON via `nsl_write_file`.  This is done in-process inside
//! `emit_and_link_calibration_binary` before emitting the Cranelift object,
//! so IdentityHook continues to work.
//!
//! **Subprocess argv convention (Task 6):**
//!   argv[1] = data_path
//!   argv[2] = sidecar_path
//!   argv[3] = weights_path
//!
//! No environment variables are used.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{
    types as cl_types, AbiParam, Function, InstBuilder, MemFlags, StackSlotData, StackSlotKind,
    UserFuncName,
};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, DataId, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};
use object::{Object, ObjectSymbol};

use crate::calibration::ctx::CalibCtx;
use crate::calibration::hooks::{CalibrationResult, FinalizePlanEntry, ObservePlanEntry};
use crate::calibration::registry::HookRegistry;
use crate::calibration::retention_pass::build_arena_layout;
use crate::calibration::sidecar::{Sidecar, SIDECAR_VERSION};
use crate::calibration::subprocess::{run_subprocess, SubprocessOutcome};
use crate::calibration::{HarnessConfig, HarnessError, HarnessOutput};

pub fn real_subprocess_entry(
    cfg: &HarnessConfig,
    registry: &HookRegistry,
) -> Result<HarnessOutput, HarnessError> {
    let tmp = create_tmp_dir()?;
    let _guard = TmpCleanup(tmp.clone());

    let sidecar_path = tmp.join("sidecar.json");

    let needs_forward = registry.iter().any(|h| h.requires().needs_forward_pass());

    // For non-forward hooks: compute sidecar in-process so we can embed the
    // JSON in the binary.
    // For forward hooks: the subprocess writes the sidecar itself via
    // nsl_awq_write_sidecar; we embed a placeholder JSON that is never read.
    let sidecar_json_for_binary = if needs_forward {
        // The subprocess writes its own sidecar via nsl_awq_write_sidecar.
        // Embed a minimal placeholder so the binary emission code has something
        // to pass — it is not used by the forward-pass code path in main().
        b"{}".to_vec()
    } else {
        let sidecar = build_sidecar_from_stub(cfg, registry)?;
        serde_json::to_vec(&sidecar).map_err(|e| HarnessError::Infrastructure {
            reason: format!("serializing sidecar JSON: {e}"),
        })?
    };

    // Determine weights path (use first checkpoint if available).
    let weights_path = cfg
        .checkpoints
        .first()
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_default();

    let forward_batch_seq = if needs_forward {
        Some(
            nsl_runtime::calibration_data::peek_batch_seq(&cfg.calibration_data).map_err(|e| {
                HarnessError::Infrastructure {
                    reason: format!("peek_batch_seq: {e}"),
                }
            })?,
        )
    } else {
        None
    };

    // Build the arena layout and collect observe/finalize plans from hooks.
    let (arena_batch, arena_seq) = forward_batch_seq
        .map(|(_, seq)| (1, seq))
        .unwrap_or((8, 4));
    let arena_layout = build_arena_layout(&cfg.projections, arena_batch, arena_seq);
    let observe_plan: Vec<ObservePlanEntry> = registry
        .iter()
        .flat_map(|h| h.observe_plan(&arena_layout))
        .collect();
    let finalize_plan: Vec<FinalizePlanEntry> = registry
        .iter()
        .flat_map(|h| h.finalize_plan())
        .collect();

    // Emit and link the calibration binary.
    let binary_path = if needs_forward {
        let compile_bundle = cfg
            .compile_bundle
            .clone()
            .ok_or_else(|| HarnessError::Infrastructure {
                reason: "real_subprocess_entry requires compile_bundle for forward-pass calibration"
                    .into(),
            })?;
        let (_count, seq) = forward_batch_seq.expect("forward path computed batch/seq above");
        let mut model_opts = crate::CompileOptions::default();
        model_opts.calibration_batch_seq = Some((1, seq));
        model_opts.calibration_retention = Some(cfg.projections.clone());
        model_opts.calibration_compile_bundle = Some(compile_bundle.clone());

        let model_obj = tmp.join("calib_model.o");
        emit_calibration_model_object(
            &compile_bundle.ast,
            &model_opts,
            &arena_layout,
            &model_obj,
        )?;

        let scaffolding_obj = tmp.join("scaffolding.o");
        emit_calibration_scaffolding_object(
            &observe_plan,
            &finalize_plan,
            &arena_layout,
            &sidecar_json_for_binary,
            true,
            &scaffolding_obj,
        )?;

        let binary_path = tmp.join(if cfg!(windows) {
            "calibration.exe"
        } else {
            "calibration"
        });
        link_calibration_binary(&scaffolding_obj, &model_obj, &binary_path)?;
        binary_path
    } else {
        emit_and_link_calibration_binary(
            &tmp,
            &sidecar_json_for_binary,
            false,
            &observe_plan,
            &finalize_plan,
        )?
    };

    // Spawn the subprocess passing paths as CLI args.
    let data_path_str = cfg.calibration_data.to_string_lossy().into_owned();
    let sidecar_path_str = sidecar_path.to_string_lossy().into_owned();
    let outcome = run_subprocess(
        &binary_path,
        &[&data_path_str, &sidecar_path_str, &weights_path],
        Duration::from_secs(cfg.timeout_secs),
    )
    .map_err(|e| HarnessError::Infrastructure {
        reason: format!("spawn failed: {e}"),
    })?;

    match outcome {
        SubprocessOutcome::Clean => {}
        SubprocessOutcome::Degenerate => {
            return Err(HarnessError::Degenerate {
                hook_id: "<subprocess>".into(),
                reason: "calibration subprocess exited with degenerate status (1)".into(),
            });
        }
        SubprocessOutcome::Infrastructure(reason) => {
            let reason = if reason.contains("status 3") {
                match fs::read_to_string(&sidecar_path) {
                    Ok(detail) if !detail.trim().is_empty() => {
                        format!("{reason}\n{}", detail.trim())
                    }
                    _ => reason,
                }
            } else {
                reason
            };
            return Err(HarnessError::Infrastructure { reason });
        }
    }

    // Read sidecar JSON the subprocess wrote.
    let json = fs::read(&sidecar_path).map_err(|e| HarnessError::Infrastructure {
        reason: format!("reading sidecar {}: {e}", sidecar_path.display()),
    })?;
    let parsed: Sidecar = serde_json::from_slice(&json).map_err(|e| {
        HarnessError::Infrastructure {
            reason: format!("parsing sidecar: {e}"),
        }
    })?;

    Ok(HarnessOutput {
        sidecar: parsed,
        outcome_repr: "clean",
    })
}

/// Run each hook's init/per-step/finalize against a stub ctx and
/// assemble a `Sidecar`.  Only valid for observation-free hooks.
fn build_sidecar_from_stub(
    cfg: &HarnessConfig,
    registry: &HookRegistry,
) -> Result<Sidecar, HarnessError> {
    let mut ctx = CalibCtx::stub_for_tests();
    ctx.total_samples = cfg.samples;

    for hook in registry.iter() {
        hook.emit_init(&mut ctx);
    }
    for i in 0..cfg.samples {
        ctx.sample_idx = i;
        for hook in registry.iter() {
            hook.emit_per_step(&mut ctx);
        }
    }

    let mut hooks_out: BTreeMap<String, Vec<u8>> = BTreeMap::new();
    for hook in registry.iter() {
        match hook.emit_finalize(&mut ctx) {
            CalibrationResult::Ok(bytes) => {
                hooks_out.insert(hook.id().to_string(), bytes);
            }
            CalibrationResult::Degenerate { reason } => {
                return Err(HarnessError::Degenerate {
                    hook_id: hook.id().to_string(),
                    reason,
                });
            }
        }
    }

    Ok(Sidecar {
        version: SIDECAR_VERSION,
        checkpoint_sha256: String::new(),
        calibration_data_sha256: String::new(),
        hook_set_sha256: String::new(),
        cache_key_digest: String::new(),
        num_samples_used: cfg.samples,
        hooks: hooks_out,
        wggo_head_gradients: None,
    })
}

/// Emit a Cranelift IR loop inside `b` that performs a 2-D max-abs reduction
/// over `rows × channels` f32 elements, reading from `arena_base + src_offset`
/// and accumulating into `running_base`.
///
/// Block layout:
///   caller_block → row_header(i=0)
///   row_header(i): i < rows → row_body / row_exit
///   row_body: j=0 → col_header(j)
///   col_header(j): j < channels → col_body / col_exit
///   col_body: abs(arena[i*ch+j]) → fmax → running[j]; j++ → col_header
///   col_exit: i++ → row_header
///   row_exit: (fall through to caller)
///
/// All blocks are sealed here; the caller must NOT seal them.
fn emit_2d_max_abs_loop(
    b: &mut FunctionBuilder,
    arena_base: cranelift_codegen::ir::Value,
    src_offset: u32,
    rows: u32,
    channels: u32,
    running_base: cranelift_codegen::ir::Value,
) {
    let ptr_ty = cl_types::I64;

    let row_header = b.create_block();
    let row_body   = b.create_block();
    let col_header = b.create_block();
    let col_body   = b.create_block();
    let col_exit   = b.create_block();
    let row_exit   = b.create_block();

    b.append_block_param(row_header, cl_types::I32); // i
    b.append_block_param(col_header, cl_types::I32); // j

    let zero_i32 = b.ins().iconst(cl_types::I32, 0);
    b.ins().jump(row_header, &[zero_i32]);

    // row_header: if i < rows → row_body else row_exit
    b.switch_to_block(row_header);
    let i = b.block_params(row_header)[0];
    let rows_v = b.ins().iconst(cl_types::I32, rows as i64);
    let cmp_r = b.ins().icmp(IntCC::UnsignedLessThan, i, rows_v);
    b.ins().brif(cmp_r, row_body, &[], row_exit, &[]);

    // row_body: jump to col_header(j=0)
    b.switch_to_block(row_body);
    b.seal_block(row_body);
    b.ins().jump(col_header, &[zero_i32]);

    // col_header: if j < channels → col_body else col_exit
    b.switch_to_block(col_header);
    let j = b.block_params(col_header)[0];
    let ch_v = b.ins().iconst(cl_types::I32, channels as i64);
    let cmp_c = b.ins().icmp(IntCC::UnsignedLessThan, j, ch_v);
    b.ins().brif(cmp_c, col_body, &[], col_exit, &[]);

    // col_body: compute source and running addresses, do fmax(running, |src|)
    b.switch_to_block(col_body);
    b.seal_block(col_body);

    // src_addr = arena_base + src_offset + (i * channels + j) * 4
    let i_ch  = b.ins().imul(i, ch_v);
    let lin   = b.ins().iadd(i_ch, j);
    let four  = b.ins().iconst(cl_types::I32, 4);
    let lin4  = b.ins().imul(lin, four);
    let soff  = b.ins().iconst(cl_types::I32, src_offset as i64);
    let soff_t = b.ins().iadd(lin4, soff);
    let soff_p = b.ins().uextend(ptr_ty, soff_t);
    let src_addr = b.ins().iadd(arena_base, soff_p);

    // run_addr = running_base + j * 4
    let j_off = b.ins().imul(j, four);
    let j_off_p = b.ins().uextend(ptr_ty, j_off);
    let run_addr = b.ins().iadd(running_base, j_off_p);

    let v    = b.ins().load(cl_types::F32, MemFlags::new(), src_addr, 0);
    let absv = b.ins().fabs(v);
    let cur  = b.ins().load(cl_types::F32, MemFlags::new(), run_addr, 0);
    let new  = b.ins().fmax(cur, absv);
    b.ins().store(MemFlags::new(), new, run_addr, 0);

    let jp1 = b.ins().iadd_imm(j, 1);
    b.ins().jump(col_header, &[jp1]);

    // col_exit: i++ → row_header
    b.switch_to_block(col_exit);
    b.seal_block(col_exit);
    let ip1 = b.ins().iadd_imm(i, 1);
    b.ins().jump(row_header, &[ip1]);

    // seal the back-edge blocks now that all predecessors are known
    b.seal_block(row_header);
    b.seal_block(col_header);

    b.switch_to_block(row_exit);
    b.seal_block(row_exit);
    // caller continues from row_exit
}

/// Size of `AwqProjectionDescriptor` on 64-bit platforms (matching `#[repr(C)]`):
///   path_ptr:    *const u8  = 8 bytes
///   path_len:    usize      = 8 bytes
///   channels:    u32        = 4 bytes
///   _pad:        u32        = 4 bytes
///   running_ptr: *const f32 = 8 bytes
///   Total                   = 32 bytes
const AWQ_DESC_BYTES: u32 = 32;

fn new_calibration_object_module(
    name: &str,
) -> Result<(ObjectModule, cranelift_codegen::isa::CallConv), HarnessError> {
    let flag_builder = cranelift_codegen::settings::builder();
    let isa = cranelift_native::builder()
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("cranelift native builder: {e}"),
        })?
        .finish(cranelift_codegen::settings::Flags::new(flag_builder))
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("cranelift ISA: {e}"),
        })?;

    let call_conv = {
        let detected = isa.default_call_conv();
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        {
            if detected != cranelift_codegen::isa::CallConv::WindowsFastcall {
                cranelift_codegen::isa::CallConv::WindowsFastcall
            } else {
                detected
            }
        }
        #[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
        {
            detected
        }
    };

    let builder = ObjectBuilder::new(
        isa,
        name,
        cranelift_module::default_libcall_names(),
    )
    .map_err(|e| HarnessError::Infrastructure {
        reason: format!("object builder: {e}"),
    })?;

    Ok((ObjectModule::new(builder), call_conv))
}

fn write_object_module(module: ObjectModule, out_path: &Path) -> Result<(), HarnessError> {
    let product = module.finish();
    let obj_bytes = product.emit().map_err(|e| HarnessError::Infrastructure {
        reason: format!("emit object: {e}"),
    })?;
    fs::write(out_path, &obj_bytes).map_err(|e| HarnessError::Infrastructure {
        reason: format!("writing {}: {e}", out_path.display()),
    })
}

fn map_codegen_error<T>(
    context: &str,
    result: Result<T, crate::error::CodegenError>,
) -> Result<T, HarnessError> {
    result.map_err(|e| HarnessError::Infrastructure {
        reason: format!("{context}: {}", e.message),
    })
}

fn awq_model_def_from_ast<'a>(
    ast: &'a nsl_ast::Module,
    interner: &nsl_lexer::Interner,
    model_name: &str,
) -> Option<&'a nsl_ast::decl::ModelDef> {
    ast.stmts.iter().find_map(|stmt| {
        let md = match &stmt.kind {
            nsl_ast::stmt::StmtKind::ModelDef(md) => md,
            nsl_ast::stmt::StmtKind::Decorated { stmt: inner, .. } => match &inner.kind {
                nsl_ast::stmt::StmtKind::ModelDef(md) => md,
                _ => return None,
            },
            _ => return None,
        };
        (interner.resolve(md.name.0) == Some(model_name)).then_some(md)
    })
}

fn awq_tensor_field_names(
    model_def: &nsl_ast::decl::ModelDef,
    interner: &nsl_lexer::Interner,
) -> Vec<String> {
    model_def
        .members
        .iter()
        .filter_map(|member| match member {
            nsl_ast::decl::ModelMember::LayerDecl { name, type_ann, .. } => {
                match &type_ann.kind {
                    nsl_ast::types::TypeExprKind::Named(sym)
                        if interner.resolve(sym.0) == Some("Tensor") =>
                    {
                        interner.resolve(name.0).map(str::to_string)
                    }
                    _ => None,
                }
            }
            _ => None,
        })
        .collect()
}

fn declare_i64_rodata(
    module: &mut ObjectModule,
    symbol: &str,
    values: &[i64],
) -> Result<DataId, HarnessError> {
    let mut bytes = Vec::with_capacity(std::mem::size_of_val(values));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    let data_id = module
        .declare_data(symbol, Linkage::Local, false, false)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare {symbol}: {e}"),
        })?;
    let mut desc = DataDescription::new();
    desc.define(bytes.into_boxed_slice());
    module
        .define_data(data_id, &desc)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("define {symbol}: {e}"),
        })?;
    Ok(data_id)
}

fn declare_cstr_rodata(
    module: &mut ObjectModule,
    symbol: &str,
    value: &str,
) -> Result<DataId, HarnessError> {
    if value.as_bytes().contains(&0) {
        return Err(HarnessError::Infrastructure {
            reason: format!("declare {symbol}: embedded NUL byte in C string literal"),
        });
    }
    let data_id = module
        .declare_data(symbol, Linkage::Local, false, false)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare {symbol}: {e}"),
        })?;
    let mut bytes = value.as_bytes().to_vec();
    bytes.push(0);
    let mut desc = DataDescription::new();
    desc.define(bytes.into_boxed_slice());
    module
        .define_data(data_id, &desc)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("define {symbol}: {e}"),
        })?;
    Ok(data_id)
}

fn emit_model_forward_bridge(
    compiler: &mut crate::compiler::Compiler<'_>,
    model_name: &str,
    model_def: &nsl_ast::decl::ModelDef,
    transpose_fields: &HashSet<String>,
) -> Result<cranelift_module::FuncId, HarnessError> {
    let mut model_get_weight_sig = compiler.module.make_signature();
    model_get_weight_sig.call_conv = compiler.call_conv;
    model_get_weight_sig.params.push(AbiParam::new(cl_types::I64));
    model_get_weight_sig.params.push(AbiParam::new(cl_types::I64));
    model_get_weight_sig.params.push(AbiParam::new(cl_types::I64));
    model_get_weight_sig.returns.push(AbiParam::new(cl_types::I64));
    let model_get_weight_id = compiler
        .module
        .declare_function(
            "nsl_model_get_weight",
            Linkage::Import,
            &model_get_weight_sig,
        )
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_model_get_weight: {e}"),
        })?;

    let mut tensor_transpose_sig = compiler.module.make_signature();
    tensor_transpose_sig.call_conv = compiler.call_conv;
    tensor_transpose_sig.params.push(AbiParam::new(cl_types::I64));
    tensor_transpose_sig.params.push(AbiParam::new(cl_types::I64));
    tensor_transpose_sig.params.push(AbiParam::new(cl_types::I64));
    tensor_transpose_sig.returns.push(AbiParam::new(cl_types::I64));
    let tensor_transpose_id = compiler
        .module
        .declare_function("nsl_tensor_transpose", Linkage::Import, &tensor_transpose_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_tensor_transpose: {e}"),
        })?;

    let mut sig = compiler.module.make_signature();
    sig.call_conv = compiler.call_conv;
    sig.params.push(AbiParam::new(cl_types::I64));
    sig.params.push(AbiParam::new(cl_types::I64));
    sig.params.push(AbiParam::new(cl_types::I64));
    let func_id = compiler
        .module
        .declare_function("model_forward", Linkage::Export, &sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare model_forward: {e}"),
        })?;

    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, compiler.next_func_index()),
        sig,
    ));
    let mut fn_builder_ctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);

        let model_ptr = b.block_params(entry)[0];
        let _num_weights = b.block_params(entry)[1];
        let input_tensor = b.block_params(entry)[2];

        let layout = compiler
            .types
            .struct_layouts
            .get(model_name)
            .cloned()
            .ok_or_else(|| HarnessError::Infrastructure {
                reason: format!("missing struct layout for model '{model_name}'"),
            })?;

        let model_slot = b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            layout.total_size.max(8) as u32,
            3,
        ));
        let model_addr = b.ins().stack_addr(cl_types::I64, model_slot, 0);

        for field in &layout.fields {
            let zero = if field.cl_type == cl_types::F64 {
                b.ins().f64const(0.0)
            } else if field.cl_type == cl_types::F32 {
                b.ins().f32const(0.0)
            } else {
                b.ins().iconst(field.cl_type, 0)
            };
            b.ins()
                .store(MemFlags::trusted(), zero, model_addr, field.offset as i32);
        }
        if let Some(slot_off) = layout.adapter_sidetable_offset {
            let zero = b.ins().iconst(cl_types::I64, 0);
            b.ins()
                .store(MemFlags::trusted(), zero, model_addr, slot_off as i32);
        }

        let tensor_fields = awq_tensor_field_names(model_def, compiler.interner);
        let model_get_weight_ref = compiler
            .module
            .declare_func_in_func(model_get_weight_id, b.func);
        let tensor_transpose_ref = compiler
            .module
            .declare_func_in_func(tensor_transpose_id, b.func);
        let zero_i64 = b.ins().iconst(cl_types::I64, 0);
        let one_i64 = b.ins().iconst(cl_types::I64, 1);
        let mut transposed_tensors = Vec::new();
        for field_name in &tensor_fields {
            let Some(field) = layout.fields.iter().find(|field| field.name == *field_name) else {
                continue;
            };
            let qualified_name = format!("{model_name}.{field_name}");
            let symbol = format!(
                "__nsl_calib_weight_name_{}_{}",
                model_name,
                field_name
            );
            let name_data = declare_cstr_rodata(&mut compiler.module, &symbol, &qualified_name)?;
            let name_ref = compiler.module.declare_data_in_func(name_data, b.func);
            let name_ptr = b.ins().symbol_value(cl_types::I64, name_ref);
            let name_len = b.ins().iconst(cl_types::I64, qualified_name.len() as i64);
            let tensor_call = b.ins().call(model_get_weight_ref, &[model_ptr, name_ptr, name_len]);
            let mut tensor_ptr = b.inst_results(tensor_call)[0];
            let should_transpose = transpose_fields.contains(field_name)
                || compiler
                    .models
                    .model_tensor_field_shapes
                    .get(model_name)
                    .is_some_and(|fields| fields.contains_key(field_name));
            if should_transpose {
                let transpose_call = b.ins().call(tensor_transpose_ref, &[tensor_ptr, zero_i64, one_i64]);
                tensor_ptr = b.inst_results(transpose_call)[0];
                transposed_tensors.push(tensor_ptr);
            }
            b.ins().store(
                MemFlags::trusted(),
                tensor_ptr,
                model_addr,
                field.offset as i32,
            );
        }

        let method_mangled = compiler
            .models
            .model_methods
            .get(model_name)
            .and_then(|methods| methods.get("forward"))
            .cloned()
            .ok_or_else(|| HarnessError::Infrastructure {
                reason: format!("model '{model_name}' has no forward method"),
            })?;
        let (forward_id, forward_sig) = compiler
            .registry
            .functions
            .get(&method_mangled)
            .cloned()
            .ok_or_else(|| HarnessError::Infrastructure {
                reason: format!("forward impl '{method_mangled}' not registered"),
            })?;
        if forward_sig.params.len() != 2 {
            return Err(HarnessError::Infrastructure {
                reason: format!(
                    "model '{model_name}' forward must take exactly one non-self argument"
                ),
            });
        }
        let forward_ref = compiler.module.declare_func_in_func(forward_id, b.func);
        let forward_call = b.ins().call(forward_ref, &[model_addr, input_tensor]);

        if !forward_sig.returns.is_empty() {
            let output_tensor = b.inst_results(forward_call)[0];
            let tensor_free_id = compiler.registry.runtime_fns["nsl_tensor_free"].0;
            let tensor_free_ref = compiler.module.declare_func_in_func(tensor_free_id, b.func);
            b.ins().call(tensor_free_ref, &[output_tensor]);
            for tensor_ptr in transposed_tensors {
                b.ins().call(tensor_free_ref, &[tensor_ptr]);
            }
        } else {
            let tensor_free_id = compiler.registry.runtime_fns["nsl_tensor_free"].0;
            let tensor_free_ref = compiler.module.declare_func_in_func(tensor_free_id, b.func);
            for tensor_ptr in transposed_tensors {
                b.ins().call(tensor_free_ref, &[tensor_ptr]);
            }
        }

        b.ins().return_(&[]);
        b.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut ctx)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("define model_forward: {e}"),
        })?;
    Ok(func_id)
}

fn emit_calibration_forward_wrapper(
    compiler: &mut crate::compiler::Compiler<'_>,
    model_forward_id: cranelift_module::FuncId,
    batch: u32,
    seq: u32,
    channels: i64,
) -> Result<(), HarnessError> {
    let (shape_vals, input_ndim) = if batch == 1 {
        (vec![seq as i64, channels], 2_i32)
    } else {
        (vec![batch as i64, seq as i64, channels], 3_i32)
    };
    let shape_data = declare_i64_rodata(
        &mut compiler.module,
        "__nsl_calib_input_shape",
        &shape_vals,
    )?;

    let mut desc_to_tensor_sig = compiler.module.make_signature();
    desc_to_tensor_sig.call_conv = compiler.call_conv;
    desc_to_tensor_sig.params.push(AbiParam::new(cl_types::I64));
    desc_to_tensor_sig.returns.push(AbiParam::new(cl_types::I64));
    let desc_to_tensor_id = compiler
        .module
        .declare_function("nsl_desc_to_tensor", Linkage::Import, &desc_to_tensor_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_desc_to_tensor: {e}"),
        })?;

    // v2 follow-up (spec §5.1): wrapper returns i32 status so the
    // caller can detect per-batch shape mismatch mid-loop, not just the
    // first-batch preflight in `batch_preflight`. 0 = ok, 3 = mismatch
    // (matches the subprocess exit code for §5.1).
    let mut wrapper_sig = compiler.module.make_signature();
    wrapper_sig.call_conv = compiler.call_conv;
    wrapper_sig.params.push(AbiParam::new(cl_types::I64));
    wrapper_sig.params.push(AbiParam::new(cl_types::I64));
    wrapper_sig.params.push(AbiParam::new(cl_types::I64));
    wrapper_sig.params.push(AbiParam::new(cl_types::I64));
    wrapper_sig.returns.push(AbiParam::new(cl_types::I32));
    let wrapper_id = compiler
        .module
        .declare_function("nsl_calib_model_forward", Linkage::Export, &wrapper_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calib_model_forward: {e}"),
        })?;

    // Compile-time element count derived from the same shape array we
    // wrote into the rodata; both descriptor and check share one source
    // of truth, so a future shape change can't desync them.
    let expected_elem_count: i64 = shape_vals.iter().product();

    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, compiler.next_func_index()),
        wrapper_sig,
    ));
    let mut fn_builder_ctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);

        let weight_ptrs = b.block_params(entry)[0];
        let num_weights = b.block_params(entry)[1];
        let batch_ptr = b.block_params(entry)[2];
        let batch_elem_count = b.block_params(entry)[3];

        // Per-batch shape check: refuse with status 3 if the runtime
        // batch element count differs from the compile-time expected.
        // Defends against `nsl_calibration_batch_at` returning a
        // variable-size view on a heterogeneous calibration dataset —
        // the scaffolding's `batch_preflight` only validates batch 0.
        let shape_ok = b.create_block();
        let shape_mismatch = b.create_block();
        let expected_v = b.ins().iconst(cl_types::I64, expected_elem_count);
        let shape_eq = b.ins().icmp(IntCC::Equal, batch_elem_count, expected_v);
        b.ins().brif(shape_eq, shape_ok, &[], shape_mismatch, &[]);

        b.switch_to_block(shape_mismatch);
        b.seal_block(shape_mismatch);
        let three = b.ins().iconst(cl_types::I32, 3);
        b.ins().return_(&[three]);

        b.switch_to_block(shape_ok);
        b.seal_block(shape_ok);

        let desc_slot = b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            40,
            3,
        ));
        let desc_addr = b.ins().stack_addr(cl_types::I64, desc_slot, 0);
        let zero_i64 = b.ins().iconst(cl_types::I64, 0);
        let zero_i32 = b.ins().iconst(cl_types::I32, 0);
        let ndim_i32 = b.ins().iconst(cl_types::I32, input_ndim as i64);

        let shape_ref = compiler.module.declare_data_in_func(shape_data, b.func);
        let shape_ptr = b.ins().symbol_value(cl_types::I64, shape_ref);

        b.ins().store(MemFlags::trusted(), batch_ptr, desc_addr, 0);
        b.ins().store(MemFlags::trusted(), shape_ptr, desc_addr, 8);
        b.ins().store(MemFlags::trusted(), zero_i64, desc_addr, 16);
        b.ins().store(MemFlags::trusted(), ndim_i32, desc_addr, 24);
        b.ins().store(MemFlags::trusted(), zero_i32, desc_addr, 28);
        b.ins().store(MemFlags::trusted(), zero_i32, desc_addr, 32);
        b.ins().store(MemFlags::trusted(), zero_i32, desc_addr, 36);

        let desc_to_tensor_ref = compiler.module.declare_func_in_func(desc_to_tensor_id, b.func);
        let desc_call = b.ins().call(desc_to_tensor_ref, &[desc_addr]);
        let input_tensor = b.inst_results(desc_call)[0];

        let model_forward_ref = compiler.module.declare_func_in_func(model_forward_id, b.func);
        b.ins()
            .call(model_forward_ref, &[weight_ptrs, num_weights, input_tensor]);

        let tensor_free_id = compiler.registry.runtime_fns["nsl_tensor_free"].0;
        let tensor_free_ref = compiler.module.declare_func_in_func(tensor_free_id, b.func);
        b.ins().call(tensor_free_ref, &[input_tensor]);
        b.ins().return_(&[zero_i32]);
        b.finalize();
    }

    compiler
        .module
        .define_function(wrapper_id, &mut ctx)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("define nsl_calib_model_forward: {e}"),
        })?;
    Ok(())
}

pub fn emit_calibration_model_object(
    ast: &nsl_ast::Module,
    opts: &crate::CompileOptions,
    _arena_layout: &crate::calibration::ArenaLayout,
    out_path: &Path,
) -> Result<(), HarnessError> {
    let bundle = opts
        .calibration_compile_bundle
        .as_ref()
        .ok_or_else(|| HarnessError::Infrastructure {
            reason: "emit_calibration_model_object requires calibration_compile_bundle"
                .into(),
        })?;

    let mut compile_opts = opts.clone();
    if compile_opts.calibration_retention.is_none() {
        let discovered =
            crate::calibration::pre_scan_awq_projections_from_ast(ast, &bundle.interner);
        if !discovered.is_empty() {
            compile_opts.calibration_retention = Some(discovered);
        }
    }
    let projections = compile_opts
        .calibration_retention
        .clone()
        .ok_or_else(|| HarnessError::Infrastructure {
            reason: "emit_calibration_model_object requires AWQ projections".into(),
        })?;
    let first_projection = projections.first().ok_or_else(|| HarnessError::Infrastructure {
        reason: "emit_calibration_model_object requires a non-empty projection list".into(),
    })?;
    let channels = first_projection.weight_shape[1] as i64;
    let (batch, seq) = compile_opts.calibration_batch_seq.unwrap_or((8, 4));
    let model_name = first_projection
        .projection
        .0
        .split('.')
        .next()
        .ok_or_else(|| HarnessError::Infrastructure {
            reason: "invalid projection path in calibration retention".into(),
        })?
        .to_string();
    let transpose_fields: HashSet<String> = projections
        .iter()
        .filter_map(|projection| {
            projection
                .projection
                .0
                .strip_prefix(&(model_name.clone() + "."))
                .map(str::to_string)
        })
        .collect();
    let model_def = awq_model_def_from_ast(ast, &bundle.interner, &model_name).ok_or_else(|| {
        HarnessError::Infrastructure {
            reason: format!("cannot find AWQ model '{model_name}' in AST"),
        }
    })?;
    let model_only_stmts: Vec<nsl_ast::stmt::Stmt> = bundle
        .ast
        .stmts
        .iter()
        .filter(|stmt| match &stmt.kind {
            nsl_ast::stmt::StmtKind::ModelDef(md) => {
                bundle.interner.resolve(md.name.0) == Some(model_name.as_str())
            }
            nsl_ast::stmt::StmtKind::Decorated { stmt: inner, .. } => match &inner.kind {
                nsl_ast::stmt::StmtKind::ModelDef(md) => {
                    bundle.interner.resolve(md.name.0) == Some(model_name.as_str())
                }
                _ => false,
            },
            _ => false,
        })
        .cloned()
        .collect();

    let (module, call_conv) = new_calibration_object_module("nsl_calibration_model")?;
    let mut compiler = map_codegen_error(
        "Compiler::new(calibration model object)",
        crate::compiler::Compiler::new(&bundle.interner, &bundle.type_map, &compile_opts),
    )?;
    compiler.module = module;
    compiler.call_conv = call_conv;

    map_codegen_error("intern empty string", compiler.intern_string(""))?;
    map_codegen_error("collect strings", compiler.collect_strings(&bundle.ast.stmts))?;
    map_codegen_error("collect enums", compiler.collect_enums(&bundle.ast.stmts))?;
    map_codegen_error("collect structs", compiler.collect_structs(&bundle.ast.stmts))?;
    map_codegen_error("collect models", compiler.collect_models(&bundle.ast.stmts))?;
    map_codegen_error("emit retention arena", compiler.emit_retention_arena())?;
    // Task 10: backward (WGGO grad) sibling arena — spec §7.2 ordering invariant #2.
    map_codegen_error(
        "emit grad retention arena",
        compiler.emit_grad_retention_arena(),
    )?;
    map_codegen_error("declare runtime functions", compiler.declare_runtime_functions())?;
    map_codegen_error(
        "declare user functions",
        compiler.declare_user_functions(&model_only_stmts),
    )?;
    let vmap_results = compiler.apply_vmap_transforms(&bundle.ast);
    compiler.register_batched_functions(&vmap_results);
    map_codegen_error(
        "compile datatype defs",
        compiler.compile_datatype_defs(&bundle.ast.stmts),
    )?;
    map_codegen_error("compile kernels", compiler.compile_kernels(&bundle.ast.stmts))?;
    map_codegen_error(
        "compile flash-attention kernels",
        compiler.compile_flash_attention_kernels(&bundle.ast.stmts),
    )?;
    crate::wrga_prescan::prescan_adapter_sites_from_decorators(&mut compiler);
    crate::wrga_prescan::rewrite_model_method_bodies_with_adapter_sites(&mut compiler);
    map_codegen_error(
        "compile user functions",
        compiler.compile_user_functions(&model_only_stmts),
    )?;
    map_codegen_error(
        "compile batched functions",
        compiler.compile_batched_functions(&vmap_results),
    )?;
    map_codegen_error(
        "compile pending lambdas",
        compiler.compile_pending_lambdas(),
    )?;

    let model_forward_id = emit_model_forward_bridge(
        &mut compiler,
        &model_name,
        model_def,
        &transpose_fields,
    )?;
    emit_calibration_forward_wrapper(&mut compiler, model_forward_id, batch, seq, channels)?;

    let obj_bytes = map_codegen_error("finalize calib_model.o", compiler.finalize())?;
    fs::write(out_path, &obj_bytes).map_err(|e| HarnessError::Infrastructure {
        reason: format!("writing {}: {e}", out_path.display()),
    })
}

pub fn emit_calibration_scaffolding_object(
    observe_plan: &[ObservePlanEntry],
    finalize_plan: &[FinalizePlanEntry],
    arena_layout: &crate::calibration::ArenaLayout,
    sidecar_json: &[u8],
    needs_forward_pass: bool,
    out_path: &Path,
) -> Result<(), HarnessError> {
    if needs_forward_pass && observe_plan.is_empty() {
        return Err(HarnessError::Infrastructure {
            reason: "calibration: forward pass emitted but no observations declared.\n\
 requested:  run calibration subprocess with forward pass\n\
 expected:   observe_plan is non-empty when a model_forward call is emitted\n\
 found:      observe_plan is empty but needs_forward_pass() returned true.\n\
             Did a hook's requires() change without updating observe_plan()?"
                .into(),
        });
    }

    if !needs_forward_pass && sidecar_json.contains(&0u8) {
        return Err(HarnessError::Infrastructure {
            reason: "sidecar JSON contains embedded NUL byte".into(),
        });
    }

    let (mut module, call_conv) = new_calibration_object_module("nsl_calibration")?;
    let ptr_ty = cl_types::I64;

    let mut json_bytes: Vec<u8> = sidecar_json.to_vec();
    json_bytes.push(0);
    let json_data = module
        .declare_data("__nsl_calibration_sidecar_json", Linkage::Local, false, false)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare sidecar-json data: {e}"),
        })?;
    let mut json_desc = DataDescription::new();
    json_desc.define(json_bytes.into_boxed_slice());
    module
        .define_data(json_data, &json_desc)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("define sidecar-json data: {e}"),
        })?;

    let batch_shape_mismatch_data = if needs_forward_pass {
        let first_entry = arena_layout
            .entries
            .first()
            .ok_or_else(|| HarnessError::Infrastructure {
                reason: "calibration: forward pass emitted but arena layout is empty".into(),
            })?;
        let expected_bytes = first_entry.2 as u64;
        let expected_elems = expected_bytes / 4;
        Some(declare_cstr_rodata(
            &mut module,
            "__nsl_calibration_batch_shape_mismatch",
            &format!(
                "calibration: batch shape mismatch at subprocess model-forward call.\n\
 requested:  run calibration forward on a batch matching the emitted arena layout\n\
 expected:   flattened batch payload == {expected_elems} f32 elements ({expected_bytes} bytes)\n\
 found:      nsl_calibration_batch_at returned a batch payload whose size did not match the emitted arena layout.\n\
 Action:     regenerate calibration metadata from the same model/data pair, or fix the projection shape metadata passed into calibration."
            ),
        )?)
    } else {
        None
    };

    let mut running_data_ids: HashMap<String, DataId> = HashMap::new();
    for entry in finalize_plan {
        let mut data = DataDescription::new();
        data.define_zeroinit((entry.channels * 4) as usize);
        let id = module
            .declare_data(&entry.running_symbol, Linkage::Export, true, false)
            .map_err(|e| HarnessError::Infrastructure {
                reason: format!("declare running buf '{}': {e}", entry.running_symbol),
            })?;
        module
            .define_data(id, &data)
            .map_err(|e| HarnessError::Infrastructure {
                reason: format!("define running buf '{}': {e}", entry.running_symbol),
            })?;
        running_data_ids.insert(entry.running_symbol.clone(), id);
    }

    let arena_data_id: Option<DataId> = if !observe_plan.is_empty() {
        let id = module
            .declare_data("__nsl_calib_retention_arena", Linkage::Import, false, false)
            .map_err(|e| HarnessError::Infrastructure {
                reason: format!("declare retention arena import: {e}"),
            })?;
        Some(id)
    } else {
        None
    };

    let mut path_data_ids: HashMap<String, DataId> = HashMap::new();
    for entry in finalize_plan {
        let sym = format!(
            "__nsl_awq_pathstr.{}",
            entry.projection.0.replace(['.', '-'], "_")
        );
        let mut data = DataDescription::new();
        data.define(entry.projection.0.as_bytes().to_vec().into_boxed_slice());
        let id = module
            .declare_data(&sym, Linkage::Local, false, false)
            .map_err(|e| HarnessError::Infrastructure {
                reason: format!("declare path string '{}': {e}", sym),
            })?;
        module
            .define_data(id, &data)
            .map_err(|e| HarnessError::Infrastructure {
                reason: format!("define path string '{}': {e}", sym),
            })?;
        path_data_ids.insert(entry.projection.0.clone(), id);
    }

    let mut strlen_sig = module.make_signature();
    strlen_sig.call_conv = call_conv;
    strlen_sig.params.push(AbiParam::new(cl_types::I64));
    strlen_sig.returns.push(AbiParam::new(cl_types::I64));
    let strlen_id = module
        .declare_function("strlen", Linkage::Import, &strlen_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare strlen: {e}"),
        })?;

    let mut write_file_sig = module.make_signature();
    write_file_sig.call_conv = call_conv;
    write_file_sig.params.push(AbiParam::new(cl_types::I64));
    write_file_sig.params.push(AbiParam::new(cl_types::I64));
    let write_file_id = module
        .declare_function("nsl_write_file", Linkage::Import, &write_file_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_write_file: {e}"),
        })?;

    let mut calib_load_sig = module.make_signature();
    calib_load_sig.call_conv = call_conv;
    calib_load_sig.params.push(AbiParam::new(cl_types::I64));
    calib_load_sig.params.push(AbiParam::new(cl_types::I64));
    calib_load_sig.returns.push(AbiParam::new(cl_types::I64));
    let calib_load_id = module
        .declare_function("nsl_calibration_load", Linkage::Import, &calib_load_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calibration_load: {e}"),
        })?;

    let mut calib_count_sig = module.make_signature();
    calib_count_sig.call_conv = call_conv;
    calib_count_sig.params.push(AbiParam::new(cl_types::I64));
    calib_count_sig.returns.push(AbiParam::new(cl_types::I64));
    let calib_count_id = module
        .declare_function("nsl_calibration_count", Linkage::Import, &calib_count_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calibration_count: {e}"),
        })?;

    let mut calib_batch_sig = module.make_signature();
    calib_batch_sig.call_conv = call_conv;
    calib_batch_sig.params.push(AbiParam::new(cl_types::I64));
    calib_batch_sig.params.push(AbiParam::new(cl_types::I64));
    calib_batch_sig.params.push(AbiParam::new(cl_types::I64));
    calib_batch_sig.params.push(AbiParam::new(cl_types::I64));
    let calib_batch_id = module
        .declare_function("nsl_calibration_batch_at", Linkage::Import, &calib_batch_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calibration_batch_at: {e}"),
        })?;

    let mut calib_free_sig = module.make_signature();
    calib_free_sig.call_conv = call_conv;
    calib_free_sig.params.push(AbiParam::new(cl_types::I64));
    let _calib_free_id = module
        .declare_function("nsl_calibration_free", Linkage::Import, &calib_free_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calibration_free: {e}"),
        })?;

    let mut write_sidecar_sig = module.make_signature();
    write_sidecar_sig.call_conv = call_conv;
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty));
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty));
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty));
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty));
    write_sidecar_sig.returns.push(AbiParam::new(cl_types::I32));
    let write_sidecar_id = module
        .declare_function("nsl_awq_write_sidecar", Linkage::Import, &write_sidecar_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_awq_write_sidecar: {e}"),
        })?;

    let mut model_create_sig = module.make_signature();
    model_create_sig.call_conv = call_conv;
    model_create_sig.params.push(AbiParam::new(ptr_ty));
    model_create_sig.returns.push(AbiParam::new(ptr_ty));
    let model_create_id = module
        .declare_function("nsl_model_create", Linkage::Import, &model_create_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_model_create: {e}"),
        })?;

    let mut model_destroy_sig = module.make_signature();
    model_destroy_sig.call_conv = call_conv;
    model_destroy_sig.params.push(AbiParam::new(ptr_ty));
    model_destroy_sig.returns.push(AbiParam::new(cl_types::I64));
    let model_destroy_id = module
        .declare_function("nsl_model_destroy", Linkage::Import, &model_destroy_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_model_destroy: {e}"),
        })?;

    let mut model_num_weights_sig = module.make_signature();
    model_num_weights_sig.call_conv = call_conv;
    model_num_weights_sig.params.push(AbiParam::new(ptr_ty));
    model_num_weights_sig.returns.push(AbiParam::new(cl_types::I64));
    let model_num_weights_id = module
        .declare_function(
            "nsl_model_get_num_weights",
            Linkage::Import,
            &model_num_weights_sig,
        )
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_model_get_num_weights: {e}"),
        })?;

    let mut model_weight_ptrs_sig = module.make_signature();
    model_weight_ptrs_sig.call_conv = call_conv;
    model_weight_ptrs_sig.params.push(AbiParam::new(ptr_ty));
    model_weight_ptrs_sig.returns.push(AbiParam::new(ptr_ty));
    let model_weight_ptrs_id = module
        .declare_function(
            "nsl_model_get_weight_ptrs",
            Linkage::Import,
            &model_weight_ptrs_sig,
        )
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_model_get_weight_ptrs: {e}"),
        })?;

    // v2 follow-up (spec §5.1): wrapper returns i32 status (0=ok, 3=batch
    // shape mismatch). Scaffolding must declare the same signature or the
    // ABI desyncs at link time (return-register vs caller-clobber).
    let mut calib_model_forward_sig = module.make_signature();
    calib_model_forward_sig.call_conv = call_conv;
    calib_model_forward_sig.params.push(AbiParam::new(ptr_ty));
    calib_model_forward_sig.params.push(AbiParam::new(cl_types::I64));
    calib_model_forward_sig.params.push(AbiParam::new(ptr_ty));
    calib_model_forward_sig.params.push(AbiParam::new(cl_types::I64));
    calib_model_forward_sig.returns.push(AbiParam::new(cl_types::I32));
    let calib_model_forward_id = module
        .declare_function(
            "nsl_calib_model_forward",
            Linkage::Import,
            &calib_model_forward_sig,
        )
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calib_model_forward: {e}"),
        })?;

    let mut main_sig = module.make_signature();
    main_sig.call_conv = call_conv;
    main_sig.params.push(AbiParam::new(cl_types::I32));
    main_sig.params.push(AbiParam::new(cl_types::I64));
    main_sig.returns.push(AbiParam::new(cl_types::I32));
    let main_id = module
        .declare_function("main", Linkage::Export, &main_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare main: {e}"),
        })?;

    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, 0),
        main_sig,
    ));
    let mut fn_builder_ctx = FunctionBuilderContext::new();

    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

        let entry = b.create_block();
        let argc_err = b.create_block();
        let no_forward_write = b.create_block();
        let load_data = b.create_block();
        let data_null = b.create_block();
        let load_model = b.create_block();
        let model_create_entry = b.create_block();
        let batch_preflight = b.create_block();
        let batch_shape_err = b.create_block();
        let model_ready = b.create_block();
        let model_null = b.create_block();
        let loop_header = b.create_block();
        let loop_body = b.create_block();
        let loop_exit = b.create_block();
        let finalize = b.create_block();
        let write_ok = b.create_block();
        let write_err = b.create_block();

        b.append_block_params_for_function_params(entry);
        b.append_block_param(load_model, cl_types::I64);
        b.append_block_param(load_model, cl_types::I64);
        b.append_block_param(model_ready, cl_types::I64);
        b.append_block_param(model_ready, cl_types::I64);
        b.append_block_param(model_ready, cl_types::I64);
        b.append_block_param(loop_header, cl_types::I64);
        b.append_block_param(loop_header, cl_types::I64);
        b.append_block_param(loop_header, cl_types::I64);
        b.append_block_param(loop_exit, cl_types::I64);
        b.append_block_param(loop_exit, cl_types::I64);
        b.append_block_param(finalize, cl_types::I64);
        b.append_block_param(finalize, cl_types::I64);

        b.switch_to_block(entry);
        b.seal_block(entry);

        let argc = b.block_params(entry)[0];
        let argv = b.block_params(entry)[1];
        let four = b.ins().iconst(cl_types::I32, 4);
        let argc_ok = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, argc, four);
        b.ins().brif(argc_ok, load_data, &[], argc_err, &[]);

        b.switch_to_block(argc_err);
        b.seal_block(argc_err);
        let two = b.ins().iconst(cl_types::I32, 2);
        b.ins().return_(&[two]);

        b.switch_to_block(load_data);
        b.seal_block(load_data);

        let argv1_addr = b.ins().load(cl_types::I64, MemFlags::new(), argv, 8);
        let argv2_addr = b.ins().load(cl_types::I64, MemFlags::new(), argv, 16);

        if !needs_forward_pass {
            let json_gv = module.declare_data_in_func(json_data, b.func);
            let json_ptr = b.ins().symbol_value(cl_types::I64, json_gv);
            let write_ref = module.declare_func_in_func(write_file_id, b.func);
            b.ins().call(write_ref, &[argv2_addr, json_ptr]);
            b.ins().jump(no_forward_write, &[]);

            b.switch_to_block(no_forward_write);
            b.seal_block(no_forward_write);
            let zero = b.ins().iconst(cl_types::I32, 0);
            b.ins().return_(&[zero]);
        } else {
            let model_ptr_slot = b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                8,
                0,
            ));
            let weight_ptrs_slot = b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                8,
                0,
            ));
            let num_weights_slot = b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                8,
                0,
            ));
            let model_ptr_addr = b.ins().stack_addr(cl_types::I64, model_ptr_slot, 0);
            let weight_ptrs_addr = b.ins().stack_addr(cl_types::I64, weight_ptrs_slot, 0);
            let num_weights_addr = b.ins().stack_addr(cl_types::I64, num_weights_slot, 0);

            let strlen_ref = module.declare_func_in_func(strlen_id, b.func);
            let path_len_call = b.ins().call(strlen_ref, &[argv1_addr]);
            let data_path_len = b.inst_results(path_len_call)[0];

            let calib_load_ref = module.declare_func_in_func(calib_load_id, b.func);
            let load_call = b.ins().call(calib_load_ref, &[argv1_addr, data_path_len]);
            let batches_ptr = b.inst_results(load_call)[0];
            let zero_i64 = b.ins().iconst(cl_types::I64, 0);
            let batches_ok = b.ins().icmp(IntCC::NotEqual, batches_ptr, zero_i64);
            b.ins().brif(
                batches_ok,
                load_model,
                &[batches_ptr, argv2_addr],
                data_null,
                &[],
            );

            b.switch_to_block(data_null);
            b.seal_block(data_null);
            let four_ret = b.ins().iconst(cl_types::I32, 4);
            b.ins().return_(&[four_ret]);

            b.switch_to_block(load_model);
            b.seal_block(load_model);
            let lm_batches = b.block_params(load_model)[0];
            let lm_sidecar_ptr = b.block_params(load_model)[1];

            let calib_count_ref = module.declare_func_in_func(calib_count_id, b.func);
            let count_call = b.ins().call(calib_count_ref, &[lm_batches]);
            let count = b.inst_results(count_call)[0];
            let has_batches = b.ins().icmp(IntCC::NotEqual, count, zero_i64);
            b.ins().brif(has_batches, batch_preflight, &[], model_create_entry, &[]);

            b.switch_to_block(batch_preflight);
            b.seal_block(batch_preflight);
            let out_ptr_slot = b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                8,
                0,
            ));
            let out_len_slot = b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                8,
                0,
            ));
            let out_ptr_addr = b.ins().stack_addr(cl_types::I64, out_ptr_slot, 0);
            let out_len_addr = b.ins().stack_addr(cl_types::I64, out_len_slot, 0);
            let calib_batch_ref = module.declare_func_in_func(calib_batch_id, b.func);
            b.ins().call(calib_batch_ref, &[lm_batches, zero_i64, out_ptr_addr, out_len_addr]);

            let batch_len = b.ins().load(cl_types::I64, MemFlags::new(), out_len_addr, 0);
                let actual_batch_elems = b.ins().ushr_imm(batch_len, 2);
                let expected_batch_elems = b.ins().iconst(
                cl_types::I64,
                arena_layout
                    .entries
                    .first()
                    .map(|(_, _, nbytes)| (*nbytes / 4) as i64)
                    .unwrap_or(0),
            );
                let batch_ok = b.ins().icmp(IntCC::Equal, actual_batch_elems, expected_batch_elems);
            b.ins().brif(batch_ok, model_create_entry, &[], batch_shape_err, &[]);

            b.switch_to_block(batch_shape_err);
            b.seal_block(batch_shape_err);
            let mismatch_ref = module.declare_data_in_func(
                batch_shape_mismatch_data.expect("forward path defines batch-shape mismatch data"),
                b.func,
            );
            let mismatch_ptr = b.ins().symbol_value(ptr_ty, mismatch_ref);
            let write_ref = module.declare_func_in_func(write_file_id, b.func);
            b.ins().call(write_ref, &[lm_sidecar_ptr, mismatch_ptr]);
            let calib_free_ref = module.declare_func_in_func(_calib_free_id, b.func);
            b.ins().call(calib_free_ref, &[lm_batches]);
            let three = b.ins().iconst(cl_types::I32, 3);
            b.ins().return_(&[three]);

            b.switch_to_block(model_create_entry);
            b.seal_block(model_create_entry);
            let argv3_addr = b.ins().load(cl_types::I64, MemFlags::new(), argv, 24);

            let model_create_ref = module.declare_func_in_func(model_create_id, b.func);
            let model_create_call = b.ins().call(model_create_ref, &[argv3_addr]);
            let model_ptr = b.inst_results(model_create_call)[0];
            let model_ok = b.ins().icmp(IntCC::NotEqual, model_ptr, zero_i64);
            b.ins().brif(
                model_ok,
                model_ready,
                &[lm_batches, lm_sidecar_ptr, model_ptr],
                model_null,
                &[],
            );

            b.switch_to_block(model_null);
            b.seal_block(model_null);
            let four_ret = b.ins().iconst(cl_types::I32, 4);
            b.ins().return_(&[four_ret]);

            b.switch_to_block(model_ready);
            b.seal_block(model_ready);
            let mr_batches = b.block_params(model_ready)[0];
            let mr_sidecar_ptr = b.block_params(model_ready)[1];
            let mr_model_ptr = b.block_params(model_ready)[2];

            let model_num_weights_ref = module.declare_func_in_func(model_num_weights_id, b.func);
            let num_weights_call = b.ins().call(model_num_weights_ref, &[mr_model_ptr]);
            let num_weights = b.inst_results(num_weights_call)[0];

            let model_weight_ptrs_ref =
                module.declare_func_in_func(model_weight_ptrs_id, b.func);
            let weight_ptrs_call = b.ins().call(model_weight_ptrs_ref, &[mr_model_ptr]);
            let weight_ptrs = b.inst_results(weight_ptrs_call)[0];

            b.ins().store(MemFlags::new(), mr_model_ptr, model_ptr_addr, 0);
            b.ins().store(MemFlags::new(), weight_ptrs, weight_ptrs_addr, 0);
            b.ins().store(MemFlags::new(), num_weights, num_weights_addr, 0);
            b.ins().jump(loop_header, &[mr_batches, mr_sidecar_ptr, zero_i64]);

            b.switch_to_block(loop_header);
            let lh_batches = b.block_params(loop_header)[0];
            let lh_sidecar_ptr = b.block_params(loop_header)[1];
            let i_cur = b.block_params(loop_header)[2];

            let calib_count_ref = module.declare_func_in_func(calib_count_id, b.func);
            let count_call = b.ins().call(calib_count_ref, &[lh_batches]);
            let count = b.inst_results(count_call)[0];
            let loop_cond = b.ins().icmp(IntCC::UnsignedLessThan, i_cur, count);
            b.ins().brif(loop_cond, loop_body, &[], loop_exit, &[lh_batches, lh_sidecar_ptr]);

            b.switch_to_block(loop_body);
            b.seal_block(loop_body);

            let out_ptr_slot = b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                8,
                0,
            ));
            let out_len_slot = b.create_sized_stack_slot(StackSlotData::new(
                StackSlotKind::ExplicitSlot,
                8,
                0,
            ));
            let out_ptr_addr = b.ins().stack_addr(cl_types::I64, out_ptr_slot, 0);
            let out_len_addr = b.ins().stack_addr(cl_types::I64, out_len_slot, 0);

            let calib_batch_ref = module.declare_func_in_func(calib_batch_id, b.func);
            b.ins().call(calib_batch_ref, &[lh_batches, i_cur, out_ptr_addr, out_len_addr]);

            let batch_ptr = b.ins().load(cl_types::I64, MemFlags::new(), out_ptr_addr, 0);
            let batch_len = b.ins().load(cl_types::I64, MemFlags::new(), out_len_addr, 0);
            let batch_elems = b.ins().ushr_imm(batch_len, 2);
            let model_ptr = b.ins().load(cl_types::I64, MemFlags::new(), model_ptr_addr, 0);
            let num_weights = b.ins().load(cl_types::I64, MemFlags::new(), num_weights_addr, 0);
            let calib_model_forward_ref =
                module.declare_func_in_func(calib_model_forward_id, b.func);
            // Spec §4.3: 4th arg is element count, not byte length.
            // `nsl_calibration_batch_at` returns out_len in bytes, so we
            // shift right by 2 (f32 = 4 bytes). Pre-v2 the wrapper ignored
            // this argument so the bytes-vs-elements mismatch was latent.
            let fwd_call = b.ins().call(
                calib_model_forward_ref,
                &[model_ptr, num_weights, batch_ptr, batch_elems],
            );
            let fwd_status = b.inst_results(fwd_call)[0];

            // v2 follow-up (spec §5.1): wrapper returns 3 on per-batch
            // shape mismatch. The scaffolding's `batch_preflight` only
            // validates batch 0; this catches drift mid-loop. Reuse the
            // same `batch_shape_mismatch_data` payload + exit code as
            // the preflight path so the surfaced reason is identical.
            let observe_continue = b.create_block();
            let batch_mismatch_mid_loop = b.create_block();
            let zero_status = b.ins().iconst(cl_types::I32, 0);
            let status_ok = b.ins().icmp(IntCC::Equal, fwd_status, zero_status);
            b.ins().brif(
                status_ok,
                observe_continue,
                &[],
                batch_mismatch_mid_loop,
                &[],
            );

            b.switch_to_block(batch_mismatch_mid_loop);
            b.seal_block(batch_mismatch_mid_loop);
            let mid_mismatch_ref = module.declare_data_in_func(
                batch_shape_mismatch_data
                    .expect("forward path defines batch-shape mismatch data"),
                b.func,
            );
            let mid_mismatch_ptr = b.ins().symbol_value(ptr_ty, mid_mismatch_ref);
            let mid_write_ref = module.declare_func_in_func(write_file_id, b.func);
            b.ins().call(mid_write_ref, &[lh_sidecar_ptr, mid_mismatch_ptr]);
            let mid_free_ref = module.declare_func_in_func(_calib_free_id, b.func);
            b.ins().call(mid_free_ref, &[lh_batches]);
            let mid_three = b.ins().iconst(cl_types::I32, 3);
            b.ins().return_(&[mid_three]);

            b.switch_to_block(observe_continue);
            b.seal_block(observe_continue);

            for entry in observe_plan {
                let arena_id = arena_data_id.expect("arena_data_id is Some when observe_plan is non-empty");
                let arena_gv = module.declare_data_in_func(arena_id, b.func);
                let arena_base = b.ins().symbol_value(ptr_ty, arena_gv);
                let run_id = running_data_ids[&entry.running_symbol];
                let run_gv = module.declare_data_in_func(run_id, b.func);
                let running_base = b.ins().symbol_value(ptr_ty, run_gv);
                emit_2d_max_abs_loop(
                    &mut b,
                    arena_base,
                    entry.src_offset,
                    entry.rows,
                    entry.channels,
                    running_base,
                );
            }

            let one_i64 = b.ins().iconst(cl_types::I64, 1);
            let i_next = b.ins().iadd(i_cur, one_i64);
            b.ins().jump(loop_header, &[lh_batches, lh_sidecar_ptr, i_next]);

            b.seal_block(loop_header);

            b.switch_to_block(loop_exit);
            b.seal_block(loop_exit);
            let le_batches = b.block_params(loop_exit)[0];
            let le_sidecar_ptr = b.block_params(loop_exit)[1];
            b.ins().jump(finalize, &[le_batches, le_sidecar_ptr]);

            b.switch_to_block(finalize);
            b.seal_block(finalize);
            let fin_batches = b.block_params(finalize)[0];
            let fin_sidecar_ptr = b.block_params(finalize)[1];
            let model_destroy_ref = module.declare_func_in_func(model_destroy_id, b.func);
            let calib_free_ref = module.declare_func_in_func(_calib_free_id, b.func);

            if finalize_plan.is_empty() {
                let json_gv = module.declare_data_in_func(json_data, b.func);
                let json_ptr = b.ins().symbol_value(cl_types::I64, json_gv);
                let write_ref = module.declare_func_in_func(write_file_id, b.func);
                b.ins().call(write_ref, &[fin_sidecar_ptr, json_ptr]);
                let model_ptr = b.ins().load(cl_types::I64, MemFlags::new(), model_ptr_addr, 0);
                b.ins().call(model_destroy_ref, &[model_ptr]);
                b.ins().call(calib_free_ref, &[fin_batches]);
                let zero = b.ins().iconst(cl_types::I32, 0);
                b.ins().return_(&[zero]);
            } else {
                let total_desc_bytes = (finalize_plan.len() as u32) * AWQ_DESC_BYTES;
                let stack_slot = b.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    total_desc_bytes,
                    8,
                ));
                let descs_base = b.ins().stack_addr(ptr_ty, stack_slot, 0);

                for (index, fp) in finalize_plan.iter().enumerate() {
                    let off = (index as i32) * (AWQ_DESC_BYTES as i32);
                    let path_id = path_data_ids[&fp.projection.0];
                    let path_gv = module.declare_data_in_func(path_id, b.func);
                    let path_ptr_v = b.ins().symbol_value(ptr_ty, path_gv);
                    b.ins().stack_store(path_ptr_v, stack_slot, off);

                    let path_len_v = b.ins().iconst(ptr_ty, fp.projection.0.len() as i64);
                    b.ins().stack_store(path_len_v, stack_slot, off + 8);

                    let channels_v = b.ins().iconst(cl_types::I32, fp.channels as i64);
                    b.ins().stack_store(channels_v, stack_slot, off + 16);

                    let pad_v = b.ins().iconst(cl_types::I32, 0);
                    b.ins().stack_store(pad_v, stack_slot, off + 20);

                    let run_id = running_data_ids[&fp.running_symbol];
                    let run_gv = module.declare_data_in_func(run_id, b.func);
                    let run_ptr_v = b.ins().symbol_value(ptr_ty, run_gv);
                    b.ins().stack_store(run_ptr_v, stack_slot, off + 24);
                }

                let strlen_ref2 = module.declare_func_in_func(strlen_id, b.func);
                let sc_len_call = b.ins().call(strlen_ref2, &[fin_sidecar_ptr]);
                let sidecar_path_len = b.inst_results(sc_len_call)[0];

                let desc_count = b.ins().iconst(ptr_ty, finalize_plan.len() as i64);
                let write_sidecar_ref = module.declare_func_in_func(write_sidecar_id, b.func);
                let ws_call = b.ins().call(
                    write_sidecar_ref,
                    &[fin_sidecar_ptr, sidecar_path_len, descs_base, desc_count],
                );
                let rc = b.inst_results(ws_call)[0];
                let model_ptr = b.ins().load(cl_types::I64, MemFlags::new(), model_ptr_addr, 0);
                b.ins().call(model_destroy_ref, &[model_ptr]);
                b.ins().call(calib_free_ref, &[fin_batches]);

                let zero_i32 = b.ins().iconst(cl_types::I32, 0);
                let is_zero = b.ins().icmp(IntCC::Equal, rc, zero_i32);
                b.ins().brif(is_zero, write_ok, &[], write_err, &[]);

                b.switch_to_block(write_err);
                b.seal_block(write_err);
                b.ins().return_(&[rc]);

                b.switch_to_block(write_ok);
                b.seal_block(write_ok);
                let zero = b.ins().iconst(cl_types::I32, 0);
                b.ins().return_(&[zero]);
            }
        }

        b.finalize();
    }

    module
        .define_function(main_id, &mut ctx)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("define main: {e}"),
        })?;

    write_object_module(module, out_path)
}

pub fn link_calibration_binary(
    scaffolding_obj: &Path,
    calib_model_obj: &Path,
    out_binary: &Path,
) -> Result<(), HarnessError> {
    let obj_bytes = std::fs::read(calib_model_obj).map_err(|e| HarnessError::Infrastructure {
        reason: format!(
            "link_calibration_binary: read {}: {e}",
            calib_model_obj.display()
        ),
    })?;
    let obj = object::File::parse(&*obj_bytes).map_err(|e| HarnessError::Infrastructure {
        reason: format!(
            "link_calibration_binary: parse {}: {e}",
            calib_model_obj.display()
        ),
    })?;
    if !obj
        .symbols()
        .any(|symbol| symbol.name() == Ok("nsl_calib_model_forward") && !symbol.is_undefined())
    {
        return Err(HarnessError::Infrastructure {
            reason: format!(
                "calibration: model-forward wrapper missing from calib_model.o.\n\
 requested:  link scaffolding.o <- calib_model.o with nsl_calib_model_forward resolved\n\
 expected:   calib_model.o exports nsl_calib_model_forward (the f32-buffer wrapper around model_forward)\n\
 found:      {} does not export nsl_calib_model_forward",
                calib_model_obj.display()
            ),
        });
    }

    crate::linker::link_multi(
        &[scaffolding_obj.to_path_buf(), calib_model_obj.to_path_buf()],
        out_binary,
    )
    .map_err(|e| {
        let err_str = format!("link_calibration_binary: {e}");
        if err_str.contains("nsl_calib_model_forward") {
            HarnessError::Infrastructure {
                reason: format!(
                    "calibration: model-forward wrapper missing from calib_model.o.\n\
 requested:  link scaffolding.o <- calib_model.o with nsl_calib_model_forward resolved\n\
 expected:   calib_model.o exports nsl_calib_model_forward (the f32-buffer wrapper around model_forward)\n\
 found:      {err_str}"
                ),
            }
        } else {
            HarnessError::Infrastructure { reason: err_str }
        }
    })
}

/// Emit a Cranelift object + link a calibration binary whose `main()`:
///
///   1. Validates `argc >= 4` → returns 2 if not.
///   2. Parses `argv[1]` (data_path), `argv[2]` (sidecar_path),
///      `argv[3]` (weights_path) using `strlen` for lengths.
///   3. Calls `nsl_calibration_load(data_path_ptr, data_path_len)`.
///      Returns 4 if null.
///   4. If `needs_forward_pass`: loops over batches calling
///      `nsl_calibration_batch_at`; after each batch, runs a plan-driven
///      2D max-abs reduction into per-projection running-buffer globals.
///   5. After the batch loop: builds a descriptor array on the stack and
///      calls `nsl_awq_write_sidecar(sidecar_path, descriptors)`.
///      Propagates non-zero return code.
///   6. Calls `nsl_calibration_free(batches)` (deferred to final step).
///   7. Returns 0.
///
/// For non-forward hooks, emits the simpler path that pre-bakes the JSON
/// and writes it via `nsl_write_file`.
fn emit_and_link_calibration_binary(
    tmp: &Path,
    sidecar_json: &[u8],
    needs_forward_pass: bool,
    observe_plan: &[ObservePlanEntry],
    finalize_plan: &[FinalizePlanEntry],
) -> Result<PathBuf, HarnessError> {
    if !needs_forward_pass && sidecar_json.contains(&0u8) {
        return Err(HarnessError::Infrastructure {
            reason: "sidecar JSON contains embedded NUL byte".into(),
        });
    }

    // ── Build the ObjectModule targeting the host ───────────────────
    let flag_builder = cranelift_codegen::settings::builder();
    let isa = cranelift_native::builder()
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("cranelift native builder: {e}"),
        })?
        .finish(cranelift_codegen::settings::Flags::new(flag_builder))
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("cranelift ISA: {e}"),
        })?;

    let ptr_ty = cl_types::I64; // 64-bit pointers

    let call_conv = {
        let detected = isa.default_call_conv();
        #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
        {
            if detected != cranelift_codegen::isa::CallConv::WindowsFastcall {
                cranelift_codegen::isa::CallConv::WindowsFastcall
            } else {
                detected
            }
        }
        #[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
        {
            detected
        }
    };

    let builder = ObjectBuilder::new(
        isa,
        "nsl_calibration",
        cranelift_module::default_libcall_names(),
    )
    .map_err(|e| HarnessError::Infrastructure {
        reason: format!("object builder: {e}"),
    })?;
    let mut module = ObjectModule::new(builder);

    // ── Embed the sidecar JSON (NUL-terminated) — used by non-forward path ──
    let mut json_bytes: Vec<u8> = sidecar_json.to_vec();
    json_bytes.push(0);
    let json_data = module
        .declare_data("__nsl_calibration_sidecar_json", Linkage::Local, false, false)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare sidecar-json data: {e}"),
        })?;
    let mut json_desc = DataDescription::new();
    json_desc.define(json_bytes.into_boxed_slice());
    module
        .define_data(json_data, &json_desc)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("define sidecar-json data: {e}"),
        })?;

    // ── Step 2: Declare per-projection running buffers as zeroinit .data globals ──
    // These are the targets of the 2D max-abs reduction loop.
    let mut running_data_ids: HashMap<String, DataId> = HashMap::new();
    for entry in finalize_plan {
        let mut data = DataDescription::new();
        data.define_zeroinit((entry.channels * 4) as usize);
        let id = module
            .declare_data(
                &entry.running_symbol,
                Linkage::Export,
                /* writable */ true,
                /* tls */ false,
            )
            .map_err(|e| HarnessError::Infrastructure {
                reason: format!("declare running buf '{}': {e}", entry.running_symbol),
            })?;
        module
            .define_data(id, &data)
            .map_err(|e| HarnessError::Infrastructure {
                reason: format!("define running buf '{}': {e}", entry.running_symbol),
            })?;
        running_data_ids.insert(entry.running_symbol.clone(), id);
    }

    // ── Declare the retention arena as an imported symbol ──────────────────
    // Only declared when there are actual observe_plan entries that will
    // reference it in IR.  Declaring it unconditionally causes a linker error
    // for binaries (like IdentityHook) where no model object is linked.
    let arena_data_id: Option<DataId> = if !observe_plan.is_empty() {
        let id = module
            .declare_data(
                "__nsl_calib_retention_arena",
                Linkage::Import,
                /* writable */ false,
                /* tls */ false,
            )
            .map_err(|e| HarnessError::Infrastructure {
                reason: format!("declare retention arena import: {e}"),
            })?;
        Some(id)
    } else {
        None
    };

    // ── Declare path strings for each finalize_plan projection ─────────────
    let mut path_data_ids: HashMap<String, DataId> = HashMap::new();
    for entry in finalize_plan {
        let sym = format!(
            "__nsl_awq_pathstr.{}",
            entry.projection.0.replace(['.', '-'], "_")
        );
        let mut data = DataDescription::new();
        // Store the projection name as raw bytes (no NUL terminator needed;
        // nsl_awq_write_sidecar receives the length explicitly).
        data.define(entry.projection.0.as_bytes().to_vec().into_boxed_slice());
        let id = module
            .declare_data(&sym, Linkage::Local, false, false)
            .map_err(|e| HarnessError::Infrastructure {
                reason: format!("declare path string '{}': {e}", sym),
            })?;
        module
            .define_data(id, &data)
            .map_err(|e| HarnessError::Infrastructure {
                reason: format!("define path string '{}': {e}", sym),
            })?;
        path_data_ids.insert(entry.projection.0.clone(), id);
    }

    // ── Declare extern fns ──────────────────────────────────────────

    // strlen(const char*) -> usize
    let mut strlen_sig = module.make_signature();
    strlen_sig.call_conv = call_conv;
    strlen_sig.params.push(AbiParam::new(cl_types::I64)); // const char*
    strlen_sig.returns.push(AbiParam::new(cl_types::I64)); // size_t (use I64 on 64-bit)
    let strlen_id = module
        .declare_function("strlen", Linkage::Import, &strlen_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare strlen: {e}"),
        })?;

    // nsl_write_file(path: *const u8, content: *const u8) -> void
    let mut write_file_sig = module.make_signature();
    write_file_sig.call_conv = call_conv;
    write_file_sig.params.push(AbiParam::new(cl_types::I64)); // path
    write_file_sig.params.push(AbiParam::new(cl_types::I64)); // content
    let write_file_id = module
        .declare_function("nsl_write_file", Linkage::Import, &write_file_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_write_file: {e}"),
        })?;

    // nsl_calibration_load(path_ptr: *const u8, path_len: usize) -> *mut Batches
    let mut calib_load_sig = module.make_signature();
    calib_load_sig.call_conv = call_conv;
    calib_load_sig.params.push(AbiParam::new(cl_types::I64)); // path_ptr
    calib_load_sig.params.push(AbiParam::new(cl_types::I64)); // path_len
    calib_load_sig.returns.push(AbiParam::new(cl_types::I64)); // *mut Batches
    let calib_load_id = module
        .declare_function("nsl_calibration_load", Linkage::Import, &calib_load_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calibration_load: {e}"),
        })?;

    // nsl_calibration_count(batches: *mut Batches) -> usize
    let mut calib_count_sig = module.make_signature();
    calib_count_sig.call_conv = call_conv;
    calib_count_sig.params.push(AbiParam::new(cl_types::I64)); // *mut Batches
    calib_count_sig.returns.push(AbiParam::new(cl_types::I64)); // usize
    let calib_count_id = module
        .declare_function("nsl_calibration_count", Linkage::Import, &calib_count_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calibration_count: {e}"),
        })?;

    // nsl_calibration_batch_at(batches, i, *out_ptr, *out_len) -> void
    let mut calib_batch_sig = module.make_signature();
    calib_batch_sig.call_conv = call_conv;
    calib_batch_sig.params.push(AbiParam::new(cl_types::I64)); // batches
    calib_batch_sig.params.push(AbiParam::new(cl_types::I64)); // i: usize
    calib_batch_sig.params.push(AbiParam::new(cl_types::I64)); // *out_ptr
    calib_batch_sig.params.push(AbiParam::new(cl_types::I64)); // *out_len
    let calib_batch_id = module
        .declare_function("nsl_calibration_batch_at", Linkage::Import, &calib_batch_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calibration_batch_at: {e}"),
        })?;

    // nsl_calibration_free(batches: *mut Batches) -> void
    let mut calib_free_sig = module.make_signature();
    calib_free_sig.call_conv = call_conv;
    calib_free_sig.params.push(AbiParam::new(cl_types::I64)); // *mut Batches
    let _calib_free_id = module
        .declare_function("nsl_calibration_free", Linkage::Import, &calib_free_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calibration_free: {e}"),
        })?;

    // nsl_awq_write_sidecar(path_ptr, path_len, desc_ptr, desc_len) -> i32
    let mut write_sidecar_sig = module.make_signature();
    write_sidecar_sig.call_conv = call_conv;
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // sidecar_path_ptr
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // sidecar_path_len
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // desc_ptr
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // desc_len
    write_sidecar_sig.returns.push(AbiParam::new(cl_types::I32));
    let write_sidecar_id = module
        .declare_function("nsl_awq_write_sidecar", Linkage::Import, &write_sidecar_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_awq_write_sidecar: {e}"),
        })?;

    // ── Declare main() ──────────────────────────────────────────────
    let mut main_sig = module.make_signature();
    main_sig.call_conv = call_conv;
    main_sig.params.push(AbiParam::new(cl_types::I32)); // argc
    main_sig.params.push(AbiParam::new(cl_types::I64)); // argv: **u8
    main_sig.returns.push(AbiParam::new(cl_types::I32));
    let main_id = module
        .declare_function("main", Linkage::Export, &main_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare main: {e}"),
        })?;

    // ── Build main() body ───────────────────────────────────────────
    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, 0),
        main_sig,
    ));
    let mut fn_builder_ctx = FunctionBuilderContext::new();

    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

        // Block layout differs by `needs_forward_pass`:
        //
        // NON-FORWARD (observation-free hooks, e.g. IdentityHook):
        //   entry → (argc_err | no_forward_write)
        //   argc_err: return 2
        //   no_forward_write: argv[2]=sidecar_path → nsl_write_file → return 0
        //
        // FORWARD (AWQ / hooks that need model_forward):
        //   entry → (argc_err | load_data)
        //   load_data: load argv[1]=data argv[2]=sidecar argv[3]=weights
        //              → nsl_calibration_load → (data_null | loop_header)
        //   data_null: return 4
        //   loop_header(batches_ptr, sidecar_ptr, i): i < count → loop_body / loop_exit
        //   loop_body: batch_at + plan-driven 2D max-abs loops → loop_header
        //   loop_exit(sidecar_ptr): → finalize
        //   finalize(sidecar_ptr): build descriptors → nsl_awq_write_sidecar
        //                          → (write_ok | write_err)
        //   write_err: return rc
        //   write_ok: return 0

        let entry       = b.create_block();
        let argc_err    = b.create_block();

        b.append_block_params_for_function_params(entry);

        // ── Entry: validate argc >= 4 ──────────────────────────────
        b.switch_to_block(entry);
        b.seal_block(entry);
        let argc = b.block_params(entry)[0];
        let argv = b.block_params(entry)[1];
        let four_i32 = b.ins().iconst(cl_types::I32, 4);
        let argc_ok = b.ins().icmp(IntCC::SignedGreaterThanOrEqual, argc, four_i32);

        if !needs_forward_pass {
            // ── Observation-free path: just write pre-baked sidecar JSON ──
            // argv[2] is sidecar_path.  No calibration loading needed.
            let write_block = b.create_block();
            b.ins().brif(argc_ok, write_block, &[], argc_err, &[]);

            // argc_err block
            b.switch_to_block(argc_err);
            b.seal_block(argc_err);
            let two = b.ins().iconst(cl_types::I32, 2);
            b.ins().return_(&[two]);

            b.switch_to_block(write_block);
            b.seal_block(write_block);

            let off2 = b.ins().iconst(cl_types::I64, 16i64);
            let argv2_addr = b.ins().iadd(argv, off2);
            let sidecar_path_ptr = b.ins().load(cl_types::I64, MemFlags::trusted(), argv2_addr, 0);

            let json_gv = module.declare_data_in_func(json_data, b.func);
            let json_ptr = b.ins().symbol_value(cl_types::I64, json_gv);
            let write_ref = module.declare_func_in_func(write_file_id, b.func);
            b.ins().call(write_ref, &[sidecar_path_ptr, json_ptr]);

            let zero = b.ins().iconst(cl_types::I32, 0);
            b.ins().return_(&[zero]);

            b.finalize();
        } else {
            // ── Forward-pass path: load calibration data + batch loop ──
            let load_data   = b.create_block();
            let data_null   = b.create_block();
            let loop_header = b.create_block();
            let loop_body   = b.create_block();
            let loop_exit   = b.create_block();
            let finalize    = b.create_block();
            let write_ok    = b.create_block();
            let write_err   = b.create_block();

            // loop_header params: (batches_ptr: I64, sidecar_path_ptr: I64, i: I64)
            b.append_block_param(loop_header, cl_types::I64);
            b.append_block_param(loop_header, cl_types::I64);
            b.append_block_param(loop_header, cl_types::I64);
            // loop_exit param: (sidecar_path_ptr: I64)
            b.append_block_param(loop_exit, cl_types::I64);
            // finalize param: (sidecar_path_ptr: I64)
            b.append_block_param(finalize, cl_types::I64);

            b.ins().brif(argc_ok, load_data, &[], argc_err, &[]);

            // argc_err: return 2
            b.switch_to_block(argc_err);
            b.seal_block(argc_err);
            let two = b.ins().iconst(cl_types::I32, 2);
            b.ins().return_(&[two]);

            // load_data: parse argv[1..3], call nsl_calibration_load
            b.switch_to_block(load_data);
            b.seal_block(load_data);

            let off1 = b.ins().iconst(cl_types::I64, 8i64);
            let off2 = b.ins().iconst(cl_types::I64, 16i64);
            let argv1_addr = b.ins().iadd(argv, off1);
            let argv2_addr = b.ins().iadd(argv, off2);

            let data_path_ptr    = b.ins().load(cl_types::I64, MemFlags::trusted(), argv1_addr, 0);
            let sidecar_path_ptr = b.ins().load(cl_types::I64, MemFlags::trusted(), argv2_addr, 0);

            let strlen_ref = module.declare_func_in_func(strlen_id, b.func);
            let data_len_call = b.ins().call(strlen_ref, &[data_path_ptr]);
            let data_path_len = b.inst_results(data_len_call)[0];

            let calib_load_ref = module.declare_func_in_func(calib_load_id, b.func);
            let calib_call = b.ins().call(calib_load_ref, &[data_path_ptr, data_path_len]);
            let batches_ptr = b.inst_results(calib_call)[0];

            let batches_ok = b.ins().icmp_imm(IntCC::NotEqual, batches_ptr, 0);
            let zero_i64 = b.ins().iconst(cl_types::I64, 0);
            b.ins().brif(
                batches_ok,
                loop_header, &[batches_ptr, sidecar_path_ptr, zero_i64],
                data_null, &[],
            );

            // data_null: return 4
            b.switch_to_block(data_null);
            b.seal_block(data_null);
            let four_ret = b.ins().iconst(cl_types::I32, 4);
            b.ins().return_(&[four_ret]);

            // loop_header: if i < count → loop_body else loop_exit
            b.switch_to_block(loop_header);
            let lh_batches     = b.block_params(loop_header)[0];
            let lh_sidecar_ptr = b.block_params(loop_header)[1];
            let i_cur          = b.block_params(loop_header)[2];

            let calib_count_ref = module.declare_func_in_func(calib_count_id, b.func);
            let count_call = b.ins().call(calib_count_ref, &[lh_batches]);
            let count = b.inst_results(count_call)[0];

            let loop_cond = b.ins().icmp(IntCC::UnsignedLessThan, i_cur, count);
            b.ins().brif(loop_cond, loop_body, &[], loop_exit, &[lh_sidecar_ptr]);

            // loop_body: call nsl_calibration_batch_at + plan-driven 2D max-abs loops
            b.switch_to_block(loop_body);
            b.seal_block(loop_body);

            let out_ptr_slot = b.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 8, 0,
            ));
            let out_len_slot = b.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot, 8, 0,
            ));
            let out_ptr_addr = b.ins().stack_addr(cl_types::I64, out_ptr_slot, 0);
            let out_len_addr = b.ins().stack_addr(cl_types::I64, out_len_slot, 0);

            let calib_batch_ref = module.declare_func_in_func(calib_batch_id, b.func);
            b.ins().call(calib_batch_ref, &[lh_batches, i_cur, out_ptr_addr, out_len_addr]);

            // ── Step 4: Plan-driven 2D max-abs reduction per projection ──
            // For each ObservePlanEntry, read from the retention arena and
            // accumulate into the corresponding running-buffer global.
            // `arena_data_id` is Some iff observe_plan is non-empty (declared together).
            for entry in observe_plan {
                let arena_id = arena_data_id.expect("arena_data_id is Some when observe_plan is non-empty");
                let arena_gv = module.declare_data_in_func(arena_id, b.func);
                let arena_base = b.ins().symbol_value(ptr_ty, arena_gv);
                let run_id = running_data_ids[&entry.running_symbol];
                let run_gv = module.declare_data_in_func(run_id, b.func);
                let running_base = b.ins().symbol_value(ptr_ty, run_gv);
                emit_2d_max_abs_loop(
                    &mut b,
                    arena_base,
                    entry.src_offset,
                    entry.rows,
                    entry.channels,
                    running_base,
                );
                // emit_2d_max_abs_loop leaves the builder positioned at row_exit.
                // We need to continue emitting in that block (it becomes our
                // "current block" after the helper returns).
            }

            let one_i64 = b.ins().iconst(cl_types::I64, 1);
            let i_next = b.ins().iadd(i_cur, one_i64);
            b.ins().jump(loop_header, &[lh_batches, lh_sidecar_ptr, i_next]);

            b.seal_block(loop_header);

            // loop_exit → finalize
            b.switch_to_block(loop_exit);
            b.seal_block(loop_exit);
            let le_sidecar_ptr = b.block_params(loop_exit)[0];
            b.ins().jump(finalize, &[le_sidecar_ptr]);

            // ── Step 5: finalize — build descriptor array, call nsl_awq_write_sidecar ──
            b.switch_to_block(finalize);
            b.seal_block(finalize);
            let fin_sidecar_ptr = b.block_params(finalize)[0];

            if finalize_plan.is_empty() {
                // No AWQ projections — write the pre-baked JSON (empty placeholder)
                // so the host can read a valid sidecar file.
                let json_gv = module.declare_data_in_func(json_data, b.func);
                let json_ptr = b.ins().symbol_value(cl_types::I64, json_gv);
                let write_ref = module.declare_func_in_func(write_file_id, b.func);
                b.ins().call(write_ref, &[fin_sidecar_ptr, json_ptr]);
                let zero = b.ins().iconst(cl_types::I32, 0);
                b.ins().return_(&[zero]);
            } else {
                // Allocate a stack slot large enough for all descriptors.
                let total_desc_bytes = (finalize_plan.len() as u32) * AWQ_DESC_BYTES;
                let stack_slot = b.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    total_desc_bytes,
                    8, // 8-byte alignment (pointer alignment)
                ));
                let descs_base = b.ins().stack_addr(ptr_ty, stack_slot, 0);

                for (i, fp) in finalize_plan.iter().enumerate() {
                    let off = (i as i32) * (AWQ_DESC_BYTES as i32);

                    // path_ptr (offset 0, 8 bytes)
                    let path_id = path_data_ids[&fp.projection.0];
                    let path_gv = module.declare_data_in_func(path_id, b.func);
                    let path_ptr_v = b.ins().symbol_value(ptr_ty, path_gv);
                    b.ins().stack_store(path_ptr_v, stack_slot, off);

                    // path_len (offset 8, 8 bytes — usize on 64-bit)
                    let path_len_v = b.ins().iconst(ptr_ty, fp.projection.0.len() as i64);
                    b.ins().stack_store(path_len_v, stack_slot, off + 8);

                    // channels (offset 16, 4 bytes — u32)
                    let channels_v = b.ins().iconst(cl_types::I32, fp.channels as i64);
                    b.ins().stack_store(channels_v, stack_slot, off + 16);

                    // _pad (offset 20, 4 bytes — u32)
                    let pad_v = b.ins().iconst(cl_types::I32, 0);
                    b.ins().stack_store(pad_v, stack_slot, off + 20);

                    // running_ptr (offset 24, 8 bytes — *const f32)
                    let run_id = running_data_ids[&fp.running_symbol];
                    let run_gv = module.declare_data_in_func(run_id, b.func);
                    let run_ptr_v = b.ins().symbol_value(ptr_ty, run_gv);
                    b.ins().stack_store(run_ptr_v, stack_slot, off + 24);
                }

                // Compute the sidecar path length via strlen.
                let strlen_ref2 = module.declare_func_in_func(strlen_id, b.func);
                let sc_len_call = b.ins().call(strlen_ref2, &[fin_sidecar_ptr]);
                let sidecar_path_len = b.inst_results(sc_len_call)[0];

                // Call nsl_awq_write_sidecar(path_ptr, path_len, descs_ptr, descs_len).
                let desc_count = b.ins().iconst(ptr_ty, finalize_plan.len() as i64);
                let write_sidecar_ref = module.declare_func_in_func(write_sidecar_id, b.func);
                let ws_call = b.ins().call(
                    write_sidecar_ref,
                    &[fin_sidecar_ptr, sidecar_path_len, descs_base, desc_count],
                );
                let rc = b.inst_results(ws_call)[0];

                // If rc != 0, return rc; else fall through to return 0.
                let zero_i32 = b.ins().iconst(cl_types::I32, 0);
                let is_zero = b.ins().icmp(IntCC::Equal, rc, zero_i32);
                b.ins().brif(is_zero, write_ok, &[], write_err, &[]);

                b.switch_to_block(write_err);
                b.seal_block(write_err);
                b.ins().return_(&[rc]);

                b.switch_to_block(write_ok);
                b.seal_block(write_ok);
                let zero = b.ins().iconst(cl_types::I32, 0);
                b.ins().return_(&[zero]);
            }

            b.finalize();
        }
    }

    module
        .define_function(main_id, &mut ctx)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("define main: {e}"),
        })?;

    // ── Finish → object bytes on disk ───────────────────────────────
    let product = module.finish();
    let obj_bytes = product.emit().map_err(|e| HarnessError::Infrastructure {
        reason: format!("emit object: {e}"),
    })?;

    let obj_path = tmp.join("calibration.o");
    fs::write(&obj_path, &obj_bytes).map_err(|e| HarnessError::Infrastructure {
        reason: format!("writing {}: {e}", obj_path.display()),
    })?;

    let binary_path = if cfg!(target_os = "windows") {
        tmp.join("calibration.exe")
    } else {
        tmp.join("calibration")
    };
    crate::linker::link_multi(&[obj_path], &binary_path).map_err(|e| {
        HarnessError::Infrastructure {
            reason: format!("linking calibration binary: {e}"),
        }
    })?;

    Ok(binary_path)
}

fn create_tmp_dir() -> Result<PathBuf, HarnessError> {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    let p = std::env::temp_dir().join(format!(
        "nsl-calibration-{}-{}",
        std::process::id(),
        nanos
    ));
    fs::create_dir_all(&p).map_err(|e| HarnessError::Infrastructure {
        reason: format!("cannot create {}: {e}", p.display()),
    })?;
    Ok(p)
}

struct TmpCleanup(PathBuf);
impl Drop for TmpCleanup {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use object::{Object, ObjectSymbol};
    use nsl_errors::{FileId, Level};
    use nsl_lexer::{tokenize, Interner};

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }

    fn fixture(name: &str) -> PathBuf {
        repo_root().join("tests").join("fixtures").join(name)
    }

    fn parse_awq_fixture() -> (nsl_ast::Module, Interner) {
        let source = std::fs::read_to_string(fixture("awq_calibration_mlp.nsl"))
            .expect("fixture readable");
        let mut interner = Interner::new();
        let (tokens, lex_diags) = tokenize(&source, FileId(0), &mut interner);
        assert!(
            lex_diags.iter().all(|diag| !matches!(diag.level, Level::Error)),
            "fixture must lex cleanly: {lex_diags:?}"
        );

        let parsed = nsl_parser::parse(&tokens, &mut interner);
        assert!(
            parsed
                .diagnostics
                .iter()
                .all(|diag| !matches!(diag.level, Level::Error)),
            "fixture must parse cleanly: {:?}",
            parsed.diagnostics
        );

        (parsed.module, interner)
    }

    fn awq_fixture_compile_options(
        ast: &nsl_ast::Module,
        interner: &Interner,
    ) -> crate::CompileOptions {
        let mut analysis_interner = interner.clone();
        let analysis = nsl_semantic::analyze(ast, &mut analysis_interner);
        let mut opts = crate::CompileOptions::default();
        opts.calibration_batch_seq = Some((8, 4));
        opts.calibration_compile_bundle = Some(std::sync::Arc::new(
            crate::calibration::CalibrationCompileBundle {
                ast: ast.clone(),
                interner: analysis_interner,
                type_map: analysis.type_map.clone(),
            },
        ));
        opts.weight_index_map = analysis.weight_index_map.clone();
        opts
    }

    #[test]
    fn emit_calibration_model_object_exports_wrapper_symbol() {
        let (ast, interner) = parse_awq_fixture();
        let projections = crate::calibration::pre_scan_awq_projections_from_ast(&ast, &interner);
        let arena_layout = build_arena_layout(&projections, 8, 4);
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("calib_model.o");
        let opts = awq_fixture_compile_options(&ast, &interner);

        emit_calibration_model_object(
            &ast,
            &opts,
            &arena_layout,
            &out_path,
        )
        .expect("emit succeeds");

        assert!(out_path.exists(), "calib_model.o must be written");
        let obj_bytes = std::fs::read(&out_path).expect("object readable");
        let obj = object::File::parse(&*obj_bytes).expect("object::File::parse");

        assert!(
            obj.symbols()
                .any(|symbol| symbol.name() == Ok("nsl_calib_model_forward")),
            "calib_model.o must export nsl_calib_model_forward"
        );
    }

    #[test]
    fn calib_model_object_contains_model_forward_body() {
        let (ast, interner) = parse_awq_fixture();
        let projections = crate::calibration::pre_scan_awq_projections_from_ast(&ast, &interner);
        let arena_layout = build_arena_layout(&projections, 8, 4);
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("calib_model.o");
        let opts = awq_fixture_compile_options(&ast, &interner);

        emit_calibration_model_object(&ast, &opts, &arena_layout, &out_path)
            .expect("emit succeeds");

        let obj_bytes = std::fs::read(&out_path).expect("object readable");
        let obj = object::File::parse(&*obj_bytes).expect("object::File::parse");

        assert!(
            obj.symbols().any(|symbol| symbol.name() == Ok("model_forward")),
            "calib_model.o must export model_forward"
        );
    }

    #[test]
    fn emit_calibration_scaffolding_imports_calib_model_forward() {
        let observe_plan = vec![ObservePlanEntry {
            projection: crate::calibration::ProjectionRef("TinyMLP.up_proj".into()),
            src_offset: 0,
            rows: 32,
            channels: 64,
            running_symbol: "__nsl_awq_running_up_proj".into(),
        }];
        let finalize_plan = vec![FinalizePlanEntry {
            projection: crate::calibration::ProjectionRef("TinyMLP.up_proj".into()),
            running_symbol: "__nsl_awq_running_up_proj".into(),
            channels: 64,
        }];
        let arena_layout = build_arena_layout(
            &[crate::calibration::DiscoveredProjection {
                projection: crate::calibration::ProjectionRef("TinyMLP.up_proj".into()),
                weight_shape: [128, 64],
            }],
            8,
            4,
        );
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("scaffolding.o");

        emit_calibration_scaffolding_object(
            &observe_plan,
            &finalize_plan,
            &arena_layout,
            b"{}",
            true,
            &out_path,
        )
        .expect("emit succeeds");

        let obj_bytes = std::fs::read(&out_path).expect("object readable");
        let obj = object::File::parse(&*obj_bytes).expect("object::File::parse");

        let imports: Vec<String> = obj
            .symbols()
            .filter(|symbol| symbol.is_undefined())
            .filter_map(|symbol| symbol.name().ok().map(str::to_string))
            .collect();
        assert!(
            imports.iter().any(|symbol| symbol == "nsl_calib_model_forward"),
            "scaffolding.o must import nsl_calib_model_forward"
        );
        assert!(
            imports.iter().any(|symbol| symbol == "nsl_calibration_load"),
            "scaffolding.o must import nsl_calibration_load"
        );
        assert!(
            imports.iter().any(|symbol| symbol == "nsl_calibration_batch_at"),
            "scaffolding.o must import nsl_calibration_batch_at"
        );
        assert!(
            imports.iter().any(|symbol| symbol == "nsl_awq_write_sidecar"),
            "scaffolding.o must import nsl_awq_write_sidecar"
        );

        assert!(
            obj.symbols().any(|symbol| symbol.name() == Ok("main") && !symbol.is_undefined()),
            "scaffolding.o must export main"
        );
    }

    #[test]
    fn empty_observe_plan_with_forward_pass_refuses() {
        let arena_layout = build_arena_layout(
            &[crate::calibration::DiscoveredProjection {
                projection: crate::calibration::ProjectionRef("TinyMLP.up_proj".into()),
                weight_shape: [128, 64],
            }],
            8,
            4,
        );
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("scaffolding.o");

        let err = emit_calibration_scaffolding_object(
            &[],
            &[],
            &arena_layout,
            b"{}",
            true,
            &out_path,
        )
        .expect_err("must refuse defensively");

        let err_str = format!("{err}");
        assert!(err_str.contains("calibration: forward pass emitted but no observations declared"));
        assert!(err_str.contains("requested:"));
        assert!(err_str.contains("expected:"));
        assert!(err_str.contains("found:"));
    }
}

#[cfg(test)]
mod grad_arena_emission {
    use super::*;
    use object::{Object, ObjectSection, ObjectSymbol};
    use nsl_lexer::Interner;

    use crate::calibration::discovery::WggoGradTarget;
    use crate::calibration::observation::ProjectionRef;
    use crate::calibration::retention::build_grad_arena_layout;

    /// Build two-layer fixture: two attention layers with 64x64 weight matrices.
    fn sample_two_attention_layers() -> Vec<WggoGradTarget> {
        (0..2)
            .map(|i| WggoGradTarget {
                layer_key: format!("gpt.blocks.{i}.attn"),
                class_name: "Attention".into(),
                head_dim: 64,
                w_q: ProjectionRef(format!("gpt.blocks.{i}.attn.q_proj")),
                w_k: ProjectionRef(format!("gpt.blocks.{i}.attn.k_proj")),
                w_v: ProjectionRef(format!("gpt.blocks.{i}.attn.v_proj")),
                w_o: ProjectionRef(format!("gpt.blocks.{i}.attn.o_proj")),
                w_q_shape: [64, 64],
                w_k_shape: [64, 64],
                w_v_shape: [64, 64],
                w_o_shape: [64, 64],
            })
            .collect()
    }

    #[test]
    fn emit_grad_retention_arena_declares_bss_global_with_total_bytes() {
        let targets = sample_two_attention_layers();
        let layout = build_grad_arena_layout(&targets);
        // 2 layers × 4 projections × (64 × 64 × 4 bytes) = 8 × 16384 = 131072
        let expected_total = 2 * 4 * 64 * 64 * 4u32;
        assert_eq!(layout.total_bytes, expected_total, "fixture sanity");

        // Build a minimal Compiler with calibration_grad_retention set.
        let interner = Interner::new();
        let type_map = nsl_semantic::checker::TypeMap::default();
        let mut opts = crate::CompileOptions::default();
        opts.calibration_grad_retention = Some(targets);

        let mut compiler =
            crate::compiler::Compiler::new(&interner, &type_map, &opts)
                .expect("Compiler::new");

        // Should succeed and populate grad_arena_layout.
        compiler
            .emit_grad_retention_arena()
            .expect("emit_grad_retention_arena must succeed");

        assert!(
            compiler.grad_arena_layout.is_some(),
            "grad_arena_layout must be populated after emit"
        );
        assert_eq!(
            compiler.grad_arena_layout.as_ref().unwrap().total_bytes,
            expected_total,
            "stored layout total_bytes must match"
        );

        // Finalize and verify the symbol appears in the emitted object with
        // the expected size and Export linkage.
        let obj_bytes = compiler.finalize().expect("finalize");
        let obj = object::File::parse(&*obj_bytes).expect("object::File::parse");

        // Find the __nsl_calib_grad_arena symbol.
        let sym = obj
            .symbols()
            .find(|s| s.name() == Ok("__nsl_calib_grad_arena"))
            .expect("__nsl_calib_grad_arena must be exported");

        assert!(
            !sym.is_undefined(),
            "__nsl_calib_grad_arena must be defined (not just imported)"
        );

        // On ELF the symbol size field reflects the BSS region directly.
        // On COFF (Windows) symbol sizes are stored in the containing section,
        // not in the symbol table entry, so sym.size() returns 0 for BSS.
        // We check the section's size instead, which is format-agnostic.
        let section_idx = sym.section_index()
            .expect("__nsl_calib_grad_arena must belong to a section");
        let section = obj.section_by_index(section_idx)
            .expect("section must be readable");
        assert_eq!(
            section.size(),
            expected_total as u64,
            "__nsl_calib_grad_arena section size must equal layout.total_bytes \
             (ELF: symbol carries it; COFF: section carries it)"
        );
    }

    #[test]
    fn emit_grad_retention_arena_no_ops_when_option_is_none() {
        let interner = Interner::new();
        let type_map = nsl_semantic::checker::TypeMap::default();
        let opts = crate::CompileOptions::default(); // calibration_grad_retention is None

        let mut compiler =
            crate::compiler::Compiler::new(&interner, &type_map, &opts)
                .expect("Compiler::new");

        compiler
            .emit_grad_retention_arena()
            .expect("no-op must not error");

        assert!(
            compiler.grad_arena_layout.is_none(),
            "grad_arena_layout must remain None when option is not set"
        );
    }
}
