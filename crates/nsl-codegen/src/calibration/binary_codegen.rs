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
use crate::calibration::sidecar::{Sidecar, WggoHeadGradients, SIDECAR_VERSION};
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
    let needs_backward = {
        let sets: Vec<_> = registry.iter().map(|h| h.requires()).collect();
        crate::calibration::observation::ObservationPlan::union_of(&sets).needs_backward()
    };

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
    // Spec §4.5: collect running-buffer symbols from hooks that require backward.
    // Passed into the scaffolding so the loop body emits per-step symbol references.
    let per_step_backward_symbols: Vec<String> = registry
        .iter()
        .filter(|h| h.requires_backward())
        .flat_map(|h| h.finalize_plan().into_iter().map(|fp| fp.running_symbol))
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
            needs_backward,
            &per_step_backward_symbols,
            &scaffolding_obj,
        )?;

        let binary_path = tmp.join(if cfg!(windows) {
            "calibration.exe"
        } else {
            "calibration"
        });
        link_calibration_binary(&scaffolding_obj, &model_obj, &binary_path, needs_backward)?;
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
        wggo_head_gradients: hooks_out
            .get("wggo_head_gradients")
            .and_then(|bytes| serde_json::from_slice::<WggoHeadGradients>(bytes).ok()),
        hooks: hooks_out,
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

/// Emit a Cranelift IR loop inside `b` that performs a per-head `|grad · weight|`
/// accumulation over `n_proj_heads × elements_per_head` f32 elements, writing the
/// f64 running sum into `running_base`.
///
/// **Spec §4.6 invariants:**
/// - f32 input values are promoted to f64 **before** accumulation (preserves
///   numerical fidelity for the 524K-element reduction on large models).
/// - The running buffer is f64 (8 bytes per element); `bytes_per_element = 8` in
///   the matching `FinalizePlanEntry`.
/// - MQA/GQA replication: K/V projections have fewer heads than Q/O.  Each K/V
///   head `h` contributes to `replication` Q-head slots at indices
///   `h * replication .. h * replication + replication` in the running buffer.
///   The unrolled write loop `for r in 0..replication` implements this.
///
/// **Block layout (6 blocks):**
///   caller_block → head_header(h=0)
///   head_header(h): h < n_proj_heads → head_body / head_exit
///   head_body: e=0 → elem_header(e)
///   elem_header(e): e < elements_per_head → elem_body / elem_exit
///   elem_body: grad*weight|→f64 → for r: running[h*rep+r] += v64; e++ → elem_header
///   elem_exit: h++ → head_header
///   head_exit: (fall through to caller)
///
/// All blocks are sealed here; the caller must NOT seal them.
/// After this call the builder is positioned at `head_exit` (just like
/// `emit_2d_max_abs_loop` leaves the builder at `row_exit`).
// Task 22 (scaffolding wire-up) will call this from the scaffolding loop body
// to perform the per-batch |g·w| reduction per (target, projection).
// Task 21 implemented WggoGradientHook::emit_per_step and unit-tested the
// iteration logic; Task 22 threads weight-data pointers in and removes the
// allow(dead_code) suppression below.
#[allow(dead_code)]
pub(crate) fn emit_per_head_dot_abs_accum(
    b: &mut FunctionBuilder,
    grad_arena_base: cranelift_codegen::ir::Value,
    src_offset: u32,
    weight_data_base: cranelift_codegen::ir::Value,
    running_base: cranelift_codegen::ir::Value,
    n_proj_heads: u32,
    elements_per_head: u32,
    replication: u32,
) {
    let ptr_ty = cl_types::I64;

    let head_header = b.create_block();
    let head_body   = b.create_block();
    let elem_header = b.create_block();
    let elem_body   = b.create_block();
    let elem_exit   = b.create_block();
    let head_exit   = b.create_block();

    // Block params: head_header carries h (I32), elem_header carries e (I32).
    b.append_block_param(head_header, cl_types::I32); // h
    b.append_block_param(elem_header, cl_types::I32); // e

    let zero_i32      = b.ins().iconst(cl_types::I32, 0);
    let four          = b.ins().iconst(cl_types::I32, 4);
    let n_heads_v     = b.ins().iconst(cl_types::I32, n_proj_heads as i64);
    let elems_v       = b.ins().iconst(cl_types::I32, elements_per_head as i64);
    let soff_c        = b.ins().iconst(cl_types::I32, src_offset as i64);

    // Jump from caller into the outer loop.
    b.ins().jump(head_header, &[zero_i32]);

    // ── head_header: if h < n_proj_heads → head_body else head_exit ────────
    b.switch_to_block(head_header);
    let h = b.block_params(head_header)[0];
    let cmp_h = b.ins().icmp(IntCC::UnsignedLessThan, h, n_heads_v);
    b.ins().brif(cmp_h, head_body, &[], head_exit, &[]);

    // ── head_body: jump to elem_header(e=0) ────────────────────────────────
    b.switch_to_block(head_body);
    b.seal_block(head_body);
    b.ins().jump(elem_header, &[zero_i32]);

    // ── elem_header: if e < elements_per_head → elem_body else elem_exit ───
    b.switch_to_block(elem_header);
    let e = b.block_params(elem_header)[0];
    let cmp_e = b.ins().icmp(IntCC::UnsignedLessThan, e, elems_v);
    b.ins().brif(cmp_e, elem_body, &[], elem_exit, &[]);

    // ── elem_body: compute addresses, load grad + weight, accumulate ────────
    b.switch_to_block(elem_body);
    b.seal_block(elem_body);

    // Linear element index within this head: lin = h * elements_per_head + e
    let h_elems = b.ins().imul(h, elems_v);
    let lin     = b.ins().iadd(h_elems, e);

    // grad_addr = grad_arena_base + src_offset + lin * 4
    let lin4_g   = b.ins().imul(lin, four);
    let raw_g    = b.ins().iadd(lin4_g, soff_c);
    let raw_g_p  = b.ins().uextend(ptr_ty, raw_g);
    let grad_addr = b.ins().iadd(grad_arena_base, raw_g_p);

    // weight_addr = weight_data_base + lin * 4  (weights are packed [out, in] f32;
    // no src_offset needed — weight_data_base already points at this projection's
    // weight slice, which Task 21 threads in from nsl_model_get_weight_ptrs).
    let lin4_w    = b.ins().imul(lin, four);
    let raw_w_p   = b.ins().uextend(ptr_ty, lin4_w);
    let wt_addr   = b.ins().iadd(weight_data_base, raw_w_p);

    // Load f32 values.
    let gv  = b.ins().load(cl_types::F32, MemFlags::new(), grad_addr, 0);
    let wv  = b.ins().load(cl_types::F32, MemFlags::new(), wt_addr,   0);

    // |grad * weight| — compute in f32, then promote to f64.
    let prod    = b.ins().fmul(gv, wv);
    let abs_prod = b.ins().fabs(prod);
    let v64     = b.ins().fpromote(cl_types::F64, abs_prod);

    // Unrolled replication loop: for r in 0..replication { running[h*rep+r] += v64 }
    // Each K/V head contributes to `replication` Q-head running slots.
    // The running buffer is f64 so each slot is 8 bytes.
    let eight      = b.ins().iconst(cl_types::I32, 8);
    let rep_v      = b.ins().iconst(cl_types::I32, replication as i64);
    let base_slot  = b.ins().imul(h, rep_v); // h * replication

    for r in 0..replication {
        let r_v      = b.ins().iconst(cl_types::I32, r as i64);
        let slot_idx = b.ins().iadd(base_slot, r_v);      // h*rep + r
        let slot_off = b.ins().imul(slot_idx, eight);      // * 8 bytes
        let slot_p   = b.ins().uextend(ptr_ty, slot_off);
        let run_addr = b.ins().iadd(running_base, slot_p);
        let cur      = b.ins().load(cl_types::F64, MemFlags::new(), run_addr, 0);
        let new_val  = b.ins().fadd(cur, v64);
        b.ins().store(MemFlags::new(), new_val, run_addr, 0);
    }

    // e++ → elem_header
    let ep1 = b.ins().iadd_imm(e, 1);
    b.ins().jump(elem_header, &[ep1]);

    // ── elem_exit: h++ → head_header ────────────────────────────────────────
    b.switch_to_block(elem_exit);
    b.seal_block(elem_exit);
    let hp1 = b.ins().iadd_imm(h, 1);
    b.ins().jump(head_header, &[hp1]);

    // Seal back-edge blocks now that all predecessors are known.
    b.seal_block(head_header);
    b.seal_block(elem_header);

    b.switch_to_block(head_exit);
    b.seal_block(head_exit);
    // caller continues from head_exit
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

    // model_forward: (weight_ptrs: i64, num_weights: i64, input_tensor: i64) -> i64
    // Returns the output tensor (y) — callers are responsible for freeing it.
    // Previously this was void-returning and freed y internally; changed in Task 15
    // so that `emit_calibration_backward_wrapper` can capture y to compute dy = 2y.
    let mut sig = compiler.module.make_signature();
    sig.call_conv = compiler.call_conv;
    sig.params.push(AbiParam::new(cl_types::I64));
    sig.params.push(AbiParam::new(cl_types::I64));
    sig.params.push(AbiParam::new(cl_types::I64));
    sig.returns.push(AbiParam::new(cl_types::I64)); // returns y (output tensor)
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

        // Free transposed weight views (these are always intermediate tensors).
        let tensor_free_id = compiler.registry.runtime_fns["nsl_tensor_free"].0;
        let tensor_free_ref = compiler.module.declare_func_in_func(tensor_free_id, b.func);
        for tensor_ptr in transposed_tensors {
            b.ins().call(tensor_free_ref, &[tensor_ptr]);
        }

        // Return y to the caller; the caller is responsible for freeing it.
        // If the forward method has no return value, return null (0) — callers
        // that depend on y (e.g. the backward wrapper) must not be used with
        // void-returning models, but this keeps the bridge valid.
        let output_tensor = if !forward_sig.returns.is_empty() {
            b.inst_results(forward_call)[0]
        } else {
            b.ins().iconst(cl_types::I64, 0)
        };
        b.ins().return_(&[output_tensor]);
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
        let fwd_call = b.ins()
            .call(model_forward_ref, &[weight_ptrs, num_weights, input_tensor]);
        // model_forward now returns y; free it here (forward wrapper only observes
        // activations via the retention arena — it does not need y itself).
        let y_tensor = b.inst_results(fwd_call)[0];

        let tensor_free_id = compiler.registry.runtime_fns["nsl_tensor_free"].0;
        let tensor_free_ref = compiler.module.declare_func_in_func(tensor_free_id, b.func);
        b.ins().call(tensor_free_ref, &[y_tensor]);
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

/// Emit the `model_backward` bridge: source-AD splice into
/// `__nsl_calib_grad_arena` via the `on_param_grad` callback.
///
/// Spec §4.2 last paragraph + projection-identity invariant.
///
/// Signature (Task 16):
///   model_backward(weight_ptrs: i64, num_weights: i64,
///                  input_handle: i64, dy_handle: i64) -> void
///
/// Body:
///  1. Load weight tensors from weight_ptrs (same pattern as model_forward).
///  2. Extract the model's forward WengertList via WengertExtractor.
///  3. Build primal_vars: map x VarId → input_handle, weight VarIds → loaded tensors.
///  4. Lower the primal forward to obtain full intermediate values.
///  5. Generate the adjoint WengertList from the primal.
///  6. Lower the adjoint with on_param_grad callback: for each WGGO target weight
///     gradient, copy to __nsl_calib_grad_arena at the correct byte offset.
///  7. Free loaded weight views and return.
///
/// Forward-equivalence note: the primal forward re-runs as source-AD ops
/// (FFI calls), independent of the compiled NSL forward.  The retention arena
/// writes (Task 4's splice) happen in model_forward, NOT here — this function
/// only writes to the GRAD arena.  This is intentional: spec §7.2 requires the
/// two arenas to be written by separate paths.
fn emit_model_backward_bridge(
    compiler: &mut crate::compiler::Compiler<'_>,
    model_def: &nsl_ast::decl::ModelDef,
    model_name: &str,
    // reserved for Task 17+: cross-arena consistency checks (forward retention vs backward retention symmetry)
    _arena_layout: &crate::calibration::ArenaLayout,
    grad_arena_layout: &crate::calibration::retention::GradArenaLayout,
) -> Result<cranelift_module::FuncId, HarnessError> {
    // ── Signature ────────────────────────────────────────────────────────────
    // Task 16 expands from the Task-14/15 2-param stub to 4 params so the
    // bridge can load weights for source-AD (same pattern as model_forward).
    let mut sig = compiler.module.make_signature();
    sig.call_conv = compiler.call_conv;
    sig.params.push(AbiParam::new(cl_types::I64)); // weight_ptrs
    sig.params.push(AbiParam::new(cl_types::I64)); // num_weights
    sig.params.push(AbiParam::new(cl_types::I64)); // input_handle (NslTensor* as i64)
    sig.params.push(AbiParam::new(cl_types::I64)); // dy_handle    (NslTensor* as i64)

    let func_id = compiler
        .module
        .declare_function("model_backward", Linkage::Local, &sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare model_backward: {e}"),
        })?;

    // ── Pre-compute grad-arena layout map: projection path → (offset, bytes) ──
    // Keyed by the bare field name (e.g. "up_proj") for fast lookup.
    // GradArenaLayout entries carry ProjectionRef with form "<ModelName>.<field>".
    let prefix = format!("{model_name}.");
    let field_to_arena: HashMap<String, (u32, u32)> = grad_arena_layout
        .entries
        .iter()
        .filter_map(|(proj_ref, offset, nbytes)| {
            proj_ref
                .0
                .strip_prefix(&prefix)
                .map(|field| (field.to_string(), (*offset, *nbytes)))
        })
        .collect();

    // Early exit: if no grad arena entries match this model's fields, emit a
    // trivial stub to avoid confusing the Wengert extraction path below.
    // This mirrors the Task-14 stub and is correct when the WGGO targets list
    // references a different model name than the one being compiled.
    if field_to_arena.is_empty() {
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
            b.ins().return_(&[]);
            b.finalize();
        }
        compiler
            .module
            .define_function(func_id, &mut ctx)
            .map_err(|e| HarnessError::Infrastructure {
                reason: format!("define model_backward (empty-layout stub): {e}"),
            })?;
        return Ok(func_id);
    }

    // ── Declare helper FFIs ───────────────────────────────────────────────────
    // nsl_model_get_weight: (model_ptr: i64, name_ptr: i64, name_len: i64) -> i64
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
            reason: format!("declare nsl_model_get_weight (backward bridge): {e}"),
        })?;

    // nsl_tensor_free: (tensor: i64) -> void
    let tensor_free_id = compiler.registry.runtime_fns["nsl_tensor_free"].0;

    // __nsl_calib_grad_arena: declared as Local by emit_grad_retention_arena.
    // Reference it from this function via Import — the linker resolves within
    // the same object (both Local and Export are resolved before relocation).
    let grad_arena_data_id = compiler
        .module
        .declare_data("__nsl_calib_grad_arena", Linkage::Import, true, false)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare __nsl_calib_grad_arena import (backward bridge): {e}"),
        })?;

    // ── Extract the model's forward WengertList ───────────────────────────────
    // Collect tensor field names (same logic as awq_tensor_field_names).
    let tensor_fields: Vec<String> = model_def
        .members
        .iter()
        .filter_map(|member| match member {
            nsl_ast::decl::ModelMember::LayerDecl { name, type_ann, .. } => {
                match &type_ann.kind {
                    nsl_ast::types::TypeExprKind::Named(sym)
                        if compiler.interner.resolve(sym.0) == Some("Tensor") =>
                    {
                        compiler.interner.resolve(name.0).map(str::to_string)
                    }
                    _ => None,
                }
            }
            _ => None,
        })
        .collect();

    // Find the forward method body.  collect_models() / declare_user_functions()
    // have already run before this function is called, so model_method_bodies is
    // populated.  If "forward" is missing, emit the trivial stub.
    let forward_fn_def = compiler
        .models
        .model_method_bodies
        .get(model_name)
        .and_then(|m| m.get("forward"))
        .cloned();

    let forward_fn_def = match forward_fn_def {
        Some(fd) => fd,
        None => {
            // No forward method known — emit trivial stub.
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
                b.ins().return_(&[]);
                b.finalize();
            }
            compiler
                .module
                .define_function(func_id, &mut ctx)
                .map_err(|e| HarnessError::Infrastructure {
                    reason: format!("define model_backward (no-forward stub): {e}"),
                })?;
            return Ok(func_id);
        }
    };

    // Run WengertExtractor on the forward body.
    // We set self_context = "self" so SelfRef + field → "self.<field>".
    // The model instance is registered with a synthetic Symbol whose name is "self".
    let mut extractor = crate::source_ad::WengertExtractor::new(compiler.interner);
    extractor.set_model_method_bodies(compiler.models.model_method_bodies.clone());
    extractor.set_model_field_types(compiler.models.model_field_types.clone());

    // Register the model's `self` symbol as the model instance.
    // forward_fn_def.params[0] is the `self` parameter whose Symbol maps
    // to the context name "self" in the WengertExtractor.  If the method has
    // no params (malformed model — rejected by the front-end), the extraction
    // will fail and we fall through to the stub path below.
    let self_sym = forward_fn_def.params.first().map(|p| p.name);
    let self_param_name: Option<String> = self_sym.and_then(|sym| {
        compiler.interner.resolve(sym.0).map(str::to_string)
    });
    if let Some(sym) = self_sym {
        extractor.register_model_instance(sym, model_name);
    }
    // Set self_context so that ExprKind::SelfRef and pipe-with-bare-field-ident
    // arms in extract_expr can resolve "self.<field>" correctly when the extractor
    // runs directly on the method body (not via method inlining from an outer loop).
    extractor.set_self_context(self_param_name);

    // Register the input parameter (second param of forward, e.g. `x`).
    let input_sym_opt = forward_fn_def.params.get(1).map(|p| p.name);
    if let Some(input_sym) = input_sym_opt {
        extractor.register_input(input_sym);
    }

    let extraction_ok = extractor.extract_stmts(&forward_fn_def.body.stmts);

    if !extraction_ok || extractor.wengert_list().ops.is_empty() {
        // Extraction failed — emit trivial stub, log warning.
        eprintln!(
            "[nsl] model_backward: WengertExtractor failed for '{model_name}' — emitting stub"
        );
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
            b.ins().return_(&[]);
            b.finalize();
        }
        compiler
            .module
            .define_function(func_id, &mut ctx)
            .map_err(|e| HarnessError::Infrastructure {
                reason: format!("define model_backward (extraction-failed stub): {e}"),
            })?;
        return Ok(func_id);
    }

    // Build adjoint-variable → grad_arena entry map.
    // named_param_var_ids() yields ("self.<field>", VarId).
    // We map "self.<field>" → "<model_name>.<field>" to match grad_arena keys.
    let self_prefix = "self.";
    let mut param_adj_set = std::collections::HashSet::<crate::wengert::VarId>::new();
    // Maps adjoint VarId → (byte_offset, byte_size) in grad_arena.
    let mut adj_to_arena: std::collections::HashMap<crate::wengert::VarId, (u32, u32)> =
        std::collections::HashMap::new();

    let start_var = extractor.next_var_id();
    let mut gen = crate::source_ad::AdjointGenerator::new(start_var);
    // Generate adjoint before we need adjoint_of() — it consumes the primal list.
    // Capture the primal output VarId first so we can retrieve loss_seed_var_id
    // after generate() maps primal.output → loss_bar.
    let primal_output_var = extractor.wengert_list().output;
    let adjoint = gen.generate(extractor.wengert_list());
    // Spec §4.2: retrieve the loss seed VarId so the adjoint lowering can be
    // given the real upstream gradient (dy_handle = dL/dy = 2·y from the L2
    // wrapper) instead of the default Constant(1.0).
    let loss_seed_vid = gen.loss_seed_var_id(primal_output_var);

    for (compound_name, primal_vid) in extractor.named_param_var_ids() {
        // Strip "self." prefix to get bare field name.
        let field_name = match compound_name.strip_prefix(self_prefix) {
            Some(f) => f,
            None => continue,
        };
        let Some(&(offset, nbytes)) = field_to_arena.get(field_name) else {
            continue;
        };
        let Some(adj_vid) = gen.adjoint_of(*primal_vid) else {
            continue;
        };
        param_adj_set.insert(adj_vid);
        adj_to_arena.insert(adj_vid, (offset, nbytes));
    }

    // ── Emit IR ──────────────────────────────────────────────────────────────
    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, compiler.next_func_index()),
        sig,
    ));
    let mut fn_builder_ctx = FunctionBuilderContext::new();
    let result = (|| {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        b.switch_to_block(entry);
        b.seal_block(entry);

        let weight_ptrs   = b.block_params(entry)[0];
        let _num_weights  = b.block_params(entry)[1];
        let input_handle  = b.block_params(entry)[2];
        // dy_handle: the upstream loss gradient dL/dy = 2·y computed by the
        // L2 wrapper (spec §4.2).  Must be used to seed the adjoint's loss_bar
        // VarId so weight gradients are scaled correctly; ignoring it would
        // silently corrupt per-head scoring magnitudes used by WGGO pruning.
        let dy_handle     = b.block_params(entry)[3];

        // ── Step 1: load weight tensors from weight_ptrs ──────────────────
        // Mirrors emit_model_forward_bridge: call nsl_model_get_weight per field.
        // We build a map: field_name → Cranelift Value (the loaded tensor handle).
        let model_get_weight_ref = compiler
            .module
            .declare_func_in_func(model_get_weight_id, b.func);
        let tensor_free_ref = compiler
            .module
            .declare_func_in_func(tensor_free_id, b.func);

        let mut field_values: HashMap<String, cranelift_codegen::ir::Value> = HashMap::new();
        let mut loaded_weight_vals: Vec<cranelift_codegen::ir::Value> = Vec::new();
        for field_name in &tensor_fields {
            let qualified_name = format!("{model_name}.{field_name}");
            let symbol = format!(
                "__nsl_calib_bwd_weight_name_{}_{}",
                model_name, field_name
            );
            let name_data = declare_cstr_rodata(&mut compiler.module, &symbol, &qualified_name)?;
            let name_ref = compiler.module.declare_data_in_func(name_data, b.func);
            let name_ptr = b.ins().symbol_value(cl_types::I64, name_ref);
            let name_len = b.ins().iconst(cl_types::I64, qualified_name.len() as i64);
            let call_inst = b.ins().call(model_get_weight_ref, &[weight_ptrs, name_ptr, name_len]);
            let tensor_val = b.inst_results(call_inst)[0];
            field_values.insert(field_name.clone(), tensor_val);
            loaded_weight_vals.push(tensor_val);
        }

        // ── Step 2: build primal_vars for source-AD lowering ──────────────
        // Map:
        //   input symbol VarId → input_handle (from block param)
        //   each weight symbol VarId → the loaded tensor value
        let mut primal_vars = crate::wengert_lower::VarMap::new();

        // Map input symbol.
        if let Some(input_sym) = input_sym_opt {
            if let Some(&vid) = extractor.symbol_var_map().get(&input_sym) {
                primal_vars.insert(vid, input_handle);
            }
        }
        // Map weight params via named_param_var_ids.
        for (compound_name, primal_vid) in extractor.named_param_var_ids() {
            let field_name = match compound_name.strip_prefix(self_prefix) {
                Some(f) => f,
                None => continue,
            };
            if let Some(&tensor_val) = field_values.get(field_name) {
                primal_vars.insert(*primal_vid, tensor_val);
            }
        }
        // Also seed Input ops by name in the Wengert list.
        // Resolve the input symbol name once for the name-match path.
        let input_sym_name = input_sym_opt
            .and_then(|sym| compiler.interner.resolve(sym.0))
            .unwrap_or("x");
        for op in &extractor.wengert_list().ops {
            if let crate::wengert::PrimalOp::Input(name) = &op.op {
                if primal_vars.contains_key(&op.result) {
                    continue;
                }
                if name == input_sym_name {
                    primal_vars.insert(op.result, input_handle);
                }
            }
        }

        // ── Step 3: lower primal forward ──────────────────────────────────
        let mut func_state = crate::context::FuncState::default();
        let full_lowered = map_codegen_error(
            "model_backward: lower primal forward",
            crate::wengert_lower::compile_wengert_ops(
                compiler,
                &mut b,
                &mut func_state,
                extractor.wengert_list(),
                &primal_vars,
                None,
            ),
        )?;
        let mut full_vars = full_lowered.var_map.clone();

        // ── Spec §4.2: seed adjoint with upstream dy_handle ──────────────
        // `loss_seed_vid` is the VarId that `AdjointGenerator::generate` assigned
        // `PrimalOp::Constant(1.0)` to (the default "loss seed = 1").
        // By inserting `dy_handle` for that VarId before calling
        // `compile_wengert_ops` on the adjoint, the lowerer's pre-mapped-constant
        // check (wengert_lower.rs: `if var_map.contains_key(&op.result)`) will
        // short-circuit the Constant(1.0) materialisation and use the real
        // upstream gradient `dL/dy = 2·y` instead.  Without this, weight
        // gradients are off by the 2·y factor, corrupting per-head scoring
        // magnitudes and potentially the relative head ordering used by WGGO pruning.
        if let Some(seed_vid) = loss_seed_vid {
            full_vars.insert(seed_vid, dy_handle);
        }

        // ── Step 4: prepare grad-arena global value reference ────────────
        // The GV is declared once per function so the module can link it.
        // The actual `symbol_value` instruction is emitted INSIDE the
        // on_param_grad callback — one per weight gradient — so that each
        // splice produces an independent relocation targeting
        // `__nsl_calib_grad_arena` in the object's text section.
        // Spec §4.2 / test-spec §wggo_grad_arena_splice: "one relocation
        // per distinct W_* that the on_param_grad callback writes to."
        let grad_arena_gv = compiler.module.declare_data_in_func(grad_arena_data_id, b.func);

        // ── Step 5: lower adjoint backward with on_param_grad callback ────
        // The callback: for each fired gradient VarId, load the raw data
        // pointer from the NslTensor (offset 8 = data field), compute
        // dst = arena_base + byte_offset, and emit a byte-copy loop.
        // Then free the gradient tensor.
        let arena_map = &adj_to_arena;
        let mut grad_cb = |c: &mut crate::compiler::Compiler<'_>,
                           var_id: crate::wengert::VarId,
                           grad_val: cranelift_codegen::ir::Value,
                           fb: &mut FunctionBuilder|
         -> Result<(), crate::error::CodegenError> {
            let Some(&(offset, nbytes)) = arena_map.get(&var_id) else {
                // Not in our target set — free and skip.
                c.compile_call_by_name(fb, "nsl_tensor_free", &[grad_val])?;
                return Ok(());
            };
            // Emit symbol_value here (not once outside the callback) so that
            // each splice generates an independent relocation targeting
            // __nsl_calib_grad_arena.  This satisfies the §4.2 invariant
            // that the linker can observe one arena reference per W_*.
            let arena_base = fb.ins().symbol_value(cl_types::I64, grad_arena_gv);
            // Get raw data pointer from the NslTensor struct.
            // NslTensor.data is at offset NSL_TENSOR_DATA_OFFSET (= 8 bytes after magic).
            const DATA_OFF: i32 =
                nsl_runtime::tensor::NSL_TENSOR_DATA_OFFSET as i32;
            let src_ptr = fb.ins().load(
                cl_types::I64,
                MemFlags::new(),
                grad_val,
                DATA_OFF,
            );
            // Emit inline byte-copy loop to grad arena slot.
            crate::calibration::retention::emit_splice_memcpy(
                fb,
                arena_base,
                offset,
                src_ptr,
                nbytes as u64,
            );
            // Free the gradient tensor (callback owns it).
            c.compile_call_by_name(fb, "nsl_tensor_free", &[grad_val])?;
            Ok(())
        };

        let mut adj_func_state = crate::context::FuncState::default();
        map_codegen_error(
            "model_backward: lower adjoint backward",
            crate::wengert_lower::compile_wengert_ops(
                compiler,
                &mut b,
                &mut adj_func_state,
                &adjoint,
                &full_vars,
                Some((&param_adj_set, &mut grad_cb)),
            ),
        )?;

        // ── Step 6: free loaded weight views ──────────────────────────────
        // The weights loaded via nsl_model_get_weight are borrowed views;
        // release them now.  (Matching the forward bridge pattern.)
        for weight_val in loaded_weight_vals {
            b.ins().call(tensor_free_ref, &[weight_val]);
        }

        b.ins().return_(&[]);
        b.finalize();
        Ok::<(), HarnessError>(())
    })();

    result?;

    compiler
        .module
        .define_function(func_id, &mut ctx)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("define model_backward: {e}"),
        })?;
    Ok(func_id)
}

/// Emit `nsl_calib_model_backward` — the ABI wrapper the calibration
/// scaffolding calls to run the backward pass for a single batch.
///
/// Spec §4.2 / §4.4.
///
/// Signature (Task 15): `(weight_ptrs: i64, num_weights: i64, batch_ptr: i64,
///                        batch_elem_count: i64) -> i32`.
/// Matches `nsl_calib_model_forward` so the scaffolding can call both with the
/// same (model_ptr, num_weights, batch_ptr, batch_elems) quad.
/// Returns 0 on success, 3 on batch-shape mismatch.
///
/// Body (spec §4.4):
///  1. Validate batch_elem_count.
///  2. Wrap batch_ptr in a NslTensorDesc and convert to NslTensor (input).
///  3. Call model_forward(weight_ptrs, num_weights, input) → y.
///  4. Compute dy = nsl_tensor_mul_scalar(y, 2.0, 0)  [L2 loss: dL/dy = 2·y].
///  5. Call model_backward(input, dy).
///  6. Free input, y, dy.
///  7. Return 0.
fn emit_calibration_backward_wrapper(
    compiler: &mut crate::compiler::Compiler<'_>,
    model_forward_id: cranelift_module::FuncId,
    model_backward_id: cranelift_module::FuncId,
    batch: u32,
    seq: u32,
    channels: i64,
) -> Result<(), HarnessError> {
    let (shape_vals, input_ndim) = if batch == 1 {
        (vec![seq as i64, channels], 2_i32)
    } else {
        (vec![batch as i64, seq as i64, channels], 3_i32)
    };

    // Compile-time element count — single source of truth.
    let expected_elem_count: i64 = shape_vals.iter().product();

    // Shape rodata for NslTensorDesc — same layout as the forward wrapper.
    let shape_data = declare_i64_rodata(
        &mut compiler.module,
        "__nsl_calib_bwd_input_shape",
        &shape_vals,
    )?;

    // nsl_desc_to_tensor: converts an NslTensorDesc (stack slot) into a heap-allocated
    // NslTensor handle (i64).  Same declaration pattern as the forward wrapper.
    let mut desc_to_tensor_sig = compiler.module.make_signature();
    desc_to_tensor_sig.call_conv = compiler.call_conv;
    desc_to_tensor_sig.params.push(AbiParam::new(cl_types::I64));
    desc_to_tensor_sig.returns.push(AbiParam::new(cl_types::I64));
    let desc_to_tensor_id = compiler
        .module
        .declare_function("nsl_desc_to_tensor", Linkage::Import, &desc_to_tensor_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_desc_to_tensor (backward wrapper): {e}"),
        })?;

    // nsl_tensor_mul_scalar: (tensor: i64, scalar: f64, flags: i8) -> i64.
    // Used to compute dy = 2·y (L2-loss gradient).
    let mul_scalar_id = compiler.registry.runtime_fns["nsl_tensor_mul_scalar"].0;

    // nsl_tensor_free: (tensor: i64) -> void.
    let tensor_free_id = compiler.registry.runtime_fns["nsl_tensor_free"].0;

    // Backward wrapper ABI matches the forward wrapper:
    //   (weight_ptrs: i64, num_weights: i64, batch_ptr: i64, batch_elem_count: i64) -> i32
    let mut wrapper_sig = compiler.module.make_signature();
    wrapper_sig.call_conv = compiler.call_conv;
    wrapper_sig.params.push(AbiParam::new(cl_types::I64)); // weight_ptrs
    wrapper_sig.params.push(AbiParam::new(cl_types::I64)); // num_weights
    wrapper_sig.params.push(AbiParam::new(cl_types::I64)); // batch_ptr (*const f32 as i64)
    wrapper_sig.params.push(AbiParam::new(cl_types::I64)); // batch_elem_count
    wrapper_sig.returns.push(AbiParam::new(cl_types::I32)); // status: 0 = ok, 3 = mismatch

    let func_id = compiler
        .module
        .declare_function("nsl_calib_model_backward", Linkage::Export, &wrapper_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calib_model_backward: {e}"),
        })?;

    let mut ctx = Context::for_function(Function::with_name_signature(
        UserFuncName::user(0, compiler.next_func_index()),
        wrapper_sig,
    ));
    let mut fn_builder_ctx = FunctionBuilderContext::new();
    {
        let mut b = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
        let entry = b.create_block();
        b.append_block_params_for_function_params(entry);
        let shape_ok = b.create_block();
        let shape_mismatch = b.create_block();

        b.switch_to_block(entry);
        b.seal_block(entry);

        let weight_ptrs = b.block_params(entry)[0];
        let num_weights = b.block_params(entry)[1];
        let batch_ptr = b.block_params(entry)[2];
        let batch_elem_count = b.block_params(entry)[3];

        // Validate batch element count against compile-time expected shape product.
        let expected_v = b.ins().iconst(cl_types::I64, expected_elem_count);
        let shape_eq = b.ins().icmp(IntCC::Equal, batch_elem_count, expected_v);
        b.ins().brif(shape_eq, shape_ok, &[], shape_mismatch, &[]);

        b.switch_to_block(shape_mismatch);
        b.seal_block(shape_mismatch);
        let three = b.ins().iconst(cl_types::I32, 3);
        b.ins().return_(&[three]);

        b.switch_to_block(shape_ok);
        b.seal_block(shape_ok);

        // ── Step 1: build NslTensorDesc for the input batch (mirrors forward wrapper). ──
        //
        // NslTensorDesc layout (40 bytes, 8-byte aligned):
        //   offset  0: data_ptr (i64)
        //   offset  8: shape_ptr (i64)
        //   offset 16: strides_ptr (i64, 0 = row-major default)
        //   offset 24: ndim (i32)
        //   offset 28: dtype (i32) — C API convention: 0=f32, 1=f64 (opposite of NSL internal!)
        //                            calibration batches are f32, so dtype=0 here.
        //   offset 32: device (i32, 0 = CPU)
        //   offset 36: pad (i32)
        let desc_slot = b.create_sized_stack_slot(StackSlotData::new(
            StackSlotKind::ExplicitSlot,
            40,
            3,
        ));
        let desc_addr = b.ins().stack_addr(cl_types::I64, desc_slot, 0);
        let zero_i64 = b.ins().iconst(cl_types::I64, 0);
        let zero_i32 = b.ins().iconst(cl_types::I32, 0);
        let ndim_i32 = b.ins().iconst(cl_types::I32, input_ndim as i64);
        // dtype=0 (C API f32) — mirrors the forward wrapper (binary_codegen.rs offset 28).
        // NslTensorDesc.dtype follows the C API convention where 0=f32, NOT the NSL
        // internal convention (where 0=f64). nsl_desc_to_tensor calls capi_dtype_to_nsl
        // which maps 0 → NSL 1 (f32). Calibration batches are always raw f32.
        let dtype_f32 = b.ins().iconst(cl_types::I32, 0);

        let shape_ref = compiler.module.declare_data_in_func(shape_data, b.func);
        let shape_ptr = b.ins().symbol_value(cl_types::I64, shape_ref);

        b.ins().store(MemFlags::trusted(), batch_ptr, desc_addr, 0);
        b.ins().store(MemFlags::trusted(), shape_ptr, desc_addr, 8);
        b.ins().store(MemFlags::trusted(), zero_i64, desc_addr, 16);
        b.ins().store(MemFlags::trusted(), ndim_i32, desc_addr, 24);
        b.ins().store(MemFlags::trusted(), dtype_f32, desc_addr, 28);
        b.ins().store(MemFlags::trusted(), zero_i32, desc_addr, 32);
        b.ins().store(MemFlags::trusted(), zero_i32, desc_addr, 36);

        let desc_to_tensor_ref =
            compiler.module.declare_func_in_func(desc_to_tensor_id, b.func);
        let desc_call = b.ins().call(desc_to_tensor_ref, &[desc_addr]);
        let input_handle = b.inst_results(desc_call)[0];

        // ── Step 2: call model_forward(weight_ptrs, num_weights, input) → y. ──
        let model_forward_ref =
            compiler.module.declare_func_in_func(model_forward_id, b.func);
        let fwd_call = b
            .ins()
            .call(model_forward_ref, &[weight_ptrs, num_weights, input_handle]);
        let y_handle = b.inst_results(fwd_call)[0];

        // ── Step 3: dy = nsl_tensor_mul_scalar(y, 2.0, 0)  [L2: dL/dy = 2·y]. ──
        let mul_scalar_ref =
            compiler.module.declare_func_in_func(mul_scalar_id, b.func);
        let two_f64 = b.ins().f64const(2.0_f64);
        let flags_zero = b.ins().iconst(cl_types::I8, 0); // no FBIP relinquish flags
        let dy_call = b.ins().call(mul_scalar_ref, &[y_handle, two_f64, flags_zero]);
        let dy_handle = b.inst_results(dy_call)[0];

        // ── Step 4: call model_backward(weight_ptrs, num_weights, input_handle, dy_handle). ──
        // Task 16: backward bridge now takes weight_ptrs and num_weights so it can
        // load weight tensors for source-AD primal forward re-run.
        let model_backward_ref =
            compiler.module.declare_func_in_func(model_backward_id, b.func);
        b.ins().call(model_backward_ref, &[weight_ptrs, num_weights, input_handle, dy_handle]);

        // ── Step 5: free intermediates and return 0 (success). ──
        let tensor_free_ref =
            compiler.module.declare_func_in_func(tensor_free_id, b.func);
        b.ins().call(tensor_free_ref, &[input_handle]);
        b.ins().call(tensor_free_ref, &[y_handle]);
        b.ins().call(tensor_free_ref, &[dy_handle]);

        let ok = b.ins().iconst(cl_types::I32, 0);
        b.ins().return_(&[ok]);

        b.finalize();
    }

    compiler
        .module
        .define_function(func_id, &mut ctx)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("define nsl_calib_model_backward: {e}"),
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

    // Spec safety: backward wrapper synthesises L2 loss from forward's output (dy = 2y).
    // A void-returning forward has no tensor to differentiate against — passing a null
    // y_handle into nsl_tensor_mul_scalar is silent UB / segfault at runtime.
    //
    // Check the AST return annotation here, before any Cranelift IR is emitted, so we
    // get an actionable error rather than a Cranelift verifier panic inside
    // compile_user_functions (which cannot lower a void model method to a returning IR).
    if compile_opts.calibration_grad_retention.is_some() {
        let forward_has_return = model_def.members.iter().any(|m| {
            if let nsl_ast::decl::ModelMember::Method(fn_def, _) = m {
                if bundle.interner.resolve(fn_def.name.0) == Some("forward") {
                    return fn_def.return_type.is_some();
                }
            }
            false
        });
        if !forward_has_return {
            return Err(HarnessError::Infrastructure {
                reason: format!(
                    "calibration: --wggo-importance=grad requires a model whose `forward` returns a tensor.\n\
  requested: gradient-importance scoring with synthetic L2 loss\n\
  expected:  model.forward returns a Tensor for L2 = sum(y\u{00B2})\n\
  found:     model '{model_name}' forward has no return type\n\
  fix:       return the activations from forward, or use --wggo-importance=magnitude."
                ),
            });
        }
    }

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

    // §5.3: defensive check — backward emission with empty grad observations is a bug.
    // `calibration_grad_retention.is_some()` signals that the caller requested a backward
    // pass; `grad_arena_layout` must be populated by `emit_grad_retention_arena` above.
    // If it isn't, the targets list was present but yielded zero bytes (corrupt shape
    // metadata or a hook that updated `requires()` without updating its targets list).
    // This fires when callers bypass the pre-scan refusal gate (`enforce_grad_mode_refusals`
    // in entry_points.rs) and pre-populate `calibration_grad_retention` themselves;
    // normal flows produce more actionable messages via the §5.4–§5.6 refusals above.
    if compile_opts.calibration_grad_retention.is_some()
        && compiler.grad_arena_layout.is_none()
    {
        return Err(HarnessError::Infrastructure {
            reason: "calibration: backward pass emitted but no grad observations declared.\n\
  requested:  run calibration subprocess with backward pass\n\
  expected:   __nsl_calib_grad_arena has at least one entry when a\n\
              model_backward call is emitted\n\
  found:      grad_arena_layout is empty but calibration_grad_retention\n\
              was set. Did a hook's requires() change without\n\
              updating its targets() list?\n\
  fix:        verify hook.requires() and pre_scan_wggo_targets_from_ast\n\
              produce consistent target sets."
                .into(),
        });
    }

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

    // Spec §4.2: when grad retention is requested, also emit the backward path.
    if compile_opts.calibration_grad_retention.is_some() {
        // Clone the layout before the mutable borrow of compiler so there is no
        // simultaneous &/&mut conflict.  The layout is small (a Vec of tuples).
        // Safe to unwrap: the `calibration_grad_retention.is_some()` guard above
        // plus the defensive check at the start of this function guarantees
        // `grad_arena_layout` is populated when we reach this point.
        let grad_arena_layout_clone = compiler
            .grad_arena_layout
            .clone()
            .expect("grad_arena_layout must be set when calibration_grad_retention is Some");
        let model_backward_id = emit_model_backward_bridge(
            &mut compiler,
            model_def,
            &model_name,
            _arena_layout,
            &grad_arena_layout_clone,
        )?;
        emit_calibration_backward_wrapper(
            &mut compiler,
            model_forward_id,
            model_backward_id,
            batch,
            seq,
            channels,
        )?;
    }

    let obj_bytes = map_codegen_error("finalize calib_model.o", compiler.finalize())?;
    fs::write(out_path, &obj_bytes).map_err(|e| HarnessError::Infrastructure {
        reason: format!("writing {}: {e}", out_path.display()),
    })
}

/// Emit the calibration scaffolding object (`main` + loop body).
///
/// `per_step_backward_symbols`: running-buffer symbol names that backward hooks
/// need referenced per step (spec §4.5).  Each symbol must already be declared
/// as an Export global via `finalize_plan` — the scaffolding reuses the
/// `running_data_ids` entry.  The placeholder `symbol_value` + `stack_store`
/// keeps the relocation alive so the linker sees a reference even before Task 21
/// replaces this stub with real per-head dot+abs reduction IR.
pub fn emit_calibration_scaffolding_object(
    observe_plan: &[ObservePlanEntry],
    finalize_plan: &[FinalizePlanEntry],
    arena_layout: &crate::calibration::ArenaLayout,
    sidecar_json: &[u8],
    needs_forward_pass: bool,
    needs_backward: bool,
    per_step_backward_symbols: &[String],
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

    // AWQ entries use bytes_per_element=4 (f32 max-abs running buffer, spec §4.3).
    // WGGO entries use bytes_per_element=8 (f64 per-head accumulator, spec §4.6).
    // Do NOT collapse these to a single constant — the sizes differ by design.
    let mut running_data_ids: HashMap<String, DataId> = HashMap::new();
    for entry in finalize_plan {
        let mut data = DataDescription::new();
        data.define_zeroinit((entry.channels * (entry.bytes_per_element as u32)) as usize);
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
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // path_ptr
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // path_len
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // awq_descs_ptr
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // awq_count
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // wggo_descs_ptr
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // wggo_count
    write_sidecar_sig.returns.push(AbiParam::new(cl_types::I32));
    let write_sidecar_id = module
        .declare_function("nsl_calib_write_sidecar", Linkage::Import, &write_sidecar_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calib_write_sidecar: {e}"),
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
    let calib_model_forward_id = if !needs_backward {
        Some(
            module
                .declare_function(
                    "nsl_calib_model_forward",
                    Linkage::Import,
                    &calib_model_forward_sig,
                )
                .map_err(|e| HarnessError::Infrastructure {
                    reason: format!("declare nsl_calib_model_forward: {e}"),
                })?,
        )
    } else {
        None
    };

    // Task 17 (spec §4.5): when needs_backward, the scaffolding calls
    // nsl_calib_model_backward INSTEAD of nsl_calib_model_forward.
    // The backward wrapper internally runs forward then backward, so calling
    // forward separately would be redundant.  Same 4-param ABI as forward.
    let calib_model_backward_id = if needs_backward {
        let mut sig = module.make_signature();
        sig.call_conv = call_conv;
        sig.params.push(AbiParam::new(ptr_ty));
        sig.params.push(AbiParam::new(cl_types::I64));
        sig.params.push(AbiParam::new(ptr_ty));
        sig.params.push(AbiParam::new(cl_types::I64));
        sig.returns.push(AbiParam::new(cl_types::I32));
        Some(
            module
                .declare_function("nsl_calib_model_backward", Linkage::Import, &sig)
                .map_err(|e| HarnessError::Infrastructure {
                    reason: format!("declare nsl_calib_model_backward: {e}"),
                })?,
        )
    } else {
        None
    };

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

            // Task 17 (spec §4.5): dispatch to backward wrapper when needs_backward,
            // otherwise call forward.  The backward wrapper internally runs forward
            // then backward — calling forward separately when backward is needed would
            // be redundant and incorrect.  Both share the same 4-param ABI:
            //   (model_ptr, num_weights, batch_ptr, batch_elems) -> i32
            // Spec §4.3: 4th arg is element count, not byte length.
            // `nsl_calibration_batch_at` returns out_len in bytes, so we
            // shift right by 2 (f32 = 4 bytes). Pre-v2 the wrapper ignored
            // this argument so the bytes-vs-elements mismatch was latent.
            let call_status = if needs_backward {
                let bwd_ref = module.declare_func_in_func(
                    calib_model_backward_id.expect("calib_model_backward_id is Some when needs_backward"),
                    b.func,
                );
                let bwd_call = b.ins().call(
                    bwd_ref,
                    &[model_ptr, num_weights, batch_ptr, batch_elems],
                );
                b.inst_results(bwd_call)[0]
            } else {
                let fwd_ref = module.declare_func_in_func(
                    calib_model_forward_id.expect("calib_model_forward_id is Some when !needs_backward"),
                    b.func,
                );
                let fwd_call = b.ins().call(
                    fwd_ref,
                    &[model_ptr, num_weights, batch_ptr, batch_elems],
                );
                b.inst_results(fwd_call)[0]
            };

            // v2 follow-up (spec §5.1): wrapper returns 3 on per-batch
            // shape mismatch. The scaffolding's `batch_preflight` only
            // validates batch 0; this catches drift mid-loop. Reuse the
            // same `batch_shape_mismatch_data` payload + exit code as
            // the preflight path so the surfaced reason is identical.
            let observe_continue = b.create_block();
            let batch_mismatch_mid_loop = b.create_block();
            let zero_status = b.ins().iconst(cl_types::I32, 0);
            let status_ok = b.ins().icmp(IntCC::Equal, call_status, zero_status);
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

            // Spec §4.5: hook.emit_per_step for backward observers.
            //
            // Task 21 status: WggoGradientHook::emit_per_step is implemented and
            // unit-tested (4 tests in wggo_gradient_hook.rs).  The hook's body
            // iterates (target × projection), verifies presence in grad_arena_layout,
            // and calls ctx.record_per_head_dot_call() for each valid pair.
            //
            // Full scaffolding wiring (calling emit_per_head_dot_abs_accum here)
            // requires threading per-projection weight data pointers into this
            // loop body.  In production, weight pointers come from
            // `nsl_model_get_weight_ptrs` (stored in weight_ptrs_addr above),
            // but indexing individual projections requires either:
            //   (a) `nsl_model_get_weight`-by-name for each projection (needs new
            //       import declaration + weight name string globals), or
            //   (b) pre-computed weight-index offsets passed via a new parameter
            //       to emit_calibration_scaffolding_object.
            //
            // The placeholder below (Task 18) is kept to maintain the relocation
            // entries that verify `__nsl_wggo_grad.*` symbols are referenced.
            // Replace with real emit_per_head_dot_abs_accum calls in Task 22.
            if needs_backward && !per_step_backward_symbols.is_empty() {
                // Placeholder: single scratch slot is intentional. All stores
                // overwrite offset 0; the slot exists only to prevent Cranelift DCE
                // of the symbol_value instructions. Task 22 will replace the loop
                // with a real per-symbol reduction (one slot per accumulator).
                let scratch_slot = b.create_sized_stack_slot(StackSlotData::new(
                    StackSlotKind::ExplicitSlot,
                    8,
                    0,
                ));
                for sym in per_step_backward_symbols {
                    let run_id = running_data_ids
                        .get(sym.as_str())
                        .unwrap_or_else(|| panic!(
                            "per_step_backward_symbols entry '{sym}' not found in \
                             running_data_ids — ensure it appears in finalize_plan"
                        ));
                    let run_gv = module.declare_data_in_func(*run_id, b.func);
                    let run_ptr = b.ins().symbol_value(ptr_ty, run_gv);
                    // Stack-store the pointer value to prevent DCE.
                    b.ins().stack_store(run_ptr, scratch_slot, 0);
                }
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
                let zero_ptr = b.ins().iconst(ptr_ty, 0);
                let zero_count = b.ins().iconst(ptr_ty, 0);
                let write_sidecar_ref = module.declare_func_in_func(write_sidecar_id, b.func);
                let ws_call = b.ins().call(
                    write_sidecar_ref,
                    &[fin_sidecar_ptr, sidecar_path_len, descs_base, desc_count, zero_ptr, zero_count],
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
    needs_backward: bool,
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
    if needs_backward
        && !obj
            .symbols()
            .any(|symbol| symbol.name() == Ok("nsl_calib_model_backward") && !symbol.is_undefined())
    {
        return Err(HarnessError::Infrastructure {
            reason: format!(
                "calibration: model-backward wrapper missing from calib_model.o.\n\
 requested:  link scaffolding.o <- calib_model.o with nsl_calib_model_backward resolved\n\
 expected:   calib_model.o exports nsl_calib_model_backward (the f32-buffer\n\
             wrapper around model_forward + L2 loss + model_backward)\n\
 found:      {} does not export `nsl_calib_model_backward`. Either the\n\
             NSL source lacks a model_forward function, or the\n\
             calibration-scoped compile pass skipped backward emission.",
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
        if err_str.contains("nsl_calib_model_backward") {
            HarnessError::Infrastructure {
                reason: format!(
                    "calibration: model-backward wrapper missing from calib_model.o.\n\
 requested:  link scaffolding.o <- calib_model.o with nsl_calib_model_backward resolved\n\
 expected:   calib_model.o exports nsl_calib_model_backward (the f32-buffer\n\
             wrapper around model_forward + L2 loss + model_backward)\n\
 found:      {err_str}"
                ),
            }
        } else if err_str.contains("nsl_calib_model_forward") {
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
    // AWQ entries use bytes_per_element=4 (f32 max-abs running buffer, spec §4.3).
    // WGGO entries use bytes_per_element=8 (f64 per-head accumulator, spec §4.6).
    // Do NOT collapse these to a single constant — the sizes differ by design.
    let mut running_data_ids: HashMap<String, DataId> = HashMap::new();
    for entry in finalize_plan {
        let mut data = DataDescription::new();
        data.define_zeroinit((entry.channels * (entry.bytes_per_element as u32)) as usize);
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

    // nsl_calib_write_sidecar(path_ptr, path_len, awq_descs_ptr, awq_count, wggo_descs_ptr, wggo_count) -> i32
    let mut write_sidecar_sig = module.make_signature();
    write_sidecar_sig.call_conv = call_conv;
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // path_ptr
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // path_len
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // awq_descs_ptr
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // awq_count
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // wggo_descs_ptr
    write_sidecar_sig.params.push(AbiParam::new(ptr_ty)); // wggo_count
    write_sidecar_sig.returns.push(AbiParam::new(cl_types::I32));
    let write_sidecar_id = module
        .declare_function("nsl_calib_write_sidecar", Linkage::Import, &write_sidecar_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calib_write_sidecar: {e}"),
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

                // Call nsl_calib_write_sidecar(path_ptr, path_len, awq_descs_ptr, awq_count, wggo_descs_ptr, wggo_count).
                let desc_count = b.ins().iconst(ptr_ty, finalize_plan.len() as i64);
                let zero_ptr = b.ins().iconst(ptr_ty, 0);
                let zero_count = b.ins().iconst(ptr_ty, 0);
                let write_sidecar_ref = module.declare_func_in_func(write_sidecar_id, b.func);
                let ws_call = b.ins().call(
                    write_sidecar_ref,
                    &[fin_sidecar_ptr, sidecar_path_len, descs_base, desc_count, zero_ptr, zero_count],
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
            bytes_per_element: 4, // AWQ f32 max-abs running buffer
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
            false, // needs_backward = false for this AWQ-only test
            &[],   // no per-step backward symbols for AWQ-only
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
            imports.iter().any(|symbol| symbol == "nsl_calib_write_sidecar"),
            "scaffolding.o must import nsl_calib_write_sidecar"
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
            false, // needs_backward = false
            &[],   // no per-step backward symbols
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

#[cfg(test)]
mod backward_wrapper {
    use super::*;
    use object::{Object, ObjectSymbol};
    use nsl_errors::{FileId, Level};
    use nsl_lexer::{tokenize, Interner};

    use crate::calibration::discovery::WggoGradTarget;
    use crate::calibration::observation::ProjectionRef;
    use crate::calibration::retention_pass::build_arena_layout;

    fn repo_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("..")
            .join("..")
    }

    fn fixture(name: &str) -> PathBuf {
        repo_root().join("tests").join("fixtures").join(name)
    }

    /// Parse the AWQ MLP fixture — same model used by existing calib model tests.
    fn parse_awq_fixture() -> (nsl_ast::Module, Interner) {
        let source = std::fs::read_to_string(fixture("awq_calibration_mlp.nsl"))
            .expect("fixture readable");
        let mut interner = Interner::new();
        let (tokens, lex_diags) = tokenize(&source, FileId(0), &mut interner);
        assert!(
            lex_diags.iter().all(|d| !matches!(d.level, Level::Error)),
            "fixture must lex cleanly: {lex_diags:?}"
        );
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        assert!(
            parsed.diagnostics.iter().all(|d| !matches!(d.level, Level::Error)),
            "fixture must parse cleanly: {:?}",
            parsed.diagnostics
        );
        (parsed.module, interner)
    }

    /// Build `CompileOptions` with both AWQ calibration bundle *and*
    /// `calibration_grad_retention` set to a minimal single-layer WGGO target
    /// whose projection names match the TinyMLP fixture.
    ///
    /// The WGGO target uses the same 128×64 weight shape as `TinyMLP.up_proj`
    /// so that `build_grad_arena_layout` produces a non-zero BSS section,
    /// which is required for the backward guard in `emit_calibration_model_object`
    /// to pass.
    fn compile_options_with_backward(
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

        // One WGGO target that mirrors TinyMLP's weight shapes (128×64).
        let targets = vec![WggoGradTarget {
            layer_key: "TinyMLP".into(),
            class_name: "TinyMLP".into(),
            head_dim: 64,
            w_q: ProjectionRef("TinyMLP.up_proj".into()),
            w_k: ProjectionRef("TinyMLP.up_proj".into()),
            w_v: ProjectionRef("TinyMLP.up_proj".into()),
            w_o: ProjectionRef("TinyMLP.down_proj".into()),
            w_q_shape: [128, 64],
            w_k_shape: [128, 64],
            w_v_shape: [128, 64],
            w_o_shape: [64, 128],
        }];
        opts.calibration_grad_retention = Some(targets);
        opts
    }

    #[test]
    fn calib_model_object_exports_nsl_calib_model_backward() {
        let (ast, interner) = parse_awq_fixture();
        let projections = crate::calibration::pre_scan_awq_projections_from_ast(&ast, &interner);
        let arena_layout = build_arena_layout(&projections, 8, 4);
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("calib_model.o");
        let opts = compile_options_with_backward(&ast, &interner);

        emit_calibration_model_object(&ast, &opts, &arena_layout, &out_path)
            .expect("emit succeeds with grad_retention set");

        let obj_bytes = std::fs::read(&out_path).expect("object readable");
        let obj = object::File::parse(&*obj_bytes).expect("object::File::parse");

        assert!(
            obj.symbols().any(|s| {
                s.name() == Ok("nsl_calib_model_backward") && !s.is_undefined()
            }),
            "calib_model.o must export nsl_calib_model_backward when grad-retention is set"
        );
    }

    #[test]
    fn calib_model_object_no_backward_when_grad_retention_absent() {
        let (ast, interner) = parse_awq_fixture();
        let projections = crate::calibration::pre_scan_awq_projections_from_ast(&ast, &interner);
        let arena_layout = build_arena_layout(&projections, 8, 4);
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("calib_model.o");

        // opts WITHOUT calibration_grad_retention.
        let opts = {
            let mut analysis_interner = interner.clone();
            let analysis = nsl_semantic::analyze(&ast, &mut analysis_interner);
            let mut o = crate::CompileOptions::default();
            o.calibration_batch_seq = Some((8, 4));
            o.calibration_compile_bundle = Some(std::sync::Arc::new(
                crate::calibration::CalibrationCompileBundle {
                    ast: ast.clone(),
                    interner: analysis_interner,
                    type_map: analysis.type_map.clone(),
                },
            ));
            o.weight_index_map = analysis.weight_index_map.clone();
            o
        };
        // Confirm no grad retention.
        assert!(opts.calibration_grad_retention.is_none());

        emit_calibration_model_object(&ast, &opts, &arena_layout, &out_path)
            .expect("emit succeeds without grad_retention");

        let obj_bytes = std::fs::read(&out_path).expect("object readable");
        let obj = object::File::parse(&*obj_bytes).expect("object::File::parse");

        assert!(
            !obj.symbols().any(|s| s.name() == Ok("nsl_calib_model_backward")),
            "calib_model.o must NOT export nsl_calib_model_backward when grad-retention is absent"
        );
    }

    /// Task 15: verify the backward wrapper accepts 4 ABI params (matching the forward
    /// wrapper) and compiles successfully with the full L2-loss body.
    ///
    /// This test confirms that:
    ///  1. The object still exports `nsl_calib_model_backward`.
    ///  2. `nsl_calib_model_backward` now has the 4-param ABI
    ///     (weight_ptrs, num_weights, batch_ptr, batch_elem_count) matching the forward
    ///     wrapper, not the 2-param Task-14 stub.
    ///  3. The model_forward bridge now RETURNS y (so this test would fail to compile if
    ///     the bridge still returned void and the backward wrapper tried to capture its
    ///     result).
    ///
    /// We verify #2 indirectly: if the wrapper is compiled with the wrong number of
    /// block params the Cranelift verifier would panic/error during `define_function`;
    /// the call to `emit_calibration_model_object` would return Err. We assert it
    /// returns Ok AND check that both wrappers are exported.
    #[test]
    fn backward_wrapper_emits_l2_loss_dy_alloc_and_scale() {
        let (ast, interner) = parse_awq_fixture();
        let projections = crate::calibration::pre_scan_awq_projections_from_ast(&ast, &interner);
        let arena_layout = build_arena_layout(&projections, 8, 4);
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("calib_model.o");
        let opts = compile_options_with_backward(&ast, &interner);

        // This will fail with a Cranelift verifier error if the backward wrapper's body
        // tries to call model_forward but model_forward still has a void return (Task 14).
        // After Task 15, model_forward returns y (i64), so the call captures the result.
        emit_calibration_model_object(&ast, &opts, &arena_layout, &out_path)
            .expect("emit_calibration_model_object succeeds with Task-15 L2-loss body");

        let obj_bytes = std::fs::read(&out_path).expect("object readable");
        let obj = object::File::parse(&*obj_bytes).expect("object::File::parse");

        // Both wrappers must be exported — confirmed by symbol presence.
        let fwd_exported = obj.symbols().any(|s| {
            s.name() == Ok("nsl_calib_model_forward") && !s.is_undefined()
        });
        assert!(fwd_exported, "nsl_calib_model_forward must be exported");

        let bwd_exported = obj.symbols().any(|s| {
            s.name() == Ok("nsl_calib_model_backward") && !s.is_undefined()
        });
        assert!(bwd_exported, "nsl_calib_model_backward must be exported");

        // model_forward must now be a LOCAL (non-undefined) symbol — it's defined in
        // this object (not imported).  Before Task 15 it was Export; after refactoring
        // for backward use it remains defined here.
        let model_fwd_defined = obj.symbols().any(|s| {
            s.name() == Ok("model_forward") && !s.is_undefined()
        });
        assert!(
            model_fwd_defined,
            "model_forward bridge must be defined (local/export) in calib_model.o"
        );

        // nsl_tensor_mul_scalar must be imported (undefined) — the backward wrapper
        // uses it to compute dy = 2·y.  If the Task-14 stub were still in place, this
        // function would NOT be needed by nsl_calib_model_backward specifically (though
        // the compiled model code may already reference it).
        // We verify the emission compiles without error as the primary assertion;
        // the symbol check is a belt-and-suspenders indicator.
        let has_mul_scalar = obj.symbols().any(|s| {
            s.name() == Ok("nsl_tensor_mul_scalar")
        });
        assert!(
            has_mul_scalar,
            "calib_model.o must reference nsl_tensor_mul_scalar"
        );
    }

    /// Regression test for Issue 1: a void-returning `forward` with
    /// `calibration_grad_retention` set must be refused at compile time, not
    /// silently produce a null y_handle that crashes `nsl_tensor_mul_scalar`.
    ///
    /// The backward wrapper synthesises `dy = 2·y` from the forward output;
    /// this is undefined when `y` is null (void-returning forward).
    #[test]
    fn void_forward_with_grad_retention_is_refused() {
        let source = std::fs::read_to_string(fixture("awq_calibration_mlp_void_forward.nsl"))
            .expect("void-forward fixture readable");
        let mut interner = Interner::new();
        let (tokens, lex_diags) = tokenize(&source, FileId(0), &mut interner);
        assert!(
            lex_diags.iter().all(|d| !matches!(d.level, Level::Error)),
            "fixture must lex cleanly: {lex_diags:?}"
        );
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        assert!(
            parsed.diagnostics.iter().all(|d| !matches!(d.level, Level::Error)),
            "fixture must parse cleanly: {:?}",
            parsed.diagnostics
        );
        let ast = parsed.module;

        let projections = crate::calibration::pre_scan_awq_projections_from_ast(&ast, &interner);
        let arena_layout = build_arena_layout(&projections, 8, 4);
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("calib_model_void.o");

        let opts = compile_options_with_backward(&ast, &interner);

        let result = emit_calibration_model_object(&ast, &opts, &arena_layout, &out_path);

        assert!(
            result.is_err(),
            "emit_calibration_model_object must refuse a void-returning forward \
             when calibration_grad_retention is set; got: {result:?}"
        );
        let err_msg = match result.unwrap_err() {
            HarnessError::Infrastructure { reason } => reason,
            other => panic!("expected Infrastructure error, got {other:?}"),
        };
        assert!(
            err_msg.contains("forward has no return type")
                || err_msg.contains("--wggo-importance=magnitude"),
            "error message must mention the void-forward constraint; got: {err_msg}"
        );
    }
}

/// Task 17 (spec §4.5): verify that the scaffolding loop body dispatches to
/// `nsl_calib_model_backward` when `needs_backward = true`, and to
/// `nsl_calib_model_forward` when `needs_backward = false`.
///
/// Test strategy: relocation-symbol inspection on the emitted `.o` file.
/// When `needs_backward = true`:
///   - scaffolding.o must import (undefined reference to) `nsl_calib_model_backward`
///   - scaffolding.o must NOT import `nsl_calib_model_forward`
/// When `needs_backward = false`:
///   - scaffolding.o must import `nsl_calib_model_forward`
///   - scaffolding.o must NOT import `nsl_calib_model_backward`
///
/// This ensures the codegen branches to ONE OR THE OTHER at compile time,
/// not both, which is the spec §4.5 requirement.
#[cfg(test)]
mod loop_body_dispatch {
    use object::{Object, ObjectSymbol};

    use crate::calibration::hooks::{FinalizePlanEntry, ObservePlanEntry};
    use crate::calibration::observation::ProjectionRef;
    use crate::calibration::retention_pass::build_arena_layout;

    use super::emit_calibration_scaffolding_object;

    /// Build a minimal observe/finalize plan and arena layout for dispatch tests.
    fn fixture_plans() -> (
        Vec<ObservePlanEntry>,
        Vec<FinalizePlanEntry>,
        crate::calibration::ArenaLayout,
    ) {
        let observe_plan = vec![ObservePlanEntry {
            projection: ProjectionRef("TinyMLP.up_proj".into()),
            src_offset: 0,
            rows: 32,
            channels: 64,
            running_symbol: "__nsl_awq_running_up_proj".into(),
        }];
        let finalize_plan = vec![FinalizePlanEntry {
            projection: ProjectionRef("TinyMLP.up_proj".into()),
            running_symbol: "__nsl_awq_running_up_proj".into(),
            channels: 64,
            bytes_per_element: 4, // AWQ f32 max-abs running buffer
        }];
        let arena_layout = build_arena_layout(
            &[crate::calibration::DiscoveredProjection {
                projection: ProjectionRef("TinyMLP.up_proj".into()),
                weight_shape: [128, 64],
            }],
            8,
            4,
        );
        (observe_plan, finalize_plan, arena_layout)
    }

    #[test]
    fn loop_body_calls_model_backward_when_backward_needed() {
        let (observe_plan, finalize_plan, arena_layout) = fixture_plans();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("scaffolding_bwd.o");

        emit_calibration_scaffolding_object(
            &observe_plan,
            &finalize_plan,
            &arena_layout,
            b"{}",
            true,
            true, // needs_backward = true
            &[],  // no wggo per-step symbols for this forward-dispatch test
            &out_path,
        )
        .expect("emit succeeds with needs_backward=true");

        let obj_bytes = std::fs::read(&out_path).expect("object readable");
        let obj = object::File::parse(&*obj_bytes).expect("object::File::parse");

        let imports: Vec<String> = obj
            .symbols()
            .filter(|s| s.is_undefined())
            .filter_map(|s| s.name().ok().map(str::to_string))
            .collect();

        assert!(
            imports.iter().any(|s| s == "nsl_calib_model_backward"),
            "scaffolding.o must import nsl_calib_model_backward when needs_backward=true; \
             imports: {imports:?}"
        );
        assert!(
            !imports.iter().any(|s| s == "nsl_calib_model_forward"),
            "scaffolding.o must NOT import nsl_calib_model_forward when needs_backward=true \
             (backward subsumes forward); imports: {imports:?}"
        );
    }

    #[test]
    fn loop_body_calls_only_forward_when_backward_not_needed() {
        let (observe_plan, finalize_plan, arena_layout) = fixture_plans();
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("scaffolding_fwd.o");

        emit_calibration_scaffolding_object(
            &observe_plan,
            &finalize_plan,
            &arena_layout,
            b"{}",
            true,
            false, // needs_backward = false
            &[],   // no wggo per-step symbols for this forward-only test
            &out_path,
        )
        .expect("emit succeeds with needs_backward=false");

        let obj_bytes = std::fs::read(&out_path).expect("object readable");
        let obj = object::File::parse(&*obj_bytes).expect("object::File::parse");

        let imports: Vec<String> = obj
            .symbols()
            .filter(|s| s.is_undefined())
            .filter_map(|s| s.name().ok().map(str::to_string))
            .collect();

        assert!(
            imports.iter().any(|s| s == "nsl_calib_model_forward"),
            "scaffolding.o must import nsl_calib_model_forward when needs_backward=false; \
             imports: {imports:?}"
        );
        assert!(
            !imports.iter().any(|s| s == "nsl_calib_model_backward"),
            "scaffolding.o must NOT import nsl_calib_model_backward when needs_backward=false; \
             imports: {imports:?}"
        );
    }

    /// Task 18 (spec §4.5): verify that the scaffolding loop body emits
    /// relocation entries referencing `__nsl_wggo_grad.*` symbols when
    /// `per_step_backward_symbols` is populated and `needs_backward = true`.
    ///
    /// Test strategy: relocation-symbol inspection on the emitted `.o` file.
    /// The placeholder `symbol_value` + `stack_store` in the loop body creates
    /// at least one relocation entry per symbol so the linker sees a reference.
    ///
    /// Two WGGO symbols are passed to verify the loop iterates correctly —
    /// the assertion is tightened to `>= 2` to catch any single-iteration bug.
    #[test]
    fn loop_body_invokes_emit_per_step_for_backward_hooks_after_backward_call() {
        // Build a finalize_plan that includes TWO WGGO running buffers alongside
        // the AWQ running buffer so all three are in running_data_ids.
        let observe_plan = vec![ObservePlanEntry {
            projection: ProjectionRef("TinyMLP.up_proj".into()),
            src_offset: 0,
            rows: 32,
            channels: 64,
            running_symbol: "__nsl_awq_running_up_proj".into(),
        }];
        let finalize_plan = vec![
            FinalizePlanEntry {
                projection: ProjectionRef("TinyMLP.up_proj".into()),
                running_symbol: "__nsl_awq_running_up_proj".into(),
                channels: 64,
                bytes_per_element: 4, // AWQ f32 max-abs running buffer
            },
            // WGGO running buffers — these are what emit_per_step references.
            // Uses bytes_per_element=8 (f64 per-head accumulator, spec §4.6).
            FinalizePlanEntry {
                projection: ProjectionRef("TinyMLP.__wggo_grad".into()),
                running_symbol: "__nsl_wggo_grad.TinyMLP".into(),
                channels: 4,
                bytes_per_element: 8, // WGGO f64 per-head accumulator
            },
            FinalizePlanEntry {
                projection: ProjectionRef("TinyMLP2.__wggo_grad".into()),
                running_symbol: "__nsl_wggo_grad.TinyMLP2".into(),
                channels: 4,
                bytes_per_element: 8, // WGGO f64 per-head accumulator
            },
        ];
        let arena_layout = build_arena_layout(
            &[crate::calibration::DiscoveredProjection {
                projection: ProjectionRef("TinyMLP.up_proj".into()),
                weight_shape: [128, 64],
            }],
            8,
            4,
        );
        // Two distinct WGGO symbols — verifies the loop iterates over all of them.
        let per_step_backward_symbols = vec![
            "__nsl_wggo_grad.TinyMLP".to_string(),
            "__nsl_wggo_grad.TinyMLP2".to_string(),
        ];
        let tmp = tempfile::tempdir().expect("tempdir");
        let out_path = tmp.path().join("scaffolding_wggo.o");

        emit_calibration_scaffolding_object(
            &observe_plan,
            &finalize_plan,
            &arena_layout,
            b"{}",
            true,
            true, // needs_backward = true
            &per_step_backward_symbols,
            &out_path,
        )
        .expect("emit succeeds with two WGGO per-step symbols");

        let obj_bytes = std::fs::read(&out_path).expect("object readable");
        let obj = object::File::parse(&*obj_bytes).expect("object::File::parse");

        let count = count_wggo_grad_refs(&obj);
        assert!(
            count >= 2,
            "expected ≥2 __nsl_wggo_grad.* relocation references in scaffolding.o \
             (one per symbol in per_step_backward_symbols); got {count}"
        );
    }

    /// Count relocations in the .text section that target a `__nsl_wggo_grad.*` symbol.
    fn count_wggo_grad_refs(obj: &object::File) -> usize {
        use object::{ObjectSection, ObjectSymbol, RelocationTarget};
        let mut count = 0;
        for sec in obj.sections() {
            if sec.kind() != object::SectionKind::Text {
                continue;
            }
            for (_offset, reloc) in sec.relocations() {
                if let RelocationTarget::Symbol(sym_idx) = reloc.target() {
                    if let Ok(sym) = obj.symbol_by_index(sym_idx) {
                        if sym
                            .name()
                            .map(|n| n.starts_with("__nsl_wggo_grad."))
                            .unwrap_or(false)
                        {
                            count += 1;
                        }
                    }
                }
            }
        }
        count
    }
}

/// Tests for `emit_per_head_dot_abs_accum` (Task 20, spec §4.6).
///
/// These tests operate directly on Cranelift IR text (via `func.display()`)
/// so they run without a GPU, without the subprocess, and without a fixture
/// NSL source file.  They check:
///   1. f32 → f64 promotion happens before the accumulating fadd.
///   2. The running buffer is read/written as f64 (8 bytes).
///   3. MQA/GQA replication emits one f64 store per replication slot.
#[cfg(test)]
mod per_head_dot {
    use cranelift_codegen::ir::types as cl_types;
    use cranelift_codegen::ir::{AbiParam, Function, InstBuilder, UserFuncName};
    use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};

    use super::emit_per_head_dot_abs_accum;

    /// Build a tiny test `Function`, call `emit_per_head_dot_abs_accum` with the
    /// given dimensions, finalise, and return the IR text.
    ///
    /// The three "base" pointer Values are materialised as `iconst(I64, 0)` — safe
    /// for IR text inspection; we never execute this function.
    fn emit_helper_for_test(
        n_proj_heads: u32,
        elements_per_head: u32,
        replication: u32,
    ) -> String {
        // Build a no-arg, no-return signature.
        let mut sig = cranelift_codegen::ir::Signature::new(
            cranelift_codegen::isa::CallConv::SystemV,
        );
        sig.params.push(AbiParam::new(cl_types::I64)); // grad_arena_base
        sig.params.push(AbiParam::new(cl_types::I64)); // weight_data_base
        sig.params.push(AbiParam::new(cl_types::I64)); // running_base

        let mut func = Function::with_name_signature(UserFuncName::user(0, 0), sig);
        let mut fn_builder_ctx = FunctionBuilderContext::new();
        {
            let mut b = FunctionBuilder::new(&mut func, &mut fn_builder_ctx);
            let entry = b.create_block();
            b.append_block_params_for_function_params(entry);
            b.switch_to_block(entry);
            b.seal_block(entry);

            let grad_base   = b.block_params(entry)[0];
            let weight_base = b.block_params(entry)[1];
            let running_base = b.block_params(entry)[2];

            emit_per_head_dot_abs_accum(
                &mut b,
                grad_base,
                /*src_offset=*/ 0,
                weight_base,
                running_base,
                n_proj_heads,
                elements_per_head,
                replication,
            );

            // head_exit is now current; terminate.
            b.ins().return_(&[]);
            b.finalize();
        }

        format!("{}", func.display())
    }

    /// §4.6 invariant: f32 inputs must be promoted to f64 before accumulation.
    /// Verified by checking the IR text for `fpromote`, `fadd`, and `load.f64`.
    #[test]
    fn emit_per_head_dot_abs_accum_emits_f64_accumulator() {
        let ir = emit_helper_for_test(/*n_proj_heads=*/4, /*elements_per_head=*/16, /*replication=*/1);
        assert!(
            ir.contains("fpromote"),
            "must promote f32 → f64 before accumulating; got:\n{ir}"
        );
        assert!(
            ir.contains("fadd"),
            "must accumulate (fadd); got:\n{ir}"
        );
        // Cranelift displays f64 loads as "load.f64" in the textual IR.
        assert!(
            ir.contains("load.f64"),
            "running buffer must be loaded as f64; got:\n{ir}"
        );
    }

    /// §4.6 MQA/GQA replication: K/V with n_proj_heads=1, replication=4 must
    /// write into 4 distinct f64 slots (one per Q-head it serves).
    #[test]
    fn emit_per_head_dot_abs_accum_handles_replication_for_mqa() {
        // MQA: n_proj_heads=1 (K/V), replication=4 (q-heads).
        let n_proj_heads = 1;
        let elements_per_head = 16;
        let replication = 4;
        let ir = emit_helper_for_test(/*n_proj_heads=*/n_proj_heads, /*elements_per_head=*/elements_per_head, /*replication=*/replication);

        // The replication loop is unrolled at codegen time; element and head loops remain.
        // Textual IR shows one dynamic trace per loop iteration, with `replication` unrolled
        // fadd-store pairs per element iteration. With replication=4, each element iteration
        // contributes 4 fadds. The full execution (1 head × 16 elements × 4 replication) = 64 fadds total,
        // but the static IR text shows ≥replication fadds.
        let expected_fadds = replication as usize;
        let fadd_count = ir.matches("fadd").count();
        assert!(
            fadd_count >= expected_fadds,
            "expected ≥{expected_fadds} fadd instructions (one per replication slot) for replication={replication}; \
             got {fadd_count}:\n{ir}"
        );

        // Also verify the load.f64 count is >= replication (one per replication slot per element iteration).
        let expected_loads = replication as usize;
        let load_f64_count = ir.matches("load.f64").count();
        assert!(
            load_f64_count >= expected_loads,
            "expected ≥{expected_loads} load.f64 (one per replication slot) for replication={replication}; got {load_f64_count}:\n{ir}"
        );
    }
}
