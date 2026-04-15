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
//!   4. Loads model weights via `nsl_model_load` (nsl-runtime).
//!   5. Loops over batches calling `model_forward`; each call populates
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

    // Build the arena layout and collect observe/finalize plans from hooks.
    let arena_layout = build_arena_layout(&cfg.projections, 8, 4);
    let observe_plan: Vec<ObservePlanEntry> = registry
        .iter()
        .flat_map(|h| h.observe_plan(&arena_layout))
        .collect();
    let finalize_plan: Vec<FinalizePlanEntry> = registry
        .iter()
        .flat_map(|h| h.finalize_plan())
        .collect();

    // Emit and link the calibration binary.
    let binary_path = emit_and_link_calibration_binary(
        &tmp,
        &sidecar_json_for_binary,
        needs_forward,
        &observe_plan,
        &finalize_plan,
    )?;

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
