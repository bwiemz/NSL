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
//!   6. Calls `hook.emit_observe_batch` per batch (no-op for AWQ in this
//!      task; Task 7 wires the real reduction).
//!   7. Calls `hook.emit_finalize` → serialises result to sidecar_path
//!      via `nsl_write_file`.
//!   8. Calls `nsl_calibration_free`; returns 0.
//!
//! For hooks whose `requires()` returns `ObservationSet::Empty` or similar
//! (i.e. `needs_forward_pass() == false`), we still run through the same
//! path but skip the forward-loop and go directly to emit_finalize.  This
//! is done in-process inside `emit_and_link_calibration_binary` before
//! emitting the Cranelift object, so IdentityHook continues to work.
//!
//! **Subprocess argv convention (Task 6):**
//!   argv[1] = data_path
//!   argv[2] = sidecar_path
//!   argv[3] = weights_path
//!
//! No environment variables are used.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::{types as cl_types, AbiParam, Function, InstBuilder, MemFlags, UserFuncName};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{DataDescription, Linkage, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

use crate::calibration::ctx::CalibCtx;
use crate::calibration::hooks::CalibrationResult;
use crate::calibration::registry::HookRegistry;
use crate::calibration::sidecar::{Sidecar, SIDECAR_VERSION};
use crate::calibration::subprocess::{run_subprocess, SubprocessOutcome};
use crate::calibration::{HarnessConfig, HarnessError, HarnessOutput};

pub fn real_subprocess_entry(
    cfg: &HarnessConfig,
    registry: &HookRegistry,
) -> Result<HarnessOutput, HarnessError> {
    // Build the calibration binary (handles both forward-needing and
    // forward-free hooks via the same code path).
    let tmp = create_tmp_dir()?;
    let _guard = TmpCleanup(tmp.clone());

    let sidecar_path = tmp.join("sidecar.json");

    // Build in-process sidecar for forward-free hooks; forward-needing hooks
    // produce their sidecar from the subprocess run after model_forward.
    // The emitted Cranelift binary always calls nsl_write_file(sidecar_path, json)
    // at the end, so for forward-free hooks we bake the JSON into the binary's
    // .rodata and write it out after the (empty) batch loop.
    let needs_forward = registry.iter().any(|h| h.requires().needs_forward_pass());

    // For non-forward hooks: compute sidecar in-process so we can embed the
    // JSON in the binary. For forward hooks: the sidecar_json embedded is a
    // placeholder; the subprocess computes real scales (Task 7 wires this).
    let sidecar_json_for_binary = if needs_forward {
        // Embed placeholder JSON; the subprocess will overwrite it with real data
        // once Task 7 is implemented. For now the binary writes this placeholder.
        let placeholder = Sidecar {
            version: SIDECAR_VERSION,
            checkpoint_sha256: String::new(),
            calibration_data_sha256: String::new(),
            hook_set_sha256: String::new(),
            cache_key_digest: String::new(),
            num_samples_used: 0,
            hooks: BTreeMap::new(),
        };
        serde_json::to_vec(&placeholder).map_err(|e| HarnessError::Infrastructure {
            reason: format!("serializing placeholder sidecar: {e}"),
        })?
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

    // Emit and link the calibration binary.
    let binary_path = emit_and_link_calibration_binary(
        &tmp,
        &sidecar_json_for_binary,
        needs_forward,
        &cfg.calibration_data.to_string_lossy(),
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
///      `nsl_calibration_batch_at` and `nsl_model_load` (weights).
///      (Task 7 wires in real per-batch observe_batch; for now just
///       does the loop structure to validate the plumbing.)
///   5. Calls `nsl_calibration_free(batches)`.
///   6. Calls `nsl_write_file(sidecar_path_ptr, sidecar_json_ptr)`.
///   7. Returns 0.
///
/// The `sidecar_json` bytes (already computed in-process for non-forward
/// hooks, or a placeholder for forward hooks) are embedded in `.rodata`.
fn emit_and_link_calibration_binary(
    tmp: &Path,
    sidecar_json: &[u8],
    needs_forward_pass: bool,
    _data_path_hint: &str,
) -> Result<PathBuf, HarnessError> {
    if sidecar_json.contains(&0u8) {
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

    // ── Embed the sidecar JSON (NUL-terminated) ─────────────────────
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
    // Declared for completeness; Task 7 will call it once batches_ptr is
    // properly threaded to the loop_exit block.  Currently not emitted
    // because the subprocess exits immediately after write_file.
    let mut calib_free_sig = module.make_signature();
    calib_free_sig.call_conv = call_conv;
    calib_free_sig.params.push(AbiParam::new(cl_types::I64)); // *mut Batches
    let _calib_free_id = module
        .declare_function("nsl_calibration_free", Linkage::Import, &calib_free_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_calibration_free: {e}"),
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
        //   loop_body: batch_at + [observe placeholder] → loop_header
        //   loop_exit(sidecar_ptr): → finalize
        //   finalize(sidecar_ptr): nsl_write_file → return 0

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

            // write_block: single predecessor (entry), seal immediately.
            // argv is visible here because write_block is dominated by entry
            // and argv is a block param of entry.  In Cranelift FunctionBuilder,
            // values from dominating blocks ARE visible in successor blocks
            // with a single predecessor (entry seals before branching, so
            // write_block sees entry's values directly).
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

            // loop_body: call nsl_calibration_batch_at + (Task 7: observe_batch)
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
            // Task 7: emit_observe_batch IR goes here.  Currently a no-op.

            let one_i64 = b.ins().iconst(cl_types::I64, 1);
            let i_next = b.ins().iadd(i_cur, one_i64);
            b.ins().jump(loop_header, &[lh_batches, lh_sidecar_ptr, i_next]);

            b.seal_block(loop_header);

            // loop_exit: jump to finalize (nsl_calibration_free deferred to Task 7
            // when batches_ptr is threaded via a second block param)
            b.switch_to_block(loop_exit);
            b.seal_block(loop_exit);
            let le_sidecar_ptr = b.block_params(loop_exit)[0];
            b.ins().jump(finalize, &[le_sidecar_ptr]);

            // finalize: nsl_write_file(sidecar_path, json_ptr); return 0
            b.switch_to_block(finalize);
            b.seal_block(finalize);
            let fin_sidecar_ptr = b.block_params(finalize)[0];

            let json_gv = module.declare_data_in_func(json_data, b.func);
            let json_ptr = b.ins().symbol_value(cl_types::I64, json_gv);
            let write_ref = module.declare_func_in_func(write_file_id, b.func);
            b.ins().call(write_ref, &[fin_sidecar_ptr, json_ptr]);

            let zero = b.ins().iconst(cl_types::I32, 0);
            b.ins().return_(&[zero]);

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
