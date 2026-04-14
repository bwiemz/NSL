//! Emits the `calibration_main()` entrypoint and links it into a
//! standalone binary the harness driver spawns as a subprocess.
//!
//! **MVP scope (Task 11):** only hooks with `ObservationSet::Empty`,
//! `Weights(..)`, or `BackwardGradients(..)` requirements are
//! supported.  Hooks requiring `ForwardActivations` or
//! `LinearInputActivations` (e.g. AWQ) return an `Infrastructure`
//! error — full model-forward emission into `calibration_main` is a
//! follow-up plan.  Today AWQ instead routes through the stub
//! simulator seam (`run_harness_simulated` + `run_harness_stub`).
//!
//! **Strategy for this MVP:**
//!
//! 1. Because IdentityHook (and similar observation-free hooks) produce
//!    deterministic output that does not depend on any runtime state,
//!    we can run each hook's `emit_init` / `emit_per_step` /
//!    `emit_finalize` *at codegen time* against `CalibCtx::stub_for_tests()`.
//! 2. The resulting sidecar JSON is baked into the emitted object as a
//!    `.rodata` byte array.
//! 3. The emitted `main()` reads the env var `NSL_CALIBRATION_SIDECAR_PATH`
//!    (via libc `getenv`) and writes the embedded JSON bytes to that
//!    path by calling `nsl_write_file` from `nsl-runtime` (which is
//!    already linked by `linker::link_multi`).
//! 4. Returns 0 on success, 2 on infra failure (env var missing),
//!    which `run_subprocess` classifies as `Infrastructure`.
//!
//! This still exercises the real emit → link → spawn → read-sidecar
//! loop end-to-end; only the IR body is trivial.  When AWQ arrives,
//! `calibration_main` will actually include the model forward pass and
//! hook IR, but the object-file / link / spawn plumbing established
//! here is reused as-is.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use cranelift_codegen::ir::{types as cl_types, AbiParam, Function, InstBuilder, UserFuncName};
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

const ENV_VAR_NAME: &str = "NSL_CALIBRATION_SIDECAR_PATH";

pub fn real_subprocess_entry(
    cfg: &HarnessConfig,
    registry: &HookRegistry,
) -> Result<HarnessOutput, HarnessError> {
    // MVP scope gate: reject hooks that need forward activations or
    // linear-input activations.  Full model-forward emission lands in
    // a follow-up plan.
    for hook in registry.iter() {
        if hook.requires().needs_forward_pass() {
            return Err(HarnessError::Infrastructure {
                reason: format!(
                    "hook '{}' requires LinearInputActivations which is not yet supported by real_subprocess_entry (follow-up plan)",
                    hook.id()
                ),
            });
        }
    }

    // 1. Run hooks in-process against a stub ctx to collect sidecar bytes.
    //    Valid here only because observation-free hooks (IdentityHook and
    //    similar) don't read any per-step state.
    let sidecar = build_sidecar_from_stub(cfg, registry)?;
    let sidecar_json = serde_json::to_vec(&sidecar).map_err(|e| HarnessError::Infrastructure {
        reason: format!("serializing sidecar JSON: {e}"),
    })?;

    // 2. Set up temp dir with RAII cleanup.
    let tmp = create_tmp_dir()?;
    let _guard = TmpCleanup(tmp.clone());

    let sidecar_path = tmp.join("sidecar.json");

    // 3. Emit object → link binary.
    let binary_path = emit_and_link_calibration_binary(&tmp, &sidecar_json)?;

    // 4. Spawn via env-var convention.
    // SAFETY: setting a process-wide env var is not thread-safe if the
    // harness is ever invoked concurrently from the same process.  The
    // compiler driver runs calibration sequentially, so this is fine
    // today.  Document the constraint; follow-up plan can pass the
    // path via a CLI arg instead once calibration_main is no longer
    // trivial.
    std::env::set_var(ENV_VAR_NAME, &sidecar_path);

    let outcome = run_subprocess(&binary_path, Duration::from_secs(cfg.timeout_secs))
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

    // 5. Read sidecar JSON the subprocess wrote.
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
/// assemble a `Sidecar`.  Driver callers overwrite the identity fields
/// (`checkpoint_sha256` etc.) afterwards — we leave them as placeholders
/// here.  Returns `Degenerate` / `Infrastructure` on hook-reported
/// failure.
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

/// Emit a Cranelift object file whose `main()`:
///   * calls `getenv("NSL_CALIBRATION_SIDECAR_PATH")`
///   * if NULL, returns 2 (Infrastructure per subprocess status map)
///   * else calls `nsl_write_file(path, sidecar_json)` (both C-strings)
///   * returns 0
///
/// Then link it against `nsl-runtime` (which provides `nsl_write_file`)
/// + libc (provides `getenv`) into a standalone executable.  Returns
/// the executable path on success.
fn emit_and_link_calibration_binary(
    tmp: &Path,
    sidecar_json: &[u8],
) -> Result<PathBuf, HarnessError> {
    // Sidecar bytes are valid JSON → pure ASCII/UTF-8, no NUL bytes.
    // `nsl_write_file` takes a C-string, so we need a trailing NUL.
    // Sanity-check we don't silently truncate.
    if sidecar_json.contains(&0u8) {
        return Err(HarnessError::Infrastructure {
            reason: "sidecar JSON contains embedded NUL byte (would truncate when passed as C-string)"
                .into(),
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
        // Match the main compiler's Windows x64 override for ABI correctness.
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

    // ── Embed the env-var name (NUL-terminated) ─────────────────────
    let mut env_name_bytes: Vec<u8> = ENV_VAR_NAME.as_bytes().to_vec();
    env_name_bytes.push(0);
    let env_name_data = module
        .declare_data(
            "__nsl_calibration_env_name",
            Linkage::Local,
            false, // not writable
            false, // not thread-local
        )
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare env-name data: {e}"),
        })?;
    let mut env_name_desc = DataDescription::new();
    env_name_desc.define(env_name_bytes.into_boxed_slice());
    module
        .define_data(env_name_data, &env_name_desc)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("define env-name data: {e}"),
        })?;

    // ── Embed the sidecar JSON (NUL-terminated) ─────────────────────
    let mut json_bytes: Vec<u8> = sidecar_json.to_vec();
    json_bytes.push(0);
    let json_data = module
        .declare_data(
            "__nsl_calibration_sidecar_json",
            Linkage::Local,
            false,
            false,
        )
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

    // ── Declare extern fns: getenv + nsl_write_file ─────────────────
    // getenv uses the C call conv (SystemV/Win64 depending on platform);
    // cranelift's default_call_conv on ISA matches the C ABI on the
    // host, and we override to WindowsFastcall on Windows x64 above —
    // which is ALSO the Win64 C ABI.  So `call_conv` is correct for
    // both libc and nsl-runtime `extern "C"` entry points.
    let mut getenv_sig = module.make_signature();
    getenv_sig.call_conv = call_conv;
    getenv_sig.params.push(AbiParam::new(cl_types::I64)); // const char* name
    getenv_sig.returns.push(AbiParam::new(cl_types::I64)); // char*
    let getenv_id = module
        .declare_function("getenv", Linkage::Import, &getenv_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare getenv: {e}"),
        })?;

    let mut write_file_sig = module.make_signature();
    write_file_sig.call_conv = call_conv;
    write_file_sig.params.push(AbiParam::new(cl_types::I64)); // path
    write_file_sig.params.push(AbiParam::new(cl_types::I64)); // content
    let write_file_id = module
        .declare_function("nsl_write_file", Linkage::Import, &write_file_sig)
        .map_err(|e| HarnessError::Infrastructure {
            reason: format!("declare nsl_write_file: {e}"),
        })?;

    // ── Declare main() ──────────────────────────────────────────────
    let mut main_sig = module.make_signature();
    main_sig.call_conv = call_conv;
    main_sig.params.push(AbiParam::new(cl_types::I32)); // argc
    main_sig.params.push(AbiParam::new(cl_types::I64)); // argv
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

        let entry = b.create_block();
        let infra_err = b.create_block();
        let do_write = b.create_block();
        b.append_block_params_for_function_params(entry);

        // Entry: call getenv(env_name_ptr)
        b.switch_to_block(entry);
        b.seal_block(entry);

        let env_name_gv = module.declare_data_in_func(env_name_data, b.func);
        let env_name_ptr = b.ins().symbol_value(cl_types::I64, env_name_gv);

        let getenv_ref = module.declare_func_in_func(getenv_id, b.func);
        let getenv_call = b.ins().call(getenv_ref, &[env_name_ptr]);
        let path_ptr = b.inst_results(getenv_call)[0];

        // if path_ptr == 0 → infra_err; else → do_write
        b.ins().brif(path_ptr, do_write, &[], infra_err, &[]);

        // infra_err: return 2
        b.switch_to_block(infra_err);
        b.seal_block(infra_err);
        let two = b.ins().iconst(cl_types::I32, 2);
        b.ins().return_(&[two]);

        // do_write: call nsl_write_file(path_ptr, json_ptr); return 0
        b.switch_to_block(do_write);
        b.seal_block(do_write);
        let json_gv = module.declare_data_in_func(json_data, b.func);
        let json_ptr = b.ins().symbol_value(cl_types::I64, json_gv);
        let write_ref = module.declare_func_in_func(write_file_id, b.func);
        b.ins().call(write_ref, &[path_ptr, json_ptr]);
        let zero = b.ins().iconst(cl_types::I32, 0);
        b.ins().return_(&[zero]);

        b.finalize();
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

    // ── Link (runtime lib auto-discovered by linker::link_multi) ────
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
