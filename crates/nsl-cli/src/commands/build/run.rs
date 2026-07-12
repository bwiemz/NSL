//! `nsl run` — compile to a temp executable, run it, forward the exit code.
//!
//! Extracted verbatim from the former monolithic `build.rs`; behavior is
//! unchanged. Split into `build_to_temp` + `execute_temp_build` so the
//! dev-tools live health monitor (`nsl run --monitor` on a train block) can
//! spawn its poller thread BETWEEN the build and the child execution while
//! sharing exactly this code — see `commands/run.rs`.

use std::path::PathBuf;
use std::process;

use super::normal::run_build_inner;

/// A compiled program staged in a temp directory, ready to execute.
pub(crate) struct TempBuild {
    pub exe_path: PathBuf,
    pub temp_dir: PathBuf,
}

/// Build `file` to a temp executable and render the post-compile CPDT report
/// (mirroring the `nsl build` path). Shared by `run_run` and the live-monitor
/// path so both stay in lock-step by construction.
pub(crate) fn build_to_temp(
    file: &PathBuf,
    options: &nsl_codegen::CompileOptions,
) -> TempBuild {
    let temp_dir = std::env::temp_dir().join(format!("nsl_run_{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("error: could not create temp dir: {e}");
        process::exit(1);
    }

    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("program");
    let exe_name = if cfg!(target_os = "windows") {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    };
    let exe_path = temp_dir.join(&exe_name);

    // Build to temp dir (reuse existing build logic, quiet mode)
    run_build_inner(file, Some(exe_path.clone()), false, false, true, options, None);

    // CPDT: post-compile rendering, mirroring the `nsl build` path. Stderr
    // diagnostics always fire when CPDT ran; the stdout plan only with
    // --cpdt-report. The plan slot was populated during the compile above.
    if let Some(slot) = options.cpdt.plan_out.as_ref() {
        if let Some(plan) = slot.lock().ok().and_then(|g| g.clone()) {
            for diag in &plan.override_diagnostics {
                eprintln!(
                    "[cpdt] scope:global wggo-override-rejected requested={} applied={} reason={:?}",
                    diag.requested, diag.applied, diag.reason
                );
            }
            if options.cpdt.report_requested {
                print!("{}", plan.render_report());
                println!();
                println!("=== Defaults Assumed ===");
                println!("precision_cfg: BF16-mixed (override: --cpdt-precision, future)");
                let jc = nsl_codegen::cpdt_joint::JointConfig::default();
                println!("joint_cfg:     {:?} (override: --cpdt-budget, future)", jc);
                println!("expert_cfg:    none (no MoE block detected)");
                match &options.weight_file {
                    Some(p) => println!("weights:       {}", p.display()),
                    None => println!(
                        "weights:       none (no --weights flag and no AST load_safetensors)"
                    ),
                }
            }
        }
    }

    TempBuild { exe_path, temp_dir }
}

/// Execute a staged `TempBuild`, merge profile traces, clean up, and RETURN
/// the child's exit code (unlike `run_run`, which forwards it via
/// `process::exit`). A spawn failure prints and `process::exit(1)`s exactly
/// as the original monolithic path did.
#[allow(clippy::too_many_arguments)] // CLI dispatcher, not a library API
pub(crate) fn execute_temp_build(
    build: &TempBuild,
    program_args: &[String],
    profile_memory: bool,
    profile_kernels: bool,
    profile: bool,
    cuda_sync: bool,
    gpu_mem_report: bool,
) -> i32 {
    let mut cmd = std::process::Command::new(&build.exe_path);
    cmd.args(program_args);
    if profile_memory || profile {
        cmd.env("NSL_PROFILE_MEMORY", "1");
    }
    if profile_kernels || profile {
        cmd.env("NSL_PROFILE_KERNELS", "1");
    }
    if cuda_sync {
        cmd.env("NSL_CUDA_SYNC", "1");
    }
    if gpu_mem_report {
        // ELTLS: instructs the runtime (via atexit hook in nsl_args_init)
        // to print the GPU memory report after the compiled main returns.
        cmd.env("NSL_GPU_MEM_REPORT", "1");
    }
    let status = cmd.status().unwrap_or_else(|e| {
        eprintln!("error: could not execute '{}': {e}", build.exe_path.display());
        process::exit(1);
    });

    // Merge profile traces before returning.
    if profile {
        crate::commands::profile_merge::merge_profile_traces(
            "memory_profile.json",
            "kernel_profile.json",
            "profile.json",
        );
    }

    // Clean up
    let _ = std::fs::remove_file(&build.exe_path);
    let _ = std::fs::remove_dir(&build.temp_dir);

    status.code().unwrap_or(1)
}

#[allow(clippy::too_many_arguments)] // CLI dispatcher, not a library API
pub(crate) fn run_run(file: &PathBuf, program_args: &[String], profile_memory: bool, profile_kernels: bool, profile: bool, cuda_sync: bool, gpu_mem_report: bool, options: &nsl_codegen::CompileOptions) {
    let build = build_to_temp(file, options);
    let code = execute_temp_build(
        &build,
        program_args,
        profile_memory,
        profile_kernels,
        profile,
        cuda_sync,
        gpu_mem_report,
    );
    // Forward exit code
    process::exit(code);
}
