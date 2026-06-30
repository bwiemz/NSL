//! `nsl run` — compile to a temp executable, run it, forward the exit code.
//!
//! Extracted verbatim from the former monolithic `build.rs`; behavior is
//! unchanged.

use std::path::PathBuf;
use std::process;

use super::normal::run_build_inner;

#[allow(clippy::too_many_arguments)] // CLI dispatcher, not a library API
pub(crate) fn run_run(file: &PathBuf, program_args: &[String], profile_memory: bool, profile_kernels: bool, profile: bool, cuda_sync: bool, gpu_mem_report: bool, options: &nsl_codegen::CompileOptions) {
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

    // Execute the compiled program
    let mut cmd = std::process::Command::new(&exe_path);
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
    let status = cmd
        .status()
        .unwrap_or_else(|e| {
            eprintln!("error: could not execute '{}': {e}", exe_path.display());
            process::exit(1);
        });

    // Merge profile traces before exiting (process::exit won't return)
    if profile {
        crate::commands::profile_merge::merge_profile_traces("memory_profile.json", "kernel_profile.json", "profile.json");
    }

    // Clean up
    let _ = std::fs::remove_file(&exe_path);
    let _ = std::fs::remove_dir(&temp_dir);

    // Forward exit code
    process::exit(status.code().unwrap_or(1));
}
