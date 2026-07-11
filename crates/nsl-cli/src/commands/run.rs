//! `nsl run` — compile and execute an NSL program (with profiling/monitor).
//!
//! Extracted from main_inner; behavior is unchanged.

use std::path::PathBuf;
use std::process;

use nsl_lexer::Interner;

pub(crate) fn dispatch(args: crate::args::RunArgs) {
    let crate::args::RunArgs {
            file,
            args,
            profile_memory,
            profile_kernels,
            profile,
            devices,
            prefill_workers,
            decode_workers,
            target,
            disable_fusion,
            tape_ad,
            source_ad,
            pretrain_optimized,
            debug_training,
            trace_ops,
            deterministic,
            distribute: _distribute,
            zero_stage,
            wcet,
            wcet_cert,
            gpu,
            cpu,
            do178c_report,
            wcet_target,
            fpga_device,
            cuda_sync,
            gpu_mem_report,
            monitor,
            inspect,
            csha,
            csha_report,
            linear_types,
            cpdt,
            cpdt_num_gpus,
            cpdt_intra_bw,
            cpdt_inter_bw,
            cpdt_report,
            wggo,
            wggo_report,
            wggo_weights,
            wggo_importance,
            wggo_prune_fraction,
            weights,
    } = args;

    // Meta-flag expansion (roadmap 3.3) — see the twin call in build/options.rs.
    let mut wggo = wggo;
    let mut csha = csha;
    let mut source_ad = source_ad;
    crate::meta_flags::expand_pretrain_optimized(
        pretrain_optimized,
        &mut wggo,
        &mut csha,
        &mut source_ad,
    );

            // CSHA: validate the same way `nsl build --csha` does so an
            // unrecognised mode fails fast rather than silently disabling
            // the planner.
            if let Some(ref m) = csha {
                if nsl_codegen::csha::CshaMode::parse(m).is_none() {
                    eprintln!(
                        "error: --csha value '{}' is not one of auto|boundary|pipeline|block|off",
                        m
                    );
                    process::exit(1);
                }
            }

            // WGGO: validate the same way `nsl build --wggo` does (build's
            // options.rs) so an unrecognised mode / out-of-range prune fraction
            // / missing sidecar fails fast rather than silently no-op'ing. The
            // error strings are kept byte-identical to the build path.
            if let Some(ref m) = wggo {
                if nsl_codegen::wggo::WggoMode::parse(m).is_none() {
                    eprintln!(
                        "error: --wggo value '{}' is not one of full|greedy|off|auto",
                        m
                    );
                    process::exit(1);
                }
            }
            // wggo_importance is a typed CliWggoImportance enum; clap rejects
            // unknown values before we get here. The Grad variant requires a
            // calibration sidecar — enforced downstream at compile time.
            if let Some(f) = wggo_prune_fraction {
                if !(0.0..=0.9).contains(&f) {
                    eprintln!(
                        "error: --wggo-prune-fraction must be in [0.0, 0.9], got {}",
                        f
                    );
                    process::exit(1);
                }
            }
            if let Some(ref p) = wggo_weights {
                if !p.exists() {
                    eprintln!(
                        "error: --wggo-weights path does not exist: {}",
                        p.display()
                    );
                    process::exit(1);
                }
            }

            // CPDT: mirror the `nsl build` setup so precision-adaptive
            // training executes end-to-end via `nsl run`. The cpdt_mode,
            // cpdt_cluster and cpdt_plan_out are threaded into the compile
            // call via CompileOptions fields below (the compiler copies
            // options.cpdt_cluster into Compiler::cpdt_cluster and reads
            // compile_options.cpdt_plan_out during train-block codegen).
            //
            // --cpdt-report implies --cpdt (full mode unless explicit).
            let cpdt_mode_str: Option<String> = match (cpdt.as_deref(), cpdt_report) {
                (Some(s), _) => Some(s.to_string()),
                (None, true) => Some("full".to_string()),
                (None, false) => None,
            };
            let cpdt_mode = match cpdt_mode_str.as_deref() {
                None => nsl_codegen::cpdt::CpdtMode::Off,
                Some(s) => match nsl_codegen::cpdt::CpdtMode::parse(s) {
                    Some(m) => m,
                    None => {
                        eprintln!(
                            "error: --cpdt value '{}' is not one of full|zero_only|off",
                            s
                        );
                        process::exit(2);
                    }
                },
            };
            let cpdt_cluster = if cpdt_mode != nsl_codegen::cpdt::CpdtMode::Off {
                let n = match cpdt_num_gpus {
                    Some(n) if n >= 1 => n,
                    Some(_) => {
                        eprintln!("nsl: --cpdt-num-gpus must be >= 1");
                        process::exit(2);
                    }
                    None => {
                        eprintln!("nsl: --cpdt requires --cpdt-num-gpus N");
                        process::exit(2);
                    }
                };
                Some(nsl_codegen::cpdt_zero::ClusterSpec {
                    num_gpus: n,
                    memory_budget_bytes: 80u64 * 1024 * 1024 * 1024,
                    intra_bw_bps: cpdt_intra_bw,
                    inter_bw_bps: cpdt_inter_bw,
                    gpus_per_node: n.min(8),
                })
            } else {
                None
            };
            let cpdt_plan_out: Option<
                std::sync::Arc<std::sync::Mutex<Option<nsl_codegen::cpdt::CpdtPlan>>>,
            > = if cpdt_mode != nsl_codegen::cpdt::CpdtMode::Off {
                Some(std::sync::Arc::new(std::sync::Mutex::new(None)))
            } else {
                None
            };

            // Phase 1 CPDT: AST-scan for load_safetensors(...) + @cpdt(weight_aware=...).
            // Resolve the effective weight file via the four-case decision
            // table from the Phase 1 spec §2.1 (identical to the build path).
            // `nsl run` has no --standalone flag, so the build path's
            // `!standalone` guard is always satisfied here.
            let resolved_weight_file: Option<PathBuf> = {
                let ast_source = std::fs::read_to_string(&file).unwrap_or_default();
                let mut ast_interner = Interner::new();
                let ast_file_id = nsl_errors::FileId(0);
                let (ast_tokens, _) =
                    nsl_lexer::tokenize(&ast_source, ast_file_id, &mut ast_interner);
                let ast_parse = nsl_parser::parse(&ast_tokens, &mut ast_interner);
                let ast_weight_ref =
                    crate::ast_scan::find_ast_weight_ref(&ast_parse.module, &ast_interner);
                let ast_weight_aware =
                    crate::ast_scan::find_ast_cpdt_weight_aware(&ast_parse.module, &ast_interner);

                match (&weights, &ast_weight_ref) {
                    (Some(flag_path), Some(ast_path)) => {
                        eprintln!(
                            "warning: --weights {} overrides AST-declared load_safetensors({:?}).",
                            flag_path.display(),
                            ast_path.display(),
                        );
                        Some(flag_path.clone())
                    }
                    (Some(flag_path), None) => Some(flag_path.clone()),
                    (None, Some(ast_path)) => Some(ast_path.clone()),
                    (None, None) => {
                        let cpdt_enabled = cpdt_mode != nsl_codegen::cpdt::CpdtMode::Off;
                        let weight_aware = ast_weight_aware.unwrap_or(true);
                        if cpdt_enabled && weight_aware {
                            eprintln!(
                                "error: --cpdt {} requires weights. Resolve by ONE of:\n\
                                 \n\
                                 1. Add --weights <path.safetensors> to this invocation.\n\
                                 2. Add `let w = load_safetensors(\"<path>\")` to your NSL source.\n\
                                 3. Add `@cpdt(weight_aware=false)` to opt out of the weight-aware path\n\
                                    (produces a CPDT plan without weight-derived tier assignments).",
                                cpdt_mode.as_str(),
                            );
                            process::exit(1);
                        }
                        None
                    }
                }
            };

            // Dev Tools Phase 4 Task 6: when --monitor is set, auto-detect
            // whether this program has a `train { }` block.  If so, enable
            // the health monitor and skip the Phase 1/2 kernel-timing path
            // (the two are mutually exclusive per spec § 4.6).  Non-train
            // programs keep the existing kernel-timing behavior.
            let detected_train_block: bool = if monitor {
                let src = std::fs::read_to_string(&file).unwrap_or_default();
                match nsl_cli::shape_debug::ShapeDebugInput::from_source(
                    &src,
                    file.to_str().unwrap_or("<file>"),
                ) {
                    Ok(input) => crate::has_train_block(&input.module),
                    Err(_) => false,
                }
            } else {
                false
            };

            if monitor && !detected_train_block {
                let profile_args = nsl_cli::profile::ProfileArgs {
                    file: file.clone(),
                    target: gpu.clone().unwrap_or_else(|| "h100".to_string()),
                    dtype: "bf16".into(),
                    batch: 1,
                    seq: 2048,
                    dim: vec![],
                    fusion: true,
                    memory: false,
                    entry: "auto".into(),
                    json: true,
                    explain_wggo: false,
                };
                let report_json = match nsl_cli::profile::run_profile(&profile_args) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("error: {e}");
                        process::exit(1);
                    }
                };
                let report: nsl_codegen::profiling::types::ProfileReport =
                    match serde_json::from_str(&report_json) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("error: profile JSON parse: {e}");
                            process::exit(1);
                        }
                    };
                let manifest_path =
                    match nsl_cli::monitor::write_manifest_beside(&file, &report) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("error: {e}");
                            process::exit(1);
                        }
                    };
                let actual_path = file.with_extension("nsl-profile-actual.json");
                match nsl_cli::monitor::run_monitor(&file, &manifest_path, &actual_path) {
                    Ok(rendered) => {
                        println!("{}", rendered);
                        return;
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        process::exit(1);
                    }
                }
            }
            let compile_opts = nsl_codegen::CompileOptions {
                no_autotune: false,
                autotune_fresh: false,
                // Clamp like `nsl build` (build/options.rs) — `--devices 0` must not
                // produce world_size=0 (WGGO ZeRO/TP math assumes >= 1 rank).
                world_size: (devices as usize).max(1),
                fusion_report: false,
                vram_budget: None,
                memory_report: false,
                target,
                disable_fusion,
                tape_ad,
                source_ad,
                trace_ops,
                nan_analysis: false,
                deterministic,
                // CPDT: pass through the four-case-resolved weight file so the
                // weight-aware tier assignment runs during train-block codegen.
                weight_file: resolved_weight_file.clone(),
                weight_config: Default::default(),
                weight_analysis: false,
                unikernel_config: None,
                wcet: nsl_codegen::WcetOptions {
                    enabled: wcet,
                    gpu,
                    cpu,
                    report_path: wcet_cert,
                    safety_margin: 1.05,
                    do178c_report,
                    target: wcet_target,
                    fpga_device,
                },
                // M55: ZK flags not exposed on `run`; use defaults.
                zk: nsl_codegen::ZkOptions::default(),
                linear_types_enabled: linear_types, // Task 20: nsl run now exposes --linear-types
                ownership_info: std::collections::HashMap::new(),
                zero_stage: zero_stage.map(|s| s as u8),
                debug_training,
                shared_lib: false,
                emit_export_table: false,
                wrga_inputs: None,
                fused_ce_configs: Vec::new(),
                pca_user_strategies: Vec::new(),
                wrga_fold_allocations: false,
                // S3: thread the `--wggo*` surface through so the WGGO
                // mode-table dispatch reaches `emit_unified_optim_step_dispatch`
                // via `nsl run` (previously hardcoded to defaults, which
                // blocked end-to-end activation of the Part II FP16 optim wrap).
                // Mirrors `nsl build` (build/options.rs).
                wggo: nsl_codegen::WggoOptions {
                    mode: wggo.clone(),
                    report: wggo_report,
                    weights: wggo_weights.clone(),
                    importance: nsl_codegen::WggoImportance::from(wggo_importance),
                    prune_fraction: wggo_prune_fraction,
                },
                cfie: nsl_codegen::CfieOptions::default(),
                // Phase 4 Task 6: when a train block is detected with --monitor,
                // the health monitor takes over; disable the Phase 1/2 kernel
                // timing path so they don't stomp on each other.
                profile_kernels: if detected_train_block { false } else { monitor },
                target_gpu: "h100".to_string(),
                dtype: "bf16".to_string(),
                manifest_output_path: if monitor {
                    Some(file.with_extension("nsl-profile.json"))
                } else {
                    None
                },
                profile_source_text: if monitor {
                    std::fs::read_to_string(&file).ok()
                } else {
                    None
                },
                profile_source_file_name: if monitor {
                    Some(file.display().to_string())
                } else {
                    None
                },
                health_monitor: detected_train_block,
                health_flush_interval: None,
                inspect_enabled: inspect,
                csha: nsl_codegen::CshaOptions {
                    mode: csha.clone(),
                    report: csha_report,
                },
                // CSHA Sprint 2: default to empty here; `run_build_inner` /
                // `run_run` route through `run_build_single` (build.rs) which
                // overwrites this from semantic analysis via
                // pipeline::analysis_to_csha_configs.
                csha_configs: std::collections::HashMap::new(),
                // Cycle-10 §5.3 Task 6: default to empty here; overwritten
                // downstream in `run_build_single` via
                // pipeline::analysis_to_checkpoint_policies. Empty = byte-identity.
                checkpoint_policies: std::collections::HashMap::new(),
                // CPDT: thread the planner mode + cluster + plan-out slot into
                // codegen exactly as `nsl build` does — the compiler copies
                // cpdt.cluster into Compiler::cpdt_cluster and reads
                // cpdt.plan_out during train-block codegen via
                // invoke_cpdt_if_enabled.
                cpdt: nsl_codegen::CpdtOptions {
                    mode: cpdt_mode,
                    cluster: cpdt_cluster.clone(),
                    report_requested: cpdt_report,
                    moe_roofline_slack: 0.0,
                    plan_out: cpdt_plan_out.clone(),
                },
                // `nsl run` never sets WRGA check-mode overrides.
                wrga_check: nsl_codegen::WrgaCheckContext::default(),
                export_functions_out: None,
                calibration_data: None,
                calibration_mode: Some("required".to_string()),
                calibration_samples: 512,
                calibration_batch_size: 8,
                calibration_timeout_secs: 600,
                calibration_sidecar: None,
                calibration_retention: None,
                calibration_batch_seq: None,
                // M62 Task 6: weight_index_map is populated from analysis in
                // run_build_single (where analysis is in scope).
                weight_index_map: std::collections::HashMap::new(),
                // PR #127 (AWQ v2) added this field; CLI run site doesn't
                // perform calibration so the bundle is unset.
                calibration_compile_bundle: None,
                // PR #132 (WGGO Phase 2) added this field; the run site
                // doesn't drive calibration, so this stays None.
                calibration_grad_retention: None,
            };
            // M41: Disaggregated inference — spawn router + prefill + decode workers.
            // Each runs the same compiled binary with NSL_ROLE and NSL_LOCAL_RANK env vars.
            if prefill_workers > 1 || decode_workers > 1 {
                // Build to a temp directory first, then spawn worker processes
                let temp_dir =
                    std::env::temp_dir().join(format!("nsl_disagg_{}", std::process::id()));
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
                let binary_path = temp_dir.join(&exe_name);

                // Build the binary
                crate::commands::build::run_build_inner(&file, Some(binary_path.clone()), false, false, true, &compile_opts, None);

                let mut children: Vec<(&str, std::process::Child)> = Vec::new();
                let total_workers = 1 + prefill_workers + decode_workers;

                // Spawn router (rank 0)
                let router_child = match std::process::Command::new(&binary_path)
                    .env("NSL_ROLE", "router")
                    .env("NSL_LOCAL_RANK", "0")
                    .env("NSL_WORLD_SIZE", format!("{}", total_workers))
                    .stderr(std::process::Stdio::inherit())
                    .spawn()
                {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("[nsl] failed to spawn router process: {e}");
                        std::process::exit(1);
                    }
                };
                children.push(("router:0", router_child));

                // Spawn prefill workers
                for i in 0..prefill_workers {
                    let child = match std::process::Command::new(&binary_path)
                        .env("NSL_ROLE", "prefill")
                        .env("NSL_LOCAL_RANK", format!("{}", i))
                        .env("NSL_WORLD_SIZE", format!("{}", total_workers))
                        .stderr(std::process::Stdio::inherit())
                        .spawn()
                    {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("[nsl] failed to spawn prefill worker {i}: {e}");
                            std::process::exit(1);
                        }
                    };
                    children.push(("prefill", child));
                }

                // Spawn decode workers
                for i in 0..decode_workers {
                    let child = match std::process::Command::new(&binary_path)
                        .env("NSL_ROLE", "decode")
                        .env("NSL_LOCAL_RANK", format!("{}", i))
                        .env("NSL_WORLD_SIZE", format!("{}", total_workers))
                        .stderr(std::process::Stdio::inherit())
                        .spawn()
                    {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("[nsl] failed to spawn decode worker {i}: {e}");
                            std::process::exit(1);
                        }
                    };
                    children.push(("decode", child));
                }

                // Wait for all processes
                let mut exit_code = 0;
                for (name, mut child) in children {
                    let status = match child.wait() {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[nsl] failed to wait on {name}: {e}");
                            exit_code = 1;
                            continue;
                        }
                    };
                    if !status.success() {
                        eprintln!("[nsl] {} exited with {}", name, status);
                        exit_code = 1;
                    }
                }

                // Cleanup
                let _ = std::fs::remove_file(&binary_path);
                let _ = std::fs::remove_dir(&temp_dir);

                std::process::exit(exit_code);
            } else if devices > 1 {
                // Build-then-spawn SPMD: spawn N children, each runs the same program
                let exe = match std::env::current_exe() {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("[nsl] could not find current executable: {e}");
                        std::process::exit(1);
                    }
                };

                // Create shared memory file for SimulatedBackend
                let shm_size = 64 + devices as usize * 64 * 1024 * 1024; // header + 64MB per rank
                let shm_path = std::env::temp_dir().join(format!(
                    "nsl_tp_{}_{}.shm",
                    devices,
                    std::process::id()
                ));
                {
                    let f = match std::fs::File::create(&shm_path) {
                        Ok(f) => f,
                        Err(e) => {
                            eprintln!("[nsl] failed to create shm file: {e}");
                            std::process::exit(1);
                        }
                    };
                    if let Err(e) = f.set_len(shm_size as u64) {
                        eprintln!("[nsl] failed to set shm size: {e}");
                        std::process::exit(1);
                    }
                    // Zero the header by mapping and dropping
                    let mmap = match unsafe { memmap2::MmapMut::map_mut(&f) } {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("[nsl] failed to mmap shm file: {e}");
                            std::process::exit(1);
                        }
                    };
                    drop(mmap);
                }

                let mut children = Vec::new();
                for rank in 0..devices {
                    let mut cmd = std::process::Command::new(&exe);
                    cmd.arg("run")
                        .arg(&file)
                        .env("NSL_LOCAL_RANK", rank.to_string())
                        .env("NSL_WORLD_SIZE", devices.to_string())
                        .env("NSL_SIMULATED_TP", "1")
                        .env("NSL_TP_SHM_PATH", shm_path.to_str().unwrap_or(""));

                    // Forward profiling flags
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
                        cmd.env("NSL_GPU_MEM_REPORT", "1");
                    }

                    // Pass through program args
                    if !args.is_empty() {
                        cmd.arg("--").args(&args);
                    }

                    // Only rank 0 gets stdout
                    if rank > 0 {
                        cmd.stdout(std::process::Stdio::null());
                    }
                    cmd.stderr(std::process::Stdio::inherit());

                    let child = match cmd.spawn() {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("[nsl] failed to spawn rank {rank}: {e}");
                            std::process::exit(1);
                        }
                    };
                    children.push((rank, child));
                }

                // Wait for all children
                let mut failed = false;
                for (rank, mut child) in children {
                    let status = match child.wait() {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[nsl] failed to wait on rank {rank}: {e}");
                            failed = true;
                            continue;
                        }
                    };
                    if !status.success() {
                        eprintln!("[nsl] rank {} exited with: {}", rank, status);
                        failed = true;
                    }
                }

                // Cleanup shared memory
                let _ = std::fs::remove_file(&shm_path);

                if failed {
                    std::process::exit(1);
                }
            } else {
                crate::commands::build::run_run(&file, &args, profile_memory, profile_kernels, profile, cuda_sync, gpu_mem_report, &compile_opts);
                // Phase 4 Task 6: after the child process exits, load and
                // render the health snapshot written by the runtime flush
                // hook.  Only runs when `--monitor` + train-block detection
                // enabled the health monitor upstream.
                if monitor && detected_train_block {
                    let health_path = file.with_extension("nsl-health.json");
                    match std::fs::read_to_string(&health_path) {
                        Ok(s) => match serde_json::from_str::<
                            nsl_runtime::health::collector::HealthSnapshot,
                        >(&s)
                        {
                            Ok(snap) => {
                                let mut renderer =
                                    nsl_cli::health_monitor::HealthRenderer::new();
                                renderer.render(&snap);
                            }
                            Err(e) => eprintln!(
                                "warning: health snapshot at {} failed to parse: {}",
                                health_path.display(),
                                e
                            ),
                        },
                        Err(_) => eprintln!(
                            "warning: no health snapshot at {} — train step may not have reached first flush",
                            health_path.display()
                        ),
                    }
                }
                // Phase 5 Task 8: summarize @inspect dumps written by the
                // runtime hook to `.nsl-inspect/`.
                if inspect {
                    let dir = std::path::PathBuf::from(".nsl-inspect");
                    match std::fs::read_dir(&dir) {
                        Ok(entries) => {
                            let (mut stats_count, mut full_count) = (0usize, 0usize);
                            for entry in entries.flatten() {
                                let name = entry.file_name();
                                let name_str = name.to_string_lossy();
                                if name_str.ends_with(".stats.bin") {
                                    stats_count += 1;
                                } else if name_str.ends_with(".tensor.bin") {
                                    full_count += 1;
                                }
                            }
                            eprintln!(
                                "[inspect] Wrote {} stats records, {} full dumps to {}/",
                                stats_count,
                                full_count,
                                dir.display()
                            );
                        }
                        Err(_) => {
                            eprintln!(
                                "[inspect] No inspect output directory at {} — @inspect sites may not have fired",
                                dir.display()
                            );
                        }
                    }
                }
            }
}
