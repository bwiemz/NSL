//! `nsl build` argument handling: validate flags, build the `CompileOptions`,
//! and dispatch to the standalone / shared-lib / ZK / normal build path.
//!
//! Extracted verbatim from the former monolithic `build.rs`; behavior is
//! unchanged.

use std::path::PathBuf;
use std::process;

use nsl_lexer::Interner;

pub(crate) fn dispatch(args: crate::args::BuildArgs) {
    let crate::args::BuildArgs {
            file,
            output,
            emit_obj,
            dump_ir,
            standalone,
            weights,
            embed_weights,
            embed_threshold,
            no_autotune,
            autotune_fresh,
            autotune_clean,
            fusion_report,
            vram_budget,
            memory_report,
            linear_types,
            target,
            disable_fusion,
            tape_ad: _tape_ad,
            source_ad: _source_ad,
            pretrain_optimized,
            debug_training,
            grad_integrity,
            training_reference,
            nan_analysis,
            distribute: _distribute,
            zero_stage,
            deterministic: _deterministic,
            dead_weight_threshold,
            sparse_threshold,
            no_constant_fold,
            no_dead_weight,
            no_sparse_codegen,
            shared_lib,
            unikernel,
            listen,
            memory,
            wcet,
            wcet_cert,
            cpu,
            do178c_report,
            wcet_target,
            fpga_device,
            zk_circuit,
            zk_backend,
            zk_field,
            zk_solidity,
            zk_weights,
            wrga_report,
            wrga_fold_allocations,
            wggo,
            wggo_report,
            wggo_moment_precision,
            wggo_weights,
            wggo_importance,
            wggo_prune_fraction,
            wggo_memory_budget,
            optim_state_offload,
            checkpoint_blocks,
            checkpoint_selective,
            checkpoint_budget_mib,
            checkpoint_stride,
            fuse_rmsnorm_backward,
            checkpoint_compress,
            layerwise_accum,
            weight_stream,
            stream_arena,
            devices,
            csha,
            csha_report,
            cpdt,
            cpdt_num_gpus,
            cpdt_intra_bw,
            cpdt_inter_bw,
            cpdt_report,
            cfie,
            cfie_report,
            calibration_data,
            calibrate,
            calibration_samples,
            calibration_batch_size,
            calibration_timeout,
            cep_prune,
            cep_joint,
            cep_target,
            cep_sparsity,
            cep_out,
            cep_emit_weights,
            cep_emit_source,
    } = args;

    // Meta-flag expansion (roadmap 3.3): must run BEFORE mode-string
    // validation below so bundle-filled values take the same validation
    // path as hand-written flags. Shared helper with `nsl run` so the two
    // dispatchers cannot drift.
    let mut wggo = wggo;
    let mut csha = csha;
    let mut _source_ad = _source_ad;
    crate::meta_flags::expand_pretrain_optimized(
        pretrain_optimized,
        &mut wggo,
        &mut csha,
        &mut _source_ad,
    );

            // M62a: shared_lib flag is threaded through compile_opts and handled
            // in the build path below.

            // --wggo-memory-budget: reject 0 loudly (clap already rejects an
            // unparseable value at the u64 parse). The flag IMPLIES
            // --wggo-moment-precision — a budget that forces sub-32 moments is
            // meaningless unless the bits are actually lowered to storage.
            // Convert MiB -> bytes here so codegen only ever sees bytes. This
            // runs before the CompileOptions build below because the wggo
            // sub-struct is constructed there. (Mirrors `nsl run`.)
            let wggo_memory_budget_bytes = match wggo_memory_budget {
                Some(0) => {
                    eprintln!("error: --wggo-memory-budget must be > 0 MiB");
                    process::exit(1);
                }
                Some(mib) => Some(mib.saturating_mul(1024 * 1024)),
                None => None,
            };
            let wggo_moment_precision =
                wggo_moment_precision || wggo_memory_budget_bytes.is_some();

            if cep_prune && cep_joint {
                eprintln!(
                    "error: --cep-prune and --cep-joint are mutually exclusive (use one)"
                );
                std::process::exit(1);
            }

            if cep_prune {
                let ov = nsl_codegen::cep::CliOverrides {
                    target: cep_target,
                    sparsity: cep_sparsity,
                    cep_out,
                    cep_emit_weights,
                    cep_emit_source,
                };
                std::process::exit(crate::commands::cep::run_cep_prune(&file, weights.as_deref(), &ov));
            }

            if cep_joint {
                let ov = nsl_codegen::cep::CliOverrides {
                    target: cep_target,
                    sparsity: cep_sparsity,
                    cep_out,
                    cep_emit_weights,
                    cep_emit_source,
                };
                std::process::exit(crate::commands::cep::run_cep_joint(&file, weights.as_deref(), &ov));
            }

            if autotune_clean {
                let cache_dir = std::path::Path::new(".nsl-cache/autotune");
                if cache_dir.exists() {
                    std::fs::remove_dir_all(cache_dir).ok();
                    eprintln!("[nsl] autotune cache cleaned");
                } else {
                    eprintln!("[nsl] no autotune cache to clean");
                }
                return;
            }

            // M54: Parse unikernel configuration if --unikernel is set.
            let unikernel_config = if unikernel {
                let listen_addr = match nsl_codegen::unikernel::parse_listen_addr(&listen) {
                    Ok(addr) => addr,
                    Err(e) => {
                        eprintln!("error: invalid --listen value: {e}");
                        process::exit(1);
                    }
                };
                let memory_bytes = match memory.as_deref() {
                    Some(s) => match nsl_codegen::unikernel::parse_memory_size(s) {
                        Ok(n) => n,
                        Err(e) => {
                            eprintln!("error: invalid --memory value: {e}");
                            process::exit(1);
                        }
                    },
                    None => 0, // auto-detect at boot
                };
                let cfg = nsl_codegen::unikernel::UnikernelConfig {
                    listen_addr,
                    memory_bytes,
                    ..Default::default()
                };
                cfg.print_summary();
                Some(cfg)
            } else {
                None
            };

            // Calibration-flag validation per spec §8.
            if calibration_data.is_none() && calibrate.as_str() != "required" {
                eprintln!(
                    "error: --calibrate={} requires --calibration-data <PATH>",
                    calibrate
                );
                process::exit(1);
            }
            match calibrate.as_str() {
                "required" | "best-effort" => {}
                other => {
                    eprintln!(
                        "error: --calibrate value '{}' is not one of required|best-effort",
                        other
                    );
                    process::exit(1);
                }
            }
            if let Some(ref p) = calibration_data {
                if !p.exists() {
                    eprintln!("error: --calibration-data path does not exist: {}", p.display());
                    process::exit(1);
                }
                let ext = p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
                match ext.as_deref() {
                    Some("bin") | Some("safetensors") => {}
                    other => {
                        eprintln!(
                            "error: --calibration-data extension {:?} is not one of .bin|.safetensors",
                            other
                        );
                        process::exit(1);
                    }
                }
            }
            if calibration_samples == 0 {
                eprintln!("error: --calibration-samples must be > 0");
                process::exit(1);
            }
            if calibration_batch_size == 0 {
                eprintln!("error: --calibration-batch-size must be > 0");
                process::exit(1);
            }
            if calibration_timeout == 0 {
                eprintln!("error: --calibration-timeout must be > 0");
                process::exit(1);
            }

            // CPDT: --cpdt-report implies --cpdt (full mode unless explicit).
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
            // Resolve the effective weight file via the four-case decision table
            // from the Phase 1 spec §2.1. Design:
            // docs/superpowers/specs/2026-04-21-cpdt-ast-autodetect-design.md.
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
                        // The `!standalone` guard is load-bearing: a separate
                        // "--standalone requires --weights" error fires later
                        // (~line 1298). Without this guard, the four-case
                        // message would fire first and replace the standalone-
                        // specific error. If a future refactor moves the
                        // standalone check earlier, this guard can be dropped.
                        if cpdt_enabled && weight_aware && !standalone {
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

            let mut compile_opts = nsl_codegen::CompileOptions {
                no_autotune,
                autotune_fresh,
                world_size: devices.max(1) as usize, // --devices drives WGGO ZeRO + TP world_size
                fusion_report,
                vram_budget: vram_budget.as_deref()
                    .and_then(nsl_codegen::memory_planner::parse_vram_budget),
                memory_report,
                target,
                disable_fusion,
                tape_ad: _tape_ad,
                source_ad: _source_ad,
                trace_ops: false,
                nan_analysis,
                deterministic: _deterministic,
                // M52: When --standalone, weights are handled by standalone pipeline;
                // otherwise pass through the four-case-resolved weight file from
                // above (AST auto-detect + --weights flag decision table).
                weight_file: if standalone { None } else { resolved_weight_file.clone() },
                weight_config: nsl_codegen::weight_aware::WeightAwareConfig {
                    dead_weight_threshold,
                    sparse_threshold,
                    constant_fold: !no_constant_fold,
                    dead_weight_elim: !no_dead_weight,
                    sparse_codegen: !no_sparse_codegen,
                },
                weight_analysis: false,
                unikernel_config,
                wcet: nsl_codegen::WcetOptions {
                    enabled: wcet,
                    gpu: None, // reuse --gpu from Check variant; Build uses target for backend
                    cpu,
                    report_path: wcet_cert,
                    safety_margin: 1.05,
                    do178c_report,
                    target: wcet_target,
                    fpga_device,
                },
                zk: nsl_codegen::ZkOptions {
                    circuit: zk_circuit,
                    backend: zk_backend,
                    field: zk_field,
                    solidity: zk_solidity,
                    weights_path: zk_weights.clone(),
                },
                linear_types_enabled: linear_types,
                ownership_info: std::collections::HashMap::new(), // populated by loader
                zero_stage: zero_stage.map(|s| s as u8),
                optim_state_offload,
                checkpoint_blocks,
                checkpoint_selective,
                checkpoint_budget_mib,
                checkpoint_stride: crate::meta_flags::parse_checkpoint_stride(&checkpoint_stride),
                fuse_rmsnorm_backward,
                checkpoint_compress,
                layerwise_accum,
                weight_stream,
                stream_arena,
                debug_training,
                grad_integrity,
                training_reference,
                shared_lib,
                emit_export_table: shared_lib,
                wrga_inputs: None,
                fused_ce_configs: Vec::new(),
                fused_kl_ce_configs: Vec::new(),
                pca_user_strategies: Vec::new(),
                wrga_fold_allocations,
                wggo: nsl_codegen::WggoOptions {
                    mode: wggo.clone(),
                    report: wggo_report,
                    moment_precision: wggo_moment_precision,
                    weights: wggo_weights.clone(),
                    importance: nsl_codegen::WggoImportance::from(wggo_importance),
                    prune_fraction: wggo_prune_fraction,
                    memory_budget_bytes: wggo_memory_budget_bytes,
                },
                cfie: nsl_codegen::CfieOptions {
                    mode_override: cfie.clone(),
                    report_path: cfie_report.clone(),
                },
                profile_kernels: false,
                target_gpu: "h100".to_string(),
                dtype: "bf16".to_string(),
                manifest_output_path: None,
                profile_source_text: None,
                profile_source_file_name: None,
                health_monitor: false,
                health_flush_interval: None,
                inspect_enabled: false,
                csha: nsl_codegen::CshaOptions {
                    mode: csha.clone(),
                    report: csha_report,
                },
                // CSHA Sprint 2: default to empty here; the six build-path
                // entry points (run_build_shared_single, run_build_shared_multi,
                // run_build_zk, run_build_standalone, run_build_single,
                // run_build_multi) overwrite this from semantic analysis via
                // pipeline::{analysis,module_data}_to_csha_configs.
                csha_configs: std::collections::HashMap::new(),
                // Cycle-10 §5.3 Task 6: default to empty here; overwritten
                // per build path by pipeline::{analysis,module_data}_to_checkpoint_policies
                // once the semantic checker has run. Empty = byte-identity.
                checkpoint_policies: std::collections::HashMap::new(),
                cpdt: nsl_codegen::CpdtOptions {
                    mode: cpdt_mode,
                    cluster: cpdt_cluster.clone(),
                    report_requested: cpdt_report,
                    moe_roofline_slack: 0.0,
                    plan_out: cpdt_plan_out.clone(),
                },
                // Normal `nsl build` never sets WRGA check-mode overrides;
                // `nsl check --wrga-analyze | --wrga-compare` builds its own
                // CompileOptions with a populated `wrga_check` (wrga_check.rs).
                wrga_check: nsl_codegen::WrgaCheckContext::default(),
                export_functions_out: None,
                calibration_data: calibration_data.clone(),
                calibration_mode: Some(calibrate.clone()),
                calibration_samples,
                calibration_batch_size,
                calibration_timeout_secs: calibration_timeout,
                calibration_sidecar: None,
                calibration_retention: None,
                // Task 6: peek_batch_seq is called inside the compiler when
                // calibration_data is set; the CLI passes None here and the
                // compiler resolves the real (batch, seq) from the data header.
                calibration_batch_seq: None,
                // M62 Task 6: weight_index_map is populated from analysis in
                // run_build_single/run_build_multi (where analysis is in scope).
                weight_index_map: std::collections::HashMap::new(),
                // PR #127 (AWQ v2) added this field; CLI build site populates
                // it via the calibration plumbing further down, not here.
                calibration_compile_bundle: None,
                // PR #132 (WGGO Phase 2) added this field; codegen populates
                // it from AST pre-scan inside `run_pre_scan_phase`, so the
                // CLI initializes to None and lets entry_points.rs do it.
                calibration_grad_retention: None,
            };
            // P1.7: force the field-controlled optimizations off for the
            // reference training path (decorator/pattern-driven ones are gated
            // in codegen on compile_opts.training_reference).
            crate::meta_flags::apply_training_reference(&mut compile_opts);

            // Validate WGGO mode string early so users get a clear error
            // instead of a silent no-op.
            if let Some(ref m) = wggo {
                if nsl_codegen::wggo::WggoMode::parse(m).is_none() {
                    eprintln!(
                        "error: --wggo value '{}' is not one of full|greedy|off|auto",
                        m
                    );
                    process::exit(1);
                }
            }
            // wggo_importance is now a typed CliWggoImportance enum; clap
            // rejects unknown values before we get here.  The Grad variant
            // requires a calibration sidecar — build_scorer enforces that at
            // compile time and emits the --calibration-data error message.
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
            // Validate CSHA mode string early.
            if let Some(ref m) = csha {
                if nsl_codegen::csha::CshaMode::parse(m).is_none() {
                    eprintln!(
                        "error: --csha value '{}' is not one of auto|boundary|pipeline|block|off",
                        m
                    );
                    process::exit(1);
                }
            }

            if standalone {
                if weights.is_none() {
                    eprintln!("error: --standalone requires -w/--weights <path>");
                    process::exit(1);
                }
                let embed_mode = match embed_weights.to_lowercase().as_str() {
                    "auto" => crate::standalone::EmbedMode::Auto,
                    "always" => crate::standalone::EmbedMode::Always,
                    "never" => crate::standalone::EmbedMode::Never,
                    other => {
                        eprintln!(
                            "error: unknown --embed-weights value '{}'. \
                             Expected: auto, always, never",
                            other
                        );
                        process::exit(1);
                    }
                };
                crate::commands::build::run_build_standalone(
                    &file,
                    output.as_deref(),
                    weights.as_deref().unwrap(),
                    embed_mode,
                    embed_threshold,
                    &compile_opts,
                    wrga_report.as_deref(),
                );
            } else if shared_lib {
                crate::commands::build::run_build_shared(&file, output, dump_ir, &compile_opts, wrga_report.as_deref());
            } else if zk_circuit {
                crate::commands::build::run_build_zk(
                    &file,
                    output,
                    emit_obj,
                    dump_ir,
                    zk_weights.as_deref(),
                    &compile_opts,
                    wrga_report.as_deref(),
                );
            } else {
                crate::commands::build::run_build(&file, output, emit_obj, dump_ir, &compile_opts, wrga_report.as_deref());
            }

            // CPDT: post-compile rendering. Stderr diagnostics always fire
            // when CPDT ran; stdout plan only with --cpdt-report.
            if let Some(slot) = cpdt_plan_out.as_ref() {
                if let Some(plan) = slot.lock().ok().and_then(|g| g.clone()) {
                    for diag in &plan.override_diagnostics {
                        eprintln!(
                            "[cpdt] scope:global wggo-override-rejected requested={} applied={} reason={:?}",
                            diag.requested, diag.applied, diag.reason
                        );
                    }
                    if cpdt_report {
                        print!("{}", plan.render_report());
                        println!();
                        println!("=== Defaults Assumed ===");
                        println!("precision_cfg: BF16-mixed (override: --cpdt-precision, future)");
                        let jc = nsl_codegen::cpdt_joint::JointConfig::default();
                        println!("joint_cfg:     {:?} (override: --cpdt-budget, future)", jc);
                        println!("expert_cfg:    none (no MoE block detected)");
                        match &resolved_weight_file {
                            Some(p) => println!("weights:       {}", p.display()),
                            None => println!(
                                "weights:       none (no --weights flag and no AST load_safetensors)"
                            ),
                        }
                    }
                }
            }
}
