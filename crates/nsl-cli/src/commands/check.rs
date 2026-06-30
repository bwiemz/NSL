//! `nsl check` — type-check / shape-check without running, with optional
//! token/AST/type dumps.
//!
//! Extracted verbatim from `main.rs`; behavior is unchanged.

use std::path::PathBuf;
use std::process;

use nsl_errors::{Level, SourceMap};
use nsl_lexer::Interner;
use crate::args::TrainingReportFormat;

pub(crate) fn run_check(file: &PathBuf, dump_tokens: bool, dump_ast: bool, dump_types: bool, linear_types: bool) {
    let source = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read file '{}': {e}", file.display());
            process::exit(1);
        }
    };

    let mut source_map = SourceMap::new();
    let file_id = source_map.add_file(file.display().to_string(), source.clone());

    let mut interner = Interner::new();

    // Lex
    let (tokens, lex_errors) = nsl_lexer::tokenize(&source, file_id, &mut interner);

    if dump_tokens {
        for token in &tokens {
            println!("{:?}", token);
        }
    }

    for diag in &lex_errors {
        source_map.emit_diagnostic(diag);
    }

    // Parse
    let parse_result = nsl_parser::parse(&tokens, &mut interner);

    for diag in &parse_result.diagnostics {
        source_map.emit_diagnostic(diag);
    }

    if dump_ast {
        match serde_json::to_string_pretty(&parse_result.module) {
            Ok(json) => println!("{json}"),
            Err(e) => eprintln!("error serializing AST: {e}"),
        }
    }

    // Semantic analysis (with ownership checking when --linear-types is active)
    let analysis = nsl_semantic::analyze_with_imports(
        &parse_result.module, &mut interner, &std::collections::HashMap::new(), linear_types,
    );

    for diag in &analysis.diagnostics {
        source_map.emit_diagnostic(diag);
    }

    if dump_types {
        println!("=== Type Map ===");
        let mut entries: Vec<_> = analysis.type_map.iter().collect();
        entries.sort_by_key(|(id, _)| id.0);
        for (id, ty) in entries {
            println!("  node {}: {}", id.0, nsl_semantic::types::display_type(ty));
        }
    }

    let total_errors = lex_errors
        .iter()
        .chain(parse_result.diagnostics.iter())
        .chain(analysis.diagnostics.iter())
        .filter(|d| d.level == Level::Error)
        .count();

    if total_errors > 0 {
        eprintln!("{total_errors} error(s) found");
        process::exit(1);
    } else {
        println!(
            "OK: {} checked successfully ({} statements)",
            file.display(),
            parse_result.module.stmts.len()
        );
    }
}


pub(crate) fn dispatch(args: crate::args::CheckArgs) {
    let crate::args::CheckArgs {
            file,
            dump_tokens,
            dump_ast,
            dump_types,
            shapes,
            perf: _perf,
            gpu: _gpu,
            trace: _trace,
            linear_types,
            nan_analysis,
            deterministic,
            weight_analysis,
            weights,
            dead_weight_threshold,
            sparse_threshold,
            wcet: _wcet,
            wcet_cert: _wcet_cert,
            cpu: _cpu,
            do178c_report: _do178c_report,
            wcet_target: _wcet_target,
            fpga_device: _fpga_device,
            training_report,
            cep_search,
            cep_profile,
            cep_target,
            cep_out,
            wrga_analyze,
            wrga_target,
            wrga_compare,
            csha,
            csha_report,
    } = args;

            if cep_search && cep_profile {
                eprintln!("error: --cep-search and --cep-profile are mutually exclusive");
                std::process::exit(1);
            }
            if wrga_analyze.is_some() && wrga_compare.is_some() {
                eprintln!("error: --wrga-analyze and --wrga-compare are mutually exclusive");
                std::process::exit(1);
            }
            // WRGA paper §8.3: `--wrga-analyze` short-circuits the check path
            // to emit just the WRGA compilation report (no .o, no shape trace,
            // no NaN analysis side effects). Mutually exclusive with CEP modes
            // since CEP early-exits above.
            if let Some(ref report_path) = wrga_analyze {
                std::process::exit(crate::commands::build::run_check_wrga_analyze(
                    &file,
                    report_path,
                    wrga_target.as_deref(),
                ));
            }
            // WRGA paper §8.3: `--wrga-compare` — same dispatch shape as
            // `--wrga-analyze`, but renders the PEFT comparison report.
            if let Some(ref report_path) = wrga_compare {
                std::process::exit(crate::commands::build::run_check_wrga_compare(
                    &file,
                    report_path,
                    wrga_target.as_deref(),
                ));
            }

            // CSHA (Sprint 3, paper §6.3): planner-only diagnostic on `nsl check`.
            // Mirrors `nsl run --csha[-report]` / `nsl build --csha[-report]` but
            // emits no kernels — runs the compile pipeline with the planner
            // wired in via `CompileOptions::csha_mode`/`csha_report` and
            // discards the resulting object bytes. The CSHA report fires
            // from inside `compile_train_block`, so files without a train
            // block produce a `note:` instead of a report.
            if csha_report || csha.is_some() {
                // Validate the mode string up front (matches Run/Build).
                if let Some(ref m) = csha {
                    if nsl_codegen::csha::CshaMode::parse(m).is_none() {
                        eprintln!(
                            "error: --csha value '{}' is not one of auto|boundary|pipeline|block|off",
                            m
                        );
                        process::exit(1);
                    }
                }
                let source = match std::fs::read_to_string(&file) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("error: could not read file '{}': {e}", file.display());
                        process::exit(1);
                    }
                };
                // `--csha-report` without an explicit `--csha` defaults to
                // `auto` so the planner actually fires — mirrors the
                // CPDT/WRGA convention where `--<feature>-report` implies
                // the planner is on.
                let mode_str = csha.clone().unwrap_or_else(|| "auto".to_string());
                // Best-effort detection of whether the source has a train
                // block so we can emit a helpful `note:` when the planner
                // won't fire. Same quick scan `needs_multi_file` uses.
                let has_train_block = source.lines().any(|line| {
                    let t = line.trim();
                    if t.starts_with('#') || t.starts_with("//") { return false; }
                    t.starts_with("train(") || t.starts_with("train (")
                });
                if !has_train_block {
                    eprintln!(
                        "note: --csha-report: no `train(...)` block in {} — CSHA planner has nothing to do.",
                        file.display(),
                    );
                } else {
                    let mut opts = nsl_codegen::CompileOptions::default();
                    opts.csha_mode = Some(mode_str);
                    opts.csha_report = csha_report;
                    // Force source-AD so the train block lowers to a
                    // Wengert list (the planner runs against
                    // `extractor.wengert_list()`). Without this the
                    // tape-AD path bypasses CSHA entirely.
                    opts.source_ad = true;
                    // Suppress unrelated noise: no calibration data.
                    opts.calibration_data = None;
                    // Drive the same compile entry the `nsl build` path
                    // uses (`compile_returning_plan`) so top-level
                    // `train(...)` blocks reach `compile_train_block`
                    // where the CSHA hook lives. The alternative
                    // `compile_module_with_imports_*` family stops
                    // after `compile_user_functions` and never runs
                    // `compile_main`, so top-level train blocks would
                    // silently skip the planner.
                    let mut csha_interner = Interner::new();
                    let mut csha_source_map = SourceMap::new();
                    let csha_file_id = csha_source_map
                        .add_file(file.display().to_string(), source.clone());
                    let (csha_tokens, csha_lex_diags) =
                        nsl_lexer::tokenize(&source, csha_file_id, &mut csha_interner);
                    if csha_lex_diags
                        .iter()
                        .any(|d| d.level == Level::Error)
                    {
                        eprintln!(
                            "note: --csha-report: lex errors prevented planner from running on {}",
                            file.display(),
                        );
                    } else {
                        let csha_parse = nsl_parser::parse(&csha_tokens, &mut csha_interner);
                        if csha_parse
                            .diagnostics
                            .iter()
                            .any(|d| d.level == Level::Error)
                        {
                            eprintln!(
                                "note: --csha-report: parse errors prevented planner from running on {}",
                                file.display(),
                            );
                        } else {
                            let csha_analysis = nsl_semantic::analyze_with_imports(
                                &csha_parse.module,
                                &mut csha_interner,
                                &std::collections::HashMap::new(),
                                false, // linear_types not needed for the planner-only path
                            );
                            // Forward @csha decorator configs (Sprint 2)
                            // exactly the way `run_build_single` does.
                            opts.csha_configs = crate::pipeline::analysis_to_csha_configs(&csha_analysis);
                            // The planner runs as a side-effect inside
                            // `compile_train_block` (which sits inside
                            // `compile_user_functions` → `compile_main`)
                            // and emits its report to stderr BEFORE the
                            // rest of codegen. Downstream codegen often
                            // fails on `nsl check`'s minimal compile
                            // (e.g. unresolved optimizer stdlib symbols
                            // like `nsl_optim_sgd__sgd_step`) — we
                            // intentionally swallow those errors with a
                            // brief note so the CSHA report stays the
                            // headline output.
                            if let Err(e) = nsl_codegen::compile_returning_plan(
                                &csha_parse.module,
                                &csha_interner,
                                &csha_analysis.type_map,
                                false,
                                &opts,
                            ) {
                                eprintln!(
                                    "note: --csha-report: compile pipeline stopped after the planner ran (full codegen needs `nsl build`): {}",
                                    e.message,
                                );
                            }
                        }
                    }
                }
            }

            if cep_search {
                let ov = nsl_codegen::cep::CliOverrides {
                    target: cep_target,
                    sparsity: None,
                    cep_out,
                    cep_emit_weights: None,
                    cep_emit_source: None,
                };
                std::process::exit(crate::commands::cep::run_cep_search(&file, &ov));
            }
            if cep_profile {
                let ov = nsl_codegen::cep::CliOverrides {
                    target: cep_target,
                    sparsity: None,
                    cep_out: None,
                    cep_emit_weights: None,
                    cep_emit_source: None,
                };
                std::process::exit(crate::commands::cep::run_cep_profile(&file, weights.as_deref(), &ov));
            }
            if shapes {
                let src = match std::fs::read_to_string(&file) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("error: could not read file '{}': {e}", file.display());
                        process::exit(1);
                    }
                };
                match nsl_cli::shape_debug::ShapeDebugInput::from_source(
                    &src,
                    &file.display().to_string(),
                ) {
                    Ok(input) => {
                        println!("{}", nsl_cli::shape_debug::format_trace(&input));
                        return;
                    }
                    Err(e) => {
                        eprintln!("error: shape debug failed: {e}");
                        process::exit(1);
                    }
                }
            }
            crate::commands::check::run_check(&file, dump_tokens, dump_ast, dump_types, linear_types);
            // M37: --perf, --gpu, --trace flags parsed but dormant.

            if nan_analysis {
                // M45: Run compile-time NaN risk analysis via AST walker.
                // Re-lex/parse the file to get a Module + Interner for the walker.
                let source = std::fs::read_to_string(&file).unwrap_or_default();
                let mut nan_interner = Interner::new();
                let mut nan_source_map = SourceMap::new();
                let nan_file_id = nan_source_map.add_file(file.display().to_string(), source.clone());
                let (tokens, _) = nsl_lexer::tokenize(&source, nan_file_id, &mut nan_interner);
                let parse_result = nsl_parser::parse(&tokens, &mut nan_interner);

                let mut analyzer = nsl_semantic::nan_analysis::NanAnalyzer::new();
                analyzer.analyze_module(&parse_result.module, &nan_interner);

                if analyzer.diagnostics.is_empty() {
                    eprintln!("note: --nan-analysis: no NaN/Inf risks detected");
                } else {
                    eprintln!(
                        "note: --nan-analysis: {} warning(s) detected",
                        analyzer.diagnostics.len()
                    );
                    for diag in &analyzer.diagnostics {
                        nan_source_map.emit_diagnostic(diag);
                    }
                }
            }

            // M46: Determinism analysis — scan for non-deterministic ops
            if deterministic {
                let source = std::fs::read_to_string(&file).unwrap_or_default();
                let mut det_interner = Interner::new();
                let mut det_source_map = SourceMap::new();
                let det_file_id = det_source_map.add_file(file.display().to_string(), source.clone());
                let (tokens, _) = nsl_lexer::tokenize(&source, det_file_id, &mut det_interner);
                let parse_result = nsl_parser::parse(&tokens, &mut det_interner);

                let mut checker = nsl_semantic::determinism::DeterminismChecker::new(
                    nsl_semantic::determinism::DeterminismMode::Global,
                );
                checker.scan_module(&parse_result.module, &det_interner);

                let errors: Vec<_> = checker.diagnostics.iter()
                    .filter(|d| d.level == nsl_errors::Level::Error)
                    .collect();
                let warnings: Vec<_> = checker.diagnostics.iter()
                    .filter(|d| d.level == nsl_errors::Level::Warning)
                    .collect();

                if checker.diagnostics.is_empty() {
                    eprintln!("note: --deterministic: no non-deterministic ops detected");
                } else {
                    eprintln!(
                        "note: --deterministic: {} warning(s), {} error(s)",
                        warnings.len(),
                        errors.len()
                    );
                    for diag in &checker.diagnostics {
                        det_source_map.emit_diagnostic(diag);
                    }
                    if !errors.is_empty() {
                        process::exit(1);
                    }
                }
            }

            // M52: Weight analysis report
            if weight_analysis {
                if let Some(ref weights_path) = weights {
                    let config = nsl_codegen::weight_aware::WeightAwareConfig {
                        dead_weight_threshold,
                        sparse_threshold,
                        ..Default::default()
                    };
                    match nsl_codegen::weight_aware::WeightMap::load(weights_path.as_path()) {
                        Ok(wmap) => {
                            nsl_codegen::weight_aware::print_weight_analysis_report(&wmap, &config);
                        }
                        Err(e) => {
                            eprintln!("error: failed to load weights: {}", e);
                            process::exit(1);
                        }
                    }
                } else {
                    eprintln!("error: --weight-analysis requires --weights <path>");
                    process::exit(1);
                }
            }

            // FASE: Training-pipeline decision audit
            if let Some(format) = training_report {
                let source = std::fs::read_to_string(&file).unwrap_or_default();
                let mut tr_interner = Interner::new();
                let mut tr_source_map = SourceMap::new();
                let tr_file_id = tr_source_map.add_file(file.display().to_string(), source.clone());
                let (tr_tokens, _) = nsl_lexer::tokenize(&source, tr_file_id, &mut tr_interner);
                let tr_parse = nsl_parser::parse(&tr_tokens, &mut tr_interner);
                let report = nsl_codegen::training_report::build_report(
                    &tr_parse.module,
                    &tr_interner,
                    file.as_path(),
                );
                match format {
                    TrainingReportFormat::Text => {
                        println!("{}", report);
                    }
                    TrainingReportFormat::Json => {
                        let json = serde_json::to_string_pretty(&report)
                            .expect("serialize training report");
                        println!("{}", json);
                    }
                }
            }
}
