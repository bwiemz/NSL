//! CEP (compiler-extracted pruning) command frontend.
//!
//! `nsl build --cep-prune` / `nsl build --cep-joint` / `nsl check --cep-search`
//! / `nsl check --cep-profile`. These orchestrators tie together the AST
//! recognizer (`cep_extract`), the semantic decorator validators
//! (`nsl_semantic::cep`), the codegen bridge (`nsl_codegen::cep`), and the
//! delta writers. Extracted verbatim from `main.rs`; behavior is unchanged.

use nsl_errors::{Level, SourceMap};
use std::path::PathBuf;

/// Locate the `@cep_prune` / `@cep_search` decorator attached to the model
/// definition. Returns the first matching decorator in source order.
fn find_cep_decorator<'a>(
    module: &'a nsl_ast::Module,
    interner: &nsl_lexer::Interner,
    want: &str,
) -> Option<&'a nsl_ast::decl::Decorator> {
    use nsl_ast::stmt::StmtKind;
    for stmt in &module.stmts {
        if let StmtKind::Decorated { decorators, stmt: inner } = &stmt.kind {
            if matches!(inner.kind, StmtKind::ModelDef(_)) {
                for deco in decorators {
                    if deco.name.len() == 1
                        && interner.resolve(deco.name[0].0).unwrap_or("") == want
                    {
                        return Some(deco);
                    }
                }
            }
        }
    }
    None
}

/// Resolve the effective CEP target string. CLI `--cep-target` wins; otherwise
/// the decorator's `target = ...` ident; otherwise the default "H100-SXM".
fn resolve_cep_target(
    cli: Option<&str>,
    deco_target: Option<nsl_ast::Symbol>,
    resolve: &dyn Fn(nsl_ast::Symbol) -> String,
) -> String {
    if let Some(t) = cli {
        return t.to_string();
    }
    if let Some(sym) = deco_target {
        let s = resolve(sym);
        if !s.is_empty() {
            return s;
        }
    }
    "H100-SXM".to_string()
}

/// Default delta output path: `<model>.cep.json` next to the source file.
fn default_cep_out(file: &std::path::Path) -> PathBuf {
    file.with_extension("cep.json")
}

/// Emit CEP diagnostics with source context (same mechanism `run_check` uses:
/// `SourceMap::add_file` + `emit_diagnostic`). Returns `true` if any diagnostic
/// is at `Level::Error`.
fn emit_cep_diags(file: &std::path::Path, diags: &[nsl_errors::Diagnostic]) -> bool {
    let source = std::fs::read_to_string(file).unwrap_or_default();
    let mut source_map = SourceMap::new();
    source_map.add_file(file.display().to_string(), source);
    for diag in diags {
        source_map.emit_diagnostic(diag);
    }
    diags.iter().any(|d| d.level == Level::Error)
}

pub(crate) fn run_cep_prune(
    file: &PathBuf,
    weights: Option<&std::path::Path>,
    ov: &nsl_codegen::cep::CliOverrides,
) -> i32 {
    use nsl_codegen::cep_extract::{cross_check_dims, extract_model_spec};

    let Some(weights_path) = weights else {
        eprintln!(
            "error: --cep-prune requires --weights <file.safetensors> \
             (a prune from synthetic scores would be silently wrong)"
        );
        return 1;
    };

    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(file, false);
    // Analysis-error gate — same accessor as run_check: filter
    // `analysis.diagnostics` on `Level::Error`. `frontend_with_flags` already
    // exits on errors, but we keep a defensive gate that prints with source
    // context before bailing.
    let analysis_errors: Vec<_> = analysis
        .diagnostics
        .iter()
        .filter(|d| d.level == Level::Error)
        .cloned()
        .collect();
    if !analysis_errors.is_empty() {
        emit_cep_diags(file, &analysis_errors);
        return 1;
    }

    let module = &parse_result.module;
    let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();

    let Some(deco) = find_cep_decorator(module, &interner, "cep_prune") else {
        eprintln!("error: --cep-prune requires a @cep_prune(...) decorator on the model");
        return 1;
    };
    let mut diags = Vec::new();
    let Some(cfg) = nsl_semantic::cep::validate_cep_prune_decorator(deco, &resolve, &mut diags)
    else {
        emit_cep_diags(file, &diags);
        return 1;
    };
    // The validator may return `Some` while still pushing error diagnostics for
    // recoverable problems; treat any pushed error as fatal.
    if emit_cep_diags(file, &diags) {
        return 1;
    }

    let spec = match extract_model_spec(module, &resolve) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };

    let wm = match nsl_codegen::weight_aware::WeightMap::load(weights_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error: failed to load weights: {e}");
            return 1;
        }
    };
    if let Err(e) = cross_check_dims(&spec, &wm, &resolve) {
        eprintln!("{e}");
        return 1;
    }

    let target = resolve_cep_target(ov.target.as_deref(), cfg.target, &resolve);
    let input =
        match nsl_codegen::cep::build_prune_input(&cfg, spec.clone(), Some(&wm), &target, ov.sparsity) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        };
    let plan = nsl_codegen::cep::run_prune(input);
    println!("{}", plan.render_report());

    let out_path = ov.cep_out.clone().unwrap_or_else(|| default_cep_out(file));
    if let Err(e) = nsl_codegen::cep::write_prune_delta(&plan, &spec, &out_path) {
        eprintln!("error: failed to write delta: {e}");
        return 1;
    }
    println!("CEP delta written to {}", out_path.display());

    if let Some(weights_out) = ov.cep_emit_weights.as_ref() {
        let delta = nsl_codegen::cep::plan_to_prune_delta(&plan, &spec);
        match nsl_codegen::cep_slice::apply_prune_delta_to_weights(&wm, &spec, &delta) {
            Ok(sliced) => {
                let orig_params: usize = wm.entries().map(|(_, e)| e.num_elements).sum();
                let new_params: usize = sliced.values().map(|e| e.num_elements).sum();
                if let Err(e) = nsl_codegen::cep_slice::write_sliced_weights(&sliced, weights_out) {
                    eprintln!("error: failed to write sliced weights: {e}");
                    return 1;
                }
                println!(
                    "CEP sliced weights written to {} ({orig_params} -> {new_params} params)",
                    weights_out.display()
                );
            }
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        }
    }

    // SP2: rewrite the source with chosen pruned dims and write to `cep_emit_source`.
    if let Some(source_out) = ov.cep_emit_source.as_ref() {
        let delta = nsl_codegen::cep::plan_to_prune_delta(&plan, &spec);
        let original_source = match std::fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: failed to re-read source for SP2 emission: {e}");
                return 1;
            }
        };
        match nsl_codegen::cep_emit_source::apply_prune_delta_to_source(
            &original_source,
            module,
            &resolve,
            &spec,
            &delta,
        ) {
            Ok(out_src) => {
                if let Err(e) = std::fs::write(source_out, &out_src) {
                    eprintln!("error: failed to write rewritten source: {e}");
                    return 1;
                }
                println!("CEP rewritten source written to {}", source_out.display());
            }
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        }
    }
    0
}

/// CEP Mode 3 (paper §2.2) — joint prune-search. Same setup as `run_cep_prune` (reuses
/// the @cep_prune decorator for v1 config; weights required for non-synthetic
/// importance), but calls `nsl_codegen::cep::run_joint` to extend the action space with
/// layer drops on top of head + FFN pruning.
pub(crate) fn run_cep_joint(
    file: &PathBuf,
    weights: Option<&std::path::Path>,
    ov: &nsl_codegen::cep::CliOverrides,
) -> i32 {
    use nsl_codegen::cep_extract::{cross_check_dims, extract_model_spec};

    let Some(weights_path) = weights else {
        eprintln!(
            "error: --cep-joint requires --weights <file.safetensors> \
             (joint search without weight-derived importance would be silently wrong)"
        );
        return 1;
    };

    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(file, false);
    let analysis_errors: Vec<_> = analysis
        .diagnostics
        .iter()
        .filter(|d| d.level == Level::Error)
        .cloned()
        .collect();
    if !analysis_errors.is_empty() {
        emit_cep_diags(file, &analysis_errors);
        return 1;
    }

    let module = &parse_result.module;
    let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();

    let Some(deco) = find_cep_decorator(module, &interner, "cep_prune") else {
        eprintln!("error: --cep-joint requires a @cep_prune(...) decorator on the model");
        return 1;
    };
    let mut diags = Vec::new();
    let Some(cfg) = nsl_semantic::cep::validate_cep_prune_decorator(deco, &resolve, &mut diags)
    else {
        emit_cep_diags(file, &diags);
        return 1;
    };
    if emit_cep_diags(file, &diags) {
        return 1;
    }

    let spec = match extract_model_spec(module, &resolve) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };

    let wm = match nsl_codegen::weight_aware::WeightMap::load(weights_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error: failed to load weights: {e}");
            return 1;
        }
    };
    if let Err(e) = cross_check_dims(&spec, &wm, &resolve) {
        eprintln!("{e}");
        return 1;
    }

    let target = resolve_cep_target(ov.target.as_deref(), cfg.target, &resolve);
    let input = match nsl_codegen::cep::build_prune_input(
        &cfg,
        spec.clone(),
        Some(&wm),
        &target,
        ov.sparsity,
    ) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let plan = nsl_codegen::cep::run_joint(input);
    println!("{}", plan.render_report());

    let out_path = ov.cep_out.clone().unwrap_or_else(|| default_cep_out(file));
    if let Err(e) = nsl_codegen::cep::write_prune_delta(&plan, &spec, &out_path) {
        eprintln!("error: failed to write delta: {e}");
        return 1;
    }
    println!("CEP joint delta written to {}", out_path.display());

    if let Some(weights_out) = ov.cep_emit_weights.as_ref() {
        let delta = nsl_codegen::cep::plan_to_prune_delta(&plan, &spec);
        match nsl_codegen::cep_slice::apply_prune_delta_to_weights(&wm, &spec, &delta) {
            Ok(sliced) => {
                let orig_params: usize = wm.entries().map(|(_, e)| e.num_elements).sum();
                let new_params: usize = sliced.values().map(|e| e.num_elements).sum();
                if let Err(e) = nsl_codegen::cep_slice::write_sliced_weights(&sliced, weights_out)
                {
                    eprintln!("error: failed to write sliced weights: {e}");
                    return 1;
                }
                println!(
                    "CEP sliced weights written to {} ({orig_params} -> {new_params} params)",
                    weights_out.display()
                );
            }
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        }
    }
    0
}

pub(crate) fn run_cep_search(file: &PathBuf, ov: &nsl_codegen::cep::CliOverrides) -> i32 {
    use nsl_codegen::cep_extract::extract_search_axes;

    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(file, false);
    let analysis_errors: Vec<_> = analysis
        .diagnostics
        .iter()
        .filter(|d| d.level == Level::Error)
        .cloned()
        .collect();
    if !analysis_errors.is_empty() {
        emit_cep_diags(file, &analysis_errors);
        return 1;
    }

    let module = &parse_result.module;
    let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();

    let Some(deco) = find_cep_decorator(module, &interner, "cep_search") else {
        eprintln!("error: --cep-search requires a @cep_search(...) decorator on the model");
        return 1;
    };
    let mut diags = Vec::new();
    let Some(cfg) = nsl_semantic::cep::validate_cep_search_decorator(deco, &resolve, &mut diags)
    else {
        emit_cep_diags(file, &diags);
        return 1;
    };
    if emit_cep_diags(file, &diags) {
        return 1;
    }

    let axes = match extract_search_axes(module, &resolve) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let target = resolve_cep_target(ov.target.as_deref(), cfg.target, &resolve);
    let input = match nsl_codegen::cep::build_search_input(&cfg, axes, &target) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let plan = nsl_codegen::cep::run_search(input);
    println!("{}", plan.render_report());

    let out_path = ov.cep_out.clone().unwrap_or_else(|| default_cep_out(file));
    if let Err(e) = nsl_codegen::cep::write_search_delta(&plan, &out_path) {
        eprintln!("error: failed to write delta: {e}");
        return 1;
    }
    println!("CEP delta written to {}", out_path.display());
    0
}

/// Run a bare compilation profile for a model (paper §7.1 'nsl check --cep-profile').
/// Prints a one-shot CompilationProfile without modification.
pub(crate) fn run_cep_profile(
    file: &PathBuf,
    weights: Option<&std::path::Path>,
    ov: &nsl_codegen::cep::CliOverrides,
) -> i32 {
    use nsl_codegen::cep_extract::{cross_check_dims, extract_model_spec};

    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(file, false);
    let analysis_errors: Vec<_> = analysis
        .diagnostics
        .iter()
        .filter(|d| d.level == Level::Error)
        .cloned()
        .collect();
    if !analysis_errors.is_empty() {
        emit_cep_diags(file, &analysis_errors);
        return 1;
    }

    let module = &parse_result.module;
    let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();

    let spec = match extract_model_spec(module, &resolve) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };

    // Optional weight cross-check (same logic as run_cep_prune).
    if let Some(weights_path) = weights {
        let wm = match nsl_codegen::weight_aware::WeightMap::load(weights_path) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("error: failed to load weights: {e}");
                return 1;
            }
        };
        if let Err(e) = cross_check_dims(&spec, &wm, &resolve) {
            eprintln!("{e}");
            return 1;
        }
    }

    let target = resolve_cep_target(ov.target.as_deref(), None, &resolve);
    let gpu = match nsl_codegen::gpu_specs::find_gpu(&target) {
        Some(g) => g,
        None => {
            eprintln!(
                "error: unknown CEP target '{}'. Supported: {}",
                target,
                nsl_codegen::cep::supported_gpus_list()
            );
            return 1;
        }
    };
    let profile = match nsl_codegen::cep_oracle::evaluate(&spec, gpu) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: CEP oracle failed: {e:?}");
            return 1;
        }
    };

    // §6.3-style one-shot profile output.
    println!("=== CEP Compilation Profile ===");
    println!("Target: {}", gpu.name);
    println!("Params: {:.1}M", spec.param_count() as f64 / 1e6);
    println!("Binary size: {}", cep_format_bytes_si(profile.binary_size_bytes));
    println!("Peak memory: {:.1}GB", profile.peak_memory_bytes as f64 / 1e9);
    println!("Estimated latency: {:.1}us/token", profile.estimated_latency_us);
    println!("WCET (roofline upper bound): {:.1}us/token", profile.wcet_us);
    println!("Kernel launches per forward: {}", profile.kernel_launches);
    println!("Total FLOPs: {}", cep_format_flops(profile.total_flops));
    println!("Total HBM bytes: {:.1}GB", profile.total_hbm_bytes as f64 / 1e9);
    println!("Roofline utilization: {:.2}", profile.roofline_utilization);
    println!("Fusion opportunities: {}", profile.fusion_events.len());
    0
}

fn cep_format_bytes_si(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{}MB", bytes / 1_000_000)
    } else if bytes >= 1_000 {
        format!("{}KB", bytes / 1_000)
    } else {
        format!("{}B", bytes)
    }
}

fn cep_format_flops(flops: u64) -> String {
    if flops >= 1_000_000_000_000 {
        format!("{:.1}T", flops as f64 / 1e12)
    } else if flops >= 1_000_000_000 {
        format!("{:.1}G", flops as f64 / 1e9)
    } else if flops >= 1_000_000 {
        format!("{:.1}M", flops as f64 / 1e6)
    } else if flops >= 1_000 {
        format!("{:.1}K", flops as f64 / 1e3)
    } else {
        format!("{}", flops)
    }
}
