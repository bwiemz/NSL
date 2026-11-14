//! CEP (compilation-evaluated pruning) command frontend.
//!
//! `nsl build --cep-prune` / `nsl build --cep-joint` / `nsl check --cep-search`
//! / `nsl check --cep-profile`. These orchestrators tie together the AST
//! recognizer (`cep_extract`), the semantic decorator validators
//! (`nsl_semantic::cep`), the codegen bridge (`nsl_codegen::cep`), and the
//! delta, weights and source writers.

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
        if let StmtKind::Decorated { decorators, stmt: inner } = &stmt.kind
            && matches!(inner.kind, StmtKind::ModelDef(_))
        {
            for deco in decorators {
                if deco.name.len() == 1
                    && interner.resolve(deco.name[0].0).unwrap_or("") == want
                {
                    return Some(deco);
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
    run_cep_prune_mode(file, weights, ov, false)
}

/// `--cep-joint`: as `--cep-prune` (same `@cep_prune` decorator, weights
/// required), with whole-layer drops as a further kind of step.
pub(crate) fn run_cep_joint(
    file: &PathBuf,
    weights: Option<&std::path::Path>,
    ov: &nsl_codegen::cep::CliOverrides,
) -> i32 {
    run_cep_prune_mode(file, weights, ov, true)
}

/// Shared body of `--cep-prune` and `--cep-joint`.
///
/// Nothing is written unless the plan met its sparsity target and limits AND
/// every requested output could be produced: the delta, the sliced weights
/// and the rewritten source are all computed first, then written together.
fn run_cep_prune_mode(
    file: &PathBuf,
    weights: Option<&std::path::Path>,
    ov: &nsl_codegen::cep::CliOverrides,
    joint: bool,
) -> i32 {
    use nsl_codegen::cep_extract::{cross_check_dims, extract_model_spec, weights_dtype_bytes};
    use nsl_codegen::cep_weights::{load_weights, wants_nslm, WeightFormat};

    let flag = if joint { "--cep-joint" } else { "--cep-prune" };
    let Some(weights_path) = weights else {
        // One literal per flag: the feature-composition gate pins each
        // refusal's text at its log call.
        if joint {
            nsl_log::nsl_log!(ERROR, "cli",
                "error: --cep-joint requires --weights <file.nslm|file.safetensors> \
                 (importance is read from the weights; without them every component would \
                 count the same)"
            );
        } else {
            nsl_log::nsl_log!(ERROR, "cli",
                "error: --cep-prune requires --weights <file.nslm|file.safetensors> \
                 (importance is read from the weights; without them every component would \
                 count the same)"
            );
        }
        return 1;
    };

    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(file, false);
    // `frontend_with_flags` already exits on errors; this gate prints any
    // that slip through with source context before bailing.
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
        nsl_log::nsl_log!(ERROR, "cli", "error: {flag} requires a @cep_prune(...) decorator on the model");
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

    let mut spec = match extract_model_spec(module, &resolve) {
        Ok(s) => s,
        Err(e) => {
            nsl_log::nsl_log!(ERROR, "cli", "{e}");
            return 1;
        }
    };

    let loaded = match load_weights(weights_path) {
        Ok(w) => w,
        Err(e) => {
            nsl_log::nsl_log!(ERROR, "cli", "error: failed to load weights: {e}");
            return 1;
        }
    };
    if let Err(e) = cross_check_dims(&spec, &loaded.map, &resolve) {
        nsl_log::nsl_log!(ERROR, "cli", "{e}");
        return 1;
    }
    // The estimate prices compute and memory by value width; the weights say
    // what that is.
    if let Some(bytes) = weights_dtype_bytes(&loaded.map) {
        spec.dtype_bytes = bytes;
    }

    // Requests that cannot be honoured are refused before any work is done.
    let out_path = ov.cep_out.clone().unwrap_or_else(|| default_cep_out(file));
    for (flag, path) in [
        ("--cep-out", Some(&out_path)),
        ("--cep-emit-weights", ov.cep_emit_weights.as_ref()),
        ("--cep-emit-source", ov.cep_emit_source.as_ref()),
    ] {
        // Outputs are written together at the end; a directory that is not
        // there would otherwise be found after the first of them was written.
        if let Some(dir) = path.and_then(|p| p.parent())
            && !dir.as_os_str().is_empty()
            && !dir.is_dir()
        {
            nsl_log::nsl_log!(ERROR, "cli",
                "error: {flag}: the directory {} does not exist; no file was written",
                dir.display()
            );
            return 1;
        }
    }
    // An output that is also an input, or two outputs under one name, would
    // overwrite a file this run still needs or leave one result in place of two.
    let inputs = [resolved(file), resolved(weights_path)];
    let mut seen: Vec<PathBuf> = Vec::new();
    for (flag, path) in [
        ("--cep-out", Some(&out_path)),
        ("--cep-emit-weights", ov.cep_emit_weights.as_ref()),
        ("--cep-emit-source", ov.cep_emit_source.as_ref()),
    ] {
        let Some(path) = path else { continue };
        let full = resolved(path);
        if inputs.contains(&full) || seen.contains(&full) {
            nsl_log::nsl_log!(ERROR, "cli",
                "error: {flag}: {} is the model source, the input weights or another output of \
                 this run; every output needs a file of its own",
                path.display()
            );
            return 1;
        }
        seen.push(full);
    }
    if ov.cep_emit_source.is_some() && !cfg.preserve.is_empty() {
        nsl_log::nsl_log!(ERROR, "cli",
            "error: --cep-emit-source cannot be combined with `preserve`: the source declares \
             every layer with one block constructor, so a plan that leaves the preserved layers \
             at a different shape cannot be written. Remove `preserve`, or emit only the delta \
             and the weights."
        );
        return 1;
    }
    if let Some(out) = ov.cep_emit_weights.as_ref()
        && wants_nslm(out)
        && loaded.format != WeightFormat::Nslm
    {
        nsl_log::nsl_log!(ERROR, "cli",
            "error: --cep-emit-weights {} asks for an .nslm file, which `model_load` reads by \
             position, but --weights is a .safetensors file and that format does not record \
             the model's parameter order. Pass the .nslm that `model_save` wrote as --weights, \
             or emit .safetensors.",
            out.display()
        );
        return 1;
    }

    let target = resolve_cep_target(ov.target.as_deref(), cfg.target, &resolve);
    let mut input = match nsl_codegen::cep::build_prune_input(
        &cfg,
        spec.clone(),
        Some(&loaded.map),
        &target,
        ov.sparsity,
    ) {
        Ok(i) => i,
        Err(e) => {
            nsl_log::nsl_log!(ERROR, "cli", "{e}");
            return 1;
        }
    };
    // The emitted source has one constructor for all layers, so a plan that
    // is to be written as source must keep them at one shape.
    input.constraints.uniform = ov.cep_emit_source.is_some();
    let plan = if joint {
        nsl_codegen::cep::run_joint(input)
    } else {
        nsl_codegen::cep::run_prune(input)
    };
    let plan = match plan {
        Ok(p) => p,
        Err(e) => {
            nsl_log::nsl_log!(ERROR, "cli", "{e}");
            return 1;
        }
    };
    println!("{}", plan.render_report());

    if !plan.succeeded() {
        for f in plan.failures() {
            nsl_log::nsl_log!(ERROR, "cli", "error: {f}");
        }
        nsl_log::nsl_log!(ERROR, "cli",
            "error: the plan above does not do what was asked; no file was written. Lower \
             --cep-sparsity, relax the constraints{}.",
            if joint { "" } else { ", or use --cep-joint to allow dropping layers" }
        );
        return 1;
    }

    // Produce every requested output in memory first.
    let delta = nsl_codegen::cep::plan_to_prune_delta(&plan, &spec);
    let delta_json = match nsl_codegen::cep::prune_delta_json(&plan, &spec) {
        Ok(j) => j,
        Err(e) => {
            nsl_log::nsl_log!(ERROR, "cli", "error: failed to serialise the delta: {e}");
            return 1;
        }
    };
    let sliced = match ov.cep_emit_weights.as_ref() {
        None => None,
        Some(_) => match nsl_codegen::cep_slice::apply_prune_delta_to_weights(&loaded, &spec, &delta) {
            Ok(s) => Some(s),
            Err(e) => {
                nsl_log::nsl_log!(ERROR, "cli", "{e}");
                return 1;
            }
        },
    };
    let rewritten = match ov.cep_emit_source.as_ref() {
        None => None,
        Some(_) => {
            let original_source = match std::fs::read_to_string(file) {
                Ok(s) => s,
                Err(e) => {
                    nsl_log::nsl_log!(ERROR, "cli", "error: failed to re-read the source: {e}");
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
                Ok(out_src) => Some(out_src),
                Err(e) => {
                    nsl_log::nsl_log!(ERROR, "cli", "{e}");
                    nsl_log::nsl_log!(ERROR, "cli", "error: no file was written");
                    return 1;
                }
            }
        }
    };

    // Everything exists; write it, all of it or none of it.
    let weights_bytes = match (ov.cep_emit_weights.as_ref(), sliced.as_ref()) {
        (Some(path), Some(sliced)) => {
            match nsl_codegen::cep_slice::sliced_weights_bytes(sliced, path) {
                Ok(b) => Some(b),
                Err(e) => {
                    nsl_log::nsl_log!(ERROR, "cli", "error: failed to serialise the sliced weights: {e}");
                    return 1;
                }
            }
        }
        _ => None,
    };
    let mut outputs: Vec<(&std::path::Path, &[u8])> = vec![(out_path.as_path(), delta_json.as_bytes())];
    if let (Some(path), Some(bytes)) = (ov.cep_emit_weights.as_ref(), weights_bytes.as_ref()) {
        outputs.push((path.as_path(), bytes.as_slice()));
    }
    if let (Some(path), Some(src)) = (ov.cep_emit_source.as_ref(), rewritten.as_ref()) {
        outputs.push((path.as_path(), src.as_bytes()));
    }
    if let Err(e) = nsl_codegen::cep_weights::write_all_or_nothing(&outputs) {
        nsl_log::nsl_log!(ERROR, "cli", "error: failed to write {e}; no file was written");
        return 1;
    }

    println!(
        "CEP {}delta written to {}",
        if joint { "joint " } else { "" },
        out_path.display()
    );
    if let (Some(weights_out), Some(sliced)) = (ov.cep_emit_weights.as_ref(), sliced.as_ref()) {
        let orig: usize = loaded.map.entries().map(|(_, e)| e.num_elements).sum();
        let new: usize = sliced.iter().map(|e| e.num_elements).sum();
        println!(
            "CEP sliced weights written to {} ({orig} -> {new} stored values)",
            weights_out.display()
        );
    }
    if let Some(source_out) = ov.cep_emit_source.as_ref() {
        println!("CEP rewritten source written to {}", source_out.display());
    }
    0
}

/// `path` with its directory resolved, so two spellings of one file compare
/// equal whether or not the file exists yet.
fn resolved(path: &std::path::Path) -> PathBuf {
    let dir = match path.parent() {
        Some(d) if !d.as_os_str().is_empty() => d.to_path_buf(),
        _ => PathBuf::from("."),
    };
    let dir = dir.canonicalize().unwrap_or(dir);
    match path.file_name() {
        Some(name) => dir.join(name),
        None => dir,
    }
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
        nsl_log::nsl_log!(ERROR, "cli", "error: --cep-search requires a @cep_search(...) decorator on the model");
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
            nsl_log::nsl_log!(ERROR, "cli", "{e}");
            return 1;
        }
    };
    let target = resolve_cep_target(ov.target.as_deref(), cfg.target, &resolve);
    let input = match nsl_codegen::cep::build_search_input(&cfg, axes, &target) {
        Ok(i) => i,
        Err(e) => {
            nsl_log::nsl_log!(ERROR, "cli", "{e}");
            return 1;
        }
    };
    let plan = match nsl_codegen::cep::run_search(input) {
        Ok(p) => p,
        Err(e) => {
            nsl_log::nsl_log!(ERROR, "cli", "{e}");
            return 1;
        }
    };
    println!("{}", plan.render_report());
    if !plan.succeeded() {
        for f in plan.failures() {
            nsl_log::nsl_log!(ERROR, "cli", "error: {f}");
        }
        nsl_log::nsl_log!(ERROR, "cli",
            "error: there is no architecture to recommend; no file was written. Relax the              constraints or widen the @search axes."
        );
        return 1;
    }

    let out_path = ov.cep_out.clone().unwrap_or_else(|| default_cep_out(file));
    if let Err(e) = nsl_codegen::cep::write_search_delta(&plan, &out_path) {
        nsl_log::nsl_log!(ERROR, "cli", "error: failed to write delta: {e}");
        return 1;
    }
    println!("CEP delta written to {}", out_path.display());
    0
}

/// `nsl check --cep-profile`: print the cost estimate for the model as
/// written, without changing anything.
pub(crate) fn run_cep_profile(
    file: &PathBuf,
    weights: Option<&std::path::Path>,
    ov: &nsl_codegen::cep::CliOverrides,
) -> i32 {
    use nsl_codegen::cep_extract::{cross_check_dims, extract_model_spec, weights_dtype_bytes};

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

    let mut spec = match extract_model_spec(module, &resolve) {
        Ok(s) => s,
        Err(e) => {
            nsl_log::nsl_log!(ERROR, "cli", "{e}");
            return 1;
        }
    };

    // Optional weights: cross-checked against the source, and the source of
    // the value width the estimate prices.
    if let Some(weights_path) = weights {
        let loaded = match nsl_codegen::cep_weights::load_weights(weights_path) {
            Ok(w) => w,
            Err(e) => {
                nsl_log::nsl_log!(ERROR, "cli", "error: failed to load weights: {e}");
                return 1;
            }
        };
        if let Err(e) = cross_check_dims(&spec, &loaded.map, &resolve) {
            nsl_log::nsl_log!(ERROR, "cli", "{e}");
            return 1;
        }
        if let Some(bytes) = weights_dtype_bytes(&loaded.map) {
            spec.dtype_bytes = bytes;
        }
    }

    let target = resolve_cep_target(ov.target.as_deref(), None, &resolve);
    let gpu = match nsl_codegen::gpu_specs::find_gpu(&target) {
        Some(g) => g,
        None => {
            nsl_log::nsl_log!(ERROR, "cli",
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
            nsl_log::nsl_log!(ERROR, "cli", "error: CEP cost estimate failed: {e}");
            return 1;
        }
    };

    let tokens = (spec.batch as f64) * (spec.max_seq as f64);
    println!("=== CEP Compilation Profile ===");
    println!("Analytic estimate from the model's dimensions; nothing is compiled or measured.");
    println!("Target: {}", gpu.name);
    println!("{}", nsl_codegen::cep::assumptions_line(&spec));
    println!("Params: {:.1}M ({})", spec.param_count() as f64 / 1e6, spec.param_count());
    println!("Weights + nominal code size: {}", cep_format_bytes_si(profile.binary_size_bytes));
    println!(
        "Peak memory (parameters + widest activation; no KV cache, no training state): {:.2}GB",
        profile.peak_memory_bytes as f64 / 1e9
    );
    println!(
        "Estimated latency: {:.1}us per forward pass ({:.3}us/token over {} tokens)",
        profile.estimated_latency_us,
        profile.estimated_latency_us / tokens.max(1.0),
        tokens as u64
    );
    println!(
        "Roofline upper bound (compute + memory, no overlap): {:.1}us per forward pass",
        profile.wcet_us
    );
    println!("Kernel launches per forward (fusion template): {}", profile.kernel_launches);
    println!("Total FLOPs: {}", cep_format_flops(profile.total_flops));
    println!("Memory traffic: {:.1}GB", profile.total_hbm_bytes as f64 / 1e9);
    println!("Roofline utilization: {:.2}", profile.roofline_utilization);
    println!("Fusion events credited (fixed template): {}", profile.fusion_events.len());
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
