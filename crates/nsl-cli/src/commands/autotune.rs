//! `nsl autotune` — measure `@autotune` kernel variants on this GPU and
//! cache the winners (roadmap item 10, the measured half).
//!
//! This command is the ONE production caller of the measured selection path
//! (`nsl_codegen::autotune::measure_variants` → `find_best_variant`).
//! Compiles never benchmark: they read the cache, and a `Measured` record
//! written here short-circuits the cost model on every later compile of the
//! same kernel on the same device. `autotune_selection_method_is_honest`
//! pins that division of labor in both directions.
//!
//! ## Why this drives a real compile
//!
//! The cache key hashes the kernel's AST bytes, whose `Symbol` indices
//! depend on the whole frontend's interning order — a reconstruction
//! outside the compile pipeline provably drifts (measured: two different
//! keys for the same kernel), writing records no compile ever reads. So
//! this command ARMS capture mode and runs the same `compile_returning_plan`
//! entry `nsl build` uses; the capture hands back each `@autotune` kernel's
//! name, params, per-variant PTX, and the compile path's OWN key. The e2e
//! gate proves the roundtrip: tune, then compile, and the compile reports a
//! cache hit.

use std::path::Path;
use std::process;

use nsl_lexer::Interner;

pub(crate) fn run_autotune(file: &Path, elements: usize, freeze: Option<&Path>) {
    if elements == 0 {
        eprintln!("error: --elements must be at least 1 (0 elements benches nothing)");
        process::exit(1);
    }
    let source = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read file '{}': {e}", file.display());
            process::exit(1);
        }
    };

    let mut source_map = nsl_errors::SourceMap::new();
    let file_id = source_map.add_file(file.display().to_string(), source.clone());
    let mut interner = Interner::new();
    let (tokens, lex_errors) = nsl_lexer::tokenize(&source, file_id, &mut interner);
    for diag in &lex_errors {
        source_map.emit_diagnostic(diag);
    }
    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    for diag in &parse_result.diagnostics {
        source_map.emit_diagnostic(diag);
    }
    let has_errors = lex_errors
        .iter()
        .chain(parse_result.diagnostics.iter())
        .any(|d| d.level == nsl_errors::Level::Error);
    if has_errors {
        eprintln!(
            "error: '{}' does not parse — fix the errors above first",
            file.display()
        );
        process::exit(1);
    }

    // Friendly refusal before any compile work: decorated statements parse
    // as `StmtKind::Decorated` wrapping the KernelDef, so @autotune is found
    // on the wrapper.
    let has_autotune_kernel = parse_result.module.stmts.iter().any(|s| {
        if let nsl_ast::stmt::StmtKind::Decorated { decorators, stmt } = &s.kind {
            matches!(stmt.kind, nsl_ast::stmt::StmtKind::KernelDef(_))
                && decorators.iter().any(|d| {
                    d.name.len() == 1
                        && interner.resolve(d.name[0].0).unwrap_or("") == "autotune"
                })
        } else {
            false
        }
    });
    if !has_autotune_kernel {
        eprintln!(
            "error: no @autotune kernels in '{}' — nothing to measure",
            file.display()
        );
        process::exit(1);
    }

    let device = nsl_codegen::autotune::DeviceIdentity::local();
    if !device.is_real_device() {
        eprintln!(
            "error: nsl autotune measures real kernels and needs a CUDA device \
             (detected: {}). The cost-model path needs no GPU and runs at \
             every compile automatically.",
            device.describe()
        );
        process::exit(1);
    }

    let analysis = nsl_semantic::analyze_with_imports(
        &parse_result.module,
        &mut interner,
        &std::collections::HashMap::new(),
        false,
    );
    if analysis.diagnostics.iter().any(|d| d.level == nsl_errors::Level::Error) {
        for diag in &analysis.diagnostics {
            source_map.emit_diagnostic(diag);
        }
        eprintln!("error: '{}' does not type-check", file.display());
        process::exit(1);
    }

    // Run the real compile entry with capture armed. Downstream codegen may
    // fail on this minimal-options compile (missing stdlib symbols, no
    // train-block config) — kernels compile early, so the capture is
    // already complete; swallow the tail error like `nsl check --csha`
    // does.
    nsl_codegen::autotune::arm_capture();
    let opts = nsl_codegen::CompileOptions::default();
    if let Err(e) = nsl_codegen::compile_returning_plan(
        &parse_result.module,
        &interner,
        &analysis.type_map,
        false,
        &opts,
    ) {
        eprintln!(
            "note: compile pipeline stopped after kernel selection ({}) — \
             the @autotune capture is unaffected",
            e.message
        );
    }
    let captured = nsl_codegen::autotune::take_capture();
    if captured.is_empty() {
        eprintln!(
            "error: the compile captured no @autotune kernels from '{}' — \
             kernel selection never ran (the kernels may be unreachable from \
             this file's compile)",
            file.display()
        );
        process::exit(1);
    }

    let mut frozen: Vec<nsl_codegen::autotune::AutotuneCacheRecord> = Vec::new();
    let mut failures = 0usize;
    for cap in &captured {
        println!(
            "[autotune] measuring '{}' ({} variants, {} elements)...",
            cap.name,
            cap.variant_ptx.len(),
            elements
        );
        match measure(cap, &device, elements) {
            Ok(winner) => {
                // Read back BEFORE claiming anything: "measured winner" must
                // mean a Measured record exists. The all-variants-failed
                // median fallback returns Ok but writes nothing, and a
                // cost-model record must never be presented (or frozen) as
                // a measurement (review findings, 2026-08-06).
                match nsl_codegen::autotune::load_cache_record(&cap.name, &cap.hash, &device, None)
                {
                    Ok(Some(rec))
                        if matches!(
                            rec.selection,
                            nsl_codegen::autotune::SelectionMethod::Measured
                        ) =>
                    {
                        println!("[autotune] '{}': measured winner {winner:?}", cap.name);
                        frozen.push(rec);
                    }
                    other => {
                        eprintln!(
                            "[autotune] '{}': no Measured record after tuning \
                             ({other:?}) — every variant failed to benchmark, so \
                             the selection fell back to an estimate; nothing to \
                             report or freeze",
                            cap.name
                        );
                        failures += 1;
                    }
                }
            }
            Err(e) => {
                eprintln!("[autotune] '{}': measurement failed: {e}", cap.name);
                failures += 1;
            }
        }
    }

    if let Some(out) = freeze {
        if frozen.is_empty() {
            eprintln!("error: --freeze: no measured records to write");
            process::exit(1);
        }
        let (json, sha) = nsl_codegen::autotune::freeze_records(&mut frozen);
        if let Err(e) = std::fs::write(out, &json) {
            eprintln!("error: --freeze {}: {e}", out.display());
            process::exit(1);
        }
        println!(
            "[autotune] froze {} record(s) to {}\n[autotune] pin with: --autotune-db {} --autotune-db-sha256 {}",
            frozen.len(),
            out.display(),
            out.display(),
            sha
        );
    }
    if failures > 0 {
        process::exit(1);
    }
}

#[cfg(feature = "cuda")]
fn measure(
    cap: &nsl_codegen::autotune::CapturedAutotuneKernel,
    device: &nsl_codegen::autotune::DeviceIdentity,
    elements: usize,
) -> Result<nsl_codegen::autotune::Variant, String> {
    nsl_codegen::autotune::measure_variants(cap, device, elements)
}

#[cfg(not(feature = "cuda"))]
fn measure(
    _cap: &nsl_codegen::autotune::CapturedAutotuneKernel,
    _device: &nsl_codegen::autotune::DeviceIdentity,
    _elements: usize,
) -> Result<nsl_codegen::autotune::Variant, String> {
    Err("this nsl was built without CUDA support (rebuild with --features cuda)".to_string())
}
