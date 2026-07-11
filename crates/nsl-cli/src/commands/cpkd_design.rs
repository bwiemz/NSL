//! CPKD Innovation 5 command frontend: `nsl check --cpkd-design-student`.
//!
//! Orchestrates: frontend analysis -> CEP recognizer (`extract_model_spec` +
//! `extract_search_axes`) -> optional teacher `WeightMap` load ->
//! `nsl_codegen::cpkd_student::design_student` -> report to stdout.
//! Modeled on `commands/cep.rs::run_cep_search`. Advisory: prints the design
//! report only — no delta file, no source emission in v1.

use nsl_errors::{Level, SourceMap};
use std::path::{Path, PathBuf};

/// Parse a human parameter budget: a raw integer ("350000000") or a number
/// with a case-insensitive K/M/B suffix ("125M", "1.1B", "64k").
fn parse_param_budget(s: &str) -> Result<u64, String> {
    let t = s.trim();
    let mut chars = t.chars();
    let Some(last) = chars.next_back() else {
        return Err("empty parameter budget (expected e.g. 125M, 1.1B, or 350000000)".to_string());
    };
    let (head, mult) = match last.to_ascii_uppercase() {
        'K' => (chars.as_str(), 1_000u64),
        'M' => (chars.as_str(), 1_000_000u64),
        'B' => (chars.as_str(), 1_000_000_000u64),
        _ => (t, 1u64),
    };
    let bad = || {
        format!(
            "invalid parameter budget '{s}' (expected a raw integer or a number \
             with a K/M/B suffix, e.g. 125M, 1.1B, 350000000)"
        )
    };
    if head.is_empty() {
        return Err(bad());
    }
    if mult == 1 {
        return head.parse::<u64>().map_err(|_| bad());
    }
    let v: f64 = head.parse().map_err(|_| bad())?;
    if !v.is_finite() || v < 0.0 {
        return Err(bad());
    }
    Ok((v * mult as f64).round() as u64)
}

/// True if any decorated statement in the module carries a `@search(...)`
/// decorator. `extract_search_axes` silently falls back to singleton axes
/// (the teacher itself) when no `@search` decorators exist — for student
/// design that is a degenerate one-point "search", so we refuse loudly
/// instead.
fn module_has_search_decorator(module: &nsl_ast::Module, interner: &nsl_lexer::Interner) -> bool {
    use nsl_ast::stmt::StmtKind;
    for stmt in &module.stmts {
        if let StmtKind::Decorated { decorators, .. } = &stmt.kind {
            for deco in decorators {
                if deco.name.len() == 1
                    && interner.resolve(deco.name[0].0).unwrap_or("") == "search"
                {
                    return true;
                }
            }
        }
    }
    false
}

/// Emit diagnostics with source context (same mechanism `commands/cep.rs`
/// uses). Returns `true` if any diagnostic is at `Level::Error`.
fn emit_diags(file: &Path, diags: &[nsl_errors::Diagnostic]) -> bool {
    let source = std::fs::read_to_string(file).unwrap_or_default();
    let mut source_map = SourceMap::new();
    source_map.add_file(file.display().to_string(), source);
    for diag in diags {
        source_map.emit_diagnostic(diag);
    }
    diags.iter().any(|d| d.level == Level::Error)
}

pub(crate) fn run_cpkd_design(
    file: &PathBuf,
    budget_str: &str,
    target: Option<&str>,
    weights: Option<&Path>,
) -> i32 {
    use nsl_codegen::cep_extract::{extract_model_spec, extract_search_axes};

    let target_params = match parse_param_budget(budget_str) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("error: --cpkd-design-student: {e}");
            return 1;
        }
    };

    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(file, false);
    let analysis_errors: Vec<_> = analysis
        .diagnostics
        .iter()
        .filter(|d| d.level == Level::Error)
        .cloned()
        .collect();
    if !analysis_errors.is_empty() {
        emit_diags(file, &analysis_errors);
        return 1;
    }

    let module = &parse_result.module;
    let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();

    // Teacher spec first, so unrecognizable models surface the CEP
    // recognizer's refusal (which names what was expected vs. found).
    let teacher_spec = match extract_model_spec(module, &resolve) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: {e}");
            eprintln!(
                "note: --cpkd-design-student needs a CEP-recognizable teacher model \
                 (canonical GroupedQueryAttention + SwiGLUFFN blocks)"
            );
            return 1;
        }
    };

    if !module_has_search_decorator(module, &interner) {
        eprintln!(
            "error: --cpkd-design-student requires @search(axis, [values]) decorators \
             on the model; without them the search space contains only the teacher \
             architecture itself. Add e.g. @search(d_model, [256, 384, 512]); \
             axes: d_model, n_layers, n_heads, n_kv_heads, d_ff, activation, norm"
        );
        return 1;
    }

    let axes = match extract_search_axes(module, &resolve) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("error: {e}");
            return 1;
        }
    };

    let wm = match weights {
        Some(p) => match nsl_codegen::weight_aware::WeightMap::load(p) {
            Ok(w) => Some(w),
            Err(e) => {
                eprintln!("error: failed to load --weights {}: {e}", p.display());
                return 1;
            }
        },
        None => None,
    };

    // Target resolution mirrors resolve_cep_target (commands/cep.rs): CLI
    // flag wins, else the H100-SXM default. (No decorator source here — the
    // design mode is CLI-driven in v1.)
    let target_gpu = target.unwrap_or("H100-SXM");

    let input = nsl_codegen::cpkd_student::StudentDesignInput {
        axes,
        teacher_spec: teacher_spec.clone(),
        weights: wm.as_ref(),
        target_gpu,
        target_params,
        objective: nsl_codegen::cep_search::NasObjective::ParamEfficiency,
    };
    match nsl_codegen::cpkd_student::design_student(input) {
        Ok(design) => {
            print!(
                "{}",
                nsl_codegen::cpkd_student::render_design_report(
                    &design,
                    &teacher_spec,
                    target_params,
                    target_gpu,
                )
            );
            0
        }
        Err(e) => {
            eprintln!("error: {e}");
            1
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse_param_budget;

    #[test]
    fn parses_suffixed_and_raw_budgets() {
        assert_eq!(parse_param_budget("125M").unwrap(), 125_000_000);
        assert_eq!(parse_param_budget("1.1B").unwrap(), 1_100_000_000);
        assert_eq!(parse_param_budget("350000000").unwrap(), 350_000_000);
        assert_eq!(parse_param_budget("64k").unwrap(), 64_000);
        assert_eq!(parse_param_budget("2b").unwrap(), 2_000_000_000);
        assert_eq!(parse_param_budget(" 12M ").unwrap(), 12_000_000);
        assert_eq!(parse_param_budget("0").unwrap(), 0);
    }

    #[test]
    fn rejects_garbage_budgets() {
        for bad in ["", "M", "12X", "abc", "-5", "-5M", "1.5", "1e9", "12MM", "NaNB"] {
            assert!(
                parse_param_budget(bad).is_err(),
                "expected Err for {bad:?}"
            );
        }
    }

    #[test]
    fn budget_errors_name_the_input_and_syntax() {
        let err = parse_param_budget("garbage").unwrap_err();
        assert!(err.contains("'garbage'"), "err: {err}");
        assert!(err.contains("125M"), "err shows examples: {err}");
    }
}
