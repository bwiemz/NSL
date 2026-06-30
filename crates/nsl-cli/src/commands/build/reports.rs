//! WRGA report helpers shared by every build path.
//!
//! `check_wrga_report_preconditions` fails fast when `--wrga-report` is used
//! without `--source-ad`; `emit_wrga_report` renders the plan to stdout/file
//! and offers it to the `--wrga-compare` capture slot. Extracted verbatim from
//! the former monolithic `build.rs`; behavior is unchanged.

use std::process;

use super::wrga_state::capture_wrga_plan_if_armed;

/// Task 3 (B.1): `--wrga-report` requires `--source-ad`, since WRGA only fires
/// in the source-AD lowering path. Detect and fail fast rather than silently
/// producing an empty report.
pub(super) fn check_wrga_report_preconditions(
    analysis: &nsl_semantic::AnalysisResult,
    wrga_report: Option<&std::path::Path>,
    options: &nsl_codegen::CompileOptions,
) {
    if wrga_report.is_none() || options.source_ad {
        return;
    }
    let has_wrga_decorators = !analysis.wrga_configs.is_empty()
        || !analysis.freeze_configs.is_empty()
        || !analysis.adapter_configs.is_empty();
    if has_wrga_decorators {
        eprintln!(
            "nsl: --wrga-report requires --source-ad when WRGA decorators are present; re-run with --source-ad"
        );
        process::exit(2);
    }
}

/// Task 3 (B.1): emit the WRGA report to stdout or a file. Mirrors the logic in
/// `run_build_single` / `run_build_multi`.
pub(super) fn emit_wrga_report(
    wrga_plan: &Option<nsl_codegen::wrga::WrgaPlan>,
    wrga_report: Option<&std::path::Path>,
) {
    // Always offer the plan to the capture slot; the helper is a no-op when
    // `--wrga-compare` hasn't armed it. Doing the capture up-here covers every
    // build path that funnels through `emit_wrga_report`.
    capture_wrga_plan_if_armed(wrga_plan);

    let Some(report_path) = wrga_report else { return; };
    match wrga_plan {
        Some(p) => {
            let report = p.render_report();
            if report_path == std::path::Path::new("-") {
                print!("{}", report);
            } else if let Err(e) = std::fs::write(report_path, &report) {
                eprintln!("error: could not write WRGA report: {e}");
                process::exit(1);
            }
        }
        None => {
            eprintln!(
                "nsl: --wrga-report requested but no @train block with WRGA decorators was compiled"
            );
        }
    }
}
