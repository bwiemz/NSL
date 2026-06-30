//! `nsl check --wrga-analyze` / `--wrga-compare` command implementations.
//!
//! Both reuse `run_build_inner` against a per-process temp directory to drive
//! the only path that produces a `WrgaPlan`, then render the analyze/compare
//! report and clean up so `nsl check` stays side-effect-free. Extracted
//! verbatim from the former monolithic `build.rs`; behavior is unchanged.

use std::path::PathBuf;

use super::normal::run_build_inner;
use super::wrga_state::{WrgaAblationOverrideGuard, WrgaPlanCaptureGuard, WrgaTargetOverrideGuard};

/// WRGA paper §8.3: `nsl check --wrga-analyze` — run the WRGA pass on the
/// source and emit `WrgaPlan::render_report()` without leaving a `.o` behind.
///
/// Reuses the existing `run_build_inner` to do all the heavy lifting (multi-
/// file resolution, source-AD lowering, codegen, WRGA bridge) because that
/// path is the only one that produces a `WrgaPlan` today, and reimplementing
/// it would duplicate ~200 lines of multi-file orchestration. The build is
/// redirected at a per-process temp directory; the directory is deleted after
/// the report has been written, so `nsl check` remains side-effect-free from
/// the user's perspective.
///
/// `wrga_target` (optional) overrides the `target=` field on every
/// `@wrga(...)` decorator before codegen — so `--wrga-target h100` from the
/// CLI wins over any source-level `@wrga(target="a100")`. The override is
/// installed by mutating the entry module's `wrga_configs` in
/// `frontend_with_flags`'s output via a CLI-side passthrough; we do this by
/// post-processing `WrgaInputs` inside `run_build_inner` — see
/// `apply_wrga_check_overrides` below.
///
/// Returns the process exit code: `0` on success, `2` on "no WRGA decorators
/// in source" (so CI can distinguish absence from compile failure), `1` on
/// any other error.
pub(crate) fn run_check_wrga_analyze(
    file: &PathBuf,
    report_path: &std::path::Path,
    wrga_target: Option<&str>,
    ablation: nsl_codegen::wrga::WrgaAblation,
) -> i32 {
    let _ablation_guard = if ablation.is_active() {
        Some(WrgaAblationOverrideGuard::set(ablation))
    } else {
        None
    };
    // Pre-check: surface "no decorators" as exit 2 BEFORE running codegen.
    // The build path would silently report "no plan" with exit 0, which the
    // paper's `--wrga-analyze` contract treats as a distinct error class.
    let (_interner, _parse_result, analysis) = crate::pipeline::frontend_with_flags(file, false);
    let has_wrga_decorators = !analysis.wrga_configs.is_empty()
        || !analysis.freeze_configs.is_empty()
        || !analysis.adapter_configs.is_empty();
    if !has_wrga_decorators {
        eprintln!(
            "nsl: --wrga-analyze: no @wrga / @freeze / @adapter decorators found in '{}'",
            file.display()
        );
        return 2;
    }

    // Redirect the build at a temp dir we own. Pre-create it so the linker has
    // a real path to drop the .o into, and so cleanup at the end is bounded.
    let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("nsl_check");
    let temp_dir = std::env::temp_dir().join(format!(
        "nsl_check_wrga_analyze_{}_{}",
        std::process::id(),
        stem
    ));
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("nsl: --wrga-analyze: could not create temp dir: {e}");
        return 1;
    }
    let temp_obj = temp_dir.join(format!("{stem}.o"));

    // The WRGA target override travels via a thread-local that the build path
    // checks just before constructing `WrgaInputs`. This avoids threading a
    // new param through `run_build_inner`'s six callers. The RAII guard
    // guarantees the cell is cleared even if `run_build_inner` panics, so the
    // override cannot leak into any in-process follow-on CLI invocation.
    let _override_guard = wrga_target.map(|t| WrgaTargetOverrideGuard::set(t.to_string()));

    let opts = nsl_codegen::CompileOptions {
        source_ad: true,
        ..nsl_codegen::CompileOptions::default()
    };

    // We pass our temp_obj as `output`. emit_obj=true so the linker is
    // skipped. The single-file build path ignores `output` and drops the .o
    // next to the source file (see `run_build_single` line ~4479); the
    // multi-file build path uses its own internal temp dir. To make `nsl
    // check` side-effect-free across BOTH dispatches, we explicitly clean
    // the source-adjacent .o path too, after the build returns.
    let source_adjacent_obj = file.with_file_name(format!("{stem}.o"));
    let source_adjacent_pre_existed = source_adjacent_obj.exists();

    run_build_inner(
        file,
        Some(temp_obj.clone()),
        true,   // emit_obj
        false,  // dump_ir
        true,   // quiet
        &opts,
        Some(report_path),
    );

    // Cleanup: our owned temp dir, plus any source-adjacent .o that the build
    // path dropped. Never delete a source-adjacent .o that ALREADY existed
    // before the call — that would be a user-data loss bug.
    let _ = std::fs::remove_dir_all(&temp_dir);
    if !source_adjacent_pre_existed && source_adjacent_obj.exists() {
        let _ = std::fs::remove_file(&source_adjacent_obj);
    }
    0
}

/// WRGA paper §8.3: `nsl check --wrga-compare` — run the WRGA pass on the
/// source and emit `WrgaPlan::render_compare_report()` (PEFT comparison
/// against LoRA / AdaLoRA / GaLore / ReFT) without leaving a `.o` behind.
///
/// Plumbing mirrors `run_check_wrga_analyze` exactly, except:
/// 1. The CLI-side capture slot is armed before invoking the build so the
///    plan can be pulled back out after `run_build_inner` returns.
/// 2. `wrga_report` is passed as `None`, suppressing the normal analyze
///    report from appearing on stdout (we only want the compare report).
///
/// Returns the same exit-code shape as `run_check_wrga_analyze`: `0` on
/// success, `2` on "no WRGA decorators in source", `1` on any other error.
pub(crate) fn run_check_wrga_compare(
    file: &PathBuf,
    report_path: &std::path::Path,
    wrga_target: Option<&str>,
    ablation: nsl_codegen::wrga::WrgaAblation,
) -> i32 {
    // `--wrga-compare` recovers each site's `(m + n)` weight footprint from
    // `RankAllocation.adapter_params / rank`. When `--wrga-ablate=spectral`
    // is active, `spectral_noop` returns `adapter_params: 0` (no SVD ran =
    // no shape data), which would silently zero every LoRA/AdaLoRA/ReFT row
    // in the PEFT table. Warn rather than continuing into a misleading report.
    if ablation.skip_spectral_allocation {
        eprintln!(
            "nsl: --wrga-compare: warning — --wrga-ablate=spectral disables the SVD that \
             recovers per-site weight shapes; the comparison table's LoRA / AdaLoRA / ReFT \
             rows will all show 0 params. Use --wrga-analyze if you want the bare ablated \
             plan."
        );
    }
    let _ablation_guard = if ablation.is_active() {
        Some(WrgaAblationOverrideGuard::set(ablation))
    } else {
        None
    };
    let (_interner, _parse_result, analysis) = crate::pipeline::frontend_with_flags(file, false);
    let has_wrga_decorators = !analysis.wrga_configs.is_empty()
        || !analysis.freeze_configs.is_empty()
        || !analysis.adapter_configs.is_empty();
    if !has_wrga_decorators {
        eprintln!(
            "nsl: --wrga-compare: no @wrga / @freeze / @adapter decorators found in '{}'",
            file.display()
        );
        return 2;
    }

    let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("nsl_check");
    let temp_dir = std::env::temp_dir().join(format!(
        "nsl_check_wrga_compare_{}_{}",
        std::process::id(),
        stem
    ));
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("nsl: --wrga-compare: could not create temp dir: {e}");
        return 1;
    }
    let temp_obj = temp_dir.join(format!("{stem}.o"));

    let _override_guard = wrga_target.map(|t| WrgaTargetOverrideGuard::set(t.to_string()));
    let _capture_guard = WrgaPlanCaptureGuard::arm();

    let opts = nsl_codegen::CompileOptions {
        source_ad: true,
        ..nsl_codegen::CompileOptions::default()
    };

    let source_adjacent_obj = file.with_file_name(format!("{stem}.o"));
    let source_adjacent_pre_existed = source_adjacent_obj.exists();

    run_build_inner(
        file,
        Some(temp_obj.clone()),
        true,   // emit_obj
        false,  // dump_ir
        true,   // quiet
        &opts,
        None,   // wrga_report — suppress the analyze report; we render compare below
    );

    // Belt-and-suspenders cleanup of artifacts. Same logic as the analyze path.
    let _ = std::fs::remove_dir_all(&temp_dir);
    if !source_adjacent_pre_existed && source_adjacent_obj.exists() {
        let _ = std::fs::remove_file(&source_adjacent_obj);
    }

    // Pull the captured plan out and render comparison.
    let Some(plan) = WrgaPlanCaptureGuard::take() else {
        eprintln!(
            "nsl: --wrga-compare: codegen completed but no @train block with WRGA decorators \
             was compiled; nothing to compare"
        );
        return 2;
    };
    let report = plan.render_compare_report();
    if report_path == std::path::Path::new("-") {
        print!("{}", report);
    } else if let Err(e) = std::fs::write(report_path, &report) {
        eprintln!("nsl: --wrga-compare: could not write report: {e}");
        return 1;
    }
    0
}
