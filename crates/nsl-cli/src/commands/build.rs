//! `nsl build` / `nsl run` / `nsl zk` and the WRGA analysis commands.
//!
//! This is the tightly-coupled build cluster: the build paths (single/multi,
//! shared-lib, standalone, ZK), the `nsl run` execution path, the `nsl zk`
//! verify/stats subcommands, and the WRGA report/analysis helpers (which the
//! build paths invoke). They share enough state that they live together in one
//! module. Shared frontend/config helpers remain in `main.rs` and are reached
//! via `crate::`. Extracted verbatim from `main.rs`; behavior is unchanged.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process;

use nsl_errors::SourceMap;
use nsl_lexer::Interner;

/// Check if a file has any import statements or train blocks by quick-scanning.
/// Train blocks need multi-file compilation because optimizer stdlib modules
/// are auto-imported.
fn needs_multi_file(file: &PathBuf) -> bool {
    if let Ok(source) = std::fs::read_to_string(file) {
        source.lines().any(|line| {
            let trimmed = line.trim();
            // Skip comments — they can contain import-like text
            if trimmed.starts_with('#') || trimmed.starts_with("//") {
                return false;
            }
            (trimmed.starts_with("from ") && trimmed.contains(" import "))
                || (trimmed.starts_with("import ") && trimmed.contains(" as "))
                || trimmed.starts_with("train(")
                || trimmed.starts_with("train (")
        })
    } else {
        false
    }
}

pub(crate) fn run_build(
    file: &PathBuf,
    output: Option<PathBuf>,
    emit_obj: bool,
    dump_ir: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    run_build_inner(file, output, emit_obj, dump_ir, false, options, wrga_report);
}

/// Task 3 (B.1): `--wrga-report` requires `--source-ad`, since WRGA only fires
/// in the source-AD lowering path. Detect and fail fast rather than silently
/// producing an empty report.
fn check_wrga_report_preconditions(
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

/// RAII guard that clears `WRGA_TARGET_OVERRIDE` on drop. Holding this guard
/// keeps the thread-local set for the lifetime of the build; dropping it
/// (including on panic) restores `None`.
struct WrgaTargetOverrideGuard;

impl WrgaTargetOverrideGuard {
    fn set(value: String) -> Self {
        WRGA_TARGET_OVERRIDE.with(|c| *c.borrow_mut() = Some(value));
        Self
    }
}

impl Drop for WrgaTargetOverrideGuard {
    fn drop(&mut self) {
        WRGA_TARGET_OVERRIDE.with(|c| *c.borrow_mut() = None);
    }
}

thread_local! {
    /// CLI-side override for `WrgaInputs::wrga[*].target`. Set by
    /// `run_check_wrga_analyze` before invoking the build pipeline; read by
    /// `analysis_to_wrga_inputs` / `module_data_to_wrga_inputs` to patch each
    /// decorator's target field before the bridge ships to codegen.
    static WRGA_TARGET_OVERRIDE: std::cell::RefCell<Option<String>>
        = const { std::cell::RefCell::new(None) };

    /// CLI-side capture slot for the `WrgaPlan` produced during a check-mode
    /// build. Set by `run_check_wrga_compare` before invoking the build; read
    /// at every site that has a fresh plan in scope, so the comparison report
    /// can be rendered from the plan after the build returns. `None` outside
    /// a capture window — no overhead on normal `nsl build` paths.
    static WRGA_PLAN_CAPTURE: std::cell::RefCell<Option<nsl_codegen::wrga::WrgaPlan>>
        = const { std::cell::RefCell::new(None) };

    /// CLI-side override for `WrgaInputs::ablation`. Set by
    /// `run_check_wrga_analyze` / `run_check_wrga_compare` when the user
    /// passes `--wrga-ablate=<flags>`. Read by `apply_wrga_check_overrides`
    /// (which now also forwards ablation) just before the bridge ships to
    /// codegen. `None` outside a check window so normal `nsl build` paths
    /// are not affected.
    static WRGA_ABLATION_OVERRIDE: std::cell::RefCell<Option<nsl_codegen::wrga::WrgaAblation>>
        = const { std::cell::RefCell::new(None) };
}

/// RAII guard for `WRGA_ABLATION_OVERRIDE`. Mirrors `WrgaTargetOverrideGuard`.
struct WrgaAblationOverrideGuard;

impl WrgaAblationOverrideGuard {
    fn set(value: nsl_codegen::wrga::WrgaAblation) -> Self {
        WRGA_ABLATION_OVERRIDE.with(|c| *c.borrow_mut() = Some(value));
        Self
    }
}

impl Drop for WrgaAblationOverrideGuard {
    fn drop(&mut self) {
        WRGA_ABLATION_OVERRIDE.with(|c| *c.borrow_mut() = None);
    }
}

/// RAII guard that arms `WRGA_PLAN_CAPTURE` for the lifetime of the guard,
/// then disarms on drop. Use the returned guard to keep capture live for the
/// duration of a single `run_build_inner` invocation, then call `take()` to
/// extract the captured plan before the guard drops.
struct WrgaPlanCaptureGuard;

impl WrgaPlanCaptureGuard {
    fn arm() -> Self {
        WRGA_PLAN_CAPTURE.with(|c| *c.borrow_mut() = None);
        Self
    }

    /// Extract the captured plan, leaving `None` behind. Returns `None` if no
    /// plan was captured (e.g. compile failed, or no @train block had WRGA
    /// decorators).
    fn take() -> Option<nsl_codegen::wrga::WrgaPlan> {
        WRGA_PLAN_CAPTURE.with(|c| c.borrow_mut().take())
    }
}

impl Drop for WrgaPlanCaptureGuard {
    fn drop(&mut self) {
        WRGA_PLAN_CAPTURE.with(|c| *c.borrow_mut() = None);
    }
}

/// Capture the just-produced `WrgaPlan` into `WRGA_PLAN_CAPTURE` if and only
/// if the capture slot is currently armed (i.e. someone called
/// `WrgaPlanCaptureGuard::arm`). No-op otherwise. Called at every site in
/// `run_build_single` / `run_build_multi` / `run_build_zk` / `run_build_standalone`
/// just after `compile_returning_plan` returns. Captures the FIRST non-`None`
/// plan we see, so multi-file paths that compile multiple modules don't
/// overwrite the entry-module plan with a dependency's empty one.
fn capture_wrga_plan_if_armed(plan: &Option<nsl_codegen::wrga::WrgaPlan>) {
    let Some(p) = plan else { return };
    WRGA_PLAN_CAPTURE.with(|c| {
        let mut slot = c.borrow_mut();
        if slot.is_none() {
            *slot = Some(p.clone());
        }
    });
}

/// Apply every CLI-side check-mode override onto a freshly-built `WrgaInputs`
/// before it ships to codegen. Called from both bridge functions in
/// `pipeline.rs` (single-file `analysis_to_wrga_inputs` and multi-file
/// `module_data_to_wrga_inputs`).
///
/// Two overrides, both populated by `run_check_wrga_analyze` /
/// `run_check_wrga_compare` via their respective RAII guards and read here:
///
/// 1. `WRGA_TARGET_OVERRIDE` (paper §8.3) — copied onto every
///    `WrgaDecoratorConfig::target`. When the source has no `@wrga(...)` at
///    all (only `@freeze` / `@adapter`), a minimal Auto-mode config is
///    inserted so the target choice still reaches the codegen-side
///    `wrga::run`.
/// 2. `WRGA_ABLATION_OVERRIDE` (paper §9.3) — copied onto
///    `WrgaInputs::ablation` so the codegen-side WRGA driver honours the
///    requested per-Innovation skip flags.
///
/// Both overrides share the same trigger surface (only `nsl check
/// --wrga-analyze | --wrga-compare` sets either), so a single bridge avoids
/// gratuitous fanout. On normal `nsl build` paths both thread-locals are
/// `None` and this fn is a quick noop.
pub(crate) fn apply_wrga_check_overrides(inputs: &mut nsl_codegen::WrgaInputs) {
    WRGA_TARGET_OVERRIDE.with(|c| {
        let Some(target) = c.borrow().clone() else { return };
        for cfg in &mut inputs.wrga {
            cfg.target = Some(target.clone());
        }
        if inputs.wrga.is_empty() {
            inputs.wrga.push(nsl_codegen::WrgaDecoratorConfig {
                mode: nsl_ast::block::WrgaMode::Auto,
                budget: None,
                target: Some(target),
                layers: Vec::new(),
                custom_adapter: None,
            });
        }
    });
    WRGA_ABLATION_OVERRIDE.with(|c| {
        if let Some(abl) = *c.borrow() {
            inputs.ablation = abl;
        }
    });
}

/// Task 3 (B.1): emit the WRGA report to stdout or a file. Mirrors the logic in
/// `run_build_single` / `run_build_multi`.
fn emit_wrga_report(
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

/// M62a: Build as a shared library (.so/.dylib/.dll) with stable C API.
pub(crate) fn run_build_shared(
    file: &PathBuf,
    output: Option<PathBuf>,
    dump_ir: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    if needs_multi_file(file) {
        run_build_shared_multi(file, output, dump_ir, options, wrga_report);
    } else {
        run_build_shared_single(file, output, dump_ir, options, wrga_report);
    }
}

/// M62a: Single-file shared library build.
fn run_build_shared_single(
    file: &PathBuf,
    output: Option<PathBuf>,
    dump_ir: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(file, options.linear_types_enabled);

    // Task 3 (B.1): forward WRGA decorator configs so they take effect on the
    // shared-library build path, and fail fast if --wrga-report is combined
    // with decorators but without --source-ad.
    check_wrga_report_preconditions(&analysis, wrga_report, options);
    let mut options = options.clone();
    options.wrga_inputs = Some(crate::pipeline::analysis_to_wrga_inputs(&analysis));
    options.fused_ce_configs = crate::pipeline::analysis_to_fused_ce_configs(&analysis);
    // M62 Task 6: route weight_index_map from semantic analysis into codegen.
    options.weight_index_map = analysis.weight_index_map.clone();
    // M62: allocate a slot the compiler publishes @export functions into,
    // so we can emit the C header after the shared library is linked.
    let exports_slot: std::sync::Arc<
        std::sync::Mutex<Option<Vec<nsl_codegen::c_header::ExportInfo>>>,
    > = std::sync::Arc::new(std::sync::Mutex::new(None));
    options.export_functions_out = Some(exports_slot.clone());
    let options = &options;

    // Codegen with PIC enabled (shared_lib=true in options)
    let (obj_bytes, wrga_plan) = match nsl_codegen::compile_returning_plan(
        &parse_result.module,
        &interner,
        &analysis.type_map,
        dump_ir,
        options,
    ) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("codegen error: {e}");
            process::exit(1);
        }
    };

    emit_wrga_report(&wrga_plan, wrga_report);

    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| {
            eprintln!("error: invalid input filename '{}'", file.display());
            process::exit(1);
        });
    let obj_path = file.with_file_name(format!("{stem}.o"));

    if let Err(e) = std::fs::write(&obj_path, &obj_bytes) {
        eprintln!("error: could not write object file: {e}");
        process::exit(1);
    }

    let lib_path = if let Some(out) = output {
        out
    } else {
        nsl_codegen::linker::default_shared_lib_path(file)
    };

    // M62 Task 6: collect @export symbol names so the linker can force them
    // into the DLL export table (Linkage::Export alone isn't enough on MSVC).
    let export_symbols: Vec<String> = exports_slot
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|v| v.iter().map(|e| e.symbol_name.clone()).collect()))
        .unwrap_or_default();
    // Sibling packed-array dispatch wrappers (`<name>__nsl_dispatch`) emitted
    // alongside every typed `<name>` wrapper. The runtime ExportRegistry
    // dlsyms these via the suffix; MSVC requires them in the explicit export
    // list to survive linking.
    let dispatch_symbols: Vec<String> = export_symbols
        .iter()
        .map(|s| format!("{}__nsl_dispatch", s))
        .collect();
    // M62 Task 9: also re-export the runtime lifecycle symbols so that ctypes
    // callers can call nsl_model_create / nsl_model_destroy / nsl_get_last_error
    // directly from the generated shared lib without loading a separate runtime DLL.
    // `mut` is only used when --features onnx-rt-op is on; harmless otherwise.
    #[allow(unused_mut)]
    let mut runtime_exports: Vec<&'static str> = vec![
        "nsl_model_create",
        "nsl_model_create_with_lib",
        "nsl_model_destroy",
        "nsl_model_forward",
        "nsl_model_forward_grad",
        "nsl_model_backward",
        "nsl_grad_context_destroy",
        "nsl_model_call",
        "nsl_model_call_dlpack",
        "nsl_model_export_count",
        "nsl_model_lookup_function",
        "nsl_model_get_weight_ptrs",
        "nsl_model_get_num_weights",
        "nsl_model_num_weights",
        "nsl_get_last_error",
        "nsl_clear_error",
        "nsl_set_error_cstr",
        "nsl_desc_to_tensor",
        "nsl_tensor_to_desc_ffi",
        "nsl_tensor_free",
        "nsl_get_num_exports",
        "nsl_get_export_name",
        "nsl_dispatch_apply_result",
        "nsl_dl_path_for_fn_addr",
        "nsl_free_cstr",
    ];
    // M62b Spec C — when nsl-cli is built with --features onnx-rt-op the nsl-runtime
    // crate compiles in `RegisterCustomOps` (the ORT custom-op registration entry
    // point). MSVC requires the symbol to be in the explicit export list to survive
    // linking into the .dll; on other platforms the extra entry is harmless.
    #[cfg(feature = "onnx-rt-op")]
    runtime_exports.push("RegisterCustomOps");
    let mut export_refs: Vec<&str> = export_symbols.iter().map(|s| s.as_str()).collect();
    export_refs.extend(dispatch_symbols.iter().map(|s| s.as_str()));
    export_refs.extend_from_slice(&runtime_exports);

    match nsl_codegen::linker::link_shared_with_exports(
        std::slice::from_ref(&obj_path),
        &lib_path,
        &export_refs,
    ) {
        Ok(()) => {
            let _ = std::fs::remove_file(&obj_path);
            println!("Built shared library {}", lib_path.display());
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }

    emit_c_header_if_any(&exports_slot, &lib_path);
}

/// M62: Write a matching C header next to the shared library when the
/// compile published one or more `@export` functions. No-op otherwise.
fn emit_c_header_if_any(
    exports_slot: &std::sync::Arc<
        std::sync::Mutex<Option<Vec<nsl_codegen::c_header::ExportInfo>>>,
    >,
    lib_path: &std::path::Path,
) {
    let exports = match exports_slot.lock() {
        Ok(guard) => match guard.as_ref() {
            Some(v) if !v.is_empty() => v.clone(),
            _ => return,
        },
        Err(_) => return,
    };
    let module_name = lib_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let header = nsl_codegen::c_header::emit(&exports, module_name);
    let header_path = lib_path.with_extension("h");
    match std::fs::write(&header_path, header) {
        Ok(()) => println!("Wrote C header {}", header_path.display()),
        Err(e) => eprintln!("warning: failed to write header '{}': {e}", header_path.display()),
    }
}

/// M62a: Multi-file shared library build.
fn run_build_shared_multi(
    file: &std::path::Path,
    output: Option<PathBuf>,
    dump_ir: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    let mut entry_wrga_plan: Option<nsl_codegen::wrga::WrgaPlan> = None;
    let mut source_map = SourceMap::new();
    let mut interner = Interner::new();
    // M62: allocate a slot the entry-module compile publishes @export
    // functions into, so we can emit a C header after linking.
    let exports_slot: std::sync::Arc<
        std::sync::Mutex<Option<Vec<nsl_codegen::c_header::ExportInfo>>>,
    > = std::sync::Arc::new(std::sync::Mutex::new(None));

    let graph = match crate::loader::load_all_modules(file, &mut source_map, &mut interner) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    let temp_dir = std::env::temp_dir().join(format!("nsl_shared_{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("error: could not create temp dir: {e}");
        process::exit(1);
    }

    let mut obj_files: Vec<PathBuf> = Vec::new();

    // Compile each module in dependency order (same as run_build_multi but with shared_lib options)
    for path in &graph.dep_order {
        let mod_data = &graph.modules[path];
        let is_entry = *path == graph.entry;

        let obj_bytes = if is_entry {
            let mut imported_fns = Vec::new();
            let mut imported_struct_layouts: HashMap<String, nsl_codegen::context::StructLayout> = HashMap::new();
            let mut imported_model_names = std::collections::HashSet::new();
            let mut imported_enum_variants = HashMap::new();
            let mut imported_enum_defs = HashMap::new();
            let mut imported_model_method_bodies: HashMap<String, HashMap<String, nsl_ast::decl::FnDef>> = HashMap::new();
            let mut imported_model_field_types: HashMap<String, HashMap<String, String>> = HashMap::new();

            for dep_path in &graph.dep_order {
                if dep_path == &graph.entry {
                    continue;
                }
                let dep_data = &graph.modules[dep_path];

                let mut temp_compiler = match nsl_codegen::compiler::Compiler::new(
                    &interner,
                    &dep_data.type_map,
                    &nsl_codegen::CompileOptions::default(),
                ) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("codegen error: {e}");
                        process::exit(1);
                    }
                };

                for stmt in &dep_data.ast.stmts {
                    if let nsl_ast::stmt::StmtKind::FnDef(fn_def) = &stmt.kind {
                        let raw_name = interner.resolve(fn_def.name.0).unwrap_or("<unknown>").to_string();
                        let mangled_name = crate::mangling::mangle(&dep_data.module_prefix, &raw_name);
                        let sig = temp_compiler.build_fn_signature(fn_def);
                        imported_fns.push((raw_name, mangled_name, sig));
                    }
                }

                // Inject previously-collected struct layouts from earlier deps
                for (name, layout) in &imported_struct_layouts {
                    temp_compiler.types.struct_layouts.insert(name.clone(), layout.clone());
                }

                if let Err(e) = temp_compiler.collect_structs(&dep_data.ast.stmts) {
                    eprintln!("codegen error: {e}");
                    process::exit(1);
                }
                if let Err(e) = temp_compiler.collect_models(&dep_data.ast.stmts) {
                    eprintln!("codegen error: {e}");
                    process::exit(1);
                }

                let model_sigs = temp_compiler.build_model_signatures(&dep_data.ast.stmts);
                imported_fns.extend(model_sigs);

                for stmt in &dep_data.ast.stmts {
                    if let nsl_ast::stmt::StmtKind::ModelDef(md) = &stmt.kind {
                        let model_name = interner.resolve(md.name.0).unwrap_or("<unknown>").to_string();
                        imported_model_names.insert(model_name);
                    }
                }

                for (name, layout) in temp_compiler.types.struct_layouts.drain() {
                    imported_struct_layouts.insert(name, layout);
                }

                // Propagate model field types from temp compiler (populated by collect_models)
                for (name, fields) in temp_compiler.models.model_field_types.drain() {
                    imported_model_field_types.entry(name).or_insert(fields);
                }

                // Extract model method bodies directly from AST
                for stmt in &dep_data.ast.stmts {
                    if let nsl_ast::stmt::StmtKind::ModelDef(md) = &stmt.kind {
                        let model_name = interner.resolve(md.name.0).unwrap_or("<unknown>").to_string();
                        let mut body_map = HashMap::new();
                        for member in &md.members {
                            if let nsl_ast::decl::ModelMember::Method(fn_def, _) = member {
                                let method_name = interner.resolve(fn_def.name.0).unwrap_or("<unknown>").to_string();
                                body_map.insert(method_name, fn_def.clone());
                            }
                        }
                        if !body_map.is_empty() {
                            imported_model_method_bodies.entry(model_name).or_insert(body_map);
                        }
                    }
                }

                imported_enum_variants.extend(dep_data.enum_variants.clone());
                imported_enum_defs.extend(dep_data.enum_defs.clone());
            }

            // Task 3 (B.1): forward entry-module decorator configs to codegen
            // and fail fast if --wrga-report is used without --source-ad.
            if wrga_report.is_some() && !options.source_ad {
                let has_wrga_decorators = !mod_data.wrga_configs.is_empty()
                    || !mod_data.freeze_configs.is_empty()
                    || !mod_data.adapter_configs.is_empty();
                if has_wrga_decorators {
                    eprintln!(
                        "nsl: --wrga-report requires --source-ad when WRGA decorators are present; re-run with --source-ad"
                    );
                    process::exit(2);
                }
            }
            let mut entry_options = options.clone();
            entry_options.wrga_inputs = Some(crate::pipeline::module_data_to_wrga_inputs(mod_data));
            entry_options.fused_ce_configs = crate::pipeline::module_data_to_fused_ce_configs(mod_data);
            entry_options.export_functions_out = Some(exports_slot.clone());
            // M62: route entry-module weight_index_map so @export model methods
            // can resolve `self.<field>` → weight index on the multi-file path.
            entry_options.weight_index_map = mod_data.weight_index_map.clone();
            let entry_options = &entry_options;

            match nsl_codegen::compile_entry_returning_plan(
                &mod_data.ast,
                &interner,
                &mod_data.type_map,
                &imported_fns,
                imported_struct_layouts,
                imported_model_names,
                imported_enum_variants,
                imported_enum_defs,
                imported_model_method_bodies,
                imported_model_field_types,
                dump_ir,
                entry_options,
            ) {
                Ok((bytes, plan)) => {
                    entry_wrga_plan = plan;
                    bytes
                }
                Err(e) => {
                    eprintln!("codegen error in '{}': {e}", path.display());
                    process::exit(1);
                }
            }
        } else {
            match nsl_codegen::compile_module(
                &mod_data.ast,
                &interner,
                &mod_data.type_map,
                &mod_data.module_prefix,
                dump_ir,
                options,
            ) {
                Ok(bytes) => bytes,
                Err(e) => {
                    eprintln!("codegen error in '{}': {e}", path.display());
                    process::exit(1);
                }
            }
        };

        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let obj_path = temp_dir.join(format!("{stem}_{}.o", obj_files.len()));

        if let Err(e) = std::fs::write(&obj_path, &obj_bytes) {
            eprintln!("error: could not write object file '{}': {e}", obj_path.display());
            process::exit(1);
        }

        obj_files.push(obj_path);
    }

    emit_wrga_report(&entry_wrga_plan, wrga_report);

    let lib_path = if let Some(out) = output {
        out
    } else {
        nsl_codegen::linker::default_shared_lib_path(file)
    };

    // M62 Task 6: propagate @export symbols to the linker on MSVC.
    let export_symbols: Vec<String> = exports_slot
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|v| v.iter().map(|e| e.symbol_name.clone()).collect()))
        .unwrap_or_default();
    // Sibling packed-array dispatch wrappers (`<name>__nsl_dispatch`); see
    // single-file path above for rationale.
    let dispatch_symbols: Vec<String> = export_symbols
        .iter()
        .map(|s| format!("{}__nsl_dispatch", s))
        .collect();
    // M62 Task 9: mirror the single-file path and re-export runtime lifecycle
    // symbols so ctypes callers can load weights + call exports through the
    // generated DLL without loading a separate runtime DLL.
    // `mut` is only used when --features onnx-rt-op is on; harmless otherwise.
    #[allow(unused_mut)]
    let mut runtime_exports: Vec<&'static str> = vec![
        "nsl_model_create",
        "nsl_model_create_with_lib",
        "nsl_model_destroy",
        "nsl_model_forward",
        "nsl_model_forward_grad",
        "nsl_model_backward",
        "nsl_grad_context_destroy",
        "nsl_model_call",
        "nsl_model_call_dlpack",
        "nsl_model_export_count",
        "nsl_model_lookup_function",
        "nsl_model_get_weight_ptrs",
        "nsl_model_get_num_weights",
        "nsl_model_num_weights",
        "nsl_get_last_error",
        "nsl_clear_error",
        "nsl_set_error_cstr",
        "nsl_desc_to_tensor",
        "nsl_tensor_to_desc_ffi",
        "nsl_tensor_free",
        "nsl_get_num_exports",
        "nsl_get_export_name",
        "nsl_dispatch_apply_result",
        "nsl_dl_path_for_fn_addr",
        "nsl_free_cstr",
    ];
    // M62b Spec C — mirror single-file path: surface `RegisterCustomOps` for the
    // MSVC linker when nsl-cli is built with --features onnx-rt-op.
    #[cfg(feature = "onnx-rt-op")]
    runtime_exports.push("RegisterCustomOps");
    let mut export_refs: Vec<&str> = export_symbols.iter().map(|s| s.as_str()).collect();
    export_refs.extend(dispatch_symbols.iter().map(|s| s.as_str()));
    export_refs.extend_from_slice(&runtime_exports);

    match nsl_codegen::linker::link_shared_with_exports(&obj_files, &lib_path, &export_refs) {
        Ok(()) => {
            for obj in &obj_files {
                let _ = std::fs::remove_file(obj);
            }
            println!("Built shared library {}", lib_path.display());
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }

    emit_c_header_if_any(&exports_slot, &lib_path);
}

/// M55: Build with --zk-circuit. Runs normal compilation and then invokes
/// `zk::compile_zk()` for each @zk_proof-decorated function found.
pub(crate) fn run_build_zk(
    file: &PathBuf,
    output: Option<PathBuf>,
    emit_obj: bool,
    dump_ir: bool,
    zk_weights: Option<&std::path::Path>,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(file, options.linear_types_enabled);

    // Task 3 (B.1): forward WRGA decorator configs so `@freeze`/`@adapter`/`@wrga`
    // take effect in codegen on the ZK build path.
    // Task 4 (B.2): if `--wrga-report` is set, fail fast when decorators are
    // present without `--source-ad` (mirroring the single/multi build paths).
    check_wrga_report_preconditions(&analysis, wrga_report, options);
    let mut options = options.clone();
    options.wrga_inputs = Some(crate::pipeline::analysis_to_wrga_inputs(&analysis));
    options.fused_ce_configs = crate::pipeline::analysis_to_fused_ce_configs(&analysis);
    // M62 Task 6: route weight_index_map from semantic analysis into codegen.
    options.weight_index_map = analysis.weight_index_map.clone();
    let options = &options;

    // Task 4 (B.2): use the `_returning_plan` variant so the WRGA plan is
    // observable (and reportable) even if later codegen fails.
    let (bytes_res, zk_proof_fns, zk_results, wrga_plan) =
        nsl_codegen::compile_with_zk_info_returning_plan(
            &parse_result.module,
            &interner,
            &analysis.type_map,
            dump_ir,
            options,
        );

    // Emit the WRGA report (if requested) before reporting any codegen error.
    emit_wrga_report(&wrga_plan, wrga_report);

    let obj_bytes = match bytes_res {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("codegen error: {e}");
            process::exit(1);
        }
    };

    // M55c: Write ZK proof files alongside the binary
    if zk_proof_fns.is_empty() {
        eprintln!("[nsl/zk] no @zk_proof functions found — ZK circuit output skipped");
    } else {
        eprintln!(
            "[nsl/zk] found {} @zk_proof function(s):",
            zk_proof_fns.len()
        );

        if let Some(wpath) = zk_weights {
            eprintln!("[nsl/zk]   weights: {}", wpath.display());
        }

        for (fn_name, result) in &zk_results {
            let report = nsl_codegen::zk::stats::format_stats(&result.stats, fn_name);
            eprint!("{}", report);

            // Write proof file if proof was generated
            if let Some(ref proof) = result.proof {
                let proof_path = file.with_extension(format!("{}.proof", fn_name));
                if let Err(e) = std::fs::write(&proof_path, &proof.data) {
                    eprintln!("[nsl/zk] error writing proof: {e}");
                } else {
                    eprintln!("[nsl/zk]   proof: {} ({} bytes, {} folds)",
                        proof_path.display(), proof.data.len(), proof.num_folds);
                }

                // Write public inputs as JSON
                let pi_path = file.with_extension(format!("{}.public.json", fn_name));
                let pi_entries: Vec<String> = proof.public_inputs.iter()
                    .map(|v| format!("{:?}", v))
                    .collect();
                let pi_json = format!(
                    "{{\"public_inputs\":[{}],\"num_folds\":{}}}",
                    pi_entries.join(","), proof.num_folds
                );
                if let Err(e) = std::fs::write(&pi_path, &pi_json) {
                    eprintln!("[nsl/zk] error writing public inputs: {e}");
                } else {
                    eprintln!("[nsl/zk]   public inputs: {}", pi_path.display());
                }
            }
        }
    }

    // Write normal object / link as usual.
    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| {
            eprintln!("error: invalid input filename '{}'", file.display());
            process::exit(1);
        });
    let obj_path = file.with_file_name(format!("{stem}.o"));

    if let Err(e) = std::fs::write(&obj_path, &obj_bytes) {
        eprintln!("error: could not write object file: {e}");
        process::exit(1);
    }

    if emit_obj {
        println!("Wrote {}", obj_path.display());
        return;
    }

    let exe_path = if let Some(out) = output {
        out
    } else {
        nsl_codegen::linker::default_output_path(file)
    };

    match nsl_codegen::linker::link(&obj_path, &exe_path) {
        Ok(()) => {
            let _ = std::fs::remove_file(&obj_path);
            println!("Built {}", exe_path.display());
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }
}

/// M55c: Handle `nsl zk <subcommand>`.
pub(crate) fn run_zk_cmd(cmd: crate::args::ZkCmd) {
    match cmd {
        crate::args::ZkCmd::Stats { file } => {
            match std::fs::read(&file) {
                Ok(data) => {
                    if data.len() < 12 {
                        eprintln!("[nsl/zk] invalid proof file: too short ({} bytes)", data.len());
                        process::exit(1);
                    }
                    let num_folds = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    let instance_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                    let num_rounds = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
                    println!("Proof stats:");
                    println!("  File:        {}", file.display());
                    println!("  Size:        {} bytes", data.len());
                    println!("  Folds:       {}", num_folds);
                    println!("  Instance:    {} elements", instance_len);
                    println!("  SC rounds:   {}", num_rounds);
                }
                Err(e) => {
                    eprintln!("error reading {}: {e}", file.display());
                    process::exit(1);
                }
            }
        }
        crate::args::ZkCmd::Prove { file, pk: _, input: _, output } => {
            // For the folding backend, proofs are generated during compilation.
            eprintln!("[nsl/zk] For the folding backend, proofs are generated during `nsl build --zk-circuit`.");
            eprintln!("[nsl/zk] The proof file is written alongside the binary as <file>.<fn_name>.proof");
            if let Some(ref out) = output {
                eprintln!("[nsl/zk] Requested output: {}", out.display());
            }
            eprintln!("[nsl/zk] To generate a proof, run: nsl build --zk-circuit {}", file.display());
        }
        crate::args::ZkCmd::Verify { vk: _, proof, public: _ } => {
            match std::fs::read(&proof) {
                Ok(data) => {
                    if data.len() < 12 {
                        eprintln!("INVALID: proof file too short ({} bytes)", data.len());
                        process::exit(1);
                    }

                    let num_folds = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    let instance_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
                    let num_rounds = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);

                    // Reconstruct ZkProof for verification
                    let zk_proof = nsl_codegen::zk::backend::ZkProof {
                        data: data.clone(),
                        num_folds,
                        public_inputs: vec![vec![0u8; 4]; instance_len],
                        public_outputs: Vec::new(),
                    };

                    use nsl_codegen::zk::backend::FoldingBackend;
                    type M31Prover = nsl_codegen::zk::folding::FoldingProver<nsl_codegen::zk::field_m31::Mersenne31Field>;

                    match M31Prover::verify(&zk_proof, &[]) {
                        Ok(true) => {
                            println!("VERIFIED: proof is valid ({} folds, {} sumcheck rounds)",
                                num_folds, num_rounds);
                        }
                        Ok(false) => {
                            println!("INVALID: proof verification failed");
                            process::exit(1);
                        }
                        Err(e) => {
                            eprintln!("VERIFICATION ERROR: {e}");
                            process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("error reading {}: {e}", proof.display());
                    process::exit(1);
                }
            }
        }
    }
}

pub(crate) fn run_build_standalone(
    file: &std::path::Path,
    output: Option<&std::path::Path>,
    weights: &std::path::Path,
    embed_mode: crate::standalone::EmbedMode,
    embed_threshold: u64,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    // 1. Read weights from safetensors
    let tensors = crate::standalone::read_safetensors(weights).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        process::exit(1);
    });

    // 2. Serialize to .nslweights format
    let nslweights_data = crate::standalone::serialize_nslweights(&tensors);

    // 3. Decide embed vs sidecar
    let embedded = match embed_mode {
        crate::standalone::EmbedMode::Always => true,
        crate::standalone::EmbedMode::Never => false,
        crate::standalone::EmbedMode::Auto => (nslweights_data.len() as u64) <= embed_threshold,
    };

    // 4. Run frontend (lex, parse, semantic analysis)
    let file_pb = file.to_path_buf();
    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(&file_pb, options.linear_types_enabled);

    // Task 3 (B.1): forward WRGA decorator configs so `@freeze`/`@adapter`/`@wrga`
    // take effect in codegen on the standalone build path.
    // Task 4 (B.2): if `--wrga-report` is set, fail fast when decorators are
    // present without `--source-ad`.
    check_wrga_report_preconditions(&analysis, wrga_report, options);
    let mut options = options.clone();
    options.wrga_inputs = Some(crate::pipeline::analysis_to_wrga_inputs(&analysis));
    options.fused_ce_configs = crate::pipeline::analysis_to_fused_ce_configs(&analysis);
    // M62 Task 6: route weight_index_map from semantic analysis into codegen.
    options.weight_index_map = analysis.weight_index_map.clone();
    let options = &options;

    // 5. Determine output path
    let output_path = if let Some(out) = output {
        out.to_path_buf()
    } else {
        nsl_codegen::linker::default_output_path(file)
    };

    let sidecar_path = output_path.with_extension("nslweights");

    // Pass only the filename (not full path) to codegen — the runtime resolves
    // the sidecar relative to the executable, so absolute paths would break
    // portability when the binary is moved.
    let sidecar_name = sidecar_path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "model.nslweights".to_string());

    let config = nsl_codegen::StandaloneConfig {
        embedded,
        sidecar_path: sidecar_name,
    };

    // 6. Compile with standalone config.
    // Task 4 (B.2): use the `_returning_plan` variant so the WRGA plan is
    // observable (and reportable) even if later codegen fails.
    let (bytes_res, wrga_plan) = nsl_codegen::compile_standalone_returning_plan(
        &parse_result.module,
        &interner,
        &analysis.type_map,
        config,
        false,
        options,
    );
    emit_wrga_report(&wrga_plan, wrga_report);
    let obj_bytes = bytes_res.unwrap_or_else(|e| {
        eprintln!("codegen error: {e}");
        process::exit(1);
    });

    // 7. Write main object file
    let temp_dir = std::env::temp_dir().join(format!("nsl_standalone_{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("error: could not create temp dir: {e}");
        process::exit(1);
    }

    let main_obj_path = temp_dir.join("main.o");
    if let Err(e) = std::fs::write(&main_obj_path, &obj_bytes) {
        eprintln!("error: could not write object file: {e}");
        process::exit(1);
    }

    let mut obj_paths: Vec<PathBuf> = vec![main_obj_path];

    // 8. Handle embedded weights or sidecar
    if embedded {
        // Create weight object containing the nslweights data
        let weight_obj_bytes = nsl_codegen::create_weight_object(&nslweights_data).unwrap_or_else(|e| {
            eprintln!("error: could not create weight object: {e}");
            process::exit(1);
        });
        let weight_obj_path = temp_dir.join("weights.o");
        if let Err(e) = std::fs::write(&weight_obj_path, &weight_obj_bytes) {
            eprintln!("error: could not write weight object file: {e}");
            process::exit(1);
        }
        obj_paths.push(weight_obj_path);
    } else {
        // Write sidecar .nslweights file
        crate::standalone::write_nslweights_sidecar_raw(&nslweights_data, &sidecar_path).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            process::exit(1);
        });
    }

    // 9. Link all objects
    match nsl_codegen::linker::link_multi(&obj_paths, &output_path) {
        Ok(()) => {
            // 10. Clean up temp object files
            for obj in &obj_paths {
                let _ = std::fs::remove_file(obj);
            }
            let _ = std::fs::remove_dir(&temp_dir);

            println!("Built {} (standalone{})", output_path.display(),
                if embedded { ", weights embedded" } else { ", sidecar weights" });
            if !embedded {
                println!("  Sidecar: {}", sidecar_path.display());
            }
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }
}

pub(crate) fn run_build_inner(
    file: &PathBuf,
    output: Option<PathBuf>,
    emit_obj: bool,
    dump_ir: bool,
    quiet: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    if needs_multi_file(file) {
        run_build_multi(file, output, emit_obj, dump_ir, quiet, options, wrga_report);
    } else {
        run_build_single(file, output, emit_obj, dump_ir, quiet, options, wrga_report);
    }
}

/// Single-file build (backward compatible, fast path).
fn run_build_single(
    file: &PathBuf,
    output: Option<PathBuf>,
    emit_obj: bool,
    dump_ir: bool,
    quiet: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(file, options.linear_types_enabled);

    // Task 3 (B.1): fail fast if --wrga-report is used without --source-ad
    // when decorators are present — the old silent-notice behaviour was
    // confusing.
    check_wrga_report_preconditions(&analysis, wrga_report, options);

    // Task 1 (WRGA bridge): forward decorator configs captured by nsl-semantic.
    let mut options = options.clone();
    options.wrga_inputs = Some(crate::pipeline::analysis_to_wrga_inputs(&analysis));
    options.fused_ce_configs = crate::pipeline::analysis_to_fused_ce_configs(&analysis);
    // M62 Task 6: route weight_index_map from semantic analysis into codegen so
    // compile_export_model_methods can resolve self.<field> → weight-array index.
    options.weight_index_map = analysis.weight_index_map.clone();
    let options = &options;

    // M45: Run compile-time NaN risk analysis before codegen if --nan-analysis is set.
    if options.nan_analysis {
        let mut analyzer = nsl_semantic::nan_analysis::NanAnalyzer::new();
        analyzer.analyze_module(&parse_result.module, &interner);
        if analyzer.diagnostics.is_empty() {
            eprintln!("note: --nan-analysis: no NaN/Inf risks detected");
        } else {
            eprintln!(
                "note: --nan-analysis: {} warning(s) detected",
                analyzer.diagnostics.len()
            );
            let mut sm = nsl_errors::SourceMap::new();
            let src = std::fs::read_to_string(file).unwrap_or_default();
            sm.add_file(file.display().to_string(), src);
            for diag in &analyzer.diagnostics {
                sm.emit_diagnostic(diag);
            }
        }
    }

    // Codegen
    let (obj_bytes, wrga_plan) = match nsl_codegen::compile_returning_plan(
        &parse_result.module,
        &interner,
        &analysis.type_map,
        dump_ir,
        options,
    ) {
        Ok((bytes, plan)) => (bytes, plan),
        Err(e) => {
            eprintln!("codegen error: {e}");
            process::exit(1);
        }
    };

    // WRGA Milestone B.1: emit `WrgaPlan::render_report()` if --wrga-report was set.
    // Also offer the plan to the CLI-side capture slot (`--wrga-compare`).
    capture_wrga_plan_if_armed(&wrga_plan);
    if let Some(report_path) = wrga_report {
        match &wrga_plan {
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

    // Determine output paths
    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| {
            eprintln!("error: invalid input filename '{}'", file.display());
            process::exit(1);
        });
    let obj_path = file.with_file_name(format!("{stem}.o"));

    // Write object file
    if let Err(e) = std::fs::write(&obj_path, &obj_bytes) {
        eprintln!("error: could not write object file: {e}");
        process::exit(1);
    }

    if emit_obj {
        if !quiet { println!("Wrote {}", obj_path.display()); }
        return;
    }

    // Link
    let exe_path = if let Some(out) = output {
        out
    } else {
        nsl_codegen::linker::default_output_path(file)
    };

    match nsl_codegen::linker::link(&obj_path, &exe_path) {
        Ok(()) => {
            // Clean up .o file after successful link
            let _ = std::fs::remove_file(&obj_path);
            if !quiet { println!("Built {}", exe_path.display()); }
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }
}

/// Multi-file build with module system.
fn run_build_multi(
    file: &std::path::Path,
    output: Option<PathBuf>,
    emit_obj: bool,
    dump_ir: bool,
    quiet: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    let mut entry_wrga_plan: Option<nsl_codegen::wrga::WrgaPlan> = None;
    let mut source_map = SourceMap::new();
    let mut interner = Interner::new();

    // Load and analyze all modules
    let graph = match crate::loader::load_all_modules(file, &mut source_map, &mut interner) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    let temp_dir = std::env::temp_dir().join(format!("nsl_build_{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("error: could not create temp dir: {e}");
        process::exit(1);
    }

    let mut obj_files: Vec<PathBuf> = Vec::new();

    // Compile each module in dependency order
    for path in &graph.dep_order {
        let mod_data = &graph.modules[path];
        let is_entry = *path == graph.entry;

        let obj_bytes = if is_entry {
            // Entry module: import functions from all dependencies
            let mut imported_fns = Vec::new();
            let mut imported_struct_layouts: HashMap<String, nsl_codegen::context::StructLayout> = HashMap::new();
            let mut imported_model_names = std::collections::HashSet::new();
            let mut imported_enum_variants = HashMap::new();
            let mut imported_enum_defs = HashMap::new();
            let mut imported_model_method_bodies: HashMap<String, HashMap<String, nsl_ast::decl::FnDef>> = HashMap::new();
            let mut imported_model_field_types: HashMap<String, HashMap<String, String>> = HashMap::new();

            // Collect imports from ALL dependency modules (not just direct deps)
            for dep_path in &graph.dep_order {
                if dep_path == &graph.entry {
                    continue;
                }
                let dep_data = &graph.modules[dep_path];

                // Build function signatures and struct layouts using a temporary compiler
                let mut temp_compiler = match nsl_codegen::compiler::Compiler::new(&interner, &dep_data.type_map, &nsl_codegen::CompileOptions::default()) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("codegen error: {e}");
                        process::exit(1);
                    }
                };

                for stmt in &dep_data.ast.stmts {
                    if let nsl_ast::stmt::StmtKind::FnDef(fn_def) = &stmt.kind {
                        let raw_name = interner.resolve(fn_def.name.0).unwrap_or("<unknown>").to_string();
                        let mangled_name = crate::mangling::mangle(&dep_data.module_prefix, &raw_name);
                        let sig = temp_compiler.build_fn_signature(fn_def);
                        imported_fns.push((raw_name, mangled_name, sig));
                    }
                }

                // Inject previously-collected struct layouts from earlier deps so that
                // collect_models can resolve sub-model field types (e.g., GQA referencing
                // RotaryEmbedding which was collected from the rope module earlier).
                for (name, layout) in &imported_struct_layouts {
                    temp_compiler.types.struct_layouts.insert(name.clone(), layout.clone());
                }

                // Extract struct layouts from dependency (structs first, then models which may reference structs)
                if let Err(e) = temp_compiler.collect_structs(&dep_data.ast.stmts) {
                    eprintln!("codegen error collecting structs from '{}': {e}", dep_path.display());
                    process::exit(1);
                }
                if let Err(e) = temp_compiler.collect_models(&dep_data.ast.stmts) {
                    eprintln!("codegen error collecting models from '{}': {e}", dep_path.display());
                    process::exit(1);
                }

                // Import model constructor and method signatures
                let model_sigs = temp_compiler.build_model_signatures(&dep_data.ast.stmts);
                imported_fns.extend(model_sigs);

                // Collect model names from dep (so entry module won't generate struct ctors for them)
                for stmt in &dep_data.ast.stmts {
                    if let nsl_ast::stmt::StmtKind::ModelDef(md) = &stmt.kind {
                        let model_name = interner.resolve(md.name.0).unwrap_or("<unknown>").to_string();
                        imported_model_names.insert(model_name);
                    }
                }

                for (name, layout) in temp_compiler.types.struct_layouts.drain() {
                    imported_struct_layouts.insert(name, layout);
                }

                // Propagate model field types from temp compiler (populated by collect_models)
                for (name, fields) in temp_compiler.models.model_field_types.drain() {
                    imported_model_field_types.entry(name).or_insert(fields);
                }

                // Extract model method bodies directly from AST (not from temp compiler,
                // because model_method_bodies is only populated by declare_user_functions
                // which is not called on temp compilers).
                for stmt in &dep_data.ast.stmts {
                    if let nsl_ast::stmt::StmtKind::ModelDef(md) = &stmt.kind {
                        let model_name = interner.resolve(md.name.0).unwrap_or("<unknown>").to_string();
                        let mut body_map = HashMap::new();
                        for member in &md.members {
                            if let nsl_ast::decl::ModelMember::Method(fn_def, _) = member {
                                let method_name = interner.resolve(fn_def.name.0).unwrap_or("<unknown>").to_string();
                                body_map.insert(method_name, fn_def.clone());
                            }
                        }
                        if !body_map.is_empty() {
                            imported_model_method_bodies.entry(model_name).or_insert(body_map);
                        }
                    }
                }

                // Import enum variants/defs
                imported_enum_variants.extend(dep_data.enum_variants.clone());
                imported_enum_defs.extend(dep_data.enum_defs.clone());
            }

            // WRGA Milestone B.1: forward entry-module decorator configs to codegen.
            // Task 3: fail fast if --wrga-report is used without --source-ad.
            if wrga_report.is_some() && !options.source_ad {
                let has_wrga_decorators = !mod_data.wrga_configs.is_empty()
                    || !mod_data.freeze_configs.is_empty()
                    || !mod_data.adapter_configs.is_empty();
                if has_wrga_decorators {
                    eprintln!(
                        "nsl: --wrga-report requires --source-ad when WRGA decorators are present; re-run with --source-ad"
                    );
                    process::exit(2);
                }
            }
            let mut entry_options = options.clone();
            entry_options.wrga_inputs = Some(crate::pipeline::module_data_to_wrga_inputs(mod_data));
            entry_options.fused_ce_configs = crate::pipeline::module_data_to_fused_ce_configs(mod_data);
            let entry_options = &entry_options;

            match nsl_codegen::compile_entry_returning_plan(
                &mod_data.ast,
                &interner,
                &mod_data.type_map,
                &imported_fns,
                imported_struct_layouts,
                imported_model_names,
                imported_enum_variants,
                imported_enum_defs,
                imported_model_method_bodies,
                imported_model_field_types,
                dump_ir,
                entry_options,
            ) {
                Ok((bytes, plan)) => {
                    entry_wrga_plan = plan;
                    bytes
                }
                Err(e) => {
                    eprintln!("codegen error in '{}': {e}", path.display());
                    process::exit(1);
                }
            }
        } else {
            // Library module: export all functions.
            // If this module has dependencies (imports from other modules), inject their symbols.
            let mut lib_imported_fns = Vec::new();
            let mut lib_struct_layouts: HashMap<String, nsl_codegen::context::StructLayout> = HashMap::new();
            let mut lib_model_names = std::collections::HashSet::new();

            // Check if this module has any imports
            let has_imports = mod_data.ast.stmts.iter().any(|s| {
                matches!(s.kind, nsl_ast::stmt::StmtKind::FromImport(_) | nsl_ast::stmt::StmtKind::Import(_))
            });

            if has_imports {
                // Collect symbols from all transitive deps of this module
                for dep_path in &graph.dep_order {
                    if dep_path == path || dep_path == &graph.entry {
                        continue;
                    }
                    // Only include deps that are before this module in dep_order
                    // (i.e., deps of this module, not modules that depend on it)
                    let dep_idx = graph.dep_order.iter().position(|p| p == dep_path).unwrap_or(usize::MAX);
                    let cur_idx = graph.dep_order.iter().position(|p| p == path).unwrap_or(usize::MAX);
                    if dep_idx >= cur_idx {
                        continue;
                    }

                    let dep_data = &graph.modules[dep_path];
                    let mut temp_compiler = match nsl_codegen::compiler::Compiler::new(&interner, &dep_data.type_map, &nsl_codegen::CompileOptions::default()) {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("codegen error: {e}");
                            process::exit(1);
                        }
                    };

                    for stmt in &dep_data.ast.stmts {
                        if let nsl_ast::stmt::StmtKind::FnDef(fn_def) = &stmt.kind {
                            let raw_name = interner.resolve(fn_def.name.0).unwrap_or("<unknown>").to_string();
                            let mangled_name = crate::mangling::mangle(&dep_data.module_prefix, &raw_name);
                            let sig = temp_compiler.build_fn_signature(fn_def);
                            lib_imported_fns.push((raw_name, mangled_name, sig));
                        }
                    }

                    // Inject previously-collected struct layouts from earlier deps
                    for (name, layout) in &lib_struct_layouts {
                        temp_compiler.types.struct_layouts.insert(name.clone(), layout.clone());
                    }

                    if let Err(e) = temp_compiler.collect_structs(&dep_data.ast.stmts) {
                        eprintln!("codegen error: {e}");
                        process::exit(1);
                    }
                    if let Err(e) = temp_compiler.collect_models(&dep_data.ast.stmts) {
                        eprintln!("codegen error: {e}");
                        process::exit(1);
                    }

                    let model_sigs = temp_compiler.build_model_signatures(&dep_data.ast.stmts);
                    lib_imported_fns.extend(model_sigs);

                    for stmt in &dep_data.ast.stmts {
                        if let nsl_ast::stmt::StmtKind::ModelDef(md) = &stmt.kind {
                            let model_name = interner.resolve(md.name.0).unwrap_or("<unknown>").to_string();
                            lib_model_names.insert(model_name);
                        }
                    }

                    for (name, layout) in temp_compiler.types.struct_layouts.drain() {
                        lib_struct_layouts.insert(name, layout);
                    }
                }
            }

            match nsl_codegen::compile_module_with_imports(
                &mod_data.ast,
                &interner,
                &mod_data.type_map,
                &mod_data.module_prefix,
                &lib_imported_fns,
                lib_struct_layouts,
                lib_model_names,
                dump_ir,
                options,
            ) {
                Ok(bytes) => bytes,
                Err(e) => {
                    eprintln!("codegen error in '{}': {e}", path.display());
                    process::exit(1);
                }
            }
        };

        // Write .o file — use index to avoid name collisions (e.g., math/utils.nsl vs string/utils.nsl)
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let obj_path = temp_dir.join(format!("{stem}_{}.o", obj_files.len()));

        if let Err(e) = std::fs::write(&obj_path, &obj_bytes) {
            eprintln!("error: could not write object file '{}': {e}", obj_path.display());
            process::exit(1);
        }

        obj_files.push(obj_path);
    }

    // WRGA Milestone B.1: emit `WrgaPlan::render_report()` if --wrga-report was set.
    // Also offer the plan to the CLI-side capture slot (`--wrga-compare`).
    capture_wrga_plan_if_armed(&entry_wrga_plan);
    if let Some(report_path) = wrga_report {
        match &entry_wrga_plan {
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

    if emit_obj {
        if !quiet {
            for obj in &obj_files {
                println!("Wrote {}", obj.display());
            }
        }
        return;
    }

    // Link all .o files
    let exe_path = if let Some(out) = output {
        out
    } else {
        nsl_codegen::linker::default_output_path(file)
    };

    match nsl_codegen::linker::link_multi(&obj_files, &exe_path) {
        Ok(()) => {
            // Clean up .o files
            for obj in &obj_files {
                let _ = std::fs::remove_file(obj);
            }
            if !quiet { println!("Built {}", exe_path.display()); }
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }
}

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
            debug_training,
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
            wggo_weights,
            wggo_importance,
            wggo_prune_fraction,
            devices,
            csha,
            csha_report,
            cpdt,
            cpdt_num_gpus,
            cpdt_intra_bw,
            cpdt_inter_bw,
            cpdt_report,
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

            // M62a: shared_lib flag is threaded through compile_opts and handled
            // in the build path below.

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

            let compile_opts = nsl_codegen::CompileOptions {
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
                debug_training,
                shared_lib,
                wrga_inputs: None,
                fused_ce_configs: Vec::new(),
                wrga_fold_allocations,
                wggo: nsl_codegen::WggoOptions {
                    mode: wggo.clone(),
                    report: wggo_report,
                    weights: wggo_weights.clone(),
                    importance: nsl_codegen::WggoImportance::from(wggo_importance),
                    prune_fraction: wggo_prune_fraction,
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
                cpdt: nsl_codegen::CpdtOptions {
                    mode: cpdt_mode,
                    cluster: cpdt_cluster.clone(),
                    report_requested: cpdt_report,
                    plan_out: cpdt_plan_out.clone(),
                },
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
