//! CLI-side thread-local state for WRGA check-mode overrides.
//!
//! These thread-locals and their RAII guards carry the `nsl check
//! --wrga-analyze` / `--wrga-compare` overrides (target, ablation) and the
//! captured `WrgaPlan` into the shared build pipeline without threading new
//! parameters through `run_build_inner`'s callers. Extracted verbatim from the
//! former monolithic `build.rs`; behavior is unchanged.

/// RAII guard that clears `WRGA_TARGET_OVERRIDE` on drop. Holding this guard
/// keeps the thread-local set for the lifetime of the build; dropping it
/// (including on panic) restores `None`.
pub(super) struct WrgaTargetOverrideGuard;

impl WrgaTargetOverrideGuard {
    pub(super) fn set(value: String) -> Self {
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
pub(super) struct WrgaAblationOverrideGuard;

impl WrgaAblationOverrideGuard {
    pub(super) fn set(value: nsl_codegen::wrga::WrgaAblation) -> Self {
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
pub(super) struct WrgaPlanCaptureGuard;

impl WrgaPlanCaptureGuard {
    pub(super) fn arm() -> Self {
        WRGA_PLAN_CAPTURE.with(|c| *c.borrow_mut() = None);
        Self
    }

    /// Extract the captured plan, leaving `None` behind. Returns `None` if no
    /// plan was captured (e.g. compile failed, or no @train block had WRGA
    /// decorators).
    pub(super) fn take() -> Option<nsl_codegen::wrga::WrgaPlan> {
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
pub(super) fn capture_wrga_plan_if_armed(plan: &Option<nsl_codegen::wrga::WrgaPlan>) {
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
            });
        }
    });
    WRGA_ABLATION_OVERRIDE.with(|c| {
        if let Some(abl) = *c.borrow() {
            inputs.ablation = abl;
        }
    });
}
