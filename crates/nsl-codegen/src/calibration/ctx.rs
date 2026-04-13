/// Runtime context passed to `CalibrationHook::emit_*` methods.  Later
/// tasks attach real codegen state; for now this is a struct stub that
/// hook authors can `&mut` but whose fields are only meaningful inside
/// the calibration binary codegen.
#[derive(Debug, Default)]
pub struct CalibCtx {
    /// Sample index in the current per-step call; 0 before the loop
    /// starts.
    pub sample_idx: u32,
    /// Total samples the loop will run.
    pub total_samples: u32,
}

impl CalibCtx {
    /// Construct a no-op context for unit tests that exercise hook
    /// behaviour without running actual codegen.
    pub fn stub_for_tests() -> Self {
        Self { sample_idx: 0, total_samples: 0 }
    }
}
