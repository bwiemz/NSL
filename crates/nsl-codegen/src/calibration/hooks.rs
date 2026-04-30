use crate::calibration::ctx::CalibCtx;
use crate::calibration::observation::{ObservationSet, ProjectionRef};
use crate::calibration::retention::ArenaLayout;

/// Describes a single per-batch reduction to emit inline in `calibration_main`.
#[derive(Debug, Clone)]
pub struct ObservePlanEntry {
    pub projection: ProjectionRef,
    pub src_offset: u32,
    pub rows: u32,
    pub channels: u32,
    pub running_symbol: String,
}

/// A per-projection running buffer to declare as a `.data` global and to
/// serialize at finalize time.  Paired with `ObservePlanEntry`.
#[derive(Debug, Clone)]
pub struct FinalizePlanEntry {
    pub projection: ProjectionRef,
    pub running_symbol: String,
    pub channels: u32,
    /// Number of bytes per accumulator element.  AWQ uses 4 (f32 max-abs
    /// reduction, spec §4.3); WGGO uses 8 (f64 per-head accumulator,
    /// spec §4.6).  Total BSS size = `channels * bytes_per_element`.
    pub bytes_per_element: u8,
}

/// Outcome of a hook's `emit_finalize` call.
#[derive(Debug, Clone)]
pub enum CalibrationResult {
    /// Hook produced a serialised sidecar blob for its key.
    Ok(Vec<u8>),
    /// Hook completed but its data is unusable (NaN, all-zero variance,
    /// insufficient samples).  ALWAYS triggers a hard error regardless
    /// of `--calibrate` mode — see spec §6.1.
    Degenerate { reason: String },
}

// ── CalibrationHook trait ────────────────────────────────────────────────────

/// A compiler pass's participation in calibration.  See the design doc
/// §4 for semantics.
pub trait CalibrationHook: Send + Sync {
    fn id(&self) -> &'static str;
    fn requires(&self) -> ObservationSet;
    fn emit_init(&self, ctx: &mut CalibCtx);
    fn emit_per_step(&self, ctx: &mut CalibCtx);
    fn emit_finalize(&self, ctx: &mut CalibCtx) -> CalibrationResult;

    /// Returns `true` when this hook needs backward gradients.
    ///
    /// Default impl matches on `requires()`'s `ObservationSet` variant:
    /// returns `true` iff the hook declared `BackwardGradients`.
    /// Forward-only hooks (AWQ, IdentityHook) return `false` automatically.
    fn requires_backward(&self) -> bool {
        matches!(self.requires(), ObservationSet::BackwardGradients(_))
    }

    /// Called once per calibration batch, after model_forward has populated
    /// the retention arena.  Default impl is a no-op for hooks that don't
    /// observe per-batch activations (e.g. IdentityHook).
    fn emit_observe_batch(&self, _ctx: &mut CalibCtx, _arena: &ArenaLayout) {}

    /// Describes per-batch reductions to emit inline in `calibration_main`.
    /// Default empty — hooks with no forward-pass observation (e.g. IdentityHook).
    fn observe_plan(&self, _arena: &ArenaLayout) -> Vec<ObservePlanEntry> {
        Vec::new()
    }

    /// Per-projection running buffers to declare as `.data` globals AND to
    /// serialize at finalize time.  Paired with `observe_plan`.
    fn finalize_plan(&self) -> Vec<FinalizePlanEntry> {
        Vec::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct DummyHook;
    impl CalibrationHook for DummyHook {
        fn id(&self) -> &'static str { "dummy" }
        fn requires(&self) -> ObservationSet { ObservationSet::Empty }
        fn emit_init(&self, _ctx: &mut CalibCtx) {}
        fn emit_per_step(&self, _ctx: &mut CalibCtx) {}
        fn emit_finalize(&self, _ctx: &mut CalibCtx) -> CalibrationResult {
            CalibrationResult::Ok(vec![0x42])
        }
    }

    #[test]
    fn hook_trait_is_object_safe_and_returns_bytes() {
        let h: Box<dyn CalibrationHook> = Box::new(DummyHook);
        assert_eq!(h.id(), "dummy");
        let mut ctx = CalibCtx::stub_for_tests();
        match h.emit_finalize(&mut ctx) {
            CalibrationResult::Ok(b) => assert_eq!(b, vec![0x42]),
            other => panic!("expected Ok, got {other:?}"),
        }
    }

    #[test]
    fn requires_backward_default_false_for_empty_hook() {
        let h = DummyHook;
        assert!(
            !h.requires_backward(),
            "DummyHook (ObservationSet::Empty) must not require backward"
        );
    }

    #[test]
    fn degenerate_result_carries_reason() {
        let r = CalibrationResult::Degenerate {
            reason: "all-zero variance".into(),
        };
        match r {
            CalibrationResult::Degenerate { reason } => {
                assert!(reason.contains("variance"))
            }
            _ => panic!("wrong variant"),
        }
    }
}
