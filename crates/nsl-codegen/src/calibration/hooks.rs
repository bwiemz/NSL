use crate::calibration::ctx::CalibCtx;
use crate::calibration::observation::ObservationSet;
use crate::calibration::retention::ArenaLayout;

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

/// A compiler pass's participation in calibration.  See the design doc
/// §4 for semantics.
pub trait CalibrationHook: Send + Sync {
    fn id(&self) -> &'static str;
    fn requires(&self) -> ObservationSet;
    fn emit_init(&self, ctx: &mut CalibCtx);
    fn emit_per_step(&self, ctx: &mut CalibCtx);
    fn emit_finalize(&self, ctx: &mut CalibCtx) -> CalibrationResult;

    /// Called once per calibration batch, after model_forward has populated
    /// the retention arena.  Default impl is a no-op for hooks that don't
    /// observe per-batch activations (e.g. IdentityHook).
    fn emit_observe_batch(&self, _ctx: &mut CalibCtx, _arena: &ArenaLayout) {}
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
