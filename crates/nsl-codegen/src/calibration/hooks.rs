use crate::calibration::ctx::CalibCtx;
use crate::calibration::observation::{ObservationSet, ProjectionRef};

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

// ── Arena-layout plan types ───────────────────────────────────────────────────
//
// These are used by hooks that operate over a pre-built retention arena
// (e.g. `WggoGradientHook`).  The arena maps each projection to a byte
// offset and size inside a flat buffer; hooks can query this to decide
// which projections to observe and how to slice the running buffers.

/// Flat byte-offset map of a retention arena produced by the allocator.
/// Each entry is `(projection, byte_offset, nbytes)`.
#[derive(Debug, Clone, Default)]
pub struct ArenaLayout {
    pub entries: Vec<(ProjectionRef, u32, u32)>,
}

/// One hook's per-projection observation: read `rows × channels` f32
/// elements starting at `src_offset` bytes and accumulate into
/// `running_symbol`.
#[derive(Debug, Clone)]
pub struct ObservePlanEntry {
    pub projection: ProjectionRef,
    pub src_offset: u32,
    pub rows: u32,
    pub channels: u32,
    pub running_symbol: String,
}

/// One hook's per-layer finalisation: write the aggregated `channels`-wide
/// running buffer named `running_symbol` into the sidecar under
/// `projection`.
#[derive(Debug, Clone)]
pub struct FinalizePlanEntry {
    pub projection: ProjectionRef,
    pub running_symbol: String,
    pub channels: u32,
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

    /// Optional: hooks that operate over a pre-built retention arena
    /// return the set of per-projection observations they want.
    /// Default: no arena observations.
    fn observe_plan(&self, _arena: &ArenaLayout) -> Vec<ObservePlanEntry> {
        vec![]
    }

    /// Optional: hooks that aggregate arena observations return one
    /// finalisation entry per logical output dimension (e.g. per layer).
    /// Default: no finalization entries.
    fn finalize_plan(&self) -> Vec<FinalizePlanEntry> {
        vec![]
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
