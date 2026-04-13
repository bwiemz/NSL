//! AWQ calibration hook — observes per-channel max|activation| across
//! the calibration set for each registered linear projection.  See
//! design §7.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::calibration::awq_sidecar;
use crate::calibration::ctx::{BufferHandle, CalibCtx};
use crate::calibration::hooks::{CalibrationHook, CalibrationResult};
use crate::calibration::observation::{ObservationSet, ProjectionRef};

// NOTE: spec §7 text says "RefCell-backed state", but the
// `CalibrationHook` trait requires `Send + Sync` so the registry can
// share hooks across threads. `RefCell` is `!Sync`, so we use `Mutex`
// which provides equivalent interior mutability under shared refs.
pub struct AwqCalibrationHook {
    projections: Vec<ProjectionRef>,
    handles: Mutex<BTreeMap<ProjectionRef, BufferHandle>>,
}

impl AwqCalibrationHook {
    pub fn new(projections: Vec<ProjectionRef>) -> Self {
        Self { projections, handles: Mutex::new(BTreeMap::new()) }
    }

    #[cfg(test)]
    pub fn handles_for_test(&self) -> std::sync::MutexGuard<'_, BTreeMap<ProjectionRef, BufferHandle>> {
        self.handles.lock().unwrap()
    }
}

impl CalibrationHook for AwqCalibrationHook {
    fn id(&self) -> &'static str { "awq_activation_scales" }

    fn requires(&self) -> ObservationSet {
        ObservationSet::LinearInputActivations(self.projections.clone())
    }

    fn emit_init(&self, ctx: &mut CalibCtx) {
        let mut handles = self.handles.lock().unwrap();
        for p in &self.projections {
            let shape = ctx.projection_input_shape(p);
            let in_channels = *shape.last().unwrap_or(&1) as usize;
            let h = ctx.alloc_running_vec(&p.0, in_channels);
            handles.insert(p.clone(), h);
        }
    }

    fn emit_per_step(&self, _ctx: &mut CalibCtx) {
        // Production (Task 9): for each (projection, handle), emit IR
        // that reads the scratch buffer and calls
        // ctx.running_max_abs.  In stub mode this is a no-op; tests
        // drive the reduction via ctx.stub_running_max_abs with the
        // handle retrieved from handles_for_test().
    }

    fn emit_finalize(&self, ctx: &mut CalibCtx) -> CalibrationResult {
        let handles = self.handles.lock().unwrap();
        let mut by_projection: BTreeMap<String, Vec<f32>> = BTreeMap::new();
        for p in &self.projections {
            let h = handles.get(p).copied().unwrap_or_else(|| {
                panic!("AwqCalibrationHook: emit_init was not called for {}", p.0)
            });
            let scales = ctx.finalize_read(h);
            if scales.iter().any(|v| !v.is_finite()) {
                return CalibrationResult::Degenerate {
                    reason: format!("projection {} produced non-finite scales (NaN/Inf)", p.0),
                };
            }
            if scales.iter().all(|v| *v == 0.0) {
                return CalibrationResult::Degenerate {
                    reason: format!("projection {} produced all-zero scales (no variance in calibration data?)", p.0),
                };
            }
            by_projection.insert(p.0.clone(), scales);
        }
        CalibrationResult::Ok(awq_sidecar::serialize(&by_projection))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::retention::{RetentionTable, TensorShape};

    fn setup() -> (Vec<ProjectionRef>, RetentionTable) {
        let projs = vec![
            ProjectionRef::new("blocks.0.attn.wq"),
            ProjectionRef::new("blocks.0.attn.wk"),
        ];
        let mut table = RetentionTable::new();
        for p in &projs {
            table.register(p.clone(), TensorShape::new(vec![1, 8, 4]), 4);
        }
        (projs, table)
    }

    #[test]
    fn requires_returns_all_registered_projections() {
        let (projs, _) = setup();
        let hook = AwqCalibrationHook::new(projs.clone());
        match hook.requires() {
            ObservationSet::LinearInputActivations(got) => assert_eq!(got, projs),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn id_is_awq_activation_scales() {
        let (projs, _) = setup();
        let hook = AwqCalibrationHook::new(projs);
        assert_eq!(hook.id(), "awq_activation_scales");
    }

    #[test]
    fn emit_init_populates_handles_via_interior_mutability() {
        let (projs, table) = setup();
        let hook = AwqCalibrationHook::new(projs.clone());
        let mut ctx = CalibCtx::for_tests(&table);
        hook.emit_init(&mut ctx);
        for p in &projs {
            assert!(hook.handles_for_test().contains_key(p),
                "emit_init should have allocated handle for {}", p.0);
        }
    }

    #[test]
    fn finalize_serialises_roundtrip() {
        let (projs, table) = setup();
        let hook = AwqCalibrationHook::new(projs.clone());
        let mut ctx = CalibCtx::for_tests(&table);
        hook.emit_init(&mut ctx);
        // Simulate one sample's worth of observation via the stub
        // helper, using handles populated by emit_init.
        for p in &projs {
            let h = *hook.handles_for_test().get(p).unwrap();
            ctx.stub_running_max_abs(h, &[1.0, -2.0, 3.0, 4.0]);
        }
        let result = hook.emit_finalize(&mut ctx);
        let blob = match result {
            CalibrationResult::Ok(b) => b,
            other => panic!("expected Ok, got {other:?}"),
        };
        let parsed = awq_sidecar::deserialize(&blob).unwrap();
        assert_eq!(parsed.len(), 2);
        assert_eq!(parsed[0].scales, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn finalize_degenerate_on_nan() {
        let (projs, table) = setup();
        let hook = AwqCalibrationHook::new(projs.clone());
        let mut ctx = CalibCtx::for_tests(&table);
        hook.emit_init(&mut ctx);
        for p in &projs {
            let h = *hook.handles_for_test().get(p).unwrap();
            ctx.stub_set_buffer(h, vec![1.0, f32::NAN, 3.0, 4.0]);
        }
        match hook.emit_finalize(&mut ctx) {
            CalibrationResult::Degenerate { reason } => {
                assert!(reason.contains("non-finite"));
            }
            other => panic!("expected Degenerate, got {other:?}"),
        }
    }

    #[test]
    fn finalize_degenerate_on_all_zero() {
        let (projs, table) = setup();
        let hook = AwqCalibrationHook::new(projs.clone());
        let mut ctx = CalibCtx::for_tests(&table);
        hook.emit_init(&mut ctx);
        // Buffers initialised to 0.0 by alloc_running_vec; no updates.
        match hook.emit_finalize(&mut ctx) {
            CalibrationResult::Degenerate { reason } => {
                assert!(reason.contains("all-zero") || reason.contains("no variance"));
            }
            other => panic!("expected Degenerate, got {other:?}"),
        }
    }

    #[test]
    fn end_to_end_without_manual_handle_install() {
        // Full cycle via public API only: new → emit_init → simulate
        // samples → emit_finalize.  No test-only handle installation.
        let (projs, table) = setup();
        let hook = AwqCalibrationHook::new(projs.clone());
        let mut ctx = CalibCtx::for_tests(&table);
        hook.emit_init(&mut ctx);
        for p in &projs {
            let h = *hook.handles_for_test().get(p).unwrap();
            ctx.stub_running_max_abs(h, &[2.0, 3.0, -4.0, 1.0]);
        }
        match hook.emit_finalize(&mut ctx) {
            CalibrationResult::Ok(_) => {}
            other => panic!("expected Ok after full cycle, got {other:?}"),
        }
    }
}
