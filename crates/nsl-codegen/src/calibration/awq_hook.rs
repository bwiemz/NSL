//! AWQ calibration hook — observes per-channel max|activation| across
//! the calibration set for each registered linear projection.  See
//! design §7.

use std::collections::BTreeMap;
use std::sync::Mutex;

use crate::calibration::awq_sidecar;
use crate::calibration::ctx::{BufferHandle, CalibCtx};
use crate::calibration::discovery::DiscoveredProjection;
use crate::calibration::hooks::{
    CalibrationHook, CalibrationResult, FinalizePlanEntry, ObservePlanEntry,
};
use crate::calibration::observation::{ObservationSet, ProjectionRef};
use crate::calibration::retention::ArenaLayout;

/// Produces a valid Cranelift data symbol name for a projection's running buffer.
/// Replaces `.` with `_` since Cranelift symbol names can't contain periods.
pub(crate) fn running_symbol_for(projection: &ProjectionRef) -> String {
    format!("__nsl_awq_running.{}", projection.0.replace('.', "_"))
}

// NOTE: spec §7 text says "RefCell-backed state", but the
// `CalibrationHook` trait requires `Send + Sync` so the registry can
// share hooks across threads. `RefCell` is `!Sync`, so we use `Mutex`
// which provides equivalent interior mutability under shared refs.
pub struct AwqCalibrationHook {
    projections: Vec<ProjectionRef>,
    handles: Mutex<BTreeMap<ProjectionRef, BufferHandle>>,
    /// Per-projection in-features count (`weight_shape[1]`).  Populated
    /// from `DiscoveredProjection` data; used by `emit_observe_batch` to
    /// determine the column stride of the arena slice.
    in_features: BTreeMap<ProjectionRef, u32>,
}

impl AwqCalibrationHook {
    /// Construct from a bare list of projection refs (no weight-shape
    /// information).  `emit_observe_batch` will skip any projection whose
    /// `in_features` is unknown; use `from_discovered` when the full
    /// shape context is available.
    pub fn new(projections: Vec<ProjectionRef>) -> Self {
        Self {
            projections,
            handles: Mutex::new(BTreeMap::new()),
            in_features: BTreeMap::new(),
        }
    }

    /// Construct from the output of `discover_awq_projections`, which
    /// carries weight-shape metadata.  `emit_observe_batch` is fully
    /// operational when constructed via this path.
    pub fn from_discovered(projections: &[DiscoveredProjection]) -> Self {
        let refs: Vec<ProjectionRef> = projections.iter().map(|p| p.projection.clone()).collect();
        let in_features = projections
            .iter()
            .map(|p| (p.projection.clone(), p.weight_shape[1]))
            .collect();
        Self {
            projections: refs,
            handles: Mutex::new(BTreeMap::new()),
            in_features,
        }
    }

    /// Test-only handle inspection.  Exposed (no `#[cfg(test)]`) so
    /// integration tests in `tests/` — which compile against the crate
    /// as an external dependency — can drive the stub reduction path.
    /// Not intended for production callers.
    #[doc(hidden)]
    pub fn handles_for_test(
        &self,
    ) -> std::sync::MutexGuard<'_, BTreeMap<ProjectionRef, BufferHandle>> {
        self.handles.lock().unwrap()
    }
}

impl CalibrationHook for AwqCalibrationHook {
    fn id(&self) -> &'static str {
        "awq_activation_scales"
    }

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

    /// Per-batch max-abs reduction over the retention arena.
    ///
    /// For each projection registered in `arena.entries`, computes the
    /// row-wise max|activation| and accumulates it into the running buffer
    /// allocated during `emit_init`.
    ///
    /// - `rows`     = `nbytes / (in_features * 4)` (batch×seq collapsed).
    /// - Projections missing from `self.in_features` or with `in_features == 0`
    ///   are skipped silently (log-and-continue; not a hard error).
    fn emit_observe_batch(&self, ctx: &mut CalibCtx, arena: &ArenaLayout) {
        let handles = self.handles.lock().unwrap();
        for (projection, offset, nbytes) in &arena.entries {
            let Some(&handle) = handles.get(projection) else {
                continue;
            };
            let Some(&in_feat) = self.in_features.get(projection) else {
                continue;
            };
            if in_feat == 0 {
                continue;
            }
            let total_floats = nbytes / 4;
            if total_floats % in_feat != 0 {
                continue;
            }
            let rows = total_floats / in_feat;
            ctx.emit_per_channel_max_abs_update(&projection.0, *offset, rows, in_feat, handle);
        }
    }

    fn observe_plan(&self, arena: &ArenaLayout) -> Vec<ObservePlanEntry> {
        arena
            .entries
            .iter()
            .filter_map(|(proj, offset, nbytes)| {
                let channels = *self.in_features.get(proj)?;
                if channels == 0 {
                    return None;
                }
                let total = nbytes / 4;
                if total % channels != 0 {
                    return None;
                }
                let rows = total / channels;
                Some(ObservePlanEntry {
                    projection: proj.clone(),
                    src_offset: *offset,
                    rows,
                    channels,
                    running_symbol: running_symbol_for(proj),
                })
            })
            .collect()
    }

    fn finalize_plan(&self) -> Vec<FinalizePlanEntry> {
        let mut entries: Vec<FinalizePlanEntry> = self
            .in_features
            .iter()
            .map(|(proj, &channels)| FinalizePlanEntry {
                projection: proj.clone(),
                running_symbol: running_symbol_for(proj),
                channels,
                // AWQ max-abs reduction accumulates f32 values (spec §4.3).
                bytes_per_element: 4,
            })
            .collect();
        entries.sort_by(|a, b| a.projection.0.cmp(&b.projection.0));
        entries
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
                    reason: format!(
                        "projection {} produced all-zero scales (no variance in calibration data?)",
                        p.0
                    ),
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
            assert!(
                hook.handles_for_test().contains_key(p),
                "emit_init should have allocated handle for {}",
                p.0
            );
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

    // ── Task 7 test: emit_observe_batch ────────────────────────────────────────

    #[test]
    fn observe_batch_updates_running_max_abs_per_channel() {
        // Build a hook via from_discovered so in_features is populated.
        let discovered = vec![DiscoveredProjection {
            projection: ProjectionRef::new("model.up"),
            weight_shape: [8, 4], // out=8, in_features=4
        }];
        let hook = AwqCalibrationHook::from_discovered(&discovered);

        // ctx doesn't need a retention table for this test; alloc the
        // running buffer manually by calling emit_init via a stub table
        // that knows about model.up with shape [1, 2, 4] (so in_channels=4).
        let mut table = RetentionTable::new();
        table.register(
            ProjectionRef::new("model.up"),
            TensorShape::new(vec![1, 2, 4]),
            4,
        );
        let mut ctx = CalibCtx::for_tests(&table);
        hook.emit_init(&mut ctx);

        // Build an arena layout: 2 rows × 4 channels = 32 bytes.
        let layout = ArenaLayout {
            entries: vec![(ProjectionRef::new("model.up"), 0, 2 * 4 * 4)],
        };

        // 2 rows (batch*seq=2) × 4 channels.
        // Channel-wise max-abs should end up [1.0, 2.0, 7.0, 4.0].
        ctx.stub_set_arena_buffer(
            "model.up",
            &[
                1.0, 2.0, 3.0, 4.0, // row 0
                -1.0, -2.0, -7.0, -4.0, // row 1
            ],
        );
        hook.emit_observe_batch(&mut ctx, &layout);

        let running = ctx.stub_running_max_abs_named("model.up");
        assert_eq!(running, vec![1.0, 2.0, 7.0, 4.0]);
    }

    #[test]
    fn observe_batch_accumulates_across_multiple_calls() {
        let discovered = vec![DiscoveredProjection {
            projection: ProjectionRef::new("model.down"),
            weight_shape: [4, 3], // in_features=3
        }];
        let hook = AwqCalibrationHook::from_discovered(&discovered);

        let mut table = RetentionTable::new();
        table.register(
            ProjectionRef::new("model.down"),
            TensorShape::new(vec![1, 1, 3]),
            4,
        );
        let mut ctx = CalibCtx::for_tests(&table);
        hook.emit_init(&mut ctx);

        let layout = ArenaLayout {
            entries: vec![(ProjectionRef::new("model.down"), 0, 1 * 3 * 4)],
        };

        // First batch: [1.0, -5.0, 2.0] → running = [1.0, 5.0, 2.0]
        ctx.stub_set_arena_buffer("model.down", &[1.0, -5.0, 2.0]);
        hook.emit_observe_batch(&mut ctx, &layout);

        // Second batch: [-3.0, 4.0, -1.5] → running = [3.0, 5.0, 2.0]
        ctx.stub_set_arena_buffer("model.down", &[-3.0, 4.0, -1.5]);
        hook.emit_observe_batch(&mut ctx, &layout);

        let running = ctx.stub_running_max_abs_named("model.down");
        assert_eq!(running, vec![3.0, 5.0, 2.0]);
    }

    #[test]
    fn awq_observe_plan_has_one_entry_per_projection() {
        use crate::calibration::build_arena_layout;
        use crate::calibration::discovery::DiscoveredProjection;
        use crate::calibration::observation::ProjectionRef;

        let ps = vec![
            DiscoveredProjection {
                projection: ProjectionRef("TinyMLP.up_proj".into()),
                weight_shape: [128, 64],
            },
            DiscoveredProjection {
                projection: ProjectionRef("TinyMLP.down_proj".into()),
                weight_shape: [64, 128],
            },
        ];
        let hook = AwqCalibrationHook::from_discovered(&ps);
        let layout = build_arena_layout(&ps, 8, 4);

        let observe = hook.observe_plan(&layout);
        assert_eq!(observe.len(), 2);

        // Look up entries by projection (order is arena-entry order, which is insertion order).
        let up_entry = observe
            .iter()
            .find(|e| e.projection.0 == "TinyMLP.up_proj")
            .unwrap();
        assert_eq!(up_entry.running_symbol, "__nsl_awq_running.TinyMLP_up_proj");
        assert_eq!(up_entry.channels, 64);
        assert_eq!(up_entry.rows, 32);
        let down_entry = observe
            .iter()
            .find(|e| e.projection.0 == "TinyMLP.down_proj")
            .unwrap();
        assert_eq!(
            down_entry.running_symbol,
            "__nsl_awq_running.TinyMLP_down_proj"
        );
        assert_eq!(down_entry.channels, 128);

        let finalize = hook.finalize_plan();
        assert_eq!(finalize.len(), 2);
        // Finalize entries are sorted by projection path for determinism.
        assert_eq!(finalize[0].projection.0, "TinyMLP.down_proj");
        assert_eq!(finalize[0].channels, 128);
        assert_eq!(
            finalize[0].running_symbol,
            "__nsl_awq_running.TinyMLP_down_proj"
        );
        assert_eq!(finalize[1].projection.0, "TinyMLP.up_proj");
        assert_eq!(finalize[1].channels, 64);
    }

    #[test]
    fn observe_batch_skips_projection_with_unknown_in_features() {
        // Hook built via new() has empty in_features map.
        let proj = ProjectionRef::new("model.x");
        let hook = AwqCalibrationHook::new(vec![proj.clone()]);

        let mut table = RetentionTable::new();
        table.register(proj.clone(), TensorShape::new(vec![1, 2, 8]), 4);
        let mut ctx = CalibCtx::for_tests(&table);
        hook.emit_init(&mut ctx);

        let layout = ArenaLayout {
            entries: vec![(proj.clone(), 0, 2 * 8 * 4)],
        };
        ctx.stub_set_arena_buffer("model.x", &vec![1.0f32; 16]);

        // Should silently skip (no panic).
        hook.emit_observe_batch(&mut ctx, &layout);

        // Running buffer stays all-zero.
        let h = *hook.handles_for_test().get(&proj).unwrap();
        let out = ctx.finalize_read(h);
        assert!(out.iter().all(|v| *v == 0.0));
    }
}
