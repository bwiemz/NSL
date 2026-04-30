//! WGGO Phase 2: calibration hook that will retain per-head |g·w| reductions
//! during the compile-time backward pass.
//!
//! In this milestone the hook is structural only — the subprocess backward
//! pass remains blocked on the AWQ retention debug. `observe_plan` and
//! `finalize_plan` return shaped entries so the rest of the pipeline
//! (running buffer declaration, sidecar JSON layout) is exercised, but
//! running buffers stay zero until the backward pass fires for real.

use crate::calibration::{
    ArenaLayout, CalibCtx, CalibrationHook, CalibrationResult, FinalizePlanEntry,
    ObservePlanEntry,
};
use crate::calibration::observation::{ObservationSet, ProjectionRef};
use crate::calibration::discovery::WggoGradTarget;

// ── Symbol naming ─────────────────────────────────────────────────────────────

fn running_symbol_for(layer_key: &str) -> String {
    // Prefix MUST be `__nsl_wggo_grad.` — distinct from AWQ's
    // `__nsl_awq_running.` — then sanitise `.` → `_` in the layer key so
    // the symbol is a valid IR identifier.
    format!("__nsl_wggo_grad.{}", layer_key.replace('.', "_"))
}

// ── Hook ──────────────────────────────────────────────────────────────────────

#[derive(Debug)]
pub struct WggoGradientHook {
    targets: Vec<WggoGradTarget>,
}

impl WggoGradientHook {
    /// Construct a hook from pre-scanned WGGO targets.
    ///
    /// **No `loss_anchor` argument:** the synthetic L2 loss is computed
    /// over `model_forward`'s return value (spec §4.4), so the hook does
    /// not need to know the LM head's projection ref. The previous
    /// `loss_anchor` parameter has been removed.
    pub fn new(targets: Vec<WggoGradTarget>) -> Self {
        Self { targets }
    }

    pub fn targets(&self) -> &[WggoGradTarget] {
        &self.targets
    }
}

impl CalibrationHook for WggoGradientHook {
    fn id(&self) -> &'static str {
        "wggo_head_gradients"
    }

    fn requires(&self) -> ObservationSet {
        // Needs backward gradients for every registered layer.
        use crate::calibration::observation::LayerRef;
        let layers: Vec<LayerRef> = self
            .targets
            .iter()
            .map(|t| LayerRef::new(&t.layer_key))
            .collect();
        ObservationSet::BackwardGradients(layers)
    }

    fn emit_init(&self, ctx: &mut CalibCtx) {
        for target in &self.targets {
            // n_o_heads = d_model / head_dim where d_model = w_o_shape[0]
            // (output dim of the o_proj). Floor to 1 to guard against
            // degenerate fixtures where head_dim > d_model.
            //
            // Belt-and-suspenders: pre-scan rejects head_dim == 0 (discovery.rs),
            // but guard the division here too in case a registry path bypasses pre-scan.
            let n_o_heads = if target.head_dim == 0 {
                1
            } else {
                (target.w_o_shape[0] / target.head_dim).max(1)
            };
            let symbol = running_symbol_for(&target.layer_key);
            // Each head gets one f64 accumulator — 8 bytes per spec §4.6.
            ctx.declare_bss_global(&symbol, n_o_heads * 8);
        }
    }

    fn emit_per_step(&self, _ctx: &mut CalibCtx) {
        // Stub — see emit_init comment.
    }

    fn emit_finalize(&self, _ctx: &mut CalibCtx) -> CalibrationResult {
        // Stub — returns an empty payload so the harness driver records a
        // key for this hook but with no data.  Tests can assert on this
        // until the real reduction lands.
        CalibrationResult::Ok(vec![])
    }

    // ── Arena-layout plan methods ─────────────────────────────────────────

    fn observe_plan(&self, arena: &ArenaLayout) -> Vec<ObservePlanEntry> {
        let mut out = Vec::new();
        for t in &self.targets {
            // n_heads = d_model / head_dim (w_o_shape[0] is d_model);
            // floor to 1 to avoid div-by-zero.
            //
            // Belt-and-suspenders: pre-scan rejects head_dim == 0 (discovery.rs),
            // but guard the division here too in case a registry path bypasses pre-scan.
            let num_heads = if t.head_dim == 0 {
                1
            } else {
                (t.w_o_shape[0] / t.head_dim).max(1)
            };
            for proj in [&t.w_q, &t.w_k, &t.w_v, &t.w_o] {
                if let Some((_, offset, nbytes)) =
                    arena.entries.iter().find(|(p, _, _)| p == proj)
                {
                    // Validate the arena entry is evenly divisible by n_heads.
                    let total_f32 = *nbytes / 4;
                    if total_f32 % num_heads != 0 {
                        continue;
                    }
                    let rows = total_f32 / num_heads;
                    out.push(ObservePlanEntry {
                        projection: proj.clone(),
                        src_offset: *offset,
                        rows,
                        channels: num_heads,
                        running_symbol: running_symbol_for(&t.layer_key),
                    });
                }
                // Projections absent from the arena are silently skipped —
                // the retention-arena builder may or may not include all four
                // projections per layer depending on how the retention pass evolves.
            }
        }
        out
    }

    fn finalize_plan(&self) -> Vec<FinalizePlanEntry> {
        // One entry per layer (not per projection): all four projections'
        // gradients accumulate into a single per-layer per-head running buffer.
        self.targets
            .iter()
            .map(|t| {
                // Belt-and-suspenders: pre-scan rejects head_dim == 0 (discovery.rs),
                // but guard the division here too in case a registry path bypasses pre-scan.
                let num_heads = if t.head_dim == 0 {
                    1
                } else {
                    (t.w_o_shape[0] / t.head_dim).max(1)
                };
                FinalizePlanEntry {
                    projection: ProjectionRef(format!("{}.__wggo_grad", t.layer_key)),
                    running_symbol: running_symbol_for(&t.layer_key),
                    channels: num_heads,
                }
            })
            .collect()
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calibration::retention::ArenaLayout;
    use crate::calibration::discovery::WggoGradTarget;

    fn proj(name: &str) -> ProjectionRef {
        ProjectionRef(name.to_string())
    }

    /// Build a `WggoGradTarget` fixture.
    ///
    /// `head_dim` and `w_o_shape` are explicit so that tests can control
    /// the head-count calculation independently.  The other three
    /// projection shapes default to the same value as `w_o_shape` — fine
    /// for observe/finalize plan tests that only care about `w_o_shape`.
    fn fixture_target(layer_key: &str, head_dim: u32, w_o_shape: [u32; 2]) -> WggoGradTarget {
        WggoGradTarget {
            layer_key: layer_key.to_string(),
            class_name: "Attention".to_string(),
            head_dim,
            w_q: proj(&format!("{layer_key}.w_q")),
            w_k: proj(&format!("{layer_key}.w_k")),
            w_v: proj(&format!("{layer_key}.w_v")),
            w_o: proj(&format!("{layer_key}.w_o")),
            w_q_shape: w_o_shape,
            w_k_shape: w_o_shape,
            w_v_shape: w_o_shape,
            w_o_shape,
        }
    }

    /// Convenience wrapper matching the old zero-argument style:
    /// 4 heads, d_model=256, head_dim=64 (same as `LayerShape::default_for_test_4heads`).
    fn fixture_target_4heads(layer_key: &str) -> WggoGradTarget {
        fixture_target(layer_key, 64, [256, 256])
    }

    // ── observe_plan ─────────────────────────────────────────────────────

    #[test]
    fn observe_plan_matches_per_layer_targets() {
        let t = fixture_target_4heads("model.layers.0");
        // 4 heads × 4 bytes × 1 row = 16 bytes per projection
        let entries = vec![
            (t.w_q.clone(), 0u32, 16u32),
            (t.w_k.clone(), 16u32, 16u32),
            (t.w_v.clone(), 32u32, 16u32),
            (t.w_o.clone(), 48u32, 16u32),
        ];
        let arena = ArenaLayout { entries };
        let hook = WggoGradientHook::new(vec![t]);
        let plan = hook.observe_plan(&arena);
        assert_eq!(plan.len(), 4, "all four projections should produce entries");
        for e in &plan {
            assert_eq!(e.channels, 4, "n_heads should be 4 (256/64)");
            assert_eq!(e.rows, 1, "16 bytes / 4 bytes_per_f32 / 4 heads = 1 row");
            assert_eq!(
                e.running_symbol,
                "__nsl_wggo_grad.model_layers_0",
                "dots in layer_key must be replaced by underscores"
            );
        }
        // Verify offsets are preserved correctly.
        assert_eq!(plan[0].src_offset, 0);
        assert_eq!(plan[1].src_offset, 16);
        assert_eq!(plan[2].src_offset, 32);
        assert_eq!(plan[3].src_offset, 48);
    }

    #[test]
    fn observe_plan_skips_projections_missing_from_arena() {
        let t = fixture_target_4heads("model.layers.0");
        // Only w_q in arena; w_k, w_v, w_o absent.
        let entries = vec![(t.w_q.clone(), 0u32, 16u32)];
        let arena = ArenaLayout { entries };
        let hook = WggoGradientHook::new(vec![t]);
        let plan = hook.observe_plan(&arena);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].projection, proj("model.layers.0.w_q"));
    }

    #[test]
    fn observe_plan_empty_arena_produces_no_entries() {
        let t = fixture_target_4heads("model.layers.0");
        let arena = ArenaLayout { entries: vec![] };
        let hook = WggoGradientHook::new(vec![t]);
        assert!(hook.observe_plan(&arena).is_empty());
    }

    #[test]
    fn observe_plan_multiple_layers() {
        let t0 = fixture_target_4heads("model.layers.0");
        let t1 = fixture_target_4heads("model.layers.1");
        let entries = vec![
            (t0.w_q.clone(), 0u32, 16u32),
            (t1.w_q.clone(), 16u32, 16u32),
        ];
        let arena = ArenaLayout { entries };
        let hook = WggoGradientHook::new(vec![t0, t1]);
        let plan = hook.observe_plan(&arena);
        // One projection per layer (only w_q present in arena for each).
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].running_symbol, "__nsl_wggo_grad.model_layers_0");
        assert_eq!(plan[1].running_symbol, "__nsl_wggo_grad.model_layers_1");
    }

    // ── finalize_plan ────────────────────────────────────────────────────

    #[test]
    fn finalize_plan_one_entry_per_layer() {
        let hook = WggoGradientHook::new(
            vec![
                fixture_target_4heads("model.layers.0"),
                fixture_target_4heads("model.layers.1"),
            ],
        );
        let plan = hook.finalize_plan();
        assert_eq!(plan.len(), 2);
        assert_eq!(plan[0].channels, 4);
        assert_eq!(plan[0].running_symbol, "__nsl_wggo_grad.model_layers_0");
        assert_eq!(plan[0].projection.0, "model.layers.0.__wggo_grad");
        assert_eq!(plan[1].channels, 4);
        assert_eq!(plan[1].running_symbol, "__nsl_wggo_grad.model_layers_1");
        assert_eq!(plan[1].projection.0, "model.layers.1.__wggo_grad");
    }

    #[test]
    fn finalize_plan_empty_targets() {
        let hook = WggoGradientHook::new(vec![]);
        assert!(hook.finalize_plan().is_empty());
    }

    // ── CalibrationHook trait compliance ─────────────────────────────────

    #[test]
    fn hook_id_is_stable() {
        let hook = WggoGradientHook::new(vec![]);
        assert_eq!(hook.id(), "wggo_head_gradients");
    }

    #[test]
    fn requires_backward_returns_true_for_wggo_hook() {
        let hook = WggoGradientHook::new(vec![fixture_target_4heads("model.layers.0")]);
        assert!(
            hook.requires_backward(),
            "WggoGradientHook must return true for requires_backward() \
             (it declares BackwardGradients)"
        );
    }

    #[test]
    fn requires_returns_backward_gradients_for_all_layers() {
        let hook = WggoGradientHook::new(
            vec![
                fixture_target_4heads("model.layers.0"),
                fixture_target_4heads("model.layers.1"),
            ],
        );
        match hook.requires() {
            ObservationSet::BackwardGradients(refs) => {
                assert_eq!(refs.len(), 2);
                assert_eq!(refs[0].0, "model.layers.0");
                assert_eq!(refs[1].0, "model.layers.1");
            }
            other => panic!("expected BackwardGradients, got {other:?}"),
        }
    }

    #[test]
    fn emit_finalize_stub_returns_empty_ok() {
        let hook = WggoGradientHook::new(vec![]);
        let mut ctx = crate::calibration::ctx::CalibCtx::stub_for_tests();
        match hook.emit_finalize(&mut ctx) {
            CalibrationResult::Ok(b) => assert!(b.is_empty()),
            other => panic!("expected Ok(empty), got {other:?}"),
        }
    }

    #[test]
    fn hook_is_object_safe_via_box() {
        let hook: Box<dyn CalibrationHook> = Box::new(WggoGradientHook::new(
            vec![fixture_target_4heads("model.layers.0")],
        ));
        assert_eq!(hook.id(), "wggo_head_gradients");
    }

    #[test]
    fn running_symbol_sanitises_dots() {
        // Indirectly test through finalize_plan.
        let hook = WggoGradientHook::new(
            vec![fixture_target_4heads("a.b.c")],
        );
        let plan = hook.finalize_plan();
        assert_eq!(plan[0].running_symbol, "__nsl_wggo_grad.a_b_c");
    }

    // ── head_dim=0 defensive guard ────────────────────────────────────

    /// Regression: pre-scan rejects head_dim=0 (discovery.rs), but a registry
    /// path could bypass it.  All three division sites must not panic.
    #[test]
    fn head_dim_zero_does_not_panic_in_emit_init() {
        let t = fixture_target("model.layers.0", /*head_dim=*/ 0, /*w_o_shape=*/ [256, 256]);
        let hook = WggoGradientHook::new(vec![t]);
        let mut ctx = crate::calibration::ctx::CalibCtx::stub_for_tests();
        // Must not panic; should emit a 1-head buffer (8 bytes).
        hook.emit_init(&mut ctx);
        let g = ctx.lookup_bss_global("__nsl_wggo_grad.model_layers_0").unwrap();
        assert_eq!(g.size_bytes, 8, "head_dim=0 → 1 head → 8 bytes");
    }

    #[test]
    fn head_dim_zero_does_not_panic_in_observe_plan() {
        let t = fixture_target("model.layers.0", 0, [256, 256]);
        let entries = vec![(t.w_q.clone(), 0u32, 16u32)];
        let arena = ArenaLayout { entries };
        let hook = WggoGradientHook::new(vec![t]);
        // Must not panic; num_heads falls back to 1.
        let plan = hook.observe_plan(&arena);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].channels, 1);
    }

    #[test]
    fn head_dim_zero_does_not_panic_in_finalize_plan() {
        let t = fixture_target("model.layers.0", 0, [256, 256]);
        let hook = WggoGradientHook::new(vec![t]);
        // Must not panic; channels falls back to 1.
        let plan = hook.finalize_plan();
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].channels, 1);
    }

    // ── emit_init ────────────────────────────────────────────────────────

    #[test]
    fn emit_init_declares_per_layer_running_buffers_as_bss_globals() {
        let targets = vec![
            fixture_target("model.layers.0", /*head_dim=*/ 128, /*w_o_shape=*/ [4096, 4096]),
            fixture_target("model.layers.1", 128, [4096, 4096]),
        ];
        let hook = WggoGradientHook::new(targets);
        let mut ctx = crate::calibration::ctx::CalibCtx::stub_for_tests();
        hook.emit_init(&mut ctx);

        // n_o_heads = 4096 / 128 = 32; buffer = 32 × 8 bytes = 256 bytes per layer
        let g0 = ctx.lookup_bss_global("__nsl_wggo_grad.model_layers_0").unwrap();
        assert_eq!(g0.size_bytes, 256);
        let g1 = ctx.lookup_bss_global("__nsl_wggo_grad.model_layers_1").unwrap();
        assert_eq!(g1.size_bytes, 256);
    }
}
