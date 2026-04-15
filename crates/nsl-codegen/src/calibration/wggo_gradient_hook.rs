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
use crate::wggo_cost::LayerShape;

/// Per-layer targets the hook observes. Each layer contributes four
/// projections (`W_q`, `W_k`, `W_v`, `W_o`) to the retention arena; the
/// per-head scalar for each is accumulated into a single per-layer running
/// buffer of size `n_heads`.
#[derive(Debug, Clone)]
pub struct LayerGradTarget {
    pub layer_key: String,
    pub layer_shape: LayerShape,
    pub w_q: ProjectionRef,
    pub w_k: ProjectionRef,
    pub w_v: ProjectionRef,
    pub w_o: ProjectionRef,
}

// ── Discovery error ───────────────────────────────────────────────────────────

#[derive(Debug)]
pub enum WggoAnchorError {
    /// No LM-head candidate found; add `@wggo_loss_anchor` to the model
    /// output projection.
    NoCandidate,
    /// Multiple candidates found; disambiguate with `@wggo_loss_anchor`.
    AmbiguousCandidates(Vec<String>),
}

impl std::fmt::Display for WggoAnchorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoCandidate => write!(
                f,
                "no LM-head candidate found; add @wggo_loss_anchor to the model output projection"
            ),
            Self::AmbiguousCandidates(v) => write!(
                f,
                "multiple LM-head candidates: {:?}; disambiguate with @wggo_loss_anchor",
                v
            ),
        }
    }
}

impl std::error::Error for WggoAnchorError {}

// ── Discovery function ────────────────────────────────────────────────────────

/// Auto-discovers the LM-head loss anchor among the model's projections by
/// matching path suffixes. Exactly one match must exist.
pub fn discover_loss_anchor(
    projections: &[ProjectionRef],
) -> Result<ProjectionRef, WggoAnchorError> {
    const SUFFIXES: &[&str] = &[".lm_head", ".output", ".output_proj", ".final_linear"];
    let hits: Vec<ProjectionRef> = projections
        .iter()
        .filter(|p| SUFFIXES.iter().any(|s| p.0.ends_with(s)))
        .cloned()
        .collect();
    match hits.len() {
        0 => Err(WggoAnchorError::NoCandidate),
        1 => Ok(hits.into_iter().next().unwrap()),
        _ => Err(WggoAnchorError::AmbiguousCandidates(
            hits.into_iter().map(|p| p.0).collect(),
        )),
    }
}

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
    targets: Vec<LayerGradTarget>,
    loss_anchor: ProjectionRef,
}

impl WggoGradientHook {
    pub fn new(targets: Vec<LayerGradTarget>, loss_anchor: ProjectionRef) -> Self {
        Self { targets, loss_anchor }
    }

    pub fn targets(&self) -> &[LayerGradTarget] {
        &self.targets
    }

    pub fn loss_anchor(&self) -> &ProjectionRef {
        &self.loss_anchor
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

    fn emit_init(&self, _ctx: &mut CalibCtx) {
        // Stub — real reduction IR lands in the follow-up plan when the AWQ
        // subprocess retention debug resolves.  Running buffers stay zero.
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
            // n_heads = d_model / head_dim; floor to 1 to avoid div-by-zero.
            let num_heads = (t.layer_shape.d_model / t.layer_shape.head_dim).max(1) as u32;
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
                let num_heads =
                    (t.layer_shape.d_model / t.layer_shape.head_dim).max(1) as u32;
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
    use crate::wggo_cost::LayerShape;

    fn proj(name: &str) -> ProjectionRef {
        ProjectionRef(name.to_string())
    }

    fn fixture_target(layer_key: &str) -> LayerGradTarget {
        LayerGradTarget {
            layer_key: layer_key.to_string(),
            layer_shape: LayerShape::default_for_test_4heads(),
            w_q: proj(&format!("{layer_key}.w_q")),
            w_k: proj(&format!("{layer_key}.w_k")),
            w_v: proj(&format!("{layer_key}.w_v")),
            w_o: proj(&format!("{layer_key}.w_o")),
        }
    }

    // ── discover_loss_anchor ──────────────────────────────────────────────

    #[test]
    fn discover_single_lm_head_match() {
        let ps = vec![
            proj("model.embed"),
            proj("model.lm_head"),
            proj("model.layers.0.up_proj"),
        ];
        let anchor = discover_loss_anchor(&ps).unwrap();
        assert_eq!(anchor.0, "model.lm_head");
    }

    #[test]
    fn discover_output_suffix_matches() {
        let ps = vec![proj("model.output")];
        let anchor = discover_loss_anchor(&ps).unwrap();
        assert_eq!(anchor.0, "model.output");
    }

    #[test]
    fn discover_output_proj_suffix_matches() {
        let ps = vec![proj("model.output_proj")];
        let anchor = discover_loss_anchor(&ps).unwrap();
        assert_eq!(anchor.0, "model.output_proj");
    }

    #[test]
    fn discover_final_linear_suffix_matches() {
        let ps = vec![proj("model.final_linear")];
        let anchor = discover_loss_anchor(&ps).unwrap();
        assert_eq!(anchor.0, "model.final_linear");
    }

    #[test]
    fn discover_zero_matches_errors() {
        let ps = vec![proj("model.foo")];
        assert!(matches!(
            discover_loss_anchor(&ps),
            Err(WggoAnchorError::NoCandidate)
        ));
    }

    #[test]
    fn discover_multiple_matches_errors() {
        let ps = vec![proj("a.lm_head"), proj("b.output_proj")];
        assert!(matches!(
            discover_loss_anchor(&ps),
            Err(WggoAnchorError::AmbiguousCandidates(_))
        ));
    }

    #[test]
    fn discover_empty_list_errors() {
        assert!(matches!(
            discover_loss_anchor(&[]),
            Err(WggoAnchorError::NoCandidate)
        ));
    }

    // ── observe_plan ─────────────────────────────────────────────────────

    #[test]
    fn observe_plan_matches_per_layer_targets() {
        let t = fixture_target("model.layers.0");
        // 4 heads × 4 bytes × 1 row = 16 bytes per projection
        let entries = vec![
            (t.w_q.clone(), 0u32, 16u32),
            (t.w_k.clone(), 16u32, 16u32),
            (t.w_v.clone(), 32u32, 16u32),
            (t.w_o.clone(), 48u32, 16u32),
        ];
        let arena = ArenaLayout { entries };
        let hook = WggoGradientHook::new(vec![t], proj("model.lm_head"));
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
        let t = fixture_target("model.layers.0");
        // Only w_q in arena; w_k, w_v, w_o absent.
        let entries = vec![(t.w_q.clone(), 0u32, 16u32)];
        let arena = ArenaLayout { entries };
        let hook = WggoGradientHook::new(vec![t], proj("model.lm_head"));
        let plan = hook.observe_plan(&arena);
        assert_eq!(plan.len(), 1);
        assert_eq!(plan[0].projection, proj("model.layers.0.w_q"));
    }

    #[test]
    fn observe_plan_empty_arena_produces_no_entries() {
        let t = fixture_target("model.layers.0");
        let arena = ArenaLayout { entries: vec![] };
        let hook = WggoGradientHook::new(vec![t], proj("model.lm_head"));
        assert!(hook.observe_plan(&arena).is_empty());
    }

    #[test]
    fn observe_plan_multiple_layers() {
        let t0 = fixture_target("model.layers.0");
        let t1 = fixture_target("model.layers.1");
        let entries = vec![
            (t0.w_q.clone(), 0u32, 16u32),
            (t1.w_q.clone(), 16u32, 16u32),
        ];
        let arena = ArenaLayout { entries };
        let hook = WggoGradientHook::new(vec![t0, t1], proj("model.lm_head"));
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
                fixture_target("model.layers.0"),
                fixture_target("model.layers.1"),
            ],
            proj("model.lm_head"),
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
        let hook = WggoGradientHook::new(vec![], proj("model.lm_head"));
        assert!(hook.finalize_plan().is_empty());
    }

    // ── CalibrationHook trait compliance ─────────────────────────────────

    #[test]
    fn hook_id_is_stable() {
        let hook = WggoGradientHook::new(vec![], proj("model.lm_head"));
        assert_eq!(hook.id(), "wggo_head_gradients");
    }

    #[test]
    fn requires_returns_backward_gradients_for_all_layers() {
        let hook = WggoGradientHook::new(
            vec![
                fixture_target("model.layers.0"),
                fixture_target("model.layers.1"),
            ],
            proj("model.lm_head"),
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
        let hook = WggoGradientHook::new(vec![], proj("model.lm_head"));
        let mut ctx = crate::calibration::ctx::CalibCtx::stub_for_tests();
        match hook.emit_finalize(&mut ctx) {
            CalibrationResult::Ok(b) => assert!(b.is_empty()),
            other => panic!("expected Ok(empty), got {other:?}"),
        }
    }

    #[test]
    fn hook_is_object_safe_via_box() {
        let hook: Box<dyn CalibrationHook> = Box::new(WggoGradientHook::new(
            vec![fixture_target("model.layers.0")],
            proj("model.lm_head"),
        ));
        assert_eq!(hook.id(), "wggo_head_gradients");
    }

    #[test]
    fn running_symbol_sanitises_dots() {
        // Indirectly test through finalize_plan.
        let hook = WggoGradientHook::new(
            vec![fixture_target("a.b.c")],
            proj("model.lm_head"),
        );
        let plan = hook.finalize_plan();
        assert_eq!(plan[0].running_symbol, "__nsl_wggo_grad.a_b_c");
    }
}
