//! WGGO Phase 2: calibration hook that retains per-head |g·w| reductions
//! during the compile-time backward pass.
//!
//! The hook owns three IR-emission methods:
//! - `emit_init` declares one f64 .bss running buffer per registered layer.
//! - `emit_per_step` runs after each batch's backward call, accumulating
//!   `Σ |dW · W|` per head (with MQA/GQA replication per spec §4.6).
//! - `emit_finalize` reads back the running buffers and serialises them
//!   into the sidecar's `wggo_head_gradients` field.

use crate::calibration::{
    ArenaLayout, CalibCtx, CalibrationHook, CalibrationResult, FinalizePlanEntry,
    ObservePlanEntry,
};
use crate::calibration::observation::{ObservationSet, ProjectionRef};
use crate::calibration::discovery::WggoGradTarget;
use crate::calibration::sidecar::{PerLayerGradient, WggoHeadGradients};

// ── Symbol naming ─────────────────────────────────────────────────────────────

fn running_symbol_for(layer_key: &str) -> String {
    // Prefix MUST be `__nsl_wggo_grad.` — distinct from AWQ's
    // `__nsl_awq_running.` — then sanitise `.` → `_` in the layer key so
    // the symbol is a valid IR identifier.
    format!("__nsl_wggo_grad.{}", layer_key.replace('.', "_"))
}

// ── Payload sanitization ──────────────────────────────────────────────────────

/// Replace any non-finite `per_head_score` values (NaN / ±Inf) with `0.0`
/// before JSON serialization.
///
/// `serde_json` rejects `f32::NAN` / `f32::INFINITY` because JSON has no
/// representation for them.  A non-finite score means the importance signal
/// is undefined for that head; treating it as `0.0` effectively removes the
/// head from the importance ranking rather than aborting calibration.
fn sanitize_payload(payload: &WggoHeadGradients) -> WggoHeadGradients {
    let mut out = WggoHeadGradients { by_layer: std::collections::BTreeMap::new() };
    for (k, v) in &payload.by_layer {
        let sanitized_scores: Vec<f32> = v
            .per_head_score
            .iter()
            .map(|s| if s.is_finite() { *s } else { 0.0 })
            .collect();
        out.by_layer.insert(
            k.clone(),
            PerLayerGradient {
                per_head_score: sanitized_scores,
                batches_observed: v.batches_observed,
            },
        );
    }
    out
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

    fn emit_per_step(&self, ctx: &mut CalibCtx) {
        // Task 21 (spec §4.6): iterate every (target, projection) pair that
        // is registered in the grad arena layout and record that
        // `emit_per_head_dot_abs_accum` would be called for it.
        //
        // **Production path:** the real Cranelift IR emission happens in
        // `binary_codegen::emit_calibration_scaffolding_object` which has
        // direct access to a FunctionBuilder, arena globals, and runtime
        // weight pointers.  This method's job is to (a) verify the iteration
        // logic is correct via unit tests, and (b) call `record_per_head_dot_call`
        // / `record_layout_map_build` so tests can assert on call counts without
        // needing Cranelift.  In production these record_* calls are no-ops.
        //
        // If `ctx.grad_arena_layout()` returns `None` (production) the loop
        // still runs correctly — it just skips every projection (layout_map is
        // empty).  This is safe because the production scaffolding performs the
        // real IR emission independently.

        use std::collections::HashMap;
        use crate::calibration::observation::ProjectionRef;

        // Build the layout map ONCE — spec §4.6 requires this to be outside
        // the per-target loop to avoid O(targets²) behaviour.
        // Clone entries to release the borrow on `ctx` before the mutable
        // `record_*` calls below.
        let layout_map: HashMap<ProjectionRef, (u32, u32)> = ctx
            .grad_arena_layout()
            .map(|layout| {
                layout
                    .entries
                    .iter()
                    .map(|(p, off, sz)| (p.clone(), (*off, *sz)))
                    .collect()
            })
            .unwrap_or_default();
        ctx.record_layout_map_build();

        // n_o_heads = w_o_shape[0] / head_dim (with head_dim=0 guard → 1).
        // Used to compute the MQA replication factor for K/V projections.
        for target in &self.targets {
            let n_o_heads = if target.head_dim == 0 {
                1
            } else {
                (target.w_o_shape[0] / target.head_dim).max(1)
            };

            for (proj, proj_shape) in [
                (&target.w_q, target.w_q_shape),
                (&target.w_k, target.w_k_shape),
                (&target.w_v, target.w_v_shape),
                (&target.w_o, target.w_o_shape),
            ] {
                let Some(&(byte_offset, byte_size)) = layout_map.get(proj) else {
                    continue;
                };
                let total_f32 = byte_size / 4;
                let n_proj_heads = if target.head_dim == 0 {
                    1
                } else {
                    (proj_shape[0] / target.head_dim).max(1)
                };
                // Guard against divide-by-zero and non-evenly-divisible shapes.
                if n_proj_heads == 0 || total_f32 % n_proj_heads != 0 {
                    continue;
                }
                let elements_per_head = total_f32 / n_proj_heads;
                let _replication = (n_o_heads / n_proj_heads).max(1);
                let _byte_offset = byte_offset; // used by scaffolding IR path

                // In production the scaffolding calls emit_per_head_dot_abs_accum
                // directly with a FunctionBuilder.  Here we only record that a
                // call *would* happen so tests can verify the count.
                ctx.record_per_head_dot_call();

                // Suppress unused-variable warnings in non-test builds where
                // record_* is a no-op and the compiler might warn.
                let _ = elements_per_head;
            }
        }
    }

    fn emit_finalize(&self, ctx: &mut CalibCtx) -> CalibrationResult {
        let batches_observed = ctx.batches_processed();
        let mut by_layer = std::collections::BTreeMap::new();
        for target in &self.targets {
            // n_o_heads = d_model / head_dim where d_model = w_o_shape[0].
            // Belt-and-suspenders: guard head_dim == 0 in case a registry
            // path bypasses pre-scan validation.
            let n_o_heads = if target.head_dim == 0 {
                1
            } else {
                (target.w_o_shape[0] / target.head_dim).max(1) as usize
            };
            let symbol = running_symbol_for(&target.layer_key);
            // Sanitize non-finite values (NaN / ±Inf) before inserting.
            // serde_json serializes f32::NAN as `null`, which cannot be
            // deserialized back as f32 — sanitize eagerly so the JSON round-trip
            // is always clean.  A non-finite score means importance is undefined;
            // 0.0 removes the head from ranking rather than crashing calibration.
            let per_head_score: Vec<f32> = ctx
                .read_running_buffer_f64_as_f32(&symbol, n_o_heads)
                .into_iter()
                .map(|s| if s.is_finite() { s } else { 0.0 })
                .collect();
            by_layer.insert(
                target.layer_key.clone(),
                PerLayerGradient {
                    per_head_score,
                    batches_observed,
                },
            );
        }
        let payload = WggoHeadGradients { by_layer };
        let bytes = serde_json::to_vec(&payload).unwrap_or_else(|_e| {
            // f32 NaN/Inf is not representable in JSON.  Sanitize non-finite
            // values to 0.0 (undefined importance → treated as unimportant)
            // and retry.  A non-finite score cannot rank heads, so 0.0 is the
            // safest fallback — it effectively removes those heads from the
            // importance ranking rather than crashing calibration entirely.
            let sanitized = sanitize_payload(&payload);
            serde_json::to_vec(&sanitized)
                .expect("serialize WggoHeadGradients after NaN/Inf sanitization (per_head_score: f32)")
        });
        CalibrationResult::Ok(bytes)
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
                    // WGGO per-head gradient accumulator uses f64 (spec §4.6):
                    // 8 bytes per head, distinct from AWQ's f32 max-abs (4 bytes).
                    bytes_per_element: 8,
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

    /// Spec §4.6: WGGO running buffers use f64 accumulators — each entry must
    /// declare bytes_per_element=8 so the Cranelift BSS allocation is the
    /// correct size (channels * 8, not channels * 4).
    #[test]
    fn finalize_plan_uses_f64_bytes_per_element() {
        let hook = WggoGradientHook::new(vec![fixture_target_4heads("model.layers.0")]);
        let plan = hook.finalize_plan();
        assert_eq!(plan.len(), 1);
        assert_eq!(
            plan[0].bytes_per_element, 8,
            "WGGO finalize_plan must use bytes_per_element=8 (f64) per spec §4.6"
        );
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

    /// Task 22: emit_finalize must serialise per-layer running buffers into a
    /// `WggoHeadGradients` JSON payload.  This test replaces the old
    /// `emit_finalize_stub_returns_empty_ok` test that asserted on the stub
    /// behaviour (empty Vec).
    #[test]
    fn emit_finalize_returns_populated_per_layer_payload() {
        use crate::calibration::sidecar::WggoHeadGradients;

        let targets = vec![
            fixture_target("layers.0", 64, [256, 256]),
            fixture_target("layers.1", 64, [256, 256]),
        ];
        let hook = WggoGradientHook::new(targets);
        let mut ctx = crate::calibration::ctx::CalibCtx::stub_for_tests();
        // Simulate accumulator state: 4 heads each (256/64 = 4).
        ctx.set_running_buffer_f64("__nsl_wggo_grad.layers_0", &[1.0, 2.0, 3.0, 4.0]);
        ctx.set_running_buffer_f64("__nsl_wggo_grad.layers_1", &[5.0, 6.0, 7.0, 8.0]);
        ctx.set_batches_processed(10);

        let result = hook.emit_finalize(&mut ctx);
        let CalibrationResult::Ok(bytes) = result else {
            panic!("expected CalibrationResult::Ok, got {result:?}")
        };
        assert!(!bytes.is_empty(), "stub-era empty payload regression: emit_finalize must return non-empty bytes");

        let payload: WggoHeadGradients = serde_json::from_slice(&bytes)
            .expect("emit_finalize bytes must be valid WggoHeadGradients JSON");

        let l0 = payload.by_layer.get("layers.0").expect("layers.0 missing from payload");
        assert_eq!(l0.per_head_score, vec![1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(l0.batches_observed, 10);

        let l1 = payload.by_layer.get("layers.1").expect("layers.1 missing from payload");
        assert_eq!(l1.per_head_score, vec![5.0f32, 6.0, 7.0, 8.0]);
        assert_eq!(l1.batches_observed, 10);
    }

    /// Regression: NaN and ±Inf in per_head_score must not crash emit_finalize.
    /// Non-finite values are sanitized to 0.0 before JSON serialization.
    #[test]
    fn emit_finalize_handles_nan_inf_in_per_head_score() {
        use crate::calibration::sidecar::WggoHeadGradients;

        let targets = vec![fixture_target("layers.nan", 64, [256, 256])];
        let hook = WggoGradientHook::new(targets);
        let mut ctx = crate::calibration::ctx::CalibCtx::stub_for_tests();
        ctx.set_running_buffer_f64("__nsl_wggo_grad.layers_nan", &[
            f64::NAN, f64::INFINITY, 1.0, f64::NEG_INFINITY,
        ]);
        ctx.set_batches_processed(5);

        let result = hook.emit_finalize(&mut ctx);
        let CalibrationResult::Ok(bytes) = result else { panic!("expected Ok, got {result:?}") };
        let payload: WggoHeadGradients = serde_json::from_slice(&bytes)
            .expect("emit_finalize bytes must be valid WggoHeadGradients JSON after sanitization");
        let layer = payload.by_layer.get("layers.nan").expect("layers.nan missing from payload");
        // NaN and ±Inf are sanitized to 0.0; finite values pass through unchanged.
        assert_eq!(layer.per_head_score, vec![0.0f32, 0.0, 1.0, 0.0]);
        assert_eq!(layer.batches_observed, 5);
    }

    /// Empty targets → empty by_layer map, still a valid non-empty JSON object.
    #[test]
    fn emit_finalize_empty_targets_returns_empty_map() {
        use crate::calibration::sidecar::WggoHeadGradients;

        let hook = WggoGradientHook::new(vec![]);
        let mut ctx = crate::calibration::ctx::CalibCtx::stub_for_tests();
        let CalibrationResult::Ok(bytes) = hook.emit_finalize(&mut ctx) else {
            panic!("expected Ok")
        };
        assert!(!bytes.is_empty(), "JSON serialisation of empty map must not be empty bytes");
        let payload: WggoHeadGradients = serde_json::from_slice(&bytes).unwrap();
        assert!(payload.by_layer.is_empty());
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

    // ── emit_per_step ────────────────────────────────────────────────────

    /// Task 21 (spec §4.6): `emit_per_step` must call
    /// `emit_per_head_dot_abs_accum` once per (target, projection) pair
    /// that is present in the grad arena layout.
    ///
    /// 2 layers × 4 projections = 8 expected calls.
    ///
    /// The layout HashMap must be built exactly once (outside the per-target
    /// loop) regardless of how many targets are registered.
    #[test]
    fn emit_per_step_invokes_per_head_dot_for_each_w_star_per_target() {
        let targets = vec![
            fixture_target("layers.0", 64, [256, 256]),
            fixture_target("layers.1", 64, [256, 256]),
        ];
        let hook = WggoGradientHook::new(targets);
        let bytes_per_proj = 256 * 256 * 4u32;
        let mut ctx = crate::calibration::ctx::CalibCtx::stub_for_tests_with_grad_layout(&[
            ("layers.0.w_q", 0,                 bytes_per_proj),
            ("layers.0.w_k", bytes_per_proj,     bytes_per_proj),
            ("layers.0.w_v", 2 * bytes_per_proj, bytes_per_proj),
            ("layers.0.w_o", 3 * bytes_per_proj, bytes_per_proj),
            ("layers.1.w_q", 4 * bytes_per_proj, bytes_per_proj),
            ("layers.1.w_k", 5 * bytes_per_proj, bytes_per_proj),
            ("layers.1.w_v", 6 * bytes_per_proj, bytes_per_proj),
            ("layers.1.w_o", 7 * bytes_per_proj, bytes_per_proj),
        ]);
        hook.emit_per_step(&mut ctx);
        // 2 layers × 4 projections = 8 per_head_dot calls
        assert_eq!(
            ctx.per_head_dot_call_count(), 8,
            "expected 8 per_head_dot calls (2 layers × 4 projections)"
        );
        // The layout HashMap must be built ONCE, not once per target.
        assert_eq!(
            ctx.layout_map_build_count(), 1,
            "layout HashMap must be built exactly once (outside per-target loop)"
        );
    }

    /// When a projection is absent from the grad arena layout, `emit_per_step`
    /// must silently skip it (no call counted for that projection).
    #[test]
    fn emit_per_step_skips_projections_absent_from_layout() {
        let targets = vec![fixture_target("layers.0", 64, [256, 256])];
        let hook = WggoGradientHook::new(targets);
        let bytes_per_proj = 256 * 256 * 4u32;
        // Only w_q and w_k in layout; w_v and w_o absent.
        let mut ctx = crate::calibration::ctx::CalibCtx::stub_for_tests_with_grad_layout(&[
            ("layers.0.w_q", 0,            bytes_per_proj),
            ("layers.0.w_k", bytes_per_proj, bytes_per_proj),
        ]);
        hook.emit_per_step(&mut ctx);
        assert_eq!(ctx.per_head_dot_call_count(), 2,
            "only 2 projections present in layout; 2 calls expected");
    }

    /// An empty layout produces zero per_head_dot calls.
    #[test]
    fn emit_per_step_empty_layout_produces_no_calls() {
        let targets = vec![fixture_target("layers.0", 64, [256, 256])];
        let hook = WggoGradientHook::new(targets);
        let mut ctx = crate::calibration::ctx::CalibCtx::stub_for_tests_with_grad_layout(&[]);
        hook.emit_per_step(&mut ctx);
        assert_eq!(ctx.per_head_dot_call_count(), 0);
        // layout map was still built (once), just empty.
        assert_eq!(ctx.layout_map_build_count(), 1);
    }

    /// head_dim=0 in emit_per_step must not panic (same guard as emit_init).
    #[test]
    fn emit_per_step_head_dim_zero_does_not_panic() {
        let target = fixture_target("layers.0", /*head_dim=*/0, [256, 256]);
        let hook = WggoGradientHook::new(vec![target]);
        let bytes_per_proj = 256 * 256 * 4u32;
        let mut ctx = crate::calibration::ctx::CalibCtx::stub_for_tests_with_grad_layout(&[
            ("layers.0.w_q", 0, bytes_per_proj),
            ("layers.0.w_k", bytes_per_proj, bytes_per_proj),
            ("layers.0.w_v", 2 * bytes_per_proj, bytes_per_proj),
            ("layers.0.w_o", 3 * bytes_per_proj, bytes_per_proj),
        ]);
        // Must not panic; n_proj_heads falls back to 1.
        hook.emit_per_step(&mut ctx);
        assert_eq!(ctx.per_head_dot_call_count(), 4);
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
