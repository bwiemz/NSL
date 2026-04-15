//! WGGO — Stage 3: weight analysis (importance score derivation).
//!
//! Reads the model's actual parameter tensors (when available) and
//! produces per-head importance scores that Stage 4 / Stage 5 consume
//! via `LayerIlpConstraints::head_importance` + the existing
//! `importance_ok()` constraint in `wggo_ilp.rs`.
//!
//! The analyzer is structured around a trait so callers can plug in
//! whatever weight source they have — an in-memory `Checkpoint`, a
//! memory-mapped `.nslm` file, or a test-only `HashMap`.
//!
//! **Algorithm (Phase 1, magnitude-based).** For each Q head `h`:
//!
//! ```text
//! head_score[h] = ‖W_q[h]‖₂ + ‖W_o[:, h]‖₂
//!               + ‖W_k[h / group_size]‖₂ + ‖W_v[h / group_size]‖₂
//! ```
//!
//! where `group_size = n_heads / n_kv_heads`.  GQA models (Wk, Wv shaped
//! over `n_kv_heads`) distribute each KV-head score across its
//! corresponding Q-head group.  When a required tensor is absent, the
//! contribution is silently skipped (so a partial checkpoint still
//! yields usable scores).  A layer whose weights are entirely absent
//! falls back to uniform scores and is counted in
//! `WeightAnalysisReport::layers_without_weights`.
//!
//! See `docs/plans/2026-04-13-wggo-weight-analysis-design.md` for the
//! full design, rationale, and Phase-2 plan.

use serde::{Deserialize, Serialize};

use crate::wggo_cost::LayerShape;
use crate::wggo_graph::OptGraph;
use crate::wggo_ilp::{HeadImportance, LayerIlpConstraints};

/// Interface for whatever weight source the caller has.
///
/// `get` returns the flattened f32 buffer for the named parameter; the
/// caller interprets the shape via [`WeightProvider::shape`].  Both
/// methods return `None` for unknown names, allowing a partial
/// checkpoint to pass through without failing the pass.
pub trait WeightProvider: std::fmt::Debug {
    fn get(&self, name: &str) -> Option<&[f32]>;
    fn shape(&self, name: &str) -> Option<&[u64]>;
}

/// Provider that returns `None` for every name — used when the caller
/// has no weights (e.g., un-initialised model, test runs).  The
/// analyzer falls back to uniform per-head scores.
#[derive(Debug, Default, Clone, Copy)]
pub struct NullWeightProvider;

impl WeightProvider for NullWeightProvider {
    fn get(&self, _: &str) -> Option<&[f32]> {
        None
    }
    fn shape(&self, _: &str) -> Option<&[u64]> {
        None
    }
}

/// Importance data for one layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerImportance {
    /// Per-Q-head score; length == `num_heads`.  Normalised so
    /// `Σ head_scores ≈ num_heads` (uniform distribution when all
    /// weights are absent or zero).
    pub head_scores: Vec<f64>,
    /// Default `min_retained_importance` — sum of the top
    /// `num_heads × (1 - default_prune_fraction)` scores.  Callers may
    /// override via `LayerIlpConstraints::min_retained_importance`.
    pub default_min_retained: f64,
}

/// Aggregate report from [`analyze`].
#[derive(Debug, Clone, Default, Serialize)]
pub struct WeightAnalysisReport {
    pub per_layer: Vec<LayerImportance>,
    /// Layers whose weights were entirely absent → uniform fallback.
    pub layers_without_weights: u32,
}

impl WeightAnalysisReport {
    /// Merge analyzer output into an existing parallel vector of ILP
    /// constraints.  Overwrites `head_importance`; only overwrites
    /// `min_retained_importance` when the caller left it at 0.0
    /// (preserves explicit overrides).
    pub fn apply_to(&self, constraints: &mut [LayerIlpConstraints]) {
        for (c, imp) in constraints.iter_mut().zip(self.per_layer.iter()) {
            c.head_importance = imp.head_scores.clone();
            if c.min_retained_importance == 0.0 {
                c.min_retained_importance = imp.default_min_retained;
            }
        }
    }
}

/// Tunables.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Fraction of heads the default threshold allows to be pruned
    /// (clamped to `[0.0, 0.9]`).  Default 0.25.
    pub default_prune_fraction: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            default_prune_fraction: 0.25,
        }
    }
}

/// Run Stage 3 over every layer in `graph`.  `num_heads` and the KV-head
/// count from `layer_shape` drive head-slice addressing.
pub fn analyze(
    graph: &OptGraph,
    layer_shape: &LayerShape,
    num_heads: usize,
    weights: &dyn WeightProvider,
    config: &AnalysisConfig,
) -> WeightAnalysisReport {
    let mut report = WeightAnalysisReport::default();
    let prune_fraction = config.default_prune_fraction.clamp(0.0, 0.9);
    for layer in &graph.layers {
        let raw = score_layer(&layer.name, layer_shape, num_heads, weights);
        let any_signal = raw.iter().any(|s| *s > 0.0);
        if !any_signal {
            report.layers_without_weights += 1;
        }
        let head_scores = normalise(raw, num_heads);
        let default_min_retained = percentile_threshold(&head_scores, prune_fraction);
        report.per_layer.push(LayerImportance {
            head_scores,
            default_min_retained,
        });
    }
    report
}

/// Score one layer's Q heads.
///
/// Returns a length-`num_heads` vector of raw (un-normalised) scores.
/// Handles GQA: Wk / Wv are iterated over `layer_shape.n_kv_heads` and
/// each KV-head score is distributed across the Q-head group of size
/// `num_heads / n_kv_heads`.
pub fn score_layer(
    layer_name: &str,
    layer_shape: &LayerShape,
    num_heads: usize,
    w: &dyn WeightProvider,
) -> Vec<f64> {
    let head_dim = layer_shape.head_dim as usize;
    let d_model = layer_shape.d_model as usize;
    let n_kv_heads = (layer_shape.n_kv_heads as usize).max(1);
    let group_size = (num_heads / n_kv_heads).max(1);

    let mut scores = vec![0.0f64; num_heads];

    // Wq: [d_model, num_heads * head_dim].
    if let Some(data) = w.get(&format!("{layer_name}.attn.wq")) {
        add_columnar_scores(&mut scores, data, d_model, num_heads, head_dim);
    }

    // Wk, Wv: [d_model, n_kv_heads * head_dim] → distribute to Q groups.
    for proj in &["wk", "wv"] {
        let key = format!("{layer_name}.attn.{proj}");
        let Some(data) = w.get(&key) else { continue };
        let mut kv_scores = vec![0.0f64; n_kv_heads];
        add_columnar_scores(&mut kv_scores, data, d_model, n_kv_heads, head_dim);
        for h in 0..num_heads {
            scores[h] += kv_scores[h / group_size];
        }
    }

    // Wo: [num_heads * head_dim, d_model] — row block per Q head.
    if let Some(data) = w.get(&format!("{layer_name}.attn.wo")) {
        add_row_scores(&mut scores, data, d_model, num_heads, head_dim);
    }

    scores
}

/// Per-layer entry point for the magnitude scorer, suitable for Phase 2
/// scorer impls that call this layer-by-layer rather than driving
/// [`analyze`] over the full graph.
///
/// Derives `num_heads` from `shape.d_model / shape.head_dim`, computes
/// magnitude-based raw scores via [`score_layer`], normalises them, and
/// wraps the result in a [`HeadImportance`].  Falls back to uniform
/// scores when all weights are absent or all-zero (same semantics as the
/// `any_signal` path in [`analyze`]).
pub fn score_layer_magnitude(
    layer_key: &str,
    shape: &LayerShape,
    provider: &dyn WeightProvider,
) -> HeadImportance {
    let num_heads = (shape.d_model / shape.head_dim.max(1)).max(1) as usize;
    let raw = score_layer(layer_key, shape, num_heads, provider);
    let normalised = normalise(raw, num_heads);
    HeadImportance {
        per_head: normalised.iter().map(|&v| v as f32).collect(),
    }
}

/// Normalise `scores` so they sum to `num_heads` (making thresholds
/// invariant to the absolute magnitude of weights).  All-zero or empty
/// input yields a uniform vector.
pub fn normalise(mut scores: Vec<f64>, num_heads: usize) -> Vec<f64> {
    if num_heads == 0 {
        return Vec::new();
    }
    if scores.len() != num_heads {
        // Defensive: length mismatch would be a caller bug.  Fall back
        // to uniform rather than panic.
        return vec![1.0; num_heads];
    }
    let total: f64 = scores.iter().sum();
    if total <= 0.0 || !total.is_finite() {
        return vec![1.0; num_heads];
    }
    let scale = num_heads as f64 / total;
    for s in &mut scores {
        *s *= scale;
    }
    scores
}

/// Threshold for `min_retained_importance`.  The ILP's
/// `importance_ok()` constraint checks
/// `Σ keep[k] × head_scores[k] ≥ min_retained_importance`, so the
/// threshold is set to the sum of the top
/// `num_heads × (1 - prune_fraction)` scores — any keep-mask that drops
/// a more-important head falls below and is rejected.
///
/// **Worked example.** 8 heads with normalised `head_scores`
/// `[0.3, 0.5, 0.6, 0.8, 1.1, 1.2, 1.5, 2.0]` (sums to 8.0);
/// `prune_fraction = 0.25` → `drop_count = 2`.  Sorted ascending,
/// `skip(2).sum() = 0.6 + 0.8 + 1.1 + 1.2 + 1.5 + 2.0 = 7.2`.  Any
/// keep-mask that retains only 5 heads (or drops one of the top 6)
/// sums to at most 6.2 < 7.2 and fails the constraint.  The result is
/// **not** exactly `n_heads × (1 - prune_fraction) = 6.0` — normalise
/// preserves relative magnitudes, not uniformity.
pub fn percentile_threshold(scores: &[f64], prune_fraction: f64) -> f64 {
    if scores.is_empty() {
        return 0.0;
    }
    let mut sorted: Vec<f64> = scores.iter().copied().collect();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let drop_count = (sorted.len() as f64 * prune_fraction).floor() as usize;
    sorted.iter().skip(drop_count).sum()
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Add L2-norm-per-column-block into `out`.  Matrix `data` has shape
/// `[rows, n_blocks * block_cols]` in row-major layout.  `out[b]` gains
/// the L2 norm of block `b`'s columns.
fn add_columnar_scores(
    out: &mut [f64],
    data: &[f32],
    rows: usize,
    n_blocks: usize,
    block_cols: usize,
) {
    let total_cols = n_blocks * block_cols;
    if data.len() < rows * total_cols {
        return; // shape mismatch — skip silently rather than panic
    }
    for b in 0..n_blocks {
        let col_start = b * block_cols;
        let col_end = col_start + block_cols;
        let mut sumsq = 0.0f64;
        for row in 0..rows {
            for col in col_start..col_end {
                let v = data[row * total_cols + col] as f64;
                sumsq += v * v;
            }
        }
        out[b] += sumsq.sqrt();
    }
}

/// Add L2-norm-per-row-block into `out`.  Matrix `data` has shape
/// `[n_blocks * block_rows, cols]` in row-major layout.
fn add_row_scores(
    out: &mut [f64],
    data: &[f32],
    cols: usize,
    n_blocks: usize,
    block_rows: usize,
) {
    let total_rows = n_blocks * block_rows;
    if data.len() < total_rows * cols {
        return;
    }
    for b in 0..n_blocks {
        let row_start = b * block_rows;
        let row_end = row_start + block_rows;
        let mut sumsq = 0.0f64;
        for row in row_start..row_end {
            for col in 0..cols {
                let v = data[row * cols + col] as f64;
                sumsq += v * v;
            }
        }
        out[b] += sumsq.sqrt();
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wggo_graph::{build as build_graph, OptGraph};
    use crate::wengert::{PrimalOp, WengertOp, WengertList};
    use std::collections::HashMap;

    /// Test-only provider backed by a `HashMap`.
    #[derive(Debug, Default)]
    struct MapProvider {
        data: HashMap<String, Vec<f32>>,
        shapes: HashMap<String, Vec<u64>>,
    }
    impl MapProvider {
        fn insert(&mut self, name: &str, shape: Vec<u64>, data: Vec<f32>) {
            self.shapes.insert(name.to_string(), shape);
            self.data.insert(name.to_string(), data);
        }
    }
    impl WeightProvider for MapProvider {
        fn get(&self, name: &str) -> Option<&[f32]> {
            self.data.get(name).map(|v| v.as_slice())
        }
        fn shape(&self, name: &str) -> Option<&[u64]> {
            self.shapes.get(name).map(|v| v.as_slice())
        }
    }

    fn shape(n_kv_heads: u64) -> LayerShape {
        LayerShape {
            batch: 1,
            seq: 64,
            d_model: 16,
            head_dim: 4,
            n_kv_heads,
            dtype_bytes: 2,
        }
    }

    fn op(id: u32, o: PrimalOp, inputs: Vec<u32>) -> WengertOp {
        WengertOp {
            id,
            result: id,
            op: o,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    fn single_block_graph() -> OptGraph {
        let ops = vec![
            op(0, PrimalOp::Input("x".into()), vec![]),
            op(1, PrimalOp::Param("blocks.0.attn.wq".into()), vec![]),
            op(2, PrimalOp::Matmul, vec![1, 0]),
        ];
        let wl = WengertList {
            ops,
            output: 2,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        build_graph(&wl)
    }

    fn find_layer_scores<'a>(
        g: &OptGraph,
        rep: &'a WeightAnalysisReport,
        name: &str,
    ) -> &'a [f64] {
        let idx = g
            .layers
            .iter()
            .position(|l| l.name == name)
            .unwrap_or_else(|| panic!("layer {name} not found"));
        &rep.per_layer[idx].head_scores
    }

    #[test]
    fn null_provider_yields_equal_scores() {
        let g = single_block_graph();
        let rep = analyze(&g, &shape(4), 4, &NullWeightProvider, &AnalysisConfig::default());
        assert_eq!(rep.per_layer.len(), g.layers.len());
        assert_eq!(rep.layers_without_weights, g.layers.len() as u32);
        for imp in &rep.per_layer {
            assert_eq!(imp.head_scores.len(), 4);
            for &s in &imp.head_scores {
                assert!((s - 1.0).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn graph_contains_blocks_0_layer() {
        // Guardrail: the rest of the tests assume the graph builder
        // produces a "blocks.0" layer.  If this ever changes the other
        // tests will fail silently-looking ways — catch it here.
        let g = single_block_graph();
        assert!(g.layers.iter().any(|l| l.name == "blocks.0"));
    }

    #[test]
    fn magnitude_analyzer_picks_high_norm_heads() {
        // 4 Q heads, 4 KV heads (no grouping), head_dim=4, d_model=16.
        // Wq shape [16, 16]; head h occupies columns [4h..4h+4).
        // Set head 2's columns to 10×; other columns = 1.
        let mut wq = vec![1.0f32; 16 * 16];
        for row in 0..16 {
            for col in 8..12 {
                wq[row * 16 + col] = 10.0;
            }
        }
        let mut prov = MapProvider::default();
        prov.insert("blocks.0.attn.wq", vec![16, 16], wq);

        let g = single_block_graph();
        let rep = analyze(&g, &shape(4), 4, &prov, &AnalysisConfig::default());
        let s = find_layer_scores(&g, &rep, "blocks.0");
        // Head 2 must dominate.
        assert!(s[2] > s[0]);
        assert!(s[2] > s[1]);
        assert!(s[2] > s[3]);
    }

    #[test]
    fn normalisation_preserves_relative_ordering() {
        let raw = vec![1.0, 5.0, 3.0, 2.0];
        let norm = normalise(raw.clone(), 4);
        // Ordering must match.
        let rank = |v: &Vec<f64>, i: usize| v.iter().filter(|&&x| x < v[i]).count();
        for i in 0..4 {
            assert_eq!(rank(&raw, i), rank(&norm, i));
        }
        // Sum is n_heads.
        let total: f64 = norm.iter().sum();
        assert!((total - 4.0).abs() < 1e-9);
    }

    #[test]
    fn percentile_threshold_allows_specified_fraction() {
        let s = vec![0.3, 0.5, 0.6, 0.8, 1.1, 1.2, 1.5, 2.0];
        let t = percentile_threshold(&s, 0.25); // drop 2 → keep top 6
        let expected = 0.6 + 0.8 + 1.1 + 1.2 + 1.5 + 2.0;
        assert!((t - expected).abs() < 1e-9, "got {t}");
    }

    #[test]
    fn analyzer_zero_scores_when_all_zero_weights() {
        let mut prov = MapProvider::default();
        prov.insert("blocks.0.attn.wq", vec![16, 16], vec![0.0f32; 256]);
        let g = single_block_graph();
        let rep = analyze(&g, &shape(4), 4, &prov, &AnalysisConfig::default());
        // Zero weights in "blocks.0" + absent weights in "other"
        // → both layers count as without_weights, both fall back to
        // uniform scores.
        assert_eq!(rep.layers_without_weights, g.layers.len() as u32);
        let s = find_layer_scores(&g, &rep, "blocks.0");
        for &v in s {
            assert!((v - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn gqa_kv_score_distributes_across_q_group() {
        // 6 Q heads, 3 KV heads, group_size=2.  Shape: n_kv_heads=3,
        // head_dim=4, d_model=16.  Wk shape [16, 12].  Boost KV head 1
        // (columns 4..8) to 10×; expect Q heads 2 and 3 (group 1) to
        // receive the elevated contribution, and no out-of-bounds read.
        let mut shape = shape(3);
        shape.n_kv_heads = 3;
        let mut wk = vec![1.0f32; 16 * 12];
        for row in 0..16 {
            for col in 4..8 {
                wk[row * 12 + col] = 10.0;
            }
        }
        let mut prov = MapProvider::default();
        prov.insert("blocks.0.attn.wk", vec![16, 12], wk);

        let g = single_block_graph();
        let rep = analyze(&g, &shape, 6, &prov, &AnalysisConfig::default());
        let s = find_layer_scores(&g, &rep, "blocks.0");
        assert_eq!(s.len(), 6);
        // Q heads 2 and 3 (KV group 1) must dominate.
        assert!(s[2] > s[0] && s[2] > s[1]);
        assert!(s[3] > s[0] && s[3] > s[1]);
        assert!(s[2] > s[4] && s[2] > s[5]);
        // And the two Q heads in the same KV group must be ~equal.
        assert!((s[2] - s[3]).abs() < 1e-9);
    }

    #[test]
    fn caller_override_wins_over_default_threshold() {
        let g = single_block_graph();
        let rep = analyze(&g, &shape(4), 4, &NullWeightProvider, &AnalysisConfig::default());
        let mut cons = vec![LayerIlpConstraints {
            num_heads: 4,
            min_retained_importance: 42.0, // explicit override
            ..Default::default()
        }];
        rep.apply_to(&mut cons);
        // head_importance is overwritten, min_retained_importance is not.
        assert_eq!(cons[0].head_importance.len(), 4);
        assert_eq!(cons[0].min_retained_importance, 42.0);
    }

    #[test]
    fn default_threshold_fills_when_caller_leaves_zero() {
        // Non-zero normalised scores → default_min_retained > 0.
        let mut wq = vec![1.0f32; 16 * 16];
        for row in 0..16 {
            for col in 8..12 {
                wq[row * 16 + col] = 10.0;
            }
        }
        let mut prov = MapProvider::default();
        prov.insert("blocks.0.attn.wq", vec![16, 16], wq);
        let g = single_block_graph();
        let rep = analyze(&g, &shape(4), 4, &prov, &AnalysisConfig::default());
        let mut cons = vec![LayerIlpConstraints {
            num_heads: 4,
            min_retained_importance: 0.0,
            ..Default::default()
        }];
        rep.apply_to(&mut cons);
        assert!(cons[0].min_retained_importance > 0.0);
        assert_eq!(cons[0].min_retained_importance, rep.per_layer[0].default_min_retained);
    }

    #[test]
    fn score_layer_magnitude_with_null_provider_yields_uniform_scores() {
        let s = shape(4); // n_kv_heads=4, d_model=16, head_dim=4 → n_heads=4
        let hi = score_layer_magnitude("layer.0", &s, &NullWeightProvider);
        let n_heads = (s.d_model / s.head_dim) as usize;
        assert_eq!(hi.per_head.len(), n_heads);
        // All uniform (null provider → no signal → normalise to 1.0 each).
        let first = hi.per_head[0];
        assert!(
            hi.per_head.iter().all(|&v| (v - first).abs() < 1e-6),
            "expected uniform scores, got {:?}",
            hi.per_head
        );
    }

    #[test]
    fn analysis_config_clamps_prune_fraction() {
        // Out-of-range prune_fraction shouldn't panic or produce NaN.
        let mut wq = vec![1.0f32; 16 * 16];
        for row in 0..16 {
            for col in 8..12 {
                wq[row * 16 + col] = 5.0;
            }
        }
        let mut prov = MapProvider::default();
        prov.insert("blocks.0.attn.wq", vec![16, 16], wq);
        let g = single_block_graph();
        let bad = AnalysisConfig {
            default_prune_fraction: 2.0, // clamped to 0.9
        };
        let rep = analyze(&g, &shape(4), 4, &prov, &bad);
        assert!(rep.per_layer[0].default_min_retained.is_finite());
        assert!(rep.per_layer[0].default_min_retained >= 0.0);
    }
}
