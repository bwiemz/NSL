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
//! contribution is skipped (so a partial checkpoint still yields usable
//! scores).  A layer whose weights are entirely absent falls back to
//! uniform scores and is counted in
//! `WeightAnalysisReport::layers_without_weights`.  A tensor that is
//! present with a shape other than the layer's is an error: slicing it
//! with the wrong strides would produce scores for heads that do not
//! exist.
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
    /// Whether the analyzer had a REAL weight signal for this layer
    /// (any nonzero raw score). False under NullWeightProvider or a
    /// checkpoint missing this layer. Gates sub-32-bit optimizer-moment
    /// recommendations: quantizing moments on zero signal would be a
    /// cost-only choice with no sensitivity evidence behind it.
    /// `serde(default)`: pre-existing sidecar caches lack the field and
    /// deserialize to `false` — conservative (no quantization enabled).
    #[serde(default)]
    pub has_signal: bool,
}

/// Aggregate report from [`analyze`].
#[derive(Debug, Clone, Default, Serialize)]
pub struct WeightAnalysisReport {
    pub per_layer: Vec<LayerImportance>,
    /// Layers whose weights were entirely absent → uniform fallback.
    pub layers_without_weights: u32,
    /// The head-prune fraction the thresholds were built for. Zero means
    /// the user did not ask for head pruning, and [`Self::apply_to`] then
    /// leaves the structural axes pinned to keep-all whatever the scores.
    pub prune_fraction: f64,
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
            // Sub-32-bit moment precision requires evidence; see
            // `prec_allowed`. Only ever ENABLE — never clear a flag a
            // caller set explicitly.
            if imp.has_signal {
                c.sensitivity_informed = true;
                // Head pruning is opt-in. A weights file alone is not a
                // request to prune: the scores of a freshly initialised
                // model differ by noise, and pruning on them by default
                // removed a quarter of the heads of any model compiled
                // with `--wggo-weights`. The structural axes open only
                // when the user also gave a prune fraction above zero.
                if self.prune_fraction > 0.0 {
                    c.importance_informed = true;
                }
            }
        }
    }
}

/// Tunables.
#[derive(Debug, Clone)]
pub struct AnalysisConfig {
    /// Fraction of heads the default threshold allows to be pruned
    /// (clamped to `[0.0, 0.9]`).  Default 0.0: no head is pruned unless
    /// the user asks (`--wggo-prune-fraction`).
    pub default_prune_fraction: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            default_prune_fraction: 0.0,
        }
    }
}

/// The attention shape of one layer, as the head scorer needs it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeadLayout {
    pub d_model: usize,
    pub num_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
}

impl HeadLayout {
    /// The layout a nominal [`LayerShape`] and head count describe.
    pub fn nominal(shape: &LayerShape, num_heads: usize) -> Self {
        Self {
            d_model: shape.d_model as usize,
            num_heads,
            n_kv_heads: (shape.n_kv_heads as usize).max(1),
            head_dim: shape.head_dim as usize,
        }
    }
}

/// A weights-file tensor whose shape is not the layer's.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WeightShapeMismatch {
    pub tensor: String,
    pub found: Vec<u64>,
    pub expected: Vec<u64>,
    pub layout: HeadLayout,
}

impl std::fmt::Display for WeightShapeMismatch {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "tensor `{}` has shape {:?}, but the model's layer has {} heads of {} ({} key/value \
             heads) over a {}-wide stream, which makes it {:?}",
            self.tensor,
            self.found,
            self.layout.num_heads,
            self.layout.head_dim,
            self.layout.n_kv_heads,
            self.layout.d_model,
            self.expected,
        )
    }
}

/// Run Stage 3 over every layer in `graph`.
///
/// `layouts` is parallel to `graph.layers`: the attention shape of each
/// layer, `None` for a layer with no attention (it gets no head scores).
/// `Err` when a tensor in the weights source has a shape other than its
/// layer's.
pub fn analyze_layers(
    graph: &OptGraph,
    layouts: &[Option<HeadLayout>],
    weights: &dyn WeightProvider,
    config: &AnalysisConfig,
) -> Result<WeightAnalysisReport, WeightShapeMismatch> {
    let prune_fraction = config.default_prune_fraction.clamp(0.0, 0.9);
    let mut report = WeightAnalysisReport {
        prune_fraction,
        ..Default::default()
    };
    for (i, layer) in graph.layers.iter().enumerate() {
        let Some(layout) = layouts.get(i).copied().flatten() else {
            report.per_layer.push(LayerImportance {
                head_scores: Vec::new(),
                default_min_retained: 0.0,
                has_signal: false,
            });
            continue;
        };
        let raw = score_layer_checked(&layer.name, &layout, weights)?;
        let any_signal = raw.iter().any(|s| *s > 0.0);
        if !any_signal {
            report.layers_without_weights += 1;
        }
        let head_scores = normalise(raw, layout.num_heads);
        let default_min_retained = percentile_threshold(&head_scores, prune_fraction);
        report.per_layer.push(LayerImportance {
            head_scores,
            default_min_retained,
            has_signal: any_signal,
        });
    }
    Ok(report)
}

/// [`analyze_layers`] with one nominal shape for every layer. For callers
/// that size the model themselves (unit tests, tools); the compile pipeline
/// passes each layer's own layout.
pub fn analyze(
    graph: &OptGraph,
    layer_shape: &LayerShape,
    num_heads: usize,
    weights: &dyn WeightProvider,
    config: &AnalysisConfig,
) -> Result<WeightAnalysisReport, WeightShapeMismatch> {
    let layouts = vec![Some(HeadLayout::nominal(layer_shape, num_heads)); graph.layers.len()];
    analyze_layers(graph, &layouts, weights, config)
}

/// Score one layer's Q heads from a nominal shape. See
/// [`score_layer_checked`].
pub fn score_layer(
    layer_name: &str,
    layer_shape: &LayerShape,
    num_heads: usize,
    w: &dyn WeightProvider,
) -> Result<Vec<f64>, WeightShapeMismatch> {
    score_layer_checked(layer_name, &HeadLayout::nominal(layer_shape, num_heads), w)
}

/// Fetch a tensor and check it has the shape the layer's layout gives it.
fn checked_tensor<'a>(
    w: &'a dyn WeightProvider,
    name: &str,
    expected: [usize; 2],
    layout: &HeadLayout,
) -> Result<Option<&'a [f32]>, WeightShapeMismatch> {
    let Some(data) = w.get(name) else {
        return Ok(None);
    };
    let expected_u64 = vec![expected[0] as u64, expected[1] as u64];
    let found: Vec<u64> = match w.shape(name) {
        Some(s) => s.to_vec(),
        // A source that records no shape is held to the element count.
        None if data.len() == expected[0] * expected[1] => expected_u64.clone(),
        None => vec![data.len() as u64],
    };
    if found != expected_u64 {
        return Err(WeightShapeMismatch {
            tensor: name.to_string(),
            found,
            expected: expected_u64,
            layout: *layout,
        });
    }
    Ok(Some(data))
}

/// Score one layer's Q heads.
///
/// Returns a length-`num_heads` vector of raw (un-normalised) scores.
/// Handles GQA: Wk / Wv are iterated over the key/value heads and each
/// KV-head score is distributed across its Q-head group. Every tensor is
/// checked against the shape the layout gives it before it is sliced.
pub fn score_layer_checked(
    layer_name: &str,
    layout: &HeadLayout,
    w: &dyn WeightProvider,
) -> Result<Vec<f64>, WeightShapeMismatch> {
    let HeadLayout { d_model, num_heads, n_kv_heads, head_dim } = *layout;
    let n_kv_heads = n_kv_heads.max(1);
    let group_size = (num_heads / n_kv_heads).max(1);

    let mut scores = vec![0.0f64; num_heads];

    // Wq: [d_model, num_heads * head_dim].
    let wq = format!("{layer_name}.attn.wq");
    if let Some(data) = checked_tensor(w, &wq, [d_model, num_heads * head_dim], layout)? {
        add_columnar_scores(&mut scores, data, d_model, num_heads, head_dim);
    }

    // Wk, Wv: [d_model, n_kv_heads * head_dim] → distribute to Q groups.
    for proj in &["wk", "wv"] {
        let key = format!("{layer_name}.attn.{proj}");
        let Some(data) = checked_tensor(w, &key, [d_model, n_kv_heads * head_dim], layout)? else {
            continue;
        };
        let mut kv_scores = vec![0.0f64; n_kv_heads];
        add_columnar_scores(&mut kv_scores, data, d_model, n_kv_heads, head_dim);
        for (h, score) in scores.iter_mut().enumerate() {
            *score += kv_scores[(h / group_size).min(n_kv_heads - 1)];
        }
    }

    // Wo: [num_heads * head_dim, d_model] — row block per Q head.
    let wo = format!("{layer_name}.attn.wo");
    if let Some(data) = checked_tensor(w, &wo, [num_heads * head_dim, d_model], layout)? {
        add_row_scores(&mut scores, data, d_model, num_heads, head_dim);
    }

    Ok(scores)
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
    // A tensor of another shape contributes nothing here; the pipeline's
    // own pass over the same source (`analyze_layers`) is what reports it.
    let raw = score_layer(layer_key, shape, num_heads, provider)
        .unwrap_or_else(|_| vec![0.0; num_heads]);
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
    let mut sorted: Vec<f64> = scores.to_vec();
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
        return; // unreachable from `score_layer_checked`, which checks first
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
        let rep = analyze(&g, &shape(4), 4, &NullWeightProvider, &AnalysisConfig::default()).unwrap();
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
        let rep = analyze(&g, &shape(4), 4, &prov, &AnalysisConfig::default()).unwrap();
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
        let rep = analyze(&g, &shape(4), 4, &prov, &AnalysisConfig::default()).unwrap();
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
        let rep = analyze(&g, &shape, 6, &prov, &AnalysisConfig::default()).unwrap();
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
        let rep = analyze(&g, &shape(4), 4, &NullWeightProvider, &AnalysisConfig::default()).unwrap();
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
        let rep = analyze(&g, &shape(4), 4, &prov, &AnalysisConfig::default()).unwrap();
        let mut cons = vec![LayerIlpConstraints {
            num_heads: 4,
            min_retained_importance: 0.0,
            ..Default::default()
        }];
        rep.apply_to(&mut cons);
        assert!(cons[0].min_retained_importance > 0.0);
        assert_eq!(cons[0].min_retained_importance, rep.per_layer[0].default_min_retained);
    }

    /// Scores from a weights file are evidence about precision, not a
    /// request to prune: the structural axes open only with a prune
    /// fraction above zero. The default fraction used to be 0.25, so any
    /// weights file, including freshly initialised weights whose scores
    /// differ by noise, pruned a quarter of the heads.
    #[test]
    fn a_weights_file_alone_does_not_open_head_pruning() {
        let mut wq = vec![1.0f32; 16 * 16];
        for row in 0..16 {
            for col in 8..12 {
                wq[row * 16 + col] = 10.0;
            }
        }
        let mut prov = MapProvider::default();
        prov.insert("blocks.0.attn.wq", vec![16, 16], wq);
        let g = single_block_graph();
        let idx = g.layers.iter().position(|l| l.name == "blocks.0").unwrap();
        assert_eq!(AnalysisConfig::default().default_prune_fraction, 0.0);

        let apply = |fraction: f64| {
            let cfg = AnalysisConfig { default_prune_fraction: fraction };
            let rep = analyze(&g, &shape(4), 4, &prov, &cfg).unwrap();
            let mut cons = vec![
                LayerIlpConstraints { num_heads: 4, ..Default::default() };
                g.layers.len()
            ];
            rep.apply_to(&mut cons);
            (rep, cons.swap_remove(idx))
        };

        let (rep, c) = apply(0.0);
        assert!(rep.per_layer[idx].has_signal);
        assert!(c.sensitivity_informed, "the scores are still evidence");
        assert!(!c.importance_informed, "no fraction given: keep-all only");
        // With nothing prunable the threshold is the whole: 4 heads' worth.
        assert!((c.min_retained_importance - 4.0).abs() < 1e-9);

        let (_, c) = apply(0.25);
        assert!(c.importance_informed, "an explicit fraction opens pruning");
    }

    /// A tensor of another model's shape is an error, not eight scores
    /// read out of the wrong columns. The 12-head, 768-wide matrix below
    /// used to be sliced as 8 heads of 64 over its first 512 x 512 values.
    #[test]
    fn a_tensor_of_the_wrong_shape_is_refused() {
        let mut prov = MapProvider::default();
        prov.insert("blocks.0.attn.wq", vec![768, 768], vec![1.0f32; 768 * 768]);
        let nominal = LayerShape {
            batch: 1,
            seq: 1,
            d_model: 512,
            head_dim: 64,
            n_kv_heads: 4,
            dtype_bytes: 2,
        };
        let err = score_layer("blocks.0", &nominal, 8, &prov).unwrap_err();
        assert_eq!(err.tensor, "blocks.0.attn.wq");
        assert_eq!(err.found, vec![768, 768]);
        assert_eq!(err.expected, vec![512, 512]);
        let msg = err.to_string();
        assert!(msg.contains("blocks.0.attn.wq") && msg.contains("[768, 768]"), "{msg}");

        // With the layer's own layout the same tensor scores twelve heads,
        // all alike (every column block has norm sqrt(768 x 64)).
        let layout = HeadLayout { d_model: 768, num_heads: 12, n_kv_heads: 4, head_dim: 64 };
        let scores = score_layer_checked("blocks.0", &layout, &prov).unwrap();
        assert_eq!(scores.len(), 12);
        let expected = (768.0f64 * 64.0).sqrt();
        for s in &scores {
            assert!((s - expected).abs() < 1e-6, "{s} vs {expected}");
        }

        // A smaller tensor is refused too; it used to be skipped, so a
        // model narrower than 512 read as "no weights".
        let mut small = MapProvider::default();
        small.insert("blocks.0.attn.wq", vec![64, 64], vec![1.0f32; 64 * 64]);
        assert!(score_layer("blocks.0", &nominal, 8, &small).is_err());
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
        let rep = analyze(&g, &shape(4), 4, &prov, &bad).unwrap();
        assert!(rep.per_layer[0].default_min_retained.is_finite());
        assert!(rep.per_layer[0].default_min_retained >= 0.0);
    }
}
