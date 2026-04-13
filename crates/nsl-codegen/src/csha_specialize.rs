//! CSHA Per-Layer Specialization (paper §3).
//!
//! Extracts per-head, per-layer characteristics from pre-trained weights
//! so the compiler can emit specialised kernels:
//!
//!   * Frobenius-norm head importance → head pruning
//!   * Effective entropy estimate     → sparse-attention selection
//!   * Per-head precision hint        → mixed-precision kernel
//!
//! This module is deliberately narrow: it reads from a `WeightMap` and
//! produces a `SpecializationPlan`.  Kernel selection happens later in
//! `csha.rs`.

use serde::Serialize;

use crate::csha_boundary::{BoundaryScan, ProjKind};
use crate::weight_aware::{WeightDType, WeightEntry, WeightMap};

/// Precision hint for one attention head.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum HeadPrecision {
    /// FP16 — default, preserves dynamic range.
    Fp16,
    /// FP8 E4M3 — higher precision, narrower range (Q/K).
    Fp8E4m3,
    /// FP8 E5M2 — wider range (V accumulation).
    Fp8E5m2,
}

impl HeadPrecision {
    pub fn as_str(self) -> &'static str {
        match self {
            HeadPrecision::Fp16 => "fp16",
            HeadPrecision::Fp8E4m3 => "fp8_e4m3",
            HeadPrecision::Fp8E5m2 => "fp8_e5m2",
        }
    }
}

/// Per-head analysis result.
#[derive(Debug, Clone, Serialize)]
pub struct HeadStats {
    pub head_idx: u32,
    /// Combined Frobenius norm of `Wq[:, head]` + `Wk[:, head]`.
    pub qk_norm: f64,
    /// Max absolute value across the head's Q/K weights.
    pub max_abs: f64,
    /// True if the head should be pruned (qk_norm below threshold).
    pub pruned: bool,
    /// Precision selected for this head.
    pub precision: HeadPrecision,
}

/// Per-layer specialization plan.
#[derive(Debug, Clone, Serialize)]
pub struct LayerSpec {
    pub layer: String,
    pub n_heads: u32,
    pub n_active_heads: u32,
    /// Coarse attention-entropy bucket (low → sparse, medium → block-sparse,
    /// high → dense).  Currently a stub that derives from the Q/K norm
    /// distribution; refined analysis is future work.
    pub entropy_bucket: EntropyBucket,
    pub heads: Vec<HeadStats>,
    /// Which projection weights were present for this layer (some
    /// models only ship Q/K, not V, in checkpoints).
    pub seen_projs: Vec<ProjKind>,
}

/// Coarse bucket for attention entropy — drives sparse-kernel selection.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum EntropyBucket {
    /// Focused attention — good candidate for sparse top-k kernels.
    Low,
    /// Mixed — block-sparse.
    Medium,
    /// Diffuse — dense FA is best.
    High,
}

impl EntropyBucket {
    pub fn as_str(self) -> &'static str {
        match self {
            EntropyBucket::Low => "low",
            EntropyBucket::Medium => "medium",
            EntropyBucket::High => "high",
        }
    }
}

/// Full specialization plan: one entry per layer in the scan.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SpecializationPlan {
    pub layers: Vec<LayerSpec>,
}

impl SpecializationPlan {
    /// Count of heads eliminated across all layers.
    pub fn total_pruned_heads(&self) -> u32 {
        self.layers
            .iter()
            .map(|l| l.n_heads - l.n_active_heads)
            .sum()
    }

    pub fn get(&self, layer: &str) -> Option<&LayerSpec> {
        self.layers.iter().find(|l| l.layer == layer)
    }
}

/// Thresholds used by the analyser.  All relative to per-layer maxima,
/// so the defaults work across model scales.
#[derive(Debug, Clone, Copy)]
pub struct SpecConfig {
    /// Prune heads whose `qk_norm` is below this fraction of the
    /// layer's max `qk_norm`.  Paper mentions "norm < 0.01" — we use a
    /// *relative* threshold so small-scale weights are not all pruned.
    pub prune_threshold_rel: f64,
    /// FP8 E4M3 cutoff: heads with `max_abs` below this (absolute) get
    /// the narrower-range precision.
    pub fp8_max_abs_cutoff: f64,
}

impl Default for SpecConfig {
    fn default() -> Self {
        Self {
            prune_threshold_rel: 0.01,
            fp8_max_abs_cutoff: 6.0,
        }
    }
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------

/// Build a specialization plan from the boundary scan + weight map.
///
/// Layers without weights in the map get a fallback [`LayerSpec`] with
/// all heads kept at `Fp16` — matches the "no --weights" build behaviour.
pub fn analyze(
    scan: &BoundaryScan,
    weights: Option<&WeightMap>,
    n_heads: u32,
    cfg: &SpecConfig,
) -> SpecializationPlan {
    let mut layers = Vec::new();

    for (layer_name, chains) in scan.by_layer() {
        let mut seen_projs: Vec<ProjKind> = chains.iter().map(|c| c.kind).collect();
        seen_projs.sort_by_key(|p| *p as u8);
        seen_projs.dedup();

        // Locate Wq / Wk entries for this layer.
        let wq = chains
            .iter()
            .find(|c| c.kind == ProjKind::Q)
            .and_then(|c| weights.and_then(|w| w.get(&c.weight_param)));
        let wk = chains
            .iter()
            .find(|c| c.kind == ProjKind::K)
            .and_then(|c| weights.and_then(|w| w.get(&c.weight_param)));

        let heads = if let (Some(q), Some(k)) = (wq, wk) {
            analyze_heads(q, k, n_heads, cfg)
        } else {
            fallback_heads(n_heads)
        };

        let n_active = heads.iter().filter(|h| !h.pruned).count() as u32;

        // Coarse entropy bucket — uses the spread of qk_norms.  Wide spread →
        // focused heads dominate → low entropy.  Tight spread → diffuse.
        let bucket = entropy_bucket(&heads);

        layers.push(LayerSpec {
            layer: layer_name,
            n_heads,
            n_active_heads: n_active,
            entropy_bucket: bucket,
            heads,
            seen_projs,
        });
    }

    SpecializationPlan { layers }
}

/// Per-head analysis when both Wq and Wk are available.
///
/// Assumes weights are stored as `[d_model, d_proj]` with `d_proj =
/// n_heads * head_dim`.  Each head's columns are a contiguous slice.
fn analyze_heads(
    wq: &WeightEntry,
    wk: &WeightEntry,
    n_heads: u32,
    cfg: &SpecConfig,
) -> Vec<HeadStats> {
    let q_stats = per_head_frobenius(wq, n_heads);
    let k_stats = per_head_frobenius(wk, n_heads);

    let q_stats: Vec<(f64, f64)> = match q_stats {
        Some(s) => s,
        None => return fallback_heads(n_heads),
    };
    let k_stats: Vec<(f64, f64)> = match k_stats {
        Some(s) => s,
        None => return fallback_heads(n_heads),
    };
    if q_stats.len() != n_heads as usize || k_stats.len() != n_heads as usize {
        return fallback_heads(n_heads);
    }

    // Combined per-head signal.
    let combined: Vec<(f64, f64)> = q_stats
        .iter()
        .zip(k_stats.iter())
        .map(|(q, k)| ((q.0 + k.0).sqrt(), q.1.max(k.1)))
        .collect();

    // Relative prune threshold.
    let max_norm = combined.iter().map(|(n, _)| *n).fold(0.0, f64::max);
    let prune_cutoff = max_norm * cfg.prune_threshold_rel;

    combined
        .into_iter()
        .enumerate()
        .map(|(i, (norm, max_abs))| {
            let pruned = norm < prune_cutoff;
            let precision = if pruned {
                HeadPrecision::Fp16 // irrelevant, kernel skips it anyway
            } else if max_abs < cfg.fp8_max_abs_cutoff {
                HeadPrecision::Fp8E4m3
            } else {
                HeadPrecision::Fp16
            };
            HeadStats {
                head_idx: i as u32,
                qk_norm: norm,
                max_abs,
                pruned,
                precision,
            }
        })
        .collect()
}

/// Return `(sum_of_squares, max_abs)` per head.
fn per_head_frobenius(w: &WeightEntry, n_heads: u32) -> Option<Vec<(f64, f64)>> {
    if w.shape.len() != 2 {
        return None;
    }
    let rows = w.shape[0];
    let cols = w.shape[1];
    if n_heads as usize == 0 || cols % n_heads as usize != 0 {
        return None;
    }
    let head_width = cols / n_heads as usize;
    let bw = w.dtype.byte_width();
    if bw == 0 || w.data.len() < rows * cols * bw {
        return None;
    }

    let mut out = vec![(0.0f64, 0.0f64); n_heads as usize];
    for r in 0..rows {
        for c in 0..cols {
            let offset = (r * cols + c) * bw;
            // Bounds-check each read — weights may be f64-truncated, etc.
            if offset + bw > w.data.len() {
                return None;
            }
            let val = read_scalar(w.dtype, &w.data[offset..offset + bw]);
            let head = c / head_width;
            out[head].0 += val * val;
            if val.abs() > out[head].1 {
                out[head].1 = val.abs();
            }
        }
    }
    Some(out)
}

fn read_scalar(dtype: WeightDType, bytes: &[u8]) -> f64 {
    dtype.to_f64(bytes)
}

fn fallback_heads(n_heads: u32) -> Vec<HeadStats> {
    (0..n_heads)
        .map(|i| HeadStats {
            head_idx: i,
            qk_norm: 1.0,
            max_abs: 1.0,
            pruned: false,
            precision: HeadPrecision::Fp16,
        })
        .collect()
}

fn entropy_bucket(heads: &[HeadStats]) -> EntropyBucket {
    // Standard deviation of `qk_norm`, normalised by the mean.  Wide
    // spread → focused attention → low entropy.
    let active: Vec<f64> = heads
        .iter()
        .filter(|h| !h.pruned)
        .map(|h| h.qk_norm)
        .collect();
    if active.len() < 2 {
        return EntropyBucket::High;
    }
    let mean = active.iter().sum::<f64>() / active.len() as f64;
    if mean <= 0.0 {
        return EntropyBucket::High;
    }
    let var = active.iter().map(|n| (n - mean).powi(2)).sum::<f64>() / active.len() as f64;
    let cv = var.sqrt() / mean; // coefficient of variation
    if cv > 0.5 {
        EntropyBucket::Low
    } else if cv > 0.2 {
        EntropyBucket::Medium
    } else {
        EntropyBucket::High
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::csha_boundary::BoundaryChain;

    fn synthetic_weight(name: &str, rows: usize, cols: usize, fill: impl Fn(usize, usize) -> f32) -> WeightEntry {
        let mut data = Vec::with_capacity(rows * cols * 4);
        for r in 0..rows {
            for c in 0..cols {
                data.extend_from_slice(&fill(r, c).to_le_bytes());
            }
        }
        WeightEntry {
            name: name.to_string(),
            data,
            shape: vec![rows, cols],
            dtype: WeightDType::F32,
            num_elements: rows * cols,
            sparsity: None,
            eliminated: false,
        }
    }

    #[test]
    fn per_head_frobenius_partitions_columns() {
        // 8 heads, head_width = 2, columns = 16; head i has uniform value
        // (i + 1).  Norm of head i = sqrt(rows * 2 * (i+1)^2).
        let w = synthetic_weight("wq", 4, 16, |_, c| (c / 2 + 1) as f32);
        let stats = per_head_frobenius(&w, 8).unwrap();
        assert_eq!(stats.len(), 8);
        // Norms monotonically increase with head index.
        for i in 1..8 {
            assert!(stats[i].0 > stats[i - 1].0);
        }
    }

    #[test]
    fn analyze_heads_prunes_zero_magnitude() {
        // Head 0 is all zeros; it should be pruned.
        let wq = synthetic_weight("wq", 4, 16, |_, c| if c < 2 { 0.0 } else { 1.0 });
        let wk = synthetic_weight("wk", 4, 16, |_, c| if c < 2 { 0.0 } else { 1.0 });
        let cfg = SpecConfig::default();
        let heads = analyze_heads(&wq, &wk, 8, &cfg);
        assert!(heads[0].pruned);
        assert!(heads.iter().skip(1).all(|h| !h.pruned));
    }

    #[test]
    fn fallback_heads_keep_all() {
        let heads = fallback_heads(8);
        assert_eq!(heads.len(), 8);
        assert!(heads.iter().all(|h| !h.pruned));
        assert!(heads.iter().all(|h| h.precision == HeadPrecision::Fp16));
    }

    #[test]
    fn analyze_without_weights_falls_back() {
        let scan = BoundaryScan {
            chains: vec![BoundaryChain {
                layer: Some("blocks.0".into()),
                kind: ProjKind::Q,
                norm_op: 0,
                matmul_op: 1,
                rope_op: Some(2),
                weight_param: "blocks.0.attn.wq".into(),
            }],
        };
        let plan = analyze(&scan, None, 8, &SpecConfig::default());
        assert_eq!(plan.layers.len(), 1);
        assert_eq!(plan.layers[0].n_active_heads, 8);
        assert_eq!(plan.total_pruned_heads(), 0);
    }

    #[test]
    fn entropy_bucket_varies_with_spread() {
        let uniform = (0..8)
            .map(|i| HeadStats {
                head_idx: i,
                qk_norm: 1.0,
                max_abs: 0.5,
                pruned: false,
                precision: HeadPrecision::Fp16,
            })
            .collect::<Vec<_>>();
        assert_eq!(entropy_bucket(&uniform), EntropyBucket::High);

        let spread = (0..8)
            .map(|i| HeadStats {
                head_idx: i,
                qk_norm: i as f64 + 1.0,
                max_abs: 0.5,
                pruned: false,
                precision: HeadPrecision::Fp16,
            })
            .collect::<Vec<_>>();
        // Coefficient of variation > 0.5 → Low.
        assert_eq!(entropy_bucket(&spread), EntropyBucket::Low);
    }

    #[test]
    fn total_pruned_heads_is_zero_for_healthy_weights() {
        let wq = synthetic_weight("wq", 4, 16, |r, c| (r + c + 1) as f32);
        let wk = synthetic_weight("wk", 4, 16, |r, c| (r + c + 2) as f32);
        let heads = analyze_heads(&wq, &wk, 8, &SpecConfig::default());
        let pruned = heads.iter().filter(|h| h.pruned).count();
        assert_eq!(pruned, 0);
    }

    #[test]
    fn mismatched_shape_returns_fallback() {
        let wq = synthetic_weight("wq", 4, 17, |_, _| 1.0); // not divisible by 8
        let wk = synthetic_weight("wk", 4, 17, |_, _| 1.0);
        let heads = analyze_heads(&wq, &wk, 8, &SpecConfig::default());
        // Fallback returns 8 heads all unpruned.
        assert_eq!(heads.len(), 8);
        assert!(heads.iter().all(|h| !h.pruned));
    }
}
