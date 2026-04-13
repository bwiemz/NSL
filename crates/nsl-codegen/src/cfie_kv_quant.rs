//! CFIE — per-layer KV-cache precision selection.
//!
//! Paper §7: instead of a uniform INT8 / FP16 KV cache across every
//! layer, the compiler analyses attention projection weights and
//! assigns a precision per layer:
//!
//!   * Layer 0         → FP16 K, FP16 V  (high position sensitivity)
//!   * Layer 1         → FP16 K, INT8 V  (K needs position precision)
//!   * Mid layers 2..L-2 → INT8 K, INT8 V  (low sensitivity)
//!   * Final layer     → FP16 K, FP16 V  (high impact on output)
//!
//! When weight data is available we refine this with spectral analysis;
//! without weights, we fall back to the position-based heuristic.

use serde::Serialize;

use crate::weight_aware::{WeightEntry, WeightMap};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum KvPrecision {
    Fp16,
    Bf16,
    Int8,
    Int4,
}

impl KvPrecision {
    pub fn byte_width(self) -> u32 {
        match self {
            KvPrecision::Fp16 | KvPrecision::Bf16 => 2,
            KvPrecision::Int8 => 1,
            KvPrecision::Int4 => 1, // 2× per byte — approx for reporting
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            KvPrecision::Fp16 => "fp16",
            KvPrecision::Bf16 => "bf16",
            KvPrecision::Int8 => "int8",
            KvPrecision::Int4 => "int4",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Sensitivity {
    High,
    Medium,
    Low,
}

#[derive(Debug, Clone, Serialize)]
pub struct LayerKvDecision {
    pub layer: u32,
    pub k_precision: KvPrecision,
    pub v_precision: KvPrecision,
    pub sensitivity: Sensitivity,
    pub sensitivity_score: f64,
    pub rationale: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct KvQuantPlan {
    pub layers: Vec<LayerKvDecision>,
    pub bytes_per_token_uniform_fp16: u64,
    pub bytes_per_token_selected: u64,
}

impl KvQuantPlan {
    pub fn memory_savings_ratio(&self) -> f64 {
        if self.bytes_per_token_uniform_fp16 == 0 {
            return 0.0;
        }
        1.0 - (self.bytes_per_token_selected as f64
            / self.bytes_per_token_uniform_fp16 as f64)
    }
    pub fn int8_layer_count(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| matches!(l.k_precision, KvPrecision::Int8 | KvPrecision::Int4))
            .count()
    }
}

#[derive(Debug, Clone)]
pub struct KvQuantConfig {
    /// Sensitivity threshold above which the layer stays at FP16.
    pub high_threshold: f64,
    /// Threshold below which the layer goes full INT8.
    pub low_threshold: f64,
    pub n_layers: u32,
    pub n_kv_heads: u32,
    pub head_dim: u32,
}

impl KvQuantConfig {
    pub fn new(n_layers: u32, n_kv_heads: u32, head_dim: u32) -> Self {
        // Thresholds tuned against the paper's position-weighted
        // scoring: with the default 0.7 pos + 0.3 spectral formula and
        // a 0.5 spectral default (no weights), middle layers end up
        // around 0.25.  `low_threshold = 0.3` cleanly puts them below
        // the Medium band.
        Self {
            high_threshold: 0.6,
            low_threshold: 0.3,
            n_layers,
            n_kv_heads,
            head_dim,
        }
    }
}

// ---------------------------------------------------------------------------
// Sensitivity scoring
// ---------------------------------------------------------------------------

fn position_factor(layer: u32, n_layers: u32) -> f64 {
    if n_layers <= 1 {
        return 1.0;
    }
    // U-shape with steeper endpoints — first and last layers must
    // survive at FP16.  Normalised to [0, 1] with peaks at 1.0.
    let t = layer as f64 / (n_layers - 1) as f64;
    let d = (2.0 * t - 1.0).abs();
    d
}

fn spectral_energy_proxy(entry: &WeightEntry) -> f64 {
    // Cheap proxy: Frobenius / (sqrt(n) × max_abs).  High value → low
    // rank → lower sensitivity.
    if entry.num_elements == 0 {
        return 0.0;
    }
    let bw = entry.dtype.byte_width();
    let mut sum_sq = 0.0_f64;
    let mut max_abs = 0.0_f64;
    for i in 0..entry.num_elements {
        let off = i * bw;
        if off + bw > entry.data.len() {
            break;
        }
        let v = entry.dtype.to_f64(&entry.data[off..off + bw]);
        sum_sq += v * v;
        let a = v.abs();
        if a > max_abs {
            max_abs = a;
        }
    }
    if max_abs < 1e-30 {
        return 0.0;
    }
    sum_sq.sqrt() / ((entry.num_elements as f64).sqrt() * max_abs)
}

fn classify(score: f64, cfg: &KvQuantConfig) -> Sensitivity {
    if score >= cfg.high_threshold {
        Sensitivity::High
    } else if score >= cfg.low_threshold {
        Sensitivity::Medium
    } else {
        Sensitivity::Low
    }
}

fn precisions_for(sensitivity: Sensitivity) -> (KvPrecision, KvPrecision) {
    match sensitivity {
        Sensitivity::High => (KvPrecision::Fp16, KvPrecision::Fp16),
        Sensitivity::Medium => (KvPrecision::Fp16, KvPrecision::Int8),
        Sensitivity::Low => (KvPrecision::Int8, KvPrecision::Int8),
    }
}

/// Score a single layer; fall back to pure-position sensitivity when
/// weights are unavailable.
fn score_layer(
    layer: u32,
    cfg: &KvQuantConfig,
    wk: Option<&WeightEntry>,
    wv: Option<&WeightEntry>,
) -> f64 {
    let pos = position_factor(layer, cfg.n_layers);
    // Spectral contribution ∈ [0, 1].  With no weights we weight pure
    // position.
    let spectral = match (wk, wv) {
        (Some(k), Some(v)) => 0.5 * (spectral_energy_proxy(k) + spectral_energy_proxy(v)),
        (Some(k), None) => spectral_energy_proxy(k),
        (None, Some(v)) => spectral_energy_proxy(v),
        (None, None) => 0.5, // unknown → medium
    };
    // Final score: position weight dominates, spectral is a refinement.
    0.7 * pos + 0.3 * spectral
}

fn find_weight<'a>(wm: &'a WeightMap, layer: u32, suffixes: &[&str]) -> Option<&'a WeightEntry> {
    for prefix in [
        format!("blocks.{layer}."),
        format!("layers.{layer}."),
        format!("h.{layer}."),
    ] {
        for suf in suffixes {
            let name = format!("{prefix}{suf}");
            if let Some(e) = wm.get(&name) {
                return Some(e);
            }
        }
    }
    None
}

/// Build the full per-layer plan.
pub fn plan(cfg: &KvQuantConfig, weights: Option<&WeightMap>) -> KvQuantPlan {
    let mut layers = Vec::with_capacity(cfg.n_layers as usize);
    let bytes_per_head_token = (cfg.n_kv_heads as u64) * (cfg.head_dim as u64);
    let mut total_selected_bytes = 0u64;
    let total_fp16_bytes = bytes_per_head_token * 2 /* K+V */ * 2 /* FP16 */;
    let mut cumulative_fp16 = 0u64;

    for layer in 0..cfg.n_layers {
        let (wk, wv) = match weights {
            Some(wm) => (
                find_weight(wm, layer, &["attn.wk", "attn.W_k", "k_proj.weight", "wk"]),
                find_weight(wm, layer, &["attn.wv", "attn.W_v", "v_proj.weight", "wv"]),
            ),
            None => (None, None),
        };
        let score = score_layer(layer, cfg, wk, wv);
        let sens = classify(score, cfg);
        let (kp, vp) = precisions_for(sens);
        let rationale = if weights.is_some() {
            format!(
                "layer {layer}: score={score:.3} → {} (K={}, V={})",
                match sens {
                    Sensitivity::High => "high",
                    Sensitivity::Medium => "medium",
                    Sensitivity::Low => "low",
                },
                kp.as_str(),
                vp.as_str()
            )
        } else {
            format!(
                "layer {layer}: no weights → position score={score:.2} → {}",
                match sens {
                    Sensitivity::High => "FP16 K+V",
                    Sensitivity::Medium => "FP16 K, INT8 V",
                    Sensitivity::Low => "INT8 K+V",
                }
            )
        };
        total_selected_bytes += bytes_per_head_token
            * (kp.byte_width() as u64 + vp.byte_width() as u64);
        cumulative_fp16 += total_fp16_bytes;
        layers.push(LayerKvDecision {
            layer,
            k_precision: kp,
            v_precision: vp,
            sensitivity: sens,
            sensitivity_score: score,
            rationale,
        });
    }

    KvQuantPlan {
        layers,
        bytes_per_token_uniform_fp16: cumulative_fp16,
        bytes_per_token_selected: total_selected_bytes,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_factor_peaks_at_endpoints() {
        assert!((position_factor(0, 8) - 1.0).abs() < 1e-9);
        assert!((position_factor(7, 8) - 1.0).abs() < 1e-9);
        assert!(position_factor(3, 8) < 0.5);
    }

    #[test]
    fn plan_without_weights_uses_position_heuristic() {
        let cfg = KvQuantConfig::new(8, 4, 64);
        let plan = plan(&cfg, None);
        assert_eq!(plan.layers.len(), 8);
        // First and last layer must be High sensitivity.
        assert_eq!(plan.layers[0].sensitivity, Sensitivity::High);
        assert_eq!(plan.layers[7].sensitivity, Sensitivity::High);
        // Middle layer should not be High.
        assert_ne!(plan.layers[3].sensitivity, Sensitivity::High);
    }

    #[test]
    fn savings_vs_uniform_fp16_are_positive() {
        let cfg = KvQuantConfig::new(8, 4, 64);
        let plan = plan(&cfg, None);
        assert!(plan.memory_savings_ratio() > 0.0);
    }

    #[test]
    fn int8_layers_counted_correctly() {
        let cfg = KvQuantConfig::new(8, 4, 64);
        let plan = plan(&cfg, None);
        assert_eq!(plan.int8_layer_count(), plan.layers.iter().filter(|l| matches!(l.k_precision, KvPrecision::Int8)).count());
        assert!(plan.int8_layer_count() > 0);
    }

    #[test]
    fn single_layer_is_always_high_sensitivity() {
        let cfg = KvQuantConfig::new(1, 4, 64);
        let plan = plan(&cfg, None);
        assert_eq!(plan.layers[0].sensitivity, Sensitivity::High);
    }

    #[test]
    fn thresholds_swap_behaviour() {
        let mut cfg = KvQuantConfig::new(8, 4, 64);
        cfg.high_threshold = 0.0;
        cfg.low_threshold = 0.0;
        let plan = plan(&cfg, None);
        // Everything above 0 → all High.
        for l in &plan.layers {
            assert_eq!(l.sensitivity, Sensitivity::High);
        }
    }

    #[test]
    fn byte_widths_match_declared_precision() {
        assert_eq!(KvPrecision::Fp16.byte_width(), 2);
        assert_eq!(KvPrecision::Bf16.byte_width(), 2);
        assert_eq!(KvPrecision::Int8.byte_width(), 1);
    }

    #[test]
    fn rationale_contains_layer_number() {
        let cfg = KvQuantConfig::new(8, 4, 64);
        let plan = plan(&cfg, None);
        for (i, l) in plan.layers.iter().enumerate() {
            assert!(l.rationale.contains(&format!("layer {i}")));
        }
    }
}
