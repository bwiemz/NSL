//! CPDT — per-parameter optimizer precision selection.
//!
//! Implements §4 of the paper.  Given weight-aware metrics, assigns each
//! parameter tensor a precision from a four-tier table:
//!
//!   * **High sensitivity** → FP32 m + FP32 v  (8 bytes/param)
//!   * **Medium**           → FP16 m + FP32 v  (6 bytes/param)
//!   * **Low**              → INT8 m + FP16 v  (3 bytes/param)
//!   * **Very low**         → INT8 m + INT8 v  (2 bytes/param)
//!
//! Sensitivity formula:
//!
//! ```text
//! spectral_condition × gradient_magnitude × position_criticality
//! / element_count
//! ```
//!
//! Every factor is computable from pre-trained weights + compilation
//! analysis — no training step required.

use serde::Serialize;

use crate::weight_aware::{WeightEntry, WeightMap};

/// Four supported optimizer-state precisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum OptimPrecision {
    Fp32,
    Fp16,
    Int8,
}

impl OptimPrecision {
    pub fn bytes(self) -> u32 {
        match self {
            OptimPrecision::Fp32 => 4,
            OptimPrecision::Fp16 => 2,
            OptimPrecision::Int8 => 1,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            OptimPrecision::Fp32 => "fp32",
            OptimPrecision::Fp16 => "fp16",
            OptimPrecision::Int8 => "int8",
        }
    }
}

/// Sensitivity tier → precision pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SensitivityTier {
    /// Norms, embeddings, first/last layer.  Full FP32 m + v.
    High,
    /// Attention projections.  FP16 m, FP32 v.
    Medium,
    /// Middle-layer FFN.  INT8 m, FP16 v.
    Low,
    /// Provably-stable parameters (very low sensitivity).  INT8 m + v.
    VeryLow,
}

impl SensitivityTier {
    pub fn precision(self) -> (OptimPrecision, OptimPrecision) {
        match self {
            SensitivityTier::High => (OptimPrecision::Fp32, OptimPrecision::Fp32),
            SensitivityTier::Medium => (OptimPrecision::Fp16, OptimPrecision::Fp32),
            SensitivityTier::Low => (OptimPrecision::Int8, OptimPrecision::Fp16),
            SensitivityTier::VeryLow => (OptimPrecision::Int8, OptimPrecision::Int8),
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SensitivityTier::High => "high",
            SensitivityTier::Medium => "medium",
            SensitivityTier::Low => "low",
            SensitivityTier::VeryLow => "very_low",
        }
    }
}

/// Per-parameter precision decision.
#[derive(Debug, Clone, Serialize)]
pub struct ParamPrecision {
    pub name: String,
    pub layer: Option<u32>,
    pub tier: SensitivityTier,
    pub m_precision: OptimPrecision,
    pub v_precision: OptimPrecision,
    /// Whether the parameter needs stochastic rounding (embeddings).
    pub stochastic_rounding: bool,
    pub sensitivity_score: f64,
    pub param_bytes: u64,
    pub optim_bytes: u64,
}

/// Aggregate precision plan.
#[derive(Debug, Clone, Default, Serialize)]
pub struct PrecisionPlan {
    pub params: Vec<ParamPrecision>,
    /// Total optimizer-state bytes across all parameters.
    pub total_optim_bytes: u64,
    /// Baseline (all-FP32) optimizer-state bytes for the same weights.
    pub baseline_fp32_bytes: u64,
}

impl PrecisionPlan {
    pub fn savings_ratio(&self) -> f64 {
        if self.baseline_fp32_bytes == 0 {
            return 0.0;
        }
        1.0 - (self.total_optim_bytes as f64 / self.baseline_fp32_bytes as f64)
    }

    pub fn tier_counts(&self) -> (usize, usize, usize, usize) {
        let mut h = 0;
        let mut m = 0;
        let mut l = 0;
        let mut v = 0;
        for p in &self.params {
            match p.tier {
                SensitivityTier::High => h += 1,
                SensitivityTier::Medium => m += 1,
                SensitivityTier::Low => l += 1,
                SensitivityTier::VeryLow => v += 1,
            }
        }
        (h, m, l, v)
    }
}

/// User-tunable thresholds.
#[derive(Debug, Clone)]
pub struct PrecisionConfig {
    pub high_threshold: f64,
    pub medium_threshold: f64,
    pub low_threshold: f64,
    /// When `true`, embedding tensors always get stochastic rounding —
    /// Q-Adam-mini showed this is required for INT8 stability on
    /// embeddings.
    pub embedding_stochastic_rounding: bool,
    /// Total layer count (for position-criticality computation).
    pub n_layers: u32,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            high_threshold: 0.5,
            medium_threshold: 0.1,
            low_threshold: 0.02,
            embedding_stochastic_rounding: true,
            n_layers: 8,
        }
    }
}

// ---------------------------------------------------------------------------
// Scoring primitives
// ---------------------------------------------------------------------------

/// Approximate the spectral condition number via the Frobenius norm
/// divided by an estimate of the smallest singular value.  Cheap
/// power-iteration-free proxy: `‖W‖_F / ‖W‖_max`.  High values indicate
/// an ill-conditioned matrix.
fn spectral_condition_proxy(entry: &WeightEntry) -> f64 {
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
    sum_sq.sqrt() / max_abs
}

/// Gradient-magnitude estimator: for a pre-trained model, the magnitude
/// of updates is well-approximated by the weight's L2 norm divided by
/// the number of elements (a "per-element energy" measure).
fn gradient_magnitude_estimate(entry: &WeightEntry) -> f64 {
    if entry.num_elements == 0 {
        return 0.0;
    }
    let bw = entry.dtype.byte_width();
    let mut sum_sq = 0.0_f64;
    for i in 0..entry.num_elements {
        let off = i * bw;
        if off + bw > entry.data.len() {
            break;
        }
        let v = entry.dtype.to_f64(&entry.data[off..off + bw]);
        sum_sq += v * v;
    }
    (sum_sq / entry.num_elements as f64).sqrt()
}

/// Position-criticality curve: early (l=0) and late (l=L-1) layers
/// score higher because they're empirically more sensitive to
/// quantisation.  U-shaped, peaks at 1.3 at the endpoints.
pub fn position_criticality(layer: u32, n_layers: u32) -> f64 {
    if n_layers <= 1 {
        return 1.0;
    }
    let normalized = layer as f64 / (n_layers - 1) as f64;
    1.0 + 0.3 * (2.0 * normalized - 1.0).abs()
}

fn layer_of(name: &str) -> Option<u32> {
    for prefix in ["blocks.", "layers.", "h."] {
        if let Some(rest) = name.strip_prefix(prefix) {
            if let Some(end) = rest.find('.') {
                return rest[..end].parse::<u32>().ok();
            }
        }
    }
    None
}

fn is_embedding(name: &str) -> bool {
    let lname = name.to_ascii_lowercase();
    lname.contains("embed") || lname == "tok_embeddings.weight" || lname == "wte.weight"
}

fn is_norm(name: &str) -> bool {
    let lname = name.to_ascii_lowercase();
    lname.contains("norm") && !lname.contains("normalize")
}

fn is_first_or_last_layer(layer: Option<u32>, n_layers: u32) -> bool {
    match layer {
        Some(l) => l == 0 || l + 1 == n_layers,
        None => false,
    }
}

fn sensitivity(
    entry: &WeightEntry,
    layer: Option<u32>,
    n_layers: u32,
) -> f64 {
    let condition = spectral_condition_proxy(entry);
    let grad = gradient_magnitude_estimate(entry);
    let pos = layer
        .map(|l| position_criticality(l, n_layers))
        .unwrap_or(1.5); // unknown layer → treat as borderline high
    let elements = entry.num_elements.max(1) as f64;
    condition * grad * pos / elements.sqrt()
}

fn choose_tier(
    name: &str,
    layer: Option<u32>,
    n_layers: u32,
    score: f64,
    cfg: &PrecisionConfig,
) -> SensitivityTier {
    if is_norm(name) || is_embedding(name) || is_first_or_last_layer(layer, n_layers) {
        return SensitivityTier::High;
    }
    if score >= cfg.high_threshold {
        SensitivityTier::High
    } else if score >= cfg.medium_threshold {
        SensitivityTier::Medium
    } else if score >= cfg.low_threshold {
        SensitivityTier::Low
    } else {
        SensitivityTier::VeryLow
    }
}

/// Score a single parameter tensor and produce its precision decision.
pub fn classify_param(
    entry: &WeightEntry,
    cfg: &PrecisionConfig,
) -> ParamPrecision {
    let layer = layer_of(&entry.name);
    let score = sensitivity(entry, layer, cfg.n_layers);
    let tier = choose_tier(&entry.name, layer, cfg.n_layers, score, cfg);
    let (m, v) = tier.precision();
    let param_bytes = (entry.num_elements as u64) * (entry.dtype.byte_width() as u64);
    let optim_bytes = (entry.num_elements as u64) * (m.bytes() as u64 + v.bytes() as u64);
    let stochastic = cfg.embedding_stochastic_rounding && is_embedding(&entry.name);
    ParamPrecision {
        name: entry.name.clone(),
        layer,
        tier,
        m_precision: m,
        v_precision: v,
        stochastic_rounding: stochastic,
        sensitivity_score: score,
        param_bytes,
        optim_bytes,
    }
}

/// Classify every tensor in a [`WeightMap`] and produce the aggregate
/// plan.
pub fn plan_map(wm: &WeightMap, cfg: &PrecisionConfig) -> PrecisionPlan {
    let mut params = Vec::new();
    let mut total_optim = 0u64;
    let mut baseline_fp32 = 0u64;
    for (_name, entry) in wm.entries() {
        let p = classify_param(entry, cfg);
        total_optim += p.optim_bytes;
        baseline_fp32 += (entry.num_elements as u64) * 8; // FP32 m + v
        params.push(p);
    }
    PrecisionPlan {
        params,
        total_optim_bytes: total_optim,
        baseline_fp32_bytes: baseline_fp32,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_aware::{WeightDType, WeightEntry};

    fn make_f32(name: &str, rows: usize, cols: usize, value: f32) -> WeightEntry {
        let mut data = vec![0u8; rows * cols * 4];
        for i in 0..rows * cols {
            let off = i * 4;
            data[off..off + 4].copy_from_slice(&value.to_le_bytes());
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
    fn optim_precision_byte_widths() {
        assert_eq!(OptimPrecision::Fp32.bytes(), 4);
        assert_eq!(OptimPrecision::Fp16.bytes(), 2);
        assert_eq!(OptimPrecision::Int8.bytes(), 1);
    }

    #[test]
    fn position_criticality_u_shape() {
        assert!((position_criticality(0, 8) - 1.3).abs() < 1e-9);
        assert!((position_criticality(7, 8) - 1.3).abs() < 1e-9);
        assert!(position_criticality(3, 8) < 1.1);
    }

    #[test]
    fn embeddings_get_high_tier_and_stochastic_rounding() {
        let w = make_f32("tok_embeddings.weight", 16, 16, 0.01);
        let decision = classify_param(&w, &PrecisionConfig::default());
        assert_eq!(decision.tier, SensitivityTier::High);
        assert!(decision.stochastic_rounding);
    }

    #[test]
    fn norms_get_high_tier() {
        let w = make_f32("blocks.3.attn_norm.weight", 4, 4, 1.0);
        let decision = classify_param(&w, &PrecisionConfig::default());
        assert_eq!(decision.tier, SensitivityTier::High);
    }

    #[test]
    fn first_layer_forced_high_tier() {
        let w = make_f32("blocks.0.attn.wq.weight", 8, 8, 0.01);
        let cfg = PrecisionConfig { n_layers: 8, ..Default::default() };
        let decision = classify_param(&w, &cfg);
        assert_eq!(decision.tier, SensitivityTier::High);
    }

    #[test]
    fn last_layer_forced_high_tier() {
        let w = make_f32("blocks.7.attn.wq.weight", 8, 8, 0.01);
        let cfg = PrecisionConfig { n_layers: 8, ..Default::default() };
        let decision = classify_param(&w, &cfg);
        assert_eq!(decision.tier, SensitivityTier::High);
    }

    #[test]
    fn middle_layer_low_sensitivity_gets_lower_tier() {
        let w = make_f32("blocks.4.ffn.w_gate.weight", 512, 512, 0.0001);
        let cfg = PrecisionConfig { n_layers: 8, ..Default::default() };
        let decision = classify_param(&w, &cfg);
        assert!(matches!(
            decision.tier,
            SensitivityTier::Low | SensitivityTier::VeryLow | SensitivityTier::Medium
        ));
    }

    /// Helper that mirrors `plan_map` but operates on a slice of
    /// [`WeightEntry`], so the test suite doesn't need to construct a
    /// full [`WeightMap`] — constructing one requires a `load()` from
    /// a real safetensors file.
    fn plan_from_slice(entries: &[WeightEntry], cfg: &PrecisionConfig) -> PrecisionPlan {
        let mut params = Vec::new();
        let mut total_optim = 0u64;
        let mut baseline_fp32 = 0u64;
        for entry in entries {
            let p = classify_param(entry, cfg);
            total_optim += p.optim_bytes;
            baseline_fp32 += (entry.num_elements as u64) * 8;
            params.push(p);
        }
        PrecisionPlan {
            params,
            total_optim_bytes: total_optim,
            baseline_fp32_bytes: baseline_fp32,
        }
    }

    #[test]
    fn plan_aggregate_matches_sum_of_params() {
        let entries = vec![
            make_f32("tok_embeddings.weight", 8, 8, 0.01),
            make_f32("blocks.0.attn.wq.weight", 4, 4, 0.01),
            make_f32("blocks.3.ffn.w_gate.weight", 64, 64, 0.001),
        ];
        let cfg = PrecisionConfig { n_layers: 8, ..Default::default() };
        let plan = plan_from_slice(&entries, &cfg);
        let summed: u64 = plan.params.iter().map(|p| p.optim_bytes).sum();
        assert_eq!(plan.total_optim_bytes, summed);
    }

    #[test]
    fn tier_count_sums_to_param_count() {
        let entries = vec![
            make_f32("tok_embeddings.weight", 4, 4, 0.01),
            make_f32("blocks.2.ffn.w_gate.weight", 16, 16, 0.01),
            make_f32("blocks.3.attn.wq.weight", 16, 16, 0.01),
        ];
        let cfg = PrecisionConfig { n_layers: 8, ..Default::default() };
        let plan = plan_from_slice(&entries, &cfg);
        let (h, m, l, v) = plan.tier_counts();
        assert_eq!(h + m + l + v, plan.params.len());
    }

    #[test]
    fn layer_of_recognises_common_patterns() {
        assert_eq!(layer_of("blocks.6.attn.wq"), Some(6));
        assert_eq!(layer_of("layers.12.norm"), Some(12));
        assert_eq!(layer_of("h.3.mlp.fc"), Some(3));
        assert_eq!(layer_of("embedding.weight"), None);
    }
}
