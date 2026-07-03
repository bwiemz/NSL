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
//! When weight data is available we perform *real* spectral analysis:
//! the top singular value (sigma_max) of each layer's K-projection and
//! V-projection weight is computed via the deterministic randomized SVD
//! in [`crate::wrga_spectral`] (fixed seed → identical results across
//! runs).  Raw sigma_max values are normalised **per metric** (K-proj
//! sigmas across layers, V-proj sigmas across layers) with min-max
//! scaling to [0, 1]; when every measured value is equal (degenerate
//! spectrum) the normalised value is defined as 0.5.
//!
//! With weights present, the measured spectral score dominates:
//! `score = 0.7 * spectral + 0.3 * position`.  Without weights (or for
//! a layer whose projections cannot be found by name) we fall back to
//! the position-based prior: `score = 0.7 * position + 0.3 * 0.5`.
//!
//! Note on the "first RoPE layer": true RoPE-awareness requires model
//! IR that is not available at this planning layer, so layer 0 is used
//! as a *positional proxy* for the first RoPE layer.  The position
//! factor peaks at both ends of the stack (first + final layer).

use serde::Serialize;

use crate::weight_aware::{WeightEntry, WeightMap};
use crate::wrga_spectral::randomized_svd;

/// Fixed seed for the randomized SVD so two plan() invocations over the
/// same weights produce bit-identical sensitivity scores.
const SPECTRAL_SEED: u64 = 0xCF1E_5EED;

/// Blend weights when a measured spectral score is available: measured
/// spectral dominates, position is a refinement.
const SPECTRAL_WEIGHT: f64 = 0.7;
const POSITION_WEIGHT: f64 = 0.3;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum KvPrecision {
    Fp16,
    Bf16,
    Int8,
    Int4,
}

impl KvPrecision {
    /// Whole bytes for a SINGLE element.  Int4 packs two elements per
    /// byte, so a single element still costs one byte here (you cannot
    /// store half a byte in isolation) — this is the coarse per-element
    /// figure.  For accurate multi-element sizing use
    /// [`Self::bytes_for_elems`], which is half-byte exact for Int4.
    pub fn byte_width(self) -> u32 {
        match self {
            KvPrecision::Fp16 | KvPrecision::Bf16 => 2,
            KvPrecision::Int8 => 1,
            KvPrecision::Int4 => 1, // one element rounds up to a byte
        }
    }

    /// Exact byte footprint of `n` elements at this precision (G24b).
    /// Int4 packs two 4-bit elements per byte, so `n` elements take
    /// `ceil(n / 2)` bytes; every other precision is `n * byte_width`.
    /// This is the figure the KV byte accounting and pool sizing must
    /// use so an odd element count is charged the right half-byte.
    pub fn bytes_for_elems(self, n: u64) -> u64 {
        match self {
            KvPrecision::Int4 => n.div_ceil(2),
            _ => n * (self.byte_width() as u64),
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
        // Thresholds tuned against both scoring paths:
        //   * No weights (positional prior, 0.7 pos + 0.3 × 0.5):
        //     middle layers land around 0.25, endpoints at 0.85 —
        //     `low_threshold = 0.3` cleanly puts the middle below the
        //     Medium band while endpoints clear `high_threshold`.
        //   * Weights present (0.7 spectral + 0.3 pos): a layer whose
        //     normalised sigma_max is ~1.0 scores >= 0.7 (High) even in
        //     the middle of the stack; a ~0.0 layer scores <= 0.3 (Low
        //     in the middle, Medium at the endpoints where position
        //     alone contributes 0.3).
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
    // survive at FP16 under the positional prior.  Layer 0 doubles as
    // the positional proxy for the first RoPE layer (real RoPE
    // detection needs model IR, out of scope here).  Normalised to
    // [0, 1] with peaks at 1.0.
    let t = layer as f64 / (n_layers - 1) as f64;
    (2.0 * t - 1.0).abs()
}

/// Top singular value of a projection weight via the deterministic
/// randomized SVD in `wrga_spectral`.  Tensors with ndim > 2 are viewed
/// as `[dim0, prod(rest)]` (same convention as `wrga_spectral`).
/// Returns `None` when the entry is not a usable matrix.
fn sigma_max(entry: &WeightEntry) -> Option<f64> {
    if entry.shape.len() < 2 {
        return None;
    }
    let m = entry.shape[0];
    let n: usize = entry.shape[1..].iter().product::<usize>().max(1);
    if m == 0 || n == 0 {
        return None;
    }
    let bw = entry.dtype.byte_width();
    if entry.data.len() < m * n * bw {
        return None;
    }
    let mut mat = Vec::with_capacity(m * n);
    for idx in 0..(m * n) {
        let off = idx * bw;
        mat.push(entry.dtype.to_f64(&entry.data[off..off + bw]));
    }
    let sv = randomized_svd(&mat, m, n, 1, 8, SPECTRAL_SEED);
    sv.first().copied().filter(|s| s.is_finite())
}

/// Min-max normalise the measured values across layers to [0, 1].
/// `None` entries (layer had no weight) stay `None`.  Degenerate case
/// (all measured values equal, including a single measurement) maps
/// every measured value to 0.5.
fn min_max_normalize(raw: &[Option<f64>]) -> Vec<Option<f64>> {
    let measured: Vec<f64> = raw.iter().filter_map(|v| *v).collect();
    if measured.is_empty() {
        return raw.to_vec();
    }
    let min = measured.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = measured.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if (max - min).abs() < 1e-12 {
        return raw.iter().map(|v| v.map(|_| 0.5)).collect();
    }
    raw.iter()
        .map(|v| v.map(|x| (x - min) / (max - min)))
        .collect()
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

/// Position-only fallback score used when no projection weights are
/// available for a layer (or no weights at all): the unknown spectral
/// contribution is pinned at the medium value 0.5.
fn positional_fallback_score(layer: u32, n_layers: u32) -> f64 {
    0.7 * position_factor(layer, n_layers) + 0.3 * 0.5
}

/// Locate layer `layer`'s K- or V-projection weight by name.
///
/// `kind` must be `"k_proj"` or `"v_proj"` (anything else returns
/// `None`).  Accepts the common checkpoint naming families:
///
///   * `layers.{i}.self_attn.k_proj.weight`          (Llama-style)
///   * `model.layers.{i}.self_attn.k_proj.weight`    (HF-prefixed)
///   * `blocks.{i}.attn.k_proj.weight`               (block-style)
///   * `transformer.h.{i}.attn.k_proj.weight`        (GPT-2-style)
///   * legacy NSL names: `attn.wk` / `attn.W_k` / `wk` under any of the
///     block prefixes above (and `wv` variants for `"v_proj"`).
pub fn find_layer_proj<'a>(
    wm: &'a WeightMap,
    layer: u32,
    kind: &str,
) -> Option<&'a WeightEntry> {
    let suffixes: &[&str] = match kind {
        "k_proj" => &[
            "self_attn.k_proj.weight",
            "attn.k_proj.weight",
            "k_proj.weight",
            "attn.wk",
            "attn.W_k",
            "wk",
        ],
        "v_proj" => &[
            "self_attn.v_proj.weight",
            "attn.v_proj.weight",
            "v_proj.weight",
            "attn.wv",
            "attn.W_v",
            "wv",
        ],
        _ => return None,
    };
    let block_prefixes = [
        format!("layers.{layer}."),
        format!("blocks.{layer}."),
        format!("h.{layer}."),
    ];
    for model_prefix in ["", "model.", "transformer."] {
        for block_prefix in &block_prefixes {
            for suf in suffixes {
                let name = format!("{model_prefix}{block_prefix}{suf}");
                if let Some(e) = wm.get(&name) {
                    return Some(e);
                }
            }
        }
    }
    None
}

/// Build the full per-layer plan.
///
/// With `weights: Some(..)` each layer's sensitivity is
/// `0.7 * spectral + 0.3 * position`, where `spectral` is the mean of
/// the layer's per-metric min-max-normalised sigma_max values (K-proj
/// and V-proj, whichever are present).  Layers whose projections cannot
/// be found — and the whole plan when `weights: None` — use the
/// position-based prior instead.
pub fn plan(cfg: &KvQuantConfig, weights: Option<&WeightMap>) -> KvQuantPlan {
    let n = cfg.n_layers as usize;

    // Pass 1: measure raw sigma_max per layer and metric.
    let mut k_raw: Vec<Option<f64>> = vec![None; n];
    let mut v_raw: Vec<Option<f64>> = vec![None; n];
    if let Some(wm) = weights {
        for layer in 0..cfg.n_layers {
            if let Some(e) = find_layer_proj(wm, layer, "k_proj") {
                k_raw[layer as usize] = sigma_max(e);
            }
            if let Some(e) = find_layer_proj(wm, layer, "v_proj") {
                v_raw[layer as usize] = sigma_max(e);
            }
        }
    }
    // Pass 2: normalise each metric across layers.
    let k_norm = min_max_normalize(&k_raw);
    let v_norm = min_max_normalize(&v_raw);

    let mut layers = Vec::with_capacity(n);
    let bytes_per_head_token = (cfg.n_kv_heads as u64) * (cfg.head_dim as u64);
    let mut total_selected_bytes = 0u64;
    let total_fp16_bytes = bytes_per_head_token * 2 /* K+V */ * 2 /* FP16 */;
    let mut cumulative_fp16 = 0u64;

    for layer in 0..cfg.n_layers {
        let i = layer as usize;
        let pos = position_factor(layer, cfg.n_layers);
        let spectral = match (k_norm[i], v_norm[i]) {
            (Some(k), Some(v)) => Some(0.5 * (k + v)),
            (Some(k), None) => Some(k),
            (None, Some(v)) => Some(v),
            (None, None) => None,
        };
        let (score, rationale) = match spectral {
            Some(s) => {
                let score = SPECTRAL_WEIGHT * s + POSITION_WEIGHT * pos;
                let sens = classify(score, cfg);
                let (kp, vp) = precisions_for(sens);
                (
                    score,
                    format!(
                        "layer {layer}: spectral={s:.3} pos={pos:.3} score={score:.3} → {} (K={}, V={})",
                        sens_str(sens),
                        kp.as_str(),
                        vp.as_str()
                    ),
                )
            }
            None => {
                let score = positional_fallback_score(layer, cfg.n_layers);
                let sens = classify(score, cfg);
                let msg = if weights.is_some() {
                    format!(
                        "layer {layer}: no k/v projection weights found → positional fallback score={score:.2} → {}",
                        fallback_sens_str(sens)
                    )
                } else {
                    format!(
                        "layer {layer}: no weights → position score={score:.2} → {}",
                        fallback_sens_str(sens)
                    )
                };
                (score, msg)
            }
        };
        let sens = classify(score, cfg);
        let (kp, vp) = precisions_for(sens);
        // G24b: half-byte-exact sizing — Int4 K/V halves charge
        // ceil(n/2) bytes, not one byte per element.
        total_selected_bytes +=
            kp.bytes_for_elems(bytes_per_head_token) + vp.bytes_for_elems(bytes_per_head_token);
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

fn sens_str(s: Sensitivity) -> &'static str {
    match s {
        Sensitivity::High => "high",
        Sensitivity::Medium => "medium",
        Sensitivity::Low => "low",
    }
}

fn fallback_sens_str(s: Sensitivity) -> &'static str {
    match s {
        Sensitivity::High => "FP16 K+V",
        Sensitivity::Medium => "FP16 K, INT8 V",
        Sensitivity::Low => "INT8 K+V",
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;
    use crate::weight_aware::WeightDType;

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
        // Paper pattern: first and last layer FP16 K+V.
        assert_eq!(plan.layers[0].sensitivity, Sensitivity::High);
        assert_eq!(plan.layers[7].sensitivity, Sensitivity::High);
        assert_eq!(plan.layers[0].k_precision, KvPrecision::Fp16);
        assert_eq!(plan.layers[0].v_precision, KvPrecision::Fp16);
        assert_eq!(plan.layers[7].k_precision, KvPrecision::Fp16);
        // Innermost layers: INT8 K+V; shoulders (2, 5) at most Medium.
        for i in 3..=4 {
            assert_eq!(plan.layers[i].sensitivity, Sensitivity::Low, "layer {i}");
            assert_eq!(plan.layers[i].k_precision, KvPrecision::Int8);
            assert_eq!(plan.layers[i].v_precision, KvPrecision::Int8);
        }
        assert_ne!(plan.layers[2].sensitivity, Sensitivity::High);
        assert_ne!(plan.layers[5].sensitivity, Sensitivity::High);
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
    fn bytes_for_elems_is_half_byte_exact_for_int4() {
        // G24b: Int4 packs two elements per byte -> ceil(n/2).
        assert_eq!(KvPrecision::Int4.bytes_for_elems(0), 0);
        assert_eq!(KvPrecision::Int4.bytes_for_elems(1), 1); // odd: rounds up
        assert_eq!(KvPrecision::Int4.bytes_for_elems(2), 1);
        assert_eq!(KvPrecision::Int4.bytes_for_elems(3), 2); // odd
        assert_eq!(KvPrecision::Int4.bytes_for_elems(4), 2);
        assert_eq!(KvPrecision::Int4.bytes_for_elems(511), 256); // odd
        assert_eq!(KvPrecision::Int4.bytes_for_elems(512), 256);
        // Other precisions are exactly n * byte_width.
        assert_eq!(KvPrecision::Int8.bytes_for_elems(7), 7);
        assert_eq!(KvPrecision::Fp16.bytes_for_elems(7), 14);
        assert_eq!(KvPrecision::Bf16.bytes_for_elems(7), 14);
    }

    #[test]
    fn int4_never_over_counts_vs_int8() {
        // For any element count, Int4 must cost <= Int8 (half or one
        // fewer byte on odd counts), and roughly half.
        for n in [1u64, 2, 3, 8, 63, 64, 129, 256, 1000] {
            let i4 = KvPrecision::Int4.bytes_for_elems(n);
            let i8 = KvPrecision::Int8.bytes_for_elems(n);
            assert!(i4 <= i8, "Int4 {i4} must not exceed Int8 {i8} for n={n}");
            assert_eq!(i4, n.div_ceil(2));
        }
    }

    #[test]
    fn rationale_contains_layer_number() {
        let cfg = KvQuantConfig::new(8, 4, 64);
        let plan = plan(&cfg, None);
        for (i, l) in plan.layers.iter().enumerate() {
            assert!(l.rationale.contains(&format!("layer {i}")));
        }
    }

    // -- Synthetic-weight helpers ------------------------------------------

    const DIM: usize = 8;

    fn entry_f32(name: &str, vals: &[f32]) -> WeightEntry {
        let mut data = Vec::with_capacity(vals.len() * 4);
        for v in vals {
            data.extend_from_slice(&v.to_le_bytes());
        }
        WeightEntry {
            name: name.to_string(),
            data,
            shape: vec![DIM, DIM],
            dtype: WeightDType::F32,
            num_elements: DIM * DIM,
            sparsity: None,
            eliminated: false,
        }
    }

    /// DIM×DIM matrix: deterministic small noise plus `diag` on the
    /// diagonal, so sigma_max ~ diag and the matrix is full rank.
    fn matrix(diag: f32) -> Vec<f32> {
        let mut vals = vec![0.0f32; DIM * DIM];
        for (idx, v) in vals.iter_mut().enumerate() {
            let (i, j) = (idx / DIM, idx % DIM);
            *v = (((i * 31 + j * 17) % 13) as f32 - 6.0) * 0.001;
            if i == j {
                *v += diag;
            }
        }
        vals
    }

    fn wm_from(entries: Vec<WeightEntry>) -> WeightMap {
        let map: HashMap<String, WeightEntry> =
            entries.into_iter().map(|e| (e.name.clone(), e)).collect();
        WeightMap::from_entries(map)
    }

    /// Build a full weight map for `n_layers` layers where `sigma(i)`
    /// gives the diagonal magnitude of layer i's K and V projections.
    fn synthetic_weights(n_layers: u32, sigma: impl Fn(u32) -> f32) -> WeightMap {
        let mut entries = Vec::new();
        for i in 0..n_layers {
            let s = sigma(i);
            entries.push(entry_f32(
                &format!("layers.{i}.self_attn.k_proj.weight"),
                &matrix(s),
            ));
            entries.push(entry_f32(
                &format!("layers.{i}.self_attn.v_proj.weight"),
                &matrix(s),
            ));
        }
        wm_from(entries)
    }

    // -- find_layer_proj ----------------------------------------------------

    #[test]
    fn find_layer_proj_matches_common_families() {
        let names = [
            "layers.0.self_attn.k_proj.weight",
            "model.layers.1.self_attn.k_proj.weight",
            "blocks.2.attn.k_proj.weight",
            "transformer.h.3.attn.k_proj.weight",
            "layers.4.attn.wk",
        ];
        for name in names {
            let wm = wm_from(vec![entry_f32(name, &matrix(1.0))]);
            let layer: u32 = name
                .chars()
                .find(|c| c.is_ascii_digit())
                .unwrap()
                .to_digit(10)
                .unwrap();
            let found = find_layer_proj(&wm, layer, "k_proj");
            assert!(found.is_some(), "pattern not matched: {name}");
            assert_eq!(found.unwrap().name, name);
            // Wrong layer index must not match.
            assert!(find_layer_proj(&wm, layer + 1, "k_proj").is_none());
        }
    }

    #[test]
    fn find_layer_proj_v_proj_and_unknown_kind() {
        let wm = wm_from(vec![
            entry_f32("model.layers.0.self_attn.v_proj.weight", &matrix(1.0)),
            entry_f32("blocks.1.attn.wv", &matrix(1.0)),
        ]);
        assert!(find_layer_proj(&wm, 0, "v_proj").is_some());
        assert!(find_layer_proj(&wm, 1, "v_proj").is_some());
        // k_proj lookup must not return a v_proj weight.
        assert!(find_layer_proj(&wm, 0, "k_proj").is_none());
        // Unknown kinds return None rather than guessing.
        assert!(find_layer_proj(&wm, 0, "q_proj").is_none());
    }

    // -- weights-present spectral scoring ------------------------------------

    #[test]
    fn high_sigma_middle_layer_earns_fp16() {
        let cfg = KvQuantConfig::new(8, 4, 64);
        // All layers low-sigma except middle layer 3.
        let wm = synthetic_weights(8, |i| if i == 3 { 10.0 } else { 0.05 });
        let plan = plan(&cfg, Some(&wm));
        // Layer 3: normalised spectral ~1.0 → 0.7*1.0 + 0.3*pos ≥ 0.7 → High.
        assert_eq!(plan.layers[3].sensitivity, Sensitivity::High);
        assert_eq!(plan.layers[3].k_precision, KvPrecision::Fp16);
        assert_eq!(plan.layers[3].v_precision, KvPrecision::Fp16);
        // A low-sigma middle layer: spectral ~0 → 0.3*pos < 0.3 → Low → INT8.
        assert_eq!(plan.layers[4].sensitivity, Sensitivity::Low);
        assert_eq!(plan.layers[4].k_precision, KvPrecision::Int8);
        assert_eq!(plan.layers[4].v_precision, KvPrecision::Int8);
        // Measured spectral dominates: a low-sigma endpoint is no longer
        // forced High by position alone (0.3*1.0 = 0.3 → Medium).
        assert_ne!(plan.layers[0].sensitivity, Sensitivity::High);
        assert_eq!(plan.layers[0].k_precision, KvPrecision::Fp16);
    }

    #[test]
    fn degenerate_equal_sigmas_normalize_to_half() {
        let cfg = KvQuantConfig::new(8, 4, 64);
        // Every layer identical → min-max degenerate → spectral = 0.5.
        let wm = synthetic_weights(8, |_| 1.0);
        let plan = plan(&cfg, Some(&wm));
        // score = 0.7*0.5 + 0.3*pos: endpoints 0.65 (High), middle ~0.39 (Medium).
        assert_eq!(plan.layers[0].sensitivity, Sensitivity::High);
        assert_eq!(plan.layers[7].sensitivity, Sensitivity::High);
        assert_eq!(plan.layers[3].sensitivity, Sensitivity::Medium);
        for l in &plan.layers {
            assert!(
                (l.sensitivity_score - (0.7 * 0.5 + 0.3 * position_factor(l.layer, 8))).abs()
                    < 1e-9,
                "layer {} score {}",
                l.layer,
                l.sensitivity_score
            );
        }
    }

    #[test]
    fn missing_projections_fall_back_to_position() {
        let cfg = KvQuantConfig::new(8, 4, 64);
        // Projections only for layers 0..4; layers 4..8 absent.
        let mut entries = Vec::new();
        for i in 0..4u32 {
            let s = if i == 2 { 10.0 } else { 0.05 };
            entries.push(entry_f32(
                &format!("layers.{i}.self_attn.k_proj.weight"),
                &matrix(s),
            ));
            entries.push(entry_f32(
                &format!("layers.{i}.self_attn.v_proj.weight"),
                &matrix(s),
            ));
        }
        let wm = wm_from(entries);
        let with_weights = plan(&cfg, Some(&wm));
        let without = plan(&cfg, None);
        // Layers with no projections score exactly like the no-weights prior.
        for i in 4..8 {
            assert!(
                (with_weights.layers[i].sensitivity_score
                    - without.layers[i].sensitivity_score)
                    .abs()
                    < 1e-12,
                "layer {i} fallback score mismatch"
            );
            assert!(
                with_weights.layers[i].rationale.contains("positional fallback"),
                "layer {i} rationale: {}",
                with_weights.layers[i].rationale
            );
        }
        // Measured layers still get spectral treatment.
        assert_eq!(with_weights.layers[2].sensitivity, Sensitivity::High);
        assert!(with_weights.layers[2].rationale.contains("spectral"));
    }

    #[test]
    fn k_only_projection_uses_k_metric() {
        let cfg = KvQuantConfig::new(8, 4, 64);
        // Only K projections present (V missing everywhere).
        let mut entries = Vec::new();
        for i in 0..8u32 {
            let s = if i == 3 { 10.0 } else { 0.05 };
            entries.push(entry_f32(
                &format!("layers.{i}.self_attn.k_proj.weight"),
                &matrix(s),
            ));
        }
        let wm = wm_from(entries);
        let plan = plan(&cfg, Some(&wm));
        assert_eq!(plan.layers[3].sensitivity, Sensitivity::High);
        assert_eq!(plan.layers[4].sensitivity, Sensitivity::Low);
    }

    #[test]
    fn spectral_plan_is_deterministic_across_runs() {
        let cfg = KvQuantConfig::new(8, 4, 64);
        let wm = synthetic_weights(8, |i| 0.1 + 0.5 * i as f32);
        let a = plan(&cfg, Some(&wm));
        let b = plan(&cfg, Some(&wm));
        for (la, lb) in a.layers.iter().zip(b.layers.iter()) {
            // Bit-identical: randomized SVD runs with a fixed seed.
            assert_eq!(
                la.sensitivity_score.to_bits(),
                lb.sensitivity_score.to_bits(),
                "layer {} nondeterministic",
                la.layer
            );
            assert_eq!(la.sensitivity, lb.sensitivity);
        }
    }

    #[test]
    fn sigma_max_refuses_non_matrices() {
        // 1-D tensor → None (fallback path, no panic).
        let mut e = entry_f32("layers.0.self_attn.k_proj.weight", &matrix(1.0));
        e.shape = vec![DIM * DIM];
        assert!(sigma_max(&e).is_none());
        // Truncated data → None.
        let mut e2 = entry_f32("layers.0.self_attn.k_proj.weight", &matrix(1.0));
        e2.data.truncate(8);
        assert!(sigma_max(&e2).is_none());
    }

    #[test]
    fn sigma_max_tracks_dominant_singular_value() {
        let big = entry_f32("w", &matrix(10.0));
        let small = entry_f32("w", &matrix(0.05));
        let sb = sigma_max(&big).unwrap();
        let ss = sigma_max(&small).unwrap();
        assert!(sb > 9.0 && sb < 11.0, "sigma_max(big) = {sb}");
        assert!(ss < 0.2, "sigma_max(small) = {ss}");
    }
}
