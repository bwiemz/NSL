//! CPDT Phase 1 — unified sensitivity scorer.
//!
//! ## Formula
//!
//! ```text
//! sensitivity(W, l, kind) = gradient_magnitude_est(W) × position_criticality(l, L)
//!                         / element_count(W)
//! ```
//!
//! Phase 2 adds `spectral_condition(W)` as a fourth multiplicative factor,
//! with a `.cpdt-sensitivity.json` sidecar cache modeled on
//! `wggo_weight_analysis_cache.rs`.
//!
//! ## `ANALYSIS_VERSION` bump rule
//!
//! Any change to the sensitivity formula, factor computation
//! (`gradient_magnitude_est`, `position_criticality`, `element_count`),
//! or tier-boundary policy (`assign_tier`) MUST bump [`ANALYSIS_VERSION`]
//! in the same commit. Phase 2's sidecar-cache key includes this field;
//! caches from older versions are ignored automatically. Phase 2 adds a
//! CI check that flags diffs touching the scoring / tier-boundary
//! functions without a matching bump.

use crate::cpdt_tier_apply::{PrecisionConfig, Tier};
use crate::weight_aware::WeightEntry;

// ---------------------------------------------------------------------------
// ANALYSIS_VERSION
// ---------------------------------------------------------------------------

pub const ANALYSIS_VERSION: u32 = 2;

// ---------------------------------------------------------------------------
// Calibration constants (Phase 1 retune — 2026-04-19)
//
// Emitted by `cpdt_calibrate --emit-calibration` on the pooled corpus
// {calib_tiny, calib_small, calib_medium}. T0/T1/T2 are geomeans of adjacent
// weights-present band mins/maxes; CALIB_K is the log-midpoint of the
// longest K-range where both calib_small and calib_medium disagreement stay
// <= 20% (the monitoring threshold). Residual per-fixture disagreement at
// this K: calib_small 15.91%, calib_medium 2.51%, corpus-wide 3.76%.
//
// The SwiGLU numel-collision floors calib_small disagreement above 5%:
// ffn.w_gate, ffn.w_up, ffn.w_down share numel = d_model * d_ffn, so the
// no-weights formula K * pos / numel cannot separate them. Invariant #7 is
// reframed to monitoring-gate at 20% per spec §5; Phase 2 spectral is the
// intervention that returns it to <5% hard-gate.
//
// Retune design: docs/superpowers/specs/2026-04-19-cpdt-calibration-correction-design.md
// ---------------------------------------------------------------------------

/// Neutral value of `gradient_magnitude_est` when weights are absent.
/// Calibrated by plateau-midpoint selection in `cpdt_calibrate --emit-calibration`.
pub const CALIB_K: f64 = 6.309573e-2;

pub const CALIB_T0: f64 = 6.105631e-8; // High   ↔ Medium
pub const CALIB_T1: f64 = 2.232401e-8; // Medium ↔ Low
pub const CALIB_T2: f64 = 7.255675e-10; // Low   ↔ VeryLow

/// Position-criticality near-extreme boost (for `L ≥ 4`). Unchanged from
/// Phase 1 ship; the retune touches only the sensitivity-band constants.
pub const CALIB_ALPHA: f64 = 0.3;

// ---------------------------------------------------------------------------
// LayerKind
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerKind {
    /// Embedding tables (token or position embeddings). Hard-overridden to `Tier::High`.
    Embedding,
    /// Any form of normalization layer weight. Hard-overridden to `Tier::High`.
    Norm,
    /// First or last decoder block. Hard-overridden to `Tier::High`.
    FirstOrLast,
    /// Any non-overridden parameter; scored via the formula.
    Generic,
}

impl LayerKind {
    /// Layer kinds that bypass formula scoring and land at `Tier::High`.
    pub fn is_kind_overridden(self) -> bool {
        matches!(
            self,
            LayerKind::Embedding | LayerKind::Norm | LayerKind::FirstOrLast
        )
    }
}

// ---------------------------------------------------------------------------
// Unified scorer
// ---------------------------------------------------------------------------

pub struct SensitivityScorer {
    n_layers: u32,
}

impl SensitivityScorer {
    pub fn from_config(cfg: &PrecisionConfig) -> Self {
        Self {
            n_layers: cfg.n_layers,
        }
    }

    /// Score a single weight entry. Returns `(tier, raw_score, layer, kind)`.
    /// Raw score is stored on the resulting `ParamPrecision` for debugging
    /// and future cache keying (Phase 2).
    pub fn score_entry(&self, entry: &WeightEntry) -> (Tier, f64, Option<u32>, LayerKind) {
        let layer = layer_of(&entry.name);
        let kind = classify_layer_kind(&entry.name, layer, self.n_layers);
        let gm = gradient_magnitude_est(Some(entry));
        let pos = position_criticality(layer, self.n_layers, CALIB_ALPHA);
        let elts = entry.num_elements.max(1) as f64;
        let score = gm * pos / elts;
        let tier = assign_tier(score, kind);
        (tier, score, layer, kind)
    }

    /// Score without a live `WeightEntry` — used by `plan_map_noweights`
    /// and the tier-agreement diagnostic in Commit 5.
    pub fn score_optional(
        &self,
        name: &str,
        element_count: usize,
        entry: Option<&WeightEntry>,
    ) -> (Tier, f64, Option<u32>, LayerKind) {
        let layer = layer_of(name);
        let kind = classify_layer_kind(name, layer, self.n_layers);
        let gm = gradient_magnitude_est(entry);
        let pos = position_criticality(layer, self.n_layers, CALIB_ALPHA);
        let elts = element_count.max(1) as f64;
        let score = gm * pos / elts;
        let tier = assign_tier(score, kind);
        (tier, score, layer, kind)
    }
}

// ---------------------------------------------------------------------------
// Formula primitives
// ---------------------------------------------------------------------------

/// Raw RMS magnitude when weights present; calibrated neutral constant when
/// absent. No clamping, no normalization.
pub fn gradient_magnitude_est(entry: Option<&WeightEntry>) -> f64 {
    let Some(w) = entry else {
        return CALIB_K;
    };
    if w.num_elements == 0 {
        return 0.0;
    }
    let bw = w.dtype.byte_width();
    let mut sum_sq = 0.0_f64;
    for i in 0..w.num_elements {
        let off = i * bw;
        if off + bw > w.data.len() {
            break;
        }
        let v = w.dtype.to_f64(&w.data[off..off + bw]);
        sum_sq += v * v;
    }
    (sum_sq / w.num_elements as f64).sqrt()
}

/// Piecewise position criticality with explicit `L < 4` guard.
/// For `L < 4`, the near-extreme branch is definitionally unreachable.
pub fn position_criticality(layer: Option<u32>, n_layers: u32, alpha: f64) -> f64 {
    let Some(l) = layer else {
        return 1.5; // unknown layer → borderline high
    };
    if n_layers == 0 {
        return 1.0;
    }
    let l = l as i64;
    let big_l = n_layers as i64;
    debug_assert!(l >= 0 && l < big_l);
    if l == 0 || l == big_l - 1 {
        return 2.0;
    }
    if big_l >= 4 && (l == 1 || l == big_l - 2) {
        return 1.0 + alpha;
    }
    1.0
}

/// Tier-boundary decision. Layer-kind override fires FIRST — scored value
/// only matters when `kind == LayerKind::Generic`.
pub fn assign_tier(score: f64, kind: LayerKind) -> Tier {
    if kind.is_kind_overridden() {
        return Tier::High;
    }
    if score > CALIB_T0 {
        Tier::High
    } else if score > CALIB_T1 {
        Tier::Medium
    } else if score > CALIB_T2 {
        Tier::Low
    } else {
        Tier::VeryLow
    }
}

// ---------------------------------------------------------------------------
// Name-based layer-kind detection (migrated from cpdt_precision.rs)
// ---------------------------------------------------------------------------

pub fn classify_layer_kind(name: &str, layer: Option<u32>, n_layers: u32) -> LayerKind {
    if is_embedding(name) {
        return LayerKind::Embedding;
    }
    if is_norm(name) {
        return LayerKind::Norm;
    }
    if is_first_or_last_layer(layer, n_layers) {
        return LayerKind::FirstOrLast;
    }
    LayerKind::Generic
}

pub fn layer_of(name: &str) -> Option<u32> {
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

// ---------------------------------------------------------------------------
// Validation pass against AppliedPlan (called from invoke_cpdt_if_enabled)
//
// Phase 1 ships layer-prefix validation only: catches "wrong checkpoint
// entirely" at plan-time by asserting every hierarchical layer (blocks.N /
// layers.N / h.N) has at least one matching tensor in the WeightMap.
// Per-tensor shape/dtype validation is deferred to Phase 2 where the
// spectral-factor wiring provides per-tensor metadata. Until then, the
// MissingTensor / ShapeMismatch / DtypeMismatch variants are unused —
// preserved in the enum so Phase 2 can activate them without schema drift.
// Design: docs/superpowers/specs/2026-04-20-cpdt-validate-body-design.md
// ---------------------------------------------------------------------------

use crate::weight_aware::WeightMap;
use crate::wggo_apply::AppliedPlan;

/// Max number of unique tensor-name prefixes shown in the
/// `LayersMissing` error's WeightMap summary. Keeps the error message
/// diagnostic without dumping hundreds of tensor names on a large checkpoint.
const PREFIX_SUMMARY_MAX: usize = 8;

#[derive(Debug)]
pub enum ValidationError {
    /// Phase 2: a specific tensor declared by the model is absent from
    /// the weight map. Unused in Phase 1.
    MissingTensor { tensor_name: String },
    /// Phase 2: a tensor's declared shape doesn't match the weight map.
    /// Unused in Phase 1.
    ShapeMismatch {
        tensor_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// Phase 2: a tensor's declared dtype doesn't match the weight map.
    /// Unused in Phase 1.
    DtypeMismatch {
        tensor_name: String,
        expected: String,
        actual: String,
    },
    /// Phase 1 (active): one or more AppliedPlan layers have no matching
    /// tensors in the WeightMap. Aggregates every missing layer plus a
    /// summary of the WeightMap's top-level tensor prefixes so the user
    /// can diagnose format mismatches (HuggingFace vs NSL-native, etc.).
    LayersMissing {
        /// Layer names whose prefix-match returned zero tensors.
        missing: Vec<String>,
        /// Total number of hierarchical layers checked (i.e. non-"other").
        total_layers_checked: usize,
        /// Deduplicated, sorted, up-to-`PREFIX_SUMMARY_MAX` entries.
        weight_map_prefix_summary: Vec<String>,
        /// Total tensors in the WeightMap (for the error message's summary line).
        weight_map_total_tensors: usize,
    },
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTensor { tensor_name } => write!(
                f,
                "CPDT weight validation: tensor `{tensor_name}` declared in AppliedPlan but not present in WeightMap",
            ),
            Self::ShapeMismatch { tensor_name, expected, actual } => write!(
                f,
                "CPDT weight validation: tensor `{tensor_name}` shape mismatch — expected {expected:?}, got {actual:?}",
            ),
            Self::DtypeMismatch { tensor_name, expected, actual } => write!(
                f,
                "CPDT weight validation: tensor `{tensor_name}` dtype mismatch — expected {expected}, got {actual}",
            ),
            Self::LayersMissing {
                missing,
                total_layers_checked,
                weight_map_prefix_summary,
                weight_map_total_tensors,
            } => {
                writeln!(
                    f,
                    "CPDT weight validation: weight map does not match model declaration."
                )?;
                writeln!(
                    f,
                    "  Missing layers ({} of {}): {}",
                    missing.len(),
                    total_layers_checked,
                    missing.join(", "),
                )?;
                if weight_map_prefix_summary.is_empty() {
                    write!(
                        f,
                        "  WeightMap contains {} tensors (no top-level prefixes to summarize).",
                        weight_map_total_tensors,
                    )
                } else {
                    write!(
                        f,
                        "  WeightMap contains {} tensors with top-level prefixes: {}",
                        weight_map_total_tensors,
                        weight_map_prefix_summary.join(", "),
                    )
                }
            }
        }
    }
}

impl std::error::Error for ValidationError {}

/// Top-level prefix of a tensor name: substring up to the second dot.
/// `transformer.h.0.attn.wq.weight` → `transformer.h`. `tok_embeddings.weight`
/// → `tok_embeddings`. `no_dots` → `no_dots`. The second-dot rule gives
/// enough signal to diagnose format mismatches (HuggingFace's `transformer.h`,
/// `model.layers`, etc.) without listing every individual tensor.
fn top_level_prefix(name: &str) -> &str {
    let mut dot_count = 0;
    for (i, c) in name.char_indices() {
        if c == '.' {
            dot_count += 1;
            if dot_count == 2 {
                return &name[..i];
            }
        }
    }
    name
}

/// Whether `layer_name` follows one of the hierarchical prefixes we can
/// validate. Layer names outside this set (currently only `"other"` per
/// `wggo_graph::layer_prefix`) are skipped.
fn is_hierarchical_layer(layer_name: &str) -> bool {
    layer_name.starts_with("blocks.")
        || layer_name.starts_with("layers.")
        || layer_name.starts_with("h.")
}

/// Cross-check the loaded WeightMap against WGGO's AppliedPlan.
///
/// For each hierarchical layer (`blocks.N` / `layers.N` / `h.N`) in
/// `applied`, require at least one tensor in `wm` whose name starts with
/// `layer_name + "."`. The `"other"` catch-all layer is skipped — its
/// contents (embeddings, norms, LM head) are heterogeneous and per-tensor
/// validation is Phase 2 work. If any hierarchical layer is unmatched, all
/// missing layers are aggregated into a single `LayersMissing` error with
/// a summary of the WeightMap's top-level tensor prefixes so the user can
/// diagnose format mismatches.
///
/// Phase 1 scope; per-tensor shape/dtype validation deferred. See spec at
/// `docs/superpowers/specs/2026-04-20-cpdt-validate-body-design.md`.
pub fn validate(wm: &WeightMap, applied: &AppliedPlan) -> Result<(), ValidationError> {
    let mut missing: Vec<String> = Vec::new();
    let mut checked: usize = 0;

    for layer in &applied.layers {
        if !is_hierarchical_layer(&layer.layer_name) {
            continue;
        }
        checked += 1;
        let prefix = format!("{}.", layer.layer_name);
        let has_match = wm.entries().any(|(name, _)| name.starts_with(&prefix));
        if !has_match {
            missing.push(layer.layer_name.clone());
        }
    }

    if missing.is_empty() {
        return Ok(());
    }

    // Aggregate the WeightMap's top-level prefixes for the diagnostic message.
    use std::collections::BTreeSet;
    let mut prefixes: BTreeSet<String> = BTreeSet::new();
    let mut total_tensors = 0usize;
    for (name, _) in wm.entries() {
        total_tensors += 1;
        prefixes.insert(top_level_prefix(name).to_string());
    }
    let truncated = prefixes.len() > PREFIX_SUMMARY_MAX;
    let mut summary: Vec<String> = prefixes.into_iter().take(PREFIX_SUMMARY_MAX).collect();
    if truncated {
        // Signal that the list was cut off; the actual count is embedded in
        // the prefix sort order, but "..." alerts the reader that there are
        // more prefixes than shown.
        summary.push("...".to_string());
    }

    Err(ValidationError::LayersMissing {
        missing,
        total_layers_checked: checked,
        weight_map_prefix_summary: summary,
        weight_map_total_tensors: total_tensors,
    })
}
