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

pub const ANALYSIS_VERSION: u32 = 1;

// ---------------------------------------------------------------------------
// Calibration constants
//
// All constants are placeholders pending the Task 1.11 calibration run
// that regenerates `tests/fixtures/cpdt_calibration/baseline_heuristic.json`
// and emits the finalized values. Running `tools/cpdt_calibrate.rs`
// (dev-only, gated behind `[features] calibrate = []`) reproduces them.
// ---------------------------------------------------------------------------

/// Neutral value of `gradient_magnitude_est` when weights are absent.
/// Calibrated so the no-weights path's tier assignments agree with the
/// weights-present path within 5% parameter-weighted on the baseline corpus.
pub const CALIB_K: f64 = 0.0312;

pub const CALIB_T0: f64 = 0.50; // High   ↔ Medium
pub const CALIB_T1: f64 = 0.10; // Medium ↔ Low
pub const CALIB_T2: f64 = 0.02; // Low    ↔ VeryLow

/// Position-criticality near-extreme boost (for `L ≥ 4`).
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
// Validation pass against AppliedPlan (called by cpdt::run before scoring)
//
// Phase 1 Commit 1 ships a stub; Commit 5 fills in the body after confirming
// the AppliedLayer weight-metadata surface.
// ---------------------------------------------------------------------------

use crate::weight_aware::WeightMap;
use crate::wggo_apply::AppliedPlan;

#[derive(Debug)]
pub enum ValidationError {
    MissingTensor {
        tensor_name: String,
    },
    ShapeMismatch {
        tensor_name: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    DtypeMismatch {
        tensor_name: String,
        expected: String,
        actual: String,
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
        }
    }
}

impl std::error::Error for ValidationError {}

/// Cross-check the loaded WeightMap against WGGO's AppliedPlan. Phase 1
/// Commit 1 ships a stub; the body is filled in during Commit 5 (Task 5.2)
/// after confirming the exact `AppliedLayer` weight-metadata surface.
pub fn validate(wm: &WeightMap, applied: &AppliedPlan) -> Result<(), ValidationError> {
    // TODO(plan): cross-check against AppliedLayer.weight_name/shape/dtype.
    // The exact AppliedLayer field names are read during Commit 5 Task 5.2
    // when the validation is wired into cpdt::run.
    let _ = (wm, applied);
    Ok(())
}
