//! CPDT tier application — migrated from the now-deleted `cpdt_precision.rs`.
//!
//! This module owns the *application* of sensitivity tiers to parameters:
//! - [`OptimPrecision`] / [`Tier`] enums and their precision-pair mapping.
//! - [`ParamPrecision`] / [`PrecisionPlan`] structs that downstream CPDT passes
//!   (cpdt_joint, cpdt_optim, cpdt_comm) consume.
//! - [`PrecisionConfig`] with `n_layers` + stochastic-rounding fields.
//! - [`classify_param`] and [`plan_map`] entry points delegating to
//!   [`crate::cpdt_sensitivity::SensitivityScorer`].
//!
//! Tier *assignment* (deciding which tier a parameter belongs to) lives in
//! `cpdt_sensitivity.rs`. This separation keeps filenames honest: sensitivity
//! decides, tier_apply emits.

use serde::Serialize;

use crate::cpdt_sensitivity::{LayerKind, SensitivityScorer};
use crate::weight_aware::{WeightEntry, WeightMap};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Tier {
    High,
    Medium,
    Low,
    VeryLow,
}

impl Tier {
    pub fn precision(self) -> (OptimPrecision, OptimPrecision) {
        match self {
            Tier::High => (OptimPrecision::Fp32, OptimPrecision::Fp32),
            Tier::Medium => (OptimPrecision::Fp16, OptimPrecision::Fp32),
            Tier::Low => (OptimPrecision::Int8, OptimPrecision::Fp16),
            Tier::VeryLow => (OptimPrecision::Int8, OptimPrecision::Int8),
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            Tier::High => "high",
            Tier::Medium => "medium",
            Tier::Low => "low",
            Tier::VeryLow => "very_low",
        }
    }
}

/// Back-compat alias for the old `SensitivityTier` name. Downstream callers
/// that haven't migrated to `Tier` yet keep compiling.
pub type SensitivityTier = Tier;

#[derive(Debug, Clone, Serialize)]
pub struct ParamPrecision {
    pub name: String,
    pub layer: Option<u32>,
    pub tier: Tier,
    pub m_precision: OptimPrecision,
    pub v_precision: OptimPrecision,
    pub stochastic_rounding: bool,
    pub sensitivity_score: f64,
    pub param_bytes: u64,
    pub optim_bytes: u64,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct PrecisionPlan {
    pub params: Vec<ParamPrecision>,
    pub total_optim_bytes: u64,
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
                Tier::High => h += 1,
                Tier::Medium => m += 1,
                Tier::Low => l += 1,
                Tier::VeryLow => v += 1,
            }
        }
        (h, m, l, v)
    }
}

#[derive(Debug, Clone)]
pub struct PrecisionConfig {
    /// Total layer count (for position-criticality computation).
    pub n_layers: u32,
    /// When `true`, embedding tensors always get stochastic rounding —
    /// Q-Adam-mini showed this is required for INT8 stability on embeddings.
    pub embedding_stochastic_rounding: bool,
}

impl Default for PrecisionConfig {
    fn default() -> Self {
        Self {
            n_layers: 8,
            embedding_stochastic_rounding: true,
        }
    }
}

/// Score a single tensor and produce its precision decision. Delegates to
/// [`SensitivityScorer`] for the tier assignment; this function packages
/// the tier into a [`ParamPrecision`] record that downstream CPDT passes
/// consume.
pub fn classify_param(entry: &WeightEntry, cfg: &PrecisionConfig) -> ParamPrecision {
    let scorer = SensitivityScorer::from_config(cfg);
    let (tier, score, layer, kind) = scorer.score_entry(entry);
    let (m, v) = tier.precision();
    let param_bytes = (entry.num_elements as u64) * (entry.dtype.byte_width() as u64);
    let optim_bytes = (entry.num_elements as u64) * (m.bytes() as u64 + v.bytes() as u64);
    let stochastic = cfg.embedding_stochastic_rounding && matches!(kind, LayerKind::Embedding);
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

/// Classify every tensor in a [`WeightMap`] and produce the aggregate plan.
pub fn plan_map(wm: &WeightMap, cfg: &PrecisionConfig) -> PrecisionPlan {
    let mut params = Vec::new();
    let mut total_optim = 0u64;
    let mut baseline_fp32 = 0u64;
    for (_name, entry) in wm.entries() {
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

/// Score every tensor in `wm` using the no-weights path (gradient_magnitude_est
/// replaced by `CALIB_K`). The outer caller passes the same `WeightMap` the
/// weights-present pass used so layer identity and element counts match — this
/// is what makes the two plans directly comparable in
/// [`compute_tier_agreement`].
pub fn plan_map_noweights(wm: &WeightMap, cfg: &PrecisionConfig) -> PrecisionPlan {
    let scorer = SensitivityScorer::from_config(cfg);
    let mut params = Vec::new();
    let mut total_optim = 0u64;
    let mut baseline_fp32 = 0u64;
    for (name, entry) in wm.entries() {
        let (tier, score, layer, kind) = scorer.score_optional(name, entry.num_elements, None);
        let (m, v) = tier.precision();
        let param_bytes = (entry.num_elements as u64) * (entry.dtype.byte_width() as u64);
        let optim_bytes = (entry.num_elements as u64) * (m.bytes() as u64 + v.bytes() as u64);
        let stochastic = cfg.embedding_stochastic_rounding && matches!(kind, LayerKind::Embedding);
        params.push(ParamPrecision {
            name: name.clone(),
            layer,
            tier,
            m_precision: m,
            v_precision: v,
            stochastic_rounding: stochastic,
            sensitivity_score: score,
            param_bytes,
            optim_bytes,
        });
        total_optim += optim_bytes;
        baseline_fp32 += (entry.num_elements as u64) * 8;
    }
    PrecisionPlan {
        params,
        total_optim_bytes: total_optim,
        baseline_fp32_bytes: baseline_fp32,
    }
}

/// Tier-agreement between a weights-present `plan` and a no-weights `plan_nw`.
///
/// Returns `(agree_layers, total_layers, agree_params, total_params)`. The two
/// plans must have been produced from the same `WeightMap`; layer alignment
/// goes by name, and any name present in one but not the other is counted as
/// disagreement for the layers that are shared (mismatched layers are
/// ignored — the caller should ensure plans share identities).
pub fn compute_tier_agreement(
    plan: &PrecisionPlan,
    plan_nw: &PrecisionPlan,
) -> (u64, u64, u64, u64) {
    use std::collections::HashMap;
    let by_name_nw: HashMap<&str, &ParamPrecision> =
        plan_nw.params.iter().map(|p| (p.name.as_str(), p)).collect();
    let mut agree_layers: u64 = 0;
    let mut total_layers: u64 = 0;
    let mut agree_params: u64 = 0;
    let mut total_params: u64 = 0;
    for p in &plan.params {
        let Some(pnw) = by_name_nw.get(p.name.as_str()) else {
            continue;
        };
        total_layers += 1;
        total_params += p.param_bytes.max(1);
        if p.tier == pnw.tier {
            agree_layers += 1;
            agree_params += p.param_bytes.max(1);
        }
    }
    (agree_layers, total_layers, agree_params, total_params)
}
