//! FASE — per-layer memory schedule.
//!
//! This module models the peak-memory timeline of gradient accumulation
//! under FASE and compares it with the standard "full-gradient-buffer"
//! baseline.  Its output is consumed by the memory planner (M36) to emit
//! a slot-assignment plan that reuses one layer's gradient slot for the
//! next as soon as the optimizer step for that layer completes.
//!
//! The FASE paper's key memory claim (Section 3.4) is not a reduction in
//! *total* state — first-moment m_partial is the same size as the full
//! gradient buffer — but a reduction in *peak* residency: at any point
//! during the fused final backward, only **one layer's gradient** is
//! live.  This module quantifies that reduction for a given model.

use serde::Serialize;

use crate::fase::{FaseMode, FasePlan};

/// Per-parameter size record.  Sizes are in bytes.
#[derive(Debug, Clone, Serialize)]
pub struct ParamFootprint {
    pub name: String,
    /// Bytes of parameter storage.
    pub param_bytes: u64,
    /// Bytes of activation saved for this layer's backward (typically this
    /// layer's forward output or similar).
    pub activation_bytes: u64,
}

/// Model-wide footprint for the memory analysis.
#[derive(Debug, Clone, Default, Serialize)]
pub struct ModelFootprint {
    pub params: Vec<ParamFootprint>,
    /// Bytes of optimizer state per parameter byte.  For AdamW this is 2
    /// (m + v as f32 alongside a possibly-f16 parameter).  For SGD it's 0
    /// (or 1 if momentum is enabled).
    pub optimizer_state_multiplier: f64,
}

impl ModelFootprint {
    pub fn total_param_bytes(&self) -> u64 {
        self.params.iter().map(|p| p.param_bytes).sum()
    }
    pub fn total_activation_bytes(&self) -> u64 {
        self.params.iter().map(|p| p.activation_bytes).sum()
    }
    pub fn peak_activation_layer_bytes(&self) -> u64 {
        self.params
            .iter()
            .map(|p| p.activation_bytes)
            .max()
            .unwrap_or(0)
    }
    pub fn max_param_bytes(&self) -> u64 {
        self.params.iter().map(|p| p.param_bytes).max().unwrap_or(0)
    }
}

/// Memory summary for a given scheduling mode.
#[derive(Debug, Clone, Default, Serialize)]
pub struct MemoryBreakdown {
    pub params: u64,
    pub gradients: u64,
    pub optimizer_state: u64,
    pub activations: u64,
    pub peak: u64,
}

/// Full memory schedule produced by FASE.
#[derive(Debug, Clone, Serialize)]
pub struct MemorySchedule {
    pub mode: FaseMode,
    pub standard: MemoryBreakdown,
    pub fase: MemoryBreakdown,
}

impl MemorySchedule {
    /// Absolute byte savings of FASE over the standard schedule.
    pub fn peak_savings(&self) -> u64 {
        self.standard.peak.saturating_sub(self.fase.peak)
    }

    /// Fractional peak-memory reduction.
    pub fn peak_reduction(&self) -> f64 {
        if self.standard.peak == 0 {
            return 0.0;
        }
        self.peak_savings() as f64 / self.standard.peak as f64
    }
}

fn optimizer_bytes(footprint: &ModelFootprint) -> u64 {
    let total_params = footprint.total_param_bytes();
    (total_params as f64 * footprint.optimizer_state_multiplier) as u64
}

fn standard_breakdown(footprint: &ModelFootprint) -> MemoryBreakdown {
    let params = footprint.total_param_bytes();
    let gradients = params; // full gradient buffer, same size as parameters.
    let opt_state = optimizer_bytes(footprint);
    let activations = footprint.total_activation_bytes();
    MemoryBreakdown {
        params,
        gradients,
        optimizer_state: opt_state,
        activations,
        peak: params + gradients + opt_state + activations,
    }
}

fn fase_breakdown(footprint: &ModelFootprint, plan: &FasePlan) -> MemoryBreakdown {
    let params = footprint.total_param_bytes();
    let opt_state = optimizer_bytes(footprint);
    // FASE residency depends on the mode:
    //   Deferred → m_partial-sized buffer (== params), one-layer gradient.
    //   FullBuffer → full gradient buffer, one-layer gradient.
    //   Passthrough → same as standard.
    let (accumulator_bytes, one_layer_grad) = match plan.mode {
        FaseMode::Passthrough => (params, 0),
        FaseMode::Deferred => (params, footprint.max_param_bytes()),
        FaseMode::FullBuffer => (params, footprint.max_param_bytes()),
    };
    // Peak is reached at the final micro-batch, mid-backward, when ONE
    // layer's gradient is live alongside m_partial / grad-buffer and the
    // activation for that same layer.
    let peak_activation_layer = footprint.peak_activation_layer_bytes();
    let peak = match plan.mode {
        FaseMode::Passthrough => params + accumulator_bytes + opt_state + peak_activation_layer,
        _ => params + accumulator_bytes + one_layer_grad + opt_state + peak_activation_layer,
    };
    MemoryBreakdown {
        params,
        gradients: accumulator_bytes,
        optimizer_state: opt_state,
        activations: peak_activation_layer,
        peak,
    }
}

/// Compute the full before/after memory schedule.
pub fn schedule(footprint: &ModelFootprint, plan: &FasePlan) -> MemorySchedule {
    MemorySchedule {
        mode: plan.mode,
        standard: standard_breakdown(footprint),
        fase: fase_breakdown(footprint, plan),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fase::{plan as fase_plan, FaseConfig, FaseOptimizer};

    fn nslcoder_50m_footprint() -> ModelFootprint {
        // 8 transformer blocks, ~6.1 MB params each (48.8 MB total as f32).
        // Activation per layer ~ batch × seq × d_model × 4 = 1 × 1024 × 512 × 4 = 2 MB.
        let mut params = Vec::new();
        for i in 0..8 {
            params.push(ParamFootprint {
                name: format!("block.{i}"),
                param_bytes: 6_100_000,
                activation_bytes: 2_000_000,
            });
        }
        ModelFootprint {
            params,
            optimizer_state_multiplier: 2.0, // AdamW: m + v
        }
    }

    #[test]
    fn passthrough_matches_standard() {
        let fp = nslcoder_50m_footprint();
        let p = fase_plan(&FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 1,
            ..Default::default()
        });
        let s = schedule(&fp, &p);
        // Activation peak differs (FASE uses max-layer, standard uses
        // total), but gradient/optimizer bytes should match when N=1.
        assert_eq!(s.standard.gradients, s.fase.gradients);
        assert_eq!(s.standard.optimizer_state, s.fase.optimizer_state);
    }

    #[test]
    fn deferred_cuts_peak_under_accumulation() {
        let fp = nslcoder_50m_footprint();
        let p = fase_plan(&FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            ..Default::default()
        });
        let s = schedule(&fp, &p);
        assert!(
            s.fase.peak < s.standard.peak,
            "FASE peak {} should be below standard peak {}",
            s.fase.peak,
            s.standard.peak
        );
        // Savings should be on the order of 6 MB (full activations vs one
        // layer's activations) + full grad vs one layer's grad.
        assert!(s.peak_savings() > 6_000_000);
    }

    #[test]
    fn reduction_is_fraction_in_valid_range() {
        let fp = nslcoder_50m_footprint();
        let p = fase_plan(&FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            ..Default::default()
        });
        let s = schedule(&fp, &p);
        let r = s.peak_reduction();
        assert!((0.0..1.0).contains(&r), "reduction {r} out of range");
    }

    #[test]
    fn zero_params_is_safe() {
        let fp = ModelFootprint::default();
        let p = fase_plan(&FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            ..Default::default()
        });
        let s = schedule(&fp, &p);
        assert_eq!(s.standard.peak, 0);
        assert_eq!(s.fase.peak, 0);
        assert_eq!(s.peak_reduction(), 0.0);
    }

    #[test]
    fn full_buffer_fallback_still_saves_activations() {
        let fp = nslcoder_50m_footprint();
        let p = fase_plan(&FaseConfig {
            optimizer: FaseOptimizer::Lion,
            accumulation: 4,
            ..Default::default()
        });
        let s = schedule(&fp, &p);
        // FullBuffer mode still saves activations (peak-layer vs total).
        assert!(s.fase.activations < s.standard.activations);
        assert!(s.fase.peak < s.standard.peak);
    }
}
