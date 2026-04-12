//! FASE — gradient-norm clipping (two-phase final backward).
//!
//! When the train block enables `grad_clip=<τ>`, FASE emits a two-phase
//! final micro-batch:
//!
//!   **Phase A (norm accumulation):**
//!     for each parameter θ_k in reverse topological order:
//!         g_k ← compute_gradient(θ_k)
//!         m_partial_k ← m_partial_k + (1/N) · g_k
//!         running_sq_sum ← running_sq_sum + Σ m_partial_k²
//!         free g_k
//!     global_norm ← √running_sq_sum
//!     clip_factor ← min(1, τ / (global_norm + ε))
//!
//!   **Phase B (clipped update):**
//!     for each parameter θ_k:
//!         clipped ← clip_factor · m_partial_k
//!         run optimizer update using `clipped` as g_acc
//!         m_partial_k ← 0
//!
//! This module produces a [`ClipPlan`] describing the two phases: which
//! reductions run, what register the clip factor lives in, and how the
//! second pass references it.

use serde::Serialize;

/// Norm type used for gradient clipping.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ClipNorm {
    /// Global L2 norm across every parameter.
    L2Global,
    /// Per-parameter L∞ norm (max-abs).
    LinfPerParam,
}

impl ClipNorm {
    pub fn as_str(self) -> &'static str {
        match self {
            ClipNorm::L2Global => "l2_global",
            ClipNorm::LinfPerParam => "linf_per_param",
        }
    }
}

/// Per-phase recipe for the clipped final backward.
#[derive(Debug, Clone, Serialize)]
pub struct ClipPlan {
    pub enabled: bool,
    pub threshold: f64,
    pub norm: ClipNorm,
    /// Numerical floor added to the norm before forming the clip factor, to
    /// prevent division by zero when the gradient is exactly zero.
    pub eps: f64,
    /// Whether Phase A *also* performs the per-parameter accumulator update,
    /// leaving Phase B as a pure multiply-and-step (the recommended path).
    pub accumulate_during_phase_a: bool,
}

impl ClipPlan {
    pub fn disabled() -> Self {
        Self {
            enabled: false,
            threshold: 0.0,
            norm: ClipNorm::L2Global,
            eps: 0.0,
            accumulate_during_phase_a: false,
        }
    }

    /// Construct a standard L2-global plan with sensible defaults.
    pub fn l2_global(threshold: f64) -> Self {
        Self {
            enabled: threshold > 0.0 && threshold.is_finite(),
            threshold,
            norm: ClipNorm::L2Global,
            eps: 1e-6,
            accumulate_during_phase_a: true,
        }
    }

    /// Compute the clip factor for a given pre-clip norm.
    ///
    /// Matches PyTorch's `torch.nn.utils.clip_grad_norm_` (max_norm) formula:
    /// `factor = min(1, τ / (norm + ε))`.  Returns `1.0` when disabled.
    pub fn clip_factor(&self, norm: f64) -> f64 {
        if !self.enabled {
            return 1.0;
        }
        if norm <= 0.0 {
            return 1.0;
        }
        let raw = self.threshold / (norm + self.eps);
        raw.min(1.0)
    }
}

/// Label for ops emitted by the clip planner.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ClipOp {
    /// Accumulator update (`m_partial += (1/N) g`).
    Accumulate,
    /// Running norm² accumulation.
    NormAccumulate,
    /// Barrier between Phase A and Phase B (global sync + compute
    /// `clip_factor`).
    PhaseBarrier,
    /// Pre-update multiply: `m_partial *= clip_factor`.
    ApplyClip,
    /// Run optimizer update using clipped m_partial as g_acc.
    OptimizerStep,
    /// Reset `m_partial = 0`.
    Reset,
}

/// Deterministic program describing what the clipped final backward does.
#[derive(Debug, Clone, Serialize)]
pub struct ClipProgram {
    pub plan: ClipPlan,
    /// Ops in execution order.
    pub ops: Vec<ClipOp>,
}

impl ClipProgram {
    /// Number of ops in Phase A (before the barrier).
    pub fn phase_a_len(&self) -> usize {
        self.ops
            .iter()
            .take_while(|o| !matches!(o, ClipOp::PhaseBarrier))
            .count()
    }

    /// Number of ops in Phase B (after the barrier).
    pub fn phase_b_len(&self) -> usize {
        self.ops
            .iter()
            .skip_while(|o| !matches!(o, ClipOp::PhaseBarrier))
            .skip(1)
            .count()
    }
}

/// Emit the clip program for the configured plan.  `disabled` plans emit
/// a trivial single-phase program (Accumulate → OptimizerStep → Reset).
pub fn emit_program(plan: ClipPlan) -> ClipProgram {
    let mut ops = Vec::new();
    if !plan.enabled {
        ops.push(ClipOp::Accumulate);
        ops.push(ClipOp::OptimizerStep);
        ops.push(ClipOp::Reset);
        return ClipProgram { plan, ops };
    }
    // Phase A
    if plan.accumulate_during_phase_a {
        ops.push(ClipOp::Accumulate);
    }
    ops.push(ClipOp::NormAccumulate);
    ops.push(ClipOp::PhaseBarrier);
    // Phase B
    ops.push(ClipOp::ApplyClip);
    ops.push(ClipOp::OptimizerStep);
    ops.push(ClipOp::Reset);
    ClipProgram { plan, ops }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_plan_returns_factor_one() {
        let p = ClipPlan::disabled();
        assert_eq!(p.clip_factor(10.0), 1.0);
    }

    #[test]
    fn l2_plan_below_threshold_no_clip() {
        let p = ClipPlan::l2_global(1.0);
        // norm 0.5 < 1.0 → clip factor = min(1, 1/0.5) = 1 (capped).
        assert_eq!(p.clip_factor(0.5), 1.0);
    }

    #[test]
    fn l2_plan_above_threshold_scales_down() {
        let p = ClipPlan::l2_global(1.0);
        let f = p.clip_factor(4.0);
        assert!(f < 1.0 && f > 0.0);
        // Rough formula: 1 / (4 + eps) ≈ 0.25.
        assert!((f - 0.25).abs() < 0.01);
    }

    #[test]
    fn program_without_clip_is_three_ops() {
        let prog = emit_program(ClipPlan::disabled());
        assert_eq!(prog.ops.len(), 3);
        assert!(matches!(prog.ops[0], ClipOp::Accumulate));
        assert!(matches!(prog.ops[2], ClipOp::Reset));
    }

    #[test]
    fn program_with_clip_has_both_phases() {
        let prog = emit_program(ClipPlan::l2_global(1.0));
        assert!(prog.ops.iter().any(|o| matches!(o, ClipOp::PhaseBarrier)));
        assert!(prog.phase_a_len() > 0);
        assert!(prog.phase_b_len() >= 3);
    }

    #[test]
    fn zero_norm_never_divides() {
        let p = ClipPlan::l2_global(1.0);
        assert_eq!(p.clip_factor(0.0), 1.0);
    }

    #[test]
    fn negative_threshold_is_treated_as_disabled() {
        let p = ClipPlan::l2_global(-5.0);
        assert!(!p.enabled);
        assert_eq!(p.clip_factor(100.0), 1.0);
    }

    #[test]
    fn norm_enum_name_is_nonempty() {
        assert_eq!(ClipNorm::L2Global.as_str(), "l2_global");
        assert_eq!(ClipNorm::LinfPerParam.as_str(), "linf_per_param");
    }
}
