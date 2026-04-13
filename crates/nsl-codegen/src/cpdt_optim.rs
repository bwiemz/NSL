//! CPDT — quantized optimizer step codegen.
//!
//! Emits the structured op-program for a per-layer fused optimizer step
//! that honours the [`PrecisionPlan`] from `cpdt_precision`.  Mirrors
//! FASE's approach (paper §4 composition): the optimizer is inlined
//! into the backward pass as a sequence of [`QuantizedOptimOp`]s that
//! the downstream PTX / Cranelift emitters consume.
//!
//! Critical invariant: quant/dequant happens in registers.  The program
//! never introduces additional HBM traffic beyond one m/v read and one
//! m/v write per step.

use serde::Serialize;

use crate::cpdt_precision::{OptimPrecision, ParamPrecision};

/// One instruction in the fused optimizer program.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum QuantizedOptimOp {
    /// Load `m` from HBM (stored in `src_precision`), dequantize to
    /// compute precision (FP32), and place in the register file.
    DequantLoadM {
        src_precision: OptimPrecision,
    },
    /// Load `v` similarly.
    DequantLoadV {
        src_precision: OptimPrecision,
    },
    /// `m = β₁·m + (1-β₁)·g`.
    MomentumUpdate {
        beta1: f64,
    },
    /// `v = β₂·v + (1-β₂)·g²`.
    VarianceUpdate {
        beta2: f64,
    },
    /// `θ ← θ - lr · (m̂ / (√v̂ + ε) + wd·θ)`.
    ParamUpdate {
        lr: f64,
        eps: f64,
        weight_decay: f64,
    },
    /// Quantize `m` back to its stored precision and write to HBM.
    /// When `stochastic` is true, random rounding is used (required
    /// for INT8 embeddings per Q-Adam-mini).
    QuantStoreM {
        dst_precision: OptimPrecision,
        stochastic: bool,
        blockwise: bool,
    },
    /// Quantize `v` back.
    QuantStoreV {
        dst_precision: OptimPrecision,
        stochastic: bool,
        blockwise: bool,
    },
    /// Free the per-layer gradient register — gradients are never
    /// stored in HBM under the fused step.
    FreeGradient,
}

/// Complete per-parameter program.
#[derive(Debug, Clone, Serialize)]
pub struct QuantizedOptimProgram {
    pub param_name: String,
    pub ops: Vec<QuantizedOptimOp>,
    pub m_precision: OptimPrecision,
    pub v_precision: OptimPrecision,
    pub stochastic_rounding: bool,
}

impl QuantizedOptimProgram {
    /// Whether the program uses any quantization (either m or v < FP32).
    pub fn uses_quantization(&self) -> bool {
        self.m_precision != OptimPrecision::Fp32 || self.v_precision != OptimPrecision::Fp32
    }

    /// Number of ops in the program.
    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// AdamW hyperparameters.
#[derive(Debug, Clone, Copy)]
pub struct AdamWHyperparams {
    pub lr: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub eps: f64,
    pub weight_decay: f64,
}

impl Default for AdamWHyperparams {
    fn default() -> Self {
        Self {
            lr: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.01,
        }
    }
}

/// Emit the fused-step program for a single parameter.
pub fn emit_step(precision: &ParamPrecision, hyper: &AdamWHyperparams) -> QuantizedOptimProgram {
    let m_stochastic = precision.stochastic_rounding && precision.m_precision == OptimPrecision::Int8;
    let v_stochastic = precision.stochastic_rounding && precision.v_precision == OptimPrecision::Int8;
    let blockwise_m = precision.m_precision == OptimPrecision::Int8;
    let blockwise_v = precision.v_precision == OptimPrecision::Int8;

    let ops = vec![
        QuantizedOptimOp::DequantLoadM {
            src_precision: precision.m_precision,
        },
        QuantizedOptimOp::DequantLoadV {
            src_precision: precision.v_precision,
        },
        QuantizedOptimOp::MomentumUpdate { beta1: hyper.beta1 },
        QuantizedOptimOp::VarianceUpdate { beta2: hyper.beta2 },
        QuantizedOptimOp::ParamUpdate {
            lr: hyper.lr,
            eps: hyper.eps,
            weight_decay: hyper.weight_decay,
        },
        QuantizedOptimOp::QuantStoreM {
            dst_precision: precision.m_precision,
            stochastic: m_stochastic,
            blockwise: blockwise_m,
        },
        QuantizedOptimOp::QuantStoreV {
            dst_precision: precision.v_precision,
            stochastic: v_stochastic,
            blockwise: blockwise_v,
        },
        QuantizedOptimOp::FreeGradient,
    ];

    QuantizedOptimProgram {
        param_name: precision.name.clone(),
        ops,
        m_precision: precision.m_precision,
        v_precision: precision.v_precision,
        stochastic_rounding: precision.stochastic_rounding,
    }
}

/// Emit per-parameter programs for a whole precision plan.
pub fn emit_plan(
    plan: &crate::cpdt_precision::PrecisionPlan,
    hyper: &AdamWHyperparams,
) -> Vec<QuantizedOptimProgram> {
    plan.params.iter().map(|p| emit_step(p, hyper)).collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpdt_precision::{OptimPrecision, ParamPrecision, SensitivityTier};

    fn param(name: &str, m: OptimPrecision, v: OptimPrecision, stoch: bool) -> ParamPrecision {
        ParamPrecision {
            name: name.to_string(),
            layer: Some(3),
            tier: SensitivityTier::Low,
            m_precision: m,
            v_precision: v,
            stochastic_rounding: stoch,
            sensitivity_score: 0.01,
            param_bytes: 1024,
            optim_bytes: 1024,
        }
    }

    #[test]
    fn step_has_eight_ops_including_free() {
        let p = param("w", OptimPrecision::Int8, OptimPrecision::Fp16, false);
        let prog = emit_step(&p, &AdamWHyperparams::default());
        assert_eq!(prog.len(), 8);
        assert!(matches!(prog.ops.last(), Some(QuantizedOptimOp::FreeGradient)));
    }

    #[test]
    fn fp32_fp32_does_not_use_quantization() {
        let p = param("w", OptimPrecision::Fp32, OptimPrecision::Fp32, false);
        let prog = emit_step(&p, &AdamWHyperparams::default());
        assert!(!prog.uses_quantization());
    }

    #[test]
    fn int8_embedding_gets_stochastic_rounding() {
        let p = param(
            "tok_embeddings.weight",
            OptimPrecision::Int8,
            OptimPrecision::Int8,
            true,
        );
        let prog = emit_step(&p, &AdamWHyperparams::default());
        let stoch_stores: Vec<_> = prog
            .ops
            .iter()
            .filter(|op| {
                matches!(op, QuantizedOptimOp::QuantStoreM { stochastic: true, .. })
                    || matches!(op, QuantizedOptimOp::QuantStoreV { stochastic: true, .. })
            })
            .collect();
        assert_eq!(stoch_stores.len(), 2);
    }

    #[test]
    fn fp16_v_store_not_blockwise() {
        let p = param("w", OptimPrecision::Int8, OptimPrecision::Fp16, false);
        let prog = emit_step(&p, &AdamWHyperparams::default());
        for op in &prog.ops {
            if let QuantizedOptimOp::QuantStoreV { blockwise, .. } = op {
                assert!(!blockwise); // FP16 doesn't need blockwise quantization
            }
        }
    }

    #[test]
    fn program_carries_param_name() {
        let p = param("blocks.4.ffn.w_gate.weight", OptimPrecision::Int8, OptimPrecision::Fp16, false);
        let prog = emit_step(&p, &AdamWHyperparams::default());
        assert_eq!(prog.param_name, "blocks.4.ffn.w_gate.weight");
    }

    #[test]
    fn emit_plan_has_one_program_per_param() {
        use crate::cpdt_precision::PrecisionPlan;
        let plan = PrecisionPlan {
            params: vec![
                param("a", OptimPrecision::Fp32, OptimPrecision::Fp32, false),
                param("b", OptimPrecision::Int8, OptimPrecision::Fp16, false),
            ],
            total_optim_bytes: 2048,
            baseline_fp32_bytes: 4096,
        };
        let programs = emit_plan(&plan, &AdamWHyperparams::default());
        assert_eq!(programs.len(), 2);
    }

    #[test]
    fn hyperparams_propagate_into_ops() {
        let p = param("w", OptimPrecision::Int8, OptimPrecision::Fp16, false);
        let hyper = AdamWHyperparams {
            lr: 5e-4,
            beta1: 0.85,
            beta2: 0.95,
            eps: 1e-6,
            weight_decay: 0.1,
        };
        let prog = emit_step(&p, &hyper);
        let has_lr = prog.ops.iter().any(|op| matches!(op, QuantizedOptimOp::ParamUpdate { lr, .. } if (*lr - 5e-4).abs() < 1e-12));
        assert!(has_lr);
        let has_b1 = prog
            .ops
            .iter()
            .any(|op| matches!(op, QuantizedOptimOp::MomentumUpdate { beta1 } if (*beta1 - 0.85).abs() < 1e-12));
        assert!(has_b1);
    }

    #[test]
    fn order_matches_paper_section_4_2() {
        // The CPDT paper prescribes: dequant → moment/variance update →
        // param update → quant → free.  Assert that the emitted op
        // sequence follows exactly that order.
        let p = param("w", OptimPrecision::Int8, OptimPrecision::Fp16, false);
        let prog = emit_step(&p, &AdamWHyperparams::default());
        let names: Vec<&str> = prog
            .ops
            .iter()
            .map(|op| match op {
                QuantizedOptimOp::DequantLoadM { .. } => "dequant_m",
                QuantizedOptimOp::DequantLoadV { .. } => "dequant_v",
                QuantizedOptimOp::MomentumUpdate { .. } => "momentum",
                QuantizedOptimOp::VarianceUpdate { .. } => "variance",
                QuantizedOptimOp::ParamUpdate { .. } => "param_update",
                QuantizedOptimOp::QuantStoreM { .. } => "quant_m",
                QuantizedOptimOp::QuantStoreV { .. } => "quant_v",
                QuantizedOptimOp::FreeGradient => "free",
            })
            .collect();
        assert_eq!(
            names,
            vec![
                "dequant_m",
                "dequant_v",
                "momentum",
                "variance",
                "param_update",
                "quant_m",
                "quant_v",
                "free",
            ]
        );
    }
}
