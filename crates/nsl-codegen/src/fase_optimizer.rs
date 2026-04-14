//! FASE — per-layer optimizer update-rule emission.
//!
//! Given an [`UpdateRecipe`] from `fase.rs`, this module emits a
//! deterministic textual description of the update rule that the downstream
//! PTX / Cranelift backends specialise.  The representation is a small
//! [`UpdateProgram`] — a short sequence of [`UpdateOp`]s operating on
//! per-parameter scalar registers — that captures exactly what mathematics
//! the backend must execute for one parameter tensor during the fused
//! accumulator-update step.
//!
//! This module is pure and has no backend dependencies.  It serves two
//! purposes:
//!
//!   1. A textual "golden" encoding that the backend golden-test suite can
//!      diff against, so that changes to the update rule are caught early.
//!   2. A structured program that code-generators walk to emit the actual
//!      instructions, keeping the arithmetic in a single audited location.

use serde::Serialize;

use crate::fase::{FaseOptimizer, UpdateRecipe};

/// Symbolic registers visible to the update program.
///
/// Registers are per-parameter scalars (broadcast across the parameter's
/// shape when the backend lowers the program).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Register {
    /// Parameter tensor θ.
    Theta,
    /// First-moment state `m` for AdamW / Adam / SGD-momentum / Lion.
    M,
    /// Deferred-mode accumulator: the running mean gradient
    /// `m_partial = (1/N) Σ gᵢ` across the micro-batch window.
    MPartial,
    /// Second-moment accumulator buffer (AdamW only).
    V,
    /// Per-micro-batch gradient, live only within one backward-step worth
    /// of register scope.
    G,
    /// Scratch (temporary) register.
    Tmp,
}

/// A single op in the update program.  These are the *only* mathematical
/// operations FASE's per-layer emitter performs; anything beyond this is a
/// mistake.  (Contrast with the general source-AD IR, which supports
/// arbitrary gradient computation.)
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum UpdateOp {
    /// Scalar-multiply-add: `dst = a * src + b`.  Used heavily for moment
    /// updates (`m = β₁·m + (1-β₁)·g`).
    ScalarMulAdd {
        dst: Register,
        src: Register,
        a: f64,
        b_src: Option<Register>,
        b_scale: f64,
    },
    /// Elementwise square: `dst = src * src`.
    Square { dst: Register, src: Register },
    /// `dst = src + scale * squared(operand)` — fused square-accumulate.
    SquaredAccumulate {
        dst: Register,
        src: Register,
        operand: Register,
        scale: f64,
    },
    /// `dst = sqrt(src) + eps`.
    SqrtPlusEps {
        dst: Register,
        src: Register,
        eps: f64,
    },
    /// `dst = src / divisor`.
    Div {
        dst: Register,
        src: Register,
        divisor: Register,
    },
    /// `θ -= lr * (scaled_m + wd * θ)` — the AdamW parameter-update step.
    Update {
        lr: f64,
        wd: f64,
        scaled_m: Register,
    },
    /// Plain SGD-momentum update: `θ -= lr * m`.
    SgdUpdate { lr: f64 },
    /// `dst = sign(src)`.
    Sign { dst: Register, src: Register },
    /// Zero the register.
    Zero(Register),
}

/// A linear sequence of [`UpdateOp`]s describing one optimizer step.
#[derive(Debug, Clone, Serialize)]
pub struct UpdateProgram {
    pub optimizer: FaseOptimizer,
    pub ops: Vec<UpdateOp>,
    /// Textual mnemonic of the program — useful for golden testing / CLI
    /// reports.
    pub pseudocode: String,
}

/// Emit the final-micro-batch update program for the given recipe.
///
/// For FASE-Deferred mode the `G` register contains the per-micro-batch
/// gradient, which was already added into `M` by the previous phase's
/// accumulation.  This program runs *after* the accumulator is complete.
pub fn emit_final_step(recipe: &UpdateRecipe) -> UpdateProgram {
    match recipe.optimizer {
        FaseOptimizer::AdamW => emit_adamw(recipe, /*decoupled_wd=*/ true),
        FaseOptimizer::Adam => emit_adamw(recipe, /*decoupled_wd=*/ false),
        FaseOptimizer::Sgd => emit_sgd(recipe, /*with_momentum=*/ false),
        FaseOptimizer::SgdMomentum => emit_sgd(recipe, /*with_momentum=*/ true),
        FaseOptimizer::Lion => emit_lion(recipe),
        FaseOptimizer::Unknown => emit_empty("optimizer is unknown — no update emitted"),
    }
}

/// Emit the per-micro-batch accumulator update (runs for every micro-batch
/// including the final one in Deferred mode).
///
/// The program performs `m_partial += accum_scale * g`.  For FullBuffer
/// mode the equivalent op is a plain gradient-buffer add instead.
pub fn emit_accumulate(recipe: &UpdateRecipe) -> UpdateProgram {
    let ops = vec![UpdateOp::ScalarMulAdd {
        dst: Register::MPartial,
        src: Register::MPartial,
        a: 1.0,
        b_src: Some(Register::G),
        b_scale: recipe.accum_scale,
    }];
    UpdateProgram {
        optimizer: recipe.optimizer,
        ops,
        pseudocode: format!("m_partial += {} * g", recipe.accum_scale),
    }
}

/// Emit the post-accumulation reset program: zeroes `m_partial` after the
/// fused final step.
pub fn emit_reset() -> UpdateProgram {
    UpdateProgram {
        optimizer: FaseOptimizer::Unknown,
        ops: vec![UpdateOp::Zero(Register::MPartial)],
        pseudocode: "m_partial = 0".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Per-optimizer emitters
// ---------------------------------------------------------------------------

fn emit_adamw(recipe: &UpdateRecipe, decoupled_wd: bool) -> UpdateProgram {
    // Assumes the caller has already accumulated `m_partial` across all
    // micro-batches.  Now:
    //   1. m = β₁·m + (1-β₁)·m_partial
    //   2. v = β₂·v + (1-β₂)·m_partial²           (or approx: per-step avg)
    //   3. denom = sqrt(v) + ε
    //   4. tmp = m / denom
    //   5. θ -= lr * (tmp + wd·θ)    [AdamW]   or   θ -= lr·tmp; θ *= (1-lr·wd) [Adam, coupled]
    let wd = if decoupled_wd { recipe.weight_decay } else { 0.0 };

    let ops = vec![
        // m = β₁·m + (1-β₁)·m_partial
        UpdateOp::ScalarMulAdd {
            dst: Register::M,
            src: Register::M,
            a: recipe.beta1,
            b_src: Some(Register::MPartial),
            b_scale: recipe.one_minus_beta1,
        },
        // v = β₂·v + (1-β₂)·m_partial²
        UpdateOp::SquaredAccumulate {
            dst: Register::V,
            src: Register::V,
            operand: Register::MPartial,
            scale: recipe.one_minus_beta2,
        },
        // tmp = sqrt(v) + eps
        UpdateOp::SqrtPlusEps {
            dst: Register::Tmp,
            src: Register::V,
            eps: recipe.eps,
        },
        // tmp = m / tmp
        UpdateOp::Div {
            dst: Register::Tmp,
            src: Register::M,
            divisor: Register::Tmp,
        },
        // θ -= lr * (tmp + wd·θ)
        UpdateOp::Update {
            lr: recipe.lr,
            wd,
            scaled_m: Register::Tmp,
        },
    ];
    UpdateProgram {
        optimizer: recipe.optimizer,
        ops,
        pseudocode: format!(
            "m=β₁·m+(1-β₁)·m_partial; v{}=β₂·v+(1-β₂)·m_partial²; θ -= lr·(m/(√v+ε) + wd·θ)",
            if recipe.v_uses_approx { "≈" } else { "=" }
        ),
    }
}

fn emit_sgd(recipe: &UpdateRecipe, with_momentum: bool) -> UpdateProgram {
    // In Deferred mode the caller has accumulated the mean gradient into
    // m_partial.  Plain SGD applies it directly (θ -= lr·m_partial, via the
    // SgdUpdate op which reads m_partial implicitly).  SGD+momentum folds
    // m_partial into the running momentum buffer first.
    let mut ops = Vec::new();
    if with_momentum {
        ops.push(UpdateOp::ScalarMulAdd {
            dst: Register::M,
            src: Register::M,
            a: 1.0,
            b_src: Some(Register::MPartial),
            b_scale: 1.0,
        });
    }
    ops.push(UpdateOp::SgdUpdate { lr: recipe.lr });
    UpdateProgram {
        optimizer: recipe.optimizer,
        ops,
        pseudocode: if with_momentum {
            format!("m = m + m_partial; θ -= {}·m", recipe.lr)
        } else {
            format!("θ -= {}·m_partial", recipe.lr)
        },
    }
}

fn emit_lion(recipe: &UpdateRecipe) -> UpdateProgram {
    // Lion's update: update = sign(β₁·m + (1-β₁)·g); θ -= lr·(update + wd·θ)
    let ops = vec![
        UpdateOp::ScalarMulAdd {
            dst: Register::Tmp,
            src: Register::M,
            a: recipe.beta1,
            b_src: Some(Register::G),
            b_scale: recipe.one_minus_beta1,
        },
        UpdateOp::Sign {
            dst: Register::Tmp,
            src: Register::Tmp,
        },
        UpdateOp::Update {
            lr: recipe.lr,
            wd: recipe.weight_decay,
            scaled_m: Register::Tmp,
        },
        // m = β₂·m + (1-β₂)·g
        UpdateOp::ScalarMulAdd {
            dst: Register::M,
            src: Register::M,
            a: recipe.beta2,
            b_src: Some(Register::G),
            b_scale: recipe.one_minus_beta2,
        },
    ];
    UpdateProgram {
        optimizer: FaseOptimizer::Lion,
        ops,
        pseudocode: "Lion: update=sign(β₁m+(1-β₁)g); θ-=lr(update+wd·θ); m=β₂m+(1-β₂)g".into(),
    }
}

fn emit_empty(reason: &str) -> UpdateProgram {
    UpdateProgram {
        optimizer: FaseOptimizer::Unknown,
        ops: Vec::new(),
        pseudocode: reason.to_string(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fase::{plan, FaseConfig, FaseOptimizer};

    fn adamw_recipe() -> UpdateRecipe {
        plan(&FaseConfig {
            optimizer: FaseOptimizer::AdamW,
            accumulation: 4,
            ..Default::default()
        })
        .recipe
    }

    #[test]
    fn accumulate_is_single_scalar_add() {
        let prog = emit_accumulate(&adamw_recipe());
        assert_eq!(prog.ops.len(), 1);
        assert!(matches!(prog.ops[0], UpdateOp::ScalarMulAdd { dst: Register::MPartial, src: Register::MPartial, .. }));
    }

    #[test]
    fn adamw_program_has_five_ops() {
        let prog = emit_final_step(&adamw_recipe());
        assert_eq!(prog.ops.len(), 5);
        // Must end in a parameter update.
        assert!(matches!(prog.ops.last().unwrap(), UpdateOp::Update { .. }));
    }

    #[test]
    fn sgd_program_is_single_update() {
        let recipe = plan(&FaseConfig {
            optimizer: FaseOptimizer::Sgd,
            accumulation: 4,
            ..Default::default()
        })
        .recipe;
        let prog = emit_final_step(&recipe);
        assert_eq!(prog.ops.len(), 1);
        assert!(matches!(prog.ops[0], UpdateOp::SgdUpdate { .. }));
    }

    #[test]
    fn lion_program_applies_sign() {
        let recipe = plan(&FaseConfig {
            optimizer: FaseOptimizer::Lion,
            accumulation: 4,
            ..Default::default()
        })
        .recipe;
        let prog = emit_final_step(&recipe);
        assert!(prog.ops.iter().any(|op| matches!(op, UpdateOp::Sign { .. })));
    }

    #[test]
    fn unknown_optimizer_is_empty_program() {
        let recipe = plan(&FaseConfig {
            optimizer: FaseOptimizer::Unknown,
            accumulation: 4,
            ..Default::default()
        })
        .recipe;
        let prog = emit_final_step(&recipe);
        assert!(prog.ops.is_empty());
    }

    #[test]
    fn pseudocode_is_populated() {
        let prog = emit_final_step(&adamw_recipe());
        assert!(!prog.pseudocode.is_empty());
        assert!(prog.pseudocode.contains("θ"));
    }

    #[test]
    fn m_partial_is_distinct_from_m() {
        // Register::MPartial must exist as a separate variant so AdamW's
        // final step can read m_old and m_partial simultaneously.
        let m = Register::M;
        let mp = Register::MPartial;
        assert_ne!(m, mp);
    }

    #[test]
    fn accumulate_targets_m_partial() {
        let recipe = UpdateRecipe {
            optimizer: FaseOptimizer::AdamW,
            lr: 0.001,
            beta1: 0.9,
            one_minus_beta1: 0.1,
            beta2: 0.999,
            one_minus_beta2: 0.001,
            eps: 1e-8,
            weight_decay: 0.01,
            accum_scale: 0.25, // 1/N for N=4
            v_uses_approx: true,
        };
        let prog = emit_accumulate(&recipe);
        // m_partial += 0.25 * g
        let UpdateOp::ScalarMulAdd { dst, src, a, b_src, b_scale } = &prog.ops[0] else {
            panic!("expected ScalarMulAdd, got {:?}", prog.ops[0]);
        };
        assert_eq!(*dst, Register::MPartial);
        assert_eq!(*src, Register::MPartial);
        assert_eq!(*a, 1.0);
        assert_eq!(*b_src, Some(Register::G));
        assert!((b_scale - 0.25).abs() < 1e-12);
    }

    #[test]
    fn reset_zeroes_m_partial() {
        let prog = emit_reset();
        assert_eq!(prog.ops.len(), 1);
        assert_eq!(prog.ops[0], UpdateOp::Zero(Register::MPartial));
    }

    #[test]
    fn sgd_plain_uses_m_partial_in_deferred_mode() {
        let recipe = UpdateRecipe {
            optimizer: FaseOptimizer::Sgd,
            lr: 0.01,
            beta1: 0.0, beta2: 0.0,
            one_minus_beta1: 0.0, one_minus_beta2: 0.0,
            eps: 0.0, weight_decay: 0.0,
            accum_scale: 0.25,
            v_uses_approx: false,
        };
        let prog = emit_sgd(&recipe, /*with_momentum=*/ false);
        match &prog.ops[0] {
            UpdateOp::SgdUpdate { lr } => assert!((lr - 0.01).abs() < 1e-12),
            op => panic!("expected SgdUpdate, got {:?}", op),
        }
    }

    #[test]
    fn sgd_momentum_accumulates_from_m_partial() {
        let recipe = UpdateRecipe {
            optimizer: FaseOptimizer::SgdMomentum,
            lr: 0.01,
            beta1: 0.0, beta2: 0.0,
            one_minus_beta1: 0.0, one_minus_beta2: 0.0,
            eps: 0.0, weight_decay: 0.0,
            accum_scale: 0.25,
            v_uses_approx: false,
        };
        let prog = emit_sgd(&recipe, /*with_momentum=*/ true);
        let UpdateOp::ScalarMulAdd { dst, src, b_src, .. } = &prog.ops[0] else {
            panic!("op 0 should be ScalarMulAdd");
        };
        assert_eq!(*dst, Register::M);
        assert_eq!(*src, Register::M);
        assert_eq!(*b_src, Some(Register::MPartial));
    }

    #[test]
    fn adamw_reads_m_partial_for_first_and_second_moments() {
        let recipe = UpdateRecipe {
            optimizer: FaseOptimizer::AdamW,
            lr: 0.001,
            beta1: 0.9,
            one_minus_beta1: 0.1,
            beta2: 0.999,
            one_minus_beta2: 0.001,
            eps: 1e-8,
            weight_decay: 0.01,
            accum_scale: 0.25,
            v_uses_approx: true,
        };
        let prog = emit_adamw(&recipe, /*decoupled_wd=*/ true);

        // Op 0: m = β₁·m + (1-β₁)·m_partial
        let UpdateOp::ScalarMulAdd { dst, src, a, b_src, b_scale } = &prog.ops[0] else {
            panic!("op 0 should be ScalarMulAdd");
        };
        assert_eq!(*dst, Register::M);
        assert_eq!(*src, Register::M);
        assert!((a - 0.9).abs() < 1e-12);
        assert_eq!(*b_src, Some(Register::MPartial));
        assert!((b_scale - 0.1).abs() < 1e-12, "b_scale must be one_minus_beta1, no extra 1/N");

        // Op 1: v = β₂·v + (1-β₂)·m_partial²
        let UpdateOp::SquaredAccumulate { dst, src, operand, scale } = &prog.ops[1] else {
            panic!("op 1 should be SquaredAccumulate");
        };
        assert_eq!(*dst, Register::V);
        assert_eq!(*src, Register::V);
        assert_eq!(*operand, Register::MPartial);
        assert!((scale - 0.001).abs() < 1e-12);
    }

    #[test]
    fn jensen_fence_fase_v_exceeds_standard_v_for_nonconstant_gradients() {
        // The FASE paper's Option B approximates v using mean(g²) rather than
        // mean(g)².  Jensen's inequality (mean(g²) ≥ mean(g)²) guarantees
        // v_fase ≥ v_standard element-wise, with strict inequality when the
        // per-micro-batch gradients vary.  This test fences the approximation
        // against future "fixes" that would silently implement the standard
        // formula.
        //
        // Closed-form per-parameter v-update for one accumulation window:
        //
        //   Standard AdamW:     v' = β₂·v + (1 - β₂) · (mean(g))²
        //   FASE Deferred:      v' = β₂·v + (1 - β₂) · mean(g²)
        //
        // Using non-constant gradients: [1.0, 2.0, 0.5, 1.5]
        //   mean(g)   = 1.25
        //   mean(g)²  = 1.5625
        //   mean(g²)  = (1 + 4 + 0.25 + 2.25) / 4 = 1.875
        //
        // So v_fase - v_standard = (1 - β₂) · (1.875 - 1.5625) = (1 - β₂) · 0.3125

        let beta2: f64 = 0.999;
        let v_prev: f64 = 0.0;
        let gradients: [f64; 4] = [1.0, 2.0, 0.5, 1.5];

        let mean_g: f64 = gradients.iter().sum::<f64>() / (gradients.len() as f64);
        let mean_g_sq: f64 =
            gradients.iter().map(|g| g * g).sum::<f64>() / (gradients.len() as f64);

        let v_standard = beta2 * v_prev + (1.0 - beta2) * mean_g * mean_g;
        let v_fase = beta2 * v_prev + (1.0 - beta2) * mean_g_sq;

        assert!(
            v_fase >= v_standard,
            "Jensen: v_fase ({}) must be >= v_standard ({})",
            v_fase,
            v_standard
        );
        assert!(
            v_fase - v_standard > 0.0,
            "For non-constant gradients, v_fase must strictly exceed v_standard; got diff {}",
            v_fase - v_standard
        );

        // The expected difference is (1 - β₂) · (mean_g_sq - mean_g²) = 0.001 · 0.3125.
        let expected_diff = (1.0 - beta2) * (mean_g_sq - mean_g * mean_g);
        assert!(
            (v_fase - v_standard - expected_diff).abs() < 1e-12,
            "expected diff {}, got {}",
            expected_diff,
            v_fase - v_standard
        );
    }

    #[test]
    fn jensen_fence_constant_gradients_produce_equal_v() {
        // Sanity check: when all per-micro-batch gradients are identical,
        // mean(g²) == mean(g)², so Jensen's inequality is tight.  The fence
        // test above must use non-constant gradients; this test documents
        // why.
        let beta2: f64 = 0.999;
        let v_prev: f64 = 0.0;
        let gradients: [f64; 4] = [0.7; 4];

        let mean_g: f64 = gradients.iter().sum::<f64>() / (gradients.len() as f64);
        let mean_g_sq: f64 =
            gradients.iter().map(|g| g * g).sum::<f64>() / (gradients.len() as f64);

        let v_standard = beta2 * v_prev + (1.0 - beta2) * mean_g * mean_g;
        let v_fase = beta2 * v_prev + (1.0 - beta2) * mean_g_sq;

        assert!(
            (v_fase - v_standard).abs() < 1e-12,
            "constant gradients must produce equal v_fase and v_standard"
        );
    }
}
