//! M40: Reverse-mode AD rules — maps each primal operation to its adjoint computation.

use crate::wengert::{PrimalOp, VarId, WengertOp};

/// Primitive and compound backward operations used in adjoint expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum AdjointExpr {
    MulElementwise(VarId, VarId),
    MatmulTransposeLeft(VarId, VarId),
    MatmulTransposeRight(VarId, VarId),
    Scale(VarId, f64),
    Negate(VarId),
    Broadcast(VarId),
    ScaleBroadcast(VarId, f64),
    Transpose(VarId, usize, usize),
    Reshape(VarId),
    Identity(VarId),
    // Compound backward rules (multi-step, lowered to op sequences in M40b)
    ExpBackward(VarId, VarId),
    ReluBackward(VarId, VarId),
    SigmoidBackward(VarId, VarId),
    TanhBackward(VarId, VarId),
    LogBackward(VarId, VarId),
    SqrtBackward(VarId, VarId),
    DivNumeratorBackward(VarId, VarId),
    DivDenominatorBackward(VarId, VarId, VarId),
    // Control flow backward rules
    /// SelectTrue: adj_true = cond ? adj_out : 0.  inputs: (adj_out, cond_var)
    SelectTrue(VarId, VarId),
    /// SelectFalse: adj_false = !cond ? adj_out : 0.  inputs: (adj_out, cond_var)
    SelectFalse(VarId, VarId),
}

/// A single input-adjoint pair from applying an AD rule.
#[derive(Debug, Clone)]
pub struct InputAdjoint {
    pub input_var: VarId,
    pub expr: AdjointExpr,
}

/// Apply the reverse-mode AD rule for a primal operation.
pub fn apply_ad_rule(op: &WengertOp, output_bar: VarId) -> Vec<InputAdjoint> {
    match &op.op {
        PrimalOp::Add => vec![
            InputAdjoint { input_var: op.inputs[0], expr: AdjointExpr::Identity(output_bar) },
            InputAdjoint { input_var: op.inputs[1], expr: AdjointExpr::Identity(output_bar) },
        ],
        PrimalOp::Sub => vec![
            InputAdjoint { input_var: op.inputs[0], expr: AdjointExpr::Identity(output_bar) },
            InputAdjoint { input_var: op.inputs[1], expr: AdjointExpr::Negate(output_bar) },
        ],
        PrimalOp::Mul => vec![
            InputAdjoint { input_var: op.inputs[0], expr: AdjointExpr::MulElementwise(output_bar, op.inputs[1]) },
            InputAdjoint { input_var: op.inputs[1], expr: AdjointExpr::MulElementwise(output_bar, op.inputs[0]) },
        ],
        PrimalOp::Div => vec![
            InputAdjoint { input_var: op.inputs[0], expr: AdjointExpr::DivNumeratorBackward(output_bar, op.inputs[1]) },
            InputAdjoint { input_var: op.inputs[1], expr: AdjointExpr::DivDenominatorBackward(output_bar, op.inputs[0], op.inputs[1]) },
        ],
        PrimalOp::Neg => vec![
            InputAdjoint { input_var: op.inputs[0], expr: AdjointExpr::Negate(output_bar) },
        ],
        PrimalOp::Relu => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::ReluBackward(output_bar, op.inputs[0]),
        }],
        PrimalOp::Sigmoid => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::SigmoidBackward(output_bar, op.result),
        }],
        PrimalOp::Tanh => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::TanhBackward(output_bar, op.result),
        }],
        PrimalOp::Exp => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::ExpBackward(output_bar, op.result),
        }],
        PrimalOp::Log => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::LogBackward(output_bar, op.inputs[0]),
        }],
        PrimalOp::Sqrt => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::SqrtBackward(output_bar, op.result),
        }],
        PrimalOp::Matmul => vec![
            InputAdjoint { input_var: op.inputs[0], expr: AdjointExpr::MatmulTransposeLeft(output_bar, op.inputs[1]) },
            InputAdjoint { input_var: op.inputs[1], expr: AdjointExpr::MatmulTransposeRight(op.inputs[0], output_bar) },
        ],
        PrimalOp::Transpose { dim0, dim1 } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::Transpose(output_bar, *dim0, *dim1),
        }],
        PrimalOp::Sum { .. } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::Broadcast(output_bar),
        }],
        PrimalOp::Mean { .. } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::ScaleBroadcast(output_bar, 1.0),
        }],
        PrimalOp::Reshape { .. } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::Reshape(output_bar),
        }],
        // Select(cond, true_val, false_val) -> result
        // d(result)/d(true_val) = cond ? 1 : 0, so adj_true = cond ? adj_out : 0
        // d(result)/d(false_val) = cond ? 0 : 1, so adj_false = !cond ? adj_out : 0
        // cond is non-differentiable — no adjoint propagated to inputs[0]
        PrimalOp::Select => {
            let cond_var = op.inputs[0];
            vec![
                InputAdjoint {
                    input_var: op.inputs[1],
                    expr: AdjointExpr::SelectTrue(output_bar, cond_var),
                },
                InputAdjoint {
                    input_var: op.inputs[2],
                    expr: AdjointExpr::SelectFalse(output_bar, cond_var),
                },
            ]
        }
        // Condition is non-differentiable — no adjoints to propagate
        PrimalOp::Condition(_) => vec![],
        _ => vec![],
    }
}

/// What a backward rule needs saved from the forward pass.
#[derive(Debug, Clone, PartialEq)]
pub enum SavedRequirement {
    Nothing,
    Inputs,
    Output,
}

/// Determine which variables an AD rule needs saved from the forward pass.
pub fn saved_for_backward(op: &PrimalOp) -> SavedRequirement {
    match op {
        PrimalOp::Add | PrimalOp::Sub | PrimalOp::Neg
        | PrimalOp::Transpose { .. } | PrimalOp::Reshape { .. }
        | PrimalOp::Sum { .. } | PrimalOp::Mean { .. }
        | PrimalOp::Broadcast => SavedRequirement::Nothing,
        PrimalOp::Mul | PrimalOp::Div | PrimalOp::Matmul
        | PrimalOp::Relu | PrimalOp::Log | PrimalOp::Abs
        | PrimalOp::Gelu | PrimalOp::Silu => SavedRequirement::Inputs,
        PrimalOp::Sigmoid | PrimalOp::Tanh | PrimalOp::Exp
        | PrimalOp::Sqrt | PrimalOp::Softmax { .. } => SavedRequirement::Output,
        // Select needs the condition saved (input[0]), plus both branch values
        PrimalOp::Select => SavedRequirement::Inputs,
        // Condition is non-differentiable — nothing needed
        PrimalOp::Condition(_) => SavedRequirement::Nothing,
        _ => SavedRequirement::Nothing,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::WengertOp;

    fn make_op(result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp { id: 0, result, op, inputs, saved_for_backward: false, checkpointed: false }
    }

    #[test]
    fn test_add_rule() {
        let op = make_op(2, PrimalOp::Add, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 2);
        assert!(matches!(adj[0].expr, AdjointExpr::Identity(100)));
        assert!(matches!(adj[1].expr, AdjointExpr::Identity(100)));
    }

    #[test]
    fn test_sub_rule() {
        let op = make_op(2, PrimalOp::Sub, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::Identity(100)));
        assert!(matches!(adj[1].expr, AdjointExpr::Negate(100)));
    }

    #[test]
    fn test_mul_rule() {
        let op = make_op(2, PrimalOp::Mul, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::MulElementwise(100, 1)));
        assert!(matches!(adj[1].expr, AdjointExpr::MulElementwise(100, 0)));
    }

    #[test]
    fn test_matmul_rule() {
        let op = make_op(2, PrimalOp::Matmul, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::MatmulTransposeLeft(100, 1)));
        assert!(matches!(adj[1].expr, AdjointExpr::MatmulTransposeRight(0, 100)));
    }

    #[test]
    fn test_relu_backward() {
        let op = make_op(1, PrimalOp::Relu, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::ReluBackward(100, 0)));
    }

    #[test]
    fn test_sigmoid_backward() {
        let op = make_op(1, PrimalOp::Sigmoid, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::SigmoidBackward(100, 1)));
    }

    #[test]
    fn test_tanh_backward() {
        let op = make_op(1, PrimalOp::Tanh, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::TanhBackward(100, 1)));
    }

    #[test]
    fn test_log_backward() {
        let op = make_op(1, PrimalOp::Log, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::LogBackward(100, 0)));
    }

    #[test]
    fn test_sqrt_backward() {
        let op = make_op(1, PrimalOp::Sqrt, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::SqrtBackward(100, 1)));
    }

    #[test]
    fn test_div_backward() {
        let op = make_op(2, PrimalOp::Div, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::DivNumeratorBackward(100, 1)));
        assert!(matches!(adj[1].expr, AdjointExpr::DivDenominatorBackward(100, 0, 1)));
    }

    #[test]
    fn test_sum_broadcasts() {
        let op = make_op(1, PrimalOp::Sum { dim: Some(0) }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::Broadcast(100)));
    }

    #[test]
    fn test_mean_scale_broadcasts() {
        let op = make_op(1, PrimalOp::Mean { dim: Some(0) }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::ScaleBroadcast(100, _)));
    }

    #[test]
    fn test_transpose_rule() {
        let op = make_op(1, PrimalOp::Transpose { dim0: 0, dim1: 1 }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::Transpose(100, 0, 1)));
    }

    #[test]
    fn test_saved_nothing() {
        assert_eq!(saved_for_backward(&PrimalOp::Add), SavedRequirement::Nothing);
        assert_eq!(saved_for_backward(&PrimalOp::Transpose { dim0: 0, dim1: 1 }), SavedRequirement::Nothing);
    }

    #[test]
    fn test_saved_inputs() {
        assert_eq!(saved_for_backward(&PrimalOp::Mul), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::Matmul), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::Relu), SavedRequirement::Inputs);
    }

    #[test]
    fn test_saved_output() {
        assert_eq!(saved_for_backward(&PrimalOp::Sigmoid), SavedRequirement::Output);
        assert_eq!(saved_for_backward(&PrimalOp::Tanh), SavedRequirement::Output);
        assert_eq!(saved_for_backward(&PrimalOp::Exp), SavedRequirement::Output);
    }

    // --- Control-flow AD rules ---

    #[test]
    fn test_select_rule() {
        // Select(cond=0, true_val=1, false_val=2) -> result=3
        // apply_ad_rule should return two InputAdjoint entries:
        //   inputs[1] (true_val)  -> SelectTrue(output_bar, cond_var)
        //   inputs[2] (false_val) -> SelectFalse(output_bar, cond_var)
        // No adjoint is propagated to inputs[0] (the condition is non-differentiable).
        let op = make_op(3, PrimalOp::Select, vec![0, 1, 2]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 2, "Select should produce exactly two input adjoints");
        // First: adjoint for the true branch value
        assert_eq!(adj[0].input_var, 1, "First adjoint targets true_val (inputs[1])");
        assert!(
            matches!(adj[0].expr, AdjointExpr::SelectTrue(100, 0)),
            "Expected SelectTrue(output_bar=100, cond_var=0), got {:?}",
            adj[0].expr
        );
        // Second: adjoint for the false branch value
        assert_eq!(adj[1].input_var, 2, "Second adjoint targets false_val (inputs[2])");
        assert!(
            matches!(adj[1].expr, AdjointExpr::SelectFalse(100, 0)),
            "Expected SelectFalse(output_bar=100, cond_var=0), got {:?}",
            adj[1].expr
        );
    }

    #[test]
    fn test_condition_rule() {
        // Condition is non-differentiable — apply_ad_rule should return empty vec.
        use crate::wengert::CompareKind;
        let op = make_op(1, PrimalOp::Condition(CompareKind::Gt), vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(adj.is_empty(), "Condition op should have no adjoints (non-differentiable)");
    }

    #[test]
    fn test_saved_select() {
        // Select needs the condition + both branch values saved for backward.
        assert_eq!(
            saved_for_backward(&PrimalOp::Select),
            SavedRequirement::Inputs,
            "Select should save its inputs (cond, true_val, false_val)"
        );
    }

    #[test]
    fn test_saved_condition() {
        // Condition is non-differentiable — nothing needs to be saved.
        use crate::wengert::CompareKind;
        assert_eq!(
            saved_for_backward(&PrimalOp::Condition(CompareKind::Gt)),
            SavedRequirement::Nothing,
            "Condition op should not save anything"
        );
    }
}
