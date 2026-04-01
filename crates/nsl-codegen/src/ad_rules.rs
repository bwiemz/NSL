//! M40: Reverse-mode AD rules — maps each primal operation to its adjoint computation.

use crate::wengert::{PrimalOp, VarId, WengertOp};

/// Primitive and compound backward operations used in adjoint expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum AdjointExpr {
    MulElementwise(VarId, VarId),
    MatmulTransposeLeft(VarId, VarId),
    /// MatmulTransposeRight(a, grad, b) — grad_b = reduce_to_shape(a.T @ grad, b)
    /// The third field `b` is the original weight for shape reduction.
    MatmulTransposeRight(VarId, VarId, VarId),
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
    // New elementwise backward rules
    /// GELU backward: grad * (0.5*(1+erf(x/√2)) + x*exp(-x²/2)/√(2π))
    GeluBackward(VarId, VarId),
    /// SiLU backward: grad * (σ(x) + x*σ(x)*(1-σ(x)))
    SiluBackward(VarId, VarId),
    /// Abs backward: sign(x) * grad
    SignMul(VarId, VarId),
    /// Clamp backward: grad * (min <= x <= max), with actual min/max bounds
    ClampBackward(VarId, VarId, f64, f64),

    // Softmax/LogSoftmax backward
    /// Softmax backward: grad - sum(grad * y) * y  (y = softmax output)
    SoftmaxBackward(VarId, VarId),
    /// LogSoftmax backward: grad - exp(y) * sum(grad)  (y = log_softmax output)
    LogSoftmaxBackward(VarId, VarId),

    // Normalization backward
    /// LayerNorm backward: 3 adjoint components for input, gamma, beta
    /// args: (grad, input, mean_unused, rstd_unused, eps)
    LayerNormBackward(VarId, VarId, VarId, VarId, f64),
    /// BatchNorm backward: similar to LayerNorm but over batch dimension
    /// args: (grad, input, mean_unused, rstd_unused, eps)
    BatchNormBackward(VarId, VarId, VarId, VarId, f64),
    /// Gamma gradient for normalization: grad * x_hat where x_hat = (x - mean) / std
    /// args: (grad, input, eps, dim, weight) — recomputes x_hat from input, reduces to weight shape
    NormGammaBackward(VarId, VarId, f64, i64, VarId),

    // Regularization
    /// Dropout backward: grad * mask / (1-p).  args: (grad, mask)
    DropoutBackward(VarId, VarId, f64),

    // Indexing backward
    /// Embedding backward: scatter_add(grad, indices, weight). args: (grad, indices, weight_var)
    EmbeddingBackward(VarId, VarId, VarId),
    /// Gather backward: scatter_add(grad, indices, dim). args: (grad, indices, dim)
    GatherBackward(VarId, VarId, i64),
    /// ScatterAdd backward for src: gather(grad, indices). args: (grad, indices, dim)
    ScatterAddSrcBackward(VarId, VarId, i64),

    // Shape backward
    /// Concat backward: split grad along dim. args: (grad, dim, offset, size)
    ConcatSplit(VarId, i64, usize, usize),
    /// Split backward: concat grads along dim
    SplitConcat(VarId, i64),
    /// Slice backward: zero-pad grad into original shape.
    /// args: (grad, dim, start, end, orig_dim_size)
    SliceBackward(VarId, i64, i64, i64, i64),

    // Convolution/pooling backward
    /// Conv2d backward for input: conv_transpose(grad, weight, stride, padding)
    ConvTransposeInput(VarId, VarId, usize, usize),
    /// Conv2d backward for weight: conv(input^T, grad)
    ConvTransposeWeight(VarId, VarId),
    /// MaxPool backward: grad * (input == max_value). args: (grad, argmax_indices)
    MaxPoolBackward(VarId, VarId),
    /// AvgPool backward: grad / pool_size, broadcast to pool region
    AvgPoolBackward(VarId, usize),

    // Loss backward
    /// CrossEntropy backward: y_bar * (softmax(logits) - one_hot(target)). args: (y_bar, logits, targets)
    CrossEntropyBackward(VarId, VarId, VarId),
    /// MSE backward: 2*(pred - target)/n. args: (grad, pred, target)
    MSEBackward(VarId, VarId, VarId),
    /// L1 backward: sign(pred - target)/n. args: (grad, pred, target)
    L1Backward(VarId, VarId, VarId),

    // Attention backward — per-component (Q, K, V) for correct causal masking
    /// Attention backward for Q: args: (grad, Q, K, V, causal)
    AttentionBackwardQ(VarId, VarId, VarId, VarId, bool),
    /// Attention backward for K: args: (grad, Q, K, V, causal)
    AttentionBackwardK(VarId, VarId, VarId, VarId, bool),
    /// Attention backward for V: args: (grad, Q, K, V, causal)
    AttentionBackwardV(VarId, VarId, VarId, VarId, bool),
    /// RoPE backward: rotate grad by negative angle. args: (grad, dim)
    RoPEBackward(VarId, usize),

    // Shape backward (broadcast reduction)
    /// Expand backward: sum-reduce gradient over expanded dims to match original shape.
    /// args: (grad, original_input) — original_input provides the target shape.
    ReduceToShape(VarId, VarId),

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
            InputAdjoint { input_var: op.inputs[1], expr: AdjointExpr::MatmulTransposeRight(op.inputs[0], output_bar, op.inputs[1]) },
        ],
        PrimalOp::Transpose { dim0, dim1 } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::Transpose(output_bar, *dim0, *dim1),
        }],
        PrimalOp::Sum { .. } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::Broadcast(output_bar),
        }],
        // Mean backward: broadcast(grad) / n.  The exact 1/n factor depends on the
        // reduced dimension size which isn't available in the Wengert list.  The tape-based
        // runtime backward (backward.rs) handles this correctly with num_elements.
        // For source AD analysis, we emit Broadcast (correct structure, conservative scale).
        PrimalOp::Mean { .. } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::Broadcast(output_bar),
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
        // --- New elementwise unary rules ---
        PrimalOp::Gelu => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::GeluBackward(output_bar, op.inputs[0]),
        }],
        PrimalOp::Silu => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::SiluBackward(output_bar, op.inputs[0]),
        }],
        PrimalOp::Abs => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::SignMul(output_bar, op.inputs[0]),
        }],
        PrimalOp::Clamp { min, max } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::ClampBackward(output_bar, op.inputs[0], *min, *max),
        }],

        // --- Softmax / LogSoftmax ---
        PrimalOp::Softmax { .. } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::SoftmaxBackward(output_bar, op.result),
        }],
        PrimalOp::LogSoftmax { .. } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::LogSoftmaxBackward(output_bar, op.result),
        }],

        // --- Normalization ---
        // LayerNorm(input, gamma, beta) -> output
        // Backward needs: grad, input, saved_mean (inputs[3]), saved_rstd (inputs[4])
        // For simplicity, we save input and use the output's VarId for mean/rstd
        PrimalOp::LayerNorm { eps } => {
            let input = op.inputs[0];
            let mut adjoints = vec![InputAdjoint {
                input_var: input,
                expr: AdjointExpr::LayerNormBackward(output_bar, input, op.result, op.result, *eps),
            }];
            // gamma gradient: grad * x_hat (normalized input, NOT the output)
            if op.inputs.len() > 1 {
                adjoints.push(InputAdjoint {
                    input_var: op.inputs[1],
                    expr: AdjointExpr::NormGammaBackward(output_bar, input, *eps, -1, op.inputs[1]),
                });
            }
            // beta gradient: identity (grad flows through)
            if op.inputs.len() > 2 {
                adjoints.push(InputAdjoint {
                    input_var: op.inputs[2],
                    expr: AdjointExpr::Identity(output_bar),
                });
            }
            adjoints
        }
        // RMSNorm(input, weight) -> output  (no bias, eps is compile-time constant)
        PrimalOp::RMSNorm { eps } => {
            let input = op.inputs[0];
            let mut adjoints = vec![InputAdjoint {
                input_var: input,
                expr: AdjointExpr::LayerNormBackward(output_bar, input, op.result, op.result, *eps),
            }];
            // weight gradient: grad * x_hat
            if op.inputs.len() > 1 {
                adjoints.push(InputAdjoint {
                    input_var: op.inputs[1],
                    expr: AdjointExpr::NormGammaBackward(output_bar, input, *eps, -1, op.inputs[1]),
                });
            }
            adjoints
        }
        PrimalOp::BatchNorm { eps, .. } => {
            let input = op.inputs[0];
            let mut adjoints = vec![InputAdjoint {
                input_var: input,
                expr: AdjointExpr::BatchNormBackward(output_bar, input, op.result, op.result, *eps),
            }];
            // gamma gradient: grad * x_hat (normalized input, NOT the output)
            if op.inputs.len() > 1 {
                adjoints.push(InputAdjoint {
                    input_var: op.inputs[1],
                    expr: AdjointExpr::NormGammaBackward(output_bar, input, *eps, 0, op.inputs[1]),
                });
            }
            if op.inputs.len() > 2 {
                adjoints.push(InputAdjoint {
                    input_var: op.inputs[2],
                    expr: AdjointExpr::Identity(output_bar),
                });
            }
            adjoints
        }

        // --- Dropout ---
        PrimalOp::Dropout { p } => vec![InputAdjoint {
            input_var: op.inputs[0],
            // inputs[1] is the dropout mask (saved from forward)
            expr: AdjointExpr::DropoutBackward(
                output_bar,
                if op.inputs.len() > 1 { op.inputs[1] } else { op.inputs[0] },
                1.0 / (1.0 - p),
            ),
        }],

        // --- Indexing ---
        PrimalOp::Embedding => vec![InputAdjoint {
            input_var: op.inputs[0],
            // Pass weight (inputs[0]) so backward can size the output to match
            expr: AdjointExpr::EmbeddingBackward(output_bar, op.inputs[1], op.inputs[0]),
        }],
        PrimalOp::Gather { dim } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::GatherBackward(output_bar, op.inputs[1], *dim),
        }],
        PrimalOp::ScatterAdd { dim } => {
            // scatter_add(input, indices, src) -> output
            // d/d(input) = identity, d/d(src) = gather(grad, indices)
            let mut adjoints = vec![InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::Identity(output_bar),
            }];
            if op.inputs.len() > 2 {
                adjoints.push(InputAdjoint {
                    input_var: op.inputs[2],
                    expr: AdjointExpr::ScatterAddSrcBackward(output_bar, op.inputs[1], *dim),
                });
            }
            adjoints
        }

        // --- Shape ops ---
        PrimalOp::Concat { dim } => {
            // Each input gets a slice of the gradient
            let mut adjoints = Vec::with_capacity(op.inputs.len());
            for (i, &input) in op.inputs.iter().enumerate() {
                adjoints.push(InputAdjoint {
                    input_var: input,
                    expr: AdjointExpr::ConcatSplit(output_bar, *dim, i, 1),
                });
            }
            adjoints
        }
        PrimalOp::Split { dim, .. } => {
            // Split backward = concat all grads along the split dim
            vec![InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::SplitConcat(output_bar, *dim),
            }]
        }
        PrimalOp::Slice { dim, start, end, orig_dim_size } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::SliceBackward(output_bar, *dim, *start, *end, *orig_dim_size),
        }],

        // --- Convolution ---
        PrimalOp::Conv2d { stride, padding } => vec![
            InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::ConvTransposeInput(output_bar, op.inputs[1], *stride, *padding),
            },
            InputAdjoint {
                input_var: op.inputs[1],
                expr: AdjointExpr::ConvTransposeWeight(op.inputs[0], output_bar),
            },
        ],

        // --- Pooling ---
        PrimalOp::MaxPool2d { .. } => vec![InputAdjoint {
            input_var: op.inputs[0],
            // inputs[1] would be argmax indices saved from forward
            expr: AdjointExpr::MaxPoolBackward(
                output_bar,
                if op.inputs.len() > 1 { op.inputs[1] } else { op.result },
            ),
        }],
        PrimalOp::AvgPool2d { kernel, .. } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::AvgPoolBackward(output_bar, kernel * kernel),
        }],

        // --- Loss functions ---
        PrimalOp::CrossEntropyLoss => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::CrossEntropyBackward(output_bar, op.inputs[0], op.inputs[1]),
        }],
        PrimalOp::MSELoss => {
            let pred = op.inputs[0];
            let target = op.inputs[1];
            vec![InputAdjoint {
                input_var: pred,
                expr: AdjointExpr::MSEBackward(output_bar, pred, target),
            }]
        }
        PrimalOp::L1Loss => {
            let pred = op.inputs[0];
            let target = op.inputs[1];
            vec![InputAdjoint {
                input_var: pred,
                expr: AdjointExpr::L1Backward(output_bar, pred, target),
            }]
        }

        // --- Attention ---
        PrimalOp::ScaledDotProductAttention { causal } => {
            // Q, K, V are inputs[0..3]
            let q = op.inputs[0];
            let k = op.inputs[1];
            let v = op.inputs[2];
            vec![
                InputAdjoint { input_var: q, expr: AdjointExpr::AttentionBackwardQ(output_bar, q, k, v, *causal) },
                InputAdjoint { input_var: k, expr: AdjointExpr::AttentionBackwardK(output_bar, q, k, v, *causal) },
                InputAdjoint { input_var: v, expr: AdjointExpr::AttentionBackwardV(output_bar, q, k, v, *causal) },
            ]
        }
        PrimalOp::RoPE { dim } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::RoPEBackward(output_bar, *dim),
        }],

        // Condition is non-differentiable — no adjoints to propagate
        PrimalOp::Condition(_) => vec![],
        // Passthrough ops: only shape-preserving ones propagate gradients.
        // Non-differentiable metadata ops (shape, ndim, item, int, subscript, list, arange, etc.)
        // do NOT propagate gradients — they produce non-tensor values.
        PrimalOp::Passthrough(name) => {
            match name.as_str() {
                // Shape-preserving identity: gradient flows through unchanged
                "reshape" | "contiguous" | "squeeze" | "unsqueeze"
                | "cos" | "sin" | "rotate_half" => {
                    if op.inputs.is_empty() { vec![] }
                    else {
                        vec![InputAdjoint {
                            input_var: op.inputs[0],
                            expr: AdjointExpr::Identity(output_bar),
                        }]
                    }
                }
                // Expand backward: sum-reduce gradient over broadcast-expanded dims
                "expand" => {
                    if op.inputs.is_empty() { vec![] }
                    else {
                        vec![InputAdjoint {
                            input_var: op.inputs[0],
                            expr: AdjointExpr::ReduceToShape(output_bar, op.inputs[0]),
                        }]
                    }
                }
                // Non-differentiable: no gradient propagation
                _ => vec![],
            }
        }
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
        // Nothing saved — gradient is independent of forward values
        PrimalOp::Add | PrimalOp::Sub | PrimalOp::Neg
        | PrimalOp::Transpose { .. } | PrimalOp::Reshape { .. }
        | PrimalOp::Sum { .. } | PrimalOp::Mean { .. }
        | PrimalOp::Broadcast
        | PrimalOp::Concat { .. } | PrimalOp::Split { .. } | PrimalOp::Slice { .. }
        | PrimalOp::RoPE { .. } | PrimalOp::RoPEInverse { .. }
        | PrimalOp::AvgPool2d { .. }
        | PrimalOp::Passthrough(_) => SavedRequirement::Nothing,

        // Save inputs — gradient depends on forward input values
        PrimalOp::Mul | PrimalOp::Div | PrimalOp::Matmul
        | PrimalOp::Relu | PrimalOp::Log | PrimalOp::Abs
        | PrimalOp::Gelu | PrimalOp::Silu | PrimalOp::Clamp { .. }
        | PrimalOp::LayerNorm { .. } | PrimalOp::RMSNorm { .. } | PrimalOp::BatchNorm { .. }
        | PrimalOp::Dropout { .. }
        | PrimalOp::Embedding | PrimalOp::Gather { .. } | PrimalOp::ScatterAdd { .. }
        | PrimalOp::Conv2d { .. }
        | PrimalOp::CrossEntropyLoss | PrimalOp::MSELoss | PrimalOp::L1Loss
        | PrimalOp::ScaledDotProductAttention { .. }
        | PrimalOp::Select => SavedRequirement::Inputs,

        // Save output — gradient depends on forward output values
        PrimalOp::Sigmoid | PrimalOp::Tanh | PrimalOp::Exp
        | PrimalOp::Sqrt | PrimalOp::Softmax { .. } | PrimalOp::LogSoftmax { .. }
        | PrimalOp::MaxPool2d { .. } => SavedRequirement::Output,

        // Non-differentiable — nothing needed
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
        assert!(matches!(adj[1].expr, AdjointExpr::MatmulTransposeRight(0, 100, 1)));
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
    fn test_mean_broadcasts() {
        // Mean backward: broadcast grad (1/n scaling deferred to runtime tape backward)
        let op = make_op(1, PrimalOp::Mean { dim: Some(0) }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::Broadcast(100)));
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

    // --- Tier 1: Trivial new rules ---

    #[test]
    fn test_gelu_backward() {
        let op = make_op(1, PrimalOp::Gelu, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::GeluBackward(100, 0)));
        assert_eq!(saved_for_backward(&PrimalOp::Gelu), SavedRequirement::Inputs);
    }

    #[test]
    fn test_silu_backward() {
        let op = make_op(1, PrimalOp::Silu, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::SiluBackward(100, 0)));
        assert_eq!(saved_for_backward(&PrimalOp::Silu), SavedRequirement::Inputs);
    }

    #[test]
    fn test_abs_backward() {
        let op = make_op(1, PrimalOp::Abs, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::SignMul(100, 0)));
        assert_eq!(saved_for_backward(&PrimalOp::Abs), SavedRequirement::Inputs);
    }

    #[test]
    fn test_softmax_backward() {
        let op = make_op(1, PrimalOp::Softmax { dim: -1 }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::SoftmaxBackward(100, 1)));
        assert_eq!(saved_for_backward(&PrimalOp::Softmax { dim: -1 }), SavedRequirement::Output);
    }

    #[test]
    fn test_clamp_backward() {
        let op = make_op(1, PrimalOp::Clamp { min: 0.0, max: 1.0 }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::ClampBackward(100, 0, _, _)));
        assert_eq!(saved_for_backward(&PrimalOp::Clamp { min: 0.0, max: 1.0 }), SavedRequirement::Inputs);
    }

    // --- Tier 2: Normalization / indexing / shape ---

    #[test]
    fn test_layer_norm_backward() {
        let op = make_op(3, PrimalOp::LayerNorm { eps: 1e-5 }, vec![0, 1, 2]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 3, "LayerNorm should produce 3 adjoints (input, gamma, beta)");
        assert!(matches!(adj[0].expr, AdjointExpr::LayerNormBackward(100, 0, 3, 3, _)));
        assert!(matches!(adj[1].expr, AdjointExpr::NormGammaBackward(100, 0, _, -1, 1))); // gamma grad
        assert!(matches!(adj[2].expr, AdjointExpr::Identity(100))); // beta grad
        assert_eq!(saved_for_backward(&PrimalOp::LayerNorm { eps: 1e-5 }), SavedRequirement::Inputs);
    }

    #[test]
    fn test_dropout_backward() {
        let op = make_op(2, PrimalOp::Dropout { p: 0.1 }, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        match &adj[0].expr {
            AdjointExpr::DropoutBackward(y_bar, mask, scale) => {
                assert_eq!(*y_bar, 100);
                assert_eq!(*mask, 1); // inputs[1] = mask
                assert!((scale - 1.0 / 0.9).abs() < 1e-6); // 1/(1-0.1)
            }
            other => panic!("Expected DropoutBackward, got {:?}", other),
        }
        assert_eq!(saved_for_backward(&PrimalOp::Dropout { p: 0.1 }), SavedRequirement::Inputs);
    }

    #[test]
    fn test_embedding_backward() {
        let op = make_op(2, PrimalOp::Embedding, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::EmbeddingBackward(100, 1, 0)));
    }

    #[test]
    fn test_gather_backward() {
        let op = make_op(2, PrimalOp::Gather { dim: 1 }, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::GatherBackward(100, 1, 1)));
    }

    #[test]
    fn test_concat_backward() {
        let op = make_op(3, PrimalOp::Concat { dim: 0 }, vec![0, 1, 2]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 3, "Concat of 3 inputs should produce 3 adjoints");
        for (i, a) in adj.iter().enumerate() {
            assert_eq!(a.input_var, i as VarId);
            assert!(matches!(a.expr, AdjointExpr::ConcatSplit(100, 0, _, 1)));
        }
    }

    #[test]
    fn test_slice_backward() {
        let op = make_op(1, PrimalOp::Slice { dim: 0, start: 2, end: 5, orig_dim_size: 10 }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::SliceBackward(100, 0, 2, 5, 10)));
    }

    // --- Tier 3: Compound ops ---

    #[test]
    fn test_conv2d_backward() {
        let op = make_op(2, PrimalOp::Conv2d { stride: 1, padding: 0 }, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 2, "Conv2d should produce 2 adjoints (input, weight)");
        assert!(matches!(adj[0].expr, AdjointExpr::ConvTransposeInput(100, 1, 1, 0)));
        assert!(matches!(adj[1].expr, AdjointExpr::ConvTransposeWeight(0, 100)));
    }

    #[test]
    fn test_maxpool_backward() {
        let op = make_op(2, PrimalOp::MaxPool2d { kernel: 2, stride: 2 }, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::MaxPoolBackward(100, 1)));
        assert_eq!(saved_for_backward(&PrimalOp::MaxPool2d { kernel: 2, stride: 2 }), SavedRequirement::Output);
    }

    #[test]
    fn test_avgpool_backward() {
        let op = make_op(1, PrimalOp::AvgPool2d { kernel: 3, stride: 3 }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::AvgPoolBackward(100, 9))); // kernel*kernel = 9
        assert_eq!(saved_for_backward(&PrimalOp::AvgPool2d { kernel: 3, stride: 3 }), SavedRequirement::Nothing);
    }

    #[test]
    fn test_cross_entropy_backward() {
        let op = make_op(2, PrimalOp::CrossEntropyLoss, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::CrossEntropyBackward(100, 0, 1)));
        assert_eq!(saved_for_backward(&PrimalOp::CrossEntropyLoss), SavedRequirement::Inputs);
    }

    #[test]
    fn test_mse_loss_backward() {
        let op = make_op(2, PrimalOp::MSELoss, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::MSEBackward(100, 0, 1)));
    }

    #[test]
    fn test_l1_loss_backward() {
        let op = make_op(2, PrimalOp::L1Loss, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::L1Backward(100, 0, 1)));
    }

    // --- Tier 4: Attention ---

    #[test]
    fn test_attention_backward() {
        let op = make_op(3, PrimalOp::ScaledDotProductAttention { causal: true }, vec![0, 1, 2]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 3, "Attention should produce 3 adjoints (Q, K, V)");
        assert!(matches!(adj[0].expr, AdjointExpr::AttentionBackwardQ(100, 0, 1, 2, true)));
        assert!(matches!(adj[1].expr, AdjointExpr::AttentionBackwardK(100, 0, 1, 2, true)));
        assert!(matches!(adj[2].expr, AdjointExpr::AttentionBackwardV(100, 0, 1, 2, true)));
        assert_eq!(
            saved_for_backward(&PrimalOp::ScaledDotProductAttention { causal: true }),
            SavedRequirement::Inputs
        );
    }

    #[test]
    fn test_rope_backward() {
        let op = make_op(1, PrimalOp::RoPE { dim: 64 }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::RoPEBackward(100, 64)));
        assert_eq!(saved_for_backward(&PrimalOp::RoPE { dim: 64 }), SavedRequirement::Nothing);
    }

    #[test]
    fn test_log_softmax_backward() {
        let op = make_op(1, PrimalOp::LogSoftmax { dim: -1 }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::LogSoftmaxBackward(100, 1)));
        assert_eq!(saved_for_backward(&PrimalOp::LogSoftmax { dim: -1 }), SavedRequirement::Output);
    }

    // --- Saved requirement tests for new ops ---

    #[test]
    fn test_saved_new_nothing_ops() {
        assert_eq!(saved_for_backward(&PrimalOp::Concat { dim: 0 }), SavedRequirement::Nothing);
        assert_eq!(saved_for_backward(&PrimalOp::Split { dim: 0, chunks: 2 }), SavedRequirement::Nothing);
        assert_eq!(saved_for_backward(&PrimalOp::Slice { dim: 0, start: 0, end: 5, orig_dim_size: 10 }), SavedRequirement::Nothing);
        assert_eq!(saved_for_backward(&PrimalOp::AvgPool2d { kernel: 2, stride: 2 }), SavedRequirement::Nothing);
        assert_eq!(saved_for_backward(&PrimalOp::RoPE { dim: 64 }), SavedRequirement::Nothing);
    }

    #[test]
    fn test_saved_new_input_ops() {
        assert_eq!(saved_for_backward(&PrimalOp::Gelu), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::Silu), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::Clamp { min: 0.0, max: 1.0 }), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::LayerNorm { eps: 1e-5 }), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::BatchNorm { eps: 1e-5, training: true }), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::Dropout { p: 0.1 }), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::Embedding), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::Gather { dim: 0 }), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::Conv2d { stride: 1, padding: 0 }), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::CrossEntropyLoss), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::MSELoss), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::ScaledDotProductAttention { causal: false }), SavedRequirement::Inputs);
    }

    #[test]
    fn test_saved_new_output_ops() {
        assert_eq!(saved_for_backward(&PrimalOp::Softmax { dim: -1 }), SavedRequirement::Output);
        assert_eq!(saved_for_backward(&PrimalOp::LogSoftmax { dim: -1 }), SavedRequirement::Output);
        assert_eq!(saved_for_backward(&PrimalOp::MaxPool2d { kernel: 2, stride: 2 }), SavedRequirement::Output);
    }

    #[test]
    fn test_expand_backward_reduce_to_shape() {
        // expand backward should produce ReduceToShape, NOT Identity
        let op = make_op(2, PrimalOp::Passthrough("expand".into()), vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert_eq!(adj[0].input_var, 0);
        assert!(matches!(adj[0].expr, AdjointExpr::ReduceToShape(100, 0)));
    }

    #[test]
    fn test_reshape_backward_identity() {
        // reshape backward should still be Identity (not ReduceToShape)
        let op = make_op(1, PrimalOp::Passthrough("reshape".into()), vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::Identity(100)));
    }
}
