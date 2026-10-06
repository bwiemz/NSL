//! M40: Reverse-mode AD rules — maps each primal operation to its adjoint computation.

use crate::csha_apply::FusionMark;
use crate::flash_attention_v2::smem_layout::{self, Direction};
use crate::wengert::{ConvGradKind, NormEps, PrimalOp, VarId, WengertOp};

/// Tier C (T5.2): decision the reverse-walk dispatcher makes when it
/// encounters a Wengert op that belongs to a CSHA-claimed chain.
///
/// The dispatcher's contract is: walk the Wengert list in reverse
/// topological order; on the FIRST claimed op it sees (which must be
/// the chain's output op per spec §5.4), decide how to handle the
/// chain's entire backward pass. Subsequent claimed ops for the same
/// chain are no-ops (the fused backward already emitted gradients for
/// all of them in one kernel).
#[derive(Debug, Clone, PartialEq)]
pub enum CshaDispatchDecision {
    /// Emit the fused Tier C backward kernel for this mark. The caller
    /// is expected to:
    ///   1. Invoke `synthesize_backward(&mark.config)` + launch via
    ///      `nsl_flash_attention_csha_backward`.
    ///   2. Register the 7 output VarIds (dQ/dK/dV/dWq/dWk/dWv/dx) in
    ///      the tape's gradient map.
    /// After emission, `mark.backward_emitted` is set to true so the
    /// same chain's other claimed ops return `AlreadyEmitted`.
    EmitFused,
    /// This claimed op belongs to a chain whose fused backward has
    /// already been emitted earlier in the reverse walk. The
    /// dispatcher does nothing — gradients are already in the tape.
    AlreadyEmitted,
    /// The config validator rejected this chain for the backward
    /// direction (e.g. SMEM budget exceeded). Caller must fall back to
    /// per-op adjoint rules (`apply_ad_rule` for each individual op).
    /// The diagnostic string is propagated so the user sees WHY the
    /// fused path was rejected.
    Fallback { diagnostic: String },
}

/// Dispatch decision for a single reverse-walk encounter of a claimed
/// op. See `CshaDispatchDecision` for the contract.
///
/// `op_idx` is the Wengert op index currently being processed. It is
/// NOT used to select the mark (the caller supplies the matching mark
/// already) but is kept in the signature so a future orchestrator can
/// verify the spec §5.4 reverse-walk invariant via debug_assert — i.e.
/// the FIRST claimed op the walk hits must be the chain's output op.
pub fn csha_dispatch_for_op(mark: &FusionMark, op_idx: u32) -> CshaDispatchDecision {
    let _ = op_idx; // reserved for spec §5.4 reverse-walk invariant check

    if mark.backward_emitted.get() {
        return CshaDispatchDecision::AlreadyEmitted;
    }

    // The mark must carry a config — Tier C dispatchers need the full
    // FlashAttentionConfig to run the backward-direction validator.
    // Without it we can't safely emit the fused path; fall back.
    let Some(cfg) = mark.config.as_ref() else {
        return CshaDispatchDecision::Fallback {
            diagnostic: format!(
                "CSHA fused backward unavailable for layer {}: \
                 mark carries no FlashAttentionConfig (pre-Tier-C plan)",
                mark.layer
            ),
        };
    };

    match smem_layout::validate_scalar_v2_config(cfg, Direction::Backward) {
        Ok(()) => {
            mark.backward_emitted.set(true);
            CshaDispatchDecision::EmitFused
        }
        Err(e) => CshaDispatchDecision::Fallback {
            diagnostic: format!(
                "CSHA fused backward rejected for layer {}: {e}; \
                 falling back to per-op adjoints",
                mark.layer
            ),
        },
    }
}

/// Primitive and compound backward operations used in adjoint expressions.
#[derive(Debug, Clone, PartialEq)]
pub enum AdjointExpr {
    /// MulElementwise(grad, other, target) — emits `grad * other` then
    /// `reduce_to_shape(prod, target)`. `target` is the input variable whose
    /// gradient we are accumulating; the reduce-to-shape is a no-op when no
    /// broadcasting occurred, and a sum-reduction over the broadcast axes
    /// when it did. Mirrors `MatmulTransposeRight`'s third-field pattern.
    MulElementwise(VarId, VarId, VarId),
    /// MatmulTransposeLeft(grad, b, a) — grad_a = reduce_to_shape(grad @ b.T, a).
    /// `a @ b` broadcasts a lower-rank `a` against a batched `b`, so the raw
    /// product can carry batch dims `a` does not have.
    MatmulTransposeLeft(VarId, VarId, VarId),
    /// MatmulTransposeRight(a, grad, b) — grad_b = reduce_to_shape(a.T @ grad, b)
    /// The third field `b` is the original weight for shape reduction.
    MatmulTransposeRight(VarId, VarId, VarId),
    Scale(VarId, f64),
    Negate(VarId),
    Broadcast(VarId),
    /// MeanBackward(grad, input, result): the adjoint of a full mean, `grad`
    /// scaled by `numel(result) / numel(input)` (1/N), then expanded to the
    /// input's shape. The ratio is taken at run time, from the mean's own
    /// operand and result, because the Wengert list does not carry shapes.
    MeanBackward(VarId, VarId, VarId),
    /// ExpandLike(grad, input): the adjoint of a full sum, the one-element
    /// `grad` expanded to (and materialized at) the input's shape. Every
    /// element of the input contributed once, so every element receives the
    /// gradient; leaving it one element relied on each consumer to broadcast
    /// it, which a matmul's backward does not.
    ExpandLike(VarId, VarId),
    /// SumDimBackward(grad, input, dim): the adjoint of a sum over `dim`
    /// (keepdim off). The reduced dim is re-inserted at `dim` and the gradient
    /// expanded to the input's shape.
    SumDimBackward(VarId, VarId, i64),
    /// MeanDimBackward(grad, input, result, dim): a mean over `dim`, scaled
    /// like `MeanBackward` (by numel(result)/numel(input), i.e. 1/size(dim))
    /// and then re-expanded like `SumDimBackward`.
    MeanDimBackward(VarId, VarId, VarId, i64),
    ScaleBroadcast(VarId, f64),
    Transpose(VarId, usize, usize),
    ReshapeLike(VarId, VarId),
    Identity(VarId),
    // Compound backward rules (multi-step, lowered to op sequences in M40b)
    ExpBackward(VarId, VarId),
    ReluBackward(VarId, VarId),
    SigmoidBackward(VarId, VarId),
    TanhBackward(VarId, VarId),
    LogBackward(VarId, VarId),
    SqrtBackward(VarId, VarId),
    /// DivNumeratorBackward(grad, b, a): `grad / b`, reduced to the
    /// numerator `a`'s shape.
    DivNumeratorBackward(VarId, VarId, VarId),
    /// DivDenominatorBackward(grad, a, b): `-grad * a / b²`, reduced to the
    /// denominator `b`'s shape (`x / sum(x)` broadcasts a scalar `b`).
    DivDenominatorBackward(VarId, VarId, VarId),
    // New elementwise backward rules
    /// GELU backward: grad * (0.5*(1+erf(x/√2)) + x*exp(-x²/2)/√(2π))
    GeluBackward(VarId, VarId),
    /// SiLU backward: grad * (σ(x) + x*σ(x)*(1-σ(x)))
    SiluBackward(VarId, VarId),
    /// Abs backward: sign(x) * grad
    SignMul(VarId, VarId),
    /// cos backward: `-sin(x) * grad`. args: (grad, x)
    CosBackward(VarId, VarId),
    /// sin backward: `cos(x) * grad`. args: (grad, x)
    SinBackward(VarId, VarId),
    /// Clamp backward: grad * (min <= x <= max), with actual min/max bounds
    ClampBackward(VarId, VarId, f64, f64),

    // Softmax/LogSoftmax backward
    /// Softmax backward: y * (grad - sum_dim(grad * y))  (y = softmax output),
    /// the sum taken along the softmax's own `dim`. args: (grad, y, dim)
    SoftmaxBackward(VarId, VarId, i64),
    /// LogSoftmax backward: grad - exp(y) * sum_dim(grad)  (y = log_softmax
    /// output), along `dim`. args: (grad, y, dim)
    LogSoftmaxBackward(VarId, VarId, i64),

    // Normalization backward
    /// LayerNorm INPUT gradient. args: (grad, input, gamma, eps). For
    /// `y = gamma * x_hat + beta` the input gradient is the plain
    /// normalization backward of `grad * gamma`; `None` means no gamma.
    LayerNormBackward(VarId, VarId, Option<VarId>, NormEps),
    /// BatchNorm backward: similar to LayerNorm but over batch dimension
    /// args: (grad, input, mean_unused, rstd_unused, eps)
    BatchNormBackward(VarId, VarId, VarId, VarId, f64),
    /// Gamma gradient for LayerNorm / BatchNorm: grad * x_hat where
    /// `x_hat = (x - mean) / std`.
    /// args: (grad, input, eps, dim, weight) — recomputes x_hat from input, reduces to weight shape.
    ///
    /// NOT valid for RMSNorm (which does NOT mean-subtract); use
    /// `RmsNormGammaBackward` instead.
    NormGammaBackward(VarId, VarId, NormEps, i64, VarId),
    /// Gamma gradient for RMSNorm: `grad * x_hat` where
    /// `x_hat = x / rms` and `rms = sqrt(mean(x^2) + eps)` over the last
    /// dimension (keepdim).  Unlike `NormGammaBackward`, this does NOT
    /// subtract the per-row mean from `x` before normalizing, which matches
    /// RMSNorm's forward definition `y = gamma * x / rms`.
    /// args: (grad, input, eps, weight)
    RmsNormGammaBackward(VarId, VarId, NormEps, VarId),
    /// INPUT gradient for RMSNorm — the correct dx that does NOT mean-subtract
    /// (RMSNorm's forward is `y = gamma * x / rms`, `rms = sqrt(mean(x²)+eps)`,
    /// with no per-row mean removal). Reusing `LayerNormBackward` here — as the
    /// code did before — computes the LayerNorm dx (which centers x), giving
    /// WRONG input gradients for RMSNorm and matching neither tape-AD nor the
    /// math. Formula: `dx_j = g_j·ȳ_j/rms − x_j·mean_k(ȳ_k·g_k·x_k)/rms³`.
    /// args: (grad, input, gamma, eps)
    RmsNormInputBackward(VarId, VarId, VarId, NormEps),

    // Regularization
    /// Dropout backward: grad * mask / (1-p).  args: (grad, mask)
    DropoutBackward(VarId, VarId, f64),

    // Indexing backward
    /// Embedding backward: scatter_add(grad, indices, weight). args: (grad, indices, weight_var)
    EmbeddingBackward(VarId, VarId, VarId),
    /// Gather backward: `grad` scattered into zeros shaped like `input`, at
    /// `indices` along `dim`. args: (grad, input, indices, dim)
    GatherBackward(VarId, VarId, VarId, i64),
    /// ScatterAdd backward for src: gather(grad, indices). args: (grad, indices, dim)
    ScatterAddSrcBackward(VarId, VarId, i64),

    // Shape backward
    /// Concat backward: the slice of `grad` along `dim` that concat operand
    /// `index` occupied. Its offset is the sum of the preceding operands'
    /// sizes along `dim`, read at run time (the Wengert list has no shapes).
    /// args: (grad, dim, index, operands)
    ConcatSplit(VarId, i64, usize, Vec<VarId>),
    /// Split backward: concat grads along dim
    SplitConcat(VarId, i64),
    /// Slice backward: zero-pad grad into original shape.
    /// args: (grad, dim, start, end, orig_dim_size)
    SliceBackward(VarId, i64, i64, i64, i64),

    // Convolution/pooling backward
    /// Conv2d gradient: `(kind, grad_output, input, weight, stride, padding)`.
    /// Lowers to `PrimalOp::Conv2dBackward`, delegating to the runtime FFI that
    /// wraps the verified nested-loop `conv2d_backward` (shared with tape AD).
    Conv2dBackward(ConvGradKind, VarId, VarId, VarId, usize, usize),
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
    /// The TARGET's gradient of an MSE / L1 loss: the pred gradient negated
    /// (both losses depend on `pred - target` alone) and reduced to the
    /// target's shape, in case the target was broadcast. args: (grad, pred,
    /// target)
    MSETargetBackward(VarId, VarId, VarId),
    L1TargetBackward(VarId, VarId, VarId),

    // Attention backward — per-component (Q, K, V) for correct causal masking
    /// Attention backward for Q: args: (grad, Q, K, V, fwd_result, causal,
    /// scale). `scale` is the forward's scale operand; `None` means the
    /// default `1/sqrt(head_dim)`.
    AttentionBackwardQ(VarId, VarId, VarId, VarId, VarId, bool, Option<VarId>),
    /// PCA Stage C: packed (segment-masked) attention backward —
    /// (output_bar, q, k, v, fwd_out, segment_ids). Causal-within-segment
    /// by contract, so no causal flag.
    AttentionBackwardQPacked(VarId, VarId, VarId, VarId, VarId, VarId, VarId),
    /// See [`AdjointExpr::AttentionBackwardQPacked`].
    AttentionBackwardKPacked(VarId, VarId, VarId, VarId, VarId, VarId, VarId),
    /// See [`AdjointExpr::AttentionBackwardQPacked`].
    AttentionBackwardVPacked(VarId, VarId, VarId, VarId, VarId, VarId, VarId),
    /// Attention backward for K: see [`AdjointExpr::AttentionBackwardQ`].
    AttentionBackwardK(VarId, VarId, VarId, VarId, VarId, bool, Option<VarId>),
    /// Attention backward for V: see [`AdjointExpr::AttentionBackwardQ`].
    AttentionBackwardV(VarId, VarId, VarId, VarId, VarId, bool, Option<VarId>),
    /// CFTP §4.4 G3 (Sprint 4): fused linear-CE backward — per-component extract.
    ///
    /// args: (grad, x, W, bias, targets, fwd_result, component,
    ///        vocab_size, hidden_size, batch_size, seq_len, vocab_tile,
    ///        ignore_index)
    ///
    /// Lowers to `PrimalOp::FusedLinearCeBackwardExtract` with the same
    /// `component` so the three components share one backward FFI launch
    /// via the cache `Compiler.fused_ce_bwd_cache` (mirrors the
    /// `FlashAttentionBackwardExtract` pattern).
    FusedLinearCeBackward {
        grad: VarId,
        x: VarId,
        w: VarId,
        bias: VarId,
        targets: VarId,
        fwd_result: VarId,
        component: u8,
        vocab_size: u32,
        hidden_size: u32,
        batch_size: u32,
        seq_len: u32,
        vocab_tile: u32,
        ignore_index: i64,
        /// Sprint 2.5 (see `PrimalOp::FusedLinearCe::has_bias`).
        has_bias: bool,
        /// Sprint 2.5 (see `PrimalOp::FusedLinearCe::x_rank3`).
        x_rank3: bool,
    },
    /// CPKD: fused KL-CE distillation backward — STUDENT components only
    /// (0 = dx_s, 1 = dW_s, 2 = dbias_s). The teacher inputs receive NO
    /// adjoints by construction (composition-paper invariant I-11); the
    /// rule simply never emits them.
    ///
    /// Lowers to `PrimalOp::FusedKlCeBackwardExtract` sharing one backward
    /// FFI launch across the three components via
    /// `Compiler.fused_kl_ce_bwd_cache` (mirrors `FusedLinearCeBackward`).
    FusedKlCeBackward {
        grad: VarId,
        /// Forward inputs in op order: [x_s, W_s, bias_s, x_t, W_t, bias_t, targets].
        fwd_inputs: [VarId; 7],
        fwd_result: VarId,
        component: u8,
        vocab_size: u32,
        student_hidden: u32,
        teacher_hidden: u32,
        batch_size: u32,
        seq_len: u32,
        vocab_tile: u32,
        ignore_index: i64,
        alpha_bits: u64,
        temperature_bits: u64,
    },
    /// RoPE backward: rotate grad by negative angle. args: (grad, dim)
    RoPEBackward(VarId, usize),
    /// rotate_half backward: -rotate_half(grad).
    /// rotate_half is the Jacobian of itself but reflected; the inverse is
    /// the negation of itself. args: (grad)
    RotateHalfBackward(VarId),

    // Shape backward (broadcast reduction)
    /// Expand backward: sum-reduce gradient over expanded dims to match original shape.
    /// args: (grad, original_input) — original_input provides the target shape.
    /// Also the Add/Sub operand adjoint: either operand may have been
    /// broadcast, and its gradient must be summed back to its own shape.
    ReduceToShape(VarId, VarId),
    /// Sub's subtrahend adjoint: `-reduce_to_shape(grad, target)`.
    NegReduceToShape(VarId, VarId),

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
        // Either operand may have been broadcast (a bias `x @ w + b`, a
        // scalar `x + mean(y)`), so each gradient is summed back to its
        // operand's shape. The reduce returns its input when no broadcast
        // happened, so the common case costs a shape compare.
        PrimalOp::Add => vec![
            InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::ReduceToShape(output_bar, op.inputs[0]),
            },
            InputAdjoint {
                input_var: op.inputs[1],
                expr: AdjointExpr::ReduceToShape(output_bar, op.inputs[1]),
            },
        ],
        PrimalOp::Sub => vec![
            InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::ReduceToShape(output_bar, op.inputs[0]),
            },
            InputAdjoint {
                input_var: op.inputs[1],
                expr: AdjointExpr::NegReduceToShape(output_bar, op.inputs[1]),
            },
        ],
        PrimalOp::Mul => vec![
            InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::MulElementwise(
                    output_bar,
                    op.inputs[1],
                    op.inputs[0],
                ),
            },
            InputAdjoint {
                input_var: op.inputs[1],
                expr: AdjointExpr::MulElementwise(
                    output_bar,
                    op.inputs[0],
                    op.inputs[1],
                ),
            },
        ],
        PrimalOp::Div => vec![
            InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::DivNumeratorBackward(output_bar, op.inputs[1], op.inputs[0]),
            },
            InputAdjoint {
                input_var: op.inputs[1],
                expr: AdjointExpr::DivDenominatorBackward(output_bar, op.inputs[0], op.inputs[1]),
            },
        ],
        PrimalOp::Neg => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::Negate(output_bar),
        }],
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
            InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::MatmulTransposeLeft(output_bar, op.inputs[1], op.inputs[0]),
            },
            InputAdjoint {
                input_var: op.inputs[1],
                expr: AdjointExpr::MatmulTransposeRight(op.inputs[0], output_bar, op.inputs[1]),
            },
        ],
        PrimalOp::Transpose { dim0, dim1 } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::Transpose(output_bar, *dim0, *dim1),
        }],
        // `sum(x)` is a full reduction; `sum(x, d)` / `sum(x, d, 0)` reduces
        // one dim (the extractor sends a keepdim sum to the tape). The dim's
        // gradient is the upstream gradient re-inserted at `d` and expanded.
        PrimalOp::Sum { dim: None } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::ExpandLike(output_bar, op.inputs[0]),
        }],
        PrimalOp::Sum { dim: Some(d) } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::SumDimBackward(output_bar, op.inputs[0], *d),
        }],
        // Mean backward: broadcast(grad) / n. The 1/n factor depends on the
        // reduced size, which the Wengert list does not carry, so it is taken
        // at run time (`MeanBackward` -> the `mean_grad_scale` passthrough).
        //
        // This rule used to emit a bare Broadcast, commented as "for source AD
        // analysis": but source AD LOWERS these rules, and never runs the
        // tape's `MeanReduce` backward. Every `mean(...)` in a source-AD step
        // got a gradient N times too large. AdamW is invariant to a uniform
        // scale, which hid it for a mean-reduced loss; SGD, clipping and any
        // mean inside a forward (where it scales only some paths) were wrong.
        PrimalOp::Mean { dim: None } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::MeanBackward(output_bar, op.inputs[0], op.result),
        }],
        PrimalOp::Mean { dim: Some(d) } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::MeanDimBackward(output_bar, op.inputs[0], op.result, *d),
        }],
        PrimalOp::Reshape { .. } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::ReshapeLike(output_bar, op.inputs[0]),
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
        PrimalOp::Softmax { dim } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::SoftmaxBackward(output_bar, op.result, *dim),
        }],
        PrimalOp::LogSoftmax { dim } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::LogSoftmaxBackward(output_bar, op.result, *dim),
        }],

        // --- Normalization ---
        // LayerNorm(input, gamma, beta) -> output
        // Backward needs: grad, input, saved_mean (inputs[3]), saved_rstd (inputs[4])
        // For simplicity, we save input and use the output's VarId for mean/rstd
        PrimalOp::LayerNorm { eps } => {
            let input = op.inputs[0];
            let mut adjoints = vec![InputAdjoint {
                input_var: input,
                expr: AdjointExpr::LayerNormBackward(output_bar, input, op.inputs.get(1).copied(), *eps),
            }];
            // gamma gradient: grad * x_hat (normalized input, NOT the output)
            if op.inputs.len() > 1 {
                adjoints.push(InputAdjoint {
                    input_var: op.inputs[1],
                    expr: AdjointExpr::NormGammaBackward(output_bar, input, *eps, -1, op.inputs[1]),
                });
            }
            // beta gradient: grad summed over every dim beta was broadcast
            // across (it used to flow through unreduced, at the input's shape)
            if op.inputs.len() > 2 {
                adjoints.push(InputAdjoint {
                    input_var: op.inputs[2],
                    expr: AdjointExpr::ReduceToShape(output_bar, op.inputs[2]),
                });
            }
            adjoints
        }
        // RMSNorm(input, weight, eps) -> output (no bias; eps is a constant or
        // the run-time operand, `NormEps`)
        PrimalOp::RMSNorm { eps } => {
            let input = op.inputs[0];
            let mut adjoints = Vec::new();
            // INPUT gradient. RMSNorm does NOT mean-subtract, so the correct dx
            // needs gamma and the RMSNorm rms. When gamma is present use the
            // dedicated `RmsNormInputBackward`; only if a (rare) gamma-less
            // RMSNorm appears do we fall back to the old `LayerNormBackward`
            // path (documented as approximate for that degenerate case).
            if op.inputs.len() > 1 {
                adjoints.push(InputAdjoint {
                    input_var: input,
                    expr: AdjointExpr::RmsNormInputBackward(output_bar, input, op.inputs[1], *eps),
                });
                // weight gradient: grad * (x / rms) — RMSNorm does NOT
                // mean-subtract. Using `NormGammaBackward` here would compute
                // x_hat = (x-mean)/std (the LayerNorm formulation), yielding
                // dgamma=0 for any constant row, masking real gradients.
                adjoints.push(InputAdjoint {
                    input_var: op.inputs[1],
                    expr: AdjointExpr::RmsNormGammaBackward(output_bar, input, *eps, op.inputs[1]),
                });
            } else {
                adjoints.push(InputAdjoint {
                    input_var: input,
                    expr: AdjointExpr::LayerNormBackward(output_bar, input, None, *eps),
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
                    expr: AdjointExpr::NormGammaBackward(output_bar, input, NormEps::Const(*eps), 0, op.inputs[1]),
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
        // Two-op split (see wengert.rs): `DropoutMask` draws the RNG
        // keep-mask (result = mask, inputs = [x]); `Dropout` applies it
        // (result = out, inputs = [x, mask]). inputs[1] is genuinely the
        // exact forward mask here — the pre-split layout put the raw call
        // args at inputs, so this rule multiplied the gradient by the p
        // ARGUMENT var (dx = dy·p/(1-p), decorrelated from the forward);
        // pinned wrong until 2026-08-16, now gated by
        // `dropout_backward_parity_gate`.
        PrimalOp::Dropout { p } => {
            let mask = *op.inputs.get(1).unwrap_or_else(|| {
                panic!(
                    "PrimalOp::Dropout requires inputs [x, mask] — got {} \
                     input(s). The extraction two-op split must emit \
                     DropoutMask first (wengert.rs)",
                    op.inputs.len()
                )
            });
            vec![InputAdjoint {
                input_var: op.inputs[0],
                expr: AdjointExpr::DropoutBackward(output_bar, mask, 1.0 / (1.0 - p)),
            }]
        }
        // The mask is an RNG draw — no gradient flows through it to x.
        PrimalOp::DropoutMask { .. } => vec![],

        // --- Indexing ---
        PrimalOp::Embedding => vec![InputAdjoint {
            input_var: op.inputs[0],
            // Pass weight (inputs[0]) so backward can size the output to match
            expr: AdjointExpr::EmbeddingBackward(output_bar, op.inputs[1], op.inputs[0]),
        }],
        PrimalOp::Gather { dim } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::GatherBackward(output_bar, op.inputs[0], op.inputs[1], *dim),
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
                    expr: AdjointExpr::ConcatSplit(output_bar, *dim, i, op.inputs.clone()),
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
        PrimalOp::Slice {
            dim,
            start,
            end,
            orig_dim_size,
        } => vec![InputAdjoint {
            input_var: op.inputs[0],
            expr: AdjointExpr::SliceBackward(output_bar, *dim, *start, *end, *orig_dim_size),
        }],

        // --- Convolution ---
        PrimalOp::Conv2d { stride, padding } => {
            // inputs = [input, weight] or [input, weight, bias]. Every gradient
            // uses the full (grad_output, input, weight) triple + stride/padding.
            let (input, weight) = (op.inputs[0], op.inputs[1]);
            let mut adjoints = vec![
                InputAdjoint {
                    input_var: input,
                    expr: AdjointExpr::Conv2dBackward(
                        ConvGradKind::Input,
                        output_bar,
                        input,
                        weight,
                        *stride,
                        *padding,
                    ),
                },
                InputAdjoint {
                    input_var: weight,
                    expr: AdjointExpr::Conv2dBackward(
                        ConvGradKind::Weight,
                        output_bar,
                        input,
                        weight,
                        *stride,
                        *padding,
                    ),
                },
            ];
            // Bias gradient only when a bias tensor was passed (3rd input).
            if op.inputs.len() >= 3 {
                adjoints.push(InputAdjoint {
                    input_var: op.inputs[2],
                    expr: AdjointExpr::Conv2dBackward(
                        ConvGradKind::Bias,
                        output_bar,
                        input,
                        weight,
                        *stride,
                        *padding,
                    ),
                });
            }
            adjoints
        }

        // --- Pooling ---
        PrimalOp::MaxPool2d { .. } => vec![InputAdjoint {
            input_var: op.inputs[0],
            // inputs[1] would be argmax indices saved from forward
            expr: AdjointExpr::MaxPoolBackward(
                output_bar,
                if op.inputs.len() > 1 {
                    op.inputs[1]
                } else {
                    op.result
                },
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
        // CFTP §4.4 G3 (Sprint 4): fused linear-CE backward.
        //
        // Produces three adjoints — dx, dW, dbias — each via a
        // `FusedLinearCeBackwardExtract` op that shares one backward
        // FFI launch via the codegen-side cache.
        //
        // inputs: [x, W, bias, targets].  Result is the scalar loss VarId,
        // which the lowering side uses as the cache key.
        PrimalOp::FusedLinearCe {
            vocab_size,
            hidden_size,
            batch_size,
            seq_len,
            vocab_tile,
            ignore_index,
            is_large: _,
            has_bias,
            x_rank3,
        } => {
            let x = op.inputs[0];
            let w = op.inputs[1];
            let bias = op.inputs[2];
            let targets = op.inputs[3];
            let fwd_result = op.result;
            let mk = |component: u8| AdjointExpr::FusedLinearCeBackward {
                grad: output_bar,
                x,
                w,
                bias,
                targets,
                fwd_result,
                component,
                vocab_size: *vocab_size,
                hidden_size: *hidden_size,
                batch_size: *batch_size,
                seq_len: *seq_len,
                vocab_tile: *vocab_tile,
                ignore_index: *ignore_index,
                has_bias: *has_bias,
                x_rank3: *x_rank3,
            };
            let mut adjoints = vec![
                InputAdjoint { input_var: x, expr: mk(0) },
                InputAdjoint { input_var: w, expr: mk(1) },
            ];
            // Sprint 2.5: a biasless head has no bias adjoint — inputs[2]
            // is a placeholder (w_var) that must NOT accumulate a dbias.
            // The backward cache evicts on the last EMITTED component,
            // which the lowering derives from has_bias.
            if *has_bias {
                adjoints.push(InputAdjoint { input_var: bias, expr: mk(2) });
            }
            adjoints
        }
        // CPKD: fused KL-CE distillation loss. Adjoints ONLY for the three
        // STUDENT inputs (x_s, W_s, bias_s = inputs[0..3]); the teacher
        // inputs (x_t, W_t, bias_t = inputs[3..6]) and targets get none —
        // zero-grad by omission enforces invariant I-11 at the rule level
        // (the teacher backward is never generated, not merely skipped).
        PrimalOp::FusedKlCe {
            vocab_size,
            student_hidden,
            teacher_hidden,
            batch_size,
            seq_len,
            vocab_tile,
            ignore_index,
            alpha_bits,
            temperature_bits,
        } => {
            let fwd_inputs: [VarId; 7] = [
                op.inputs[0],
                op.inputs[1],
                op.inputs[2],
                op.inputs[3],
                op.inputs[4],
                op.inputs[5],
                op.inputs[6],
            ];
            let fwd_result = op.result;
            (0u8..3u8)
                .map(|component| InputAdjoint {
                    input_var: fwd_inputs[component as usize],
                    expr: AdjointExpr::FusedKlCeBackward {
                        grad: output_bar,
                        fwd_inputs,
                        fwd_result,
                        component,
                        vocab_size: *vocab_size,
                        student_hidden: *student_hidden,
                        teacher_hidden: *teacher_hidden,
                        batch_size: *batch_size,
                        seq_len: *seq_len,
                        vocab_tile: *vocab_tile,
                        ignore_index: *ignore_index,
                        alpha_bits: *alpha_bits,
                        temperature_bits: *temperature_bits,
                    },
                })
                .collect()
        }
        PrimalOp::MSELoss => {
            let pred = op.inputs[0];
            let target = op.inputs[1];
            vec![
                InputAdjoint {
                    input_var: pred,
                    expr: AdjointExpr::MSEBackward(output_bar, pred, target),
                },
                InputAdjoint {
                    input_var: target,
                    expr: AdjointExpr::MSETargetBackward(output_bar, pred, target),
                },
            ]
        }
        PrimalOp::L1Loss => {
            let pred = op.inputs[0];
            let target = op.inputs[1];
            vec![
                InputAdjoint {
                    input_var: pred,
                    expr: AdjointExpr::L1Backward(output_bar, pred, target),
                },
                InputAdjoint {
                    input_var: target,
                    expr: AdjointExpr::L1TargetBackward(output_bar, pred, target),
                },
            ]
        }

        // --- Attention ---
        PrimalOp::ScaledDotProductAttention { causal } => {
            // Q, K, V are inputs[0..3]; fwd_result is the forward output VarId
            // needed by the backward lowering to look up the logsumexp buffer.
            let q = op.inputs[0];
            let k = op.inputs[1];
            let v = op.inputs[2];
            let fwd_result = op.result;
            // The backward must use the forward's scale; it used to assume
            // 1/sqrt(head_dim) whatever scale the call passed.
            let scale = op.inputs.get(3).copied();
            vec![
                InputAdjoint {
                    input_var: q,
                    expr: AdjointExpr::AttentionBackwardQ(
                        output_bar, q, k, v, fwd_result, *causal, scale,
                    ),
                },
                InputAdjoint {
                    input_var: k,
                    expr: AdjointExpr::AttentionBackwardK(
                        output_bar, q, k, v, fwd_result, *causal, scale,
                    ),
                },
                InputAdjoint {
                    input_var: v,
                    expr: AdjointExpr::AttentionBackwardV(
                        output_bar, q, k, v, fwd_result, *causal, scale,
                    ),
                },
            ]
        }
        // PCA Stage C: packed attention. Q/K/V get fused-backward extracts;
        // scale and segment_ids are non-differentiable (the Stage B
        // decomposed form produced a discarded mask adjoint — this op has
        // no mask input at all: the fallback derives it from segment_ids).
        PrimalOp::ScaledDotProductAttentionPacked => {
            let q = op.inputs[0];
            let k = op.inputs[1];
            let v = op.inputs[2];
            let scale = op.inputs[3];
            let seg = op.inputs[4];
            let fwd_result = op.result;
            vec![
                InputAdjoint {
                    input_var: q,
                    expr: AdjointExpr::AttentionBackwardQPacked(
                        output_bar, q, k, v, fwd_result, seg, scale,
                    ),
                },
                InputAdjoint {
                    input_var: k,
                    expr: AdjointExpr::AttentionBackwardKPacked(
                        output_bar, q, k, v, fwd_result, seg, scale,
                    ),
                },
                InputAdjoint {
                    input_var: v,
                    expr: AdjointExpr::AttentionBackwardVPacked(
                        output_bar, q, k, v, fwd_result, seg, scale,
                    ),
                },
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
            // Only listed passthroughs have a rule (and a certification
            // status); see `RULED_PASSTHROUGHS`.
            if !RULED_PASSTHROUGHS.contains(&name.as_str()) {
                return vec![];
            }
            match name.as_str() {
                // Reshape-like ops must restore the original input shape in backward.
                "reshape" | "squeeze" | "unsqueeze" => {
                    if op.inputs.is_empty() {
                        vec![]
                    } else {
                        vec![InputAdjoint {
                            input_var: op.inputs[0],
                            expr: AdjointExpr::ReshapeLike(output_bar, op.inputs[0]),
                        }]
                    }
                }
                // Shape-preserving identity: gradient flows through unchanged.
                "contiguous" | "sum_keepdim_last" | "mean_keepdim_last" => {
                    if op.inputs.is_empty() {
                        vec![]
                    } else {
                        vec![InputAdjoint {
                            input_var: op.inputs[0],
                            expr: AdjointExpr::Identity(output_bar),
                        }]
                    }
                }
                // cos/sin used to share the identity rule above, on the
                // theory that they only ever touch frozen RoPE tables; the
                // certificates `cos`/`sin` hold the real derivatives.
                "cos" => {
                    if op.inputs.is_empty() {
                        vec![]
                    } else {
                        vec![InputAdjoint {
                            input_var: op.inputs[0],
                            expr: AdjointExpr::CosBackward(output_bar, op.inputs[0]),
                        }]
                    }
                }
                "sin" => {
                    if op.inputs.is_empty() {
                        vec![]
                    } else {
                        vec![InputAdjoint {
                            input_var: op.inputs[0],
                            expr: AdjointExpr::SinBackward(output_bar, op.inputs[0]),
                        }]
                    }
                }
                // rotate_half is its own inverse up to sign:
                //   forward: y[..h] = -x[h..],   y[h..] = x[..h]
                //   backward: dx[..h] = dy[h..], dx[h..] = -dy[..h]
                // which equals -rotate_half(dy). Identity here is WRONG —
                // it routes the sin-channel gradient onto the wrong q/k
                // halves with the wrong sign every transformer layer.
                "rotate_half" => {
                    if op.inputs.is_empty() {
                        vec![]
                    } else {
                        vec![InputAdjoint {
                            input_var: op.inputs[0],
                            expr: AdjointExpr::RotateHalfBackward(output_bar),
                        }]
                    }
                }
                // Expand backward: sum-reduce gradient over broadcast-expanded dims
                "expand" => {
                    if op.inputs.is_empty() {
                        vec![]
                    } else {
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
        PrimalOp::Neg
        | PrimalOp::Transpose { .. }
        | PrimalOp::Reshape { .. }
        | PrimalOp::Broadcast
        | PrimalOp::Concat { .. }
        | PrimalOp::Split { .. }
        | PrimalOp::Slice { .. }
        | PrimalOp::RoPE { .. }
        | PrimalOp::RoPEInverse { .. }
        | PrimalOp::AvgPool2d { .. } => SavedRequirement::Nothing,
        // d cos(x) = -sin(x), d sin(x) = cos(x): both read x.
        PrimalOp::Passthrough(name) if name == "cos" || name == "sin" => {
            SavedRequirement::Inputs
        }
        PrimalOp::Passthrough(_) => SavedRequirement::Nothing,

        // Save inputs — gradient depends on forward input values. Add/Sub
        // read only their operands' SHAPES, to undo a broadcast, and a full
        // Sum its operand's, to expand its gradient back; the shape is read
        // off the live tensor.
        PrimalOp::Add
        | PrimalOp::Sub
        | PrimalOp::Sum { .. }
        | PrimalOp::Mul
        | PrimalOp::Div
        | PrimalOp::Matmul
        | PrimalOp::Relu
        | PrimalOp::Log
        | PrimalOp::Abs
        | PrimalOp::Gelu
        | PrimalOp::Silu
        | PrimalOp::Clamp { .. }
        | PrimalOp::LayerNorm { .. }
        | PrimalOp::RMSNorm { .. }
        | PrimalOp::BatchNorm { .. }
        | PrimalOp::Dropout { .. }
        | PrimalOp::Embedding
        | PrimalOp::Gather { .. }
        | PrimalOp::ScatterAdd { .. }
        | PrimalOp::Conv2d { .. }
        | PrimalOp::CrossEntropyLoss
        | PrimalOp::MSELoss
        | PrimalOp::L1Loss
        | PrimalOp::ScaledDotProductAttention { .. }
        | PrimalOp::ScaledDotProductAttentionPacked
        | PrimalOp::Select => SavedRequirement::Inputs,

        // Save output — gradient depends on forward output values
        PrimalOp::Sigmoid
        | PrimalOp::Tanh
        | PrimalOp::Exp
        | PrimalOp::Sqrt
        | PrimalOp::Softmax { .. }
        | PrimalOp::LogSoftmax { .. }
        | PrimalOp::MaxPool2d { .. } => SavedRequirement::Output,

        // The mask op's OUTPUT is what the paired Dropout's backward
        // multiplies by (DropoutBackward reads the mask var, which is this
        // op's result). The RNG draw is not replayable, so the mask must
        // survive to the backward.
        PrimalOp::DropoutMask { .. } => SavedRequirement::Output,

        // Non-differentiable — nothing needed
        PrimalOp::Condition(_) => SavedRequirement::Nothing,
        // `MeanBackward` reads the mean's operand AND its result (their
        // lengths give the 1/N); this enum has no "both", and the live table
        // that drives saving is `wrga_prune::save_requirements`.
        PrimalOp::Mean { .. } => SavedRequirement::Inputs,
        _ => SavedRequirement::Nothing,
    }
}

// ---------------------------------------------------------------------------
// Source-AD rule certification: coverage
// ---------------------------------------------------------------------------

/// How a `PrimalOp`'s source-AD gradient is certified against an independent
/// oracle. [`ad_cert_status`] matches every variant with no wildcard, so a new
/// op does not compile until it states one; `crates/nsl-cli/tests/
/// source_ad_rule_cert.rs` checks that each named certificate exists.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdCertStatus {
    /// Certified by these certificates in `source_ad_rule_cert.rs`: raw
    /// gradients of a source-level grad block against an f64
    /// central-difference oracle, under both source AD and the tape. An op
    /// that exists only in a GPU `train` block is certified instead by a
    /// compiled SGD step in `fused_loss_gradient_cert_gpu.rs` (that file's
    /// `GPU_CERTS` list).
    Certified(&'static [&'static str]),
    /// No gradient flows through it (leaves, comparisons, markers).
    NotDifferentiable(&'static str),
    /// Emitted only inside generated adjoint code, never as a primal op that
    /// `apply_ad_rule` differentiates.
    AdjointOnly(&'static str),
    /// No NSL source spelling reaches it as a primal op of a grad block, so
    /// there is nothing end-to-end to certify.
    Unreachable(&'static str),
    /// Reachable and differentiable, with no raw-gradient certificate of the
    /// source-AD rule yet; the reason says what exists instead. This is the
    /// campaign's visible debt.
    Uncertified(&'static str),
}

/// The certification status of every `PrimalOp` variant.
pub fn ad_cert_status(op: &PrimalOp) -> AdCertStatus {
    use AdCertStatus::*;
    match op {
        PrimalOp::Relu => Certified(&["relu"]),
        PrimalOp::Sigmoid => Certified(&["sigmoid"]),
        PrimalOp::Tanh => Certified(&["tanh"]),
        PrimalOp::Gelu => Certified(&["gelu"]),
        PrimalOp::Silu => Certified(&["silu"]),
        PrimalOp::Exp => Certified(&["exp"]),
        PrimalOp::Log => Certified(&["log"]),
        PrimalOp::Sqrt => Certified(&["sqrt"]),
        PrimalOp::Abs => Certified(&["abs"]),
        PrimalOp::Neg => Certified(&["neg"]),
        PrimalOp::Clamp { .. } => Certified(&["clamp"]),
        PrimalOp::Add => Certified(&["add_same", "add_row", "add_col", "add_one", "add_literal"]),
        PrimalOp::Sub => Certified(&["sub_row", "sub_col"]),
        PrimalOp::Mul => Certified(&["mul_row", "mul_col", "mul_literal"]),
        PrimalOp::Div => Certified(&["div_row", "div_col", "div_numerator_broadcast"]),
        PrimalOp::Matmul => {
            Certified(&["matmul_2d", "matmul_3d_2d", "matmul_2d_3d", "matmul_3d_3d"])
        }
        PrimalOp::Transpose { .. } => Certified(&["transpose", "transpose_3d"]),
        PrimalOp::Sum { .. } => {
            Certified(&["sum_all", "sum_dim", "sum_dim_neg", "sum_dim_keepdim", "sum_dim_last"])
        }
        PrimalOp::Mean { .. } => Certified(&["mean_all", "mean_dim", "mean_dim_last"]),
        PrimalOp::Softmax { .. } => Certified(&["softmax_last", "softmax_dim0", "softmax_mid"]),
        PrimalOp::LogSoftmax { .. } => {
            Certified(&["log_softmax_last", "log_softmax_dim0", "log_softmax_mid"])
        }
        PrimalOp::Reshape { .. } => AdjointOnly(
            "the full-reduction adjoints; NSL `.reshape` lowers to Passthrough(\"reshape\")",
        ),
        PrimalOp::Broadcast => AdjointOnly("the full-reduction and mean adjoints"),
        PrimalOp::Concat { .. } => Certified(&["cat_dim0", "cat_dim1", "cat_three", "cat_neg"]),
        PrimalOp::Split { .. } => Unreachable("the extractor never builds a Split primal"),
        PrimalOp::Slice { .. } => Unreachable("the extractor never builds a Slice primal"),
        PrimalOp::PadZero { .. } => AdjointOnly("the Slice adjoint"),
        PrimalOp::Gather { .. } => {
            Certified(&["gather", "gather_neg", "gather_dim0", "gather_mid"])
        }
        PrimalOp::ScatterAdd { .. } => AdjointOnly("the Gather and Embedding adjoints"),
        PrimalOp::Embedding => Certified(&["embedding"]),
        PrimalOp::LayerNorm { .. } => Certified(&[
            "layernorm",
            "layernorm_eps",
            "layernorm_3d",
            "layernorm_field_eps",
        ]),
        PrimalOp::RMSNorm { .. } => Certified(&["rmsnorm", "rmsnorm_eps", "rmsnorm_field_eps", "rmsnorm_field_eps_reused"]),
        PrimalOp::BatchNorm { .. } => {
            Unreachable("the extractor maps `batch_norm`, but no builtin or stdlib fn defines it")
        }
        PrimalOp::MaxPool2d { .. } => Unreachable(
            "the extractor has no `maxpool2d` mapping; a grad block using it falls back",
        ),
        PrimalOp::AvgPool2d { .. } => Unreachable("the extractor never builds an AvgPool2d primal"),
        PrimalOp::Conv2d { .. } => Certified(&["conv2d"]),
        PrimalOp::ConvTranspose2d { .. } => {
            Unreachable("the extractor never builds a ConvTranspose2d primal")
        }
        PrimalOp::Conv2dBackward { .. } => AdjointOnly("the Conv2d adjoint"),
        PrimalOp::MaterializeConvOutputGrad { .. } => AdjointOnly("the Conv2d adjoint"),
        PrimalOp::Repeat { .. } => AdjointOnly("the pooling adjoints"),
        PrimalOp::CrossEntropyLoss => Certified(&["cross_entropy"]),
        PrimalOp::MSELoss => Certified(&["mse_loss"]),
        PrimalOp::L1Loss => Certified(&["l1_loss"]),
        PrimalOp::ScaledDotProductAttention { .. } => {
            Certified(&["sdpa", "sdpa_causal", "sdpa_scale"])
        }
        PrimalOp::FlashAttentionBackwardExtract { .. } => AdjointOnly("the SDPA adjoint"),
        PrimalOp::ScaledDotProductAttentionPacked => Certified(&[
            "sdpa_packed",
            "sdpa_packed_docs",
            "sdpa_packed_batch",
            "sdpa_packed_scale",
            "sdpa_packed_step",
        ]),
        PrimalOp::FlashAttentionBackwardExtractPacked { .. } => {
            AdjointOnly("the packed SDPA adjoint")
        }
        PrimalOp::CshaFusedBackwardExtract { .. } => AdjointOnly("the CSHA fused backward"),
        PrimalOp::FusedCshaBackward { .. } => AdjointOnly("the CSHA fused backward"),
        PrimalOp::PrologueRecompute { .. } => NotDifferentiable("a CCR recompute marker"),
        PrimalOp::FreeTensor => NotDifferentiable("a lifetime marker"),
        PrimalOp::RoPE { .. } => Unreachable(
            "built only by CSHA chain matching; stdlib RoPE is `x*cos + rotate_half(x)*sin`",
        ),
        PrimalOp::RoPEInverse { .. } => AdjointOnly("the RoPE adjoint"),
        PrimalOp::FusedGatedLoraMatmul { .. } => Uncertified(
            "held to the unfused graph (wrga_adapter_runtime_equivalence.rs), not to an oracle",
        ),
        PrimalOp::FusedLoraMatmul { .. } => Uncertified(
            "sm>=80 only; the unfused LoRA path is held to an f64 reference by lora_adapter_training_gate.rs",
        ),
        PrimalOp::FusedIa3Matmul { .. } => Uncertified("forward fixtures only"),
        PrimalOp::FusedLinearCe { .. } => Certified(&["fused_linear_ce_step"]),
        PrimalOp::FusedLinearCeBackwardExtract { .. } => AdjointOnly("the FusedLinearCe adjoint"),
        PrimalOp::FusedKlCe { .. } => Certified(&["fused_kl_ce_step"]),
        PrimalOp::FusedKlCeBackwardExtract { .. } => AdjointOnly("the FusedKlCe adjoint"),
        PrimalOp::Dropout { .. } => Uncertified(
            "held to tape parity (dropout_backward_parity_gate.rs); a random mask has no finite-difference oracle",
        ),
        PrimalOp::DropoutMask { .. } => NotDifferentiable(
            "the RNG mask; the gradient flows through the Dropout that applies it",
        ),
        PrimalOp::Select => Uncertified("a tensor `if`/`else`; no certificate yet"),
        PrimalOp::Condition(_) => NotDifferentiable("a comparison"),
        PrimalOp::Input(_) | PrimalOp::Param(_) => NotDifferentiable("a leaf"),
        PrimalOp::Constant(_) => NotDifferentiable("a constant"),
        PrimalOp::Passthrough(name) => ad_cert_status_passthrough(name),
    }
}

/// The passthroughs `apply_ad_rule` gives an adjoint. Its passthrough arm
/// returns no adjoint for any other name, so a rule cannot fire until it is
/// listed here (and [`ad_cert_status_passthrough`] states its status).
pub const RULED_PASSTHROUGHS: &[&str] = &[
    "reshape",
    "squeeze",
    "unsqueeze",
    "contiguous",
    "cos",
    "sin",
    "sum_keepdim_last",
    "mean_keepdim_last",
    "rotate_half",
    "expand",
];

/// The certification status of a passthrough, by name.
pub fn ad_cert_status_passthrough(name: &str) -> AdCertStatus {
    use AdCertStatus::*;
    match name {
        "reshape" => Certified(&["reshape"]),
        "unsqueeze" => Certified(&["unsqueeze"]),
        "squeeze" => Uncertified("shares the reshape-like rule certified by `reshape`/`unsqueeze`"),
        "contiguous" => Certified(&["contiguous"]),
        "cos" => Certified(&["cos"]),
        "sin" => Certified(&["sin"]),
        "rotate_half" => Certified(&["rotate_half"]),
        "expand" => Certified(&["expand"]),
        "sum_keepdim_last" | "mean_keepdim_last" => {
            AdjointOnly("the softmax and normalization adjoints")
        }
        _ => NotDifferentiable("a passthrough with no adjoint rule"),
    }
}

/// Every `PrimalOp` variant (one sample each, passthroughs by rule name) with
/// its status. The coverage gate reads this; a unit test holds it to the
/// enum's definition in `wengert.rs`.
pub fn ad_cert_inventory() -> Vec<(String, AdCertStatus)> {
    let mut out: Vec<(String, AdCertStatus)> = ad_cert_samples()
        .iter()
        .map(|op| (variant_name(op), ad_cert_status(op)))
        .collect();
    out.extend(
        RULED_PASSTHROUGHS
            .iter()
            .map(|n| (format!("Passthrough({n})"), ad_cert_status_passthrough(n))),
    );
    out
}

/// The variant name of `op` (its `Debug` text up to the first field).
fn variant_name(op: &PrimalOp) -> String {
    format!("{op:?}")
        .chars()
        .take_while(|c| c.is_alphanumeric())
        .collect()
}

/// One sample of every `PrimalOp` variant except `Passthrough`.
fn ad_cert_samples() -> Vec<PrimalOp> {
    use crate::wengert::{CompareKind, ConvGradKind, SubgraphId};
    vec![
        PrimalOp::Relu,
        PrimalOp::Sigmoid,
        PrimalOp::Tanh,
        PrimalOp::Gelu,
        PrimalOp::Silu,
        PrimalOp::Exp,
        PrimalOp::Log,
        PrimalOp::Sqrt,
        PrimalOp::Abs,
        PrimalOp::Neg,
        PrimalOp::Clamp { min: 0.0, max: 1.0 },
        PrimalOp::Add,
        PrimalOp::Sub,
        PrimalOp::Mul,
        PrimalOp::Div,
        PrimalOp::Matmul,
        PrimalOp::Transpose { dim0: 0, dim1: 1 },
        PrimalOp::Sum { dim: None },
        PrimalOp::Mean { dim: None },
        PrimalOp::Softmax { dim: -1 },
        PrimalOp::LogSoftmax { dim: -1 },
        PrimalOp::Reshape { target_ndim: 1 },
        PrimalOp::Broadcast,
        PrimalOp::Concat { dim: 0 },
        PrimalOp::Split { dim: 0, chunks: 2 },
        PrimalOp::Slice {
            dim: 0,
            start: 0,
            end: 1,
            orig_dim_size: 2,
        },
        PrimalOp::PadZero {
            dim: 0,
            pad_before: 0,
            pad_after: 1,
        },
        PrimalOp::Gather { dim: 0 },
        PrimalOp::ScatterAdd { dim: 0 },
        PrimalOp::Embedding,
        PrimalOp::LayerNorm { eps: NormEps::Const(1e-5) },
        PrimalOp::RMSNorm { eps: NormEps::Const(1e-5) },
        PrimalOp::BatchNorm {
            eps: 1e-5,
            training: true,
        },
        PrimalOp::MaxPool2d {
            kernel: 2,
            stride: 2,
        },
        PrimalOp::AvgPool2d {
            kernel: 2,
            stride: 2,
        },
        PrimalOp::Conv2d {
            stride: 1,
            padding: 0,
        },
        PrimalOp::ConvTranspose2d {
            stride: 1,
            padding: 0,
        },
        PrimalOp::Conv2dBackward {
            kind: ConvGradKind::Input,
            stride: 1,
            padding: 0,
        },
        PrimalOp::MaterializeConvOutputGrad {
            stride: 1,
            padding: 0,
        },
        PrimalOp::Repeat { kernel: 2 },
        PrimalOp::CrossEntropyLoss,
        PrimalOp::MSELoss,
        PrimalOp::L1Loss,
        PrimalOp::ScaledDotProductAttention { causal: false },
        PrimalOp::FlashAttentionBackwardExtract {
            causal: false,
            component: 0,
        },
        PrimalOp::ScaledDotProductAttentionPacked,
        PrimalOp::FlashAttentionBackwardExtractPacked { component: 0 },
        PrimalOp::CshaFusedBackwardExtract { component: 0 },
        PrimalOp::FusedCshaBackward {
            layer: String::new(),
        },
        PrimalOp::PrologueRecompute {
            subgraph_id: SubgraphId(0),
        },
        PrimalOp::FreeTensor,
        PrimalOp::RoPE { dim: 2 },
        PrimalOp::RoPEInverse { dim: 2 },
        PrimalOp::FusedGatedLoraMatmul {
            scale: 1.0,
            kernel_handle: 0,
        },
        PrimalOp::FusedLoraMatmul {
            scale: 1.0,
            kernel_handle: 0,
        },
        PrimalOp::FusedIa3Matmul { kernel_handle: 0 },
        PrimalOp::FusedLinearCe {
            vocab_size: 0,
            hidden_size: 0,
            batch_size: 0,
            seq_len: 0,
            vocab_tile: 0,
            ignore_index: -100,
            is_large: false,
            has_bias: false,
            x_rank3: false,
        },
        PrimalOp::FusedLinearCeBackwardExtract {
            component: 0,
            vocab_size: 0,
            hidden_size: 0,
            batch_size: 0,
            seq_len: 0,
            vocab_tile: 0,
            ignore_index: -100,
            has_bias: false,
            x_rank3: false,
        },
        PrimalOp::FusedKlCe {
            vocab_size: 0,
            student_hidden: 0,
            teacher_hidden: 0,
            batch_size: 0,
            seq_len: 0,
            vocab_tile: 0,
            ignore_index: -100,
            alpha_bits: 0,
            temperature_bits: 0,
        },
        PrimalOp::FusedKlCeBackwardExtract {
            component: 0,
            vocab_size: 0,
            student_hidden: 0,
            teacher_hidden: 0,
            batch_size: 0,
            seq_len: 0,
            vocab_tile: 0,
            ignore_index: -100,
            alpha_bits: 0,
            temperature_bits: 0,
        },
        PrimalOp::Dropout { p: 0.1 },
        PrimalOp::DropoutMask { p: 0.1 },
        PrimalOp::Select,
        PrimalOp::Condition(CompareKind::Gt),
        PrimalOp::Input(String::new()),
        PrimalOp::Param(String::new()),
        PrimalOp::Constant(0.0),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::WengertOp;

    fn make_op(result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp {
            id: 0,
            result,
            op,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    /// Every `PrimalOp` variant in `wengert.rs` has an entry in
    /// [`ad_cert_inventory`] (and nothing else does), so the coverage gate in
    /// `source_ad_rule_cert.rs` sees every certificate name.
    #[test]
    fn ad_cert_inventory_covers_every_variant() {
        let src = include_str!("wengert.rs");
        let body = &src[src.find("pub enum PrimalOp {").expect("PrimalOp")..];
        let body = &body[..body.find("\n}\n").expect("end of PrimalOp")];
        let mut declared: Vec<String> = body
            .lines()
            .skip(1)
            .filter_map(|l| {
                let rest = l.strip_prefix("    ")?;
                if !rest.starts_with(|c: char| c.is_ascii_uppercase()) {
                    return None;
                }
                let name: String = rest.chars().take_while(|c| c.is_alphanumeric()).collect();
                Some(name)
            })
            .collect();
        declared.sort();
        let mut listed: Vec<String> = ad_cert_inventory()
            .into_iter()
            .map(|(n, _)| n.split('(').next().unwrap_or("").to_string())
            .collect();
        listed.sort();
        listed.dedup();
        assert!(
            declared.len() > 60,
            "the scan found only {} variants",
            declared.len()
        );
        assert_eq!(
            declared, listed,
            "ad_cert_inventory is out of step with PrimalOp"
        );
    }

    /// The passthrough names `apply_ad_rule` matches are exactly
    /// [`RULED_PASSTHROUGHS`]: a new passthrough rule must be listed (and so
    /// given a status) before it can fire.
    #[test]
    fn ruled_passthroughs_match_the_rule_arm() {
        let src = include_str!("ad_rules.rs");
        let arm = &src[src
            .find("PrimalOp::Passthrough(name) => {\n            // Only listed")
            .expect("arm")..];
        let arm = &arm[..arm.find("_ => vec![],").expect("end of arm")];
        let mut matched: Vec<&str> = arm
            .lines()
            .map(str::trim)
            .filter(|l| l.starts_with('"') && l.ends_with("=> {"))
            .flat_map(|l| l.trim_end_matches("=> {").split('|'))
            .map(|n| n.trim().trim_matches('"'))
            .collect();
        matched.sort();
        let mut listed = RULED_PASSTHROUGHS.to_vec();
        listed.sort();
        assert_eq!(matched, listed);
        for name in RULED_PASSTHROUGHS {
            let op = make_op(1, PrimalOp::Passthrough((*name).to_string()), vec![0]);
            assert!(
                !apply_ad_rule(&op, 9).is_empty(),
                "{name} is listed but has no rule"
            );
        }
        let unlisted = make_op(1, PrimalOp::Passthrough("floor".into()), vec![0]);
        assert!(apply_ad_rule(&unlisted, 9).is_empty());
    }

    /// A status agrees with whether the op has a rule (in `apply_ad_rule`, or
    /// one of the fused-adapter rules `source_ad.rs` applies before it):
    /// certified and uncertified ops have one, non-differentiable ops do not,
    /// and the unreachable or adjoint-only ops that carry a rule anyway are
    /// pinned, so a rule added to any op forces its status to be revisited.
    #[test]
    fn ad_cert_status_agrees_with_the_rules() {
        const RULED_BUT_UNCERTIFIABLE: &[&str] = &[
            "Reshape",
            "Split",
            "Slice",
            "ScatterAdd",
            "BatchNorm",
            "MaxPool2d",
            "AvgPool2d",
            "RoPE",
        ];
        let source_ad = include_str!("source_ad.rs");
        let inputs: Vec<VarId> = (0..8).collect();
        let mut ruled_uncertifiable = Vec::new();
        for op in ad_cert_samples() {
            let (name, status) = (variant_name(&op), ad_cert_status(&op));
            let inline_rule = source_ad.contains(&format!("if let PrimalOp::{name} {{"));
            let has_rule =
                inline_rule || !apply_ad_rule(&make_op(100, op, inputs.clone()), 101).is_empty();
            match status {
                AdCertStatus::Certified(_) | AdCertStatus::Uncertified(_) => {
                    assert!(
                        has_rule,
                        "{name} is {status:?} but apply_ad_rule has no rule"
                    );
                }
                AdCertStatus::NotDifferentiable(_) => {
                    assert!(!has_rule, "{name} is NotDifferentiable but has a rule");
                }
                AdCertStatus::Unreachable(_) | AdCertStatus::AdjointOnly(_) => {
                    if has_rule {
                        ruled_uncertifiable.push(name);
                    }
                }
            }
        }
        assert_eq!(
            ruled_uncertifiable, RULED_BUT_UNCERTIFIABLE,
            "the unreachable/adjoint-only ops with a rule changed; revisit their statuses"
        );
    }

    #[test]
    fn test_add_rule() {
        let op = make_op(2, PrimalOp::Add, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 2);
        assert!(matches!(adj[0].expr, AdjointExpr::ReduceToShape(100, 0)));
        assert!(matches!(adj[1].expr, AdjointExpr::ReduceToShape(100, 1)));
    }

    #[test]
    fn test_sub_rule() {
        let op = make_op(2, PrimalOp::Sub, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::ReduceToShape(100, 0)));
        assert!(matches!(adj[1].expr, AdjointExpr::NegReduceToShape(100, 1)));
    }

    #[test]
    fn test_mul_rule() {
        let op = make_op(2, PrimalOp::Mul, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(
            adj[0].expr,
            AdjointExpr::MulElementwise(100, 1, 0)
        ));
        assert!(matches!(
            adj[1].expr,
            AdjointExpr::MulElementwise(100, 0, 1)
        ));
    }

    #[test]
    fn test_matmul_rule() {
        let op = make_op(2, PrimalOp::Matmul, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(
            adj[0].expr,
            AdjointExpr::MatmulTransposeLeft(100, 1, 0)
        ));
        assert!(matches!(
            adj[1].expr,
            AdjointExpr::MatmulTransposeRight(0, 100, 1)
        ));
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
        assert!(matches!(
            adj[0].expr,
            AdjointExpr::DivNumeratorBackward(100, 1, 0)
        ));
        assert!(matches!(
            adj[1].expr,
            AdjointExpr::DivDenominatorBackward(100, 0, 1)
        ));
    }

    #[test]
    fn test_sum_over_a_dim_reexpands_along_it() {
        let op = make_op(1, PrimalOp::Sum { dim: Some(-1) }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::SumDimBackward(100, 0, -1)));
    }

    /// A full sum's gradient is expanded to its operand's shape; a bare
    /// Broadcast left it one element, which a matmul backward rejects.
    #[test]
    fn test_full_sum_expands_to_its_operand() {
        let op = make_op(1, PrimalOp::Sum { dim: None }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::ExpandLike(100, 0)));
    }

    #[test]
    fn test_mean_scales_then_broadcasts() {
        // Mean backward: grad scaled by numel(out)/numel(in) at run time, then
        // broadcast. A bare Broadcast is the N-times-too-large gradient this
        // rule used to produce.
        let op = make_op(1, PrimalOp::Mean { dim: None }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::MeanBackward(100, 0, 1)));
        let op = make_op(1, PrimalOp::Mean { dim: Some(0) }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::MeanDimBackward(100, 0, 1, 0)));
    }

    #[test]
    fn test_transpose_rule() {
        let op = make_op(1, PrimalOp::Transpose { dim0: 0, dim1: 1 }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert!(matches!(adj[0].expr, AdjointExpr::Transpose(100, 0, 1)));
    }

    #[test]
    fn test_saved_nothing() {
        assert_eq!(
            saved_for_backward(&PrimalOp::Neg),
            SavedRequirement::Nothing
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::Transpose { dim0: 0, dim1: 1 }),
            SavedRequirement::Nothing
        );
    }

    #[test]
    fn test_saved_inputs() {
        assert_eq!(saved_for_backward(&PrimalOp::Mul), SavedRequirement::Inputs);
        // Add/Sub read their operands' shapes to undo a broadcast.
        assert_eq!(saved_for_backward(&PrimalOp::Add), SavedRequirement::Inputs);
        assert_eq!(saved_for_backward(&PrimalOp::Sub), SavedRequirement::Inputs);
        assert_eq!(
            saved_for_backward(&PrimalOp::Matmul),
            SavedRequirement::Inputs
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::Relu),
            SavedRequirement::Inputs
        );
    }

    #[test]
    fn test_saved_output() {
        assert_eq!(
            saved_for_backward(&PrimalOp::Sigmoid),
            SavedRequirement::Output
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::Tanh),
            SavedRequirement::Output
        );
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
        assert_eq!(
            adj.len(),
            2,
            "Select should produce exactly two input adjoints"
        );
        // First: adjoint for the true branch value
        assert_eq!(
            adj[0].input_var, 1,
            "First adjoint targets true_val (inputs[1])"
        );
        assert!(
            matches!(adj[0].expr, AdjointExpr::SelectTrue(100, 0)),
            "Expected SelectTrue(output_bar=100, cond_var=0), got {:?}",
            adj[0].expr
        );
        // Second: adjoint for the false branch value
        assert_eq!(
            adj[1].input_var, 2,
            "Second adjoint targets false_val (inputs[2])"
        );
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
        assert!(
            adj.is_empty(),
            "Condition op should have no adjoints (non-differentiable)"
        );
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
        assert_eq!(
            saved_for_backward(&PrimalOp::Gelu),
            SavedRequirement::Inputs
        );
    }

    #[test]
    fn test_silu_backward() {
        let op = make_op(1, PrimalOp::Silu, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::SiluBackward(100, 0)));
        assert_eq!(
            saved_for_backward(&PrimalOp::Silu),
            SavedRequirement::Inputs
        );
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
        assert!(matches!(adj[0].expr, AdjointExpr::SoftmaxBackward(100, 1, -1)));
        let op = make_op(1, PrimalOp::Softmax { dim: 0 }, vec![0]);
        assert!(matches!(
            apply_ad_rule(&op, 100)[0].expr,
            AdjointExpr::SoftmaxBackward(100, 1, 0)
        ));
        assert_eq!(
            saved_for_backward(&PrimalOp::Softmax { dim: -1 }),
            SavedRequirement::Output
        );
    }

    #[test]
    fn test_clamp_backward() {
        let op = make_op(1, PrimalOp::Clamp { min: 0.0, max: 1.0 }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(
            adj[0].expr,
            AdjointExpr::ClampBackward(100, 0, _, _)
        ));
        assert_eq!(
            saved_for_backward(&PrimalOp::Clamp { min: 0.0, max: 1.0 }),
            SavedRequirement::Inputs
        );
    }

    // --- Tier 2: Normalization / indexing / shape ---

    #[test]
    fn test_layer_norm_backward() {
        let op = make_op(3, PrimalOp::LayerNorm { eps: NormEps::Const(1e-5) }, vec![0, 1, 2]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(
            adj.len(),
            3,
            "LayerNorm should produce 3 adjoints (input, gamma, beta)"
        );
        assert!(matches!(
            adj[0].expr,
            AdjointExpr::LayerNormBackward(100, 0, Some(1), _)
        ));
        assert!(matches!(
            adj[1].expr,
            AdjointExpr::NormGammaBackward(100, 0, _, -1, 1)
        )); // gamma grad
        assert!(matches!(adj[2].expr, AdjointExpr::ReduceToShape(100, 2))); // beta grad
        assert_eq!(
            saved_for_backward(&PrimalOp::LayerNorm { eps: NormEps::Const(1e-5) }),
            SavedRequirement::Inputs
        );
    }

    #[test]
    fn test_dropout_backward() {
        // Two-op split layout: inputs = [x, mask] where mask is the paired
        // DropoutMask op's result — the EXACT forward RNG mask, an SSA var.
        // (Before 2026-08-16 extraction put the raw call args here, so
        // "inputs[1]" was the p ARGUMENT var and this test pinned a wrong
        // wiring as correct.)
        let op = make_op(2, PrimalOp::Dropout { p: 0.1 }, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        match &adj[0].expr {
            AdjointExpr::DropoutBackward(y_bar, mask, scale) => {
                assert_eq!(*y_bar, 100);
                assert_eq!(*mask, 1); // inputs[1] = the DropoutMask result
                assert!((scale - 1.0 / 0.9).abs() < 1e-6); // 1/(1-0.1)
            }
            other => panic!("Expected DropoutBackward, got {:?}", other),
        }
        assert_eq!(
            saved_for_backward(&PrimalOp::Dropout { p: 0.1 }),
            SavedRequirement::Inputs
        );
    }

    #[test]
    fn test_dropout_mask_has_no_adjoint_and_saves_output() {
        // The mask is an RNG draw: no gradient flows through it to x, and
        // its OUTPUT (the mask tensor) must survive to the backward.
        let op = make_op(1, PrimalOp::DropoutMask { p: 0.5 }, vec![0]);
        assert!(apply_ad_rule(&op, 100).is_empty());
        assert_eq!(
            saved_for_backward(&PrimalOp::DropoutMask { p: 0.5 }),
            SavedRequirement::Output
        );
    }

    #[test]
    #[should_panic(expected = "requires inputs [x, mask]")]
    fn test_dropout_without_mask_input_panics() {
        // A 1-input Dropout op is malformed post-split: the old rule fell
        // back to multiplying the gradient by x ITSELF — silently wrong.
        let op = make_op(1, PrimalOp::Dropout { p: 0.1 }, vec![0]);
        let _ = apply_ad_rule(&op, 100);
    }

    #[test]
    fn test_embedding_backward() {
        let op = make_op(2, PrimalOp::Embedding, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(
            adj[0].expr,
            AdjointExpr::EmbeddingBackward(100, 1, 0)
        ));
    }

    #[test]
    fn test_gather_backward() {
        let op = make_op(2, PrimalOp::Gather { dim: 1 }, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(
            adj[0].expr,
            AdjointExpr::GatherBackward(100, 0, 1, 1)
        ));
    }

    #[test]
    fn test_concat_backward() {
        let op = make_op(3, PrimalOp::Concat { dim: 0 }, vec![0, 1, 2]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 3, "Concat of 3 inputs should produce 3 adjoints");
        for (i, a) in adj.iter().enumerate() {
            assert_eq!(a.input_var, i as VarId);
            assert_eq!(a.expr, AdjointExpr::ConcatSplit(100, 0, i, vec![0, 1, 2]));
        }
    }

    #[test]
    fn test_slice_backward() {
        let op = make_op(
            1,
            PrimalOp::Slice {
                dim: 0,
                start: 2,
                end: 5,
                orig_dim_size: 10,
            },
            vec![0],
        );
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(
            adj[0].expr,
            AdjointExpr::SliceBackward(100, 0, 2, 5, 10)
        ));
    }

    // --- Tier 3: Compound ops ---

    #[test]
    fn test_conv2d_backward() {
        // No bias: inputs [input=0, weight=1] -> 2 adjoints (input, weight).
        // Each adjoint carries (kind, grad_output=100, input=0, weight=1, stride, padding).
        let op = make_op(
            2,
            PrimalOp::Conv2d {
                stride: 1,
                padding: 0,
            },
            vec![0, 1],
        );
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(
            adj.len(),
            2,
            "Conv2d without bias should produce 2 adjoints (input, weight)"
        );
        assert!(matches!(
            adj[0].expr,
            AdjointExpr::Conv2dBackward(ConvGradKind::Input, 100, 0, 1, 1, 0)
        ));
        assert_eq!(adj[0].input_var, 0);
        assert!(matches!(
            adj[1].expr,
            AdjointExpr::Conv2dBackward(ConvGradKind::Weight, 100, 0, 1, 1, 0)
        ));
        assert_eq!(adj[1].input_var, 1);

        // With bias: inputs [input=0, weight=1, bias=2] -> 3 adjoints, and the
        // stride/padding are threaded through to every gradient.
        let op_b = make_op(
            3,
            PrimalOp::Conv2d {
                stride: 2,
                padding: 1,
            },
            vec![0, 1, 2],
        );
        let adj_b = apply_ad_rule(&op_b, 100);
        assert_eq!(
            adj_b.len(),
            3,
            "Conv2d with bias should produce 3 adjoints (input, weight, bias)"
        );
        assert!(matches!(
            adj_b[2].expr,
            AdjointExpr::Conv2dBackward(ConvGradKind::Bias, 100, 0, 1, 2, 1)
        ));
        assert_eq!(adj_b[2].input_var, 2);
    }

    #[test]
    fn test_maxpool_backward() {
        let op = make_op(
            2,
            PrimalOp::MaxPool2d {
                kernel: 2,
                stride: 2,
            },
            vec![0, 1],
        );
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::MaxPoolBackward(100, 1)));
        assert_eq!(
            saved_for_backward(&PrimalOp::MaxPool2d {
                kernel: 2,
                stride: 2
            }),
            SavedRequirement::Output
        );
    }

    #[test]
    fn test_avgpool_backward() {
        let op = make_op(
            1,
            PrimalOp::AvgPool2d {
                kernel: 3,
                stride: 3,
            },
            vec![0],
        );
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(adj[0].expr, AdjointExpr::AvgPoolBackward(100, 9))); // kernel*kernel = 9
        assert_eq!(
            saved_for_backward(&PrimalOp::AvgPool2d {
                kernel: 3,
                stride: 3
            }),
            SavedRequirement::Nothing
        );
    }

    #[test]
    fn test_cross_entropy_backward() {
        let op = make_op(2, PrimalOp::CrossEntropyLoss, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(
            adj[0].expr,
            AdjointExpr::CrossEntropyBackward(100, 0, 1)
        ));
        assert_eq!(
            saved_for_backward(&PrimalOp::CrossEntropyLoss),
            SavedRequirement::Inputs
        );
    }

    #[test]
    fn test_mse_loss_backward() {
        let op = make_op(2, PrimalOp::MSELoss, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 2);
        assert!(matches!(adj[0].expr, AdjointExpr::MSEBackward(100, 0, 1)));
        assert_eq!(adj[1].input_var, 1);
        assert!(matches!(adj[1].expr, AdjointExpr::MSETargetBackward(100, 0, 1)));
    }

    #[test]
    fn test_l1_loss_backward() {
        let op = make_op(2, PrimalOp::L1Loss, vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 2);
        assert!(matches!(adj[0].expr, AdjointExpr::L1Backward(100, 0, 1)));
        assert_eq!(adj[1].input_var, 1);
        assert!(matches!(adj[1].expr, AdjointExpr::L1TargetBackward(100, 0, 1)));
    }

    // --- Tier 4: Attention ---

    #[test]
    fn test_attention_backward() {
        let op = make_op(
            3,
            PrimalOp::ScaledDotProductAttention { causal: true },
            vec![0, 1, 2],
        );
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(
            adj.len(),
            3,
            "Attention should produce 3 adjoints (Q, K, V)"
        );
        assert!(matches!(
            adj[0].expr,
            AdjointExpr::AttentionBackwardQ(100, 0, 1, 2, 3, true, None)
        ));
        assert!(matches!(
            adj[1].expr,
            AdjointExpr::AttentionBackwardK(100, 0, 1, 2, 3, true, None)
        ));
        assert!(matches!(
            adj[2].expr,
            AdjointExpr::AttentionBackwardV(100, 0, 1, 2, 3, true, None)
        ));
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
        assert_eq!(
            saved_for_backward(&PrimalOp::RoPE { dim: 64 }),
            SavedRequirement::Nothing
        );
    }

    #[test]
    fn test_log_softmax_backward() {
        let op = make_op(1, PrimalOp::LogSoftmax { dim: -1 }, vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert!(matches!(
            adj[0].expr,
            AdjointExpr::LogSoftmaxBackward(100, 1, -1)
        ));
        assert_eq!(
            saved_for_backward(&PrimalOp::LogSoftmax { dim: -1 }),
            SavedRequirement::Output
        );
    }

    // --- Saved requirement tests for new ops ---

    #[test]
    fn test_saved_new_nothing_ops() {
        assert_eq!(
            saved_for_backward(&PrimalOp::Concat { dim: 0 }),
            SavedRequirement::Nothing
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::Split { dim: 0, chunks: 2 }),
            SavedRequirement::Nothing
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::Slice {
                dim: 0,
                start: 0,
                end: 5,
                orig_dim_size: 10
            }),
            SavedRequirement::Nothing
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::AvgPool2d {
                kernel: 2,
                stride: 2
            }),
            SavedRequirement::Nothing
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::RoPE { dim: 64 }),
            SavedRequirement::Nothing
        );
    }

    #[test]
    fn test_saved_new_input_ops() {
        assert_eq!(
            saved_for_backward(&PrimalOp::Gelu),
            SavedRequirement::Inputs
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::Silu),
            SavedRequirement::Inputs
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::Clamp { min: 0.0, max: 1.0 }),
            SavedRequirement::Inputs
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::LayerNorm { eps: NormEps::Const(1e-5) }),
            SavedRequirement::Inputs
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::BatchNorm {
                eps: 1e-5,
                training: true
            }),
            SavedRequirement::Inputs
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::Dropout { p: 0.1 }),
            SavedRequirement::Inputs
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::Embedding),
            SavedRequirement::Inputs
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::Gather { dim: 0 }),
            SavedRequirement::Inputs
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::Conv2d {
                stride: 1,
                padding: 0
            }),
            SavedRequirement::Inputs
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::CrossEntropyLoss),
            SavedRequirement::Inputs
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::MSELoss),
            SavedRequirement::Inputs
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::ScaledDotProductAttention { causal: false }),
            SavedRequirement::Inputs
        );
    }

    #[test]
    fn test_saved_new_output_ops() {
        assert_eq!(
            saved_for_backward(&PrimalOp::Softmax { dim: -1 }),
            SavedRequirement::Output
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::LogSoftmax { dim: -1 }),
            SavedRequirement::Output
        );
        assert_eq!(
            saved_for_backward(&PrimalOp::MaxPool2d {
                kernel: 2,
                stride: 2
            }),
            SavedRequirement::Output
        );
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
    fn test_reshape_backward_restores_shape() {
        let op = make_op(1, PrimalOp::Passthrough("reshape".into()), vec![0, 1]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert_eq!(adj[0].input_var, 0);
        assert!(matches!(adj[0].expr, AdjointExpr::ReshapeLike(100, 0)));
    }

    // ── T5.2 CSHA fused-backward dispatcher tests ─────────────────────────

    fn mark_with_cfg(cfg: Option<crate::flash_attention::FlashAttentionConfig>) -> FusionMark {
        FusionMark {
            layer: "blocks.0".into(),
            kind: Some(crate::csha_boundary::ProjKind::Q),
            param_name: "blocks.0.attn.wq".into(),
            role: crate::csha_apply::MarkRole::NormPrologue,
            config: cfg,
            backward_emitted: std::cell::Cell::new(false),
            chain_varids: None,
        }
    }

    fn base_cfg_fused_backward(
        block_q: i64, block_kv: i64, head_dim: i64, heads: u32, d_model: u32,
    ) -> crate::flash_attention::FlashAttentionConfig {
        let _ = heads;
        crate::flash_attention::FlashAttentionConfig {
            block_q, block_kv, head_dim,
            causal: false, paged: false, rope_q: false,
            rope_style: crate::flash_attention::RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0, gpu_sm: 75,
            segment_masked: false,
            csha: Some(crate::flash_attention::CshaExtras {
                fused_projections: true,
                save_activations_for_backward: true,
                d_model,
                ..crate::flash_attention::CshaExtras::default()
            }),
            checkpoint: None,
        }
    }

    #[test]
    fn ad_dispatcher_emits_fused_backward_on_claimed_chain_output_op() {
        let mark = mark_with_cfg(Some(base_cfg_fused_backward(32, 32, 32, 4, 32)));
        // First encounter (topological output op): EmitFused + flag flips.
        match csha_dispatch_for_op(&mark, /*op_idx=*/ 99) {
            CshaDispatchDecision::EmitFused => {}
            other => panic!("expected EmitFused, got {other:?}"),
        }
        assert!(
            mark.backward_emitted.get(),
            "backward_emitted flag must be set after EmitFused"
        );
        // Second encounter (same chain, different claimed op): no-op.
        match csha_dispatch_for_op(&mark, /*op_idx=*/ 98) {
            CshaDispatchDecision::AlreadyEmitted => {}
            other => panic!("expected AlreadyEmitted, got {other:?}"),
        }
    }

    #[test]
    fn ad_dispatcher_falls_back_on_validator_reject_with_diagnostic() {
        // (64,64,64,8,64) exceeds the backward SMEM budget (T2.1 test).
        let mark = mark_with_cfg(Some(base_cfg_fused_backward(64, 64, 64, 8, 64)));
        match csha_dispatch_for_op(&mark, 99) {
            CshaDispatchDecision::Fallback { diagnostic } => {
                assert!(
                    diagnostic.contains("CSHA fused backward rejected"),
                    "diagnostic missing rejection prefix: {diagnostic}"
                );
                assert!(
                    diagnostic.contains("Backward"),
                    "diagnostic must name direction: {diagnostic}"
                );
                assert!(
                    diagnostic.contains("blocks.0"),
                    "diagnostic must name the layer: {diagnostic}"
                );
                assert!(
                    diagnostic.contains("bytes >"),
                    "diagnostic must surface T2.1 byte-comparison: {diagnostic}"
                );
            }
            other => panic!("expected Fallback, got {other:?}"),
        }
        // Validator rejected — backward_emitted MUST NOT be flipped so
        // the fallback per-op dispatch path still fires on all
        // constituent ops.
        assert!(
            !mark.backward_emitted.get(),
            "validator-reject must leave backward_emitted=false"
        );
    }

    #[test]
    fn ad_dispatcher_falls_back_when_mark_has_no_config() {
        // Pre-Tier-C plans create FusionMark with config=None. Dispatcher
        // must recognise this and fall back cleanly instead of panicking.
        let mark = mark_with_cfg(None);
        match csha_dispatch_for_op(&mark, 99) {
            CshaDispatchDecision::Fallback { diagnostic } => {
                assert!(
                    diagnostic.contains("no FlashAttentionConfig"),
                    "diagnostic must explain missing-config: {diagnostic}"
                );
            }
            other => panic!("expected Fallback, got {other:?}"),
        }
    }

    #[test]
    fn ad_dispatcher_is_idempotent_under_repeated_calls() {
        let mark = mark_with_cfg(Some(base_cfg_fused_backward(32, 32, 32, 4, 32)));
        let first = csha_dispatch_for_op(&mark, 99);
        assert!(matches!(first, CshaDispatchDecision::EmitFused));
        for _ in 0..10 {
            let d = csha_dispatch_for_op(&mark, 99);
            assert!(
                matches!(d, CshaDispatchDecision::AlreadyEmitted),
                "repeated dispatch must be idempotent, got {d:?}"
            );
        }
    }

    #[test]
    fn test_rotate_half_backward_is_negated_rotate_half() {
        // rotate_half is its own inverse up to sign — the backward must NOT
        // be Identity. It must be RotateHalfBackward (which lowers to
        // -rotate_half(grad)). Identity here was the source of the RoPE
        // gradient corruption that prevented training convergence.
        let op = make_op(1, PrimalOp::Passthrough("rotate_half".into()), vec![0]);
        let adj = apply_ad_rule(&op, 100);
        assert_eq!(adj.len(), 1);
        assert_eq!(adj[0].input_var, 0);
        assert!(matches!(adj[0].expr, AdjointExpr::RotateHalfBackward(100)));
    }
}
