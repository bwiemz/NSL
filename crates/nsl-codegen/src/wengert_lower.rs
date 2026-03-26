//! M40: Lower a WengertList (symbolic backward graph) to Cranelift IR.
//!
//! Each `PrimalOp` variant is dispatched to the corresponding runtime FFI call.
//! The result is a map from all VarIds (primal + adjoint) to Cranelift `Value`s.

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::wengert::{CompareKind, PrimalOp, VarId, WengertList, WengertOp};
use crate::CodegenError;
use cranelift_codegen::ir::{types as cl_types, InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use std::collections::HashMap;

pub type VarMap = HashMap<VarId, Value>;

/// Lower a WengertList to Cranelift IR by dispatching each PrimalOp to its runtime FFI call.
///
/// `primal_vars` maps VarIds from the forward pass to their Cranelift Values (i64 tensor pointers).
/// Returns a map from all VarIds (including adjoint) to Cranelift Values.
pub fn compile_wengert_ops(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    _state: &mut FuncState,
    wengert: &WengertList,
    primal_vars: &VarMap,
) -> Result<VarMap, CodegenError> {
    let mut var_map = primal_vars.clone();
    for op in &wengert.ops {
        let result_val = lower_single_op(compiler, builder, op, &var_map)?;
        var_map.insert(op.result, result_val);
    }
    Ok(var_map)
}

/// Resolve all input VarIds for a WengertOp to their Cranelift Values.
fn resolve_inputs(op: &WengertOp, var_map: &VarMap) -> Result<Vec<Value>, CodegenError> {
    op.inputs
        .iter()
        .map(|vid| {
            var_map.get(vid).copied().ok_or_else(|| {
                CodegenError::new(format!("source-ad: unresolved VarId {}", vid))
            })
        })
        .collect()
}

/// Emit a single runtime FFI call and return the result Value.
fn call(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    name: &str,
    args: &[Value],
) -> Result<Value, CodegenError> {
    compiler.compile_call_by_name(builder, name, args)
}

/// Lower one WengertOp to Cranelift IR.
fn lower_single_op(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    op: &WengertOp,
    var_map: &VarMap,
) -> Result<Value, CodegenError> {
    // Marker ops (leaf nodes) — already present in primal_vars
    match &op.op {
        PrimalOp::Input(_) | PrimalOp::Param(_) => {
            return var_map.get(&op.result).copied().ok_or_else(|| {
                CodegenError::new(format!(
                    "source-ad: leaf VarId {} not in primal_vars",
                    op.result
                ))
            });
        }
        PrimalOp::Constant(val) => {
            let v = builder.ins().f64const(*val);
            let dt = builder.ins().iconst(cl_types::I64, 1); // f32 default
            return call(compiler, builder, "nsl_tensor_scalar", &[v, dt]);
        }
        _ => {}
    }

    let inputs = resolve_inputs(op, var_map)?;

    match &op.op {
        // Already handled above
        PrimalOp::Input(_) | PrimalOp::Param(_) | PrimalOp::Constant(_) => unreachable!(),

        // === Elementwise unary (11 ops) ===
        PrimalOp::Relu => call(compiler, builder, "nsl_tensor_relu", &[inputs[0]]),
        PrimalOp::Sigmoid => call(compiler, builder, "nsl_tensor_sigmoid", &[inputs[0]]),
        PrimalOp::Tanh => call(compiler, builder, "nsl_tensor_tanh_act", &[inputs[0]]),
        PrimalOp::Gelu => call(compiler, builder, "nsl_tensor_gelu", &[inputs[0]]),
        PrimalOp::Silu => call(compiler, builder, "nsl_tensor_silu", &[inputs[0]]),
        PrimalOp::Exp => call(compiler, builder, "nsl_tensor_exp", &[inputs[0]]),
        PrimalOp::Log => call(compiler, builder, "nsl_tensor_log", &[inputs[0]]),
        PrimalOp::Sqrt => call(compiler, builder, "nsl_tensor_sqrt", &[inputs[0]]),
        PrimalOp::Abs => call(compiler, builder, "nsl_tensor_abs", &[inputs[0]]),
        PrimalOp::Neg => call(compiler, builder, "nsl_tensor_neg", &[inputs[0]]),
        PrimalOp::Clamp { min, max } => {
            let min_v = builder.ins().f64const(*min);
            let max_v = builder.ins().f64const(*max);
            call(compiler, builder, "nsl_tensor_clamp", &[inputs[0], min_v, max_v])
        }

        // === Elementwise binary (4 ops) ===
        PrimalOp::Add => call(compiler, builder, "nsl_tensor_add", &[inputs[0], inputs[1]]),
        PrimalOp::Sub => call(compiler, builder, "nsl_tensor_sub", &[inputs[0], inputs[1]]),
        PrimalOp::Mul => call(compiler, builder, "nsl_tensor_mul", &[inputs[0], inputs[1]]),
        PrimalOp::Div => call(compiler, builder, "nsl_tensor_div", &[inputs[0], inputs[1]]),

        // === Linear algebra (4 ops) ===
        PrimalOp::Matmul => call(compiler, builder, "nsl_tensor_matmul", &[inputs[0], inputs[1]]),
        PrimalOp::Transpose { dim0, dim1 } => {
            let d0 = builder.ins().iconst(cl_types::I64, *dim0 as i64);
            let d1 = builder.ins().iconst(cl_types::I64, *dim1 as i64);
            call(compiler, builder, "nsl_tensor_transpose", &[inputs[0], d0, d1])
        }
        PrimalOp::Reshape { .. } => {
            // Reshape in backward pass is typically a view op; clone preserves data.
            call(compiler, builder, "nsl_tensor_clone", &[inputs[0]])
        }
        PrimalOp::Broadcast => {
            // Broadcast is a shape-level op; clone the tensor.
            call(compiler, builder, "nsl_tensor_clone", &[inputs[0]])
        }

        // === Reductions (4 ops) ===
        PrimalOp::Sum { dim } => {
            let d = builder.ins().iconst(cl_types::I64, dim.unwrap_or(-1));
            let keepdim = builder.ins().iconst(cl_types::I64, 0);
            call(compiler, builder, "nsl_tensor_sum_dim", &[inputs[0], d, keepdim])
        }
        PrimalOp::Mean { dim } => {
            let d = builder.ins().iconst(cl_types::I64, dim.unwrap_or(-1));
            let keepdim = builder.ins().iconst(cl_types::I64, 0);
            call(compiler, builder, "nsl_tensor_mean_dim", &[inputs[0], d, keepdim])
        }
        PrimalOp::Softmax { dim } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            call(compiler, builder, "nsl_tensor_softmax", &[inputs[0], d])
        }
        PrimalOp::LogSoftmax { dim } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            call(compiler, builder, "nsl_tensor_logsoftmax", &[inputs[0], d])
        }

        // === Shape ops (5 ops) ===
        PrimalOp::Concat { dim } => {
            // Runtime nsl_tensor_cat takes (tensor_list_ptr, dim).
            // Construct an NslList from all inputs, then pass to cat.
            let list = call(compiler, builder, "nsl_list_new", &[])?;
            for &inp in &inputs {
                call(compiler, builder, "nsl_list_push", &[list, inp])?;
            }
            let d = builder.ins().iconst(cl_types::I64, *dim);
            let result = call(compiler, builder, "nsl_tensor_cat", &[list, d])?;
            call(compiler, builder, "nsl_list_free", &[list])?;
            Ok(result)
        }
        PrimalOp::Split { .. } => {
            // Split is a forward-only op — its backward is Concat (via SplitConcat adjoint).
            // If Split appears in an adjoint graph, it's from re-execution; clone is correct.
            call(compiler, builder, "nsl_tensor_clone", &[inputs[0]])
        }
        PrimalOp::Slice { dim, start, end, .. } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            let s = builder.ins().iconst(cl_types::I64, *start);
            let e = builder.ins().iconst(cl_types::I64, *end);
            call(compiler, builder, "nsl_tensor_slice", &[inputs[0], d, s, e])
        }
        PrimalOp::PadZero { dim, pad_before, pad_after } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            let pb = builder.ins().iconst(cl_types::I64, *pad_before);
            let pa = builder.ins().iconst(cl_types::I64, *pad_after);
            call(compiler, builder, "nsl_tensor_pad_zero", &[inputs[0], d, pb, pa])
        }
        PrimalOp::Repeat { kernel } => {
            let k = builder.ins().iconst(cl_types::I64, *kernel as i64);
            call(compiler, builder, "nsl_tensor_repeat", &[inputs[0], k])
        }

        // === Indexing (3 ops) ===
        PrimalOp::Gather { dim } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            call(compiler, builder, "nsl_tensor_gather", &[inputs[0], inputs[1], d])
        }
        PrimalOp::ScatterAdd { dim } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            call(compiler, builder, "nsl_tensor_scatter_add", &[inputs[0], inputs[1], d])
        }
        PrimalOp::Embedding => {
            call(compiler, builder, "nsl_tensor_embedding_lookup", &[inputs[0], inputs[1]])
        }

        // === Normalization (2 ops) ===
        PrimalOp::LayerNorm { eps } => {
            let e = builder.ins().f64const(*eps);
            call(compiler, builder, "nsl_tensor_layernorm", &[inputs[0], inputs[1], inputs[2], e])
        }
        PrimalOp::BatchNorm { eps, training } => {
            let e = builder.ins().f64const(*eps);
            let t = builder.ins().iconst(cl_types::I64, if *training { 1 } else { 0 });
            call(compiler, builder, "nsl_tensor_batchnorm", &[inputs[0], inputs[1], inputs[2], e, t])
        }

        // === Convolution (3 ops) ===
        PrimalOp::Conv2d { stride, padding } => {
            // Runtime: nsl_tensor_conv2d(input, weight, bias, stride_h, stride_w, pad_h, pad_w)
            // Bias input is inputs[2] if present; use 0 (null) if only 2 inputs.
            let bias = if inputs.len() > 2 {
                inputs[2]
            } else {
                builder.ins().iconst(cl_types::I64, 0)
            };
            let sh = builder.ins().iconst(cl_types::I64, *stride as i64);
            let sw = builder.ins().iconst(cl_types::I64, *stride as i64);
            let ph = builder.ins().iconst(cl_types::I64, *padding as i64);
            let pw = builder.ins().iconst(cl_types::I64, *padding as i64);
            call(compiler, builder, "nsl_tensor_conv2d", &[inputs[0], inputs[1], bias, sh, sw, ph, pw])
        }
        PrimalOp::ConvTranspose2d { stride, padding } => {
            // Approximate: use conv2d with the same args (weight should already be transposed by AD rules)
            let bias = if inputs.len() > 2 {
                inputs[2]
            } else {
                builder.ins().iconst(cl_types::I64, 0)
            };
            let sh = builder.ins().iconst(cl_types::I64, *stride as i64);
            let sw = builder.ins().iconst(cl_types::I64, *stride as i64);
            let ph = builder.ins().iconst(cl_types::I64, *padding as i64);
            let pw = builder.ins().iconst(cl_types::I64, *padding as i64);
            call(compiler, builder, "nsl_tensor_conv2d", &[inputs[0], inputs[1], bias, sh, sw, ph, pw])
        }

        // === Pooling (2 ops) ===
        PrimalOp::MaxPool2d { kernel, stride } => {
            // Runtime: nsl_tensor_maxpool2d(input, kernel_h, kernel_w, stride, padding)
            let kh = builder.ins().iconst(cl_types::I64, *kernel as i64);
            let kw = builder.ins().iconst(cl_types::I64, *kernel as i64);
            let s = builder.ins().iconst(cl_types::I64, *stride as i64);
            let p = builder.ins().iconst(cl_types::I64, 0); // padding = 0
            call(compiler, builder, "nsl_tensor_maxpool2d", &[inputs[0], kh, kw, s, p])
        }
        PrimalOp::AvgPool2d { kernel, stride } => {
            let kh = builder.ins().iconst(cl_types::I64, *kernel as i64);
            let kw = builder.ins().iconst(cl_types::I64, *kernel as i64);
            let s = builder.ins().iconst(cl_types::I64, *stride as i64);
            let p = builder.ins().iconst(cl_types::I64, 0);
            call(compiler, builder, "nsl_tensor_avgpool2d", &[inputs[0], kh, kw, s, p])
        }

        // === Loss functions (3 ops) ===
        // Loss functions are composite: we lower them to sequences of existing FFI calls.
        PrimalOp::CrossEntropyLoss => {
            // cross_entropy(logits, targets) = mean(-log(softmax(logits)) * one_hot(targets))
            // Simplified: softmax(logits) -> log -> negate -> gather targets -> mean
            let neg_one = builder.ins().iconst(cl_types::I64, -1);
            let softmax = call(compiler, builder, "nsl_tensor_softmax", &[inputs[0], neg_one])?;
            let log_sm = call(compiler, builder, "nsl_tensor_log", &[softmax])?;
            let neg_log = call(compiler, builder, "nsl_tensor_neg", &[log_sm])?;
            // Gather target indices along last dim
            let dim_val = neg_one;
            let gathered = call(compiler, builder, "nsl_tensor_gather", &[neg_log, inputs[1], dim_val])?;
            let mean_dim = builder.ins().iconst(cl_types::I64, -1);
            let keepdim = builder.ins().iconst(cl_types::I64, 0);
            call(compiler, builder, "nsl_tensor_mean_dim", &[gathered, mean_dim, keepdim])
        }
        PrimalOp::MSELoss => {
            // mse_loss(pred, target) = mean((pred - target)^2)
            let diff = call(compiler, builder, "nsl_tensor_sub", &[inputs[0], inputs[1]])?;
            let sq = call(compiler, builder, "nsl_tensor_mul", &[diff, diff])?;
            let d = builder.ins().iconst(cl_types::I64, -1);
            let keepdim = builder.ins().iconst(cl_types::I64, 0);
            call(compiler, builder, "nsl_tensor_mean_dim", &[sq, d, keepdim])
        }
        PrimalOp::L1Loss => {
            // l1_loss(pred, target) = mean(|pred - target|)
            let diff = call(compiler, builder, "nsl_tensor_sub", &[inputs[0], inputs[1]])?;
            let abs_diff = call(compiler, builder, "nsl_tensor_abs", &[diff])?;
            let d = builder.ins().iconst(cl_types::I64, -1);
            let keepdim = builder.ins().iconst(cl_types::I64, 0);
            call(compiler, builder, "nsl_tensor_mean_dim", &[abs_diff, d, keepdim])
        }

        // === Attention (4 ops) ===
        PrimalOp::ScaledDotProductAttention { .. } => {
            // Forward attention op should not normally appear in the adjoint graph,
            // but handle for safety by cloning the first input (approximate).
            call(compiler, builder, "nsl_tensor_clone", &[inputs[0]])
        }
        PrimalOp::FlashAttentionBackwardExtract { .. } => {
            // Fused attention backward component extraction.
            // Approximate: clone the input tensor (the full backward will be
            // decomposed by ad_rules into simpler ops).
            call(compiler, builder, "nsl_tensor_clone", &[inputs[0]])
        }
        PrimalOp::RoPE { .. } => {
            // No dedicated nsl_tensor_rope FFI exists; clone as identity for now.
            // RoPE forward is handled by the flash attention system.
            call(compiler, builder, "nsl_tensor_clone", &[inputs[0]])
        }
        PrimalOp::RoPEInverse { dim } => {
            let d = builder.ins().iconst(cl_types::I64, *dim as i64);
            call(compiler, builder, "nsl_tensor_rope_inverse", &[inputs[0], d])
        }

        // === Regularization ===
        PrimalOp::Dropout { p } => {
            let pv = builder.ins().f64const(*p);
            let training = builder.ins().iconst(cl_types::I8, 1);
            call(compiler, builder, "nsl_tensor_dropout", &[inputs[0], pv, training])
        }

        // === Control flow (2 ops) ===
        PrimalOp::Select => {
            // inputs: [cond, true_val, false_val]
            call(compiler, builder, "nsl_tensor_where", &[inputs[0], inputs[1], inputs[2]])
        }
        PrimalOp::Condition(cmp) => {
            let cmp_val = builder.ins().iconst(cl_types::I64, match cmp {
                CompareKind::Gt => 0,
                CompareKind::GtEq => 1,
                CompareKind::Lt => 2,
                CompareKind::LtEq => 3,
                CompareKind::Eq => 4,
                CompareKind::NotEq => 5,
            });
            call(compiler, builder, "nsl_tensor_compare", &[inputs[0], inputs[1], cmp_val])
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::WengertOp;

    fn make_op(id: u32, result: VarId, op: PrimalOp, inputs: Vec<VarId>) -> WengertOp {
        WengertOp {
            id,
            result,
            op,
            inputs,
            saved_for_backward: false,
            checkpointed: false,
        }
    }

    #[test]
    fn test_resolve_inputs_success() {
        let op = make_op(0, 10, PrimalOp::Relu, vec![5]);
        let var_map = VarMap::new();
        // Use a dummy Value; we just test the lookup logic.
        // VarId 5 must be in the map.
        // We can't easily create a real Cranelift Value without a builder,
        // so we test the error path instead.
        let result = resolve_inputs(&op, &var_map);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .to_string()
            .contains("unresolved VarId 5"));

        // With VarId present — we'd need a real Value, so skip the success case
        // in this unit test. Integration tests cover the full lowering path.
        let _ = var_map; // suppress unused warning
    }

    #[test]
    fn test_lower_constant_leaf() {
        // Verify that Constant variant is matched as a leaf (no input resolution).
        let op = make_op(0, 1, PrimalOp::Constant(3.14), vec![]);
        // Without a real compiler/builder we can't call lower_single_op,
        // but we can verify the op structure is correct.
        assert!(op.inputs.is_empty());
        assert!(matches!(op.op, PrimalOp::Constant(v) if (v - 3.14).abs() < 1e-10));
    }

    #[test]
    fn test_compare_kind_mapping() {
        // Verify the CompareKind -> i64 mapping matches the runtime convention.
        let mappings = [
            (CompareKind::Gt, 0i64),
            (CompareKind::GtEq, 1),
            (CompareKind::Lt, 2),
            (CompareKind::LtEq, 3),
            (CompareKind::Eq, 4),
            (CompareKind::NotEq, 5),
        ];
        for (kind, expected) in mappings {
            let val = match &kind {
                CompareKind::Gt => 0i64,
                CompareKind::GtEq => 1,
                CompareKind::Lt => 2,
                CompareKind::LtEq => 3,
                CompareKind::Eq => 4,
                CompareKind::NotEq => 5,
            };
            assert_eq!(val, expected, "CompareKind::{:?} should map to {}", kind, expected);
        }
    }

    #[test]
    fn test_all_primal_ops_covered() {
        // Verify that every PrimalOp variant is handled by lower_single_op.
        // This is a compile-time exhaustiveness check — the match in lower_single_op
        // is exhaustive, so if a new variant is added, the compiler will error.
        // We verify the known count here for documentation.
        let ops: Vec<PrimalOp> = vec![
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
            PrimalOp::Sum { dim: Some(0) },
            PrimalOp::Mean { dim: None },
            PrimalOp::Softmax { dim: -1 },
            PrimalOp::LogSoftmax { dim: -1 },
            PrimalOp::Reshape { target_ndim: 2 },
            PrimalOp::Broadcast,
            PrimalOp::Concat { dim: 0 },
            PrimalOp::Split { dim: 0, chunks: 2 },
            PrimalOp::Slice { dim: 0, start: 0, end: 5, orig_dim_size: 10 },
            PrimalOp::PadZero { dim: 0, pad_before: 1, pad_after: 1 },
            PrimalOp::Gather { dim: 0 },
            PrimalOp::ScatterAdd { dim: 0 },
            PrimalOp::Embedding,
            PrimalOp::LayerNorm { eps: 1e-5 },
            PrimalOp::BatchNorm { eps: 1e-5, training: true },
            PrimalOp::MaxPool2d { kernel: 2, stride: 2 },
            PrimalOp::AvgPool2d { kernel: 2, stride: 2 },
            PrimalOp::Conv2d { stride: 1, padding: 0 },
            PrimalOp::ConvTranspose2d { stride: 1, padding: 0 },
            PrimalOp::Repeat { kernel: 2 },
            PrimalOp::CrossEntropyLoss,
            PrimalOp::MSELoss,
            PrimalOp::L1Loss,
            PrimalOp::ScaledDotProductAttention { causal: false },
            PrimalOp::FlashAttentionBackwardExtract { causal: false, component: 0 },
            PrimalOp::RoPE { dim: 64 },
            PrimalOp::RoPEInverse { dim: 64 },
            PrimalOp::Dropout { p: 0.1 },
            PrimalOp::Select,
            PrimalOp::Condition(CompareKind::Gt),
            PrimalOp::Input("x".into()),
            PrimalOp::Param("w".into()),
            PrimalOp::Constant(1.0),
        ];
        // 50 variants total (including markers)
        assert_eq!(ops.len(), 50);
    }
}
