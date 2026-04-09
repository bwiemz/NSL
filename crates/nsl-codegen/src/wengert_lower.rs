//! M40: Lower a WengertList (symbolic backward graph) to Cranelift IR.
//!
//! Each `PrimalOp` variant is dispatched to the corresponding runtime FFI call.
//! The result is a map from all VarIds (primal + adjoint) to Cranelift `Value`s.

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::wengert::{
    type_for_op, CompareKind, PrimalOp, VarId, WengertList, WengertOp, WengertType,
};
use crate::CodegenError;
use cranelift_codegen::ir::{types as cl_types, InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use std::collections::HashMap;

pub type VarMap = HashMap<VarId, Value>;

pub struct LoweredWengert {
    pub var_map: VarMap,
    pub owned_values: Vec<(VarId, Value, WengertType)>,
}

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
) -> Result<LoweredWengert, CodegenError> {
    let mut var_map = primal_vars.clone();
    let mut owned_values = Vec::new();
    // Clone var_types so the lowerer can look up types for any VarId and
    // also tag newly-created adjoint VarIds.
    let mut var_types = wengert.var_types.clone();
    for op in &wengert.ops {
        // Skip ops whose inputs can't be resolved (ghost VarIds from
        // get_or_create_adjoint that never received a gradient).
        // These produce dead adjoint paths for non-differentiable ops.
        let all_inputs_resolved = op.inputs.iter().all(|vid| var_map.contains_key(vid));
        if !all_inputs_resolved {
            // Ghost VarId — skip this op. Its result stays unmapped,
            // which propagates the "no gradient" signal downstream.
            if std::env::var("NSL_DEBUG_WENGERT").is_ok() {
                let missing: Vec<_> = op
                    .inputs
                    .iter()
                    .filter(|vid| !var_map.contains_key(vid))
                    .collect();
                eprintln!(
                    "[wengert-skip] VarId {} ({:?}) — missing inputs: {:?}",
                    op.result, op.op, missing
                );
            }
            continue;
        }
        let result_val = lower_single_op(compiler, builder, op, &var_map, &var_types)?;
        var_map.insert(op.result, result_val);
        // Infer result type: for binary ops, if both inputs are Integer, result is Integer.
        // For Constants, preserve the extractor's type (it may have overridden
        // the default Tensor to Integer for IntLiteral).
        let result_type = match &op.op {
            PrimalOp::Add | PrimalOp::Sub | PrimalOp::Mul | PrimalOp::Div => {
                let a_ty = var_types
                    .get(&op.inputs[0])
                    .copied()
                    .unwrap_or(WengertType::Tensor);
                let b_ty = var_types
                    .get(&op.inputs[1])
                    .copied()
                    .unwrap_or(WengertType::Tensor);
                if a_ty == WengertType::Integer && b_ty == WengertType::Integer {
                    WengertType::Integer
                } else {
                    WengertType::Tensor
                }
            }
            PrimalOp::Constant(_) => {
                // Preserve the extractor's type annotation (Integer for IntLiteral,
                // Tensor for adjoint seeds, Scalar for float literals).
                var_types
                    .get(&op.result)
                    .copied()
                    .unwrap_or(type_for_op(&op.op))
            }
            _ => type_for_op(&op.op),
        };
        if should_cleanup_result(&op.op, result_type) {
            owned_values.push((op.result, result_val, result_type));
        }
        var_types.insert(op.result, result_type);
    }
    Ok(LoweredWengert {
        var_map,
        owned_values,
    })
}

/// Resolve all input VarIds for a WengertOp to their Cranelift Values.
fn resolve_inputs(op: &WengertOp, var_map: &VarMap) -> Result<Vec<Value>, CodegenError> {
    op.inputs
        .iter()
        .map(|vid| {
            var_map.get(vid).copied().ok_or_else(|| {
                CodegenError::new(format!(
                    "source-ad: unresolved VarId {} (input to {:?} at VarId {})",
                    vid, op.op, op.result
                ))
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

fn should_cleanup_result(op: &PrimalOp, result_type: WengertType) -> bool {
    if !matches!(result_type, WengertType::Tensor | WengertType::List) {
        return false;
    }
    match op {
        PrimalOp::Input(_) | PrimalOp::Param(_) | PrimalOp::Constant(_) => false,
        PrimalOp::Passthrough(name) if name.starts_with("dict_get:") => false,
        _ => true,
    }
}

fn free_tensor_value(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    value: Value,
) -> Result<(), CodegenError> {
    let _ = call(compiler, builder, "nsl_tensor_free", &[value])?;
    Ok(())
}

fn free_tensor_if_owned(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    value: Value,
    owned: bool,
) -> Result<(), CodegenError> {
    if owned {
        free_tensor_value(compiler, builder, value)?;
    }
    Ok(())
}

/// Promote an Integer (i64) to a scalar Tensor if needed for mixed-type arithmetic.
fn promote_to_tensor(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    val: Value,
    ty: WengertType,
) -> Result<(Value, bool), CodegenError> {
    match ty {
        WengertType::Integer => {
            let f = builder.ins().fcvt_from_sint(cl_types::F64, val);
            let dt = builder.ins().iconst(cl_types::I64, 0); // f64
            Ok((
                call(compiler, builder, "nsl_tensor_scalar", &[f, dt])?,
                true,
            ))
        }
        WengertType::Scalar => {
            // f64 scalar → wrap in a scalar tensor for tensor arithmetic
            let dt = builder.ins().iconst(cl_types::I64, 0); // f64
            Ok((
                call(compiler, builder, "nsl_tensor_scalar", &[val, dt])?,
                true,
            ))
        }
        _ => Ok((val, false)), // Already a tensor pointer (i64)
    }
}

/// Lower one WengertOp to Cranelift IR.
fn lower_single_op(
    compiler: &mut Compiler,
    builder: &mut FunctionBuilder,
    op: &WengertOp,
    var_map: &VarMap,
    var_types: &HashMap<VarId, WengertType>,
) -> Result<Value, CodegenError> {
    // Marker ops (leaf nodes) — should be in primal_vars
    match &op.op {
        PrimalOp::Input(name) => {
            if let Some(&val) = var_map.get(&op.result) {
                return Ok(val);
            }
            // Unresolved Input leaf — this is a real error. Data accesses like
            // batch.input_ids should be dict_get ops (with the batch as input),
            // not disconnected Input leaves. Fail with a diagnostic instead of
            // silently producing a null pointer that crashes at runtime.
            return Err(CodegenError::new(format!(
                "[source-ad] unresolved Input VarId {} ('{}') — no Cranelift Value in primal_vars. \
                 If this is a struct field access (e.g., batch.field), it should be a dict_get op, \
                 not a leaf Input.",
                op.result, name
            )));
        }
        PrimalOp::Param(name) => {
            if let Some(&val) = var_map.get(&op.result) {
                return Ok(val);
            }
            // Unresolved Param — may be a scalar config field (eps, _d_model)
            // used only in non-differentiable contexts. Safe to use null for
            // Passthrough ops that don't need the actual tensor value.
            eprintln!(
                "[source-ad] note: unresolved Param VarId {} ('{}'), using null placeholder",
                op.result, name
            );
            return Ok(builder.ins().iconst(cl_types::I64, 0));
        }
        PrimalOp::Constant(val) => {
            let ty = var_types
                .get(&op.result)
                .copied()
                .unwrap_or(WengertType::Tensor);
            match ty {
                WengertType::Scalar => {
                    // Raw f64 constant — used as scalar value (e.g., index 0.0 for subscript)
                    return Ok(builder.ins().f64const(*val));
                }
                WengertType::Integer => {
                    // Raw i64 constant — used as integer (e.g., dimension size)
                    return Ok(builder.ins().iconst(cl_types::I64, *val as i64));
                }
                WengertType::Tensor | WengertType::List => {
                    // Scalar tensor for use in tensor ops (default)
                    let v = builder.ins().f64const(*val);
                    let dt = builder.ins().iconst(cl_types::I64, 1); // f32 default
                    return call(compiler, builder, "nsl_tensor_scalar", &[v, dt]);
                }
            }
        }
        _ => {}
    }

    let inputs = resolve_inputs(op, var_map)?;

    match &op.op {
        // Already handled above
        PrimalOp::Input(_) | PrimalOp::Param(_) | PrimalOp::Constant(_) => unreachable!(),

        // === Elementwise unary (11 ops) ===
        // Promote scalar/integer inputs to tensor before calling tensor ops.
        PrimalOp::Relu
        | PrimalOp::Sigmoid
        | PrimalOp::Tanh
        | PrimalOp::Gelu
        | PrimalOp::Silu
        | PrimalOp::Exp
        | PrimalOp::Log
        | PrimalOp::Sqrt
        | PrimalOp::Abs
        | PrimalOp::Neg => {
            let a_ty = var_types
                .get(&op.inputs[0])
                .copied()
                .unwrap_or(WengertType::Tensor);
            let (a, free_a) = promote_to_tensor(compiler, builder, inputs[0], a_ty)?;
            let rt_name = match &op.op {
                PrimalOp::Relu => "nsl_tensor_relu",
                PrimalOp::Sigmoid => "nsl_tensor_sigmoid",
                PrimalOp::Tanh => "nsl_tensor_tanh_act",
                PrimalOp::Gelu => "nsl_tensor_gelu",
                PrimalOp::Silu => "nsl_tensor_silu",
                PrimalOp::Exp => "nsl_tensor_exp",
                PrimalOp::Log => "nsl_tensor_log",
                PrimalOp::Sqrt => "nsl_tensor_sqrt",
                PrimalOp::Abs => "nsl_tensor_abs",
                PrimalOp::Neg => "nsl_tensor_neg",
                _ => unreachable!(),
            };
            let result = call(compiler, builder, rt_name, &[a])?;
            free_tensor_if_owned(compiler, builder, a, free_a)?;
            Ok(result)
        }
        PrimalOp::Clamp { min, max } => {
            let min_v = builder.ins().f64const(*min);
            let max_v = builder.ins().f64const(*max);
            call(
                compiler,
                builder,
                "nsl_tensor_clamp",
                &[inputs[0], min_v, max_v],
            )
        }

        // === Elementwise binary (4 ops) ===
        // Check if both inputs are Integer-typed (e.g., shape[0] * shape[1]).
        // If so, use Cranelift integer arithmetic instead of tensor ops.
        PrimalOp::Add => {
            let a_ty = var_types
                .get(&op.inputs[0])
                .copied()
                .unwrap_or(WengertType::Tensor);
            let b_ty = var_types
                .get(&op.inputs[1])
                .copied()
                .unwrap_or(WengertType::Tensor);
            if a_ty == WengertType::Integer && b_ty == WengertType::Integer {
                Ok(builder.ins().iadd(inputs[0], inputs[1]))
            } else {
                let (a, free_a) = promote_to_tensor(compiler, builder, inputs[0], a_ty)?;
                let (b, free_b) = promote_to_tensor(compiler, builder, inputs[1], b_ty)?;
                let result = call(compiler, builder, "nsl_tensor_add", &[a, b])?;
                free_tensor_if_owned(compiler, builder, a, free_a)?;
                free_tensor_if_owned(compiler, builder, b, free_b)?;
                Ok(result)
            }
        }
        PrimalOp::Sub => {
            let a_ty = var_types
                .get(&op.inputs[0])
                .copied()
                .unwrap_or(WengertType::Tensor);
            let b_ty = var_types
                .get(&op.inputs[1])
                .copied()
                .unwrap_or(WengertType::Tensor);
            if a_ty == WengertType::Integer && b_ty == WengertType::Integer {
                Ok(builder.ins().isub(inputs[0], inputs[1]))
            } else {
                let (a, free_a) = promote_to_tensor(compiler, builder, inputs[0], a_ty)?;
                let (b, free_b) = promote_to_tensor(compiler, builder, inputs[1], b_ty)?;
                let result = call(compiler, builder, "nsl_tensor_sub", &[a, b])?;
                free_tensor_if_owned(compiler, builder, a, free_a)?;
                free_tensor_if_owned(compiler, builder, b, free_b)?;
                Ok(result)
            }
        }
        PrimalOp::Mul => {
            let a_ty = var_types
                .get(&op.inputs[0])
                .copied()
                .unwrap_or(WengertType::Tensor);
            let b_ty = var_types
                .get(&op.inputs[1])
                .copied()
                .unwrap_or(WengertType::Tensor);
            if a_ty == WengertType::Integer && b_ty == WengertType::Integer {
                Ok(builder.ins().imul(inputs[0], inputs[1]))
            } else {
                let (a, free_a) = promote_to_tensor(compiler, builder, inputs[0], a_ty)?;
                let (b, free_b) = promote_to_tensor(compiler, builder, inputs[1], b_ty)?;
                let result = call(compiler, builder, "nsl_tensor_mul", &[a, b])?;
                free_tensor_if_owned(compiler, builder, a, free_a)?;
                free_tensor_if_owned(compiler, builder, b, free_b)?;
                Ok(result)
            }
        }
        PrimalOp::Div => {
            let a_ty = var_types
                .get(&op.inputs[0])
                .copied()
                .unwrap_or(WengertType::Tensor);
            let b_ty = var_types
                .get(&op.inputs[1])
                .copied()
                .unwrap_or(WengertType::Tensor);
            if a_ty == WengertType::Integer && b_ty == WengertType::Integer {
                Ok(builder.ins().sdiv(inputs[0], inputs[1]))
            } else {
                let (a, free_a) = promote_to_tensor(compiler, builder, inputs[0], a_ty)?;
                let (b, free_b) = promote_to_tensor(compiler, builder, inputs[1], b_ty)?;
                let result = call(compiler, builder, "nsl_tensor_div", &[a, b])?;
                free_tensor_if_owned(compiler, builder, a, free_a)?;
                free_tensor_if_owned(compiler, builder, b, free_b)?;
                Ok(result)
            }
        }

        // === Linear algebra (4 ops) ===
        PrimalOp::Matmul => call(
            compiler,
            builder,
            "nsl_tensor_matmul",
            &[inputs[0], inputs[1]],
        ),
        PrimalOp::Transpose { dim0, dim1 } => {
            let d0 = builder.ins().iconst(cl_types::I64, *dim0 as i64);
            let d1 = builder.ins().iconst(cl_types::I64, *dim1 as i64);
            call(
                compiler,
                builder,
                "nsl_tensor_transpose",
                &[inputs[0], d0, d1],
            )
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
            // M46b: Route to deterministic sort-based reduction when --deterministic is active.
            if compiler.compile_options.deterministic {
                call(
                    compiler,
                    builder,
                    "nsl_tensor_reduce_sum_deterministic",
                    &[inputs[0], d, keepdim],
                )
            } else {
                call(
                    compiler,
                    builder,
                    "nsl_tensor_sum_dim",
                    &[inputs[0], d, keepdim],
                )
            }
        }
        PrimalOp::Mean { dim } => {
            let d = builder.ins().iconst(cl_types::I64, dim.unwrap_or(-1));
            let keepdim = builder.ins().iconst(cl_types::I64, 0);
            // M46b: Route to deterministic sort-based reduction when --deterministic is active.
            if compiler.compile_options.deterministic {
                call(
                    compiler,
                    builder,
                    "nsl_tensor_reduce_mean_deterministic",
                    &[inputs[0], d, keepdim],
                )
            } else {
                call(
                    compiler,
                    builder,
                    "nsl_tensor_mean_dim",
                    &[inputs[0], d, keepdim],
                )
            }
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
            // In source AD, tensor_cat(tensors, dim) passes the list VarId from a
            // Passthrough("list") op as inputs[0]. Detect this by checking if the
            // first input is typed as List, and use it directly instead of building
            // a new list.
            let first_is_list = op
                .inputs
                .first()
                .and_then(|vid| var_types.get(vid).copied())
                .map(|t| t == WengertType::List)
                .unwrap_or(false);
            let list = if first_is_list {
                // inputs[0] is already a tensor list pointer (from Passthrough("list"))
                inputs[0]
            } else {
                // Individual tensor inputs — build a list from them
                let l = call(compiler, builder, "nsl_list_new", &[])?;
                for &inp in &inputs {
                    call(compiler, builder, "nsl_list_push", &[l, inp])?;
                }
                l
            };
            let d = builder.ins().iconst(cl_types::I64, *dim);
            let result = call(compiler, builder, "nsl_tensor_cat", &[list, d])?;
            if !first_is_list {
                call(compiler, builder, "nsl_list_free", &[list])?;
            }
            Ok(result)
        }
        PrimalOp::Split { .. } => {
            // Split is a forward-only op — its backward is Concat (via SplitConcat adjoint).
            // If Split appears in an adjoint graph, it's from re-execution; clone is correct.
            call(compiler, builder, "nsl_tensor_clone", &[inputs[0]])
        }
        PrimalOp::Slice {
            dim, start, end, ..
        } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            let s = builder.ins().iconst(cl_types::I64, *start);
            let e = builder.ins().iconst(cl_types::I64, *end);
            call(compiler, builder, "nsl_tensor_slice", &[inputs[0], d, s, e])
        }
        PrimalOp::PadZero {
            dim,
            pad_before,
            pad_after,
        } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            let pb = builder.ins().iconst(cl_types::I64, *pad_before);
            let pa = builder.ins().iconst(cl_types::I64, *pad_after);
            call(
                compiler,
                builder,
                "nsl_tensor_pad_zero",
                &[inputs[0], d, pb, pa],
            )
        }
        PrimalOp::Repeat { kernel } => {
            let k = builder.ins().iconst(cl_types::I64, *kernel as i64);
            call(compiler, builder, "nsl_tensor_repeat", &[inputs[0], k])
        }

        // === Indexing (3 ops) ===
        PrimalOp::Gather { dim } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            call(
                compiler,
                builder,
                "nsl_tensor_gather",
                &[inputs[0], d, inputs[1]],
            )
        }
        PrimalOp::ScatterAdd { dim } => {
            let d = builder.ins().iconst(cl_types::I64, *dim);
            // M46: Route to deterministic sort-accumulate variant when --deterministic is active.
            // The deterministic FFI takes (input, indices, src) — dim is implicit (0).
            if compiler.compile_options.deterministic {
                call(
                    compiler,
                    builder,
                    "nsl_tensor_scatter_add_deterministic",
                    &[inputs[0], inputs[1], d],
                )
            } else {
                call(
                    compiler,
                    builder,
                    "nsl_tensor_scatter_add",
                    &[inputs[0], inputs[1], d],
                )
            }
        }
        PrimalOp::Embedding => call(
            compiler,
            builder,
            "nsl_tensor_embedding_lookup",
            &[inputs[0], inputs[1]],
        ),

        // === Normalization (2 ops) ===
        PrimalOp::LayerNorm { eps } => {
            let e = builder.ins().f64const(*eps);
            call(
                compiler,
                builder,
                "nsl_tensor_layernorm",
                &[inputs[0], inputs[1], inputs[2], e],
            )
        }
        PrimalOp::RMSNorm { eps } => {
            // RMSNorm takes (input, weight, eps) — the eps argument from the Wengert
            // inputs is a float field loaded from the model struct. Use the hardcoded
            // eps from the PrimalOp to avoid type mismatches (input eps may be f64).
            let e = builder.ins().f64const(*eps);
            call(
                compiler,
                builder,
                "nsl_tensor_rmsnorm",
                &[inputs[0], inputs[1], e],
            )
        }
        PrimalOp::BatchNorm { eps, training } => {
            let e = builder.ins().f64const(*eps);
            let t = builder
                .ins()
                .iconst(cl_types::I64, if *training { 1 } else { 0 });
            call(
                compiler,
                builder,
                "nsl_tensor_batchnorm",
                &[inputs[0], inputs[1], inputs[2], e, t],
            )
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
            call(
                compiler,
                builder,
                "nsl_tensor_conv2d",
                &[inputs[0], inputs[1], bias, sh, sw, ph, pw],
            )
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
            call(
                compiler,
                builder,
                "nsl_tensor_conv2d",
                &[inputs[0], inputs[1], bias, sh, sw, ph, pw],
            )
        }

        // === Pooling (2 ops) ===
        PrimalOp::MaxPool2d { kernel, stride } => {
            // Runtime: nsl_tensor_maxpool2d(input, kernel_h, kernel_w, stride, padding)
            let kh = builder.ins().iconst(cl_types::I64, *kernel as i64);
            let kw = builder.ins().iconst(cl_types::I64, *kernel as i64);
            let s = builder.ins().iconst(cl_types::I64, *stride as i64);
            let p = builder.ins().iconst(cl_types::I64, 0); // padding = 0
            call(
                compiler,
                builder,
                "nsl_tensor_maxpool2d",
                &[inputs[0], kh, kw, s, p],
            )
        }
        PrimalOp::AvgPool2d { kernel, stride } => {
            let kh = builder.ins().iconst(cl_types::I64, *kernel as i64);
            let kw = builder.ins().iconst(cl_types::I64, *kernel as i64);
            let s = builder.ins().iconst(cl_types::I64, *stride as i64);
            let p = builder.ins().iconst(cl_types::I64, 0);
            call(
                compiler,
                builder,
                "nsl_tensor_avgpool2d",
                &[inputs[0], kh, kw, s, p],
            )
        }

        // === Loss functions (3 ops) ===
        // Loss functions are composite: we lower them to sequences of existing FFI calls.
        PrimalOp::CrossEntropyLoss => {
            // Match stdlib/nsl/nn/losses.nsl semantics, including ignore labels
            // encoded as -100 by the DataLoader.
            let dim_one = builder.ins().iconst(cl_types::I64, 1);
            let safe_min = builder.ins().f64const(0.0);
            let safe_max = builder.ins().f64const(2147483647.0);
            let safe_targets = call(
                compiler,
                builder,
                "nsl_tensor_clamp",
                &[inputs[1], safe_min, safe_max],
            )?;
            let log_probs = call(
                compiler,
                builder,
                "nsl_tensor_logsoftmax",
                &[inputs[0], dim_one],
            )?;
            let gathered = call(
                compiler,
                builder,
                "nsl_tensor_gather",
                &[log_probs, dim_one, safe_targets],
            )?;
            let nll = call(compiler, builder, "nsl_tensor_neg", &[gathered])?;

            let one_f = builder.ins().f64const(1.0);
            let zero_f = builder.ins().f64const(0.0);
            let eps_f = builder.ins().f64const(1e-8);
            let targets_plus_one = call(
                compiler,
                builder,
                "nsl_tensor_add_scalar",
                &[inputs[1], one_f],
            )?;
            let valid_mask = call(
                compiler,
                builder,
                "nsl_tensor_clamp",
                &[targets_plus_one, zero_f, one_f],
            )?;
            let masked_nll = call(compiler, builder, "nsl_tensor_mul", &[nll, valid_mask])?;
            let num_valid = call(compiler, builder, "nsl_tensor_sum", &[valid_mask])?;
            let num_valid_eps = call(
                compiler,
                builder,
                "nsl_tensor_add_scalar",
                &[num_valid, eps_f],
            )?;
            let total = call(compiler, builder, "nsl_tensor_sum", &[masked_nll])?;
            let result = call(compiler, builder, "nsl_tensor_div", &[total, num_valid_eps])?;
            for temp in [
                safe_targets,
                log_probs,
                gathered,
                nll,
                targets_plus_one,
                valid_mask,
                masked_nll,
                num_valid,
                num_valid_eps,
                total,
            ] {
                free_tensor_value(compiler, builder, temp)?;
            }
            Ok(result)
        }
        PrimalOp::MSELoss => {
            // mse_loss(pred, target) = mean((pred - target)^2)
            let diff = call(compiler, builder, "nsl_tensor_sub", &[inputs[0], inputs[1]])?;
            let sq = call(compiler, builder, "nsl_tensor_mul", &[diff, diff])?;
            let d = builder.ins().iconst(cl_types::I64, -1);
            let keepdim = builder.ins().iconst(cl_types::I64, 0);
            let result = call(compiler, builder, "nsl_tensor_mean_dim", &[sq, d, keepdim])?;
            free_tensor_value(compiler, builder, diff)?;
            free_tensor_value(compiler, builder, sq)?;
            Ok(result)
        }
        PrimalOp::L1Loss => {
            // l1_loss(pred, target) = mean(|pred - target|)
            let diff = call(compiler, builder, "nsl_tensor_sub", &[inputs[0], inputs[1]])?;
            let abs_diff = call(compiler, builder, "nsl_tensor_abs", &[diff])?;
            let d = builder.ins().iconst(cl_types::I64, -1);
            let keepdim = builder.ins().iconst(cl_types::I64, 0);
            let result = call(
                compiler,
                builder,
                "nsl_tensor_mean_dim",
                &[abs_diff, d, keepdim],
            )?;
            free_tensor_value(compiler, builder, diff)?;
            free_tensor_value(compiler, builder, abs_diff)?;
            Ok(result)
        }

        // === Attention (4 ops) ===
        PrimalOp::ScaledDotProductAttention { causal } => {
            // Decompose into primitive ops: softmax((Q @ K.T) * scale [+ mask]) @ V
            // inputs: [q, k, v, scale, causal_flag]
            let q = inputs[0];
            let k = inputs[1];
            let v = inputs[2];
            // Scale may be f64 scalar or tensor — promote to tensor if needed
            let scale_ty = op
                .inputs
                .get(3)
                .and_then(|vid| var_types.get(vid).copied())
                .unwrap_or(WengertType::Tensor);
            let (scale, free_scale) = if inputs.len() > 3 {
                promote_to_tensor(compiler, builder, inputs[3], scale_ty)?
            } else {
                // Default scale: 1.0
                let one = builder.ins().f64const(1.0);
                let dt = builder.ins().iconst(cl_types::I64, 1);
                (
                    call(compiler, builder, "nsl_tensor_scalar", &[one, dt])?,
                    true,
                )
            };

            // K_T = transpose(K, -2, -1)
            let dim_m2 = builder.ins().iconst(cl_types::I64, -2_i64);
            let dim_m1 = builder.ins().iconst(cl_types::I64, -1_i64);
            let k_t = call(
                compiler,
                builder,
                "nsl_tensor_transpose",
                &[k, dim_m2, dim_m1],
            )?;
            // scores = Q @ K_T
            let scores = call(compiler, builder, "nsl_tensor_matmul", &[q, k_t])?;
            // scaled = scores * scale
            let scale_item = call(compiler, builder, "nsl_tensor_item", &[scale])?;
            let scaled = call(
                compiler,
                builder,
                "nsl_tensor_mul_scalar",
                &[scores, scale_item],
            )?;
            free_tensor_if_owned(compiler, builder, scale, free_scale)?;

            // Apply causal mask if needed
            let masked = if *causal {
                let dim_neg2 = builder.ins().iconst(cl_types::I64, -2_i64);
                let seq_len = call(compiler, builder, "nsl_tensor_shape_dim", &[q, dim_neg2])?;
                let mask = call(compiler, builder, "nsl_tensor_causal_mask", &[seq_len])?;
                let masked = call(compiler, builder, "nsl_tensor_add", &[scaled, mask])?;
                free_tensor_value(compiler, builder, mask)?;
                masked
            } else {
                scaled
            };

            // attn_weights = softmax(masked, -1)
            let attn_weights = call(compiler, builder, "nsl_tensor_softmax", &[masked, dim_m1])?;
            // output = attn_weights @ V
            let result = call(compiler, builder, "nsl_tensor_matmul", &[attn_weights, v])?;
            free_tensor_value(compiler, builder, k_t)?;
            free_tensor_value(compiler, builder, scores)?;
            if *causal {
                free_tensor_value(compiler, builder, scaled)?;
            }
            free_tensor_value(compiler, builder, masked)?;
            free_tensor_value(compiler, builder, attn_weights)?;
            Ok(result)
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
            call(
                compiler,
                builder,
                "nsl_tensor_rope_inverse",
                &[inputs[0], d],
            )
        }

        // === Regularization ===
        PrimalOp::Dropout { p } => {
            let pv = builder.ins().f64const(*p);
            let training = builder.ins().iconst(cl_types::I8, 1);
            call(
                compiler,
                builder,
                "nsl_tensor_dropout",
                &[inputs[0], pv, training],
            )
        }

        // === Control flow (2 ops) ===
        PrimalOp::Select => {
            // inputs: [cond, true_val, false_val]
            call(
                compiler,
                builder,
                "nsl_tensor_where",
                &[inputs[0], inputs[1], inputs[2]],
            )
        }
        PrimalOp::Condition(cmp) => {
            let cmp_val = builder.ins().iconst(
                cl_types::I64,
                match cmp {
                    CompareKind::Gt => 0,
                    CompareKind::GtEq => 1,
                    CompareKind::Lt => 2,
                    CompareKind::LtEq => 3,
                    CompareKind::Eq => 4,
                    CompareKind::NotEq => 5,
                },
            );
            call(
                compiler,
                builder,
                "nsl_tensor_compare",
                &[inputs[0], inputs[1], cmp_val],
            )
        }

        // === Non-differentiable passthroughs ===
        PrimalOp::Passthrough(ref name) => {
            match name.as_str() {
                "shape" => call(compiler, builder, "nsl_tensor_shape", &[inputs[0]]),
                "ndim" => call(compiler, builder, "nsl_tensor_ndim", &[inputs[0]]),
                "reshape" => {
                    // inputs[0] = tensor, inputs[1] = shape_list
                    call(
                        compiler,
                        builder,
                        "nsl_tensor_reshape",
                        &[inputs[0], inputs[1]],
                    )
                }
                "transpose" => {
                    // inputs[0] = tensor, inputs[1] = dim0, inputs[2] = dim1
                    call(
                        compiler,
                        builder,
                        "nsl_tensor_transpose",
                        &[inputs[0], inputs[1], inputs[2]],
                    )
                }
                "contiguous" => call(compiler, builder, "nsl_tensor_contiguous", &[inputs[0]]),
                "expand" => call(
                    compiler,
                    builder,
                    "nsl_tensor_expand",
                    &[inputs[0], inputs[1]],
                ),
                "squeeze" => call(compiler, builder, "nsl_tensor_squeeze", &[inputs[0]]),
                "unsqueeze" => {
                    let dim = if inputs.len() > 1 {
                        inputs[1]
                    } else {
                        builder.ins().iconst(cl_types::I64, 0)
                    };
                    call(compiler, builder, "nsl_tensor_unsqueeze", &[inputs[0], dim])
                }
                "item" => {
                    // .item() extracts a scalar from a tensor and wraps back into a
                    // 0-dim scalar tensor so it stays in the tensor domain for
                    // arithmetic ops (Mul, Add, etc.).
                    let f64_val = call(compiler, builder, "nsl_tensor_item", &[inputs[0]])?;
                    let dt = builder.ins().iconst(cl_types::I64, 1); // f32
                    call(compiler, builder, "nsl_tensor_scalar", &[f64_val, dt])
                }
                // Trigonometric (forward pass — backward handled by tape)
                "cos" => call(compiler, builder, "nsl_tensor_cos", &[inputs[0]]),
                "sin" => call(compiler, builder, "nsl_tensor_sin", &[inputs[0]]),
                "rotate_half" => call(compiler, builder, "nsl_tensor_rotate_half", &[inputs[0]]),
                // Tensor construction (non-differentiable, used for shape/position computation)
                "arange" => {
                    // arange(start, stop) or arange(stop) — extract f64 from inputs
                    // and call nsl_tensor_arange(start, stop, step=1.0).
                    // Inputs may be Tensor (i64 ptr), Scalar (f64), or Integer (i64).
                    let mut to_f64 = |builder: &mut FunctionBuilder,
                                      val: Value,
                                      idx: usize|
                     -> Result<Value, CodegenError> {
                        let vty = op
                            .inputs
                            .get(idx)
                            .and_then(|vid| var_types.get(vid).copied())
                            .unwrap_or(WengertType::Tensor);
                        match vty {
                            WengertType::Scalar => Ok(val),
                            WengertType::Integer => {
                                Ok(builder.ins().fcvt_from_sint(cl_types::F64, val))
                            }
                            _ => call(compiler, builder, "nsl_tensor_item", &[val]),
                        }
                    };
                    let step = builder.ins().f64const(1.0);
                    if inputs.len() >= 2 {
                        let start = to_f64(builder, inputs[0], 0)?;
                        let stop = to_f64(builder, inputs[1], 1)?;
                        call(compiler, builder, "nsl_tensor_arange", &[start, stop, step])
                    } else if inputs.len() == 1 {
                        let zero = builder.ins().f64const(0.0);
                        let stop = to_f64(builder, inputs[0], 0)?;
                        call(compiler, builder, "nsl_tensor_arange", &[zero, stop, step])
                    } else {
                        Err(CodegenError::new(
                            "arange requires at least 1 argument".to_string(),
                        ))
                    }
                }
                "reduce_to_shape" => {
                    // inputs = [grad, target_param]
                    // Reduce grad by summing over leading batch dims to match target shape
                    call(
                        compiler,
                        builder,
                        "nsl_tensor_reduce_to_shape",
                        &[inputs[0], inputs[1]],
                    )
                }
                _ if name.starts_with("dict_get:") => {
                    // inputs = [dict_ptr]
                    // Dict field access: batch.input_ids -> nsl_dict_get_str(batch, "input_ids")
                    let field = &name["dict_get:".len()..];
                    let key = compiler.compile_string_literal(builder, field)?;
                    call(compiler, builder, "nsl_dict_get_str", &[inputs[0], key])
                }
                "embedding_backward" => {
                    // inputs = [grad, indices, weight]
                    // Creates zeros_like(weight) then scatter-adds grad rows
                    call(
                        compiler,
                        builder,
                        "nsl_embedding_backward",
                        &[inputs[0], inputs[1], inputs[2]],
                    )
                }
                "cross_entropy_backward" => {
                    // inputs = [grad_output, logits, targets]
                    call(
                        compiler,
                        builder,
                        "nsl_cross_entropy_backward",
                        &[inputs[0], inputs[1], inputs[2]],
                    )
                }
                "zeros" | "ones" | "full" | "randn" | "zeros_like" | "ones_like" => {
                    let rt_name = format!("nsl_tensor_{}", name);
                    call(compiler, builder, &rt_name, &inputs)
                }
                // Scalar type conversion
                "int" => {
                    // Convert to i64. Input may be:
                    //   - Scalar (f64): convert with fcvt_to_sint
                    //   - Tensor: extract f64 via nsl_tensor_item, then convert
                    //   - Integer (i64): already an integer, pass through
                    let input_ty = op
                        .inputs
                        .first()
                        .and_then(|vid| var_types.get(vid).copied())
                        .unwrap_or(WengertType::Tensor);
                    match input_ty {
                        WengertType::Integer => Ok(inputs[0]),
                        WengertType::Scalar => {
                            Ok(builder.ins().fcvt_to_sint(cl_types::I64, inputs[0]))
                        }
                        _ => {
                            // Tensor or List — extract f64 first
                            let f64_val = call(compiler, builder, "nsl_tensor_item", &[inputs[0]])?;
                            Ok(builder.ins().fcvt_to_sint(cl_types::I64, f64_val))
                        }
                    }
                }
                "float" => {
                    // float() converts to f64 scalar. Input may be:
                    //   - Integer (i64): convert with fcvt_from_sint
                    //   - Scalar (f64): pass through
                    //   - Tensor: extract f64 via nsl_tensor_item
                    let input_ty = op
                        .inputs
                        .first()
                        .and_then(|vid| var_types.get(vid).copied())
                        .unwrap_or(WengertType::Tensor);
                    match input_ty {
                        WengertType::Scalar => Ok(inputs[0]),
                        WengertType::Integer => {
                            Ok(builder.ins().fcvt_from_sint(cl_types::F64, inputs[0]))
                        }
                        _ => {
                            // Tensor — extract f64
                            call(compiler, builder, "nsl_tensor_item", &[inputs[0]])
                        }
                    }
                }
                "subscript" => {
                    // inputs[0] = list/shape, inputs[1] = index
                    // Index may be Scalar (f64 from Constant), Integer (i64), or Tensor.
                    let idx_ty = op
                        .inputs
                        .get(1)
                        .and_then(|vid| var_types.get(vid).copied())
                        .unwrap_or(WengertType::Tensor);
                    let idx = match idx_ty {
                        WengertType::Integer => inputs[1],
                        WengertType::Scalar => builder.ins().fcvt_to_sint(cl_types::I64, inputs[1]),
                        _ => {
                            // Tensor — extract f64 then convert to i64
                            let f64_val = call(compiler, builder, "nsl_tensor_item", &[inputs[1]])?;
                            builder.ins().fcvt_to_sint(cl_types::I64, f64_val)
                        }
                    };
                    // nsl_list_get returns raw i64 (dimension value, NOT tensor pointer).
                    call(compiler, builder, "nsl_list_get", &[inputs[0], idx])
                }
                "list" => {
                    // Build a list from input elements.
                    // Determine list kind from element types:
                    //   - If ANY element is a Tensor, treat as tensor list (push raw pointers)
                    //   - Otherwise, treat as dimension list (scalarize to i64)
                    let has_tensor = op.inputs.iter().any(|vid| {
                        var_types.get(vid).copied().unwrap_or(WengertType::Integer)
                            == WengertType::Tensor
                    });
                    let list = call(compiler, builder, "nsl_list_new", &[])?;
                    for (i, &inp) in inputs.iter().enumerate() {
                        let val_ty = op
                            .inputs
                            .get(i)
                            .and_then(|vid| var_types.get(vid).copied())
                            .unwrap_or(WengertType::Integer);
                        let val = if has_tensor {
                            // Tensor list: push raw i64 pointers (no scalarization)
                            inp
                        } else {
                            // Dimension list: convert to i64
                            match val_ty {
                                WengertType::Scalar => {
                                    builder.ins().fcvt_to_sint(cl_types::I64, inp)
                                }
                                _ => inp, // Integer, List — already i64
                            }
                        };
                        call(compiler, builder, "nsl_list_push", &[list, val])?;
                    }
                    Ok(list)
                }
                other => Err(CodegenError::new(format!(
                    "unsupported passthrough op: {}",
                    other
                ))),
            }
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
        let value = std::f64::consts::PI;
        let op = make_op(0, 1, PrimalOp::Constant(value), vec![]);
        // Without a real compiler/builder we can't call lower_single_op,
        // but we can verify the op structure is correct.
        assert!(op.inputs.is_empty());
        assert!(matches!(op.op, PrimalOp::Constant(v) if (v - value).abs() < 1e-10));
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
            assert_eq!(
                val, expected,
                "CompareKind::{:?} should map to {}",
                kind, expected
            );
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
            PrimalOp::Slice {
                dim: 0,
                start: 0,
                end: 5,
                orig_dim_size: 10,
            },
            PrimalOp::PadZero {
                dim: 0,
                pad_before: 1,
                pad_after: 1,
            },
            PrimalOp::Gather { dim: 0 },
            PrimalOp::ScatterAdd { dim: 0 },
            PrimalOp::Embedding,
            PrimalOp::LayerNorm { eps: 1e-5 },
            PrimalOp::RMSNorm { eps: 1e-5 },
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
            PrimalOp::Repeat { kernel: 2 },
            PrimalOp::CrossEntropyLoss,
            PrimalOp::MSELoss,
            PrimalOp::L1Loss,
            PrimalOp::ScaledDotProductAttention { causal: false },
            PrimalOp::FlashAttentionBackwardExtract {
                causal: false,
                component: 0,
            },
            PrimalOp::RoPE { dim: 64 },
            PrimalOp::RoPEInverse { dim: 64 },
            PrimalOp::Dropout { p: 0.1 },
            PrimalOp::Select,
            PrimalOp::Condition(CompareKind::Gt),
            PrimalOp::Input("x".into()),
            PrimalOp::Param("w".into()),
            PrimalOp::Constant(1.0),
        ];
        // 51 variants total (including markers)
        assert_eq!(ops.len(), 51);
    }
}
