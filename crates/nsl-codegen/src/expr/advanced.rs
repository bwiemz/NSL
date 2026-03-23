use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{Linkage, Module};

use nsl_ast::expr::{Expr, ExprKind, SubscriptKind, FStringPart};
use nsl_ast::operator::BinOp;
use nsl_ast::Symbol;
use nsl_semantic::types::Type;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::types::{nsl_type_to_cl, pointer_type};

impl Compiler<'_> {
    pub(crate) fn compile_if_expr(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        condition: &Expr,
        then_expr: &Expr,
        else_expr: &Expr,
        _full_expr: &Expr,
    ) -> Result<Value, CodegenError> {
        let cond_val = self.compile_expr(builder, state, condition)?;
        let then_block = builder.create_block();
        let else_block = builder.create_block();
        let merge_block = builder.create_block();

        let result_type = nsl_type_to_cl(&self.node_type(then_expr.id).clone());
        builder.append_block_param(merge_block, result_type);
        builder.ins().brif(cond_val, then_block, &[], else_block, &[]);

        builder.switch_to_block(then_block);
        builder.seal_block(then_block);
        let then_val = self.compile_expr(builder, state, then_expr)?;
        builder.ins().jump(merge_block, &[then_val]);

        builder.switch_to_block(else_block);
        builder.seal_block(else_block);
        let else_val = self.compile_expr(builder, state, else_expr)?;
        builder.ins().jump(merge_block, &[else_val]);

        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);
        Ok(builder.block_params(merge_block)[0])
    }

    pub(crate) fn compile_lambda(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        params: &[nsl_ast::expr::LambdaParam],
        body: &Expr,
    ) -> Result<Value, CodegenError> {
        let lambda_name = format!("__nsl_lambda_{}", self.next_func_index());

        // Detect free variables (captures from outer scope)
        let param_syms: std::collections::HashSet<nsl_ast::Symbol> =
            params.iter().map(|p| p.name).collect();
        let captures = self.find_free_variables(body, &param_syms, state);

        // Build the Cranelift signature: normal params + capture params
        let mut sig = self.module.make_signature();
        sig.call_conv = self.call_conv;

        let mut param_info = Vec::new();
        for param in params {
            let cl_type = cl_types::I64; // Default to I64 for all params
            sig.params.push(AbiParam::new(cl_type));
            param_info.push((param.name, cl_type));
        }

        // Add capture params to signature (all I64 -- pointers or ints)
        let mut capture_info = Vec::new();
        for &(sym, cl_type) in &captures {
            sig.params.push(AbiParam::new(cl_type));
            capture_info.push((sym, cl_type));
        }

        // Return type from type map
        let ret_type = nsl_type_to_cl(&self.node_type(body.id).clone());
        sig.returns.push(AbiParam::new(ret_type));

        // Declare the function in the module
        let func_id = self.module
            .declare_function(&lambda_name, Linkage::Local, &sig)
            .map_err(|e| CodegenError::new(format!("failed to declare lambda '{lambda_name}': {e}")))?;

        // Store for compilation and in functions table
        self.functions.insert(lambda_name.clone(), (func_id, sig.clone()));
        self.pending_lambdas.push(crate::compiler::PendingLambda {
            name: lambda_name,
            func_id,
            sig,
            params: param_info,
            body: body.clone(),
            captures: capture_info,
        });

        // Get function address
        let func_ref = self.module.declare_func_in_func(func_id, builder.func);
        let addr = builder.ins().func_addr(pointer_type(), func_ref);

        if captures.is_empty() {
            // No captures -- return bare function pointer (backward compatible)
            Ok(addr)
        } else {
            // Allocate closure struct: { fn_ptr (8 bytes), num_captures (8 bytes), captures[] (8 bytes each) }
            let struct_size = 8 + 8 + captures.len() * 8;
            let alloc_id = self.runtime_fns["nsl_alloc"].0;
            let alloc_ref = self.module.declare_func_in_func(alloc_id, builder.func);
            let size_val = builder.ins().iconst(cl_types::I64, struct_size as i64);
            let call = builder.ins().call(alloc_ref, &[size_val]);
            let closure_ptr = builder.inst_results(call)[0];

            // Store fn_ptr at offset 0
            builder.ins().store(MemFlags::trusted(), addr, closure_ptr, 0);
            // Store num_captures at offset 8
            let num_cap = builder.ins().iconst(cl_types::I64, captures.len() as i64);
            builder.ins().store(MemFlags::trusted(), num_cap, closure_ptr, 8);
            // Store each captured variable at offset 16 + i*8
            for (i, (sym, _cl_type)) in captures.iter().enumerate() {
                let (var, _) = *state.variables.get(sym).ok_or_else(|| {
                    CodegenError::new(format!(
                        "captured variable '{}' not found in outer scope",
                        self.resolve_sym(*sym)
                    ))
                })?;
                let val = builder.use_var(var);
                let offset = (16 + i * 8) as i32;
                builder.ins().store(MemFlags::trusted(), val, closure_ptr, offset);
            }

            // TODO(M19): Implement nsl_closure_free -- closure_ptr is heap-allocated via nsl_alloc
            // and currently leaks when the variable holding it goes out of scope, since it's not
            // hooked into the NslTensor refcounting or scope-teardown logic.

            // Record how many captures this closure has (keyed by a sentinel -- caller will
            // transfer this to the variable that receives the closure result).
            self.last_lambda_capture_count = Some(captures.len());

            Ok(closure_ptr)
        }
    }

    /// Walk an expression AST to find variables that reference outer-scope bindings,
    /// i.e. not lambda params and not function/builtin names.
    pub(crate) fn find_free_variables(
        &self,
        expr: &Expr,
        param_syms: &std::collections::HashSet<nsl_ast::Symbol>,
        state: &FuncState,
    ) -> Vec<(nsl_ast::Symbol, cl_types::Type)> {
        let mut free_vars = Vec::new();
        let mut seen = std::collections::HashSet::new();
        self.collect_free_vars(expr, param_syms, state, &mut free_vars, &mut seen);
        free_vars
    }

    pub(crate) fn collect_free_vars(
        &self,
        expr: &Expr,
        param_syms: &std::collections::HashSet<nsl_ast::Symbol>,
        state: &FuncState,
        out: &mut Vec<(nsl_ast::Symbol, cl_types::Type)>,
        seen: &mut std::collections::HashSet<nsl_ast::Symbol>,
    ) {
        match &expr.kind {
            ExprKind::Ident(sym) => {
                if !param_syms.contains(sym) && !seen.contains(sym) {
                    if let Some((_var, cl_type)) = state.variables.get(sym) {
                        // It's an outer-scope variable -- it's a capture
                        seen.insert(*sym);
                        out.push((*sym, *cl_type));
                    }
                    // If not in state.variables, it's a global/builtin -- not a capture
                }
            }
            ExprKind::BinaryOp { left, right, .. } | ExprKind::Pipe { left, right } => {
                self.collect_free_vars(left, param_syms, state, out, seen);
                self.collect_free_vars(right, param_syms, state, out, seen);
            }
            ExprKind::UnaryOp { operand, .. } => {
                self.collect_free_vars(operand, param_syms, state, out, seen);
            }
            ExprKind::Call { callee, args } => {
                self.collect_free_vars(callee, param_syms, state, out, seen);
                for arg in args {
                    self.collect_free_vars(&arg.value, param_syms, state, out, seen);
                }
            }
            ExprKind::MemberAccess { object, .. } => {
                self.collect_free_vars(object, param_syms, state, out, seen);
            }
            ExprKind::Subscript { object, index } => {
                self.collect_free_vars(object, param_syms, state, out, seen);
                match index.as_ref() {
                    SubscriptKind::Index(e) => self.collect_free_vars(e, param_syms, state, out, seen),
                    SubscriptKind::Slice { lower, upper, step } => {
                        if let Some(e) = lower { self.collect_free_vars(e, param_syms, state, out, seen); }
                        if let Some(e) = upper { self.collect_free_vars(e, param_syms, state, out, seen); }
                        if let Some(e) = step { self.collect_free_vars(e, param_syms, state, out, seen); }
                    }
                    _ => {}
                }
            }
            ExprKind::FString(parts) => {
                for part in parts {
                    if let FStringPart::Expr(e) = part {
                        self.collect_free_vars(e, param_syms, state, out, seen);
                    }
                }
            }
            ExprKind::ListLiteral(elems) | ExprKind::TupleLiteral(elems) => {
                for e in elems {
                    self.collect_free_vars(e, param_syms, state, out, seen);
                }
            }
            ExprKind::DictLiteral(pairs) => {
                for (k, v) in pairs {
                    self.collect_free_vars(k, param_syms, state, out, seen);
                    self.collect_free_vars(v, param_syms, state, out, seen);
                }
            }
            ExprKind::IfExpr { condition, then_expr, else_expr } => {
                self.collect_free_vars(condition, param_syms, state, out, seen);
                self.collect_free_vars(then_expr, param_syms, state, out, seen);
                self.collect_free_vars(else_expr, param_syms, state, out, seen);
            }
            // Literals and other leaf nodes -- no free variables
            _ => {}
        }
    }

    pub(crate) fn compile_higher_order_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        func_name: &str,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        if args.len() != 2 {
            return Err(CodegenError::new(format!("{func_name}() takes exactly 2 arguments")));
        }
        // Error if first arg is a closure (captures variables) -- HOFs in C runtime expect bare fn ptrs
        if let ExprKind::Ident(sym) = &args[0].value.kind {
            if self.closure_info.contains_key(sym) {
                let name = self.resolve_sym(*sym).to_string();
                return Err(CodegenError::new(format!(
                    "cannot pass closure '{name}' to {func_name}() -- closures with captured variables \
                     are not supported in higher-order functions yet. Use a non-capturing lambda instead."
                )));
            }
        }
        let fn_val = self.compile_expr(builder, state, &args[0].value)?;
        // Also catch inline capturing lambdas (not just named variables)
        if self.last_lambda_capture_count.take().is_some() {
            return Err(CodegenError::new(format!(
                "cannot pass capturing closure to {func_name}() -- closures with captured variables \
                 are not supported in higher-order functions yet. Use a non-capturing lambda instead."
            )));
        }
        let list_val = self.compile_expr(builder, state, &args[1].value)?;
        let rt_name = format!("nsl_{func_name}");
        self.compile_call_by_name(builder, &rt_name, &[fn_val, list_val])
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn compile_tensor_binary_op(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        lhs: Value,
        rhs: Value,
        op: BinOp,
        left_is_tensor: bool,
        right_is_tensor: bool,
    ) -> Result<Value, CodegenError> {
        if left_is_tensor && right_is_tensor {
            // Tensor-tensor ops
            let rt_name = match op {
                BinOp::Add => "nsl_tensor_add",
                BinOp::Sub => "nsl_tensor_sub",
                BinOp::Mul => "nsl_tensor_mul",
                BinOp::Div => "nsl_tensor_div",
                BinOp::MatMul => if state.is_fp8_compute {
                    "nsl_fp8_matmul_training"
                } else {
                    "nsl_tensor_matmul"
                },
                _ => return Err(CodegenError::new(format!("unsupported tensor op: {op:?}"))),
            };
            let result = self.compile_traced_call(builder, rt_name, &[lhs, rhs])?;
            state.tensor_temporaries.push(result);
            Ok(result)
        } else if left_is_tensor {
            // Tensor-scalar: ensure scalar is f64 for the runtime FFI
            // (runtime handles f64->f32 demotion when tensor is f32)
            let rhs_ty = builder.func.dfg.value_type(rhs);
            let scalar = if rhs_ty == cl_types::F64 {
                rhs
            } else if rhs_ty == cl_types::F32 {
                // F32 scalar (e.g., from model weight .item()) -- promote to f64
                builder.ins().fpromote(cl_types::F64, rhs)
            } else if rhs_ty.is_int() {
                builder.ins().fcvt_from_sint(cl_types::F64, rhs)
            } else {
                rhs
            };
            let result = match op {
                BinOp::Add => self.compile_traced_call(builder, "nsl_tensor_add_scalar", &[lhs, scalar])?,
                BinOp::Mul => self.compile_traced_call(builder, "nsl_tensor_mul_scalar", &[lhs, scalar])?,
                BinOp::Sub => {
                    // tensor - scalar = tensor + (-scalar)
                    let neg = builder.ins().fneg(scalar);
                    self.compile_traced_call(builder, "nsl_tensor_add_scalar", &[lhs, neg])?
                }
                BinOp::Div => {
                    // tensor / scalar = tensor * (1/scalar)
                    let one = builder.ins().f64const(1.0);
                    let inv = builder.ins().fdiv(one, scalar);
                    self.compile_traced_call(builder, "nsl_tensor_mul_scalar", &[lhs, inv])?
                }
                _ => return Err(CodegenError::new(format!("unsupported tensor-scalar op: {op:?}"))),
            };
            state.tensor_temporaries.push(result);
            Ok(result)
        } else {
            // Scalar-tensor: ensure scalar is f64
            let lhs_ty = builder.func.dfg.value_type(lhs);
            let scalar = if lhs_ty == cl_types::F64 { lhs } else { builder.ins().fcvt_from_sint(cl_types::F64, lhs) };
            let result = match op {
                BinOp::Add => self.compile_traced_call(builder, "nsl_tensor_add_scalar", &[rhs, scalar])?,
                BinOp::Mul => self.compile_traced_call(builder, "nsl_tensor_mul_scalar", &[rhs, scalar])?,
                BinOp::Sub => {
                    // scalar - tensor = neg(tensor) + scalar
                    let neg_tensor = self.compile_call_by_name(builder, "nsl_tensor_neg", &[rhs])?;
                    let result = self.compile_call_by_name(builder, "nsl_tensor_add_scalar", &[neg_tensor, scalar])?;
                    // Free the intermediate neg_tensor (consumed by add_scalar)
                    self.compile_call_by_name(builder, "nsl_tensor_free", &[neg_tensor])?;
                    result
                }
                _ => return Err(CodegenError::new(format!("unsupported scalar-tensor op: {op:?}"))),
            };
            state.tensor_temporaries.push(result);
            Ok(result)
        }
    }

    pub(crate) fn compile_model_method_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        object: &Expr,
        model_name: &str,
        method_name: &str,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        // Compile object to get self pointer
        let self_val = self.compile_expr(builder, state, object)?;

        // Look up mangled method name.
        // For locally defined models, use the model_methods map.
        // For imported models, fall back to the standard naming convention.
        let mangled = if let Some(methods) = self.model_methods.get(model_name) {
            methods.get(method_name).ok_or_else(|| {
                CodegenError::new(format!("model '{model_name}' has no method '{method_name}'"))
            })?.clone()
        } else {
            // Imported model: derive mangled name from convention
            format!("__nsl_model_{model_name}_{method_name}")
        };

        // Compile args
        let mut arg_vals = vec![self_val];
        for arg in args {
            arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
        }

        self.compile_call_by_name(builder, &mangled, &arg_vals)
    }

    pub(crate) fn compile_tensor_method_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        object: &Expr,
        method: &str,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        let obj_val = self.compile_expr(builder, state, object)?;
        match method {
            // M46: Full reductions (no dim arg) don't have 1-arg deterministic variants;
            // deterministic swap only applies to the 3-arg sum_dim/mean_dim in calls.rs.
            "sum" => self.compile_traced_call(builder, "nsl_tensor_sum", &[obj_val]),
            "mean" => self.compile_traced_call(builder, "nsl_tensor_mean", &[obj_val]),
            "reshape" => {
                if args.len() != 1 {
                    return Err(CodegenError::new("reshape() takes exactly 1 argument (shape list)"));
                }
                let shape_val = self.compile_expr(builder, state, &args[0].value)?;
                self.compile_call_by_name(builder, "nsl_tensor_reshape", &[obj_val, shape_val])
            }
            "transpose" => {
                if args.len() != 2 {
                    return Err(CodegenError::new("transpose() takes exactly 2 arguments (dim0, dim1)"));
                }
                let d0 = self.compile_expr(builder, state, &args[0].value)?;
                let d1 = self.compile_expr(builder, state, &args[1].value)?;
                self.compile_call_by_name(builder, "nsl_tensor_transpose", &[obj_val, d0, d1])
            }
            "clone" => {
                // FBIP Phase 2: skip clone when source binding is single-use
                // (the binding won't be referenced again, so cloning is unnecessary)
                if let ExprKind::Ident(sym) = &object.kind {
                    if let Some(ref uc) = state.use_counts {
                        if uc.is_single_use(sym) && !state.in_tape_region {
                            return Ok(obj_val); // elide clone
                        }
                    }
                }
                self.compile_call_by_name(builder, "nsl_tensor_clone", &[obj_val])
            }
            "item" => self.compile_call_by_name(builder, "nsl_tensor_item", &[obj_val]),
            "to" => {
                if args.len() != 1 {
                    return Err(CodegenError::new("to() takes exactly 1 argument (device or dtype)"));
                }
                // Check if the argument is a custom dtype name
                if let ExprKind::Ident(arg_sym) = &args[0].value.kind {
                    let arg_name = self.resolve_sym(*arg_sym).to_string();
                    if let Some(dtype_id) = self.resolve_custom_dtype(&arg_name) {
                        // .to(CustomDtype) -- convert to custom dtype
                        let id_val = builder.ins().iconst(cl_types::I64, dtype_id as i64);
                        return self.compile_call_by_name(builder, "nsl_tensor_to_custom_dtype", &[obj_val, id_val]);
                    }
                    // .to(f32) / .to(f64) -- convert from custom dtype back to standard
                    if matches!(arg_name.as_str(), "f32" | "f64" | "float") {
                        return self.compile_call_by_name(builder, "nsl_tensor_from_custom_dtype", &[obj_val]);
                    }
                }
                // Fall through: device transfer (.to(cuda), .to(cpu), etc.)
                let device_val = self.compile_expr(builder, state, &args[0].value)?;
                self.compile_call_by_name(builder, "nsl_tensor_to_device", &[obj_val, device_val])
            }
            // M18a shape ops
            "unsqueeze" => {
                if args.len() != 1 {
                    return Err(CodegenError::new("unsqueeze() takes exactly 1 argument (dim)"));
                }
                let dim_val = self.compile_expr(builder, state, &args[0].value)?;
                self.compile_call_by_name(builder, "nsl_tensor_unsqueeze", &[obj_val, dim_val])
            }
            "select" => {
                if args.len() != 2 {
                    return Err(CodegenError::new("select() takes exactly 2 arguments (dim, index)"));
                }
                let dim_val = self.compile_expr(builder, state, &args[0].value)?;
                let idx_val = self.compile_expr(builder, state, &args[1].value)?;
                self.compile_call_by_name(builder, "nsl_tensor_select", &[obj_val, dim_val, idx_val])
            }
            "expand" => {
                if args.len() != 1 {
                    return Err(CodegenError::new("expand() takes exactly 1 argument (shape list)"));
                }
                let shape_val = self.compile_expr(builder, state, &args[0].value)?;
                self.compile_call_by_name(builder, "nsl_tensor_expand", &[obj_val, shape_val])
            }
            "contiguous" => {
                self.compile_call_by_name(builder, "nsl_tensor_contiguous", &[obj_val])
            }
            "slice" => {
                if args.len() != 3 {
                    return Err(CodegenError::new("slice() takes exactly 3 arguments (dim, start, end)"));
                }
                let dim_val = self.compile_expr(builder, state, &args[0].value)?;
                let start_val = self.compile_expr(builder, state, &args[1].value)?;
                let end_val = self.compile_expr(builder, state, &args[2].value)?;
                self.compile_call_by_name(builder, "nsl_tensor_slice", &[obj_val, dim_val, start_val, end_val])
            }
            // M19: cumsum(dim)
            "cumsum" => {
                if args.len() != 1 {
                    return Err(CodegenError::new("cumsum() takes exactly 1 argument (dim)"));
                }
                let dim_val = self.compile_expr(builder, state, &args[0].value)?;
                self.compile_call_by_name(builder, "nsl_tensor_cumsum", &[obj_val, dim_val])
            }
            // M19: shape(dim) -- returns a single dimension size as i64
            "shape" => {
                if args.len() != 1 {
                    return Err(CodegenError::new("shape() takes exactly 1 argument (dim)"));
                }
                let dim_val = self.compile_expr(builder, state, &args[0].value)?;
                self.compile_call_by_name(builder, "nsl_tensor_shape_dim", &[obj_val, dim_val])
            }
            _ => Err(CodegenError::new(format!("unknown tensor method '.{method}()'"))),
        }
    }

    pub(crate) fn compile_str_method_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        object: &Expr,
        method: &str,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        let obj_val = self.compile_expr(builder, state, object)?;
        let rt_name = format!("nsl_str_{method}");

        if !self.runtime_fns.contains_key(&rt_name) {
            return Err(CodegenError::new(format!("unknown string method '.{method}()'")));
        }

        let mut arg_vals = vec![obj_val];
        for arg in args {
            arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
        }
        self.compile_call_by_name(builder, &rt_name, &arg_vals)
    }

    pub(crate) fn compile_list_method_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        object: &Expr,
        method: &str,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        let obj_val = self.compile_expr(builder, state, object)?;
        match method {
            "append" | "push" => {
                if let Some(arg) = args.first() {
                    let val = self.compile_expr(builder, state, &arg.value)?;
                    let fid = self.runtime_fns["nsl_list_push"].0;
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    builder.ins().call(fref, &[obj_val, val]);
                    Ok(builder.ins().iconst(cl_types::I64, 0))
                } else {
                    Err(CodegenError::new("append/push requires 1 argument"))
                }
            }
            "len" => {
                let fid = self.runtime_fns["nsl_list_len"].0;
                let fref = self.module.declare_func_in_func(fid, builder.func);
                let call = builder.ins().call(fref, &[obj_val]);
                Ok(builder.inst_results(call)[0])
            }
            _ => Err(CodegenError::new(format!("unknown list method '.{method}()'")))
        }
    }

    pub(crate) fn compile_flash_attention_call(
        &mut self,
        builder: &mut FunctionBuilder,
        q_val: Value,
        k_val: Value,
        v_val: Value,
        scale_val: Value,
    ) -> Result<Value, CodegenError> {
        let ctx = self
            .flash_attention_context
            .as_ref()
            .ok_or_else(|| CodegenError::new("flash_attention_context not set"))?;

        let ptx_data_id = ctx.ptx_data_id;
        let name_data_id = ctx.name_data_id;
        let block_q = ctx.config.block_q;
        let block_kv = ctx.config.block_kv;
        let is_causal = ctx.config.causal;
        let shmem_bytes = crate::flash_attention::shared_mem_bytes(&ctx.config) as i64;

        // Allocate output tensor (same shape as Q, same device)
        let out_val = self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[q_val])?;

        // Get PTX and name pointers from .rodata
        let ptx_gv = self.module.declare_data_in_func(ptx_data_id, builder.func);
        let ptx_ptr = builder.ins().symbol_value(pointer_type(), ptx_gv);
        let name_gv = self.module.declare_data_in_func(name_data_id, builder.func);
        let name_ptr = builder.ins().symbol_value(pointer_type(), name_gv);

        let block_q_val = builder.ins().iconst(cl_types::I64, block_q);
        let block_kv_val = builder.ins().iconst(cl_types::I64, block_kv);
        let shmem_val = builder.ins().iconst(cl_types::I64, shmem_bytes);
        let causal_val = builder.ins().iconst(cl_types::I64, if is_causal { 1 } else { 0 });

        let null = builder.ins().iconst(cl_types::I64, 0);

        // Extract tensor dimensions: Q is [batch, heads, seq_len, head_dim]
        let dim0 = builder.ins().iconst(cl_types::I64, 0);
        let dim1 = builder.ins().iconst(cl_types::I64, 1);
        let dim2 = builder.ins().iconst(cl_types::I64, 2);
        let dim3 = builder.ins().iconst(cl_types::I64, 3);
        let batch = self.compile_call_by_name(builder, "nsl_tensor_shape_dim", &[q_val, dim0])?;
        let heads = self.compile_call_by_name(builder, "nsl_tensor_shape_dim", &[q_val, dim1])?;
        let seq_len = self.compile_call_by_name(builder, "nsl_tensor_shape_dim", &[q_val, dim2])?;
        let head_dim = self.compile_call_by_name(builder, "nsl_tensor_shape_dim", &[q_val, dim3])?;

        // Allocate logsumexp auxiliary tensor: shape [batch, heads, seq_len]
        // Must be on the same device as Q — the GPU kernel writes to it via st.global.f32,
        // so it needs managed memory (cuMemAllocManaged) not regular CPU malloc.
        // Strategy: allocate on CPU via nsl_tensor_zeros, then move to GPU via to_device.
        // Flash attention always implies CUDA, so device=1 is safe.
        let lse_shape = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        self.compile_call_by_name(builder, "nsl_list_push", &[lse_shape, batch])?;
        self.compile_call_by_name(builder, "nsl_list_push", &[lse_shape, heads])?;
        self.compile_call_by_name(builder, "nsl_list_push", &[lse_shape, seq_len])?;
        let lse_cpu = self.compile_call_by_name(builder, "nsl_tensor_zeros", &[lse_shape])?;
        // Move to GPU so the PTX kernel can write to it (managed memory)
        let cuda_device = builder.ins().iconst(cl_types::I64, 1);
        let lse_val = self.compile_call_by_name(builder, "nsl_tensor_to_device", &[lse_cpu, cuda_device])?;

        // Call flash attention FFI — returns error code (i64), output written to out_val
        let _err = self.compile_call_by_name(builder, "nsl_flash_attention", &[
            q_val, k_val, v_val, out_val,
            lse_val,                        // logsumexp auxiliary output
            scale_val,
            batch, heads, seq_len, head_dim,
            null, null, null, null, // paged params (null for now)
            null, null,             // RoPE params (null for now)
            null, null,             // seq_ids, seq_lens (M29-ready)
            shmem_val,
            ptx_ptr, name_ptr,
            block_q_val, block_kv_val,
            causal_val,
        ])?;

        // Logsumexp is consumed by the runtime's tape recording in nsl_flash_attention.
        // The TapeOp::FlashAttention variant is recorded there with the logsumexp pointer.
        let _ = lse_val;

        // Return the output tensor (not the FFI error code)
        Ok(out_val)
    }

    pub(crate) fn compile_to_onnx(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        if args.len() < 3 {
            return Err(CodegenError::new(
                "to_onnx() requires 3 arguments: to_onnx(model, input, \"output.onnx\")",
            ));
        }

        // arg 0: model instance
        let model_val = self.compile_expr(builder, state, &args[0].value)?;

        let model_type = self.node_type(args[0].value.id).clone();
        let model_name = match &model_type {
            Type::Model { name, .. } => self.resolve_sym(*name).to_string(),
            _ => {
                return Err(CodegenError::new(
                    "to_onnx(): first argument must be a model instance",
                ));
            }
        };

        // arg 1: input tensor
        let input_val = self.compile_expr(builder, state, &args[1].value)?;

        // arg 2: output path string
        let path_val = self.compile_expr(builder, state, &args[2].value)?;

        let path_str = match &args[2].value.kind {
            ExprKind::StringLiteral(s) => s.clone(),
            _ => {
                return Err(CodegenError::new(
                    "to_onnx(): third argument must be a string literal (output path)",
                ));
            }
        };
        let path_len_val = builder
            .ins()
            .iconst(cl_types::I64, path_str.len() as i64);

        // 1. nsl_trace_start()
        self.compile_call_by_name(builder, "nsl_trace_start", &[])?;

        // 2. nsl_trace_register_input(input, "input_0")
        self.intern_string("input_0")?;
        let input_name_ptr = self.compile_string_literal(builder, "input_0")?;
        self.compile_call_by_name(
            builder,
            "nsl_trace_register_input",
            &[input_val, input_name_ptr],
        )?;

        // 3. Call model's forward method
        let mangled_forward = if let Some(methods) = self.model_methods.get(&model_name) {
            methods
                .get("forward")
                .ok_or_else(|| {
                    CodegenError::new(format!(
                        "to_onnx(): model '{model_name}' has no 'forward' method"
                    ))
                })?
                .clone()
        } else {
            format!("__nsl_model_{model_name}_forward")
        };

        let result = self.compile_call_by_name(
            builder,
            &mangled_forward,
            &[model_val, input_val],
        )?;

        // 4. nsl_trace_register_output(result, "output_0")
        self.intern_string("output_0")?;
        let output_name_ptr = self.compile_string_literal(builder, "output_0")?;
        self.compile_call_by_name(
            builder,
            "nsl_trace_register_output",
            &[result, output_name_ptr],
        )?;

        // 5. let trace = nsl_trace_stop()
        let trace = self.compile_call_by_name(builder, "nsl_trace_stop", &[])?;

        // 6. nsl_onnx_export(trace, path_ptr, path_len)
        self.compile_call_by_name(
            builder,
            "nsl_onnx_export",
            &[trace, path_val, path_len_val],
        )?;

        // Return the forward result so callers can use the output
        Ok(result)
    }

    pub(crate) fn compile_from_hf(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        if args.len() < 2 {
            return Err(CodegenError::new(
                "from_hf() requires at least 2 arguments: from_hf(\"repo/id\", model)",
            ));
        }

        // arg 0: repo_id string
        let repo_val = self.compile_expr(builder, state, &args[0].value)?;

        // Extract the string literal for length calculation
        let repo_str = match &args[0].value.kind {
            ExprKind::StringLiteral(s) => s.clone(),
            _ => {
                return Err(CodegenError::new(
                    "from_hf(): first argument must be a string literal (repo ID)",
                ));
            }
        };
        let repo_len_val =
            builder
                .ins()
                .iconst(cl_types::I64, repo_str.len() as i64);

        // arg 1: model variable
        let model_val = self.compile_expr(builder, state, &args[1].value)?;

        let model_type = self.node_type(args[1].value.id).clone();
        let model_name = match &model_type {
            Type::Model { name, .. } => self.resolve_sym(*name).to_string(),
            _ => {
                return Err(CodegenError::new(
                    "from_hf(): second argument must be a model instance",
                ));
            }
        };

        // optional arg 2: device (default 0 = CPU)
        let device_val = if args.len() > 2 {
            self.compile_expr(builder, state, &args[2].value)?
        } else {
            builder.ins().iconst(cl_types::I64, 0)
        };

        // generate compile-time param metadata
        let meta_entries = self.generate_param_metadata(&model_name, "", 0);
        let meta_count = meta_entries.len();

        if meta_count == 0 {
            return Err(CodegenError::new(format!(
                "from_hf(): model '{model_name}' has no parameter fields"
            )));
        }

        // Each ParamMeta entry is 40 bytes (5 x i64):
        //   [0]  name_ptr   (i64)
        //   [8]  offset     (i64)
        //   [16] shape_ptr  (i64) -- 0 for now
        //   [24] ndim       (i64) -- 0 for now
        //   [32] transpose  (i64) -- 0 or 1
        let entry_size: usize = 40;
        let total_bytes = meta_count * entry_size;

        // Allocate the ParamMeta array at runtime
        let alloc_size = builder.ins().iconst(cl_types::I64, total_bytes as i64);
        let meta_ptr = self.compile_call_by_name(builder, "nsl_alloc", &[alloc_size])?;

        // Populate each entry
        for (i, (field_path, byte_offset, transpose)) in meta_entries.iter().enumerate() {
            let base = (i * entry_size) as i32;

            // Intern the field path string and get a pointer to it
            let name_data_id = self.intern_string(field_path)?;
            let gv = self.module.declare_data_in_func(name_data_id, builder.func);
            let name_ptr = builder.ins().symbol_value(cl_types::I64, gv);

            // offset 0: name_ptr
            builder.ins().store(
                MemFlags::trusted(),
                name_ptr,
                meta_ptr,
                cranelift_codegen::ir::immediates::Offset32::new(base),
            );

            // offset 8: byte_offset into model struct
            let offset_val = builder.ins().iconst(cl_types::I64, *byte_offset as i64);
            builder.ins().store(
                MemFlags::trusted(),
                offset_val,
                meta_ptr,
                cranelift_codegen::ir::immediates::Offset32::new(base + 8),
            );

            // offset 16: shape_ptr (0 for now)
            let zero = builder.ins().iconst(cl_types::I64, 0);
            builder.ins().store(
                MemFlags::trusted(),
                zero,
                meta_ptr,
                cranelift_codegen::ir::immediates::Offset32::new(base + 16),
            );

            // offset 24: ndim (0 for now)
            let zero2 = builder.ins().iconst(cl_types::I64, 0);
            builder.ins().store(
                MemFlags::trusted(),
                zero2,
                meta_ptr,
                cranelift_codegen::ir::immediates::Offset32::new(base + 24),
            );

            // offset 32: transpose flag
            let transpose_val =
                builder
                    .ins()
                    .iconst(cl_types::I64, if *transpose { 1 } else { 0 });
            builder.ins().store(
                MemFlags::trusted(),
                transpose_val,
                meta_ptr,
                cranelift_codegen::ir::immediates::Offset32::new(base + 32),
            );
        }

        // call nsl_hf_load
        let meta_len_val = builder.ins().iconst(cl_types::I64, meta_count as i64);

        // nsl_hf_load(model_ptr, meta_ptr, meta_len, repo_id_ptr, repo_id_len, device)
        self.compile_call_by_name(
            builder,
            "nsl_hf_load",
            &[model_val, meta_ptr, meta_len_val, repo_val, repo_len_val, device_val],
        )?;

        // Free the temporary ParamMeta array
        self.compile_call_by_name(builder, "nsl_free", &[meta_ptr])?;

        // Return the model pointer so callers can chain: let m = from_hf("repo", m)
        Ok(model_val)
    }

    /// Walk a model's struct layout tree and collect a flat list of
    /// `(field_path, byte_offset, needs_transpose)` entries for every leaf
    /// tensor field.
    pub(crate) fn generate_param_metadata(
        &self,
        model_name: &str,
        prefix: &str,
        base_offset: usize,
    ) -> Vec<(String, usize, bool)> {
        let mut entries = Vec::new();

        let layout = match self.struct_layouts.get(model_name) {
            Some(l) => l.clone(),
            None => return entries,
        };
        let field_types = self
            .model_field_types
            .get(model_name)
            .cloned()
            .unwrap_or_default();

        for field in &layout.fields {
            let field_path = if prefix.is_empty() {
                field.name.clone()
            } else {
                format!("{}.{}", prefix, field.name)
            };
            let abs_offset = base_offset + field.offset;

            if let Some(type_marker) = field_types.get(&field.name) {
                if type_marker.starts_with('[') {
                    // Fixed array like "[TransformerBlock;6]"
                    // Parse element type and count from "[ElemType;N]"
                    let inner = &type_marker[1..type_marker.len() - 1]; // strip [ ]
                    if let Some((elem_type, count_str)) = inner.rsplit_once(';') {
                        let count: usize = count_str.trim().parse().unwrap_or(0);
                        // Determine element size from the element type's struct layout
                        let elem_size = self
                            .struct_layouts
                            .get(elem_type)
                            .map(|l| l.total_size)
                            .unwrap_or(8);
                        for i in 0..count {
                            let indexed_path = format!("{}[{}]", field_path, i);
                            let indexed_offset = abs_offset + i * elem_size;
                            let sub =
                                self.generate_param_metadata(elem_type, &indexed_path, indexed_offset);
                            entries.extend(sub);
                        }
                    }
                } else {
                    // Nested model (e.g., "Linear", "Attention")
                    let sub =
                        self.generate_param_metadata(type_marker, &field_path, abs_offset);
                    entries.extend(sub);
                }
            } else {
                // Leaf tensor field -- determine transpose flag
                let is_linear_like = {
                    let names: Vec<&str> = layout.fields.iter().map(|f| f.name.as_str()).collect();
                    (names.contains(&"w") && names.contains(&"b"))
                        || (names.contains(&"weight") && names.contains(&"bias"))
                };
                let needs_transpose =
                    is_linear_like && (field.name == "weight" || field.name == "w");
                entries.push((field_path, abs_offset, needs_transpose));
            }
        }

        entries
    }

    /// Attempt auto-fusion on an expression. Called from compile_expr() before
    /// normal dispatch for BinaryOp and Call nodes.
    /// Returns Ok(Some(val)) if fused, Ok(None) to fall through to normal compile.
    pub(crate) fn try_auto_fuse(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        expr: &Expr,
    ) -> Result<Option<Value>, CodegenError> {
        if state.in_fuse_bypass || self.disable_fusion {
            return Ok(None);
        }
        // Don't fuse during tape recording — autodiff needs individual op recordings
        if state.in_tape_region {
            return Ok(None);
        }

        let interner = self.interner;
        let resolve = |sym: Symbol| -> Option<String> {
            interner.resolve(sym.0).map(|s| s.to_string())
        };

        if let Some((ops, inputs)) = crate::fusion::analyze_fusible_chain(expr, &resolve) {
            if ops.len() >= 2 && inputs.len() >= 1 && inputs.len() <= 2 {
                // Convert op names to runtime op codes
                let op_codes: Vec<i64> = ops.iter().filter_map(|op| match op.as_str() {
                    "add" => Some(0), // FUSED_OP_ADD
                    "mul" => Some(1),
                    "sub" => Some(2),
                    "div" => Some(3),
                    "relu" => Some(4),
                    "sigmoid" => Some(5),
                    "tanh" => Some(6),
                    "neg" => Some(7),
                    "exp" => Some(8),
                    "log" => Some(9),
                    "sqrt" => Some(10),
                    "abs" => Some(11),
                    "gelu" => Some(12),
                    "silu" => Some(13),
                    _ => None, // Unknown op — skip fusion
                }).collect();
                if op_codes.len() != ops.len() {
                    return Ok(None); // Some op couldn't be fused — fall back
                }

                // Check if all operand types are tensors
                let all_tensors = inputs.iter().all(|inp| {
                    let ty = self.node_type(inp.id).clone();
                    ty.is_tensor() || ty.is_indeterminate()
                });
                if !all_tensors {
                    return Ok(None);
                }

                // Compile inputs (bypass fusion to avoid infinite recursion)
                state.in_fuse_bypass = true;
                let compiled_inputs: Vec<Value> = inputs.iter()
                    .map(|inp| self.compile_expr(builder, state, inp))
                    .collect::<Result<Vec<_>, _>>()?;
                state.in_fuse_bypass = false;

                // Build op-codes list
                let ops_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
                for &code in &op_codes {
                    let code_val = builder.ins().iconst(cl_types::I64, code);
                    self.compile_call_by_name(builder, "nsl_list_push", &[ops_list, code_val])?;
                }
                let num_ops = builder.ins().iconst(cl_types::I64, op_codes.len() as i64);

                let result = if compiled_inputs.len() == 2 {
                    // Binary + unary chain: nsl_fused_elementwise_2(a, b, ops, num_ops)
                    self.compile_call_by_name(
                        builder, "nsl_fused_elementwise_2",
                        &[compiled_inputs[0], compiled_inputs[1], ops_list, num_ops],
                    )?
                } else {
                    // Single input + unary chain: nsl_fused_elementwise_1(a, ops, num_ops)
                    self.compile_call_by_name(
                        builder, "nsl_fused_elementwise_1",
                        &[compiled_inputs[0], ops_list, num_ops],
                    )?
                };

                // Free the ops list (small allocation, not a tensor)
                self.compile_call_by_name(builder, "nsl_list_free", &[ops_list])?;

                return Ok(Some(result));
            }
        }

        Ok(None)
    }

    /// Compile `model_save(model, "path.nslm")` — serialize model parameters to disk.
    ///
    /// Codegen emits:
    ///   1. Build NslList of parameter name C-string pointers
    ///   2. Build NslList of parameter tensor pointers (from struct fields)
    ///   3. Call nsl_model_save(path_ptr, path_len, names_list, tensors_list)
    pub(crate) fn compile_model_save(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        if args.len() != 2 {
            return Err(CodegenError::new(
                "model_save() takes exactly 2 arguments (model, path)",
            ));
        }

        // arg 0: model instance
        let model_val = self.compile_expr(builder, state, &args[0].value)?;
        let model_type = self.node_type(args[0].value.id).clone();
        let model_name = match &model_type {
            Type::Model { name, .. } => self.resolve_sym(*name).to_string(),
            _ => {
                return Err(CodegenError::new(
                    "model_save(): first argument must be a model instance",
                ));
            }
        };

        // arg 1: path string
        let path_val = self.compile_expr(builder, state, &args[1].value)?;
        let path_str = match &args[1].value.kind {
            ExprKind::StringLiteral(s) => s.clone(),
            _ => {
                return Err(CodegenError::new(
                    "model_save(): second argument must be a string literal (file path)",
                ));
            }
        };
        let path_len = builder
            .ins()
            .iconst(cl_types::I64, path_str.len() as i64);

        // Look up model struct layout
        let layout = self.struct_layouts.get(&model_name).cloned().ok_or_else(|| {
            CodegenError::new(format!(
                "model_save(): no struct layout found for model '{}'",
                model_name
            ))
        })?;

        // Build NslList of parameter name pointers (null-terminated C strings)
        let names_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        for field in &layout.fields {
            let name_data_id = self.intern_string(&field.name)?;
            let gv = self.module.declare_data_in_func(name_data_id, builder.func);
            let name_ptr = builder.ins().symbol_value(cl_types::I64, gv);
            self.compile_call_by_name(builder, "nsl_list_push", &[names_list, name_ptr])?;
        }

        // Build NslList of parameter tensor pointers (load from model struct fields)
        let tensors_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        for field in &layout.fields {
            let field_val = builder.ins().load(
                field.cl_type,
                MemFlags::trusted(),
                model_val,
                field.offset as i32,
            );
            self.compile_call_by_name(builder, "nsl_list_push", &[tensors_list, field_val])?;
        }

        // Call nsl_model_save(path_ptr, path_len, names_list, tensors_list)
        self.compile_call_by_name(
            builder,
            "nsl_model_save",
            &[path_val, path_len, names_list, tensors_list],
        )
    }

    /// Compile `model_load(model, "path.nslm")` — load model parameters from disk.
    ///
    /// Codegen emits:
    ///   1. Build NslList of parameter tensor pointers (from struct fields)
    ///   2. Call nsl_model_load(path_ptr, path_len, tensors_list)
    pub(crate) fn compile_model_load(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        if args.len() != 2 {
            return Err(CodegenError::new(
                "model_load() takes exactly 2 arguments (model, path)",
            ));
        }

        // arg 0: model instance
        let model_val = self.compile_expr(builder, state, &args[0].value)?;
        let model_type = self.node_type(args[0].value.id).clone();
        let model_name = match &model_type {
            Type::Model { name, .. } => self.resolve_sym(*name).to_string(),
            _ => {
                return Err(CodegenError::new(
                    "model_load(): first argument must be a model instance",
                ));
            }
        };

        // arg 1: path string
        let path_val = self.compile_expr(builder, state, &args[1].value)?;
        let path_str = match &args[1].value.kind {
            ExprKind::StringLiteral(s) => s.clone(),
            _ => {
                return Err(CodegenError::new(
                    "model_load(): second argument must be a string literal (file path)",
                ));
            }
        };
        let path_len = builder
            .ins()
            .iconst(cl_types::I64, path_str.len() as i64);

        // Look up model struct layout
        let layout = self.struct_layouts.get(&model_name).cloned().ok_or_else(|| {
            CodegenError::new(format!(
                "model_load(): no struct layout found for model '{}'",
                model_name
            ))
        })?;

        // Build NslList of parameter tensor pointers (load from model struct fields)
        let tensors_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        for field in &layout.fields {
            let field_val = builder.ins().load(
                field.cl_type,
                MemFlags::trusted(),
                model_val,
                field.offset as i32,
            );
            self.compile_call_by_name(builder, "nsl_list_push", &[tensors_list, field_val])?;
        }

        // Call nsl_model_load(path_ptr, path_len, tensors_list)
        self.compile_call_by_name(
            builder,
            "nsl_model_load",
            &[path_val, path_len, tensors_list],
        )
    }
}
