use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;

use nsl_ast::expr::{Expr, ExprKind};
use nsl_semantic::types::Type;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::types::{is_float_type, is_int_type, nsl_type_to_cl, pointer_type};

fn tensor_unary_runtime_alias(func_name: &str, arity: usize) -> Option<&'static str> {
    if arity != 1 {
        return None;
    }
    match func_name {
        "relu" => Some("nsl_tensor_relu"),
        "sigmoid" => Some("nsl_tensor_sigmoid"),
        "tanh" => Some("nsl_tensor_tanh"),
        "exp" => Some("nsl_tensor_exp"),
        "log" => Some("nsl_tensor_log"),
        "sqrt" => Some("nsl_tensor_sqrt"),
        "abs" => Some("nsl_tensor_abs"),
        "gelu" => Some("nsl_tensor_gelu"),
        "silu" => Some("nsl_tensor_silu"),
        _ => None,
    }
}

fn static_dim_value(dim: &nsl_semantic::types::Dim) -> Option<i64> {
    match dim {
        nsl_semantic::types::Dim::Concrete(value) => Some(*value),
        nsl_semantic::types::Dim::Named { size, .. } => static_dim_value(size),
        nsl_semantic::types::Dim::Computed(expr) => expr.as_lit(),
        nsl_semantic::types::Dim::Symbolic(_)
        | nsl_semantic::types::Dim::Bounded { .. }
        | nsl_semantic::types::Dim::Wildcard => None,
    }
}

fn static_tensor_numel(ty: &Type) -> Option<i64> {
    let (shape, _, _) = ty.as_tensor_parts()?;
    if shape.dims.is_empty() {
        return Some(1);
    }

    let mut total = 1_i64;
    for dim in &shape.dims {
        total = total.checked_mul(static_dim_value(dim)?)?;
    }
    Some(total)
}

// CPDT Part III v2.19 introduced the per-model decorator-config
// resolver to fix multi-model `@moe` composition; v2.20 generalized
// the resolver to `crate::moe::resolve_decorator_config_for_call_site`
// and applied it to `@context_parallel` + `@speculative` too. See
// that function's rustdoc for the resolution rules and the agent/
// model name-collision footgun.

impl Compiler<'_> {
    pub(crate) fn compile_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        callee: &Expr,
        args: &[nsl_ast::expr::Arg],
        call_expr: &Expr,
    ) -> Result<Value, CodegenError> {
        // Intercept method calls (e.g., s.upper()) before extracting func_name
        if let ExprKind::MemberAccess { object, member } = &callee.kind {
            let member_name = self.resolve_sym(*member).to_string();
            let obj_type = self.node_type(object.id).clone();
            // Module alias call: math.clamp(...) -> call "clamp"
            if matches!(obj_type, Type::Module { .. }) {
                let mut arg_vals = Vec::new();
                for arg in args {
                    arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
                }
                let result = self.compile_call_by_name(builder, &member_name, &arg_vals)?;
                self.register_ffi_result_ownership(
                    builder,
                    state,
                    &member_name,
                    result,
                    &arg_vals,
                );
                return Ok(result);
            }
            if matches!(obj_type, Type::Str) {
                return self.compile_str_method_call(builder, state, object, &member_name, args);
            }
            if matches!(obj_type, Type::List(_)) {
                return self.compile_list_method_call(builder, state, object, &member_name, args);
            }
            if obj_type.is_tensor() {
                return self.compile_tensor_method_call(builder, state, object, &member_name, args);
            }
            if let Type::Model { name, .. } = &obj_type {
                let model_name = self.resolve_sym(*name).to_string();
                // m.to(device): transfer all tensor fields, recurse into sub-models
                if member_name == "to" && args.len() == 1 {
                    let model_val = self.compile_expr(builder, state, object)?;
                    let device_val = self.compile_expr(builder, state, &args[0].value)?;
                    self.emit_model_to_device(builder, &model_name, model_val, device_val)?;
                    return Ok(model_val);
                }
                return self.compile_model_method_call(
                    builder,
                    state,
                    object,
                    &model_name,
                    &member_name,
                    args,
                );
            }
            // M56 Task 17: agent method calls — `agent_var.method(args)`.
            // Spec §4.1: mangling is `__nsl_agent_{AgentName}_{MethodName}`.
            if let Type::Agent { name, .. } = &obj_type {
                let agent_name = self.resolve_sym(*name).to_string();
                let mangled = format!("__nsl_agent_{}_{}", agent_name, member_name);
                let self_val = self.compile_expr(builder, state, object)?;
                let mut arg_vals = vec![self_val];
                for arg in args {
                    arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
                }
                return self.compile_call_by_name(builder, &mangled, &arg_vals);
            }
            // Fallback for model array loop variables (type is Unknown but var was bound from model array).
            // M56 Task 18: also handles `Type::Error` (semantic pass assigns Error to undefined
            // bindings like `drafter` in @pipeline_agent fn bodies where type checking doesn't
            // know about the synthesised agent vars — they are wired at codegen time).
            if matches!(obj_type, Type::Unknown | Type::Error) {
                if let ExprKind::Ident(obj_sym) = &object.kind {
                    if let Some(model_name) = self.models.model_var_types.get(obj_sym).cloned() {
                        return self.compile_model_method_call(
                            builder,
                            state,
                            object,
                            &model_name,
                            &member_name,
                            args,
                        );
                    }
                    // M56 Task 18 (spec §5.2, Option A): agent-var synthesised at
                    // @pipeline_agent fn entry — dispatch to __nsl_agent_{Name}_{method}.
                    if let Some(agent_name) = self.models.agent_var_types.get(obj_sym).cloned() {
                        let mangled = format!("__nsl_agent_{}_{}", agent_name, member_name);
                        let self_val = self.compile_expr(builder, state, object)?;
                        // M56 Task 19: compile user args and apply @auto_device_transfer
                        // transfers where the target method annotates tensor params with
                        // an explicit device.  `nsl_tensor_to_device` is a no-op when the
                        // tensor is already on the correct device, so the transfer is safe
                        // to insert unconditionally without a static source-device check.
                        let device_transfers = self
                            .models
                            .agent_auto_device_params
                            .get(&(agent_name.clone(), member_name.clone()))
                            .cloned()
                            .unwrap_or_default();
                        let mut arg_vals = vec![self_val];
                        for (call_arg_idx, arg) in args.iter().enumerate() {
                            let compiled = self.compile_expr(builder, state, &arg.value)?;
                            // Check whether this argument position has a device-transfer
                            // requirement AND the compiled value is a tensor pointer (I64).
                            // The transfer is only safe when the value is pointer-sized;
                            // non-pointer values (e.g. i32 scalars) are passed unchanged.
                            // `nsl_tensor_to_device` is a no-op when already on the target
                            // device, so insertion is safe-by-default for I64 tensor values.
                            let compiled_ty = builder.func.dfg.value_type(compiled);
                            if compiled_ty == cl_types::I64 {
                                if let Some(&(_, target_device)) =
                                    device_transfers.iter().find(|(idx, _)| *idx == call_arg_idx)
                                {
                                    let device_val =
                                        builder.ins().iconst(cl_types::I64, target_device);
                                    let transferred = self.compile_call_by_name(
                                        builder,
                                        "nsl_tensor_to_device",
                                        &[compiled, device_val],
                                    )?;
                                    arg_vals.push(transferred);
                                    continue;
                                }
                            }
                            arg_vals.push(compiled);
                        }
                        return self.compile_call_by_name(builder, &mangled, &arg_vals);
                    }
                }
                // Only warn for Unknown — Error is expected for @pipeline_agent bodies
                // where type inference doesn't cover synthesised agent vars.
                if matches!(obj_type, Type::Unknown) {
                    eprintln!(
                        "[nsl-codegen] warning: method '.{member_name}()' called on Unknown-typed \
                         object — defaulting to tensor dispatch. This may indicate a type inference gap."
                    );
                }
                return self.compile_tensor_method_call(builder, state, object, &member_name, args);
            }
        }

        let func_name = match self.expr_as_func_name(callee) {
            Some(name) => name,
            None => {
                // Try indirect call (e.g., variable holding a lambda)
                return self.compile_indirect_call(builder, state, callee, args);
            }
        };

        // M39b: matmul rewrite dispatch — VmapTransformer records (NodeId → target_name)
        // for matmul call sites that need batched semantics.  "batched_matmul" and
        // "batched_matmul_right" are logical names; both currently lower to
        // nsl_tensor_matmul which handles broadcast/batched shapes natively.
        // Future milestones (M39c+) will wire specialised batched kernels here.
        if let Some(rewrite_target) = self.vmap.matmul_rewrites.get(&call_expr.id).cloned() {
            let rt_name = match rewrite_target.as_str() {
                "batched_matmul" | "batched_matmul_right" => "nsl_tensor_matmul",
                other => other, // forward any future named variants unchanged
            };
            if args.len() < 2 {
                return Err(CodegenError::new(
                    "matmul rewrite: expected at least 2 arguments",
                ));
            }
            let lhs = self.compile_expr(builder, state, &args[0].value)?;
            let rhs = self.compile_expr(builder, state, &args[1].value)?;
            // ELTLS (FBIP-3): nsl_tensor_matmul takes a flags byte.
            let flags_zero = builder.ins().iconst(cl_types::I8, 0);
            return self.compile_traced_call(builder, rt_name, &[lhs, rhs, flags_zero]);
        }

        // Check if this is a kernel call (compiled GPU kernel)
        if let Some((ptx_data_id, name_data_id)) =
            self.kernels.kernel_ptx_data.get(&func_name).cloned()
        {
            return self.compile_kernel_call(
                builder,
                state,
                &func_name.clone(),
                args,
                ptx_data_id,
                name_data_id,
                call_expr.id,
                call_expr.span,
            );
        }

        // @fuse decorated function: emit fused elementwise kernel launch
        if let Some((op_chain, num_inputs)) = self.fusion.fused_fns.get(&func_name).cloned() {
            if !self.fusion.disabled && !state.flags.in_tape_region {
                let op_refs: Vec<&str> = op_chain.iter().map(|s| s.as_str()).collect();
                if let Some(kernel) = crate::fusion::try_synthesize_fused(&op_refs, num_inputs) {
                    // Compile input arguments
                    let mut compiled_args = Vec::new();
                    for arg in args {
                        compiled_args.push(self.compile_expr(builder, state, &arg.value)?);
                    }

                    // Build op-codes list for nsl_fused_elementwise_N
                    let op_codes: Vec<i64> = op_chain
                        .iter()
                        .filter_map(|op| match op.as_str() {
                            "add" => Some(0),
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
                            _ => None,
                        })
                        .collect();

                    if op_codes.len() == op_chain.len() && !compiled_args.is_empty() {
                        let ops_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
                        for &code in &op_codes {
                            let code_val = builder.ins().iconst(cl_types::I64, code);
                            self.compile_call_by_name(
                                builder,
                                "nsl_list_push",
                                &[ops_list, code_val],
                            )?;
                        }
                        let num_ops = builder.ins().iconst(cl_types::I64, op_codes.len() as i64);

                        let result = if compiled_args.len() >= 2 {
                            self.compile_call_by_name(
                                builder,
                                "nsl_fused_elementwise_2",
                                &[compiled_args[0], compiled_args[1], ops_list, num_ops],
                            )?
                        } else {
                            self.compile_call_by_name(
                                builder,
                                "nsl_fused_elementwise_1",
                                &[compiled_args[0], ops_list, num_ops],
                            )?
                        };
                        self.compile_call_by_name(builder, "nsl_list_free", &[ops_list])?;

                        let _ = kernel; // PTX is synthesized but dispatch uses runtime fused ops
                        return Ok(result);
                    }
                }
            }
            // Fall through to normal call if fusion disabled or synthesis failed
        }

        if func_name == "print" {
            return self.compile_print_call(builder, state, args);
        }
        if func_name == "len" {
            if let Some(arg) = args.first() {
                let val = self.compile_expr(builder, state, &arg.value)?;
                let arg_type = self.node_type(arg.value.id).clone();
                let fn_name = if matches!(arg_type, Type::Dict(_, _)) {
                    "nsl_dict_len"
                } else {
                    "nsl_list_len"
                };
                let fid = self.registry.runtime_fns[fn_name].0;
                let fref = self.module.declare_func_in_func(fid, builder.func);
                let call = builder.ins().call(fref, &[val]);
                return Ok(builder.inst_results(call)[0]);
            }
        }
        if func_name == "range" {
            return self.compile_range_call(builder, state, args);
        }
        // Benchmarking intrinsics: clock(), alloc_reset(), alloc_count(), alloc_bytes()
        if func_name == "clock" && args.is_empty() {
            return self.compile_call_by_name(builder, "nsl_clock", &[]);
        }
        if func_name == "alloc_reset" && args.is_empty() {
            return self.compile_call_by_name(builder, "nsl_alloc_reset", &[]);
        }
        if func_name == "alloc_count" && args.is_empty() {
            return self.compile_call_by_name(builder, "nsl_alloc_count", &[]);
        }
        if func_name == "alloc_bytes" && args.is_empty() {
            return self.compile_call_by_name(builder, "nsl_alloc_bytes", &[]);
        }
        if matches!(func_name.as_str(), "int" | "float" | "str" | "bool") {
            return self.compile_type_conversion(builder, state, &func_name, args);
        }
        // Higher-order functions: map(fn, list), filter(fn, list)
        if matches!(func_name.as_str(), "map" | "filter") {
            return self.compile_higher_order_call(builder, state, &func_name, args);
        }
        // Paged KV cache builtins (M25) -- dispatch nsl_kv_cache_* and nsl_profiler_*
        if matches!(
            func_name.as_str(),
            "kv_cache_init"
                | "kv_cache_init_gpu"
                | "kv_cache_alloc_seq"
                | "kv_cache_append"
                | "kv_cache_k_ptr"
                | "kv_cache_v_ptr"
                | "kv_cache_free_seq"
                | "kv_cache_seq_len"
                | "kv_cache_seq_blocks"
                | "kv_cache_seq_num_blocks"
                | "kv_cache_utilization"
                | "kv_cache_destroy"
                | "profiler_start"
                | "profiler_stop"
                | "profiler_peak"
                | "kernel_profiler_start"
                | "kernel_profiler_stop"
        ) {
            let mut arg_vals = Vec::new();
            for arg in args {
                arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
            }
            // M42: For kv_cache_init variants, append KV compression params.
            // Use the first policy found across all models (v1: single-policy heuristic).
            if func_name == "kv_cache_init" || func_name == "kv_cache_init_gpu" {
                let (compress_scheme, compress_window, compress_sinks) = self
                    .features
                    .kv_compress_policies
                    .values()
                    .next()
                    .and_then(|policies| policies.first())
                    .map(|p| (p.scheme as i64, p.window as i64, p.sinks as i64))
                    .unwrap_or((0, 0, 0));
                if compress_scheme > 0 {
                    eprintln!(
                        "[nsl] KV cache compression active: scheme={}, window={}, sinks={}",
                        compress_scheme, compress_window, compress_sinks
                    );
                }
                let scheme_val = builder.ins().iconst(cl_types::I64, compress_scheme);
                let window_val = builder.ins().iconst(cl_types::I64, compress_window);
                let sinks_val = builder.ins().iconst(cl_types::I64, compress_sinks);
                arg_vals.push(scheme_val);
                arg_vals.push(window_val);
                arg_vals.push(sinks_val);
            }
            let rt_name = format!("nsl_{func_name}");
            return self.compile_call_by_name(builder, &rt_name, &arg_vals);
        }
        // Direct runtime builtins: enumerate, zip, sorted, reversed
        if matches!(
            func_name.as_str(),
            "enumerate" | "zip" | "sorted" | "reversed"
        ) {
            let mut arg_vals = Vec::new();
            for arg in args {
                arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
            }
            let rt_name = format!("nsl_{func_name}");
            return self.compile_call_by_name(builder, &rt_name, &arg_vals);
        }

        // Math builtins: sqrt, log, exp, sin, cos (scalar path: f64 in, f64 out)
        // For tensor arguments, fall through to the M14 tensor dispatch below.
        if matches!(func_name.as_str(), "sqrt" | "log" | "exp" | "sin" | "cos") {
            if args.len() != 1 {
                return Err(CodegenError::new(format!(
                    "{func_name}() takes exactly 1 argument"
                )));
            }
            let arg_type = self.node_type(args[0].value.id).clone();
            if is_float_type(&arg_type) || is_int_type(&arg_type) {
                let val = self.compile_expr(builder, state, &args[0].value)?;
                let float_val = if is_float_type(&arg_type) {
                    if matches!(arg_type, Type::F32) {
                        builder.ins().fpromote(cl_types::F64, val)
                    } else {
                        val
                    }
                } else {
                    let widened = match &arg_type {
                        Type::Int32 | Type::Int16 | Type::Int8 => {
                            builder.ins().sextend(cl_types::I64, val)
                        }
                        _ => val,
                    };
                    builder.ins().fcvt_from_sint(cl_types::F64, widened)
                };
                let rt_name = format!("nsl_{func_name}");
                return self.compile_call_by_name(builder, &rt_name, &[float_val]);
            }
            // else: tensor type -- fall through to M14 tensor dispatch
        }
        // abs: dispatch to int or float variant (scalar only)
        // For tensor arguments, fall through to the M14 tensor dispatch below.
        if func_name == "abs" {
            if args.len() != 1 {
                return Err(CodegenError::new("abs() takes exactly 1 argument"));
            }
            let arg_type = self.node_type(args[0].value.id).clone();
            if is_float_type(&arg_type) || is_int_type(&arg_type) {
                let val = self.compile_expr(builder, state, &args[0].value)?;
                let rt_name = if is_float_type(&arg_type) {
                    "nsl_abs_float"
                } else {
                    "nsl_abs_int"
                };
                return self.compile_call_by_name(builder, rt_name, &[val]);
            }
            // else: tensor type -- fall through to M14 tensor dispatch
        }
        // floor: scalar f64 dispatch
        if func_name == "floor" {
            if args.len() != 1 {
                return Err(CodegenError::new("floor() takes exactly 1 argument"));
            }
            let val = self.compile_nested_expr(builder, state, &args[0].value)?;
            let arg_type = self.node_type(args[0].value.id).clone();
            let float_val = if is_float_type(&arg_type) {
                if matches!(arg_type, Type::F32) {
                    builder.ins().fpromote(cl_types::F64, val)
                } else {
                    val
                }
            } else {
                let widened = match &arg_type {
                    Type::Int32 | Type::Int16 | Type::Int8 => {
                        builder.ins().sextend(cl_types::I64, val)
                    }
                    _ => val,
                };
                builder.ins().fcvt_from_sint(cl_types::F64, widened)
            };
            return self.compile_call_by_name(builder, "nsl_floor", &[float_val]);
        }
        // min/max: dispatch to int or float variant
        if matches!(func_name.as_str(), "min" | "max") {
            if args.len() != 2 {
                return Err(CodegenError::new(format!(
                    "{func_name}() takes exactly 2 arguments"
                )));
            }
            let a = self.compile_expr(builder, state, &args[0].value)?;
            let b = self.compile_expr(builder, state, &args[1].value)?;
            let a_type = self.node_type(args[0].value.id).clone();
            let b_type = self.node_type(args[1].value.id).clone();
            let is_float = is_float_type(&a_type) || is_float_type(&b_type);
            let rt_name = format!(
                "nsl_{}_{}",
                func_name,
                if is_float { "float" } else { "int" }
            );
            if is_float {
                let a_f = if matches!(a_type, Type::F32) {
                    builder.ins().fpromote(cl_types::F64, a)
                } else if is_float_type(&a_type) {
                    a
                } else {
                    let w = match &a_type {
                        Type::Int32 | Type::Int16 | Type::Int8 => {
                            builder.ins().sextend(cl_types::I64, a)
                        }
                        _ => a,
                    };
                    builder.ins().fcvt_from_sint(cl_types::F64, w)
                };
                let b_f = if matches!(b_type, Type::F32) {
                    builder.ins().fpromote(cl_types::F64, b)
                } else if is_float_type(&b_type) {
                    b
                } else {
                    let w = match &b_type {
                        Type::Int32 | Type::Int16 | Type::Int8 => {
                            builder.ins().sextend(cl_types::I64, b)
                        }
                        _ => b,
                    };
                    builder.ins().fcvt_from_sint(cl_types::F64, w)
                };
                return self.compile_call_by_name(builder, &rt_name, &[a_f, b_f]);
            }
            return self.compile_call_by_name(builder, &rt_name, &[a, b]);
        }
        // Assert
        if func_name == "assert" {
            if args.is_empty() {
                return Err(CodegenError::new("assert() requires at least 1 argument"));
            }
            let cond_val = self.compile_expr(builder, state, &args[0].value)?;
            let cond_type = self.node_type(args[0].value.id).clone();
            let cond_i8 = if crate::types::is_float_type(&cond_type) {
                let zero = builder.ins().f64const(0.0);
                builder.ins().fcmp(FloatCC::NotEqual, cond_val, zero)
            } else {
                builder.ins().icmp_imm(IntCC::NotEqual, cond_val, 0)
            };
            let msg = if args.len() > 1 {
                self.compile_expr(builder, state, &args[1].value)?
            } else {
                self.intern_string("assertion failed")?;
                self.compile_string_literal(builder, "assertion failed")?
            };
            let fid = self.registry.runtime_fns["nsl_assert"].0;
            let fref = self.module.declare_func_in_func(fid, builder.func);
            builder.ins().call(fref, &[cond_i8, msg]);
            return Ok(builder.ins().iconst(cl_types::I64, 0));
        }
        // assert_eq(a, b)
        if func_name == "assert_eq" {
            if args.len() != 2 {
                return Err(CodegenError::new("assert_eq() takes exactly 2 arguments"));
            }
            let a = self.compile_expr(builder, state, &args[0].value)?;
            let b = self.compile_expr(builder, state, &args[1].value)?;
            let a_type = self.node_type(args[0].value.id).clone();
            let b_type = self.node_type(args[1].value.id).clone();
            let is_float = is_float_type(&a_type) || is_float_type(&b_type);

            let msg_str = "assert_eq";
            self.intern_string(msg_str)?;
            let msg_ptr = self.compile_string_literal(builder, msg_str)?;
            let msg_len = builder.ins().iconst(cl_types::I64, msg_str.len() as i64);

            if is_float {
                let a_f = if is_float_type(&a_type) {
                    a
                } else {
                    builder.ins().fcvt_from_sint(cl_types::F64, a)
                };
                let b_f = if is_float_type(&b_type) {
                    b
                } else {
                    builder.ins().fcvt_from_sint(cl_types::F64, b)
                };
                return self.compile_call_by_name(
                    builder,
                    "nsl_assert_eq_float",
                    &[a_f, b_f, msg_ptr, msg_len],
                );
            }
            return self.compile_call_by_name(
                builder,
                "nsl_assert_eq_int",
                &[a, b, msg_ptr, msg_len],
            );
        }
        // assert_close(a, b, rtol, atol)
        if func_name == "assert_close" {
            if args.len() != 4 {
                return Err(CodegenError::new(
                    "assert_close() takes exactly 4 arguments (tensor, tensor, rtol, atol)",
                ));
            }
            let a = self.compile_expr(builder, state, &args[0].value)?;
            let b = self.compile_expr(builder, state, &args[1].value)?;
            let rtol = self.compile_expr(builder, state, &args[2].value)?;
            let atol = self.compile_expr(builder, state, &args[3].value)?;

            // Coerce rtol/atol to f64 if they are int
            let rtol_type = self.node_type(args[2].value.id).clone();
            let atol_type = self.node_type(args[3].value.id).clone();
            let rtol_f = if is_float_type(&rtol_type) {
                rtol
            } else {
                builder.ins().fcvt_from_sint(cl_types::F64, rtol)
            };
            let atol_f = if is_float_type(&atol_type) {
                atol
            } else {
                builder.ins().fcvt_from_sint(cl_types::F64, atol)
            };

            let msg_str = "assert_close";
            self.intern_string(msg_str)?;
            let msg_ptr = self.compile_string_literal(builder, msg_str)?;
            let msg_len = builder.ins().iconst(cl_types::I64, msg_str.len() as i64);

            return self.compile_call_by_name(
                builder,
                "nsl_assert_close",
                &[a, b, rtol_f, atol_f, msg_ptr, msg_len],
            );
        }
        // Exit
        if func_name == "exit" {
            if args.len() != 1 {
                return Err(CodegenError::new("exit() takes exactly 1 argument"));
            }
            let code_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_exit", &[code_val]);
        }
        // Stdin I/O
        if func_name == "read_line" && !self.registry.functions.contains_key(&func_name) {
            if !args.is_empty() {
                return Err(CodegenError::new("read_line() takes no arguments"));
            }
            return self.compile_call_by_name(builder, "nsl_read_line", &[]);
        }
        // File I/O
        if matches!(
            func_name.as_str(),
            "read_file" | "write_file" | "append_file" | "file_exists"
        ) {
            let mut arg_vals = Vec::new();
            for arg in args {
                arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
            }
            let rt_name = format!("nsl_{func_name}");
            return self.compile_call_by_name(builder, &rt_name, &arg_vals);
        }
        // Command-line args
        if func_name == "args" {
            return self.compile_call_by_name(builder, "nsl_args", &[]);
        }
        // Training mode
        if func_name == "is_training" {
            return self.compile_call_by_name(builder, "nsl_is_training", &[]);
        }
        if func_name == "set_training_mode" {
            if args.len() != 1 {
                return Err(CodegenError::new(
                    "set_training_mode() takes exactly 1 argument (bool)",
                ));
            }
            let mode_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_set_training_mode", &[mode_val]);
        }
        // Tensor creation builtins
        if matches!(func_name.as_str(), "zeros" | "ones" | "rand" | "randn") {
            if args.len() != 1 {
                return Err(CodegenError::new(format!(
                    "{func_name}() takes exactly 1 argument (shape list)"
                )));
            }
            let shape_val = self.compile_expr(builder, state, &args[0].value)?;
            let rt_name = format!("nsl_tensor_{func_name}");
            return self.compile_call_by_name(builder, &rt_name, &[shape_val]);
        }
        if func_name == "full" {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "full() takes exactly 2 arguments (shape, value)",
                ));
            }
            let shape_val = self.compile_expr(builder, state, &args[0].value)?;
            let fill_val = self.compile_expr(builder, state, &args[1].value)?;
            let arg_type = self.node_type(args[1].value.id).clone();
            let float_val = if is_float_type(&arg_type) {
                fill_val
            } else {
                builder.ins().fcvt_from_sint(cl_types::F64, fill_val)
            };
            return self.compile_call_by_name(builder, "nsl_tensor_full", &[shape_val, float_val]);
        }
        if func_name == "arange" {
            if args.is_empty() || args.len() > 3 {
                return Err(CodegenError::new(
                    "arange() takes 1-3 arguments (stop) or (start, stop[, step])",
                ));
            }
            let (start, stop, step) = match args.len() {
                1 => {
                    let stop_val = self.compile_expr(builder, state, &args[0].value)?;
                    let stop_type = self.node_type(args[0].value.id).clone();
                    let stop_f = if is_float_type(&stop_type) {
                        stop_val
                    } else {
                        builder.ins().fcvt_from_sint(cl_types::F64, stop_val)
                    };
                    let zero = builder.ins().f64const(0.0);
                    let one = builder.ins().f64const(1.0);
                    (zero, stop_f, one)
                }
                2 => {
                    let start_val = self.compile_expr(builder, state, &args[0].value)?;
                    let start_type = self.node_type(args[0].value.id).clone();
                    let start_f = if is_float_type(&start_type) {
                        start_val
                    } else {
                        builder.ins().fcvt_from_sint(cl_types::F64, start_val)
                    };
                    let stop_val = self.compile_expr(builder, state, &args[1].value)?;
                    let stop_type = self.node_type(args[1].value.id).clone();
                    let stop_f = if is_float_type(&stop_type) {
                        stop_val
                    } else {
                        builder.ins().fcvt_from_sint(cl_types::F64, stop_val)
                    };
                    let one = builder.ins().f64const(1.0);
                    (start_f, stop_f, one)
                }
                3 => {
                    let start_val = self.compile_expr(builder, state, &args[0].value)?;
                    let start_type = self.node_type(args[0].value.id).clone();
                    let start_f = if is_float_type(&start_type) {
                        start_val
                    } else {
                        builder.ins().fcvt_from_sint(cl_types::F64, start_val)
                    };
                    let stop_val = self.compile_expr(builder, state, &args[1].value)?;
                    let stop_type = self.node_type(args[1].value.id).clone();
                    let stop_f = if is_float_type(&stop_type) {
                        stop_val
                    } else {
                        builder.ins().fcvt_from_sint(cl_types::F64, stop_val)
                    };
                    let step_val = self.compile_expr(builder, state, &args[2].value)?;
                    let step_type = self.node_type(args[2].value.id).clone();
                    let step_f = if is_float_type(&step_type) {
                        step_val
                    } else {
                        builder.ins().fcvt_from_sint(cl_types::F64, step_val)
                    };
                    (start_f, stop_f, step_f)
                }
                _ => unreachable!(),
            };
            return self.compile_call_by_name(builder, "nsl_tensor_arange", &[start, stop, step]);
        }

        // Element-wise tensor builtins (M14)
        // Skip if there's a user-defined function with the same name (e.g., nsl.math.sign)
        if matches!(
            func_name.as_str(),
            "exp" | "log" | "sqrt" | "abs" | "sign" | "neg"
        ) && !self.registry.functions.contains_key(&func_name)
        {
            if args.len() != 1 {
                return Err(CodegenError::new(format!(
                    "{func_name}() takes exactly 1 argument"
                )));
            }
            let val = self.compile_nested_expr(builder, state, &args[0].value)?;
            // FBIP Phase 2: emit inplace variant when arg is single-use
            let rt_name = self.fbip_select_variant(state, &args[0].value, &func_name);
            return self.compile_traced_call(builder, &rt_name, &[val]);
        }
        // Activation functions (M15): relu, gelu, silu, sigmoid -- single tensor arg
        if matches!(func_name.as_str(), "relu" | "gelu" | "silu" | "sigmoid")
            && !self.registry.functions.contains_key(&func_name)
        {
            if args.len() != 1 {
                return Err(CodegenError::new(format!(
                    "{func_name}() takes exactly 1 argument"
                )));
            }
            let val = self.compile_nested_expr(builder, state, &args[0].value)?;
            // FBIP Phase 2: emit inplace variant when arg is single-use
            let rt_name = self.fbip_select_variant(state, &args[0].value, &func_name);
            return self.compile_traced_call(builder, &rt_name, &[val]);
        }
        // Tensor trig: tensor_sin, tensor_cos (RoPE support)
        if matches!(func_name.as_str(), "tensor_sin" | "tensor_cos")
            && !self.registry.functions.contains_key(&func_name)
        {
            if args.len() != 1 {
                return Err(CodegenError::new(format!(
                    "{func_name}() takes exactly 1 argument"
                )));
            }
            let val = self.compile_nested_expr(builder, state, &args[0].value)?;
            let rt_name = format!("nsl_{func_name}");
            return self.compile_traced_call(builder, &rt_name, &[val]);
        }
        // Fused rotate_half for RoPE
        if func_name == "rotate_half" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new("rotate_half() takes exactly 1 argument"));
            }
            let val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_traced_call(builder, "nsl_tensor_rotate_half", &[val]);
        }
        // tanh activation: maps NSL name "tanh" to runtime "nsl_tensor_tanh_act"
        if func_name == "tanh" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new("tanh() takes exactly 1 argument"));
            }
            let val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_tanh_act", &[val]);
        }
        // softmax(tensor, dim) -- two args
        if func_name == "softmax" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "softmax() takes exactly 2 arguments (tensor, dim)",
                ));
            }
            let tensor_val = self.compile_nested_expr(builder, state, &args[0].value)?;
            let dim_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_traced_call(builder, "nsl_tensor_softmax", &[tensor_val, dim_val]);
        }
        // log_softmax(tensor, dim) -- two args
        if func_name == "log_softmax" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "log_softmax() takes exactly 2 arguments (tensor, dim)",
                ));
            }
            let tensor_val = self.compile_nested_expr(builder, state, &args[0].value)?;
            let dim_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_traced_call(
                builder,
                "nsl_tensor_logsoftmax",
                &[tensor_val, dim_val],
            );
        }
        if func_name == "clamp" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 3 {
                return Err(CodegenError::new(
                    "clamp() takes exactly 3 arguments (tensor, min, max)",
                ));
            }
            let tensor_val = self.compile_nested_expr(builder, state, &args[0].value)?;
            let min_val = self.compile_expr(builder, state, &args[1].value)?;
            let max_val = self.compile_expr(builder, state, &args[2].value)?;
            // Ensure min/max are f64
            let min_type = self.node_type(args[1].value.id).clone();
            let min_f = if is_float_type(&min_type) {
                min_val
            } else {
                builder.ins().fcvt_from_sint(cl_types::F64, min_val)
            };
            let max_type = self.node_type(args[2].value.id).clone();
            let max_f = if is_float_type(&max_type) {
                max_val
            } else {
                builder.ins().fcvt_from_sint(cl_types::F64, max_val)
            };
            return self.compile_call_by_name(
                builder,
                "nsl_tensor_clamp",
                &[tensor_val, min_f, max_f],
            );
        }
        if func_name == "copy_data" {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "copy_data() takes exactly 2 arguments (dest, src)",
                ));
            }
            let dest = self.compile_expr(builder, state, &args[0].value)?;
            let src = self.compile_expr(builder, state, &args[1].value)?;
            self.compile_call_by_name(builder, "nsl_tensor_copy_data", &[dest, src])?;
            return Ok(dest);
        }
        if func_name == "add_inplace" {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "add_inplace() takes exactly 2 arguments (dest, src)",
                ));
            }
            let dest = self.compile_expr(builder, state, &args[0].value)?;
            let src = self.compile_expr(builder, state, &args[1].value)?;
            self.compile_call_by_name(builder, "nsl_tensor_add_inplace", &[dest, src])?;
            return Ok(dest);
        }
        if func_name == "zero_inplace" {
            if args.len() != 1 {
                return Err(CodegenError::new("zero_inplace() takes exactly 1 argument"));
            }
            let val = self.compile_expr(builder, state, &args[0].value)?;
            self.compile_call_by_name(builder, "nsl_tensor_zero_inplace", &[val])?;
            return Ok(val);
        }
        if func_name == "zeros_like" {
            if args.len() != 1 {
                return Err(CodegenError::new("zeros_like() takes exactly 1 argument"));
            }
            let val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[val]);
        }
        if func_name == "reduce_max" {
            if args.len() != 3 {
                return Err(CodegenError::new(
                    "reduce_max() takes exactly 3 arguments (tensor, dim, keepdim)",
                ));
            }
            let t = self.compile_expr(builder, state, &args[0].value)?;
            let dim = self.compile_expr(builder, state, &args[1].value)?;
            let keepdim = self.compile_expr(builder, state, &args[2].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_reduce_max", &[t, dim, keepdim]);
        }
        if func_name == "gather" {
            if args.len() != 3 {
                return Err(CodegenError::new(
                    "gather() takes exactly 3 arguments (tensor, dim, indices)",
                ));
            }
            let t = self.compile_expr(builder, state, &args[0].value)?;
            let dim = self.compile_expr(builder, state, &args[1].value)?;
            let indices = self.compile_expr(builder, state, &args[2].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_gather", &[t, dim, indices]);
        }
        // layernorm(input, weight, bias, eps) -> tensor
        if func_name == "layernorm" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 4 {
                return Err(CodegenError::new(
                    "layernorm() takes exactly 4 arguments (input, weight, bias, eps)",
                ));
            }
            let input_val = self.compile_expr(builder, state, &args[0].value)?;
            let weight_val = self.compile_expr(builder, state, &args[1].value)?;
            let bias_val = self.compile_expr(builder, state, &args[2].value)?;
            let eps_val = self.compile_expr(builder, state, &args[3].value)?;
            // Ensure eps is f64
            let eps_type = self.node_type(args[3].value.id).clone();
            let eps_f = if is_float_type(&eps_type) {
                eps_val
            } else {
                builder.ins().fcvt_from_sint(cl_types::F64, eps_val)
            };
            return self.compile_call_by_name(
                builder,
                "nsl_tensor_layernorm",
                &[input_val, weight_val, bias_val, eps_f],
            );
        }
        // rmsnorm(input, weight, eps) -> tensor
        if func_name == "rmsnorm" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 3 {
                return Err(CodegenError::new(
                    "rmsnorm() takes exactly 3 arguments (input, weight, eps)",
                ));
            }
            let input_val = self.compile_expr(builder, state, &args[0].value)?;
            let weight_val = self.compile_expr(builder, state, &args[1].value)?;
            let eps_val = self.compile_expr(builder, state, &args[2].value)?;
            let eps_type = self.node_type(args[2].value.id).clone();
            let eps_f = if is_float_type(&eps_type) {
                eps_val
            } else {
                builder.ins().fcvt_from_sint(cl_types::F64, eps_val)
            };
            return self.compile_call_by_name(
                builder,
                "nsl_tensor_rmsnorm",
                &[input_val, weight_val, eps_f],
            );
        }
        // dropout(tensor, p, training) -> tensor
        if func_name == "dropout" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 3 {
                return Err(CodegenError::new(
                    "dropout() takes exactly 3 arguments (tensor, p, training)",
                ));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let p_val = self.compile_expr(builder, state, &args[1].value)?;
            let p_type = self.node_type(args[1].value.id).clone();
            let p_f = if is_float_type(&p_type) {
                p_val
            } else {
                builder.ins().fcvt_from_sint(cl_types::F64, p_val)
            };
            let training_val = self.compile_expr(builder, state, &args[2].value)?;
            let training_i8 = {
                let vt = builder.func.dfg.value_type(training_val);
                if vt == cl_types::I8 {
                    training_val
                } else {
                    builder.ins().ireduce(cl_types::I8, training_val)
                }
            };
            return self.compile_call_by_name(
                builder,
                "nsl_tensor_dropout",
                &[tensor_val, p_f, training_i8],
            );
        }
        // conv2d(input, weight, bias, stride_h, stride_w, pad_h, pad_w) -> tensor
        if func_name == "conv2d" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 7 {
                return Err(CodegenError::new("conv2d() takes exactly 7 arguments (input, weight, bias, stride_h, stride_w, pad_h, pad_w)"));
            }
            let input_val = self.compile_expr(builder, state, &args[0].value)?;
            let weight_val = self.compile_expr(builder, state, &args[1].value)?;
            let bias_val = self.compile_expr(builder, state, &args[2].value)?;
            let stride_h = self.compile_expr(builder, state, &args[3].value)?;
            let stride_w = self.compile_expr(builder, state, &args[4].value)?;
            let pad_h = self.compile_expr(builder, state, &args[5].value)?;
            let pad_w = self.compile_expr(builder, state, &args[6].value)?;
            return self.compile_call_by_name(
                builder,
                "nsl_tensor_conv2d",
                &[
                    input_val, weight_val, bias_val, stride_h, stride_w, pad_h, pad_w,
                ],
            );
        }
        // maxpool2d(input, kernel_h, kernel_w, stride, padding) -> tensor
        if func_name == "maxpool2d" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 5 {
                return Err(CodegenError::new("maxpool2d() takes exactly 5 arguments (input, kernel_h, kernel_w, stride, padding)"));
            }
            let input_val = self.compile_expr(builder, state, &args[0].value)?;
            let kernel_h = self.compile_expr(builder, state, &args[1].value)?;
            let kernel_w = self.compile_expr(builder, state, &args[2].value)?;
            let stride_val = self.compile_expr(builder, state, &args[3].value)?;
            let padding_val = self.compile_expr(builder, state, &args[4].value)?;
            return self.compile_call_by_name(
                builder,
                "nsl_tensor_maxpool2d",
                &[input_val, kernel_h, kernel_w, stride_val, padding_val],
            );
        }
        // embedding_lookup(weight, indices) -> tensor
        if func_name == "embedding_lookup" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "embedding_lookup() takes exactly 2 arguments (weight, indices)",
                ));
            }
            let weight_val = self.compile_expr(builder, state, &args[0].value)?;
            let indices_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(
                builder,
                "nsl_tensor_embedding_lookup",
                &[weight_val, indices_val],
            );
        }
        // bias_add(tensor, bias) -> tensor -- broadcasts 1D bias over 2D tensor
        if func_name == "bias_add" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "bias_add() takes exactly 2 arguments (tensor, bias)",
                ));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let bias_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(
                builder,
                "nsl_tensor_bias_add",
                &[tensor_val, bias_val],
            );
        }
        // tensor_slice(tensor, dim, start, end) -> tensor
        if func_name == "tensor_slice" {
            if args.len() != 4 {
                return Err(CodegenError::new(
                    "tensor_slice() takes exactly 4 arguments (tensor, dim, start, end)",
                ));
            }
            let t = self.compile_expr(builder, state, &args[0].value)?;
            let dim = self.compile_expr(builder, state, &args[1].value)?;
            let start = self.compile_expr(builder, state, &args[2].value)?;
            let end = self.compile_expr(builder, state, &args[3].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_slice", &[t, dim, start, end]);
        }
        // tensor_cat(list_of_tensors, dim) -> tensor
        if func_name == "tensor_cat" {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "tensor_cat() takes exactly 2 arguments (tensor_list, dim)",
                ));
            }
            let list = self.compile_expr(builder, state, &args[0].value)?;
            let dim = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_cat", &[list, dim]);
        }
        // M18a: unsqueeze(tensor, dim) -- free function form
        if func_name == "unsqueeze" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "unsqueeze() takes exactly 2 arguments (tensor, dim)",
                ));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let dim_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(
                builder,
                "nsl_tensor_unsqueeze",
                &[tensor_val, dim_val],
            );
        }
        // M18a: stack(list_of_tensors, dim) -> tensor
        if func_name == "stack" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "stack() takes exactly 2 arguments (tensor_list, dim)",
                ));
            }
            let list_val = self.compile_expr(builder, state, &args[0].value)?;
            let dim_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_stack", &[list_val, dim_val]);
        }
        // contiguous(tensor) -> tensor — materialize non-contiguous views
        if func_name == "contiguous" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new(
                    "contiguous() takes exactly 1 argument (tensor)",
                ));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_contiguous", &[tensor_val]);
        }
        // M18a: causal_mask(seq_len) -> tensor
        if func_name == "causal_mask" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new(
                    "causal_mask() takes exactly 1 argument (seq_len)",
                ));
            }
            let seq_len_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_causal_mask", &[seq_len_val]);
        }
        // M27: scaled_dot_product_attention(Q, K, V, scale, causal=false)
        // Naive path: softmax((Q @ K.T) * scale [+ causal_mask]) @ V
        // Flash path: PTX dispatch via nsl_flash_attention when @flash_attention decorator is present
        if func_name == "scaled_dot_product_attention"
            && !self.registry.functions.contains_key(&func_name)
        {
            if args.len() < 4 {
                return Err(CodegenError::new(
                    "scaled_dot_product_attention() requires at least 4 arguments (Q, K, V, scale)",
                ));
            }

            let q_val = self.compile_expr(builder, state, &args[0].value)?;
            let k_val = self.compile_expr(builder, state, &args[1].value)?;
            let v_val = self.compile_expr(builder, state, &args[2].value)?;
            let scale_val = self.compile_expr(builder, state, &args[3].value)?;

            // Check for causal named arg
            let mut causal = false;
            if args.len() > 4 {
                for arg in &args[4..] {
                    if let Some(name_sym) = arg.name {
                        let name = self.resolve_sym(name_sym).to_string();
                        if name == "causal" {
                            if let ExprKind::BoolLiteral(b) = arg.value.kind {
                                causal = b;
                            }
                        }
                    }
                }
            }

            // Check if the enclosing function has @flash_attention decorator
            if self.kernels.flash_attention_context.is_some() {
                return self.compile_flash_attention_call(builder, state, q_val, k_val, v_val, scale_val);
            }

            // M34: `@context_parallel` decorator (ring attention).
            //
            // CPDT Part III v2.19 fixed the multi-model lookup via
            // `resolve_decorator_config_for_call_site`; v2.20 generalized
            // the helper; v2.21 fixed the F64/I64 scale mismatch. But
            // v2.20 bug #2 (codegen/FFI positional misalignment — only
            // position 0 aligns; every non-zero slot carries semantically
            // wrong bytes) and bug #3 (runtime impl in
            // `crates/nsl-runtime/src/context_parallel/ffi.rs` returns 0
            // unconditionally) remained. Both are M34-completion work
            // requiring a runtime redesign, not v2.x scope.
            //
            // v2.22 (this cycle): delete the buggy ring-attention FFI
            // chain entirely and fall through to the naive attention
            // path below. `@context_parallel` becomes advisory — the
            // build emits a WARNING that the ring-attention runtime is
            // incomplete, but the method still produces mathematically
            // correct (undistributed) output via the naive
            // softmax(QK^T · scale)V path. When M34 lands, someone
            // rewires the emission against the finalized runtime FFI
            // shape here, and the warning + fallback go away together.
            let cp_config = crate::moe::resolve_decorator_config_for_call_site(
                self.current_method_model_name.as_deref(),
                &self.features.context_parallel_configs,
            )
            .map(|(_, info)| info);

            if let Some(cp_info) = cp_config {
                eprintln!(
                    "[nsl] warning: @context_parallel(ring_size={}) recognized but the ring-attention runtime is incomplete (M34 in progress); falling through to naive attention. Output is mathematically correct but not distributed.",
                    cp_info.ring_size
                );
                // Fall through to the naive path below. Do NOT emit
                // any `nsl_cp_init` / `nsl_sequence_partition` /
                // `nsl_ring_attention` / `nsl_sequence_gather` /
                // `nsl_cp_destroy` FFI calls — all previously landed
                // in stub runtime slots and produced wrong output.
            }

            // Naive path: softmax(apply_causal_mask((Q @ K.T) * scale)) @ V
            let dim_m2 = builder.ins().iconst(cl_types::I64, -2_i64);
            let dim_m1 = builder.ins().iconst(cl_types::I64, -1_i64);
            let k_t = self.compile_call_by_name(
                builder,
                "nsl_tensor_transpose",
                &[k_val, dim_m2, dim_m1],
            )?;
            // ELTLS (FBIP-3): nsl_tensor_matmul takes a flags byte.
            let matmul_flags0 = builder.ins().iconst(cl_types::I8, 0);
            let scores = self.compile_traced_call(builder, "nsl_tensor_matmul", &[q_val, k_t, matmul_flags0])?;
            let scaled =
                self.compile_traced_call(builder, "nsl_tensor_mul_scalar", &[scores, scale_val, matmul_flags0])?;

            let masked = if causal {
                let dim_neg2 = builder.ins().iconst(cl_types::I64, -2_i64);
                let seq_len =
                    self.compile_call_by_name(builder, "nsl_tensor_shape_dim", &[q_val, dim_neg2])?;
                let mask =
                    self.compile_call_by_name(builder, "nsl_tensor_causal_mask", &[seq_len])?;
                {
                    // ELTLS (FBIP-3): nsl_tensor_add takes flags=0 here.
                    let flags_zero = builder.ins().iconst(cl_types::I8, 0);
                    self.compile_traced_call(builder, "nsl_tensor_add", &[scaled, mask, flags_zero])?
                }
            } else {
                scaled
            };

            let dim_neg1 = builder.ins().iconst(cl_types::I64, -1_i64);
            let attn_weights =
                self.compile_traced_call(builder, "nsl_tensor_softmax", &[masked, dim_neg1])?;
            // ELTLS (FBIP-3): nsl_tensor_matmul takes a flags byte.
            let final_matmul_flags0 = builder.ins().iconst(cl_types::I8, 0);
            return self.compile_traced_call(builder, "nsl_tensor_matmul", &[attn_weights, v_val, final_matmul_flags0]);
        }

        // sum/mean with dim args -- overload: sum(tensor) or sum(tensor, dim, keepdim)
        if matches!(func_name.as_str(), "sum" | "mean") {
            if args.len() == 1 {
                // M46: Full reductions (no dim arg) don't have 1-arg deterministic variants;
                // deterministic swap only applies to the 3-arg sum_dim/mean_dim below.
                let val = self.compile_expr(builder, state, &args[0].value)?;
                let rt_name = format!("nsl_tensor_{func_name}");
                return self.compile_traced_call(builder, &rt_name, &[val]);
            } else if args.len() == 3 {
                let t = self.compile_expr(builder, state, &args[0].value)?;
                let dim = self.compile_expr(builder, state, &args[1].value)?;
                let keepdim = self.compile_expr(builder, state, &args[2].value)?;
                // M46: Swap to deterministic kernel variants when deterministic mode is active
                let rt_name = if self.compile_options.deterministic {
                    match func_name.as_str() {
                        "sum" => "nsl_tensor_reduce_sum_deterministic".to_string(),
                        "mean" => "nsl_tensor_reduce_mean_deterministic".to_string(),
                        _ => format!("nsl_tensor_{func_name}_dim"),
                    }
                } else {
                    format!("nsl_tensor_{func_name}_dim")
                };
                return self.compile_traced_call(builder, &rt_name, &[t, dim, keepdim]);
            } else {
                return Err(CodegenError::new(format!(
                    "{func_name}() takes 1 or 3 arguments"
                )));
            }
        }

        // Tokenizer functions (M15)
        if func_name == "byte_tokenizer_new" {
            return self.compile_call_by_name(builder, "nsl_byte_tokenizer_new", &[]);
        }
        if func_name == "bpe_train" {
            if args.len() != 4 {
                return Err(CodegenError::new("bpe_train() takes exactly 4 arguments (corpus_path, vocab_size, min_freq, special_tokens)"));
            }
            let path_val = self.compile_expr(builder, state, &args[0].value)?;
            let vocab_size = self.compile_expr(builder, state, &args[1].value)?;
            let min_freq = self.compile_expr(builder, state, &args[2].value)?;
            let special_tokens = self.compile_expr(builder, state, &args[3].value)?;
            return self.compile_call_by_name(
                builder,
                "nsl_bpe_train",
                &[path_val, vocab_size, min_freq, special_tokens],
            );
        }
        if func_name == "tokenizer_load" {
            if args.len() != 1 {
                return Err(CodegenError::new(
                    "tokenizer_load() takes exactly 1 argument (path)",
                ));
            }
            let path_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_tokenizer_load", &[path_val]);
        }
        if func_name == "tokenizer_save" {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "tokenizer_save() takes exactly 2 arguments (handle, path)",
                ));
            }
            let handle = self.compile_expr(builder, state, &args[0].value)?;
            let path_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tokenizer_save", &[handle, path_val]);
        }
        if func_name == "tokenizer_encode" {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "tokenizer_encode() takes exactly 2 arguments (handle, text)",
                ));
            }
            let handle = self.compile_expr(builder, state, &args[0].value)?;
            let text_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tokenizer_encode", &[handle, text_val]);
        }
        if func_name == "tokenizer_decode" {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "tokenizer_decode() takes exactly 2 arguments (handle, tensor)",
                ));
            }
            let handle = self.compile_expr(builder, state, &args[0].value)?;
            let tensor_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(
                builder,
                "nsl_tokenizer_decode",
                &[handle, tensor_val],
            );
        }
        if func_name == "tokenizer_vocab_size" {
            if args.len() != 1 {
                return Err(CodegenError::new(
                    "tokenizer_vocab_size() takes exactly 1 argument (handle)",
                ));
            }
            let handle = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_tokenizer_vocab_size", &[handle]);
        }
        if func_name == "tokenizer_encode_batch" {
            if args.len() != 5 {
                return Err(CodegenError::new("tokenizer_encode_batch() takes exactly 5 arguments (handle, texts, padding, truncation, max_len)"));
            }
            let handle = self.compile_expr(builder, state, &args[0].value)?;
            let texts = self.compile_expr(builder, state, &args[1].value)?;
            let padding = self.compile_expr(builder, state, &args[2].value)?;
            let padding_i8 = if builder.func.dfg.value_type(padding) == cl_types::I8 {
                padding
            } else {
                builder.ins().ireduce(cl_types::I8, padding)
            };
            let truncation = self.compile_expr(builder, state, &args[3].value)?;
            let truncation_i8 = if builder.func.dfg.value_type(truncation) == cl_types::I8 {
                truncation
            } else {
                builder.ins().ireduce(cl_types::I8, truncation)
            };
            let max_len = self.compile_expr(builder, state, &args[4].value)?;
            return self.compile_call_by_name(
                builder,
                "nsl_tokenizer_encode_batch",
                &[handle, texts, padding_i8, truncation_i8, max_len],
            );
        }

        // Quantization functions (M16)
        if func_name == "nsl_qtensor_quantize" && !self.registry.functions.contains_key(&func_name)
        {
            if args.len() != 5 {
                return Err(CodegenError::new("nsl_qtensor_quantize() takes exactly 5 arguments (tensor, dtype, granularity, axis, group_size)"));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let dtype_val = self.compile_expr(builder, state, &args[1].value)?;
            let gran_val = self.compile_expr(builder, state, &args[2].value)?;
            let axis_val = self.compile_expr(builder, state, &args[3].value)?;
            let group_val = self.compile_expr(builder, state, &args[4].value)?;
            return self.compile_call_by_name(
                builder,
                "nsl_qtensor_quantize",
                &[tensor_val, dtype_val, gran_val, axis_val, group_val],
            );
        }
        if func_name == "nsl_qtensor_dequantize"
            && !self.registry.functions.contains_key(&func_name)
        {
            if args.len() != 1 {
                return Err(CodegenError::new(
                    "nsl_qtensor_dequantize() takes exactly 1 argument (qtensor)",
                ));
            }
            let qt_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_qtensor_dequantize", &[qt_val]);
        }
        if func_name == "nsl_qtensor_matmul_mixed"
            && !self.registry.functions.contains_key(&func_name)
        {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "nsl_qtensor_matmul_mixed() takes exactly 2 arguments (tensor, qtensor)",
                ));
            }
            let x_val = self.compile_expr(builder, state, &args[0].value)?;
            let qw_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(
                builder,
                "nsl_qtensor_matmul_mixed",
                &[x_val, qw_val],
            );
        }
        if func_name == "nsl_qtensor_dtype" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new(
                    "nsl_qtensor_dtype() takes exactly 1 argument (qtensor)",
                ));
            }
            let qt_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_qtensor_dtype", &[qt_val]);
        }
        if func_name == "nsl_qtensor_shape" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new(
                    "nsl_qtensor_shape() takes exactly 1 argument (qtensor)",
                ));
            }
            let qt_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_qtensor_shape", &[qt_val]);
        }

        // HuggingFace model loading intrinsic: from_hf("repo_id", model_var)
        if func_name == "from_hf" {
            return self.compile_from_hf(builder, state, args);
        }

        // ONNX export intrinsic: to_onnx(model, input, "output.onnx")
        if func_name == "to_onnx" {
            return self.compile_to_onnx(builder, state, args);
        }

        // Checkpoint I/O: model_save(model, "path.nslm")
        if func_name == "model_save" {
            return self.compile_model_save(builder, state, args);
        }

        // Checkpoint I/O: model_load(model, "path.nslm")
        if func_name == "model_load" {
            return self.compile_model_load(builder, state, args);
        }

        // Safetensors load intrinsic: load_safetensors("path.safetensors")
        // Optional second arg: device (0=CPU default, 1=CUDA)
        if func_name == "load_safetensors" {
            if args.is_empty() || args.len() > 2 {
                return Err(CodegenError::new(
                    "load_safetensors() takes 1-2 arguments (path, device=0)",
                ));
            }
            let path_val = self.compile_expr(builder, state, &args[0].value)?;
            let path_str = match &args[0].value.kind {
                ExprKind::StringLiteral(s) => s.clone(),
                _ => {
                    return Err(CodegenError::new(
                        "load_safetensors(): first argument must be a string literal (file path)",
                    ));
                }
            };
            let path_len = builder.ins().iconst(cl_types::I64, path_str.len() as i64);
            let device_val = if args.len() > 1 {
                self.compile_expr(builder, state, &args[1].value)?
            } else {
                builder.ins().iconst(cl_types::I64, 0)
            };
            return self.compile_call_by_name(
                builder,
                "nsl_safetensors_load",
                &[path_val, path_len, device_val],
            );
        }

        // Safetensors save intrinsic: save_safetensors(weights_dict, "path.safetensors")
        if func_name == "save_safetensors" {
            if args.len() != 2 {
                return Err(CodegenError::new(
                    "save_safetensors() takes exactly 2 arguments (weights_dict, path)",
                ));
            }
            let dict_val = self.compile_expr(builder, state, &args[0].value)?;
            let path_val = self.compile_expr(builder, state, &args[1].value)?;
            let path_str = match &args[1].value.kind {
                ExprKind::StringLiteral(s) => s.clone(),
                _ => {
                    return Err(CodegenError::new(
                        "save_safetensors(): second argument must be a string literal (file path)",
                    ));
                }
            };
            let path_len = builder.ins().iconst(cl_types::I64, path_str.len() as i64);
            return self.compile_call_by_name(
                builder,
                "nsl_safetensors_save",
                &[dict_val, path_val, path_len],
            );
        }

        // CPDT Part III v2.5 SwiGLU MoE FFN lowering:
        // `moe_dispatch_swiglu(tokens, logits, experts_gate, experts_up,
        // experts_down)` is the 5-arg variant that emits
        // `nsl_moe_dispatch_full_v4` (Mixtral's default FFN structure).
        // Coexists with v2 `moe_dispatch` and v3 `moe_dispatch_ffn`;
        // users opt in by switching intrinsics.
        //
        // No silent fallback to v3/v2 — users that call this intrinsic
        // explicitly want SwiGLU. Failure is loud, message lists the
        // expected WeightMap keys.
        if func_name == "moe_dispatch_swiglu" {
            // CPDT Part III v2.14 — `moe_dispatch_swiglu` accepts the
            // 5-arg form (no bias) for the Mixtral-without-bias case
            // (the SwiGLU paper convention) OR the 8-arg form (...,
            // experts_gate_bias, experts_up_bias, experts_down_bias)
            // for the rare biased variant (e.g., NSL-converted GPT-2
            // ported to SwiGLU). The v2.14 FFI (14 i64 args) accepts
            // both forms via the nullable bias pointers. Mirrors v3's
            // 4/6 arity pattern from v2.12.
            if args.len() != 5 && args.len() != 8 {
                // v2.14 fix F7 (IMPORTANT adversarial review): cite a
                // concrete biased-v4 family. v3's F7 names GPT-2/OPT
                // for the bias case (well-known); v4 biased variants
                // are rarer but T5-style models with SwiGLU + bias do
                // exist (and any NSL-converted GeGLU/ReGLU MoE could
                // ship biases). Naming the convention makes the
                // 8-arg form recognizable.
                return Err(crate::error::CodegenError::new(
                    "moe_dispatch_swiglu() takes 5 arguments (tokens, router_logits, experts_gate, \
                     experts_up, experts_down) OR 8 arguments (..., experts_gate_bias, \
                     experts_up_bias, experts_down_bias). The 8-arg form is required when the \
                     loaded weight bundle has v4 FFN biases — all 3 (gate, up, down) are \
                     required if any are present (v4 paper convention; mirrors v3's GPT-2 / OPT \
                     pattern but with 3 directions instead of 2). The 5-arg form is the default \
                     for the Mixtral / DeepSeek convention which omits FFN bias.",
                ));
            }
            let tokens_val = self.compile_expr(builder, state, &args[0].value)?;
            let logits_val = self.compile_expr(builder, state, &args[1].value)?;
            let experts_gate_val = self.compile_expr(builder, state, &args[2].value)?;
            let experts_up_val = self.compile_expr(builder, state, &args[3].value)?;
            let experts_down_val = self.compile_expr(builder, state, &args[4].value)?;
            // v2.14: 8-arg form compiles 3 bias expressions; 5-arg
            // leaves bias_vals = None and emits iconst(0)/iconst(0)/
            // iconst(0) at the call site (preserves v2.5/v2.7 v4
            // emission byte-identical for bias-free MoE).
            let bias_vals: Option<(_, _, _)> = if args.len() == 8 {
                let g = self.compile_expr(builder, state, &args[5].value)?;
                let u = self.compile_expr(builder, state, &args[6].value)?;
                let d = self.compile_expr(builder, state, &args[7].value)?;
                Some((g, u, d))
            } else {
                None
            };

            // CPDT Part III v2.19 — use the model-method-aware helper
            // so multi-model `@moe` builds resolve each call site to
            // its own model's MoE config. Multi-MoE-PER-model is the
            // documented v2.next deferral (helper returns None and
            // the call site falls into the hard-error path).
            let config_with_key = crate::moe::resolve_decorator_config_for_call_site(
                self.current_method_model_name.as_deref(),
                &self.features.moe_configs,
            );
            let (cfg_key, num_experts, top_k, capacity_factor, activation, weight_prefix) =
                match &config_with_key {
                    Some((k, info)) => (
                        Some(k.clone()),
                        info.num_experts,
                        info.top_k,
                        info.capacity_factor,
                        info.activation,
                        info.weight_prefix.clone(),
                    ),
                    // Fallback path is unreachable when v4 dims resolve
                    // (config_with_key=None ⇒ lookup_key=None ⇒
                    // v4_dims=None ⇒ hard error before activation is
                    // consumed). Mirrors the v3 arm's use of
                    // MoeActivation::default() so a future flip of the
                    // default can't silently diverge v3 vs v4 fallbacks.
                    None => (None, 8, 2, 1.25f32, crate::moe::MoeActivation::default(), None),
                };

            // CPDT Part III v2.8 — gate-activation selection.
            // v4 now accepts gate_activation_kind ∈ {1, 2, 3}
            // (SwiGLU/GeGLU/ReGLU). `@moe(activation="…")` is the source
            // of truth, mapping silu→1, gelu→2, relu→3.
            //
            // Identity is REFUSED at the codegen boundary (matching the
            // runtime FFI's upfront gate). A GLU with identity gate
            // degenerates to `gate * up @ down`, which is structurally
            // non-Mixtral; callers that genuinely want no gate should
            // use `moe_dispatch_ffn` (v3 FFI, 2-weight FFN) with
            // `@moe(activation="identity")`, not v4.
            if activation == crate::moe::MoeActivation::Identity {
                return Err(crate::error::CodegenError::new(
                    "moe_dispatch_swiglu: @moe(activation=\"identity\") is invalid \
                     for the SwiGLU FFI — a GLU with identity gate degenerates to \
                     `gate * up @ down`, which is not a known production MoE \
                     structure. Use `moe_dispatch_ffn` (v3 FFI) for a 2-weight FFN \
                     without a gate activation. Valid v4 activations: silu \
                     (SwiGLU, default), gelu (GeGLU), relu (ReGLU)."
                        .to_string(),
                ));
            }

            let num_experts_val = builder
                .ins()
                .iconst(cranelift_codegen::ir::types::I64, num_experts as i64);
            let top_k_val = builder
                .ins()
                .iconst(cranelift_codegen::ir::types::I64, top_k as i64);
            let cap_bits = builder.ins().iconst(
                cranelift_codegen::ir::types::I64,
                capacity_factor.to_bits() as i64,
            );
            let gate_activation_val = builder
                .ins()
                .iconst(cranelift_codegen::ir::types::I64, activation as i64);

            // CPDT Part III v2.7 — `@moe(weight_prefix="…")` overrides
            // the moe_configs key for WeightMap lookups. When Some,
            // the v4 arm derives dims under that prefix (so a user
            // running against an HF Mixtral safetensors file can say
            // `weight_prefix="model.layers.0.block_sparse_moe"` even
            // though NSL field names cannot contain `.`).
            let lookup_key = weight_prefix.as_ref().or(cfg_key.as_ref()).cloned();
            let v4_dims = if let (Some(key), Some(weight_map)) =
                (lookup_key.as_ref(), self.features.weight_map.as_ref())
            {
                crate::moe::derive_v4_dims(weight_map, key, num_experts)
            } else {
                None
            };

            // CPDT Part III v2.14 fix F1 (IMPORTANT adversarial review):
            // orphan-bias pre-pass for v4, symmetric to v2.12 F2 for v3.
            // If bias entries are present but the weight dims fail to
            // resolve (e.g., malformed bundle: biases present but
            // missing or mis-shaped weight tensors), surface a bias-
            // aware error BEFORE falling through to the generic v4-
            // dims-not-resolvable diagnostic. Without this gate the
            // user only sees "missing experts.gate/up/down.weight"
            // with no hint that orphan v4 biases also exist in the
            // bundle and would have activated with the right weight
            // pack.
            if v4_dims.is_none() {
                if let (Some(key), Some(weight_map)) =
                    (lookup_key.as_ref(), self.features.weight_map.as_ref())
                {
                    if crate::moe::any_v4_bias_entry_present(weight_map, key) {
                        return Err(crate::error::CodegenError::new(format!(
                            "moe_dispatch_swiglu: WeightMap under '{key}' contains v4 bias \
                             entries (`experts.gate.bias` and/or `experts.up.bias` and/or \
                             `experts.down.bias`) BUT the v4 weight dimensions could not \
                             be resolved (missing or mis-shaped `experts.gate.weight` / \
                             `experts.up.weight` / `experts.down.weight`). This is an \
                             orphaned-bias bundle: the biases would never activate \
                             because their parent projections are absent. Fix the bundle \
                             by adding the missing weight tensors (and re-running any \
                             auto-pack step), or remove the orphaned biases."
                        )));
                    }
                }
            }

            if let Some((hidden_dim, intermediate_dim)) = v4_dims {
                // CPDT Part III v2.14 source-activation gate. Mirrors
                // v3's v2.12 detection: when the WeightMap has v4
                // biases, the source MUST use the 8-arg form so the
                // bias pointers reach the FFI; the 5-arg form would
                // silently drop them. Partial / mis-shaped biases
                // also refuse here with actionable element counts.
                let weight_map_has_biases = if let (Some(key), Some(weight_map)) =
                    (lookup_key.as_ref(), self.features.weight_map.as_ref())
                {
                    match crate::moe::detect_v4_biases(
                        weight_map,
                        key,
                        num_experts,
                        hidden_dim,
                        intermediate_dim,
                    ) {
                        Ok(opt) => opt.is_some(),
                        Err(msg) => {
                            return Err(crate::error::CodegenError::new(format!(
                                "moe_dispatch_swiglu: {msg}"
                            )));
                        }
                    }
                } else {
                    false
                };
                if weight_map_has_biases && bias_vals.is_none() {
                    return Err(crate::error::CodegenError::new(format!(
                        "moe_dispatch_swiglu: WeightMap under '{}' has v4 FFN biases \
                         (`experts.gate.bias` + `experts.up.bias` + `experts.down.bias`), but \
                         the call site uses the 5-arg form (no bias). To activate the loaded \
                         biases, add the bias model fields and call the 8-arg form: \
                         `moe_dispatch_swiglu(tokens, logits, experts_gate, experts_up, \
                         experts_down, experts_gate_bias, experts_up_bias, experts_down_bias)`. \
                         Or, to drop the loaded biases, remove them from the weight bundle.",
                        lookup_key.as_deref().unwrap_or("?")
                    )));
                }

                let hidden_val = builder.ins().iconst(
                    cranelift_codegen::ir::types::I64,
                    hidden_dim as i64,
                );
                let intermediate_val = builder.ins().iconst(
                    cranelift_codegen::ir::types::I64,
                    intermediate_dim as i64,
                );
                // v2.14: bias args. The 5-arg form emits 3 separate
                // iconst(0) values (matches v2.12 F1 Cranelift IR
                // hygiene — no Value reuse across arg slots). The
                // 8-arg form threads the compiled bias expressions.
                let (bias_gate_arg, bias_up_arg, bias_down_arg) = match bias_vals {
                    Some((g, u, d)) => (g, u, d),
                    None => {
                        let zg = builder
                            .ins()
                            .iconst(cranelift_codegen::ir::types::I64, 0_i64);
                        let zu = builder
                            .ins()
                            .iconst(cranelift_codegen::ir::types::I64, 0_i64);
                        let zd = builder
                            .ins()
                            .iconst(cranelift_codegen::ir::types::I64, 0_i64);
                        (zg, zu, zd)
                    }
                };
                let result = self.compile_call_by_name(
                    builder,
                    "nsl_moe_dispatch_full_v4",
                    &[
                        tokens_val,
                        logits_val,
                        experts_gate_val,
                        experts_up_val,
                        experts_down_val,
                        num_experts_val,
                        top_k_val,
                        cap_bits,
                        hidden_val,
                        intermediate_val,
                        gate_activation_val,
                        bias_gate_arg,
                        bias_up_arg,
                        bias_down_arg,
                    ],
                )?;
                return Ok(result);
            }

            // No silent fallback. SwiGLU is opt-in.
            return Err(crate::error::CodegenError::new(format!(
                "moe_dispatch_swiglu: v4 (SwiGLU) requires resolvable router + \
                 experts.gate + experts.up + experts.down entries in the WeightMap. \
                 None of the standard names resolved under the lookup prefix{}, or \
                 their shapes were inconsistent (router must be [hidden, num_experts]; \
                 each of experts.gate / experts.up must be [num_experts, hidden * \
                 intermediate]; experts.down must be [num_experts, intermediate * \
                 hidden]; all three projections must agree on intermediate). Pass \
                 --weights with a safetensors bundle containing <prefix>.router.weight \
                 AND <prefix>.experts.gate.weight AND <prefix>.experts.up.weight AND \
                 <prefix>.experts.down.weight (with `.weight` suffix optional). Raw HF \
                 Mixtral safetensors use per-expert experts.{{e}}.w1/w2/w3 — v2.6's \
                 packing primitive (auto-run at WeightMap::load since v2.7) rewrites \
                 them in place under the HF prefix; declare `@moe(weight_prefix=\"...\")` \
                 in source to point this lookup at that prefix. For biased v4 \
                 checkpoints, also include <prefix>.experts.{{gate,up,down}}.bias and \
                 call the 8-arg form of moe_dispatch_swiglu (v2.14).",
                lookup_key
                    .as_deref()
                    .map(|k| format!(" '{}'", k))
                    .unwrap_or_default(),
            )));
        }

        // CPDT Part III v2.3 paper-faithful MoE FFN lowering:
        // `moe_dispatch_ffn(tokens, logits, experts_up, experts_down)` is
        // the opt-in 4-arg variant that emits `nsl_moe_dispatch_full_v3`.
        // It coexists with the 3-arg `moe_dispatch` (v1/v2 path) — users
        // opt into v3 by switching source intrinsics; existing v1/v2
        // callers are unaffected.
        //
        // Shape-derivation rules (mirror v1/v2 with the up/down split):
        //   - cpdt_mode == Full → require both router AND experts.up AND
        //     experts.down to resolve in the WeightMap under the matching
        //     MoeConfig key with consistent dims. None → hard CodegenError.
        //   - cpdt_mode != Full → if v3 dims resolve, emit v3; otherwise
        //     return an error (no silent v2 fallback from this site —
        //     callers using moe_dispatch_ffn explicitly want v3).
        //
        // Activation: hardcoded SiLU (activation_kind=1) for v2.3.
        // Per-decorator `@moe(activation="gelu" | "swiglu" | ...)` is a
        // v2.4 deferral — when added it threads through the existing
        // `extract_moe_decorator` path in moe.rs.
        if func_name == "moe_dispatch_ffn" {
            // CPDT Part III v2.12 — `moe_dispatch_ffn` accepts the 4-arg
            // form (tokens, logits, experts_up, experts_down) for the
            // bias-free case (Llama-1+, Mixtral SwiGLU pre-bias variant)
            // OR the 6-arg form (..., experts_up_bias, experts_down_bias)
            // for GPT-2 / OPT FFN-bias families. Source-level activation
            // gate: when the WeightMap has bias entries but the 4-arg
            // form is used, refuse loudly (see the detect_v3_biases call
            // below). The v2.11 FFI signature (12 i64 args) accepts both
            // forms via the nullable bias pointers.
            if args.len() != 4 && args.len() != 6 {
                return Err(crate::error::CodegenError::new(
                    "moe_dispatch_ffn() takes 4 arguments (tokens, router_logits, \
                     experts_up, experts_down) OR 6 arguments (..., experts_up_bias, \
                     experts_down_bias). The 6-arg form is required when the loaded \
                     weight bundle has FFN biases (GPT-2 / OPT convention); the 4-arg \
                     form is required when it does not (Llama-1+ / Mixtral).",
                ));
            }
            let tokens_val = self.compile_expr(builder, state, &args[0].value)?;
            let logits_val = self.compile_expr(builder, state, &args[1].value)?;
            let experts_up_val = self.compile_expr(builder, state, &args[2].value)?;
            let experts_down_val = self.compile_expr(builder, state, &args[3].value)?;
            // v2.12: 6-arg form compiles the bias expressions. The
            // 4-arg form leaves these as None and emits `iconst(0)` at
            // the call site below (preserves v2.5/v2.7 byte-identical
            // emission for bias-free MoE).
            let bias_vals: Option<(_, _)> = if args.len() == 6 {
                let bias_up_val = self.compile_expr(builder, state, &args[4].value)?;
                let bias_down_val = self.compile_expr(builder, state, &args[5].value)?;
                Some((bias_up_val, bias_down_val))
            } else {
                None
            };

            // CPDT Part III v2.19 — multi-model `@moe` composition.
            // The helper picks the call site's MoE config under the
            // current method's model. Multi-MoE-per-MODEL (≥2 `@moe`
            // decorators on one model) remains the documented v2.next
            // deferral and surfaces as `None` here, hitting the v3
            // hard-error path below.
            let config_with_key = crate::moe::resolve_decorator_config_for_call_site(
                self.current_method_model_name.as_deref(),
                &self.features.moe_configs,
            );
            let (cfg_key, num_experts, top_k, capacity_factor, activation, weight_prefix) =
                match &config_with_key {
                    Some((k, info)) => (
                        Some(k.clone()),
                        info.num_experts,
                        info.top_k,
                        info.capacity_factor,
                        info.activation,
                        info.weight_prefix.clone(),
                    ),
                    // No MoeConfig matched. Fall back to single-MoE defaults
                    // for the numeric fields and the default activation
                    // (SiLU). The error path below still fires loud when v3
                    // dims can't be derived.
                    None => (
                        None,
                        8,
                        2,
                        1.25f32,
                        crate::moe::MoeActivation::default(),
                        None,
                    ),
                };

            let num_experts_val = builder
                .ins()
                .iconst(cranelift_codegen::ir::types::I64, num_experts as i64);
            let top_k_val = builder
                .ins()
                .iconst(cranelift_codegen::ir::types::I64, top_k as i64);
            let cap_bits = builder.ins().iconst(
                cranelift_codegen::ir::types::I64,
                capacity_factor.to_bits() as i64,
            );

            // CPDT Part III v2.7 — `@moe(weight_prefix="…")` overrides
            // the moe_configs key for WeightMap lookups (same pattern as
            // the v4 arm above). Lets a user point the v3 lowering at
            // an HF-style prefix that contains `.` (NSL field names
            // can't).
            let lookup_key = weight_prefix.as_ref().or(cfg_key.as_ref()).cloned();
            let v3_dims = if let (Some(key), Some(weight_map)) =
                (lookup_key.as_ref(), self.features.weight_map.as_ref())
            {
                crate::moe::derive_v3_dims(weight_map, key, num_experts)
            } else {
                None
            };

            // CPDT Part III v2.12 fix F2 (HIGH adversarial review): if
            // bias entries are present but the weight dims fail to
            // resolve (e.g., the user shipped a malformed bundle with
            // biases but missing weights), surface a bias-aware error
            // BEFORE falling through to the generic v3-dims-not-
            // resolvable diagnostic. Otherwise the user sees only
            // "missing experts.up.weight" and never realizes their
            // bundle also has orphaned bias entries that would have
            // activated with the right weight pack.
            if v3_dims.is_none() {
                if let (Some(key), Some(weight_map)) =
                    (lookup_key.as_ref(), self.features.weight_map.as_ref())
                {
                    if crate::moe::any_v3_bias_entry_present(weight_map, key) {
                        return Err(crate::error::CodegenError::new(format!(
                            "moe_dispatch_ffn: WeightMap under '{key}' contains v3 bias \
                             entries (`experts.up.bias` and/or `experts.down.bias`) BUT \
                             the v3 weight dimensions could not be resolved (missing or \
                             mis-shaped `experts.up.weight` / `experts.down.weight`). \
                             This is an orphaned-bias bundle: the biases would never \
                             activate because their parent projections are absent. Fix \
                             the bundle by adding the missing weight tensors (and \
                             re-running any auto-pack step), or remove the orphaned \
                             biases."
                        )));
                    }
                }
            }

            if let Some((hidden_dim, intermediate_dim)) = v3_dims {
                // CPDT Part III v2.12 source-activation gate. When the
                // WeightMap has v3 biases, the source MUST use the 6-arg
                // form so the bias pointers reach the FFI; the 4-arg
                // form would silently drop them. Partial / mis-shaped
                // biases also refuse here so the diagnostic lands at
                // codegen with the actionable element counts.
                let weight_map_has_biases = if let (Some(key), Some(weight_map)) =
                    (lookup_key.as_ref(), self.features.weight_map.as_ref())
                {
                    match crate::moe::detect_v3_biases(
                        weight_map,
                        key,
                        num_experts,
                        hidden_dim,
                        intermediate_dim,
                    ) {
                        Ok(opt) => opt.is_some(),
                        Err(msg) => {
                            return Err(crate::error::CodegenError::new(format!(
                                "moe_dispatch_ffn: {msg}"
                            )));
                        }
                    }
                } else {
                    false
                };
                if weight_map_has_biases && bias_vals.is_none() {
                    return Err(crate::error::CodegenError::new(format!(
                        "moe_dispatch_ffn: WeightMap under '{}' has v3 FFN biases \
                         (`experts.up.bias` + `experts.down.bias`), but the call site \
                         uses the 4-arg form (no bias). To activate the loaded biases, \
                         add the bias model fields and call the 6-arg form: \
                         `moe_dispatch_ffn(tokens, logits, experts_up, experts_down, \
                         experts_up_bias, experts_down_bias)`. Or, to drop the loaded \
                         biases, remove them from the weight bundle.",
                        lookup_key.as_deref().unwrap_or("?")
                    )));
                }
                let hidden_val = builder.ins().iconst(
                    cranelift_codegen::ir::types::I64,
                    hidden_dim as i64,
                );
                let intermediate_val = builder.ins().iconst(
                    cranelift_codegen::ir::types::I64,
                    intermediate_dim as i64,
                );
                // CPDT Part III v2.4: activation_kind sourced from
                // MoeInfo.activation (the @moe(activation="…") kwarg
                // or the SiLU default). The enum's `repr(i64)` matches
                // the runtime FFI's literal switch on 0/1/2/3 — pinned
                // by `moe_activation_repr_matches_ffi_contract` in
                // moe.rs tests.
                let activation_kind_val = builder
                    .ins()
                    .iconst(cranelift_codegen::ir::types::I64, activation as i64);
                // CPDT Part III v2.12 — bias args. The runtime FFI's
                // experts_up_bias_ptr + experts_down_bias_ptr (null=0
                // means no bias). The 4-arg `moe_dispatch_ffn` form
                // emits `iconst(0)` here so v2.5/v2.7 v3 emission stays
                // byte-identical for bias-free MoE. The 6-arg form
                // threads the compiled bias expressions through the
                // FFI; v2.11's upfront-dtype + device-CPU + len gates
                // validate them at runtime entry.
                let (bias_up_arg, bias_down_arg) = match bias_vals {
                    Some((u, d)) => (u, d),
                    None => {
                        // v2.12 fix F1 (HIGH adversarial review): emit
                        // two separate `iconst(0)` Values rather than
                        // aliasing one Value into two call-arg slots.
                        // Cranelift's IR technically permits a Value to
                        // be reused across arg positions, but the
                        // verifier in debug builds has flagged this
                        // pattern in past Cranelift versions. Two
                        // iconsts cost zero (constant-folded by the
                        // backend) and eliminate the brittleness.
                        let zero_up = builder
                            .ins()
                            .iconst(cranelift_codegen::ir::types::I64, 0_i64);
                        let zero_down = builder
                            .ins()
                            .iconst(cranelift_codegen::ir::types::I64, 0_i64);
                        (zero_up, zero_down)
                    }
                };
                let result = self.compile_call_by_name(
                    builder,
                    "nsl_moe_dispatch_full_v3",
                    &[
                        tokens_val,
                        logits_val,
                        experts_up_val,
                        experts_down_val,
                        num_experts_val,
                        top_k_val,
                        cap_bits,
                        hidden_val,
                        intermediate_val,
                        activation_kind_val,
                        bias_up_arg,
                        bias_down_arg,
                    ],
                )?;
                return Ok(result);
            }

            // No silent fallback from moe_dispatch_ffn. Users that call
            // this intrinsic explicitly want v3; if v3 dims can't
            // resolve, the build fails loudly. This is stricter than
            // moe_dispatch's v1-fallback for non-CPDT — that fallback
            // exists for source-API backward compat, which isn't a
            // concern here (moe_dispatch_ffn is a new entry point).
            return Err(crate::error::CodegenError::new(format!(
                "moe_dispatch_ffn: v3 (paper-faithful FFN) requires resolvable router + \
                 experts.up + experts.down entries in the WeightMap. None of the standard \
                 names resolved under the lookup prefix{}, or their shapes were inconsistent \
                 (router must be [hidden, num_experts]; experts.up must be [num_experts, \
                 hidden * intermediate]; experts.down must be [num_experts, intermediate * \
                 hidden]; up- and down-derived intermediate dims must agree). Pass --weights \
                 with a safetensors bundle that contains <prefix>.router.weight AND \
                 <prefix>.experts.up.weight AND <prefix>.experts.down.weight (with `.weight` \
                 suffix optional). Declare `@moe(weight_prefix=\"...\")` in source to point \
                 this lookup at a non-NSL-identifier prefix (e.g., an HF Mixtral path \
                 segment with dots). For GPT-2 / OPT FFN-bias checkpoints, also include \
                 <prefix>.experts.up.bias + <prefix>.experts.down.bias and call the 6-arg \
                 form of moe_dispatch_ffn (v2.12).",
                lookup_key
                    .as_deref()
                    .map(|k| format!(" '{}'", k))
                    .unwrap_or_default(),
            )));
        }

        // M32: MoE dispatch intrinsic
        //
        // CPDT Part III v1 production-forward (M32 gap closure): when an
        // expert WeightMap entry is resolvable AND its shape implies a
        // well-formed `(hidden_dim, intermediate_dim)` split, emit
        // `nsl_moe_dispatch_full_v2` against the real expert weight tensor
        // (replaces the M32 identity skeleton at moe/ffi.rs:287). Otherwise
        // fall through to the v1 identity emission — silent-fallback to be
        // hard-erred in S4.
        if func_name == "moe_dispatch" {
            if args.len() != 3 {
                return Err(crate::error::CodegenError::new(
                    "moe_dispatch() takes 3 arguments (tokens, router_logits, experts)",
                ));
            }
            let tokens_val = self.compile_expr(builder, state, &args[0].value)?;
            let logits_val = self.compile_expr(builder, state, &args[1].value)?;
            let experts_val = self.compile_expr(builder, state, &args[2].value)?;

            // CPDT Part III v2.19 — multi-model `@moe` composition.
            // The helper picks the call site's MoE config under the
            // current method's model (`self.current_method_model_name`),
            // filtered by `"{model}."` prefix to scope the WeightMap
            // lookup identically to `cpdt_expert_prune::prune_moe_weights_in_map`.
            //
            // Multi-MoE-per-MODEL (≥2 `@moe` decorators on different
            // fields of ONE model) remains the documented v2.next
            // deferral: the helper returns `None`, the v2 fallback
            // below kicks in with default `num_experts=8`/`top_k=2`,
            // and `cfg_key=None`. Subsequent behaviour depends on
            // `cpdt_mode`: under `CpdtMode::Full` the v2-dims gate
            // below raises a hard codegen error; outside CPDT Full
            // the v1 fallback (`nsl_moe_dispatch_full`) emits the
            // M32 identity skeleton without diagnostic. Both
            // sub-cases match pre-v2.19 behaviour for the deferral.
            let config_with_key = crate::moe::resolve_decorator_config_for_call_site(
                self.current_method_model_name.as_deref(),
                &self.features.moe_configs,
            );
            let (cfg_key, num_experts, top_k, capacity_factor) = match &config_with_key {
                Some((k, info)) => {
                    (Some(k.clone()), info.num_experts, info.top_k, info.capacity_factor)
                }
                None => (None, 8, 2, 1.25f32), // defaults
            };

            let num_experts_val = builder
                .ins()
                .iconst(cranelift_codegen::ir::types::I64, num_experts as i64);
            let top_k_val = builder
                .ins()
                .iconst(cranelift_codegen::ir::types::I64, top_k as i64);
            let cap_bits = builder.ins().iconst(
                cranelift_codegen::ir::types::I64,
                capacity_factor.to_bits() as i64,
            );

            // CPDT Part III v1: derive (hidden_dim, intermediate_dim) from
            // the WeightMap router/experts shapes via the shared helper
            // `crate::moe::derive_v2_dims`. None → silent v1 fallback (S4
            // promotes that path to a hard compile error for cpdt_mode=Full).
            let v2_dims = if let (Some(key), Some(weight_map)) =
                (cfg_key.as_ref(), self.features.weight_map.as_ref())
            {
                crate::moe::derive_v2_dims(weight_map, key, num_experts)
            } else {
                None
            };

            if let Some((hidden_dim, intermediate_dim)) = v2_dims {
                let hidden_val = builder.ins().iconst(
                    cranelift_codegen::ir::types::I64,
                    hidden_dim as i64,
                );
                let intermediate_val = builder.ins().iconst(
                    cranelift_codegen::ir::types::I64,
                    intermediate_dim as i64,
                );
                let result = self.compile_call_by_name(
                    builder,
                    "nsl_moe_dispatch_full_v2",
                    &[
                        tokens_val,
                        logits_val,
                        experts_val,
                        num_experts_val,
                        top_k_val,
                        cap_bits,
                        hidden_val,
                        intermediate_val,
                    ],
                )?;
                return Ok(result);
            }

            // CPDT Part III v1 (S4) — no silent v1 fallback when CPDT is
            // active. When `cpdt_mode == Full` AND v2 emission failed, raise
            // a hard compile error so the running program does not silently
            // route through the M32 identity skeleton (which would mask a
            // broken WeightMap / mismatched router/experts shapes).
            // Non-CPDT consumers still get v1 (the M32 fallback) since they
            // have no expectation of the production-forward path.
            if matches!(self.cpdt_mode, crate::cpdt::CpdtMode::Full) {
                return Err(crate::error::CodegenError::new(format!(
                    "moe_dispatch: CPDT Full mode requires resolvable router + experts \
                     entries in the WeightMap. None of the standard names resolved under \
                     the MoeConfig key{}, or their shapes were inconsistent (router must be \
                     [hidden, num_experts]; experts must be [num_experts, hidden * \
                     intermediate]). Pass --weights with a safetensors bundle that \
                     contains <key>.router.weight (or gate.weight / router / gate) AND \
                     <key>.experts.weight (or experts).",
                    cfg_key
                        .as_deref()
                        .map(|k| format!(" '{}'", k))
                        .unwrap_or_default(),
                )));
            }

            // Non-CPDT fallback: emit v1 (M32 identity skeleton). Kept for
            // backward compatibility with consumers that don't supply a
            // WeightMap and don't activate CPDT.
            let result = self.compile_call_by_name(
                builder,
                "nsl_moe_dispatch_full",
                &[tokens_val, logits_val, num_experts_val, top_k_val, cap_bits],
            )?;

            return Ok(result);
        }

        // M33: Speculative decode step intrinsic
        if func_name == "speculative_decode" {
            // speculative_decode(draft_tokens, draft_logits, verifier_logits, vocab_size)
            // Returns NslTensor of accepted token IDs.
            if args.len() != 4 {
                return Err(crate::error::CodegenError::new(
                    "speculative_decode() takes 4 arguments (draft_tokens, draft_logits, verifier_logits, vocab_size)",
                ));
            }
            let draft_tokens = self.compile_expr(builder, state, &args[0].value)?;
            let draft_logits = self.compile_expr(builder, state, &args[1].value)?;
            let verifier_logits = self.compile_expr(builder, state, &args[2].value)?;
            let vocab_size = self.compile_expr(builder, state, &args[3].value)?;

            // CPDT Part III v2.20 — multi-model `@speculative`
            // composition. Pre-v2.20 used the same broken
            // `current_function_name.split("__").next()` pattern as the
            // MoE sites (fixed in v2.19): `current_function_name` is
            // None for model methods, so the lookup always returned
            // None and `speculative_decode` silently fell through to
            // hard-coded defaults (`temperature=0.0`, `num_tokens=5`,
            // `method=Draft`, `tree_width=4`) regardless of user
            // `@speculative(...)` values — making the decorator a no-op
            // inside any model method body.
            let config = crate::moe::resolve_decorator_config_for_call_site(
                self.current_method_model_name.as_deref(),
                &self.features.speculative_configs,
            )
            .map(|(_, info)| info);

            let (temperature, num_tokens, method_id, tree_width) = match &config {
                Some(info) => (
                    info.temperature,
                    info.num_tokens as i64,
                    match info.method {
                        crate::speculative::SpeculativeMethod::Draft => 0i64,
                        crate::speculative::SpeculativeMethod::Medusa => 1,
                        crate::speculative::SpeculativeMethod::Eagle2 => 2,
                        crate::speculative::SpeculativeMethod::Lookahead => 3,
                    },
                    info.tree_width as i64,
                ),
                None => (0.0f32, 5i64, 0i64, 4i64), // defaults
            };

            // Get num_draft_tokens from draft_tokens tensor length
            let draft_len = builder.ins().load(
                cranelift_codegen::ir::types::I64,
                cranelift_codegen::ir::MemFlags::trusted(),
                draft_tokens,
                cranelift_codegen::ir::immediates::Offset32::new(40), // NslTensor.len offset (magic:0 + pad:4 + data:8 + shape:16 + strides:24 + ndim:32 → len:40)
            );
            let temp_bits = builder.ins().iconst(
                cranelift_codegen::ir::types::I64,
                temperature.to_bits() as i64,
            );
            let num_tokens_val = builder
                .ins()
                .iconst(cranelift_codegen::ir::types::I64, num_tokens);
            let method_val = builder
                .ins()
                .iconst(cranelift_codegen::ir::types::I64, method_id);
            let tree_width_val = builder
                .ins()
                .iconst(cranelift_codegen::ir::types::I64, tree_width);

            let result = self.compile_call_by_name(
                builder,
                "nsl_speculative_decode_step",
                &[
                    draft_tokens,
                    draft_logits,
                    verifier_logits,
                    draft_len,
                    vocab_size,
                    temp_bits,
                    num_tokens_val,
                    method_val,
                    tree_width_val,
                ],
            )?;
            return Ok(result);
        }

        // M19: Sampling intrinsics

        // manual_seed(seed)
        if func_name == "manual_seed" {
            let seed_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_manual_seed", &[seed_val]);
        }

        // topk(tensor, k, dim=-1)
        if func_name == "topk" {
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let k_val = self.compile_expr(builder, state, &args[1].value)?;
            let dim_val = if args.len() > 2 {
                self.compile_expr(builder, state, &args[2].value)?
            } else {
                builder.ins().iconst(cl_types::I64, -1_i64)
            };
            return self.compile_call_by_name(
                builder,
                "nsl_tensor_topk",
                &[tensor_val, k_val, dim_val],
            );
        }

        // multinomial(tensor, num_samples)
        if func_name == "multinomial" {
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let n_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(
                builder,
                "nsl_tensor_multinomial",
                &[tensor_val, n_val],
            );
        }

        // M44b: generate(model, tokens, max_tokens=256, temperature=0.7, top_p=0.9, schema=...)
        if func_name == "generate" {
            if args.len() >= 2 {
                let model_val = self.compile_expr(builder, state, &args[0].value)?;
                let tokens_val = self.compile_expr(builder, state, &args[1].value)?;
                // Extract max_tokens (default 256), temperature (default 0.7), top_p (default 0.9)
                let max_tokens = builder.ins().iconst(cl_types::I64, 256);
                let temperature = builder.ins().f64const(0.7);
                let top_p = builder.ins().f64const(0.9);
                // Enqueue the request — nsl_serve_enqueue(prompt_ptr, prompt_len, max_tokens, temp, top_p)
                // tokens_val is used as prompt_ptr; model_val as prompt_len placeholder
                let request_id = self.compile_call_by_name(
                    builder,
                    "nsl_serve_enqueue",
                    &[tokens_val, model_val, max_tokens, temperature, top_p],
                )?;

                // M44: Check for schema= kwarg → compile grammar at codegen time → set FSM on request
                let mut schema_val: Option<String> = None;
                for arg in args {
                    if let Some(name_sym) = arg.name {
                        let name = self.resolve_sym(name_sym).to_string();
                        if name == "schema" {
                            if let nsl_ast::expr::ExprKind::StringLiteral(s) = &arg.value.kind {
                                schema_val = Some(s.clone());
                            }
                        }
                    }
                }

                if let Some(ref schema) = schema_val {
                    // Compile grammar at codegen time; pass start_state=0 (DFA start) to runtime.
                    // The runtime interprets start_state > 0 as "constrained decoding active".
                    // We use 1 to distinguish "grammar set" from "no grammar" (0).
                    let start_state = builder.ins().iconst(cl_types::I64, 1);
                    self.compile_call_by_name(
                        builder,
                        "nsl_serve_set_grammar",
                        &[request_id, start_state],
                    )?;
                    eprintln!("[nsl] Constrained decoding active: schema='{}'", schema);
                } else {
                    // Check if the current function has a model-level @grammar decorator
                    let current_fn_name = state.current_function_name.clone().unwrap_or_default();
                    if let Some(grammar_info) = self.features.grammar_configs.get(&current_fn_name)
                    {
                        if !grammar_info.grammar_source.is_empty()
                            || !grammar_info.start_rule.is_empty()
                        {
                            let src = grammar_info.grammar_source.clone();
                            let start_state = builder.ins().iconst(cl_types::I64, 1);
                            self.compile_call_by_name(
                                builder,
                                "nsl_serve_set_grammar",
                                &[request_id, start_state],
                            )?;
                            eprintln!("[nsl] Constrained decoding active via @grammar: '{}'", src);
                        }
                    }
                }

                let _ = model_val; // suppress unused warning — model used as prompt_len placeholder
                return Ok(request_id);
            }
            return Err(CodegenError::new(
                "generate() requires at least 2 arguments (model, tokens)",
            ));
        }

        // argmax(tensor, dim=-1)
        if func_name == "argmax" {
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let dim_val = if args.len() > 1 {
                self.compile_expr(builder, state, &args[1].value)?
            } else {
                builder.ins().iconst(cl_types::I64, -1_i64)
            };
            return self.compile_call_by_name(builder, "nsl_tensor_argmax", &[tensor_val, dim_val]);
        }

        // cumsum(tensor, dim)
        if func_name == "cumsum" {
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let dim_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_cumsum", &[tensor_val, dim_val]);
        }

        // lt_scalar(tensor, scalar)
        if func_name == "lt_scalar" {
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let scalar_val = self.compile_expr(builder, state, &args[1].value)?;
            // Ensure scalar is f64
            let scalar_ty = builder.func.dfg.value_type(scalar_val);
            let scalar_f64 = if scalar_ty == cl_types::F64 {
                scalar_val
            } else {
                builder.ins().fcvt_from_sint(cl_types::F64, scalar_val)
            };
            return self.compile_call_by_name(
                builder,
                "nsl_tensor_lt_scalar",
                &[tensor_val, scalar_f64],
            );
        }

        // M19: Data source intrinsics

        // load_jsonl("path", "field")
        if func_name == "load_jsonl" {
            let path_val = self.compile_expr(builder, state, &args[0].value)?;
            let path_len = if let ExprKind::StringLiteral(s) = &args[0].value.kind {
                builder.ins().iconst(cl_types::I64, s.len() as i64)
            } else {
                self.compile_call_by_name(builder, "nsl_str_len", &[path_val])?
            };
            let field_val = self.compile_expr(builder, state, &args[1].value)?;
            let field_len = if let ExprKind::StringLiteral(s) = &args[1].value.kind {
                builder.ins().iconst(cl_types::I64, s.len() as i64)
            } else {
                self.compile_call_by_name(builder, "nsl_str_len", &[field_val])?
            };
            return self.compile_call_by_name(
                builder,
                "nsl_load_jsonl",
                &[path_val, path_len, field_val, field_len],
            );
        }

        // load_csv("path", col_idx, has_header=1)
        if func_name == "load_csv" {
            let path_val = self.compile_expr(builder, state, &args[0].value)?;
            let path_len = if let ExprKind::StringLiteral(s) = &args[0].value.kind {
                builder.ins().iconst(cl_types::I64, s.len() as i64)
            } else {
                self.compile_call_by_name(builder, "nsl_str_len", &[path_val])?
            };
            let col_val = self.compile_expr(builder, state, &args[1].value)?;
            let header_val = if args.len() > 2 {
                self.compile_expr(builder, state, &args[2].value)?
            } else {
                builder.ins().iconst(cl_types::I64, 1)
            };
            return self.compile_call_by_name(
                builder,
                "nsl_load_csv",
                &[path_val, path_len, col_val, header_val],
            );
        }

        // load_mmap("path", dtype)
        if func_name == "load_mmap" {
            let path_val = self.compile_expr(builder, state, &args[0].value)?;
            let path_len = if let ExprKind::StringLiteral(s) = &args[0].value.kind {
                builder.ins().iconst(cl_types::I64, s.len() as i64)
            } else {
                self.compile_call_by_name(builder, "nsl_str_len", &[path_val])?
            };
            let dtype_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(
                builder,
                "nsl_load_mmap",
                &[path_val, path_len, dtype_val],
            );
        }

        // M19: DataLoader intrinsic

        // DataLoader(data, batch_size=32, seq_len=128, ...)
        if func_name == "DataLoader" {
            if state.flags.conditional_depth > 0
                || state.flags.in_dataloader_batch_scope
                || !state.loop_stack.is_empty()
            {
                return Err(CodegenError::new(
                    "DataLoader creation is only supported in straight-line function scope; move it out of conditionals and loops",
                ));
            }
            let data_expr = &args[0].value;
            let data_ty = self.node_type(data_expr.id).clone();
            if !data_ty.is_tensor() && !data_ty.is_indeterminate() {
                return Err(CodegenError::new(format!(
                    "DataLoader(): data must be a tensor, got {}",
                    nsl_semantic::types::display_type(&data_ty)
                )));
            }

            let data_val = self.compile_expr(builder, state, data_expr)?;
            let mut labels_arg: Option<&nsl_ast::expr::Expr> = None;

            // Build JSON config from keyword args at compile time
            let mut config = serde_json::Map::new();
            for arg in args.iter().skip(1) {
                if let Some(name_sym) = arg.name {
                    let key = self.resolve_sym(name_sym).to_string();
                    if key == "labels" {
                        if labels_arg.is_some() {
                            return Err(CodegenError::new(
                                "DataLoader(): labels provided more than once",
                            ));
                        }
                        labels_arg = Some(&arg.value);
                        continue;
                    }
                    match &arg.value.kind {
                        ExprKind::IntLiteral(v) => {
                            config.insert(
                                key,
                                serde_json::Value::Number(serde_json::Number::from(*v)),
                            );
                        }
                        ExprKind::BoolLiteral(v) => {
                            config.insert(key, serde_json::Value::Bool(*v));
                        }
                        ExprKind::FloatLiteral(v) => {
                            if let Some(n) = serde_json::Number::from_f64(*v) {
                                config.insert(key, serde_json::Value::Number(n));
                            }
                        }
                        _ => {
                            return Err(CodegenError::new(format!(
                                "DataLoader(): keyword argument '{}' must be a compile-time constant", key
                            )));
                        }
                    }
                } else if labels_arg.is_none() {
                    labels_arg = Some(&arg.value);
                } else {
                    return Err(CodegenError::new(
                        "DataLoader(): only one optional labels tensor is allowed after data",
                    ));
                }
            }

            // pin_memory is accepted but currently a no-op (actual memory
            // pinning deferred to tensor allocator integration).

            // Default drop_last to true when unspecified.
            if config.get("drop_last").and_then(|v| v.as_bool()).is_none() {
                config.insert("drop_last".to_string(), serde_json::Value::Bool(true));
            }

            // Packing is accepted; the runtime builds packed batches with
            // attention_mask. Model attention must read the mask if present.

            if let Some(labels_expr) = labels_arg {
                let labels_ty = self.node_type(labels_expr.id).clone();
                if matches!(labels_ty.strip_borrow(), Type::NoneType) {
                    labels_arg = None;
                } else if !labels_ty.is_tensor() && !labels_ty.is_indeterminate() {
                    return Err(CodegenError::new(format!(
                        "DataLoader(): labels must be a tensor, got {}",
                        nsl_semantic::types::display_type(&labels_ty)
                    )));
                } else if let (Some(data_numel), Some(labels_numel)) = (
                    static_tensor_numel(&data_ty),
                    static_tensor_numel(&labels_ty),
                ) {
                    if data_numel != labels_numel {
                        return Err(CodegenError::new(format!(
                            "DataLoader(): data and labels must have the same number of elements ({} vs {})",
                            data_numel, labels_numel
                        )));
                    }
                }
            }

            let config_json = serde_json::Value::Object(config).to_string();

            // Intern the config string as data
            let config_data_id = self.intern_string(&config_json)?;
            let config_gv = self
                .module
                .declare_data_in_func(config_data_id, builder.func);
            let config_ptr = builder.ins().symbol_value(pointer_type(), config_gv);
            let config_len = builder
                .ins()
                .iconst(cl_types::I64, config_json.len() as i64);

            // Normalize loader inputs to CPU-contiguous tensors and let the runtime keep them alive.
            let cpu_device = builder.ins().iconst(cl_types::I64, 0);
            let data_cpu = self.compile_call_by_name(
                builder,
                "nsl_tensor_to_device",
                &[data_val, cpu_device],
            )?;
            let data_tensor =
                self.compile_call_by_name(builder, "nsl_tensor_contiguous", &[data_cpu])?;
            self.compile_call_by_name(builder, "nsl_tensor_free", &[data_cpu])?;

            let zero = builder.ins().iconst(cl_types::I64, 0);
            let labels_tensor = if let Some(labels_expr) = labels_arg {
                let labels_val = self.compile_expr(builder, state, labels_expr)?;
                let labels_cpu = self.compile_call_by_name(
                    builder,
                    "nsl_tensor_to_device",
                    &[labels_val, cpu_device],
                )?;
                let labels_tensor =
                    self.compile_call_by_name(builder, "nsl_tensor_contiguous", &[labels_cpu])?;
                self.compile_call_by_name(builder, "nsl_tensor_free", &[labels_cpu])?;
                labels_tensor
            } else {
                zero
            };

            let dl_ptr = self.compile_call_by_name(
                builder,
                "nsl_dataloader_create",
                &[data_tensor, labels_tensor, config_ptr, config_len],
            )?;
            self.compile_call_by_name(builder, "nsl_dataloader_start", &[dl_ptr])?;
            state.cleanup.dataloader_vars.push(dl_ptr);
            return Ok(dl_ptr);
        }

        // Check if it's a known function or variable holding a function pointer
        if self.registry.functions.contains_key(&func_name)
            || self.registry.runtime_fns.contains_key(&func_name)
        {
            let mut arg_vals = Vec::new();
            for arg in args {
                arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
            }
            let result = self.compile_call_by_name(builder, &func_name, &arg_vals)?;
            self.register_ffi_result_ownership(
                builder,
                state,
                &func_name,
                result,
                &arg_vals,
            );
            return Ok(result);
        }

        // Forward dispatch: if callee is a model instance, invoke its forward method
        let callee_type = self.node_type(callee.id).clone();
        if let Type::Model { name, .. } = &callee_type {
            let model_name = self.resolve_sym(*name).to_string();
            return self.compile_model_method_call(
                builder,
                state,
                callee,
                &model_name,
                "forward",
                args,
            );
        }

        // Fall through to indirect call for variables holding function pointers
        self.compile_indirect_call(builder, state, callee, args)
    }

    pub(crate) fn compile_call_by_name(
        &mut self,
        builder: &mut FunctionBuilder,
        func_name: &str,
        arg_vals: &[Value],
    ) -> Result<Value, CodegenError> {
        // M39c: Vmap dispatch — check if function has a compiled batched variant.
        // Since @vmap is an explicit opt-in decorator, we dispatch to the batched
        // variant whenever it has been successfully compiled.  The VmapTransformer
        // has already validated that the function body is batchable during the
        // transform pass.  If compilation of the batched variant failed (non-fatal),
        // the function won't be in self.registry.functions and we fall through to the original.
        let effective_name: &str =
            if let Some(batched_name) = self.vmap.batched_fn_names.get(func_name) {
                if self.registry.functions.contains_key(batched_name.as_str()) {
                    // Batched variant exists and is compiled — dispatch to it.
                    batched_name.as_str()
                } else {
                    // Batched variant declared but not compiled — use original
                    func_name
                }
            } else {
                func_name
            };

        let (func_id, sig) = if let Some(e) = self.registry.functions.get(effective_name) {
            e.clone()
        } else if let Some(e) = self.registry.runtime_fns.get(effective_name) {
            e.clone()
        } else if let Some(alias) = tensor_unary_runtime_alias(effective_name, arg_vals.len()) {
            self.registry
                .runtime_fns
                .get(alias)
                .cloned()
                .ok_or_else(|| CodegenError::new(format!("undefined function '{effective_name}'")))?
        } else {
            return Err(CodegenError::new(format!(
                "undefined function '{effective_name}'"
            )));
        };

        let func_ref = self.module.declare_func_in_func(func_id, builder.func);
        let call = builder.ins().call(func_ref, arg_vals);
        if sig.returns.is_empty() {
            Ok(builder.ins().iconst(cl_types::I64, 0))
        } else {
            Ok(builder.inst_results(call)[0])
        }
    }

    /// Look up a custom dtype name in the registry and return its numeric id.
    pub(crate) fn resolve_custom_dtype(&self, name: &str) -> Option<u16> {
        self.types.custom_dtype_ids.get(name).copied()
    }

    /// ELTLS: register the ownership of a tensor value produced by a known
    /// runtime FFI. Consults the ffi_ownership table and routes the result
    /// through set_ownership / Borrowed propagation / NotATensor skip.
    /// Un-listed FFIs bump the unknown-ownership counter for rollout
    /// measurement (spec §6.2).
    pub(crate) fn register_ffi_result_ownership(
        &self,
        builder: &FunctionBuilder,
        state: &mut FuncState,
        fn_name: &str,
        result: Value,
        arg_vals: &[Value],
    ) {
        use crate::ffi_ownership::{ffi_ownership_kind, FfiOwnershipKind};
        use crate::ownership_expr::Ownership;

        // Only consult the table for runtime FFIs; user-defined functions
        // have their ownership resolved at the callee's return site.
        if !self.registry.runtime_fns.contains_key(fn_name) {
            return;
        }
        match ffi_ownership_kind(fn_name) {
            Some(FfiOwnershipKind::OwnedNewResult) => {
                self.set_ownership(builder, state, result, Ownership::Owned);
            }
            Some(FfiOwnershipKind::BorrowedFromInput(idx)) => {
                if let Some(&src_val) = arg_vals.get(idx) {
                    let src_own = self.get_ownership(state, src_val);
                    self.set_ownership(builder, state, result, src_own);
                }
            }
            Some(FfiOwnershipKind::NotATensor) => {
                // Scalar result — no tracking.
            }
            None => {
                // Un-listed FFI — record the fallback hit so the
                // instrumentation counter surfaces migration gaps.
                self.note_unknown_fallback(state, result);
            }
        }
    }

    pub(crate) fn compile_print_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        if args.is_empty() {
            let empty = self.compile_string_literal(builder, "")?;
            let fid = self.registry.runtime_fns["nsl_print_str"].0;
            let fref = self.module.declare_func_in_func(fid, builder.func);
            builder.ins().call(fref, &[empty]);
            return Ok(builder.ins().iconst(cl_types::I64, 0));
        }

        let arg = &args[0];
        let val = self.compile_expr(builder, state, &arg.value)?;
        let arg_type = self.node_type(arg.value.id).clone();

        let rt_fn = match &arg_type {
            Type::Int | Type::Int64 | Type::Int32 | Type::Int16 | Type::Int8 => "nsl_print_int",
            Type::Float | Type::F64 | Type::F32 => "nsl_print_float",
            Type::Str => "nsl_print_str",
            Type::Bool => "nsl_print_bool",
            t if t.is_tensor() => "nsl_tensor_print",
            _ => {
                // For Unknown types, check the Cranelift value type
                let cl_type = builder.func.dfg.value_type(val);
                if cl_type == cl_types::F64 {
                    "nsl_print_float"
                } else {
                    "nsl_print_int"
                }
            }
        };

        // Widen sub-I64 integer values for nsl_print_int (expects I64),
        // or widen F32 for nsl_print_float (expects F64)
        let print_val = match rt_fn {
            "nsl_print_int" => {
                let vt = builder.func.dfg.value_type(val);
                if vt.bits() < 64 && vt.is_int() {
                    builder.ins().sextend(cl_types::I64, val)
                } else {
                    val
                }
            }
            "nsl_print_float" => {
                let vt = builder.func.dfg.value_type(val);
                if vt == cl_types::F32 {
                    builder.ins().fpromote(cl_types::F64, val)
                } else {
                    val
                }
            }
            _ => val,
        };
        let fid = self.registry.runtime_fns[rt_fn].0;
        let fref = self.module.declare_func_in_func(fid, builder.func);
        builder.ins().call(fref, &[print_val]);
        Ok(builder.ins().iconst(cl_types::I64, 0))
    }

    pub(crate) fn compile_range_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        let (start, stop, step) = match args.len() {
            1 => {
                let stop = self.compile_expr(builder, state, &args[0].value)?;
                let start = builder.ins().iconst(cl_types::I64, 0);
                let step = builder.ins().iconst(cl_types::I64, 1);
                (start, stop, step)
            }
            2 => {
                let start = self.compile_expr(builder, state, &args[0].value)?;
                let stop = self.compile_expr(builder, state, &args[1].value)?;
                let step = builder.ins().iconst(cl_types::I64, 1);
                (start, stop, step)
            }
            3 => {
                let start = self.compile_expr(builder, state, &args[0].value)?;
                let stop = self.compile_expr(builder, state, &args[1].value)?;
                let step = self.compile_expr(builder, state, &args[2].value)?;
                (start, stop, step)
            }
            _ => return Err(CodegenError::new("range() takes 1-3 arguments")),
        };
        let fid = self.registry.runtime_fns["nsl_range"].0;
        let fref = self.module.declare_func_in_func(fid, builder.func);
        let call = builder.ins().call(fref, &[start, stop, step]);
        Ok(builder.inst_results(call)[0])
    }

    pub(crate) fn compile_indirect_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        callee: &Expr,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        let callee_type = self.node_type(callee.id).clone();

        // Check if callee is a closure variable (has captures)
        let closure_captures = if let ExprKind::Ident(sym) = &callee.kind {
            self.registry.closure_info.get(sym).copied()
        } else {
            None
        };

        let raw_val = self.compile_expr(builder, state, callee)?;

        if let Type::Function {
            params: param_types,
            ret,
            ..
        } = &callee_type
        {
            if let Some(num_captures) = closure_captures {
                // Closure call: raw_val is a pointer to { fn_ptr, num_captures, captures[] }
                let closure_ptr = raw_val;

                // Load fn_ptr from offset 0
                let fn_ptr = builder
                    .ins()
                    .load(cl_types::I64, MemFlags::trusted(), closure_ptr, 0);

                // Build signature: normal params + capture params (all I64)
                let mut sig = self.module.make_signature();
                sig.call_conv = self.call_conv;
                for pt in param_types {
                    sig.params.push(AbiParam::new(nsl_type_to_cl(pt)));
                }
                for _ in 0..num_captures {
                    sig.params.push(AbiParam::new(cl_types::I64));
                }
                let is_void = matches!(ret.as_ref(), Type::Void);
                if !is_void {
                    sig.returns.push(AbiParam::new(nsl_type_to_cl(ret)));
                }
                let sig_ref = builder.import_signature(sig);

                // Compile normal args
                let mut arg_vals = Vec::new();
                for arg in args {
                    arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
                }
                // Load captured values from closure struct
                for i in 0..num_captures {
                    let offset = (16 + i * 8) as i32;
                    let cap_val =
                        builder
                            .ins()
                            .load(cl_types::I64, MemFlags::trusted(), closure_ptr, offset);
                    arg_vals.push(cap_val);
                }

                let call = builder.ins().call_indirect(sig_ref, fn_ptr, &arg_vals);
                if is_void {
                    return Ok(builder.ins().iconst(cl_types::I64, 0));
                }
                return Ok(builder.inst_results(call)[0]);
            } else {
                // Bare function pointer call (non-capturing lambda)
                let fn_ptr = raw_val;
                let mut sig = self.module.make_signature();
                sig.call_conv = self.call_conv;
                for pt in param_types {
                    sig.params.push(AbiParam::new(nsl_type_to_cl(pt)));
                }
                let is_void = matches!(ret.as_ref(), Type::Void);
                if !is_void {
                    sig.returns.push(AbiParam::new(nsl_type_to_cl(ret)));
                }
                let sig_ref = builder.import_signature(sig);

                let mut arg_vals = Vec::new();
                for arg in args {
                    arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
                }
                let call = builder.ins().call_indirect(sig_ref, fn_ptr, &arg_vals);
                if is_void {
                    return Ok(builder.ins().iconst(cl_types::I64, 0));
                }
                return Ok(builder.inst_results(call)[0]);
            }
        }

        Err(CodegenError::new(format!(
            "cannot call expression of type {:?}",
            callee_type
        )))
    }

    pub(crate) fn compile_kernel_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        kernel_name: &str,
        args: &[nsl_ast::expr::Arg],
        ptx_data_id: cranelift_module::DataId,
        name_data_id: cranelift_module::DataId,
        origin_node: nsl_ast::NodeId,
        span: nsl_errors::Span,
    ) -> Result<Value, CodegenError> {
        // Separate positional args (tensor params) from named args (grid, block)
        let mut tensor_args: Vec<Value> = Vec::new();
        let mut grid_val: Option<Value> = None;
        let mut block_val: Option<Value> = None;

        for arg in args {
            if let Some(name_sym) = arg.name {
                let name = self.resolve_sym(name_sym).to_string();
                match name.as_str() {
                    "grid" => {
                        grid_val = Some(self.compile_expr(builder, state, &arg.value)?);
                    }
                    "block" => {
                        block_val = Some(self.compile_expr(builder, state, &arg.value)?);
                    }
                    _ => {
                        tensor_args.push(self.compile_expr(builder, state, &arg.value)?);
                    }
                }
            } else {
                tensor_args.push(self.compile_expr(builder, state, &arg.value)?);
            }
        }

        let grid_x = grid_val.unwrap_or_else(|| builder.ins().iconst(cl_types::I64, 1));
        let grid_y = builder.ins().iconst(cl_types::I64, 1);
        let grid_z = builder.ins().iconst(cl_types::I64, 1);
        let block_x = block_val.unwrap_or_else(|| builder.ins().iconst(cl_types::I64, 256));
        let block_y = builder.ins().iconst(cl_types::I64, 1);
        let block_z = builder.ins().iconst(cl_types::I64, 1);

        // Get PTX pointer from .rodata
        let ptx_gv = self.module.declare_data_in_func(ptx_data_id, builder.func);
        let ptx_ptr = builder.ins().symbol_value(pointer_type(), ptx_gv);

        // Get kernel name pointer from .rodata
        let name_gv = self.module.declare_data_in_func(name_data_id, builder.func);
        let name_ptr = builder.ins().symbol_value(pointer_type(), name_gv);

        // Dev Tools Phase 2, Task 5: reserve a kernel_id and emit
        // `nsl_profile_kernel_begin` / `_end` around the launch when the
        // profile pre-pass populated `manifest_builder`.  When profile_kernels
        // is disabled, `kernel_id` stays `None` and codegen is byte-identical
        // to the pre-Phase-2 path.
        let kernel_id = if self.manifest_builder.is_some() {
            // Resolve fusion constituents and aggregate predicted cost
            // BEFORE taking a mutable borrow of `manifest_builder`.
            let constituents = self.fusion_constituents(origin_node);
            let mut total_us = 0.0_f64;
            let mut total_flops = 0u64;
            let mut total_hbm = 0u64;
            for nid in &constituents {
                if let Some(p) = self.prediction_map.get(nid) {
                    total_us += p.estimated_time_us;
                    total_flops = total_flops.saturating_add(p.flops);
                    total_hbm = total_hbm
                        .saturating_add(p.bytes_read)
                        .saturating_add(p.bytes_written);
                }
            }
            let span_json = crate::profiling::instrument::SourceSpanJson::from_span(
                span,
                &self.source_file_name,
                &self.source_text,
            );
            // Split the is_some / as_mut so the earlier immutable uses of
            // `self` don't collide with the mutable borrow — the `expect`
            // form matches that control-flow shape more clearly than an
            // `if let Some(...)` refactor here.
            #[allow(clippy::unnecessary_unwrap)]
            let mb = self
                .manifest_builder
                .as_mut()
                .expect("manifest_builder presence checked above");
            let id = mb.reserve_id();
            mb.record_kernel_at(id, kernel_name, span_json, total_us, total_flops, total_hbm);
            Some(id)
        } else {
            None
        };

        if let Some(id) = kernel_id {
            let id_val = builder.ins().iconst(cl_types::I32, id as i64);
            self.compile_call_by_name(builder, "nsl_profile_kernel_begin", &[id_val])?;
        }

        // Build args array on stack: each tensor arg is an i64 (pointer)
        let num_args = tensor_args.len();
        let launch_result = if num_args == 0 {
            // No tensor args -- pass null pointer and 0 count
            let null_ptr = builder.ins().iconst(cl_types::I64, 0);
            let num_args_val = builder.ins().iconst(cl_types::I64, 0);
            let shared_mem = builder.ins().iconst(cl_types::I64, 0);
            self.compile_call_by_name(
                builder,
                "nsl_kernel_launch",
                &[
                    ptx_ptr,
                    name_ptr,
                    grid_x,
                    grid_y,
                    grid_z,
                    block_x,
                    block_y,
                    block_z,
                    null_ptr,
                    num_args_val,
                    shared_mem,
                ],
            )?
        } else {
            let slot_size = (num_args * 8) as u32;
            let ss = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
                cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                slot_size,
                8,
            ));

            for (i, arg_val) in tensor_args.iter().enumerate() {
                let offset = (i * 8) as i32;
                builder.ins().stack_store(*arg_val, ss, offset);
            }

            let args_ptr = builder.ins().stack_addr(cl_types::I64, ss, 0);
            let num_args_val = builder.ins().iconst(cl_types::I64, num_args as i64);
            let shared_mem = builder.ins().iconst(cl_types::I64, 0);

            // Call nsl_kernel_launch(ptx_ptr, name_ptr, grid_x, grid_y, grid_z,
            //                        block_x, block_y, block_z, args_ptr, num_args,
            //                        shared_mem_bytes)
            self.compile_call_by_name(
                builder,
                "nsl_kernel_launch",
                &[
                    ptx_ptr,
                    name_ptr,
                    grid_x,
                    grid_y,
                    grid_z,
                    block_x,
                    block_y,
                    block_z,
                    args_ptr,
                    num_args_val,
                    shared_mem,
                ],
            )?
        };

        if let Some(id) = kernel_id {
            let id_val = builder.ins().iconst(cl_types::I32, id as i64);
            self.compile_call_by_name(builder, "nsl_profile_kernel_end", &[id_val])?;
        }

        Ok(launch_result)
    }
}
