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
                return self.compile_call_by_name(builder, &member_name, &arg_vals);
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
                return self.compile_model_method_call(builder, state, object, &model_name, &member_name, args);
            }
            // Fallback for model array loop variables (type is Unknown but var was bound from model array)
            if matches!(obj_type, Type::Unknown) {
                if let ExprKind::Ident(obj_sym) = &object.kind {
                    if let Some(model_name) = self.models.model_var_types.get(obj_sym).cloned() {
                        return self.compile_model_method_call(builder, state, object, &model_name, &member_name, args);
                    }
                }
                eprintln!(
                    "[nsl-codegen] warning: method '.{member_name}()' called on Unknown-typed object \
                     — defaulting to tensor dispatch. This may indicate a type inference gap."
                );
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
                return Err(CodegenError::new("matmul rewrite: expected at least 2 arguments"));
            }
            let lhs = self.compile_expr(builder, state, &args[0].value)?;
            let rhs = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_traced_call(builder, rt_name, &[lhs, rhs]);
        }

        // Check if this is a kernel call (compiled GPU kernel)
        if let Some((ptx_data_id, name_data_id)) = self.kernels.kernel_ptx_data.get(&func_name).cloned() {
            return self.compile_kernel_call(builder, state, &func_name.clone(), args, ptx_data_id, name_data_id);
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
        if matches!(func_name.as_str(),
            "kv_cache_init" | "kv_cache_init_gpu" |
            "kv_cache_alloc_seq" | "kv_cache_append" |
            "kv_cache_k_ptr" | "kv_cache_v_ptr" |
            "kv_cache_free_seq" | "kv_cache_seq_len" |
            "kv_cache_seq_blocks" | "kv_cache_seq_num_blocks" |
            "kv_cache_utilization" | "kv_cache_destroy" |
            "profiler_start" | "profiler_stop" | "profiler_peak" |
            "kernel_profiler_start" | "kernel_profiler_stop")
        {
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
        if matches!(func_name.as_str(), "enumerate" | "zip" | "sorted" | "reversed") {
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
                return Err(CodegenError::new(format!("{func_name}() takes exactly 1 argument")));
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
                        Type::Int32 | Type::Int16 | Type::Int8 => builder.ins().sextend(cl_types::I64, val),
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
                let rt_name = if is_float_type(&arg_type) { "nsl_abs_float" } else { "nsl_abs_int" };
                return self.compile_call_by_name(builder, rt_name, &[val]);
            }
            // else: tensor type -- fall through to M14 tensor dispatch
        }
        // floor: scalar f64 dispatch
        if func_name == "floor" {
            if args.len() != 1 {
                return Err(CodegenError::new("floor() takes exactly 1 argument"));
            }
            let val = self.compile_expr(builder, state, &args[0].value)?;
            let arg_type = self.node_type(args[0].value.id).clone();
            let float_val = if is_float_type(&arg_type) {
                if matches!(arg_type, Type::F32) {
                    builder.ins().fpromote(cl_types::F64, val)
                } else {
                    val
                }
            } else {
                let widened = match &arg_type {
                    Type::Int32 | Type::Int16 | Type::Int8 => builder.ins().sextend(cl_types::I64, val),
                    _ => val,
                };
                builder.ins().fcvt_from_sint(cl_types::F64, widened)
            };
            return self.compile_call_by_name(builder, "nsl_floor", &[float_val]);
        }
        // min/max: dispatch to int or float variant
        if matches!(func_name.as_str(), "min" | "max") {
            if args.len() != 2 {
                return Err(CodegenError::new(format!("{func_name}() takes exactly 2 arguments")));
            }
            let a = self.compile_expr(builder, state, &args[0].value)?;
            let b = self.compile_expr(builder, state, &args[1].value)?;
            let a_type = self.node_type(args[0].value.id).clone();
            let b_type = self.node_type(args[1].value.id).clone();
            let is_float = is_float_type(&a_type) || is_float_type(&b_type);
            let rt_name = format!("nsl_{}_{}", func_name, if is_float { "float" } else { "int" });
            if is_float {
                let a_f = if matches!(a_type, Type::F32) {
                    builder.ins().fpromote(cl_types::F64, a)
                } else if is_float_type(&a_type) {
                    a
                } else {
                    let w = match &a_type {
                        Type::Int32 | Type::Int16 | Type::Int8 => builder.ins().sextend(cl_types::I64, a),
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
                        Type::Int32 | Type::Int16 | Type::Int8 => builder.ins().sextend(cl_types::I64, b),
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
                let a_f = if is_float_type(&a_type) { a } else { builder.ins().fcvt_from_sint(cl_types::F64, a) };
                let b_f = if is_float_type(&b_type) { b } else { builder.ins().fcvt_from_sint(cl_types::F64, b) };
                return self.compile_call_by_name(builder, "nsl_assert_eq_float", &[a_f, b_f, msg_ptr, msg_len]);
            }
            return self.compile_call_by_name(builder, "nsl_assert_eq_int", &[a, b, msg_ptr, msg_len]);
        }
        // assert_close(a, b, rtol, atol)
        if func_name == "assert_close" {
            if args.len() != 4 {
                return Err(CodegenError::new("assert_close() takes exactly 4 arguments (tensor, tensor, rtol, atol)"));
            }
            let a = self.compile_expr(builder, state, &args[0].value)?;
            let b = self.compile_expr(builder, state, &args[1].value)?;
            let rtol = self.compile_expr(builder, state, &args[2].value)?;
            let atol = self.compile_expr(builder, state, &args[3].value)?;

            // Coerce rtol/atol to f64 if they are int
            let rtol_type = self.node_type(args[2].value.id).clone();
            let atol_type = self.node_type(args[3].value.id).clone();
            let rtol_f = if is_float_type(&rtol_type) { rtol } else { builder.ins().fcvt_from_sint(cl_types::F64, rtol) };
            let atol_f = if is_float_type(&atol_type) { atol } else { builder.ins().fcvt_from_sint(cl_types::F64, atol) };

            let msg_str = "assert_close";
            self.intern_string(msg_str)?;
            let msg_ptr = self.compile_string_literal(builder, msg_str)?;
            let msg_len = builder.ins().iconst(cl_types::I64, msg_str.len() as i64);

            return self.compile_call_by_name(builder, "nsl_assert_close", &[a, b, rtol_f, atol_f, msg_ptr, msg_len]);
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
        if matches!(func_name.as_str(), "read_file" | "write_file" | "append_file" | "file_exists") {
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
                return Err(CodegenError::new("set_training_mode() takes exactly 1 argument (bool)"));
            }
            let mode_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_set_training_mode", &[mode_val]);
        }
        // Tensor creation builtins
        if matches!(func_name.as_str(), "zeros" | "ones" | "rand" | "randn") {
            if args.len() != 1 {
                return Err(CodegenError::new(format!("{func_name}() takes exactly 1 argument (shape list)")));
            }
            let shape_val = self.compile_expr(builder, state, &args[0].value)?;
            let rt_name = format!("nsl_tensor_{func_name}");
            return self.compile_call_by_name(builder, &rt_name, &[shape_val]);
        }
        if func_name == "full" {
            if args.len() != 2 {
                return Err(CodegenError::new("full() takes exactly 2 arguments (shape, value)"));
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
                return Err(CodegenError::new("arange() takes 1-3 arguments (stop) or (start, stop[, step])"));
            }
            let (start, stop, step) = match args.len() {
                1 => {
                    let stop_val = self.compile_expr(builder, state, &args[0].value)?;
                    let stop_type = self.node_type(args[0].value.id).clone();
                    let stop_f = if is_float_type(&stop_type) { stop_val } else { builder.ins().fcvt_from_sint(cl_types::F64, stop_val) };
                    let zero = builder.ins().f64const(0.0);
                    let one = builder.ins().f64const(1.0);
                    (zero, stop_f, one)
                }
                2 => {
                    let start_val = self.compile_expr(builder, state, &args[0].value)?;
                    let start_type = self.node_type(args[0].value.id).clone();
                    let start_f = if is_float_type(&start_type) { start_val } else { builder.ins().fcvt_from_sint(cl_types::F64, start_val) };
                    let stop_val = self.compile_expr(builder, state, &args[1].value)?;
                    let stop_type = self.node_type(args[1].value.id).clone();
                    let stop_f = if is_float_type(&stop_type) { stop_val } else { builder.ins().fcvt_from_sint(cl_types::F64, stop_val) };
                    let one = builder.ins().f64const(1.0);
                    (start_f, stop_f, one)
                }
                3 => {
                    let start_val = self.compile_expr(builder, state, &args[0].value)?;
                    let start_type = self.node_type(args[0].value.id).clone();
                    let start_f = if is_float_type(&start_type) { start_val } else { builder.ins().fcvt_from_sint(cl_types::F64, start_val) };
                    let stop_val = self.compile_expr(builder, state, &args[1].value)?;
                    let stop_type = self.node_type(args[1].value.id).clone();
                    let stop_f = if is_float_type(&stop_type) { stop_val } else { builder.ins().fcvt_from_sint(cl_types::F64, stop_val) };
                    let step_val = self.compile_expr(builder, state, &args[2].value)?;
                    let step_type = self.node_type(args[2].value.id).clone();
                    let step_f = if is_float_type(&step_type) { step_val } else { builder.ins().fcvt_from_sint(cl_types::F64, step_val) };
                    (start_f, stop_f, step_f)
                }
                _ => unreachable!(),
            };
            return self.compile_call_by_name(builder, "nsl_tensor_arange", &[start, stop, step]);
        }

        // Element-wise tensor builtins (M14)
        // Skip if there's a user-defined function with the same name (e.g., nsl.math.sign)
        if matches!(func_name.as_str(), "exp" | "log" | "sqrt" | "abs" | "sign" | "neg")
            && !self.registry.functions.contains_key(&func_name)
        {
            if args.len() != 1 {
                return Err(CodegenError::new(format!("{func_name}() takes exactly 1 argument")));
            }
            let val = self.compile_expr(builder, state, &args[0].value)?;
            // FBIP Phase 2: emit inplace variant when arg is single-use
            let rt_name = self.fbip_select_variant(state, &args[0].value, &func_name);
            return self.compile_traced_call(builder, &rt_name, &[val]);
        }
        // Activation functions (M15): relu, gelu, silu, sigmoid -- single tensor arg
        if matches!(func_name.as_str(), "relu" | "gelu" | "silu" | "sigmoid")
            && !self.registry.functions.contains_key(&func_name)
        {
            if args.len() != 1 {
                return Err(CodegenError::new(format!("{func_name}() takes exactly 1 argument")));
            }
            let val = self.compile_expr(builder, state, &args[0].value)?;
            // FBIP Phase 2: emit inplace variant when arg is single-use
            let rt_name = self.fbip_select_variant(state, &args[0].value, &func_name);
            return self.compile_traced_call(builder, &rt_name, &[val]);
        }
        // Tensor trig: tensor_sin, tensor_cos (RoPE support)
        if matches!(func_name.as_str(), "tensor_sin" | "tensor_cos")
            && !self.registry.functions.contains_key(&func_name)
        {
            if args.len() != 1 {
                return Err(CodegenError::new(format!("{func_name}() takes exactly 1 argument")));
            }
            let val = self.compile_expr(builder, state, &args[0].value)?;
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
                return Err(CodegenError::new("softmax() takes exactly 2 arguments (tensor, dim)"));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let dim_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_traced_call(builder, "nsl_tensor_softmax", &[tensor_val, dim_val]);
        }
        // log_softmax(tensor, dim) -- two args
        if func_name == "log_softmax" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new("log_softmax() takes exactly 2 arguments (tensor, dim)"));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let dim_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_traced_call(builder, "nsl_tensor_logsoftmax", &[tensor_val, dim_val]);
        }
        if func_name == "clamp" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 3 {
                return Err(CodegenError::new("clamp() takes exactly 3 arguments (tensor, min, max)"));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let min_val = self.compile_expr(builder, state, &args[1].value)?;
            let max_val = self.compile_expr(builder, state, &args[2].value)?;
            // Ensure min/max are f64
            let min_type = self.node_type(args[1].value.id).clone();
            let min_f = if is_float_type(&min_type) { min_val } else { builder.ins().fcvt_from_sint(cl_types::F64, min_val) };
            let max_type = self.node_type(args[2].value.id).clone();
            let max_f = if is_float_type(&max_type) { max_val } else { builder.ins().fcvt_from_sint(cl_types::F64, max_val) };
            return self.compile_call_by_name(builder, "nsl_tensor_clamp", &[tensor_val, min_f, max_f]);
        }
        if func_name == "copy_data" {
            if args.len() != 2 {
                return Err(CodegenError::new("copy_data() takes exactly 2 arguments (dest, src)"));
            }
            let dest = self.compile_expr(builder, state, &args[0].value)?;
            let src = self.compile_expr(builder, state, &args[1].value)?;
            self.compile_call_by_name(builder, "nsl_tensor_copy_data", &[dest, src])?;
            return Ok(dest);
        }
        if func_name == "add_inplace" {
            if args.len() != 2 {
                return Err(CodegenError::new("add_inplace() takes exactly 2 arguments (dest, src)"));
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
                return Err(CodegenError::new("reduce_max() takes exactly 3 arguments (tensor, dim, keepdim)"));
            }
            let t = self.compile_expr(builder, state, &args[0].value)?;
            let dim = self.compile_expr(builder, state, &args[1].value)?;
            let keepdim = self.compile_expr(builder, state, &args[2].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_reduce_max", &[t, dim, keepdim]);
        }
        if func_name == "gather" {
            if args.len() != 3 {
                return Err(CodegenError::new("gather() takes exactly 3 arguments (tensor, dim, indices)"));
            }
            let t = self.compile_expr(builder, state, &args[0].value)?;
            let dim = self.compile_expr(builder, state, &args[1].value)?;
            let indices = self.compile_expr(builder, state, &args[2].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_gather", &[t, dim, indices]);
        }
        // layernorm(input, weight, bias, eps) -> tensor
        if func_name == "layernorm" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 4 {
                return Err(CodegenError::new("layernorm() takes exactly 4 arguments (input, weight, bias, eps)"));
            }
            let input_val = self.compile_expr(builder, state, &args[0].value)?;
            let weight_val = self.compile_expr(builder, state, &args[1].value)?;
            let bias_val = self.compile_expr(builder, state, &args[2].value)?;
            let eps_val = self.compile_expr(builder, state, &args[3].value)?;
            // Ensure eps is f64
            let eps_type = self.node_type(args[3].value.id).clone();
            let eps_f = if is_float_type(&eps_type) { eps_val } else { builder.ins().fcvt_from_sint(cl_types::F64, eps_val) };
            return self.compile_call_by_name(builder, "nsl_tensor_layernorm", &[input_val, weight_val, bias_val, eps_f]);
        }
        // rmsnorm(input, weight, eps) -> tensor
        if func_name == "rmsnorm" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 3 {
                return Err(CodegenError::new("rmsnorm() takes exactly 3 arguments (input, weight, eps)"));
            }
            let input_val = self.compile_expr(builder, state, &args[0].value)?;
            let weight_val = self.compile_expr(builder, state, &args[1].value)?;
            let eps_val = self.compile_expr(builder, state, &args[2].value)?;
            let eps_type = self.node_type(args[2].value.id).clone();
            let eps_f = if is_float_type(&eps_type) { eps_val } else { builder.ins().fcvt_from_sint(cl_types::F64, eps_val) };
            return self.compile_call_by_name(builder, "nsl_tensor_rmsnorm", &[input_val, weight_val, eps_f]);
        }
        // dropout(tensor, p, training) -> tensor
        if func_name == "dropout" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 3 {
                return Err(CodegenError::new("dropout() takes exactly 3 arguments (tensor, p, training)"));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let p_val = self.compile_expr(builder, state, &args[1].value)?;
            let p_type = self.node_type(args[1].value.id).clone();
            let p_f = if is_float_type(&p_type) { p_val } else { builder.ins().fcvt_from_sint(cl_types::F64, p_val) };
            let training_val = self.compile_expr(builder, state, &args[2].value)?;
            let training_i8 = {
                let vt = builder.func.dfg.value_type(training_val);
                if vt == cl_types::I8 { training_val } else { builder.ins().ireduce(cl_types::I8, training_val) }
            };
            return self.compile_call_by_name(builder, "nsl_tensor_dropout", &[tensor_val, p_f, training_i8]);
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
            return self.compile_call_by_name(builder, "nsl_tensor_conv2d", &[input_val, weight_val, bias_val, stride_h, stride_w, pad_h, pad_w]);
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
            return self.compile_call_by_name(builder, "nsl_tensor_maxpool2d", &[input_val, kernel_h, kernel_w, stride_val, padding_val]);
        }
        // embedding_lookup(weight, indices) -> tensor
        if func_name == "embedding_lookup" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new("embedding_lookup() takes exactly 2 arguments (weight, indices)"));
            }
            let weight_val = self.compile_expr(builder, state, &args[0].value)?;
            let indices_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_embedding_lookup", &[weight_val, indices_val]);
        }
        // bias_add(tensor, bias) -> tensor -- broadcasts 1D bias over 2D tensor
        if func_name == "bias_add" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new("bias_add() takes exactly 2 arguments (tensor, bias)"));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let bias_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_bias_add", &[tensor_val, bias_val]);
        }
        // tensor_slice(tensor, dim, start, end) -> tensor
        if func_name == "tensor_slice" {
            if args.len() != 4 {
                return Err(CodegenError::new("tensor_slice() takes exactly 4 arguments (tensor, dim, start, end)"));
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
                return Err(CodegenError::new("tensor_cat() takes exactly 2 arguments (tensor_list, dim)"));
            }
            let list = self.compile_expr(builder, state, &args[0].value)?;
            let dim = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_cat", &[list, dim]);
        }
        // M18a: unsqueeze(tensor, dim) -- free function form
        if func_name == "unsqueeze" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new("unsqueeze() takes exactly 2 arguments (tensor, dim)"));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let dim_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_unsqueeze", &[tensor_val, dim_val]);
        }
        // M18a: stack(list_of_tensors, dim) -> tensor
        if func_name == "stack" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new("stack() takes exactly 2 arguments (tensor_list, dim)"));
            }
            let list_val = self.compile_expr(builder, state, &args[0].value)?;
            let dim_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_stack", &[list_val, dim_val]);
        }
        // contiguous(tensor) -> tensor — materialize non-contiguous views
        if func_name == "contiguous" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new("contiguous() takes exactly 1 argument (tensor)"));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_contiguous", &[tensor_val]);
        }
        // M18a: causal_mask(seq_len) -> tensor
        if func_name == "causal_mask" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new("causal_mask() takes exactly 1 argument (seq_len)"));
            }
            let seq_len_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_causal_mask", &[seq_len_val]);
        }
        // M27: scaled_dot_product_attention(Q, K, V, scale, causal=false)
        // Naive path: softmax((Q @ K.T) * scale [+ causal_mask]) @ V
        // Flash path: PTX dispatch via nsl_flash_attention when @flash_attention decorator is present
        if func_name == "scaled_dot_product_attention" && !self.registry.functions.contains_key(&func_name) {
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
                return self.compile_flash_attention_call(builder, q_val, k_val, v_val, scale_val);
            }

            // M34: Check for context parallelism (ring attention)
            // Look up config using the model name prefix from the mangled function name
            // ("ModelName__method_name"), mirroring the MoE/speculative patterns.
            let cp_config = state.current_function_name.as_ref()
                .and_then(|fn_name| {
                    let model_prefix = fn_name.split("__").next().unwrap_or("");
                    self.features.context_parallel_configs.iter()
                        .find(|(key, _)| key.starts_with(model_prefix))
                        .map(|(_, info)| info.clone())
                });

            if let Some(cp_info) = cp_config {
                eprintln!("[nsl] Context parallelism active: ring_size={}", cp_info.ring_size);

                // Initialize context parallel context:
                // nsl_cp_init(ring_size, local_seq_len, num_heads, num_kv_heads, head_dim, dtype)
                // Pass zeros for dims inferred by runtime; dtype=1 means f32.
                let ring_size = builder.ins().iconst(cl_types::I64, cp_info.ring_size as i64);
                let zero = builder.ins().iconst(cl_types::I64, 0);
                let dtype_val = builder.ins().iconst(cl_types::I64, 1); // f32
                let cp_ctx = self.compile_call_by_name(
                    builder,
                    "nsl_cp_init",
                    &[ring_size, zero, zero, zero, zero, dtype_val],
                )?;

                // Partition Q across the ring:
                // nsl_sequence_partition(tensor, batch, seq_len, hidden, ring_size, rank, overlap)
                // Runtime infers dims from tensor shape; rank=0 (host rank), overlap=0.
                let q_part = self.compile_call_by_name(
                    builder,
                    "nsl_sequence_partition",
                    &[q_val, zero, zero, zero, ring_size, zero, zero],
                )?;

                // Ring attention core:
                // nsl_ring_attention(cp_ctx, q_part, k, v, scale, causal,
                //                    null, null, null, null, null, null, null)
                // Trailing nulls are reserved for future paged/GQA extensions.
                let causal_flag = builder.ins().iconst(cl_types::I64, if causal { 1 } else { 0 });
                let null = builder.ins().iconst(cl_types::I64, 0);
                let ring_result = self.compile_call_by_name(
                    builder,
                    "nsl_ring_attention",
                    &[cp_ctx, q_part, k_val, v_val, scale_val, causal_flag,
                      null, null, null, null, null, null, null],
                )?;

                // Gather partitioned outputs back to full sequence:
                // nsl_sequence_gather(result, ring_size, batch, seq_len, hidden, rank, overlap, null)
                let output = self.compile_call_by_name(
                    builder,
                    "nsl_sequence_gather",
                    &[ring_result, ring_size, zero, zero, zero, zero, zero, null],
                )?;

                // Destroy context parallel context.
                self.compile_call_by_name(builder, "nsl_cp_destroy", &[cp_ctx])?;

                return Ok(output);
            }

            // Naive path: softmax(apply_causal_mask((Q @ K.T) * scale)) @ V
            let dim_m2 = builder.ins().iconst(cl_types::I64, -2_i64);
            let dim_m1 = builder.ins().iconst(cl_types::I64, -1_i64);
            let k_t = self.compile_call_by_name(builder, "nsl_tensor_transpose", &[k_val, dim_m2, dim_m1])?;
            let scores = self.compile_traced_call(builder, "nsl_tensor_matmul", &[q_val, k_t])?;
            let scaled = self.compile_traced_call(builder, "nsl_tensor_mul_scalar", &[scores, scale_val])?;

            let masked = if causal {
                let dim_neg2 = builder.ins().iconst(cl_types::I64, -2_i64);
                let seq_len = self.compile_call_by_name(builder, "nsl_tensor_shape_dim", &[q_val, dim_neg2])?;
                let mask = self.compile_call_by_name(builder, "nsl_tensor_causal_mask", &[seq_len])?;
                self.compile_traced_call(builder, "nsl_tensor_add", &[scaled, mask])?
            } else {
                scaled
            };

            let dim_neg1 = builder.ins().iconst(cl_types::I64, -1_i64);
            let attn_weights = self.compile_traced_call(builder, "nsl_tensor_softmax", &[masked, dim_neg1])?;
            return self.compile_traced_call(builder, "nsl_tensor_matmul", &[attn_weights, v_val]);
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
                return Err(CodegenError::new(format!("{func_name}() takes 1 or 3 arguments")));
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
            return self.compile_call_by_name(builder, "nsl_bpe_train", &[path_val, vocab_size, min_freq, special_tokens]);
        }
        if func_name == "tokenizer_load" {
            if args.len() != 1 {
                return Err(CodegenError::new("tokenizer_load() takes exactly 1 argument (path)"));
            }
            let path_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_tokenizer_load", &[path_val]);
        }
        if func_name == "tokenizer_save" {
            if args.len() != 2 {
                return Err(CodegenError::new("tokenizer_save() takes exactly 2 arguments (handle, path)"));
            }
            let handle = self.compile_expr(builder, state, &args[0].value)?;
            let path_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tokenizer_save", &[handle, path_val]);
        }
        if func_name == "tokenizer_encode" {
            if args.len() != 2 {
                return Err(CodegenError::new("tokenizer_encode() takes exactly 2 arguments (handle, text)"));
            }
            let handle = self.compile_expr(builder, state, &args[0].value)?;
            let text_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tokenizer_encode", &[handle, text_val]);
        }
        if func_name == "tokenizer_decode" {
            if args.len() != 2 {
                return Err(CodegenError::new("tokenizer_decode() takes exactly 2 arguments (handle, tensor)"));
            }
            let handle = self.compile_expr(builder, state, &args[0].value)?;
            let tensor_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tokenizer_decode", &[handle, tensor_val]);
        }
        if func_name == "tokenizer_vocab_size" {
            if args.len() != 1 {
                return Err(CodegenError::new("tokenizer_vocab_size() takes exactly 1 argument (handle)"));
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
            return self.compile_call_by_name(builder, "nsl_tokenizer_encode_batch", &[handle, texts, padding_i8, truncation_i8, max_len]);
        }

        // Quantization functions (M16)
        if func_name == "nsl_qtensor_quantize" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 5 {
                return Err(CodegenError::new("nsl_qtensor_quantize() takes exactly 5 arguments (tensor, dtype, granularity, axis, group_size)"));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let dtype_val = self.compile_expr(builder, state, &args[1].value)?;
            let gran_val = self.compile_expr(builder, state, &args[2].value)?;
            let axis_val = self.compile_expr(builder, state, &args[3].value)?;
            let group_val = self.compile_expr(builder, state, &args[4].value)?;
            return self.compile_call_by_name(builder, "nsl_qtensor_quantize", &[tensor_val, dtype_val, gran_val, axis_val, group_val]);
        }
        if func_name == "nsl_qtensor_dequantize" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new("nsl_qtensor_dequantize() takes exactly 1 argument (qtensor)"));
            }
            let qt_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_qtensor_dequantize", &[qt_val]);
        }
        if func_name == "nsl_qtensor_matmul_mixed" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new("nsl_qtensor_matmul_mixed() takes exactly 2 arguments (tensor, qtensor)"));
            }
            let x_val = self.compile_expr(builder, state, &args[0].value)?;
            let qw_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_qtensor_matmul_mixed", &[x_val, qw_val]);
        }
        if func_name == "nsl_qtensor_dtype" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new("nsl_qtensor_dtype() takes exactly 1 argument (qtensor)"));
            }
            let qt_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_qtensor_dtype", &[qt_val]);
        }
        if func_name == "nsl_qtensor_shape" && !self.registry.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new("nsl_qtensor_shape() takes exactly 1 argument (qtensor)"));
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
                return Err(CodegenError::new("load_safetensors() takes 1-2 arguments (path, device=0)"));
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
            return self.compile_call_by_name(builder, "nsl_safetensors_load", &[path_val, path_len, device_val]);
        }

        // Safetensors save intrinsic: save_safetensors(weights_dict, "path.safetensors")
        if func_name == "save_safetensors" {
            if args.len() != 2 {
                return Err(CodegenError::new("save_safetensors() takes exactly 2 arguments (weights_dict, path)"));
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
            return self.compile_call_by_name(builder, "nsl_safetensors_save", &[dict_val, path_val, path_len]);
        }

        // M32: MoE dispatch intrinsic
        if func_name == "moe_dispatch" {
            if args.len() != 3 {
                return Err(crate::error::CodegenError::new(
                    "moe_dispatch() takes 3 arguments (tokens, router_logits, experts)",
                ));
            }
            let tokens_val = self.compile_expr(builder, state, &args[0].value)?;
            let logits_val = self.compile_expr(builder, state, &args[1].value)?;
            let _experts_val = self.compile_expr(builder, state, &args[2].value)?;

            // Look up MoE config from decorator extraction, using model name prefix
            // from the mangled function name ("ModelName__method_name") to correctly
            // resolve the right config in multi-layer models instead of grabbing an
            // arbitrary entry via .values().next().
            // Look up MoE config by matching model name prefix from mangled function name.
            // Try both "ModelName__method" pattern and any config key matching.
            let config = state.current_function_name.as_ref()
                .and_then(|fn_name| {
                    let model_prefix = fn_name.split("__").next().unwrap_or("");
                    self.features.moe_configs.iter()
                        .find(|(key, _)| key.starts_with(model_prefix))
                        .map(|(_, info)| info.clone())
                })
                .or_else(|| {
                    // Fallback: if only one MoE config exists, use it
                    if self.features.moe_configs.len() == 1 {
                        self.features.moe_configs.values().next().cloned()
                    } else {
                        None
                    }
                });
            let (num_experts, top_k, capacity_factor) = match config {
                Some(info) => (info.num_experts, info.top_k, info.capacity_factor),
                None => (8, 2, 1.25f32), // defaults
            };

            let num_experts_val = builder.ins().iconst(cranelift_codegen::ir::types::I64, num_experts as i64);
            let top_k_val = builder.ins().iconst(cranelift_codegen::ir::types::I64, top_k as i64);
            let cap_bits = builder.ins().iconst(cranelift_codegen::ir::types::I64, capacity_factor.to_bits() as i64);

            // Call nsl_moe_dispatch_full(tokens, logits, num_experts, top_k, capacity_factor_bits)
            // Returns a new NslTensor with the MoE output
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

            // Look up @speculative config using model name prefix from the mangled
            // function name ("ModelName__method_name"), mirroring the MoE pattern (M32).
            let config = state.current_function_name.as_ref()
                .and_then(|fn_name| {
                    let model_prefix = fn_name.split("__").next().unwrap_or("");
                    self.features.speculative_configs.iter()
                        .find(|(key, _)| key.starts_with(model_prefix))
                        .map(|(_, info)| info.clone())
                });

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
            let num_tokens_val = builder.ins().iconst(cranelift_codegen::ir::types::I64, num_tokens);
            let method_val = builder.ins().iconst(cranelift_codegen::ir::types::I64, method_id);
            let tree_width_val = builder.ins().iconst(cranelift_codegen::ir::types::I64, tree_width);

            let result = self.compile_call_by_name(
                builder,
                "nsl_speculative_decode_step",
                &[draft_tokens, draft_logits, verifier_logits, draft_len, vocab_size, temp_bits,
                  num_tokens_val, method_val, tree_width_val],
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
            return self.compile_call_by_name(builder, "nsl_tensor_topk", &[tensor_val, k_val, dim_val]);
        }

        // multinomial(tensor, num_samples)
        if func_name == "multinomial" {
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let n_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_multinomial", &[tensor_val, n_val]);
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
                    builder, "nsl_serve_enqueue",
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
                    self.compile_call_by_name(builder, "nsl_serve_set_grammar", &[request_id, start_state])?;
                    eprintln!("[nsl] Constrained decoding active: schema='{}'", schema);
                } else {
                    // Check if the current function has a model-level @grammar decorator
                    let current_fn_name = state.current_function_name.clone().unwrap_or_default();
                    if let Some(grammar_info) = self.features.grammar_configs.get(&current_fn_name) {
                        if !grammar_info.grammar_source.is_empty() || !grammar_info.start_rule.is_empty() {
                            let src = grammar_info.grammar_source.clone();
                            let start_state = builder.ins().iconst(cl_types::I64, 1);
                            self.compile_call_by_name(builder, "nsl_serve_set_grammar", &[request_id, start_state])?;
                            eprintln!("[nsl] Constrained decoding active via @grammar: '{}'", src);
                        }
                    }
                }

                let _ = model_val; // suppress unused warning — model used as prompt_len placeholder
                return Ok(request_id);
            }
            return Err(CodegenError::new("generate() requires at least 2 arguments (model, tokens)"));
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
            return self.compile_call_by_name(builder, "nsl_tensor_lt_scalar", &[tensor_val, scalar_f64]);
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
            return self.compile_call_by_name(builder, "nsl_load_jsonl", &[path_val, path_len, field_val, field_len]);
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
            return self.compile_call_by_name(builder, "nsl_load_csv", &[path_val, path_len, col_val, header_val]);
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
            return self.compile_call_by_name(builder, "nsl_load_mmap", &[path_val, path_len, dtype_val]);
        }

        // M19: DataLoader intrinsic

        // DataLoader(data, batch_size=32, seq_len=128, ...)
        if func_name == "DataLoader" {
            let data_val = self.compile_expr(builder, state, &args[0].value)?;

            // Build JSON config from keyword args at compile time
            let mut config = serde_json::Map::new();
            for arg in args.iter().skip(1) {
                if let Some(name_sym) = arg.name {
                    let key = self.resolve_sym(name_sym).to_string();
                    match &arg.value.kind {
                        ExprKind::IntLiteral(v) => {
                            config.insert(key, serde_json::Value::Number(serde_json::Number::from(*v)));
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
                }
            }
            let config_json = serde_json::Value::Object(config).to_string();

            // Intern the config string as data
            let config_data_id = self.intern_string(&config_json)?;
            let config_gv = self.module.declare_data_in_func(config_data_id, builder.func);
            let config_ptr = builder.ins().symbol_value(pointer_type(), config_gv);
            let config_len = builder.ins().iconst(cl_types::I64, config_json.len() as i64);

            // Ensure tensor is on CPU (DataLoader reads raw f64 data pointer)
            let cpu_device = builder.ins().iconst(cl_types::I64, 0);
            let data_val = self.compile_call_by_name(builder, "nsl_tensor_to_device", &[data_val, cpu_device])?;

            // Read tensor .len field (offset 32: data:8 + shape:8 + strides:8 + ndim:8)
            let tensor_len = builder.ins().load(
                cl_types::I64,
                MemFlags::trusted(),
                data_val,
                cranelift_codegen::ir::immediates::Offset32::new(40), // NslTensor.len offset (magic+pad shifts +8)
            );

            // Read tensor .data field (offset 8, after magic:u32 + 4-byte pad)
            let tensor_data = builder.ins().load(
                cl_types::I64,
                MemFlags::trusted(),
                data_val,
                cranelift_codegen::ir::immediates::Offset32::new(8),
            );

            // Read tensor dtype via safe FFI call (avoids struct offset assumptions)
            let tensor_dtype = self.compile_call_by_name(builder, "nsl_tensor_get_dtype", &[data_val])?;
            let dl_ptr = self.compile_call_by_name(builder, "nsl_dataloader_create", &[tensor_data, tensor_len, config_ptr, config_len, tensor_dtype])?;
            self.compile_call_by_name(builder, "nsl_dataloader_start", &[dl_ptr])?;
            state.cleanup.dataloader_vars.push(dl_ptr);
            return Ok(dl_ptr);
        }

        // Check if it's a known function or variable holding a function pointer
        if self.registry.functions.contains_key(&func_name) || self.registry.runtime_fns.contains_key(&func_name) {
            let mut arg_vals = Vec::new();
            for arg in args {
                arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
            }
            return self.compile_call_by_name(builder, &func_name, &arg_vals);
        }

        // Forward dispatch: if callee is a model instance, invoke its forward method
        let callee_type = self.node_type(callee.id).clone();
        if let Type::Model { name, .. } = &callee_type {
            let model_name = self.resolve_sym(*name).to_string();
            return self.compile_model_method_call(
                builder, state, callee, &model_name, "forward", args,
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
        let effective_name: &str = if let Some(batched_name) = self.vmap.batched_fn_names.get(func_name) {
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
        } else {
            return Err(CodegenError::new(format!("undefined function '{effective_name}'")));
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

        if let Type::Function { params: param_types, ret, .. } = &callee_type {
            if let Some(num_captures) = closure_captures {
                // Closure call: raw_val is a pointer to { fn_ptr, num_captures, captures[] }
                let closure_ptr = raw_val;

                // Load fn_ptr from offset 0
                let fn_ptr = builder.ins().load(cl_types::I64, MemFlags::trusted(), closure_ptr, 0);

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
                    let cap_val = builder.ins().load(cl_types::I64, MemFlags::trusted(), closure_ptr, offset);
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
        _kernel_name: &str,
        args: &[nsl_ast::expr::Arg],
        ptx_data_id: cranelift_module::DataId,
        name_data_id: cranelift_module::DataId,
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

        // Build args array on stack: each tensor arg is an i64 (pointer)
        let num_args = tensor_args.len();
        if num_args == 0 {
            // No tensor args -- pass null pointer and 0 count
            let null_ptr = builder.ins().iconst(cl_types::I64, 0);
            let num_args_val = builder.ins().iconst(cl_types::I64, 0);
            let shared_mem = builder.ins().iconst(cl_types::I64, 0);
            return self.compile_call_by_name(
                builder,
                "nsl_kernel_launch",
                &[ptx_ptr, name_ptr, grid_x, grid_y, grid_z, block_x, block_y, block_z, null_ptr, num_args_val, shared_mem],
            );
        }

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
            &[ptx_ptr, name_ptr, grid_x, grid_y, grid_z, block_x, block_y, block_z, args_ptr, num_args_val, shared_mem],
        )
    }
}
