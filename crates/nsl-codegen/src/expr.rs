use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{AbiParam, InstBuilder, MemFlags, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::{DataId, Linkage, Module};

use nsl_ast::expr::{Expr, ExprKind, FStringPart, SubscriptKind};
use nsl_ast::operator::{BinOp, UnaryOp};
use nsl_ast::pattern::PatternKind;
use nsl_semantic::types::Type;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::types::{is_float_type, is_int_type, nsl_type_to_cl, pointer_type};

impl Compiler<'_> {
    pub fn compile_expr(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        expr: &Expr,
    ) -> Result<Value, CodegenError> {
        match &expr.kind {
            ExprKind::IntLiteral(n) => Ok(builder.ins().iconst(cl_types::I64, *n)),
            ExprKind::FloatLiteral(f) => Ok(builder.ins().f64const(*f)),
            ExprKind::BoolLiteral(b) => {
                Ok(builder.ins().iconst(cl_types::I8, if *b { 1 } else { 0 }))
            }
            ExprKind::StringLiteral(s) => self.compile_string_literal(builder, s),
            ExprKind::NoneLiteral => Ok(builder.ins().iconst(cl_types::I64, 0)),

            ExprKind::Ident(sym) => {
                if let Some((var, _)) = state.variables.get(sym) {
                    Ok(builder.use_var(*var))
                } else {
                    let name = self.resolve_sym(*sym).to_string();
                    // Device constants for .to(device) calls
                    if name == "cuda" {
                        return Ok(builder.ins().iconst(cl_types::I64, 1));
                    }
                    if name == "cpu" {
                        return Ok(builder.ins().iconst(cl_types::I64, 0));
                    }
                    // Check if it's an enum variant
                    if let Some(tag) = self.lookup_enum_variant_tag(&name) {
                        Ok(builder.ins().iconst(cl_types::I64, tag))
                    } else {
                        Err(CodegenError::new(format!(
                            "undefined variable '{name}'"
                        )))
                    }
                }
            }

            ExprKind::BinaryOp { left, op, right } => {
                self.compile_binary_op(builder, state, left, *op, right, expr)
            }
            ExprKind::UnaryOp { op, operand } => {
                self.compile_unary_op(builder, state, *op, operand)
            }
            ExprKind::Call { callee, args } => {
                self.compile_call(builder, state, callee, args, expr)
            }
            ExprKind::MemberAccess { object, member } => {
                self.compile_member_access(builder, state, object, *member, expr)
            }
            ExprKind::ListLiteral(elements) => {
                self.compile_list_literal(builder, state, elements)
            }
            ExprKind::Subscript { object, index } => {
                self.compile_subscript(builder, state, object, index)
            }
            ExprKind::ListComp { element, generators } => {
                self.compile_list_comp(builder, state, element, generators)
            }
            ExprKind::Lambda { params, body } => {
                self.compile_lambda(builder, state, params, body)
            }
            ExprKind::TupleLiteral(elements) => {
                self.compile_tuple_literal(builder, state, elements)
            }
            ExprKind::DictLiteral(pairs) => {
                self.compile_dict_literal(builder, state, pairs)
            }
            ExprKind::MatchExpr { subject, arms } => {
                self.compile_match_expr(builder, state, subject, arms, expr)
            }
            ExprKind::Range { start, end, inclusive } => {
                self.compile_range_expr(builder, state, start.as_deref(), end.as_deref(), *inclusive)
            }
            ExprKind::FString(parts) => self.compile_fstring(builder, state, parts),
            ExprKind::Paren(inner) => self.compile_expr(builder, state, inner),
            ExprKind::IfExpr { condition, then_expr, else_expr } => {
                self.compile_if_expr(builder, state, condition, then_expr, else_expr, expr)
            }
            ExprKind::Pipe { left, right } => {
                let left_val = self.compile_expr(builder, state, left)?;
                match &right.kind {
                    ExprKind::Ident(_) => {
                        let func_name = self.expr_as_func_name(right);
                        if let Some(name) = func_name {
                            self.compile_call_by_name(builder, &name, &[left_val])
                        } else {
                            Err(CodegenError::new("pipe target is not a function"))
                        }
                    }
                    ExprKind::Call { callee, args } => {
                        let func_name = self.expr_as_func_name(callee);
                        if let Some(name) = func_name {
                            let mut arg_vals = vec![left_val];
                            for arg in args {
                                arg_vals.push(self.compile_expr(builder, state, &arg.value)?);
                            }
                            self.compile_call_by_name(builder, &name, &arg_vals)
                        } else {
                            Err(CodegenError::new("pipe call target is not a function"))
                        }
                    }
                    _ => Err(CodegenError::new("unsupported pipe target")),
                }
            }

            ExprKind::SelfRef => {
                // Find the "self" variable in state (set by model method compilation)
                for (sym, (var, _ty)) in &state.variables {
                    if self.resolve_sym(*sym) == "self" {
                        return Ok(builder.use_var(*var));
                    }
                }
                Err(CodegenError::new("`self` used outside of model method"))
            }

            ExprKind::Error => Ok(builder.ins().iconst(cl_types::I64, 0)),
            _ => Err(CodegenError::new(format!(
                "unsupported expression in M3 codegen: {:?}",
                std::mem::discriminant(&expr.kind)
            ))),
        }
    }

    fn compile_string_literal(
        &mut self,
        builder: &mut FunctionBuilder,
        s: &str,
    ) -> Result<Value, CodegenError> {
        let data_id = *self
            .string_pool
            .get(s)
            .ok_or_else(|| CodegenError::new(format!("string not in pool: {s:?}")))?;
        let gv = self.module.declare_data_in_func(data_id, builder.func);
        Ok(builder.ins().symbol_value(pointer_type(), gv))
    }

    // ── Binary ops ──────────────────────────────────────────────────

    fn compile_binary_op(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        left: &Expr,
        op: BinOp,
        right: &Expr,
        _full_expr: &Expr,
    ) -> Result<Value, CodegenError> {
        if matches!(op, BinOp::And | BinOp::Or) {
            return self.compile_short_circuit(builder, state, left, op, right);
        }

        if matches!(op, BinOp::In) {
            let lhs = self.compile_expr(builder, state, left)?;
            let rhs = self.compile_expr(builder, state, right)?;
            let rhs_type = self.node_type(right.id).clone();
            let fn_name = if matches!(rhs_type, Type::Dict(_, _)) {
                "nsl_dict_contains"
            } else {
                "nsl_list_contains"
            };
            let fid = self.runtime_fns[fn_name].0;
            let fref = self.module.declare_func_in_func(fid, builder.func);
            let call = builder.ins().call(fref, &[rhs, lhs]);
            return Ok(builder.inst_results(call)[0]);
        }

        let mut lhs = self.compile_expr(builder, state, left)?;
        let mut rhs = self.compile_expr(builder, state, right)?;
        let left_type = self.node_type(left.id).clone();
        let right_type = self.node_type(right.id).clone();

        // String concatenation with +
        if matches!(op, BinOp::Add) && (matches!(left_type, Type::Str) || matches!(right_type, Type::Str)) {
            // Convert non-string operand to string if needed
            if !matches!(left_type, Type::Str) {
                lhs = self.value_to_string(builder, lhs, &left_type)?;
            }
            if !matches!(right_type, Type::Str) {
                rhs = self.value_to_string(builder, rhs, &right_type)?;
            }
            let concat_id = self.runtime_fns["nsl_str_concat"].0;
            let concat_ref = self.module.declare_func_in_func(concat_id, builder.func);
            let call = builder.ins().call(concat_ref, &[lhs, rhs]);
            return Ok(builder.inst_results(call)[0]);
        }

        // String repeat with *
        if matches!(op, BinOp::Mul) && (matches!(left_type, Type::Str) || matches!(right_type, Type::Str)) {
            let (str_val, int_val) = if matches!(left_type, Type::Str) {
                (lhs, rhs)
            } else {
                (rhs, lhs)
            };
            let fid = self.runtime_fns["nsl_str_repeat"].0;
            let fref = self.module.declare_func_in_func(fid, builder.func);
            let call = builder.ins().call(fref, &[str_val, int_val]);
            return Ok(builder.inst_results(call)[0]);
        }

        // Tensor operations
        // Unknown (indeterminate) types are treated as tensors when:
        //  - The op is MatMul (@ is tensor-only)
        //  - The other operand is a tensor
        //  - Both operands are indeterminate (likely from .to(device) or other runtime returns)
        let both_indeterminate = left_type.is_indeterminate() && right_type.is_indeterminate();
        let left_is_tensor = left_type.is_tensor()
            || (left_type.is_indeterminate() && (matches!(op, BinOp::MatMul) || right_type.is_tensor() || both_indeterminate));
        let right_is_tensor = right_type.is_tensor()
            || (right_type.is_indeterminate() && (matches!(op, BinOp::MatMul) || left_type.is_tensor() || both_indeterminate));
        if left_is_tensor || right_is_tensor {
            return self.compile_tensor_binary_op(builder, state, lhs, rhs, op, left_is_tensor, right_is_tensor);
        }

        let left_is_float = is_float_type(&left_type);
        let right_is_float = is_float_type(&right_type);
        let is_float = left_is_float || right_is_float;

        // Promote int → float if mixed types
        if is_float {
            if !left_is_float {
                lhs = builder.ins().fcvt_from_sint(cl_types::F64, lhs);
            }
            if !right_is_float {
                rhs = builder.ins().fcvt_from_sint(cl_types::F64, rhs);
            }
        }

        match op {
            BinOp::Add if is_float => Ok(builder.ins().fadd(lhs, rhs)),
            BinOp::Add => Ok(builder.ins().iadd(lhs, rhs)),
            BinOp::Sub if is_float => Ok(builder.ins().fsub(lhs, rhs)),
            BinOp::Sub => Ok(builder.ins().isub(lhs, rhs)),
            BinOp::Mul if is_float => Ok(builder.ins().fmul(lhs, rhs)),
            BinOp::Mul => Ok(builder.ins().imul(lhs, rhs)),
            BinOp::Div if is_float => Ok(builder.ins().fdiv(lhs, rhs)),
            BinOp::Div => Ok(builder.ins().sdiv(lhs, rhs)),
            BinOp::FloorDiv if is_float => {
                let div = builder.ins().fdiv(lhs, rhs);
                Ok(builder.ins().floor(div))
            }
            BinOp::FloorDiv => Ok(builder.ins().sdiv(lhs, rhs)),
            BinOp::Mod if is_float => {
                // a % b = a - floor(a/b) * b
                let div = builder.ins().fdiv(lhs, rhs);
                let floored = builder.ins().floor(div);
                let prod = builder.ins().fmul(floored, rhs);
                Ok(builder.ins().fsub(lhs, prod))
            }
            BinOp::Mod => Ok(builder.ins().srem(lhs, rhs)),
            BinOp::Pow => self.compile_pow(builder, lhs, rhs, is_float),

            BinOp::Eq if is_float => Ok(builder.ins().fcmp(FloatCC::Equal, lhs, rhs)),
            BinOp::Eq if matches!(left_type, Type::Str) => {
                let fid = self.runtime_fns["nsl_str_eq"].0;
                let fref = self.module.declare_func_in_func(fid, builder.func);
                let call = builder.ins().call(fref, &[lhs, rhs]);
                Ok(builder.inst_results(call)[0])
            }
            BinOp::Eq => Ok(builder.ins().icmp(IntCC::Equal, lhs, rhs)),
            BinOp::NotEq if is_float => Ok(builder.ins().fcmp(FloatCC::NotEqual, lhs, rhs)),
            BinOp::NotEq if matches!(left_type, Type::Str) => {
                let fid = self.runtime_fns["nsl_str_eq"].0;
                let fref = self.module.declare_func_in_func(fid, builder.func);
                let call = builder.ins().call(fref, &[lhs, rhs]);
                let eq_val = builder.inst_results(call)[0];
                // Invert: not-equal = 1 - eq
                let one = builder.ins().iconst(cl_types::I64, 1);
                Ok(builder.ins().isub(one, eq_val))
            }
            BinOp::NotEq => Ok(builder.ins().icmp(IntCC::NotEqual, lhs, rhs)),
            BinOp::Lt if is_float => Ok(builder.ins().fcmp(FloatCC::LessThan, lhs, rhs)),
            BinOp::Lt => Ok(builder.ins().icmp(IntCC::SignedLessThan, lhs, rhs)),
            BinOp::Gt if is_float => Ok(builder.ins().fcmp(FloatCC::GreaterThan, lhs, rhs)),
            BinOp::Gt => Ok(builder.ins().icmp(IntCC::SignedGreaterThan, lhs, rhs)),
            BinOp::LtEq if is_float => Ok(builder.ins().fcmp(FloatCC::LessThanOrEqual, lhs, rhs)),
            BinOp::LtEq => Ok(builder.ins().icmp(IntCC::SignedLessThanOrEqual, lhs, rhs)),
            BinOp::GtEq if is_float => Ok(builder.ins().fcmp(FloatCC::GreaterThanOrEqual, lhs, rhs)),
            BinOp::GtEq => Ok(builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, lhs, rhs)),

            BinOp::BitAnd => Ok(builder.ins().band(lhs, rhs)),
            BinOp::BitOr => Ok(builder.ins().bor(lhs, rhs)),

            BinOp::And | BinOp::Or => unreachable!(),
            _ => Err(CodegenError::new(format!("unsupported binary op: {op:?}"))),
        }
    }

    fn compile_short_circuit(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        left: &Expr,
        op: BinOp,
        right: &Expr,
    ) -> Result<Value, CodegenError> {
        let lhs_raw = self.compile_expr(builder, state, left)?;
        // Normalize to I8 boolean (handles I64 values from comparisons etc.)
        let lhs = builder.ins().icmp_imm(IntCC::NotEqual, lhs_raw, 0);

        let rhs_block = builder.create_block();
        let merge_block = builder.create_block();
        builder.append_block_param(merge_block, cl_types::I8);

        match op {
            BinOp::And => builder.ins().brif(lhs, rhs_block, &[], merge_block, &[lhs]),
            BinOp::Or => builder.ins().brif(lhs, merge_block, &[lhs], rhs_block, &[]),
            _ => unreachable!(),
        };

        builder.switch_to_block(rhs_block);
        builder.seal_block(rhs_block);
        state.current_block = Some(rhs_block);
        let rhs_raw = self.compile_expr(builder, state, right)?;
        let rhs = builder.ins().icmp_imm(IntCC::NotEqual, rhs_raw, 0);
        builder.ins().jump(merge_block, &[rhs]);

        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);
        state.current_block = Some(merge_block);
        Ok(builder.block_params(merge_block)[0])
    }

    fn compile_pow(
        &mut self,
        builder: &mut FunctionBuilder,
        lhs: Value,
        rhs: Value,
        is_float: bool,
    ) -> Result<Value, CodegenError> {
        let rt_fn = if is_float { "nsl_pow_float" } else { "nsl_pow_int" };
        let func_id = self.runtime_fns[rt_fn].0;
        let func_ref = self.module.declare_func_in_func(func_id, builder.func);
        let call = builder.ins().call(func_ref, &[lhs, rhs]);
        Ok(builder.inst_results(call)[0])
    }

    // ── Unary ops ───────────────────────────────────────────────────

    fn compile_unary_op(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        op: UnaryOp,
        operand: &Expr,
    ) -> Result<Value, CodegenError> {
        let val = self.compile_expr(builder, state, operand)?;
        let ty = self.node_type(operand.id).clone();
        match op {
            UnaryOp::Neg if matches!(ty, Type::Tensor { .. }) => {
                self.compile_call_by_name(builder, "nsl_tensor_neg", &[val])
            }
            UnaryOp::Neg if is_float_type(&ty) => Ok(builder.ins().fneg(val)),
            UnaryOp::Neg => Ok(builder.ins().ineg(val)),
            UnaryOp::Not => Ok(builder.ins().icmp_imm(IntCC::Equal, val, 0)),
        }
    }

    // ── Calls ───────────────────────────────────────────────────────

    fn compile_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        callee: &Expr,
        args: &[nsl_ast::expr::Arg],
        _call_expr: &Expr,
    ) -> Result<Value, CodegenError> {
        // Intercept method calls (e.g., s.upper()) before extracting func_name
        if let ExprKind::MemberAccess { object, member } = &callee.kind {
            let member_name = self.resolve_sym(*member).to_string();
            let obj_type = self.node_type(object.id).clone();
            // Module alias call: math.clamp(...) → call "clamp"
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
                return self.compile_model_method_call(builder, state, object, &model_name, &member_name, args);
            }
            // Fallback for model array loop variables (type is Unknown but var was bound from model array)
            if matches!(obj_type, Type::Unknown) {
                if let ExprKind::Ident(obj_sym) = &object.kind {
                    if let Some(model_name) = self.model_var_types.get(obj_sym).cloned() {
                        return self.compile_model_method_call(builder, state, object, &model_name, &member_name, args);
                    }
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

        // Check if this is a kernel call (compiled GPU kernel)
        if let Some((ptx_data_id, name_data_id)) = self.kernel_ptx_data.get(&func_name).cloned() {
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
                let fid = self.runtime_fns[fn_name].0;
                let fref = self.module.declare_func_in_func(fid, builder.func);
                let call = builder.ins().call(fref, &[val]);
                return Ok(builder.inst_results(call)[0]);
            }
        }
        if func_name == "range" {
            return self.compile_range_call(builder, state, args);
        }
        if matches!(func_name.as_str(), "int" | "float" | "str" | "bool") {
            return self.compile_type_conversion(builder, state, &func_name, args);
        }
        // Higher-order functions: map(fn, list), filter(fn, list)
        if matches!(func_name.as_str(), "map" | "filter") {
            return self.compile_higher_order_call(builder, state, &func_name, args);
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
                    val
                } else {
                    builder.ins().fcvt_from_sint(cl_types::F64, val)
                };
                let rt_name = format!("nsl_{func_name}");
                return self.compile_call_by_name(builder, &rt_name, &[float_val]);
            }
            // else: tensor type — fall through to M14 tensor dispatch
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
            // else: tensor type — fall through to M14 tensor dispatch
        }
        // floor: scalar f64 dispatch
        if func_name == "floor" {
            if args.len() != 1 {
                return Err(CodegenError::new("floor() takes exactly 1 argument"));
            }
            let val = self.compile_expr(builder, state, &args[0].value)?;
            let arg_type = self.node_type(args[0].value.id).clone();
            let float_val = if is_float_type(&arg_type) {
                val
            } else {
                builder.ins().fcvt_from_sint(cl_types::F64, val)
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
                let a_f = if is_float_type(&a_type) { a } else { builder.ins().fcvt_from_sint(cl_types::F64, a) };
                let b_f = if is_float_type(&b_type) { b } else { builder.ins().fcvt_from_sint(cl_types::F64, b) };
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
            let cond_i8 = builder.ins().icmp_imm(IntCC::NotEqual, cond_val, 0);
            let msg = if args.len() > 1 {
                self.compile_expr(builder, state, &args[1].value)?
            } else {
                self.intern_string("assertion failed")?;
                self.compile_string_literal(builder, "assertion failed")?
            };
            let fid = self.runtime_fns["nsl_assert"].0;
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
            && !self.functions.contains_key(&func_name)
        {
            if args.len() != 1 {
                return Err(CodegenError::new(format!("{func_name}() takes exactly 1 argument")));
            }
            let val = self.compile_expr(builder, state, &args[0].value)?;
            let rt_name = format!("nsl_tensor_{func_name}");
            return self.compile_call_by_name(builder, &rt_name, &[val]);
        }
        // Activation functions (M15): relu, gelu, silu, sigmoid — single tensor arg
        if matches!(func_name.as_str(), "relu" | "gelu" | "silu" | "sigmoid")
            && !self.functions.contains_key(&func_name)
        {
            if args.len() != 1 {
                return Err(CodegenError::new(format!("{func_name}() takes exactly 1 argument")));
            }
            let val = self.compile_expr(builder, state, &args[0].value)?;
            let rt_name = format!("nsl_tensor_{func_name}");
            return self.compile_call_by_name(builder, &rt_name, &[val]);
        }
        // tanh activation: maps NSL name "tanh" to runtime "nsl_tensor_tanh_act"
        if func_name == "tanh" && !self.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new("tanh() takes exactly 1 argument"));
            }
            let val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_tanh_act", &[val]);
        }
        // softmax(tensor, dim) — two args
        if func_name == "softmax" && !self.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new("softmax() takes exactly 2 arguments (tensor, dim)"));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let dim_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_softmax", &[tensor_val, dim_val]);
        }
        if func_name == "clamp" && !self.functions.contains_key(&func_name) {
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
        if func_name == "layernorm" && !self.functions.contains_key(&func_name) {
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
        if func_name == "rmsnorm" && !self.functions.contains_key(&func_name) {
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
        if func_name == "dropout" && !self.functions.contains_key(&func_name) {
            if args.len() != 3 {
                return Err(CodegenError::new("dropout() takes exactly 3 arguments (tensor, p, training)"));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let p_val = self.compile_expr(builder, state, &args[1].value)?;
            let p_type = self.node_type(args[1].value.id).clone();
            let p_f = if is_float_type(&p_type) { p_val } else { builder.ins().fcvt_from_sint(cl_types::F64, p_val) };
            let training_val = self.compile_expr(builder, state, &args[2].value)?;
            let training_i8 = builder.ins().ireduce(cl_types::I8, training_val);
            return self.compile_call_by_name(builder, "nsl_tensor_dropout", &[tensor_val, p_f, training_i8]);
        }
        // conv2d(input, weight, bias, stride_h, stride_w, pad_h, pad_w) -> tensor
        if func_name == "conv2d" && !self.functions.contains_key(&func_name) {
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
        if func_name == "maxpool2d" && !self.functions.contains_key(&func_name) {
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
        if func_name == "embedding_lookup" && !self.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new("embedding_lookup() takes exactly 2 arguments (weight, indices)"));
            }
            let weight_val = self.compile_expr(builder, state, &args[0].value)?;
            let indices_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_embedding_lookup", &[weight_val, indices_val]);
        }
        // bias_add(tensor, bias) -> tensor — broadcasts 1D bias over 2D tensor
        if func_name == "bias_add" && !self.functions.contains_key(&func_name) {
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
        // M18a: unsqueeze(tensor, dim) — free function form
        if func_name == "unsqueeze" && !self.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new("unsqueeze() takes exactly 2 arguments (tensor, dim)"));
            }
            let tensor_val = self.compile_expr(builder, state, &args[0].value)?;
            let dim_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_unsqueeze", &[tensor_val, dim_val]);
        }
        // M18a: stack(list_of_tensors, dim) -> tensor
        if func_name == "stack" && !self.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new("stack() takes exactly 2 arguments (tensor_list, dim)"));
            }
            let list_val = self.compile_expr(builder, state, &args[0].value)?;
            let dim_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_stack", &[list_val, dim_val]);
        }
        // M18a: causal_mask(seq_len) -> tensor
        if func_name == "causal_mask" && !self.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new("causal_mask() takes exactly 1 argument (seq_len)"));
            }
            let seq_len_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_tensor_causal_mask", &[seq_len_val]);
        }
        // sum/mean with dim args — overload: sum(tensor) or sum(tensor, dim, keepdim)
        if matches!(func_name.as_str(), "sum" | "mean") {
            if args.len() == 1 {
                let val = self.compile_expr(builder, state, &args[0].value)?;
                let rt_name = format!("nsl_tensor_{func_name}");
                return self.compile_call_by_name(builder, &rt_name, &[val]);
            } else if args.len() == 3 {
                let t = self.compile_expr(builder, state, &args[0].value)?;
                let dim = self.compile_expr(builder, state, &args[1].value)?;
                let keepdim = self.compile_expr(builder, state, &args[2].value)?;
                let rt_name = format!("nsl_tensor_{func_name}_dim");
                return self.compile_call_by_name(builder, &rt_name, &[t, dim, keepdim]);
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
            let padding_i8 = builder.ins().ireduce(cl_types::I8, padding);
            let truncation = self.compile_expr(builder, state, &args[3].value)?;
            let truncation_i8 = builder.ins().ireduce(cl_types::I8, truncation);
            let max_len = self.compile_expr(builder, state, &args[4].value)?;
            return self.compile_call_by_name(builder, "nsl_tokenizer_encode_batch", &[handle, texts, padding_i8, truncation_i8, max_len]);
        }

        // Quantization functions (M16)
        if func_name == "nsl_qtensor_quantize" && !self.functions.contains_key(&func_name) {
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
        if func_name == "nsl_qtensor_dequantize" && !self.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new("nsl_qtensor_dequantize() takes exactly 1 argument (qtensor)"));
            }
            let qt_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_qtensor_dequantize", &[qt_val]);
        }
        if func_name == "nsl_qtensor_matmul_mixed" && !self.functions.contains_key(&func_name) {
            if args.len() != 2 {
                return Err(CodegenError::new("nsl_qtensor_matmul_mixed() takes exactly 2 arguments (tensor, qtensor)"));
            }
            let x_val = self.compile_expr(builder, state, &args[0].value)?;
            let qw_val = self.compile_expr(builder, state, &args[1].value)?;
            return self.compile_call_by_name(builder, "nsl_qtensor_matmul_mixed", &[x_val, qw_val]);
        }
        if func_name == "nsl_qtensor_dtype" && !self.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new("nsl_qtensor_dtype() takes exactly 1 argument (qtensor)"));
            }
            let qt_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_qtensor_dtype", &[qt_val]);
        }
        if func_name == "nsl_qtensor_shape" && !self.functions.contains_key(&func_name) {
            if args.len() != 1 {
                return Err(CodegenError::new("nsl_qtensor_shape() takes exactly 1 argument (qtensor)"));
            }
            let qt_val = self.compile_expr(builder, state, &args[0].value)?;
            return self.compile_call_by_name(builder, "nsl_qtensor_shape", &[qt_val]);
        }

        // Check if it's a known function or variable holding a function pointer
        if self.functions.contains_key(&func_name) || self.runtime_fns.contains_key(&func_name) {
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
        let (func_id, sig) = if let Some(e) = self.functions.get(func_name) {
            e.clone()
        } else if let Some(e) = self.runtime_fns.get(func_name) {
            e.clone()
        } else {
            return Err(CodegenError::new(format!("undefined function '{func_name}'")));
        };

        let func_ref = self.module.declare_func_in_func(func_id, builder.func);
        let call = builder.ins().call(func_ref, arg_vals);
        if sig.returns.is_empty() {
            Ok(builder.ins().iconst(cl_types::I64, 0))
        } else {
            Ok(builder.inst_results(call)[0])
        }
    }

    fn compile_print_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        if args.is_empty() {
            let empty = self.compile_string_literal(builder, "")?;
            let fid = self.runtime_fns["nsl_print_str"].0;
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

        let fid = self.runtime_fns[rt_fn].0;
        let fref = self.module.declare_func_in_func(fid, builder.func);
        builder.ins().call(fref, &[val]);
        Ok(builder.ins().iconst(cl_types::I64, 0))
    }

    fn compile_range_call(
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
        let fid = self.runtime_fns["nsl_range"].0;
        let fref = self.module.declare_func_in_func(fid, builder.func);
        let call = builder.ins().call(fref, &[start, stop, step]);
        Ok(builder.inst_results(call)[0])
    }

    // ── List literal ────────────────────────────────────────────────

    fn compile_list_literal(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        elements: &[Expr],
    ) -> Result<Value, CodegenError> {
        let new_id = self.runtime_fns["nsl_list_new"].0;
        let new_ref = self.module.declare_func_in_func(new_id, builder.func);
        let call = builder.ins().call(new_ref, &[]);
        let list_ptr = builder.inst_results(call)[0];

        let push_id = self.runtime_fns["nsl_list_push"].0;
        let push_ref = self.module.declare_func_in_func(push_id, builder.func);
        for elem in elements {
            let val = self.compile_expr(builder, state, elem)?;
            builder.ins().call(push_ref, &[list_ptr, val]);
        }
        Ok(list_ptr)
    }

    // ── Member access ───────────────────────────────────────────────

    fn compile_member_access(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        object: &Expr,
        member: nsl_ast::Symbol,
        _expr: &Expr,
    ) -> Result<Value, CodegenError> {
        let member_name = self.resolve_sym(member).to_string();

        // Check if this is a module alias access: math.clamp (non-call context)
        {
            let obj_type = self.node_type(object.id).clone();
            if matches!(obj_type, Type::Module { .. }) {
                // Module member access in non-call context — return function reference
                // This is handled at call site; for now return a dummy value
                if let Some((func_id, _sig)) = self.functions.get(&member_name) {
                    let fref = self.module.declare_func_in_func(*func_id, builder.func);
                    return Ok(builder.ins().func_addr(crate::types::pointer_type(), fref));
                }
                return Err(CodegenError::new(format!(
                    "module has no export '{member_name}'"
                )));
            }
        }

        // Check if this is an enum variant access: EnumName.VariantName
        if let ExprKind::Ident(obj_sym) = &object.kind {
            let obj_name = self.resolve_sym(*obj_sym).to_string();
            if self.enum_defs.contains_key(&obj_name) {
                if let Some(tag) = self.lookup_qualified_variant(&obj_name, &member_name) {
                    return Ok(builder.ins().iconst(cl_types::I64, tag));
                }
                return Err(CodegenError::new(format!(
                    "enum '{obj_name}' has no variant '{member_name}'"
                )));
            }
        }

        let obj_val = self.compile_expr(builder, state, object)?;
        let obj_type = self.node_type(object.id).clone();

        if let Type::Struct { name, .. } = &obj_type {
            let struct_name = self.resolve_sym(*name).to_string();
            if let Some(layout) = self.struct_layouts.get(&struct_name) {
                for field in &layout.fields {
                    if field.name == member_name {
                        let val = builder.ins().load(
                            field.cl_type,
                            MemFlags::trusted(),
                            obj_val,
                            field.offset as i32,
                        );
                        return Ok(val);
                    }
                }
                return Err(CodegenError::new(format!(
                    "struct '{struct_name}' has no field '{member_name}'"
                )));
            }
        }

        if let Type::Model { name, .. } = &obj_type {
            let model_name = self.resolve_sym(*name).to_string();
            // Check if this field is a FixedModelArray — return the address, not a loaded value
            if let Some(field_type_map) = self.model_field_types.get(&model_name).cloned() {
                if let Some(array_marker) = field_type_map.get(&member_name).cloned() {
                    if array_marker.starts_with('[') {
                        if let Some(layout) = self.struct_layouts.get(&model_name).cloned() {
                            for field in &layout.fields {
                                if field.name == member_name {
                                    let offset_val = builder.ins().iconst(cl_types::I64, field.offset as i64);
                                    return Ok(builder.ins().iadd(obj_val, offset_val));
                                }
                            }
                        }
                    }
                }
            }
            if let Some(layout) = self.struct_layouts.get(&model_name).cloned() {
                for field in &layout.fields {
                    if field.name == member_name {
                        let val = builder.ins().load(
                            field.cl_type,
                            MemFlags::trusted(),
                            obj_val,
                            field.offset as i32,
                        );
                        return Ok(val);
                    }
                }
                return Err(CodegenError::new(format!(
                    "model '{model_name}' has no field '{member_name}'"
                )));
            }
        }

        // Tensor property access
        if obj_type.is_tensor() {
            return match member_name.as_str() {
                "shape" => self.compile_call_by_name(builder, "nsl_tensor_shape", &[obj_val]),
                "ndim" => self.compile_call_by_name(builder, "nsl_tensor_ndim", &[obj_val]),
                _ => Err(CodegenError::new(format!("unknown tensor property '.{member_name}'"))),
            };
        }

        Err(CodegenError::new(format!("member access not supported: .{member_name}")))
    }

    // ── F-string ────────────────────────────────────────────────────

    fn compile_fstring(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        parts: &[FStringPart],
    ) -> Result<Value, CodegenError> {
        if parts.is_empty() {
            return self.compile_string_literal(builder, "");
        }

        let concat_id = self.runtime_fns["nsl_str_concat"].0;
        let mut result: Option<Value> = None;

        for part in parts {
            let part_str = match part {
                FStringPart::Text(s) => self.compile_string_literal(builder, s)?,
                FStringPart::Expr(expr) => {
                    let val = self.compile_expr(builder, state, expr)?;
                    let expr_type = self.node_type(expr.id).clone();
                    self.value_to_string(builder, val, &expr_type)?
                }
            };

            result = Some(match result {
                None => part_str,
                Some(prev) => {
                    let concat_ref = self.module.declare_func_in_func(concat_id, builder.func);
                    let call = builder.ins().call(concat_ref, &[prev, part_str]);
                    builder.inst_results(call)[0]
                }
            });
        }
        Ok(result.unwrap())
    }

    fn value_to_string(
        &mut self,
        builder: &mut FunctionBuilder,
        val: Value,
        ty: &Type,
    ) -> Result<Value, CodegenError> {
        let rt_fn = match ty {
            Type::Int | Type::Int64 | Type::Int32 | Type::Int16 | Type::Int8 => "nsl_int_to_str",
            Type::Float | Type::F64 | Type::F32 => "nsl_float_to_str",
            Type::Bool => "nsl_bool_to_str",
            Type::Str => return Ok(val),
            _ => "nsl_int_to_str",
        };
        let fid = self.runtime_fns[rt_fn].0;
        let fref = self.module.declare_func_in_func(fid, builder.func);
        let call = builder.ins().call(fref, &[val]);
        Ok(builder.inst_results(call)[0])
    }

    // ── If expression ───────────────────────────────────────────────

    fn compile_if_expr(
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

    pub fn expr_as_func_name(&self, expr: &Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::Ident(sym) => Some(self.resolve_sym(*sym).to_string()),
            _ => None,
        }
    }

    // ── Subscript access ────────────────────────────────────────────

    fn compile_subscript(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        object: &Expr,
        index: &SubscriptKind,
    ) -> Result<Value, CodegenError> {
        let obj_val = self.compile_expr(builder, state, object)?;
        match index {
            SubscriptKind::Index(idx_expr) => {
                let idx_val = self.compile_expr(builder, state, idx_expr)?;
                let obj_type = self.node_type(object.id).clone();
                match &obj_type {
                    Type::Dict { .. } => {
                        let fid = self.runtime_fns["nsl_dict_get_str"].0;
                        let fref = self.module.declare_func_in_func(fid, builder.func);
                        let call = builder.ins().call(fref, &[obj_val, idx_val]);
                        Ok(builder.inst_results(call)[0])
                    }
                    _ => {
                        // Default: list subscript
                        let fid = self.runtime_fns["nsl_list_get"].0;
                        let fref = self.module.declare_func_in_func(fid, builder.func);
                        let call = builder.ins().call(fref, &[obj_val, idx_val]);
                        Ok(builder.inst_results(call)[0])
                    }
                }
            }
            SubscriptKind::Slice { lower, upper, step } => {
                let sentinel = builder.ins().iconst(cl_types::I64, i64::MIN);
                let lo = if let Some(e) = lower {
                    self.compile_expr(builder, state, e)?
                } else {
                    sentinel
                };
                let hi = if let Some(e) = upper {
                    self.compile_expr(builder, state, e)?
                } else {
                    builder.ins().iconst(cl_types::I64, i64::MIN)
                };
                let st = if let Some(e) = step {
                    self.compile_expr(builder, state, e)?
                } else {
                    builder.ins().iconst(cl_types::I64, i64::MIN)
                };

                let obj_type = self.node_type(object.id).clone();
                let fn_name = if matches!(obj_type, Type::Str) {
                    "nsl_str_slice"
                } else {
                    "nsl_list_slice"
                };
                let fid = self.runtime_fns[fn_name].0;
                let fref = self.module.declare_func_in_func(fid, builder.func);
                let call = builder.ins().call(fref, &[obj_val, lo, hi, st]);
                Ok(builder.inst_results(call)[0])
            }
            _ => Err(CodegenError::new("multi-dim subscript not supported yet")),
        }
    }

    // ── Type conversions ────────────────────────────────────────────

    fn compile_type_conversion(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        target_type: &str,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        if args.len() != 1 {
            return Err(CodegenError::new(format!("{target_type}() takes exactly 1 argument")));
        }
        let val = self.compile_expr(builder, state, &args[0].value)?;
        let src_type = self.node_type(args[0].value.id).clone();

        match target_type {
            "int" => match &src_type {
                Type::Int | Type::Int64 | Type::Int32 | Type::Int16 | Type::Int8 => Ok(val),
                Type::Float | Type::F64 | Type::F32 => {
                    Ok(builder.ins().fcvt_to_sint_sat(cl_types::I64, val))
                }
                Type::Bool => Ok(builder.ins().uextend(cl_types::I64, val)),
                Type::Str => {
                    let fid = self.runtime_fns["nsl_str_to_int"].0;
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[val]);
                    Ok(builder.inst_results(call)[0])
                }
                _ => Ok(val),
            },
            "float" => match &src_type {
                Type::Float | Type::F64 | Type::F32 => Ok(val),
                Type::Int | Type::Int64 | Type::Int32 | Type::Int16 | Type::Int8 => {
                    Ok(builder.ins().fcvt_from_sint(cl_types::F64, val))
                }
                Type::Bool => {
                    let ext = builder.ins().uextend(cl_types::I64, val);
                    Ok(builder.ins().fcvt_from_sint(cl_types::F64, ext))
                }
                Type::Str => {
                    let fid = self.runtime_fns["nsl_str_to_float"].0;
                    let fref = self.module.declare_func_in_func(fid, builder.func);
                    let call = builder.ins().call(fref, &[val]);
                    Ok(builder.inst_results(call)[0])
                }
                _ => Ok(builder.ins().fcvt_from_sint(cl_types::F64, val)),
            },
            "str" => self.value_to_string(builder, val, &src_type),
            "bool" => match &src_type {
                Type::Bool => Ok(val),
                Type::Int | Type::Int64 | Type::Int32 | Type::Int16 | Type::Int8 => {
                    Ok(builder.ins().icmp_imm(IntCC::NotEqual, val, 0))
                }
                Type::Float | Type::F64 | Type::F32 => {
                    let zero = builder.ins().f64const(0.0);
                    Ok(builder.ins().fcmp(FloatCC::NotEqual, val, zero))
                }
                _ => Ok(builder.ins().icmp_imm(IntCC::NotEqual, val, 0)),
            },
            _ => Err(CodegenError::new(format!("unknown type conversion: {target_type}()"))),
        }
    }

    // ── List comprehension ──────────────────────────────────────────

    fn compile_list_comp(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        element: &Expr,
        generators: &[nsl_ast::expr::CompGenerator],
    ) -> Result<Value, CodegenError> {
        if generators.is_empty() {
            return Err(CodegenError::new("list comprehension needs at least one generator"));
        }
        // Create a new list
        let new_id = self.runtime_fns["nsl_list_new"].0;
        let new_ref = self.module.declare_func_in_func(new_id, builder.func);
        let call = builder.ins().call(new_ref, &[]);
        let result_list = builder.inst_results(call)[0];

        // Only support single generator for now
        let gen = &generators[0];
        let sym = match &gen.pattern.kind {
            PatternKind::Ident(sym) => *sym,
            _ => return Err(CodegenError::new("only simple ident patterns in list comprehensions")),
        };

        let iter_val = self.compile_expr(builder, state, &gen.iterable)?;

        // Get list length
        let len_id = self.runtime_fns["nsl_list_len"].0;
        let len_ref = self.module.declare_func_in_func(len_id, builder.func);
        let call = builder.ins().call(len_ref, &[iter_val]);
        let list_len = builder.inst_results(call)[0];

        // Counter variable
        let counter_var = state.new_variable();
        builder.declare_var(counter_var, cl_types::I64);
        let zero = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(counter_var, zero);

        // Element variable
        let elem_var = state.new_variable();
        builder.declare_var(elem_var, cl_types::I64);
        builder.def_var(elem_var, zero);
        state.variables.insert(sym, (elem_var, cl_types::I64));

        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let increment_block = builder.create_block();
        let exit_block = builder.create_block();

        builder.ins().jump(header_block, &[]);

        // Header: check counter < len
        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let counter = builder.use_var(counter_var);
        let cond = builder.ins().icmp(IntCC::SignedLessThan, counter, list_len);
        builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        // Body: get element, evaluate conditions, push to result
        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        let get_id = self.runtime_fns["nsl_list_get"].0;
        let get_ref = self.module.declare_func_in_func(get_id, builder.func);
        let counter = builder.use_var(counter_var);
        let call = builder.ins().call(get_ref, &[iter_val, counter]);
        let elem = builder.inst_results(call)[0];
        builder.def_var(elem_var, elem);

        // Handle filter conditions
        let push_block = if gen.conditions.is_empty() {
            body_block // no filter, push directly
        } else {
            let push_bb = builder.create_block();
            let mut current_block_id = body_block;
            for cond_expr in &gen.conditions {
                let cond_val = self.compile_expr(builder, state, cond_expr)?;
                let next_check = builder.create_block();
                builder.ins().brif(cond_val, next_check, &[], increment_block, &[]);
                builder.switch_to_block(next_check);
                builder.seal_block(next_check);
                state.current_block = Some(next_check);
                current_block_id = next_check;
            }
            // Jump from last condition check to push block
            builder.ins().jump(push_bb, &[]);
            builder.switch_to_block(push_bb);
            builder.seal_block(push_bb);
            state.current_block = Some(push_bb);
            let _ = current_block_id;
            push_bb
        };

        // Evaluate element expression and push
        let elem_val = self.compile_expr(builder, state, element)?;
        let push_id = self.runtime_fns["nsl_list_push"].0;
        let push_ref = self.module.declare_func_in_func(push_id, builder.func);
        builder.ins().call(push_ref, &[result_list, elem_val]);

        let _ = push_block;
        builder.ins().jump(increment_block, &[]);

        // Increment
        builder.switch_to_block(increment_block);
        builder.seal_block(increment_block);
        state.current_block = Some(increment_block);
        let counter = builder.use_var(counter_var);
        let one = builder.ins().iconst(cl_types::I64, 1);
        let next = builder.ins().iadd(counter, one);
        builder.def_var(counter_var, next);
        builder.ins().jump(header_block, &[]);

        builder.seal_block(header_block);
        builder.switch_to_block(exit_block);
        builder.seal_block(exit_block);
        state.current_block = Some(exit_block);

        Ok(result_list)
    }

    // ── Tuple literal ──────────────────────────────────────────────

    fn compile_tuple_literal(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        elements: &[Expr],
    ) -> Result<Value, CodegenError> {
        // Tuples reuse list representation (heap-allocated i64 array)
        let new_id = self.runtime_fns["nsl_list_new"].0;
        let new_ref = self.module.declare_func_in_func(new_id, builder.func);
        let call = builder.ins().call(new_ref, &[]);
        let tuple_ptr = builder.inst_results(call)[0];

        let push_id = self.runtime_fns["nsl_list_push"].0;
        let push_ref = self.module.declare_func_in_func(push_id, builder.func);
        for elem in elements {
            let val = self.compile_expr(builder, state, elem)?;
            builder.ins().call(push_ref, &[tuple_ptr, val]);
        }
        Ok(tuple_ptr)
    }

    // ── Lambda (with optional closure support) ─────────────────────

    fn compile_lambda(
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

        // Add capture params to signature (all I64 — pointers or ints)
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
            // No captures — return bare function pointer (backward compatible)
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

            // Record how many captures this closure has (keyed by a sentinel — caller will
            // transfer this to the variable that receives the closure result).
            self.last_lambda_capture_count = Some(captures.len());

            Ok(closure_ptr)
        }
    }

    /// Walk an expression AST to find variables that reference outer-scope bindings,
    /// i.e. not lambda params and not function/builtin names.
    fn find_free_variables(
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

    fn collect_free_vars(
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
                        // It's an outer-scope variable — it's a capture
                        seen.insert(*sym);
                        out.push((*sym, *cl_type));
                    }
                    // If not in state.variables, it's a global/builtin — not a capture
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
            // Literals and other leaf nodes — no free variables
            _ => {}
        }
    }

    // ── Dict literal ────────────────────────────────────────────────

    fn compile_dict_literal(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        pairs: &[(Expr, Expr)],
    ) -> Result<Value, CodegenError> {
        let new_id = self.runtime_fns["nsl_dict_new"].0;
        let new_ref = self.module.declare_func_in_func(new_id, builder.func);
        let call = builder.ins().call(new_ref, &[]);
        let dict_ptr = builder.inst_results(call)[0];

        let set_id = self.runtime_fns["nsl_dict_set_str"].0;
        let set_ref = self.module.declare_func_in_func(set_id, builder.func);
        for (key_expr, val_expr) in pairs {
            // Bare ident keys are treated as string literals (matching checker behavior)
            let key_val = match &key_expr.kind {
                ExprKind::Ident(sym) => {
                    let name = self.resolve_sym(*sym).to_string();
                    self.compile_string_literal(builder, &name)?
                }
                _ => self.compile_expr(builder, state, key_expr)?,
            };
            let val_val = self.compile_expr(builder, state, val_expr)?;
            builder.ins().call(set_ref, &[dict_ptr, key_val, val_val]);
        }
        Ok(dict_ptr)
    }

    // ── Indirect call (for lambda variables) ───────────────────────

    fn compile_indirect_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        callee: &Expr,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        let callee_type = self.node_type(callee.id).clone();

        // Check if callee is a closure variable (has captures)
        let closure_captures = if let ExprKind::Ident(sym) = &callee.kind {
            self.closure_info.get(sym).copied()
        } else {
            None
        };

        let raw_val = self.compile_expr(builder, state, callee)?;

        if let Type::Function { params: param_types, ret } = &callee_type {
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

    // ── String method calls ────────────────────────────────────────

    fn compile_str_method_call(
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

    // ── List method calls ──────────────────────────────────────────

    fn compile_list_method_call(
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

    // ── Higher-order function calls ────────────────────────────────

    fn compile_higher_order_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        func_name: &str,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        if args.len() != 2 {
            return Err(CodegenError::new(format!("{func_name}() takes exactly 2 arguments")));
        }
        // Error if first arg is a closure (captures variables) — HOFs in C runtime expect bare fn ptrs
        if let ExprKind::Ident(sym) = &args[0].value.kind {
            if self.closure_info.contains_key(sym) {
                let name = self.resolve_sym(*sym).to_string();
                return Err(CodegenError::new(format!(
                    "cannot pass closure '{name}' to {func_name}() — closures with captured variables \
                     are not supported in higher-order functions yet. Use a non-capturing lambda instead."
                )));
            }
        }
        let fn_val = self.compile_expr(builder, state, &args[0].value)?;
        // Also catch inline capturing lambdas (not just named variables)
        if self.last_lambda_capture_count.take().is_some() {
            return Err(CodegenError::new(format!(
                "cannot pass capturing closure to {func_name}() — closures with captured variables \
                 are not supported in higher-order functions yet. Use a non-capturing lambda instead."
            )));
        }
        let list_val = self.compile_expr(builder, state, &args[1].value)?;
        let rt_name = format!("nsl_{func_name}");
        self.compile_call_by_name(builder, &rt_name, &[fn_val, list_val])
    }

    // ── Match expression ───────────────────────────────────────────

    fn compile_match_expr(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        subject: &Expr,
        arms: &[nsl_ast::expr::MatchArm],
        full_expr: &Expr,
    ) -> Result<Value, CodegenError> {
        let subject_val = self.compile_expr(builder, state, subject)?;
        let result_type = nsl_type_to_cl(&self.node_type(full_expr.id).clone());
        let merge_block = builder.create_block();
        builder.append_block_param(merge_block, result_type);

        let mut remaining_arms: Vec<_> = arms.iter().collect();
        let mut needs_default_block = false;
        while !remaining_arms.is_empty() {
            let arm = remaining_arms.remove(0);
            let is_last = remaining_arms.is_empty();

            match &arm.pattern.kind {
                PatternKind::Wildcard => {
                    let arm_val = self.compile_match_arm_value(builder, state, &arm.body, result_type)?;
                    builder.ins().jump(merge_block, &[arm_val]);
                    break;
                }
                PatternKind::Ident(sym) => {
                    let name = self.resolve_sym(*sym).to_string();
                    if let Some(tag) = self.lookup_enum_variant_tag(&name) {
                        let tag_val = builder.ins().iconst(cl_types::I64, tag);
                        let cmp = builder.ins().icmp(IntCC::Equal, subject_val, tag_val);
                        let arm_block = builder.create_block();
                        let next_block = builder.create_block();
                        builder.ins().brif(cmp, arm_block, &[], next_block, &[]);

                        builder.switch_to_block(arm_block);
                        builder.seal_block(arm_block);
                        state.current_block = Some(arm_block);
                        let arm_val = self.compile_match_arm_value(builder, state, &arm.body, result_type)?;
                        builder.ins().jump(merge_block, &[arm_val]);

                        builder.switch_to_block(next_block);
                        builder.seal_block(next_block);
                        state.current_block = Some(next_block);
                        if is_last { needs_default_block = true; }
                    } else {
                        // Binding: bind subject to variable
                        let var = state.new_variable();
                        builder.declare_var(var, cl_types::I64);
                        builder.def_var(var, subject_val);
                        state.variables.insert(*sym, (var, cl_types::I64));
                        let arm_val = self.compile_match_arm_value(builder, state, &arm.body, result_type)?;
                        builder.ins().jump(merge_block, &[arm_val]);
                        break;
                    }
                }
                PatternKind::Literal(lit_expr) => {
                    let lit_val = self.compile_expr(builder, state, lit_expr)?;
                    let lit_type = self.node_type(lit_expr.id).clone();
                    let cmp = if is_float_type(&lit_type) {
                        builder.ins().fcmp(FloatCC::Equal, subject_val, lit_val)
                    } else {
                        builder.ins().icmp(IntCC::Equal, subject_val, lit_val)
                    };
                    let arm_block = builder.create_block();
                    let next_block = builder.create_block();
                    builder.ins().brif(cmp, arm_block, &[], next_block, &[]);

                    builder.switch_to_block(arm_block);
                    builder.seal_block(arm_block);
                    state.current_block = Some(arm_block);
                    let arm_val = self.compile_match_arm_value(builder, state, &arm.body, result_type)?;
                    builder.ins().jump(merge_block, &[arm_val]);

                    builder.switch_to_block(next_block);
                    builder.seal_block(next_block);
                    state.current_block = Some(next_block);
                    if is_last { needs_default_block = true; }
                }
                PatternKind::Constructor { path, .. } => {
                    let variant_name = self.resolve_sym(*path.last().unwrap()).to_string();
                    if let Some(tag) = self.lookup_enum_variant_tag(&variant_name) {
                        let tag_val = builder.ins().iconst(cl_types::I64, tag);
                        let cmp = builder.ins().icmp(IntCC::Equal, subject_val, tag_val);
                        let arm_block = builder.create_block();
                        let next_block = builder.create_block();
                        builder.ins().brif(cmp, arm_block, &[], next_block, &[]);

                        builder.switch_to_block(arm_block);
                        builder.seal_block(arm_block);
                        state.current_block = Some(arm_block);
                        let arm_val = self.compile_match_arm_value(builder, state, &arm.body, result_type)?;
                        builder.ins().jump(merge_block, &[arm_val]);

                        builder.switch_to_block(next_block);
                        builder.seal_block(next_block);
                        state.current_block = Some(next_block);
                        if is_last { needs_default_block = true; }
                    } else {
                        return Err(CodegenError::new(format!("unknown variant '{variant_name}' in match expr")));
                    }
                }
                _ => return Err(CodegenError::new("unsupported pattern in match expression")),
            }
        }

        // If last arm was not a wildcard, emit a default value for the no-match path
        if needs_default_block {
            let default_val = if result_type.is_float() {
                builder.ins().f64const(0.0)
            } else {
                builder.ins().iconst(result_type, 0)
            };
            builder.ins().jump(merge_block, &[default_val]);
        }

        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);
        state.current_block = Some(merge_block);
        Ok(builder.block_params(merge_block)[0])
    }

    fn compile_match_arm_value(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        body: &nsl_ast::stmt::Block,
        result_type: cl_types::Type,
    ) -> Result<Value, CodegenError> {
        if body.stmts.is_empty() {
            let zero = if result_type.is_float() {
                builder.ins().f64const(0.0)
            } else {
                builder.ins().iconst(result_type, 0)
            };
            return Ok(zero);
        }
        // Compile all statements except the last
        for s in &body.stmts[..body.stmts.len() - 1] {
            self.compile_stmt(builder, state, s)?;
        }
        // The last statement should be an expression producing the arm's value
        let last = &body.stmts[body.stmts.len() - 1];
        match &last.kind {
            nsl_ast::stmt::StmtKind::Expr(expr) => {
                self.compile_expr(builder, state, expr)
            }
            _ => {
                self.compile_stmt(builder, state, last)?;
                let zero = if result_type.is_float() {
                    builder.ins().f64const(0.0)
                } else {
                    builder.ins().iconst(result_type, 0)
                };
                Ok(zero)
            }
        }
    }

    // ── Range expression ───────────────────────────────────────────

    fn compile_range_expr(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        start: Option<&Expr>,
        end: Option<&Expr>,
        inclusive: bool,
    ) -> Result<Value, CodegenError> {
        let start_val = if let Some(s) = start {
            self.compile_expr(builder, state, s)?
        } else {
            builder.ins().iconst(cl_types::I64, 0)
        };
        let end_val = if let Some(e) = end {
            let val = self.compile_expr(builder, state, e)?;
            if inclusive {
                let one = builder.ins().iconst(cl_types::I64, 1);
                builder.ins().iadd(val, one)
            } else {
                val
            }
        } else {
            return Err(CodegenError::new("open-ended range not supported"));
        };
        let step = builder.ins().iconst(cl_types::I64, 1);
        let fid = self.runtime_fns["nsl_range"].0;
        let fref = self.module.declare_func_in_func(fid, builder.func);
        let call = builder.ins().call(fref, &[start_val, end_val, step]);
        Ok(builder.inst_results(call)[0])
    }

    // ── Tensor operations ──────────────────────────────────────────

    fn compile_tensor_binary_op(
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
                BinOp::MatMul => "nsl_tensor_matmul",
                _ => return Err(CodegenError::new(format!("unsupported tensor op: {op:?}"))),
            };
            let result = self.compile_call_by_name(builder, rt_name, &[lhs, rhs])?;
            state.tensor_temporaries.push(result);
            Ok(result)
        } else if left_is_tensor {
            // Tensor-scalar: ensure scalar is f64
            let rhs_ty = builder.func.dfg.value_type(rhs);
            let scalar = if rhs_ty == cl_types::F64 { rhs } else { builder.ins().fcvt_from_sint(cl_types::F64, rhs) };
            let result = match op {
                BinOp::Add => self.compile_call_by_name(builder, "nsl_tensor_add_scalar", &[lhs, scalar])?,
                BinOp::Mul => self.compile_call_by_name(builder, "nsl_tensor_mul_scalar", &[lhs, scalar])?,
                BinOp::Sub => {
                    // tensor - scalar = tensor + (-scalar)
                    let neg = builder.ins().fneg(scalar);
                    self.compile_call_by_name(builder, "nsl_tensor_add_scalar", &[lhs, neg])?
                }
                BinOp::Div => {
                    // tensor / scalar = tensor * (1/scalar)
                    let one = builder.ins().f64const(1.0);
                    let inv = builder.ins().fdiv(one, scalar);
                    self.compile_call_by_name(builder, "nsl_tensor_mul_scalar", &[lhs, inv])?
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
                BinOp::Add => self.compile_call_by_name(builder, "nsl_tensor_add_scalar", &[rhs, scalar])?,
                BinOp::Mul => self.compile_call_by_name(builder, "nsl_tensor_mul_scalar", &[rhs, scalar])?,
                BinOp::Sub => {
                    // scalar - tensor = neg(tensor) + scalar
                    let neg_tensor = self.compile_call_by_name(builder, "nsl_tensor_neg", &[rhs])?;
                    state.tensor_temporaries.push(neg_tensor);
                    self.compile_call_by_name(builder, "nsl_tensor_add_scalar", &[neg_tensor, scalar])?
                }
                _ => return Err(CodegenError::new(format!("unsupported scalar-tensor op: {op:?}"))),
            };
            state.tensor_temporaries.push(result);
            Ok(result)
        }
    }

    fn compile_model_method_call(
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

    fn compile_tensor_method_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        object: &Expr,
        method: &str,
        args: &[nsl_ast::expr::Arg],
    ) -> Result<Value, CodegenError> {
        let obj_val = self.compile_expr(builder, state, object)?;
        match method {
            "sum" => self.compile_call_by_name(builder, "nsl_tensor_sum", &[obj_val]),
            "mean" => self.compile_call_by_name(builder, "nsl_tensor_mean", &[obj_val]),
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
            "clone" => self.compile_call_by_name(builder, "nsl_tensor_clone", &[obj_val]),
            "item" => self.compile_call_by_name(builder, "nsl_tensor_item", &[obj_val]),
            "to" => {
                if args.len() != 1 {
                    return Err(CodegenError::new("to() takes exactly 1 argument (device)"));
                }
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
            "slice" => {
                if args.len() != 3 {
                    return Err(CodegenError::new("slice() takes exactly 3 arguments (dim, start, end)"));
                }
                let dim_val = self.compile_expr(builder, state, &args[0].value)?;
                let start_val = self.compile_expr(builder, state, &args[1].value)?;
                let end_val = self.compile_expr(builder, state, &args[2].value)?;
                self.compile_call_by_name(builder, "nsl_tensor_slice", &[obj_val, dim_val, start_val, end_val])
            }
            _ => Err(CodegenError::new(format!("unknown tensor method '.{method}()'"))),
        }
    }

    fn compile_kernel_call(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        _kernel_name: &str,
        args: &[nsl_ast::expr::Arg],
        ptx_data_id: DataId,
        name_data_id: DataId,
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
        let ptx_ptr = builder.ins().global_value(cl_types::I64, ptx_gv);

        // Get kernel name pointer from .rodata
        let name_gv = self.module.declare_data_in_func(name_data_id, builder.func);
        let name_ptr = builder.ins().global_value(cl_types::I64, name_gv);

        // Build args array on stack: each tensor arg is an i64 (pointer)
        let num_args = tensor_args.len();
        if num_args == 0 {
            // No tensor args — pass null pointer and 0 count
            let null_ptr = builder.ins().iconst(cl_types::I64, 0);
            let num_args_val = builder.ins().iconst(cl_types::I64, 0);
            return self.compile_call_by_name(
                builder,
                "nsl_kernel_launch",
                &[ptx_ptr, name_ptr, grid_x, grid_y, grid_z, block_x, block_y, block_z, null_ptr, num_args_val],
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

        // Call nsl_kernel_launch(ptx_ptr, name_ptr, grid_x, grid_y, grid_z,
        //                        block_x, block_y, block_z, args_ptr, num_args)
        self.compile_call_by_name(
            builder,
            "nsl_kernel_launch",
            &[ptx_ptr, name_ptr, grid_x, grid_y, grid_z, block_x, block_y, block_z, args_ptr, num_args_val],
        )
    }
}
