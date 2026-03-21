use cranelift_codegen::ir::condcodes::{FloatCC, IntCC};
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;

use nsl_ast::expr::{Expr, ExprKind, FStringPart};
use nsl_ast::pattern::PatternKind;
use nsl_semantic::types::Type;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::types::{is_float_type, nsl_type_to_cl, pointer_type};

impl Compiler<'_> {
    pub(crate) fn compile_string_literal(
        &mut self,
        builder: &mut FunctionBuilder,
        s: &str,
    ) -> Result<Value, CodegenError> {
        // Auto-intern if not already in pool (handles strings inside train blocks,
        // callbacks, and dict member access keys that the collection pass missed)
        if !self.string_pool.contains_key(s) {
            self.intern_string(s)?;
        }
        let data_id = self.string_pool[s];
        let gv = self.module.declare_data_in_func(data_id, builder.func);
        Ok(builder.ins().symbol_value(pointer_type(), gv))
    }

    pub(crate) fn compile_list_literal(
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
            // nsl_list_push expects i64; convert f64 values (e.g. from Int / Int
            // which the semantic pass types as Float) to i64 via fcvt_to_sint.
            let val = if builder.func.dfg.value_type(val).is_float() {
                builder.ins().fcvt_to_sint_sat(cl_types::I64, val)
            } else {
                val
            };
            builder.ins().call(push_ref, &[list_ptr, val]);
        }
        Ok(list_ptr)
    }

    pub(crate) fn compile_tuple_literal(
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

    pub(crate) fn compile_dict_literal(
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

    pub(crate) fn compile_fstring(
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

    pub(crate) fn value_to_string(
        &mut self,
        builder: &mut FunctionBuilder,
        val: Value,
        ty: &Type,
    ) -> Result<Value, CodegenError> {
        let (rt_fn, converted_val) = match ty {
            Type::Int | Type::Int64 => ("nsl_int_to_str", val),
            Type::Int32 | Type::Int16 | Type::Int8 => {
                ("nsl_int_to_str", builder.ins().sextend(cl_types::I64, val))
            }
            Type::Float | Type::F64 => ("nsl_float_to_str", val),
            Type::F32 => {
                ("nsl_float_to_str", builder.ins().fpromote(cl_types::F64, val))
            }
            Type::Bool => ("nsl_bool_to_str", val),
            Type::Str => return Ok(val),
            _ => ("nsl_int_to_str", val),
        };
        let fid = self.runtime_fns[rt_fn].0;
        let fref = self.module.declare_func_in_func(fid, builder.func);
        let call = builder.ins().call(fref, &[converted_val]);
        Ok(builder.inst_results(call)[0])
    }

    pub(crate) fn compile_range_expr(
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

    pub(crate) fn compile_list_comp(
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

    pub(crate) fn compile_match_expr(
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
            let default_val = if result_type == cl_types::F64 {
                builder.ins().f64const(0.0)
            } else if result_type == cl_types::F32 {
                builder.ins().f32const(0.0)
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

    pub(crate) fn compile_match_arm_value(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        body: &nsl_ast::stmt::Block,
        result_type: cl_types::Type,
    ) -> Result<Value, CodegenError> {
        if body.stmts.is_empty() {
            let zero = if result_type == cl_types::F64 {
                builder.ins().f64const(0.0)
            } else if result_type == cl_types::F32 {
                builder.ins().f32const(0.0)
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
}
