use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, MemFlags};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;

use nsl_ast::operator::AssignOp;
use nsl_ast::pattern::PatternKind;
use nsl_ast::expr::{ExprKind, SubscriptKind};
use nsl_ast::block::{QuantDtype, QuantGranularity, TrainSection};
use nsl_ast::stmt::{Stmt, StmtKind};

use crate::compiler::Compiler;
use crate::context::{FuncState, LoopContext};
use crate::error::CodegenError;
use crate::types::{is_block_filled, is_float_type, nsl_type_to_cl};

impl Compiler<'_> {
    pub fn compile_stmt(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        stmt: &Stmt,
    ) -> Result<(), CodegenError> {
        if let Some(block) = state.current_block {
            if is_block_filled(builder, block) {
                return Ok(());
            }
        }

        // Clear any stale lambda capture count from a previous statement
        // (only VarDecl should consume this; if it leaks past a statement boundary it's a bug)
        self.last_lambda_capture_count = None;

        match &stmt.kind {
            StmtKind::VarDecl { pattern, value, .. } => {
                match &pattern.kind {
                    PatternKind::Ident(sym) => {
                        let sym = *sym;
                        let init_val = if let Some(expr) = value {
                            self.compile_expr(builder, state, expr)?
                        } else {
                            builder.ins().iconst(cl_types::I64, 0)
                        };

                        let cl_type = if let Some(expr) = value {
                            nsl_type_to_cl(&self.node_type(expr.id).clone())
                        } else {
                            cl_types::I64
                        };

                        if let Some((var, _)) = state.variables.get(&sym) {
                            builder.def_var(*var, init_val);
                        } else {
                            let var = state.new_variable();
                            builder.declare_var(var, cl_type);
                            builder.def_var(var, init_val);
                            state.variables.insert(sym, (var, cl_type));
                        }

                        // If the value was a closure lambda, record capture count for indirect call dispatch
                        if let Some(count) = self.last_lambda_capture_count.take() {
                            self.closure_info.insert(sym, count);
                        }
                    }
                    PatternKind::Tuple(sub_patterns) => {
                        let tuple_val = if let Some(expr) = value {
                            self.compile_expr(builder, state, expr)?
                        } else {
                            return Err(CodegenError::new("tuple destructuring requires a value"));
                        };

                        let get_id = self.runtime_fns["nsl_list_get"].0;
                        let get_ref = self.module.declare_func_in_func(get_id, builder.func);

                        for (i, sub_pat) in sub_patterns.iter().enumerate() {
                            match &sub_pat.kind {
                                PatternKind::Ident(sym) => {
                                    let idx = builder.ins().iconst(cl_types::I64, i as i64);
                                    let call = builder.ins().call(get_ref, &[tuple_val, idx]);
                                    let elem = builder.inst_results(call)[0];
                                    let var = state.new_variable();
                                    builder.declare_var(var, cl_types::I64);
                                    builder.def_var(var, elem);
                                    state.variables.insert(*sym, (var, cl_types::I64));
                                }
                                PatternKind::Wildcard => {}
                                _ => return Err(CodegenError::new("nested destructuring not yet supported")),
                            }
                        }
                    }
                    _ => return Err(CodegenError::new("only ident and tuple patterns supported")),
                }
            }

            StmtKind::Assign { target, op, value } => {
                self.compile_assign(builder, state, target, *op, value)?;
            }

            StmtKind::Return(expr) => {
                if let Some(e) = expr {
                    let val = self.compile_expr(builder, state, e)?;
                    // @no_grad: resume tape before explicit return
                    if state.is_no_grad {
                        self.compile_call_by_name(builder, "nsl_tape_resume", &[])?;
                    }
                    builder.ins().return_(&[val]);
                } else {
                    // @no_grad: resume tape before explicit return
                    if state.is_no_grad {
                        self.compile_call_by_name(builder, "nsl_tape_resume", &[])?;
                    }
                    builder.ins().return_(&[]);
                }
            }

            StmtKind::Expr(expr) => {
                let _ = self.compile_expr(builder, state, expr)?;
            }

            StmtKind::If { condition, then_block, elif_clauses, else_block } => {
                self.compile_if_stmt(builder, state, condition, then_block, elif_clauses, else_block)?;
            }

            StmtKind::While { condition, body } => {
                self.compile_while(builder, state, condition, body)?;
            }

            StmtKind::For { pattern, iterable, body } => {
                self.compile_for(builder, state, pattern, iterable, body)?;
            }

            StmtKind::Match { subject, arms } => {
                self.compile_match(builder, state, subject, arms)?;
            }

            StmtKind::Break => {
                if let Some(lc) = state.loop_stack.last() {
                    builder.ins().jump(lc.exit_block, &[]);
                } else {
                    return Err(CodegenError::new("break outside loop"));
                }
            }

            StmtKind::Continue => {
                if let Some(lc) = state.loop_stack.last() {
                    builder.ins().jump(lc.continue_block, &[]);
                } else {
                    return Err(CodegenError::new("continue outside loop"));
                }
            }

            StmtKind::FnDef(fn_def) => {
                // Nested function definition: declare, compile, and bind name
                let base_name = self.resolve_sym(fn_def.name).to_string();
                let unique_name = format!("__nsl_nested_{}_{}", base_name, self.next_func_index());
                let sig = self.build_fn_signature(fn_def);
                let func_id = self.module
                    .declare_function(&unique_name, cranelift_module::Linkage::Local, &sig)
                    .map_err(|e| CodegenError::new(format!("failed to declare nested fn '{base_name}': {e}")))?;
                // Temporarily insert under base_name for compile_fn_def lookup, then remove
                self.functions.insert(base_name.clone(), (func_id, sig.clone()));

                // Compile the nested function body
                self.compile_fn_def(fn_def)?;

                // Remove temp entry; the function is accessed via local variable binding below
                self.functions.remove(&base_name);

                // Bind function name as a variable holding the function pointer
                let func_ref = self.module.declare_func_in_func(func_id, builder.func);
                let addr = builder.ins().func_addr(crate::types::pointer_type(), func_ref);
                let var = state.new_variable();
                builder.declare_var(var, cl_types::I64);
                builder.def_var(var, addr);
                state.variables.insert(fn_def.name, (var, cl_types::I64));
            }

            StmtKind::GradBlock(grad) => {
                self.compile_grad_block(builder, state, grad)?;
            }

            StmtKind::TrainBlock(train) => {
                self.compile_train_block(builder, state, train)?;
            }

            StmtKind::StructDef(_) | StmtKind::ModelDef(_)
            | StmtKind::EnumDef(_) | StmtKind::TraitDef(_)
            | StmtKind::Import(_) | StmtKind::FromImport(_)
            | StmtKind::DatasetDef(_) | StmtKind::TokenizerDef(_)
            | StmtKind::KernelDef(_) => {}

            StmtKind::QuantBlock(ref quant) => {
                self.compile_quant_block(builder, state, quant)?;
            }

            StmtKind::WhileLet { pattern, expr, body } => {
                self.compile_while_let(builder, state, pattern, expr, body)?;
            }

            StmtKind::Decorated { decorators, stmt } => {
                // Check for @no_grad on nested function definitions
                if let StmtKind::FnDef(fn_def) = &stmt.kind {
                    for d in decorators {
                        if d.name.len() == 1 {
                            let dname = self.resolve_sym(d.name[0]);
                            if dname == "no_grad" {
                                let fname = self.resolve_sym(fn_def.name).to_string();
                                self.no_grad_fns.insert(fname);
                            }
                        }
                    }
                }
                self.compile_stmt(builder, state, stmt)?;
            }

            _ => {
                return Err(CodegenError::new(format!(
                    "unsupported statement in M3 codegen"
                )));
            }
        }
        Ok(())
    }

    fn compile_assign(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        target: &nsl_ast::expr::Expr,
        op: AssignOp,
        value: &nsl_ast::expr::Expr,
    ) -> Result<(), CodegenError> {
        let new_val = self.compile_expr(builder, state, value)?;
        match &target.kind {
            nsl_ast::expr::ExprKind::Ident(sym) => {
                let (var, _) = *state.variables.get(sym).ok_or_else(|| {
                    CodegenError::new(format!(
                        "undefined variable '{}' in assignment",
                        self.resolve_sym(*sym)
                    ))
                })?;

                let target_type = self.node_type(target.id).clone();
                let is_float = is_float_type(&target_type);

                let final_val = match op {
                    AssignOp::Assign => new_val,
                    AssignOp::AddAssign => {
                        let old = builder.use_var(var);
                        if is_float { builder.ins().fadd(old, new_val) } else { builder.ins().iadd(old, new_val) }
                    }
                    AssignOp::SubAssign => {
                        let old = builder.use_var(var);
                        if is_float { builder.ins().fsub(old, new_val) } else { builder.ins().isub(old, new_val) }
                    }
                    AssignOp::MulAssign => {
                        let old = builder.use_var(var);
                        if is_float { builder.ins().fmul(old, new_val) } else { builder.ins().imul(old, new_val) }
                    }
                    AssignOp::DivAssign => {
                        let old = builder.use_var(var);
                        if is_float { builder.ins().fdiv(old, new_val) } else { builder.ins().sdiv(old, new_val) }
                    }
                };
                builder.def_var(var, final_val);
            }
            nsl_ast::expr::ExprKind::Subscript { object, index } => {
                let obj_val = self.compile_expr(builder, state, object)?;
                let obj_type = self.node_type(object.id).clone();
                match index.as_ref() {
                    SubscriptKind::Index(idx_expr) => {
                        let idx_val = self.compile_expr(builder, state, idx_expr)?;
                        let is_dict = matches!(obj_type, nsl_semantic::types::Type::Dict { .. });

                        let final_val = if matches!(op, AssignOp::Assign) {
                            new_val
                        } else {
                            // Read-modify-write: get old value, apply op, write back
                            let get_fn = if is_dict { "nsl_dict_get_str" } else { "nsl_list_get" };
                            let get_id = self.runtime_fns[get_fn].0;
                            let get_ref = self.module.declare_func_in_func(get_id, builder.func);
                            let call = builder.ins().call(get_ref, &[obj_val, idx_val]);
                            let old_val = builder.inst_results(call)[0];

                            match op {
                                AssignOp::AddAssign => builder.ins().iadd(old_val, new_val),
                                AssignOp::SubAssign => builder.ins().isub(old_val, new_val),
                                AssignOp::MulAssign => builder.ins().imul(old_val, new_val),
                                AssignOp::DivAssign => builder.ins().sdiv(old_val, new_val),
                                _ => unreachable!(),
                            }
                        };

                        let set_fn = if is_dict { "nsl_dict_set_str" } else { "nsl_list_set" };
                        let set_id = self.runtime_fns[set_fn].0;
                        let set_ref = self.module.declare_func_in_func(set_id, builder.func);
                        builder.ins().call(set_ref, &[obj_val, idx_val, final_val]);
                    }
                    _ => return Err(CodegenError::new("only simple index assignment supported")),
                }
            }
            nsl_ast::expr::ExprKind::MemberAccess { object, member } => {
                let obj_val = self.compile_expr(builder, state, object)?;
                let member_name = self.resolve_sym(*member).to_string();
                let obj_type = self.node_type(object.id).clone();
                if let nsl_semantic::types::Type::Struct { name, .. } = &obj_type {
                    let struct_name = self.resolve_sym(*name).to_string();
                    if let Some(layout) = self.struct_layouts.get(&struct_name) {
                        for field in &layout.fields {
                            if field.name == member_name {
                                let final_val = if matches!(op, AssignOp::Assign) {
                                    new_val
                                } else {
                                    let old_val = builder.ins().load(
                                        field.cl_type,
                                        cranelift_codegen::ir::MemFlags::trusted(),
                                        obj_val,
                                        field.offset as i32,
                                    );
                                    match op {
                                        AssignOp::AddAssign => builder.ins().iadd(old_val, new_val),
                                        AssignOp::SubAssign => builder.ins().isub(old_val, new_val),
                                        AssignOp::MulAssign => builder.ins().imul(old_val, new_val),
                                        AssignOp::DivAssign => builder.ins().sdiv(old_val, new_val),
                                        _ => unreachable!(),
                                    }
                                };
                                builder.ins().store(
                                    cranelift_codegen::ir::MemFlags::trusted(),
                                    final_val,
                                    obj_val,
                                    field.offset as i32,
                                );
                                return Ok(());
                            }
                        }
                        return Err(CodegenError::new(format!(
                            "struct '{struct_name}' has no field '{member_name}'"
                        )));
                    }
                }
                if let nsl_semantic::types::Type::Model { name, .. } = &obj_type {
                    let model_name = self.resolve_sym(*name).to_string();
                    if let Some(layout) = self.struct_layouts.get(&model_name) {
                        for field in &layout.fields {
                            if field.name == member_name {
                                let final_val = if matches!(op, AssignOp::Assign) {
                                    new_val
                                } else {
                                    let old_val = builder.ins().load(
                                        field.cl_type,
                                        cranelift_codegen::ir::MemFlags::trusted(),
                                        obj_val,
                                        field.offset as i32,
                                    );
                                    match op {
                                        AssignOp::AddAssign => builder.ins().iadd(old_val, new_val),
                                        AssignOp::SubAssign => builder.ins().isub(old_val, new_val),
                                        AssignOp::MulAssign => builder.ins().imul(old_val, new_val),
                                        AssignOp::DivAssign => builder.ins().sdiv(old_val, new_val),
                                        _ => unreachable!(),
                                    }
                                };
                                builder.ins().store(
                                    cranelift_codegen::ir::MemFlags::trusted(),
                                    final_val,
                                    obj_val,
                                    field.offset as i32,
                                );
                                return Ok(());
                            }
                        }
                        return Err(CodegenError::new(format!(
                            "model '{model_name}' has no field '{member_name}'"
                        )));
                    }
                }
                return Err(CodegenError::new(format!("member assignment not supported for .{member_name}")));
            }
            _ => return Err(CodegenError::new("only variable/subscript/member assignment supported in M4")),
        }
        Ok(())
    }

    fn compile_if_stmt(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        condition: &nsl_ast::expr::Expr,
        then_block: &nsl_ast::stmt::Block,
        elif_clauses: &[(nsl_ast::expr::Expr, nsl_ast::stmt::Block)],
        else_block: &Option<nsl_ast::stmt::Block>,
    ) -> Result<(), CodegenError> {
        let merge_block = builder.create_block();
        let cond_val = self.compile_expr(builder, state, condition)?;

        let then_bb = builder.create_block();
        let next_bb = if !elif_clauses.is_empty() || else_block.is_some() {
            builder.create_block()
        } else {
            merge_block
        };
        builder.ins().brif(cond_val, then_bb, &[], next_bb, &[]);

        builder.switch_to_block(then_bb);
        builder.seal_block(then_bb);
        state.current_block = Some(then_bb);
        for s in &then_block.stmts { self.compile_stmt(builder, state, s)?; }
        if !is_block_filled(builder, then_bb) { builder.ins().jump(merge_block, &[]); }

        let mut current_else = next_bb;
        for (i, (elif_cond, elif_body)) in elif_clauses.iter().enumerate() {
            builder.switch_to_block(current_else);
            builder.seal_block(current_else);
            state.current_block = Some(current_else);
            let elif_cond_val = self.compile_expr(builder, state, elif_cond)?;

            let elif_then = builder.create_block();
            let elif_next = if i + 1 < elif_clauses.len() || else_block.is_some() {
                builder.create_block()
            } else {
                merge_block
            };
            builder.ins().brif(elif_cond_val, elif_then, &[], elif_next, &[]);

            builder.switch_to_block(elif_then);
            builder.seal_block(elif_then);
            state.current_block = Some(elif_then);
            for s in &elif_body.stmts { self.compile_stmt(builder, state, s)?; }
            if !is_block_filled(builder, elif_then) { builder.ins().jump(merge_block, &[]); }

            current_else = elif_next;
        }

        if let Some(else_body) = else_block {
            builder.switch_to_block(current_else);
            builder.seal_block(current_else);
            state.current_block = Some(current_else);
            for s in &else_body.stmts { self.compile_stmt(builder, state, s)?; }
            if !is_block_filled(builder, current_else) { builder.ins().jump(merge_block, &[]); }
        } else if current_else != merge_block {
            builder.switch_to_block(current_else);
            builder.seal_block(current_else);
            state.current_block = Some(current_else);
            builder.ins().jump(merge_block, &[]);
        }

        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);
        state.current_block = Some(merge_block);
        Ok(())
    }

    fn compile_while(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        condition: &nsl_ast::expr::Expr,
        body: &nsl_ast::stmt::Block,
    ) -> Result<(), CodegenError> {
        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let exit_block = builder.create_block();

        builder.ins().jump(header_block, &[]);

        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let cond_val = self.compile_expr(builder, state, condition)?;
        builder.ins().brif(cond_val, body_block, &[], exit_block, &[]);

        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        state.loop_stack.push(LoopContext { continue_block: header_block, exit_block });
        for s in &body.stmts { self.compile_stmt(builder, state, s)?; }
        state.loop_stack.pop();

        if !is_block_filled(builder, body_block) { builder.ins().jump(header_block, &[]); }

        builder.seal_block(header_block);
        builder.switch_to_block(exit_block);
        builder.seal_block(exit_block);
        state.current_block = Some(exit_block);
        Ok(())
    }

    fn compile_while_let(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        pattern: &nsl_ast::pattern::Pattern,
        expr: &nsl_ast::expr::Expr,
        body: &nsl_ast::stmt::Block,
    ) -> Result<(), CodegenError> {
        // Pre-declare the pattern variable before the loop (once per function, not per iteration)
        let pattern_var = match &pattern.kind {
            PatternKind::Ident(sym) => {
                let var = state.new_variable();
                builder.declare_var(var, cl_types::I64);
                let zero = builder.ins().iconst(cl_types::I64, 0);
                builder.def_var(var, zero);
                state.variables.insert(*sym, (var, cl_types::I64));
                Some(var)
            }
            PatternKind::Wildcard => None,
            _ => return Err(CodegenError::new("only ident or wildcard patterns in while-let")),
        };

        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let exit_block = builder.create_block();

        builder.ins().jump(header_block, &[]);

        // Header: evaluate expression, check truthiness (non-zero = continue)
        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let val = self.compile_expr(builder, state, expr)?;
        let cond = builder.ins().icmp_imm(IntCC::NotEqual, val, 0);
        builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        // Body: update pattern variable with current value, execute body
        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        // Update the pattern variable with the value from this iteration
        if let Some(var) = pattern_var {
            builder.def_var(var, val);
        }

        state.loop_stack.push(LoopContext { continue_block: header_block, exit_block });
        for s in &body.stmts { self.compile_stmt(builder, state, s)?; }
        state.loop_stack.pop();

        if !is_block_filled(builder, body_block) {
            builder.ins().jump(header_block, &[]);
        }

        builder.seal_block(header_block);
        builder.switch_to_block(exit_block);
        builder.seal_block(exit_block);
        state.current_block = Some(exit_block);
        Ok(())
    }

    fn compile_for(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        pattern: &nsl_ast::pattern::Pattern,
        iterable: &nsl_ast::expr::Expr,
        body: &nsl_ast::stmt::Block,
    ) -> Result<(), CodegenError> {
        let list_val = self.compile_expr(builder, state, iterable)?;

        let len_id = self.runtime_fns["nsl_list_len"].0;
        let len_ref = self.module.declare_func_in_func(len_id, builder.func);
        let call = builder.ins().call(len_ref, &[list_val]);
        let list_len = builder.inst_results(call)[0];

        let counter_var = state.new_variable();
        builder.declare_var(counter_var, cl_types::I64);
        let zero = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(counter_var, zero);

        // Pre-declare pattern variables before the loop
        match &pattern.kind {
            PatternKind::Ident(sym) => {
                let elem_var = state.new_variable();
                builder.declare_var(elem_var, cl_types::I64);
                builder.def_var(elem_var, zero);
                state.variables.insert(*sym, (elem_var, cl_types::I64));
            }
            PatternKind::Tuple(sub_patterns) => {
                for sub_pat in sub_patterns {
                    if let PatternKind::Ident(sym) = &sub_pat.kind {
                        let var = state.new_variable();
                        builder.declare_var(var, cl_types::I64);
                        builder.def_var(var, zero);
                        state.variables.insert(*sym, (var, cl_types::I64));
                    }
                }
            }
            _ => return Err(CodegenError::new("only ident and tuple patterns in for loops")),
        }

        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let increment_block = builder.create_block();
        let exit_block = builder.create_block();

        builder.ins().jump(header_block, &[]);

        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let counter = builder.use_var(counter_var);
        let cond = builder.ins().icmp(IntCC::SignedLessThan, counter, list_len);
        builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        let get_id = self.runtime_fns["nsl_list_get"].0;
        let get_ref = self.module.declare_func_in_func(get_id, builder.func);
        let counter = builder.use_var(counter_var);
        let call = builder.ins().call(get_ref, &[list_val, counter]);
        let elem = builder.inst_results(call)[0];

        // Bind element to pattern variable(s)
        match &pattern.kind {
            PatternKind::Ident(sym) => {
                let (var, _) = state.variables[sym];
                builder.def_var(var, elem);
            }
            PatternKind::Tuple(sub_patterns) => {
                // elem is a tuple (NslList ptr) — destructure it
                for (i, sub_pat) in sub_patterns.iter().enumerate() {
                    if let PatternKind::Ident(sym) = &sub_pat.kind {
                        let idx = builder.ins().iconst(cl_types::I64, i as i64);
                        let inner_get_ref = self.module.declare_func_in_func(get_id, builder.func);
                        let call = builder.ins().call(inner_get_ref, &[elem, idx]);
                        let sub_elem = builder.inst_results(call)[0];
                        let (var, _) = state.variables[sym];
                        builder.def_var(var, sub_elem);
                    }
                }
            }
            _ => unreachable!(),
        }

        // continue jumps to increment_block (not header) so counter is incremented
        state.loop_stack.push(LoopContext { continue_block: increment_block, exit_block });
        for s in &body.stmts { self.compile_stmt(builder, state, s)?; }
        state.loop_stack.pop();

        if !is_block_filled(builder, body_block) {
            builder.ins().jump(increment_block, &[]);
        }

        // Increment block: counter++ then jump to header
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
        Ok(())
    }

    // ── Match/case ──────────────────────────────────────────────────

    fn compile_match(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        subject: &nsl_ast::expr::Expr,
        arms: &[nsl_ast::expr::MatchArm],
    ) -> Result<(), CodegenError> {
        let subject_val = self.compile_expr(builder, state, subject)?;
        let merge_block = builder.create_block();

        let mut remaining_arms: Vec<_> = arms.iter().collect();
        while !remaining_arms.is_empty() {
            let arm = remaining_arms.remove(0);
            let is_last = remaining_arms.is_empty();

            match &arm.pattern.kind {
                PatternKind::Wildcard => {
                    // Default arm — always taken
                    for s in &arm.body.stmts {
                        self.compile_stmt(builder, state, s)?;
                    }
                    if let Some(block) = state.current_block {
                        if !is_block_filled(builder, block) {
                            builder.ins().jump(merge_block, &[]);
                        }
                    }
                    break;
                }
                PatternKind::Ident(sym) => {
                    // Could be an enum variant or a binding
                    let name = self.resolve_sym(*sym).to_string();
                    if let Some(tag) = self.lookup_enum_variant_tag(&name) {
                        // Enum variant comparison
                        let tag_val = builder.ins().iconst(cl_types::I64, tag);
                        let cmp = builder.ins().icmp(IntCC::Equal, subject_val, tag_val);
                        let arm_block = builder.create_block();
                        let next_block = if is_last { merge_block } else { builder.create_block() };
                        builder.ins().brif(cmp, arm_block, &[], next_block, &[]);

                        builder.switch_to_block(arm_block);
                        builder.seal_block(arm_block);
                        state.current_block = Some(arm_block);
                        for s in &arm.body.stmts { self.compile_stmt(builder, state, s)?; }
                        if !is_block_filled(builder, arm_block) {
                            builder.ins().jump(merge_block, &[]);
                        }

                        if !is_last {
                            builder.switch_to_block(next_block);
                            builder.seal_block(next_block);
                            state.current_block = Some(next_block);
                        }
                    } else {
                        // Binding — bind subject to variable, always taken
                        let var = state.new_variable();
                        builder.declare_var(var, cl_types::I64);
                        builder.def_var(var, subject_val);
                        state.variables.insert(*sym, (var, cl_types::I64));
                        for s in &arm.body.stmts { self.compile_stmt(builder, state, s)?; }
                        if let Some(block) = state.current_block {
                            if !is_block_filled(builder, block) {
                                builder.ins().jump(merge_block, &[]);
                            }
                        }
                        break;
                    }
                }
                PatternKind::Literal(lit_expr) => {
                    let lit_val = self.compile_expr(builder, state, lit_expr)?;
                    let lit_type = self.node_type(lit_expr.id).clone();
                    let cmp = if is_float_type(&lit_type) {
                        builder.ins().fcmp(cranelift_codegen::ir::condcodes::FloatCC::Equal, subject_val, lit_val)
                    } else {
                        builder.ins().icmp(IntCC::Equal, subject_val, lit_val)
                    };
                    let arm_block = builder.create_block();
                    let next_block = if is_last { merge_block } else { builder.create_block() };
                    builder.ins().brif(cmp, arm_block, &[], next_block, &[]);

                    builder.switch_to_block(arm_block);
                    builder.seal_block(arm_block);
                    state.current_block = Some(arm_block);
                    for s in &arm.body.stmts { self.compile_stmt(builder, state, s)?; }
                    if !is_block_filled(builder, arm_block) {
                        builder.ins().jump(merge_block, &[]);
                    }

                    if !is_last {
                        builder.switch_to_block(next_block);
                        builder.seal_block(next_block);
                        state.current_block = Some(next_block);
                    }
                }
                PatternKind::Constructor { path, .. } => {
                    // Enum variant via path: e.g., Activation.ReLU → check tag
                    let variant_name = if path.len() >= 1 {
                        self.resolve_sym(*path.last().unwrap()).to_string()
                    } else {
                        return Err(CodegenError::new("empty constructor path in match"));
                    };
                    if let Some(tag) = self.lookup_enum_variant_tag(&variant_name) {
                        let tag_val = builder.ins().iconst(cl_types::I64, tag);
                        let cmp = builder.ins().icmp(IntCC::Equal, subject_val, tag_val);
                        let arm_block = builder.create_block();
                        let next_block = if is_last { merge_block } else { builder.create_block() };
                        builder.ins().brif(cmp, arm_block, &[], next_block, &[]);

                        builder.switch_to_block(arm_block);
                        builder.seal_block(arm_block);
                        state.current_block = Some(arm_block);
                        for s in &arm.body.stmts { self.compile_stmt(builder, state, s)?; }
                        if !is_block_filled(builder, arm_block) {
                            builder.ins().jump(merge_block, &[]);
                        }

                        if !is_last {
                            builder.switch_to_block(next_block);
                            builder.seal_block(next_block);
                            state.current_block = Some(next_block);
                        }
                    } else {
                        return Err(CodegenError::new(format!(
                            "unknown enum variant '{variant_name}' in match"
                        )));
                    }
                }
                _ => return Err(CodegenError::new("unsupported pattern in match arm")),
            }
        }

        // If we didn't break (no wildcard/binding), need to jump to merge from final else
        if let Some(block) = state.current_block {
            if block != merge_block && !is_block_filled(builder, block) {
                builder.ins().jump(merge_block, &[]);
            }
        }

        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);
        state.current_block = Some(merge_block);
        Ok(())
    }

    fn compile_train_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        train: &nsl_ast::block::TrainBlock,
    ) -> Result<(), CodegenError> {
        // ── 1. Extract config from train(...) args ──────────────────────
        let mut model_sym: Option<nsl_ast::Symbol> = None;
        let mut epochs: i64 = 1;

        for arg in &train.config {
            if let Some(name_sym) = arg.name {
                let name = self.resolve_sym(name_sym).to_string();
                match name.as_str() {
                    "model" => {
                        if let ExprKind::Ident(sym) = &arg.value.kind {
                            model_sym = Some(*sym);
                        } else {
                            return Err(CodegenError::new("train 'model' arg must be an identifier"));
                        }
                    }
                    "epochs" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            epochs = *n;
                        } else {
                            return Err(CodegenError::new("train 'epochs' arg must be an integer literal"));
                        }
                    }
                    _ => {} // ignore unknown config for forward compat
                }
            }
        }

        let model_sym = model_sym.ok_or_else(|| {
            CodegenError::new("train block requires 'model=<ident>' config argument")
        })?;

        // ── 2. Extract optimizer info from sections ─────────────────────
        let mut optimizer_name = String::new();
        let mut lr_value: f64 = 0.01;
        let mut momentum_value: f64 = 0.0;
        let mut dampening_value: f64 = 0.0;
        let mut weight_decay_value: f64 = 0.0;
        let mut nesterov_value: bool = false;
        let mut beta1_value: f64 = 0.9;
        let mut beta2_value: f64 = 0.999;
        let mut eps_value: f64 = 1e-8;
        let mut step_body: Option<&nsl_ast::stmt::Block> = None;
        let mut callbacks: Vec<&nsl_ast::block::CallbackDef> = Vec::new();
        let mut scheduler_name = String::new();
        let mut scheduler_args: Vec<(String, f64)> = Vec::new();

        for section in &train.sections {
            match section {
                TrainSection::Optimizer(expr) => {
                    // Parse call like SGD(lr=0.01, momentum=0.9)
                    if let ExprKind::Call { callee, args } = &expr.kind {
                        if let ExprKind::Ident(sym) = &callee.kind {
                            optimizer_name = self.resolve_sym(*sym).to_string().to_lowercase();
                        }
                        for arg in args {
                            if let Some(name_sym) = arg.name {
                                let name = self.resolve_sym(name_sym).to_string();
                                match name.as_str() {
                                    "lr" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            lr_value = *f;
                                        } else if let ExprKind::IntLiteral(n) = &arg.value.kind {
                                            lr_value = *n as f64;
                                        }
                                    }
                                    "momentum" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            momentum_value = *f;
                                        }
                                    }
                                    "dampening" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            dampening_value = *f;
                                        }
                                    }
                                    "weight_decay" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            weight_decay_value = *f;
                                        }
                                    }
                                    "nesterov" => {
                                        if let ExprKind::BoolLiteral(b) = &arg.value.kind {
                                            nesterov_value = *b;
                                        }
                                    }
                                    "beta1" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            beta1_value = *f;
                                        }
                                    }
                                    "beta2" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            beta2_value = *f;
                                        }
                                    }
                                    "eps" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            eps_value = *f;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                TrainSection::Step { body, .. } => {
                    step_body = Some(body);
                }
                TrainSection::Callbacks(cbs) => {
                    callbacks.extend(cbs.iter());
                }
                TrainSection::Scheduler(expr) => {
                    if let ExprKind::Call { callee, args } = &expr.kind {
                        if let ExprKind::Ident(sym) = &callee.kind {
                            scheduler_name = self.resolve_sym(*sym).to_string();
                        }
                        for arg in args {
                            if let Some(name_sym) = arg.name {
                                let name = self.resolve_sym(name_sym).to_string();
                                if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                    scheduler_args.push((name, *f));
                                } else if let ExprKind::IntLiteral(n) = &arg.value.kind {
                                    scheduler_args.push((name, *n as f64));
                                }
                            }
                        }
                    }
                }
                _ => {} // Data, Eval, Distribute, Stmt — not yet implemented
            }
        }

        if optimizer_name.is_empty() {
            return Err(CodegenError::new("train block requires an optimizer section"));
        }

        let step_body = step_body.ok_or_else(|| {
            CodegenError::new("train block requires a step section")
        })?;

        // ── 3. Resolve model type and build param list ──────────────────
        // Get the model pointer from state
        let (model_var, _) = *state.variables.get(&model_sym).ok_or_else(|| {
            CodegenError::new(format!(
                "undefined model variable '{}' in train block",
                self.resolve_sym(model_sym)
            ))
        })?;
        let model_ptr = builder.use_var(model_var);

        // Resolve model type name — search type_map for any entry with Model/Struct type
        // that matches what we know about this variable, or fall back to the variable name
        let model_var_name = self.resolve_sym(model_sym).to_string();
        let model_type_name = {
            // Try to find the model type by checking all type_map entries
            let mut found_name = None;
            for (_node_id, ty) in self.type_map.iter() {
                match ty {
                    nsl_semantic::types::Type::Model { name, .. } => {
                        let n = self.resolve_sym(*name).to_string();
                        if self.struct_layouts.contains_key(&n) {
                            found_name = Some(n);
                            break;
                        }
                    }
                    nsl_semantic::types::Type::Struct { name, .. } => {
                        let n = self.resolve_sym(*name).to_string();
                        if self.struct_layouts.contains_key(&n) {
                            // Only use if it looks like a model (has tensor fields)
                            found_name = Some(n);
                        }
                    }
                    _ => {}
                }
            }
            // Fallback: try variable name directly, or capitalize it
            found_name.unwrap_or_else(|| {
                // Try capitalized form (common: variable 'm' → type 'M' not useful)
                // Just use the variable name and hope struct_layouts has it
                model_var_name.clone()
            })
        };

        let layout = self.struct_layouts.get(&model_type_name).cloned().ok_or_else(|| {
            CodegenError::new(format!(
                "no struct layout found for model '{}' in train block",
                model_type_name
            ))
        })?;

        let num_params = layout.fields.len();

        // Build param_list: NslList of tensor pointers (model fields)
        let param_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        for field in &layout.fields {
            let field_val = builder.ins().load(
                field.cl_type,
                MemFlags::trusted(),
                model_ptr,
                field.offset as i32,
            );
            self.compile_call_by_name(builder, "nsl_list_push", &[param_list, field_val])?;
        }

        // ── 4. Create optimizer state buffers ─────────────────────────
        // Number of state buffers per param depends on optimizer:
        //   SGD/Lion/Muon: 1 (velocity/momentum)
        //   Adam/AdamW/SOAP: 2 (first moment m, second moment v)
        let num_state_buffers = match optimizer_name.as_str() {
            "adam" | "adamw" | "soap" => 2,
            _ => 1,
        };

        let mut state_buf_1: Vec<cranelift_codegen::ir::Value> = Vec::new();
        let mut state_buf_2: Vec<cranelift_codegen::ir::Value> = Vec::new();
        for field in &layout.fields {
            let field_val = builder.ins().load(
                field.cl_type,
                MemFlags::trusted(),
                model_ptr,
                field.offset as i32,
            );
            let buf1 = self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[field_val])?;
            state_buf_1.push(buf1);
            if num_state_buffers >= 2 {
                let buf2 = self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[field_val])?;
                state_buf_2.push(buf2);
            }
        }

        // ── 5. Initialize lr and step_count variables ───────────────────
        let lr_var = state.new_variable();
        builder.declare_var(lr_var, cl_types::F64);
        let lr_const = builder.ins().f64const(lr_value);
        builder.def_var(lr_var, lr_const);

        let step_count_var = state.new_variable();
        builder.declare_var(step_count_var, cl_types::I64);
        let zero_i64 = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_count_var, zero_i64);

        // ── 6. Emit epoch loop ──────────────────────────────────────────
        let epoch_counter_var = state.new_variable();
        builder.declare_var(epoch_counter_var, cl_types::I64);
        let zero = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(epoch_counter_var, zero);

        let epochs_val = builder.ins().iconst(cl_types::I64, epochs);

        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let increment_block = builder.create_block();
        let exit_block = builder.create_block();

        builder.ins().jump(header_block, &[]);

        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let counter = builder.use_var(epoch_counter_var);
        let cond = builder.ins().icmp(IntCC::SignedLessThan, counter, epochs_val);
        builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        // ── 7. Inside body: tape_start → step body → find loss → backward → optimizer → callbacks ──

        // 7a. Set training mode = true, then start tape recording
        let true_val = builder.ins().iconst(cl_types::I8, 1);
        self.compile_call_by_name(builder, "nsl_set_training_mode", &[true_val])?;
        self.compile_call_by_name(builder, "nsl_tape_start", &[param_list])?;

        // 7b. Compile step body stmts
        for stmt in &step_body.stmts {
            self.compile_stmt(builder, state, stmt)?;
        }

        // 7c. Find loss variable — look for "loss" in state.variables by name
        let loss_val = {
            let mut found = None;
            for (sym, (var, _)) in &state.variables {
                if self.resolve_sym(*sym) == "loss" {
                    found = Some(builder.use_var(*var));
                    break;
                }
            }
            found.ok_or_else(|| {
                CodegenError::new("train step body must assign to a variable named 'loss'")
            })?
        };

        // 7d. Run backward pass
        let grads_list = self.compile_call_by_name(
            builder,
            "nsl_tape_backward",
            &[loss_val, param_list],
        )?;

        // 7e. Stop tape and restore eval mode
        self.compile_call_by_name(builder, "nsl_tape_stop", &[])?;
        let false_val = builder.ins().iconst(cl_types::I8, 0);
        self.compile_call_by_name(builder, "nsl_set_training_mode", &[false_val])?;

        // 7f. Optimizer step: for each param, call optimizer step function
        let optimizer_fn_name = match optimizer_name.as_str() {
            "sgd" => "nsl__optim__sgd__sgd_step",
            "adam" => "nsl__optim__adam__adam_step",
            "adamw" => "nsl__optim__adamw__adamw_step",
            "lion" => "nsl__optim__lion__lion_step",
            "muon" => "nsl__optim__muon__muon_step",
            "soap" => "nsl__optim__soap__soap_step",
            _ => {
                return Err(CodegenError::new(format!(
                    "unsupported optimizer '{}' in train block", optimizer_name
                )));
            }
        };

        // Check if optimizer function exists, try fallback name patterns
        let opt_fn = if self.functions.contains_key(optimizer_fn_name) {
            optimizer_fn_name.to_string()
        } else {
            // Try simpler name: e.g. "sgd_step"
            let simple = format!("{}_step", optimizer_name);
            if self.functions.contains_key(&simple) {
                simple
            } else if self.runtime_fns.contains_key(optimizer_fn_name) {
                optimizer_fn_name.to_string()
            } else if self.runtime_fns.contains_key(&simple) {
                simple
            } else {
                // Register as runtime function so it can be resolved at link time
                optimizer_fn_name.to_string()
            }
        };

        let lr = builder.use_var(lr_var);
        let momentum_const = builder.ins().f64const(momentum_value);
        let dampening_const = builder.ins().f64const(dampening_value);
        let weight_decay_const = builder.ins().f64const(weight_decay_value);
        let nesterov_const = builder.ins().iconst(cl_types::I8, if nesterov_value { 1 } else { 0 });
        let beta1_const = builder.ins().f64const(beta1_value);
        let beta2_const = builder.ins().f64const(beta2_value);
        let eps_const = builder.ins().f64const(eps_value);

        for i in 0..num_params {
            let idx = builder.ins().iconst(cl_types::I64, i as i64);

            // Get param tensor from param_list
            let param_val = self.compile_call_by_name(
                builder, "nsl_list_get", &[param_list, idx],
            )?;

            // Get gradient from grads_list
            let grad_val = self.compile_call_by_name(
                builder, "nsl_list_get", &[grads_list, idx],
            )?;

            match optimizer_name.as_str() {
                "sgd" => {
                    self.compile_call_by_name(
                        builder, &opt_fn,
                        &[param_val, grad_val, state_buf_1[i], lr, momentum_const, dampening_const, weight_decay_const, nesterov_const],
                    )?;
                }
                "adam" | "adamw" => {
                    // Adam/AdamW need t = step_count + 1 (1-based) as float
                    let t_val = builder.use_var(step_count_var);
                    let one = builder.ins().iconst(cl_types::I64, 1);
                    let t_plus_one = builder.ins().iadd(t_val, one);
                    let t_float = builder.ins().fcvt_from_sint(cl_types::F64, t_plus_one);
                    self.compile_call_by_name(
                        builder, &opt_fn,
                        &[param_val, grad_val, state_buf_1[i], state_buf_2[i], lr, beta1_const, beta2_const, eps_const, weight_decay_const, t_float],
                    )?;
                }
                "lion" => {
                    self.compile_call_by_name(
                        builder, &opt_fn,
                        &[param_val, grad_val, state_buf_1[i], lr, beta1_const, beta2_const, weight_decay_const],
                    )?;
                }
                "muon" => {
                    self.compile_call_by_name(
                        builder, &opt_fn,
                        &[param_val, grad_val, state_buf_1[i], lr, momentum_const, nesterov_const],
                    )?;
                }
                "soap" => {
                    self.compile_call_by_name(
                        builder, &opt_fn,
                        &[param_val, grad_val, state_buf_1[i], state_buf_2[i], lr, beta1_const, beta2_const, eps_const],
                    )?;
                }
                _ => {
                    return Err(CodegenError::new(format!(
                        "unsupported optimizer '{}' in train block", optimizer_name
                    )));
                }
            }
        }

        // 7g. Increment step count
        let sc = builder.use_var(step_count_var);
        let one_i64 = builder.ins().iconst(cl_types::I64, 1);
        let sc_next = builder.ins().iadd(sc, one_i64);
        builder.def_var(step_count_var, sc_next);

        // 7g2. Scheduler: update learning rate if scheduler is configured
        if !scheduler_name.is_empty() {
            let sched_fn_name = match scheduler_name.to_lowercase().as_str() {
                "constant_lr" | "constantlr" => "constant_lr",
                "step_lr" | "steplr" => "step_lr",
                "exponential_lr" | "exponentiallr" => "exponential_lr",
                "linear_decay" | "lineardecay" => "linear_decay",
                "cosine_anneal" | "cosineanneal" => "cosine_anneal",
                "warmup_cosine" | "warmupcosine" => "warmup_cosine",
                "one_cycle" | "onecycle" => "one_cycle",
                _ => &scheduler_name,
            };
            let mangled = format!("nsl__optim__schedulers__{}", sched_fn_name);

            // Find the actual function name (check functions/runtime_fns with fallback)
            let sched_fn = if self.functions.contains_key(mangled.as_str()) {
                mangled.clone()
            } else {
                let simple = sched_fn_name.to_string();
                if self.functions.contains_key(simple.as_str()) {
                    simple
                } else if self.runtime_fns.contains_key(mangled.as_str()) {
                    mangled.clone()
                } else if self.runtime_fns.contains_key(simple.as_str()) {
                    simple
                } else {
                    mangled.clone()
                }
            };

            let base_lr_val = builder.ins().f64const(lr_value);
            let step_count_val = builder.use_var(step_count_var);
            let step_float = builder.ins().fcvt_from_sint(cl_types::F64, step_count_val);

            let new_lr = match sched_fn_name {
                "constant_lr" => {
                    self.compile_call_by_name(builder, &sched_fn, &[base_lr_val, step_float])?
                }
                "step_lr" => {
                    let step_size = scheduler_args.iter().find(|(n, _)| n == "step_size").map(|(_, v)| *v).unwrap_or(10.0);
                    let gamma = scheduler_args.iter().find(|(n, _)| n == "gamma").map(|(_, v)| *v).unwrap_or(0.1);
                    let ss_val = builder.ins().f64const(step_size);
                    let g_val = builder.ins().f64const(gamma);
                    self.compile_call_by_name(builder, &sched_fn, &[base_lr_val, step_float, ss_val, g_val])?
                }
                "exponential_lr" => {
                    let gamma = scheduler_args.iter().find(|(n, _)| n == "gamma").map(|(_, v)| *v).unwrap_or(0.95);
                    let g_val = builder.ins().f64const(gamma);
                    self.compile_call_by_name(builder, &sched_fn, &[base_lr_val, step_float, g_val])?
                }
                "linear_decay" => {
                    let total_steps = scheduler_args.iter().find(|(n, _)| n == "total_steps").map(|(_, v)| *v).unwrap_or(1000.0);
                    let end_factor = scheduler_args.iter().find(|(n, _)| n == "end_factor").map(|(_, v)| *v).unwrap_or(0.0);
                    let ts_val = builder.ins().f64const(total_steps);
                    let ef_val = builder.ins().f64const(end_factor);
                    self.compile_call_by_name(builder, &sched_fn, &[base_lr_val, step_float, ts_val, ef_val])?
                }
                "cosine_anneal" => {
                    let t_max = scheduler_args.iter().find(|(n, _)| n == "t_max").map(|(_, v)| *v).unwrap_or(1000.0);
                    let eta_min = scheduler_args.iter().find(|(n, _)| n == "eta_min").map(|(_, v)| *v).unwrap_or(0.0);
                    let tm_val = builder.ins().f64const(t_max);
                    let em_val = builder.ins().f64const(eta_min);
                    self.compile_call_by_name(builder, &sched_fn, &[base_lr_val, step_float, tm_val, em_val])?
                }
                "warmup_cosine" => {
                    let ws = scheduler_args.iter().find(|(n, _)| n == "warmup_steps").map(|(_, v)| *v).unwrap_or(100.0);
                    let ts = scheduler_args.iter().find(|(n, _)| n == "total_steps").map(|(_, v)| *v).unwrap_or(1000.0);
                    let ml = scheduler_args.iter().find(|(n, _)| n == "min_lr").map(|(_, v)| *v).unwrap_or(1e-5);
                    let ws_val = builder.ins().f64const(ws);
                    let ts_val = builder.ins().f64const(ts);
                    let ml_val = builder.ins().f64const(ml);
                    self.compile_call_by_name(builder, &sched_fn, &[base_lr_val, step_float, ws_val, ts_val, ml_val])?
                }
                "one_cycle" => {
                    let max_lr = scheduler_args.iter().find(|(n, _)| n == "max_lr").map(|(_, v)| *v).unwrap_or(lr_value * 10.0);
                    let total_steps = scheduler_args.iter().find(|(n, _)| n == "total_steps").map(|(_, v)| *v).unwrap_or(1000.0);
                    let pct_start = scheduler_args.iter().find(|(n, _)| n == "pct_start").map(|(_, v)| *v).unwrap_or(0.3);
                    let ml_val = builder.ins().f64const(max_lr);
                    let ts_val = builder.ins().f64const(total_steps);
                    let ps_val = builder.ins().f64const(pct_start);
                    self.compile_call_by_name(builder, &sched_fn, &[base_lr_val, step_float, ml_val, ts_val, ps_val])?
                }
                _ => base_lr_val, // fallback: no change
            };

            builder.def_var(lr_var, new_lr);
        }

        // 7h. Callbacks: compile on_step body with step_count and loss bound
        for cb in &callbacks {
            let cb_name = self.resolve_sym(cb.name).to_string();
            if cb_name == "on_step" {
                // Bind callback params: on_step(step, loss)
                for param in &cb.params {
                    let pname = self.resolve_sym(param.name).to_string();
                    match pname.as_str() {
                        "step" => {
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            let step_val = builder.use_var(step_count_var);
                            builder.def_var(var, step_val);
                            state.variables.insert(param.name, (var, cl_types::I64));
                        }
                        "loss" => {
                            // loss is already in state.variables, but rebind for callback scope
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            builder.def_var(var, loss_val);
                            state.variables.insert(param.name, (var, cl_types::I64));
                        }
                        _ => {
                            // Unknown callback param — bind to zero
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            let z = builder.ins().iconst(cl_types::I64, 0);
                            builder.def_var(var, z);
                            state.variables.insert(param.name, (var, cl_types::I64));
                        }
                    }
                }
                // Compile callback body
                for stmt in &cb.body.stmts {
                    self.compile_stmt(builder, state, stmt)?;
                }
            } else if cb_name == "on_epoch" || cb_name == "on_epoch_end" {
                // Bind epoch counter and loss
                for param in &cb.params {
                    let pname = self.resolve_sym(param.name).to_string();
                    match pname.as_str() {
                        "epoch" => {
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            let epoch_val = builder.use_var(epoch_counter_var);
                            builder.def_var(var, epoch_val);
                            state.variables.insert(param.name, (var, cl_types::I64));
                        }
                        "loss" => {
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            builder.def_var(var, loss_val);
                            state.variables.insert(param.name, (var, cl_types::I64));
                        }
                        _ => {
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            let z = builder.ins().iconst(cl_types::I64, 0);
                            builder.def_var(var, z);
                            state.variables.insert(param.name, (var, cl_types::I64));
                        }
                    }
                }
                for stmt in &cb.body.stmts {
                    self.compile_stmt(builder, state, stmt)?;
                }
            }
        }

        // ── 8. Increment epoch counter and loop back ────────────────────
        if !is_block_filled(builder, body_block) {
            builder.ins().jump(increment_block, &[]);
        }

        builder.switch_to_block(increment_block);
        builder.seal_block(increment_block);
        state.current_block = Some(increment_block);
        let counter = builder.use_var(epoch_counter_var);
        let one = builder.ins().iconst(cl_types::I64, 1);
        let next = builder.ins().iadd(counter, one);
        builder.def_var(epoch_counter_var, next);
        builder.ins().jump(header_block, &[]);

        builder.seal_block(header_block);
        builder.switch_to_block(exit_block);
        builder.seal_block(exit_block);
        state.current_block = Some(exit_block);

        Ok(())
    }

    fn compile_grad_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        grad: &nsl_ast::block::GradBlock,
    ) -> Result<(), CodegenError> {
        // 1. Compile targets expression to get param tensor ptr
        let targets_val = self.compile_expr(builder, state, &grad.targets)?;

        // 2. Wrap single tensor in a 1-element list for the tape API
        let param_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        self.compile_call_by_name(builder, "nsl_list_push", &[param_list, targets_val])?;

        // 3. Start tape recording
        self.compile_call_by_name(builder, "nsl_tape_start", &[param_list])?;

        // 4. Compile body — all tensor ops auto-record on the global tape.
        //    The last expression is the loss (a scalar tensor).
        let mut loss_val = None;
        for (i, stmt) in grad.body.stmts.iter().enumerate() {
            if i == grad.body.stmts.len() - 1 {
                if let StmtKind::Expr(ref expr) = stmt.kind {
                    loss_val = Some(self.compile_expr(builder, state, expr)?);
                } else {
                    self.compile_stmt(builder, state, stmt)?;
                }
            } else {
                self.compile_stmt(builder, state, stmt)?;
            }
        }

        let loss_tensor = loss_val.ok_or_else(|| {
            CodegenError::new("grad block must end with an expression (the loss)")
        })?;

        // 5. Run backward pass
        let grads_list = self.compile_call_by_name(
            builder,
            "nsl_tape_backward",
            &[loss_tensor, param_list],
        )?;

        // 6. Stop tape (cleans up saved tensor refcounts)
        self.compile_call_by_name(builder, "nsl_tape_stop", &[])?;

        // 7. Get gradient for the single param (index 0)
        let zero = builder.ins().iconst(cl_types::I64, 0);
        let grad_tensor = self.compile_call_by_name(
            builder,
            "nsl_list_get",
            &[grads_list, zero],
        )?;

        // 8. Bind output variables if pattern exists
        //    loss is bound as scalar tensor ptr (I64) — use .item() for f64
        //    grads is bound as gradient tensor ptr (I64)
        if let Some(ref pattern) = grad.outputs {
            match &pattern.kind {
                PatternKind::Tuple(pats) if pats.len() == 2 => {
                    // Bind loss (scalar tensor ptr)
                    if let PatternKind::Ident(loss_sym) = &pats[0].kind {
                        let var = state.new_variable();
                        builder.declare_var(var, cl_types::I64);
                        builder.def_var(var, loss_tensor);
                        state.variables.insert(*loss_sym, (var, cl_types::I64));
                    }
                    // Bind grads (tensor ptr)
                    if let PatternKind::Ident(grads_sym) = &pats[1].kind {
                        let var = state.new_variable();
                        builder.declare_var(var, cl_types::I64);
                        builder.def_var(var, grad_tensor);
                        state.variables.insert(*grads_sym, (var, cl_types::I64));
                    }
                }
                _ => {
                    return Err(CodegenError::new(
                        "grad block output must be `let (loss, grads) = grad(...):`"
                    ));
                }
            }
        }

        Ok(())
    }

    // ── Quant block codegen ──────────────────────────────────────────

    pub fn compile_quant_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        quant: &nsl_ast::block::QuantBlock,
    ) -> Result<(), CodegenError> {
        // 1. Get source model variable
        let source_sym = quant.source;
        let source_val = {
            let (var, _) = state.variables.get(&source_sym).ok_or_else(|| {
                CodegenError::new(format!(
                    "undefined model variable '{}' in quant block",
                    self.resolve_sym(source_sym)
                ))
            })?;
            builder.use_var(*var)
        };

        // 2. Resolve model type name using the same strategy as train blocks:
        //    scan the type_map for a Model type with a known struct layout.
        let model_type_name = {
            let mut found_name = None;
            for (_node_id, ty) in self.type_map.iter() {
                match ty {
                    nsl_semantic::types::Type::Model { name, .. } => {
                        let n = self.resolve_sym(*name).to_string();
                        if self.struct_layouts.contains_key(&n) {
                            found_name = Some(n);
                            break;
                        }
                    }
                    nsl_semantic::types::Type::Struct { name, .. } => {
                        let n = self.resolve_sym(*name).to_string();
                        if self.struct_layouts.contains_key(&n) {
                            found_name = Some(n);
                        }
                    }
                    _ => {}
                }
            }
            found_name.unwrap_or_else(|| self.resolve_sym(source_sym).to_string())
        };

        let layout = self.struct_layouts.get(&model_type_name).cloned().ok_or_else(|| {
            CodegenError::new(format!(
                "no struct layout found for model '{}' in quant block",
                model_type_name
            ))
        })?;

        // 3. Compute dtype/granularity integer codes for the runtime call
        let dtype_code: i64 = match quant.default_dtype {
            Some(QuantDtype::Int4) => 1,
            Some(QuantDtype::Int8) | None => 0,
        };
        let (gran_code, axis_val, gs_val): (i64, i64, i64) = match &quant.default_granularity {
            Some(QuantGranularity::PerChannel(a)) => (1, *a, 0),
            Some(QuantGranularity::PerGroup(a, gs)) => (2, *a, *gs),
            Some(QuantGranularity::PerTensor) | None => (0, 0, 0),
        };

        let dtype_v = builder.ins().iconst(cl_types::I64, dtype_code);
        let gran_v = builder.ins().iconst(cl_types::I64, gran_code);
        let axis_v = builder.ins().iconst(cl_types::I64, axis_val);
        let gs_v = builder.ins().iconst(cl_types::I64, gs_val);

        // 4. Allocate a new struct with the same layout as the source model
        let alloc_size = builder.ins().iconst(cl_types::I64, layout.total_size.max(8) as i64);
        let new_ptr = self.compile_call_by_name(builder, "nsl_alloc", &[alloc_size])?;

        // 5. For each field: quantize→dequantize (or clone if excluded)
        for field in &layout.fields {
            let is_excluded = quant.exclude.iter().any(|pat| glob_match(pat, &field.name));
            let src_val = builder.ins().load(
                field.cl_type,
                MemFlags::trusted(),
                source_val,
                field.offset as i32,
            );

            if is_excluded {
                // Copy as-is via clone (bumps refcount internally)
                let cloned = self.compile_call_by_name(builder, "nsl_tensor_clone", &[src_val])?;
                builder.ins().store(MemFlags::trusted(), cloned, new_ptr, field.offset as i32);
            } else {
                // Quantize then immediately dequantize — validates the roundtrip and
                // shows quantization effects (precision loss) while storing a regular
                // NslTensor that the original forward method can consume directly.
                let qt = self.compile_call_by_name(
                    builder,
                    "nsl_qtensor_quantize",
                    &[src_val, dtype_v, gran_v, axis_v, gs_v],
                )?;
                let deq = self.compile_call_by_name(builder, "nsl_qtensor_dequantize", &[qt])?;
                // Release the intermediate QuantizedTensor (refcount-aware)
                self.compile_call_by_name(builder, "nsl_qtensor_release", &[qt])?;
                builder.ins().store(MemFlags::trusted(), deq, new_ptr, field.offset as i32);
            }
        }

        // 6. Register the quantized model with the same struct layout and methods
        //    so that forward dispatch works identically to the source model.
        let quant_name = self.resolve_sym(quant.name).to_string();
        if !self.struct_layouts.contains_key(&quant_name) {
            self.struct_layouts.insert(quant_name.clone(), layout);
        }
        if let Some(methods) = self.model_methods.get(&model_type_name).cloned() {
            self.model_methods.insert(quant_name, methods);
        }

        // 7. Bind the new struct pointer as the output variable
        let var = state.new_variable();
        builder.declare_var(var, cl_types::I64);
        builder.def_var(var, new_ptr);
        state.variables.insert(quant.name, (var, cl_types::I64));

        Ok(())
    }
}

/// Simple glob matching supporting `*` (any sequence) and `?` (single char) wildcards.
fn glob_match(pattern: &str, text: &str) -> bool {
    let pb = pattern.as_bytes();
    let tb = text.as_bytes();
    let mut pi = 0usize;
    let mut ti = 0usize;
    let mut star_pi = usize::MAX;
    let mut star_ti = 0usize;

    while ti < tb.len() {
        if pi < pb.len() && (pb[pi] == b'?' || pb[pi] == tb[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pb.len() && pb[pi] == b'*' {
            star_pi = pi;
            star_ti = ti;
            pi += 1;
        } else if star_pi != usize::MAX {
            pi = star_pi + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }
    while pi < pb.len() && pb[pi] == b'*' {
        pi += 1;
    }
    pi == pb.len()
}
