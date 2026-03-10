use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::InstBuilder;
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;

use nsl_ast::operator::AssignOp;
use nsl_ast::pattern::PatternKind;
use nsl_ast::expr::SubscriptKind;
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

            StmtKind::StructDef(_) | StmtKind::ModelDef(_)
            | StmtKind::EnumDef(_) | StmtKind::TraitDef(_)
            | StmtKind::Import(_) | StmtKind::FromImport(_)
            | StmtKind::DatasetDef(_) | StmtKind::TokenizerDef(_)
            | StmtKind::TrainBlock(_) | StmtKind::QuantBlock(_)
            | StmtKind::KernelDef(_) => {}

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
}
