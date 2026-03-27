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
use nsl_semantic::types::Type;

use cranelift_codegen::ir::Value;
use crate::compiler::Compiler;
use crate::context::{FuncState, LoopContext};
use crate::error::CodegenError;
use crate::types::{is_block_filled, is_float_type, nsl_type_to_cl};

impl Compiler<'_> {
    /// M36: Try to compile a tensor creation as a slab-managed allocation.
    /// Returns Ok(Some(value)) if the variable is slab-planned and the RHS is a
    /// tensor creation (zeros, ones, etc.). Returns Ok(None) to fall through to normal codegen.
    fn try_compile_slab_tensor(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        sym: &nsl_ast::Symbol,
        expr: &nsl_ast::expr::Expr,
    ) -> Result<Option<Value>, CodegenError> {
        // Check if slab is active and this variable is planned
        let slab_var = match state.slab_ptr_var {
            Some(v) => v,
            None => return Ok(None),
        };
        let var_name = match self.interner.resolve(sym.0) {
            Some(n) => n.to_string(),
            None => return Ok(None),
        };
        let offset = match self.slab_name_offsets.get(&var_name) {
            Some(&o) => o,
            None => return Ok(None),
        };

        // Check if the RHS is a tensor creation call (zeros, ones, rand, zeros_on)
        let is_tensor_creation = match &expr.kind {
            ExprKind::Call { callee, .. } => {
                match &callee.kind {
                    ExprKind::Ident(func_sym) => {
                        let func_name = self.interner.resolve(func_sym.0).unwrap_or("");
                        matches!(func_name, "zeros" | "ones" | "rand" | "randn" | "zeros_like")
                    }
                    _ => false,
                }
            }
            // zeros_on is typically a method call: Tensor.zeros_on(shape, device)
            _ => false,
        };

        if !is_tensor_creation {
            return Ok(None);
        }

        // Extract the shape argument from the call
        let shape_val = if let ExprKind::Call { args, .. } = &expr.kind {
            if args.is_empty() {
                return Ok(None);
            }
            self.compile_expr(builder, state, &args[0].value)?
        } else {
            return Ok(None);
        };

        // Compute data pointer: slab_base + offset
        let slab_ptr = builder.use_var(slab_var);
        let offset_val = builder.ins().iconst(cl_types::I64, offset as i64);
        let data_ptr = self.compile_call_by_name(builder, "nsl_slab_offset", &[slab_ptr, offset_val])?;

        // Determine device and dtype from the expression type
        let (device, dtype) = if let Some(ty) = self.type_map.get(&expr.id) {
            if let Some((_shape, dt, dev)) = ty.as_tensor_parts() {
                let dev_val = match dev {
                    nsl_semantic::types::Device::Cuda(_) => 1i64,
                    nsl_semantic::types::Device::Cpu => 0i64,
                    _ => 0i64,
                };
                let dt_val = match dt {
                    nsl_semantic::types::DType::F32 => 1i64,
                    nsl_semantic::types::DType::F64 => 0i64,
                    _ => 1i64, // default GPU dtype
                };
                (dev_val, dt_val)
            } else {
                (0, 1) // fallback
            }
        } else {
            (0, 1)
        };

        let device_val = builder.ins().iconst(cl_types::I64, device);
        let dtype_val = builder.ins().iconst(cl_types::I64, dtype);

        let tensor = self.compile_call_by_name(
            builder, "nsl_tensor_from_slab",
            &[data_ptr, shape_val, device_val, dtype_val],
        )?;

        Ok(Some(tensor))
    }

    /// Recursively destructure patterns from a list/tuple value.
    /// Each `PatternKind::Ident` binds a variable, `Wildcard` is skipped,
    /// `Tuple`/`List` recurse into nested `nsl_list_get` calls, and
    /// `Struct` destructures by field name via `nsl_dict_get`.
    fn compile_destructure_patterns(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        patterns: &[nsl_ast::pattern::Pattern],
        container_val: cranelift_codegen::ir::Value,
    ) -> Result<(), CodegenError> {
        let get_id = self.runtime_fns["nsl_list_get"].0;
        let get_ref = self.module.declare_func_in_func(get_id, builder.func);

        for (i, sub_pat) in patterns.iter().enumerate() {
            match &sub_pat.kind {
                PatternKind::Ident(sym) => {
                    let idx = builder.ins().iconst(cl_types::I64, i as i64);
                    let call = builder.ins().call(get_ref, &[container_val, idx]);
                    let elem = builder.inst_results(call)[0];
                    let var = state.new_variable();
                    builder.declare_var(var, cl_types::I64);
                    builder.def_var(var, elem);
                    state.variables.insert(*sym, (var, cl_types::I64));
                }
                PatternKind::Wildcard => {}
                PatternKind::Tuple(nested) | PatternKind::List(nested) => {
                    // Extract the i-th element, then recurse into it
                    let idx = builder.ins().iconst(cl_types::I64, i as i64);
                    let call = builder.ins().call(get_ref, &[container_val, idx]);
                    let nested_val = builder.inst_results(call)[0];
                    self.compile_destructure_patterns(builder, state, nested, nested_val)?;
                }
                PatternKind::Struct { fields, .. } => {
                    // Extract the i-th element (the struct/dict), then destructure fields
                    let idx = builder.ins().iconst(cl_types::I64, i as i64);
                    let call = builder.ins().call(get_ref, &[container_val, idx]);
                    let struct_val = builder.inst_results(call)[0];
                    for field in fields {
                        let field_name = self.resolve_sym(field.name).to_string();
                        // Ensure string is in pool, then get pointer for dict lookup
                        if !self.string_pool.contains_key(field_name.as_str()) {
                            self.intern_string(&field_name)?;
                        }
                        let key_str = self.compile_string_literal(builder, &field_name)?;
                        let field_val = self.compile_call_by_name(
                            builder, "nsl_dict_get_str", &[struct_val, key_str],
                        )?;
                        if let Some(ref pat) = field.pattern {
                            // Nested pattern: { x: (a, b) } → destructure the field value
                            match &pat.kind {
                                PatternKind::Ident(sym) => {
                                    let var = state.new_variable();
                                    builder.declare_var(var, cl_types::I64);
                                    builder.def_var(var, field_val);
                                    state.variables.insert(*sym, (var, cl_types::I64));
                                }
                                PatternKind::Tuple(nested) | PatternKind::List(nested) => {
                                    self.compile_destructure_patterns(builder, state, nested, field_val)?;
                                }
                                PatternKind::Wildcard => {}
                                _ => {
                                    return Err(CodegenError::new(format!(
                                        "unsupported nested pattern in struct field '{}'", field_name
                                    )));
                                }
                            }
                        } else {
                            // Simple field binding: { name } binds `name` to the value
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            builder.def_var(var, field_val);
                            state.variables.insert(field.name, (var, cl_types::I64));
                        }
                    }
                }
                PatternKind::Typed { pattern, .. } => {
                    // Type annotation is semantic-only — recurse into the inner pattern
                    let idx = builder.ins().iconst(cl_types::I64, i as i64);
                    let call = builder.ins().call(get_ref, &[container_val, idx]);
                    let elem = builder.inst_results(call)[0];
                    match &pattern.kind {
                        PatternKind::Ident(sym) => {
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            builder.def_var(var, elem);
                            state.variables.insert(*sym, (var, cl_types::I64));
                        }
                        PatternKind::Tuple(nested) | PatternKind::List(nested) => {
                            self.compile_destructure_patterns(builder, state, nested, elem)?;
                        }
                        PatternKind::Wildcard => {}
                        _ => {
                            return Err(CodegenError::new("unsupported typed pattern variant"));
                        }
                    }
                }
                PatternKind::Rest(_) => {
                    // Rest patterns (...rest) — skip for now (would need slice extraction)
                    // TODO: implement rest pattern as nsl_list_slice(container, i, len)
                }
                _ => {
                    return Err(CodegenError::new(format!(
                        "unsupported pattern kind in destructuring at position {}", i
                    )));
                }
            }
        }
        Ok(())
    }

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
                            // M36: Check if this variable is slab-planned for zero-alloc
                            let slab_result = self.try_compile_slab_tensor(builder, state, &sym, expr);
                            match slab_result {
                                Ok(Some(val)) => val, // Slab allocation succeeded
                                _ => self.compile_expr(builder, state, expr)?, // Normal path
                            }
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

                        // M50: Track sparse tensor variables for end-to-end dispatch.
                        // Check if the RHS is a call to a sparse function or has Type::Sparse.
                        if let Some(expr) = value {
                            let is_sparse_type = matches!(self.node_type(expr.id), nsl_semantic::types::Type::Sparse { .. });
                            let is_sparse_call = if let ExprKind::Call { callee, .. } = &expr.kind {
                                if let ExprKind::Ident(fn_sym) = &callee.kind {
                                    let fn_name = self.resolve_sym(*fn_sym);
                                    fn_name.contains("sparse") || fn_name == "from_dense"
                                } else { false }
                            } else { false };
                            if is_sparse_type || is_sparse_call {
                                state.sparse_vars.insert(sym);
                            }
                        }

                        // Free intermediate tensor temporaries (keep init_val which is now owned by the variable)
                        self.free_tensor_temporaries(builder, state, Some(init_val));
                        // M38b: Free linear tensors consumed during this let-binding's RHS
                        self.free_linear_consumes(builder, state, Some(init_val));

                        // If the value was a closure lambda, record capture count for indirect call dispatch
                        if let Some(count) = self.last_lambda_capture_count.take() {
                            self.closure_info.insert(sym, count);
                        }
                    }
                    PatternKind::Tuple(sub_patterns) | PatternKind::List(sub_patterns) => {
                        let tuple_val = if let Some(expr) = value {
                            self.compile_expr(builder, state, expr)?
                        } else {
                            return Err(CodegenError::new("tuple/list destructuring requires a value"));
                        };

                        self.compile_destructure_patterns(builder, state, sub_patterns, tuple_val)?;
                    }
                    PatternKind::Struct { fields, .. } => {
                        // Top-level struct destructuring: let { x, y } = expr
                        let struct_val = if let Some(expr) = value {
                            self.compile_expr(builder, state, expr)?
                        } else {
                            return Err(CodegenError::new("struct destructuring requires a value"));
                        };
                        for field in fields {
                            let field_name = self.resolve_sym(field.name).to_string();
                            if !self.string_pool.contains_key(field_name.as_str()) {
                                self.intern_string(&field_name)?;
                            }
                            let key_str = self.compile_string_literal(builder, &field_name)?;
                            let field_val = self.compile_call_by_name(
                                builder, "nsl_dict_get_str", &[struct_val, key_str],
                            )?;
                            if let Some(ref pat) = field.pattern {
                                match &pat.kind {
                                    PatternKind::Ident(sym) => {
                                        let var = state.new_variable();
                                        builder.declare_var(var, cl_types::I64);
                                        builder.def_var(var, field_val);
                                        state.variables.insert(*sym, (var, cl_types::I64));
                                    }
                                    PatternKind::Tuple(nested) | PatternKind::List(nested) => {
                                        self.compile_destructure_patterns(builder, state, nested, field_val)?;
                                    }
                                    PatternKind::Wildcard => {}
                                    _ => {
                                        return Err(CodegenError::new(format!(
                                            "unsupported pattern in struct field '{}'", field_name
                                        )));
                                    }
                                }
                            } else {
                                let var = state.new_variable();
                                builder.declare_var(var, cl_types::I64);
                                builder.def_var(var, field_val);
                                state.variables.insert(field.name, (var, cl_types::I64));
                            }
                        }
                    }
                    _ => return Err(CodegenError::new("only ident, tuple, list, and struct patterns supported")),
                }
            }

            StmtKind::Assign { target, op, value } => {
                self.compile_assign(builder, state, target, *op, value)?;
            }

            StmtKind::Return(expr) => {
                if let Some(e) = expr {
                    let mut val = self.compile_expr(builder, state, e)?;
                    // Free intermediate tensor temporaries before returning (keep return value)
                    self.free_tensor_temporaries(builder, state, Some(val));
                    // M38b: Free linear tensors consumed during the return expression
                    self.free_linear_consumes(builder, state, Some(val));
                    // Stop and free any DataLoaders created in this scope
                    self.teardown_dataloaders(builder, state);
                    // @no_grad: resume tape before explicit return
                    if state.is_no_grad {
                        self.compile_call_by_name(builder, "nsl_tape_resume", &[])?;
                    }
                    // In element-wise unpack methods, bitcast f64→i64 for the return value
                    if state.dtype_unpack_ret_bitcast {
                        let vt = builder.func.dfg.value_type(val);
                        if vt == cranelift_codegen::ir::types::F64 {
                            val = builder.ins().bitcast(cranelift_codegen::ir::types::I64, cranelift_codegen::ir::MemFlags::new(), val);
                        }
                    }
                    builder.ins().return_(&[val]);
                } else {
                    // Stop and free any DataLoaders created in this scope
                    self.teardown_dataloaders(builder, state);
                    // @no_grad: resume tape before explicit return
                    if state.is_no_grad {
                        self.compile_call_by_name(builder, "nsl_tape_resume", &[])?;
                    }
                    builder.ins().return_(&[]);
                }
            }

            StmtKind::Expr(expr) => {
                let _ = self.compile_expr(builder, state, expr)?;
                // Free all tensor temporaries from this expression (none are kept)
                self.free_tensor_temporaries(builder, state, None);
                // M38b: Free linear tensors consumed during this expression statement
                self.free_linear_consumes(builder, state, None);
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
                let exit = state.loop_stack.last()
                    .map(|lc| lc.exit_block)
                    .ok_or_else(|| CodegenError::new("break outside loop"))?;
                // Free tensor temporaries from current loop iteration before jumping out
                self.emit_loop_scope_cleanup(builder, state);
                builder.ins().jump(exit, &[]);
            }

            StmtKind::Continue => {
                let cont = state.loop_stack.last()
                    .map(|lc| lc.continue_block)
                    .ok_or_else(|| CodegenError::new("continue outside loop"))?;
                // Free tensor temporaries from current loop iteration before restarting
                self.emit_loop_scope_cleanup(builder, state);
                builder.ins().jump(cont, &[]);
            }

            StmtKind::FnDef(fn_def) => {
                // Nested function definition: declare, compile, and bind name
                let base_name = self.resolve_sym(fn_def.name).to_string();
                let unique_name = format!("__nsl_nested_{}_{}", base_name, self.next_func_index());
                let sig = self.build_fn_signature(fn_def);
                let func_id = self.module
                    .declare_function(&unique_name, cranelift_module::Linkage::Local, &sig)
                    .map_err(|e| CodegenError::new(format!("failed to declare nested fn '{base_name}': {e}")))?;
                // Temporarily insert under base_name for compile_fn_def lookup, then restore
                let prev_entry = self.functions.remove(&base_name);
                self.functions.insert(base_name.clone(), (func_id, sig.clone()));

                // Compile the nested function body
                self.compile_fn_def(fn_def)?;

                // Remove temp entry and restore any previous function with the same name
                self.functions.remove(&base_name);
                if let Some(prev) = prev_entry {
                    self.functions.insert(base_name, prev);
                }

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
            | StmtKind::DatasetDef(_) | StmtKind::TokenizerDef(_) => {}

            StmtKind::DatatypeDef(_) => {
                // M23: custom datatype codegen — implemented in Task 9
            }

            StmtKind::ServeBlock(serve) => {
                self.compile_serve_block(builder, state, serve)?;
            }

            StmtKind::KernelDef(_) => {
                // Kernels are compiled in the compile_kernels pass (before functions).
            }

            StmtKind::QuantBlock(ref quant) => {
                self.compile_quant_block(builder, state, quant)?;
            }

            StmtKind::WhileLet { pattern, expr, body } => {
                self.compile_while_let(builder, state, pattern, expr, body)?;
            }

            StmtKind::Decorated { decorators, stmt } => {
                // Check for @no_grad and @fuse on nested function definitions
                if let StmtKind::FnDef(fn_def) = &stmt.kind {
                    for d in decorators {
                        if d.name.len() == 1 {
                            let dname = self.resolve_sym(d.name[0]);
                            if dname == "no_grad" {
                                let fname = self.resolve_sym(fn_def.name).to_string();
                                self.no_grad_fns.insert(fname);
                            } else if dname == "fp8_compute" {
                                let fname = self.resolve_sym(fn_def.name).to_string();
                                self.features.fp8_compute_fns.insert(fname);
                            } else if dname == "fuse" {
                                // Validate that the body contains only fusible elementwise ops
                                self.validate_fuse_body(fn_def)?;
                                // TODO(M26): emit training branch + fused kernel launch for @fuse fn
                                // For now, compile normally (unfused path is always correct).
                            } else if dname == "grammar" {
                                // M44: @grammar decorator on nested function
                                let fname = self.resolve_sym(fn_def.name).to_string();
                                let mut start_rule = String::new();
                                let mut grammar_source = String::new();
                                if let Some(ref dargs) = d.args {
                                    for arg in dargs {
                                        if let Some(name_sym) = arg.name {
                                            let arg_name = self.resolve_sym(name_sym).to_string();
                                            if arg_name == "start_rule" {
                                                if let nsl_ast::expr::ExprKind::StringLiteral(s) = &arg.value.kind {
                                                    start_rule = s.clone();
                                                }
                                            }
                                        } else if let nsl_ast::expr::ExprKind::StringLiteral(s) = &arg.value.kind {
                                            grammar_source = s.clone();
                                        }
                                    }
                                }
                                self.features.grammar_configs.insert(
                                    fname,
                                    crate::compiler::GrammarInfo { start_rule, grammar_source },
                                );
                            }
                        }
                    }
                }
                self.compile_stmt(builder, state, stmt)?;
            }

            _ => {
                return Err(CodegenError::new(
                    "unsupported statement in M3 codegen".to_string()
                ));
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
                        if is_float {
                            builder.ins().fdiv(old, new_val)
                        } else {
                            self.compile_divmod_guard(builder, state, new_val)?;
                            builder.ins().sdiv(old, new_val)
                        }
                    }
                };
                builder.def_var(var, final_val);
                // Free intermediate tensor temporaries (keep final_val which is now owned by the variable)
                self.free_tensor_temporaries(builder, state, Some(final_val));
                // M38b: Free linear tensors consumed during this assignment's RHS
                self.free_linear_consumes(builder, state, Some(final_val));
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
                                AssignOp::DivAssign => {
                                    self.compile_divmod_guard(builder, state, new_val)?;
                                    builder.ins().sdiv(old_val, new_val)
                                }
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
                                    let is_float = field.cl_type == cl_types::F64 || field.cl_type == cl_types::F32;
                                    match (op, is_float) {
                                        (AssignOp::AddAssign, true) => builder.ins().fadd(old_val, new_val),
                                        (AssignOp::SubAssign, true) => builder.ins().fsub(old_val, new_val),
                                        (AssignOp::MulAssign, true) => builder.ins().fmul(old_val, new_val),
                                        (AssignOp::DivAssign, true) => builder.ins().fdiv(old_val, new_val),
                                        (AssignOp::AddAssign, false) => builder.ins().iadd(old_val, new_val),
                                        (AssignOp::SubAssign, false) => builder.ins().isub(old_val, new_val),
                                        (AssignOp::MulAssign, false) => builder.ins().imul(old_val, new_val),
                                        (AssignOp::DivAssign, false) => {
                                            // Inline div-by-zero guard (can't call method due to borrow)
                                            let ok_blk = builder.create_block();
                                            let trap_blk = builder.create_block();
                                            let is_zero = builder.ins().icmp_imm(IntCC::Equal, new_val, 0);
                                            builder.ins().brif(is_zero, trap_blk, &[], ok_blk, &[]);
                                            builder.switch_to_block(trap_blk);
                                            builder.seal_block(trap_blk);
                                            builder.ins().trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
                                            builder.switch_to_block(ok_blk);
                                            builder.seal_block(ok_blk);
                                            state.current_block = Some(ok_blk);
                                            builder.ins().sdiv(old_val, new_val)
                                        }
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
                                    let is_float = field.cl_type == cl_types::F64 || field.cl_type == cl_types::F32;
                                    match (op, is_float) {
                                        (AssignOp::AddAssign, true) => builder.ins().fadd(old_val, new_val),
                                        (AssignOp::SubAssign, true) => builder.ins().fsub(old_val, new_val),
                                        (AssignOp::MulAssign, true) => builder.ins().fmul(old_val, new_val),
                                        (AssignOp::DivAssign, true) => builder.ins().fdiv(old_val, new_val),
                                        (AssignOp::AddAssign, false) => builder.ins().iadd(old_val, new_val),
                                        (AssignOp::SubAssign, false) => builder.ins().isub(old_val, new_val),
                                        (AssignOp::MulAssign, false) => builder.ins().imul(old_val, new_val),
                                        (AssignOp::DivAssign, false) => {
                                            // Inline div-by-zero guard (can't call method due to borrow)
                                            let ok_blk = builder.create_block();
                                            let trap_blk = builder.create_block();
                                            let is_zero = builder.ins().icmp_imm(IntCC::Equal, new_val, 0);
                                            builder.ins().brif(is_zero, trap_blk, &[], ok_blk, &[]);
                                            builder.switch_to_block(trap_blk);
                                            builder.seal_block(trap_blk);
                                            builder.ins().trap(cranelift_codegen::ir::TrapCode::unwrap_user(1));
                                            builder.switch_to_block(ok_blk);
                                            builder.seal_block(ok_blk);
                                            state.current_block = Some(ok_blk);
                                            builder.ins().sdiv(old_val, new_val)
                                        }
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

    /// Free intermediate tensor temporaries accumulated during expression compilation.
    /// `keep` is the final result value that should NOT be freed (it's owned by a variable).
    /// All other temporaries are intermediates from compound expressions (e.g. `a + b` in `a + b + c`).
    fn free_tensor_temporaries(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        keep: Option<Value>,
    ) {
        let temps = std::mem::take(&mut state.tensor_temporaries);
        // When inside a tape-recorded region (train step body), the autodiff tape
        // holds raw pointers to intermediate tensors (TapeOp `a`/`out` fields).
        // Freeing them here causes use-after-free during backward.
        if state.in_tape_region {
            return;
        }
        for temp in &temps {
            if Some(*temp) == keep {
                continue;
            }
            // Emit nsl_tensor_free(temp) — but only if block is not already filled
            if let Some(block) = state.current_block {
                if is_block_filled(builder, block) {
                    break;
                }
            }
            let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[*temp]);
        }
    }

    /// M38b: Free linear tensors that were consumed during the current statement.
    /// Called after `free_tensor_temporaries` at each statement boundary.
    /// `keep` is the value being assigned to a variable (should NOT be freed).
    ///
    /// Only active when `state.ownership_lowering.is_some()` — the pending list
    /// is empty otherwise so the loop is a no-op.
    fn free_linear_consumes(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        keep: Option<Value>,
    ) {
        if state.linear_consume_pending.is_empty() {
            return;
        }
        // Don't free inside tape-recorded regions — backward needs the data alive.
        if state.in_tape_region {
            state.linear_consume_pending.clear();
            return;
        }
        let pending = std::mem::take(&mut state.linear_consume_pending);
        for val in &pending {
            if Some(*val) == keep {
                continue;
            }
            if let Some(block) = state.current_block {
                if is_block_filled(builder, block) {
                    break;
                }
            }
            let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[*val]);
        }
    }

    /// Emit nsl_tensor_free calls for all tensor temporaries accumulated since the
    /// current loop scope started. Used at break/continue points.
    /// CRITICAL: Does NOT truncate tensor_temporaries — that only happens at natural scope exit.
    fn emit_loop_scope_cleanup(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
    ) {
        if let Some(&scope_start) = state.temp_scope_stack.last() {
            for &temp in &state.tensor_temporaries[scope_start..] {
                if let Some(block) = state.current_block {
                    if is_block_filled(builder, block) {
                        break;
                    }
                }
                let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[temp]);
            }
        }
    }

    /// Emit cleanup AND truncate temporaries at natural loop exit.
    fn cleanup_loop_scope(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
    ) {
        if let Some(scope_start) = state.temp_scope_stack.pop() {
            for &temp in &state.tensor_temporaries[scope_start..] {
                if let Some(block) = state.current_block {
                    if is_block_filled(builder, block) {
                        break;
                    }
                }
                let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[temp]);
            }
            state.tensor_temporaries.truncate(scope_start);
        }
    }

    /// Emit nsl_dataloader_stop + nsl_dataloader_free for all DataLoaders
    /// created in this scope. Called before function returns to prevent
    /// thread leaks and resource leaks.
    pub(crate) fn teardown_dataloaders(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
    ) {
        let loaders = std::mem::take(&mut state.dataloader_vars);
        for dl in &loaders {
            if let Some(block) = state.current_block {
                if is_block_filled(builder, block) {
                    break;
                }
            }
            let _ = self.compile_call_by_name(builder, "nsl_dataloader_stop", &[*dl]);
            let _ = self.compile_call_by_name(builder, "nsl_dataloader_free", &[*dl]);
        }
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
        let current = state.current_block.unwrap_or(then_bb);
        if !is_block_filled(builder, current) { builder.ins().jump(merge_block, &[]); }

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
            let current = state.current_block.unwrap_or(elif_then);
            if !is_block_filled(builder, current) { builder.ins().jump(merge_block, &[]); }

            current_else = elif_next;
        }

        if let Some(else_body) = else_block {
            builder.switch_to_block(current_else);
            builder.seal_block(current_else);
            state.current_block = Some(current_else);
            for s in &else_body.stmts { self.compile_stmt(builder, state, s)?; }
            let current = state.current_block.unwrap_or(current_else);
            if !is_block_filled(builder, current) { builder.ins().jump(merge_block, &[]); }
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

        state.temp_scope_stack.push(state.tensor_temporaries.len());
        state.loop_stack.push(LoopContext { continue_block: header_block, exit_block });
        for s in &body.stmts { self.compile_stmt(builder, state, s)?; }
        state.loop_stack.pop();

        let current = state.current_block.unwrap_or(body_block);
        if !is_block_filled(builder, current) {
            self.cleanup_loop_scope(builder, state);
            builder.ins().jump(header_block, &[]);
        } else {
            state.temp_scope_stack.pop();
        }

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

        state.temp_scope_stack.push(state.tensor_temporaries.len());
        state.loop_stack.push(LoopContext { continue_block: header_block, exit_block });
        for s in &body.stmts { self.compile_stmt(builder, state, s)?; }
        state.loop_stack.pop();

        let current = state.current_block.unwrap_or(body_block);
        if !is_block_filled(builder, current) {
            self.cleanup_loop_scope(builder, state);
            builder.ins().jump(header_block, &[]);
        } else {
            state.temp_scope_stack.pop();
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
        // Check if iterating over a fixed model array
        let iter_type = self.node_type(iterable.id).clone();
        if let Type::FixedModelArray { element_model, size } = &iter_type {
            return self.compile_for_model_array(builder, state, pattern, iterable, body, *element_model, *size);
        }

        // DataLoader iteration: `for batch in loader:`
        // Detect DataLoader by checking if the iterable variable was assigned from a DataLoader() call.
        // The semantic type is List<Dict<Str, Tensor>> but the runtime handle is an opaque pointer,
        // not a real list — we must use the DataLoader-specific iteration protocol.
        // Only route to DataLoader protocol when the type is positively known to be
        // a DataLoader (List<Dict<...>>). Previously, Type::Unknown also fell through
        // here, which could miscompile non-DataLoader iterables whose type inference failed.
        if self.is_dataloader_iterable(iterable) {
            return self.compile_for_dataloader(builder, state, pattern, iterable, body);
        }
        if matches!(iter_type, Type::Unknown) {
            eprintln!(
                "[nsl-codegen] warning: for-loop iterable has Unknown type — compiling as list iteration. \
                 If this is a DataLoader, ensure the variable type is inferred correctly."
            );
        }

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
        state.temp_scope_stack.push(state.tensor_temporaries.len());
        state.loop_stack.push(LoopContext { continue_block: increment_block, exit_block });
        for s in &body.stmts { self.compile_stmt(builder, state, s)?; }
        state.loop_stack.pop();

        let current = state.current_block.unwrap_or(body_block);
        if !is_block_filled(builder, current) {
            self.cleanup_loop_scope(builder, state);
            builder.ins().jump(increment_block, &[]);
        } else {
            state.temp_scope_stack.pop();
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

    #[allow(clippy::too_many_arguments)]
    fn compile_for_model_array(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        pattern: &nsl_ast::pattern::Pattern,
        iterable: &nsl_ast::expr::Expr,
        body: &nsl_ast::stmt::Block,
        element_model: nsl_ast::Symbol,
        size: i64,
    ) -> Result<(), CodegenError> {
        // Compile iterable to get base address of the array
        let base_val = self.compile_expr(builder, state, iterable)?;

        // Declare loop variable
        let loop_var_sym = match &pattern.kind {
            PatternKind::Ident(sym) => *sym,
            _ => return Err(CodegenError::new("only ident patterns supported in model array for-loops")),
        };
        let zero = builder.ins().iconst(cl_types::I64, 0);
        let elem_var = state.new_variable();
        builder.declare_var(elem_var, cl_types::I64);
        builder.def_var(elem_var, zero);
        state.variables.insert(loop_var_sym, (elem_var, cl_types::I64));

        // Register the loop variable's model type for method dispatch
        let model_name = self.resolve_sym(element_model).to_string();
        self.model_var_types.insert(loop_var_sym, model_name);

        // Counter variable
        let counter_var = state.new_variable();
        builder.declare_var(counter_var, cl_types::I64);
        builder.def_var(counter_var, zero);

        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let increment_block = builder.create_block();
        let exit_block = builder.create_block();

        builder.ins().jump(header_block, &[]);

        // Header: check i < size
        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let counter = builder.use_var(counter_var);
        let limit = builder.ins().iconst(cl_types::I64, size);
        let cond = builder.ins().icmp(IntCC::SignedLessThan, counter, limit);
        builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        // Body: load element pointer from base_val + i*8
        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);
        let counter = builder.use_var(counter_var);
        let eight = builder.ins().iconst(cl_types::I64, 8);
        let elem_offset = builder.ins().imul(counter, eight);
        let addr = builder.ins().iadd(base_val, elem_offset);
        let elem_ptr = builder.ins().load(cl_types::I64, MemFlags::trusted(), addr, 0);
        builder.def_var(elem_var, elem_ptr);

        // Compile body statements
        state.temp_scope_stack.push(state.tensor_temporaries.len());
        state.loop_stack.push(LoopContext { continue_block: increment_block, exit_block });
        for s in &body.stmts {
            self.compile_stmt(builder, state, s)?;
        }
        state.loop_stack.pop();

        let current = state.current_block.unwrap_or(body_block);
        if !is_block_filled(builder, current) {
            self.cleanup_loop_scope(builder, state);
            builder.ins().jump(increment_block, &[]);
        } else {
            state.temp_scope_stack.pop();
        }

        // Increment: counter++ then jump to header
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

    /// Check if an iterable expression is a DataLoader (typed as List<Dict<Str, Tensor>>).
    fn is_dataloader_iterable(&self, iterable: &nsl_ast::expr::Expr) -> bool {
        let ty = self.node_type(iterable.id).clone();
        matches!(ty, Type::List(ref elem) if matches!(**elem, Type::Dict(_, _)))
    }

    // ── DataLoader for-loop ──────────────────────────────────────────

    fn compile_for_dataloader(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        pattern: &nsl_ast::pattern::Pattern,
        iterable: &nsl_ast::expr::Expr,
        body: &nsl_ast::stmt::Block,
    ) -> Result<(), CodegenError> {
        let dl_val = self.compile_expr(builder, state, iterable)?;

        // Extract loop variable symbol (must be simple ident)
        let loop_var_sym = match &pattern.kind {
            PatternKind::Ident(sym) => *sym,
            _ => return Err(CodegenError::new("only ident patterns supported in DataLoader for-loops")),
        };

        // Declare cranelift variable for the batch pointer
        let batch_var = state.new_variable();
        builder.declare_var(batch_var, cl_types::I64);
        let zero = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(batch_var, zero);
        state.variables.insert(loop_var_sym, (batch_var, cl_types::I64));

        // Create blocks: header, body, cleanup, break_exit, exit
        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let cleanup_block = builder.create_block();
        let break_exit_block = builder.create_block();
        let exit_block = builder.create_block();

        builder.ins().jump(header_block, &[]);

        // Header: call nsl_dataloader_next_batch, branch on null
        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let batch_ptr = self.compile_call_by_name(builder, "nsl_dataloader_next_batch", &[dl_val])?;
        builder.def_var(batch_var, batch_ptr);
        let is_null = builder.ins().icmp_imm(IntCC::Equal, batch_ptr, 0);
        builder.ins().brif(is_null, exit_block, &[], body_block, &[]);

        // Body: compile loop body statements
        // break → break_exit_block (frees batch, then stops DL)
        // continue → cleanup_block (frees batch, loops back)
        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        // Rely on codegen-level tensor_temporaries for per-statement cleanup.
        // Do NOT use scope_begin/scope_end — it double-frees tensors that are
        // already freed by free_tensor_temporaries in called functions.
        state.temp_scope_stack.push(state.tensor_temporaries.len());
        state.loop_stack.push(LoopContext { continue_block: cleanup_block, exit_block: break_exit_block });
        for s in &body.stmts {
            self.compile_stmt(builder, state, s)?;
        }
        state.loop_stack.pop();

        let current = state.current_block.unwrap_or(body_block);
        if !is_block_filled(builder, current) {
            self.cleanup_loop_scope(builder, state);
            builder.ins().jump(cleanup_block, &[]);
        } else {
            state.temp_scope_stack.pop();
        }

        // Cleanup: free batch dict, loop back
        builder.switch_to_block(cleanup_block);
        builder.seal_block(cleanup_block);
        state.current_block = Some(cleanup_block);
        let batch_to_free = builder.use_var(batch_var);
        self.compile_call_by_name(builder, "nsl_dict_free", &[batch_to_free])?;
        builder.ins().jump(header_block, &[]);

        // Break exit: end tensor scope, free the current batch dict, then stop the DataLoader
        builder.switch_to_block(break_exit_block);
        builder.seal_block(break_exit_block);
        state.current_block = Some(break_exit_block);
        let batch_to_free = builder.use_var(batch_var);
        self.compile_call_by_name(builder, "nsl_dict_free", &[batch_to_free])?;
        self.compile_call_by_name(builder, "nsl_dataloader_stop", &[dl_val])?;
        builder.ins().jump(exit_block, &[]);

        // Exit (normal exhaustion): reset the dataloader for potential next epoch
        builder.seal_block(header_block);
        builder.switch_to_block(exit_block);
        builder.seal_block(exit_block);
        state.current_block = Some(exit_block);
        self.compile_call_by_name(builder, "nsl_dataloader_reset", &[dl_val])?;

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
                        let current = state.current_block.unwrap_or(arm_block);
                        if !is_block_filled(builder, current) {
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
                    let current = state.current_block.unwrap_or(arm_block);
                    if !is_block_filled(builder, current) {
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
                    let variant_name = if !path.is_empty() {
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
                        let current = state.current_block.unwrap_or(arm_block);
                        if !is_block_filled(builder, current) {
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
        // M43b: Pipeline parallel detection
        if self.features.pipeline_config.is_some() {
            return self.compile_train_block_pipelined(builder, state, train);
        }

        // ── 1. Extract config from train(...) args ──────────────────────
        let mut model_sym: Option<nsl_ast::Symbol> = None;
        let mut epochs: i64 = 1;
        let mut grad_accumulation_steps: i64 = 1;
        let mut grad_clip: f64 = f64::MAX; // default: no clipping

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
                    "grad_accumulation" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            grad_accumulation_steps = (*n).max(1);
                        }
                    }
                    "grad_clip" => {
                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                            grad_clip = *f;
                        } else if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            grad_clip = *n as f64;
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
        let mut step_body: Option<(&nsl_ast::stmt::Block, nsl_ast::Symbol)> = None;
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
                TrainSection::Step { param, body } => {
                    step_body = Some((body, *param));
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
                TrainSection::Data(stmts) => {
                    // Compile data section stmts — typically creates a DataLoader
                    for stmt in stmts {
                        self.compile_stmt(builder, state, stmt)?;
                    }
                }
                _ => {} // Eval, Distribute, Stmt — not yet implemented
            }
        }

        if optimizer_name.is_empty() {
            return Err(CodegenError::new("train block requires an optimizer section"));
        }

        let (step_body, step_param_sym) = step_body.ok_or_else(|| {
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

        // ── 5b. Allocate gradient accumulation buffers (if grad_accumulation_steps > 1) ──
        // These persist across batches within each accumulation window. Each buffer
        // is zeros_like(param) and gets += each batch's grads, then zeroed after
        // the optimizer step every N batches.
        let accum_list = if grad_accumulation_steps > 1 {
            let list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            for field in &layout.fields {
                let field_val = builder.ins().load(
                    field.cl_type,
                    MemFlags::trusted(),
                    model_ptr,
                    field.offset as i32,
                );
                let zeros = self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[field_val])?;
                self.compile_call_by_name(builder, "nsl_list_push", &[list, zeros])?;
            }
            Some(list)
        } else {
            None
        };

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

        // ── 7. Inner batch loop (when DataLoader exists) or single-step (backward compat) ──

        let has_dataloader = state.dataloader_vars.last().copied();

        // Declare step parameter variable
        let step_param_var = state.new_variable();
        builder.declare_var(step_param_var, cl_types::I64);
        let init_null = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_param_var, init_null);
        state.variables.insert(step_param_sym, (step_param_var, cl_types::I64));

        // If DataLoader exists: emit inner batch loop
        // Structure: reset → [batch_header: next_batch → null check → batch_body | batch_exit]
        let batch_header_block;
        let batch_body_block;
        let batch_exit_block = builder.create_block();

        if let Some(dl_handle) = has_dataloader {
            // Reset DataLoader at epoch start
            self.compile_call_by_name(builder, "nsl_dataloader_reset", &[dl_handle])?;

            batch_header_block = builder.create_block();
            batch_body_block = builder.create_block();

            builder.ins().jump(batch_header_block, &[]);

            // Batch header: get next batch, check for null (exhausted)
            builder.switch_to_block(batch_header_block);
            builder.seal_block(batch_header_block);
            let batch_ptr = self.compile_call_by_name(builder, "nsl_dataloader_next_batch", &[dl_handle])?;
            let null_check = builder.ins().iconst(cl_types::I64, 0);
            let is_done = builder.ins().icmp(IntCC::Equal, batch_ptr, null_check);
            builder.ins().brif(is_done, batch_exit_block, &[], batch_body_block, &[]);

            // Batch body
            builder.switch_to_block(batch_body_block);
            builder.seal_block(batch_body_block);
            state.current_block = Some(batch_body_block);
            builder.def_var(step_param_var, batch_ptr);
        } else {
            // No DataLoader — step body runs once per epoch (backward compat)
            batch_header_block = body_block; // unused, just needs a value
            batch_body_block = body_block;   // we're already in it
        }

        // 7a. Prefetch batch tensors to GPU (if on GPU) to overlap
        // page migration with tape setup. This reduces first-access latency
        // from unified memory page faults.
        if has_dataloader.is_some() {
            let batch_val = builder.use_var(step_param_var);
            // Prefetch input_ids and labels from the batch dict
            let k_ids = self.compile_string_literal(builder, "input_ids")?;
            let k_lbl = self.compile_string_literal(builder, "labels")?;
            let ids_tensor = self.compile_call_by_name(builder, "nsl_dict_get_str", &[batch_val, k_ids])?;
            let lbl_tensor = self.compile_call_by_name(builder, "nsl_dict_get_str", &[batch_val, k_lbl])?;
            let device_1 = builder.ins().iconst(cl_types::I64, 1);
            self.compile_call_by_name(builder, "nsl_tensor_prefetch", &[ids_tensor, device_1])?;
            self.compile_call_by_name(builder, "nsl_tensor_prefetch", &[lbl_tensor, device_1])?;
        }

        // ── 7b. Forward pass + backward pass ─────────────────────────
        // When source AD is enabled, attempt compile-time backward graph
        // generation. If extraction fails (dynamic control flow), fall back
        // to the tape-based AD path.
        let (grads_list, loss_val) = if self.features.source_ad_enabled {
            // === Source AD path (compile-time backward) ===
            eprintln!("[nsl] Using source-to-source AD for backward pass");

            // 1. Set training mode
            let true_val = builder.ins().iconst(cl_types::I8, 1);
            self.compile_call_by_name(builder, "nsl_set_training_mode", &[true_val])?;

            // 2. Try to extract Wengert list from step body
            let mut extractor = crate::source_ad::WengertExtractor::new(self.interner);
            let extraction_ok = extractor.extract_stmts(&step_body.stmts);

            if !extraction_ok {
                // Source AD extraction failed — fall back to tape
                eprintln!("[nsl] source AD extraction failed, falling back to tape-based AD");

                // Undo training mode — tape path sets it itself
                let false_val = builder.ins().iconst(cl_types::I8, 0);
                self.compile_call_by_name(builder, "nsl_set_training_mode", &[false_val])?;

                self.compile_tape_backward(builder, state, step_body, param_list)?
            } else {
                // 3. Compile forward pass normally (no tape wrapping)
                state.in_tape_region = false;
                for stmt in &step_body.stmts {
                    self.compile_stmt(builder, state, stmt)?;
                }

                // 4. Find loss variable
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

                // 5. Generate adjoint backward graph
                let start_var = extractor.next_var_id();
                let mut gen = crate::source_ad::AdjointGenerator::new(start_var);
                let adjoint = gen.generate(extractor.wengert_list());

                // 6. Build VarMap: map symbolic VarIds to Cranelift Values
                let mut primal_vars = crate::wengert_lower::VarMap::new();
                for (sym, vid) in extractor.symbol_var_map() {
                    if let Some(&(cvar, _)) = state.variables.get(sym) {
                        primal_vars.insert(*vid, builder.use_var(cvar));
                    }
                }

                // 7. Lower adjoint to Cranelift IR
                // If lowering fails (e.g. unsupported op), fall back to tape AD.
                // Note: at this point the forward pass is already emitted, so we
                // cannot cleanly re-run through tape. Instead we propagate the error
                // with a clear diagnostic so the user can disable --source-ad.
                let grad_vars = match crate::wengert_lower::compile_wengert_ops(
                    self, builder, state, &adjoint, &primal_vars,
                ) {
                    Ok(gv) => gv,
                    Err(e) => {
                        eprintln!(
                            "[nsl] source AD lowering failed ({}), \
                             falling back to tape AD is not possible after forward emit; \
                             rerun without --source-ad",
                            e
                        );
                        return Err(e);
                    }
                };

                // 8. Collect parameter gradients into grads_list (NslList)
                let grads = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
                for field in &layout.fields {
                    // Find the param's symbol -> VarId -> adjoint VarId -> Cranelift Value
                    let grad_val = extractor
                        .param_var_ids()
                        .iter()
                        .find(|(sym, _)| self.resolve_sym(*sym) == field.name)
                        .and_then(|(_, vid)| gen.adjoint_of(*vid))
                        .and_then(|adj_vid| grad_vars.get(&adj_vid).copied())
                        .unwrap_or_else(|| builder.ins().iconst(cl_types::I64, 0));
                    self.compile_call_by_name(builder, "nsl_list_push", &[grads, grad_val])?;
                }

                let false_val = builder.ins().iconst(cl_types::I8, 0);
                self.compile_call_by_name(builder, "nsl_set_training_mode", &[false_val])?;

                (grads, loss_val)
            }
        } else {
            // === Tape AD path (runtime backward) ===
            self.compile_tape_backward(builder, state, step_body, param_list)?
        };

        // 7e1b. Debug training: emit gradient checksum to catch silent corruption.
        // Prints sum(abs(grad)) per parameter — detects NaN, zero, and misrouted gradients.
        if self.compile_options.debug_training {
            let num_p = builder.ins().iconst(cl_types::I64, num_params as i64);
            self.compile_call_by_name(builder, "nsl_debug_grad_checksum", &[grads_list, num_p])?;
        }

        // 7e2. Gradient clipping (only if grad_clip was specified)
        if grad_clip < f64::MAX {
            let max_norm_val = builder.ins().f64const(grad_clip);
            self.compile_call_by_name(builder, "nsl_clip_grad_norm", &[grads_list, max_norm_val])?;
        }

        // 7e3. Gradient accumulation: accumulate this batch's grads into
        // persistent buffers, then free the per-batch grads immediately.
        // When not accumulating (steps == 1), grads_list is used directly
        // by the optimizer and freed after the step.
        if let Some(accum) = accum_list {
            // accum[i] += grads[i] for each parameter
            for i in 0..num_params {
                let idx = builder.ins().iconst(cl_types::I64, i as i64);
                let accum_buf = self.compile_call_by_name(builder, "nsl_list_get", &[accum, idx])?;
                let grad = self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, idx])?;
                let n_elems = self.compile_call_by_name(builder, "nsl_tensor_len", &[accum_buf])?;
                self.compile_call_by_name(builder, "nsl_grad_accumulate_add", &[accum_buf, grad, n_elems])?;
            }

            // Free this batch's gradient tensors (accumulated into buffers)
            for i in 0..num_params {
                let idx = builder.ins().iconst(cl_types::I64, i as i64);
                let grad_val = self.compile_call_by_name(
                    builder, "nsl_list_get", &[grads_list, idx],
                )?;
                self.compile_call_by_name(builder, "nsl_tensor_free", &[grad_val])?;
            }
            self.compile_call_by_name(builder, "nsl_list_free", &[grads_list])?;
        }

        // 7e4. Gradient accumulation gate: only step optimizer every N batches
        let optimizer_block = builder.create_block();
        let post_optimizer_block = builder.create_block();

        if grad_accumulation_steps > 1 {
            let sc = builder.use_var(step_count_var);
            let accum_val = builder.ins().iconst(cl_types::I64, grad_accumulation_steps);
            let one_ga = builder.ins().iconst(cl_types::I64, 1);
            let sc_plus_one = builder.ins().iadd(sc, one_ga);
            let rem = builder.ins().srem(sc_plus_one, accum_val);
            let zero_check = builder.ins().iconst(cl_types::I64, 0);
            let should_step = builder.ins().icmp(IntCC::Equal, rem, zero_check);
            builder.ins().brif(should_step, optimizer_block, &[], post_optimizer_block, &[]);

            builder.switch_to_block(optimizer_block);
            builder.seal_block(optimizer_block);
            state.current_block = Some(optimizer_block);
        } else {
            // No accumulation — always step
            builder.ins().jump(optimizer_block, &[]);
            builder.switch_to_block(optimizer_block);
            builder.seal_block(optimizer_block);
            state.current_block = Some(optimizer_block);
        }

        // 7f. Optimizer step: for each param, call optimizer step function
        // When accumulating, use accum_list (accumulated grads); otherwise use grads_list directly.
        let opt_grads = if let Some(accum) = accum_list { accum } else { grads_list };

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

        // M43b: ZeRO Stage 1+ — all-reduce gradients before optimizer step
        let zero_enabled = self.features.zero_stage.filter(|&s| s >= 1).is_some();
        if zero_enabled {
            let num_p = builder.ins().iconst(cl_types::I64, num_params as i64);
            self.compile_call_by_name(builder, "nsl_zero_reduce_grads", &[opt_grads, num_p])?;
        }

        for i in 0..num_params {
            let idx = builder.ins().iconst(cl_types::I64, i as i64);

            // M43b: ZeRO Stage 1+ — skip optimizer step for params not owned by this rank
            let zero_merge_block = if zero_enabled {
                let owns = self.compile_call_by_name(builder, "nsl_zero_owns_param", &[idx])?;
                let one_val = builder.ins().iconst(cl_types::I64, 1);
                let should_skip = builder.ins().icmp(IntCC::NotEqual, owns, one_val);
                let skip_block = builder.create_block();
                let step_block = builder.create_block();
                builder.ins().brif(should_skip, skip_block, &[], step_block, &[]);

                builder.switch_to_block(step_block);
                builder.seal_block(step_block);
                Some(skip_block)
            } else {
                None
            };

            // Get param tensor from param_list
            let param_val = self.compile_call_by_name(
                builder, "nsl_list_get", &[param_list, idx],
            )?;

            // Get gradient from opt_grads (accumulated buffer or direct grads)
            let grad_val = self.compile_call_by_name(
                builder, "nsl_list_get", &[opt_grads, idx],
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
                        &[param_val, grad_val, state_buf_1[i], lr, momentum_const, weight_decay_const, nesterov_const],
                    )?;
                }
                "soap" => {
                    let t_val_s = builder.use_var(step_count_var);
                    let one_s = builder.ins().iconst(cl_types::I64, 1);
                    let t_plus_s = builder.ins().iadd(t_val_s, one_s);
                    let t_float_s = builder.ins().fcvt_from_sint(cl_types::F64, t_plus_s);
                    self.compile_call_by_name(
                        builder, &opt_fn,
                        &[param_val, grad_val, state_buf_1[i], state_buf_2[i], lr, beta1_const, beta2_const, eps_const, t_float_s],
                    )?;
                }
                _ => {
                    return Err(CodegenError::new(format!(
                        "unsupported optimizer '{}' in train block", optimizer_name
                    )));
                }
            }

            // M43b: ZeRO — merge the skip and step paths
            if let Some(skip_block) = zero_merge_block {
                builder.ins().jump(skip_block, &[]);
                builder.switch_to_block(skip_block);
                builder.seal_block(skip_block);
            }
        }

        // M43b: ZeRO Stage 1+ — all-gather updated params after optimizer step
        if zero_enabled {
            self.compile_call_by_name(builder, "nsl_zero_step", &[])?;
        }

        // 7g. Post-optimizer cleanup: zero accum buffers or free direct grads
        if let Some(accum) = accum_list {
            // Zero the accumulation buffers after optimizer step so the next
            // N batches start from a clean slate.
            for i in 0..num_params {
                let idx = builder.ins().iconst(cl_types::I64, i as i64);
                let buf = self.compile_call_by_name(builder, "nsl_list_get", &[accum, idx])?;
                let n_elems = self.compile_call_by_name(builder, "nsl_tensor_len", &[buf])?;
                self.compile_call_by_name(builder, "nsl_grad_zero", &[buf, n_elems])?;
            }
        } else {
            // No accumulation — free gradient tensors and grads_list every batch
            for i in 0..num_params {
                let idx = builder.ins().iconst(cl_types::I64, i as i64);
                let grad_val = self.compile_call_by_name(
                    builder, "nsl_list_get", &[grads_list, idx],
                )?;
                self.compile_call_by_name(builder, "nsl_tensor_free", &[grad_val])?;
            }
            self.compile_call_by_name(builder, "nsl_list_free", &[grads_list])?;
        }

        // Jump to post-optimizer block (merges optimizer and skip paths)
        builder.ins().jump(post_optimizer_block, &[]);
        builder.switch_to_block(post_optimizer_block);
        builder.seal_block(post_optimizer_block);
        state.current_block = Some(post_optimizer_block);

        // 7g2. Scheduler: update learning rate if scheduler is configured
        // NOTE: step_count is incremented AFTER the scheduler call so that
        // step 0 produces the step-0 learning rate (e.g. warmup starts correctly).
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

        // 7h. Increment step count (after scheduler so step 0 uses the initial LR)
        let sc = builder.use_var(step_count_var);
        let one_i64 = builder.ins().iconst(cl_types::I64, 1);
        let sc_next = builder.ins().iadd(sc, one_i64);
        builder.def_var(step_count_var, sc_next);

        // 7i. Callbacks: compile on_step body with step_count and loss bound
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

        // ── 8. Close batch loop (if DataLoader) and increment epoch ──────
        let current = state.current_block.unwrap_or(batch_body_block);
        if has_dataloader.is_some() {
            // Jump back to batch_header for next batch
            if !is_block_filled(builder, current) {
                builder.ins().jump(batch_header_block, &[]);
            }
            // batch_exit: all batches done → jump to epoch increment
            builder.switch_to_block(batch_exit_block);
            builder.seal_block(batch_exit_block);
            state.current_block = Some(batch_exit_block);
            builder.ins().jump(increment_block, &[]);
        } else {
            // No DataLoader — single step per epoch, jump to increment
            if !is_block_filled(builder, current) {
                builder.ins().jump(increment_block, &[]);
            }
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

        // Free param_list after training loop completes
        self.compile_call_by_name(builder, "nsl_list_free", &[param_list])?;

        // Free optimizer state buffers (momentum/velocity tensors)
        for &buf in &state_buf_1 {
            self.compile_call_by_name(builder, "nsl_tensor_free", &[buf])?;
        }
        for &buf in &state_buf_2 {
            self.compile_call_by_name(builder, "nsl_tensor_free", &[buf])?;
        }

        // Free gradient accumulation buffers (if allocated)
        if let Some(accum) = accum_list {
            for i in 0..num_params {
                let idx = builder.ins().iconst(cl_types::I64, i as i64);
                let buf = self.compile_call_by_name(builder, "nsl_list_get", &[accum, idx])?;
                self.compile_call_by_name(builder, "nsl_tensor_free", &[buf])?;
            }
            self.compile_call_by_name(builder, "nsl_list_free", &[accum])?;
        }

        Ok(())
    }

    /// Emit the tape-based AD backward pass: tape_start, compile forward,
    /// find loss, tape_backward, tape_stop. Returns `(grads_list, loss_val)`.
    fn compile_tape_backward(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        step_body: &nsl_ast::stmt::Block,
        param_list: Value,
    ) -> Result<(Value, Value), CodegenError> {
        // Set training mode = true, then start tape recording
        let true_val = builder.ins().iconst(cl_types::I8, 1);
        self.compile_call_by_name(builder, "nsl_set_training_mode", &[true_val])?;
        self.compile_call_by_name(builder, "nsl_tape_start", &[param_list])?;

        // Compile step body stmts
        // Suppress tensor temporary cleanup — tape holds raw pointers to intermediates.
        state.in_tape_region = true;
        for stmt in &step_body.stmts {
            self.compile_stmt(builder, state, stmt)?;
        }
        state.in_tape_region = false;

        // Find loss variable — look for "loss" in state.variables by name
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

        // Run backward pass
        let grads_list = self.compile_call_by_name(
            builder,
            "nsl_tape_backward",
            &[loss_val, param_list],
        )?;

        // Stop tape and restore eval mode
        self.compile_call_by_name(builder, "nsl_tape_stop", &[])?;
        let false_val = builder.ins().iconst(cl_types::I8, 0);
        self.compile_call_by_name(builder, "nsl_set_training_mode", &[false_val])?;

        Ok((grads_list, loss_val))
    }

    /// M43b: Emit pipeline-parallel training loop with gradient serialization.
    ///
    /// When a model carries `@pipeline(stages=N)`, the train block emits:
    ///   1. `nsl_pipeline_init(num_stages, schedule_type, num_micro_batches)`
    ///   2. Extract model param_list, optimizer config, and step body.
    ///   3. Forward pass under tape recording — compile step body to produce
    ///      activations and loss.
    ///   4. Activation send — serialize the loss tensor to the next pipeline
    ///      stage via `nsl_pipeline_send`.
    ///   5. Backward pass — `nsl_tape_backward` computes per-parameter
    ///      gradients from the recorded tape.
    ///   6. Gradient send — serialize each parameter gradient to the previous
    ///      pipeline stage via `nsl_pipeline_send_grad`.
    ///   7. Optimizer step — apply optimizer update using the computed
    ///      gradients (same dispatch as the non-pipelined path).
    ///   8. `nsl_pipeline_barrier()` — synchronize all stages.
    ///   9. Cleanup — free gradient tensors, param_list, optimizer buffers,
    ///      and `nsl_pipeline_destroy()`.
    ///
    /// Model partitioning (which layers run on which stage) is deferred to
    /// M43c; the initial implementation runs the full model in a single
    /// process with logical stage-to-stage communication.
    fn compile_train_block_pipelined(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        train: &nsl_ast::block::TrainBlock,
    ) -> Result<(), CodegenError> {
        let config = self.features.pipeline_config.clone().unwrap();
        let num_stages = config.num_stages;

        // ── 1. Pipeline init ────────────────────────────────────────────
        let v_stages = builder.ins().iconst(cl_types::I64, num_stages as i64);
        let v_schedule = builder.ins().iconst(
            cl_types::I64,
            match config.schedule_type {
                crate::pipeline::ScheduleType::OneF1B => 0i64,
                crate::pipeline::ScheduleType::GPipe => 1i64,
            },
        );
        let v_micro = builder.ins().iconst(cl_types::I64, 8); // default micro-batches
        self.compile_call_by_name(builder, "nsl_pipeline_init", &[v_stages, v_schedule, v_micro])?;

        // ── 2. Extract config from train(...) args ──────────────────────
        let mut model_sym: Option<nsl_ast::Symbol> = None;
        let mut optimizer_name = String::new();
        let mut lr_value: f64 = 0.01;
        let mut momentum_value: f64 = 0.0;
        let mut dampening_value: f64 = 0.0;
        let mut weight_decay_value: f64 = 0.0;
        let mut nesterov_value: bool = false;
        let mut beta1_value: f64 = 0.9;
        let mut beta2_value: f64 = 0.999;
        let mut eps_value: f64 = 1e-8;
        let mut step_body: Option<(&nsl_ast::stmt::Block, nsl_ast::Symbol)> = None;

        for arg in &train.config {
            if let Some(name_sym) = arg.name {
                let name = self.resolve_sym(name_sym).to_string();
                if name == "model" {
                    if let ExprKind::Ident(sym) = &arg.value.kind {
                        model_sym = Some(*sym);
                    }
                }
            }
        }

        for section in &train.sections {
            match section {
                TrainSection::Optimizer(expr) => {
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
                TrainSection::Step { param, body } => {
                    step_body = Some((body, *param));
                }
                TrainSection::Data(stmts) => {
                    for stmt in stmts {
                        self.compile_stmt(builder, state, stmt)?;
                    }
                }
                _ => {}
            }
        }

        let model_sym = model_sym.ok_or_else(|| {
            CodegenError::new("pipelined train block requires 'model=<ident>' config argument")
        })?;

        if optimizer_name.is_empty() {
            return Err(CodegenError::new("pipelined train block requires an optimizer section"));
        }

        let (step_body, step_param_sym) = step_body.ok_or_else(|| {
            CodegenError::new("pipelined train block requires a step section")
        })?;

        // ── 3. Resolve model and build param_list ───────────────────────
        let (model_var, _) = *state.variables.get(&model_sym).ok_or_else(|| {
            CodegenError::new(format!(
                "undefined model variable '{}' in pipelined train block",
                self.resolve_sym(model_sym)
            ))
        })?;
        let model_ptr = builder.use_var(model_var);

        let model_var_name = self.resolve_sym(model_sym).to_string();
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
            found_name.unwrap_or_else(|| model_var_name.clone())
        };

        let layout = self.struct_layouts.get(&model_type_name).cloned().ok_or_else(|| {
            CodegenError::new(format!(
                "no struct layout found for model '{}' in pipelined train block",
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

        // ── 4. Create optimizer state buffers ───────────────────────────
        let num_state_buffers = match optimizer_name.as_str() {
            "adam" | "adamw" | "soap" => 2,
            _ => 1,
        };

        let mut state_buf_1: Vec<Value> = Vec::new();
        let mut state_buf_2: Vec<Value> = Vec::new();
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

        // ── 5. Declare step parameter and step counter ──────────────────
        let step_param_var = state.new_variable();
        builder.declare_var(step_param_var, cl_types::I64);
        let init_null = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_param_var, init_null);
        state.variables.insert(step_param_sym, (step_param_var, cl_types::I64));

        let step_count_var = state.new_variable();
        builder.declare_var(step_count_var, cl_types::I64);
        let zero_i64 = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_count_var, zero_i64);

        let lr_var = state.new_variable();
        builder.declare_var(lr_var, cl_types::F64);
        let lr_const = builder.ins().f64const(lr_value);
        builder.def_var(lr_var, lr_const);

        // ── 6. Forward pass under tape recording ────────────────────────
        let true_val = builder.ins().iconst(cl_types::I8, 1);
        self.compile_call_by_name(builder, "nsl_set_training_mode", &[true_val])?;
        self.compile_call_by_name(builder, "nsl_tape_start", &[param_list])?;

        state.in_tape_region = true;
        for stmt in &step_body.stmts {
            self.compile_stmt(builder, state, stmt)?;
        }
        state.in_tape_region = false;

        // Find loss variable
        let loss_val = {
            let mut found = None;
            for (sym, (var, _)) in &state.variables {
                if self.resolve_sym(*sym) == "loss" {
                    found = Some(builder.use_var(*var));
                    break;
                }
            }
            found.ok_or_else(|| {
                CodegenError::new("pipelined train step body must assign to a variable named 'loss'")
            })?
        };

        // ── 7. Activation send — send loss to next stage ────────────────
        // In single-process pipeline, stage 0 sends activations to logical
        // stage 1. The runtime's shared-memory backend serializes the tensor
        // into a mailbox keyed by (dst_rank, tag).
        let zero_tag = builder.ins().iconst(cl_types::I64, 0);
        let zero_stream = builder.ins().iconst(cl_types::I64, 0);
        let next_stage = builder.ins().iconst(cl_types::I64, 1);
        self.compile_call_by_name(
            builder, "nsl_pipeline_send",
            &[loss_val, next_stage, zero_tag, zero_stream],
        )?;

        // ── 8. Backward pass — tape backward + stop ─────────────────────
        let grads_list = self.compile_call_by_name(
            builder, "nsl_tape_backward",
            &[loss_val, param_list],
        )?;
        self.compile_call_by_name(builder, "nsl_tape_stop", &[])?;

        let false_val = builder.ins().iconst(cl_types::I8, 0);
        self.compile_call_by_name(builder, "nsl_set_training_mode", &[false_val])?;

        // ── 9. Gradient send — serialize each param gradient ────────────
        // Send gradients to the previous stage (stage 0 receives gradients
        // from stage 1 in the backward direction). Each gradient is tagged
        // with its parameter index for correct matching.
        let prev_stage = builder.ins().iconst(cl_types::I64, 0);
        for i in 0..num_params {
            let idx = builder.ins().iconst(cl_types::I64, i as i64);
            let grad_val = self.compile_call_by_name(
                builder, "nsl_list_get",
                &[grads_list, idx],
            )?;
            let grad_tag = builder.ins().iconst(cl_types::I64, i as i64);
            self.compile_call_by_name(
                builder, "nsl_pipeline_send_grad",
                &[grad_val, prev_stage, grad_tag, zero_stream],
            )?;
        }

        // ── 10. Optimizer step ──────────────────────────────────────────
        let optimizer_fn_name = match optimizer_name.as_str() {
            "sgd" => "nsl__optim__sgd__sgd_step",
            "adam" => "nsl__optim__adam__adam_step",
            "adamw" => "nsl__optim__adamw__adamw_step",
            "lion" => "nsl__optim__lion__lion_step",
            "muon" => "nsl__optim__muon__muon_step",
            "soap" => "nsl__optim__soap__soap_step",
            _ => {
                return Err(CodegenError::new(format!(
                    "unsupported optimizer '{}' in pipelined train block", optimizer_name
                )));
            }
        };

        let opt_fn = if self.functions.contains_key(optimizer_fn_name) {
            optimizer_fn_name.to_string()
        } else {
            let simple = format!("{}_step", optimizer_name);
            if self.functions.contains_key(&simple) {
                simple
            } else if self.runtime_fns.contains_key(optimizer_fn_name) {
                optimizer_fn_name.to_string()
            } else if self.runtime_fns.contains_key(&simple) {
                simple
            } else {
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
            let param_val = self.compile_call_by_name(
                builder, "nsl_list_get", &[param_list, idx],
            )?;
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
                        &[param_val, grad_val, state_buf_1[i], lr, momentum_const, weight_decay_const, nesterov_const],
                    )?;
                }
                "soap" => {
                    let t_val_p = builder.use_var(step_count_var);
                    let one_p = builder.ins().iconst(cl_types::I64, 1);
                    let t_plus_p = builder.ins().iadd(t_val_p, one_p);
                    let t_float_p = builder.ins().fcvt_from_sint(cl_types::F64, t_plus_p);
                    self.compile_call_by_name(
                        builder, &opt_fn,
                        &[param_val, grad_val, state_buf_1[i], state_buf_2[i], lr, beta1_const, beta2_const, eps_const, t_float_p],
                    )?;
                }
                _ => {
                    return Err(CodegenError::new(format!(
                        "unsupported optimizer '{}' in pipelined train block", optimizer_name
                    )));
                }
            }
        }

        // ── 11. Increment step count ────────────────────────────────────
        let sc = builder.use_var(step_count_var);
        let one_i64 = builder.ins().iconst(cl_types::I64, 1);
        let sc_next = builder.ins().iadd(sc, one_i64);
        builder.def_var(step_count_var, sc_next);

        // ── 12. Barrier — synchronize all pipeline stages ───────────────
        self.compile_call_by_name(builder, "nsl_pipeline_barrier", &[])?;

        // ── 13. Cleanup — free gradients, param_list, optimizer buffers ─
        for i in 0..num_params {
            let idx = builder.ins().iconst(cl_types::I64, i as i64);
            let grad_val = self.compile_call_by_name(
                builder, "nsl_list_get", &[grads_list, idx],
            )?;
            self.compile_call_by_name(builder, "nsl_tensor_free", &[grad_val])?;
        }
        self.compile_call_by_name(builder, "nsl_list_free", &[grads_list])?;
        self.compile_call_by_name(builder, "nsl_list_free", &[param_list])?;

        for &buf in &state_buf_1 {
            self.compile_call_by_name(builder, "nsl_tensor_free", &[buf])?;
        }
        for &buf in &state_buf_2 {
            self.compile_call_by_name(builder, "nsl_tensor_free", &[buf])?;
        }

        // ── 14. Pipeline destroy ────────────────────────────────────────
        self.compile_call_by_name(builder, "nsl_pipeline_destroy", &[])?;

        Ok(())
    }

    fn compile_grad_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        grad: &nsl_ast::block::GradBlock,
    ) -> Result<(), CodegenError> {
        // M40b: Source-to-source AD strategy selection
        // When source AD is enabled, attempt to extract a static computation graph
        // (Wengert list) from the grad block body. If extraction succeeds, the
        // backward pass can be emitted directly as Cranelift IR (deferred: backward
        // codegen). If extraction fails (e.g., dynamic control flow), fall through
        // to the tape-based AD path below.
        if self.features.source_ad_enabled {
            // Future: WengertExtractor would analyze grad.body here.
            //   let extractor = crate::wengert::WengertExtractor::new();
            //   if let Ok(graph) = extractor.extract(&grad.body) {
            //       return self.compile_source_ad_backward(builder, state, grad, &graph);
            //   }
            // For now, always fall through to tape AD — backward Cranelift emission
            // is not yet wired.
        }
        // Existing tape-based AD path continues below.

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

        // 7b. Free the temporary lists (grad_tensor was extracted, still alive)
        self.compile_call_by_name(builder, "nsl_list_free", &[grads_list])?;
        self.compile_call_by_name(builder, "nsl_list_free", &[param_list])?;

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
            Some(QuantDtype::Awq4) => 2,
            Some(QuantDtype::Gptq4) => 3,
            Some(QuantDtype::Gptq8) => 4,
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
