use cranelift_codegen::ir::{types as cl_types, Function, InstBuilder, MemFlags, UserFuncName};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::Module;

use nsl_ast::decl::ModelMember;
use nsl_ast::stmt::{Stmt, StmtKind};

use super::{Compiler, PendingLambda};
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::types::pointer_type;

impl Compiler<'_> {
    // ── Pass 2: Compile function bodies ─────────────────────────────

    pub fn compile_user_functions(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        let fn_defs: Vec<_> = stmts
            .iter()
            .filter_map(|s| match &s.kind {
                StmtKind::FnDef(fn_def) => Some(fn_def.clone()),
                StmtKind::Decorated { stmt, .. } => {
                    if let StmtKind::FnDef(fn_def) = &stmt.kind { Some(fn_def.clone()) } else { None }
                }
                _ => None,
            })
            .collect();

        for fn_def in &fn_defs {
            self.compile_fn_def(fn_def)?;
        }

        #[allow(clippy::type_complexity)]
        let layouts: Vec<(String, Vec<(cl_types::Type, usize)>, usize)> = self
            .struct_layouts.iter()
            .filter(|(name, _)| !self.model_methods.contains_key(*name) && !self.imported_model_names.contains(*name))
            .map(|(name, layout)| {
                let fields: Vec<_> = layout.fields.iter().map(|f| (f.cl_type, f.offset)).collect();
                (name.clone(), fields, layout.total_size)
            })
            .collect();

        for (name, fields, total_size) in &layouts {
            self.compile_struct_constructor(name, fields, *total_size)?;
        }

        // Compile model constructors and methods
        let model_defs: Vec<_> = stmts
            .iter()
            .filter_map(|s| {
                if let StmtKind::ModelDef(md) = &s.kind { Some(md.clone()) } else { None }
            })
            .collect();
        for md in &model_defs {
            self.compile_model_constructor(md)?;
        }
        self.compile_model_methods(stmts)?;

        self.compile_pending_lambdas()?;
        Ok(())
    }

    /// Compile all pending lambda bodies. Drains in a loop to handle nested lambdas.
    pub fn compile_pending_lambdas(&mut self) -> Result<(), CodegenError> {
        while !self.pending_lambdas.is_empty() {
            let lambdas: Vec<PendingLambda> = self.pending_lambdas.drain(..).collect();
            for lambda in lambdas {
                self.compile_lambda_body(&lambda)?;
            }
        }
        Ok(())
    }

    fn compile_lambda_body(&mut self, lambda: &PendingLambda) -> Result<(), CodegenError> {
        let mut ctx = Context::for_function(Function::with_name_signature(
            UserFuncName::user(0, self.next_func_index()),
            lambda.sig.clone(),
        ));
        let mut fn_builder_ctx = FunctionBuilderContext::new();

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
            let mut state = FuncState::new();

            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);
            state.current_block = Some(entry);

            // Bind normal parameters
            for (i, (sym, cl_type)) in lambda.params.iter().enumerate() {
                let param_val = builder.block_params(entry)[i];
                let var = state.new_variable();
                builder.declare_var(var, *cl_type);
                builder.def_var(var, param_val);
                state.variables.insert(*sym, (var, *cl_type));
            }

            // Bind captured variables (come after normal params in the signature)
            let capture_offset = lambda.params.len();
            for (i, (sym, cl_type)) in lambda.captures.iter().enumerate() {
                let param_val = builder.block_params(entry)[capture_offset + i];
                let var = state.new_variable();
                builder.declare_var(var, *cl_type);
                builder.def_var(var, param_val);
                state.variables.insert(*sym, (var, *cl_type));
            }

            // Compile body expression
            let result = self.compile_expr(&mut builder, &mut state, &lambda.body)?;
            let current = state.current_block.unwrap_or(entry);
            if !crate::types::is_block_filled(&builder, current) {
                builder.ins().return_(&[result]);
            }
            builder.finalize();
        }

        if self.dump_ir {
            eprintln!("--- IR: lambda '{}' ---\n{}", lambda.name, ctx.func.display());
        }

        self.module
            .define_function(lambda.func_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define lambda '{}': {e}", lambda.name)))?;
        Ok(())
    }

    fn compile_struct_constructor(
        &mut self,
        name: &str,
        fields: &[(cl_types::Type, usize)],
        total_size: usize,
    ) -> Result<(), CodegenError> {
        let (func_id, sig) = self.functions[name].clone();
        let mut ctx = Context::for_function(Function::with_name_signature(
            UserFuncName::user(0, self.next_func_index()),
            sig.clone(),
        ));
        let mut fn_builder_ctx = FunctionBuilderContext::new();

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            let alloc_id = self.runtime_fns["nsl_alloc"].0;
            let alloc_ref = self.module.declare_func_in_func(alloc_id, builder.func);
            let size_val = builder.ins().iconst(cl_types::I64, total_size as i64);
            let call = builder.ins().call(alloc_ref, &[size_val]);
            let ptr = builder.inst_results(call)[0];

            for (i, (_cl_type, offset)) in fields.iter().enumerate() {
                let param_val = builder.block_params(entry)[i];
                builder.ins().store(MemFlags::trusted(), param_val, ptr, *offset as i32);
            }

            builder.ins().return_(&[ptr]);
            builder.finalize();
        }

        if self.dump_ir {
            eprintln!("--- IR: struct ctor '{name}' ---\n{}", ctx.func.display());
        }

        self.module.define_function(func_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define struct ctor '{name}': {e}")))?;
        Ok(())
    }

    fn compile_model_constructor(
        &mut self,
        md: &nsl_ast::decl::ModelDef,
    ) -> Result<(), CodegenError> {
        let model_name = self.resolve_sym(md.name).to_string();
        let (func_id, sig) = self.functions[&model_name].clone();

        let mut ctx = Context::for_function(Function::with_name_signature(
            UserFuncName::user(0, self.next_func_index()),
            sig.clone(),
        ));
        let mut fn_builder_ctx = FunctionBuilderContext::new();

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
            let mut state = FuncState::new();

            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);
            state.current_block = Some(entry);

            // Bind constructor params as variables
            for (i, param) in md.params.iter().enumerate() {
                let param_val = builder.block_params(entry)[i];
                let cl_type = if i < sig.params.len() {
                    sig.params[i].value_type
                } else {
                    cl_types::I64
                };
                let var = state.new_variable();
                builder.declare_var(var, cl_type);
                builder.def_var(var, param_val);
                state.variables.insert(param.name, (var, cl_type));
            }

            // Allocate model memory
            let layout = self.struct_layouts.get(&model_name).cloned();
            let total_size = layout.as_ref().map(|l| l.total_size).unwrap_or(0);
            let alloc_size = total_size.max(8) as i64;

            let alloc_id = self.runtime_fns["nsl_alloc"].0;
            let alloc_ref = self.module.declare_func_in_func(alloc_id, builder.func);
            let size_val = builder.ins().iconst(cl_types::I64, alloc_size);
            let call = builder.ins().call(alloc_ref, &[size_val]);
            let ptr = builder.inst_results(call)[0];

            // Initialize fields
            if let Some(ref layout) = layout {
                let field_info: Vec<_> = layout.fields.iter().map(|f| (f.name.clone(), f.cl_type, f.offset)).collect();
                let members_clone: Vec<_> = md.members.clone();
                for member in &members_clone {
                    if let ModelMember::LayerDecl { name, init, type_ann, .. } = member {
                        let field_name = self.resolve_sym(*name).to_string();

                        // Check if this is a FixedArray field → generate init loop
                        if let nsl_ast::types::TypeExprKind::FixedArray { size, .. } = &type_ann.kind {
                            let arr_size = *size;
                            if let Some((_fname, _cl_type, field_offset)) = field_info.iter().find(|(n, _, _)| *n == field_name) {
                                let field_offset = *field_offset;
                                if let Some(init_expr) = init {
                                    let init_expr_clone = init_expr.clone();
                                    // Generate loop: for i in 0..arr_size, compile init, store at ptr+field_offset+i*8
                                    let counter_var = state.new_variable();
                                    builder.declare_var(counter_var, cl_types::I64);
                                    let zero = builder.ins().iconst(cl_types::I64, 0);
                                    builder.def_var(counter_var, zero);

                                    let loop_header = builder.create_block();
                                    let loop_body = builder.create_block();
                                    let loop_exit = builder.create_block();

                                    builder.ins().jump(loop_header, &[]);

                                    builder.switch_to_block(loop_header);
                                    state.current_block = Some(loop_header);
                                    let counter = builder.use_var(counter_var);
                                    let limit = builder.ins().iconst(cl_types::I64, arr_size);
                                    let cond = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::SignedLessThan, counter, limit);
                                    builder.ins().brif(cond, loop_body, &[], loop_exit, &[]);

                                    builder.switch_to_block(loop_body);
                                    builder.seal_block(loop_body);
                                    state.current_block = Some(loop_body);

                                    let init_val = self.compile_expr(&mut builder, &mut state, &init_expr_clone)?;
                                    let counter = builder.use_var(counter_var);
                                    let eight = builder.ins().iconst(cl_types::I64, 8);
                                    let elem_byte_offset = builder.ins().imul(counter, eight);
                                    let base_off = builder.ins().iconst(cl_types::I64, field_offset as i64);
                                    let total_off = builder.ins().iadd(base_off, elem_byte_offset);
                                    let addr = builder.ins().iadd(ptr, total_off);
                                    builder.ins().store(MemFlags::trusted(), init_val, addr, 0);

                                    let next_counter = builder.use_var(counter_var);
                                    let one = builder.ins().iconst(cl_types::I64, 1);
                                    let next = builder.ins().iadd(next_counter, one);
                                    builder.def_var(counter_var, next);
                                    builder.ins().jump(loop_header, &[]);

                                    builder.seal_block(loop_header);
                                    builder.switch_to_block(loop_exit);
                                    builder.seal_block(loop_exit);
                                    state.current_block = Some(loop_exit);
                                }
                            }
                            continue;
                        }

                        if let Some((_fname, cl_type, offset)) = field_info.iter().find(|(n, _, _)| *n == field_name) {
                            if let Some(init_expr) = init {
                                let val = self.compile_expr(&mut builder, &mut state, init_expr)?;
                                builder.ins().store(MemFlags::trusted(), val, ptr, *offset as i32);
                            } else {
                                // Store zero for uninitialized fields
                                let zero = if cl_type.is_float() {
                                    builder.ins().f64const(0.0)
                                } else {
                                    builder.ins().iconst(*cl_type, 0)
                                };
                                builder.ins().store(MemFlags::trusted(), zero, ptr, *offset as i32);
                            }
                        }
                    }
                }
            }

            // M25: Initialize paged KV cache if model has @paged_kv
            if let Some(&(num_blocks, block_size, num_heads, head_dim, num_layers)) = self.paged_kv_configs.get(&model_name) {
                let init_id = self.runtime_fns["nsl_kv_cache_init"].0;
                let init_ref = self.module.declare_func_in_func(init_id, builder.func);
                let nb = builder.ins().iconst(cl_types::I64, num_blocks);
                let bs = builder.ins().iconst(cl_types::I64, block_size);
                let nh = builder.ins().iconst(cl_types::I64, num_heads);
                let hd = builder.ins().iconst(cl_types::I64, head_dim);
                let nl = builder.ins().iconst(cl_types::I64, num_layers);
                let call = builder.ins().call(init_ref, &[nb, bs, nh, hd, nl]);
                // Store the handle — for now just discard it (the handle is returned by init)
                let _handle = builder.inst_results(call)[0];
            }

            builder.ins().return_(&[ptr]);
            builder.finalize();
        }

        if self.dump_ir {
            eprintln!("--- IR: model ctor '{model_name}' ---\n{}", ctx.func.display());
        }

        self.module.define_function(func_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define model ctor '{model_name}': {e}")))?;
        Ok(())
    }

    fn compile_model_methods(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        let model_defs: Vec<_> = stmts
            .iter()
            .filter_map(|s| {
                if let StmtKind::ModelDef(md) = &s.kind { Some(md.clone()) } else { None }
            })
            .collect();

        for md in &model_defs {
            let model_name = self.resolve_sym(md.name).to_string();
            for member in &md.members {
                if let ModelMember::Method(fn_def, _decos) = member {
                    let method_name = self.resolve_sym(fn_def.name).to_string();
                    let mangled = format!("__nsl_model_{model_name}_{method_name}");
                    let (func_id, sig) = self.functions.get(&mangled)
                        .ok_or_else(|| CodegenError::new(format!("model method '{}' not registered", mangled)))?
                        .clone();

                    let mut ctx = Context::for_function(Function::with_name_signature(
                        UserFuncName::user(0, self.next_func_index()),
                        sig.clone(),
                    ));
                    let mut fn_builder_ctx = FunctionBuilderContext::new();

                    {
                        let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
                        let mut state = FuncState::new();

                        let entry = builder.create_block();
                        builder.append_block_params_for_function_params(entry);
                        builder.switch_to_block(entry);
                        builder.seal_block(entry);
                        state.current_block = Some(entry);

                        // First Cranelift param is self (pointer)
                        let self_val = builder.block_params(entry)[0];
                        // Find the "self" symbol from the AST params
                        let self_sym = fn_def.params.iter()
                            .find(|p| self.resolve_sym(p.name) == "self")
                            .map(|p| p.name)
                            .unwrap_or_else(|| fn_def.params[0].name);
                        let self_var = state.new_variable();
                        builder.declare_var(self_var, pointer_type());
                        builder.def_var(self_var, self_val);
                        state.variables.insert(self_sym, (self_var, pointer_type()));

                        // Bind remaining params (skip "self" in AST)
                        let mut cl_param_idx = 1usize;
                        for param in &fn_def.params {
                            let pname = self.resolve_sym(param.name).to_string();
                            if pname == "self" {
                                continue;
                            }
                            let param_val = builder.block_params(entry)[cl_param_idx];
                            let cl_type = if cl_param_idx < sig.params.len() {
                                sig.params[cl_param_idx].value_type
                            } else {
                                cl_types::I64
                            };
                            let var = state.new_variable();
                            builder.declare_var(var, cl_type);
                            builder.def_var(var, param_val);
                            state.variables.insert(param.name, (var, cl_type));
                            cl_param_idx += 1;
                        }

                        // Compile method body
                        for stmt in &fn_def.body.stmts {
                            self.compile_stmt(&mut builder, &mut state, stmt)?;
                        }

                        // Add implicit return if body doesn't end with one
                        let current = state.current_block.unwrap_or(entry);
                        if !crate::types::is_block_filled(&builder, current) {
                            if sig.returns.is_empty() {
                                builder.ins().return_(&[]);
                            } else {
                                let ret_type = sig.returns[0].value_type;
                                let zero = if ret_type == cl_types::F64 {
                                    builder.ins().f64const(0.0)
                                } else if ret_type == cl_types::F32 {
                                    builder.ins().f32const(0.0)
                                } else {
                                    builder.ins().iconst(ret_type, 0)
                                };
                                builder.ins().return_(&[zero]);
                            }
                        }

                        builder.finalize();
                    }

                    if self.dump_ir {
                        eprintln!("--- IR: model method '{mangled}' ---\n{}", ctx.func.display());
                    }

                    self.module.define_function(func_id, &mut ctx)
                        .map_err(|e| CodegenError::new(format!("failed to define model method '{mangled}': {e}")))?;
                }
            }
        }
        Ok(())
    }
}
