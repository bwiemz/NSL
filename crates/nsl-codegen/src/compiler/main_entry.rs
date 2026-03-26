use cranelift_codegen::ir::{types as cl_types, AbiParam, Function, InstBuilder, UserFuncName};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{Linkage, Module};

use nsl_ast::stmt::StmtKind;

use super::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;

impl Compiler<'_> {
    // ── Pass 3: Compile top-level stmts into main() ─────────────────

    pub fn compile_main(&mut self, stmts: &[nsl_ast::stmt::Stmt]) -> Result<(), CodegenError> {
        let top_level: Vec<_> = stmts
            .iter()
            .filter(|s| {
                // Filter out decorated function definitions (compiled as top-level fns)
                if let StmtKind::Decorated { stmt, .. } = &s.kind {
                    if matches!(stmt.kind, StmtKind::FnDef(_)) {
                        return false;
                    }
                }
                !matches!(
                    s.kind,
                    StmtKind::FnDef(_) | StmtKind::StructDef(_) | StmtKind::ModelDef(_)
                        | StmtKind::EnumDef(_) | StmtKind::TraitDef(_)
                        | StmtKind::Import(_) | StmtKind::FromImport(_)
                        | StmtKind::DatasetDef(_) | StmtKind::TokenizerDef(_)
                        | StmtKind::KernelDef(_) | StmtKind::DatatypeDef(_)
                )
            })
            .cloned()
            .collect();

        if top_level.is_empty() {
            return Ok(());
        }

        let mut sig = self.module.make_signature();
        sig.call_conv = self.call_conv;
        sig.params.push(AbiParam::new(cl_types::I32));  // argc
        sig.params.push(AbiParam::new(cl_types::I64));  // argv
        sig.returns.push(AbiParam::new(cl_types::I32));

        let main_id = self.module
            .declare_function("main", Linkage::Export, &sig)
            .map_err(|e| CodegenError::new(format!("failed to declare main: {e}")))?;

        let mut ctx = Context::for_function(Function::with_name_signature(
            UserFuncName::user(0, self.next_func_index()),
            sig,
        ));
        let mut fn_builder_ctx = FunctionBuilderContext::new();

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);
            let mut state = FuncState::new();

            // Run use-count analysis on the main body for FBIP protection
            let main_block = nsl_ast::stmt::Block {
                stmts: top_level.clone(),
                span: nsl_ast::Span::DUMMY,
            };
            // Disable FBIP in debug_training mode to prevent in-place mutation
            // of tensors needed by backward pass
            if !self.compile_options.debug_training {
                state.use_counts = Some(crate::use_count::analyze_use_counts(&main_block));
            }

            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);
            state.current_block = Some(entry);

            // Initialize command-line args for args() support
            let argc_val = builder.block_params(entry)[0];
            let argv_val = builder.block_params(entry)[1];
            let init_id = self.runtime_fns["nsl_args_init"].0;
            let init_ref = self.module.declare_func_in_func(init_id, builder.func);
            builder.ins().call(init_ref, &[argc_val, argv_val]);

            // Register custom dtypes before user code runs
            self.emit_dtype_registration(&mut builder, stmts)?;

            // M36: Initialize GPU memory slab if the planner computed a layout
            let slab_ptr_var = if let Some(ref plan) = self.slab_plan {
                if plan.total_bytes > 0 {
                    let total = builder.ins().iconst(cl_types::I64, plan.total_bytes as i64);
                    let slab_ptr = self.compile_call_by_name(&mut builder, "nsl_gpu_slab_init", &[total])?;
                    // Store slab_ptr in a variable for slab_offset calls
                    let var = state.new_variable();
                    builder.declare_var(var, cl_types::I64);
                    builder.def_var(var, slab_ptr);
                    state.slab_ptr_var = Some(var);
                    Some(var)
                } else { None }
            } else { None };

            for stmt in &top_level {
                self.compile_stmt(&mut builder, &mut state, stmt)?;
            }

            let current = state.current_block.unwrap_or(entry);
            if !crate::types::is_block_filled(&builder, current) {
                // Stop and free any DataLoaders before implicit main() return
                self.teardown_dataloaders(&mut builder, &mut state);
                // M36: Destroy GPU memory slab before exit
                if slab_ptr_var.is_some() {
                    let _ = self.compile_call_by_name(&mut builder, "nsl_gpu_slab_destroy", &[]);
                }
                let zero = builder.ins().iconst(cl_types::I32, 0);
                builder.ins().return_(&[zero]);
            }

            builder.finalize();
        }

        if self.dump_ir {
            eprintln!("--- IR: main ---\n{}", ctx.func.display());
        }

        self.module.define_function(main_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define main: {e}")))?;
        Ok(())
    }

    /// Compile a test-dispatch main() that reads `--run <name>` from argv and
    /// calls the corresponding @test function. Used by `nsl test`.
    pub fn compile_test_main(&mut self) -> Result<(), CodegenError> {
        let test_fns = self.test_fns.clone();
        if test_fns.is_empty() {
            return Err(CodegenError::new("no @test functions found".to_string()));
        }

        // Ensure test function name strings are in the string pool
        let run_flag = "--run".to_string();
        self.intern_string(&run_flag)?;
        for name in &test_fns {
            self.intern_string(name)?;
        }

        let mut sig = self.module.make_signature();
        sig.call_conv = self.call_conv;
        sig.params.push(AbiParam::new(cl_types::I32)); // argc
        sig.params.push(AbiParam::new(cl_types::I64)); // argv
        sig.returns.push(AbiParam::new(cl_types::I32));

        let main_id = self.module
            .declare_function("main", Linkage::Export, &sig)
            .map_err(|e| CodegenError::new(format!("failed to declare main: {e}")))?;

        let mut ctx = Context::for_function(Function::with_name_signature(
            UserFuncName::user(0, self.next_func_index()),
            sig,
        ));
        let mut fn_builder_ctx = FunctionBuilderContext::new();

        {
            let mut builder = FunctionBuilder::new(&mut ctx.func, &mut fn_builder_ctx);

            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);

            // Initialize args
            let argc_val = builder.block_params(entry)[0];
            let argv_val = builder.block_params(entry)[1];
            let init_id = self.runtime_fns["nsl_args_init"].0;
            let init_ref = self.module.declare_func_in_func(init_id, builder.func);
            builder.ins().call(init_ref, &[argc_val, argv_val]);

            // Get args list
            let args_id = self.runtime_fns["nsl_args"].0;
            let args_ref = self.module.declare_func_in_func(args_id, builder.func);
            let args_call = builder.ins().call(args_ref, &[]);
            let args_list = builder.inst_results(args_call)[0];

            // Get list length
            let len_id = self.runtime_fns["nsl_list_len"].0;
            let len_ref = self.module.declare_func_in_func(len_id, builder.func);
            let len_call = builder.ins().call(len_ref, &[args_list]);
            let args_len = builder.inst_results(len_call)[0];

            // Check argc >= 3 (program, --run, test_name)
            let three = builder.ins().iconst(cl_types::I64, 3);
            let has_args = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThanOrEqual, args_len, three);

            let dispatch_block = builder.create_block();
            let exit_one_block = builder.create_block();

            builder.ins().brif(has_args, dispatch_block, &[], exit_one_block, &[]);

            // exit(1) block — no --run flag
            builder.switch_to_block(exit_one_block);
            builder.seal_block(exit_one_block);
            let one = builder.ins().iconst(cl_types::I32, 1);
            builder.ins().return_(&[one]);

            // Dispatch block: check argv[1] == "--run"
            builder.switch_to_block(dispatch_block);
            builder.seal_block(dispatch_block);

            let get_id = self.runtime_fns["nsl_list_get"].0;
            let get_ref = self.module.declare_func_in_func(get_id, builder.func);
            let eq_id = self.runtime_fns["nsl_str_eq"].0;
            let eq_ref = self.module.declare_func_in_func(eq_id, builder.func);

            // argv[1]
            let idx1 = builder.ins().iconst(cl_types::I64, 1);
            let get_call1 = builder.ins().call(get_ref, &[args_list, idx1]);
            let arg1 = builder.inst_results(get_call1)[0];

            // "--run" string constant
            let run_data_id = self.string_pool[&run_flag];
            let run_gv = self.module.declare_data_in_func(run_data_id, builder.func);
            let run_ptr = builder.ins().symbol_value(cl_types::I64, run_gv);

            let eq_call = builder.ins().call(eq_ref, &[arg1, run_ptr]);
            let is_run = builder.inst_results(eq_call)[0];
            let zero_i64 = builder.ins().iconst(cl_types::I64, 0);
            let run_check = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, is_run, zero_i64);

            let name_check_block = builder.create_block();
            let exit_one_block2 = builder.create_block();

            builder.ins().brif(run_check, name_check_block, &[], exit_one_block2, &[]);

            builder.switch_to_block(exit_one_block2);
            builder.seal_block(exit_one_block2);
            let one2 = builder.ins().iconst(cl_types::I32, 1);
            builder.ins().return_(&[one2]);

            // Name check block: get argv[2] and match against test names
            builder.switch_to_block(name_check_block);
            builder.seal_block(name_check_block);

            let idx2 = builder.ins().iconst(cl_types::I64, 2);
            let get_call2 = builder.ins().call(get_ref, &[args_list, idx2]);
            let arg2 = builder.inst_results(get_call2)[0];

            // Build chain: if arg2 == "test_name" { call test_name(); return 0; }
            let exit_fail_block = builder.create_block();

            for (i, test_name) in test_fns.iter().enumerate() {
                let match_block = builder.create_block();
                let next_block = if i + 1 < test_fns.len() {
                    builder.create_block()
                } else {
                    exit_fail_block
                };

                // Compare arg2 with test_name string constant
                let name_data_id = self.string_pool[test_name];
                let name_gv = self.module.declare_data_in_func(name_data_id, builder.func);
                let name_ptr = builder.ins().symbol_value(cl_types::I64, name_gv);
                let eq_ref2 = self.module.declare_func_in_func(eq_id, builder.func);
                let eq_call2 = builder.ins().call(eq_ref2, &[arg2, name_ptr]);
                let is_match = builder.inst_results(eq_call2)[0];
                let zero2 = builder.ins().iconst(cl_types::I64, 0);
                let match_check = builder.ins().icmp(cranelift_codegen::ir::condcodes::IntCC::NotEqual, is_match, zero2);

                builder.ins().brif(match_check, match_block, &[], next_block, &[]);

                // Match block: call the test function and return 0
                builder.switch_to_block(match_block);
                builder.seal_block(match_block);

                let (func_id, _) = &self.functions[test_name];
                let fn_ref = self.module.declare_func_in_func(*func_id, builder.func);
                builder.ins().call(fn_ref, &[]);
                let zero_ret = builder.ins().iconst(cl_types::I32, 0);
                builder.ins().return_(&[zero_ret]);

                if i + 1 < test_fns.len() {
                    builder.switch_to_block(next_block);
                    builder.seal_block(next_block);
                }
            }

            // No match — exit with code 2
            builder.switch_to_block(exit_fail_block);
            builder.seal_block(exit_fail_block);
            let two = builder.ins().iconst(cl_types::I32, 2);
            builder.ins().return_(&[two]);

            builder.finalize();
        }

        if self.dump_ir {
            eprintln!("--- IR: main (test dispatch) ---\n{}", ctx.func.display());
        }

        self.module.define_function(main_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define test main: {e}")))?;
        Ok(())
    }

    /// M39b: Apply vmap AST transformations to produce batched function variants.
    ///
    /// For each function with a @vmap config, clones the FnDef, runs the
    /// VmapTransformer, and returns the batched variants.
    /// Must be called before compile_user_functions().
    pub fn apply_vmap_transforms(&self, module: &nsl_ast::Module) -> Vec<nsl_ast::decl::FnDef> {
        let mut batched_fns = Vec::new();

        for stmt in &module.stmts {
            let fn_def = match &stmt.kind {
                StmtKind::FnDef(f) => Some(f),
                StmtKind::Decorated { stmt, .. } => {
                    if let StmtKind::FnDef(f) = &stmt.kind {
                        Some(f)
                    } else {
                        None
                    }
                }
                _ => None,
            };

            if let Some(fn_def) = fn_def {
                let name = self.resolve_sym(fn_def.name).to_string();
                if let Some(config) = self.features.vmap_configs.get(&name).cloned() {
                    let mut transformer = crate::vmap::VmapTransformer::new(
                        self.interner,
                        &config,
                    );
                    match transformer.transform(fn_def) {
                        Ok(result) => {
                            batched_fns.push(result.batched_fn);
                        }
                        Err(e) => {
                            eprintln!("vmap transform error for '{}': {}", name, e);
                        }
                    }
                }
            }
        }

        batched_fns
    }
}
