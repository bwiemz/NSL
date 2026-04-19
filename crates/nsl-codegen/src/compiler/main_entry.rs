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
                    StmtKind::FnDef(_)
                        | StmtKind::StructDef(_)
                        | StmtKind::ModelDef(_)
                        | StmtKind::EnumDef(_)
                        | StmtKind::TraitDef(_)
                        | StmtKind::Import(_)
                        | StmtKind::FromImport(_)
                        | StmtKind::DatasetDef(_)
                        | StmtKind::TokenizerDef(_)
                        | StmtKind::KernelDef(_)
                        | StmtKind::DatatypeDef(_)
                )
            })
            .cloned()
            .collect();

        if top_level.is_empty() {
            return Ok(());
        }

        let mut sig = self.module.make_signature();
        sig.call_conv = self.call_conv;
        sig.params.push(AbiParam::new(cl_types::I32)); // argc
        sig.params.push(AbiParam::new(cl_types::I64)); // argv
        sig.returns.push(AbiParam::new(cl_types::I32));

        let main_id = self
            .module
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
            let init_id = self.registry.runtime_fns["nsl_args_init"].0;
            let init_ref = self.module.declare_func_in_func(init_id, builder.func);
            builder.ins().call(init_ref, &[argc_val, argv_val]);

            // B.3 Task 5.6: register synthesized fused PTX kernels with the
            // runtime registry so `NSL_WRGA_FUSED_CUDA=1` can launch them.
            self.emit_fused_ptx_registration(&mut builder)?;

            // M46: Set global deterministic mode flag + seed RNG at program start
            if self.compile_options.deterministic {
                let one = builder.ins().iconst(cl_types::I64, 1);
                self.compile_call_by_name(&mut builder, "nsl_set_deterministic", &[one])?;
                let seed = builder.ins().iconst(cl_types::I64, 42);
                self.compile_call_by_name(&mut builder, "nsl_rng_seed", &[seed])?;
            }

            // Register custom dtypes before user code runs
            self.emit_dtype_registration(&mut builder, stmts)?;

            // M36: Initialize GPU memory slab if the planner computed a layout
            let slab_ptr_var = if let Some(ref plan) = self.memory.slab_plan {
                if plan.total_bytes > 0 {
                    let total = builder.ins().iconst(cl_types::I64, plan.total_bytes as i64);
                    let slab_ptr =
                        self.compile_call_by_name(&mut builder, "nsl_gpu_slab_init", &[total])?;
                    // Store slab_ptr in a variable for slab_offset calls
                    let var = state.new_variable();
                    builder.declare_var(var, cl_types::I64);
                    builder.def_var(var, slab_ptr);
                    state.slab_ptr_var = Some(var);
                    Some(var)
                } else {
                    None
                }
            } else {
                None
            };

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

        self.module
            .define_function(main_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define main: {e}")))?;
        Ok(())
    }

    /// Compile a test-dispatch main() that reads `--run <name>` from argv and
    /// calls the corresponding @test function. Used by `nsl test`.
    pub fn compile_test_main(&mut self) -> Result<(), CodegenError> {
        let test_fns = self.registry.test_fns.clone();
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

        let main_id = self
            .module
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
            let init_id = self.registry.runtime_fns["nsl_args_init"].0;
            let init_ref = self.module.declare_func_in_func(init_id, builder.func);
            builder.ins().call(init_ref, &[argc_val, argv_val]);

            // Get args list
            let args_id = self.registry.runtime_fns["nsl_args"].0;
            let args_ref = self.module.declare_func_in_func(args_id, builder.func);
            let args_call = builder.ins().call(args_ref, &[]);
            let args_list = builder.inst_results(args_call)[0];

            // Get list length
            let len_id = self.registry.runtime_fns["nsl_list_len"].0;
            let len_ref = self.module.declare_func_in_func(len_id, builder.func);
            let len_call = builder.ins().call(len_ref, &[args_list]);
            let args_len = builder.inst_results(len_call)[0];

            // Check argc >= 3 (program, --run, test_name)
            let three = builder.ins().iconst(cl_types::I64, 3);
            let has_args = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::SignedGreaterThanOrEqual,
                args_len,
                three,
            );

            let dispatch_block = builder.create_block();
            let exit_one_block = builder.create_block();

            builder
                .ins()
                .brif(has_args, dispatch_block, &[], exit_one_block, &[]);

            // exit(1) block — no --run flag
            builder.switch_to_block(exit_one_block);
            builder.seal_block(exit_one_block);
            let one = builder.ins().iconst(cl_types::I32, 1);
            builder.ins().return_(&[one]);

            // Dispatch block: check argv[1] == "--run"
            builder.switch_to_block(dispatch_block);
            builder.seal_block(dispatch_block);

            let get_id = self.registry.runtime_fns["nsl_list_get"].0;
            let get_ref = self.module.declare_func_in_func(get_id, builder.func);
            let eq_id = self.registry.runtime_fns["nsl_str_eq"].0;
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
            let run_check = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::NotEqual,
                is_run,
                zero_i64,
            );

            let name_check_block = builder.create_block();
            let exit_one_block2 = builder.create_block();

            builder
                .ins()
                .brif(run_check, name_check_block, &[], exit_one_block2, &[]);

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
                let match_check = builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::NotEqual,
                    is_match,
                    zero2,
                );

                builder
                    .ins()
                    .brif(match_check, match_block, &[], next_block, &[]);

                // Match block: call the test function and return 0
                builder.switch_to_block(match_block);
                builder.seal_block(match_block);

                let (func_id, _) = &self.registry.functions[test_name];
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

        self.module
            .define_function(main_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define test main: {e}")))?;
        Ok(())
    }

    /// B.3 Task 5.6: emit `nsl_wrga_register_fused_ptx(handle, ptx_ptr,
    /// ptx_len, name_ptr, name_len)` calls at program start, one per unique
    /// fused-LoRA kernel in `self.fused_ptx_kernels`.  The handle assigned
    /// here MUST match the one embedded in each rewritten adapter Call —
    /// both sides sort by the same `(m, n, k, rank, target_sm)` tuple used
    /// by the AST rewrite's `fused_kernel_order`.
    ///
    /// No-op when the map is empty (non-WRGA compiles or non-sm_80 targets).
    fn emit_fused_ptx_registration(
        &mut self,
        builder: &mut FunctionBuilder,
    ) -> Result<(), CodegenError> {
        if self.fused_ptx_kernels.is_empty() && self.fused_gatedlora_ptx_kernels.is_empty() {
            return Ok(());
        }

        // Deterministic order — must match wrga_adapter_rewrite's
        // `fused_kernel_order` (sorted by the same key tuple).
        let mut order: Vec<(crate::wrga_fused_ptx::LoraKernelKey, String)> = self
            .fused_ptx_kernels
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        order.sort_by_key(|(k, _)| (k.m, k.n, k.k, k.rank, k.target_sm));

        // B.3.1 Task 5.0.c: GatedLoRA kernels get handles starting at
        // lora_count so there is no overlap with LoRA handles.  Must match
        // the offset baked into each GatedLoRA call site by
        // `synthesize_gatedlora_fused_call`.
        let lora_count = order.len();
        let mut gl_order: Vec<(crate::wrga_fused_ptx::LoraKernelKey, String)> = self
            .fused_gatedlora_ptx_kernels
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        gl_order.sort_by_key(|(k, _)| (k.m, k.n, k.k, k.rank, k.target_sm));

        let register_id = self.registry.runtime_fns["nsl_wrga_register_fused_ptx"].0;
        let register_ref = self
            .module
            .declare_func_in_func(register_id, builder.func);

        /// Emit one `nsl_wrga_register_fused_ptx` call.
        ///
        /// `tag` is `"lora"` or `"gatedlora"` — used in the Cranelift data
        /// symbol names so that LoRA and GatedLoRA symbols are distinct
        /// (Cranelift rejects duplicate data-section symbol names).
        fn emit_one_registration(
            module: &mut cranelift_object::ObjectModule,
            builder: &mut FunctionBuilder<'_>,
            register_ref: cranelift_codegen::ir::FuncRef,
            handle: usize,
            key: &crate::wrga_fused_ptx::LoraKernelKey,
            ptx: &str,
            kernel_name: &str,
            tag: &str,
        ) -> Result<(), CodegenError> {
            // PTX bytes + null terminator (cudarc's load_ptx needs a C string).
            let mut ptx_bytes = ptx.as_bytes().to_vec();
            ptx_bytes.push(0);
            let ptx_len = (ptx_bytes.len() - 1) as i64; // length excludes NUL
            let ptx_label = format!(
                "__nsl_wrga_{}_ptx_m{}n{}k{}r{}sm{}",
                tag, key.m, key.n, key.k, key.rank, key.target_sm
            );
            let ptx_data_id = module
                .declare_data(&ptx_label, cranelift_module::Linkage::Local, false, false)
                .map_err(|e| {
                    CodegenError::new(format!(
                        "failed to declare fused-PTX data '{ptx_label}': {e}"
                    ))
                })?;
            let mut ptx_desc = cranelift_module::DataDescription::new();
            ptx_desc.define(ptx_bytes.into_boxed_slice());
            module
                .define_data(ptx_data_id, &ptx_desc)
                .map_err(|e| {
                    CodegenError::new(format!(
                        "failed to define fused-PTX data '{ptx_label}': {e}"
                    ))
                })?;

            let name_len = kernel_name.len() as i64;
            let mut name_bytes = kernel_name.as_bytes().to_vec();
            name_bytes.push(0);
            let name_label = format!(
                "__nsl_wrga_{}_name_m{}n{}k{}r{}sm{}",
                tag, key.m, key.n, key.k, key.rank, key.target_sm
            );
            let name_data_id = module
                .declare_data(&name_label, cranelift_module::Linkage::Local, false, false)
                .map_err(|e| {
                    CodegenError::new(format!(
                        "failed to declare fused-PTX name data '{name_label}': {e}"
                    ))
                })?;
            let mut name_desc = cranelift_module::DataDescription::new();
            name_desc.define(name_bytes.into_boxed_slice());
            module
                .define_data(name_data_id, &name_desc)
                .map_err(|e| {
                    CodegenError::new(format!(
                        "failed to define fused-PTX name data '{name_label}': {e}"
                    ))
                })?;

            // Emit the call: nsl_wrga_register_fused_ptx(handle, ptx_ptr,
            //   ptx_len, name_ptr, name_len)
            let ptx_gv = module.declare_data_in_func(ptx_data_id, builder.func);
            let ptx_ptr = builder.ins().symbol_value(cl_types::I64, ptx_gv);
            let name_gv = module.declare_data_in_func(name_data_id, builder.func);
            let name_ptr = builder.ins().symbol_value(cl_types::I64, name_gv);
            let handle_val = builder.ins().iconst(cl_types::I64, handle as i64);
            let ptx_len_val = builder.ins().iconst(cl_types::I64, ptx_len);
            let name_len_val = builder.ins().iconst(cl_types::I64, name_len);
            builder.ins().call(
                register_ref,
                &[handle_val, ptx_ptr, ptx_len_val, name_ptr, name_len_val],
            );
            Ok(())
        }

        // ── LoRA kernels (handles 0..lora_count) ─────────────────────────
        for (handle, (key, ptx)) in order.into_iter().enumerate() {
            // Kernel entry symbol name — must match `.visible .entry` in PTX.
            let kernel_name = format!(
                "nsl_wrga_fused_lora_m{}n{}k{}r{}",
                key.m, key.n, key.k, key.rank
            );
            emit_one_registration(
                &mut self.module,
                builder,
                register_ref,
                handle,
                &key,
                &ptx,
                &kernel_name,
                "fused_lora",
            )?;
        }

        // ── GatedLoRA kernels (handles lora_count..lora_count+gl_count) ──
        for (gl_idx, (key, ptx)) in gl_order.into_iter().enumerate() {
            // Kernel entry symbol name — must match `.visible .entry` in PTX
            // emitted by `synthesize_fused_gatedlora_ptx`.
            let kernel_name = format!(
                "nsl_wrga_fused_gatedlora_m{}n{}k{}r{}",
                key.m, key.n, key.k, key.rank
            );
            emit_one_registration(
                &mut self.module,
                builder,
                register_ref,
                lora_count + gl_idx,
                &key,
                &ptx,
                &kernel_name,
                "fused_gatedlora",
            )?;
        }

        Ok(())
    }

    /// M39b: Apply vmap AST transformations to produce batched function variants.
    ///
    /// For each function with a @vmap config, clones the FnDef, runs the
    /// VmapTransformer, and returns full [`VmapResult`]s (batched FnDef +
    /// batched name + matmul rewrites).
    /// Must be called before compile_user_functions().
    pub fn apply_vmap_transforms(&self, module: &nsl_ast::Module) -> Vec<crate::vmap::VmapResult> {
        let mut results = Vec::new();

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
                    let mut transformer = crate::vmap::VmapTransformer::new(self.interner, &config);
                    match transformer.transform(fn_def) {
                        Ok(result) => {
                            results.push(result);
                        }
                        Err(e) => {
                            eprintln!("vmap transform error for '{}': {}", name, e);
                        }
                    }
                }
            }
        }

        results
    }

    /// M39b: Register batched function variants produced by [`apply_vmap_transforms`].
    ///
    /// For each `VmapResult`, declares the batched function in the Cranelift module
    /// (using the original function's signature), records the original→batched name
    /// mapping, and stores matmul rewrite targets.
    ///
    /// Body compilation is deferred — the functions are declared (have a `FuncId`)
    /// but their bodies are not yet compiled.  Call-site dispatch will fall back to
    /// the original function if the batched variant has no compiled body.
    pub fn register_batched_functions(&mut self, results: &[crate::vmap::VmapResult]) {
        for result in results {
            let batched_name = &result.batched_name;
            let original_name = batched_name.trim_end_matches("_batched");

            if let Some((_, sig)) = self.registry.functions.get(original_name) {
                let sig_clone = sig.clone();
                let mangled = super::mangle_name(&self.module_prefix, batched_name);
                match self
                    .module
                    .declare_function(&mangled, Linkage::Local, &sig_clone)
                {
                    Ok(func_id) => {
                        self.registry
                            .functions
                            .insert(batched_name.clone(), (func_id, sig_clone));
                        self.vmap
                            .batched_fn_names
                            .insert(original_name.to_string(), batched_name.clone());
                    }
                    Err(e) => {
                        eprintln!("[nsl] vmap: failed to declare '{}': {}", batched_name, e);
                    }
                }
            }

            // Store matmul rewrites for later use during call-site dispatch
            for (node_id, target) in &result.matmul_rewrites {
                self.vmap.matmul_rewrites.insert(*node_id, target.clone());
            }
        }
    }

    /// M39c: Compile batched function bodies using name_override to bypass
    /// the placeholder Symbol in fn_def.name.
    ///
    /// For each VmapResult whose batched function has been declared (present in
    /// `self.registry.functions`), compiles the batched FnDef body via
    /// `compile_fn_def_named`. Failures are non-fatal — the original function
    /// still works via dispatch fallback.
    pub fn compile_batched_functions(
        &mut self,
        results: &[crate::vmap::VmapResult],
    ) -> Result<(), CodegenError> {
        for result in results {
            let batched_name = &result.batched_name;
            // Only compile if the function was successfully declared during registration
            if self.registry.functions.contains_key(batched_name) {
                match self.compile_fn_def_named(&result.batched_fn, Some(batched_name)) {
                    Ok(()) => {}
                    Err(e) => {
                        eprintln!(
                            "[nsl] vmap: failed to compile batched variant '{}': {}",
                            batched_name, e
                        );
                        // Non-fatal — original function still works via dispatch fallback
                    }
                }
            }
        }
        Ok(())
    }
}
