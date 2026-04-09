/// Standalone utilities for M24 standalone export.
///
/// This module provides helpers used by the build pipeline (nsl-cli) to create
/// separate object files that carry weight data independently of the compiled
/// NSL program object.  Also contains `compile_standalone_main()` which generates
/// the entry-point main() for standalone binaries (weight init + arg parser).
use cranelift_codegen::ir::{
    types as cl_types, AbiParam, Function, InstBuilder, MemFlags, UserFuncName,
};
use cranelift_codegen::Context;
use cranelift_frontend::{FunctionBuilder, FunctionBuilderContext};
use cranelift_module::{Linkage, Module};

/// Create a Cranelift object file containing .nslweights data in .rodata.
/// Exports two symbols: __nsl_weights_data (the bytes) and __nsl_weights_size (u64 length).
pub fn create_weight_object(nslweights_data: &[u8]) -> Result<Vec<u8>, String> {
    // 1. Create ISA from native target
    let flag_builder = cranelift_codegen::settings::builder();
    let isa = cranelift_native::builder()
        .map_err(|e| format!("cranelift native builder: {}", e))?
        .finish(cranelift_codegen::settings::Flags::new(flag_builder))
        .map_err(|e| format!("cranelift ISA: {}", e))?;

    // 2. Create ObjectModule
    let builder = cranelift_object::ObjectBuilder::new(
        isa,
        "nsl_weights",
        cranelift_module::default_libcall_names(),
    )
    .map_err(|e| format!("object builder: {}", e))?;
    let mut module = cranelift_object::ObjectModule::new(builder);

    // 3. Embed weight data
    let data_id = module
        .declare_data(
            "__nsl_weights_data",
            cranelift_module::Linkage::Export,
            false,
            false,
        )
        .map_err(|e| format!("declare weights data: {}", e))?;
    let mut data_desc = cranelift_module::DataDescription::new();
    data_desc.define(nslweights_data.to_vec().into_boxed_slice());
    module
        .define_data(data_id, &data_desc)
        .map_err(|e| format!("define weights data: {}", e))?;

    // 4. Embed size as u64
    let size_id = module
        .declare_data(
            "__nsl_weights_size",
            cranelift_module::Linkage::Export,
            false,
            false,
        )
        .map_err(|e| format!("declare weights size: {}", e))?;
    let mut size_desc = cranelift_module::DataDescription::new();
    size_desc.define(Box::new((nslweights_data.len() as u64).to_le_bytes()));
    module
        .define_data(size_id, &size_desc)
        .map_err(|e| format!("define weights size: {}", e))?;

    // 5. Emit object
    let product = module.finish();
    product
        .emit()
        .map_err(|e| format!("emit weight object: {}", e))
}

// ── Standalone main() codegen ──────────────────────────────────────────

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use nsl_ast::stmt::{Stmt, StmtKind};

impl Compiler<'_> {
    /// Generate a standalone main() for exported binaries.
    ///
    /// This is similar to `compile_main()` but additionally:
    /// - Initialises the standalone arg parser (`nsl_standalone_args_init`)
    /// - Initialises the weight provider (embedded data or sidecar file)
    /// - Calls `nsl_standalone_args_finish()` before returning
    pub fn compile_standalone_main(&mut self, stmts: &[Stmt]) -> Result<(), CodegenError> {
        // ── Filter top-level statements (same rules as compile_main) ───
        let top_level: Vec<_> = stmts
            .iter()
            .filter(|s| {
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

        // ── main(argc: i32, argv: i64) -> i32 ─────────────────────────
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

            let entry = builder.create_block();
            builder.append_block_params_for_function_params(entry);
            builder.switch_to_block(entry);
            builder.seal_block(entry);
            state.current_block = Some(entry);

            let argc_val = builder.block_params(entry)[0]; // I32
            let argv_val = builder.block_params(entry)[1]; // I64

            // 1. Initialise standard args: nsl_args_init(argc, argv)
            let init_id = self.registry.runtime_fns["nsl_args_init"].0;
            let init_ref = self.module.declare_func_in_func(init_id, builder.func);
            builder.ins().call(init_ref, &[argc_val, argv_val]);

            // 2. Initialise standalone arg parser: nsl_standalone_args_init(argc_i64, argv)
            let argc_i64 = builder.ins().sextend(cl_types::I64, argc_val);
            let sa_init_id = self.registry.runtime_fns["nsl_standalone_args_init"].0;
            let sa_init_ref = self.module.declare_func_in_func(sa_init_id, builder.func);
            builder.ins().call(sa_init_ref, &[argc_i64, argv_val]);

            // 3. Initialise WeightProvider
            let config = self
                .standalone_config
                .as_ref()
                .expect("standalone_config must be set for compile_standalone_main");

            if config.embedded {
                // Imported data symbols provided by the linker (from the weight object)
                let data_gv_id = self
                    .module
                    .declare_data("__nsl_weights_data", Linkage::Import, false, false)
                    .map_err(|e| {
                        CodegenError::new(format!("failed to declare __nsl_weights_data: {e}"))
                    })?;
                let size_gv_id = self
                    .module
                    .declare_data("__nsl_weights_size", Linkage::Import, false, false)
                    .map_err(|e| {
                        CodegenError::new(format!("failed to declare __nsl_weights_size: {e}"))
                    })?;

                let data_gv = self.module.declare_data_in_func(data_gv_id, builder.func);
                let size_gv = self.module.declare_data_in_func(size_gv_id, builder.func);

                let data_ptr = builder.ins().symbol_value(cl_types::I64, data_gv);
                let size_ptr = builder.ins().symbol_value(cl_types::I64, size_gv);
                let size_val = builder
                    .ins()
                    .load(cl_types::I64, MemFlags::trusted(), size_ptr, 0);

                let embed_init_id = self.registry.runtime_fns["nsl_standalone_init_embedded"].0;
                let embed_init_ref = self
                    .module
                    .declare_func_in_func(embed_init_id, builder.func);
                builder.ins().call(embed_init_ref, &[data_ptr, size_val]);
            } else {
                // Sidecar mode: pass the path string to the runtime
                let sidecar_path = config.sidecar_path.clone();
                let path_len = sidecar_path.len() as i64;
                let path_data_id = self.intern_string(&sidecar_path)?;
                let path_gv = self.module.declare_data_in_func(path_data_id, builder.func);
                let path_ptr = builder.ins().symbol_value(cl_types::I64, path_gv);
                let path_len_val = builder.ins().iconst(cl_types::I64, path_len);

                let sidecar_init_id = self.registry.runtime_fns["nsl_standalone_init_sidecar"].0;
                let sidecar_init_ref = self
                    .module
                    .declare_func_in_func(sidecar_init_id, builder.func);
                builder
                    .ins()
                    .call(sidecar_init_ref, &[path_ptr, path_len_val]);
            }

            // 4. Register custom dtypes before user code runs
            self.emit_dtype_registration(&mut builder, stmts)?;

            // 5. Execute user's top-level code
            for stmt in &top_level {
                self.compile_stmt(&mut builder, &mut state, stmt)?;
            }

            // 6. Teardown: finish arg parser, dataloaders, and return 0
            let current = state.current_block.unwrap_or(entry);
            if !crate::types::is_block_filled(&builder, current) {
                // Finish standalone arg parser
                let finish_id = self.registry.runtime_fns["nsl_standalone_args_finish"].0;
                let finish_ref = self.module.declare_func_in_func(finish_id, builder.func);
                builder.ins().call(finish_ref, &[]);

                // Stop and free any DataLoaders before implicit main() return
                self.teardown_dataloaders(&mut builder, &mut state);

                let zero = builder.ins().iconst(cl_types::I32, 0);
                builder.ins().return_(&[zero]);
            }

            builder.finalize();
        }

        if self.dump_ir {
            eprintln!("--- IR: main (standalone) ---\n{}", ctx.func.display());
        }

        self.module
            .define_function(main_id, &mut ctx)
            .map_err(|e| CodegenError::new(format!("failed to define main: {e}")))?;
        Ok(())
    }
}
