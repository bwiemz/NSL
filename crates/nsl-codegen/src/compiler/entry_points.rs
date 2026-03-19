use std::collections::{HashMap, HashSet};

use cranelift_codegen::ir::Signature;
use cranelift_module::Linkage;

use nsl_lexer::Interner;
use nsl_semantic::checker::TypeMap;

use super::{Compiler, StandaloneConfig};
use crate::error::CodegenError;

/// Main entry point (single-file, backward compatible).
pub fn compile(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> Result<Vec<u8>, CodegenError> {
    let mut compiler = Compiler::new(interner, type_map, options)?;

    // M52: Load weights if --weights was provided
    if let Some(ref weight_path) = options.weight_file {
        match crate::weight_aware::WeightMap::load(weight_path) {
            Ok(mut wmap) => {
                let integrity = crate::weight_aware::WeightIntegrity::new(*wmap.hash());

                // Run sparsity analysis on all weight entries
                if options.weight_config.sparse_codegen || options.weight_config.dead_weight_elim {
                    let config = &options.weight_config;
                    let names: Vec<String> = wmap.names().map(|s| s.to_string()).collect();
                    for name in &names {
                        if let Some(entry) = wmap.get_mut(name) {
                            entry.analyze_sparsity(config);
                        }
                    }
                }

                // Run dead weight elimination
                if options.weight_config.dead_weight_elim {
                    let eliminator = crate::weight_aware::DeadWeightEliminator::new(&options.weight_config);
                    let names: Vec<String> = wmap.names().map(|s| s.to_string()).collect();
                    for name in &names {
                        if let Some(entry) = wmap.get_mut(name) {
                            eliminator.eliminate(entry);
                        }
                    }
                }

                // Weight analysis report (when --weight-analysis is set)
                if options.weight_analysis {
                    crate::weight_aware::print_weight_analysis_report(&wmap, &options.weight_config);
                }

                eprintln!(
                    "[nsl] loaded {} weights from {} (SHA-256: {})",
                    wmap.len(),
                    wmap.source_path(),
                    integrity.hash_hex
                );

                compiler.weight_integrity = Some(integrity);
                compiler.weight_map = Some(wmap);
            }
            Err(e) => {
                return Err(crate::error::CodegenError::new(
                    format!("failed to load weights: {}", e)
                ));
            }
        }
    }

    compiler.dump_ir = dump_ir;
    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;
    compiler.declare_runtime_functions()?;
    compiler.declare_user_functions(&ast.stmts)?;
    // M39b: Apply vmap AST transforms to generate batched function variants
    let _batched_fns = compiler.apply_vmap_transforms(ast);
    // TODO: register batched_fns for compilation when call-site dispatch is wired
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    compiler.compile_flash_attention_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    compiler.compile_main(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;

    // M36: Memory planner integration
    // The memory planner requires TensorAlloc records populated during codegen.
    // TODO: Integrate planner.record_alloc() calls into tensor creation codegen,
    // then invoke plan_slab() here. For now, report that the flags are recognized.
    if options.vram_budget.is_some() {
        eprintln!("[nsl] --vram-budget set to {} bytes (planner integration in progress)",
            options.vram_budget.unwrap());
    }
    if options.memory_report {
        eprintln!("[nsl] --memory-report requested (planner integration in progress)");
    }

    // M52: Embed weight hash if weights were loaded
    compiler.embed_weight_hash()?;
    compiler.finalize()
}

/// Compile for standalone export: like `compile()` but uses `compile_standalone_main()`
/// which initialises the weight provider and standalone arg parser before user code.
pub fn compile_standalone(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    config: StandaloneConfig,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> Result<Vec<u8>, CodegenError> {
    let mut compiler = Compiler::new(interner, type_map, options)?;
    compiler.dump_ir = dump_ir;
    compiler.standalone_config = Some(config);
    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;
    compiler.declare_runtime_functions()?;
    compiler.declare_user_functions(&ast.stmts)?;
    // M39b: Apply vmap AST transforms to generate batched function variants
    let _batched_fns = compiler.apply_vmap_transforms(ast);
    // TODO: register batched_fns for compilation when call-site dispatch is wired
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    compiler.compile_flash_attention_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    compiler.compile_standalone_main(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;
    compiler.finalize()
}

/// Compile in test mode: functions are compiled normally but main() dispatches
/// to @test functions based on `--run <name>` argv. Returns (object_bytes, test_fn_names).
pub fn compile_test(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> Result<(Vec<u8>, Vec<String>), CodegenError> {
    let mut compiler = Compiler::new(interner, type_map, options)?;
    compiler.dump_ir = dump_ir;
    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;
    compiler.declare_runtime_functions()?;
    compiler.declare_user_functions(&ast.stmts)?;
    // M39b: Apply vmap AST transforms to generate batched function variants
    let _batched_fns = compiler.apply_vmap_transforms(ast);
    // TODO: register batched_fns for compilation when call-site dispatch is wired
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    compiler.compile_flash_attention_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;
    let test_fns = compiler.test_fns.clone();
    if test_fns.is_empty() {
        return Err(CodegenError::new("no @test functions found".to_string()));
    }
    compiler.compile_test_main()?;
    let bytes = compiler.finalize()?;
    Ok((bytes, test_fns))
}

/// Compile a library module (non-entry). Functions use Linkage::Export, no main().
pub fn compile_module(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    module_prefix: &str,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> Result<Vec<u8>, CodegenError> {
    compile_module_with_imports(ast, interner, type_map, module_prefix, &[], HashMap::new(), HashSet::new(), dump_ir, options)
}

/// Compile a library module with imported symbols from its own dependencies.
#[allow(clippy::too_many_arguments)]
pub fn compile_module_with_imports(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    module_prefix: &str,
    imported_fns: &[(String, String, Signature)],
    imported_struct_layouts: HashMap<String, crate::context::StructLayout>,
    imported_model_names: HashSet<String>,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> Result<Vec<u8>, CodegenError> {
    let mut compiler = Compiler::new(interner, type_map, options)?;
    compiler.dump_ir = dump_ir;
    compiler.module_prefix = module_prefix.to_string();

    // Register imported structs/models from dependencies
    for (name, layout) in imported_struct_layouts {
        compiler.struct_layouts.insert(name, layout);
    }
    for name in imported_model_names {
        compiler.imported_model_names.insert(name);
    }

    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;
    compiler.declare_runtime_functions()?;
    compiler.declare_imported_functions(imported_fns)?;
    compiler.declare_user_functions_with_linkage(&ast.stmts, Linkage::Export)?;
    // M39b: Apply vmap AST transforms to generate batched function variants
    let _batched_fns = compiler.apply_vmap_transforms(ast);
    // TODO: register batched_fns for compilation when call-site dispatch is wired
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    compiler.compile_flash_attention_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;
    compiler.finalize()
}

/// Compile the entry module with imported functions from other modules.
/// Own functions use Linkage::Export, imported functions use Linkage::Import.
/// imported_fns entries are (raw_name, mangled_name, signature).
#[allow(clippy::too_many_arguments)]
pub fn compile_entry(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    imported_fns: &[(String, String, Signature)],
    imported_struct_layouts: HashMap<String, crate::context::StructLayout>,
    imported_model_names: HashSet<String>,
    imported_enum_variants: HashMap<String, i64>,
    imported_enum_defs: HashMap<String, Vec<(String, i64)>>,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> Result<Vec<u8>, CodegenError> {
    let mut compiler = Compiler::new(interner, type_map, options)?;
    compiler.dump_ir = dump_ir;

    // Register imported structs/enums so the entry module can reference them
    for (name, layout) in imported_struct_layouts {
        compiler.struct_layouts.insert(name, layout);
    }
    // Mark imported model names so we don't generate struct ctors for them
    for name in imported_model_names {
        compiler.imported_model_names.insert(name);
    }
    for (name, tag) in imported_enum_variants {
        compiler.enum_variants.insert(name, tag);
    }
    for (name, variants) in imported_enum_defs {
        compiler.enum_defs.insert(name, variants);
    }

    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;
    compiler.declare_runtime_functions()?;
    compiler.declare_imported_functions(imported_fns)?;
    compiler.declare_user_functions_with_linkage(&ast.stmts, Linkage::Export)?;
    // M39b: Apply vmap AST transforms to generate batched function variants
    let _batched_fns = compiler.apply_vmap_transforms(ast);
    // TODO: register batched_fns for compilation when call-site dispatch is wired
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    compiler.compile_flash_attention_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    compiler.compile_main(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;
    // M31: Print fusion report if enabled
    if compiler.fusion_report_enabled {
        crate::fusion_report::print_fusion_report(&compiler.fusion_events, &compiler.fusion_barriers);
    }
    compiler.finalize()
}
