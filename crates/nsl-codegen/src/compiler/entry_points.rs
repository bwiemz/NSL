use std::collections::{HashMap, HashSet};

use cranelift_codegen::ir::Signature;
use cranelift_module::Linkage;

use nsl_lexer::Interner;
use nsl_semantic::checker::TypeMap;

use super::{Compiler, StandaloneConfig};
use crate::error::CodegenError;

/// Dev Tools Phase 2, Task 4: run the profiling walker once and stash its
/// output on the compiler so kernel-launch sites can attach compile-time
/// predictions by `NodeId`.  Non-fatal: a walker error is logged to stderr
/// (under `NSL_DEBUG`) and leaves the maps empty — profiling is advisory and
/// must never break a build.
fn run_profile_pre_pass(
    compiler: &mut Compiler<'_>,
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    options: &crate::CompileOptions,
) {
    use crate::gpu_specs::find_gpu;
    use crate::profiling::instrument::ManifestBuilder;
    use crate::profiling::shape_env::ShapeEnv;
    use crate::profiling::types::EntryKind;
    use crate::profiling::walker::walk_ops;

    let env = ShapeEnv::with_defaults();
    let target_gpu = options.target_gpu.as_str();
    let dtype = options.dtype.as_str();

    let gpu = match find_gpu(target_gpu) {
        Some(g) => g,
        None => {
            if std::env::var("NSL_DEBUG").is_ok() {
                eprintln!(
                    "[nsl] profile_kernels: unknown GPU target {:?}, skipping pre-pass",
                    target_gpu
                );
            }
            return;
        }
    };

    // The walker expects an `&AnalysisResult` but only reads `type_map`.
    // Construct a minimal synthetic result from the `TypeMap` we already have.
    let analysis = nsl_semantic::AnalysisResult {
        diagnostics: Vec::new(),
        type_map: type_map.clone(),
        scopes: nsl_semantic::scope::ScopeMap::new(),
        ownership_info: std::collections::HashMap::new(),
        wrga_configs: Vec::new(),
        freeze_configs: Vec::new(),
        adapter_configs: Vec::new(),
    };

    // Task 6 + Phase 2.5 Task 4: populate source text/name so
    // SourceSpanJson::from_span produces real line numbers in manifest records.
    // Source-text priority: explicit options → disk fallback → empty.  Runs
    // before walk_ops so source context is ready even if the walker fails,
    // and before any fusion decisions are made.
    compiler.source_text = match &options.profile_source_text {
        Some(s) => s.clone(),
        None => options
            .profile_source_file_name
            .as_ref()
            .and_then(|p| std::fs::read_to_string(p).ok())
            .unwrap_or_default(),
    };
    compiler.source_file_name = options
        .profile_source_file_name
        .clone()
        .unwrap_or_default();

    match walk_ops(ast, &analysis, interner, EntryKind::Auto, &env, gpu, dtype) {
        Ok(report) => {
            compiler.prediction_map = report
                .ops
                .iter()
                .filter_map(|op| op.origin_node.map(|nid| (nid, op.clone())))
                .collect();
            compiler.manifest_builder = Some(ManifestBuilder::new(target_gpu, dtype));
            // Phase 2.5 Task 3: seed the Compiler-owned plan so later fusion
            // passes (apply_epilogue_fusion, apply_reduction_fusion) can write
            // into the same instance that `fusion_constituents` reads at
            // launch-emit time. Copy WRGA-level adapter-fusion groups if a
            // recent @train compile produced them; otherwise start empty.
            let seeded = compiler
                .last_wrga_plan
                .as_ref()
                .map(|p| p.fusion.clone())
                .unwrap_or_default();
            compiler.fusion_plan_for_profile = Some(seeded);
        }
        Err(e) => {
            if std::env::var("NSL_DEBUG").is_ok() {
                eprintln!(
                    "[nsl] profile_kernels: walker failed ({}), continuing without predictions",
                    e
                );
            }
        }
    }
}

/// Dev Tools Phase 2, Task 6: drain `Compiler.manifest_builder` (if set) and
/// write the resulting kernel-profile manifest to
/// `options.manifest_output_path`.  Non-fatal on any failure — profiling is
/// advisory and must never break a build.  Called from each codegen entry
/// function's latest reliable success path.
fn write_manifest_if_needed(compiler: &mut Compiler<'_>, options: &crate::CompileOptions) {
    let Some(mb) = compiler.manifest_builder.take() else {
        return;
    };
    let manifest = mb.finish();
    if let Some(out_path) = &options.manifest_output_path {
        match crate::profiling::instrument::write_manifest(out_path, &manifest) {
            Ok(_) => {
                if std::env::var("NSL_DEBUG").is_ok() {
                    eprintln!(
                        "[profile] wrote manifest with {} kernels to {}",
                        manifest.kernels.len(),
                        out_path.display()
                    );
                }
            }
            Err(e) => {
                eprintln!(
                    "warning: failed to write profile manifest to {}: {}",
                    out_path.display(),
                    e
                );
            }
        }
    } else if std::env::var("NSL_DEBUG").is_ok() {
        eprintln!(
            "[profile] manifest_builder set but no manifest_output_path — \
             skipping write ({} kernels)",
            manifest.kernels.len()
        );
    }
}

/// Main entry point (single-file, backward compatible).
pub fn compile(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> Result<Vec<u8>, CodegenError> {
    compile_returning_plan(ast, interner, type_map, dump_ir, options).map(|(b, _)| b)
}

/// Same as [`compile`] but also returns any `WrgaPlan` stashed on the compiler
/// during `@train` block lowering (Milestone B.1 Task 2).
///
/// Used by `nsl build --wrga-report` to surface WRGA analysis to the user.
pub fn compile_returning_plan(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> Result<(Vec<u8>, Option<crate::wrga::WrgaPlan>), CodegenError> {
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
                    let eliminator =
                        crate::weight_aware::DeadWeightEliminator::new(&options.weight_config);
                    let names: Vec<String> = wmap.names().map(|s| s.to_string()).collect();
                    for name in &names {
                        if let Some(entry) = wmap.get_mut(name) {
                            eliminator.eliminate(entry);
                        }
                    }
                }

                // Weight analysis report (when --weight-analysis is set)
                if options.weight_analysis {
                    crate::weight_aware::print_weight_analysis_report(
                        &wmap,
                        &options.weight_config,
                    );
                }

                // M52d: Compute and store quantization scales for FP8/INT8 weights
                {
                    let names: Vec<String> = wmap.names().map(|s| s.to_string()).collect();
                    for name in &names {
                        if let Some(entry) = wmap.get(name) {
                            if let Some(scale) = entry.compute_scale() {
                                compiler.memory.weight_scales.insert(name.clone(), scale);
                            }
                        }
                    }
                    if !compiler.memory.weight_scales.is_empty() {
                        eprintln!(
                            "[nsl] M52d: computed compile-time scales for {} quantized weights",
                            compiler.memory.weight_scales.len()
                        );
                    }
                }

                eprintln!(
                    "[nsl] loaded {} weights from {} (SHA-256: {})",
                    wmap.len(),
                    wmap.source_path(),
                    integrity.hash_hex
                );

                compiler.features.weight_integrity = Some(integrity);
                compiler.features.weight_map = Some(wmap);
            }
            Err(e) => {
                return Err(crate::error::CodegenError::new(format!(
                    "failed to load weights: {}",
                    e
                )));
            }
        }
    }

    compiler.dump_ir = dump_ir;

    // Dev Tools Phase 2, Task 4/6: run the kernel-profile pre-pass before any
    // body codegen so downstream kernel-launch sites can record manifest
    // entries keyed by `NodeId`.
    if options.profile_kernels {
        run_profile_pre_pass(&mut compiler, ast, interner, type_map, options);
    }

    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;
    compiler.declare_runtime_functions()?;
    compiler.declare_user_functions(&ast.stmts)?;
    // M39b: Apply vmap AST transforms and register batched function variants
    let vmap_results = compiler.apply_vmap_transforms(ast);
    compiler.register_batched_functions(&vmap_results);
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    compiler.compile_flash_attention_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    // M39c: Compile batched function bodies (after user functions, before main)
    compiler.compile_batched_functions(&vmap_results)?;
    // M36: Memory planner — compile-time slab allocation for GPU tensors.
    // Run BEFORE compile_main so the slab plan is available during codegen.
    {
        use crate::memory_planner::*;
        let allocs = analyze_ast_liveness(&ast.stmts, type_map, interner);
        if !allocs.is_empty() {
            let plannable: Vec<_> = allocs
                .iter()
                .filter(|a| a.is_plannable())
                .cloned()
                .collect();
            if !plannable.is_empty() {
                let graph = InterferenceGraph::build(&plannable);
                let plan = plan_slab(&plannable, &graph);

                if options.memory_report || plan.total_bytes > 0 {
                    let report = format_memory_report(&allocs, &plan);
                    eprintln!("[nsl] {}", report);
                }

                if let Some(budget) = options.vram_budget {
                    if let Some(err_msg) = check_vram_budget(&plan, budget) {
                        return Err(crate::error::CodegenError::new(err_msg));
                    }
                }

                // Build name → offset map for codegen
                for alloc in &plannable {
                    if let Some(&(_slot_id, offset)) = plan.assignments.get(&alloc.id) {
                        compiler
                            .memory
                            .slab_name_offsets
                            .insert(alloc.name.clone(), offset);
                    }
                }
                compiler.memory.slab_plan = Some(plan);
            }
        } else if options.memory_report {
            eprintln!("[nsl] Memory plan: no static-shape tensor allocations found");
        }
    }

    compiler.compile_main(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;

    // M53: Run WCET analysis for @real_time functions (after codegen, before finalize)
    if compiler.compile_options.wcet_enabled {
        compiler.run_wcet_analysis()?;
    }

    // M52: Embed weight hash if weights were loaded
    compiler.embed_weight_hash()?;
    let plan = compiler.last_wrga_plan.clone();
    // Dev Tools Phase 2, Task 6: write the profile manifest before finalize
    // consumes the compiler.
    write_manifest_if_needed(&mut compiler, options);
    let bytes = compiler.finalize()?;
    Ok((bytes, plan))
}

/// Like [`compile`] but also returns the collected `@zk_proof` function map.
///
/// This allows the CLI to iterate over the ZK-decorated functions after normal
/// compilation and invoke `zk::compile_zk()` on each one (Task 13 / M55).
///
/// Returns `(object_bytes, zk_proof_fns)` where `zk_proof_fns` maps mangled
/// function names to their [`crate::zk::backend::ZkMode`].
#[allow(clippy::type_complexity)]
pub fn compile_with_zk_info(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> Result<
    (
        Vec<u8>,
        HashMap<String, crate::zk::backend::ZkMode>,
        Vec<(String, crate::zk::ZkCompileResult)>,
    ),
    CodegenError,
> {
    let (res, zk_modes, zk_results, _plan) =
        compile_with_zk_info_best_effort_plan(ast, interner, type_map, dump_ir, options);
    res.map(|bytes| (bytes, zk_modes, zk_results))
}

/// Task 4 (B.2): same as [`compile_with_zk_info`] but also returns any `WrgaPlan`
/// stashed on the compiler during `@train` block lowering.  Used by
/// `nsl build --zk-circuit --wrga-report` to surface WRGA analysis on the ZK
/// build path.  The plan is returned even when later codegen stages fail, so
/// the CLI can emit the report before reporting the error.
#[allow(clippy::type_complexity)]
pub fn compile_with_zk_info_returning_plan(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> (
    Result<Vec<u8>, CodegenError>,
    HashMap<String, crate::zk::backend::ZkMode>,
    Vec<(String, crate::zk::ZkCompileResult)>,
    Option<crate::wrga::WrgaPlan>,
) {
    compile_with_zk_info_best_effort_plan(ast, interner, type_map, dump_ir, options)
}

#[allow(clippy::type_complexity, clippy::field_reassign_with_default)]
fn compile_with_zk_info_best_effort_plan(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> (
    Result<Vec<u8>, CodegenError>,
    HashMap<String, crate::zk::backend::ZkMode>,
    Vec<(String, crate::zk::ZkCompileResult)>,
    Option<crate::wrga::WrgaPlan>,
) {
    let mut compiler = match Compiler::new(interner, type_map, options) {
        Ok(c) => c,
        Err(e) => return (Err(e), HashMap::new(), Vec::new(), None),
    };

    compiler.dump_ir = dump_ir;

    // Run every pass up to (but not including) finalize so we can observe
    // `last_wrga_plan` even on an error path before consuming the compiler.
    let pre_finalize = (|| -> Result<(), CodegenError> {
        // M52: Load weights if --weights was provided
        if let Some(ref weight_path) = options.weight_file {
            let mut wmap = crate::weight_aware::WeightMap::load(weight_path).map_err(|e| {
                crate::error::CodegenError::new(format!("failed to load weights: {}", e))
            })?;
            let integrity = crate::weight_aware::WeightIntegrity::new(*wmap.hash());
            if options.weight_config.sparse_codegen || options.weight_config.dead_weight_elim {
                let config = &options.weight_config;
                let names: Vec<String> = wmap.names().map(|s| s.to_string()).collect();
                for name in &names {
                    if let Some(entry) = wmap.get_mut(name) {
                        entry.analyze_sparsity(config);
                    }
                }
            }
            if options.weight_config.dead_weight_elim {
                let eliminator =
                    crate::weight_aware::DeadWeightEliminator::new(&options.weight_config);
                let names: Vec<String> = wmap.names().map(|s| s.to_string()).collect();
                for name in &names {
                    if let Some(entry) = wmap.get_mut(name) {
                        eliminator.eliminate(entry);
                    }
                }
            }
            if options.weight_analysis {
                crate::weight_aware::print_weight_analysis_report(&wmap, &options.weight_config);
            }
            eprintln!(
                "[nsl] loaded {} weights from {} (SHA-256: {})",
                wmap.len(),
                wmap.source_path(),
                integrity.hash_hex
            );
            compiler.features.weight_integrity = Some(integrity);
            compiler.features.weight_map = Some(wmap);
        }

        compiler.intern_string("")?;
        compiler.collect_strings(&ast.stmts)?;
        compiler.collect_enums(&ast.stmts)?;
        compiler.collect_structs(&ast.stmts)?;
        compiler.collect_models(&ast.stmts)?;
        compiler.declare_runtime_functions()?;
        compiler.declare_user_functions(&ast.stmts)?;
        let vmap_results = compiler.apply_vmap_transforms(ast);
        compiler.register_batched_functions(&vmap_results);
        compiler.compile_datatype_defs(&ast.stmts)?;
        compiler.compile_kernels(&ast.stmts)?;
        compiler.compile_flash_attention_kernels(&ast.stmts)?;
        compiler.compile_user_functions(&ast.stmts)?;
        compiler.compile_batched_functions(&vmap_results)?;
        compiler.compile_main(&ast.stmts)?;
        compiler.compile_pending_lambdas()?;

        if let Some(budget) = options.vram_budget {
            eprintln!(
                "[nsl] --vram-budget set to {} bytes (planner integration in progress)",
                budget
            );
        }
        if options.memory_report {
            eprintln!("[nsl] --memory-report requested (planner integration in progress)");
        }

        // M53: Run WCET analysis for @real_time functions
        if compiler.compile_options.wcet_enabled {
            compiler.run_wcet_analysis()?;
        }

        // M52: Embed weight hash if weights were loaded
        compiler.embed_weight_hash()?;
        Ok(())
    })();

    let plan = compiler.last_wrga_plan.clone();

    // Capture ZK fn map before finalize() consumes the compiler.
    let zk_proof_fns = compiler.features.zk_proof_fns.clone();

    // If the pre-finalize pipeline failed, bail out with the plan preserved.
    if let Err(e) = pre_finalize {
        return (Err(e), zk_proof_fns, Vec::new(), plan);
    }

    // M55: Compile @zk_proof functions to ZK circuits
    let mut zk_results: Vec<(String, crate::zk::ZkCompileResult)> = Vec::new();
    for (fn_name, mode) in &zk_proof_fns {
        if let Some(fn_def) = compiler.features.zk_fn_defs.get(fn_name) {
            let zk_config = {
                let mut cfg = crate::zk::backend::ZkConfig::default();
                // Wire --zk-backend flag to select backend
                cfg.backend = match compiler.compile_options.zk_backend.to_lowercase().as_str() {
                    "plonky3" | "fri" => crate::zk::backend::ZkBackendType::Plonky3,
                    "halo2" => crate::zk::backend::ZkBackendType::Halo2,
                    "folding" | "nova" | "" => crate::zk::backend::ZkBackendType::Folding,
                    other => {
                        eprintln!(
                            "[nsl] warning: unknown ZK backend '{}', using folding",
                            other
                        );
                        crate::zk::backend::ZkBackendType::Folding
                    }
                };
                // Wire --zk-field flag to select finite field
                cfg.field = match compiler.compile_options.zk_field.to_lowercase().as_str() {
                    "bn254" | "bn256" => crate::zk::backend::ZkField::BN254,
                    "mersenne31" | "m31" | "" => crate::zk::backend::ZkField::Mersenne31,
                    other => {
                        eprintln!(
                            "[nsl] warning: unknown ZK field '{}', using Mersenne31",
                            other
                        );
                        crate::zk::backend::ZkField::Mersenne31
                    }
                };
                cfg.emit_solidity = compiler.compile_options.zk_solidity;
                // Wire --zk-weights flag to load weight file for witness generation
                if let Some(ref weights_path) = compiler.compile_options.zk_weights_path {
                    eprintln!(
                        "[nsl] ZK: loading weights from {} for witness generation",
                        weights_path.display()
                    );
                    cfg.weights_path = Some(weights_path.clone());
                }
                cfg
            };
            match crate::zk::compile_zk(fn_def, *mode, &zk_config, type_map, interner) {
                Ok(result) => {
                    eprintln!(
                        "[nsl] M55: compiled ZK circuit for '{}' — {} constraints, proof ~{} KB",
                        fn_name,
                        result.stats.num_constraints,
                        result.stats.estimated_proof_size_bytes / 1024,
                    );
                    zk_results.push((fn_name.clone(), result));
                }
                Err(e) => {
                    eprintln!("[nsl] M55: ZK compilation warning for '{}': {}", fn_name, e);
                }
            }
        }
    }

    // Dev Tools Phase 2, Task 6: write the profile manifest before finalize.
    write_manifest_if_needed(&mut compiler, options);
    let bytes = compiler.finalize();
    (bytes, zk_proof_fns, zk_results, plan)
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
    let (res, _plan) =
        compile_standalone_best_effort_plan(ast, interner, type_map, config, dump_ir, options);
    res
}

/// Task 4 (B.2): same as [`compile_standalone`] but also returns any `WrgaPlan`
/// stashed on the compiler during `@train` block lowering.  Used by
/// `nsl build --standalone --wrga-report` to surface WRGA analysis even when
/// later codegen stages fail.
pub fn compile_standalone_returning_plan(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    config: StandaloneConfig,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> (Result<Vec<u8>, CodegenError>, Option<crate::wrga::WrgaPlan>) {
    compile_standalone_best_effort_plan(ast, interner, type_map, config, dump_ir, options)
}

fn compile_standalone_best_effort_plan(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    config: StandaloneConfig,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> (Result<Vec<u8>, CodegenError>, Option<crate::wrga::WrgaPlan>) {
    let mut compiler = match Compiler::new(interner, type_map, options) {
        Ok(c) => c,
        Err(e) => return (Err(e), None),
    };
    compiler.dump_ir = dump_ir;
    compiler.standalone_config = Some(config);
    let pre_finalize = (|| -> Result<(), CodegenError> {
        compiler.intern_string("")?;
        compiler.collect_strings(&ast.stmts)?;
        compiler.collect_enums(&ast.stmts)?;
        compiler.collect_structs(&ast.stmts)?;
        compiler.collect_models(&ast.stmts)?;
        compiler.declare_runtime_functions()?;
        compiler.declare_user_functions(&ast.stmts)?;
        let vmap_results = compiler.apply_vmap_transforms(ast);
        compiler.register_batched_functions(&vmap_results);
        compiler.compile_datatype_defs(&ast.stmts)?;
        compiler.compile_kernels(&ast.stmts)?;
        compiler.compile_flash_attention_kernels(&ast.stmts)?;
        compiler.compile_user_functions(&ast.stmts)?;
        compiler.compile_batched_functions(&vmap_results)?;
        compiler.compile_standalone_main(&ast.stmts)?;
        compiler.compile_pending_lambdas()?;
        if compiler.compile_options.wcet_enabled {
            compiler.run_wcet_analysis()?;
        }
        Ok(())
    })();
    let plan = compiler.last_wrga_plan.clone();
    // Dev Tools Phase 2, Task 6: write the profile manifest before finalize.
    if pre_finalize.is_ok() {
        write_manifest_if_needed(&mut compiler, options);
    }
    let result = pre_finalize.and_then(|()| compiler.finalize());
    (result, plan)
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
    // M39b: Apply vmap AST transforms and register batched function variants
    let vmap_results = compiler.apply_vmap_transforms(ast);
    compiler.register_batched_functions(&vmap_results);
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    compiler.compile_flash_attention_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    // M39c: Compile batched function bodies
    compiler.compile_batched_functions(&vmap_results)?;
    compiler.compile_pending_lambdas()?;
    let test_fns = compiler.registry.test_fns.clone();
    if test_fns.is_empty() {
        return Err(CodegenError::new("no @test functions found".to_string()));
    }
    compiler.compile_test_main()?;
    // Dev Tools Phase 2, Task 6: write the profile manifest before finalize.
    write_manifest_if_needed(&mut compiler, options);
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
    compile_module_with_imports(
        ast,
        interner,
        type_map,
        module_prefix,
        &[],
        HashMap::new(),
        HashSet::new(),
        dump_ir,
        options,
    )
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
    compile_module_with_imports_returning_plan(
        ast,
        interner,
        type_map,
        module_prefix,
        imported_fns,
        imported_struct_layouts,
        imported_model_names,
        dump_ir,
        options,
    )
    .map(|(bytes, _)| bytes)
}

/// Task 4: same as `compile_module_with_imports` but also returns the last
/// `WrgaPlan` stashed on the `Compiler` during this compile (if any).  On
/// error, the plan is *discarded* — use
/// [`compile_module_with_imports_best_effort_plan`] when you need to observe
/// the plan produced before a later codegen failure.
#[allow(clippy::too_many_arguments)]
pub fn compile_module_with_imports_returning_plan(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    module_prefix: &str,
    imported_fns: &[(String, String, Signature)],
    imported_struct_layouts: HashMap<String, crate::context::StructLayout>,
    imported_model_names: HashSet<String>,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> Result<(Vec<u8>, Option<crate::wrga::WrgaPlan>), CodegenError> {
    let (res, plan) = compile_module_with_imports_best_effort_plan(
        ast,
        interner,
        type_map,
        module_prefix,
        imported_fns,
        imported_struct_layouts,
        imported_model_names,
        dump_ir,
        options,
    );
    res.map(|bytes| (bytes, plan))
}

/// Task 4: like `compile_module_with_imports_returning_plan` but returns the
/// plan *even when codegen fails*, so test harnesses can assert on WRGA
/// behaviour without a full link-ready object.
#[allow(clippy::too_many_arguments)]
pub fn compile_module_with_imports_best_effort_plan(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    module_prefix: &str,
    imported_fns: &[(String, String, Signature)],
    imported_struct_layouts: HashMap<String, crate::context::StructLayout>,
    imported_model_names: HashSet<String>,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> (Result<Vec<u8>, CodegenError>, Option<crate::wrga::WrgaPlan>) {
    let mut compiler = match Compiler::new(interner, type_map, options) {
        Ok(c) => c,
        Err(e) => return (Err(e), None),
    };
    compiler.dump_ir = dump_ir;
    compiler.module_prefix = module_prefix.to_string();
    for (name, layout) in imported_struct_layouts {
        compiler.types.struct_layouts.insert(name, layout);
    }
    for name in imported_model_names {
        compiler.models.imported_model_names.insert(name);
    }

    // Dev Tools Phase 2, Task 4: run the kernel-profile pre-pass once at the
    // start of codegen so downstream kernel-launch sites can attach
    // compile-time predictions by `NodeId`.
    if options.profile_kernels {
        run_profile_pre_pass(&mut compiler, ast, interner, type_map, options);
    }

    // Run every pass up to (but not including) `finalize`, so we can observe
    // `last_wrga_plan` even on an error path before consuming the compiler.
    let pre_finalize = (|| -> Result<(), CodegenError> {
        compiler.intern_string("")?;
        compiler.collect_strings(&ast.stmts)?;
        compiler.collect_enums(&ast.stmts)?;
        compiler.collect_structs(&ast.stmts)?;
        compiler.collect_models(&ast.stmts)?;
        compiler.declare_runtime_functions()?;
        compiler.declare_imported_functions(imported_fns)?;
        compiler.declare_user_functions_with_linkage(&ast.stmts, Linkage::Export)?;
        let vmap_results = compiler.apply_vmap_transforms(ast);
        compiler.register_batched_functions(&vmap_results);
        compiler.compile_datatype_defs(&ast.stmts)?;
        compiler.compile_kernels(&ast.stmts)?;
        compiler.compile_flash_attention_kernels(&ast.stmts)?;
        compiler.compile_user_functions(&ast.stmts)?;
        compiler.compile_batched_functions(&vmap_results)?;
        compiler.compile_pending_lambdas()?;
        Ok(())
    })();
    let plan = compiler.last_wrga_plan.clone();
    // Dev Tools Phase 2, Task 6: write the profile manifest before finalize.
    if pre_finalize.is_ok() {
        write_manifest_if_needed(&mut compiler, options);
    }
    let result = pre_finalize.and_then(|()| compiler.finalize());
    (result, plan)
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
    imported_model_method_bodies: HashMap<String, HashMap<String, nsl_ast::decl::FnDef>>,
    imported_model_field_types: HashMap<String, HashMap<String, String>>,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> Result<Vec<u8>, CodegenError> {
    compile_entry_returning_plan(
        ast,
        interner,
        type_map,
        imported_fns,
        imported_struct_layouts,
        imported_model_names,
        imported_enum_variants,
        imported_enum_defs,
        imported_model_method_bodies,
        imported_model_field_types,
        dump_ir,
        options,
    )
    .map(|(b, _)| b)
}

/// Same as [`compile_entry`] but also returns any `WrgaPlan` stashed on the
/// compiler during `@train` block lowering (Milestone B.1 Task 2).
#[allow(clippy::too_many_arguments)]
pub fn compile_entry_returning_plan(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    imported_fns: &[(String, String, Signature)],
    imported_struct_layouts: HashMap<String, crate::context::StructLayout>,
    imported_model_names: HashSet<String>,
    imported_enum_variants: HashMap<String, i64>,
    imported_enum_defs: HashMap<String, Vec<(String, i64)>>,
    imported_model_method_bodies: HashMap<String, HashMap<String, nsl_ast::decl::FnDef>>,
    imported_model_field_types: HashMap<String, HashMap<String, String>>,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> Result<(Vec<u8>, Option<crate::wrga::WrgaPlan>), CodegenError> {
    let mut compiler = Compiler::new(interner, type_map, options)?;
    compiler.dump_ir = dump_ir;

    // Register imported structs/enums so the entry module can reference them
    for (name, layout) in imported_struct_layouts {
        compiler.types.struct_layouts.insert(name, layout);
    }
    // Mark imported model names so we don't generate struct ctors for them
    for name in imported_model_names {
        compiler.models.imported_model_names.insert(name);
    }
    for (name, tag) in imported_enum_variants {
        compiler.types.enum_variants.insert(name, tag);
    }
    for (name, variants) in imported_enum_defs {
        compiler.types.enum_defs.insert(name, variants);
    }

    // Dev Tools Phase 2, Task 4/6: run the kernel-profile pre-pass before any
    // body codegen.
    if options.profile_kernels {
        run_profile_pre_pass(&mut compiler, ast, interner, type_map, options);
    }

    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;

    // Merge imported model method bodies and field types from dependency modules.
    // This enables source AD to inline method calls on imported model types
    // (e.g., RMSNorm.forward, GroupedQueryAttention.forward from stdlib).
    // Local models (from collect_models above) take precedence over imports.
    for (model_name, methods) in imported_model_method_bodies {
        compiler
            .models
            .model_method_bodies
            .entry(model_name)
            .or_insert(methods);
    }
    for (model_name, fields) in imported_model_field_types {
        compiler
            .models
            .model_field_types
            .entry(model_name)
            .or_insert(fields);
    }
    compiler.declare_runtime_functions()?;
    compiler.declare_imported_functions(imported_fns)?;
    compiler.declare_user_functions_with_linkage(&ast.stmts, Linkage::Export)?;
    // M39b: Apply vmap AST transforms and register batched function variants
    let vmap_results = compiler.apply_vmap_transforms(ast);
    compiler.register_batched_functions(&vmap_results);
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    compiler.compile_flash_attention_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    // M39c: Compile batched function bodies
    compiler.compile_batched_functions(&vmap_results)?;
    compiler.compile_main(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;
    // M53: Run WCET analysis for @real_time functions
    if compiler.compile_options.wcet_enabled {
        compiler.run_wcet_analysis()?;
    }
    // M31: Print fusion report if enabled
    if compiler.fusion.report_enabled {
        crate::fusion_report::print_fusion_report(
            &compiler.fusion.events,
            &compiler.fusion.barriers,
        );
    }
    let plan = compiler.last_wrga_plan.clone();
    // Dev Tools Phase 2, Task 6: write the profile manifest before finalize.
    write_manifest_if_needed(&mut compiler, options);
    let bytes = compiler.finalize()?;
    Ok((bytes, plan))
}
