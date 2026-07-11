use std::collections::{HashMap, HashSet};
use std::sync::Arc;

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
        csha_configs: Vec::new(),
        weight_index_map: std::collections::HashMap::new(),
        checkpoint_policies: std::collections::HashMap::new(),
        paged_kv_models: std::collections::HashSet::new(),
        fused_ce_configs: Vec::new(),
        fused_kl_ce_configs: Vec::new(),
        pca_configs: Vec::new(),
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

fn install_calibration_compile_bundle(
    compiler: &mut Compiler<'_>,
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
) {
    compiler.compile_options.calibration_compile_bundle = Some(Arc::new(
        crate::calibration::CalibrationCompileBundle {
            ast: ast.clone(),
            interner: interner.clone(),
            type_map: type_map.clone(),
        },
    ));
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

/// M52 / CPDT follow-up: shared weight-loading path for every codegen entry
/// point.  When `options.weight_file` is set, loads the safetensors file,
/// runs sparsity analysis + dead-weight elimination + optional weight-analysis
/// report + M52d scale computation, and stashes the resulting `WeightMap` +
/// integrity hash on the compiler.  A missing/unreadable file is a hard error.
/// A `None` weight path is a silent no-op — callers that intentionally skip
/// weights (standalone export, library-module precompile) can simply not set
/// `weight_file`.
///
/// Load-bearing for CPDT Phase 1: `stmt.rs::invoke_cpdt_if_enabled` reads
/// `compiler.features.weight_map.as_ref()`, so the multi-file entry points
/// `compile_module_with_imports_best_effort_plan` and
/// `compile_entry_returning_plan` must call this helper before compiling any
/// user-function body — otherwise CPDT silently no-ops on every real program
/// that uses `from X import Y` or a `train(...)` block.
fn load_and_register_weights_if_needed(
    compiler: &mut Compiler<'_>,
    options: &crate::CompileOptions,
) -> Result<(), CodegenError> {
    let Some(weight_path) = options.weight_file.as_ref() else {
        return Ok(());
    };

    let mut wmap = crate::weight_aware::WeightMap::load(weight_path).map_err(|e| {
        crate::error::CodegenError::new(format!("failed to load weights: {}", e))
    })?;
    let integrity = crate::weight_aware::WeightIntegrity::new(*wmap.hash());

    // CPDT Part III v2.7 — HF Mixtral auto-pack.
    //
    // Scan the freshly-loaded WeightMap for HF Mixtral per-expert
    // patterns (`<prefix>.experts.{e}.w{1,2,3}.weight`) and rewrite
    // each detected block in place into NSL packed convention so the
    // downstream `derive_v3_dims` / `derive_v4_dims` can resolve them.
    // Per-block atomicity is provided by v2.6's two-phase commit;
    // cross-block failures don't roll back successful blocks (a user
    // with one bad MoE layer still gets every other layer packed).
    //
    // The auto-pack is an UNCONDITIONAL post-load pass — a no-op when
    // no HF patterns are found, so no CLI flag is needed to gate it.
    // Users with non-HF safetensors (NSL-packed or other MoE
    // conventions) see no behavior change. Users with HF Mixtral
    // safetensors get the pack for free.
    let auto_pack = crate::moe_hf_pack::pack_all_detected_hf_mixtral_blocks(&mut wmap);
    if !auto_pack.packed.is_empty() {
        eprintln!(
            "[nsl] CPDT v2.7: auto-packed {} HF Mixtral MoE block{} into NSL convention",
            auto_pack.packed.len(),
            if auto_pack.packed.len() == 1 { "" } else { "s" },
        );
        for (block, _) in &auto_pack.packed {
            eprintln!(
                "  - {} (num_experts={})",
                block.hf_prefix, block.num_experts,
            );
        }
    }
    // CPDT Part III v2.15 — bias auto-pack telemetry. Blocks that
    // shipped per-expert HF `.w{1,2,3}.bias` keys get their biases
    // packed into NSL convention alongside the weights. Blocks
    // without biases produce no entry here (Ok(None) no-op).
    if !auto_pack.bias_packed.is_empty() {
        eprintln!(
            "[nsl] CPDT v2.15: auto-packed v4 biases for {} HF Mixtral MoE block{}",
            auto_pack.bias_packed.len(),
            if auto_pack.bias_packed.len() == 1 { "" } else { "s" },
        );
        for (block, _) in &auto_pack.bias_packed {
            eprintln!(
                "  - {} (num_experts={})",
                block.hf_prefix, block.num_experts,
            );
        }
    }
    // v2.7 adversarial-review F2 fix: hard-refuse on any auto-pack
    // failure. Previously the wrapper only printed failures to stderr
    // and let the build continue, which created a silent-corruption
    // hazard for the mixed HF+already-packed case: detection finds
    // the HF block, packer trips `TargetAlreadyExists` on the stale
    // pre-existing packed entry, the failure is logged-but-ignored,
    // and `derive_v4_dims` then succeeds against the STALE shadow
    // data while the freshly-loaded HF per-expert tensors become
    // orphans. Per the "deferral must refuse" / "no silent fallback"
    // invariant carried from v2.3..v2.6, any HF block detected MUST
    // either pack successfully or surface as a build error.
    //
    // v2.15 extends the same refuse-loudly contract to bias-pack
    // failures: a partial bias bundle would silently drop biases at
    // the v4 lowering's `detect_v4_biases` (no `.experts.gate.bias`
    // resolves because the HF `.w1.bias` keys remain unpacked), so
    // the 5-arg no-bias path runs and produces wrong numerics. Same
    // surface treatment as weight-pack failures.
    if !auto_pack.failed.is_empty() || !auto_pack.bias_failed.is_empty() {
        let mut msg = String::from(
            "CPDT v2.7/v2.15/v2.16: HF Mixtral auto-pack failed for one or more detected blocks:\n",
        );
        use std::fmt::Write as _;
        for (block, err) in &auto_pack.failed {
            let _ = writeln!(msg, "  - {} (weights): {}", block.hf_prefix, err);
        }
        for (block, err) in &auto_pack.bias_failed {
            // v2.16-B: BiasesWithoutWeights uses a synthetic block
            // with num_experts=0. Surface a clearer label so users
            // know the failure isn't tied to a fully-detected block.
            let kind = if matches!(err, crate::moe_hf_pack::PackError::BiasesWithoutWeights { .. }) {
                "biases-without-weights"
            } else {
                "biases"
            };
            let _ = writeln!(msg, "  - {} ({}): {}", block.hf_prefix, kind, err);
        }
        if !auto_pack.packed.is_empty() || !auto_pack.bias_packed.is_empty() {
            let _ = std::fmt::Write::write_str(
                &mut msg,
                "\nNote: other blocks WERE successfully packed in place (per-block \
                 atomicity preserved). Rebuilding after fixing the failed blocks will \
                 not double-pack the successful ones.",
            );
        }
        return Err(crate::error::CodegenError::new(msg));
    }

    // Sparsity analysis for sparse codegen / dead-weight elimination.
    if options.weight_config.sparse_codegen || options.weight_config.dead_weight_elim {
        let names: Vec<String> = wmap.names().map(|s| s.to_string()).collect();
        for name in &names {
            if let Some(entry) = wmap.get_mut(name) {
                entry.analyze_sparsity(&options.weight_config);
            }
        }
    }

    // Dead-weight elimination.
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

    // Optional --weight-analysis report.
    if options.weight_analysis {
        crate::weight_aware::print_weight_analysis_report(&wmap, &options.weight_config);
    }

    // M52d: compile-time quantization scales for FP8/INT8 weights.
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
    Ok(())
}

/// Run both AWQ and WGGO pre-scan discovery passes against the AST and
/// populate `opts` fields that are still `None`.  This is the test-facing
/// seam: all compiler entry points delegate to
/// [`populate_calibration_retention_from_ast_if_unset`] which calls this
/// helper internally.
///
/// Idempotent: if a field is already `Some`, it is left unchanged.
///
/// §5.7: when `opts.wggo.importance == WggoImportance::Auto` and any of the
/// §5.4/§5.5/§5.6 conditions are met, an informational note is emitted to
/// stderr and `opts.wggo.importance` is demoted to `WggoImportance::Magnitude`.
pub(crate) fn run_pre_scan_phase(
    ast: &nsl_ast::Module,
    interner: &Interner,
    mut opts: crate::CompileOptions,
) -> crate::CompileOptions {
    if opts.calibration_retention.is_none() {
        let discovered_awq = crate::calibration::pre_scan_awq_projections_from_ast(ast, interner);
        if !discovered_awq.is_empty() {
            opts.calibration_retention = Some(discovered_awq);
        }
    }
    if opts.calibration_grad_retention.is_none() {
        let discovered_wggo =
            crate::calibration::discovery::pre_scan_wggo_targets_from_ast(ast, interner);
        if !discovered_wggo.is_empty() {
            opts.calibration_grad_retention = Some(discovered_wggo);
        }
    }

    // §5.7: --wggo-importance=auto soft-fallback to magnitude.
    // When auto mode encounters any of the §5.4/§5.5/§5.6 conditions, emit
    // an informational note and demote the mode to magnitude.
    apply_auto_mode_fallback_note(ast, interner, &mut opts);

    opts
}

/// §5.7 soft-fallback for `WggoImportance::Auto`.
///
/// If `opts.wggo.importance == Auto` and any of the §5.4/§5.5/§5.6
/// conditions are met, this function:
///   1. Emits a single informational note to stderr.
///   2. Demotes `opts.wggo.importance` to `WggoImportance::Magnitude`.
///
/// This is the parallel to [`enforce_grad_mode_refusals`] — same three
/// conditions, but soft (note + demotion) rather than hard (error).
///
/// Must be called **after** `calibration_grad_retention` has been populated
/// by discovery (see [`run_pre_scan_phase`]).
fn apply_auto_mode_fallback_note(
    ast: &nsl_ast::Module,
    interner: &Interner,
    opts: &mut crate::CompileOptions,
) {
    use crate::WggoImportance;

    if opts.wggo.importance != WggoImportance::Auto {
        return;
    }

    let calibration_data_present = opts.calibration_data.is_some();
    let decorators_in_ast =
        crate::calibration::discovery::ast_has_wggo_target_decorators(ast, interner);
    let resolved_targets = opts.calibration_grad_retention.as_deref().unwrap_or(&[]);

    // §5.4: no @wggo_target decorators in source.
    // §5.5: decorators present but none reachable from entry point.
    // §5.6: decorators reachable but no calibration data provided.
    let trigger_reason: Option<String> = if !decorators_in_ast {
        Some("no @wggo_target decorators in source".to_string())
    } else if resolved_targets.is_empty() {
        let decorated =
            crate::calibration::discovery::list_decorated_class_names(ast, interner);
        let entry = crate::calibration::discovery::entry_point_fn_name(ast, interner)
            .unwrap_or_else(|| "(unknown)".to_string());
        Some(format!(
            "decorated classes {decorated:?} not reachable from entry `{entry}`"
        ))
    } else if !calibration_data_present {
        Some("decorators present but no calibration data provided".to_string())
    } else {
        None
    };

    if let Some(reason) = trigger_reason {
        eprintln!(
            "note: --wggo-importance=auto fell back to magnitude scoring.\n  \
             reason: {reason}\n  \
             effect: WGGO ILP runs against magnitude (||W||₂)-based importance scores.\n  \
             to silence: add @wggo_target + calibration data, OR set\n             \
             --wggo-importance=magnitude to opt out of grad scoring entirely."
        );
        opts.wggo.importance = WggoImportance::Magnitude;
    }
}

/// §5.4-§5.6 refusal checks for grad-mode WGGO.
///
/// Only fires when `opts.wggo.importance == WggoImportance::Grad`.  In `Auto`
/// or `Magnitude` modes the user never asked for gradient scoring, so there is
/// nothing to refuse.
///
/// Must be called **after** `run_pre_scan_phase` has already populated
/// `calibration_grad_retention` (or left it `None` when no reachable target
/// was found) so the checks below see the post-discovery state.
fn enforce_grad_mode_refusals(
    ast: &nsl_ast::Module,
    interner: &Interner,
    opts: &crate::CompileOptions,
) -> Result<(), CodegenError> {
    use crate::WggoImportance;

    if opts.wggo.importance != WggoImportance::Grad {
        return Ok(());
    }

    let decorators_in_ast =
        crate::calibration::discovery::ast_has_wggo_target_decorators(ast, interner);
    let resolved_targets = opts.calibration_grad_retention.as_deref().unwrap_or(&[]);
    let calibration_data_present = opts.calibration_data.is_some();

    // §5.4: grad mode requested but the source has no @wggo_target decorators at all.
    if !decorators_in_ast {
        return Err(CodegenError::new(
            "calibration: --wggo-importance=grad set but no @wggo_target decorators in source.\n\
  requested: gradient-importance scoring on the compiled model\n\
  expected:  ≥1 @wggo_target decorator on a model's `forward` method\n\
  found:     zero @wggo_target decorators in the AST\n\
  fix:       either add @wggo_target(...) to your attention block's\n\
             `forward` method, or use --wggo-importance=magnitude."
                .to_string(),
        ));
    }

    // §5.5: decorators present in source but none is reachable from the entry point.
    if resolved_targets.is_empty() {
        let decorated = crate::calibration::discovery::list_decorated_class_names(ast, interner);
        let entry_fn = crate::calibration::discovery::entry_point_fn_name(ast, interner)
            .unwrap_or_else(|| "(unknown)".to_string());
        let n = decorated.len();
        return Err(CodegenError::new(format!(
            "calibration: --wggo-importance=grad set but no decorated model is reachable.\n\
  requested: gradient-importance scoring on the compiled model\n\
  expected:  ≥1 @wggo_target-decorated model reachable from the entry point\n\
  found:     {n} decorator(s) in AST; 0 reachable\n\
             decorated classes: {decorated:?}\n\
             entry function:    {entry_fn}\n\
  fix:       either compile a model that uses one of the decorated classes,\n\
             or move the decorator to the class your entry point uses."
        )));
    }

    // §5.6: decorators present and model reachable, but no calibration data provided.
    if !calibration_data_present {
        let n = resolved_targets.len();
        return Err(CodegenError::new(format!(
            "calibration: --wggo-importance=grad requires calibration data, but none was provided.\n\
  requested: gradient-importance scoring with @wggo_target-decorated attention\n\
  expected:  a calibration data file via `quant awq {{ calibration_data = \"...\" }}`\n\
  found:     {n} @wggo_target target(s) discovered, model instantiated, but\n\
             no calibration data was provided.\n\
  fix:       add a `quant awq {{ calibration_data = \"path/to/data.safetensors\" }}`\n\
             block. If you don't have calibration data,\n\
             use --wggo-importance=magnitude."
        )));
    }

    Ok(())
}

fn populate_calibration_retention_from_ast_if_unset(
    compiler: &mut Compiler<'_>,
    ast: &nsl_ast::Module,
    interner: &Interner,
) -> Result<(), CodegenError> {
    // Delegate to the shared helper for pure-discovery work.
    let updated = run_pre_scan_phase(ast, interner, compiler.compile_options.clone());
    compiler.compile_options.calibration_retention = updated.calibration_retention;
    compiler.compile_options.calibration_grad_retention = updated.calibration_grad_retention;

    // AWQ-specific error: if an @quantize decorator is present but discovery
    // returned nothing, that is a user error we must surface.
    if compiler.compile_options.calibration_retention.is_none()
        && crate::calibration::discovery::ast_has_awq_quantize_decorator(ast, interner)
    {
        let model_name =
            crate::calibration::discovery::first_awq_quantized_model_name(ast, interner)
                .unwrap_or_else(|| "<unknown>".into());
        return Err(CodegenError::new(format!(
            "calibration: @quantize model declared but no AWQ projections discovered.\n\
 requested:  auto-discover AWQ projections for model '{model_name}'\n\
 expected:   @quantize(dtype=\"awq4\") model -> at least one DiscoveredProjection\n\
 found:      model '{model_name}' is AWQ-quantized but discovery returned zero projections.\n\
 Action:     either remove the @quantize decorator, or add the Linear/tensor projections the decorator is meant to target."
        )));
    }

    // §5.4-§5.6: grad-mode WGGO refusals.
    enforce_grad_mode_refusals(ast, interner, &compiler.compile_options)?;

    Ok(())
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

/// Test-helper: runs the codegen pipeline through `compile_user_functions`
/// (where model method bodies — and therefore retention splices — are
/// emitted), skipping `compile_main` so tests with models that have
/// non-trivial top-level constructors don't fail on unrelated setup issues.
/// Returns the splice-emission count snapped off the compiler.  Downstream
/// errors during body codegen are swallowed so the test can assert on the
/// splice counter even when the fixture's pipe targets don't fully lower
/// (e.g. pipe-as-function-call lookup failing — the splice already fired
/// *before* the call is dispatched, so the counter is correct even if the
/// later call step errors).
#[doc(hidden)]
pub fn compile_returning_splice_count_for_tests(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    options: &crate::CompileOptions,
) -> Result<u32, CodegenError> {
    let mut compiler = Compiler::new(interner, type_map, options)?;
    install_calibration_compile_bundle(&mut compiler, ast, interner, type_map);

    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;
    // M56 Task 17: compute agent struct layouts (after models, same pass ordering).
    compiler.collect_agents(&ast.stmts)?;
    populate_calibration_retention_from_ast_if_unset(&mut compiler, ast, interner)?;
    compiler.emit_retention_arena()?;
    // Task 10: backward (WGGO grad) sibling arena — spec §7.2 ordering invariant #2.
    compiler.emit_grad_retention_arena()?;
    compiler.declare_runtime_functions()?;
    compiler.declare_user_functions(&ast.stmts)?;
    // M56 Task 17: declare agent method FuncIds (Linkage::Local mirrors non-export modules).
    compiler.declare_agent_methods(&ast.stmts, cranelift_module::Linkage::Local)?;
    let vmap_results = compiler.apply_vmap_transforms(ast);
    compiler.register_batched_functions(&vmap_results);
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    crate::wrga_prescan::prescan_adapter_sites_from_decorators(&mut compiler);
    crate::wrga_prescan::rewrite_model_method_bodies_with_adapter_sites(&mut compiler);
    compiler.compile_flash_attention_kernels(&ast.stmts)?;
    // Swallow the result intentionally.  The splice runs BEFORE
    // `compile_call_by_name` inside `ExprKind::Pipe` lowering, so the counter
    // is valid even if a later pipe target's function-call lookup fails — a
    // common case in minimal test fixtures whose pipe RHS identifiers
    // resolve to model fields rather than registered functions.  All earlier
    // setup steps still propagate errors via `?` above, so a genuine
    // infrastructure failure (Compiler::new, emit_retention_arena,
    // declare_runtime_functions, etc.) is still surfaced as an Err.
    let _ = compiler.compile_user_functions(&ast.stmts);
    // M56 Task 17: compile agent method bodies.
    let _ = compiler.compile_agent_methods(&ast.stmts);

    Ok(compiler.retention_splices_emitted)
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
    install_calibration_compile_bundle(&mut compiler, ast, interner, type_map);

    // M52: load weights if --weights was provided.
    load_and_register_weights_if_needed(&mut compiler, options)?;

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
    // M56 Task 17: compute agent struct layouts.
    compiler.collect_agents(&ast.stmts)?;
    // CPDT Part III: non-WGGO MoE dead-expert prune. Runs here (not under the
    // WGGO-gated invoke_cpdt_if_enabled) so it's reachable with --cpdt --weights
    // alone. No-op unless cpdt Full + a @moe config + a loaded WeightMap.
    // CPDT §6.1 — `@cpdt(...)` train-block decorator (must precede the
    // mode/cluster-reading passes below so the decorator's source-level
    // configuration is authoritative when present).
    let cpdt_decor_outcome = crate::cpdt_decorator::apply_cpdt_decorator_from_ast(
        ast, interner, &mut compiler,
    );
    crate::cpdt_decorator::report_outcome(&cpdt_decor_outcome);
    crate::cpdt_expert_prune::run_moe_prune_pass(&mut compiler);
    // CPDT §4.1 — roofline-derived capacity-factor override. Runs after the
    // prune pass (so n_live is settled in moe_configs) and before any body
    // codegen (the moe_dispatch lowering reads MoeInfo.capacity_factor).
    // Same non-WGGO blocker-sidestep posture as the prune pass.
    crate::cpdt_moe_capacity::run_moe_capacity_pass(&mut compiler);
    populate_calibration_retention_from_ast_if_unset(&mut compiler, ast, interner)?;
    // Task 4: declare the calibration retention arena BEFORE method-body
    // codegen.  The pipe-site splice in `try_emit_retention_splice` early-
    // returns when `retention_arena_data_id` is `None`, so emitting the
    // arena later (after `compile_user_functions`) made every splice site
    // silently no-op.  No-op when `calibration_retention` is `None`, so
    // shipped binaries are unaffected.
    compiler.emit_retention_arena()?;
    // Task 10: backward (WGGO grad) sibling arena — spec §7.2 ordering invariant #2.
    compiler.emit_grad_retention_arena()?;
    compiler.declare_runtime_functions()?;
    compiler.declare_user_functions(&ast.stmts)?;
    // M56 Task 17: declare agent method FuncIds.
    compiler.declare_agent_methods(&ast.stmts, cranelift_module::Linkage::Local)?;
    // M39b: Apply vmap AST transforms and register batched function variants
    let vmap_results = compiler.apply_vmap_transforms(ast);
    compiler.register_batched_functions(&vmap_results);
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    // B.2.1 Task 5.5: pre-populate adapter_sites + last_wrga_plan from the
    // user-facing @adapter decorators BEFORE model methods (e.g. `forward`)
    // are compiled. Without this, the Task 3 LoRA AST rewrite never fires
    // because adapter_sites is empty at that point (train-block WRGA
    // invocation happens later, inside compile_main).
    crate::wrga_prescan::prescan_adapter_sites_from_decorators(&mut compiler);
    // B.3.2 Option 3 phase 3e: re-apply the adapter rewrite to
    // `model_method_bodies` (the source-AD-walked copy) so the fused FFI
    // call is visible in the code path source-AD traverses during train
    // blocks. Mirrors the rewrite inside `compile_user_functions`.
    crate::wrga_prescan::rewrite_model_method_bodies_with_adapter_sites(&mut compiler);
    compiler.compile_flash_attention_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    // M56 Task 17: compile agent method bodies.
    compiler.compile_agent_methods(&ast.stmts)?;
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
    if compiler.compile_options.wcet.enabled {
        compiler.run_wcet_analysis()?;
    }

    // M52: Embed weight hash if weights were loaded
    compiler.embed_weight_hash()?;
    let plan = compiler.last_wrga_plan.clone();
    // M62: Emit C-ABI wrapper bodies for @export functions before finalize.
    compiler.emit_export_wrappers()?;
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
    install_calibration_compile_bundle(&mut compiler, ast, interner, type_map);

    compiler.dump_ir = dump_ir;

    // Run every pass up to (but not including) finalize so we can observe
    // `last_wrga_plan` even on an error path before consuming the compiler.
    let pre_finalize = (|| -> Result<(), CodegenError> {
        // M52: load weights if --weights was provided.
        load_and_register_weights_if_needed(&mut compiler, options)?;

        compiler.intern_string("")?;
        compiler.collect_strings(&ast.stmts)?;
        compiler.collect_enums(&ast.stmts)?;
        compiler.collect_structs(&ast.stmts)?;
        compiler.collect_models(&ast.stmts)?;
        // M56 Task 17: compute agent struct layouts.
        compiler.collect_agents(&ast.stmts)?;
        populate_calibration_retention_from_ast_if_unset(&mut compiler, ast, interner)?;
        // Task 4: declare the calibration retention arena BEFORE method-body
        // codegen — see `compile_returning_plan` for the full rationale.
        compiler.emit_retention_arena()?;
        // Task 10: backward (WGGO grad) sibling arena — spec §7.2 ordering invariant #2.
        compiler.emit_grad_retention_arena()?;
        compiler.declare_runtime_functions()?;
        compiler.declare_user_functions(&ast.stmts)?;
        // M56 Task 17: declare agent method FuncIds.
        compiler.declare_agent_methods(&ast.stmts, cranelift_module::Linkage::Local)?;
        let vmap_results = compiler.apply_vmap_transforms(ast);
        compiler.register_batched_functions(&vmap_results);
        compiler.compile_datatype_defs(&ast.stmts)?;
        compiler.compile_kernels(&ast.stmts)?;
        crate::wrga_prescan::prescan_adapter_sites_from_decorators(&mut compiler);
        // B.3.2 Option 3 phase 3e: re-apply rewrite to model_method_bodies
        // so source-AD's inline expansion sees the fused FFI call.
        crate::wrga_prescan::rewrite_model_method_bodies_with_adapter_sites(&mut compiler);
        compiler.compile_flash_attention_kernels(&ast.stmts)?;
        compiler.compile_user_functions(&ast.stmts)?;
        // M56 Task 17: compile agent method bodies.
        compiler.compile_agent_methods(&ast.stmts)?;
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
        if compiler.compile_options.wcet.enabled {
            compiler.run_wcet_analysis()?;
        }

        // M52: Embed weight hash if weights were loaded
        compiler.embed_weight_hash()?;
        // M62: Emit C-ABI wrapper bodies for @export functions before finalize.
        compiler.emit_export_wrappers()?;
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
                cfg.backend = match compiler.compile_options.zk.backend.to_lowercase().as_str() {
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
                cfg.field = match compiler.compile_options.zk.field.to_lowercase().as_str() {
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
                cfg.emit_solidity = compiler.compile_options.zk.solidity;
                // Wire --zk-weights flag to load weight file for witness generation
                if let Some(ref weights_path) = compiler.compile_options.zk.weights_path {
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
    install_calibration_compile_bundle(&mut compiler, ast, interner, type_map);
    compiler.dump_ir = dump_ir;
    compiler.standalone_config = Some(config);
    let pre_finalize = (|| -> Result<(), CodegenError> {
        compiler.intern_string("")?;
        compiler.collect_strings(&ast.stmts)?;
        compiler.collect_enums(&ast.stmts)?;
        compiler.collect_structs(&ast.stmts)?;
        compiler.collect_models(&ast.stmts)?;
        // M56 Task 17: compute agent struct layouts.
        compiler.collect_agents(&ast.stmts)?;
        populate_calibration_retention_from_ast_if_unset(&mut compiler, ast, interner)?;
        // Task 4: declare the calibration retention arena BEFORE method-body
        // codegen — see `compile_returning_plan` for the full rationale.
        compiler.emit_retention_arena()?;
        // Task 10: backward (WGGO grad) sibling arena — spec §7.2 ordering invariant #2.
        compiler.emit_grad_retention_arena()?;
        compiler.declare_runtime_functions()?;
        compiler.declare_user_functions(&ast.stmts)?;
        // M56 Task 17: declare agent method FuncIds.
        compiler.declare_agent_methods(&ast.stmts, cranelift_module::Linkage::Local)?;
        let vmap_results = compiler.apply_vmap_transforms(ast);
        compiler.register_batched_functions(&vmap_results);
        compiler.compile_datatype_defs(&ast.stmts)?;
        compiler.compile_kernels(&ast.stmts)?;
        crate::wrga_prescan::prescan_adapter_sites_from_decorators(&mut compiler);
        // B.3.2 Option 3 phase 3e: re-apply rewrite to model_method_bodies
        // so source-AD's inline expansion sees the fused FFI call.
        crate::wrga_prescan::rewrite_model_method_bodies_with_adapter_sites(&mut compiler);
        compiler.compile_flash_attention_kernels(&ast.stmts)?;
        compiler.compile_user_functions(&ast.stmts)?;
        // M56 Task 17: compile agent method bodies.
        compiler.compile_agent_methods(&ast.stmts)?;
        compiler.compile_batched_functions(&vmap_results)?;
        compiler.compile_standalone_main(&ast.stmts)?;
        compiler.compile_pending_lambdas()?;
        if compiler.compile_options.wcet.enabled {
            compiler.run_wcet_analysis()?;
        }
        // M62: Emit C-ABI wrapper bodies for @export functions before finalize.
        compiler.emit_export_wrappers()?;
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
    install_calibration_compile_bundle(&mut compiler, ast, interner, type_map);
    compiler.dump_ir = dump_ir;
    compiler.intern_string("")?;
    compiler.collect_strings(&ast.stmts)?;
    compiler.collect_enums(&ast.stmts)?;
    compiler.collect_structs(&ast.stmts)?;
    compiler.collect_models(&ast.stmts)?;
    // M56 Task 17: compute agent struct layouts.
    compiler.collect_agents(&ast.stmts)?;
    populate_calibration_retention_from_ast_if_unset(&mut compiler, ast, interner)?;
    // Task 4: declare the calibration retention arena BEFORE method-body
    // codegen — parity with all other entry points.  `compile_test` compiles
    // `@test` functions; a `@test` body that calls a model.forward with a
    // pipe site would otherwise no-op the splice if the caller set
    // `calibration_retention` on CompileOptions.
    compiler.emit_retention_arena()?;
    // Task 10: backward (WGGO grad) sibling arena — spec §7.2 ordering invariant #2.
    compiler.emit_grad_retention_arena()?;
    compiler.declare_runtime_functions()?;
    compiler.declare_user_functions(&ast.stmts)?;
    // M56 Task 17: declare agent method FuncIds.
    compiler.declare_agent_methods(&ast.stmts, cranelift_module::Linkage::Local)?;
    // M39b: Apply vmap AST transforms and register batched function variants
    let vmap_results = compiler.apply_vmap_transforms(ast);
    compiler.register_batched_functions(&vmap_results);
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    compiler.compile_flash_attention_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    // M56 Task 17: compile agent method bodies.
    compiler.compile_agent_methods(&ast.stmts)?;
    // M39c: Compile batched function bodies
    compiler.compile_batched_functions(&vmap_results)?;
    compiler.compile_pending_lambdas()?;
    let test_fns = compiler.registry.test_fns.clone();
    if test_fns.is_empty() {
        return Err(CodegenError::new("no @test functions found".to_string()));
    }
    compiler.compile_test_main()?;
    // M62: Emit C-ABI wrapper bodies for @export functions before finalize.
    compiler.emit_export_wrappers()?;
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
    let (result, wrga_plan, _cfie_plan) = compile_module_with_imports_best_effort_plans(
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
    (result, wrga_plan)
}

/// Like [`compile_module_with_imports_best_effort_plan`] but also
/// surfaces the CFIE plan produced while compiling any serve block
/// (CFIE Tier-A wiring observability, mirroring the WRGA pattern).
#[allow(clippy::too_many_arguments)]
#[allow(clippy::type_complexity)]
pub fn compile_module_with_imports_best_effort_plans(
    ast: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    module_prefix: &str,
    imported_fns: &[(String, String, Signature)],
    imported_struct_layouts: HashMap<String, crate::context::StructLayout>,
    imported_model_names: HashSet<String>,
    dump_ir: bool,
    options: &crate::CompileOptions,
) -> (
    Result<Vec<u8>, CodegenError>,
    Option<crate::wrga::WrgaPlan>,
    Option<crate::cfie::CfiePlan>,
) {
    let mut compiler = match Compiler::new(interner, type_map, options) {
        Ok(c) => c,
        Err(e) => return (Err(e), None, None),
    };
    install_calibration_compile_bundle(&mut compiler, ast, interner, type_map);
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
        // M56 Task 17: compute agent struct layouts.
        compiler.collect_agents(&ast.stmts)?;
        populate_calibration_retention_from_ast_if_unset(&mut compiler, ast, interner)?;
        // Task 4: declare the calibration retention arena BEFORE method-body
        // codegen — see `compile_returning_plan` for the full rationale.
        compiler.emit_retention_arena()?;
        // Task 10: backward (WGGO grad) sibling arena — spec §7.2 ordering invariant #2.
        compiler.emit_grad_retention_arena()?;
        compiler.declare_runtime_functions()?;
        compiler.declare_imported_functions(imported_fns)?;
        compiler.declare_user_functions_with_linkage(&ast.stmts, Linkage::Export)?;
        // M56 Task 17: declare agent method FuncIds (Export linkage mirrors module compile).
        compiler.declare_agent_methods(&ast.stmts, Linkage::Export)?;
        let vmap_results = compiler.apply_vmap_transforms(ast);
        compiler.register_batched_functions(&vmap_results);
        compiler.compile_datatype_defs(&ast.stmts)?;
        compiler.compile_kernels(&ast.stmts)?;
        crate::wrga_prescan::prescan_adapter_sites_from_decorators(&mut compiler);
        // B.3.2 Option 3 phase 3e: re-apply rewrite to model_method_bodies
        // so source-AD's inline expansion sees the fused FFI call.
        crate::wrga_prescan::rewrite_model_method_bodies_with_adapter_sites(&mut compiler);
        compiler.compile_flash_attention_kernels(&ast.stmts)?;
        compiler.compile_user_functions(&ast.stmts)?;
        // M56 Task 17: compile agent method bodies.
        compiler.compile_agent_methods(&ast.stmts)?;
        compiler.compile_batched_functions(&vmap_results)?;
        compiler.compile_pending_lambdas()?;
        // M62: Emit C-ABI wrapper bodies for @export functions before finalize.
        compiler.emit_export_wrappers()?;
        Ok(())
    })();
    let plan = compiler.last_wrga_plan.clone();
    let cfie_plan = compiler.last_cfie_plan.clone();
    // Dev Tools Phase 2, Task 6: write the profile manifest before finalize.
    if pre_finalize.is_ok() {
        write_manifest_if_needed(&mut compiler, options);
    }
    let result = pre_finalize.and_then(|()| compiler.finalize());
    (result, plan, cfie_plan)
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
    install_calibration_compile_bundle(&mut compiler, ast, interner, type_map);
    compiler.dump_ir = dump_ir;

    // M52 / CPDT Phase 1: load weights if --weights was provided.  Load-bearing
    // for the multi-file build path: without this call, `compile_main` sees
    // `compiler.features.weight_map == None` and CPDT Phase 1 silently
    // degrades to the no-weights tier-assignment path for every user program
    // that has `from X import Y` or a `train(...)` block.  Equivalent single-
    // file path runs through `compile_returning_plan` which calls the same
    // helper.
    load_and_register_weights_if_needed(&mut compiler, options)?;

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
    // M56 Task 17: compute agent struct layouts.
    compiler.collect_agents(&ast.stmts)?;

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
    populate_calibration_retention_from_ast_if_unset(&mut compiler, ast, interner)?;
    // Task 4: declare the calibration retention arena BEFORE method-body
    // codegen — see `compile_returning_plan` for the full rationale.
    compiler.emit_retention_arena()?;
    // Task 10: backward (WGGO grad) sibling arena — spec §7.2 ordering invariant #2.
    compiler.emit_grad_retention_arena()?;
    compiler.declare_runtime_functions()?;
    compiler.declare_imported_functions(imported_fns)?;
    compiler.declare_user_functions_with_linkage(&ast.stmts, Linkage::Export)?;
    // M56 Task 17: declare agent method FuncIds (Export linkage for entry compile).
    compiler.declare_agent_methods(&ast.stmts, Linkage::Export)?;
    // M39b: Apply vmap AST transforms and register batched function variants
    let vmap_results = compiler.apply_vmap_transforms(ast);
    compiler.register_batched_functions(&vmap_results);
    compiler.compile_datatype_defs(&ast.stmts)?;
    compiler.compile_kernels(&ast.stmts)?;
    crate::wrga_prescan::prescan_adapter_sites_from_decorators(&mut compiler);
    // B.3.2 Option 3 phase 3e: re-apply rewrite to model_method_bodies
    // so source-AD's inline expansion sees the fused FFI call.
    crate::wrga_prescan::rewrite_model_method_bodies_with_adapter_sites(&mut compiler);
    compiler.compile_flash_attention_kernels(&ast.stmts)?;
    compiler.compile_user_functions(&ast.stmts)?;
    // M56 Task 17: compile agent method bodies.
    compiler.compile_agent_methods(&ast.stmts)?;
    // M39c: Compile batched function bodies
    compiler.compile_batched_functions(&vmap_results)?;
    compiler.compile_main(&ast.stmts)?;
    compiler.compile_pending_lambdas()?;
    // M53: Run WCET analysis for @real_time functions
    if compiler.compile_options.wcet.enabled {
        compiler.run_wcet_analysis()?;
    }
    // M31: Print fusion report if enabled
    if compiler.fusion.report_enabled {
        crate::fusion_report::print_fusion_report(
            &compiler.fusion.events,
            &compiler.fusion.barriers,
        );
    }
    // M52: embed weight hash if weights were loaded (parity with single-file
    // `compile_returning_plan`).  Called before the `last_wrga_plan` clone so
    // that if a future change makes `embed_weight_hash` stash anything into
    // the plan, the returned plan reflects it — matches the single-file path.
    compiler.embed_weight_hash()?;
    let plan = compiler.last_wrga_plan.clone();
    // M62: Emit C-ABI wrapper bodies for @export functions before finalize.
    compiler.emit_export_wrappers()?;
    // Dev Tools Phase 2, Task 6: write the profile manifest before finalize.
    write_manifest_if_needed(&mut compiler, options);
    let bytes = compiler.finalize()?;
    Ok((bytes, plan))
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_lexer::Interner;

    fn parse_module(source: &str) -> (nsl_ast::Module, Interner) {
        let mut interner = Interner::new();
        let (tokens, lex_diags) =
            nsl_lexer::tokenize(source, nsl_errors::FileId(0), &mut interner);
        assert!(
            lex_diags
                .iter()
                .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
            "fixture must lex cleanly: {lex_diags:?}"
        );
        let parsed = nsl_parser::parse(&tokens, &mut interner);
        assert!(
            parsed
                .diagnostics
                .iter()
                .all(|d| !matches!(d.level, nsl_errors::Level::Error)),
            "fixture must parse cleanly: {:?}",
            parsed.diagnostics
        );
        (parsed.module, interner)
    }

    #[test]
    fn entry_point_populates_calibration_grad_retention_from_ast() {
        let source = include_str!("../../../../tests/fixtures/wggo_attention_mlp.nsl");
        let opts = crate::CompileOptions::default();
        let (ast, interner) = parse_module(source);
        let resolved_opts = run_pre_scan_phase(&ast, &interner, opts);
        let targets = resolved_opts
            .calibration_grad_retention
            .expect("@wggo_target decorators should populate calibration_grad_retention");
        assert!(
            !targets.is_empty(),
            "@wggo_target decorators should populate the field"
        );
        assert_eq!(targets[0].class_name, "TinyAttn");
        assert_eq!(targets[0].layer_key, "m");
    }

    #[test]
    fn run_pre_scan_phase_does_not_overwrite_existing_grad_retention() {
        let source = include_str!("../../../../tests/fixtures/wggo_attention_mlp.nsl");
        let (ast, interner) = parse_module(source);
        // Pre-populate with a sentinel empty vec to verify idempotency.
        let mut opts = crate::CompileOptions::default();
        opts.calibration_grad_retention = Some(Vec::new());
        let resolved_opts = run_pre_scan_phase(&ast, &interner, opts);
        // Must remain the caller-supplied empty vec, not the discovered targets.
        assert_eq!(
            resolved_opts.calibration_grad_retention,
            Some(Vec::new()),
            "run_pre_scan_phase must not overwrite a pre-populated field"
        );
    }

    // -----------------------------------------------------------------------
    // §5.7 auto-mode soft-fallback tests
    // -----------------------------------------------------------------------

    /// §5.4 condition: no @wggo_target decorators → Auto demotes to Magnitude.
    #[test]
    fn auto_mode_demotes_to_magnitude_when_no_decorators_in_ast() {
        // Plain model with no @wggo_target decorators.
        let source =
            "model NoAttn:\n    weight: Tensor = zeros([4, 4])\n\nfn main():\n    let m = NoAttn()\n";
        let mut opts = crate::CompileOptions::default();
        opts.wggo.importance = crate::WggoImportance::Auto;
        // No calibration_data, no decorators.
        let (ast, interner) = parse_module(source);
        let resolved = run_pre_scan_phase(&ast, &interner, opts);
        assert_eq!(
            resolved.wggo.importance,
            crate::WggoImportance::Magnitude,
            "§5.7: Auto must demote to Magnitude when no @wggo_target decorators are present"
        );
    }

    /// §5.6 condition: decorators present + model reachable, but no calibration data
    /// → Auto demotes to Magnitude.
    #[test]
    fn auto_mode_demotes_to_magnitude_when_no_calibration_data() {
        // Fixture has @wggo_target + a main() that instantiates the decorated model.
        let source = include_str!("../../../../tests/fixtures/wggo_attention_mlp.nsl");
        let mut opts = crate::CompileOptions::default();
        opts.wggo.importance = crate::WggoImportance::Auto;
        // calibration_data is None (default) — §5.6 fires.
        let (ast, interner) = parse_module(source);
        let resolved = run_pre_scan_phase(&ast, &interner, opts);
        assert_eq!(
            resolved.wggo.importance,
            crate::WggoImportance::Magnitude,
            "§5.7: Auto must demote to Magnitude when calibration data is absent"
        );
    }

    /// When wggo_importance is NOT Auto the fallback note must not fire
    /// (Grad and Magnitude modes are unaffected).
    #[test]
    fn non_auto_modes_are_not_demoted_by_fallback_note() {
        let source =
            "model NoAttn:\n    weight: Tensor = zeros([4, 4])\n\nfn main():\n    let m = NoAttn()\n";
        let (ast, interner) = parse_module(source);

        for importance in [crate::WggoImportance::Grad, crate::WggoImportance::Magnitude] {
            let mut opts = crate::CompileOptions::default();
            opts.wggo.importance = importance;
            let resolved = run_pre_scan_phase(&ast, &interner, opts);
            assert_eq!(
                resolved.wggo.importance, importance,
                "§5.7 fallback must not alter non-Auto mode {importance:?}"
            );
        }
    }
}
