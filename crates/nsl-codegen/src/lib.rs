pub mod autotune;
pub mod builtins;
pub mod calibration;
pub mod compiler;
pub mod context;
pub mod context_parallel;
pub mod cost_model;
pub mod deterministic_kernels;
pub mod profiling;
pub mod grammar_compiler;
pub mod schema_convert;
pub mod stdlib_loader;

pub mod ad_rules;
pub mod backend_amdgpu;
pub mod backend_metal;
pub mod backend_ptx;
pub mod backend_wgsl;
pub mod cep;
pub mod cep_importance;
pub mod cep_oracle;
pub mod cep_rewrite;
pub mod cep_search;
pub mod cfie;
pub mod cfie_fused_sample;
pub mod cfie_grammar;
pub mod cfie_kv_plan;
pub mod cfie_kv_quant;
pub mod cfie_persistent;
pub mod cfie_speculative;
pub mod cpdt;
pub mod cpdt_comm;
pub mod cpdt_expert;
pub mod cpdt_joint;
pub mod cpdt_optim;
pub mod cpdt_precision;
pub mod cpdt_zero;
pub mod csha;
pub mod csha_apply;
pub mod csha_boundary;
pub mod csha_pipeline;
pub mod csha_specialize;
pub mod dynamic_shapes;
pub mod epilogue_fusion;
pub mod error;
pub mod expr;
pub mod fase;
pub mod fase_clip;
pub mod fase_memory;
pub mod fase_optimizer;
pub mod flash_attention;
pub mod fp8;
pub mod func;
pub mod fusion;
pub mod fusion_graph;
pub mod fusion_report;
pub mod gpu_specs;
pub mod gpu_target;
pub mod inspect;
pub mod kernel;
pub mod kernel_ir;
pub mod kernel_lower;
pub mod linker;
pub mod memory_planner;
pub mod moe;
pub mod moe_kernels;
pub mod multimodal;
pub mod ffi_ownership;
pub mod ownership;
pub mod ownership_expr;
pub mod pca_detect;
pub mod pca_rope;
pub mod pca_segment;
pub mod pca_tileskip;
pub mod pipeline;
pub mod reduction_fusion;
pub mod serve;
pub mod source_ad;
pub mod sparse;
pub mod speculative;
pub mod standalone;
pub mod stmt;
pub mod tensor_parallel;
pub mod types;
pub mod unikernel;
pub mod unikernel_boot;
pub mod use_count;
pub mod vmap;
pub mod wcet;
pub mod weight_aware;
pub mod wengert;
pub mod wengert_lower;
pub mod wggo;
pub mod wggo_apply;
pub mod wggo_conflicts;
pub mod wggo_cost;
pub mod wggo_dp;
pub mod wggo_graph;
pub mod wggo_ilp;
pub mod wggo_schedule;
pub mod wggo_weight_analysis;
pub mod wggo_weight_analysis_cache;
pub mod wggo_weight_analysis_nslweights;
pub mod wrga;
pub mod matmul_mma;
pub mod wrga_adapter_init;
pub mod wrga_adapter_inject;
pub mod wrga_adapter_rewrite;
pub mod wrga_fused_ptx;
pub mod wrga_fusion;
pub mod wrga_memory;
pub mod wrga_prescan;
pub mod wrga_prune;
pub mod wrga_roofline;
pub mod wrga_spectral;
pub mod zk;

#[cfg(any(test, feature = "test-helpers"))]
pub mod test_helpers;

pub use compiler::{
    compile, compile_entry, compile_module, compile_module_with_imports,
    compile_entry_returning_plan, compile_module_with_imports_returning_plan,
    compile_returning_plan, compile_standalone, compile_standalone_returning_plan,
    compile_test, compile_with_zk_info, compile_with_zk_info_returning_plan,
    StandaloneConfig,
};

/// Task 4 test helper: compile a module and return any `WrgaPlan` produced
/// during `@train` block lowering.  The plan is returned even when codegen
/// fails *after* `wrga::run` has fired — this is deliberate: WRGA's observable
/// effect is the stashed plan, independent of whether the full object file
/// links.
#[doc(hidden)]
pub fn debug_compile_and_return_plan_from_ast(
    ast: &nsl_ast::Module,
    interner: &nsl_lexer::Interner,
    type_map: &nsl_semantic::checker::TypeMap,
    options: &CompileOptions,
) -> Result<Option<crate::wrga::WrgaPlan>, CodegenError> {
    debug_compile_and_return_plan_with_imports(ast, interner, type_map, &[], options)
}

fn debug_compile_and_return_plan_with_imports(
    ast: &nsl_ast::Module,
    interner: &nsl_lexer::Interner,
    type_map: &nsl_semantic::checker::TypeMap,
    imported_fns: &[(String, String, cranelift_codegen::ir::Signature)],
    options: &CompileOptions,
) -> Result<Option<crate::wrga::WrgaPlan>, CodegenError> {
    let (res, plan) = crate::compiler::compile_module_with_imports_best_effort_plan(
        ast,
        interner,
        type_map,
        "",
        imported_fns,
        HashMap::new(),
        std::collections::HashSet::new(),
        false,
        options,
    );
    match (res, plan) {
        (Ok(_), plan) => Ok(plan),
        (Err(e), Some(plan)) => {
            eprintln!(
                "[debug_compile_and_return_plan] codegen failed but a WRGA plan was produced: {}",
                e.message
            );
            Ok(Some(plan))
        }
        (Err(e), None) => Err(e),
    }
}

/// B.2 Task 1 test helper: compile a raw NSL source string and return any
/// `WrgaPlan` produced.  Loads stdlib optimizer signatures as needed so that
/// real `train` blocks can be compiled in-process without an on-disk workspace.
#[doc(hidden)]
pub fn debug_compile_and_return_plan(
    src: &str,
    options: &CompileOptions,
) -> Result<Option<crate::wrga::WrgaPlan>, CodegenError> {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    if lex_diags
        .iter()
        .any(|d| matches!(d.level, nsl_errors::Level::Error))
    {
        return Err(CodegenError::new(format!(
            "lex errors: {:?}",
            lex_diags
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        )));
    }
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    if parsed
        .diagnostics
        .iter()
        .any(|d| matches!(d.level, nsl_errors::Level::Error))
    {
        return Err(CodegenError::new(format!(
            "parse errors: {:?}",
            parsed
                .diagnostics
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        )));
    }

    // Build imported_fns for any stdlib optimizer modules referenced in
    // train-block `optimizer:` sections.  This mutates the interner, so it
    // must run before semantic analysis / codegen (which borrow immutably).
    // Semantic analysis doesn't need the stdlib imports here because the
    // train block's optimizer expression is captured structurally and not
    // type-checked as a regular call — matching the existing baseline tests.
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    if analysis
        .diagnostics
        .iter()
        .any(|d| matches!(d.level, nsl_errors::Level::Error))
    {
        return Err(CodegenError::new(format!(
            "semantic errors: {:?}",
            analysis
                .diagnostics
                .iter()
                .map(|d| d.message.clone())
                .collect::<Vec<_>>()
        )));
    }

    let imported_fns = crate::stdlib_loader::build_imported_fns_for_entry(
        &parsed.module,
        &mut interner,
        &analysis.type_map,
        options,
    )?;

    debug_compile_and_return_plan_with_imports(
        &parsed.module,
        &interner,
        &analysis.type_map,
        &imported_fns,
        options,
    )
}
#[doc(hidden)]
pub mod debug_channels {
    use std::cell::Cell;
    thread_local! {
        pub static ADJOINT_OPS_DROPPED: Cell<Option<usize>> = const { Cell::new(None) };
        pub static ALLOC_SLOTS_PRE_HINT: Cell<Option<usize>> = const { Cell::new(None) };
        pub static ALLOC_SLOTS_POST_HINT: Cell<Option<usize>> = const { Cell::new(None) };
        pub static CONSUME_HINTS_CALLS: Cell<usize> = const { Cell::new(0) };
    }
}

#[doc(hidden)]
pub fn debug_bump_consume_hints_calls() {
    debug_channels::CONSUME_HINTS_CALLS.with(|c| c.set(c.get() + 1));
}

#[doc(hidden)]
pub fn debug_last_consume_hints_calls() -> Option<usize> {
    let n = debug_channels::CONSUME_HINTS_CALLS.with(|c| c.get());
    if n == 0 { None } else { Some(n) }
}

#[doc(hidden)]
pub fn debug_reset_consume_hints_calls() {
    debug_channels::CONSUME_HINTS_CALLS.with(|c| c.set(0));
}

#[doc(hidden)]
pub fn debug_set_adjoint_ops_dropped(n: usize) {
    debug_channels::ADJOINT_OPS_DROPPED.with(|c| c.set(Some(n)));
}

#[doc(hidden)]
pub fn debug_last_adjoint_ops_dropped() -> Option<usize> {
    debug_channels::ADJOINT_OPS_DROPPED.with(|c| c.get())
}

#[doc(hidden)]
pub fn debug_set_allocator_slot_count_pre_hint(n: usize) {
    debug_channels::ALLOC_SLOTS_PRE_HINT.with(|c| c.set(Some(n)));
}
#[doc(hidden)]
pub fn debug_last_allocator_slot_count_pre_hint() -> Option<usize> {
    debug_channels::ALLOC_SLOTS_PRE_HINT.with(|c| c.get())
}
#[doc(hidden)]
pub fn debug_set_allocator_slot_count_post_hint(n: usize) {
    debug_channels::ALLOC_SLOTS_POST_HINT.with(|c| c.set(Some(n)));
}
#[doc(hidden)]
pub fn debug_last_allocator_slot_count_post_hint() -> Option<usize> {
    debug_channels::ALLOC_SLOTS_POST_HINT.with(|c| c.get())
}
#[doc(hidden)]
pub fn debug_clear_allocator_slot_channels() {
    debug_channels::ALLOC_SLOTS_PRE_HINT.with(|c| c.set(None));
    debug_channels::ALLOC_SLOTS_POST_HINT.with(|c| c.set(None));
}

pub use error::CodegenError;
pub use standalone::create_weight_object;
pub use wrga_fusion::{FusionDecision, FusionPlan, FusionTarget};

use std::collections::HashMap;

/// Decorator configs captured by nsl-semantic and forwarded to
/// `nsl_codegen::wrga::run` during `@train` block compilation.
///
/// Derived from `nsl_semantic::AnalysisResult.{wrga,freeze,adapter}_configs`.
/// Kept as a codegen-side newtype so nsl-codegen does not depend on
/// nsl-semantic (mirroring how `imported_fns` is passed as raw signatures).
#[derive(Debug, Clone, Default)]
pub struct WrgaInputs {
    /// Validated `@wrga(...)` configs (one per occurrence).
    pub wrga: Vec<WrgaDecoratorConfig>,
    /// Validated `@freeze(...)` configs.
    pub freeze: Vec<FreezeDecoratorConfig>,
    /// Validated `@adapter(...)` configs.
    pub adapter: Vec<AdapterDecoratorConfig>,
}

#[derive(Debug, Clone)]
pub struct WrgaDecoratorConfig {
    pub mode: nsl_ast::block::WrgaMode,
    pub budget: Option<i64>,
    pub target: Option<String>,
    pub layers: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct FreezeDecoratorConfig {
    pub exclude: Vec<String>,
    pub include: Vec<String>,
}

#[derive(Debug, Clone)]
pub struct AdapterDecoratorConfig {
    pub kind: AdapterKind,
    pub targets: Vec<String>,
    pub rank: Option<i64>,
    pub alpha: Option<i64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdapterKind {
    Lora,
    Ia3,
    GatedLora,
}

/// Compiler configuration flags passed from CLI.
#[derive(Clone)]
pub struct CompileOptions {
    pub no_autotune: bool,
    pub autotune_fresh: bool,
    pub world_size: usize,
    pub fusion_report: bool,
    /// M36: VRAM budget in bytes (None = no limit, Some(n) = fail if plan exceeds n)
    pub vram_budget: Option<u64>,
    /// M36: Print memory plan report to stderr
    pub memory_report: bool,
    /// M47: GPU compilation target name.
    pub target: String,
    /// Disable all fusion optimizations (for differential testing baseline).
    pub disable_fusion: bool,
    /// M40b: Force tape-based AD (disable source-to-source AD).
    pub tape_ad: bool,
    /// M40: Use compile-time source-to-source AD for training (default: false = tape AD).
    pub source_ad: bool,
    /// M45: Enable tensor operation tracing.
    pub trace_ops: bool,
    /// M45: Enable compile-time NaN risk analysis.
    pub nan_analysis: bool,
    /// M46: Enable deterministic mode.
    pub deterministic: bool,
    /// M52: Path to safetensors weight file for weight-aware compilation
    pub weight_file: Option<std::path::PathBuf>,
    /// M52: Weight-aware compilation configuration
    pub weight_config: weight_aware::WeightAwareConfig,
    /// M52: Whether to emit a weight analysis report (nsl check --weight-analysis)
    pub weight_analysis: bool,
    /// M54: Unikernel build configuration (None = normal build)
    pub unikernel_config: Option<crate::unikernel::UnikernelConfig>,
    /// M53: Enable WCET analysis for @real_time functions
    pub wcet_enabled: bool,
    /// M53: GPU target name for WCET analysis (e.g., "Orin", "H100")
    pub wcet_gpu: Option<String>,
    /// M53: CPU target name for WCET analysis (e.g., "cortex-a78")
    pub wcet_cpu: Option<String>,
    /// M53: Path to write WCET certificate JSON
    pub wcet_report_path: Option<std::path::PathBuf>,
    /// M53: Safety margin multiplier for WCET (default: 1.05 = 5%)
    pub wcet_safety_margin: f64,
    /// M53: Path to write DO-178C compliance report
    pub do178c_report: Option<std::path::PathBuf>,
    /// M53: WCET target type: "gpu" (statistical), "fpga" (certified), "groq" (blocked)
    pub wcet_target: String,
    /// M53: FPGA device name for certified WCET (e.g., "xcvu440", "xczu9eg")
    pub fpga_device: Option<String>,
    /// M38a: Enable linear types ownership checking.
    pub linear_types_enabled: bool,
    /// M38a: Per-function ownership metadata from semantic analysis.
    /// Keys are function names, values have linear_params and shared_params.
    pub ownership_info: HashMap<String, crate::ownership::FunctionOwnership>,
    /// M55: Emit a ZK inference circuit alongside compiled output.
    pub zk_circuit: bool,
    /// M55: ZK backend to use ("folding", "halo2", or "plonky3").
    pub zk_backend: String,
    /// M55: ZK field to use ("m31" or "bn254").
    pub zk_field: String,
    /// M55: Also emit a Solidity verifier contract.
    pub zk_solidity: bool,
    /// M55: Path to safetensors weight file used as ZK witness.
    pub zk_weights_path: Option<std::path::PathBuf>,
    /// M43b: ZeRO optimizer sharding stage (1, 2, or 3)
    pub zero_stage: Option<u8>,
    /// Debug training mode: disables fusion, disables FBIP, and emits
    /// gradient checksum assertions after each backward pass.
    pub debug_training: bool,
    /// M62a: Build as a shared library (.so/.dylib/.dll) instead of an executable.
    pub shared_lib: bool,
    /// WRGA: decorator configs forwarded from nsl-semantic (Task 1 of bridge).
    pub wrga_inputs: Option<WrgaInputs>,
    /// WRGA Milestone B.2 Task 3: fold WRGA memory hints into real
    /// allocations (vs. B.1's observational-only path). Default false.
    pub wrga_fold_allocations: bool,
    /// WGGO: global optimization mode ("full", "greedy", "off"). When `None`,
    /// WGGO is not run.  See `crates/nsl-codegen/src/wggo.rs`.
    pub wggo_mode: Option<String>,
    /// WGGO: print the global-optimization report to stderr.
    pub wggo_report: bool,
    /// Dev Tools Phase 2: enable the kernel-profile pre-pass.
    pub profile_kernels: bool,
    /// Dev Tools Phase 2: target GPU name for the profile walker.
    pub target_gpu: String,
    /// Dev Tools Phase 2: tensor dtype assumed by the profile walker.
    pub dtype: String,
    /// Dev Tools Phase 2, Task 6: manifest output path.
    pub manifest_output_path: Option<std::path::PathBuf>,
    /// Dev Tools Phase 2, Task 6: source text for span line numbers.
    pub profile_source_text: Option<String>,
    /// Dev Tools Phase 2, Task 6: source file name for manifest spans.
    pub profile_source_file_name: Option<String>,
    /// Dev Tools Phase 4, Task 4: enable per-step health hook emission.
    pub health_monitor: bool,
    /// Dev Tools Phase 4, Task 4: optional explicit flush-interval setter.
    pub health_flush_interval: Option<u64>,
    /// Dev Tools Phase 5, Task 7: enable `@inspect` decorator emission.
    pub inspect_enabled: bool,
    /// WGGO Stage 3: path to a `.nslweights` sidecar file.
    pub wggo_weights: Option<std::path::PathBuf>,
    /// WGGO Stage 3: scoring mode ("none", "magnitude").
    pub wggo_importance: Option<String>,
    /// WGGO Stage 3: default fraction of heads allowed to be pruned.
    pub wggo_prune_fraction: Option<f64>,
    /// CSHA: fusion mode.
    pub csha_mode: Option<String>,
    /// CSHA: print the attention-fusion report.
    pub csha_report: bool,
    /// Path to calibration dataset.
    pub calibration_data: Option<std::path::PathBuf>,
    /// Failure-handling mode: `"required"` or `"best-effort"`.
    pub calibration_mode: Option<String>,
    pub calibration_samples: u32,
    pub calibration_batch_size: u32,
    pub calibration_timeout_secs: u64,
    /// Populated by the harness after a successful calibration run.
    /// Downstream passes (AWQ quantizer, future hooks' consumers) read
    /// their sidecar key from here.  `None` when calibration didn't
    /// run or produced no usable output.
    pub calibration_sidecar: Option<crate::calibration::sidecar::Sidecar>,
    /// When `Some`, codegen runs the retention pass on `model_forward`, splicing
    /// memcpys of input activations before matched linear sites. `None` = shipped
    /// binary, zero IR impact.
    pub calibration_retention: Option<Vec<crate::calibration::DiscoveredProjection>>,
    /// Batch count and sequence length read from the calibration-data header
    /// (`peek_batch_seq`).  Must be set whenever `calibration_retention` is
    /// `Some`.  Callers that exercise retention in tests without real calib
    /// data fall back to (8, 4) when this is `None`.
    pub calibration_batch_seq: Option<(u32, u32)>,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            no_autotune: false,
            autotune_fresh: false,
            world_size: 1,
            fusion_report: false,
            vram_budget: None,
            memory_report: false,
            target: "cuda".to_string(),
            disable_fusion: false,
            tape_ad: false,
            source_ad: false,
            trace_ops: false,
            nan_analysis: false,
            deterministic: false,
            weight_file: None,
            weight_config: weight_aware::WeightAwareConfig::default(),
            weight_analysis: false,
            unikernel_config: None,
            wcet_enabled: false,
            wcet_gpu: None,
            wcet_cpu: None,
            wcet_report_path: None,
            wcet_safety_margin: 1.05,
            do178c_report: None,
            wcet_target: "gpu".to_string(),
            fpga_device: None,
            linear_types_enabled: false,
            ownership_info: HashMap::new(),
            zk_circuit: false,
            zk_backend: "folding".to_string(),
            zk_field: "m31".to_string(),
            zk_solidity: false,
            zk_weights_path: None,
            zero_stage: None,
            debug_training: false,
            shared_lib: false,
            wrga_inputs: None,
            wrga_fold_allocations: false,
            wggo_mode: None,
            wggo_report: false,
            profile_kernels: false,
            target_gpu: "h100".to_string(),
            dtype: "bf16".to_string(),
            manifest_output_path: None,
            profile_source_text: None,
            profile_source_file_name: None,
            health_monitor: false,
            health_flush_interval: None,
            inspect_enabled: false,
            wggo_weights: None,
            wggo_importance: None,
            wggo_prune_fraction: None,
            csha_mode: None,
            csha_report: false,
            calibration_data: None,
            calibration_mode: Some("required".to_string()),
            calibration_samples: 512,
            calibration_batch_size: 8,
            calibration_timeout_secs: 600,
            calibration_sidecar: None,
            calibration_retention: None,
            calibration_batch_seq: None,
        }
    }
}

/// Compile an NSL source file, run the AWQ calibration harness, and return
/// the resulting `Sidecar`.
///
/// This is a convenience wrapper that:
/// 1. Reads the calibration-data header to obtain `(batch, seq)`.
/// 2. Sets up `CompileOptions` with `calibration_data`, `weight_file`,
///    `calibration_batch_seq`, and `calibration_mode = "required"`.
/// 3. Lexes, parses, and semantically analyses the source.
/// 4. Constructs a `Compiler` directly and runs all passes.  During
///    `compile_train_block`, `discover_awq_projections` fires and stores the
///    discovered projections in `compile_options.calibration_retention`.
/// 5. After codegen succeeds, loads calibration data and weights from
///    the supplied safetensors files and runs the in-process AWQ observation
///    loop (per-projection forward-pass + per-channel max-abs reduction).
/// 6. Wraps the resulting scales in a `Sidecar` and returns it.
///
/// The NSL source must contain a `train` block and at least one model decorated
/// with `@quantize(dtype="awq4")` whose `forward` method uses `|>` pipe syntax.
pub fn compile_and_calibrate(
    source_path: &std::path::Path,
    data_path: &std::path::Path,
    weights_path: &std::path::Path,
) -> Result<crate::calibration::sidecar::Sidecar, CodegenError> {
    let source = std::fs::read_to_string(source_path).map_err(|e| {
        CodegenError::new(format!("reading source {}: {e}", source_path.display()))
    })?;

    // Step 1: peek at the calibration data to get (batch, seq).
    let (batch, seq) = nsl_runtime::calibration_data::peek_batch_seq(data_path)
        .map_err(|e| CodegenError::new(format!("reading calibration data header: {e}")))?;

    // Step 2: lex, parse, and semantically analyse.
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(&source, nsl_errors::FileId(0), &mut interner);
    if lex_diags.iter().any(|d| matches!(d.level, nsl_errors::Level::Error)) {
        return Err(CodegenError::new(format!(
            "lex errors in {}: {:?}",
            source_path.display(),
            lex_diags.iter().map(|d| d.message.clone()).collect::<Vec<_>>()
        )));
    }
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    if parsed.diagnostics.iter().any(|d| matches!(d.level, nsl_errors::Level::Error)) {
        return Err(CodegenError::new(format!(
            "parse errors in {}: {:?}",
            source_path.display(),
            parsed.diagnostics.iter().map(|d| d.message.clone()).collect::<Vec<_>>()
        )));
    }
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    if analysis.diagnostics.iter().any(|d| matches!(d.level, nsl_errors::Level::Error)) {
        return Err(CodegenError::new(format!(
            "semantic errors in {}: {:?}",
            source_path.display(),
            analysis.diagnostics.iter().map(|d| d.message.clone()).collect::<Vec<_>>()
        )));
    }

    // Step 3: assemble options.
    let mut opts = CompileOptions::default();
    opts.calibration_data = Some(data_path.to_path_buf());
    opts.weight_file = Some(weights_path.to_path_buf());
    opts.calibration_batch_seq = Some((batch, seq));
    opts.calibration_mode = Some("required".to_string());

    // Step 4: Run stdlib imports for any train block references.
    let imported_fns = crate::stdlib_loader::build_imported_fns_for_entry(
        &parsed.module,
        &mut interner,
        &analysis.type_map,
        &opts,
    )?;

    // Step 5: construct the Compiler directly so we can read back
    // `compiler.compile_options.calibration_sidecar` after all passes run.
    // (This is Blocker A's fix: compile_module takes &CompileOptions so the
    // sidecar written into the Compiler's internal copy cannot be recovered
    // via the public API.  Driving Compiler directly avoids this limitation.)
    let mut compiler = crate::compiler::Compiler::new(&interner, &analysis.type_map, &opts)?;

    let pre_finalize = (|| -> Result<(), CodegenError> {
        compiler.intern_string("")?;
        compiler.collect_strings(&parsed.module.stmts)?;
        compiler.collect_enums(&parsed.module.stmts)?;
        compiler.collect_structs(&parsed.module.stmts)?;
        compiler.collect_models(&parsed.module.stmts)?;
        compiler.declare_runtime_functions()?;
        compiler.declare_imported_functions(&imported_fns)?;
        compiler.declare_user_functions_with_linkage(
            &parsed.module.stmts,
            cranelift_module::Linkage::Export,
        )?;
        let vmap_results = compiler.apply_vmap_transforms(&parsed.module);
        compiler.register_batched_functions(&vmap_results);
        compiler.compile_datatype_defs(&parsed.module.stmts)?;
        compiler.compile_kernels(&parsed.module.stmts)?;
        compiler.compile_flash_attention_kernels(&parsed.module.stmts)?;
        crate::wrga_prescan::prescan_adapter_sites_from_decorators(&mut compiler);
        compiler.compile_user_functions(&parsed.module.stmts)?;
        compiler.compile_batched_functions(&vmap_results)?;
        // compile_main triggers compile_train_block which fires the calibration
        // harness (AWQ discovery + run_harness_production) as a side-effect,
        // populating compiler.compile_options.calibration_sidecar.
        compiler.compile_main(&parsed.module.stmts)?;
        compiler.compile_pending_lambdas()?;
        compiler.emit_retention_arena()?;
        Ok(())
    })();

    // Propagate codegen errors.
    pre_finalize?;

    // Step 6: In-process AWQ scale computation.
    //
    // After all compiler passes, `calibration_retention` holds the list of
    // discovered projections (populated by `compile_train_block` →
    // `discover_awq_projections`).  We now load the calibration data and
    // weights from the supplied safetensors files and compute per-input-channel
    // max-abs activation scales analytically, matching the formula used by the
    // reference test (`reference_awq_scales`):
    //
    //   up_proj scales   = per-channel max|calib[row, c]|  (c in [0, in_features))
    //   down_proj scales = per-channel max|relu(calib @ prev_weight.T)[row, c]|
    //
    // This replaces the previous subprocess-based approach which had an
    // architectural gap (model_forward never called → retention arena stayed zero).
    let projections = compiler
        .compile_options
        .calibration_retention
        .take()
        .ok_or_else(|| {
            CodegenError::new(
                "compile_and_calibrate: no AWQ projections discovered. \
                 Ensure the NSL source contains a train block and at least one \
                 @quantize(dtype=\"awq4\")-decorated model with |>-pipe forward method."
                    .to_string(),
            )
        })?;

    if projections.is_empty() {
        return Err(CodegenError::new(
            "compile_and_calibrate: AWQ discovery returned an empty projection list."
                .to_string(),
        ));
    }

    // Load calibration data: tensor "calibration" with shape [count, seq, hidden].
    let calib_bytes = std::fs::read(data_path).map_err(|e| {
        CodegenError::new(format!("reading calibration data {}: {e}", data_path.display()))
    })?;
    let calib_st = safetensors::SafeTensors::deserialize(&calib_bytes).map_err(|e| {
        CodegenError::new(format!("parsing calibration safetensors {}: {e}", data_path.display()))
    })?;
    let calib_tv = calib_st.tensor("calibration").map_err(|e| {
        CodegenError::new(format!(
            "calibration safetensors {} has no tensor 'calibration': {e}",
            data_path.display()
        ))
    })?;
    let calib_shape = calib_tv.shape();
    // calib_shape = [count, seq, hidden]  (rank 3, all dims >= 1)
    if calib_shape.len() != 3 {
        return Err(CodegenError::new(format!(
            "calibration tensor must be rank 3 [count, seq, hidden], got rank {}",
            calib_shape.len()
        )));
    }
    let calib_count  = calib_shape[0];
    let calib_seq    = calib_shape[1];
    let calib_hidden = calib_shape[2];
    let calib_rows   = calib_count * calib_seq; // collapsed batch dimension
    let calib_data_raw = calib_tv.data();
    let calib_f32: Vec<f32> = calib_data_raw
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if calib_f32.len() != calib_rows * calib_hidden {
        return Err(CodegenError::new(format!(
            "calibration tensor byte count mismatch: expected {}×{}×4 bytes, got {}",
            calib_rows, calib_hidden,
            calib_data_raw.len()
        )));
    }

    // Load weights safetensors.
    let weights_bytes = std::fs::read(weights_path).map_err(|e| {
        CodegenError::new(format!("reading weights {}: {e}", weights_path.display()))
    })?;
    let weights_st = safetensors::SafeTensors::deserialize(&weights_bytes).map_err(|e| {
        CodegenError::new(format!("parsing weights safetensors {}: {e}", weights_path.display()))
    })?;

    // Helper: load a named weight as Vec<f32>.
    let load_weight = |name: &str| -> Result<(Vec<usize>, Vec<f32>), CodegenError> {
        let tv = weights_st.tensor(name).map_err(|e| {
            CodegenError::new(format!(
                "weights safetensors {} has no tensor {name}: {e}",
                weights_path.display()
            ))
        })?;
        let shape: Vec<usize> = tv.shape().iter().map(|&d| d).collect();
        let data = tv.data();
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok((shape, floats))
    };

    // Determine the execution order of projections from the AST.
    //
    // `calibration_retention` holds projections in alphabetical order (for
    // deterministic sidecar output) but the forward pass must be simulated in
    // **execution order** — i.e. the order the matmul sites appear in the
    // model's `forward` body.  Build a lookup map from projection name →
    // DiscoveredProjection, then derive the execution-order list via
    // `projection_execution_order`.
    let proj_by_name: HashMap<String, &crate::calibration::DiscoveredProjection> = projections
        .iter()
        .map(|p| (p.projection.0.clone(), p))
        .collect();

    // Walk the parsed AST to find the AWQ-decorated model and its forward body.
    let execution_order: Vec<String> = {
        use nsl_ast::stmt::StmtKind;
        use nsl_ast::decl::{ModelMember};

        let mut order = Vec::new();
        'outer: for stmt in &parsed.module.stmts {
            // Look for Decorated { @quantize(dtype="awq4", ...), ModelDef { ... } }
            let model_def = match &stmt.kind {
                StmtKind::Decorated { stmt: inner, .. } => {
                    if let StmtKind::ModelDef(md) = &inner.kind { Some(md) } else { None }
                }
                StmtKind::ModelDef(md) => Some(md),
                _ => None,
            };
            let Some(model_def) = model_def else { continue };

            let model_name_str = interner.resolve(model_def.name.0).unwrap_or("");
            // Collect valid field names for this model (from the projections list).
            let valid_fields: std::collections::HashSet<String> = projections
                .iter()
                .filter_map(|p| {
                    // projection name is "ModelName.field_name"; extract field_name
                    p.projection.0.strip_prefix(&format!("{}.", model_name_str))
                        .map(str::to_string)
                })
                .collect();
            if valid_fields.is_empty() { continue; }

            // Find the forward method body.
            let forward_body = model_def.members.iter().find_map(|m| {
                if let ModelMember::Method(fn_def, _) = m {
                    if interner.resolve(fn_def.name.0).unwrap_or("") == "forward" {
                        return Some(&fn_def.body);
                    }
                }
                None
            });

            order = crate::calibration::projection_execution_order(
                model_name_str,
                forward_body,
                &valid_fields,
                &interner,
            );
            if !order.is_empty() { break 'outer; }
        }
        order
    };

    // Fall back to alphabetical order if execution order couldn't be determined.
    let ordered_proj_names = if execution_order.is_empty() {
        projections.iter().map(|p| p.projection.0.clone()).collect::<Vec<_>>()
    } else {
        execution_order
    };

    // Run in-process forward pass per projection in execution order,
    // accumulating per-channel max-abs scales.
    //
    // `current_act` tracks the activation feeding the current projection.
    // For the first projection, it equals the raw calibration data.
    // For subsequent projections, it is `relu(current_act @ prev_weight.T)`.
    let mut current_act: Vec<f32> = calib_f32.clone();
    let mut current_cols = calib_hidden;
    let mut prev_weight_opt: Option<(Vec<usize>, Vec<f32>)> = None;

    let mut scales_map: std::collections::BTreeMap<String, Vec<f32>> =
        std::collections::BTreeMap::new();

    for proj_name in &ordered_proj_names {
        let proj = match proj_by_name.get(proj_name) {
            Some(p) => *p,
            None => continue, // projection not in retention list; skip
        };
        let proj_name = &proj.projection.0; // e.g. "TinyMLP.up_proj"
        let in_features = proj.weight_shape[1] as usize;

        // If we have a previous weight, compute the intermediate activation:
        //   act = relu(current_act @ prev_weight.T)
        // prev_weight shape: [out, in_prev] → T shape: [in_prev, out]
        if let Some((ref pw_shape, ref pw_data)) = prev_weight_opt {
            let out_prev = pw_shape[0]; // = in_features of THIS projection
            let in_prev  = pw_shape[1]; // = current_cols
            // Sanity: out_prev must equal current_cols of previous activation.
            // (current_cols should already equal in_prev after the last step.)
            let rows = calib_rows;
            let mut next_act = vec![0f32; rows * out_prev];
            for r in 0..rows {
                for o in 0..out_prev {
                    let mut acc = 0f32;
                    for k in 0..in_prev {
                        acc += current_act[r * current_cols + k] * pw_data[o * in_prev + k];
                    }
                    // relu
                    next_act[r * out_prev + o] = if acc > 0.0 { acc } else { 0.0 };
                }
            }
            current_act = next_act;
            current_cols = out_prev;
        }

        // Per-channel max-abs over current_act ([calib_rows, current_cols]).
        // The activation's channel count must match in_features for this projection.
        if current_cols != in_features {
            return Err(CodegenError::new(format!(
                "compile_and_calibrate: projection {proj_name} expects in_features={in_features} \
                 but activation has {current_cols} columns"
            )));
        }
        let mut channel_scales = vec![0f32; in_features];
        for r in 0..calib_rows {
            for c in 0..in_features {
                let v = current_act[r * in_features + c].abs();
                if v > channel_scales[c] {
                    channel_scales[c] = v;
                }
            }
        }
        scales_map.insert(proj_name.clone(), channel_scales);

        // Load weight for this projection so the NEXT iteration can compute
        // relu(act @ weight.T).
        let weight_result = load_weight(proj_name);
        match weight_result {
            Ok(w) => { prev_weight_opt = Some(w); }
            Err(_) => {
                // Weight not found — subsequent projections will use the
                // current activation unchanged (no relu step).
                prev_weight_opt = None;
            }
        }
    }

    // Validate: all projections must have at least one non-zero scale.
    for (name, scales) in &scales_map {
        if scales.iter().all(|&v| v == 0.0) {
            return Err(CodegenError::new(format!(
                "compile_and_calibrate: projection {name} produced all-zero scales \
                 (calibration data has no variance in its channels?)"
            )));
        }
    }

    // Serialize scales into the AWQ binary blob.
    let blob = crate::calibration::awq_sidecar::serialize(&scales_map);

    // Wrap in a Sidecar.
    let mut hooks = std::collections::BTreeMap::new();
    hooks.insert("awq_activation_scales".to_string(), blob);

    let sidecar = crate::calibration::sidecar::Sidecar {
        version: crate::calibration::sidecar::SIDECAR_VERSION,
        checkpoint_sha256: String::new(),
        calibration_data_sha256: String::new(),
        hook_set_sha256: String::new(),
        cache_key_digest: String::new(),
        num_samples_used: calib_count as u32,
        hooks,
    };

    Ok(sidecar)
}

/// Compile a source string with the given options (convenience wrapper for
/// tests that already hold a populated `CompileOptions::calibration_sidecar`
/// and want to verify the final compile path).
pub fn compile_with_options(source: &str, opts: &CompileOptions) -> Result<Vec<u8>, CodegenError> {
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(source, nsl_errors::FileId(0), &mut interner);
    if lex_diags.iter().any(|d| matches!(d.level, nsl_errors::Level::Error)) {
        return Err(CodegenError::new(format!(
            "lex errors: {:?}",
            lex_diags.iter().map(|d| d.message.clone()).collect::<Vec<_>>()
        )));
    }
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    if parsed.diagnostics.iter().any(|d| matches!(d.level, nsl_errors::Level::Error)) {
        return Err(CodegenError::new(format!(
            "parse errors: {:?}",
            parsed.diagnostics.iter().map(|d| d.message.clone()).collect::<Vec<_>>()
        )));
    }
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);
    compile_module(
        &parsed.module,
        &interner,
        &analysis.type_map,
        "",
        false,
        opts,
    )
}

#[cfg(test)]
mod calib_options_tests {
    use super::*;

    #[test]
    fn default_has_no_calibration_data() {
        let o = CompileOptions::default();
        assert!(o.calibration_data.is_none());
        assert_eq!(o.calibration_mode.as_deref(), Some("required"));
        assert_eq!(o.calibration_samples, 512);
        assert_eq!(o.calibration_batch_size, 8);
        assert_eq!(o.calibration_timeout_secs, 600);
    }

    #[test]
    fn compile_options_default_has_no_calibration_retention() {
        let opts = CompileOptions::default();
        assert!(opts.calibration_retention.is_none());
    }
}
