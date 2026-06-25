// Clippy style-lint churn new in 1.95+ that's not worth per-site fixes here:
// - doc_overindented_list_items / doc_lazy_continuation / unused_parens: doc
//   formatting pedantry that fires on preexisting ASCII diagrams.
// - manual_checked_ops (clippy 1.95+): explicit `if divisor == 0` fallback is
//   clearer than `.checked_div(...).unwrap_or(...)` at the call sites here.
//   `unknown_lints` is listed first so compilers older than 1.95 don't error on
//   the unrecognised lint name under `-D warnings`.
// - needless_range_loop: indexed loops where the index threads into multiple
//   parallel buffers in lock-step.
// - too_many_arguments: numerical-kernel / codegen-emit signatures have many
//   parameters by nature (block_q, block_kv, head_dim, rope flags, ...) and
//   refactoring into structs is churn with no callsite win.
// - sort_by_key_reverse / unstable_sort: cosmetic; ordering is fine either way.
// - has_len_but_not_is_empty: `len()` is the semantic we want; empty state is
//   represented via separate invariants, not "len == 0".
// - field_reassign_with_default / type_complexity: style, not correctness.
#![allow(
    unknown_lints,
    unused_parens,
    clippy::doc_overindented_list_items,
    clippy::doc_lazy_continuation,
    clippy::manual_checked_ops,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::field_reassign_with_default,
    clippy::type_complexity,
    clippy::len_without_is_empty
)]

pub mod agent;
pub mod autotune;
pub mod builtins;
pub(crate) mod c_export_table;
pub mod c_header;
pub mod c_wrapper;
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
pub mod bitnet;
pub mod cep;
pub mod cep_extract;
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
pub mod cpdt_decorator;
pub mod cpdt_expert_prune;
pub mod cpdt_moe_capacity;
pub mod cpdt_joint;
pub mod cpdt_optim;
pub mod cpdt_sensitivity;
pub mod cpdt_tier_apply;
pub mod cpdt_precision_exec;
pub mod cpdt_zero;
pub mod csha;
pub mod csha_apply;
pub mod csha_boundary;
pub mod csha_patterns;
pub mod csha_pipeline;
pub mod csha_specialize;
pub mod dynamic_shapes;
pub mod epilogue_fusion;
pub mod error;
pub mod expr;
pub mod fase;
pub mod fase_clip;
pub mod fase_codegen_table;
pub mod fase_memory;
pub mod fase_optimizer;
pub mod stmt_fase;
pub mod flash_attention;
pub mod flash_attention_v2;
pub mod flash_attention_selector;
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
pub mod kernel_skeleton;
pub mod linker;
pub mod memory_planner;
pub mod moe;
pub mod moe_hf_pack;
pub mod moe_kernels;
pub mod multimodal;
pub mod ffi_ownership;
pub mod ownership;
pub mod ownership_expr;
pub mod pca_activation;
pub mod pca_detect;
pub mod pca_rope;
pub mod pca_segment;
pub mod pca_tier_b;
pub mod pca_tile_config;
pub mod pca_tilerange;
pub mod pca_tileskip;
pub mod training_report;
pub mod pipeline;
pub mod ptxas_validation;
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
pub mod wggo_gradient_scorer;
pub mod wggo_graph;
pub mod wggo_ilp;
pub mod wggo_schedule;
pub mod wggo_weight_analysis;
pub mod wggo_weight_analysis_cache;
pub mod wggo_weight_analysis_nslweights;
pub mod wggo_overrides;
pub mod wggo_prune;
pub use wggo_overrides::{
    OverrideDiagnostic, OverrideRejectReason, PerLayerOverride, WggoOverrides,
};
pub mod fpga_error;
pub mod hir;
pub mod backend_verilog;
pub mod kernel_lower_fpga;  // M57.1 §3.3
pub mod wrga;
pub mod matmul_mma;
pub mod wrga_adapter_init;
pub mod wrga_adapter_inject;
pub mod wrga_adapter_rewrite;
pub mod wrga_fused_ptx;
pub mod wrga_fusion;
pub mod wrga_kernel_helpers;
pub mod wrga_memory;
pub mod wrga_prescan;
pub mod wrga_prune;
pub mod wrga_roofline;
pub mod wrga_spectral;
pub mod zk;

/// Binary-internal modules re-exposed at the library level so integration
/// tests can reach them without the source code living twice. The actual
/// binary entry points are in `src/bin/`; the submodules they consume live
/// under `src/bin/<binname>/`.
///
/// Gated behind the binary's required Cargo features so non-`cuda` builds
/// don't try to compile CUDA-dependent helpers.
#[cfg(all(feature = "cuda", feature = "debug_kernel_instrumentation"))]
pub mod bin {
    pub mod bench {
        #[path = "../../bin/bench/cli.rs"]
        pub mod cli;
        #[path = "../../bin/bench/fixtures.rs"]
        pub mod fixtures;
        #[path = "../../bin/bench/launch.rs"]
        pub mod launch;
        #[path = "../../bin/bench/output.rs"]
        pub mod output;
    }
}

// Note on the `#[path]` attributes above: when an inline `mod` is nested
// inside another inline `mod`, Rust treats the directory as having descended
// once per level. From `lib.rs` (in `src/`), entering `pub mod bin` notionally
// descends into `src/bin/`, and entering `pub mod bench` descends again into
// `src/bin/bench/`. The actual files live at `src/bin/bench/{cli,launch,output}.rs`,
// so the relative path from `src/bin/bench/` UP two levels and back down is
// `../../bin/bench/<file>.rs`. This keeps the binary and the library reading
// from the same files.

#[cfg(any(test, feature = "test-helpers"))]
pub mod test_helpers;

pub use compiler::{
    compile, compile_entry, compile_module, compile_module_with_imports,
    compile_entry_returning_plan, compile_module_with_imports_returning_plan,
    compile_returning_plan, compile_returning_splice_count_for_tests,
    compile_standalone, compile_standalone_returning_plan,
    compile_test, compile_with_zk_info, compile_with_zk_info_returning_plan,
    StandaloneConfig,
};

/// M57.1 §3.2: re-exported from the (private) `compiler::kernel` module so that
/// integration tests can pin the production redirect message without copying
/// the literal. See `tests/fpga_target_redirect.rs`.
pub use crate::compiler::kernel::FPGA_TARGET_REDIRECT_MSG;

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

/// §5.7 test helper: parse `source`, run only the pre-scan phase (AWQ + WGGO
/// discovery + auto-mode fallback note), and return the resolved
/// `CompileOptions`.
///
/// Panics on lex/parse errors so callers can use it in tests with clean
/// `expect`-style assertions on the returned opts.  Does **not** run codegen.
#[doc(hidden)]
pub fn debug_resolve_pre_scan_opts(source: &str, opts: CompileOptions) -> CompileOptions {
    use nsl_errors::{FileId, Level};
    let mut interner = nsl_lexer::Interner::new();
    let (tokens, lex_diags) = nsl_lexer::tokenize(source, FileId(0), &mut interner);
    assert!(
        !lex_diags.iter().any(|d| matches!(d.level, Level::Error)),
        "lex errors in test fixture: {:?}",
        lex_diags.iter().map(|d| d.message.clone()).collect::<Vec<_>>()
    );
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    assert!(
        !parsed.diagnostics.iter().any(|d| matches!(d.level, Level::Error)),
        "parse errors in test fixture: {:?}",
        parsed.diagnostics.iter().map(|d| d.message.clone()).collect::<Vec<_>>()
    );
    compiler::run_pre_scan_phase(&parsed.module, &interner, opts)
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

/// User-facing knob that gates how WGGO scores head importance.
/// - `Auto`: use gradient scoring when a calibration sidecar is present
///   (with per-layer magnitude fallback); otherwise pure magnitude.
/// - `Magnitude`: force pure magnitude scoring even if calibration is available.
/// - `Grad`: require gradient scoring; error if no calibration sidecar.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum WggoImportance {
    #[default]
    Auto,
    Magnitude,
    Grad,
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
    /// WGGO Stage 3: scoring mode (Auto/Magnitude/Grad).
    pub wggo_importance: WggoImportance,
    /// WGGO Stage 3: default fraction of heads allowed to be pruned.
    pub wggo_prune_fraction: Option<f64>,
    /// CSHA: fusion mode.
    pub csha_mode: Option<String>,
    /// CSHA: print the attention-fusion report.
    pub csha_report: bool,
    /// CPDT: planner mode (Off/ZeroOnly/Full).
    pub cpdt_mode: crate::cpdt::CpdtMode,
    /// CPDT: cluster topology.  Required when `cpdt_mode != Off`.
    pub cpdt_cluster: Option<crate::cpdt_zero::ClusterSpec>,
    /// CPDT: whether `--cpdt-report` was requested (stdout full plan).
    pub cpdt_report_requested: bool,
    /// CPDT §4.1: roofline slack ratio for the per-MoE capacity-factor
    /// override (`cap = max(1.0, slack × top_k / n_experts)`). Default 1.0
    /// (compute-bound — capacity clamps to 1.0). Higher values reflect
    /// memory-bound expert FFNs and raise capacity.
    pub cpdt_moe_roofline_slack: f64,
    /// CPDT: shared output slot the CLI reads after compile returns.
    /// `Compiler::invoke_cpdt_if_enabled` stashes the plan here so callers
    /// can render it without extending every entry-point return type.
    pub cpdt_plan_out:
        Option<std::sync::Arc<std::sync::Mutex<Option<crate::cpdt::CpdtPlan>>>>,
    /// M62: shared output slot the CLI reads after compile returns so it can
    /// emit a matching C header alongside the shared library. Populated by
    /// `Compiler::finalize` from `features.export_functions`.
    pub export_functions_out: Option<
        std::sync::Arc<std::sync::Mutex<Option<Vec<crate::c_header::ExportInfo>>>>,
    >,
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
    /// M62 Task 6: maps `self.<field>` `NodeId`s to weight-array indices for
    /// `@export` model methods compiled via `WeightPtrsArray` self-resolution.
    /// Populated from `nsl_semantic::AnalysisResult.weight_index_map` before
    /// calling any codegen entry point.  Empty map = no @export methods (safe
    /// default, nothing to look up).
    pub weight_index_map: HashMap<nsl_ast::NodeId, usize>,
    /// Owned AST/interner/type-map bundle used by the calibration subprocess
    /// path to emit a dedicated model object.
    pub calibration_compile_bundle:
        Option<std::sync::Arc<crate::calibration::CalibrationCompileBundle>>,
    /// When `Some`, codegen emits `model_backward` IR with grad-retention
    /// splices for the listed attention layers. `None` = forward-only AWQ
    /// or no calibration. Spec §4.8.
    pub calibration_grad_retention:
        Option<Vec<crate::calibration::discovery::WggoGradTarget>>,
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
            wggo_importance: WggoImportance::Auto,
            wggo_prune_fraction: None,
            csha_mode: None,
            csha_report: false,
            cpdt_mode: crate::cpdt::CpdtMode::Off,
            cpdt_cluster: None,
            cpdt_report_requested: false,
            cpdt_moe_roofline_slack: 1.0,
            cpdt_plan_out: None,
            export_functions_out: None,
            calibration_data: None,
            calibration_mode: Some("required".to_string()),
            calibration_samples: 512,
            calibration_batch_size: 8,
            calibration_timeout_secs: 600,
            calibration_sidecar: None,
            calibration_retention: None,
            calibration_batch_seq: None,
            weight_index_map: HashMap::new(),
            calibration_compile_bundle: None,
            calibration_grad_retention: None,
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
/// 4. Constructs a `Compiler` directly, runs all pre-`compile_main` passes
///    (string/enum/struct/model collection, function declaration, kernel
///    compilation, user-function compilation), then fires the calibration
///    harness at the wrapper level by invoking `real_subprocess_entry`
///    directly — the same canonical path used by AWQ + WGGO end-to-end tests.
///    After the harness completes, `compile_main` runs with
///    `calibration_sidecar` already populated.
/// 5. Reads back `compiler.compile_options.calibration_sidecar` and returns it.
///
/// The NSL source no longer needs a `train` block for calibration to fire.
/// Calibration fires whenever `calibration_data.is_some()`, driven by the
/// wrapper-level block (see `#134 (c-i)`).  Models decorated with
/// `@quantize(dtype="awq4")` will have their AWQ projections discovered via
/// `discover_awq_projections`; WGGO gradient targets are wired via
/// `WggoGradientHook` when `calibration_grad_retention` is populated.
pub fn compile_and_calibrate(
    source_path: &std::path::Path,
    data_path: &std::path::Path,
    weights_path: &std::path::Path,
) -> Result<crate::calibration::sidecar::Sidecar, CodegenError> {
    let source = std::fs::read_to_string(source_path).map_err(|e| {
        CodegenError::new(format!("reading source {}: {e}", source_path.display()))
    })?;

    // Step 1: peek at the calibration data to get (_count, seq).
    let (_count, seq) = nsl_runtime::calibration_data::peek_batch_seq(data_path)
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
    opts.calibration_batch_seq = Some((1, seq));
    opts.calibration_mode = Some("required".to_string());
    opts.calibration_compile_bundle = Some(std::sync::Arc::new(
        crate::calibration::CalibrationCompileBundle {
            ast: parsed.module.clone(),
            interner: interner.clone(),
            type_map: analysis.type_map.clone(),
        },
    ));

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
    compiler.compile_options.calibration_compile_bundle = opts.calibration_compile_bundle.clone();

    let pre_finalize = (|| -> Result<(), CodegenError> {
        compiler.intern_string("")?;
        compiler.collect_strings(&parsed.module.stmts)?;
        compiler.collect_enums(&parsed.module.stmts)?;
        compiler.collect_structs(&parsed.module.stmts)?;
        compiler.collect_models(&parsed.module.stmts)?;
        // M56 Task 17: compute agent struct layouts.
        compiler.collect_agents(&parsed.module.stmts)?;
        compiler.declare_runtime_functions()?;
        compiler.declare_imported_functions(&imported_fns)?;
        compiler.declare_user_functions_with_linkage(
            &parsed.module.stmts,
            cranelift_module::Linkage::Export,
        )?;
        // M56 Task 17: declare agent method FuncIds.
        compiler.declare_agent_methods(&parsed.module.stmts, cranelift_module::Linkage::Export)?;
        let vmap_results = compiler.apply_vmap_transforms(&parsed.module);
        compiler.register_batched_functions(&vmap_results);
        compiler.compile_datatype_defs(&parsed.module.stmts)?;
        compiler.compile_kernels(&parsed.module.stmts)?;
        compiler.compile_flash_attention_kernels(&parsed.module.stmts)?;
        crate::wrga_prescan::prescan_adapter_sites_from_decorators(&mut compiler);
        compiler.compile_user_functions(&parsed.module.stmts)?;
        // M56 Task 17: compile agent method bodies.
        compiler.compile_agent_methods(&parsed.module.stmts)?;
        compiler.compile_batched_functions(&vmap_results)?;
        // #134 (c-i) wrapper-level firing — spec §4.1 + §8.1 commit 3.
        // Previously: compile_train_block fired the calibration harness as a
        // side-effect inside compile_main, coupling calibration to the
        // presence of a `train` block.
        // Now: calibration fires here, at the wrapper level, regardless of
        // whether the source contains a `train` block. Path 1 (compile_train_
        // block's calibration block at stmt.rs:3960-4046) is deleted in this
        // same commit. real_subprocess_entry is invoked directly — same
        // canonical path that AWQ + WGGO end-to-end tests already use.
        //
        // ORDERING INVARIANT (preserved from the deleted stmt.rs block):
        // Calibration must fire BEFORE compile_main because compile_main
        // triggers compile_train_block's WGGO pass, which reads
        // compile_options.calibration_sidecar via build_scorer. Firing after
        // compile_main would leave the sidecar None at WGGO's read site,
        // silently degrading wggo_importance=Auto to magnitude scoring and
        // breaking wggo_importance=Grad outright. Spec §4.1's "after
        // compile_main returns" wording was over-specified — the architectural
        // goal "calibration runs around the compiled code" (§4.3) is equally
        // satisfied by firing before compile_main, with the WGGO ordering
        // invariant preserved.
        if let Some(data_path) = compiler.compile_options.calibration_data.clone() {
            let mut registry = crate::calibration::registry::HookRegistry::new();
            let awq_projections = compiler.discover_awq_projections().unwrap_or_default();
            if let Some(pre_scan) = compiler.compile_options.calibration_retention.as_ref() {
                crate::calibration::discovery::check_discovery_agreement(
                    pre_scan,
                    &awq_projections,
                )
                .map_err(|err| CodegenError::new(err.to_string()))?;
            }
            if !awq_projections.is_empty() {
                let proj_refs: Vec<crate::calibration::ProjectionRef> = awq_projections
                    .iter()
                    .map(|dp| dp.projection.clone())
                    .collect();
                registry.register(Box::new(
                    crate::calibration::awq_hook::AwqCalibrationHook::new(proj_refs),
                ));
                compiler.compile_options.calibration_retention = Some(awq_projections);
            }
            if let Some(targets) = compiler.compile_options.calibration_grad_retention.as_ref() {
                if !targets.is_empty() {
                    registry.register(Box::new(
                        crate::calibration::wggo_gradient_hook::WggoGradientHook::new(
                            targets.clone(),
                        ),
                    ));
                }
            }
            if registry.is_empty() {
                eprintln!(
                    "warning: --calibration-data {} supplied but no calibration hooks \
                     registered (no consumers yet — this is a no-op in MVP)",
                    data_path.display()
                );
            } else {
                let mode = match compiler
                    .compile_options
                    .calibration_mode
                    .as_deref()
                    .unwrap_or("required")
                {
                    "best-effort" => crate::calibration::HarnessMode::BestEffort,
                    _ => crate::calibration::HarnessMode::Required,
                };
                let cfg = crate::calibration::HarnessConfig {
                    checkpoints: compiler
                        .compile_options
                        .weight_file
                        .as_ref()
                        .map(|p| vec![p.clone()])
                        .unwrap_or_default(),
                    calibration_data: data_path.clone(),
                    samples: compiler.compile_options.calibration_samples,
                    batch_size: compiler.compile_options.calibration_batch_size,
                    timeout_secs: compiler.compile_options.calibration_timeout_secs,
                    mode,
                    projections: compiler
                        .compile_options
                        .calibration_retention
                        .clone()
                        .unwrap_or_default(),
                    compile_bundle: compiler.compile_options.calibration_compile_bundle.clone(),
                };
                match crate::calibration::binary_codegen::real_subprocess_entry(&cfg, &registry) {
                    Ok(out) => {
                        eprintln!(
                            "[calibration] {} ({} hooks)",
                            out.outcome_repr,
                            out.sidecar.hooks.len()
                        );
                        compiler.compile_options.calibration_sidecar = Some(out.sidecar);
                    }
                    Err(err) => {
                        return Err(CodegenError::new(format!("calibration: {err}")));
                    }
                }
            }
        }
        compiler.compile_main(&parsed.module.stmts)?;
        compiler.compile_pending_lambdas()?;
        compiler.emit_retention_arena()?;
        // Task 10: backward (WGGO grad) sibling arena — spec §7.2 ordering invariant #2.
        compiler.emit_grad_retention_arena()?;
        Ok(())
    })();

    // The calibration sidecar is stored in the Compiler's copy of compile_options.
    // Extract it before handling errors (the sidecar may have been populated even
    // if a later codegen pass failed, but we want the sidecar from a successful run).
    let sidecar = compiler.compile_options.calibration_sidecar.take();

    // Propagate codegen errors only after extracting the sidecar.
    pre_finalize?;

    sidecar.ok_or_else(|| {
        CodegenError::new(
            "compile_and_calibrate: calibration harness ran but produced no sidecar. \
             Ensure at least one @quantize(dtype=\"awq4\")-decorated model is present \
             (for AWQ projections) and/or @wggo_target decorators (for WGGO gradients), \
             and that the registered hook set is non-empty."
                .to_string(),
        )
    })
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

    #[test]
    fn compile_options_default_has_no_calibration_grad_retention() {
        let opts = CompileOptions::default();
        assert!(opts.calibration_grad_retention.is_none());
    }
}
