//! # nsl-codegen — Cranelift IR generation and native compilation
//!
//! This crate lowers the type-checked NSL AST into Cranelift IR, emits object
//! code, and orchestrates the many analysis/optimization passes that run in
//! between. It is the largest crate in the workspace, so its modules are
//! organized into the subsystem groups below.
//!
//! ## Public API
//!
//! The **stable entry points** are re-exported at the crate root:
//! [`compile`], [`compile_module`], [`compile_entry`], [`CompileOptions`],
//! [`CodegenError`], and the `compile_*_returning_plan` family.
//!
//! For navigating internals, the modules are also surfaced through subsystem
//! **facade namespaces** that mirror the architecture:
//!
//! - [`core`] — the compilation pipeline itself (compiler driver, statement /
//!   expression lowering, linker, C-export/header emission, ownership).
//! - [`gpu`] — GPU backends (PTX, AMDGPU, Metal, WGSL) and kernel lowering.
//! - [`training`] — autodiff (tape + source-to-source), Wengert lists, `vmap`.
//! - [`quantization`] — FP8, BitNet, AWQ/PCA precision tiering, weight analysis.
//! - [`distributed`] — tensor / context / pipeline parallelism, MoE, CPDT.
//! - [`analysis`] — cost model, autotuning, fusion, memory planning, WCET,
//!   FlashAttention codegen, calibration.
//! - [`experimental`] — research subsystems (CEP, CFIE, CSHA, WGGO, WRGA,
//!   FASE, ZK, FPGA, unikernel, sparse, speculative, multimodal). These APIs
//!   are **not stable** and may change or be removed between releases.
//!
//! These facades re-export the same modules that remain available at the crate
//! root for backward compatibility; they exist to make the crate navigable and
//! to flag which subsystems are experimental.
//!
//! See `ARCHITECTURE.md` in this crate for a fuller description.

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

// ===========================================================================
// Module declarations
//
// Every module is declared at the crate root (this keeps internal `crate::foo`
// paths and existing `nsl_codegen::foo` consumers stable). They are grouped
// below by subsystem for navigability, and re-surfaced through the facade
// namespaces (`core`, `gpu`, `training`, `quantization`, `distributed`,
// `analysis`, `experimental`) defined at the end of this file.
// ===========================================================================

// --- Core compilation pipeline -------------------------------------------
pub mod agent;
pub mod builtins;
pub(crate) mod c_export_table;
pub mod c_header;
pub mod c_wrapper;
pub mod compiler;
pub mod context;
pub mod dynamic_shapes;
pub mod error;
pub mod expr;
pub mod ffi_ownership;
pub mod func;
pub mod grammar_compiler;
pub mod linker;
pub mod ownership;
pub mod ownership_expr;
pub mod schema_convert;
pub mod standalone;
pub mod stdlib_loader;
pub mod stmt;
pub mod stmt_fase;
pub mod types;
pub mod use_count;

// --- GPU backends & kernel lowering --------------------------------------
pub mod backend_amdgpu;
pub mod backend_metal;
pub mod backend_ptx;
pub mod backend_wgsl;
pub mod gpu_specs;
pub mod gpu_target;
pub mod kernel;
pub mod kernel_ir;
pub mod kernel_lower;
pub mod kernel_skeleton;
pub mod matmul_mma;
pub mod ptx_metadata;
pub mod ptxas_validation;

// --- Autodiff & training -------------------------------------------------
pub mod ad_rules;
pub mod ccr;
pub mod source_ad;
pub mod training_report;
pub mod vmap;
pub mod wengert;
pub mod wengert_lower;

// --- Quantization & precision --------------------------------------------
pub mod bitnet;
pub mod fp8;
pub mod pca_activation;
pub mod pca_detect;
pub mod pca_per_doc;
pub mod pca_rope;
pub mod pca_segment;
pub mod pca_tier_b;
pub mod pca_tile_config;
pub mod pca_tilerange;
pub mod pca_tileskip;
pub mod weight_aware;

// --- Distributed & parallelism -------------------------------------------
pub mod context_parallel;
pub mod cpdt;
pub mod cpdt_comm;
pub mod cpdt_decorator;
pub mod cpdt_expert;
pub mod cpdt_expert_prune;
pub mod cpdt_joint;
pub mod cpdt_moe_capacity;
pub mod cpdt_optim;
pub mod cpdt_precision_exec;
pub mod cpdt_sensitivity;
pub mod cpdt_tier_apply;
pub mod cpdt_zero;
pub mod moe;
pub mod moe_hf_pack;
pub mod moe_kernels;
pub mod muon_roles;
pub mod pipeline;
pub mod tensor_parallel;

// --- Cost model, fusion & analysis ---------------------------------------
pub mod autotune;
pub mod calibration;
pub mod cost_model;
pub mod epilogue_fusion;
pub mod flash_attention;
pub mod flash_attention_selector;
pub mod flash_attention_v2;
pub mod fused_linear_ce;
pub mod precision_cast_ptx;
pub mod fusion;
pub mod fusion_graph;
pub mod fusion_report;
pub mod inspect;
pub mod memory_planner;
pub mod profiling;
pub mod reduction_fusion;
pub mod serve;
pub mod wcet;

// --- Experimental research subsystems ------------------------------------
// These are NOT part of the stable API. See the `experimental` facade.
pub mod cep;
pub mod cep_extract;
pub mod cep_importance;
pub mod cep_oracle;
pub mod cep_rewrite;
pub mod cep_search;
pub mod cep_emit_source;
pub mod cep_slice;
pub mod cfie;
pub mod cfie_cost;
pub mod cfie_decode_attention;
pub mod cfie_fused_sample;
pub mod cfie_grammar;
pub mod cfie_grammar_ptx;
pub mod cfie_kv_plan;
pub mod cfie_kv_quant;
pub mod cfie_kv_quant_ptx;
pub mod cfie_persistent;
pub mod cfie_persistent_ptx;
pub mod cfie_sample_ptx;
pub mod cfie_serve;
pub mod cfie_spec_sampler_ptx;
pub mod cfie_speculative;
pub mod cfie_speculative_ptx;
pub mod cpkd;
pub mod cpkd_fused_loss;
pub mod cpkd_spectral;
pub mod cpkd_student;
pub mod csha;
pub mod csha_apply;
pub mod csha_boundary;
pub mod csha_patterns;
pub mod csha_pipeline;
pub mod csha_specialize;
pub mod fase;
pub mod fase_clip;
pub mod fase_codegen_table;
pub mod fase_memory;
pub mod fase_optimizer;
pub mod multimodal;
pub mod sparse;
pub mod speculative;
pub mod unikernel;
pub mod unikernel_boot;
pub mod wggo;
pub mod wggo_apply;
pub mod wggo_cfie;
pub mod wggo_cpkd;
pub mod wggo_conflicts;
pub mod wggo_cost;
pub mod wggo_dp;
pub mod wggo_gradient_scorer;
pub mod wggo_graph;
pub mod layerwise;
pub mod transient_arena;
pub mod wggo_ilp;
pub mod wggo_overrides;
pub mod wggo_prepass;
pub mod wggo_prune;
pub mod wggo_schedule;
pub mod wggo_shape;
pub mod wggo_weight_analysis;
pub mod wggo_weight_analysis_cache;
pub mod wggo_weight_analysis_nslweights;
pub use wggo_overrides::{
    OverrideDiagnostic, OverrideRejectReason, PerLayerOverride, WggoOverrides,
};
pub mod wrga;
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

// FPGA / hardware-synthesis path (experimental).
pub mod backend_verilog;
pub mod fpga_error;
pub mod hir;
pub mod kernel_lower_fpga;  // M57.1 §3.3

// ===========================================================================
// Subsystem facade namespaces
//
// These re-export the modules above under architecture-oriented namespaces so
// the crate is navigable without touching the (backward-compatible) crate-root
// paths. New code is encouraged to import through these facades.
// ===========================================================================

/// Core compilation pipeline: the compiler driver, statement/expression
/// lowering, linking, C-export/header emission, and ownership analysis.
pub mod core {
    pub use crate::{
        agent, builtins, c_header, c_wrapper, compiler, context, dynamic_shapes,
        error, expr, ffi_ownership, func, grammar_compiler, linker, ownership,
        ownership_expr, schema_convert, standalone, stdlib_loader, stmt, stmt_fase,
        types, use_count,
    };
}

/// GPU code generation: device backends and kernel lowering.
pub mod gpu {
    pub use crate::{
        backend_amdgpu, backend_metal, backend_ptx, backend_wgsl, gpu_specs,
        gpu_target, kernel, kernel_ir, kernel_lower, kernel_skeleton,
        matmul_mma, ptx_metadata, ptxas_validation,
    };
}

/// Automatic differentiation and training-time codegen.
pub mod training {
    pub use crate::{ad_rules, source_ad, training_report, vmap, wengert, wengert_lower};
}

/// Quantization and reduced-precision execution.
pub mod quantization {
    pub use crate::{
        bitnet, fp8, pca_activation, pca_detect, pca_per_doc, pca_rope, pca_segment, pca_tier_b,
        pca_tile_config, pca_tilerange, pca_tileskip, weight_aware,
    };
}

/// Distributed execution and parallelism strategies.
pub mod distributed {
    pub use crate::{
        context_parallel, cpdt, cpdt_comm, cpdt_expert, cpdt_joint, cpdt_optim,
        cpdt_precision_exec, cpdt_sensitivity, cpdt_tier_apply, cpdt_zero, moe,
        moe_kernels, pipeline, tensor_parallel,
    };
}

/// Cost modeling, fusion, memory planning, and other analysis passes.
pub mod analysis {
    pub use crate::{
        autotune, calibration, cost_model, epilogue_fusion, flash_attention,
        flash_attention_selector, flash_attention_v2, fused_linear_ce, fusion, fusion_graph,
        fusion_report, inspect, memory_planner, profiling, reduction_fusion, serve,
        wcet,
    };
}

/// Experimental research subsystems. **APIs here are unstable** and may change
/// or be removed between releases.
pub mod experimental {
    pub use crate::{
        cep, cep_extract, cep_importance, cep_oracle, cep_rewrite, cep_search,
        cep_emit_source, cep_slice, cfie, cfie_fused_sample, cfie_grammar, cfie_kv_plan,
        cfie_kv_quant, cfie_persistent, cfie_speculative, csha, csha_apply,
        csha_boundary, csha_patterns, csha_pipeline, csha_specialize, fase,
        fase_clip, fase_codegen_table, fase_memory, fase_optimizer, multimodal,
        sparse, speculative, unikernel, unikernel_boot, wggo, wggo_apply,
        wggo_conflicts, wggo_cost, wggo_dp, wggo_gradient_scorer, wggo_graph,
        wggo_ilp, wggo_overrides, wggo_prune, wggo_schedule, wggo_weight_analysis,
        wggo_weight_analysis_cache, wggo_weight_analysis_nslweights, wrga,
        wrga_adapter_init, wrga_adapter_inject, wrga_adapter_rewrite,
        wrga_fused_ptx, wrga_fusion, wrga_kernel_helpers, wrga_memory,
        wrga_prescan, wrga_prune, wrga_roofline, wrga_spectral, zk,
    };

    /// FPGA / hardware-synthesis path (Verilog emission, HIR lowering).
    pub mod fpga {
        pub use crate::{backend_verilog, fpga_error, hir, kernel_lower_fpga};
    }
}

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
    compile_test, compile_with_profile_captures, compile_with_zk_info,
    compile_with_zk_info_returning_plan,
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

/// CFIE Tier-A observability: compile a module through the full
/// pipeline and return the [`crate::cfie::CfiePlan`] produced while
/// compiling its serve block (`None` when no serve block opted into
/// CFIE).  Mirrors [`debug_compile_and_return_plan_from_ast`].
pub fn debug_compile_and_return_cfie_plan_from_ast(
    ast: &nsl_ast::Module,
    interner: &nsl_lexer::Interner,
    type_map: &nsl_semantic::checker::TypeMap,
    options: &CompileOptions,
) -> Result<Option<crate::cfie::CfiePlan>, CodegenError> {
    let (res, _wrga_plan, cfie_plan) =
        crate::compiler::compile_module_with_imports_best_effort_plans(
            ast,
            interner,
            type_map,
            "",
            &[],
            HashMap::new(),
            std::collections::HashSet::new(),
            false,
            options,
        );
    match (res, cfie_plan) {
        (Ok(_), plan) => Ok(plan),
        (Err(e), Some(plan)) => {
            eprintln!(
                "[debug_compile_and_return_cfie_plan] codegen failed but a CFIE plan was produced: {}",
                e.message
            );
            Ok(Some(plan))
        }
        (Err(e), None) => Err(e),
    }
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
    /// Paper §9.3 ablation harness — per-Innovation skip flags forwarded to
    /// `wrga::run`. Default `WrgaAblation::default()` (= all false = no
    /// ablation, standard WRGA). Set via `nsl check --wrga-ablate=<name>`.
    pub ablation: crate::wrga::WrgaAblation,
}

#[derive(Debug, Clone)]
pub struct WrgaDecoratorConfig {
    pub mode: nsl_ast::block::WrgaMode,
    pub budget: Option<i64>,
    pub target: Option<String>,
    pub layers: Vec<String>,
    /// WRGA paper §8.2: name of the user-defined custom adapter model
    /// (e.g. `"GatedLoRA"`), or `None` for the built-in `lora`/`ia3`/
    /// `gatedlora` flavour selected by the compiler.  Resolved from the
    /// `adapter=<Ident>` argument on `@wrga(...)` at the semantic layer.
    pub custom_adapter: Option<String>,
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

/// CFTP §4.4 G3 (Sprint 2): codegen-side mirror of
/// `nsl_semantic::cftp::FusedCeConfig`.
///
/// Derived from `AnalysisResult.fused_ce_configs` and forwarded into
/// `CompileOptions.fused_ce_configs` by the CLI bridge.  Kept as a
/// codegen-side newtype so nsl-codegen does NOT depend on nsl-semantic
/// directly (matches the `WrgaInputs` / `FreezeDecoratorConfig` pattern).
///
/// v1 carries the two knobs surfaced by the decorator parser:
///
/// * `enabled` — whether the fused linear-CE kernel should fire on
///   matching cross_entropy occurrences inside the decorated `train` block.
/// * `vocab_tile` — explicit vocab-tile override (must be a multiple of
///   128, validated in semantic).  `None` falls back to the codegen
///   default (`FusedLinearCEConfig::default().vocab_tile` = 1024).
///
/// Sprint 2 wires only the plumbing; the lowering-site auto-substitution
/// is documented as deferred to Sprint 2.5 — see the inline marker near
/// `PrimalOp::CrossEntropyLoss` in `wengert_lower.rs`.
#[derive(Debug, Clone)]
pub struct FusedCeDecoratorConfig {
    /// `enabled = true|false` from the decorator. Defaults to `false` if
    /// the keyword arg is missing (preserves opt-in semantics).
    pub enabled: bool,
    /// `vocab_tile = N` override. `None` → codegen default.
    pub vocab_tile: Option<u32>,
    /// CFTP §4.4 G3 (Sprint 4): `vocab_size = N` from the decorator.
    /// Required at codegen time to bake `V` into the synthesised PTX.
    /// `None` → the fused emitter falls back to the composite
    /// `cross_entropy` lowering for this train block.
    pub vocab_size: Option<u32>,
    /// CFTP §4.4 G3 (Sprint 4): `hidden_size = N` from the decorator.
    /// Required at codegen time to bake `H` into the synthesised PTX.
    pub hidden_size: Option<u32>,
    /// CFTP §4.4 G3 (Sprint 4): `batch_size = N` from the decorator.
    /// Required at codegen time to size the per-row output buffers.
    pub batch_size: Option<u32>,
    /// CFTP §4.4 G3 (Sprint 4): `seq_len = N` from the decorator.
    /// Required at codegen time to size the per-row output buffers.
    pub seq_len: Option<u32>,
    /// CFTP §4.4 G3 v4-2: dtype hint from the decorator, mapped to the
    /// runtime FFI sentinel: `None`/`F32` → 0, `F16` → 1, `Bf16` → 2.
    /// Mirror of `nsl_semantic::cftp::FusedCeConfig.dtype`; kept as a
    /// codegen-local enum so nsl-codegen does not depend on nsl-semantic
    /// types (same convention as the rest of `FusedCeDecoratorConfig`).
    pub dtype: Option<FusedCeDtypeHint>,
    /// CFTP v10 (item 3): AST NodeId of the `train` block Stmt this
    /// decorator belongs to.  Codegen looks up the active config for
    /// each train block by matching this id — so multiple decorators
    /// in one compilation unit each dispatch to their own train block.
    /// Mirror of `nsl_semantic::cftp::FusedCeConfig.train_block_stmt_id`.
    pub train_block_stmt_id: nsl_ast::NodeId,
}

/// CPKD: codegen-side mirror of `nsl_semantic::cpkd::FusedKlCeConfig` —
/// the `@fused_kl_ce(...)` decorator on a `distill` block.
///
/// Same opt-in contract as `FusedCeDecoratorConfig`: the fused KL-CE op
/// only fires when `enabled = true` AND all five shape hints are present;
/// otherwise `fused_kl_ce(...)` calls fall through to the stdlib
/// composite (which forces tape AD — refused inside distill blocks, so
/// in practice an incomplete decorator is a loud compile error there).
///
/// v1 is f32-only: there is deliberately no dtype hint field (the
/// fp16/bf16 emitters are a documented deferral).
#[derive(Debug, Clone)]
pub struct FusedKlCeDecoratorConfig {
    pub enabled: bool,
    pub vocab_size: Option<u32>,
    /// Student hidden dim (`hidden_size =` in the decorator, matching the
    /// @fused_lm_ce naming for the trainable side).
    pub student_hidden: Option<u32>,
    /// Teacher hidden dim (`teacher_hidden =`).
    pub teacher_hidden: Option<u32>,
    pub batch_size: Option<u32>,
    pub seq_len: Option<u32>,
    pub vocab_tile: Option<u32>,
    /// AST NodeId of the `distill` block Stmt this decorator belongs to
    /// (per-block dispatch, mirroring CFTP v10 item 3).
    pub distill_block_stmt_id: nsl_ast::NodeId,
}

/// Codegen-side mirror of `nsl_semantic::cftp::FusedCeDtypeHint`.
///
/// Maps to the runtime FFI `dtype_tag: i64` sentinel:
/// * `F32` → 0 (v3-2 / pre-v3-2 byte-identical)
/// * `F16` → 1 (v3-2 fp16 emitters)
/// * `Bf16` → 2 (v4-1 bf16 emitters)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusedCeDtypeHint {
    F32,
    F16,
    Bf16,
}

impl FusedCeDtypeHint {
    /// FFI sentinel value matching the runtime contract.
    #[inline]
    pub fn dtype_tag(self) -> i64 {
        match self {
            FusedCeDtypeHint::F32 => 0,
            FusedCeDtypeHint::F16 => 1,
            FusedCeDtypeHint::Bf16 => 2,
        }
    }
}

/// CFTP §4.3 G2 Strategy 3 (Item 4): codegen-side mirror of
/// `nsl_semantic::cftp::PcaStrategy` carried through `CompileOptions`.
///
/// Mirrors the semantic enum 1-for-1 but lives in nsl-codegen so the
/// codegen crate does not need to depend on nsl-semantic types directly
/// (same convention as `FusedCeDtypeHint`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcaUserStrategy {
    /// Decorator-default — let the auto planner decide.
    Auto,
    /// Force segment-ID-masked kernel even if per-doc CTA would apply.
    SegmentId,
    /// Request per-document CTA scheduling. Activates the existing
    /// `PerDocAdmitConfig::enable_per_doc_cta` gate at the CSHA
    /// training-PTX synthesis site; admission still has the final say
    /// (block_q / num_docs / causal / fused-projection gates).
    PerDocument,
    /// Disable PCA entirely for this train block (force-fallback to
    /// the plain FA-2 forward / backward).
    Off,
}

impl PcaUserStrategy {
    /// True iff at least one `@pca(strategy=per_document)` (or its
    /// aliases) was collected on this compile.
    pub fn is_per_document(self) -> bool {
        matches!(self, PcaUserStrategy::PerDocument)
    }

    /// Map from the semantic `PcaStrategy` enum without forcing
    /// nsl-codegen to take a direct dependency on its discriminants.
    /// Callers provide the textual name (`PcaStrategy::as_str`) — keeps
    /// the wire-format stable across the crate boundary.
    pub fn from_semantic_str(s: &str) -> Self {
        match s {
            "auto" => PcaUserStrategy::Auto,
            "segment_id" => PcaUserStrategy::SegmentId,
            "per_document" => PcaUserStrategy::PerDocument,
            "off" => PcaUserStrategy::Off,
            // Forward-compat fallback: an unrecognised string from a
            // newer semantic crate decays to `Auto` rather than crashing
            // the build. Tests pin all four canonical strings AND the
            // catch-all so a future rename of `PcaStrategy::Auto::as_str`
            // does not silently mask via this fallback.
            _ => PcaUserStrategy::Auto,
        }
    }
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

/// WGGO: weight-graph global-optimization options.
///
/// Grouped out of [`CompileOptions`] as part of decomposing that god-config
/// struct into cohesive sub-structs (architecture-hardening review).
#[derive(Clone, Debug, PartialEq, Default)]
pub struct WggoOptions {
    /// Global optimization mode ("full", "greedy", "off"). When `None`, WGGO is
    /// not run. See `crates/nsl-codegen/src/wggo.rs`.
    pub mode: Option<String>,
    /// Print the global-optimization report to stderr.
    pub report: bool,
    /// Stage 3: path to a `.nslweights` sidecar file.
    pub weights: Option<std::path::PathBuf>,
    /// Stage 3: scoring mode (Auto/Magnitude/Grad).
    pub importance: WggoImportance,
    /// Stage 3: default fraction of heads allowed to be pruned.
    pub prune_fraction: Option<f64>,
    /// Lower the plan's optim_m/v_bits decisions to real optimizer-state
    /// storage dtypes (opt-in: reduced-precision moments change training
    /// numerics, and the v1 dequant→step→quant envelope is CPU-only —
    /// GPU-resident training is refused loudly at train-block setup, when
    /// `nsl_tensor_zeros_like_dtype` allocates the moment buffers). When
    /// false, sub-32 plan bits stay advisory and a not-lowered notice is
    /// printed.
    pub moment_precision: bool,
    /// Per-device resident training-memory budget in BYTES for the WGGO
    /// plan (`--wggo-memory-budget <MiB>`, converted at the CLI boundary,
    /// which also validates non-zero and sets `moment_precision` — the
    /// flag implies it). `None` = today's behavior, byte-identical
    /// (WGGO opt-in precedent: `zero_stage_search`, `snap_to_grid`).
    /// When `Some`, the budget is enforced by the Level-1 DP
    /// (`ClusterSpec::memory_budget`) and the per-layer ILP
    /// (`LayerIlpConstraints::{memory_budget, budget_informed}`), and a
    /// plan that cannot fit even at the fp16-moment floor is a HARD
    /// compile failure (no silent degradation).
    pub memory_budget_bytes: Option<u64>,
}

/// CFIE: compiler-fused inference-engine options (paper: docs/research/CFIE.pdf).
///
/// Grouped as a cohesive sub-struct per the architecture-hardening
/// CompileOptions decomposition convention (cf. [`CshaOptions`] /
/// [`CpdtOptions`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CfieOptions {
    /// CLI mode override (`--cfie full|sampling|off`).  `None` defers to
    /// the serve block's `@cfie` decorator / config keys.
    pub mode_override: Option<String>,
    /// Write the CFIE build report to this path in addition to stderr
    /// (`--cfie-report <path>`).
    pub report_path: Option<std::path::PathBuf>,
}

/// M53: Worst-case-execution-time (WCET) analysis and certification options.
///
/// Grouped out of [`CompileOptions`] as part of decomposing that god-config
/// struct into cohesive sub-structs (architecture-hardening review).
#[derive(Clone, Debug, PartialEq)]
pub struct WcetOptions {
    /// Enable WCET analysis for `@real_time` functions.
    pub enabled: bool,
    /// GPU target name for WCET analysis (e.g., "Orin", "H100").
    pub gpu: Option<String>,
    /// CPU target name for WCET analysis (e.g., "cortex-a78").
    pub cpu: Option<String>,
    /// Path to write the WCET certificate JSON.
    pub report_path: Option<std::path::PathBuf>,
    /// Safety-margin multiplier for WCET (default: 1.05 = 5%).
    pub safety_margin: f64,
    /// Path to write a DO-178C compliance report.
    pub do178c_report: Option<std::path::PathBuf>,
    /// WCET target type: "gpu" (statistical), "fpga" (certified), "groq" (blocked).
    pub target: String,
    /// FPGA device name for certified WCET (e.g., "xcvu440", "xczu9eg").
    pub fpga_device: Option<String>,
}

impl Default for WcetOptions {
    fn default() -> Self {
        Self {
            enabled: false,
            gpu: None,
            cpu: None,
            report_path: None,
            safety_margin: 1.05,
            do178c_report: None,
            target: "gpu".to_string(),
            fpga_device: None,
        }
    }
}

/// M55: Zero-knowledge proof-circuit emission options.
///
/// Grouped out of [`CompileOptions`] as part of decomposing that god-config
/// struct into cohesive sub-structs (architecture-hardening review).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ZkOptions {
    /// Emit a ZK inference circuit alongside compiled output.
    pub circuit: bool,
    /// ZK backend to use ("folding", "halo2", or "plonky3").
    pub backend: String,
    /// ZK field to use ("m31" or "bn254").
    pub field: String,
    /// Also emit a Solidity verifier contract.
    pub solidity: bool,
    /// Path to safetensors weight file used as ZK witness.
    pub weights_path: Option<std::path::PathBuf>,
}

impl Default for ZkOptions {
    fn default() -> Self {
        Self {
            circuit: false,
            backend: "folding".to_string(),
            field: "m31".to_string(),
            solidity: false,
            weights_path: None,
        }
    }
}

/// CSHA (compiler-specialized hardware attention) codegen options.
///
/// Grouped out of [`CompileOptions`] as part of decomposing that god-config
/// struct into cohesive sub-structs (architecture-hardening review).
#[derive(Clone, Debug, Default)]
pub struct CshaOptions {
    /// Fusion mode (`None` = off; e.g. `Some("auto".into())`).
    pub mode: Option<String>,
    /// Print the attention-fusion report.
    pub report: bool,
}

/// CPDT (compiler-planned distributed training) options.
///
/// Grouped out of [`CompileOptions`] as part of decomposing that god-config
/// struct into cohesive sub-structs (architecture-hardening review).
#[derive(Clone)]
pub struct CpdtOptions {
    /// Planner mode (Off/ZeroOnly/Full).
    pub mode: crate::cpdt::CpdtMode,
    /// Cluster topology.  Required when `mode != Off`.
    pub cluster: Option<crate::cpdt_zero::ClusterSpec>,
    /// Whether `--cpdt-report` was requested (stdout full plan).
    pub report_requested: bool,
    /// CPDT Part III §4.1: roofline-derived MoE `capacity_factor` override
    /// slack — fraction of theoretical peak the planner targets when
    /// upper-bounding `capacity_factor`. Default 0.0 = no override
    /// (paper-faithful behavior). Set via `--cpdt-moe-roofline-slack`.
    /// Consumed by
    /// [`crate::cpdt_moe_capacity::apply_roofline_capacity_override`].
    pub moe_roofline_slack: f64,
    /// Shared output slot the CLI reads after compile returns.
    /// `Compiler::invoke_cpdt_if_enabled` stashes the plan here so callers
    /// can render it without extending every entry-point return type.
    pub plan_out:
        Option<std::sync::Arc<std::sync::Mutex<Option<crate::cpdt::CpdtPlan>>>>,
}

impl Default for CpdtOptions {
    fn default() -> Self {
        Self {
            mode: crate::cpdt::CpdtMode::Off,
            cluster: None,
            report_requested: false,
            moe_roofline_slack: 0.0,
            plan_out: None,
        }
    }
}

/// WRGA check-mode override context (`nsl check --wrga-analyze | --wrga-compare`).
///
/// Replaces the former CLI-side thread-locals (`WRGA_TARGET_OVERRIDE` /
/// `WRGA_ABLATION_OVERRIDE` / `WRGA_PLAN_CAPTURE`) with explicit state threaded
/// through [`CompileOptions`], mirroring [`CpdtOptions::plan_out`]. Every field
/// is `None` on the normal `nsl build` path, so this is zero-overhead there.
///
/// The two override fields are consumed CLI-side (the WRGA bridge pre-applies
/// them onto [`WrgaInputs`] before it reaches codegen); codegen itself never
/// reads them. `plan_capture` is written by the CLI build paths from the
/// `WrgaPlan` returned by `compile_returning_plan`.
#[derive(Clone, Default)]
pub struct WrgaCheckContext {
    /// `--wrga-target <gpu>` — copied onto every `WrgaDecoratorConfig::target`
    /// before the bridge ships `WrgaInputs` to codegen. When the source has no
    /// `@wrga(...)` at all, a minimal Auto-mode config is synthesised so the
    /// target choice still reaches `wrga::run`.
    pub target_override: Option<String>,
    /// `--wrga-ablate=<flags>` — copied onto `WrgaInputs::ablation`.
    pub ablation_override: Option<crate::wrga::WrgaAblation>,
    /// Capture slot for the produced `WrgaPlan`, read by `--wrga-compare` after
    /// the build returns (mirrors [`CpdtOptions::plan_out`]). `None` outside a
    /// compare run; captures the FIRST non-`None` plan so multi-file paths
    /// don't overwrite the entry-module plan with a dependency's empty one.
    pub plan_capture:
        Option<std::sync::Arc<std::sync::Mutex<Option<crate::wrga::WrgaPlan>>>>,
}

/// Periodic-checkpointing stride request (Item 8, `--checkpoint-stride`).
/// `Fixed(k)` coalesces every `k` block anchors into one CCR super-segment;
/// `Auto` lets the activation-budget scheduler pick `k`. `Fixed(1)` (the
/// default) is the classic per-block behavior — no coalescing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CheckpointStride {
    Fixed(usize),
    Auto,
}

impl Default for CheckpointStride {
    fn default() -> Self {
        CheckpointStride::Fixed(1)
    }
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
    /// P0 certification: RNG seed for `randn`/`rand`/stochastic ops.
    /// `None` keeps the historical behavior (seed 42 under
    /// --deterministic, unseeded otherwise). `Some(s)` seeds the RNG at
    /// program start regardless of --deterministic (multi-seed training
    /// campaigns need distinct reproducible inits; bit-reproducibility of
    /// the DEVICE path still requires --deterministic).
    pub rng_seed: Option<u64>,
    /// M52: Path to safetensors weight file for weight-aware compilation
    pub weight_file: Option<std::path::PathBuf>,
    /// M52: Weight-aware compilation configuration
    pub weight_config: weight_aware::WeightAwareConfig,
    /// M52: Whether to emit a weight analysis report (nsl check --weight-analysis)
    pub weight_analysis: bool,
    /// M54: Unikernel build configuration (None = normal build)
    pub unikernel_config: Option<crate::unikernel::UnikernelConfig>,
    /// M53: Worst-case-execution-time analysis / certification options.
    pub wcet: WcetOptions,
    /// M38a: Enable linear types ownership checking.
    pub linear_types_enabled: bool,
    /// M38a: Per-function ownership metadata from semantic analysis.
    /// Keys are function names, values have linear_params and shared_params.
    pub ownership_info: HashMap<String, crate::ownership::FunctionOwnership>,
    /// M55: Zero-knowledge proof-circuit emission options.
    pub zk: ZkOptions,
    /// M43b: ZeRO optimizer sharding stage (1, 2, or 3)
    pub zero_stage: Option<u8>,
    /// P4 item 17 (`--param-dtype bf16-sr`): authoritative BF16 parameter
    /// storage with counter-based stochastic rounding on the fused AdamW
    /// update — no FP32 master copy. Rides the weight-stream residency
    /// schedule (bf16 device mirrors, transient f32 working views).
    pub param_dtype_bf16sr: bool,
    /// P4 item 18 rung 2 (`--muon-state-dtype bf16`): Muon first-moment
    /// buffers stored in BF16 with an FP32 working buffer per update and a
    /// counter-based SR quant-store (v stays f32 — it is null-sloted per
    /// param on the Muon route).
    pub muon_state_bf16: bool,
    /// Debug training mode: disables fusion, disables FBIP, and emits
    /// gradient checksum assertions after each backward pass.
    pub debug_training: bool,
    /// P0.3: gradient-integrity gate. Emits a per-step check that every
    /// trainable parameter received a finite, mostly-nonzero gradient, and
    /// prints the `[grad-integrity]` worst-case snapshot at exit (under
    /// `NSL_GRAD_INTEGRITY=1`, which the flag sets). Unlike `debug_training`
    /// it changes NO optimization decisions — it only observes gradients.
    pub grad_integrity: bool,
    /// P1.7: permanent reference-training mode (`--training-reference`). Forces
    /// the SIMPLEST correct training path so an optimized stack can be compared
    /// against an independent baseline (stronger than comparing two optimized
    /// paths that may share the same bug). Disables: CCR, CSLA, weight
    /// streaming, optimizer-state offload, WGGO, CPDT/reduced-precision moments,
    /// CSHA, kernel `@fuse` fusion (the field-controlled ones, forced off at the
    /// CLI) plus FBIP in-place, the fused FASE optimizer step, and the fused-CE
    /// (`@fused_lm_ce`) / fused-KL-CE substitution and `@checkpoint` decorators
    /// (gated in codegen on this field). The source-AD fused activation
    /// backward is retained — it is bit-exact-equivalent to the unfused form
    /// (FASE≡AdamW gates) and is not a distinct-numerics surface.
    pub training_reference: bool,
    /// M62a: Build as a shared library (.so/.dylib/.dll) instead of an executable.
    /// Also controls PIC codegen (`is_pic`), which every object linked into
    /// the shared library needs — including non-entry modules on the
    /// multi-file path. See `emit_export_table` for the (distinct) decision
    /// of which compilation unit emits the `nsl_get_num_exports` /
    /// `nsl_get_export_name` C ABI, since only one may define them.
    pub shared_lib: bool,
    /// Whether *this* compilation unit should emit the shared-library
    /// export-table FFIs (`nsl_get_num_exports` / `nsl_get_export_name`).
    /// On the single-file shared-lib path this mirrors `shared_lib`. On the
    /// multi-file path every module needs `shared_lib = true` for PIC, but
    /// only the entry module's object may define these symbols — every
    /// other module defining them too causes a "multiple definition"
    /// linker error when the objects are joined. Defaults to `false`.
    pub emit_export_table: bool,
    /// WRGA: decorator configs forwarded from nsl-semantic (Task 1 of bridge).
    pub wrga_inputs: Option<WrgaInputs>,
    /// CFTP §4.4 G3 (Sprint 2): `@fused_lm_ce(...)` configs forwarded from
    /// nsl-semantic.  Empty when no decorator is present; codegen consults
    /// the first `enabled = true` entry to gate the fused linear-CE
    /// kernel emission (Sprint 2.5 substitution; v1 plumbing-only).
    pub fused_ce_configs: Vec<FusedCeDecoratorConfig>,
    /// CPKD: `@fused_kl_ce(...)` decorator configs, one per decorated
    /// distill block. Empty when no decorator is present.
    pub fused_kl_ce_configs: Vec<FusedKlCeDecoratorConfig>,
    /// CFTP §4.3 G2 Strategy 3 (Item 4): `@pca(strategy=...)` strategies
    /// forwarded from nsl-semantic. Empty when no `@pca` decorator is
    /// present. The CSHA training-PTX synthesis site consults this list
    /// to flip `PerDocAdmitConfig::enable_per_doc_cta=true` when at least
    /// one entry requests `PerDocument`.
    /// Stored as the codegen-local `PcaUserStrategy` enum so nsl-codegen
    /// does not depend directly on nsl-semantic types (mirrors the
    /// `FusedCeDecoratorConfig` / `WrgaInputs` pattern).
    pub pca_user_strategies: Vec<PcaUserStrategy>,
    /// WRGA Milestone B.2 Task 3: fold WRGA memory hints into real
    /// allocations (vs. B.1's observational-only path). Default false.
    pub wrga_fold_allocations: bool,
    /// WGGO: weight-graph global-optimization options.
    pub wggo: WggoOptions,
    /// CFIE: compiler-fused inference-engine options.
    pub cfie: CfieOptions,
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
    /// Optimizer-state offload (scaling campaign item 4, the single-GPU
    /// ZeRO-Offload analog): allocate m/v HOST-resident (CPU f32) and wrap
    /// every optimizer step in a stage-in → GPU-f32 update → copy-back
    /// envelope. The update math runs on the device exactly as without the
    /// flag (same kernels, same dtype), so FASE≡AdamW exactness is
    /// preserved; the cost is one HtoD+DtoH round-trip of the optimizer
    /// state per step. Frees 2×param bytes of VRAM for Adam-family
    /// optimizers (1× for momentum-SGD/Lion/Muon). Mutually exclusive with
    /// reduced-precision moments (`nsl_tensor_cast_into` cannot cross
    /// devices) — enforced with a loud compile error in stmt.rs.
    pub optim_state_offload: bool,
    /// CCR P1.a (`--checkpoint-blocks`): block-granular activation
    /// checkpointing on the source-AD path. Interiors of each transformer
    /// block are freed after the forward and recomputed just before that
    /// block's adjoint runs — activation residency drops from
    /// O(layers x interiors) to O(layers x boundaries + one block).
    /// Bit-exact (same kernels, same order, same inputs); Dropout results
    /// and CSHA-claimed chains are force-saved. No-ops with a loud stderr
    /// note when the tape has no `blocks.N` structure.
    pub checkpoint_blocks: bool,
    /// CCR P1.b: with `checkpoint_blocks`, use the SELECTIVE policy —
    /// matmul-class outputs stay saved, only cheap bandwidth-bound ops
    /// (norms, RoPE, elementwise, softmax, reshapes) are replayed. Less
    /// memory reduction than the block policy at near-zero recompute cost.
    pub checkpoint_selective: bool,
    /// CCR P1.c (`--checkpoint-budget-mib`): allowed SAVED-interior bytes.
    /// The per-tensor SAVE/RECOMPUTE knapsack (ccr::apply_budget) flips the
    /// highest-value tensors back to SAVE within this budget, minimizing
    /// recompute cost. Under FASE Deferred the C-01 credit (the gradient
    /// buffer Deferred never materializes) is added when parameter sizes
    /// are statically known. None = pure policy decision, no arbitration.
    pub checkpoint_budget_mib: Option<u64>,
    /// Item 8 (`--checkpoint-stride N|auto`): periodic checkpointing. With
    /// `Fixed(k)`, CCR coalesces every `k` transformer-block anchors into one
    /// super-segment — saving only every k-th block boundary and recomputing
    /// the k-block span. Trades recompute for a k× smaller saved-boundary
    /// surface (the activation surface CSLA buffers across the accumulation
    /// window). `Auto` searches strides with `ccr::project_activation_peak`
    /// and picks the smallest projected peak within `checkpoint_budget_mib`
    /// (or the min-peak stride if none fits). `Fixed(1)` = classic per-block.
    /// Bit-exact regardless of stride (recompute replays the same kernels).
    ///
    /// NOTE: under `Auto`, `checkpoint_budget_mib` is consulted twice with two
    /// meanings — first as a peak-concurrent-activation budget to pick the
    /// stride (`select_stride`), then as a saved-interior-bytes budget by the
    /// per-tensor knapsack (`ccr::apply_budget`). Both keep the plan valid (the
    /// number is a soft target, not a hard cap), so the dual use is safe but
    /// deliberate; unifying them is future work (the full DP scheduler).
    pub checkpoint_stride: CheckpointStride,
    /// Item 9 (`--fuse-rmsnorm-backward`): lower the source-AD RMSNorm INPUT
    /// gradient to a single fused `nsl_rmsnorm_dx_backward` op (native GPU kernel
    /// / CPU reference) instead of the ~11-op decomposition — fewer launches,
    /// temporaries, and HBM traffic. Off by default; the fused kernel matches
    /// the decomposition to an f32 tolerance (approx rsqrt/div), so it is an
    /// opt-in speedup, not a bit-exact substitution.
    pub fuse_rmsnorm_backward: bool,
    /// CCR phases 5-6 (`--checkpoint-compress fp16|bf16`): compress the
    /// Selective policy's saved matmul-class interiors to half precision
    /// between forward and backward (cast-on-save, dequant-on-load via the
    /// CFTP-v7 GPU cast kernels). NOT bit-exact — backward reads rounded
    /// activations; gated by the repo's 3-4 dp loss-parity standard.
    pub checkpoint_compress: Option<String>,
    /// CSLA Stage-2 (`--layerwise-accum`): window-buffered training schedule.
    /// The N micro-batches of a FASE-Deferred accumulation window run their
    /// forwards first (saving only the adjoint-read, batch-dependent tensors
    /// plus the batch dicts), then the backward phase replays the whole window
    /// through a runtime loop over the buffered micro-batches. Bit-exact with
    /// the interleaved baseline (same kernels, same inputs, same per-parameter
    /// accumulation order). Requires `--source-ad`, `--checkpoint-blocks`, and
    /// a FASE-Deferred plan (AdamW/Adam + grad_accumulation >= 2); refuses
    /// loudly on grad_clip, WGGO mode tables, `--optim-state-offload`,
    /// `--checkpoint-compress`, and the pipelined/tape paths.
    pub layerwise_accum: bool,
    /// D2b (`--weight-stream`, requires `--layerwise-accum`): layer weight
    /// streaming across the WHOLE training loop. Layer-grouped params keep
    /// pinned host mirrors; the forward (sliced per CCR segment) uploads
    /// each layer before its segment and evicts after its last primal
    /// read, and the window backward re-uploads per replay range and
    /// writes back after that layer's update. Teardown restores residency
    /// for model_save/eval. Tensor POINTERS never change (side-table
    /// mechanism), so param_list / struct fields / the tie guard stay
    /// valid. Byte-preserving — bit-exact.
    pub weight_stream: bool,
    /// Item 10 (`--stream-arena`, requires `--weight-stream`): batch each
    /// layer's per-param host<->device transfers into ONE contiguous transfer
    /// through a stable, reused device staging arena. Cuts CUDA calls, lands
    /// one large PCIe transaction per layer, keeps device addresses stable
    /// across steps, and bounds fragmentation. Bit-exact with the per-param
    /// path (same mirror bytes, same order).
    pub stream_arena: bool,
    /// Item 11 (`--stream-prefetch`, requires `--stream-arena`): double-buffer
    /// the backward weight stream. Each layer's pack is prefetched (async HtoD
    /// on the transfer stream) while the PREVIOUS layer computes, and the
    /// compute stream waits on a per-pack CUDA event before reading it — the
    /// CADENCE-style assume/guarantee transfer certificate. WGGO's calibration
    /// activates the overlap only where estimated compute hides the transfer;
    /// otherwise it falls back to the synchronous arena upload. Bit-exact.
    pub stream_prefetch: bool,
    /// Item 11 writeback half (`--stream-async-writeback`, requires
    /// `--stream-arena`): issue each layer pack's post-update DtoH on the
    /// transfer stream instead of blocking, deferring the mirror scatter to
    /// the runtime's drain points (writeback-queue cap / affected re-upload /
    /// teardown). Completes the double-buffer schedule: compute L, prefetch
    /// L+1, write back L-1. Bit-exact (same bytes, different timing).
    pub stream_async_writeback: bool,
    /// Dev Tools Phase 5, Task 7: enable `@inspect` decorator emission.
    pub inspect_enabled: bool,
    /// CSHA (compiler-specialized hardware attention) codegen options.
    pub csha: CshaOptions,
    /// CSHA Sprint 2 (paper §6.2 binding fix): per-model `@csha(...)` config
    /// captured by the semantic checker, keyed by the decorated model's name
    /// (the `<Type>` in `model <Type>:` or the LHS binding name in
    /// `@csha let m = SomeModel()`).  The CSHA hook in
    /// `nsl-codegen/src/stmt.rs::compile_train_block` looks up the current
    /// `model_type_name` in this map and:
    ///   * `disabled = true` -> skip the CSHA pipeline for that compile.
    ///   * `level    = Some(L)` -> clamp the planner's `mode_str` to L.
    ///   * `target   = Some(T)` -> override `csha::run_on_wengert`'s target.
    /// Empty map = no `@csha` decorators in the program (the default), which
    /// preserves the pre-Sprint-2 behaviour driven solely by `--csha`.
    pub csha_configs: HashMap<String, nsl_semantic::csha::CshaConfig>,
    /// Cycle-10 §5.3 paper checkpointing-aware backward (Task 6):
    /// per-function `@checkpoint(policy=...)` policies collected by
    /// `EffectChecker::checkpoint_policies()` and routed through the
    /// `nsl-cli` loader (`crates/nsl-cli/src/loader.rs:414`). Consumed
    /// by extractor construction sites in `stmt.rs` /
    /// `binary_codegen.rs` via `WengertExtractor::with_checkpoint_policies`.
    /// Empty map = no checkpointing transformations = byte-identity preserved.
    pub checkpoint_policies:
        HashMap<String, nsl_semantic::effects::CheckpointPolicy>,
    /// CPDT (compiler-planned distributed training) options.
    pub cpdt: CpdtOptions,
    /// WRGA check-mode override context (`nsl check --wrga-analyze | --wrga-compare`).
    /// Carries the `--wrga-target` / `--wrga-ablate` overrides and the
    /// `--wrga-compare` plan-capture slot. All-`None` on normal builds.
    pub wrga_check: WrgaCheckContext,
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
            rng_seed: None,
            weight_file: None,
            weight_config: weight_aware::WeightAwareConfig::default(),
            weight_analysis: false,
            unikernel_config: None,
            wcet: WcetOptions::default(),
            linear_types_enabled: false,
            ownership_info: HashMap::new(),
            zk: ZkOptions::default(),
            zero_stage: None,
            param_dtype_bf16sr: false,
            muon_state_bf16: false,
            debug_training: false,
            grad_integrity: false,
            training_reference: false,
            shared_lib: false,
            emit_export_table: false,
            wrga_inputs: None,
            fused_ce_configs: Vec::new(),
            fused_kl_ce_configs: Vec::new(),
            pca_user_strategies: Vec::new(),
            wrga_fold_allocations: false,
            wggo: WggoOptions::default(),
            cfie: CfieOptions::default(),
            profile_kernels: false,
            target_gpu: "h100".to_string(),
            dtype: "bf16".to_string(),
            manifest_output_path: None,
            profile_source_text: None,
            profile_source_file_name: None,
            health_monitor: false,
            health_flush_interval: None,
            optim_state_offload: false,
            checkpoint_blocks: false,
            checkpoint_selective: false,
            checkpoint_budget_mib: None,
            checkpoint_stride: CheckpointStride::default(),
            fuse_rmsnorm_backward: false,
            checkpoint_compress: None,
            layerwise_accum: false,
            weight_stream: false,
            stream_arena: false,
            stream_prefetch: false,
            stream_async_writeback: false,
            inspect_enabled: false,
            csha: CshaOptions::default(),
            csha_configs: HashMap::new(),
            checkpoint_policies: HashMap::new(),
            cpdt: CpdtOptions::default(),
            wrga_check: WrgaCheckContext::default(),
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
        crate::wrga_prescan::prescan_adapter_sites_from_decorators(&mut compiler);
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
        // ORDERING INVARIANT: calibration must fire BEFORE
        // compile_flash_attention_kernels (and therefore before compile_main).
        // TWO WGGO passes read compile_options.calibration_sidecar via
        // build_scorer: (1) the WGGO pre-pass inside
        // compile_flash_attention_kernels (the WGGO-before-kernels restructure)
        // and (2) compile_train_block's in-place WGGO pass under compile_main.
        // Firing calibration after either read site leaves the sidecar None
        // there, silently degrading wggo_importance=Auto to magnitude scoring
        // and breaking wggo_importance=Grad outright — and would score the
        // pre-pass and the in-place planner with DIFFERENT importance, so the
        // pre-plan's graph fingerprint would never match and it would be
        // rejected. Firing here (right after wrga_prescan, before any kernel
        // synthesis) keeps both read sites consistent. discover_awq_projections
        // reads only collect_models-era state, which wrga_prescan has already
        // finalized, so nothing the harness needs is produced by the synthesis
        // calls now sequenced below it. Spec §4.1's "after compile_main
        // returns" wording was over-specified; the architectural goal
        // "calibration runs around the compiled code" (§4.3) is preserved.
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
                    // Test-only fault-injection seam — production never
                    // overrides the subprocess's runtime data file.
                    runtime_data_override: None,
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
        compiler.compile_flash_attention_kernels(&parsed.module.stmts)?;
        compiler.compile_user_functions(&parsed.module.stmts)?;
        // M56 Task 17: compile agent method bodies.
        compiler.compile_agent_methods(&parsed.module.stmts)?;
        compiler.compile_batched_functions(&vmap_results)?;
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
