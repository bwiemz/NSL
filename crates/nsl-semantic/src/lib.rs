pub mod agent;
pub mod builtins;
pub mod cep;
pub mod cfie;
pub mod cftp;
pub mod cpkd;
pub mod csha;
pub mod checker;
pub mod cpdt;
pub mod context_parallel;
pub mod determinism;
pub mod effects;
pub mod export;
pub mod fp8;
pub mod grammar;
pub mod inspect;
pub mod kv_compress;
pub mod moe;
pub mod multimodal;
pub mod nan_analysis;
pub mod ownership;
pub mod pipeline;
pub mod ownership_autodiff;
pub mod ownership_walker;
pub mod perf_budget;
pub mod resolve;
pub mod scope;
pub mod shape_algebra;
pub mod shapes;
pub mod sparse;
pub mod sparse_layout;
pub mod speculative;
pub mod target;
pub mod types;
pub mod vmap;
pub mod wggo;
pub mod wrga;

use std::collections::{HashMap, HashSet};

use nsl_ast::{Module, Symbol};
use nsl_errors::Diagnostic;
use nsl_lexer::Interner;

use crate::checker::{TypeChecker, TypeMap};
use crate::export::WeightIndexMap;
use crate::scope::ScopeMap;
use crate::types::Type;

/// Maps imported symbol names to their resolved types from other modules.
pub type ImportTypes = HashMap<Symbol, Type>;

/// Per-function ownership metadata extracted during M38a analysis.
#[derive(Debug, Clone, Default)]
pub struct FunctionOwnershipInfo {
    /// Parameters that are consumed (moved) by this function.
    pub linear_params: Vec<Symbol>,
    /// Parameters that are @shared (refcounted, multiple uses OK).
    pub shared_params: Vec<Symbol>,
}

/// Result of semantic analysis.
pub struct AnalysisResult {
    pub diagnostics: Vec<Diagnostic>,
    pub type_map: TypeMap,
    pub scopes: ScopeMap,
    /// M38a: Per-function ownership metadata (function name → info).
    /// Only populated when linear_types analysis is enabled.
    pub ownership_info: HashMap<String, FunctionOwnershipInfo>,
    /// WRGA: validated `@wrga(...)` configurations for later consumption by
    /// `nsl-codegen::wrga::run`.  One entry per decorator occurrence.
    pub wrga_configs: Vec<crate::wrga::WrgaConfig>,
    /// WRGA: validated `@freeze(...)` configurations.
    pub freeze_configs: Vec<crate::wrga::FreezeConfig>,
    /// WRGA: validated `@adapter(...)` configurations.
    pub adapter_configs: Vec<crate::wrga::AdapterConfig>,
    /// CSHA Sprint 2: validated `@csha(...)` configurations, keyed by the
    /// decorated model's name (or the LHS binding name for
    /// `@csha let m = SomeModel()`). The CLI converts these into the
    /// `CompileOptions.csha_configs` side-channel that the CSHA hook in
    /// `nsl-codegen/src/stmt.rs` consults per-model so `level=`, `target=`,
    /// and `disable=` take observable effect on codegen.
    pub csha_configs: Vec<(String, crate::csha::CshaConfig)>,
    /// M62: Maps each `self.<field>` MemberAccess expression NodeId (inside an
    /// `@export` model method) to the declaration-order index of that field in
    /// the model's tensor-weight list.  Consumed by codegen to lower
    /// `self.W` → `load(weight_ptrs + index * 8)`.
    pub weight_index_map: WeightIndexMap,
    /// Cycle-10 §5.3 (Task 3): per-function checkpointing policies, as
    /// parsed from `@checkpoint(policy="...")` kwargs. Consumed by the
    /// `nsl-cli` loader at `crates/nsl-cli/src/loader.rs` (Task 6) to
    /// pass into `WengertExtractor::with_checkpoint_policy`.
    pub checkpoint_policies: HashMap<String, crate::effects::CheckpointPolicy>,
    /// Cycle-10 §5.3 (Task 4): models annotated `@paged_kv`. Consumed by
    /// the codegen-side R9 cross-scope refusal predicate at
    /// `flash_attention_v2/mod.rs::synthesize_backward_with_tier`.
    pub paged_kv_models: HashSet<String>,
    /// CFTP §4.4 G3 (Sprint 2): validated `@fused_lm_ce(...)` configurations
    /// captured during semantic analysis.  One entry per decorated `train`
    /// block.  Codegen plumbs these via `CompileOptions.fused_ce_configs`
    /// (Sprint 2 wires the collection; Sprint 2.5 will use these to drive
    /// the lowering-site auto-substitution of composite cross_entropy → the
    /// fused linear-CE kernel).
    pub fused_ce_configs: Vec<crate::cftp::FusedCeConfig>,
    /// CPKD: validated `@fused_kl_ce(...)` decorator configs, one per
    /// decorated `distill` block. Codegen plumbs these via
    /// `CompileOptions.fused_kl_ce_configs` (same bridge pattern as
    /// `fused_ce_configs`).
    pub fused_kl_ce_configs: Vec<crate::cpkd::FusedKlCeConfig>,
    /// CFTP §4.3 G2 Strategy 3 (Item 4): validated `@pca(...)` configurations.
    /// One entry per decorator occurrence. Codegen reads these at the
    /// CSHA training-PTX synthesis site so that
    /// `@pca(strategy=per_document)` flips
    /// `PerDocAdmitConfig::enable_per_doc_cta=true` for the planner.
    /// Pre-Item-4 the validated config was dropped at
    /// `nsl_semantic::checker::stmt::stmt_decorator`.
    pub pca_configs: Vec<crate::cftp::PcaConfig>,
}

/// Run semantic analysis on a parsed module (single-file, backward compatible).
pub fn analyze(module: &Module, interner: &mut Interner) -> AnalysisResult {
    analyze_with_imports(module, interner, &HashMap::new(), false)
}

/// Run semantic analysis with pre-resolved import types from other modules.
pub fn analyze_with_imports(
    module: &Module,
    interner: &mut Interner,
    import_types: &ImportTypes,
    linear_types: bool,
) -> AnalysisResult {
    let mut scopes = ScopeMap::new();

    // Phase 1: Register built-in types and functions
    builtins::register_builtins(&mut scopes, interner);

    // Phase 2: Run the type checker with import context
    let mut checker = TypeChecker::new(interner, &mut scopes);
    checker.set_import_types(import_types);
    // M56: thread the flag so E0610 can fire during collect_top_level_decls.
    checker.linear_types_enabled = linear_types;
    checker.check_module(module);

    // M51: Effect analysis is now integrated into TypeChecker — propagation
    // and validation run at the end of check_module(). Diagnostics are merged.

    // Extract results from type checker, releasing its borrows on interner/scopes
    let mut diagnostics = checker.diagnostics;
    let type_map = checker.type_map;
    let wrga_configs = checker.wrga_configs;
    let freeze_configs = checker.freeze_configs;
    let adapter_configs = checker.adapter_configs;
    let csha_configs = checker.csha_configs;
    // Cycle-10 §5.3 (Tasks 3/4): extract checkpoint policies and
    // @paged_kv membership from the effect checker for downstream
    // codegen consumption.
    let checkpoint_policies = checker.effect_checker.checkpoint_policies().clone();
    let paged_kv_models = checker.effect_checker.paged_kv_models().clone();
    let fused_ce_configs = checker.fused_ce_configs;
    let fused_kl_ce_configs = checker.fused_kl_ce_configs;
    let pca_configs = checker.pca_configs;

    // M62: Run `@export` decorator validation.  Pure-additive — appends
    // diagnostics without touching other analysis state.  Also returns the
    // weight-index side-table for codegen consumption.
    let (export_diags, weight_index_map) = crate::export::validate_exports(module, interner);
    diagnostics.extend(export_diags);

    // WRGA paper §8.2: validate `@wrga(adapter=<Ident>, ...)` references to
    // user-defined adapter models.  Runs after `check_module` so all top-
    // level model declarations are visible; reports undeclared / empty /
    // forward-less adapters loudly.
    {
        let resolve = |sym: nsl_ast::Symbol| {
            interner.resolve(sym.0).unwrap_or("<unknown>").to_string()
        };
        let custom_adapter_diags = crate::wrga::validate_wrga_custom_adapters(
            &wrga_configs,
            module,
            &scopes,
            &resolve,
        );
        diagnostics.extend(custom_adapter_diags);
    }

    // M38a: Run ownership analysis when --linear-types is active.
    // Runs after type checking so we have the TypeMap for tensor type detection.
    let mut ownership_info = HashMap::new();
    if linear_types {
        let (ownership_diags, info) = ownership_walker::analyze_ownership(
            module, interner, &type_map, &scopes,
        );
        diagnostics.extend(ownership_diags);
        ownership_info = info;
    }

    // M56 Task 12: agent analysis pipeline.
    // Order: registry → APG extraction → cycle detection → device compat
    // → cross-agent access → cross-agent mutation → fan-out.
    //
    // The E0610 flag gate already ran inside TypeChecker::check_module above;
    // we do NOT re-run it here. We still run the rest of the pipeline
    // unconditionally so the user sees ALL agent-related errors in one
    // compile pass, not just the flag-gate complaint.
    {
        let mut agent_registry = crate::agent::AgentRegistry::new();
        agent_registry.register_module(module, interner);

        let mut apgs = Vec::new();
        crate::agent::extract_apgs(
            module,
            &agent_registry,
            interner,
            &mut apgs,
            &mut diagnostics,
        );

        for apg in &apgs {
            crate::agent::detect_cycles(apg, interner, &mut diagnostics);
            crate::agent::check_device_compatibility(
                apg,
                module,
                &agent_registry,
                interner,
                &mut diagnostics,
            );
            crate::agent::check_fan_out(apg, interner, &mut diagnostics);
        }

        crate::agent::check_cross_agent_field_access(
            module,
            &agent_registry,
            interner,
            &mut diagnostics,
        );
        crate::agent::check_cross_agent_mutation(
            module,
            &agent_registry,
            interner,
            &mut diagnostics,
        );
    }

    AnalysisResult {
        diagnostics,
        type_map,
        scopes,
        ownership_info,
        wrga_configs,
        freeze_configs,
        adapter_configs,
        csha_configs,
        weight_index_map,
        checkpoint_policies,
        paged_kv_models,
        fused_ce_configs,
        fused_kl_ce_configs,
        pca_configs,
    }
}
