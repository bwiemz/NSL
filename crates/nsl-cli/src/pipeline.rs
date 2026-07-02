//! Compiler-frontend plumbing shared by the command handlers: lex/parse/
//! semantic-analyze a file, and build WRGA / fused-CE codegen inputs.
//!
//! Extracted from main.rs; behavior is unchanged.

use std::path::PathBuf;
use std::process;

use nsl_errors::{Level, SourceMap};
use nsl_lexer::Interner;

pub(crate) fn frontend(file: &PathBuf) -> (Interner, nsl_parser::ParseResult, nsl_semantic::AnalysisResult) {
    frontend_with_flags(file, false)
}

pub(crate) fn frontend_with_flags(
    file: &PathBuf,
    linear_types: bool,
) -> (Interner, nsl_parser::ParseResult, nsl_semantic::AnalysisResult) {
    let source = match std::fs::read_to_string(file) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: could not read file '{}': {e}", file.display());
            process::exit(1);
        }
    };

    let mut source_map = SourceMap::new();
    let file_id = source_map.add_file(file.display().to_string(), source.clone());

    let mut interner = Interner::new();

    // Lex
    let (tokens, lex_errors) = nsl_lexer::tokenize(&source, file_id, &mut interner);

    for diag in &lex_errors {
        source_map.emit_diagnostic(diag);
    }

    // Parse
    let parse_result = nsl_parser::parse(&tokens, &mut interner);

    for diag in &parse_result.diagnostics {
        source_map.emit_diagnostic(diag);
    }

    // Semantic analysis — thread linear_types so E0610 fires correctly.
    let analysis = nsl_semantic::analyze_with_imports(
        &parse_result.module,
        &mut interner,
        &std::collections::HashMap::new(),
        linear_types,
    );

    for diag in &analysis.diagnostics {
        source_map.emit_diagnostic(diag);
    }

    let total_errors = lex_errors
        .iter()
        .chain(parse_result.diagnostics.iter())
        .chain(analysis.diagnostics.iter())
        .filter(|d| d.level == Level::Error)
        .count();

    if total_errors > 0 {
        eprintln!("{total_errors} error(s) found");
        process::exit(1);
    }

    (interner, parse_result, analysis)
}

/// Convert WRGA decorator configs captured by nsl-semantic into the codegen-side
/// `WrgaInputs` newtype (Task 1 of WRGA bridge). Keeps nsl-codegen free of a
/// direct dependency on nsl-semantic.
pub(crate) fn module_data_to_wrga_inputs(
    m: &crate::loader::ModuleData,
    ctx: &nsl_codegen::WrgaCheckContext,
) -> nsl_codegen::WrgaInputs {
    use nsl_codegen::{
        AdapterDecoratorConfig, AdapterKind, FreezeDecoratorConfig, WrgaDecoratorConfig,
        WrgaInputs,
    };
    let mut inputs = WrgaInputs {
        wrga: m
            .wrga_configs
            .iter()
            .map(|c| WrgaDecoratorConfig {
                mode: c.block.mode,
                budget: c.block.budget,
                target: None,
                layers: c.block.layers.clone(),
                custom_adapter: c.adapter_name.clone(),
            })
            .collect(),
        freeze: m
            .freeze_configs
            .iter()
            .map(|c| FreezeDecoratorConfig {
                exclude: c.exclude.clone(),
                include: c.include.clone(),
            })
            .collect(),
        adapter: m
            .adapter_configs
            .iter()
            .map(|c| AdapterDecoratorConfig {
                kind: match c.kind {
                    nsl_semantic::wrga::AdapterKind::Lora => AdapterKind::Lora,
                    nsl_semantic::wrga::AdapterKind::Ia3 => AdapterKind::Ia3,
                    nsl_semantic::wrga::AdapterKind::GatedLora => AdapterKind::GatedLora,
                },
                targets: c.targets.clone(),
                rank: c.rank,
                alpha: c.alpha,
            })
            .collect(),
        ablation: Default::default(),
    };
    apply_wrga_check_overrides(&mut inputs, ctx);
    inputs
}

/// CFTP §4.4 G3 (Sprint 2): bridge `@fused_lm_ce(...)` configs from
/// `AnalysisResult` into the codegen-side `FusedCeDecoratorConfig` newtype.
/// Mirrors `analysis_to_wrga_inputs` — keeps nsl-codegen free of a direct
/// nsl-semantic dependency.
pub(crate) fn analysis_to_fused_ce_configs(
    a: &nsl_semantic::AnalysisResult,
) -> Vec<nsl_codegen::FusedCeDecoratorConfig> {
    a.fused_ce_configs
        .iter()
        .map(|c| nsl_codegen::FusedCeDecoratorConfig {
            enabled: c.enabled,
            vocab_tile: c.vocab_tile,
            vocab_size: c.vocab_size,
            hidden_size: c.hidden_size,
            batch_size: c.batch_size,
            seq_len: c.seq_len,
            dtype: c.dtype.map(|d| match d {
                nsl_semantic::cftp::FusedCeDtypeHint::F32 => nsl_codegen::FusedCeDtypeHint::F32,
                nsl_semantic::cftp::FusedCeDtypeHint::F16 => nsl_codegen::FusedCeDtypeHint::F16,
                nsl_semantic::cftp::FusedCeDtypeHint::Bf16 => nsl_codegen::FusedCeDtypeHint::Bf16,
            }),
            // CFTP v10 (item 3): forward the AST NodeId so codegen can
            // dispatch the right decorator per train block.
            train_block_stmt_id: c.train_block_stmt_id,
        })
        .collect()
}

/// CFTP §4.3 G2 Strategy 3 (Item 4): bridge `@pca(strategy=...)` configs
/// from `AnalysisResult` into the codegen-side `PcaUserStrategy` enum.
/// Mirrors `analysis_to_fused_ce_configs` — keeps nsl-codegen free of a
/// direct nsl-semantic dependency. Crosses the crate boundary via
/// `PcaStrategy::as_str` (the wire-format the codegen-side parser
/// understands; future strategies extend both sides in lock-step).
pub(crate) fn analysis_to_pca_user_strategies(
    a: &nsl_semantic::AnalysisResult,
) -> Vec<nsl_codegen::PcaUserStrategy> {
    a.pca_configs
        .iter()
        .map(|c| nsl_codegen::PcaUserStrategy::from_semantic_str(c.strategy.as_str()))
        .collect()
}

/// Mirror of `analysis_to_pca_user_strategies` for the multi-file path
/// that consumes `ModuleData` rather than `AnalysisResult`.
pub(crate) fn module_data_to_pca_user_strategies(
    m: &crate::loader::ModuleData,
) -> Vec<nsl_codegen::PcaUserStrategy> {
    m.pca_configs
        .iter()
        .map(|c| nsl_codegen::PcaUserStrategy::from_semantic_str(c.strategy.as_str()))
        .collect()
}

/// Mirror of `module_data_to_wrga_inputs` for `@fused_lm_ce` configs.
/// Used by multi-file paths that consume the entry module's `ModuleData`
/// rather than a single `AnalysisResult`.
pub(crate) fn module_data_to_fused_ce_configs(
    m: &crate::loader::ModuleData,
) -> Vec<nsl_codegen::FusedCeDecoratorConfig> {
    m.fused_ce_configs
        .iter()
        .map(|c| nsl_codegen::FusedCeDecoratorConfig {
            enabled: c.enabled,
            vocab_tile: c.vocab_tile,
            vocab_size: c.vocab_size,
            hidden_size: c.hidden_size,
            batch_size: c.batch_size,
            seq_len: c.seq_len,
            dtype: c.dtype.map(|d| match d {
                nsl_semantic::cftp::FusedCeDtypeHint::F32 => nsl_codegen::FusedCeDtypeHint::F32,
                nsl_semantic::cftp::FusedCeDtypeHint::F16 => nsl_codegen::FusedCeDtypeHint::F16,
                nsl_semantic::cftp::FusedCeDtypeHint::Bf16 => nsl_codegen::FusedCeDtypeHint::Bf16,
            }),
            // CFTP v10 (item 3): forward the AST NodeId so codegen can
            // dispatch the right decorator per train block.
            train_block_stmt_id: c.train_block_stmt_id,
        })
        .collect()
}

/// Sprint 2 (paper §6.2): convert the semantic side-table of validated
/// `@csha(...)` decorators into the `HashMap<String, CshaConfig>` shape
/// `CompileOptions.csha_configs` expects. The CSHA hook in
/// `nsl-codegen/src/stmt.rs` looks up `model_type_name` here and applies the
/// per-model `disable=` / `level=` / `target=` overrides. Empty when the
/// program has no `@csha` decorators.
pub(crate) fn analysis_to_csha_configs(
    a: &nsl_semantic::AnalysisResult,
) -> std::collections::HashMap<String, nsl_semantic::csha::CshaConfig> {
    a.csha_configs.iter().cloned().collect()
}

/// Mirror of `analysis_to_csha_configs` for multi-file paths that consume the
/// entry module's `ModuleData` rather than a single `AnalysisResult`.
pub(crate) fn module_data_to_csha_configs(
    m: &crate::loader::ModuleData,
) -> std::collections::HashMap<String, nsl_semantic::csha::CshaConfig> {
    m.csha_configs.iter().cloned().collect()
}

/// Cycle-10 §5.3 paper checkpointing-aware backward (Task 6):
/// expose `EffectChecker::checkpoint_policies()` (already published on
/// `AnalysisResult.checkpoint_policies`) in the
/// `HashMap<String, CheckpointPolicy>` shape `CompileOptions.checkpoint_policies`
/// expects. Empty when no `@checkpoint(policy="...")` decorators are present.
pub(crate) fn analysis_to_checkpoint_policies(
    a: &nsl_semantic::AnalysisResult,
) -> std::collections::HashMap<String, nsl_semantic::effects::CheckpointPolicy> {
    a.checkpoint_policies.clone()
}

/// Mirror of `analysis_to_checkpoint_policies` for multi-file paths that
/// consume the entry module's `ModuleData` rather than a single
/// `AnalysisResult`.
pub(crate) fn module_data_to_checkpoint_policies(
    m: &crate::loader::ModuleData,
) -> std::collections::HashMap<String, nsl_semantic::effects::CheckpointPolicy> {
    m.checkpoint_policies.clone()
}

pub(crate) fn analysis_to_wrga_inputs(
    a: &nsl_semantic::AnalysisResult,
    ctx: &nsl_codegen::WrgaCheckContext,
) -> nsl_codegen::WrgaInputs {
    use nsl_codegen::{
        AdapterDecoratorConfig, AdapterKind, FreezeDecoratorConfig, WrgaDecoratorConfig,
        WrgaInputs,
    };
    let mut inputs = WrgaInputs {
        wrga: a
            .wrga_configs
            .iter()
            .map(|c| WrgaDecoratorConfig {
                mode: c.block.mode,
                budget: c.block.budget,
                target: None, // Symbol->string resolution happens at codegen if needed
                layers: c.block.layers.clone(),
                custom_adapter: c.adapter_name.clone(),
            })
            .collect(),
        freeze: a
            .freeze_configs
            .iter()
            .map(|c| FreezeDecoratorConfig {
                exclude: c.exclude.clone(),
                include: c.include.clone(),
            })
            .collect(),
        adapter: a
            .adapter_configs
            .iter()
            .map(|c| AdapterDecoratorConfig {
                kind: match c.kind {
                    nsl_semantic::wrga::AdapterKind::Lora => AdapterKind::Lora,
                    nsl_semantic::wrga::AdapterKind::Ia3 => AdapterKind::Ia3,
                    nsl_semantic::wrga::AdapterKind::GatedLora => AdapterKind::GatedLora,
                },
                targets: c.targets.clone(),
                rank: c.rank,
                alpha: c.alpha,
            })
            .collect(),
        ablation: Default::default(),
    };
    apply_wrga_check_overrides(&mut inputs, ctx);
    inputs
}

/// Apply the CLI-side `nsl check --wrga-analyze | --wrga-compare` overrides onto
/// a freshly-built `WrgaInputs` before it ships to codegen. Both overrides are
/// carried explicitly on [`nsl_codegen::WrgaCheckContext`] (formerly CLI
/// thread-locals); on normal `nsl build` paths every field is `None` and this
/// is a quick no-op.
///
/// 1. `target_override` (paper §8.3) — copied onto every
///    `WrgaDecoratorConfig::target`. When the source has no `@wrga(...)` at all
///    (only `@freeze` / `@adapter`), a minimal Auto-mode config is inserted so
///    the target choice still reaches the codegen-side `wrga::run`.
/// 2. `ablation_override` (paper §9.3) — copied onto `WrgaInputs::ablation` so
///    the codegen-side WRGA driver honours the requested per-Innovation skip
///    flags.
fn apply_wrga_check_overrides(
    inputs: &mut nsl_codegen::WrgaInputs,
    ctx: &nsl_codegen::WrgaCheckContext,
) {
    if let Some(target) = ctx.target_override.clone() {
        for cfg in &mut inputs.wrga {
            cfg.target = Some(target.clone());
        }
        if inputs.wrga.is_empty() {
            inputs.wrga.push(nsl_codegen::WrgaDecoratorConfig {
                mode: nsl_ast::block::WrgaMode::Auto,
                budget: None,
                target: Some(target),
                layers: Vec::new(),
                custom_adapter: None,
            });
        }
    }
    if let Some(abl) = ctx.ablation_override {
        inputs.ablation = abl;
    }
}

#[cfg(test)]
mod checkpoint_decorator_cli_wireup_tests {
    //! Lock-in test for the phase2.6 <- main merge regression that silently
    //! dropped `ModuleData.csha_configs` + `ModuleData.checkpoint_policies`
    //! and the two corresponding `pipeline::*` helpers when taking main's
    //! refactored loader.rs. Without these helpers the per-model
    //! `@csha(disable|level|target)` decorator effects and ALL
    //! `@checkpoint(policy="full")` decorator effects silently no-op
    //! because the codegen-side consumers (stmt.rs train/grad extractors,
    //! compiler/kernel.rs CSHA training PTX backward,
    //! calibration/binary_codegen.rs model_backward) read empty maps.
    //!
    //! This test exists ONLY to prevent the regression from silently
    //! re-occurring; it asserts the helper return-type round-trips
    //! non-empty when the analysis has captured the decorator.
    //! Keep it minimal — full per-flag round-trip coverage lives in
    //! crates/nsl-semantic/tests/csha_decorator_binding.rs.
    use super::*;
    use nsl_errors::FileId;
    use nsl_lexer::Interner;

    fn analyze_source(src: &str) -> nsl_semantic::AnalysisResult {
        let mut interner = Interner::new();
        let (tokens, _lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
        let parse_result = nsl_parser::parse(&tokens, &mut interner);
        nsl_semantic::analyze(&parse_result.module, &mut interner)
    }

    /// Lock-in: `@checkpoint(policy="full")` on a fn body must flow from
    /// `AnalysisResult.checkpoint_policies` through
    /// `pipeline::analysis_to_checkpoint_policies` non-empty.  If this
    /// asserts empty, the loader->CompileOptions wiring has regressed
    /// (the codegen-side `WengertExtractor::with_checkpoint_policies`
    /// installer will silently no-op and `@checkpoint(policy="full")`
    /// will have zero effect on backward-pass codegen).
    #[test]
    fn checkpoint_full_policy_survives_pipeline_handoff() {
        let src = r#"
@checkpoint(policy="full")
fn forward_step(x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
    return x
"#;
        let analysis = analyze_source(src);
        let policies = analysis_to_checkpoint_policies(&analysis);
        assert!(
            !policies.is_empty(),
            "regression: @checkpoint(policy=\"full\") was captured by \
             semantic analysis ({} entry/entries) but pipeline helper \
             returned an empty map — the phase2.6<-main loader.rs merge \
             likely dropped ModuleData.checkpoint_policies again. \
             analysis.checkpoint_policies = {:?}",
            analysis.checkpoint_policies.len(),
            analysis.checkpoint_policies,
        );
    }

    /// Companion lock-in for the @csha decorator wiring (Sprint 2 — paper
    /// §6.2). Same regression class as the checkpoint case above:
    /// `ModuleData.csha_configs` was dropped along with checkpoint_policies
    /// in the merge, so per-model @csha overrides silently no-op.
    #[test]
    fn csha_decorator_survives_pipeline_handoff() {
        let src = r#"
@csha(level=2)
model Toy:
    layer lin: Linear<16, 16>

    fn forward(self, x: Tensor<[4, 16], f32>) -> Tensor<[4, 16], f32>:
        return self.lin(x)
"#;
        let analysis = analyze_source(src);
        let configs = analysis_to_csha_configs(&analysis);
        assert!(
            !configs.is_empty(),
            "regression: @csha(level=2) was captured by semantic analysis \
             ({} entry/entries) but pipeline helper returned an empty map. \
             analysis.csha_configs = {:?}",
            analysis.csha_configs.len(),
            analysis.csha_configs,
        );
    }
}
