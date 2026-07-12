//! Decorator bridges: pure `AnalysisResult -> CompileOptions field`
//! transforms shared by every build path (standalone / shared_lib / zk /
//! check) AND the `nsl profile` real-capture path.
//!
//! These moved out of the bin-only `pipeline.rs` so the lib-side
//! `profile::try_capture_real` can enrich its capture `CompileOptions` from
//! the SAME `AnalysisResult` it compiles (the configs are NodeId-keyed, so
//! enrichment built from a different parse would silently fail to match).
//! `pipeline.rs` re-exports these names, so the existing bin call sites are
//! unchanged. Only the `module_data_to_*` mirrors (which need the bin-only
//! `loader::ModuleData`) stay in `pipeline.rs`.

use std::collections::HashMap;

pub fn analysis_to_fused_ce_configs(
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

/// CPKD: bridge `@fused_kl_ce(...)` configs from `AnalysisResult` into
/// the codegen-side `FusedKlCeDecoratorConfig` newtype. Mirrors
/// `analysis_to_fused_ce_configs`.
pub fn analysis_to_fused_kl_ce_configs(
    a: &nsl_semantic::AnalysisResult,
) -> Vec<nsl_codegen::FusedKlCeDecoratorConfig> {
    a.fused_kl_ce_configs
        .iter()
        .map(|c| nsl_codegen::FusedKlCeDecoratorConfig {
            enabled: c.enabled,
            vocab_size: c.vocab_size,
            student_hidden: c.hidden_size,
            teacher_hidden: c.teacher_hidden,
            batch_size: c.batch_size,
            seq_len: c.seq_len,
            vocab_tile: c.vocab_tile,
            distill_block_stmt_id: c.distill_block_stmt_id,
        })
        .collect()
}

/// CFTP §4.3 G2 Strategy 3 (Item 4): bridge `@pca(strategy=...)` configs
/// from `AnalysisResult` into the codegen-side `PcaUserStrategy` enum.
/// Crosses the crate boundary via `PcaStrategy::as_str` (the wire-format
/// the codegen-side parser understands).
pub fn analysis_to_pca_user_strategies(
    a: &nsl_semantic::AnalysisResult,
) -> Vec<nsl_codegen::PcaUserStrategy> {
    a.pca_configs
        .iter()
        .map(|c| nsl_codegen::PcaUserStrategy::from_semantic_str(c.strategy.as_str()))
        .collect()
}

/// Sprint 2 (paper §6.2): convert the semantic side-table of validated
/// `@csha(...)` decorators into the `HashMap<String, CshaConfig>` shape
/// `CompileOptions.csha_configs` expects. Empty when the program has no
/// `@csha` decorators.
pub fn analysis_to_csha_configs(
    a: &nsl_semantic::AnalysisResult,
) -> HashMap<String, nsl_semantic::csha::CshaConfig> {
    a.csha_configs.iter().cloned().collect()
}

/// Cycle-10 §5.3 paper checkpointing-aware backward (Task 6): expose
/// `EffectChecker::checkpoint_policies()` in the
/// `HashMap<String, CheckpointPolicy>` shape `CompileOptions.checkpoint_policies`
/// expects. Empty when no `@checkpoint(policy="...")` decorators are present.
pub fn analysis_to_checkpoint_policies(
    a: &nsl_semantic::AnalysisResult,
) -> HashMap<String, nsl_semantic::effects::CheckpointPolicy> {
    a.checkpoint_policies.clone()
}

pub fn analysis_to_wrga_inputs(
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

/// Apply the CLI-side `nsl check --wrga-analyze | --wrga-compare` overrides
/// onto a freshly-built `WrgaInputs` before it ships to codegen. Both
/// overrides are carried explicitly on [`nsl_codegen::WrgaCheckContext`]; on
/// normal `nsl build` / `nsl profile` paths every field is `None` and this
/// is a quick no-op. `pub` so the bin-side `module_data_to_wrga_inputs`
/// (multi-file path, `ModuleData`-based) can share it.
pub fn apply_wrga_check_overrides(
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
