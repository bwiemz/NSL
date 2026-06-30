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
pub(crate) fn module_data_to_wrga_inputs(m: &crate::loader::ModuleData) -> nsl_codegen::WrgaInputs {
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
    crate::commands::build::apply_wrga_check_overrides(&mut inputs);
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
        })
        .collect()
}

pub(crate) fn analysis_to_wrga_inputs(a: &nsl_semantic::AnalysisResult) -> nsl_codegen::WrgaInputs {
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
    crate::commands::build::apply_wrga_check_overrides(&mut inputs);
    inputs
}
