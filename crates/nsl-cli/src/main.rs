mod args;
mod ast_scan;
mod commands;
mod debug;
mod formatter;
mod loader;
mod mangling;
mod resolver;
mod standalone;

use std::path::PathBuf;
use std::process;

use clap::Parser as ClapParser;

use nsl_errors::{Level, SourceMap};
use nsl_lexer::Interner;
use crate::args::Cli;

/// Scan a parsed module for a top-level `train { ... }` block.
///
/// Dev Tools Phase 4 Task 6: `nsl run --monitor` uses this detection to
/// decide whether to enable the health monitor (train program) or fall
/// back to the Phase 1/2 kernel-timing profile path (non-train program).
/// Train blocks in NSL are a top-level construct, so a flat scan suffices.
pub(crate) fn has_train_block(module: &nsl_ast::Module) -> bool {
    module
        .stmts
        .iter()
        .any(|s| matches!(s.kind, nsl_ast::stmt::StmtKind::TrainBlock(_)))
}


fn main() {
    // Windows debug builds easily overflow the 1MB main-thread stack
    // because NSL's compile pipeline has deeply-nested passes (WRGA +
    // WGGO + source-AD + Cranelift lowering).  Run the real entry
    // point on a thread with a 16MB stack.  Release builds don't need
    // this but the cost is negligible.
    let child = std::thread::Builder::new()
        .name("nsl-main".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(main_inner)
        .expect("failed to spawn nsl-main thread");
    match child.join() {
        Ok(()) => {}
        Err(_) => std::process::exit(101),
    }
}

fn main_inner() {
    let cli = Cli::parse();

    match cli {
        Cli::Check(args) => commands::check::dispatch(args),
        Cli::Build(args) => commands::build::dispatch(args),
        Cli::Run(args) => commands::run::dispatch(args),
        Cli::Test { file, filter } => {
            commands::test::run_test(&file, filter.as_deref());
        }
        Cli::Export {
            file,
            output,
            format,
        } => {
            commands::export::run_export(&file, output.as_deref(), format.as_deref());
        }
        Cli::Convert { input, output } => {
            commands::convert::run_convert(&input, &output);
        }
        Cli::Init { name } => {
            commands::init::run_init(&name);
        }
        Cli::Fmt { files, check } => {
            commands::fmt::run_fmt(&files, check);
        }
        Cli::Debug {
            file,
            find_nan,
            diff,
            export_chrome,
        } => {
            debug::run_debug(
                &file,
                find_nan,
                diff.as_deref(),
                export_chrome.as_deref(),
            );
        }
        Cli::Zk { cmd } => {
            commands::build::run_zk_cmd(cmd);
        }
        Cli::Profile {
            file,
            target,
            dtype,
            batch,
            seq,
            dim,
            no_fusion,
            memory,
            entry,
            json,
            explain_wggo,
        } => {
            let args = nsl_cli::profile::ProfileArgs {
                file,
                target,
                dtype,
                batch,
                seq,
                dim,
                fusion: !no_fusion,
                memory,
                entry,
                json,
                explain_wggo,
            };
            match nsl_cli::profile::run_profile(&args) {
                Ok(s) => println!("{s}"),
                Err(e) => {
                    eprintln!("error: {e}");
                    process::exit(1);
                }
            }
        }
        Cli::Tokenize { dirs, output, vocab_size, min_freq, ext } => {
            commands::tokenize::run_tokenize(&dirs, &output, vocab_size, min_freq, &ext);
        }

        Cli::FpgaCompile { file, output_dir, fixture, test_taps, seq } => {
            if let Err(e) =
                commands::fpga::run_fpga_compile(&file, fixture.as_ref(), output_dir.as_ref(), test_taps, seq)
            {
                eprintln!("error: {e}");
                process::exit(1);
            }
        }
    }
}

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
    };
    commands::build::apply_wrga_target_override(&mut inputs);
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
    };
    commands::build::apply_wrga_target_override(&mut inputs);
    inputs
}
