mod args;
mod ast_scan;
mod commands;
mod debug;
mod formatter;
mod loader;
mod pipeline;
mod mangling;
mod resolver;
mod standalone;

use std::process;

use clap::Parser as ClapParser;

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
        Cli::PtxMetadata { file } => {
            commands::ptx_metadata::run(&file);
        }
    }
}
