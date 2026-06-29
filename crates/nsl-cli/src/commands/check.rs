//! `nsl check` — type-check / shape-check without running, with optional
//! token/AST/type dumps.
//!
//! Extracted verbatim from `main.rs`; behavior is unchanged.

use std::path::PathBuf;
use std::process;

use nsl_errors::{Level, SourceMap};
use nsl_lexer::Interner;

pub(crate) fn run_check(file: &PathBuf, dump_tokens: bool, dump_ast: bool, dump_types: bool, linear_types: bool) {
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

    if dump_tokens {
        for token in &tokens {
            println!("{:?}", token);
        }
    }

    for diag in &lex_errors {
        source_map.emit_diagnostic(diag);
    }

    // Parse
    let parse_result = nsl_parser::parse(&tokens, &mut interner);

    for diag in &parse_result.diagnostics {
        source_map.emit_diagnostic(diag);
    }

    if dump_ast {
        match serde_json::to_string_pretty(&parse_result.module) {
            Ok(json) => println!("{json}"),
            Err(e) => eprintln!("error serializing AST: {e}"),
        }
    }

    // Semantic analysis (with ownership checking when --linear-types is active)
    let analysis = nsl_semantic::analyze_with_imports(
        &parse_result.module, &mut interner, &std::collections::HashMap::new(), linear_types,
    );

    for diag in &analysis.diagnostics {
        source_map.emit_diagnostic(diag);
    }

    if dump_types {
        println!("=== Type Map ===");
        let mut entries: Vec<_> = analysis.type_map.iter().collect();
        entries.sort_by_key(|(id, _)| id.0);
        for (id, ty) in entries {
            println!("  node {}: {}", id.0, nsl_semantic::types::display_type(ty));
        }
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
    } else {
        println!(
            "OK: {} checked successfully ({} statements)",
            file.display(),
            parse_result.module.stmts.len()
        );
    }
}

