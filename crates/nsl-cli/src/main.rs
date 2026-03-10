mod loader;
mod mangling;
mod resolver;

use std::collections::HashMap;
use std::path::PathBuf;
use std::process;

use clap::Parser as ClapParser;

use nsl_errors::{Level, SourceMap};
use nsl_lexer::Interner;

#[derive(ClapParser)]
#[command(name = "nsl", about = "NeuralScript Language Toolchain", version)]
enum Cli {
    /// Parse and type-check an NSL file
    Check {
        /// Path to the .nsl file
        file: PathBuf,

        /// Print the token stream
        #[arg(long)]
        dump_tokens: bool,

        /// Print the AST as JSON
        #[arg(long)]
        dump_ast: bool,

        /// Print the inferred type map
        #[arg(long)]
        dump_types: bool,
    },

    /// Compile and execute an NSL program
    Run {
        /// Path to the .nsl file
        file: PathBuf,

        /// Arguments to pass to the compiled program
        #[arg(last = true)]
        args: Vec<String>,
    },

    /// Compile an NSL file to a native executable
    Build {
        /// Path to the .nsl file
        file: PathBuf,

        /// Output file path (default: input stem + .exe on Windows)
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Emit only the object file (skip linking)
        #[arg(long)]
        emit_obj: bool,

        /// Print the Cranelift IR for each function
        #[arg(long)]
        dump_ir: bool,
    },
}

fn main() {
    let cli = Cli::parse();

    match cli {
        Cli::Check {
            file,
            dump_tokens,
            dump_ast,
            dump_types,
        } => {
            run_check(&file, dump_tokens, dump_ast, dump_types);
        }
        Cli::Build {
            file,
            output,
            emit_obj,
            dump_ir,
        } => {
            run_build(&file, output, emit_obj, dump_ir);
        }
        Cli::Run { file, args } => {
            run_run(&file, &args);
        }
    }
}

fn frontend(file: &PathBuf) -> (Interner, nsl_parser::ParseResult, nsl_semantic::AnalysisResult) {
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

    // Semantic analysis
    let analysis = nsl_semantic::analyze(&parse_result.module, &mut interner);

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

fn run_check(file: &PathBuf, dump_tokens: bool, dump_ast: bool, dump_types: bool) {
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

    // Semantic analysis
    let analysis = nsl_semantic::analyze(&parse_result.module, &mut interner);

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

/// Check if a file has any import statements by quick-scanning.
fn has_imports(file: &PathBuf) -> bool {
    if let Ok(source) = std::fs::read_to_string(file) {
        source.lines().any(|line| {
            let trimmed = line.trim();
            (trimmed.starts_with("from ") && trimmed.contains(" import "))
                || (trimmed.starts_with("import ") && trimmed.contains(" as "))
        })
    } else {
        false
    }
}

fn run_build(file: &PathBuf, output: Option<PathBuf>, emit_obj: bool, dump_ir: bool) {
    run_build_inner(file, output, emit_obj, dump_ir, false);
}

fn run_build_inner(file: &PathBuf, output: Option<PathBuf>, emit_obj: bool, dump_ir: bool, quiet: bool) {
    if has_imports(file) {
        run_build_multi(file, output, emit_obj, dump_ir, quiet);
    } else {
        run_build_single(file, output, emit_obj, dump_ir, quiet);
    }
}

/// Single-file build (backward compatible, fast path).
fn run_build_single(file: &PathBuf, output: Option<PathBuf>, emit_obj: bool, dump_ir: bool, quiet: bool) {
    let (interner, parse_result, analysis) = frontend(file);

    // Codegen
    let obj_bytes = match nsl_codegen::compile(
        &parse_result.module,
        &interner,
        &analysis.type_map,
        dump_ir,
    ) {
        Ok(bytes) => bytes,
        Err(e) => {
            eprintln!("codegen error: {e}");
            process::exit(1);
        }
    };

    // Determine output paths
    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| {
            eprintln!("error: invalid input filename '{}'", file.display());
            process::exit(1);
        });
    let obj_path = file.with_file_name(format!("{stem}.o"));

    // Write object file
    if let Err(e) = std::fs::write(&obj_path, &obj_bytes) {
        eprintln!("error: could not write object file: {e}");
        process::exit(1);
    }

    if emit_obj {
        if !quiet { println!("Wrote {}", obj_path.display()); }
        return;
    }

    // Link
    let exe_path = if let Some(out) = output {
        out
    } else {
        nsl_codegen::linker::default_output_path(file)
    };

    match nsl_codegen::linker::link(&obj_path, &exe_path) {
        Ok(()) => {
            // Clean up .o file after successful link
            let _ = std::fs::remove_file(&obj_path);
            if !quiet { println!("Built {}", exe_path.display()); }
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }
}

/// Multi-file build with module system.
fn run_build_multi(file: &PathBuf, output: Option<PathBuf>, emit_obj: bool, dump_ir: bool, quiet: bool) {
    let mut source_map = SourceMap::new();
    let mut interner = Interner::new();

    // Load and analyze all modules
    let graph = match loader::load_all_modules(file, &mut source_map, &mut interner) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    let temp_dir = std::env::temp_dir().join("nsl_build");
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("error: could not create temp dir: {e}");
        process::exit(1);
    }

    let mut obj_files: Vec<PathBuf> = Vec::new();

    // Compile each module in dependency order
    for path in &graph.dep_order {
        let mod_data = &graph.modules[path];
        let is_entry = *path == graph.entry;

        let obj_bytes = if is_entry {
            // Entry module: import functions from all dependencies
            let mut imported_fns = Vec::new();
            let mut imported_struct_layouts = HashMap::new();
            let mut imported_enum_variants = HashMap::new();
            let mut imported_enum_defs = HashMap::new();

            // Collect imports from ALL dependency modules (not just direct deps)
            for dep_path in &graph.dep_order {
                if dep_path == &graph.entry {
                    continue;
                }
                let dep_data = &graph.modules[dep_path];

                // Build function signatures and struct layouts using a temporary compiler
                let mut temp_compiler = match nsl_codegen::compiler::Compiler::new(&interner, &dep_data.type_map) {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("codegen error: {e}");
                        process::exit(1);
                    }
                };

                for stmt in &dep_data.ast.stmts {
                    if let nsl_ast::stmt::StmtKind::FnDef(fn_def) = &stmt.kind {
                        let raw_name = interner.resolve(fn_def.name.0).unwrap_or("<unknown>").to_string();
                        let mangled_name = crate::mangling::mangle(&dep_data.module_prefix, &raw_name);
                        let sig = temp_compiler.build_fn_signature(fn_def);
                        imported_fns.push((raw_name, mangled_name, sig));
                    }
                }

                // Extract struct layouts from dependency
                if let Err(e) = temp_compiler.collect_structs(&dep_data.ast.stmts) {
                    eprintln!("codegen error collecting structs from '{}': {e}", dep_path.display());
                    process::exit(1);
                }
                for (name, layout) in temp_compiler.struct_layouts.drain() {
                    imported_struct_layouts.insert(name, layout);
                }

                // Import enum variants/defs
                imported_enum_variants.extend(dep_data.enum_variants.clone());
                imported_enum_defs.extend(dep_data.enum_defs.clone());
            }

            match nsl_codegen::compile_entry(
                &mod_data.ast,
                &interner,
                &mod_data.type_map,
                &imported_fns,
                imported_struct_layouts,
                imported_enum_variants,
                imported_enum_defs,
                dump_ir,
            ) {
                Ok(bytes) => bytes,
                Err(e) => {
                    eprintln!("codegen error in '{}': {e}", path.display());
                    process::exit(1);
                }
            }
        } else {
            // Library module: export all functions
            match nsl_codegen::compile_module(
                &mod_data.ast,
                &interner,
                &mod_data.type_map,
                &mod_data.module_prefix,
                dump_ir,
            ) {
                Ok(bytes) => bytes,
                Err(e) => {
                    eprintln!("codegen error in '{}': {e}", path.display());
                    process::exit(1);
                }
            }
        };

        // Write .o file — use index to avoid name collisions (e.g., math/utils.nsl vs string/utils.nsl)
        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let obj_path = temp_dir.join(format!("{stem}_{}.o", obj_files.len()));

        if let Err(e) = std::fs::write(&obj_path, &obj_bytes) {
            eprintln!("error: could not write object file '{}': {e}", obj_path.display());
            process::exit(1);
        }

        obj_files.push(obj_path);
    }

    if emit_obj {
        if !quiet {
            for obj in &obj_files {
                println!("Wrote {}", obj.display());
            }
        }
        return;
    }

    // Link all .o files
    let exe_path = if let Some(out) = output {
        out
    } else {
        nsl_codegen::linker::default_output_path(file)
    };

    match nsl_codegen::linker::link_multi(&obj_files, &exe_path) {
        Ok(()) => {
            // Clean up .o files
            for obj in &obj_files {
                let _ = std::fs::remove_file(obj);
            }
            if !quiet { println!("Built {}", exe_path.display()); }
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }
}

fn run_run(file: &PathBuf, program_args: &[String]) {
    let temp_dir = std::env::temp_dir().join("nsl_run");
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("error: could not create temp dir: {e}");
        process::exit(1);
    }

    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("program");
    let exe_name = if cfg!(target_os = "windows") {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    };
    let exe_path = temp_dir.join(&exe_name);

    // Build to temp dir (reuse existing build logic, quiet mode)
    run_build_inner(file, Some(exe_path.clone()), false, false, true);

    // Execute the compiled program
    let status = std::process::Command::new(&exe_path)
        .args(program_args)
        .status()
        .unwrap_or_else(|e| {
            eprintln!("error: could not execute '{}': {e}", exe_path.display());
            process::exit(1);
        });

    // Clean up
    let _ = std::fs::remove_file(&exe_path);
    let _ = std::fs::remove_dir(&temp_dir);

    // Forward exit code
    process::exit(status.code().unwrap_or(1));
}
