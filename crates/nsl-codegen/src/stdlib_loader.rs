//! Minimal stdlib loader for test helpers.
//!
//! Intentionally narrower than nsl-cli's `load_all_modules` — file I/O stays
//! OUT of nsl-runtime; the CLI loader is unchanged.  This helper exists so
//! `debug_compile_and_return_plan` can take a raw NSL source string and still
//! resolve stdlib optimizer signatures required by `train` blocks.
//!
//! Scope (B.2 Task 1): for each `optimizer: <name>(...)` section found in
//! the entry's train block(s), locate the matching `stdlib/nsl/optim/<name>.nsl`,
//! parse its `FnDef`s, and produce `(raw_name, mangled_name, Signature)`
//! triples suitable for feeding to
//! `compile_module_with_imports_best_effort_plan` as `imported_fns`.
//!
//! Limitations (accepted):
//! - No transitive stdlib import resolution.  Only directly-needed optimizer
//!   modules are loaded.
//! - No analysis of stdlib modules; signatures are derived straight from the
//!   AST via `Compiler::build_fn_signature`.  That's sufficient for codegen's
//!   fallback lookup of `sgd_step`/`adam_step`/... in
//!   `stmt.rs::compile_optimizer_call`.

use std::path::PathBuf;

use cranelift_codegen::ir::Signature;
use nsl_ast::block::TrainSection;
use nsl_ast::decl::FnDef;
use nsl_ast::expr::ExprKind;
use nsl_ast::stmt::StmtKind;
use nsl_ast::Module;

use crate::CodegenError;

/// Return all valid stdlib root directories, ordered by priority.
///
/// Mirrors `nsl-cli::resolver::stdlib_roots` but kept local to avoid a
/// nsl-cli -> nsl-codegen reverse dependency.
fn stdlib_roots() -> Vec<PathBuf> {
    let mut roots = Vec::new();
    if let Ok(env_path) = std::env::var("NSL_STDLIB_PATH") {
        let p = PathBuf::from(&env_path);
        if p.is_dir() {
            roots.push(p);
        }
    }
    if let Ok(cwd) = std::env::current_dir() {
        let stdlib = cwd.join("stdlib");
        if stdlib.is_dir() && !roots.iter().any(|r| r == &stdlib) {
            roots.push(stdlib);
        }
    }
    roots
}

/// Locate a stdlib file by dotted module path (e.g. `"nsl.optim.sgd"`).
fn locate_stdlib_file(dotted: &str) -> Option<PathBuf> {
    let rel: PathBuf = dotted.split('.').collect::<PathBuf>().with_extension("nsl");
    for root in stdlib_roots() {
        let candidate = root.join(&rel);
        if candidate.is_file() {
            return Some(candidate);
        }
    }
    None
}

/// Scan the entry module for train-block optimizer names and return the
/// stdlib modules they need.  Walks top-level statements *and* function
/// bodies so that `train` blocks nested inside `fn main()` are found.
fn discover_optimizer_modules(entry: &Module, interner: &nsl_lexer::Interner) -> Vec<String> {
    let mut modules = Vec::new();
    fn walk(
        stmts: &[nsl_ast::stmt::Stmt],
        interner: &nsl_lexer::Interner,
        modules: &mut Vec<String>,
    ) {
        for stmt in stmts {
            match &stmt.kind {
                StmtKind::TrainBlock(train) => {
                    for section in &train.sections {
                        match section {
                            TrainSection::Optimizer(expr) => {
                                if let ExprKind::Call { callee, .. } = &expr.kind {
                                    if let ExprKind::Ident(sym) = &callee.kind {
                                        let name =
                                            interner.resolve(sym.0).unwrap_or("").to_lowercase();
                                        let module = match name.as_str() {
                                            "sgd" => Some("nsl.optim.sgd"),
                                            "adam" => Some("nsl.optim.adam"),
                                            "adamw" => Some("nsl.optim.adamw"),
                                            "lion" => Some("nsl.optim.lion"),
                                            "muon" => Some("nsl.optim.muon"),
                                            "soap" => Some("nsl.optim.soap"),
                                            _ => None,
                                        };
                                        if let Some(m) = module {
                                            if !modules.iter().any(|x: &String| x == m) {
                                                modules.push(m.to_string());
                                            }
                                        }
                                    }
                                }
                            }
                            TrainSection::Scheduler(_) => {
                                let m = "nsl.optim.schedulers".to_string();
                                if !modules.iter().any(|x| x == &m) {
                                    modules.push(m);
                                }
                            }
                            _ => {}
                        }
                    }
                }
                StmtKind::FnDef(fn_def) => {
                    walk(&fn_def.body.stmts, interner, modules);
                }
                _ => {}
            }
        }
    }
    walk(&entry.stmts, interner, &mut modules);
    modules
}

/// Compile a mangled symbol prefix from a dotted stdlib path using CLI's
/// convention: `stripped-rel-path-with-os-sep-replaced-by-underscore`.
/// `nsl.optim.sgd` -> `nsl_optim_sgd`.
fn module_prefix_for(dotted: &str) -> String {
    dotted.replace('.', "_")
}

/// Parse a stdlib file and return its top-level `FnDef`s.  Returns an empty
/// list if the file can't be read or parsed (the caller will then fail later
/// at codegen time with a clearer error).
fn load_stdlib_fn_defs(path: &PathBuf, interner: &mut nsl_lexer::Interner) -> Vec<FnDef> {
    let source = match std::fs::read_to_string(path) {
        Ok(s) => s,
        Err(_) => return Vec::new(),
    };
    let (tokens, _lex_diags) = nsl_lexer::tokenize(&source, nsl_errors::FileId(0), interner);
    let parse_result = nsl_parser::parse(&tokens, interner);
    parse_result
        .module
        .stmts
        .into_iter()
        .filter_map(|stmt| match stmt.kind {
            StmtKind::FnDef(fn_def) => Some(fn_def),
            _ => None,
        })
        .collect()
}

/// Build `imported_fns` triples for all stdlib optimizer modules needed by
/// `entry`.  Each triple is `(raw_name, mangled_name, Signature)`.
pub fn build_imported_fns_for_entry(
    entry: &Module,
    interner: &mut nsl_lexer::Interner,
    type_map: &nsl_semantic::checker::TypeMap,
    options: &crate::CompileOptions,
) -> Result<Vec<(String, String, Signature)>, CodegenError> {
    let modules = discover_optimizer_modules(entry, interner);
    if modules.is_empty() {
        return Ok(Vec::new());
    }

    // Phase 1: parse stdlib sources into AST fragments.  All mutation of
    // `interner` happens here, before we borrow it immutably for codegen.
    let mut module_fn_defs: Vec<(String, Vec<FnDef>)> = Vec::new();
    for module in &modules {
        let Some(path) = locate_stdlib_file(module) else {
            // Silently skip unresolved stdlib modules — codegen will emit its
            // usual "undefined function" error if the symbol is really needed.
            continue;
        };
        let fn_defs = load_stdlib_fn_defs(&path, interner);
        if !fn_defs.is_empty() {
            module_fn_defs.push((module.clone(), fn_defs));
        }
    }

    // Phase 2: build signatures via a throwaway Compiler.  We deliberately
    // do NOT run any codegen passes on the stdlib AST — we only need the
    // signature shape so codegen's optimizer-call fallback lookup succeeds.
    let sig_compiler = crate::compiler::Compiler::new(interner, type_map, options)?;

    let mut imported_fns = Vec::new();
    for (module, fn_defs) in &module_fn_defs {
        let prefix = module_prefix_for(module);
        for fn_def in fn_defs {
            let raw = sig_compiler.resolve_sym(fn_def.name).to_string();
            let mangled = format!("{prefix}__{raw}");
            let sig = sig_compiler.build_fn_signature(fn_def);
            imported_fns.push((raw, mangled, sig));
        }
    }
    Ok(imported_fns)
}
