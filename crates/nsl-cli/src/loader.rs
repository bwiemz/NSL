use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use nsl_ast::decl::{FromImportStmt, ImportItems, ImportStmt};
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_ast::{Module, Symbol};
use nsl_errors::{Level, SourceMap};
use nsl_lexer::Interner;
use nsl_semantic::checker::TypeMap;
use nsl_semantic::scope::{ScopeId, ScopeMap};
use nsl_semantic::types::Type;

use nsl_ast::expr::ExprKind;

use crate::resolver;

/// Data for a single parsed and analyzed module.
#[allow(dead_code)] // Fields reserved for future tooling (LSP, debugger, etc.)
pub struct ModuleData {
    pub path: PathBuf,
    pub ast: Module,
    pub type_map: TypeMap,
    pub scopes: ScopeMap,
    /// Exported top-level symbols: name → Type
    pub exports: HashMap<Symbol, Type>,
    /// Struct layouts extracted from codegen (populated during codegen phase)
    pub struct_names: Vec<String>,
    /// Enum variant → tag mapping
    pub enum_variants: HashMap<String, i64>,
    /// Enum name → [(variant_name, tag)]
    pub enum_defs: HashMap<String, Vec<(String, i64)>>,
    /// Paths this module imports from
    pub dependencies: Vec<PathBuf>,
    /// Module prefix for name mangling (empty for entry module)
    pub module_prefix: String,
}

/// The complete module graph for a compilation unit.
pub struct ModuleGraph {
    pub entry: PathBuf,
    pub modules: HashMap<PathBuf, ModuleData>,
    /// Topological order: dependencies before dependents. Entry module is last.
    pub dep_order: Vec<PathBuf>,
}

/// Represents a discovered import — either `from X import Y` or `import X as alias`.
#[derive(Clone)]
enum ImportInfo {
    From(FromImportStmt),
    Alias { #[allow(dead_code)] stmt: ImportStmt, alias: Symbol },
}

/// Scan a module's AST for import statements and resolve them to file paths.
fn discover_imports(
    stmts: &[Stmt],
    source_file: &Path,
    interner: &Interner,
) -> Result<Vec<(PathBuf, ImportInfo)>, String> {
    let mut imports = Vec::new();

    for stmt in stmts {
        match &stmt.kind {
            StmtKind::FromImport(from_import) => {
                let resolved = resolver::resolve_import(
                    &from_import.module_path,
                    source_file,
                    interner,
                )?;
                imports.push((resolved, ImportInfo::From(from_import.clone())));
            }
            StmtKind::Import(import_stmt) if import_stmt.alias.is_some() => {
                let resolved = resolver::resolve_import(
                    &import_stmt.path,
                    source_file,
                    interner,
                )?;
                imports.push((resolved, ImportInfo::Alias {
                    stmt: import_stmt.clone(),
                    alias: import_stmt.alias.unwrap(),
                }));
            }
            StmtKind::Import(import_stmt)
                if matches!(import_stmt.items, ImportItems::Glob | ImportItems::Named(_)) =>
            {
                let resolved = resolver::resolve_import(
                    &import_stmt.path,
                    source_file,
                    interner,
                )?;
                // Treat `import X.*` / `import X.{Y,Z}` like `from X import *` / `from X import {Y,Z}`
                imports.push((resolved, ImportInfo::From(FromImportStmt {
                    module_path: import_stmt.path.clone(),
                    items: import_stmt.items.clone(),
                    span: import_stmt.span,
                })));
            }
            _ => {}
        }
    }

    Ok(imports)
}

/// Scan statements for train blocks and inject synthetic `from nsl.optim.X import *`
/// imports so the optimizer stdlib modules are compiled and linked automatically.
/// Also injects `from nsl.nn.losses import *` since train blocks commonly use loss functions.
fn inject_train_block_imports(
    stmts: &[Stmt],
    interner: &mut Interner,
) -> Vec<Stmt> {
    let mut synthetic_stmts = Vec::new();

    for stmt in stmts {
        if let StmtKind::TrainBlock(train) = &stmt.kind {
            for section in &train.sections {
                if let nsl_ast::block::TrainSection::Scheduler(expr) = section {
                    // Auto-import nsl.optim.schedulers when a scheduler section is present
                    if let ExprKind::Call { .. } = &expr.kind {
                        let sched_segments = ["nsl", "optim", "schedulers"];
                        let module_path: Vec<Symbol> = sched_segments
                            .iter()
                            .map(|s| Symbol(interner.get_or_intern(s)))
                            .collect();

                        let dummy_span = nsl_ast::Span {
                            file_id: nsl_errors::FileId(0),
                            start: nsl_errors::BytePos(0),
                            end: nsl_errors::BytePos(0),
                        };

                        synthetic_stmts.push(Stmt {
                            kind: StmtKind::FromImport(FromImportStmt {
                                module_path,
                                items: ImportItems::Glob,
                                span: dummy_span,
                            }),
                            span: dummy_span,
                            id: nsl_ast::NodeId::next(),
                        });
                    }
                }
                if let nsl_ast::block::TrainSection::Optimizer(expr) = section {
                    // Extract optimizer name from call expression: e.g. SGD(lr=0.01)
                    if let ExprKind::Call { callee, .. } = &expr.kind {
                        if let ExprKind::Ident(sym) = &callee.kind {
                            let name = interner.resolve(sym.0).unwrap_or("").to_lowercase();
                            let module_segments = match name.as_str() {
                                "sgd" => Some(["nsl", "optim", "sgd"]),
                                "adam" => Some(["nsl", "optim", "adam"]),
                                "adamw" => Some(["nsl", "optim", "adamw"]),
                                "lion" => Some(["nsl", "optim", "lion"]),
                                "muon" => Some(["nsl", "optim", "muon"]),
                                "soap" => Some(["nsl", "optim", "soap"]),
                                _ => None,
                            };

                            if let Some(segments) = module_segments {
                                let module_path: Vec<Symbol> = segments
                                    .iter()
                                    .map(|s| Symbol(interner.get_or_intern(s)))
                                    .collect();

                                let dummy_span = nsl_ast::Span {
                                    file_id: nsl_errors::FileId(0),
                                    start: nsl_errors::BytePos(0),
                                    end: nsl_errors::BytePos(0),
                                };

                                synthetic_stmts.push(Stmt {
                                    kind: StmtKind::FromImport(FromImportStmt {
                                        module_path,
                                        items: ImportItems::Glob,
                                        span: dummy_span,
                                    }),
                                    span: dummy_span,
                                    id: nsl_ast::NodeId::next(),
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    synthetic_stmts
}

/// Extract exported symbols from an analyzed module.
/// Walks top-level statements and looks up their types in the root scope.
fn extract_exports(
    stmts: &[Stmt],
    scopes: &ScopeMap,
    _interner: &Interner,
) -> HashMap<Symbol, Type> {
    let mut exports = HashMap::new();

    for stmt in stmts {
        let sym = match &stmt.kind {
            StmtKind::FnDef(fn_def) => Some(fn_def.name),
            StmtKind::StructDef(sd) => Some(sd.name),
            StmtKind::EnumDef(ed) => Some(ed.name),
            StmtKind::ModelDef(md) => Some(md.name),
            _ => None,
        };

        if let Some(name) = sym {
            if let Some((_scope_id, info)) = scopes.lookup(ScopeId::ROOT, name) {
                exports.insert(name, info.ty.clone());
            }
        }
    }

    exports
}

/// Extract enum info from an analyzed module's AST + types for codegen import.
#[allow(clippy::type_complexity)]
fn extract_enum_info(
    stmts: &[Stmt],
    interner: &Interner,
) -> (HashMap<String, i64>, HashMap<String, Vec<(String, i64)>>) {
    let mut variants = HashMap::new();
    let mut defs = HashMap::new();

    for stmt in stmts {
        if let StmtKind::EnumDef(ed) = &stmt.kind {
            let enum_name = interner.resolve(ed.name.0).unwrap_or("<unknown>").to_string();
            let mut variant_list = Vec::new();

            for (i, variant) in ed.variants.iter().enumerate() {
                let vname = interner.resolve(variant.name.0).unwrap_or("<unknown>").to_string();
                let tag = i as i64;
                // Only store qualified names for cross-module imports to avoid
                // collisions when two modules export enums with the same variant name.
                // The codegen adds bare names for locally-defined enums.
                variants.insert(format!("{enum_name}.{vname}"), tag);
                variant_list.push((vname, tag));
            }

            defs.insert(enum_name, variant_list);
        }
    }

    (variants, defs)
}

/// Load all modules starting from the entry file.
///
/// Algorithm:
/// 1. Parse entry file, discover its imports
/// 2. Iteratively resolve and parse imported modules
/// 3. Topologically sort by dependencies
/// 4. Analyze each module in dependency order, passing real types from predecessors
pub fn load_all_modules(
    entry_file: &Path,
    source_map: &mut SourceMap,
    interner: &mut Interner,
) -> Result<ModuleGraph, String> {
    let entry_path = entry_file
        .canonicalize()
        .map_err(|e| format!("cannot canonicalize entry file '{}': {e}", entry_file.display()))?;

    // Phase 1: Parse all modules (iterative worklist)
    let mut parsed: HashMap<PathBuf, (Module, Vec<(PathBuf, ImportInfo)>)> = HashMap::new();
    let mut worklist: Vec<PathBuf> = vec![entry_path.clone()];

    while let Some(file_path) = worklist.pop() {
        if parsed.contains_key(&file_path) {
            continue;
        }

        // Read and parse
        let source = std::fs::read_to_string(&file_path)
            .map_err(|e| format!("could not read '{}': {e}", file_path.display()))?;

        let file_id = source_map.add_file(file_path.display().to_string(), source.clone());
        let (tokens, lex_errors) = nsl_lexer::tokenize(&source, file_id, interner);

        for diag in &lex_errors {
            source_map.emit_diagnostic(diag);
        }
        let has_lex_errors = lex_errors.iter().any(|d| d.level == Level::Error);

        let parse_result = nsl_parser::parse(&tokens, interner);
        for diag in &parse_result.diagnostics {
            source_map.emit_diagnostic(diag);
        }
        let has_parse_errors = parse_result.diagnostics.iter().any(|d| d.level == Level::Error);

        if has_lex_errors || has_parse_errors {
            return Err(format!("errors in '{}'", file_path.display()));
        }

        // For the entry file, auto-inject imports for train block optimizer/scheduler modules
        let mut module = parse_result.module;
        if file_path == entry_path {
            let synthetic = inject_train_block_imports(&module.stmts, interner);
            if !synthetic.is_empty() {
                // Prepend synthetic imports so discover_imports picks them up
                let mut new_stmts = synthetic;
                new_stmts.extend(module.stmts);
                module.stmts = new_stmts;
            }
        }

        // Discover imports from this module
        let imports = discover_imports(&module.stmts, &file_path, interner)?;

        // Add new dependencies to worklist
        for (dep_path, _) in &imports {
            if !parsed.contains_key(dep_path) {
                worklist.push(dep_path.clone());
            }
        }

        parsed.insert(file_path.clone(), (module, imports));
    }

    // Circular imports are detected by topological_sort (DFS with in_stack detection)

    // Phase 2: Topological sort
    let dep_order = topological_sort(&entry_path, &parsed)?;

    // Phase 3: Analyze in dependency order
    let mut modules: HashMap<PathBuf, ModuleData> = HashMap::new();

    for path in &dep_order {
        let (ast, imports) = parsed.remove(path).unwrap();

        // Build import_types from already-analyzed dependencies
        let mut import_types: HashMap<Symbol, Type> = HashMap::new();
        for (dep_path, import_info) in &imports {
            if let Some(dep_data) = modules.get(dep_path) {
                match import_info {
                    ImportInfo::From(from_import) => {
                        inject_import_types(from_import, &dep_data.exports, &mut import_types);
                    }
                    ImportInfo::Alias { alias, .. } => {
                        inject_alias_import(*alias, &dep_data.exports, &mut import_types);
                    }
                }
            }
        }

        // Analyze with real imported types
        let analysis = nsl_semantic::analyze_with_imports(&ast, interner, &import_types, false);

        for diag in &analysis.diagnostics {
            source_map.emit_diagnostic(diag);
        }
        let has_errors = analysis.diagnostics.iter().any(|d| d.level == Level::Error);
        if has_errors {
            return Err(format!("type errors in '{}'", path.display()));
        }

        // Extract exports for downstream modules
        let exports = extract_exports(&ast.stmts, &analysis.scopes, interner);

        // Extract enum info
        let (enum_variants, enum_defs) = extract_enum_info(&ast.stmts, interner);

        // Collect struct names for later codegen import
        let struct_names: Vec<String> = ast.stmts.iter().filter_map(|s| {
            if let StmtKind::StructDef(sd) = &s.kind {
                Some(interner.resolve(sd.name.0).unwrap_or("<unknown>").to_string())
            } else {
                None
            }
        }).collect();

        let dep_paths: Vec<PathBuf> = imports.iter().map(|(p, _)| p.clone()).collect();

        // Compute module prefix for name mangling.
        // Entry module gets empty prefix (no mangling).
        let base_dir = entry_path.parent().unwrap_or(Path::new("."));
        let module_prefix = if *path == entry_path {
            String::new()
        } else {
            crate::mangling::module_prefix(path, base_dir)
        };

        modules.insert(path.clone(), ModuleData {
            path: path.clone(),
            ast,
            type_map: analysis.type_map,
            scopes: analysis.scopes,
            exports,
            struct_names,
            enum_variants,
            enum_defs,
            dependencies: dep_paths,
            module_prefix,
        });
    }

    Ok(ModuleGraph {
        entry: entry_path,
        modules,
        dep_order,
    })
}

/// Inject types from a dependency's exports into the import_types map,
/// based on the `from X import {a, b}` statement.
fn inject_import_types(
    from_import: &FromImportStmt,
    dep_exports: &HashMap<Symbol, Type>,
    import_types: &mut HashMap<Symbol, Type>,
) {
    match &from_import.items {
        ImportItems::Named(items) => {
            for item in items {
                let local_name = item.alias.unwrap_or(item.name);
                if let Some(ty) = dep_exports.get(&item.name) {
                    import_types.insert(local_name, ty.clone());
                }
                // If not found in exports, it will remain Type::Unknown
                // and the semantic checker will report it
            }
        }
        ImportItems::Glob => {
            // Import everything from the dependency
            for (sym, ty) in dep_exports {
                import_types.insert(*sym, ty.clone());
            }
        }
        ImportItems::Module => {
            // `from X import` without names — no-op for now
        }
    }
}

/// Inject all exports from a dependency as a `Type::Module` under the alias symbol.
fn inject_alias_import(
    alias: Symbol,
    dep_exports: &HashMap<Symbol, Type>,
    import_types: &mut HashMap<Symbol, Type>,
) {
    let exports: HashMap<Symbol, Box<Type>> = dep_exports
        .iter()
        .map(|(sym, ty)| (*sym, Box::new(ty.clone())))
        .collect();
    import_types.insert(alias, Type::Module { exports });
}

/// Topological sort of modules. Dependencies come before dependents.
#[allow(clippy::type_complexity)]
fn topological_sort(
    entry: &PathBuf,
    parsed: &HashMap<PathBuf, (Module, Vec<(PathBuf, ImportInfo)>)>,
) -> Result<Vec<PathBuf>, String> {
    let mut order = Vec::new();
    let mut visited = HashSet::new();
    let mut in_stack = HashSet::new();

    #[allow(clippy::type_complexity)]
    fn visit(
        path: &PathBuf,
        parsed: &HashMap<PathBuf, (Module, Vec<(PathBuf, ImportInfo)>)>,
        visited: &mut HashSet<PathBuf>,
        in_stack: &mut HashSet<PathBuf>,
        order: &mut Vec<PathBuf>,
    ) -> Result<(), String> {
        if visited.contains(path) {
            return Ok(());
        }
        if in_stack.contains(path) {
            return Err(format!("circular import detected involving '{}'", path.display()));
        }

        in_stack.insert(path.clone());

        if let Some((_ast, imports)) = parsed.get(path) {
            for (dep_path, _) in imports {
                visit(dep_path, parsed, visited, in_stack, order)?;
            }
        }

        in_stack.remove(path);
        visited.insert(path.clone());
        order.push(path.clone());
        Ok(())
    }

    visit(entry, parsed, &mut visited, &mut in_stack, &mut order)?;
    Ok(order)
}
