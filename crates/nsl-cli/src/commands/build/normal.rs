//! The default `nsl build` path: single-file and multi-file object emit + link.
//!
//! `needs_multi_file` decides which path runs; `run_build` / `run_build_inner`
//! are the thin dispatchers other commands call. Extracted verbatim from the
//! former monolithic `build.rs`; behavior is unchanged.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process;

use nsl_errors::SourceMap;
use nsl_lexer::Interner;

use super::reports::{capture_wrga_plan, check_wrga_report_preconditions};

/// Check if a file has any import statements or train blocks by quick-scanning.
/// Train blocks need multi-file compilation because optimizer stdlib modules
/// are auto-imported.
pub(super) fn needs_multi_file(file: &PathBuf) -> bool {
    if let Ok(source) = std::fs::read_to_string(file) {
        source.lines().any(|line| {
            let trimmed = line.trim();
            // Skip comments — they can contain import-like text
            if trimmed.starts_with('#') || trimmed.starts_with("//") {
                return false;
            }
            (trimmed.starts_with("from ") && trimmed.contains(" import "))
                || (trimmed.starts_with("import ") && trimmed.contains(" as "))
                || trimmed.starts_with("train(")
                || trimmed.starts_with("train (")
        })
    } else {
        false
    }
}

pub(crate) fn run_build(
    file: &PathBuf,
    output: Option<PathBuf>,
    emit_obj: bool,
    dump_ir: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    run_build_inner(file, output, emit_obj, dump_ir, false, options, wrga_report);
}

pub(crate) fn run_build_inner(
    file: &PathBuf,
    output: Option<PathBuf>,
    emit_obj: bool,
    dump_ir: bool,
    quiet: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    if needs_multi_file(file) {
        run_build_multi(file, output, emit_obj, dump_ir, quiet, options, wrga_report);
    } else {
        run_build_single(file, output, emit_obj, dump_ir, quiet, options, wrga_report);
    }
}

/// Single-file build (backward compatible, fast path).
fn run_build_single(
    file: &PathBuf,
    output: Option<PathBuf>,
    emit_obj: bool,
    dump_ir: bool,
    quiet: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(file, options.linear_types_enabled);

    // Task 3 (B.1): fail fast if --wrga-report is used without --source-ad
    // when decorators are present — the old silent-notice behaviour was
    // confusing.
    check_wrga_report_preconditions(&analysis, wrga_report, options);

    // Task 1 (WRGA bridge): forward decorator configs captured by nsl-semantic.
    let mut options = options.clone();
    options.wrga_inputs =
        Some(crate::pipeline::analysis_to_wrga_inputs(&analysis, &options.wrga_check));
    options.fused_ce_configs = crate::pipeline::analysis_to_fused_ce_configs(&analysis);
    options.pca_user_strategies = crate::pipeline::analysis_to_pca_user_strategies(&analysis);
    // Sprint 2 (paper §6.2): forward @csha decorator configs so per-model
    // disable/level/target overrides take effect on the multi-module path.
    options.csha_configs = crate::pipeline::analysis_to_csha_configs(&analysis);
    // Cycle-10 §5.3 Task 6: route @checkpoint(policy=...) policies from
    // EffectChecker into CompileOptions so WengertExtractor::with_checkpoint_policies
    // can stamp the prologue + emit a PrologueRecompute marker.
    options.checkpoint_policies = crate::pipeline::analysis_to_checkpoint_policies(&analysis);
    // M62 Task 6: route weight_index_map from semantic analysis into codegen so
    // compile_export_model_methods can resolve self.<field> → weight-array index.
    options.weight_index_map = analysis.weight_index_map.clone();
    let options = &options;

    // M45: Run compile-time NaN risk analysis before codegen if --nan-analysis is set.
    if options.nan_analysis {
        let mut analyzer = nsl_semantic::nan_analysis::NanAnalyzer::new();
        analyzer.analyze_module(&parse_result.module, &interner);
        if analyzer.diagnostics.is_empty() {
            eprintln!("note: --nan-analysis: no NaN/Inf risks detected");
        } else {
            eprintln!(
                "note: --nan-analysis: {} warning(s) detected",
                analyzer.diagnostics.len()
            );
            let mut sm = nsl_errors::SourceMap::new();
            let src = std::fs::read_to_string(file).unwrap_or_default();
            sm.add_file(file.display().to_string(), src);
            for diag in &analyzer.diagnostics {
                sm.emit_diagnostic(diag);
            }
        }
    }

    // Codegen
    let (obj_bytes, wrga_plan) = match nsl_codegen::compile_returning_plan(
        &parse_result.module,
        &interner,
        &analysis.type_map,
        dump_ir,
        options,
    ) {
        Ok((bytes, plan)) => (bytes, plan),
        Err(e) => {
            eprintln!("codegen error: {e}");
            process::exit(1);
        }
    };

    // WRGA Milestone B.1: emit `WrgaPlan::render_report()` if --wrga-report was set.
    // Also offer the plan to the CLI-side capture slot (`--wrga-compare`).
    capture_wrga_plan(&wrga_plan, &options.wrga_check);
    if let Some(report_path) = wrga_report {
        match &wrga_plan {
            Some(p) => {
                let report = p.render_report();
                if report_path == std::path::Path::new("-") {
                    print!("{}", report);
                } else if let Err(e) = std::fs::write(report_path, &report) {
                    eprintln!("error: could not write WRGA report: {e}");
                    process::exit(1);
                }
            }
            None => {
                eprintln!(
                    "nsl: --wrga-report requested but no @train block with WRGA decorators was compiled"
                );
            }
        }
    }

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
fn run_build_multi(
    file: &std::path::Path,
    output: Option<PathBuf>,
    emit_obj: bool,
    dump_ir: bool,
    quiet: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    let mut entry_wrga_plan: Option<nsl_codegen::wrga::WrgaPlan> = None;
    let mut source_map = SourceMap::new();
    let mut interner = Interner::new();

    // Load and analyze all modules
    let graph = match crate::loader::load_all_modules(file, &mut source_map, &mut interner) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    let temp_dir = std::env::temp_dir().join(format!("nsl_build_{}", std::process::id()));
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
            let mut imported_struct_layouts: HashMap<String, nsl_codegen::context::StructLayout> = HashMap::new();
            let mut imported_model_names = std::collections::HashSet::new();
            let mut imported_enum_variants = HashMap::new();
            let mut imported_enum_defs = HashMap::new();
            let mut imported_model_method_bodies: HashMap<String, HashMap<String, nsl_ast::decl::FnDef>> = HashMap::new();
            let mut imported_model_field_types: HashMap<String, HashMap<String, String>> = HashMap::new();

            // Collect imports from ALL dependency modules (not just direct deps)
            for dep_path in &graph.dep_order {
                if dep_path == &graph.entry {
                    continue;
                }
                let dep_data = &graph.modules[dep_path];

                // Build function signatures and struct layouts using a temporary compiler
                let mut temp_compiler = match nsl_codegen::compiler::Compiler::new(&interner, &dep_data.type_map, &nsl_codegen::CompileOptions::default()) {
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

                // Inject previously-collected struct layouts from earlier deps so that
                // collect_models can resolve sub-model field types (e.g., GQA referencing
                // RotaryEmbedding which was collected from the rope module earlier).
                for (name, layout) in &imported_struct_layouts {
                    temp_compiler.types.struct_layouts.insert(name.clone(), layout.clone());
                }

                // Extract struct layouts from dependency (structs first, then models which may reference structs)
                if let Err(e) = temp_compiler.collect_structs(&dep_data.ast.stmts) {
                    eprintln!("codegen error collecting structs from '{}': {e}", dep_path.display());
                    process::exit(1);
                }
                if let Err(e) = temp_compiler.collect_models(&dep_data.ast.stmts) {
                    eprintln!("codegen error collecting models from '{}': {e}", dep_path.display());
                    process::exit(1);
                }

                // Import model constructor and method signatures
                let model_sigs = temp_compiler.build_model_signatures(&dep_data.ast.stmts);
                imported_fns.extend(model_sigs);

                // Collect model names from dep (so entry module won't generate struct ctors for them)
                for stmt in &dep_data.ast.stmts {
                    if let nsl_ast::stmt::StmtKind::ModelDef(md) = &stmt.kind {
                        let model_name = interner.resolve(md.name.0).unwrap_or("<unknown>").to_string();
                        imported_model_names.insert(model_name);
                    }
                }

                for (name, layout) in temp_compiler.types.struct_layouts.drain() {
                    imported_struct_layouts.insert(name, layout);
                }

                // Propagate model field types from temp compiler (populated by collect_models)
                for (name, fields) in temp_compiler.models.model_field_types.drain() {
                    imported_model_field_types.entry(name).or_insert(fields);
                }

                // Extract model method bodies directly from AST (not from temp compiler,
                // because model_method_bodies is only populated by declare_user_functions
                // which is not called on temp compilers).
                for stmt in &dep_data.ast.stmts {
                    if let nsl_ast::stmt::StmtKind::ModelDef(md) = &stmt.kind {
                        let model_name = interner.resolve(md.name.0).unwrap_or("<unknown>").to_string();
                        let mut body_map = HashMap::new();
                        for member in &md.members {
                            if let nsl_ast::decl::ModelMember::Method(fn_def, _) = member {
                                let method_name = interner.resolve(fn_def.name.0).unwrap_or("<unknown>").to_string();
                                body_map.insert(method_name, fn_def.clone());
                            }
                        }
                        if !body_map.is_empty() {
                            imported_model_method_bodies.entry(model_name).or_insert(body_map);
                        }
                    }
                }

                // Import enum variants/defs
                imported_enum_variants.extend(dep_data.enum_variants.clone());
                imported_enum_defs.extend(dep_data.enum_defs.clone());
            }

            // WRGA Milestone B.1: forward entry-module decorator configs to codegen.
            // Task 3: fail fast if --wrga-report is used without --source-ad.
            if wrga_report.is_some() && !options.source_ad {
                let has_wrga_decorators = !mod_data.wrga_configs.is_empty()
                    || !mod_data.freeze_configs.is_empty()
                    || !mod_data.adapter_configs.is_empty();
                if has_wrga_decorators {
                    eprintln!(
                        "nsl: --wrga-report requires --source-ad when WRGA decorators are present; re-run with --source-ad"
                    );
                    process::exit(2);
                }
            }
            let mut entry_options = options.clone();
            entry_options.wrga_inputs = Some(crate::pipeline::module_data_to_wrga_inputs(
                mod_data,
                &entry_options.wrga_check,
            ));
            entry_options.fused_ce_configs = crate::pipeline::module_data_to_fused_ce_configs(mod_data);
            entry_options.pca_user_strategies = crate::pipeline::module_data_to_pca_user_strategies(mod_data);
            // Sprint 2 (paper §6.2): forward entry-module @csha decorator
            // configs so per-model disable/level/target overrides take
            // effect on the multi-file standalone path.
            entry_options.csha_configs = crate::pipeline::module_data_to_csha_configs(mod_data);
            // Cycle-10 §5.3 Task 6: forward @checkpoint(policy=...) policies
            // from the entry module's semantic analysis into CompileOptions.
            entry_options.checkpoint_policies =
                crate::pipeline::module_data_to_checkpoint_policies(mod_data);
            let entry_options = &entry_options;

            match nsl_codegen::compile_entry_returning_plan(
                &mod_data.ast,
                &interner,
                &mod_data.type_map,
                &imported_fns,
                imported_struct_layouts,
                imported_model_names,
                imported_enum_variants,
                imported_enum_defs,
                imported_model_method_bodies,
                imported_model_field_types,
                dump_ir,
                entry_options,
            ) {
                Ok((bytes, plan)) => {
                    entry_wrga_plan = plan;
                    bytes
                }
                Err(e) => {
                    eprintln!("codegen error in '{}': {e}", path.display());
                    process::exit(1);
                }
            }
        } else {
            // Library module: export all functions.
            // If this module has dependencies (imports from other modules), inject their symbols.
            let mut lib_imported_fns = Vec::new();
            let mut lib_struct_layouts: HashMap<String, nsl_codegen::context::StructLayout> = HashMap::new();
            let mut lib_model_names = std::collections::HashSet::new();

            // Check if this module has any imports
            let has_imports = mod_data.ast.stmts.iter().any(|s| {
                matches!(s.kind, nsl_ast::stmt::StmtKind::FromImport(_) | nsl_ast::stmt::StmtKind::Import(_))
            });

            if has_imports {
                // Collect symbols from all transitive deps of this module
                for dep_path in &graph.dep_order {
                    if dep_path == path || dep_path == &graph.entry {
                        continue;
                    }
                    // Only include deps that are before this module in dep_order
                    // (i.e., deps of this module, not modules that depend on it)
                    let dep_idx = graph.dep_order.iter().position(|p| p == dep_path).unwrap_or(usize::MAX);
                    let cur_idx = graph.dep_order.iter().position(|p| p == path).unwrap_or(usize::MAX);
                    if dep_idx >= cur_idx {
                        continue;
                    }

                    let dep_data = &graph.modules[dep_path];
                    let mut temp_compiler = match nsl_codegen::compiler::Compiler::new(&interner, &dep_data.type_map, &nsl_codegen::CompileOptions::default()) {
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
                            lib_imported_fns.push((raw_name, mangled_name, sig));
                        }
                    }

                    // Inject previously-collected struct layouts from earlier deps
                    for (name, layout) in &lib_struct_layouts {
                        temp_compiler.types.struct_layouts.insert(name.clone(), layout.clone());
                    }

                    if let Err(e) = temp_compiler.collect_structs(&dep_data.ast.stmts) {
                        eprintln!("codegen error: {e}");
                        process::exit(1);
                    }
                    if let Err(e) = temp_compiler.collect_models(&dep_data.ast.stmts) {
                        eprintln!("codegen error: {e}");
                        process::exit(1);
                    }

                    let model_sigs = temp_compiler.build_model_signatures(&dep_data.ast.stmts);
                    lib_imported_fns.extend(model_sigs);

                    for stmt in &dep_data.ast.stmts {
                        if let nsl_ast::stmt::StmtKind::ModelDef(md) = &stmt.kind {
                            let model_name = interner.resolve(md.name.0).unwrap_or("<unknown>").to_string();
                            lib_model_names.insert(model_name);
                        }
                    }

                    for (name, layout) in temp_compiler.types.struct_layouts.drain() {
                        lib_struct_layouts.insert(name, layout);
                    }
                }
            }

            match nsl_codegen::compile_module_with_imports(
                &mod_data.ast,
                &interner,
                &mod_data.type_map,
                &mod_data.module_prefix,
                &lib_imported_fns,
                lib_struct_layouts,
                lib_model_names,
                dump_ir,
                options,
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

    // WRGA Milestone B.1: emit `WrgaPlan::render_report()` if --wrga-report was set.
    // Also offer the plan to the CLI-side capture slot (`--wrga-compare`).
    capture_wrga_plan(&entry_wrga_plan, &options.wrga_check);
    if let Some(report_path) = wrga_report {
        match &entry_wrga_plan {
            Some(p) => {
                let report = p.render_report();
                if report_path == std::path::Path::new("-") {
                    print!("{}", report);
                } else if let Err(e) = std::fs::write(report_path, &report) {
                    eprintln!("error: could not write WRGA report: {e}");
                    process::exit(1);
                }
            }
            None => {
                eprintln!(
                    "nsl: --wrga-report requested but no @train block with WRGA decorators was compiled"
                );
            }
        }
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
