//! M62a: `nsl build --shared-lib` — emit a `.so`/`.dylib`/`.dll` with the
//! stable C API, plus a matching C header when `@export` functions exist.
//!
//! Extracted verbatim from the former monolithic `build.rs`; behavior is
//! unchanged.

use std::collections::HashMap;
use std::path::PathBuf;
use std::process;

use nsl_errors::SourceMap;
use nsl_lexer::Interner;

use super::normal::needs_multi_file;
use super::reports::{check_wrga_report_preconditions, emit_wrga_report};

/// M62a: Build as a shared library (.so/.dylib/.dll) with stable C API.
pub(crate) fn run_build_shared(
    file: &PathBuf,
    output: Option<PathBuf>,
    dump_ir: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    if needs_multi_file(file) {
        run_build_shared_multi(file, output, dump_ir, options, wrga_report);
    } else {
        run_build_shared_single(file, output, dump_ir, options, wrga_report);
    }
}

/// M62a: Single-file shared library build.
fn run_build_shared_single(
    file: &PathBuf,
    output: Option<PathBuf>,
    dump_ir: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    let (interner, parse_result, analysis) = crate::pipeline::frontend_with_flags(file, options.linear_types_enabled);

    // Task 3 (B.1): forward WRGA decorator configs so they take effect on the
    // shared-library build path, and fail fast if --wrga-report is combined
    // with decorators but without --source-ad.
    check_wrga_report_preconditions(&analysis, wrga_report, options);
    let mut options = options.clone();
    options.wrga_inputs =
        Some(crate::pipeline::analysis_to_wrga_inputs(&analysis, &options.wrga_check));
    options.fused_ce_configs = crate::pipeline::analysis_to_fused_ce_configs(&analysis);
    options.pca_user_strategies = crate::pipeline::analysis_to_pca_user_strategies(&analysis);
    // M62 Task 6: route weight_index_map from semantic analysis into codegen.
    options.weight_index_map = analysis.weight_index_map.clone();
    // M62: allocate a slot the compiler publishes @export functions into,
    // so we can emit the C header after the shared library is linked.
    let exports_slot: std::sync::Arc<
        std::sync::Mutex<Option<Vec<nsl_codegen::c_header::ExportInfo>>>,
    > = std::sync::Arc::new(std::sync::Mutex::new(None));
    options.export_functions_out = Some(exports_slot.clone());
    let options = &options;

    // Codegen with PIC enabled (shared_lib=true in options)
    let (obj_bytes, wrga_plan) = match nsl_codegen::compile_returning_plan(
        &parse_result.module,
        &interner,
        &analysis.type_map,
        dump_ir,
        options,
    ) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("codegen error: {e}");
            process::exit(1);
        }
    };

    emit_wrga_report(&wrga_plan, wrga_report, &options.wrga_check);

    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or_else(|| {
            eprintln!("error: invalid input filename '{}'", file.display());
            process::exit(1);
        });
    let obj_path = file.with_file_name(format!("{stem}.o"));

    if let Err(e) = std::fs::write(&obj_path, &obj_bytes) {
        eprintln!("error: could not write object file: {e}");
        process::exit(1);
    }

    let lib_path = if let Some(out) = output {
        out
    } else {
        nsl_codegen::linker::default_shared_lib_path(file)
    };

    // M62 Task 6: collect @export symbol names so the linker can force them
    // into the DLL export table (Linkage::Export alone isn't enough on MSVC).
    let export_symbols: Vec<String> = exports_slot
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|v| v.iter().map(|e| e.symbol_name.clone()).collect()))
        .unwrap_or_default();
    // Sibling packed-array dispatch wrappers (`<name>__nsl_dispatch`) emitted
    // alongside every typed `<name>` wrapper. The runtime ExportRegistry
    // dlsyms these via the suffix; MSVC requires them in the explicit export
    // list to survive linking.
    let dispatch_symbols: Vec<String> = export_symbols
        .iter()
        .map(|s| format!("{}__nsl_dispatch", s))
        .collect();
    // M62 Task 9: also re-export the runtime lifecycle symbols so that ctypes
    // callers can call nsl_model_create / nsl_model_destroy / nsl_get_last_error
    // directly from the generated shared lib without loading a separate runtime DLL.
    // `mut` is only used when --features onnx-rt-op is on; harmless otherwise.
    #[allow(unused_mut)]
    let mut runtime_exports: Vec<&'static str> = vec![
        "nsl_model_create",
        "nsl_model_create_with_lib",
        "nsl_model_destroy",
        "nsl_model_forward",
        "nsl_model_forward_grad",
        "nsl_model_backward",
        "nsl_grad_context_destroy",
        "nsl_model_call",
        "nsl_model_call_dlpack",
        "nsl_model_export_count",
        "nsl_model_lookup_function",
        "nsl_model_get_weight_ptrs",
        "nsl_model_get_num_weights",
        "nsl_model_num_weights",
        "nsl_get_last_error",
        "nsl_clear_error",
        "nsl_set_error_cstr",
        "nsl_desc_to_tensor",
        "nsl_tensor_to_desc_ffi",
        "nsl_tensor_free",
        "nsl_get_num_exports",
        "nsl_get_export_name",
        "nsl_dispatch_apply_result",
        "nsl_dl_path_for_fn_addr",
        "nsl_free_cstr",
    ];
    // M62b Spec C — when nsl-cli is built with --features onnx-rt-op the nsl-runtime
    // crate compiles in `RegisterCustomOps` (the ORT custom-op registration entry
    // point). MSVC requires the symbol to be in the explicit export list to survive
    // linking into the .dll; on other platforms the extra entry is harmless.
    #[cfg(feature = "onnx-rt-op")]
    runtime_exports.push("RegisterCustomOps");
    let mut export_refs: Vec<&str> = export_symbols.iter().map(|s| s.as_str()).collect();
    export_refs.extend(dispatch_symbols.iter().map(|s| s.as_str()));
    export_refs.extend_from_slice(&runtime_exports);

    match nsl_codegen::linker::link_shared_with_exports(
        std::slice::from_ref(&obj_path),
        &lib_path,
        &export_refs,
    ) {
        Ok(()) => {
            let _ = std::fs::remove_file(&obj_path);
            println!("Built shared library {}", lib_path.display());
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }

    emit_c_header_if_any(&exports_slot, &lib_path);
}

/// M62: Write a matching C header next to the shared library when the
/// compile published one or more `@export` functions. No-op otherwise.
fn emit_c_header_if_any(
    exports_slot: &std::sync::Arc<
        std::sync::Mutex<Option<Vec<nsl_codegen::c_header::ExportInfo>>>,
    >,
    lib_path: &std::path::Path,
) {
    let exports = match exports_slot.lock() {
        Ok(guard) => match guard.as_ref() {
            Some(v) if !v.is_empty() => v.clone(),
            _ => return,
        },
        Err(_) => return,
    };
    let module_name = lib_path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    let header = nsl_codegen::c_header::emit(&exports, module_name);
    let header_path = lib_path.with_extension("h");
    match std::fs::write(&header_path, header) {
        Ok(()) => println!("Wrote C header {}", header_path.display()),
        Err(e) => eprintln!("warning: failed to write header '{}': {e}", header_path.display()),
    }
}

/// M62a: Multi-file shared library build.
fn run_build_shared_multi(
    file: &std::path::Path,
    output: Option<PathBuf>,
    dump_ir: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    let mut entry_wrga_plan: Option<nsl_codegen::wrga::WrgaPlan> = None;
    let mut source_map = SourceMap::new();
    let mut interner = Interner::new();
    // M62: allocate a slot the entry-module compile publishes @export
    // functions into, so we can emit a C header after linking.
    let exports_slot: std::sync::Arc<
        std::sync::Mutex<Option<Vec<nsl_codegen::c_header::ExportInfo>>>,
    > = std::sync::Arc::new(std::sync::Mutex::new(None));

    let graph = match crate::loader::load_all_modules(file, &mut source_map, &mut interner) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("error: {e}");
            process::exit(1);
        }
    };

    let temp_dir = std::env::temp_dir().join(format!("nsl_shared_{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("error: could not create temp dir: {e}");
        process::exit(1);
    }

    let mut obj_files: Vec<PathBuf> = Vec::new();

    // Compile each module in dependency order (same as run_build_multi but with shared_lib options)
    for path in &graph.dep_order {
        let mod_data = &graph.modules[path];
        let is_entry = *path == graph.entry;

        let obj_bytes = if is_entry {
            let mut imported_fns = Vec::new();
            let mut imported_struct_layouts: HashMap<String, nsl_codegen::context::StructLayout> = HashMap::new();
            let mut imported_model_names = std::collections::HashSet::new();
            let mut imported_enum_variants = HashMap::new();
            let mut imported_enum_defs = HashMap::new();
            let mut imported_model_method_bodies: HashMap<String, HashMap<String, nsl_ast::decl::FnDef>> = HashMap::new();
            let mut imported_model_field_types: HashMap<String, HashMap<String, String>> = HashMap::new();

            for dep_path in &graph.dep_order {
                if dep_path == &graph.entry {
                    continue;
                }
                let dep_data = &graph.modules[dep_path];

                let mut temp_compiler = match nsl_codegen::compiler::Compiler::new(
                    &interner,
                    &dep_data.type_map,
                    &nsl_codegen::CompileOptions::default(),
                ) {
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

                // Inject previously-collected struct layouts from earlier deps
                for (name, layout) in &imported_struct_layouts {
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
                imported_fns.extend(model_sigs);

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

                // Extract model method bodies directly from AST
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

                imported_enum_variants.extend(dep_data.enum_variants.clone());
                imported_enum_defs.extend(dep_data.enum_defs.clone());
            }

            // Task 3 (B.1): forward entry-module decorator configs to codegen
            // and fail fast if --wrga-report is used without --source-ad.
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
            entry_options.export_functions_out = Some(exports_slot.clone());
            // M62: route entry-module weight_index_map so @export model methods
            // can resolve `self.<field>` → weight index on the multi-file path.
            entry_options.weight_index_map = mod_data.weight_index_map.clone();
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
            match nsl_codegen::compile_module(
                &mod_data.ast,
                &interner,
                &mod_data.type_map,
                &mod_data.module_prefix,
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

        let stem = path.file_stem().unwrap_or_default().to_string_lossy();
        let obj_path = temp_dir.join(format!("{stem}_{}.o", obj_files.len()));

        if let Err(e) = std::fs::write(&obj_path, &obj_bytes) {
            eprintln!("error: could not write object file '{}': {e}", obj_path.display());
            process::exit(1);
        }

        obj_files.push(obj_path);
    }

    emit_wrga_report(&entry_wrga_plan, wrga_report, &options.wrga_check);

    let lib_path = if let Some(out) = output {
        out
    } else {
        nsl_codegen::linker::default_shared_lib_path(file)
    };

    // M62 Task 6: propagate @export symbols to the linker on MSVC.
    let export_symbols: Vec<String> = exports_slot
        .lock()
        .ok()
        .and_then(|g| g.as_ref().map(|v| v.iter().map(|e| e.symbol_name.clone()).collect()))
        .unwrap_or_default();
    // Sibling packed-array dispatch wrappers (`<name>__nsl_dispatch`); see
    // single-file path above for rationale.
    let dispatch_symbols: Vec<String> = export_symbols
        .iter()
        .map(|s| format!("{}__nsl_dispatch", s))
        .collect();
    // M62 Task 9: mirror the single-file path and re-export runtime lifecycle
    // symbols so ctypes callers can load weights + call exports through the
    // generated DLL without loading a separate runtime DLL.
    // `mut` is only used when --features onnx-rt-op is on; harmless otherwise.
    #[allow(unused_mut)]
    let mut runtime_exports: Vec<&'static str> = vec![
        "nsl_model_create",
        "nsl_model_create_with_lib",
        "nsl_model_destroy",
        "nsl_model_forward",
        "nsl_model_forward_grad",
        "nsl_model_backward",
        "nsl_grad_context_destroy",
        "nsl_model_call",
        "nsl_model_call_dlpack",
        "nsl_model_export_count",
        "nsl_model_lookup_function",
        "nsl_model_get_weight_ptrs",
        "nsl_model_get_num_weights",
        "nsl_model_num_weights",
        "nsl_get_last_error",
        "nsl_clear_error",
        "nsl_set_error_cstr",
        "nsl_desc_to_tensor",
        "nsl_tensor_to_desc_ffi",
        "nsl_tensor_free",
        "nsl_get_num_exports",
        "nsl_get_export_name",
        "nsl_dispatch_apply_result",
        "nsl_dl_path_for_fn_addr",
        "nsl_free_cstr",
    ];
    // M62b Spec C — mirror single-file path: surface `RegisterCustomOps` for the
    // MSVC linker when nsl-cli is built with --features onnx-rt-op.
    #[cfg(feature = "onnx-rt-op")]
    runtime_exports.push("RegisterCustomOps");
    let mut export_refs: Vec<&str> = export_symbols.iter().map(|s| s.as_str()).collect();
    export_refs.extend(dispatch_symbols.iter().map(|s| s.as_str()));
    export_refs.extend_from_slice(&runtime_exports);

    match nsl_codegen::linker::link_shared_with_exports(&obj_files, &lib_path, &export_refs) {
        Ok(()) => {
            for obj in &obj_files {
                let _ = std::fs::remove_file(obj);
            }
            println!("Built shared library {}", lib_path.display());
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }

    emit_c_header_if_any(&exports_slot, &lib_path);
}
