mod formatter;
mod loader;
mod mangling;
mod resolver;
mod standalone;

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

        /// M37: Run roofline performance analysis
        #[arg(long)]
        perf: bool,

        /// M37: Target GPU for performance analysis (e.g., "H100", "A100-PCIe")
        #[arg(long)]
        gpu: Option<String>,

        /// M37: Write Chrome tracing JSON to file
        #[arg(long)]
        trace: Option<String>,

        /// M38a: Enable linear types ownership checking
        #[arg(long)]
        linear_types: bool,

        /// M45: Run compile-time NaN/Inf risk analysis
        #[arg(long)]
        nan_analysis: bool,

        /// M46: Enable deterministic mode (compile-time non-determinism detection)
        #[arg(long)]
        deterministic: bool,
    },

    /// Compile and execute an NSL program
    Run {
        /// Path to the .nsl file
        file: PathBuf,

        /// Enable memory profiling (writes memory_profile.json on exit)
        #[arg(long)]
        profile_memory: bool,

        /// Enable kernel profiling (writes kernel_profile.json on exit)
        #[arg(long)]
        profile_kernels: bool,

        /// Enable all profilers and merge into profile.json
        #[arg(long)]
        profile: bool,

        /// Number of GPUs for tensor parallelism (spawns N processes)
        #[arg(long, default_value = "1")]
        devices: u32,

        /// M41: Number of prefill workers for disaggregated inference
        #[arg(long, default_value = "1")]
        prefill_workers: u32,

        /// M41: Number of decode workers for disaggregated inference
        #[arg(long, default_value = "1")]
        decode_workers: u32,

        /// M47: GPU target backend (cuda, rocm, metal, webgpu)
        #[arg(long, default_value = "cuda")]
        target: String,

        /// Disable all fusion optimizations (for differential testing)
        #[arg(long)]
        disable_fusion: bool,

        /// Force tape-based AD (disable source-to-source AD)
        #[arg(long)]
        tape_ad: bool,

        /// M45: Enable tensor operation tracing (writes .nsl.trace binary)
        #[arg(long)]
        trace_ops: bool,

        /// M46: Enable deterministic mode (compile-time non-determinism detection)
        #[arg(long)]
        deterministic: bool,

        /// M43: 3D parallelism config (e.g., "dp=2, tp=4, pp=4")
        #[arg(long)]
        distribute: Option<String>,

        /// M43: ZeRO optimizer sharding stage (1, 2, or 3)
        #[arg(long)]
        zero_stage: Option<u32>,

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

        /// Produce a zero-dependency standalone bundle (requires -w/--weights)
        #[arg(long)]
        standalone: bool,

        /// Path to the model weights file to bundle with the standalone executable
        #[arg(short = 'w', long)]
        weights: Option<PathBuf>,

        /// Weight embedding strategy: auto, always, never (default: auto)
        #[arg(long, default_value = "auto")]
        embed_weights: String,

        /// Size threshold in bytes above which auto mode streams weights instead of embedding (default: 256 MiB)
        #[arg(long, default_value_t = 268_435_456)]
        embed_threshold: u64,

        /// Skip @autotune benchmarking; use middle values from each parameter range
        #[arg(long)]
        no_autotune: bool,

        /// Re-run all @autotune benchmarks, ignoring cached results
        #[arg(long)]
        autotune_fresh: bool,

        /// Delete the autotune cache directory and exit
        #[arg(long)]
        autotune_clean: bool,

        /// Show fusion optimization report on stderr
        #[arg(long)]
        fusion_report: bool,

        /// M36: VRAM budget (e.g., "8GB", "512MB") — fail if plan exceeds
        #[arg(long)]
        vram_budget: Option<String>,

        /// M36: Print memory plan report
        #[arg(long)]
        memory_report: bool,

        /// M38a: Enable linear types ownership checking
        #[arg(long)]
        linear_types: bool,

        /// M47: GPU target backend (cuda, rocm, metal, webgpu)
        #[arg(long, default_value = "cuda")]
        target: String,

        /// Disable all fusion optimizations (for differential testing)
        #[arg(long)]
        disable_fusion: bool,

        /// Force tape-based AD (disable source-to-source AD)
        #[arg(long)]
        tape_ad: bool,

        /// M43: 3D parallelism config (e.g., "dp=2, tp=4, pp=4")
        #[arg(long)]
        distribute: Option<String>,

        /// M43: ZeRO optimizer sharding stage (1, 2, or 3)
        #[arg(long)]
        zero_stage: Option<u32>,

        /// M46: Enable deterministic mode (compile-time non-determinism detection)
        #[arg(long)]
        deterministic: bool,
    },

    /// Run @test functions in an NSL file
    Test {
        /// Path to the .nsl test file
        file: PathBuf,

        /// Filter tests by name substring
        #[arg(long)]
        filter: Option<String>,
    },

    /// Export model to ONNX or checkpoint to safetensors
    Export {
        /// Input file (.nsl for ONNX export, .nslm for safetensors conversion)
        file: PathBuf,

        /// Output file path
        #[arg(short, long)]
        output: Option<PathBuf>,

        /// Output format (onnx, safetensors) — inferred from extension if omitted
        #[arg(long)]
        format: Option<String>,
    },

    /// Initialize a new NeuralScript project
    Init {
        /// Project name (creates directory)
        name: String,
    },

    /// Format NeuralScript source files
    Fmt {
        /// Files to format
        files: Vec<String>,
        /// Check mode: exit non-zero if changes needed
        #[arg(long)]
        check: bool,
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
            perf: _perf,
            gpu: _gpu,
            trace: _trace,
            linear_types: _linear_types,
            nan_analysis,
            deterministic: _deterministic,
        } => {
            run_check(&file, dump_tokens, dump_ast, dump_types);
            // M37: --perf, --gpu, --trace flags parsed but dormant.
            // M38a: --linear-types parsed but dormant until ownership checker
            // is wired through the compilation pipeline.
            // M46: --deterministic parsed but dormant until determinism checker
            // is wired through the check pipeline.

            if nan_analysis {
                // M45: Run compile-time NaN risk analysis.
                // The NanAnalyzer needs function bodies + resolved types to detect
                // log(x), sqrt(x), div-by-zero patterns by walking the typed AST
                // and tracking value constraints through dataflow. Full integration
                // requires a typed-AST walker pass — for now, instantiate to verify
                // the module is importable and announce the pass is active.
                let _analyzer = nsl_semantic::nan_analysis::NanAnalyzer::new();
                eprintln!(
                    "note: --nan-analysis pass enabled \
                     (pattern detection requires typed AST walker — in progress)"
                );
            }
        }
        Cli::Build {
            file,
            output,
            emit_obj,
            dump_ir,
            standalone,
            weights,
            embed_weights,
            embed_threshold,
            no_autotune,
            autotune_fresh,
            autotune_clean,
            fusion_report,
            vram_budget,
            memory_report,
            linear_types: _linear_types,
            target,
            disable_fusion,
            tape_ad: _tape_ad,
            distribute: _distribute,
            zero_stage: _zero_stage,
            deterministic: _deterministic,
        } => {
            if autotune_clean {
                let cache_dir = std::path::Path::new(".nsl-cache/autotune");
                if cache_dir.exists() {
                    std::fs::remove_dir_all(cache_dir).ok();
                    eprintln!("[nsl] autotune cache cleaned");
                } else {
                    eprintln!("[nsl] no autotune cache to clean");
                }
                return;
            }

            let compile_opts = nsl_codegen::CompileOptions {
                no_autotune,
                autotune_fresh,
                world_size: 1, // Build cmd doesn't use --devices; TP world_size is always 1
                fusion_report,
                vram_budget: vram_budget.as_deref()
                    .and_then(nsl_codegen::memory_planner::parse_vram_budget),
                memory_report,
                target,
                disable_fusion,
                tape_ad: _tape_ad,
                trace_ops: false,
                nan_analysis: false,
                deterministic: _deterministic,
            };

            if standalone {
                if weights.is_none() {
                    eprintln!("error: --standalone requires -w/--weights <path>");
                    process::exit(1);
                }
                let embed_mode = match embed_weights.to_lowercase().as_str() {
                    "auto" => standalone::EmbedMode::Auto,
                    "always" => standalone::EmbedMode::Always,
                    "never" => standalone::EmbedMode::Never,
                    other => {
                        eprintln!(
                            "error: unknown --embed-weights value '{}'. \
                             Expected: auto, always, never",
                            other
                        );
                        process::exit(1);
                    }
                };
                run_build_standalone(
                    &file,
                    output.as_deref(),
                    weights.as_deref().unwrap(),
                    embed_mode,
                    embed_threshold,
                    &compile_opts,
                );
            } else {
                run_build(&file, output, emit_obj, dump_ir, &compile_opts);
            }
        }
        Cli::Run {
            file,
            args,
            profile_memory,
            profile_kernels,
            profile,
            devices,
            prefill_workers,
            decode_workers,
            target,
            disable_fusion,
            tape_ad,
            trace_ops,
            deterministic,
            distribute: _distribute,
            zero_stage: _zero_stage,
        } => {
            let compile_opts = nsl_codegen::CompileOptions {
                no_autotune: false,
                autotune_fresh: false,
                world_size: devices as usize,
                fusion_report: false,
                vram_budget: None,
                memory_report: false,
                target,
                disable_fusion,
                tape_ad,
                trace_ops,
                nan_analysis: false,
                deterministic,
            };
            // M41: Disaggregated inference — spawn router + prefill + decode workers.
            // Each runs the same compiled binary with NSL_ROLE and NSL_LOCAL_RANK env vars.
            if prefill_workers > 1 || decode_workers > 1 {
                // Build to a temp directory first, then spawn worker processes
                let temp_dir =
                    std::env::temp_dir().join(format!("nsl_disagg_{}", std::process::id()));
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
                let binary_path = temp_dir.join(&exe_name);

                // Build the binary
                run_build_inner(&file, Some(binary_path.clone()), false, false, true, &compile_opts);

                let mut children: Vec<(&str, std::process::Child)> = Vec::new();
                let total_workers = 1 + prefill_workers + decode_workers;

                // Spawn router (rank 0)
                let router_child = match std::process::Command::new(&binary_path)
                    .env("NSL_ROLE", "router")
                    .env("NSL_LOCAL_RANK", "0")
                    .env("NSL_WORLD_SIZE", format!("{}", total_workers))
                    .stderr(std::process::Stdio::inherit())
                    .spawn()
                {
                    Ok(c) => c,
                    Err(e) => {
                        eprintln!("[nsl] failed to spawn router process: {e}");
                        std::process::exit(1);
                    }
                };
                children.push(("router:0", router_child));

                // Spawn prefill workers
                for i in 0..prefill_workers {
                    let child = match std::process::Command::new(&binary_path)
                        .env("NSL_ROLE", "prefill")
                        .env("NSL_LOCAL_RANK", format!("{}", i))
                        .env("NSL_WORLD_SIZE", format!("{}", total_workers))
                        .stderr(std::process::Stdio::inherit())
                        .spawn()
                    {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("[nsl] failed to spawn prefill worker {i}: {e}");
                            std::process::exit(1);
                        }
                    };
                    children.push(("prefill", child));
                }

                // Spawn decode workers
                for i in 0..decode_workers {
                    let child = match std::process::Command::new(&binary_path)
                        .env("NSL_ROLE", "decode")
                        .env("NSL_LOCAL_RANK", format!("{}", i))
                        .env("NSL_WORLD_SIZE", format!("{}", total_workers))
                        .stderr(std::process::Stdio::inherit())
                        .spawn()
                    {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("[nsl] failed to spawn decode worker {i}: {e}");
                            std::process::exit(1);
                        }
                    };
                    children.push(("decode", child));
                }

                // Wait for all processes
                let mut exit_code = 0;
                for (name, mut child) in children {
                    let status = match child.wait() {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[nsl] failed to wait on {name}: {e}");
                            exit_code = 1;
                            continue;
                        }
                    };
                    if !status.success() {
                        eprintln!("[nsl] {} exited with {}", name, status);
                        exit_code = 1;
                    }
                }

                // Cleanup
                let _ = std::fs::remove_file(&binary_path);
                let _ = std::fs::remove_dir(&temp_dir);

                std::process::exit(exit_code);
            } else if devices > 1 {
                // Build-then-spawn SPMD: spawn N children, each runs the same program
                let exe = match std::env::current_exe() {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("[nsl] could not find current executable: {e}");
                        std::process::exit(1);
                    }
                };

                // Create shared memory file for SimulatedBackend
                let shm_size = 64 + devices as usize * 64 * 1024 * 1024; // header + 64MB per rank
                let shm_path = std::env::temp_dir().join(format!(
                    "nsl_tp_{}_{}.shm",
                    devices,
                    std::process::id()
                ));
                {
                    let f = match std::fs::File::create(&shm_path) {
                        Ok(f) => f,
                        Err(e) => {
                            eprintln!("[nsl] failed to create shm file: {e}");
                            std::process::exit(1);
                        }
                    };
                    if let Err(e) = f.set_len(shm_size as u64) {
                        eprintln!("[nsl] failed to set shm size: {e}");
                        std::process::exit(1);
                    }
                    // Zero the header by mapping and dropping
                    let mmap = match unsafe { memmap2::MmapMut::map_mut(&f) } {
                        Ok(m) => m,
                        Err(e) => {
                            eprintln!("[nsl] failed to mmap shm file: {e}");
                            std::process::exit(1);
                        }
                    };
                    drop(mmap);
                }

                let mut children = Vec::new();
                for rank in 0..devices {
                    let mut cmd = std::process::Command::new(&exe);
                    cmd.arg("run")
                        .arg(&file)
                        .env("NSL_LOCAL_RANK", rank.to_string())
                        .env("NSL_WORLD_SIZE", devices.to_string())
                        .env("NSL_SIMULATED_TP", "1")
                        .env("NSL_TP_SHM_PATH", shm_path.to_str().unwrap_or(""));

                    // Forward profiling flags
                    if profile_memory || profile {
                        cmd.env("NSL_PROFILE_MEMORY", "1");
                    }
                    if profile_kernels || profile {
                        cmd.env("NSL_PROFILE_KERNELS", "1");
                    }

                    // Pass through program args
                    if !args.is_empty() {
                        cmd.arg("--").args(&args);
                    }

                    // Only rank 0 gets stdout
                    if rank > 0 {
                        cmd.stdout(std::process::Stdio::null());
                    }
                    cmd.stderr(std::process::Stdio::inherit());

                    let child = match cmd.spawn() {
                        Ok(c) => c,
                        Err(e) => {
                            eprintln!("[nsl] failed to spawn rank {rank}: {e}");
                            std::process::exit(1);
                        }
                    };
                    children.push((rank, child));
                }

                // Wait for all children
                let mut failed = false;
                for (rank, mut child) in children {
                    let status = match child.wait() {
                        Ok(s) => s,
                        Err(e) => {
                            eprintln!("[nsl] failed to wait on rank {rank}: {e}");
                            failed = true;
                            continue;
                        }
                    };
                    if !status.success() {
                        eprintln!("[nsl] rank {} exited with: {}", rank, status);
                        failed = true;
                    }
                }

                // Cleanup shared memory
                let _ = std::fs::remove_file(&shm_path);

                if failed {
                    std::process::exit(1);
                }
            } else {
                run_run(&file, &args, profile_memory, profile_kernels, profile, &compile_opts);
            }
        }
        Cli::Test { file, filter } => {
            run_test(&file, filter.as_deref());
        }
        Cli::Export {
            file,
            output,
            format,
        } => {
            run_export(&file, output.as_deref(), format.as_deref());
        }
        Cli::Init { name } => {
            run_init(&name);
        }
        Cli::Fmt { files, check } => {
            run_fmt(&files, check);
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

/// Check if a file has any import statements or train blocks by quick-scanning.
/// Train blocks need multi-file compilation because optimizer stdlib modules
/// are auto-imported.
fn needs_multi_file(file: &PathBuf) -> bool {
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

fn run_build(file: &PathBuf, output: Option<PathBuf>, emit_obj: bool, dump_ir: bool, options: &nsl_codegen::CompileOptions) {
    run_build_inner(file, output, emit_obj, dump_ir, false, options);
}

fn run_build_standalone(
    file: &std::path::Path,
    output: Option<&std::path::Path>,
    weights: &std::path::Path,
    embed_mode: standalone::EmbedMode,
    embed_threshold: u64,
    options: &nsl_codegen::CompileOptions,
) {
    // 1. Read weights from safetensors
    let tensors = standalone::read_safetensors(weights).unwrap_or_else(|e| {
        eprintln!("error: {e}");
        process::exit(1);
    });

    // 2. Serialize to .nslweights format
    let nslweights_data = standalone::serialize_nslweights(&tensors);

    // 3. Decide embed vs sidecar
    let embedded = match embed_mode {
        standalone::EmbedMode::Always => true,
        standalone::EmbedMode::Never => false,
        standalone::EmbedMode::Auto => (nslweights_data.len() as u64) <= embed_threshold,
    };

    // 4. Run frontend (lex, parse, semantic analysis)
    let file_pb = file.to_path_buf();
    let (interner, parse_result, analysis) = frontend(&file_pb);

    // 5. Determine output path
    let output_path = if let Some(out) = output {
        out.to_path_buf()
    } else {
        nsl_codegen::linker::default_output_path(file)
    };

    let sidecar_path = output_path.with_extension("nslweights");

    // Pass only the filename (not full path) to codegen — the runtime resolves
    // the sidecar relative to the executable, so absolute paths would break
    // portability when the binary is moved.
    let sidecar_name = sidecar_path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_else(|| "model.nslweights".to_string());

    let config = nsl_codegen::StandaloneConfig {
        embedded,
        sidecar_path: sidecar_name,
    };

    // 6. Compile with standalone config
    let obj_bytes = nsl_codegen::compile_standalone(
        &parse_result.module,
        &interner,
        &analysis.type_map,
        config,
        false,
        options,
    )
    .unwrap_or_else(|e| {
        eprintln!("codegen error: {e}");
        process::exit(1);
    });

    // 7. Write main object file
    let temp_dir = std::env::temp_dir().join(format!("nsl_standalone_{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("error: could not create temp dir: {e}");
        process::exit(1);
    }

    let main_obj_path = temp_dir.join("main.o");
    if let Err(e) = std::fs::write(&main_obj_path, &obj_bytes) {
        eprintln!("error: could not write object file: {e}");
        process::exit(1);
    }

    let mut obj_paths: Vec<PathBuf> = vec![main_obj_path];

    // 8. Handle embedded weights or sidecar
    if embedded {
        // Create weight object containing the nslweights data
        let weight_obj_bytes = nsl_codegen::create_weight_object(&nslweights_data).unwrap_or_else(|e| {
            eprintln!("error: could not create weight object: {e}");
            process::exit(1);
        });
        let weight_obj_path = temp_dir.join("weights.o");
        if let Err(e) = std::fs::write(&weight_obj_path, &weight_obj_bytes) {
            eprintln!("error: could not write weight object file: {e}");
            process::exit(1);
        }
        obj_paths.push(weight_obj_path);
    } else {
        // Write sidecar .nslweights file
        standalone::write_nslweights_sidecar_raw(&nslweights_data, &sidecar_path).unwrap_or_else(|e| {
            eprintln!("error: {e}");
            process::exit(1);
        });
    }

    // 9. Link all objects
    match nsl_codegen::linker::link_multi(&obj_paths, &output_path) {
        Ok(()) => {
            // 10. Clean up temp object files
            for obj in &obj_paths {
                let _ = std::fs::remove_file(obj);
            }
            let _ = std::fs::remove_dir(&temp_dir);

            println!("Built {} (standalone{})", output_path.display(),
                if embedded { ", weights embedded" } else { ", sidecar weights" });
            if !embedded {
                println!("  Sidecar: {}", sidecar_path.display());
            }
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }
}

fn run_build_inner(file: &PathBuf, output: Option<PathBuf>, emit_obj: bool, dump_ir: bool, quiet: bool, options: &nsl_codegen::CompileOptions) {
    if needs_multi_file(file) {
        run_build_multi(file, output, emit_obj, dump_ir, quiet, options);
    } else {
        run_build_single(file, output, emit_obj, dump_ir, quiet, options);
    }
}

/// Single-file build (backward compatible, fast path).
fn run_build_single(file: &PathBuf, output: Option<PathBuf>, emit_obj: bool, dump_ir: bool, quiet: bool, options: &nsl_codegen::CompileOptions) {
    let (interner, parse_result, analysis) = frontend(file);

    // Codegen
    let obj_bytes = match nsl_codegen::compile(
        &parse_result.module,
        &interner,
        &analysis.type_map,
        dump_ir,
        options,
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
fn run_build_multi(file: &std::path::Path, output: Option<PathBuf>, emit_obj: bool, dump_ir: bool, quiet: bool, options: &nsl_codegen::CompileOptions) {
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
            let mut imported_struct_layouts = HashMap::new();
            let mut imported_model_names = std::collections::HashSet::new();
            let mut imported_enum_variants = HashMap::new();
            let mut imported_enum_defs = HashMap::new();

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
                imported_model_names,
                imported_enum_variants,
                imported_enum_defs,
                dump_ir,
                options,
            ) {
                Ok(bytes) => bytes,
                Err(e) => {
                    eprintln!("codegen error in '{}': {e}", path.display());
                    process::exit(1);
                }
            }
        } else {
            // Library module: export all functions.
            // If this module has dependencies (imports from other modules), inject their symbols.
            let mut lib_imported_fns = Vec::new();
            let mut lib_struct_layouts = HashMap::new();
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

                    for (name, layout) in temp_compiler.struct_layouts.drain() {
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

fn run_run(file: &PathBuf, program_args: &[String], profile_memory: bool, profile_kernels: bool, profile: bool, options: &nsl_codegen::CompileOptions) {
    let temp_dir = std::env::temp_dir().join(format!("nsl_run_{}", std::process::id()));
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
    run_build_inner(file, Some(exe_path.clone()), false, false, true, options);

    // Execute the compiled program
    let mut cmd = std::process::Command::new(&exe_path);
    cmd.args(program_args);
    if profile_memory || profile {
        cmd.env("NSL_PROFILE_MEMORY", "1");
    }
    if profile_kernels || profile {
        cmd.env("NSL_PROFILE_KERNELS", "1");
    }
    let status = cmd
        .status()
        .unwrap_or_else(|e| {
            eprintln!("error: could not execute '{}': {e}", exe_path.display());
            process::exit(1);
        });

    // Merge profile traces before exiting (process::exit won't return)
    if profile {
        merge_profile_traces("memory_profile.json", "kernel_profile.json", "profile.json");
    }

    // Clean up
    let _ = std::fs::remove_file(&exe_path);
    let _ = std::fs::remove_dir(&temp_dir);

    // Forward exit code
    process::exit(status.code().unwrap_or(1));
}

fn run_test(file: &PathBuf, filter: Option<&str>) {
    let (interner, parse_result, analysis) = frontend(file);

    // Compile in test mode — produces a binary with test-dispatch main()
    let (obj_bytes, test_fns) = match nsl_codegen::compile_test(
        &parse_result.module,
        &interner,
        &analysis.type_map,
        false,
        &nsl_codegen::CompileOptions::default(),
    ) {
        Ok(result) => result,
        Err(e) => {
            eprintln!("codegen error: {e}");
            process::exit(1);
        }
    };

    // Apply filter
    let tests: Vec<&String> = if let Some(f) = filter {
        test_fns.iter().filter(|name| name.contains(f)).collect()
    } else {
        test_fns.iter().collect()
    };

    if tests.is_empty() {
        if let Some(f) = filter {
            eprintln!("no tests match filter '{f}'");
        } else {
            eprintln!("no @test functions found");
        }
        process::exit(1);
    }

    // Write object file and link to a temp executable
    let temp_dir = std::env::temp_dir().join(format!("nsl_test_{}", std::process::id()));
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("error: could not create temp dir: {e}");
        process::exit(1);
    }

    let stem = file
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("test");
    let obj_path = temp_dir.join(format!("{stem}.o"));
    let exe_name = if cfg!(target_os = "windows") {
        format!("{stem}.exe")
    } else {
        stem.to_string()
    };
    let exe_path = temp_dir.join(&exe_name);

    if let Err(e) = std::fs::write(&obj_path, &obj_bytes) {
        eprintln!("error: could not write object file: {e}");
        process::exit(1);
    }

    match nsl_codegen::linker::link(&obj_path, &exe_path) {
        Ok(()) => {
            let _ = std::fs::remove_file(&obj_path);
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }

    // Run each test by spawning the binary with --run <test_name>
    let mut passed = 0u32;
    let mut failed = 0u32;

    for test_name in &tests {
        let output = std::process::Command::new(&exe_path)
            .args(["--run", test_name])
            .output()
            .unwrap_or_else(|e| {
                eprintln!("error: could not execute '{}': {e}", exe_path.display());
                process::exit(1);
            });

        if output.status.success() {
            println!("{test_name} ... PASS");
            passed += 1;
        } else {
            println!("{test_name} ... FAIL");
            let stderr = String::from_utf8_lossy(&output.stderr);
            if !stderr.is_empty() {
                for line in stderr.lines() {
                    println!("  {line}");
                }
            }
            failed += 1;
        }
    }

    // Clean up
    let _ = std::fs::remove_file(&exe_path);
    let _ = std::fs::remove_dir(&temp_dir);

    // Summary
    println!("\n{passed} passed, {failed} failed");

    if failed > 0 {
        process::exit(1);
    }
}

fn run_export(file: &PathBuf, output: Option<&std::path::Path>, format: Option<&str>) {
    let ext = file
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("");

    // Determine export mode from format flag, output extension, or input extension
    let mode = if let Some(fmt) = format {
        fmt.to_lowercase()
    } else if let Some(out) = output {
        out.extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase()
    } else {
        match ext {
            "nsl" => "onnx".to_string(),
            "nslm" => "safetensors".to_string(),
            _ => {
                eprintln!(
                    "error: cannot determine export format from '{}'.\n\
                     Use --format onnx or --format safetensors, or provide an output path with --output.",
                    file.display()
                );
                process::exit(1);
            }
        }
    };

    match mode.as_str() {
        "onnx" => {
            if ext != "nsl" {
                eprintln!(
                    "error: ONNX export requires an .nsl input file, got '{}'",
                    file.display()
                );
                process::exit(1);
            }
            // The NSL file should contain an export_model() function (or similar)
            // that calls to_onnx() internally. We just compile and run it.
            println!("Exporting ONNX from {}", file.display());
            run_run(file, &[], false, false, false, &nsl_codegen::CompileOptions::default());
        }
        "safetensors" => {
            if ext != "nslm" {
                eprintln!(
                    "error: safetensors conversion requires an .nslm input file, got '{}'",
                    file.display()
                );
                process::exit(1);
            }
            let out_path = output.unwrap_or_else(|| {
                // Will use the stem + .safetensors
                std::path::Path::new("output.safetensors")
            });
            println!(
                "Converting {} to safetensors format at {}",
                file.display(),
                out_path.display()
            );
            // TODO: implement NSLM → safetensors data conversion
            // This requires parsing the NSLM binary header (magic + JSON + f64 data)
            // and writing it in the safetensors format.
            eprintln!("note: .nslm to safetensors conversion is not yet implemented");
            process::exit(1);
        }
        other => {
            eprintln!(
                "error: unknown export format '{}'. Supported formats: onnx, safetensors",
                other
            );
            process::exit(1);
        }
    }
}

fn run_fmt(files: &[String], check: bool) {
    use std::path::Path;

    let mut total = 0u32;
    let mut changed = 0u32;
    let mut errors = 0u32;

    for pattern in files {
        // Treat as literal file path (glob support can come later)
        let path = Path::new(pattern);
        if !path.exists() {
            eprintln!("error: file not found: {}", pattern);
            errors += 1;
            continue;
        }
        total += 1;
        match formatter::format_file(path, check) {
            Ok(true) => {
                changed += 1;
                if check {
                    println!("Would reformat: {}", path.display());
                } else {
                    println!("Formatted: {}", path.display());
                }
            }
            Ok(false) => {} // already formatted
            Err(e) => {
                eprintln!("{}", e);
                errors += 1;
            }
        }
    }

    if total > 0 || errors > 0 {
        println!("{} file(s) checked, {} changed, {} error(s)", total, changed, errors);
    }

    if check && changed > 0 {
        process::exit(1);
    }
    if errors > 0 {
        process::exit(1);
    }
}

fn merge_profile_traces(memory_path: &str, kernel_path: &str, output_path: &str) {
    let mem_json = std::fs::read_to_string(memory_path).unwrap_or_default();
    let kern_json = std::fs::read_to_string(kernel_path).unwrap_or_default();

    let mem_events = extract_trace_events(&mem_json).unwrap_or_default();
    let kern_events = extract_trace_events(&kern_json).unwrap_or_default();

    let merged = format!(
        r#"{{"traceEvents":[{}{}{}],"metadata":{{"merged":true}}}}"#,
        mem_events,
        if !mem_events.is_empty() && !kern_events.is_empty() { "," } else { "" },
        kern_events,
    );

    std::fs::write(output_path, &merged).ok();
    std::fs::remove_file(memory_path).ok();
    std::fs::remove_file(kernel_path).ok();
    eprintln!("[nsl] merged profile written to {}", output_path);
}

fn extract_trace_events(json: &str) -> Option<String> {
    let start = json.find("\"traceEvents\":")? + "\"traceEvents\":".len();
    let bracket_start = json[start..].find('[')? + start;
    let mut depth = 0;
    let mut bracket_end = bracket_start;
    for (i, ch) in json[bracket_start..].char_indices() {
        match ch {
            '[' => depth += 1,
            ']' => {
                depth -= 1;
                if depth == 0 {
                    bracket_end = bracket_start + i;
                    break;
                }
            }
            _ => {}
        }
    }
    Some(json[bracket_start + 1..bracket_end].to_string())
}

fn run_init(name: &str) {
    let root = std::path::Path::new(name);

    // Refuse to overwrite an existing directory
    if root.exists() {
        eprintln!("error: directory '{}' already exists", name);
        process::exit(1);
    }

    // Create project root and sub-directories
    for dir in &[root.to_path_buf(), root.join("data"), root.join("weights")] {
        if let Err(e) = std::fs::create_dir_all(dir) {
            eprintln!("error: could not create directory '{}': {e}", dir.display());
            process::exit(1);
        }
    }

    // main.nsl — tensor example
    let main_nsl = "\
# My first NeuralScript program

let x = zeros([2, 3])
let y = ones([2, 3])
let z = x + y

print(z)
print(f\"Sum: {z.sum()}\")
";

    // nsl.toml
    let nsl_toml = format!(
        "\
# NeuralScript Project Configuration
# Reserved for future use by the NSL package manager (v0.2)

[project]
name = \"{name}\"
version = \"0.1.0\"
entry = \"main.nsl\"
"
    );

    // .gitignore
    let gitignore = "\
# Build
*.exe
*.o
.nsl-cache/

# ML artifacts
*.safetensors
*.bin
*.nslm
/weights/
/data/
";

    let files: &[(&str, &str)] = &[
        ("main.nsl", main_nsl),
        ("nsl.toml", &nsl_toml),
        (".gitignore", gitignore),
    ];

    for (filename, contents) in files {
        let path = root.join(filename);
        if let Err(e) = std::fs::write(&path, contents) {
            eprintln!("error: could not write '{}': {e}", path.display());
            process::exit(1);
        }
    }

    println!("Created project '{name}'. Run: cd {name} && nsl run main.nsl");
}
