mod debug;
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

        /// M52: Run weight analysis report
        #[arg(long)]
        weight_analysis: bool,

        /// M52: Path to safetensors weight file for weight analysis
        #[arg(long)]
        weights: Option<PathBuf>,

        /// M52: Dead weight threshold for analysis (default: 1e-6)
        #[arg(long, default_value_t = 1e-6)]
        dead_weight_threshold: f64,

        /// M52: Sparsity threshold for analysis (default: 0.5)
        #[arg(long, default_value_t = 0.5)]
        sparse_threshold: f64,

        /// M53: Enable WCET analysis for @real_time functions
        #[arg(long)]
        wcet: bool,

        /// M53: Write WCET certificate JSON to file
        #[arg(long)]
        wcet_cert: Option<PathBuf>,

        /// M53: CPU target for WCET analysis (e.g., "cortex-a78")
        #[arg(long)]
        cpu: Option<String>,

        /// M53: Write DO-178C compliance report to file (FPGA only)
        #[arg(long)]
        do178c_report: Option<PathBuf>,

        /// M53: WCET target: "gpu" (statistical advisory), "fpga" (certified DO-178C), "groq" (blocked)
        #[arg(long, default_value = "gpu")]
        wcet_target: String,

        /// M53: FPGA device for certified WCET (e.g., "xcvu440", "xczu9eg", "ve2302")
        #[arg(long)]
        fpga_device: Option<String>,
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

        /// Use compile-time source-to-source AD instead of runtime tape AD
        #[arg(long)]
        source_ad: bool,

        /// Debug training: disable fusion + FBIP, emit gradient checksums
        #[arg(long)]
        debug_training: bool,

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

        /// M53: Enable WCET analysis for @real_time functions
        #[arg(long)]
        wcet: bool,

        /// M53: Write WCET certificate JSON to file
        #[arg(long)]
        wcet_cert: Option<PathBuf>,

        /// M53: GPU target for WCET analysis (e.g., "A100-SXM", "Orin")
        #[arg(long)]
        gpu: Option<String>,

        /// M53: CPU target for WCET analysis (e.g., "cortex-a78")
        #[arg(long)]
        cpu: Option<String>,

        /// M53: Write DO-178C compliance report to file (FPGA only)
        #[arg(long)]
        do178c_report: Option<PathBuf>,

        /// M53: WCET target: "gpu" (statistical advisory), "fpga" (certified DO-178C), "groq" (blocked)
        #[arg(long, default_value = "gpu")]
        wcet_target: String,

        /// M53: FPGA device for certified WCET (e.g., "xcvu440", "xczu9eg", "ve2302")
        #[arg(long)]
        fpga_device: Option<String>,

        /// Synchronize after every CUDA kernel launch (debug: surfaces async GPU errors)
        #[arg(long)]
        cuda_sync: bool,

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

        /// Use compile-time source-to-source AD instead of runtime tape AD
        #[arg(long)]
        source_ad: bool,

        /// Debug training: disable fusion + FBIP, emit gradient checksums
        #[arg(long)]
        debug_training: bool,

        /// M45: Run compile-time NaN/Inf risk analysis before codegen
        #[arg(long)]
        nan_analysis: bool,

        /// M43: 3D parallelism config (e.g., "dp=2, tp=4, pp=4")
        #[arg(long)]
        distribute: Option<String>,

        /// M43: ZeRO optimizer sharding stage (1, 2, or 3)
        #[arg(long)]
        zero_stage: Option<u32>,

        /// M46: Enable deterministic mode (compile-time non-determinism detection)
        #[arg(long)]
        deterministic: bool,

        /// M52: Dead weight threshold (default: 1e-6). Weights with |value| < threshold are zeroed.
        #[arg(long, default_value_t = 1e-6)]
        dead_weight_threshold: f64,

        /// M52: Sparsity threshold (default: 0.5). Matrices with near-zero fraction >= threshold get sparse annotation.
        #[arg(long, default_value_t = 0.5)]
        sparse_threshold: f64,

        /// M52: Disable constant folding of weight expressions
        #[arg(long)]
        no_constant_fold: bool,

        /// M52: Disable dead weight elimination
        #[arg(long)]
        no_dead_weight: bool,

        /// M52: Disable sparsity-aware codegen annotations
        #[arg(long)]
        no_sparse_codegen: bool,

        /// M62: Build as shared library (.so/.dylib/.dll) with stable C API
        #[arg(long)]
        shared_lib: bool,

        /// M54: Build as a bare-metal unikernel image
        #[arg(long)]
        unikernel: bool,

        /// M54: Unikernel listen address (default: "0.0.0.0:8080")
        #[arg(long, default_value = "0.0.0.0:8080")]
        listen: String,

        /// M54: Unikernel total memory (e.g., "16G", "512M"). 0 or omitted = auto-detect at boot.
        #[arg(long)]
        memory: Option<String>,

        /// M53: Enable WCET analysis for @real_time functions
        #[arg(long)]
        wcet: bool,

        /// M53: Write WCET certificate JSON to file
        #[arg(long)]
        wcet_cert: Option<PathBuf>,

        /// M53: CPU target for WCET analysis (e.g., "cortex-a78")
        #[arg(long)]
        cpu: Option<String>,

        /// M53: Write DO-178C compliance report to file (FPGA only)
        #[arg(long)]
        do178c_report: Option<PathBuf>,

        /// M53: WCET target: "gpu" (statistical advisory), "fpga" (certified DO-178C), "groq" (blocked)
        #[arg(long, default_value = "gpu")]
        wcet_target: String,

        /// M53: FPGA device for certified WCET (e.g., "xcvu440", "xczu9eg", "ve2302")
        #[arg(long)]
        fpga_device: Option<String>,

        /// M55: Compile @zk_proof functions to ZK inference circuits
        #[arg(long)]
        zk_circuit: bool,

        /// M55: ZK proving backend: folding (default), halo2, or plonky3
        #[arg(long, default_value = "folding")]
        zk_backend: String,

        /// M55: ZK field: m31 (default, ~10x faster) or bn254 (EVM-compatible)
        #[arg(long, default_value = "m31")]
        zk_field: String,

        /// M55: Emit a Solidity verifier contract alongside the ZK circuit
        #[arg(long)]
        zk_solidity: bool,

        /// M55: Path to .safetensors weights file used as ZK witness
        #[arg(long)]
        zk_weights: Option<PathBuf>,
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

    /// Convert between checkpoint formats (.nslm ↔ .safetensors)
    Convert {
        /// Input file (.nslm or .safetensors)
        input: PathBuf,

        /// Output file (.safetensors or .nslm) — extension determines direction
        output: PathBuf,
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

    /// Read and analyze a tensor operation trace (.nsltrace binary)
    Debug {
        /// Path to the .nsltrace binary file
        file: PathBuf,

        /// Find first NaN/Inf operation
        #[arg(long)]
        find_nan: bool,

        /// Compare with another trace file
        #[arg(long)]
        diff: Option<PathBuf>,

        /// Export to Chrome tracing JSON file
        #[arg(long)]
        export_chrome: Option<PathBuf>,
    },

    /// M55: ZK inference circuit operations (stats, prove, verify)
    Zk {
        #[command(subcommand)]
        cmd: ZkCmd,
    },

    /// Train a BPE tokenizer from source files
    Tokenize {
        /// Directories to scan for source files (default: stdlib/ examples/ tests/)
        #[arg(long, num_args = 1..)]
        dirs: Vec<String>,

        /// Output path for the tokenizer JSON file
        #[arg(short, long, default_value = "tokenizer.json")]
        output: PathBuf,

        /// Vocabulary size (default: 49152 — matches coder50m config)
        #[arg(long, default_value_t = 49152)]
        vocab_size: usize,

        /// Minimum token frequency (default: 2)
        #[arg(long, default_value_t = 2)]
        min_freq: u64,

        /// File extensions to include (default: .nsl)
        #[arg(long, default_value = "nsl")]
        ext: String,
    },
}

/// M55: ZK subcommands.
#[derive(clap::Subcommand)]
enum ZkCmd {
    /// Show circuit statistics for a compiled .zkir file
    Stats {
        /// Path to the .zkir file
        file: PathBuf,
    },

    /// Generate a ZK proof from a compiled circuit
    Prove {
        /// Path to the .zkir file
        file: PathBuf,

        /// Path to the proving key file
        #[arg(long)]
        pk: PathBuf,

        /// Path to JSON file containing circuit inputs
        #[arg(long)]
        input: PathBuf,

        /// Output path for the generated proof (default: <file>.proof)
        #[arg(short, long)]
        output: Option<PathBuf>,
    },

    /// Verify a ZK proof against public inputs
    Verify {
        /// Path to the verification key file
        vk: PathBuf,

        /// Path to the proof file to verify
        #[arg(long)]
        proof: PathBuf,

        /// Path to JSON file containing public inputs
        #[arg(long)]
        public: PathBuf,
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
            linear_types,
            nan_analysis,
            deterministic,
            weight_analysis,
            weights,
            dead_weight_threshold,
            sparse_threshold,
            wcet: _wcet,
            wcet_cert: _wcet_cert,
            cpu: _cpu,
            do178c_report: _do178c_report,
            wcet_target: _wcet_target,
            fpga_device: _fpga_device,
        } => {
            run_check(&file, dump_tokens, dump_ast, dump_types, linear_types);
            // M37: --perf, --gpu, --trace flags parsed but dormant.

            if nan_analysis {
                // M45: Run compile-time NaN risk analysis via AST walker.
                // Re-lex/parse the file to get a Module + Interner for the walker.
                let source = std::fs::read_to_string(&file).unwrap_or_default();
                let mut nan_interner = Interner::new();
                let mut nan_source_map = SourceMap::new();
                let nan_file_id = nan_source_map.add_file(file.display().to_string(), source.clone());
                let (tokens, _) = nsl_lexer::tokenize(&source, nan_file_id, &mut nan_interner);
                let parse_result = nsl_parser::parse(&tokens, &mut nan_interner);

                let mut analyzer = nsl_semantic::nan_analysis::NanAnalyzer::new();
                analyzer.analyze_module(&parse_result.module, &nan_interner);

                if analyzer.diagnostics.is_empty() {
                    eprintln!("note: --nan-analysis: no NaN/Inf risks detected");
                } else {
                    eprintln!(
                        "note: --nan-analysis: {} warning(s) detected",
                        analyzer.diagnostics.len()
                    );
                    for diag in &analyzer.diagnostics {
                        nan_source_map.emit_diagnostic(diag);
                    }
                }
            }

            // M46: Determinism analysis — scan for non-deterministic ops
            if deterministic {
                let source = std::fs::read_to_string(&file).unwrap_or_default();
                let mut det_interner = Interner::new();
                let mut det_source_map = SourceMap::new();
                let det_file_id = det_source_map.add_file(file.display().to_string(), source.clone());
                let (tokens, _) = nsl_lexer::tokenize(&source, det_file_id, &mut det_interner);
                let parse_result = nsl_parser::parse(&tokens, &mut det_interner);

                let mut checker = nsl_semantic::determinism::DeterminismChecker::new(
                    nsl_semantic::determinism::DeterminismMode::Global,
                );
                checker.scan_module(&parse_result.module, &det_interner);

                let errors: Vec<_> = checker.diagnostics.iter()
                    .filter(|d| d.level == nsl_errors::Level::Error)
                    .collect();
                let warnings: Vec<_> = checker.diagnostics.iter()
                    .filter(|d| d.level == nsl_errors::Level::Warning)
                    .collect();

                if checker.diagnostics.is_empty() {
                    eprintln!("note: --deterministic: no non-deterministic ops detected");
                } else {
                    eprintln!(
                        "note: --deterministic: {} warning(s), {} error(s)",
                        warnings.len(),
                        errors.len()
                    );
                    for diag in &checker.diagnostics {
                        det_source_map.emit_diagnostic(diag);
                    }
                    if !errors.is_empty() {
                        process::exit(1);
                    }
                }
            }

            // M52: Weight analysis report
            if weight_analysis {
                if let Some(ref weights_path) = weights {
                    let config = nsl_codegen::weight_aware::WeightAwareConfig {
                        dead_weight_threshold,
                        sparse_threshold,
                        ..Default::default()
                    };
                    match nsl_codegen::weight_aware::WeightMap::load(weights_path.as_path()) {
                        Ok(wmap) => {
                            nsl_codegen::weight_aware::print_weight_analysis_report(&wmap, &config);
                        }
                        Err(e) => {
                            eprintln!("error: failed to load weights: {}", e);
                            process::exit(1);
                        }
                    }
                } else {
                    eprintln!("error: --weight-analysis requires --weights <path>");
                    process::exit(1);
                }
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
            linear_types,
            target,
            disable_fusion,
            tape_ad: _tape_ad,
            source_ad: _source_ad,
            debug_training,
            nan_analysis,
            distribute: _distribute,
            zero_stage,
            deterministic: _deterministic,
            dead_weight_threshold,
            sparse_threshold,
            no_constant_fold,
            no_dead_weight,
            no_sparse_codegen,
            shared_lib,
            unikernel,
            listen,
            memory,
            wcet,
            wcet_cert,
            cpu,
            do178c_report,
            wcet_target,
            fpga_device,
            zk_circuit,
            zk_backend,
            zk_field,
            zk_solidity,
            zk_weights,
        } => {
            // M62a: shared_lib flag is threaded through compile_opts and handled
            // in the build path below.

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

            // M54: Parse unikernel configuration if --unikernel is set.
            let unikernel_config = if unikernel {
                let listen_addr = match nsl_codegen::unikernel::parse_listen_addr(&listen) {
                    Ok(addr) => addr,
                    Err(e) => {
                        eprintln!("error: invalid --listen value: {e}");
                        process::exit(1);
                    }
                };
                let memory_bytes = match memory.as_deref() {
                    Some(s) => match nsl_codegen::unikernel::parse_memory_size(s) {
                        Ok(n) => n,
                        Err(e) => {
                            eprintln!("error: invalid --memory value: {e}");
                            process::exit(1);
                        }
                    },
                    None => 0, // auto-detect at boot
                };
                let cfg = nsl_codegen::unikernel::UnikernelConfig {
                    listen_addr,
                    memory_bytes,
                    ..Default::default()
                };
                cfg.print_summary();
                Some(cfg)
            } else {
                None
            };

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
                source_ad: _source_ad,
                trace_ops: false,
                nan_analysis,
                deterministic: _deterministic,
                // M52: When --standalone, weights are handled by standalone pipeline;
                // otherwise pass through for weight-aware compilation
                weight_file: if standalone { None } else { weights.clone() },
                weight_config: nsl_codegen::weight_aware::WeightAwareConfig {
                    dead_weight_threshold,
                    sparse_threshold,
                    constant_fold: !no_constant_fold,
                    dead_weight_elim: !no_dead_weight,
                    sparse_codegen: !no_sparse_codegen,
                },
                weight_analysis: false,
                unikernel_config,
                wcet_enabled: wcet,
                wcet_gpu: None, // reuse --gpu from Check variant; Build uses target for backend
                wcet_cpu: cpu,
                wcet_report_path: wcet_cert,
                wcet_safety_margin: 1.05,
                do178c_report,
                wcet_target,
                fpga_device,
                zk_circuit,
                zk_backend,
                zk_field,
                zk_solidity,
                zk_weights_path: zk_weights.clone(),
                linear_types_enabled: linear_types,
                ownership_info: std::collections::HashMap::new(), // populated by loader
                zero_stage: zero_stage.map(|s| s as u8),
                debug_training,
                shared_lib,
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
            } else if shared_lib {
                run_build_shared(&file, output, dump_ir, &compile_opts);
            } else if zk_circuit {
                run_build_zk(&file, output, emit_obj, dump_ir, zk_weights.as_deref(), &compile_opts);
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
            source_ad,
            debug_training,
            trace_ops,
            deterministic,
            distribute: _distribute,
            zero_stage,
            wcet,
            wcet_cert,
            gpu,
            cpu,
            do178c_report,
            wcet_target,
            fpga_device,
            cuda_sync,
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
                source_ad,
                trace_ops,
                nan_analysis: false,
                deterministic,
                weight_file: None,
                weight_config: Default::default(),
                weight_analysis: false,
                unikernel_config: None,
                wcet_enabled: wcet,
                wcet_gpu: gpu,
                wcet_cpu: cpu,
                wcet_report_path: wcet_cert,
                wcet_safety_margin: 1.05,
                do178c_report,
                wcet_target,
                fpga_device,
                // M55: ZK flags not exposed on `run`; use defaults.
                zk_circuit: false,
                zk_backend: "folding".to_string(),
                zk_field: "m31".to_string(),
                zk_solidity: false,
                zk_weights_path: None,
                linear_types_enabled: false, // run command doesn't expose --linear-types
                ownership_info: std::collections::HashMap::new(),
                zero_stage: zero_stage.map(|s| s as u8),
                debug_training,
                shared_lib: false,
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
                    if cuda_sync {
                        cmd.env("NSL_CUDA_SYNC", "1");
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
                run_run(&file, &args, profile_memory, profile_kernels, profile, cuda_sync, &compile_opts);
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
        Cli::Convert { input, output } => {
            run_convert(&input, &output);
        }
        Cli::Init { name } => {
            run_init(&name);
        }
        Cli::Fmt { files, check } => {
            run_fmt(&files, check);
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
            run_zk_cmd(cmd);
        }
        Cli::Tokenize { dirs, output, vocab_size, min_freq, ext } => {
            run_tokenize(&dirs, &output, vocab_size, min_freq, &ext);
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

fn run_check(file: &PathBuf, dump_tokens: bool, dump_ast: bool, dump_types: bool, linear_types: bool) {
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

/// M62a: Build as a shared library (.so/.dylib/.dll) with stable C API.
fn run_build_shared(
    file: &PathBuf,
    output: Option<PathBuf>,
    dump_ir: bool,
    options: &nsl_codegen::CompileOptions,
) {
    if needs_multi_file(file) {
        run_build_shared_multi(file, output, dump_ir, options);
    } else {
        run_build_shared_single(file, output, dump_ir, options);
    }
}

/// M62a: Single-file shared library build.
fn run_build_shared_single(
    file: &PathBuf,
    output: Option<PathBuf>,
    dump_ir: bool,
    options: &nsl_codegen::CompileOptions,
) {
    let (interner, parse_result, analysis) = frontend(file);

    // Codegen with PIC enabled (shared_lib=true in options)
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

    match nsl_codegen::linker::link_shared(&[obj_path.clone()], &lib_path) {
        Ok(()) => {
            let _ = std::fs::remove_file(&obj_path);
            println!("Built shared library {}", lib_path.display());
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }
}

/// M62a: Multi-file shared library build.
fn run_build_shared_multi(
    file: &std::path::Path,
    output: Option<PathBuf>,
    dump_ir: bool,
    options: &nsl_codegen::CompileOptions,
) {
    let mut source_map = SourceMap::new();
    let mut interner = Interner::new();

    let graph = match loader::load_all_modules(file, &mut source_map, &mut interner) {
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
            let mut imported_struct_layouts = HashMap::new();
            let mut imported_model_names = std::collections::HashSet::new();
            let mut imported_enum_variants = HashMap::new();
            let mut imported_enum_defs = HashMap::new();

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

                for (name, layout) in temp_compiler.struct_layouts.drain() {
                    imported_struct_layouts.insert(name, layout);
                }

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

    let lib_path = if let Some(out) = output {
        out
    } else {
        nsl_codegen::linker::default_shared_lib_path(file)
    };

    match nsl_codegen::linker::link_shared(&obj_files, &lib_path) {
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
}

/// M55: Build with --zk-circuit. Runs normal compilation and then invokes
/// `zk::compile_zk()` for each @zk_proof-decorated function found.
fn run_build_zk(
    file: &PathBuf,
    output: Option<PathBuf>,
    emit_obj: bool,
    dump_ir: bool,
    zk_weights: Option<&std::path::Path>,
    options: &nsl_codegen::CompileOptions,
) {
    let (interner, parse_result, analysis) = frontend(file);

    let (obj_bytes, zk_proof_fns, zk_results) = match nsl_codegen::compile_with_zk_info(
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

    // M55c: Write ZK proof files alongside the binary
    if zk_proof_fns.is_empty() {
        eprintln!("[nsl/zk] no @zk_proof functions found — ZK circuit output skipped");
    } else {
        eprintln!(
            "[nsl/zk] found {} @zk_proof function(s):",
            zk_proof_fns.len()
        );

        if let Some(wpath) = zk_weights {
            eprintln!("[nsl/zk]   weights: {}", wpath.display());
        }

        for (fn_name, result) in &zk_results {
            let report = nsl_codegen::zk::stats::format_stats(&result.stats, fn_name);
            eprint!("{}", report);

            // Write proof file if proof was generated
            if let Some(ref proof) = result.proof {
                let proof_path = file.with_extension(format!("{}.proof", fn_name));
                if let Err(e) = std::fs::write(&proof_path, &proof.data) {
                    eprintln!("[nsl/zk] error writing proof: {e}");
                } else {
                    eprintln!("[nsl/zk]   proof: {} ({} bytes, {} folds)",
                        proof_path.display(), proof.data.len(), proof.num_folds);
                }

                // Write public inputs as JSON
                let pi_path = file.with_extension(format!("{}.public.json", fn_name));
                let pi_entries: Vec<String> = proof.public_inputs.iter()
                    .map(|v| format!("{:?}", v))
                    .collect();
                let pi_json = format!(
                    "{{\"public_inputs\":[{}],\"num_folds\":{}}}",
                    pi_entries.join(","), proof.num_folds
                );
                if let Err(e) = std::fs::write(&pi_path, &pi_json) {
                    eprintln!("[nsl/zk] error writing public inputs: {e}");
                } else {
                    eprintln!("[nsl/zk]   public inputs: {}", pi_path.display());
                }
            }
        }
    }

    // Write normal object / link as usual.
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

    if emit_obj {
        println!("Wrote {}", obj_path.display());
        return;
    }

    let exe_path = if let Some(out) = output {
        out
    } else {
        nsl_codegen::linker::default_output_path(file)
    };

    match nsl_codegen::linker::link(&obj_path, &exe_path) {
        Ok(()) => {
            let _ = std::fs::remove_file(&obj_path);
            println!("Built {}", exe_path.display());
        }
        Err(e) => {
            eprintln!("link error: {e}");
            process::exit(1);
        }
    }
}

/// M55c: Handle `nsl zk <subcommand>`.
fn run_zk_cmd(cmd: ZkCmd) {
    match cmd {
        ZkCmd::Stats { file } => {
            match std::fs::read(&file) {
                Ok(data) => {
                    if data.len() < 12 {
                        eprintln!("[nsl/zk] invalid proof file: too short ({} bytes)", data.len());
                        process::exit(1);
                    }
                    let num_folds = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    let instance_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]);
                    let num_rounds = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);
                    println!("Proof stats:");
                    println!("  File:        {}", file.display());
                    println!("  Size:        {} bytes", data.len());
                    println!("  Folds:       {}", num_folds);
                    println!("  Instance:    {} elements", instance_len);
                    println!("  SC rounds:   {}", num_rounds);
                }
                Err(e) => {
                    eprintln!("error reading {}: {e}", file.display());
                    process::exit(1);
                }
            }
        }
        ZkCmd::Prove { file, pk: _, input: _, output } => {
            // For the folding backend, proofs are generated during compilation.
            eprintln!("[nsl/zk] For the folding backend, proofs are generated during `nsl build --zk-circuit`.");
            eprintln!("[nsl/zk] The proof file is written alongside the binary as <file>.<fn_name>.proof");
            if let Some(ref out) = output {
                eprintln!("[nsl/zk] Requested output: {}", out.display());
            }
            eprintln!("[nsl/zk] To generate a proof, run: nsl build --zk-circuit {}", file.display());
        }
        ZkCmd::Verify { vk: _, proof, public: _ } => {
            match std::fs::read(&proof) {
                Ok(data) => {
                    if data.len() < 12 {
                        eprintln!("INVALID: proof file too short ({} bytes)", data.len());
                        process::exit(1);
                    }

                    let num_folds = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
                    let instance_len = u32::from_le_bytes([data[4], data[5], data[6], data[7]]) as usize;
                    let num_rounds = u32::from_le_bytes([data[8], data[9], data[10], data[11]]);

                    // Reconstruct ZkProof for verification
                    let zk_proof = nsl_codegen::zk::backend::ZkProof {
                        data: data.clone(),
                        num_folds,
                        public_inputs: vec![vec![0u8; 4]; instance_len],
                        public_outputs: Vec::new(),
                    };

                    use nsl_codegen::zk::backend::FoldingBackend;
                    type M31Prover = nsl_codegen::zk::folding::FoldingProver<nsl_codegen::zk::field_m31::Mersenne31Field>;

                    match M31Prover::verify(&zk_proof, &[]) {
                        Ok(true) => {
                            println!("VERIFIED: proof is valid ({} folds, {} sumcheck rounds)",
                                num_folds, num_rounds);
                        }
                        Ok(false) => {
                            println!("INVALID: proof verification failed");
                            process::exit(1);
                        }
                        Err(e) => {
                            eprintln!("VERIFICATION ERROR: {e}");
                            process::exit(1);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("error reading {}: {e}", proof.display());
                    process::exit(1);
                }
            }
        }
    }
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

fn run_run(file: &PathBuf, program_args: &[String], profile_memory: bool, profile_kernels: bool, profile: bool, cuda_sync: bool, options: &nsl_codegen::CompileOptions) {
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
    if cuda_sync {
        cmd.env("NSL_CUDA_SYNC", "1");
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

// ─────────────────────────────────────────────────────────────────────────────
// NSLM ↔ Safetensors bidirectional conversion
// ─────────────────────────────────────────────────────────────────────────────

/// NSLM binary format:
///   [0..4]   magic: b"NSLM"
///   [4..8]   version: u32 LE (currently 1)
///   [8..16]  header_size: u64 LE  (byte length of JSON that follows)
///   [16 .. 16+header_size]  JSON: {"params":[{"name":...,"shape":[...],"dtype":"f32"|"f64","offset":N,"nbytes":N}]}
///   padding to next 64-byte boundary (from offset 16+header_size)
///   raw little-endian tensor data concatenated in params order

fn convert_nslm_to_safetensors(input_path: &std::path::Path, output_path: &std::path::Path) -> Result<(), String> {
    use std::collections::HashMap;

    let data = std::fs::read(input_path)
        .map_err(|e| format!("cannot read '{}': {}", input_path.display(), e))?;

    if data.len() < 16 {
        return Err(format!(
            "'{}' is too small to be a valid NSLM file ({} bytes)",
            input_path.display(), data.len()
        ));
    }
    if &data[0..4] != b"NSLM" {
        return Err(format!("'{}' is not a valid NSLM file (bad magic)", input_path.display()));
    }
    let version = u32::from_le_bytes(data[4..8].try_into().unwrap());
    if version != 1 {
        return Err(format!(
            "unsupported NSLM version {} in '{}' (expected 1)",
            version, input_path.display()
        ));
    }
    let header_size = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
    if 16 + header_size > data.len() {
        return Err(format!(
            "NSLM header claims {} bytes but file is only {} bytes",
            header_size, data.len()
        ));
    }
    let header_json: serde_json::Value = serde_json::from_slice(&data[16..16 + header_size])
        .map_err(|e| format!("NSLM header JSON parse error: {}", e))?;

    let params = header_json["params"]
        .as_array()
        .ok_or_else(|| "NSLM header missing 'params' array".to_string())?;

    // Compute start of raw data (64-byte aligned from byte 16+header_size)
    let total_header = 16 + header_size;
    let padding = (64 - (total_header % 64)) % 64;
    let data_start = total_header + padding;

    // Build safetensors entries: convert f64 → f32
    let mut owned: Vec<(String, Vec<u8>, Vec<usize>)> = Vec::with_capacity(params.len());

    for param in params {
        let name = param["name"]
            .as_str()
            .ok_or_else(|| "NSLM param missing 'name'".to_string())?
            .to_owned();
        let dtype_str = param["dtype"]
            .as_str()
            .ok_or_else(|| format!("NSLM param '{}' missing 'dtype'", name))?;
        let offset = param["offset"]
            .as_u64()
            .ok_or_else(|| format!("NSLM param '{}' missing 'offset'", name))? as usize;
        let nbytes = param["nbytes"]
            .as_u64()
            .ok_or_else(|| format!("NSLM param '{}' missing 'nbytes'", name))? as usize;
        let shape: Vec<usize> = param["shape"]
            .as_array()
            .ok_or_else(|| format!("NSLM param '{}' missing 'shape'", name))?
            .iter()
            .map(|v| v.as_i64().unwrap_or(0) as usize)
            .collect();

        let abs_start = data_start + offset;
        let abs_end = abs_start + nbytes;
        if abs_end > data.len() {
            return Err(format!(
                "NSLM tensor '{}' data [{}..{}] exceeds file size {}",
                name, abs_start, abs_end, data.len()
            ));
        }
        let raw = &data[abs_start..abs_end];

        // Convert to f32 LE bytes for safetensors
        let f32_bytes: Vec<u8> = match dtype_str {
            "f64" => {
                let count = raw.len() / 8;
                let mut out = Vec::with_capacity(count * 4);
                for i in 0..count {
                    let b: [u8; 8] = raw[i * 8..(i + 1) * 8].try_into().unwrap();
                    let v = f64::from_le_bytes(b) as f32;
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
            "f32" => raw.to_vec(),
            other => {
                return Err(format!("NSLM tensor '{}' has unsupported dtype '{}'", name, other));
            }
        };

        owned.push((name, f32_bytes, shape));
    }

    // Serialize to safetensors
    let st_data: HashMap<String, safetensors::tensor::TensorView<'_>> = owned
        .iter()
        .map(|(name, bytes, shape)| {
            let view = safetensors::tensor::TensorView::new(
                safetensors::Dtype::F32,
                shape.clone(),
                bytes.as_slice(),
            )
            .map_err(|e| format!("safetensors TensorView error for '{}': {}", name, e))?;
            Ok((name.clone(), view))
        })
        .collect::<Result<HashMap<_, _>, String>>()?;

    let serialized = safetensors::tensor::serialize(&st_data, &None)
        .map_err(|e| format!("safetensors serialize error: {}", e))?;

    std::fs::write(output_path, &serialized)
        .map_err(|e| format!("cannot write '{}': {}", output_path.display(), e))?;

    Ok(())
}

/// Shard of metadata used when writing NSLM header.
#[derive(Debug)]
struct NslmParamMeta {
    name: String,
    shape: Vec<i64>,
    dtype: &'static str,
    offset: u64,
    nbytes: u64,
}

fn convert_safetensors_to_nslm(input_path: &std::path::Path, output_path: &std::path::Path) -> Result<(), String> {
    use std::io::Write;

    let bytes = std::fs::read(input_path)
        .map_err(|e| format!("cannot read '{}': {}", input_path.display(), e))?;

    let st = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| format!("safetensors parse error in '{}': {}", input_path.display(), e))?;

    // Collect tensors in iteration order
    let mut metas: Vec<NslmParamMeta> = Vec::new();
    let mut data_blocks: Vec<Vec<u8>> = Vec::new();
    let mut data_offset: u64 = 0;

    for (name, view) in st.tensors() {
        let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();

        // Convert incoming dtype to f32 LE bytes (NSLM will store as f32)
        let f32_bytes: Vec<u8> = match view.dtype() {
            safetensors::Dtype::F32 => view.data().to_vec(),
            safetensors::Dtype::F64 => {
                let raw = view.data();
                let count = raw.len() / 8;
                let mut out = Vec::with_capacity(count * 4);
                for i in 0..count {
                    let b: [u8; 8] = raw[i * 8..(i + 1) * 8].try_into().unwrap();
                    let v = f64::from_le_bytes(b) as f32;
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
            safetensors::Dtype::F16 => {
                let raw = view.data();
                let count = raw.len() / 2;
                let mut out = Vec::with_capacity(count * 4);
                for i in 0..count {
                    let b: [u8; 2] = raw[i * 2..(i + 1) * 2].try_into().unwrap();
                    let v = f32::from(half::f16::from_le_bytes(b));
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
            safetensors::Dtype::BF16 => {
                let raw = view.data();
                let count = raw.len() / 2;
                let mut out = Vec::with_capacity(count * 4);
                for i in 0..count {
                    let b: [u8; 2] = raw[i * 2..(i + 1) * 2].try_into().unwrap();
                    let v = f32::from(half::bf16::from_le_bytes(b));
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
            safetensors::Dtype::I32 => {
                let raw = view.data();
                let count = raw.len() / 4;
                let mut out = Vec::with_capacity(count * 4);
                for i in 0..count {
                    let b: [u8; 4] = raw[i * 4..(i + 1) * 4].try_into().unwrap();
                    let v = i32::from_le_bytes(b) as f32;
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
            safetensors::Dtype::I64 => {
                let raw = view.data();
                let count = raw.len() / 8;
                let mut out = Vec::with_capacity(count * 4);
                for i in 0..count {
                    let b: [u8; 8] = raw[i * 8..(i + 1) * 8].try_into().unwrap();
                    let v = i64::from_le_bytes(b) as f32;
                    out.extend_from_slice(&v.to_le_bytes());
                }
                out
            }
            other => {
                return Err(format!(
                    "safetensors tensor '{}' has unsupported dtype {:?}",
                    name, other
                ));
            }
        };

        let nbytes = f32_bytes.len() as u64;
        metas.push(NslmParamMeta {
            name: name.to_owned(),
            shape,
            dtype: "f32",
            offset: data_offset,
            nbytes,
        });
        data_offset += nbytes;
        data_blocks.push(f32_bytes);
    }

    // Build NSLM JSON header
    let params_json: Vec<String> = metas
        .iter()
        .map(|m| {
            format!(
                r#"{{"name":"{}","shape":{:?},"dtype":"{}","offset":{},"nbytes":{}}}"#,
                m.name, m.shape, m.dtype, m.offset, m.nbytes
            )
        })
        .collect();
    let header = format!(r#"{{"params":[{}]}}"#, params_json.join(","));
    let header_bytes = header.as_bytes();

    // Write NSLM file
    let mut file = std::fs::File::create(output_path)
        .map_err(|e| format!("cannot create '{}': {}", output_path.display(), e))?;

    let magic: &[u8; 4] = b"NSLM";
    let version: u32 = 1;
    let header_size = header_bytes.len() as u64;

    file.write_all(magic)
        .map_err(|e| format!("write error (magic): {}", e))?;
    file.write_all(&version.to_le_bytes())
        .map_err(|e| format!("write error (version): {}", e))?;
    file.write_all(&header_size.to_le_bytes())
        .map_err(|e| format!("write error (header_size): {}", e))?;
    file.write_all(header_bytes)
        .map_err(|e| format!("write error (header): {}", e))?;

    // Pad to 64-byte alignment (measured from byte 0)
    let total_header = 4 + 4 + 8 + header_bytes.len(); // magic + version + header_size + header
    let padding = (64 - (total_header % 64)) % 64;
    let pad_buf = [0u8; 64];
    file.write_all(&pad_buf[..padding])
        .map_err(|e| format!("write error (padding): {}", e))?;

    // Raw tensor data
    for block in &data_blocks {
        file.write_all(block)
            .map_err(|e| format!("write error (tensor data): {}", e))?;
    }

    Ok(())
}

fn run_convert(input: &std::path::Path, output: &std::path::Path) {
    let in_ext = input.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    let out_ext = output.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();

    match (in_ext.as_str(), out_ext.as_str()) {
        ("nslm", "safetensors") => {
            match convert_nslm_to_safetensors(input, output) {
                Ok(()) => println!("Converted {} → {}", input.display(), output.display()),
                Err(e) => {
                    eprintln!("error: {}", e);
                    process::exit(1);
                }
            }
        }
        ("safetensors", "nslm") => {
            match convert_safetensors_to_nslm(input, output) {
                Ok(()) => println!("Converted {} → {}", input.display(), output.display()),
                Err(e) => {
                    eprintln!("error: {}", e);
                    process::exit(1);
                }
            }
        }
        _ => {
            eprintln!(
                "error: unsupported conversion '{}.{}' → '{}.{}'.\n\
                 Supported: .nslm → .safetensors, .safetensors → .nslm",
                input.display(), in_ext, output.display(), out_ext
            );
            process::exit(1);
        }
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
            run_run(file, &[], false, false, false, false, &nsl_codegen::CompileOptions::default());
        }
        "safetensors" => {
            if ext != "nslm" {
                eprintln!(
                    "error: safetensors conversion requires an .nslm input file, got '{}'",
                    file.display()
                );
                process::exit(1);
            }
            // Compute default output path: same stem with .safetensors extension
            let default_out;
            let out_path: &std::path::Path = if let Some(o) = output {
                o
            } else {
                default_out = file.with_extension("safetensors");
                &default_out
            };
            match convert_nslm_to_safetensors(file, out_path) {
                Ok(()) => println!("Converted {} → {}", file.display(), out_path.display()),
                Err(e) => {
                    eprintln!("error: {}", e);
                    process::exit(1);
                }
            }
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

fn run_tokenize(dirs: &[String], output: &PathBuf, vocab_size: usize, min_freq: u64, ext: &str) {
    use std::io::Write;

    // Default directories if none specified
    let search_dirs: Vec<String> = if dirs.is_empty() {
        vec!["stdlib".into(), "examples".into(), "tests".into(), "models".into()]
    } else {
        dirs.to_vec()
    };

    // Collect all source files
    let mut source_files: Vec<PathBuf> = Vec::new();
    for dir in &search_dirs {
        let dir_path = PathBuf::from(dir);
        if !dir_path.exists() {
            eprintln!("warning: directory '{}' not found, skipping", dir);
            continue;
        }
        collect_files_recursive(&dir_path, ext, &mut source_files);
    }
    source_files.sort();

    if source_files.is_empty() {
        eprintln!("error: no .{ext} files found in {:?}", search_dirs);
        process::exit(1);
    }

    eprintln!("[tokenize] Found {} .{} files across {} directories", source_files.len(), ext, search_dirs.len());

    // Concatenate all source text into a temporary corpus file
    let corpus_path = std::env::temp_dir().join("nsl_tokenizer_corpus.txt");
    {
        let mut corpus = std::fs::File::create(&corpus_path).unwrap_or_else(|e| {
            eprintln!("error: could not create corpus file: {e}");
            process::exit(1);
        });
        let mut total_bytes: usize = 0;
        for file in &source_files {
            match std::fs::read_to_string(file) {
                Ok(content) => {
                    total_bytes += content.len();
                    let _ = corpus.write_all(content.as_bytes());
                    let _ = corpus.write_all(b"\n");
                }
                Err(e) => {
                    eprintln!("warning: could not read '{}': {e}", file.display());
                }
            }
        }
        eprintln!("[tokenize] Corpus: {} bytes from {} files", total_bytes, source_files.len());
    }

    // Train BPE tokenizer using the runtime's nsl_bpe_train
    eprintln!("[tokenize] Training BPE tokenizer (vocab_size={}, min_freq={})...", vocab_size, min_freq);

    // We call the runtime's BPE trainer directly via Rust (not through FFI)
    use tokenizers::models::bpe::{BPE, BpeTrainer};
    use tokenizers::pre_tokenizers::byte_level::ByteLevel;
    use tokenizers::Tokenizer;

    let trainer = BpeTrainer::builder()
        .vocab_size(vocab_size)
        .min_frequency(min_freq)
        .special_tokens(vec![
            tokenizers::AddedToken::from("<|endoftext|>".to_string(), true),
            tokenizers::AddedToken::from("<|padding|>".to_string(), true),
            tokenizers::AddedToken::from("<|fim_prefix|>".to_string(), true),
            tokenizers::AddedToken::from("<|fim_middle|>".to_string(), true),
            tokenizers::AddedToken::from("<|fim_suffix|>".to_string(), true),
        ])
        .build();

    let mut tokenizer = Tokenizer::new(BPE::default());
    tokenizer.with_pre_tokenizer(Some(
        tokenizers::PreTokenizerWrapper::ByteLevel(ByteLevel::default()),
    ));

    let mut trainer_wrapper = tokenizers::models::TrainerWrapper::BpeTrainer(trainer);
    match tokenizer.train_from_files(&mut trainer_wrapper, vec![corpus_path.to_string_lossy().to_string()]) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("error: BPE training failed: {e}");
            process::exit(1);
        }
    }

    // Ensure output directory exists
    if let Some(parent) = output.parent() {
        if !parent.exists() {
            let _ = std::fs::create_dir_all(parent);
        }
    }

    // Save tokenizer
    match tokenizer.save(output.to_string_lossy().as_ref(), true) {
        Ok(_) => {}
        Err(e) => {
            eprintln!("error: could not save tokenizer to '{}': {e}", output.display());
            process::exit(1);
        }
    }

    let final_vocab = tokenizer.get_vocab_size(true);
    eprintln!("[tokenize] Saved tokenizer to '{}' (vocab_size={})", output.display(), final_vocab);

    // Clean up corpus
    let _ = std::fs::remove_file(&corpus_path);

    // Test: encode a sample string
    let sample = "fn forward(self, x: Tensor) -> Tensor:";
    match tokenizer.encode(sample, false) {
        Ok(encoding) => {
            let tokens = encoding.get_tokens();
            eprintln!("[tokenize] Sample: \"{}\" -> {} tokens: {:?}", sample, tokens.len(), &tokens[..tokens.len().min(10)]);
        }
        Err(_) => {}
    }
}

fn collect_files_recursive(dir: &PathBuf, ext: &str, out: &mut Vec<PathBuf>) {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_files_recursive(&path, ext, out);
        } else if path.extension().and_then(|e| e.to_str()) == Some(ext) {
            out.push(path);
        }
    }
}
