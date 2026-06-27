mod ast_scan;
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

/// Output format for `--training-report`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingReportFormat {
    Text,
    Json,
}

/// Scan a parsed module for a top-level `train { ... }` block.
///
/// Dev Tools Phase 4 Task 6: `nsl run --monitor` uses this detection to
/// decide whether to enable the health monitor (train program) or fall
/// back to the Phase 1/2 kernel-timing profile path (non-train program).
/// Train blocks in NSL are a top-level construct, so a flat scan suffices.
fn has_train_block(module: &nsl_ast::Module) -> bool {
    module
        .stmts
        .iter()
        .any(|s| matches!(s.kind, nsl_ast::stmt::StmtKind::TrainBlock(_)))
}

/// CLI wrapper for `nsl_codegen::WggoImportance` — keeps `clap` out of `nsl-codegen`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default)]
#[clap(rename_all = "lower")]
enum CliWggoImportance {
    #[default]
    Auto,
    Magnitude,
    Grad,
}

impl From<CliWggoImportance> for nsl_codegen::WggoImportance {
    fn from(v: CliWggoImportance) -> Self {
        match v {
            CliWggoImportance::Auto => Self::Auto,
            CliWggoImportance::Magnitude => Self::Magnitude,
            CliWggoImportance::Grad => Self::Grad,
        }
    }
}

// The clap command enum carries many subcommand-specific flags, so keeping it
// as a single enum is clearer than splitting every large variant into boxes.
#[allow(clippy::large_enum_variant)]
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

        /// Print a compile-time shape-propagation trace
        #[arg(long)]
        shapes: bool,

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

        /// Emit a training-pipeline decision audit for every train block in the file.
        /// Pass without value for text output, or `--training-report=json` for JSON.
        #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "text")]
        training_report: Option<TrainingReportFormat>,

        /// CEP: run hardware-aware architecture search.
        #[arg(long)]
        cep_search: bool,
        /// CEP: print a bare CompilationProfile for the model (paper §7.1
        /// differentiator — equivalent of PyTorch's missing 'nsl check --perf').
        #[arg(long)]
        cep_profile: bool,
        /// CEP: target GPU for analysis (e.g. H100-SXM).
        #[arg(long)]
        cep_target: Option<String>,
        /// CEP: write the delta JSON here (default: <model>.cep.json).
        #[arg(long)]
        cep_out: Option<PathBuf>,

        /// WRGA paper §8.3: emit the WRGA compilation report without running
        /// codegen output. Pass without value (or `-`) for stdout; provide a
        /// path to write to a file. Sister of `nsl build --wrga-report` but
        /// avoids producing a `.o`. Requires `@wrga` decorators in the source.
        #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "-")]
        wrga_analyze: Option<PathBuf>,

        /// WRGA paper §8.3: target GPU for the WRGA roofline analysis (e.g.
        /// "H100-SXM", "A100-PCIe", "RTX-4090"). Overrides any `target=` set
        /// on `@wrga(...)` decorators. Looked up in the codegen GPU spec
        /// database; empty / unknown falls back to the database default.
        #[arg(long)]
        wrga_target: Option<String>,
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

        /// ELTLS instrumentation: print a GPU memory report at end of run.
        /// Prints epilog_frees_total plus (with --features cuda) the
        /// live caching-allocator block summary and driver/allocator stats.
        #[arg(long)]
        gpu_mem_report: bool,

        /// Render predicted-vs-actual kernel timings instead of running the program
        #[arg(long)]
        monitor: bool,

        /// Activate @inspect hooks: dump tensor stats/contents to .nsl-inspect/
        #[arg(long)]
        inspect: bool,

        /// CSHA: attention-fusion mode ("auto", "boundary", "pipeline",
        /// "block", or "off").  Passing `--csha` without a value enables
        /// auto mode.  Mirrors `nsl build --csha`.
        #[arg(long, value_name = "MODE", num_args = 0..=1, default_missing_value = "auto")]
        csha: Option<String>,

        /// CSHA: print the attention-fusion report to stderr
        #[arg(long)]
        csha_report: bool,

        /// M38a/M56: Enable linear types ownership checking. Required for
        /// agent declarations (M56). Closes Task 20 of the M56 plan.
        #[arg(long)]
        linear_types: bool,

        /// CPDT: planner mode ("full", "zero_only", or "off").
        /// Passing `--cpdt` without a value enables full mode.
        /// Mirrors `nsl build --cpdt` so precision-adaptive training
        /// executes end-to-end via `nsl run`.
        #[arg(long, value_name = "MODE", num_args = 0..=1, default_missing_value = "full")]
        cpdt: Option<String>,

        /// CPDT: number of GPUs in the target cluster. Required when `--cpdt` is set.
        #[arg(long, value_name = "N")]
        cpdt_num_gpus: Option<u32>,

        /// CPDT: intra-node bandwidth in bytes/sec (default 9e11 = 900 GB/s).
        #[arg(long, value_name = "BPS", default_value_t = 9e11)]
        cpdt_intra_bw: f64,

        /// CPDT: inter-node bandwidth in bytes/sec (default 1e11 = 100 GB/s).
        #[arg(long, value_name = "BPS", default_value_t = 1e11)]
        cpdt_inter_bw: f64,

        /// CPDT: emit the full plan to stdout. Implies `--cpdt` (full mode).
        #[arg(long, default_value_t = false)]
        cpdt_report: bool,

        /// Path to the model weights file (.safetensors) for the
        /// weight-aware CPDT path. Mirrors `nsl build -w/--weights`.
        #[arg(short = 'w', long)]
        weights: Option<PathBuf>,

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

        /// WRGA Milestone B.1: Emit the WRGA compilation report.
        /// With no value, prints to stdout.  With a path, writes to that file.
        #[arg(long, value_name = "PATH", num_args = 0..=1, default_missing_value = "-")]
        wrga_report: Option<PathBuf>,

        /// WRGA Milestone B.2 Task 3: fold WRGA memory hints into real
        /// allocations.  Default off — observational mode ships per B.1.
        #[arg(long, default_value_t = false)]
        wrga_fold_allocations: bool,

        /// WGGO: global-optimization mode ("full", "greedy", or "off").
        /// Passing `--wggo` without a value enables full mode.
        #[arg(long, value_name = "MODE", num_args = 0..=1, default_missing_value = "full")]
        wggo: Option<String>,

        /// WGGO: print the global-optimization report to stderr
        #[arg(long)]
        wggo_report: bool,

        /// WGGO Stage 3: path to a `.nslweights` sidecar for real
        /// weight-based importance scoring.  Without this flag the
        /// analyzer falls back to uniform head scores.
        #[arg(long, value_name = "PATH")]
        wggo_weights: Option<PathBuf>,

        /// WGGO head-importance scoring mode.
        /// - `auto` (default): gradient scoring when calibration sidecar present, else magnitude.
        /// - `magnitude`: force magnitude scoring even with calibration data.
        /// - `grad`: require gradient scoring; errors if no calibration sidecar present.
        #[arg(long, value_enum, default_value_t = CliWggoImportance::Auto)]
        wggo_importance: CliWggoImportance,

        /// WGGO Stage 3: fraction of heads the default
        /// `min_retained_importance` threshold allows to be pruned.
        /// Clamped to [0.0, 0.9]; default 0.25.
        #[arg(long, value_name = "F")]
        wggo_prune_fraction: Option<f64>,

        /// CSHA: attention-fusion mode ("auto", "boundary", "pipeline",
        /// "block", or "off").  Passing `--csha` without a value enables
        /// auto mode.
        #[arg(long, value_name = "MODE", num_args = 0..=1, default_missing_value = "auto")]
        csha: Option<String>,

        /// CSHA: print the attention-fusion report to stderr
        #[arg(long)]
        csha_report: bool,

        /// CPDT: planner mode ("full", "zero_only", or "off").
        /// Passing `--cpdt` without a value enables full mode.
        #[arg(long, value_name = "MODE", num_args = 0..=1, default_missing_value = "full")]
        cpdt: Option<String>,

        /// CPDT: number of GPUs in the target cluster. Required when `--cpdt` is set.
        #[arg(long, value_name = "N")]
        cpdt_num_gpus: Option<u32>,

        /// CPDT: intra-node bandwidth in bytes/sec (default 9e11 = 900 GB/s).
        #[arg(long, value_name = "BPS", default_value_t = 9e11)]
        cpdt_intra_bw: f64,

        /// CPDT: inter-node bandwidth in bytes/sec (default 1e11 = 100 GB/s).
        #[arg(long, value_name = "BPS", default_value_t = 1e11)]
        cpdt_inter_bw: f64,

        /// CPDT: emit the full plan to stdout. Implies `--cpdt` (full mode).
        #[arg(long, default_value_t = false)]
        cpdt_report: bool,

        /// Path to calibration dataset (.bin or .safetensors).  When
        /// omitted, calibration is skipped entirely.
        #[arg(long, value_name = "PATH")]
        calibration_data: Option<PathBuf>,

        /// Calibration failure policy.  Default `required` aborts the
        /// build on infrastructure errors; `best-effort` warns and
        /// falls back.  Degenerate-data errors are always fatal.
        #[arg(long, value_name = "MODE", default_value = "required")]
        calibrate: String,

        /// Number of calibration samples to consume (default 512).
        /// Truncated to the dataset size with a warning when smaller.
        #[arg(long, value_name = "N", default_value_t = 512)]
        calibration_samples: u32,

        #[arg(long, value_name = "N", default_value_t = 8)]
        calibration_batch_size: u32,

        #[arg(long, value_name = "SECONDS", default_value_t = 600)]
        calibration_timeout: u64,

        /// CEP: run compilation-verified pruning (requires --weights).
        #[arg(long)]
        cep_prune: bool,
        /// CEP Mode 3 (paper §2.2): run joint prune-search (heads + FFN + layer drops).
        /// Requires --weights. Mutually exclusive with --cep-prune.
        #[arg(long)]
        cep_joint: bool,
        /// CEP: target GPU for analysis (e.g. H100-SXM).
        #[arg(long)]
        cep_target: Option<String>,
        /// CEP: override the decorator's prune sparsity.
        #[arg(long)]
        cep_sparsity: Option<f64>,
        /// CEP: write the delta JSON here (default: <model>.cep.json).
        #[arg(long)]
        cep_out: Option<PathBuf>,
        /// CEP: also emit the pruned (sliced) weights to this .safetensors path.
        #[arg(long)]
        cep_emit_weights: Option<PathBuf>,
        /// CEP SP2: also emit the rewritten NSL source with pruned dims to this path.
        #[arg(long)]
        cep_emit_source: Option<PathBuf>,
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

    /// Predictive performance profile for a target GPU
    Profile {
        /// Path to the .nsl file
        file: PathBuf,

        /// Target GPU (e.g., "h100", "a100", "rtx-4090")
        #[arg(long, default_value = "h100")]
        target: String,

        /// Dtype to cost (e.g., "bf16", "fp16", "fp8")
        #[arg(long, default_value = "bf16")]
        dtype: String,

        /// Concrete batch size to resolve shape symbols
        #[arg(long, default_value_t = 1)]
        batch: u64,

        /// Concrete sequence length to resolve shape symbols
        #[arg(long, default_value_t = 2048)]
        seq: u64,

        /// Extra shape-env bindings as "name=N" (repeatable)
        #[arg(long)]
        dim: Vec<String>,

        /// Disable fusion plan in the report
        #[arg(long)]
        no_fusion: bool,

        /// Include memory timeline in the report (Task 5)
        #[arg(long)]
        memory: bool,

        /// Entry point: "auto", "train", or "fn:<name>"
        #[arg(long, default_value = "auto")]
        entry: String,

        /// Emit machine-readable JSON instead of a formatted table
        #[arg(long)]
        json: bool,

        /// Run WGGO Full mode and append/attach the decision explanation
        #[arg(long)]
        explain_wggo: bool,
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

    /// M57: Compile an NSL file to synthesizable Verilog for FPGA targets.
    ///
    /// NOTE: --target fpga is not yet end-to-end functional.
    /// PRs 1-4 shipped the HIR + KIR->HIR + HIR->Verilog + Yosys-gate +
    /// fixture infrastructure (M57 milestone).  AST -> structured KIR dispatch,
    /// HIR port/wire generation, and CLI dispatch wiring are deferred to
    /// M57.1 (v1 closure follow-on).
    FpgaCompile {
        /// Path to the .nsl file to compile
        file: PathBuf,

        /// Output directory for the emitted .v file
        #[arg(short, long)]
        output_dir: Option<PathBuf>,

        /// Path to the weight fixture binary (.bin) for the v1 MLP.
        /// Defaults to <input_dir>/<source_basename>_weights.bin (sidecar convention).
        /// See spec §6.1 for the sidecar lookup rules.
        #[arg(long)]
        fixture: Option<PathBuf>,

        /// Emit test-tap ports on all intermediate signals (Layer 2 + Layer 3 gates).
        /// Only valid with --target fpga.
        #[arg(long)]
        test_taps: bool,

        /// M57.2: emit the clocked sequential FSM instead of the combinational netlist.
        /// Prints `total_cycles=<N>` to stdout after writing the .v file.
        #[arg(long)]
        seq: bool,
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
        Cli::Check {
            file,
            dump_tokens,
            dump_ast,
            dump_types,
            shapes,
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
            training_report,
            cep_search,
            cep_profile,
            cep_target,
            cep_out,
            wrga_analyze,
            wrga_target,
        } => {
            if cep_search && cep_profile {
                eprintln!("error: --cep-search and --cep-profile are mutually exclusive");
                std::process::exit(1);
            }
            // WRGA paper §8.3: `--wrga-analyze` short-circuits the check path
            // to emit just the WRGA compilation report (no .o, no shape trace,
            // no NaN analysis side effects). Mutually exclusive with CEP modes
            // since CEP early-exits above.
            if let Some(ref report_path) = wrga_analyze {
                std::process::exit(run_check_wrga_analyze(
                    &file,
                    report_path,
                    wrga_target.as_deref(),
                ));
            }
            if cep_search {
                let ov = nsl_codegen::cep::CliOverrides {
                    target: cep_target,
                    sparsity: None,
                    cep_out,
                    cep_emit_weights: None,
                    cep_emit_source: None,
                };
                std::process::exit(run_cep_search(&file, &ov));
            }
            if cep_profile {
                let ov = nsl_codegen::cep::CliOverrides {
                    target: cep_target,
                    sparsity: None,
                    cep_out: None,
                    cep_emit_weights: None,
                    cep_emit_source: None,
                };
                std::process::exit(run_cep_profile(&file, weights.as_deref(), &ov));
            }
            if shapes {
                let src = match std::fs::read_to_string(&file) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("error: could not read file '{}': {e}", file.display());
                        process::exit(1);
                    }
                };
                match nsl_cli::shape_debug::ShapeDebugInput::from_source(
                    &src,
                    &file.display().to_string(),
                ) {
                    Ok(input) => {
                        println!("{}", nsl_cli::shape_debug::format_trace(&input));
                        return;
                    }
                    Err(e) => {
                        eprintln!("error: shape debug failed: {e}");
                        process::exit(1);
                    }
                }
            }
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

            // FASE: Training-pipeline decision audit
            if let Some(format) = training_report {
                let source = std::fs::read_to_string(&file).unwrap_or_default();
                let mut tr_interner = Interner::new();
                let mut tr_source_map = SourceMap::new();
                let tr_file_id = tr_source_map.add_file(file.display().to_string(), source.clone());
                let (tr_tokens, _) = nsl_lexer::tokenize(&source, tr_file_id, &mut tr_interner);
                let tr_parse = nsl_parser::parse(&tr_tokens, &mut tr_interner);
                let report = nsl_codegen::training_report::build_report(
                    &tr_parse.module,
                    &tr_interner,
                    file.as_path(),
                );
                match format {
                    TrainingReportFormat::Text => {
                        println!("{}", report);
                    }
                    TrainingReportFormat::Json => {
                        let json = serde_json::to_string_pretty(&report)
                            .expect("serialize training report");
                        println!("{}", json);
                    }
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
            wrga_report,
            wrga_fold_allocations,
            wggo,
            wggo_report,
            wggo_weights,
            wggo_importance,
            wggo_prune_fraction,
            csha,
            csha_report,
            cpdt,
            cpdt_num_gpus,
            cpdt_intra_bw,
            cpdt_inter_bw,
            cpdt_report,
            calibration_data,
            calibrate,
            calibration_samples,
            calibration_batch_size,
            calibration_timeout,
            cep_prune,
            cep_joint,
            cep_target,
            cep_sparsity,
            cep_out,
            cep_emit_weights,
            cep_emit_source,
        } => {
            // M62a: shared_lib flag is threaded through compile_opts and handled
            // in the build path below.

            if cep_prune && cep_joint {
                eprintln!(
                    "error: --cep-prune and --cep-joint are mutually exclusive (use one)"
                );
                std::process::exit(1);
            }

            if cep_prune {
                let ov = nsl_codegen::cep::CliOverrides {
                    target: cep_target,
                    sparsity: cep_sparsity,
                    cep_out,
                    cep_emit_weights,
                    cep_emit_source,
                };
                std::process::exit(run_cep_prune(&file, weights.as_deref(), &ov));
            }

            if cep_joint {
                let ov = nsl_codegen::cep::CliOverrides {
                    target: cep_target,
                    sparsity: cep_sparsity,
                    cep_out,
                    cep_emit_weights,
                    cep_emit_source,
                };
                std::process::exit(run_cep_joint(&file, weights.as_deref(), &ov));
            }

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

            // Calibration-flag validation per spec §8.
            if calibration_data.is_none() && calibrate.as_str() != "required" {
                eprintln!(
                    "error: --calibrate={} requires --calibration-data <PATH>",
                    calibrate
                );
                process::exit(1);
            }
            match calibrate.as_str() {
                "required" | "best-effort" => {}
                other => {
                    eprintln!(
                        "error: --calibrate value '{}' is not one of required|best-effort",
                        other
                    );
                    process::exit(1);
                }
            }
            if let Some(ref p) = calibration_data {
                if !p.exists() {
                    eprintln!("error: --calibration-data path does not exist: {}", p.display());
                    process::exit(1);
                }
                let ext = p.extension().and_then(|e| e.to_str()).map(|s| s.to_ascii_lowercase());
                match ext.as_deref() {
                    Some("bin") | Some("safetensors") => {}
                    other => {
                        eprintln!(
                            "error: --calibration-data extension {:?} is not one of .bin|.safetensors",
                            other
                        );
                        process::exit(1);
                    }
                }
            }
            if calibration_samples == 0 {
                eprintln!("error: --calibration-samples must be > 0");
                process::exit(1);
            }
            if calibration_batch_size == 0 {
                eprintln!("error: --calibration-batch-size must be > 0");
                process::exit(1);
            }
            if calibration_timeout == 0 {
                eprintln!("error: --calibration-timeout must be > 0");
                process::exit(1);
            }

            // CPDT: --cpdt-report implies --cpdt (full mode unless explicit).
            let cpdt_mode_str: Option<String> = match (cpdt.as_deref(), cpdt_report) {
                (Some(s), _) => Some(s.to_string()),
                (None, true) => Some("full".to_string()),
                (None, false) => None,
            };
            let cpdt_mode = match cpdt_mode_str.as_deref() {
                None => nsl_codegen::cpdt::CpdtMode::Off,
                Some(s) => match nsl_codegen::cpdt::CpdtMode::parse(s) {
                    Some(m) => m,
                    None => {
                        eprintln!(
                            "error: --cpdt value '{}' is not one of full|zero_only|off",
                            s
                        );
                        process::exit(2);
                    }
                },
            };
            let cpdt_cluster = if cpdt_mode != nsl_codegen::cpdt::CpdtMode::Off {
                let n = match cpdt_num_gpus {
                    Some(n) if n >= 1 => n,
                    Some(_) => {
                        eprintln!("nsl: --cpdt-num-gpus must be >= 1");
                        process::exit(2);
                    }
                    None => {
                        eprintln!("nsl: --cpdt requires --cpdt-num-gpus N");
                        process::exit(2);
                    }
                };
                Some(nsl_codegen::cpdt_zero::ClusterSpec {
                    num_gpus: n,
                    memory_budget_bytes: 80u64 * 1024 * 1024 * 1024,
                    intra_bw_bps: cpdt_intra_bw,
                    inter_bw_bps: cpdt_inter_bw,
                    gpus_per_node: n.min(8),
                })
            } else {
                None
            };
            let cpdt_plan_out: Option<
                std::sync::Arc<std::sync::Mutex<Option<nsl_codegen::cpdt::CpdtPlan>>>,
            > = if cpdt_mode != nsl_codegen::cpdt::CpdtMode::Off {
                Some(std::sync::Arc::new(std::sync::Mutex::new(None)))
            } else {
                None
            };

            // Phase 1 CPDT: AST-scan for load_safetensors(...) + @cpdt(weight_aware=...).
            // Resolve the effective weight file via the four-case decision table
            // from the Phase 1 spec §2.1. Design:
            // docs/superpowers/specs/2026-04-21-cpdt-ast-autodetect-design.md.
            let resolved_weight_file: Option<PathBuf> = {
                let ast_source = std::fs::read_to_string(&file).unwrap_or_default();
                let mut ast_interner = Interner::new();
                let ast_file_id = nsl_errors::FileId(0);
                let (ast_tokens, _) =
                    nsl_lexer::tokenize(&ast_source, ast_file_id, &mut ast_interner);
                let ast_parse = nsl_parser::parse(&ast_tokens, &mut ast_interner);
                let ast_weight_ref =
                    ast_scan::find_ast_weight_ref(&ast_parse.module, &ast_interner);
                let ast_weight_aware =
                    ast_scan::find_ast_cpdt_weight_aware(&ast_parse.module, &ast_interner);

                match (&weights, &ast_weight_ref) {
                    (Some(flag_path), Some(ast_path)) => {
                        eprintln!(
                            "warning: --weights {} overrides AST-declared load_safetensors({:?}).",
                            flag_path.display(),
                            ast_path.display(),
                        );
                        Some(flag_path.clone())
                    }
                    (Some(flag_path), None) => Some(flag_path.clone()),
                    (None, Some(ast_path)) => Some(ast_path.clone()),
                    (None, None) => {
                        let cpdt_enabled = cpdt_mode != nsl_codegen::cpdt::CpdtMode::Off;
                        let weight_aware = ast_weight_aware.unwrap_or(true);
                        // The `!standalone` guard is load-bearing: a separate
                        // "--standalone requires --weights" error fires later
                        // (~line 1298). Without this guard, the four-case
                        // message would fire first and replace the standalone-
                        // specific error. If a future refactor moves the
                        // standalone check earlier, this guard can be dropped.
                        if cpdt_enabled && weight_aware && !standalone {
                            eprintln!(
                                "error: --cpdt {} requires weights. Resolve by ONE of:\n\
                                 \n\
                                 1. Add --weights <path.safetensors> to this invocation.\n\
                                 2. Add `let w = load_safetensors(\"<path>\")` to your NSL source.\n\
                                 3. Add `@cpdt(weight_aware=false)` to opt out of the weight-aware path\n\
                                    (produces a CPDT plan without weight-derived tier assignments).",
                                cpdt_mode.as_str(),
                            );
                            process::exit(1);
                        }
                        None
                    }
                }
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
                // otherwise pass through the four-case-resolved weight file from
                // above (AST auto-detect + --weights flag decision table).
                weight_file: if standalone { None } else { resolved_weight_file.clone() },
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
                wrga_inputs: None,
                fused_ce_configs: Vec::new(),
                wrga_fold_allocations,
                wggo_mode: wggo.clone(),
                wggo_report,
                profile_kernels: false,
                target_gpu: "h100".to_string(),
                dtype: "bf16".to_string(),
                manifest_output_path: None,
                profile_source_text: None,
                profile_source_file_name: None,
                health_monitor: false,
                health_flush_interval: None,
                inspect_enabled: false,
                wggo_weights: wggo_weights.clone(),
                wggo_importance: nsl_codegen::WggoImportance::from(wggo_importance),
                wggo_prune_fraction,
                csha_mode: csha.clone(),
                csha_report,
                cpdt_mode,
                cpdt_cluster: cpdt_cluster.clone(),
                cpdt_report_requested: cpdt_report,
                cpdt_plan_out: cpdt_plan_out.clone(),
                export_functions_out: None,
                calibration_data: calibration_data.clone(),
                calibration_mode: Some(calibrate.clone()),
                calibration_samples,
                calibration_batch_size,
                calibration_timeout_secs: calibration_timeout,
                calibration_sidecar: None,
                calibration_retention: None,
                // Task 6: peek_batch_seq is called inside the compiler when
                // calibration_data is set; the CLI passes None here and the
                // compiler resolves the real (batch, seq) from the data header.
                calibration_batch_seq: None,
                // M62 Task 6: weight_index_map is populated from analysis in
                // run_build_single/run_build_multi (where analysis is in scope).
                weight_index_map: std::collections::HashMap::new(),
                // PR #127 (AWQ v2) added this field; CLI build site populates
                // it via the calibration plumbing further down, not here.
                calibration_compile_bundle: None,
                // PR #132 (WGGO Phase 2) added this field; codegen populates
                // it from AST pre-scan inside `run_pre_scan_phase`, so the
                // CLI initializes to None and lets entry_points.rs do it.
                calibration_grad_retention: None,
            };

            // Validate WGGO mode string early so users get a clear error
            // instead of a silent no-op.
            if let Some(ref m) = wggo {
                if nsl_codegen::wggo::WggoMode::parse(m).is_none() {
                    eprintln!(
                        "error: --wggo value '{}' is not one of full|greedy|off|auto",
                        m
                    );
                    process::exit(1);
                }
            }
            // wggo_importance is now a typed CliWggoImportance enum; clap
            // rejects unknown values before we get here.  The Grad variant
            // requires a calibration sidecar — build_scorer enforces that at
            // compile time and emits the --calibration-data error message.
            if let Some(f) = wggo_prune_fraction {
                if !(0.0..=0.9).contains(&f) {
                    eprintln!(
                        "error: --wggo-prune-fraction must be in [0.0, 0.9], got {}",
                        f
                    );
                    process::exit(1);
                }
            }
            if let Some(ref p) = wggo_weights {
                if !p.exists() {
                    eprintln!(
                        "error: --wggo-weights path does not exist: {}",
                        p.display()
                    );
                    process::exit(1);
                }
            }
            // Validate CSHA mode string early.
            if let Some(ref m) = csha {
                if nsl_codegen::csha::CshaMode::parse(m).is_none() {
                    eprintln!(
                        "error: --csha value '{}' is not one of auto|boundary|pipeline|block|off",
                        m
                    );
                    process::exit(1);
                }
            }

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
                    wrga_report.as_deref(),
                );
            } else if shared_lib {
                run_build_shared(&file, output, dump_ir, &compile_opts, wrga_report.as_deref());
            } else if zk_circuit {
                run_build_zk(
                    &file,
                    output,
                    emit_obj,
                    dump_ir,
                    zk_weights.as_deref(),
                    &compile_opts,
                    wrga_report.as_deref(),
                );
            } else {
                run_build(&file, output, emit_obj, dump_ir, &compile_opts, wrga_report.as_deref());
            }

            // CPDT: post-compile rendering. Stderr diagnostics always fire
            // when CPDT ran; stdout plan only with --cpdt-report.
            if let Some(slot) = cpdt_plan_out.as_ref() {
                if let Some(plan) = slot.lock().ok().and_then(|g| g.clone()) {
                    for diag in &plan.override_diagnostics {
                        eprintln!(
                            "[cpdt] scope:global wggo-override-rejected requested={} applied={} reason={:?}",
                            diag.requested, diag.applied, diag.reason
                        );
                    }
                    if cpdt_report {
                        print!("{}", plan.render_report());
                        println!();
                        println!("=== Defaults Assumed ===");
                        println!("precision_cfg: BF16-mixed (override: --cpdt-precision, future)");
                        let jc = nsl_codegen::cpdt_joint::JointConfig::default();
                        println!("joint_cfg:     {:?} (override: --cpdt-budget, future)", jc);
                        println!("expert_cfg:    none (no MoE block detected)");
                        match &resolved_weight_file {
                            Some(p) => println!("weights:       {}", p.display()),
                            None => println!(
                                "weights:       none (no --weights flag and no AST load_safetensors)"
                            ),
                        }
                    }
                }
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
            gpu_mem_report,
            monitor,
            inspect,
            csha,
            csha_report,
            linear_types,
            cpdt,
            cpdt_num_gpus,
            cpdt_intra_bw,
            cpdt_inter_bw,
            cpdt_report,
            weights,
        } => {
            // CSHA: validate the same way `nsl build --csha` does so an
            // unrecognised mode fails fast rather than silently disabling
            // the planner.
            if let Some(ref m) = csha {
                if nsl_codegen::csha::CshaMode::parse(m).is_none() {
                    eprintln!(
                        "error: --csha value '{}' is not one of auto|boundary|pipeline|block|off",
                        m
                    );
                    process::exit(1);
                }
            }

            // CPDT: mirror the `nsl build` setup so precision-adaptive
            // training executes end-to-end via `nsl run`. The cpdt_mode,
            // cpdt_cluster and cpdt_plan_out are threaded into the compile
            // call via CompileOptions fields below (the compiler copies
            // options.cpdt_cluster into Compiler::cpdt_cluster and reads
            // compile_options.cpdt_plan_out during train-block codegen).
            //
            // --cpdt-report implies --cpdt (full mode unless explicit).
            let cpdt_mode_str: Option<String> = match (cpdt.as_deref(), cpdt_report) {
                (Some(s), _) => Some(s.to_string()),
                (None, true) => Some("full".to_string()),
                (None, false) => None,
            };
            let cpdt_mode = match cpdt_mode_str.as_deref() {
                None => nsl_codegen::cpdt::CpdtMode::Off,
                Some(s) => match nsl_codegen::cpdt::CpdtMode::parse(s) {
                    Some(m) => m,
                    None => {
                        eprintln!(
                            "error: --cpdt value '{}' is not one of full|zero_only|off",
                            s
                        );
                        process::exit(2);
                    }
                },
            };
            let cpdt_cluster = if cpdt_mode != nsl_codegen::cpdt::CpdtMode::Off {
                let n = match cpdt_num_gpus {
                    Some(n) if n >= 1 => n,
                    Some(_) => {
                        eprintln!("nsl: --cpdt-num-gpus must be >= 1");
                        process::exit(2);
                    }
                    None => {
                        eprintln!("nsl: --cpdt requires --cpdt-num-gpus N");
                        process::exit(2);
                    }
                };
                Some(nsl_codegen::cpdt_zero::ClusterSpec {
                    num_gpus: n,
                    memory_budget_bytes: 80u64 * 1024 * 1024 * 1024,
                    intra_bw_bps: cpdt_intra_bw,
                    inter_bw_bps: cpdt_inter_bw,
                    gpus_per_node: n.min(8),
                })
            } else {
                None
            };
            let cpdt_plan_out: Option<
                std::sync::Arc<std::sync::Mutex<Option<nsl_codegen::cpdt::CpdtPlan>>>,
            > = if cpdt_mode != nsl_codegen::cpdt::CpdtMode::Off {
                Some(std::sync::Arc::new(std::sync::Mutex::new(None)))
            } else {
                None
            };

            // Phase 1 CPDT: AST-scan for load_safetensors(...) + @cpdt(weight_aware=...).
            // Resolve the effective weight file via the four-case decision
            // table from the Phase 1 spec §2.1 (identical to the build path).
            // `nsl run` has no --standalone flag, so the build path's
            // `!standalone` guard is always satisfied here.
            let resolved_weight_file: Option<PathBuf> = {
                let ast_source = std::fs::read_to_string(&file).unwrap_or_default();
                let mut ast_interner = Interner::new();
                let ast_file_id = nsl_errors::FileId(0);
                let (ast_tokens, _) =
                    nsl_lexer::tokenize(&ast_source, ast_file_id, &mut ast_interner);
                let ast_parse = nsl_parser::parse(&ast_tokens, &mut ast_interner);
                let ast_weight_ref =
                    ast_scan::find_ast_weight_ref(&ast_parse.module, &ast_interner);
                let ast_weight_aware =
                    ast_scan::find_ast_cpdt_weight_aware(&ast_parse.module, &ast_interner);

                match (&weights, &ast_weight_ref) {
                    (Some(flag_path), Some(ast_path)) => {
                        eprintln!(
                            "warning: --weights {} overrides AST-declared load_safetensors({:?}).",
                            flag_path.display(),
                            ast_path.display(),
                        );
                        Some(flag_path.clone())
                    }
                    (Some(flag_path), None) => Some(flag_path.clone()),
                    (None, Some(ast_path)) => Some(ast_path.clone()),
                    (None, None) => {
                        let cpdt_enabled = cpdt_mode != nsl_codegen::cpdt::CpdtMode::Off;
                        let weight_aware = ast_weight_aware.unwrap_or(true);
                        if cpdt_enabled && weight_aware {
                            eprintln!(
                                "error: --cpdt {} requires weights. Resolve by ONE of:\n\
                                 \n\
                                 1. Add --weights <path.safetensors> to this invocation.\n\
                                 2. Add `let w = load_safetensors(\"<path>\")` to your NSL source.\n\
                                 3. Add `@cpdt(weight_aware=false)` to opt out of the weight-aware path\n\
                                    (produces a CPDT plan without weight-derived tier assignments).",
                                cpdt_mode.as_str(),
                            );
                            process::exit(1);
                        }
                        None
                    }
                }
            };

            // Dev Tools Phase 4 Task 6: when --monitor is set, auto-detect
            // whether this program has a `train { }` block.  If so, enable
            // the health monitor and skip the Phase 1/2 kernel-timing path
            // (the two are mutually exclusive per spec § 4.6).  Non-train
            // programs keep the existing kernel-timing behavior.
            let detected_train_block: bool = if monitor {
                let src = std::fs::read_to_string(&file).unwrap_or_default();
                match nsl_cli::shape_debug::ShapeDebugInput::from_source(
                    &src,
                    file.to_str().unwrap_or("<file>"),
                ) {
                    Ok(input) => has_train_block(&input.module),
                    Err(_) => false,
                }
            } else {
                false
            };

            if monitor && !detected_train_block {
                let profile_args = nsl_cli::profile::ProfileArgs {
                    file: file.clone(),
                    target: gpu.clone().unwrap_or_else(|| "h100".to_string()),
                    dtype: "bf16".into(),
                    batch: 1,
                    seq: 2048,
                    dim: vec![],
                    fusion: true,
                    memory: false,
                    entry: "auto".into(),
                    json: true,
                    explain_wggo: false,
                };
                let report_json = match nsl_cli::profile::run_profile(&profile_args) {
                    Ok(s) => s,
                    Err(e) => {
                        eprintln!("error: {e}");
                        process::exit(1);
                    }
                };
                let report: nsl_codegen::profiling::types::ProfileReport =
                    match serde_json::from_str(&report_json) {
                        Ok(r) => r,
                        Err(e) => {
                            eprintln!("error: profile JSON parse: {e}");
                            process::exit(1);
                        }
                    };
                let manifest_path =
                    match nsl_cli::monitor::write_manifest_beside(&file, &report) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("error: {e}");
                            process::exit(1);
                        }
                    };
                let actual_path = file.with_extension("nsl-profile-actual.json");
                match nsl_cli::monitor::run_monitor(&file, &manifest_path, &actual_path) {
                    Ok(rendered) => {
                        println!("{}", rendered);
                        return;
                    }
                    Err(e) => {
                        eprintln!("error: {e}");
                        process::exit(1);
                    }
                }
            }
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
                // CPDT: pass through the four-case-resolved weight file so the
                // weight-aware tier assignment runs during train-block codegen.
                weight_file: resolved_weight_file.clone(),
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
                linear_types_enabled: linear_types, // Task 20: nsl run now exposes --linear-types
                ownership_info: std::collections::HashMap::new(),
                zero_stage: zero_stage.map(|s| s as u8),
                debug_training,
                shared_lib: false,
                wrga_inputs: None,
                fused_ce_configs: Vec::new(),
                wrga_fold_allocations: false,
                wggo_mode: None,
                wggo_report: false,
                // Phase 4 Task 6: when a train block is detected with --monitor,
                // the health monitor takes over; disable the Phase 1/2 kernel
                // timing path so they don't stomp on each other.
                profile_kernels: if detected_train_block { false } else { monitor },
                target_gpu: "h100".to_string(),
                dtype: "bf16".to_string(),
                manifest_output_path: if monitor {
                    Some(file.with_extension("nsl-profile.json"))
                } else {
                    None
                },
                profile_source_text: if monitor {
                    std::fs::read_to_string(&file).ok()
                } else {
                    None
                },
                profile_source_file_name: if monitor {
                    Some(file.display().to_string())
                } else {
                    None
                },
                health_monitor: detected_train_block,
                health_flush_interval: None,
                inspect_enabled: inspect,
                wggo_weights: None,
                wggo_importance: nsl_codegen::WggoImportance::Auto,
                wggo_prune_fraction: None,
                csha_mode: csha.clone(),
                csha_report,
                // CPDT: thread the planner mode + cluster + plan-out slot into
                // codegen exactly as `nsl build` does — the compiler copies
                // cpdt_cluster into Compiler::cpdt_cluster and reads
                // cpdt_plan_out during train-block codegen via
                // invoke_cpdt_if_enabled.
                cpdt_mode,
                cpdt_cluster: cpdt_cluster.clone(),
                cpdt_report_requested: cpdt_report,
                cpdt_plan_out: cpdt_plan_out.clone(),
                export_functions_out: None,
                calibration_data: None,
                calibration_mode: Some("required".to_string()),
                calibration_samples: 512,
                calibration_batch_size: 8,
                calibration_timeout_secs: 600,
                calibration_sidecar: None,
                calibration_retention: None,
                calibration_batch_seq: None,
                // M62 Task 6: weight_index_map is populated from analysis in
                // run_build_single (where analysis is in scope).
                weight_index_map: std::collections::HashMap::new(),
                // PR #127 (AWQ v2) added this field; CLI run site doesn't
                // perform calibration so the bundle is unset.
                calibration_compile_bundle: None,
                // PR #132 (WGGO Phase 2) added this field; the run site
                // doesn't drive calibration, so this stays None.
                calibration_grad_retention: None,
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
                run_build_inner(&file, Some(binary_path.clone()), false, false, true, &compile_opts, None);

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
                    if gpu_mem_report {
                        cmd.env("NSL_GPU_MEM_REPORT", "1");
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
                run_run(&file, &args, profile_memory, profile_kernels, profile, cuda_sync, gpu_mem_report, &compile_opts);
                // Phase 4 Task 6: after the child process exits, load and
                // render the health snapshot written by the runtime flush
                // hook.  Only runs when `--monitor` + train-block detection
                // enabled the health monitor upstream.
                if monitor && detected_train_block {
                    let health_path = file.with_extension("nsl-health.json");
                    match std::fs::read_to_string(&health_path) {
                        Ok(s) => match serde_json::from_str::<
                            nsl_runtime::health::collector::HealthSnapshot,
                        >(&s)
                        {
                            Ok(snap) => {
                                let mut renderer =
                                    nsl_cli::health_monitor::HealthRenderer::new();
                                renderer.render(&snap);
                            }
                            Err(e) => eprintln!(
                                "warning: health snapshot at {} failed to parse: {}",
                                health_path.display(),
                                e
                            ),
                        },
                        Err(_) => eprintln!(
                            "warning: no health snapshot at {} — train step may not have reached first flush",
                            health_path.display()
                        ),
                    }
                }
                // Phase 5 Task 8: summarize @inspect dumps written by the
                // runtime hook to `.nsl-inspect/`.
                if inspect {
                    let dir = std::path::PathBuf::from(".nsl-inspect");
                    match std::fs::read_dir(&dir) {
                        Ok(entries) => {
                            let (mut stats_count, mut full_count) = (0usize, 0usize);
                            for entry in entries.flatten() {
                                let name = entry.file_name();
                                let name_str = name.to_string_lossy();
                                if name_str.ends_with(".stats.bin") {
                                    stats_count += 1;
                                } else if name_str.ends_with(".tensor.bin") {
                                    full_count += 1;
                                }
                            }
                            eprintln!(
                                "[inspect] Wrote {} stats records, {} full dumps to {}/",
                                stats_count,
                                full_count,
                                dir.display()
                            );
                        }
                        Err(_) => {
                            eprintln!(
                                "[inspect] No inspect output directory at {} — @inspect sites may not have fired",
                                dir.display()
                            );
                        }
                    }
                }
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
            run_tokenize(&dirs, &output, vocab_size, min_freq, &ext);
        }

        Cli::FpgaCompile { file, output_dir, fixture, test_taps, seq } => {
            if let Err(e) =
                run_fpga_compile(&file, fixture.as_ref(), output_dir.as_ref(), test_taps, seq)
            {
                eprintln!("error: {e}");
                process::exit(1);
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// M57.1 §3.6 (Task 4.1): `nsl fpga-compile` end-to-end pipeline dispatch.
//
// Pipeline stages:
//   1. Read NSL source
//   2. Lex + parse + semantic resolve (Phase 1's i8 alias lives in nsl-lexer)
//   3. AST → KIR via nsl_codegen::kernel_lower_fpga::lower
//   4. KIR → HIR via nsl_codegen::hir::KirToHirPass
//   5. Bake fixture weight/bias values into LocalParams by name
//   6. HIR → Verilog via nsl_codegen::backend_verilog::VerilogEmitter
//   7. Write <output-dir>/tiny_mlp.v
//
// The fixture sidecar (`<stem>_weights.bin`) is auto-discovered if `--fixture`
// is omitted; the optional `<stem>.toml` manifest's `sha256` is verified when
// present. Spec §6.1.
// ──────────────────────────────────────────────────────────────────────────────

/// The fixed top-level module name emitted by the v1 KIR→HIR pass.
/// Mirrors `KirToHirPass::lower(&kir, "tiny_mlp")` below.
const FPGA_V1_MODULE_NAME: &str = "tiny_mlp";

/// Sequential variant module name — distinct from the combinational `tiny_mlp`
/// so a single Verilator testbench can instantiate BOTH for the v1↔v2
/// cross-check.  Matches `Vtiny_mlp_seq.h` (clocked testbench) and the
/// `sequential_v1_mlp_skeleton` snapshot.
const FPGA_V1_SEQ_MODULE_NAME: &str = "tiny_mlp_seq";

/// Derive `<dir>/<stem>_weights.bin` from the .nsl path (sidecar convention,
/// spec §6.1). Returns `None` when the source path has no stem.
fn derive_default_fixture_path(source: &std::path::Path) -> Option<PathBuf> {
    let stem = source.file_stem()?.to_owned();
    let mut name = stem;
    name.push("_weights.bin");
    Some(source.with_file_name(name))
}

/// Derive `<dir>/<stem>.toml` from a fixture .bin path. Returns `None` when
/// the fixture path has no stem.
fn derive_toml_path(fixture: &std::path::Path) -> Option<PathBuf> {
    Some(fixture.with_extension("toml"))
}

/// Walks the fixture's blocks and writes element values into the matching
/// `W<i>` (weight) and `b<i>` (bias) `LocalParamArray`s declared by the
/// KIR→HIR pass (spec §3.2 / Task W6).
///
/// Pre-W5 (commit `94a05f8e`) the KIR→HIR pass emitted per-element scalar
/// `LocalParam`s named `W<i>_<k>_<o>` and `b<i>_<o>`; W5 collapsed those into
/// `LocalParamArray { name: "W<i>", dims: [k_dim, n_outputs], values: <flat> }`
/// and `LocalParamArray { name: "b<i>", dims: [n_outputs], values: <flat> }`
/// to feed the genvar-indexed ripple body (`SignalRef::IndexedLocalParam`).
/// The bake helper now writes directly into `LocalParamArray.values`, which is
/// row-major flat: for a 2-D W of shape `[K, N]`, `values[k * N + o]` is the
/// `[k][o]` cell (matching the fixture's on-disk layout — see
/// `nsl_test::fixture::FixtureBlock::shape_used`).
///
/// Missing `LocalParamArray`s are not silently tolerated: a mismatch between
/// the fixture's declared block shape and the HIR's declared arrays indicates
/// a divergence between the .nsl source (which the KIR→HIR pass walks) and the
/// fixture file the CLI was pointed at. We surface this as an error string so
/// the caller's `eprintln!("error: {e}")` reaches the user.
fn bake_fixture_into_localparams(
    module: &mut nsl_codegen::hir::HirModule,
    fixture: &nsl_test::fixture::FixtureFile,
) -> Result<(), String> {
    use nsl_test::fixture::{BlockDtype, BlockKind};

    // Decode block values to a uniform i128 representation up front. (The
    // `LocalParamArray.values` field is `Vec<i128>` so any v1 dtype fits.)
    let decode_values = |block: &nsl_test::fixture::FixtureBlock| -> Result<Vec<i128>, String> {
        Ok(match block.dtype {
            BlockDtype::I8 => block.as_i8().into_iter().map(|v| v as i128).collect(),
            BlockDtype::I16 => {
                return Err(format!(
                    "fixture block (layer {}) dtype i16 is not supported in v1",
                    block.layer
                ));
            }
            BlockDtype::I32 => block.as_i32().into_iter().map(|v| v as i128).collect(),
            BlockDtype::I64 => block.as_i64().into_iter().map(|v| v as i128).collect(),
        })
    };

    // Helper: locate a `LocalParamArray` by name and return its index in
    // `module.local_param_arrays_mut()`. Errors fail-loud with a structural
    // message — the KIR→HIR pass either declared it (under exactly the
    // `W<i>` / `b<i>` convention) or the .nsl source is structurally
    // incompatible with this fixture.
    let find_lpa_idx = |module: &nsl_codegen::hir::HirModule,
                        name: &str|
     -> Result<usize, String> {
        module
            .local_param_arrays()
            .iter()
            .position(|lpa| lpa.name == name)
            .ok_or_else(|| {
                format!(
                    "fixture references LocalParamArray `{name}` not declared by the \
                     KIR→HIR pass (source/.nsl and fixture/.bin out of sync — \
                     regenerate the fixture or update the .nsl source)"
                )
            })
    };

    for block in &fixture.blocks {
        match block.kind {
            BlockKind::Weight => {
                let used = block.shape_used();
                if used.len() != 2 {
                    return Err(format!(
                        "fixture weight block (layer {}) has rank {}; expected 2",
                        block.layer,
                        used.len()
                    ));
                }
                let k_dim = used[0] as usize;
                let n_outputs = used[1] as usize;
                let values = decode_values(block)?;
                if values.len() != k_dim * n_outputs {
                    return Err(format!(
                        "fixture weight block (layer {}): element count {} != K·N = {}·{} = {}",
                        block.layer,
                        values.len(),
                        k_dim,
                        n_outputs,
                        k_dim * n_outputs
                    ));
                }
                let arr_name = format!("W{}", block.layer);
                let idx = find_lpa_idx(module, &arr_name)?;
                {
                    // Shape-check against the HIR's declared dims. The
                    // KIR→HIR pass derived these from the .nsl source's
                    // `const W<i>: Tensor<[K, N], i8>` declaration; the
                    // fixture must match exactly (row-major [K, N]).
                    let lpa = &module.local_param_arrays()[idx];
                    if lpa.dims != vec![k_dim, n_outputs] {
                        return Err(format!(
                            "LocalParamArray `{}` dims mismatch: HIR declares {:?} \
                             but fixture provides [{}, {}]",
                            arr_name, lpa.dims, k_dim, n_outputs
                        ));
                    }
                }
                // Row-major [K, N]: index = k * n_outputs + o — same layout
                // as `LocalParamArray.values` (`values[r * dims[1] + c]`
                // per nodes.rs:77-79).
                module.local_param_arrays_mut()[idx].values = values;
            }
            BlockKind::Bias => {
                let used = block.shape_used();
                if used.len() != 1 {
                    return Err(format!(
                        "fixture bias block (layer {}) has rank {}; expected 1",
                        block.layer,
                        used.len()
                    ));
                }
                let n_outputs = used[0] as usize;
                let values = decode_values(block)?;
                if values.len() != n_outputs {
                    return Err(format!(
                        "fixture bias block (layer {}): element count {} != N = {}",
                        block.layer,
                        values.len(),
                        n_outputs
                    ));
                }
                let arr_name = format!("b{}", block.layer);
                let idx = find_lpa_idx(module, &arr_name)?;
                {
                    let lpa = &module.local_param_arrays()[idx];
                    if lpa.dims != vec![n_outputs] {
                        return Err(format!(
                            "LocalParamArray `{}` dims mismatch: HIR declares {:?} \
                             but fixture provides [{}]",
                            arr_name, lpa.dims, n_outputs
                        ));
                    }
                }
                module.local_param_arrays_mut()[idx].values = values;
            }
        }
    }

    Ok(())
}

/// End-to-end pipeline. Returns a flat String error for the simple
/// "eprintln!('error: {e}'); exit(1)" pattern used by the rest of the CLI.
fn run_fpga_compile(
    source_path: &std::path::Path,
    fixture_arg: Option<&PathBuf>,
    output_dir_arg: Option<&PathBuf>,
    test_taps: bool,
    seq: bool,
) -> Result<(), String> {
    // ── 1. Read NSL source ──────────────────────────────────────────────────
    let source_text = std::fs::read_to_string(source_path)
        .map_err(|e| format!("read NSL source `{}`: {e}", source_path.display()))?;

    // ── 2. Lex + parse ──────────────────────────────────────────────────────
    //
    // NOTE: semantic analysis (`nsl_semantic::analyze_*`) is INTENTIONALLY
    // skipped on the FPGA path. The v1 KIR→HIR recognizer in
    // `kernel_lower_fpga::lower` walks the AST directly and matches on
    // structural shape (`model TinyMlp:` + `relu(matmul(...) + bias)`); it
    // does not require `matmul`/`relu` to be resolvable as standard-library
    // symbols. Running the full semantic analyzer here would emit
    // "undefined variable `matmul`" errors that are spurious in the FPGA
    // dispatch context. The recognizer's own surface (any structural
    // mismatch) maps to `UnsupportedV1Shape` with a clear "expected …, found …"
    // explanation, which is the appropriate error surface for v1's
    // recognizer-only frontend.
    use nsl_errors::{Level, SourceMap};
    let mut source_map = SourceMap::new();
    let file_id = source_map.add_file(source_path.display().to_string(), source_text.clone());
    let mut interner = Interner::new();

    let (tokens, lex_errors) = nsl_lexer::tokenize(&source_text, file_id, &mut interner);
    for diag in &lex_errors {
        source_map.emit_diagnostic(diag);
    }

    let parse_result = nsl_parser::parse(&tokens, &mut interner);
    for diag in &parse_result.diagnostics {
        source_map.emit_diagnostic(diag);
    }

    let total_errors = lex_errors
        .iter()
        .chain(parse_result.diagnostics.iter())
        .filter(|d| d.level == Level::Error)
        .count();
    if total_errors > 0 {
        return Err(format!(
            "{total_errors} lex/parse error(s) in NSL source `{}` — see diagnostics above",
            source_path.display()
        ));
    }

    // ── 3. AST → KIR (Fpga target dispatch) ─────────────────────────────────
    let kir = nsl_codegen::kernel_lower_fpga::lower(&parse_result.module, &interner)
        .map_err(|e| format!("AST→KIR (FPGA target): {e}"))?;

    // ── 4. KIR → HIR ────────────────────────────────────────────────────────
    // M57.2: `seq` selects the sequential clocked FSM lowering; combinational
    // is the default (`sequential: false`).  Use the struct literal directly so
    // we don't need to touch `KirToHirPass::new`'s other callers.
    // The sequential module uses a distinct name (`tiny_mlp_seq`) so that a
    // single Verilator testbench can instantiate both variants simultaneously
    // for the v1↔v2 cross-check (Vtiny_mlp_seq.h / Vtiny_mlp.h).
    let module_name = if seq { FPGA_V1_SEQ_MODULE_NAME } else { FPGA_V1_MODULE_NAME };
    let mut hir = nsl_codegen::hir::KirToHirPass { test_taps, sequential: seq }
        .lower(&kir, module_name)
        .map_err(|e| format!("KIR→HIR: {e}"))?;

    // Stash cycle_count before hir is borrowed by bake_fixture_into_localparams
    // and emit_module (both take &/&mut HirModule); reading it here lets us print
    // after emit without keeping an extra borrow alive.
    let cycle_count = hir.cycle_count;

    // ── 5. Bake fixture into LocalParams ────────────────────────────────────
    let fixture_path: PathBuf = match fixture_arg {
        Some(p) => p.clone(),
        None => derive_default_fixture_path(source_path).ok_or_else(|| {
            format!(
                "no --fixture provided and could not derive default sidecar from `{}`",
                source_path.display()
            )
        })?,
    };
    let fixture_bytes = std::fs::read(&fixture_path)
        .map_err(|e| format!("read fixture `{}`: {e}", fixture_path.display()))?;

    // Optional manifest hash check (spec §6.1). Missing .toml is not an
    // error — the bin is still parsed; we just skip the integrity check.
    if let Some(toml_path) = derive_toml_path(&fixture_path) {
        if toml_path.exists() {
            let toml_text = std::fs::read_to_string(&toml_path).map_err(|e| {
                format!("read manifest `{}`: {e}", toml_path.display())
            })?;
            let manifest: toml::Value = toml::from_str(&toml_text).map_err(|e| {
                format!("parse manifest `{}`: {e}", toml_path.display())
            })?;
            let expected_hash = manifest
                .get("sha256")
                .and_then(|v| v.as_str())
                .ok_or_else(|| {
                    format!(
                        "manifest `{}` missing `sha256` top-level field",
                        toml_path.display()
                    )
                })?;
            nsl_test::fixture::verify_hash(&fixture_bytes, expected_hash).map_err(|e| {
                format!(
                    "fixture `{}` hash mismatch against manifest `{}`: {e}",
                    fixture_path.display(),
                    toml_path.display()
                )
            })?;
        }
    }

    let fixture_file = nsl_test::fixture::parse(&fixture_bytes)
        .map_err(|e| format!("parse fixture `{}`: {e}", fixture_path.display()))?;
    bake_fixture_into_localparams(&mut hir, &fixture_file)?;

    // ── 6. HIR → Verilog ────────────────────────────────────────────────────
    let verilog = nsl_codegen::backend_verilog::VerilogEmitter::emit_module(&hir);

    // ── 7. Write <output-dir>/<module_name>.v ───────────────────────────────
    // Combinational → tiny_mlp.v; sequential → tiny_mlp_seq.v.
    let out_dir = output_dir_arg
        .cloned()
        .unwrap_or_else(|| PathBuf::from("target/fpga"));
    std::fs::create_dir_all(&out_dir)
        .map_err(|e| format!("create output dir `{}`: {e}", out_dir.display()))?;
    let out_path = out_dir.join(format!("{module_name}.v"));
    std::fs::write(&out_path, &verilog)
        .map_err(|e| format!("write `{}`: {e}", out_path.display()))?;
    println!("Wrote {}", out_path.display());

    // M57.2: print the deterministic cycle count when --seq was requested.
    if seq {
        if let Some(cycles) = cycle_count {
            println!("total_cycles={cycles}");
        }
    }

    Ok(())
}

fn frontend(file: &PathBuf) -> (Interner, nsl_parser::ParseResult, nsl_semantic::AnalysisResult) {
    frontend_with_flags(file, false)
}

fn frontend_with_flags(
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
fn module_data_to_wrga_inputs(m: &crate::loader::ModuleData) -> nsl_codegen::WrgaInputs {
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
    apply_wrga_target_override(&mut inputs);
    inputs
}

/// CFTP §4.4 G3 (Sprint 2): bridge `@fused_lm_ce(...)` configs from
/// `AnalysisResult` into the codegen-side `FusedCeDecoratorConfig` newtype.
/// Mirrors `analysis_to_wrga_inputs` — keeps nsl-codegen free of a direct
/// nsl-semantic dependency.
fn analysis_to_fused_ce_configs(
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
fn module_data_to_fused_ce_configs(
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

fn analysis_to_wrga_inputs(a: &nsl_semantic::AnalysisResult) -> nsl_codegen::WrgaInputs {
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
    apply_wrga_target_override(&mut inputs);
    inputs
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

// ──────────────────────────────────────────────────────────────────────────────
// CEP frontend (DR-3, DR-9): `nsl build --cep-prune` / `nsl check --cep-search`.
//
// These orchestrators tie the already-built pieces together: the AST recognizer
// (`cep_extract`), the semantic decorator validators (`nsl_semantic::cep`), the
// codegen bridge (`nsl_codegen::cep`), and the delta writers. They short-circuit
// the normal build/check path only when the respective flag is passed.
// ──────────────────────────────────────────────────────────────────────────────

/// Locate the `@cep_prune` / `@cep_search` decorator attached to the model
/// definition. Returns the first matching decorator in source order.
fn find_cep_decorator<'a>(
    module: &'a nsl_ast::Module,
    interner: &nsl_lexer::Interner,
    want: &str,
) -> Option<&'a nsl_ast::decl::Decorator> {
    use nsl_ast::stmt::StmtKind;
    for stmt in &module.stmts {
        if let StmtKind::Decorated { decorators, stmt: inner } = &stmt.kind {
            if matches!(inner.kind, StmtKind::ModelDef(_)) {
                for deco in decorators {
                    if deco.name.len() == 1
                        && interner.resolve(deco.name[0].0).unwrap_or("") == want
                    {
                        return Some(deco);
                    }
                }
            }
        }
    }
    None
}

/// Resolve the effective CEP target string. CLI `--cep-target` wins; otherwise
/// the decorator's `target = ...` ident; otherwise the default "H100-SXM".
fn resolve_cep_target(
    cli: Option<&str>,
    deco_target: Option<nsl_ast::Symbol>,
    resolve: &dyn Fn(nsl_ast::Symbol) -> String,
) -> String {
    if let Some(t) = cli {
        return t.to_string();
    }
    if let Some(sym) = deco_target {
        let s = resolve(sym);
        if !s.is_empty() {
            return s;
        }
    }
    "H100-SXM".to_string()
}

/// Default delta output path: `<model>.cep.json` next to the source file.
fn default_cep_out(file: &std::path::Path) -> PathBuf {
    file.with_extension("cep.json")
}

/// Emit CEP diagnostics with source context (same mechanism `run_check` uses:
/// `SourceMap::add_file` + `emit_diagnostic`). Returns `true` if any diagnostic
/// is at `Level::Error`.
fn emit_cep_diags(file: &std::path::Path, diags: &[nsl_errors::Diagnostic]) -> bool {
    let source = std::fs::read_to_string(file).unwrap_or_default();
    let mut source_map = SourceMap::new();
    source_map.add_file(file.display().to_string(), source);
    for diag in diags {
        source_map.emit_diagnostic(diag);
    }
    diags.iter().any(|d| d.level == Level::Error)
}

fn run_cep_prune(
    file: &PathBuf,
    weights: Option<&std::path::Path>,
    ov: &nsl_codegen::cep::CliOverrides,
) -> i32 {
    use nsl_codegen::cep_extract::{cross_check_dims, extract_model_spec};

    let Some(weights_path) = weights else {
        eprintln!(
            "error: --cep-prune requires --weights <file.safetensors> \
             (a prune from synthetic scores would be silently wrong)"
        );
        return 1;
    };

    let (interner, parse_result, analysis) = frontend_with_flags(file, false);
    // Analysis-error gate — same accessor as run_check: filter
    // `analysis.diagnostics` on `Level::Error`. `frontend_with_flags` already
    // exits on errors, but we keep a defensive gate that prints with source
    // context before bailing.
    let analysis_errors: Vec<_> = analysis
        .diagnostics
        .iter()
        .filter(|d| d.level == Level::Error)
        .cloned()
        .collect();
    if !analysis_errors.is_empty() {
        emit_cep_diags(file, &analysis_errors);
        return 1;
    }

    let module = &parse_result.module;
    let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();

    let Some(deco) = find_cep_decorator(module, &interner, "cep_prune") else {
        eprintln!("error: --cep-prune requires a @cep_prune(...) decorator on the model");
        return 1;
    };
    let mut diags = Vec::new();
    let Some(cfg) = nsl_semantic::cep::validate_cep_prune_decorator(deco, &resolve, &mut diags)
    else {
        emit_cep_diags(file, &diags);
        return 1;
    };
    // The validator may return `Some` while still pushing error diagnostics for
    // recoverable problems; treat any pushed error as fatal.
    if emit_cep_diags(file, &diags) {
        return 1;
    }

    let spec = match extract_model_spec(module, &resolve) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };

    let wm = match nsl_codegen::weight_aware::WeightMap::load(weights_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error: failed to load weights: {e}");
            return 1;
        }
    };
    if let Err(e) = cross_check_dims(&spec, &wm, &resolve) {
        eprintln!("{e}");
        return 1;
    }

    let target = resolve_cep_target(ov.target.as_deref(), cfg.target, &resolve);
    let input =
        match nsl_codegen::cep::build_prune_input(&cfg, spec.clone(), Some(&wm), &target, ov.sparsity) {
            Ok(i) => i,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        };
    let plan = nsl_codegen::cep::run_prune(input);
    println!("{}", plan.render_report());

    let out_path = ov.cep_out.clone().unwrap_or_else(|| default_cep_out(file));
    if let Err(e) = nsl_codegen::cep::write_prune_delta(&plan, &spec, &out_path) {
        eprintln!("error: failed to write delta: {e}");
        return 1;
    }
    println!("CEP delta written to {}", out_path.display());

    if let Some(weights_out) = ov.cep_emit_weights.as_ref() {
        let delta = nsl_codegen::cep::plan_to_prune_delta(&plan, &spec);
        match nsl_codegen::cep_slice::apply_prune_delta_to_weights(&wm, &spec, &delta) {
            Ok(sliced) => {
                let orig_params: usize = wm.entries().map(|(_, e)| e.num_elements).sum();
                let new_params: usize = sliced.values().map(|e| e.num_elements).sum();
                if let Err(e) = nsl_codegen::cep_slice::write_sliced_weights(&sliced, weights_out) {
                    eprintln!("error: failed to write sliced weights: {e}");
                    return 1;
                }
                println!(
                    "CEP sliced weights written to {} ({orig_params} -> {new_params} params)",
                    weights_out.display()
                );
            }
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        }
    }

    // SP2: rewrite the source with chosen pruned dims and write to `cep_emit_source`.
    if let Some(source_out) = ov.cep_emit_source.as_ref() {
        let delta = nsl_codegen::cep::plan_to_prune_delta(&plan, &spec);
        let original_source = match std::fs::read_to_string(file) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("error: failed to re-read source for SP2 emission: {e}");
                return 1;
            }
        };
        match nsl_codegen::cep_emit_source::apply_prune_delta_to_source(
            &original_source,
            module,
            &resolve,
            &spec,
            &delta,
        ) {
            Ok(out_src) => {
                if let Err(e) = std::fs::write(source_out, &out_src) {
                    eprintln!("error: failed to write rewritten source: {e}");
                    return 1;
                }
                println!("CEP rewritten source written to {}", source_out.display());
            }
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        }
    }
    0
}

/// CEP Mode 3 (paper §2.2) — joint prune-search. Same setup as `run_cep_prune` (reuses
/// the @cep_prune decorator for v1 config; weights required for non-synthetic
/// importance), but calls `nsl_codegen::cep::run_joint` to extend the action space with
/// layer drops on top of head + FFN pruning.
fn run_cep_joint(
    file: &PathBuf,
    weights: Option<&std::path::Path>,
    ov: &nsl_codegen::cep::CliOverrides,
) -> i32 {
    use nsl_codegen::cep_extract::{cross_check_dims, extract_model_spec};

    let Some(weights_path) = weights else {
        eprintln!(
            "error: --cep-joint requires --weights <file.safetensors> \
             (joint search without weight-derived importance would be silently wrong)"
        );
        return 1;
    };

    let (interner, parse_result, analysis) = frontend_with_flags(file, false);
    let analysis_errors: Vec<_> = analysis
        .diagnostics
        .iter()
        .filter(|d| d.level == Level::Error)
        .cloned()
        .collect();
    if !analysis_errors.is_empty() {
        emit_cep_diags(file, &analysis_errors);
        return 1;
    }

    let module = &parse_result.module;
    let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();

    let Some(deco) = find_cep_decorator(module, &interner, "cep_prune") else {
        eprintln!("error: --cep-joint requires a @cep_prune(...) decorator on the model");
        return 1;
    };
    let mut diags = Vec::new();
    let Some(cfg) = nsl_semantic::cep::validate_cep_prune_decorator(deco, &resolve, &mut diags)
    else {
        emit_cep_diags(file, &diags);
        return 1;
    };
    if emit_cep_diags(file, &diags) {
        return 1;
    }

    let spec = match extract_model_spec(module, &resolve) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };

    let wm = match nsl_codegen::weight_aware::WeightMap::load(weights_path) {
        Ok(w) => w,
        Err(e) => {
            eprintln!("error: failed to load weights: {e}");
            return 1;
        }
    };
    if let Err(e) = cross_check_dims(&spec, &wm, &resolve) {
        eprintln!("{e}");
        return 1;
    }

    let target = resolve_cep_target(ov.target.as_deref(), cfg.target, &resolve);
    let input = match nsl_codegen::cep::build_prune_input(
        &cfg,
        spec.clone(),
        Some(&wm),
        &target,
        ov.sparsity,
    ) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let plan = nsl_codegen::cep::run_joint(input);
    println!("{}", plan.render_report());

    let out_path = ov.cep_out.clone().unwrap_or_else(|| default_cep_out(file));
    if let Err(e) = nsl_codegen::cep::write_prune_delta(&plan, &spec, &out_path) {
        eprintln!("error: failed to write delta: {e}");
        return 1;
    }
    println!("CEP joint delta written to {}", out_path.display());

    if let Some(weights_out) = ov.cep_emit_weights.as_ref() {
        let delta = nsl_codegen::cep::plan_to_prune_delta(&plan, &spec);
        match nsl_codegen::cep_slice::apply_prune_delta_to_weights(&wm, &spec, &delta) {
            Ok(sliced) => {
                let orig_params: usize = wm.entries().map(|(_, e)| e.num_elements).sum();
                let new_params: usize = sliced.values().map(|e| e.num_elements).sum();
                if let Err(e) = nsl_codegen::cep_slice::write_sliced_weights(&sliced, weights_out)
                {
                    eprintln!("error: failed to write sliced weights: {e}");
                    return 1;
                }
                println!(
                    "CEP sliced weights written to {} ({orig_params} -> {new_params} params)",
                    weights_out.display()
                );
            }
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        }
    }
    0
}

fn run_cep_search(file: &PathBuf, ov: &nsl_codegen::cep::CliOverrides) -> i32 {
    use nsl_codegen::cep_extract::extract_search_axes;

    let (interner, parse_result, analysis) = frontend_with_flags(file, false);
    let analysis_errors: Vec<_> = analysis
        .diagnostics
        .iter()
        .filter(|d| d.level == Level::Error)
        .cloned()
        .collect();
    if !analysis_errors.is_empty() {
        emit_cep_diags(file, &analysis_errors);
        return 1;
    }

    let module = &parse_result.module;
    let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();

    let Some(deco) = find_cep_decorator(module, &interner, "cep_search") else {
        eprintln!("error: --cep-search requires a @cep_search(...) decorator on the model");
        return 1;
    };
    let mut diags = Vec::new();
    let Some(cfg) = nsl_semantic::cep::validate_cep_search_decorator(deco, &resolve, &mut diags)
    else {
        emit_cep_diags(file, &diags);
        return 1;
    };
    if emit_cep_diags(file, &diags) {
        return 1;
    }

    let axes = match extract_search_axes(module, &resolve) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let target = resolve_cep_target(ov.target.as_deref(), cfg.target, &resolve);
    let input = match nsl_codegen::cep::build_search_input(&cfg, axes, &target) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };
    let plan = nsl_codegen::cep::run_search(input);
    println!("{}", plan.render_report());

    let out_path = ov.cep_out.clone().unwrap_or_else(|| default_cep_out(file));
    if let Err(e) = nsl_codegen::cep::write_search_delta(&plan, &out_path) {
        eprintln!("error: failed to write delta: {e}");
        return 1;
    }
    println!("CEP delta written to {}", out_path.display());
    0
}

/// Run a bare compilation profile for a model (paper §7.1 'nsl check --cep-profile').
/// Prints a one-shot CompilationProfile without modification.
fn run_cep_profile(
    file: &PathBuf,
    weights: Option<&std::path::Path>,
    ov: &nsl_codegen::cep::CliOverrides,
) -> i32 {
    use nsl_codegen::cep_extract::{cross_check_dims, extract_model_spec};

    let (interner, parse_result, analysis) = frontend_with_flags(file, false);
    let analysis_errors: Vec<_> = analysis
        .diagnostics
        .iter()
        .filter(|d| d.level == Level::Error)
        .cloned()
        .collect();
    if !analysis_errors.is_empty() {
        emit_cep_diags(file, &analysis_errors);
        return 1;
    }

    let module = &parse_result.module;
    let resolve = |s: nsl_ast::Symbol| interner.resolve(s.0).unwrap_or("").to_string();

    let spec = match extract_model_spec(module, &resolve) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{e}");
            return 1;
        }
    };

    // Optional weight cross-check (same logic as run_cep_prune).
    if let Some(weights_path) = weights {
        let wm = match nsl_codegen::weight_aware::WeightMap::load(weights_path) {
            Ok(w) => w,
            Err(e) => {
                eprintln!("error: failed to load weights: {e}");
                return 1;
            }
        };
        if let Err(e) = cross_check_dims(&spec, &wm, &resolve) {
            eprintln!("{e}");
            return 1;
        }
    }

    let target = resolve_cep_target(ov.target.as_deref(), None, &resolve);
    let gpu = match nsl_codegen::gpu_specs::find_gpu(&target) {
        Some(g) => g,
        None => {
            eprintln!(
                "error: unknown CEP target '{}'. Supported: {}",
                target,
                nsl_codegen::cep::supported_gpus_list()
            );
            return 1;
        }
    };
    let profile = match nsl_codegen::cep_oracle::evaluate(&spec, gpu) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("error: CEP oracle failed: {e:?}");
            return 1;
        }
    };

    // §6.3-style one-shot profile output.
    println!("=== CEP Compilation Profile ===");
    println!("Target: {}", gpu.name);
    println!("Params: {:.1}M", spec.param_count() as f64 / 1e6);
    println!("Binary size: {}", cep_format_bytes_si(profile.binary_size_bytes));
    println!("Peak memory: {:.1}GB", profile.peak_memory_bytes as f64 / 1e9);
    println!("Estimated latency: {:.1}us/token", profile.estimated_latency_us);
    println!("WCET (roofline upper bound): {:.1}us/token", profile.wcet_us);
    println!("Kernel launches per forward: {}", profile.kernel_launches);
    println!("Total FLOPs: {}", cep_format_flops(profile.total_flops));
    println!("Total HBM bytes: {:.1}GB", profile.total_hbm_bytes as f64 / 1e9);
    println!("Roofline utilization: {:.2}", profile.roofline_utilization);
    println!("Fusion opportunities: {}", profile.fusion_events.len());
    0
}

fn cep_format_bytes_si(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{}MB", bytes / 1_000_000)
    } else if bytes >= 1_000 {
        format!("{}KB", bytes / 1_000)
    } else {
        format!("{}B", bytes)
    }
}

fn cep_format_flops(flops: u64) -> String {
    if flops >= 1_000_000_000_000 {
        format!("{:.1}T", flops as f64 / 1e12)
    } else if flops >= 1_000_000_000 {
        format!("{:.1}G", flops as f64 / 1e9)
    } else if flops >= 1_000_000 {
        format!("{:.1}M", flops as f64 / 1e6)
    } else if flops >= 1_000 {
        format!("{:.1}K", flops as f64 / 1e3)
    } else {
        format!("{}", flops)
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

fn run_build(
    file: &PathBuf,
    output: Option<PathBuf>,
    emit_obj: bool,
    dump_ir: bool,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    run_build_inner(file, output, emit_obj, dump_ir, false, options, wrga_report);
}

/// Task 3 (B.1): `--wrga-report` requires `--source-ad`, since WRGA only fires
/// in the source-AD lowering path. Detect and fail fast rather than silently
/// producing an empty report.
fn check_wrga_report_preconditions(
    analysis: &nsl_semantic::AnalysisResult,
    wrga_report: Option<&std::path::Path>,
    options: &nsl_codegen::CompileOptions,
) {
    if wrga_report.is_none() || options.source_ad {
        return;
    }
    let has_wrga_decorators = !analysis.wrga_configs.is_empty()
        || !analysis.freeze_configs.is_empty()
        || !analysis.adapter_configs.is_empty();
    if has_wrga_decorators {
        eprintln!(
            "nsl: --wrga-report requires --source-ad when WRGA decorators are present; re-run with --source-ad"
        );
        process::exit(2);
    }
}

/// WRGA paper §8.3: `nsl check --wrga-analyze` — run the WRGA pass on the
/// source and emit `WrgaPlan::render_report()` without leaving a `.o` behind.
///
/// Reuses the existing `run_build_inner` to do all the heavy lifting (multi-
/// file resolution, source-AD lowering, codegen, WRGA bridge) because that
/// path is the only one that produces a `WrgaPlan` today, and reimplementing
/// it would duplicate ~200 lines of multi-file orchestration. The build is
/// redirected at a per-process temp directory; the directory is deleted after
/// the report has been written, so `nsl check` remains side-effect-free from
/// the user's perspective.
///
/// `wrga_target` (optional) overrides the `target=` field on every
/// `@wrga(...)` decorator before codegen — so `--wrga-target h100` from the
/// CLI wins over any source-level `@wrga(target="a100")`. The override is
/// installed by mutating the entry module's `wrga_configs` in
/// `frontend_with_flags`'s output via a CLI-side passthrough; we do this by
/// post-processing `WrgaInputs` inside `run_build_inner` — see
/// `apply_wrga_target_override` below.
///
/// Returns the process exit code: `0` on success, `2` on "no WRGA decorators
/// in source" (so CI can distinguish absence from compile failure), `1` on
/// any other error.
fn run_check_wrga_analyze(
    file: &PathBuf,
    report_path: &std::path::Path,
    wrga_target: Option<&str>,
) -> i32 {
    // Pre-check: surface "no decorators" as exit 2 BEFORE running codegen.
    // The build path would silently report "no plan" with exit 0, which the
    // paper's `--wrga-analyze` contract treats as a distinct error class.
    let (_interner, _parse_result, analysis) = frontend_with_flags(file, false);
    let has_wrga_decorators = !analysis.wrga_configs.is_empty()
        || !analysis.freeze_configs.is_empty()
        || !analysis.adapter_configs.is_empty();
    if !has_wrga_decorators {
        eprintln!(
            "nsl: --wrga-analyze: no @wrga / @freeze / @adapter decorators found in '{}'",
            file.display()
        );
        return 2;
    }

    // Redirect the build at a temp dir we own. Pre-create it so the linker has
    // a real path to drop the .o into, and so cleanup at the end is bounded.
    let stem = file.file_stem().and_then(|s| s.to_str()).unwrap_or("nsl_check");
    let temp_dir = std::env::temp_dir().join(format!(
        "nsl_check_wrga_analyze_{}_{}",
        std::process::id(),
        stem
    ));
    if let Err(e) = std::fs::create_dir_all(&temp_dir) {
        eprintln!("nsl: --wrga-analyze: could not create temp dir: {e}");
        return 1;
    }
    let temp_obj = temp_dir.join(format!("{stem}.o"));

    // The WRGA target override travels via a thread-local that the build path
    // checks just before constructing `WrgaInputs`. This avoids threading a
    // new param through `run_build_inner`'s six callers. The RAII guard
    // guarantees the cell is cleared even if `run_build_inner` panics, so the
    // override cannot leak into any in-process follow-on CLI invocation.
    let _override_guard = wrga_target.map(|t| WrgaTargetOverrideGuard::set(t.to_string()));

    let opts = nsl_codegen::CompileOptions {
        source_ad: true,
        ..nsl_codegen::CompileOptions::default()
    };

    // We pass our temp_obj as `output`. emit_obj=true so the linker is
    // skipped. The single-file build path ignores `output` and drops the .o
    // next to the source file (see `run_build_single` line ~4479); the
    // multi-file build path uses its own internal temp dir. To make `nsl
    // check` side-effect-free across BOTH dispatches, we explicitly clean
    // the source-adjacent .o path too, after the build returns.
    let source_adjacent_obj = file.with_file_name(format!("{stem}.o"));
    let source_adjacent_pre_existed = source_adjacent_obj.exists();

    run_build_inner(
        file,
        Some(temp_obj.clone()),
        true,   // emit_obj
        false,  // dump_ir
        true,   // quiet
        &opts,
        Some(report_path),
    );

    // Cleanup: our owned temp dir, plus any source-adjacent .o that the build
    // path dropped. Never delete a source-adjacent .o that ALREADY existed
    // before the call — that would be a user-data loss bug.
    let _ = std::fs::remove_dir_all(&temp_dir);
    if !source_adjacent_pre_existed && source_adjacent_obj.exists() {
        let _ = std::fs::remove_file(&source_adjacent_obj);
    }
    0
}

/// RAII guard that clears `WRGA_TARGET_OVERRIDE` on drop. Holding this guard
/// keeps the thread-local set for the lifetime of the build; dropping it
/// (including on panic) restores `None`.
struct WrgaTargetOverrideGuard;

impl WrgaTargetOverrideGuard {
    fn set(value: String) -> Self {
        WRGA_TARGET_OVERRIDE.with(|c| *c.borrow_mut() = Some(value));
        Self
    }
}

impl Drop for WrgaTargetOverrideGuard {
    fn drop(&mut self) {
        WRGA_TARGET_OVERRIDE.with(|c| *c.borrow_mut() = None);
    }
}

thread_local! {
    /// CLI-side override for `WrgaInputs::wrga[*].target`. Set by
    /// `run_check_wrga_analyze` before invoking the build pipeline; read by
    /// `analysis_to_wrga_inputs` / `module_data_to_wrga_inputs` to patch each
    /// decorator's target field before the bridge ships to codegen.
    static WRGA_TARGET_OVERRIDE: std::cell::RefCell<Option<String>>
        = const { std::cell::RefCell::new(None) };
}

/// Apply the thread-local `WRGA_TARGET_OVERRIDE` (set by
/// `run_check_wrga_analyze`) to a freshly-built `WrgaInputs`. No-op when no
/// override is active. When the source has no `@wrga(...)` decorator at all
/// (only `@freeze` / `@adapter`), a minimal Auto-mode config is inserted so
/// the bridge surfaces the user's target choice.
fn apply_wrga_target_override(inputs: &mut nsl_codegen::WrgaInputs) {
    WRGA_TARGET_OVERRIDE.with(|c| {
        let Some(target) = c.borrow().clone() else { return };
        for cfg in &mut inputs.wrga {
            cfg.target = Some(target.clone());
        }
        if inputs.wrga.is_empty() {
            inputs.wrga.push(nsl_codegen::WrgaDecoratorConfig {
                mode: nsl_ast::block::WrgaMode::Auto,
                budget: None,
                target: Some(target),
                layers: Vec::new(),
            });
        }
    });
}

/// Task 3 (B.1): emit the WRGA report to stdout or a file. Mirrors the logic in
/// `run_build_single` / `run_build_multi`.
fn emit_wrga_report(
    wrga_plan: &Option<nsl_codegen::wrga::WrgaPlan>,
    wrga_report: Option<&std::path::Path>,
) {
    let Some(report_path) = wrga_report else { return; };
    match wrga_plan {
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

/// M62a: Build as a shared library (.so/.dylib/.dll) with stable C API.
fn run_build_shared(
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
    let (interner, parse_result, analysis) = frontend_with_flags(file, options.linear_types_enabled);

    // Task 3 (B.1): forward WRGA decorator configs so they take effect on the
    // shared-library build path, and fail fast if --wrga-report is combined
    // with decorators but without --source-ad.
    check_wrga_report_preconditions(&analysis, wrga_report, options);
    let mut options = options.clone();
    options.wrga_inputs = Some(analysis_to_wrga_inputs(&analysis));
    options.fused_ce_configs = analysis_to_fused_ce_configs(&analysis);
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

    emit_wrga_report(&wrga_plan, wrga_report);

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
            entry_options.wrga_inputs = Some(module_data_to_wrga_inputs(mod_data));
            entry_options.fused_ce_configs = module_data_to_fused_ce_configs(mod_data);
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

    emit_wrga_report(&entry_wrga_plan, wrga_report);

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

/// M55: Build with --zk-circuit. Runs normal compilation and then invokes
/// `zk::compile_zk()` for each @zk_proof-decorated function found.
fn run_build_zk(
    file: &PathBuf,
    output: Option<PathBuf>,
    emit_obj: bool,
    dump_ir: bool,
    zk_weights: Option<&std::path::Path>,
    options: &nsl_codegen::CompileOptions,
    wrga_report: Option<&std::path::Path>,
) {
    let (interner, parse_result, analysis) = frontend_with_flags(file, options.linear_types_enabled);

    // Task 3 (B.1): forward WRGA decorator configs so `@freeze`/`@adapter`/`@wrga`
    // take effect in codegen on the ZK build path.
    // Task 4 (B.2): if `--wrga-report` is set, fail fast when decorators are
    // present without `--source-ad` (mirroring the single/multi build paths).
    check_wrga_report_preconditions(&analysis, wrga_report, options);
    let mut options = options.clone();
    options.wrga_inputs = Some(analysis_to_wrga_inputs(&analysis));
    options.fused_ce_configs = analysis_to_fused_ce_configs(&analysis);
    // M62 Task 6: route weight_index_map from semantic analysis into codegen.
    options.weight_index_map = analysis.weight_index_map.clone();
    let options = &options;

    // Task 4 (B.2): use the `_returning_plan` variant so the WRGA plan is
    // observable (and reportable) even if later codegen fails.
    let (bytes_res, zk_proof_fns, zk_results, wrga_plan) =
        nsl_codegen::compile_with_zk_info_returning_plan(
            &parse_result.module,
            &interner,
            &analysis.type_map,
            dump_ir,
            options,
        );

    // Emit the WRGA report (if requested) before reporting any codegen error.
    emit_wrga_report(&wrga_plan, wrga_report);

    let obj_bytes = match bytes_res {
        Ok(bytes) => bytes,
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
    wrga_report: Option<&std::path::Path>,
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
    let (interner, parse_result, analysis) = frontend_with_flags(&file_pb, options.linear_types_enabled);

    // Task 3 (B.1): forward WRGA decorator configs so `@freeze`/`@adapter`/`@wrga`
    // take effect in codegen on the standalone build path.
    // Task 4 (B.2): if `--wrga-report` is set, fail fast when decorators are
    // present without `--source-ad`.
    check_wrga_report_preconditions(&analysis, wrga_report, options);
    let mut options = options.clone();
    options.wrga_inputs = Some(analysis_to_wrga_inputs(&analysis));
    options.fused_ce_configs = analysis_to_fused_ce_configs(&analysis);
    // M62 Task 6: route weight_index_map from semantic analysis into codegen.
    options.weight_index_map = analysis.weight_index_map.clone();
    let options = &options;

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

    // 6. Compile with standalone config.
    // Task 4 (B.2): use the `_returning_plan` variant so the WRGA plan is
    // observable (and reportable) even if later codegen fails.
    let (bytes_res, wrga_plan) = nsl_codegen::compile_standalone_returning_plan(
        &parse_result.module,
        &interner,
        &analysis.type_map,
        config,
        false,
        options,
    );
    emit_wrga_report(&wrga_plan, wrga_report);
    let obj_bytes = bytes_res.unwrap_or_else(|e| {
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

fn run_build_inner(
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
    let (interner, parse_result, analysis) = frontend_with_flags(file, options.linear_types_enabled);

    // Task 3 (B.1): fail fast if --wrga-report is used without --source-ad
    // when decorators are present — the old silent-notice behaviour was
    // confusing.
    check_wrga_report_preconditions(&analysis, wrga_report, options);

    // Task 1 (WRGA bridge): forward decorator configs captured by nsl-semantic.
    let mut options = options.clone();
    options.wrga_inputs = Some(analysis_to_wrga_inputs(&analysis));
    options.fused_ce_configs = analysis_to_fused_ce_configs(&analysis);
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
            entry_options.wrga_inputs = Some(module_data_to_wrga_inputs(mod_data));
            entry_options.fused_ce_configs = module_data_to_fused_ce_configs(mod_data);
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

#[allow(clippy::too_many_arguments)] // CLI dispatcher, not a library API
fn run_run(file: &PathBuf, program_args: &[String], profile_memory: bool, profile_kernels: bool, profile: bool, cuda_sync: bool, gpu_mem_report: bool, options: &nsl_codegen::CompileOptions) {
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
    run_build_inner(file, Some(exe_path.clone()), false, false, true, options, None);

    // CPDT: post-compile rendering, mirroring the `nsl build` path. Stderr
    // diagnostics always fire when CPDT ran; the stdout plan only with
    // --cpdt-report. The plan slot was populated during the compile above.
    if let Some(slot) = options.cpdt_plan_out.as_ref() {
        if let Some(plan) = slot.lock().ok().and_then(|g| g.clone()) {
            for diag in &plan.override_diagnostics {
                eprintln!(
                    "[cpdt] scope:global wggo-override-rejected requested={} applied={} reason={:?}",
                    diag.requested, diag.applied, diag.reason
                );
            }
            if options.cpdt_report_requested {
                print!("{}", plan.render_report());
                println!();
                println!("=== Defaults Assumed ===");
                println!("precision_cfg: BF16-mixed (override: --cpdt-precision, future)");
                let jc = nsl_codegen::cpdt_joint::JointConfig::default();
                println!("joint_cfg:     {:?} (override: --cpdt-budget, future)", jc);
                println!("expert_cfg:    none (no MoE block detected)");
                match &options.weight_file {
                    Some(p) => println!("weights:       {}", p.display()),
                    None => println!(
                        "weights:       none (no --weights flag and no AST load_safetensors)"
                    ),
                }
            }
        }
    }

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
    if gpu_mem_report {
        // ELTLS: instructs the runtime (via atexit hook in nsl_args_init)
        // to print the GPU memory report after the compiled main returns.
        cmd.env("NSL_GPU_MEM_REPORT", "1");
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
            run_run(file, &[], false, false, false, false, false, &nsl_codegen::CompileOptions::default());
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

fn run_tokenize(dirs: &[String], output: &std::path::Path, vocab_size: usize, min_freq: u64, ext: &str) {
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
    if let Ok(encoding) = tokenizer.encode(sample, false) {
        let tokens = encoding.get_tokens();
        eprintln!("[tokenize] Sample: \"{}\" -> {} tokens: {:?}", sample, tokens.len(), &tokens[..tokens.len().min(10)]);
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
