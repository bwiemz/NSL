mod ast_scan;
mod commands;
mod debug;
mod formatter;
mod loader;
mod mangling;
mod resolver;
mod standalone;

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

        /// WRGA paper §8.3: emit a structural PEFT comparison report (WRGA vs.
        /// LoRA / AdaLoRA / GaLore / ReFT) without running codegen output.
        /// Pass without value (or `-`) for stdout; provide a path to write to
        /// a file. Mutually exclusive with `--wrga-analyze`. Requires `@wrga`
        /// decorators in the source.
        #[arg(long, num_args = 0..=1, require_equals = true, default_missing_value = "-")]
        wrga_compare: Option<PathBuf>,
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

        /// Number of devices in the target cluster (compile-time
        /// `world_size`).  Drives WGGO's ZeRO sharding budget and the
        /// tensor-parallel `world_size` baked into the artifact.  Unlike the
        /// `run` command's `--devices`, this does not spawn processes — it
        /// only informs the compiler's global-optimization plan.
        #[arg(long, default_value_t = 1)]
        devices: u32,

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
pub(crate) enum ZkCmd {
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
            wrga_compare,
        } => {
            if cep_search && cep_profile {
                eprintln!("error: --cep-search and --cep-profile are mutually exclusive");
                std::process::exit(1);
            }
            if wrga_analyze.is_some() && wrga_compare.is_some() {
                eprintln!("error: --wrga-analyze and --wrga-compare are mutually exclusive");
                std::process::exit(1);
            }
            // WRGA paper §8.3: `--wrga-analyze` short-circuits the check path
            // to emit just the WRGA compilation report (no .o, no shape trace,
            // no NaN analysis side effects). Mutually exclusive with CEP modes
            // since CEP early-exits above.
            if let Some(ref report_path) = wrga_analyze {
                std::process::exit(commands::build::run_check_wrga_analyze(
                    &file,
                    report_path,
                    wrga_target.as_deref(),
                ));
            }
            // WRGA paper §8.3: `--wrga-compare` — same dispatch shape as
            // `--wrga-analyze`, but renders the PEFT comparison report.
            if let Some(ref report_path) = wrga_compare {
                std::process::exit(commands::build::run_check_wrga_compare(
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
                std::process::exit(commands::cep::run_cep_search(&file, &ov));
            }
            if cep_profile {
                let ov = nsl_codegen::cep::CliOverrides {
                    target: cep_target,
                    sparsity: None,
                    cep_out: None,
                    cep_emit_weights: None,
                    cep_emit_source: None,
                };
                std::process::exit(commands::cep::run_cep_profile(&file, weights.as_deref(), &ov));
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
            commands::check::run_check(&file, dump_tokens, dump_ast, dump_types, linear_types);
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
            devices,
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
                std::process::exit(commands::cep::run_cep_prune(&file, weights.as_deref(), &ov));
            }

            if cep_joint {
                let ov = nsl_codegen::cep::CliOverrides {
                    target: cep_target,
                    sparsity: cep_sparsity,
                    cep_out,
                    cep_emit_weights,
                    cep_emit_source,
                };
                std::process::exit(commands::cep::run_cep_joint(&file, weights.as_deref(), &ov));
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
                world_size: devices.max(1) as usize, // --devices drives WGGO ZeRO + TP world_size
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
                commands::build::run_build_standalone(
                    &file,
                    output.as_deref(),
                    weights.as_deref().unwrap(),
                    embed_mode,
                    embed_threshold,
                    &compile_opts,
                    wrga_report.as_deref(),
                );
            } else if shared_lib {
                commands::build::run_build_shared(&file, output, dump_ir, &compile_opts, wrga_report.as_deref());
            } else if zk_circuit {
                commands::build::run_build_zk(
                    &file,
                    output,
                    emit_obj,
                    dump_ir,
                    zk_weights.as_deref(),
                    &compile_opts,
                    wrga_report.as_deref(),
                );
            } else {
                commands::build::run_build(&file, output, emit_obj, dump_ir, &compile_opts, wrga_report.as_deref());
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
                commands::build::run_build_inner(&file, Some(binary_path.clone()), false, false, true, &compile_opts, None);

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
                commands::build::run_run(&file, &args, profile_memory, profile_kernels, profile, cuda_sync, gpu_mem_report, &compile_opts);
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
    }
}

pub(crate) fn frontend(file: &PathBuf) -> (Interner, nsl_parser::ParseResult, nsl_semantic::AnalysisResult) {
    frontend_with_flags(file, false)
}

pub(crate) fn frontend_with_flags(
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
pub(crate) fn module_data_to_wrga_inputs(m: &crate::loader::ModuleData) -> nsl_codegen::WrgaInputs {
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
    commands::build::apply_wrga_target_override(&mut inputs);
    inputs
}

/// CFTP §4.4 G3 (Sprint 2): bridge `@fused_lm_ce(...)` configs from
/// `AnalysisResult` into the codegen-side `FusedCeDecoratorConfig` newtype.
/// Mirrors `analysis_to_wrga_inputs` — keeps nsl-codegen free of a direct
/// nsl-semantic dependency.
pub(crate) fn analysis_to_fused_ce_configs(
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
pub(crate) fn module_data_to_fused_ce_configs(
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

pub(crate) fn analysis_to_wrga_inputs(a: &nsl_semantic::AnalysisResult) -> nsl_codegen::WrgaInputs {
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
    commands::build::apply_wrga_target_override(&mut inputs);
    inputs
}
