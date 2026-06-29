//! Command-line argument definitions (clap) for the `nsl` binary.
//!
//! The `Cli` enum and its subcommand/value enums were extracted verbatim
//! from `main.rs`; `main_inner` (still in `main.rs`) matches on them.

use std::path::PathBuf;

use clap::Parser as ClapParser;

/// Output format for `--training-report`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, PartialEq, Eq)]
pub enum TrainingReportFormat {
    Text,
    Json,
}

/// CLI wrapper for `nsl_codegen::WggoImportance` — keeps `clap` out of `nsl-codegen`.
#[derive(clap::ValueEnum, Clone, Copy, Debug, Default)]
#[clap(rename_all = "lower")]
pub(crate) enum CliWggoImportance {
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
pub(crate) enum Cli {
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
