pub mod autotune;
pub mod builtins;
pub mod compiler;
pub mod context;
pub mod context_parallel;
pub mod cost_model;
pub mod deterministic_kernels;
pub mod grammar_compiler;
pub mod schema_convert;

pub mod ad_rules;
pub mod backend_amdgpu;
pub mod backend_metal;
pub mod backend_ptx;
pub mod backend_wgsl;
pub mod cep;
pub mod cep_importance;
pub mod cep_oracle;
pub mod cep_rewrite;
pub mod cep_search;
pub mod cfie;
pub mod cfie_fused_sample;
pub mod cfie_grammar;
pub mod cfie_kv_plan;
pub mod cfie_kv_quant;
pub mod cfie_persistent;
pub mod cfie_speculative;
pub mod cpdt;
pub mod cpdt_comm;
pub mod cpdt_expert;
pub mod cpdt_joint;
pub mod cpdt_optim;
pub mod cpdt_precision;
pub mod cpdt_zero;
pub mod csha;
pub mod csha_apply;
pub mod csha_boundary;
pub mod csha_pipeline;
pub mod csha_specialize;
pub mod dynamic_shapes;
pub mod epilogue_fusion;
pub mod error;
pub mod expr;
pub mod fase;
pub mod fase_clip;
pub mod fase_memory;
pub mod fase_optimizer;
pub mod flash_attention;
pub mod fp8;
pub mod func;
pub mod fusion;
pub mod fusion_graph;
pub mod fusion_report;
pub mod gpu_specs;
pub mod gpu_target;
pub mod kernel;
pub mod kernel_ir;
pub mod kernel_lower;
pub mod linker;
pub mod memory_planner;
pub mod moe;
pub mod moe_kernels;
pub mod multimodal;
pub mod ffi_ownership;
pub mod ownership;
pub mod ownership_expr;
pub mod pca_detect;
pub mod pca_rope;
pub mod pca_segment;
pub mod pca_tileskip;
pub mod pipeline;
pub mod reduction_fusion;
pub mod serve;
pub mod source_ad;
pub mod sparse;
pub mod speculative;
pub mod standalone;
pub mod stmt;
pub mod tensor_parallel;
pub mod types;
pub mod unikernel;
pub mod unikernel_boot;
pub mod use_count;
pub mod vmap;
pub mod wcet;
pub mod weight_aware;
pub mod wengert;
pub mod wengert_lower;
pub mod wggo;
pub mod wggo_apply;
pub mod wggo_conflicts;
pub mod wggo_cost;
pub mod wggo_dp;
pub mod wggo_graph;
pub mod wggo_ilp;
pub mod wrga;
pub mod wrga_fusion;
pub mod wrga_memory;
pub mod wrga_prune;
pub mod wrga_roofline;
pub mod wrga_spectral;
pub mod zk;

pub use compiler::{
    compile, compile_entry, compile_module, compile_module_with_imports, compile_standalone,
    compile_test, compile_with_zk_info, StandaloneConfig,
};
pub use error::CodegenError;
pub use standalone::create_weight_object;

use std::collections::HashMap;

/// Compiler configuration flags passed from CLI.
#[derive(Clone)]
pub struct CompileOptions {
    pub no_autotune: bool,
    pub autotune_fresh: bool,
    pub world_size: usize,
    pub fusion_report: bool,
    /// M36: VRAM budget in bytes (None = no limit, Some(n) = fail if plan exceeds n)
    pub vram_budget: Option<u64>,
    /// M36: Print memory plan report to stderr
    pub memory_report: bool,
    /// M47: GPU compilation target name.
    pub target: String,
    /// Disable all fusion optimizations (for differential testing baseline).
    pub disable_fusion: bool,
    /// M40b: Force tape-based AD (disable source-to-source AD).
    pub tape_ad: bool,
    /// M40: Use compile-time source-to-source AD for training (default: false = tape AD).
    pub source_ad: bool,
    /// M45: Enable tensor operation tracing.
    pub trace_ops: bool,
    /// M45: Enable compile-time NaN risk analysis.
    pub nan_analysis: bool,
    /// M46: Enable deterministic mode.
    pub deterministic: bool,
    /// M52: Path to safetensors weight file for weight-aware compilation
    pub weight_file: Option<std::path::PathBuf>,
    /// M52: Weight-aware compilation configuration
    pub weight_config: weight_aware::WeightAwareConfig,
    /// M52: Whether to emit a weight analysis report (nsl check --weight-analysis)
    pub weight_analysis: bool,
    /// M54: Unikernel build configuration (None = normal build)
    pub unikernel_config: Option<crate::unikernel::UnikernelConfig>,
    /// M53: Enable WCET analysis for @real_time functions
    pub wcet_enabled: bool,
    /// M53: GPU target name for WCET analysis (e.g., "Orin", "H100")
    pub wcet_gpu: Option<String>,
    /// M53: CPU target name for WCET analysis (e.g., "cortex-a78")
    pub wcet_cpu: Option<String>,
    /// M53: Path to write WCET certificate JSON
    pub wcet_report_path: Option<std::path::PathBuf>,
    /// M53: Safety margin multiplier for WCET (default: 1.05 = 5%)
    pub wcet_safety_margin: f64,
    /// M53: Path to write DO-178C compliance report
    pub do178c_report: Option<std::path::PathBuf>,
    /// M53: WCET target type: "gpu" (statistical), "fpga" (certified), "groq" (blocked)
    pub wcet_target: String,
    /// M53: FPGA device name for certified WCET (e.g., "xcvu440", "xczu9eg")
    pub fpga_device: Option<String>,
    /// M38a: Enable linear types ownership checking.
    pub linear_types_enabled: bool,
    /// M38a: Per-function ownership metadata from semantic analysis.
    /// Keys are function names, values have linear_params and shared_params.
    pub ownership_info: HashMap<String, crate::ownership::FunctionOwnership>,
    /// M55: Emit a ZK inference circuit alongside compiled output.
    pub zk_circuit: bool,
    /// M55: ZK backend to use ("folding", "halo2", or "plonky3").
    pub zk_backend: String,
    /// M55: ZK field to use ("m31" or "bn254").
    pub zk_field: String,
    /// M55: Also emit a Solidity verifier contract.
    pub zk_solidity: bool,
    /// M55: Path to safetensors weight file used as ZK witness.
    pub zk_weights_path: Option<std::path::PathBuf>,
    /// M43b: ZeRO optimizer sharding stage (1, 2, or 3)
    pub zero_stage: Option<u8>,
    /// Debug training mode: disables fusion, disables FBIP, and emits
    /// gradient checksum assertions after each backward pass.
    pub debug_training: bool,
    /// M62a: Build as a shared library (.so/.dylib/.dll) instead of an executable.
    pub shared_lib: bool,
    /// WGGO: global optimization mode ("full", "greedy", "off"). When `None`,
    /// WGGO is not run.  See `crates/nsl-codegen/src/wggo.rs`.
    pub wggo_mode: Option<String>,
    /// WGGO: print the global-optimization report to stderr.
    pub wggo_report: bool,
    /// CSHA: fusion mode ("auto", "boundary", "pipeline", "block", "off").
    /// When `None`, CSHA is not run.  See `crates/nsl-codegen/src/csha.rs`.
    pub csha_mode: Option<String>,
    /// CSHA: print the attention-fusion report to stderr.
    pub csha_report: bool,
}

impl Default for CompileOptions {
    fn default() -> Self {
        Self {
            no_autotune: false,
            autotune_fresh: false,
            world_size: 1,
            fusion_report: false,
            vram_budget: None,
            memory_report: false,
            target: "cuda".to_string(),
            disable_fusion: false,
            tape_ad: false,
            source_ad: false,
            trace_ops: false,
            nan_analysis: false,
            deterministic: false,
            weight_file: None,
            weight_config: weight_aware::WeightAwareConfig::default(),
            weight_analysis: false,
            unikernel_config: None,
            wcet_enabled: false,
            wcet_gpu: None,
            wcet_cpu: None,
            wcet_report_path: None,
            wcet_safety_margin: 1.05,
            do178c_report: None,
            wcet_target: "gpu".to_string(),
            fpga_device: None,
            linear_types_enabled: false,
            ownership_info: HashMap::new(),
            zk_circuit: false,
            zk_backend: "folding".to_string(),
            zk_field: "m31".to_string(),
            zk_solidity: false,
            zk_weights_path: None,
            zero_stage: None,
            debug_training: false,
            shared_lib: false,
            wggo_mode: None,
            wggo_report: false,
            csha_mode: None,
            csha_report: false,
        }
    }
}
