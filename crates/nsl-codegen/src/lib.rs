pub mod autotune;
pub mod builtins;
pub mod compiler;
pub mod context;
pub mod context_parallel;
pub mod cost_model;
pub mod deterministic_kernels;
pub mod grammar_compiler;
pub mod schema_convert;

pub mod error;
pub mod expr;
pub mod func;
pub mod dynamic_shapes;
pub mod flash_attention;
pub mod fusion;
pub mod fusion_graph;
pub mod gpu_specs;
pub mod moe;
pub mod pipeline;
pub mod moe_kernels;
pub mod epilogue_fusion;
pub mod fp8;
pub mod memory_planner;
pub mod ownership;
pub mod reduction_fusion;
pub mod fusion_report;
pub mod gpu_target;
pub mod kernel_ir;
pub mod backend_ptx;
pub mod backend_amdgpu;
pub mod backend_metal;
pub mod backend_wgsl;
pub mod kernel_lower;
pub mod kernel;
pub mod linker;
pub mod serve;
pub mod sparse;
pub mod speculative;
pub mod source_ad;
pub mod standalone;
pub mod unikernel;
pub mod tensor_parallel;
pub mod stmt;
pub mod types;
pub mod multimodal;
pub mod vmap;
pub mod wcet;
pub mod weight_aware;
pub mod wengert;
pub mod ad_rules;

pub use compiler::{compile, compile_entry, compile_module, compile_module_with_imports, compile_test, compile_standalone, StandaloneConfig};
pub use error::CodegenError;
pub use standalone::create_weight_object;

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
        }
    }
}
