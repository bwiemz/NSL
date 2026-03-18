pub mod autotune;
pub mod builtins;
pub mod compiler;
pub mod context;
pub mod context_parallel;
pub mod cost_model;
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
pub mod kernel_lower;
pub mod kernel;
pub mod linker;
pub mod serve;
pub mod speculative;
pub mod source_ad;
pub mod standalone;
pub mod tensor_parallel;
pub mod stmt;
pub mod types;
pub mod vmap;
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
        }
    }
}
