mod collection;
mod declaration;
mod entry_points;
mod functions;
mod kernel;
mod main_entry;
mod ownership_api;

use std::collections::{HashMap, HashSet};

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::isa::CallConv;
use cranelift_codegen::settings::{self, Configurable};
use cranelift_module::{DataDescription, DataId, FuncId, Module};
use cranelift_object::{ObjectBuilder, ObjectModule};

use nsl_ast::decl::FnDef;
use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::stmt::StmtKind;
use nsl_ast::{NodeId, Symbol};
use nsl_lexer::Interner;
use nsl_semantic::checker::TypeMap;
use nsl_semantic::types::Type;

use crate::context::StructLayout;
use crate::error::CodegenError;

// Re-export the free functions for `pub use compiler::{...}` in lib.rs
pub use entry_points::{
    compile, compile_entry, compile_module, compile_module_with_imports, compile_standalone,
    compile_test, compile_with_zk_info,
};

/// Compile-time context for FlashAttention PTX dispatch.
/// Populated by `compile_flash_attention_kernels()` when a `@flash_attention`-decorated
/// function is found, consumed by `compile_flash_attention_call()` in expr.rs.
pub struct FlashAttentionCompileContext {
    pub ptx_data_id: DataId,
    pub name_data_id: DataId,
    pub config: crate::flash_attention::FlashAttentionConfig,
}

/// A lambda body to be compiled after the current function.
pub struct PendingLambda {
    pub name: String,
    pub func_id: FuncId,
    pub sig: cranelift_codegen::ir::Signature,
    pub params: Vec<(Symbol, cl_types::Type)>,
    pub body: Expr,
    /// Captured variables from outer scope (name, cranelift type). These become extra params
    /// after the normal params in the lambda's Cranelift function signature.
    pub captures: Vec<(Symbol, cl_types::Type)>,
}

/// Sub-struct grouping all type-system registration state out of the `Compiler` god-object.
pub struct TypeRegistry {
    pub struct_layouts: HashMap<String, StructLayout>,
    pub enum_variants: HashMap<String, i64>,
    pub enum_defs: HashMap<String, Vec<(String, i64)>>,
    pub custom_dtype_ids: HashMap<String, u16>,
}

impl TypeRegistry {
    fn new() -> Self {
        Self {
            struct_layouts: HashMap::new(),
            enum_variants: HashMap::new(),
            enum_defs: HashMap::new(),
            custom_dtype_ids: HashMap::new(),
        }
    }
}

/// Sub-struct grouping all function/lambda registration state out of the `Compiler` god-object.
pub struct FunctionRegistry {
    pub functions: HashMap<String, (FuncId, cranelift_codegen::ir::Signature)>,
    pub runtime_fns: HashMap<String, (FuncId, cranelift_codegen::ir::Signature)>,
    pub pending_lambdas: Vec<PendingLambda>,
    pub closure_info: HashMap<Symbol, usize>,
    pub last_lambda_capture_count: Option<usize>,
    pub no_grad_fns: HashSet<String>,
    pub test_fns: Vec<String>,
}

impl FunctionRegistry {
    fn new() -> Self {
        Self {
            functions: HashMap::new(),
            runtime_fns: HashMap::new(),
            pending_lambdas: Vec::new(),
            closure_info: HashMap::new(),
            last_lambda_capture_count: None,
            no_grad_fns: HashSet::new(),
            test_fns: Vec::new(),
        }
    }
}

/// Sub-struct grouping model metadata fields out of the `Compiler` god-object.
pub struct ModelMetadata {
    pub model_methods: HashMap<String, HashMap<String, String>>,
    pub model_field_types: HashMap<String, HashMap<String, String>>,
    pub model_var_types: HashMap<Symbol, String>,
    pub imported_model_names: HashSet<String>,
    pub paged_kv_configs: HashMap<String, (i64, i64, i64, i64, i64)>,
    /// Model method AST bodies for source AD inlining: model_type_name -> method_name -> FnDef
    pub model_method_bodies: HashMap<String, HashMap<String, nsl_ast::decl::FnDef>>,
}

impl ModelMetadata {
    fn new() -> Self {
        Self {
            model_methods: HashMap::new(),
            model_field_types: HashMap::new(),
            model_var_types: HashMap::new(),
            imported_model_names: HashSet::new(),
            paged_kv_configs: HashMap::new(),
            model_method_bodies: HashMap::new(),
        }
    }
}

/// Sub-struct grouping fusion reporting state out of the `Compiler` god-object.
pub struct FusionState {
    pub events: Vec<crate::fusion_report::FusionEvent>,
    pub barriers: Vec<crate::fusion_report::FusionBarrierEvent>,
    pub report_enabled: bool,
    pub disabled: bool,
    /// @fuse-decorated functions: maps function name → (op_chain, num_inputs).
    /// At call time, these are launched as fused kernels instead of scalar calls.
    pub fused_fns: std::collections::HashMap<String, (Vec<String>, usize)>,
}

impl FusionState {
    fn new(options: &crate::CompileOptions) -> Self {
        Self {
            events: Vec::new(),
            barriers: Vec::new(),
            report_enabled: options.fusion_report,
            disabled: options.disable_fusion || options.debug_training,
            fused_fns: std::collections::HashMap::new(),
        }
    }
}

/// Sub-struct grouping memory planning state out of the `Compiler` god-object.
pub struct MemoryPlanState {
    pub slab_plan: Option<crate::memory_planner::SlabPlan>,
    pub slab_name_offsets: HashMap<String, u64>,
    pub weight_scales: HashMap<String, f32>,
}

impl MemoryPlanState {
    fn new() -> Self {
        Self {
            slab_plan: None,
            slab_name_offsets: HashMap::new(),
            weight_scales: HashMap::new(),
        }
    }
}

/// Sub-struct grouping GPU kernel compilation state out of the `Compiler` god-object.
pub struct GpuKernelState {
    pub kernel_ptx_data: HashMap<String, (DataId, DataId)>,
    pub flash_attention_context: Option<FlashAttentionCompileContext>,
}

impl GpuKernelState {
    fn new() -> Self {
        Self {
            kernel_ptx_data: HashMap::new(),
            flash_attention_context: None,
        }
    }
}

/// Sub-struct grouping vmap batched function state out of the `Compiler` god-object.
pub struct VmapState {
    pub batched_fn_names: HashMap<String, String>,
    pub matmul_rewrites: HashMap<NodeId, String>,
}

impl VmapState {
    fn new() -> Self {
        Self {
            batched_fn_names: HashMap::new(),
            matmul_rewrites: HashMap::new(),
        }
    }
}

/// Configuration for standalone export codegen.
pub struct StandaloneConfig {
    /// If true, weights are embedded in the binary via linker symbols.
    /// If false, weights are loaded at runtime from the sidecar path.
    pub embedded: bool,
    /// Path to the `.nslweights` sidecar file (used when `embedded == false`).
    pub sidecar_path: String,
}

/// Feature-specific decorator and configuration state extracted from model/function
/// decorators during compilation. These are set during declaration processing and
/// consumed during codegen. Grouped here to keep the main Compiler struct manageable.
pub struct FeatureConfigs {
    // ── Scaling & Parallelism (M30, M34, M41, M43) ───────────────────
    /// M30: Sharded layers — "ModelName.layer_name" → ShardInfo
    pub shard_configs: HashMap<String, crate::tensor_parallel::ShardInfo>,
    /// M30: Activation distribution states (for future all-reduce insertion)
    pub activation_states: HashMap<String, crate::tensor_parallel::DistState>,
    /// M30: Tensor parallelism world size (number of devices)
    pub world_size: usize,
    /// M34: Context parallelism configs — "ModelName.layer_name" → ContextParallelInfo
    pub context_parallel_configs: HashMap<String, crate::context_parallel::ContextParallelInfo>,
    /// M34: Ring size for context parallelism
    pub cp_ring_size: usize,
    /// M41: Whether this serve block uses disaggregated inference.
    pub disaggregated: bool,
    /// M41: Number of prefill workers (from serve config).
    pub prefill_workers: usize,
    /// M41: Number of decode workers (from serve config).
    pub decode_workers: usize,
    /// M43: Pipeline parallelism configuration.
    pub pipeline_config: Option<crate::pipeline::PipelineConfig>,
    /// M43: 3D parallelism configuration.
    pub parallelism_config: Option<crate::pipeline::ParallelismConfig>,
    /// M43: ZeRO optimizer sharding stage.
    pub zero_stage: Option<u8>,

    // ── Inference Features (M32, M33, M42, M44) ──────────────────────
    /// M32: MoE layer configs — "ModelName.layer_name" → MoeInfo
    pub moe_configs: HashMap<String, crate::moe::MoeInfo>,
    /// M33: Speculative decoding configs — "ModelName.layer_name" → SpeculativeInfo
    pub speculative_configs: HashMap<String, crate::speculative::SpeculativeInfo>,
    /// M42: Per-model KV compression policies from @kv_compress decorators.
    pub kv_compress_policies: HashMap<String, Vec<KvCompressPolicy>>,
    /// M44: Compiled grammar FSMs from @grammar decorators and generate() schema= args.
    pub grammar_configs: HashMap<String, GrammarInfo>,

    // ── Quantization & Precision (M35) ───────────────────────────────
    /// M35: Functions with @fp8_compute decorator
    pub fp8_compute_fns: HashSet<String>,
    /// M35: Model quantization configs — "ModelName" -> QuantConfig
    pub quant_configs: HashMap<String, QuantConfig>,

    // ── Ownership & Safety (M38, M39, M40) ───────────────────────────
    /// M38b: Whether --linear-types flag is active
    pub linear_types_enabled: bool,
    /// M38b: Per-function ownership metadata from semantic pass
    pub ownership_info: HashMap<String, crate::ownership::FunctionOwnership>,
    /// M39: Functions with @vmap decorator and their batch configuration
    pub vmap_configs: HashMap<String, crate::vmap::VmapConfig>,
    /// M40: Source-to-source AD enabled (default true; --tape-ad forces tape-only)
    pub source_ad_enabled: bool,

    // ── Weight Intelligence (M52) ────────────────────────────────────
    /// M52: Loaded weight map (populated when --weights is passed)
    pub weight_map: Option<crate::weight_aware::WeightMap>,
    /// M52: Sparsity hints per matmul operation (AST NodeId -> SparsityHint)
    pub sparsity_hints: HashMap<NodeId, crate::weight_aware::SparsityHint>,
    /// M52: Weight integrity hash for embedding in .rodata
    pub weight_integrity: Option<crate::weight_aware::WeightIntegrity>,

    // ── Safety & Verification (M53, M55) ─────────────────────────────
    /// M53: Functions decorated with @real_time — name -> constraint
    pub real_time_fns: HashMap<String, crate::wcet::RealTimeConstraint>,
    /// M53: Functions decorated with @wcet_budget — name -> constraint
    pub wcet_budget_fns: HashMap<String, crate::wcet::WcetBudgetConstraint>,
    /// M53: Collected WCET analysis results for @real_time functions
    pub wcet_results: Vec<crate::wcet::FunctionWcet>,
    /// M55: Functions (and model methods) decorated with @zk_proof — name -> ZkMode
    pub zk_proof_fns: HashMap<String, crate::zk::backend::ZkMode>,
    /// M55: Functions decorated with @zk_lookup — name -> (input_bits, output_bits)
    pub zk_lookup_fns: HashMap<String, (u32, u32)>,
    /// M55: ASTs of @zk_proof functions — stored during declaration for ZK compilation
    pub zk_fn_defs: HashMap<String, nsl_ast::decl::FnDef>,
}

impl FeatureConfigs {
    fn new(options: &crate::CompileOptions) -> Self {
        Self {
            shard_configs: HashMap::new(),
            activation_states: HashMap::new(),
            world_size: options.world_size,
            context_parallel_configs: HashMap::new(),
            cp_ring_size: 1,
            disaggregated: false,
            prefill_workers: 1,
            decode_workers: 1,
            pipeline_config: None,
            parallelism_config: None,
            zero_stage: options.zero_stage,
            moe_configs: HashMap::new(),
            speculative_configs: HashMap::new(),
            kv_compress_policies: HashMap::new(),
            grammar_configs: HashMap::new(),
            fp8_compute_fns: HashSet::new(),
            quant_configs: HashMap::new(),
            linear_types_enabled: options.linear_types_enabled,
            ownership_info: options.ownership_info.clone(),
            vmap_configs: HashMap::new(),
            source_ad_enabled: options.source_ad,
            weight_map: None,
            sparsity_hints: HashMap::new(),
            weight_integrity: None,
            real_time_fns: HashMap::new(),
            wcet_budget_fns: HashMap::new(),
            wcet_results: Vec::new(),
            zk_proof_fns: HashMap::new(),
            zk_lookup_fns: HashMap::new(),
            zk_fn_defs: HashMap::new(),
        }
    }
}

pub struct Compiler<'a> {
    // ── Core infrastructure ──────────────────────────────────────────
    pub module: ObjectModule,
    pub interner: &'a Interner,
    pub type_map: &'a TypeMap,
    pub call_conv: CallConv,
    pub compile_options: crate::CompileOptions,
    pub dump_ir: bool,
    func_index: u32,

    // ── Function registry ────────────────────────────────────────────
    pub registry: FunctionRegistry,

    // ── Type system ──────────────────────────────────────────────────
    pub types: TypeRegistry,

    // ── Model metadata ───────────────────────────────────────────────
    pub models: ModelMetadata,

    // ── Module & linking ─────────────────────────────────────────────
    pub module_prefix: String,
    pub string_pool: HashMap<String, DataId>,
    pub(crate) standalone_config: Option<StandaloneConfig>,

    // ── GPU kernels ──────────────────────────────────────────────────
    pub kernels: GpuKernelState,

    // ── Fusion reporting (M31) ───────────────────────────────────────
    pub fusion: FusionState,

    // ── Memory planning (M36) ────────────────────────────────────────
    pub memory: MemoryPlanState,

    // ── vmap batched function registry (M39) ─────────────────────────
    pub vmap: VmapState,

    // ── Feature configs (M30-M55 decorators) ─────────────────────────
    /// All feature-specific decorator state extracted into a single sub-struct.
    /// Access via `self.features.field_name`.
    pub features: FeatureConfigs,
}

/// Quantization configuration for a model.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub dtype: String, // "awq4", "gptq4", "gptq8"
    pub group_size: i64,
}

/// M44: Codegen-side grammar configuration.
pub struct GrammarInfo {
    pub start_rule: String,
    pub grammar_source: String, // "json_schema", "bnf", "regex"
}

/// M42: Codegen-side KV compression config.
pub struct KvCompressPolicy {
    pub method: String,
    pub scheme: u8,
    pub window: usize,
    pub sinks: usize,
    pub budget: usize,
}

/// Mangle a function name with a module prefix for unique Cranelift symbols.
/// If prefix is empty, returns the name unchanged.
pub(crate) fn mangle_name(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}__{name}")
    }
}

impl<'a> Compiler<'a> {
    pub fn new(
        interner: &'a Interner,
        type_map: &'a TypeMap,
        options: &crate::CompileOptions,
    ) -> Result<Self, CodegenError> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| CodegenError::new(e.to_string()))?;

        // Enable stack probing for large stack frames (>4KB).
        // On Windows x64, stack pages are committed lazily via guard pages.
        // Without probing, functions with large stack frames (e.g., train steps
        // with many tensor temporaries) skip past the guard page and trigger
        // STATUS_STACK_BUFFER_OVERRUN instead of growing the stack.
        flag_builder
            .enable("enable_probestack")
            .map_err(|e| CodegenError::new(format!("failed to enable probestack: {e}")))?;
        flag_builder
            .set("probestack_strategy", "inline")
            .map_err(|e| CodegenError::new(format!("failed to set probestack strategy: {e}")))?;

        // M62a: Enable position-independent code for shared library emission.
        if options.shared_lib {
            flag_builder
                .set("is_pic", "true")
                .map_err(|e| CodegenError::new(format!("failed to enable PIC: {e}")))?;
        }

        let isa_builder = cranelift_native::builder()
            .map_err(|e| CodegenError::new(format!("failed to create ISA builder: {e}")))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| CodegenError::new(format!("failed to build ISA: {e}")))?;

        // Detect calling convention from ISA, with Windows x64 safety override
        let call_conv = {
            let detected = isa.default_call_conv();
            if std::env::var("NSL_DEBUG").is_ok() {
                eprintln!("[nsl] Cranelift ISA call convention: {:?}", detected);
            }
            // Force WindowsFastcall on Windows x64 — Cranelift may misdetect
            #[cfg(all(target_os = "windows", target_arch = "x86_64"))]
            {
                if detected != cranelift_codegen::isa::CallConv::WindowsFastcall {
                    eprintln!(
                        "[nsl] WARNING: Cranelift detected {:?} on Windows x64, \
                         forcing WindowsFastcall for ABI correctness",
                        detected
                    );
                    cranelift_codegen::isa::CallConv::WindowsFastcall
                } else {
                    detected
                }
            }
            #[cfg(not(all(target_os = "windows", target_arch = "x86_64")))]
            {
                detected
            }
        };

        let builder =
            ObjectBuilder::new(isa, "nsl_module", cranelift_module::default_libcall_names())
                .map_err(|e| CodegenError::new(format!("failed to create object builder: {e}")))?;

        let module = ObjectModule::new(builder);

        Ok(Compiler {
            module,
            interner,
            type_map,
            call_conv,
            compile_options: options.clone(),
            dump_ir: false,
            func_index: 0,
            registry: FunctionRegistry::new(),
            types: TypeRegistry::new(),
            models: ModelMetadata::new(),
            module_prefix: String::new(),
            string_pool: HashMap::new(),
            standalone_config: None,
            kernels: GpuKernelState::new(),
            fusion: FusionState::new(options),
            memory: MemoryPlanState::new(),
            vmap: VmapState::new(),
            features: FeatureConfigs::new(options),
        })
    }

    /// Resolve a Symbol to its interned string.
    pub fn resolve_sym(&self, sym: Symbol) -> &str {
        self.interner.resolve(sym.0).unwrap_or("<unknown>")
    }

    /// Get the inferred type of an AST node by NodeId.
    pub fn node_type(&self, id: NodeId) -> &Type {
        self.type_map.get(&id).unwrap_or(&Type::Unknown)
    }

    /// M47b: Resolve the GPU compilation target from `--target` CLI flag.
    /// Defaults to CUDA when no target is specified.
    pub fn gpu_target(&self) -> crate::gpu_target::GpuTarget {
        crate::gpu_target::GpuTarget::from_target_string(&self.compile_options.target)
    }

    /// M42: Look up the first KV compression policy for a specific model layer.
    /// Key format: `"ModelName.layer_name"` (matches collection.rs insertion key).
    /// Returns `None` when no `@kv_compress` decorator was found for that layer.
    pub fn kv_compress_for_layer(
        &self,
        model_name: &str,
        layer_name: &str,
    ) -> Option<&KvCompressPolicy> {
        let key = format!("{}.{}", model_name, layer_name);
        self.features
            .kv_compress_policies
            .get(&key)
            .and_then(|policies| policies.first())
    }

    /// Allocate a unique function index for Cranelift.
    pub fn next_func_index(&mut self) -> u32 {
        let idx = self.func_index;
        self.func_index += 1;
        idx
    }

    // CompileOptions is now threaded through the compilation pipeline.
    // fusion_report_enabled is set from compile_options in each public compile function.

    /// Enable fusion report collection (called when --fusion-report or @fuse_graph is present).
    pub fn enable_fusion_report(&mut self) {
        self.fusion.report_enabled = true;
    }

    /// Resolve a type name (from AST annotation) to a Cranelift type.
    pub(crate) fn resolve_type_name_to_cl(&self, sym: Symbol) -> cl_types::Type {
        let name = self.resolve_sym(sym);
        match name {
            "int" => cl_types::I64,
            "float" => cl_types::F64,
            "f32" => cl_types::F32,
            "f64" => cl_types::F64,
            "bool" => cl_types::I8,
            "str" => cl_types::I64,
            "i8" | "int8" => cl_types::I8,
            "i16" | "int16" => cl_types::I16,
            "i32" | "int32" => cl_types::I32,
            "i64" | "int64" => cl_types::I64,
            "bf16" | "fp16" => cl_types::F32,
            _ => cl_types::I64,
        }
    }

    // ── @fuse fn validation ─────────────────────────────────────────

    /// Validate that a `@fuse`-decorated function body contains only fusible
    /// elementwise operations, let bindings, and return statements.
    pub fn validate_fuse_body(&self, fn_def: &FnDef) -> Result<(), CodegenError> {
        for stmt in &fn_def.body.stmts {
            match &stmt.kind {
                StmtKind::Return(Some(expr)) => {
                    self.validate_fusible_expr(expr)?;
                }
                StmtKind::Return(None) => {
                    // bare return is OK (no-op)
                }
                StmtKind::Expr(expr) => {
                    self.validate_fusible_expr(expr)?;
                }
                StmtKind::VarDecl {
                    value: Some(expr), ..
                } => {
                    self.validate_fusible_expr(expr)?;
                }
                StmtKind::ModelDef(_) => {
                    return Err(CodegenError::new(
                        "model blocks cannot appear inside @fuse fn".to_string(),
                    ));
                }
                _ => {
                    return Err(CodegenError::new(
                        "@fuse function body must contain only fusible expressions and let bindings"
                            .to_string(),
                    ));
                }
            }
        }
        Ok(())
    }

    /// Recursively validate that an expression tree contains only fusible ops.
    fn validate_fusible_expr(&self, expr: &Expr) -> Result<(), CodegenError> {
        match &expr.kind {
            ExprKind::BinaryOp { left, op, right } => {
                if !crate::fusion::is_fusible_binop(&format!("{:?}", op)) {
                    return Err(CodegenError::new(format!(
                        "@fuse: non-fusible binary op {:?} (only Add/Sub/Mul/Div/Pow allowed)",
                        op
                    )));
                }
                self.validate_fusible_expr(left)?;
                self.validate_fusible_expr(right)?;
            }
            ExprKind::Call { callee, args } => {
                if let ExprKind::Ident(sym) = &callee.kind {
                    let name = self.resolve_sym(*sym);
                    if !crate::fusion::is_fusible_op(name) {
                        return Err(CodegenError::new(format!(
                            "@fuse: non-fusible function call '{}' (only elementwise ops allowed)",
                            name
                        )));
                    }
                }
                for arg in args {
                    self.validate_fusible_expr(&arg.value)?;
                }
            }
            ExprKind::Ident(_) | ExprKind::FloatLiteral(_) | ExprKind::IntLiteral(_) => {
                // Leaf nodes: always OK
            }
            ExprKind::UnaryOp { operand, .. } => {
                self.validate_fusible_expr(operand)?;
            }
            _ => {
                return Err(CodegenError::new(
                    "@fuse: unsupported expression in fused function body".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Intern a string literal into the module's data section.
    pub fn intern_string(&mut self, s: &str) -> Result<DataId, CodegenError> {
        if let Some(&data_id) = self.string_pool.get(s) {
            return Ok(data_id);
        }
        let name = format!(".str.{}", self.string_pool.len());
        let data_id = self
            .module
            .declare_data(&name, cranelift_module::Linkage::Local, false, false)
            .map_err(|e| CodegenError::new(format!("failed to declare string data: {e}")))?;

        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);

        let mut desc = DataDescription::new();
        desc.define(bytes.into_boxed_slice());
        self.module
            .define_data(data_id, &desc)
            .map_err(|e| CodegenError::new(format!("failed to define string data: {e}")))?;

        self.string_pool.insert(s.to_string(), data_id);
        Ok(data_id)
    }

    /// M53: Run WCET analysis for all `@real_time` functions.
    ///
    /// For each function with a `@real_time(max_latency_ms=N)` decorator, this creates
    /// placeholder operation estimates, runs no-heap and static control-flow proofs,
    /// checks the computed WCET against the declared bound, and optionally emits
    /// certificate JSON and/or DO-178C compliance reports.
    pub fn run_wcet_analysis(&mut self) -> Result<(), CodegenError> {
        use crate::gpu_specs::{find_fpga, find_gpu};
        use crate::wcet::*;

        if self.features.real_time_fns.is_empty() {
            return Ok(());
        }

        let safety_margin = self.compile_options.wcet_safety_margin;
        let target_kind = self.compile_options.wcet_target.as_str();

        // Build the WcetTarget based on CLI flags
        let target = match target_kind {
            "fpga" => {
                let fpga_name = self
                    .compile_options
                    .fpga_device
                    .as_deref()
                    .unwrap_or("xcvu440");
                let fpga = find_fpga(fpga_name).ok_or_else(|| {
                    CodegenError::new(format!(
                        "WCET: unknown FPGA '{}'. Available: xcvu440, xczu9eg, ve2302",
                        fpga_name
                    ))
                })?;
                WcetTarget::Fpga {
                    device_name: fpga.device_name.to_string(),
                    ocm_size_kb: fpga.ocm_size_kb,
                }
            }
            "groq" => {
                return Err(CodegenError::new(
                    "Groq LPU WCET not yet supported (ISA not public). \
                     Use --wcet-target gpu or --wcet-target fpga."
                        .to_string(),
                ));
            }
            _ => {
                // Default: GPU statistical
                let gpu_name = self
                    .compile_options
                    .wcet_gpu
                    .as_deref()
                    .unwrap_or("A100-SXM");
                let _ = find_gpu(gpu_name).ok_or_else(|| {
                    CodegenError::new(format!(
                        "WCET: unknown GPU '{}'. Use --gpu to specify.",
                        gpu_name
                    ))
                })?;
                WcetTarget::Gpu {
                    device_name: gpu_name.to_string(),
                }
            }
        };

        // M53: If --wcet-cpu is specified, validate the CPU model and emit an advisory note.
        // CPU WCET estimates are advisory — they use the CpuSpec database in gpu_specs.rs.
        if let Some(ref cpu_name) = self.compile_options.wcet_cpu {
            if !cpu_name.is_empty() {
                match crate::gpu_specs::find_cpu(cpu_name) {
                    Some(cpu) => {
                        eprintln!(
                            "[nsl] WCET CPU target: {} @ {} MHz, {:.0} GFLOPS/core (fp32), {} cores",
                            cpu.name,
                            cpu.base_clock_mhz,
                            (cpu.fp32_flops_per_cycle as f64 * cpu.base_clock_mhz as f64) / 1000.0,
                            cpu.num_cores,
                        );
                    }
                    None => {
                        eprintln!(
                            "[nsl] warning: unknown --cpu '{}'. Known models: cortex-a78, x86-64-v4. \
                             CPU WCET will use GPU/FPGA estimates only.",
                            cpu_name
                        );
                    }
                }
            }
        }

        // Clone to avoid borrowing self.features.real_time_fns while mutating self.features.wcet_results
        for (fn_name, constraint) in self.features.real_time_fns.clone() {
            let ops = match &target {
                WcetTarget::Gpu { device_name } => {
                    let gpu = find_gpu(device_name).unwrap();
                    eprintln!(
                        "note: GPU WCET is a statistical p95 estimate (not a certified proof). \
                         For safety-critical applications, use --wcet-target fpga."
                    );
                    vec![
                        estimate_matmul_gpu_statistical(1, 512, 512, "fp32", gpu),
                        estimate_elementwise_gpu_statistical(512, "fp32", "relu", gpu),
                    ]
                }
                WcetTarget::Fpga { device_name, .. } => {
                    let fpga = find_fpga(device_name).unwrap();
                    vec![
                        wcet_matmul_fpga_certified(1, 512, 512, "fp32", fpga),
                        wcet_elementwise_fpga_certified(512, "fp32", "relu", fpga),
                    ]
                }
                WcetTarget::GroqLpu => unreachable!(),
            };

            let total_ns: u64 = ops.iter().map(|o| o.worst_case_ns).sum();
            let total_ms = total_ns as f64 / 1_000_000.0;
            let final_ms = total_ms * safety_margin;
            let bound_satisfied = final_ms <= constraint.max_latency_ms;

            let no_heap = prove_no_heap(self.memory.slab_plan.as_ref(), &fn_name);
            let static_cf = prove_static_cf(&target, &ops);

            let func_wcet = FunctionWcet {
                name: fn_name.clone(),
                ops: ops.clone(),
                total_wcet_ns: total_ns,
                total_wcet_ms: total_ms,
                safety_margin,
                final_wcet_ms: final_ms,
                constraint: Some(constraint.clone()),
                bound_satisfied,
                no_heap_proven: no_heap.proven,
                static_cf_proven: static_cf.proven,
            };

            if !bound_satisfied {
                let suggestions = generate_suggestions(&ops, constraint.max_latency_ms, final_ms);
                let violation = WcetViolation {
                    function: fn_name.clone(),
                    declared_bound_ms: constraint.max_latency_ms,
                    computed_wcet_ms: final_ms,
                    ops,
                    suggestions,
                };
                eprintln!("{}", format_wcet_violation(&violation));
                return Err(CodegenError::new(format!(
                    "WCET bound exceeded for '{}': {:.3} ms > {:.3} ms",
                    fn_name, final_ms, constraint.max_latency_ms
                )));
            }

            // Emit certificate if requested
            if let Some(ref cert_path) = self.compile_options.wcet_report_path {
                let cert =
                    build_certificate(&func_wcet, &no_heap, &static_cf, "source.nsl", &target);
                emit_certificate(&cert, cert_path).map_err(|e| {
                    CodegenError::new(format!("WCET certificate write failed: {}", e))
                })?;
                eprintln!("WCET certificate: {}", cert_path.display());
            }

            // Emit DO-178C report if requested (guarded: FPGA only)
            if let Some(ref do178c_path) = self.compile_options.do178c_report {
                let cert =
                    build_certificate(&func_wcet, &no_heap, &static_cf, "source.nsl", &target);
                emit_do178c_report(&cert, do178c_path)
                    .map_err(|e| CodegenError::new(format!("DO-178C report: {}", e)))?;
            }

            self.features.wcet_results.push(func_wcet);
        }

        Ok(())
    }

    /// M52: Embed the weight file SHA-256 hash as a .rodata symbol.
    /// The symbol `__nsl_weight_hash` contains 32 bytes of the SHA-256 hash.
    pub fn embed_weight_hash(&mut self) -> Result<(), CodegenError> {
        if let Some(ref integrity) = self.features.weight_integrity {
            let data_id = self
                .module
                .declare_data(
                    "__nsl_weight_hash",
                    cranelift_module::Linkage::Export,
                    false,
                    false,
                )
                .map_err(|e| {
                    CodegenError::new(format!("failed to declare weight hash data: {e}"))
                })?;

            let mut desc = DataDescription::new();
            desc.define(integrity.compile_hash.to_vec().into_boxed_slice());
            self.module.define_data(data_id, &desc).map_err(|e| {
                CodegenError::new(format!("failed to define weight hash data: {e}"))
            })?;
        }
        Ok(())
    }

    pub fn finalize(self) -> Result<Vec<u8>, CodegenError> {
        let product = self.module.finish();
        product
            .emit()
            .map_err(|e| CodegenError::new(format!("failed to emit object: {e}")))
    }
}
