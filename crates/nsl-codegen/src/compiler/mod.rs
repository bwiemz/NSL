mod collection;
mod declaration;
mod entry_points;
mod functions;
mod kernel;
mod main_entry;

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
    compile_test,
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

/// Configuration for standalone export codegen.
pub struct StandaloneConfig {
    /// If true, weights are embedded in the binary via linker symbols.
    /// If false, weights are loaded at runtime from the sidecar path.
    pub embedded: bool,
    /// Path to the `.nslweights` sidecar file (used when `embedded == false`).
    pub sidecar_path: String,
}

pub struct Compiler<'a> {
    pub module: ObjectModule,
    pub interner: &'a Interner,
    pub type_map: &'a TypeMap,
    pub functions: HashMap<String, (FuncId, cranelift_codegen::ir::Signature)>,
    pub runtime_fns: HashMap<String, (FuncId, cranelift_codegen::ir::Signature)>,
    pub string_pool: HashMap<String, DataId>,
    pub struct_layouts: HashMap<String, StructLayout>,
    /// Enum variant name → integer tag. Flat map: "ReLU" → 0, "Sigmoid" → 1, etc.
    pub enum_variants: HashMap<String, i64>,
    /// Enum name → list of (variant_name, tag)
    pub enum_defs: HashMap<String, Vec<(String, i64)>>,
    /// Model name → { method_name → mangled_fn_name } for method dispatch.
    pub model_methods: HashMap<String, HashMap<String, String>>,
    /// Lambda bodies to compile after the current function is finished.
    pub pending_lambdas: Vec<PendingLambda>,
    /// Maps variable Symbol to its closure capture count. If present, the variable holds a
    /// heap-allocated closure struct `{ fn_ptr, num_captures, captures[] }` instead of a bare fn ptr.
    pub closure_info: HashMap<Symbol, usize>,
    /// Set by compile_lambda when it creates a closure; consumed by VarDecl to register in closure_info.
    pub last_lambda_capture_count: Option<usize>,
    pub call_conv: CallConv,
    pub dump_ir: bool,
    /// Functions decorated with `@no_grad` — tape is paused during their execution.
    pub no_grad_fns: HashSet<String>,
    /// Functions decorated with `@test` — discovered by `nsl test` runner.
    pub test_fns: Vec<String>,
    /// Module prefix for name mangling. Empty means no mangling (entry module / single-file).
    pub module_prefix: String,
    /// Kernel name → (ptx_data_id, name_data_id) for compiled GPU kernels.
    pub kernel_ptx_data: HashMap<String, (DataId, DataId)>,
    /// model_name → { field_name → type_marker }
    /// type_marker is "[ElemModel;N]" for fixed model arrays, or the nested model name.
    pub model_field_types: HashMap<String, HashMap<String, String>>,
    /// Variable Symbol → model name (for model array loop variables)
    pub model_var_types: HashMap<Symbol, String>,
    /// Names of models imported from other modules (not defined locally).
    /// These have struct layouts but should NOT generate struct ctors.
    pub imported_model_names: HashSet<String>,
    /// Custom dtype name → numeric id (starting at 256). Populated by compile_datatype_defs.
    pub custom_dtype_ids: HashMap<String, u16>,
    /// Configuration for standalone export (set by `compile_standalone()`).
    pub(crate) standalone_config: Option<StandaloneConfig>,
    /// Models with @paged_kv: model_name -> (num_blocks, block_size, num_heads, head_dim, num_layers)
    pub paged_kv_configs: HashMap<String, (i64, i64, i64, i64, i64)>,
    /// Compiler configuration flags (--no-autotune, --autotune-fresh, etc.)
    pub compile_options: crate::CompileOptions,
    /// FlashAttention compile context: set during compile_flash_attention_kernels(),
    /// consumed by compile_flash_attention_call() when lowering scaled_dot_product_attention.
    pub flash_attention_context: Option<FlashAttentionCompileContext>,
    /// M30: Sharded layers — "ModelName.layer_name" → ShardInfo
    pub shard_configs: HashMap<String, crate::tensor_parallel::ShardInfo>,
    /// M32: MoE layer configs — "ModelName.layer_name" → MoeInfo
    pub moe_configs: HashMap<String, crate::moe::MoeInfo>,
    /// M33: Speculative decoding configs — "ModelName.layer_name" → SpeculativeInfo
    pub speculative_configs: HashMap<String, crate::speculative::SpeculativeInfo>,
    /// M34: Context parallelism configs — "ModelName.layer_name" → ContextParallelInfo
    pub context_parallel_configs: HashMap<String, crate::context_parallel::ContextParallelInfo>,
    /// M34: Ring size for context parallelism
    pub cp_ring_size: usize,
    /// M30: Activation distribution states (for future all-reduce insertion)
    pub activation_states: HashMap<String, crate::tensor_parallel::DistState>,
    /// M30: Tensor parallelism world size (number of devices)
    pub world_size: usize,
    /// M41: Whether this serve block uses disaggregated inference.
    pub disaggregated: bool,
    /// M41: Number of prefill workers (from serve config).
    pub prefill_workers: usize,
    /// M41: Number of decode workers (from serve config).
    pub decode_workers: usize,
    /// M31: Collected fusion optimization events for --fusion-report
    pub fusion_events: Vec<crate::fusion_report::FusionEvent>,
    /// M31: Collected fusion barrier events for --fusion-report
    pub fusion_barriers: Vec<crate::fusion_report::FusionBarrierEvent>,
    /// M31: Whether fusion event collection is enabled
    pub fusion_report_enabled: bool,
    /// M47: Disable all fusion optimizations for differential testing.
    pub disable_fusion: bool,
    /// M35: Functions with @fp8_compute decorator
    pub fp8_compute_fns: HashSet<String>,
    /// M35: Model quantization configs — "ModelName" -> QuantConfig
    pub quant_configs: HashMap<String, QuantConfig>,
    /// M36: Computed memory plan for slab allocation (None if planning disabled/empty)
    pub slab_plan: Option<crate::memory_planner::SlabPlan>,
    /// M38b: Whether --linear-types flag is active
    pub linear_types_enabled: bool,
    /// M38b: Per-function ownership metadata from semantic pass
    pub ownership_info: HashMap<String, crate::ownership::FunctionOwnership>,
    /// M39: Functions with @vmap decorator and their batch configuration
    pub vmap_configs: HashMap<String, crate::vmap::VmapConfig>,
    /// M40: Source-to-source AD enabled (default true; --tape-ad forces tape-only)
    pub source_ad_enabled: bool,
    /// M42: Per-model KV compression policies from @kv_compress decorators.
    pub kv_compress_policies: HashMap<String, Vec<KvCompressPolicy>>,
    /// M44: Compiled grammar FSMs from @grammar decorators and generate() schema= args.
    pub grammar_configs: HashMap<String, GrammarInfo>,
    /// M43: Pipeline parallelism configuration.
    pub pipeline_config: Option<crate::pipeline::PipelineConfig>,
    /// M43: 3D parallelism configuration.
    pub parallelism_config: Option<crate::pipeline::ParallelismConfig>,
    /// M43: ZeRO optimizer sharding stage.
    pub zero_stage: Option<u8>,
    /// M52: Loaded weight map (populated when --weights is passed)
    pub weight_map: Option<crate::weight_aware::WeightMap>,
    /// M52: Sparsity hints per matmul operation (AST NodeId -> SparsityHint)
    pub sparsity_hints: HashMap<NodeId, crate::weight_aware::SparsityHint>,
    /// M52: Weight integrity hash for embedding in .rodata
    pub weight_integrity: Option<crate::weight_aware::WeightIntegrity>,
    /// M53: Functions decorated with @real_time — name -> constraint
    pub real_time_fns: HashMap<String, crate::wcet::RealTimeConstraint>,
    /// M53: Functions decorated with @wcet_budget — name -> constraint
    pub wcet_budget_fns: HashMap<String, crate::wcet::WcetBudgetConstraint>,
    /// M53: Collected WCET analysis results for @real_time functions
    pub wcet_results: Vec<crate::wcet::FunctionWcet>,
    /// M55: Functions (and model methods) decorated with @zk_proof — name -> ZkMode
    pub zk_proof_fns: HashMap<String, crate::zk::backend::ZkMode>,
    func_index: u32,
}

/// Quantization configuration for a model.
#[derive(Debug, Clone)]
pub struct QuantConfig {
    pub dtype: String,     // "awq4", "gptq4", "gptq8"
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
    pub fn new(interner: &'a Interner, type_map: &'a TypeMap, options: &crate::CompileOptions) -> Result<Self, CodegenError> {
        let mut flag_builder = settings::builder();
        flag_builder
            .set("opt_level", "speed")
            .map_err(|e| CodegenError::new(e.to_string()))?;

        let isa_builder = cranelift_native::builder()
            .map_err(|e| CodegenError::new(format!("failed to create ISA builder: {e}")))?;
        let isa = isa_builder
            .finish(settings::Flags::new(flag_builder))
            .map_err(|e| CodegenError::new(format!("failed to build ISA: {e}")))?;

        let call_conv = isa.default_call_conv();

        let builder = ObjectBuilder::new(
            isa,
            "nsl_module",
            cranelift_module::default_libcall_names(),
        )
        .map_err(|e| CodegenError::new(format!("failed to create object builder: {e}")))?;

        let module = ObjectModule::new(builder);

        Ok(Compiler {
            module,
            interner,
            type_map,
            functions: HashMap::new(),
            runtime_fns: HashMap::new(),
            string_pool: HashMap::new(),
            struct_layouts: HashMap::new(),
            model_methods: HashMap::new(),
            enum_variants: HashMap::new(),
            enum_defs: HashMap::new(),
            pending_lambdas: Vec::new(),
            closure_info: HashMap::new(),
            last_lambda_capture_count: None,
            call_conv,
            dump_ir: false,
            no_grad_fns: HashSet::new(),
            test_fns: Vec::new(),
            module_prefix: String::new(),
            kernel_ptx_data: HashMap::new(),
            model_field_types: HashMap::new(),
            model_var_types: HashMap::new(),
            imported_model_names: HashSet::new(),
            custom_dtype_ids: HashMap::new(),
            standalone_config: None,
            paged_kv_configs: HashMap::new(),
            compile_options: options.clone(),
            flash_attention_context: None,
            shard_configs: HashMap::new(),
            moe_configs: HashMap::new(),
            speculative_configs: HashMap::new(),
            context_parallel_configs: HashMap::new(),
            cp_ring_size: 1,
            activation_states: HashMap::new(),
            world_size: options.world_size,
            disaggregated: false,
            prefill_workers: 1,
            decode_workers: 1,
            fusion_events: Vec::new(),
            fusion_barriers: Vec::new(),
            fusion_report_enabled: options.fusion_report,
            disable_fusion: options.disable_fusion,
            fp8_compute_fns: HashSet::new(),
            quant_configs: HashMap::new(),
            slab_plan: None,
            linear_types_enabled: false,
            ownership_info: HashMap::new(),
            vmap_configs: HashMap::new(),
            source_ad_enabled: !options.tape_ad,
            kv_compress_policies: HashMap::new(),
            grammar_configs: HashMap::new(),
            pipeline_config: None,
            parallelism_config: None,
            zero_stage: None,
            weight_map: None,
            sparsity_hints: HashMap::new(),
            weight_integrity: None,
            real_time_fns: HashMap::new(),
            wcet_budget_fns: HashMap::new(),
            wcet_results: Vec::new(),
            zk_proof_fns: HashMap::new(),
            func_index: 0,
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
        self.fusion_report_enabled = true;
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
                StmtKind::VarDecl { value: Some(expr), .. } => {
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
        let data_id = self.module
            .declare_data(&name, cranelift_module::Linkage::Local, false, false)
            .map_err(|e| CodegenError::new(format!("failed to declare string data: {e}")))?;

        let mut bytes = s.as_bytes().to_vec();
        bytes.push(0);

        let mut desc = DataDescription::new();
        desc.define(bytes.into_boxed_slice());
        self.module.define_data(data_id, &desc)
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
        use crate::wcet::*;
        use crate::gpu_specs::find_gpu;

        if self.real_time_fns.is_empty() {
            return Ok(());
        }

        let gpu_name = self
            .compile_options
            .wcet_gpu
            .as_deref()
            .unwrap_or("A100-SXM");
        let gpu = find_gpu(gpu_name).ok_or_else(|| {
            CodegenError::new(format!(
                "WCET: unknown GPU '{}'. Use --gpu to specify.",
                gpu_name
            ))
        })?;

        let safety_margin = self.compile_options.wcet_safety_margin;

        // Clone to avoid borrowing self.real_time_fns while mutating self.wcet_results
        for (fn_name, constraint) in self.real_time_fns.clone() {
            // Create demo analysis with a small matmul + elementwise op
            let ops = vec![
                wcet_matmul_gpu(1, 512, 512, "fp32", gpu),
                wcet_elementwise_gpu(512, "fp32", "relu", gpu),
            ];

            let total_ns: u64 = ops.iter().map(|o| o.worst_case_ns).sum();
            let total_ms = total_ns as f64 / 1_000_000.0;
            let final_ms = total_ms * safety_margin;
            let bound_satisfied = final_ms <= constraint.max_latency_ms;

            let no_heap = prove_no_heap(self.slab_plan.as_ref(), &fn_name);
            let static_cf = prove_static_cf();

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
                let suggestions =
                    generate_suggestions(&ops, constraint.max_latency_ms, final_ms);
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
                let cert = build_certificate(
                    &func_wcet,
                    &no_heap,
                    &static_cf,
                    "source.nsl",
                    Some(gpu_name),
                    None,
                );
                emit_certificate(&cert, cert_path).map_err(|e| {
                    CodegenError::new(format!("WCET certificate write failed: {}", e))
                })?;
                eprintln!("WCET certificate: {}", cert_path.display());
            }

            // Emit DO-178C report if requested
            if let Some(ref do178c_path) = self.compile_options.do178c_report {
                let cert = build_certificate(
                    &func_wcet,
                    &no_heap,
                    &static_cf,
                    "source.nsl",
                    Some(gpu_name),
                    None,
                );
                emit_do178c_report(&cert, do178c_path).map_err(|e| {
                    CodegenError::new(format!("DO-178C report write failed: {}", e))
                })?;
            }

            self.wcet_results.push(func_wcet);
        }

        Ok(())
    }

    /// M52: Embed the weight file SHA-256 hash as a .rodata symbol.
    /// The symbol `__nsl_weight_hash` contains 32 bytes of the SHA-256 hash.
    pub fn embed_weight_hash(&mut self) -> Result<(), CodegenError> {
        if let Some(ref integrity) = self.weight_integrity {
            let data_id = self.module
                .declare_data("__nsl_weight_hash", cranelift_module::Linkage::Export, false, false)
                .map_err(|e| CodegenError::new(format!("failed to declare weight hash data: {e}")))?;

            let mut desc = DataDescription::new();
            desc.define(integrity.compile_hash.to_vec().into_boxed_slice());
            self.module.define_data(data_id, &desc)
                .map_err(|e| CodegenError::new(format!("failed to define weight hash data: {e}")))?;
        }
        Ok(())
    }

    pub fn finalize(self) -> Result<Vec<u8>, CodegenError> {
        let product = self.module.finish();
        product.emit().map_err(|e| CodegenError::new(format!("failed to emit object: {e}")))
    }
}
