//! M29: Serve block codegen.

use cranelift_codegen::ir::{types as cl_types, InstBuilder};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module as _;

use nsl_ast::block::ServeBlock;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::speculative::SpeculativeMethod;

const WORKER_CONFIG_SIZE: u32 = 48;
const WORKER_CONFIG_ALIGN: u8 = 8;
const WORKER_DEFAULT_BLOCK_SIZE: i64 = 16;
const WORKER_DEFAULT_NUM_KV_HEADS: i64 = 32;
const WORKER_DEFAULT_HEAD_DIM: i64 = 128;
const WORKER_DEFAULT_NUM_LAYERS: i64 = 32;
const WORKER_DEFAULT_EOS_TOKEN_ID: i64 = 2;

const WORKER_MAX_SEQ_LEN_OFFSET: i32 = 0;
const WORKER_KV_BLOCKS_OFFSET: i32 = 4;
const WORKER_BLOCK_SIZE_OFFSET: i32 = 8;
const WORKER_NUM_KV_HEADS_OFFSET: i32 = 12;
const WORKER_HEAD_DIM_OFFSET: i32 = 16;
const WORKER_NUM_LAYERS_OFFSET: i32 = 20;
const WORKER_SPEC_TOKENS_OFFSET: i32 = 24;
const WORKER_SPEC_METHOD_OFFSET: i32 = 28;
const WORKER_SPEC_TREE_WIDTH_OFFSET: i32 = 32;
const WORKER_SPEC_TEMP_BITS_OFFSET: i32 = 36;
const WORKER_EOS_TOKEN_OFFSET: i32 = 40;

/// Values the compiled serve binary passes to the `nsl_cfie_*` init
/// FFIs, sized from the compile-time CFIE plan.
struct CfieRuntimeInit {
    /// `(slot_count, per_slot_tokens)` for the KV slot free-list —
    /// present exactly when the decode block was emitted.
    kv_slots: Option<(i64, i64)>,
    /// Kernel registrations (`cfie::kernel_registrations`).  Each entry
    /// becomes one `nsl_cfie_register_kernel` call at serve init.
    kernels: Vec<crate::cfie::CfieKernelReg>,
    /// Device KV-pool bytes for `nsl_cfie_kv_pool_alloc` — `Some`
    /// exactly when the decode block was emitted: `n_layers * 2 *
    /// (max_slots * per_slot_tokens) * n_kv_heads * head_dim * 2` (f16).
    pool_bytes: Option<i64>,
    /// Model-binding + generation parameters resolved from the serve
    /// config + the compile-time model shape.  `Some` exactly when the
    /// decode path (decode block + fused sampler) is in the build.
    endpoint: Option<CfieEndpointInit>,
    /// Why `generate()` cannot run in this build, when `endpoint` is
    /// `None`: a `generate()` call is then a compile error with this
    /// text.
    generate_refusal: Option<String>,
}

/// The most static shared memory one thread block may declare, in bytes.
const CFIE_STATIC_SMEM_LIMIT: u32 = 48 * 1024;

/// Cycle 11: everything the monolithic serve path needs to emit the
/// model-binding + generation-driver calls, resolved at compile time.
struct CfieEndpointInit {
    /// Absolute path to the `.safetensors` weights, from the serve
    /// `weights:` key or (fallback) the `--weights` CLI path.  Empty
    /// string when neither is set — the emitted `nsl_model_create` then
    /// receives an empty path and returns a null handle, and the runtime
    /// `nsl_cfie_bind_model` refuses it cleanly (documented boundary:
    /// there is no served model without a weights path).
    weights_path: String,
    /// Optional tokenizer path for the runtime prompt-encode + the
    /// decode-and-print tail (Cycle 12).
    tokenizer_path: Option<String>,
    /// The FALLBACK prompt token ids, baked at compile time (byte-level
    /// ids clamped into the vocab; `[0]` sentinel when no `prompt:` is
    /// configured).  When a tokenizer + prompt are both configured the
    /// serve init ALSO emits the runtime encode
    /// (`nsl_tokenizer_encode` -> `nsl_cfie_tensor_to_tokens`) and a
    /// Cranelift `select` picks the runtime-encoded buffer whenever the
    /// encode produced tokens (n > 0), falling back to these baked ids
    /// otherwise — so the binary still runs if the encode comes back
    /// empty.  The runtime `nsl_cfie_generate` prompt ABI is a host i64
    /// array, so these are baked as raw i64s in .rodata.
    prompt_tokens: Vec<i64>,
    /// The raw `prompt:` string for the runtime tokenizer encode
    /// (Cycle 12).  `None` when no non-empty prompt is configured — the
    /// baked sentinel is then the only prompt source.
    prompt_text: Option<String>,
    /// Resolved model shape (passed to `nsl_cfie_bind_model`).
    shape: crate::cfie_serve::ResolvedModelShape,
    max_new_tokens: i64,
    eos_token_id: i64,
    /// The `sampling: seed:` value (default 0).
    rng_seed: i64,
    /// Cycle 13 (G15): the resolved draft-model wiring — `Some` exactly
    /// when the plan emitted the two draft kernels.  Serve init
    /// then ALSO emits `nsl_model_create(draft) ->
    /// nsl_cfie_bind_draft_model -> nsl_cfie_draft_pool_alloc` after
    /// the target binding, and `generate()` rewrites to
    /// `nsl_cfie_speculative_generate` (k = `k_tokens`).
    draft: Option<crate::cfie::CfieSpeculativeDraft>,
}

/// CFIE Cycle 11 side-channel: the `generate()` intrinsic's driver
/// parameters, published on `Compiler::cfie_serve_gen` while a
/// CFIE-active serve body compiles.  Presence of this value is exactly
/// the "CFIE serve context" the `generate()` rewrite tests for.
pub struct CfieServeGen {
    /// `Some(why)` when this build has no decode path: the serve block is
    /// CFIE-active, but a `generate()` call must be refused at compile
    /// time with this reason (the other fields are then unused).
    pub refusal: Option<String>,
    pub max_new_tokens: i64,
    pub eos_token_id: i64,
    /// The `sampling: seed:` value (default 0).  The runtime keys each
    /// request's draws by it, the request's index and the position.
    pub rng_seed: i64,
    /// FALLBACK prompt token count baked at serve init (host i64 array
    /// length) — used when `prompt_len_val` is `None`.
    pub prompt_len: i64,
    /// Cycle 12: the RUNTIME prompt length (a Cranelift value selected
    /// at serve init between the tokenizer-encoded count and the baked
    /// length).  `Some` exactly when tokenizer + prompt are configured;
    /// `generate()` then passes this instead of the baked constant.
    /// Valid because serve init and the endpoint bodies compile into the
    /// SAME Cranelift function (the init block dominates the bodies).
    pub prompt_len_val: Option<cranelift_codegen::ir::Value>,
    /// Cycle 12: the tokenizer handle loaded at serve init.  `Some`
    /// exactly when `tokenizer:` is configured; `generate()` then emits
    /// the decode-and-print tail (tokens -> tensor -> text -> stdout).
    pub tok_handle: Option<cranelift_codegen::ir::Value>,
    /// Cycle 13 (G15): `Some(k)` exactly when a speculative draft model
    /// is compiled into the binary — `generate()` then emits
    /// `nsl_cfie_speculative_generate(..., k, ...)` instead of
    /// `nsl_cfie_generate`.
    pub speculative_k: Option<i64>,
}

/// What `emit_cfie_endpoint_init` hands back to `compile_serve_block`:
/// the (possibly runtime-selected) prompt pointer the endpoint's first
/// param binds to, the runtime prompt length when the Cycle-12 encode
/// path was emitted, and the tokenizer handle when one was loaded.
struct CfieEndpointWiring {
    prompt_ptr: cranelift_codegen::ir::Value,
    prompt_len_val: Option<cranelift_codegen::ir::Value>,
    tok_handle: Option<cranelift_codegen::ir::Value>,
}

struct WorkerConfigSpec {
    max_seq_len: i64,
    kv_blocks: i64,
    speculative_tokens: i64,
    speculative_method: i64,
    speculative_tree_width: i64,
    speculative_temperature_bits: i64,
}

/// Embed raw bytes in .rodata and return the DataId (the
/// `Compiler::embed_raw_data` pattern, kernel.rs:1609 — duplicated here
/// because that helper is private to `compiler::kernel`).  Callers own
/// NUL termination: push the trailing 0 BEFORE calling when the bytes
/// feed a C-string consumer.
fn embed_cfie_bytes(
    module: &mut cranelift_object::ObjectModule,
    label: &str,
    bytes: Vec<u8>,
) -> Result<cranelift_module::DataId, CodegenError> {
    let data_id = module
        .declare_data(label, cranelift_module::Linkage::Local, false, false)
        .map_err(|e| CodegenError::new(format!("failed to declare CFIE data '{label}': {e}")))?;
    let mut desc = cranelift_module::DataDescription::new();
    desc.define(bytes.into_boxed_slice());
    module
        .define_data(data_id, &desc)
        .map_err(|e| CodegenError::new(format!("failed to define CFIE data '{label}': {e}")))?;
    Ok(data_id)
}

/// NUL-terminate `s` for .rodata embedding and return
/// `(bytes_with_nul, len_without_nul)`.  The runtime copies
/// `[ptr, ptr + len)` and appends its OWN terminator, so the registered
/// length must EXCLUDE the NUL (PR #251 missing-NUL precedent).
fn cfie_nul_terminated(s: &str) -> (Vec<u8>, i64) {
    let mut bytes = s.as_bytes().to_vec();
    bytes.push(0);
    let len = (bytes.len() - 1) as i64;
    (bytes, len)
}

/// Sanitize a serve-block name into a data-section label fragment so
/// two CFIE-active serve blocks in one module get distinct labels
/// (same-name blocks still collide loudly via `DuplicateDefinition`).
fn cfie_label_scope(name: &str) -> String {
    let s: String = name
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() { c } else { '_' })
        .collect();
    if s.is_empty() { "serve".to_string() } else { s }
}

impl Compiler<'_> {
    fn build_worker_config_slot(
        &self,
        builder: &mut FunctionBuilder,
        spec: WorkerConfigSpec,
    ) -> cranelift_codegen::ir::StackSlot {
        let slot = builder.create_sized_stack_slot(cranelift_codegen::ir::StackSlotData::new(
            cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
            WORKER_CONFIG_SIZE,
            WORKER_CONFIG_ALIGN,
        ));

        let max_seq_len_val = builder.ins().iconst(cl_types::I32, spec.max_seq_len);
        let kv_blocks_val = builder.ins().iconst(cl_types::I32, spec.kv_blocks);
        let block_size_val = builder
            .ins()
            .iconst(cl_types::I32, WORKER_DEFAULT_BLOCK_SIZE);
        let num_kv_heads_val = builder
            .ins()
            .iconst(cl_types::I32, WORKER_DEFAULT_NUM_KV_HEADS);
        let head_dim_val = builder.ins().iconst(cl_types::I32, WORKER_DEFAULT_HEAD_DIM);
        let num_layers_val = builder
            .ins()
            .iconst(cl_types::I32, WORKER_DEFAULT_NUM_LAYERS);
        let speculative_tokens_val = builder.ins().iconst(cl_types::I32, spec.speculative_tokens);
        let speculative_method_val = builder.ins().iconst(cl_types::I32, spec.speculative_method);
        let speculative_tree_width_val =
            builder.ins().iconst(cl_types::I32, spec.speculative_tree_width);
        let speculative_temp_bits_val = builder
            .ins()
            .iconst(cl_types::I32, spec.speculative_temperature_bits);
        let eos_val = builder
            .ins()
            .iconst(cl_types::I32, WORKER_DEFAULT_EOS_TOKEN_ID);

        builder
            .ins()
            .stack_store(cl_types::I64, max_seq_len_val, slot, WORKER_MAX_SEQ_LEN_OFFSET);
        builder
            .ins()
            .stack_store(cl_types::I64, kv_blocks_val, slot, WORKER_KV_BLOCKS_OFFSET);
        builder
            .ins()
            .stack_store(cl_types::I64, block_size_val, slot, WORKER_BLOCK_SIZE_OFFSET);
        builder
            .ins()
            .stack_store(cl_types::I64, num_kv_heads_val, slot, WORKER_NUM_KV_HEADS_OFFSET);
        builder
            .ins()
            .stack_store(cl_types::I64, head_dim_val, slot, WORKER_HEAD_DIM_OFFSET);
        builder
            .ins()
            .stack_store(cl_types::I64, num_layers_val, slot, WORKER_NUM_LAYERS_OFFSET);
        builder
            .ins()
            .stack_store(cl_types::I64, speculative_tokens_val, slot, WORKER_SPEC_TOKENS_OFFSET);
        builder
            .ins()
            .stack_store(cl_types::I64, speculative_method_val, slot, WORKER_SPEC_METHOD_OFFSET);
        builder.ins().stack_store(
            cl_types::I64,
            speculative_tree_width_val,
            slot,
            WORKER_SPEC_TREE_WIDTH_OFFSET,
        );
        builder.ins().stack_store(
            cl_types::I64,
            speculative_temp_bits_val,
            slot,
            WORKER_SPEC_TEMP_BITS_OFFSET,
        );
        builder
            .ins()
            .stack_store(cl_types::I64, eos_val, slot, WORKER_EOS_TOKEN_OFFSET);

        slot
    }

    pub fn compile_serve_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        serve: &ServeBlock,
    ) -> Result<(), CodegenError> {
        // M30: Initialize tensor parallelism if multi-device
        if self.features.world_size > 1 {
            self.compile_call_by_name(builder, "nsl_tp_init", &[])?;
        }

        // Extract config values with defaults
        let mut max_batch: i64 = 32;
        let mut max_seq_len: i64 = 4096;
        let mut kv_blocks: i64 = 2048;
        let mut prefill_chunk: i64 = 512;
        let mut prefill_workers: i64 = 1;
        let mut decode_workers: i64 = 1;
        let mut kv_transfer_backend: String = "auto".to_string();

        for entry in &serve.config {
            let key_name = self.resolve_sym(entry.key).to_string();
            match key_name.as_str() {
                "max_batch" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        max_batch = *v;
                    }
                }
                // CFIE: the paper spells this key `max_seq`; accept both.
                "max_seq_len" | "max_seq" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        max_seq_len = *v;
                    }
                }
                "kv_blocks" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        kv_blocks = *v;
                    }
                }
                "prefill_chunk" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        prefill_chunk = *v;
                    }
                }
                "prefill_workers" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        prefill_workers = *v;
                    }
                }
                "decode_workers" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &entry.value.kind {
                        decode_workers = *v;
                    }
                }
                "kv_transfer" => {
                    if let nsl_ast::expr::ExprKind::StringLiteral(s) = &entry.value.kind {
                        kv_transfer_backend = s.clone();
                    }
                }
                // CFIE config keys: consumed by `run_cfie_for_serve`
                // below — not runtime expressions.  The Cycle 11
                // endpoint-wiring keys (weights/tokenizer/prompt/
                // max_new_tokens/eos_token_id) are string/int literals
                // consumed by the CFIE serve-init emission, so they
                // must be skipped here too or `compile_expr` would treat
                // e.g. `weights` as a bare variable reference.
                "kv_layout" | "kv_quant" | "target_gpu" | "n_layers" | "n_kv_heads"
                | "kv_heads" | "n_heads" | "head_dim" | "d_model" | "d_ff" | "vocab_size"
                | "rope_theta" | "norm_eps" | "weights" | "tokenizer" | "prompt"
                | "max_new_tokens" | "eos_token_id" => {}
                _ => {
                    self.compile_expr(builder, state, &entry.value)?;
                }
            }
        }

        // Override from CLI flags if set
        if self.features.prefill_workers > 1 {
            prefill_workers = self.features.prefill_workers as i64;
        }
        if self.features.decode_workers > 1 {
            decode_workers = self.features.decode_workers as i64;
        }

        let is_disaggregated = prefill_workers > 1 || decode_workers > 1;

        // ── CFIE Tier-A wiring (audit gap G1) ────────────────────────
        // Extract CFIE config from the serve block, resolve the mode
        // (CLI --cfie > @cfie decorator > implicit via CFIE keys), run
        // the orchestrator, and surface the build report.  The plan
        // drives the report, the KV-slot init call, and the kernel
        // registration/finalize emission — the
        // report must know the path up front because the disaggregated
        // branch emits NONE of that runtime wiring.
        let cfie_init = self.run_cfie_for_serve(serve, is_disaggregated)?;

        if is_disaggregated {
            self.compile_disaggregated_serve(
                builder,
                state,
                serve,
                max_batch,
                max_seq_len,
                kv_blocks,
                prefill_chunk,
                prefill_workers,
                decode_workers,
                &kv_transfer_backend,
            )?;
        } else {
            // Monolithic M29 path (unchanged)
            let v_max_batch = builder.ins().iconst(cl_types::I64, max_batch);
            let v_max_seq_len = builder.ins().iconst(cl_types::I64, max_seq_len);
            let v_kv_blocks = builder.ins().iconst(cl_types::I64, kv_blocks);
            let v_prefill_chunk = builder.ins().iconst(cl_types::I64, prefill_chunk);

            self.compile_call_by_name(
                builder,
                "nsl_serve_init",
                &[v_max_batch, v_max_seq_len, v_kv_blocks, v_prefill_chunk],
            )?;

            // CFIE Cycle 11: the baked prompt-token host i64 array
            // pointer, produced by `emit_cfie_endpoint_init` and bound to
            // the endpoint's first param below.  `None` when CFIE is
            // inactive or the decode path was not wired.
            let mut cfie_prompt_ptr: Option<cranelift_codegen::ir::Value> = None;

            // CFIE: size the KV sequence-slot free-list from the
            // compile-time plan.  Monolithic path only; the
            // disaggregated workers keep their M41 queueing.
            if let Some(init) = &cfie_init {
                if let Some((slot_count, per_slot_tokens)) = init.kv_slots {
                    let v_slots = builder.ins().iconst(cl_types::I64, slot_count);
                    let v_tokens = builder.ins().iconst(cl_types::I64, per_slot_tokens);
                    self.compile_call_by_name(
                        builder,
                        "nsl_cfie_kv_slots_init",
                        &[v_slots, v_tokens],
                    )?;
                }
                // Register the compiled kernels, allocate the device KV
                // pool, and finalize the engine (one cuModuleLoadData +
                // cuModuleGetFunction per kernel) before any endpoint
                // body runs.  Return values are intentionally ignored —
                // the runtime warns on stderr and every launch FFI
                // refuses cleanly (-1) when finalize did not succeed.
                if !init.kernels.is_empty() {
                    let scope = cfie_label_scope(self.resolve_sym(serve.name));
                    for reg in &init.kernels {
                        self.emit_cfie_register_kernel(builder, &scope, reg)?;
                    }
                    if let Some(bytes) = init.pool_bytes {
                        let v_bytes = builder.ins().iconst(cl_types::I64, bytes);
                        self.compile_call_by_name(
                            builder,
                            "nsl_cfie_kv_pool_alloc",
                            &[v_bytes],
                        )?;
                    }
                    self.compile_call_by_name(builder, "nsl_cfie_engine_finalize", &[])?;
                }
                // CFIE Cycle 11: after the engine is finalized, bind the
                // served model (nsl_model_create + nsl_cfie_bind_model),
                // optionally load the tokenizer, bake the prompt token
                // array, and publish the `generate()` driver params.  The
                // baked prompt ptr becomes the endpoint's first param.
                if let Some(ep) = init.endpoint.as_ref() {
                    let scope = cfie_label_scope(self.resolve_sym(serve.name));
                    let wired = self.emit_cfie_endpoint_init(builder, &scope, ep)?;
                    self.bus.publish_cfie_serve_gen(CfieServeGen {
                        refusal: None,
                        max_new_tokens: ep.max_new_tokens,
                        eos_token_id: ep.eos_token_id,
                        rng_seed: ep.rng_seed,
                        prompt_len: ep.prompt_tokens.len() as i64,
                        prompt_len_val: wired.prompt_len_val,
                        tok_handle: wired.tok_handle,
                        // Cycle 13: a compiled-in draft flips generate()
                        // to the speculative driver with this K.
                        speculative_k: ep.draft.as_ref().map(|d| d.k_tokens as i64),
                    });
                    cfie_prompt_ptr = Some(wired.prompt_ptr);
                } else if let Some(why) = init.generate_refusal.clone() {
                    // CFIE is active but the decode path is not in the
                    // build: a `generate()` in an endpoint body is a
                    // compile error that says why.
                    self.bus.publish_cfie_serve_gen(CfieServeGen {
                        refusal: Some(why),
                        max_new_tokens: 0,
                        eos_token_id: -1,
                        rng_seed: 0,
                        prompt_len: 0,
                        prompt_len_val: None,
                        tok_handle: None,
                        speculative_k: None,
                    });
                }
            }

            // CFIE Cycle 11: bind each endpoint's params as real
            // function-scope locals so the body can reference them
            // (mirrors func.rs::compile_fn_def_named param binding).
            // Since a one-shot serve binary has no caller, the FIRST
            // param (the prompt) is initialised to the baked prompt-token
            // host i64 array pointer; any remaining params are bound to a
            // defined 0 sentinel (DEFERRED: live per-request param
            // sourcing needs the request loop this v1 does not build).
            for endpoint in &serve.endpoints {
                for (i, param) in endpoint.params.iter().enumerate() {
                    let init_val = if i == 0 {
                        cfie_prompt_ptr
                            .unwrap_or_else(|| builder.ins().iconst(cl_types::I64, 0))
                    } else {
                        builder.ins().iconst(cl_types::I64, 0)
                    };
                    let var = builder.declare_var(cl_types::I64);
                    builder.def_var(var, init_val);
                    state.variables.insert(param.name, (var, cl_types::I64));
                    state.param_symbols.insert(param.name);
                }
                // Endpoint bodies are checker-scoped; scope their
                // fn-value bindings (review MEDIUM on 682641ca, same
                // class as the grad-body gap).
                state.push_fn_binding_scope();
                for stmt in &endpoint.body.stmts {
                    self.compile_stmt(builder, state, stmt)?;
                }
                state.pop_fn_binding_scope();
            }

            // CFIE Cycle 11: the serve body is done — the `generate()`
            // side-channel must not leak into any function compiled
            // after this serve block.
            self.bus.clear_cfie_serve_gen();

            // CFIE Cycle 6: free the pool + clear registrations before
            // the serve runtime tears down (CUmodules may stay loaded —
            // module leak-by-design precedent, the runtime's context module cache).
            if cfie_init.as_ref().is_some_and(|i| !i.kernels.is_empty()) {
                self.compile_call_by_name(builder, "nsl_cfie_engine_destroy", &[])?;
            }
            self.compile_call_by_name(builder, "nsl_serve_destroy", &[])?;
        }

        // M30: Tear down tensor parallelism if multi-device
        if self.features.world_size > 1 {
            self.compile_call_by_name(builder, "nsl_tp_destroy", &[])?;
        }

        Ok(())
    }

    /// CFIE Cycle 6: emit one `nsl_cfie_register_kernel` call.  PTX and
    /// kernel-name bytes are embedded in .rodata WITH a trailing NUL
    /// (cuModuleLoadData / cuModuleGetFunction consume C strings — the
    /// PR #251 missing-NUL precedent); the registered lengths EXCLUDE
    /// the NUL per the frozen ABI.  Data labels follow the contract,
    /// scoped by serve-block name so two CFIE-active serve blocks don't
    /// collide: `__nsl_cfie_{ptx,name}_{scope}_{kind}`.
    fn emit_cfie_register_kernel(
        &mut self,
        builder: &mut FunctionBuilder,
        label_scope: &str,
        reg: &crate::cfie::CfieKernelReg,
    ) -> Result<(), CodegenError> {
        let suffix = format!("{label_scope}_{}", reg.kind);

        let (ptx_bytes, ptx_len) = cfie_nul_terminated(&reg.ptx);
        let ptx_id = embed_cfie_bytes(
            &mut self.module,
            &format!("__nsl_cfie_ptx_{suffix}"),
            ptx_bytes,
        )?;

        let (name_bytes, name_len) = cfie_nul_terminated(&reg.name);
        let name_id = embed_cfie_bytes(
            &mut self.module,
            &format!("__nsl_cfie_name_{suffix}"),
            name_bytes,
        )?;

        let ptx_gv = self.module.declare_data_in_func(ptx_id, builder.func);
        let ptx_ptr = builder.ins().symbol_value(cl_types::I64, ptx_gv);
        let name_gv = self.module.declare_data_in_func(name_id, builder.func);
        let name_ptr = builder.ins().symbol_value(cl_types::I64, name_gv);

        let v_kind = builder.ins().iconst(cl_types::I64, reg.kind as i64);
        let v_layer = builder.ins().iconst(cl_types::I64, reg.layer_idx as i64);
        let v_ptx_len = builder.ins().iconst(cl_types::I64, ptx_len);
        let v_name_len = builder.ins().iconst(cl_types::I64, name_len);
        let v_grid = builder.ins().iconst(cl_types::I64, reg.grid_x as i64);
        let v_block = builder.ins().iconst(cl_types::I64, reg.block_x as i64);
        let v_smem = builder.ins().iconst(cl_types::I64, reg.smem_dyn as i64);
        self.compile_call_by_name(
            builder,
            "nsl_cfie_register_kernel",
            &[
                v_kind, v_layer, ptx_ptr, v_ptx_len, name_ptr, v_name_len, v_grid,
                v_block, v_smem,
            ],
        )?;
        Ok(())
    }

    /// CFIE Cycle 11 (+12): emit the model-binding + tokenizer-load +
    /// prompt wiring, and return the values the endpoint bodies consume
    /// (prompt pointer/length + tokenizer handle).
    ///
    /// Emission order (AFTER `nsl_cfie_engine_finalize`):
    ///   1. `model_handle = nsl_model_create(weights_path_cstr)`.  A null
    ///      handle (missing/failed weights) is well-formed: the runtime
    ///      `nsl_cfie_bind_model` refuses a null handle cleanly.
    ///   2. `nsl_cfie_bind_model(model_handle, n_layers, d_model,
    ///      n_heads, n_kv_heads, head_dim, d_ff, vocab_size)` with the
    ///      resolved-shape constants.
    ///   3. `tok_handle = nsl_tokenizer_load(tokenizer_path_cstr)` when a
    ///      tokenizer is configured.  Cycle 12: the handle is KEPT — it
    ///      feeds the runtime prompt encode below and the `generate()`
    ///      decode-and-print tail (calls.rs).
    ///   4. bake the FALLBACK prompt token ids as a raw host i64 array
    ///      in .rodata.
    ///   5. Cycle 12, when tokenizer + prompt are both configured: emit
    ///      the RUNTIME prompt encode —
    ///        enc  = nsl_tokenizer_encode(tok_handle, prompt_text_cstr)
    ///        n    = nsl_cfie_tensor_to_tokens(enc, prompt_buf, cap)
    ///        nsl_tensor_free(enc)
    ///      then a Cranelift `select` on `n > 0` picks
    ///      `(prompt_buf, min(n, cap))` over the baked `(ptr, len)` — a
    ///      RUNTIME branch, so the binary still runs when the encode
    ///      yields nothing.  (A missing/corrupt tokenizer FILE never
    ///      reaches the select: `nsl_tokenizer_load` aborts with its own
    ///      clean diagnostic — the runtime's documented load contract.)
    ///      `cap` is `4 * prompt_bytes + 16`, comfortably above any real
    ///      tokenizer's token count for the text; if a tokenizer still
    ///      exceeds it, `min(n, cap)` clamps to the tokens actually
    ///      written (no overrun, truncated prompt).  `prompt_buf` stays
    ///      allocated for the process lifetime (the endpoint body may
    ///      call `generate()` repeatedly; a one-shot serve binary exits
    ///      right after).
    ///
    /// Data labels are scoped by serve-block name so two CFIE serve
    /// blocks in one module do not collide (same-name blocks collide
    /// loudly via `DuplicateDefinition`, matching the kernel-reg path).
    fn emit_cfie_endpoint_init(
        &mut self,
        builder: &mut FunctionBuilder,
        scope: &str,
        ep: &CfieEndpointInit,
    ) -> Result<CfieEndpointWiring, CodegenError> {
        // 1. model_handle = nsl_model_create(weights_path_cstr)
        let (wbytes, _wlen) = cfie_nul_terminated(&ep.weights_path);
        let w_id = embed_cfie_bytes(
            &mut self.module,
            &format!("__nsl_cfie_weights_{scope}"),
            wbytes,
        )?;
        let w_gv = self.module.declare_data_in_func(w_id, builder.func);
        let w_ptr = builder.ins().symbol_value(cl_types::I64, w_gv);
        let model_handle = self.compile_call_by_name(builder, "nsl_model_create", &[w_ptr])?;

        // 2. nsl_cfie_bind_model(model_handle, shape...)
        let s = &ep.shape;
        let v_layers = builder.ins().iconst(cl_types::I64, s.n_layers as i64);
        let v_dmodel = builder.ins().iconst(cl_types::I64, s.d_model as i64);
        let v_nheads = builder.ins().iconst(cl_types::I64, s.n_heads as i64);
        let v_nkv = builder.ins().iconst(cl_types::I64, s.n_kv_heads as i64);
        let v_hdim = builder.ins().iconst(cl_types::I64, s.head_dim as i64);
        let v_dff = builder.ins().iconst(cl_types::I64, s.d_ff as i64);
        let v_vocab = builder.ins().iconst(cl_types::I64, s.vocab_size as i64);
        self.compile_call_by_name(
            builder,
            "nsl_cfie_bind_model",
            &[
                model_handle, v_layers, v_dmodel, v_nheads, v_nkv, v_hdim, v_dff, v_vocab,
            ],
        )?;

        // 2b. Cycle 13 (G15): draft-model binding + draft KV pool,
        // emitted AFTER the target bind (nsl_cfie_bind_draft_model
        // refuses when no target is bound — the emission order IS the
        // runtime ordering contract).  Return values are intentionally
        // ignored, matching the target-bind discipline: the runtime
        // warns on stderr and nsl_cfie_speculative_generate refuses
        // cleanly when the draft binding is absent.
        if let Some(dr) = ep.draft.as_ref() {
            let (dbytes, _dlen) = cfie_nul_terminated(&dr.weights_path);
            let d_id = embed_cfie_bytes(
                &mut self.module,
                &format!("__nsl_cfie_draft_weights_{scope}"),
                dbytes,
            )?;
            let d_gv = self.module.declare_data_in_func(d_id, builder.func);
            let d_ptr = builder.ins().symbol_value(cl_types::I64, d_gv);
            let draft_handle =
                self.compile_call_by_name(builder, "nsl_model_create", &[d_ptr])?;
            let v_dlayers = builder.ins().iconst(cl_types::I64, dr.n_layers as i64);
            let v_ddmodel = builder.ins().iconst(cl_types::I64, dr.d_model as i64);
            let v_dnheads = builder.ins().iconst(cl_types::I64, dr.n_heads as i64);
            let v_dnkv = builder.ins().iconst(cl_types::I64, dr.n_kv_heads as i64);
            let v_dhdim = builder.ins().iconst(cl_types::I64, dr.head_dim as i64);
            let v_ddff = builder.ins().iconst(cl_types::I64, dr.d_ff as i64);
            let v_dvocab = builder.ins().iconst(cl_types::I64, dr.vocab_size as i64);
            self.compile_call_by_name(
                builder,
                "nsl_cfie_bind_draft_model",
                &[
                    draft_handle, v_dlayers, v_ddmodel, v_dnheads, v_dnkv, v_dhdim,
                    v_ddff, v_dvocab,
                ],
            )?;
            let v_pool = builder.ins().iconst(cl_types::I64, dr.pool_bytes);
            self.compile_call_by_name(builder, "nsl_cfie_draft_pool_alloc", &[v_pool])?;
        }

        // 3. optional tokenizer_load — Cycle 12 keeps the handle for the
        // runtime prompt encode + the generate() decode tail.
        let mut tok_handle: Option<cranelift_codegen::ir::Value> = None;
        if let Some(tok_path) = ep.tokenizer_path.as_ref() {
            let (tbytes, _tlen) = cfie_nul_terminated(tok_path);
            let t_id = embed_cfie_bytes(
                &mut self.module,
                &format!("__nsl_cfie_tokenizer_{scope}"),
                tbytes,
            )?;
            let t_gv = self.module.declare_data_in_func(t_id, builder.func);
            let t_ptr = builder.ins().symbol_value(cl_types::I64, t_gv);
            tok_handle =
                Some(self.compile_call_by_name(builder, "nsl_tokenizer_load", &[t_ptr])?);
        }

        // 4. bake the FALLBACK prompt token ids as a host i64 array
        // (little-endian raw bytes).
        let mut prompt_bytes: Vec<u8> = Vec::with_capacity(ep.prompt_tokens.len() * 8);
        for tok in &ep.prompt_tokens {
            prompt_bytes.extend_from_slice(&tok.to_le_bytes());
        }
        let p_id = embed_cfie_bytes(
            &mut self.module,
            &format!("__nsl_cfie_prompt_{scope}"),
            prompt_bytes,
        )?;
        let p_gv = self.module.declare_data_in_func(p_id, builder.func);
        let baked_ptr = builder.ins().symbol_value(cl_types::I64, p_gv);
        let baked_len = builder
            .ins()
            .iconst(cl_types::I64, ep.prompt_tokens.len() as i64);

        // 5. Cycle 12: runtime prompt encode + select-fallback, emitted
        // only when tokenizer + prompt are BOTH configured (without a
        // prompt there is nothing to encode; without a tokenizer there
        // is nothing to encode WITH).
        let (prompt_ptr, prompt_len_val) = match (tok_handle, ep.prompt_text.as_ref()) {
            (Some(tok), Some(text)) => {
                let (pbytes, _plen) = cfie_nul_terminated(text);
                let cap_tokens = 4 * (pbytes.len() as i64 - 1) + 16;
                let pt_id = embed_cfie_bytes(
                    &mut self.module,
                    &format!("__nsl_cfie_prompt_text_{scope}"),
                    pbytes,
                )?;
                let pt_gv = self.module.declare_data_in_func(pt_id, builder.func);
                let pt_ptr = builder.ins().symbol_value(cl_types::I64, pt_gv);

                // enc = nsl_tokenizer_encode(tok, prompt_text) — a 1-D
                // f64 tensor of token ids (the runtime tokenizer ABI).
                let enc =
                    self.compile_call_by_name(builder, "nsl_tokenizer_encode", &[tok, pt_ptr])?;
                // prompt_buf = nsl_alloc(cap * 8) host i64 buffer.
                let v_cap = builder.ins().iconst(cl_types::I64, cap_tokens);
                let v_cap_bytes = builder.ins().iconst(cl_types::I64, cap_tokens * 8);
                let prompt_buf =
                    self.compile_call_by_name(builder, "nsl_alloc", &[v_cap_bytes])?;
                // n = nsl_cfie_tensor_to_tokens(enc, prompt_buf, cap);
                // then free the encode tensor — the ids now live in
                // prompt_buf (free-every-tensor-on-every-path: this is
                // the only path, no branches between encode and free).
                let n = self.compile_call_by_name(
                    builder,
                    "nsl_cfie_tensor_to_tokens",
                    &[enc, prompt_buf, v_cap],
                )?;
                self.compile_call_by_name(builder, "nsl_tensor_free", &[enc])?;

                // Runtime select: use (prompt_buf, min(n, cap)) when the
                // encode produced tokens (n > 0), else the baked prompt.
                use cranelift_codegen::ir::condcodes::IntCC;
                let zero = builder.ins().iconst(cl_types::I64, 0);
                let use_rt = builder.ins().icmp(IntCC::SignedGreaterThan, n, zero);
                let n_over_cap = builder.ins().icmp(IntCC::SignedGreaterThan, n, v_cap);
                let n_clamped = builder.ins().select(n_over_cap, v_cap, n);
                let sel_ptr = builder.ins().select(use_rt, prompt_buf, baked_ptr);
                let sel_len = builder.ins().select(use_rt, n_clamped, baked_len);
                (sel_ptr, Some(sel_len))
            }
            _ => (baked_ptr, None),
        };

        Ok(CfieEndpointWiring {
            prompt_ptr,
            prompt_len_val,
            tok_handle,
        })
    }

    /// CFIE Tier-A wiring: extract serve-block CFIE config, resolve the
    /// mode, run the six-pass orchestrator against real inputs (GPU
    /// database + `--weights` WeightMap + serve config keys), print the
    /// build report, and publish the plan on `bus.cfie_plan`.
    ///
    /// Returns the runtime-init values the compiled binary must pass to
    /// the `nsl_cfie_*` init FFIs, or `None` when CFIE is not active.
    ///
    /// `is_disaggregated`: the disaggregated serve branch emits NONE of
    /// the CFIE runtime wiring (slots/registration/finalize) — the
    /// report must say so instead of claiming monolithic wiring.
    fn run_cfie_for_serve(
        &mut self,
        serve: &ServeBlock,
        is_disaggregated: bool,
    ) -> Result<Option<CfieRuntimeInit>, CodegenError> {
        let interner = self.interner;
        let resolve = |sym: nsl_ast::Symbol| -> String {
            interner.resolve(sym.0).unwrap_or("").to_string()
        };
        let cfg = crate::cfie_serve::extract(serve, &resolve);

        let decorator_mode = self.cfie_decorator_mode.take();
        let decorator_target = self.cfie_decorator_target.take();

        let Some(mode) = crate::cfie_serve::resolve_mode(
            self.compile_options.cfie.mode_override.as_deref(),
            decorator_mode,
            &cfg,
        )?
        else {
            return Ok(None);
        };
        if mode == crate::cfie::CfieMode::Off {
            // Explicit opt-out: leave the M29/M41 dynamic path untouched.
            return Ok(None);
        }

        // GPU resolution: serve config key > @cfie(target=...) > CLI
        // default.  An unknown GPU is a hard error — planning against a
        // guessed budget is exactly what the CFIE audit flagged.
        let gpu_name = cfg
            .target_gpu
            .clone()
            .or(decorator_target)
            .unwrap_or_else(|| self.compile_options.target_gpu.clone());
        let gpu = crate::gpu_specs::find_gpu(&gpu_name).ok_or_else(|| {
            CodegenError::new(format!(
                "CFIE: unknown target GPU '{gpu_name}'; known GPUs: {}",
                crate::gpu_specs::GPU_DATABASE
                    .iter()
                    .map(|g| g.name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })?;

        let weights = self.features.weight_map.as_ref();
        let prepared = crate::cfie_serve::prepare(&cfg, mode, gpu, weights)?;
        let mut plan = crate::cfie::run(prepared.input);

        // The `kv_layout` key is honoured, not just used to switch CFIE
        // on: "paged" forces the paged plan, and "static" with an
        // envelope that does not fit is an error rather than a build
        // whose pool is larger than the card.
        match cfg.kv_layout.as_deref() {
            Some("paged") => {
                plan.kv = crate::cfie_kv_plan::KvLayoutPlan::paged(
                    plan.kv.available_bytes,
                    "kv_layout: \"paged\" was requested",
                );
            }
            Some("static") if !plan.kv.uses_direct_indexing() => {
                return Err(CodegenError::new(format!(
                    "CFIE kv_layout: \"static\" cannot be honoured on {}: {}. Lower max_batch \
                     or max_seq, or use kv_layout: \"auto\" (which then plans the paged \
                     layout, without the compiled decode path).",
                    gpu.name, plan.kv.rationale
                )));
            }
            _ => {}
        }

        let s = &prepared.shape;
        let per_slot = plan
            .kv
            .direct
            .as_ref()
            .map(|d| d.per_sequence_max_tokens)
            .unwrap_or(0);
        let max_slots = plan.persistent.scheduler.max_active;
        // Serve keys with the common-architecture defaults; a model with
        // a different RoPE base (e.g. 500000) or norm epsilon MUST set
        // them or the baked constants are numerically wrong.  Every
        // kernel that normalises takes the same epsilon.
        let rope_theta = cfg.rope_theta.unwrap_or(10_000.0) as f32;
        let norm_eps = cfg
            .norm_eps
            .map(|e| e as f32)
            .unwrap_or(crate::cfie_sample_ptx::DEFAULT_NORM_EPS);

        // The decode block: one launch runs one layer's decode step for
        // one token.  It needs the static KV pool it appends to, the
        // planner's whole-layer fusion verdict, the emitter's shape
        // preconditions, and a static shared-memory footprint one thread
        // block may declare.  A miss is recorded with its reason; the
        // report prints it and `generate()` refuses with it.
        plan.decode_block = if !plan.kv.uses_direct_indexing() {
            crate::cfie::CfieKernel::refused(format!(
                "the KV layout is paged ({}), and the decode block addresses a static pool",
                plan.kv.rationale
            ))
        } else if plan.persistent.fusion != crate::cfie_persistent::FusionLevel::Level3 {
            crate::cfie::CfieKernel::refused(format!(
                "the planner's fusion level is {} (the whole-layer kernel does not fit the \
                 target GPU's shared-memory budget), and only the whole-layer kernel has an \
                 emitter",
                plan.persistent.fusion.as_str()
            ))
        } else {
            let shape_ok = s.head_dim >= 2
                && s.head_dim % 2 == 0
                && s.head_dim <= 128
                && s.d_model >= 1
                && s.d_model <= 8192
                && s.d_ff >= 1
                && s.d_ff <= 32768
                && s.n_heads % s.n_kv_heads.max(1) == 0
                && s.n_heads * s.head_dim <= 8192
                && per_slot >= 1
                && max_slots >= 1
                && (max_slots as u64) * (per_slot as u64) <= u32::MAX as u64;
            if !shape_ok {
                crate::cfie::CfieKernel::refused(format!(
                    "the model shape is outside what the kernel supports (head_dim {} even \
                     in 2..=128, d_model {} <= 8192, d_ff {} <= 32768, n_heads {} divisible \
                     by n_kv_heads {}, n_heads*head_dim <= 8192, max_batch*max_seq below \
                     2^32)",
                    s.head_dim, s.d_model, s.d_ff, s.n_heads, s.n_kv_heads
                ))
            } else {
                let blk_cfg = crate::cfie_persistent_ptx::DecodeBlockConfig {
                    d_model: s.d_model,
                    head_dim: s.head_dim,
                    n_heads: s.n_heads,
                    n_kv_heads: s.n_kv_heads,
                    d_ff: s.d_ff,
                    per_slot_max_tokens: per_slot,
                    max_slots,
                    n_layers: s.n_layers,
                    rope_theta,
                    eps: norm_eps,
                };
                // Static .shared declarations are capped at 48 KB per
                // thread block; a bigger footprint would fail module
                // load (and the KIR verifier refuses to build it).
                let smem = crate::cfie_persistent_ptx::smem_bytes(&blk_cfg);
                if smem > CFIE_STATIC_SMEM_LIMIT {
                    crate::cfie::CfieKernel::refused(format!(
                        "its static shared memory would be {smem} bytes and one thread block \
                         may declare at most {CFIE_STATIC_SMEM_LIMIT} (d_model {}, {} heads of \
                         {}): this model is too wide for the one-block kernel",
                        s.d_model, s.n_heads, s.head_dim
                    ))
                } else {
                    let (ptx, meta) = crate::cfie_persistent_ptx::emit(&blk_cfg);
                    crate::cfie::CfieKernel::emitted(
                        meta.kernel_name,
                        ptx,
                        crate::cfie::CfieLaunchSpec {
                            grid_x: 1,
                            block_x: meta.block_dim,
                            smem_dyn_bytes: 0,
                        },
                    )
                }
            }
        };

        // The fused decode-sample kernel, launched by the host decode
        // loop (nsl_cfie_decode_step) on the final hidden state.  The
        // plan's sampler program bakes vocab_tile 256 (the accounting
        // default from `prepare`); the kernel tile is fixed at 128, so
        // the program is re-emitted against the kernel tile with the
        // same parameters.
        plan.fused_sample = {
            let top_k = plan.sampling.params.top_k;
            let supported = s.d_model >= 1
                && s.d_model <= 8192
                && s.vocab_size >= 128
                && s.vocab_size % 128 == 0
                && (1..=64).contains(&top_k);
            if !supported {
                crate::cfie::CfieKernel::refused(format!(
                    "the kernel needs d_model in 1..=8192 (is {}), a vocab_size that is a \
                     positive multiple of 128 (is {}) and top_k in 1..=64 (is {})",
                    s.d_model, s.vocab_size, top_k
                ))
            } else {
                let shape128 = crate::cfie_fused_sample::LmHeadShape {
                    vocab_tile: 128,
                    ..plan.sampling.shape
                };
                let program =
                    crate::cfie_fused_sample::emit_program(plan.sampling.params, shape128);
                let sample_cfg = crate::cfie_sample_ptx::FusedSampleKernelConfig {
                    d_model: s.d_model,
                    vocab_size: s.vocab_size,
                    vocab_tile: 128,
                    top_k,
                    // No grammar is ever bound: a `grammar:` section is
                    // refused (cfie_serve::prepare).
                    grammar_states: 0,
                    norm_eps,
                };
                let (ptx, meta) = crate::cfie_sample_ptx::emit(&program, &sample_cfg);
                crate::cfie::CfieKernel::emitted(
                    meta.kernel_name,
                    ptx,
                    crate::cfie::CfieLaunchSpec {
                        grid_x: 1, // single-CTA latency path
                        block_x: meta.block_dim,
                        smem_dyn_bytes: 0,
                    },
                )
            }
        };

        // The draft model of a `speculative:` section: its decode block
        // (the same emitter with the draft dims, max_slots = 1, per-slot
        // capacity = the TARGET's per the engine's draft-pool sizing
        // contract) and its greedy sampler, plus the resolved draft info
        // that drives the serve-init binding and the generate() ->
        // speculative_generate rewrite.  A section that cannot be wired
        // is an error: it would otherwise build a binary whose
        // generate() ran plain decoding, or refused, with no hint.
        if let (Some(spec_cfg), Some(sp_sec)) = (plan.speculative.clone(), cfg.speculative.as_ref())
        {
            let draft_path = sp_sec.draft_weights.clone().unwrap_or_default();
            let refuse = |what: String| {
                Err(CodegenError::new(format!(
                    "CFIE speculative (draft_weights '{draft_path}'): {what}. Fix the serve \
                     config or remove the speculative: section."
                )))
            };
            if let Some(why) = plan.generate_unavailable() {
                return refuse(format!(
                    "the speculative driver verifies with the target's decode block and \
                     fused sampler, and {why}"
                ));
            }
            // Draft dims: explicit draft_* keys, defaulting to the
            // TARGET's resolved values (self-speculation shape).  A
            // mismatch against the actual checkpoint refuses loudly at
            // serve init (bind_draft_model validates every shape).
            let pick = |v: Option<i64>, dflt: u32| -> u32 {
                match v {
                    Some(x) if x > 0 => x as u32,
                    _ => dflt,
                }
            };
            let dn_layers = pick(sp_sec.draft_n_layers, s.n_layers);
            let dd_model = pick(sp_sec.draft_d_model, s.d_model);
            let dn_heads = pick(sp_sec.draft_n_heads, s.n_heads);
            let dn_kv = pick(sp_sec.draft_n_kv_heads, s.n_kv_heads);
            let dhead = pick(sp_sec.draft_head_dim, s.head_dim);
            let dd_ff = pick(sp_sec.draft_d_ff, s.d_ff);
            let draft_ok = dhead >= 2
                && dhead % 2 == 0
                && dhead <= 128
                && (1..=8192).contains(&dd_model)
                && (1..=32768).contains(&dd_ff)
                && dn_heads % dn_kv.max(1) == 0
                && dn_heads * dhead <= 8192
                && per_slot >= 1;
            if !draft_ok {
                return refuse(
                    "the draft dims fail the decode-block emitter preconditions (head_dim \
                     even in 2..=128, d_model <= 8192, d_ff <= 32768, n_heads divisible by \
                     n_kv_heads, n_heads*head_dim <= 8192, static per-slot capacity >= 1)"
                        .to_string(),
                );
            }
            let draft_blk_cfg = crate::cfie_persistent_ptx::DecodeBlockConfig {
                d_model: dd_model,
                head_dim: dhead,
                n_heads: dn_heads,
                n_kv_heads: dn_kv,
                d_ff: dd_ff,
                // Sizing contract: the draft pool's per-slot capacity
                // equals the TARGET's, so the driver's capacity probe
                // (bounded by the target slot) bounds draft writes too.
                per_slot_max_tokens: per_slot,
                max_slots: 1,
                n_layers: dn_layers,
                rope_theta,
                eps: norm_eps,
            };
            if crate::cfie_persistent_ptx::smem_bytes(&draft_blk_cfg) > CFIE_STATIC_SMEM_LIMIT {
                return refuse(format!(
                    "the draft decode block's static shared memory exceeds the \
                     {CFIE_STATIC_SMEM_LIMIT}-byte limit of one thread block (shrink draft \
                     d_model/d_ff/head_dim)"
                ));
            }
            let (dptx, dmeta) = crate::cfie_persistent_ptx::emit(&draft_blk_cfg);
            let sampler_cfg = crate::cfie_spec_sampler_ptx::SpecSamplerConfig {
                d_model: dd_model,
                vocab_size: s.vocab_size, // shared vocab
                vocab_tile: 128,
                norm_eps,
            };
            let (sptx, smeta) = crate::cfie_spec_sampler_ptx::emit_draft_sample(&sampler_cfg);
            let one_block = |block_x: u32| crate::cfie::CfieLaunchSpec {
                grid_x: 1,
                block_x,
                smem_dyn_bytes: 0,
            };
            plan.draft_block =
                crate::cfie::CfieKernel::emitted(dmeta.kernel_name, dptx, one_block(dmeta.block_dim));
            plan.draft_sample =
                crate::cfie::CfieKernel::emitted(smeta.kernel_name, sptx, one_block(smeta.block_dim));
            // Draft KV pool: [n_layers][2][1 * per_slot][n_kv][hd] f16.
            let pool_bytes =
                (dn_layers as i64) * 2 * (per_slot as i64) * (dn_kv as i64) * (dhead as i64) * 2;
            plan.speculative_draft = Some(crate::cfie::CfieSpeculativeDraft {
                weights_path: draft_path,
                n_layers: dn_layers,
                d_model: dd_model,
                n_heads: dn_heads,
                n_kv_heads: dn_kv,
                head_dim: dhead,
                d_ff: dd_ff,
                vocab_size: s.vocab_size,
                pool_bytes,
                k_tokens: spec_cfg.k_tokens,
                target_n_layers: s.n_layers,
            });
        }

        // Build report: what this build bakes into the binary.
        plan.runtime_wiring_emitted = !is_disaggregated;
        let mut report = plan.render_report();
        report.push_str(&format!(
            "Model-shape provenance: {}\n",
            prepared.shape.provenance
        ));
        if is_disaggregated {
            report.push_str(
                "Kernel wiring: DISAGGREGATED serve — CFIE runtime wiring \
                 (KV slots, kernel registration, engine finalize) is NOT \
                 emitted on this path; the plan and kernels above are \
                 report-only for this build (monolithic serve gets the \
                 wiring).\n",
            );
        } else if plan.decode_path_emitted() {
            report.push_str(
                "Kernel wiring: KV-slot init + kernel registration + engine \
                 finalize in this build; per-token launches go through the \
                 host decode loop (nsl_cfie_decode_step).\n",
            );
        } else {
            report.push_str(
                "Kernel wiring: none in this build — the decode path is \
                 incomplete, so no kernel is registered.\n",
            );
        }
        eprint!("{report}");
        if let Some(path) = self.compile_options.cfie.report_path.clone()
            && let Err(e) = std::fs::write(&path, &report)
        {
            nsl_log::nsl_log!(WARN, "codegen", 
                "warning: --cfie-report: failed to write {}: {e}",
                path.display()
            );
        }

        // Registrations, the KV slot free-list and the device pool: all
        // exist exactly when the decode path does (the decode block is
        // the only kernel that reads or writes KV, and without the
        // sampler nothing would drive it).  The pool is the uniform f16
        // layout over ALL layers.
        let kernels = crate::cfie::kernel_registrations(&plan);
        let kv_slots = plan
            .decode_path_emitted()
            .then_some((max_slots as i64, per_slot as i64));
        let pool_bytes = kv_slots.map(|(slots, per_slot)| {
            (s.n_layers as i64)
                * 2 // K + V halves
                * (slots * per_slot)
                * (s.n_kv_heads as i64)
                * (s.head_dim as i64)
                * 2 // f16 bytes
        });

        // The endpoint's generation driver is wired only when the decode
        // path is in the build.  Otherwise `generate()` is a COMPILE
        // error that says which kernel is missing and why: the runtime
        // would refuse every call.
        let generate_refusal = plan.generate_unavailable();
        let endpoint = if generate_refusal.is_some() {
            None
        } else {
            // Weights: serve `weights:` key > `--weights` CLI path (the
            // loaded WeightMap's source path) > empty (null handle;
            // bind_model refuses cleanly at runtime).
            let weights_path = cfg
                .weights_path
                .clone()
                .or_else(|| weights.map(|w| w.source_path().to_string()))
                .unwrap_or_default();
            // FALLBACK prompt token ids.  Runtime `nsl_cfie_generate`
            // reads a host i64 array, so we bake raw i64s.  A configured
            // `prompt:` is baked as byte-level token ids (the runtime
            // byte-tokenizer convention), clamped into [0, vocab_size) so
            // the runtime's per-token range guard passes; absent that, a
            // single [0] sentinel keeps the body runnable.  When
            // `tokenizer:` is ALSO configured, serve init emits the
            // runtime tokenizer encode (`nsl_tokenizer_encode` ->
            // `nsl_cfie_tensor_to_tokens`) and selects it over these
            // baked ids at runtime whenever it produces tokens.  A
            // tokenizer id >= vocab_size is NOT clamped on that path —
            // `nsl_cfie_generate`'s per-token range guard refuses it
            // loudly (honest refusal beats a silently-wrong prompt).
            let vocab = prepared.shape.vocab_size.max(1) as i64;
            let prompt_text = cfg.prompt.clone().filter(|p| !p.is_empty());
            let prompt_tokens: Vec<i64> = match cfg.prompt.as_deref() {
                Some(p) if !p.is_empty() => p
                    .bytes()
                    .map(|b| (b as i64) % vocab)
                    .collect(),
                _ => vec![0],
            };
            Some(CfieEndpointInit {
                weights_path,
                tokenizer_path: cfg.tokenizer_path.clone(),
                prompt_tokens,
                prompt_text,
                shape: prepared.shape.clone(),
                max_new_tokens: cfg.max_new_tokens.unwrap_or(64).max(1),
                eos_token_id: cfg.eos_token_id.unwrap_or(-1),
                rng_seed: cfg.sampling.as_ref().and_then(|s| s.seed).unwrap_or(0).max(0),
                // Present exactly when the draft kernels were emitted.
                draft: plan.speculative_draft.clone(),
            })
        };

        self.bus.publish_cfie_plan(plan);
        Ok(Some(CfieRuntimeInit {
            kv_slots,
            kernels,
            pool_bytes,
            endpoint,
            generate_refusal,
        }))
    }

    /// M41: Compile a disaggregated serve block.
    ///
    /// Generates role-dispatch code by reading NSL_ROLE env var at runtime:
    ///   - role == 0 (router):  nsl_disagg_init() + nsl_disagg_router_loop()
    ///   - role == 1 (prefill): nsl_disagg_worker_init(1, rank, 0) + nsl_disagg_prefill_loop(0)
    ///   - role == 2 (decode):  nsl_disagg_worker_init(2, rank, 0) + nsl_disagg_decode_loop(0)
    ///
    /// The runtime FFI `nsl_disagg_get_role()` reads NSL_ROLE env var and returns
    /// 0/1/2. This avoids string comparison in Cranelift IR.
    #[allow(clippy::too_many_arguments)]
    fn compile_disaggregated_serve(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        serve: &ServeBlock,
        max_batch: i64,
        _max_seq_len: i64,
        kv_blocks: i64,
        _prefill_chunk: i64,
        prefill_workers: i64,
        decode_workers: i64,
        kv_transfer_backend: &str,
    ) -> Result<(), CodegenError> {
        // M41b: Map kv_transfer backend name to ID for runtime init
        let kv_backend_id: i64 = match kv_transfer_backend {
            "shared_mem" => 0,
            "nvlink" => 1,
            "rdma" => 2,
            "tcp" => 3,
            _ => -1, // auto-detect at runtime (includes "auto")
        };

        // Step 1: Get role from env var via FFI (returns 0=router, 1=prefill, 2=decode)
        let role = self.compile_call_by_name(builder, "nsl_disagg_get_role", &[])?;

        // ONE worker config slot, so exactly one @speculative configuration can
        // be represented. `values().next()` picked an arbitrary HashMap entry:
        // two differing decorators baked whichever the hash seed happened to
        // yield, differently from build to build. Refuse instead.
        let speculative_config = crate::speculative::unique_worker_config(
            &self.features.speculative_configs,
        )
        .map_err(|keys| {
            CodegenError::new(format!(
                "a disaggregated `serve` block carries {} differing `@speculative` \
                 configurations ({}), but a worker config slot holds exactly one. \
                 Give the speculative layers matching decorator arguments, or move \
                 all but one out of the served model.",
                keys.len(),
                keys.join(", ")
            ))
        })?;
        let speculative_tokens = speculative_config
            .map(|info| info.num_tokens as i64)
            .unwrap_or(0);
        let speculative_method = speculative_config
            .map(|info| match info.method {
                SpeculativeMethod::Draft => 0,
                SpeculativeMethod::Medusa => 1,
                SpeculativeMethod::Eagle2 => 2,
                SpeculativeMethod::Lookahead => 3,
            })
            .unwrap_or(0);
        let speculative_tree_width = speculative_config
            .map(|info| info.tree_width as i64)
            .unwrap_or(1);
        let speculative_temperature_bits = speculative_config
            .map(|info| info.temperature.to_bits() as i64)
            .unwrap_or(0);
        let prefill_config_slot = self.build_worker_config_slot(
            builder,
            WorkerConfigSpec {
                max_seq_len: _max_seq_len,
                kv_blocks,
                speculative_tokens: 0,
                speculative_method: 0,
                speculative_tree_width: 1,
                speculative_temperature_bits: 0,
            },
        );
        let decode_config_slot = self.build_worker_config_slot(
            builder,
            WorkerConfigSpec {
                max_seq_len: _max_seq_len,
                kv_blocks,
                speculative_tokens,
                speculative_method,
                speculative_tree_width,
                speculative_temperature_bits,
            },
        );

        // Step 2: Branch on role
        let router_block = builder.create_block();
        let check_prefill_block = builder.create_block();
        let prefill_block = builder.create_block();
        let decode_block = builder.create_block();
        let merge_block = builder.create_block();

        // if role == 0 → router_block, else → check_prefill_block
        let zero = builder.ins().iconst(cl_types::I64, 0);
        let is_router =
            builder
                .ins()
                .icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, role, zero);
        builder
            .ins()
            .brif(is_router, router_block, &[], check_prefill_block, &[]);

        // check_prefill_block: if role == 1 → prefill_block, else → decode_block
        builder.switch_to_block(check_prefill_block);
        builder.seal_block(check_prefill_block);
        let one = builder.ins().iconst(cl_types::I64, 1);
        let is_prefill =
            builder
                .ins()
                .icmp(cranelift_codegen::ir::condcodes::IntCC::Equal, role, one);
        builder
            .ins()
            .brif(is_prefill, prefill_block, &[], decode_block, &[]);

        // --- Router block ---
        builder.switch_to_block(router_block);
        builder.seal_block(router_block);
        let v_prefill = builder.ins().iconst(cl_types::I64, prefill_workers);
        let v_decode = builder.ins().iconst(cl_types::I64, decode_workers);
        let v_batch = builder.ins().iconst(cl_types::I64, max_batch);
        let v_kv = builder.ins().iconst(cl_types::I64, kv_blocks);
        self.compile_call_by_name(
            builder,
            "nsl_disagg_init",
            &[v_prefill, v_decode, v_batch, v_kv],
        )?;
        // Router runs endpoint bodies (sets up the event loop)
        for endpoint in &serve.endpoints {
            // Same scoping as the monolithic serve path above.
            state.push_fn_binding_scope();
            for stmt in &endpoint.body.stmts {
                self.compile_stmt(builder, state, stmt)?;
            }
            state.pop_fn_binding_scope();
        }
        self.compile_call_by_name(builder, "nsl_disagg_destroy", &[])?;
        builder.ins().jump(merge_block, &[]);

        // --- Prefill worker block ---
        builder.switch_to_block(prefill_block);
        builder.seal_block(prefill_block);
        let role_prefill = builder.ins().iconst(cl_types::I64, 1);
        let rank = self.compile_call_by_name(builder, "nsl_disagg_get_rank", &[])?;
        let model_zero = builder.ins().iconst(cl_types::I64, 0); // model ptr placeholder
        self.compile_call_by_name(
            builder,
            "nsl_disagg_worker_init",
            &[role_prefill, rank, model_zero],
        )?;
        // M41b: Initialize KV transfer backend for this worker
        let v_kv_backend = builder.ins().iconst(cl_types::I64, kv_backend_id);
        self.compile_call_by_name(builder, "nsl_kv_transfer_init", &[v_kv_backend, rank])?;
        let prefill_config = builder
            .ins()
            .stack_addr(cl_types::I64, prefill_config_slot, 0);
        self.compile_call_by_name(builder, "nsl_disagg_prefill_loop", &[prefill_config])?;
        self.compile_call_by_name(builder, "nsl_kv_transfer_destroy", &[])?;
        self.compile_call_by_name(builder, "nsl_disagg_worker_destroy", &[])?;
        builder.ins().jump(merge_block, &[]);

        // --- Decode worker block ---
        builder.switch_to_block(decode_block);
        builder.seal_block(decode_block);
        let role_decode = builder.ins().iconst(cl_types::I64, 2);
        let rank2 = self.compile_call_by_name(builder, "nsl_disagg_get_rank", &[])?;
        let model_zero2 = builder.ins().iconst(cl_types::I64, 0);
        self.compile_call_by_name(
            builder,
            "nsl_disagg_worker_init",
            &[role_decode, rank2, model_zero2],
        )?;
        // M41b: Initialize KV transfer backend for this worker
        let v_kv_backend2 = builder.ins().iconst(cl_types::I64, kv_backend_id);
        self.compile_call_by_name(builder, "nsl_kv_transfer_init", &[v_kv_backend2, rank2])?;
        let decode_config = builder
            .ins()
            .stack_addr(cl_types::I64, decode_config_slot, 0);
        self.compile_call_by_name(builder, "nsl_disagg_decode_loop", &[decode_config])?;
        self.compile_call_by_name(builder, "nsl_kv_transfer_destroy", &[])?;
        self.compile_call_by_name(builder, "nsl_disagg_worker_destroy", &[])?;
        builder.ins().jump(merge_block, &[]);

        // --- Merge block ---
        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);
        state.current_block = Some(merge_block);

        Ok(())
    }
}

#[cfg(test)]
mod cfie_emit_tests {
    use super::{cfie_label_scope, cfie_nul_terminated};

    // The runtime copies [ptr, ptr + len) and appends its own NUL, so a
    // regression to len-INCLUDING-NUL would hand the driver an interior
    // NUL (PR #251 class of bug).  Pin the convention here.
    #[test]
    fn nul_terminated_len_excludes_terminator() {
        let (bytes, len) = cfie_nul_terminated("abc");
        assert_eq!(bytes, b"abc\0");
        assert_eq!(len, 3);
        let (bytes, len) = cfie_nul_terminated("");
        assert_eq!(bytes, b"\0");
        assert_eq!(len, 0);
    }

    #[test]
    fn label_scope_sanitizes_to_identifier_chars() {
        assert_eq!(cfie_label_scope("Inference"), "Inference");
        assert_eq!(cfie_label_scope("My-Serve.2"), "My_Serve_2");
        assert_eq!(cfie_label_scope(""), "serve");
    }
}
