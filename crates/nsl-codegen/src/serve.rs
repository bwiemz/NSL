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
    ring_capacity: i64,
    /// `(slot_count, per_slot_tokens)` for the KV slot free-list —
    /// present only when the plan selected a static layout and the
    /// decode-attention kernel was emitted.
    kv_slots: Option<(i64, i64)>,
    /// Cycle 6: kernel registrations for the compile-time-chosen
    /// family (`cfie::kernel_registrations` — quant wins whenever the
    /// per-layer kernels were emitted, else uniform).  Each entry
    /// becomes one `nsl_cfie_register_kernel` call at serve init.
    kernels: Vec<crate::cfie::CfieKernelReg>,
    /// Device KV-pool bytes for `nsl_cfie_kv_pool_alloc` — `Some` only
    /// when at least one KV-consuming kernel (kind 0/2/3/5) registers.
    /// Quant family: `cfie_kv_quant_ptx::total_pool_bytes`; uniform:
    /// `n_layers * 2 * (max_slots * per_slot_tokens) * n_kv_heads *
    /// head_dim * 2` (f16).
    pool_bytes: Option<i64>,
    /// Cycle 11: model-binding + generation parameters resolved from the
    /// serve config + the compile-time model shape.  `Some` only when
    /// the plan actually wired the decode path (kernels emitted); `None`
    /// leaves the generation FFIs unemitted so `generate()` refuses.
    endpoint: Option<CfieEndpointInit>,
}

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
    /// Optional tokenizer path for the decode-and-print demo tail.
    tokenizer_path: Option<String>,
    /// The prompt token ids, baked at compile time.  When a tokenizer +
    /// prompt are configured this is the offline-encoded prompt; else a
    /// tiny sentinel (`[0]`) so the endpoint body still compiles + runs.
    /// The runtime `nsl_cfie_generate` prompt ABI is a host i64 array,
    /// so these are baked as raw i64s in .rodata (NOT an f64 tokenizer
    /// tensor — the tokenizer-encode -> generate bridge would need an
    /// f64-tensor -> i64-host-array runtime FFI that is out of Cycle-11
    /// scope; see the deferral note in `emit_cfie_endpoint_init`).
    prompt_tokens: Vec<i64>,
    /// Resolved model shape (passed to `nsl_cfie_bind_model`).
    shape: crate::cfie_serve::ResolvedModelShape,
    max_new_tokens: i64,
    eos_token_id: i64,
}

/// CFIE Cycle 11 side-channel: the `generate()` intrinsic's driver
/// parameters, published on `Compiler::cfie_serve_gen` while a
/// CFIE-active serve body compiles.  Presence of this value is exactly
/// the "CFIE serve context" the `generate()` rewrite tests for.
pub struct CfieServeGen {
    pub max_new_tokens: i64,
    pub eos_token_id: i64,
    /// Prompt token count baked at serve init (host i64 array length).
    pub prompt_len: i64,
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
            .stack_store(max_seq_len_val, slot, WORKER_MAX_SEQ_LEN_OFFSET);
        builder
            .ins()
            .stack_store(kv_blocks_val, slot, WORKER_KV_BLOCKS_OFFSET);
        builder
            .ins()
            .stack_store(block_size_val, slot, WORKER_BLOCK_SIZE_OFFSET);
        builder
            .ins()
            .stack_store(num_kv_heads_val, slot, WORKER_NUM_KV_HEADS_OFFSET);
        builder
            .ins()
            .stack_store(head_dim_val, slot, WORKER_HEAD_DIM_OFFSET);
        builder
            .ins()
            .stack_store(num_layers_val, slot, WORKER_NUM_LAYERS_OFFSET);
        builder
            .ins()
            .stack_store(speculative_tokens_val, slot, WORKER_SPEC_TOKENS_OFFSET);
        builder
            .ins()
            .stack_store(speculative_method_val, slot, WORKER_SPEC_METHOD_OFFSET);
        builder.ins().stack_store(
            speculative_tree_width_val,
            slot,
            WORKER_SPEC_TREE_WIDTH_OFFSET,
        );
        builder.ins().stack_store(
            speculative_temp_bits_val,
            slot,
            WORKER_SPEC_TEMP_BITS_OFFSET,
        );
        builder
            .ins()
            .stack_store(eos_val, slot, WORKER_EOS_TOKEN_OFFSET);

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
        // drives the report, the request-ring + KV-slot init calls, and
        // the Cycle-6 kernel registration/finalize emission — the
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

            // CFIE: size the continuous-batching request ring and the
            // KV sequence-slot free-list from the compile-time plan
            // (audit gaps G5/G8 — the runtime FFIs are now reachable
            // from emitted code).  Monolithic path only; the
            // disaggregated workers keep their M41 queueing until the
            // persistent decode kernel lands (G16).
            if let Some(init) = &cfie_init {
                let v_capacity = builder.ins().iconst(cl_types::I64, init.ring_capacity);
                self.compile_call_by_name(builder, "nsl_cfie_ring_init", &[v_capacity])?;
                if let Some((slot_count, per_slot_tokens)) = init.kv_slots {
                    let v_slots = builder.ins().iconst(cl_types::I64, slot_count);
                    let v_tokens = builder.ins().iconst(cl_types::I64, per_slot_tokens);
                    self.compile_call_by_name(
                        builder,
                        "nsl_cfie_kv_slots_init",
                        &[v_slots, v_tokens],
                    )?;
                }
                // CFIE Cycle 6: register the compiled kernel family,
                // allocate the device KV pool, and finalize the engine
                // (one cuModuleLoadData + cuModuleGetFunction per
                // kernel) before any endpoint body runs.  Return values
                // are intentionally ignored — the runtime warns on
                // stderr and every launch FFI refuses cleanly (-1) when
                // finalize did not succeed.
                if !init.kernels.is_empty() {
                    let scope = cfie_label_scope(&self.resolve_sym(serve.name).to_string());
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
                    let scope =
                        cfie_label_scope(&self.resolve_sym(serve.name).to_string());
                    let prompt_ptr = self.emit_cfie_endpoint_init(builder, &scope, ep)?;
                    self.cfie_serve_gen = Some(CfieServeGen {
                        max_new_tokens: ep.max_new_tokens,
                        eos_token_id: ep.eos_token_id,
                        prompt_len: ep.prompt_tokens.len() as i64,
                    });
                    cfie_prompt_ptr = Some(prompt_ptr);
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
                    let var = state.new_variable();
                    builder.declare_var(var, cl_types::I64);
                    builder.def_var(var, init_val);
                    state.variables.insert(param.name, (var, cl_types::I64));
                    state.param_symbols.insert(param.name);
                }
                for stmt in &endpoint.body.stmts {
                    self.compile_stmt(builder, state, stmt)?;
                }
            }

            // CFIE Cycle 11: the serve body is done — the `generate()`
            // side-channel must not leak into any function compiled
            // after this serve block.
            self.cfie_serve_gen = None;

            // CFIE Cycle 6: free the pool + clear registrations before
            // the serve runtime tears down (CUmodules may stay loaded —
            // module leak-by-design precedent, cuda/mod.rs module_cache).
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
    /// collide: `__nsl_cfie_{ptx,name}_{scope}_{kind}` (`_l{N}` suffix
    /// for kind 5).
    fn emit_cfie_register_kernel(
        &mut self,
        builder: &mut FunctionBuilder,
        label_scope: &str,
        reg: &crate::cfie::CfieKernelReg,
    ) -> Result<(), CodegenError> {
        let suffix = if reg.kind == 5 {
            format!("{label_scope}_{}_l{}", reg.kind, reg.layer_idx)
        } else {
            format!("{label_scope}_{}", reg.kind)
        };

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

    /// CFIE Cycle 11: emit the model-binding + tokenizer-load + baked
    /// prompt-token array, and return the baked prompt host i64 array
    /// pointer (the endpoint's first param is bound to it).
    ///
    /// Emission order (AFTER `nsl_cfie_engine_finalize`):
    ///   1. `model_handle = nsl_model_create(weights_path_cstr)`.  A null
    ///      handle (missing/failed weights) is well-formed: the runtime
    ///      `nsl_cfie_bind_model` refuses a null handle cleanly.
    ///   2. `nsl_cfie_bind_model(model_handle, n_layers, d_model,
    ///      n_heads, n_kv_heads, head_dim, d_ff, vocab_size)` with the
    ///      resolved-shape constants.
    ///   3. `tok_handle = nsl_tokenizer_load(tokenizer_path_cstr)` when a
    ///      tokenizer is configured (loaded so the decode-and-print demo
    ///      tail can resolve it; the handle is currently unused by the
    ///      body because the i64-token-buffer -> f64-tensor decode bridge
    ///      is a documented Cycle-11 deferral).
    ///   4. bake the prompt token ids as a raw host i64 array in .rodata
    ///      and return its pointer.
    ///
    /// Data labels are scoped by serve-block name so two CFIE serve
    /// blocks in one module do not collide (same-name blocks collide
    /// loudly via `DuplicateDefinition`, matching the kernel-reg path).
    fn emit_cfie_endpoint_init(
        &mut self,
        builder: &mut FunctionBuilder,
        scope: &str,
        ep: &CfieEndpointInit,
    ) -> Result<cranelift_codegen::ir::Value, CodegenError> {
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

        // 3. optional tokenizer_load
        if let Some(tok_path) = ep.tokenizer_path.as_ref() {
            let (tbytes, _tlen) = cfie_nul_terminated(tok_path);
            let t_id = embed_cfie_bytes(
                &mut self.module,
                &format!("__nsl_cfie_tokenizer_{scope}"),
                tbytes,
            )?;
            let t_gv = self.module.declare_data_in_func(t_id, builder.func);
            let t_ptr = builder.ins().symbol_value(cl_types::I64, t_gv);
            // Return value (tok handle) intentionally dropped in v1 — see
            // the decode-bridge deferral note in the method doc.
            self.compile_call_by_name(builder, "nsl_tokenizer_load", &[t_ptr])?;
        }

        // 4. bake the prompt token ids as a host i64 array (little-endian
        // raw bytes) and return its pointer.
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
        let p_ptr = builder.ins().symbol_value(cl_types::I64, p_gv);
        Ok(p_ptr)
    }

    /// CFIE Tier-A wiring: extract serve-block CFIE config, resolve the
    /// mode, run the six-pass orchestrator against real inputs (GPU
    /// database + `--weights` WeightMap + serve config keys), print the
    /// build report, and stash the plan on `self.last_cfie_plan`.
    ///
    /// Returns the runtime-init values the compiled binary must pass to
    /// the `nsl_cfie_*` init FFIs, or `None` when CFIE is not active.
    ///
    /// `is_disaggregated`: the disaggregated serve branch emits NONE of
    /// the CFIE runtime wiring (ring/slots/registration/finalize) — the
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

        // Feature 1 (G7): emit the direct-indexing decode-attention
        // kernel when the plan selected a static layout and the v1
        // emitter's preconditions hold.  Precondition misses downgrade
        // to a report note, not a build failure — the paged path stays
        // available.
        let mut kv_slots: Option<(i64, i64)> = None;
        if plan.kv.uses_direct_indexing() {
            let per_slot = plan
                .kv
                .direct
                .as_ref()
                .map(|d| d.per_sequence_max_tokens)
                .unwrap_or(0);
            let max_slots = plan.persistent.scheduler.max_active;
            let attn_cfg = crate::cfie_decode_attention::DecodeAttentionConfig {
                n_layers: prepared.shape.n_layers,
                n_heads: prepared.shape.n_heads,
                n_kv_heads: prepared.shape.n_kv_heads,
                head_dim: prepared.shape.head_dim,
                per_slot_max_tokens: per_slot,
                max_slots,
                kv_dtype_bytes: 2,
                sm_version: gpu.sm_version,
            };
            let supported = prepared.shape.head_dim <= 128
                && prepared.shape.n_heads % prepared.shape.n_kv_heads.max(1) == 0
                && per_slot >= 1
                && max_slots >= 1
                && (max_slots as u64) * (per_slot as u64) <= u32::MAX as u64;
            if supported {
                let (ptx, meta) = crate::cfie_decode_attention::emit(&attn_cfg);
                plan.decode_attention_launch = Some(crate::cfie::CfieLaunchSpec {
                    grid_x: if meta.grid_dim_is_n_heads {
                        prepared.shape.n_heads
                    } else {
                        1
                    },
                    block_x: meta.block_dim,
                    smem_dyn_bytes: 0,
                });
                plan.decode_attention_kernel = Some(meta.kernel_name);
                plan.decode_attention_ptx = Some(ptx);
                kv_slots = Some((max_slots as i64, per_slot as i64));
            }
        }

        // Feature 4 (G16): emit the persistent decode-block kernel only
        // when the planner sustained full Level-3 block fusion — the
        // paper's whole-layer claim; Level1/2 stay plan-level with the
        // honest report note — AND the same static-layout preconditions
        // as the decode-attention kernel hold (the block appends into
        // the same baked KV pool that kernel reads).
        if plan.persistent.fusion == crate::cfie_persistent::FusionLevel::Level3
            && plan.kv.uses_direct_indexing()
        {
            let per_slot = plan
                .kv
                .direct
                .as_ref()
                .map(|d| d.per_sequence_max_tokens)
                .unwrap_or(0);
            let max_slots = plan.persistent.scheduler.max_active;
            let s = &prepared.shape;
            let supported = s.head_dim >= 2
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
            if supported {
                let blk_cfg = crate::cfie_persistent_ptx::DecodeBlockConfig {
                    d_model: s.d_model,
                    head_dim: s.head_dim,
                    n_heads: s.n_heads,
                    n_kv_heads: s.n_kv_heads,
                    d_ff: s.d_ff,
                    per_slot_max_tokens: per_slot,
                    max_slots,
                    n_layers: s.n_layers,
                    // Serve keys with the common-architecture defaults;
                    // a model with a different RoPE base (e.g. 500000)
                    // or norm epsilon MUST set them or the baked
                    // constants are numerically wrong.
                    rope_theta: cfg.rope_theta.unwrap_or(10_000.0) as f32,
                    eps: cfg.norm_eps.unwrap_or(1e-5) as f32,
                    sm_version: gpu.sm_version,
                };
                let (ptx, meta) = crate::cfie_persistent_ptx::emit(&blk_cfg);
                // Static .shared declarations are capped at 48 KB per
                // CTA; a bigger footprint would fail module load, so
                // downgrade to plan-level instead of shipping dead PTX.
                if meta.smem_bytes <= 48 * 1024 {
                    // Persistent block: ONE CTA runs one layer's decode
                    // step (grid = 1 by construction, see emitter doc).
                    plan.decode_block_launch = Some(crate::cfie::CfieLaunchSpec {
                        grid_x: 1,
                        block_x: meta.block_dim,
                        smem_dyn_bytes: 0,
                    });
                    plan.decode_block_kernel = Some(meta.kernel_name);
                    plan.decode_block_ptx = Some(ptx);
                }
            }
        }

        // Feature 3 (G13/G14): emit the compiled speculative
        // verification kernels — tree-mask verify attention (each node
        // row's ancestor mask a baked u64 immediate; no mask tensor
        // parameter) + rejection-sampling epilogue.  Requires the same
        // static direct-index KV pool the decode kernels bake: the
        // verify kernel reads draft rows the host appends at
        // seq_len..seq_len+num_nodes, and the host rolls rejected rows
        // back via nsl_cfie_kv_slot_rollback.  Non-tree methods verify
        // a linear K+1 chain (a width-1 tree mask).
        let spec_inputs = plan.speculative.as_ref().map(|spec| {
            let mask = match spec.tree_mask.as_ref() {
                Some(m) => m.clone(),
                None => crate::cfie_speculative::build_tree_mask(spec.config.k_tokens + 1, 1),
            };
            (mask, spec.config.k_tokens)
        });
        if let Some((mask, k_tokens)) = spec_inputs {
            if plan.kv.uses_direct_indexing() {
                let per_slot = plan
                    .kv
                    .direct
                    .as_ref()
                    .map(|d| d.per_sequence_max_tokens)
                    .unwrap_or(0);
                let max_slots = plan.persistent.scheduler.max_active;
                let s = &prepared.shape;
                let num_nodes = mask.num_nodes;
                let supported = s.head_dim >= 1
                    && s.head_dim <= 128
                    && s.n_heads % s.n_kv_heads.max(1) == 0
                    && num_nodes >= 1
                    // K+1 <= 33: a node row's mask bits fit one u64
                    // immediate; wider trees stay plan-level.
                    && num_nodes <= 33
                    && per_slot >= num_nodes
                    && max_slots >= 1
                    && (max_slots as u64) * (per_slot as u64) <= u32::MAX as u64
                    && k_tokens >= 1
                    && (k_tokens as u64) * (s.vocab_size as u64) <= u32::MAX as u64;
                if supported {
                    let verify_cfg = crate::cfie_speculative_ptx::VerifyAttentionConfig {
                        n_heads: s.n_heads,
                        n_kv_heads: s.n_kv_heads,
                        head_dim: s.head_dim,
                        per_slot_max_tokens: per_slot,
                        max_slots,
                        num_nodes,
                        mask_bits: crate::cfie_speculative_ptx::mask_bits_from_tree(&mask),
                        sm_version: gpu.sm_version,
                    };
                    let (vptx, vmeta) =
                        crate::cfie_speculative_ptx::emit_verify_attention(&verify_cfg);
                    let reject_cfg = crate::cfie_speculative_ptx::RejectionConfig {
                        k_tokens,
                        vocab_size: s.vocab_size,
                        sm_version: gpu.sm_version,
                    };
                    let (rptx, rmeta) =
                        crate::cfie_speculative_ptx::emit_rejection_kernel(&reject_cfg);
                    plan.spec_verify_launch = Some(crate::cfie::CfieLaunchSpec {
                        grid_x: if vmeta.grid_dim_is_n_heads { s.n_heads } else { 1 },
                        block_x: vmeta.block_dim,
                        smem_dyn_bytes: 0,
                    });
                    plan.spec_reject_launch = Some(crate::cfie::CfieLaunchSpec {
                        grid_x: if rmeta.grid_dim_is_n_heads { s.n_heads } else { 1 },
                        block_x: rmeta.block_dim,
                        smem_dyn_bytes: 0,
                    });
                    plan.spec_verify_kernel = Some(vmeta.kernel_name);
                    plan.spec_verify_ptx = Some(vptx);
                    plan.spec_reject_kernel = Some(rmeta.kernel_name);
                    plan.spec_reject_ptx = Some(rptx);
                }
            }
        }

        // Feature 6 (G11): bake the grammar's valid-token bitmask into
        // the module image as an initialized .global — the data is a
        // compile-time constant; the decode loop binds its device
        // address to the sampler's grammar_mask_ptr param (G16).
        if let Some(dfa) = plan.grammar.as_ref() {
            plan.grammar_mask_ptx = Some(crate::cfie_grammar_ptx::emit_mask_global(dfa));
        }

        // Feature 5 (G18): per-layer decode-attention kernels with the
        // KV-quant plan's precision baked into each layer's load path.
        // Emitted only when the plan mixes in INT8 (an all-FP16 plan is
        // exactly the base kernel) and every decision is FP16/INT8 (the
        // v1 emitter refuses INT4/BF16).  This family bakes the
        // mixed-precision pool layout, which differs from the uniform
        // f16 pool the base/block/verify kernels assume — the decode
        // loop picks ONE family per build at integration time.
        let mut quant_pool_bytes: Option<i64> = None;
        if plan.kv.uses_direct_indexing() && !plan.kv_quant.layers.is_empty() {
            use crate::cfie_kv_quant::KvPrecision;
            let s = &prepared.shape;
            let precisions: Vec<(KvPrecision, KvPrecision)> = plan
                .kv_quant
                .layers
                .iter()
                .map(|l| (l.k_precision, l.v_precision))
                .collect();
            let all_supported = precisions.iter().all(|(k, v)| {
                matches!(k, KvPrecision::Fp16 | KvPrecision::Int8)
                    && matches!(v, KvPrecision::Fp16 | KvPrecision::Int8)
            });
            let any_int8 = precisions.iter().any(|(k, v)| {
                matches!(k, KvPrecision::Int8) || matches!(v, KvPrecision::Int8)
            });
            let per_slot = plan
                .kv
                .direct
                .as_ref()
                .map(|d| d.per_sequence_max_tokens)
                .unwrap_or(0);
            let max_slots = plan.persistent.scheduler.max_active;
            let supported = all_supported
                && any_int8
                && precisions.len() as u32 == s.n_layers
                && s.head_dim >= 1
                && s.head_dim <= 128
                && s.n_heads % s.n_kv_heads.max(1) == 0
                && per_slot >= 1
                && max_slots >= 1
                && (max_slots as u64) * (per_slot as u64) <= u32::MAX as u64;
            if supported {
                let qcfg = crate::cfie_kv_quant_ptx::QuantDecodeAttentionConfig {
                    n_layers: s.n_layers,
                    n_heads: s.n_heads,
                    n_kv_heads: s.n_kv_heads,
                    head_dim: s.head_dim,
                    per_slot_max_tokens: per_slot,
                    max_slots,
                    sm_version: gpu.sm_version,
                    layer_precisions: precisions,
                };
                // Mixed-precision pool sizing (Cycle 6): when this
                // family wins registration, the device pool is sized
                // by the quant layout, not the uniform-f16 formula.
                quant_pool_bytes =
                    Some(crate::cfie_kv_quant_ptx::total_pool_bytes(&qcfg) as i64);
                let kernels = crate::cfie_kv_quant_ptx::emit_all(&qcfg);
                if let Some((_, meta0)) = kernels.first() {
                    // All layer kernels share one launch shape (same
                    // flash-decode scheme as the base kernel).
                    plan.quant_attention_launch = Some(crate::cfie::CfieLaunchSpec {
                        grid_x: if meta0.grid_dim_is_n_heads { s.n_heads } else { 1 },
                        block_x: meta0.block_dim,
                        smem_dyn_bytes: 0,
                    });
                }
                plan.quant_attention_kernels = kernels
                    .into_iter()
                    .map(|(ptx, meta)| (meta.kernel_name, ptx))
                    .collect();
            }
        }

        // Feature 2 (F2, Cycle 6): the fused decode-sample kernel —
        // kernel kind 1, registered in BOTH families; the host decode
        // loop (nsl_cfie_decode_step) launches it on the final hidden
        // state.  Preconditions mirror the emitter's asserts, F1-style
        // (miss => skip + report note, never a build failure): d_model
        // 1..=8192 (hidden staged in static SMEM), vocab_size a
        // positive multiple of the 128-wide tile (thread t owns row
        // tile_base+t), top_k 1..=64 (serial SMEM candidate list).
        // The plan's sampler program bakes vocab_tile 256 (the
        // HBM-accounting default from `prepare`); the kernel tile is
        // fixed at 128, so re-emit the program against the kernel tile
        // with the SAME params (incl. grammar_masked).
        let mut fused_sample_note: Option<String> = None;
        if plan.sampling.is_fused() {
            let s = &prepared.shape;
            let top_k = plan.sampling.params.top_k;
            let supported = s.d_model >= 1
                && s.d_model <= 8192
                && s.vocab_size >= 128
                && s.vocab_size % 128 == 0
                && (1..=64).contains(&top_k);
            if supported {
                let shape128 = crate::cfie_fused_sample::LmHeadShape {
                    vocab_tile: 128,
                    ..plan.sampling.shape
                };
                let program =
                    crate::cfie_fused_sample::emit_program(plan.sampling.params, shape128);
                // DFA state count sizes the grammar hook's comment and
                // gates its emission; 0 = no grammar hook.
                let grammar_states =
                    plan.grammar.as_ref().map(|d| d.num_states).unwrap_or(0);
                let sample_cfg = crate::cfie_sample_ptx::FusedSampleKernelConfig {
                    d_model: s.d_model,
                    vocab_size: s.vocab_size,
                    vocab_tile: 128,
                    top_k,
                    sm_version: gpu.sm_version,
                    grammar_states,
                };
                let (ptx, meta) = crate::cfie_sample_ptx::emit(&program, &sample_cfg);
                // Splice the baked mask .global into the sampler module
                // so the host can cuModuleGetGlobal it at finalize.
                let ptx = match plan.grammar_mask_ptx.as_ref() {
                    Some(mask) => {
                        crate::cfie_grammar_ptx::splice_mask_into_module(&ptx, mask)
                    }
                    None => ptx,
                };
                plan.fused_sample_launch = Some(crate::cfie::CfieLaunchSpec {
                    grid_x: 1, // single-CTA latency path
                    block_x: meta.block_dim,
                    smem_dyn_bytes: 0,
                });
                plan.fused_sample_kernel = Some(meta.kernel_name);
                plan.fused_sample_ptx = Some(ptx);
            } else {
                fused_sample_note = Some(format!(
                    "note: fused-sample kernel skipped (preconditions: d_model {} \
                     in 1..=8192, vocab_size {} a positive multiple of 128, \
                     top_k {} in 1..=64)\n",
                    s.d_model, s.vocab_size, top_k
                ));
            }
        }

        // G22: estimated decode latency + throughput from the explicit
        // roofline cost model.  Computed here because this is the only
        // point with the resolved model shape, the weights precision, the
        // per-layer KV-quant plan, and the GPU spec all in scope.  The
        // baseline KV footprint is the plan's uniform-FP16 count when the
        // quant pass ran (Full mode); otherwise a shape-derived fallback.
        {
            let s = &prepared.shape;
            let baseline_kv_stored = if plan.kv_quant.bytes_per_token_uniform_fp16 > 0 {
                plan.kv_quant.bytes_per_token_uniform_fp16
            } else {
                // Uniform FP16 KV per stored token across all layers:
                // 2 (K+V) * n_kv_heads * head_dim * 2 bytes * n_layers.
                2 * (s.n_kv_heads as u64) * (s.head_dim as u64) * 2 * (s.n_layers as u64)
            };
            let cost_inputs = crate::cfie_cost::CostModelInputs {
                n_layers: s.n_layers,
                n_heads: s.n_heads,
                n_kv_heads: s.n_kv_heads,
                head_dim: s.head_dim,
                d_model: s.d_model,
                d_ff: s.d_ff,
                vocab_size: s.vocab_size,
                weight_dtype_bytes: s.dtype_bytes,
                cfie_launches_per_token: plan.kernel_launches_per_token_cfie,
                baseline_launches_per_token: plan.kernel_launches_per_token_baseline,
                kv_quant: &plan.kv_quant,
                baseline_kv_bytes_per_stored_token: baseline_kv_stored,
                max_seq: cfg.max_seq.unwrap_or(4096).max(1) as u32,
                batch: plan.persistent.scheduler.max_active,
                gpu,
            };
            plan.cost_estimate = Some(crate::cfie_cost::estimate(&cost_inputs));
        }

        // Build report: the paper's visible artifact (§8).  Provenance +
        // wiring status keep it honest about what this build actually
        // bakes into the binary.
        plan.runtime_wiring_emitted = !is_disaggregated;
        let mut report = plan.render_report();
        report.push_str(&format!(
            "Model-shape provenance: {}\n",
            prepared.shape.provenance
        ));
        if let Some(note) = fused_sample_note {
            report.push_str(&note);
        }
        if is_disaggregated {
            report.push_str(
                "Kernel wiring: DISAGGREGATED serve — CFIE runtime wiring \
                 (request ring, KV slots, kernel registration, engine \
                 finalize) is NOT emitted on this path; the plan and \
                 kernels above are report-only for this build \
                 (monolithic serve gets the wiring).\n",
            );
        } else {
            report.push_str(
                "Kernel wiring: plan + request-ring + KV-slot init + \
                 kernel-family registration + engine finalize in this build; \
                 per-token launches go through the host decode loop \
                 (nsl_cfie_decode_step).\n",
            );
        }
        eprint!("{report}");
        if let Some(path) = self.compile_options.cfie.report_path.clone() {
            if let Err(e) = std::fs::write(&path, &report) {
                eprintln!(
                    "warning: --cfie-report: failed to write {}: {e}",
                    path.display()
                );
            }
        }

        // Cycle 6: registration list for the compile-time-chosen family
        // + KV pool sizing.  The pool allocates only when a KV-consuming
        // kernel (kind 0/2/3/5) registers; the quant family sizes it
        // from the mixed-precision layout, the uniform family from the
        // f16 formula (both cover ALL layers of the baked pool).
        let kernels = crate::cfie::kernel_registrations(&plan);
        let has_kv_consumer = kernels.iter().any(|r| matches!(r.kind, 0 | 2 | 3 | 5));
        let pool_bytes = if !has_kv_consumer {
            None
        } else {
            match crate::cfie::choose_kernel_family(&plan) {
                crate::cfie::CfieKernelFamily::Quant => quant_pool_bytes,
                crate::cfie::CfieKernelFamily::Uniform => {
                    kv_slots.map(|(slots, per_slot)| {
                        let s = &prepared.shape;
                        (s.n_layers as i64)
                            * 2 // K + V halves
                            * (slots * per_slot)
                            * (s.n_kv_heads as i64)
                            * (s.head_dim as i64)
                            * 2 // f16 bytes
                    })
                }
            }
        };

        // Cycle 11: the endpoint's generation driver is wired ONLY when
        // the decode path was actually emitted (kernels registered ->
        // engine finalize will run -> nsl_cfie_generate can drive
        // decode_step).  With no kernels the engine never finalizes, so
        // emitting the binding + generate calls would produce a runtime
        // that always refuses; instead leave `endpoint = None` so the
        // `generate()` rewrite refuses at COMPILE time with a clear
        // message (honest: no half-wired binary).
        let endpoint = if kernels.is_empty() {
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
            // Baked prompt token ids.  Runtime `nsl_cfie_generate` reads a
            // host i64 array, so we bake raw i64s.  A configured `prompt:`
            // is baked as byte-level token ids (the runtime byte-tokenizer
            // convention), clamped into [0, vocab_size) so the runtime's
            // per-token range guard passes; absent that, a single [0]
            // sentinel keeps the body runnable.  NOTE (deferral): a live
            // HF-tokenizer-encoded prompt would need an f64-tensor ->
            // i64-host-array runtime bridge that Cycle 11 does not add;
            // the baked byte-id prompt is the v1 demo path.
            let vocab = prepared.shape.vocab_size.max(1) as i64;
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
                shape: prepared.shape.clone(),
                max_new_tokens: cfg.max_new_tokens.unwrap_or(64).max(1),
                eos_token_id: cfg.eos_token_id.unwrap_or(-1),
            })
        };

        let capacity = plan.persistent.scheduler.ring_buffer.capacity as i64;
        self.last_cfie_plan = Some(plan);
        Ok(Some(CfieRuntimeInit {
            ring_capacity: capacity,
            kv_slots,
            kernels,
            pool_bytes,
            endpoint,
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

        let speculative_config = self.features.speculative_configs.values().next();
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
            for stmt in &endpoint.body.stmts {
                self.compile_stmt(builder, state, stmt)?;
            }
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
