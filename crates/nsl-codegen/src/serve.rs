//! M29: Serve block codegen.

use cranelift_codegen::ir::{types as cl_types, InstBuilder};
use cranelift_frontend::FunctionBuilder;

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
}

struct WorkerConfigSpec {
    max_seq_len: i64,
    kv_blocks: i64,
    speculative_tokens: i64,
    speculative_method: i64,
    speculative_tree_width: i64,
    speculative_temperature_bits: i64,
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
                // below — not runtime expressions.
                "kv_layout" | "kv_quant" | "target_gpu" | "n_layers" | "n_kv_heads"
                | "kv_heads" | "n_heads" | "head_dim" | "d_model" | "d_ff" | "vocab_size"
                | "rope_theta" | "norm_eps" => {}
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

        // ── CFIE Tier-A wiring (audit gap G1) ────────────────────────
        // Extract CFIE config from the serve block, resolve the mode
        // (CLI --cfie > @cfie decorator > implicit via CFIE keys), run
        // the orchestrator, and surface the build report.  The plan
        // drives the report, the request-ring + KV-slot init calls, and
        // (static layouts) the direct-index decode-attention kernel;
        // the decode-loop launch path lands with audit gap G16.
        let cfie_init = self.run_cfie_for_serve(serve)?;

        let is_disaggregated = prefill_workers > 1 || decode_workers > 1;

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
            }

            for endpoint in &serve.endpoints {
                for stmt in &endpoint.body.stmts {
                    self.compile_stmt(builder, state, stmt)?;
                }
            }

            self.compile_call_by_name(builder, "nsl_serve_destroy", &[])?;
        }

        // M30: Tear down tensor parallelism if multi-device
        if self.features.world_size > 1 {
            self.compile_call_by_name(builder, "nsl_tp_destroy", &[])?;
        }

        Ok(())
    }

    /// CFIE Tier-A wiring: extract serve-block CFIE config, resolve the
    /// mode, run the six-pass orchestrator against real inputs (GPU
    /// database + `--weights` WeightMap + serve config keys), print the
    /// build report, and stash the plan on `self.last_cfie_plan`.
    ///
    /// Returns the runtime-init values the compiled binary must pass to
    /// the `nsl_cfie_*` init FFIs, or `None` when CFIE is not active.
    fn run_cfie_for_serve(
        &mut self,
        serve: &ServeBlock,
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
                plan.quant_attention_kernels = crate::cfie_kv_quant_ptx::emit_all(&qcfg)
                    .into_iter()
                    .map(|(ptx, meta)| (meta.kernel_name, ptx))
                    .collect();
            }
        }

        // Build report: the paper's visible artifact (§8).  Provenance +
        // wiring status keep it honest about what this build actually
        // bakes into the binary.
        let mut report = plan.render_report();
        report.push_str(&format!(
            "Model-shape provenance: {}\n",
            prepared.shape.provenance
        ));
        report.push_str(
            "Kernel wiring: plan + request-ring + KV-slot init in this \
             build; kernel launch paths land with the decode-loop \
             integration cycle.\n",
        );
        eprint!("{report}");
        if let Some(path) = self.compile_options.cfie.report_path.clone() {
            if let Err(e) = std::fs::write(&path, &report) {
                eprintln!(
                    "warning: --cfie-report: failed to write {}: {e}",
                    path.display()
                );
            }
        }

        let capacity = plan.persistent.scheduler.ring_buffer.capacity as i64;
        self.last_cfie_plan = Some(plan);
        Ok(Some(CfieRuntimeInit {
            ring_capacity: capacity,
            kv_slots,
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
