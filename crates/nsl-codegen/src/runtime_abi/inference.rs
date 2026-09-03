//! Serving: KV cache, continuous batching, MoE and decoding.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_ABI_INFERENCE: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // Standalone weight provider and arg parser (M24)
    (
        "nsl_standalone_init_embedded",
        &[types::I64, types::I64],
        None,
    ),
    (
        "nsl_standalone_init_sidecar",
        &[types::I64, types::I64],
        None,
    ),
    ("nsl_standalone_has_weights", &[], Some(types::I64)),
    ("nsl_standalone_args_init", &[types::I64, types::I64], None),
    (
        "nsl_standalone_arg_str",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_standalone_arg_str_default",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_standalone_arg_int",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_standalone_arg_int_default",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_standalone_arg_float",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_standalone_arg_float_default",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_standalone_args_finish", &[], None),
    // Paged KV-cache (M25)
    (
        "nsl_kv_cache_init",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_kv_cache_init_gpu",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    ("nsl_kv_cache_alloc_seq", &[types::I64], Some(types::I64)),
    (
        "nsl_kv_cache_append",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_cache_k_ptr",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_cache_v_ptr",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_kv_cache_free_seq", &[types::I64, types::I64], None),
    (
        "nsl_kv_cache_seq_len",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_cache_seq_blocks",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_cache_seq_num_blocks",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_kv_cache_utilization", &[types::I64], Some(types::F64)),
    ("nsl_kv_cache_destroy", &[types::I64], None),
    // M29: Serving engine
    (
        "nsl_serve_init",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_serve_enqueue",
        &[types::I64, types::I64, types::I64, types::F64, types::F64],
        Some(types::I64),
    ),
    ("nsl_serve_step", &[], Some(types::I64)),
    (
        "nsl_serve_record_token",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_serve_drain_completed", &[], Some(types::I64)),
    ("nsl_serve_has_work", &[], Some(types::I64)),
    ("nsl_serve_completed_count", &[], Some(types::I64)),
    ("nsl_serve_preempt", &[types::I64], Some(types::I64)),
    ("nsl_serve_destroy", &[], Some(types::I64)),
    // --- CFIE: continuous-batching request ring + grammar-table helper ---
    ("nsl_cfie_ring_init", &[types::I64], Some(types::I64)),
    (
        "nsl_cfie_ring_push",
        &[
            types::I64,
            types::I64, // sequence_id, prompt_ptr
            types::I64,
            types::I64, // prompt_len, max_new_tokens
            types::I64,
            types::I64, // grammar_start_state, sampling_packed
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_ring_pop",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64, // six out-pointers mirroring ring_push fields
        ],
        Some(types::I64),
    ),
    ("nsl_cfie_ring_len", &[], Some(types::I64)),
    (
        "nsl_cfie_grammar_transition",
        &[
            types::I64, // table_ptr
            types::I64,
            types::I64, // num_states, vocab_size
            types::I64,
            types::I64, // state, token
        ],
        Some(types::I64),
    ),
    // --- CFIE: KV sequence-slot free-list ---
    (
        "nsl_cfie_kv_slots_init",
        &[types::I64, types::I64], // slot_count, per_slot_tokens
        Some(types::I64),
    ),
    ("nsl_cfie_kv_slot_acquire", &[], Some(types::I64)),
    ("nsl_cfie_kv_slot_release", &[types::I64], Some(types::I64)),
    (
        "nsl_cfie_kv_slot_advance",
        &[types::I64, types::I64], // slot, n_tokens
        Some(types::I64),
    ),
    (
        "nsl_cfie_kv_slot_rollback",
        &[types::I64, types::I64], // slot, n_tokens
        Some(types::I64),
    ),
    ("nsl_cfie_kv_slots_active", &[], Some(types::I64)),
    (
        "nsl_cfie_kv_attach_device",
        &[types::I64, types::I64], // base, bytes
        Some(types::I64),
    ),
    // --- CFIE Cycle 6: compiled-engine registration/lifecycle + launch
    // FFIs (frozen ABI).  All params/returns i64; f32 kernel params are
    // passed as f32::to_bits in the LOW 32 bits.  Kernel kinds:
    // 0=decode_attn, 1=fused_sample, 2=decode_block, 3=spec_verify,
    // 4=spec_reject, 5=quant_attn (layer_idx meaningful only for 5). ---
    (
        "nsl_cfie_register_kernel",
        &[
            types::I64, // kind
            types::I64, // layer_idx
            types::I64, // ptx_ptr
            types::I64, // ptx_len (excludes NUL)
            types::I64, // name_ptr
            types::I64, // name_len (excludes NUL)
            types::I64, // grid_x
            types::I64, // block_x
            types::I64, // smem_dyn_bytes
        ],
        Some(types::I64),
    ),
    ("nsl_cfie_kv_pool_alloc", &[types::I64], Some(types::I64)), // bytes
    ("nsl_cfie_engine_finalize", &[], Some(types::I64)),
    ("nsl_cfie_engine_destroy", &[], Some(types::I64)),
    // --- CFIE Cycle 9: runtime weight binding (production upload FFIs).
    // Cast host f32 [out][in] row-major -> device f16/f32, persistent
    // pool, engine-tracked; reset frees them. ---
    (
        "nsl_cfie_upload_weight_f16",
        &[
            types::I64, // host_f32_ptr
            types::I64, // n_elems
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_upload_weight_f32",
        &[
            types::I64, // host_f32_ptr
            types::I64, // n_elems
        ],
        Some(types::I64),
    ),
    ("nsl_cfie_weights_reset", &[], Some(types::I64)),
    (
        "nsl_cfie_launch_decode_attn",
        &[
            types::I64, // q_ptr
            types::I64, // out_ptr
            types::I64, // layer_idx
            types::I64, // slot_idx
            types::I64, // seq_len
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_fused_sample",
        &[
            types::I64, // hidden_ptr
            types::I64, // norm_w_ptr
            types::I64, // lm_head_ptr
            types::I64, // out_token_ptr
            types::I64, // rng_seed
            types::I64, // grammar_state
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_decode_block",
        &[
            types::I64, // x_in
            types::I64, // x_out
            types::I64, // wq
            types::I64, // wk
            types::I64, // wv
            types::I64, // wo
            types::I64, // w_gate
            types::I64, // w_up
            types::I64, // w_down
            types::I64, // norm1_w
            types::I64, // norm2_w
            types::I64, // layer_idx
            types::I64, // slot_idx
            types::I64, // pos
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_spec_verify",
        &[
            types::I64, // q_ptr
            types::I64, // out_ptr
            types::I64, // layer_idx
            types::I64, // slot_idx
            types::I64, // seq_len
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_spec_reject",
        &[
            types::I64, // target_probs_ptr
            types::I64, // draft_probs_ptr
            types::I64, // draft_tokens_ptr
            types::I64, // rng_seed
            types::I64, // out_accepted_ptr
            types::I64, // out_correction_token_ptr
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_quant_attn",
        &[
            types::I64, // layer_idx (selects the (5, layer) registration)
            types::I64, // q_ptr
            types::I64, // out_ptr
            types::I64, // slot_idx
            types::I64, // seq_len
            types::I64, // k_scale_bits (f32 bits, low 32)
            types::I64, // v_scale_bits (f32 bits, low 32)
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_decode_step",
        &[
            types::I64, // x_buf_a
            types::I64, // x_buf_b
            types::I64, // layer_weights_ptr (host array: n_layers x 9 device ptrs)
            types::I64, // n_layers
            types::I64, // norm_w_ptr (final-norm gamma)
            types::I64, // lm_head_ptr
            types::I64, // slot_idx
            types::I64, // pos
            types::I64, // rng_seed
            types::I64, // grammar_state
            types::I64, // out_token_ptr (device u32)
        ],
        Some(types::I64),
    ),
    // --- CFIE Cycle 10: model binding + generation driver. bind_model
    // resolves an NslModel's host f32 weights by the HF-Llama name
    // convention, uploads them, and records the device weight table;
    // generate drives the decode loop over a prompt; generate_reset
    // clears the binding without freeing the weight buffers. ---
    (
        "nsl_cfie_bind_model",
        &[
            types::I64, // model_handle (NslModel*)
            types::I64, // n_layers
            types::I64, // d_model
            types::I64, // n_heads
            types::I64, // n_kv_heads
            types::I64, // head_dim
            types::I64, // d_ff
            types::I64, // vocab_size
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_generate",
        &[
            types::I64, // prompt_tokens_ptr (host i64 array)
            types::I64, // prompt_len
            types::I64, // max_new_tokens
            types::I64, // eos_token_id
            types::I64, // rng_seed
            types::I64, // out_tokens_ptr (host i64 array)
            types::I64, // out_cap
        ],
        Some(types::I64),
    ),
    ("nsl_cfie_generate_reset", &[], Some(types::I64)),
    // --- CFIE Cycle 12: host token-buffer <-> tokenizer-tensor bridge.
    // tokens_to_tensor turns generate's out-buffer into the 1-D f64
    // tensor nsl_tokenizer_decode consumes (text output); tensor_to_tokens
    // turns nsl_tokenizer_encode's tensor into generate's host i64 prompt
    // array (runtime-encoded prompt). ---
    (
        "nsl_cfie_tokens_to_tensor",
        &[
            types::I64, // tokens_ptr (host i64 array)
            types::I64, // count
        ],
        Some(types::I64), // NslTensor* (1-D f64), or 0 on bad args
    ),
    (
        "nsl_cfie_tensor_to_tokens",
        &[
            types::I64, // tensor_ptr (1-D f64 NslTensor*)
            types::I64, // out_ptr (host i64 buffer)
            types::I64, // cap
        ],
        Some(types::I64), // FULL token count (> cap = truncated), or -1
    ),
    // --- CFIE Cycle 13 (G15 draft-model-in-binary): draft-model binding
    // + engine-held draft KV pool + speculative decode driver.  The
    // serve wiring emits bind_draft_model/draft_pool_alloc at serve init
    // (after the target bind) and speculative_generate from the
    // endpoint's generate() when the speculative draft is configured;
    // the launch FFIs are the kind-6/7/8 wrappers the driver uses
    // internally (registered for ABI completeness + direct testing). ---
    (
        "nsl_cfie_bind_draft_model",
        &[
            types::I64, // model_handle (NslModel*)
            types::I64, // n_layers (draft)
            types::I64, // d_model (draft)
            types::I64, // n_heads (draft)
            types::I64, // n_kv_heads (draft)
            types::I64, // head_dim (draft)
            types::I64, // d_ff (draft)
            types::I64, // vocab_size (MUST equal the target binding's)
        ],
        Some(types::I64),
    ),
    ("nsl_cfie_draft_pool_alloc", &[types::I64], Some(types::I64)), // bytes
    ("nsl_cfie_draft_reset", &[], Some(types::I64)),
    (
        "nsl_cfie_launch_draft_block",
        &[
            types::I64, // x_in
            types::I64, // x_out
            types::I64, // layer_idx (draft weight table is engine-held)
            types::I64, // pos
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_draft_sample",
        &[
            types::I64, // hidden_ptr
            types::I64, // out_token_ptr (device u32)
            types::I64, // out_prob_ptr (device f32)
            types::I64, // rng_seed (accepted-unused; greedy v1)
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_launch_verify_probs",
        &[
            types::I64, // hidden_ptr
            types::I64, // out_probs_ptr (device f32 x vocab)
        ],
        Some(types::I64),
    ),
    (
        "nsl_cfie_speculative_generate",
        &[
            types::I64, // prompt_tokens_ptr (host i64 array)
            types::I64, // prompt_len
            types::I64, // max_new_tokens
            types::I64, // eos_token_id
            types::I64, // rng_seed
            types::I64, // k_tokens (MUST match the kind-4 kernel's K)
            types::I64, // out_tokens_ptr (host i64 array)
            types::I64, // out_cap
        ],
        Some(types::I64),
    ),
    // --- M32: MoE runtime functions ---
    (
        "nsl_moe_route",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_moe_scatter",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_expert_parallel_matmul",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_moe_gather",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_moe_all_to_all",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_moe_aux_loss",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_moe_dispatch_full",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // CPDT Part III v1 production-forward (M32 gap closure): same as v1
    // plus `experts_ptr`, `hidden_dim`, `intermediate_dim` (3 extra i64
    // args, total 8). Returns NslTensor `[total_tokens, intermediate_dim]`
    // (note: trailing dim differs from v1's `[total_tokens, hidden_dim]`
    // identity output). See crates/nsl-runtime/src/moe/ffi.rs.
    (
        "nsl_moe_dispatch_full_v2",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // CPDT Part III v2.2 paper-faithful MoE FFN: per-expert kernel is
    // `up → SiLU → down` instead of v2's single matmul. 10 i64 args:
    // tokens, logits, experts_up, experts_down, num_experts, top_k,
    // capacity_factor_bits, hidden_dim, intermediate_dim, activation_kind,
    // experts_up_bias_ptr, experts_down_bias_ptr (v2.11: bias args are
    // nullable — pass 0 for no bias).
    // Returns NslTensor `[total_tokens, hidden_dim]` (back to hidden,
    // unlike v2's intermediate). See nsl-runtime/src/moe/ffi.rs.
    (
        "nsl_moe_dispatch_full_v3",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // CPDT Part III v2.5+v2.8 Mixtral gated MoE FFN: per-expert kernel is
    // `gate_act(gate) * up → down` where gate_act is selected by
    // gate_activation_kind. 11 i64 args: tokens, logits, experts_gate,
    // experts_up, experts_down, num_experts, top_k, capacity_factor_bits,
    // hidden_dim, intermediate_dim, gate_activation_kind (v2.8: 1=SwiGLU,
    // 2=GeGLU, 3=ReGLU). Output shape matches v3
    // `[total_tokens, hidden_dim]`. See nsl-runtime/src/moe/ffi.rs.
    (
        "nsl_moe_dispatch_full_v4",
        // v2.14: 14 i64 args. Positions 12+13+14 are
        // experts_{gate,up,down}_bias_ptr (nullable, 0 = no bias).
        // Codegen always emits iconst(0) for these in the 5-arg
        // source form; the 8-arg form threads source-supplied bias
        // expressions through (mirrors v3's 4/6 pattern in v2.12).
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // --- M33: Speculative decoding runtime functions ---
    (
        "nsl_speculative_draft",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_speculative_verify",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_speculative_build_tree",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_speculative_verify_tree",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_page_branch",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_page_cow_copy",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tree_attention",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    (
        "nsl_speculative_cleanup",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_speculative_decode_step",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // --- M44: Constrained decoding (grammar FSM) ---
    (
        "nsl_grammar_init",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_grammar_step",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_grammar_apply_mask",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_grammar_is_accept", &[types::I64], Some(types::I64)),
    ("nsl_grammar_start_state", &[], Some(types::I64)),
    ("nsl_grammar_destroy", &[], Some(types::I64)),
    // M44b: Constrained decoding serve integration
    (
        "nsl_serve_apply_grammar",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_serve_advance_grammar",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_serve_set_grammar",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
];
