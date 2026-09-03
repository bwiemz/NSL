//! The training loop's runtime: tape, checkpointing, data loading,
//! gradient plumbing and weight streaming.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_ABI_TRAINING: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // Training mode
    ("nsl_set_training_mode", &[types::I8], None),
    ("nsl_is_training", &[], Some(types::I8)),
    (
        "nsl_tensor_full",
        &[types::I64, types::F64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_arange",
        &[types::F64, types::F64, types::F64],
        Some(types::I64),
    ),
    // Autodiff tape management
    ("nsl_tape_start", &[types::I64], None),
    ("nsl_tape_stop", &[], None),
    (
        "nsl_tape_backward",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tape_backward_train",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tape_pause", &[], None),
    ("nsl_tape_resume", &[], None),
    // Gradient checkpointing
    (
        "nsl_checkpoint_record",
        &[types::I64, types::I64, types::I64, types::I64],
        None,
    ),
    // Gradient clipping (M14)
    ("nsl_clip_grad_norm", &[types::I64, types::F64], None),
    // Collect all tensor params from a model struct (recursive, magic-probed)
    (
        "nsl_collect_model_params",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // Debug training: gradient checksum (--debug-training)
    ("nsl_debug_grad_checksum", &[types::I64, types::I64], None),
    // P0.3 gradient-integrity gate (--grad-integrity)
    ("nsl_grad_integrity_arm", &[], None),
    ("nsl_grad_integrity_check", &[types::I64, types::I64], None),
    // (num_params, expected_notes_per_param) — the second argument is what
    // lets the runtime distinguish "every micro-batch reached this param"
    // from "k of N did", which the CSLA window bracket otherwise merges away.
    (
        "nsl_grad_integrity_step_begin",
        &[types::I64, types::I64],
        None,
    ),
    ("nsl_grad_integrity_note", &[types::I64, types::I64], None),
    ("nsl_grad_integrity_step_end", &[], None),
    // Prefetch tensor to GPU asynchronously
    ("nsl_tensor_prefetch", &[types::I64, types::I64], None),
    // Item 2 (2026-08-25): refuse a host-resident dense-float step input on
    // a GPU-parameter train — the left-operand device-reconciliation rule
    // would otherwise silently drag the whole graph to the host.
    ("nsl_train_input_device_guard", &[types::I64, types::I64], None),
    // Checkpoint I/O (M14)
    (
        "nsl_model_save",
        &[types::I64, types::I64, types::I64, types::I64],
        None,
    ),
    (
        "nsl_model_load",
        &[types::I64, types::I64, types::I64],
        None,
    ),
    // Milestone B + item 8: full training-state checkpoint (θ .nslm + .optim
    // sidecar with m/v moments, micro-batch step counter, data position and
    // RNG state). Save: (path_ptr, path_len, names_list, param_list,
    // state_list_1, state_list_2, step_count, dataloader_handle_or_0,
    // train_epoch). Load: (path_ptr, path_len, param_list, state_list_1,
    // state_list_2, dataloader_handle_or_0) -> saved step counter; the
    // restored training epoch comes back through nsl_train_resume_epoch.
    (
        "nsl_train_checkpoint_save",
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
        None,
    ),
    (
        "nsl_train_checkpoint_load",
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
    ("nsl_train_resume_epoch", &[], Some(types::I64)),
    // Data sources (M19)
    (
        "nsl_load_jsonl",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_load_csv",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_load_mmap",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // DataLoader (M19)
    (
        "nsl_dataloader_create",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_dataloader_start", &[types::I64], None),
    ("nsl_dataloader_next_batch", &[types::I64], Some(types::I64)),
    ("nsl_dataloader_reset", &[types::I64], None),
    ("nsl_dataloader_stop", &[types::I64], None),
    ("nsl_dataloader_free", &[types::I64], None),
    // Item 8: resumable data position. `slot` is the loader's next DELIVERY
    // slot (not a batch count — the ragged-packed-tail sentinel makes those
    // differ), `identity` fingerprints the corpus + geometry it indexes.
    ("nsl_dataloader_epoch", &[types::I64], Some(types::I64)),
    ("nsl_dataloader_slot", &[types::I64], Some(types::I64)),
    ("nsl_dataloader_identity", &[types::I64], Some(types::I64)),
    (
        "nsl_dataloader_resume_to",
        &[types::I64, types::I64, types::I64],
        None,
    ),
    // Packing efficiency (M19)
    ("nsl_packing_efficiency", &[types::I64], Some(types::F64)),
    // CFTP §4.4 G3 (Sprint 4): fused linear-CE FFI signatures.
    // Sprint v3-2 added trailing `dtype_tag` (0=F32 sentinel preserves
    // pre-v3-2 ABI; 1=F16). Sprint v4-1 extended the sentinel space
    // with 2=Bf16 (same single-i64 trailing arg — no ABI bump).
    // The Cranelift IR call sites in wengert_lower.rs derive the tag
    // from the @fused_lm_ce(dtype=...) decorator via
    // `fused_ce_dtype_for_compiler`. Note: v4-2 wengert refuses
    // tag != 0 pending precision_cast plumbing (see review Finding 2);
    // direct FFI tests with caller-managed 16-bit HBM allocation
    // exercise tags 1 and 2 end-to-end.
    // Forward v1 (small vocab, single CTA per row).
    (
        "nsl_fused_linear_ce_forward",
        &[
            types::I64, // ptx_ptr
            types::I64, // kname_ptr
            types::I64, types::I64, types::I64, types::I64, // x, W, bias, targets (raw device ptrs)
            types::I64, types::I64, // loss_out, lse_out
            types::I64, types::I64, types::I64, types::I64, // b, s, v, h
            types::I64, // smem_bytes
            types::I64, // dtype_tag (Sprint v3-2 / extended v4-1; 0=F32, 1=F16, 2=Bf16)
        ],
        Some(types::I64),
    ),
    // Forward large-vocab (Sprint 3 two-kernel path, vocab > 8192).
    (
        "nsl_fused_linear_ce_forward_large",
        &[
            types::I64, // ptx_ptr
            types::I64, // partials_kname_ptr
            types::I64, // finalize_kname_ptr
            types::I64, types::I64, types::I64, types::I64, // x, W, bias, targets
            types::I64, // partials_ptr (caller-owned scratch)
            types::I64, types::I64, // loss_out, lse_out
            types::I64, types::I64, types::I64, types::I64, // b, s, v, h
            types::I64, // num_tiles
            types::I64, // smem_bytes
            types::I64, // dtype_tag (Sprint v3-2 / extended v4-1; 0=F32, 1=F16, 2=Bf16)
        ],
        Some(types::I64),
    ),
    // Backward (shared between v1 and large-vocab forward paths).
    (
        "nsl_fused_linear_ce_backward",
        &[
            types::I64, // ptx_ptr
            types::I64, // kname_ptr
            types::I64, // grad_output_bits (f32 bits packed into i64)
            types::I64, types::I64, types::I64, types::I64, // x, W, bias, targets
            types::I64, // lse_ptr
            types::I64, types::I64, types::I64, // dx_out, dW_out, dbias_out
            types::I64, types::I64, types::I64, types::I64, // b, s, v, h
            types::I64, // num_valid
            types::I64, // smem_bytes
            types::I64, // dtype_tag (Sprint v3-2 / extended v4-1; 0=F32, 1=F16, 2=Bf16)
        ],
        Some(types::I64),
    ),
    // Sprint 2.5: GEMM-chunked fused linear-CE (production path for large
    // vocab and every biasless head). No PTX/kname/smem args — chunk
    // kernels are runtime constants, heavy math is cuBLAS. bias/dbias
    // pointers are the literal 0 when has_bias == 0.
    (
        "nsl_fused_linear_ce_forward_gemm",
        &[
            types::I64, types::I64, types::I64, types::I64, // x, W, bias_or_0, targets
            types::I64, types::I64, // loss_out, lse_out
            types::I64, types::I64, types::I64, types::I64, // b, s, v, h
            types::I64, // has_bias
        ],
        Some(types::I64),
    ),
    (
        "nsl_fused_linear_ce_backward_gemm",
        &[
            types::I64, // grad_output_bits (f32 bits packed into i64)
            types::I64, types::I64, types::I64, types::I64, // x, W, bias_or_0, targets
            types::I64, // lse_ptr
            types::I64, types::I64, types::I64, // dx_out, dW_out, dbias_or_0
            types::I64, types::I64, types::I64, types::I64, // b, s, v, h
            types::I64, // num_valid
            types::I64, // has_bias
        ],
        Some(types::I64),
    ),
    // CPKD: fused KL-CE distillation loss (forward + backward).
    //
    // ABI LOCK-STEP: these declarations, the call sites in
    // wengert_lower.rs (lower_fused_kl_ce_forward / _backward_extract),
    // and the runtime extern "C" fns in
    // crates/nsl-runtime/src/fused_kl_ce.rs must agree on arg count and
    // order BY HAND — there is no compile-time cross-check (see the
    // 16-vs-18-arg Tier-B lesson above nsl_flash_attention_backward).
    // Forward = 20 args; backward = 23 args.
    (
        "nsl_fused_kl_ce_forward",
        &[
            types::I64, // ptx_ptr
            types::I64, // kname_ptr
            types::I64, types::I64, types::I64, // x_s, W_s, bias_s (raw device ptrs)
            types::I64, types::I64, types::I64, // x_t, W_t, bias_t
            types::I64, // targets
            types::I64, // loss_out
            types::I64, types::I64, types::I64, // lse_s1_out, lse_st_out, lse_tt_out
            types::I64, types::I64, types::I64, types::I64, // rows, v, hs, ht
            types::I64, types::I64, // alpha_bits, temp_bits (f32 bits in i64)
            types::I64, // smem_bytes
        ],
        Some(types::I64),
    ),
    (
        "nsl_fused_kl_ce_backward",
        &[
            types::I64, // ptx_ptr
            types::I64, // kname_ptr
            types::I64, // grad_output_bits (f32 bits in i64)
            types::I64, types::I64, types::I64, // x_s, W_s, bias_s
            types::I64, types::I64, types::I64, // x_t, W_t, bias_t
            types::I64, // targets
            types::I64, types::I64, types::I64, // lse_s1, lse_st, lse_tt
            types::I64, types::I64, types::I64, // dxs_out, dws_out, dbs_out (student only — I-11)
            types::I64, types::I64, types::I64, types::I64, // rows, v, hs, ht
            types::I64, types::I64, // alpha_bits, temp_bits
            types::I64, // num_valid
        ],
        Some(types::I64),
    ),
    // --- M40b: Backward context for source-to-source AD (handle-based) ---
    ("nsl_backward_ctx_new", &[types::I64], Some(types::I64)),
    (
        "nsl_backward_ctx_save",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_backward_ctx_load",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_backward_ctx_free", &[types::I64], Some(types::I64)),
    // --- M43: Gradient accumulation (ABI-fixed) ---
    (
        "nsl_grad_accumulate_add",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ), // (dst, src, num_elems)
    ("nsl_grad_zero", &[types::I64, types::I64], Some(types::I64)), // (grad_ptr, num_elems)
    (
        "nsl_grad_all_reduce",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // Item 4 (2026-08-25): the resolved train/optimizer/scheduler record,
    // installed at TRAIN-BLOCK entry (per block, not per program — a module
    // can hold several train blocks). Joins the .optim sidecar as checkpoint
    // identity; see nsl-runtime/src/train_config_record.rs for the policy.
    (
        "nsl_set_train_config_record",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // D1 (CSLA Stage-2): window-backward anti-vacuity counter — mark once per
    // accumulation window at the head of the buffered backward phase, plus the
    // in-process getter for gates.
    ("nsl_csla_window_mark", &[], None),
    ("nsl_csla_window_count", &[], Some(types::I64)),
    // D1b: one-time pointer-tie guard over param_list (aborts loudly on any
    // aliased pair — per-layer in-place updates would corrupt the alias).
    ("nsl_csla_assert_params_unaliased", &[types::I64], None),
    // Fused-CE tape-carry gates: fused linear-CE launch counter (0 =
    // forward, 1 = forward_large, 2 = backward) — NSL-callable as
    // fused_lce_launch_count(k).
    ("nsl_fused_lce_launch_count", &[types::I64], Some(types::I64)),
    // Milestone B: weight-stream stat surface — NSL-callable as
    // weight_stream_stat(kind); kinds documented on the runtime fn.
    ("nsl_weight_stream_stat", &[types::I64], Some(types::I64)),
    // Fused-CE targets dtype bridge: the kernels read targets as s64 but
    // NSL GPU labels are f32 — materialize/free a device i64 copy around
    // each fused forward/backward FFI.
    // Second arg: expected row count (decorator batch*seq) — the runtime
    // aborts loudly on mismatch instead of overreading the staging buffer.
    ("nsl_fused_lce_targets_i64_alloc", &[types::I64, types::I64], Some(types::I64)),
    // (x_tensor, w_tensor, batch, seq, vocab_size, hidden_size, site_code)
    // -> void. Aborts when the decorator hints disagree with the head
    // tensors. batch and seq stay SEPARATE: collapsing them to rows here
    // would let a swapped pair through, and the backward builds dx from the
    // pair. site_code names the caller in the diagnostic (0 = @fused_lm_ce,
    // 1/2 = @fused_kl_ce student/teacher) so a refusal never blames the
    // wrong decorator or the wrong hint name.
    (
        "nsl_fused_lce_pin_hint_extents",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        None,
    ),
    ("nsl_fused_lce_targets_i64_free", &[types::I64], None),
    // D2b weight streaming: pointer-identity host offload of model params
    // (side-table mirrors; tensor pointers never change).
    ("nsl_weight_stream_register", &[types::I64], None),
    ("nsl_weight_stream_upload", &[types::I64], None),
    ("nsl_weight_stream_evict", &[types::I64, types::I64], None),
    ("nsl_weight_stream_upload_all", &[], None),
    // Item 12: re-evict everything after a scoped `upload_all` around a
    // model-touching callback. Arg = writeback (1 if the callback may mutate).
    ("nsl_weight_stream_reevict_all", &[types::I64], None),
    // Item 10: contiguous layer-pack transfers. Arg = NslList of the pack's
    // param tensor pointers; evict also takes writeback.
    ("nsl_weight_stream_upload_pack", &[types::I64], None),
    ("nsl_weight_stream_evict_pack", &[types::I64, types::I64], None),
    // Item 11: async double-buffer prefetch + event-ordered await.
    ("nsl_weight_stream_prefetch_pack", &[types::I64], None),
    ("nsl_weight_stream_evict_pack_async", &[types::I64], None),
    ("nsl_weight_stream_await_pack", &[types::I64], None),
    ("nsl_weight_stream_teardown", &[], None),
    ("nsl_weight_stream_upload_count", &[], Some(types::I64)),
    (
        "nsl_tensor_logsoftmax",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_repeat",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_rope_inverse",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
];
