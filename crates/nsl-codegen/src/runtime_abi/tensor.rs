//! Tensor kernels behind the language surface: fused elementwise chains,
//! in-place variants, casts, attention, and hand-written backward kernels.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_ABI_TENSOR: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // Fused elementwise-chain launcher (MFU campaign C3):
    // (ptx, kname, descriptor, desc_len, in0..in5, n_inputs) -> result handle.
    (
        "nsl_fused_ew_chain",
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
        ],
        Some(types::I64),
    ),
    // FASE fused scaled-add epilogue (p4): m += s * g, in place, void.
    (
        "nsl_tensor_scalar_mul_add_inplace",
        &[types::I64, types::I64, types::F64],
        None,
    ),
    // Item 7 fused weight-gradient accumulate: m += s * (x^T @ g), in place,
    // void. Args (m_partial, x, g, s) — x/g are the PRE-transpose activation
    // and the upstream gradient; the transpose is folded into the GEMM.
    (
        "nsl_tensor_wgrad_accum",
        &[types::I64, types::I64, types::I64, types::F64],
        None,
    ),
    // WRGA B.3 Task 4: fused LoRA/IA³ adapter matmul FFIs.
    // LoRA args: (x_ptr, w_ptr, a_ptr, b_ptr, scale_f64, kernel_handle_i64).
    // The scale is f64 at the FFI boundary because NSL FloatLiteral is f64;
    // the runtime narrows to f32 internally.
    (
        "nsl_adapter_fused_lora_matmul",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::F64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // IA³ args: (x_ptr, w_ptr, ia3_scale_ptr, kernel_handle_i64)
    (
        "nsl_adapter_fused_ia3_matmul",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // B.3.1 GatedLoRA args: (x_ptr, w_ptr, a_ptr, b_ptr, scale_f64, gate_ptr, kernel_handle_i64).
    // Body is a stub returning the base matmul (x @ W) until Task 5.0.c registers the real
    // fused PTX kernel.  The FFI declaration is needed here so compile_call resolves the
    // callee symbol and falls through to compile_traced_call rather than compile_indirect_call.
    (
        "nsl_adapter_fused_gatedlora_matmul",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::F64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // WRGA B.3 Task 5.6: fused-PTX runtime registry registration.
    // Args: (handle_i64, ptx_ptr_i64, ptx_len_i64, name_ptr_i64, name_len_i64).
    // Called from `main` preamble, one call per unique (m,n,k,rank,sm) key.
    (
        "nsl_wrga_register_fused_ptx",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        None,
    ),
    // Fused elementwise operations (M31 fusion lowering)
    (
        "nsl_fused_elementwise_2",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_fused_elementwise_1",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_fused_matmul_epilogue",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // FBIP Phase 2: unconditional in-place variants (compiler-guaranteed single-use)
    ("nsl_tensor_relu_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_exp_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_log_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_sqrt_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_abs_inplace", &[types::I64], Some(types::I64)),
    (
        "nsl_tensor_sigmoid_inplace",
        &[types::I64],
        Some(types::I64),
    ),
    ("nsl_tensor_tanh_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_neg_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_sign_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_gelu_inplace", &[types::I64], Some(types::I64)),
    ("nsl_tensor_silu_inplace", &[types::I64], Some(types::I64)),
    // CFTP §4.3 / Tier A activation — extract raw device pointer from
    // NslTensor* for use by nsl_packing_metadata_set. Returns 0 when
    // tensor_ptr == 0. See spec 2026-05-17-pca-rope-activation-design.md.
    ("nsl_tensor_data_ptr", &[types::I64], Some(types::I64)),
    // CFTP §4.3 / Tier A activation — thread-local registry for the
    // segment_ids/doc_starts pointers. Train block sets per step;
    // CSHA call sites read.
    ("nsl_packing_metadata_set", &[types::I64, types::I64], None),
    ("nsl_packing_metadata_get_segment_ids", &[], Some(types::I64)),
    ("nsl_packing_metadata_get_doc_starts", &[], Some(types::I64)),
    // PCA Tier A (spec §6.1) — mismatch warning: warns once if a
    // segment-masked module sees no segment_ids in the first N steps.
    ("nsl_pca_packing_mismatch_check", &[types::I64], None),
    // CPDT precision-adaptive optimizer: cast / zeros helpers
    ("nsl_tensor_cast", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_cast_into", &[types::I64, types::I64], None),
    ("nsl_tensor_zeros_like_dtype", &[types::I64, types::I64], Some(types::I64)),
    // Optimizer-state offload (scaling campaign item 4): host-resident f32
    // zeros with the template's shape, regardless of the template's device.
    ("nsl_tensor_zeros_like_host_f32", &[types::I64], Some(types::I64)),
    // Offload P0.2: async copy-back (CONSUMES src — replaces the emitted
    // copy_data+free pair) + the once-per-step drain point.
    ("nsl_tensor_copy_data_async", &[types::I64, types::I64], None),
    ("nsl_offload_drain", &[], None),
    // Offload P0.3 (offload x reduced-precision composition): host state
    // at the planned dtype + the cross-device quant/dequant envelope.
    ("nsl_tensor_zeros_like_host_dtype", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_cast_to_host_into", &[types::I64, types::I64], None),
    ("nsl_tensor_cast_from_host", &[types::I64, types::I64], Some(types::I64)),
    // CFTP v6 forward inline-cast wrappers: src_ptr -> new tensor (scope-tracked).
    ("nsl_tensor_to_bf16", &[types::I64], Some(types::I64)),
    ("nsl_tensor_to_fp16", &[types::I64], Some(types::I64)),
    ("nsl_tensor_to_f32", &[types::I64], Some(types::I64)),
    // M52c: CSR sparse matmul (row_ptrs, col_indices, values, B, nrows, ncols, nnz) -> C
    (
        "nsl_sparse_matmul",
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
    // Fused RoPE backward: -rotate_half(dy) in one launch (bit-exact)
    ("nsl_tensor_rotate_half_neg", &[types::I64], Some(types::I64)),
    (
        "nsl_tensor_softmax",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // Item 9: fused RMSNorm input-gradient (dy, x, gamma, eps) -> dx.
    (
        "nsl_rmsnorm_dx_backward",
        &[types::I64, types::I64, types::I64, types::F64],
        Some(types::I64),
    ),
    // P5 slice C: fused RMSNorm dx + residual fold (dy, x, gamma, res, eps) -> dx+res.
    (
        "nsl_rmsnorm_dx_backward_add",
        &[types::I64, types::I64, types::I64, types::I64, types::F64],
        Some(types::I64),
    ),
    // P5 item 20 slice A: fused RMSNorm gamma-gradient (dy, x, gamma, eps) -> dgamma.
    (
        "nsl_rmsnorm_dgamma_backward",
        &[types::I64, types::I64, types::I64, types::F64],
        Some(types::I64),
    ),
    // Source AD: reduce gradient to match parameter shape (matmul broadcast backward)
    (
        "nsl_tensor_reduce_to_shape",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // Source-AD dropout forward: returns an NslList [out, mask] so the
    // compiled backward can consume the exact RNG mask (two-op split,
    // wengert.rs DropoutMask).
    (
        "nsl_tensor_dropout_fwd_mask",
        &[types::I64, types::F64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_conv2d",
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
    // Source-AD conv2d backward FFIs: (grad, input, weight, sh, sw, ph, pw) -> grad.
    (
        "nsl_conv2d_input_backward",
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
        "nsl_conv2d_weight_backward",
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
        "nsl_conv2d_bias_backward",
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
    // Reify grad_output to the conv2d output shape once per node, shared by
    // the 3 FFIs above: (grad, input, weight, sh, sw, ph, pw) -> grad.
    (
        "nsl_materialize_conv_output_grad",
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
        "nsl_tensor_maxpool2d",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // FlashAttention-2 launch wrappers (M27)
    (
        "nsl_flash_attention",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64, // q, k, v, out
            types::I64, // logsumexp (backward aux, 0=skip)
            types::I64, // scale
            types::I64,
            types::I64,
            types::I64,
            types::I64, // batch, heads, seq_len, head_dim
            types::I64,
            types::I64,
            types::I64,
            types::I64, // block_table, k_pool, v_pool, block_size
            types::I64,
            types::I64, // cos, sin (RoPE)
            types::I64,
            types::I64, // seq_ids, seq_lens (M29-ready)
            types::I64, // shared_mem_bytes
            types::I64,
            types::I64, // ptx_ptr, name_ptr
            types::I64,
            types::I64, // block_q, block_kv
            types::I64, // causal (0=false, 1=true)
            // PCA Tier B planner spec §4 — must match the runtime signature
            // (nsl_runtime::flash_attention::nsl_flash_attention, lines 149-150).
            // Non-CSHA path has no Tier-B-on emission, so the caller always
            // passes the disabled sentinel (0, 0). Without these slots in the
            // Cranelift signature, the runtime's `assert_tier_b_sentinels`
            // entry guard reads stack garbage and may abort.
            types::I64, // tier_b_ptx_ptr  (always 0 from non-CSHA caller)
            types::I64, // tier_b_name_ptr (always 0 from non-CSHA caller)
        ],
        Some(types::I64),
    ),
    // CSHA Tier A.1: FA launcher variant carrying per-layer CSHA extras.
    // Same 24-arg prelude as `nsl_flash_attention`, then 9 CSHA args:
    //   x, norm_weight, Wq, Wk, Wv, Wo, rmsnorm_eps_bits, active_heads, d_model.
    // Stub today — forwards to `nsl_flash_attention`; A.2 will light up
    // the CSHA PTX body.
    (
        "nsl_flash_attention_csha",
        &[
            types::I64, types::I64, types::I64, types::I64, // q, k, v, out
            types::I64, // logsumexp
            types::I64, // scale_bits
            types::I64, types::I64, types::I64, types::I64, // batch, heads, seq_len, head_dim
            types::I64, types::I64, types::I64, types::I64, // block_table, k_pool, v_pool, block_size
            types::I64, types::I64, // cos, sin
            types::I64, types::I64, // seq_ids, seq_lens
            types::I64, // shared_mem_bytes
            types::I64, types::I64, // ptx_ptr, name_ptr
            types::I64, types::I64, // block_q, block_kv
            types::I64, // causal
            // CSHA extras:
            types::I64, // x_ptr
            types::I64, // norm_weight_ptr
            types::I64, types::I64, types::I64, types::I64, // Wq, Wk, Wv, Wo
            types::I64, // rmsnorm_eps_bits (f32 as i64)
            types::I64, // active_heads
            types::I64, // d_model
            // PCA Tier A: segment_ids device pointer (0 = unpacked path)
            types::I64, // segment_ids_ptr
            // Tier B extension (planner spec §4):
            types::I64, // tier_b_ptx_ptr
            types::I64, // tier_b_name_ptr
            // PCA §4.3: doc_starts device pointer (0 = identity positions)
            types::I64, // doc_starts_ptr
            // PCA per-doc CTA (Strategy 3 v1): num_docs_or_zero — grid_x
            // override when the kernel name carries the `_per_doc_cta`
            // suffix. Pass 0 for all non-per-doc topologies.
            types::I64, // num_docs_or_zero
        ],
        Some(types::I64),
    ),
    // Gap A: CSHA FA launcher with activation-save pointers for source-AD
    // backward. Identical to `nsl_flash_attention_csha` plus 6 trailing
    // save-pointer args (q_proj, k_proj, v_proj, row_max, row_sum, x_raw).
    // Emitted by `compile_flash_attention_call` when
    // `CshaExtras.save_activations_for_backward` is true (i.e. inside a
    // `@train` block with CSHA fused).
    (
        "nsl_flash_attention_csha_with_saves",
        &[
            types::I64, types::I64, types::I64, types::I64, // q, k, v, out
            types::I64, // logsumexp
            types::I64, // scale_bits
            types::I64, types::I64, types::I64, types::I64, // batch, heads, seq_len, head_dim
            types::I64, types::I64, types::I64, types::I64, // block_table, k_pool, v_pool, block_size
            types::I64, types::I64, // cos, sin
            types::I64, types::I64, // seq_ids, seq_lens
            types::I64, // shared_mem_bytes
            types::I64, types::I64, // ptx_ptr, name_ptr
            types::I64, types::I64, // block_q, block_kv
            types::I64, // causal
            // CSHA extras (9):
            types::I64, // x_ptr
            types::I64, // norm_weight_ptr
            types::I64, types::I64, types::I64, types::I64, // Wq, Wk, Wv, Wo
            types::I64, // rmsnorm_eps_bits
            types::I64, // active_heads
            types::I64, // d_model
            // Tier C activation-save pointers (6):
            types::I64, // q_proj_ptr
            types::I64, // k_proj_ptr
            types::I64, // v_proj_ptr
            types::I64, // row_max_ptr
            types::I64, // row_sum_ptr
            types::I64, // x_raw_ptr
            // PCA Tier A: segment_ids device pointer (0 = unpacked path)
            types::I64, // segment_ids_ptr
            // Tier B extension (planner spec §4):
            types::I64, // tier_b_ptx_ptr
            types::I64, // tier_b_name_ptr
            // PCA §4.3: doc_starts device pointer (0 = identity positions)
            types::I64, // doc_starts_ptr
            // PCA per-doc CTA (Strategy 3 v1): num_docs_or_zero — grid_x
            // override when the kernel name carries the `_per_doc_cta`
            // suffix. Pass 0 for all non-per-doc topologies.
            types::I64, // num_docs_or_zero
        ],
        Some(types::I64),
    ),
    // Gap A: codegen-side allocator for the 6 CSHA backward-activation
    // HBM buffers. Writes 6 device-pointer i64s contiguously into
    // `out_ptr` (caller-supplied stack-slot). Returns 0 on success.
    (
        "nsl_csha_alloc_backward_activations_into",
        &[
            types::I64, // batch
            types::I64, // heads
            types::I64, // seq
            types::I64, // head_dim
            types::I64, // out_ptr (writes 6 * i64)
        ],
        Some(types::I64),
    ),
    // Gap A: codegen-side free helper. Takes 6 i64 pointers matching the
    // layout written by `nsl_csha_alloc_backward_activations_into`.
    (
        "nsl_csha_free_backward_activations_from",
        &[
            types::I64, // q_proj
            types::I64, // k_proj
            types::I64, // v_proj
            types::I64, // row_max
            types::I64, // row_sum
            types::I64, // x_raw
        ],
        None,
    ),
    // Gap D / Tier C (extended by Gap I.5 Option A): CSHA fused backward
    // launch. i64 args matching the wengert_lower.rs
    // `PrimalOp::FusedCshaBackward` emission order, plus the trailing
    // tier_b2_active flag (CSHA Tier B.2 Phase 3 T6):
    //   36-arg forward-side prelude mirrored off `_with_saves`,
    //   + 6 forward-saved activation pointers,
    //   + dO input + 8 gradient outputs
    //     (dq, dk, dv, dwq, dwk, dwv, dx, dx_norm).
    // First surfaced as "undefined function" in the Gap I.3 smoke once
    // A+F let the backward launch actually fire. Gap I.5 appended the
    // 8th output (`dx_norm`) so the AD-side `RmsNormGammaBackward` gets
    // the correct `dy_norm` input.
    (
        "nsl_flash_attention_csha_backward",
        &[
            types::I64, types::I64, types::I64, // q, k, v
            types::I64, types::I64,             // out, logsumexp
            types::I64,                         // scale_bits
            types::I64, types::I64, types::I64, types::I64, // batch, heads, seq_len, head_dim
            types::I64, types::I64, types::I64, types::I64, // block_table, k_pool, v_pool, block_size
            types::I64, types::I64,             // cos, sin
            types::I64, types::I64,             // seq_ids, seq_lens
            types::I64,                         // shmem_bytes
            types::I64, types::I64,             // bwd_ptx_ptr, bwd_name_ptr
            types::I64, types::I64,             // block_q, block_kv
            types::I64,                         // causal
            types::I64, types::I64,             // x_ptr, norm_weight_ptr
            types::I64, types::I64, types::I64, // wq, wk, wv
            types::I64,                         // wo (null)
            types::I64,                         // rmsnorm_eps_bits
            types::I64, types::I64,             // active_heads, d_model
            // Saved activations (6):
            types::I64, types::I64, types::I64, // q_proj, k_proj, v_proj
            types::I64, types::I64,             // row_max, row_sum
            types::I64,                         // x_raw
            // Gradient outputs (dO + 8):
            types::I64,                         // do_ptr
            types::I64, types::I64, types::I64, // dq, dk, dv
            types::I64, types::I64, types::I64, // dwq, dwk, dwv
            types::I64,                         // dx
            types::I64,                         // dx_norm (Gap I.5)
            types::I64,                         // segment_ids_ptr (PCA Tier A Task 4B)
            types::I64,                         // tier_b_ptx_ptr (planner spec §4)
            types::I64,                         // tier_b_name_ptr (planner spec §4)
            types::I64,                         // doc_starts_ptr (PCA §4.3 Task 3)
            types::I64,                         // tier_b2_active (CSHA Tier B.2 Phase 3 T6)
            types::I64,                         // num_docs_or_zero (PCA per-doc CTA backward, Sprint 5)
        ],
        Some(types::I64),
    ),
    // FlashAttention-2 backward (M27 backward pass)
    // Returns NslList [dQ, dK, dV]. When logsumexp_ptr == 0, auto-computes lse.
    //
    // The trailing Tier-B sentinel pair (planner spec §4) was added to the
    // runtime FFI but this declaration and the wengert_lower call site were
    // never extended — Cranelift emitted 16-arg calls against the 18-param C
    // function, so the runtime read undefined stack/registers for the pair
    // and `assert_tier_b_sentinels` aborted the process at the FIRST plain-
    // SDPA training backward (found by the roadmap-4.2 pretrain e2e; no prior
    // test ran the compiler-EMITTED call — GPU parity tests build FFI args by
    // hand). Keep this in lock-step with `nsl_flash_attention_backward` in
    // nsl-runtime/src/flash_attention.rs.
    (
        "nsl_flash_attention_backward",
        &[
            types::I64, // dout
            types::I64, types::I64, types::I64, // q, k, v
            types::I64, // out (forward output)
            types::I64, // logsumexp (0 = auto-compute)
            types::I64, // scale_bits (f32 as i64)
            types::I64, types::I64, types::I64, types::I64, // batch, heads, seq_len, head_dim
            types::I64, // causal
            types::I64, // phase1_ptx_ptr (0 = CPU fallback)
            types::I64, // phase1_name_ptr
            types::I64, // phase2_ptx_ptr
            types::I64, // phase2_name_ptr
            types::I64, // tier_b_ptx_ptr (planner spec §4 sentinel pair;
            types::I64, // tier_b_name_ptr  both 0 = no Tier-B-on variant)
            types::I64, // segment_ids (PCA Stage C: NslTensor* [b,s], 0 = plain)
        ],
        Some(types::I64),
    ),
    // PCA Stage C: plain fused SDPA forward with saves. Launches the v2
    // scalar forward (csha: None) selected by the wengert lowering's
    // per-head_dim variant table; returns an NslList* [out, lse] or 0 to
    // DECLINE (caller's decomposed fallback runs). segment_ids != 0 selects
    // the segment-masked kernel family (packed attention); the Tier-B pair
    // is the tile-skip variant behind the runtime gate. Keep in lock-step
    // with `nsl_sdpa_fused_forward` in nsl-runtime/src/flash_attention.rs.
    (
        "nsl_sdpa_fused_forward",
        &[
            types::I64, types::I64, types::I64, // q, k, v (NslTensor*)
            types::I64, // scale_bits (f32 as i64)
            types::I64, // causal
            types::I64, // segment_ids (NslTensor* [b,s] or 0)
            types::I64, // ptx_ptr (base kernel; 0 = decline)
            types::I64, // name_ptr
            types::I64, // tier_b_ptx_ptr (sentinel pair, both 0 = none)
            types::I64, // tier_b_name_ptr
            types::I64, // block_q
            types::I64, // block_kv
            types::I64, // shared_mem_bytes
        ],
        Some(types::I64),
    ),
    // PCA Stage C: align packed-batch mask/segment tensors to the params'
    // device at step start (train-block batch prep). No-op for CPU models
    // and unpacked batches. Keep in lock-step with
    // `nsl_packed_batch_align_device` in nsl-runtime/src/packing.rs.
    (
        "nsl_packed_batch_align_device",
        &[
            types::I64, // batch dict (NslDict*)
            types::I64, // param list (NslList*) — device reference
        ],
        Some(types::I64),
    ),
    // Campaign item 5: derive the dense [b,1,s,s] packed mask from
    // segment_ids at the decomposed-fallback site (the DataLoader no
    // longer ships attention_mask by default). Keep in lock-step with
    // `nsl_packed_mask_from_segment_ids` in nsl-runtime/src/packing.rs.
    (
        "nsl_packed_mask_from_segment_ids",
        &[
            types::I64, // segment_ids (NslTensor* [b,s], f32/f64, CPU or GPU)
        ],
        Some(types::I64),
    ),
    // --- M46: Deterministic kernel variants ---
    (
        "nsl_tensor_reduce_sum_deterministic",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_reduce_mean_deterministic",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_scatter_add_deterministic",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // --- M46: Global deterministic mode flag + RNG seeding ---
    ("nsl_set_deterministic", &[types::I64], Some(types::I64)),
    ("nsl_rng_seed", &[types::I64], Some(types::I64)),
    // Item 4 (2026-09-02): the matmul arithmetic configuration, promoted from
    // the NSL_MATMUL_BF16* environment family. Installed before user code, from
    // the same compile options the execution fingerprint renders -- so the
    // runtime and the fingerprint cannot disagree about which arithmetic ran.
    // (mode, rounding, min_ratio, cast_cache, lt, lt_workspace_mib, lt_tune)
    (
        "nsl_set_matmul_config",
        &[
            types::I64,
            types::I64,
            types::F64,
            types::I64,
            types::I64,
            types::I64,
            types::I64,
        ],
        Some(types::I64),
    ),
    // --- M50: Sparse tensors ---
    (
        "nsl_sparse_coo",
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
        "nsl_sparse_from_dense",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ), // M50b: +threshold
    ("nsl_sparse_to_dense", &[types::I64], Some(types::I64)),
    ("nsl_sparse_nnz", &[types::I64], Some(types::I64)),
    ("nsl_sparse_density", &[types::I64], Some(types::I64)),
    (
        "nsl_sparse_spmm",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_sparse_spmv",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_sparse_coo_to_csr", &[types::I64], Some(types::I64)),
    ("nsl_sparse_coo_to_csc", &[types::I64], Some(types::I64)),
    ("nsl_sparse_csr_to_csc", &[types::I64], Some(types::I64)),
    ("nsl_sparse_csc_to_csr", &[types::I64], Some(types::I64)),
    ("nsl_sparse_csr_to_coo", &[types::I64], Some(types::I64)),
    ("nsl_sparse_csc_to_coo", &[types::I64], Some(types::I64)),
    (
        "nsl_sparse_add",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_sparse_mul",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_sparse_free", &[types::I64], Some(types::I64)),
    // --- M40: Source AD runtime helpers ---
    (
        "nsl_tensor_compare",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_where",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_scalar",
        &[types::F64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_pad_zero",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_scatter_add",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_embedding_backward",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_cross_entropy_backward",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_mse_backward",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_l1_backward",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // p4 slice 2: fused SiLU backward — grad * σ(x)*(1 + x*(1-σ(x))).
    (
        "nsl_tensor_silu_backward",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // P5 item 20 slice B: fused SwiGLU gate backward (y_bar, up, gate_in) -> dgate.
    (
        "nsl_tensor_swiglu_gate_backward",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // p4 slice 3: fused Sigmoid backward — grad * y*(1-y), y = σ output.
    (
        "nsl_tensor_sigmoid_backward",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // p4 slice 3: fused Tanh backward — grad * (1 - y*y), y = tanh output.
    (
        "nsl_tensor_tanh_backward",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // p4 slice 4: source-AD in-place suppression guard (on!=0 enter / on==0 leave)
    // — raised around the source-AD forward so FBIP preserves primal inputs.
    ("nsl_set_inplace_suppressed", &[types::I64], None),
    // p4 GELU fix: fused GELU backward — grad * gelu'(x), per-device derivative.
    (
        "nsl_tensor_gelu_backward",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // LSE tape-carry gates: fused-SDPA launch counter (0 = base fwd kernel,
    // 1 = Tier-B tile-skip) — NSL-callable as sdpa_fused_launch_count(v).
    ("nsl_sdpa_fused_launch_count", &[types::I64], Some(types::I64)),
];
