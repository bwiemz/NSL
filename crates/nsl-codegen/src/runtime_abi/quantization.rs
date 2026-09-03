//! Quantized formats: INT8, FP8, AWQ, GPTQ and KV compression.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_ABI_QUANTIZATION: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // CPDT §3.2: INT8 blockwise quantization (the headline 4× memory result)
    ("nsl_tensor_quant_int8_blockwise", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_dequant_int8_blockwise", &[types::I64], Some(types::I64)),
    // Quantization (M16)
    (
        "nsl_qtensor_quantize",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_qtensor_dequantize", &[types::I64], Some(types::I64)),
    (
        "nsl_qtensor_matmul_mixed",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_qtensor_free", &[types::I64], None),
    ("nsl_qtensor_addref", &[types::I64], None),
    ("nsl_qtensor_release", &[types::I64], None),
    ("nsl_qtensor_dtype", &[types::I64], Some(types::I64)),
    ("nsl_qtensor_shape", &[types::I64], Some(types::I64)),
    // Custom dtype registry (M23)
    (
        "nsl_register_custom_dtype",
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
        None,
    ),
    ("nsl_finalize_dtype_registry", &[], None),
    (
        "nsl_tensor_to_custom_dtype",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_from_custom_dtype",
        &[types::I64],
        Some(types::I64),
    ),
    // M42b: Quantized FlashAttention (KV-cache in INT8/FP8)
    (
        "nsl_flash_attention_quantized",
        &[
            types::I64,
            types::I64,
            types::I64,
            types::I64,
            types::I64, // q, k, v, out, scale
            types::I64,
            types::I64,
            types::I64,
            types::I64, // batch, heads, seq_len, head_dim
            types::I64,
            types::I64,
            types::I64,
            types::I64, // block_table, k_pool, v_pool, block_size
            types::I64,
            types::I64,
            types::I64, // meta_k, meta_v, kv_quant_scheme
            types::I64, // shared_mem_bytes
            types::I64,
            types::I64, // ptx_ptr, name_ptr
            types::I64,
            types::I64, // block_q, block_kv
            // PCA Tier B planner spec §4 — the runtime
            // `nsl_flash_attention_quantized` takes the SAME 2 trailing Tier-B
            // sentinel slots as `nsl_flash_attention` and reads them via
            // `assert_tier_b_sentinels`. They were missing here (declared 21 vs
            // the runtime's 23) — a latent ABI drift: no call site emits this
            // today, but any future caller would push 21 args and the runtime
            // would read 2 slots of stack garbage. Caught by the nsl-abi
            // signature-agreement gate.
            types::I64, // tier_b_ptx_ptr  (disabled sentinel 0)
            types::I64, // tier_b_name_ptr (disabled sentinel 0)
        ],
        Some(types::I64),
    ),
    (
        "nsl_rope_cache_write",
        &[
            types::I64,
            types::I64, // k_projected, v_projected
            types::I64,
            types::I64,
            types::I64, // cos, sin, positions
            types::I64,
            types::I64,
            types::I64, // k_pool, v_pool, block_table
            types::I64,
            types::I64, // seq_ids, seq_lens (M29-ready)
            types::I64,
            types::I64,
            types::I64,
            types::I64, // num_tokens, num_heads, head_dim, block_size
            types::I64,
            types::I64, // ptx_ptr, name_ptr
        ],
        Some(types::I64),
    ),
    // --- M34: Context parallelism (ring attention) ---
    // Extern signatures REMOVED across two cycles:
    //   * CPDT Part III v2.22 unlinked the ring FFI chain from codegen
    //     (`nsl_cp_init` / `nsl_sequence_partition` / `nsl_ring_attention`
    //     / `nsl_ring_send_recv` / `nsl_sequence_gather` / `nsl_cp_destroy`)
    //     and fell `@context_parallel` through to naive attention.
    //   * M34 v1 (this cycle) deleted the six runtime stubs themselves from
    //     `crates/nsl-runtime/src/context_parallel/ffi.rs` (they were dead
    //     symbols with wrong positional layouts) and shipped the
    //     single-node ring-attention composer
    //     (`run_ring_attention_full`) verified against `naive_attention`
    //     on a matrix of shapes and ring sizes. Multi-device distribution
    //     is deferred until NCCL/IPC lands.
    // When multi-device distribution lands, a fresh runtime FFI shape gets
    // designed and the extern table + emission + runtime impl all get
    // wired together against the new shape.
    // --- M35: FP8 compute ---
    (
        "nsl_fp8_cast",
        &[types::I64, types::I64, types::F64],
        Some(types::I64),
    ),
    (
        "nsl_fp8_matmul",
        &[types::I64, types::I64, types::F64, types::F64],
        Some(types::I64),
    ),
    (
        "nsl_fp8_matmul_training",
        &[types::I64, types::I64, types::I8],
        Some(types::I64),
    ),
    (
        "nsl_fp8_compute_scale",
        &[types::I64, types::I64],
        Some(types::F64),
    ),
    (
        "nsl_fp8_quantize_e5m2",
        &[types::I64, types::F64],
        Some(types::I64),
    ),
    ("nsl_fp8_gradient_scale", &[types::I64], Some(types::F64)),
    ("nsl_fp8_cache_e5m2_ptx", &[types::I64, types::I64], None),
    (
        "nsl_fp8_update_calibration",
        &[types::I64, types::I64, types::F64],
        Some(types::F64),
    ),
    // --- M35: AWQ 4-bit quantization ---
    (
        "nsl_awq_quantize",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_awq_matmul",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_awq_free", &[types::I64], None),
    // AWQ calibration sidecar: apply per-channel scales to weight tensor before quantizing.
    // Signature: (weight_ptr, scales_ptr, scales_len, alpha) -> scaled_weight_ptr
    (
        "nsl_awq_pre_scale_weight",
        &[types::I64, types::I64, types::I64, types::F64],
        Some(types::I64),
    ),
    // --- M35: GPTQ quantization ---
    (
        "nsl_gptq_quantize",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_gptq_quantize_ext",
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
        "nsl_gptq_matmul",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_gptq_free", &[types::I64], None),
    ("nsl_gptq_hessian_init", &[types::I64], Some(types::I64)),
    (
        "nsl_gptq_hessian_add_batch",
        &[types::I64],
        Some(types::I64),
    ),
    ("nsl_gptq_hessian_finalize", &[], Some(types::I64)),
    // --- M42: KV-cache compression ---
    (
        "nsl_kv_quantize_and_store",
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
    (
        "nsl_kv_sliding_window_init",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_sliding_window_check",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_kv_sliding_window_destroy", &[], Some(types::I64)),
    (
        "nsl_kv_h2o_init",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_h2o_accumulate",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_h2o_check",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_kv_h2o_remove_sequence",
        &[types::I64],
        Some(types::I64),
    ),
    ("nsl_kv_h2o_destroy", &[], Some(types::I64)),
    ("nsl_kv_compress_ratio", &[types::I64], Some(types::I64)),
];
