//! The tensor surface a program writes directly: creation, shape,
//! indexing, elementwise arithmetic, reductions and the common layers.
//!
//! Part of the runtime-function registry. Every table here is named
//! `RUNTIME_FUNCTIONS*` so `nsl-abi`'s signature gate finds it; see
//! `builtins/mod.rs` for how the tables are declared and why the
//! grouping is free to change.

use cranelift_codegen::ir::types;

#[rustfmt::skip]
pub(crate) const RUNTIME_FUNCTIONS_TENSOR: &[(&str, &[types::Type], Option<types::Type>)] = &[
    // Tensor creation
    ("nsl_tensor_zeros", &[types::I64], Some(types::I64)),
    ("nsl_tensor_ones", &[types::I64], Some(types::I64)),
    ("nsl_tensor_rand", &[types::I64], Some(types::I64)),
    ("nsl_tensor_randn", &[types::I64], Some(types::I64)),
    // Tensor element access
    (
        "nsl_tensor_get",
        &[types::I64, types::I64],
        Some(types::F64),
    ),
    (
        "nsl_tensor_set",
        &[types::I64, types::I64, types::F64],
        None,
    ),
    // Tensor shape ops
    ("nsl_tensor_shape", &[types::I64], Some(types::I64)),
    (
        "nsl_tensor_shape_dim",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // M28: Dynamic shape assertions
    (
        "nsl_tensor_assert_dim",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_assert_dim_bound",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tensor_ndim", &[types::I64], Some(types::I64)),
    // PCA Stage C: non-aborting shape probe (0 for out-of-range dims).
    ("nsl_tensor_dim_or_zero", &[types::I64, types::I64], Some(types::I64)),
    ("nsl_tensor_len", &[types::I64], Some(types::I64)),
    ("nsl_tensor_get_dtype", &[types::I64], Some(types::I64)),
    (
        "nsl_tensor_reshape",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_transpose",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // Tensor arithmetic (elementwise)
    (
        "nsl_tensor_add",
        &[types::I64, types::I64, types::I8],
        Some(types::I64),
    ),
    (
        "nsl_tensor_sub",
        &[types::I64, types::I64, types::I8],
        Some(types::I64),
    ),
    (
        "nsl_tensor_mul",
        &[types::I64, types::I64, types::I8],
        Some(types::I64),
    ),
    (
        "nsl_tensor_div",
        &[types::I64, types::I64, types::I8],
        Some(types::I64),
    ),
    ("nsl_tensor_neg", &[types::I64], Some(types::I64)),
    // Tensor scalar ops
    (
        "nsl_tensor_add_scalar",
        &[types::I64, types::F64, types::I8],
        Some(types::I64),
    ),
    (
        "nsl_tensor_mul_scalar",
        &[types::I64, types::F64, types::I8],
        Some(types::I64),
    ),
    // Scalar-immediate siblings (MFU campaign C3): x / s and x - s with the
    // scalar as an argument instead of a broadcast-materialized tensor.
    (
        "nsl_tensor_div_scalar",
        &[types::I64, types::F64, types::I8],
        Some(types::I64),
    ),
    (
        "nsl_tensor_sub_scalar",
        &[types::I64, types::F64, types::I8],
        Some(types::I64),
    ),
    // Dispatching scalar-immediate entry (review fix): f32 -> the dedicated
    // scalar kernel; any other dtype -> the literal decomposed baseline
    // (preserves the mixed-dtype "f32 wins" narrowing AND output dtype).
    // (tensor, f64 scalar, descriptor-v1 opcode) -> result handle.
    (
        "nsl_tensor_scalar_rhs",
        &[types::I64, types::F64, types::I64],
        Some(types::I64),
    ),
    // Tensor matmul
    (
        "nsl_tensor_matmul",
        &[types::I64, types::I64, types::I8],
        Some(types::I64),
    ),
    // Tensor reductions (return scalar tensor ptr, not f64)
    ("nsl_tensor_sum", &[types::I64], Some(types::I64)),
    ("nsl_tensor_mean", &[types::I64], Some(types::I64)),
    // Tensor scalar extraction
    ("nsl_tensor_item", &[types::I64], Some(types::F64)),
    ("nsl_tensor_l2_norm", &[types::I64], Some(types::F64)),
    // Tensor display
    ("nsl_tensor_print", &[types::I64], None),
    // Tensor memory
    ("nsl_tensor_clone", &[types::I64], Some(types::I64)),
    ("nsl_tensor_clone_if_valid", &[types::I64], Some(types::I64)),
    ("nsl_tensor_free", &[types::I64], None),
    ("nsl_tensor_free_if_valid", &[types::I64], None),
    ("nsl_tensor_free_transient", &[types::I64], None),
    ("nsl_tensor_retain", &[types::I64], None),
    ("nsl_tensor_release", &[types::I64], None),
    ("nsl_tensor_scope_begin", &[], None),
    ("nsl_tensor_scope_end", &[types::I64], None),
    // Element-wise tensor ops (M14)
    ("nsl_tensor_exp", &[types::I64], Some(types::I64)),
    ("nsl_tensor_log", &[types::I64], Some(types::I64)),
    ("nsl_tensor_sqrt", &[types::I64], Some(types::I64)),
    ("nsl_tensor_abs", &[types::I64], Some(types::I64)),
    ("nsl_tensor_sign", &[types::I64], Some(types::I64)),
    (
        "nsl_tensor_clamp",
        &[types::I64, types::F64, types::F64],
        Some(types::I64),
    ),
    // Dimensional reductions (M14)
    (
        "nsl_tensor_sum_dim",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_mean_dim",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_reduce_max",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_gather",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // In-place mutation ops (M14)
    ("nsl_tensor_copy_data", &[types::I64, types::I64], None),
    ("nsl_tensor_add_inplace", &[types::I64, types::I64], None),
    ("nsl_tensor_zero_inplace", &[types::I64], None),
    ("nsl_tensor_zeros_like", &[types::I64], Some(types::I64)),
    // M52b: Create tensor from static .rodata data (compile-time constant folded)
    (
        "nsl_tensor_from_static",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    // Activation functions (M15)
    ("nsl_tensor_relu", &[types::I64], Some(types::I64)),
    ("nsl_tensor_gelu", &[types::I64], Some(types::I64)),
    ("nsl_tensor_silu", &[types::I64], Some(types::I64)),
    ("nsl_tensor_sigmoid", &[types::I64], Some(types::I64)),
    ("nsl_tensor_tanh_act", &[types::I64], Some(types::I64)),
    // Tensor trig (RoPE support)
    ("nsl_tensor_sin", &[types::I64], Some(types::I64)),
    ("nsl_tensor_cos", &[types::I64], Some(types::I64)),
    // Fused rotate_half (RoPE support)
    ("nsl_tensor_rotate_half", &[types::I64], Some(types::I64)),
    // Slice & Cat (M15)
    (
        "nsl_tensor_slice",
        &[types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_cat",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // Embedding lookup (M15)
    (
        "nsl_tensor_embedding_lookup",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // LayerNorm & RMSNorm (M15)
    (
        "nsl_tensor_layernorm",
        &[types::I64, types::I64, types::I64, types::F64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_rmsnorm",
        &[types::I64, types::I64, types::F64],
        Some(types::I64),
    ),
    // Dropout, Conv2d, MaxPool2d (M15)
    (
        "nsl_tensor_dropout",
        &[types::I64, types::F64, types::I8],
        Some(types::I64),
    ),
    // Bias add (M15 — broadcast 1D bias over 2D tensor)
    (
        "nsl_tensor_bias_add",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // Tensor creation helpers (M17)
    (
        "nsl_tensor_zeros_on",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    // CSHA Gap I.3 (A+F): f16 (dtype=2, 2 bytes/element) zeros allocator.
    // The Tier C backward kernel writes dq/dk/dv/dwq/dwk/dwv via
    // `st.global.u16`; the f32 `_zeros_on` variant over-allocates by 2×
    // and leaves every second byte uninitialised → host-side f32 reads
    // then interpret raw f16 bits as f32 → garbage → weight corruption.
    // `dx` stays on `_zeros_on` because the kernel writes it as f32.
    (
        "nsl_tensor_zeros_f16_on",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tensor_ones_like", &[types::I64], Some(types::I64)),
    // Shape manipulation ops (M18a)
    (
        "nsl_tensor_unsqueeze",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_select",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_stack",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_expand",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    ("nsl_tensor_contiguous", &[types::I64], Some(types::I64)),
    ("nsl_tensor_causal_mask", &[types::I64], Some(types::I64)),
    // Sampling primitives (M19)
    ("nsl_manual_seed", &[types::I64], None),
    (
        "nsl_tensor_topk",
        &[types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_multinomial",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_argmax",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_cumsum",
        &[types::I64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_lt_scalar",
        &[types::I64, types::F64],
        Some(types::I64),
    ),
    // Tensor mutation (M19)
    (
        "nsl_tensor_set_element",
        &[types::I64, types::I64, types::I64, types::F64],
        None,
    ),
    (
        "nsl_tensor_slice_assign",
        &[types::I64, types::I64, types::I64, types::I64],
        None,
    ),
    // BatchNorm + AvgPool2d (proper implementations replacing approximations)
    (
        "nsl_tensor_batchnorm",
        &[types::I64, types::I64, types::I64, types::F64, types::I64],
        Some(types::I64),
    ),
    (
        "nsl_tensor_avgpool2d",
        &[types::I64, types::I64, types::I64, types::I64, types::I64],
        Some(types::I64),
    ),
];
