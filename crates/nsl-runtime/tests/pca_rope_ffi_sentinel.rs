//! PCA §4.3 RoPE Position Reset — Task 3 FFI sentinel test.
//!
//! Verifies the three CSHA-fused FFI entry points have been extended with
//! the trailing `doc_starts_ptr: i64` parameter (per spec v3 §3). The
//! type-level coercion below fails to compile if any of the function
//! pointer signatures does not accept the documented argument count.
//!
//! Sentinel value `0` preserves identity-position semantics — the runtime
//! body must treat `doc_starts_ptr == 0` as "RoPE position reset disabled"
//! and route through the existing unpacked path.
//!
//! Test order (PCA-FFI-1, PCA-FFI-2, PCA-FFI-3) mirrors the spec's three
//! entry-point enumeration so regression failures are easy to localise.

use nsl_runtime::flash_attention::{
    nsl_flash_attention_csha, nsl_flash_attention_csha_backward,
    nsl_flash_attention_csha_with_saves,
};

/// PCA-FFI-1: `nsl_flash_attention_csha` accepts a trailing `doc_starts_ptr: i64`.
///
/// Count: 24 base + 9 CSHA extras + 1 segment_ids_ptr + 1 doc_starts_ptr = 35 params.
#[test]
fn csha_forward_ffi_accepts_trailing_doc_starts_ptr() {
    let _: extern "C" fn(
        i64, i64, i64, i64, i64, i64,   // q, k, v, out, lse, scale_bits
        i64, i64, i64, i64,             // batch, heads, seq_len, head_dim
        i64, i64, i64, i64,             // block_table, k_pool, v_pool, block_size
        i64, i64,                       // cos, sin
        i64, i64,                       // seq_ids, seq_lens
        i64, i64, i64,                  // shmem_bytes, ptx, name
        i64, i64, i64,                  // block_q, block_kv, causal
        // CSHA extras (9):
        i64, i64, i64, i64, i64, i64,   // x, norm_weight, Wq, Wk, Wv, Wo
        i64, i64, i64,                  // rmsnorm_eps_bits, active_heads, d_model
        // PCA Tier A: segment_ids_ptr
        i64,
        // PCA Tier B Planner (PR #175): tier_b_ptx_ptr, tier_b_name_ptr.
        i64, i64,
        // PCA §4.3: doc_starts_ptr — the new trailing param under test.
        i64,
        // PCA per-doc CTA (Strategy 3 v1): num_docs_or_zero — CFTP v2 follow-on.
        i64,
    ) -> i64 = nsl_flash_attention_csha;
}

/// PCA-FFI-2: `nsl_flash_attention_csha_with_saves` accepts a trailing
/// `doc_starts_ptr: i64`.
#[test]
fn csha_with_saves_ffi_accepts_trailing_doc_starts_ptr() {
    let _: extern "C" fn(
        i64, i64, i64, i64, i64, i64,   // q, k, v, out, lse, scale_bits
        i64, i64, i64, i64,             // batch, heads, seq_len, head_dim
        i64, i64, i64, i64,             // block_table, k_pool, v_pool, block_size
        i64, i64,                       // cos, sin
        i64, i64,                       // seq_ids, seq_lens
        i64, i64, i64,                  // shmem_bytes, ptx, name
        i64, i64, i64,                  // block_q, block_kv, causal
        // CSHA extras (9):
        i64, i64, i64, i64, i64, i64,   // x, norm_weight, Wq, Wk, Wv, Wo
        i64, i64, i64,                  // rmsnorm_eps_bits, active_heads, d_model
        // Tier C save pointers (6):
        i64, i64, i64, i64, i64, i64,
        // PCA Tier A: segment_ids_ptr
        i64,
        // PCA Tier B Planner (PR #175): tier_b_ptx_ptr, tier_b_name_ptr.
        i64, i64,
        // PCA §4.3: doc_starts_ptr — the new trailing param under test.
        i64,
        // PCA per-doc CTA (Strategy 3 v1): num_docs_or_zero — CFTP v2 follow-on.
        i64,
    ) -> i64 = nsl_flash_attention_csha_with_saves;
}

/// PCA-FFI-3: `nsl_flash_attention_csha_backward` accepts a trailing
/// `doc_starts_ptr: i64`.
#[test]
fn csha_backward_ffi_accepts_trailing_doc_starts_ptr() {
    let _: extern "C" fn(
        i64, i64, i64, i64, i64, i64,   // q, k, v, out, lse, scale_bits
        i64, i64, i64, i64,             // batch, heads, seq_len, head_dim
        i64, i64, i64, i64,             // block_table, k_pool, v_pool, block_size
        i64, i64,                       // cos, sin
        i64, i64,                       // seq_ids, seq_lens
        i64, i64, i64,                  // shmem_bytes, ptx, name
        i64, i64, i64,                  // block_q, block_kv, causal
        // CSHA extras (9):
        i64, i64, i64, i64, i64, i64,   // x, norm_weight, Wq, Wk, Wv, Wo
        i64, i64, i64,                  // rmsnorm_eps_bits, active_heads, d_model
        // Saved activations (6):
        i64, i64, i64, i64, i64, i64,
        // dO + 8 gradient outputs:
        i64,                            // do_ptr
        i64, i64, i64,                  // dq, dk, dv
        i64, i64, i64,                  // dwq, dwk, dwv
        i64,                            // dx
        i64,                            // dx_norm
        // PCA Tier A: segment_ids_ptr
        i64,
        // PCA Tier B Planner (PR #175): tier_b_ptx_ptr, tier_b_name_ptr.
        i64, i64,
        // PCA §4.3: doc_starts_ptr — the new trailing param under test.
        i64,
        // PCA per-doc CTA (Strategy 3 v1): num_docs_or_zero — CFTP v2 follow-on.
        i64,
    ) -> i64 = nsl_flash_attention_csha_backward;
}

/// PCA-FFI-4 (non-CUDA): sentinel-0 path returns -1 cleanly on a no-CUDA
/// build. The runtime body must accept the new arg without panicking.
#[test]
#[cfg(not(feature = "cuda"))]
fn csha_forward_ffi_sentinel_zero_returns_minus_one_without_panic() {
    let r = nsl_flash_attention_csha(
        0, 0, 0, 0, 0, 1.0f32.to_bits() as i64,
        1, 1, 16, 64,
        0, 0, 0, 0,
        0, 0, 0, 0,
        0, 0, 0,
        64, 64, 0,
        // CSHA extras (all null / zero).
        0, 0, 0, 0, 0, 0,
        1e-5f32.to_bits() as i64,
        0, 0,
        // PCA Tier A: segment_ids_ptr (0 = unpacked).
        0,
        // PCA Tier B Planner (PR #175): tier_b_ptx_ptr, tier_b_name_ptr.
        0, 0,
        // PCA §4.3: doc_starts_ptr (0 = identity positions).
        0,
        // PCA per-doc CTA (Strategy 3 v1): num_docs_or_zero (0 = classic topology).
        0,
    );
    assert_eq!(r, -1, "non-CUDA build must return -1");
}
