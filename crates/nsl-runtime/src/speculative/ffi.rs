//! M33: Speculative decoding FFI — rejection sampling on pre-computed logits.

use super::verify;
use crate::memory::checked_alloc;
use crate::tensor::NslTensor;
use std::os::raw::c_void;
use std::slice;

/// Draft K tokens autoregressively.
/// Stub — actual draft model forward pass requires codegen-generated function calls.
/// The codegen calls the draft model's forward method directly and passes the
/// resulting logits to nsl_speculative_decode_step.
#[no_mangle]
pub extern "C" fn nsl_speculative_draft(
    _draft_model_ptr: i64,
    _input_ids_ptr: i64,
    _seq_len: i64,
    _num_tokens: i64,
    _draft_tokens_ptr: i64,
    _draft_logits_ptr: i64,
    _temperature_bits: i64,
) -> i64 {
    0
}

/// Verify draft tokens via rejection sampling on pre-computed logits.
///
/// Takes draft tokens, draft logits, and verifier logits (all pre-computed by codegen
/// calling the respective model forward methods). Runs rejection sampling and writes
/// the result to result_ptr.
///
/// draft_tokens_ptr:    *const i64, [K] draft token IDs
/// draft_logits_ptr:    *const f32, [K, vocab_size] draft model logits
/// verifier_logits_ptr: first 8 bytes of result_ptr used for verifier logits pointer
/// num_draft_tokens:    K
/// temperature_bits:    f32::to_bits() as i64
/// result_ptr:          *mut i64 — output buffer:
///                        [0] = num_accepted
///                        [1] = has_bonus (0 or 1)
///                        [2..2+num_accepted+has_bonus] = accepted token IDs
#[no_mangle]
pub extern "C" fn nsl_speculative_verify(
    _verifier_model_ptr: i64,
    _input_ids_ptr: i64,
    _seq_len: i64,
    _draft_tokens_ptr: i64,
    _draft_logits_ptr: i64,
    _num_draft_tokens: i64,
    _temperature_bits: i64,
    _result_ptr: i64,
) -> i64 {
    // This stub path: codegen should use nsl_speculative_decode_step instead
    0
}

/// High-level speculative decode step.
///
/// Takes pre-computed draft and verifier logits, runs rejection sampling,
/// returns a new NslTensor containing the accepted token IDs.
///
/// This is the primary entry point for speculative decoding from codegen.
/// The codegen is responsible for:
///   1. Running the draft model K times to get draft_tokens + draft_logits
///   2. Running the verifier model once on [context + draft_tokens] to get verifier_logits
///   3. Calling this function with both sets of logits
///
/// Parameters:
///   draft_tokens_ptr:     NslTensor* [K] i64 draft token IDs
///   draft_logits_ptr:     NslTensor* [K, vocab_size] f32 draft logits
///   verifier_logits_ptr:  NslTensor* [K+1, vocab_size] f32 verifier logits
///   num_draft_tokens:     K (number of drafted tokens)
///   vocab_size:           vocabulary size
///   temperature_bits:     f32::to_bits() as i64
///
/// Returns: NslTensor* [num_accepted] i64 — accepted token IDs
#[no_mangle]
pub extern "C" fn nsl_speculative_decode_step(
    draft_tokens_ptr: i64,
    draft_logits_ptr: i64,
    verifier_logits_ptr: i64,
    num_draft_tokens: i64,
    vocab_size: i64,
    temperature_bits: i64,
) -> i64 {
    let k = num_draft_tokens as usize;
    let vocab = vocab_size as usize;
    let temperature = f32::from_bits(temperature_bits as u32);

    // Read draft tokens — these are token IDs stored as f64 or f32 values
    let draft_tensor = NslTensor::from_ptr(draft_tokens_ptr);
    let draft_tokens: Vec<i64> = if draft_tensor.dtype == 0 {
        let data = unsafe { slice::from_raw_parts(draft_tensor.data as *const f64, k) };
        data.iter().map(|&v| v as i64).collect()
    } else {
        // f32 tensor — read as f32 and convert to i64
        let data = unsafe { slice::from_raw_parts(draft_tensor.data as *const f32, k) };
        data.iter().map(|&v| v as i64).collect()
    };

    // Read draft logits [K, vocab_size] as f32
    let draft_logits_tensor = NslTensor::from_ptr(draft_logits_ptr);
    let draft_logits: Vec<f32> = if draft_logits_tensor.dtype == 0 {
        let data = unsafe { slice::from_raw_parts(draft_logits_tensor.data as *const f64, k * vocab) };
        data.iter().map(|&v| v as f32).collect()
    } else {
        let data = unsafe { slice::from_raw_parts(draft_logits_tensor.data as *const f32, k * vocab) };
        data.to_vec()
    };

    // Read verifier logits [K+1, vocab_size] as f32
    let verifier_tensor = NslTensor::from_ptr(verifier_logits_ptr);
    let verifier_logits: Vec<f32> = if verifier_tensor.dtype == 0 {
        let data = unsafe { slice::from_raw_parts(verifier_tensor.data as *const f64, (k + 1) * vocab) };
        data.iter().map(|&v| v as f32).collect()
    } else {
        let data = unsafe { slice::from_raw_parts(verifier_tensor.data as *const f32, (k + 1) * vocab) };
        data.to_vec()
    };

    let seed = draft_tokens.first().copied().unwrap_or(42) as u64;

    // Run rejection sampling
    let result = verify::rejection_sample(
        &verifier_logits,
        &draft_logits,
        &draft_tokens,
        vocab,
        temperature,
        seed,
    );

    // Create output tensor: [num_accepted_tokens] of f64 (token IDs)
    let num_tokens = result.accepted_tokens.len();
    let data = checked_alloc(num_tokens * std::mem::size_of::<f64>()) as *mut f64;
    for (i, &token) in result.accepted_tokens.iter().enumerate() {
        unsafe { *data.add(i) = token as f64 };
    }

    let shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *shape = num_tokens as i64 };

    let strides = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *strides = 1 };

    let tensor = Box::new(NslTensor {
        data: data as *mut c_void,
        shape,
        strides,
        ndim: 1,
        len: num_tokens as i64,
        refcount: 1,
        device: 0,
        dtype: 0,
        owns_data: 1,
    });

    Box::into_raw(tensor) as i64
}

/// Build speculation tree from draft logits.
#[no_mangle]
pub extern "C" fn nsl_speculative_build_tree(
    _draft_logits_ptr: i64,
    _num_heads: i64,
    _tree_width: i64,
    _vocab_size: i64,
    _tree_ptr: i64,
) -> i64 {
    0
}

/// Verify speculation tree.
#[no_mangle]
pub extern "C" fn nsl_speculative_verify_tree(
    _verifier_model_ptr: i64,
    _input_ids_ptr: i64,
    _seq_len: i64,
    _tree_ptr: i64,
    _temperature_bits: i64,
    _result_ptr: i64,
) -> i64 {
    0
}

/// CoW branch a sequence's page table.
#[no_mangle]
pub extern "C" fn nsl_page_branch(_page_table_handle: i64, _parent_seq: i64) -> i64 {
    -1
}

/// Copy-on-write for a specific page.
#[no_mangle]
pub extern "C" fn nsl_page_cow_copy(
    _page_table_handle: i64,
    _seq_id: i64,
    _logical_block_idx: i64,
) -> i64 {
    0
}

/// FlashAttention with tree-structured causal mask.
#[no_mangle]
pub extern "C" fn nsl_tree_attention(
    _q_ptr: i64,
    _k_ptr: i64,
    _v_ptr: i64,
    _out_ptr: i64,
    _scale_bits: i64,
    _batch: i64,
    _heads: i64,
    _seq_len: i64,
    _head_dim: i64,
    _block_table_ptr: i64,
    _k_pool_ptr: i64,
    _v_pool_ptr: i64,
    _block_size: i64,
    _tree_parent_ptr: i64,
    _dfs_enter_ptr: i64,
    _dfs_exit_ptr: i64,
    _num_tree_nodes: i64,
    _shared_mem_bytes: i64,
    _ptx_ptr: i64,
    _name_ptr: i64,
    _block_q: i64,
    _block_kv: i64,
) -> i64 {
    0
}

/// Clean up speculative branch.
#[no_mangle]
pub extern "C" fn nsl_speculative_cleanup(_page_table_handle: i64, _seq_id: i64) -> i64 {
    0
}
