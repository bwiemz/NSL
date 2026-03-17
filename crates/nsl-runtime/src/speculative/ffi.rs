//! M33: Speculative decoding FFI stubs.

/// Draft K tokens autoregressively.
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

/// Verify draft tokens via rejection sampling.
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
    0
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
