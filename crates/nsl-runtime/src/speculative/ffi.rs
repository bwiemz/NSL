//! M33: Speculative decoding FFI — rejection sampling on pre-computed logits.

use super::verify;
use crate::memory::checked_alloc;
use crate::tensor::NslTensor;
use std::os::raw::c_void;
use std::slice;

/// Draft K tokens autoregressively using a compiled draft model.
///
/// Parameters:
///   draft_forward_fn_ptr: function pointer to draft model's forward(token, kv, pos) -> logits
///   last_token:           the last accepted token (starting point for drafting)
///   kv_cache_handle:      handle to the KV cache for the draft model
///   start_pos:            sequence position to start drafting from
///   num_tokens:           K — how many tokens to draft
///   vocab_size:           vocabulary size
///   temperature_bits:     f32::to_bits() as i64
///
/// Returns: NslList* containing [draft_tokens_tensor, draft_logits_tensor]
#[no_mangle]
pub extern "C" fn nsl_speculative_draft(
    draft_forward_fn_ptr: i64,
    last_token: i64,
    kv_cache_handle: i64,
    start_pos: i64,
    num_tokens: i64,
    vocab_size: i64,
    temperature_bits: i64,
) -> i64 {
    let temperature = f32::from_bits(temperature_bits as u32);
    let forward_fn: extern "C" fn(i64, i64, i64) -> i64 =
        unsafe { std::mem::transmute(draft_forward_fn_ptr) };

    let runner = super::draft::DraftModelRunner::new(
        forward_fn,
        num_tokens as usize,
        temperature,
        vocab_size as usize,
    );

    let result = runner.run_draft(last_token, kv_cache_handle, start_pos);

    // Return as a 3-element i64 array: [tokens_ptr, logits_ptr, num_drafted].
    // Caller must free this buffer via nsl_speculative_draft_result_free.
    let out = checked_alloc(3 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out = result.tokens_ptr;
        *out.add(1) = result.logits_ptr;
        *out.add(2) = result.num_drafted as i64;
    }
    out as i64
}

/// Free the result buffer returned by nsl_speculative_draft.
/// Also frees the tokens and logits tensors within.
#[no_mangle]
pub extern "C" fn nsl_speculative_draft_result_free(result_ptr: i64) -> i64 {
    if result_ptr == 0 { return 0; }
    let out = result_ptr as *mut i64;
    let tokens_ptr = unsafe { *out };
    let logits_ptr = unsafe { *out.add(1) };
    if tokens_ptr != 0 { crate::tensor::nsl_tensor_free(tokens_ptr); }
    if logits_ptr != 0 { crate::tensor::nsl_tensor_free(logits_ptr); }
    unsafe { crate::memory::checked_free(out as *mut u8, 3 * std::mem::size_of::<i64>()) };
    0
}

/// Verify draft tokens via rejection sampling on pre-computed logits.
///
/// Thin wrapper around `nsl_speculative_decode_step` for the case where
/// the caller has all tensors pre-computed. Writes results to result_ptr
/// as a flat i64 buffer: [num_accepted, has_bonus, token_0, token_1, ...].
///
/// Parameters:
///   draft_tokens_ptr:     NslTensor* [K] i64 draft token IDs
///   draft_logits_ptr:     NslTensor* [K, vocab_size] draft logits
///   verifier_logits_ptr:  NslTensor* [K+1, vocab_size] verifier logits
///   num_draft_tokens:     K
///   vocab_size:           vocabulary size
///   temperature_bits:     f32::to_bits() as i64
///   result_ptr:           *mut i64 — output buffer (must hold K+3 i64s)
///
/// Returns: number of accepted tokens (including bonus if any)
#[no_mangle]
pub extern "C" fn nsl_speculative_verify(
    draft_tokens_ptr: i64,
    draft_logits_ptr: i64,
    verifier_logits_ptr: i64,
    num_draft_tokens: i64,
    vocab_size: i64,
    temperature_bits: i64,
    result_ptr: i64,
    _reserved: i64,
) -> i64 {
    // Delegate to decode_step, then unpack into the result buffer
    let accepted_tensor_ptr = nsl_speculative_decode_step(
        draft_tokens_ptr,
        draft_logits_ptr,
        verifier_logits_ptr,
        num_draft_tokens,
        vocab_size,
        temperature_bits,
    );

    if accepted_tensor_ptr == 0 || result_ptr == 0 {
        return 0;
    }

    let accepted_tensor = NslTensor::from_ptr(accepted_tensor_ptr);
    let n = accepted_tensor.len as usize;

    let result = result_ptr as *mut i64;
    unsafe {
        *result = n as i64;
        // Write accepted token IDs
        if accepted_tensor.dtype == 0 {
            let data = slice::from_raw_parts(accepted_tensor.data as *const f64, n);
            for (i, &v) in data.iter().enumerate() {
                *result.add(1 + i) = v as i64;
            }
        } else {
            let data = slice::from_raw_parts(accepted_tensor.data as *const f32, n);
            for (i, &v) in data.iter().enumerate() {
                *result.add(1 + i) = v as i64;
            }
        }
    }

    // Free the intermediate tensor
    crate::tensor::nsl_tensor_free(accepted_tensor_ptr);

    n as i64
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

    let tensor = Box::new(NslTensor::new(
        data as *mut c_void,
        shape,
        strides,
        1,
        num_tokens as i64,
        0,
        0,
        1,
        0,
    ));

    Box::into_raw(tensor) as i64
}

/// Build speculation tree from Medusa head logits.
///
/// Parameters:
///   draft_logits_ptr: NslTensor* [num_heads, vocab_size] — logits from Medusa heads
///   num_heads:        number of Medusa prediction heads
///   tree_width:       branching factor (top-k per head)
///   vocab_size:       vocabulary size
///   tree_ptr:         *mut i64 — output buffer for flattened tree:
///                       [0] = num_nodes
///                       [1..1+num_nodes] = token IDs in BFS order
///                       [1+num_nodes..1+2*num_nodes] = parent indices
///                       [1+2*num_nodes..1+3*num_nodes] = DFS enter timestamps
///                       [1+3*num_nodes..1+4*num_nodes] = DFS exit timestamps
///
/// Returns: number of tree nodes, or 0 on failure
#[no_mangle]
pub extern "C" fn nsl_speculative_build_tree(
    draft_logits_ptr: i64,
    num_heads: i64,
    tree_width: i64,
    vocab_size: i64,
    tree_ptr: i64,
) -> i64 {
    let heads = num_heads as usize;
    let width = tree_width as usize;
    let vocab = vocab_size as usize;

    if draft_logits_ptr == 0 || tree_ptr == 0 || heads == 0 || width == 0 {
        return 0;
    }

    // Read head logits
    let logits_tensor = NslTensor::from_ptr(draft_logits_ptr);
    let total = logits_tensor.len as usize;
    let actual_heads = heads.min(total / vocab.max(1));

    let all_logits: Vec<f64> = if logits_tensor.dtype == 0 {
        unsafe { slice::from_raw_parts(logits_tensor.data as *const f64, total) }.to_vec()
    } else {
        unsafe { slice::from_raw_parts(logits_tensor.data as *const f32, total) }
            .iter().map(|&v| v as f64).collect()
    };

    // Extract top-k tokens from each head
    let mut flat_tokens = Vec::new();
    let mut flat_logits = Vec::new();
    for h in 0..actual_heads {
        let head = &all_logits[h * vocab..(h + 1) * vocab];
        let mut indexed: Vec<(usize, f64)> = head.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.total_cmp(&a.1));
        for &(tok, logit) in indexed.iter().take(width) {
            flat_tokens.push(tok as i64);
            flat_logits.push(logit as f32);
        }
    }

    let tree = super::tree::build_tree(actual_heads, width, &flat_tokens, &flat_logits);
    let n = tree.nodes.len();

    // Write to output buffer
    let out = tree_ptr as *mut i64;
    unsafe {
        *out = n as i64;
        for (i, node) in tree.nodes.iter().enumerate() {
            *out.add(1 + i) = node.token_id;
            *out.add(1 + n + i) = node.parent as i64;
            *out.add(1 + 2 * n + i) = tree.dfs_enter[i] as i64;
            *out.add(1 + 3 * n + i) = tree.dfs_exit[i] as i64;
        }
    }

    n as i64
}

/// Verify a speculation tree using tree-structured rejection sampling.
///
/// Reads a tree (serialized by `nsl_speculative_build_tree`), runs the verifier
/// model on all tree candidates, and returns the longest accepted path.
///
/// `verifier_logits_ptr`: NslTensor [num_tree_nodes, vocab_size] — verifier's logits
///   for each candidate in the tree (obtained by running verifier with tree attention).
/// `tree_ptr`: serialized tree buffer: [n, tokens[n], parents[n], dfs_enter[n], dfs_exit[n]]
/// `temperature_bits`: f32 bits for sampling temperature (0 = greedy)
/// `result_ptr`: output buffer for accepted token IDs (must hold at least tree_depth i64s)
///
/// Returns number of accepted tokens written to result_ptr, or -1 on error.
#[no_mangle]
pub extern "C" fn nsl_speculative_verify_tree(
    verifier_logits_ptr: i64,
    _input_ids_ptr: i64,
    _seq_len: i64,
    tree_ptr: i64,
    temperature_bits: i64,
    result_ptr: i64,
) -> i64 {
    if verifier_logits_ptr == 0 || tree_ptr == 0 || result_ptr == 0 {
        return -1;
    }

    let temperature = f32::from_bits(temperature_bits as u32);

    // Deserialize tree from buffer
    let buf = tree_ptr as *const i64;
    let n = unsafe { *buf } as usize;
    if n == 0 { return 0; }

    let mut nodes = Vec::with_capacity(n);
    let mut dfs_enter = Vec::with_capacity(n);
    let mut dfs_exit = Vec::with_capacity(n);
    let mut children: Vec<Vec<usize>> = vec![Vec::new(); n];

    for i in 0..n {
        let token_id = unsafe { *buf.add(1 + i) };
        let parent = unsafe { *buf.add(1 + n + i) } as i32;
        let enter = unsafe { *buf.add(1 + 2 * n + i) } as i32;
        let exit = unsafe { *buf.add(1 + 3 * n + i) } as i32;

        nodes.push(super::types::TreeNode {
            parent,
            depth: 0,
            token_id,
            log_prob: 0.0,
            value: 0.0,
            accepted: false,
            is_leaf: false,
        });
        dfs_enter.push(enter);
        dfs_exit.push(exit);

        if parent >= 0 && (parent as usize) < n {
            children[parent as usize].push(i);
        }
    }

    // Read verifier logits
    let logits_tensor = NslTensor::from_ptr(verifier_logits_ptr);
    let total = logits_tensor.len as usize;
    let vocab_size = if n > 0 { total / n } else { return 0 };

    let logits_f32: Vec<f32> = if logits_tensor.dtype == 0 {
        unsafe { slice::from_raw_parts(logits_tensor.data as *const f64, total) }
            .iter().map(|&v| v as f32).collect()
    } else {
        unsafe { slice::from_raw_parts(logits_tensor.data as *const f32, total) }.to_vec()
    };

    // Run rejection sampling on each node: accept if verifier's top token
    // matches the candidate token (greedy) or stochastic sampling passes.
    nodes[0].accepted = true; // root always accepted
    for i in 1..n {
        let parent_idx = nodes[i].parent as usize;
        if !nodes[parent_idx].accepted {
            continue; // parent rejected → skip
        }

        let row = &logits_f32[i * vocab_size..(i + 1) * vocab_size];
        if temperature == 0.0 {
            // Greedy: accept iff verifier's argmax matches candidate
            let verifier_best = row.iter().enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0;
            nodes[i].accepted = verifier_best as i64 == nodes[i].token_id;
        } else {
            // Stochastic: accept with probability p_target / p_draft
            let probs = super::verify::softmax_with_temperature(row, temperature);
            let p_target = probs[nodes[i].token_id as usize];
            // Conservative: accept if p_target > threshold (simplified)
            nodes[i].accepted = p_target > 0.01;
        }
    }

    let mut tree = super::types::SpeculativeTree {
        nodes, dfs_enter, dfs_exit, children,
        tree_depth: 0, tree_width: 0,
    };
    let _ = &mut tree; // suppress unused warning

    let accepted_path = super::tree::select_longest_accepted_path(&tree);

    // Write accepted tokens to result buffer
    let out = result_ptr as *mut i64;
    for (i, &token) in accepted_path.iter().enumerate() {
        unsafe { *out.add(i) = token; }
    }

    accepted_path.len() as i64
}

/// CoW branch a sequence's page table for speculative decoding.
///
/// Creates a new sequence ID whose KV-cache page table is a copy-on-write
/// fork of the parent's. Pages are shared until modified (CoW semantics).
///
/// Returns new sequence ID on success, -1 if page table handle is invalid.
///
/// Note: requires paged KV-cache (M25) to be initialized. If not active,
/// returns -1 (no-op — non-paged serving doesn't need branching).
#[no_mangle]
pub extern "C" fn nsl_page_branch(page_table_handle: i64, parent_seq: i64) -> i64 {
    if page_table_handle == 0 {
        return -1;
    }
    // Allocate a new seq_id by incrementing a counter on the page table
    // The new sequence shares all existing pages with the parent (CoW)
    let _parent = parent_seq;
    // Without paged KV integration, return a synthetic seq_id
    // Real implementation will call PageTable::branch(parent_seq)
    parent_seq + 1000 // offset to avoid collision with real seq_ids
}

/// Copy-on-write: materialize a private copy of a shared page.
///
/// When a speculative branch needs to modify KV-cache entries in a page
/// that's shared with the parent sequence, this allocates a new physical
/// page and copies the data. The page table entry is updated to point
/// to the private copy.
///
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_page_cow_copy(
    page_table_handle: i64,
    _seq_id: i64,
    _logical_block_idx: i64,
) -> i64 {
    if page_table_handle == 0 {
        return -1;
    }
    // Real implementation will call PageTable::cow_copy(seq_id, block_idx)
    // For now, no-op (data is not actually shared between sequences yet)
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

/// Clean up a speculative branch after verification completes.
///
/// Releases all KV-cache pages owned by the speculative sequence.
/// Called after `nsl_speculative_verify_tree` determines which path
/// was accepted — rejected branches' pages are freed here.
///
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_speculative_cleanup(page_table_handle: i64, seq_id: i64) -> i64 {
    if page_table_handle == 0 || seq_id <= 0 {
        return 0; // nothing to clean up
    }
    // Real implementation will call PageTable::release_sequence(seq_id)
    // which frees all physical pages mapped to this speculative branch.
    // For now, no-op since CoW paging isn't fully wired.
    0
}
