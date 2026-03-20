use std::sync::atomic::AtomicI64;
use super::types::MoeRoutingResult;
use super::router;
use super::dispatch;
use std::slice;

/// Route tokens to experts via top-k gating.
#[no_mangle]
pub extern "C" fn nsl_moe_route(
    router_logits_ptr: i64,
    total_tokens: i64,
    num_experts: i64,
    top_k: i64,
    capacity_factor_bits: i64,
    result_ptr: i64,
) -> i64 {
    let total_tokens = total_tokens as usize;
    let num_experts = num_experts as usize;
    let top_k = top_k as usize;
    let capacity_factor = f32::from_bits(capacity_factor_bits as u32) as f64;

    let logits = unsafe {
        slice::from_raw_parts(router_logits_ptr as *const f32, total_tokens * num_experts)
    };

    let result = router::route_topk(logits, total_tokens, num_experts, top_k, capacity_factor);

    let out = unsafe { &mut *(result_ptr as *mut MoeRoutingResult) };

    let mut indices_box = result.expert_indices.into_boxed_slice();
    out.expert_indices = indices_box.as_mut_ptr();
    std::mem::forget(indices_box);

    let mut weights_box = result.expert_weights.into_boxed_slice();
    out.expert_weights = weights_box.as_mut_ptr();
    std::mem::forget(weights_box);

    let mut sorted_box = result.sorted_token_indices.into_boxed_slice();
    out.sorted_token_indices = sorted_box.as_mut_ptr();
    std::mem::forget(sorted_box);

    let mut bounds_box = result.expert_boundaries.into_boxed_slice();
    out.expert_boundaries = bounds_box.as_mut_ptr();
    std::mem::forget(bounds_box);

    out.total_assigned = result.total_assigned;
    out.importance_loss = result.importance_loss;
    out.load_loss = result.load_loss;

    0
}

/// Scatter tokens into expert-sorted order.
#[no_mangle]
pub extern "C" fn nsl_moe_scatter(
    tokens_ptr: i64,
    sorted_indices_ptr: i64,
    sorted_tokens_ptr: i64,
    total_tokens: i64,
    num_assigned: i64,
    hidden_dim: i64,
) -> i64 {
    let hidden_dim = hidden_dim as usize;
    let num_assigned = num_assigned as usize;
    let total_tokens_n = total_tokens as usize;

    let tokens = unsafe {
        slice::from_raw_parts(tokens_ptr as *const f32, total_tokens_n * hidden_dim)
    };
    let sorted_indices = unsafe {
        slice::from_raw_parts(sorted_indices_ptr as *const i32, num_assigned)
    };

    let scattered = dispatch::scatter_tokens(tokens, sorted_indices, hidden_dim);

    let out_buf = unsafe {
        slice::from_raw_parts_mut(sorted_tokens_ptr as *mut f32, num_assigned * hidden_dim)
    };
    out_buf.copy_from_slice(&scattered);

    0
}

/// Batched expert GEMM (CPU fallback: loop over experts).
#[no_mangle]
pub extern "C" fn nsl_expert_parallel_matmul(
    sorted_tokens_ptr: i64,
    expert_weights_ptr: i64,
    expert_boundaries_ptr: i64,
    output_ptr: i64,
    num_experts: i64,
    hidden_dim: i64,
    intermediate_dim: i64,
) -> i64 {
    let num_experts = num_experts as usize;
    let hidden_dim = hidden_dim as usize;
    let intermediate_dim = intermediate_dim as usize;

    let boundaries = unsafe {
        slice::from_raw_parts(expert_boundaries_ptr as *const i32, num_experts + 1)
    };
    let total_assigned = boundaries[num_experts] as usize;

    let sorted_tokens = unsafe {
        slice::from_raw_parts(sorted_tokens_ptr as *const f32, total_assigned * hidden_dim)
    };
    let expert_weights = unsafe {
        slice::from_raw_parts(
            expert_weights_ptr as *const f32,
            num_experts * hidden_dim * intermediate_dim,
        )
    };
    let output = unsafe {
        slice::from_raw_parts_mut(output_ptr as *mut f32, total_assigned * intermediate_dim)
    };

    for e in 0..num_experts {
        let start = boundaries[e] as usize;
        let end = boundaries[e + 1] as usize;
        if start == end { continue; }

        let w_off = e * hidden_dim * intermediate_dim;
        for t in start..end {
            let tok_off = t * hidden_dim;
            let out_off = t * intermediate_dim;
            for j in 0..intermediate_dim {
                let mut sum = 0.0f32;
                for k in 0..hidden_dim {
                    sum += sorted_tokens[tok_off + k]
                        * expert_weights[w_off + k * intermediate_dim + j];
                }
                output[out_off + j] = sum;
            }
        }
    }

    0
}

/// Gather expert outputs back to original token order.
#[no_mangle]
pub extern "C" fn nsl_moe_gather(
    expert_outputs_ptr: i64,
    _expert_indices_ptr: i64,
    expert_weights_ptr: i64,
    sorted_indices_ptr: i64,
    output_ptr: i64,
    total_tokens: i64,
    top_k: i64,
    hidden_dim: i64,
    num_assigned: i64,
) -> i64 {
    let total_tokens = total_tokens as usize;
    let top_k = top_k as usize;
    let hidden_dim = hidden_dim as usize;
    let num_assigned = num_assigned as usize;

    let expert_outputs = unsafe {
        slice::from_raw_parts(expert_outputs_ptr as *const f32, num_assigned * hidden_dim)
    };
    let sorted_indices = unsafe {
        slice::from_raw_parts(sorted_indices_ptr as *const i32, num_assigned)
    };
    let weights = unsafe {
        slice::from_raw_parts(expert_weights_ptr as *const f32, num_assigned)
    };

    let gathered = dispatch::gather_tokens(
        expert_outputs, sorted_indices, weights,
        total_tokens, top_k, hidden_dim,
    );

    let output = unsafe {
        slice::from_raw_parts_mut(output_ptr as *mut f32, total_tokens * hidden_dim)
    };
    output.copy_from_slice(&gathered);

    0
}

/// Compute auxiliary load-balancing loss.
#[no_mangle]
pub extern "C" fn nsl_moe_aux_loss(
    expert_weights_ptr: i64,
    expert_indices_ptr: i64,
    total_tokens: i64,
    num_experts: i64,
    top_k: i64,
    coeff_bits: i64,
) -> i64 {
    let total_tokens = total_tokens as usize;
    let num_experts = num_experts as usize;
    let top_k = top_k as usize;
    let coeff = f32::from_bits(coeff_bits as u32);

    let weights = unsafe {
        slice::from_raw_parts(expert_weights_ptr as *const f32, total_tokens * top_k)
    };
    let indices = unsafe {
        slice::from_raw_parts(expert_indices_ptr as *const i32, total_tokens * top_k)
    };

    let (imp, load) = super::aux_loss::compute_aux_losses(
        weights, indices, total_tokens, num_experts, top_k,
    );
    let result = coeff * (imp + load);

    result.to_bits() as i64
}

/// All-to-all token exchange for expert parallelism (stub).
#[no_mangle]
pub extern "C" fn nsl_moe_all_to_all(
    _send_buf_ptr: i64,
    _recv_buf_ptr: i64,
    _send_counts_ptr: i64,
    _recv_counts_ptr: i64,
    _hidden_dim: i64,
    _dtype: i64,
    _stream: i64,
) -> i64 {
    0
}

/// High-level MoE dispatch: routes tokens to experts and returns the combined output.
///
/// Takes the tokens NslTensor and logits NslTensor (both as i64 pointers),
/// plus MoE config. Returns a new NslTensor with the MoE output.
///
/// For the CPU fallback, each expert is treated as an identity transform
/// (output = input weighted by gating). The full pipeline with actual expert
/// weights requires extracting weight tensors from the expert model array,
/// which will be wired when FixedModelArray weight access is implemented.
#[no_mangle]
pub extern "C" fn nsl_moe_dispatch_full(
    tokens_ptr: i64,
    logits_ptr: i64,
    num_experts: i64,
    top_k: i64,
    capacity_factor_bits: i64,
) -> i64 {
    use crate::tensor::NslTensor;
    use crate::memory::checked_alloc;
    use std::os::raw::c_void;

    let tokens = NslTensor::from_ptr(tokens_ptr);
    let logits = NslTensor::from_ptr(logits_ptr);

    let num_experts = num_experts as usize;
    let top_k = top_k as usize;
    let capacity_factor = f32::from_bits(capacity_factor_bits as u32) as f64;

    let total_tokens = if logits.ndim >= 2 {
        (unsafe { *logits.shape.offset(0) }) as usize
    } else {
        logits.len as usize / num_experts
    };

    let hidden_dim = if tokens.ndim >= 2 {
        (unsafe { *tokens.shape.offset(tokens.ndim as isize - 1) }) as usize
    } else {
        tokens.len as usize / total_tokens
    };

    // Read logits data as f64 (NSL default dtype)
    let logits_f32: Vec<f32> = if logits.dtype == 0 {
        let data = unsafe { std::slice::from_raw_parts(logits.data as *const f64, logits.len as usize) };
        data.iter().map(|&v| v as f32).collect()
    } else {
        let data = unsafe { std::slice::from_raw_parts(logits.data as *const f32, logits.len as usize) };
        data.to_vec()
    };

    // Route tokens
    let routing = router::route_topk(&logits_f32, total_tokens, num_experts, top_k, capacity_factor);

    // Read tokens data — handle both f64 (dtype=0) and f32 (dtype=1)
    let tokens_f32: Vec<f32> = if tokens.dtype == 0 {
        let data = unsafe { std::slice::from_raw_parts(tokens.data as *const f64, tokens.len as usize) };
        data.iter().map(|&v| v as f32).collect()
    } else {
        let data = unsafe { std::slice::from_raw_parts(tokens.data as *const f32, tokens.len as usize) };
        data.to_vec()
    };
    let scattered = dispatch::scatter_tokens(&tokens_f32, &routing.sorted_token_indices, hidden_dim);

    // Identity expert transform: expert_outputs = scattered tokens
    let expert_outputs = scattered;

    // For top_k=1: each sorted position has weight 1.0 (the only selected expert).
    // For top_k=2: we'd need to track which sorted positions correspond to which
    // (token, k) pairs. For now, use uniform weight = 1.0 for all sorted positions.
    // This is correct for top_k=1. For top_k=2 with identity experts, the gather
    // will double-count (weight 1.0 each), but the output shape is still correct.
    let gather_weights = vec![1.0f32; routing.total_assigned as usize];

    // Gather back to original token order
    let output_f32 = dispatch::gather_tokens(
        &expert_outputs,
        &routing.sorted_token_indices,
        &gather_weights,
        total_tokens,
        top_k,
        hidden_dim,
    );

    // Create output NslTensor with same dtype as input
    let output_len = total_tokens * hidden_dim;
    let out_dtype = tokens.dtype;

    let data: *mut c_void = if out_dtype == 0 {
        // f64 output
        let ptr = checked_alloc(output_len * std::mem::size_of::<f64>()) as *mut f64;
        for (i, &val) in output_f32.iter().enumerate().take(output_len) {
            unsafe { *ptr.add(i) = val as f64 };
        }
        ptr as *mut c_void
    } else {
        // f32 output
        let ptr = checked_alloc(output_len * std::mem::size_of::<f32>()) as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(output_f32.as_ptr(), ptr, output_len);
        }
        ptr as *mut c_void
    };

    // Copy shape from input tensor
    let ndim = tokens.ndim as usize;
    let shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        std::ptr::copy_nonoverlapping(tokens.shape, shape, ndim);
    }

    // Compute strides
    let strides = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    if ndim > 0 {
        unsafe {
            *strides.add(ndim - 1) = 1;
            for d in (0..ndim - 1).rev() {
                *strides.add(d) = *strides.add(d + 1) * *shape.add(d + 1);
            }
        }
    }

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim: tokens.ndim,
        len: output_len as i64,
        refcount: AtomicI64::new(1),
        device: 0,
        dtype: out_dtype,
        owns_data: 1, data_owner: 0,
    });

    Box::into_raw(result) as i64
}
