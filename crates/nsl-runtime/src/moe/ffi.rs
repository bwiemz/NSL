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
