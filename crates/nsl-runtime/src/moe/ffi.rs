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

    let result = Box::new(NslTensor::new(
        data,
        shape,
        strides,
        tokens.ndim,
        output_len as i64,
        0,
        out_dtype,
        1,
        0,
    ));

    Box::into_raw(result) as i64
}

/// CPDT Part III v1 production-forward (M32 gap closure): high-level MoE
/// dispatch that routes tokens through REAL per-expert FFN weights — no
/// identity skeleton. Single-matmul "expert" semantics (matches the
/// granular-op reference at crates/nsl-codegen/tests/cpdt_expert_prune.rs
/// `moe_forward`): for each routed token, output = sorted_tokens × W_expert.
///
/// ABI: distinct symbol from `nsl_moe_dispatch_full` (kept byte-identical).
/// v2 adds `experts_ptr` (NslTensor packed `[n_experts, hidden_dim *
/// intermediate_dim]` row-major, f32 only) plus the two shape dimensions.
///
/// OUTPUT SHAPE: `[total_tokens, intermediate_dim]` (NOT `[total_tokens,
/// hidden_dim]` — every paper-faithful MoE expert FFN changes the trailing
/// dim from hidden to intermediate). Callers that previously consumed v1's
/// hidden-dim identity output must adjust accordingly.
///
/// REFUSALS (return 0 — caller MUST treat as compile-time error):
///   - experts_ptr == 0
///   - num_experts == 0
///   - top_k not in {1, 2} — `route_topk` itself asserts this; v2 fails
///     closed at the FFI boundary rather than panicking inside the router.
///   - experts.dtype != tokens.dtype (mixed-dtype silent-corruption hazard)
///   - experts.len != num_experts * hidden_dim * intermediate_dim
///
/// GATING-WEIGHT BROADCAST: top_k > 1 now uses the real per-token-per-
/// expert routing weights from `routing.sorted_assignment_weights`
/// (router-normalized so the surviving pair sums to 1.0 per token).
/// `gather_tokens` performs the weighted sum across each token's
/// surviving assignments.
///
/// Deferrals (v2.next): bias, activation, down-proj, capacity overflow
/// handling, GPU expert_parallel_matmul, FP16 experts.
#[no_mangle]
pub extern "C" fn nsl_moe_dispatch_full_v2(
    tokens_ptr: i64,
    logits_ptr: i64,
    experts_ptr: i64,
    num_experts: i64,
    top_k: i64,
    capacity_factor_bits: i64,
    hidden_dim: i64,
    intermediate_dim: i64,
) -> i64 {
    use crate::tensor::NslTensor;
    use crate::memory::checked_alloc;
    use std::os::raw::c_void;

    // ── Refusal gates — return 0 (null) on bad input ────────────────────
    if experts_ptr == 0 || num_experts <= 0 || hidden_dim <= 0 || intermediate_dim <= 0 {
        return 0;
    }
    // route_topk asserts top_k in {1, 2}. Fail closed at the FFI boundary
    // instead of letting the assertion panic across the C ABI.
    if top_k != 1 && top_k != 2 {
        return 0;
    }

    let tokens = NslTensor::from_ptr(tokens_ptr);
    let logits = NslTensor::from_ptr(logits_ptr);
    let experts = NslTensor::from_ptr(experts_ptr);

    // Mixed-dtype refusal: v1 byte-slices experts dtype-agnostically at
    // compile time (cpdt_expert_prune::MixedDtypeUnsupported), but at
    // runtime v2 needs a uniform dtype to safely interpret the buffer.
    if experts.dtype != tokens.dtype {
        return 0;
    }

    let num_experts_us = num_experts as usize;
    let hidden_dim_us = hidden_dim as usize;
    let intermediate_dim_us = intermediate_dim as usize;
    let top_k = top_k as usize;
    let capacity_factor = f32::from_bits(capacity_factor_bits as u32) as f64;

    // Shape check: experts must be exactly [num_experts, hidden * intermediate].
    let expected_experts_elems = num_experts_us
        .saturating_mul(hidden_dim_us)
        .saturating_mul(intermediate_dim_us);
    if (experts.len as usize) != expected_experts_elems {
        return 0;
    }

    let total_tokens = if logits.ndim >= 2 {
        (unsafe { *logits.shape.offset(0) }) as usize
    } else {
        logits.len as usize / num_experts_us
    };

    // Read logits/tokens/experts as f32 — supporting both NSL CPU f64 default
    // (dtype=0) and explicit f32 (dtype=1). For dtype symmetry we cast all
    // three to f32 for the matmul.
    let read_f32 = |t: &NslTensor| -> Vec<f32> {
        if t.dtype == 0 {
            let data = unsafe { std::slice::from_raw_parts(t.data as *const f64, t.len as usize) };
            data.iter().map(|&v| v as f32).collect()
        } else {
            let data = unsafe { std::slice::from_raw_parts(t.data as *const f32, t.len as usize) };
            data.to_vec()
        }
    };

    let logits_f32 = read_f32(&logits);
    let tokens_f32 = read_f32(&tokens);
    let experts_f32 = read_f32(&experts);

    // Route tokens — same as v1.
    let routing = router::route_topk(
        &logits_f32,
        total_tokens,
        num_experts_us,
        top_k,
        capacity_factor,
    );
    let total_assigned = routing.total_assigned as usize;

    // Scatter tokens into sorted-by-expert layout [total_assigned, hidden].
    let sorted_tokens = dispatch::scatter_tokens(
        &tokens_f32,
        &routing.sorted_token_indices,
        hidden_dim_us,
    );

    // Per-expert matmul: sorted_outputs[t, j] = Σ_k sorted_tokens[t, k] * W_e[k, j]
    // where t ranges over [boundaries[e], boundaries[e+1]) for expert e. Mirrors
    // nsl_expert_parallel_matmul (ffi.rs:85) inline-replicated so we don't
    // need to allocate an intermediate raw-buffer just for that FFI call.
    let mut sorted_outputs = vec![0.0_f32; total_assigned * intermediate_dim_us];
    let boundaries = &routing.expert_boundaries;
    for e in 0..num_experts_us {
        let start = boundaries[e] as usize;
        let end = boundaries[e + 1] as usize;
        if start == end {
            continue;
        }
        let w_off = e * hidden_dim_us * intermediate_dim_us;
        for t in start..end {
            let tok_off = t * hidden_dim_us;
            let out_off = t * intermediate_dim_us;
            for j in 0..intermediate_dim_us {
                let mut sum = 0.0_f32;
                for k in 0..hidden_dim_us {
                    sum += sorted_tokens[tok_off + k]
                        * experts_f32[w_off + k * intermediate_dim_us + j];
                }
                sorted_outputs[out_off + j] = sum;
            }
        }
    }

    // Gather back to token order using the real per-(token, expert)
    // routing weights from sorted_assignment_weights. For top_k=1 every
    // weight is 1.0 (single-element softmax), so this remains bit-exact
    // against the prior uniform-1.0 path. For top_k=2 each surviving
    // assignment carries its router-normalized share; gather_tokens
    // accumulates the weighted sum back to the original token row.
    let output_f32 = dispatch::gather_tokens(
        &sorted_outputs,
        &routing.sorted_token_indices,
        &routing.sorted_assignment_weights,
        total_tokens,
        top_k,
        intermediate_dim_us,
    );

    // Allocate output NslTensor with shape [total_tokens, intermediate_dim].
    // This is the v1↔v2 ABI difference downstream callers must absorb.
    let output_len = total_tokens * intermediate_dim_us;
    let out_dtype = tokens.dtype;
    let data: *mut c_void = if out_dtype == 0 {
        let ptr = checked_alloc(output_len * std::mem::size_of::<f64>()) as *mut f64;
        for (i, &val) in output_f32.iter().enumerate().take(output_len) {
            unsafe { *ptr.add(i) = val as f64 };
        }
        ptr as *mut c_void
    } else {
        let ptr = checked_alloc(output_len * std::mem::size_of::<f32>()) as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(output_f32.as_ptr(), ptr, output_len);
        }
        ptr as *mut c_void
    };

    // Output is always 2D: [total_tokens, intermediate_dim].
    let ndim: i64 = 2;
    let shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *shape = total_tokens as i64;
        *shape.add(1) = intermediate_dim;
    }
    let strides = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *strides.add(1) = 1;
        *strides = intermediate_dim;
    }

    let result = Box::new(NslTensor::new(
        data,
        shape,
        strides,
        ndim,
        output_len as i64,
        0,
        out_dtype,
        1,
        0,
    ));

    Box::into_raw(result) as i64
}

/// CPDT Part III v2.2 — paper-faithful MoE FFN.
///
/// Layers a second matmul + activation on top of v2's single-matmul
/// "expert", producing the standard MoE FFN structure used by
/// Mixtral / DeepSeek MoE / etc.:
///
///   x_up   = x_token  @ W_up_e                # [hidden] -> [intermediate]
///   x_act  = activation(x_up)                 # in-place; SiLU = x * sigmoid(x)
///   x_down = x_act    @ W_down_e              # [intermediate] -> [hidden]
///   out    = Σ_e routing_weight[t, e] * x_down
///
/// Output shape is `[total_tokens, hidden_dim]` — distinct from v2 which
/// stopped at intermediate_dim. Downstream consumers reading the
/// trailing dim of v3 vs v2 outputs will see different sizes.
///
/// CONTRACT vs v1/v2:
///   - v1 (`nsl_moe_dispatch_full`): identity skeleton; output ==
///     scattered tokens at hidden_dim. KEPT BYTE-IDENTICAL.
///   - v2 (`nsl_moe_dispatch_full_v2`): real W_up matmul; output at
///     intermediate_dim. KEPT BYTE-IDENTICAL.
///   - v3 (this fn): adds activation + W_down; output back at
///     hidden_dim. Routing/scatter/gather pipeline IS SHARED with
///     v2 — only the per-expert compute kernel differs.
///
/// WEIGHTS LAYOUT (both packed [n_experts, ...] row-major):
///   - experts_up_ptr   shape == [n_experts, hidden_dim * intermediate_dim]
///   - experts_down_ptr shape == [n_experts, intermediate_dim * hidden_dim]
///
/// ACTIVATION SELECTION:
///   - activation_kind == 0 → identity (no activation; only useful to
///     differentiate v3-vs-v2 at the test level; production should use
///     a real activation)
///   - activation_kind == 1 → SiLU = x * sigmoid(x). Mixtral / DeepSeek
///     MoE default.
///   - activation_kind == 2 → GELU (tanh approximation):
///       0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3))).
///     Matches PyTorch's `F.gelu(approximate='tanh')`. Used by GPT-2
///     and earlier transformer FFNs.
///   - activation_kind == 3 → ReLU = max(0, x). Pre-modern; included
///     mostly for completeness + as a sanity-check baseline.
///   - activation_kind ∈ other → refused (return 0). SwiGLU = (silu(gate)
///     ⊙ up) @ down needs a third weight matrix and is a separate
///     v2.5 FFI variant (v4).
///
/// REFUSALS (return 0):
///   - any of experts_up_ptr, experts_down_ptr is null
///   - num_experts/hidden/intermediate <= 0
///   - top_k ∉ {1, 2} (route_topk asserts {1,2})
///   - activation_kind ∉ {0, 1, 2, 3}
///   - mixed dtype between tokens / experts_up / experts_down
///   - experts_up.len   != n_experts * hidden * intermediate
///   - experts_down.len != n_experts * intermediate * hidden
///
/// Deferrals (v2.next): bias on either matmul,
/// gating-projection variant (SwiGLU = (gate ⊙ silu(up)) @ down),
/// capacity-overflow renorm, GPU expert_parallel_matmul,
/// FP16/mixed-precision experts, codegen lowering (v3 is reachable
/// only via direct FFI today, mirroring v2's pre-codegen state).
#[no_mangle]
pub extern "C" fn nsl_moe_dispatch_full_v3(
    tokens_ptr: i64,
    logits_ptr: i64,
    experts_up_ptr: i64,
    experts_down_ptr: i64,
    num_experts: i64,
    top_k: i64,
    capacity_factor_bits: i64,
    hidden_dim: i64,
    intermediate_dim: i64,
    activation_kind: i64,
) -> i64 {
    use crate::tensor::NslTensor;
    use crate::memory::checked_alloc;
    use std::os::raw::c_void;

    // ── Refusal gates ──────────────────────────────────────────────────
    if experts_up_ptr == 0 || experts_down_ptr == 0
        || num_experts <= 0 || hidden_dim <= 0 || intermediate_dim <= 0
    {
        return 0;
    }
    if top_k != 1 && top_k != 2 {
        return 0;
    }
    if !(0..=3).contains(&activation_kind) {
        return 0;
    }

    let tokens = NslTensor::from_ptr(tokens_ptr);
    let logits = NslTensor::from_ptr(logits_ptr);
    let experts_up = NslTensor::from_ptr(experts_up_ptr);
    let experts_down = NslTensor::from_ptr(experts_down_ptr);

    // Uniform dtype required across tokens / both expert tensors. v2.next
    // will add mixed-precision; for v2.2 we mirror v2's refusal.
    if experts_up.dtype != tokens.dtype || experts_down.dtype != tokens.dtype {
        return 0;
    }

    let num_experts_us = num_experts as usize;
    let hidden_dim_us = hidden_dim as usize;
    let intermediate_dim_us = intermediate_dim as usize;
    let top_k = top_k as usize;
    let capacity_factor = f32::from_bits(capacity_factor_bits as u32) as f64;

    // Shape checks.
    let expected_up = num_experts_us
        .saturating_mul(hidden_dim_us)
        .saturating_mul(intermediate_dim_us);
    let expected_down = num_experts_us
        .saturating_mul(intermediate_dim_us)
        .saturating_mul(hidden_dim_us);
    if (experts_up.len as usize) != expected_up || (experts_down.len as usize) != expected_down {
        return 0;
    }

    let total_tokens = if logits.ndim >= 2 {
        (unsafe { *logits.shape.offset(0) }) as usize
    } else {
        logits.len as usize / num_experts_us
    };

    let read_f32 = |t: &NslTensor| -> Vec<f32> {
        if t.dtype == 0 {
            let data = unsafe { std::slice::from_raw_parts(t.data as *const f64, t.len as usize) };
            data.iter().map(|&v| v as f32).collect()
        } else {
            let data = unsafe { std::slice::from_raw_parts(t.data as *const f32, t.len as usize) };
            data.to_vec()
        }
    };

    let logits_f32 = read_f32(&logits);
    let tokens_f32 = read_f32(&tokens);
    let experts_up_f32 = read_f32(&experts_up);
    let experts_down_f32 = read_f32(&experts_down);

    let routing = router::route_topk(
        &logits_f32, total_tokens, num_experts_us, top_k, capacity_factor,
    );
    let total_assigned = routing.total_assigned as usize;

    let sorted_tokens = dispatch::scatter_tokens(
        &tokens_f32, &routing.sorted_token_indices, hidden_dim_us,
    );

    // Per-expert up-matmul → activation → down-matmul. Output buffer is
    // sized for hidden_dim trailing axis (back to the input size).
    let mut sorted_outputs = vec![0.0_f32; total_assigned * hidden_dim_us];
    // Intermediate scratch reused per expert range to avoid reallocating.
    let mut intermediate = vec![0.0_f32; intermediate_dim_us];
    let boundaries = &routing.expert_boundaries;
    for e in 0..num_experts_us {
        let start = boundaries[e] as usize;
        let end = boundaries[e + 1] as usize;
        if start == end {
            continue;
        }
        let up_off = e * hidden_dim_us * intermediate_dim_us;
        let down_off = e * intermediate_dim_us * hidden_dim_us;
        for t in start..end {
            let tok_off = t * hidden_dim_us;
            let out_off = t * hidden_dim_us;

            // x_up[j] = Σ_k token[k] * W_up[e][k, j]
            for j in 0..intermediate_dim_us {
                let mut sum = 0.0_f32;
                for k in 0..hidden_dim_us {
                    sum += sorted_tokens[tok_off + k]
                        * experts_up_f32[up_off + k * intermediate_dim_us + j];
                }
                intermediate[j] = sum;
            }

            // Activation. activation_kind selects the elementwise
            // nonlinearity. All variants are numerically safe for all
            // finite f32; NaN inputs propagate NaN (caller's
            // responsibility, consistent with the rest of moe/ffi.rs).
            //
            //   0 → identity (skip)
            //   1 → SiLU = x * sigmoid(x)
            //   2 → GELU (tanh approximation, matches torch.gelu approx="tanh")
            //   3 → ReLU = max(0, x)
            match activation_kind {
                0 => {}
                1 => {
                    for v in intermediate.iter_mut() {
                        let s = 1.0_f32 / (1.0_f32 + (-*v).exp());
                        *v *= s;
                    }
                }
                2 => {
                    // GELU(x) ≈ 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))
                    // 0.7978845608 = sqrt(2.0 / std::f32::consts::PI), precomputed
                    // as an f32 constant to avoid bringing the dependency into
                    // this inner loop.
                    const SQRT_2_OVER_PI: f32 = 0.7978845608028654_f32;
                    const GELU_CUBIC: f32 = 0.044715_f32;
                    for v in intermediate.iter_mut() {
                        let x = *v;
                        let inner = SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x);
                        *v = 0.5_f32 * x * (1.0_f32 + inner.tanh());
                    }
                }
                3 => {
                    for v in intermediate.iter_mut() {
                        if *v < 0.0_f32 {
                            *v = 0.0_f32;
                        }
                    }
                }
                _ => unreachable!("activation_kind {} bypassed the upfront refusal gate", activation_kind),
            }

            // x_down[h] = Σ_j x_act[j] * W_down[e][j, h]
            for h in 0..hidden_dim_us {
                let mut sum = 0.0_f32;
                for j in 0..intermediate_dim_us {
                    sum += intermediate[j]
                        * experts_down_f32[down_off + j * hidden_dim_us + h];
                }
                sorted_outputs[out_off + h] = sum;
            }
        }
    }

    // Gather back with the real routing weights — same alignment as v2.1.
    let output_f32 = dispatch::gather_tokens(
        &sorted_outputs,
        &routing.sorted_token_indices,
        &routing.sorted_assignment_weights,
        total_tokens,
        top_k,
        hidden_dim_us,
    );

    let output_len = total_tokens * hidden_dim_us;
    let out_dtype = tokens.dtype;
    let data: *mut c_void = if out_dtype == 0 {
        let ptr = checked_alloc(output_len * std::mem::size_of::<f64>()) as *mut f64;
        for (i, &val) in output_f32.iter().enumerate().take(output_len) {
            unsafe { *ptr.add(i) = val as f64 };
        }
        ptr as *mut c_void
    } else {
        let ptr = checked_alloc(output_len * std::mem::size_of::<f32>()) as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(output_f32.as_ptr(), ptr, output_len);
        }
        ptr as *mut c_void
    };

    let ndim: i64 = 2;
    let shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *shape = total_tokens as i64;
        *shape.add(1) = hidden_dim;
    }
    let strides = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *strides.add(1) = 1;
        *strides = hidden_dim;
    }

    let result = Box::new(NslTensor::new(
        data,
        shape,
        strides,
        ndim,
        output_len as i64,
        0,
        out_dtype,
        1,
        0,
    ));

    Box::into_raw(result) as i64
}

/// CPDT Part III v2.5 — Mixtral's SwiGLU MoE FFN.
///
/// SwiGLU is the dominant production MoE activation (Mixtral, DeepSeek
/// MoE, Llama-3 dense FFNs). It uses a THIRD weight matrix beyond v3's
/// up + down, splitting the up-projection into a gate path and a
/// straight path, then element-wise multiplying after the nonlinearity:
///
///   x_gate = token  @ W_gate[e]    # [hidden] -> [intermediate]
///   x_up   = token  @ W_up[e]      # [hidden] -> [intermediate]
///   x_act  = silu(x_gate) * x_up   # GLU step (element-wise)
///   x_down = x_act  @ W_down[e]    # [intermediate] -> [hidden]
///   out    = sum over e of routing_weight[t, e] * x_down
///
/// Output shape `[total_tokens, hidden_dim]` — same as v3 (returns to
/// the input hidden dim via down-proj).
///
/// CONTRACT vs v1/v2/v3:
///   - v1 (`nsl_moe_dispatch_full`): identity skeleton at hidden_dim.
///   - v2 (`nsl_moe_dispatch_full_v2`): real W_up matmul; output at
///     intermediate_dim.
///   - v3 (`nsl_moe_dispatch_full_v3`): W_up -> activation -> W_down,
///     output at hidden_dim, single scalar activation.
///   - v4 (this fn): GATE + W_up -> SiLU-gated multiply -> W_down,
///     output at hidden_dim. Routing / scatter / gather pipeline IS
///     SHARED with v2/v3; only the per-expert kernel is new.
///
/// WEIGHTS LAYOUT (all packed `[n_experts, ...]` row-major):
///   - experts_gate_ptr shape == [n_experts, hidden_dim * intermediate_dim]
///   - experts_up_ptr   shape == [n_experts, hidden_dim * intermediate_dim]
///   - experts_down_ptr shape == [n_experts, intermediate_dim * hidden_dim]
///
/// The gate-activation is hardcoded to SiLU per the Mixtral / DeepSeek
/// MoE convention. GeGLU (gate-activation = GELU) and ReGLU would be
/// v2.next variants; they share the same FFI shape with a different
/// inner nonlinearity. For now we don't expose an activation_kind arg
/// — adding it later is a non-breaking signature extension.
///
/// REFUSALS (return 0):
///   - tokens_ptr or logits_ptr is null
///   - any of experts_gate_ptr, experts_up_ptr, experts_down_ptr is null
///   - num_experts/hidden/intermediate <= 0
///   - top_k not in {1, 2} (route_topk asserts {1, 2})
///   - mixed dtype between tokens / experts_{gate, up, down}
///   - experts_gate.len != n_experts * hidden * intermediate
///   - experts_up.len   != n_experts * hidden * intermediate
///   - experts_down.len != n_experts * intermediate * hidden
///
/// Deferrals (v2.next): bias on any matmul, GeGLU / ReGLU variants,
/// capacity-overflow renorm, GPU expert_parallel_matmul,
/// FP16/mixed-precision experts.
#[no_mangle]
pub extern "C" fn nsl_moe_dispatch_full_v4(
    tokens_ptr: i64,
    logits_ptr: i64,
    experts_gate_ptr: i64,
    experts_up_ptr: i64,
    experts_down_ptr: i64,
    num_experts: i64,
    top_k: i64,
    capacity_factor_bits: i64,
    hidden_dim: i64,
    intermediate_dim: i64,
) -> i64 {
    use crate::tensor::NslTensor;
    use crate::memory::checked_alloc;
    use std::os::raw::c_void;

    // Refusal gates. Null pointers MUST be rejected BEFORE any
    // NslTensor::from_ptr — that call materializes a `&mut NslTensor`
    // from the raw integer, and forming a reference from a null
    // pointer is instant UB under Rust's reference-creation rules
    // (the debug_assert! on the magic field reads the struct AFTER
    // the reference is formed, and is elided in release). v4 is a
    // public `#[no_mangle] extern "C"` boundary, so direct FFI
    // callers (C ABI, tests, third-party wrappers) can pass 0 here.
    if tokens_ptr == 0 || logits_ptr == 0 {
        return 0;
    }
    if experts_gate_ptr == 0
        || experts_up_ptr == 0
        || experts_down_ptr == 0
        || num_experts <= 0
        || hidden_dim <= 0
        || intermediate_dim <= 0
    {
        return 0;
    }
    if top_k != 1 && top_k != 2 {
        return 0;
    }

    let tokens = NslTensor::from_ptr(tokens_ptr);
    let logits = NslTensor::from_ptr(logits_ptr);
    let experts_gate = NslTensor::from_ptr(experts_gate_ptr);
    let experts_up = NslTensor::from_ptr(experts_up_ptr);
    let experts_down = NslTensor::from_ptr(experts_down_ptr);

    if experts_gate.dtype != tokens.dtype
        || experts_up.dtype != tokens.dtype
        || experts_down.dtype != tokens.dtype
    {
        return 0;
    }

    let num_experts_us = num_experts as usize;
    let hidden_dim_us = hidden_dim as usize;
    let intermediate_dim_us = intermediate_dim as usize;
    let top_k = top_k as usize;
    let capacity_factor = f32::from_bits(capacity_factor_bits as u32) as f64;

    let expected_gate_up = num_experts_us
        .saturating_mul(hidden_dim_us)
        .saturating_mul(intermediate_dim_us);
    let expected_down = num_experts_us
        .saturating_mul(intermediate_dim_us)
        .saturating_mul(hidden_dim_us);
    if (experts_gate.len as usize) != expected_gate_up
        || (experts_up.len as usize) != expected_gate_up
        || (experts_down.len as usize) != expected_down
    {
        return 0;
    }

    let total_tokens = if logits.ndim >= 2 {
        (unsafe { *logits.shape.offset(0) }) as usize
    } else {
        logits.len as usize / num_experts_us
    };

    let read_f32 = |t: &NslTensor| -> Vec<f32> {
        if t.dtype == 0 {
            let data = unsafe { std::slice::from_raw_parts(t.data as *const f64, t.len as usize) };
            data.iter().map(|&v| v as f32).collect()
        } else {
            let data = unsafe { std::slice::from_raw_parts(t.data as *const f32, t.len as usize) };
            data.to_vec()
        }
    };

    let logits_f32 = read_f32(&logits);
    let tokens_f32 = read_f32(&tokens);
    let experts_gate_f32 = read_f32(&experts_gate);
    let experts_up_f32 = read_f32(&experts_up);
    let experts_down_f32 = read_f32(&experts_down);

    let routing = router::route_topk(
        &logits_f32, total_tokens, num_experts_us, top_k, capacity_factor,
    );
    let total_assigned = routing.total_assigned as usize;

    let sorted_tokens = dispatch::scatter_tokens(
        &tokens_f32, &routing.sorted_token_indices, hidden_dim_us,
    );

    let mut sorted_outputs = vec![0.0_f32; total_assigned * hidden_dim_us];
    let mut gate_scratch = vec![0.0_f32; intermediate_dim_us];
    let mut up_scratch = vec![0.0_f32; intermediate_dim_us];
    let boundaries = &routing.expert_boundaries;
    for e in 0..num_experts_us {
        let start = boundaries[e] as usize;
        let end = boundaries[e + 1] as usize;
        if start == end {
            continue;
        }
        let gate_off = e * hidden_dim_us * intermediate_dim_us;
        let up_off = e * hidden_dim_us * intermediate_dim_us;
        let down_off = e * intermediate_dim_us * hidden_dim_us;
        for t in start..end {
            let tok_off = t * hidden_dim_us;
            let out_off = t * hidden_dim_us;

            // Combined gate+up matmul — token bytes loaded once per
            // k-iteration and applied to both gate and up weights.
            for j in 0..intermediate_dim_us {
                let mut sum_g = 0.0_f32;
                let mut sum_u = 0.0_f32;
                for k in 0..hidden_dim_us {
                    let tk = sorted_tokens[tok_off + k];
                    sum_g += tk * experts_gate_f32[gate_off + k * intermediate_dim_us + j];
                    sum_u += tk * experts_up_f32[up_off + k * intermediate_dim_us + j];
                }
                gate_scratch[j] = sum_g;
                up_scratch[j] = sum_u;
            }

            // GLU step: x_act[j] = silu(x_gate[j]) * x_up[j].
            // SiLU numerical edges already proven for v3. Writes back
            // into up_scratch so the down-matmul reads a single buffer.
            for j in 0..intermediate_dim_us {
                let g = gate_scratch[j];
                let s = 1.0_f32 / (1.0_f32 + (-g).exp());
                up_scratch[j] = (g * s) * up_scratch[j];
            }

            // x_down[h] = sum over j of x_act[j] * W_down[e][j, h]
            for h in 0..hidden_dim_us {
                let mut sum = 0.0_f32;
                for j in 0..intermediate_dim_us {
                    sum += up_scratch[j]
                        * experts_down_f32[down_off + j * hidden_dim_us + h];
                }
                sorted_outputs[out_off + h] = sum;
            }
        }
    }

    let output_f32 = dispatch::gather_tokens(
        &sorted_outputs,
        &routing.sorted_token_indices,
        &routing.sorted_assignment_weights,
        total_tokens,
        top_k,
        hidden_dim_us,
    );

    let output_len = total_tokens * hidden_dim_us;
    let out_dtype = tokens.dtype;
    let data: *mut c_void = if out_dtype == 0 {
        let ptr = checked_alloc(output_len * std::mem::size_of::<f64>()) as *mut f64;
        for (i, &val) in output_f32.iter().enumerate().take(output_len) {
            unsafe { *ptr.add(i) = val as f64 };
        }
        ptr as *mut c_void
    } else {
        let ptr = checked_alloc(output_len * std::mem::size_of::<f32>()) as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(output_f32.as_ptr(), ptr, output_len);
        }
        ptr as *mut c_void
    };

    let ndim: i64 = 2;
    let shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *shape = total_tokens as i64;
        *shape.add(1) = hidden_dim;
    }
    let strides = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *strides.add(1) = 1;
        *strides = hidden_dim;
    }

    let result = Box::new(NslTensor::new(
        data,
        shape,
        strides,
        ndim,
        output_len as i64,
        0,
        out_dtype,
        1,
        0,
    ));

    Box::into_raw(result) as i64
}
