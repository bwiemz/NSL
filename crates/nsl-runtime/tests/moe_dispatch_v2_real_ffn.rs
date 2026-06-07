//! CPDT Part III v1 production-forward (S1): validates
//! `nsl_moe_dispatch_full_v2` against a hand-composed granular-op reference
//! and confirms it is NOT a bit-identical identity on non-trivial inputs.
//!
//! The v1 dispatch (`nsl_moe_dispatch_full`) silently aliased `expert_outputs
//! = scattered` (file: nsl-runtime/src/moe/ffi.rs:287 pre-S1), so its output
//! was bit-equal to its input under top_k=1. v2 replaces that with a real
//! per-expert matmul against extracted weight tensors. These tests pin the
//! v2 contract:
//!
//!   - Numerical equality vs reference within 1e-6 f32 max_abs
//!   - Output is NOT bit-equal to input (proves the identity is gone)
//!   - Output shape is `[total_tokens, intermediate_dim]` (not hidden_dim)
//!   - Refusal gates return 0 (null) for bad inputs

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_zeros_on};

extern "C" {
    fn nsl_moe_dispatch_full_v2(
        tokens_ptr: i64,
        logits_ptr: i64,
        experts_ptr: i64,
        num_experts: i64,
        top_k: i64,
        capacity_factor_bits: i64,
        hidden_dim: i64,
        intermediate_dim: i64,
    ) -> i64;
}

/// Allocate a CPU f32 tensor of the given shape, filled with `vals`.
/// `nsl_tensor_zeros_on(_, 0)` produces dtype=f32 (`tensor_from_shape_list`
/// allocates `len * size_of::<f32>()`), so we can write the f32 values
/// directly into the buffer without an extra cast.
fn make_f32_tensor(shape: &[i64], vals: &[f32]) -> i64 {
    let shape_list = nsl_list_new();
    for &d in shape {
        nsl_list_push(shape_list, d);
    }
    let ptr = nsl_tensor_zeros_on(shape_list, 0);
    nsl_list_free(shape_list);
    let len: i64 = shape.iter().product();
    assert_eq!(vals.len() as i64, len, "shape vs vals length mismatch");
    let data = nsl_tensor_data_ptr(ptr) as *mut f32;
    for (i, &v) in vals.iter().enumerate() {
        unsafe { *data.add(i) = v };
    }
    ptr
}

fn read_f32(ptr: i64, len: usize) -> Vec<f32> {
    let data = nsl_tensor_data_ptr(ptr) as *const f32;
    (0..len).map(|i| unsafe { *data.add(i) }).collect()
}

/// Hand-composed granular-op reference: route_topk + scatter + per-expert
/// matmul + gather. Mirrors `crates/nsl-codegen/tests/cpdt_expert_prune.rs::
/// moe_forward` but computed directly in this test file for self-containment.
fn moe_forward_reference(
    tokens: &[f32],
    logits: &[f32],
    experts: &[f32],
    total_tokens: usize,
    num_experts: usize,
    hidden_dim: usize,
    intermediate_dim: usize,
    top_k: usize,
    capacity_factor: f64,
) -> Vec<f32> {
    use nsl_runtime::moe::{dispatch, router};
    let routing = router::route_topk(logits, total_tokens, num_experts, top_k, capacity_factor);
    let total_assigned = routing.total_assigned as usize;
    let sorted = dispatch::scatter_tokens(tokens, &routing.sorted_token_indices, hidden_dim);
    let boundaries = &routing.expert_boundaries;
    let mut sorted_outputs = vec![0.0_f32; total_assigned * intermediate_dim];
    for e in 0..num_experts {
        let start = boundaries[e] as usize;
        let end = boundaries[e + 1] as usize;
        if start == end {
            continue;
        }
        let w_off = e * hidden_dim * intermediate_dim;
        for t in start..end {
            let tok_off = t * hidden_dim;
            let out_off = t * intermediate_dim;
            for j in 0..intermediate_dim {
                let mut sum = 0.0_f32;
                for k in 0..hidden_dim {
                    sum += sorted[tok_off + k] * experts[w_off + k * intermediate_dim + j];
                }
                sorted_outputs[out_off + j] = sum;
            }
        }
    }
    // Use the routing's real per-(token, expert) weights so this reference
    // is correct for ANY top_k, not just top_k=1 where uniform 1.0 happens
    // to match. (For top_k=1, sorted_assignment_weights are all 1.0 anyway,
    // so callers passing top_k=1 see byte-identical output to the prior
    // uniform-vector path — the existing top_k=1 regression anchor stays
    // green and now also pins the new weights pipeline.)
    dispatch::gather_tokens(
        &sorted_outputs,
        &routing.sorted_token_indices,
        &routing.sorted_assignment_weights,
        total_tokens,
        top_k,
        intermediate_dim,
    )
}

#[test]
fn dispatch_v2_matches_granular_reference() {
    let total_tokens = 6_usize;
    let hidden_dim = 4_usize;
    let intermediate_dim = 5_usize;
    let num_experts = 3_usize;
    let top_k = 1_usize;
    let capacity_factor = 2.0_f64;

    // Non-trivial tokens: deterministic but varied per-token magnitudes.
    let tokens: Vec<f32> = (0..total_tokens * hidden_dim)
        .map(|i| 0.1 + 0.05 * (i as f32))
        .collect();
    // Logits that prefer different experts for different tokens — exercises
    // the per-expert boundaries logic with non-uniform assignment.
    let logits: Vec<f32> = (0..total_tokens * num_experts)
        .map(|i| {
            let token = i / num_experts;
            let expert = i % num_experts;
            ((token + 2 * expert) % 3) as f32
        })
        .collect();
    // Experts: distinct weight pattern per expert, so picking expert e vs e'
    // produces detectably different outputs.
    let experts: Vec<f32> = (0..num_experts * hidden_dim * intermediate_dim)
        .map(|i| {
            let e = i / (hidden_dim * intermediate_dim);
            0.01 * ((1 + e) as f32) * (1 + (i % 7)) as f32
        })
        .collect();

    let tokens_ptr = make_f32_tensor(&[total_tokens as i64, hidden_dim as i64], &tokens);
    let logits_ptr = make_f32_tensor(&[total_tokens as i64, num_experts as i64], &logits);
    let experts_ptr = make_f32_tensor(
        &[num_experts as i64, (hidden_dim * intermediate_dim) as i64],
        &experts,
    );

    let capacity_bits = (capacity_factor as f32).to_bits() as i64;
    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v2(
            tokens_ptr,
            logits_ptr,
            experts_ptr,
            num_experts as i64,
            top_k as i64,
            capacity_bits,
            hidden_dim as i64,
            intermediate_dim as i64,
        )
    };
    assert_ne!(out_ptr, 0, "v2 returned null pointer for valid inputs");

    // Output shape is verified implicitly: reading `total_tokens *
    // intermediate_dim` f32 elements and matching the granular-op reference
    // proves the buffer is the expected size + layout. If v2 produced a
    // hidden_dim trailing dim we'd see numerical garbage at the tail.
    let got = read_f32(out_ptr, total_tokens * intermediate_dim);
    let want = moe_forward_reference(
        &tokens, &logits, &experts,
        total_tokens, num_experts, hidden_dim, intermediate_dim, top_k, capacity_factor,
    );

    let mut max_abs_diff = 0.0_f32;
    for (g, w) in got.iter().zip(want.iter()) {
        let d = (g - w).abs();
        if d > max_abs_diff {
            max_abs_diff = d;
        }
    }
    assert!(
        max_abs_diff <= 1e-6,
        "v2 output diverges from hand-composed granular-op reference: max_abs_diff={max_abs_diff:.6e} (gate: <= 1e-6)"
    );

    // Cleanup
    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(experts_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v2_is_not_identity_on_distinct_experts() {
    // Use a setup where intermediate_dim == hidden_dim, so the v1 vs v2
    // shapes match and we can do a direct bit-comparison vs the input
    // tokens (the v1 identity behaviour). v2 should NEVER produce
    // tokens-bytes-back unless the expert weights are literally the
    // identity matrix.
    let total_tokens = 4_usize;
    let dim = 3_usize;
    let num_experts = 2_usize;

    let tokens: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        4.0, 5.0, 6.0,
        7.0, 8.0, 9.0,
        10.0, 11.0, 12.0,
    ];
    // Make every token route to expert 0.
    let logits: Vec<f32> = vec![5.0, 0.0, 5.0, 0.0, 5.0, 0.0, 5.0, 0.0];
    // Expert 0: scale-by-2; expert 1: identity (unused here).
    let experts: Vec<f32> = vec![
        2.0, 0.0, 0.0,
        0.0, 2.0, 0.0,
        0.0, 0.0, 2.0,
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
    ];

    let tokens_ptr = make_f32_tensor(&[total_tokens as i64, dim as i64], &tokens);
    let logits_ptr = make_f32_tensor(&[total_tokens as i64, num_experts as i64], &logits);
    let experts_ptr = make_f32_tensor(
        &[num_experts as i64, (dim * dim) as i64],
        &experts,
    );

    let capacity_bits = (2.0_f32).to_bits() as i64;
    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v2(
            tokens_ptr, logits_ptr, experts_ptr,
            num_experts as i64, 1, capacity_bits,
            dim as i64, dim as i64,
        )
    };
    assert_ne!(out_ptr, 0);

    let got = read_f32(out_ptr, total_tokens * dim);
    let want_scale_by_two: Vec<f32> = tokens.iter().map(|v| 2.0 * v).collect();
    for (g, w) in got.iter().zip(want_scale_by_two.iter()) {
        assert!(
            (g - w).abs() <= 1e-6,
            "v2 with scale-by-2 expert did not produce scaled tokens: got={got:?} want={want_scale_by_two:?}"
        );
    }
    // Identity refutation — v2 output must NOT equal tokens bitwise.
    let tokens_bits: Vec<u32> = tokens.iter().map(|v| v.to_bits()).collect();
    let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
    assert_ne!(
        tokens_bits, got_bits,
        "v2 output bit-equal to tokens — identity skeleton not actually replaced"
    );

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(experts_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v2_refuses_null_experts_ptr() {
    let tokens_ptr = make_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let logits_ptr = make_f32_tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v2(
            tokens_ptr, logits_ptr, 0 /* null experts */,
            2, 1, (1.0_f32).to_bits() as i64,
            3, 3,
        )
    };
    assert_eq!(
        out_ptr, 0,
        "v2 must return null (0) when experts_ptr is null — caller must treat as compile-time error"
    );
    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
}

#[test]
fn dispatch_v2_top_k_two_matches_hand_computed_weighted_sum() {
    // Tiny case where we can verify the entire gating-weight broadcast by
    // hand. 2 tokens, 2 experts (so top_k=2 routes every token to BOTH
    // experts), hidden=2, intermediate=2. Reference is computed
    // INDEPENDENTLY of route_topk / gather_tokens — pure scalar math on the
    // softmax + per-expert matmul + weighted sum. This catches:
    //
    //   - sorted_assignment_weights threaded into the wrong gather slot
    //   - The weights being normalized differently than expected
    //   - Any sign / index bug in the per-expert matmul under non-unit
    //     gating weights
    //
    // With 2 tokens × 2 experts and capacity_factor large enough (>= 1.0
    // since each expert sees exactly 2 tokens = total_tokens), no capacity
    // drop occurs and every assignment survives.
    let tokens: Vec<f32> = vec![
        1.0, 2.0, // token 0
        3.0, 4.0, // token 1
    ];
    // Logits give token 0 a 75/25 preference for expert 0, token 1 a
    // 25/75 preference for expert 1. Computed manually so we know the
    // reference softmax:
    //   token 0: logits = [ln 3, 0] → exp = [3, 1] → probs = [0.75, 0.25]
    //   token 1: logits = [0, ln 3] → exp = [1, 3] → probs = [0.25, 0.75]
    let ln3 = 3.0_f32.ln();
    let logits: Vec<f32> = vec![
        ln3, 0.0,
        0.0, ln3,
    ];
    // expert 0: W_0 = [[1, 0], [0, 1]]  (identity 2x2)
    // expert 1: W_1 = [[0, 1], [1, 0]]  (swap columns)
    let experts: Vec<f32> = vec![
        1.0, 0.0,  0.0, 1.0,   // expert 0 (row-major [hidden=2, intermediate=2])
        0.0, 1.0,  1.0, 0.0,   // expert 1
    ];
    // Reference, computed by hand:
    //   token 0 ⊗ W_0 = [1, 2]
    //   token 0 ⊗ W_1 = [2, 1]
    //   token 0 out   = 0.75 * [1, 2] + 0.25 * [2, 1] = [1.25, 1.75]
    //
    //   token 1 ⊗ W_0 = [3, 4]
    //   token 1 ⊗ W_1 = [4, 3]
    //   token 1 out   = 0.25 * [3, 4] + 0.75 * [4, 3] = [3.75, 3.25]
    let expected: Vec<f32> = vec![1.25, 1.75, 3.75, 3.25];

    let tokens_ptr = make_f32_tensor(&[2, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[2, 2], &logits);
    let experts_ptr = make_f32_tensor(&[2, 4], &experts);
    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v2(
            tokens_ptr, logits_ptr, experts_ptr,
            2,      // num_experts
            2,      // top_k
            (2.0_f32).to_bits() as i64, // capacity_factor: ample
            2, 2,   // hidden, intermediate
        )
    };
    assert_ne!(out_ptr, 0, "v2 returned null for valid top_k=2 inputs");
    let got = read_f32(out_ptr, 4);

    let mut max_abs = 0.0_f32;
    for (g, w) in got.iter().zip(expected.iter()) {
        let d = (g - w).abs();
        if d > max_abs { max_abs = d; }
    }
    assert!(
        max_abs <= 1e-6,
        "top_k=2 weighted sum diverges from hand-computed reference: got={got:?} want={expected:?} max_abs_diff={max_abs:.6e}"
    );

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(experts_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v2_accepts_top_k_two_after_gating_broadcast() {
    // top_k=2 used to be hard-refused because the gather hardcoded weight
    // 1.0 and would have double-counted. After threading
    // sorted_assignment_weights through, top_k=2 is supported: the
    // surviving pair of assignments per token are weighted-summed.
    // This test pins that v2 NO LONGER returns null for top_k=2.
    let tokens_ptr = make_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let logits_ptr = make_f32_tensor(&[2, 2], &[1.0, 0.5, 0.5, 1.0]);
    let experts_ptr = make_f32_tensor(&[2, 9], &vec![0.1_f32; 18]);
    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v2(
            tokens_ptr, logits_ptr, experts_ptr,
            2,
            2, /* top_k=2 now supported */
            (2.0_f32).to_bits() as i64,
            3, 3,
        )
    };
    assert_ne!(
        out_ptr, 0,
        "v2 must accept top_k=2 after gating-weight broadcast lands"
    );
    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(experts_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v2_refuses_top_k_out_of_router_range() {
    // route_topk asserts top_k in {1, 2}. The FFI fails closed BEFORE the
    // assertion so we don't panic across the C ABI. Pin both ends:
    // top_k=0 and top_k=3 must return null.
    let tokens_ptr = make_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let logits_ptr = make_f32_tensor(&[2, 2], &[1.0, 0.5, 0.5, 1.0]);
    let experts_ptr = make_f32_tensor(&[2, 9], &vec![0.1_f32; 18]);
    for &bad_top_k in &[0_i64, 3, 4] {
        let out_ptr = unsafe {
            nsl_moe_dispatch_full_v2(
                tokens_ptr, logits_ptr, experts_ptr,
                2,
                bad_top_k,
                (2.0_f32).to_bits() as i64,
                3, 3,
            )
        };
        assert_eq!(
            out_ptr, 0,
            "v2 must refuse top_k={} (router asserts top_k in {{1, 2}})",
            bad_top_k
        );
    }
    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(experts_ptr);
}

#[test]
fn dispatch_v2_refuses_wrong_experts_length() {
    let tokens_ptr = make_f32_tensor(&[2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let logits_ptr = make_f32_tensor(&[2, 2], &[1.0, 0.0, 0.0, 1.0]);
    // experts shape says n_experts=2, hidden=3, intermediate=3 → expected 18 elems.
    // But supply only 10 elements.
    let bad_experts = make_f32_tensor(&[10], &vec![0.5_f32; 10]);
    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v2(
            tokens_ptr, logits_ptr, bad_experts,
            2, 1, (1.0_f32).to_bits() as i64,
            3, 3,
        )
    };
    assert_eq!(
        out_ptr, 0,
        "v2 must return null when experts.len mismatches n_experts * hidden * intermediate"
    );
    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(bad_experts);
}
