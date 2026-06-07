//! CPDT Part III v2.2: paper-faithful MoE FFN (`nsl_moe_dispatch_full_v3`).
//!
//! v2 was a single matmul per "expert", producing `[total_tokens,
//! intermediate_dim]`. v3 adds the second half of the FFN: activation +
//! W_down matmul, returning to `[total_tokens, hidden_dim]`. Pinned with
//! hand-computed references (no shared internals with the v2 tests so
//! a regression in route_topk / scatter / gather cannot mask a v3 bug).

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_zeros_on};

extern "C" {
    fn nsl_moe_dispatch_full_v3(
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
    ) -> i64;
}

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

fn silu(x: f32) -> f32 {
    x * (1.0_f32 / (1.0_f32 + (-x).exp()))
}

/// Single-token, single-expert forward pass. Used as the building block
/// for the hand-computed multi-token reference, independent of
/// route_topk / scatter / gather. Returns `out[hidden]`.
fn ffn_expert_silu(
    token: &[f32],
    w_up: &[f32],   // shape [hidden, intermediate], row-major
    w_down: &[f32], // shape [intermediate, hidden], row-major
    hidden: usize,
    intermediate: usize,
) -> Vec<f32> {
    let mut x_up = vec![0.0_f32; intermediate];
    for j in 0..intermediate {
        let mut s = 0.0_f32;
        for k in 0..hidden {
            s += token[k] * w_up[k * intermediate + j];
        }
        x_up[j] = s;
    }
    let x_act: Vec<f32> = x_up.iter().map(|&v| silu(v)).collect();
    let mut x_down = vec![0.0_f32; hidden];
    for h in 0..hidden {
        let mut s = 0.0_f32;
        for j in 0..intermediate {
            s += x_act[j] * w_down[j * hidden + h];
        }
        x_down[h] = s;
    }
    x_down
}

#[test]
fn dispatch_v3_silu_top_k_two_matches_hand_computed_reference() {
    // 2 tokens, 2 experts, top_k=2 → every token visits BOTH experts so
    // the gather is genuinely a weighted sum (not just identity through
    // one expert). hidden=2, intermediate=2 keeps the by-hand math
    // tractable but exercises every code path.
    //
    // Routing: logits [ln3, 0] → probs [0.75, 0.25] for token 0,
    //                [0, ln3] → probs [0.25, 0.75] for token 1.
    let ln3 = 3.0_f32.ln();
    let tokens: Vec<f32> = vec![
        1.0, 2.0,   // token 0
        3.0, 4.0,   // token 1
    ];
    let logits: Vec<f32> = vec![
        ln3, 0.0,
        0.0, ln3,
    ];
    // W_up_0 = identity, W_up_1 = swap. (shape [hidden=2, intermediate=2])
    let w_up: Vec<f32> = vec![
        1.0, 0.0,  0.0, 1.0,  // expert 0
        0.0, 1.0,  1.0, 0.0,  // expert 1
    ];
    // W_down_0 = identity, W_down_1 = swap. (shape [intermediate=2, hidden=2])
    let w_down: Vec<f32> = vec![
        1.0, 0.0,  0.0, 1.0,  // expert 0
        0.0, 1.0,  1.0, 0.0,  // expert 1
    ];

    // Hand-compose the reference:
    //   token 0 @ expert 0:
    //     x_up   = [1, 2]
    //     silu   = [silu(1), silu(2)]
    //     x_down = silu(1) * [1, 0] + silu(2) * [0, 1] = [silu(1), silu(2)]
    //   token 0 @ expert 1:
    //     x_up   = [2, 1]
    //     silu   = [silu(2), silu(1)]
    //     x_down = silu(2) * [0, 1] + silu(1) * [1, 0] = [silu(1), silu(2)]
    //   weighted: 0.75 * [silu(1), silu(2)] + 0.25 * [silu(1), silu(2)]
    //           = [silu(1), silu(2)]  (both experts produce the same output here)
    //
    //   token 1 @ expert 0: x_up=[3,4] → silu → x_down=[silu(3), silu(4)]
    //   token 1 @ expert 1: x_up=[4,3] → silu → x_down=[silu(3), silu(4)]
    //   weighted: 0.25 * [silu(3), silu(4)] + 0.75 * [silu(3), silu(4)]
    //           = [silu(3), silu(4)]
    //
    // The symmetry of identity + swap collapses the expert-pair to a
    // common output; we still cover every code path (two matmuls, the
    // activation, the gather weighted sum), and the by-hand prediction
    // is deterministic.
    let expected: Vec<f32> = vec![
        silu(1.0), silu(2.0),
        silu(3.0), silu(4.0),
    ];

    let tokens_ptr = make_f32_tensor(&[2, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[2, 2], &logits);
    let w_up_ptr = make_f32_tensor(&[2, 4], &w_up);
    let w_down_ptr = make_f32_tensor(&[2, 4], &w_down);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_ptr, logits_ptr,
            w_up_ptr, w_down_ptr,
            2,                              // num_experts
            2,                              // top_k
            (2.0_f32).to_bits() as i64,     // capacity
            2,                              // hidden_dim
            2,                              // intermediate_dim
            1,                              // activation_kind = SiLU
        )
    };
    assert_ne!(out_ptr, 0, "v3 returned null for valid SiLU top_k=2 inputs");
    let got = read_f32(out_ptr, 4);

    let mut max_abs = 0.0_f32;
    for (g, w) in got.iter().zip(expected.iter()) {
        let d = (g - w).abs();
        if d > max_abs { max_abs = d; }
    }
    assert!(
        max_abs <= 1e-6,
        "v3 SiLU MoE FFN diverges from hand-computed reference: got={got:?} want={expected:?} max_abs_diff={max_abs:.6e}"
    );

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v3_distinct_experts_top_k_one_matches_per_expert_reference() {
    // top_k=1 with experts that have DISTINCT outputs (so the test can
    // catch a bug where the wrong expert is consulted or the weights
    // are silently swapped). 4 tokens routed deterministically across 3
    // experts. Hand-compute each token's output via the single-expert
    // ffn_expert_silu helper.
    let total_tokens = 4_usize;
    let hidden = 3_usize;
    let intermediate = 2_usize;
    let num_experts = 3_usize;

    let tokens: Vec<f32> = vec![
        1.0, 0.0, 0.0,
        0.0, 1.0, 0.0,
        0.0, 0.0, 1.0,
        1.0, 1.0, 1.0,
    ];
    // Force routing: tokens 0+3 → expert 0, token 1 → expert 1, token 2 → expert 2.
    let logits: Vec<f32> = vec![
        5.0, 0.0, 0.0,
        0.0, 5.0, 0.0,
        0.0, 0.0, 5.0,
        5.0, 0.0, 0.0,
    ];
    // Distinct W_up per expert.
    let w_up: Vec<f32> = (0..num_experts * hidden * intermediate)
        .map(|i| {
            let e = i / (hidden * intermediate);
            0.1 * ((e + 1) as f32) + 0.01 * ((i % 7) as f32)
        })
        .collect();
    // Distinct W_down per expert.
    let w_down: Vec<f32> = (0..num_experts * intermediate * hidden)
        .map(|i| {
            let e = i / (intermediate * hidden);
            -0.2 * ((e + 1) as f32) + 0.03 * ((i % 5) as f32)
        })
        .collect();

    // Reference: route each token directly to its known expert.
    let routes: [usize; 4] = [0, 1, 2, 0];
    let mut expected: Vec<f32> = vec![0.0_f32; total_tokens * hidden];
    for (t, &e) in routes.iter().enumerate() {
        let tok = &tokens[t * hidden..(t + 1) * hidden];
        let wu = &w_up[e * hidden * intermediate..(e + 1) * hidden * intermediate];
        let wd = &w_down[e * intermediate * hidden..(e + 1) * intermediate * hidden];
        let out = ffn_expert_silu(tok, wu, wd, hidden, intermediate);
        // top_k=1 routing weight is exactly 1.0 (verified by v2.1 tests).
        for h in 0..hidden {
            expected[t * hidden + h] = out[h];
        }
    }

    let tokens_ptr = make_f32_tensor(&[total_tokens as i64, hidden as i64], &tokens);
    let logits_ptr = make_f32_tensor(&[total_tokens as i64, num_experts as i64], &logits);
    let w_up_ptr = make_f32_tensor(
        &[num_experts as i64, (hidden * intermediate) as i64], &w_up,
    );
    let w_down_ptr = make_f32_tensor(
        &[num_experts as i64, (intermediate * hidden) as i64], &w_down,
    );

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_ptr, logits_ptr,
            w_up_ptr, w_down_ptr,
            num_experts as i64,
            1,
            (2.0_f32).to_bits() as i64,
            hidden as i64,
            intermediate as i64,
            1,
        )
    };
    assert_ne!(out_ptr, 0);
    let got = read_f32(out_ptr, total_tokens * hidden);

    let mut max_abs = 0.0_f32;
    for (g, w) in got.iter().zip(expected.iter()) {
        let d = (g - w).abs();
        if d > max_abs { max_abs = d; }
    }
    assert!(
        max_abs <= 1e-6,
        "v3 top_k=1 diverges from per-token-per-expert reference: max_abs_diff={max_abs:.6e}"
    );

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v3_activation_kind_zero_is_identity_matmul_chain() {
    // activation_kind=0 means no activation. Then v3's "expert" is just
    // (W_up @ W_down). For W_up=identity + W_down=swap, the chain
    // collapses to: token @ identity = token, then token @ swap = swapped
    // token. We can verify this without any nonlinearity.
    let tokens: Vec<f32> = vec![
        1.0, 2.0,
        3.0, 4.0,
    ];
    let logits: Vec<f32> = vec![10.0, 0.0,  10.0, 0.0]; // both tokens → expert 0
    let w_up: Vec<f32> = vec![
        1.0, 0.0,  0.0, 1.0, // expert 0 = identity
        0.0, 0.0,  0.0, 0.0, // expert 1 = zeros (unused)
    ];
    let w_down: Vec<f32> = vec![
        0.0, 1.0,  1.0, 0.0, // expert 0 = swap
        0.0, 0.0,  0.0, 0.0,
    ];
    let expected: Vec<f32> = vec![
        2.0, 1.0,   // swap of [1,2]
        4.0, 3.0,   // swap of [3,4]
    ];

    let tokens_ptr = make_f32_tensor(&[2, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[2, 2], &logits);
    let w_up_ptr = make_f32_tensor(&[2, 4], &w_up);
    let w_down_ptr = make_f32_tensor(&[2, 4], &w_down);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr,
            2, 1, (2.0_f32).to_bits() as i64, 2, 2, 0, // activation_kind=0
        )
    };
    assert_ne!(out_ptr, 0);
    let got = read_f32(out_ptr, 4);
    for (g, w) in got.iter().zip(expected.iter()) {
        assert!((g - w).abs() < 1e-6, "got={got:?} want={expected:?}");
    }

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v3_refuses_invalid_inputs() {
    let tokens_ptr = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let logits_ptr = make_f32_tensor(&[2, 2], &[1.0, 0.5, 0.5, 1.0]);
    let w_up_ptr = make_f32_tensor(&[2, 4], &vec![0.1_f32; 8]);
    let w_down_ptr = make_f32_tensor(&[2, 4], &vec![0.1_f32; 8]);
    let cap = (2.0_f32).to_bits() as i64;

    // Null up.
    assert_eq!(unsafe {
        nsl_moe_dispatch_full_v3(tokens_ptr, logits_ptr, 0, w_down_ptr, 2, 1, cap, 2, 2, 1)
    }, 0, "must refuse null experts_up_ptr");
    // Null down.
    assert_eq!(unsafe {
        nsl_moe_dispatch_full_v3(tokens_ptr, logits_ptr, w_up_ptr, 0, 2, 1, cap, 2, 2, 1)
    }, 0, "must refuse null experts_down_ptr");
    // top_k out of range.
    assert_eq!(unsafe {
        nsl_moe_dispatch_full_v3(tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr, 2, 3, cap, 2, 2, 1)
    }, 0, "must refuse top_k=3");
    // Unknown activation kind.
    assert_eq!(unsafe {
        nsl_moe_dispatch_full_v3(tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr, 2, 1, cap, 2, 2, 2)
    }, 0, "must refuse activation_kind=2 (only 0 or 1 supported in v2.2)");
    // Wrong up shape.
    let bad_up = make_f32_tensor(&[10], &vec![0.5_f32; 10]);
    assert_eq!(unsafe {
        nsl_moe_dispatch_full_v3(tokens_ptr, logits_ptr, bad_up, w_down_ptr, 2, 1, cap, 2, 2, 1)
    }, 0, "must refuse wrong experts_up.len");
    // Wrong down shape.
    let bad_down = make_f32_tensor(&[10], &vec![0.5_f32; 10]);
    assert_eq!(unsafe {
        nsl_moe_dispatch_full_v3(tokens_ptr, logits_ptr, w_up_ptr, bad_down, 2, 1, cap, 2, 2, 1)
    }, 0, "must refuse wrong experts_down.len");

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(bad_up);
    nsl_tensor_free(bad_down);
}
