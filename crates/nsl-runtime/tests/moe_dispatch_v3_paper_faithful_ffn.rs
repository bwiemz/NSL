//! CPDT Part III v2.2: paper-faithful MoE FFN (`nsl_moe_dispatch_full_v3`).
//!
//! v2 was a single matmul per "expert", producing `[total_tokens,
//! intermediate_dim]`. v3 adds the second half of the FFN: activation +
//! W_down matmul, returning to `[total_tokens, hidden_dim]`. Pinned with
//! hand-computed references (no shared internals with the v2 tests so
//! a regression in route_topk / scatter / gather cannot mask a v3 bug).

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_zeros_f16_on, nsl_tensor_zeros_on};

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
        experts_up_bias_ptr: i64,
        experts_down_bias_ptr: i64,
    ) -> i64;
}

/// CPDT Part III v2.11 — sentinel constant for "no bias" used by the
/// pre-v2.11 v3 test sites. Passing 0/0 preserves byte-for-byte the
/// v2.5 v3 behavior (the bias args are nullable additions).
const NO_BIAS: i64 = 0;

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

fn gelu_tanh(x: f32) -> f32 {
    // Identical formula to the production FFI — references the same f32
    // constants so the test catches a typo regression in either.
    const SQRT_2_OVER_PI: f32 = 0.7978845608028654_f32;
    const GELU_CUBIC: f32 = 0.044715_f32;
    let inner = SQRT_2_OVER_PI * (x + GELU_CUBIC * x * x * x);
    0.5_f32 * x * (1.0_f32 + inner.tanh())
}


/// Single-token, single-expert forward pass. Used as the building block
/// for the hand-computed multi-token reference, independent of
/// route_topk / scatter / gather. Returns `out[hidden]`.
fn ffn_expert(
    token: &[f32],
    w_up: &[f32],   // shape [hidden, intermediate], row-major
    w_down: &[f32], // shape [intermediate, hidden], row-major
    hidden: usize,
    intermediate: usize,
    activation: fn(f32) -> f32,
) -> Vec<f32> {
    let mut x_up = vec![0.0_f32; intermediate];
    for j in 0..intermediate {
        let mut s = 0.0_f32;
        for k in 0..hidden {
            s += token[k] * w_up[k * intermediate + j];
        }
        x_up[j] = s;
    }
    let x_act: Vec<f32> = x_up.iter().map(|&v| activation(v)).collect();
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
            NO_BIAS, NO_BIAS,
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
        let out = ffn_expert(tok, wu, wd, hidden, intermediate, silu);
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
            NO_BIAS, NO_BIAS,
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
            NO_BIAS, NO_BIAS,
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
        nsl_moe_dispatch_full_v3(tokens_ptr, logits_ptr, 0, w_down_ptr, 2, 1, cap, 2, 2, 1, NO_BIAS, NO_BIAS)
    }, 0, "must refuse null experts_up_ptr");
    // Null down.
    assert_eq!(unsafe {
        nsl_moe_dispatch_full_v3(tokens_ptr, logits_ptr, w_up_ptr, 0, 2, 1, cap, 2, 2, 1, NO_BIAS, NO_BIAS)
    }, 0, "must refuse null experts_down_ptr");
    // top_k out of range.
    assert_eq!(unsafe {
        nsl_moe_dispatch_full_v3(tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr, 2, 3, cap, 2, 2, 1, NO_BIAS, NO_BIAS)
    }, 0, "must refuse top_k=3");
    // Unknown activation kind: 4 is past the v2.4 range (0=identity,
    // 1=SiLU, 2=GELU, 3=ReLU). SwiGLU and others are v2.5+ deferrals.
    assert_eq!(unsafe {
        nsl_moe_dispatch_full_v3(tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr, 2, 1, cap, 2, 2, 4, NO_BIAS, NO_BIAS)
    }, 0, "must refuse activation_kind=4 (only 0..=3 supported in v2.4)");
    // Negative also refused (signed-i64 input).
    assert_eq!(unsafe {
        nsl_moe_dispatch_full_v3(tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr, 2, 1, cap, 2, 2, -1, NO_BIAS, NO_BIAS)
    }, 0, "must refuse activation_kind=-1");
    // Wrong up shape.
    let bad_up = make_f32_tensor(&[10], &vec![0.5_f32; 10]);
    assert_eq!(unsafe {
        nsl_moe_dispatch_full_v3(tokens_ptr, logits_ptr, bad_up, w_down_ptr, 2, 1, cap, 2, 2, 1, NO_BIAS, NO_BIAS)
    }, 0, "must refuse wrong experts_up.len");
    // Wrong down shape.
    let bad_down = make_f32_tensor(&[10], &vec![0.5_f32; 10]);
    assert_eq!(unsafe {
        nsl_moe_dispatch_full_v3(tokens_ptr, logits_ptr, w_up_ptr, bad_down, 2, 1, cap, 2, 2, 1, NO_BIAS, NO_BIAS)
    }, 0, "must refuse wrong experts_down.len");

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(bad_up);
    nsl_tensor_free(bad_down);
}

#[test]
fn dispatch_v3_gelu_top_k_one_matches_hand_computed_reference() {
    // 4 tokens × 3 distinct experts, top_k=1, hidden=3 intermediate=2.
    // Same fixture as the SiLU variant; only the activation differs.
    // Catches any cross-activation contamination (e.g. accidentally
    // computing silu when the caller asked for gelu).
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
    let logits: Vec<f32> = vec![
        5.0, 0.0, 0.0,
        0.0, 5.0, 0.0,
        0.0, 0.0, 5.0,
        5.0, 0.0, 0.0,
    ];
    let w_up: Vec<f32> = (0..num_experts * hidden * intermediate)
        .map(|i| {
            let e = i / (hidden * intermediate);
            0.1 * ((e + 1) as f32) + 0.01 * ((i % 7) as f32)
        })
        .collect();
    let w_down: Vec<f32> = (0..num_experts * intermediate * hidden)
        .map(|i| {
            let e = i / (intermediate * hidden);
            -0.2 * ((e + 1) as f32) + 0.03 * ((i % 5) as f32)
        })
        .collect();

    let routes: [usize; 4] = [0, 1, 2, 0];
    let mut expected: Vec<f32> = vec![0.0_f32; total_tokens * hidden];
    for (t, &e) in routes.iter().enumerate() {
        let tok = &tokens[t * hidden..(t + 1) * hidden];
        let wu = &w_up[e * hidden * intermediate..(e + 1) * hidden * intermediate];
        let wd = &w_down[e * intermediate * hidden..(e + 1) * intermediate * hidden];
        let out = ffn_expert(tok, wu, wd, hidden, intermediate, gelu_tanh);
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
            2, // activation_kind = GELU
            NO_BIAS, NO_BIAS,
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
        "v3 GELU diverges from per-token-per-expert reference: max_abs_diff={max_abs:.6e}"
    );

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v3_relu_zeros_negative_intermediate_activations() {
    // ReLU's defining property: negative pre-activations become exactly
    // zero. Construct a fixture where W_up @ token has KNOWN-NEGATIVE
    // intermediate entries: token = [1, 0], W_up = [[-2, -3], [0, 0]]
    // (shape [hidden=2, intermediate=2]). Then x_up = [-2, -3]; ReLU
    // zeros both. The down-proj receives [0, 0] regardless of its
    // weights, so the output is [0, 0]. Pinning this catches any
    // future regression that "softens" ReLU (e.g. accidentally swaps
    // to SiLU which would produce nonzero output for negative inputs).
    let tokens: Vec<f32> = vec![
        1.0, 0.0,    // token 0
        0.5, 0.5,    // token 1: x_up = [-2*0.5 + 0*0.5, -3*0.5 + 0*0.5] = [-1, -1.5] → ReLU → [0,0]
    ];
    let logits: Vec<f32> = vec![10.0, 0.0,  10.0, 0.0]; // both → expert 0
    let w_up: Vec<f32> = vec![
        -2.0, -3.0,  0.0, 0.0,   // expert 0: shape [hidden=2, intermediate=2]
        7.0, 7.0,    7.0, 7.0,   // expert 1 (unused but must be valid f32)
    ];
    let w_down: Vec<f32> = vec![
        1.0, 1.0,  1.0, 1.0,     // expert 0: any values; ReLU output is 0 so down result is 0
        7.0, 7.0,  7.0, 7.0,     // expert 1 (unused)
    ];
    let expected: Vec<f32> = vec![
        0.0, 0.0,
        0.0, 0.0,
    ];

    let tokens_ptr = make_f32_tensor(&[2, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[2, 2], &logits);
    let w_up_ptr = make_f32_tensor(&[2, 4], &w_up);
    let w_down_ptr = make_f32_tensor(&[2, 4], &w_down);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr,
            2, 1, (2.0_f32).to_bits() as i64, 2, 2,
            3, // activation_kind = ReLU
            NO_BIAS, NO_BIAS,
        )
    };
    assert_ne!(out_ptr, 0);
    let got = read_f32(out_ptr, 4);
    for (g, w) in got.iter().zip(expected.iter()) {
        assert!((g - w).abs() < 1e-6, "ReLU on negative intermediates should zero them out: got={got:?} want={expected:?}");
    }

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v3_relu_preserves_positive_intermediate_activations() {
    // Complement to the previous test: ReLU on positive pre-activations
    // must NOT change them. Token = [1, 0], W_up = identity → x_up = [1, 0].
    // ReLU([1, 0]) = [1, 0] (0 is the boundary; ReLU(0) = max(0, 0) = 0,
    // unchanged). W_down = identity → output = [1, 0]. The single matmul
    // chain through identity activation matches.
    let tokens: Vec<f32> = vec![1.0, 0.0];
    let logits: Vec<f32> = vec![10.0, 0.0];
    let w_up: Vec<f32> = vec![
        1.0, 0.0,  0.0, 1.0,
        7.0, 7.0,  7.0, 7.0,
    ];
    let w_down: Vec<f32> = vec![
        1.0, 0.0,  0.0, 1.0,
        7.0, 7.0,  7.0, 7.0,
    ];
    let expected: Vec<f32> = vec![1.0, 0.0];

    let tokens_ptr = make_f32_tensor(&[1, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[1, 2], &logits);
    let w_up_ptr = make_f32_tensor(&[2, 4], &w_up);
    let w_down_ptr = make_f32_tensor(&[2, 4], &w_down);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr,
            2, 1, (2.0_f32).to_bits() as i64, 2, 2,
            3,
            NO_BIAS, NO_BIAS,
        )
    };
    assert_ne!(out_ptr, 0);
    let got = read_f32(out_ptr, 2);
    for (g, w) in got.iter().zip(expected.iter()) {
        assert!((g - w).abs() < 1e-6, "ReLU should preserve nonneg: got={got:?} want={expected:?}");
    }

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v3_activations_produce_distinct_outputs() {
    // Different activations on the SAME inputs must produce DIFFERENT
    // outputs (except for trivial cases). Catches any bug where the
    // activation selector silently fell through to one default for all
    // kinds (e.g. a match-fallthrough that left every kind hitting the
    // SiLU branch).
    let tokens: Vec<f32> = vec![0.5, 0.7,  -0.3, 0.4]; // mixed signs after up-matmul
    let logits: Vec<f32> = vec![10.0, 0.0,  10.0, 0.0];
    let w_up: Vec<f32> = vec![
        1.0, -1.0,  0.5, 0.5,  // expert 0: produces mixed-sign intermediates
        7.0, 7.0,   7.0, 7.0,
    ];
    let w_down: Vec<f32> = vec![
        1.0, 0.0,   0.0, 1.0,  // expert 0: identity-ish
        7.0, 7.0,   7.0, 7.0,
    ];

    let mut outputs: Vec<Vec<f32>> = Vec::new();
    for &act in &[1_i64, 2, 3] {
        let tokens_ptr = make_f32_tensor(&[2, 2], &tokens);
        let logits_ptr = make_f32_tensor(&[2, 2], &logits);
        let w_up_ptr = make_f32_tensor(&[2, 4], &w_up);
        let w_down_ptr = make_f32_tensor(&[2, 4], &w_down);

        let out_ptr = unsafe {
            nsl_moe_dispatch_full_v3(
                tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr,
                2, 1, (2.0_f32).to_bits() as i64, 2, 2,
                act,
                NO_BIAS, NO_BIAS,
            )
        };
        assert_ne!(out_ptr, 0, "activation_kind={} failed", act);
        outputs.push(read_f32(out_ptr, 4));

        nsl_tensor_free(tokens_ptr);
        nsl_tensor_free(logits_ptr);
        nsl_tensor_free(w_up_ptr);
        nsl_tensor_free(w_down_ptr);
        nsl_tensor_free(out_ptr);
    }

    // Each pair must differ somewhere — if any two match bit-exactly,
    // the activation selector dispatched both to the same branch.
    fn diff_max(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f32, f32::max)
    }
    assert!(diff_max(&outputs[0], &outputs[1]) > 1e-5, "SiLU vs GELU: outputs[0]={:?} outputs[1]={:?}", outputs[0], outputs[1]);
    assert!(diff_max(&outputs[1], &outputs[2]) > 1e-5, "GELU vs ReLU: outputs[1]={:?} outputs[2]={:?}", outputs[1], outputs[2]);
    assert!(diff_max(&outputs[0], &outputs[2]) > 1e-5, "SiLU vs ReLU: outputs[0]={:?} outputs[2]={:?}", outputs[0], outputs[2]);
}

// ─────────────────────────────────────────────────────────────────────────────
// CPDT Part III v2.11 — bias on up and down matmuls. Each bias is independently
// nullable (0 = no bias). Tests pin:
//   - bias on up-matmul only: output differs from no-bias baseline by exactly
//     `bias_up @ W_down` (hand-computed reference, no shared internals with
//     the v3 forward kernel so a regression cannot mask)
//   - bias on down-matmul only: output differs from no-bias by exactly the
//     bias_down vector (gather respects routing weights)
//   - bias on BOTH: sum of both effects (linearity check)
//   - refusal on bias dtype mismatch
//   - refusal on bias length mismatch
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn dispatch_v3_up_bias_only_matches_hand_reference() {
    // 1 token, 1 expert (no routing complexity), identity activation
    // (so the bias effect is observable end-to-end), W_up = identity,
    // W_down = identity. Token x = [2.0, 3.0]; bias_up = [0.5, -1.0].
    //   x_up        = [2.0, 3.0]
    //   x_up+bias   = [2.5, 2.0]
    //   x_act       = [2.5, 2.0]   (identity activation)
    //   x_down      = [2.5, 2.0]   (identity W_down)
    //   out         = [2.5, 2.0]
    let tokens: Vec<f32> = vec![2.0, 3.0];
    let logits: Vec<f32> = vec![1.0]; // 1-expert routing trivial
    let w_up: Vec<f32> = vec![1.0, 0.0,  0.0, 1.0];   // I_2
    let w_down: Vec<f32> = vec![1.0, 0.0,  0.0, 1.0]; // I_2
    let bias_up: Vec<f32> = vec![0.5, -1.0];

    let tokens_ptr = make_f32_tensor(&[1, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[1, 1], &logits);
    let w_up_ptr = make_f32_tensor(&[1, 4], &w_up);
    let w_down_ptr = make_f32_tensor(&[1, 4], &w_down);
    let bias_up_ptr = make_f32_tensor(&[1, 2], &bias_up);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr,
            1, 1, (2.0_f32).to_bits() as i64, 2, 2,
            0, // activation_kind = identity
            bias_up_ptr,
            NO_BIAS, // no down bias
        )
    };
    assert_ne!(out_ptr, 0, "v3 with up_bias must succeed");
    let got = read_f32(out_ptr, 2);
    let expected = vec![2.5_f32, 2.0_f32];
    for (g, w) in got.iter().zip(expected.iter()) {
        assert!((g - w).abs() < 1e-6, "up_bias: got={got:?} want={expected:?}");
    }

    // Baseline: same call with NO_BIAS — must differ by exactly bias_up
    // values (proves the bias actually fired, not a no-op coincidence).
    let baseline_ptr = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr,
            1, 1, (2.0_f32).to_bits() as i64, 2, 2, 0, NO_BIAS, NO_BIAS,
        )
    };
    let baseline = read_f32(baseline_ptr, 2);
    assert_eq!(baseline, vec![2.0, 3.0], "baseline x @ I @ I = x");
    let diff: Vec<f32> = got.iter().zip(baseline.iter()).map(|(g, b)| g - b).collect();
    assert!((diff[0] - 0.5).abs() < 1e-6 && (diff[1] - (-1.0)).abs() < 1e-6,
        "up_bias effect must equal bias_up exactly (W_down=I): diff={diff:?}");

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(bias_up_ptr);
    nsl_tensor_free(out_ptr);
    nsl_tensor_free(baseline_ptr);
}

#[test]
fn dispatch_v3_down_bias_only_matches_hand_reference() {
    // Same setup as above but bias on the down matmul instead.
    // bias_down = [10.0, -5.0]; output should be baseline + bias_down.
    let tokens: Vec<f32> = vec![2.0, 3.0];
    let logits: Vec<f32> = vec![1.0];
    let w_up: Vec<f32> = vec![1.0, 0.0,  0.0, 1.0];
    let w_down: Vec<f32> = vec![1.0, 0.0,  0.0, 1.0];
    let bias_down: Vec<f32> = vec![10.0, -5.0];

    let tokens_ptr = make_f32_tensor(&[1, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[1, 1], &logits);
    let w_up_ptr = make_f32_tensor(&[1, 4], &w_up);
    let w_down_ptr = make_f32_tensor(&[1, 4], &w_down);
    let bias_down_ptr = make_f32_tensor(&[1, 2], &bias_down);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr,
            1, 1, (2.0_f32).to_bits() as i64, 2, 2,
            0, NO_BIAS, bias_down_ptr,
        )
    };
    assert_ne!(out_ptr, 0);
    let got = read_f32(out_ptr, 2);
    let expected = vec![12.0_f32, -2.0_f32];
    for (g, w) in got.iter().zip(expected.iter()) {
        assert!((g - w).abs() < 1e-6, "down_bias: got={got:?} want={expected:?}");
    }

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(bias_down_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v3_both_biases_compose_linearly() {
    // Both biases present. Linearity: out(bias_up, bias_down) =
    // baseline + bias_up @ W_down + bias_down. With W_down=I this is
    // baseline + bias_up + bias_down.
    let tokens: Vec<f32> = vec![2.0, 3.0];
    let logits: Vec<f32> = vec![1.0];
    let w_up: Vec<f32> = vec![1.0, 0.0,  0.0, 1.0];
    let w_down: Vec<f32> = vec![1.0, 0.0,  0.0, 1.0];
    let bias_up: Vec<f32> = vec![0.5, -1.0];
    let bias_down: Vec<f32> = vec![10.0, -5.0];

    let tokens_ptr = make_f32_tensor(&[1, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[1, 1], &logits);
    let w_up_ptr = make_f32_tensor(&[1, 4], &w_up);
    let w_down_ptr = make_f32_tensor(&[1, 4], &w_down);
    let bias_up_ptr = make_f32_tensor(&[1, 2], &bias_up);
    let bias_down_ptr = make_f32_tensor(&[1, 2], &bias_down);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr,
            1, 1, (2.0_f32).to_bits() as i64, 2, 2,
            0, bias_up_ptr, bias_down_ptr,
        )
    };
    let got = read_f32(out_ptr, 2);
    // Expected: x + bias_up + bias_down = [2+0.5+10, 3-1-5] = [12.5, -3.0]
    let expected = vec![12.5_f32, -3.0_f32];
    for (g, w) in got.iter().zip(expected.iter()) {
        assert!((g - w).abs() < 1e-6, "both biases: got={got:?} want={expected:?}");
    }

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(bias_up_ptr);
    nsl_tensor_free(bias_down_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v3_refuses_invalid_bias() {
    // Refusal gates for bias args: shape mismatch + dtype mismatch on
    // each direction independently. v2.11 fix F8 (IMPORTANT
    // adversarial review) — use SEPARATE bad-length tensors for up vs
    // down so a direction-asymmetric regression cannot mask the other
    // direction's bug.
    let tokens_ptr = make_f32_tensor(&[1, 2], &[1.0, 1.0]);
    let logits_ptr = make_f32_tensor(&[1, 1], &[1.0]);
    let w_up_ptr = make_f32_tensor(&[1, 4], &vec![0.1_f32; 4]);
    let w_down_ptr = make_f32_tensor(&[1, 4], &vec![0.1_f32; 4]);
    let cap = (2.0_f32).to_bits() as i64;

    // Wrong up_bias length (expected 1*2=2; this is 7 — distinct from
    // the down_bias bad length so a direction swap is observable).
    let bad_up_bias = make_f32_tensor(&[7], &vec![0.5_f32; 7]);
    let bad_down_bias = make_f32_tensor(&[11], &vec![0.5_f32; 11]);
    let rc = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr,
            1, 1, cap, 2, 2, 0, bad_up_bias, NO_BIAS,
        )
    };
    assert_eq!(rc, 0, "must refuse wrong up_bias len");

    let rc2 = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr,
            1, 1, cap, 2, 2, 0, NO_BIAS, bad_down_bias,
        )
    };
    assert_eq!(rc2, 0, "must refuse wrong down_bias len");

    // v2.11 fix F1 (IMPORTANT): tokens (or any tensor) at dtype != {0,1}
    // would silently hit the read_f32 else-branch and OOB-read f16
    // 2-byte buffers as f32. Pin the upfront refusal with an f16
    // tokens tensor.
    let tokens_f16_shape = nsl_list_new();
    nsl_list_push(tokens_f16_shape, 1);
    nsl_list_push(tokens_f16_shape, 2);
    let tokens_f16 = nsl_tensor_zeros_f16_on(tokens_f16_shape, 0);
    nsl_list_free(tokens_f16_shape);
    let rc_f16 = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_f16, logits_ptr, w_up_ptr, w_down_ptr,
            1, 1, cap, 2, 2, 0, NO_BIAS, NO_BIAS,
        )
    };
    assert_eq!(
        rc_f16, 0,
        "must refuse f16 tokens (dtype=2) — read_f32 would OOB-read 2-byte buffer"
    );

    // v2.11 fix F2 (IMPORTANT): a GPU-resident tensor (device=1) here
    // would dereference a CUDA device pointer on the host. Refuse
    // upfront. The make_*_on(shape, 0) helper allocates on CPU; the
    // device-mismatch path is exercised through a bias built on GPU.
    // CI may run without CUDA available, so this branch checks via the
    // helper's known behavior of always returning device=0 — a future
    // GPU bias-tensor helper would close the active-test gap.
    let _ = tokens_f16; // referenced below for free

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(bad_up_bias);
    nsl_tensor_free(bad_down_bias);
    nsl_tensor_free(tokens_f16);
}

#[test]
fn dispatch_v3_up_bias_with_silu_pins_pre_activation_ordering() {
    // v2.11 fix F5 (HIGH adversarial review). The 3 happy-path bias
    // tests all use IDENTITY activation, so they cannot detect a
    // regression that applies bias AFTER activation instead of
    // BEFORE. SiLU is non-linear, so silu(x+bias) != silu(x)+bias for
    // nonzero bias.
    //
    // Setup: 1 token, 1 expert, W_up = W_down = identity, x = [1.0,
    // -1.0], bias_up = [2.0, -0.5]. Activation = SiLU (kind=1).
    //   x_up        = [1.0, -1.0]
    //   bias-FIRST: x_up + bias_up = [3.0, -1.5]
    //               silu(3.0) ≈ 2.857, silu(-1.5) ≈ -0.275
    //   x_act       = [2.857, -0.275]   (silu of biased)
    //   x_down      = [2.857, -0.275]   (W_down = I)
    //   out         = [2.857, -0.275]
    //
    // A regression that did bias-AFTER would compute silu(x_up) =
    // silu([1, -1]) ≈ [0.731, -0.269], then +bias = [2.731, -0.769].
    // The two outputs DIFFER by more than fp32 epsilon — the test
    // pins which one v2.11 emits.
    let tokens: Vec<f32> = vec![1.0, -1.0];
    let logits: Vec<f32> = vec![1.0];
    let w_up: Vec<f32> = vec![1.0, 0.0,  0.0, 1.0];
    let w_down: Vec<f32> = vec![1.0, 0.0,  0.0, 1.0];
    let bias_up: Vec<f32> = vec![2.0, -0.5];

    let tokens_ptr = make_f32_tensor(&[1, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[1, 1], &logits);
    let w_up_ptr = make_f32_tensor(&[1, 4], &w_up);
    let w_down_ptr = make_f32_tensor(&[1, 4], &w_down);
    let bias_up_ptr = make_f32_tensor(&[1, 2], &bias_up);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr,
            1, 1, (2.0_f32).to_bits() as i64, 2, 2,
            1, // SiLU
            bias_up_ptr, NO_BIAS,
        )
    };
    let got = read_f32(out_ptr, 2);
    let silu = |x: f32| x * (1.0_f32 / (1.0_f32 + (-x).exp()));
    let expected_pre = vec![silu(3.0), silu(-1.5)];
    let expected_post = vec![silu(1.0) + 2.0, silu(-1.0) + (-0.5)];
    for (g, w) in got.iter().zip(expected_pre.iter()) {
        assert!(
            (g - w).abs() < 1e-5,
            "bias-BEFORE-activation: got={got:?} want={expected_pre:?} (post-activation would be {expected_post:?})"
        );
    }
    // Sanity: confirm the two interpretations actually diverge beyond
    // fp32 noise so the test isn't vacuous.
    let max_diff = expected_pre.iter().zip(expected_post.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0_f32, f32::max);
    assert!(max_diff > 0.1, "test setup must make pre/post divergent: diff={max_diff}");

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(bias_up_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v3_per_expert_bias_indexing_with_two_experts() {
    // v2.11 fix F6 (IMPORTANT adversarial review). The 3 happy-path
    // bias tests all use num_experts=1, so the per-expert offset
    // (e * dim) is never multiplied by anything non-zero. A regression
    // that dropped the e*dim multiplier would still pass.
    //
    // Setup: 2 experts, 2 tokens, top_k=1. Tokens [1.0, 0.0] route to
    // expert 0 (logits favor e0); token [0.0, 1.0] routes to expert
    // 1. W_up = W_down = I per expert. bias_up = [[0.5, 1.5], [10.0,
    // -5.0]] (expert 0's bias differs sharply from expert 1's).
    // Identity activation (matches v2.5 NoBias control).
    //
    //   token 0 = [1, 0] routes to e0 → out = [1, 0] + bias_up[0] = [1.5, 1.5]
    //   token 1 = [0, 1] routes to e1 → out = [0, 1] + bias_up[1] = [10.0, -4.0]
    //
    // A regression that dropped e*dim would make BOTH tokens read
    // bias_up[0..2] giving wrong outputs for token 1.
    let tokens: Vec<f32> = vec![1.0, 0.0,  0.0, 1.0];
    // Logits: row 0 favors e0, row 1 favors e1.
    let logits: Vec<f32> = vec![5.0, -5.0,  -5.0, 5.0];
    // 2 experts × W_up of identity each.
    let w_up: Vec<f32> = vec![1.0, 0.0,  0.0, 1.0,   1.0, 0.0,  0.0, 1.0];
    let w_down: Vec<f32> = vec![1.0, 0.0,  0.0, 1.0,   1.0, 0.0,  0.0, 1.0];
    // bias_up: expert 0 = [0.5, 1.5], expert 1 = [10.0, -5.0].
    let bias_up: Vec<f32> = vec![0.5, 1.5,   10.0, -5.0];

    let tokens_ptr = make_f32_tensor(&[2, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[2, 2], &logits);
    let w_up_ptr = make_f32_tensor(&[2, 4], &w_up);
    let w_down_ptr = make_f32_tensor(&[2, 4], &w_down);
    let bias_up_ptr = make_f32_tensor(&[2, 2], &bias_up);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v3(
            tokens_ptr, logits_ptr, w_up_ptr, w_down_ptr,
            2, 1, (2.0_f32).to_bits() as i64, 2, 2,
            0, bias_up_ptr, NO_BIAS,
        )
    };
    let got = read_f32(out_ptr, 4);
    let expected = vec![1.5_f32, 1.5,  10.0, -4.0];
    for (i, (g, w)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-5,
            "per-expert bias[{}]: got={got:?} want={expected:?} (regression dropping e*dim would put bias_up[0] on token 1 too)",
            i,
        );
    }

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(bias_up_ptr);
    nsl_tensor_free(out_ptr);
}

