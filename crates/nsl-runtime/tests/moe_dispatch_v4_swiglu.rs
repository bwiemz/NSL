//! CPDT Part III v2.5: Mixtral SwiGLU MoE FFN
//! (`nsl_moe_dispatch_full_v4`).
//!
//! v3 was up→activation→down with one weight per direction. v4 adds a
//! THIRD weight matrix (gate) and an element-wise multiply step before
//! the down-projection:
//!
//!   x_act = silu(token @ W_gate) * (token @ W_up)
//!   out   = x_act @ W_down
//!
//! Tests pin the contract against multiple INDEPENDENT references so a
//! single regression in route_topk / scatter / gather can't mask a v4
//! bug across all tests.

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_zeros_f16_on, nsl_tensor_zeros_on,
};

extern "C" {
    fn nsl_moe_dispatch_full_v4(
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
        gate_activation_kind: i64,
        // v2.14: 3 nullable bias pointers (NO_BIAS = 0 → no bias for
        // that direction). Existing pre-v2.14 call sites pass
        // NO_BIAS, NO_BIAS, NO_BIAS to preserve byte-identical
        // behavior.
        experts_gate_bias_ptr: i64,
        experts_up_bias_ptr: i64,
        experts_down_bias_ptr: i64,
    ) -> i64;
}

/// v2.8 SwiGLU constant — preserves byte-for-byte the v2.5/v2.7
/// behavior in this test file (which all predate gate_activation_kind).
const SWIGLU: i64 = 1;

/// v2.14 no-bias sentinel — preserves byte-for-byte the v2.5/v2.7/v2.8
/// behavior for pre-v2.14 v4 call sites.
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

/// Hand-rolled per-token single-expert SwiGLU forward. Independent of
/// route_topk / scatter / gather so a routing-pipeline regression
/// cannot silently corrupt this reference.
fn swiglu_expert(
    token: &[f32],
    w_gate: &[f32], // [hidden, intermediate] row-major
    w_up: &[f32],   // [hidden, intermediate] row-major
    w_down: &[f32], // [intermediate, hidden] row-major
    hidden: usize,
    intermediate: usize,
) -> Vec<f32> {
    let mut x_gate = vec![0.0_f32; intermediate];
    let mut x_up = vec![0.0_f32; intermediate];
    for j in 0..intermediate {
        let mut sg = 0.0_f32;
        let mut su = 0.0_f32;
        for k in 0..hidden {
            sg += token[k] * w_gate[k * intermediate + j];
            su += token[k] * w_up[k * intermediate + j];
        }
        x_gate[j] = sg;
        x_up[j] = su;
    }
    let x_act: Vec<f32> = x_gate
        .iter()
        .zip(x_up.iter())
        .map(|(&g, &u)| silu(g) * u)
        .collect();
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

// ─────────────────────────────────────────────────────────────────────
// TEST 1 — Top-k=1, deterministic routing per token, hand-computed
// reference via swiglu_expert. The strongest correctness gate because
// it exercises every matmul + the GLU multiply + the SiLU.
// ─────────────────────────────────────────────────────────────────────
#[test]
fn dispatch_v4_top_k_one_distinct_experts_matches_reference() {
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
    // Force routing: tokens 0+3 -> expert 0, token 1 -> expert 1, token 2 -> expert 2.
    let logits: Vec<f32> = vec![
        5.0, 0.0, 0.0,
        0.0, 5.0, 0.0,
        0.0, 0.0, 5.0,
        5.0, 0.0, 0.0,
    ];
    let w_gate: Vec<f32> = (0..num_experts * hidden * intermediate)
        .map(|i| {
            let e = i / (hidden * intermediate);
            0.07 * ((e + 1) as f32) + 0.013 * ((i % 7) as f32)
        })
        .collect();
    let w_up: Vec<f32> = (0..num_experts * hidden * intermediate)
        .map(|i| {
            let e = i / (hidden * intermediate);
            0.11 * ((e + 1) as f32) - 0.017 * ((i % 5) as f32)
        })
        .collect();
    let w_down: Vec<f32> = (0..num_experts * intermediate * hidden)
        .map(|i| {
            let e = i / (intermediate * hidden);
            -0.19 * ((e + 1) as f32) + 0.029 * ((i % 5) as f32)
        })
        .collect();

    let routes: [usize; 4] = [0, 1, 2, 0];
    let mut expected: Vec<f32> = vec![0.0_f32; total_tokens * hidden];
    for (t, &e) in routes.iter().enumerate() {
        let tok = &tokens[t * hidden..(t + 1) * hidden];
        let wg = &w_gate[e * hidden * intermediate..(e + 1) * hidden * intermediate];
        let wu = &w_up[e * hidden * intermediate..(e + 1) * hidden * intermediate];
        let wd = &w_down[e * intermediate * hidden..(e + 1) * intermediate * hidden];
        let out = swiglu_expert(tok, wg, wu, wd, hidden, intermediate);
        for h in 0..hidden {
            expected[t * hidden + h] = out[h];
        }
    }

    let tokens_ptr = make_f32_tensor(&[total_tokens as i64, hidden as i64], &tokens);
    let logits_ptr = make_f32_tensor(&[total_tokens as i64, num_experts as i64], &logits);
    let w_gate_ptr = make_f32_tensor(
        &[num_experts as i64, (hidden * intermediate) as i64],
        &w_gate,
    );
    let w_up_ptr = make_f32_tensor(
        &[num_experts as i64, (hidden * intermediate) as i64],
        &w_up,
    );
    let w_down_ptr = make_f32_tensor(
        &[num_experts as i64, (intermediate * hidden) as i64],
        &w_down,
    );

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr,
            logits_ptr,
            w_gate_ptr,
            w_up_ptr,
            w_down_ptr,
            num_experts as i64,
            1,
            (2.0_f32).to_bits() as i64,
            hidden as i64,
            intermediate as i64,
            SWIGLU,
            NO_BIAS, NO_BIAS, NO_BIAS,
        )
    };
    assert_ne!(out_ptr, 0, "v4 returned null for valid SwiGLU inputs");
    let got = read_f32(out_ptr, total_tokens * hidden);

    let mut max_abs = 0.0_f32;
    for (g, w) in got.iter().zip(expected.iter()) {
        let d = (g - w).abs();
        if d > max_abs {
            max_abs = d;
        }
    }
    assert!(
        max_abs <= 1e-6,
        "v4 SwiGLU diverges from per-token-per-expert reference: max_abs_diff={max_abs:.6e}"
    );

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_gate_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(out_ptr);
}

// ─────────────────────────────────────────────────────────────────────
// TEST 2 — top_k=2 with a hand-derived 2-token / 2-expert / hidden=2
// case. Reference computed by closed-form scalar math (NOT by
// swiglu_expert), so it catches a bug that contaminates both v4 and
// the helper. The symmetric setup makes the expected output derivable
// by inspection.
// ─────────────────────────────────────────────────────────────────────
#[test]
fn dispatch_v4_top_k_two_symmetric_closed_form() {
    // Token 0 = [1, 0], token 1 = [0, 1].
    // Logits engineered so token 0 goes 0.75/0.25 to experts 0/1 and
    // token 1 goes 0.25/0.75 — same logits as v3's top_k=2 test.
    let ln3 = 3.0_f32.ln();
    let tokens: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let logits: Vec<f32> = vec![ln3, 0.0, 0.0, ln3];

    // Expert 0: W_gate = identity, W_up = identity, W_down = identity.
    // Expert 1: same as expert 0 — symmetric collapse means both
    // experts produce the same output per token, so the weighted sum
    // collapses to a single closed-form expression.
    let w_gate: Vec<f32> = vec![
        1.0, 0.0,  0.0, 1.0,
        1.0, 0.0,  0.0, 1.0,
    ];
    let w_up: Vec<f32> = vec![
        1.0, 0.0,  0.0, 1.0,
        1.0, 0.0,  0.0, 1.0,
    ];
    let w_down: Vec<f32> = vec![
        1.0, 0.0,  0.0, 1.0,
        1.0, 0.0,  0.0, 1.0,
    ];

    // Hand derivation for token 0 = [1, 0] under either expert:
    //   x_gate = [1, 0]
    //   x_up   = [1, 0]
    //   x_act  = [silu(1) * 1, silu(0) * 0] = [silu(1), 0]
    //   x_down = [silu(1), 0]
    // Both experts agree → weighted sum (0.75 + 0.25) * [silu(1), 0]
    //                    = [silu(1), 0].
    //
    // Token 1 = [0, 1]:
    //   x_gate = [0, 1]
    //   x_up   = [0, 1]
    //   x_act  = [silu(0) * 0, silu(1) * 1] = [0, silu(1)]
    //   x_down = [0, silu(1)]
    let s1 = silu(1.0_f32);
    let expected: Vec<f32> = vec![s1, 0.0, 0.0, s1];

    let tokens_ptr = make_f32_tensor(&[2, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[2, 2], &logits);
    let w_gate_ptr = make_f32_tensor(&[2, 4], &w_gate);
    let w_up_ptr = make_f32_tensor(&[2, 4], &w_up);
    let w_down_ptr = make_f32_tensor(&[2, 4], &w_down);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr,
            w_gate_ptr, w_up_ptr, w_down_ptr,
            2,
            2,
            (2.0_f32).to_bits() as i64,
            2,
            2,
            SWIGLU,
            NO_BIAS, NO_BIAS, NO_BIAS,
        )
    };
    assert_ne!(out_ptr, 0);
    let got = read_f32(out_ptr, 4);

    let mut max_abs = 0.0_f32;
    for (g, w) in got.iter().zip(expected.iter()) {
        let d = (g - w).abs();
        if d > max_abs { max_abs = d; }
    }
    assert!(
        max_abs <= 1e-6,
        "top_k=2 symmetric closed-form: got={got:?} want={expected:?} max_abs_diff={max_abs:.6e}"
    );

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_gate_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(out_ptr);
}

// ─────────────────────────────────────────────────────────────────────
// TEST 3 — GLU contract: gate-zero kills the contribution.
// When x_gate is all zeros, silu(0) = 0, so x_act = 0 * x_up = 0, so
// output must be exactly zero regardless of W_up or W_down or token.
// This pins the multiply step independently of the matmul + SiLU.
// ─────────────────────────────────────────────────────────────────────
#[test]
fn dispatch_v4_zero_gate_kills_expert_contribution_exactly() {
    let tokens: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2 tokens × hidden=3
    let logits: Vec<f32> = vec![10.0, 0.0, 10.0, 0.0]; // both → expert 0
    let w_gate: Vec<f32> = vec![
        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // expert 0: ALL ZEROS → silu(0) = 0
        7.0, 7.0, 7.0, 7.0, 7.0, 7.0, // expert 1: nonzero but unused
    ];
    // W_up and W_down completely arbitrary nonzero values — they must
    // not influence the output because the GLU multiply gates them off.
    let w_up: Vec<f32> = vec![
        17.0, 23.0, 29.0, 31.0, 37.0, 41.0,
        43.0, 47.0, 53.0, 59.0, 61.0, 67.0,
    ];
    let w_down: Vec<f32> = vec![
        71.0, 73.0, 79.0, 83.0, 89.0, 97.0,
        101.0, 103.0, 107.0, 109.0, 113.0, 127.0,
    ];
    let expected: Vec<f32> = vec![0.0; 6];

    let tokens_ptr = make_f32_tensor(&[2, 3], &tokens);
    let logits_ptr = make_f32_tensor(&[2, 2], &logits);
    let w_gate_ptr = make_f32_tensor(&[2, 6], &w_gate);
    let w_up_ptr = make_f32_tensor(&[2, 6], &w_up);
    let w_down_ptr = make_f32_tensor(&[2, 6], &w_down);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr,
            w_gate_ptr, w_up_ptr, w_down_ptr,
            2, 1, (2.0_f32).to_bits() as i64, 3, 2, SWIGLU,
            NO_BIAS, NO_BIAS, NO_BIAS,
        )
    };
    assert_ne!(out_ptr, 0);
    let got = read_f32(out_ptr, 6);
    for (g, w) in got.iter().zip(expected.iter()) {
        assert!(
            (g - w).abs() < 1e-6,
            "GLU contract violated: zero gate should kill output. got={got:?} want=zeros"
        );
    }

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_gate_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(out_ptr);
}

// ─────────────────────────────────────────────────────────────────────
// TEST 4 — Distinguishability: SwiGLU output must NOT equal v3 SiLU
// output on the same inputs (with W_gate = W_up). If it does, the gate
// matmul is being ignored.
// ─────────────────────────────────────────────────────────────────────
#[test]
fn dispatch_v4_swiglu_differs_from_silu_when_gate_equals_up() {
    // Inputs chosen so x_up is nonzero AND silu(x_up) != x_up (the
    // case where SwiGLU = silu(x) * x diverges from plain SiLU(x)).
    let tokens: Vec<f32> = vec![1.5, 2.5];
    let logits: Vec<f32> = vec![10.0, 0.0]; // → expert 0
    let w: Vec<f32> = vec![
        1.0, 0.0,  0.0, 1.0,        // expert 0: identity 2×2
        7.0, 7.0,  7.0, 7.0,        // expert 1: unused
    ];
    let w_down: Vec<f32> = vec![
        1.0, 0.0,  0.0, 1.0,
        7.0, 7.0,  7.0, 7.0,
    ];

    let tokens_ptr = make_f32_tensor(&[1, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[1, 2], &logits);
    let w_gate_ptr = make_f32_tensor(&[2, 4], &w);
    let w_up_ptr = make_f32_tensor(&[2, 4], &w);
    let w_down_ptr = make_f32_tensor(&[2, 4], &w_down);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr,
            w_gate_ptr, w_up_ptr, w_down_ptr,
            2, 1, (2.0_f32).to_bits() as i64, 2, 2, SWIGLU,
            NO_BIAS, NO_BIAS, NO_BIAS,
        )
    };
    assert_ne!(out_ptr, 0);
    let got = read_f32(out_ptr, 2);

    // With W_gate = W_up = identity and W_down = identity, token [1.5, 2.5]:
    //   x_gate = [1.5, 2.5], x_up = [1.5, 2.5]
    //   x_act  = [silu(1.5) * 1.5, silu(2.5) * 2.5]
    //   x_down = x_act (identity W_down)
    let expected: Vec<f32> = vec![
        silu(1.5_f32) * 1.5_f32,
        silu(2.5_f32) * 2.5_f32,
    ];
    let mut max_abs = 0.0_f32;
    for (g, w) in got.iter().zip(expected.iter()) {
        let d = (g - w).abs();
        if d > max_abs { max_abs = d; }
    }
    assert!(max_abs <= 1e-6, "SwiGLU formula error: got={got:?} want={expected:?}");

    // The plain-SiLU value for the same input would be just silu(1.5)
    // (≈1.2256) and silu(2.5) (≈2.2818). SwiGLU multiplies these by
    // x_up — so the SwiGLU output should be larger than plain SiLU.
    let silu_only: Vec<f32> = vec![silu(1.5_f32), silu(2.5_f32)];
    assert!(
        got.iter().zip(silu_only.iter()).any(|(g, s)| (g - s).abs() > 0.1),
        "SwiGLU output must clearly differ from plain SiLU when gate equals up"
    );

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_gate_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(out_ptr);
}

// ─────────────────────────────────────────────────────────────────────
// TEST 5 — Refusal contract. Each of the 6 refusal gates exercised in
// isolation. A future regression that drops a refusal (e.g. someone
// removes the null-gate check) would let the FFI dereference a null
// pointer.
// ─────────────────────────────────────────────────────────────────────
#[test]
fn dispatch_v4_refuses_invalid_inputs() {
    let tokens_ptr = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let logits_ptr = make_f32_tensor(&[2, 2], &[1.0, 0.5, 0.5, 1.0]);
    let w_gate_ptr = make_f32_tensor(&[2, 4], &vec![0.1_f32; 8]);
    let w_up_ptr = make_f32_tensor(&[2, 4], &vec![0.1_f32; 8]);
    let w_down_ptr = make_f32_tensor(&[2, 4], &vec![0.1_f32; 8]);
    let cap = (2.0_f32).to_bits() as i64;

    let v = |gate, up, down, ne, tk, h, i| unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr, gate, up, down, ne, tk, cap, h, i, SWIGLU,
            NO_BIAS, NO_BIAS, NO_BIAS,
        )
    };

    assert_eq!(v(0, w_up_ptr, w_down_ptr, 2, 1, 2, 2), 0, "null gate");
    assert_eq!(v(w_gate_ptr, 0, w_down_ptr, 2, 1, 2, 2), 0, "null up");
    assert_eq!(v(w_gate_ptr, w_up_ptr, 0, 2, 1, 2, 2), 0, "null down");
    assert_eq!(v(w_gate_ptr, w_up_ptr, w_down_ptr, 0, 1, 2, 2), 0, "n_experts=0");
    assert_eq!(v(w_gate_ptr, w_up_ptr, w_down_ptr, 2, 1, 0, 2), 0, "hidden=0");
    assert_eq!(v(w_gate_ptr, w_up_ptr, w_down_ptr, 2, 1, 2, 0), 0, "intermediate=0");
    assert_eq!(v(w_gate_ptr, w_up_ptr, w_down_ptr, 2, 3, 2, 2), 0, "top_k=3");
    assert_eq!(v(w_gate_ptr, w_up_ptr, w_down_ptr, 2, 0, 2, 2), 0, "top_k=0");

    // Wrong gate shape: 10 elements instead of expected 8.
    let bad = make_f32_tensor(&[10], &vec![0.5_f32; 10]);
    assert_eq!(v(bad, w_up_ptr, w_down_ptr, 2, 1, 2, 2), 0, "wrong gate.len");
    assert_eq!(v(w_gate_ptr, bad, w_down_ptr, 2, 1, 2, 2), 0, "wrong up.len");
    assert_eq!(v(w_gate_ptr, w_up_ptr, bad, 2, 1, 2, 2), 0, "wrong down.len");

    // Null tokens / null logits — added per v2.5 adversarial review:
    // the v4 FFI is `#[no_mangle] pub extern "C"`, so a third-party
    // wrapper or direct C ABI caller can pass 0 for either pointer.
    // NslTensor::from_ptr would form a `&mut NslTensor` from null,
    // which is instant UB. The refusal must precede from_ptr.
    let v_null = |t, l| unsafe {
        nsl_moe_dispatch_full_v4(
            t, l, w_gate_ptr, w_up_ptr, w_down_ptr, 2, 1, cap, 2, 2, SWIGLU,
            NO_BIAS, NO_BIAS, NO_BIAS,
        )
    };
    assert_eq!(v_null(0, logits_ptr), 0, "null tokens");
    assert_eq!(v_null(tokens_ptr, 0), 0, "null logits");
    assert_eq!(v_null(0, 0), 0, "both null");

    // Dtype-mismatch refusals — pin the FFI guard that v2.5 review
    // flagged as untested. The guard rejects when tokens.dtype !=
    // experts_*.dtype for ANY of the three expert tensors. We
    // construct fp16-tagged expert tensors (dtype != f32) and
    // verify each of the three checks fires independently. A future
    // refactor that drops one of the three checks would regress one
    // of these assertions — uniform-f32 tests don't cover this
    // surface at all.
    let make_f16_zeros = |shape: &[i64]| -> i64 {
        let shape_list = nsl_list_new();
        for &d in shape {
            nsl_list_push(shape_list, d);
        }
        let ptr = nsl_tensor_zeros_f16_on(shape_list, 0);
        nsl_list_free(shape_list);
        ptr
    };
    let w_gate_fp16 = make_f16_zeros(&[2, 4]);
    let w_up_fp16 = make_f16_zeros(&[2, 4]);
    let w_down_fp16 = make_f16_zeros(&[2, 4]);

    assert_eq!(
        v(w_gate_fp16, w_up_ptr, w_down_ptr, 2, 1, 2, 2),
        0,
        "gate dtype != tokens dtype"
    );
    assert_eq!(
        v(w_gate_ptr, w_up_fp16, w_down_ptr, 2, 1, 2, 2),
        0,
        "up dtype != tokens dtype"
    );
    assert_eq!(
        v(w_gate_ptr, w_up_ptr, w_down_fp16, 2, 1, 2, 2),
        0,
        "down dtype != tokens dtype"
    );

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_gate_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(w_gate_fp16);
    nsl_tensor_free(w_up_fp16);
    nsl_tensor_free(w_down_fp16);
    nsl_tensor_free(bad);
}

// ─────────────────────────────────────────────────────────────────────
// CPDT Part III v2.8 — GeGLU + ReGLU + activation-kind refusal tests.
// v4 was extended with `gate_activation_kind` ∈ {1, 2, 3} after the
// v2.5/v2.7 cycles fixed signature at 10 args. These tests pin:
//   - kind=2 (GeGLU) matches a hand-rolled `gelu(gate) * up @ down`
//     reference and DIFFERS from kind=1 (SwiGLU) by more than fp32 epsilon
//   - kind=3 (ReGLU) matches a hand-rolled `relu(gate) * up @ down`
//     reference and DIFFERS from kind=1 by more than fp32 epsilon
//   - kind=0 (identity) is refused (return 0) — a GLU with identity gate
//     degenerates to gate*up@down and is not a known production MoE
//   - kind∈{-1, 4, i64::MIN, i64::MAX} are all refused (full out-of-range
//     pin so the bounds check can't silently widen)
// ─────────────────────────────────────────────────────────────────────

fn gelu_tanh(x: f32) -> f32 {
    // tanh-approx GELU, matches torch.gelu(approximate='tanh') and v3
    // activation_kind=2. SQRT_2_OVER_PI computed inline (no const dep
    // on the FFI module).
    let sqrt_2_over_pi = 0.7978845608028654_f32;
    let inner = sqrt_2_over_pi * (x + 0.044715_f32 * x * x * x);
    0.5_f32 * x * (1.0_f32 + inner.tanh())
}

fn relu(x: f32) -> f32 {
    if x < 0.0_f32 { 0.0_f32 } else { x }
}

/// Hand-rolled GeGLU expert — `gelu(gate) * up @ down`. Independent
/// of swiglu_expert so a SwiGLU bug cannot mask a GeGLU bug.
fn geglu_expert(
    token: &[f32], w_gate: &[f32], w_up: &[f32], w_down: &[f32],
    hidden: usize, intermediate: usize,
) -> Vec<f32> {
    let mut x_gate = vec![0.0_f32; intermediate];
    let mut x_up = vec![0.0_f32; intermediate];
    for j in 0..intermediate {
        let mut sg = 0.0_f32;
        let mut su = 0.0_f32;
        for k in 0..hidden {
            sg += token[k] * w_gate[k * intermediate + j];
            su += token[k] * w_up[k * intermediate + j];
        }
        x_gate[j] = sg;
        x_up[j] = su;
    }
    let x_act: Vec<f32> = x_gate.iter().zip(x_up.iter())
        .map(|(&g, &u)| gelu_tanh(g) * u).collect();
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

/// Hand-rolled ReGLU expert — `relu(gate) * up @ down`.
fn reglu_expert(
    token: &[f32], w_gate: &[f32], w_up: &[f32], w_down: &[f32],
    hidden: usize, intermediate: usize,
) -> Vec<f32> {
    let mut x_gate = vec![0.0_f32; intermediate];
    let mut x_up = vec![0.0_f32; intermediate];
    for j in 0..intermediate {
        let mut sg = 0.0_f32;
        let mut su = 0.0_f32;
        for k in 0..hidden {
            sg += token[k] * w_gate[k * intermediate + j];
            su += token[k] * w_up[k * intermediate + j];
        }
        x_gate[j] = sg;
        x_up[j] = su;
    }
    let x_act: Vec<f32> = x_gate.iter().zip(x_up.iter())
        .map(|(&g, &u)| relu(g) * u).collect();
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
fn dispatch_v4_geglu_matches_hand_reference() {
    // 2 tokens, 1 expert (top_k=1, expert 0 wins both), hidden=2,
    // intermediate=2. Logits force both tokens to expert 0. Weights
    // chosen with non-trivial magnitudes so silu/gelu differ
    // by more than fp32 noise (silu(0.7)=0.4500 vs gelu(0.7)=0.5249).
    let tokens: Vec<f32> = vec![0.7, 0.3,  0.5, 0.9];
    // num_experts=1, top_k=1 → logits shape [total_tokens, 1].
    let logits: Vec<f32> = vec![1.0,  1.0];
    let w_gate: Vec<f32> = vec![1.1, -0.4,  0.6, 1.2];
    let w_up:   Vec<f32> = vec![0.8,  0.5, -0.3, 1.0];
    let w_down: Vec<f32> = vec![0.7, -0.2,  0.4, 0.9];

    let tokens_ptr = make_f32_tensor(&[2, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[2, 1], &logits);
    let w_gate_ptr = make_f32_tensor(&[1, 4], &w_gate);
    let w_up_ptr   = make_f32_tensor(&[1, 4], &w_up);
    let w_down_ptr = make_f32_tensor(&[1, 4], &w_down);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr, w_gate_ptr, w_up_ptr, w_down_ptr,
            1, 1, (4.0_f32).to_bits() as i64, 2, 2, 2, /* GeGLU */
            NO_BIAS, NO_BIAS, NO_BIAS,
        )
    };
    assert_ne!(out_ptr, 0, "v4 returned null for valid GeGLU inputs");
    let got = read_f32(out_ptr, 4);

    let mut expected = vec![0.0_f32; 4];
    for t in 0..2 {
        let tok = &tokens[t * 2..(t + 1) * 2];
        let row = geglu_expert(tok, &w_gate, &w_up, &w_down, 2, 2);
        expected[t * 2..(t + 1) * 2].copy_from_slice(&row);
    }

    let mut max_abs = 0.0_f32;
    for (g, w) in got.iter().zip(expected.iter()) {
        let d = (g - w).abs();
        if d > max_abs { max_abs = d; }
    }
    assert!(
        max_abs <= 1e-6,
        "GeGLU diverges from hand reference: got={got:?} want={expected:?} max_abs_diff={max_abs:.6e}"
    );

    // Also confirm GeGLU output DIFFERS from SwiGLU output for these
    // weights — proves we are not silently routing kind=2 through the
    // kind=1 branch.
    let swiglu_out_ptr = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr, w_gate_ptr, w_up_ptr, w_down_ptr,
            1, 1, (4.0_f32).to_bits() as i64, 2, 2, SWIGLU,
            NO_BIAS, NO_BIAS, NO_BIAS,
        )
    };
    let swiglu_out = read_f32(swiglu_out_ptr, 4);
    let mut max_diff = 0.0_f32;
    for (g, s) in got.iter().zip(swiglu_out.iter()) {
        let d = (g - s).abs();
        if d > max_diff { max_diff = d; }
    }
    assert!(
        max_diff > 1e-3,
        "GeGLU output too close to SwiGLU — branch may be aliased: max_diff={max_diff:.6e}"
    );

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_gate_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(out_ptr);
    nsl_tensor_free(swiglu_out_ptr);
}

#[test]
fn dispatch_v4_reglu_matches_hand_reference() {
    // Construct weights where SOME gate-projections are negative (so
    // ReLU clips them to zero) and others are positive — exercises
    // both branches of the ReLU. SwiGLU/silu would output a small but
    // nonzero value for the clipped lanes, so the SwiGLU-vs-ReGLU
    // divergence test below catches a silent alias.
    let tokens: Vec<f32> = vec![1.0, -0.5,  -0.3, 0.8];
    // num_experts=1, top_k=1 → logits shape [total_tokens, 1].
    let logits: Vec<f32> = vec![1.0,  1.0];
    // w_gate row 0 col 0 = +1.5 → x_gate[0] for token 0 = 0.75
    // w_gate row 0 col 1 = -1.0 → x_gate[1] for token 0 = -0.5 (ReLU→0)
    let w_gate: Vec<f32> = vec![1.5, -1.0,  0.3, 0.7];
    let w_up:   Vec<f32> = vec![0.8,  0.5, -0.3, 1.0];
    let w_down: Vec<f32> = vec![0.7, -0.2,  0.4, 0.9];

    let tokens_ptr = make_f32_tensor(&[2, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[2, 1], &logits);
    let w_gate_ptr = make_f32_tensor(&[1, 4], &w_gate);
    let w_up_ptr   = make_f32_tensor(&[1, 4], &w_up);
    let w_down_ptr = make_f32_tensor(&[1, 4], &w_down);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr, w_gate_ptr, w_up_ptr, w_down_ptr,
            1, 1, (4.0_f32).to_bits() as i64, 2, 2, 3, /* ReGLU */
            NO_BIAS, NO_BIAS, NO_BIAS,
        )
    };
    assert_ne!(out_ptr, 0, "v4 returned null for valid ReGLU inputs");
    let got = read_f32(out_ptr, 4);

    let mut expected = vec![0.0_f32; 4];
    for t in 0..2 {
        let tok = &tokens[t * 2..(t + 1) * 2];
        let row = reglu_expert(tok, &w_gate, &w_up, &w_down, 2, 2);
        expected[t * 2..(t + 1) * 2].copy_from_slice(&row);
    }

    let mut max_abs = 0.0_f32;
    for (g, w) in got.iter().zip(expected.iter()) {
        let d = (g - w).abs();
        if d > max_abs { max_abs = d; }
    }
    assert!(
        max_abs <= 1e-6,
        "ReGLU diverges from hand reference: got={got:?} want={expected:?} max_abs_diff={max_abs:.6e}"
    );

    // Confirm ReGLU output DIFFERS from SwiGLU for these weights.
    let swiglu_out_ptr = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr, w_gate_ptr, w_up_ptr, w_down_ptr,
            1, 1, (4.0_f32).to_bits() as i64, 2, 2, SWIGLU,
            NO_BIAS, NO_BIAS, NO_BIAS,
        )
    };
    let swiglu_out = read_f32(swiglu_out_ptr, 4);
    let mut max_diff = 0.0_f32;
    for (g, s) in got.iter().zip(swiglu_out.iter()) {
        let d = (g - s).abs();
        if d > max_diff { max_diff = d; }
    }
    assert!(
        max_diff > 1e-3,
        "ReGLU output too close to SwiGLU — branch may be aliased: max_diff={max_diff:.6e}"
    );

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_gate_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(out_ptr);
    nsl_tensor_free(swiglu_out_ptr);
}

#[test]
fn dispatch_v4_refuses_invalid_gate_activation_kind() {
    // Bounds check: kind must be in {1, 2, 3}. 0 (identity) is
    // structurally non-Mixtral so it's REFUSED at this FFI; callers
    // wanting identity gate should use v3 (a 2-weight FFN). Negative
    // and large-positive inputs are pinned to catch a silent widening
    // of the bounds check.
    let tokens_ptr = make_f32_tensor(&[2, 2], &[1.0, 2.0, 3.0, 4.0]);
    let logits_ptr = make_f32_tensor(&[2, 2], &[1.0, 0.5, 0.5, 1.0]);
    let w_gate_ptr = make_f32_tensor(&[2, 4], &vec![0.1_f32; 8]);
    let w_up_ptr = make_f32_tensor(&[2, 4], &vec![0.1_f32; 8]);
    let w_down_ptr = make_f32_tensor(&[2, 4], &vec![0.1_f32; 8]);
    let cap = (2.0_f32).to_bits() as i64;

    let v_kind = |kind: i64| unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr, w_gate_ptr, w_up_ptr, w_down_ptr,
            2, 1, cap, 2, 2, kind,
            NO_BIAS, NO_BIAS, NO_BIAS,
        )
    };

    assert_eq!(v_kind(0), 0, "identity (kind=0) refused");
    assert_eq!(v_kind(-1), 0, "kind=-1 refused");
    assert_eq!(v_kind(4), 0, "kind=4 refused");
    assert_eq!(v_kind(100), 0, "kind=100 refused");
    assert_eq!(v_kind(i64::MIN), 0, "kind=i64::MIN refused");
    assert_eq!(v_kind(i64::MAX), 0, "kind=i64::MAX refused");

    // Sanity: kind ∈ {1, 2, 3} all succeed (return non-zero) with
    // identical inputs — confirms the refusal is kind-specific.
    let ok1 = v_kind(1);
    let ok2 = v_kind(2);
    let ok3 = v_kind(3);
    assert_ne!(ok1, 0, "SwiGLU kind=1 must succeed");
    assert_ne!(ok2, 0, "GeGLU kind=2 must succeed");
    assert_ne!(ok3, 0, "ReGLU kind=3 must succeed");

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_gate_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(ok1);
    nsl_tensor_free(ok2);
    nsl_tensor_free(ok3);
}

// ── CPDT Part III v2.14 — v4 FFN bias tests ──────────────────────────────

/// SwiGLU expert with biases:
///   x_gate = (token @ W_gate) + gate_bias
///   x_up   = (token @ W_up)   + up_bias
///   x_act  = silu(x_gate) * x_up
///   out    = (x_act @ W_down) + down_bias
///
/// Independent of route_topk / scatter / gather so a routing-pipeline
/// regression cannot silently corrupt this reference.
#[allow(clippy::too_many_arguments)]
fn swiglu_expert_with_biases(
    token: &[f32],
    w_gate: &[f32],
    w_up: &[f32],
    w_down: &[f32],
    gate_bias: &[f32],
    up_bias: &[f32],
    down_bias: &[f32],
    hidden: usize,
    intermediate: usize,
) -> Vec<f32> {
    let mut x_gate = vec![0.0_f32; intermediate];
    let mut x_up = vec![0.0_f32; intermediate];
    for j in 0..intermediate {
        let mut sg = 0.0_f32;
        let mut su = 0.0_f32;
        for k in 0..hidden {
            sg += token[k] * w_gate[k * intermediate + j];
            su += token[k] * w_up[k * intermediate + j];
        }
        x_gate[j] = sg + gate_bias[j];
        x_up[j] = su + up_bias[j];
    }
    let mut x_act = vec![0.0_f32; intermediate];
    for j in 0..intermediate {
        x_act[j] = silu(x_gate[j]) * x_up[j];
    }
    let mut out = vec![0.0_f32; hidden];
    for h in 0..hidden {
        let mut s = 0.0_f32;
        for j in 0..intermediate {
            s += x_act[j] * w_down[j * hidden + h];
        }
        out[h] = s + down_bias[h];
    }
    out
}

/// All 3 biases (gate + up + down) supplied. Numerical match against
/// `swiglu_expert_with_biases` reference. Uses num_experts=2 so the
/// per-expert bias offset (e * dim) is exercised — a regression that
/// dropped the `e *` factor would still pass num_experts=1.
#[test]
fn dispatch_v4_all_three_biases_matches_hand_reference() {
    let hidden = 2_usize;
    let intermediate = 2_usize;
    let n_experts = 2_usize;
    // Tokens [2, 2]: both route deterministically — token 0 to expert 0, token 1 to expert 1.
    let tokens: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    // Logits [2, 2]: row 0 picks col 0, row 1 picks col 1.
    let logits: Vec<f32> = vec![10.0, -10.0, -10.0, 10.0];

    // Distinct per-expert weights so any mis-routing surfaces.
    let w_gate: Vec<f32> = vec![
        // expert 0: [hidden=2, intermediate=2]
        0.5, 1.0,
        -0.5, 0.25,
        // expert 1
        2.0, -1.0,
        0.5, 1.5,
    ];
    let w_up: Vec<f32> = vec![
        0.1, 0.2,
        0.3, 0.4,
        1.0, 0.5,
        -0.5, 0.25,
    ];
    let w_down: Vec<f32> = vec![
        // expert 0: [intermediate=2, hidden=2]
        1.0, 0.0,
        0.0, 1.0,
        // expert 1
        0.5, 0.25,
        -0.25, 0.5,
    ];
    // Per-expert biases.
    let gate_bias: Vec<f32> = vec![
        // expert 0: [intermediate=2]
        1.0, -0.5,
        // expert 1
        -1.0, 0.5,
    ];
    let up_bias: Vec<f32> = vec![
        0.25, -0.25,
        -0.5, 1.0,
    ];
    let down_bias: Vec<f32> = vec![
        // expert 0: [hidden=2]
        0.1, 0.2,
        // expert 1
        -0.1, -0.2,
    ];

    let tokens_ptr = make_f32_tensor(&[2, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[2, 2], &logits);
    let w_gate_ptr = make_f32_tensor(&[2, 4], &w_gate);
    let w_up_ptr = make_f32_tensor(&[2, 4], &w_up);
    let w_down_ptr = make_f32_tensor(&[2, 4], &w_down);
    let gate_bias_ptr = make_f32_tensor(&[2, 2], &gate_bias);
    let up_bias_ptr = make_f32_tensor(&[2, 2], &up_bias);
    let down_bias_ptr = make_f32_tensor(&[2, 2], &down_bias);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr, w_gate_ptr, w_up_ptr, w_down_ptr,
            n_experts as i64, 1, (2.0_f32).to_bits() as i64,
            hidden as i64, intermediate as i64, SWIGLU,
            gate_bias_ptr, up_bias_ptr, down_bias_ptr,
        )
    };
    assert_ne!(out_ptr, 0, "v4 returned null for valid biased SwiGLU inputs");
    let got = read_f32(out_ptr, 2 * hidden);

    // Reference: token 0 → expert 0; token 1 → expert 1.
    let token0 = &tokens[0..2];
    let token1 = &tokens[2..4];
    let exp0_out = swiglu_expert_with_biases(
        token0,
        &w_gate[0..4], &w_up[0..4], &w_down[0..4],
        &gate_bias[0..2], &up_bias[0..2], &down_bias[0..2],
        hidden, intermediate,
    );
    let exp1_out = swiglu_expert_with_biases(
        token1,
        &w_gate[4..8], &w_up[4..8], &w_down[4..8],
        &gate_bias[2..4], &up_bias[2..4], &down_bias[2..4],
        hidden, intermediate,
    );
    let mut expected = vec![0.0_f32; 2 * hidden];
    expected[0..hidden].copy_from_slice(&exp0_out);
    expected[hidden..2 * hidden].copy_from_slice(&exp1_out);

    for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
        assert!(
            (g - e).abs() < 1e-5,
            "v4 biased output mismatch at i={i}: got {g}, expected {e}",
        );
    }
}

/// Gate-bias-only path — verifies the bias-BEFORE-activation ordering
/// for gate. silu(x + bias) ≠ silu(x) + bias (the activation is
/// nonlinear), so a regression applying gate_bias AFTER silu would
/// produce numerically distinct output (>0.1 here) from the
/// hand-computed pre-activation path.
#[test]
fn dispatch_v4_gate_bias_only_pins_bias_before_activation_ordering() {
    let hidden = 1_usize;
    let intermediate = 1_usize;
    let n_experts = 1_usize;
    let tokens: Vec<f32> = vec![1.0, -1.0];
    let logits: Vec<f32> = vec![1.0, 1.0];
    // W_gate = [[1.0]], W_up = [[1.0]], W_down = [[1.0]].
    let w_gate: Vec<f32> = vec![1.0];
    let w_up: Vec<f32> = vec![1.0];
    let w_down: Vec<f32> = vec![1.0];
    // Strong gate_bias → silu(x+2) vs silu(x)+2 diverges by >0.1.
    let gate_bias: Vec<f32> = vec![2.0];

    let tokens_ptr = make_f32_tensor(&[2, 1], &tokens);
    let logits_ptr = make_f32_tensor(&[2, 1], &logits);
    let w_gate_ptr = make_f32_tensor(&[1, 1], &w_gate);
    let w_up_ptr = make_f32_tensor(&[1, 1], &w_up);
    let w_down_ptr = make_f32_tensor(&[1, 1], &w_down);
    let gate_bias_ptr = make_f32_tensor(&[1, 1], &gate_bias);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr, w_gate_ptr, w_up_ptr, w_down_ptr,
            n_experts as i64, 1, (4.0_f32).to_bits() as i64,
            hidden as i64, intermediate as i64, SWIGLU,
            gate_bias_ptr, NO_BIAS, NO_BIAS,
        )
    };
    assert_ne!(out_ptr, 0);
    let got = read_f32(out_ptr, 2 * hidden);

    // Expected per-token pre-bias-activation: silu(token + 2) * token.
    // Token 0 = 1.0: silu(3.0) * 1.0 ≈ 2.857
    // Token 1 = -1.0: silu(1.0) * -1.0 ≈ -0.731
    let want_t0 = silu(1.0 + 2.0) * 1.0;
    let want_t1 = silu(-1.0 + 2.0) * -1.0;
    assert!((got[0] - want_t0).abs() < 1e-5, "got[0]={} expected {}", got[0], want_t0);
    assert!((got[1] - want_t1).abs() < 1e-5, "got[1]={} expected {}", got[1], want_t1);

    // Confirm divergence vs post-activation application.
    let post_act_t0 = silu(1.0) * 1.0 + 2.0; // wrong: bias added AFTER silu
    assert!(
        (got[0] - post_act_t0).abs() > 0.1,
        "gate bias must produce distinct output vs post-silu addition (sanity check)",
    );
}

/// v2.14 fix F3 (IMPORTANT adversarial review) — non-square dims
/// (hidden=2, intermediate=4) so the 3 bias offsets diverge:
///   gate_bias_off = e * intermediate = e * 4
///   up_bias_off   = e * intermediate = e * 4   (same as gate)
///   down_bias_off = e * hidden       = e * 2
/// A regression that swapped gate_bias and down_bias offsets would
/// have undetectable consequences with hidden=intermediate (both
/// offsets equal e * D); non-square dims surface the bug.
#[test]
fn dispatch_v4_all_three_biases_hidden_neq_intermediate_matches_reference() {
    let hidden = 2_usize;
    let intermediate = 4_usize;
    let n_experts = 2_usize;
    let tokens: Vec<f32> = vec![1.0, 0.0, 0.0, 1.0];
    let logits: Vec<f32> = vec![10.0, -10.0, -10.0, 10.0];

    let w_gate: Vec<f32> = (0..n_experts * hidden * intermediate)
        .map(|i| 0.1_f32 * (i as f32 + 1.0))
        .collect();
    let w_up: Vec<f32> = (0..n_experts * hidden * intermediate)
        .map(|i| 0.2_f32 * (i as f32 + 1.0))
        .collect();
    let w_down: Vec<f32> = (0..n_experts * intermediate * hidden)
        .map(|i| 0.05_f32 * (i as f32 + 1.0))
        .collect();
    // Per-expert biases — distinct values + asymmetric direction
    // sizes ensure offsets cannot be swapped silently.
    let gate_bias: Vec<f32> = (0..n_experts * intermediate)
        .map(|i| (i as f32) + 1.0)
        .collect();
    let up_bias: Vec<f32> = (0..n_experts * intermediate)
        .map(|i| -(i as f32) - 0.5)
        .collect();
    let down_bias: Vec<f32> = (0..n_experts * hidden)
        .map(|i| 0.25_f32 * (i as f32 + 1.0))
        .collect();

    let tokens_ptr = make_f32_tensor(&[2, hidden as i64], &tokens);
    let logits_ptr = make_f32_tensor(&[2, n_experts as i64], &logits);
    let w_gate_ptr = make_f32_tensor(&[n_experts as i64, (hidden * intermediate) as i64], &w_gate);
    let w_up_ptr = make_f32_tensor(&[n_experts as i64, (hidden * intermediate) as i64], &w_up);
    let w_down_ptr = make_f32_tensor(&[n_experts as i64, (intermediate * hidden) as i64], &w_down);
    let gate_bias_ptr = make_f32_tensor(&[n_experts as i64, intermediate as i64], &gate_bias);
    let up_bias_ptr = make_f32_tensor(&[n_experts as i64, intermediate as i64], &up_bias);
    let down_bias_ptr = make_f32_tensor(&[n_experts as i64, hidden as i64], &down_bias);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr, w_gate_ptr, w_up_ptr, w_down_ptr,
            n_experts as i64, 1, (2.0_f32).to_bits() as i64,
            hidden as i64, intermediate as i64, SWIGLU,
            gate_bias_ptr, up_bias_ptr, down_bias_ptr,
        )
    };
    assert_ne!(out_ptr, 0);
    let got = read_f32(out_ptr, 2 * hidden);

    let exp0 = swiglu_expert_with_biases(
        &tokens[0..hidden],
        &w_gate[0..hidden * intermediate],
        &w_up[0..hidden * intermediate],
        &w_down[0..intermediate * hidden],
        &gate_bias[0..intermediate],
        &up_bias[0..intermediate],
        &down_bias[0..hidden],
        hidden, intermediate,
    );
    let exp1 = swiglu_expert_with_biases(
        &tokens[hidden..2 * hidden],
        &w_gate[hidden * intermediate..2 * hidden * intermediate],
        &w_up[hidden * intermediate..2 * hidden * intermediate],
        &w_down[intermediate * hidden..2 * intermediate * hidden],
        &gate_bias[intermediate..2 * intermediate],
        &up_bias[intermediate..2 * intermediate],
        &down_bias[hidden..2 * hidden],
        hidden, intermediate,
    );

    for h in 0..hidden {
        assert!(
            (got[h] - exp0[h]).abs() < 1e-4,
            "token 0 expert 0 h={h}: got {} expected {}",
            got[h], exp0[h],
        );
        assert!(
            (got[hidden + h] - exp1[h]).abs() < 1e-4,
            "token 1 expert 1 h={h}: got {} expected {}",
            got[hidden + h], exp1[h],
        );
    }
}

/// Refuse cases for bias args: f16 bias dtype mismatch, wrong-length,
/// device != 0. v2.11/v2.14 require all biases match tokens.dtype and
/// be on CPU.
///
/// v2.14 fix F4 (IMPORTANT adversarial review): coverage now includes
/// the middle direction (wrong-length up_bias) — the original test
/// covered gate + down but skipped up. A regression that mis-indexed
/// the up_bias check would otherwise pass silently.
#[test]
fn dispatch_v4_refuses_invalid_bias() {
    let hidden = 2_usize;
    let intermediate = 2_usize;
    let n_experts = 1_usize;
    let tokens: Vec<f32> = vec![1.0, 0.0];
    let logits: Vec<f32> = vec![1.0];
    let w: Vec<f32> = vec![0.5, 0.5, 0.5, 0.5];
    let gate_bias_ok: Vec<f32> = vec![0.0, 0.0];

    let tokens_ptr = make_f32_tensor(&[1, 2], &tokens);
    let logits_ptr = make_f32_tensor(&[1, 1], &logits);
    let w_gate_ptr = make_f32_tensor(&[1, 4], &w);
    let w_up_ptr = make_f32_tensor(&[1, 4], &w);
    let w_down_ptr = make_f32_tensor(&[1, 4], &w);

    // F16 bias → dtype mismatch → refuse.
    let f16_shape_list = nsl_list_new();
    nsl_list_push(f16_shape_list, 1);
    nsl_list_push(f16_shape_list, 2);
    let f16_bias_ptr = nsl_tensor_zeros_f16_on(f16_shape_list, 0);
    nsl_list_free(f16_shape_list);

    let r1 = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr, w_gate_ptr, w_up_ptr, w_down_ptr,
            n_experts as i64, 1, (4.0_f32).to_bits() as i64,
            hidden as i64, intermediate as i64, SWIGLU,
            f16_bias_ptr, NO_BIAS, NO_BIAS,
        )
    };
    assert_eq!(r1, 0, "f16 gate_bias against f32 tokens must refuse");

    // Wrong-length gate bias (expected = 1*2 = 2, got 3).
    let bad_gate_bias: Vec<f32> = vec![0.0, 0.0, 0.0];
    let bad_gate_bias_ptr = make_f32_tensor(&[3], &bad_gate_bias);

    let r2 = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr, w_gate_ptr, w_up_ptr, w_down_ptr,
            n_experts as i64, 1, (4.0_f32).to_bits() as i64,
            hidden as i64, intermediate as i64, SWIGLU,
            bad_gate_bias_ptr, NO_BIAS, NO_BIAS,
        )
    };
    assert_eq!(r2, 0, "wrong-length gate_bias must refuse");

    // v2.14 fix F4 — wrong-length UP_bias (middle direction; the
    // original test had asymmetric coverage on gate + down only).
    // expected = 1 * 2 = 2, got 7.
    let bad_up_bias: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let gate_bias_ptr_a = make_f32_tensor(&[1, 2], &gate_bias_ok);
    let bad_up_bias_ptr = make_f32_tensor(&[7], &bad_up_bias);
    let down_bias_ptr_ok = make_f32_tensor(&[1, 2], &gate_bias_ok);

    let r_up = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr, w_gate_ptr, w_up_ptr, w_down_ptr,
            n_experts as i64, 1, (4.0_f32).to_bits() as i64,
            hidden as i64, intermediate as i64, SWIGLU,
            gate_bias_ptr_a, bad_up_bias_ptr, down_bias_ptr_ok,
        )
    };
    assert_eq!(r_up, 0, "wrong-length up_bias must refuse");

    // Wrong-length down bias (expected = 1*2 = 2, got 5).
    let bad_down_bias: Vec<f32> = vec![0.0, 0.0, 0.0, 0.0, 0.0];
    let gate_bias_ptr = make_f32_tensor(&[1, 2], &gate_bias_ok);
    let up_bias_ptr = make_f32_tensor(&[1, 2], &gate_bias_ok);
    let bad_down_bias_ptr = make_f32_tensor(&[5], &bad_down_bias);

    let r3 = unsafe {
        nsl_moe_dispatch_full_v4(
            tokens_ptr, logits_ptr, w_gate_ptr, w_up_ptr, w_down_ptr,
            n_experts as i64, 1, (4.0_f32).to_bits() as i64,
            hidden as i64, intermediate as i64, SWIGLU,
            gate_bias_ptr, up_bias_ptr, bad_down_bias_ptr,
        )
    };
    assert_eq!(r3, 0, "wrong-length down_bias must refuse");

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(w_gate_ptr);
    nsl_tensor_free(w_up_ptr);
    nsl_tensor_free(w_down_ptr);
    nsl_tensor_free(f16_bias_ptr);
    nsl_tensor_free(bad_gate_bias_ptr);
    nsl_tensor_free(gate_bias_ptr);
    nsl_tensor_free(up_bias_ptr);
    nsl_tensor_free(bad_down_bias_ptr);
    nsl_tensor_free(gate_bias_ptr_a);
    nsl_tensor_free(bad_up_bias_ptr);
    nsl_tensor_free(down_bias_ptr_ok);
}
