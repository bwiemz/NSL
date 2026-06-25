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
            2, 1, (2.0_f32).to_bits() as i64, 3, 2,
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
            2, 1, (2.0_f32).to_bits() as i64, 2, 2,
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
            tokens_ptr, logits_ptr, gate, up, down, ne, tk, cap, h, i,
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
            t, l, w_gate_ptr, w_up_ptr, w_down_ptr, 2, 1, cap, 2, 2,
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
