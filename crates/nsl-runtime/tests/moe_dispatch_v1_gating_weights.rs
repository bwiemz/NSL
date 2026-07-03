//! Pins the gating-weighted gather semantics of the v1 MoE dispatch
//! (`nsl_moe_dispatch_full`, the M32 identity-skeleton backward-compat
//! fallback emitted by codegen for non-CPDT builds without a WeightMap).
//!
//! Historically v1 gathered every surviving expert copy with a uniform
//! weight of 1.0, which DOUBLE-COUNTED token contributions at top_k=2
//! (output ~= 2x input for identity experts). The fix threads the router's
//! `sorted_assignment_weights` through `dispatch::gather_tokens` — the same
//! pipeline v2/v3/v4 already consume. These tests pin the fixed contract:
//!
//!   - top_k=2, no capacity drops: per-token weights sum to 1.0, so with
//!     identity experts each output row ~= its input row (NOT 2x).
//!   - top_k=1: weights are exactly 1.0 (single-element renormalization is
//!     IEEE x/x), so the output is BIT-EXACT pass-through.
//!   - Capacity drop: dropped assignments contribute 0, leaving a partial
//!     (< 1.0 total weight) sum for the affected token — consistent with
//!     v2/v3/v4 semantics.

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_zeros_on};

extern "C" {
    fn nsl_moe_dispatch_full(
        tokens_ptr: i64,
        logits_ptr: i64,
        num_experts: i64,
        top_k: i64,
        capacity_factor_bits: i64,
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

#[test]
fn dispatch_v1_top_k_two_weights_sum_to_one_not_double_count() {
    // 4 tokens, 2 experts, top_k=2 => every token routes to BOTH experts.
    // capacity_factor=4.0 => capacity = ceil((4/2)*4) = 8 >= 4 assignments
    // per expert, so nothing is dropped. With identity experts and
    // renormalized gate weights (w0 + w1 == 1.0 per token), each output row
    // must equal its input row. The pre-fix uniform-1.0 gather produced
    // 2x each row — the 1e-5 gate below refutes that double-count.
    let total_tokens = 4_usize;
    let hidden_dim = 3_usize;
    let num_experts = 2_usize;

    let tokens: Vec<f32> = vec![
        1.0, 2.0, 3.0,
        -4.0, 5.0, -6.0,
        7.0, -8.0, 9.0,
        0.5, 0.25, -0.125,
    ];
    // Strongly separated logits (per-token expert preference alternates).
    // The specific values do not matter for the sum-to-one property — the
    // separation just makes the two per-token weights clearly unequal, so
    // an accidental uniform-0.5 gather would also be caught by test 3.
    let logits: Vec<f32> = vec![
        3.0, 0.0,
        0.0, 3.0,
        3.0, 0.0,
        0.0, 3.0,
    ];

    let tokens_ptr = make_f32_tensor(&[total_tokens as i64, hidden_dim as i64], &tokens);
    let logits_ptr = make_f32_tensor(&[total_tokens as i64, num_experts as i64], &logits);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full(
            tokens_ptr,
            logits_ptr,
            num_experts as i64,
            2, // top_k
            (4.0_f32).to_bits() as i64,
        )
    };
    assert_ne!(out_ptr, 0, "v1 returned null for valid top_k=2 inputs");

    let got = read_f32(out_ptr, total_tokens * hidden_dim);
    for (i, (g, w)) in got.iter().zip(tokens.iter()).enumerate() {
        assert!(
            (g - w).abs() <= 1e-5 * w.abs().max(1.0),
            "v1 top_k=2 output[{}] = {} != input {} — gating weights do not \
             sum to 1.0 (a value of ~{} would indicate the old uniform-1.0 \
             double-count)",
            i,
            g,
            w,
            2.0 * w,
        );
    }

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v1_top_k_one_is_bit_exact_pass_through() {
    // top_k=1: the single surviving weight per token is p/p == 1.0 exactly
    // (IEEE division of a finite non-zero value by itself), so the identity
    // expert must yield a BIT-EXACT copy of the input. This pins the
    // backward-compat contract the m32_moe_basic e2e baseline relies on.
    let total_tokens = 4_usize;
    let hidden_dim = 3_usize;
    let num_experts = 2_usize;

    let tokens: Vec<f32> = vec![
        1.5, -2.25, 3.125,
        4.0, 5.5, -6.75,
        -7.0, 8.125, 9.5,
        10.0, -11.25, 12.5,
    ];
    // 2 tokens per expert; capacity = ceil((4/2)*2) = 4 => no drops.
    let logits: Vec<f32> = vec![
        5.0, 0.0,
        0.0, 5.0,
        5.0, 0.0,
        0.0, 5.0,
    ];

    let tokens_ptr = make_f32_tensor(&[total_tokens as i64, hidden_dim as i64], &tokens);
    let logits_ptr = make_f32_tensor(&[total_tokens as i64, num_experts as i64], &logits);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full(
            tokens_ptr,
            logits_ptr,
            num_experts as i64,
            1, // top_k
            (2.0_f32).to_bits() as i64,
        )
    };
    assert_ne!(out_ptr, 0, "v1 returned null for valid top_k=1 inputs");

    let got = read_f32(out_ptr, total_tokens * hidden_dim);
    let got_bits: Vec<u32> = got.iter().map(|v| v.to_bits()).collect();
    let want_bits: Vec<u32> = tokens.iter().map(|v| v.to_bits()).collect();
    assert_eq!(
        got_bits, want_bits,
        "v1 top_k=1 must be a bit-exact identity pass-through: got={:?} want={:?}",
        got, tokens
    );

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(out_ptr);
}

#[test]
fn dispatch_v1_capacity_drop_yields_partial_weight_sum() {
    // 2 tokens, 3 experts, top_k=2, capacity_factor=1.0 =>
    // capacity = ceil((2/3)*1.0) = 1 slot per expert.
    //
    // token 0 logits [2, 1, -5]  => top-2 = experts {0, 1}
    // token 1 logits [2, 0, 1]   => top-2 = experts {0, 2}
    //
    // Assignment order (token-major): (t0,e0) fills expert 0, (t0,e1) fills
    // expert 1, (t1,e0) is DROPPED (expert 0 full), (t1,e2) fills expert 2.
    //
    //   token 0 out = (w00 + w01) * input0 = 1.0 * input0   (full weight)
    //   token 1 out = w12 * input1, w12 = p2 / (p0 + p2) < 1  (partial sum)
    //
    // consistent with v2/v3/v4 drop semantics (dropped assignments
    // contribute 0; no re-scaling of survivors).
    let hidden_dim = 2_usize;
    let num_experts = 3_usize;

    let tokens: Vec<f32> = vec![
        2.0, -3.0, // token 0
        4.0, 8.0, // token 1
    ];
    let logits: Vec<f32> = vec![
        2.0, 1.0, -5.0, // token 0
        2.0, 0.0, 1.0, // token 1
    ];

    // Replicate the router's f32 stable-softmax + top-2 renormalization for
    // token 1 to derive the exact surviving weight for expert 2.
    let row = [2.0_f32, 0.0, 1.0];
    let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
    let sum: f32 = exps.iter().sum();
    let probs: Vec<f32> = exps.iter().map(|&e| e / sum).collect();
    // top-2 of [p0 ~ 0.665, p1 ~ 0.090, p2 ~ 0.245] = experts 0 and 2.
    let weight_sum = probs[0] + probs[2];
    let w12 = probs[2] / weight_sum;
    assert!(w12 < 1.0 && w12 > 0.0, "test setup: expected partial weight");

    let tokens_ptr = make_f32_tensor(&[2, hidden_dim as i64], &tokens);
    let logits_ptr = make_f32_tensor(&[2, num_experts as i64], &logits);

    let out_ptr = unsafe {
        nsl_moe_dispatch_full(
            tokens_ptr,
            logits_ptr,
            num_experts as i64,
            2, // top_k
            (1.0_f32).to_bits() as i64,
        )
    };
    assert_ne!(out_ptr, 0, "v1 returned null for valid capacity-drop inputs");

    let got = read_f32(out_ptr, 2 * hidden_dim);

    // Token 0: both assignments survive => full renormalized weight 1.0.
    for d in 0..hidden_dim {
        assert!(
            (got[d] - tokens[d]).abs() <= 1e-6 * tokens[d].abs().max(1.0),
            "token 0 (no drop) output[{}] = {} != input {}",
            d,
            got[d],
            tokens[d],
        );
    }
    // Token 1: expert-0 assignment dropped => partial sum w12 * input.
    for d in 0..hidden_dim {
        let want = w12 * tokens[hidden_dim + d];
        assert!(
            (got[hidden_dim + d] - want).abs() <= 1e-6 * want.abs().max(1.0),
            "token 1 (dropped assignment) output[{}] = {} != w12*input = {} \
             (w12 = {}); full-weight output would be {}",
            d,
            got[hidden_dim + d],
            want,
            w12,
            tokens[hidden_dim + d],
        );
        assert!(
            got[hidden_dim + d].abs() < tokens[hidden_dim + d].abs(),
            "token 1 output must be strictly smaller in magnitude than the \
             input (partial weight after capacity drop)",
        );
    }

    nsl_tensor_free(tokens_ptr);
    nsl_tensor_free(logits_ptr);
    nsl_tensor_free(out_ptr);
}
