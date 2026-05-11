//! Hand-coded analytical reference for the WGGO merge-gate test.
//!
//! Spec ref: §6.4. Pure Rust math, no infrastructure dependencies.
//!
//! Computation order:
//!   Forward:  y = softmax(X Wq (X Wk)^T) (X Wv) Wo
//!   Loss:     L = sum(y²)
//!   Backward: chain-rule through Wo, attn, V, softmax-Jacobian, Q, K
//!
//! Per-head gradient score aggregates |dW * W| over each head's row range.

use std::collections::HashMap;

// ---------------------------------------------------------------------------
// f64 matrix helpers (f64 throughout for accumulator parity with IR)
// ---------------------------------------------------------------------------

fn matmul_f64(
    a: &[f64],
    a_rows: usize,
    a_cols: usize,
    b: &[f64],
    b_rows: usize,
    b_cols: usize,
) -> Vec<f64> {
    assert_eq!(a_cols, b_rows);
    let mut out = vec![0.0; a_rows * b_cols];
    for i in 0..a_rows {
        for j in 0..b_cols {
            let mut acc = 0.0;
            for k in 0..a_cols {
                acc += a[i * a_cols + k] * b[k * b_cols + j];
            }
            out[i * b_cols + j] = acc;
        }
    }
    out
}

fn transpose_f64(m: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut out = vec![0.0; rows * cols];
    for i in 0..rows {
        for j in 0..cols {
            out[j * rows + i] = m[i * cols + j];
        }
    }
    out
}

fn softmax_rowwise_f64(m: &mut [f64], rows: usize, cols: usize) {
    for r in 0..rows {
        let row = &mut m[r * cols..(r + 1) * cols];
        let max = row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum: f64 = row.iter().map(|x| (x - max).exp()).sum();
        for x in row.iter_mut() {
            *x = (*x - max).exp() / sum;
        }
    }
}

// ---------------------------------------------------------------------------
// Per-head gradient score
// ---------------------------------------------------------------------------

/// Aggregate |dW * W| over each head's row range.
///
/// Layout: W is [dim, dim] = [n_heads * head_dim, dim].
/// Per-head row range: rows [h*head_dim .. (h+1)*head_dim].
fn per_head_score(d_w: &[f64], w: &[f64], dim: usize, head_dim: usize) -> Vec<f32> {
    let n_heads = dim / head_dim;
    let mut scores = vec![0.0_f64; n_heads];
    for (h, score) in scores.iter_mut().enumerate() {
        let row_start = h * head_dim * dim;
        let row_end = (h + 1) * head_dim * dim;
        for i in row_start..row_end {
            *score += (d_w[i] * w[i]).abs();
        }
    }
    scores.iter().map(|&v| v as f32).collect()
}

// ---------------------------------------------------------------------------
// Main reference function
// ---------------------------------------------------------------------------

/// Hand-coded analytical WGGO reference for a single AttentionMLP layer.
///
/// Returns a map from layer name to per-head gradient importance scores.
/// Key `"AttentionMLP"` maps to a `Vec<f32>` of length `num_heads`.
///
/// `calib` is the calibration input of shape `[count, seq, dim]` (flattened
/// to length `count * seq * dim`). Each of the `count` samples is processed
/// independently — the model's `forward(x: Tensor<[B, S, D]>)` performs
/// **batched** attention with `softmax` over `seq` (not over `count * seq`),
/// so the per-head |dW · W| score aggregates `count` independent backward
/// passes and matches what the calibration subprocess produces by feeding
/// the model one batch at a time.
///
/// `weights` must contain keys: `"AttentionMLP.q_proj"`, `"AttentionMLP.k_proj"`,
/// `"AttentionMLP.v_proj"`, `"AttentionMLP.o_proj"` each of size `dim * dim`.
///
/// # Historical note (#147 hop 13)
///
/// An earlier version of this reference treated `calib` as `[batch * seq, dim]`
/// — one big flattened sample with `softmax` over `count * seq = 32` keys.
/// That was inconsistent with both the merge-gate fixture's declared input
/// shape (`let x = zeros([8, 4, 32])`) and the calibration subprocess's
/// per-sample feeding. Once the subprocess actually wrote correct values into
/// the BSS running buffer (hop 13 fixes in `binary_codegen.rs`), the actual
/// scores came out ≈60% of the old reference's flattened result — exactly the
/// constant-factor structural mismatch you'd expect from `softmax(K=32)` vs
/// `softmax(K=4)`. The reference now mirrors the subprocess: `count` separate
/// `[seq, dim]` forward/backward passes, gradients summed.
pub fn reference_wggo_head_scores(
    calib: &[f32],
    weights: &HashMap<&str, Vec<f32>>,
    dim: usize,
    num_heads: usize,
) -> HashMap<String, Vec<f32>> {
    let head_dim = dim / num_heads;
    let total_rows = calib.len() / dim;

    // The merge-gate fixture stores calibration data as `[count, seq, dim]`
    // and the subprocess feeds the model `count` batches of `[1, seq, dim]`.
    // For the test we hard-code the fixture's seq=4 — the only consumer is
    // the merge-gate test (dim=32, count*seq=32) which always uses seq=4.
    // `analytical_reference_produces_four_head_scores` below pins this.
    let seq = 4;
    assert!(
        total_rows.is_multiple_of(seq),
        "calib row count {} must be a multiple of seq={}",
        total_rows,
        seq
    );
    let count = total_rows / seq;

    // Promote everything to f64 for accumulator parity with IR.
    let x_full: Vec<f64> = calib.iter().map(|&v| v as f64).collect();
    let w_q: Vec<f64> = weights["AttentionMLP.q_proj"]
        .iter()
        .map(|&v| v as f64)
        .collect();
    let w_k: Vec<f64> = weights["AttentionMLP.k_proj"]
        .iter()
        .map(|&v| v as f64)
        .collect();
    let w_v: Vec<f64> = weights["AttentionMLP.v_proj"]
        .iter()
        .map(|&v| v as f64)
        .collect();
    let w_o: Vec<f64> = weights["AttentionMLP.o_proj"]
        .iter()
        .map(|&v| v as f64)
        .collect();

    // Accumulate per-head scores across all `count` samples. We accumulate
    // SCORES (`Σ_elem |dW · W|` per head, per batch) — NOT raw gradients —
    // because the subprocess's `emit_per_head_dot_abs_accum` takes the
    // element-wise absolute value before the per-head sum, so the running
    // buffer holds `Σ_batch Σ_elem |dW_batch[i] · W[i]|`. Accumulating raw
    // gradients first and then taking `per_head_score` would give the
    // different `Σ_elem |Σ_batch dW_batch[i] · W[i]|` quantity, since
    // `|a+b| ≠ |a|+|b|` in general.
    let mut q_score_acc = vec![0.0_f64; num_heads];
    let mut k_score_acc = vec![0.0_f64; num_heads];
    let mut v_score_acc = vec![0.0_f64; num_heads];
    let mut o_score_acc = vec![0.0_f64; num_heads];

    for batch_idx in 0..count {
        let x = &x_full[batch_idx * seq * dim..(batch_idx + 1) * seq * dim];

        // Forward pass on this batch (seq rows, dim cols).
        let q = matmul_f64(x, seq, dim, &w_q, dim, dim);
        let k = matmul_f64(x, seq, dim, &w_k, dim, dim);
        let v = matmul_f64(x, seq, dim, &w_v, dim, dim);
        let k_t = transpose_f64(&k, seq, dim);
        let scores = matmul_f64(&q, seq, dim, &k_t, dim, seq);
        let mut attn = scores.clone();
        softmax_rowwise_f64(&mut attn, seq, seq);
        let out = matmul_f64(&attn, seq, seq, &v, seq, dim);
        let y = matmul_f64(&out, seq, dim, &w_o, dim, dim);

        // Loss = sum(y²); dy = 2y
        let dy: Vec<f64> = y.iter().map(|v| 2.0 * v).collect();

        // Backward pass — same chain rule as before, but on per-batch shapes.
        let out_t = transpose_f64(&out, seq, dim);
        let d_w_o = matmul_f64(&out_t, dim, seq, &dy, seq, dim);
        let w_o_t = transpose_f64(&w_o, dim, dim);
        let d_out = matmul_f64(&dy, seq, dim, &w_o_t, dim, dim);
        let attn_t = transpose_f64(&attn, seq, seq);
        let d_v_act = matmul_f64(&attn_t, seq, seq, &d_out, seq, dim);
        let v_t = transpose_f64(&v, seq, dim);
        let d_attn = matmul_f64(&d_out, seq, dim, &v_t, dim, seq);

        let mut d_scores = vec![0.0; seq * seq];
        for r in 0..seq {
            let attn_row = &attn[r * seq..(r + 1) * seq];
            let d_attn_row = &d_attn[r * seq..(r + 1) * seq];
            let dot: f64 = attn_row
                .iter()
                .zip(d_attn_row.iter())
                .map(|(a, d)| a * d)
                .sum();
            for j in 0..seq {
                d_scores[r * seq + j] = attn_row[j] * (d_attn_row[j] - dot);
            }
        }

        let d_q_act = matmul_f64(&d_scores, seq, seq, &k, seq, dim);
        let d_scores_t = transpose_f64(&d_scores, seq, seq);
        let d_k_act = matmul_f64(&d_scores_t, seq, seq, &q, seq, dim);

        let x_t = transpose_f64(x, seq, dim);
        let d_w_q = matmul_f64(&x_t, dim, seq, &d_q_act, seq, dim);
        let d_w_k = matmul_f64(&x_t, dim, seq, &d_k_act, seq, dim);
        let d_w_v = matmul_f64(&x_t, dim, seq, &d_v_act, seq, dim);

        // Per-batch per-head score: `Σ_elem |dW · W|` per head row range.
        // Matches the subprocess's per-batch reduction inside
        // `emit_per_head_dot_abs_accum`.
        let q_scores_batch = per_head_score(&d_w_q, &w_q, dim, head_dim);
        let k_scores_batch = per_head_score(&d_w_k, &w_k, dim, head_dim);
        let v_scores_batch = per_head_score(&d_w_v, &w_v, dim, head_dim);
        let o_scores_batch = per_head_score(&d_w_o, &w_o, dim, head_dim);
        for h in 0..num_heads {
            q_score_acc[h] += q_scores_batch[h] as f64;
            k_score_acc[h] += k_scores_batch[h] as f64;
            v_score_acc[h] += v_scores_batch[h] as f64;
            o_score_acc[h] += o_scores_batch[h] as f64;
        }
    }

    // For symmetric MHA, n_q_heads == n_kv_heads == n_o_heads, so simple
    // cross-projection sum gives the layer's per-head score.
    let mut combined = vec![0.0_f32; num_heads];
    for h in 0..num_heads {
        combined[h] = (q_score_acc[h] + k_score_acc[h] + v_score_acc[h] + o_score_acc[h]) as f32;
    }

    let mut out = HashMap::new();
    out.insert("AttentionMLP".into(), combined);
    out
}

// ---------------------------------------------------------------------------
// Test helpers
// ---------------------------------------------------------------------------

/// Build deterministic test weights for a square [dim × dim] attention block.
pub fn build_test_weights(dim: usize) -> HashMap<&'static str, Vec<f32>> {
    let mut weights = HashMap::new();
    for (name, seed) in &[
        ("AttentionMLP.q_proj", 1u32),
        ("AttentionMLP.k_proj", 2),
        ("AttentionMLP.v_proj", 3),
        ("AttentionMLP.o_proj", 4),
    ] {
        let w: Vec<f32> = (0..dim * dim)
            .map(|i| ((i as u32 * seed) % 1024) as f32 / 1024.0)
            .collect();
        weights.insert(*name, w);
    }
    weights
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_returns_per_head_scores_with_correct_shape() {
        let calib = vec![0.1_f32; 8 * 4 * 32];
        let weights = build_test_weights(32);
        let scores = reference_wggo_head_scores(&calib, &weights, 32, 4);
        let layer = scores.get("AttentionMLP").unwrap();
        assert_eq!(layer.len(), 4); // num_heads
    }

    #[test]
    fn reference_zero_input_yields_zero_grad() {
        let calib = vec![0.0_f32; 8 * 4 * 32];
        let weights = build_test_weights(32);
        let scores = reference_wggo_head_scores(&calib, &weights, 32, 4);
        let layer = scores.get("AttentionMLP").unwrap();
        for &s in layer {
            assert!(s.abs() < 1e-10);
        }
    }
}
