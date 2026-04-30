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
/// `calib` is the calibration input of shape `[batch * seq, dim]` (flattened).
/// `weights` must contain keys: `"AttentionMLP.q_proj"`, `"AttentionMLP.k_proj"`,
/// `"AttentionMLP.v_proj"`, `"AttentionMLP.o_proj"` each of size `dim * dim`.
pub fn reference_wggo_head_scores(
    calib: &[f32],
    weights: &HashMap<&str, Vec<f32>>,
    dim: usize,
    num_heads: usize,
) -> HashMap<String, Vec<f32>> {
    let head_dim = dim / num_heads;
    let n_rows = calib.len() / dim; // batch × seq flattened

    // Promote everything to f64 for accumulator parity with IR.
    let x: Vec<f64> = calib.iter().map(|&v| v as f64).collect();
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

    // Forward pass: y = softmax(X Wq (X Wk)^T) (X Wv) Wo
    let q = matmul_f64(&x, n_rows, dim, &w_q, dim, dim);
    let k = matmul_f64(&x, n_rows, dim, &w_k, dim, dim);
    let v = matmul_f64(&x, n_rows, dim, &w_v, dim, dim);
    let k_t = transpose_f64(&k, n_rows, dim);
    let scores = matmul_f64(&q, n_rows, dim, &k_t, dim, n_rows);
    let mut attn = scores.clone();
    softmax_rowwise_f64(&mut attn, n_rows, n_rows);
    let out = matmul_f64(&attn, n_rows, n_rows, &v, n_rows, dim);
    let y = matmul_f64(&out, n_rows, dim, &w_o, dim, dim);

    // Loss = sum(y²); dy = 2y
    let dy: Vec<f64> = y.iter().map(|v| 2.0 * v).collect();

    // Backward pass
    // dW_o = out.T @ dy
    let out_t = transpose_f64(&out, n_rows, dim);
    let d_w_o = matmul_f64(&out_t, dim, n_rows, &dy, n_rows, dim);
    // d_out = dy @ W_o.T
    let w_o_t = transpose_f64(&w_o, dim, dim);
    let d_out = matmul_f64(&dy, n_rows, dim, &w_o_t, dim, dim);
    // dV = attn.T @ d_out
    let attn_t = transpose_f64(&attn, n_rows, n_rows);
    let d_v_act = matmul_f64(&attn_t, n_rows, n_rows, &d_out, n_rows, dim);
    // d_attn = d_out @ V.T
    let v_t = transpose_f64(&v, n_rows, dim);
    let d_attn = matmul_f64(&d_out, n_rows, dim, &v_t, dim, n_rows);

    // Softmax-Jacobian-vector-product:
    //   d_scores[i,j] = attn[i,j] * (d_attn[i,j] - sum_k(attn[i,k] * d_attn[i,k]))
    let mut d_scores = vec![0.0; n_rows * n_rows];
    for r in 0..n_rows {
        let attn_row = &attn[r * n_rows..(r + 1) * n_rows];
        let d_attn_row = &d_attn[r * n_rows..(r + 1) * n_rows];
        let dot: f64 = attn_row
            .iter()
            .zip(d_attn_row.iter())
            .map(|(a, d)| a * d)
            .sum();
        for j in 0..n_rows {
            d_scores[r * n_rows + j] = attn_row[j] * (d_attn_row[j] - dot);
        }
    }

    // dQ = d_scores @ K
    let d_q_act = matmul_f64(&d_scores, n_rows, n_rows, &k, n_rows, dim);
    // dK = d_scores.T @ Q
    let d_scores_t = transpose_f64(&d_scores, n_rows, n_rows);
    let d_k_act = matmul_f64(&d_scores_t, n_rows, n_rows, &q, n_rows, dim);

    // dW_q = x.T @ dQ;  dW_k = x.T @ dK;  dW_v = x.T @ dV
    let x_t = transpose_f64(&x, n_rows, dim);
    let d_w_q = matmul_f64(&x_t, dim, n_rows, &d_q_act, n_rows, dim);
    let d_w_k = matmul_f64(&x_t, dim, n_rows, &d_k_act, n_rows, dim);
    let d_w_v = matmul_f64(&x_t, dim, n_rows, &d_v_act, n_rows, dim);

    // Per-head reduction: aggregate per-projection per-head into a single score vec.
    // For symmetric MHA, n_q_heads == n_kv_heads == n_o_heads, so simple sum works.
    let q_scores = per_head_score(&d_w_q, &w_q, dim, head_dim);
    let k_scores = per_head_score(&d_w_k, &w_k, dim, head_dim);
    let v_scores = per_head_score(&d_w_v, &w_v, dim, head_dim);
    let o_scores = per_head_score(&d_w_o, &w_o, dim, head_dim);
    let mut combined = vec![0.0_f32; num_heads];
    for h in 0..num_heads {
        combined[h] = q_scores[h] + k_scores[h] + v_scores[h] + o_scores[h];
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
