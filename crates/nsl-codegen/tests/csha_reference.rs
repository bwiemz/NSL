//! Test-only CPU reference for CSHA fused computation.
//!
//! Computation order (must match v2 kernel's fused-output-disabled path):
//!     RMSNorm(x) -> x_norm
//!     x_norm @ Wq -> Q; x_norm @ Wk -> K; x_norm @ Wv -> V
//!     RoPE(Q, cos, sin); RoPE(K, cos, sin)
//!     S = Q @ K^T / sqrt(head_dim)
//!     if causal: mask upper triangle
//!     P = softmax(S) row-wise
//!     O = P @ V
//!
//! Wo + residual are OUT OF SCOPE per spec §9a (SMEM budget). The kernel
//! emits O directly; any Wo matmul + residual add is user-side NSL.
//!
//! ANY change to the v2 kernel phase order must update this file in the
//! same commit.

pub struct CshaInputs<'a> {
    pub x: &'a [f32],           // [seq, d_model]
    pub wq: &'a [f32],          // [d_model, head_dim * heads]
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    pub norm_weight: &'a [f32], // [d_model]
    pub cos: &'a [f32],         // [seq, head_dim/2]
    pub sin: &'a [f32],         // [seq, head_dim/2]
}

pub struct CshaShape {
    pub seq: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub d_model: usize,
    pub causal: bool,
    pub norm_eps: f32,
}

/// Apply RMSNorm to a single row of length `d_model`.
/// Returns a new vec of the same length.
fn rmsnorm_row(row: &[f32], weight: &[f32], eps: f32) -> Vec<f32> {
    let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
    let rms = (mean_sq + eps).sqrt();
    row.iter()
        .zip(weight.iter())
        .map(|(v, w)| v / rms * w)
        .collect()
}

/// Matrix multiply: C = A @ B
/// A: [m, k], B: [k, n] -> C: [m, n]  (all row-major flat)
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut sum = 0.0f32;
            for p in 0..k {
                sum += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = sum;
        }
    }
    c
}

/// Apply RoPE in-place to a tensor of shape [seq, heads, head_dim].
/// cos/sin have shape [seq, head_dim/2].
///
/// **Implements `RopeStyle::Adjacent` (GPT-NeoX / GPT-J layout) ONLY.**
/// Pair `i` rotates `(x[2i], x[2i+1])`.  This matches `emit_rope_pair_sweep`
/// in `csha_hooks.rs`.  Do NOT call this function for `RopeStyle::HalfSplit`
/// (LLaMA / Qwen: `x[i]` paired with `x[i + head_dim/2]`) — the kernel and
/// CPU reference would diverge silently.
///
/// Rotation formula (matching v2 kernel §A.2.4):
///   new_x[2i]   = x[2i]*cos[i] - x[2i+1]*sin[i]
///   new_x[2i+1] = x[2i]*sin[i] + x[2i+1]*cos[i]
fn apply_rope(q: &mut [f32], seq: usize, heads: usize, head_dim: usize, cos: &[f32], sin: &[f32]) {
    // Consistency guard: this function only implements Adjacent layout.
    // If head_dim is 0 somehow, the loop is a no-op — but flag odd configs.
    debug_assert!(head_dim > 0 && head_dim % 2 == 0, "apply_rope requires even head_dim > 0");
    let half = head_dim / 2;
    for s in 0..seq {
        for h in 0..heads {
            let base = (s * heads + h) * head_dim;
            for pair in 0..half {
                let cos_val = cos[s * half + pair];
                let sin_val = sin[s * half + pair];
                let x0 = q[base + 2 * pair];
                let x1 = q[base + 2 * pair + 1];
                q[base + 2 * pair]     = x0 * cos_val - x1 * sin_val;
                q[base + 2 * pair + 1] = x0 * sin_val + x1 * cos_val;
            }
        }
    }
}

/// Row-wise softmax over a [rows, cols] matrix in-place.
fn softmax_rows(s: &mut [f32], rows: usize, cols: usize) {
    for i in 0..rows {
        let row = &mut s[i * cols..(i + 1) * cols];
        let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - max).exp();
            sum += *v;
        }
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// CPU reference for CSHA fused computation.
///
/// Returns O of shape `[seq, heads * head_dim]` — the pre-Wo attention output.
pub fn csha_reference(inputs: &CshaInputs<'_>, shape: &CshaShape) -> Vec<f32> {
    let CshaShape { seq, heads, head_dim, d_model, causal, norm_eps } = *shape;
    let kv_dim = heads * head_dim;

    // Step 1: RMSNorm(x) -> x_norm   shape [seq, d_model]
    let mut x_norm = Vec::with_capacity(seq * d_model);
    for s in 0..seq {
        let row = &inputs.x[s * d_model..(s + 1) * d_model];
        x_norm.extend(rmsnorm_row(row, inputs.norm_weight, norm_eps));
    }

    // Step 2: x_norm @ Wq/Wk/Wv -> Q, K, V   each [seq, kv_dim]
    let mut q = matmul(&x_norm, inputs.wq, seq, d_model, kv_dim);
    let mut k = matmul(&x_norm, inputs.wk, seq, d_model, kv_dim);
    let     v = matmul(&x_norm, inputs.wv, seq, d_model, kv_dim);

    // Reshape Q, K from [seq, kv_dim] to [seq, heads, head_dim] is logical;
    // the flat layout [seq * heads * head_dim] is the same either way.

    // Step 3: RoPE(Q, cos, sin); RoPE(K, cos, sin)
    apply_rope(&mut q, seq, heads, head_dim, inputs.cos, inputs.sin);
    apply_rope(&mut k, seq, heads, head_dim, inputs.cos, inputs.sin);

    // Step 4a: S = Q @ K^T / sqrt(head_dim)  per head, shape [seq, seq]
    // Step 4b: mask upper triangle if causal
    // Step 4c: softmax(S) row-wise
    // Step 4d: O_head = P @ V_head
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut out = vec![0.0f32; seq * kv_dim];

    for h in 0..heads {
        // Extract Q_h, K_h, V_h: each [seq, head_dim]
        let mut q_h = vec![0.0f32; seq * head_dim];
        let mut k_h = vec![0.0f32; seq * head_dim];
        let mut v_h = vec![0.0f32; seq * head_dim];
        for s in 0..seq {
            let q_base = s * kv_dim + h * head_dim;
            let k_base = s * kv_dim + h * head_dim;
            let v_base = s * kv_dim + h * head_dim;
            q_h[s * head_dim..(s + 1) * head_dim].copy_from_slice(&q[q_base..q_base + head_dim]);
            k_h[s * head_dim..(s + 1) * head_dim].copy_from_slice(&k[k_base..k_base + head_dim]);
            v_h[s * head_dim..(s + 1) * head_dim].copy_from_slice(&v[v_base..v_base + head_dim]);
        }

        // S = Q_h @ K_h^T * scale  [seq, seq]
        let mut s_mat = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q_h[i * head_dim + d] * k_h[j * head_dim + d];
                }
                s_mat[i * seq + j] = dot * scale;
            }
        }

        // Causal mask: upper triangle -> -inf
        if causal {
            for i in 0..seq {
                for j in (i + 1)..seq {
                    s_mat[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }

        // P = softmax(S) row-wise
        softmax_rows(&mut s_mat, seq, seq);

        // O_h = P @ V_h  [seq, head_dim]
        let o_h = matmul(&s_mat, &v_h, seq, seq, head_dim);

        // Write O_h into output at [s, h * head_dim .. (h+1) * head_dim]
        for s in 0..seq {
            let out_base = s * kv_dim + h * head_dim;
            out[out_base..out_base + head_dim]
                .copy_from_slice(&o_h[s * head_dim..(s + 1) * head_dim]);
        }
    }

    out
}

// ---------------------------------------------------------------------------
// Helpers shared by tests below (also used by C2's csha_cuda_launch_fused.rs
// via `#[path = "csha_reference.rs"] mod csha_reference;`).
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random sequence via a linear congruential generator.
pub fn det_seq(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as f32 / 65535.0) - 0.5
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn golden_value_4x4_attention_identity_weights() {
        // Deterministic tiny input: seq=4, heads=1, head_dim=4, d_model=4.
        // Identity weights + zero RoPE angles -> reference reduces to
        // softmax(x @ x^T / sqrt(d)) @ x with unit RMSNorm scaling.
        let shape = CshaShape {
            seq: 4,
            heads: 1,
            head_dim: 4,
            d_model: 4,
            causal: false,
            norm_eps: 1e-5,
        };
        #[rustfmt::skip]
        let x = [
            1.0f32, 0.0, 0.0, 0.0,
            0.0,    1.0, 0.0, 0.0,
            0.0,    0.0, 1.0, 0.0,
            0.0,    0.0, 0.0, 1.0,
        ];
        #[rustfmt::skip]
        let eye = [
            1.0f32, 0.0, 0.0, 0.0,
            0.0,    1.0, 0.0, 0.0,
            0.0,    0.0, 1.0, 0.0,
            0.0,    0.0, 0.0, 1.0,
        ];
        let ones = [1.0f32; 4];
        let cos = [1.0f32; 4 * 2]; // cos=1
        let sin = [0.0f32; 4 * 2]; // sin=0 -> RoPE is identity

        let inputs = CshaInputs {
            x: &x,
            wq: &eye,
            wk: &eye,
            wv: &eye,
            norm_weight: &ones,
            cos: &cos,
            sin: &sin,
        };
        let out = csha_reference(&inputs, &shape);

        // Golden values captured from first deterministic run on this machine.
        // With identity weights + RMSNorm of unit vectors + zero RoPE:
        //   x_norm row i = x_i / rms(x_i) * 1  (unit vectors -> rms = 1/sqrt(d=4))
        //   Q=K=V=x_norm  (identity Wq/Wk/Wv)
        //   S_ij = Q_i . K_j / sqrt(4)
        //   softmax(S) -> 4x4 attention weights
        //   O = P @ V
        // softmax([2, 0, 0, 0]) ≈ [0.711, 0.096, 0.096, 0.096] per row
        // (diagonal-dominant because S = 2*I, off-diagonal = 0)
        // O[i,i] = 2 * softmax_diag ≈ 1.422, O[i,j≠i] = 2 * softmax_off ≈ 0.193
        #[rustfmt::skip]
        let expected: [f32; 16] = [
            1.4224076, 0.19251737, 0.19251737, 0.19251737,
            0.19251737, 1.4224076, 0.19251737, 0.19251737,
            0.19251738, 0.19251738, 1.4224077, 0.19251738,
            0.19251738, 0.19251738, 0.19251738, 1.4224077,
        ];
        assert_eq!(out.len(), 16);
        for (i, (a, b)) in out.iter().zip(expected.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "index {i}: got {a}, expected {b}"
            );
        }
    }

    #[test]
    fn csha_reference_runs_for_matrix_shape() {
        let shape = CshaShape {
            seq: 8,
            heads: 4,
            head_dim: 64,
            d_model: 128,
            causal: true,
            norm_eps: 1e-5,
        };
        let x = det_seq(42, shape.seq * shape.d_model);
        let w_size = shape.d_model * shape.heads * shape.head_dim;
        let wq = det_seq(43, w_size);
        let wk = det_seq(44, w_size);
        let wv = det_seq(45, w_size);
        let norm_weight = vec![1.0f32; shape.d_model];
        let cos = det_seq(46, shape.seq * shape.head_dim / 2);
        let sin = det_seq(47, shape.seq * shape.head_dim / 2);

        let inputs = CshaInputs {
            x: &x,
            wq: &wq,
            wk: &wk,
            wv: &wv,
            norm_weight: &norm_weight,
            cos: &cos,
            sin: &sin,
        };
        let out = csha_reference(&inputs, &shape);
        assert_eq!(out.len(), shape.seq * shape.heads * shape.head_dim);
        // Sanity: no NaNs, no infs.
        for (i, v) in out.iter().enumerate() {
            assert!(v.is_finite(), "out[{i}] = {v} is not finite");
        }
    }
}
