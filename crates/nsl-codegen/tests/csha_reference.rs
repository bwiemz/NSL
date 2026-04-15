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
// Tier C backward reference
// ---------------------------------------------------------------------------

/// Chain-rule gradients produced by `csha_reference_backward`.
///
/// Shapes (inverse of forward outputs):
///   dq, dk, dv:   [seq, heads * head_dim]
///   dwq, dwk, dwv: [d_model, heads * head_dim]
///   dx:           [seq, d_model]
///
/// All stored as row-major flat `Vec<f32>` to match the forward
/// convention in `CshaInputs`.
#[derive(Debug, Clone)]
pub struct CshaGradients {
    pub dq: Vec<f32>,
    pub dk: Vec<f32>,
    pub dv: Vec<f32>,
    pub dwq: Vec<f32>,
    pub dwk: Vec<f32>,
    pub dwv: Vec<f32>,
    pub dx: Vec<f32>,
}

/// All forward intermediate tensors needed by the backward pass.
/// Re-computed from inputs rather than cached on CshaInputs so the
/// forward reference stays stateless.
struct Intermediates {
    x_norm: Vec<f32>,          // [seq, d_model]
    rms: Vec<f32>,             // [seq]
    q: Vec<f32>,               // [seq, heads*head_dim], post-RoPE
    k: Vec<f32>,
    v: Vec<f32>,
    p_per_head: Vec<Vec<f32>>, // heads × [seq, seq]
}

/// Compute the full set of forward intermediates. Keeps the chain-rule
/// backward arithmetically closed-form (no numerical re-derivation).
fn forward_intermediates(inputs: &CshaInputs<'_>, shape: &CshaShape) -> Intermediates {
    let CshaShape { seq, heads, head_dim, d_model, causal, norm_eps } = *shape;
    let kv_dim = heads * head_dim;

    let mut x_norm = Vec::with_capacity(seq * d_model);
    let mut rms = Vec::with_capacity(seq);
    for s in 0..seq {
        let row = &inputs.x[s * d_model..(s + 1) * d_model];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / row.len() as f32;
        let r = (mean_sq + norm_eps).sqrt();
        rms.push(r);
        x_norm.extend(row.iter().zip(inputs.norm_weight).map(|(v, w)| v / r * w));
    }

    let mut q = matmul(&x_norm, inputs.wq, seq, d_model, kv_dim);
    let mut k = matmul(&x_norm, inputs.wk, seq, d_model, kv_dim);
    let v = matmul(&x_norm, inputs.wv, seq, d_model, kv_dim);

    apply_rope(&mut q, seq, heads, head_dim, inputs.cos, inputs.sin);
    apply_rope(&mut k, seq, heads, head_dim, inputs.cos, inputs.sin);

    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let mut p_per_head = Vec::with_capacity(heads);
    for h in 0..heads {
        let mut s_mat = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[(i * kv_dim) + h * head_dim + d]
                         * k[(j * kv_dim) + h * head_dim + d];
                }
                s_mat[i * seq + j] = dot * scale;
            }
        }
        if causal {
            for i in 0..seq {
                for j in (i + 1)..seq {
                    s_mat[i * seq + j] = f32::NEG_INFINITY;
                }
            }
        }
        softmax_rows(&mut s_mat, seq, seq);
        p_per_head.push(s_mat);
    }

    Intermediates { x_norm, rms, q, k, v, p_per_head }
}

/// Reverse-mode backward through the full CSHA chain: O → (dQ, dK, dV,
/// dWq, dWk, dWv, dx). Used as the debugging ground truth for the
/// fused PTX backward kernel in Tier C.
///
/// `do_out` is the upstream gradient w.r.t. `O` in `[seq, heads*head_dim]`
/// layout, matching the forward output shape.
pub fn csha_reference_backward(
    inputs: &CshaInputs<'_>,
    shape: &CshaShape,
    do_out: &[f32],
) -> CshaGradients {
    let CshaShape { seq, heads, head_dim, d_model, causal: _, norm_eps: _ } = *shape;
    let kv_dim = heads * head_dim;
    let inter = forward_intermediates(inputs, shape);
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Per-head accumulators for dQ_post_rope, dK_post_rope, dV.
    let mut d_q = vec![0.0f32; seq * kv_dim];
    let mut d_k = vec![0.0f32; seq * kv_dim];
    let mut d_v = vec![0.0f32; seq * kv_dim];

    for h in 0..heads {
        let p = &inter.p_per_head[h];

        // dP = dO_h @ V_h^T   (shape [seq, seq])
        let mut d_p = vec![0.0f32; seq * seq];
        for i in 0..seq {
            for j in 0..seq {
                let mut acc = 0.0f32;
                for d in 0..head_dim {
                    acc += do_out[i * kv_dim + h * head_dim + d]
                         * inter.v[j * kv_dim + h * head_dim + d];
                }
                d_p[i * seq + j] = acc;
            }
        }

        // dS[i,j] = P[i,j] * (dP[i,j] - sum_k(dP[i,k] * P[i,k]))
        let mut d_s = vec![0.0f32; seq * seq];
        for i in 0..seq {
            let mut row_sum = 0.0f32;
            for k in 0..seq {
                row_sum += d_p[i * seq + k] * p[i * seq + k];
            }
            for j in 0..seq {
                d_s[i * seq + j] = p[i * seq + j] * (d_p[i * seq + j] - row_sum);
            }
        }

        // dV_h[j,d] += sum_i P[i,j] * dO[i,d]
        for j in 0..seq {
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for i in 0..seq {
                    acc += p[i * seq + j] * do_out[i * kv_dim + h * head_dim + d];
                }
                d_v[j * kv_dim + h * head_dim + d] += acc;
            }
        }

        // S = Q @ K^T * scale   →   dQ += dS*scale @ K;  dK += (dS*scale)^T @ Q
        for i in 0..seq {
            for d in 0..head_dim {
                let mut acc_q = 0.0f32;
                for j in 0..seq {
                    acc_q += d_s[i * seq + j] * inter.k[j * kv_dim + h * head_dim + d];
                }
                d_q[i * kv_dim + h * head_dim + d] += acc_q * scale;
            }
        }
        for j in 0..seq {
            for d in 0..head_dim {
                let mut acc_k = 0.0f32;
                for i in 0..seq {
                    acc_k += d_s[i * seq + j] * inter.q[i * kv_dim + h * head_dim + d];
                }
                d_k[j * kv_dim + h * head_dim + d] += acc_k * scale;
            }
        }
    }

    // Reverse RoPE on dQ and dK (V untouched). Pair layout = Adjacent.
    // forward: y0 = x0*cos - x1*sin; y1 = x0*sin + x1*cos
    // inverse (since rotation is orthogonal): dx0 = dy0*cos + dy1*sin
    //                                         dx1 = -dy0*sin + dy1*cos
    let half = head_dim / 2;
    for tensor in [&mut d_q, &mut d_k] {
        for s in 0..seq {
            for h in 0..heads {
                let base = s * kv_dim + h * head_dim;
                for pair in 0..half {
                    let cos_v = inputs.cos[s * half + pair];
                    let sin_v = inputs.sin[s * half + pair];
                    let y0 = tensor[base + 2 * pair];
                    let y1 = tensor[base + 2 * pair + 1];
                    tensor[base + 2 * pair]     =  y0 * cos_v + y1 * sin_v;
                    tensor[base + 2 * pair + 1] = -y0 * sin_v + y1 * cos_v;
                }
            }
        }
    }
    // d_q, d_k now hold dQ_pre_rope, dK_pre_rope.

    // dWq = x_norm^T @ dQ_pre_rope   (shape [d_model, kv_dim])
    // dWk, dWv similarly.
    let mut d_wq = vec![0.0f32; d_model * kv_dim];
    let mut d_wk = vec![0.0f32; d_model * kv_dim];
    let mut d_wv = vec![0.0f32; d_model * kv_dim];
    for p in 0..d_model {
        for j in 0..kv_dim {
            let mut aq = 0.0f32;
            let mut ak = 0.0f32;
            let mut av = 0.0f32;
            for s in 0..seq {
                let xn = inter.x_norm[s * d_model + p];
                aq += xn * d_q[s * kv_dim + j];
                ak += xn * d_k[s * kv_dim + j];
                av += xn * d_v[s * kv_dim + j];
            }
            d_wq[p * kv_dim + j] = aq;
            d_wk[p * kv_dim + j] = ak;
            d_wv[p * kv_dim + j] = av;
        }
    }

    // dx_norm = dQ_pre_rope @ Wq^T + dK_pre_rope @ Wk^T + dV @ Wv^T
    // Shape: [seq, d_model].
    let mut d_x_norm = vec![0.0f32; seq * d_model];
    for s in 0..seq {
        for p in 0..d_model {
            let mut acc = 0.0f32;
            for j in 0..kv_dim {
                acc += d_q[s * kv_dim + j] * inputs.wq[p * kv_dim + j]
                     + d_k[s * kv_dim + j] * inputs.wk[p * kv_dim + j]
                     + d_v[s * kv_dim + j] * inputs.wv[p * kv_dim + j];
            }
            d_x_norm[s * d_model + p] = acc;
        }
    }

    // Reverse RMSNorm. Forward:
    //   x_norm[i,d] = x[i,d] * (1/rms[i]) * norm_weight[d]
    //   rms[i] = sqrt(mean_sq[i] + eps); mean_sq[i] = (1/D) sum_d x[i,d]^2
    // Backward (row i, dim D = d_model):
    //   Let g_d = dx_norm[i,d] * norm_weight[d]
    //   s_grad = sum_d g_d * x[i,d]
    //   dx[i,d] = g_d / rms[i] - x[i,d] * s_grad / (D * rms[i]^3)
    let mut d_x = vec![0.0f32; seq * d_model];
    for i in 0..seq {
        let r = inter.rms[i];
        let r3 = r * r * r;
        let mut s_grad = 0.0f32;
        for d in 0..d_model {
            let g_d = d_x_norm[i * d_model + d] * inputs.norm_weight[d];
            s_grad += g_d * inputs.x[i * d_model + d];
        }
        for d in 0..d_model {
            let g_d = d_x_norm[i * d_model + d] * inputs.norm_weight[d];
            d_x[i * d_model + d] = g_d / r - inputs.x[i * d_model + d] * s_grad / (d_model as f32 * r3);
        }
    }

    CshaGradients { dq: d_q, dk: d_k, dv: d_v, dwq: d_wq, dwk: d_wk, dwv: d_wv, dx: d_x }
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

    // ── T6.1 / T6.2 backward reference tests ──────────────────────────────

    #[test]
    fn csha_gradients_struct_has_seven_fields() {
        let g = CshaGradients {
            dq: vec![0.0],
            dk: vec![0.0],
            dv: vec![0.0],
            dwq: vec![0.0],
            dwk: vec![0.0],
            dwv: vec![0.0],
            dx: vec![0.0],
        };
        assert_eq!(g.dq.len(), 1);
        assert_eq!(g.dwv.len(), 1);
        assert_eq!(g.dx.len(), 1);
    }

    /// Sum(O) → upstream dO = ones. Verify all seven gradients come back
    /// with the expected shape and are finite on a deterministic input.
    /// Numerical correctness is validated by the finite-difference
    /// gradcheck test below.
    #[test]
    fn csha_reference_backward_shape_and_finiteness() {
        let shape = CshaShape {
            seq: 8,
            heads: 2,
            head_dim: 16,
            d_model: 32,
            causal: true,
            norm_eps: 1e-5,
        };
        let kv_dim = shape.heads * shape.head_dim;
        let x = det_seq(42, shape.seq * shape.d_model);
        let wq = det_seq(43, shape.d_model * kv_dim);
        let wk = det_seq(44, shape.d_model * kv_dim);
        let wv = det_seq(45, shape.d_model * kv_dim);
        let norm_weight = vec![1.0f32; shape.d_model];
        let cos = det_seq(46, shape.seq * shape.head_dim / 2);
        let sin = det_seq(47, shape.seq * shape.head_dim / 2);
        let inputs = CshaInputs {
            x: &x, wq: &wq, wk: &wk, wv: &wv,
            norm_weight: &norm_weight, cos: &cos, sin: &sin,
        };
        let do_out = vec![1.0f32; shape.seq * kv_dim];

        let g = csha_reference_backward(&inputs, &shape, &do_out);

        assert_eq!(g.dq.len(), shape.seq * kv_dim);
        assert_eq!(g.dk.len(), shape.seq * kv_dim);
        assert_eq!(g.dv.len(), shape.seq * kv_dim);
        assert_eq!(g.dwq.len(), shape.d_model * kv_dim);
        assert_eq!(g.dwk.len(), shape.d_model * kv_dim);
        assert_eq!(g.dwv.len(), shape.d_model * kv_dim);
        assert_eq!(g.dx.len(), shape.seq * shape.d_model);

        for (name, arr) in [
            ("dq", &g.dq), ("dk", &g.dk), ("dv", &g.dv),
            ("dwq", &g.dwq), ("dwk", &g.dwk), ("dwv", &g.dwv),
            ("dx", &g.dx),
        ] {
            for (i, &v) in arr.iter().enumerate() {
                assert!(v.is_finite(), "{name}[{i}] = {v} not finite");
            }
            // At least one non-zero entry — all-zero would mean the
            // backward path is silently dropping gradients.
            assert!(arr.iter().any(|&v| v.abs() > 0.0),
                "{name} is all-zero; backward chain may be broken");
        }
    }

    /// Finite-difference gradcheck on a small 2-head config. Numerically
    /// perturbs a handful of input entries and compares the analytical
    /// gradients from `csha_reference_backward` against central-difference
    /// estimates. This is the real correctness test — if the backward
    /// pass is algebraically wrong, this catches it at f32 precision
    /// (max_abs_err ≲ 1e-3 is expected; scalar sum-of-O loss is O(1)).
    #[test]
    fn csha_reference_backward_finite_difference_gradcheck() {
        let shape = CshaShape {
            seq: 3,
            heads: 1,
            head_dim: 4,
            d_model: 4,
            causal: false,
            norm_eps: 1e-5,
        };
        let kv_dim = shape.heads * shape.head_dim;
        let mut x = det_seq(7, shape.seq * shape.d_model);
        let mut wq = det_seq(8, shape.d_model * kv_dim);
        let mut wk = det_seq(9, shape.d_model * kv_dim);
        let wv = det_seq(10, shape.d_model * kv_dim);
        let norm_weight = vec![1.0f32; shape.d_model];
        // Non-trivial RoPE angles.
        let cos: Vec<f32> = (0..shape.seq * shape.head_dim / 2)
            .map(|i| ((i as f32) * 0.1).cos()).collect();
        let sin: Vec<f32> = (0..shape.seq * shape.head_dim / 2)
            .map(|i| ((i as f32) * 0.1).sin()).collect();
        let do_out = vec![1.0f32; shape.seq * kv_dim]; // loss = sum(O)

        // Compute analytical gradients once.
        let g = {
            let inputs = CshaInputs {
                x: &x, wq: &wq, wk: &wk, wv: &wv,
                norm_weight: &norm_weight, cos: &cos, sin: &sin,
            };
            csha_reference_backward(&inputs, &shape, &do_out)
        };

        // Loss helper: sum of forward output (dO = ones → dL/dparam == g.*).
        let eps = 1e-3f32;
        let loss_of = |x_: &[f32], wq_: &[f32], wk_: &[f32]| {
            let inputs = CshaInputs {
                x: x_, wq: wq_, wk: wk_, wv: &wv,
                norm_weight: &norm_weight, cos: &cos, sin: &sin,
            };
            csha_reference(&inputs, &shape).iter().sum::<f32>()
        };

        // Probe a handful of positions per tensor.
        let probes_x = [0, 3, 5, shape.seq * shape.d_model - 1];
        let probes_w = [0, 2, 7, shape.d_model * kv_dim - 1];

        for &i in &probes_x {
            let saved = x[i];
            x[i] = saved + eps;
            let lp = loss_of(&x, &wq, &wk);
            x[i] = saved - eps;
            let lm = loss_of(&x, &wq, &wk);
            x[i] = saved;
            let fd = (lp - lm) / (2.0 * eps);
            let diff = (fd - g.dx[i]).abs();
            assert!(
                diff < 1e-2,
                "dx[{i}]: analytical={}, fd={}, diff={}",
                g.dx[i], fd, diff
            );
        }
        for &i in &probes_w {
            let saved = wq[i];
            wq[i] = saved + eps;
            let lp = loss_of(&x, &wq, &wk);
            wq[i] = saved - eps;
            let lm = loss_of(&x, &wq, &wk);
            wq[i] = saved;
            let fd = (lp - lm) / (2.0 * eps);
            let diff = (fd - g.dwq[i]).abs();
            assert!(
                diff < 1e-2,
                "dwq[{i}]: analytical={}, fd={}, diff={}",
                g.dwq[i], fd, diff
            );
        }
        for &i in &probes_w {
            let saved = wk[i];
            wk[i] = saved + eps;
            let lp = loss_of(&x, &wq, &wk);
            wk[i] = saved - eps;
            let lm = loss_of(&x, &wq, &wk);
            wk[i] = saved;
            let fd = (lp - lm) / (2.0 * eps);
            let diff = (fd - g.dwk[i]).abs();
            assert!(
                diff < 1e-2,
                "dwk[{i}]: analytical={}, fd={}, diff={}",
                g.dwk[i], fd, diff
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
