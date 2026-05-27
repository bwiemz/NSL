//! CPU-naive backward dQ -- fully independent reference (Phase 2.6 T5).
//!
//! Takes Q/K/V/O from ForwardOutputs (caller passes fwd.q_saved/k_saved/v_saved/o).
//! Recomputes row_max/row_sum/S/P/D internally -- does NOT take saved row_max/row_sum
//! or the D-pre-pass D. This catches adapter stats-reshape + D-pre-pass bugs the
//! comparator otherwise couldn't.
//!
//! Math:
//!   S        = (1/sqrt(D_dim)) * Q @ K^T   (causal mask if cfg.causal)
//!   row_max  = max_k S[q,k]
//!   row_sum  = sum_k exp(S - row_max)
//!   P        = exp(S - row_max) / row_sum
//!   D_local  = rowsum(dO * O)
//!   dP       = dO @ V^T
//!   dS       = (1/sqrt(D_dim)) * P * (dP - D_local)
//!   dQ       = dS @ K
//!
//! Callers MUST pass fwd.{q_saved, k_saved, v_saved, o} -- NOT raw test inputs.

use half::f16;
use nsl_codegen::flash_attention::FlashAttentionConfig;

/// CPU-naive backward dQ. q/k/v/o are row-major [B,H,S,D] f16 (pass fwd fields);
/// d_o is row-major [B,H,S,D] f16. Returns dQ row-major [B,H,S,D] f32.
#[allow(clippy::too_many_arguments)]
pub fn cpu_naive_backward_dq(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    o: &[f16],
    d_o: &[f16],
    batch: usize,
    heads: usize,
    seq: usize,
    cfg: &FlashAttentionConfig,
) -> Vec<f32> {
    let d = cfg.head_dim as usize;
    let expected = batch * heads * seq * d;
    assert_eq!(q.len(), expected, "q length mismatch: got {} expected {}", q.len(), expected);
    assert_eq!(k.len(), expected, "k length mismatch");
    assert_eq!(v.len(), expected, "v length mismatch");
    assert_eq!(o.len(), expected, "o length mismatch");
    assert_eq!(d_o.len(), expected, "d_o length mismatch");

    let mut dq = vec![0.0f32; expected];
    let scale = 1.0f32 / (d as f32).sqrt();
    let causal = cfg.causal;

    for bi in 0..batch {
        for hi in 0..heads {
            for qi in 0..seq {
                let qbase = ((bi * heads + hi) * seq + qi) * d;
                let k_limit = if causal { qi + 1 } else { seq };

                let s_row: Vec<f32> = (0..seq)
                    .map(|ki| {
                        if ki >= k_limit {
                            return f32::NEG_INFINITY;
                        }
                        let kbase = ((bi * heads + hi) * seq + ki) * d;
                        let dot: f32 = (0..d)
                            .map(|di| q[qbase + di].to_f32() * k[kbase + di].to_f32())
                            .sum();
                        dot * scale
                    })
                    .collect();

                let rmax = s_row[..k_limit].iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
                let rsum: f32 = (0..k_limit).map(|ki| (s_row[ki] - rmax).exp()).sum();
                let p_row: Vec<f32> = (0..seq)
                    .map(|ki| {
                        if ki >= k_limit {
                            0.0
                        } else {
                            (s_row[ki] - rmax).exp() / rsum
                        }
                    })
                    .collect();

                let d_local: f32 = (0..d)
                    .map(|di| d_o[qbase + di].to_f32() * o[qbase + di].to_f32())
                    .sum();

                let dp_row: Vec<f32> = (0..seq)
                    .map(|ki| {
                        if ki >= k_limit {
                            return 0.0;
                        }
                        let kbase = ((bi * heads + hi) * seq + ki) * d;
                        (0..d)
                            .map(|di| d_o[qbase + di].to_f32() * v[kbase + di].to_f32())
                            .sum()
                    })
                    .collect();

                let ds_row: Vec<f32> = (0..seq)
                    .map(|ki| scale * p_row[ki] * (dp_row[ki] - d_local))
                    .collect();

                // `ki` indexes both ds_row and the flat-strided k base, so a plain
                // enumerate() would not remove the second index -- keep the range loop.
                #[allow(clippy::needless_range_loop)]
                for ki in 0..k_limit {
                    let kbase = ((bi * heads + hi) * seq + ki) * d;
                    for di in 0..d {
                        dq[qbase + di] += ds_row[ki] * k[kbase + di].to_f32();
                    }
                }
            }
        }
    }
    dq
}

/// CPU-naive backward dK/dV. q/k/v/o/d_o are row-major [B,H,S,D] f16 (pass fwd fields).
/// Returns (dV, dK), both row-major [B,H,S,D] f32.
///
/// Math (kv-outer perspective):
///   dV[kv,d] = Σ_{q >= kv (causal)}  P[q,kv] · dO[q,d]
///   dK[kv,d] = Σ_{q >= kv (causal)}  dS[q,kv] · Q[q,d]
///
/// where P and dS are recomputed from q/k/v/o identically to cpu_naive_backward_dq:
///   S[q,k]   = (1/sqrt(D)) * Q[q] · K[k]^T   (causal: k > q → masked)
///   row_max  = max_k S[q,k]
///   row_sum  = Σ_k exp(S[q,k] - row_max)
///   P[q,k]   = exp(S[q,k] - row_max) / row_sum
///   D_local  = rowsum(dO[q] * O[q])
///   dP[q,k]  = dO[q] · V[k]^T
///   dS[q,k]  = (1/sqrt(D)) * P[q,k] * (dP[q,k] - D_local)
#[allow(clippy::too_many_arguments)]
pub fn cpu_naive_backward_dkdv(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    o: &[f16],
    d_o: &[f16],
    batch: usize,
    heads: usize,
    seq: usize,
    cfg: &FlashAttentionConfig,
) -> (Vec<f32>, Vec<f32>) {
    let d = cfg.head_dim as usize;
    let expected = batch * heads * seq * d;
    assert_eq!(q.len(), expected, "q length mismatch: got {} expected {}", q.len(), expected);
    assert_eq!(k.len(), expected, "k length mismatch");
    assert_eq!(v.len(), expected, "v length mismatch");
    assert_eq!(o.len(), expected, "o length mismatch");
    assert_eq!(d_o.len(), expected, "d_o length mismatch");

    let mut dv = vec![0.0f32; expected];
    let mut dk = vec![0.0f32; expected];
    let scale = 1.0f32 / (d as f32).sqrt();
    let causal = cfg.causal;

    for bi in 0..batch {
        for hi in 0..heads {
            for qi in 0..seq {
                let qbase = ((bi * heads + hi) * seq + qi) * d;
                let k_limit = if causal { qi + 1 } else { seq };

                let s_row: Vec<f32> = (0..seq)
                    .map(|ki| {
                        if ki >= k_limit {
                            return f32::NEG_INFINITY;
                        }
                        let kbase = ((bi * heads + hi) * seq + ki) * d;
                        let dot: f32 = (0..d)
                            .map(|di| q[qbase + di].to_f32() * k[kbase + di].to_f32())
                            .sum();
                        dot * scale
                    })
                    .collect();

                let rmax = s_row[..k_limit].iter().fold(f32::NEG_INFINITY, |a, &x| a.max(x));
                let rsum: f32 = (0..k_limit).map(|ki| (s_row[ki] - rmax).exp()).sum();
                let p_row: Vec<f32> = (0..seq)
                    .map(|ki| {
                        if ki >= k_limit {
                            0.0
                        } else {
                            (s_row[ki] - rmax).exp() / rsum
                        }
                    })
                    .collect();

                let d_local: f32 = (0..d)
                    .map(|di| d_o[qbase + di].to_f32() * o[qbase + di].to_f32())
                    .sum();

                let dp_row: Vec<f32> = (0..seq)
                    .map(|ki| {
                        if ki >= k_limit {
                            return 0.0;
                        }
                        let kbase = ((bi * heads + hi) * seq + ki) * d;
                        (0..d)
                            .map(|di| d_o[qbase + di].to_f32() * v[kbase + di].to_f32())
                            .sum()
                    })
                    .collect();

                let ds_row: Vec<f32> = (0..seq)
                    .map(|ki| scale * p_row[ki] * (dp_row[ki] - d_local))
                    .collect();

                // Accumulate into the kv-indexed outputs.
                // dV[kv,d] += P[q,kv]  * dO[q,d]
                // dK[kv,d] += dS[q,kv] * Q[q,d]
                // `ki` is the loop variable for the kv dimension (ki < k_limit).
                #[allow(clippy::needless_range_loop)]
                for ki in 0..k_limit {
                    let kbase = ((bi * heads + hi) * seq + ki) * d;
                    for di in 0..d {
                        dv[kbase + di] += p_row[ki]  * d_o[qbase + di].to_f32();
                        dk[kbase + di] += ds_row[ki] * q[qbase + di].to_f32();
                    }
                }
            }
        }
    }
    (dv, dk)
}

/// CPU-naive projection backward (smoke scope: heads==1, d_model==head_dim, seq==block_q).
/// dq/dk/dv: row-major [batch,heads,seq,head_dim] f32. x_raw: [batch,heads,seq,d_model] f16.
/// wq/wk/wv: [d_model, kv_dim] f16 (kv_dim = heads*head_dim). norm_weight: [d_model] f16.
/// Returns (dWq, dWk, dWv, dx): dW* row-major [d_model, kv_dim] f32; dx [batch,heads,seq,d_model] f32.
///
/// The GPU folds the RMSNorm gain (gamma = norm_weight[p]) into x_norm BEFORE the
/// projection: `x_norm[s,p] = (x_raw[s,p] / rms[s]) * norm_weight[p]` (emit_xnorm_recompute).
/// Both gradient paths therefore carry gamma:
///   dWq[p,j] = sum_s (x_raw[s,p]/rms[s]) * norm_weight[p] * dQ[s,j]   (likewise dWk/dWv)
///   dx[s,p]  = g[p]/rms - x_raw[s,p]*sdot/(d_model*rms^3),
///              where g[p] = dx_norm[s,p]*norm_weight[p], sdot = sum_p g[p]*x_raw[s,p].
#[allow(clippy::too_many_arguments)]
pub fn cpu_naive_backward_proj(
    dq: &[f32], dk: &[f32], dv: &[f32],
    x_raw: &[f16],
    wq: &[f16], wk: &[f16], wv: &[f16],
    norm_weight: &[f16],
    eps: f32,
    batch: usize, heads: usize, seq: usize, head_dim: usize, d_model: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    assert_eq!(heads, 1, "cpu_naive_backward_proj: multi-head dx reduction not implemented; heads must be 1 (smoke scope)");
    let kv_dim = heads * head_dim;
    let mut dwq = vec![0.0f32; d_model * kv_dim];
    let mut dwk = vec![0.0f32; d_model * kv_dim];
    let mut dwv = vec![0.0f32; d_model * kv_dim];
    let mut dx = vec![0.0f32; batch * heads * seq * d_model];

    for bi in 0..batch {
        for hi in 0..heads {
            for s in 0..seq {
                let x_base = ((bi * heads + hi) * seq + s) * d_model;
                let y_base = ((bi * heads + hi) * seq + s) * head_dim;

                let mean_sq: f32 = (0..d_model)
                    .map(|p| { let v = x_raw[x_base + p].to_f32(); v * v })
                    .sum::<f32>() / d_model as f32;
                let rms = (mean_sq + eps).sqrt();

                for p in 0..d_model {
                    // x_norm = (x_raw/rms) * gamma -- the GPU folds the RMSNorm gain
                    // (norm_weight[p]) into x_norm BEFORE the projection
                    // (emit_xnorm_recompute), so the weight gradient MUST include it:
                    //   dWq[p,j] = sum_s (x_raw[s,p]/rms[s]) * norm_weight[p] * dQ[s,j]
                    // (the dx path below already applies gamma via g = dx_norm[p]*norm_weight[p];
                    // omitting it here was an internal inconsistency).
                    let xn = x_raw[x_base + p].to_f32() / rms * norm_weight[p].to_f32();
                    for j in 0..head_dim {
                        let col = hi * head_dim + j;
                        dwq[p * kv_dim + col] += xn * dq[y_base + j];
                        dwk[p * kv_dim + col] += xn * dk[y_base + j];
                        dwv[p * kv_dim + col] += xn * dv[y_base + j];
                    }
                }

                let mut dx_norm = vec![0.0f32; d_model];
                for p in 0..d_model {
                    let mut acc = 0.0f32;
                    for j in 0..head_dim {
                        let col = hi * head_dim + j;
                        acc += dq[y_base + j] * wq[p * kv_dim + col].to_f32()
                             + dk[y_base + j] * wk[p * kv_dim + col].to_f32()
                             + dv[y_base + j] * wv[p * kv_dim + col].to_f32();
                    }
                    dx_norm[p] = acc;
                }
                let sdot: f32 = (0..d_model)
                    .map(|p| dx_norm[p] * norm_weight[p].to_f32() * x_raw[x_base + p].to_f32())
                    .sum();
                for p in 0..d_model {
                    let g = dx_norm[p] * norm_weight[p].to_f32();
                    dx[x_base + p] =
                        g / rms - x_raw[x_base + p].to_f32() * sdot / (d_model as f32 * rms * rms * rms);
                }
            }
        }
    }
    (dwq, dwk, dwv, dx)
}

#[cfg(test)]
mod proj_tests {
    use super::cpu_naive_backward_proj;
    use half::f16;

    fn h(x: f32) -> f16 { f16::from_f32(x) }

    #[test]
    fn proj_dw_is_xnorm_t_times_dy_smoke() {
        // heads=1, d_model=head_dim=2, seq=1 -> single row, kv_dim=2.
        // x = [3,4] -> mean_sq = (9+16)/2 = 12.5; rms = sqrt(12.5+eps).
        //
        // NON-UNIT gamma (RMSNorm gain): gamma = [1.5, 0.5]. The GPU folds gamma
        // into x_norm BEFORE the projection (x_norm = (x/rms)*gamma), so BOTH the
        // weight-gradient and the input-gradient paths must carry gamma. With
        // gamma != 1 this test FAILS if gamma is dropped from EITHER path:
        //   dWq[p,j] = (x_raw[0,p]/rms) * gamma[p] * dq[0,j]   (gamma in dW path)
        //   dx path:  dx_norm[p] = sum_j dq[j]*Wq[p,j];  g[p] = dx_norm[p]*gamma[p]
        //             sdot = sum_p g[p]*x_raw[p];
        //             dx[p] = g[p]/rms - x_raw[p]*sdot/(d_model*rms^3)
        let d_model = 2usize;
        let head_dim = 2usize;
        let seq = 1usize;
        let eps = 1e-5f32;
        let x_raw = vec![h(3.0), h(4.0)];                 // [seq, d_model]
        let dq = vec![1.0f32, 2.0];                        // [seq, head_dim]
        let dk = vec![0.0f32, 0.0];
        let dv = vec![0.0f32, 0.0];
        // Identity Wq (row-major [d_model, kv_dim]=[2,2]); Wk=Wv=0 so dx_norm = dq @ Wq^T.
        let wq = vec![h(1.0), h(0.0), h(0.0), h(1.0)];
        let wk = vec![h(0.0); d_model * head_dim];
        let wv = vec![h(0.0); d_model * head_dim];
        let norm_weight = vec![h(1.5), h(0.5)];            // NON-UNIT gamma

        let (dwq, _dwk, _dwv, dx) = cpu_naive_backward_proj(
            &dq, &dk, &dv, &x_raw, &wq, &wk, &wv, &norm_weight,
            eps, /*batch*/1, /*heads*/1, seq, head_dim, d_model,
        );

        let rms = (12.5f32 + eps).sqrt();
        let gamma = [1.5f32, 0.5f32];
        let xr = [3.0f32, 4.0f32];
        // dWq[p,j] = (x_raw[0,p]/rms) * gamma[p] * dq[0,j]  -- gamma MUST be present.
        let xn0 = xr[0] / rms * gamma[0]; // (3/rms)*1.5
        let xn1 = xr[1] / rms * gamma[1]; // (4/rms)*0.5
        assert!((dwq[0 * 2 + 0] - xn0 * 1.0).abs() < 1e-4, "dWq[0,0]: got {} want {}", dwq[0], xn0 * 1.0);
        assert!((dwq[0 * 2 + 1] - xn0 * 2.0).abs() < 1e-4, "dWq[0,1]: got {} want {}", dwq[1], xn0 * 2.0);
        assert!((dwq[1 * 2 + 0] - xn1 * 1.0).abs() < 1e-4, "dWq[1,0]: got {} want {}", dwq[2], xn1 * 1.0);
        assert!((dwq[1 * 2 + 1] - xn1 * 2.0).abs() < 1e-4, "dWq[1,1]: got {} want {}", dwq[3], xn1 * 2.0);

        // dx path (independent recomputation; identity Wq => dx_norm = [dq0, dq1] = [1,2]).
        let dxn = [1.0f32, 2.0f32];
        let g = [dxn[0] * gamma[0], dxn[1] * gamma[1]]; // [1*1.5, 2*0.5] = [1.5, 1.0]
        let sdot = g[0] * xr[0] + g[1] * xr[1]; // 1.5*3 + 1.0*4 = 8.5
        let exp = |p: usize| g[p] / rms - xr[p] * sdot / (d_model as f32 * rms * rms * rms);
        assert!((dx[0] - exp(0)).abs() < 1e-4, "dx[0]: got {} want {}", dx[0], exp(0));
        assert!((dx[1] - exp(1)).abs() < 1e-4, "dx[1]: got {} want {}", dx[1], exp(1));
    }
}
