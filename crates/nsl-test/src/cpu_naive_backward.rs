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
