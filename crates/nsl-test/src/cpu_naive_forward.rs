//! CPU-naive forward reference for Phase 2.5's standalone dQ-kernel validation.
//!
//! Implements standard attention forward from raw (q, k, v), producing
//! row_max / row_sum / O. INDEPENDENT of the dQ-kernel's inline P recompute --
//! does NOT share code with the kernel; this is the trustworthiness pin from
//! Phase 2.5 spec s2.2.
//!
//! Output layout: row-major [B, H, S, D] f16 for O; [B, H, S] f32 for
//! row_max / row_sum (the Phase 2.5 backward-input convention from spec s2.1).
//!
//! row_sum is the SUM, not the reciprocal -- Sigma_k exp(S - row_max). The kernel
//! divides at use site.

use half::f16;

/// Outputs of the CPU-naive attention forward pass.
///
/// Layout invariants (per Phase 2.5 spec s2.1):
/// - `row_max`: f32 [B, H, S] — per-query max logit (used as softmax stabilizer)
/// - `row_sum`: f32 [B, H, S] — sum of exp(S - row_max) over attended keys (NOT reciprocal)
/// - `o`:       f16 [B, H, S, D] row-major — attention output
pub struct ForwardOutputs {
    /// Forward inputs as the kernel sees them, row-major [B,H,S,D] f16.
    /// CPU-naive path: direct copies of test inputs. B.1 path: adapter reshape.
    pub q_saved: Vec<f16>,
    pub k_saved: Vec<f16>,
    pub v_saved: Vec<f16>,
    pub row_max: Vec<f32>,
    pub row_sum: Vec<f32>,
    pub o:       Vec<f16>,
}

/// CPU-naive attention forward.
///
/// Computes standard scaled dot-product attention:
///   S[q, k] = (1/sqrt(D)) * sum_d q[q,d] * k[k,d]   (f32 throughout)
///   row_max[q] = max_k S[q, k]
///   row_sum[q] = sum_k exp(S[q,k] - row_max[q])
///   P[q, k]   = exp(S[q,k] - row_max[q]) / row_sum[q]
///   O[q, d]   = sum_k P[q,k] * v[k,d]   (then cast to f16)
///
/// When `causal=true`, query position qi only attends to key positions 0..=qi.
///
/// # Arguments
/// - `q`, `k`, `v`: f16 slices in row-major [B, H, S, D] order
/// - `batch`, `heads`, `seq`, `hd`: tensor dimensions
/// - `causal`: if true, mask upper triangle (k > qi is excluded)
#[allow(clippy::too_many_arguments)]
pub fn cpu_naive_forward(
    q: &[f16],
    k: &[f16],
    v: &[f16],
    batch: usize,
    heads: usize,
    seq: usize,
    hd: usize,
    causal: bool,
) -> ForwardOutputs {
    assert_eq!(q.len(), batch * heads * seq * hd, "q size mismatch");
    assert_eq!(k.len(), batch * heads * seq * hd, "k size mismatch");
    assert_eq!(v.len(), batch * heads * seq * hd, "v size mismatch");

    let scale = 1.0f32 / (hd as f32).sqrt();
    let mut row_max = vec![0.0f32; batch * heads * seq];
    let mut row_sum = vec![0.0f32; batch * heads * seq];
    let mut o       = vec![f16::ZERO; batch * heads * seq * hd];

    for b in 0..batch {
        for h in 0..heads {
            for qi in 0..seq {
                let q_base  = ((b * heads + h) * seq + qi) * hd;
                // Causal: only attend to key positions 0..=qi
                let k_limit = if causal { qi + 1 } else { seq };

                // Step 1: compute scaled dot-product S[qi, ki] in f32
                let mut s_row = vec![f32::NEG_INFINITY; seq];
                for (ki, s_slot) in s_row.iter_mut().enumerate().take(k_limit) {
                    let k_base = ((b * heads + h) * seq + ki) * hd;
                    let mut s = 0.0f32;
                    for di in 0..hd {
                        s += q[q_base + di].to_f32() * k[k_base + di].to_f32();
                    }
                    *s_slot = s * scale;
                }

                // Step 2: row_max = max over attended positions
                let rmax = s_row[..k_limit]
                    .iter()
                    .fold(f32::NEG_INFINITY, |a, &x| a.max(x));
                row_max[(b * heads + h) * seq + qi] = rmax;

                // Step 3: row_sum = sum_k exp(S[qi,ki] - rmax)
                let mut rsum = 0.0f32;
                for &s in s_row.iter().take(k_limit) {
                    rsum += (s - rmax).exp();
                }
                row_sum[(b * heads + h) * seq + qi] = rsum;

                // Step 4: O[qi, di] = sum_k P[qi,ki] * v[ki, di]
                for di in 0..hd {
                    let mut acc = 0.0f32;
                    for (ki, &s) in s_row.iter().enumerate().take(k_limit) {
                        let p = (s - rmax).exp() / rsum;
                        let k_base = ((b * heads + h) * seq + ki) * hd;
                        acc += p * v[k_base + di].to_f32();
                    }
                    o[q_base + di] = f16::from_f32(acc);
                }
            }
        }
    }

    ForwardOutputs {
        q_saved: q.to_vec(),
        k_saved: k.to_vec(),
        v_saved: v.to_vec(),
        row_max,
        row_sum,
        o,
    }
}
