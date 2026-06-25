//! Shared CPU f64 reference for `fused_linear_ce` forward + backward.
//!
//! ## Motivation
//! Multiple integration tests (`fused_linear_ce_numerical`,
//! `fused_linear_ce_fp16_numerical`, `fused_linear_ce_bf16_numerical`,
//! `fused_linear_ce_large_vocab_numerical`) each duplicate a structurally
//! identical CPU reference: a row-major `(row, vi, hi)` triple loop for
//! logits followed by per-row `max + sum_exp + ln` for the LSE, and
//! `dlogits = softmax - one_hot(target)` for the backward gradient.
//!
//! This module lifts that reference into a single helper computing **all**
//! gradients (`dx`, `dW`, `dbias`) in pure f64. Callers cast their f32/bf16/
//! fp16 host buffers to f64 (after any low-precision rounding) before
//! invoking; the helper itself is precision-agnostic and uses no ndarray
//! dependency — just plain `Vec<f64>` buffers.
//!
//! ## Conventions
//! * `x` is row-major `[B*S, H]`.
//! * `w` is row-major `[V, H]` (each row is one vocab entry's weight vector).
//! * `bias` is `[V]`.
//! * `targets` is `[B*S]` `i32`; `-100` means "ignore this row" (no
//!   contribution to mean_loss, dx/dW/dbias gradients for that row remain
//!   exactly zero).
//! * `mean_loss = sum_over_valid(loss_per_row) / max(num_valid, 1)`.
//! * Backward uses `grad_loss` as the upstream cotangent; for a loss
//!   that the caller will scale by `1 / num_valid`, the per-logit
//!   gradient is `(softmax_v - one_hot_v) * (grad_loss / num_valid)`.

#![allow(dead_code)]

pub const IGNORE_INDEX: i32 = -100;

/// Result of the forward pass.
pub struct CpuLceForward {
    /// Per-row NLL loss (`-log_softmax(logits)[target]`). Zero for ignored rows.
    pub loss_per_row: Vec<f64>,
    /// Per-row log-sum-exp of the logits. Zero for ignored rows.
    pub lse_per_row: Vec<f64>,
    /// Mean loss over rows whose target is not `IGNORE_INDEX`.
    pub mean_loss: f64,
}

/// Result of the backward pass.
pub struct CpuLceBackward {
    /// `dx` flat `[B*S, H]`.
    pub dx: Vec<f64>,
    /// `dW` flat `[V, H]`.
    pub dw: Vec<f64>,
    /// `dbias` flat `[V]`.
    pub dbias: Vec<f64>,
}

/// Forward reference: full f64 logits, per-row LSE, per-row NLL, masked mean.
///
/// Inputs are slices of length `b_times_s*h`, `v*h`, `v`, `b_times_s`.
/// Panics on size mismatch — these are test helpers, not production code.
pub fn cpu_lce_forward_f64(
    x: &[f64],
    w: &[f64],
    bias: &[f64],
    targets: &[i32],
    b_times_s: usize,
    v: usize,
    h: usize,
) -> CpuLceForward {
    assert_eq!(x.len(), b_times_s * h, "x length mismatch");
    assert_eq!(w.len(), v * h, "w length mismatch");
    assert_eq!(bias.len(), v, "bias length mismatch");
    assert_eq!(targets.len(), b_times_s, "targets length mismatch");

    let mut loss_per_row = vec![0f64; b_times_s];
    let mut lse_per_row = vec![0f64; b_times_s];
    let mut total_loss = 0f64;
    let mut num_valid = 0usize;

    // Per-row scratch for logits — avoids allocating B*S*V at once for
    // large-vocab fixtures (e.g. V=49152 × rows=128 = 25 MB of f64).
    let mut row_logits = vec![0f64; v];

    for row in 0..b_times_s {
        let t = targets[row];
        if t == IGNORE_INDEX {
            continue;
        }
        let tgt = t as usize;
        assert!(tgt < v, "target {} out of range [0,{})", tgt, v);

        // Compute logits[row, :] = x[row,:] @ W^T + bias  (row-major, W is [V,H]).
        let x_row = &x[row * h..(row + 1) * h];
        for (vi, logit_slot) in row_logits.iter_mut().enumerate() {
            let mut acc = bias[vi];
            let w_row = &w[vi * h..(vi + 1) * h];
            for hi in 0..h {
                acc += x_row[hi] * w_row[hi];
            }
            *logit_slot = acc;
        }

        // LSE via max-shift for numerical stability.
        let max_logit = row_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let mut sum_exp = 0f64;
        for &lv in row_logits.iter() {
            sum_exp += (lv - max_logit).exp();
        }
        let lse_val = sum_exp.ln() + max_logit;
        lse_per_row[row] = lse_val;

        let loss_row = -(row_logits[tgt] - lse_val);
        loss_per_row[row] = loss_row;
        total_loss += loss_row;
        num_valid += 1;
    }

    let mean_loss = total_loss / num_valid.max(1) as f64;
    CpuLceForward { loss_per_row, lse_per_row, mean_loss }
}

/// Backward reference: `dx = dlogits @ W`, `dW = dlogits^T @ x`,
/// `dbias = sum_rows(dlogits)`, where `dlogits_{r,v} = (softmax_v -
/// one_hot_v) * (grad_loss / max(num_valid, 1))` for valid rows, zero
/// for ignored rows.
///
/// `lse_per_row` MUST come from a prior `cpu_lce_forward_f64` call with
/// the same `(x, w, bias)` inputs.
#[allow(clippy::too_many_arguments)]
pub fn cpu_lce_backward_f64(
    x: &[f64],
    w: &[f64],
    bias: &[f64],
    targets: &[i32],
    lse_per_row: &[f64],
    grad_loss: f64,
    b_times_s: usize,
    v: usize,
    h: usize,
) -> CpuLceBackward {
    assert_eq!(x.len(), b_times_s * h, "x length mismatch");
    assert_eq!(w.len(), v * h, "w length mismatch");
    assert_eq!(bias.len(), v, "bias length mismatch");
    assert_eq!(targets.len(), b_times_s, "targets length mismatch");
    assert_eq!(lse_per_row.len(), b_times_s, "lse length mismatch");

    let num_valid = targets.iter().filter(|&&t| t != IGNORE_INDEX).count();
    let inv_nv = grad_loss / num_valid.max(1) as f64;

    let mut dx = vec![0f64; b_times_s * h];
    let mut dw = vec![0f64; v * h];
    let mut dbias = vec![0f64; v];

    // Per-row scratch buffers — keeps peak memory at O((V + H)) rather than
    // O(B*S*V) for the dlogits matrix.
    let mut row_logits = vec![0f64; v];
    let mut dlogits = vec![0f64; v];

    for row in 0..b_times_s {
        let t = targets[row];
        if t == IGNORE_INDEX {
            // dx, dW, dbias contributions for this row are exactly 0.
            continue;
        }
        let tgt = t as usize;
        assert!(tgt < v, "target {} out of range [0,{})", tgt, v);

        // Recompute logits[row, :].
        for vi in 0..v {
            let mut acc = bias[vi];
            let w_row = &w[vi * h..(vi + 1) * h];
            let x_row = &x[row * h..(row + 1) * h];
            for hi in 0..h {
                acc += x_row[hi] * w_row[hi];
            }
            row_logits[vi] = acc;
        }

        let lse_val = lse_per_row[row];

        // dlogits_v = (softmax_v - one_hot_v) * inv_nv
        for vi in 0..v {
            let p_v = (row_logits[vi] - lse_val).exp();
            let one_hot = if vi == tgt { 1.0f64 } else { 0.0 };
            dlogits[vi] = (p_v - one_hot) * inv_nv;
        }

        // dbias += dlogits[row, :]
        for vi in 0..v {
            dbias[vi] += dlogits[vi];
        }

        // dx[row, h] = sum_v dlogits[v] * W[v, h]
        let x_row = &x[row * h..(row + 1) * h];
        let dx_row = &mut dx[row * h..(row + 1) * h];
        for vi in 0..v {
            let dl_v = dlogits[vi];
            let w_row = &w[vi * h..(vi + 1) * h];
            for hi in 0..h {
                dx_row[hi] += dl_v * w_row[hi];
            }
        }

        // dW[v, h] += dlogits[v] * x[row, h]
        for vi in 0..v {
            let dl_v = dlogits[vi];
            let dw_row = &mut dw[vi * h..(vi + 1) * h];
            for hi in 0..h {
                dw_row[hi] += dl_v * x_row[hi];
            }
        }
    }

    CpuLceBackward { dx, dw, dbias }
}

// ─── Unit tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Hand-computed single-token case: V=3, H=2, one row, target=1.
    ///
    /// x = [1.0, 2.0]
    /// W = [[0.1, 0.2],
    ///      [0.3, 0.4],
    ///      [0.5, 0.6]]
    /// bias = [0.0, 0.0, 0.0]
    /// logits = [0.1*1 + 0.2*2, 0.3*1 + 0.4*2, 0.5*1 + 0.6*2]
    ///        = [0.5, 1.1, 1.7]
    /// max = 1.7
    /// sum_exp(shifted) = exp(-1.2) + exp(-0.6) + exp(0)
    ///                  = 0.30119421191220...
    ///                    + 0.54881163609402...
    ///                    + 1.0
    ///                  = 1.85000584800622...
    /// lse = ln(1.85000584800622) + 1.7 = 0.61518563798...  + 1.7 = 2.31518563798
    /// loss = -(logits[1] - lse) = -(1.1 - 2.31518563798) = 1.21518563798
    #[test]
    fn forward_hand_computed_v3_h2() {
        let x = vec![1.0, 2.0];
        let w = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6];
        let bias = vec![0.0, 0.0, 0.0];
        let targets = vec![1i32];

        let out = cpu_lce_forward_f64(&x, &w, &bias, &targets, 1, 3, 2);

        let expected_lse: f64 = {
            let logits = [0.5_f64, 1.1, 1.7];
            let m = 1.7_f64;
            let s = (logits[0] - m).exp() + (logits[1] - m).exp() + (logits[2] - m).exp();
            s.ln() + m
        };
        let expected_loss = -(1.1 - expected_lse);

        assert!((out.lse_per_row[0] - expected_lse).abs() < 1e-12,
            "lse={} expected={}", out.lse_per_row[0], expected_lse);
        assert!((out.loss_per_row[0] - expected_loss).abs() < 1e-12,
            "loss={} expected={}", out.loss_per_row[0], expected_loss);
        assert!((out.mean_loss - expected_loss).abs() < 1e-12,
            "mean_loss={} expected={}", out.mean_loss, expected_loss);
    }

    /// Larger single-row case V=10, H=4 — exercises generic indexing.
    #[test]
    fn forward_v10_h4_softmax_invariant() {
        let h = 4usize;
        let v = 10usize;
        let x: Vec<f64> = (0..h).map(|i| (i as f64) * 0.1 - 0.2).collect();
        let w: Vec<f64> = (0..v * h)
            .map(|i| ((i as f64) * 0.037).sin() * 0.5)
            .collect();
        let bias: Vec<f64> = (0..v).map(|i| (i as f64) * 0.01).collect();
        let targets = vec![3i32];

        let out = cpu_lce_forward_f64(&x, &w, &bias, &targets, 1, v, h);

        // Softmax invariant: sum_v exp(logit_v - lse) == 1.
        // Recompute logits and check.
        let mut logits = vec![0f64; v];
        for vi in 0..v {
            let mut acc = bias[vi];
            for hi in 0..h {
                acc += x[hi] * w[vi * h + hi];
            }
            logits[vi] = acc;
        }
        let lse = out.lse_per_row[0];
        let s: f64 = logits.iter().map(|&l| (l - lse).exp()).sum();
        assert!((s - 1.0).abs() < 1e-12, "softmax sum {} != 1", s);

        // loss = -(logit_tgt - lse) >= 0 (since p in (0, 1])
        let expected_loss = -(logits[3] - lse);
        assert!((out.loss_per_row[0] - expected_loss).abs() < 1e-12);
    }

    /// Ignored rows contribute nothing to the mean and have zero loss/lse.
    #[test]
    fn forward_ignore_index_skipped() {
        let h = 2usize;
        let v = 4usize;
        let rows = 4usize;
        // Two valid rows (idx 1, 3) and two ignored (idx 0, 2).
        let x: Vec<f64> = (0..rows * h).map(|i| (i as f64) * 0.1).collect();
        let w: Vec<f64> = (0..v * h).map(|i| (i as f64) * 0.05).collect();
        let bias: Vec<f64> = vec![0.0; v];
        let targets = vec![IGNORE_INDEX, 1, IGNORE_INDEX, 2];

        let out = cpu_lce_forward_f64(&x, &w, &bias, &targets, rows, v, h);

        assert_eq!(out.loss_per_row[0], 0.0, "ignored row 0 should have 0 loss");
        assert_eq!(out.lse_per_row[0], 0.0, "ignored row 0 should have 0 lse");
        assert_eq!(out.loss_per_row[2], 0.0, "ignored row 2 should have 0 loss");
        assert_eq!(out.lse_per_row[2], 0.0, "ignored row 2 should have 0 lse");
        assert!(out.loss_per_row[1] > 0.0, "valid row 1 should have positive loss");
        assert!(out.loss_per_row[3] > 0.0, "valid row 3 should have positive loss");

        // Mean is over 2 valid rows.
        let expected_mean = (out.loss_per_row[1] + out.loss_per_row[3]) / 2.0;
        assert!((out.mean_loss - expected_mean).abs() < 1e-15,
            "mean_loss={} expected={}", out.mean_loss, expected_mean);
    }

    /// All-ignored corner: mean is 0 (denominator clamped to 1).
    #[test]
    fn forward_all_ignored_is_zero_mean() {
        let h = 2usize;
        let v = 3usize;
        let rows = 2usize;
        let x = vec![1.0; rows * h];
        let w = vec![0.5; v * h];
        let bias = vec![0.0; v];
        let targets = vec![IGNORE_INDEX, IGNORE_INDEX];

        let out = cpu_lce_forward_f64(&x, &w, &bias, &targets, rows, v, h);
        assert_eq!(out.mean_loss, 0.0);
    }

    /// Finite-differences gate on `dx`. The analytic dx must match
    /// `(L(x + eps) - L(x - eps)) / (2 eps)` for each element of `x`.
    #[test]
    fn backward_dx_matches_finite_differences() {
        let h = 3usize;
        let v = 5usize;
        let rows = 2usize;
        let x: Vec<f64> = (0..rows * h).map(|i| (i as f64) * 0.07 - 0.1).collect();
        let w: Vec<f64> = (0..v * h)
            .map(|i| ((i as f64) * 0.13).cos() * 0.3)
            .collect();
        let bias: Vec<f64> = (0..v).map(|i| (i as f64) * 0.02 - 0.05).collect();
        let targets = vec![2i32, 4i32];

        let fwd = cpu_lce_forward_f64(&x, &w, &bias, &targets, rows, v, h);
        let bwd = cpu_lce_backward_f64(
            &x, &w, &bias, &targets, &fwd.lse_per_row, 1.0, rows, v, h,
        );

        // grad_loss=1.0 with the helper's inv_nv = 1/num_valid means
        // bwd.dx is d(mean_loss)/dx. The FD probe must use mean_loss too.
        let eps = 1e-5;
        for idx in 0..rows * h {
            let mut x_plus = x.clone();
            let mut x_minus = x.clone();
            x_plus[idx] += eps;
            x_minus[idx] -= eps;
            let f_plus = cpu_lce_forward_f64(&x_plus, &w, &bias, &targets, rows, v, h);
            let f_minus = cpu_lce_forward_f64(&x_minus, &w, &bias, &targets, rows, v, h);
            let fd = (f_plus.mean_loss - f_minus.mean_loss) / (2.0 * eps);
            let an = bwd.dx[idx];
            assert!((an - fd).abs() < 1e-7,
                "dx[{idx}]: analytic={an} fd={fd} diff={}", (an - fd).abs());
        }
    }

    /// Finite-differences gate on `dW`.
    #[test]
    fn backward_dw_matches_finite_differences() {
        let h = 2usize;
        let v = 4usize;
        let rows = 2usize;
        let x: Vec<f64> = (0..rows * h).map(|i| (i as f64) * 0.05).collect();
        let w: Vec<f64> = (0..v * h)
            .map(|i| ((i as f64) * 0.11).sin() * 0.2)
            .collect();
        let bias: Vec<f64> = vec![0.01, -0.02, 0.03, -0.04];
        let targets = vec![1i32, 3i32];

        let fwd = cpu_lce_forward_f64(&x, &w, &bias, &targets, rows, v, h);
        let bwd = cpu_lce_backward_f64(
            &x, &w, &bias, &targets, &fwd.lse_per_row, 1.0, rows, v, h,
        );

        let eps = 1e-5;
        for idx in 0..v * h {
            let mut w_plus = w.clone();
            let mut w_minus = w.clone();
            w_plus[idx] += eps;
            w_minus[idx] -= eps;
            let f_plus = cpu_lce_forward_f64(&x, &w_plus, &bias, &targets, rows, v, h);
            let f_minus = cpu_lce_forward_f64(&x, &w_minus, &bias, &targets, rows, v, h);
            let fd = (f_plus.mean_loss - f_minus.mean_loss) / (2.0 * eps);
            let an = bwd.dw[idx];
            assert!((an - fd).abs() < 1e-7,
                "dw[{idx}]: analytic={an} fd={fd} diff={}", (an - fd).abs());
        }
    }

    /// Finite-differences gate on `dbias`.
    #[test]
    fn backward_dbias_matches_finite_differences() {
        let h = 2usize;
        let v = 3usize;
        let rows = 2usize;
        let x: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
        let w: Vec<f64> = (0..v * h).map(|i| (i as f64) * 0.07).collect();
        let bias: Vec<f64> = vec![0.0, 0.0, 0.0];
        let targets = vec![0i32, 2i32];

        let fwd = cpu_lce_forward_f64(&x, &w, &bias, &targets, rows, v, h);
        let bwd = cpu_lce_backward_f64(
            &x, &w, &bias, &targets, &fwd.lse_per_row, 1.0, rows, v, h,
        );

        let eps = 1e-5;
        for idx in 0..v {
            let mut b_plus = bias.clone();
            let mut b_minus = bias.clone();
            b_plus[idx] += eps;
            b_minus[idx] -= eps;
            let f_plus = cpu_lce_forward_f64(&x, &w, &b_plus, &targets, rows, v, h);
            let f_minus = cpu_lce_forward_f64(&x, &w, &b_minus, &targets, rows, v, h);
            let fd = (f_plus.mean_loss - f_minus.mean_loss) / (2.0 * eps);
            let an = bwd.dbias[idx];
            assert!((an - fd).abs() < 1e-7,
                "dbias[{idx}]: analytic={an} fd={fd} diff={}", (an - fd).abs());
        }
    }

    /// Ignored rows must not contribute to dx / dW / dbias.
    /// Specifically: zeroing-out the ignored row's x-slice must leave
    /// dW and dbias unchanged (its dx slice is zero by construction).
    #[test]
    fn backward_ignored_rows_contribute_zero() {
        let h = 2usize;
        let v = 3usize;
        let rows = 3usize;
        let x: Vec<f64> = (0..rows * h).map(|i| (i as f64) * 0.1 + 0.05).collect();
        let w: Vec<f64> = (0..v * h).map(|i| (i as f64) * 0.04).collect();
        let bias: Vec<f64> = vec![0.0, 0.0, 0.0];
        // Row 1 is ignored.
        let targets = vec![0i32, IGNORE_INDEX, 2i32];

        let fwd = cpu_lce_forward_f64(&x, &w, &bias, &targets, rows, v, h);
        let bwd = cpu_lce_backward_f64(
            &x, &w, &bias, &targets, &fwd.lse_per_row, 1.0, rows, v, h,
        );

        // dx for the ignored row must be exactly zero.
        for hi in 0..h {
            assert_eq!(bwd.dx[h + hi], 0.0,
                "ignored row dx[1,{}] should be 0, got {}", hi, bwd.dx[h + hi]);
        }

        // Permute x[ignored row] to garbage and re-run — dW / dbias must
        // be byte-identical because the ignored row contributes nothing.
        let mut x_perturbed = x.clone();
        x_perturbed[h] = 9999.0;
        x_perturbed[h + 1] = -9999.0;
        let fwd2 = cpu_lce_forward_f64(&x_perturbed, &w, &bias, &targets, rows, v, h);
        let bwd2 = cpu_lce_backward_f64(
            &x_perturbed, &w, &bias, &targets, &fwd2.lse_per_row, 1.0, rows, v, h,
        );
        for i in 0..bwd.dw.len() {
            assert_eq!(bwd.dw[i], bwd2.dw[i],
                "dW[{i}] changed when ignored row x was perturbed");
        }
        for i in 0..bwd.dbias.len() {
            assert_eq!(bwd.dbias[i], bwd2.dbias[i],
                "dbias[{i}] changed when ignored row x was perturbed");
        }
    }

    /// All-ignored corner: gradients are entirely zero.
    #[test]
    fn backward_all_ignored_is_zero() {
        let h = 2usize;
        let v = 3usize;
        let rows = 2usize;
        let x = vec![1.0; rows * h];
        let w = vec![0.5; v * h];
        let bias = vec![0.0; v];
        let targets = vec![IGNORE_INDEX, IGNORE_INDEX];

        let fwd = cpu_lce_forward_f64(&x, &w, &bias, &targets, rows, v, h);
        let bwd = cpu_lce_backward_f64(
            &x, &w, &bias, &targets, &fwd.lse_per_row, 1.0, rows, v, h,
        );
        assert!(bwd.dx.iter().all(|&v| v == 0.0));
        assert!(bwd.dw.iter().all(|&v| v == 0.0));
        assert!(bwd.dbias.iter().all(|&v| v == 0.0));
    }

    /// `grad_loss` scales gradients linearly.
    #[test]
    fn backward_grad_loss_scales_linearly() {
        let h = 2usize;
        let v = 3usize;
        let rows = 2usize;
        let x: Vec<f64> = vec![0.1, 0.2, 0.3, 0.4];
        let w: Vec<f64> = (0..v * h).map(|i| (i as f64) * 0.07).collect();
        let bias: Vec<f64> = vec![0.0, 0.0, 0.0];
        let targets = vec![0i32, 2i32];

        let fwd = cpu_lce_forward_f64(&x, &w, &bias, &targets, rows, v, h);
        let bwd1 = cpu_lce_backward_f64(
            &x, &w, &bias, &targets, &fwd.lse_per_row, 1.0, rows, v, h,
        );
        let bwd3 = cpu_lce_backward_f64(
            &x, &w, &bias, &targets, &fwd.lse_per_row, 3.0, rows, v, h,
        );

        for i in 0..bwd1.dx.len() {
            let scaled = bwd1.dx[i] * 3.0;
            assert!((bwd3.dx[i] - scaled).abs() < 1e-14,
                "dx[{i}] not linear in grad_loss: 1.0->{} 3.0->{} expected {}",
                bwd1.dx[i], bwd3.dx[i], scaled);
        }
    }
}
