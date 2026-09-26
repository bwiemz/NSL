//! Finite-difference gradcheck of the CPU-naive attention backward oracles
//! (`cpu_naive_backward_{dq,dkdv,proj}`) that the tier-B2 GPU backward tests
//! compare against.
//!
//! The analytic gradients are checked against central differences of an
//! independent f64 forward — `O = softmax(Q K^T / sqrt(d) [causal]) V` and
//! `Q = (x / rms(x) * gamma) Wq` — at the f16 input values. Two batches and
//! two heads, causal and not, with a random `dO`.
//!
//! Bounds are relative to each tensor's largest gradient. dQ and dK read the
//! f16 forward output `O` for `D = rowsum(dO * O)`, so they carry f16-level
//! error: measured at most 4.6e-4, bound 2e-3. dV and the projection oracle
//! are exact up to f32 accumulation: measured at most 1.8e-7, bound 1e-5.

use half::f16;
use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_test::cpu_naive_backward::{cpu_naive_backward_dkdv, cpu_naive_backward_dq, cpu_naive_backward_proj};
use nsl_test::cpu_naive_forward::cpu_naive_forward;

/// dQ/dK: f16 `O` in `D = rowsum(dO * O)`.
const TOL_F16_O: f64 = 2e-3;
/// dV and the projection oracle: f32 accumulation only.
const TOL_F32: f64 = 1e-5;

fn cfg(hd: usize, causal: bool) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: hd as i64, causal, paged: false,
        rope_q: false, rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, num_sink_tokens: 0, gpu_sm: 80, segment_masked: false,
        csha: Some(CshaExtras { level: 2, ..Default::default() }),
        checkpoint: None,
    }
}

fn seq_f16(n: usize, seed: u64, amp: f32) -> Vec<f16> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            f16::from_f32(((s >> 40) as f32 / (1u64 << 24) as f32 - 0.5) * 2.0 * amp)
        })
        .collect()
}

fn f64s(v: &[f16]) -> Vec<f64> {
    v.iter().map(|x| x.to_f64()).collect()
}

/// `sum(O * dO)` for `O = softmax(Q K^T * scale, causal) V`, in f64.
#[allow(clippy::too_many_arguments)]
fn attention_loss(q: &[f64], k: &[f64], v: &[f64], d_o: &[f64], bh: usize, s: usize, d: usize, causal: bool) -> f64 {
    let scale = 1.0 / (d as f64).sqrt();
    let mut loss = 0.0;
    for g in 0..bh {
        let at = |t: &[f64], row: usize, c: usize| t[(g * s + row) * d + c];
        for qi in 0..s {
            let lim = if causal { qi + 1 } else { s };
            let sc: Vec<f64> = (0..lim).map(|ki| (0..d).map(|c| at(q, qi, c) * at(k, ki, c)).sum::<f64>() * scale).collect();
            let m = sc.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = sc.iter().map(|x| (x - m).exp()).collect();
            let z: f64 = e.iter().sum();
            for c in 0..d {
                let o: f64 = (0..lim).map(|ki| e[ki] / z * at(v, ki, c)).sum();
                loss += o * at(d_o, qi, c);
            }
        }
    }
    loss
}

/// Worst |analytic - fd| over `base`'s entries, relative to max|analytic|.
fn check(name: &str, tol: f64, analytic: &[f32], base: &[f64], loss: impl Fn(&[f64]) -> f64) {
    let eps = 1e-4;
    let scale = analytic.iter().fold(0.0f64, |m, &g| m.max(g.abs() as f64));
    assert!(scale > 1e-3, "{name}: gradient too small to check (max |g| = {scale})");
    let mut worst = 0.0f64;
    let mut at = 0;
    for i in 0..base.len() {
        let mut p = base.to_vec();
        p[i] = base[i] + eps;
        let lp = loss(&p);
        p[i] = base[i] - eps;
        let lm = loss(&p);
        let err = ((lp - lm) / (2.0 * eps) - analytic[i] as f64).abs();
        if err > worst {
            worst = err;
            at = i;
        }
    }
    assert!(
        worst <= tol * scale,
        "{name}[{at}]: analytic {} vs central difference off by {worst:.3e} ({:.3e} of max |g| = {scale:.3e})",
        analytic[at],
        worst / scale
    );
}

#[test]
fn attention_backward_oracles_match_central_differences() {
    let (b, h, s, d) = (2usize, 2usize, 6usize, 4usize);
    let n = b * h * s * d;
    for causal in [false, true] {
        let q = seq_f16(n, 1, 1.0);
        let k = seq_f16(n, 2, 1.0);
        let v = seq_f16(n, 3, 1.0);
        let d_o = seq_f16(n, 4, 1.0);
        let fwd = cpu_naive_forward(&q, &k, &v, b, h, s, d, causal);
        let c = cfg(d, causal);
        let dq = cpu_naive_backward_dq(&fwd.q_saved, &fwd.k_saved, &fwd.v_saved, &fwd.o, &d_o, b, h, s, &c);
        let (dv, dk) = cpu_naive_backward_dkdv(&fwd.q_saved, &fwd.k_saved, &fwd.v_saved, &fwd.o, &d_o, b, h, s, &c);
        let (qf, kf, vf, dof) = (f64s(&q), f64s(&k), f64s(&v), f64s(&d_o));
        let tag = if causal { "causal" } else { "full" };
        check(&format!("{tag} dq"), TOL_F16_O, &dq, &qf, |p| attention_loss(p, &kf, &vf, &dof, b * h, s, d, causal));
        check(&format!("{tag} dk"), TOL_F16_O, &dk, &kf, |p| attention_loss(&qf, p, &vf, &dof, b * h, s, d, causal));
        check(&format!("{tag} dv"), TOL_F32, &dv, &vf, |p| attention_loss(&qf, &kf, p, &dof, b * h, s, d, causal));
    }
}

#[test]
fn projection_backward_oracle_matches_central_differences() {
    // heads = 1: the oracle's documented scope.
    let (b, s, hd, dm) = (2usize, 5usize, 4usize, 6usize);
    let eps_norm = 1e-5f32;
    let x = seq_f16(b * s * dm, 5, 1.0);
    let wq = seq_f16(dm * hd, 6, 1.0);
    let wk = seq_f16(dm * hd, 7, 1.0);
    let wv = seq_f16(dm * hd, 8, 1.0);
    let gamma: Vec<f16> = seq_f16(dm, 9, 0.5).iter().map(|g| f16::from_f32(1.0 + g.to_f32())).collect();
    // Upstream gradients of Q, K, V: the loss is sum(Q*gq + K*gk + V*gv).
    let to_f32 = |v: Vec<f16>| v.iter().map(|x| x.to_f32()).collect::<Vec<f32>>();
    let gq = to_f32(seq_f16(b * s * hd, 10, 1.0));
    let gk = to_f32(seq_f16(b * s * hd, 11, 1.0));
    let gv = to_f32(seq_f16(b * s * hd, 12, 1.0));
    let (dwq, dwk, dwv, dx) = cpu_naive_backward_proj(&gq, &gk, &gv, &x, &wq, &wk, &wv, &gamma, eps_norm, b, 1, s, hd, dm);

    let gm = f64s(&gamma);
    let loss = |x: &[f64], wq: &[f64], wk: &[f64], wv: &[f64]| -> f64 {
        let mut l = 0.0;
        for r in 0..b * s {
            let row = &x[r * dm..(r + 1) * dm];
            let rms = (row.iter().map(|v| v * v).sum::<f64>() / dm as f64 + eps_norm as f64).sqrt();
            for j in 0..hd {
                let proj = |w: &[f64]| (0..dm).map(|p| row[p] / rms * gm[p] * w[p * hd + j]).sum::<f64>();
                l += proj(wq) * gq[r * hd + j] as f64 + proj(wk) * gk[r * hd + j] as f64 + proj(wv) * gv[r * hd + j] as f64;
            }
        }
        l
    };
    let (xf, wqf, wkf, wvf) = (f64s(&x), f64s(&wq), f64s(&wk), f64s(&wv));
    check("dx", TOL_F32, &dx, &xf, |p| loss(p, &wqf, &wkf, &wvf));
    check("dwq", TOL_F32, &dwq, &wqf, |p| loss(&xf, p, &wkf, &wvf));
    check("dwk", TOL_F32, &dwk, &wkf, |p| loss(&xf, &wqf, p, &wvf));
    check("dwv", TOL_F32, &dwv, &wvf, |p| loss(&xf, &wqf, &wkf, p));
}
