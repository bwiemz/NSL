//! f64 reference for the CSHA fused attention backward, pinned to PyTorch.
//!
//! The GPU gates in `csha_cuda_backward.rs` used to compare the kernel with
//! an f32 CPU routine that was handed the kernel's own f16-rounded inputs,
//! at gradient magnitudes where f16 is harmless. Both sides rounded alike,
//! so a kernel that flushed every realistic gradient to zero passed.
//!
//! This file is the independent side of the comparison:
//!
//! * [`fused_backward_f64`] evaluates the layer and its gradients in f64
//!   from the UNROUNDED f32 inputs.
//! * The tests below hold that routine to literals produced by PyTorch's
//!   float64 autograd on the same inputs (printed by
//!   `tools/refcheck/csha_backward_literals.py`, which differentiates
//!   `sum(O * dO)`; the inputs are the `det_seq` LCG, which the script
//!   reproduces exactly). They run on the CPU, in CI.
//!
//! The model differentiated, for one head:
//!
//! ```text
//! x_norm = x / sqrt(mean(x^2) + eps) * gamma        x: [batch, seq, d_model]
//! Q = x_norm @ Wq;  K = x_norm @ Wk;  V = x_norm @ Wv     W: [d_model, head_dim]
//! O = softmax(rope(Q) rope(K)^T * scale, causal mask optional) V
//! ```
//!
//! `rope` is the optional rotary embedding over adjacent pairs. dQ and dK are
//! the gradients with respect to the projections, before the rotation.
//!
//! Included as a module by the GPU tests (`#[path]`), and compiled on its
//! own as the CPU test target that pins it.

#![allow(dead_code)]

/// The LCG the CSHA GPU harnesses draw their inputs from: `n` values in
/// `[-0.5, 0.5]`.
pub fn det_seq(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as f32 / 65535.0) - 0.5
        })
        .collect()
}

/// A non-trivial RMSNorm weight: `n` values in `[0.75, 1.25]`. An all-ones
/// weight cannot tell a kernel that applies it from one that drops it.
pub fn gamma_seq(n: usize) -> Vec<f32> {
    det_seq(46, n).into_iter().map(|u| 0.75 + 0.5 * (u + 0.5)).collect()
}

/// `det_seq(99, n)` scaled to a gradient magnitude, in f32 as the kernel
/// receives it.
pub fn d_o_seq(n: usize, magnitude: f32) -> Vec<f32> {
    det_seq(99, n).into_iter().map(|u| u * magnitude).collect()
}

/// One fused-projection attention layer and the gradient arriving at its
/// output. A single head: `head_dim` is the projection width.
pub struct FusedCase<'a> {
    pub batch: usize,
    pub seq: usize,
    pub head_dim: usize,
    pub d_model: usize,
    pub causal: bool,
    pub eps: f32,
    /// `[batch, seq, d_model]`
    pub x: &'a [f32],
    /// `[d_model, head_dim]` each
    pub wq: &'a [f32],
    pub wk: &'a [f32],
    pub wv: &'a [f32],
    /// `[d_model]`
    pub gamma: &'a [f32],
    /// `[batch, seq, head_dim]`
    pub d_o: &'a [f32],
    /// Rotary tables `(cos, sin)`, each `[seq, head_dim / 2]`, applied to Q
    /// and K over adjacent pairs `(2i, 2i+1)` before the attention product.
    /// `None` leaves Q and K unrotated.
    pub rope: Option<(&'a [f32], &'a [f32])>,
}

/// Rotary tables `(cos, sin)`, each `[seq, head_dim / 2]`: one rotation per
/// (position, pair), built from IEEE-exact f32 operations only (multiply,
/// subtract, square root) so that NumPy and Rust produce the same bits.
/// `c = 1.8 * det_seq(47)` lies in `[-0.9, 0.9]` and `s = sqrt(1 - c*c)`.
pub fn rope_tables(seq: usize, head_dim: usize) -> (Vec<f32>, Vec<f32>) {
    let cos: Vec<f32> = det_seq(47, seq * head_dim / 2).into_iter().map(|u| u * 1.8).collect();
    let sin: Vec<f32> = cos.iter().map(|&c| (1.0f32 - c * c).sqrt()).collect();
    (cos, sin)
}

/// Rotate adjacent pairs of a `[seq, head_dim]` tensor:
/// `y0 = x0 cos - x1 sin`, `y1 = x0 sin + x1 cos`. With `transpose` the
/// inverse rotation, which is what carries a gradient back through it.
fn rope_adjacent(t: &mut [f64], seq: usize, head_dim: usize, cos: &[f32], sin: &[f32], transpose: bool) {
    let half = head_dim / 2;
    for s in 0..seq {
        for i in 0..half {
            let c = f64::from(cos[s * half + i]);
            let sn = f64::from(sin[s * half + i]);
            let sn = if transpose { -sn } else { sn };
            let a = t[s * head_dim + 2 * i];
            let b = t[s * head_dim + 2 * i + 1];
            t[s * head_dim + 2 * i] = a * c - b * sn;
            t[s * head_dim + 2 * i + 1] = a * sn + b * c;
        }
    }
}

/// Forward output and every gradient of `sum(O * dO)`.
pub struct Grads64 {
    /// The projections the attention ran on (after RoPE for Q and K),
    /// `[batch, seq, head_dim]`: what the forward saves for the backward.
    pub q: Vec<f64>,
    pub k: Vec<f64>,
    pub v: Vec<f64>,
    /// The softmax statistics the forward saves, `[batch, seq]`.
    pub row_max: Vec<f64>,
    pub row_sum: Vec<f64>,
    /// `[batch, seq, head_dim]`
    pub o: Vec<f64>,
    pub dq: Vec<f64>,
    pub dk: Vec<f64>,
    pub dv: Vec<f64>,
    /// `[d_model, head_dim]`
    pub dwq: Vec<f64>,
    pub dwk: Vec<f64>,
    pub dwv: Vec<f64>,
    /// `[batch, seq, d_model]`
    pub dx: Vec<f64>,
    pub dx_norm: Vec<f64>,
}

/// Attention weights of one `[seq, head_dim]` slice and the per-row softmax
/// statistics the forward kernel saves for the backward.
pub struct AttnWeights64 {
    /// `P = softmax(Q K^T * scale)`, row-major `[seq, seq]`.
    pub p: Vec<f64>,
    /// `max_j S[i, j]` over the unmasked columns, `[seq]`.
    pub row_max: Vec<f64>,
    /// `sum_j exp(S[i, j] - row_max[i])`, `[seq]`.
    pub row_sum: Vec<f64>,
}

pub fn attention_weights_f64(
    q: &[f64],
    k: &[f64],
    seq: usize,
    head_dim: usize,
    scale: f64,
    causal: bool,
) -> AttnWeights64 {
    let mut p = vec![0.0f64; seq * seq];
    let mut row_max = vec![0.0f64; seq];
    let mut row_sum = vec![0.0f64; seq];
    for i in 0..seq {
        let last = if causal { i } else { seq - 1 };
        let mut m = f64::NEG_INFINITY;
        for j in 0..=last {
            let mut s = 0.0f64;
            for d in 0..head_dim {
                s += q[i * head_dim + d] * k[j * head_dim + d];
            }
            let s = s * scale;
            p[i * seq + j] = s;
            m = m.max(s);
        }
        let mut sum = 0.0f64;
        for j in 0..=last {
            let e = (p[i * seq + j] - m).exp();
            p[i * seq + j] = e;
            sum += e;
        }
        for j in 0..=last {
            p[i * seq + j] /= sum;
        }
        row_max[i] = m;
        row_sum[i] = sum;
    }
    AttnWeights64 { p, row_max, row_sum }
}

/// Output and gradients of one attention slice (one batch element, one head).
pub struct AttnGrads64 {
    /// `[seq, head_dim]` each.
    pub o: Vec<f64>,
    pub dq: Vec<f64>,
    pub dk: Vec<f64>,
    pub dv: Vec<f64>,
    /// `[seq]` each; see [`AttnWeights64`].
    pub row_max: Vec<f64>,
    pub row_sum: Vec<f64>,
}

/// `O = softmax(Q K^T * scale) V` for one `[seq, head_dim]` slice, and the
/// gradients of `sum(O * dO)` with respect to Q, K and V:
///
/// ```text
/// dV = P^T dO
/// dP = dO V^T
/// dS = P * (dP - rowsum(dP * P))          (softmax Jacobian, row-wise)
/// dQ = scale * dS K        dK = scale * dS^T Q
/// ```
#[allow(clippy::too_many_arguments)]
pub fn attention_backward_f64(
    q: &[f64],
    k: &[f64],
    v: &[f64],
    d_o: &[f64],
    seq: usize,
    head_dim: usize,
    scale: f64,
    causal: bool,
) -> AttnGrads64 {
    let (s_n, hd) = (seq, head_dim);
    let AttnWeights64 { p: p_attn, row_max, row_sum } =
        attention_weights_f64(q, k, s_n, hd, scale, causal);
    let mut g = AttnGrads64 {
        o: vec![0.0; s_n * hd],
        dq: vec![0.0; s_n * hd],
        dk: vec![0.0; s_n * hd],
        dv: vec![0.0; s_n * hd],
        row_max,
        row_sum,
    };
    let mut d_s = vec![0.0f64; s_n * s_n];
    for i in 0..s_n {
        let mut dp = vec![0.0f64; s_n];
        let mut corr = 0.0f64;
        for j in 0..s_n {
            let w = p_attn[i * s_n + j];
            let mut acc = 0.0f64;
            for d in 0..hd {
                g.o[i * hd + d] += w * v[j * hd + d];
                acc += d_o[i * hd + d] * v[j * hd + d];
            }
            dp[j] = acc;
            corr += acc * w;
        }
        for j in 0..s_n {
            let w = p_attn[i * s_n + j];
            d_s[i * s_n + j] = w * (dp[j] - corr);
            for d in 0..hd {
                g.dv[j * hd + d] += w * d_o[i * hd + d];
            }
        }
    }
    for i in 0..s_n {
        for j in 0..s_n {
            let ds = d_s[i * s_n + j] * scale;
            for d in 0..hd {
                g.dq[i * hd + d] += ds * k[j * hd + d];
                g.dk[j * hd + d] += ds * q[i * hd + d];
            }
        }
    }
    g
}

/// [`attention_backward_f64`] for f32 inputs, with the scale the launchers
/// pass (`1/sqrt(head_dim)` rounded to f32). For the kernels that take Q, K
/// and V directly, without projections.
pub fn attention_backward_f64_from_f32(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    d_o: &[f32],
    seq: usize,
    head_dim: usize,
    causal: bool,
) -> AttnGrads64 {
    let widen = |a: &[f32]| -> Vec<f64> { a.iter().map(|&x| f64::from(x)).collect() };
    let scale = f64::from(1.0f32 / (head_dim as f32).sqrt());
    attention_backward_f64(&widen(q), &widen(k), &widen(v), &widen(d_o), seq, head_dim, scale, causal)
}

/// The layer's output and gradients in f64, from the unrounded f32 inputs.
pub fn fused_backward_f64(c: &FusedCase<'_>) -> Grads64 {
    let (b_n, s_n, hd, dm) = (c.batch, c.seq, c.head_dim, c.d_model);
    assert_eq!(c.x.len(), b_n * s_n * dm);
    assert_eq!(c.d_o.len(), b_n * s_n * hd);
    for w in [c.wq, c.wk, c.wv] {
        assert_eq!(w.len(), dm * hd);
    }
    assert_eq!(c.gamma.len(), dm);
    // The scale the launchers pass: 1/sqrt(head_dim) rounded to f32.
    let scale = f64::from(1.0f32 / (hd as f32).sqrt());
    let eps = f64::from(c.eps);

    let mut g = Grads64 {
        q: vec![0.0; b_n * s_n * hd],
        k: vec![0.0; b_n * s_n * hd],
        v: vec![0.0; b_n * s_n * hd],
        row_max: vec![0.0; b_n * s_n],
        row_sum: vec![0.0; b_n * s_n],
        o: vec![0.0; b_n * s_n * hd],
        dq: vec![0.0; b_n * s_n * hd],
        dk: vec![0.0; b_n * s_n * hd],
        dv: vec![0.0; b_n * s_n * hd],
        dwq: vec![0.0; dm * hd],
        dwk: vec![0.0; dm * hd],
        dwv: vec![0.0; dm * hd],
        dx: vec![0.0; b_n * s_n * dm],
        dx_norm: vec![0.0; b_n * s_n * dm],
    };

    for b in 0..b_n {
        let x = &c.x[b * s_n * dm..(b + 1) * s_n * dm];
        let d_o = &c.d_o[b * s_n * hd..(b + 1) * s_n * hd];

        // RMSNorm.
        let mut rms = vec![0.0f64; s_n];
        let mut xn = vec![0.0f64; s_n * dm];
        for s in 0..s_n {
            let mut ss = 0.0f64;
            for p in 0..dm {
                let v = f64::from(x[s * dm + p]);
                ss += v * v;
            }
            rms[s] = (ss / dm as f64 + eps).sqrt();
            for p in 0..dm {
                xn[s * dm + p] = f64::from(x[s * dm + p]) / rms[s] * f64::from(c.gamma[p]);
            }
        }

        // Projections.
        let project = |w: &[f32]| -> Vec<f64> {
            let mut out = vec![0.0f64; s_n * hd];
            for s in 0..s_n {
                for p in 0..dm {
                    let a = xn[s * dm + p];
                    for j in 0..hd {
                        out[s * hd + j] += a * f64::from(w[p * hd + j]);
                    }
                }
            }
            out
        };
        let mut q = project(c.wq);
        let mut k = project(c.wk);
        let v = project(c.wv);
        if let Some((cos, sin)) = c.rope {
            rope_adjacent(&mut q, s_n, hd, cos, sin, false);
            rope_adjacent(&mut k, s_n, hd, cos, sin, false);
        }

        // Attention and its gradients, on the (rotated) projections.
        let d_o64: Vec<f64> = d_o.iter().map(|&a| f64::from(a)).collect();
        let attn = attention_backward_f64(&q, &k, &v, &d_o64, s_n, hd, scale, c.causal);
        g.o[b * s_n * hd..(b + 1) * s_n * hd].copy_from_slice(&attn.o);
        g.q[b * s_n * hd..(b + 1) * s_n * hd].copy_from_slice(&q);
        g.k[b * s_n * hd..(b + 1) * s_n * hd].copy_from_slice(&k);
        g.v[b * s_n * hd..(b + 1) * s_n * hd].copy_from_slice(&v);
        g.row_max[b * s_n..(b + 1) * s_n].copy_from_slice(&attn.row_max);
        g.row_sum[b * s_n..(b + 1) * s_n].copy_from_slice(&attn.row_sum);
        let (mut dq, mut dk, dv) = (attn.dq, attn.dk, attn.dv);

        // Back through the rotation: dQ and dK are reported with respect to
        // the projections, before RoPE.
        if let Some((cos, sin)) = c.rope {
            rope_adjacent(&mut dq, s_n, hd, cos, sin, true);
            rope_adjacent(&mut dk, s_n, hd, cos, sin, true);
        }

        // dW = x_norm^T dY (summed over the batch);
        // dx_norm = dQ Wq^T + dK Wk^T + dV Wv^T.
        let dxn = &mut g.dx_norm[b * s_n * dm..(b + 1) * s_n * dm];
        for s in 0..s_n {
            for p in 0..dm {
                let a = xn[s * dm + p];
                let mut acc = 0.0f64;
                for j in 0..hd {
                    g.dwq[p * hd + j] += a * dq[s * hd + j];
                    g.dwk[p * hd + j] += a * dk[s * hd + j];
                    g.dwv[p * hd + j] += a * dv[s * hd + j];
                    acc += dq[s * hd + j] * f64::from(c.wq[p * hd + j])
                        + dk[s * hd + j] * f64::from(c.wk[p * hd + j])
                        + dv[s * hd + j] * f64::from(c.wv[p * hd + j]);
                }
                dxn[s * dm + p] = acc;
            }
        }

        // RMSNorm backward: with g_p = dx_norm_p * gamma_p and r = rms,
        // dx_p = g_p / r - x_p * sum_p(g_p x_p) / (D r^3).
        let dx = &mut g.dx[b * s_n * dm..(b + 1) * s_n * dm];
        for s in 0..s_n {
            let mut dot = 0.0f64;
            for p in 0..dm {
                dot += dxn[s * dm + p] * f64::from(c.gamma[p]) * f64::from(x[s * dm + p]);
            }
            let r = rms[s];
            for p in 0..dm {
                let gp = dxn[s * dm + p] * f64::from(c.gamma[p]);
                dx[s * dm + p] = gp / r - f64::from(x[s * dm + p]) * dot / (dm as f64 * r * r * r);
            }
        }

        g.dq[b * s_n * hd..(b + 1) * s_n * hd].copy_from_slice(&dq);
        g.dk[b * s_n * hd..(b + 1) * s_n * hd].copy_from_slice(&dk);
        g.dv[b * s_n * hd..(b + 1) * s_n * hd].copy_from_slice(&dv);
    }
    g
}

/// `|got - want|_2 / |want|_2`. Panics on a non-finite `got`, which a ratio
/// would otherwise turn into a NaN that fails no `<` comparison's intent
/// clearly, and on an all-zero `want`, which has no relative error.
pub fn rel_l2(name: &str, got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "{name}: length");
    if let Some(i) = got.iter().position(|v| !v.is_finite()) {
        panic!("{name}: non-finite value {} at index {i}", got[i]);
    }
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    for (&g, &w) in got.iter().zip(want) {
        let d = f64::from(g) - w;
        num += d * d;
        den += w * w;
    }
    assert!(den > 0.0, "{name}: the reference is all zero");
    (num / den).sqrt()
}

/// Fraction of exactly-zero entries.
pub fn zero_fraction(xs: &[f32]) -> f64 {
    xs.iter().filter(|v| **v == 0.0).count() as f64 / xs.len().max(1) as f64
}

// ── PyTorch float64 autograd literals ──────────────────────────────────────
//
// Per tensor: its L2 norm, and three (index, value) samples of the flattened
// tensor. Inputs: x = det_seq(42), Wq/Wk/Wv = det_seq(43/44/45),
// gamma = gamma_seq, dO = d_o_seq(magnitude), eps = 1e-5 (f32).

// S32_HD32_NONCAUSAL_1E6: seq=32 hd=32 d_model=32 causal=False dO x 1e-06 batch=1 rope=False
const S32_HD32_NONCAUSAL_1E6: &[(&str, f64, [(usize, f64); 3])] = &[
    ("o", 28.693363063489812, [(1, 0.15537435602280858), (515, 1.976291944410024), (1022, -0.4410521736449225)]),
    ("dq", 7.316117498062245e-06, [(1, -3.840651027420346e-07), (515, -4.39187612737427e-08), (1022, -5.5417037691673765e-08)]),
    ("dk", 8.10315379315088e-06, [(1, -2.0010405553712444e-07), (515, -5.883636401445171e-08), (1022, 5.5262324847763506e-08)]),
    ("dv", 4.861985887234666e-06, [(1, -2.1835468093392342e-07), (515, 4.51528123847062e-08), (1022, -1.1444837714251861e-07)]),
    ("dwq", 4.1752330483103804e-05, [(1, -2.677721456363939e-06), (515, 1.0841425698027245e-06), (1022, -1.5887972165043473e-06)]),
    ("dwk", 4.619411480610333e-05, [(1, 4.944585202716834e-06), (515, 6.071623285678344e-07), (1022, 2.090849338991005e-08)]),
    ("dwv", 2.9615410105725048e-05, [(1, -6.794728826894626e-07), (515, 6.090857734077894e-07), (1022, 9.828847809755967e-08)]),
    ("dx", 6.188048753505943e-05, [(1, -1.914101014587729e-06), (515, -1.2105302818323555e-06), (1022, -2.725358348603201e-07)]),
    ("dx_norm", 1.833141431075105e-05, [(1, -4.7050147656960535e-07), (515, -3.936248037096787e-07), (1022, 5.8238014217955904e-08)]),
];
// S32_HD32_CAUSAL_1E7: seq=32 hd=32 d_model=32 causal=True dO x 1e-07 batch=1 rope=False
const S32_HD32_CAUSAL_1E7: &[(&str, f64, [(usize, f64); 3])] = &[
    ("o", 34.68368342014046, [(1, -0.5492001017171542), (515, 0.6914057703633103), (1022, -0.4410521736449225)]),
    ("dq", 7.657963454397016e-07, [(1, 0.0), (515, -1.89801539609361e-08), (1022, -5.541704036582477e-09)]),
    ("dk", 9.085308395449071e-07, [(1, 3.849657613162824e-08), (515, -1.2451231119217511e-08), (1022, -6.760091316493574e-14)]),
    ("dv", 6.055812705772303e-07, [(1, -8.554210362035117e-08), (515, 6.682707767317943e-09), (1022, 7.756990001674922e-13)]),
    ("dwq", 4.6480541173358e-06, [(1, -2.4122791980187055e-07), (515, 7.657594121291914e-08), (1022, -3.368295777892249e-08)]),
    ("dwk", 5.356786257207007e-06, [(1, 1.5518254129628312e-07), (515, -5.58110706160775e-08), (1022, 6.944018329710509e-08)]),
    ("dwv", 3.592900371463531e-06, [(1, -1.4250428263780523e-07), (515, 1.1324028301378759e-07), (1022, 7.649951113962286e-08)]),
    ("dx", 6.819906877349134e-06, [(1, -1.4422752967807817e-07), (515, 4.9383540521280473e-08), (1022, 4.146550934409606e-08)]),
    ("dx_norm", 1.9704751983767285e-06, [(1, -3.358876325349122e-08), (515, 6.57195116603343e-10), (1022, 1.6390602547040023e-08)]),
];
// S128_HD32_CAUSAL_1E6: seq=128 hd=32 d_model=32 causal=True dO x 1e-06 batch=1 rope=False
const S128_HD32_CAUSAL_1E6: &[(&str, f64, [(usize, f64); 3])] = &[
    ("o", 59.9050869501488, [(1, -0.5492001017171542), (2051, -0.8813600829996554), (4094, 0.09279471083957727)]),
    ("dq", 1.4094119802872857e-05, [(1, 0.0), (2051, -1.7399095214667348e-08), (4094, -2.228034587320969e-08)]),
    ("dk", 1.4322593382064692e-05, [(1, 3.2764528523287333e-07), (2051, -1.4170306664284047e-08), (4094, -6.007179622454394e-10)]),
    ("dv", 9.08593281607397e-06, [(1, -9.11102706966037e-07), (2051, 6.358938030346377e-09), (4094, 3.599472363856609e-11)]),
    ("dwq", 7.90489445017038e-05, [(1, -4.7183662947924343e-07), (515, -7.342170275139119e-07), (1022, 2.6599962326285648e-06)]),
    ("dwk", 8.381971346837835e-05, [(1, 3.1636539432759745e-06), (515, -7.714346141504823e-07), (1022, 3.921190754807493e-07)]),
    ("dwv", 5.950837327955344e-05, [(1, -3.777479179990873e-06), (515, 2.57878197241778e-06), (1022, 4.637654234833252e-07)]),
    ("dx", 0.00011361255923190874, [(1, -1.9033847329468324e-06), (2051, 1.7695512326891683e-06), (4094, 1.6863412170780822e-06)]),
    ("dx_norm", 3.326141850409622e-05, [(1, -3.7836179478641986e-07), (2051, 4.534065974270317e-07), (4094, 5.586958990376914e-07)]),
];
// S32_HD32_DM64_CAUSAL_1E6: seq=32 hd=32 d_model=64 causal=True dO x 1e-06 batch=1 rope=False
const S32_HD32_DM64_CAUSAL_1E6: &[(&str, f64, [(usize, f64); 3])] = &[
    ("o", 68.4828399473364, [(1, -2.7361864025126916), (515, 0.5387373186512753), (1022, -1.6945717158461655)]),
    ("dq", 8.205167558497386e-06, [(1, 0.0), (515, 6.924923347831655e-08), (1022, -2.0877923528687243e-07)]),
    ("dk", 7.800801963796877e-06, [(1, 4.4302154662214644e-08), (515, 7.618514755641893e-08), (1022, 3.6985869599215195e-12)]),
    ("dv", 8.025701355079286e-06, [(1, -3.1424286712983693e-07), (515, 3.3998312797905336e-09), (1022, 2.1415199566423745e-12)]),
    ("dwq", 6.675755802811383e-05, [(1, 8.567034932090117e-07), (1027, -8.6886974301204e-07), (2046, 5.053015767420149e-09)]),
    ("dwk", 6.344919098767858e-05, [(1, -8.606081664495275e-07), (1027, 2.2629068159637343e-07), (2046, -1.5138237257905348e-07)]),
    ("dwv", 6.776761624737402e-05, [(1, -3.0160302179180425e-06), (1027, -9.861714467598472e-07), (2046, -8.509733270849183e-08)]),
    ("dx", 0.00011255164115283571, [(1, 1.6314120780526725e-06), (1027, 6.853044634174114e-08), (2046, 7.850659779256469e-07)]),
    ("dx_norm", 3.2149638893328426e-05, [(1, 4.6739835221821784e-07), (1027, 4.560546015560033e-09), (2046, 1.794187679552973e-07)]),
];
// S32_HD32_ROPE_CAUSAL_1E6: seq=32 hd=32 d_model=32 causal=True dO x 1e-06 batch=1 rope=True
const S32_HD32_ROPE_CAUSAL_1E6: &[(&str, f64, [(usize, f64); 3])] = &[
    ("o", 37.43221648650992, [(1, -0.5492001017171542), (515, 0.8571946499282307), (1022, -0.3761225691361709)]),
    ("dq", 6.7476065719373425e-06, [(1, 0.0), (515, -6.46154187872583e-08), (1022, -4.2822951385440774e-08)]),
    ("dk", 7.332778246330664e-06, [(1, -2.385567110907401e-07), (515, -7.093541650424288e-08), (1022, 9.46923436206697e-12)]),
    ("dv", 6.227552807857112e-06, [(1, -9.332183944265342e-07), (515, 8.932916887743207e-08), (1022, 3.248467108101791e-11)]),
    ("dwq", 3.7170994655309395e-05, [(1, 5.796341993864878e-07), (515, 3.9037388180874236e-07), (1022, -1.6481222521302016e-06)]),
    ("dwk", 4.235339634627925e-05, [(1, -1.319523666176619e-06), (515, -1.4439698967213513e-07), (1022, -1.0207946398670578e-06)]),
    ("dwv", 3.714746649489214e-05, [(1, -2.367619049038312e-06), (515, 9.202483349821972e-07), (1022, 2.8256601711423025e-07)]),
    ("dx", 6.88773531382394e-05, [(1, -5.1726273355710196e-08), (515, -1.0520177403247268e-06), (1022, -3.6317532901031114e-07)]),
    ("dx_norm", 1.9624247100019012e-05, [(1, 1.6936004161930032e-07), (515, -3.2953515540148284e-07), (1022, -1.5096644550491536e-07)]),
];
// S32_HD32_BATCH2_CAUSAL_1E6: seq=32 hd=32 d_model=32 causal=True dO x 1e-06 batch=2 rope=False
const S32_HD32_BATCH2_CAUSAL_1E6: &[(&str, f64, [(usize, f64); 3])] = &[
    ("o", 48.82290261977923, [(1, -0.5492001017171542), (1027, -2.1216229550870906), (2046, 0.014332302287479708)]),
    ("dq", 1.1977955072805968e-05, [(1, 0.0), (1027, 0.0), (2046, 3.859574412858245e-08)]),
    ("dk", 1.2428889692173102e-05, [(1, 3.8496576415291335e-07), (1027, -1.8283734867466146e-08), (2046, 3.5481517940590105e-11)]),
    ("dv", 8.485797378304058e-06, [(1, -8.554210112924467e-07), (1027, 3.4795223054535907e-07), (2046, 3.3467664460565604e-10)]),
    ("dwq", 7.142050939429562e-05, [(1, -2.2691745108311086e-06), (515, -1.6568004410073808e-06), (1022, -4.0224083459251176e-07)]),
    ("dwk", 7.60632441536282e-05, [(1, 4.036082362563214e-06), (515, -6.355668373826782e-07), (1022, 5.134921811362197e-07)]),
    ("dwv", 4.9706331079750544e-05, [(1, -5.605797608926608e-07), (515, 1.048995759551675e-06), (1022, 9.598453808181012e-08)]),
    ("dx", 9.984236144334898e-05, [(1, -1.4422753798216555e-06), (1027, -1.9021331485031898e-06), (2046, -2.270271415061483e-07)]),
    ("dx_norm", 2.8711867483368464e-05, [(1, -3.3588765092201773e-07), (1027, -5.281863111442481e-07), (2046, -8.716713463595235e-08)]),
];

type Pinned = &'static [(&'static str, f64, [(usize, f64); 3])];

#[cfg(test)]
#[allow(clippy::too_many_arguments)]
fn check_against_pytorch(
    name: &str,
    batch: usize,
    seq: usize,
    head_dim: usize,
    d_model: usize,
    causal: bool,
    rope: bool,
    magnitude: f32,
    pinned: Pinned,
) {
    let x = det_seq(42, batch * seq * d_model);
    let wq = det_seq(43, d_model * head_dim);
    let wk = det_seq(44, d_model * head_dim);
    let wv = det_seq(45, d_model * head_dim);
    let gamma = gamma_seq(d_model);
    let d_o = d_o_seq(batch * seq * head_dim, magnitude);
    let tables = rope_tables(seq, head_dim);
    let g = fused_backward_f64(&FusedCase {
        batch,
        seq,
        head_dim,
        d_model,
        causal,
        eps: 1e-5,
        x: &x,
        wq: &wq,
        wk: &wk,
        wv: &wv,
        gamma: &gamma,
        d_o: &d_o,
        rope: rope.then_some((&tables.0, &tables.1)),
    });
    for &(key, want_norm, samples) in pinned {
        let got: &[f64] = match key {
            "o" => &g.o,
            "dq" => &g.dq,
            "dk" => &g.dk,
            "dv" => &g.dv,
            "dwq" => &g.dwq,
            "dwk" => &g.dwk,
            "dwv" => &g.dwv,
            "dx" => &g.dx,
            "dx_norm" => &g.dx_norm,
            other => panic!("unknown tensor {other}"),
        };
        let norm = got.iter().map(|v| v * v).sum::<f64>().sqrt();
        assert!(
            ((norm - want_norm) / want_norm).abs() < 1e-10,
            "{name} {key}: norm {norm:e} vs PyTorch {want_norm:e}"
        );
        for (i, want) in samples {
            let have = got[i];
            // Relative to the tensor's typical element, so an exact zero in
            // the reference (a causally masked cell) is still checked.
            let typical = want_norm / (got.len() as f64).sqrt();
            assert!(
                (have - want).abs() < 1e-9 * typical,
                "{name} {key}[{i}]: {have:e} vs PyTorch {want:e}"
            );
        }
    }
}

#[test]
fn f64_reference_matches_pytorch_noncausal_at_1e6() {
    check_against_pytorch("s32 non-causal", 1, 32, 32, 32, false, false, 1e-6, S32_HD32_NONCAUSAL_1E6);
}

#[test]
fn f64_reference_matches_pytorch_causal_at_1e7() {
    check_against_pytorch("s32 causal", 1, 32, 32, 32, true, false, 1e-7, S32_HD32_CAUSAL_1E7);
}

#[test]
fn f64_reference_matches_pytorch_four_tiles() {
    check_against_pytorch("s128 causal", 1, 128, 32, 32, true, false, 1e-6, S128_HD32_CAUSAL_1E6);
}

#[test]
fn f64_reference_matches_pytorch_wide_model() {
    check_against_pytorch("d_model 64", 1, 32, 32, 64, true, false, 1e-6, S32_HD32_DM64_CAUSAL_1E6);
}

#[test]
fn f64_reference_matches_pytorch_with_rope() {
    check_against_pytorch("rope", 1, 32, 32, 32, true, true, 1e-6, S32_HD32_ROPE_CAUSAL_1E6);
}

#[test]
fn f64_reference_matches_pytorch_batch_of_two() {
    check_against_pytorch("batch 2", 2, 32, 32, 32, true, false, 1e-6, S32_HD32_BATCH2_CAUSAL_1E6);
}

/// The reference must be able to fail: a dropped RMSNorm weight is caught.
#[test]
fn f64_reference_check_rejects_a_wrong_gamma() {
    let r = std::panic::catch_unwind(|| {
        let seq = 32;
        let x = det_seq(42, seq * 32);
        let w = det_seq(43, 32 * 32);
        let ones = vec![1.0f32; 32];
        let d_o = d_o_seq(seq * 32, 1e-6);
        let g = fused_backward_f64(&FusedCase {
            batch: 1, seq, head_dim: 32, d_model: 32, causal: false, eps: 1e-5,
            x: &x, wq: &w, wk: &w, wv: &w, gamma: &ones, d_o: &d_o, rope: None,
        });
        let norm = g.dq.iter().map(|v| v * v).sum::<f64>().sqrt();
        let want = S32_HD32_NONCAUSAL_1E6[1].1;
        assert!(((norm - want) / want).abs() < 1e-10);
    });
    assert!(r.is_err(), "a different model must not match the PyTorch literals");
}
