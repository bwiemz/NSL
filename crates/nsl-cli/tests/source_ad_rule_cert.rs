//! Source-AD rule certification: every differentiable primitive's RAW gradient
//! against an independent f64 oracle, in both AD modes.
//!
//! Each certificate is one primitive spelled as NSL source, with the operands
//! to differentiate and an independent Rust f64 forward of the same
//! expression. The generated program draws its inputs (`randn`, so every
//! element differs), prints them, and takes `grad(operand)` of
//! `sum(EXPR * r)` for a random `r` of the output's shape -- so every output
//! element is weighted differently and no symmetry can hide a wrong rule. The
//! test reads back the inputs, the loss and the raw gradients, and holds them
//! to the oracle: the loss to the oracle's forward (a wrong FORWARD fails
//! here, which a gradient check alone could miss), and each gradient to
//! central differences of the oracle in f64.
//!
//! Both modes run every certificate: `nsl run` (tape AD) and `nsl run
//! --source-ad`. A source-AD run must prove it engaged -- one
//! `Using source-to-source AD for grad block` per block and no fallback line
//! -- or a silent fall back to the tape would certify the tape twice.
//!
//! No optimizer is involved: AdamW's scale invariance hides gradient-scale
//! bugs, and these are raw gradients.
//!
//! Known failures are a RATCHET, listed per certificate and mode with the
//! reason: a listed failure must still fail (a fix flips it, and the list must
//! say so), and an unlisted one must pass.
//!
//! Coverage: `nsl_codegen::ad_rules::ad_cert_status` gives every `PrimalOp`
//! variant a status with no wildcard arm, so a new op does not compile without
//! one, and its unit tests tie the statuses to the rules that exist.
//! `certificates_match_the_codegen_inventory` below holds the certificate
//! names on both sides to each other.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;

use nsl_codegen::ad_rules::{AdCertStatus, ad_cert_inventory};

// ---------------------------------------------------------------------------
// Certificate table
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mode {
    Tape,
    Source,
}

#[derive(Clone, Copy)]
enum Init {
    /// `randn(shape)`.
    Randn,
    /// `abs(randn(shape)) + 0.5`, for operands a primitive needs positive.
    Positive,
    /// An NSL expression yielding the tensor (index tensors).
    Expr(&'static str),
}

#[derive(Clone, Copy)]
struct Input {
    name: &'static str,
    shape: &'static [usize],
    init: Init,
}

/// Row-major data and its shape.
#[derive(Clone, Debug)]
struct Arr {
    shape: Vec<usize>,
    data: Vec<f64>,
}

type Env = HashMap<&'static str, Arr>;

struct Cert {
    name: &'static str,
    inputs: Vec<Input>,
    expr: &'static str,
    wrt: &'static [&'static str],
    out_shape: &'static [usize],
    oracle: fn(&Env) -> Arr,
    /// Known failures: (mode, why). Empty = must pass in both modes.
    known: &'static [(Mode, &'static str)],
    /// Lines before the inputs (imports).
    prelude: &'static str,
}

const fn inp(name: &'static str, shape: &'static [usize]) -> Input {
    Input {
        name,
        shape,
        init: Init::Randn,
    }
}
const fn pos(name: &'static str, shape: &'static [usize]) -> Input {
    Input {
        name,
        shape,
        init: Init::Positive,
    }
}
const fn idx(name: &'static str, shape: &'static [usize], expr: &'static str) -> Input {
    Input {
        name,
        shape,
        init: Init::Expr(expr),
    }
}

// ---------------------------------------------------------------------------
// Oracle helpers (f64, row-major)
// ---------------------------------------------------------------------------

fn get<'a>(env: &'a Env, name: &str) -> &'a Arr {
    env.get(name)
        .unwrap_or_else(|| panic!("oracle input {name} missing"))
}

fn numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut s = vec![1; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

/// Right-aligned numpy broadcasting of two shapes.
fn broadcast_shape(a: &[usize], b: &[usize]) -> Vec<usize> {
    let n = a.len().max(b.len());
    (0..n)
        .map(|i| {
            let da = if i + a.len() >= n {
                a[i + a.len() - n]
            } else {
                1
            };
            let db = if i + b.len() >= n {
                b[i + b.len() - n]
            } else {
                1
            };
            assert!(
                da == db || da == 1 || db == 1,
                "shapes {a:?} {b:?} do not broadcast"
            );
            da.max(db)
        })
        .collect()
}

/// The element of `a` that output index `out_idx` (of shape `out`) reads.
fn bcast_at(a: &Arr, out: &[usize], out_idx: &[usize]) -> f64 {
    let off = out.len() - a.shape.len();
    let st = strides(&a.shape);
    let mut flat = 0;
    for (d, &sz) in a.shape.iter().enumerate() {
        let i = if sz == 1 { 0 } else { out_idx[d + off] };
        flat += i * st[d];
    }
    a.data[flat]
}

fn unravel(mut flat: usize, shape: &[usize]) -> Vec<usize> {
    let mut idx = vec![0; shape.len()];
    for d in (0..shape.len()).rev() {
        idx[d] = flat % shape[d];
        flat /= shape[d];
    }
    idx
}

fn binary(a: &Arr, b: &Arr, f: impl Fn(f64, f64) -> f64) -> Arr {
    let shape = broadcast_shape(&a.shape, &b.shape);
    let data = (0..numel(&shape))
        .map(|k| {
            let ix = unravel(k, &shape);
            f(bcast_at(a, &shape, &ix), bcast_at(b, &shape, &ix))
        })
        .collect();
    Arr { shape, data }
}

fn unary(a: &Arr, f: impl Fn(f64) -> f64) -> Arr {
    Arr {
        shape: a.shape.clone(),
        data: a.data.iter().map(|&v| f(v)).collect(),
    }
}

fn scalar(v: f64) -> Arr {
    Arr {
        shape: vec![],
        data: vec![v],
    }
}

/// Batched matmul with numpy broadcasting over the leading dims.
fn matmul(a: &Arr, b: &Arr) -> Arr {
    let (ar, br) = (a.shape.len(), b.shape.len());
    let (m, k) = (a.shape[ar - 2], a.shape[ar - 1]);
    let (k2, n) = (b.shape[br - 2], b.shape[br - 1]);
    assert_eq!(k, k2);
    let batch = broadcast_shape(&a.shape[..ar - 2], &b.shape[..br - 2]);
    let nb = numel(&batch);
    let mut data = Vec::with_capacity(nb * m * n);
    let a_batch = Arr {
        shape: a.shape[..ar - 2].to_vec(),
        data: (0..numel(&a.shape[..ar - 2])).map(|i| i as f64).collect(),
    };
    let b_batch = Arr {
        shape: b.shape[..br - 2].to_vec(),
        data: (0..numel(&b.shape[..br - 2])).map(|i| i as f64).collect(),
    };
    for bi in 0..nb {
        let ix = unravel(bi, &batch);
        let ai = bcast_at(&a_batch, &batch, &ix) as usize;
        let bj = bcast_at(&b_batch, &batch, &ix) as usize;
        for i in 0..m {
            for j in 0..n {
                data.push(
                    (0..k)
                        .map(|t| a.data[ai * m * k + i * k + t] * b.data[bj * k * n + t * n + j])
                        .sum(),
                );
            }
        }
    }
    let mut shape = batch;
    shape.extend([m, n]);
    Arr { shape, data }
}

/// Reduce `a` over `dim` with `f` over the slice (no keepdim).
fn reduce_dim(a: &Arr, dim: usize, f: impl Fn(&[f64]) -> f64) -> Arr {
    let mut out_shape = a.shape.clone();
    out_shape.remove(dim);
    let st = strides(&a.shape);
    let data = (0..numel(&out_shape))
        .map(|k| {
            let oi = unravel(k, &out_shape);
            let mut full = oi.clone();
            full.insert(dim, 0);
            let base: usize = full.iter().zip(&st).map(|(i, s)| i * s).sum();
            let vals: Vec<f64> = (0..a.shape[dim])
                .map(|t| a.data[base + t * st[dim]])
                .collect();
            f(&vals)
        })
        .collect();
    Arr {
        shape: out_shape,
        data,
    }
}

/// Apply `f` to every 1-D slice along `dim` (same shape out).
fn along_dim(a: &Arr, dim: usize, f: impl Fn(&[f64]) -> Vec<f64>) -> Arr {
    let st = strides(&a.shape);
    let mut out = a.data.clone();
    let mut outer = a.shape.clone();
    outer.remove(dim);
    for k in 0..numel(&outer) {
        let mut full = unravel(k, &outer);
        full.insert(dim, 0);
        let base: usize = full.iter().zip(&st).map(|(i, s)| i * s).sum();
        let vals: Vec<f64> = (0..a.shape[dim])
            .map(|t| a.data[base + t * st[dim]])
            .collect();
        for (t, v) in f(&vals).into_iter().enumerate() {
            out[base + t * st[dim]] = v;
        }
    }
    Arr {
        shape: a.shape.clone(),
        data: out,
    }
}

fn softmax_vec(v: &[f64]) -> Vec<f64> {
    let m = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let e: Vec<f64> = v.iter().map(|x| (x - m).exp()).collect();
    let s: f64 = e.iter().sum();
    e.iter().map(|x| x / s).collect()
}

fn sigmoid_ref(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The CPU GELU NSL implements: the tanh approximation.
fn gelu_tanh(x: f64) -> f64 {
    let c = (2.0 / std::f64::consts::PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

fn transpose2(a: &Arr, d0: usize, d1: usize) -> Arr {
    let mut shape = a.shape.clone();
    shape.swap(d0, d1);
    let st = strides(&a.shape);
    let data = (0..numel(&shape))
        .map(|k| {
            let mut ix = unravel(k, &shape);
            ix.swap(d0, d1);
            a.data[ix.iter().zip(&st).map(|(i, s)| i * s).sum::<usize>()]
        })
        .collect();
    Arr { shape, data }
}

fn concat(a: &Arr, b: &Arr, dim: usize) -> Arr {
    let mut shape = a.shape.clone();
    shape[dim] += b.shape[dim];
    let (sa, sb) = (strides(&a.shape), strides(&b.shape));
    let data = (0..numel(&shape))
        .map(|k| {
            let mut ix = unravel(k, &shape);
            if ix[dim] < a.shape[dim] {
                a.data[ix.iter().zip(&sa).map(|(i, s)| i * s).sum::<usize>()]
            } else {
                ix[dim] -= a.shape[dim];
                b.data[ix.iter().zip(&sb).map(|(i, s)| i * s).sum::<usize>()]
            }
        })
        .collect();
    Arr { shape, data }
}

fn layernorm_rows(x: &Arr, w: &Arr, b: Option<&Arr>, eps: f64, rms: bool) -> Arr {
    let d = *x.shape.last().unwrap();
    let mut out = x.data.clone();
    for row in 0..x.data.len() / d {
        let v = &x.data[row * d..(row + 1) * d];
        let mean = if rms {
            0.0
        } else {
            v.iter().sum::<f64>() / d as f64
        };
        let var = v.iter().map(|t| (t - mean) * (t - mean)).sum::<f64>() / d as f64;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..d {
            out[row * d + j] = (v[j] - mean) * inv * w.data[j] + b.map_or(0.0, |b| b.data[j]);
        }
    }
    Arr {
        shape: x.shape.clone(),
        data: out,
    }
}

/// NCHW conv2d, weight [Cout, Cin, kh, kw].
fn conv2d_ref(x: &Arr, w: &Arr, b: &Arr, stride: usize, pad: usize) -> Arr {
    let (n, cin, h, wd) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let (cout, _, kh, kw) = (w.shape[0], w.shape[1], w.shape[2], w.shape[3]);
    let oh = (h + 2 * pad - kh) / stride + 1;
    let ow = (wd + 2 * pad - kw) / stride + 1;
    let mut data = vec![0.0; n * cout * oh * ow];
    for ni in 0..n {
        for co in 0..cout {
            for oy in 0..oh {
                for ox in 0..ow {
                    let mut acc = b.data[co];
                    for ci in 0..cin {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let iy = (oy * stride + ky) as isize - pad as isize;
                                let ix = (ox * stride + kx) as isize - pad as isize;
                                if iy < 0 || ix < 0 || iy >= h as isize || ix >= wd as isize {
                                    continue;
                                }
                                let xv =
                                    x.data[((ni * cin + ci) * h + iy as usize) * wd + ix as usize];
                                acc += xv * w.data[((co * cin + ci) * kh + ky) * kw + kx];
                            }
                        }
                    }
                    data[((ni * cout + co) * oh + oy) * ow + ox] = acc;
                }
            }
        }
    }
    Arr {
        shape: vec![n, cout, oh, ow],
        data,
    }
}

/// Scaled dot-product attention over [B, H, S, D].
fn sdpa_ref(q: &Arr, k: &Arr, v: &Arr, scale: f64, causal: bool) -> Arr {
    let (bsz, h, s, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let mut out = vec![0.0; q.data.len()];
    for bh in 0..bsz * h {
        let base = bh * s * d;
        for i in 0..s {
            let scores: Vec<f64> = (0..s)
                .map(|j| {
                    if causal && j > i {
                        f64::NEG_INFINITY
                    } else {
                        (0..d)
                            .map(|t| q.data[base + i * d + t] * k.data[base + j * d + t])
                            .sum::<f64>()
                            * scale
                    }
                })
                .collect();
            let p = softmax_vec(&scores);
            for t in 0..d {
                out[base + i * d + t] = (0..s).map(|j| p[j] * v.data[base + j * d + t]).sum();
            }
        }
    }
    Arr {
        shape: q.shape.clone(),
        data: out,
    }
}

// ---------------------------------------------------------------------------
// The certificates
// ---------------------------------------------------------------------------

macro_rules! oracle {
    (|$env:ident| $body:expr) => {{
        fn f($env: &Env) -> Arr {
            $body
        }
        f
    }};
}

const XY_ROW: &[Input] = &[inp("x", &[3, 4]), inp("y", &[4])];
const XY_COL: &[Input] = &[inp("x", &[3, 4]), inp("y", &[3, 1])];
const XY_SAME: &[Input] = &[inp("x", &[3, 4]), inp("y", &[3, 4])];
const XY_ONE: &[Input] = &[inp("x", &[3, 4]), inp("y", &[1])];
const X_34: &[Input] = &[inp("x", &[3, 4])];
const X_POS: &[Input] = &[pos("x", &[3, 4])];

/// One row per certificate; kept one-per-line so the table reads as a table.
#[rustfmt::skip]
fn certs() -> Vec<Cert> {
    vec![
        // --- binary elementwise, broadcast variants ---------------------
        Cert { name: "add_same", inputs: XY_SAME.to_vec(), expr: "x + y", wrt: &["x", "y"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(get(e, "x"), get(e, "y"), |a, b| a + b)), known: &[], prelude: "" },
        Cert { name: "add_row", inputs: XY_ROW.to_vec(), expr: "x + y", wrt: &["x", "y"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(get(e, "x"), get(e, "y"), |a, b| a + b)), known: &[], prelude: "" },
        Cert { name: "add_col", inputs: XY_COL.to_vec(), expr: "x + y", wrt: &["x", "y"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(get(e, "x"), get(e, "y"), |a, b| a + b)), known: &[], prelude: "" },
        Cert { name: "add_one", inputs: XY_ONE.to_vec(), expr: "x + y", wrt: &["x", "y"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(get(e, "x"), get(e, "y"), |a, b| a + b)), known: &[], prelude: "" },
        Cert { name: "add_literal", inputs: X_34.to_vec(), expr: "x + 2.0", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), |a| a + 2.0)), known: &[], prelude: "" },
        Cert { name: "sub_row", inputs: XY_ROW.to_vec(), expr: "x - y", wrt: &["x", "y"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(get(e, "x"), get(e, "y"), |a, b| a - b)), known: &[], prelude: "" },
        Cert { name: "sub_col", inputs: XY_COL.to_vec(), expr: "y - x", wrt: &["x", "y"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(get(e, "y"), get(e, "x"), |a, b| a - b)), known: &[], prelude: "" },
        Cert { name: "mul_row", inputs: XY_ROW.to_vec(), expr: "x * y", wrt: &["x", "y"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(get(e, "x"), get(e, "y"), |a, b| a * b)), known: &[], prelude: "" },
        Cert { name: "mul_col", inputs: XY_COL.to_vec(), expr: "x * y", wrt: &["x", "y"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(get(e, "x"), get(e, "y"), |a, b| a * b)), known: &[], prelude: "" },
        Cert { name: "mul_literal", inputs: X_34.to_vec(), expr: "x * 0.5", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), |a| a * 0.5)), known: &[], prelude: "" },
        Cert { name: "div_row", inputs: vec![inp("x", &[3, 4]), pos("y", &[4])], expr: "x / y", wrt: &["x", "y"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(get(e, "x"), get(e, "y"), |a, b| a / b)), known: &[], prelude: "" },
        Cert { name: "div_col", inputs: vec![inp("x", &[3, 4]), pos("y", &[3, 1])], expr: "x / y", wrt: &["x", "y"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(get(e, "x"), get(e, "y"), |a, b| a / b)), known: &[], prelude: "" },
        Cert { name: "div_numerator_broadcast", inputs: vec![inp("x", &[4]), pos("y", &[3, 4])], expr: "x / y", wrt: &["x", "y"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(get(e, "x"), get(e, "y"), |a, b| a / b)), known: &[], prelude: "" },
        // --- matmul ------------------------------------------------------
        Cert { name: "matmul_2d", inputs: vec![inp("a", &[3, 4]), inp("b", &[4, 5])], expr: "a @ b", wrt: &["a", "b"], out_shape: &[3, 5],
            oracle: oracle!(|e| matmul(get(e, "a"), get(e, "b"))), known: &[], prelude: "" },
        Cert { name: "matmul_3d_2d", inputs: vec![inp("a", &[2, 3, 4]), inp("b", &[4, 5])], expr: "a @ b", wrt: &["a", "b"], out_shape: &[2, 3, 5],
            oracle: oracle!(|e| matmul(get(e, "a"), get(e, "b"))), known: &[], prelude: "" },
        Cert { name: "matmul_2d_3d", inputs: vec![inp("a", &[3, 4]), inp("b", &[2, 4, 5])], expr: "a @ b", wrt: &["a", "b"], out_shape: &[2, 3, 5],
            oracle: oracle!(|e| matmul(get(e, "a"), get(e, "b"))), known: &[], prelude: "" },
        Cert { name: "matmul_3d_3d", inputs: vec![inp("a", &[2, 3, 4]), inp("b", &[2, 4, 5])], expr: "a @ b", wrt: &["a", "b"], out_shape: &[2, 3, 5],
            oracle: oracle!(|e| matmul(get(e, "a"), get(e, "b"))), known: &[], prelude: "" },
        // --- unary -------------------------------------------------------
        Cert { name: "neg", inputs: X_34.to_vec(), expr: "-x", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), |a| -a)), known: &[], prelude: "" },
        Cert { name: "relu", inputs: X_34.to_vec(), expr: "relu(x)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), |a| a.max(0.0))), known: &[], prelude: "" },
        Cert { name: "sigmoid", inputs: X_34.to_vec(), expr: "sigmoid(x)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), sigmoid_ref)), known: &[], prelude: "" },
        Cert { name: "tanh", inputs: X_34.to_vec(), expr: "tanh(x)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), f64::tanh)), known: &[], prelude: "" },
        Cert { name: "exp", inputs: X_34.to_vec(), expr: "exp(x)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), f64::exp)), known: &[], prelude: "" },
        Cert { name: "log", inputs: X_POS.to_vec(), expr: "log(x)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), f64::ln)), known: &[], prelude: "" },
        Cert { name: "sqrt", inputs: X_POS.to_vec(), expr: "sqrt(x)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), f64::sqrt)), known: &[], prelude: "" },
        Cert { name: "gelu", inputs: X_34.to_vec(), expr: "gelu(x)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), gelu_tanh)), known: &[], prelude: "" },
        Cert { name: "silu", inputs: X_34.to_vec(), expr: "silu(x)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), |a| a * sigmoid_ref(a))), known: &[], prelude: "" },
        Cert { name: "abs", inputs: X_34.to_vec(), expr: "abs(x)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), f64::abs)), known: &[], prelude: "" },
        Cert { name: "clamp", inputs: X_34.to_vec(), expr: "clamp(x, -0.5, 0.5)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), |a| a.clamp(-0.5, 0.5))), known: &[], prelude: "" },
        Cert { name: "cos", inputs: X_34.to_vec(), expr: "tensor_cos(x)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), f64::cos)), known: &[], prelude: "" },
        Cert { name: "sin", inputs: X_34.to_vec(), expr: "tensor_sin(x)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), f64::sin)), known: &[], prelude: "" },
        Cert { name: "rotate_half", inputs: X_34.to_vec(), expr: "rotate_half(x)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| along_dim(get(e, "x"), 1, |v| {
                let h = v.len() / 2;
                v[h..].iter().map(|a| -a).chain(v[..h].iter().copied()).collect()
            })), known: &[], prelude: "" },
        // --- shape -------------------------------------------------------
        Cert { name: "transpose", inputs: vec![inp("x", &[3, 4])], expr: "x.transpose(0, 1)", wrt: &["x"], out_shape: &[4, 3],
            oracle: oracle!(|e| transpose2(get(e, "x"), 0, 1)), known: &[], prelude: "" },
        Cert { name: "transpose_3d", inputs: vec![inp("x", &[2, 3, 4])], expr: "x.transpose(1, 2)", wrt: &["x"], out_shape: &[2, 4, 3],
            oracle: oracle!(|e| transpose2(get(e, "x"), 1, 2)), known: &[], prelude: "" },
        Cert { name: "reshape", inputs: X_34.to_vec(), expr: "x.reshape([2, 6])", wrt: &["x"], out_shape: &[2, 6],
            oracle: oracle!(|e| Arr { shape: vec![2, 6], data: get(e, "x").data.clone() }), known: &[], prelude: "" },
        Cert { name: "unsqueeze", inputs: X_34.to_vec(), expr: "x.unsqueeze(0)", wrt: &["x"], out_shape: &[1, 3, 4],
            oracle: oracle!(|e| Arr { shape: vec![1, 3, 4], data: get(e, "x").data.clone() }), known: &[], prelude: "" },
        Cert { name: "expand", inputs: vec![inp("x", &[1, 4])], expr: "x.expand([3, 4])", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(get(e, "x"), &Arr { shape: vec![3, 4], data: vec![0.0; 12] }, |a, _| a)), known: &[], prelude: "" },
        Cert { name: "contiguous", inputs: X_34.to_vec(), expr: "x.transpose(0, 1).contiguous()", wrt: &["x"], out_shape: &[4, 3],
            oracle: oracle!(|e| transpose2(get(e, "x"), 0, 1)), known: &[], prelude: "" },
        // --- reductions --------------------------------------------------
        Cert { name: "sum_all", inputs: X_34.to_vec(), expr: "sum(x)", wrt: &["x"], out_shape: &[],
            oracle: oracle!(|e| scalar(get(e, "x").data.iter().sum())), known: &[], prelude: "" },
        Cert { name: "mean_all", inputs: X_34.to_vec(), expr: "mean(x)", wrt: &["x"], out_shape: &[],
            oracle: oracle!(|e| scalar(get(e, "x").data.iter().sum::<f64>() / 12.0)), known: &[], prelude: "" },
        Cert { name: "sum_dim", inputs: X_34.to_vec(), expr: "sum(x, 1, 0).reshape([3])", wrt: &["x"], out_shape: &[3],
            oracle: oracle!(|e| reduce_dim(get(e, "x"), 1, |v| v.iter().sum())), known: &[], prelude: "" },
        Cert { name: "mean_dim", inputs: X_34.to_vec(), expr: "mean(x, 0, 0).reshape([4])", wrt: &["x"], out_shape: &[4],
            oracle: oracle!(|e| reduce_dim(get(e, "x"), 0, |v| v.iter().sum::<f64>() / v.len() as f64)), known: &[], prelude: "" },
        Cert { name: "sum_dim_neg", inputs: vec![inp("x", &[2, 3, 4])], expr: "sum(x, -2, 0).reshape([2, 4])", wrt: &["x"], out_shape: &[2, 4],
            oracle: oracle!(|e| reduce_dim(get(e, "x"), 1, |v| v.iter().sum())), known: &[], prelude: "" },
        Cert { name: "sum_dim_keepdim", inputs: X_34.to_vec(), expr: "sum(x, 0, 1)", wrt: &["x"], out_shape: &[1, 4],
            oracle: oracle!(|e| Arr { shape: vec![1, 4], data: reduce_dim(get(e, "x"), 0, |v| v.iter().sum()).data }),
            known: &[(Mode::Source, "a keepdim sum is not extracted: the grad block falls back to the tape")], prelude: "" },
        Cert { name: "sum_dim_last", inputs: X_34.to_vec(), expr: "sum(x, -1, 0).reshape([3])", wrt: &["x"], out_shape: &[3],
            oracle: oracle!(|e| reduce_dim(get(e, "x"), 1, |v| v.iter().sum())), known: &[], prelude: "" },
        Cert { name: "mean_dim_last", inputs: X_34.to_vec(), expr: "mean(x, -1, 0).reshape([3])", wrt: &["x"], out_shape: &[3],
            oracle: oracle!(|e| reduce_dim(get(e, "x"), 1, |v| v.iter().sum::<f64>() / v.len() as f64)), known: &[], prelude: "" },
        Cert { name: "softmax_last", inputs: X_34.to_vec(), expr: "softmax(x, -1)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| along_dim(get(e, "x"), 1, softmax_vec)), known: &[], prelude: "" },
        Cert { name: "softmax_dim0", inputs: X_34.to_vec(), expr: "softmax(x, 0)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| along_dim(get(e, "x"), 0, softmax_vec)), known: &[], prelude: "" },
        Cert { name: "log_softmax_last", inputs: X_34.to_vec(), expr: "log_softmax(x, -1)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| along_dim(get(e, "x"), 1, |v| softmax_vec(v).iter().map(|p| p.ln()).collect())), known: &[], prelude: "" },
        Cert { name: "log_softmax_dim0", inputs: X_34.to_vec(), expr: "log_softmax(x, 0)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| along_dim(get(e, "x"), 0, |v| softmax_vec(v).iter().map(|p| p.ln()).collect())), known: &[], prelude: "" },
        Cert { name: "softmax_mid", inputs: vec![inp("x", &[2, 3, 4])], expr: "softmax(x, 1)", wrt: &["x"], out_shape: &[2, 3, 4],
            oracle: oracle!(|e| along_dim(get(e, "x"), 1, softmax_vec)), known: &[], prelude: "" },
        Cert { name: "log_softmax_mid", inputs: vec![inp("x", &[2, 3, 4])], expr: "log_softmax(x, 1)", wrt: &["x"], out_shape: &[2, 3, 4],
            oracle: oracle!(|e| along_dim(get(e, "x"), 1, |v| softmax_vec(v).iter().map(|p| p.ln()).collect())), known: &[], prelude: "" },
        // --- normalization -----------------------------------------------
        Cert { name: "layernorm", inputs: vec![inp("x", &[3, 4]), inp("w", &[4]), inp("b", &[4])],
            expr: "layernorm(x, w, b, 0.00001)", wrt: &["x", "w", "b"], out_shape: &[3, 4],
            oracle: oracle!(|e| layernorm_rows(get(e, "x"), get(e, "w"), Some(get(e, "b")), 0.00001, false)), known: &[(Mode::Source, "dx is wrong and db keeps the [3, 4] shape (12 values, not 4)")], prelude: "" },
        Cert { name: "layernorm_eps", inputs: vec![inp("x", &[3, 4]), inp("w", &[4]), inp("b", &[4])],
            expr: "layernorm(x, w, b, 0.5)", wrt: &["x", "w", "b"], out_shape: &[3, 4],
            oracle: oracle!(|e| layernorm_rows(get(e, "x"), get(e, "w"), Some(get(e, "b")), 0.5, false)), known: &[(Mode::Source, "the extractor hard-codes eps 1e-5 (wrong forward), on top of the layernorm adjoint bugs")], prelude: "" },
        Cert { name: "rmsnorm", inputs: vec![inp("x", &[3, 4]), inp("w", &[4])],
            expr: "rmsnorm(x, w, 0.00001)", wrt: &["x", "w"], out_shape: &[3, 4],
            oracle: oracle!(|e| layernorm_rows(get(e, "x"), get(e, "w"), None, 0.00001, true)), known: &[], prelude: "" },
        Cert { name: "rmsnorm_eps", inputs: vec![inp("x", &[3, 4]), inp("w", &[4])],
            expr: "rmsnorm(x, w, 0.5)", wrt: &["x", "w"], out_shape: &[3, 4],
            oracle: oracle!(|e| layernorm_rows(get(e, "x"), get(e, "w"), None, 0.5, true)), known: &[(Mode::Source, "the extractor hard-codes eps 1e-5: forward and gradients use the wrong eps")], prelude: "" },
        // --- indexing / joining ------------------------------------------
        Cert { name: "embedding", inputs: vec![inp("w", &[4, 3]), idx("i", &[6], "abs(arange(-2.0, 4.0))")],
            expr: "embedding_lookup(w, i)", wrt: &["w"], out_shape: &[6, 3],
            oracle: oracle!(|e| {
                let (w, i) = (get(e, "w"), get(e, "i"));
                let data = i.data.iter().flat_map(|&r| w.data[r as usize * 3..r as usize * 3 + 3].to_vec()).collect();
                Arr { shape: vec![6, 3], data }
            }), known: &[], prelude: "" },
        Cert { name: "gather", inputs: vec![inp("t", &[3, 4]), idx("g", &[3], "abs(arange(-1.0, 2.0)) + 1.0")],
            expr: "gather(t, 1, g).reshape([3])", wrt: &["t"], out_shape: &[3],
            oracle: oracle!(|e| {
                let (t, g) = (get(e, "t"), get(e, "g"));
                Arr { shape: vec![3], data: (0..3).map(|r| t.data[r * 4 + g.data[r] as usize]).collect() }
            }), known: &[(Mode::Source, "the dim literal is read as the index tensor: invalid tensor handle 0x1")], prelude: "" },
        Cert { name: "cat_dim0", inputs: vec![inp("a", &[2, 3]), inp("b", &[1, 3])], expr: "tensor_cat([a, b], 0)", wrt: &["a", "b"], out_shape: &[3, 3],
            oracle: oracle!(|e| concat(get(e, "a"), get(e, "b"), 0)), known: &[(Mode::Source, "the tensor list operand gets no adjoint: both gradients are zero")], prelude: "" },
        Cert { name: "cat_dim1", inputs: vec![inp("a", &[2, 3]), inp("b", &[2, 2])], expr: "tensor_cat([a, b], 1)", wrt: &["a", "b"], out_shape: &[2, 5],
            oracle: oracle!(|e| concat(get(e, "a"), get(e, "b"), 1)), known: &[(Mode::Source, "the tensor list operand gets no adjoint: both gradients are zero")], prelude: "" },
        // --- losses (scalar outputs) -------------------------------------
        Cert { name: "cross_entropy", inputs: vec![inp("z", &[4, 5]), idx("t", &[4], "abs(arange(-1.0, 3.0)) + 1.0")],
            expr: "cross_entropy(z, t)", wrt: &["z"], out_shape: &[],
            oracle: oracle!(|e| {
                let (z, t) = (get(e, "z"), get(e, "t"));
                let l: f64 = (0..4).map(|r| {
                    let row = &z.data[r * 5..r * 5 + 5];
                    let m = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let lse = m + row.iter().map(|v| (v - m).exp()).sum::<f64>().ln();
                    lse - row[t.data[r] as usize]
                }).sum();
                scalar(l / 4.0)
            }), known: &[], prelude: "from nsl.nn.losses import cross_entropy, mse_loss, l1_loss" },
        Cert { name: "mse_loss", inputs: XY_SAME.to_vec(), expr: "mse_loss(x, y)", wrt: &["x", "y"], out_shape: &[],
            oracle: oracle!(|e| {
                let (x, y) = (get(e, "x"), get(e, "y"));
                scalar(x.data.iter().zip(&y.data).map(|(a, b)| (a - b) * (a - b)).sum::<f64>() / 12.0)
            }), known: &[], prelude: "from nsl.nn.losses import cross_entropy, mse_loss, l1_loss" },
        Cert { name: "l1_loss", inputs: XY_SAME.to_vec(), expr: "l1_loss(x, y)", wrt: &["x", "y"], out_shape: &[],
            oracle: oracle!(|e| {
                let (x, y) = (get(e, "x"), get(e, "y"));
                scalar(x.data.iter().zip(&y.data).map(|(a, b)| (a - b).abs()).sum::<f64>() / 12.0)
            }), known: &[], prelude: "from nsl.nn.losses import cross_entropy, mse_loss, l1_loss" },
        // --- conv / attention --------------------------------------------
        Cert { name: "conv2d", inputs: vec![inp("x", &[1, 2, 5, 5]), inp("w", &[3, 2, 3, 3]), inp("b", &[3])],
            expr: "conv2d(x, w, b, 1, 1, 1, 1)", wrt: &["x", "w", "b"], out_shape: &[1, 3, 5, 5],
            oracle: oracle!(|e| conv2d_ref(get(e, "x"), get(e, "w"), get(e, "b"), 1, 1)), known: &[], prelude: "" },
        Cert { name: "sdpa", inputs: vec![inp("q", &[1, 2, 4, 8]), inp("k", &[1, 2, 4, 8]), inp("v", &[1, 2, 4, 8])],
            expr: "scaled_dot_product_attention(q, k, v, 0.35355339059327373, false)", wrt: &["q", "k", "v"], out_shape: &[1, 2, 4, 8],
            oracle: oracle!(|e| sdpa_ref(get(e, "q"), get(e, "k"), get(e, "v"), 0.35355339059327373, false)), known: &[(Mode::Source, "compile panic in the grad block: FunctionBuilder finalized, but block3 is not filled")], prelude: "" },
        Cert { name: "sdpa_causal", inputs: vec![inp("q", &[1, 2, 4, 8]), inp("k", &[1, 2, 4, 8]), inp("v", &[1, 2, 4, 8])],
            expr: "scaled_dot_product_attention(q, k, v, 0.35355339059327373, true)", wrt: &["q", "k", "v"], out_shape: &[1, 2, 4, 8],
            oracle: oracle!(|e| sdpa_ref(get(e, "q"), get(e, "k"), get(e, "v"), 0.35355339059327373, true)), known: &[(Mode::Source, "compile panic in the grad block: FunctionBuilder finalized, but block3 is not filled")], prelude: "" },
        Cert { name: "sdpa_scale", inputs: vec![inp("q", &[1, 2, 4, 8]), inp("k", &[1, 2, 4, 8]), inp("v", &[1, 2, 4, 8])],
            expr: "scaled_dot_product_attention(q, k, v, 0.9, false)", wrt: &["q", "k", "v"], out_shape: &[1, 2, 4, 8],
            oracle: oracle!(|e| sdpa_ref(get(e, "q"), get(e, "k"), get(e, "v"), 0.9, false)), known: &[(Mode::Source, "compile panic in the grad block: FunctionBuilder finalized, but block3 is not filled")], prelude: "" },
    ]
}

// ---------------------------------------------------------------------------
// Running a certificate
// ---------------------------------------------------------------------------

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap()
        .to_path_buf()
}

fn shape_list(shape: &[usize]) -> String {
    let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    format!("[{}]", dims.join(", "))
}

fn program(c: &Cert) -> String {
    let mut s = String::new();
    if !c.prelude.is_empty() {
        s += c.prelude;
        s += "\n";
    }
    for i in &c.inputs {
        let init = match i.init {
            Init::Randn => format!("randn({})", shape_list(i.shape)),
            Init::Positive => format!("abs(randn({})) + 0.5", shape_list(i.shape)),
            Init::Expr(e) => e.to_string(),
        };
        s += &format!("let {} = {init}\n", i.name);
    }
    let r_shape: &[usize] = if c.out_shape.is_empty() {
        &[1]
    } else {
        c.out_shape
    };
    s += &format!("let cert_r = randn({})\n", shape_list(r_shape));
    for name in c.inputs.iter().map(|i| i.name).chain(["cert_r"]) {
        s += &format!("print(\"IN_{name}_BEGIN\")\nprint({name})\nprint(\"IN_{name}_END\")\n");
    }
    for w in c.wrt {
        s += &format!(
            "let (cert_l_{w}, cert_d_{w}) = grad({w}):\n    sum(({}) * cert_r)\n\
             print(\"LOSS_{w}_BEGIN\")\nprint(cert_l_{w})\nprint(\"LOSS_{w}_END\")\n\
             print(\"GRAD_{w}_BEGIN\")\nprint(cert_d_{w})\nprint(\"GRAD_{w}_END\")\n",
            c.expr
        );
    }
    s
}

fn parse_between(stdout: &str, begin: &str, end: &str) -> Option<Vec<f64>> {
    let after = stdout.split_once(begin)?.1;
    let inner = after.split_once(end)?.0;
    Some(
        inner
            .split(|c: char| {
                !(c.is_ascii_digit() || c == '.' || c == '-' || c == 'e' || c == 'E' || c == '+')
            })
            .filter(|t| !t.is_empty() && t.chars().any(|c| c.is_ascii_digit()))
            .filter_map(|t| t.parse::<f64>().ok())
            .collect(),
    )
}

/// The oracle's loss: `sum(out * r)`, `r` broadcast for a scalar output.
fn oracle_loss(c: &Cert, env: &Env) -> f64 {
    let out = (c.oracle)(env);
    assert_eq!(
        numel(&out.shape).max(1),
        numel(c.out_shape).max(1),
        "{}: oracle output shape {:?} disagrees with the declared {:?}",
        c.name,
        out.shape,
        c.out_shape
    );
    let r = &env["cert_r"].data;
    out.data
        .iter()
        .enumerate()
        .map(|(i, v)| v * r[i % r.len()])
        .sum()
}

const FALLBACK_MARKERS: &[&str] = &[
    "falling back to tape-based AD",
    "unrecognized FFI callee",
    "unresolved method call",
    "extraction failed",
];

/// Run one certificate in one mode; `Err` describes how it failed.
fn run_cert(c: &Cert, mode: Mode) -> Result<(), String> {
    let root = repo_root();
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join(format!("{}.nsl", c.name));
    std::fs::write(&path, program(c)).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("run");
    if mode == Mode::Source {
        cmd.arg("--source-ad");
    }
    let out = cmd
        .arg(&path)
        .current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let tail = || {
        stderr
            .lines()
            .filter(|l| !l.trim().is_empty())
            .rev()
            .take(6)
            .collect::<Vec<_>>()
            .join(" | ")
    };
    if !out.status.success() {
        return Err(format!("the program failed: {}", tail()));
    }
    if mode == Mode::Source {
        let engaged = stderr
            .matches("Using source-to-source AD for grad block")
            .count();
        if engaged != c.wrt.len() {
            return Err(format!(
                "source AD engaged {engaged} of {} grad blocks",
                c.wrt.len()
            ));
        }
        if let Some(m) = FALLBACK_MARKERS.iter().find(|m| stderr.contains(*m)) {
            return Err(format!("source AD fell back ({m})"));
        }
    }

    let mut env: Env = HashMap::new();
    for (name, shape) in c
        .inputs
        .iter()
        .map(|i| (i.name, i.shape))
        .chain([("cert_r", c.out_shape)])
    {
        let data = parse_between(
            &stdout,
            &format!("IN_{name}_BEGIN"),
            &format!("IN_{name}_END"),
        )
        .ok_or_else(|| format!("input {name} not printed"))?;
        let want = numel(shape).max(1);
        if data.len() != want {
            return Err(format!(
                "input {name}: {} values printed, {want} expected",
                data.len()
            ));
        }
        env.insert(
            name,
            Arr {
                shape: shape.to_vec(),
                data,
            },
        );
    }

    let base_loss = oracle_loss(c, &env);
    let mut failures = Vec::new();
    for &w in c.wrt {
        let loss = parse_between(
            &stdout,
            &format!("LOSS_{w}_BEGIN"),
            &format!("LOSS_{w}_END"),
        )
        .and_then(|v| v.first().copied())
        .ok_or_else(|| format!("loss for {w} not printed"))?;
        if (loss - base_loss).abs() > 1e-4 * base_loss.abs().max(1.0) {
            failures.push(format!("forward: loss {loss} vs oracle {base_loss}"));
        }
        let got = parse_between(
            &stdout,
            &format!("GRAD_{w}_BEGIN"),
            &format!("GRAD_{w}_END"),
        )
        .ok_or_else(|| format!("gradient for {w} not printed"))?;
        let n = env[w].data.len();
        if got.len() != n {
            failures.push(format!(
                "d{w}: {} values, {n} expected (wrong shape)",
                got.len()
            ));
            continue;
        }
        let want: Vec<f64> = (0..n)
            .map(|k| {
                let h = 1e-6;
                let mut up = env.clone();
                up.get_mut(w).unwrap().data[k] += h;
                let mut down = env.clone();
                down.get_mut(w).unwrap().data[k] -= h;
                (oracle_loss(c, &up) - oracle_loss(c, &down)) / (2.0 * h)
            })
            .collect();
        let scale = want.iter().fold(1.0f64, |m, v| m.max(v.abs()));
        let (worst_k, worst) = got
            .iter()
            .zip(&want)
            .map(|(g, t)| (g - t).abs())
            .enumerate()
            .fold(
                (0, 0.0f64),
                |acc, (k, d)| if d > acc.1 { (k, d) } else { acc },
            );
        if worst > 2e-4 * scale {
            failures.push(format!(
                "d{w}[{worst_k}] = {} vs oracle {} (max |err| {:.2e}, scale {:.2})",
                got[worst_k], want[worst_k], worst, scale
            ));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("; "))
    }
}

fn check(name: &str) {
    let all = certs();
    let c = all
        .iter()
        .find(|c| c.name == name)
        .unwrap_or_else(|| panic!("no certificate {name}"));
    let mut problems = Vec::new();
    for mode in [Mode::Tape, Mode::Source] {
        let result = run_cert(c, mode);
        let known = c.known.iter().find(|(m, _)| *m == mode);
        match (result, known) {
            (Ok(()), None) => eprintln!("{name} [{mode:?}]: certified"),
            (Err(e), Some((_, why))) => eprintln!("{name} [{mode:?}]: known failure ({why}): {e}"),
            (Err(e), None) => problems.push(format!("{mode:?}: {e}")),
            (Ok(()), Some((_, why))) => problems.push(format!(
                "{mode:?} now PASSES but is listed as a known failure ({why}): remove it from `known`"
            )),
        }
    }
    assert!(problems.is_empty(), "{name}:\n  {}", problems.join("\n  "));
}

#[test]
fn certificate_names_are_unique() {
    let all = certs();
    let mut names: Vec<&str> = all.iter().map(|c| c.name).collect();
    names.sort_unstable();
    let before = names.len();
    names.dedup();
    assert_eq!(before, names.len(), "duplicate certificate names");
}

/// The coverage gate: every certificate a `PrimalOp` status names exists here,
/// and every certificate here is named by some status.
#[test]
fn certificates_match_the_codegen_inventory() {
    let all = certs();
    let names: Vec<&str> = all.iter().map(|c| c.name).collect();
    let mut claimed: Vec<&str> = Vec::new();
    for (op, status) in ad_cert_inventory() {
        if let AdCertStatus::Certified(named) = status {
            for n in named {
                assert!(
                    names.contains(n),
                    "{op} names certificate `{n}`, which is not in certs()"
                );
                claimed.push(n);
            }
        }
    }
    for n in &names {
        assert!(
            claimed.contains(n),
            "certificate `{n}` is not named by any PrimalOp status in ad_cert_status"
        );
    }
}

macro_rules! cert_tests {
    ($($name:ident),* $(,)?) => {
        $(
            #[test]
            fn $name() {
                check(stringify!($name));
            }
        )*
        /// Every certificate in the table has a test, and every test a certificate.
        #[test]
        fn every_certificate_has_a_test() {
            let tests: Vec<&str> = vec![$(stringify!($name)),*];
            let all = certs();
            for c in &all {
                assert!(tests.contains(&c.name), "certificate {} has no test in cert_tests!", c.name);
            }
            assert_eq!(tests.len(), all.len(), "cert_tests! names a certificate the table lacks");
        }
    };
}

cert_tests! {
    add_same, add_row, add_col, add_one, add_literal, sub_row, sub_col,
    mul_row, mul_col, mul_literal, div_row, div_col, div_numerator_broadcast,
    matmul_2d, matmul_3d_2d, matmul_2d_3d, matmul_3d_3d,
    neg, relu, sigmoid, tanh, exp, log, sqrt, gelu, silu, abs, clamp, cos, sin,
    transpose, transpose_3d, reshape, unsqueeze, expand, contiguous,
    rotate_half,
    sum_all, mean_all, sum_dim, mean_dim, sum_dim_neg, sum_dim_keepdim, sum_dim_last, mean_dim_last,
    softmax_last, softmax_dim0, softmax_mid, log_softmax_last, log_softmax_dim0, log_softmax_mid,
    layernorm, layernorm_eps, rmsnorm, rmsnorm_eps,
    embedding, gather, cat_dim0, cat_dim1,
    cross_entropy, mse_loss, l1_loss,
    conv2d, sdpa, sdpa_causal, sdpa_scale,
}
