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
//!
//! The tape has its own inventory: `nsl_runtime::autodiff::tape_cert_status`
//! gives every `TapeOp` (the tape's backward arms) a status the same way.
//! `certificates_match_the_tape_inventory` holds those names to this table,
//! and every tape run here is traced (`NSL_DEBUG_MEM_TRACE=1` makes the tape
//! log `[tape-trace] record <Variant>` per recorded op), so a certificate a
//! `TapeOp` status names must actually record that op. Some ops exist only on
//! the tape (no source-AD extraction): their certificates list the source run
//! as a known failure that must be a fall back to the tape, so the day source
//! AD extracts the op the ratchet flips.
//!
//! Layout is an axis: each base certificate hands its op contiguous tensors,
//! and its `_vgrad` / `_vin` variants (`LAYOUT_VARIANTS`) hand it a strided
//! output gradient / strided inputs. A `TapeOp` whose variants fail is
//! `Defective`: it names the failing certificates as defects, ratcheted here.
//!
//! A certificate that fails in every mode certifies nothing; it is a DEFECT
//! PROBE, and only a `Defective` status may name it (as a defect). The
//! defect's fix flips the ratchet, and the checks then move it to certified.

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;

use nsl_codegen::ad_rules::{AdCertStatus, ad_cert_inventory};
use nsl_runtime::autodiff::{TapeCertStatus, tape_cert_inventory};

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
    /// An NSL expression yielding the tensor (index tensors, the seeded
    /// dropout mask).
    Expr(&'static str),
}

#[derive(Clone, Copy)]
struct Input {
    name: &'static str,
    shape: &'static [usize],
    init: Init,
    /// Drawn transposed in its last two dims and transposed back: a strided
    /// (non-contiguous) view of `shape` with the same logical values. Set
    /// only by a `Layout::ViewIn` variant.
    view: bool,
}

/// Row-major data and its shape.
#[derive(Clone, Debug)]
struct Arr {
    shape: Vec<usize>,
    data: Vec<f64>,
}

type Env = HashMap<&'static str, Arr>;

/// A certificate's known failures: (mode, why).
type Known = &'static [(Mode, &'static str)];

/// The f64 forward a certificate is held to.
#[derive(Clone, Copy)]
enum Oracle {
    Fn(fn(&Env) -> Arr),
    /// `Fn`'s output transposed in two dims (a `Layout::ViewGrad` variant).
    Transposed(fn(&Env) -> Arr, usize, usize),
}

impl Oracle {
    fn eval(self, env: &Env) -> Arr {
        match self {
            Oracle::Fn(f) => f(env),
            Oracle::Transposed(f, d0, d1) => transpose2(&f(env), d0, d1),
        }
    }
}

struct Cert {
    name: &'static str,
    inputs: Vec<Input>,
    expr: &'static str,
    wrt: &'static [&'static str],
    out_shape: &'static [usize],
    oracle: Oracle,
    /// Known failures: (mode, why). Empty = must pass in both modes.
    known: Known,
    /// Lines before the inputs (imports).
    prelude: &'static str,
}

const fn inp(name: &'static str, shape: &'static [usize]) -> Input {
    Input {
        name,
        shape,
        init: Init::Randn,
        view: false,
    }
}
const fn pos(name: &'static str, shape: &'static [usize]) -> Input {
    Input {
        name,
        shape,
        init: Init::Positive,
        view: false,
    }
}
const fn idx(name: &'static str, shape: &'static [usize], expr: &'static str) -> Input {
    Input {
        name,
        shape,
        init: Init::Expr(expr),
        view: false,
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

/// Packed attention over [B, H, S, D]: causal within each document, where
/// `seg` [B, S] names the document of every position (PCA Stage C). Row i
/// attends to j <= i with seg[b][j] == seg[b][i].
fn sdpa_packed_ref(q: &Arr, k: &Arr, v: &Arr, scale: f64, seg: &Arr) -> Arr {
    let (bsz, h, s, d) = (q.shape[0], q.shape[1], q.shape[2], q.shape[3]);
    let mut out = vec![0.0; q.data.len()];
    for b in 0..bsz {
        let doc = |i: usize| seg.data[b * s + i];
        for hh in 0..h {
            let base = (b * h + hh) * s * d;
            for i in 0..s {
                let scores: Vec<f64> = (0..s)
                    .map(|j| {
                        if j > i || doc(j) != doc(i) {
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
    }
    Arr {
        shape: q.shape.clone(),
        data: out,
    }
}

fn max_of(v: &[f64]) -> f64 {
    v.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
}

/// Elements `[start, end)` of `a` along `dim`.
fn slice_ref(a: &Arr, dim: usize, start: usize, end: usize) -> Arr {
    let mut shape = a.shape.clone();
    shape[dim] = end - start;
    let st = strides(&a.shape);
    let data = (0..numel(&shape))
        .map(|k| {
            let mut ix = unravel(k, &shape);
            ix[dim] += start;
            a.data[ix.iter().zip(&st).map(|(i, s)| i * s).sum::<usize>()]
        })
        .collect();
    Arr { shape, data }
}

/// Equal-shape `parts` stacked along a new output axis `dim`.
fn stack_ref(parts: &[&Arr], dim: usize) -> Arr {
    let inner = &parts[0].shape;
    let mut shape = inner.clone();
    shape.insert(dim, parts.len());
    let st = strides(inner);
    let data = (0..numel(&shape))
        .map(|k| {
            let mut ix = unravel(k, &shape);
            let p = ix.remove(dim);
            parts[p].data[ix.iter().zip(&st).map(|(i, s)| i * s).sum::<usize>()]
        })
        .collect();
    Arr { shape, data }
}

/// NCHW max pooling over a `k` x `k` window. Padding cells are skipped, not
/// read as zeros (the runtime never lets a pad cell win).
fn maxpool2d_ref(x: &Arr, k: usize, stride: usize, pad: usize) -> Arr {
    let (n, c, h, w) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3]);
    let oh = (h + 2 * pad - k) / stride + 1;
    let ow = (w + 2 * pad - k) / stride + 1;
    let mut data = Vec::with_capacity(n * c * oh * ow);
    for nc in 0..n * c {
        for oy in 0..oh {
            for ox in 0..ow {
                let mut m = f64::NEG_INFINITY;
                for ky in 0..k {
                    for kx in 0..k {
                        let iy = (oy * stride + ky) as isize - pad as isize;
                        let ix = (ox * stride + kx) as isize - pad as isize;
                        if iy >= 0 && ix >= 0 && (iy as usize) < h && (ix as usize) < w {
                            m = m.max(x.data[(nc * h + iy as usize) * w + ix as usize]);
                        }
                    }
                }
                data.push(m);
            }
        }
    }
    Arr {
        shape: vec![n, c, oh, ow],
        data,
    }
}

/// The seeded dropout's forward, `x * m`. The mask is the one quantity the
/// oracle cannot draw, so the program supplies it: `m` is a same-seed
/// `dropout(ones)`, i.e. mask / (1 - p). It is checked instead of trusted --
/// every element exactly 0 or 1/(1-p), both present -- so the scale is held
/// to the oracle's p, and a trivial mask (all kept: only a scale; all
/// dropped: nothing) cannot pass for a certificate.
fn seeded_dropout_ref(x: &Arr, m: &Arr, p: f64) -> Arr {
    let keep = 1.0 / (1.0 - p);
    assert!(
        m.data.iter().all(|&v| v == 0.0 || v == keep),
        "dropout mask {:?} is not 0 / {keep}: the forward's scale is wrong",
        m.data
    );
    assert!(
        m.data.contains(&0.0) && m.data.contains(&keep),
        "dropout mask {:?} is trivial: pick another seed",
        m.data
    );
    binary(x, m, |a, b| a * b)
}

// ---------------------------------------------------------------------------
// The certificates
// ---------------------------------------------------------------------------

macro_rules! oracle {
    (|$env:ident| $body:expr) => {{
        fn f($env: &Env) -> Arr {
            $body
        }
        Oracle::Fn(f)
    }};
}

const XY_ROW: &[Input] = &[inp("x", &[3, 4]), inp("y", &[4])];
const XY_COL: &[Input] = &[inp("x", &[3, 4]), inp("y", &[3, 1])];
const XY_SAME: &[Input] = &[inp("x", &[3, 4]), inp("y", &[3, 4])];
const XY_ONE: &[Input] = &[inp("x", &[3, 4]), inp("y", &[1])];
const X_34: &[Input] = &[inp("x", &[3, 4])];
const X_POS: &[Input] = &[pos("x", &[3, 4])];
const X_234: &[Input] = &[inp("x", &[2, 3, 4])];
const AB_34: &[Input] = &[inp("a", &[3, 4]), inp("b", &[3, 4])];
const X_POOL: &[Input] = &[inp("x", &[1, 2, 4, 4])];

/// The source-run entry of a tape-only certificate: source AD has no
/// extraction for the op, so the grad block falls back to the tape. `check`
/// holds such an entry to a FALL BACK (`FALLS_BACK`), not to any failure.
macro_rules! tape_only {
    ($what:literal) => {
        &[(
            Mode::Source,
            concat!($what, ": the grad block falls back to the tape"),
        )]
    };
}

/// A fixed mask: `manual_seed` before each draw makes the program's mask
/// input and the grad block's dropout the same draw.
const DROPOUT_PRELUDE: &str = "fn cert_dropout_mask() -> Tensor:\n    manual_seed(7)\n    \
     return dropout(ones([3, 4]), 0.5, true)\nfn cert_seeded_dropout(t: Tensor) -> Tensor:\n    \
     manual_seed(7)\n    return dropout(t, 0.5, true)";

/// One row per certificate; kept one-per-line so the table reads as a table.
#[rustfmt::skip]
fn base_certs() -> Vec<Cert> {
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
        // keepdim over a middle dim with no reshape after it (the tape's
        // SumReduce/MeanReduce arms see the output's gradient directly).
        Cert { name: "sum_dim_keepdim_mid", inputs: X_234.to_vec(), expr: "sum(x, 1, 1)", wrt: &["x"], out_shape: &[2, 1, 4],
            oracle: oracle!(|e| Arr { shape: vec![2, 1, 4], data: reduce_dim(get(e, "x"), 1, |v| v.iter().sum()).data }),
            known: &[(Mode::Source, "a keepdim sum is not extracted: the grad block falls back to the tape")], prelude: "" },
        Cert { name: "mean_dim_keepdim_mid", inputs: X_234.to_vec(), expr: "mean(x, 1, 1)", wrt: &["x"], out_shape: &[2, 1, 4],
            oracle: oracle!(|e| Arr { shape: vec![2, 1, 4], data: reduce_dim(get(e, "x"), 1, |v| v.iter().sum::<f64>() / v.len() as f64).data }),
            known: &[(Mode::Source, "a keepdim mean is not extracted: the grad block falls back to the tape")], prelude: "" },
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
            oracle: oracle!(|e| layernorm_rows(get(e, "x"), get(e, "w"), Some(get(e, "b")), 0.00001, false)), known: &[], prelude: "" },
        Cert { name: "layernorm_eps", inputs: vec![inp("x", &[3, 4]), inp("w", &[4]), inp("b", &[4])],
            expr: "layernorm(x, w, b, 0.5)", wrt: &["x", "w", "b"], out_shape: &[3, 4],
            oracle: oracle!(|e| layernorm_rows(get(e, "x"), get(e, "w"), Some(get(e, "b")), 0.5, false)), known: &[], prelude: "" },
        Cert { name: "rmsnorm", inputs: vec![inp("x", &[3, 4]), inp("w", &[4])],
            expr: "rmsnorm(x, w, 0.00001)", wrt: &["x", "w"], out_shape: &[3, 4],
            oracle: oracle!(|e| layernorm_rows(get(e, "x"), get(e, "w"), None, 0.00001, true)), known: &[], prelude: "" },
        Cert { name: "rmsnorm_eps", inputs: vec![inp("x", &[3, 4]), inp("w", &[4])],
            expr: "rmsnorm(x, w, 0.5)", wrt: &["x", "w"], out_shape: &[3, 4],
            oracle: oracle!(|e| layernorm_rows(get(e, "x"), get(e, "w"), None, 0.5, true)), known: &[], prelude: "" },
        Cert { name: "layernorm_3d", inputs: vec![inp("x", &[2, 3, 4]), inp("w", &[4]), inp("b", &[4])],
            expr: "layernorm(x, w, b, 0.00001)", wrt: &["x", "w", "b"], out_shape: &[2, 3, 4],
            oracle: oracle!(|e| layernorm_rows(get(e, "x"), get(e, "w"), Some(get(e, "b")), 0.00001, false)), known: &[], prelude: "" },
        // A model-field eps (the stdlib norms' `self.eps`) is read at run time
        // by source AD; 0.5 is far from the 1e-5 it used to bake.
        Cert { name: "layernorm_field_eps", inputs: vec![inp("x", &[3, 4]), inp("w", &[4]), inp("b", &[4])],
            expr: "layernorm(x, w, b, cfg.eps)", wrt: &["x", "w", "b"], out_shape: &[3, 4],
            oracle: oracle!(|e| layernorm_rows(get(e, "x"), get(e, "w"), Some(get(e, "b")), 0.5, false)),
            known: &[],
            prelude: "model EpsCfg(d: int):\n    eps: float = 0.5\nlet cfg = EpsCfg(1)" },
        Cert { name: "rmsnorm_field_eps", inputs: vec![inp("x", &[3, 4]), inp("w", &[4])],
            expr: "rmsnorm(x, w, cfg.eps)", wrt: &["x", "w"], out_shape: &[3, 4],
            oracle: oracle!(|e| layernorm_rows(get(e, "x"), get(e, "w"), None, 0.5, true)),
            known: &[],
            prelude: "model EpsCfg(d: int):\n    eps: float = 0.5\nlet cfg = EpsCfg(1)" },
        // The eps field read twice: the scalar operand must stay a scalar for
        // its second consumer too (it used to be re-typed a tensor after the
        // first).
        Cert { name: "rmsnorm_field_eps_reused", inputs: vec![inp("x", &[3, 4]), inp("w", &[4])],
            expr: "rmsnorm(x, w, cfg.eps) * cfg.eps", wrt: &["x", "w"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(&layernorm_rows(get(e, "x"), get(e, "w"), None, 0.5, true), &scalar(0.5), |a, b| a * b)),
            known: &[],
            prelude: "model EpsCfg(d: int):\n    eps: float = 0.5\nlet cfg = EpsCfg(1)" },
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
            }), known: &[], prelude: "" },
        Cert { name: "gather_neg", inputs: vec![inp("t", &[3, 4]), idx("g", &[3], "abs(arange(-1.0, 2.0)) + 1.0")],
            expr: "gather(t, -1, g).reshape([3])", wrt: &["t"], out_shape: &[3],
            oracle: oracle!(|e| {
                let (t, g) = (get(e, "t"), get(e, "g"));
                Arr { shape: vec![3], data: (0..3).map(|r| t.data[r * 4 + g.data[r] as usize]).collect() }
            }), known: &[], prelude: "" },
        Cert { name: "gather_dim0", inputs: vec![inp("t", &[3, 4]), idx("g", &[1], "abs(arange(-1.0, 0.0)) + 1.0")],
            expr: "gather(t, 0, g).reshape([4])", wrt: &["t"], out_shape: &[4],
            oracle: oracle!(|e| {
                let (t, g) = (get(e, "t"), get(e, "g"));
                let r = g.data[0] as usize;
                Arr { shape: vec![4], data: t.data[r * 4..r * 4 + 4].to_vec() }
            }), known: &[], prelude: "" },
        Cert { name: "gather_mid", inputs: vec![inp("t", &[2, 3, 4]), idx("g", &[2], "abs(arange(-1.0, 1.0))")],
            expr: "gather(t, 1, g).reshape([2, 4])", wrt: &["t"], out_shape: &[2, 4],
            oracle: oracle!(|e| {
                let (t, g) = (get(e, "t"), get(e, "g"));
                let data = (0..2).flat_map(|o| {
                    let r = g.data[o] as usize;
                    t.data[o * 12 + r * 4..o * 12 + r * 4 + 4].to_vec()
                }).collect();
                Arr { shape: vec![2, 4], data }
            }), known: &[], prelude: "" },
        Cert { name: "cat_dim0", inputs: vec![inp("a", &[2, 3]), inp("b", &[1, 3])], expr: "tensor_cat([a, b], 0)", wrt: &["a", "b"], out_shape: &[3, 3],
            oracle: oracle!(|e| concat(get(e, "a"), get(e, "b"), 0)), known: &[], prelude: "" },
        Cert { name: "cat_dim1", inputs: vec![inp("a", &[2, 3]), inp("b", &[2, 2])], expr: "tensor_cat([a, b], 1)", wrt: &["a", "b"], out_shape: &[2, 5],
            oracle: oracle!(|e| concat(get(e, "a"), get(e, "b"), 1)), known: &[], prelude: "" },
        Cert { name: "cat_three", inputs: vec![inp("a", &[2, 3]), inp("b", &[2, 1]), inp("c", &[2, 2])],
            expr: "tensor_cat([a, b, c], 1)", wrt: &["a", "b", "c"], out_shape: &[2, 6],
            oracle: oracle!(|e| concat(&concat(get(e, "a"), get(e, "b"), 1), get(e, "c"), 1)), known: &[], prelude: "" },
        Cert { name: "cat_neg", inputs: vec![inp("a", &[2, 3]), inp("b", &[2, 2])], expr: "tensor_cat([a, b], -1)", wrt: &["a", "b"], out_shape: &[2, 5],
            oracle: oracle!(|e| concat(get(e, "a"), get(e, "b"), 1)), known: &[], prelude: "" },
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
            oracle: oracle!(|e| sdpa_ref(get(e, "q"), get(e, "k"), get(e, "v"), 0.35355339059327373, false)), known: &[], prelude: "" },
        Cert { name: "sdpa_causal", inputs: vec![inp("q", &[1, 2, 4, 8]), inp("k", &[1, 2, 4, 8]), inp("v", &[1, 2, 4, 8])],
            expr: "scaled_dot_product_attention(q, k, v, 0.35355339059327373, true)", wrt: &["q", "k", "v"], out_shape: &[1, 2, 4, 8],
            oracle: oracle!(|e| sdpa_ref(get(e, "q"), get(e, "k"), get(e, "v"), 0.35355339059327373, true)), known: &[], prelude: "" },
        Cert { name: "sdpa_scale", inputs: vec![inp("q", &[1, 2, 4, 8]), inp("k", &[1, 2, 4, 8]), inp("v", &[1, 2, 4, 8])],
            expr: "scaled_dot_product_attention(q, k, v, 0.9, false)", wrt: &["q", "k", "v"], out_shape: &[1, 2, 4, 8],
            oracle: oracle!(|e| sdpa_ref(get(e, "q"), get(e, "k"), get(e, "v"), 0.9, false)), known: &[], prelude: "" },
        // Packed (segment-masked) attention: causal within each document.
        // On the CPU the forward is the decomposed chain under the mask
        // derived from `seg` and the backward is the segment-aware flash
        // reference, so these certify the source-AD wiring and those CPU
        // paths, not the GPU kernels (held to an f64 oracle by
        // nsl-codegen/tests/sdpa_fused_packed_gpu_parity.rs).
        Cert { name: "sdpa_packed",
            inputs: vec![inp("q", &[1, 2, 6, 8]), inp("k", &[1, 2, 6, 8]), inp("v", &[1, 2, 6, 8]),
                idx("seg", &[1, 6], "tensor_cat([zeros([1, 3]), ones([1, 3])], 1)")],
            expr: "scaled_dot_product_attention_packed(q, k, v, 0.35355339059327373, seg)", wrt: &["q", "k", "v"], out_shape: &[1, 2, 6, 8],
            oracle: oracle!(|e| sdpa_packed_ref(get(e, "q"), get(e, "k"), get(e, "v"), 0.35355339059327373, get(e, "seg"))), known: &[], prelude: "" },
        // Three uneven documents, one of length 1 (a row that sees only itself).
        Cert { name: "sdpa_packed_docs",
            inputs: vec![inp("q", &[1, 2, 6, 8]), inp("k", &[1, 2, 6, 8]), inp("v", &[1, 2, 6, 8]),
                idx("seg", &[1, 6], "tensor_cat([zeros([1, 2]), ones([1, 3]), full([1, 1], 2.0)], 1)")],
            expr: "scaled_dot_product_attention_packed(q, k, v, 0.35355339059327373, seg)", wrt: &["q", "k", "v"], out_shape: &[1, 2, 6, 8],
            oracle: oracle!(|e| sdpa_packed_ref(get(e, "q"), get(e, "k"), get(e, "v"), 0.35355339059327373, get(e, "seg"))), known: &[], prelude: "" },
        // Two batch rows packed differently: the mask is per row.
        Cert { name: "sdpa_packed_batch",
            inputs: vec![inp("q", &[2, 2, 6, 8]), inp("k", &[2, 2, 6, 8]), inp("v", &[2, 2, 6, 8]),
                idx("seg", &[2, 6], "tensor_cat([tensor_cat([zeros([1, 3]), ones([1, 3])], 1), tensor_cat([zeros([1, 5]), ones([1, 1])], 1)], 0)")],
            expr: "scaled_dot_product_attention_packed(q, k, v, 0.35355339059327373, seg)", wrt: &["q", "k", "v"], out_shape: &[2, 2, 6, 8],
            oracle: oracle!(|e| sdpa_packed_ref(get(e, "q"), get(e, "k"), get(e, "v"), 0.35355339059327373, get(e, "seg"))), known: &[], prelude: "" },
        // A scale other than 1/sqrt(head_dim). The builtin's doc says the
        // fused backward re-derives the scale from Q's shape; the CPU paths
        // take the argument.
        Cert { name: "sdpa_packed_scale",
            inputs: vec![inp("q", &[1, 2, 6, 8]), inp("k", &[1, 2, 6, 8]), inp("v", &[1, 2, 6, 8]),
                idx("seg", &[1, 6], "tensor_cat([zeros([1, 3]), ones([1, 3])], 1)")],
            expr: "scaled_dot_product_attention_packed(q, k, v, 0.9, seg)", wrt: &["q", "k", "v"], out_shape: &[1, 2, 6, 8],
            oracle: oracle!(|e| sdpa_packed_ref(get(e, "q"), get(e, "k"), get(e, "v"), 0.9, get(e, "seg"))), known: &[], prelude: "" },
        // --- tape-only ops (no source-AD extraction; named by tape_cert_status) ---
        // A lossless cast round trip: both Cast directions (f32 -> f64 and back)
        // and, in the square, f64 arithmetic between them with two Cast grads
        // accumulating into one f32 input. A lossy cast (fp16/bf16) is
        // piecewise constant, so no finite difference can hold its
        // straight-through gradient.
        Cert { name: "cast_f64_round_trip", inputs: X_34.to_vec(), expr: "x.to(f64).to(f32)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), |a| a)), known: tape_only!("source AD does not extract `.to(dtype)`"), prelude: "" },
        Cert { name: "cast_f64_square", inputs: X_34.to_vec(), expr: "(x.to(f64) * x.to(f64)).to(f32)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| unary(get(e, "x"), |a| a * a)), known: tape_only!("source AD does not extract `.to(dtype)`"), prelude: "" },
        // randn inputs: a tie for the max (where the gradient is undefined)
        // has probability zero, and the draw is seeded, so none occurs.
        Cert { name: "reduce_max_dim1", inputs: X_34.to_vec(), expr: "reduce_max(x, 1, 0).reshape([3])", wrt: &["x"], out_shape: &[3],
            oracle: oracle!(|e| reduce_dim(get(e, "x"), 1, max_of)), known: tape_only!("source AD does not extract `reduce_max`"), prelude: "" },
        Cert { name: "reduce_max_dim0", inputs: X_34.to_vec(), expr: "reduce_max(x, 0, 0).reshape([4])", wrt: &["x"], out_shape: &[4],
            oracle: oracle!(|e| reduce_dim(get(e, "x"), 0, max_of)), known: tape_only!("source AD does not extract `reduce_max`"), prelude: "" },
        // keepdim, with no reshape after it: the output feeds the loss (or a
        // layout variant's transpose) directly.
        Cert { name: "reduce_max_keepdim_last", inputs: X_234.to_vec(), expr: "reduce_max(x, -1, 1)", wrt: &["x"], out_shape: &[2, 3, 1],
            oracle: oracle!(|e| Arr { shape: vec![2, 3, 1], data: reduce_dim(get(e, "x"), 2, max_of).data }),
            known: tape_only!("source AD does not extract `reduce_max`"), prelude: "" },
        // keepdim over a non-last dim: every dim after the kept size-1 one
        // must read its own index (`scatter_grad_to_argmax` once read the
        // kept dim's 0 for all of them, routing the whole gradient to the
        // last dim's first slot).
        Cert { name: "reduce_max_keepdim_mid", inputs: X_234.to_vec(), expr: "reduce_max(x, -2, 1)", wrt: &["x"], out_shape: &[2, 1, 4],
            oracle: oracle!(|e| Arr { shape: vec![2, 1, 4], data: reduce_dim(get(e, "x"), 1, max_of).data }),
            known: tape_only!("source AD does not extract `reduce_max`"), prelude: "" },
        Cert { name: "reduce_max_mid", inputs: X_234.to_vec(), expr: "reduce_max(x, 1, 0).reshape([2, 4])", wrt: &["x"], out_shape: &[2, 4],
            oracle: oracle!(|e| reduce_dim(get(e, "x"), 1, max_of)), known: tape_only!("source AD does not extract `reduce_max`"), prelude: "" },
        Cert { name: "slice_dim1", inputs: X_34.to_vec(), expr: "x.slice(1, 1, 3)", wrt: &["x"], out_shape: &[3, 2],
            oracle: oracle!(|e| slice_ref(get(e, "x"), 1, 1, 3)), known: tape_only!("source AD does not extract `.slice`"), prelude: "" },
        Cert { name: "slice_dim0", inputs: X_34.to_vec(), expr: "tensor_slice(x, 0, 1, 3)", wrt: &["x"], out_shape: &[2, 4],
            oracle: oracle!(|e| slice_ref(get(e, "x"), 0, 1, 3)), known: tape_only!("source AD does not extract `tensor_slice`"), prelude: "" },
        Cert { name: "slice_neg", inputs: X_34.to_vec(), expr: "x.slice(-1, -3, 4)", wrt: &["x"], out_shape: &[3, 3],
            oracle: oracle!(|e| slice_ref(get(e, "x"), 1, 1, 4)), known: tape_only!("source AD does not extract `.slice`"), prelude: "" },
        Cert { name: "slice_mid", inputs: X_234.to_vec(), expr: "x.slice(1, 1, 3)", wrt: &["x"], out_shape: &[2, 2, 4],
            oracle: oracle!(|e| slice_ref(get(e, "x"), 1, 1, 3)), known: tape_only!("source AD does not extract `.slice`"), prelude: "" },
        Cert { name: "stack_dim0", inputs: AB_34.to_vec(), expr: "stack([a, b], 0)", wrt: &["a", "b"], out_shape: &[2, 3, 4],
            oracle: oracle!(|e| stack_ref(&[get(e, "a"), get(e, "b")], 0)), known: tape_only!("source AD does not extract `stack`"), prelude: "" },
        Cert { name: "stack_dim1", inputs: AB_34.to_vec(), expr: "stack([a, b], 1)", wrt: &["a", "b"], out_shape: &[3, 2, 4],
            oracle: oracle!(|e| stack_ref(&[get(e, "a"), get(e, "b")], 1)), known: tape_only!("source AD does not extract `stack`"), prelude: "" },
        Cert { name: "stack_neg", inputs: vec![inp("a", &[3, 4]), inp("b", &[3, 4]), inp("c", &[3, 4])],
            expr: "stack([a, b, c], -1)", wrt: &["a", "b", "c"], out_shape: &[3, 4, 3],
            oracle: oracle!(|e| stack_ref(&[get(e, "a"), get(e, "b"), get(e, "c")], 2)), known: tape_only!("source AD does not extract `stack`"), prelude: "" },
        // The stdlib Linear's bias: `bias_add(x @ w.transpose(0, 1), b)`.
        Cert { name: "bias_add", inputs: XY_ROW.to_vec(), expr: "bias_add(x, y)", wrt: &["x", "y"], out_shape: &[3, 4],
            oracle: oracle!(|e| binary(get(e, "x"), get(e, "y"), |a, b| a + b)), known: tape_only!("source AD does not extract `bias_add`"), prelude: "" },
        // Non-overlapping, overlapping (an input can win several windows, so
        // its gradient accumulates) and padded windows.
        Cert { name: "maxpool2d", inputs: X_POOL.to_vec(), expr: "maxpool2d(x, 2, 2, 2, 0)", wrt: &["x"], out_shape: &[1, 2, 2, 2],
            oracle: oracle!(|e| maxpool2d_ref(get(e, "x"), 2, 2, 0)), known: tape_only!("source AD does not extract `maxpool2d`"), prelude: "" },
        Cert { name: "maxpool2d_overlap", inputs: X_POOL.to_vec(), expr: "maxpool2d(x, 2, 2, 1, 0)", wrt: &["x"], out_shape: &[1, 2, 3, 3],
            oracle: oracle!(|e| maxpool2d_ref(get(e, "x"), 2, 1, 0)), known: tape_only!("source AD does not extract `maxpool2d`"), prelude: "" },
        Cert { name: "maxpool2d_pad", inputs: vec![inp("x", &[1, 1, 5, 5])], expr: "maxpool2d(x, 3, 3, 2, 1)", wrt: &["x"], out_shape: &[1, 1, 3, 3],
            oracle: oracle!(|e| maxpool2d_ref(get(e, "x"), 3, 2, 1)), known: tape_only!("source AD does not extract `maxpool2d`"), prelude: "" },
        // p = 0.5 with a fixed seed: the mask input `m` is the program's own
        // same-seed draw (seeded_dropout_ref checks it), so the forward and
        // the backward are held to one known mask. p = 0 cannot certify this
        // op: it records a Reshape, not a Dropout.
        Cert { name: "dropout_seeded", inputs: vec![inp("x", &[3, 4]), idx("m", &[3, 4], "cert_dropout_mask()")],
            expr: "cert_seeded_dropout(x)", wrt: &["x"], out_shape: &[3, 4],
            oracle: oracle!(|e| seeded_dropout_ref(get(e, "x"), get(e, "m"), 0.5)),
            known: tape_only!("source AD does not extract the seeding fn (nor `manual_seed` inside a method)"),
            prelude: DROPOUT_PRELUDE },
    ]
}

// ---------------------------------------------------------------------------
// Layout variants
// ---------------------------------------------------------------------------
//
// Every base certificate hands each op contiguous tensors: fresh `randn`
// inputs, and an output gradient that is the loss's own contiguous `r`. Real
// programs hand the backward strided views as well -- a `.transpose` after
// the op makes its output gradient one (the transpose's backward is a view),
// and a transposed input is one. A backward that reads storage linearly is
// right on every base certificate and wrong on these, so the layout is an
// axis of its own.

#[derive(Clone, Copy)]
enum Layout {
    /// `<base>_vgrad`: the op's output is transposed in dims (d0, d1)
    /// before the loss, so the gradient its backward receives is a strided
    /// view. Both dims must exceed 1 (or the view's storage order is the
    /// logical order and the variant is vacuous).
    ViewGrad(usize, usize),
    /// `<base>_vin`: every differentiated input of rank >= 2 is a strided
    /// view (`Input::view`), so the forward reads and the tape saves one.
    ViewIn,
}

/// The layout variants, one row per (base, layout) with the known failures
/// the variant adds to its base's (a tape-only base's fall back carries
/// over). Chosen so the transpose meets the op's output directly: a base
/// whose output passes through a `.reshape` first (`sum_dim`, `gather_mid`)
/// would hand the op a materialized gradient and certify nothing new.
///
/// What these catch: CPU code that indexes a tensor's storage linearly
/// (`data.add(i)`, a `memcpy` of the buffer), as if every tensor were
/// row-major. A strided gradient view (`_vgrad`) or a strided input -- read by
/// the forward and saved for the backward (`_vin`) -- then pairs elements in
/// storage order. When they were added, 31 of these failed for that reason
/// in tape mode (and conv2d / embedding in source mode too, through the
/// shared CPU kernels); every CPU kernel involved now reads its operands
/// through `nsl_runtime::tensor::with_row_major`, and none is listed.
#[rustfmt::skip]
const LAYOUT_VARIANTS: &[(&str, Layout, Known)] = &[
    ("add_row", Layout::ViewGrad(0, 1), &[]),
    ("sub_row", Layout::ViewGrad(0, 1), &[]),
    ("mul_col", Layout::ViewGrad(0, 1), &[]),
    ("div_row", Layout::ViewGrad(0, 1), &[]),
    ("matmul_2d", Layout::ViewGrad(0, 1), &[]),
    ("neg", Layout::ViewGrad(0, 1), &[]),
    ("cast_f64_round_trip", Layout::ViewGrad(0, 1), &[]),
    ("mul_literal", Layout::ViewGrad(0, 1), &[]),
    ("add_literal", Layout::ViewGrad(0, 1), &[]),
    ("transpose_3d", Layout::ViewGrad(0, 2), &[]),
    ("sum_dim_keepdim_mid", Layout::ViewGrad(0, 2), &[]),
    ("mean_dim_keepdim_mid", Layout::ViewGrad(0, 2), &[]),
    ("reduce_max_keepdim_last", Layout::ViewGrad(0, 1), &[]),
    ("exp", Layout::ViewGrad(0, 1), &[]),
    ("log", Layout::ViewGrad(0, 1), &[]),
    ("sqrt", Layout::ViewGrad(0, 1), &[]),
    ("abs", Layout::ViewGrad(0, 1), &[]),
    ("clamp", Layout::ViewGrad(0, 1), &[]),
    ("relu", Layout::ViewGrad(0, 1), &[]),
    ("gelu", Layout::ViewGrad(0, 1), &[]),
    ("silu", Layout::ViewGrad(0, 1), &[]),
    ("sin", Layout::ViewGrad(0, 1), &[]),
    ("cos", Layout::ViewGrad(0, 1), &[]),
    ("sigmoid", Layout::ViewGrad(0, 1), &[]),
    ("tanh", Layout::ViewGrad(0, 1), &[]),
    ("softmax_last", Layout::ViewGrad(0, 1), &[]),
    ("log_softmax_last", Layout::ViewGrad(0, 1), &[]),
    ("slice_dim1", Layout::ViewGrad(0, 1), &[]),
    ("reshape", Layout::ViewGrad(0, 1), &[]),
    ("cat_dim1", Layout::ViewGrad(0, 1), &[]),
    ("embedding", Layout::ViewGrad(0, 1), &[]),
    ("layernorm", Layout::ViewGrad(0, 1), &[]),
    ("rmsnorm", Layout::ViewGrad(0, 1), &[]),
    ("dropout_seeded", Layout::ViewGrad(0, 1), &[]),
    ("conv2d", Layout::ViewGrad(2, 3), &[]),
    ("maxpool2d", Layout::ViewGrad(2, 3), &[]),
    ("rotate_half", Layout::ViewGrad(0, 1), &[]),
    ("bias_add", Layout::ViewGrad(0, 1), &[]),
    ("unsqueeze", Layout::ViewGrad(1, 2), &[]),
    ("expand", Layout::ViewGrad(0, 1), &[]),
    ("stack_dim0", Layout::ViewGrad(0, 2), &[]),
    ("add_row", Layout::ViewIn, &[]),
    ("sub_row", Layout::ViewIn, &[]),
    ("mul_col", Layout::ViewIn, &[]),
    ("div_row", Layout::ViewIn, &[]),
    ("matmul_2d", Layout::ViewIn, &[]),
    ("neg", Layout::ViewIn, &[]),
    ("cast_f64_round_trip", Layout::ViewIn, &[]),
    ("mul_literal", Layout::ViewIn, &[]),
    ("add_literal", Layout::ViewIn, &[]),
    ("transpose_3d", Layout::ViewIn, &[]),
    ("sum_dim", Layout::ViewIn, &[]),
    ("mean_dim", Layout::ViewIn, &[]),
    ("reduce_max_dim1", Layout::ViewIn, &[]),
    ("gather", Layout::ViewIn, &[]),
    ("exp", Layout::ViewIn, &[]),
    ("log", Layout::ViewIn, &[]),
    ("sqrt", Layout::ViewIn, &[]),
    ("abs", Layout::ViewIn, &[]),
    ("clamp", Layout::ViewIn, &[]),
    ("relu", Layout::ViewIn, &[]),
    ("gelu", Layout::ViewIn, &[]),
    ("silu", Layout::ViewIn, &[]),
    ("sin", Layout::ViewIn, &[]),
    ("cos", Layout::ViewIn, &[]),
    ("sigmoid", Layout::ViewIn, &[]),
    ("tanh", Layout::ViewIn, &[]),
    ("softmax_last", Layout::ViewIn, &[]),
    ("log_softmax_last", Layout::ViewIn, &[]),
    ("slice_dim1", Layout::ViewIn, &[]),
    ("reshape", Layout::ViewIn, &[]),
    ("cat_dim1", Layout::ViewIn, &[]),
    ("embedding", Layout::ViewIn, &[]),
    ("layernorm", Layout::ViewIn, &[]),
    ("rmsnorm", Layout::ViewIn, &[]),
    ("dropout_seeded", Layout::ViewIn, &[]),
    ("conv2d", Layout::ViewIn, &[]),
    ("maxpool2d", Layout::ViewIn, &[]),
    ("rotate_half", Layout::ViewIn, &[]),
    ("bias_add", Layout::ViewIn, &[]),
    ("unsqueeze", Layout::ViewIn, &[]),
    ("stack_dim0", Layout::ViewIn, &[]),
];

/// Leak a runtime string into the `'static` table (once per process:
/// `certs()` builds the table once).
fn leak(s: String) -> &'static str {
    Box::leak(s.into_boxed_str())
}

/// The `layout` variant of `base`, failing as `base` does plus `known`.
fn layout_variant(base: &Cert, layout: Layout, known: Known) -> Cert {
    let Oracle::Fn(f) = base.oracle else {
        panic!(
            "{}: a layout variant's base must be a base certificate",
            base.name
        )
    };
    let mut all_known: Vec<(Mode, &'static str)> = base.known.to_vec();
    for k in known {
        assert!(
            all_known.iter().all(|(m, _)| *m != k.0),
            "{}: a layout variant re-lists its base's {:?} known failure",
            base.name,
            k.0
        );
        all_known.push(*k);
    }
    let known: Known = Box::leak(all_known.into_boxed_slice());
    match layout {
        Layout::ViewGrad(d0, d1) => {
            let mut shape = base.out_shape.to_vec();
            assert!(
                shape.len() > d0.max(d1) && shape[d0] > 1 && shape[d1] > 1,
                "{}: transposing dims {d0},{d1} of {shape:?} does not stride the gradient",
                base.name
            );
            shape.swap(d0, d1);
            Cert {
                name: leak(format!("{}_vgrad", base.name)),
                inputs: base.inputs.clone(),
                expr: leak(format!("({}).transpose({d0}, {d1})", base.expr)),
                wrt: base.wrt,
                out_shape: Box::leak(shape.into_boxed_slice()),
                oracle: Oracle::Transposed(f, d0, d1),
                known,
                prelude: base.prelude,
            }
        }
        Layout::ViewIn => {
            let inputs: Vec<Input> = base
                .inputs
                .iter()
                .map(|i| Input {
                    view: base.wrt.contains(&i.name)
                        && i.shape.len() >= 2
                        && !matches!(i.init, Init::Expr(_)),
                    ..*i
                })
                .collect();
            assert!(
                inputs.iter().any(|i| {
                    let r = i.shape.len();
                    i.view && i.shape[r - 1] > 1 && i.shape[r - 2] > 1
                }),
                "{}: no differentiated input whose view is strided",
                base.name
            );
            Cert {
                name: leak(format!("{}_vin", base.name)),
                inputs,
                expr: base.expr,
                wrt: base.wrt,
                out_shape: base.out_shape,
                oracle: base.oracle,
                known,
                prelude: base.prelude,
            }
        }
    }
}

/// Every certificate: the base table, then its layout variants.
fn certs() -> &'static [Cert] {
    static ALL: OnceLock<Vec<Cert>> = OnceLock::new();
    ALL.get_or_init(|| {
        let mut all = base_certs();
        let variants: Vec<Cert> = LAYOUT_VARIANTS
            .iter()
            .map(|&(base, layout, known)| {
                let b = all
                    .iter()
                    .find(|c| c.name == base)
                    .unwrap_or_else(|| panic!("layout variant of unknown certificate `{base}`"));
                layout_variant(b, layout, known)
            })
            .collect();
        all.extend(variants);
        all
    })
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
        // A view is drawn in the transposed shape and transposed back.
        let mut drawn = i.shape.to_vec();
        let r = drawn.len();
        if i.view {
            drawn.swap(r - 2, r - 1);
        }
        let init = match i.init {
            Init::Randn => format!("randn({})", shape_list(&drawn)),
            Init::Positive => format!("abs(randn({})) + 0.5", shape_list(&drawn)),
            Init::Expr(e) => e.to_string(),
        };
        let init = if i.view {
            format!("({init}).transpose({}, {})", r - 2, r - 1)
        } else {
            init
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
    let out = c.oracle.eval(env);
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

/// The prefix of a source run's `Err` when the grad block fell back to the
/// tape, and the phrase a `known` entry uses to claim exactly that.
const FELL_BACK: &str = "source AD fell back";
const FALLS_BACK: &str = "falls back to the tape";

/// The ops a tape run recorded, by `TapeOp` variant name: the run sets
/// `NSL_DEBUG_MEM_TRACE=1`, under which `maybe_record` logs
/// `[tape-trace] record <Variant> ...` for every op it pushes.
fn recorded_tape_ops(stderr: &str) -> BTreeSet<String> {
    stderr
        .lines()
        .filter_map(|l| l.split_once("[tape-trace] record ").map(|(_, rest)| rest))
        .filter_map(|rest| rest.split_whitespace().next())
        .map(str::to_string)
        .collect()
}

/// Run one certificate in one mode; `Err` describes how it failed. A tape
/// run also returns the ops it recorded (empty for a source run).
fn run_cert(c: &Cert, mode: Mode, recorded: &mut BTreeSet<String>) -> Result<(), String> {
    let root = repo_root();
    let tmp = tempfile::TempDir::new().unwrap();
    let path = tmp.path().join(format!("{}.nsl", c.name));
    std::fs::write(&path, program(c)).unwrap();
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_nsl"));
    cmd.arg("run");
    if mode == Mode::Source {
        cmd.arg("--source-ad");
    } else {
        cmd.env("NSL_DEBUG_MEM_TRACE", "1");
    }
    let out = cmd
        .arg(&path)
        .current_dir(tmp.path())
        .env("NSL_STDLIB_PATH", root.join("stdlib"))
        .output()
        .expect("spawn nsl run");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    if mode == Mode::Tape {
        *recorded = recorded_tape_ops(&stderr);
    }
    let tail = || {
        stderr
            .lines()
            .filter(|l| {
                !l.trim().is_empty()
                    && !l.starts_with("[tape-trace]")
                    && !l.starts_with("[tensor-trace]")
            })
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
            return Err(format!("{FELL_BACK} ({m})"));
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
    // The oracle is mode-independent, so a broken one panics rather than
    // returning an Err (which a `known` entry would absorb). A NaN oracle
    // would also pass the loss check below: `x > NaN` is false.
    assert!(
        base_loss.is_finite(),
        "{}: the f64 oracle loss is non-finite ({base_loss}) — the oracle or its \
         inputs are broken, not the AD rule",
        c.name
    );
    let mut failures = Vec::new();
    for &w in c.wrt {
        let loss = parse_between(
            &stdout,
            &format!("LOSS_{w}_BEGIN"),
            &format!("LOSS_{w}_END"),
        )
        .and_then(|v| v.first().copied())
        .ok_or_else(|| format!("loss for {w} not printed"))?;
        // `!is_finite` first: `NaN > tol` is false, so a NaN loss would pass.
        if !loss.is_finite() || (loss - base_loss).abs() > 1e-4 * base_loss.abs().max(1.0) {
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
        // `m.max(nan)` keeps `m` and `d > acc.1` is false for a NaN `d`, so a
        // NaN oracle gradient would drop out of `scale` and of the worst-error
        // fold below. Mode-independent, so it panics (see `base_loss`).
        if let Some(k) = want.iter().position(|v| !v.is_finite()) {
            panic!(
                "{}: the f64 finite-difference oracle d{w}[{k}] is non-finite ({})",
                c.name, want[k]
            );
        }
        // `d > acc.1` is false for a NaN `d`, so the fold below would skip a
        // NaN gradient entry. `parse_between` drops non-numeric tokens today
        // (the length check above then fails), but the comparator must not
        // depend on that.
        if let Some(k) = got.iter().position(|v| !v.is_finite()) {
            failures.push(format!("d{w}[{k}] = {} is non-finite", got[k]));
            continue;
        }
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
        let mut recorded = BTreeSet::new();
        let result = run_cert(c, mode, &mut recorded);
        if mode == Mode::Tape {
            eprintln!("{name} [Tape]: recorded {recorded:?}");
            problems.extend(tape_claim_problems(name, &recorded));
        }
        let known = c.known.iter().find(|(m, _)| *m == mode);
        match (result, known) {
            (Ok(()), None) => eprintln!("{name} [{mode:?}]: certified"),
            // A listed fall back must BE a fall back: once source AD extracts
            // the op, a wrong gradient must not hide behind the entry.
            (Err(e), Some((_, why))) if why.contains(FALLS_BACK) && !e.starts_with(FELL_BACK) => {
                problems.push(format!(
                    "{mode:?}: listed as a fall back ({why}), but it failed otherwise: {e}"
                ))
            }
            (Err(e), Some((_, why))) => eprintln!("{name} [{mode:?}]: known failure ({why}): {e}"),
            (Err(e), None) => problems.push(format!("{mode:?}: {e}")),
            (Ok(()), Some((_, why))) => problems.push(format!(
                "{mode:?} now PASSES but is listed as a known failure ({why}): remove it from `known`"
            )),
        }
    }
    assert!(problems.is_empty(), "{name}:\n  {}", problems.join("\n  "));
}

/// The tape inventory's claims about one certificate, held to what its tape
/// run recorded: every `TapeOp` whose status names it (as a certificate or a
/// defect) was recorded, and nothing it recorded is an op the inventory calls
/// GPU-only or unreachable.
fn tape_claim_problems(name: &str, recorded: &BTreeSet<String>) -> Vec<String> {
    let mut problems = Vec::new();
    for (op, status) in tape_cert_inventory() {
        let named = status.certified().contains(&name) || status.defects().contains(&name);
        if named && !recorded.contains(op) {
            problems.push(format!(
                "Tape: tape_cert_status names `{name}` for {op}, but its tape run recorded \
                 no {op} (it recorded {recorded:?})"
            ));
        }
        if matches!(
            status,
            TapeCertStatus::GpuOnly(_) | TapeCertStatus::Unreachable(_)
        ) && recorded.contains(op)
        {
            problems.push(format!(
                "Tape: the tape run recorded {op}, which tape_cert_status calls {status:?}"
            ));
        }
    }
    problems
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

/// Certificates for ops that exist only in a GPU `train` block (no CPU
/// `grad` spelling reaches them): one compiled SGD step whose update is the
/// raw gradient, against f64 central differences. They live in
/// `fused_loss_gradient_cert_gpu.rs` and run in the GPU cert lane.
const GPU_CERTS: &[&str] = &["fused_linear_ce_step", "fused_kl_ce_step", "sdpa_packed_step"];

/// The coverage gate: every certificate a `PrimalOp` status names exists here
/// (or in `GPU_CERTS`). `every_certificate_is_named_by_a_status` holds the
/// other direction.
#[test]
fn certificates_match_the_codegen_inventory() {
    let all = certs();
    let names: Vec<&str> = all.iter().map(|c| c.name).collect();
    let mut claimed: Vec<&str> = Vec::new();
    for (op, status) in ad_cert_inventory() {
        if let AdCertStatus::Certified(named) = status {
            for n in named {
                assert!(
                    names.contains(n) || GPU_CERTS.contains(n),
                    "{op} names certificate `{n}`, which is not in certs() or GPU_CERTS"
                );
                claimed.push(n);
            }
        }
    }
    // A GPU certificate is a GPU-ignored test fn of that exact name in
    // fused_loss_gradient_cert_gpu.rs, run by the cert lane.
    let gpu_src = include_str!("fused_loss_gradient_cert_gpu.rs");
    // Assembled, not written out: scripts/gpu-gate-inventory.awk reads the
    // attribute text line by line and would take a literal for a gate.
    let gpu_attr = concat!("#[", "ignore = \"requires CUDA GPU\"]");
    for n in GPU_CERTS {
        let at = gpu_src
            .find(&format!("\nfn {n}() {{"))
            .unwrap_or_else(|| panic!("GPU certificate `{n}` has no test fn in fused_loss_gradient_cert_gpu.rs"));
        let attrs = gpu_src[..at].trim_end();
        assert!(
            attrs.ends_with(gpu_attr) && attrs[..attrs.len() - gpu_attr.len()].trim_end().ends_with("#[test]"),
            "GPU certificate `{n}` must be a #[test] carrying {gpu_attr}, so the cert lane runs it"
        );
        assert!(claimed.contains(n), "GPU certificate `{n}` is not named by any PrimalOp status");
    }
}

/// The tape coverage gate: every certificate a `TapeOp` status names exists
/// here; one it names as CERTIFYING the op is not a known failure in tape
/// mode (a failing run certifies nothing), and one it names as a DEFECT is
/// (so a fix that flips the ratchet must move it to `certified`). `check`
/// holds each named certificate's tape run to recording the op.
#[test]
fn certificates_match_the_tape_inventory() {
    let all = certs();
    let find = |op: &str, n: &str| {
        all.iter()
            .find(|c| c.name == n)
            .unwrap_or_else(|| panic!("{op} names certificate `{n}`, which is not in certs()"))
    };
    let tape_fails = |c: &Cert| c.known.iter().any(|(m, _)| *m == Mode::Tape);
    for (op, status) in tape_cert_inventory() {
        for n in status.certified() {
            assert!(
                !tape_fails(find(op, n)),
                "{op} is certified by `{n}`, whose tape run is a known failure: a failing run \
                 certifies nothing (list it as a defect)"
            );
        }
        for n in status.defects() {
            assert!(
                tape_fails(find(op, n)),
                "{op} lists `{n}` as a defect, but its tape run is not a known failure: \
                 move it to `certified`"
            );
        }
    }
}

/// Every certificate is named by some status: as certifying a `PrimalOp` or
/// a `TapeOp`, or as a `TapeOp` defect. A DEFECT PROBE -- a known failure in
/// every mode -- certifies nothing, so it may only be named as a defect; its
/// fix flips the ratchet, and then the checks above move it.
#[test]
fn every_certificate_is_named_by_a_status() {
    let mut certifying: Vec<&str> = Vec::new();
    let mut defects: Vec<&str> = Vec::new();
    for (_, status) in ad_cert_inventory() {
        if let AdCertStatus::Certified(n) = status {
            certifying.extend_from_slice(n);
        }
    }
    for (_, status) in tape_cert_inventory() {
        certifying.extend_from_slice(status.certified());
        defects.extend_from_slice(status.defects());
    }
    for c in certs() {
        let probe = [Mode::Tape, Mode::Source]
            .iter()
            .all(|m| c.known.iter().any(|(k, _)| k == m));
        if probe {
            assert!(
                !certifying.contains(&c.name) && defects.contains(&c.name),
                "`{}` fails in every mode (a defect probe): name it as a TapeOp defect, and \
                 nowhere as a certificate",
                c.name
            );
        } else {
            assert!(
                certifying.contains(&c.name) || defects.contains(&c.name),
                "certificate `{}` is named by no PrimalOp status (ad_cert_status) and no \
                 TapeOp status (tape_cert_status)",
                c.name
            );
        }
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
            for c in all {
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
    layernorm, layernorm_eps, rmsnorm, rmsnorm_eps, layernorm_3d, layernorm_field_eps, rmsnorm_field_eps, rmsnorm_field_eps_reused,
    embedding, gather, gather_neg, gather_dim0, gather_mid, cat_dim0, cat_dim1, cat_three, cat_neg,
    cross_entropy, mse_loss, l1_loss,
    conv2d, sdpa, sdpa_causal, sdpa_scale,
    sdpa_packed, sdpa_packed_docs, sdpa_packed_batch, sdpa_packed_scale,
    cast_f64_round_trip, cast_f64_square,
    reduce_max_dim1, reduce_max_dim0, reduce_max_keepdim_last, reduce_max_keepdim_mid, reduce_max_mid,
    slice_dim1, slice_dim0, slice_neg, slice_mid,
    stack_dim0, stack_dim1, stack_neg,
    bias_add, maxpool2d, maxpool2d_overlap, maxpool2d_pad, dropout_seeded,
    sum_dim_keepdim_mid, mean_dim_keepdim_mid,
    add_row_vgrad, sub_row_vgrad, mul_col_vgrad, div_row_vgrad, matmul_2d_vgrad, neg_vgrad,
    cast_f64_round_trip_vgrad, mul_literal_vgrad, add_literal_vgrad, transpose_3d_vgrad,
    sum_dim_keepdim_mid_vgrad, mean_dim_keepdim_mid_vgrad, reduce_max_keepdim_last_vgrad, exp_vgrad,
    log_vgrad, sqrt_vgrad, abs_vgrad, clamp_vgrad, relu_vgrad, gelu_vgrad, silu_vgrad, sin_vgrad,
    cos_vgrad, sigmoid_vgrad, tanh_vgrad, softmax_last_vgrad, log_softmax_last_vgrad,
    slice_dim1_vgrad, reshape_vgrad, cat_dim1_vgrad, embedding_vgrad, layernorm_vgrad,
    rmsnorm_vgrad, dropout_seeded_vgrad, conv2d_vgrad, maxpool2d_vgrad, rotate_half_vgrad,
    bias_add_vgrad, unsqueeze_vgrad, expand_vgrad, stack_dim0_vgrad,
    add_row_vin, sub_row_vin, mul_col_vin, div_row_vin, matmul_2d_vin, neg_vin,
    cast_f64_round_trip_vin, mul_literal_vin, add_literal_vin, transpose_3d_vin, sum_dim_vin,
    mean_dim_vin, reduce_max_dim1_vin, gather_vin, exp_vin, log_vin, sqrt_vin, abs_vin, clamp_vin,
    relu_vin, gelu_vin, silu_vin, sin_vin, cos_vin, sigmoid_vin, tanh_vin, softmax_last_vin,
    log_softmax_last_vin, slice_dim1_vin, reshape_vin, cat_dim1_vin, embedding_vin, layernorm_vin,
    rmsnorm_vin, dropout_seeded_vin, conv2d_vin, maxpool2d_vin, rotate_half_vin, bias_add_vin,
    unsqueeze_vin, stack_dim0_vin,
}
