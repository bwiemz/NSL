//! The row-chunked fused linear-CE's softmax kernel
//! (`nsl_kir::kernels::lce_rows`, `nsl_lce_rows_grad_{f32,bf16}`) on the
//! cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`), with no
//! GPU:
//!
//! 1. **Bits.** One 256-thread block per row plus one past the last (which
//!    must write nothing), shared memory poisoned, under all four thread
//!    schedules: all of global memory after the launch is exactly what a
//!    model of the kernel's arithmetic leaves (the per-thread folds in
//!    column order, thread 0's in thread order, `lse = fma(lg2(s), ln 2, m)`,
//!    the gradient rounded once to the logits' storage), and nothing past the
//!    rows, the losses or the logits is touched. Rows of 1 to 1,000 columns
//!    (under, at and over one 256-column stride, with a tail), targets at
//!    the first and last column, inside, the ignore index and `i64::MIN`.
//! 2. **Math.** The model is the softmax cross-entropy: its loss and
//!    gradient agree with an f64 evaluation from the same logits to f32's
//!    precision (a model that replicated a wrong formula would pass 1 and
//!    fail here).

use std::collections::HashMap;

use half::bf16;
use nsl_kir::kernels::lce_rows::{LCE_ROWS_BLOCK, LN_2, PARAM_NAMES, kernel_name, ptx};
use nsl_kir::kernels::storage::Storage;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const LOG2_E: f32 = f32::from_bits(0x3FB8_AA3B);
const POISON: u32 = 0x7FA5_A5A5;
const TAIL: usize = 16;
const LOGITS: u64 = 0x1000_0000;
const TARGETS: u64 = 0x2000_0000;
const LOSS: u64 = 0x3000_0000;
const ORDERS: [Order; 4] = [
    Order::Ascending,
    Order::Descending,
    Order::WarpsAscending,
    Order::WarpsDescending,
];
const SCALE: f32 = 0.0137;

fn lcg(s: &mut u64) -> u64 {
    *s = s
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    *s >> 33
}

/// A row-major `rows × cols` logits block in `[-6, 6)`, with one row whose
/// max sits one stride in.
fn logits(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut v: Vec<f32> = (0..rows * cols)
        .map(|_| (lcg(&mut s) as f32 / (1u64 << 31) as f32) * 12.0 - 6.0)
        .collect();
    if cols > LCE_ROWS_BLOCK as usize {
        v[LCE_ROWS_BLOCK as usize] = 9.5;
    }
    v
}

fn targets(rows: usize, cols: usize, seed: u64) -> Vec<i64> {
    let mut s = seed;
    let kinds = [0, cols as i64 - 1, (cols / 2) as i64, -100, i64::MIN];
    (0..rows)
        .map(|r| match r % 6 {
            5 => (lcg(&mut s) % cols as u64) as i64,
            k => kinds[k % kinds.len()],
        })
        .collect()
}

/// The stored logits as f32 (bf16 storage rounds them first).
fn stored(v: &[f32], s: Storage) -> Vec<f32> {
    match s {
        Storage::F32 => v.to_vec(),
        Storage::Bf16 => v.iter().map(|&x| bf16::from_f32(x).to_f32()).collect(),
    }
}

fn encode(v: &[f32], s: Storage) -> Vec<u8> {
    let mut out: Vec<u8> = match s {
        Storage::F32 => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
        Storage::Bf16 => v
            .iter()
            .flat_map(|x| bf16::from_f32(*x).to_bits().to_le_bytes())
            .collect(),
    };
    out.extend(std::iter::repeat_n(0xA5, TAIL * 4));
    out
}

/// The kernel's arithmetic for one row: (loss, gradient as stored f32).
fn model_row(z: &[f32], t: i64, s: Storage) -> (f32, Vec<f32>) {
    let n = LCE_ROWS_BLOCK as usize;
    let cols = z.len();
    let mut m_t = vec![f32::NEG_INFINITY; n];
    for (k, &zk) in z.iter().enumerate() {
        m_t[k % n] = ptx_max_f32(m_t[k % n], zk);
    }
    let m = m_t[1..].iter().fold(m_t[0], |a, &b| ptx_max_f32(a, b));
    let ex2 = |x: f32, at: f32| ((x - at) * LOG2_E).exp2();
    let mut s_t = vec![0.0f32; n];
    for (k, &zk) in z.iter().enumerate() {
        s_t[k % n] += ex2(zk, m);
    }
    let sum = s_t[1..].iter().fold(s_t[0], |a, &b| a + b);
    let lse = sum.log2().mul_add(LN_2, m);
    let valid = t >= 0;
    let loss = if valid { lse - z[t as usize] } else { 0.0 };
    let grad = (0..cols)
        .map(|k| {
            let p = ex2(z[k], lse);
            let g = if k as i64 != t { p } else { p - 1.0 };
            let dl = if valid { g * SCALE } else { 0.0 };
            stored(&[dl], s)[0]
        })
        .collect();
    (loss, grad)
}

struct Out {
    logits: Vec<u8>,
    targets: Vec<u8>,
    loss: Vec<u8>,
}

fn run(
    prog: &Program,
    z: &[f32],
    t: &[i64],
    rows: usize,
    cols: usize,
    s: Storage,
    order: Order,
) -> Out {
    let mut global = vec![
        Segment {
            base: LOGITS,
            bytes: encode(z, s),
        },
        Segment {
            base: TARGETS,
            bytes: t.iter().flat_map(|x| x.to_le_bytes()).collect(),
        },
        Segment {
            base: LOSS,
            bytes: std::iter::repeat_n(POISON.to_le_bytes(), rows + TAIL)
                .flatten()
                .collect(),
        },
    ];
    let values = [
        LOGITS,
        TARGETS,
        LOSS,
        rows as u64,
        cols as u64,
        u64::from(SCALE.to_bits()),
    ];
    let args: HashMap<String, u64> = PARAM_NAMES
        .iter()
        .zip(values)
        .map(|(p, v)| (p.to_string(), v))
        .collect();
    for ctaid in 0..=rows as u32 {
        let mut l = Launch {
            prog,
            args: &args,
            global: &mut global,
            shared: std::iter::repeat_n(POISON.to_le_bytes(), LCE_ROWS_BLOCK as usize)
                .flatten()
                .collect(),
            ctaid,
            ctaid_y: 0,
            nctaid_x: rows as u32 + 1,
            nctaid_y: 1,
            ntid: LCE_ROWS_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    let mut it = global.into_iter();
    let logits = it.next().expect("logits").bytes;
    let targets = it.next().expect("targets").bytes;
    let loss = it.next().expect("loss").bytes;
    Out {
        logits,
        targets,
        loss,
    }
}

/// `(rows, cols)`.
const SHAPES: [(usize, usize); 7] = [
    (1, 1),
    (3, 5),
    (6, 256),
    (2, 257),
    (7, 300),
    (2, 1000),
    (12, 40),
];

#[test]
fn the_kernel_leaves_exactly_the_models_bits() {
    for s in Storage::ALL {
        let text = String::from_utf8(ptx(s)).expect("ASCII");
        assert!(text.contains(kernel_name(s)));
        let prog = parse(text.trim_end_matches('\0'));
        for (i, &(rows, cols)) in SHAPES.iter().enumerate() {
            let raw = logits(rows, cols, 31 + i as u64);
            let z = stored(&raw, s);
            let t = targets(rows, cols, 77 + i as u64);
            let mut want_logits = Vec::new();
            let mut want_loss = Vec::new();
            for r in 0..rows {
                let (loss, grad) = model_row(&z[r * cols..(r + 1) * cols], t[r], s);
                want_loss.push(loss);
                want_logits.extend(grad);
            }
            let mut want_loss_bytes: Vec<u8> =
                want_loss.iter().flat_map(|x| x.to_le_bytes()).collect();
            want_loss_bytes.extend(std::iter::repeat_n(POISON.to_le_bytes(), TAIL).flatten());
            let want_t: Vec<u8> = t.iter().flat_map(|x| x.to_le_bytes()).collect();
            for order in ORDERS {
                let got = run(&prog, &z, &t, rows, cols, s, order);
                let what = format!("{s:?} rows {rows} cols {cols} {order:?}");
                assert!(
                    got.logits == encode(&want_logits, s),
                    "{what}: gradient bits"
                );
                assert!(got.loss == want_loss_bytes, "{what}: loss bits");
                assert!(got.targets == want_t, "{what}: targets written");
            }
        }
    }
}

#[test]
fn the_model_is_the_softmax_cross_entropy() {
    for &(rows, cols) in &SHAPES {
        let z = logits(rows, cols, 5 + cols as u64);
        let t = targets(rows, cols, 9 + rows as u64);
        for r in 0..rows {
            let zr = &z[r * cols..(r + 1) * cols];
            let (loss, grad) = model_row(zr, t[r], Storage::F32);
            let m = zr
                .iter()
                .fold(f64::NEG_INFINITY, |a, &b| a.max(f64::from(b)));
            let sum: f64 = zr.iter().map(|&x| (f64::from(x) - m).exp()).sum();
            let lse = m + sum.ln();
            if t[r] < 0 {
                assert_eq!(loss, 0.0);
                assert!(grad.iter().all(|&g| g == 0.0));
                continue;
            }
            let want = lse - f64::from(zr[t[r] as usize]);
            assert!(
                (f64::from(loss) - want).abs() <= 2e-5 * want.abs().max(1.0),
                "rows {rows} cols {cols}: {loss} vs {want}"
            );
            for (k, &g) in grad.iter().enumerate() {
                let p = (f64::from(zr[k]) - lse).exp();
                let want = (p - if k as i64 == t[r] { 1.0 } else { 0.0 }) * f64::from(SCALE);
                assert!(
                    (f64::from(g) - want).abs() <= 1e-5 * f64::from(SCALE),
                    "rows {rows} cols {cols} k {k}: {g} vs {want}"
                );
            }
        }
    }
}
