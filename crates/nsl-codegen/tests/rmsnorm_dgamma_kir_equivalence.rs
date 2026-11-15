//! The gate for the fused RMSNorm gamma-backward kernels built by
//! `nsl_kir::kernels::rmsnorm_dgamma`: `nsl_rmsnorm_rinv_rows_f32`,
//! `nsl_rmsnorm_dgamma_partial_f32` and `nsl_rmsnorm_dgamma_final_f32`, run
//! on the cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`).
//!
//! Until 2026-11-15 this file held the KIR kernels to the frozen hand-written
//! ones byte for byte. Those were a thread per row and a thread per column,
//! each a sequential loop; the kernels that replaced them are block-parallel
//! reductions, so their sums run in a different order and the bytes differ.
//! The gate now holds each kernel, on shapes with tails against every block
//! and chunk size, with one block more than the work needs in every grid
//! dimension, to:
//!
//! 1. **An f64 reference, within the derived bound.** The module docs derive
//!    the bounds; `bounds` below restates them from the documented reduction
//!    shape. This is the independent check: the inputs are exact f32 values,
//!    the reference sums their products in f64 (exact products, sums good to
//!    ~1e-16), and a dropped chunk, a wrong stride or a missing final pass is
//!    many times the bound (the `the_f64_check_bites_*` tests show it).
//! 2. **The documented order, bit for bit.** An f32 restatement of each
//!    kernel's summation order (`model_*`). The bounds are derived for that
//!    order, so this is what makes them the kernels' bounds.
//! 3. **Determinism.** The same bits under every interpreter schedule
//!    (threads ascending or descending, warps run whole barrier intervals
//!    ahead in either order) and with the CTAs in either order.
//! 4. **Nothing else.** Every input comes back untouched and nothing is
//!    written past an output.
//!
//! Gradients are drawn at realistic magnitudes (`dy` from 1e-9 to 1e-6 of
//! either sign, as the 2026-11-12 attention bug taught), activations from
//! 2^-10 to 2^10, with zeros and signed zeros.

use std::collections::HashMap;

use nsl_kir::kernels::rmsnorm_dgamma::{
    dgamma_plan, ptx, DgammaPlan, RmsNormDgammaOp, FINAL_BLOCK, FINAL_COLS, PARTIAL_BLOCK, RINV_BLOCK, WARPS,
};

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 64;
const POISON: u32 = 0x7FA5_A5A5;
const DY: u64 = 0x1000_0000;
const X: u64 = 0x2000_0000;
const RINV: u64 = 0x3000_0000;
const PARTIAL: u64 = 0x4000_0000;
const OUT: u64 = 0x5000_0000;
const EPS: f32 = 1e-5;
const U: f64 = 1.0 / (1u64 << 24) as f64;
const ORDERS: [Order; 4] = [Order::Ascending, Order::Descending, Order::WarpsAscending, Order::WarpsDescending];

fn kir_ptx(op: RmsNormDgammaOp) -> String {
    String::from_utf8(ptx(op)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    *s >> 33
}

/// `n` values `±m · 2^e`, `m` in [1, 2) with ten significant bits, `e` in
/// `lo..=hi`, with a zero and a `-0.0` every 13 and 17 elements.
fn values(n: usize, seed: u64, lo: i32, hi: i32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|k| {
            if k % 13 == 4 {
                return 0.0;
            }
            if k % 17 == 9 {
                return -0.0;
            }
            let e = lo + (lcg(&mut s) % (hi - lo + 1) as u64) as i32;
            let m = 1.0 + (lcg(&mut s) % 1024) as f32 / 1024.0;
            let sign = if lcg(&mut s).is_multiple_of(2) { 1.0 } else { -1.0 };
            sign * m * 2f32.powi(e)
        })
        .collect()
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// Activations: 2^-10 .. 2^10.
fn activations(n: usize, seed: u64) -> Vec<f32> {
    values(n, seed, -10, 10)
}

/// Gradients: 2^-30 .. 2^-20, about 1e-9 .. 1e-6.
fn gradients(n: usize, seed: u64) -> Vec<f32> {
    values(n, seed, -30, -20)
}

/// Positive, varied `1 / rms` values.
fn rinvs(n: usize, seed: u64) -> Vec<f32> {
    values(n, seed, -6, 6).into_iter().map(|v| if v == 0.0 { 0.75 } else { v.abs() }).collect()
}

// ---------------------------------------------------------------------------
// Running a kernel
// ---------------------------------------------------------------------------

/// One launch of `ptx` over a `grid` of CTAs (x, y), visiting them in order
/// or reversed. Shared memory starts poisoned, so a read before a write is a
/// NaN in the result, never a plausible zero.
fn launch(ptx: &str, args: &HashMap<String, u64>, global: &mut [Segment], grid: (u32, u32), block: u32, order: Order, reverse: bool) {
    let prog = parse(ptx);
    let mut ctas: Vec<(u32, u32)> = (0..grid.1).flat_map(|y| (0..grid.0).map(move |x| (x, y))).collect();
    if reverse {
        ctas.reverse();
    }
    for (x, y) in ctas {
        let shared = le32(&vec![POISON; prog.shared_bytes.div_ceil(4)]);
        let mut l = Launch {
            prog: &prog,
            args,
            global,
            shared,
            ctaid: x,
            ctaid_y: y,
            nctaid_x: grid.0,
            nctaid_y: grid.1,
            ntid: block,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
}

fn args_of(op: RmsNormDgammaOp, vals: &[u64]) -> HashMap<String, u64> {
    assert_eq!(op.param_names().len(), vals.len());
    op.param_names().iter().zip(vals).map(|(p, v)| (p.to_string(), *v)).collect()
}

/// The output segment's words, after checking every other segment came back
/// as it went in and nothing was written past `n_out`.
fn take_output(global: &[Segment], inputs: &[Vec<u32>], out: usize, n_out: usize) -> Result<Vec<u32>, String> {
    for (k, seg) in global.iter().enumerate().filter(|(k, _)| *k != out) {
        if words(&seg.bytes) != inputs[k] {
            return Err(format!("input segment {k} was written"));
        }
    }
    let w = words(&global[out].bytes);
    if w[n_out..].iter().any(|&x| x != POISON) {
        return Err("wrote past the output".into());
    }
    Ok(w[..n_out].to_vec())
}

/// Run `op` on the given segments with one CTA more than the work in each
/// grid dimension; the last segment is the output. Panics (a fault, an
/// unknown instruction, a barrier some threads never reach) are returned as
/// errors, so a mutant that faults counts as caught.
fn run(op: RmsNormDgammaOp, ptx: &str, segs: &[(u64, Vec<u32>)], n_out: usize, vals: &[u64], grid: (u32, u32), order: Order, reverse: bool) -> Result<Vec<u32>, String> {
    let mut global: Vec<Segment> = segs.iter().map(|(base, w)| Segment { base: *base, bytes: le32(w) }).collect();
    let mut inputs: Vec<Vec<u32>> = segs.iter().map(|(_, w)| w.clone()).collect();
    let out = segs.len() - 1;
    inputs[out] = vec![];
    let args = args_of(op, vals);
    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| launch(ptx, &args, &mut global, grid, op.block(), order, reverse)))
        .map_err(|e| format!("the launch failed: {:?}", e.downcast_ref::<String>().cloned().or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))))?;
    take_output(&global, &inputs, out, n_out)
}

fn poisoned(n: usize) -> Vec<u32> {
    vec![POISON; n + TAIL]
}

fn run_rinv(ptx: &str, x: &[f32], rows: usize, cols: usize, order: Order, reverse: bool) -> Result<Vec<u32>, String> {
    let segs = [(X, bits(x)), (RINV, poisoned(rows))];
    let vals = [X, RINV, rows as u64, cols as u64, u64::from(EPS.to_bits())];
    run(RmsNormDgammaOp::RinvRows, ptx, &segs, rows, &vals, (rows as u32 + 1, 1), order, reverse)
}

struct PartialCase<'a> {
    dy: &'a [f32],
    x: &'a [f32],
    rinv: &'a [f32],
    rows: usize,
    cols: usize,
    chunk_rows: usize,
}

impl PartialCase<'_> {
    fn chunks(&self) -> usize {
        self.rows.div_ceil(self.chunk_rows)
    }
}

fn run_partial(ptx: &str, c: &PartialCase, order: Order, reverse: bool) -> Result<Vec<u32>, String> {
    let n_out = c.chunks() * c.cols;
    let segs = [(DY, bits(c.dy)), (X, bits(c.x)), (RINV, bits(c.rinv)), (PARTIAL, poisoned(n_out))];
    let vals = [DY, X, RINV, PARTIAL, c.rows as u64, c.cols as u64, c.chunk_rows as u64];
    let grid = (c.cols.div_ceil(PARTIAL_BLOCK as usize) as u32 + 1, c.chunks() as u32 + 1);
    run(RmsNormDgammaOp::Partial, ptx, &segs, n_out, &vals, grid, order, reverse)
}

fn run_final(ptx: &str, partial: &[f32], chunks: usize, cols: usize, order: Order, reverse: bool) -> Result<Vec<u32>, String> {
    let segs = [(PARTIAL, bits(partial)), (OUT, poisoned(cols))];
    let vals = [PARTIAL, OUT, chunks as u64, cols as u64];
    let grid = (DgammaPlan::final_blocks(cols as u64) as u32 + 1, 1);
    run(RmsNormDgammaOp::Final, ptx, &segs, cols, &vals, grid, order, reverse)
}

// ---------------------------------------------------------------------------
// The bounds (module docs, restated from the reductions' shapes)
// ---------------------------------------------------------------------------

fn gamma(n: usize) -> f64 {
    let nu = n as f64 * U;
    nu / (1.0 - nu)
}

/// Kernel 1: `ceil(cols / 256)` fma steps, 5 butterfly levels, 3 tree
/// levels, the division and `+ eps`: `1 + θ_{m+10}` on `s / cols + eps`;
/// then the square root and the reciprocal round once each.
fn rinv_bound(cols: usize) -> f64 {
    let m = cols.div_ceil(RINV_BLOCK as usize);
    (1.0 + U) / ((1.0 - U) * (1.0 - gamma(m + 10)).sqrt()) - 1.0
}

/// Kernel 2 alone: two products, then `chunk_rows` adds.
fn partial_depth(chunk_rows: usize) -> usize {
    2 + chunk_rows
}

/// Kernel 3 alone: `ceil(chunks / 8)` adds in a warp, 3 in the tree.
fn final_depth(chunks: usize) -> usize {
    chunks.div_ceil(WARPS as usize) + 3
}

/// The f64 work of the f64 reference itself: a sum of `n` f64 terms is
/// within `n · 2^-53` of the exact sum, relative to the sum of magnitudes.
fn f64_slack(n: usize) -> f64 {
    (n as f64 + 4.0) * f64::EPSILON
}

fn f64_rinv(x: &[f32], rows: usize, cols: usize) -> Vec<f64> {
    (0..rows)
        .map(|r| {
            let s: f64 = (0..cols).map(|j| f64::from(x[r * cols + j]).powi(2)).sum();
            1.0 / (s / cols as f64 + f64::from(EPS)).sqrt()
        })
        .collect()
}

/// `got` against `want` within `bound[k]`: the first violation.
fn within(got: &[u32], want: &[f64], bound: &[f64], what: &str) -> Result<(), String> {
    for (k, (&g, (&w, &b))) in got.iter().zip(want.iter().zip(bound)).enumerate() {
        let g = f64::from(f32::from_bits(g));
        if !((g - w).abs() <= b) {
            return Err(format!("{what}[{k}] = {g:e}, f64 reference {w:e}, |error| {:e} > bound {b:e}", (g - w).abs()));
        }
    }
    Ok(())
}

fn check_rinv_f64(got: &[u32], x: &[f32], rows: usize, cols: usize) -> Result<(), String> {
    let want = f64_rinv(x, rows, cols);
    let bound: Vec<f64> = want.iter().map(|w| w * (rinv_bound(cols) + f64_slack(cols))).collect();
    within(got, &want, &bound, "rinv")
}

/// Column sums of `t[i, j]` over the row ranges `ranges`, in f64, with the
/// bound `γ_depth · Σ|t| + slack`.
fn f64_column_sums(t: impl Fn(usize, usize) -> f64, ranges: &[(usize, usize)], cols: usize, depth: usize) -> (Vec<f64>, Vec<f64>) {
    let mut want = vec![];
    let mut bound = vec![];
    for &(r0, r1) in ranges {
        for j in 0..cols {
            let s: f64 = (r0..r1).map(|i| t(i, j)).sum();
            let a: f64 = (r0..r1).map(|i| t(i, j).abs()).sum();
            want.push(s);
            bound.push(a * (gamma(depth) + f64_slack(r1 - r0)));
        }
    }
    (want, bound)
}

fn check_partial_f64(got: &[u32], c: &PartialCase) -> Result<(), String> {
    let ranges: Vec<(usize, usize)> = (0..c.chunks()).map(|k| (k * c.chunk_rows, ((k + 1) * c.chunk_rows).min(c.rows))).collect();
    let t = |i: usize, j: usize| f64::from(c.dy[i * c.cols + j]) * f64::from(c.x[i * c.cols + j]) * f64::from(c.rinv[i]);
    let (want, bound) = f64_column_sums(t, &ranges, c.cols, partial_depth(c.chunk_rows));
    within(got, &want, &bound, "partial")
}

fn check_final_f64(got: &[u32], partial: &[f32], chunks: usize, cols: usize) -> Result<(), String> {
    let t = |i: usize, j: usize| f64::from(partial[i * cols + j]);
    let (want, bound) = f64_column_sums(t, &[(0, chunks)], cols, final_depth(chunks));
    within(got, &want, &bound, "dgamma")
}

// ---------------------------------------------------------------------------
// The documented orders, in f32
// ---------------------------------------------------------------------------

/// `((v0 + v1) + (v2 + v3)) + ((v4 + v5) + (v6 + v7))`.
fn tree8(v: [f32; 8]) -> f32 {
    ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]))
}

/// A warp's butterfly: at each level every lane adds its partner's value.
fn butterfly(mut lanes: [f32; 32]) -> f32 {
    for d in [16, 8, 4, 2, 1] {
        let before = lanes;
        for (l, v) in lanes.iter_mut().enumerate() {
            *v = before[l] + before[l ^ d];
        }
    }
    assert!(lanes.iter().all(|v| v.to_bits() == lanes[0].to_bits()), "every lane ends with the same bits");
    lanes[0]
}

fn model_rinv(x: &[f32], rows: usize, cols: usize) -> Vec<u32> {
    let b = RINV_BLOCK as usize;
    (0..rows)
        .map(|r| {
            let thread = |t: usize| (t..cols).step_by(b).fold(0.0f32, |s, j| x[r * cols + j].mul_add(x[r * cols + j], s));
            let warps: Vec<f32> = (0..WARPS as usize).map(|w| butterfly(std::array::from_fn(|l| thread(w * 32 + l)))).collect();
            let s = tree8(warps.try_into().unwrap());
            (1.0f32 / (s / cols as f32 + EPS).sqrt()).to_bits()
        })
        .collect()
}

fn model_partial(c: &PartialCase) -> Vec<u32> {
    let mut out = vec![];
    for k in 0..c.chunks() {
        for j in 0..c.cols {
            let r = k * c.chunk_rows..((k + 1) * c.chunk_rows).min(c.rows);
            let s = r.fold(0.0f32, |a, i| a + (c.dy[i * c.cols + j] * c.x[i * c.cols + j]) * c.rinv[i]);
            out.push(s.to_bits());
        }
    }
    out
}

fn model_final(partial: &[f32], chunks: usize, cols: usize) -> Vec<u32> {
    let w = WARPS as usize;
    (0..cols)
        .map(|j| tree8(std::array::from_fn(|q| (q..chunks).step_by(w).fold(0.0f32, |a, k| a + partial[k * cols + j]))).to_bits())
        .collect()
}

// ---------------------------------------------------------------------------
// The cases
// ---------------------------------------------------------------------------

/// `(rows, cols)` for kernel 1: a single element; widths under a warp, at
/// and past the block, past two blocks; single-column rows, some of them
/// zero (so `eps` alone sets `rinv`).
const RINV_SHAPES: [(usize, usize); 8] = [(1, 1), (3, 7), (2, 256), (2, 257), (3, 600), (5, 33), (1, 1000), (37, 1)];

/// `(rows, cols, chunk_rows)` for kernel 2: one chunk, one row per chunk,
/// ragged last chunks, a chunk larger than the rows, column tails against
/// the 256-thread block, the plan's own choice.
fn partial_cases() -> Vec<(usize, usize, usize)> {
    let mut v = vec![(1, 1, 8), (37, 1, 1), (9, 33, 4), (70, 13, 3), (20, 300, 7), (5, 513, 32), (12, 257, 12)];
    let p = dgamma_plan(130, 260);
    v.push((130, 260, p.chunk_rows as usize));
    v
}

/// `(chunks, cols)` for kernel 3: no chunks, fewer chunks than warps, one
/// per warp, one past, a ragged spread; widths under, at and past the
/// 32-column block.
const FINAL_CASES: [(usize, usize); 8] = [(0, 5), (1, 1), (7, 31), (8, 32), (9, 33), (17, 100), (40, 3), (205, 40)];

/// `(rows, cols)` for the three kernels in sequence, partitioned by the
/// runtime's plan.
const PIPELINE_SHAPES: [(usize, usize); 6] = [(1, 1), (3, 7), (37, 1), (9, 33), (130, 260), (300, 40)];

#[test]
fn rinv_is_within_its_bound_in_the_documented_order_on_every_schedule() {
    let p = kir_ptx(RmsNormDgammaOp::RinvRows);
    for (n, &(rows, cols)) in RINV_SHAPES.iter().enumerate() {
        let x = activations(rows * cols, n as u64 + 1);
        let first = run_rinv(&p, &x, rows, cols, Order::Ascending, false).unwrap();
        check_rinv_f64(&first, &x, rows, cols).unwrap_or_else(|e| panic!("{rows}x{cols}: {e}"));
        assert_eq!(first, model_rinv(&x, rows, cols), "{rows}x{cols}: the documented order");
        for order in ORDERS {
            for reverse in [false, true] {
                assert_eq!(run_rinv(&p, &x, rows, cols, order, reverse).unwrap(), first, "{rows}x{cols} {order:?} {reverse}");
            }
        }
    }
}

#[test]
fn partial_is_within_its_bound_in_the_documented_order_on_every_schedule() {
    let p = kir_ptx(RmsNormDgammaOp::Partial);
    for (n, (rows, cols, chunk_rows)) in partial_cases().into_iter().enumerate() {
        let (dy, x, rinv) = (gradients(rows * cols, n as u64 + 11), activations(rows * cols, n as u64 + 21), rinvs(rows, n as u64 + 31));
        let c = PartialCase { dy: &dy, x: &x, rinv: &rinv, rows, cols, chunk_rows };
        let first = run_partial(&p, &c, Order::Ascending, false).unwrap();
        check_partial_f64(&first, &c).unwrap_or_else(|e| panic!("{rows}x{cols}/{chunk_rows}: {e}"));
        assert_eq!(first, model_partial(&c), "{rows}x{cols}/{chunk_rows}: the documented order");
        for order in [Order::Descending, Order::WarpsDescending] {
            assert_eq!(run_partial(&p, &c, order, true).unwrap(), first, "{rows}x{cols}/{chunk_rows} {order:?}");
        }
    }
}

#[test]
fn final_is_within_its_bound_in_the_documented_order_on_every_schedule() {
    let p = kir_ptx(RmsNormDgammaOp::Final);
    for (n, &(chunks, cols)) in FINAL_CASES.iter().enumerate() {
        // Partials at the magnitude of a column sum of gradients.
        let partial = values(chunks * cols, n as u64 + 41, -24, -14);
        let first = run_final(&p, &partial, chunks, cols, Order::Ascending, false).unwrap();
        check_final_f64(&first, &partial, chunks, cols).unwrap_or_else(|e| panic!("{chunks}x{cols}: {e}"));
        assert_eq!(first, model_final(&partial, chunks, cols), "{chunks}x{cols}: the documented order");
        for order in ORDERS {
            for reverse in [false, true] {
                assert_eq!(run_final(&p, &partial, chunks, cols, order, reverse).unwrap(), first, "{chunks}x{cols} {order:?} {reverse}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// The three kernels in sequence, as the runtime launches them
// ---------------------------------------------------------------------------

struct Kernels {
    rinv: String,
    partial: String,
    fin: String,
}

fn kernels() -> Kernels {
    Kernels {
        rinv: kir_ptx(RmsNormDgammaOp::RinvRows),
        partial: kir_ptx(RmsNormDgammaOp::Partial),
        fin: kir_ptx(RmsNormDgammaOp::Final),
    }
}

/// What the harness does between the launches: `Runtime` is the runtime's
/// sequence; the others are the broken launchers the f64 check must catch.
#[derive(Clone, Copy, PartialEq, Debug)]
enum Harness {
    Runtime,
    /// The final pass is told one chunk fewer: the last chunk's rows drop.
    DroppedChunk,
    /// The final pass never runs: dgamma is what the buffer held.
    NoFinalPass,
}

/// `dgamma` for `(rows, cols)` through the three kernels, and the f64
/// check of the whole formula: `|dĝ - Σ dy·x·rinv| ≤ ((1 + ε_rinv)(1 + γ_D)
/// - 1) · Σ|dy·x·rinv|` with `rinv` the f64 one, `D` the plan's depth.
fn pipeline(k: &Kernels, rows: usize, cols: usize, seed: u64, harness: Harness) -> Result<Vec<u32>, String> {
    let plan = dgamma_plan(rows as u64, cols as u64);
    let (chunk_rows, chunks) = (plan.chunk_rows as usize, plan.chunks as usize);
    assert_eq!(plan.depth() as usize, partial_depth(chunk_rows) + final_depth(chunks), "the module's depth is the derivation's");
    let (dy, x) = (gradients(rows * cols, seed), activations(rows * cols, seed + 7));
    let rinv: Vec<f32> = run_rinv(&k.rinv, &x, rows, cols, Order::Ascending, false)?.into_iter().map(f32::from_bits).collect();
    let c = PartialCase { dy: &dy, x: &x, rinv: &rinv, rows, cols, chunk_rows };
    let partial: Vec<f32> = run_partial(&k.partial, &c, Order::Ascending, false)?.into_iter().map(f32::from_bits).collect();
    let dgamma = match harness {
        Harness::Runtime => run_final(&k.fin, &partial, chunks, cols, Order::Ascending, false)?,
        Harness::DroppedChunk => run_final(&k.fin, &partial, chunks.saturating_sub(1), cols, Order::Ascending, false)?,
        Harness::NoFinalPass => poisoned(cols)[..cols].to_vec(),
    };
    let r = f64_rinv(&x, rows, cols);
    let t = |i: usize, j: usize| f64::from(dy[i * cols + j]) * f64::from(x[i * cols + j]) * r[i];
    let depth = partial_depth(chunk_rows) + final_depth(chunks);
    let scale = (1.0 + rinv_bound(cols)) * (1.0 + gamma(depth)) - 1.0 + f64_slack(rows + cols);
    let want: Vec<f64> = (0..cols).map(|j| (0..rows).map(|i| t(i, j)).sum()).collect();
    let bound: Vec<f64> = (0..cols).map(|j| (0..rows).map(|i| t(i, j).abs()).sum::<f64>() * scale).collect();
    within(&dgamma, &want, &bound, "dgamma")?;
    Ok(dgamma)
}

#[test]
fn the_three_kernels_compute_dgamma_within_the_bound() {
    let k = kernels();
    for (n, &(rows, cols)) in PIPELINE_SHAPES.iter().enumerate() {
        pipeline(&k, rows, cols, n as u64 + 51, Harness::Runtime).unwrap_or_else(|e| panic!("{rows}x{cols}: {e}"));
    }
}

/// With no rows only the final pass runs (the runtime skips the others),
/// and it writes zeros.
#[test]
fn no_rows_gives_zeros() {
    let out = run_final(&kir_ptx(RmsNormDgammaOp::Final), &[], 0, 37, Order::Ascending, false).unwrap();
    assert_eq!(out, vec![0u32; 37]);
}

// ---------------------------------------------------------------------------
// The f64 check bites
// ---------------------------------------------------------------------------

/// The three breakages the bound must see, through the f64 check alone: a
/// dropped chunk, a missing final pass, and (below) a wrong stride.
#[test]
fn the_f64_check_bites_a_dropped_chunk_and_a_missing_final_pass() {
    let k = kernels();
    for &(rows, cols) in &[(130usize, 260usize), (300, 40), (9, 33)] {
        for harness in [Harness::DroppedChunk, Harness::NoFinalPass] {
            let e = pipeline(&k, rows, cols, 77, harness).expect_err("passed the f64 check");
            assert!(e.contains("f64 reference"), "{rows}x{cols} {harness:?}: not the bound that failed: {e}");
        }
    }
}

/// `ptx` with the `i`-th line containing `from` rewritten by `g`.
fn edit(ptx: &str, from: &str, i: usize, g: impl Fn(&str) -> String) -> String {
    let at = ptx
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains(from))
        .nth(i)
        .unwrap_or_else(|| panic!("no line {i} with `{from}`"))
        .0;
    ptx.lines().enumerate().map(|(n, l)| if n == at { g(l) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
}

fn nudge(ptx: &str, from: &str, to: &str, i: usize) -> String {
    edit(ptx, from, i, |l| l.replacen(from, to, 1))
}

/// `op.ty d, a, b;` as `mov.ty d, a;`: the operation dropped.
fn drop_op(ptx: &str, op: &str, ty: &str, i: usize) -> String {
    edit(ptx, op, i, |l| {
        let (head, operands) = l.split_once(op).expect("the op");
        let mut ops: Vec<&str> = operands.trim_end_matches(';').split(',').map(str::trim).collect();
        ops.truncate(2);
        format!("{head}mov.{ty} {};", ops.join(", "))
    })
}

/// Whether the f64 check of the pipeline's output fails with `k` on some
/// shape: a value outside the bound, not a fault.
fn f64_caught(k: &Kernels) -> bool {
    PIPELINE_SHAPES.iter().enumerate().any(|(n, &(rows, cols))| {
        matches!(pipeline(k, rows, cols, n as u64 + 51, Harness::Runtime), Err(e) if e.contains("f64 reference"))
    })
}

/// The register `ld.param.u64` loads `param` into.
fn param_reg(ptx: &str, param: &str) -> String {
    let l = ptx.lines().find(|l| l.contains(&format!("[param_{param}];"))).unwrap_or_else(|| panic!("no param {param}"));
    l.split_whitespace().nth(1).expect("a register").trim_end_matches(',').to_string()
}

/// The partial kernel's row stride (`at += cols`) dropped, so every row
/// reads the chunk's first; the same stride off by one column; and the
/// final pass's chunk stride (`c += 8`, `at += 8·cols`) as 16: every other
/// chunk dropped.
#[test]
fn the_f64_check_bites_a_wrong_stride() {
    let base = kernels();
    let p = &base.partial;
    // `at += cols` is the loop body's one add of `cols` (the body starts at
    // the first load).
    let cols = param_reg(p, "cols");
    let body_start = p.lines().position(|l| l.contains("ld.global.f32")).unwrap();
    let adds: Vec<(usize, &str)> = p.lines().enumerate().filter(|(_, l)| l.contains("add.u64 ")).collect();
    let stride: Vec<usize> = adds.iter().enumerate().filter(|(_, (n, l))| *n > body_start && l.ends_with(&format!(", {cols};"))).map(|(i, _)| i).collect();
    assert_eq!(stride.len(), 1, "one `at += cols`\n{p}");
    let k = Kernels { partial: drop_op(p, "add.u64 ", "u64", stride[0]), ..kernels() };
    assert!(f64_caught(&k), "a partial row stride of 0 passed");
    // `cols` itself one larger for the stride only: `at += cols + 1`.
    let wider = edit(p, "add.u64 ", stride[0], |l| {
        let (head, _) = l.split_once("add.u64 ").unwrap();
        let ops: Vec<&str> = l.split_once("add.u64 ").unwrap().1.trim_end_matches(';').split(',').map(str::trim).collect();
        format!("{head}add.u64 {d}, {a}, {c}; add.u64 {d}, {d}, 1;", d = ops[0], a = ops[1], c = ops[2])
    });
    let k = Kernels { partial: wider, ..kernels() };
    assert!(f64_caught(&k), "a partial row stride of cols + 1 passed");
    let f = &base.fin;
    let at = f.lines().filter(|l| l.contains("mov.u64 ")).position(|l| l.ends_with(", 8;")).expect("the warp count");
    let k = Kernels { fin: edit(f, "mov.u64 ", at, |l| l.replace(", 8;", ", 16;")), ..kernels() };
    assert!(f64_caught(&k), "a chunk stride of 16 passed");
}

// ---------------------------------------------------------------------------
// Mutations of every kernel
// ---------------------------------------------------------------------------

/// Whether `mutate` applied to `op`'s PTX is told apart from the kernel on
/// any of its cases, under the f64 check or the documented order (or
/// faults).
fn caught(op: RmsNormDgammaOp, mutate: impl Fn(&str) -> String) -> bool {
    let p = mutate(&kir_ptx(op));
    match op {
        RmsNormDgammaOp::RinvRows => RINV_SHAPES.iter().enumerate().any(|(n, &(rows, cols))| {
            let x = activations(rows * cols, n as u64 + 1);
            ORDERS.iter().any(|&order| match run_rinv(&p, &x, rows, cols, order, false) {
                Ok(got) => got != model_rinv(&x, rows, cols) || check_rinv_f64(&got, &x, rows, cols).is_err(),
                Err(_) => true,
            })
        }),
        RmsNormDgammaOp::Partial => partial_cases().into_iter().enumerate().any(|(n, (rows, cols, chunk_rows))| {
            let (dy, x, rinv) = (gradients(rows * cols, n as u64 + 11), activations(rows * cols, n as u64 + 21), rinvs(rows, n as u64 + 31));
            let c = PartialCase { dy: &dy, x: &x, rinv: &rinv, rows, cols, chunk_rows };
            match run_partial(&p, &c, Order::Ascending, false) {
                Ok(got) => got != model_partial(&c) || check_partial_f64(&got, &c).is_err(),
                Err(_) => true,
            }
        }),
        RmsNormDgammaOp::Final => FINAL_CASES.iter().enumerate().any(|(n, &(chunks, cols))| {
            let partial = values(chunks * cols, n as u64 + 41, -24, -14);
            ORDERS.iter().any(|&order| match run_final(&p, &partial, chunks, cols, order, false) {
                Ok(got) => got != model_final(&partial, chunks, cols) || check_final_f64(&got, &partial, chunks, cols).is_err(),
                Err(_) => true,
            })
        }),
    }
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for op in RmsNormDgammaOp::ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

/// Every bound: kernel 1's row and column tests, kernel 2's column, chunk
/// and row tests, kernel 3's chunk test, warp-0 test and column test.
#[test]
fn relaxing_a_bound_is_caught() {
    for op in RmsNormDgammaOp::ALL {
        let p = kir_ptx(op);
        let sites: Vec<&str> = p.lines().filter(|l| l.contains("setp.ge.")).collect();
        let want = match op {
            RmsNormDgammaOp::RinvRows => 2,
            RmsNormDgammaOp::Partial => 3,
            RmsNormDgammaOp::Final => 3,
        };
        assert_eq!(sites.len(), want, "{op:?}\n{p}");
        for (i, site) in sites.iter().enumerate() {
            // Kernel 3's `tid >= 32`: relaxed, thread 32 (lane 0 of warp 1)
            // also stores, but the same column and the same tree, so the
            // same bits: an equivalent mutant. Its threshold is pinned below.
            if op == RmsNormDgammaOp::Final && site.contains(".u32 ") {
                continue;
            }
            assert!(caught(op, |p| nudge(p, "setp.ge.", "setp.gt.", i)), "{op:?} bound {i}: {site}");
        }
    }
    // Kernel 3's live-column test, which picks a dead lane's empty range.
    assert!(caught(RmsNormDgammaOp::Final, |p| nudge(p, "setp.lt.u64 ", "setp.le.u64 ", 0)), "the live-column select");
    // Kernel 3's warp-0 threshold as 0: nothing stores.
    let f = kir_ptx(RmsNormDgammaOp::Final);
    let at = f.lines().filter(|l| l.contains("mov.u32 ")).position(|l| l.ends_with(", 32;")).expect("the warp-0 threshold");
    assert!(caught(RmsNormDgammaOp::Final, |p| edit(p, "mov.u32 ", at, |l| l.replace(", 32;", ", 0;"))), "the warp-0 test");
}

#[test]
fn every_block_index_is_caught() {
    for (op, sites) in [(RmsNormDgammaOp::RinvRows, &["%ctaid.x"][..]), (RmsNormDgammaOp::Partial, &["%ctaid.x", "%ctaid.y"]), (RmsNormDgammaOp::Final, &["%ctaid.x"])] {
        for s in sites {
            assert!(caught(op, |p| p.replacen(s, "0", 1)), "{op:?} {s}");
        }
    }
}

/// Kernel 1's tree reads `swarp[0]` first: an address whose offset is the
/// constant 0, so scaling the offset or dropping the add of the base to it
/// leaves the address unchanged (equivalent mutants). The lines of that
/// address: its `mul.lo` and its `add`.
fn equivalent_address_lines(op: RmsNormDgammaOp, ptx: &str) -> [usize; 2] {
    if op != RmsNormDgammaOp::RinvRows {
        return [usize::MAX; 2];
    }
    let lines: Vec<&str> = ptx.lines().collect();
    let ld = lines.iter().position(|l| l.contains("ld.shared.f32")).expect("the tree's first load");
    assert!(lines[ld - 2].contains("mul.lo.u64") && lines[ld - 1].contains("add.u64 "), "{}\n{}", lines[ld - 2], lines[ld - 1]);
    [ld - 2, ld - 1]
}

/// Every address's element size: loads, stores, shared and global.
#[test]
fn nudging_an_element_size_is_caught() {
    for op in RmsNormDgammaOp::ALL {
        let p = kir_ptx(op);
        let skip = equivalent_address_lines(op, &p);
        let sites: Vec<usize> =
            p.lines().enumerate().filter(|(n, l)| l.contains("mul.lo.u64") && l.ends_with(", 4;") && !skip.contains(n)).map(|(n, _)| n).collect();
        assert!(!sites.is_empty(), "{op:?}");
        for at in sites {
            let mutate = |p: &str| p.lines().enumerate().map(|(n, l)| if n == at { l.replacen(", 4;", ", 8;", 1) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n";
            assert!(caught(op, mutate), "{op:?} element size at line {at}");
        }
    }
}

/// Every 64-bit add and every 64-bit multiply that is not an element size:
/// the bases, the strides, the steps, the chunk bounds.
#[test]
fn dropping_an_add_or_a_multiply_is_caught() {
    for op in RmsNormDgammaOp::ALL {
        let p = kir_ptx(op);
        let skip = equivalent_address_lines(op, &p);
        let adds: Vec<usize> = p.lines().enumerate().filter(|(_, l)| l.contains("add.u64 ")).map(|(n, _)| n).collect();
        for (i, line) in adds.into_iter().enumerate() {
            if skip.contains(&line) {
                continue;
            }
            // An address add (base + offset) dropped leaves the base: a
            // read of the wrong element. A counter's add dropped never ends
            // the loop: the interpreter's step limit.
            assert!(caught(op, |p| drop_op(p, "add.u64 ", "u64", i)), "{op:?} add {i}");
        }
        let muls: Vec<usize> = p.lines().filter(|l| l.contains("mul.lo.u64")).enumerate().filter(|(_, l)| !l.ends_with(", 4;")).map(|(n, _)| n).collect();
        assert!(!muls.is_empty(), "{op:?}");
        for i in muls {
            assert!(caught(op, |p| drop_op(p, "mul.lo.u64 ", "u64", i)), "{op:?} multiply {i}");
        }
    }
}

/// Every float operation: kernel 1's square-accumulate, butterfly and tree
/// adds, mean, `+ eps`, square root and reciprocal; kernel 2's products and
/// accumulate; kernel 3's accumulate and tree.
#[test]
fn every_float_operation_is_caught() {
    for op in RmsNormDgammaOp::ALL {
        let p = kir_ptx(op);
        for i in 0..p.matches("add.rn.f32 ").count() {
            assert!(caught(op, |p| nudge(p, "add.rn.f32 ", "sub.rn.f32 ", i)), "{op:?} add {i}");
        }
        for i in 0..p.matches("mul.rn.f32 ").count() {
            assert!(caught(op, |p| nudge(p, "mul.rn.f32 ", "add.rn.f32 ", i)), "{op:?} product {i}");
        }
    }
    let op = RmsNormDgammaOp::RinvRows;
    assert!(caught(op, |p| edit(p, "fma.rn.f32 ", 0, |l| {
        let (head, ops) = l.split_once("fma.rn.f32 ").expect("fma");
        let ops: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
        format!("{head}add.rn.f32 {}, {}, {};", ops[0], ops[1], ops[3])
    })), "sum of values, not squares");
    assert!(caught(op, |p| nudge(p, "div.rn.f32 ", "mul.rn.f32 ", 0)), "the mean");
    assert!(caught(op, |p| drop_op(p, "sqrt.rn.f32 ", "f32", 0)), "the square root");
    assert!(caught(op, |p| p.replacen("0f3F800000;", "0f40000000;", 1)), "the reciprocal's 1");
    assert!(caught(op, |p| nudge(p, "div.rn.f32 ", "mul.rn.f32 ", 1)), "the reciprocal");
}

/// The butterfly's lane masks (each the `mov.u32` just before its
/// `shfl.sync.bfly`) and the accumulators' starts.
#[test]
fn the_butterfly_and_the_starts_are_pinned() {
    let op = RmsNormDgammaOp::RinvRows;
    let p = kir_ptx(op);
    let lines: Vec<&str> = p.lines().collect();
    let shfl: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.contains("shfl.sync.bfly")).map(|(n, _)| n).collect();
    assert_eq!(shfl.len(), 5);
    for (&at, d) in shfl.iter().zip([16, 8, 4, 2, 1]) {
        let mask = at - 1;
        assert!(lines[mask].contains("mov.u32 ") && lines[mask].ends_with(&format!(", {d};")), "{}", lines[mask]);
        let to = if d == 16 { 8 } else { d * 2 };
        let mutate = |p: &str| {
            p.lines().enumerate().map(|(n, l)| if n == mask { l.replace(&format!(", {d};"), &format!(", {to};")) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
        };
        assert!(caught(op, mutate), "butterfly {d} -> {to}");
    }
    for op in RmsNormDgammaOp::ALL {
        assert!(caught(op, |p| p.replacen("0f00000000;", "0f3F800000;", 1)), "{op:?}: the accumulator starts at +0.0");
    }
}

/// The barriers: without one, a warp reads its neighbours' slots before
/// they are written (poison) on the warp-at-a-time schedules.
#[test]
fn dropping_a_barrier_is_caught() {
    for op in [RmsNormDgammaOp::RinvRows, RmsNormDgammaOp::Final] {
        assert!(caught(op, |p| p.replacen("bar.sync 0;", "", 1)), "{op:?}");
    }
}

/// The blocks are the ones the module names, and the final pass's block is
/// its 32 columns by 8 warps.
#[test]
fn the_block_shapes_are_consistent() {
    assert_eq!(FINAL_BLOCK, FINAL_COLS * WARPS);
    assert_eq!(RINV_BLOCK, 32 * WARPS);
    for op in RmsNormDgammaOp::ALL {
        assert!(kir_ptx(op).contains(&format!(".maxntid {}, 1, 1", op.block())), "{op:?}");
    }
}
