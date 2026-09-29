//! The differential equivalence gate for `nsl_softmax_f32` and
//! `nsl_log_softmax_f32` from `nsl_runtime::cuda::fused_kernels`
//! (new-roadmap item 5), now built by `nsl_kir::kernels::softmax`.
//!
//! This file runs the frozen hand modules (`tests/fixtures/softmax_hand.rs`)
//! and the KIR ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), as the runtime launches them (one
//! block of 256 threads per row), plus one block past the last row, under all
//! four thread schedules:
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory. The rows are empty, one column, and ragged widths longer than
//!    the block. The data is order-sensitive with full 24-bit significands,
//!    and includes masked (`-inf`) columns, a fully masked row, a NaN, `+inf`
//!    and signed zeros.
//! 2. **Correctness**: every row is the hand kernels' order restated in
//!    Rust, bit for bit. Thread `k` folds columns `k, k + 256, …`; thread 0
//!    then folds the partials `1 .. 256` into its own, in order, for the max
//!    and again for the sum of `2^((x - max) · log2 e)`. The softmax scales
//!    each exponential by `1 / sum`; the log-softmax subtracts `lg2(sum) ·
//!    ln 2` from `x - max`. Nothing past the last row is written and the
//!    input is untouched.
//! 3. **The gate bites**:
//!    - the row bound, each column loop's bound and stride, each fold's
//!      start, bound and step, the thread-0 test, and all four barriers;
//!    - the row base, the sum region's offset, every element size and every
//!      64-bit add;
//!    - every combine, the exponential's scale and `ex2`, the finish (`rcp`,
//!      or `lg2` and `ln 2`), the exponentials' store, and each identity.
//!
//!    Named equivalent mutants: the max is idempotent, so a max fold that
//!    starts on thread 0's own partial and a max loop that strides by 128
//!    change nothing; the log-softmax's last pass is a pure map, so it too
//!    may stride by 128; the sum's `+0` identity may be `-0`, because every
//!    exponential is `+0` or more; and the element size of the fold's final
//!    store, which is `sm[0]`, has no multiply to change.

use std::collections::HashMap;

use nsl_kir::kernels::softmax::{ptx, SoftmaxOp, SOFTMAX_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/softmax_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 64;
const POISON: u32 = 0x7FA5_A5A5;
const INP: u64 = 0x1000_0000;
const OUT: u64 = 0x2000_0000;
const B: usize = SOFTMAX_BLOCK as usize;
const SHARED: usize = 2 * B * 4;
const LOG2_E: f32 = f32::from_bits(0x3FB8_AA3B);
const LN_2: f32 = f32::from_bits(0x3F31_7218);

fn kir_ptx(op: SoftmaxOp) -> String {
    String::from_utf8(ptx(op)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(op: SoftmaxOp) -> String {
    match op {
        SoftmaxOp::Softmax => hand::SOFTMAX_F32_PTX,
        SoftmaxOp::LogSoftmax => hand::LOG_SOFTMAX_F32_PTX,
    }
    .trim_end_matches('\0')
    .to_string()
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

/// Logits whose exponentials' sums depend on the order of the adds: full
/// 24-bit significands, magnitudes from `2^-6` to `2^2` (so no few
/// exponentials swamp the rest), both signs, and some signed zeros.
fn data(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| match lcg(&mut s) % 40 {
            0 => 0.0f32.to_bits(),
            1 => (-0.0f32).to_bits(),
            _ => {
                let e = (lcg(&mut s) % 9) as i32 - 6;
                let mant = 1.0 + (lcg(&mut s) % (1 << 23)) as f32 / (1u32 << 23) as f32;
                let sign = if lcg(&mut s).is_multiple_of(2) { 1.0 } else { -1.0 };
                (sign * mant * 2f32.powi(e)).to_bits()
            }
        })
        .collect()
}

/// `rows` rows of `cols` columns, row-major.
struct Case {
    rows: usize,
    cols: usize,
    x: Vec<u32>,
}

const WIDTHS: [usize; 6] = [0, 1, 255, 256, 257, 700];

fn cases() -> Vec<Case> {
    let mut v: Vec<Case> =
        WIDTHS.iter().enumerate().map(|(i, &cols)| Case { rows: 2, cols, x: data(2 * cols, 3 + i as u64) }).collect();
    // No rows: only the block past the end runs.
    v.push(Case { rows: 0, cols: 300, x: vec![] });
    // Special rows of 300 columns: masked columns, a fully masked row, a
    // NaN, `+inf`, every value negative (the max's `-inf` start shows), and
    // one spike on the stride's second trip.
    let cols = 300;
    let mut rows: Vec<Vec<u32>> = Vec::new();
    let mut masked = data(cols, 90);
    for i in (0..cols).step_by(7) {
        masked[i] = f32::NEG_INFINITY.to_bits();
    }
    rows.push(masked);
    rows.push(vec![f32::NEG_INFINITY.to_bits(); cols]);
    let mut nan = data(cols, 91);
    nan[123] = f32::NAN.to_bits();
    rows.push(nan);
    let mut inf = data(cols, 92);
    inf[200] = f32::INFINITY.to_bits();
    rows.push(inf);
    rows.push(data(cols, 93).into_iter().map(|w| (-f32::from_bits(w).abs() - 1.0).to_bits()).collect());
    let mut spike = data(cols, 94);
    spike[299] = 99.0f32.to_bits();
    rows.push(spike);
    v.push(Case { rows: rows.len(), cols, x: rows.concat() });
    v
}

/// All of global memory after the launch: `[inp, out + tail]`.
fn run(ptx: &str, c: &Case, order: Order) -> Vec<u32> {
    let prog = parse(ptx);
    let mut global = vec![
        Segment { base: INP, bytes: le32(&c.x) },
        Segment { base: OUT, bytes: le32(&vec![POISON; c.rows * c.cols + TAIL]) },
    ];
    let args: HashMap<String, u64> = SoftmaxOp::PARAMS
        .iter()
        .zip([INP, OUT, c.rows as u64, c.cols as u64])
        .map(|(p, v)| (p.to_string(), v))
        .collect();
    for row in 0..=c.rows {
        let mut l = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![0; SHARED],
            ctaid: row as u32,
            ctaid_y: 0,
            nctaid_x: 0,
            nctaid_y: 1,
            ntid: SOFTMAX_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    global.iter().flat_map(|s| words(&s.bytes)).collect()
}

const ORDERS: [Order; 4] = [Order::Ascending, Order::Descending, Order::WarpsAscending, Order::WarpsDescending];

#[test]
fn the_kernels_agree() {
    for op in SoftmaxOp::ALL {
        let (hand, kir) = (hand_ptx(op), kir_ptx(op));
        for c in cases() {
            for order in ORDERS {
                assert!(run(&hand, &c, order) == run(&kir, &c, order), "{op:?} {}×{} {order:?}", c.rows, c.cols);
            }
        }
    }
}

/// Thread 0's in-order fold of the per-thread partials `fold(k, …)`.
fn block_fold(partial: impl Fn(usize) -> f32, combine: impl Fn(f32, f32) -> f32) -> f32 {
    (1..B).fold(partial(0), |acc, k| combine(acc, partial(k)))
}

/// The hand kernels' order, restated, for one row.
fn reference(op: SoftmaxOp, row: &[u32]) -> Vec<u32> {
    reference_with(op, row, false)
}

/// [`reference`], or with thread 0 folding the partial sums in reverse
/// (`reversed`).
fn reference_with(op: SoftmaxOp, row: &[u32], reversed: bool) -> Vec<u32> {
    let x: Vec<f32> = row.iter().map(|&w| f32::from_bits(w)).collect();
    let mine = |k: usize| (k..x.len()).step_by(B);
    let mx = block_fold(|k| mine(k).fold(f32::NEG_INFINITY, |m, i| m.max(x[i])), f32::max);
    let e: Vec<f32> = x.iter().map(|&v| ((v - mx) * LOG2_E).exp2()).collect();
    let partial = |k: usize| mine(k).fold(0.0f32, |s, i| s + e[i]);
    let sum = if reversed {
        (1..B).rev().fold(partial(0), |acc, k| acc + partial(k))
    } else {
        block_fold(partial, |a, b| a + b)
    };
    match op {
        SoftmaxOp::Softmax => {
            let r = 1.0 / sum;
            e.iter().map(|&v| (v * r).to_bits()).collect()
        }
        SoftmaxOp::LogSoftmax => {
            let l = sum.log2() * LN_2;
            x.iter().map(|&v| ((v - mx) - l).to_bits()).collect()
        }
    }
}

fn same(a: u32, b: u32) -> bool {
    a == b || (f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan())
}

#[test]
fn the_kernels_are_the_row_softmax() {
    for op in SoftmaxOp::ALL {
        for which in [hand_ptx(op), kir_ptx(op)] {
            for c in cases() {
                let got = run(&which, &c, Order::Ascending);
                let (back, out) = got.split_at(c.x.len());
                assert_eq!(back, &c.x[..], "{op:?} {}×{}: input untouched", c.rows, c.cols);
                for r in 0..c.rows {
                    let row = &c.x[r * c.cols..(r + 1) * c.cols];
                    let want = reference(op, row);
                    let got = &out[r * c.cols..(r + 1) * c.cols];
                    assert!(got.iter().zip(&want).all(|(g, w)| same(*g, *w)), "{op:?} {}×{} row {r}", c.rows, c.cols);
                }
                assert!(out[c.rows * c.cols..].iter().all(|&w| w == POISON), "{op:?} {}×{}: wrote past out", c.rows, c.cols);
            }
        }
    }
}

/// The data reaches the cases the kernels distinguish: several trips of the
/// stride, sums that depend on their order, a fully masked row, a NaN and
/// `+inf`.
#[test]
fn the_data_covers_every_case() {
    assert!(WIDTHS.iter().any(|&n| n > 2 * B) && WIDTHS.contains(&0));
    let order_shows = cases().iter().any(|c| {
        (0..c.rows).any(|r| {
            let row = &c.x[r * c.cols..(r + 1) * c.cols];
            reference_with(SoftmaxOp::Softmax, row, true) != reference_with(SoftmaxOp::Softmax, row, false)
        })
    });
    assert!(order_shows, "the sum depends on order");
    let all: Vec<f32> = cases().iter().flat_map(|c| c.x.clone()).map(f32::from_bits).collect();
    assert!(all.iter().any(|v| v.is_nan()) && all.contains(&f32::INFINITY) && all.contains(&f32::NEG_INFINITY));
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on any case or
/// schedule (or faults).
fn caught(op: SoftmaxOp, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(op);
    let mutant = mutate(&kir_ptx(op));
    cases().iter().any(|c| {
        ORDERS.into_iter().any(|order| {
            let expect = run(&hand, c, order);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mutant, c, order))) {
                Ok(r) => r != expect,
                Err(_) => true,
            }
        })
    })
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

/// `op d, a, b;` as `mov.<ty> d, a;`: the operation dropped.
fn drop_op(ptx: &str, op: &str, ty: &str, i: usize) -> String {
    edit(ptx, op, i, |l| {
        let (head, operands) = l.split_once(op).expect("the op");
        let mut ops: Vec<&str> = operands.trim_end_matches(';').split(',').map(str::trim).collect();
        ops.truncate(2);
        format!("{head}mov.{ty} {};", ops.join(", "))
    })
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for op in SoftmaxOp::ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

/// The row bound and each column loop's bound and stride. Striding by 512
/// skips columns in every pass. Striding by 128 revisits them: the max is
/// idempotent and the log-softmax's last pass is a pure map, so there it is
/// a named equivalent mutant.
#[test]
fn the_row_and_column_loops_are_pinned() {
    for op in SoftmaxOp::ALL {
        assert_eq!(kir_ptx(op).matches("setp.ge.u64 ").count(), 4, "{op:?}");
        for i in 0..4 {
            assert!(caught(op, |p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "{op:?} bound {i}");
        }
        assert_eq!(kir_ptx(op).matches(", 256;").count(), 3, "{op:?}");
        for pass in 0..3 {
            assert!(caught(op, |p| nudge(p, ", 256;", ", 512;", pass)), "{op:?} stride {pass} doubled");
            let equivalent = pass == 0 || (pass == 2 && op == SoftmaxOp::LogSoftmax);
            assert_eq!(caught(op, |p| nudge(p, ", 256;", ", 128;", pass)), !equivalent, "{op:?} stride {pass} halved");
        }
    }
}

/// Each fold's start, bound and step, the thread-0 test, and all four
/// barriers. The max fold starting on thread 0's own partial is a named
/// equivalent mutant: the max is idempotent.
#[test]
fn the_folds_and_barriers_are_pinned() {
    for op in SoftmaxOp::ALL {
        let p = kir_ptx(op);
        let starts: Vec<usize> =
            p.lines().enumerate().filter(|(_, l)| l.trim().starts_with("mov.u32 ") && l.ends_with(", 1;")).map(|(n, _)| n).collect();
        assert_eq!(starts.len(), 4, "{op:?}: each fold's start and step");
        // Each fold's start (as 0) then its step (as 2). The first start is
        // the max fold's.
        for (j, &at) in starts.iter().enumerate() {
            let to = if j % 2 == 0 { ", 0;" } else { ", 2;" };
            let mutate = |p: &str| {
                p.lines().enumerate().map(|(n, l)| if n == at { l.replacen(", 1;", to, 1) } else { l.to_string() }).collect::<Vec<_>>().join("\n")
                    + "\n"
            };
            assert_eq!(caught(op, mutate), j != 0, "{op:?} fold constant {j}");
        }
        assert_eq!(p.matches("setp.ge.u32 ").count(), 2, "{op:?}");
        for i in 0..2 {
            assert!(caught(op, |p| nudge(p, "setp.ge.u32 ", "setp.gt.u32 ", i)), "{op:?} fold bound {i}");
            assert!(caught(op, |p| nudge(p, "setp.ne.u32 ", "setp.eq.u32 ", i)), "{op:?} thread-0 test {i}");
        }
        assert_eq!(p.matches("bar.sync 0;").count(), 4, "{op:?}");
        for i in 0..4 {
            assert!(caught(op, |p| edit(p, "bar.sync 0;", i, |_| String::new())), "{op:?} barrier {i}");
        }
    }
}

/// The row base, the sum region's offset, every element size and every
/// 64-bit add.
#[test]
fn every_address_is_pinned() {
    for op in SoftmaxOp::ALL {
        let p = kir_ptx(op);
        assert!(caught(op, |p| drop_op(p, "mul.lo.u64 ", "u64", 0)), "{op:?} row base");
        assert!(caught(op, |p| p.replacen(", 1024;", ", 0;", 1)), "{op:?} sum region");
        let sizes = p.lines().filter(|l| l.contains("mul.lo.u64") && l.ends_with(", 4;")).count();
        for i in 0..sizes {
            assert!(caught(op, |p| nudge(p, ", 4;", ", 8;", i)), "{op:?} element size {i}");
        }
        // Every 64-bit add but the column loops' steps, whose stride is
        // pinned above (dropped, the loop would not end).
        let lines: Vec<&str> = p.lines().collect();
        let adds: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.contains("add.u64 ")).map(|(n, _)| n).collect();
        let steps = adds.iter().filter(|&&n| lines[n - 1].ends_with(", 256;")).count();
        assert_eq!(steps, 3, "{op:?}");
        for (i, &n) in adds.iter().enumerate() {
            if !lines[n - 1].ends_with(", 256;") {
                assert!(caught(op, |p| drop_op(p, "add.u64 ", "u64", i)), "{op:?} add {i}");
            }
        }
    }
}

/// Every combine, the exponential's scale and `ex2`, the finish, the
/// softmax's store of the exponentials, and each identity. The sum's `+0`
/// as `-0` is a named equivalent mutant: every exponential is `+0` or more.
#[test]
fn the_arithmetic_is_pinned() {
    for op in SoftmaxOp::ALL {
        let p = kir_ptx(op);
        for (from, to) in [("max.f32 ", "min.f32 "), ("add.rn.f32 ", "sub.rn.f32 "), ("sub.rn.f32 ", "add.rn.f32 "), ("mul.rn.f32 ", "add.rn.f32 ")] {
            let n = p.matches(from).count();
            assert!(n > 0, "{op:?} {from}");
            for i in 0..n {
                assert!(caught(op, |p| nudge(p, from, to, i)), "{op:?} {from} {i}");
            }
        }
        for (ins, ty) in [("mul.rn.f32 ", "f32"), ("sub.rn.f32 ", "f32")] {
            for i in 0..p.matches(ins).count() {
                assert!(caught(op, |p| drop_op(p, ins, ty, i)), "{op:?} {ins} {i} dropped");
            }
        }
        assert!(caught(op, |p| nudge(p, "0f3FB8AA3B", "0f3F800000", 0)), "{op:?} log2 e");
        assert!(caught(op, |p| edit(p, "ex2.approx.f32 ", 0, |l| l.replacen("ex2.approx.f32", "mov.f32", 1))), "{op:?} ex2");
        match op {
            SoftmaxOp::Softmax => {
                assert!(caught(op, |p| edit(p, "rcp.approx.f32 ", 0, |l| l.replacen("rcp.approx.f32", "mov.f32", 1))), "rcp");
                assert_eq!(p.matches("st.global.f32").count(), 2);
                assert!(caught(op, |p| edit(p, "st.global.f32", 0, |_| String::new())), "the exponentials' store");
            }
            SoftmaxOp::LogSoftmax => {
                assert!(caught(op, |p| edit(p, "lg2.approx.f32 ", 0, |l| l.replacen("lg2.approx.f32", "mov.f32", 1))), "lg2");
                assert!(caught(op, |p| nudge(p, "0f3F317218", "0f3F800000", 0)), "ln 2");
                assert_eq!(p.matches("st.global.f32").count(), 1);
            }
        }
        assert!(caught(op, |p| nudge(p, "0fFF800000", "0f00000000", 0)), "{op:?} max identity");
        assert!(caught(op, |p| nudge(p, "0f00000000", "0f3F800000", 0)), "{op:?} sum identity as 1");
        assert!(!caught(op, |p| nudge(p, "0f00000000", "0f80000000", 0)), "{op:?} sum identity as -0");
    }
}
