//! The differential equivalence gate for `nsl_layernorm_f32` and
//! `nsl_rmsnorm_f32` from `nsl_runtime::cuda::fused_kernels` (new-roadmap
//! item 5), now built by `nsl_kir::kernels::norm`.
//!
//! This file runs the frozen hand modules (`tests/fixtures/norm_hand.rs`)
//! and the KIR ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), as the runtime launches them (one
//! block of 256 threads per row), plus one block past the last row. Shared
//! memory starts as NaN poison, as uninitialised shared memory may.
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory. RMSNorm agrees under all four thread schedules; the hand
//!    LayerNorm only where thread 0 runs last (`Descending`), see 3.
//! 2. **Correctness**: every row is the hand kernels' order restated in
//!    Rust, bit for bit, for the KIR kernels under every schedule. Thread `k`
//!    folds columns `k, k + 256, …`; thread 0 then adds the partials `1 ..
//!    256` to its own, in order. The statistic is divided by `cols`, `eps`
//!    added, and `rsqrt` taken. Every multiply and add rounds on its own.
//!    Nothing past the last row is written, and the inputs are untouched.
//! 3. **The hand LayerNorm's race**: it reduces the mean and the variance
//!    through one shared region, and thread 0 stores its variance partial to
//!    the mean's slot with no barrier after the other threads read the mean.
//!    Under every schedule that lets thread 0 run ahead of another thread,
//!    that thread reads thread 0's partial as the mean. (On hardware a
//!    warp's lanes read together, so it takes a warp falling a pass behind
//!    warp 0; the PTX memory model allows it either way.) The KIR kernel
//!    gives the variance its own region; putting it back on the mean's is
//!    caught.
//! 4. **The gate bites**: the row bound, each column loop's bound and
//!    stride, each fold's start, bound and step, the thread-0 test, every
//!    barrier, the row base, the variance region's offset, every element
//!    size and 64-bit add, every add, subtract and multiply (flipped and
//!    dropped), the `div.approx`, the `rsqrt`, and each sum's identity.
//!    Named equivalent mutants: the last pass striding by 128 (it is a pure
//!    map), and each sum's `+0` identity as `-0` (a sum of squares is never
//!    `-0`, and the data holds no row whose values are all `-0`).

use std::collections::HashMap;

use nsl_kir::kernels::norm::{ptx, NormOp, NORM_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/norm_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 64;
const POISON: u32 = 0x7FA5_A5A5;
const INP: u64 = 0x1000_0000;
const OUT: u64 = 0x2000_0000;
const GAMMA: u64 = 0x3000_0000;
const BETA: u64 = 0x4000_0000;
const B: usize = NORM_BLOCK as usize;
const SHARED: usize = 2 * B * 4;

fn kir_ptx(op: NormOp) -> String {
    String::from_utf8(ptx(op)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(op: NormOp) -> String {
    match op {
        NormOp::LayerNorm => hand::LAYERNORM_F32_PTX,
        NormOp::RmsNorm => hand::RMSNORM_F32_PTX,
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

/// Values whose sums depend on the order of the adds: full 24-bit
/// significands, magnitudes from `2^-6` to `2^2`, both signs, about
/// `offset`, and some signed zeros.
fn data(n: usize, seed: u64, offset: f32) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| match lcg(&mut s) % 40 {
            0 => 0.0f32.to_bits(),
            1 => (-0.0f32).to_bits(),
            _ => {
                let e = (lcg(&mut s) % 9) as i32 - 6;
                let mant = 1.0 + (lcg(&mut s) % (1 << 23)) as f32 / (1u32 << 23) as f32;
                let sign = if lcg(&mut s).is_multiple_of(2) { 1.0 } else { -1.0 };
                (offset + sign * mant * 2f32.powi(e)).to_bits()
            }
        })
        .collect()
}

/// `rows` rows of `cols` columns, row-major, with the per-column `gamma`
/// and `beta` and the launch's `eps`.
struct Case {
    rows: usize,
    cols: usize,
    x: Vec<u32>,
    gamma: Vec<u32>,
    beta: Vec<u32>,
    eps: f32,
}

fn case(rows: Vec<Vec<u32>>, cols: usize, seed: u64, eps: f32) -> Case {
    Case { rows: rows.len(), cols, x: rows.concat(), gamma: data(cols, seed ^ 0x55, 1.0), beta: data(cols, seed ^ 0xAA, 0.0), eps }
}

const WIDTHS: [usize; 6] = [0, 1, 255, 256, 257, 700];

fn cases() -> Vec<Case> {
    let mut v: Vec<Case> = WIDTHS
        .iter()
        .enumerate()
        .map(|(i, &cols)| {
            let seed = 3 + i as u64;
            case(vec![data(cols, seed, 1.5), data(cols, seed + 100, -0.25)], cols, seed, 1e-5)
        })
        .collect();
    // No rows: only the block past the end runs.
    v.push(Case { rows: 0, cols: 300, x: vec![], gamma: data(300, 1, 1.0), beta: data(300, 2, 0.0), eps: 1e-5 });
    // Rows of 300 columns: constant (the variance is 0, so `eps` shows), a
    // NaN, `+inf`, squares that overflow, and one spike on the stride's
    // second trip.
    let cols = 300;
    let mut nan = data(cols, 91, 0.5);
    nan[123] = f32::NAN.to_bits();
    let mut inf = data(cols, 92, 0.5);
    inf[200] = f32::INFINITY.to_bits();
    let mut spike = data(cols, 94, 0.0);
    spike[299] = 40.0f32.to_bits();
    let rows = vec![
        vec![2.5f32.to_bits(); cols],
        nan,
        inf,
        data(cols, 93, 0.0).into_iter().map(|w| (f32::from_bits(w) * 1e19).to_bits()).collect(),
        spike,
    ];
    v.push(case(rows, cols, 90, 1e-5));
    // `eps` 0: a zero row's statistic is 0, and `rsqrt(0)` is `inf`.
    v.push(case(vec![vec![0; 40], data(40, 95, 0.0)], 40, 95, 0.0));
    v
}

/// All of global memory after the launch: `[inp, out + tail, gamma, beta]`.
fn run(op: NormOp, ptx: &str, c: &Case, order: Order) -> Vec<u32> {
    let prog = parse(ptx);
    let mut global = vec![
        Segment { base: INP, bytes: le32(&c.x) },
        Segment { base: OUT, bytes: le32(&vec![POISON; c.rows * c.cols + TAIL]) },
        Segment { base: GAMMA, bytes: le32(&c.gamma) },
        Segment { base: BETA, bytes: le32(&c.beta) },
    ];
    let values: Vec<u64> = match op {
        NormOp::LayerNorm => vec![INP, OUT, GAMMA, BETA, c.rows as u64, c.cols as u64, u64::from(c.eps.to_bits())],
        NormOp::RmsNorm => vec![INP, OUT, GAMMA, c.rows as u64, c.cols as u64, u64::from(c.eps.to_bits())],
    };
    let args: HashMap<String, u64> = op.params().iter().zip(values).map(|(p, v)| (p.to_string(), v)).collect();
    for row in 0..=c.rows {
        let mut l = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: le32(&vec![POISON; SHARED / 4]),
            ctaid: row as u32,
            ctaid_y: 0,
            nctaid_x: 0,
            nctaid_y: 1,
            ntid: NORM_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    global.iter().flat_map(|s| words(&s.bytes)).collect()
}

const ORDERS: [Order; 4] = [Order::Ascending, Order::Descending, Order::WarpsAscending, Order::WarpsDescending];

/// The schedules under which the hand kernel is race-free: all four for
/// RMSNorm; for LayerNorm, only the one that runs thread 0 last.
fn hand_orders(op: NormOp) -> Vec<Order> {
    match op {
        NormOp::RmsNorm => ORDERS.to_vec(),
        NormOp::LayerNorm => vec![Order::Descending],
    }
}

#[test]
fn the_kernels_agree() {
    for op in NormOp::ALL {
        let (hand, kir) = (hand_ptx(op), kir_ptx(op));
        for c in cases() {
            for order in hand_orders(op) {
                assert!(run(op, &hand, &c, order) == run(op, &kir, &c, order), "{op:?} {}×{} {order:?}", c.rows, c.cols);
            }
        }
    }
}

/// Thread 0's in-order sum of the per-thread partials.
fn block_sum(partial: impl Fn(usize) -> f32, reversed: bool) -> f32 {
    if reversed {
        (1..B).rev().fold(partial(0), |acc, k| acc + partial(k))
    } else {
        (1..B).fold(partial(0), |acc, k| acc + partial(k))
    }
}

/// The hand kernels' order, restated, for one row.
fn reference(op: NormOp, c: &Case, r: usize) -> Vec<u32> {
    reference_with(op, c, r, false)
}

/// [`reference`], or with thread 0 adding the partials in reverse
/// (`reversed`).
fn reference_with(op: NormOp, c: &Case, r: usize, reversed: bool) -> Vec<u32> {
    let f = |w: &u32| f32::from_bits(*w);
    let x: Vec<f32> = c.x[r * c.cols..(r + 1) * c.cols].iter().map(f).collect();
    let (g, bt): (Vec<f32>, Vec<f32>) = (c.gamma.iter().map(f).collect(), c.beta.iter().map(f).collect());
    let n = c.cols as f32;
    let mine = |k: usize| (k..x.len()).step_by(B);
    let inv_std = |total: f32| 1.0 / (total / n + c.eps).sqrt();
    match op {
        NormOp::LayerNorm => {
            let mean = block_sum(|k| mine(k).fold(0.0f32, |s, i| s + x[i]), reversed) / n;
            let q = block_sum(
                |k| {
                    mine(k).fold(0.0f32, |s, i| {
                        let d = x[i] - mean;
                        s + d * d
                    })
                },
                reversed,
            );
            let inv = inv_std(q);
            (0..x.len()).map(|i| (((x[i] - mean) * inv) * g[i] + bt[i]).to_bits()).collect()
        }
        NormOp::RmsNorm => {
            let inv = inv_std(block_sum(|k| mine(k).fold(0.0f32, |s, i| s + x[i] * x[i]), reversed));
            (0..x.len()).map(|i| ((x[i] * inv) * g[i]).to_bits()).collect()
        }
    }
}

fn same(a: u32, b: u32) -> bool {
    a == b || (f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan())
}

/// `run`'s output matches the reference on every row, writes nothing past
/// the last row, and leaves the inputs alone.
fn is_reference(op: NormOp, c: &Case, got: &[u32]) -> bool {
    let (back, rest) = got.split_at(c.x.len());
    let (out, params) = rest.split_at(c.rows * c.cols + TAIL);
    let rows_ok = (0..c.rows).all(|r| {
        let got = &out[r * c.cols..(r + 1) * c.cols];
        got.iter().zip(reference(op, c, r)).all(|(g, w)| same(*g, w))
    });
    back == &c.x[..] && rows_ok && out[c.rows * c.cols..].iter().all(|&w| w == POISON) && params == [c.gamma.clone(), c.beta.clone()].concat()
}

#[test]
fn the_kir_kernels_are_the_reference_under_every_schedule() {
    for op in NormOp::ALL {
        let kir = kir_ptx(op);
        for c in cases() {
            for order in ORDERS {
                assert!(is_reference(op, &c, &run(op, &kir, &c, order)), "{op:?} {}×{} {order:?}", c.rows, c.cols);
            }
        }
    }
}

#[test]
fn the_hand_kernels_are_the_reference_where_they_are_race_free() {
    for op in NormOp::ALL {
        let hand = hand_ptx(op);
        for c in cases() {
            for order in hand_orders(op) {
                assert!(is_reference(op, &c, &run(op, &hand, &c, order)), "{op:?} {}×{} {order:?}", c.rows, c.cols);
            }
        }
    }
}

/// With thread 0 ahead, the hand LayerNorm's other threads read thread 0's
/// variance partial as the mean: every case with a column past thread 0's
/// goes wrong under the three schedules that let it run ahead.
#[test]
fn the_hand_layernorm_races_on_the_shared_slot() {
    let hand = hand_ptx(NormOp::LayerNorm);
    for order in [Order::Ascending, Order::WarpsAscending, Order::WarpsDescending] {
        for c in cases().iter().filter(|c| c.rows > 0 && c.cols > 1) {
            assert!(!is_reference(NormOp::LayerNorm, c, &run(NormOp::LayerNorm, &hand, c, order)), "{}×{} {order:?}", c.rows, c.cols);
        }
    }
}

/// The data reaches the cases the kernels distinguish: several trips of the
/// stride, sums that depend on their order, a NaN, `+inf`, squares that
/// overflow, a zero variance and a zero `eps`.
#[test]
fn the_data_covers_every_case() {
    assert!(WIDTHS.iter().any(|&n| n > 2 * B) && WIDTHS.contains(&0));
    for op in NormOp::ALL {
        let order_shows = cases()
            .iter()
            .any(|c| (0..c.rows).any(|r| reference_with(op, c, r, true) != reference_with(op, c, r, false)));
        assert!(order_shows, "{op:?}: the sums depend on order");
    }
    let all: Vec<f32> = cases().iter().flat_map(|c| c.x.clone()).map(f32::from_bits).collect();
    assert!(all.iter().any(|v| v.is_nan()) && all.contains(&f32::INFINITY));
    assert!(all.iter().any(|v| (v * v).is_infinite() && v.is_finite()));
    assert!(cases().iter().any(|c| c.eps == 0.0) && cases().iter().any(|c| c.x.len() >= c.cols && c.cols > 0 && c.x[..c.cols].iter().all(|&w| w == c.x[0])));
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the reference on any case or
/// schedule (or faults).
fn caught(op: NormOp, mutate: impl Fn(&str) -> String) -> bool {
    let mutant = mutate(&kir_ptx(op));
    cases().iter().any(|c| {
        ORDERS.into_iter().any(|order| {
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(op, &mutant, c, order))) {
                Ok(r) => !is_reference(op, c, &r),
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
    for op in NormOp::ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

/// The number of statistics, and so of folds and summing passes.
fn stats(op: NormOp) -> usize {
    match op {
        NormOp::LayerNorm => 2,
        NormOp::RmsNorm => 1,
    }
}

/// The row bound and each column loop's bound and stride. Striding by 512
/// skips columns in every pass. Striding by 128 revisits them: the summing
/// passes count them twice, but the last pass is a pure map, so there it
/// is a named equivalent mutant.
#[test]
fn the_row_and_column_loops_are_pinned() {
    for op in NormOp::ALL {
        let passes = stats(op) + 1;
        assert_eq!(kir_ptx(op).matches("setp.ge.u64 ").count(), 1 + passes, "{op:?}");
        for i in 0..=passes {
            assert!(caught(op, |p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "{op:?} bound {i}");
        }
        assert_eq!(kir_ptx(op).matches(", 256;").count(), passes, "{op:?}");
        for pass in 0..passes {
            assert!(caught(op, |p| nudge(p, ", 256;", ", 512;", pass)), "{op:?} stride {pass} doubled");
            assert_eq!(caught(op, |p| nudge(p, ", 256;", ", 128;", pass)), pass + 1 != passes, "{op:?} stride {pass} halved");
        }
    }
}

/// Each fold's start (as 0) and step (as 2), bound and thread-0 test, and
/// every barrier.
#[test]
fn the_folds_and_barriers_are_pinned() {
    for op in NormOp::ALL {
        let p = kir_ptx(op);
        let consts: Vec<usize> =
            p.lines().enumerate().filter(|(_, l)| l.trim().starts_with("mov.u32 ") && l.ends_with(", 1;")).map(|(n, _)| n).collect();
        assert_eq!(consts.len(), 2 * stats(op), "{op:?}: each fold's start and step");
        for (j, &at) in consts.iter().enumerate() {
            let to = if j % 2 == 0 { ", 0;" } else { ", 2;" };
            let mutate = |p: &str| {
                p.lines().enumerate().map(|(n, l)| if n == at { l.replacen(", 1;", to, 1) } else { l.to_string() }).collect::<Vec<_>>().join("\n")
                    + "\n"
            };
            assert!(caught(op, mutate), "{op:?} fold constant {j}");
        }
        assert_eq!(p.matches("setp.ge.u32 ").count(), stats(op), "{op:?}");
        for i in 0..stats(op) {
            assert!(caught(op, |p| nudge(p, "setp.ge.u32 ", "setp.gt.u32 ", i)), "{op:?} fold bound {i}");
            assert!(caught(op, |p| nudge(p, "setp.ne.u32 ", "setp.eq.u32 ", i)), "{op:?} thread-0 test {i}");
        }
        assert_eq!(p.matches("bar.sync 0;").count(), 2 * stats(op), "{op:?}");
        for i in 0..2 * stats(op) {
            assert!(caught(op, |p| edit(p, "bar.sync 0;", i, |_| String::new())), "{op:?} barrier {i}");
        }
    }
}

/// The row base, every element size and every 64-bit add but the column
/// loops' steps (their stride is pinned above; dropped, a loop would not
/// end). Putting LayerNorm's variance back on the mean's region brings the
/// hand kernel's race back, and is caught.
#[test]
fn every_address_is_pinned() {
    for op in NormOp::ALL {
        let p = kir_ptx(op);
        assert!(caught(op, |p| drop_op(p, "mul.lo.u64 ", "u64", 0)), "{op:?} row base");
        if op == NormOp::LayerNorm {
            assert!(caught(op, |p| p.replacen(", 1024;", ", 0;", 1)), "the variance region");
        }
        let sizes = p.lines().filter(|l| l.contains("mul.lo.u64") && l.ends_with(", 4;")).count();
        for i in 0..sizes {
            assert!(caught(op, |p| nudge(p, ", 4;", ", 8;", i)), "{op:?} element size {i}");
        }
        let lines: Vec<&str> = p.lines().collect();
        let adds: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.contains("add.u64 ")).map(|(n, _)| n).collect();
        let steps = adds.iter().filter(|&&n| lines[n - 1].ends_with(", 256;")).count();
        assert_eq!(steps, stats(op) + 1, "{op:?}");
        for (i, &n) in adds.iter().enumerate() {
            if !lines[n - 1].ends_with(", 256;") {
                assert!(caught(op, |p| drop_op(p, "add.u64 ", "u64", i)), "{op:?} add {i}");
            }
        }
    }
}

/// Every add, subtract and multiply, flipped and dropped; the `div.approx`
/// and the `rsqrt`; and each sum's identity. The identity as `-0` is a
/// named equivalent mutant.
#[test]
fn the_arithmetic_is_pinned() {
    for op in NormOp::ALL {
        let p = kir_ptx(op);
        for (from, to) in [("add.rn.f32 ", "sub.rn.f32 "), ("sub.rn.f32 ", "add.rn.f32 "), ("mul.rn.f32 ", "add.rn.f32 ")] {
            let n = p.matches(from).count();
            assert_eq!(n > 0, from != "sub.rn.f32 " || op == NormOp::LayerNorm, "{op:?} {from}");
            for i in 0..n {
                assert!(caught(op, |p| nudge(p, from, to, i)), "{op:?} {from} {i}");
                assert!(caught(op, |p| drop_op(p, from, "f32", i)), "{op:?} {from} {i} dropped");
            }
        }
        for i in 0..stats(op) {
            assert!(caught(op, |p| drop_op(p, "div.approx.f32 ", "f32", i)), "{op:?} div {i}");
        }
        assert!(caught(op, |p| edit(p, "rsqrt.approx.f32 ", 0, |l| l.replacen("rsqrt.approx.f32", "mov.f32", 1))), "{op:?} rsqrt");
        assert_eq!(p.matches("0f00000000").count(), stats(op), "{op:?}");
        for i in 0..stats(op) {
            assert!(caught(op, |p| nudge(p, "0f00000000", "0f3F800000", i)), "{op:?} sum identity {i} as 1");
            assert!(!caught(op, |p| nudge(p, "0f00000000", "0f80000000", i)), "{op:?} sum identity {i} as -0");
        }
    }
}

