//! The differential equivalence gate for `nsl_rmsnorm_dx_bwd_f32` and
//! `nsl_rmsnorm_dx_bwd_add_f32` from `nsl_runtime::cuda::fused_kernels`
//! (new-roadmap item 5), now built by `nsl_kir::kernels::rmsnorm_dx`.
//!
//! This file runs the frozen hand modules
//! (`tests/fixtures/rmsnorm_dx_hand.rs`) and the KIR ones side by side on
//! the cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`), as
//! the runtime launches them (one block of 256 threads per row), plus one
//! block past the last row, under all four thread schedules. Shared memory
//! starts as NaN poison, as uninitialised shared memory may.
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory.
//! 2. **Correctness**: every row is the hand kernels' order restated in
//!    Rust, bit for bit. Thread `k` folds columns `k, k + 256, …` into `S1 =
//!    fma(x, x, S1)` and `S2 = fma(dy · γ, x, S2)`; thread 0 then adds the
//!    partials `1 .. 256` to its own, in order. `rinv = min(rsqrt(S1 / cols +
//!    eps), 1e12)`, `coeff = ((rinv · rinv) · rinv) · S2 / cols`, and `dx =
//!    (γ · dy) · rinv - x · coeff` (`+ res` for the twin), every multiply and
//!    add rounded on its own. Nothing past the last row is written, and the
//!    inputs are untouched.
//! 3. **The gate bites**: the row bound, each column loop's bound and
//!    stride, the fold's start, bound and step, the thread-0 test, both
//!    barriers, the row base, the second region's offset, every element size
//!    and 64-bit add, both `fma`s, every add, subtract and multiply (flipped
//!    and dropped), both `div.approx`, the `rsqrt`, the clamp (flipped and
//!    dropped), and each identity. Named equivalent mutants: the write pass
//!    striding by 128 (it is a pure map), and each sum's `+0` identity as
//!    `-0` (a `-0` partial only survives in a sum that is itself zero, where
//!    it multiplies an `x` of zero).

use std::collections::HashMap;

use nsl_kir::kernels::rmsnorm_dx::{ptx, RmsNormDxOp, RINV_MAX, RMSNORM_DX_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/rmsnorm_dx_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 64;
const POISON: u32 = 0x7FA5_A5A5;
const DY: u64 = 0x1000_0000;
const X: u64 = 0x2000_0000;
const GAMMA: u64 = 0x3000_0000;
const DXOUT: u64 = 0x4000_0000;
const RES: u64 = 0x5000_0000;
const B: usize = RMSNORM_DX_BLOCK as usize;
const SHARED: usize = 2 * B * 4;

fn kir_ptx(op: RmsNormDxOp) -> String {
    String::from_utf8(ptx(op)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(op: RmsNormDxOp) -> String {
    match op {
        RmsNormDxOp::Dx => hand::RMSNORM_DX_BWD_F32_PTX,
        RmsNormDxOp::DxAdd => hand::RMSNORM_DX_BWD_ADD_F32_PTX,
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

/// `rows` rows of `cols` columns, row-major, for `x`, `dy` and `res`, with
/// the per-column `gamma` and the launch's `eps`.
struct Case {
    rows: usize,
    cols: usize,
    x: Vec<u32>,
    dy: Vec<u32>,
    res: Vec<u32>,
    gamma: Vec<u32>,
    eps: f32,
}

fn case(x: Vec<Vec<u32>>, cols: usize, seed: u64, eps: f32) -> Case {
    let rows = x.len();
    Case {
        rows,
        cols,
        x: x.concat(),
        dy: data(rows * cols, seed ^ 0x33, 0.0),
        res: data(rows * cols, seed ^ 0x66, 0.0),
        gamma: data(cols, seed ^ 0x55, 1.0),
        eps,
    }
}

const WIDTHS: [usize; 6] = [0, 1, 255, 256, 257, 700];

fn cases() -> Vec<Case> {
    let mut v: Vec<Case> = WIDTHS
        .iter()
        .enumerate()
        .map(|(i, &cols)| {
            let seed = 3 + i as u64;
            case(vec![data(cols, seed, 0.5), data(cols, seed + 100, -0.25)], cols, seed, 1e-5)
        })
        .collect();
    // No rows: only the block past the end runs.
    v.push(case(vec![], 300, 1, 1e-5));
    // Rows of 300 columns: constant, a NaN, `+inf`, squares that overflow,
    // and one spike on the stride's second trip.
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
    // `eps` 0: a zero row's `rsqrt(0)` is `inf`, which the clamp holds at
    // 1e12; a tiny row's is past 1e12 and clamped too.
    let tiny: Vec<u32> = data(40, 96, 0.0).into_iter().map(|w| (f32::from_bits(w) * 1e-15).to_bits()).collect();
    v.push(case(vec![vec![0; 40], tiny, data(40, 95, 0.0)], 40, 95, 0.0));
    v
}

/// All of global memory after the launch: `[dy, x, gamma, dxout + tail,
/// res]`.
fn run(op: RmsNormDxOp, ptx: &str, c: &Case, order: Order) -> Vec<u32> {
    let prog = parse(ptx);
    let mut global = vec![
        Segment { base: DY, bytes: le32(&c.dy) },
        Segment { base: X, bytes: le32(&c.x) },
        Segment { base: GAMMA, bytes: le32(&c.gamma) },
        Segment { base: DXOUT, bytes: le32(&vec![POISON; c.rows * c.cols + TAIL]) },
        Segment { base: RES, bytes: le32(&c.res) },
    ];
    let (rows, cols, eps) = (c.rows as u64, c.cols as u64, u64::from(c.eps.to_bits()));
    let values: Vec<u64> = match op {
        RmsNormDxOp::Dx => vec![DY, X, GAMMA, DXOUT, rows, cols, eps],
        RmsNormDxOp::DxAdd => vec![DY, X, GAMMA, DXOUT, RES, rows, cols, eps],
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
            ntid: RMSNORM_DX_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    global.iter().flat_map(|s| words(&s.bytes)).collect()
}

const ORDERS: [Order; 4] = [Order::Ascending, Order::Descending, Order::WarpsAscending, Order::WarpsDescending];

#[test]
fn the_kernels_agree() {
    for op in RmsNormDxOp::ALL {
        let (hand, kir) = (hand_ptx(op), kir_ptx(op));
        for c in cases() {
            for order in ORDERS {
                assert!(run(op, &hand, &c, order) == run(op, &kir, &c, order), "{op:?} {}×{} {order:?}", c.rows, c.cols);
            }
        }
    }
}

/// Thread 0's in-order sum of the per-thread partials, or in reverse.
fn block_sum(partial: impl Fn(usize) -> f32, reversed: bool) -> f32 {
    if reversed {
        (1..B).rev().fold(partial(0), |acc, k| acc + partial(k))
    } else {
        (1..B).fold(partial(0), |acc, k| acc + partial(k))
    }
}

/// The hand kernels' order, restated, for one row.
fn reference(op: RmsNormDxOp, c: &Case, r: usize) -> Vec<u32> {
    reference_with(op, c, r, false)
}

/// [`reference`], or with thread 0 adding the partials in reverse
/// (`reversed`).
fn reference_with(op: RmsNormDxOp, c: &Case, r: usize, reversed: bool) -> Vec<u32> {
    let f = |w: &u32| f32::from_bits(*w);
    let row = |v: &[u32]| -> Vec<f32> { v[r * c.cols..(r + 1) * c.cols].iter().map(f).collect() };
    let (x, dy, res) = (row(&c.x), row(&c.dy), row(&c.res));
    let g: Vec<f32> = c.gamma.iter().map(f).collect();
    let n = c.cols as f32;
    let mine = |k: usize| (k..x.len()).step_by(B);
    let s1 = block_sum(|k| mine(k).fold(0.0f32, |s, i| x[i].mul_add(x[i], s)), reversed);
    let s2 = block_sum(|k| mine(k).fold(0.0f32, |s, i| (dy[i] * g[i]).mul_add(x[i], s)), reversed);
    let rinv = (1.0 / (s1 / n + c.eps).sqrt()).min(RINV_MAX);
    let coeff = ((rinv * rinv) * rinv) * s2 / n;
    (0..x.len())
        .map(|i| {
            let dx = (g[i] * dy[i]) * rinv - x[i] * coeff;
            match op {
                RmsNormDxOp::Dx => dx,
                RmsNormDxOp::DxAdd => dx + res[i],
            }
            .to_bits()
        })
        .collect()
}

fn same(a: u32, b: u32) -> bool {
    a == b || (f32::from_bits(a).is_nan() && f32::from_bits(b).is_nan())
}

/// `run`'s output matches the reference on every row, writes nothing past
/// the last row, and leaves the inputs alone.
fn is_reference(op: RmsNormDxOp, c: &Case, got: &[u32]) -> bool {
    let (dy, rest) = got.split_at(c.dy.len());
    let (x, rest) = rest.split_at(c.x.len());
    let (gamma, rest) = rest.split_at(c.gamma.len());
    let (out, res) = rest.split_at(c.rows * c.cols + TAIL);
    let rows_ok = (0..c.rows).all(|r| {
        let got = &out[r * c.cols..(r + 1) * c.cols];
        got.iter().zip(reference(op, c, r)).all(|(g, w)| same(*g, w))
    });
    let inputs_ok = dy == &c.dy[..] && x == &c.x[..] && gamma == &c.gamma[..] && res == &c.res[..];
    rows_ok && inputs_ok && out[c.rows * c.cols..].iter().all(|&w| w == POISON)
}

#[test]
fn the_kernels_are_the_reference() {
    for op in RmsNormDxOp::ALL {
        for which in [hand_ptx(op), kir_ptx(op)] {
            for c in cases() {
                for order in ORDERS {
                    assert!(is_reference(op, &c, &run(op, &which, &c, order)), "{op:?} {}×{} {order:?}", c.rows, c.cols);
                }
            }
        }
    }
}

/// The data reaches the cases the kernels distinguish: several trips of the
/// stride, sums that depend on their order, a NaN, `+inf`, squares that
/// overflow, and an `rsqrt` past the clamp.
#[test]
fn the_data_covers_every_case() {
    assert!(WIDTHS.iter().any(|&n| n > 2 * B) && WIDTHS.contains(&0));
    let order_shows = cases()
        .iter()
        .any(|c| (0..c.rows).any(|r| reference_with(RmsNormDxOp::Dx, c, r, true) != reference_with(RmsNormDxOp::Dx, c, r, false)));
    assert!(order_shows, "the sums depend on order");
    let all: Vec<f32> = cases().iter().flat_map(|c| c.x.clone()).map(f32::from_bits).collect();
    assert!(all.iter().any(|v| v.is_nan()) && all.contains(&f32::INFINITY));
    assert!(all.iter().any(|v| (v * v).is_infinite() && v.is_finite()));
    let clamped = cases().iter().any(|c| {
        (0..c.rows).any(|r| {
            let x: Vec<f32> = c.x[r * c.cols..(r + 1) * c.cols].iter().map(|w| f32::from_bits(*w)).collect();
            let s1 = x.iter().fold(0.0f32, |s, v| v.mul_add(*v, s));
            1.0 / (s1 / c.cols as f32 + c.eps).sqrt() > RINV_MAX
        })
    });
    assert!(clamped, "some row's rsqrt passes the clamp");
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the reference on any case or
/// schedule (or faults).
fn caught(op: RmsNormDxOp, mutate: impl Fn(&str) -> String) -> bool {
    let mutant = mutate(&kir_ptx(op));
    cases().iter().any(|c| {
        ORDERS.into_iter().any(|order| match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(op, &mutant, c, order))) {
            Ok(r) => !is_reference(op, c, &r),
            Err(_) => true,
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

/// `op d, a, b[, c];` as `mov.<ty> d, a;`: the operation dropped.
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
    for op in RmsNormDxOp::ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

/// The row bound and each column loop's bound and stride. Striding by 512
/// skips columns in both passes. Striding by 128 revisits them: the
/// reducing pass counts them twice, but the write pass is a pure map, so
/// there it is a named equivalent mutant.
#[test]
fn the_row_and_column_loops_are_pinned() {
    for op in RmsNormDxOp::ALL {
        assert_eq!(kir_ptx(op).matches("setp.ge.u64 ").count(), 3, "{op:?}");
        for i in 0..3 {
            assert!(caught(op, |p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "{op:?} bound {i}");
        }
        assert_eq!(kir_ptx(op).matches(", 256;").count(), 2, "{op:?}");
        for pass in 0..2 {
            assert!(caught(op, |p| nudge(p, ", 256;", ", 512;", pass)), "{op:?} stride {pass} doubled");
            assert_eq!(caught(op, |p| nudge(p, ", 256;", ", 128;", pass)), pass == 0, "{op:?} stride {pass} halved");
        }
    }
}

/// The fold's start (as 0) and step (as 2), bound and thread-0 test, and
/// both barriers.
#[test]
fn the_fold_and_barriers_are_pinned() {
    for op in RmsNormDxOp::ALL {
        let p = kir_ptx(op);
        let consts: Vec<usize> =
            p.lines().enumerate().filter(|(_, l)| l.trim().starts_with("mov.u32 ") && l.ends_with(", 1;")).map(|(n, _)| n).collect();
        assert_eq!(consts.len(), 2, "{op:?}: the fold's start and step");
        for (j, &at) in consts.iter().enumerate() {
            let to = if j == 0 { ", 0;" } else { ", 2;" };
            let mutate = |p: &str| {
                p.lines().enumerate().map(|(n, l)| if n == at { l.replacen(", 1;", to, 1) } else { l.to_string() }).collect::<Vec<_>>().join("\n")
                    + "\n"
            };
            assert!(caught(op, mutate), "{op:?} fold constant {j}");
        }
        assert!(caught(op, |p| nudge(p, "setp.ge.u32 ", "setp.gt.u32 ", 0)), "{op:?} fold bound");
        assert!(caught(op, |p| nudge(p, "setp.ne.u32 ", "setp.eq.u32 ", 0)), "{op:?} thread-0 test");
        assert_eq!(p.matches("bar.sync 0;").count(), 2, "{op:?}");
        for i in 0..2 {
            assert!(caught(op, |p| edit(p, "bar.sync 0;", i, |_| String::new())), "{op:?} barrier {i}");
        }
    }
}

/// The row base, the second region's offset, every element size and every
/// 64-bit add but the column loops' steps (their stride is pinned above;
/// dropped, a loop would not end).
#[test]
fn every_address_is_pinned() {
    for op in RmsNormDxOp::ALL {
        let p = kir_ptx(op);
        assert!(caught(op, |p| drop_op(p, "mul.lo.u64 ", "u64", 0)), "{op:?} row base");
        assert!(caught(op, |p| p.replacen(", 1024;", ", 0;", 1)), "{op:?} the second region");
        let sizes = p.lines().filter(|l| l.contains("mul.lo.u64") && l.ends_with(", 4;")).count();
        for i in 0..sizes {
            assert!(caught(op, |p| nudge(p, ", 4;", ", 8;", i)), "{op:?} element size {i}");
        }
        let lines: Vec<&str> = p.lines().collect();
        let adds: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.contains("add.u64 ")).map(|(n, _)| n).collect();
        assert_eq!(adds.iter().filter(|&&n| lines[n - 1].ends_with(", 256;")).count(), 2, "{op:?}");
        for (i, &n) in adds.iter().enumerate() {
            if !lines[n - 1].ends_with(", 256;") {
                assert!(caught(op, |p| drop_op(p, "add.u64 ", "u64", i)), "{op:?} add {i}");
            }
        }
    }
}

/// Both `fma`s, every add, subtract and multiply (flipped and dropped),
/// both `div.approx`, the `rsqrt`, the clamp, and each identity. The
/// identity as `-0` is a named equivalent mutant.
#[test]
fn the_arithmetic_is_pinned() {
    for op in RmsNormDxOp::ALL {
        let p = kir_ptx(op);
        for i in 0..2 {
            assert!(caught(op, |p| drop_op(p, "fma.rn.f32 ", "f32", i)), "{op:?} fma {i} dropped");
            // `fma(a, b, c)` as `a · b`: the accumulate dropped.
            assert!(
                caught(op, |p| edit(p, "fma.rn.f32 ", i, |l| {
                    let (head, operands) = l.split_once("fma.rn.f32 ").expect("fma");
                    let ops: Vec<&str> = operands.trim_end_matches(';').split(',').map(str::trim).collect();
                    format!("{head}mul.rn.f32 {}, {}, {};", ops[0], ops[1], ops[2])
                })),
                "{op:?} fma {i} accumulate"
            );
        }
        for (from, to) in [("add.rn.f32 ", "sub.rn.f32 "), ("sub.rn.f32 ", "add.rn.f32 "), ("mul.rn.f32 ", "add.rn.f32 ")] {
            let n = p.matches(from).count();
            assert!(n > 0, "{op:?} {from}");
            for i in 0..n {
                assert!(caught(op, |p| nudge(p, from, to, i)), "{op:?} {from} {i}");
                assert!(caught(op, |p| drop_op(p, from, "f32", i)), "{op:?} {from} {i} dropped");
            }
        }
        for i in 0..2 {
            assert!(caught(op, |p| drop_op(p, "div.approx.f32 ", "f32", i)), "{op:?} div {i}");
        }
        assert!(caught(op, |p| edit(p, "rsqrt.approx.f32 ", 0, |l| l.replacen("rsqrt.approx.f32", "mov.f32", 1))), "{op:?} rsqrt");
        assert!(caught(op, |p| nudge(p, "min.f32 ", "max.f32 ", 0)), "{op:?} clamp flipped");
        assert!(caught(op, |p| drop_op(p, "min.f32 ", "f32", 0)), "{op:?} clamp dropped");
        assert_eq!(p.matches("0f00000000").count(), 2, "{op:?}");
        for i in 0..2 {
            assert!(caught(op, |p| nudge(p, "0f00000000", "0f3F800000", i)), "{op:?} sum identity {i} as 1");
            assert!(!caught(op, |p| nudge(p, "0f00000000", "0f80000000", i)), "{op:?} sum identity {i} as -0");
        }
    }
}
