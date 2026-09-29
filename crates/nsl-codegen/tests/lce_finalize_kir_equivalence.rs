//! The differential equivalence gate for the fused linear cross-entropy's
//! finalize kernel from `nsl_runtime::cuda::fused_kernels` (new-roadmap item
//! 5): `nsl_lce_finalize_f32`, now built by `nsl_kir::kernels::lce_finalize`.
//!
//! This file runs the frozen hand module (`tests/fixtures/lce_finalize_hand.rs`)
//! and the KIR one side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`). Each launch is a whole grid of
//! 256-thread blocks with `%nctaid.x` set, so the grid stride is real. The
//! grids are the runtime's (`ceil(rows / 256)`, at least 1), a single block,
//! and three blocks, which do not divide the work. They run under all four
//! thread schedules and both block orders.
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory. The rows are empty, short and ragged past two strides. The data
//!    has sums with full significands and sums of zero, a subnormal, `inf`
//!    and NaN. The targets are valid, zero, negative and `i64::MIN`.
//! 2. **Correctness**: `lse[r] = fma(log2(s[r]), ln 2, m[r])`, and `loss[r]
//!    = lse[r] - tl[r]` for a target `>= 0`, else `+0`. Both match bit for
//!    bit, nothing past `rows` is written, and the state is untouched.
//! 3. **The gate bites**:
//!    - the bound, the start's block index, and a stride that skips rows or
//!      never advances;
//!    - the `lg2`, the `ln 2` constant, and the `fma`'s operands;
//!    - the `fma` split into a multiply and an add, which rounds twice, as
//!      `KirOp::Log` would;
//!    - the target's sign test (signed, strict, against zero), the invalid
//!      row's zero, and the subtraction;
//!    - every element size, every 64-bit add, and every pointer.
//!
//!    A stride that shrinks but stays nonzero is named as an equivalent
//!    mutant: each row is still covered, and a row computed twice is
//!    written with the same bytes.

use std::collections::HashMap;

use nsl_kir::kernels::lce_finalize::{ptx, LCE_FINALIZE_BLOCK, LN_2, PARAM_NAMES};

#[allow(dead_code)]
#[path = "fixtures/lce_finalize_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 16;
const POISON: u32 = 0x7FA5_A5A5;
const M: u64 = 0x1000_0000;
const S: u64 = 0x2000_0000;
const TL: u64 = 0x3000_0000;
const TGT: u64 = 0x4000_0000;
const LOSS: u64 = 0x5000_0000;
const LSE: u64 = 0x6000_0000;
const B: usize = LCE_FINALIZE_BLOCK as usize;

fn kir_ptx() -> String {
    String::from_utf8(ptx()).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx() -> String {
    hand::LCE_FINALIZE_F32_PTX.trim_end_matches('\0').to_string()
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    *s >> 33
}

/// A value with a full 24-bit significand, `2^e` for `e` in `lo..=hi`.
fn full(s: &mut u64, lo: i32, hi: i32, signed: bool) -> f32 {
    let e = (lcg(s) % (hi - lo + 1) as u64) as i32 + lo;
    let mant = 1.0 + (lcg(s) % (1 << 23)) as f32 / (1u32 << 23) as f32;
    let sign = if signed && lcg(s).is_multiple_of(2) { -1.0 } else { 1.0 };
    sign * mant * 2f32.powi(e)
}

/// One row's state: `(m, s, tl, target)`.
struct Rows {
    m: Vec<f32>,
    s: Vec<f32>,
    tl: Vec<f32>,
    tgt: Vec<i64>,
}

fn rows(n: usize, seed: u64) -> Rows {
    let mut r = seed;
    let mut v = Rows { m: vec![], s: vec![], tl: vec![], tgt: vec![] };
    for _ in 0..n {
        v.m.push(full(&mut r, -4, 6, true));
        v.s.push(match lcg(&mut r) % 40 {
            0 => 0.0,
            1 => f32::from_bits(0x0000_0200), // a subnormal (`powi` would flush it)
            2 => f32::INFINITY,
            3 => f32::NAN,
            _ => full(&mut r, 0, 20, false),
        });
        v.tl.push(full(&mut r, -4, 6, true));
        v.tgt.push(match lcg(&mut r) % 6 {
            0 => -1,
            1 => -100,
            2 => 0,
            3 => i64::MIN,
            _ => (lcg(&mut r) % 50_000) as i64,
        });
    }
    v
}

/// `(rows, grid)`.
fn cases() -> Vec<(usize, u32)> {
    let mut v = vec![];
    for n in [0usize, 1, 255, 256, 257, 700, 1000] {
        v.push((n, n.div_ceil(B).max(1) as u32));
        v.push((n, 1));
        v.push((n, 3));
    }
    v
}

fn le32(v: impl IntoIterator<Item = u32>) -> Vec<u8> {
    v.into_iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// All of global memory after the launch, as 4-byte words, segment by
/// segment: `[m, s, tl, targets, loss + tail, lse + tail]`.
fn run(ptx: &str, n: usize, grid: u32, order: Order, reverse: bool) -> Vec<Vec<u32>> {
    let prog = parse(ptx);
    let r = rows(n, 11 + n as u64);
    let pad = |mut b: Vec<u8>| {
        b.resize(b.len().max(8), 0);
        b
    };
    let out = || le32(std::iter::repeat_n(POISON, n + TAIL));
    let mut global = vec![
        Segment { base: M, bytes: pad(le32(r.m.iter().map(|x| x.to_bits()))) },
        Segment { base: S, bytes: pad(le32(r.s.iter().map(|x| x.to_bits()))) },
        Segment { base: TL, bytes: pad(le32(r.tl.iter().map(|x| x.to_bits()))) },
        Segment { base: TGT, bytes: pad(r.tgt.iter().flat_map(|t| t.to_le_bytes()).collect()) },
        Segment { base: LOSS, bytes: out() },
        Segment { base: LSE, bytes: out() },
    ];
    let args: HashMap<String, u64> =
        PARAM_NAMES.iter().zip([M, S, TL, TGT, LOSS, LSE, n as u64]).map(|(p, v)| (p.to_string(), v)).collect();
    let mut ctas: Vec<u32> = (0..grid).collect();
    if reverse {
        ctas.reverse();
    }
    for ctaid in ctas {
        let mut l = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![],
            ctaid,
            ctaid_y: 0,
            nctaid_x: grid,
            nctaid_y: 1,
            ntid: LCE_FINALIZE_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    global.iter().map(|s| words(&s.bytes)).collect()
}

const ORDERS: [Order; 4] = [Order::Ascending, Order::Descending, Order::WarpsAscending, Order::WarpsDescending];

#[test]
fn the_kernels_agree() {
    let (hand, kir) = (hand_ptx(), kir_ptx());
    for (n, grid) in cases() {
        for order in ORDERS {
            for rev in [false, true] {
                assert!(run(&hand, n, grid, order, rev) == run(&kir, n, grid, order, rev), "n={n} grid={grid} {order:?} rev={rev}");
            }
        }
    }
}

/// The hand kernel's formulas, restated: `(loss, lse)` per row. The
/// interpreter reads `lg2.approx.f32` as `log2`.
fn reference(n: usize) -> (Vec<u32>, Vec<u32>) {
    let r = rows(n, 11 + n as u64);
    (0..n)
        .map(|i| {
            let lse = r.s[i].log2().mul_add(LN_2, r.m[i]);
            let loss = if r.tgt[i] < 0 { 0.0 } else { lse - r.tl[i] };
            (loss.to_bits(), lse.to_bits())
        })
        .unzip()
}

fn same(got: &[u32], want: &[u32]) -> bool {
    got.len() == want.len()
        && got.iter().zip(want).all(|(g, w)| g == w || (f32::from_bits(*g).is_nan() && f32::from_bits(*w).is_nan()))
}

#[test]
fn the_kernels_are_the_formulas() {
    for which in [hand_ptx(), kir_ptx()] {
        for (n, grid) in cases() {
            let got = run(&which, n, grid, Order::Ascending, false);
            let r = rows(n, 11 + n as u64);
            assert_eq!(&got[0][..n], &r.m.iter().map(|x| x.to_bits()).collect::<Vec<_>>()[..], "n={n}: m untouched");
            assert_eq!(&got[1][..n], &r.s.iter().map(|x| x.to_bits()).collect::<Vec<_>>()[..], "n={n}: s untouched");
            let (loss, lse) = reference(n);
            assert!(same(&got[4][..n], &loss), "n={n} grid={grid}: loss");
            assert!(same(&got[5][..n], &lse), "n={n} grid={grid}: lse");
            assert!(got[4][n..].iter().chain(&got[5][n..]).all(|&w| w == POISON), "n={n} grid={grid}: wrote past rows");
        }
    }
}

/// The data reaches every case the kernel distinguishes: each kind of
/// target, sums whose log is special, threads with several rows, and rows on
/// which a fused `ln` and one split into a multiply and an add differ.
#[test]
fn the_data_covers_every_case() {
    let r = rows(1000, 1011);
    for t in [-1, -100, 0, i64::MIN] {
        assert!(r.tgt.contains(&t), "target {t}");
    }
    assert!(r.tgt.iter().any(|&t| t > 0));
    assert!(r.s.contains(&0.0) && r.s.iter().any(|s| s.is_nan()) && r.s.iter().any(|s| s.is_infinite()));
    assert!(r.s.iter().any(|s| s.is_subnormal()));
    assert!(cases().iter().any(|&(n, g)| n > 2 * B * g as usize), "a thread takes three rows");
    let split = (0..1000).filter(|&i| {
        let l = r.s[i].log2();
        let fused = l.mul_add(LN_2, r.m[i]);
        let twice = l * LN_2 + r.m[i];
        fused.to_bits() != twice.to_bits() && !fused.is_nan()
    });
    assert!(split.count() > 0, "rounding the multiply shows");
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

fn caught(mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx();
    let mutant = mutate(&kir_ptx());
    cases().into_iter().any(|(n, grid)| {
        ORDERS.into_iter().any(|order| {
            [false, true].into_iter().any(|rev| {
                let expect = run(&hand, n, grid, order, rev);
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mutant, n, grid, order, rev))) {
                    Ok(r) => r != expect,
                    Err(_) => true,
                }
            })
        })
    })
}

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

/// `op d, a, b...;` as `mov.<ty> d, a;`: the operation dropped.
fn drop_op(ptx: &str, op: &str, ty: &str, i: usize) -> String {
    edit(ptx, op, i, |l| {
        let (head, operands) = l.split_once(op).expect("the op");
        let mut ops: Vec<&str> = operands.trim_end_matches(';').split(',').map(str::trim).collect();
        ops.truncate(2);
        format!("{head}mov.{ty} {};", ops.join(", "))
    })
}

/// The operands of the one line holding `op`.
fn operands(ptx: &str, op: &str) -> (String, Vec<String>) {
    let line = ptx.lines().find(|l| l.contains(op)).unwrap_or_else(|| panic!("no `{op}`"));
    let (head, ops) = line.split_once(op).expect("the op");
    (head.to_string(), ops.trim_end_matches(';').split(',').map(|o| o.trim().to_string()).collect())
}

/// `fma.rn.f32 d, a, b, c;` rewritten by `g(d, a, b, c)`.
fn fma_as(ptx: &str, g: impl Fn(&str, &str, &str, &str) -> String) -> String {
    let (head, o) = operands(ptx, "fma.rn.f32 ");
    let line = format!("{head}fma.rn.f32 {};", o.join(", "));
    ptx.replacen(&line, &format!("{head}{}", g(&o[0], &o[1], &o[2], &o[3])), 1)
}

#[test]
fn the_unmutated_kernel_is_not_caught() {
    assert!(!caught(|p| p.to_string()));
}

/// The stride's line: `mul.lo.u32 d, %ntid, %nctaid;`, the second
/// `mul.lo.u32` (the first forms the global id).
fn stride_line(ptx: &str) -> String {
    ptx.lines().filter(|l| l.contains("mul.lo.u32 ")).nth(1).expect("the stride").to_string()
}

/// `stride_line` followed by `extra` (with `{d}` its destination).
fn after_stride(ptx: &str, extra: &str) -> String {
    let line = stride_line(ptx);
    let d = line.split_whitespace().nth(1).expect("dst").trim_end_matches(',').to_string();
    ptx.replacen(&line, &format!("{line}\n    {}", extra.replace("{d}", &d)), 1)
}

/// The bound, the start's block index, and the grid stride: doubled, a
/// thread skips rows; zero, it never leaves the loop.
#[test]
fn the_bound_and_stride_are_pinned() {
    assert_eq!(kir_ptx().matches("setp.ge.u64 ").count(), 1);
    assert!(caught(|p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", 0)), "bound");
    assert!(caught(|p| p.replacen("%ctaid.x", "0", 1)), "start's block index");
    assert_eq!(kir_ptx().matches("mul.lo.u32 ").count(), 2, "the global id's and the stride's");
    assert!(stride_line(&kir_ptx()).contains("%r"), "{}", stride_line(&kir_ptx()));
    assert!(caught(|p| after_stride(p, "add.u32 {d}, {d}, {d};")), "stride doubled");
    assert!(caught(|p| after_stride(p, "mov.u32 {d}, 0;")), "stride zero");
}

/// Named equivalent mutants: a stride that shrinks but stays nonzero (the
/// grid width read as 1, the block width read as 1, the product dropped).
/// The kernel maps each row to itself alone, so a thread that also takes
/// rows another thread takes writes the same bytes to them; every row is
/// still covered. Only a stride that skips rows, or none, shows.
#[test]
fn a_shrunk_stride_is_an_equivalent_mutant() {
    assert!(!caught(|p| p.replacen("%nctaid.x", "1", 1)), "grid width as 1");
    assert_eq!(kir_ptx().matches("%ntid.x").count(), 2, "the start's and the stride's");
    assert!(!caught(|p| nudge(p, "%ntid.x", "1", 1)), "block width as 1");
    assert!(!caught(|p| drop_op(p, "mul.lo.u32 ", "u32", 1)), "product dropped");
}

/// The bare `lg2`, the `ln 2` constant, and each of the `fma`'s operands;
/// the `fma` rounded twice (a multiply, then an add) is told apart.
#[test]
fn the_log_sum_exp_is_pinned() {
    assert!(caught(|p| drop_op(p, "lg2.approx.f32 ", "f32", 0)), "lg2");
    assert!(caught(|p| p.replacen("0f3F317218", "0f3F800000", 1)), "ln 2");
    assert!(caught(|p| p.replacen("0f3F317218", "0f3F317219", 1)), "ln 2, one ulp");
    assert!(caught(|p| fma_as(p, |d, a, b, _| format!("mul.rn.f32 {d}, {a}, {b};"))), "addend dropped");
    assert!(caught(|p| fma_as(p, |d, _, b, c| format!("add.rn.f32 {d}, {b}, {c};"))), "log dropped");
    assert!(caught(|p| fma_as(p, |d, a, _, c| format!("add.rn.f32 {d}, {a}, {c};"))), "ln 2 dropped");
    assert!(
        caught(|p| fma_as(p, |d, a, b, c| format!("mul.rn.f32 {d}, {a}, {b};\n    add.rn.f32 {d}, {d}, {c};"))),
        "rounded twice"
    );
}

/// The target's sign test (signed, strict, against zero), the invalid
/// row's zero, and the subtraction's operation and order.
#[test]
fn the_loss_is_pinned() {
    assert!(caught(|p| nudge(p, "setp.lt.s64 ", "setp.le.s64 ", 0)), "strict");
    assert!(caught(|p| nudge(p, "setp.lt.s64 ", "setp.lt.u64 ", 0)), "signed");
    assert!(caught(|p| nudge(p, "mov.s64 %rd7, 0;", "mov.s64 %rd7, 1;", 0)), "against zero");
    assert!(caught(|p| p.replacen("mov.f32 %f0, 0f00000000;", "mov.f32 %f0, 0f80000000;", 1)), "the zero's sign");
    assert!(caught(|p| nudge(p, "sub.rn.f32 ", "add.rn.f32 ", 0)), "subtract");
    let (head, o) = operands(&kir_ptx(), "sub.rn.f32 ");
    let line = format!("{head}sub.rn.f32 {};", o.join(", "));
    let swapped = format!("{head}sub.rn.f32 {}, {}, {};", o[0], o[2], o[1]);
    assert!(caught(|p| p.replacen(&line, &swapped, 1)), "order");
    assert!(caught(|p| drop_op(p, "sub.rn.f32 ", "f32", 0)), "tl dropped");
}

/// Every element size (4 for the f32 state, 8 for the targets), every
/// 64-bit add, and every pointer (each parameter read as its neighbour).
#[test]
fn every_address_is_pinned() {
    let p = kir_ptx();
    let sizes: Vec<(usize, &str)> = p
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains("mul.lo.u64") && (l.ends_with(", 4;") || l.ends_with(", 8;")))
        .map(|(n, l)| (n, if l.ends_with(", 4;") { ", 4;" } else { ", 8;" }))
        .collect();
    let fours = sizes.iter().filter(|s| s.1 == ", 4;").count();
    assert_eq!((fours, sizes.len() - fours), (5, 1), "m, s, lse, tl, loss; the target");
    for (at, size) in sizes {
        let other = if size == ", 4;" { ", 8;" } else { ", 4;" };
        let mutate = |p: &str| {
            p.lines().enumerate().map(|(n, l)| if n == at { l.replacen(size, other, 1) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
        };
        assert!(caught(mutate), "element size at line {at}");
    }
    let adds = p.matches("add.u64 ").count();
    for i in 0..adds {
        assert!(caught(|p| drop_op(p, "add.u64 ", "u64", i)), "add.u64 {i}");
    }
    for (i, name) in PARAM_NAMES.iter().enumerate().take(6) {
        let other = PARAM_NAMES[(i + 1) % 6];
        assert!(caught(|p| p.replacen(&format!("[param_{name}]"), &format!("[param_{other}]"), 1)), "{name} as {other}");
    }
}
