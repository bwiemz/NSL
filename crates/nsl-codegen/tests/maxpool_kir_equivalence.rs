//! The differential equivalence gate for `nsl_maxpool2d_f32` from
//! `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5), now built by
//! `nsl_kir::kernels::maxpool`.
//!
//! This file runs the frozen hand module (`tests/fixtures/maxpool_hand.rs`)
//! and the KIR one side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`) over whole grids, with a block more
//! than the work needs, on the runtime's 256-thread block and a 32-thread
//! one, under two schedules:
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory.
//! 2. **Correctness**: every output is the hand kernel's window walk restated
//!    in Rust: taps in `ky`, `kx` order, padding and out-of-range taps
//!    skipped, the running `(max, argmax)` from `(-inf, 0)` replaced unless
//!    `x <= max`. So ties keep the first tap, a NaN tap wins and the next
//!    tap displaces it, and a window wholly in the padding stays at `(-inf,
//!    0)`. Nothing past the outputs is written and the input is untouched.
//! 3. **The gate bites**: every bound and padding test, every `div`/`rem` of
//!    the index split, every index multiply, add and subtract, the loop
//!    starts and steps, the tie test, both `selp`s, the running pair's
//!    start, every element size, and the argmax's 64-bit store.
//!
//! The shapes cover strides below, at and past the window (overlapping,
//! tiling and gapped windows), padding of 0, 1 and 2 (the last with a
//! window wholly in the padding), and one to three channels and images.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;
use nsl_kir::kernels::maxpool::{ptx, PARAM_NAMES};

#[allow(dead_code)]
#[path = "fixtures/maxpool_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 16;
const POISON: u32 = 0x7FA5_A5A5;
const INP: u64 = 0x1000_0000;
const OUT: u64 = 0x2000_0000;
const ARGMAX: u64 = 0x3000_0000;

fn kir_ptx() -> String {
    String::from_utf8(ptx()).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx() -> String {
    hand::MAXPOOL2D_F32_PTX.trim_end_matches('\0').to_string()
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

/// `(N, C, H, W, kH, kW, stride, padding)`.
#[derive(Clone, Copy, Debug)]
struct Shape {
    n: usize,
    c: usize,
    h: usize,
    w: usize,
    kh: usize,
    kw: usize,
    stride: usize,
    pad: usize,
}

impl Shape {
    fn h_out(&self) -> usize {
        (self.h + 2 * self.pad - self.kh) / self.stride + 1
    }
    fn w_out(&self) -> usize {
        (self.w + 2 * self.pad - self.kw) / self.stride + 1
    }
    fn total(&self) -> usize {
        self.n * self.c * self.h_out() * self.w_out()
    }
    fn len(&self) -> usize {
        self.n * self.c * self.h * self.w
    }
}

#[allow(clippy::too_many_arguments)]
const fn shape(n: usize, c: usize, h: usize, w: usize, kh: usize, kw: usize, stride: usize, pad: usize) -> Shape {
    Shape { n, c, h, w, kh, kw, stride, pad }
}

const SHAPES: [Shape; 7] = [
    shape(1, 1, 1, 1, 1, 1, 1, 0),
    shape(2, 3, 5, 7, 2, 2, 2, 0),
    shape(1, 2, 6, 6, 3, 3, 1, 1),
    shape(2, 1, 9, 4, 3, 2, 2, 1),
    shape(1, 1, 2, 3, 2, 2, 1, 2),
    shape(3, 2, 7, 8, 2, 2, 3, 0),
    shape(2, 3, 16, 16, 3, 3, 2, 1),
];

/// Values from a small set, so windows hold ties (including `-0` against
/// `+0`), with a few NaNs and infinities, and all-negative stretches.
fn data(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| match lcg(&mut s) % 32 {
            0 => f32::NAN.to_bits(),
            1 => f32::INFINITY.to_bits(),
            2 => f32::NEG_INFINITY.to_bits(),
            3 => (-0.0f32).to_bits(),
            4 => 0.0f32.to_bits(),
            _ => ((lcg(&mut s) % 9) as f32 - 7.5).to_bits(),
        })
        .collect()
}

struct Case {
    s: Shape,
    x: Vec<u32>,
}

fn cases() -> Vec<Case> {
    SHAPES.iter().enumerate().map(|(i, &s)| Case { s, x: data(s.len(), 7 + i as u64) }).collect()
}

const BLOCKS: [u32; 2] = [ELEMENTWISE_BLOCK, 32];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

/// `(out + tail, argmax + tail)` as words, after the launch.
fn run(ptx: &str, c: &Case, block: u32, order: Order) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let prog = parse(ptx);
    let s = c.s;
    let total = s.total();
    let mut global = vec![
        Segment { base: INP, bytes: le32(&c.x) },
        Segment { base: OUT, bytes: le32(&vec![POISON; total + TAIL]) },
        Segment { base: ARGMAX, bytes: le32(&vec![POISON; 2 * (total + TAIL)]) },
    ];
    let values = [
        INP,
        OUT,
        ARGMAX,
        s.n as u64,
        s.c as u64,
        s.h as u64,
        s.w as u64,
        s.kh as u64,
        s.kw as u64,
        s.stride as u64,
        s.pad as u64,
        s.h_out() as u64,
        s.w_out() as u64,
        total as u64,
    ];
    let args: HashMap<String, u64> = PARAM_NAMES.iter().zip(values).map(|(p, v)| (p.to_string(), v)).collect();
    let grid = total.div_ceil(block as usize) as u32 + 1;
    let mut ctas: Vec<u32> = (0..grid).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for ctaid in ctas {
        let mut l = Launch { prog: &prog, args: &args, global: &mut global, shared: vec![], ctaid, ctaid_y: 0, nctaid_x: 0, nctaid_y: 1, ntid: block, steps: 0 };
        run_cta(&mut l, order);
    }
    (words(&global[0].bytes), words(&global[1].bytes), words(&global[2].bytes))
}

#[test]
fn the_kernels_agree() {
    let (hand, kir) = (hand_ptx(), kir_ptx());
    for c in cases() {
        for block in BLOCKS {
            for order in ORDERS {
                assert!(run(&hand, &c, block, order) == run(&kir, &c, block, order), "{:?} {block} {order:?}", c.s);
            }
        }
    }
}

/// The hand kernel's window walk, restated: `(out, argmax)` per output.
fn reference(c: &Case) -> (Vec<u32>, Vec<u64>) {
    let s = c.s;
    let (h_out, w_out) = (s.h_out(), s.w_out());
    let mut out = vec![];
    let mut arg = vec![];
    for i in 0..s.total() {
        let (ow, oh) = (i % w_out, (i / w_out) % h_out);
        let (ch, n) = ((i / w_out / h_out) % s.c, i / w_out / h_out / s.c);
        let (mut m, mut a) = (f32::NEG_INFINITY, 0u64);
        for ky in 0..s.kh {
            for kx in 0..s.kw {
                let (ih0, iw0) = (oh * s.stride + ky, ow * s.stride + kx);
                if ih0 < s.pad || iw0 < s.pad {
                    continue;
                }
                let (ih, iw) = (ih0 - s.pad, iw0 - s.pad);
                if ih >= s.h || iw >= s.w {
                    continue;
                }
                let j = ((n * s.c + ch) * s.h + ih) * s.w + iw;
                let x = f32::from_bits(c.x[j]);
                // The hand kernel's test, NaN included: replace unless `x <= m`.
                #[allow(clippy::neg_cmp_op_on_partial_ord)]
                let replace = !(x <= m);
                if replace {
                    m = x;
                    a = j as u64;
                }
            }
        }
        out.push(m.to_bits());
        arg.push(a);
    }
    (out, arg)
}

#[test]
fn the_kernels_are_the_window_walk() {
    for which in [hand_ptx(), kir_ptx()] {
        for c in cases() {
            for block in BLOCKS {
                let (inp, out, arg) = run(&which, &c, block, Order::Ascending);
                let total = c.s.total();
                let (want_out, want_arg) = reference(&c);
                assert_eq!(inp, c.x, "{:?}: input untouched", c.s);
                assert_eq!(&out[..total], &want_out[..], "{:?} {block}: out", c.s);
                let got_arg: Vec<u64> = (0..total).map(|k| u64::from(arg[2 * k]) | (u64::from(arg[2 * k + 1]) << 32)).collect();
                assert_eq!(got_arg, want_arg, "{:?} {block}: argmax", c.s);
                assert!(out[total..].iter().chain(&arg[2 * total..]).all(|&w| w == POISON), "{:?}: wrote past the outputs", c.s);
            }
        }
    }
}

/// The data reaches the cases the walk distinguishes: tied taps, a NaN
/// tap, a window wholly in the padding, and more outputs than a block.
#[test]
fn the_data_covers_every_case() {
    let cs = cases();
    assert!(cs.iter().any(|c| c.s.total() > ELEMENTWISE_BLOCK as usize), "more outputs than a block");
    let mut ties = false;
    let mut nan_wins = false;
    let mut padded = false;
    for c in &cs {
        let (out, arg) = reference(c);
        padded |= out.iter().zip(&arg).any(|(&o, &a)| o == f32::NEG_INFINITY.to_bits() && a == 0);
        nan_wins |= out.iter().any(|&o| f32::from_bits(o).is_nan());
        ties |= arg.iter().zip(&out).any(|(&a, &o)| c.x.iter().enumerate().any(|(j, &x)| x == o && j as u64 != a && f32::from_bits(o).is_finite()));
    }
    assert!(ties && nan_wins && padded, "ties {ties} nan {nan_wins} padded {padded}");
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on any case,
/// block or schedule (or faults).
fn caught(mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx();
    let mutant = mutate(&kir_ptx());
    cases().iter().any(|c| {
        BLOCKS.into_iter().any(|block| {
            ORDERS.into_iter().any(|order| {
                let expect = run(&hand, c, block, order);
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mutant, c, block, order))) {
                    Ok(r) => r != expect,
                    Err(_) => true,
                }
            })
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

/// The line numbers of the loop steps: an `add.u64` right after the
/// `mov.u64 _, 1;` it adds.
fn steps(p: &str) -> Vec<usize> {
    let lines: Vec<&str> = p.lines().collect();
    (1..lines.len()).filter(|&n| lines[n].contains("add.u64 ") && lines[n - 1].trim().starts_with("mov.u64 ") && lines[n - 1].ends_with(", 1;")).collect()
}

#[test]
fn the_unmutated_kernel_is_not_caught() {
    assert!(!caught(|p| p.to_string()));
}

/// Every bound (the thread's, both loops', both input edges) and both
/// padding tests.
#[test]
fn every_bound_and_padding_test_is_pinned() {
    let p = kir_ptx();
    assert_eq!(p.matches("setp.ge.u64 ").count(), 5);
    for i in 0..5 {
        assert!(caught(|p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "bound {i}");
    }
    assert_eq!(p.matches("setp.lt.u64 ").count(), 2);
    for i in 0..2 {
        assert!(caught(|p| nudge(p, "setp.lt.u64 ", "setp.le.u64 ", i)), "padding test {i}");
    }
}

/// The index split: each `div` and `rem`, swapped for the other and
/// dropped.
#[test]
fn the_index_split_is_pinned() {
    for (op, other) in [("div.u64 ", "rem.u64 "), ("rem.u64 ", "div.u64 ")] {
        for i in 0..3 {
            assert!(caught(|p| nudge(p, op, other, i)), "{op} {i} swapped");
            assert!(caught(|p| drop_op(p, op, "u64", i)), "{op} {i} dropped");
        }
    }
}

/// Every index multiply, add and subtract (the loop steps are nudged: a
/// dropped step would never end), each loop's start and step, and every
/// element size.
#[test]
fn every_index_is_pinned() {
    let p = kir_ptx();
    let lines: Vec<&str> = p.lines().collect();
    let step_lines = steps(&p);
    assert_eq!(step_lines.len(), 2, "the two loop steps");
    let adds: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.contains("add.u64 ")).map(|(n, _)| n).collect();
    for (i, n) in adds.iter().enumerate() {
        if !step_lines.contains(n) {
            assert!(caught(|p| drop_op(p, "add.u64 ", "u64", i)), "add {i}");
        }
    }
    for i in 0..p.matches("sub.u64 ").count() {
        assert!(caught(|p| drop_op(p, "sub.u64 ", "u64", i)), "sub {i}");
    }
    let muls: Vec<&str> = lines.iter().copied().filter(|l| l.contains("mul.lo.u64 ")).collect();
    for (i, l) in muls.iter().enumerate() {
        if l.ends_with(", 4;") || l.ends_with(", 8;") {
            let (size, other) = if l.ends_with(", 4;") { (", 4;", ", 8;") } else { (", 8;", ", 4;") };
            assert!(caught(|p| edit(p, "mul.lo.u64 ", i, |l| l.replacen(size, other, 1))), "element size {i}: {l}");
        } else {
            assert!(caught(|p| drop_op(p, "mul.lo.u64 ", "u64", i)), "index multiply {i}: {l}");
        }
    }
    let ones: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.trim().starts_with("mov.u64 ") && l.ends_with(", 1;")).map(|(n, _)| n).collect();
    assert_eq!(ones.len(), 2, "the loop steps' 1s");
    for at in ones {
        let mutate = |p: &str| {
            p.lines().enumerate().map(|(n, l)| if n == at { l.replacen(", 1;", ", 2;", 1) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
        };
        assert!(caught(mutate), "step at line {at}");
    }
    let zeros = lines.iter().filter(|l| l.trim().starts_with("mov.u64 ") && l.ends_with(", 0;")).count();
    for i in 0..zeros {
        assert!(caught(|p| nudge(p, ", 0;", ", 1;", i)), "start {i}");
    }
}

/// The tie test, both `selp`s, the running max's start, and the argmax's
/// 64-bit store.
#[test]
fn the_update_is_pinned() {
    assert!(caught(|p| nudge(p, "setp.le.f32 ", "setp.lt.f32 ", 0)), "ties keep the first tap");
    assert!(caught(|p| nudge(p, "setp.le.f32 ", "setp.ge.f32 ", 0)), "the comparison's direction");
    for i in 0..2 {
        assert!(caught(|p| edit(p, "selp.", i, |l| {
            let (head, ops) = l.split_once(' ').expect("selp");
            let ops: Vec<&str> = ops.trim().trim_end_matches(';').split(',').map(str::trim).collect();
            format!("{head} {}, {}, {}, {};", ops[0], ops[2], ops[1], ops[3])
        })), "selp {i} swapped");
    }
    assert!(caught(|p| p.replacen("0fFF800000", "0f00000000", 1)), "the max's start");
    assert!(caught(|p| p.replacen("st.global.u64", "st.global.u32", 1)), "the argmax's width");
}
