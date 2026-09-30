//! The differential equivalence gate for `nsl_conv2d_f32` from
//! `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5), now built by
//! `nsl_kir::kernels::conv2d`.
//!
//! This file runs the frozen hand module (`tests/fixtures/conv2d_hand.rs`)
//! and the KIR one side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`) over whole grids, with a block more
//! than the work needs, on the runtime's 256-thread block and a 32-thread
//! one, under two schedules:
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory.
//! 2. **Correctness**: every output is the hand kernel's walk restated in
//!    Rust: taps in `ci`, `ky`, `kx` order, padding and out-of-range taps
//!    skipped, the accumulator from `0` updated by a fused multiply-add
//!    (`f32::mul_add`, one rounding, as `fma.rn`), then `bias[co]` added
//!    when the bias is not null. Nothing past the outputs is written and
//!    the inputs are untouched.
//! 3. **The gate bites**: every bound and padding test, every `div`/`rem` of
//!    the index split, every index multiply, add and subtract, the loop
//!    starts and steps, the `fma`'s accumulator and its fusion, the bias
//!    add and its null test, and every element size.
//!
//! The shapes differ in stride and padding between the two axes (so a
//! swapped `stride_h`/`stride_w` or `pad_h`/`pad_w` shows), cover strides
//! below, at and past the window, a window wholly in the padding, one to
//! three input channels, and more outputs than a block. Every shape runs
//! with a bias and with a null one. The data has full mantissas, so a
//! separately rounded multiply and add differ from the `fma`.

use std::collections::HashMap;

use nsl_kir::kernels::conv2d::{ptx, PARAM_NAMES};
use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;

#[allow(dead_code)]
#[path = "fixtures/conv2d_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 16;
const POISON: u32 = 0x7FA5_A5A5;
const INP: u64 = 0x1000_0000;
const WT: u64 = 0x2000_0000;
const BIAS: u64 = 0x3000_0000;
const OUT: u64 = 0x4000_0000;

fn kir_ptx() -> String {
    String::from_utf8(ptx()).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx() -> String {
    hand::CONV2D_F32_PTX.trim_end_matches('\0').to_string()
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

/// `(N, C_in, H, W, C_out, kH, kW, stride_h, stride_w, pad_h, pad_w)`.
#[derive(Clone, Copy, Debug)]
struct Shape {
    n: usize,
    ci: usize,
    h: usize,
    w: usize,
    co: usize,
    kh: usize,
    kw: usize,
    sh: usize,
    sw: usize,
    ph: usize,
    pw: usize,
}

impl Shape {
    fn h_out(&self) -> usize {
        (self.h + 2 * self.ph - self.kh) / self.sh + 1
    }
    fn w_out(&self) -> usize {
        (self.w + 2 * self.pw - self.kw) / self.sw + 1
    }
    fn total(&self) -> usize {
        self.n * self.co * self.h_out() * self.w_out()
    }
    fn inp_len(&self) -> usize {
        self.n * self.ci * self.h * self.w
    }
    fn wt_len(&self) -> usize {
        self.co * self.ci * self.kh * self.kw
    }
}

#[allow(clippy::too_many_arguments)]
const fn shape(n: usize, ci: usize, h: usize, w: usize, co: usize, kh: usize, kw: usize, sh: usize, sw: usize, ph: usize, pw: usize) -> Shape {
    Shape { n, ci, h, w, co, kh, kw, sh, sw, ph, pw }
}

const SHAPES: [Shape; 6] = [
    shape(1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0),
    shape(2, 3, 5, 7, 2, 2, 3, 2, 1, 0, 1),
    shape(1, 2, 6, 6, 3, 3, 3, 1, 2, 1, 2),
    shape(2, 1, 9, 4, 2, 3, 2, 2, 1, 1, 0),
    shape(1, 1, 2, 3, 1, 2, 2, 1, 1, 2, 2),
    shape(2, 2, 16, 16, 3, 3, 3, 2, 2, 1, 1),
];

/// Full-mantissa values in `[-2, 2)`.
fn data(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n).map(|_| ((lcg(&mut s) as f32 / (1u64 << 31) as f32) * 4.0 - 2.0).to_bits()).collect()
}

struct Case {
    s: Shape,
    x: Vec<u32>,
    wt: Vec<u32>,
    /// `None`: the bias pointer is null.
    bias: Option<Vec<u32>>,
}

fn cases() -> Vec<Case> {
    SHAPES
        .iter()
        .enumerate()
        .flat_map(|(i, &s)| {
            let seed = 11 + 3 * i as u64;
            let (x, wt, bias) = (data(s.inp_len(), seed), data(s.wt_len(), seed + 1), data(s.co, seed + 2));
            [Case { s, x: x.clone(), wt: wt.clone(), bias: Some(bias) }, Case { s, x, wt, bias: None }]
        })
        .collect()
}

const BLOCKS: [u32; 2] = [ELEMENTWISE_BLOCK, 32];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

/// `(inp, wt, bias, out + tail)` as words, after the launch.
fn run(ptx: &str, c: &Case, block: u32, order: Order) -> Vec<Vec<u32>> {
    let prog = parse(ptx);
    let s = c.s;
    let total = s.total();
    let mut global = vec![
        Segment { base: INP, bytes: le32(&c.x) },
        Segment { base: WT, bytes: le32(&c.wt) },
        Segment { base: BIAS, bytes: le32(c.bias.as_deref().unwrap_or(&[])) },
        Segment { base: OUT, bytes: le32(&vec![POISON; total + TAIL]) },
    ];
    let values = [
        INP,
        WT,
        if c.bias.is_some() { BIAS } else { 0 },
        OUT,
        s.n as u64,
        s.ci as u64,
        s.h as u64,
        s.w as u64,
        s.co as u64,
        s.kh as u64,
        s.kw as u64,
        s.sh as u64,
        s.sw as u64,
        s.ph as u64,
        s.pw as u64,
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
    global.iter().map(|g| words(&g.bytes)).collect()
}

#[test]
fn the_kernels_agree() {
    let (hand, kir) = (hand_ptx(), kir_ptx());
    for c in cases() {
        for block in BLOCKS {
            for order in ORDERS {
                assert!(run(&hand, &c, block, order) == run(&kir, &c, block, order), "{:?} bias {} {block} {order:?}", c.s, c.bias.is_some());
            }
        }
    }
}

/// The hand kernel's walk, restated; `fused` picks `fma` or a separately
/// rounded multiply and add.
fn reference(c: &Case, fused: bool) -> Vec<u32> {
    let s = c.s;
    let (h_out, w_out) = (s.h_out(), s.w_out());
    let f = |v: &[u32], k: usize| f32::from_bits(v[k]);
    (0..s.total())
        .map(|i| {
            let (ow, oh) = (i % w_out, (i / w_out) % h_out);
            let (co, n) = ((i / w_out / h_out) % s.co, i / w_out / h_out / s.co);
            let mut acc = 0.0f32;
            for ci in 0..s.ci {
                for ky in 0..s.kh {
                    for kx in 0..s.kw {
                        let (ih0, iw0) = (oh * s.sh + ky, ow * s.sw + kx);
                        if ih0 < s.ph || iw0 < s.pw {
                            continue;
                        }
                        let (ih, iw) = (ih0 - s.ph, iw0 - s.pw);
                        if ih >= s.h || iw >= s.w {
                            continue;
                        }
                        let x = f(&c.x, ((n * s.ci + ci) * s.h + ih) * s.w + iw);
                        let wv = f(&c.wt, ((co * s.ci + ci) * s.kh + ky) * s.kw + kx);
                        acc = if fused { x.mul_add(wv, acc) } else { x * wv + acc };
                    }
                }
            }
            if let Some(b) = &c.bias {
                acc += f(b, co);
            }
            acc.to_bits()
        })
        .collect()
}

#[test]
fn the_kernels_are_the_convolution() {
    for which in [hand_ptx(), kir_ptx()] {
        for c in cases() {
            for block in BLOCKS {
                let g = run(&which, &c, block, Order::Ascending);
                let total = c.s.total();
                assert_eq!(g[0], c.x, "{:?}: input untouched", c.s);
                assert_eq!(g[1], c.wt, "{:?}: weight untouched", c.s);
                assert_eq!(g[2], c.bias.clone().unwrap_or_default(), "{:?}: bias untouched", c.s);
                assert_eq!(&g[3][..total], &reference(&c, true)[..], "{:?} bias {} {block}", c.s, c.bias.is_some());
                assert!(g[3][total..].iter().all(|&w| w == POISON), "{:?}: wrote past the outputs", c.s);
            }
        }
    }
}

/// The data reaches the cases the walk distinguishes: a window wholly in
/// the padding (an output of exactly the bias, or `0`), a fused sum that a
/// separately rounded one misses, and more outputs than a block.
#[test]
fn the_data_covers_every_case() {
    let cs = cases();
    assert!(cs.iter().any(|c| c.s.total() > ELEMENTWISE_BLOCK as usize), "more outputs than a block");
    assert!(cs.iter().any(|c| c.bias.is_none() && reference(c, true).contains(&0)), "a window wholly in the padding");
    assert!(cs.iter().any(|c| reference(c, true) != reference(c, false)), "the fma's single rounding shows");
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

/// The operands of the `i`-th line containing `op`, and the line rebuilt
/// from a permutation of them.
fn permute(ptx: &str, op: &str, i: usize, order: &[usize]) -> String {
    edit(ptx, op, i, |l| {
        let (head, operands) = l.split_once(op).expect("the op");
        let ops: Vec<&str> = operands.trim_end_matches(';').split(',').map(str::trim).collect();
        let picked: Vec<&str> = order.iter().map(|&k| ops[k]).collect();
        format!("{head}{op}{};", picked.join(", "))
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

/// Every bound (the thread's, the three loops', both input edges), both
/// padding tests, and the null test.
#[test]
fn every_bound_padding_and_null_test_is_pinned() {
    let p = kir_ptx();
    assert_eq!(p.matches("setp.ge.u64 ").count(), 6);
    for i in 0..6 {
        assert!(caught(|p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "bound {i}");
    }
    assert_eq!(p.matches("setp.lt.u64 ").count(), 2);
    for i in 0..2 {
        assert!(caught(|p| nudge(p, "setp.lt.u64 ", "setp.le.u64 ", i)), "padding test {i}");
    }
    assert!(caught(|p| nudge(p, "setp.eq.u64 ", "setp.ne.u64 ", 0)), "the null test");
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
/// dropped step would never end), each loop's start and step, the null
/// constant, and every element size.
#[test]
fn every_index_is_pinned() {
    let p = kir_ptx();
    let lines: Vec<&str> = p.lines().collect();
    let step_lines = steps(&p);
    assert_eq!(step_lines.len(), 3, "the three loop steps");
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
        if l.ends_with(", 4;") {
            assert!(caught(|p| edit(p, "mul.lo.u64 ", i, |l| l.replacen(", 4;", ", 8;", 1))), "element size {i}: {l}");
        } else {
            assert!(caught(|p| drop_op(p, "mul.lo.u64 ", "u64", i)), "index multiply {i}: {l}");
        }
    }
    let ones: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.trim().starts_with("mov.u64 ") && l.ends_with(", 1;")).map(|(n, _)| n).collect();
    assert_eq!(ones.len(), 3, "the loop steps' 1s");
    for at in ones {
        let mutate = |p: &str| {
            p.lines().enumerate().map(|(n, l)| if n == at { l.replacen(", 1;", ", 2;", 1) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
        };
        assert!(caught(mutate), "step at line {at}");
    }
    let zeros = lines.iter().filter(|l| l.trim().starts_with("mov.u64 ") && l.ends_with(", 0;")).count();
    assert_eq!(zeros, 4, "three loop starts and the null pointer");
    for i in 0..zeros {
        assert!(caught(|p| nudge(p, ", 0;", ", 1;", i)), "start {i}");
    }
}

/// The accumulator: its start, the `fma`'s addend, its fusion (a
/// separately rounded `mul` then `add` differs), and the bias add.
/// Swapping the `fma`'s two factors is an equivalent mutant (the product
/// commutes), so it is not claimed.
#[test]
fn the_accumulation_is_pinned() {
    assert!(caught(|p| nudge(p, ", 0f00000000;", ", 0f3F800000;", 0)), "the accumulator's start");
    assert!(caught(|p| permute(p, "fma.rn.f32 ", 0, &[0, 3, 2, 1])), "the fma's addend");
    assert!(caught(|p| edit(p, "fma.rn.f32 ", 0, |l| {
        let (head, operands) = l.split_once("fma.rn.f32 ").expect("fma");
        let ops: Vec<&str> = operands.trim_end_matches(';').split(',').map(str::trim).collect();
        format!("{head}mul.rn.f32 {d}, {a}, {b};\n{head}add.rn.f32 {d}, {d}, {c};", d = ops[0], a = ops[1], b = ops[2], c = ops[3])
    })), "the fma's single rounding");
    assert!(caught(|p| drop_op(p, "add.f32 ", "f32", 0)), "the bias add");
    assert!(!caught(|p| permute(p, "fma.rn.f32 ", 0, &[0, 2, 1, 3])), "the factors commute");
}
