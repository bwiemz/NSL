//! The differential equivalence gate for the flash-attention log-sum-exp
//! kernels from `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5):
//! `nsl_flash_lse_f32` and `nsl_flash_lse_gqa_f32`, now built by
//! `nsl_kir::kernels::flash_lse`.
//!
//! This file runs the frozen hand modules (`tests/fixtures/flash_lse_hand.rs`)
//! and the KIR ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`) over whole grids, with a block more
//! than the work needs, on the runtime's 256-thread block and a 32-thread
//! one, under two schedules, causal and not:
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory. The interpreter rounds each PTX `mul.f32` and `add.f32` on its
//!    own, as the PTX text says; ptxas contracted the hand kernels' pairs
//!    into `FFMA` on hardware, and the KIR kernels' `.rn` keeps them apart
//!    there too (the SASS comparison is in the KIR spec).
//! 2. **Correctness**: every `lse` is `compute_logsumexp_gqa`'s two passes
//!    restated: the dot product summed from `+0` in `d` order, the score
//!    scaled, the max from `-inf` replaced when `score > max`, then `max +
//!    log2(Σ 2^((score - max) · log2 e)) · ln 2`, every step rounded on its
//!    own, `exp2`/`log2` as the interpreter's `ex2`/`lg2`. The GQA kernel
//!    reads kv-head `(bh / heads) · kv_heads + (bh % heads) / (heads /
//!    kv_heads)`. Nothing past `total` is written and Q and K are
//!    untouched.
//! 3. **The gate bites**: every bound, the causal test and its `+ 1`, both
//!    `selp`s, the max's comparison, every `div`/`rem` of the index split and
//!    the GQA head map, every index multiply and add, the loop starts and
//!    steps, every element size, the accumulators' starts, every float
//!    multiply, add and subtract, the `ex2` and `lg2`, both constants, and
//!    the dot product's and the final sum's two roundings (fusing either
//!    into an `fma` shows).
//!
//! The shapes cover one to eight Q heads per kv-head, head dims of 1 to 8,
//! sequences of 1 to 40, more rows than a block, and each launch runs
//! causal and not. The data has full mantissas, with a row of NaN and a
//! row of `+inf`.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;
use nsl_kir::kernels::flash_lse::{ptx, LseOp, LN_2_BITS, LOG2_E_BITS};

#[allow(dead_code)]
#[path = "fixtures/flash_lse_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 16;
const POISON: u32 = 0x7FA5_A5A5;
const Q: u64 = 0x1000_0000;
const K: u64 = 0x2000_0000;
const LSE: u64 = 0x3000_0000;

fn kir_ptx(op: LseOp) -> String {
    String::from_utf8(ptx(op)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(op: LseOp) -> String {
    match op {
        LseOp::Mha => hand::FLASH_LSE_F32_PTX,
        LseOp::Gqa => hand::FLASH_LSE_GQA_F32_PTX,
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

/// `(batch, heads, kv_heads, seq, hd)`; the MHA kernel runs only the shapes
/// with `kv_heads == heads`.
#[derive(Clone, Copy, Debug)]
struct Shape {
    b: usize,
    h: usize,
    kvh: usize,
    s: usize,
    hd: usize,
}

impl Shape {
    fn total(&self) -> usize {
        self.b * self.h * self.s
    }
}

const fn shape(b: usize, h: usize, kvh: usize, s: usize, hd: usize) -> Shape {
    Shape { b, h, kvh, s, hd }
}

const SHAPES: [Shape; 7] = [
    shape(1, 1, 1, 1, 1),
    shape(2, 2, 2, 5, 3),
    shape(1, 4, 2, 7, 4),
    shape(2, 6, 2, 9, 8),
    shape(1, 8, 1, 40, 5),
    shape(1, 3, 3, 6, 2),
    shape(2, 4, 4, 40, 2),
];

struct Case {
    s: Shape,
    q: Vec<u32>,
    k: Vec<u32>,
    scale: f32,
    causal: bool,
}

/// Full-mantissa values in `[-2, 2)`; the first shape with more than one
/// row gets a NaN in one Q row and an `inf` in another.
fn data(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n).map(|_| ((lcg(&mut s) as f32 / (1u64 << 31) as f32) * 4.0 - 2.0).to_bits()).collect()
}

fn cases(op: LseOp) -> Vec<Case> {
    let mut out = vec![];
    for (i, &s) in SHAPES.iter().enumerate() {
        if op == LseOp::Mha && s.kvh != s.h {
            continue;
        }
        let kv = if op == LseOp::Mha { s.h } else { s.kvh };
        let mut q = data(s.b * s.h * s.s * s.hd, 5 + 7 * i as u64);
        let k = data(s.b * kv * s.s * s.hd, 6 + 7 * i as u64);
        if i == 1 {
            q[0] = f32::NAN.to_bits();
            q[s.hd] = f32::INFINITY.to_bits();
        }
        let scale = 1.0 / (s.hd as f32).sqrt();
        for causal in [false, true] {
            out.push(Case { s, q: q.clone(), k: k.clone(), scale, causal });
        }
    }
    out
}

const BLOCKS: [u32; 2] = [ELEMENTWISE_BLOCK, 32];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

/// `(q, k, lse + tail)` as words, after the launch.
fn run(op: LseOp, ptx: &str, c: &Case, block: u32, order: Order) -> Vec<Vec<u32>> {
    let prog = parse(ptx);
    let s = c.s;
    let total = s.total();
    let mut global = vec![
        Segment { base: Q, bytes: le32(&c.q) },
        Segment { base: K, bytes: le32(&c.k) },
        Segment { base: LSE, bytes: le32(&vec![POISON; total + TAIL]) },
    ];
    let mut values = vec![
        Q,
        K,
        LSE,
        total as u64,
        s.s as u64,
        s.hd as u64,
        u64::from(c.scale.to_bits()),
        u64::from(c.causal),
    ];
    if op == LseOp::Gqa {
        values.extend([s.h as u64, s.kvh as u64]);
    }
    let args: HashMap<String, u64> = op.param_names().iter().zip(values).map(|(p, v)| (p.to_string(), v)).collect();
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
    for op in LseOp::ALL {
        let (hand, kir) = (hand_ptx(op), kir_ptx(op));
        for c in cases(op) {
            for block in BLOCKS {
                for order in ORDERS {
                    assert!(run(op, &hand, &c, block, order) == run(op, &kir, &c, block, order), "{op:?} {:?} causal {} {block} {order:?}", c.s, c.causal);
                }
            }
        }
    }
}

/// `compute_logsumexp_gqa`'s passes, restated; `fused` sums the dot
/// product with `mul_add` instead.
fn reference(op: LseOp, c: &Case, fused: bool) -> Vec<u32> {
    let s = c.s;
    let kv = if op == LseOp::Mha { s.h } else { s.kvh };
    let groups = s.h / kv;
    let f = |v: &[u32], i: usize| f32::from_bits(v[i]);
    (0..s.total())
        .map(|i| {
            let (qi, bh) = (i % s.s, i / s.s);
            let kbh = (bh / s.h) * kv + (bh % s.h) / groups;
            let n_keys = if c.causal { qi + 1 } else { s.s };
            let score = |j: usize| {
                let mut acc = 0.0f32;
                for d in 0..s.hd {
                    let (x, y) = (f(&c.q, i * s.hd + d), f(&c.k, (kbh * s.s + j) * s.hd + d));
                    acc = if fused { x.mul_add(y, acc) } else { acc + x * y };
                }
                acc * c.scale
            };
            let mut max = f32::NEG_INFINITY;
            for j in 0..n_keys {
                let sc = score(j);
                if sc > max {
                    max = sc;
                }
            }
            let mut sum = 0.0f32;
            for j in 0..n_keys {
                sum += ((score(j) - max) * f32::from_bits(LOG2_E_BITS)).exp2();
            }
            (max + sum.log2() * f32::from_bits(LN_2_BITS)).to_bits()
        })
        .collect()
}

#[test]
fn the_kernels_are_the_two_passes() {
    for op in LseOp::ALL {
        for which in [hand_ptx(op), kir_ptx(op)] {
            for c in cases(op) {
                for block in BLOCKS {
                    let g = run(op, &which, &c, block, Order::Ascending);
                    let total = c.s.total();
                    assert_eq!(g[0], c.q, "{op:?} {:?}: q untouched", c.s);
                    assert_eq!(g[1], c.k, "{op:?} {:?}: k untouched", c.s);
                    assert_eq!(&g[2][..total], &reference(op, &c, false)[..], "{op:?} {:?} causal {} {block}", c.s, c.causal);
                    assert!(g[2][total..].iter().all(|&w| w == POISON), "{op:?} {:?}: wrote past the rows", c.s);
                }
            }
        }
    }
}

/// The data reaches what the passes distinguish: more rows than a block, a
/// NaN row, an `inf` row, finite rows, and dot products whose fused sum
/// differs from the separately rounded one.
#[test]
fn the_data_covers_every_case() {
    for op in LseOp::ALL {
        let cs = cases(op);
        assert!(cs.iter().any(|c| c.s.total() > ELEMENTWISE_BLOCK as usize), "{op:?}: more rows than a block");
        let all: Vec<u32> = cs.iter().flat_map(|c| reference(op, c, false)).collect();
        assert!(all.iter().any(|&w| f32::from_bits(w).is_nan()), "{op:?}: a NaN row");
        assert!(all.iter().filter(|&&w| f32::from_bits(w).is_finite()).count() > 100, "{op:?}: finite rows");
        assert!(cs.iter().any(|c| reference(op, c, false) != reference(op, c, true)), "{op:?}: the dot product's rounding shows");
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on any case,
/// block or schedule (or faults).
fn caught(op: LseOp, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(op);
    let mutant = mutate(&kir_ptx(op));
    cases(op).iter().any(|c| {
        BLOCKS.into_iter().any(|block| {
            ORDERS.into_iter().any(|order| {
                let expect = run(op, &hand, c, block, order);
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(op, &mutant, c, block, order))) {
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

fn operands(l: &str, op: &str) -> (String, Vec<String>) {
    let (head, rest) = l.split_once(op).expect("the op");
    (head.to_string(), rest.trim_end_matches(';').split(',').map(|s| s.trim().to_string()).collect())
}

/// `op d, a, b;` as `mov.<ty> d, a;`: the operation dropped.
fn drop_op(ptx: &str, op: &str, ty: &str, i: usize) -> String {
    edit(ptx, op, i, |l| {
        let (head, ops) = operands(l, op);
        format!("{head}mov.{ty} {}, {};", ops[0], ops[1])
    })
}

/// The `i`-th `mul.rn.f32 t, a, b;` that the next line adds (`add.rn.f32
/// d, c, t;`), fused into `fma.rn.f32 d, a, b, c;`.
fn fuse(ptx: &str, i: usize) -> String {
    let lines: Vec<&str> = ptx.lines().collect();
    let pairs: Vec<usize> = (0..lines.len() - 1)
        .filter(|&n| {
            lines[n].contains("mul.rn.f32 ") && lines[n + 1].contains("add.rn.f32 ") && {
                let (_, m) = operands(lines[n], "mul.rn.f32 ");
                let (_, a) = operands(lines[n + 1], "add.rn.f32 ");
                a[2] == m[0]
            }
        })
        .collect();
    let at = pairs[i];
    let (head, m) = operands(lines[at], "mul.rn.f32 ");
    let (_, a) = operands(lines[at + 1], "add.rn.f32 ");
    let mut out: Vec<String> = lines.iter().map(|l| l.to_string()).collect();
    out[at] = format!("{head}fma.rn.f32 {}, {}, {}, {};", a[0], m[1], m[2], a[1]);
    out.remove(at + 1);
    out.join("\n") + "\n"
}

/// The line numbers of the `mov.u64 _, 1;` lines.
fn one_lines(p: &str) -> Vec<usize> {
    p.lines().enumerate().filter(|(_, l)| l.trim().starts_with("mov.u64 ") && l.ends_with(", 1;")).map(|(n, _)| n).collect()
}

/// The line numbers of the `add.u64`s whose last operand is the register
/// the latest `mov.u64 _, 1;` set: the loop steps and the causal `+ 1`,
/// which the `1` nudges pin (a dropped step would never end).
fn plus_one_adds(p: &str) -> Vec<usize> {
    let mut one = String::new();
    let mut out = vec![];
    for (n, l) in p.lines().enumerate() {
        let t = l.trim();
        if t.starts_with("mov.u64 ") && t.ends_with(", 1;") {
            one = t["mov.u64 ".len()..].split(',').next().unwrap_or("").to_string();
        } else if t.starts_with("add.u64 ") && t.trim_end_matches(';').rsplit(", ").next() == Some(one.as_str()) {
            out.push(n);
        }
    }
    out
}

fn on_line(p: &str, at: usize, g: impl Fn(&str) -> String) -> String {
    p.lines().enumerate().map(|(n, l)| if n == at { g(l) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for op in LseOp::ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

/// Every bound (the thread's, both key loops', both dot loops'), the
/// causal test, the causal `+ 1`, both `selp`s, and the max's comparison.
#[test]
fn every_bound_and_selection_is_pinned() {
    for op in LseOp::ALL {
        let p = kir_ptx(op);
        assert_eq!(p.matches("setp.ge.u64 ").count(), 5, "{op:?}");
        for i in 0..5 {
            assert!(caught(op, |p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "{op:?} bound {i}");
        }
        assert!(caught(op, |p| nudge(p, "setp.ne.u64 ", "setp.eq.u64 ", 0)), "{op:?} the causal test");
        assert!(caught(op, |p| nudge(p, "setp.gt.f32 ", "setp.lt.f32 ", 0)), "{op:?} the max's direction");
        for i in 0..2 {
            assert!(caught(op, |p| edit(p, "selp.", i, |l| {
                let (head, ops) = l.split_once(' ').expect("selp");
                let ops: Vec<&str> = ops.trim().trim_end_matches(';').split(',').map(str::trim).collect();
                format!("{head} {}, {}, {}, {};", ops[0], ops[2], ops[1], ops[3])
            })), "{op:?} selp {i} swapped");
        }
    }
}

/// The index split and the GQA head map: each `div` and `rem`, swapped for
/// the other and dropped.
#[test]
fn the_index_split_is_pinned() {
    for op in LseOp::ALL {
        let p = kir_ptx(op);
        for (name, other) in [("div.u64 ", "rem.u64 "), ("rem.u64 ", "div.u64 ")] {
            for i in 0..p.matches(name).count() {
                assert!(caught(op, |p| nudge(p, name, other, i)), "{op:?} {name} {i} swapped");
                assert!(caught(op, |p| drop_op(p, name, "u64", i)), "{op:?} {name} {i} dropped");
            }
        }
    }
}

/// Every index multiply and add, every `1` (the causal `+ 1`, the steps)
/// and `0` (the starts, the causal test's zero), and every element size. A
/// loop step's add is nudged through its `1` rather than dropped: a
/// dropped step would never end.
#[test]
fn every_index_is_pinned() {
    for op in LseOp::ALL {
        let p = kir_ptx(op);
        let lines: Vec<&str> = p.lines().collect();
        for (i, l) in lines.iter().filter(|l| l.contains("mul.lo.u64 ")).enumerate() {
            if l.ends_with(", 4;") {
                assert!(caught(op, |p| edit(p, "mul.lo.u64 ", i, |l| l.replacen(", 4;", ", 8;", 1))), "{op:?} element size {i}: {l}");
            } else {
                assert!(caught(op, |p| drop_op(p, "mul.lo.u64 ", "u64", i)), "{op:?} index multiply {i}: {l}");
            }
        }
        let ones = one_lines(&p);
        let plus_one = plus_one_adds(&p);
        for (i, (n, l)) in lines.iter().enumerate().filter(|(_, l)| l.contains("add.u64 ")).enumerate() {
            if !plus_one.contains(&n) {
                assert!(caught(op, |p| drop_op(p, "add.u64 ", "u64", i)), "{op:?} add {i}: {l}");
            }
        }
        for at in ones {
            assert!(caught(op, |p| on_line(p, at, |l| l.replacen(", 1;", ", 2;", 1))), "{op:?} the 1 at line {at}");
        }
        let zeros = lines.iter().filter(|l| l.trim().starts_with("mov.u64 ") && l.ends_with(", 0;")).count();
        for i in 0..zeros {
            assert!(caught(op, |p| nudge(p, ", 0;", ", 1;", i)), "{op:?} the 0 {i}");
        }
    }
}

/// Every float multiply, add and subtract (dropped), both starts, the `ex2`
/// and `lg2` (dropped), both constants, and the roundings of the dot
/// product and the final sum (fused into an `fma`).
#[test]
fn the_arithmetic_is_pinned() {
    for op in LseOp::ALL {
        let p = kir_ptx(op);
        for name in ["mul.rn.f32 ", "add.rn.f32 ", "sub.rn.f32 "] {
            for i in 0..p.matches(name).count() {
                assert!(caught(op, |p| drop_op(p, name, "f32", i)), "{op:?} {name} {i} dropped");
            }
        }
        for name in ["ex2.approx.f32 ", "lg2.approx.f32 "] {
            assert!(caught(op, |p| edit(p, name, 0, |l| l.replacen(name, "mov.f32 ", 1))), "{op:?} {name} dropped");
        }
        assert!(caught(op, |p| p.replacen("0fFF800000", "0f00000000", 1)), "{op:?} the max's start");
        for i in 0..p.matches(", 0f00000000;").count() {
            assert!(caught(op, |p| nudge(p, ", 0f00000000;", ", 0f3F800000;", i)), "{op:?} the 0.0 {i}");
        }
        assert!(caught(op, |p| p.replacen("0f3FB8AA3B", "0f3FB8AA3C", 1)), "{op:?} log2 e");
        assert!(caught(op, |p| p.replacen("0f3F317218", "0f3F317219", 1)), "{op:?} ln 2");
        for i in 0..2 {
            assert!(caught(op, |p| fuse(p, i)), "{op:?} dot product {i} fused");
        }
        assert!(caught(op, |p| fuse(p, 2)), "{op:?} lg2 · ln 2 + max fused");
    }
}
