//! The differential equivalence gate for the deterministic sum kernels from
//! `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5):
//! `nsl_det_global_sum_f32` and `nsl_det_sum_dim_f32`, now built by
//! `nsl_kir::kernels::det_sum`.
//!
//! This file runs the frozen hand modules (`tests/fixtures/det_sum_hand.rs`)
//! and the KIR ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), with the runtime's launch shape: one
//! one-thread block for the global sum, and one one-thread block per output
//! for the per-dim sum, plus two blocks past the end, run in ascending and in
//! descending block order:
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory, over empty, single-element and ragged extents, with inputs
//!    whose sum depends on the order they are added in, signed zeros, and
//!    an infinity.
//! 2. **Correctness**: each result is the ascending left-to-right f32 sum
//!    from `+0.0`, bit for bit; nothing past the output is written.
//! 3. **The gate bites**: each bound, the block index, the output's split
//!    into `(o, i)`, every element size, every 64-bit add and multiply, the
//!    step, the accumulator's start and the add itself are caught.

use std::collections::HashMap;

use nsl_kir::kernels::det_sum::{ptx, DetSumOp, DET_SUM_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/det_sum_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 64;
const POISON: u32 = 0x7FA5_A5A5;
const INP: u64 = 0x1000_0000;
const OUT: u64 = 0x2000_0000;

fn kir_ptx(op: DetSumOp) -> String {
    String::from_utf8(ptx(op)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(op: DetSumOp) -> String {
    match op {
        DetSumOp::Global => hand::DET_GLOBAL_SUM_F32_PTX,
        DetSumOp::Dim => hand::DET_SUM_DIM_F32_PTX,
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

/// The input: magnitudes from `2^-12` to `2^12` of both signs, so that
/// adding in any other order, or skipping or repeating an element, changes
/// the rounded sum; `-0.0` and, for `seed == 3`, an infinity at element 2.
/// Past the input, `TAIL` words of a large value a stray read would add.
fn input(len: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    let mut v: Vec<u32> = (0..len)
        .map(|n| {
            if n % 11 == 5 {
                return (-0.0f32).to_bits();
            }
            let e = (lcg(&mut s) % 25) as i32 - 12;
            let m = 1.0 + (lcg(&mut s) % 1024) as f32 / 1024.0;
            let sign = if lcg(&mut s).is_multiple_of(2) { 1.0 } else { -1.0 };
            (sign * m * 2f32.powi(e)).to_bits()
        })
        .collect();
    if seed == 3 && len > 2 {
        v[2] = f32::INFINITY.to_bits();
    }
    v.extend(std::iter::repeat_n(3.0e6f32.to_bits(), TAIL));
    v
}

/// `(outer, reduce_size, inner)`. For the global sum, the length is the
/// product.
const SHAPES: [(usize, usize, usize); 8] =
    [(1, 0, 1), (1, 1, 1), (2, 0, 3), (1, 17, 1), (3, 1, 5), (2, 3, 4), (4, 9, 3), (5, 13, 2)];

/// All-`-0.0` input: a sum from `-0.0` would stay `-0.0`, one from `+0.0`
/// does not.
const NEG_ZERO_SEED: u64 = 99;

fn data(len: usize, seed: u64) -> Vec<u32> {
    if seed == NEG_ZERO_SEED {
        let mut v = vec![(-0.0f32).to_bits(); len];
        v.extend(std::iter::repeat_n(3.0e6f32.to_bits(), TAIL));
        v
    } else {
        input(len, seed)
    }
}

const SEEDS: [u64; 3] = [1, 3, NEG_ZERO_SEED];

struct Run {
    inp: Vec<u32>,
    out: Vec<u32>,
}

fn outputs(op: DetSumOp, (outer, _, inner): (usize, usize, usize)) -> usize {
    match op {
        DetSumOp::Global => 1,
        DetSumOp::Dim => outer * inner,
    }
}

fn run(op: DetSumOp, ptx: &str, shape: (usize, usize, usize), seed: u64, order: Order) -> Run {
    let (outer, reduce, inner) = shape;
    let inp = data(outer * reduce * inner, seed);
    let n_out = outputs(op, shape);
    let prog = parse(ptx);
    let mut global = vec![
        Segment { base: INP, bytes: le32(&inp) },
        Segment { base: OUT, bytes: le32(&vec![POISON; n_out + TAIL]) },
    ];
    let values: Vec<u64> = match op {
        DetSumOp::Global => vec![INP, OUT, (outer * reduce * inner) as u64],
        DetSumOp::Dim => vec![INP, OUT, outer as u64, reduce as u64, inner as u64],
    };
    let args: HashMap<String, u64> = op.param_names().iter().zip(values).map(|(n, v)| (n.to_string(), v)).collect();
    let grid = match op {
        DetSumOp::Global => 1,
        DetSumOp::Dim => n_out as u32 + 2,
    };
    let mut ctas: Vec<u32> = (0..grid).collect();
    if order == Order::Descending {
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
            nctaid_y: 1,
            ntid: DET_SUM_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    Run { inp, out: words(&global[1].bytes) }
}

const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

/// The ascending f32 sum from `+0.0`.
fn sum(xs: impl Iterator<Item = u32>) -> u32 {
    xs.fold(0.0f32, |acc, x| acc + f32::from_bits(x)).to_bits()
}

#[test]
fn the_kernels_agree_and_are_the_ascending_sum() {
    for op in DetSumOp::ALL {
        let (hand, kir) = (hand_ptx(op), kir_ptx(op));
        for shape in SHAPES {
            let (outer, reduce, inner) = shape;
            for seed in SEEDS {
                for order in ORDERS {
                    let h = run(op, &hand, shape, seed, order);
                    let q = run(op, &kir, shape, seed, order);
                    assert!(h.out == q.out, "{op:?} {shape:?} seed {seed} {order:?}: outputs differ");
                    assert!(h.inp == q.inp);
                }
                let r = run(op, &kir, shape, seed, Order::Ascending);
                let n_out = outputs(op, shape);
                for t in 0..n_out {
                    let want = match op {
                        DetSumOp::Global => sum(r.inp[..outer * reduce * inner].iter().copied()),
                        DetSumOp::Dim => {
                            let (o, i) = (t / inner, t % inner);
                            sum((0..reduce).map(|k| r.inp[(o * reduce + k) * inner + i]))
                        }
                    };
                    assert_eq!(r.out[t], want, "{op:?} {shape:?} seed {seed} output {t}");
                }
                assert!(r.out[n_out..].iter().all(|&w| w == POISON), "{op:?} {shape:?}: wrote past the output");
            }
        }
    }
}

/// The data is order-sensitive: the ascending sum differs from the
/// descending one on some case, so a reordered loop would not pass.
#[test]
fn the_data_tells_the_order_apart() {
    let differs = SHAPES.iter().any(|&(o, r, i)| {
        let v = input(o * r * i, 1);
        let n = o * r * i;
        sum(v[..n].iter().copied()) != sum(v[..n].iter().rev().copied())
    });
    assert!(differs);
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on any case,
/// under either block order (or faults).
fn caught(op: DetSumOp, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(op);
    let mutant = mutate(&kir_ptx(op));
    SHAPES.iter().any(|&shape| {
        SEEDS.iter().any(|&seed| {
            ORDERS.into_iter().any(|order| {
                let expect = run(op, &hand, shape, seed, order).out;
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(op, &mutant, shape, seed, order))) {
                    Ok(r) => r.out != expect,
                    Err(_) => true,
                }
            })
        })
    })
}

/// `ptx` with the `i`-th line containing `from` rewritten by `f`.
fn edit(ptx: &str, from: &str, i: usize, f: impl Fn(&str) -> String) -> String {
    let at = ptx
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains(from))
        .nth(i)
        .unwrap_or_else(|| panic!("no line {i} with `{from}`"))
        .0;
    ptx.lines().enumerate().map(|(n, l)| if n == at { f(l) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
}

fn nudge(ptx: &str, from: &str, to: &str, i: usize) -> String {
    edit(ptx, from, i, |l| l.replacen(from, to, 1))
}

/// `op.u64 d, a, b;` as `mov.u64 d, a;`: the operation dropped.
fn drop_op(ptx: &str, op: &str, i: usize) -> String {
    edit(ptx, op, i, |l| {
        let (head, operands) = l.split_once(op).expect("the op");
        let mut ops: Vec<&str> = operands.trim_end_matches(';').split(',').map(str::trim).collect();
        ops.truncate(2);
        format!("{head}mov.u64 {};", ops.join(", "))
    })
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for op in DetSumOp::ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

/// The loop bound and, for the per-dim sum, the output bound.
#[test]
fn relaxing_a_bound_is_caught() {
    for op in DetSumOp::ALL {
        let bounds = if op == DetSumOp::Dim { 2 } else { 1 };
        assert_eq!(kir_ptx(op).matches("setp.ge.u64 ").count(), bounds, "{op:?}");
        for i in 0..bounds {
            assert!(caught(op, |p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "{op:?} bound {i}");
        }
    }
}

/// The per-dim sum's output is its block's: `%ctaid.x`, split into
/// `o = t / inner` and `i = t % inner`.
#[test]
fn the_output_index_is_caught() {
    let op = DetSumOp::Dim;
    assert!(caught(op, |p| p.replacen("%ctaid.x;", "0;", 1)), "ctaid.x");
    assert!(caught(op, |p| nudge(p, "div.u64 ", "rem.u64 ", 0)), "the quotient");
    assert!(caught(op, |p| nudge(p, "rem.u64 ", "div.u64 ", 0)), "the remainder");
}

/// Every address's element size: the input and, for the per-dim sum, the
/// output.
#[test]
fn nudging_an_element_size_is_caught() {
    for op in DetSumOp::ALL {
        let p = kir_ptx(op);
        let sizes = p.lines().filter(|l| l.contains("mul.lo.u64") && l.ends_with(", 4;")).count();
        assert_eq!(sizes, if op == DetSumOp::Dim { 2 } else { 1 }, "{op:?}");
        for i in 0..sizes {
            let size_line = |p: &str| {
                let at = p.lines().enumerate().filter(|(_, l)| l.contains("mul.lo.u64") && l.ends_with(", 4;")).nth(i).expect("site").0;
                p.lines()
                    .enumerate()
                    .map(|(n, l)| if n == at { l.replacen(", 4;", ", 8;", 1) } else { l.to_string() })
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n"
            };
            assert!(caught(op, size_line), "{op:?} element size {i}");
        }
    }
}

/// Every 64-bit add (the addresses, the base, the step) and every multiply
/// that is not an element size (the output count, the base's strides, the
/// row stride) carries weight.
#[test]
fn dropping_an_add_or_a_stride_is_caught() {
    for op in DetSumOp::ALL {
        let p = kir_ptx(op);
        let adds = p.matches("add.u64 ").count();
        assert_eq!(adds, if op == DetSumOp::Dim { 5 } else { 2 }, "{op:?}");
        for i in 0..adds {
            assert!(caught(op, |p| drop_op(p, "add.u64 ", i)), "{op:?} add {i}");
        }
        let strides: Vec<usize> = p
            .lines()
            .filter(|l| l.contains("mul.lo.u64"))
            .enumerate()
            .filter(|(_, l)| !l.ends_with(", 4;"))
            .map(|(n, _)| n)
            .collect();
        assert_eq!(strides.len(), if op == DetSumOp::Dim { 4 } else { 0 }, "{op:?}");
        for i in strides {
            assert!(caught(op, |p| drop_op(p, "mul.lo.u64 ", i)), "{op:?} stride {i}");
        }
    }
}

/// The step is one, the accumulator starts at `+0.0`, and the add is an
/// add.
#[test]
fn the_accumulation_is_pinned() {
    for op in DetSumOp::ALL {
        let p = kir_ptx(op);
        assert_eq!(p.matches(", 1;").count(), 1, "{op:?}: the step is the only `1`");
        assert!(caught(op, |p| p.replacen(", 1;", ", 2;", 1)), "{op:?} step");
        assert_eq!(p.matches("0f00000000;").count(), 1, "{op:?}");
        assert!(caught(op, |p| p.replacen("0f00000000;", "0f80000000;", 1)), "{op:?} -0.0 start");
        assert!(caught(op, |p| p.replacen("0f00000000;", "0f3F800000;", 1)), "{op:?} 1.0 start");
        assert!(caught(op, |p| p.replacen("add.rn.f32 ", "sub.rn.f32 ", 1)), "{op:?} add");
        assert!(caught(op, |p| p.replacen("add.rn.f32 ", "mul.rn.f32 ", 1)), "{op:?} add as mul");
    }
}
