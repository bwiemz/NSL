//! The differential equivalence gate for the shared-memory tree reductions
//! from `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5):
//! `nsl_global_sum_f32`, `nsl_sum_dim_f32` and `nsl_max_dim_f32`, now built
//! by `nsl_kir::kernels::block_reduce`.
//!
//! This file runs the frozen hand modules (`tests/fixtures/block_reduce_hand.rs`)
//! and the KIR ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), on the runtime's 256-thread block,
//! with one block more than the work needs, under all four thread schedules
//! and both block orders:
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory, over empty, single-element and ragged extents longer than the
//!    block, with order-sensitive data, signed zeros, and (for the max) NaNs.
//! 2. **Correctness**: each result is the hand kernels' order restated in
//!    Rust, bit for bit: thread `k` folds `k, k + 256, …` from the identity,
//!    then the tree folds `s[k] ⊕= s[k + h]` for `h = 128, …, 1`. Nothing
//!    past `out` is written and the input is untouched.
//! 3. **The gate bites**: every bound, the block index, the quotient and
//!    remainder, every element size, 64-bit add and multiply, the stride,
//!    the tree's first half and halving, the partner index, both barriers,
//!    both combines and the identity are caught. Two mutants are named as
//!    equivalent: every thread writing the result (after the last barrier
//!    each stores the same `s[0]`), and the max's stride halved (each value
//!    is then read twice, and the max is idempotent).

use std::collections::HashMap;

use nsl_kir::kernels::block_reduce::{ptx, BlockReduceOp, BLOCK_REDUCE_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/block_reduce_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 64;
const POISON: u32 = 0x7FA5_A5A5;
const INP: u64 = 0x1000_0000;
const OUT: u64 = 0x2000_0000;
const B: usize = BLOCK_REDUCE_BLOCK as usize;

fn kir_ptx(op: BlockReduceOp) -> String {
    String::from_utf8(ptx(op)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(op: BlockReduceOp) -> String {
    match op {
        BlockReduceOp::GlobalSum => hand::GLOBAL_SUM_F32_PTX,
        BlockReduceOp::SumDim => hand::SUM_DIM_F32_PTX,
        BlockReduceOp::MaxDim => hand::MAX_DIM_F32_PTX,
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

/// Values whose sums depend on the order of the adds: magnitudes from
/// `2^-12` to `2^12`, both signs, some signed zeros, and (`nan`) a few NaNs.
fn data(n: usize, seed: u64, nan: bool) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| match lcg(&mut s) % 40 {
            0 => 0.0f32.to_bits(),
            1 => (-0.0f32).to_bits(),
            2 if nan => f32::NAN.to_bits(),
            _ => {
                let e = (lcg(&mut s) % 25) as i32 - 12;
                let mant = 1.0 + (lcg(&mut s) % 4096) as f32 / 4096.0;
                let sign = if lcg(&mut s).is_multiple_of(2) { 1.0 } else { -1.0 };
                (sign * mant * 2f32.powi(e)).to_bits()
            }
        })
        .collect()
}

/// `(outer, reduce_size, inner)`, or `(1, n, 1)` for the global sum.
struct Case {
    dims: (usize, usize, usize),
    inp: Vec<u32>,
}

const DIM_SHAPES: [(usize, usize, usize); 7] = [(1, 1, 1), (2, 5, 3), (3, 0, 2), (0, 4, 2), (3, 300, 2), (1, 700, 1), (2, 257, 3)];
const GLOBAL_LENS: [usize; 6] = [0, 1, 255, 256, 257, 1000];

fn cases(op: BlockReduceOp) -> Vec<Case> {
    let nan = op == BlockReduceOp::MaxDim;
    let mut v: Vec<Case> = match op {
        BlockReduceOp::GlobalSum => {
            GLOBAL_LENS.iter().enumerate().map(|(i, &n)| Case { dims: (1, n, 1), inp: data(n, 3 + i as u64, false) }).collect()
        }
        _ => DIM_SHAPES
            .iter()
            .enumerate()
            .map(|(i, &(o, r, n))| Case { dims: (o, r, n), inp: data(o * r * n, 11 + i as u64, nan) })
            .collect(),
    };
    // Every value `-0.0`: the sum's `+0.0` start shows, and every value
    // negative: the max's `-inf` start shows only when the extent is empty.
    let n = if op == BlockReduceOp::GlobalSum { (1, 300, 1) } else { (2, 300, 1) };
    v.push(Case { dims: n, inp: vec![(-0.0f32).to_bits(); n.0 * n.1 * n.2] });
    // One spike per output, at the last reduced position: only the second
    // trip of the stride reaches it, so a max that skips it shows.
    let s = if op == BlockReduceOp::GlobalSum { (1, 300, 1) } else { (2, 300, 3) };
    let mut inp = vec![(-1.0f32).to_bits(); s.0 * s.1 * s.2];
    for o in 0..s.0 {
        for i in 0..s.2 {
            inp[(o * s.1 + s.1 - 1) * s.2 + i] = (5.0 + (o * s.2 + i) as f32).to_bits();
        }
    }
    v.push(Case { dims: s, inp });
    v
}

fn outputs(op: BlockReduceOp, c: &Case) -> usize {
    if op == BlockReduceOp::GlobalSum {
        1
    } else {
        c.dims.0 * c.dims.2
    }
}

/// All of global memory after the launch: `[inp, out + tail]`.
fn run(op: BlockReduceOp, ptx: &str, c: &Case, order: Order, reverse_ctas: bool) -> Vec<u32> {
    let prog = parse(ptx);
    let n_out = outputs(op, c);
    let mut global = vec![
        Segment { base: INP, bytes: le32(&c.inp) },
        Segment { base: OUT, bytes: le32(&vec![POISON; n_out + TAIL]) },
    ];
    let (o, r, i) = c.dims;
    let vals: Vec<u64> = match op {
        BlockReduceOp::GlobalSum => vec![INP, OUT, r as u64],
        _ => vec![INP, OUT, o as u64, r as u64, i as u64],
    };
    let args: HashMap<String, u64> = op.param_names().iter().zip(vals).map(|(p, v)| (p.to_string(), v)).collect();
    let grid = if op == BlockReduceOp::GlobalSum { 1 } else { n_out as u32 + 1 };
    let mut ctas: Vec<u32> = (0..grid).collect();
    if reverse_ctas {
        ctas.reverse();
    }
    for ctaid in ctas {
        let mut l = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![0; B * 4],
            ctaid,
            ctaid_y: 0,
            nctaid_x: 0, nctaid_y: 1,
            ntid: BLOCK_REDUCE_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    global.iter().flat_map(|s| words(&s.bytes)).collect()
}

const ORDERS: [Order; 4] = [Order::Ascending, Order::Descending, Order::WarpsAscending, Order::WarpsDescending];

#[test]
fn the_kernels_agree() {
    for op in BlockReduceOp::ALL {
        let (hand, kir) = (hand_ptx(op), kir_ptx(op));
        for c in cases(op) {
            for order in ORDERS {
                let rev = order == Order::Descending;
                assert!(run(op, &hand, &c, order, rev) == run(op, &kir, &c, order, rev), "{op:?} {:?} {order:?}", c.dims);
            }
        }
    }
}

/// The hand kernels' order, restated.
fn reference(op: BlockReduceOp, c: &Case) -> Vec<u32> {
    let (_, reduce, inner) = c.dims;
    let comb = |a: f32, b: f32| if op == BlockReduceOp::MaxDim { a.max(b) } else { a + b };
    let init = if op == BlockReduceOp::MaxDim { f32::NEG_INFINITY } else { 0.0 };
    (0..outputs(op, c))
        .map(|t| {
            let (o, i) = if op == BlockReduceOp::GlobalSum { (0, 0) } else { (t / inner, t % inner) };
            let mut s: Vec<f32> = (0..B)
                .map(|k| {
                    (k..reduce).step_by(B).fold(init, |acc, r| comb(acc, f32::from_bits(c.inp[(o * reduce + r) * inner + i])))
                })
                .collect();
            let mut h = B / 2;
            while h >= 1 {
                for k in 0..h {
                    s[k] = comb(s[k], s[k + h]);
                }
                h /= 2;
            }
            s[0].to_bits()
        })
        .collect()
}

#[test]
fn the_kernels_are_the_tree_reduction() {
    for op in BlockReduceOp::ALL {
        for which in [hand_ptx(op), kir_ptx(op)] {
            for c in cases(op) {
                let got = run(op, &which, &c, Order::Ascending, false);
                let (inp, out) = got.split_at(c.inp.len());
                let n_out = outputs(op, &c);
                assert_eq!(inp, &c.inp[..], "{op:?} {:?}: input untouched", c.dims);
                let want = reference(op, &c);
                let same = out[..n_out].iter().zip(&want).all(|(g, w)| g == w || (f32::from_bits(*g).is_nan() && f32::from_bits(*w).is_nan()));
                assert!(same, "{op:?} {:?}", c.dims);
                assert!(out[n_out..].iter().all(|&w| w == POISON), "{op:?} {:?}: wrote past out", c.dims);
            }
        }
    }
}

/// The data reaches the cases the kernels distinguish: more than one trip
/// of the stride, an order-sensitive sum, and NaNs for the max.
#[test]
fn the_data_covers_every_case() {
    assert!(DIM_SHAPES.iter().any(|d| d.1 > 2 * B) && GLOBAL_LENS.iter().any(|&n| n > 2 * B));
    assert!(DIM_SHAPES.iter().any(|d| d.1 == 0) && DIM_SHAPES.iter().any(|d| d.0 == 0));
    let c = &cases(BlockReduceOp::SumDim)[5];
    let fwd = c.inp.iter().fold(0.0f32, |a, &w| a + f32::from_bits(w));
    let back = c.inp.iter().rev().fold(0.0f32, |a, &w| a + f32::from_bits(w));
    assert_ne!(fwd.to_bits(), back.to_bits(), "the sum depends on order");
    assert!(cases(BlockReduceOp::MaxDim).iter().any(|c| c.inp.iter().any(|&w| f32::from_bits(w).is_nan())));
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on any case or
/// schedule (or faults).
fn caught(op: BlockReduceOp, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(op);
    let mutant = mutate(&kir_ptx(op));
    cases(op).iter().any(|c| {
        ORDERS.into_iter().any(|order| {
            [false, true].into_iter().any(|rev| {
                let expect = run(op, &hand, c, order, rev);
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(op, &mutant, c, order, rev))) {
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

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for op in BlockReduceOp::ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

/// The loop bound and, per dim, the block bound.
#[test]
fn relaxing_a_bound_is_caught() {
    for op in BlockReduceOp::ALL {
        let bounds = kir_ptx(op).matches("setp.ge.u64 ").count();
        assert_eq!(bounds, if op == BlockReduceOp::GlobalSum { 1 } else { 2 }, "{op:?}");
        for i in 0..bounds {
            assert!(caught(op, |p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "{op:?} bound {i}");
        }
    }
}

/// The block index, the output's quotient and remainder.
#[test]
fn the_output_index_is_pinned() {
    for op in [BlockReduceOp::SumDim, BlockReduceOp::MaxDim] {
        assert!(caught(op, |p| p.replacen("%ctaid.x", "0", 1)), "{op:?}");
        assert!(caught(op, |p| p.replacen("div.u64 ", "rem.u64 ", 1)), "{op:?} quotient");
        assert!(caught(op, |p| p.replacen("rem.u64 ", "div.u64 ", 1)), "{op:?} remainder");
    }
}

/// Every element size, 64-bit add and 64-bit multiply (the extent, the
/// base, the first offset and the step).
#[test]
fn every_address_is_pinned() {
    for op in BlockReduceOp::ALL {
        let p = kir_ptx(op);
        let sizes: Vec<usize> =
            p.lines().enumerate().filter(|(_, l)| l.contains("mul.lo.u64") && l.ends_with(", 4;")).map(|(n, _)| n).collect();
        let want = if op == BlockReduceOp::GlobalSum { 5 } else { 6 };
        assert_eq!(sizes.len(), want, "{op:?}: the global load, the partial's and the tree's shared accesses, and out[t]");
        for at in sizes {
            let mutate = |p: &str| {
                p.lines().enumerate().map(|(n, l)| if n == at { l.replacen(", 4;", ", 8;", 1) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
            };
            assert!(caught(op, mutate), "{op:?} element size at line {at}");
        }
        let arith = p.lines().filter(|l| l.contains("mul.lo.u64 ") && !l.ends_with(", 4;")).count();
        assert_eq!(arith, if op == BlockReduceOp::GlobalSum { 0 } else { 5 }, "{op:?}: extent, base ×2, first offset, step");
        let muls = p.matches("mul.lo.u64 ").count();
        for j in 0..muls {
            let line = p.lines().filter(|l| l.contains("mul.lo.u64 ")).nth(j).expect("a multiply");
            if line.ends_with(", 4;") {
                continue;
            }
            assert!(caught(op, |p| drop_op(p, "mul.lo.u64 ", "u64", j)), "{op:?} mul.lo.u64 {j}: {line}");
        }
        let adds = p.matches("add.u64 ").count();
        for i in 0..adds {
            assert!(caught(op, |p| drop_op(p, "add.u64 ", "u64", i)), "{op:?} add {i}");
        }
    }
}

/// The stride, the tree's first half and halving, the pairwise index and
/// both barriers.
#[test]
fn the_tree_is_pinned() {
    for op in BlockReduceOp::ALL {
        if op != BlockReduceOp::MaxDim {
            assert!(caught(op, |p| p.replacen(", 256;", ", 128;", 1)), "{op:?} stride");
        }
        assert!(caught(op, |p| p.replacen(", 128;", ", 64;", 1)), "{op:?} first half");
        assert!(caught(op, |p| drop_op(p, "shr.u32 ", "u32", 0)), "{op:?} halving");
        assert!(caught(op, |p| drop_op(p, "add.u32 ", "u32", 0)), "{op:?} partner");
        assert_eq!(kir_ptx(op).matches("bar.sync 0;").count(), 2);
        for i in 0..2 {
            assert!(caught(op, |p| edit(p, "bar.sync 0;", i, |_| String::new())), "{op:?} barrier {i}");
        }
    }
}

/// Named equivalent mutant: the max's stride halved. Each element is then
/// read by two threads, and `max.f32` is idempotent and order-free (a NaN
/// only survives if every value is NaN), so the result cannot change. The
/// sums, which are neither, catch it.
#[test]
fn the_max_stride_halved_is_an_equivalent_mutant() {
    assert!(!caught(BlockReduceOp::MaxDim, |p| p.replacen(", 256;", ", 128;", 1)));
}

/// Named equivalent mutant: every thread writing the result. The write
/// follows the tree's last barrier, so each thread stores the same final
/// `s[0]` to the same `out` element; only thread 0 doing so is a saving,
/// not a difference the bytes can show.
#[test]
fn every_thread_writing_is_an_equivalent_mutant() {
    for op in BlockReduceOp::ALL {
        assert!(!caught(op, |p| nudge(p, "setp.ne.u32 ", "setp.eq.u32 ", 0)), "{op:?}");
    }
}

/// Both combines (the loop's and the tree's) and the identity.
#[test]
fn the_combine_and_identity_are_pinned() {
    for op in BlockReduceOp::ALL {
        let (comb, other) = if op == BlockReduceOp::MaxDim { ("max.f32 ", "min.f32 ") } else { ("add.rn.f32 ", "sub.rn.f32 ") };
        assert_eq!(kir_ptx(op).matches(comb).count(), 2, "{op:?}");
        for i in 0..2 {
            assert!(caught(op, |p| nudge(p, comb, other, i)), "{op:?} combine {i}");
        }
        let (init, wrong) = if op == BlockReduceOp::MaxDim { ("0fFF800000", "0f00000000") } else { ("0f00000000", "0f80000000") };
        assert!(caught(op, |p| p.replacen(init, wrong, 1)), "{op:?} identity");
        assert!(caught(op, |p| p.replacen(init, "0f3F800000", 1)), "{op:?} identity as 1");
    }
}
