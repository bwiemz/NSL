//! The differential equivalence gate for the f64-accumulated sum of squares
//! from `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5):
//! `nsl_sum_sq_f64_acc_f32`, now built by `nsl_kir::kernels::sum_sq`.
//!
//! This file runs the frozen hand module (`tests/fixtures/sum_sq_hand.rs`) and
//! the KIR one side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`). Each launch is a whole grid of
//! 256-thread blocks with `%nctaid.x` set, so the grid stride is real. The
//! grids include the one the runtime derives from the length
//! (`clamp(ceil(n / 256), 1, 256)`), a single block, and grids that do not
//! divide the work. They run under all four thread schedules and both block
//! orders.
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory over empty, short and ragged extents, with values spanning
//!    `2^-140` (subnormal) to `2^100` and both signs.
//! 2. **Correctness**: block `b`'s partial is its threads' f64 folds,
//!    `acc = fma(x, x, acc)` over `x = inp[k]` for `k = b·256 + t, k + s, …`,
//!    combined by the shared-memory tree, bit for bit. Nothing past the
//!    partials is written and the input is untouched.
//! 3. **The gate bites**: the bound, the stride and both of its factors, the
//!    start's block index, the output's block index, the widening, the fused
//!    square-accumulate, the tree's half, halving and add, both barriers,
//!    the identity, every element size and every 64-bit add are caught.

use std::collections::HashMap;

use nsl_kir::kernels::sum_sq::{ptx, PARAM_NAMES, SUM_SQ_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/sum_sq_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 16;
const POISON: u64 = 0x7FF5_A5A5_A5A5_A5A5;
const INP: u64 = 0x1000_0000;
const OUT: u64 = 0x2000_0000;
const B: usize = SUM_SQ_BLOCK as usize;

fn kir_ptx() -> String {
    String::from_utf8(ptx()).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx() -> String {
    hand::SUM_SQ_F64_ACC_F32_PTX.trim_end_matches('\0').to_string()
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    *s >> 33
}

/// f32 values from `2^-140` (subnormal) to `2^100`, both signs, with zeros:
/// their f64 squares span far more than f32 could hold, and the sum depends
/// on the order of the adds.
fn data(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| match lcg(&mut s) % 32 {
            0 => 0.0f32.to_bits(),
            1 => (-0.0f32).to_bits(),
            2 => 0x0000_0200, // 2^-140, a subnormal (`powi` would flush it to 0)
            3 => (-(2f32.powi(100))).to_bits(),
            _ => {
                let e = (lcg(&mut s) % 61) as i32 - 30;
                let mant = 1.0 + (lcg(&mut s) % 8192) as f32 / 8192.0;
                let sign = if lcg(&mut s).is_multiple_of(2) { 1.0 } else { -1.0 };
                (sign * mant * 2f32.powi(e)).to_bits()
            }
        })
        .collect()
}

/// The runtime's grid for a length (`gpu_sum_sq_many_f32`).
fn runtime_grid(n: usize) -> u32 {
    n.div_ceil(B).clamp(1, 256) as u32
}

/// `(n, grid)`.
fn cases() -> Vec<(usize, u32)> {
    let mut v = vec![];
    for n in [0usize, 1, 255, 256, 257, 1000, 3000] {
        v.push((n, runtime_grid(n)));
        v.push((n, 1));
        v.push((n, 3));
    }
    v
}

fn words(b: &[u8]) -> Vec<u64> {
    b.chunks(8).map(|c| u64::from_le_bytes(c.try_into().expect("8 bytes"))).collect()
}

/// All of global memory after the launch, as 8-byte words:
/// `[inp (packed f32 pairs), out + tail]`.
fn run(ptx: &str, n: usize, grid: u32, order: Order, reverse: bool) -> Vec<u64> {
    let prog = parse(ptx);
    let mut inp: Vec<u8> = data(n, 7 + n as u64).iter().flat_map(|w| w.to_le_bytes()).collect();
    inp.resize(inp.len().next_multiple_of(8).max(8), 0);
    let out: Vec<u8> = std::iter::repeat_n(POISON, grid as usize + TAIL).flat_map(|w| w.to_le_bytes()).collect();
    let mut global = vec![Segment { base: INP, bytes: inp }, Segment { base: OUT, bytes: out }];
    let args: HashMap<String, u64> = PARAM_NAMES.iter().zip([INP, OUT, n as u64]).map(|(p, v)| (p.to_string(), v)).collect();
    let mut ctas: Vec<u32> = (0..grid).collect();
    if reverse {
        ctas.reverse();
    }
    for ctaid in ctas {
        let mut l = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![0; B * 8],
            ctaid,
            ctaid_y: 0,
            nctaid_x: grid,
            nctaid_y: 1,
            ntid: SUM_SQ_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    global.iter().flat_map(|s| words(&s.bytes)).collect()
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

/// The hand kernel's order, restated: each thread's fused f64 fold over its
/// grid-strided elements, then the tree.
fn reference(n: usize, grid: u32) -> Vec<u64> {
    let x: Vec<f64> = data(n, 7 + n as u64).iter().map(|&w| f32::from_bits(w) as f64).collect();
    let stride = B * grid as usize;
    (0..grid as usize)
        .map(|b| {
            let mut s: Vec<f64> =
                (0..B).map(|t| (b * B + t..n).step_by(stride).fold(0.0f64, |acc, k| x[k].mul_add(x[k], acc))).collect();
            let mut h = B / 2;
            while h >= 1 {
                for t in 0..h {
                    s[t] += s[t + h];
                }
                h /= 2;
            }
            s[0].to_bits()
        })
        .collect()
}

#[test]
fn the_partials_are_the_tree_of_fused_folds() {
    for which in [hand_ptx(), kir_ptx()] {
        for (n, grid) in cases() {
            let got = run(&which, n, grid, Order::Ascending, false);
            let inp_words = (n * 4).next_multiple_of(8).max(8) / 8;
            let (inp, out) = got.split_at(inp_words);
            let data: Vec<u64> = {
                let mut b: Vec<u8> = data(n, 7 + n as u64).iter().flat_map(|w| w.to_le_bytes()).collect();
                b.resize(inp_words * 8, 0);
                words(&b)
            };
            assert_eq!(inp, &data[..], "n={n}: input untouched");
            assert_eq!(&out[..grid as usize], &reference(n, grid)[..], "n={n} grid={grid}");
            assert!(out[grid as usize..].iter().all(|&w| w == POISON), "n={n} grid={grid}: wrote past the partials");
        }
    }
}

/// The data exercises what f64 accumulation is for: squares outside f32's
/// range, and a total whose f32 sum would differ.
#[test]
fn the_data_needs_f64() {
    let x: Vec<f32> = data(3000, 3007).iter().map(|&w| f32::from_bits(w)).collect();
    assert!(x.iter().any(|v| (*v as f64).powi(2) > f32::MAX as f64), "a square past f32's range");
    assert!(x.iter().any(|v| *v != 0.0 && ((*v as f64) * (*v as f64)) < f32::MIN_POSITIVE as f64), "a square below f32's normals");
    assert!(cases().iter().any(|&(n, g)| n > B * g as usize), "a thread folds more than one element");
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

#[test]
fn the_unmutated_kernel_is_not_caught() {
    assert!(!caught(|p| p.to_string()));
}

/// The bound, and the grid stride: both factors and the product.
#[test]
fn the_bound_and_stride_are_pinned() {
    assert_eq!(kir_ptx().matches("setp.ge.u64 ").count(), 1);
    assert!(caught(|p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", 0)), "bound");
    assert!(caught(|p| p.replacen("%nctaid.x", "1", 1)), "grid width");
    assert!(caught(|p| nudge(p, "mul.lo.u32 %r", "add.u32 %r", 0)), "stride product");
    assert!(caught(|p| drop_op(p, "mul.lo.u32 ", "u32", 1)), "stride's grid factor");
}

/// Both block indices: the start's and the output slot's.
#[test]
fn the_block_indices_are_pinned() {
    assert_eq!(kir_ptx().matches("%ctaid.x").count(), 2);
    for i in 0..2 {
        assert!(caught(|p| edit(p, "%ctaid.x", i, |l| l.replacen("%ctaid.x", "0", 1))), "ctaid {i}");
    }
}

/// The widening, the fused square-accumulate, the tree's add and the
/// identity.
#[test]
fn the_arithmetic_is_pinned() {
    assert!(caught(|p| edit(p, "fma.rn.f64 ", 0, |l| {
        let (head, ops) = l.split_once("fma.rn.f64 ").expect("fma");
        let ops: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
        format!("{head}add.rn.f64 {}, {}, {};", ops[0], ops[1], ops[3])
    })), "square");
    assert!(caught(|p| edit(p, "fma.rn.f64 ", 0, |l| {
        let (head, ops) = l.split_once("fma.rn.f64 ").expect("fma");
        let ops: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
        format!("{head}fma.rn.f64 {}, {}, {}, {};", ops[0], ops[1], ops[1], ops[1])
    })), "accumulate");
    assert!(caught(|p| drop_op(p, "add.rn.f64 ", "f64", 0)), "tree add");
    assert!(caught(|p| p.replacen("0d0000000000000000", "0d3FF0000000000000", 1)), "identity");
    assert!(caught(|p| edit(p, "cvt.f64.f32", 0, |l| {
        let (head, ops) = l.split_once("cvt.f64.f32 ").expect("cvt");
        let ops: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
        format!("{head}mov.b64 {}, 0;", ops[0])
    })), "widening");
}

/// The tree: its first half, halving, partner index and both barriers.
#[test]
fn the_tree_is_pinned() {
    assert!(caught(|p| p.replacen(", 128;", ", 64;", 1)), "first half");
    assert!(caught(|p| drop_op(p, "shr.u32 ", "u32", 0)), "halving");
    let adds = kir_ptx().matches("add.u32 ").count();
    for i in 0..adds {
        assert!(caught(|p| drop_op(p, "add.u32 ", "u32", i)), "add.u32 {i}");
    }
    assert_eq!(kir_ptx().matches("bar.sync 0;").count(), 2);
    for i in 0..2 {
        assert!(caught(|p| edit(p, "bar.sync 0;", i, |_| String::new())), "barrier {i}");
    }
}

/// Every element size (4 for the input, 8 for the f64 slots) and every
/// 64-bit add.
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
    assert_eq!((fours, sizes.len() - fours), (1, 5), "input; shared ×4 and the output slot");
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
}
