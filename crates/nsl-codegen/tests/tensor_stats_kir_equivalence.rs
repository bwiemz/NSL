//! The differential equivalence gate for `nsl_tensor_stats_f32` from
//! `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5), now built by
//! `nsl_kir::kernels::tensor_stats`.
//!
//! This file runs the frozen hand module (`tests/fixtures/tensor_stats_hand.rs`)
//! and the KIR one side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), as the runtime launches it (one block
//! of 256 threads), under all four thread schedules:
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory. The extents are empty, single-element, and ragged extents longer
//!    than the block. The data is order-sensitive with full 24-bit
//!    significands, and includes signed zeros, NaNs, infinities, a subnormal,
//!    and values whose squares leave f32's range at both ends.
//! 2. **Correctness**: `out[0..4]` is the hand kernel's order restated in
//!    Rust, bit for bit. Thread `k` folds `k, k + 256, …` from `(+inf, -inf,
//!    +0, +0)`, squaring and adding with two roundings. The tree then folds
//!    `s[k] ⊕= s[k + h]` for `h = 128, …, 1`. Nothing past `out[3]` is written
//!    and the input is untouched.
//! 3. **The gate bites**:
//!    - the bound, the stride, the tree's first half, halving, idle test and
//!      partner index, and both barriers;
//!    - every shared region's offset, element size, 64-bit add and output
//!      slot;
//!    - every combine, the square, and each identity;
//!    - the square fused into its accumulate (`fma.rn.f32`), which is what
//!      ptxas made of the hand kernel on hardware. The KIR kernel rounds
//!      twice so that it cannot.
//!
//!    Three mutants are named as equivalent. Every thread writing the result
//!    is one: after the last barrier each stores the same `s[0]`. The other
//!    two are the first output slot's element size and its address add,
//!    because its offset is `0 · 4`.

use std::collections::HashMap;

use nsl_kir::kernels::tensor_stats::{ptx, PARAM_NAMES, TENSOR_STATS_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/tensor_stats_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 64;
const POISON: u32 = 0x7FA5_A5A5;
const INP: u64 = 0x1000_0000;
const OUT: u64 = 0x2000_0000;
const B: usize = TENSOR_STATS_BLOCK as usize;

fn kir_ptx() -> String {
    String::from_utf8(ptx()).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx() -> String {
    hand::TENSOR_STATS_F32_PTX.trim_end_matches('\0').to_string()
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

/// Values whose sums depend on the order of the adds, and whose squares are
/// inexact: full 24-bit significands, magnitudes from `2^-12` to `2^12`, both
/// signs, some signed zeros, and (`nan`) a few NaNs.
fn data(n: usize, seed: u64, nan: bool) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| match lcg(&mut s) % 40 {
            0 => 0.0f32.to_bits(),
            1 => (-0.0f32).to_bits(),
            2 if nan => f32::NAN.to_bits(),
            _ => {
                let e = (lcg(&mut s) % 25) as i32 - 12;
                let mant = 1.0 + (lcg(&mut s) % (1 << 23)) as f32 / (1u32 << 23) as f32;
                let sign = if lcg(&mut s).is_multiple_of(2) { 1.0 } else { -1.0 };
                (sign * mant * 2f32.powi(e)).to_bits()
            }
        })
        .collect()
}

const LENS: [usize; 7] = [0, 1, 255, 256, 257, 1000, 3000];

fn cases() -> Vec<Vec<u32>> {
    let mut v: Vec<Vec<u32>> = LENS.iter().enumerate().map(|(i, &n)| data(n, 3 + i as u64, false)).collect();
    v.push(data(700, 41, true));
    // Every value `-0.0`: the sums' `+0.0` start shows.
    v.push(vec![(-0.0f32).to_bits(); 300]);
    // All positive, all negative, shorter and longer than the block: the
    // min's `+inf` and the max's `-inf` starts show only here.
    for (n, sign) in [(100, 1.0f32), (300, 1.0), (100, -1.0), (300, -1.0)] {
        v.push(data(n, 50 + n as u64, false).into_iter().map(|w| (sign * f32::from_bits(w).abs()).to_bits()).collect());
    }
    // Extremes: squares that overflow (`2^70`) and underflow (`2^-80`, a
    // subnormal), and the infinities, among ordinary values.
    let mut x = data(600, 77, false);
    for (i, w) in [2f32.powi(70), -(2f32.powi(-80)), f32::from_bits(0x0000_0200), f32::INFINITY, f32::NEG_INFINITY]
        .into_iter()
        .enumerate()
    {
        x[97 * i + 13] = w.to_bits();
    }
    v.push(x);
    // One spike past the first trip of the stride, for the min and max.
    let mut y = vec![1.0f32.to_bits(); 300];
    y[299] = 9.0f32.to_bits();
    y[298] = (-9.0f32).to_bits();
    v.push(y);
    // Thread 0 alone holds two values, `2^-12` then `1 + 2^-12`, and every
    // other value is zero, so its partial is the sum of squares. Rounded
    // twice, `2^-24 + round(1 + 2^-11 + 2^-24)` ties to `1 + 2^-11`; fused,
    // `2^-24 + 1 + 2^-11 + 2^-24` is `1 + 2^-11 + 2^-23` exactly. On wider
    // data the tree absorbs such a one-ulp difference in a partial.
    let mut z = vec![0.0f32.to_bits(); 257];
    z[0] = 2f32.powi(-12).to_bits();
    z[256] = (1.0 + 2f32.powi(-12)).to_bits();
    v.push(z);
    v
}

/// All of global memory after the launch: `[inp, out + tail]`.
fn run(ptx: &str, inp: &[u32], order: Order) -> Vec<u32> {
    let prog = parse(ptx);
    let mut global = vec![
        Segment { base: INP, bytes: le32(inp) },
        Segment { base: OUT, bytes: le32(&vec![POISON; 4 + TAIL]) },
    ];
    let args: HashMap<String, u64> =
        PARAM_NAMES.iter().zip([INP, OUT, inp.len() as u64]).map(|(p, v)| (p.to_string(), v)).collect();
    let mut l = Launch {
        prog: &prog,
        args: &args,
        global: &mut global,
        shared: vec![0; B * 16],
        ctaid: 0,
        ctaid_y: 0,
        nctaid_x: 0, nctaid_y: 1,
        ntid: TENSOR_STATS_BLOCK,
        steps: 0,
    };
    run_cta(&mut l, order);
    global.iter().flat_map(|s| words(&s.bytes)).collect()
}

const ORDERS: [Order; 4] = [Order::Ascending, Order::Descending, Order::WarpsAscending, Order::WarpsDescending];

#[test]
fn the_kernels_agree() {
    let (hand, kir) = (hand_ptx(), kir_ptx());
    for inp in cases() {
        for order in ORDERS {
            assert!(run(&hand, &inp, order) == run(&kir, &inp, order), "n = {} {order:?}", inp.len());
        }
    }
}

/// The hand kernel's order, restated: `[min, max, Σx, Σx²]`.
fn reference(inp: &[u32]) -> [u32; 4] {
    reference_with(inp, false)
}

/// [`reference`], or with the square fused into its accumulate (`fused`).
fn reference_with(inp: &[u32], fused: bool) -> [u32; 4] {
    let x = |i: usize| f32::from_bits(inp[i]);
    let mut s: Vec<[f32; 4]> = (0..B)
        .map(|k| {
            (k..inp.len()).step_by(B).fold([f32::INFINITY, f32::NEG_INFINITY, 0.0, 0.0], |[mn, mx, s, q], i| {
                let v = x(i);
                [mn.min(v), mx.max(v), s + v, if fused { v.mul_add(v, q) } else { q + v * v }]
            })
        })
        .collect();
    let mut h = B / 2;
    while h >= 1 {
        for k in 0..h {
            let (a, b) = (s[k], s[k + h]);
            s[k] = [a[0].min(b[0]), a[1].max(b[1]), a[2] + b[2], a[3] + b[3]];
        }
        h /= 2;
    }
    s[0].map(f32::to_bits)
}

#[test]
fn the_kernels_are_the_tree_reduction() {
    for which in [hand_ptx(), kir_ptx()] {
        for inp in cases() {
            let got = run(&which, &inp, Order::Ascending);
            let (back, out) = got.split_at(inp.len());
            assert_eq!(back, &inp[..], "n = {}: input untouched", inp.len());
            let want = reference(&inp);
            let same = out[..4].iter().zip(&want).all(|(g, w)| g == w || (f32::from_bits(*g).is_nan() && f32::from_bits(*w).is_nan()));
            assert!(same, "n = {}: {:x?} against {want:x?}", inp.len(), &out[..4]);
            assert!(out[4..].iter().all(|&w| w == POISON), "n = {}: wrote past out", inp.len());
        }
    }
}

/// The data reaches the cases the kernel distinguishes: several trips of
/// the stride, an order-sensitive sum, squares a fused multiply-add would
/// round differently, NaNs, and squares out of range.
#[test]
fn the_data_covers_every_case() {
    assert!(LENS.iter().any(|&n| n > 2 * B) && LENS.contains(&0));
    let c = &cases()[5];
    let fwd = c.iter().fold(0.0f32, |a, &w| a + f32::from_bits(w));
    let back = c.iter().rev().fold(0.0f32, |a, &w| a + f32::from_bits(w));
    assert_ne!(fwd.to_bits(), back.to_bits(), "the sum depends on order");
    assert!(cases().iter().any(|c| reference_with(c, true)[3] != reference_with(c, false)[3]), "fusing the square shows");
    assert!(cases().iter().any(|c| c.iter().any(|&w| f32::from_bits(w).is_nan())));
    let squares: Vec<f32> = cases().iter().flatten().map(|&w| f32::from_bits(w) * f32::from_bits(w)).collect();
    assert!(squares.iter().any(|q| q.is_infinite()) && cases().iter().flatten().any(|&w| w != 0 && w != 0x8000_0000 && f32::from_bits(w) * f32::from_bits(w) == 0.0));
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on any case or
/// schedule (or faults).
fn caught(mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx();
    let mutant = mutate(&kir_ptx());
    cases().iter().any(|inp| {
        ORDERS.into_iter().any(|order| {
            let expect = run(&hand, inp, order);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mutant, inp, order))) {
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
fn the_unmutated_kernel_is_not_caught() {
    assert!(!caught(|p| p.to_string()));
}

/// The loop bound, the stride, the tree's first half, halving, idle test
/// and partner index, and both barriers.
#[test]
fn the_loop_and_tree_are_pinned() {
    assert_eq!(kir_ptx().matches("setp.ge.u64 ").count(), 1);
    assert!(caught(|p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", 0)), "bound");
    assert!(caught(|p| p.replacen(", 256;", ", 128;", 1)), "stride");
    assert!(caught(|p| p.replacen(", 128;", ", 64;", 1)), "first half");
    assert!(caught(|p| drop_op(p, "shr.u32 ", "u32", 0)), "halving");
    assert!(caught(|p| nudge(p, "setp.ge.u32 ", "setp.gt.u32 ", 0)), "idle test");
    assert!(caught(|p| drop_op(p, "add.u32 ", "u32", 0)), "partner");
    assert_eq!(kir_ptx().matches("bar.sync 0;").count(), 2);
    for i in 0..2 {
        assert!(caught(|p| edit(p, "bar.sync 0;", i, |_| String::new())), "barrier {i}");
    }
}

/// Named equivalent mutant: every thread writing the result. The writes
/// follow the tree's last barrier, so each thread stores the same four
/// `s[0]` to the same `out` slots.
#[test]
fn every_thread_writing_is_an_equivalent_mutant() {
    assert!(!caught(|p| nudge(p, "setp.ne.u32 ", "setp.eq.u32 ", 0)));
}

/// Every shared region's offset, every element size, every 64-bit add, and
/// each output slot. The first output slot's element size and address add
/// are named equivalent mutants: its offset is `0 · 4`, which `0 · 8` and
/// dropping the add leave unchanged.
#[test]
fn every_address_is_pinned() {
    let p = kir_ptx();
    for off in [", 1024;", ", 2048;", ", 3072;"] {
        assert!(caught(|p| p.replacen(off, ", 0;", 1)), "region offset {off}");
    }
    let sizes: Vec<usize> = p.lines().enumerate().filter(|(_, l)| l.contains("mul.lo.u64") && l.ends_with(", 4;")).map(|(n, _)| n).collect();
    assert_eq!(sizes.len(), 1 + 4 + 12 + 4, "the load, the partials' stores, the tree's accesses, the outputs");
    let first_slot = sizes[sizes.len() - 4];
    for &at in &sizes {
        let mutate = |p: &str| {
            p.lines().enumerate().map(|(n, l)| if n == at { l.replacen(", 4;", ", 8;", 1) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
        };
        assert_eq!(caught(mutate), at != first_slot, "element size at line {at}");
    }
    let adds = p.matches("add.u64 ").count();
    assert_eq!(adds, 3 + 1 + 1 + 4 + 12 + 4, "the region offsets, the load, the stride, the stores, the tree, the outputs");
    let first_slot_add = adds - 4;
    for i in 0..adds {
        assert_eq!(caught(|p| drop_op(p, "add.u64 ", "u64", i)), i != first_slot_add, "add {i}");
    }
    for slot in 0..4 {
        let from = format!("mov.u64 %rd0, {slot};");
        assert!(p.contains(&from), "slot {slot}");
        let to = format!("mov.u64 %rd0, {};", (slot + 1) % 4);
        assert!(caught(|p| p.replacen(&from, &to, 1)), "slot {slot}");
    }
}

/// Every combine (the loop's and the tree's), the square, and each
/// identity.
#[test]
fn the_combines_and_identities_are_pinned() {
    let p = kir_ptx();
    for (op, other) in [("min.f32 ", "max.f32 "), ("max.f32 ", "min.f32 "), ("add.rn.f32 ", "sub.rn.f32 ")] {
        let n = p.matches(op).count();
        assert_eq!(n, if op.starts_with("add") { 4 } else { 2 }, "{op}");
        for i in 0..n {
            assert!(caught(|p| nudge(p, op, other, i)), "{op} {i}");
        }
    }
    assert!(caught(|p| p.replacen("mul.rn.f32 ", "add.rn.f32 ", 1)), "the square");
    assert!(caught(|p| drop_op(p, "mul.rn.f32 ", "f32", 0)), "the square dropped");
    for (init, wrong) in [("0f7F800000", "0f00000000"), ("0fFF800000", "0f00000000")] {
        assert!(caught(|p| p.replacen(init, wrong, 1)), "{init}");
    }
    for i in 0..2 {
        assert!(caught(|p| nudge(p, "0f00000000", "0f80000000", i)), "sum identity {i} as -0");
        assert!(caught(|p| nudge(p, "0f00000000", "0f3F800000", i)), "sum identity {i} as 1");
    }
}

/// The square fused into its accumulate, as ptxas contracted the hand
/// kernel's `mul.f32` + `add.f32` on hardware: one rounding instead of two
/// shows on the data.
#[test]
fn fusing_the_square_is_caught() {
    let p = kir_ptx();
    let mul = p.lines().find(|l| l.contains("mul.rn.f32 ")).expect("the square").to_string();
    let sq = mul.split_whitespace().nth(1).expect("dst").trim_end_matches(',').to_string();
    let x = mul.split_whitespace().nth(2).expect("x").trim_end_matches(',').to_string();
    let acc = p.lines().find(|l| l.contains("add.rn.f32 ") && l.trim_end_matches(';').ends_with(&format!(", {sq}"))).expect("the accumulate").to_string();
    let ops: Vec<&str> = acc.trim().trim_start_matches("add.rn.f32 ").trim_end_matches(';').split(", ").collect();
    let fused = format!("    fma.rn.f32 {}, {x}, {x}, {};", ops[0], ops[1]);
    assert!(caught(|p| p.replacen(&mul, "", 1).replacen(&acc, &fused, 1)), "{mul} / {acc} as {fused}");
}
