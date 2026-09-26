//! The differential equivalence gate for the RoPE `rotate_half` pair
//! (roadmap A2 step 11): `nsl_rotate_half_f32` (`out[..h] = -in[h..]`,
//! `out[h..] = in[..h]` over the last dimension) and the fused backward
//! `nsl_rotate_half_neg_f32` (the negation on the other half).
//!
//! The runtime carried them as hand-written PTX; they are now built by
//! `nsl_kir::kernels::elementwise::build_rotate_half`. This file runs the
//! frozen hand modules (`tests/fixtures/rotate_half_hand.rs`) and the KIR
//! ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), launched as `nsl_tensor_rotate_half`
//! launches them (`half = last_dim / 2`, `ceil(n / 256)` blocks of 256) plus
//! one block more:
//!
//! 1. **Agreement**: under two schedules and several last dimensions, the
//!    two kernels leave *the same bytes* in all of global memory, over
//!    IEEE-corner values (NaN payloads and signed zeros included).
//! 2. **Correctness**: the output is the formula bit for bit (a negation is
//!    a sign-bit flip), `_neg` is exactly the sign flip of the plain kernel,
//!    nothing past `n` is written and the input is untouched.
//! 3. **The gate bites**: relaxing the bound, reading the index from block
//!    0, nudging an element size, the partner on the wrong side, the
//!    negation dropped or in the other arm, `<=` for `<`, and the row
//!    position taken as the flat index are each caught.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::{rotate_half_ptx, RotateHalfOp, ELEMENTWISE_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/rotate_half_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const A: u64 = 0x1000_0000;
const C: u64 = 0x2000_0000;
const TAIL: usize = 300;
const POISON: u32 = 0x7FA5_A5A5;
const IN_TAIL: u32 = 0xC040_0000; // -3.0

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

fn kir_ptx(op: RotateHalfOp) -> String {
    trim(&String::from_utf8(rotate_half_ptx(op)).expect("ASCII"))
}

fn hand_ptx(op: RotateHalfOp) -> String {
    trim(match op {
        RotateHalfOp::Plain => hand::ROTATE_HALF_F32_PTX,
        RotateHalfOp::Neg => hand::ROTATE_HALF_NEG_F32_PTX,
    })
}

fn values(n: usize, seed: u64) -> Vec<u32> {
    const CORNERS: [u32; 10] = [
        0x0000_0000,
        0x8000_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0001, // NaN, payload kept: the sign flip must not canonicalise
        0xFFC0_1234,
        0x0000_0001,
        0x807F_FFFF,
        0x3F80_0000,
        0xBF00_0000,
    ];
    let mut s = seed;
    (0..n)
        .map(|k| {
            if k < CORNERS.len() {
                return CORNERS[(k + seed as usize) % CORNERS.len()];
            }
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0).to_bits()
        })
        .collect()
}

fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// Run `ptx` on `x` (rows of `last_dim`) over `ceil(n / 256) + 1` blocks.
/// Returns all of global memory: `a`, `c`.
fn run(ptx: &str, x: &[u32], last_dim: usize, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let n = x.len();
    assert_eq!(n % last_dim, 0, "whole rows");
    let mut a = le32(x);
    a.extend(le32(&vec![IN_TAIL; TAIL]));
    let mut global = vec![Segment { base: A, bytes: a }, Segment { base: C, bytes: le32(&vec![POISON; n + TAIL]) }];
    let args: HashMap<String, u64> =
        [("a", A), ("c", C), ("n", n as u64), ("last_dim", last_dim as u64), ("half", (last_dim / 2) as u64)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
    let grid = n.div_ceil(ELEMENTWISE_BLOCK as usize) as u32 + 1;
    let mut ctas: Vec<u32> = (0..grid).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for cta in ctas {
        let mut launch = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![],
            ctaid: cta,
            ctaid_y: 0,
            nctaid_y: 1,
            ntid: ELEMENTWISE_BLOCK,
            steps: 0,
        };
        run_cta(&mut launch, order);
    }
    global.into_iter().map(|s| s.bytes).collect()
}

/// `(rows, last_dim)`: a single pair, head dims 8/64/128, an odd half (6),
/// and sizes that leave a ragged last block.
const SHAPES: [(usize, usize); 6] = [(1, 2), (3, 8), (4, 64), (5, 6), (2, 128), (9, 34)];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

#[test]
fn the_kir_kernels_agree_with_the_hand_written_ones_bit_for_bit() {
    for op in RotateHalfOp::ALL {
        for (j, &(rows, d)) in SHAPES.iter().enumerate() {
            let x = values(rows * d, j as u64 + 1);
            for order in ORDERS {
                let hand = run(&hand_ptx(op), &x, d, order);
                let kir = run(&kir_ptx(op), &x, d, order);
                assert!(hand == kir, "{op:?} rows={rows} d={d} {order:?}: global memory differs");
            }
        }
    }
}

/// The formula, with the negation as the sign-bit flip it is.
fn model(op: RotateHalfOp, x: &[u32], d: usize) -> Vec<u32> {
    let h = d / 2;
    (0..x.len())
        .map(|i| {
            let col = i % d;
            let (src, first) = if col < h { (i + h, true) } else { (i - h, false) };
            let negate = first == (op == RotateHalfOp::Plain);
            x[src] ^ if negate { 0x8000_0000 } else { 0 }
        })
        .collect()
}

#[test]
fn the_answer_is_the_formula_bit_for_bit() {
    for op in RotateHalfOp::ALL {
        for &(rows, d) in &SHAPES {
            let n = rows * d;
            let x = values(n, 7);
            let mem = run(&kir_ptx(op), &x, d, Order::Ascending);
            let out = words(&mem[1]);
            assert_eq!(out[..n], model(op, &x, d)[..], "{op:?} d={d}");
            assert!(out[n..].iter().all(|&w| w == POISON), "{op:?} d={d}: wrote past n");
            assert_eq!(words(&mem[0])[..n], x[..], "{op:?} d={d}: the input is untouched");
        }
    }
}

/// `_neg` is the plain kernel's output with every sign bit flipped: the
/// fusion it exists for.
#[test]
fn the_neg_kernel_is_the_negated_plain_kernel() {
    let (rows, d) = (4, 64);
    let x = values(rows * d, 11);
    let plain = words(&run(&kir_ptx(RotateHalfOp::Plain), &x, d, Order::Ascending)[1]);
    let neg = words(&run(&kir_ptx(RotateHalfOp::Neg), &x, d, Order::Ascending)[1]);
    for i in 0..rows * d {
        assert_eq!(neg[i], plain[i] ^ 0x8000_0000, "i={i}");
    }
}

#[test]
fn the_kir_kernels_keep_the_ffi_signatures_and_entry_names() {
    for op in RotateHalfOp::ALL {
        let (h, k) = (hand_ptx(op), kir_ptx(op));
        assert_eq!(parse_signature(&k), parse_signature(&h), "{op:?}");
        let names: Vec<String> = parse_signature(&h).into_iter().map(|(_, name)| name).collect();
        assert_eq!(names, ["a", "c", "n", "last_dim", "half"], "{op:?}");
        for text in [&h, &k] {
            assert!(text.contains(&format!(".visible .entry {}(", op.kernel_name())), "{op:?}");
        }
        for form in ["rem.u64 ", "setp.lt.u64 ", "neg.f32 ", "ld.global.f32 ", "st.global.f32 "] {
            assert_eq!(k.matches(form).count(), h.matches(form).count(), "{op:?} {form}");
        }
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` of `op` is told apart from the hand kernel, over two
/// shapes (one past a single block, so the block index matters), under
/// either schedule (or faults).
fn caught(op: RotateHalfOp, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(op);
    let mutant = mutate(&kir_ptx(op));
    [(3usize, 8usize), (3, 128)].into_iter().any(|(rows, d)| {
        let x = values(rows * d, 3);
        ORDERS.into_iter().any(|order| {
            let expect = run(&hand, &x, d, order);
            let x = &x;
            match std::panic::catch_unwind(|| run(&mutant, x, d, order)) {
                Ok(mem) => mem != expect,
                Err(_) => true,
            }
        })
    })
}

fn nudge(ptx: &str, from: &str, to: &str, i: usize) -> String {
    let at = ptx.lines().enumerate().filter(|(_, l)| l.contains(from)).nth(i).expect("the line").0;
    ptx.lines()
        .enumerate()
        .map(|(k, l)| if k == at { l.replacen(from, to, 1) } else { l.to_string() })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for op in RotateHalfOp::ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

#[test]
fn relaxing_the_bound_or_ignoring_the_block_index_is_caught() {
    for op in RotateHalfOp::ALL {
        let k = kir_ptx(op);
        assert_eq!(k.matches("setp.ge.u64 ").count(), 1, "{op:?}");
        assert_eq!(k.matches("%ctaid.x;").count(), 1, "{op:?}");
        assert!(caught(op, |p| p.replacen("setp.ge.u64 ", "setp.gt.u64 ", 1)), "{op:?}: bound");
        assert!(caught(op, |p| p.replacen("%ctaid.x;", "0;", 1)), "{op:?}: block index");
    }
}

/// f32's size in every address: one load and one store per arm.
#[test]
fn nudging_an_element_size_is_caught() {
    for op in RotateHalfOp::ALL {
        let count = kir_ptx(op).matches(", 4;").count();
        assert_eq!(count, 4, "{op:?}");
        for i in 0..count {
            assert!(caught(op, |p| nudge(p, ", 4;", ", 8;", i)), "{op:?}: address {i}");
        }
    }
}

#[test]
fn the_partner_on_the_wrong_side_is_caught() {
    for op in RotateHalfOp::ALL {
        let k = kir_ptx(op);
        let adds = k.lines().filter(|l| l.contains("add.u64") && l.contains("%rd")).count();
        assert!(adds >= 1, "{op:?}");
        assert!(caught(op, |p| p.replacen("sub.u64 ", "add.u64 ", 1)), "{op:?}: i + half in the second arm");
    }
}

#[test]
fn the_negation_dropped_or_moved_is_caught() {
    for op in RotateHalfOp::ALL {
        // Dropped: neg d, a -> mov d, a.
        assert!(caught(op, |p| p.replacen("neg.f32 ", "mov.f32 ", 1)), "{op:?}: negation dropped");
        // Moved: the other kernel's arms.
        let other = if op == RotateHalfOp::Plain { RotateHalfOp::Neg } else { RotateHalfOp::Plain };
        assert!(caught(op, |_| kir_ptx(other).replace(other.kernel_name(), op.kernel_name())), "{op:?}: negation in the other arm");
    }
}

#[test]
fn the_half_comparison_is_pinned() {
    for op in RotateHalfOp::ALL {
        assert!(caught(op, |p| p.replacen("setp.lt.u64 ", "setp.le.u64 ", 1)), "{op:?}: col <= half");
    }
}

/// The row position is `i % last_dim`; comparing the flat index instead
/// treats every row but the first as its second half.
#[test]
fn the_row_position_is_pinned() {
    for op in RotateHalfOp::ALL {
        assert!(caught(op, |p| p.replacen("rem.u64 ", "min.u64 ", 1)), "{op:?}: flat index for the row position");
    }
}
