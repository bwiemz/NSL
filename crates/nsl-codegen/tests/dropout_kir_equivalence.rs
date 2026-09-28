//! The differential equivalence gate for `nsl_dropout_f32`, the GPU inverted
//! dropout kernel, now built by `nsl_kir::kernels::dropout` (new-roadmap
//! item 5).
//!
//! The runtime carried it as hand-written PTX in `cuda/fused_kernels.rs`.
//! This file runs the frozen hand module (`tests/fixtures/dropout_hand.rs`)
//! and the KIR one side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`). The grid is the one the runtime
//! launches, `ceil(len / 256)` blocks of 256, plus one block more.
//!
//! 1. **Agreement**: under two schedules, the two modules leave *the same
//!    bytes* in all of global memory. The sweep covers keep-nothing,
//!    keep-everything and in-between thresholds, several seeds (one whose
//!    `seed + i` crosses the 32-bit truncation) and IEEE-corner inputs.
//! 2. **Correctness**: every element is the restated hash's decision.
//!    `out = x · (keep ? scale : 0)`, so a dropped NaN stays NaN, and
//!    `mask = keep ? 1 : 0`. The mask depends only on `seed + i`, and
//!    nothing is written past `len`.
//! 3. **The gate bites**: the bound, the block index, each element size,
//!    each parameter slot, each hash multiplier and shift, the counter, the
//!    unsigned strict comparison (a threshold equal to an element's hash is
//!    included), both selects and the product are caught.

use std::collections::HashMap;

use nsl_kir::kernels::dropout::{dropout_ptx, DROPOUT_HASH_MULTIPLIERS, DROPOUT_HASH_SHIFTS, DROPOUT_NAME};
use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;

#[allow(dead_code)]
#[path = "fixtures/dropout_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const INP: u64 = 0x1000_0000;
const OUT: u64 = 0x2000_0000;
const MASK: u64 = 0x3000_0000;
const TAIL: usize = 300;
const POISON: u32 = 0xA5A5_A5A5;
const IN_TAIL: u32 = 0x3FC0_0000; // 1.5

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

fn kir_ptx() -> String {
    trim(&String::from_utf8(dropout_ptx()).expect("ASCII"))
}

fn hand_ptx() -> String {
    trim(hand::DROPOUT_F32_PTX)
}

/// The kernel's hash of the counter `seed + i`, restated.
fn hash(counter: u64) -> u32 {
    let mut h = counter as u32;
    for (m, s) in DROPOUT_HASH_MULTIPLIERS.into_iter().zip(DROPOUT_HASH_SHIFTS) {
        h = h.wrapping_mul(m);
        h ^= h >> s;
    }
    h
}

#[test]
fn the_restated_hash_is_the_hand_kernels() {
    // The hand kernel's constants, read off its text.
    let h = hand_ptx();
    for m in DROPOUT_HASH_MULTIPLIERS {
        assert!(h.contains(&format!(", {m:#X};").replace("0X", "0x")), "{m:#x}");
    }
    for s in DROPOUT_HASH_SHIFTS {
        assert!(h.contains(&format!(", {s};")), "{s}");
    }
}

/// Inputs: IEEE corners first, then ordinary values.
fn inputs(n: usize, seed: u64) -> Vec<u32> {
    const CORNERS: [u32; 10] =
        [0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0001, 0x0000_0001, 0x7F7F_FFFF, 0x3F80_0000, 0xBF80_0000, 0x4049_0FDB];
    let mut s = seed;
    (0..n)
        .map(|k| {
            if k < CORNERS.len() {
                CORNERS[(k + seed as usize) % CORNERS.len()]
            } else {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0).to_bits()
            }
        })
        .collect()
}

fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// One launch configuration.
#[derive(Debug, Clone, Copy)]
struct Case {
    len: usize,
    threshold: u32,
    scale: f32,
    seed: u64,
}

/// The sweep: keep nothing, keep everything, the runtime's p = 0.1 and
/// p = 0.5, a seed whose `seed + i` crosses 2^32, and the pair of
/// thresholds `hash(e7)` and `hash(e7) + 1` around element 7. The pair is
/// what makes the hash mutants deterministic kills: a change to the last
/// xorshift moves only the low bits, which the comparison sees only near
/// the threshold, and a moved `hash(e7)` lands below the first threshold
/// or at or above the second.
fn cases() -> Vec<Case> {
    let t = |p: f64| ((1.0 - p) * u32::MAX as f64) as u32;
    let boundary_seed = 1u64 << 40;
    vec![
        Case { len: 1, threshold: t(0.5), scale: 2.0, seed: 0 },
        Case { len: 257, threshold: 0, scale: 2.0, seed: 3 },
        Case { len: 257, threshold: u32::MAX, scale: 1.0, seed: 3 },
        Case { len: 1000, threshold: t(0.1), scale: (1.0 / 0.9) as f32, seed: 12_345 },
        Case { len: 511, threshold: t(0.5), scale: 2.0, seed: (1 << 32) - 200 },
        Case { len: 64, threshold: hash(boundary_seed + 7), scale: 1.25, seed: boundary_seed },
        Case { len: 64, threshold: hash(boundary_seed + 7) + 1, scale: 1.25, seed: boundary_seed },
    ]
}

fn launch(ptx: &str, c: Case, order: Order) -> (Vec<u32>, Vec<Vec<u8>>) {
    let x = inputs(c.len, c.seed ^ 0x55);
    let mut inp = x.clone();
    inp.extend(vec![IN_TAIL; TAIL]);
    let mut global = vec![
        Segment { base: INP, bytes: le32(&inp) },
        Segment { base: OUT, bytes: le32(&vec![POISON; c.len + TAIL]) },
        Segment { base: MASK, bytes: le32(&vec![POISON; c.len + TAIL]) },
    ];
    let args: HashMap<String, u64> = [
        ("inp", INP),
        ("out", OUT),
        ("mask", MASK),
        ("len", c.len as u64),
        ("threshold", c.threshold as u64),
        ("scale", c.scale.to_bits() as u64),
        ("seed", c.seed),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let prog = parse(ptx);
    let grid = c.len.div_ceil(ELEMENTWISE_BLOCK as usize) as u32 + 1;
    let mut ctas: Vec<u32> = (0..grid).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for cta in ctas {
        let mut l = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![],
            ctaid: cta,
            ctaid_y: 0,
            nctaid_x: 0, nctaid_y: 1,
            ntid: ELEMENTWISE_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    (x, global.into_iter().map(|s| s.bytes).collect())
}

const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

fn same(got: u32, want: u32) -> bool {
    got == want || (f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan())
}

#[test]
fn the_kir_kernel_agrees_with_the_hand_written_one_bit_for_bit() {
    let (hand, kir) = (hand_ptx(), kir_ptx());
    for c in cases() {
        for order in ORDERS {
            assert!(launch(&hand, c, order).1 == launch(&kir, c, order).1, "{c:?} {order:?}");
        }
    }
}

#[test]
fn every_element_is_the_hashs_decision() {
    let mut kept_any = false;
    let mut dropped_any = false;
    for c in cases() {
        let (x, mem) = launch(&kir_ptx(), c, Order::Ascending);
        let (out, mask) = (words(&mem[1]), words(&mem[2]));
        for i in 0..c.len {
            let keep = hash(c.seed.wrapping_add(i as u64)) < c.threshold;
            kept_any |= keep;
            dropped_any |= !keep;
            let factor = if keep { c.scale } else { 0.0 };
            let want = (f32::from_bits(x[i]) * factor).to_bits();
            assert!(same(out[i], want), "{c:?} i={i}: out {:#010x} vs {want:#010x}", out[i]);
            assert_eq!(mask[i], if keep { 1.0f32 } else { 0.0 }.to_bits(), "{c:?} i={i}: mask");
        }
        assert!(out[c.len..].iter().chain(&mask[c.len..]).all(|&w| w == POISON), "{c:?}: wrote past len");
        assert_eq!(words(&mem[0])[..c.len], x[..], "the input is only read");
    }
    assert!(kept_any && dropped_any);
    // The boundary pair really sits on the boundary: element 7 is dropped
    // at `hash`, kept at `hash + 1`.
    let (_, mem) = launch(&kir_ptx(), cases()[5], Order::Ascending);
    assert_eq!(words(&mem[2])[7], 0, "hash == threshold is dropped (strict <)");
    let (_, mem) = launch(&kir_ptx(), cases()[6], Order::Ascending);
    assert_eq!(words(&mem[2])[7], 1.0f32.to_bits(), "hash < threshold is kept");
}

#[test]
fn the_kir_kernel_keeps_the_ffi_signature_and_entry_name() {
    let (h, k) = (hand_ptx(), kir_ptx());
    assert_eq!(parse_signature(&k), parse_signature(&h));
    for p in [&h, &k] {
        assert!(p.contains(&format!(".visible .entry {DROPOUT_NAME}(")));
    }
}

#[test]
fn the_arithmetic_is_spelled_as_in_the_hand_kernel() {
    let (h, k) = (hand_ptx(), kir_ptx());
    for form in ["xor.b32 ", "shr.u32 ", "setp.lt.u32 ", "selp.f32 ", "mul.f32 ", "cvt.u32.u64 ", "st.global.f32 "] {
        assert_eq!(k.matches(form).count(), h.matches(form).count(), "{form}");
    }
    for never in ["fma.", "mul.rn.f32", "add.f32"] {
        assert!(!k.contains(never), "{never}");
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

fn caught(mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx();
    let mutant = mutate(&kir_ptx());
    cases().into_iter().any(|c| {
        ORDERS.into_iter().any(|o| {
            let expect = launch(&hand, c, o).1;
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| launch(&mutant, c, o).1)) {
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
fn the_unmutated_kernel_is_not_caught() {
    assert!(!caught(|p| p.to_string()));
}

#[test]
fn relaxing_the_bound_or_ignoring_the_block_index_is_caught() {
    let k = kir_ptx();
    assert_eq!((k.matches("setp.ge.u64 ").count(), k.matches("%ctaid.x;").count()), (1, 1));
    assert!(caught(|p| p.replacen("setp.ge.u64 ", "setp.gt.u64 ", 1)), "bound");
    assert!(caught(|p| p.replacen("%ctaid.x;", "0;", 1)), "block index");
}

#[test]
fn nudging_an_element_size_is_caught() {
    let k = kir_ptx();
    let sites = k.matches(", 4;").count();
    assert_eq!(sites, 3, "the input load and the two stores: {k}");
    for i in 0..sites {
        assert!(caught(|p| nudge(p, ", 4;", ", 8;", i)), "site {i}");
    }
}

#[test]
fn every_parameter_slot_is_live() {
    let ptrs = ["inp", "out", "mask"];
    let scalars = ["len", "seed"];
    for names in [&ptrs[..], &scalars[..]] {
        for (j, name) in names.iter().enumerate() {
            let other = names[(j + 1) % names.len()];
            let from = format!("[param_{name}];");
            assert!(caught(|p| p.replacen(&from, &format!("[param_{other}];"), 1)), "{name} read from {other}");
        }
    }
    // The 32-bit pair: a threshold read as the scale's bits, and back.
    assert!(caught(|p| p.replacen("[param_threshold];", "[param_scale];", 1)), "threshold");
}

/// Each multiplier's low bits and each xorshift amount.
#[test]
fn every_part_of_the_hash_is_caught() {
    let k = kir_ptx();
    for m in DROPOUT_HASH_MULTIPLIERS {
        let from = format!(", {m};");
        assert_eq!(k.matches(&from).count(), 1, "{m:#x}");
        assert!(caught(|p| p.replacen(&from, &format!(", {};", m ^ 2), 1)), "{m:#x}");
    }
    for (from, to, i) in [(", 16;", ", 15;", 0), (", 13;", ", 12;", 0), (", 16;", ", 17;", 1)] {
        assert!(caught(|p| nudge(p, from, to, i)), "{from} {i}");
    }
    assert!(caught(|p| p.replacen("xor.b32 ", "or.b32 ", 1)), "xorshift");
}

/// The counter `seed + i`, and the strict unsigned comparison.
#[test]
fn the_counter_and_the_comparison_are_caught() {
    let k = kir_ptx();
    let seed_add = k.lines().position(|l| l.contains("cvt.u32.u64")).expect("the truncation");
    let adds_before = k.lines().take(seed_add).filter(|l| l.contains("add.u64 ")).count();
    assert!(caught(|p| nudge(p, "add.u64 ", "sub.u64 ", adds_before - 1)), "seed - i");
    assert!(caught(|p| p.replacen("setp.lt.u32 ", "setp.le.u32 ", 1)), "<= on the boundary");
    assert!(caught(|p| p.replacen("setp.lt.u32 ", "setp.lt.s32 ", 1)), "signed compare");
}

/// Both selects' constants and operands, and the product.
#[test]
fn the_selects_and_the_product_are_caught() {
    let k = kir_ptx();
    assert!(caught(|p| p.replacen("mul.f32 ", "add.f32 ", 1)), "product");
    assert!(caught(|p| p.replacen("0f3F800000;", "0f3F000000;", 1)), "mask 1.0");
    let zeros = k.matches("0f00000000;").count();
    assert_eq!(zeros, 2, "{k}");
    for i in 0..zeros {
        assert!(caught(|p| nudge(p, "0f00000000;", "0f80000000;", i)), "zero {i} as -0");
    }
    for i in 0..2 {
        let line = k.lines().filter(|l| l.contains("selp.f32")).nth(i).expect("selp").trim().to_string();
        let ops: Vec<&str> = line.trim_end_matches(';').split_whitespace().skip(1).map(|s| s.trim_end_matches(',')).collect();
        let swapped = format!("selp.f32 {}, {}, {}, {};", ops[0], ops[2], ops[1], ops[3]);
        assert!(caught(|p| p.replacen(&line, &swapped, 1)), "select {i} inverted");
    }
}
