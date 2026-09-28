//! The differential equivalence gate for the integer dequantization kernels
//! (new-roadmap item 5): `nsl_dequant_int8_per_head_f32`,
//! `nsl_dequant_int8_per_token_f32` and `nsl_dequant_int4_per_group_f32`.
//!
//! The runtime carried them as hand-written PTX in `cuda/fused_kernels.rs`.
//! They are now built by `nsl_kir::kernels::dequant`. For the packed int4
//! bytes KIR gained `KirType::U8`, and the interpreter gained
//! `cvt.rn.f32.s16` (the hand kernels' widening) and `cvt.u32.u8` (KIR's).
//! This file runs the frozen hand modules (`tests/fixtures/dequant_hand.rs`)
//! and the KIR ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`). The grid is the one the runtime
//! launches, `ceil(n / 256)` blocks of 256, plus one block more.
//!
//! 1. **Agreement**: under two schedules and several layouts, the two
//!    modules leave *the same bytes* in all of global memory. Every byte
//!    value goes in, and the scales and zero points include signed zeros,
//!    subnormals, infinities, NaN and the extremes.
//! 2. **Correctness**: each kernel is its formula, bit for bit, and writes
//!    nothing past `n`. For int8 that is `f32(q) · scale` at the head's or the
//!    token's scale. For int4 it is `fma(nibble, scale, zero_point)`, low
//!    nibble first, one rounding.
//! 3. **The gate bites**: the bound, the block index, each element size, each
//!    parameter slot, the scale index arithmetic, the signed widening, the
//!    nibble shift, masks and parity test, and the arithmetic are caught.

use std::collections::HashMap;

use nsl_kir::kernels::dequant::{int4_per_group_ptx, int8_ptx, Int8Scale, INT4_PER_GROUP_NAME};
use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;

#[allow(dead_code)]
#[path = "fixtures/dequant_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const INP: u64 = 0x1000_0000;
const OUT: u64 = 0x2000_0000;
const SCALES: u64 = 0x3000_0000;
const ZPS: u64 = 0x4000_0000;
const TAIL: usize = 300;
/// What `out` holds before the launch: a pattern no kernel writes.
const POISON: u32 = 0xA5A5_A5A5;
/// What the scale and zero-point tables hold past their length: finite, so
/// an index past the end reads a value that changes the result.
const TABLE_TAIL: u32 = 0x4040_0000; // 3.0

#[derive(Clone, Copy, Debug, PartialEq)]
enum K {
    Head,
    Token,
    Int4,
}

const ALL: [K; 3] = [K::Head, K::Token, K::Int4];

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

fn kir_ptx(k: K) -> String {
    trim(&String::from_utf8(match k {
        K::Head => int8_ptx(Int8Scale::PerHead),
        K::Token => int8_ptx(Int8Scale::PerToken),
        K::Int4 => int4_per_group_ptx(),
    })
    .expect("ASCII"))
}

fn hand_ptx(k: K) -> String {
    trim(match k {
        K::Head => hand::DEQUANT_INT8_PER_HEAD_F32_PTX,
        K::Token => hand::DEQUANT_INT8_PER_TOKEN_F32_PTX,
        K::Int4 => hand::DEQUANT_INT4_PER_GROUP_F32_PTX,
    })
}

fn name(k: K) -> &'static str {
    match k {
        K::Head => Int8Scale::PerHead.kernel_name(),
        K::Token => Int8Scale::PerToken.kernel_name(),
        K::Int4 => INT4_PER_GROUP_NAME,
    }
}

// ---------------------------------------------------------------------------
// Inputs
// ---------------------------------------------------------------------------

/// Every byte value, in an order that differs per seed.
fn bytes(n: usize, seed: u64) -> Vec<u8> {
    (0..n).map(|k| (k as u64).wrapping_mul(37).wrapping_add(seed * 101) as u8).collect()
}

/// Scale-like f32 words: the IEEE corners first, then ordinary magnitudes.
fn floats(n: usize, seed: u64) -> Vec<u32> {
    const CORNERS: [u32; 11] = [
        0x3F80_0000, // 1
        0x0000_0000, // +0
        0x8000_0000, // -0
        0x0000_0001, // least subnormal
        0x7F80_0000, // +inf
        0xFF80_0000, // -inf
        0x7FC0_0001, // NaN
        0x7F7F_FFFF, // max
        0x3C00_0000, // 1/128
        0xBF00_0000, // -0.5
        0x4F00_0000, // 2^31
    ];
    let mut s = seed;
    (0..n)
        .map(|k| {
            if k < CORNERS.len() {
                CORNERS[(k + seed as usize) % CORNERS.len()]
            } else {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0).to_bits()
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

fn with_tail(v: &[u32], tail: u32) -> Vec<u8> {
    let mut w = v.to_vec();
    w.extend(vec![tail; TAIL]);
    le32(&w)
}

fn launch(ptx: &str, global: Vec<Segment>, args: &[(&str, u64)], n: usize, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let mut global = global;
    let args: HashMap<String, u64> = args.iter().map(|(k, v)| (k.to_string(), *v)).collect();
    let grid = n.div_ceil(ELEMENTWISE_BLOCK as usize) as u32 + 1;
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
    global.into_iter().map(|s| s.bytes).collect()
}

/// An int8 layout: `n` elements, `head_stride` per head, `head_dim` per
/// token.
#[derive(Debug, Clone, Copy)]
struct Int8Shape {
    n: usize,
    head_stride: usize,
    head_dim: usize,
}

const INT8_SHAPES: [Int8Shape; 5] = [
    Int8Shape { n: 1, head_stride: 1, head_dim: 1 },
    Int8Shape { n: 257, head_stride: 64, head_dim: 8 },
    Int8Shape { n: 300, head_stride: 100, head_dim: 25 },
    Int8Shape { n: 1000, head_stride: 1000, head_dim: 3 },
    Int8Shape { n: 511, head_stride: 7, head_dim: 7 },
];

/// The number of scales a layout reads.
fn int8_scales(k: K, s: Int8Shape) -> usize {
    match k {
        K::Head => s.n.div_ceil(s.head_stride),
        _ => s.head_stride.div_ceil(s.head_dim),
    }
}

/// An int8 kernel over `s`. Returns `[inp, out, scales]`.
fn run_int8(k: K, ptx: &str, s: Int8Shape, seed: u64, order: Order) -> (Vec<u8>, Vec<u32>, Vec<Vec<u8>>) {
    let q = bytes(s.n, seed);
    let sc = floats(int8_scales(k, s), seed + 7);
    let mut inp = q.clone();
    inp.extend(vec![0x7F; TAIL]);
    let global = vec![
        Segment { base: INP, bytes: inp },
        Segment { base: OUT, bytes: le32(&vec![POISON; s.n + TAIL]) },
        Segment { base: SCALES, bytes: with_tail(&sc, TABLE_TAIL) },
    ];
    let mut args = vec![("inp", INP), ("out", OUT), ("scales", SCALES), ("n", s.n as u64), ("head_stride", s.head_stride as u64)];
    if k == K::Token {
        args.push(("head_dim", s.head_dim as u64));
    }
    let mem = launch(ptx, global, &args, s.n, order);
    (q, sc, mem)
}

/// `(n, group_size)` for int4.
const INT4_SHAPES: [(usize, usize); 5] = [(1, 1), (256, 2), (257, 3), (1000, 32), (511, 511)];

/// The int4 kernel. Returns `(packed, scales, zero_points, [inp, out,
/// scales, zero_points])`.
fn run_int4(ptx: &str, (n, group): (usize, usize), seed: u64, order: Order) -> (Vec<u8>, Vec<u32>, Vec<u32>, Vec<Vec<u8>>) {
    let packed = bytes(n.div_ceil(2), seed);
    let groups = n.div_ceil(group);
    let sc = floats(groups, seed + 3);
    let zp = floats(groups, seed + 11);
    let mut inp = packed.clone();
    inp.extend(vec![0xFF; TAIL]);
    let global = vec![
        Segment { base: INP, bytes: inp },
        Segment { base: OUT, bytes: le32(&vec![POISON; n + TAIL]) },
        Segment { base: SCALES, bytes: with_tail(&sc, TABLE_TAIL) },
        Segment { base: ZPS, bytes: with_tail(&zp, TABLE_TAIL) },
    ];
    let args =
        [("inp", INP), ("out", OUT), ("scales", SCALES), ("zero_points", ZPS), ("n", n as u64), ("group_size", group as u64)];
    let mem = launch(ptx, global, &args, n, order);
    (packed, sc, zp, mem)
}

const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

fn same(got: u32, want: u32) -> bool {
    got == want || (f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan())
}

// ---------------------------------------------------------------------------
// 1. Agreement
// ---------------------------------------------------------------------------

#[test]
fn the_kir_kernels_agree_with_the_hand_written_ones_bit_for_bit() {
    for k in [K::Head, K::Token] {
        let (hand, kir) = (hand_ptx(k), kir_ptx(k));
        for (j, &s) in INT8_SHAPES.iter().enumerate() {
            for order in ORDERS {
                let seed = j as u64 + 1;
                assert!(run_int8(k, &hand, s, seed, order).2 == run_int8(k, &kir, s, seed, order).2, "{k:?} {s:?} {order:?}");
            }
        }
    }
    let (hand, kir) = (hand_ptx(K::Int4), kir_ptx(K::Int4));
    for (j, &s) in INT4_SHAPES.iter().enumerate() {
        for order in ORDERS {
            let seed = j as u64 + 1;
            assert!(run_int4(&hand, s, seed, order).3 == run_int4(&kir, s, seed, order).3, "int4 {s:?} {order:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Correctness
// ---------------------------------------------------------------------------

#[test]
fn the_int8_kernels_scale_each_signed_byte_by_its_heads_or_tokens_scale() {
    for k in [K::Head, K::Token] {
        for &s in &INT8_SHAPES {
            let (q, sc, mem) = run_int8(k, &kir_ptx(k), s, 5, Order::Ascending);
            let out = words(&mem[1]);
            for i in 0..s.n {
                let slot = match k {
                    K::Head => i / s.head_stride,
                    _ => (i % s.head_stride) / s.head_dim,
                };
                let want = (q[i] as i8 as f32 * f32::from_bits(sc[slot])).to_bits();
                assert!(same(out[i], want), "{k:?} {s:?} i={i}: {:#010x} vs {want:#010x}", out[i]);
            }
            assert!(out[s.n..].iter().all(|&w| w == POISON), "{k:?} {s:?}: wrote past n");
        }
    }
}

#[test]
fn the_int4_kernel_is_one_fma_per_nibble_low_nibble_first() {
    for &(n, group) in &INT4_SHAPES {
        let (packed, sc, zp, mem) = run_int4(&kir_ptx(K::Int4), (n, group), 5, Order::Ascending);
        let out = words(&mem[1]);
        for i in 0..n {
            let byte = packed[i / 2];
            let nib = if i % 2 == 0 { byte & 15 } else { byte >> 4 };
            let g = i / group;
            let want = (nib as f32).mul_add(f32::from_bits(sc[g]), f32::from_bits(zp[g])).to_bits();
            assert!(same(out[i], want), "n={n} group={group} i={i}: {:#010x} vs {want:#010x}", out[i]);
        }
        assert!(out[n..].iter().all(|&w| w == POISON), "n={n}: wrote past n");
    }
}

/// The contraction is visible: a case where `fma` and a rounded multiply
/// then add differ, so the one-rounding claim is tested, not assumed.
#[test]
fn the_int4_result_is_fused_not_rounded_twice() {
    let (s, z) = (f32::from_bits(0x3F80_0001), -15.0f32 * f32::from_bits(0x3F80_0001));
    let fused = 15.0f32.mul_add(s, z);
    let twice = 15.0f32 * s + z;
    assert_ne!(fused.to_bits(), twice.to_bits());
    let global = || {
        vec![
            Segment { base: INP, bytes: vec![0xFF, 0, 0, 0] },
            Segment { base: OUT, bytes: le32(&[POISON; 4]) },
            Segment { base: SCALES, bytes: le32(&[s.to_bits()]) },
            Segment { base: ZPS, bytes: le32(&[z.to_bits()]) },
        ]
    };
    let args = [("inp", INP), ("out", OUT), ("scales", SCALES), ("zero_points", ZPS), ("n", 2), ("group_size", 2)];
    for p in [kir_ptx(K::Int4), hand_ptx(K::Int4)] {
        let out = words(&launch(&p, global(), &args, 2, Order::Ascending)[1]);
        assert_eq!(out[0], fused.to_bits());
        assert_eq!(out[1], fused.to_bits());
    }
}

#[test]
fn the_kir_kernels_keep_the_ffi_signatures_and_entry_names() {
    for k in ALL {
        let (h, kir) = (hand_ptx(k), kir_ptx(k));
        assert_eq!(parse_signature(&kir), parse_signature(&h), "{k:?}");
        for p in [&h, &kir] {
            assert!(p.contains(&format!(".visible .entry {}(", name(k))), "{k:?}");
        }
    }
}

#[test]
fn the_arithmetic_is_spelled_as_in_the_hand_kernels() {
    for k in ALL {
        let (h, kir) = (hand_ptx(k), kir_ptx(k));
        for form in ["mul.f32 ", "fma.rn.f32 ", "div.u64 ", "rem.u64 ", "ld.global.s8 ", "ld.global.u8 ", "st.global.f32 "] {
            assert_eq!(kir.matches(form).count(), h.matches(form).count(), "{k:?} {form}");
        }
        for never in ["add.f32", "mul.rn.f32", "div.rn", "div.approx"] {
            assert!(!kir.contains(never), "{k:?} {never}");
        }
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on any layout,
/// under either schedule (or faults).
fn caught(k: K, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(k);
    let mutant = mutate(&kir_ptx(k));
    let differs = |run: &dyn Fn(&str) -> Vec<Vec<u8>>| {
        let expect = run(&hand);
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mutant))) {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
    };
    ORDERS.into_iter().any(|o| match k {
        K::Head | K::Token => INT8_SHAPES.iter().any(|&s| differs(&|p| run_int8(k, p, s, 3, o).2)),
        K::Int4 => INT4_SHAPES.iter().any(|&s| differs(&|p| run_int4(p, s, 3, o).3)),
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
    for k in ALL {
        assert!(!caught(k, |p| p.to_string()), "{k:?}");
    }
}

#[test]
fn relaxing_the_bound_or_ignoring_the_block_index_is_caught() {
    for k in ALL {
        let p = kir_ptx(k);
        assert_eq!((p.matches("setp.ge.u64 ").count(), p.matches("%ctaid.x;").count()), (1, 1), "{k:?}");
        assert!(caught(k, |p| p.replacen("setp.ge.u64 ", "setp.gt.u64 ", 1)), "{k:?} bound");
        assert!(caught(k, |p| p.replacen("%ctaid.x;", "0;", 1)), "{k:?} block index");
    }
}

/// Each f32 address scales by 4: the int8 kernels' scale load and store, the
/// int4 kernel's two table loads and store. (The byte loads scale by 1, which
/// KIR folds away.)
#[test]
fn nudging_an_element_size_is_caught() {
    for (k, sites) in [(K::Head, 2), (K::Token, 2), (K::Int4, 3)] {
        let p = kir_ptx(k);
        let scaled: Vec<usize> =
            p.lines().enumerate().filter(|(_, l)| l.contains("mul.lo.u64") && l.ends_with(", 4;")).map(|(n, _)| n).collect();
        assert_eq!(scaled.len(), sites, "{k:?}\n{p}");
        for at in scaled {
            let mutant = |p: &str| {
                p.lines()
                    .enumerate()
                    .map(|(n, l)| if n == at { l.replacen(", 4;", ", 8;", 1) } else { l.to_string() })
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n"
            };
            assert!(caught(k, mutant), "{k:?} line {at}");
        }
    }
}

/// Reading any parameter from its neighbour's slot is caught.
#[test]
fn every_parameter_slot_is_live() {
    for (k, names) in [
        (K::Head, &["inp", "out", "scales", "n", "head_stride"][..]),
        (K::Token, &["inp", "out", "scales", "n", "head_stride", "head_dim"][..]),
        (K::Int4, &["inp", "out", "scales", "zero_points", "n", "group_size"][..]),
    ] {
        for (j, pname) in names.iter().enumerate() {
            let other = names[(j + 1) % names.len()];
            let from = format!("[param_{pname}];");
            assert_eq!(kir_ptx(k).matches(&from).count(), 1, "{k:?} {pname}");
            assert!(caught(k, |p| p.replacen(&from, &format!("[param_{other}];"), 1)), "{k:?} {pname} read from {other}");
        }
    }
}

/// The scale index: the head's quotient, the token's remainder then
/// quotient, the group's quotient.
#[test]
fn the_scale_index_arithmetic_is_caught() {
    assert!(caught(K::Head, |p| p.replacen("div.u64 ", "rem.u64 ", 1)), "head quotient");
    assert!(caught(K::Token, |p| p.replacen("rem.u64 ", "div.u64 ", 1)), "token remainder");
    assert!(caught(K::Token, |p| p.replacen("div.u64 ", "rem.u64 ", 1)), "token quotient");
    assert!(caught(K::Int4, |p| p.replacen("div.u64 ", "rem.u64 ", 1)), "group quotient");
}

/// The int8 value is signed and the product is a product.
#[test]
fn the_int8_widening_and_product_are_caught() {
    for k in [K::Head, K::Token] {
        assert!(caught(k, |p| p.replacen("cvt.rn.f32.s8 ", "cvt.rn.f32.u32 ", 1)), "{k:?} unsigned widening");
        assert!(caught(k, |p| p.replacen("mul.f32 ", "add.f32 ", 1)), "{k:?} product");
    }
}

/// The nibble: the byte index shift, the parity mask and test, both 15
/// masks, the high nibble's shift, and the fused product's operands. The
/// byte's widening is pinned as an equivalent mutant.
#[test]
fn every_part_of_the_nibble_unpack_is_caught() {
    let k = K::Int4;
    let p = kir_ptx(k);
    // `mov.u32 %r, 1;` is the byte-index shift; `mov.u64 %rd, 1;` the parity mask.
    let shift_one = p.lines().position(|l| l.trim_start().starts_with("mov.u32") && l.ends_with(", 1;")).expect("shift");
    let parity_one = p.lines().position(|l| l.trim_start().starts_with("mov.u64") && l.ends_with(", 1;")).expect("mask");
    let replace_at = |p: &str, at: usize, to: &str| {
        p.lines()
            .enumerate()
            .map(|(k, l)| if k == at { l.replacen(", 1;", to, 1) } else { l.to_string() })
            .collect::<Vec<_>>()
            .join("\n")
            + "\n"
    };
    assert!(caught(k, |p| replace_at(p, shift_one, ", 2;")), "byte index shift");
    assert!(caught(k, |p| replace_at(p, parity_one, ", 2;")), "parity mask");
    assert!(caught(k, |p| p.replacen("setp.ne.u64 ", "setp.eq.u64 ", 1)), "parity test");
    assert_eq!(p.matches(", 15;").count(), 2);
    for i in 0..2 {
        assert!(caught(k, |p| nudge(p, ", 15;", ", 7;", i)), "mask {i}");
    }
    assert!(caught(k, |p| p.replacen(", 4;\n    shr.u32", ", 3;\n    shr.u32", 1)), "high-nibble shift");
    // A named equivalent mutant: sign-extending the byte cannot reach the
    // result, because both nibbles are masked with 15 after the shift.
    assert!(!caught(k, |p| p.replacen("cvt.u32.u8 ", "cvt.u32.s8 ", 1)), "sign extension is masked off");
    // The product's operands: scale and zero point swapped.
    let fma = p.lines().find(|l| l.contains("fma.rn.f32")).expect("fma").trim().to_string();
    let ops: Vec<&str> = fma.trim_end_matches(';').split_whitespace().skip(1).map(|s| s.trim_end_matches(',')).collect();
    let swapped = format!("fma.rn.f32 {}, {}, {}, {};", ops[0], ops[1], ops[3], ops[2]);
    assert!(caught(k, |p| p.replacen(&fma, &swapped, 1)), "scale and zero point swapped");
}
