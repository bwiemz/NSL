//! The differential equivalence gate for `nsl_dequant_fp8_e4m3_f32`, the
//! KV-cache FP8 read path, now built by
//! `nsl_kir::kernels::dequant::build_fp8_e4m3` (new-roadmap item 5).
//!
//! The runtime carried it as hand-written PTX in `cuda/fused_kernels.rs`.
//! This file runs the frozen hand module (`tests/fixtures/fp8_e4m3_hand.rs`)
//! and the KIR one side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`). Its companion in `nsl-runtime`,
//! `tests/fp8_e4m3_dequant_interp.rs`, now runs the KIR module the runtime
//! launches against the OCP reference and the CPU decoder.
//!
//! 1. **Agreement**: over all 256 codes, in ragged batches and under two
//!    schedules, the two modules leave *the same bytes* in all of global
//!    memory.
//! 2. **Correctness**: every code is the OCP E4M3 value, restated here: bias
//!    7, no infinities, `S.1111.111` the only NaN (quiet, `0x7FC00000`), and
//!    exponent field 0 the subnormals `m · 2^-9` with signed zeros.
//! 3. **The gate bites**: the bound, the block index, the output element
//!    size, the sign, exponent and mantissa extraction, the NaN mask and test,
//!    the subnormal test and scale, the exponent rebias, both shifts into
//!    place and the sign OR on the subnormal path are caught.

use std::collections::HashMap;

use nsl_kir::kernels::dequant::{fp8_e4m3_ptx, FP8_E4M3_NAME};
use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;

#[allow(dead_code)]
#[path = "fixtures/fp8_e4m3_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const INP: u64 = 0x10_0000;
const OUT: u64 = 0x20_0000;
const TAIL: usize = 300;
const POISON: u32 = 0xA5A5_A5A5;

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

fn kir_ptx() -> String {
    trim(&String::from_utf8(fp8_e4m3_ptx()).expect("ASCII"))
}

fn hand_ptx() -> String {
    trim(hand::DEQUANT_FP8_E4M3_F32_PTX)
}

/// OCP E4M3, restated: the f32 bits of `code`.
fn ocp_e4m3(code: u8) -> u32 {
    let sign = ((code >> 7) as u32) << 31;
    let e = ((code >> 3) & 15) as i32;
    let m = (code & 7) as u32;
    if code & 0x7F == 0x7F {
        return 0x7FC0_0000;
    }
    if e == 0 {
        return (m as f32 * 2f32.powi(-9)).to_bits() | sign;
    }
    sign | (((e - 7 + 127) as u32) << 23) | (m << 20)
}

#[test]
fn the_restated_reference_is_ocp_e4m3() {
    assert_eq!(f32::from_bits(ocp_e4m3(0x7E)), 448.0, "max normal");
    assert_eq!(f32::from_bits(ocp_e4m3(0x08)), 2f32.powi(-6), "min normal");
    assert_eq!(f32::from_bits(ocp_e4m3(0x01)), 2f32.powi(-9), "min subnormal");
    assert_eq!(ocp_e4m3(0x80), 0x8000_0000, "-0");
    assert_eq!(f32::from_bits(ocp_e4m3(0x38)), 1.0);
    assert!(f32::from_bits(ocp_e4m3(0xFF)).is_nan() && f32::from_bits(ocp_e4m3(0x7F)).is_nan());
}

/// Every code, in an order that differs per seed so each block sees a mix.
fn codes(n: usize, seed: u64) -> Vec<u8> {
    (0..n).map(|k| (k as u64).wrapping_mul(97).wrapping_add(seed * 31) as u8).collect()
}

fn launch(ptx: &str, input: &[u8], order: Order) -> Vec<Vec<u8>> {
    let n = input.len();
    let mut inp = input.to_vec();
    inp.extend(vec![0x38u8; TAIL]);
    let mut global = vec![
        Segment { base: INP, bytes: inp },
        Segment { base: OUT, bytes: std::iter::repeat_n(POISON.to_le_bytes(), n + TAIL).flatten().collect() },
    ];
    let args: HashMap<String, u64> =
        [("inp", INP), ("out", OUT), ("n", n as u64)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    let prog = parse(ptx);
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

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

const SIZES: [usize; 4] = [1, 256, 257, 1000];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

#[test]
fn the_kir_kernel_agrees_with_the_hand_written_one_bit_for_bit() {
    let (hand, kir) = (hand_ptx(), kir_ptx());
    for (j, &n) in SIZES.iter().enumerate() {
        let c = codes(n, j as u64);
        for order in ORDERS {
            assert!(launch(&hand, &c, order) == launch(&kir, &c, order), "n={n} {order:?}");
        }
    }
}

#[test]
fn every_code_is_its_ocp_value_bit_for_bit() {
    let all: Vec<u8> = (0..=255).collect();
    let mem = launch(&kir_ptx(), &all, Order::Ascending);
    let out = words(&mem[1]);
    for code in 0..=255u8 {
        assert_eq!(out[code as usize], ocp_e4m3(code), "code {code:#04x}");
    }
    assert!(out[256..].iter().all(|&w| w == POISON), "wrote past n");
}

#[test]
fn the_kir_kernel_keeps_the_ffi_signature_and_entry_name() {
    let (h, k) = (hand_ptx(), kir_ptx());
    assert_eq!(parse_signature(&k), parse_signature(&h));
    for p in [&h, &k] {
        assert!(p.contains(&format!(".visible .entry {FP8_E4M3_NAME}(")));
    }
}

#[test]
fn the_arithmetic_is_spelled_as_in_the_hand_kernel() {
    let (h, k) = (hand_ptx(), kir_ptx());
    for form in ["ld.global.u8 ", "cvt.rn.f32.u32 ", "mul.f32 ", "st.global.f32 "] {
        assert_eq!(k.matches(form).count(), h.matches(form).count(), "{form}");
    }
    for never in ["fma.", "mul.rn.f32", "add.f32"] {
        assert!(!k.contains(never), "{never}");
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel over all 256
/// codes (plus a ragged batch), under either schedule (or faults).
fn caught(mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx();
    let mutant = mutate(&kir_ptx());
    let batches = [(0..=255).collect::<Vec<u8>>(), codes(257, 5)];
    batches.iter().any(|c| {
        ORDERS.into_iter().any(|o| {
            let expect = launch(&hand, c, o);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| launch(&mutant, c, o))) {
                Ok(mem) => mem != expect,
                Err(_) => true,
            }
        })
    })
}

/// Replace `from` with `to` on the `i`-th line that ends with `from`.
fn nudge_end(ptx: &str, from: &str, to: &str, i: usize) -> String {
    let at = ptx.lines().enumerate().filter(|(_, l)| l.ends_with(from)).nth(i).expect("the line").0;
    ptx.lines()
        .enumerate()
        .map(|(k, l)| if k == at { format!("{}{to}", &l[..l.len() - from.len()]) } else { l.to_string() })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

fn count_end(ptx: &str, suffix: &str) -> usize {
    ptx.lines().filter(|l| l.ends_with(suffix)).count()
}

#[test]
fn the_unmutated_kernel_is_not_caught() {
    assert!(!caught(|p| p.to_string()));
}

#[test]
fn relaxing_the_bound_or_ignoring_the_block_index_is_caught() {
    assert!(caught(|p| p.replacen("setp.ge.u64 ", "setp.gt.u64 ", 1)), "bound");
    assert!(caught(|p| p.replacen("%ctaid.x;", "0;", 1)), "block index");
    assert!(caught(|p| p.replacen(", 4;", ", 8;", 1)), "output element size");
    assert!(caught(|p| p.replacen("[param_n];", "[param_out];", 1)), "n slot");
}

/// Every integer constant the decoder uses, nudged: the sign shift (7) and
/// mask (1, dropped; widening it is a named equivalent mutant) and its move
/// into place (31); the exponent shift (3) and mask
/// (15); the mantissa mask (7); the NaN mask and value (127); the rebias
/// (120) and both shifts into place (23, 20); the subnormal test (0).
#[test]
fn every_decoder_constant_is_caught() {
    let k = kir_ptx();
    for (c, to) in [
        (", 1;", ", 0;"),
        (", 31;", ", 30;"),
        (", 3;", ", 4;"),
        (", 15;", ", 7;"),
        (", 120;", ", 121;"),
        (", 23;", ", 22;"),
        (", 20;", ", 19;"),
    ] {
        let sites = count_end(&k, c);
        assert!(sites >= 1, "{c} missing: {k}");
        for i in 0..sites {
            assert!(caught(|p| nudge_end(p, c, to, i)), "{c} site {i}");
        }
    }
    // 7 is both the sign shift and the mantissa mask; 127 is both the NaN
    // mask and the value it is compared to. Every site of each matters.
    for (c, to) in [(", 7;", ", 6;"), (", 127;", ", 126;")] {
        let sites = count_end(&k, c);
        assert_eq!(sites, 2, "{c}: {k}");
        for i in 0..sites {
            assert!(caught(|p| nudge_end(p, c, to, i)), "{c} site {i}");
        }
    }
    // A named equivalent mutant: `w >> 7` of a byte is already 0 or 1, so
    // widening the sign mask from 1 to 3 changes nothing.
    let sign_mask = count_end(&k, ", 1;");
    assert!((0..sign_mask).all(|i| !caught(|p| nudge_end(p, ", 1;", ", 3;", i))), "the sign mask's width is invisible");
    assert!(caught(|p| p.replacen("0f3B000000;", "0f3B800000;", 1)), "subnormal scale 2^-9");
    assert!(caught(|p| p.replacen(", 2143289344;", ", 2139095040;", 1)), "quiet NaN as +inf");
}

/// Both branch tests inverted, and the ORs that assemble a value dropped.
#[test]
fn the_branches_and_the_assembly_are_caught() {
    let k = kir_ptx();
    let tests = k.matches("setp.eq.u32 ").count();
    assert_eq!(tests, 2, "NaN and subnormal: {k}");
    for i in 0..tests {
        let at = k.lines().enumerate().filter(|(_, l)| l.contains("setp.eq.u32 ")).nth(i).expect("setp").0;
        assert!(
            caught(|p| p
                .lines()
                .enumerate()
                .map(|(n, l)| if n == at { l.replacen("setp.eq.u32 ", "setp.ne.u32 ", 1) } else { l.to_string() })
                .collect::<Vec<_>>()
                .join("\n")
                + "\n"),
            "branch {i} inverted"
        );
    }
    let ors = k.matches("or.b32 ").count();
    assert_eq!(ors, 3, "two in the normal assembly, one sign on the subnormal: {k}");
    for i in 0..ors {
        let at = k.lines().enumerate().filter(|(_, l)| l.contains("or.b32 ")).nth(i).expect("or").0;
        assert!(
            caught(|p| p
                .lines()
                .enumerate()
                .map(|(n, l)| if n == at { l.replacen("or.b32 ", "xor.b32 ", 1).replacen("xor.b32", "and.b32", 1) } else { l.to_string() })
                .collect::<Vec<_>>()
                .join("\n")
                + "\n"),
            "or {i}"
        );
    }
    assert!(caught(|p| p.replacen("mul.f32 ", "add.f32 ", 1)), "subnormal product");
}
