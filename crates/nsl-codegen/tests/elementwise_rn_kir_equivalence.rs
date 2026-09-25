//! The differential equivalence gate for the two kernels built with KIR's
//! explicitly rounded arithmetic (`KirOp::{AddRn, MulRn}`, roadmap A2 step
//! 11): `nsl_scalar_mul_add_inplace_f32` (`m[i] += g[i] * s`) and
//! `nsl_muon_scale_inv_frob_f32` (`c[i] = x[i] * (1 / (sqrt(stats[3]) +
//! 1e-7))`).
//!
//! The runtime carried them as hand-written PTX; they are now built by
//! `nsl_kir::kernels::elementwise`. This file runs the frozen hand modules
//! (`tests/fixtures/elementwise_rn_hand.rs`) and the KIR ones side by side
//! on the cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`),
//! over the grid the runtime launches (`ceil(n / 256)` blocks of 256) plus
//! one block more:
//!
//! 1. **Agreement**: under two schedules, the hand and KIR kernels leave
//!    *the same bytes* in all of global memory, over IEEE-corner inputs,
//!    scalars and sums of squares.
//! 2. **Correctness**: the result is the formula in f32 with every operation
//!    rounded on its own (Rust never contracts), and nothing past `n` moves.
//! 3. **The spelling**: the interpreter never contracts a multiply into an
//!    add, so there the `.rn` modifier changes nothing, and the mutant that
//!    drops it is equivalent under execution. On the machine it is not:
//!    ptxas contracts the bare `mul.f32` + `add.f32` of the scaled
//!    accumulate into one `FFMA`. So both modules' `.rn` spellings are
//!    pinned here, the KIR one to the hand one.
//! 4. **The gate bites**: relaxing the bound, nudging an element size, the
//!    stats slot, the epsilon or the one, a neighbour's operation, or
//!    reading the index from block 0 is caught.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::{muon_scale_inv_frob_ptx, scalar_mul_add_inplace_ptx, ELEMENTWISE_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/elementwise_rn_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const M: u64 = 0x1000_0000;
const G: u64 = 0x2000_0000;
const C: u64 = 0x3000_0000;
const STATS: u64 = 0x4000_0000;
const TAIL: usize = 300;
const POISON: u32 = 0x7FA5_A5A5;
/// What the inputs hold past `n`: finite, so a thread past the bound
/// writes something other than the poison.
const IN_TAIL: u32 = 0xC040_0000; // -3.0

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kernel {
    ScalarMulAdd,
    Muon,
}
const KERNELS: [Kernel; 2] = [Kernel::ScalarMulAdd, Kernel::Muon];

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

fn kir_ptx(k: Kernel) -> String {
    let bytes = match k {
        Kernel::ScalarMulAdd => scalar_mul_add_inplace_ptx(),
        Kernel::Muon => muon_scale_inv_frob_ptx(),
    };
    trim(&String::from_utf8(bytes).expect("ASCII"))
}

fn hand_ptx(k: Kernel) -> String {
    trim(match k {
        Kernel::ScalarMulAdd => hand::SCALAR_MUL_ADD_INPLACE_F32_PTX,
        Kernel::Muon => hand::MUON_SCALE_INV_FROB_F32_PTX,
    })
}

fn values(n: usize, seed: u64) -> Vec<u32> {
    const CORNERS: [u32; 12] = [
        0x0000_0000,
        0x8000_0000,
        0x7F80_0000,
        0xFF80_0000,
        0x7FC0_0001,
        0x0000_0001,
        0x807F_FFFF,
        0x7F7F_FFFF,
        0xFF7F_FFFF,
        0x3F80_0000,
        0x0080_0000,
        0x5F00_0000,
    ];
    let mut s = seed;
    (0..n)
        .map(|k| {
            if k < CORNERS.len() {
                return CORNERS[(k + seed as usize) % CORNERS.len()];
            }
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let v = ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0;
            let scale = [1e-30f32, 1e-3, 1.0, 1e3, 1e30][((s >> 20) % 5) as usize];
            (v * scale).to_bits()
        })
        .collect()
}

fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn with_tail(v: &[u32]) -> Vec<u8> {
    let mut b = le32(v);
    b.extend(le32(&vec![IN_TAIL; TAIL]));
    b
}

/// One launch's inputs. The scaled accumulate reads `a` as `g` and updates
/// `b` (`m`) in place with the scalar `knob`; Muon reads `a` as `x`, writes
/// `c`, and takes `knob` as the sum of squares in `stats[3]`.
struct Case<'a> {
    n: usize,
    a: &'a [u32],
    b: &'a [u32],
    knob: u32,
}

/// Run `ptx` for `k` on `case` over `ceil(n / 256) + 1` blocks. Returns all
/// of global memory.
fn run(ptx: &str, k: Kernel, case: &Case, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let n = case.n;
    let (mut global, args): (Vec<Segment>, Vec<(&str, u64)>) = match k {
        Kernel::ScalarMulAdd => (
            vec![Segment { base: M, bytes: with_tail(case.b) }, Segment { base: G, bytes: with_tail(case.a) }],
            vec![("m", M), ("g", G), ("s", case.knob as u64), ("n", n as u64)],
        ),
        Kernel::Muon => {
            // Slots 0..2 hold values the kernel must not read; slot 3 is the
            // sum of squares.
            let stats = [0x7F80_0000, 0xFF80_0000, 0x7FC0_0002, case.knob];
            (
                vec![
                    Segment { base: M, bytes: with_tail(case.a) },
                    Segment { base: C, bytes: le32(&vec![POISON; n + TAIL]) },
                    Segment { base: STATS, bytes: le32(&stats) },
                ],
                vec![("x", M), ("c", C), ("stats", STATS), ("n", n as u64)],
            )
        }
    };
    let args: HashMap<String, u64> = args.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
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

/// The knobs each kernel runs with. For the scaled accumulate, scalars
/// across the corners; for Muon, sums of squares from 0 (the scale is
/// `1 / 1e-7`) through ones where the epsilon still shows to ordinary and
/// huge ones, and the infinity and NaN corners.
fn knobs(k: Kernel) -> Vec<u32> {
    match k {
        Kernel::ScalarMulAdd => vec![
            0x0000_0000,
            0x8000_0000,
            0x3F80_0000,
            0xC070_0000, // -3.75
            0x3DCC_CCCD, // 0.1
            0x0000_0001,
            0x7F7F_FFFF,
            0x7F80_0000,
            0x7FC0_0001,
        ],
        Kernel::Muon => vec![
            0x0000_0000,
            0x2B8C_BCCC, // 1e-12: sqrt is 1e-6, and the epsilon is a tenth of it
            0x3F80_0000,
            0x42C8_0000, // 100
            0x7F7F_FFFF,
            0x7F80_0000,
            0x7FC0_0001,
        ],
    }
}

const SIZES: [usize; 4] = [1, 255, 257, 1000];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

#[test]
fn the_kir_kernels_agree_with_the_hand_written_ones_bit_for_bit() {
    for k in KERNELS {
        for (j, &n) in SIZES.iter().enumerate() {
            let (a, b) = (values(n, j as u64 + 1), values(n, j as u64 + 40));
            for knob in knobs(k) {
                let case = Case { n, a: &a, b: &b, knob };
                for order in ORDERS {
                    let hand = run(&hand_ptx(k), k, &case, order);
                    let kir = run(&kir_ptx(k), k, &case, order);
                    assert!(hand == kir, "{k:?} n={n} knob={knob:#010x} {order:?}: global memory differs");
                }
            }
        }
    }
}

fn same(got: u32, want: u32) -> bool {
    got == want || (f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan())
}

#[test]
fn the_answer_is_the_formula_with_every_operation_rounded() {
    let n = 1000;
    let (a, b) = (values(n, 7), values(n, 70));
    for k in KERNELS {
        for knob in knobs(k) {
            let case = Case { n, a: &a, b: &b, knob };
            let mem = run(&kir_ptx(k), k, &case, Order::Ascending);
            let s = f32::from_bits(knob);
            let (out, want): (Vec<u32>, Vec<u32>) = match k {
                Kernel::ScalarMulAdd => (
                    words(&mem[0]),
                    (0..n).map(|i| (f32::from_bits(b[i]) + f32::from_bits(a[i]) * s).to_bits()).collect(),
                ),
                Kernel::Muon => {
                    let inv = 1.0f32 / (s.sqrt() + f32::from_bits(0x33D6_BF95));
                    (words(&mem[1]), (0..n).map(|i| (f32::from_bits(a[i]) * inv).to_bits()).collect())
                }
            };
            for i in 0..n {
                assert!(same(out[i], want[i]), "{k:?} knob={knob:#010x} i={i}: {:#010x} vs {:#010x}", out[i], want[i]);
            }
            assert!(out[n..].iter().all(|&w| w == if k == Kernel::Muon { POISON } else { IN_TAIL }), "{k:?}: past n");
            match k {
                Kernel::ScalarMulAdd => assert_eq!(words(&mem[1])[..n], a[..], "g is untouched"),
                Kernel::Muon => assert_eq!(words(&mem[0])[..n], a[..], "x is untouched"),
            }
        }
    }
}

#[test]
fn the_kir_kernels_keep_the_ffi_signatures_and_entry_names() {
    for (k, name) in [(Kernel::ScalarMulAdd, "nsl_scalar_mul_add_inplace_f32"), (Kernel::Muon, "nsl_muon_scale_inv_frob_f32")] {
        let (h, kk) = (hand_ptx(k), kir_ptx(k));
        assert_eq!(parse_signature(&kk), parse_signature(&h), "{k:?}");
        assert!(kk.contains(&format!(".visible .entry {name}(")), "{k:?}");
        assert!(h.contains(&format!(".visible .entry {name}(")), "{k:?}");
    }
}

/// The arithmetic spellings, counted in both modules: every multiply and
/// add explicitly rounded, none bare, no `fma` written, and the same count
/// of each in the KIR module as in the hand one.
#[test]
fn the_multiplies_and_adds_are_explicitly_rounded_as_in_the_hand_kernels() {
    let forms = ["mul.rn.f32 ", "add.rn.f32 ", "sub.rn.f32 ", "div.rn.f32 ", "sqrt.rn.f32 "];
    for k in KERNELS {
        let (h, kk) = (hand_ptx(k), kir_ptx(k));
        for form in forms {
            assert_eq!(kk.matches(form).count(), h.matches(form).count(), "{k:?} {form}");
        }
        for bare in ["mul.f32 ", "add.f32 ", "sub.f32 ", "fma."] {
            assert!(!kk.contains(bare), "{k:?}: KIR module has {bare}");
            assert!(!h.contains(bare), "{k:?}: hand module has {bare}");
        }
    }
    assert_eq!(kir_ptx(Kernel::ScalarMulAdd).matches(".rn.f32 ").count(), 2);
    assert_eq!(kir_ptx(Kernel::Muon).matches(".rn.f32 ").count(), 4);
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` of `k` is told apart from the hand kernel on a
/// ragged size, under either schedule, for any of `knobs` (or faults).
fn caught_with(k: Kernel, knobs: &[u32], mutate: impl Fn(&str) -> String) -> bool {
    let n = 257;
    let (a, b) = (values(n, 3), values(n, 30));
    let hand = hand_ptx(k);
    let mutant = mutate(&kir_ptx(k));
    knobs.iter().any(|&knob| {
        let case = Case { n, a: &a, b: &b, knob };
        ORDERS.into_iter().any(|order| {
            let expect = run(&hand, k, &case, order);
            let case = &case;
            match std::panic::catch_unwind(|| run(&mutant, k, case, order)) {
                Ok(mem) => mem != expect,
                Err(_) => true,
            }
        })
    })
}

fn caught(k: Kernel, mutate: impl Fn(&str) -> String) -> bool {
    let knob = match k {
        Kernel::ScalarMulAdd => 0xC070_0000,
        Kernel::Muon => 0x3F80_0000,
    };
    caught_with(k, &[knob], mutate)
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
    for k in KERNELS {
        assert!(!caught_with(k, &knobs(k), |p| p.to_string()), "{k:?}");
    }
}

/// Named equivalent mutant: dropping `.rn` from either operation of the
/// scaled accumulate changes nothing the interpreter can see (it never
/// contracts), which is why the spelling test above exists.
#[test]
fn dropping_rn_is_invisible_to_execution_and_pinned_by_spelling_instead() {
    for form in ["mul.rn.f32 ", "add.rn.f32 "] {
        let bare = form.replace(".rn", "");
        assert!(!caught_with(Kernel::ScalarMulAdd, &knobs(Kernel::ScalarMulAdd), |p| p.replacen(form, &bare, 1)));
    }
}

#[test]
fn relaxing_the_bound_or_ignoring_the_block_index_is_caught() {
    for k in KERNELS {
        let kk = kir_ptx(k);
        assert_eq!(kk.matches("setp.ge.u64 ").count(), 1, "{k:?}");
        assert_eq!(kk.matches("%ctaid.x;").count(), 1, "{k:?}");
        assert!(caught(k, |p| p.replacen("setp.ge.u64 ", "setp.gt.u64 ", 1)), "{k:?}: bound");
        assert!(caught(k, |p| p.replacen("%ctaid.x;", "0;", 1)), "{k:?}: block index");
    }
}

/// f32's size in every address: `g`, `m` twice (load and store) for the
/// accumulate; `stats`, `x` and `c` for Muon.
#[test]
fn nudging_an_element_size_is_caught() {
    for k in KERNELS {
        assert_eq!(kir_ptx(k).matches(", 4;").count(), 3, "{k:?}");
        for i in 0..3 {
            assert!(caught(k, |p| nudge(p, ", 4;", ", 8;", i)), "{k:?}: address {i}");
        }
    }
}

#[test]
fn a_neighbours_operation_is_caught() {
    for (k, from, to) in [
        (Kernel::ScalarMulAdd, "mul.rn.f32 ", "add.rn.f32 "),
        (Kernel::ScalarMulAdd, "add.rn.f32 ", "mul.rn.f32 "),
        (Kernel::Muon, "mul.rn.f32 ", "add.rn.f32 "),
        (Kernel::Muon, "add.rn.f32 ", "mul.rn.f32 "),
    ] {
        assert!(caught(k, |p| p.replacen(from, to, 1)), "{k:?}: {from} -> {to}");
    }
}

/// Muon's baked constants: the stats slot (3), the epsilon and the one.
/// The epsilon only shows where the norm is small, so those mutants run
/// with sums of squares of 0 and 1e-12.
#[test]
fn nudging_a_muon_constant_is_caught() {
    let k = kir_ptx(Kernel::Muon);
    let small = [0x0000_0000, 0x2B8C_BCCC];
    for (from, to, knobs) in [
        ("0f33D6BF95", "0f33D6BF96", &small[..]),
        ("0f3F800000", "0f3F800001", &[0x3F80_0000][..]),
    ] {
        assert_eq!(k.matches(from).count(), 1, "{from}");
        assert!(caught_with(Kernel::Muon, knobs, |p| p.replacen(from, to, 1)), "{from} -> {to}");
    }
    let slot = k.lines().filter(|l| l.trim_start().starts_with("mov.u64 ") && l.trim_end().ends_with(", 3;")).count();
    assert_eq!(slot, 1, "the stats slot constant");
    for to in [", 2;", ", 0;"] {
        assert!(
            caught(Kernel::Muon, |p| p
                .lines()
                .map(|l| if l.trim_start().starts_with("mov.u64 ") && l.trim_end().ends_with(", 3;") {
                    l.replacen(", 3;", to, 1)
                } else {
                    l.to_string()
                })
                .collect::<Vec<_>>()
                .join("\n")
                + "\n"),
            "stats slot 3 -> {to}"
        );
    }
}
