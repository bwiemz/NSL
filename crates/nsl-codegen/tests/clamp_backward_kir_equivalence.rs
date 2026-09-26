//! The differential equivalence gate for `nsl_clamp_backward_f32` (roadmap
//! A2 step 11): `out[i] = (input[i] >= min_val && input[i] <= max_val) ?
//! grad[i] : 0`.
//!
//! The runtime carried it as hand-written PTX; it is now built by
//! `nsl_kir::kernels::elementwise::build_clamp_backward`. This file runs the
//! frozen hand module (`tests/fixtures/clamp_backward_hand.rs`) and the KIR
//! one side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), over the grid the runtime launches
//! (`ceil(n / 256)` blocks of 256) plus one block more:
//!
//! 1. **Agreement**: under two schedules and a spread of bounds (ordinary,
//!    equal, reversed, infinite and NaN), the two kernels leave *the same
//!    bytes* in all of global memory, over IEEE-corner gradients and inputs.
//! 2. **Correctness**: the output is the formula, bit for bit: an input on
//!    either bound passes the gradient, and a NaN input or bound passes
//!    nothing (both comparisons are ordered); nothing past `n` is written and
//!    the inputs are untouched.
//! 3. **The gate bites**: relaxing the bound, reading the index from block
//!    0, nudging an element size, making either comparison strict, joining
//!    them with `or`, swapping the bounds or the select's arms, and passing
//!    the input instead of the gradient are each caught.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::{clamp_backward_ptx, CLAMP_BACKWARD_NAME, ELEMENTWISE_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/clamp_backward_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const GRAD: u64 = 0x1000_0000;
const INPUT: u64 = 0x2000_0000;
const OUT: u64 = 0x3000_0000;
const TAIL: usize = 300;
const POISON: u32 = 0x7FA5_A5A5;
/// What the inputs hold past `n`: finite and inside every ordinary bound, so
/// a thread past the bound writes something other than the poison.
const IN_TAIL: u32 = 0x3F00_0000; // 0.5

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

fn kir_ptx() -> String {
    trim(&String::from_utf8(clamp_backward_ptx()).expect("ASCII"))
}

fn hand_ptx() -> String {
    trim(hand::CLAMP_BACKWARD_F32_PTX)
}

/// The bounds each agreement run uses: ordinary, one-sided infinite, a
/// single point, reversed (nothing passes), and a NaN on either side.
const BOUNDS: [(f32, f32); 7] = [
    (-1.0, 1.0),
    (-2.5, 0.25),
    (f32::NEG_INFINITY, 0.0),
    (0.5, 0.5),
    (1.0, -1.0),
    (f32::NAN, 1.0),
    (-1.0, f32::NAN),
];

fn values(n: usize, seed: u64) -> Vec<u32> {
    const CORNERS: [u32; 14] = [
        0x0000_0000, // +0
        0x8000_0000, // -0
        0x7F80_0000, // +inf
        0xFF80_0000, // -inf
        0x7FC0_0001, // NaN
        0x0000_0001, // the least subnormal
        0x807F_FFFF,
        0x3F80_0000, // 1: on the ordinary upper bound
        0xBF80_0000, // -1: on the ordinary lower bound
        0x3F80_0001, // just above 1
        0xBF80_0001, // just below -1
        0x3F00_0000, // 0.5: the point bound
        0x3E80_0000, // 0.25
        0xC020_0000, // -2.5
    ];
    let mut s = seed;
    (0..n)
        .map(|k| {
            if k < CORNERS.len() {
                return CORNERS[(k + seed as usize) % CORNERS.len()];
            }
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 6.0 - 3.0).to_bits()
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

/// Run `ptx` on `(g, x)` with bounds `(lo, hi)` over `ceil(n / 256) + 1`
/// blocks. Returns all of global memory: `grad`, `input`, `out`.
fn run(ptx: &str, g: &[u32], x: &[u32], (lo, hi): (f32, f32), order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let n = g.len();
    let mut global = vec![
        Segment { base: GRAD, bytes: with_tail(g) },
        Segment { base: INPUT, bytes: with_tail(x) },
        Segment { base: OUT, bytes: le32(&vec![POISON; n + TAIL]) },
    ];
    let args: HashMap<String, u64> = [
        ("grad", GRAD),
        ("input", INPUT),
        ("out", OUT),
        ("min_val", lo.to_bits() as u64),
        ("max_val", hi.to_bits() as u64),
        ("n", n as u64),
    ]
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

const SIZES: [usize; 4] = [1, 255, 257, 1000];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

#[test]
fn the_kir_kernel_agrees_with_the_hand_written_one_bit_for_bit() {
    for (j, &n) in SIZES.iter().enumerate() {
        let (g, x) = (values(n, j as u64 + 1), values(n, j as u64 + 40));
        for bounds in BOUNDS {
            for order in ORDERS {
                let hand = run(&hand_ptx(), &g, &x, bounds, order);
                let kir = run(&kir_ptx(), &g, &x, bounds, order);
                assert!(hand == kir, "n={n} bounds={bounds:?} {order:?}: global memory differs");
            }
        }
    }
}

/// `input >= lo && input <= hi` passes the gradient, anything else 0; a NaN
/// on any side fails both ordered comparisons.
fn model(g: u32, x: u32, (lo, hi): (f32, f32)) -> u32 {
    let x = f32::from_bits(x);
    if x >= lo && x <= hi {
        g
    } else {
        0.0f32.to_bits()
    }
}

#[test]
fn the_answer_is_the_formula_bit_for_bit() {
    let n = 1000;
    let (g, x) = (values(n, 7), values(n, 70));
    for bounds in BOUNDS {
        let mem = run(&kir_ptx(), &g, &x, bounds, Order::Ascending);
        let out = words(&mem[2]);
        for i in 0..n {
            let want = model(g[i], x[i], bounds);
            assert_eq!(out[i], want, "bounds={bounds:?} i={i}: x={:e}", f32::from_bits(x[i]));
        }
        assert!(out[n..].iter().all(|&w| w == POISON), "bounds={bounds:?}: wrote past n");
        assert_eq!(words(&mem[0])[..n], g[..], "grad is untouched");
        assert_eq!(words(&mem[1])[..n], x[..], "input is untouched");
    }
    // The corners the formula turns on, one by one, at [-1, 1].
    let (g1, one) = (0x4040_0000u32, 1.0f32); // grad 3.0
    for (x, passes) in [(one, true), (-one, true), (f32::from_bits(0x3F80_0001), false), (f32::NAN, false), (0.0, true), (-0.0, true)] {
        let mem = run(&kir_ptx(), &[g1], &[x.to_bits()], (-1.0, 1.0), Order::Ascending);
        assert_eq!(words(&mem[2])[0], if passes { g1 } else { 0 }, "x={x:e}");
    }
}

#[test]
fn the_kir_kernel_keeps_the_ffi_signature_and_entry_name() {
    let (h, k) = (hand_ptx(), kir_ptx());
    assert_eq!(parse_signature(&k), parse_signature(&h));
    let names: Vec<String> = parse_signature(&h).into_iter().map(|(_, name)| name).collect();
    assert_eq!(names, ["grad", "input", "out", "min_val", "max_val", "n"]);
    for text in [&h, &k] {
        assert!(text.contains(&format!(".visible .entry {CLAMP_BACKWARD_NAME}(")));
    }
}

/// The comparison and the select are spelled as in the hand kernel.
#[test]
fn the_comparisons_are_spelled_as_in_the_hand_kernel() {
    let (h, k) = (hand_ptx(), kir_ptx());
    for form in ["setp.ge.f32 ", "setp.le.f32 ", "and.pred ", "selp.f32 ", "ld.param.f32 "] {
        assert_eq!(k.matches(form).count(), h.matches(form).count(), "{form}");
    }
    for never in ["setp.gt.f32", "setp.lt.f32", "or.pred", "fma.", "mul.f32", "add.f32"] {
        assert!(!k.contains(never), "{never}");
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on a ragged size,
/// under any bound set and either schedule (or faults).
fn caught(mutate: impl Fn(&str) -> String) -> bool {
    let n = 257;
    let (g, x) = (values(n, 3), values(n, 30));
    let hand = hand_ptx();
    let mutant = mutate(&kir_ptx());
    BOUNDS.into_iter().any(|bounds| {
        ORDERS.into_iter().any(|order| {
            let expect = run(&hand, &g, &x, bounds, order);
            let (g, x) = (&g, &x);
            match std::panic::catch_unwind(|| run(&mutant, g, x, bounds, order)) {
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
    assert_eq!(k.matches("setp.ge.u64 ").count(), 1);
    assert_eq!(k.matches("%ctaid.x;").count(), 1);
    assert!(caught(|p| p.replacen("setp.ge.u64 ", "setp.gt.u64 ", 1)), "bound");
    assert!(caught(|p| p.replacen("%ctaid.x;", "0;", 1)), "block index");
}

/// f32's size in every address: the two loads and the store.
#[test]
fn nudging_an_element_size_is_caught() {
    assert_eq!(kir_ptx().matches(", 4;").count(), 3);
    for i in 0..3 {
        assert!(caught(|p| nudge(p, ", 4;", ", 8;", i)), "address {i}");
    }
}

/// An input exactly on a bound passes: a strict comparison on either side
/// would drop it.
#[test]
fn a_strict_comparison_on_either_bound_is_caught() {
    assert!(caught(|p| p.replacen("setp.ge.f32 ", "setp.gt.f32 ", 1)), "lower bound strict");
    assert!(caught(|p| p.replacen("setp.le.f32 ", "setp.lt.f32 ", 1)), "upper bound strict");
}

#[test]
fn either_comparison_alone_is_caught() {
    assert!(caught(|p| p.replacen("and.pred ", "or.pred ", 1)), "or");
}

#[test]
fn swapping_the_bounds_or_the_select_is_caught() {
    assert!(caught(|p| p.replacen("[param_min_val]", "[param_max_val]", 1)), "min read as max");
    assert!(caught(|p| p.replacen("[param_max_val]", "[param_min_val]", 1)), "max read as min");
    // selp d, a, b, p -> selp d, b, a, p: the gradient where it should be 0.
    let swapped = |p: &str| {
        let line = p.lines().find(|l| l.contains("selp.f32 ")).expect("a select").to_string();
        let (head, args) = line.split_once("selp.f32 ").expect("mnemonic");
        let ops: Vec<&str> = args.trim_end_matches(';').split(',').map(str::trim).collect();
        let fixed = format!("{head}selp.f32 {}, {}, {}, {};", ops[0], ops[2], ops[1], ops[3]);
        p.replacen(&line, &fixed, 1)
    };
    assert!(caught(swapped), "select arms swapped");
}

#[test]
fn passing_the_input_instead_of_the_gradient_is_caught() {
    assert!(caught(|p| p.replacen("[param_grad]", "[param_input]", 1)));
}
