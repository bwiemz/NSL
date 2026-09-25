//! The differential equivalence gate for the unary elementwise kernels
//! `nsl_{neg,relu,exp,log,sqrt,abs,sign,sigmoid,sin,cos,silu,gelu,clamp}_f32`
//! (roadmap A2 step 11).
//!
//! The runtime carried them as hand-written PTX
//! (`nsl_runtime::cuda::kernels::*_F32_PTX`); they are now built as KIR by
//! `nsl_kir::kernels::elementwise`. This file runs the frozen hand modules
//! (`tests/fixtures/elementwise_unary_hand.rs`) and the KIR ones side by side
//! on the cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`),
//! over the grid `gpu_elementwise_unary` launches (`ceil(n / 256)` blocks of
//! 256) plus one block more:
//!
//! 1. **Agreement**: under two schedules, the hand and KIR kernels leave
//!    *the same bytes* in all of global memory, out of place and in place
//!    (`c` aliasing `a`), over IEEE corner cases.
//! 2. **Correctness**: each output is the kernel's formula evaluated in f32
//!    with the interpreter's model of the approximate instructions (exact
//!    counterparts), bit for bit, and within a tolerance of the f64
//!    mathematical function on moderate inputs; nothing past `n` is written.
//! 3. **The gate bites**: relaxing the bound, nudging the element size or
//!    any baked constant, swapping the clamp's bounds, or reading the index
//!    from block 0 is caught.
//!
//! The interpreter models `ex2`, `lg2`, `sin`, `cos` and `rcp` `.approx` by
//! their exact counterparts; the device suites carry fidelity to the
//! machine's approximations. The hand and KIR kernels use the same
//! instructions, so they agree on the machine as they do here.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::{unary_ptx, UnaryOp, ELEMENTWISE_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/elementwise_unary_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const A: u64 = 0x1000_0000;
const C: u64 = 0x3000_0000;
const TAIL: usize = 300;
const POISON: u32 = 0x7FA5_A5A5;
/// What `a` holds past `n`: finite, so every kernel maps it to something
/// other than `c`'s poison, and a thread past the bound shows.
const A_TAIL: u32 = 0xC040_0000; // -3.0
const LO: f32 = -0.75;
const HI: f32 = 1.5;

fn kir_ptx(op: UnaryOp) -> String {
    String::from_utf8(unary_ptx(op)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(op: UnaryOp) -> String {
    use UnaryOp::*;
    match op {
        Neg => hand::NEG_F32_PTX,
        Relu => hand::RELU_F32_PTX,
        Exp => hand::EXP_F32_PTX,
        Log => hand::LOG_F32_PTX,
        Sqrt => hand::SQRT_F32_PTX,
        Abs => hand::ABS_F32_PTX,
        Sign => hand::SIGN_F32_PTX,
        Sigmoid => hand::SIGMOID_F32_PTX,
        Sin => hand::SIN_F32_PTX,
        Cos => hand::COS_F32_PTX,
        Silu => hand::SILU_F32_PTX,
        Gelu => hand::GELU_F32_PTX,
        Clamp => hand::CLAMP_F32_PTX,
    }
    .trim_end_matches('\0')
    .to_string()
}

const LOG2_E: f32 = f32::from_bits(0x3FB8_AA3B);
const LN_2: f32 = f32::from_bits(0x3F31_7218);
/// 1.702f, `nsl_gelu_f32`'s slope, written out rather than taken from
/// `nsl_kir` so a change there shows here.
const GELU_K: f32 = f32::from_bits(0x3FD9_DB23);

/// `1 / (2^(-x * log2 e) + 1)`, as the kernels compute it.
fn sigmoid_model(x: f32) -> f32 {
    1.0 / ((-x * LOG2_E).exp2() + 1.0)
}

/// The kernel's formula in f32, the approximate instructions modelled
/// exactly, as the interpreter models them.
fn model(op: UnaryOp, x: f32) -> f32 {
    use UnaryOp::*;
    match op {
        Neg => f32::from_bits(x.to_bits() ^ 0x8000_0000),
        Relu => x.max(0.0),
        Exp => (x * LOG2_E).exp2(),
        Log => x.log2() * LN_2,
        Sqrt => x.sqrt(),
        Abs => f32::from_bits(x.to_bits() & 0x7FFF_FFFF),
        Sign => {
            if x > 0.0 {
                1.0
            } else if x < 0.0 {
                -1.0
            } else {
                0.0
            }
        }
        Sigmoid => sigmoid_model(x),
        Sin => x.sin(),
        Cos => x.cos(),
        Silu => x * sigmoid_model(x),
        Gelu => x * sigmoid_model(x * GELU_K),
        Clamp => x.max(LO).min(HI),
    }
}

/// The mathematical function in f64, for the sanity check.
fn exact(op: UnaryOp, x: f64) -> f64 {
    use UnaryOp::*;
    let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
    match op {
        Neg => -x,
        Relu => x.max(0.0),
        Exp => x.exp(),
        Log => x.ln(),
        Sqrt => x.sqrt(),
        Abs => x.abs(),
        Sign => x.signum() * (x != 0.0) as u8 as f64,
        Sigmoid => sig(x),
        Sin => x.sin(),
        Cos => x.cos(),
        Silu => x * sig(x),
        Gelu => x * sig(1.702 * x),
        Clamp => x.max(LO as f64).min(HI as f64),
    }
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
        0xBF00_0000,
        0x42C8_0000, // 100: exp overflows, sigmoid saturates
    ];
    let mut s = seed;
    (0..n)
        .map(|k| {
            if k < CORNERS.len() {
                return CORNERS[(k + seed as usize) % CORNERS.len()];
            }
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 20.0 - 10.0).to_bits()
        })
        .collect()
}

fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn run(ptx: &str, op: UnaryOp, n: usize, a: &[u32], in_place: bool, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let mut a_bytes = le32(a);
    a_bytes.extend(le32(&vec![A_TAIL; TAIL]));
    let mut global = vec![Segment { base: A, bytes: a_bytes }];
    if !in_place {
        global.push(Segment { base: C, bytes: le32(&vec![POISON; n + TAIL]) });
    }
    let mut args: HashMap<String, u64> = [("a", A), ("c", if in_place { A } else { C }), ("n", n as u64)]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
    if op == UnaryOp::Clamp {
        args.insert("lo".into(), LO.to_bits() as u64);
        args.insert("hi".into(), HI.to_bits() as u64);
    }
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

const SIZES: [usize; 3] = [1, 257, 1000];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

#[test]
fn the_kir_kernels_agree_with_the_hand_written_ones_bit_for_bit() {
    for op in UnaryOp::ALL {
        for (k, &n) in SIZES.iter().enumerate() {
            let a = values(n, k as u64 + 1);
            for in_place in [false, true] {
                for order in ORDERS {
                    let hand = run(&hand_ptx(op), op, n, &a, in_place, order);
                    let kir = run(&kir_ptx(op), op, n, &a, in_place, order);
                    assert!(hand == kir, "{op:?} n={n} in_place={in_place} {order:?}: global memory differs");
                }
            }
        }
    }
}

fn same(got: u32, want: u32) -> bool {
    got == want || (f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan())
}

#[test]
fn the_answer_is_the_kernels_formula() {
    for op in UnaryOp::ALL {
        let n = 1000;
        let a = values(n, 9);
        for in_place in [false, true] {
            let mem = run(&kir_ptx(op), op, n, &a, in_place, Order::Ascending);
            let c = words(if in_place { &mem[0] } else { &mem[1] });
            for i in 0..n {
                let want = model(op, f32::from_bits(a[i])).to_bits();
                assert!(same(c[i], want), "{op:?} i={i}: {:#010x} vs {want:#010x}", c[i]);
            }
            let past = if in_place { A_TAIL } else { POISON };
            assert!(c[n..].iter().all(|&w| w == past), "{op:?}: wrote past n");
            if !in_place {
                assert_eq!(words(&mem[0])[..n], a[..], "{op:?}: a is untouched");
            }
        }
    }
}

/// On moderate inputs (the random part of `values`, |x| < 10; log and sqrt
/// on |x|), the formula is the function it names.
#[test]
fn the_formula_is_the_function_it_names() {
    for op in UnaryOp::ALL {
        for &bits in &values(500, 21)[12..] {
            let mut x = f32::from_bits(bits);
            if matches!(op, UnaryOp::Log | UnaryOp::Sqrt) {
                x = x.abs().max(1e-3);
            }
            let (got, want) = (model(op, x) as f64, exact(op, x as f64));
            assert!((got - want).abs() <= 1e-5 * want.abs().max(1.0), "{op:?}({x}) = {got}, want {want}");
        }
    }
}

#[test]
fn the_kir_kernels_keep_the_ffi_signatures_and_entry_names() {
    for op in UnaryOp::ALL {
        let (h, k) = (hand_ptx(op), kir_ptx(op));
        assert_eq!(parse_signature(&k), parse_signature(&h), "{op:?}");
        assert!(h.contains(&format!(".visible .entry {}(", op.kernel_name())), "{op:?}");
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

fn caught(op: UnaryOp, mutate: impl Fn(&str) -> String) -> bool {
    let n = 257;
    let a = values(n, 3);
    let hand = hand_ptx(op);
    let mutant = mutate(&kir_ptx(op));
    ORDERS.into_iter().any(|order| {
        let expect = run(&hand, op, n, &a, false, order);
        let a = &a;
        match std::panic::catch_unwind(|| run(&mutant, op, n, a, false, order)) {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
    })
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for op in UnaryOp::ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

#[test]
fn relaxing_the_bound_or_ignoring_the_block_index_is_caught() {
    for op in UnaryOp::ALL {
        let k = kir_ptx(op);
        assert_eq!(k.matches("setp.ge.u64 ").count(), 1, "{op:?}");
        assert_eq!(k.matches("%ctaid.x;").count(), 1, "{op:?}");
        assert!(caught(op, |p| p.replacen("setp.ge.u64 ", "setp.gt.u64 ", 1)), "{op:?}: bound");
        assert!(caught(op, |p| p.replacen("%ctaid.x;", "0;", 1)), "{op:?}: block index");
    }
}

#[test]
fn nudging_the_element_size_is_caught() {
    for op in UnaryOp::ALL {
        let k = kir_ptx(op);
        assert_eq!(k.matches(", 4;").count(), 2, "{op:?}");
        for i in 0..2 {
            let nudge = |p: &str| {
                let at = p.lines().enumerate().filter(|(_, l)| l.contains(", 4;")).nth(i).unwrap().0;
                p.lines()
                    .enumerate()
                    .map(|(j, l)| if j == at { l.replacen(", 4;", ", 8;", 1) } else { l.to_string() })
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n"
            };
            assert!(caught(op, nudge), "{op:?}: address {i}");
        }
    }
}

/// Every baked f32 constant, one ulp off: log2(e), ln 2, the
/// sigmoid's 1, relu's and sign's 0, sign's +-1, GELU's slope.
#[test]
fn nudging_a_baked_constant_is_caught() {
    use UnaryOp::*;
    for (op, from, to) in [
        (Exp, "0f3FB8AA3B", "0f3FB8AA3C"),
        (Log, "0f3F317218", "0f3F317219"),
        (Sigmoid, "0f3FB8AA3B", "0f3FB8AA3C"),
        (Sigmoid, "0f3F800000", "0f3F800001"),
        (Silu, "0f3F800000", "0f3F800001"),
        (Gelu, "0f3FD9DB23", "0f3FD9DB24"),
        (Gelu, "0f3FD9DB23", "0f3FD9999A"),
        (Gelu, "0f3FB8AA3B", "0f3FB8AA3C"),
        (Gelu, "0f3F800000", "0f3F800001"),
        (Relu, "0f00000000", "0f3F800000"),
        (Sign, "0f3F800000", "0f3F800001"),
        (Sign, "0fBF800000", "0fBF800001"),
        (Sign, "0f00000000", "0f00000001"),
    ] {
        let k = kir_ptx(op);
        assert_eq!(k.matches(from).count(), 1, "{op:?}: {from}");
        assert!(caught(op, |p| p.replacen(from, to, 1)), "{op:?}: {from} -> {to} went unnoticed");
    }
}

#[test]
fn swapping_the_clamp_bounds_is_caught() {
    assert!(caught(UnaryOp::Clamp, |p| p.replacen("[param_lo]", "[param_TMP]", 1).replacen("[param_hi]", "[param_lo]", 1).replacen("[param_TMP]", "[param_hi]", 1)));
}

#[test]
fn the_interpreter_knows_the_new_forms() {
    for (op, form) in [(UnaryOp::Neg, "neg.f32 "), (UnaryOp::Abs, "abs.f32 "), (UnaryOp::Clamp, "min.f32 ")] {
        assert!(hand_ptx(op).contains(form) && kir_ptx(op).contains(form), "{op:?}");
        parse(&hand_ptx(op));
    }
}
