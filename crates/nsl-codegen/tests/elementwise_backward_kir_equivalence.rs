//! The differential equivalence gate for the activation-backward kernels
//! (roadmap A2 step 11): the tape-AD `nsl_{relu,sigmoid,tanh,silu}_backward_f32`,
//! the source-AD `nsl_{sigmoid,tanh,silu,gelu}_backward_srcad_f32`, and the
//! fused SwiGLU gate adjoint `nsl_swiglu_gate_backward_f32`.
//!
//! The runtime carried them as hand-written PTX; they are now built by
//! `nsl_kir::kernels::elementwise`. This file runs the frozen hand modules
//! (`tests/fixtures/elementwise_backward_hand.rs`) and the KIR ones side by
//! side on the cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`),
//! over the grid the runtime launches (`ceil(n / 256)` blocks of 256) plus
//! one block more:
//!
//! 1. **Agreement**: under two schedules, the hand and KIR kernels leave
//!    *the same bytes* in all of global memory, over IEEE-corner gradients
//!    and saved values.
//! 2. **Correctness**: the output is the kernel's formula in f32 with every
//!    operation rounded on its own and the approximate instructions modelled
//!    exactly (as the interpreter models them), bit for bit, and on moderate
//!    inputs within a tolerance of the f64 derivative it names; nothing past
//!    `n` is written and the inputs are untouched.
//! 3. **The spelling**: the interpreter never contracts, so it cannot tell a
//!    bare `mul.f32`/`sub.f32` pair from an explicitly rounded one. On the
//!    machine ptxas contracts the bare pair into one `FFMA`: the tape-AD
//!    kernels always allowed that, and the source-AD kernels must not (they
//!    match a chain of separate launches bit for bit). So every arithmetic
//!    mnemonic is counted in both modules and the KIR count pinned to the
//!    hand one; dropping `.rn` is named below as the equivalent mutant this
//!    pin exists for.
//! 4. **The gate bites**: relaxing the bound, reading the index from block
//!    0, nudging an element size or a baked constant, a neighbour's
//!    operation, or relu's strict comparison is caught.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::{backward_ptx, BackwardOp, ELEMENTWISE_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/elementwise_backward_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const GRAD: u64 = 0x1000_0000;
const SAVED: u64 = 0x2000_0000;
const UP: u64 = 0x3000_0000;
const OUT: u64 = 0x4000_0000;
const TAIL: usize = 300;
const POISON: u32 = 0x7FA5_A5A5;
/// What the inputs hold past `n`: finite, so a thread past the bound writes
/// something other than the poison.
const IN_TAIL: u32 = 0xC040_0000; // -3.0

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

fn kir_ptx(op: BackwardOp) -> String {
    trim(&String::from_utf8(backward_ptx(op)).expect("ASCII"))
}

fn hand_ptx(op: BackwardOp) -> String {
    use BackwardOp::*;
    trim(match op {
        Relu => hand::RELU_BACKWARD_F32_PTX,
        Sigmoid => hand::SIGMOID_BACKWARD_F32_PTX,
        Tanh => hand::TANH_BACKWARD_F32_PTX,
        Silu => hand::SILU_BACKWARD_F32_PTX,
        SigmoidSrcad => hand::SIGMOID_BACKWARD_SRCAD_F32_PTX,
        TanhSrcad => hand::TANH_BACKWARD_SRCAD_F32_PTX,
        SiluSrcad => hand::SILU_BACKWARD_SRCAD_F32_PTX,
        GeluSrcad => hand::GELU_BACKWARD_SRCAD_F32_PTX,
        SwigluGate => hand::SWIGLU_GATE_BACKWARD_F32_PTX,
    })
}

/// The saved operand's parameter name: the tape-AD sigmoid and tanh read the
/// forward's output as `saved`.
fn saved_name(op: BackwardOp) -> &'static str {
    if matches!(op, BackwardOp::Sigmoid | BackwardOp::Tanh) {
        "saved"
    } else {
        "input"
    }
}

const LOG2_E: f32 = f32::from_bits(0x3FB8_AA3B);
/// 1.702f, the GELU slope, written out rather than taken from `nsl_kir` so a
/// change there shows here.
const GELU_K: f32 = f32::from_bits(0x3FD9_DB23);

/// `1 / (2^(-x * log2 e) + 1)`, as the kernels compute it.
fn sigmoid_model(x: f32) -> f32 {
    1.0 / ((-x * LOG2_E).exp2() + 1.0)
}

/// The kernel's formula in f32, every operation rounded on its own (Rust
/// never contracts), in the kernel's operand order.
fn model(op: BackwardOp, g: f32, x: f32, u: f32) -> f32 {
    use BackwardOp::*;
    let srcad_tail = |x: f32, s: f32| s * (1.0 + x * (1.0 - s));
    match op {
        Relu => {
            if x > 0.0 {
                g
            } else {
                0.0
            }
        }
        Sigmoid | SigmoidSrcad => g * (x * (1.0 - x)),
        Tanh | TanhSrcad => g * (1.0 - x * x),
        Silu => {
            let s = sigmoid_model(x);
            g * (s + s * (x * (1.0 - s)))
        }
        SiluSrcad => g * srcad_tail(x, sigmoid_model(x)),
        GeluSrcad => {
            let kx = x * GELU_K;
            g * srcad_tail(kx, sigmoid_model(kx))
        }
        SwigluGate => (g * u) * srcad_tail(x, sigmoid_model(x)),
    }
}

/// The derivative in f64 the kernel names, times the upstream gradient.
fn exact(op: BackwardOp, g: f64, x: f64, u: f64) -> f64 {
    use BackwardOp::*;
    let sig = |v: f64| 1.0 / (1.0 + (-v).exp());
    match op {
        Relu => {
            if x > 0.0 {
                g
            } else {
                0.0
            }
        }
        Sigmoid | SigmoidSrcad => g * x * (1.0 - x),
        Tanh | TanhSrcad => g * (1.0 - x * x),
        Silu | SiluSrcad => g * (sig(x) + x * sig(x) * (1.0 - sig(x))),
        GeluSrcad => {
            let k = 1.702;
            g * (sig(k * x) + x * k * sig(k * x) * (1.0 - sig(k * x)))
        }
        SwigluGate => g * u * (sig(x) + x * sig(x) * (1.0 - sig(x))),
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
        0x42C8_0000, // 100: the sigmoid saturates
    ];
    let mut s = seed;
    (0..n)
        .map(|k| {
            if k < CORNERS.len() {
                return CORNERS[(k + seed as usize) % CORNERS.len()];
            }
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 12.0 - 6.0).to_bits()
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

/// One launch's inputs: the upstream gradient, the saved operand, and (for
/// the SwiGLU gate) the up projection.
struct Case<'a> {
    n: usize,
    g: &'a [u32],
    x: &'a [u32],
    u: &'a [u32],
}

/// Run `ptx` for `op` on `case` over `ceil(n / 256) + 1` blocks. Returns all
/// of global memory: `grad`, the saved operand, `up`, `out`.
fn run(ptx: &str, op: BackwardOp, case: &Case, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let n = case.n;
    let mut global = vec![
        Segment { base: GRAD, bytes: with_tail(case.g) },
        Segment { base: SAVED, bytes: with_tail(case.x) },
        Segment { base: UP, bytes: with_tail(case.u) },
        Segment { base: OUT, bytes: le32(&vec![POISON; n + TAIL]) },
    ];
    let mut args: HashMap<String, u64> =
        [("grad", GRAD), (saved_name(op), SAVED), ("out", OUT), ("n", n as u64)]
            .into_iter()
            .map(|(k, v)| (k.to_string(), v))
            .collect();
    if op == BackwardOp::SwigluGate {
        args.insert("up".into(), UP);
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

const SIZES: [usize; 4] = [1, 255, 257, 1000];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

#[test]
fn the_kir_kernels_agree_with_the_hand_written_ones_bit_for_bit() {
    for op in BackwardOp::ALL {
        for (j, &n) in SIZES.iter().enumerate() {
            let (g, x, u) = (values(n, j as u64 + 1), values(n, j as u64 + 40), values(n, j as u64 + 80));
            let case = Case { n, g: &g, x: &x, u: &u };
            for order in ORDERS {
                let hand = run(&hand_ptx(op), op, &case, order);
                let kir = run(&kir_ptx(op), op, &case, order);
                assert!(hand == kir, "{op:?} n={n} {order:?}: global memory differs");
            }
        }
    }
}

fn same(got: u32, want: u32) -> bool {
    got == want || (f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan())
}

#[test]
fn the_answer_is_the_kernels_formula_with_every_operation_rounded() {
    let n = 1000;
    let (g, x, u) = (values(n, 7), values(n, 70), values(n, 700));
    for op in BackwardOp::ALL {
        let case = Case { n, g: &g, x: &x, u: &u };
        let mem = run(&kir_ptx(op), op, &case, Order::Ascending);
        let out = words(&mem[3]);
        for i in 0..n {
            let want = model(op, f32::from_bits(g[i]), f32::from_bits(x[i]), f32::from_bits(u[i])).to_bits();
            assert!(same(out[i], want), "{op:?} i={i}: {:#010x} vs {want:#010x}", out[i]);
        }
        assert!(out[n..].iter().all(|&w| w == POISON), "{op:?}: wrote past n");
        assert_eq!(words(&mem[0])[..n], g[..], "{op:?}: grad is untouched");
        assert_eq!(words(&mem[1])[..n], x[..], "{op:?}: the saved operand is untouched");
        assert_eq!(words(&mem[2])[..n], u[..], "{op:?}: up is untouched");
    }
}

/// On moderate inputs (the random part of `values`, |x| < 6), the formula is
/// the derivative it names.
#[test]
fn the_formula_is_the_derivative_it_names() {
    let (g, x, u) = (values(500, 21), values(500, 22), values(500, 23));
    for op in BackwardOp::ALL {
        for i in 12..500 {
            let (gf, xf, uf) = (f32::from_bits(g[i]), f32::from_bits(x[i]), f32::from_bits(u[i]));
            let got = model(op, gf, xf, uf) as f64;
            let want = exact(op, gf as f64, xf as f64, uf as f64);
            assert!((got - want).abs() <= 1e-5 * want.abs().max(1.0), "{op:?}({gf}, {xf}, {uf}) = {got}, want {want}");
        }
    }
}

#[test]
fn the_kir_kernels_keep_the_ffi_signatures_and_entry_names() {
    for op in BackwardOp::ALL {
        let (h, k) = (hand_ptx(op), kir_ptx(op));
        assert_eq!(parse_signature(&k), parse_signature(&h), "{op:?}");
        let names: Vec<String> = parse_signature(&h).into_iter().map(|(_, name)| name).collect();
        assert_eq!(names, op.param_names(), "{op:?}");
        assert!(h.contains(&format!(".visible .entry {}(", op.kernel_name())), "{op:?}");
        assert!(k.contains(&format!(".visible .entry {}(", op.kernel_name())), "{op:?}");
    }
}

/// Every arithmetic mnemonic, counted in both modules. The tape-AD kernels
/// are bare throughout (ptxas may contract them, as it always could); the
/// source-AD kernels round every derivative operation and keep the sigmoid
/// bare, as `nsl_sigmoid_f32` does. Neither writes an `fma` or divides.
#[test]
fn the_arithmetic_is_spelled_as_in_the_hand_kernels() {
    let forms = [
        "mul.f32 ",
        "add.f32 ",
        "sub.f32 ",
        "mul.rn.f32 ",
        "add.rn.f32 ",
        "sub.rn.f32 ",
        "neg.f32 ",
        "ex2.approx.f32 ",
        "rcp.approx.f32 ",
        "setp.gt.f32 ",
        "selp.f32 ",
    ];
    for op in BackwardOp::ALL {
        let (h, k) = (hand_ptx(op), kir_ptx(op));
        for form in forms {
            assert_eq!(k.matches(form).count(), h.matches(form).count(), "{op:?} {form}");
        }
        for never in ["fma.", "div.", "mad."] {
            assert!(!k.contains(never) && !h.contains(never), "{op:?}: {never}");
        }
    }
    use BackwardOp::*;
    for (op, rounded) in [(SigmoidSrcad, 3), (TanhSrcad, 3), (SiluSrcad, 5), (GeluSrcad, 6), (SwigluGate, 6)] {
        assert_eq!(kir_ptx(op).matches(".rn.f32 ").count(), rounded, "{op:?}");
    }
    for op in [Relu, Sigmoid, Tanh, Silu] {
        assert!(!kir_ptx(op).contains(".rn."), "{op:?}");
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` of `op` is told apart from the hand kernel on a
/// ragged size, under either schedule (or faults).
fn caught(op: BackwardOp, mutate: impl Fn(&str) -> String) -> bool {
    let n = 257;
    let (g, x, u) = (values(n, 3), values(n, 30), values(n, 60));
    let case = Case { n, g: &g, x: &x, u: &u };
    let hand = hand_ptx(op);
    let mutant = mutate(&kir_ptx(op));
    ORDERS.into_iter().any(|order| {
        let expect = run(&hand, op, &case, order);
        let case = &case;
        match std::panic::catch_unwind(|| run(&mutant, op, case, order)) {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
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
    for op in BackwardOp::ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

/// Named equivalent mutant: dropping `.rn` from any source-AD operation
/// changes nothing the interpreter can see (it never contracts), which is
/// why the spelling test above exists.
#[test]
fn dropping_rn_is_invisible_to_execution_and_pinned_by_spelling_instead() {
    for op in [BackwardOp::TanhSrcad, BackwardOp::SiluSrcad, BackwardOp::GeluSrcad, BackwardOp::SwigluGate] {
        for form in ["mul.rn.f32 ", "add.rn.f32 ", "sub.rn.f32 "] {
            if kir_ptx(op).contains(form) {
                let bare = form.replace(".rn", "");
                assert!(!caught(op, |p| p.replace(form, &bare)), "{op:?}: {form}");
            }
        }
    }
}

#[test]
fn relaxing_the_bound_or_ignoring_the_block_index_is_caught() {
    for op in BackwardOp::ALL {
        let k = kir_ptx(op);
        assert_eq!(k.matches("setp.ge.u64 ").count(), 1, "{op:?}");
        assert_eq!(k.matches("%ctaid.x;").count(), 1, "{op:?}");
        assert!(caught(op, |p| p.replacen("setp.ge.u64 ", "setp.gt.u64 ", 1)), "{op:?}: bound");
        assert!(caught(op, |p| p.replacen("%ctaid.x;", "0;", 1)), "{op:?}: block index");
    }
}

/// f32's size in every address: the loads and the store.
#[test]
fn nudging_an_element_size_is_caught() {
    for op in BackwardOp::ALL {
        let addresses = if op == BackwardOp::SwigluGate { 4 } else { 3 };
        assert_eq!(kir_ptx(op).matches(", 4;").count(), addresses, "{op:?}");
        for i in 0..addresses {
            assert!(caught(op, |p| nudge(p, ", 4;", ", 8;", i)), "{op:?}: address {i}");
        }
    }
}

/// Every baked f32 constant, one ulp off, each occurrence on its own: the
/// ones, log2(e), the GELU slope, relu's zeros.
#[test]
fn nudging_a_baked_constant_is_caught() {
    use BackwardOp::*;
    for (op, from, to) in [
        (Relu, "0f00000000", "0f3F800000"),
        (Sigmoid, "0f3F800000", "0f3F800001"),
        (Tanh, "0f3F800000", "0f3F800001"),
        (Silu, "0f3F800000", "0f3F800001"),
        (Silu, "0f3FB8AA3B", "0f3FB8AA3C"),
        (SigmoidSrcad, "0f3F800000", "0f3F800001"),
        (TanhSrcad, "0f3F800000", "0f3F800001"),
        (SiluSrcad, "0f3F800000", "0f3F800001"),
        (SiluSrcad, "0f3FB8AA3B", "0f3FB8AA3C"),
        (GeluSrcad, "0f3FD9DB23", "0f3FD9DB24"),
        (GeluSrcad, "0f3FD9DB23", "0f3FD9999A"),
        (GeluSrcad, "0f3F800000", "0f3F800001"),
        (GeluSrcad, "0f3FB8AA3B", "0f3FB8AA3C"),
        (SwigluGate, "0f3F800000", "0f3F800001"),
        (SwigluGate, "0f3FB8AA3B", "0f3FB8AA3C"),
    ] {
        let count = kir_ptx(op).matches(from).count();
        assert!(count >= 1, "{op:?}: {from}");
        for i in 0..count {
            assert!(caught(op, |p| nudge(p, from, to, i)), "{op:?}: {from} #{i} -> {to} went unnoticed");
        }
    }
}

/// Each multiply turned into an add (and each add or subtract into a
/// multiply), one at a time.
#[test]
fn a_neighbours_operation_is_caught() {
    for op in BackwardOp::ALL {
        let k = kir_ptx(op);
        for (from, to) in [
            ("mul.f32 ", "add.f32 "),
            ("add.f32 ", "mul.f32 "),
            ("sub.f32 ", "mul.f32 "),
            ("mul.rn.f32 ", "add.rn.f32 "),
            ("add.rn.f32 ", "mul.rn.f32 "),
            ("sub.rn.f32 ", "mul.rn.f32 "),
        ] {
            for i in 0..k.matches(from).count() {
                assert!(caught(op, |p| nudge(p, from, to, i)), "{op:?}: {from} #{i} -> {to}");
            }
        }
    }
}

/// Relu passes the gradient only for a strictly positive input: `>=` would
/// pass it at ±0 as well.
#[test]
fn relus_strict_comparison_is_pinned() {
    assert!(caught(BackwardOp::Relu, |p| p.replacen("setp.gt.f32 ", "setp.ge.f32 ", 1)));
}

/// The SwiGLU gate multiplies by `up` before the derivative: reading `input`
/// in its place is caught.
#[test]
fn the_swiglu_gate_reads_up() {
    assert!(caught(BackwardOp::SwigluGate, |p| p.replacen("[param_up]", "[param_input]", 1)));
}
