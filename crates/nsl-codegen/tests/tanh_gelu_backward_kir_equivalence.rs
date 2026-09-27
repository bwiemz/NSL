//! The differential gate for `nsl_tanh_f32` and `nsl_gelu_backward_f32`
//! (roadmap A2 step 11). The two kernels moved to KIR, and their saturation
//! was fixed on the way.
//!
//! The runtime carried both as hand-written PTX. They are now built by
//! `nsl_kir::kernels::elementwise::{build_tanh, build_gelu_backward}`. Both
//! compute `tanh(v)` as `(e − 1) / (e + 1)`, `e = 2^(2v·log2 e)`, with
//! `div.approx.f32`. That instruction returns 0 for a divisor whose magnitude
//! is in `(2^126, 2^128)`, and the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`) models that range. The hand kernels
//! fell into it:
//!
//! * `nsl_tanh_f32` clamped its input at 44, where the divisor is about
//!   `2^126.96`. It returned 0 for every `x` from about 43.67 up, `+∞` and
//!   NaN included.
//! * `nsl_gelu_backward_f32` did not clamp. It returned garbage and then NaN
//!   once `k = 0.0356774·x³ + 0.797885·x` passed the same point (`x` about
//!   10).
//!
//! The KIR kernels keep the hand arithmetic and saturate at
//! `TANH_SATURATION` (43.5). This file runs the frozen hand modules
//! (`tests/fixtures/tanh_gelu_backward_hand.rs`) and the KIR ones side by
//! side, over the grid the runtime launches (`ceil(n / 256)` blocks of 256)
//! plus one block more:
//!
//! 1. **The predecessor's failure is real**: on the interpreter, the hand
//!    tanh returns 0 at 44, `+∞` and NaN, and the hand adjoint returns a
//!    non-finite or wrong value past `x ≈ 10`.
//! 2. **Agreement where the predecessor was right**: for `|x| ≤ 43.5` (tanh)
//!    and `|k(x)| ≤ 43.5` (the adjoint), under two schedules, the hand and
//!    KIR kernels leave *the same bytes* in all of global memory.
//! 3. **Saturation**: past the point, the KIR tanh is ±1, a NaN input comes
//!    back as it is, and the adjoint is `grad·1` above and `grad·0` below.
//! 4. **Correctness**: the KIR kernels track the f64 formulas over the whole
//!    range, and nothing past `n` is written.
//! 5. **The gate bites**: the bound, the block index, the element sizes, the
//!    old clamp, each select, the quotient's operands and the constants are
//!    each caught. `div.approx` → `div.rn` is a named equivalent mutant: the
//!    saturation keeps the divisor out of the flush range.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::{
    gelu_backward_ptx, tanh_ptx, ELEMENTWISE_BLOCK, GELU_BACKWARD_NAME, TANH_NAME, TANH_SATURATION,
};

#[allow(dead_code)]
#[path = "fixtures/tanh_gelu_backward_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const IN0: u64 = 0x1000_0000;
const IN1: u64 = 0x2000_0000;
const OUT: u64 = 0x3000_0000;
const TAIL: usize = 300;
const POISON: u32 = 0x7FA5_A5A5;
/// What the inputs hold past `n`: 0.5, inside every domain, so a thread
/// past the bound writes something other than the poison.
const IN_TAIL: u32 = 0x3F00_0000;

fn sat() -> f32 {
    f32::from_bits(TANH_SATURATION)
}

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kernel {
    Tanh,
    GeluBackward,
}

fn kir_ptx(k: Kernel) -> String {
    trim(&String::from_utf8(match k {
        Kernel::Tanh => tanh_ptx(),
        Kernel::GeluBackward => gelu_backward_ptx(),
    })
    .expect("ASCII"))
}

fn hand_ptx(k: Kernel) -> String {
    trim(match k {
        Kernel::Tanh => hand::TANH_F32_PTX,
        Kernel::GeluBackward => hand::GELU_BACKWARD_F32_PTX,
    })
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

/// Run `ptx` over `ceil(n / 256) + 1` blocks. Tanh reads `x` and ignores
/// `g`, and its memory is `[a, c]`. The adjoint's is `[grad, input, out]`.
/// Returns all of global memory.
fn run(k: Kernel, ptx: &str, g: &[u32], x: &[u32], order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let n = x.len();
    let (mut global, args): (Vec<Segment>, Vec<(&str, u64)>) = match k {
        Kernel::Tanh => (
            vec![
                Segment { base: IN0, bytes: with_tail(x) },
                Segment { base: OUT, bytes: le32(&vec![POISON; n + TAIL]) },
            ],
            vec![("a", IN0), ("c", OUT), ("n", n as u64)],
        ),
        Kernel::GeluBackward => {
            assert_eq!(g.len(), n);
            (
                vec![
                    Segment { base: IN0, bytes: with_tail(g) },
                    Segment { base: IN1, bytes: with_tail(x) },
                    Segment { base: OUT, bytes: le32(&vec![POISON; n + TAIL]) },
                ],
                vec![("grad", IN0), ("input", IN1), ("out", OUT), ("n", n as u64)],
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

/// The output words of a run.
fn out(k: Kernel, ptx: &str, g: &[u32], x: &[u32]) -> Vec<u32> {
    let mem = run(k, ptx, g, x, Order::Ascending);
    words(mem.last().expect("an output segment"))
}

fn lcg(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    (*s >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0
}

/// `n` values: the corners first (rotated by `seed`), then uniform draws in
/// `[-bound, bound]` at a spread of scales.
fn values(n: usize, seed: u64, corners: &[u32], bound: f32) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|k| {
            if k < corners.len() {
                return corners[(k + seed as usize) % corners.len()];
            }
            let scale = [1e-4f32, 0.1, 1.0, bound / 4.0, bound][(k / 7) % 5];
            (lcg(&mut s) * scale).clamp(-bound, bound).to_bits()
        })
        .collect()
}

/// `k(x)` in the adjoint's operation order, uncontracted, as the
/// interpreter runs it.
fn k_of(x: f32) -> f32 {
    x * x * x * f32::from_bits(0x3D12_4925) + x * f32::from_bits(0x3F4C_422A)
}

/// Where the adjoint's hand arithmetic is valid: `|k| ≤ 43.5` at every
/// `|x| ≤ 9.95`.
const GELU_BOUND: f32 = 9.95;

const TANH_CORNERS: [u32; 16] = [
    0x0000_0000, // +0
    0x8000_0000, // -0
    0x0000_0001, // the least subnormal
    0x807F_FFFF,
    0x3F80_0000, // 1
    0xBF80_0000,
    0x3F00_0000, // 0.5
    0x3586_37BD, // 1e-6
    0xB586_37BD,
    0x4120_0000, // 10
    0x41A0_0000, // 20
    0xC1A0_0000,
    0x422E_0000, // 43.5, the saturation point
    0xC22E_0000,
    0x422D_FFFF, // just inside it
    0xC22D_FFFF,
];

const GELU_CORNERS: [u32; 12] = [
    0x0000_0000,
    0x8000_0000,
    0x0000_0001,
    0x807F_FFFF,
    0x3F80_0000,
    0xBF80_0000,
    0x3F00_0000,
    0x4040_0000, // 3
    0xC040_0000,
    0x411F_3333, // 9.95
    0xC11F_3333,
    0x3586_37BD,
];

fn grads(n: usize, seed: u64) -> Vec<u32> {
    const G: [u32; 6] = [0x3F80_0000, 0xC020_0000, 0x0000_0000, 0x8000_0000, 0x7FC0_0001, 0x7F80_0000];
    values(n, seed, &G, 3.0)
}

/// Past the saturation point on both sides, the infinities and NaN.
fn beyond(k: Kernel) -> Vec<f32> {
    let mut v = match k {
        Kernel::Tanh => vec![43.51, 43.67, 44.0, 50.0, 1e3, 1e30],
        Kernel::GeluBackward => vec![10.0, 10.1, 10.5, 12.0, 20.0, 100.0, 1e6, 1e19, 1e30],
    };
    v.extend(v.clone().into_iter().map(|x| -x));
    v.extend([f32::MAX, -f32::MAX, f32::INFINITY, f32::NEG_INFINITY]);
    v
}

const SIZES: [usize; 4] = [1, 255, 257, 1000];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

// ---------------------------------------------------------------------------
// 1. The predecessor's failure
// ---------------------------------------------------------------------------

#[test]
fn the_hand_kernels_fall_into_the_div_approx_flush_range() {
    let hand = hand_ptx(Kernel::Tanh);
    let xs = [43.5f32, 44.0, 1e3, f32::INFINITY, f32::NAN, -44.0];
    let y: Vec<f32> = out(Kernel::Tanh, &hand, &[], &xs.map(f32::to_bits)).into_iter().map(f32::from_bits).collect();
    assert_eq!(y[0], 1.0, "inside the range it saturates correctly");
    assert_eq!(&y[1..5], &[0.0; 4], "tanh(44), tanh(1e3), tanh(+inf) and tanh(NaN) were 0");
    assert_eq!(y[5], -1.0, "the negative side never reached the range");

    let hand = hand_ptx(Kernel::GeluBackward);
    let xs = [9.95f32, 10.0, 10.5, 20.0];
    let g = [1.0f32; 4].map(f32::to_bits);
    let y: Vec<f32> = out(Kernel::GeluBackward, &hand, &g, &xs.map(f32::to_bits)).into_iter().map(f32::from_bits).collect();
    assert!((y[0] - 1.0).abs() < 1e-5, "at 9.95 the hand adjoint is right: {}", y[0]);
    for (x, y) in xs[1..].iter().zip(&y[1..]) {
        assert!(!(y.is_finite() && (y - 1.0).abs() < 0.5), "x={x}: the hand adjoint gave {y}, not ~1");
    }
}

// ---------------------------------------------------------------------------
// 2. Agreement where the predecessor was right
// ---------------------------------------------------------------------------

#[test]
fn tanh_agrees_with_the_hand_kernel_bit_for_bit_inside_the_saturation_point() {
    for (j, &n) in SIZES.iter().enumerate() {
        let x = values(n, j as u64 + 1, &TANH_CORNERS, sat());
        assert!(x.iter().all(|&w| f32::from_bits(w).abs() <= sat()));
        for order in ORDERS {
            let hand = run(Kernel::Tanh, &hand_ptx(Kernel::Tanh), &[], &x, order);
            let kir = run(Kernel::Tanh, &kir_ptx(Kernel::Tanh), &[], &x, order);
            assert!(hand == kir, "n={n} {order:?}: global memory differs");
        }
    }
}

#[test]
fn the_adjoint_agrees_with_the_hand_kernel_bit_for_bit_inside_the_saturation_point() {
    for (j, &n) in SIZES.iter().enumerate() {
        let x = values(n, j as u64 + 11, &GELU_CORNERS, GELU_BOUND);
        assert!(x.iter().all(|&w| k_of(f32::from_bits(w)).abs() <= sat()));
        let g = grads(n, j as u64 + 50);
        for order in ORDERS {
            let hand = run(Kernel::GeluBackward, &hand_ptx(Kernel::GeluBackward), &g, &x, order);
            let kir = run(Kernel::GeluBackward, &kir_ptx(Kernel::GeluBackward), &g, &x, order);
            assert!(hand == kir, "n={n} {order:?}: global memory differs");
        }
    }
}

// ---------------------------------------------------------------------------
// 3. Saturation
// ---------------------------------------------------------------------------

#[test]
fn tanh_saturates_to_one_and_returns_a_nan_as_it_is() {
    let kir = kir_ptx(Kernel::Tanh);
    let at_sat = f32::from_bits(out(Kernel::Tanh, &kir, &[], &[TANH_SATURATION])[0]);
    assert_eq!(at_sat, 1.0);
    let xs = beyond(Kernel::Tanh);
    let y = out(Kernel::Tanh, &kir, &[], &xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
    for (x, y) in xs.iter().zip(&y) {
        assert_eq!(f32::from_bits(*y), x.signum(), "tanh({x:e})");
    }
    for nan in [0x7FC0_0000u32, 0x7FC0_0001, 0xFFC0_0001, 0x7F80_0001] {
        assert_eq!(out(Kernel::Tanh, &kir, &[], &[nan])[0], nan, "NaN {nan:#x} comes back as it is");
    }
}

#[test]
fn the_adjoint_saturates_to_its_limits() {
    let kir = kir_ptx(Kernel::GeluBackward);
    let xs = beyond(Kernel::GeluBackward);
    for g in [1.0f32, -2.5, 0.0, -0.0, 3e-39] {
        let gs = vec![g.to_bits(); xs.len()];
        let y = out(Kernel::GeluBackward, &kir, &gs, &xs.iter().map(|x| x.to_bits()).collect::<Vec<_>>());
        for (x, y) in xs.iter().zip(&y) {
            let want = if *x > 0.0 { g } else { g * 0.0 };
            assert_eq!(*y, want.to_bits(), "g={g:e} x={x:e}: got {:e}", f32::from_bits(*y));
        }
    }
    // A NaN input or gradient is NaN out.
    let y = out(Kernel::GeluBackward, &kir, &[1.0f32.to_bits(), f32::NAN.to_bits()], &[f32::NAN.to_bits(), 50.0f32.to_bits()]);
    assert!(y.iter().all(|&w| f32::from_bits(w).is_nan()));
}

// ---------------------------------------------------------------------------
// 4. Correctness
// ---------------------------------------------------------------------------

/// The whole range: the domain's draws, the saturation corners, the far
/// side. Absolute error against f64 `tanh`: the interpreter's `ex2` is
/// exact, so what is left is f32 rounding in `(e − 1) / (e + 1)`,
/// cancellation near 0 included.
#[test]
fn tanh_tracks_the_f64_function_everywhere() {
    let mut xs: Vec<u32> = values(1000, 5, &TANH_CORNERS, 60.0);
    xs.extend(beyond(Kernel::Tanh).iter().map(|x| x.to_bits()));
    let mem = run(Kernel::Tanh, &kir_ptx(Kernel::Tanh), &[], &xs, Order::Ascending);
    let y = words(&mem[1]);
    for (x, y) in xs.iter().zip(&y) {
        let (x, y) = (f32::from_bits(*x) as f64, f32::from_bits(*y) as f64);
        assert!((y - x.tanh()).abs() <= 2e-7, "tanh({x:e}) = {y:e}, want {:e}", x.tanh());
    }
    assert!(y[xs.len()..].iter().all(|&w| w == POISON), "wrote past n");
    assert_eq!(words(&mem[0])[..xs.len()], xs[..], "the input is untouched");
}

/// The f64 tanh-approximation GELU derivative, in the kernel's form.
fn gelu_tanh_deriv(x: f64) -> f64 {
    let c = |w: u32| f32::from_bits(w) as f64;
    let k = c(0x3D12_4925) * x * x * x + c(0x3F4C_422A) * x;
    let t = k.tanh();
    0.5 * (1.0 + t + x * (1.0 - t * t) * (c(0x3DD8_ECA1) * x * x + c(0x3F4C_422A)))
}

#[test]
fn the_adjoint_tracks_the_f64_derivative_everywhere() {
    let mut xs: Vec<u32> = values(1000, 9, &GELU_CORNERS, 15.0);
    xs.extend(beyond(Kernel::GeluBackward).iter().map(|x| x.to_bits()));
    let g = grads(xs.len(), 90);
    let mem = run(Kernel::GeluBackward, &kir_ptx(Kernel::GeluBackward), &g, &xs, Order::Ascending);
    let y = words(&mem[2]);
    for i in 0..xs.len() {
        let (x, gv) = (f32::from_bits(xs[i]) as f64, f32::from_bits(g[i]) as f64);
        let got = f32::from_bits(y[i]) as f64;
        let want = if x.is_infinite() { if x > 0.0 { gv } else { gv * 0.0 } } else { gv * gelu_tanh_deriv(x) };
        if want.is_nan() {
            assert!(got.is_nan(), "i={i} x={x:e} g={gv:e}: got {got:e}");
            continue;
        }
        if want.is_infinite() {
            assert_eq!(got, want, "i={i} x={x:e} g={gv:e}");
            continue;
        }
        assert!((got - want).abs() <= 2e-6 * (1.0 + want.abs()), "i={i} x={x:e} g={gv:e}: got {got:e}, want {want:e}");
    }
    assert!(y[xs.len()..].iter().all(|&w| w == POISON), "wrote past n");
    assert_eq!(words(&mem[0])[..xs.len()], g[..], "grad is untouched");
    assert_eq!(words(&mem[1])[..xs.len()], xs[..], "input is untouched");
}

// ---------------------------------------------------------------------------
// Signature and spelling
// ---------------------------------------------------------------------------

#[test]
fn the_kir_kernels_keep_the_ffi_signatures_and_entry_names() {
    for (k, name, params) in [
        (Kernel::Tanh, TANH_NAME, &["a", "c", "n"][..]),
        (Kernel::GeluBackward, GELU_BACKWARD_NAME, &["grad", "input", "out", "n"][..]),
    ] {
        let (h, kir) = (hand_ptx(k), kir_ptx(k));
        assert_eq!(parse_signature(&kir), parse_signature(&h), "{k:?}");
        let names: Vec<String> = parse_signature(&h).into_iter().map(|(_, n)| n).collect();
        assert_eq!(names, params, "{k:?}");
        for text in [&h, &kir] {
            assert!(text.contains(&format!(".visible .entry {name}(")), "{k:?}");
        }
    }
}

/// The hand kernels' approximate forms, once each, and their arithmetic
/// bare (ptxas contracts the adjoint's multiply-adds as it did). The new
/// instructions are the saturation's alone.
#[test]
fn the_arithmetic_is_spelled_as_in_the_hand_kernels() {
    for k in [Kernel::Tanh, Kernel::GeluBackward] {
        let (h, kir) = (hand_ptx(k), kir_ptx(k));
        for form in ["div.approx.f32 ", "ex2.approx.f32 ", "mul.f32 ", "add.f32 ", "sub.f32 "] {
            assert_eq!(kir.matches(form).count(), h.matches(form).count(), "{k:?} {form}");
        }
        for never in [".rn.", "fma.", "rcp.", "div.rn", "div.full"] {
            assert!(!kir.contains(never), "{k:?} {never}");
        }
    }
    let t = kir_ptx(Kernel::Tanh);
    assert!(t.contains("0f422E0000") && t.contains("0fC22E0000") && !t.contains("0f42300000"), "the clamp moved to 43.5");
    assert_eq!((t.matches("min.f32 ").count(), t.matches("max.f32 ").count()), (1, 1));
    assert_eq!((t.matches("setp.eq.f32 ").count(), t.matches("selp.f32 ").count()), (1, 1));
    let g = kir_ptx(Kernel::GeluBackward);
    assert_eq!((g.matches("setp.gt.f32 ").count(), g.matches("setp.lt.f32 ").count(), g.matches("selp.f32 ").count()), (1, 1, 2));
    assert!(!g.contains("min.f32") && !g.contains("max.f32"), "the adjoint selects, it does not clamp");
    for c in ["0f3D124925", "0f3F4C422A", "0f3DD8ECA1", "0f3F000000", "0f3FB8AA3B"] {
        assert!(g.contains(c) && hand_ptx(Kernel::GeluBackward).contains(c), "{c}");
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the KIR kernel, under either
/// schedule (or faults), on a ragged size over the domain and past it.
fn caught(k: Kernel, mutate: impl Fn(&str) -> String) -> bool {
    let (corners, bound): (&[u32], f32) = match k {
        Kernel::Tanh => (&TANH_CORNERS, sat()),
        Kernel::GeluBackward => (&GELU_CORNERS, GELU_BOUND),
    };
    let mut x = values(257, 3, corners, bound);
    x.extend(beyond(k).iter().map(|v| v.to_bits()));
    x.extend([0x7FC0_0001, 0xFFC0_0001]);
    let g = grads(x.len(), 30);
    let kir = kir_ptx(k);
    let mutant = mutate(&kir);
    ORDERS.into_iter().any(|order| {
        let expect = run(k, &kir, &g, &x, order);
        let (g, x) = (&g, &x);
        match std::panic::catch_unwind(|| run(k, &mutant, g, x, order)) {
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

/// The `i`th instruction `mnemonic d, a, b, …;` rewritten by `f` over its
/// operand list.
fn rewrite(ptx: &str, mnemonic: &str, i: usize, f: impl Fn(&str, &[&str]) -> String) -> String {
    let line = ptx.lines().filter(|l| l.contains(mnemonic)).nth(i).expect("the instruction").to_string();
    let (head, args) = line.split_once(mnemonic).expect("mnemonic");
    let ops: Vec<&str> = args.trim_end_matches(';').split(',').map(str::trim).collect();
    ptx.replacen(&line, &format!("{head}{}", f(mnemonic, &ops)), 1)
}

/// `selp.f32 d, a, b, p` → `mov.f32 d, arm`: the select dropped, one arm kept.
fn keep_arm(ptx: &str, i: usize, arm: usize) -> String {
    rewrite(ptx, "selp.f32 ", i, |_, ops| format!("mov.f32 {}, {};", ops[0], ops[arm]))
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    assert!(!caught(Kernel::Tanh, |p| p.to_string()));
    assert!(!caught(Kernel::GeluBackward, |p| p.to_string()));
}

#[test]
fn relaxing_the_bound_or_ignoring_the_block_index_is_caught() {
    for k in [Kernel::Tanh, Kernel::GeluBackward] {
        let p = kir_ptx(k);
        assert_eq!((p.matches("setp.ge.u64 ").count(), p.matches("%ctaid.x;").count()), (1, 1), "{k:?}");
        assert!(caught(k, |p| p.replacen("setp.ge.u64 ", "setp.gt.u64 ", 1)), "{k:?} bound");
        assert!(caught(k, |p| p.replacen("%ctaid.x;", "0;", 1)), "{k:?} block index");
    }
}

#[test]
fn nudging_an_element_size_is_caught() {
    for (k, addresses) in [(Kernel::Tanh, 2), (Kernel::GeluBackward, 3)] {
        assert_eq!(kir_ptx(k).matches(", 4;").count(), addresses, "{k:?}");
        for i in 0..addresses {
            assert!(caught(k, |p| nudge(p, ", 4;", ", 8;", i)), "{k:?} address {i}");
        }
    }
}

/// Back at the hand kernel's 44, `tanh` is 0 from about 43.67 up.
#[test]
fn the_old_clamp_is_caught() {
    assert!(caught(Kernel::Tanh, |p| p.replacen("0f422E0000", "0f42300000", 1)));
}

#[test]
fn dropping_either_arm_of_a_select_is_caught() {
    // tanh: the NaN select, either way.
    assert!(caught(Kernel::Tanh, |p| keep_arm(p, 0, 1)), "tanh: NaN no longer returned");
    assert!(caught(Kernel::Tanh, |p| keep_arm(p, 0, 2)), "tanh: the input returned");
    // The adjoint: the upper select, then the lower.
    for (i, what) in [(0, "upper"), (1, "lower")] {
        assert!(caught(Kernel::GeluBackward, |p| keep_arm(p, i, 1)), "{what}: limit always");
        assert!(caught(Kernel::GeluBackward, |p| keep_arm(p, i, 2)), "{what}: formula always");
    }
    assert!(caught(Kernel::GeluBackward, |p| p.replacen("setp.gt.f32 ", "setp.lt.f32 ", 1)), "upper test flipped");
}

#[test]
fn swapping_the_quotient_or_nudging_a_constant_is_caught() {
    for k in [Kernel::Tanh, Kernel::GeluBackward] {
        let swapped = |p: &str| rewrite(p, "div.approx.f32 ", 0, |m, o| format!("{m}{}, {}, {};", o[0], o[2], o[1]));
        assert!(caught(k, swapped), "{k:?} quotient swapped");
        assert!(caught(k, |p| p.replacen("0f3FB8AA3B", "0f3FB8AA3C", 1)), "{k:?} log2(e)");
    }
    for c in ["0f3D124925", "0f3F4C422A", "0f3DD8ECA1", "0f3F000000"] {
        // Bit 12 of the word: a relative change of 2^-11. A one-ulp nudge
        // of the cubic term's constant moves no output at these magnitudes.
        let bumped = format!("{}{:X}{}", &c[..6], u8::from_str_radix(&c[6..7], 16).unwrap() ^ 1, &c[7..]);
        assert!(caught(Kernel::GeluBackward, |p| p.replacen(c, &bumped, 1)), "{c}");
    }
}

/// With the saturation, no divisor reaches `div.approx.f32`'s flush range,
/// so the IEEE quotient is indistinguishable here. In the hand kernels it
/// is not.
#[test]
fn div_rn_in_place_of_div_approx_is_a_named_equivalent_mutant() {
    for k in [Kernel::Tanh, Kernel::GeluBackward] {
        assert!(!caught(k, |p| p.replacen("div.approx.f32 ", "div.rn.f32 ", 1)), "{k:?}");
    }
    let hand = hand_ptx(Kernel::Tanh).replacen("div.approx.f32 ", "div.rn.f32 ", 1);
    let y = out(Kernel::Tanh, &hand, &[], &[44.0f32.to_bits()]);
    assert_eq!(f32::from_bits(y[0]), 1.0, "the IEEE quotient would have hidden the hand kernel's failure");
}
