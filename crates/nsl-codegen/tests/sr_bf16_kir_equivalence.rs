//! The differential equivalence gate for the SR-BF16 kernels (new-roadmap
//! item 5): `nsl_sr_bf16_round_probe` and the single-parameter FASE AdamW
//! step with a bf16 θ, `nsl_fase_fused_adamw_step_bf16sr`.
//!
//! The runtime carried both as hand-written PTX. They are now built by
//! `nsl_kir::kernels::optim`, from one stochastic-rounding tail
//! (`sr_bf16_bits`) and the f32 step's AdamW body. KIR needed two things
//! for them: a `U16` type (bf16 bits, `ld`/`st.global.u16`,
//! `cvt.u32.u16`) and `KirOp::Bitcast` (`mov.b32` between f32 and its
//! bits). This file runs the frozen hand modules
//! (`tests/fixtures/sr_bf16_hand.rs`) and the KIR ones side by side on the
//! cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`), over
//! the grid the runtime launches (`ceil(n / 256)` blocks of 256) plus one
//! block more:
//!
//! 1. **Agreement**: under two schedules and four `(key, counter base)`
//!    pairs, the two probes leave *the same bytes* in all of global memory.
//!    The pairs include a counter base that wraps `u64`. Every f32 bit
//!    pattern class goes in: signed zeros, subnormals, max-normal (whose
//!    dither can carry into the exponent), ±∞ and quiet, signalling and
//!    negative NaNs. The two steps agree likewise, across the f32 step's
//!    hyperparameter sets.
//! 2. **Correctness**: the probe is the host reference
//!    (`sr_bf16::sr_bf16_round` of `sr_mix64(key, ctr_base + i)`,
//!    restated here) bit for bit. The step's θ is that rounding of the f32
//!    step, and its m and v are the f32 step's. Nothing past `n` moves.
//! 3. **The gate bites**: the bound, the block index, every element size,
//!    each splitmix64 constant and shift, the dither mask, the counter, the
//!    sign, exponent and mantissa masks, each special value, the widening
//!    shift, the truncating shift and each branch are caught.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;
use nsl_kir::kernels::optim::{
    fase_adamw_step_bf16sr_ptx, sr_bf16_round_probe_ptx, FASE_ADAMW_STEP_BF16SR_NAME, SR_BF16_ROUND_PROBE_NAME,
    SR_SPLITMIX_GAMMA,
};

#[allow(dead_code)]
#[path = "fixtures/sr_bf16_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const SRC: u64 = 0x1000_0000;
const DST: u64 = 0x2000_0000;
const THETA: u64 = 0x1000_0000;
const M: u64 = 0x2000_0000;
const V: u64 = 0x3000_0000;
const MP: u64 = 0x4000_0000;
const TAIL: usize = 300;
/// What each buffer holds past `n`. An f32 input tail of 1.5 and a bf16
/// θ tail of 1.5 (`0x3FC0`) are finite, so a thread past the bound would
/// change what it wrote. The u16 output poison is a pattern the tail never
/// produces.
const IN_TAIL: u32 = 0x3FC0_0000;
const THETA_TAIL: u16 = 0x3FC0;
const POISON16: u16 = 0xA5A5;

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kernel {
    Probe,
    Step,
}

fn kir_ptx(k: Kernel) -> String {
    trim(&String::from_utf8(match k {
        Kernel::Probe => sr_bf16_round_probe_ptx(),
        Kernel::Step => fase_adamw_step_bf16sr_ptx(),
    })
    .expect("ASCII"))
}

fn hand_ptx(k: Kernel) -> String {
    trim(match k {
        Kernel::Probe => hand::SR_BF16_ROUND_PROBE_PTX,
        Kernel::Step => hand::FASE_FUSED_ADAMW_STEP_BF16SR_PTX,
    })
}

// ---------------------------------------------------------------------------
// The host reference, restated (`nsl_runtime::sr_bf16`)
// ---------------------------------------------------------------------------

/// `sr_bf16::sr_mix64`: splitmix64 of `key + counter·γ`.
fn mix64(key: u64, counter: u64) -> u64 {
    let mut z = key.wrapping_add(counter.wrapping_mul(SR_SPLITMIX_GAMMA));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// `sr_bf16::sr_bf16_round`.
fn round(bits: u32, dither: u16) -> u16 {
    let sign = ((bits >> 16) & 0x8000) as u16;
    if bits & 0x7F80_0000 == 0x7F80_0000 {
        return if bits & 0x007F_FFFF != 0 { sign | 0x7FC0 } else { sign | 0x7F80 };
    }
    let dithered = bits.wrapping_add(dither as u32);
    if dithered & 0x7F80_0000 == 0x7F80_0000 {
        return sign | 0x7F7F;
    }
    (dithered >> 16) as u16
}

/// `sr_bf16::sr_step_key`.
fn step_key(seed: u64, step: u64) -> u64 {
    seed ^ step.wrapping_mul(SR_SPLITMIX_GAMMA)
}

#[test]
fn the_restated_reference_matches_the_hosts_constants() {
    assert_eq!(SR_SPLITMIX_GAMMA, 0x9E37_79B9_7F4A_7C15, "sr_bf16::SR_STEP_SALT");
    // Max-normal with a full dither carries into the exponent: saturates.
    assert_eq!(round(0x7F7F_FFFF, 0xFFFF), 0x7F7F);
    assert_eq!(round(0xFF7F_FFFF, 0x0001), 0xFF7F);
    assert_eq!(round(0x7FC0_0001, 0), 0x7FC0);
    assert_eq!(round(0xFF80_0000, 0x1234), 0xFF80);
    assert_eq!(round(0x3F80_8000, 0x7FFF), 0x3F80, "below the half: truncates");
    assert_eq!(round(0x3F80_8000, 0x8000), 0x3F81, "at the half: carries");
}

// ---------------------------------------------------------------------------
// Inputs and launches
// ---------------------------------------------------------------------------

/// The `(sr_key, sr_ctr_base)` pairs: zero, two real step keys at real
/// parameter bases (`param << 40`), and a base that wraps `u64` inside the
/// launch.
fn keys() -> [(u64, u64); 4] {
    [(0, 0), (step_key(42, 1), 0), (step_key(7, 1000), (3 << 40) + 5), (u64::MAX, u64::MAX - 100)]
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    *s
}

/// f32 bit patterns: every class first, then uniform random words.
fn f32_bits(n: usize, seed: u64) -> Vec<u32> {
    const CORNERS: [u32; 18] = [
        0x0000_0000, // +0
        0x8000_0000, // -0
        0x0000_0001, // least subnormal
        0x007F_FFFF, // greatest subnormal
        0x807F_0000,
        0x0080_0000, // least normal
        0x7F7F_FFFF, // max normal: the dither can carry
        0xFF7F_FFFF,
        0x7F7F_0001,
        0x7F80_0000, // +inf
        0xFF80_0000, // -inf
        0x7FC0_0001, // quiet NaN
        0x7F80_0001, // signalling NaN
        0xFFC0_0000, // negative NaN
        0x3F80_0000, // 1
        0x3F80_8000, // 1 + half a bf16 ulp
        0xBF80_7FFF,
        0x4049_0FDB, // pi
    ];
    let mut s = seed;
    (0..n)
        .map(|k| if k < CORNERS.len() { CORNERS[(k + seed as usize) % CORNERS.len()] } else { (lcg(&mut s) >> 32) as u32 })
        .collect()
}

fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn halves(b: &[u8]) -> Vec<u16> {
    b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn launch(ptx: &str, mut global: Vec<Segment>, args: HashMap<String, u64>, n: usize, order: Order) -> Vec<Vec<u8>> {
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
            nctaid_y: 1,
            ntid: ELEMENTWISE_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    global.into_iter().map(|s| s.bytes).collect()
}

fn named(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
    pairs.iter().map(|(k, v)| (k.to_string(), *v)).collect()
}

/// The probe over `src`. Returns `[src, dst]`.
fn run_probe(ptx: &str, src: &[u32], (key, ctr): (u64, u64), order: Order) -> Vec<Vec<u8>> {
    let n = src.len();
    let mut s = src.to_vec();
    s.extend(vec![IN_TAIL; TAIL]);
    let global = vec![Segment { base: SRC, bytes: le32(&s) }, Segment { base: DST, bytes: le16(&vec![POISON16; n + TAIL]) }];
    let args = named(&[("src", SRC), ("dst", DST), ("n", n as u64), ("sr_key", key), ("sr_ctr_base", ctr)]);
    launch(ptx, global, args, n, order)
}

/// The nine `.f32` hyperparameters, in parameter order, and the flag.
#[derive(Debug, Clone, Copy)]
struct Hyper {
    b1: f32,
    omb1: f32,
    b2: f32,
    omb2: f32,
    eps: f32,
    neg_lr: f32,
    neg_lr_wd: f32,
    bc1: f32,
    bc2: f32,
    has_wd: u32,
}

/// The f32 step gate's sets: AdamW steps 1 and 1000 with and without
/// decay, a large-epsilon Adam step, and one with every scalar distinct.
fn hypers() -> Vec<Hyper> {
    let adamw = |t: i32, has_wd: u32| {
        let (b1, b2) = (0.9f32, 0.999f32);
        Hyper {
            b1,
            omb1: 1.0 - b1,
            b2,
            omb2: 1.0 - b2,
            eps: 1e-8,
            neg_lr: -1e-3,
            neg_lr_wd: -1e-3 * 0.01,
            bc1: 1.0 / (1.0 - b1.powi(t)),
            bc2: 1.0 / (1.0 - b2.powi(t)),
            has_wd,
        }
    };
    vec![
        adamw(1, 0),
        adamw(1, 1),
        adamw(1000, 1),
        Hyper { eps: 1.0, ..adamw(3, 0) },
        Hyper { b1: 0.7, omb1: 0.2, b2: 0.95, omb2: 0.03, eps: 1e-3, neg_lr: -0.25, neg_lr_wd: -0.125, bc1: 1.75, bc2: 3.5, has_wd: 7 },
    ]
}

struct Case {
    theta: Vec<u16>,
    m: Vec<u32>,
    v: Vec<u32>,
    mp: Vec<u32>,
}

/// bf16 θ (every class, then random bits), and f32 moments and gradients
/// from ordinary magnitudes through the corners.
fn case(n: usize, seed: u64) -> Case {
    const THETA: [u16; 10] = [0x0000, 0x8000, 0x0001, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC1, 0x3F80, 0xBE4C];
    let mut s = seed;
    let theta = (0..n).map(|k| if k < THETA.len() { THETA[(k + seed as usize) % THETA.len()] } else { (lcg(&mut s) >> 48) as u16 }).collect();
    let ordinary = |s: &mut u64, nonneg: bool| {
        let x = ((lcg(s) >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0;
        let scale = [1e-30f32, 1e-3, 1.0, 1e3, 1e30][((*s >> 20) % 5) as usize];
        let bits = (x * scale).to_bits();
        if nonneg { bits & 0x7FFF_FFFF } else { bits }
    };
    let mix = |s: &mut u64, k: usize, nonneg: bool| {
        let corners = f32_bits(18, 0);
        let bits = if k % 7 == 3 { corners[k % corners.len()] } else { ordinary(s, nonneg) };
        if nonneg { bits & 0x7FFF_FFFF } else { bits }
    };
    let m = (0..n).map(|k| mix(&mut s, k, false)).collect();
    let v = (0..n).map(|k| mix(&mut s, k + 1, true)).collect();
    let mp = (0..n).map(|k| mix(&mut s, k + 2, false)).collect();
    Case { theta, m, v, mp }
}

/// `c` plus 64 elements whose update pushes θ just past bf16's max-normal
/// under the large-epsilon set (`hypers()[3]`). θ is ±`0x7F7F`, m is ∓1e38
/// and v and the gradient are 0, so `adj` is about 1e35, some 16000 f32 ulps
/// outward. A dither above about `0xC180` then carries into the all-ones
/// exponent, and the tail saturates. Random inputs almost never reach it.
fn with_saturating(mut c: Case) -> Case {
    for k in 0..64 {
        let neg = k % 2 == 1;
        c.theta.push(if neg { 0xFF7F } else { 0x7F7F });
        c.m.push((if neg { 1e38f32 } else { -1e38f32 }).to_bits());
        c.v.push(0);
        c.mp.push(0);
    }
    c
}

/// The step over `c`. Returns `[θ, m, v, mp]`.
fn run_step(ptx: &str, c: &Case, h: &Hyper, (key, ctr): (u64, u64), order: Order) -> Vec<Vec<u8>> {
    let n = c.theta.len();
    let tail32 = |v: &[u32]| {
        let mut w = v.to_vec();
        w.extend(vec![IN_TAIL; TAIL]);
        le32(&w)
    };
    let mut th = c.theta.clone();
    th.extend(vec![THETA_TAIL; TAIL]);
    let global = vec![
        Segment { base: THETA, bytes: le16(&th) },
        Segment { base: M, bytes: tail32(&c.m) },
        Segment { base: V, bytes: tail32(&c.v) },
        Segment { base: MP, bytes: tail32(&c.mp) },
    ];
    let f = |x: f32| x.to_bits() as u64;
    let args = named(&[
        ("theta", THETA),
        ("m", M),
        ("v", V),
        ("mp", MP),
        ("n", n as u64),
        ("b1", f(h.b1)),
        ("omb1", f(h.omb1)),
        ("b2", f(h.b2)),
        ("omb2", f(h.omb2)),
        ("eps", f(h.eps)),
        ("neg_lr", f(h.neg_lr)),
        ("neg_lr_wd", f(h.neg_lr_wd)),
        ("bc1", f(h.bc1)),
        ("bc2", f(h.bc2)),
        ("has_wd", h.has_wd as u64),
        ("sr_key", key),
        ("sr_ctr_base", ctr),
    ]);
    launch(ptx, global, args, n, order)
}

const SIZES: [usize; 4] = [1, 255, 257, 1000];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

// ---------------------------------------------------------------------------
// 1. Agreement
// ---------------------------------------------------------------------------

#[test]
fn the_kir_probe_agrees_with_the_hand_written_one_bit_for_bit() {
    let (hand, kir) = (hand_ptx(Kernel::Probe), kir_ptx(Kernel::Probe));
    for (j, &n) in SIZES.iter().enumerate() {
        let src = f32_bits(n, j as u64 + 1);
        for key in keys() {
            for order in ORDERS {
                assert!(run_probe(&hand, &src, key, order) == run_probe(&kir, &src, key, order), "n={n} {key:x?} {order:?}");
            }
        }
    }
}

#[test]
fn the_kir_step_agrees_with_the_hand_written_one_bit_for_bit() {
    let (hand, kir) = (hand_ptx(Kernel::Step), kir_ptx(Kernel::Step));
    for (j, &n) in SIZES.iter().enumerate() {
        let c = case(n, j as u64 + 5);
        for (k, h) in hypers().iter().enumerate() {
            let key = keys()[k % 4];
            for order in ORDERS {
                assert!(run_step(&hand, &c, h, key, order) == run_step(&kir, &c, h, key, order), "n={n} {h:?} {order:?}");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Correctness
// ---------------------------------------------------------------------------

#[test]
fn the_probe_is_the_host_reference_bit_for_bit() {
    let n = 1000;
    let src = f32_bits(n, 9);
    for key in keys() {
        let mem = run_probe(&kir_ptx(Kernel::Probe), &src, key, Order::Ascending);
        let dst = halves(&mem[1]);
        for i in 0..n {
            let want = round(src[i], mix64(key.0, key.1.wrapping_add(i as u64)) as u16);
            assert_eq!(dst[i], want, "{key:x?} i={i} src={:#010x}", src[i]);
        }
        assert!(dst[n..].iter().all(|&h| h == POISON16), "{key:x?}: wrote past n");
        assert_eq!(words(&mem[0])[..n], src[..], "src is only read");
    }
}

/// The f32 step on the widened θ, every operation rounded on its own and
/// the quotient the interpreter's `div.approx` (IEEE).
fn step(th: f32, m: f32, v: f32, g: f32, h: &Hyper) -> (f32, f32, f32) {
    let m2 = m * h.b1 + g * h.omb1;
    let v2 = v * h.b2 + (g * g) * h.omb2;
    let u = (m2 * h.bc1) / ((v2 * h.bc2).sqrt() + h.eps);
    let mut adj = u * h.neg_lr;
    if h.has_wd != 0 {
        adj += th * h.neg_lr_wd;
    }
    (th + adj, m2, v2)
}

fn same(got: u32, want: u32) -> bool {
    got == want || (f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan())
}

#[test]
fn the_step_is_the_f32_step_then_the_host_rounding() {
    let c = with_saturating(case(1000, 17));
    let n = c.theta.len();
    let mut saturated = 0;
    for (k, h) in hypers().iter().enumerate() {
        let key = keys()[k % 4];
        let mem = run_step(&kir_ptx(Kernel::Step), &c, h, key, Order::Ascending);
        let (th, m, v, mp) = (halves(&mem[0]), words(&mem[1]), words(&mem[2]), words(&mem[3]));
        for i in 0..n {
            let f = f32::from_bits;
            let wide = f((c.theta[i] as u32) << 16);
            let (t, wm, wv) = step(wide, f(c.m[i]), f(c.v[i]), f(c.mp[i]), h);
            let want = round(t.to_bits(), mix64(key.0, key.1.wrapping_add(i as u64)) as u16);
            // A NaN θ' rounds to the canonical quiet NaN whatever its payload.
            assert_eq!(th[i], want, "{h:?} i={i}: θ' = {t:e}");
            if t.is_finite() && want & 0x7FFF == 0x7F7F && t.abs() > f32::from_bits(0x7F7F_0000) {
                saturated += 1;
            }
            assert!(same(m[i], wm.to_bits()), "{h:?} i={i}: m");
            assert!(same(v[i], wv.to_bits()), "{h:?} i={i}: v");
        }
        assert!(th[n..].iter().all(|&w| w == THETA_TAIL), "{h:?}: θ past n moved");
        for buf in [&m, &v, &mp] {
            assert!(buf[n..].iter().all(|&w| w == IN_TAIL), "{h:?}: a word past n moved");
        }
        assert_eq!(mp[..n], c.mp[..], "the gradient is only read");
    }
    assert!(saturated > 0, "no element exercised the saturating carry");
}

#[test]
fn the_kir_kernels_keep_the_ffi_signatures_and_entry_names() {
    for (k, name) in [(Kernel::Probe, SR_BF16_ROUND_PROBE_NAME), (Kernel::Step, FASE_ADAMW_STEP_BF16SR_NAME)] {
        let (h, kir) = (hand_ptx(k), kir_ptx(k));
        assert_eq!(parse_signature(&kir), parse_signature(&h), "{k:?}");
        for p in [&h, &kir] {
            assert!(p.contains(&format!(".visible .entry {name}(")), "{k:?}");
        }
    }
}

/// The bf16 path in the hand kernels' spelling: `.u16` memory, the
/// `cvt.u32.u16` widening and `cvt.u16.u32` narrowing, `mov.b32` between
/// f32 and its bits; the hash in 64-bit integer operations; and the step's
/// arithmetic counted against the hand kernel.
#[test]
fn the_bit_path_is_spelled_as_in_the_hand_kernels() {
    for k in [Kernel::Probe, Kernel::Step] {
        let (h, kir) = (hand_ptx(k), kir_ptx(k));
        for form in [
            "st.global.u16 ",
            "cvt.u16.u32 ",
            "xor.b64 ",
            "shr.u64 ",
            "cvt.u32.u64 ",
            "mul.rn.f32 ",
            "add.rn.f32 ",
            "sqrt.rn.f32 ",
            "div.approx.f32 ",
        ] {
            assert_eq!(kir.matches(form).count(), h.matches(form).count(), "{k:?} {form}");
        }
        for never in ["fma.", "mul.f32 ", "add.f32 ", "div.rn"] {
            assert!(!kir.contains(never), "{k:?} {never}");
        }
    }
    // The hash's three 64-bit multiplies (KIR also scales each address with
    // a `mul.lo.u64` by the element size; the hand kernels shift).
    for k in [Kernel::Probe, Kernel::Step] {
        let hash = kir_ptx(k).lines().filter(|l| l.contains("mul.lo.u64") && !l.ends_with(", 2;") && !l.ends_with(", 4;")).count();
        assert_eq!(hash, 3, "{k:?}");
        assert_eq!(hand_ptx(k).matches("mul.lo.u64 ").count(), 3, "{k:?}");
    }
    let (h, kir) = (hand_ptx(Kernel::Step), kir_ptx(Kernel::Step));
    for form in ["ld.global.u16 ", "cvt.u32.u16 ", "mov.b32 "] {
        assert_eq!(kir.matches(form).count(), h.matches(form).count(), "step {form}");
    }
    assert_eq!(kir.matches("mov.b32 ").count(), 2, "θ widened in, θ' out as bits");
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on a ragged
/// size, under any key pair and either schedule (or faults). The step runs
/// with a decaying and a non-decaying hyperparameter set.
fn caught(k: Kernel, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(k);
    let mutant = mutate(&kir_ptx(k));
    let differs = |run: &dyn Fn(&str) -> Vec<Vec<u8>>| {
        let expect = run(&hand);
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mutant))) {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
    };
    match k {
        Kernel::Probe => {
            let src = f32_bits(257, 3);
            keys().into_iter().any(|key| ORDERS.into_iter().any(|o| differs(&|p| run_probe(p, &src, key, o))))
        }
        Kernel::Step => {
            let c = with_saturating(case(257, 3));
            let hs = hypers();
            [(&hs[1], keys()[1]), (&hs[3], keys()[3])]
                .into_iter()
                .any(|(h, key)| ORDERS.into_iter().any(|o| differs(&|p| run_step(p, &c, h, key, o))))
        }
    }
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

const BOTH: [Kernel; 2] = [Kernel::Probe, Kernel::Step];

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for k in BOTH {
        assert!(!caught(k, |p| p.to_string()), "{k:?}");
    }
}

#[test]
fn relaxing_the_bound_or_ignoring_the_block_index_is_caught() {
    for k in BOTH {
        let p = kir_ptx(k);
        assert_eq!((p.matches("setp.ge.u64 ").count(), p.matches("%ctaid.x;").count()), (1, 1), "{k:?}");
        assert!(caught(k, |p| p.replacen("setp.ge.u64 ", "setp.gt.u64 ", 1)), "{k:?} bound");
        assert!(caught(k, |p| p.replacen("%ctaid.x;", "0;", 1)), "{k:?} block index");
    }
}

/// Every address's element size: the probe's f32 load and u16 store; the
/// step's θ load and store (2) and its five f32 accesses (4).
#[test]
fn nudging_an_element_size_is_caught() {
    for (k, sizes) in [(Kernel::Probe, [(", 4;", 1), (", 2;", 1)]), (Kernel::Step, [(", 4;", 5), (", 2;", 2)])] {
        let p = kir_ptx(k);
        for (size, sites) in sizes {
            let found = p.lines().filter(|l| l.contains("mul.lo.u64") && l.contains(size)).count();
            assert_eq!(found, sites, "{k:?} {size}");
            for i in 0..sites {
                let bigger = if size == ", 4;" { ", 8;" } else { ", 4;" };
                assert!(caught(k, |p| nudge(p, size, bigger, i)), "{k:?} {size} site {i}");
            }
        }
    }
}

/// splitmix64: each multiplier's low bit, each xorshift amount, the dither
/// mask, the counter (`sr_ctr_base + i`) and the key.
#[test]
fn every_part_of_the_hash_is_caught() {
    for k in BOTH {
        for c in [SR_SPLITMIX_GAMMA, 0xBF58_476D_1CE4_E5B9, 0x94D0_49BB_1331_11EB] {
            let (from, to) = (format!(", {c};"), format!(", {};", c ^ 2));
            assert_eq!(kir_ptx(k).matches(&from).count(), 1, "{k:?} {c:#x}");
            assert!(caught(k, |p| p.replacen(&from, &to, 1)), "{k:?} {c:#x}");
        }
        for (from, to) in [(", 30;", ", 29;"), (", 27;", ", 28;"), (", 31;", ", 32;"), (", 65535;", ", 32767;")] {
            assert_eq!(kir_ptx(k).matches(from).count(), 1, "{k:?} {from}");
            assert!(caught(k, |p| p.replacen(from, to, 1)), "{k:?} {from}");
        }
        assert!(caught(k, |p| p.replacen("[param_sr_ctr_base]", "[param_sr_key]", 1)), "{k:?} counter base");
        assert!(caught(k, |p| p.replacen("[param_sr_key]", "[param_sr_ctr_base]", 1)), "{k:?} key");
    }
}

/// The rounding: the sign, exponent and mantissa masks, each special value,
/// both shifts by 16 (the sign's and the truncation's), and each branch.
#[test]
fn every_part_of_the_rounding_is_caught() {
    for k in BOTH {
        let p = kir_ptx(k);
        for (from, to) in [
            (", 2147483648;", ", 1073741824;"), // sign mask
            (", 8388607;", ", 4194303;"),       // mantissa mask
            (", 32639;", ", 32640;"),           // saturate 0x7f7f
            (", 32704;", ", 32640;"),           // quiet NaN 0x7fc0
        ] {
            assert_eq!(p.matches(from).count(), 1, "{k:?} {from}");
            assert!(caught(k, |p| p.replacen(from, to, 1)), "{k:?} {from}");
        }
        // The exponent mask, at each of its four uses (two tests of two).
        let sites = p.matches(", 2139095040;").count();
        assert_eq!(sites, 4, "{k:?}");
        for i in 0..sites {
            assert!(caught(k, |p| nudge(p, ", 2139095040;", ", 2130706432;", i)), "{k:?} exponent mask {i}");
        }
        // ±∞ is 0x7f80.
        assert_eq!(p.matches(", 32640;").count(), 1, "{k:?}");
        assert!(caught(k, |p| p.replacen(", 32640;", ", 32639;", 1)), "{k:?} inf");
        // Every shift by 16: the sign's, the truncation's (and the step's widening).
        let sixteens = p.matches(", 16;").count();
        assert_eq!(sixteens, if k == Kernel::Step { 3 } else { 2 }, "{k:?}");
        for i in 0..sixteens {
            assert!(caught(k, |p| nudge(p, ", 16;", ", 15;", i)), "{k:?} shift {i}");
        }
        // Each branch inverted: special, overflow, NaN.
        for (i, from, to) in [(0, "setp.eq.u32 ", "setp.ne.u32 "), (1, "setp.eq.u32 ", "setp.ne.u32 "), (0, "setp.ne.u32 ", "setp.eq.u32 ")] {
            let i = if k == Kernel::Step && from == "setp.eq.u32 " { i + 1 } else { i }; // the step's has_wd test comes first
            assert!(caught(k, |p| nudge(p, from, to, i)), "{k:?} {from} {i}");
        }
    }
}

/// The step's own parts: the weight-decay test and the θ widening.
#[test]
fn the_step_specific_parts_are_caught() {
    let p = kir_ptx(Kernel::Step);
    assert_eq!(p.matches("[param_has_wd];").count(), 1);
    assert!(caught(Kernel::Step, |p| nudge(p, "setp.eq.u32 ", "setp.ne.u32 ", 0)), "decay test");
    assert!(caught(Kernel::Step, |p| p.replacen("cvt.u32.u16 ", "cvt.u32.u32 ", 1).replacen("ld.global.u16 ", "ld.global.u32 ", 1)), "θ read as u32");
}
