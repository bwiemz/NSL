//! The differential equivalence gate for `nsl_fase_fused_adamw_multi_bf16sr`,
//! the multi-tensor FASE-Deferred AdamW step with bf16 θ and stochastic
//! rounding, now built by `nsl_kir::kernels::optim` (new-roadmap item 5).
//!
//! The runtime carried it as hand-written PTX. It is now the f32 multi
//! kernel's flat-grid header, the per-parameter SR step's body and the one
//! SR-BF16 tail, shared with the kernels `fase_adamw_multi_kir_equivalence`
//! and `sr_bf16_kir_equivalence` gate. This file runs the frozen hand
//! module (`tests/fixtures/fase_adamw_multi_bf16sr_hand.rs`) and the KIR one
//! side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), over the flat grid the runtime
//! launches, for a ragged parameter list with an empty member:
//!
//! 1. **Agreement**: under two schedules, with and without weight decay and
//!    under four `(sr_key, ctrtab)` sets (one whose counter base wraps `u64`
//!    inside a parameter), the two kernels leave *the same bytes* in all of
//!    global memory.
//! 2. **Correctness**: each parameter's θ is the f32 step's θ' rounded by
//!    the host reference (`sr_bf16_round` of `sr_mix64(key, ctrtab[p] +
//!    e)`, restated), m and v are the f32 step's, `mp` is only read, and
//!    nothing past a parameter's length moves. And the contract the batching
//!    entry relies on: every parameter comes out byte-identical to the
//!    per-parameter KIR kernel `nsl_fase_fused_adamw_step_bf16sr` launched
//!    on it alone with `sr_ctr_base = ctrtab[p]`.
//! 3. **The gate bites**: the bound, the block and thread indices, each
//!    element size, each parameter slot, the counter, each hash constant and
//!    shift, each rounding mask and special value and each branch are
//!    caught.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;
use nsl_kir::kernels::optim::{
    fase_adamw_multi_bf16sr_ptx, fase_adamw_step_bf16sr_ptx, FASE_ADAMW_MULTI_BF16SR_NAME, SR_SPLITMIX_GAMMA,
};

#[allow(dead_code)]
#[path = "fixtures/fase_adamw_multi_bf16sr_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

/// Parameter lengths: ragged around the block size, and one empty member
/// (it gets no blocks and must not be touched).
const LENS: [usize; 5] = [257, 1, 0, 1000, 255];
/// Parameter 3's elements `SAT..SAT + 64` are built to saturate (see
/// [`params`]).
const SAT: usize = 900;
const TAIL: usize = 64;
/// What every f32 buffer holds past its length, and θ's bf16 tail: finite,
/// so a thread past the bound that ran the step would change it.
const IN_TAIL: u32 = 0xC040_0000; // -3.0
const THETA_TAIL: u16 = 0x3FC0; // 1.5

// Table addresses; parameter buffers live at `DATA + 0x100_0000 * slot`.
const TTAB: u64 = 0x0100_0000;
const MTAB: u64 = 0x0110_0000;
const VTAB: u64 = 0x0120_0000;
const MPTAB: u64 = 0x0130_0000;
const NTAB: u64 = 0x0140_0000;
const BPTAB: u64 = 0x0150_0000;
const BBTAB: u64 = 0x0160_0000;
const CTRTAB: u64 = 0x0170_0000;
const DATA: u64 = 0x1000_0000;
/// Segments before the parameter buffers: the eight tables.
const TABLES: usize = 8;

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

fn kir_ptx() -> String {
    trim(&String::from_utf8(fase_adamw_multi_bf16sr_ptx()).expect("ASCII"))
}

fn hand_ptx() -> String {
    trim(hand::FASE_FUSED_ADAMW_MULTI_BF16SR_PTX)
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

/// `(sr_key, ctrtab)` sets: all zero; a real step key over the real
/// registration bases (`param << 40`); a later step over offset bases; and
/// bases near `u64::MAX` so parameter 3's counter wraps mid-parameter.
fn keys() -> [(u64, [u64; 5]); 4] {
    let registered = [0, 1 << 40, 2 << 40, 3 << 40, 4 << 40];
    [
        (0, [0; 5]),
        (step_key(42, 1), registered),
        (step_key(7, 1000), registered.map(|c| c + 5)),
        (u64::MAX, [u64::MAX - 7, 3, 0, u64::MAX - 100, 1 << 63]),
    ]
}

// ---------------------------------------------------------------------------
// Inputs and launches
// ---------------------------------------------------------------------------

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    *s
}

/// f32 corners: signed zeros, subnormals, max-normal, ±∞, NaNs, ordinary.
const F32_CORNERS: [u32; 12] = [
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

/// f32 words: a corner every seventh element, otherwise magnitudes from
/// 1e-30 to 1e30. `nonneg` folds them positive, for the second moment.
fn f32_words(n: usize, s: &mut u64, nonneg: bool) -> Vec<u32> {
    (0..n)
        .map(|k| {
            let bits = if k % 7 == 3 {
                F32_CORNERS[(k / 7) % F32_CORNERS.len()]
            } else {
                let x = ((lcg(s) >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0;
                let scale = [1e-30f32, 1e-3, 1.0, 1e3, 1e30][((*s >> 20) % 5) as usize];
                (x * scale).to_bits()
            };
            if nonneg { bits & 0x7FFF_FFFF } else { bits }
        })
        .collect()
}

struct Param {
    theta: Vec<u16>,
    m: Vec<u32>,
    v: Vec<u32>,
    mp: Vec<u32>,
}

/// bf16 θ (every class first, then random bits) and f32 moments and
/// gradients. Parameter 3's elements `SAT..SAT + 64` push θ just past bf16's
/// max-normal under the large-epsilon set (`hypers()[3]`): θ is ±`0x7F7F`, m
/// is ∓1e38 and v and the gradient 0, so a large enough dither carries into
/// the all-ones exponent and the tail saturates. Random inputs almost never
/// reach it.
fn params(seed: u64) -> Vec<Param> {
    const THETA: [u16; 10] = [0x0000, 0x8000, 0x0001, 0x7F7F, 0xFF7F, 0x7F80, 0xFF80, 0x7FC1, 0x3F80, 0xBE4C];
    LENS.iter()
        .enumerate()
        .map(|(j, &n)| {
            let mut s = seed + 100 * j as u64;
            let theta = (0..n)
                .map(|k| if k < THETA.len() { THETA[(k + j) % THETA.len()] } else { (lcg(&mut s) >> 48) as u16 })
                .collect();
            let mut p = Param { theta, m: f32_words(n, &mut s, false), v: f32_words(n, &mut s, true), mp: f32_words(n, &mut s, false) };
            if j == 3 {
                for k in SAT..SAT + 64 {
                    let neg = k % 2 == 1;
                    p.theta[k] = if neg { 0xFF7F } else { 0x7F7F };
                    p.m[k] = (if neg { 1e38f32 } else { -1e38f32 }).to_bits();
                    p.v[k] = 0;
                    p.mp[k] = 0;
                }
            }
            p
        })
        .collect()
}

fn le16(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le64(v: &[u64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn halves(b: &[u8]) -> Vec<u16> {
    b.chunks(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn theta_bytes(t: &[u16]) -> Vec<u8> {
    let mut b = le16(t);
    b.extend(le16(&[THETA_TAIL; TAIL]));
    b
}

fn f32_bytes(v: &[u32]) -> Vec<u8> {
    let mut b = le32(v);
    b.extend(le32(&[IN_TAIL; TAIL]));
    b
}

/// `fase_step::build_block_tables`, restated: one block per `block`
/// elements of each parameter, zero-length parameters getting none.
fn block_tables(lens: &[usize], block: usize) -> (Vec<u32>, Vec<u32>) {
    let mut bparam = vec![];
    let mut bbase = vec![];
    for (j, &n) in lens.iter().enumerate() {
        for b in 0..n.div_ceil(block) {
            bparam.push(j as u32);
            bbase.push((b * block) as u32);
        }
    }
    (bparam, bbase)
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

impl Hyper {
    fn named(&self) -> [(&'static str, u64); 10] {
        let f = |x: f32| x.to_bits() as u64;
        [
            ("b1", f(self.b1)),
            ("omb1", f(self.omb1)),
            ("b2", f(self.b2)),
            ("omb2", f(self.omb2)),
            ("eps", f(self.eps)),
            ("neg_lr", f(self.neg_lr)),
            ("neg_lr_wd", f(self.neg_lr_wd)),
            ("bc1", f(self.bc1)),
            ("bc2", f(self.bc2)),
            ("has_wd", self.has_wd as u64),
        ]
    }
}

/// AdamW steps 1 and 1000 with and without decay, a large-epsilon Adam
/// step, and one with every scalar distinct so a swapped slot shows.
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

/// The base address of buffer `k` (0 θ, 1 m, 2 v, 3 mp) of parameter `j`.
fn buf(j: usize, k: usize) -> u64 {
    DATA + 0x100_0000 * (4 * j + k) as u64
}

fn run_ctas(prog: &Program, args: &HashMap<String, u64>, global: &mut Vec<Segment>, grid: u32, order: Order) {
    let mut ctas: Vec<u32> = (0..grid).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for cta in ctas {
        let mut launch =
            Launch { prog, args, global, shared: vec![], ctaid: cta, ctaid_y: 0, nctaid_x: 0, nctaid_y: 1, ntid: ELEMENTWISE_BLOCK, steps: 0 };
        run_cta(&mut launch, order);
    }
}

/// Run the multi kernel `ptx` over the flat grid. Returns all of global
/// memory, in segment order: the eight tables, then θ, m, v, mp of each
/// parameter.
fn run(ptx: &str, ps: &[Param], h: &Hyper, (key, ctrs): (u64, [u64; 5]), order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let (bparam, bbase) = block_tables(&LENS, ELEMENTWISE_BLOCK as usize);
    let table = |k: usize| le64(&(0..LENS.len()).map(|j| buf(j, k)).collect::<Vec<_>>());
    let mut global = vec![
        Segment { base: TTAB, bytes: table(0) },
        Segment { base: MTAB, bytes: table(1) },
        Segment { base: VTAB, bytes: table(2) },
        Segment { base: MPTAB, bytes: table(3) },
        Segment { base: NTAB, bytes: le32(&LENS.map(|n| n as u32)) },
        Segment { base: BPTAB, bytes: le32(&bparam) },
        Segment { base: BBTAB, bytes: le32(&bbase) },
        Segment { base: CTRTAB, bytes: le64(&ctrs) },
    ];
    for (j, p) in ps.iter().enumerate() {
        global.push(Segment { base: buf(j, 0), bytes: theta_bytes(&p.theta) });
        for (k, data) in [&p.m, &p.v, &p.mp].into_iter().enumerate() {
            global.push(Segment { base: buf(j, k + 1), bytes: f32_bytes(data) });
        }
    }
    let mut args: HashMap<String, u64> = [
        ("ttab", TTAB),
        ("mtab", MTAB),
        ("vtab", VTAB),
        ("mptab", MPTAB),
        ("ntab", NTAB),
        ("bptab", BPTAB),
        ("bbtab", BBTAB),
        ("ctrtab", CTRTAB),
        ("sr_key", key),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    args.extend(h.named().into_iter().map(|(k, v)| (k.to_string(), v)));
    run_ctas(&prog, &args, &mut global, bparam.len() as u32, order);
    global.into_iter().map(|s| s.bytes).collect()
}

/// The per-parameter KIR SR step on parameter `p` alone, at the multi
/// kernel's addresses. Returns `[θ, m, v, mp]`.
fn run_single(p: &Param, h: &Hyper, key: u64, ctr: u64, j: usize) -> Vec<Vec<u8>> {
    let prog = parse(&trim(&String::from_utf8(fase_adamw_step_bf16sr_ptx()).expect("ASCII")));
    let mut global = vec![
        Segment { base: buf(j, 0), bytes: theta_bytes(&p.theta) },
        Segment { base: buf(j, 1), bytes: f32_bytes(&p.m) },
        Segment { base: buf(j, 2), bytes: f32_bytes(&p.v) },
        Segment { base: buf(j, 3), bytes: f32_bytes(&p.mp) },
    ];
    let n = p.theta.len();
    let mut args: HashMap<String, u64> = [
        ("theta", buf(j, 0)),
        ("m", buf(j, 1)),
        ("v", buf(j, 2)),
        ("mp", buf(j, 3)),
        ("n", n as u64),
        ("sr_key", key),
        ("sr_ctr_base", ctr),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    args.extend(h.named().into_iter().map(|(k, v)| (k.to_string(), v)));
    run_ctas(&prog, &args, &mut global, n.div_ceil(ELEMENTWISE_BLOCK as usize) as u32, Order::Ascending);
    global.into_iter().map(|s| s.bytes).collect()
}

const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

// ---------------------------------------------------------------------------
// 1. Agreement
// ---------------------------------------------------------------------------

#[test]
fn the_kir_kernel_agrees_with_the_hand_written_one_bit_for_bit() {
    let (hand, kir) = (hand_ptx(), kir_ptx());
    for seed in [1, 2] {
        let ps = params(seed);
        for (k, h) in hypers().iter().enumerate() {
            let key = keys()[(k + seed as usize) % 4];
            for order in ORDERS {
                assert!(run(&hand, &ps, h, key, order) == run(&kir, &ps, h, key, order), "seed={seed} {h:?} {order:?}: global memory differs");
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Correctness
// ---------------------------------------------------------------------------

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
fn every_parameter_is_the_f32_step_then_the_host_rounding_at_its_own_counter() {
    let ps = params(9);
    let mut saturated = 0;
    for (k, h) in hypers().iter().enumerate() {
        let (key, ctrs) = keys()[k % 4];
        let mem = run(&kir_ptx(), &ps, h, (key, ctrs), Order::Ascending);
        for (j, p) in ps.iter().enumerate() {
            let n = LENS[j];
            let seg = |b: usize| &mem[TABLES + 4 * j + b];
            let (th, m, v, mp) = (halves(seg(0)), words(seg(1)), words(seg(2)), words(seg(3)));
            for i in 0..n {
                let f = f32::from_bits;
                let wide = f((p.theta[i] as u32) << 16);
                let (t, wm, wv) = step(wide, f(p.m[i]), f(p.v[i]), f(p.mp[i]), h);
                let want = round(t.to_bits(), mix64(key, ctrs[j].wrapping_add(i as u64)) as u16);
                assert_eq!(th[i], want, "{h:?} param {j} i={i}: θ' = {t:e}");
                if t.is_finite() && want & 0x7FFF == 0x7F7F && t.abs() > f32::from_bits(0x7F7F_0000) {
                    saturated += 1;
                }
                assert!(same(m[i], wm.to_bits()), "{h:?} param {j} i={i}: m");
                assert!(same(v[i], wv.to_bits()), "{h:?} param {j} i={i}: v");
            }
            assert!(th[n..].iter().all(|&w| w == THETA_TAIL), "{h:?} param {j}: θ past its length moved");
            for b in [&m, &v, &mp] {
                assert!(b[n..].iter().all(|&w| w == IN_TAIL), "{h:?} param {j}: a word past its length moved");
            }
            assert_eq!(mp[..n], p.mp[..], "{h:?} param {j}: the gradient is only read, not zeroed");
        }
    }
    assert!(saturated > 0, "no element exercised the saturating carry");
}

/// The batching entry's contract: one multi launch leaves every parameter
/// exactly as the per-parameter kernel would, at `sr_ctr_base = ctrtab[p]`.
#[test]
fn every_parameter_is_the_per_parameter_kernel_bit_for_bit() {
    let ps = params(4);
    for (k, h) in hypers().iter().enumerate() {
        let (key, ctrs) = keys()[k % 4];
        let mem = run(&kir_ptx(), &ps, h, (key, ctrs), Order::Descending);
        for (j, p) in ps.iter().enumerate() {
            let single = run_single(p, h, key, ctrs[j], j);
            for (b, want) in single.iter().enumerate() {
                assert!(mem[TABLES + 4 * j + b] == *want, "{h:?} param {j} buffer {b}");
            }
        }
    }
}

#[test]
fn the_kir_kernel_keeps_the_ffi_signature_and_entry_name() {
    let (h, k) = (hand_ptx(), kir_ptx());
    assert_eq!(parse_signature(&k), parse_signature(&h));
    for p in [&h, &k] {
        assert!(p.contains(&format!(".visible .entry {FASE_ADAMW_MULTI_BF16SR_NAME}(")));
    }
}

/// Every mnemonic the bit path and the arithmetic use, counted in both
/// modules; no `fma`, no bare arithmetic; neither reads the block size.
#[test]
fn every_operation_is_spelled_as_in_the_hand_kernel() {
    let (h, k) = (hand_ptx(), kir_ptx());
    for form in [
        "ld.global.u16 ",
        "st.global.u16 ",
        "cvt.u32.u16 ",
        "cvt.u16.u32 ",
        "mov.b32 ",
        "xor.b64 ",
        "shr.u64 ",
        "cvt.u32.u64 ",
        "mul.rn.f32 ",
        "add.rn.f32 ",
        "sqrt.rn.f32 ",
        "div.approx.f32 ",
        "st.global.f32 ",
    ] {
        assert_eq!(k.matches(form).count(), h.matches(form).count(), "{form}");
    }
    assert_eq!(k.matches("div.approx.f32 ").count(), 1);
    assert_eq!(k.matches("mul.rn.f32 ").count(), 9, "no mp_scale");
    assert_eq!(k.matches("st.global.f32 ").count(), 2, "m and v: no mp zeroing");
    for never in ["fma.", "mul.f32 ", "add.f32 ", "div.rn", "%ntid"] {
        assert!(!k.contains(never), "KIR module has {never}");
        assert!(!h.contains(never), "hand module has {never}");
    }
    // The hash's three 64-bit multiplies (KIR also scales each address with
    // a `mul.lo.u64` by the element size; the hand kernel shifts).
    let hash = k.lines().filter(|l| l.contains("mul.lo.u64") && ![", 2;", ", 4;", ", 8;"].iter().any(|s| l.ends_with(s))).count();
    assert_eq!(hash, 3);
    assert_eq!(h.matches("mul.lo.u64 ").count(), 3);
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel, under either
/// schedule, with a decaying set and the large-epsilon (saturating) set, at
/// the registered bases and the wrapping ones (or faults).
fn caught(mutate: impl Fn(&str) -> String) -> bool {
    let ps = params(3);
    let hand = hand_ptx();
    let mutant = mutate(&kir_ptx());
    let hs = hypers();
    [(&hs[1], keys()[1]), (&hs[3], keys()[3])].into_iter().any(|(h, key)| {
        ORDERS.into_iter().any(|order| {
            let expect = run(&hand, &ps, h, key, order);
            match std::panic::catch_unwind(|| run(&mutant, &ps, h, key, order)) {
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
fn relaxing_the_bound_or_ignoring_an_index_is_caught() {
    let k = kir_ptx();
    assert_eq!(k.matches("setp.ge.u32 ").count(), 1);
    assert_eq!(k.matches("%ctaid.x;").count(), 1);
    assert_eq!(k.matches("%tid.x;").count(), 1);
    assert!(caught(|p| p.replacen("setp.ge.u32 ", "setp.gt.u32 ", 1)), "bound");
    assert!(caught(|p| p.replacen("%ctaid.x;", "0;", 1)), "block index");
    assert!(caught(|p| p.replacen("%tid.x;", "0;", 1)), "thread index");
    assert!(caught(|p| p.replacen("add.u32 ", "sub.u32 ", 1)), "block base + thread");
}

/// Each element size: θ's load and store use 2; the three u32 table reads,
/// the three f32 loads and the two f32 stores use 4; the four pointer-table
/// reads and the counter-table read use 8.
#[test]
fn nudging_an_element_size_is_caught() {
    let k = kir_ptx();
    for (size, sites, other) in [(", 2;", 2, ", 4;"), (", 4;", 8, ", 8;"), (", 8;", 5, ", 4;")] {
        let found = k.lines().filter(|l| l.contains("mul.lo.u64") && l.ends_with(size)).count();
        assert_eq!(found, sites, "{size}");
        assert_eq!(k.matches(size).count(), sites, "{size}: only address scaling uses it");
        for i in 0..sites {
            assert!(caught(|p| nudge(p, size, other, i)), "{size} site {i}");
        }
    }
}

/// Reading any scalar, or any table, from its neighbour's slot is caught.
#[test]
fn every_parameter_slot_is_live() {
    let scalars = ["b1", "omb1", "b2", "omb2", "eps", "neg_lr", "neg_lr_wd", "bc1", "bc2", "sr_key"];
    let tables = ["ttab", "mtab", "vtab", "mptab", "ntab", "bptab", "bbtab", "ctrtab"];
    for names in [&scalars[..], &tables[..]] {
        for (j, name) in names.iter().enumerate() {
            let other = names[(j + 1) % names.len()];
            // KIR names a parameter `param_<name>`.
            let from = format!("[param_{name}];");
            assert_eq!(kir_ptx().matches(&from).count(), 1, "{name}");
            assert!(caught(|p| p.replacen(&from, &format!("[param_{other}];"), 1)), "{name} read from {other}");
        }
    }
}

/// The counter is `ctrtab[p] + e`: the add just before the first splitmix64
/// multiplier.
#[test]
fn the_counter_is_caught() {
    let k = kir_ptx();
    let gamma = k.lines().position(|l| l.contains(&format!(", {SR_SPLITMIX_GAMMA};"))).expect("γ");
    let ctr = k.lines().nth(gamma - 1).expect("the counter add");
    assert!(ctr.trim_start().starts_with("add.u64 "), "{ctr}");
    let at = k.lines().take(gamma - 1).filter(|l| l.contains("add.u64 ")).count();
    assert!(caught(|p| nudge(p, "add.u64 ", "sub.u64 ", at)), "ctrtab[p] - e");
}

/// splitmix64: each multiplier's low bit, each xorshift amount, the dither
/// mask.
#[test]
fn every_part_of_the_hash_is_caught() {
    for c in [SR_SPLITMIX_GAMMA, 0xBF58_476D_1CE4_E5B9, 0x94D0_49BB_1331_11EB] {
        let (from, to) = (format!(", {c};"), format!(", {};", c ^ 2));
        assert_eq!(kir_ptx().matches(&from).count(), 1, "{c:#x}");
        assert!(caught(|p| p.replacen(&from, &to, 1)), "{c:#x}");
    }
    for (from, to) in [(", 30;", ", 29;"), (", 27;", ", 28;"), (", 31;", ", 32;"), (", 65535;", ", 32767;")] {
        assert_eq!(kir_ptx().matches(from).count(), 1, "{from}");
        assert!(caught(|p| p.replacen(from, to, 1)), "{from}");
    }
}

/// The rounding: the sign, exponent and mantissa masks, each special value,
/// every shift by 16 (θ's widening, the sign's and the truncation's), and
/// each branch, the weight-decay test among them.
#[test]
fn every_part_of_the_rounding_and_each_branch_is_caught() {
    let k = kir_ptx();
    for (from, to) in [
        (", 2147483648;", ", 1073741824;"), // sign mask
        (", 8388607;", ", 4194303;"),       // mantissa mask
        (", 32639;", ", 32640;"),           // saturate 0x7f7f
        (", 32704;", ", 32640;"),           // quiet NaN 0x7fc0
        (", 32640;", ", 32639;"),           // ±∞ 0x7f80
    ] {
        assert_eq!(k.matches(from).count(), 1, "{from}");
        assert!(caught(|p| p.replacen(from, to, 1)), "{from}");
    }
    let sites = k.matches(", 2139095040;").count();
    assert_eq!(sites, 4);
    for i in 0..sites {
        assert!(caught(|p| nudge(p, ", 2139095040;", ", 2130706432;", i)), "exponent mask {i}");
    }
    let sixteens = k.matches(", 16;").count();
    assert_eq!(sixteens, 3);
    for i in 0..sixteens {
        assert!(caught(|p| nudge(p, ", 16;", ", 15;", i)), "shift {i}");
    }
    assert_eq!(k.matches("[param_has_wd];").count(), 1);
    // Weight decay, special, overflow (setp.eq), then NaN (setp.ne).
    for i in 0..k.matches("setp.eq.u32 ").count() {
        assert!(caught(|p| nudge(p, "setp.eq.u32 ", "setp.ne.u32 ", i)), "setp.eq {i}");
    }
    assert!(caught(|p| p.replacen("setp.ne.u32 ", "setp.eq.u32 ", 1)), "NaN test");
    for i in 0..k.matches("add.rn.f32 ").count() {
        assert!(caught(|p| nudge(p, "add.rn.f32 ", "sub.rn.f32 ", i)), "add {i}");
    }
}

#[test]
fn a_neighbours_operation_is_caught() {
    let k = kir_ptx();
    for i in 0..k.matches("mul.rn.f32 ").count() {
        assert!(caught(|p| nudge(p, "mul.rn.f32 ", "add.rn.f32 ", i)), "mul {i}");
    }
    assert!(caught(|p| p.replacen("sqrt.rn.f32 ", "abs.f32 ", 1)), "sqrt");
    assert!(caught(|p| p.replacen("div.approx.f32 ", "mul.rn.f32 ", 1)), "div");
    assert!(caught(|p| p.replacen("cvt.u32.u16 ", "cvt.u32.u32 ", 1).replacen("ld.global.u16 ", "ld.global.u32 ", 1)), "θ read as u32");
}
