//! The differential equivalence gate for `nsl_fase_fused_adamw_multi_f32`,
//! the multi-tensor FASE-Deferred AdamW/Adam step, now built by
//! `nsl_kir::kernels::optim` (new-roadmap item 5).
//!
//! The runtime carried it as hand-written PTX. This file runs the frozen
//! hand module (`tests/fixtures/fase_adamw_multi_hand.rs`) and the KIR one
//! side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), over the FLAT grid the runtime
//! launches: one block per `bptab`/`bbtab` entry, built the way
//! `fase_step::build_block_tables` builds them, for a parameter list with
//! ragged lengths and a zero-length member.
//!
//! 1. **Agreement**: under two schedules, with and without weight decay,
//!    with and without the Phase-B clip pre-scale, the two kernels leave
//!    *the same bytes* in all of global memory (every parameter's θ, m, v,
//!    mp and their tails, and the tables).
//! 2. **Correctness**: each parameter is the AdamW formula in f32 with every
//!    operation rounded on its own, on the pre-scaled gradient; `mp` is
//!    zeroed; nothing past a parameter's length moves.
//! 3. **The spelling**: the interpreter never contracts and models
//!    `div.approx.f32` as the IEEE quotient, so every mnemonic's count is
//!    pinned against the hand module instead.
//! 4. **The gate bites**: the bound, the block and thread indices, the table
//!    reads, each element size, each scalar's slot, both branches and the
//!    zeroing store are caught.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;
use nsl_kir::kernels::optim::{fase_adamw_multi_ptx, FASE_ADAMW_MULTI_NAME};

#[allow(dead_code)]
#[path = "fixtures/fase_adamw_multi_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

/// Parameter lengths: ragged around the block size, and one empty member
/// (it gets no blocks and must not be touched).
const LENS: [usize; 5] = [257, 1, 0, 1000, 255];
const TAIL: usize = 64;
/// What every buffer holds past its length: finite, so a thread past the
/// bound that ran the step would change it.
const IN_TAIL: u32 = 0xC040_0000; // -3.0

// Table addresses; parameter buffers live at `DATA + 0x100_0000 * slot`.
const TTAB: u64 = 0x0100_0000;
const MTAB: u64 = 0x0110_0000;
const VTAB: u64 = 0x0120_0000;
const MPTAB: u64 = 0x0130_0000;
const NTAB: u64 = 0x0140_0000;
const BPTAB: u64 = 0x0150_0000;
const BBTAB: u64 = 0x0160_0000;
const DATA: u64 = 0x1000_0000;

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

fn kir_ptx() -> String {
    trim(&String::from_utf8(fase_adamw_multi_ptx()).expect("ASCII"))
}

fn hand_ptx() -> String {
    trim(hand::FASE_FUSED_ADAMW_MULTI_F32_PTX)
}

/// Deterministic values: IEEE corners first, then magnitudes from 1e-30 to
/// 1e30. `nonneg` folds them positive, for the second moment.
fn values(n: usize, seed: u64, nonneg: bool) -> Vec<u32> {
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
            let bits = if k < CORNERS.len() {
                CORNERS[(k + seed as usize) % CORNERS.len()]
            } else {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                let v = ((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0;
                let scale = [1e-30f32, 1e-3, 1.0, 1e3, 1e30][((s >> 20) % 5) as usize];
                (v * scale).to_bits()
            };
            if nonneg { bits & 0x7FFF_FFFF } else { bits }
        })
        .collect()
}

fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le64(v: &[u64]) -> Vec<u8> {
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

/// The nine `.f32` hyperparameters, the flag and the clip pre-scale.
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
    mp_scale: f32,
}

impl Hyper {
    fn named(&self) -> [(&'static str, u64); 11] {
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
            ("mp_scale", f(self.mp_scale)),
        ]
    }
}

/// Step 1 and step 1000 of AdamW (β = 0.9, 0.999; ε = 1e-8; lr 1e-3; wd
/// 0.01), with and without decay and with and without a clip pre-scale, a
/// large-epsilon Adam step, and one whose scalars are all distinct so a
/// swapped parameter slot shows.
fn hypers() -> Vec<Hyper> {
    let adamw = |t: i32, has_wd: u32, mp_scale: f32| {
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
            mp_scale,
        }
    };
    vec![
        adamw(1, 0, 1.0),
        adamw(1, 1, 1.0),
        adamw(1000, 1, 0.37),
        Hyper { eps: 1.0, ..adamw(3, 0, 0.5) },
        Hyper {
            b1: 0.7,
            omb1: 0.2,
            b2: 0.95,
            omb2: 0.03,
            eps: 1e-3,
            neg_lr: -0.25,
            neg_lr_wd: -0.125,
            bc1: 1.75,
            bc2: 3.5,
            has_wd: 7,
            mp_scale: 0.8125,
        },
    ]
}

struct Param {
    theta: Vec<u32>,
    m: Vec<u32>,
    v: Vec<u32>,
    mp: Vec<u32>,
}

fn params(seed: u64) -> Vec<Param> {
    LENS.iter()
        .enumerate()
        .map(|(j, &n)| {
            let s = seed + 100 * j as u64;
            Param { theta: values(n, s, false), m: values(n, s + 11, false), v: values(n, s + 23, true), mp: values(n, s + 37, false) }
        })
        .collect()
}

/// The base address of buffer `k` (0 θ, 1 m, 2 v, 3 mp) of parameter `j`.
fn buf(j: usize, k: usize) -> u64 {
    DATA + 0x100_0000 * (4 * j + k) as u64
}

/// Run `ptx` over the flat grid. Returns all of global memory, in segment
/// order: the seven tables, then θ, m, v, mp of each parameter.
fn run(ptx: &str, ps: &[Param], h: &Hyper, order: Order) -> Vec<Vec<u8>> {
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
    ];
    for (j, p) in ps.iter().enumerate() {
        for (k, data) in [&p.theta, &p.m, &p.v, &p.mp].into_iter().enumerate() {
            global.push(Segment { base: buf(j, k), bytes: with_tail(data) });
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
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    args.extend(h.named().into_iter().map(|(k, v)| (k.to_string(), v)));
    let mut ctas: Vec<u32> = (0..bparam.len() as u32).collect();
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

const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

#[test]
fn the_kir_kernel_agrees_with_the_hand_written_one_bit_for_bit() {
    let (hand, kir) = (hand_ptx(), kir_ptx());
    for seed in [1, 2] {
        let ps = params(seed);
        for h in hypers() {
            for order in ORDERS {
                assert!(run(&hand, &ps, &h, order) == run(&kir, &ps, &h, order), "seed={seed} {h:?} {order:?}: global memory differs");
            }
        }
    }
}

fn same(got: u32, want: u32) -> bool {
    got == want || (f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan())
}

/// AdamW in f32, every operation rounded on its own (Rust never
/// contracts), the quotient the interpreter's `div.approx` (IEEE), on the
/// pre-scaled gradient.
fn step(th: f32, m: f32, v: f32, mp: f32, h: &Hyper) -> (f32, f32, f32) {
    let g = if h.mp_scale == 1.0 { mp } else { mp * h.mp_scale };
    let m2 = m * h.b1 + g * h.omb1;
    let v2 = v * h.b2 + (g * g) * h.omb2;
    let u = (m2 * h.bc1) / ((v2 * h.bc2).sqrt() + h.eps);
    let mut adj = u * h.neg_lr;
    if h.has_wd != 0 {
        adj += th * h.neg_lr_wd;
    }
    (th + adj, m2, v2)
}

#[test]
fn every_parameter_is_adamw_with_every_operation_rounded_and_mp_is_zeroed() {
    let ps = params(9);
    for h in hypers() {
        let mem = run(&kir_ptx(), &ps, &h, Order::Ascending);
        for (j, p) in ps.iter().enumerate() {
            let n = LENS[j];
            let at = |k: usize| words(&mem[7 + 4 * j + k]);
            let (th, m, v, mp) = (at(0), at(1), at(2), at(3));
            for i in 0..n {
                let f = f32::from_bits;
                let (wt, wm, wv) = step(f(p.theta[i]), f(p.m[i]), f(p.v[i]), f(p.mp[i]), &h);
                assert!(same(th[i], wt.to_bits()), "{h:?} param {j} i={i}: theta {:#010x} vs {:#010x}", th[i], wt.to_bits());
                assert!(same(m[i], wm.to_bits()), "{h:?} param {j} i={i}: m");
                assert!(same(v[i], wv.to_bits()), "{h:?} param {j} i={i}: v");
                assert_eq!(mp[i], 0, "{h:?} param {j} i={i}: the accumulated gradient is zeroed");
            }
            for buf in [&th, &m, &v, &mp] {
                assert!(buf[n..].iter().all(|&w| w == IN_TAIL), "{h:?} param {j}: a word past its length moved");
            }
        }
    }
}

/// The unclipped path branches around the pre-scale, so it is the
/// single-parameter step exactly; a scale of 0.5 halves the gradient the
/// moments see.
#[test]
fn the_clip_pre_scale_reaches_the_moments_and_one_is_the_identity() {
    let base = hypers()[0];
    let ps = params(5);
    let first = |h: &Hyper| words(&run(&kir_ptx(), &ps, h, Order::Ascending)[7 + 1]);
    let unscaled = first(&base);
    let halved = first(&Hyper { mp_scale: 0.5, ..base });
    let omb1 = base.omb1;
    for i in 12..LENS[0] {
        let g = f32::from_bits(ps[0].mp[i]);
        let m0 = f32::from_bits(ps[0].m[i]);
        assert!(same(unscaled[i], (m0 * base.b1 + g * omb1).to_bits()), "i={i}: mp_scale 1.0 is the identity");
        assert!(same(halved[i], (m0 * base.b1 + (g * 0.5) * omb1).to_bits()), "i={i}: mp_scale 0.5");
    }
}

#[test]
fn the_kir_kernel_keeps_the_ffi_signature_and_entry_name() {
    let (h, k) = (hand_ptx(), kir_ptx());
    assert_eq!(parse_signature(&k), parse_signature(&h));
    for p in [&h, &k] {
        assert!(p.contains(&format!(".visible .entry {FASE_ADAMW_MULTI_NAME}(")));
    }
}

/// Every arithmetic mnemonic, counted in both modules: the same number of
/// each, one `div.approx`, no `div.rn`, no bare arithmetic, no `fma`.
#[test]
fn every_operation_is_spelled_as_in_the_hand_kernel() {
    let (h, k) = (hand_ptx(), kir_ptx());
    for form in ["mul.rn.f32 ", "add.rn.f32 ", "sub.rn.f32 ", "sqrt.rn.f32 ", "div.approx.f32 ", "div.rn.f32 "] {
        assert_eq!(k.matches(form).count(), h.matches(form).count(), "{form}");
    }
    assert_eq!(k.matches("div.approx.f32 ").count(), 1);
    assert_eq!(k.matches("mul.rn.f32 ").count(), 10);
    assert_eq!(k.matches("add.rn.f32 ").count(), 5);
    for bare in ["mul.f32 ", "add.f32 ", "sub.f32 ", "div.f32 ", "fma."] {
        assert!(!k.contains(bare), "KIR module has {bare}");
        assert!(!h.contains(bare), "hand module has {bare}");
    }
    // The flat-grid contract: neither module reads the block size.
    assert!(!k.contains("%ntid") && !h.contains("%ntid"));
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel, under either
/// schedule, for any hyperparameter set (or faults).
fn caught(mutate: impl Fn(&str) -> String) -> bool {
    let ps = params(3);
    let hand = hand_ptx();
    let mutant = mutate(&kir_ptx());
    hypers().iter().any(|h| {
        ORDERS.into_iter().any(|order| {
            let expect = run(&hand, &ps, h, order);
            match std::panic::catch_unwind(|| run(&mutant, &ps, h, order)) {
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

/// Named equivalent mutants: the interpreter never contracts and computes
/// `div.approx` as the IEEE quotient, so these are invisible to execution —
/// which is what the spelling test pins.
#[test]
fn rounding_spellings_are_invisible_to_execution_and_pinned_instead() {
    assert!(!caught(|p| p.replacen("div.approx.f32 ", "div.rn.f32 ", 1)));
    assert!(!caught(|p| p.replacen("mul.rn.f32 ", "mul.f32 ", 1)));
    assert!(!caught(|p| p.replacen("add.rn.f32 ", "add.f32 ", 1)));
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

/// Each element size: the three u32 table reads (bptab, bbtab, ntab), the
/// four f32 loads and the four f32 stores (m, v, θ and the zeroed mp) use 4;
/// the four pointer-table reads use 8.
#[test]
fn nudging_an_element_size_is_caught() {
    let k = kir_ptx();
    let fours = k.matches(", 4;").count();
    assert_eq!(fours, 11, "{k}");
    for i in 0..fours {
        assert!(caught(|p| nudge(p, ", 4;", ", 8;", i)), "4-byte address {i}");
    }
    let eights = k.matches(", 8;").count();
    assert_eq!(eights, 4, "{k}");
    for i in 0..eights {
        assert!(caught(|p| nudge(p, ", 8;", ", 4;", i)), "pointer-table address {i}");
    }
}

/// Reading any scalar, or any table, from its neighbour's slot is caught.
#[test]
fn every_parameter_slot_is_live() {
    let scalars = ["b1", "omb1", "b2", "omb2", "eps", "neg_lr", "neg_lr_wd", "bc1", "bc2", "mp_scale"];
    let tables = ["ttab", "mtab", "vtab", "mptab", "ntab", "bptab", "bbtab"];
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

#[test]
fn both_branches_are_caught_both_ways() {
    let k = kir_ptx();
    assert_eq!(k.matches("[param_has_wd];").count(), 1);
    assert!(caught(|p| p.replacen("setp.eq.u32 ", "setp.ne.u32 ", 1)), "weight decay inverted");
    assert_eq!(k.matches("setp.eq.f32 ").count(), 1);
    assert!(caught(|p| p.replacen("setp.eq.f32 ", "setp.ne.f32 ", 1)), "pre-scale inverted");
    let adds = k.matches("add.rn.f32 ").count();
    for i in 0..adds {
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
}

/// The folded `nsl_tensor_zero_inplace`: storing anything but +0 is caught.
#[test]
fn the_gradient_zeroing_store_is_caught() {
    assert_eq!(kir_ptx().matches("0f00000000;").count(), 1);
    assert!(caught(|p| p.replacen("0f00000000;", "0f80000000;", 1)), "-0 in place of +0");
}
