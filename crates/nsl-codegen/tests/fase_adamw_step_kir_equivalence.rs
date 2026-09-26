//! The differential equivalence gate for `nsl_fase_fused_adamw_step_f32`,
//! the fused FASE-Deferred AdamW/Adam step, now built by
//! `nsl_kir::kernels::optim` (new-roadmap item 5).
//!
//! The runtime carried it as hand-written PTX. This file runs the frozen
//! hand module (`tests/fixtures/fase_adamw_step_hand.rs`) and the KIR one
//! side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), over the grid the runtime launches
//! (`ceil(n / 256)` blocks of 256) plus one block more:
//!
//! 1. **Agreement**: under two schedules, with and without weight decay and
//!    across hyperparameter sets, the two kernels leave *the same bytes* in
//!    all of global memory (θ, m, v and the read-only gradient).
//! 2. **Correctness**: θ, m and v are the AdamW formula in f32 with every
//!    operation rounded on its own, and nothing past `n` moves.
//! 3. **The spelling**: the interpreter never contracts and models
//!    `div.approx.f32` as the IEEE quotient, so dropping a `.rn` or swapping
//!    `div.approx` for `div.rn` is invisible to execution. On the machine
//!    neither is: ptxas contracts a bare multiply into the add that reads it,
//!    and `div.approx` differs from `div.rn` by up to 2 ulp. So every
//!    mnemonic's count is pinned against the hand module.
//! 4. **The gate bites**: the bound, the block index, each element size, each
//!    hyperparameter's slot, the weight-decay branch and a neighbour's
//!    operation are caught.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;
use nsl_kir::kernels::optim::{fase_adamw_step_ptx, FASE_ADAMW_STEP_NAME};

#[allow(dead_code)]
#[path = "fixtures/fase_adamw_step_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const THETA: u64 = 0x1000_0000;
const M: u64 = 0x2000_0000;
const V: u64 = 0x3000_0000;
const MP: u64 = 0x4000_0000;
const TAIL: usize = 300;
/// What every buffer holds past `n`: finite, so a thread past the bound
/// that ran the step would change it.
const IN_TAIL: u32 = 0xC040_0000; // -3.0

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

fn kir_ptx() -> String {
    trim(&String::from_utf8(fase_adamw_step_ptx()).expect("ASCII"))
}

fn hand_ptx() -> String {
    trim(hand::FASE_FUSED_ADAMW_STEP_F32_PTX)
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

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn with_tail(v: &[u32]) -> Vec<u8> {
    let mut b = le32(v);
    b.extend(le32(&vec![IN_TAIL; TAIL]));
    b
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

/// Step 1 and step 1000 of AdamW (β = 0.9, 0.999; ε = 1e-8; lr 1e-3; wd
/// 0.01), with and without decay, a large-epsilon Adam step, and one whose
/// scalars are all distinct so a swapped parameter slot shows.
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
        },
    ]
}

struct Case {
    n: usize,
    theta: Vec<u32>,
    m: Vec<u32>,
    v: Vec<u32>,
    mp: Vec<u32>,
}

fn case(n: usize, seed: u64) -> Case {
    Case {
        n,
        theta: values(n, seed, false),
        m: values(n, seed + 11, false),
        v: values(n, seed + 23, true),
        mp: values(n, seed + 37, false),
    }
}

/// Run `ptx` on `c` over `ceil(n / 256) + 1` blocks. Returns all of global
/// memory: θ, m, v, mp.
fn run(ptx: &str, c: &Case, h: &Hyper, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let mut global = vec![
        Segment { base: THETA, bytes: with_tail(&c.theta) },
        Segment { base: M, bytes: with_tail(&c.m) },
        Segment { base: V, bytes: with_tail(&c.v) },
        Segment { base: MP, bytes: with_tail(&c.mp) },
    ];
    let mut args: HashMap<String, u64> =
        [("theta", THETA), ("m", M), ("v", V), ("mp", MP), ("n", c.n as u64)].into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    args.extend(h.named().into_iter().map(|(k, v)| (k.to_string(), v)));
    let grid = c.n.div_ceil(ELEMENTWISE_BLOCK as usize) as u32 + 1;
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
    let (hand, kir) = (hand_ptx(), kir_ptx());
    for (j, &n) in SIZES.iter().enumerate() {
        let c = case(n, j as u64 + 1);
        for h in hypers() {
            for order in ORDERS {
                assert!(run(&hand, &c, &h, order) == run(&kir, &c, &h, order), "n={n} {h:?} {order:?}: global memory differs");
            }
        }
    }
}

fn same(got: u32, want: u32) -> bool {
    got == want || (f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan())
}

/// AdamW in f32, every operation rounded on its own (Rust never
/// contracts), the quotient the interpreter's `div.approx` (IEEE).
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

#[test]
fn the_answer_is_adamw_with_every_operation_rounded() {
    let n = 1000;
    let c = case(n, 9);
    for h in hypers() {
        let mem = run(&kir_ptx(), &c, &h, Order::Ascending);
        let (th, m, v, mp) = (words(&mem[0]), words(&mem[1]), words(&mem[2]), words(&mem[3]));
        for i in 0..n {
            let f = f32::from_bits;
            let (wt, wm, wv) = step(f(c.theta[i]), f(c.m[i]), f(c.v[i]), f(c.mp[i]), &h);
            assert!(same(th[i], wt.to_bits()), "{h:?} i={i}: theta {:#010x} vs {:#010x}", th[i], wt.to_bits());
            assert!(same(m[i], wm.to_bits()), "{h:?} i={i}: m");
            assert!(same(v[i], wv.to_bits()), "{h:?} i={i}: v");
        }
        for buf in [&th, &m, &v, &mp] {
            assert!(buf[n..].iter().all(|&w| w == IN_TAIL), "{h:?}: a word past n moved");
        }
        assert_eq!(mp[..n], c.mp[..], "the gradient is only read");
    }
}

/// On ordinary values the step is AdamW: step 1 with a positive gradient
/// moves θ by about -lr, the first-step property of bias-corrected Adam.
#[test]
fn the_first_step_moves_each_weight_by_about_lr() {
    let h = hypers()[0];
    let c = Case { n: 3, theta: vec![0.5f32.to_bits(); 3], m: vec![0; 3], v: vec![0; 3], mp: [0.3f32, 2.0, 40.0].map(f32::to_bits).to_vec() };
    let th = words(&run(&kir_ptx(), &c, &h, Order::Ascending)[0]);
    for w in &th[..3] {
        assert!((f32::from_bits(*w) - (0.5 - 1e-3)).abs() < 1e-7, "{}", f32::from_bits(*w));
    }
}

#[test]
fn the_kir_kernel_keeps_the_ffi_signature_and_entry_name() {
    let (h, k) = (hand_ptx(), kir_ptx());
    assert_eq!(parse_signature(&k), parse_signature(&h));
    for p in [&h, &k] {
        assert!(p.contains(&format!(".visible .entry {FASE_ADAMW_STEP_NAME}(")));
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
    assert_eq!(k.matches("mul.rn.f32 ").count(), 9);
    assert_eq!(k.matches("add.rn.f32 ").count(), 5);
    for bare in ["mul.f32 ", "add.f32 ", "sub.f32 ", "div.f32 ", "fma."] {
        assert!(!k.contains(bare), "KIR module has {bare}");
        assert!(!h.contains(bare), "hand module has {bare}");
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on a ragged
/// size, under either schedule, for any hyperparameter set (or faults).
fn caught(mutate: impl Fn(&str) -> String) -> bool {
    let c = case(257, 3);
    let hand = hand_ptx();
    let mutant = mutate(&kir_ptx());
    hypers().iter().any(|h| {
        ORDERS.into_iter().any(|order| {
            let expect = run(&hand, &c, h, order);
            match std::panic::catch_unwind(|| run(&mutant, &c, h, order)) {
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
/// `div.approx` as the IEEE quotient, so these three are invisible to
/// execution — which is what the spelling test pins.
#[test]
fn rounding_spellings_are_invisible_to_execution_and_pinned_instead() {
    assert!(!caught(|p| p.replacen("div.approx.f32 ", "div.rn.f32 ", 1)));
    assert!(!caught(|p| p.replacen("mul.rn.f32 ", "mul.f32 ", 1)));
    assert!(!caught(|p| p.replacen("add.rn.f32 ", "add.f32 ", 1)));
}

#[test]
fn relaxing_the_bound_or_ignoring_the_block_index_is_caught() {
    let k = kir_ptx();
    assert_eq!(k.matches("setp.ge.u64 ").count(), 1);
    assert_eq!(k.matches("%ctaid.x;").count(), 1);
    assert!(caught(|p| p.replacen("setp.ge.u64 ", "setp.gt.u64 ", 1)), "bound");
    assert!(caught(|p| p.replacen("%ctaid.x;", "0;", 1)), "block index");
}

/// f32's size in each of the seven addresses: four loads (θ, m, v, mp) and
/// three stores (m, v, θ).
#[test]
fn nudging_an_element_size_is_caught() {
    let k = kir_ptx();
    let sites = k.matches(", 4;").count();
    assert_eq!(sites, 7, "{k}");
    for i in 0..sites {
        assert!(caught(|p| nudge(p, ", 4;", ", 8;", i)), "address {i}");
    }
}

/// Reading any hyperparameter from its neighbour's slot is caught: each
/// `ld.param` of a scalar is redirected in turn.
#[test]
fn every_hyperparameter_slot_is_live() {
    let names = ["b1", "omb1", "b2", "omb2", "eps", "neg_lr", "neg_lr_wd", "bc1", "bc2"];
    for (j, name) in names.iter().enumerate() {
        let other = names[(j + 1) % names.len()];
        // KIR names a parameter `param_<name>`.
        let from = format!("[param_{name}];");
        assert_eq!(kir_ptx().matches(&from).count(), 1, "{name}");
        assert!(caught(|p| p.replacen(&from, &format!("[param_{other}];"), 1)), "{name} read from {other}");
    }
}

#[test]
fn the_weight_decay_branch_is_caught_both_ways() {
    let k = kir_ptx();
    // Inverting the test.
    assert_eq!(k.matches("[param_has_wd];").count(), 1);
    assert!(caught(|p| p.replacen("setp.eq.u32 ", "setp.ne.u32 ", 1)), "inverted");
    // Each add, including the decay term's, turned into a subtract.
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
