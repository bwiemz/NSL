//! The regenerated dropout mask, on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), against the kernels it replaces.
//!
//! Source-AD dropout keeps a 32-byte key instead of an f32 mask
//! (`NSL_DROPOUT_REGEN`, the default): `nsl_dropout_key_f32` is
//! `nsl_dropout_f32` with the mask store replaced by a key, and
//! `nsl_dropout_bwd_f32` redraws the mask from the key and computes the
//! stored-mask backward. That backward was the fused elementwise chain
//! `mul(i0,i1);mul(p0,imm)` over (gradient, mask) on the normal schedule,
//! and two `nsl_mul_f32` under `--layerwise-accum`; both compute
//! `(dy * m) * f32(1/(1-p))` with round-to-nearest. This gate runs the real
//! modules side by side:
//!
//! 1. **The forward is unchanged.** Under two thread schedules, the key
//!    kernel writes the mask kernel's `out` byte for byte, and its key holds
//!    the magic, the counter base, the threshold, the scale's bits and the
//!    length. Nothing is written past `len`; a launch of length 0 still
//!    writes the key.
//! 2. **The backward is the stored-mask backward.** For each case, the mask
//!    kernel's mask and the fused chain kernel built by the compiler
//!    (`ew_chain_ptx::emit` of the dropout chain) give the reference `dx`;
//!    the key kernel's key and the regenerating backward must give the same
//!    bytes, for gradients with full mantissas, signed zeros, denormals,
//!    infinities and NaN, at p = 0.1 and 0.5, for counters inside the first
//!    2^32 block, straddling into the next, and deep in a high block.
//! 3. **A key that is not the launch's poisons the gradient**: a wrong magic,
//!    length, threshold or scale makes every element NaN; a shape stub's
//!    zeros and an f32 mask in the key's place are refused the same way.
//! 4. **The replay is the forward**: `nsl_dropout_apply_f32` over the forward's
//!    input and key (what a checkpointed backward runs instead of saving the
//!    output) writes the forward's output byte for byte.

use std::collections::HashMap;

use nsl_codegen::ew_chain_fusion::{ChainSig, ChainStep, EwOpcode, Operand};
use nsl_codegen::ew_chain_ptx::emit;
use nsl_kir::kernels::dropout::{
    dropout_apply_ptx, dropout_bwd_ptx, dropout_key_ptx, dropout_ptx, DropoutKeyWord,
    DROPOUT_HASH_MULTIPLIERS, DROPOUT_HASH_SHIFTS, DROPOUT_KEY_MAGIC, DROPOUT_KEY_WORDS,
};
use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const INP: u64 = 0x1000_0000;
const OUT: u64 = 0x2000_0000;
const SINK: u64 = 0x3000_0000; // the mask, or the key
const GRAD: u64 = 0x4000_0000;
const DX: u64 = 0x5000_0000;
/// The device word a `--cuda-graphs` region passes the counter base through.
const SEED_WORD: u64 = 0x6000_0000;
const TAIL: usize = 300;
const POISON: u32 = 0xA5A5_A5A5;

fn text(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes).expect("ASCII").trim_end_matches('\0').to_string()
}

fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn round(x: u32, k: usize) -> u32 {
    let x = x.wrapping_mul(DROPOUT_HASH_MULTIPLIERS[k]);
    x ^ (x >> DROPOUT_HASH_SHIFTS[k])
}

/// The kernels' hash of the 64-bit counter (restated as in
/// `dropout_kir_equivalence.rs`, which checks it against values computed
/// outside Rust).
fn hash(counter: u64) -> u32 {
    let (lo, hi) = (counter as u32, (counter >> 32) as u32);
    let key = round(round(round(hi, 0), 1), 2);
    round(round(round(lo, 0).wrapping_add(key), 1), 2)
}

/// Values with full mantissas, and IEEE corners at fixed places: both zeros,
/// both infinities, a NaN, the smallest and a larger denormal, the largest
/// finite value, one and minus one.
fn values(n: usize, seed: u64) -> Vec<u32> {
    const CORNERS: [u32; 10] =
        [0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0001, 0x0000_0001, 0x0040_0000, 0x7F7F_FFFF, 0x3F80_0000, 0xBF80_0000];
    let mut s = seed;
    (0..n)
        .map(|k| {
            if k % 17 == 3 {
                CORNERS[(k / 17 + seed as usize) % CORNERS.len()]
            } else {
                s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
                (((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0).to_bits()
            }
        })
        .collect()
}

/// One forward launch: what the runtime passes for `p` and a counter base.
#[derive(Debug, Clone, Copy)]
struct Case {
    len: usize,
    p: f64,
    seed: u64,
}

impl Case {
    /// `cuda::dropout_threshold_scale`.
    fn threshold(self) -> u32 {
        ((1.0 - self.p) * u32::MAX as f64) as u32
    }
    fn scale(self) -> f32 {
        (1.0 / (1.0 - self.p)) as f32
    }
}

fn cases() -> Vec<Case> {
    vec![
        Case { len: 1, p: 0.5, seed: 0 },
        Case { len: 1000, p: 0.1, seed: 12_345 },
        Case { len: 1000, p: 0.5, seed: 42 },
        // `seed + i` crosses from one 2^32 block into the next mid-launch.
        Case { len: 511, p: 0.1, seed: (1 << 32) - 200 },
        Case { len: 700, p: 0.5, seed: (5 << 32) + 17 },
        Case { len: 257, p: 0.1, seed: 0xFFFF_FFFF_0000_0000 },
    ]
}

fn args(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
    let mut m = HashMap::new();
    for (k, v) in pairs {
        m.insert((*k).to_string(), *v);
        m.insert(format!("param_{k}"), *v);
    }
    m
}

/// Launch `ptx` over `ceil(len / 256) + 1` blocks (one spare) of the
/// elementwise block, and return every segment's words.
fn launch(ptx: &str, args: &HashMap<String, u64>, mut global: Vec<Segment>, len: usize, order: Order) -> Vec<Vec<u32>> {
    let prog = parse(ptx);
    let grid = len.div_ceil(ELEMENTWISE_BLOCK as usize) as u32 + 1;
    let mut ctas: Vec<u32> = (0..grid).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for cta in ctas {
        let mut l = Launch {
            prog: &prog,
            args,
            global: &mut global,
            shared: vec![],
            ctaid: cta,
            ctaid_y: 0,
            nctaid_x: 0,
            nctaid_y: 1,
            ntid: ELEMENTWISE_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    global.into_iter().map(|s| words(&s.bytes)).collect()
}

/// A forward launch of the mask kernel (`key = false`) or the key kernel:
/// `(input, out, sink)` words, the sink holding `len` mask words or the key.
/// The counter base goes by value, as everywhere outside a `--cuda-graphs`
/// region.
fn forward(c: Case, key: bool, order: Order) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    forward_base(c, key, order, false)
}

/// `forward`, with the base passed through a device word when `via_word` is set,
/// as inside a `--cuda-graphs` region. The by-value `seed` then holds a decoy,
/// so a kernel that reads the wrong one fails. The word must only be read.
fn forward_base(c: Case, key: bool, order: Order, via_word: bool) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let x = values(c.len, c.seed ^ 0x55);
    let mut inp = x.clone();
    inp.extend(vec![0x3FC0_0000; TAIL]);
    let sink_words = if key { DROPOUT_KEY_WORDS } else { c.len } + TAIL;
    let mut global = vec![
        Segment { base: INP, bytes: le32(&inp) },
        Segment { base: OUT, bytes: le32(&vec![POISON; c.len + TAIL]) },
        Segment { base: SINK, bytes: le32(&vec![POISON; sink_words]) },
    ];
    let (seed, seed_ptr) = if via_word {
        global.push(Segment { base: SEED_WORD, bytes: c.seed.to_le_bytes().to_vec() });
        (!c.seed ^ 0x0123_4567_89AB_CDEF, SEED_WORD)
    } else {
        (c.seed, 0)
    };
    let a = args(&[
        ("inp", INP),
        ("out", OUT),
        (if key { "key" } else { "mask" }, SINK),
        ("len", c.len as u64),
        ("threshold", c.threshold() as u64),
        ("scale", c.scale().to_bits() as u64),
        ("seed", seed),
        ("seed_ptr", seed_ptr),
    ]);
    let ptx = text(if key { dropout_key_ptx() } else { dropout_ptx() });
    let mut mem = launch(&ptx, &a, global, c.len, order);
    if via_word {
        let word = mem.pop().unwrap();
        assert_eq!(word, words(&c.seed.to_le_bytes()), "{c:?}: the seed word is only read");
    }
    let sink = mem.pop().unwrap();
    let out = mem.pop().unwrap();
    (x, out, sink)
}

/// The regenerating backward over `key` (the key's words) for gradient `g`.
fn backward(c: Case, key: &[u32], g: &[u32], order: Order) -> Vec<u32> {
    let mut grad = g.to_vec();
    grad.extend(vec![0x3FC0_0000; TAIL]);
    let global = vec![
        Segment { base: SINK, bytes: le32(key) },
        Segment { base: GRAD, bytes: le32(&grad) },
        Segment { base: DX, bytes: le32(&vec![POISON; c.len + TAIL]) },
    ];
    let a = args(&[
        ("grad", GRAD),
        ("dx", DX),
        ("key", SINK),
        ("len", c.len as u64),
        ("threshold", c.threshold() as u64),
        ("scale", c.scale().to_bits() as u64),
    ]);
    let mem = launch(&text(dropout_bwd_ptx()), &a, global, c.len, order);
    assert_eq!(mem[1][..c.len], g[..c.len], "the gradient is only read");
    assert_eq!(mem[0], key, "the key is only read");
    mem[2].clone()
}

/// The stored-mask backward exactly as the compiler emits it on the normal
/// schedule: the fused chain `mul(i0,i1); mul(p0, f32(1/(1-p)))` over the
/// gradient and the mask, built by `ew_chain_ptx::emit`.
fn stored_mask_backward(c: Case, mask: &[u32], g: &[u32], order: Order) -> Vec<u32> {
    let sig = ChainSig {
        n_inputs: 2,
        steps: vec![
            ChainStep { op: EwOpcode::Mul, lhs: Operand::Input(0), rhs: Some(Operand::Input(1)) },
            ChainStep { op: EwOpcode::Mul, lhs: Operand::Prev(0), rhs: Some(Operand::Imm(c.scale().to_bits())) },
        ],
    };
    assert_eq!(sig.encode_name(), format!("fused_ew:v1:mul(i0,i1);mul(p0,x{:08x})", c.scale().to_bits()));
    let ptx = text(emit(&sig, "nsl_fused_ew_dropout_ref"));
    let global = vec![
        Segment { base: DX, bytes: le32(&vec![POISON; c.len + TAIL]) },
        Segment { base: GRAD, bytes: le32(&[g, &vec![0; TAIL][..]].concat()) },
        Segment { base: SINK, bytes: le32(&[mask, &vec![0; TAIL][..]].concat()) },
    ];
    let a = args(&[("out", DX), ("in0", GRAD), ("in1", SINK), ("n", c.len as u64)]);
    launch(&ptx, &a, global, c.len, order).remove(0)
}

fn same(got: u32, want: u32) -> bool {
    got == want || (f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan())
}

const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

#[test]
fn the_key_kernel_writes_the_mask_kernels_output_and_its_key() {
    for c in cases() {
        for order in ORDERS {
            let (x, out_mask, mask) = forward(c, false, order);
            let (_, out_key, key) = forward(c, true, order);
            assert_eq!(out_key, out_mask, "{c:?} {order:?}: the key kernel's output differs");
            // The mask kernel's mask is the hash's decision (as its own gate
            // checks); the key kernel's output is the same product.
            for i in 0..c.len {
                let keep = hash(c.seed.wrapping_add(i as u64)) < c.threshold();
                assert_eq!(mask[i], if keep { 1.0f32 } else { 0.0 }.to_bits(), "{c:?} i={i}");
                let want = (f32::from_bits(x[i]) * if keep { c.scale() } else { 0.0 }).to_bits();
                assert!(same(out_key[i], want), "{c:?} i={i}");
            }
            assert!(out_key[c.len..].iter().all(|&w| w == POISON), "{c:?}: out written past len");
            let want_key = [
                DROPOUT_KEY_MAGIC,
                c.seed as u32,
                (c.seed >> 32) as u32,
                c.threshold(),
                c.scale().to_bits(),
                c.len as u32,
                (c.len as u64 >> 32) as u32,
                0,
            ];
            assert_eq!(key[..DROPOUT_KEY_WORDS], want_key, "{c:?} {order:?}: the key");
            assert!(key[DROPOUT_KEY_WORDS..].iter().all(|&w| w == POISON), "{c:?}: written past the key");
        }
    }
}

/// Inside a `--cuda-graphs` region the base comes through a device word. Both
/// forwards then write what the by-value launch writes, byte for byte. That
/// includes the key's two base words, which the backward and the replay redraw
/// from, so the gradient is unchanged too.
#[test]
fn a_base_read_through_the_word_is_the_by_value_launch_bit_for_bit() {
    for c in cases() {
        for order in ORDERS {
            for key in [false, true] {
                let by_value = forward(c, key, order);
                let via_word = forward_base(c, key, order, true);
                assert_eq!(via_word, by_value, "{c:?} {order:?} key={key}");
            }
        }
    }
    // An empty launch reads the word too: its key holds the word's base.
    let c = Case { len: 0, p: 0.1, seed: (7 << 32) + 99 };
    let (_, _, key) = forward_base(c, true, Order::Ascending, true);
    assert_eq!(key[DropoutKeyWord::SeedLo as usize], 99);
    assert_eq!(key[DropoutKeyWord::SeedHi as usize], 7);
}

/// The runtime launches the key kernel with at least one block, so a launch
/// over an empty input still leaves a valid key.
#[test]
fn an_empty_launch_still_writes_its_key() {
    let c = Case { len: 0, p: 0.1, seed: 99 };
    let (_, out, key) = forward(c, true, Order::Ascending);
    assert!(out.iter().all(|&w| w == POISON));
    assert_eq!(key[DropoutKeyWord::Magic as usize], DROPOUT_KEY_MAGIC);
    assert_eq!(key[DropoutKeyWord::SeedLo as usize], 99);
    assert_eq!(key[DropoutKeyWord::LenLo as usize], 0);
}

#[test]
fn the_backward_is_the_stored_mask_backward_bit_for_bit() {
    let mut kept = 0usize;
    let mut dropped = 0usize;
    for c in cases() {
        for order in ORDERS {
            let (_, _, mask) = forward(c, false, order);
            let (_, _, key) = forward(c, true, order);
            let g = values(c.len, c.seed ^ 0xD00D);
            let want = stored_mask_backward(c, &mask[..c.len], &g, order);
            let got = backward(c, &key, &g, order);
            for i in 0..c.len {
                // Bit for bit, NaN payloads included: both kernels multiply
                // in the same order with the same rounding.
                assert_eq!(got[i], want[i], "{c:?} {order:?} i={i}: dy {:#010x}, mask {:#010x}", g[i], mask[i]);
                if mask[i] == 0 {
                    dropped += 1;
                } else {
                    kept += 1;
                }
            }
            assert!(got[c.len..].iter().all(|&w| w == POISON), "{c:?}: dx written past len");
        }
    }
    assert!(kept > 1000 && dropped > 300, "kept {kept}, dropped {dropped}");
}

#[test]
fn a_key_that_is_not_the_launchs_poisons_every_element() {
    let c = Case { len: 600, p: 0.1, seed: (3 << 32) + 5 };
    let (_, _, key) = forward(c, true, Order::Ascending);
    let g = values(c.len, 7);
    let finite: Vec<usize> = (0..c.len).filter(|&i| f32::from_bits(g[i]).is_finite()).collect();
    let good = backward(c, &key, &g, Order::Ascending);
    assert!(finite.iter().all(|&i| !f32::from_bits(good[i]).is_nan()), "the launch's own key poisons nothing");

    let mut bad_keys: Vec<(&str, Vec<u32>)> = Vec::new();
    for (what, w, v) in [
        ("magic", DropoutKeyWord::Magic, 0x4B4F_5245),
        ("threshold", DropoutKeyWord::Threshold, c.threshold() - 1),
        ("scale", DropoutKeyWord::ScaleBits, c.scale().to_bits() ^ 1),
        ("length", DropoutKeyWord::LenLo, c.len as u32 + 1),
        ("length high word", DropoutKeyWord::LenHi, 1),
    ] {
        let mut k = key.clone();
        k[w as usize] = v;
        bad_keys.push((what, k));
    }
    // What could stand in a key's place by mistake: a shape stub's zeros, an
    // f32 mask's words.
    bad_keys.push(("zeros", vec![0; key.len()]));
    bad_keys.push(("an f32 mask", vec![1.0f32.to_bits(); key.len()]));
    for (what, k) in bad_keys {
        let dx = backward(c, &k, &g, Order::Ascending);
        assert!(
            dx[..c.len].iter().all(|&w| f32::from_bits(w).is_nan()),
            "a key with a wrong {what} must poison every element"
        );
    }
}

/// The mask comes from the key's counter base: two keys, two masks.
#[test]
fn the_backward_draws_from_the_keys_counter_base() {
    let a = Case { len: 1000, p: 0.5, seed: 1 << 40 };
    let b = Case { seed: (1 << 40) + 1000, ..a };
    let (_, _, key_a) = forward(a, true, Order::Ascending);
    let (_, _, key_b) = forward(b, true, Order::Ascending);
    let ones = vec![1.0f32.to_bits(); a.len];
    let (da, db) = (backward(a, &key_a, &ones, Order::Ascending), backward(b, &key_b, &ones, Order::Ascending));
    let agree = da[..a.len].iter().zip(&db[..a.len]).filter(|(x, y)| x == y).count();
    // Independent masks at keep probability 1/2 agree on 500 +- 16 of 1000.
    assert!((405..=595).contains(&agree), "masks of adjacent counter windows agree on {agree} of 1000");
}

/// The replay of a forward (`nsl_dropout_apply_f32`) over its input and the
/// key the forward wrote.
fn replay(c: Case, key: &[u32], x: &[u32], order: Order) -> Vec<u32> {
    let mut inp = x.to_vec();
    inp.extend(vec![0x3FC0_0000; TAIL]);
    let global = vec![
        Segment { base: SINK, bytes: le32(key) },
        Segment { base: INP, bytes: le32(&inp) },
        Segment { base: OUT, bytes: le32(&vec![POISON; c.len + TAIL]) },
    ];
    let a = args(&[
        ("inp", INP),
        ("out", OUT),
        ("key", SINK),
        ("len", c.len as u64),
        ("threshold", c.threshold() as u64),
        ("scale", c.scale().to_bits() as u64),
    ]);
    let mem = launch(&text(dropout_apply_ptx()), &a, global, c.len, order);
    assert_eq!(mem[0], key, "the key is only read");
    assert_eq!(mem[1][..c.len], x[..c.len], "the input is only read");
    mem[2].clone()
}

/// A checkpointed backward may recompute a dropout's output instead of
/// saving it: the replay must be the forward's output byte for byte, and a
/// key that is not the launch's poisons it as it poisons the backward.
#[test]
fn the_replay_is_the_forwards_output_bit_for_bit() {
    for c in cases() {
        for order in ORDERS {
            let (x, out, key) = forward(c, true, order);
            let again = replay(c, &key, &x, order);
            assert_eq!(again[..c.len], out[..c.len], "{c:?} {order:?}: the replay differs");
            assert!(again[c.len..].iter().all(|&w| w == POISON), "{c:?}: written past len");
        }
    }
    let c = Case { len: 300, p: 0.5, seed: 77 };
    let (x, _, key) = forward(c, true, Order::Ascending);
    let mut bad = key.clone();
    bad[DropoutKeyWord::LenLo as usize] += 1;
    let r = replay(c, &bad, &x, Order::Ascending);
    assert!(r[..c.len].iter().all(|&w| f32::from_bits(w).is_nan()), "a wrong key must poison the replay");
}
