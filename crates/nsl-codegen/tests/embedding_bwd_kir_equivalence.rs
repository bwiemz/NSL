//! The differential equivalence gate for the embedding backward kernels
//! from `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5): the atomic
//! pair `nsl_embedding_bwd_{f32,i32idx}` and the deterministic pair
//! `nsl_embedding_bwd_det_{f32,i32idx}`, now built by
//! `nsl_kir::kernels::embedding_bwd`.
//!
//! This file runs the frozen hand modules
//! (`tests/fixtures/embedding_bwd_hand.rs`) and the KIR ones side by side on
//! the cooperative-CTA interpreter's two-dimensional blocks. The runtime
//! launches a 16 × 16 block; the gate also runs an 8 × 4 one, where `%ntid.x`
//! and `%ntid.y` differ. Each launch covers its grid with one block more
//! than it needs in each direction.
//!
//! 1. **Agreement**: under two schedules, the two kernels leave *the same
//!    bytes* in all of global memory. The output starts non-zero, so an add
//!    is told from a store. The indices include rows hit several times,
//!    rows never hit, and indices that are skipped (negative, at or past
//!    `vocab`) or that land on row 0 (NaN, -0.5).
//! 2. **Correctness**: the atomic kernels leave `init + Σ grad`, summed in
//!    the order the interpreter's schedule runs the threads, which is
//!    increasing position. The deterministic kernels leave `Σ grad` in
//!    increasing position, from 0. Nothing past the output is written.
//! 3. **The gate bites**: each bound, both block indices, `%tid.y`,
//!    `%ntid.y`, every element size, every 64-bit add and stride, the
//!    negative-index guard, the f32 truncation, the atomic add, the
//!    deterministic loop's match test, step, start and accumulate are each
//!    caught. Two mutants are named as equivalent: the i32 index read
//!    zero-extended, and the atomic kernels' negative guard removed (the
//!    unsigned vocab bound after it rejects a negative index too).

use std::collections::HashMap;

use nsl_kir::kernels::embedding_bwd::{ptx, EmbeddingBwd};
use nsl_kir::kernels::lookup::{IndexDtype, LOOKUP_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/embedding_bwd_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

use EmbeddingBwd::{Atomic, Deterministic};

const TAIL: usize = 128;
const POISON: u32 = 0x7FA5_A5A5;
const GRAD: u64 = 0x1000_0000;
const INDICES: u64 = 0x2000_0000;
const OUT: u64 = 0x3000_0000;

#[derive(Clone, Copy, Debug, PartialEq)]
struct K(EmbeddingBwd, IndexDtype);

const ALL: [K; 4] = [
    K(Atomic, IndexDtype::F32),
    K(Atomic, IndexDtype::I32),
    K(Deterministic, IndexDtype::F32),
    K(Deterministic, IndexDtype::I32),
];

fn kir_ptx(k: K) -> String {
    String::from_utf8(ptx(k.0, k.1)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(k: K) -> String {
    match k {
        K(Atomic, IndexDtype::F32) => hand::EMBEDDING_BWD_F32_PTX,
        K(Atomic, IndexDtype::I32) => hand::EMBEDDING_BWD_I32IDX_PTX,
        K(Deterministic, IndexDtype::F32) => hand::EMBEDDING_BWD_DET_F32_PTX,
        K(Deterministic, IndexDtype::I32) => hand::EMBEDDING_BWD_DET_I32IDX_PTX,
    }
    .trim_end_matches('\0')
    .to_string()
}

fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    *s >> 33
}

fn values(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9) | 1;
    (0..n).map(|_| ((lcg(&mut s) as f32 / (1u64 << 31) as f32) * 4.0 - 2.0).to_bits()).collect()
}

/// `(seq_len, embed_dim, vocab)`: ragged against both blocks, and with more
/// positions than rows so rows repeat.
const SHAPES: [(usize, usize, usize); 5] = [(1, 1, 1), (17, 3, 5), (40, 17, 9), (9, 33, 40), (70, 5, 3)];

const BLOCKS: [[u32; 2]; 2] = [LOOKUP_BLOCK, [8, 4]];

const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

/// The index words: in range (with repeats), and the special cases.
fn indices(idx: IndexDtype, seq: usize, vocab: usize, seed: u64) -> Vec<u32> {
    let v = vocab as f32;
    let f_special = [0.0, v - 1.0, 1.7, -1.5, -0.5, f32::NAN, v, v + 3.0, 1e30, f32::NEG_INFINITY, f32::INFINITY, -0.0];
    let vi = vocab as i32;
    let i_special = [0, vi - 1, -1, vi, vi + 7, i32::MIN, i32::MAX];
    let mut s = seed;
    (0..seq)
        .map(|n| {
            let r = lcg(&mut s) % vocab as u64;
            match idx {
                IndexDtype::F32 => {
                    if n % 3 == 1 { f_special[(n / 3 + seed as usize) % f_special.len()] } else { r as f32 }.to_bits()
                }
                IndexDtype::I32 => (if n % 3 == 1 { i_special[(n / 3 + seed as usize) % i_special.len()] } else { r as i32 }) as u32,
            }
        })
        .collect()
}

/// The row an index names, or `None` when the kernels skip it.
fn row(idx: IndexDtype, word: u32, vocab: usize) -> Option<usize> {
    let r: i64 = match idx {
        // `cvt.rzi.s64.f32`: toward zero, saturating, NaN to 0.
        IndexDtype::F32 => f32::from_bits(word) as i64,
        IndexDtype::I32 => word as i32 as i64,
    };
    (0..vocab as i64).contains(&r).then_some(r as usize)
}

struct Run {
    grad: Vec<u32>,
    idx: Vec<u32>,
    init: Vec<u32>,
    mem: Vec<Vec<u8>>,
}

fn run(k: K, text: &str, (seq, embed, vocab): (usize, usize, usize), block: [u32; 2], seed: u64, order: Order) -> Run {
    let grad = values(seq * embed, seed);
    let idx = indices(k.1, seq, vocab, seed);
    let init = values(vocab * embed, seed + 99);
    let prog = parse(text);
    let mut global = vec![
        Segment { base: GRAD, bytes: le32(&grad) },
        Segment { base: INDICES, bytes: le32(&idx) },
        Segment { base: OUT, bytes: le32(&[init.clone(), vec![POISON; TAIL]].concat()) },
    ];
    let args: HashMap<String, u64> = EmbeddingBwd::PARAMS
        .iter()
        .zip([GRAD, INDICES, OUT, seq as u64, embed as u64, vocab as u64])
        .map(|(n, v)| (n.to_string(), v))
        .collect();
    let across = if k.0 == Atomic { seq } else { vocab };
    let grid_x = across.div_ceil(block[0] as usize) as u32 + 1;
    let grid_y = embed.div_ceil(block[1] as usize) as u32 + 1;
    let mut ctas: Vec<(u32, u32)> = (0..grid_y).flat_map(|y| (0..grid_x).map(move |x| (x, y))).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for (x, y) in ctas {
        let mut l = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![],
            ctaid: x,
            ctaid_y: y,
            nctaid_y: grid_y,
            ntid: block[0],
            steps: 0,
        };
        run_cta_2d(&mut l, block[1], order);
    }
    Run { grad, idx, init, mem: global.into_iter().map(|s| s.bytes).collect() }
}

// ---------------------------------------------------------------------------
// Agreement and the formulas
// ---------------------------------------------------------------------------

#[test]
fn the_kernels_agree() {
    for k in ALL {
        let (hand, kir) = (hand_ptx(k), kir_ptx(k));
        for (n, &shape) in SHAPES.iter().enumerate() {
            for block in BLOCKS {
                for order in ORDERS {
                    let h = run(k, &hand, shape, block, n as u64 + 1, order).mem;
                    let q = run(k, &kir, shape, block, n as u64 + 1, order).mem;
                    assert!(h == q, "{k:?} {shape:?} {block:?} {order:?}: global memory differs");
                }
            }
        }
    }
}

#[test]
fn the_kernels_are_the_scatter() {
    for k in ALL {
        for (n, &(seq, embed, vocab)) in SHAPES.iter().enumerate() {
            let r = run(k, &kir_ptx(k), (seq, embed, vocab), LOOKUP_BLOCK, n as u64 + 1, Order::Ascending);
            let out = words(&r.mem[2]);
            for v in 0..vocab {
                for j in 0..embed {
                    let start = if k.0 == Atomic { f32::from_bits(r.init[v * embed + j]) } else { 0.0 };
                    let want = (0..seq)
                        .filter(|&i| row(k.1, r.idx[i], vocab) == Some(v))
                        .fold(start, |s, i| s + f32::from_bits(r.grad[i * embed + j]));
                    assert_eq!(out[v * embed + j], want.to_bits(), "{k:?} ({v}, {j})");
                }
            }
            assert!(out[vocab * embed..].iter().all(|&w| w == POISON), "{k:?}: wrote past the output");
        }
    }
}

/// The data reaches every branch: skipped indices, repeated rows, and rows
/// no index names.
#[test]
fn the_data_takes_every_branch() {
    for idx in IndexDtype::ALL {
        let (seq, _, vocab) = SHAPES[2];
        let words = indices(idx, seq, vocab, 3);
        let rows: Vec<Option<usize>> = words.iter().map(|&w| row(idx, w, vocab)).collect();
        assert!(rows.iter().filter(|r| r.is_none()).count() >= 3, "{idx:?}: skips");
        let hits = |v| rows.iter().filter(|r| **r == Some(v)).count();
        assert!((0..vocab).any(|v| hits(v) >= 2), "{idx:?}: a repeated row");
        let (seq, _, vocab) = SHAPES[3];
        let rows: Vec<Option<usize>> = indices(idx, seq, vocab, 4).iter().map(|&w| row(idx, w, vocab)).collect();
        assert!((0..vocab).any(|v| !rows.contains(&Some(v))), "{idx:?}: a row never hit");
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

fn caught(k: K, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(k);
    let mutant = mutate(&kir_ptx(k));
    SHAPES.iter().enumerate().any(|(n, &shape)| {
        BLOCKS.into_iter().any(|block| {
            ORDERS.into_iter().any(|order| {
                let expect = run(k, &hand, shape, block, n as u64 + 1, order).mem;
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(k, &mutant, shape, block, n as u64 + 1, order))) {
                    Ok(r) => r.mem != expect,
                    Err(_) => true,
                }
            })
        })
    })
}

fn edit(ptx: &str, pick: impl Fn(&str) -> bool, i: usize, f: impl Fn(&str) -> String) -> String {
    let at = ptx.lines().enumerate().filter(|(_, l)| pick(l)).nth(i).unwrap_or_else(|| panic!("no line {i}")).0;
    ptx.lines().enumerate().map(|(n, l)| if n == at { f(l) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
}

fn nudge(ptx: &str, from: &str, to: &str, i: usize) -> String {
    edit(ptx, |l| l.contains(from), i, |l| l.replacen(from, to, 1))
}

fn count(k: K, pick: impl Fn(&str) -> bool) -> usize {
    kir_ptx(k).lines().filter(|l| pick(l)).count()
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for k in ALL {
        assert!(!caught(k, |p| p.to_string()), "{k:?}");
    }
}

/// The two grid bounds and the vocab bound (atomic) or the loop bound
/// (deterministic).
#[test]
fn relaxing_a_bound_is_caught() {
    for k in ALL {
        assert_eq!(count(k, |l| l.contains("setp.ge.u64 ")), 3, "{k:?}");
        for i in 0..3 {
            assert!(caught(k, |p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "{k:?} bound {i}");
        }
    }
}

#[test]
fn the_two_dimensional_index_is_caught() {
    for k in ALL {
        assert!(caught(k, |p| p.replacen("%ctaid.x;", "0;", 1)), "{k:?} ctaid.x");
        assert!(caught(k, |p| p.replacen("%ctaid.y;", "0;", 1)), "{k:?} ctaid.y");
        assert!(caught(k, |p| p.replacen("%tid.y;", "%tid.x;", 1)), "{k:?} tid.y");
        assert!(caught(k, |p| p.replacen("%ntid.y;", "%ntid.x;", 1)), "{k:?} ntid.y");
    }
}

/// Every address scaling (the index, grad and out reads and writes), every
/// 64-bit add and every row stride.
#[test]
fn the_address_arithmetic_is_caught() {
    for k in ALL {
        let is_size = |l: &str| l.contains("mul.lo.u64") && l.ends_with(", 4;");
        let sizes = count(k, is_size);
        assert_eq!(sizes, 3, "{k:?}");
        for i in 0..sizes {
            assert!(caught(k, |p| edit(p, is_size, i, |l| l.replacen(", 4;", ", 8;", 1))), "{k:?} size {i}");
        }
        let is_stride = |l: &str| l.contains("mul.lo.u64") && !l.ends_with(", 4;");
        let strides = count(k, is_stride);
        assert_eq!(strides, 2, "{k:?}: the grad row and the out row");
        for i in 0..strides {
            let m = |p: &str| {
                edit(p, is_stride, i, |l| {
                    let (head, ops) = l.split_once("mul.lo.u64 ").expect("mul");
                    let o: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
                    format!("{head}mov.u64 {}, {};", o[0], o[1])
                })
            };
            assert!(caught(k, m), "{k:?} stride {i}");
        }
        // The column and base adds; the deterministic loop's step is its own
        // test below.
        let is_add = |l: &str| l.contains("add.u64 ") && !l.contains(", 1;");
        let adds = count(k, is_add);
        for i in 0..adds {
            let m = |p: &str| {
                edit(p, is_add, i, |l| {
                    let (head, ops) = l.split_once("add.u64 ").expect("add");
                    let o: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
                    format!("{head}mov.u64 {}, {};", o[0], o[1])
                })
            };
            assert!(caught(k, m), "{k:?} add {i}");
        }
    }
}

/// The atomic kernels: the negative guard's boundary and the add itself.
///
/// The guard removed outright is an equivalent mutant, named here: the
/// vocab bound that follows compares the index's bits unsigned, and a
/// negative index read that way is at least 2^63, past every vocab. The
/// guard stays for parity with the hand kernels, which bound `vocab`
/// signed and need it.
#[test]
fn the_atomic_scatter_is_caught() {
    for idx in IndexDtype::ALL {
        let k = K(Atomic, idx);
        assert_eq!(count(k, |l| l.contains("setp.lt.s64 ")), 1, "{k:?}");
        assert!(caught(k, |p| nudge(p, "setp.lt.s64 ", "setp.le.s64 ", 0)), "{k:?} negative guard boundary");
        let never = |p: &str| {
            edit(p, |l| l.contains("setp.lt.s64 "), 0, |l| {
                let (head, ops) = l.split_once("setp.lt.s64 ").expect("setp");
                let o: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
                format!("{head}setp.lt.s64 {}, {}, {};", o[0], o[1], o[1])
            })
        };
        assert!(!caught(k, never), "{k:?}: the guard removed is a named equivalent mutant");
        assert!(caught(k, |p| p.replacen("red.global.add.f32 ", "st.global.f32 ", 1)), "{k:?} add, not store");
    }
}

/// The deterministic kernels: the match test, the loop's step and start,
/// and the accumulate.
#[test]
fn the_deterministic_walk_is_caught() {
    for idx in IndexDtype::ALL {
        let k = K(Deterministic, idx);
        assert!(caught(k, |p| nudge(p, "setp.ne.u64 ", "setp.eq.u64 ", 0)), "{k:?} match");
        assert!(caught(k, |p| nudge(p, "add.f32 ", "sub.f32 ", 0)), "{k:?} accumulate");
        let step = |l: &str| l.contains("mov.u64 ") && l.ends_with(", 1;");
        assert_eq!(count(k, step), 1, "{k:?}");
        assert!(caught(k, |p| edit(p, step, 0, |l| l.replacen(", 1;", ", 2;", 1))), "{k:?} step");
        assert!(caught(k, |p| p.replacen("0f00000000;", "0f3F800000;", 1)), "{k:?} start");
    }
}

/// The f32 index truncates signed: read unsigned (a negative index
/// saturating to row 0) it is caught. The i32 index read zero-extended is
/// not. That mutant is equivalent: a negative i32 widens to at least 2^31
/// either way, so the atomic kernels skip it (as negative, or as past
/// `vocab`) and the deterministic ones never match it.
#[test]
fn the_index_widening_is_pinned() {
    for op in EmbeddingBwd::ALL {
        let f = K(op, IndexDtype::F32);
        assert!(caught(f, |p| p.replacen("cvt.rzi.s64.f32 ", "cvt.rzi.u64.f32 ", 1)), "{op:?} signed truncation");
        let i = K(op, IndexDtype::I32);
        assert_eq!(kir_ptx(i).matches("cvt.s64.s32 ").count(), 1);
        assert!(!caught(i, |p| p.replacen("cvt.s64.s32 ", "cvt.u64.u32 ", 1)), "{op:?}: named equivalent mutant");
    }
}
