//! The differential equivalence gate for the deterministic scatter-add from
//! `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5):
//! `nsl_det_scatter_add_f32`, now built by `nsl_kir::kernels::det_scatter`.
//!
//! This file runs the frozen hand module (`tests/fixtures/det_scatter_hand.rs`)
//! and the KIR one side by side on the cooperative-CTA interpreter's
//! two-dimensional blocks. The runtime launches a 16 × 16 block; the gate
//! also runs an 8 × 4 one, where `%ntid.x` and `%ntid.y` differ. Each launch
//! covers its `(vocab_size, embed_dim)` grid with one block more than it
//! needs in each direction, under two schedules.
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory. The indices hit some rows several times and others never. They
//!    also include values the conversion bends: fractions, `-0.5`, `-1.5`,
//!    NaN and `-inf` (all row 0 under `cvt.rzi.u64.f32`), and `vocab`,
//!    `1e30` and `+inf` (no row).
//! 2. **Correctness**: `out[row, col]` is `input[row, col]` plus
//!    `src[i, col]` over every `i` with `u64(idx(i)) == row`, in increasing
//!    `i`, bit for bit. Nothing past the output is written, and the inputs
//!    are untouched.
//! 3. **The gate bites**:
//!    - each bound, both block indices, `%tid.y` and `%ntid.y`;
//!    - every element size, row stride and 64-bit add;
//!    - the match test, the loop's step, the start from `input`, and the
//!      accumulate;
//!    - the conversion read as signed, which stops negative indices landing
//!      on row 0;
//!    - every pointer.

use std::collections::HashMap;

use nsl_kir::kernels::det_scatter::{ptx, DET_SCATTER_BLOCK, PARAM_NAMES};

#[allow(dead_code)]
#[path = "fixtures/det_scatter_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 128;
const POISON: u32 = 0x7FA5_A5A5;
const SRC: u64 = 0x1000_0000;
const INDICES: u64 = 0x2000_0000;
const INPUT: u64 = 0x3000_0000;
const OUT: u64 = 0x4000_0000;

fn kir_ptx() -> String {
    String::from_utf8(ptx()).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx() -> String {
    hand::DET_SCATTER_ADD_F32_PTX.trim_end_matches('\0').to_string()
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

/// Values with full significands in `[-2, 2)`: sums that depend on order.
fn values(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9) | 1;
    (0..n).map(|_| ((lcg(&mut s) as f32 / (1u64 << 31) as f32) * 4.0 - 2.0).to_bits()).collect()
}

/// `(num_indices, embed_dim, vocab_size)`: ragged against both blocks, and
/// with more positions than rows so rows repeat.
const SHAPES: [(usize, usize, usize); 6] = [(0, 3, 5), (1, 1, 1), (17, 3, 5), (40, 17, 9), (9, 33, 40), (70, 5, 3)];

const BLOCKS: [[u32; 2]; 2] = [DET_SCATTER_BLOCK, [8, 4]];

const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

/// The index words: in range (with repeats), and every third one special.
fn indices(n: usize, vocab: usize, seed: u64) -> Vec<u32> {
    let v = vocab as f32;
    let special = [0.0, v - 1.0, 1.7, -1.5, -0.5, f32::NAN, v, v + 3.0, 1e30, f32::NEG_INFINITY, f32::INFINITY, -0.0];
    let mut s = seed;
    (0..n)
        .map(|k| {
            let r = lcg(&mut s) % vocab as u64;
            if k % 3 == 1 { special[(k / 3 + seed as usize) % special.len()] } else { r as f32 }.to_bits()
        })
        .collect()
}

/// The row an index word names: `cvt.rzi.u64.f32`, which truncates toward
/// zero and saturates (negative and NaN to 0), as Rust's `as u64` does.
fn row(word: u32) -> u64 {
    f32::from_bits(word) as u64
}

struct Run {
    src: Vec<u32>,
    idx: Vec<u32>,
    input: Vec<u32>,
    mem: Vec<Vec<u8>>,
}

fn run(text: &str, (n, embed, vocab): (usize, usize, usize), block: [u32; 2], seed: u64, order: Order) -> Run {
    let src = values(n * embed, seed);
    let idx = indices(n, vocab, seed);
    let input = values(vocab * embed, seed + 99);
    let prog = parse(text);
    let pad = |mut b: Vec<u8>| {
        b.resize(b.len().max(4), 0);
        b
    };
    let mut global = vec![
        Segment { base: SRC, bytes: pad(le32(&src)) },
        Segment { base: INDICES, bytes: pad(le32(&idx)) },
        Segment { base: INPUT, bytes: le32(&input) },
        Segment { base: OUT, bytes: le32(&vec![POISON; vocab * embed + TAIL]) },
    ];
    let args: HashMap<String, u64> = PARAM_NAMES
        .iter()
        .zip([SRC, INDICES, INPUT, OUT, n as u64, embed as u64, vocab as u64])
        .map(|(p, v)| (p.to_string(), v))
        .collect();
    let grid_x = vocab.div_ceil(block[0] as usize) as u32 + 1;
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
            nctaid_x: 0,
            nctaid_y: grid_y,
            ntid: block[0],
            steps: 0,
        };
        run_cta_2d(&mut l, block[1], order);
    }
    Run { src, idx, input, mem: global.into_iter().map(|s| s.bytes).collect() }
}

// ---------------------------------------------------------------------------
// Agreement and the formula
// ---------------------------------------------------------------------------

#[test]
fn the_kernels_agree() {
    let (hand, kir) = (hand_ptx(), kir_ptx());
    for (n, &shape) in SHAPES.iter().enumerate() {
        for block in BLOCKS {
            for order in ORDERS {
                let h = run(&hand, shape, block, n as u64 + 1, order).mem;
                let q = run(&kir, shape, block, n as u64 + 1, order).mem;
                assert!(h == q, "{shape:?} {block:?} {order:?}: global memory differs");
            }
        }
    }
}

#[test]
fn the_kernels_are_the_ordered_scatter() {
    for which in [hand_ptx(), kir_ptx()] {
        for (n, &(len, embed, vocab)) in SHAPES.iter().enumerate() {
            let r = run(&which, (len, embed, vocab), DET_SCATTER_BLOCK, n as u64 + 1, Order::Ascending);
            assert_eq!(words(&r.mem[0][..len * embed * 4]), r.src, "src untouched");
            assert_eq!(words(&r.mem[2]), r.input, "input untouched");
            let out = words(&r.mem[3]);
            for v in 0..vocab {
                for j in 0..embed {
                    let want = (0..len)
                        .filter(|&i| row(r.idx[i]) == v as u64)
                        .fold(f32::from_bits(r.input[v * embed + j]), |s, i| s + f32::from_bits(r.src[i * embed + j]));
                    assert_eq!(out[v * embed + j], want.to_bits(), "{:?} ({v}, {j})", (len, embed, vocab));
                }
            }
            assert!(out[vocab * embed..].iter().all(|&w| w == POISON), "wrote past the output");
        }
    }
}

/// The data reaches every branch: indices that land on row 0 only because
/// the conversion saturates, indices that name no row, repeated rows, and
/// rows no index names.
#[test]
fn the_data_takes_every_branch() {
    let (n, _, vocab) = SHAPES[3];
    let words = indices(n, vocab, 4);
    let bent = words.iter().filter(|&&w| {
        let f = f32::from_bits(w);
        (f.is_nan() || f < 0.0) && row(w) == 0 && f.to_bits() != (-0.0f32).to_bits()
    });
    assert!(bent.count() >= 2, "negative or NaN indices on row 0");
    assert!(words.iter().any(|&w| row(w) >= vocab as u64), "an index naming no row");
    let hits = |v: u64| words.iter().filter(|&&w| row(w) == v).count();
    assert!((0..vocab as u64).any(|v| hits(v) >= 2), "a repeated row");
    let (n, _, vocab) = SHAPES[4];
    let words = indices(n, vocab, 5);
    assert!((0..vocab as u64).any(|v| !words.iter().any(|&w| row(w) == v)), "a row never hit");
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

fn caught(mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx();
    let mutant = mutate(&kir_ptx());
    SHAPES.iter().enumerate().any(|(n, &shape)| {
        BLOCKS.into_iter().any(|block| {
            ORDERS.into_iter().any(|order| {
                let expect = run(&hand, shape, block, n as u64 + 1, order).mem;
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mutant, shape, block, n as u64 + 1, order))) {
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

fn count(pick: impl Fn(&str) -> bool) -> usize {
    kir_ptx().lines().filter(|l| pick(l)).count()
}

/// `op d, a, b;` as `mov.<ty> d, a;`.
fn drop_second(l: &str, op: &str, ty: &str) -> String {
    let (head, ops) = l.split_once(op).expect("the op");
    let o: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
    format!("{head}mov.{ty} {}, {};", o[0], o[1])
}

#[test]
fn the_unmutated_kernel_is_not_caught() {
    assert!(!caught(|p| p.to_string()));
}

/// The two grid bounds and the loop bound.
#[test]
fn relaxing_a_bound_is_caught() {
    assert_eq!(count(|l| l.contains("setp.ge.u64 ")), 3);
    for i in 0..3 {
        assert!(caught(|p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "bound {i}");
    }
}

#[test]
fn the_two_dimensional_index_is_caught() {
    assert!(caught(|p| p.replacen("%ctaid.x;", "0;", 1)), "ctaid.x");
    assert!(caught(|p| p.replacen("%ctaid.y;", "0;", 1)), "ctaid.y");
    assert!(caught(|p| p.replacen("%tid.y;", "%tid.x;", 1)), "tid.y");
    assert!(caught(|p| p.replacen("%ntid.y;", "%ntid.x;", 1)), "ntid.y");
}

/// Every element size (the index, src, input and out accesses), both row
/// strides, and every 64-bit add but the loop's step.
#[test]
fn the_address_arithmetic_is_caught() {
    let is_size = |l: &str| l.contains("mul.lo.u64") && l.ends_with(", 4;");
    assert_eq!(count(is_size), 4, "index, input, src, out");
    for i in 0..4 {
        assert!(caught(|p| edit(p, is_size, i, |l| l.replacen(", 4;", ", 8;", 1))), "size {i}");
    }
    let is_stride = |l: &str| l.contains("mul.lo.u64") && !l.ends_with(", 4;");
    assert_eq!(count(is_stride), 2, "the output row and the src row");
    for i in 0..2 {
        assert!(caught(|p| edit(p, is_stride, i, |l| drop_second(l, "mul.lo.u64 ", "u64"))), "stride {i}");
    }
    let is_add = |l: &str| l.contains("add.u64 ") && !l.contains(", 1;");
    for i in 0..count(is_add) {
        assert!(caught(|p| edit(p, is_add, i, |l| drop_second(l, "add.u64 ", "u64"))), "add {i}");
    }
}

/// The match test, the loop's step, the start from `input`, and the
/// accumulate.
#[test]
fn the_walk_is_caught() {
    assert!(caught(|p| nudge(p, "setp.ne.u64 ", "setp.eq.u64 ", 0)), "match");
    assert!(caught(|p| nudge(p, "add.rn.f32 ", "sub.rn.f32 ", 0)), "accumulate");
    assert!(caught(|p| edit(p, |l| l.contains("add.rn.f32 "), 0, |l| drop_second(l, "add.rn.f32 ", "f32"))), "accumulate dropped");
    let step = |l: &str| l.contains("mov.u64 ") && l.ends_with(", 1;");
    assert_eq!(count(step), 1);
    assert!(caught(|p| edit(p, step, 0, |l| l.replacen(", 1;", ", 2;", 1))), "step");
    assert!(caught(|p| p.replacen("[param_input]", "[param_out]", 1)), "start from input");
}

/// The index converts unsigned and saturating: read signed, a negative
/// index wraps past every row instead of landing on row 0.
#[test]
fn the_index_conversion_is_pinned() {
    assert_eq!(kir_ptx().matches("cvt.rzi.u64.f32 ").count(), 1);
    assert!(caught(|p| p.replacen("cvt.rzi.u64.f32 ", "cvt.rzi.s64.f32 ", 1)), "signed");
}

/// Every pointer: each parameter read as its neighbour.
#[test]
fn every_pointer_is_pinned() {
    for i in 0..4 {
        let (a, b) = (PARAM_NAMES[i], PARAM_NAMES[(i + 1) % 4]);
        assert!(caught(|p| p.replacen(&format!("[param_{a}]"), &format!("[param_{b}]"), 1)), "{a} as {b}");
    }
}
