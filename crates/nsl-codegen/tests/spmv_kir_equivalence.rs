//! The differential equivalence gate for the sparse matrix-vector kernels
//! from `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5):
//! `nsl_csr_spmv_f32` and `nsl_coo_spmv_f32`, now built by
//! `nsl_kir::kernels::spmv`.
//!
//! This file runs the frozen hand modules (`tests/fixtures/spmv_hand.rs`)
//! and the KIR ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), on the runtime's 256-thread block
//! and on a 32-thread one, with one block more than the work needs, under
//! two schedules:
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory, over empty rows, a single nonzero, ragged rows and repeated
//!    COO rows, with general f32 data.
//! 2. **Correctness**: on data whose products and sums are exact, CSR row
//!    `r` is `fma`-accumulated from `+0.0` over its nonzeros in order, and
//!    COO adds every nonzero's product to the `y` it started with (an add,
//!    not a store); nothing past `y` is written.
//! 3. **The gate bites**: each bound, the block index, every element size,
//!    every 64-bit add, the CSR row's `+ 1` and step, the accumulator's
//!    start, the fused multiply-add, the COO product and the atomic add are
//!    caught.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;
use nsl_kir::kernels::spmv::{ptx, SpmvFormat};

#[allow(dead_code)]
#[path = "fixtures/spmv_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 64;
const POISON: u32 = 0x7FA5_A5A5;
const ROWS: u64 = 0x1000_0000;
const COLS: u64 = 0x2000_0000;
const VALUES: u64 = 0x3000_0000;
const X: u64 = 0x4000_0000;
const Y: u64 = 0x5000_0000;

fn kir_ptx(f: SpmvFormat) -> String {
    String::from_utf8(ptx(f)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(f: SpmvFormat) -> String {
    match f {
        SpmvFormat::Csr => hand::CSR_SPMV_F32_PTX,
        SpmvFormat::Coo => hand::COO_SPMV_F32_PTX,
    }
    .trim_end_matches('\0')
    .to_string()
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

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    *s >> 33
}

/// A sparse `m × k` matrix as `(row, col, value)` triples in row order.
/// Rows get 0 to 12 nonzeros (so some are empty and some are long), and a
/// row may name a column twice.
struct Case {
    m: usize,
    k: usize,
    nz: Vec<(usize, usize, u32)>,
    x: Vec<u32>,
    y0: Vec<u32>,
}

/// `exact`: small integers, so every product and sum is exact and the
/// result does not depend on the order of the adds.
fn case(m: usize, k: usize, seed: u64, exact: bool) -> Case {
    let mut s = seed;
    let val = |s: &mut u64| -> u32 {
        if exact {
            ((lcg(s) % 7) as f32 - 3.0).to_bits()
        } else {
            let e = (lcg(s) % 17) as i32 - 8;
            let mant = 1.0 + (lcg(s) % 1024) as f32 / 1024.0;
            let sign = if lcg(s).is_multiple_of(2) { 1.0 } else { -1.0 };
            (sign * mant * 2f32.powi(e)).to_bits()
        }
    };
    let mut nz = vec![];
    for r in 0..m {
        let n = (lcg(&mut s) % 13) as usize;
        let n = if r % 5 == 2 { 0 } else { n };
        for _ in 0..n {
            let c = (lcg(&mut s) % k as u64) as usize;
            nz.push((r, c, val(&mut s)));
        }
    }
    let x = (0..k).map(|_| val(&mut s)).collect();
    let y0 = (0..m).map(|_| val(&mut s)).collect();
    Case { m, k, nz, x, y0 }
}

/// `(m, k)`, plus a single-nonzero case built by hand.
const SHAPES: [(usize, usize); 4] = [(1, 1), (7, 5), (40, 13), (300, 37)];

const BLOCKS: [u32; 2] = [ELEMENTWISE_BLOCK, 32];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

fn cases(exact: bool) -> Vec<Case> {
    let mut v: Vec<Case> = SHAPES.iter().enumerate().map(|(n, &(m, k))| case(m, k, n as u64 + 1, exact)).collect();
    v.push(Case { m: 3, k: 2, nz: vec![(1, 1, 2.5f32.to_bits())], x: vec![1.0f32.to_bits(), (-3.0f32).to_bits()], y0: vec![0; 3] });
    v
}

fn run(f: SpmvFormat, ptx: &str, c: &Case, block: u32, order: Order) -> Vec<u32> {
    let prog = parse(ptx);
    let nnz = c.nz.len();
    let (rows, cols) = match f {
        SpmvFormat::Csr => {
            let mut ptrs = vec![0u32; c.m + 1];
            for &(r, _, _) in &c.nz {
                ptrs[r + 1] += 1;
            }
            for r in 0..c.m {
                ptrs[r + 1] += ptrs[r];
            }
            (le32(&ptrs), le32(&c.nz.iter().map(|t| t.1 as u32).collect::<Vec<_>>()))
        }
        SpmvFormat::Coo => (
            le64(&c.nz.iter().map(|t| t.0 as u64).collect::<Vec<_>>()),
            le64(&c.nz.iter().map(|t| t.1 as u64).collect::<Vec<_>>()),
        ),
    };
    let y_init: Vec<u32> = match f {
        // CSR writes every row: start from poison so a skipped row shows.
        SpmvFormat::Csr => vec![POISON; c.m],
        // COO adds into `y`: start from values, so an add is told from a store.
        SpmvFormat::Coo => c.y0.clone(),
    };
    let mut y = y_init;
    y.extend(std::iter::repeat_n(POISON, TAIL));
    let mut global = vec![
        Segment { base: ROWS, bytes: rows },
        Segment { base: COLS, bytes: cols },
        Segment { base: VALUES, bytes: le32(&c.nz.iter().map(|t| t.2).collect::<Vec<_>>()) },
        Segment { base: X, bytes: le32(&c.x) },
        Segment { base: Y, bytes: le32(&y) },
    ];
    let n = match f {
        SpmvFormat::Csr => c.m,
        SpmvFormat::Coo => nnz,
    };
    let args: HashMap<String, u64> =
        f.param_names().iter().zip([ROWS, COLS, VALUES, X, Y, n as u64]).map(|(p, v)| (p.to_string(), v)).collect();
    let grid = n.div_ceil(block as usize) as u32 + 1;
    let mut ctas: Vec<u32> = (0..grid).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for ctaid in ctas {
        let mut l = Launch { prog: &prog, args: &args, global: &mut global, shared: vec![], ctaid, ctaid_y: 0, nctaid_x: 0, nctaid_y: 1, ntid: block, steps: 0 };
        run_cta(&mut l, order);
    }
    words(&global[4].bytes)
}

#[test]
fn the_kernels_agree() {
    for f in SpmvFormat::ALL {
        let (hand, kir) = (hand_ptx(f), kir_ptx(f));
        for c in cases(false) {
            for block in BLOCKS {
                for order in ORDERS {
                    assert!(run(f, &hand, &c, block, order) == run(f, &kir, &c, block, order), "{f:?} {}x{} {block} {order:?}", c.m, c.k);
                }
            }
        }
    }
}

#[test]
fn the_kernels_are_the_product() {
    for c in cases(true) {
        let fx = |w: u32| f32::from_bits(w);
        let csr = run(SpmvFormat::Csr, &kir_ptx(SpmvFormat::Csr), &c, ELEMENTWISE_BLOCK, Order::Ascending);
        let coo = run(SpmvFormat::Coo, &kir_ptx(SpmvFormat::Coo), &c, ELEMENTWISE_BLOCK, Order::Ascending);
        for r in 0..c.m {
            let terms = c.nz.iter().filter(|t| t.0 == r);
            let want = terms.clone().fold(0.0f32, |s, &(_, col, v)| fx(v).mul_add(fx(c.x[col]), s));
            assert_eq!(csr[r], want.to_bits(), "CSR {}x{} row {r}", c.m, c.k);
            let want = terms.fold(fx(c.y0[r]), |s, &(_, col, v)| s + fx(v) * fx(c.x[col]));
            assert_eq!(coo[r], want.to_bits(), "COO {}x{} row {r}", c.m, c.k);
        }
        assert!(csr[c.m..].iter().chain(&coo[c.m..]).all(|&w| w == POISON), "{}x{}: wrote past y", c.m, c.k);
    }
}

/// The data reaches every case the kernels distinguish: empty rows, rows
/// longer than the unroll, and COO rows hit more than once.
#[test]
fn the_data_covers_empty_long_and_repeated_rows() {
    let c = &cases(false)[3];
    let per_row = |r: usize| c.nz.iter().filter(|t| t.0 == r).count();
    assert!((0..c.m).any(|r| per_row(r) == 0));
    assert!((0..c.m).any(|r| per_row(r) > 8));
    assert!(c.nz.len() > ELEMENTWISE_BLOCK as usize, "COO needs more than one block");
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on any case,
/// block or schedule (or faults).
fn caught(f: SpmvFormat, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(f);
    let mutant = mutate(&kir_ptx(f));
    cases(false).iter().chain(cases(true).iter()).any(|c| {
        BLOCKS.into_iter().any(|block| {
            ORDERS.into_iter().any(|order| {
                let expect = run(f, &hand, c, block, order);
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(f, &mutant, c, block, order))) {
                    Ok(r) => r != expect,
                    Err(_) => true,
                }
            })
        })
    })
}

/// `ptx` with the `i`-th line containing `from` rewritten by `g`.
fn edit(ptx: &str, from: &str, i: usize, g: impl Fn(&str) -> String) -> String {
    let at = ptx
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains(from))
        .nth(i)
        .unwrap_or_else(|| panic!("no line {i} with `{from}`"))
        .0;
    ptx.lines().enumerate().map(|(n, l)| if n == at { g(l) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
}

fn nudge(ptx: &str, from: &str, to: &str, i: usize) -> String {
    edit(ptx, from, i, |l| l.replacen(from, to, 1))
}

/// `op.u64 d, a, b;` as `mov.u64 d, a;`: the operation dropped.
fn drop_op(ptx: &str, op: &str, i: usize) -> String {
    edit(ptx, op, i, |l| {
        let (head, operands) = l.split_once(op).expect("the op");
        let mut ops: Vec<&str> = operands.trim_end_matches(';').split(',').map(str::trim).collect();
        ops.truncate(2);
        format!("{head}mov.u64 {};", ops.join(", "))
    })
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for f in SpmvFormat::ALL {
        assert!(!caught(f, |p| p.to_string()), "{f:?}");
    }
}

/// The thread bound and, for CSR, the row's loop bound.
#[test]
fn relaxing_a_bound_is_caught() {
    for f in SpmvFormat::ALL {
        let bounds = if f == SpmvFormat::Csr { 2 } else { 1 };
        assert_eq!(kir_ptx(f).matches("setp.ge.u64 ").count(), bounds, "{f:?}");
        for i in 0..bounds {
            assert!(caught(f, |p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "{f:?} bound {i}");
        }
    }
}

#[test]
fn the_block_index_is_caught() {
    for f in SpmvFormat::ALL {
        assert!(caught(f, |p| p.replacen("%ctaid.x", "0", 1)), "{f:?}");
    }
}

/// Every address's element size: 4 bytes for the u32 CSR indices and the
/// f32 arrays, 8 for the COO indices. Each read as the other width is
/// caught.
#[test]
fn nudging_an_element_size_is_caught() {
    for f in SpmvFormat::ALL {
        let p = kir_ptx(f);
        let sites: Vec<(usize, &str)> = p
            .lines()
            .enumerate()
            .filter(|(_, l)| l.contains("mul.lo.u64") && (l.ends_with(", 4;") || l.ends_with(", 8;")))
            .map(|(n, l)| (n, if l.ends_with(", 4;") { ", 4;" } else { ", 8;" }))
            .collect();
        let want = if f == SpmvFormat::Csr { (6, 0) } else { (3, 2) };
        let fours = sites.iter().filter(|s| s.1 == ", 4;").count();
        assert_eq!((fours, sites.len() - fours), want, "{f:?}: element sizes");
        for (at, size) in sites {
            let other = if size == ", 4;" { ", 8;" } else { ", 4;" };
            let mutate = |p: &str| {
                p.lines().enumerate().map(|(n, l)| if n == at { l.replacen(size, other, 1) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
            };
            assert!(caught(f, mutate), "{f:?} element size at line {at}");
        }
    }
}

/// Every 64-bit add: the addresses, the CSR row's `+ 1` and step.
#[test]
fn dropping_an_add_is_caught() {
    for f in SpmvFormat::ALL {
        let adds = kir_ptx(f).matches("add.u64 ").count();
        assert_eq!(adds, if f == SpmvFormat::Csr { 8 } else { 5 }, "{f:?}");
        for i in 0..adds {
            assert!(caught(f, |p| drop_op(p, "add.u64 ", i)), "{f:?} add {i}");
        }
    }
}

/// CSR: the two `1`s (the next row pointer and the step), the
/// accumulator's start and the fused multiply-add.
#[test]
fn the_csr_accumulation_is_pinned() {
    let f = SpmvFormat::Csr;
    let p = kir_ptx(f);
    assert_eq!(p.matches(", 1;").count(), 2, "{p}");
    for i in 0..2 {
        assert!(caught(f, |p| nudge(p, ", 1;", ", 2;", i)), "the {i}-th 1");
    }
    assert!(caught(f, |p| p.replacen("0f00000000;", "0f3F800000;", 1)), "start");
    assert!(caught(f, |p| edit(p, "fma.rn.f32 ", 0, |l| {
        let (head, ops) = l.split_once("fma.rn.f32 ").expect("fma");
        let ops: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
        format!("{head}mul.rn.f32 {}, {}, {};", ops[0], ops[1], ops[2])
    })), "fma without its addend");
    assert!(caught(f, |p| edit(p, "fma.rn.f32 ", 0, |l| {
        let (head, ops) = l.split_once("fma.rn.f32 ").expect("fma");
        let ops: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
        format!("{head}fma.rn.f32 {}, {}, {}, {};", ops[0], ops[1], ops[1], ops[3])
    })), "fma squaring the value");
}

/// COO: the product and the atomic add.
#[test]
fn the_coo_update_is_pinned() {
    let f = SpmvFormat::Coo;
    assert!(caught(f, |p| p.replacen("mul.rn.f32 ", "add.rn.f32 ", 1)), "product");
    assert!(caught(f, |p| p.replacen("red.global.add.f32 ", "st.global.f32 ", 1)), "add as store");
}
