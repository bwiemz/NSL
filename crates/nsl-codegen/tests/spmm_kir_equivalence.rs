//! The differential equivalence gate for the sparse matrix-matrix kernels
//! from `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5):
//! `nsl_csr_spmm_f32`, `nsl_coo_spmm_f32` and `nsl_bsr_spmm_f32`, now built
//! by `nsl_kir::kernels::spmm`.
//!
//! This file runs the frozen hand modules (`tests/fixtures/spmm_hand.rs`)
//! and the KIR ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`) over whole grids, with one block
//! more than the work needs in every grid dimension, under two schedules.
//! CSR and COO run on the runtime's 256-thread block and on a 32-thread
//! one; BSR on the runtime's `min(256, N) × block_rows` block, on a
//! narrower `4 × block_rows` one, and on a `4 × (block_rows + 1)` one whose
//! last row of threads only the sub-row guard stops.
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory, over empty rows, a single nonzero, ragged rows, repeated COO
//!    rows and columns, output widths below, at and past a block, and BSR
//!    blocks of 1×1, 2×3 and 3×2, with general f32 data.
//! 2. **Correctness**: on data whose products and sums are exact, CSR and
//!    BSR outputs are `fma`-accumulated from `+0.0` over the row's terms in
//!    order, and COO adds every nonzero's product to the `C` it started
//!    with (an add, not a store). Nothing past `C` is written.
//! 3. **The gate bites**: every bound, every block and thread index, the
//!    output column's multiply and add, every element size, every 64-bit add
//!    and multiply that forms an index, the row pointer's `+ 1`, each
//!    loop's step and start, the accumulator's start, the fused
//!    multiply-adds, the COO product and the atomic add.

use std::collections::HashMap;

use nsl_kir::kernels::spmm::{ptx, SpmmFormat, SPMM_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/spmm_hand.rs"]
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
const BMAT: u64 = 0x4000_0000;
const CMAT: u64 = 0x5000_0000;

fn kir_ptx(f: SpmmFormat) -> String {
    String::from_utf8(ptx(f)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(f: SpmmFormat) -> String {
    match f {
        SpmmFormat::Csr => hand::CSR_SPMM_F32_PTX,
        SpmmFormat::Coo => hand::COO_SPMM_F32_PTX,
        SpmmFormat::Bsr => hand::BSR_SPMM_F32_PTX,
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

/// `exact`: small integers, so every product and sum is exact and the
/// result does not depend on the order of the adds.
fn val(s: &mut u64, exact: bool) -> u32 {
    if exact {
        ((lcg(s) % 7) as f32 - 3.0).to_bits()
    } else {
        let e = (lcg(s) % 17) as i32 - 8;
        let mant = 1.0 + (lcg(s) % (1 << 23)) as f32 / (1u32 << 23) as f32;
        let sign = if lcg(s).is_multiple_of(2) { 1.0 } else { -1.0 };
        (sign * mant * 2f32.powi(e)).to_bits()
    }
}

/// A sparse matrix of `m` (block) rows over `k` (block) columns, as
/// `(row, col, block)` triples in row order, each block `br × bc` values
/// (`1 × 1` for CSR and COO). Rows get 0 to 12 nonzeros (so some are empty
/// and some are long), and a row may name a column twice. `B` is `k · bc ×
/// n`, row-major; `c0` is `C`'s starting contents for COO.
struct Case {
    m: usize,
    k: usize,
    n: usize,
    br: usize,
    bc: usize,
    nz: Vec<(usize, usize, Vec<u32>)>,
    b: Vec<u32>,
    c0: Vec<u32>,
}

fn case(m: usize, k: usize, n: usize, (br, bc): (usize, usize), seed: u64, exact: bool) -> Case {
    let mut s = seed;
    let mut nz = vec![];
    for r in 0..m {
        let count = (lcg(&mut s) % 13) as usize;
        let count = if r % 5 == 2 { 0 } else { count };
        for _ in 0..count {
            let c = (lcg(&mut s) % k as u64) as usize;
            nz.push((r, c, (0..br * bc).map(|_| val(&mut s, exact)).collect()));
        }
    }
    let b = (0..k * bc * n).map(|_| val(&mut s, exact)).collect();
    let c0 = (0..m * br * n).map(|_| val(&mut s, exact)).collect();
    Case { m, k, n, br, bc, nz, b, c0 }
}

/// `(m, k, n)` for CSR and COO: output widths below, at and past one block.
const SHAPES: [(usize, usize, usize); 4] = [(1, 1, 1), (7, 5, 3), (9, 13, 256), (6, 11, 300)];

/// `(m, k, n, (br, bc))` for BSR.
const BSR_SHAPES: [(usize, usize, usize, (usize, usize)); 4] =
    [(1, 1, 1, (1, 1)), (5, 4, 7, (2, 3)), (4, 3, 257, (3, 2)), (6, 5, 20, (1, 1))];

fn cases(f: SpmmFormat, exact: bool) -> Vec<Case> {
    let mut v: Vec<Case> = match f {
        SpmmFormat::Bsr => BSR_SHAPES.iter().enumerate().map(|(i, &(m, k, n, blk))| case(m, k, n, blk, i as u64 + 11, exact)).collect(),
        _ => SHAPES.iter().enumerate().map(|(i, &(m, k, n))| case(m, k, n, (1, 1), i as u64 + 1, exact)).collect(),
    };
    // A single nonzero.
    let one = if f == SpmmFormat::Bsr { vec![2.5f32.to_bits(); 6] } else { vec![2.5f32.to_bits()] };
    let (br, bc) = if f == SpmmFormat::Bsr { (2, 3) } else { (1, 1) };
    v.push(Case {
        m: 3,
        k: 2,
        n: 4,
        br,
        bc,
        nz: vec![(1, 1, one)],
        b: (0..2 * bc * 4).map(|i| (i as f32 - 3.0).to_bits()).collect(),
        c0: vec![0; 3 * br * 4],
    });
    v
}

/// The block shapes a case runs on: `(x, y)`.
fn blocks(f: SpmmFormat, c: &Case) -> Vec<(u32, u32)> {
    match f {
        // The runtime's block, a narrower one, and one a row taller than a
        // block so the sub-row guard has a thread to stop.
        SpmmFormat::Bsr => vec![(c.n.clamp(1, 256) as u32, c.br as u32), (4, c.br as u32), (4, c.br as u32 + 1)],
        _ => vec![(SPMM_BLOCK, 1), (32, 1)],
    }
}

const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

/// `C` after the launch (with its tail).
fn run(f: SpmmFormat, ptx: &str, c: &Case, (bx, by): (u32, u32), order: Order) -> Vec<u32> {
    let prog = parse(ptx);
    let nnz = c.nz.len();
    let (rows, cols) = match f {
        SpmmFormat::Coo => (
            le64(&c.nz.iter().map(|t| t.0 as u64).collect::<Vec<_>>()),
            le64(&c.nz.iter().map(|t| t.1 as u64).collect::<Vec<_>>()),
        ),
        _ => {
            let mut ptrs = vec![0u32; c.m + 1];
            for &(r, _, _) in &c.nz {
                ptrs[r + 1] += 1;
            }
            for r in 0..c.m {
                ptrs[r + 1] += ptrs[r];
            }
            (le32(&ptrs), le32(&c.nz.iter().map(|t| t.1 as u32).collect::<Vec<_>>()))
        }
    };
    let mut cm: Vec<u32> = match f {
        // COO adds into `C`: start from values, so an add is told from a store.
        SpmmFormat::Coo => c.c0.clone(),
        // CSR and BSR write every output: start from poison so a skipped one shows.
        _ => vec![POISON; c.m * c.br * c.n],
    };
    cm.extend(std::iter::repeat_n(POISON, TAIL));
    let mut global = vec![
        Segment { base: ROWS, bytes: rows },
        Segment { base: COLS, bytes: cols },
        Segment { base: VALUES, bytes: le32(&c.nz.iter().flat_map(|t| t.2.clone()).collect::<Vec<_>>()) },
        Segment { base: BMAT, bytes: le32(&c.b) },
        Segment { base: CMAT, bytes: le32(&cm) },
    ];
    let (m, n, nnz) = (c.m as u64, c.n as u64, nnz as u64);
    let values: Vec<u64> = match f {
        SpmmFormat::Csr => vec![ROWS, COLS, VALUES, BMAT, CMAT, m, n],
        SpmmFormat::Coo => vec![ROWS, COLS, VALUES, BMAT, CMAT, n, nnz],
        SpmmFormat::Bsr => vec![ROWS, COLS, VALUES, BMAT, CMAT, n, c.br as u64, c.bc as u64, m],
    };
    let args: HashMap<String, u64> = f.param_names().iter().zip(values).map(|(p, v)| (p.to_string(), v)).collect();
    let (gx, gy) = match f {
        SpmmFormat::Coo => (c.nz.len().div_ceil(bx as usize) as u32 + 1, 1),
        _ => (c.m as u32 + 1, c.n.div_ceil(bx as usize) as u32 + 1),
    };
    let mut ctas: Vec<(u32, u32)> = (0..gy).flat_map(|y| (0..gx).map(move |x| (x, y))).collect();
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
            nctaid_y: gy,
            ntid: bx,
            steps: 0,
        };
        if by == 1 {
            run_cta(&mut l, order);
        } else {
            run_cta_2d(&mut l, by, order);
        }
    }
    words(&global[4].bytes)
}

#[test]
fn the_kernels_agree() {
    for f in SpmmFormat::ALL {
        let (hand, kir) = (hand_ptx(f), kir_ptx(f));
        for c in cases(f, false) {
            for block in blocks(f, &c) {
                for order in ORDERS {
                    assert!(run(f, &hand, &c, block, order) == run(f, &kir, &c, block, order), "{f:?} {}x{}x{} {block:?} {order:?}", c.m, c.k, c.n);
                }
            }
        }
    }
}

/// The product, restated, as `C` (without its tail).
fn reference(f: SpmmFormat, c: &Case) -> Vec<u32> {
    let fx = |w: u32| f32::from_bits(w);
    let bat = |row: usize, j: usize| fx(c.b[row * c.n + j]);
    let mut out = vec![0u32; c.m * c.br * c.n];
    for r in 0..c.m {
        let terms: Vec<&(usize, usize, Vec<u32>)> = c.nz.iter().filter(|t| t.0 == r).collect();
        for sr in 0..c.br {
            for j in 0..c.n {
                let at = (r * c.br + sr) * c.n + j;
                out[at] = match f {
                    SpmmFormat::Csr => terms.iter().fold(0.0f32, |s, t| fx(t.2[0]).mul_add(bat(t.1, j), s)),
                    SpmmFormat::Coo => terms.iter().fold(fx(c.c0[at]), |s, t| s + fx(t.2[0]) * bat(t.1, j)),
                    SpmmFormat::Bsr => terms.iter().fold(0.0f32, |s, t| {
                        (0..c.bc).fold(s, |s, sc| fx(t.2[sr * c.bc + sc]).mul_add(bat(t.1 * c.bc + sc, j), s))
                    }),
                }
                .to_bits();
            }
        }
    }
    out
}

#[test]
fn the_kernels_are_the_product() {
    for f in SpmmFormat::ALL {
        for which in [hand_ptx(f), kir_ptx(f)] {
            for c in cases(f, true) {
                for block in blocks(f, &c) {
                    let got = run(f, &which, &c, block, Order::Ascending);
                    let (out, tail) = got.split_at(c.m * c.br * c.n);
                    assert_eq!(out, &reference(f, &c)[..], "{f:?} {}x{}x{} {block:?}", c.m, c.k, c.n);
                    assert!(tail.iter().all(|&w| w == POISON), "{f:?} {}x{}x{}: wrote past C", c.m, c.k, c.n);
                }
            }
        }
    }
}

/// The data reaches every case the kernels distinguish: empty rows, rows
/// longer than an unroll, COO rows hit more than once, more nonzeros than a
/// block, and output widths past a block.
#[test]
fn the_data_covers_every_case() {
    for f in SpmmFormat::ALL {
        let cs = cases(f, false);
        let per_row = |c: &Case, r: usize| c.nz.iter().filter(|t| t.0 == r).count();
        assert!(cs.iter().any(|c| (0..c.m).any(|r| per_row(c, r) == 0)), "{f:?}");
        assert!(cs.iter().any(|c| (0..c.m).any(|r| per_row(c, r) > 8)), "{f:?}");
        assert!(cs.iter().any(|c| c.n > 256), "{f:?}");
    }
    assert!(cases(SpmmFormat::Coo, false).iter().any(|c| c.nz.len() > 32), "COO needs more than one small block");
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on any case,
/// block or schedule (or faults).
fn caught(f: SpmmFormat, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(f);
    let mutant = mutate(&kir_ptx(f));
    cases(f, false).iter().chain(cases(f, true).iter()).any(|c| {
        blocks(f, c).into_iter().any(|block| {
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

/// `op d, a, b;` as `mov.<ty> d, a;`: the operation dropped.
fn drop_op(ptx: &str, op: &str, ty: &str, i: usize) -> String {
    edit(ptx, op, i, |l| {
        let (head, operands) = l.split_once(op).expect("the op");
        let mut ops: Vec<&str> = operands.trim_end_matches(';').split(',').map(str::trim).collect();
        ops.truncate(2);
        format!("{head}mov.{ty} {};", ops.join(", "))
    })
}

/// The line numbers of each loop step: an `add.u64` right after the
/// `mov.u64 _, 1;` it adds.
fn steps(p: &str) -> Vec<usize> {
    let lines: Vec<&str> = p.lines().collect();
    (1..lines.len()).filter(|&n| lines[n].contains("add.u64 ") && lines[n - 1].trim().starts_with("mov.u64 ") && lines[n - 1].ends_with(", 1;")).collect()
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for f in SpmmFormat::ALL {
        assert!(!caught(f, |p| p.to_string()), "{f:?}");
    }
}

/// Every bound: the grid's guards and each loop's.
#[test]
fn relaxing_a_bound_is_caught() {
    for f in SpmmFormat::ALL {
        let bounds = match f {
            SpmmFormat::Csr => 3,
            SpmmFormat::Coo => 2,
            SpmmFormat::Bsr => 5,
        };
        assert_eq!(kir_ptx(f).matches("setp.ge.u64 ").count(), bounds, "{f:?}");
        for i in 0..bounds {
            assert!(caught(f, |p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "{f:?} bound {i}");
        }
    }
}

/// Every block and thread index, and the output column's 32-bit multiply
/// and add.
#[test]
fn every_index_is_pinned() {
    for f in SpmmFormat::ALL {
        let p = kir_ptx(f);
        let specials: &[&str] = match f {
            SpmmFormat::Csr => &["%ctaid.x", "%ctaid.y", "%ntid.x", "%tid.x"],
            SpmmFormat::Coo => &["%ctaid.x", "%ntid.x", "%tid.x"],
            SpmmFormat::Bsr => &["%ctaid.x", "%ctaid.y", "%ntid.x", "%tid.x", "%tid.y"],
        };
        for s in specials {
            assert!(p.contains(s), "{f:?} {s}");
            assert!(caught(f, |p| p.replacen(s, "0", 1)), "{f:?} {s} as 0");
        }
        if f != SpmmFormat::Coo {
            assert!(caught(f, |p| drop_op(p, "mul.lo.u32 ", "u32", 0)), "{f:?} column multiply");
            assert!(caught(f, |p| drop_op(p, "add.u32 ", "u32", 0)), "{f:?} column add");
        }
    }
}

/// Every address's element size: 4 bytes for the u32 CSR/BSR indices and
/// the f32 arrays, 8 for the COO indices. Each read as the other width is
/// caught.
#[test]
fn nudging_an_element_size_is_caught() {
    for f in SpmmFormat::ALL {
        let p = kir_ptx(f);
        let sites: Vec<(usize, &str)> = p
            .lines()
            .enumerate()
            .filter(|(_, l)| l.contains("mul.lo.u64") && (l.ends_with(", 4;") || l.ends_with(", 8;")))
            .map(|(n, l)| (n, if l.ends_with(", 4;") { ", 4;" } else { ", 8;" }))
            .collect();
        assert!(!sites.is_empty(), "{f:?}");
        for (at, size) in sites {
            let other = if size == ", 4;" { ", 8;" } else { ", 4;" };
            let mutate = |p: &str| {
                p.lines().enumerate().map(|(n, l)| if n == at { l.replacen(size, other, 1) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
            };
            assert!(caught(f, mutate), "{f:?} element size at line {at}");
        }
    }
}

/// Every 64-bit add and every index multiply (`a · n`) but the loop steps,
/// which are nudged instead (dropped, a loop would not end).
#[test]
fn every_index_add_and_multiply_is_pinned() {
    for f in SpmmFormat::ALL {
        let p = kir_ptx(f);
        let lines: Vec<&str> = p.lines().collect();
        let step_lines = steps(&p);
        let adds: Vec<usize> = lines.iter().enumerate().filter(|(_, l)| l.contains("add.u64 ")).map(|(n, _)| n).collect();
        for (i, n) in adds.iter().enumerate() {
            if !step_lines.contains(n) {
                assert!(caught(f, |p| drop_op(p, "add.u64 ", "u64", i)), "{f:?} add {i}");
            }
        }
        let muls: Vec<usize> = lines
            .iter()
            .enumerate()
            .filter(|(_, l)| l.contains("mul.lo.u64 ") && !l.ends_with(", 4;") && !l.ends_with(", 8;"))
            .map(|(n, _)| n)
            .collect();
        assert!(!muls.is_empty(), "{f:?}");
        for i in 0..lines.iter().filter(|l| l.contains("mul.lo.u64 ")).count() {
            let l = lines.iter().filter(|l| l.contains("mul.lo.u64 ")).nth(i).expect("mul");
            if !l.ends_with(", 4;") && !l.ends_with(", 8;") {
                assert!(caught(f, |p| drop_op(p, "mul.lo.u64 ", "u64", i)), "{f:?} index multiply {i}: {l}");
            }
        }
    }
}

/// The row pointer's `+ 1`, each loop's step (as 2) and start, and the
/// accumulator's start.
#[test]
fn the_loops_are_pinned() {
    for f in SpmmFormat::ALL {
        let p = kir_ptx(f);
        let ones = p.lines().filter(|l| l.trim().starts_with("mov.u64 ") && l.ends_with(", 1;")).count();
        let want = match f {
            SpmmFormat::Csr => 2,
            SpmmFormat::Coo => 1,
            SpmmFormat::Bsr => 3,
        };
        assert_eq!(ones, want, "{f:?}: the row pointer's + 1 and the steps");
        for i in 0..ones {
            let at = p.lines().enumerate().filter(|(_, l)| l.trim().starts_with("mov.u64 ") && l.ends_with(", 1;")).nth(i).expect("one").0;
            let mutate = |p: &str| {
                p.lines().enumerate().map(|(n, l)| if n == at { l.replacen(", 1;", ", 2;", 1) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
            };
            assert!(caught(f, mutate), "{f:?} the {i}-th 1");
        }
        let zeros = p.lines().filter(|l| l.trim().starts_with("mov.u64 ") && l.ends_with(", 0;")).count();
        assert_eq!(zeros, if f == SpmmFormat::Csr { 0 } else { 1 }, "{f:?}: the column loop's start");
        if zeros == 1 {
            assert!(caught(f, |p| nudge(p, ", 0;", ", 1;", 0)), "{f:?} the column loop's start");
        }
        if f != SpmmFormat::Coo {
            assert!(caught(f, |p| p.replacen("0f00000000;", "0f3F800000;", 1)), "{f:?} accumulator start");
        }
    }
}

/// CSR and BSR: the fused multiply-add (without its addend, or squaring the
/// value). COO: the product and the atomic add.
#[test]
fn the_arithmetic_is_pinned() {
    for f in [SpmmFormat::Csr, SpmmFormat::Bsr] {
        assert!(caught(f, |p| edit(p, "fma.rn.f32 ", 0, |l| {
            let (head, ops) = l.split_once("fma.rn.f32 ").expect("fma");
            let ops: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
            format!("{head}mul.rn.f32 {}, {}, {};", ops[0], ops[1], ops[2])
        })), "{f:?} fma without its addend");
        assert!(caught(f, |p| edit(p, "fma.rn.f32 ", 0, |l| {
            let (head, ops) = l.split_once("fma.rn.f32 ").expect("fma");
            let ops: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
            format!("{head}fma.rn.f32 {}, {}, {}, {};", ops[0], ops[1], ops[1], ops[3])
        })), "{f:?} fma squaring the value");
    }
    let f = SpmmFormat::Coo;
    assert!(caught(f, |p| p.replacen("mul.rn.f32 ", "add.rn.f32 ", 1)), "product");
    assert!(caught(f, |p| p.replacen("red.global.add.f32 ", "st.global.f32 ", 1)), "add as store");
}
