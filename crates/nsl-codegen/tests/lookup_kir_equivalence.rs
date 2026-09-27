//! The differential equivalence gate for the 2-D-block row-lookup kernels
//! from `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5):
//! `nsl_embedding_f32`, `nsl_embedding_i32idx`, `nsl_gather_f32` and
//! `nsl_gather_i32idx`, now built by `nsl_kir::kernels::lookup`.
//!
//! This file runs the frozen hand modules (`tests/fixtures/lookup_hand.rs`)
//! and the KIR ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), whose two-dimensional blocks
//! (`run_cta_2d`) these kernels are the first to need. Each launch covers
//! the output with one block more than it needs in each direction, on the
//! runtime's 16 × 16 block and on an 8 × 4 one (on a square block, `%ntid.x`
//! and `%ntid.y` are the same number, and a kernel that confused them would
//! pass):
//!
//! 1. **Agreement**: under two schedules, the two kernels leave *the same
//!    bytes* in all of global memory, over ragged shapes and indices that
//!    are, for the f32 kernels, fractional, negative, NaN and, for the
//!    gather, at the bound, past it and infinite; for the i32 gather,
//!    negative, at the bound, past it and `i32::MIN` / `i32::MAX`.
//! 2. **Correctness**: each output element is the table element its index
//!    names, bit for bit; a gather row whose index is out of range, and
//!    everything past the output, is left unwritten.
//! 3. **The gate bites**: each bound, each block index, `%tid.y`, `%ntid.y`,
//!    every element size and every 64-bit add and multiply are caught. The
//!    i32 index's sign extension is named as an equivalent mutant.

use std::collections::HashMap;

use nsl_kir::kernels::lookup::{lookup_ptx, IndexDtype, LookupOp, LOOKUP_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/lookup_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 300;
const POISON: u32 = 0x7FA5_A5A5;
const TABLE: u64 = 0x1000_0000;
const INDICES: u64 = 0x2000_0000;
const OUT: u64 = 0x3000_0000;

#[derive(Clone, Copy, Debug, PartialEq)]
struct K(LookupOp, IndexDtype);

const ALL: [K; 4] = [
    K(LookupOp::Embedding, IndexDtype::F32),
    K(LookupOp::Embedding, IndexDtype::I32),
    K(LookupOp::Gather, IndexDtype::F32),
    K(LookupOp::Gather, IndexDtype::I32),
];

fn kir_ptx(k: K) -> String {
    String::from_utf8(lookup_ptx(k.0, k.1)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(k: K) -> String {
    match k {
        K(LookupOp::Embedding, IndexDtype::F32) => hand::EMBEDDING_F32_PTX,
        K(LookupOp::Embedding, IndexDtype::I32) => hand::EMBEDDING_I32IDX_PTX,
        K(LookupOp::Gather, IndexDtype::F32) => hand::GATHER_F32_PTX,
        K(LookupOp::Gather, IndexDtype::I32) => hand::GATHER_I32IDX_PTX,
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

/// The table: distinct f32 bit patterns, so a wrong row or column shows.
fn table(len: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..len).map(|_| ((lcg(&mut s) as f32 / (1u64 << 31) as f32) * 8.0 - 4.0).to_bits()).collect()
}

/// `(rows, cols, table_rows)`: ragged against both blocks.
const SHAPES: [(usize, usize, usize); 5] = [(1, 1, 1), (17, 3, 5), (5, 33, 9), (40, 17, 7), (3, 70, 300)];

/// The block shapes: the runtime's, and a non-square one.
const BLOCKS: [[u32; 2]; 2] = [LOOKUP_BLOCK, [8, 4]];

/// The index buffer's words. The embedding's indices stay in range (its
/// kernels do not check them; the host has); the gather's do not.
fn indices(k: K, rows: usize, table_rows: usize, seed: u64) -> Vec<u32> {
    let t = table_rows as f32;
    let f32_special: &[f32] = match k.0 {
        LookupOp::Embedding => &[0.0, t - 1.0, 1.7, -1.5, f32::NAN, -0.0, t - 0.5],
        LookupOp::Gather => &[0.0, t - 1.0, 1.7, -1.5, f32::NAN, t, t + 3.0, 1e30, f32::INFINITY, -0.0],
    };
    let tr = table_rows as i32;
    let i32_special: &[i32] = match k.0 {
        LookupOp::Embedding => &[0, tr - 1],
        LookupOp::Gather => &[0, tr - 1, -1, tr, tr + 7, i32::MIN, i32::MAX],
    };
    let mut s = seed;
    (0..rows)
        .map(|n| {
            let r = lcg(&mut s) % table_rows as u64;
            match k.1 {
                IndexDtype::F32 => {
                    if n < f32_special.len() { f32_special[(n + seed as usize) % f32_special.len()] } else { r as f32 }.to_bits()
                }
                IndexDtype::I32 => {
                    (if n < i32_special.len() { i32_special[(n + seed as usize) % i32_special.len()] } else { r as i32 }) as u32
                }
            }
        })
        .collect()
}

struct Run {
    table: Vec<u32>,
    idx: Vec<u32>,
    mem: Vec<Vec<u8>>,
}

fn run(k: K, ptx: &str, (rows, cols, table_rows): (usize, usize, usize), block: [u32; 2], seed: u64, order: Order) -> Run {
    let table = table(table_rows * cols, seed);
    let idx = indices(k, rows, table_rows, seed);
    let prog = parse(ptx);
    let mut global = vec![
        Segment { base: TABLE, bytes: le32(&table) },
        Segment { base: INDICES, bytes: le32(&idx) },
        Segment { base: OUT, bytes: le32(&vec![POISON; rows * cols + TAIL]) },
    ];
    let names = k.0.param_names();
    let args: HashMap<String, u64> = [TABLE, INDICES, OUT, rows as u64, cols as u64, table_rows as u64]
        .iter()
        .zip(names)
        .map(|(v, n)| (n.to_string(), *v))
        .collect();
    let grid_x = rows.div_ceil(block[0] as usize) as u32 + 1;
    let grid_y = cols.div_ceil(block[1] as usize) as u32 + 1;
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
    Run { table, idx, mem: global.into_iter().map(|s| s.bytes).collect() }
}

const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

/// The row an index names, or `None` for a gather index out of range.
fn row(k: K, word: u32, table_rows: usize) -> Option<usize> {
    let r = match k.1 {
        // `cvt.rzi.u64.f32`: toward zero, saturating, NaN to 0.
        IndexDtype::F32 => f32::from_bits(word) as u64,
        // `cvt.u64.s32`: sign-extended.
        IndexDtype::I32 => word as i32 as i64 as u64,
    };
    (r < table_rows as u64).then_some(r as usize).or_else(|| {
        assert_eq!(k.0, LookupOp::Gather, "an embedding index out of range in the gate's own data");
        None
    })
}

#[test]
fn the_kernels_agree_and_are_the_lookup() {
    for k in ALL {
        let (hand, kir) = (hand_ptx(k), kir_ptx(k));
        for (n, &shape) in SHAPES.iter().enumerate() {
            let (rows, cols, table_rows) = shape;
            for block in BLOCKS {
                for order in ORDERS {
                    let h = run(k, &hand, shape, block, n as u64 + 1, order);
                    let q = run(k, &kir, shape, block, n as u64 + 1, order);
                    assert!(h.mem == q.mem, "{k:?} {shape:?} {block:?} {order:?}: global memory differs");
                }
            }
            let r = run(k, &kir, shape, LOOKUP_BLOCK, n as u64 + 1, Order::Ascending);
            let out = words(&r.mem[2]);
            for i in 0..rows {
                let want_row = row(k, r.idx[i], table_rows);
                for j in 0..cols {
                    let want = want_row.map_or(POISON, |t| r.table[t * cols + j]);
                    assert_eq!(out[i * cols + j], want, "{k:?} {shape:?} ({i}, {j}) index {:#x}", r.idx[i]);
                }
            }
            assert!(out[rows * cols..].iter().all(|&w| w == POISON), "{k:?} {shape:?}: wrote past the output");
        }
    }
}

/// The gate's data reaches every branch: some gather rows are skipped and
/// some are copied, for both index dtypes.
#[test]
fn the_gather_data_takes_both_sides_of_the_index_test() {
    for idx in IndexDtype::ALL {
        let k = K(LookupOp::Gather, idx);
        let (rows, _, table_rows) = SHAPES[3];
        let words = indices(k, rows, table_rows, 4);
        let skipped = words.iter().filter(|&&w| row(k, w, table_rows).is_none()).count();
        assert!(skipped >= 4 && skipped < rows, "{idx:?}: {skipped} of {rows} skipped");
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on any case,
/// under either schedule or block (or faults).
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

/// `ptx` with the `i`-th line containing `from` rewritten by `f`.
fn edit(ptx: &str, from: &str, i: usize, f: impl Fn(&str) -> String) -> String {
    let at = ptx
        .lines()
        .enumerate()
        .filter(|(_, l)| l.contains(from))
        .nth(i)
        .unwrap_or_else(|| panic!("no line {i} with `{from}`"))
        .0;
    ptx.lines().enumerate().map(|(n, l)| if n == at { f(l) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
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
    for k in ALL {
        assert!(!caught(k, |p| p.to_string()), "{k:?}");
    }
}

/// The row bound, the column bound and, for the gather, the index bound.
#[test]
fn relaxing_a_bound_is_caught() {
    for k in ALL {
        let bounds = if k.0 == LookupOp::Gather { 3 } else { 2 };
        assert_eq!(kir_ptx(k).matches("setp.ge.u64 ").count(), bounds, "{k:?}");
        for i in 0..bounds {
            assert!(caught(k, |p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "{k:?} bound {i}");
        }
    }
}

/// Both block indices, `%tid.y` and `%ntid.y`. `%ntid.y` read as
/// `%ntid.x` is caught only because the gate runs an 8 × 4 block too.
#[test]
fn the_two_dimensional_index_is_caught() {
    for k in ALL {
        let p = kir_ptx(k);
        for special in ["%ctaid.x;", "%ctaid.y;", "%tid.y;", "%ntid.y;"] {
            assert_eq!(p.matches(special).count(), 1, "{k:?} {special}");
        }
        assert!(caught(k, |p| p.replacen("%ctaid.x;", "0;", 1)), "{k:?} ctaid.x");
        assert!(caught(k, |p| p.replacen("%ctaid.y;", "0;", 1)), "{k:?} ctaid.y");
        assert!(caught(k, |p| p.replacen("%tid.y;", "%tid.x;", 1)), "{k:?} tid.y");
        assert!(caught(k, |p| p.replacen("%ntid.y;", "%ntid.x;", 1)), "{k:?} ntid.y");
    }
}

/// Every address's element size: the index, the table and the output.
#[test]
fn nudging_an_element_size_is_caught() {
    for k in ALL {
        let p = kir_ptx(k);
        let sizes = p.lines().filter(|l| l.contains("mul.lo.u64") && l.ends_with(", 4;")).count();
        assert_eq!(sizes, 3, "{k:?}");
        for i in 0..sizes {
            let size_line = |p: &str| {
                let at = p.lines().enumerate().filter(|(_, l)| l.contains("mul.lo.u64") && l.ends_with(", 4;")).nth(i).expect("site").0;
                p.lines()
                    .enumerate()
                    .map(|(n, l)| if n == at { l.replacen(", 4;", ", 8;", 1) } else { l.to_string() })
                    .collect::<Vec<_>>()
                    .join("\n")
                    + "\n"
            };
            assert!(caught(k, size_line), "{k:?} element size {i}");
        }
    }
}

/// Every 64-bit add (the column, the bases) and every multiply that is not
/// an element size (the row strides) carries weight.
#[test]
fn dropping_an_add_or_a_stride_is_caught() {
    for k in ALL {
        let p = kir_ptx(k);
        let adds = p.matches("add.u64 ").count();
        assert_eq!(adds, 5, "{k:?}: two column adds and three base adds");
        for i in 0..adds {
            assert!(caught(k, |p| drop_op(p, "add.u64 ", i)), "{k:?} add {i}");
        }
        let strides: Vec<usize> = p
            .lines()
            .filter(|l| l.contains("mul.lo.u64"))
            .enumerate()
            .filter(|(_, l)| !l.ends_with(", 4;"))
            .map(|(n, _)| n)
            .collect();
        assert_eq!(strides.len(), 2, "{k:?}: the table row and the output row");
        for i in strides {
            assert!(caught(k, |p| drop_op(p, "mul.lo.u64 ", i)), "{k:?} stride {i}");
        }
    }
}

/// The f32 index truncates: read with the i32 kernel's widening instead it
/// is caught, and the i32 index's sign extension, read as a zero extension,
/// is not. That mutant is equivalent here: a negative index widens to at
/// least `2^31` either way, past every table the gate has, so the gather
/// skips it both ways; the embedding kernels never see one.
#[test]
fn the_index_widening_is_pinned() {
    for op in LookupOp::ALL {
        let f = K(op, IndexDtype::F32);
        assert!(caught(f, |p| p.replacen("cvt.rzi.u64.f32 ", "cvt.u64.u32 ", 1)), "{op:?} f32 truncation");
        let i = K(op, IndexDtype::I32);
        assert_eq!(kir_ptx(i).matches("cvt.u64.s32 ").count(), 1, "{op:?}");
        assert!(!caught(i, |p| p.replacen("cvt.u64.s32 ", "cvt.u64.u32 ", 1)), "{op:?}: named equivalent mutant");
    }
}
