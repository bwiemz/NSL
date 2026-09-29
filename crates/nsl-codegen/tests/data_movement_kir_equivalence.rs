//! The differential equivalence gate for the data-movement kernels from
//! `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5):
//! `nsl_bias_add_f32`, `nsl_gather_dim_f32`, `nsl_strided_copy_f32` and
//! `nsl_slice_f32`, now built by `nsl_kir::kernels::data_movement`.
//!
//! This file runs the frozen hand modules
//! (`tests/fixtures/data_movement_hand.rs`) and the KIR ones side by side on
//! the cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`), over
//! the grid the runtime launches (`ceil(n / 256)` blocks of 256) plus one
//! block more:
//!
//! 1. **Agreement**: under two schedules, the two kernels leave *the same
//!    bytes* in all of global memory, over IEEE-corner values, ragged
//!    sizes and, for the gather, indices that are fractional, negative,
//!    NaN, at the bound and past it. The strided walks run over
//!    contiguous, transposed, broadcast (zero source stride), zero
//!    destination stride and 0-d views.
//! 2. **Correctness**: each output is the kernel's formula, bit for bit,
//!    and nothing past `n` is written.
//! 3. **The gate bites**: the bound, the block index, every element size,
//!    the bias's modulus, the gather's quotient and remainder, its
//!    out-of-range test, the walk's zero-stride skip, its broadcast
//!    modulus, its increment and the slice's dimension test are each
//!    caught.

use std::collections::HashMap;

use nsl_kir::kernels::data_movement::{bias_add_ptx, gather_dim_ptx, strided_ptx, StridedOp};
use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;

#[allow(dead_code)]
#[path = "fixtures/data_movement_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 300;
const POISON: u32 = 0x7FA5_A5A5;

fn trim(s: &str) -> String {
    s.trim_end_matches('\0').to_string()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum K {
    Bias,
    Gather,
    Copy,
    Slice,
}

fn kir_ptx(k: K) -> String {
    trim(&String::from_utf8(match k {
        K::Bias => bias_add_ptx(),
        K::Gather => gather_dim_ptx(),
        K::Copy => strided_ptx(StridedOp::Copy),
        K::Slice => strided_ptx(StridedOp::Slice),
    })
    .expect("ASCII"))
}

fn hand_ptx(k: K) -> String {
    trim(match k {
        K::Bias => hand::BIAS_ADD_F32_PTX,
        K::Gather => hand::GATHER_DIM_F32_PTX,
        K::Copy => hand::STRIDED_COPY_F32_PTX,
        K::Slice => hand::GPU_SLICE_F32_PTX,
    })
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

/// f32 bit patterns: IEEE corners, then ordinary values.
fn values(n: usize, seed: u64) -> Vec<u32> {
    const CORNERS: [u32; 10] =
        [0x0000_0000, 0x8000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0001, 0x0000_0001, 0x7F7F_FFFF, 0xFF7F_FFFF, 0x3F80_0000, 0xC040_0000];
    let mut s = seed;
    (0..n)
        .map(|k| {
            if k < CORNERS.len() {
                return CORNERS[(k + seed as usize) % CORNERS.len()];
            }
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 8.0 - 4.0).to_bits()
        })
        .collect()
}

/// One launch: global segments at fixed bases, named arguments, `n` threads
/// wanted. Returns all of global memory.
fn launch(ptx: &str, segments: Vec<(u64, Vec<u8>)>, args: &[(&str, u64)], n: usize, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let mut global: Vec<Segment> = segments.into_iter().map(|(base, bytes)| Segment { base, bytes }).collect();
    let args: HashMap<String, u64> = args.iter().map(|(k, v)| (k.to_string(), *v)).collect();
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
            nctaid_x: 0, nctaid_y: 1,
            ntid: ELEMENTWISE_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    global.into_iter().map(|s| s.bytes).collect()
}

fn same(got: u32, want: u32) -> bool {
    got == want || (f32::from_bits(got).is_nan() && f32::from_bits(want).is_nan())
}

const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

// ---------------------------------------------------------------------------
// bias_add
// ---------------------------------------------------------------------------

const BIAS_SHAPES: [(usize, usize); 5] = [(1, 1), (3, 85), (257, 1), (4, 300), (1, 257)];

fn run_bias(ptx: &str, rows: usize, cols: usize, seed: u64, order: Order) -> (Vec<u32>, Vec<u32>, Vec<Vec<u8>>) {
    let total = rows * cols;
    let inp = values(total, seed);
    let bias = values(cols, seed + 7);
    let mut inp_t = inp.clone();
    inp_t.extend(vec![0x3F00_0000; TAIL]);
    let mem = launch(
        ptx,
        vec![(0x1000_0000, le32(&inp_t)), (0x2000_0000, le32(&bias)), (0x3000_0000, le32(&vec![POISON; total + TAIL]))],
        &[("inp", 0x1000_0000), ("bias", 0x2000_0000), ("out", 0x3000_0000), ("total", total as u64), ("cols", cols as u64)],
        total,
        order,
    );
    (inp, bias, mem)
}

#[test]
fn bias_add_agrees_and_is_the_formula() {
    let (hand, kir) = (hand_ptx(K::Bias), kir_ptx(K::Bias));
    for (j, &(rows, cols)) in BIAS_SHAPES.iter().enumerate() {
        for order in ORDERS {
            let h = run_bias(&hand, rows, cols, j as u64 + 1, order);
            let k = run_bias(&kir, rows, cols, j as u64 + 1, order);
            assert!(h.2 == k.2, "{rows}x{cols} {order:?}: global memory differs");
        }
        let (inp, bias, mem) = run_bias(&kir, rows, cols, j as u64 + 1, Order::Ascending);
        let out = words(&mem[2]);
        for i in 0..rows * cols {
            let want = (f32::from_bits(inp[i]) + f32::from_bits(bias[i % cols])).to_bits();
            assert!(same(out[i], want), "{rows}x{cols} i={i}");
        }
        assert!(out[rows * cols..].iter().all(|&w| w == POISON), "{rows}x{cols}: wrote past n");
    }
}

// ---------------------------------------------------------------------------
// gather_dim
// ---------------------------------------------------------------------------

const GATHER_SHAPES: [(usize, usize, usize); 5] = [(1, 1, 1), (5, 7, 3), (257, 4, 1), (3, 5, 100), (40, 9, 13)];

/// Indices as f32: in range, fractional (truncated), negative and NaN
/// (saturate to 0), at and past the bound, and huge.
fn indices(outer: usize, gds: usize, seed: u64) -> Vec<u32> {
    let g = gds as f32;
    let special = [0.0, g - 1.0, 1.7, -1.5, f32::NAN, g, g + 3.0, 1e30, f32::INFINITY, -0.0];
    let mut s = seed;
    (0..outer)
        .map(|k| {
            if k < special.len() {
                return special[(k + seed as usize) % special.len()].to_bits();
            }
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((s >> 33) % gds as u64) as f32
        }
        .to_bits())
        .collect()
}

fn run_gather(ptx: &str, (outer, gds, inner): (usize, usize, usize), seed: u64, order: Order) -> (Vec<u32>, Vec<u32>, Vec<Vec<u8>>) {
    let total = outer * inner;
    let input = values(outer * gds * inner, seed);
    let idx = indices(outer, gds, seed);
    let mem = launch(
        ptx,
        vec![(0x1000_0000, le32(&input)), (0x2000_0000, le32(&idx)), (0x3000_0000, le32(&vec![POISON; total + TAIL]))],
        &[
            ("input", 0x1000_0000),
            ("indices", 0x2000_0000),
            ("out", 0x3000_0000),
            ("outer", outer as u64),
            ("gather_dim_size", gds as u64),
            ("inner", inner as u64),
        ],
        total,
        order,
    );
    (input, idx, mem)
}

#[test]
fn gather_dim_agrees_and_is_the_formula() {
    let (hand, kir) = (hand_ptx(K::Gather), kir_ptx(K::Gather));
    for (j, &shape) in GATHER_SHAPES.iter().enumerate() {
        for order in ORDERS {
            let h = run_gather(&hand, shape, j as u64 + 1, order);
            let k = run_gather(&kir, shape, j as u64 + 1, order);
            assert!(h.2 == k.2, "{shape:?} {order:?}: global memory differs");
        }
        let (outer, gds, inner) = shape;
        let (input, idx, mem) = run_gather(&kir, shape, j as u64 + 1, Order::Ascending);
        let out = words(&mem[2]);
        for i in 0..outer * inner {
            let (o, k) = (i / inner, i % inner);
            // `cvt.rzi.u64.f32`: toward zero, saturating, NaN to 0.
            let x = f32::from_bits(idx[o]) as u64;
            let want = if x >= gds as u64 { 0 } else { input[(o * gds + x as usize) * inner + k] };
            assert_eq!(out[i], want, "{shape:?} i={i} idx={}", f32::from_bits(idx[o]));
        }
        assert!(out[outer * inner..].iter().all(|&w| w == POISON), "{shape:?}: wrote past n");
    }
}

// ---------------------------------------------------------------------------
// strided_copy / slice
// ---------------------------------------------------------------------------

/// One strided view: the output shape, the source strides, the destination
/// strides (contiguous over the output unless a case says otherwise), the
/// source length, and for the slice its `(dim, start)`.
#[derive(Clone, Debug)]
struct View {
    shape: Vec<u64>,
    src_strides: Vec<u64>,
    dst_strides: Vec<u64>,
    src_len: usize,
    slice: (u64, u64),
}

fn contiguous(shape: &[u64]) -> Vec<u64> {
    let mut s = vec![1u64; shape.len()];
    for d in (0..shape.len().saturating_sub(1)).rev() {
        s[d] = s[d + 1] * shape[d + 1];
    }
    s
}

fn views() -> Vec<View> {
    let v = |shape: &[u64], src_strides: &[u64], src_len: usize, slice: (u64, u64)| View {
        shape: shape.to_vec(),
        src_strides: src_strides.to_vec(),
        dst_strides: contiguous(shape),
        src_len,
        slice,
    };
    let mut out = vec![
        // Contiguous 1-d, ragged.
        v(&[257], &[1], 300, (0, 5)),
        // A transpose: [7, 11] read from an [11, 7] source.
        v(&[7, 11], &[1, 7], 77, (1, 0)),
        // Broadcast: a [1, 13] row read as [9, 13] (stride 0 on dim 0).
        v(&[9, 13], &[0, 1], 13, (1, 0)),
        // Rank 4 with a slice window into dim 2 of [3, 2, 10, 5].
        v(&[3, 2, 4, 5], &[100, 50, 5, 1], 300, (2, 6)),
        // Rank 3, row-major source larger than the view.
        v(&[4, 5, 6], &[60, 12, 2], 300, (0, 1)),
    ];
    // A zero destination stride: that dimension is skipped by the walk.
    let mut z = v(&[1, 64], &[64, 1], 128, (1, 0));
    z.dst_strides = vec![0, 1];
    out.push(z);
    // 0-d: the loop never runs and dst[0] = src[0].
    out.push(View { shape: vec![], src_strides: vec![], dst_strides: vec![], src_len: 4, slice: (0, 0) });
    out
}

const SRC: u64 = 0x1000_0000;
const DST: u64 = 0x2000_0000;
const SHAPE: u64 = 0x3000_0000;
const SSTR: u64 = 0x4000_0000;
const DSTR: u64 = 0x5000_0000;

fn run_strided(k: K, ptx: &str, view: &View, order: Order) -> (Vec<u32>, Vec<Vec<u8>>) {
    let total: usize = view.shape.iter().product::<u64>() as usize;
    let src = values(view.src_len, view.src_len as u64);
    let mut args = vec![
        ("src", SRC),
        ("dst", DST),
        ("shape", SHAPE),
        ("src_strides", SSTR),
        ("dst_strides", DSTR),
        ("ndim", view.shape.len() as u64),
        ("total", total as u64),
    ];
    if k == K::Slice {
        args.extend([("slice_dim", view.slice.0), ("slice_start", view.slice.1)]);
    }
    // A table segment may not be empty, so a 0-d view carries one unread slot.
    let table = |t: &[u64]| le64(if t.is_empty() { &[0] } else { t });
    let mem = launch(
        ptx,
        vec![
            (SRC, le32(&src)),
            (DST, le32(&vec![POISON; total + TAIL])),
            (SHAPE, table(&view.shape)),
            (SSTR, table(&view.src_strides)),
            (DSTR, table(&view.dst_strides)),
        ],
        &args,
        total,
        order,
    );
    (src, mem)
}

/// The walk, as the kernels define it.
fn walk(view: &View, i: u64, slice: bool) -> u64 {
    let (mut rem, mut off) = (i, 0u64);
    for d in 0..view.shape.len() {
        let ds = view.dst_strides[d];
        if ds == 0 {
            continue;
        }
        let mut c = (rem / ds) % view.shape[d];
        rem %= ds;
        if slice && d as u64 == view.slice.0 {
            c += view.slice.1;
        }
        off += c * view.src_strides[d];
    }
    off
}

#[test]
fn the_strided_walks_agree_and_are_the_formula() {
    for k in [K::Copy, K::Slice] {
        let (hand, kir) = (hand_ptx(k), kir_ptx(k));
        for view in views() {
            for order in ORDERS {
                assert!(run_strided(k, &hand, &view, order).1 == run_strided(k, &kir, &view, order).1, "{k:?} {view:?} {order:?}");
            }
            let total: usize = view.shape.iter().product::<u64>() as usize;
            let (src, mem) = run_strided(k, &kir, &view, Order::Ascending);
            let dst = words(&mem[1]);
            for i in 0..total {
                assert_eq!(dst[i], src[walk(&view, i as u64, k == K::Slice) as usize], "{k:?} {view:?} i={i}");
            }
            assert!(dst[total..].iter().all(|&w| w == POISON), "{k:?} {view:?}: wrote past n");
        }
    }
}

/// On an ordinary view the walk is the obvious one: a transpose reads
/// `src[c1 * 7 + c0]`, and the slice reads the window.
#[test]
fn the_walk_means_what_it_says() {
    let t = &views()[1];
    for i in 0..77u64 {
        let (c0, c1) = (i / 11, i % 11);
        assert_eq!(walk(t, i, false), c1 * 7 + c0);
    }
    let s = &views()[3];
    // [3, 2, 4, 5] window at dim 2 from 6 in a [3, 2, 10, 5] row-major source.
    assert_eq!(walk(s, 0, true), 6 * 5);
    assert_eq!(walk(s, 5, true), 7 * 5);
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on every case
/// of that kernel, under either schedule (or faults).
fn caught(k: K, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(k);
    let mutant = mutate(&kir_ptx(k));
    let differs = |run: &dyn Fn(&str) -> Vec<Vec<u8>>| {
        let expect = run(&hand);
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(&mutant))) {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
    };
    ORDERS.into_iter().any(|o| match k {
        K::Bias => BIAS_SHAPES.iter().any(|&(r, c)| differs(&|p| run_bias(p, r, c, 3, o).2)),
        K::Gather => GATHER_SHAPES.iter().any(|&s| differs(&|p| run_gather(p, s, 3, o).2)),
        K::Copy | K::Slice => views().iter().any(|v| differs(&|p| run_strided(k, p, v, o).1)),
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

const ALL: [K; 4] = [K::Bias, K::Gather, K::Copy, K::Slice];

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for k in ALL {
        assert!(!caught(k, |p| p.to_string()), "{k:?}");
    }
}

#[test]
fn relaxing_the_bound_or_ignoring_the_block_index_is_caught() {
    for k in ALL {
        let p = kir_ptx(k);
        assert_eq!(p.matches("%ctaid.x;").count(), 1, "{k:?}");
        let bound = p.lines().position(|l| l.contains("setp.ge.u64 ")).expect("a bound");
        let relaxed = |p: &str| {
            p.lines()
                .enumerate()
                .map(|(n, l)| if n == bound { l.replacen("setp.ge.u64 ", "setp.gt.u64 ", 1) } else { l.to_string() })
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert!(caught(k, relaxed), "{k:?} bound");
        assert!(caught(k, |p| p.replacen("%ctaid.x;", "0;", 1)), "{k:?} block index");
    }
}

/// Every address's element size: f32 addresses scale by 4, the stride and
/// shape tables by 8.
#[test]
fn nudging_an_element_size_is_caught() {
    for (k, f32_sites, u64_sites) in [(K::Bias, 3, 0), (K::Gather, 3, 0), (K::Copy, 2, 3), (K::Slice, 2, 3)] {
        let p = kir_ptx(k);
        let count = |size: &str| p.lines().filter(|l| l.contains("mul.lo.u64") && l.ends_with(size)).count();
        assert_eq!((count(", 4;"), count(", 8;")), (f32_sites, u64_sites), "{k:?}");
        for i in 0..f32_sites {
            assert!(caught(k, |p| nudge(p, ", 4;", ", 8;", i)), "{k:?} f32 address {i}");
        }
        for i in 0..u64_sites {
            assert!(caught(k, |p| nudge(p, ", 8;", ", 4;", i)), "{k:?} table address {i}");
        }
    }
}

#[test]
fn the_index_arithmetic_is_caught() {
    // bias: the column is i % cols.
    assert!(caught(K::Bias, |p| p.replacen("rem.u64 ", "div.u64 ", 1)), "bias modulus");
    assert!(caught(K::Bias, |p| p.replacen("add.f32 ", "sub.f32 ", 1)), "bias add");
    // gather: o = i / inner, k = i % inner, and the out-of-range test.
    assert!(caught(K::Gather, |p| p.replacen("div.u64 ", "rem.u64 ", 1)), "gather quotient");
    assert!(caught(K::Gather, |p| p.replacen("rem.u64 ", "div.u64 ", 1)), "gather remainder");
    let ges = kir_ptx(K::Gather).matches("setp.ge.u64 ").count();
    assert_eq!(ges, 2, "the bound and the index test");
    assert!(caught(K::Gather, |p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", 1)), "index test at the bound");
}

#[test]
fn the_walk_is_caught() {
    for k in [K::Copy, K::Slice] {
        let p = kir_ptx(k);
        // The zero-stride skip, inverted.
        assert_eq!(p.matches("setp.eq.u64 ").count(), if k == K::Slice { 2 } else { 1 }, "{k:?}");
        assert!(caught(k, |p| nudge(p, "setp.eq.u64 ", "setp.ne.u64 ", 0)), "{k:?} zero-stride skip");
        // The walk's quotient, remainder and broadcast modulus.
        assert_eq!(p.matches("rem.u64 ").count(), 2, "{k:?}");
        assert!(caught(k, |p| p.replacen("div.u64 ", "rem.u64 ", 1)), "{k:?} coordinate");
        for i in 0..2 {
            assert!(caught(k, |p| nudge(p, "rem.u64 ", "div.u64 ", i)), "{k:?} rem {i}");
        }
        // The increment: `d + 1`.
        let inc = p.lines().position(|l| l.contains("mov.u64 ") && l.ends_with(", 1;")).expect("the one");
        let skip = |p: &str| {
            p.lines()
                .enumerate()
                .map(|(n, l)| if n == inc { l.replacen(", 1;", ", 2;", 1) } else { l.to_string() })
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert!(caught(k, skip), "{k:?} increment");
    }
    // The slice's dimension test and its offset.
    assert!(caught(K::Slice, |p| nudge(p, "setp.eq.u64 ", "setp.ne.u64 ", 1)), "slice dimension test");
    assert!(caught(K::Slice, |p| p.replacen("[param_slice_start]", "[param_slice_dim]", 1)), "slice start");
}
