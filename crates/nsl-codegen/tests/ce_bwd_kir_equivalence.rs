//! The differential equivalence gate for the GPU cross-entropy backward
//! kernels from `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5):
//! `nsl_ce_bwd_count_f32` and `nsl_ce_bwd_finish_f32`, now built by
//! `nsl_kir::kernels::ce_bwd`.
//!
//! This file runs the frozen hand modules (`tests/fixtures/ce_bwd_hand.rs`)
//! and the KIR ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`). The count kernel runs as the runtime
//! launches it, one 256-thread block, under all four schedules. The finish
//! kernel runs on the runtime's 256-thread block and on a 32-thread one,
//! with one block more than the work needs, under two schedules.
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory. The targets are read both ways (f32 and s32). As f32 they
//!    cover fractions, `-0.5` (which truncates to a valid `0`), NaN (`0`),
//!    and values past the s32 range (which saturate). As s32 they cover
//!    `-1`, `-100` and `i32::MIN`. The gradient scale comes from the device
//!    scalar and from the immediate.
//! 2. **Correctness**: the count is `max(#{t >= 0}, 1)` as f32. The finish
//!    kernel leaves `0` for an invalid row, and otherwise
//!    `(sm - [j == t]) · (go / denom)`, each step rounded on its own. It
//!    never reads `gop` when the scale is the immediate (the gate passes a
//!    null pointer, so a read faults), and nothing past the softmax is
//!    written.
//! 3. **The gate bites**: each bound, the stride, the tree's first half and
//!    halving, both barriers, the valid test, the f32/s32 choice, the
//!    increment, the `max`, the block index, the row/column split, the
//!    one-hot test, its `1` and subtraction, the scale's source, divide and
//!    multiply, the invalid rows' `0`, every element size and every 64-bit
//!    add are caught.

use std::collections::HashMap;

use nsl_kir::kernels::ce_bwd::{ptx, CeBwdOp, CE_BWD_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/ce_bwd_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 64;
const POISON: u32 = 0x7FA5_A5A5;
const SM: u64 = 0x1000_0000;
const TGT: u64 = 0x2000_0000;
const SCRATCH: u64 = 0x3000_0000;
const GOP: u64 = 0x4000_0000;

fn kir_ptx(op: CeBwdOp) -> String {
    String::from_utf8(ptx(op)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(op: CeBwdOp) -> String {
    match op {
        CeBwdOp::Count => hand::CE_BWD_COUNT_F32_PTX,
        CeBwdOp::Finish => hand::CE_BWD_FINISH_F32_PTX,
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

/// `n` targets for a `cols`-wide row, as the four bytes the kernel loads:
/// mostly in `0..cols` (some one past), the rest the edge cases of each
/// reading.
fn targets(n: usize, cols: usize, as_i32: bool, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            let k = lcg(&mut s) % 16;
            let c = (lcg(&mut s) % (cols as u64 + 1)) as i32;
            if as_i32 {
                match k {
                    0 => (-1i32) as u32,
                    1 => (-100i32) as u32,
                    2 => i32::MIN as u32,
                    _ => c as u32,
                }
            } else {
                match k {
                    0 => (-1.0f32).to_bits(),
                    1 => (-0.5f32).to_bits(),
                    2 => f32::NAN.to_bits(),
                    3 => 3.0e9f32.to_bits(),
                    4 => (-3.0e9f32).to_bits(),
                    5 => (c as f32 + 0.75).to_bits(),
                    6 => (-7.25f32).to_bits(),
                    _ => (c as f32).to_bits(),
                }
            }
        })
        .collect()
}

/// A target as the kernels read it: truncated toward zero from f32
/// (saturating, NaN to 0), or the s32 itself.
fn read_target(w: u32, as_i32: bool) -> i32 {
    if as_i32 {
        w as i32
    } else {
        f32::from_bits(w) as i32
    }
}

const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];
const ALL_ORDERS: [Order; 4] = [Order::Ascending, Order::Descending, Order::WarpsAscending, Order::WarpsDescending];

// ---------------------------------------------------------------------------
// The count kernel
// ---------------------------------------------------------------------------

const COUNTS: [usize; 7] = [0, 1, 5, 255, 256, 257, 700];

struct CountCase {
    tgt: Vec<u32>,
    as_i32: bool,
}

fn count_cases() -> Vec<CountCase> {
    let mut v = vec![];
    for (i, &n) in COUNTS.iter().enumerate() {
        for as_i32 in [false, true] {
            v.push(CountCase { tgt: targets(n, 9, as_i32, 11 + i as u64), as_i32 });
        }
    }
    // Every target invalid: the count is clamped to 1.
    v.push(CountCase { tgt: vec![(-1i32) as u32; 300], as_i32: true });
    v
}

/// All of global memory after one launch: `[targets + tail, scratch]`.
fn run_count(ptx: &str, c: &CountCase, order: Order) -> Vec<u32> {
    let prog = parse(ptx);
    let mut tgt = c.tgt.clone();
    tgt.extend(std::iter::repeat_n(POISON, TAIL));
    let mut global = vec![Segment { base: TGT, bytes: le32(&tgt) }, Segment { base: SCRATCH, bytes: le32(&[POISON; 4]) }];
    let args: HashMap<String, u64> = CeBwdOp::Count
        .param_names()
        .iter()
        .zip([TGT, c.tgt.len() as u64, u64::from(c.as_i32), SCRATCH])
        .map(|(p, v)| (p.to_string(), v))
        .collect();
    let mut l = Launch {
        prog: &prog,
        args: &args,
        global: &mut global,
        shared: vec![0; CE_BWD_BLOCK as usize * 4],
        ctaid: 0,
        ctaid_y: 0,
        nctaid_y: 1,
        ntid: CE_BWD_BLOCK,
        steps: 0,
    };
    run_cta(&mut l, order);
    global.iter().flat_map(|s| words(&s.bytes)).collect()
}

#[test]
fn the_count_kernels_agree() {
    let (hand, kir) = (hand_ptx(CeBwdOp::Count), kir_ptx(CeBwdOp::Count));
    for c in count_cases() {
        for order in ALL_ORDERS {
            assert!(run_count(&hand, &c, order) == run_count(&kir, &c, order), "n={} i32={} {order:?}", c.tgt.len(), c.as_i32);
        }
    }
}

#[test]
fn the_count_is_the_clamped_number_of_valid_targets() {
    for which in [hand_ptx(CeBwdOp::Count), kir_ptx(CeBwdOp::Count)] {
        for c in count_cases() {
            let got = run_count(&which, &c, Order::Ascending);
            let n = c.tgt.len();
            let valid = c.tgt.iter().filter(|&&w| read_target(w, c.as_i32) >= 0).count();
            assert_eq!(got[n + TAIL], (valid.max(1) as f32).to_bits(), "n={n} i32={}", c.as_i32);
            assert_eq!(&got[n + TAIL + 1..], &[POISON; 3], "wrote past scratch[0]");
            assert_eq!(&got[..n], &c.tgt[..], "targets untouched");
        }
    }
}

// ---------------------------------------------------------------------------
// The finish kernel
// ---------------------------------------------------------------------------

/// `(rows, cols)`.
const SHAPES: [(usize, usize); 5] = [(1, 1), (3, 5), (7, 33), (5, 100), (2, 300)];
const BLOCKS: [u32; 2] = [CE_BWD_BLOCK, 32];
const DENOM: f32 = 3.0;
const GO_DEV: f32 = 0.37;
const GO_IMM: f32 = -1.7;

struct FinishCase {
    rows: usize,
    cols: usize,
    sm: Vec<u32>,
    tgt: Vec<u32>,
    as_i32: bool,
    dev_go: bool,
}

fn finish_cases() -> Vec<FinishCase> {
    let mut v = vec![];
    for (i, &(rows, cols)) in SHAPES.iter().enumerate() {
        for (as_i32, dev_go) in [(false, false), (false, true), (true, false), (true, true)] {
            let mut s = 100 + i as u64;
            let sm = (0..rows * cols)
                .map(|_| {
                    let e = (lcg(&mut s) % 20) as i32 - 19;
                    (2f32.powi(e) * (1.0 + (lcg(&mut s) % 1024) as f32 / 1024.0)).to_bits()
                })
                .collect();
            let tgt = targets(rows, cols, as_i32, 7 + i as u64 * 3 + u64::from(as_i32));
            v.push(FinishCase { rows, cols, sm, tgt, as_i32, dev_go });
        }
    }
    v
}

/// All of global memory after the launch: `[sm + tail, targets, scratch,
/// gop]`. With the immediate scale, `gop` is null and there is no segment
/// for it.
fn run_finish(ptx: &str, c: &FinishCase, block: u32, order: Order) -> Vec<u32> {
    let prog = parse(ptx);
    let total = c.rows * c.cols;
    let mut sm = c.sm.clone();
    sm.extend(std::iter::repeat_n(POISON, TAIL));
    let mut global = vec![
        Segment { base: SM, bytes: le32(&sm) },
        Segment { base: TGT, bytes: le32(&c.tgt) },
        Segment { base: SCRATCH, bytes: le32(&[DENOM.to_bits(), POISON]) },
    ];
    if c.dev_go {
        global.push(Segment { base: GOP, bytes: le32(&[GO_DEV.to_bits(), POISON]) });
    }
    let gop = if c.dev_go { GOP } else { 0 };
    let vals = [
        SM,
        TGT,
        SCRATCH,
        gop,
        u64::from(GO_IMM.to_bits()),
        u64::from(c.dev_go),
        u64::from(c.as_i32),
        total as u64,
        c.cols as u64,
    ];
    let args: HashMap<String, u64> =
        CeBwdOp::Finish.param_names().iter().zip(vals).map(|(p, v)| (p.to_string(), v)).collect();
    let grid = total.div_ceil(block as usize) as u32 + 1;
    let mut ctas: Vec<u32> = (0..grid).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for ctaid in ctas {
        let mut l = Launch { prog: &prog, args: &args, global: &mut global, shared: vec![], ctaid, ctaid_y: 0, nctaid_y: 1, ntid: block, steps: 0 };
        run_cta(&mut l, order);
    }
    global.iter().flat_map(|s| words(&s.bytes)).collect()
}

#[test]
fn the_finish_kernels_agree() {
    let (hand, kir) = (hand_ptx(CeBwdOp::Finish), kir_ptx(CeBwdOp::Finish));
    for c in finish_cases() {
        for block in BLOCKS {
            for order in ORDERS {
                assert!(
                    run_finish(&hand, &c, block, order) == run_finish(&kir, &c, block, order),
                    "{}x{} i32={} dev={} {block} {order:?}",
                    c.rows,
                    c.cols,
                    c.as_i32,
                    c.dev_go
                );
            }
        }
    }
}

#[test]
fn the_finish_is_the_scaled_gradient() {
    for which in [hand_ptx(CeBwdOp::Finish), kir_ptx(CeBwdOp::Finish)] {
        for c in finish_cases() {
            let got = run_finish(&which, &c, CE_BWD_BLOCK, Order::Ascending);
            let go = if c.dev_go { GO_DEV } else { GO_IMM };
            let q = go / DENOM;
            for i in 0..c.rows {
                let t = read_target(c.tgt[i], c.as_i32);
                for j in 0..c.cols {
                    let at = i * c.cols + j;
                    let want = if t < 0 {
                        0.0f32
                    } else {
                        let v = f32::from_bits(c.sm[at]);
                        let v = if j as i64 == i64::from(t) { v - 1.0 } else { v };
                        v * q
                    };
                    assert_eq!(got[at], want.to_bits(), "{}x{} ({i}, {j}) t={t}", c.rows, c.cols);
                }
            }
            let total = c.rows * c.cols;
            assert!(got[total..total + TAIL].iter().all(|&w| w == POISON), "wrote past sm");
        }
    }
}

/// The data reaches every case the kernels distinguish: valid and invalid
/// rows, a one-hot hit, targets past the row, and more than one block.
#[test]
fn the_data_covers_every_case() {
    let cs = finish_cases();
    for as_i32 in [false, true] {
        let reads: Vec<(i32, usize)> =
            cs.iter().filter(|c| c.as_i32 == as_i32).flat_map(|c| c.tgt.iter().map(move |&w| (read_target(w, as_i32), c.cols))).collect();
        assert!(reads.iter().any(|&(t, _)| t < 0), "i32={as_i32}: an invalid row");
        assert!(reads.iter().any(|&(t, c)| t >= 0 && (t as usize) < c), "i32={as_i32}: a hit");
        assert!(reads.iter().any(|&(t, c)| t >= 0 && t as usize >= c), "i32={as_i32}: a target past the row");
        assert!(reads.iter().any(|&(t, _)| t == 0), "i32={as_i32}: a zero target");
    }
    assert!(cs.iter().any(|c| c.rows * c.cols > CE_BWD_BLOCK as usize));
    let f: Vec<u32> = count_cases().into_iter().filter(|c| !c.as_i32).flat_map(|c| c.tgt).collect();
    for edge in [(-0.5f32).to_bits(), f32::NAN.to_bits(), 3.0e9f32.to_bits(), (-3.0e9f32).to_bits()] {
        assert!(f.contains(&edge), "{:?}", f32::from_bits(edge));
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on any case,
/// block or schedule (or faults).
fn caught(op: CeBwdOp, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(op);
    let mutant = mutate(&kir_ptx(op));
    let differs = |a: &dyn Fn() -> Vec<u32>, b: &dyn Fn() -> Vec<u32>| {
        let expect = a();
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(b)) {
            Ok(r) => r != expect,
            Err(_) => true,
        }
    };
    match op {
        CeBwdOp::Count => count_cases().iter().any(|c| {
            ALL_ORDERS.into_iter().any(|order| differs(&|| run_count(&hand, c, order), &|| run_count(&mutant, c, order)))
        }),
        CeBwdOp::Finish => finish_cases().iter().any(|c| {
            BLOCKS.into_iter().any(|block| {
                ORDERS.into_iter().any(|order| {
                    differs(&|| run_finish(&hand, c, block, order), &|| run_finish(&mutant, c, block, order))
                })
            })
        }),
    }
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

/// `selp.T d, a, b, p;` as `selp.T d, b, a, p;`.
fn swap_selp(ptx: &str, ty: &str, i: usize) -> String {
    let op = format!("selp.{ty} ");
    edit(ptx, &op, i, |l| {
        let (head, operands) = l.split_once(op.as_str()).expect("selp");
        let ops: Vec<&str> = operands.trim_end_matches(';').split(',').map(str::trim).collect();
        format!("{head}{op}{}, {}, {}, {};", ops[0], ops[2], ops[1], ops[3])
    })
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for op in CeBwdOp::ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

/// The count loop's bound and the finish kernel's thread bound.
#[test]
fn relaxing_a_bound_is_caught() {
    for op in CeBwdOp::ALL {
        assert_eq!(kir_ptx(op).matches("setp.ge.u32 ").count(), if op == CeBwdOp::Count { 2 } else { 1 }, "{op:?}");
        assert!(caught(op, |p| nudge(p, "setp.ge.u32 ", "setp.gt.u32 ", 0)), "{op:?}");
    }
}

/// A target's reading: the f32/s32 choice, its flag, and the valid test
/// (a `0` target is valid, a negative s32 is not).
#[test]
fn the_target_reading_is_pinned() {
    for op in CeBwdOp::ALL {
        assert!(caught(op, |p| swap_selp(p, "b32", 0)), "{op:?}: the reading");
        assert!(caught(op, |p| nudge(p, "setp.ne.u32 ", "setp.eq.u32 ", 0)), "{op:?}: the flag");
        let valid = if op == CeBwdOp::Count { "setp.ge.s32 " } else { "setp.lt.s32 " };
        let strict = if op == CeBwdOp::Count { "setp.gt.s32 " } else { "setp.le.s32 " };
        let unsigned = if op == CeBwdOp::Count { "setp.ge.u32 " } else { "setp.lt.u32 " };
        assert!(caught(op, |p| nudge(p, valid, strict, 0)), "{op:?}: zero is valid");
        assert!(caught(op, |p| nudge(p, valid, unsigned, 0)), "{op:?}: signed");
    }
}

/// Every address's element size and every 64-bit add.
#[test]
fn every_address_is_pinned() {
    for op in CeBwdOp::ALL {
        let p = kir_ptx(op);
        let sizes = p.lines().filter(|l| l.contains("mul.lo.u64") && l.ends_with(", 4;")).count();
        assert_eq!(sizes, if op == CeBwdOp::Count { 5 } else { 4 }, "{op:?}");
        for i in 0..sizes {
            assert!(caught(op, |p| edit(p, "mul.lo.u64", i, |l| l.replacen(", 4;", ", 8;", 1))), "{op:?} size {i}");
        }
        let adds = p.matches("add.u64 ").count();
        assert_eq!(adds, sizes, "{op:?}");
        for i in 0..adds {
            assert!(caught(op, |p| drop_op(p, "add.u64 ", "u64", i)), "{op:?} add {i}");
        }
    }
}

/// The count: the stride, the increment, the tree's first half and its
/// halving, the pairwise add, both barriers and the clamp.
#[test]
fn the_count_reduction_is_pinned() {
    let op = CeBwdOp::Count;
    assert!(caught(op, |p| p.replacen(", 256;", ", 128;", 1)), "stride");
    assert!(caught(op, |p| swap_selp(p, "b32", 1)), "increment");
    assert!(caught(op, |p| p.replacen(", 128;", ", 64;", 1)), "first half");
    assert!(caught(op, |p| drop_op(p, "shr.u32 ", "u32", 0)), "halving");
    let adds = kir_ptx(op).matches("add.u32 ").count();
    assert_eq!(adds, 4, "count, stride, partner, pair");
    for i in 0..adds {
        assert!(caught(op, |p| drop_op(p, "add.u32 ", "u32", i)), "add.u32 {i}");
    }
    assert_eq!(kir_ptx(op).matches("bar.sync 0;").count(), 2);
    for i in 0..2 {
        assert!(caught(op, |p| edit(p, "bar.sync 0;", i, |_| String::new())), "barrier {i}");
    }
    assert!(caught(op, |p| p.replacen("max.u32 ", "min.u32 ", 1)), "clamp");
}

/// The finish: the block index, the row/column split, the one-hot test and
/// its subtraction, the scale's source, the divide and multiply, and the
/// invalid rows' `0`.
#[test]
fn the_finish_arithmetic_is_pinned() {
    let op = CeBwdOp::Finish;
    assert!(caught(op, |p| p.replacen("%ctaid.x", "0", 1)), "block index");
    assert!(caught(op, |p| p.replacen("div.u32 ", "rem.u32 ", 1)), "row");
    assert!(caught(op, |p| drop_op(p, "mul.lo.u32 ", "u32", 1)), "row · cols");
    assert!(caught(op, |p| drop_op(p, "sub.u32 ", "u32", 0)), "column");
    assert_eq!(kir_ptx(op).matches("setp.eq.u32 ").count(), 2, "one-hot, scale source");
    assert!(caught(op, |p| nudge(p, "setp.eq.u32 ", "setp.ne.u32 ", 0)), "one-hot test");
    assert!(caught(op, |p| nudge(p, "setp.eq.u32 ", "setp.ne.u32 ", 1)), "scale source");
    assert!(caught(op, |p| p.replacen("0f3F800000", "0f40000000", 1)), "the one");
    assert!(caught(op, |p| p.replacen("sub.rn.f32 ", "add.rn.f32 ", 1)), "subtraction");
    assert!(caught(op, |p| swap_selp(p, "f32", 0)), "one-hot select");
    assert!(caught(op, |p| p.replacen("div.rn.f32 ", "mul.rn.f32 ", 1)), "divide");
    assert!(caught(op, |p| p.replacen("mul.rn.f32 ", "add.rn.f32 ", 1)), "multiply");
    assert!(caught(op, |p| p.replacen("0f00000000", "0f3F800000", 1)), "invalid rows' 0");
}
