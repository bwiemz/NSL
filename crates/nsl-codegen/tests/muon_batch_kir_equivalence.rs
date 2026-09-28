//! The differential equivalence gate for the batched Muon Newton-Schulz
//! kernels from `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5):
//! `nsl_muon_batch_{mom,sumsq,pack,poly,update}_f32`, now built by
//! `nsl_kir::kernels::muon_batch`.
//!
//! This file runs the frozen hand modules (`tests/fixtures/muon_batch_hand.rs`)
//! and the KIR ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`). The elementwise kernels run on the
//! runtime's grid (`ceil(n / 256)` blocks of 256 across, one row of blocks per
//! matrix) plus one block more across. The reduction runs one block per
//! matrix. Each case batches several matrices at scattered addresses through
//! the kernels' pointer tables.
//!
//! 1. **Agreement**: under two schedules, the two kernels leave *the same
//!    bytes* in all of global memory. The shapes are square, wide and tall,
//!    with `n` below, at and above the block; both `nest` and both `tr`
//!    settings are covered, and the values include IEEE corners.
//! 2. **Correctness**: each output is the kernel's formula, rounding every
//!    multiply and add on its own. The sums follow the reduction's fixed
//!    order (a stride-256 slice per thread, then the 128-step tree), and
//!    nothing past a matrix is written.
//! 3. **The gate bites**: each bound, both block indices, every element
//!    size, every float operation, the `nest` and `tr` tests, the transpose
//!    arithmetic, the diagonal test, and the reduction's stride, tree and
//!    barriers are each caught. Poly's quotient read as a remainder is
//!    named as an equivalent mutant (it swaps row and column, and the
//!    diagonal test is symmetric).

use std::collections::HashMap;

use nsl_kir::kernels::muon_batch::{ptx, MuonBatchOp, MUON_BATCH_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/muon_batch_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

use MuonBatchOp::{Mom, Pack, Poly, Sumsq, Update};

const TAIL: usize = 64;
const POISON: u32 = 0x7FA5_A5A5;
const BLOCK: usize = MUON_BATCH_BLOCK as usize;
const ALL: [MuonBatchOp; 5] = MuonBatchOp::ALL;
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

fn kir_ptx(op: MuonBatchOp) -> String {
    String::from_utf8(ptx(op)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(op: MuonBatchOp) -> String {
    match op {
        Mom => hand::MUON_BATCH_MOM_F32_PTX,
        Sumsq => hand::MUON_BATCH_SUMSQ_F32_PTX,
        Pack => hand::MUON_BATCH_PACK_F32_PTX,
        Poly => hand::MUON_BATCH_POLY_F32_PTX,
        Update => hand::MUON_BATCH_UPDATE_F32_PTX,
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

fn f(w: u32) -> f32 {
    f32::from_bits(w)
}

/// f32 bit patterns: a few IEEE corners, then ordinary values in [-2, 2).
fn values(n: usize, seed: u64, corners: bool) -> Vec<u32> {
    const CORNERS: [u32; 6] = [0x0000_0000, 0x8000_0000, 0x7F80_0000, 0x7FC0_0001, 0x0000_0001, 0x3F80_0000];
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|k| {
            if corners && k < CORNERS.len() {
                return CORNERS[(k + seed as usize) % CORNERS.len()];
            }
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            (((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0).to_bits()
        })
        .collect()
}

/// One batched case: `k` matrices of `r × c`, and the flags.
#[derive(Clone, Copy, Debug)]
struct Case {
    k: usize,
    r: usize,
    c: usize,
    nest: bool,
    tr: bool,
    /// Whether the data carries IEEE corners (the reduction's sums are then
    /// NaN or infinite; the other cases keep them finite).
    corners: bool,
}

impl Case {
    fn n(self) -> usize {
        self.r * self.c
    }
}

fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for (i, &(k, r, c)) in [(1, 1, 1), (3, 3, 5), (2, 7, 4), (2, 16, 16), (3, 17, 19), (2, 30, 20)].iter().enumerate() {
        for (nest, tr) in [(false, false), (true, false), (false, true), (true, true)] {
            out.push(Case { k, r, c, nest, tr, corners: i % 2 == 1 });
        }
    }
    out
}

/// Matrix `j`'s data segment base; the matrices sit apart, in no order.
fn mbase(j: usize) -> u64 {
    0x1000_0000 + ((j * 7 + 3) % 11) as u64 * 0x10_0000
}
fn gbase(j: usize) -> u64 {
    0x2000_0000 + ((j * 5 + 1) % 11) as u64 * 0x10_0000
}
const MTAB: u64 = 0x3000_0000;
const GTAB: u64 = 0x3100_0000;
const NORMS: u64 = 0x3200_0000;
const Y: u64 = 0x3300_0000;
const AA: u64 = 0x3400_0000;

/// The parameters the host folds: momentum, eps and the update's decay and
/// step, plus the polynomial's coefficients.
const MU: f32 = 0.95;
const EPS: f32 = 1e-7;
const DECAY: f32 = 0.999;
const STEP: f32 = 0.02;
const NS: [f32; 3] = [3.4445, -4.7750, 2.0315];

struct Data {
    m: Vec<Vec<u32>>,
    g: Vec<Vec<u32>>,
    norms: Vec<u32>,
    y: Vec<u32>,
    aa: Vec<u32>,
}

fn data(case: Case, seed: u64) -> Data {
    let n = case.n();
    let m = (0..case.k).map(|j| values(n, seed * 31 + j as u64, case.corners)).collect();
    let g = (0..case.k).map(|j| values(n, seed * 37 + j as u64 + 100, case.corners)).collect();
    // Norms are positive (a sum of squares); a zero one is the eps floor.
    let norms = (0..case.k).map(|j| if j == 0 { 0.0f32.to_bits() } else { (1.5 + j as f32).to_bits() }).collect();
    let y = values(case.k * n, seed * 41, false);
    let aa = values(case.k * n, seed * 43, false);
    Data { m, g, norms, y, aa }
}

/// One launch of `op` over `case`; returns all of global memory.
fn run(op: MuonBatchOp, text: &str, case: Case, seed: u64, order: Order) -> Vec<Vec<u8>> {
    let d = data(case, seed);
    let n = case.n();
    let prog = parse(text);
    let mut segments = vec![
        (MTAB, le64(&(0..case.k).map(mbase).collect::<Vec<_>>())),
        (GTAB, le64(&(0..case.k).map(gbase).collect::<Vec<_>>())),
        (NORMS, le32(&[d.norms.clone(), vec![POISON; TAIL]].concat())),
    ];
    for j in 0..case.k {
        segments.push((mbase(j), le32(&[d.m[j].clone(), vec![POISON; TAIL]].concat())));
        segments.push((gbase(j), le32(&[d.g[j].clone(), vec![POISON; TAIL]].concat())));
    }
    // The workspace: Y for pack (poisoned: it writes) and update (read);
    // the Gram pair for poly, square in `r`.
    let (y, aa) = match op {
        Pack => (vec![POISON; case.k * n + TAIL], vec![0]),
        Poly => {
            let r2 = case.r * case.r;
            let a = values(case.k * r2, seed * 47, false);
            ([a, vec![POISON; TAIL]].concat(), [values(case.k * r2, seed * 53, false), vec![POISON; TAIL]].concat())
        }
        _ => ([d.y.clone(), vec![POISON; TAIL]].concat(), [d.aa.clone(), vec![POISON; TAIL]].concat()),
    };
    segments.push((Y, le32(&y)));
    segments.push((AA, le32(&aa)));
    let mut global: Vec<Segment> = segments.into_iter().map(|(base, bytes)| Segment { base, bytes }).collect();

    let (u32_, f32_) = (|v: usize| v as u64, |v: f32| v.to_bits() as u64);
    let args: Vec<(&str, u64)> = match op {
        Mom => vec![("mtab", MTAB), ("gtab", GTAB), ("mu", f32_(MU)), ("n", u32_(n))],
        Sumsq => vec![
            ("mtab", MTAB),
            ("gtab", GTAB),
            ("mu", f32_(MU)),
            ("nest", case.nest as u64),
            ("n", u32_(n)),
            ("norms", NORMS),
        ],
        Pack => vec![
            ("mtab", MTAB),
            ("gtab", GTAB),
            ("mu", f32_(MU)),
            ("nest", case.nest as u64),
            ("norms", NORMS),
            ("ybase", Y),
            ("r", u32_(case.r)),
            ("c", u32_(case.c)),
            ("tr", case.tr as u64),
            ("eps", f32_(EPS)),
        ],
        Poly => vec![
            ("abase", Y),
            ("aabase", AA),
            ("nsa", f32_(NS[0])),
            ("nsb", f32_(NS[1])),
            ("nsc", f32_(NS[2])),
            ("rdim", u32_(case.r)),
            ("r2", u32_(case.r * case.r)),
        ],
        Update => vec![
            ("ptab", MTAB),
            ("ybase", Y),
            ("r", u32_(case.r)),
            ("c", u32_(case.c)),
            ("tr", case.tr as u64),
            ("decay", f32_(DECAY)),
            ("step", f32_(STEP)),
        ],
    };
    let args: HashMap<String, u64> = args.into_iter().map(|(k, v)| (k.to_string(), v)).collect();
    let across = match op {
        Sumsq => case.k,
        Poly => (case.r * case.r).div_ceil(BLOCK) + 1,
        _ => n.div_ceil(BLOCK) + 1,
    };
    let rows = if op == Sumsq { 1 } else { case.k };
    let mut ctas: Vec<(u32, u32)> = (0..rows).flat_map(|y| (0..across).map(move |x| (x as u32, y as u32))).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for (x, y) in ctas {
        let mut l = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![0; BLOCK * 4],
            ctaid: x,
            ctaid_y: y,
            nctaid_y: rows as u32,
            ntid: MUON_BATCH_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    global.into_iter().map(|s| s.bytes).collect()
}

fn same(got: u32, want: u32) -> bool {
    got == want || (f(got).is_nan() && f(want).is_nan())
}

// ---------------------------------------------------------------------------
// Agreement and the formulas
// ---------------------------------------------------------------------------

#[test]
fn the_kernels_agree_on_every_case() {
    for op in ALL {
        let (hand, kir) = (hand_ptx(op), kir_ptx(op));
        for (s, case) in cases().into_iter().enumerate() {
            for order in ORDERS {
                let h = run(op, &hand, case, s as u64 + 1, order);
                let k = run(op, &kir, case, s as u64 + 1, order);
                assert!(h == k, "{op:?} {case:?} {order:?}: global memory differs");
            }
        }
    }
}

/// Segment index of matrix `j`'s m, and of the workspace.
fn m_seg(j: usize) -> usize {
    3 + 2 * j
}
fn y_seg(case: Case) -> usize {
    3 + 2 * case.k
}

fn direction(case: Case, m: u32, g: u32) -> f32 {
    if case.nest { f(g) + f(m) * MU } else { f(m) }
}

fn transpose(case: Case, i: usize) -> usize {
    if case.tr { (i % case.c) * case.r + i / case.c } else { i }
}

#[test]
fn mom_is_the_momentum_update() {
    for (s, case) in cases().into_iter().enumerate() {
        let d = data(case, s as u64 + 1);
        let mem = run(Mom, &kir_ptx(Mom), case, s as u64 + 1, Order::Ascending);
        for j in 0..case.k {
            let out = words(&mem[m_seg(j)]);
            for i in 0..case.n() {
                let want = (f(d.m[j][i]) * MU + f(d.g[j][i])).to_bits();
                assert!(same(out[i], want), "{case:?} matrix {j} element {i}");
            }
            assert!(out[case.n()..].iter().all(|&w| w == POISON), "{case:?}: wrote past matrix {j}");
        }
    }
}

#[test]
fn sumsq_is_the_fixed_order_sum_of_squares() {
    for (s, case) in cases().into_iter().enumerate() {
        let d = data(case, s as u64 + 1);
        let mem = run(Sumsq, &kir_ptx(Sumsq), case, s as u64 + 1, Order::Ascending);
        let norms = words(&mem[2]);
        for j in 0..case.k {
            let mut part = [0.0f32; BLOCK];
            for (t, p) in part.iter_mut().enumerate() {
                for i in (t..case.n()).step_by(BLOCK) {
                    let u = direction(case, d.m[j][i], d.g[j][i]);
                    *p += u * u;
                }
            }
            let mut h = BLOCK / 2;
            while h >= 1 {
                for t in 0..h {
                    part[t] += part[t + h];
                }
                h /= 2;
            }
            assert!(same(norms[j], part[0].to_bits()), "{case:?} matrix {j}: {} vs {}", f(norms[j]), part[0]);
        }
        assert!(norms[case.k..].iter().all(|&w| w == POISON), "{case:?}: wrote past the norms");
    }
}

#[test]
fn pack_is_the_normalised_update_in_the_workspace() {
    for (s, case) in cases().into_iter().enumerate() {
        let d = data(case, s as u64 + 1);
        let mem = run(Pack, &kir_ptx(Pack), case, s as u64 + 1, Order::Ascending);
        let y = words(&mem[y_seg(case)]);
        for j in 0..case.k {
            let inv = 1.0 / (f(d.norms[j]).sqrt() + EPS);
            for i in 0..case.n() {
                let want = (direction(case, d.m[j][i], d.g[j][i]) * inv).to_bits();
                let at = j * case.n() + transpose(case, i);
                assert!(same(y[at], want), "{case:?} matrix {j} element {i}");
            }
        }
        assert!(y[case.k * case.n()..].iter().all(|&w| w == POISON), "{case:?}: wrote past the workspace");
    }
}

#[test]
fn poly_is_the_polynomial_combine_with_the_diagonal() {
    for (s, case) in cases().into_iter().enumerate() {
        let seed = s as u64 + 1;
        let r2 = case.r * case.r;
        let a = values(case.k * r2, seed * 47, false);
        let aa = values(case.k * r2, seed * 53, false);
        let mem = run(Poly, &kir_ptx(Poly), case, seed, Order::Ascending);
        let out = words(&mem[y_seg(case)]);
        for o in 0..case.k * r2 {
            let i = o % r2;
            let v = f(a[o]) * NS[1] + f(aa[o]) * NS[2];
            let v = if i / case.r == i % case.r { v + NS[0] } else { v };
            assert!(same(out[o], v.to_bits()), "{case:?} slot {o}");
        }
        assert!(out[case.k * r2..].iter().all(|&w| w == POISON), "{case:?}: wrote past the workspace");
    }
}

#[test]
fn update_is_the_decayed_step() {
    for (s, case) in cases().into_iter().enumerate() {
        let d = data(case, s as u64 + 1);
        let mem = run(Update, &kir_ptx(Update), case, s as u64 + 1, Order::Ascending);
        for j in 0..case.k {
            let out = words(&mem[m_seg(j)]);
            for i in 0..case.n() {
                let o = f(d.y[j * case.n() + transpose(case, i)]);
                let want = (f(d.m[j][i]) * DECAY - STEP * o).to_bits();
                assert!(same(out[i], want), "{case:?} matrix {j} element {i}");
            }
            assert!(out[case.n()..].iter().all(|&w| w == POISON), "{case:?}: wrote past matrix {j}");
        }
    }
}

/// The cases reach every branch: `n` below, at and past the block, more
/// than one block across, both flags, and a tall shape.
#[test]
fn the_cases_cover_the_branches() {
    let cs = cases();
    assert!(cs.iter().any(|c| c.n() < BLOCK) && cs.iter().any(|c| c.n() == BLOCK) && cs.iter().any(|c| c.n() > 2 * BLOCK));
    assert!(cs.iter().any(|c| c.r > c.c) && cs.iter().any(|c| c.r < c.c));
    assert!(cs.iter().any(|c| c.nest && c.tr) && cs.iter().any(|c| !c.nest && !c.tr));
    assert!(cs.iter().any(|c| c.k > 1));
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on any case,
/// under either schedule (or faults).
fn caught(op: MuonBatchOp, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(op);
    let mutant = mutate(&kir_ptx(op));
    cases().into_iter().enumerate().any(|(s, case)| {
        ORDERS.into_iter().any(|order| {
            let expect = run(op, &hand, case, s as u64 + 1, order);
            match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(op, &mutant, case, s as u64 + 1, order))) {
                Ok(mem) => mem != expect,
                Err(_) => true,
            }
        })
    })
}

/// `ptx` with the `i`-th line matching `pick` rewritten by `f`.
fn edit(ptx: &str, pick: impl Fn(&str) -> bool, i: usize, f: impl Fn(&str) -> String) -> String {
    let at = ptx.lines().enumerate().filter(|(_, l)| pick(l)).nth(i).unwrap_or_else(|| panic!("no line {i}")).0;
    ptx.lines().enumerate().map(|(n, l)| if n == at { f(l) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
}

fn nudge(ptx: &str, from: &str, to: &str, i: usize) -> String {
    edit(ptx, |l| l.contains(from), i, |l| l.replacen(from, to, 1))
}

fn count(op: MuonBatchOp, pick: impl Fn(&str) -> bool) -> usize {
    kir_ptx(op).lines().filter(|l| pick(l)).count()
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for op in ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

/// The element bound of every elementwise kernel and the reduction's loop
/// bound.
#[test]
fn relaxing_a_bound_is_caught() {
    for op in [Mom, Pack, Poly, Update] {
        assert_eq!(count(op, |l| l.contains("setp.ge.u32 ")), 1, "{op:?}");
        assert!(caught(op, |p| nudge(p, "setp.ge.u32 ", "setp.gt.u32 ", 0)), "{op:?} bound");
    }
    // The reduction's two accumulation loops, one per `nest` setting.
    assert_eq!(count(Sumsq, |l| l.contains("setp.ge.u64 ")), 2);
    for i in 0..2 {
        assert!(caught(Sumsq, |p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "sumsq loop bound {i}");
    }
}

/// `%ctaid.x` everywhere; the matrix index, `%ctaid.y` (or the reduction's
/// `%ctaid.x`), everywhere.
#[test]
fn ignoring_a_block_index_is_caught() {
    for op in ALL {
        assert!(caught(op, |p| p.replacen("%ctaid.x;", "0;", 1)), "{op:?} ctaid.x");
        if op != Sumsq {
            assert!(caught(op, |p| p.replacen("%ctaid.y;", "0;", 1)), "{op:?} ctaid.y");
        }
    }
}

/// Every address scaling: the pointer tables by 8, the f32 arrays by 4.
#[test]
fn nudging_an_element_size_is_caught() {
    for op in ALL {
        let is_size = |l: &str, s: &str| l.contains("mul.lo.u64") && l.ends_with(s);
        for (size, to) in [(", 8;", ", 4;"), (", 4;", ", 8;")] {
            let sites = count(op, |l| is_size(l, size));
            for i in 0..sites {
                let m = |p: &str| edit(p, |l| is_size(l, size), i, |l| l.replacen(size, to, 1));
                assert!(caught(op, m), "{op:?} size {size} site {i}");
            }
        }
    }
    // The tables: mom, sumsq and pack read two, update one, poly none.
    let tables = |op| count(op, |l| l.contains("mul.lo.u64") && l.ends_with(", 8;"));
    assert_eq!([Mom, Sumsq, Pack, Poly, Update].map(tables), [2, 2, 2, 0, 1]);
}

/// Every rounded float operation, turned into another, is caught.
#[test]
fn every_float_operation_is_caught() {
    let swaps = [("mul.rn.f32 ", "add.rn.f32 "), ("add.rn.f32 ", "sub.rn.f32 "), ("sub.rn.f32 ", "add.rn.f32 "), ("div.rn.f32 ", "mul.rn.f32 ")];
    for op in ALL {
        let mut total = 0;
        for (from, to) in swaps {
            let sites = count(op, |l| l.contains(from));
            total += sites;
            for i in 0..sites {
                assert!(caught(op, |p| nudge(p, from, to, i)), "{op:?} {from}#{i}");
            }
        }
        let want = match op {
            Mom => 2,
            // Two unswitched loops (u = g + mu·x: four; u = x: two) and the tree.
            Sumsq => 7,
            Pack => 5,
            Poly => 4,
            Update => 3,
        };
        assert_eq!(total, want, "{op:?}");
    }
    assert!(caught(Pack, |p| p.replacen("sqrt.rn.f32 ", "abs.f32 ", 1)), "pack sqrt");
}

/// The flags: `nest` (sumsq, pack) and `tr` (pack, update), each inverted.
#[test]
fn inverting_a_flag_is_caught() {
    for op in [Sumsq, Pack] {
        assert_eq!(count(op, |l| l.contains("setp.ne.u32 ")), if op == Sumsq { 2 } else { 1 }, "{op:?}");
        assert!(caught(op, |p| nudge(p, "setp.ne.u32 ", "setp.eq.u32 ", 0)), "{op:?} nest");
    }
    for op in [Pack, Update] {
        assert_eq!(count(op, |l| l.contains("setp.eq.u32 ")), 1, "{op:?}");
        assert!(caught(op, |p| nudge(p, "setp.eq.u32 ", "setp.ne.u32 ", 0)), "{op:?} tr");
    }
}

/// The transpose `(i % c)·r + i / c` in pack and update, and poly's diagonal
/// test `i / rdim == i % rdim`.
///
/// Poly's quotient read as a remainder is an equivalent mutant, named here:
/// for `i = a·rdim + b` it takes `b` for the row and `(a − b)·rdim + b` for
/// the column, which are equal exactly when `a == b`. The diagonal test
/// cannot see the swap, so the quotient is killed by a multiply instead.
#[test]
fn the_index_arithmetic_is_caught() {
    for op in [Pack, Update, Poly] {
        assert_eq!(count(op, |l| l.contains("div.u32 ")), 1, "{op:?}");
        if op == Poly {
            assert!(!caught(op, |p| nudge(p, "div.u32 ", "rem.u32 ", 0)), "named equivalent mutant");
            assert!(caught(op, |p| nudge(p, "div.u32 ", "mul.lo.u32 ", 0)), "{op:?} quotient");
        } else {
            assert!(caught(op, |p| nudge(p, "div.u32 ", "rem.u32 ", 0)), "{op:?} quotient");
        }
        assert!(caught(op, |p| nudge(p, "sub.u32 ", "add.u32 ", 0)), "{op:?} remainder");
    }
    for op in [Pack, Update] {
        // r·c (the count and the slot stride), row·c, col·r; the global
        // index's own `%gid0` multiply aside.
        let is_mul = |l: &str| l.contains("mul.lo.u32 ") && !l.contains("%gid");
        let muls = count(op, is_mul);
        assert_eq!(muls, 3, "{op:?}");
        for i in 0..muls {
            assert!(caught(op, |p| edit(p, is_mul, i, |l| l.replacen("mul.lo.u32 ", "add.u32 ", 1))), "{op:?} mul.lo.u32 #{i}");
        }
    }
    assert!(caught(Poly, |p| nudge(p, "setp.eq.u32 ", "setp.ne.u32 ", 0)), "poly diagonal");
    assert!(caught(Poly, |p| {
        let sel = p.lines().find(|l| l.contains("selp.f32 ")).expect("the select").to_string();
        let (head, ops) = sel.split_once("selp.f32 ").expect("selp");
        let o: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
        p.replacen(&sel, &format!("{head}selp.f32 {}, {}, {}, {};", o[0], o[2], o[1], o[3]), 1)
    }), "poly select operands");
}

/// The reduction: its stride, the tree's first half-width, its step and
/// end test, and each barrier.
#[test]
fn the_reduction_is_caught() {
    let p = kir_ptx(Sumsq);
    assert!(p.contains(", 256;") && p.contains(", 128;"), "{p}");
    assert!(caught(Sumsq, |p| p.replacen(", 256;", ", 128;", 1)), "stride");
    assert!(caught(Sumsq, |p| p.replacen(", 128;", ", 64;", 1)), "tree start");
    assert!(caught(Sumsq, |p| nudge(p, "shr.u32 ", "shl.b32 ", 0)), "tree step");
    assert!(caught(Sumsq, |p| nudge(p, "setp.lt.u32 ", "setp.le.u32 ", 0)), "tree end");
    assert_eq!(p.matches("bar.sync 0;").count(), 2, "{p}");
    for i in 0..2 {
        assert!(caught(Sumsq, |p| edit(p, |l| l.contains("bar.sync 0;"), i, |_| String::new())), "barrier {i}");
    }
}
