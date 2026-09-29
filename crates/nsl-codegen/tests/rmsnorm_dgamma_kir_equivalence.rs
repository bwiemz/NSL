//! The differential equivalence gate for the fused RMSNorm gamma-backward
//! kernels from `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5):
//! `nsl_rmsnorm_rinv_rows_f32` and `nsl_rmsnorm_dgamma_f32`, now built by
//! `nsl_kir::kernels::rmsnorm_dgamma`.
//!
//! This file runs the frozen hand modules
//! (`tests/fixtures/rmsnorm_dgamma_hand.rs`) and the KIR ones side by side
//! on the cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`),
//! on the runtime's 256-thread block and on a 32-thread one, with one block
//! more than the work needs, under two schedules:
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory, over single-row, single-column and ragged shapes, with inputs
//!    whose magnitudes span 2^-10 to 2^10, zeros and signed zeros.
//! 2. **Correctness**: `rinv[r]` is `1 / sqrt(Σ fma(x, x) / f32(cols) +
//!    eps)`, each step correctly rounded, and `dgamma[j]` is
//!    `Σ_i (dy · x) · rinv[i]` in row order, bit for bit; nothing past the
//!    output is written.
//! 3. **The gate bites**: each bound, the block index, every element size,
//!    every 64-bit add and stride, the step, the accumulator's start, and
//!    every float operation are caught.

use std::collections::HashMap;

use nsl_kir::kernels::elementwise::ELEMENTWISE_BLOCK;
use nsl_kir::kernels::rmsnorm_dgamma::{ptx, RmsNormDgammaOp};

#[allow(dead_code)]
#[path = "fixtures/rmsnorm_dgamma_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 64;
const POISON: u32 = 0x7FA5_A5A5;
const DY: u64 = 0x1000_0000;
const X: u64 = 0x2000_0000;
const RINV: u64 = 0x3000_0000;
const OUT: u64 = 0x4000_0000;
const EPS: f32 = 1e-5;

fn kir_ptx(op: RmsNormDgammaOp) -> String {
    String::from_utf8(ptx(op)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(op: RmsNormDgammaOp) -> String {
    match op {
        RmsNormDgammaOp::RinvRows => hand::RMSNORM_RINV_ROWS_F32_PTX,
        RmsNormDgammaOp::Dgamma => hand::RMSNORM_DGAMMA_F32_PTX,
    }
    .trim_end_matches('\0')
    .trim_end()
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

/// `n` values from 2^-10 to 2^10 of both signs, with a zero and a `-0.0`
/// every 13 and 17 elements.
fn values(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n)
        .map(|k| {
            if k % 13 == 4 {
                return 0;
            }
            if k % 17 == 9 {
                return (-0.0f32).to_bits();
            }
            let e = (lcg(&mut s) % 21) as i32 - 10;
            let m = 1.0 + (lcg(&mut s) % 1024) as f32 / 1024.0;
            let sign = if lcg(&mut s).is_multiple_of(2) { 1.0 } else { -1.0 };
            (sign * m * 2f32.powi(e)).to_bits()
        })
        .collect()
}

/// `(rows, cols)`: single row, single column, ragged against both blocks.
const SHAPES: [(usize, usize); 5] = [(1, 1), (1, 40), (37, 1), (9, 33), (70, 13)];
const BLOCKS: [u32; 2] = [ELEMENTWISE_BLOCK, 32];
const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

struct Data {
    dy: Vec<u32>,
    x: Vec<u32>,
    rinv: Vec<u32>,
}

fn data(rows: usize, cols: usize, seed: u64) -> Data {
    let rinv = values(rows, seed + 200).into_iter().map(|w| f32::from_bits(w).abs().to_bits()).collect();
    Data { dy: values(rows * cols, seed), x: values(rows * cols, seed + 100), rinv }
}

fn run(op: RmsNormDgammaOp, ptx: &str, (rows, cols): (usize, usize), d: &Data, block: u32, order: Order) -> Vec<u32> {
    let prog = parse(ptx);
    let n_out = match op {
        RmsNormDgammaOp::RinvRows => rows,
        RmsNormDgammaOp::Dgamma => cols,
    };
    let mut global = vec![
        Segment { base: DY, bytes: le32(&d.dy) },
        Segment { base: X, bytes: le32(&d.x) },
        Segment { base: RINV, bytes: le32(&d.rinv) },
        Segment { base: OUT, bytes: le32(&vec![POISON; n_out + TAIL]) },
    ];
    let (rows, cols) = (rows as u64, cols as u64);
    let values: Vec<u64> = match op {
        RmsNormDgammaOp::RinvRows => vec![X, OUT, rows, cols, EPS.to_bits() as u64],
        RmsNormDgammaOp::Dgamma => vec![DY, X, RINV, OUT, rows, cols],
    };
    let args: HashMap<String, u64> = op.param_names().iter().zip(values).map(|(p, v)| (p.to_string(), v)).collect();
    let grid = n_out.div_ceil(block as usize) as u32 + 1;
    let mut ctas: Vec<u32> = (0..grid).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for ctaid in ctas {
        let mut l = Launch { prog: &prog, args: &args, global: &mut global, shared: vec![], ctaid, ctaid_y: 0, nctaid_x: 0, nctaid_y: 1, ntid: block, steps: 0 };
        run_cta(&mut l, order);
    }
    // Every segment but the output must come back untouched.
    assert_eq!(words(&global[0].bytes), d.dy);
    assert_eq!(words(&global[1].bytes), d.x);
    assert_eq!(words(&global[2].bytes), d.rinv);
    words(&global[3].bytes)
}

#[test]
fn the_kernels_agree_and_are_the_formulas() {
    let f = f32::from_bits;
    for op in RmsNormDgammaOp::ALL {
        let (hand, kir) = (hand_ptx(op), kir_ptx(op));
        for (n, &shape) in SHAPES.iter().enumerate() {
            let (rows, cols) = shape;
            let d = data(rows, cols, n as u64 + 1);
            for block in BLOCKS {
                for order in ORDERS {
                    assert!(run(op, &hand, shape, &d, block, order) == run(op, &kir, shape, &d, block, order), "{op:?} {shape:?} {block} {order:?}");
                }
            }
            let out = run(op, &kir, shape, &d, ELEMENTWISE_BLOCK, Order::Ascending);
            let want: Vec<u32> = match op {
                RmsNormDgammaOp::RinvRows => (0..rows)
                    .map(|r| {
                        let s = (0..cols).fold(0.0f32, |s, j| {
                            let v = f(d.x[r * cols + j]);
                            v.mul_add(v, s)
                        });
                        (1.0f32 / (s / cols as f32 + EPS).sqrt()).to_bits()
                    })
                    .collect(),
                RmsNormDgammaOp::Dgamma => (0..cols)
                    .map(|j| (0..rows).fold(0.0f32, |a, i| a + f(d.dy[i * cols + j]) * f(d.x[i * cols + j]) * f(d.rinv[i])).to_bits())
                    .collect(),
            };
            assert_eq!(&out[..want.len()], &want[..], "{op:?} {shape:?}");
            assert!(out[want.len()..].iter().all(|&w| w == POISON), "{op:?} {shape:?}: wrote past the output");
        }
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutate(kir)` is told apart from the hand kernel on any case,
/// block or schedule (or faults).
fn caught(op: RmsNormDgammaOp, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(op);
    let mutant = mutate(&kir_ptx(op));
    SHAPES.iter().enumerate().any(|(n, &shape)| {
        let d = data(shape.0, shape.1, n as u64 + 1);
        BLOCKS.into_iter().any(|block| {
            ORDERS.into_iter().any(|order| {
                let expect = run(op, &hand, shape, &d, block, order);
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(op, &mutant, shape, &d, block, order))) {
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

/// `op.ty d, a, b;` as `mov.ty d, a;`: the operation dropped.
fn drop_op(ptx: &str, op: &str, ty: &str, i: usize) -> String {
    edit(ptx, op, i, |l| {
        let (head, operands) = l.split_once(op).expect("the op");
        let mut ops: Vec<&str> = operands.trim_end_matches(';').split(',').map(str::trim).collect();
        ops.truncate(2);
        format!("{head}mov.{ty} {};", ops.join(", "))
    })
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for op in RmsNormDgammaOp::ALL {
        assert!(!caught(op, |p| p.to_string()), "{op:?}");
    }
}

/// The thread bound and the loop bound.
#[test]
fn relaxing_a_bound_is_caught() {
    for op in RmsNormDgammaOp::ALL {
        assert_eq!(kir_ptx(op).matches("setp.ge.u64 ").count(), 2, "{op:?}");
        for i in 0..2 {
            assert!(caught(op, |p| nudge(p, "setp.ge.u64 ", "setp.gt.u64 ", i)), "{op:?} bound {i}");
        }
    }
}

#[test]
fn the_block_index_is_caught() {
    for op in RmsNormDgammaOp::ALL {
        assert!(caught(op, |p| p.replacen("%ctaid.x", "0", 1)), "{op:?}");
    }
}

/// Every address's element size: the loads and the store.
#[test]
fn nudging_an_element_size_is_caught() {
    for op in RmsNormDgammaOp::ALL {
        let p = kir_ptx(op);
        let sites: Vec<usize> =
            p.lines().enumerate().filter(|(_, l)| l.contains("mul.lo.u64") && l.ends_with(", 4;")).map(|(n, _)| n).collect();
        assert_eq!(sites.len(), if op == RmsNormDgammaOp::Dgamma { 4 } else { 2 }, "{op:?}");
        for at in sites {
            let mutate = |p: &str| {
                p.lines().enumerate().map(|(n, l)| if n == at { l.replacen(", 4;", ", 8;", 1) } else { l.to_string() }).collect::<Vec<_>>().join("\n")
                    + "\n"
            };
            assert!(caught(op, mutate), "{op:?} element size at line {at}");
        }
    }
}

/// Every 64-bit add (the addresses, the row base or column step, the
/// counter) and every multiply that is not an element size (the row base).
#[test]
fn dropping_an_add_or_a_stride_is_caught() {
    for op in RmsNormDgammaOp::ALL {
        let p = kir_ptx(op);
        let adds = p.matches("add.u64 ").count();
        assert_eq!(adds, if op == RmsNormDgammaOp::Dgamma { 6 } else { 4 }, "{op:?}");
        for i in 0..adds {
            assert!(caught(op, |p| drop_op(p, "add.u64 ", "u64", i)), "{op:?} add {i}");
        }
        let strides: Vec<usize> =
            p.lines().filter(|l| l.contains("mul.lo.u64")).enumerate().filter(|(_, l)| !l.ends_with(", 4;")).map(|(n, _)| n).collect();
        assert_eq!(strides.len(), if op == RmsNormDgammaOp::RinvRows { 1 } else { 0 }, "{op:?}");
        for i in strides {
            assert!(caught(op, |p| drop_op(p, "mul.lo.u64 ", "u64", i)), "{op:?} stride {i}");
        }
    }
}

/// The step is one and the accumulator starts at `+0.0`.
#[test]
fn the_loop_is_pinned() {
    for op in RmsNormDgammaOp::ALL {
        let p = kir_ptx(op);
        assert_eq!(p.matches(", 1;").count(), 1, "{op:?}: the step is the only `1`");
        assert!(caught(op, |p| p.replacen(", 1;", ", 2;", 1)), "{op:?} step");
        assert!(caught(op, |p| p.replacen("0f00000000;", "0f3F800000;", 1)), "{op:?} start");
    }
}

/// `rinv`: the fused square-accumulate, the mean, `+ eps`, the square root
/// and the reciprocal.
#[test]
fn every_rinv_float_operation_is_caught() {
    let op = RmsNormDgammaOp::RinvRows;
    assert!(caught(op, |p| edit(p, "fma.rn.f32 ", 0, |l| {
        let (head, ops) = l.split_once("fma.rn.f32 ").expect("fma");
        let ops: Vec<&str> = ops.trim_end_matches(';').split(',').map(str::trim).collect();
        format!("{head}add.rn.f32 {}, {}, {};", ops[0], ops[1], ops[3])
    })), "sum of values, not squares");
    assert!(caught(op, |p| nudge(p, "div.rn.f32 ", "mul.rn.f32 ", 0)), "the mean");
    assert!(caught(op, |p| nudge(p, "add.rn.f32 ", "sub.rn.f32 ", 0)), "+ eps");
    assert!(caught(op, |p| drop_op(p, "sqrt.rn.f32 ", "f32", 0)), "the square root");
    assert!(caught(op, |p| p.replacen("0f3F800000;", "0f40000000;", 1)), "the reciprocal's 1");
    assert!(caught(op, |p| nudge(p, "div.rn.f32 ", "mul.rn.f32 ", 1)), "the reciprocal");
}

/// `dgamma`: both products and the accumulate.
#[test]
fn every_dgamma_float_operation_is_caught() {
    let op = RmsNormDgammaOp::Dgamma;
    for i in 0..2 {
        assert!(caught(op, |p| nudge(p, "mul.rn.f32 ", "add.rn.f32 ", i)), "product {i}");
    }
    assert!(caught(op, |p| nudge(p, "add.rn.f32 ", "sub.rn.f32 ", 0)), "the accumulate");
}
