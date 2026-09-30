//! The differential equivalence gate for `nsl_bmm_f32` from
//! `nsl_runtime::cuda::fused_kernels` (new-roadmap item 5), now built by
//! `nsl_kir::kernels::bmm`.
//!
//! This file runs the frozen hand module (`tests/fixtures/bmm_hand.rs`) and
//! the KIR one side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`) over whole three-dimensional grids
//! with a spare block along every axis, on the runtime's 16×16 block and
//! 8×4 and 4×8 ones (so a block axis read as the other skips rows or
//! columns), under all four thread schedules and both block orders:
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory.
//! 2. **Correctness**: every `C[z, row, col]` is `Σ_k A[z, row, k] · B[z, k,
//!    col]` summed from `+0` in `k` order by fused multiply-adds
//!    (`f32::mul_add`, one rounding each, as `fma.rn`), with slice `z` of an
//!    operand starting `z · stride` elements in. Nothing past `C` is written
//!    and `A` and `B` are untouched.
//! 3. **The gate bites**: a sweep over the KIR module mutates every
//!    comparison, every integer and float operation, every constant, every
//!    element size, every special register (each grid or block axis read as
//!    another), and the `fma`'s operands, and requires each mutant to be
//!    told apart (every mutant parses).
//!
//! The shapes span one block and several along both axes, `K` of 0 (a row
//! of zeros), and broadcasts of `A`, of `B` and of both (a stride of 0). The
//! data has full mantissas, so a separately rounded multiply and add differ
//! from the `fma`.

use std::collections::HashMap;

use nsl_kir::kernels::bmm::{ptx, BMM_BLOCK, PARAM_NAMES};

#[allow(dead_code)]
#[path = "fixtures/bmm_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 16;
const POISON: u32 = 0x7FA5_A5A5;
const A: u64 = 0x1000_0000;
const B: u64 = 0x2000_0000;
const C: u64 = 0x3000_0000;
const ORDERS: [Order; 4] = [Order::Ascending, Order::Descending, Order::WarpsAscending, Order::WarpsDescending];
const BLOCKS: [[u32; 2]; 3] = [[BMM_BLOCK, BMM_BLOCK], [8, 4], [4, 8]];

fn kir_ptx() -> String {
    String::from_utf8(ptx()).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx() -> String {
    hand::BMM_F32_PTX.trim_end_matches('\0').to_string()
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

/// Full-mantissa values in `[-2, 2)`.
fn data(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    (0..n).map(|_| ((lcg(&mut s) as f32 / (1u64 << 31) as f32) * 4.0 - 2.0).to_bits()).collect()
}

/// `(M, N, K, batch, broadcast A, broadcast B)`.
#[derive(Clone, Copy, Debug)]
struct Shape {
    m: usize,
    n: usize,
    k: usize,
    batch: usize,
    bcast_a: bool,
    bcast_b: bool,
}

impl Shape {
    fn stride_a(&self) -> usize {
        if self.bcast_a { 0 } else { self.m * self.k }
    }
    fn stride_b(&self) -> usize {
        if self.bcast_b { 0 } else { self.k * self.n }
    }
    fn stride_c(&self) -> usize {
        self.m * self.n
    }
}

const fn shape(m: usize, n: usize, k: usize, batch: usize, bcast_a: bool, bcast_b: bool) -> Shape {
    Shape { m, n, k, batch, bcast_a, bcast_b }
}

const SHAPES: [Shape; 6] = [
    shape(1, 1, 1, 1, false, false),
    shape(5, 7, 3, 2, false, false),
    shape(17, 18, 4, 2, false, true),
    shape(9, 20, 5, 3, true, false),
    shape(3, 4, 0, 2, false, false),
    shape(6, 5, 7, 2, true, true),
];

struct Case {
    s: Shape,
    a: Vec<u32>,
    b: Vec<u32>,
}

fn cases() -> Vec<Case> {
    SHAPES
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let slices = |bcast: bool| if bcast { 1 } else { s.batch };
            let a = data(s.m * s.k * slices(s.bcast_a), 3 + 5 * i as u64);
            let b = data(s.k * s.n * slices(s.bcast_b), 4 + 5 * i as u64);
            Case { s, a, b }
        })
        .collect()
}

fn with_tail(v: &[u32]) -> Vec<u32> {
    let mut v = v.to_vec();
    v.extend([POISON; TAIL]);
    v
}

/// `[A, B, C + tail]` as words, after the launch.
fn run(ptx: &str, c: &Case, block: [u32; 2], order: Order, reverse: bool) -> Vec<Vec<u32>> {
    let prog = parse(ptx);
    let s = c.s;
    let mut global = vec![
        Segment { base: A, bytes: le32(&with_tail(&c.a)) },
        Segment { base: B, bytes: le32(&with_tail(&c.b)) },
        Segment { base: C, bytes: le32(&vec![POISON; s.batch * s.stride_c() + TAIL]) },
    ];
    let values = [A, B, C, s.m as u64, s.n as u64, s.k as u64, s.batch as u64, s.stride_a() as u64, s.stride_b() as u64, s.stride_c() as u64];
    let base: HashMap<String, u64> = PARAM_NAMES.iter().zip(values).map(|(p, v)| (p.to_string(), v)).collect();
    let grid = [s.n.div_ceil(block[0] as usize) as u32 + 1, s.m.div_ceil(block[1] as usize) as u32 + 1, s.batch as u32 + 1];
    let mut ctas: Vec<[u32; 3]> =
        (0..grid[2]).flat_map(|z| (0..grid[1]).flat_map(move |y| (0..grid[0]).map(move |x| [x, y, z]))).collect();
    if reverse {
        ctas.reverse();
    }
    for [x, y, z] in ctas {
        let mut args = base.clone();
        args.insert("%ctaid.z".to_string(), u64::from(z));
        let mut l = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![],
            ctaid: x,
            ctaid_y: y,
            nctaid_x: 0,
            nctaid_y: grid[1],
            ntid: block[0],
            steps: 0,
        };
        run_cta_2d(&mut l, block[1], order);
    }
    global.iter().map(|g| words(&g.bytes)).collect()
}

#[test]
fn the_kernels_agree() {
    let (hand, kir) = (hand_ptx(), kir_ptx());
    for c in cases() {
        for block in BLOCKS {
            for order in ORDERS {
                for rev in [false, true] {
                    assert!(run(&hand, &c, block, order, rev) == run(&kir, &c, block, order, rev), "{:?} {block:?} {order:?} rev {rev}", c.s);
                }
            }
        }
    }
}

/// The batched product restated; `fused` picks `mul_add` or a separately
/// rounded multiply and add.
fn reference(c: &Case, fused: bool) -> Vec<u32> {
    let s = c.s;
    let mut out = vec![];
    for z in 0..s.batch {
        for row in 0..s.m {
            for col in 0..s.n {
                let mut acc = 0.0f32;
                for k in 0..s.k {
                    let x = f32::from_bits(c.a[z * s.stride_a() + row * s.k + k]);
                    let y = f32::from_bits(c.b[z * s.stride_b() + k * s.n + col]);
                    acc = if fused { x.mul_add(y, acc) } else { x * y + acc };
                }
                out.push(acc.to_bits());
            }
        }
    }
    out
}

#[test]
fn the_kernels_are_the_batched_product() {
    for which in [hand_ptx(), kir_ptx()] {
        for c in cases() {
            for block in BLOCKS {
                let g = run(&which, &c, block, Order::Ascending, false);
                assert_eq!(g[0], with_tail(&c.a), "{:?}: A untouched", c.s);
                assert_eq!(g[1], with_tail(&c.b), "{:?}: B untouched", c.s);
                assert_eq!(g[2], with_tail(&reference(&c, true)), "{:?} {block:?}", c.s);
            }
        }
    }
}

/// The data reaches what the kernel distinguishes: several blocks along
/// both axes, `K = 0`, every broadcast, and sums whose `fma`s round
/// differently from a multiply and an add.
#[test]
fn the_data_covers_every_case() {
    let cs = cases();
    assert!(cs.iter().any(|c| c.s.m > BMM_BLOCK as usize && c.s.n > BMM_BLOCK as usize));
    assert!(cs.iter().any(|c| c.s.k == 0));
    assert!(cs.iter().any(|c| c.s.bcast_a && !c.s.bcast_b) && cs.iter().any(|c| c.s.bcast_b && !c.s.bcast_a));
    assert!(cs.iter().any(|c| c.s.bcast_a && c.s.bcast_b));
    assert!(cs.iter().any(|c| reference(c, true) != reference(c, false)), "the fma's single rounding shows");
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutant` is told apart from the hand kernel on any case, block,
/// schedule or block order (or faults).
fn caught(mutant: &str) -> bool {
    parse(mutant); // a mutant that does not parse would be a hollow kill
    let hand = hand_ptx();
    cases().iter().any(|c| {
        BLOCKS.into_iter().any(|block| {
            ORDERS.into_iter().any(|order| {
                [false, true].into_iter().any(|rev| {
                    let expect = run(&hand, c, block, order, rev);
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(mutant, c, block, order, rev))) {
                        Ok(r) => r != expect,
                        Err(_) => true,
                    }
                })
            })
        })
    })
}

fn operands(rest: &str) -> Vec<String> {
    rest.trim().trim_end_matches(';').split(',').map(|s| s.trim().to_string()).collect()
}

/// The special registers a mutant may read in place of another.
const SPECIALS: [&str; 7] = ["%ctaid.x", "%ctaid.y", "%ctaid.z", "%ntid.x", "%ntid.y", "%tid.x", "%tid.y"];

/// Every mutant of one line: `(description, replacement)`.
fn line_mutants(line: &str, literal_regs: &HashMap<String, String>) -> Vec<(String, String)> {
    let t = line.trim();
    let indent = &line[..line.len() - line.trim_start().len()];
    let (mnemonic, rest) = t.split_once(' ').unwrap_or((t, ""));
    let ops = operands(rest);
    let with = |m: &str, ops: &[String]| format!("{indent}{m} {};", ops.join(", "));
    let mut out = vec![];
    let parts: Vec<&str> = mnemonic.split('.').collect();
    match parts.as_slice() {
        ["setp", cmp, ty] => {
            let other = match *cmp {
                "ge" => "gt",
                "gt" => "ge",
                "lt" => "le",
                "le" => "lt",
                "eq" => "ne",
                "ne" => "eq",
                _ => return out,
            };
            out.push((format!("{t}: {cmp} as {other}"), with(&format!("setp.{other}.{ty}"), &ops)));
        }
        ["fma", "rn", "f32"] => {
            out.push((format!("{t}: addend with first factor"), with(mnemonic, &[ops[0].clone(), ops[3].clone(), ops[2].clone(), ops[1].clone()])));
            out.push((format!("{t}: dropped"), format!("{indent}mov.f32 {}, {};", ops[0], ops[3])));
            out.push((format!("{t}: split"), format!("{indent}mul.rn.f32 {d}, {a}, {b};\n{indent}add.rn.f32 {d}, {d}, {c};", d = ops[0], a = ops[1], b = ops[2], c = ops[3])));
        }
        ["mov", ty] if ops.len() == 2 && SPECIALS.contains(&ops[1].as_str()) => {
            for other in SPECIALS.iter().filter(|o| **o != ops[1]) {
                out.push((format!("{t}: {} as {other}", ops[1]), format!("{indent}mov.{ty} {}, {other};", ops[0])));
            }
        }
        ["mov", ty] if ops.len() == 2 && !ops[1].starts_with('%') => {
            let v = &ops[1];
            let new = if let Some(hex) = v.strip_prefix("0f") {
                let bits = u32::from_str_radix(hex, 16).expect("f32 literal");
                format!("0f{:08X}", if bits == 0 { 0x3F80_0000 } else { bits ^ 1 })
            } else if let Ok(n) = v.parse::<i64>() {
                (n + 1).to_string()
            } else {
                return out;
            };
            out.push((format!("{t}: {v} as {new}"), format!("{indent}mov.{ty} {}, {new};", ops[0])));
        }
        [name @ ("add" | "sub" | "mul"), rest @ ..] if ops.len() == 3 => {
            let ty = rest.last().copied().unwrap_or("");
            if let Ok(n) = ops[2].parse::<i64>() {
                let new = if n == 4 { 8 } else { n + 1 };
                out.push((format!("{t}: {n} as {new}"), with(mnemonic, &[ops[0].clone(), ops[1].clone(), new.to_string()])));
            } else if !literal_regs.contains_key(&ops[2]) {
                let mov_ty = match ty {
                    "f32" => "f32",
                    "u32" => "u32",
                    _ => "u64",
                };
                out.push((format!("{t}: {name} dropped"), format!("{indent}mov.{mov_ty} {}, {};", ops[0], ops[1])));
            }
        }
        _ => {}
    }
    out
}

/// The sweep over every line of the KIR module. An add of a register just
/// loaded with a literal (a loop step) is nudged through its literal
/// rather than dropped: a dropped step never ends.
#[test]
fn every_mutant_is_caught() {
    let p = kir_ptx();
    let lines: Vec<&str> = p.lines().collect();
    let mut literal_regs: HashMap<String, String> = HashMap::new();
    let mut missed = vec![];
    let mut tried = 0;
    for (n, line) in lines.iter().enumerate() {
        let t = line.trim();
        let muts = line_mutants(line, &literal_regs);
        if let Some(rest) = t.strip_prefix("mov.u64 ").or_else(|| t.strip_prefix("mov.u32 ")) {
            let ops = operands(rest);
            if ops[1].parse::<i64>().is_ok() {
                literal_regs.insert(ops[0].clone(), ops[1].clone());
            } else {
                literal_regs.remove(&ops[0]);
            }
        } else if let Some(d) = t.split_whitespace().nth(1) {
            literal_regs.remove(d.trim_end_matches(','));
        }
        for (what, replacement) in muts {
            tried += 1;
            let mutant = lines.iter().enumerate().map(|(k, l)| if k == n { replacement.clone() } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n";
            if !caught(&mutant) {
                missed.push(format!("line {n}: {what}"));
            }
        }
    }
    assert!(tried > 30, "the sweep tried only {tried} mutants");
    assert!(missed.is_empty(), "{missed:#?}");
}

#[test]
fn the_unmutated_kernel_is_not_caught() {
    assert!(!caught(&kir_ptx()));
}
