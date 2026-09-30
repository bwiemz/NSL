//! The differential equivalence gate for the GEMM-chunked fused linear-CE's
//! chunk kernels from `nsl_runtime::cuda::fused_kernels` (new-roadmap item
//! 5): `nsl_lce_chunk_stats_f32` and `nsl_lce_chunk_dlogits_f32`, now built
//! by `nsl_kir::kernels::lce_chunk`.
//!
//! This file runs the frozen hand modules (`tests/fixtures/lce_chunk_hand.rs`)
//! and the KIR ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`):
//!
//! - **stats**: one 256-thread block per row plus one past the last, with
//!   shared memory poisoned, under all four thread schedules. Chunks are 1
//!   to 700 columns wide at chunk offsets 0 and past it, with and without a
//!   bias (the bias pointer null then), from a fresh state (`m = -inf`, `s =
//!   0`) and from a running one. Targets sit before, at the start of, inside,
//!   at the end of and just past the chunk, are negative (the ignore index)
//!   and `i64::MIN`.
//! - **dlogits**: whole grids with `%nctaid.x` set, the runtime's `ceil(rows
//!   · cols / 256)` blocks, one block and three (strides that do not divide
//!   the work), under both block orders and all four thread schedules, with
//!   and without a bias, over the same kinds of targets.
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory.
//! 2. **Correctness**: the stats are the hand kernel's passes restated (the
//!    per-thread folds in column order, thread 0's in thread order, the
//!    rescale as one `fma`, the target logit when the target is in the
//!    chunk); the gradient is `t < 0 ? 0 : (2^((val - lse) · log2 e) - [t ==
//!    chunk_start + j]) · scale`, and `dbias` receives each `dl` in the
//!    schedule's order. Both match bit for bit; nothing past the rows or the
//!    bias is written, and the inputs are untouched (the stats kernel's
//!    logits, both kernels' bias, targets and `lse`).
//! 3. **The gate bites**: a sweep over the KIR modules mutates every
//!    comparison, every integer and float operation, every constant, every
//!    element size, every barrier, both `selp`s' order, the `fma`'s operands,
//!    `ex2`, the bias's atomic add, and requires each mutant to be told apart
//!    (every mutant parses). The named equivalent mutants are listed where
//!    the sweep skips them.

use std::collections::HashMap;

use nsl_kir::kernels::lce_chunk::{ptx, LceChunkOp, LCE_CHUNK_BLOCK, LOG2_E_BITS};

#[allow(dead_code)]
#[path = "fixtures/lce_chunk_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 16;
const POISON: u32 = 0x7FA5_A5A5;
const LOGITS: u64 = 0x1000_0000;
const BIAS: u64 = 0x2000_0000;
const TARGETS: u64 = 0x3000_0000;
const S1: u64 = 0x4000_0000;
const S2: u64 = 0x5000_0000;
const S3: u64 = 0x6000_0000;
const SHARED: usize = LCE_CHUNK_BLOCK as usize * 4;
const ORDERS: [Order; 4] = [Order::Ascending, Order::Descending, Order::WarpsAscending, Order::WarpsDescending];

fn kir_ptx(op: LceChunkOp) -> String {
    String::from_utf8(ptx(op)).expect("ASCII").trim_end_matches('\0').to_string()
}

fn hand_ptx(op: LceChunkOp) -> String {
    match op {
        LceChunkOp::Stats => hand::LCE_CHUNK_STATS_F32_PTX,
        LceChunkOp::Dlogits => hand::LCE_CHUNK_DLOGITS_F32_PTX,
    }
    .trim_end_matches('\0')
    .to_string()
}

fn le32(v: &[u32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn le64(v: &[i64]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn words(b: &[u8]) -> Vec<u32> {
    b.chunks(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn lcg(s: &mut u64) -> u64 {
    *s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    *s >> 33
}

/// Full-mantissa values in `[-lo, hi)`.
fn data(n: usize, seed: u64, lo: f32, hi: f32) -> Vec<u32> {
    let mut s = seed;
    (0..n).map(|_| ((lcg(&mut s) as f32 / (1u64 << 31) as f32) * (hi + lo) - lo).to_bits()).collect()
}

/// One chunk: `rows × cols` logits at `chunk_start` of a longer vocabulary.
struct Case {
    rows: usize,
    cols: usize,
    chunk_start: usize,
    has_bias: bool,
    logits: Vec<u32>,
    bias: Vec<u32>,
    targets: Vec<i64>,
    /// stats: `m`, `s`, `tl` per row; dlogits: `lse` per row and the
    /// starting `dbias`.
    a: Vec<u32>,
    b: Vec<u32>,
    c: Vec<u32>,
    scale: f32,
}

/// Targets around the chunk: before, at its start, inside, at its end, just
/// past, far past, the ignore index and `i64::MIN`, cycled over the rows.
fn targets(rows: usize, cs: usize, cols: usize, seed: u64) -> Vec<i64> {
    let (cs, cols) = (cs as i64, cols as i64);
    let mut s = seed;
    let kinds = [cs - 1, cs, cs + cols / 2, cs + cols - 1, cs + cols, cs + cols + 40, -100, i64::MIN];
    (0..rows)
        .map(|r| {
            let t = kinds[(r + (lcg(&mut s) % 3) as usize) % kinds.len()];
            if t < 0 && t != -100 && t != i64::MIN { cs + (lcg(&mut s) as i64 % cols.max(1)) } else { t }
        })
        .collect()
}

/// `(rows, cols, chunk_start, has_bias, running state)`.
const SHAPES: [(usize, usize, usize, bool, bool); 9] = [
    (1, 1, 0, false, false),
    (3, 5, 0, true, false),
    (4, 256, 128, false, true),
    (3, 257, 0, true, true),
    (9, 300, 512, true, false),
    (2, 700, 64, false, true),
    (8, 40, 1000, true, true),
    (5, 3, 7, false, false),
    (32, 6, 2, true, true),
];

fn cases(op: LceChunkOp) -> Vec<Case> {
    SHAPES
        .iter()
        .enumerate()
        .map(|(i, &(rows, cols, cs, has_bias, running))| {
            let seed = 17 + 11 * i as u64;
            let v = cs + cols + 9;
            let mut logits = data(rows * cols, seed, 6.0, 6.0);
            if cols > LCE_CHUNK_BLOCK as usize {
                // A row whose max is the first column past one stride.
                logits[LCE_CHUNK_BLOCK as usize] = 9.5f32.to_bits();
            }
            let bias = if has_bias { data(v, seed + 1, 1.0, 1.0) } else { vec![] };
            let targets = targets(rows, cs, cols, seed + 2);
            let (a, b, c) = match op {
                LceChunkOp::Stats if running => (data(rows, seed + 3, 2.0, 5.0), data(rows, seed + 4, -0.5, 30.0), data(rows, seed + 5, 3.0, 3.0)),
                LceChunkOp::Stats => (vec![f32::NEG_INFINITY.to_bits(); rows], vec![0; rows], vec![0; rows]),
                LceChunkOp::Dlogits => (data(rows, seed + 3, -4.0, 9.0), data(v, seed + 4, 1.0, 1.0), vec![]),
            };
            Case { rows, cols, chunk_start: cs, has_bias, logits, bias, targets, a, b, c, scale: 0.37 }
        })
        .collect()
}

fn args(op: LceChunkOp, c: &Case) -> HashMap<String, u64> {
    let bias = if c.has_bias { BIAS } else { 0 };
    let values: Vec<u64> = match op {
        LceChunkOp::Stats => vec![LOGITS, bias, TARGETS, S1, S2, S3, c.rows as u64, c.cols as u64, c.chunk_start as u64, u64::from(c.has_bias)],
        LceChunkOp::Dlogits => vec![
            LOGITS,
            bias,
            TARGETS,
            S1,
            S2,
            c.rows as u64,
            c.cols as u64,
            c.chunk_start as u64,
            u64::from(c.scale.to_bits()),
            u64::from(c.has_bias),
        ],
    };
    op.param_names().iter().zip(values).map(|(p, v)| (p.to_string(), v)).collect()
}

fn with_tail(v: &[u32]) -> Vec<u32> {
    let mut v = v.to_vec();
    v.extend([POISON; TAIL]);
    v
}

/// All of global memory after the launch, as words. `grid` is the dlogits
/// grid (the stats kernel runs a block per row plus one).
fn run(op: LceChunkOp, ptx: &str, c: &Case, grid: u32, order: Order, reverse_ctas: bool) -> Vec<u32> {
    let prog = parse(ptx);
    let mut global = vec![
        Segment { base: LOGITS, bytes: le32(&with_tail(&c.logits)) },
        Segment { base: BIAS, bytes: le32(&with_tail(&c.bias)) },
        Segment { base: TARGETS, bytes: le64(&c.targets) },
        Segment { base: S1, bytes: le32(&with_tail(&c.a)) },
        Segment { base: S2, bytes: le32(&with_tail(&c.b)) },
        Segment { base: S3, bytes: le32(&with_tail(&c.c)) },
    ];
    let args = args(op, c);
    let (n, nctaid) = match op {
        LceChunkOp::Stats => (c.rows as u32 + 1, 0),
        LceChunkOp::Dlogits => (grid, grid),
    };
    let mut ctas: Vec<u32> = (0..n).collect();
    if reverse_ctas {
        ctas.reverse();
    }
    for ctaid in ctas {
        let mut l = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: le32(&vec![POISON; SHARED / 4]),
            ctaid,
            ctaid_y: 0,
            nctaid_x: nctaid,
            nctaid_y: 1,
            ntid: LCE_CHUNK_BLOCK,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    global.iter().flat_map(|s| words(&s.bytes)).collect()
}

/// The grids each case launches: the stats kernel's one; the dlogits
/// kernel's runtime grid, one block and three.
fn grids(op: LceChunkOp, c: &Case) -> Vec<u32> {
    match op {
        LceChunkOp::Stats => vec![0],
        LceChunkOp::Dlogits => {
            let runtime = (c.rows * c.cols).div_ceil(LCE_CHUNK_BLOCK as usize).max(1) as u32;
            let mut g = vec![runtime, 1, 3];
            g.dedup();
            g
        }
    }
}

#[test]
fn the_kernels_agree() {
    for op in LceChunkOp::ALL {
        let (hand, kir) = (hand_ptx(op), kir_ptx(op));
        for c in cases(op) {
            for grid in grids(op, &c) {
                for order in ORDERS {
                    for rev in [false, true] {
                        assert!(
                            run(op, &hand, &c, grid, order, rev) == run(op, &kir, &c, grid, order, rev),
                            "{op:?} rows {} cols {} cs {} bias {} grid {grid} {order:?} rev {rev}",
                            c.rows,
                            c.cols,
                            c.chunk_start,
                            c.has_bias
                        );
                    }
                }
            }
        }
    }
}

fn f(w: u32) -> f32 {
    f32::from_bits(w)
}

fn val(c: &Case, r: usize, j: usize) -> f32 {
    let x = f(c.logits[r * c.cols + j]);
    if c.has_bias { x + f(c.bias[c.chunk_start + j]) } else { x }
}

fn exp_shifted(x: f32, m: f32) -> f32 {
    ((x - m) * f(LOG2_E_BITS)).exp2()
}

/// The stats kernel's passes restated: `(m, s, tl)` after the launch. The
/// rescale is one `fma`, or, with `split`, a separately rounded multiply
/// and add.
fn stats_reference(c: &Case, split: bool) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let (mut m, mut s, mut tl) = (c.a.clone(), c.b.clone(), c.c.clone());
    let n = LCE_CHUNK_BLOCK as usize;
    for r in 0..c.rows {
        let partial = |t: usize, fold: &dyn Fn(f32, f32) -> f32, init: f32| (t..c.cols).step_by(n).fold(init, |acc, j| fold(acc, val(c, r, j)));
        let block_max = (1..n).fold(partial(0, &|a, v| a.max(v), f32::NEG_INFINITY), |acc, t| acc.max(partial(t, &|a, v| a.max(v), f32::NEG_INFINITY)));
        let m_old = f(m[r]);
        let mn = m_old.max(block_max);
        let sum_part = |t: usize| (t..c.cols).step_by(n).fold(0.0f32, |acc, j| acc + exp_shifted(val(c, r, j), mn));
        let total = (1..n).fold(sum_part(0), |acc, t| acc + sum_part(t));
        let (s_old, rescale) = (f(s[r]), exp_shifted(m_old, mn));
        s[r] = if split { s_old * rescale + total } else { s_old.mul_add(rescale, total) }.to_bits();
        m[r] = mn.to_bits();
        let t = c.targets[r];
        let cs = c.chunk_start as i64;
        if t >= cs && t - cs < c.cols as i64 {
            tl[r] = val(c, r, (t - cs) as usize).to_bits();
        }
    }
    (m, s, tl)
}

/// The dlogits kernel restated, for an ascending schedule over ascending
/// blocks of `grid`: `(dl, dbias)`.
fn dlogits_reference(c: &Case, grid: u32) -> (Vec<u32>, Vec<u32>) {
    let mut dl = c.logits.clone();
    let mut dbias = c.b.clone();
    let n = LCE_CHUNK_BLOCK as usize;
    let (total, stride) = (c.rows * c.cols, n * grid as usize);
    for cta in 0..grid as usize {
        for tid in 0..n {
            let mut idx = cta * n + tid;
            while idx < total {
                let (r, j) = (idx / c.cols, idx % c.cols);
                let t = c.targets[r];
                let g = if t < 0 {
                    0.0
                } else {
                    let p = exp_shifted(val(c, r, j), f(c.a[r]));
                    (if t == (c.chunk_start + j) as i64 { p - 1.0 } else { p }) * c.scale
                };
                dl[idx] = g.to_bits();
                if c.has_bias {
                    let k = c.chunk_start + j;
                    dbias[k] = (f(dbias[k]) + g).to_bits();
                }
                idx += stride;
            }
        }
    }
    (dl, dbias)
}

#[test]
fn the_kernels_match_the_restated_passes() {
    for op in LceChunkOp::ALL {
        for which in [hand_ptx(op), kir_ptx(op)] {
            for c in cases(op) {
                for grid in grids(op, &c) {
                    let g = run(op, &which, &c, grid, Order::Ascending, false);
                    let mut at = 0;
                    let mut seg = |len: usize| {
                        let s = g[at..at + len].to_vec();
                        at += len;
                        s
                    };
                    let logits = seg(c.logits.len() + TAIL);
                    let bias = seg(c.bias.len() + TAIL);
                    let tgt = seg(2 * c.rows);
                    let (s1, s2, s3) = (seg(c.a.len() + TAIL), seg(c.b.len() + TAIL), seg(c.c.len() + TAIL));
                    let ctx = format!("{op:?} rows {} cols {} bias {} grid {grid}", c.rows, c.cols, c.has_bias);
                    assert_eq!(bias, with_tail(&c.bias), "{ctx}: bias untouched");
                    assert_eq!(tgt, words(&le64(&c.targets)), "{ctx}: targets untouched");
                    match op {
                        LceChunkOp::Stats => {
                            let (m, s, tl) = stats_reference(&c, false);
                            assert_eq!(logits, with_tail(&c.logits), "{ctx}: logits untouched");
                            assert_eq!(s1, with_tail(&m), "{ctx}: m");
                            assert_eq!(s2, with_tail(&s), "{ctx}: s");
                            assert_eq!(s3, with_tail(&tl), "{ctx}: tl");
                        }
                        LceChunkOp::Dlogits => {
                            let (dl, dbias) = dlogits_reference(&c, grid);
                            assert_eq!(logits, with_tail(&dl), "{ctx}: dl");
                            assert_eq!(s1, with_tail(&c.a), "{ctx}: lse untouched");
                            assert_eq!(s2, with_tail(&dbias), "{ctx}: dbias");
                            assert_eq!(s3, with_tail(&c.c), "{ctx}");
                        }
                    }
                }
            }
        }
    }
}

/// The data reaches what the kernels distinguish: every kind of target,
/// columns past one block's stride (one of them a row's max), a running
/// state the rescale changes, and a rescale whose `fma` rounds once where a
/// multiply and an add would round twice.
#[test]
fn the_data_covers_every_case() {
    let cs = cases(LceChunkOp::Stats);
    assert!(cs.iter().any(|c| c.cols > 2 * LCE_CHUNK_BLOCK as usize));
    let kinds = |c: &Case| {
        let (lo, hi) = (c.chunk_start as i64, (c.chunk_start + c.cols) as i64);
        c.targets.iter().map(|&t| (t < 0, t >= 0 && t < lo, t == lo, t > lo && t < hi - 1, t == hi - 1, t >= hi)).collect::<Vec<_>>()
    };
    let all: Vec<_> = cs.iter().flat_map(kinds).collect();
    assert!(all.iter().any(|k| k.0) && all.iter().any(|k| k.1) && all.iter().any(|k| k.2), "{all:?}");
    assert!(all.iter().any(|k| k.3) && all.iter().any(|k| k.4) && all.iter().any(|k| k.5), "{all:?}");
    assert!(cs.iter().any(|c| c.targets.contains(&i64::MIN)));
    // A running state whose rescale is neither 0 nor 1.
    assert!(cs.iter().any(|c| {
        let (m, _, _) = stats_reference(c, false);
        c.a.iter().zip(&m).any(|(&old, &new)| f(old).is_finite() && old != new)
    }));
    assert!(cs.iter().any(|c| stats_reference(c, false).1 != stats_reference(c, true).1), "the fma's single rounding shows");
    assert!(cs.iter().any(|c| c.cols > LCE_CHUNK_BLOCK as usize && stats_reference(c, false).0[0] == 9.5f32.to_bits()), "a max past one stride");
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Whether `mutant` is told apart from the hand kernel on any case, grid,
/// schedule or block order (or faults).
fn caught(op: LceChunkOp, mutant: &str) -> bool {
    parse(mutant); // a mutant that does not parse would be a hollow kill
    let hand = hand_ptx(op);
    cases(op).iter().any(|c| {
        grids(op, c).into_iter().any(|grid| {
            ORDERS.into_iter().any(|order| {
                [false, true].into_iter().any(|rev| {
                    let expect = run(op, &hand, c, grid, order, rev);
                    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(op, mutant, c, grid, order, rev))) {
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

/// Every mutant of one line: `(description, replacement lines)`.
fn line_mutants(line: &str, last_const: &HashMap<String, String>) -> Vec<(String, String)> {
    let t = line.trim();
    let indent = &line[..line.len() - line.trim_start().len()];
    let (mnemonic, rest) = t.split_once(' ').unwrap_or((t, ""));
    let ops = operands(rest);
    let mut out = vec![];
    let with = |m: &str, ops: &[String]| format!("{indent}{m} {};", ops.join(", "));
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
        ["selp", _] => {
            let swapped = [ops[0].clone(), ops[2].clone(), ops[1].clone(), ops[3].clone()];
            out.push((format!("{t}: swapped"), with(mnemonic, &swapped)));
        }
        ["fma", "rn", "f32"] => {
            for (name, order) in [("addend with first factor", [0, 3, 2, 1]), ("dropped", [0, 3, 3, 3])] {
                let picked: Vec<String> = order.iter().map(|&k| ops[k].clone()).collect();
                if name == "dropped" {
                    out.push((format!("{t}: {name}"), format!("{indent}mov.f32 {}, {};", ops[0], ops[3])));
                } else {
                    out.push((format!("{t}: {name}"), with(mnemonic, &picked)));
                }
            }
            out.push((format!("{t}: split"), format!("{indent}mul.rn.f32 {d}, {a}, {b};\n{indent}add.rn.f32 {d}, {d}, {c};", d = ops[0], a = ops[1], b = ops[2], c = ops[3])));
        }
        ["ex2", "approx", "f32"] => out.push((format!("{t}: dropped"), format!("{indent}mov.f32 {}, {};", ops[0], ops[1]))),
        ["bar", "sync"] => out.push((format!("{t}: dropped"), String::new())),
        ["red", "global", "add", "f32"] => out.push((format!("{t}: as a store"), format!("{indent}st.global.f32 {}, {};", ops[0], ops[1]))),
        ["mov", ty] if ops.len() == 2 && !ops[1].starts_with('%') => {
            let v = &ops[1];
            let new = if let Some(hex) = v.strip_prefix("0f") {
                let bits = u32::from_str_radix(hex, 16).expect("f32 literal");
                // `0` becomes 1, `-inf` becomes 0, anything else its next bit
                // pattern (`-inf ^ 1` would be a NaN, which `max` ignores).
                format!("0f{:08X}", match bits {
                    0 => 0x3F80_0000,
                    0xFF80_0000 => 0,
                    _ => bits ^ 1,
                })
            } else if let Ok(n) = v.parse::<i64>() {
                (n + 1).to_string()
            } else {
                return out;
            };
            out.push((format!("{t}: {v} as {new}"), format!("{indent}mov.{ty} {}, {new};", ops[0])));
        }
        [name @ ("add" | "sub" | "mul" | "max" | "div"), rest @ ..] if ops.len() == 3 => {
            let ty = rest.last().copied().unwrap_or("");
            if ops[2].parse::<i64>().is_ok() {
                // An element size or a literal step: its own mutant.
                let n: i64 = ops[2].parse().expect("literal");
                let new = if n == 4 { 8 } else { n + 1 };
                out.push((format!("{t}: {n} as {new}"), with(mnemonic, &[ops[0].clone(), ops[1].clone(), new.to_string()])));
            } else if !last_const.contains_key(&ops[2]) {
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

/// The sweep: every line's mutants, with the registers whose latest
/// definition is an integer literal tracked so that an add of a literal
/// step is nudged through its literal (a dropped loop step never ends).
fn sweep(op: LceChunkOp, skip: &[&str]) -> Vec<String> {
    let p = kir_ptx(op);
    let lines: Vec<&str> = p.lines().collect();
    let mut last_const: HashMap<String, String> = HashMap::new();
    let mut missed = vec![];
    let mut tried = 0;
    for (n, line) in lines.iter().enumerate() {
        let t = line.trim();
        let muts = line_mutants(line, &last_const);
        if let Some(rest) = t.strip_prefix("mov.u64 ").or_else(|| t.strip_prefix("mov.u32 ")) {
            let ops = operands(rest);
            if ops[1].parse::<i64>().is_ok() {
                last_const.insert(ops[0].clone(), ops[1].clone());
            } else {
                last_const.remove(&ops[0]);
            }
        } else if let Some(d) = t.split_whitespace().nth(1) {
            last_const.remove(d.trim_end_matches(','));
        }
        for (what, replacement) in muts {
            if skip.iter().any(|s| what.contains(s)) {
                continue;
            }
            tried += 1;
            let mutant = lines.iter().enumerate().map(|(k, l)| if k == n { replacement.clone() } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n";
            if !caught(op, &mutant) {
                missed.push(format!("line {n}: {what}"));
            }
        }
    }
    assert!(tried > 30, "{op:?}: the sweep tried only {tried} mutants");
    missed
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for op in LceChunkOp::ALL {
        assert!(!caught(op, &kir_ptx(op)), "{op:?}");
    }
}

#[test]
fn every_stats_mutant_is_caught() {
    let missed = sweep(LceChunkOp::Stats, &[]);
    assert!(missed.is_empty(), "{missed:#?}");
}

#[test]
fn every_dlogits_mutant_is_caught() {
    let missed = sweep(LceChunkOp::Dlogits, &[]);
    assert!(missed.is_empty(), "{missed:#?}");
}
