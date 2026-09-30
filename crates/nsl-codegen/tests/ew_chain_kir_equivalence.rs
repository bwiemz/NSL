//! The differential equivalence gate for the fused adjoint elementwise-chain
//! kernel (MFU campaign C3), now built by `nsl_codegen::ew_chain_ptx`
//! (new-roadmap item 5).
//!
//! This file runs the frozen hand emitter (`tests/fixtures/ew_chain_hand.rs`)
//! and the KIR one side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), for a spread of chain signatures:
//! every opcode with each operand kind (input slot, earlier step,
//! immediate) on each side it may take, `RtsCheck` steps in the middle and
//! at the end of a chain, an input slot no step reads, the fuser's widest
//! chain (six inputs, six arithmetic steps), and seeded random chains. Each
//! runs over whole grids with a spare block, on the runtime's 256-thread
//! block and a 32-thread one, under all four thread schedules:
//!
//! 1. **Agreement**: the two kernels leave *the same bytes* in all of global
//!    memory.
//! 2. **Correctness**: every `out[i]` is the chain evaluated in tape order
//!    with f32 `+`, `-`, `*`, the interpreter's `div.approx` (`a / b`, or `a
//!    · ±0` for a divisor of magnitude in `(2^126, 2^128)`), and negation,
//!    an `RtsCheck` passing its left operand through; nothing past `out` is
//!    written and no input is touched.
//! 3. **The gate bites**: a sweep over the KIR modules of several chains
//!    mutates every comparison, every integer and float operation (dropped,
//!    swapped for its sibling, operands exchanged), every constant, every
//!    special register, every parameter read (each input read as another)
//!    and the approximate divide's rounding, and requires each mutant to be
//!    told apart (every mutant parses).
//!
//! The data has full mantissas and signed zeros, infinities, a NaN and a
//! divisor of 2^127.

use std::collections::HashMap;

use nsl_codegen::ew_chain_fusion::{ChainSig, ChainStep, EwOpcode, Operand};
use nsl_codegen::ew_chain_ptx::{emit, EW_CHAIN_BLOCK};

/// The `crate::` path the frozen emitter names, supplied from the library
/// under test (the fixture is included into this test crate).
mod ew_chain_fusion {
    pub use nsl_codegen::ew_chain_fusion::*;
}

#[allow(dead_code)]
#[path = "fixtures/ew_chain_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TAIL: usize = 8;
const POISON: u32 = 0x7FA5_A5A5;
const OUT: u64 = 0x1000_0000;
const IN0: u64 = 0x2000_0000;
const IN_STRIDE: u64 = 0x0100_0000;
const ORDERS: [Order; 4] = [Order::Ascending, Order::Descending, Order::WarpsAscending, Order::WarpsDescending];
const BLOCKS: [u32; 2] = [EW_CHAIN_BLOCK, 32];
const LENS: [usize; 3] = [1, 70, 300];
const KNAME: &str = "nsl_fused_ew_test";

fn text(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes).expect("ASCII").trim_end_matches('\0').to_string()
}

fn kir_ptx(sig: &ChainSig) -> String {
    text(emit(sig, KNAME))
}

fn hand_ptx(sig: &ChainSig) -> String {
    text(hand::synthesize_fused_chain_ptx(sig, KNAME, 80))
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

/// Values the chain distinguishes: full mantissas in `[-2, 2)`, with a
/// signed zero, an infinity, a NaN and 2^127 (a divisor whose reciprocal
/// the approximate divide flushes) at fixed places.
fn data(n: usize, seed: u64) -> Vec<u32> {
    let mut s = seed;
    let specials = [0x8000_0000, 0x0000_0000, 0x7F80_0000, 0xFF80_0000, 0x7FC0_0001, 0x7F00_0000];
    (0..n)
        .map(|i| {
            let r = lcg(&mut s);
            // Each input's specials sit at its own places, so a special
            // divisor meets a finite dividend.
            if (i + seed as usize) % 23 == 5 {
                specials[(r as usize) % specials.len()]
            } else {
                ((r as f32 / (1u64 << 31) as f32) * 4.0 - 2.0).to_bits()
            }
        })
        .collect()
}

fn step(op: EwOpcode, lhs: Operand, rhs: Option<Operand>) -> ChainStep {
    ChainStep { op, lhs, rhs }
}

use EwOpcode::{Add, Div, Mul, Neg, RtsCheck, Sub};
use Operand::{Imm, Input as I, Prev as P};

/// The fuser's canonical chain (as `fused_ew_ptx_pin.rs` pins it):
/// `mul(i0,i1); rts(p0,i2); add(p1,0.5); neg(p2); div(p3,i3)`.
fn canonical() -> ChainSig {
    ChainSig {
        n_inputs: 4,
        steps: vec![
            step(Mul, I(0), Some(I(1))),
            step(RtsCheck, P(0), Some(I(2))),
            step(Add, P(1), Some(Imm(0x3F00_0000))),
            step(Neg, P(2), None),
            step(Div, P(3), Some(I(3))),
        ],
    }
}

/// A random well-formed chain: `n_inputs` slots, `steps` steps, each
/// operand an input, an earlier step or (on the right) an immediate.
fn random_sig(seed: u64) -> ChainSig {
    let mut s = seed;
    let n_inputs = 1 + (lcg(&mut s) % 6) as u8;
    let n_steps = 2 + (lcg(&mut s) % 5) as usize;
    let ops = [Add, Sub, Mul, Div, Neg, RtsCheck];
    let mut steps = vec![];
    for j in 0..n_steps {
        let pick = |s: &mut u64, imm_ok: bool| -> Operand {
            match lcg(s) % if imm_ok { 3 } else { 2 } {
                0 if j > 0 => P((lcg(s) % j as u64) as u8),
                2 => Imm(((lcg(s) as f32 / (1u64 << 31) as f32) * 6.0 - 3.0).to_bits()),
                _ => I((lcg(s) % u64::from(n_inputs)) as u8),
            }
        };
        let op = ops[(lcg(&mut s) % ops.len() as u64) as usize];
        // A chain reads the previous step first, as the fuser's runs do.
        let lhs = if j > 0 && !lcg(&mut s).is_multiple_of(3) { P(j as u8 - 1) } else { pick(&mut s, false) };
        let rhs = match op {
            Neg => None,
            RtsCheck => Some(I((lcg(&mut s) % u64::from(n_inputs)) as u8)),
            _ => Some(pick(&mut s, true)),
        };
        steps.push(step(op, lhs, rhs));
    }
    ChainSig { n_inputs, steps }
}

fn sigs() -> Vec<ChainSig> {
    let mut v = vec![canonical()];
    for op in [Add, Sub, Mul, Div] {
        // input ∘ input, step ∘ immediate, input ∘ step (the step on the right).
        v.push(ChainSig { n_inputs: 2, steps: vec![step(op, I(1), Some(I(0))), step(op, P(0), Some(Imm(0xBFC0_0000)))] });
        v.push(ChainSig { n_inputs: 2, steps: vec![step(Neg, I(0), None), step(op, I(1), Some(P(0)))] });
    }
    // An RtsCheck last (the result is its left operand) and a slot no step reads.
    v.push(ChainSig { n_inputs: 3, steps: vec![step(Sub, I(2), Some(I(0))), step(RtsCheck, P(0), Some(I(1)))] });
    v.push(ChainSig { n_inputs: 2, steps: vec![step(RtsCheck, I(1), Some(I(0))), step(Neg, P(0), None)] });
    // The widest chain the fuser builds.
    v.push(ChainSig {
        n_inputs: 6,
        steps: vec![
            step(Mul, I(0), Some(I(1))),
            step(Add, P(0), Some(I(2))),
            step(Sub, P(1), Some(I(3))),
            step(Div, P(2), Some(I(4))),
            step(Mul, P(3), Some(I(5))),
            step(Add, P(4), Some(P(0))),
        ],
    });
    v.extend((0..24).map(|k| random_sig(0xC3 + 17 * k)));
    v
}

fn input_base(k: usize) -> u64 {
    IN0 + IN_STRIDE * k as u64
}

/// The inputs of one launch: `n_inputs` columns of `len` values.
fn inputs(sig: &ChainSig, len: usize, seed: u64) -> Vec<Vec<u32>> {
    (0..sig.n_inputs as usize).map(|k| data(len, seed * 31 + k as u64)).collect()
}

fn with_tail(v: &[u32]) -> Vec<u32> {
    let mut v = v.to_vec();
    v.extend([POISON; TAIL]);
    v
}

/// `[out + tail, in0 + tail, …]` as words, after the launch.
fn run(ptx: &str, sig: &ChainSig, xs: &[Vec<u32>], block: u32, order: Order) -> Vec<Vec<u32>> {
    let prog = parse(ptx);
    let len = xs[0].len();
    let mut global = vec![Segment { base: OUT, bytes: le32(&vec![POISON; len + TAIL]) }];
    let mut args: HashMap<String, u64> = HashMap::new();
    args.insert("param_out".into(), OUT);
    args.insert("out".into(), OUT);
    for (k, x) in xs.iter().enumerate() {
        global.push(Segment { base: input_base(k), bytes: le32(&with_tail(x)) });
        args.insert(format!("param_in{k}"), input_base(k));
        args.insert(format!("in{k}"), input_base(k));
    }
    args.insert("param_n".into(), len as u64);
    args.insert("n".into(), len as u64);
    let _ = sig;
    for ctaid in 0..len.div_ceil(block as usize) as u32 + 1 {
        let mut l = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![],
            ctaid,
            ctaid_y: 0,
            nctaid_x: 0,
            nctaid_y: 1,
            ntid: block,
            steps: 0,
        };
        run_cta(&mut l, order);
    }
    global.iter().map(|g| words(&g.bytes)).collect()
}

#[test]
fn the_kernels_agree() {
    for (s, sig) in sigs().iter().enumerate() {
        let (hand, kir) = (hand_ptx(sig), kir_ptx(sig));
        for len in LENS {
            let xs = inputs(sig, len, s as u64);
            for block in BLOCKS {
                for order in ORDERS {
                    assert!(run(&hand, sig, &xs, block, order) == run(&kir, sig, &xs, block, order), "{sig:?} len {len} block {block} {order:?}");
                }
            }
        }
    }
}

/// `div.approx.f32` as the interpreter models it.
fn div_approx(a: f32, b: f32) -> f32 {
    let m = b.abs();
    if m > 2f32.powi(126) && m.is_finite() {
        a * 0.0f32.copysign(b)
    } else {
        a / b
    }
}

/// The chain restated, element by element.
fn reference(sig: &ChainSig, xs: &[Vec<u32>]) -> Vec<u32> {
    (0..xs[0].len())
        .map(|e| {
            let mut r: Vec<f32> = vec![];
            let val = |o: Operand, r: &[f32]| match o {
                Operand::Input(k) => f32::from_bits(xs[k as usize][e]),
                Operand::Prev(p) => r[p as usize],
                Operand::Imm(bits) => f32::from_bits(bits),
            };
            for st in &sig.steps {
                let a = val(st.lhs, &r);
                let v = match st.op {
                    RtsCheck => a,
                    Neg => -a,
                    op => {
                        let b = val(st.rhs.unwrap(), &r);
                        match op {
                            Add => a + b,
                            Sub => a - b,
                            Mul => a * b,
                            _ => div_approx(a, b),
                        }
                    }
                };
                r.push(v);
            }
            r.last().unwrap().to_bits()
        })
        .collect()
}

/// Bitwise, except that any NaN matches any NaN: the interpreter's
/// arithmetic on a NaN operand is Rust's, whose payload is not the
/// hardware's canonical NaN, and both kernels agree bit for bit anyway.
fn same(a: &[u32], b: &[u32]) -> bool {
    a.len() == b.len() && a.iter().zip(b).all(|(&x, &y)| x == y || (f32::from_bits(x).is_nan() && f32::from_bits(y).is_nan()))
}

#[test]
fn the_kernels_are_the_chain() {
    for (s, sig) in sigs().iter().enumerate() {
        for which in [hand_ptx(sig), kir_ptx(sig)] {
            for len in LENS {
                let xs = inputs(sig, len, s as u64);
                let g = run(&which, sig, &xs, EW_CHAIN_BLOCK, Order::Ascending);
                assert!(same(&g[0], &with_tail(&reference(sig, &xs))), "{sig:?} len {len}");
                for (k, x) in xs.iter().enumerate() {
                    assert_eq!(g[1 + k], with_tail(x), "{sig:?}: in{k} untouched");
                }
            }
        }
    }
}

/// The signatures reach every opcode with every operand kind, and the data
/// reaches the divide's flush and non-finite values.
#[test]
fn the_cases_cover_every_form() {
    let all = sigs();
    let steps: Vec<&ChainStep> = all.iter().flat_map(|s| &s.steps).collect();
    for op in [Add, Sub, Mul, Div] {
        for kind in 0..3 {
            let is = |o: Operand| matches!((o, kind), (Operand::Input(_), 0) | (Operand::Prev(_), 1) | (Operand::Imm(_), 2));
            assert!(steps.iter().any(|st| st.op == op && st.rhs.is_some_and(is)), "{op:?} rhs kind {kind}");
            if kind < 2 {
                assert!(steps.iter().any(|st| st.op == op && is(st.lhs)), "{op:?} lhs kind {kind}");
            }
        }
    }
    assert!(all.iter().any(|s| s.steps.last().unwrap().op == RtsCheck));
    assert!(all.iter().any(|s| s.steps.iter().any(|st| st.op == RtsCheck && st.lhs == I(1))));
    assert!(all.iter().any(|s| s.n_inputs == 6 && s.steps.len() == 6));
    let d = data(300, 1);
    assert!(d.contains(&0x7F00_0000) && d.iter().any(|&x| f32::from_bits(x).is_nan()) && d.contains(&0x8000_0000));
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// The chains the sweep mutates: the canonical one and chains putting each
/// arithmetic opcode's operands through every slot.
fn sweep_sigs() -> Vec<ChainSig> {
    // The canonical chain with its like-reference slot also read, before
    // the divide: a slot no step reads is loaded for nothing, so no mutant
    // of that load could show, and a divide followed by an add or subtract
    // of O(1) would absorb the flushed quotient of a 2^127 divisor.
    let mut c = canonical();
    c.steps.insert(4, step(Sub, P(3), Some(I(2))));
    c.steps[5].lhs = P(4);
    let mut v = vec![c];
    v.push(ChainSig {
        n_inputs: 3,
        steps: vec![step(Sub, I(1), Some(I(0))), step(Mul, P(0), Some(I(2))), step(Div, I(0), Some(P(1))), step(Sub, P(2), Some(Imm(0x3FC0_0000)))],
    });
    v
}

/// Whether `mutant` is told apart from `sig`'s hand kernel on any length,
/// block or schedule (or faults).
fn caught(mutant: &str, sig: &ChainSig) -> bool {
    parse(mutant); // a mutant that does not parse would be a hollow kill
    let hand = hand_ptx(sig);
    LENS.iter().enumerate().any(|(s, &len)| {
        let xs = inputs(sig, len, 7 + s as u64);
        BLOCKS.into_iter().any(|block| {
            ORDERS.into_iter().any(|order| {
                let expect = run(&hand, sig, &xs, block, order);
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| run(mutant, sig, &xs, block, order))) {
                    Ok(r) => r != expect,
                    Err(_) => true,
                }
            })
        })
    })
}

fn operands(rest: &str) -> Vec<String> {
    rest.trim().trim_end_matches(';').split(',').map(|s| s.trim().to_string()).collect()
}

/// The special registers a mutant may read in place of another.
const SPECIALS: [&str; 4] = ["%ctaid.x", "%ntid.x", "%tid.x", "%ctaid.y"];

/// Every mutant of one line: `(description, replacement)`.
fn line_mutants(line: &str, params: &[String], literal_regs: &HashMap<String, String>) -> Vec<(String, String)> {
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
        ["ld", "param", ty] => {
            for p in params.iter().filter(|p| !ops[1].contains(p.as_str())) {
                out.push((format!("{t}: reads {p}"), format!("{indent}ld.param.{ty} {}, [{p}];", ops[0])));
            }
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
        ["neg", "f32"] => {
            out.push((format!("{t}: dropped"), format!("{indent}mov.f32 {}, {};", ops[0], ops[1])));
        }
        ["div", "approx", "f32"] => {
            out.push((format!("{t}: as div.rn"), with("div.rn.f32", &ops)));
            out.push((format!("{t}: operands exchanged"), with(mnemonic, &[ops[0].clone(), ops[2].clone(), ops[1].clone()])));
            out.push((format!("{t}: as mul"), with("mul.rn.f32", &ops)));
            out.push((format!("{t}: dropped"), format!("{indent}mov.f32 {}, {};", ops[0], ops[1])));
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
                if ty == "f32" {
                    let sibling = match *name {
                        "add" => "sub",
                        "sub" => "add",
                        _ => "add",
                    };
                    out.push((format!("{t}: as {sibling}"), with(&mnemonic.replacen(name, sibling, 1), &ops)));
                    if *name == "sub" {
                        out.push((format!("{t}: operands exchanged"), with(mnemonic, &[ops[0].clone(), ops[2].clone(), ops[1].clone()])));
                    }
                }
            }
        }
        [name @ ("shl" | "mul"), "wide", ..] | [name @ "shl", ..] if ops.len() == 3 => {
            if let Ok(n) = ops[2].parse::<i64>() {
                out.push((format!("{t}: {name} by {n} as {}", n + 1), with(mnemonic, &[ops[0].clone(), ops[1].clone(), (n + 1).to_string()])));
            }
        }
        _ => {}
    }
    out
}

/// The sweep over every line of the KIR modules.
#[test]
fn every_mutant_is_caught() {
    let mut missed = vec![];
    let mut tried = 0;
    for sig in sweep_sigs() {
        let p = kir_ptx(&sig);
        let lines: Vec<&str> = p.lines().collect();
        let params: Vec<String> = p
            .split(".param .u64 ")
            .skip(1)
            .map(|s| s.split([',', ')']).next().unwrap().trim().to_string())
            .collect();
        assert_eq!(params.len(), sig.n_inputs as usize + 2, "{p}");
        let mut literal_regs: HashMap<String, String> = HashMap::new();
        for (n, line) in lines.iter().enumerate() {
            let t = line.trim();
            let muts = line_mutants(line, &params, &literal_regs);
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
                if !caught(&mutant, &sig) {
                    missed.push(format!("{:?} line {n}: {what}", sig.steps.len()));
                }
            }
        }
    }
    assert!(tried > 100, "the sweep tried only {tried} mutants");
    assert!(missed.is_empty(), "{missed:#?}");
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for sig in sweep_sigs() {
        assert!(!caught(&kir_ptx(&sig), &sig));
    }
}
