//! The differential equivalence gate for the CSHA Tier B.1 pre-pass
//! kernels (`csha_tier_b1_prepass_x` / `_w`, roadmap A2 step 11).
//!
//! The runtime carried both as hand-written PTX
//! (`CSHA_TIER_B1_PREPASS_X_PTX`, `CSHA_TIER_B1_PREPASS_W_PTX`); they are
//! now built as KIR by `nsl_kir::kernels::tier_b1_prepass`. This file runs
//! the frozen hand modules (`tests/fixtures/tier_b1_prepass_hand.rs`) and
//! the KIR ones side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), over the grid the runtime's
//! launchers use:
//!
//! 1. **Agreement**: under two schedules (threads and CTAs in ascending or
//!    descending order), the hand and KIR kernels leave *the same bytes* in
//!    all of global memory.
//! 2. **Correctness**: the X output is the RMS-normalised, gamma-scaled
//!    row narrowed to f16 in the chunks-major layout (against an f64
//!    reference), and the W output is every weight narrowed to f16 at its
//!    col-major-within-chunk position (exactly); nothing else is written.
//! 3. **The gate bites**: deleting either barrier, relaxing any bound,
//!    nudging a baked constant, or reading the row from block 0 is caught.
//!
//! One instruction differs and the interpreter cannot tell: the hand X
//! kernel's `div.approx.f32` is KIR's `div.rn.f32`. The interpreter
//! models every approximate form by its exact counterpart, so it compares
//! them equal; on the machine the mean square is now correctly rounded
//! where it was within 2 ulp, ahead of an `rsqrt.approx.f32`.

use std::collections::HashMap;

use half::f16;
use nsl_kir::kernels::tier_b1_prepass::{ptx as kir_module, Prepass, PREPASS_BLOCK};

#[allow(dead_code)]
#[path = "fixtures/tier_b1_prepass_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const IN: u64 = 0x1000_0000;
const GAMMA: u64 = 0x2000_0000;
const OUT: u64 = 0x3000_0000;
/// What the output holds before a pre-pass: a pattern no f16 it writes has.
const POISON: u8 = 0xA5;
const EPS: f32 = 1e-5;

/// The X pre-pass over `[seq, d_model]` in chunks of `chunk`, launched on
/// `seq + extra_rows` CTAs (the extra ones must write nothing).
#[derive(Debug, Clone, Copy)]
struct XCase {
    seq: u64,
    d_model: u64,
    chunk: u64,
    extra_rows: u32,
}

/// The W pre-pass over `[d_model, hd]` in chunks of `chunk`.
#[derive(Debug, Clone, Copy)]
struct WCase {
    d_model: u64,
    hd: u64,
    chunk: u64,
}

/// Deterministic values in [-2, 2).
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0
        })
        .collect()
}

fn le32(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn text(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes).expect("ASCII").trim_end_matches('\0').to_string()
}

fn kir_ptx(kind: Prepass) -> String {
    text(kir_module(kind))
}

fn hand_ptx(kind: Prepass) -> String {
    match kind {
        Prepass::X => hand::CSHA_TIER_B1_PREPASS_X_PTX,
        Prepass::W => hand::CSHA_TIER_B1_PREPASS_W_PTX,
    }
    .trim_end_matches('\0')
    .to_string()
}

fn launch(ptx: &str, args: &HashMap<String, u64>, global: &mut [Segment], grid: u32, order: Order) {
    let prog = parse(ptx);
    let mut ctas: Vec<u32> = (0..grid).collect();
    if order == Order::Descending {
        ctas.reverse();
    }
    for cta in ctas {
        let mut launch = Launch {
            prog: &prog,
            args,
            global,
            // Poisoned with a finite f32: a read of a partial no thread
            // wrote this launch shows in the sum.
            shared: vec![0x3F; prog.shared_bytes],
            ctaid: cta,
            ctaid_y: 0,
            nctaid_y: 1,
            ntid: PREPASS_BLOCK,
            steps: 0,
        };
        run_cta(&mut launch, order);
    }
}

fn args(pairs: &[(&str, u64)]) -> HashMap<String, u64> {
    pairs.iter().map(|&(k, v)| (k.to_string(), v)).collect()
}

fn x_inputs(c: &XCase, seed: u64) -> (Vec<f32>, Vec<f32>) {
    let x = values((c.seq * c.d_model) as usize, seed);
    let gamma: Vec<f32> = values(c.d_model as usize, seed + 100).iter().map(|g| 0.5 + g.abs()).collect();
    (x, gamma)
}

fn run_x(ptx: &str, c: &XCase, (x, gamma): &(Vec<f32>, Vec<f32>), order: Order) -> Vec<Vec<u8>> {
    let mut global = vec![
        Segment { base: IN, bytes: le32(x) },
        Segment { base: GAMMA, bytes: le32(gamma) },
        Segment { base: OUT, bytes: vec![POISON; (c.seq * c.d_model * 2) as usize] },
    ];
    let a = args(&[
        ("x_in", IN),
        ("gamma", GAMMA),
        ("x_out", OUT),
        ("seq", c.seq),
        ("d_model", c.d_model),
        ("chunk", c.chunk),
        ("log2_chunk", c.chunk.trailing_zeros() as u64),
        ("eps", EPS.to_bits() as u64),
    ]);
    launch(ptx, &a, &mut global, c.seq as u32 + c.extra_rows, order);
    global.into_iter().map(|s| s.bytes).collect()
}

fn run_w(ptx: &str, c: &WCase, w: &[f32], order: Order) -> Vec<Vec<u8>> {
    let mut global = vec![
        Segment { base: IN, bytes: le32(w) },
        Segment { base: OUT, bytes: vec![POISON; (c.d_model * c.hd * 2) as usize] },
    ];
    let a = args(&[
        ("w_in", IN),
        ("w_out", OUT),
        ("d_model", c.d_model),
        ("hd", c.hd),
        ("chunk", c.chunk),
        ("log2_hd", c.hd.trailing_zeros() as u64),
        ("log2_chunk", c.chunk.trailing_zeros() as u64),
    ]);
    // `launch_w_prepass`'s grid.
    let grid = (c.d_model * c.hd).div_ceil(PREPASS_BLOCK as u64) as u32;
    launch(ptx, &a, &mut global, grid, order);
    global.into_iter().map(|s| s.bytes).collect()
}

fn x_cases() -> Vec<XCase> {
    vec![
        // Fewer columns than threads; one CTA past the rows.
        XCase { seq: 3, d_model: 64, chunk: 32, extra_rows: 1 },
        // Three trips of the column loop, the last ragged; eight-wide chunks.
        XCase { seq: 2, d_model: 600, chunk: 8, extra_rows: 0 },
        // Exactly one column per thread; two chunks.
        XCase { seq: 1, d_model: 256, chunk: 128, extra_rows: 0 },
        XCase { seq: 4, d_model: 96, chunk: 32, extra_rows: 2 },
    ]
}

fn w_cases() -> Vec<WCase> {
    vec![
        // Half a block.
        WCase { d_model: 32, hd: 4, chunk: 32 },
        // Three and a half blocks.
        WCase { d_model: 224, hd: 4, chunk: 32 },
        // Exact blocks, several chunk bands.
        WCase { d_model: 128, hd: 16, chunk: 64 },
        WCase { d_model: 96, hd: 8, chunk: 32 },
    ]
}

const ORDERS: [Order; 2] = [Order::Ascending, Order::Descending];

#[test]
fn the_kir_kernels_agree_with_the_hand_written_ones_bit_for_bit() {
    for (n, c) in x_cases().iter().enumerate() {
        let input = x_inputs(c, n as u64 + 1);
        for order in ORDERS {
            let hand = run_x(&hand_ptx(Prepass::X), c, &input, order);
            let kir = run_x(&kir_ptx(Prepass::X), c, &input, order);
            assert!(hand == kir, "X case {n} {c:?} under {order:?}: global memory differs");
        }
    }
    for (n, c) in w_cases().iter().enumerate() {
        let w = values((c.d_model * c.hd) as usize, n as u64 + 7);
        for order in ORDERS {
            let hand = run_w(&hand_ptx(Prepass::W), c, &w, order);
            let kir = run_w(&kir_ptx(Prepass::W), c, &w, order);
            assert!(hand == kir, "W case {n} {c:?} under {order:?}: global memory differs");
        }
    }
}

/// The f16 at element `i` of an output buffer.
fn half_at(bytes: &[u8], i: usize) -> f16 {
    f16::from_bits(u16::from_le_bytes([bytes[2 * i], bytes[2 * i + 1]]))
}

#[test]
fn the_x_answer_is_the_normalised_chunkified_row() {
    for (n, c) in x_cases().iter().enumerate() {
        let input = x_inputs(c, n as u64 + 1);
        let (x, gamma) = &input;
        let mem = run_x(&kir_ptx(Prepass::X), c, &input, Order::Ascending);
        assert_eq!(mem[0], le32(x), "case {n}: x is untouched");
        let (seq, dm, chunk) = (c.seq as usize, c.d_model as usize, c.chunk as usize);
        let mut seen = vec![false; seq * dm];
        for row in 0..seq {
            let r = &x[row * dm..(row + 1) * dm];
            let ms = r.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>() / dm as f64;
            let rms_inv = 1.0 / (ms + EPS as f64).sqrt();
            for d in 0..dm {
                let at = (d / chunk) * seq * chunk + row * chunk + d % chunk;
                seen[at] = true;
                let want = r[d] as f64 * rms_inv * gamma[d] as f64;
                let got = half_at(&mem[2], at).to_f64();
                assert!(
                    (got - want).abs() <= 1e-3 * want.abs() + 1e-6,
                    "case {n}: row {row} column {d}: {got} vs {want}"
                );
            }
        }
        // Every output element is written exactly by the layout above.
        assert!(seen.iter().all(|&s| s), "case {n}: the layout covers the output");
    }
}

#[test]
fn the_w_answer_is_the_col_major_chunkified_weight() {
    for (n, c) in w_cases().iter().enumerate() {
        let w = values((c.d_model * c.hd) as usize, n as u64 + 7);
        let mem = run_w(&kir_ptx(Prepass::W), c, &w, Order::Ascending);
        assert_eq!(mem[0], le32(&w), "case {n}: w is untouched");
        let (hd, chunk) = (c.hd as usize, c.chunk as usize);
        let mut want = vec![0u8; mem[1].len()];
        for d in 0..c.d_model as usize {
            for col in 0..hd {
                let at = (d / chunk) * hd * chunk + col * chunk + d % chunk;
                want[2 * at..2 * at + 2].copy_from_slice(&f16::from_f32(w[d * hd + col]).to_bits().to_le_bytes());
            }
        }
        assert!(mem[1] == want, "case {n} {c:?}: w_out is not the chunkified weight");
    }
}

#[test]
fn the_kir_kernels_keep_the_ffi_signatures_and_entry_names() {
    for kind in Prepass::ALL {
        let (h, k) = (hand_ptx(kind), kir_ptx(kind));
        assert_eq!(parse_signature(&k), parse_signature(&h), "{kind:?}");
        assert!(k.contains(&format!(".visible .entry {}(", kind.kernel_name())), "{kind:?}");
    }
}

// ---------------------------------------------------------------------------
// Mutation tests
// ---------------------------------------------------------------------------

/// Per kernel, a case where every bound and constant matters: a ragged
/// column loop and a CTA past the rows for X, a ragged last block for W.
fn caught(kind: Prepass, mutate: impl Fn(&str) -> String) -> bool {
    let hand = hand_ptx(kind);
    let mutant = mutate(&kir_ptx(kind));
    ORDERS.into_iter().any(|order| {
        let run = |p: &str| match kind {
            Prepass::X => {
                let c = XCase { seq: 2, d_model: 600, chunk: 8, extra_rows: 1 };
                run_x(p, &c, &x_inputs(&c, 5), order)
            }
            Prepass::W => {
                let c = WCase { d_model: 224, hd: 4, chunk: 32 };
                run_w(p, &c, &values(896, 9), order)
            }
        };
        let expect = run(&hand);
        match std::panic::catch_unwind(|| run(&mutant)) {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
    })
}

/// `ptx` with its `i`th line starting with `op` rewritten by `f`
/// (deleted when `f` returns `None`).
fn at_nth(ptx: &str, op: &str, i: usize, f: impl Fn(&str) -> Option<String>) -> String {
    let n = ptx.lines().enumerate().filter(|(_, l)| l.trim_start().starts_with(op)).nth(i).expect("the line").0;
    let mut out: Vec<String> = Vec::new();
    for (k, l) in ptx.lines().enumerate() {
        if k != n {
            out.push(l.to_string());
        } else if let Some(r) = f(l) {
            out.push(r);
        }
    }
    out.join("\n") + "\n"
}

fn count(ptx: &str, op: &str) -> usize {
    ptx.lines().filter(|l| l.trim_start().starts_with(op)).count()
}

/// The `i`th line containing `from`, rewritten to `to`.
fn nudge(ptx: &str, from: &str, to: &str, i: usize) -> String {
    let n = ptx.lines().enumerate().filter(|(_, l)| l.contains(from)).nth(i).expect("the constant").0;
    ptx.lines()
        .enumerate()
        .map(|(k, l)| if k == n { l.replacen(from, to, 1) } else { l.to_string() })
        .collect::<Vec<_>>()
        .join("\n")
        + "\n"
}

/// `ptx` with its one `mov.<ty>` line ending in `from` rewritten to end
/// in `to`.
fn remat(ptx: &str, ty: &str, from: &str, to: &str) -> String {
    let mov = format!("mov.{ty} ");
    let hit = |l: &str| l.trim_start().starts_with(&mov) && l.ends_with(from);
    assert_eq!(ptx.lines().filter(|l| hit(l)).count(), 1, "`{ty} {from}` is materialised once");
    ptx.lines().map(|l| if hit(l) { l.replace(from, to) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    for kind in Prepass::ALL {
        assert!(!caught(kind, |p| p.to_string()), "{kind:?}");
    }
}

/// After the partials are published, and after thread 0 publishes the
/// scale: without either, a thread reads shared memory not yet written.
#[test]
fn deleting_either_barrier_is_caught() {
    assert_eq!(count(&kir_ptx(Prepass::X), "bar.sync 0;"), 2);
    for i in 0..2 {
        assert!(caught(Prepass::X, |p| at_nth(p, "bar.sync 0;", i, |_| None)), "barrier {i}");
    }
}

/// `>=` -> `>` on every bound. X, in emission order: the row guard (the
/// extra CTA then normalises a row past the input), the sum-of-squares
/// loop (adds the next row's first square), the reduction (reads past the
/// partials) and the output loop (writes past the output). W: the guard
/// (the ragged block's tail reads past the weight).
#[test]
fn relaxing_any_bound_is_caught() {
    for (kind, ty, total) in [(Prepass::X, "u64", 3), (Prepass::X, "u32", 1), (Prepass::W, "u64", 1)] {
        let setp = format!("setp.ge.{ty} ");
        assert_eq!(count(&kir_ptx(kind), &setp), total, "{kind:?} {ty}");
        for i in 0..total {
            let relax = |p: &str| at_nth(p, &setp, i, |l| Some(l.replacen("setp.ge.", "setp.gt.", 1)));
            assert!(caught(kind, relax), "{kind:?}: bound {ty} {i} relaxed went unnoticed");
        }
    }
}

#[test]
fn nudging_a_baked_constant_is_caught() {
    // X: the column stride; f32's element size (the row, a partial's
    // address, the gamma); f16's.
    for (from, to, nth) in [
        (", 256;", ", 255;", 0),
        (", 4;", ", 2;", 0),
        (", 4;", ", 8;", 1),
        (", 4;", ", 2;", 2),
        (", 2;", ", 4;", 0),
    ] {
        assert!(caught(Prepass::X, |p| nudge(p, from, to, nth)), "X: `{from}` #{nth} -> `{to}` went unnoticed");
    }
    // X: the reduction's first partial, which is also its step; the 1 in
    // `chunk - 1`.
    for (ty, from, to) in [("u32", ", 1;", ", 2;"), ("u64", ", 1;", ", 2;")] {
        assert!(caught(Prepass::X, |p| remat(p, ty, from, to)), "X: `{ty} {from}` -> `{to}` went unnoticed");
    }
    // W: the 1 in `hd - 1` and `chunk - 1`; f32's and f16's sizes.
    for (from, to, nth) in [(", 1;", ", 2;", 0), (", 4;", ", 8;", 0), (", 2;", ", 4;", 0)] {
        assert!(caught(Prepass::W, |p| nudge(p, from, to, nth)), "W: `{from}` #{nth} -> `{to}` went unnoticed");
    }
}

/// Every CTA normalising row 0 (X), or every block converting the first
/// 256 weights (W).
#[test]
fn ignoring_the_block_index_is_caught() {
    for kind in Prepass::ALL {
        assert_eq!(kir_ptx(kind).matches("%ctaid.x;").count(), 1, "{kind:?}");
        assert!(caught(kind, |p| p.replacen("%ctaid.x;", "0;", 1)), "{kind:?}");
    }
}

#[test]
fn the_interpreter_knows_the_new_forms() {
    let hand = hand_ptx(Prepass::X);
    // An element-typed shared block addressed by name, the approximate
    // division, and the 64-bit integer conversion.
    for form in [".shared .f32 sdata[256];", "[sdata]", "div.approx.f32 ", "cvt.rn.f32.u64 "] {
        assert!(hand.contains(form), "{form}");
    }
    assert_eq!(parse(&hand).shared.get("sdata").map(|s| s.1), Some(1024));
    assert!(kir_ptx(Prepass::X).contains("cvt.rn.f32.u64 "));
    parse(&kir_ptx(Prepass::X));
}
