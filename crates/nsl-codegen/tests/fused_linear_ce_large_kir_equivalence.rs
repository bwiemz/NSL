//! The differential equivalence gate for the fused linear-CE large-vocab
//! forward (`nsl_fused_linear_ce_fwd_large_{partials,finalize}_*`, roadmap
//! A2 step 10, second slice).
//!
//! `fused_linear_ce` emitted the two large-vocab kernels by hand, one pair
//! per dtype; it now builds each from one KIR builder
//! (`build_large_partials`, `build_large_finalize`) and lowers both into one
//! module under one header. This file runs the frozen hand emitters
//! (`tests/fixtures/fused_linear_ce_hand.rs`) and the KIR ones side by side
//! on the cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`),
//! the spec's level 3, launching them as the host does: Kernel A over every
//! `(tile, row)` of its two-dimensional grid, then Kernel B over every row.
//!
//! 1. **Agreement**: for every dtype and geometry below, under two thread
//!    schedules, the hand module and the KIR module leave *the same bytes*
//!    in all of global memory — the partials buffer, `loss_out` and
//!    `lse_out`, including ignored rows.
//! 2. **Correctness**: those bytes are the cross-entropy of the linear
//!    head: a reference in f64 over the same storage-dtype inputs.
//! 3. **The gate bites**: deleting the barrier, running any loop one trip
//!    long, relaxing the vocab guard, nudging a baked constant of either
//!    kernel or the tail's `-inf` is caught. No mutant is equivalent.
//!
//! As in the v1 gate (`fused_linear_ce_fwd_kir_equivalence.rs`), level 3
//! stands in for the SASS baseline the spec named: the KIR build keeps the
//! hand kernels' floating-point operations in order, and this compares
//! their results bit for bit. The interpreter models `ex2.approx` and
//! `lg2.approx` by their exact counterparts.
//!
//! The builders are called through `synthesize_large_vocab_forward_ptx`
//! directly for vocabularies below the routing threshold, which keeps the
//! interpreted grids small; one case above it goes through the router.

use std::collections::HashMap;

use half::{bf16, f16};
use nsl_codegen::fused_linear_ce::{
    synthesize_fused_linear_ce_ptx, synthesize_large_vocab_forward_ptx, Dtype, FusedLinearCEConfig,
    LARGE_VOCAB_THRESHOLD,
};

#[allow(dead_code)]
#[path = "fixtures/fused_linear_ce_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const X_BASE: u64 = 0x1000_0000;
const W_BASE: u64 = 0x2000_0000;
const BIAS_BASE: u64 = 0x3000_0000;
const TARGETS_BASE: u64 = 0x4000_0000;
const PARTIALS_BASE: u64 = 0x5000_0000;
const LOSS_BASE: u64 = 0x6000_0000;
const LSE_BASE: u64 = 0x7000_0000;
const BLOCK: u32 = 128;
const IGNORE: i64 = -100;
/// Indices of the output buffers in [`run`]'s memory.
const PARTIALS: usize = 4;
const LOSS: usize = 5;
const LSE: usize = 6;

#[derive(Debug, Clone)]
struct Case {
    dtype: Dtype,
    vocab: u32,
    hidden: u32,
    vocab_tile: u32,
    /// One per row, each `IGNORE` or in `[0, vocab)`: Kernel B does not
    /// range-check the target (the hand kernels did not either).
    targets: Vec<i64>,
}

impl Case {
    fn cfg(&self) -> FusedLinearCEConfig {
        FusedLinearCEConfig {
            vocab_size: self.vocab,
            hidden_size: self.hidden,
            seq_len: self.targets.len() as u32,
            batch_size: 1,
            vocab_tile: self.vocab_tile,
            gpu_sm: 80,
            dtype: self.dtype,
            ignore_index: IGNORE,
            max_vocab_v1: 262_144,
        }
    }

    fn hand_cfg(&self) -> hand::FusedLinearCEConfig {
        let c = self.cfg();
        hand::FusedLinearCEConfig {
            vocab_size: c.vocab_size,
            hidden_size: c.hidden_size,
            seq_len: c.seq_len,
            batch_size: c.batch_size,
            vocab_tile: c.vocab_tile,
            gpu_sm: c.gpu_sm,
            dtype: match self.dtype {
                Dtype::F32 => hand::Dtype::F32,
                Dtype::F16 => hand::Dtype::F16,
                Dtype::Bf16 => hand::Dtype::Bf16,
            },
            ignore_index: c.ignore_index,
            max_vocab_v1: c.max_vocab_v1,
        }
    }

    fn rows(&self) -> usize {
        self.targets.len()
    }

    fn tiles(&self) -> u32 {
        self.vocab.div_ceil(self.vocab_tile)
    }
}

/// Deterministic values in [-1, 1).
struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// A value in the storage dtype: its bytes, and the f32 it widens to.
fn stored(dtype: Dtype, v: f32) -> (Vec<u8>, f32) {
    match dtype {
        Dtype::F32 => (v.to_le_bytes().to_vec(), v),
        Dtype::F16 => {
            let h = f16::from_f32(v);
            (h.to_bits().to_le_bytes().to_vec(), h.to_f32())
        }
        Dtype::Bf16 => {
            let h = bf16::from_f32(v);
            (h.to_bits().to_le_bytes().to_vec(), h.to_f32())
        }
    }
}

/// Inputs as bytes, and as the f32 values the kernels widen them to.
struct Inputs {
    bytes: [Vec<u8>; 4],
    x: Vec<f32>,
    w: Vec<f32>,
    bias: Vec<f32>,
}

fn inputs(c: &Case, seed: u64) -> Inputs {
    let mut rng = Lcg(seed);
    let (v, h) = (c.vocab as usize, c.hidden as usize);
    // Logits of O(1): a dot of `h` terms of size ~1/sqrt(h).
    let scale = 2.0 / (h as f32).sqrt();
    let mut values = |n: usize, s: f32| -> (Vec<u8>, Vec<f32>) {
        let mut bytes = Vec::new();
        let mut vals = Vec::new();
        for _ in 0..n {
            let (b, f) = stored(c.dtype, rng.next() * s);
            bytes.extend(b);
            vals.push(f);
        }
        (bytes, vals)
    };
    let (xb, x) = values(c.rows() * h, 1.0);
    let (wb, w) = values(v * h, scale);
    let (bb, bias) = values(v, 0.5);
    let tb = c.targets.iter().flat_map(|t| t.to_le_bytes()).collect();
    Inputs { bytes: [xb, wb, bb, tb], x, w, bias }
}

/// The module's entries, each as a module of its own: the text before the
/// first entry (header and module-scope declarations), then that entry.
fn entries(module: &str) -> Vec<String> {
    let starts: Vec<usize> = module.match_indices(".visible .entry ").map(|(i, _)| i).collect();
    assert_eq!(starts.len(), 2, "the large-vocab module is Kernel A then Kernel B");
    let prelude = &module[..starts[0]];
    (0..starts.len())
        .map(|k| {
            let end = starts.get(k + 1).copied().unwrap_or(module.len());
            format!("{prelude}{}", &module[starts[k]..end])
        })
        .collect()
}

/// Global memory after launching the module as the host does: `[x, w,
/// bias, targets, partials, loss_out, lse_out]`.
fn run(module: &str, c: &Case, input: &Inputs, order: Order) -> Vec<Vec<u8>> {
    let [partials_kernel, finalize_kernel] = <[String; 2]>::try_from(entries(module)).unwrap();
    let (a, b) = (parse(&partials_kernel), parse(&finalize_kernel));
    // A NaN sentinel: an output the kernels fail to write shows.
    let sentinel = |n: usize| -> Vec<u8> { (0..n).flat_map(|_| 0x7FC0_0001u32.to_le_bytes()).collect() };
    let [xb, wb, bb, tb] = input.bytes.clone();
    let mut global = vec![
        Segment { base: X_BASE, bytes: xb },
        Segment { base: W_BASE, bytes: wb },
        Segment { base: BIAS_BASE, bytes: bb },
        Segment { base: TARGETS_BASE, bytes: tb },
        Segment { base: PARTIALS_BASE, bytes: sentinel(c.rows() * c.tiles() as usize * 2) },
        Segment { base: LOSS_BASE, bytes: sentinel(c.rows()) },
        Segment { base: LSE_BASE, bytes: sentinel(c.rows()) },
    ];
    let args: HashMap<String, u64> = [
        ("x", X_BASE),
        ("w", W_BASE),
        ("bias", BIAS_BASE),
        ("targets", TARGETS_BASE),
        ("partials", PARTIALS_BASE),
        ("loss_out", LOSS_BASE),
        ("lse_out", LSE_BASE),
        ("B", 1),
        ("S", c.rows() as u64),
        ("V", c.vocab as u64),
        ("H", c.hidden as u64),
        ("num_tiles", c.tiles() as u64),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    // Kernel A takes `shared_mem_bytes()` of dynamic shared memory; Kernel
    // B is launched with none.
    assert!(a.dynamic_shared.is_some(), "Kernel A's tile is dynamic shared memory");
    let shared_len = a.shared_bytes + c.cfg().shared_mem_bytes() as usize;
    for row in 0..c.rows() as u32 {
        for tile in 0..c.tiles() {
            let mut launch = Launch {
                prog: &a,
                args: &args,
                global: &mut global,
                // Poisoned with a finite f32 (3.4e38; a NaN in 16 bits): a
                // read of shared memory no thread wrote this launch shows
                // in the max as well as the sum.
                shared: vec![0x7F; shared_len],
                ctaid: tile,
                ctaid_y: row,
                nctaid_y: c.rows() as u32,
                ntid: BLOCK,
                steps: 0,
            };
            run_cta(&mut launch, order);
        }
    }
    for row in 0..c.rows() as u32 {
        let mut launch = Launch {
            prog: &b,
            args: &args,
            global: &mut global,
            shared: vec![0x7F; b.shared_bytes],
            ctaid: row,
            ctaid_y: 0,
            nctaid_y: 1,
            ntid: BLOCK,
            steps: 0,
        };
        run_cta(&mut launch, order);
    }
    global.into_iter().map(|s| s.bytes).collect()
}

fn text(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes.strip_suffix(&[0]).expect("null-terminated").to_vec()).unwrap()
}

fn kir_ptx(c: &Case) -> String {
    text(synthesize_large_vocab_forward_ptx(&c.cfg()))
}

fn hand_ptx(c: &Case) -> String {
    text(hand::synthesize_large_vocab_forward_ptx(&c.hand_cfg()))
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// `(loss, lse)` per row. Kernel A stages each logit in the storage dtype
/// before its tile's max and sum, so the lse is over rounded logits; Kernel
/// B recomputes the target's logit in f32 and does not round it. An ignored
/// row is `(0, 0)`.
fn reference(c: &Case, input: &Inputs) -> Vec<(f64, f64)> {
    let (v, h) = (c.vocab as usize, c.hidden as usize);
    let round = |x: f64| stored(c.dtype, x as f32).1 as f64;
    c.targets
        .iter()
        .enumerate()
        .map(|(r, &t)| {
            if t == IGNORE {
                return (0.0, 0.0);
            }
            let logit = |j: usize| -> f64 {
                (0..h).map(|k| input.x[r * h + k] as f64 * input.w[j * h + k] as f64).sum::<f64>()
                    + input.bias[j] as f64
            };
            let logits: Vec<f64> = (0..v).map(|j| round(logit(j))).collect();
            let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse = m + logits.iter().map(|l| (l - m).exp()).sum::<f64>().ln();
            (lse - logit(t as usize), lse)
        })
        .collect()
}

/// The dtypes, geometries and target patterns the agreement and correctness
/// gates sweep.
fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for dtype in [Dtype::F32, Dtype::F16, Dtype::Bf16] {
        // Three tiles with a ragged last one (1100 = 2 * 512 + 76), four
        // lanes per thread; targets in every tile, at lane 0 and past it,
        // an ignored row, and the last vocab entry.
        out.push(Case {
            dtype,
            vocab: 1100,
            hidden: 24,
            vocab_tile: 512,
            targets: vec![5, IGNORE, 1099, 600, 0, 1030],
        });
        // One tile holding the whole vocab exactly (one lane per thread),
        // and a row of only ignored targets but one.
        out.push(Case { dtype, vocab: 128, hidden: 40, vocab_tile: 128, targets: vec![IGNORE, 127, IGNORE] });
        // Many tiles of the narrowest width, the last holding one column.
        out.push(Case { dtype, vocab: 641, hidden: 8, vocab_tile: 128, targets: vec![640, 64, 300] });
    }
    out
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn the_kir_kernels_agree_with_the_hand_written_ones_bit_for_bit() {
    for c in cases() {
        let (hand, kir) = (hand_ptx(&c), kir_ptx(&c));
        let input = inputs(&c, 0x5eed ^ c.vocab as u64);
        for order in [Order::Ascending, Order::Descending] {
            let expect = run(&hand, &c, &input, order);
            let got = run(&kir, &c, &input, order);
            assert!(
                expect == got,
                "{c:?} {order:?}: global memory differs\nhand partials: {:?}\nkir partials:  {:?}\nhand loss: {:?}\nkir loss:  {:?}",
                f32s(&expect[PARTIALS]),
                f32s(&got[PARTIALS]),
                f32s(&expect[LOSS]),
                f32s(&got[LOSS])
            );
        }
    }
}

#[test]
fn a_vocab_past_the_threshold_routes_here_and_agrees() {
    // The router sends it to this module; four hidden units keep the
    // interpreted grid (9 tiles x 2 rows) quick.
    for dtype in [Dtype::F32, Dtype::Bf16] {
        let c = Case {
            dtype,
            vocab: LARGE_VOCAB_THRESHOLD + 8,
            hidden: 4,
            vocab_tile: 1024,
            targets: vec![LARGE_VOCAB_THRESHOLD as i64 + 3, 17],
        };
        assert!(c.cfg().is_large_vocab());
        let routed = text(synthesize_fused_linear_ce_ptx(&c.cfg()));
        assert_eq!(routed, kir_ptx(&c));
        let input = inputs(&c, 3);
        assert!(run(&hand_ptx(&c), &c, &input, Order::Ascending) == run(&routed, &c, &input, Order::Ascending));
    }
}

#[test]
fn the_shared_answer_is_the_cross_entropy() {
    for c in cases() {
        let input = inputs(&c, 0x5eed ^ c.vocab as u64);
        let mem = run(&kir_ptx(&c), &c, &input, Order::Ascending);
        let (loss, lse) = (f32s(&mem[LOSS]), f32s(&mem[LSE]));
        // f32 accumulation against f64: the dot and the sum each drift by a
        // few ulps; a 16-bit logit may round the other way at a tie.
        let tol = if c.dtype == Dtype::F32 { 1e-5 } else { 2e-3 };
        for (r, &(want_loss, want_lse)) in reference(&c, &input).iter().enumerate() {
            assert!(
                (loss[r] as f64 - want_loss).abs() <= tol * (1.0 + want_loss.abs()),
                "{c:?}: row {r} loss {} vs {want_loss}",
                loss[r]
            );
            assert!(
                (lse[r] as f64 - want_lse).abs() <= tol * (1.0 + want_lse.abs()),
                "{c:?}: row {r} lse {} vs {want_lse}",
                lse[r]
            );
        }
    }
}

#[test]
fn inputs_are_left_untouched() {
    let c = cases().remove(1);
    let input = inputs(&c, 7);
    let mem = run(&kir_ptx(&c), &c, &input, Order::Descending);
    for (i, want) in input.bytes.iter().enumerate() {
        assert!(&mem[i] == want, "the kernels wrote to input {i}");
    }
}

#[test]
fn the_kir_kernels_keep_the_ffi_signatures_and_entry_names() {
    for c in cases() {
        let (hand, kir) = (entries(&hand_ptx(&c)), entries(&kir_ptx(&c)));
        let names = |p: &Program| p.params.iter().map(|(ty, n)| format!("{ty} {n}")).collect::<Vec<_>>();
        for k in 0..2 {
            assert_eq!(names(&parse(&hand[k])), names(&parse(&kir[k])), "the launcher marshals these positionally");
        }
        let cfg = c.cfg();
        assert!(kir[0].contains(&format!(".visible .entry {}(", cfg.large_partials_kernel_name())));
        assert!(kir[1].contains(&format!(".visible .entry {}(", cfg.large_finalize_kernel_name())));
        // One header for the module, one shared block (Kernel A's tile, in
        // what the launcher passes); Kernel B touches no shared memory.
        let module = kir_ptx(&c);
        assert_eq!(module.matches(".version ").count(), 1);
        assert_eq!(module.matches(".shared ").count(), 1, "{module}");
        assert!(c.vocab_tile * c.dtype.bytes_per_elem() <= cfg.shared_mem_bytes());
        let finalize_body = &kir[1][kir[1].find(".visible .entry ").unwrap()..];
        assert!(!finalize_body.contains(".shared"));
        // The header covers the bf16 conversions.
        let version = if c.dtype == Dtype::Bf16 { ".version 7.8\n.target sm_80" } else { ".version 7.0\n.target sm_70" };
        assert!(module.starts_with(version), "{c:?}");
    }
}

// ---------------------------------------------------------------------------
// The gate bites
// ---------------------------------------------------------------------------

/// Every mutation is judged on the first case's shape at f32 and at f16:
/// three ragged tiles, targets in every tile and at lanes other than 0,
/// an ignored row. Its baked constants are distinct within each kernel
/// wherever a test nudges one (vocab 1100, hidden 24, tile 512, lanes per
/// thread 4 in Kernel A; tiles 3 and the pair stride 2 in Kernel B).
fn mutation_cases() -> Vec<Case> {
    cases().into_iter().filter(|c| c.vocab == 1100 && c.dtype != Dtype::Bf16).collect()
}

/// Whether `mutate(kir)` is told apart from the hand module for some
/// mutation case: under either schedule its memory differs, or the
/// interpreter faults on it.
fn caught(mutate: impl Fn(&str) -> String) -> bool {
    mutation_cases().into_iter().any(|c| {
        let input = inputs(&c, 11);
        let hand = hand_ptx(&c);
        let mutant = mutate(&kir_ptx(&c));
        [Order::Ascending, Order::Descending].into_iter().any(|order| {
            let expect = run(&hand, &c, &input, order);
            let (c, input) = (&c, &input);
            match std::panic::catch_unwind(|| run(&mutant, c, input, order)) {
                Ok(mem) => mem != expect,
                Err(_) => true,
            }
        })
    })
}

/// Kernel A's text (`k = 0`: the module up to Kernel B's entry, header
/// included) or Kernel B's (`k = 1`: the rest).
fn kernel_text(module: &str, k: usize) -> &str {
    let at = module.match_indices(".visible .entry ").map(|(i, _)| i).nth(1).expect("two entries");
    if k == 0 {
        &module[..at]
    } else {
        &module[at..]
    }
}

/// Kernel A (`k = 0`) or Kernel B (`k = 1`) of `module` rewritten by `f`,
/// the other left as it is.
fn in_kernel(module: &str, k: usize, f: impl Fn(&str) -> String) -> String {
    let (a, b) = (kernel_text(module, 0), kernel_text(module, 1));
    if k == 0 {
        format!("{}{b}", f(a))
    } else {
        format!("{a}{}", f(b))
    }
}

/// `ptx` with every `mov.<ty>` line ending in `from` rewritten to end in
/// `to`, after checking there are `expect` of them.
fn remat(ptx: &str, ty: &str, from: &str, to: &str, expect: usize) -> String {
    let mov = format!("mov.{ty} ");
    let hit = |l: &str| l.trim_start().starts_with(&mov) && l.ends_with(from);
    assert_eq!(ptx.lines().filter(|l| hit(l)).count(), expect, "`{ty} {from}` materialisations");
    let mut out = ptx.lines().map(|l| if hit(l) { l.replace(from, to) } else { l.to_string() }).collect::<Vec<_>>().join("\n");
    out.push('\n');
    out
}

/// The lines of `ptx` that start with `op`, by line number.
fn setp_lines(ptx: &str, op: &str) -> Vec<(usize, String)> {
    ptx.lines().enumerate().filter(|(_, l)| l.trim_start().starts_with(op)).map(|(n, l)| (n, l.to_string())).collect()
}

/// `ptx` with the `i`th line starting with `op` rewritten from `op` to
/// `to`. Lines are found in the text being mutated: the f32 and f16
/// modules number their lines differently, and two lines may read the same.
fn at_nth(ptx: &str, op: &str, i: usize, to: &str) -> String {
    let n = setp_lines(ptx, op)[i].0;
    let mut out: Vec<String> = ptx.lines().map(str::to_string).collect();
    out[n] = out[n].replacen(op, to, 1);
    out.join("\n") + "\n"
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    // The baseline for every mutation below: without it, `caught` could be
    // reporting a difference the mutation did not cause.
    assert!(!caught(|p| p.to_string()));
}

#[test]
fn deleting_the_barrier_is_caught() {
    // Thread 0's reduction reads every lane's logit.
    let c = mutation_cases().remove(0);
    assert_eq!(kir_ptx(&c).matches("bar.sync 0;").count(), 1);
    assert!(caught(|p| p.replacen("    bar.sync 0;\n", "", 1)));
}

/// Kernel A's sub-tile fill and dot, and Kernel B's fold and dot, test
/// `i >= n` at the top (`setp.ge.u32`); Kernel A's max and sum scans test
/// `i + 1 < n` at the bottom (`setp.lt.u32`, after the fill's vocab guard).
/// One trip more reads a lane past the tile (the poisoned pad), an element
/// of the next row, or a pair of the next row's partials.
#[test]
fn running_any_loop_one_trip_long_is_caught() {
    let c = mutation_cases().remove(0);
    let module = kir_ptx(&c);
    for (k, top_tested) in [(0, 2), (1, 2)] {
        let exits = setp_lines(kernel_text(&module, k), "setp.ge.u32 ");
        assert_eq!(exits.len(), top_tested, "kernel {k}: {exits:?}");
        for (i, (_, exit)) in exits.iter().enumerate() {
            assert!(
                caught(|p| in_kernel(p, k, |t| at_nth(t, "setp.ge.u32", i, "setp.gt.u32"))),
                "kernel {k}: `{exit}` one trip long went unnoticed"
            );
        }
    }
    let lts = setp_lines(kernel_text(&module, 0), "setp.lt.u32 ");
    assert_eq!(lts.len(), 3, "{lts:?}");
    for (i, (_, exit)) in lts.iter().enumerate().skip(1) {
        assert!(
            caught(|p| in_kernel(p, 0, |t| at_nth(t, "setp.lt.u32", i, "setp.le.u32"))),
            "scan `{exit}` one trip long went unnoticed"
        );
    }
    assert!(setp_lines(kernel_text(&module, 1), "setp.lt.u32 ").is_empty());
}

#[test]
fn relaxing_the_vocab_guard_is_caught() {
    // `<` -> `<=`: the fill reads W and bias one column past the vocab.
    let c = mutation_cases().remove(0);
    let (_, guard) = setp_lines(kernel_text(&kir_ptx(&c), 0), "setp.lt.u32 ").remove(0);
    assert!(guard.contains(", %r"), "{guard}");
    assert!(caught(|p| in_kernel(p, 0, |t| at_nth(t, "setp.lt.u32", 0, "setp.le.u32"))));
}

#[test]
fn nudging_a_baked_constant_is_caught() {
    for (k, ty, from, to, expect) in [
        // Kernel A: the vocab bound; hidden as the dot's trip count and as
        // the row stride; the tile width (the tile's stride and the scans'
        // bound); lanes per thread; tiles and the pair stride of the
        // partials slot; the ignore index.
        (0, "u32", ", 1100;", ", 1099;", 1),
        (0, "u32", ", 24;", ", 23;", 1),
        (0, "u64", ", 24;", ", 25;", 1),
        (0, "u32", ", 512;", ", 511;", 1),
        (0, "u32", ", 4;", ", 3;", 1),
        (0, "u64", ", 3;", ", 2;", 1),
        (0, "u64", ", 2;", ", 1;", 1),
        (0, "s64", ", -100;", ", -99;", 1),
        // Kernel B: the fold's trip count, the row's partials (tiles and
        // pair stride), the pair stride within the row, hidden as the dot's
        // trip count and as each row stride, the ignore index.
        (1, "u32", ", 3;", ", 2;", 1),
        (1, "u64", ", 3;", ", 4;", 1),
        (1, "u64", ", 2;", ", 1;", 1),
        (1, "u32", ", 2;", ", 1;", 1),
        (1, "u32", ", 24;", ", 23;", 1),
        (1, "u64", ", 24;", ", 25;", 1),
        (1, "s64", ", 24;", ", 25;", 1),
        (1, "s64", ", -100;", ", -99;", 1),
    ] {
        assert!(
            caught(|p| in_kernel(p, k, |t| remat(t, ty, from, to, expect))),
            "kernel {k}: `{ty} {from}` -> `{to}` went unnoticed"
        );
    }
}

#[test]
fn a_finite_tail_is_caught() {
    // A lane past the vocab holds `-inf` so that neither the max nor the sum
    // sees it; 0 instead shifts the sum of the ragged tile. The same value
    // seeds the max scan, where a 0 is only caught for a tile whose logits
    // are all negative.
    let c = mutation_cases().remove(0);
    assert!(kir_ptx(&c).contains("0fFF800000;"));
    assert!(caught(|p| in_kernel(p, 0, |t| remat(t, "f32", ", 0fFF800000;", ", 0f00000000;", 1))));
}

#[test]
fn the_interpreter_knows_the_new_forms() {
    // The two-dimensional grid and the signed 64-bit arithmetic on the
    // target are modelled rather than skipped.
    let c = cases().remove(0);
    let (hand, kir) = (hand_ptx(&c), kir_ptx(&c));
    assert!(hand.contains("%ctaid.y") && kir.contains("%ctaid.y"));
    assert!(hand.contains("mul.lo.s64 ") && kir.contains("mul.lo.s64 "));
    for entry in entries(&hand).iter().chain(entries(&kir).iter()) {
        parse(entry);
    }
}
