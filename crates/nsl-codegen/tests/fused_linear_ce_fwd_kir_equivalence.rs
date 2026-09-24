//! The differential equivalence gate for the fused linear-CE v1 forward
//! kernel (`nsl_fused_linear_ce_{f32,f16,bf16}_v*_h*`, roadmap A2 step 10,
//! first slice).
//!
//! `fused_linear_ce` emitted one hand-assembled PTX kernel per dtype; it now
//! builds all three from one KIR builder (`build_forward`). This file runs
//! the frozen hand emitters (`tests/fixtures/fused_linear_ce_hand.rs`) and
//! the KIR one side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), the spec's level 3:
//!
//! 1. **Agreement**: for every dtype and geometry below, under two thread
//!    schedules, the hand module and the KIR module leave *the same bytes*
//!    in all of global memory — `loss_out` and `lse_out` for every row,
//!    including ignored rows and a row whose target is outside the vocab.
//! 2. **Correctness**: those bytes are the cross-entropy of the linear
//!    head: a reference in f64 over the same storage-dtype inputs, with each
//!    logit rounded to the storage dtype as the kernel's shared tile holds
//!    it.
//! 3. **The gate bites**: deleting any barrier, nudging a baked shape
//!    constant or the ignore index, or relaxing the vocab guard is caught.
//!
//! The A2 spec named SASS-baseline equivalence (level 2) for the loss heads,
//! because the online-softmax loops reorder under scheduling. Scheduling
//! reorders machine instructions, not the PTX's floating-point operations,
//! which the KIR build keeps in the hand kernels' order; level 3 compares
//! those operations' results bit for bit, which a SASS instruction count
//! does not. The device suites (`fused_linear_ce_numerical.rs` and the dtype
//! variants) and the ptxas gates carry fidelity to the machine. As in the
//! CFIE gates, the interpreter models `ex2.approx` and `lg2.approx` by their
//! exact counterparts.

use std::collections::HashMap;

use half::{bf16, f16};
use nsl_codegen::fused_linear_ce::{synthesize_fused_linear_ce_ptx, Dtype, FusedLinearCEConfig};

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
const LOSS_BASE: u64 = 0x5000_0000;
const LSE_BASE: u64 = 0x6000_0000;
const BLOCK: u32 = 128;
const IGNORE: i64 = -100;

#[derive(Debug, Clone)]
struct Case {
    dtype: Dtype,
    vocab: u32,
    hidden: u32,
    vocab_tile: u32,
    /// One per row; `IGNORE` rows skip, and a target at or past the vocab
    /// is never found (its loss is `+inf`, as the hand kernel's).
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
            max_vocab_v1: 8192,
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

/// The storage-dtype rounding of an f32 (the kernel's shared logits tile).
fn round_to(dtype: Dtype, v: f32) -> f32 {
    stored(dtype, v).1
}

/// Inputs as bytes, and as the f32 values the kernel widens them to.
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

/// Global memory after running `ptx` over every row: `[x, w, bias, targets,
/// loss_out, lse_out]`.
fn run(ptx: &str, c: &Case, input: &Inputs, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    // A NaN sentinel: an output the kernel fails to write shows.
    let out = || -> Vec<u8> { (0..c.rows()).flat_map(|_| 0x7FC0_0001u32.to_le_bytes()).collect() };
    let [xb, wb, bb, tb] = input.bytes.clone();
    let mut global = vec![
        Segment { base: X_BASE, bytes: xb },
        Segment { base: W_BASE, bytes: wb },
        Segment { base: BIAS_BASE, bytes: bb },
        Segment { base: TARGETS_BASE, bytes: tb },
        Segment { base: LOSS_BASE, bytes: out() },
        Segment { base: LSE_BASE, bytes: out() },
    ];
    let args: HashMap<String, u64> = [
        ("x", X_BASE),
        ("w", W_BASE),
        ("bias", BIAS_BASE),
        ("targets", TARGETS_BASE),
        ("loss_out", LOSS_BASE),
        ("lse_out", LSE_BASE),
        ("B", 1),
        ("S", c.rows() as u64),
        ("V", c.vocab as u64),
        ("H", c.hidden as u64),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    // The launcher passes `shared_mem_bytes()` of dynamic shared memory.
    assert!(prog.dynamic_shared.is_some(), "the forward's tile is dynamic shared memory");
    let shared_len = prog.shared_bytes + c.cfg().shared_mem_bytes() as usize;
    for ctaid in 0..c.rows() as u32 {
        let mut launch = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            // Poisoned (0xFF.. is a NaN as f32, f16 and bf16): a read of
            // shared memory no thread wrote this launch shows.
            shared: vec![0xFF; shared_len],
            ctaid,
            ntid: BLOCK,
            steps: 0,
        };
        run_cta(&mut launch, order);
    }
    global.into_iter().map(|s| s.bytes).collect()
}

fn kir_ptx(c: &Case) -> String {
    let bytes = synthesize_fused_linear_ce_ptx(&c.cfg());
    String::from_utf8(bytes.strip_suffix(&[0]).expect("null-terminated").to_vec()).unwrap()
}

fn hand_ptx(c: &Case) -> String {
    let bytes = hand::synthesize_fused_linear_ce_ptx(&c.hand_cfg());
    String::from_utf8(bytes.strip_suffix(&[0]).expect("null-terminated").to_vec()).unwrap()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// `(loss, lse)` per row: logits from the widened inputs (an f32 fma chain
/// in the kernel; f64 here), rounded to the storage dtype, then an f64
/// log-sum-exp. An ignored row is `(0, 0)`; a target outside the vocab
/// leaves the kernel's `-inf` slot, so its loss is `+inf`.
fn reference(c: &Case, input: &Inputs) -> Vec<(f64, f64)> {
    let (v, h) = (c.vocab as usize, c.hidden as usize);
    c.targets
        .iter()
        .enumerate()
        .map(|(r, &t)| {
            if t == IGNORE {
                return (0.0, 0.0);
            }
            let logits: Vec<f64> = (0..v)
                .map(|j| {
                    let dot: f64 = (0..h).map(|k| input.x[r * h + k] as f64 * input.w[j * h + k] as f64).sum();
                    round_to(c.dtype, (dot + input.bias[j] as f64) as f32) as f64
                })
                .collect();
            let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let lse = m + logits.iter().map(|l| (l - m).exp()).sum::<f64>().ln();
            let at = if (0..v as i64).contains(&t) { logits[t as usize] } else { f64::NEG_INFINITY };
            (lse - at, lse)
        })
        .collect()
}

/// The dtypes, geometries and target patterns the agreement and correctness
/// gates sweep.
fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for dtype in [Dtype::F32, Dtype::F16, Dtype::Bf16] {
        // Three tiles with a ragged last one (600 = 2 * 256 + 88); targets
        // in every tile, at lane 0 and past it, an ignored row, a target
        // past the vocab, and the last vocab entry.
        out.push(Case {
            dtype,
            vocab: 600,
            hidden: 32,
            vocab_tile: 256,
            targets: vec![5, IGNORE, 0, 300, 599, 612, 130, 511],
        });
        // One tile holding the whole vocab exactly; a wider hidden.
        out.push(Case { dtype, vocab: 128, hidden: 64, vocab_tile: 128, targets: vec![127, 1, IGNORE] });
        // A tile wider than the block (four lanes per thread) and a vocab
        // one past a tile.
        out.push(Case { dtype, vocab: 513, hidden: 32, vocab_tile: 512, targets: vec![512, 200, 0] });
    }
    out
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn the_kir_kernel_agrees_with_the_hand_written_one_bit_for_bit() {
    for c in cases() {
        let (hand, kir) = (hand_ptx(&c), kir_ptx(&c));
        let input = inputs(&c, 0x5eed ^ c.vocab as u64);
        for order in [Order::Ascending, Order::Descending] {
            let expect = run(&hand, &c, &input, order);
            let got = run(&kir, &c, &input, order);
            assert!(
                expect == got,
                "{c:?} {order:?}: global memory differs\nhand loss: {:?}\nkir loss:  {:?}",
                f32s(&expect[4]),
                f32s(&got[4])
            );
        }
    }
}

#[test]
fn the_shared_answer_is_the_cross_entropy() {
    for c in cases() {
        let input = inputs(&c, 0x5eed ^ c.vocab as u64);
        let mem = run(&kir_ptx(&c), &c, &input, Order::Ascending);
        let (loss, lse) = (f32s(&mem[4]), f32s(&mem[5]));
        // f32 accumulation against f64: the dot and the sum each drift by a
        // few ulps; a 16-bit logit may round the other way at a tie.
        let tol = if c.dtype == Dtype::F32 { 1e-5 } else { 2e-3 };
        for (r, &(want_loss, want_lse)) in reference(&c, &input).iter().enumerate() {
            if want_loss.is_infinite() {
                assert_eq!(loss[r], f32::INFINITY, "{c:?}: row {r}");
            } else {
                assert!((loss[r] as f64 - want_loss).abs() <= tol * (1.0 + want_loss.abs()), "{c:?}: row {r} loss {} vs {want_loss}", loss[r]);
            }
            assert!((lse[r] as f64 - want_lse).abs() <= tol * (1.0 + want_lse.abs()), "{c:?}: row {r} lse {} vs {want_lse}", lse[r]);
        }
    }
}

#[test]
fn inputs_are_left_untouched() {
    let c = cases().remove(1);
    let input = inputs(&c, 7);
    let mem = run(&kir_ptx(&c), &c, &input, Order::Descending);
    for (i, want) in input.bytes.iter().enumerate() {
        assert!(&mem[i] == want, "the kernel wrote to input {i}");
    }
}

#[test]
fn the_kir_kernel_keeps_the_ffi_signature_and_entry_name() {
    for c in cases() {
        let (hand, kir) = (parse(&hand_ptx(&c)), parse(&kir_ptx(&c)));
        let names = |p: &Program| p.params.iter().map(|(ty, n)| format!("{ty} {n}")).collect::<Vec<_>>();
        assert_eq!(names(&hand), names(&kir), "the launcher marshals these positionally");
        assert!(kir_ptx(&c).contains(&format!(".visible .entry {}(", c.cfg().kernel_name())));
        // Both tiles are dynamic shared memory the launch sizes, and the
        // KIR layout fits in what the launcher passes.
        assert!(hand.dynamic_shared.is_some() && kir.dynamic_shared.is_some());
        let need = c.vocab_tile * c.dtype.bytes_per_elem() + c.dtype.bytes_per_elem();
        assert!(need <= c.cfg().shared_mem_bytes());
    }
}

// ---------------------------------------------------------------------------
// The gate bites
// ---------------------------------------------------------------------------

/// Every mutation is judged on the first case's shape at f32 and at f16:
/// three ragged tiles, targets in every tile and at lanes other than 0,
/// an ignored row. Its baked constants are distinct wherever a test
/// nudges one (vocab 600, hidden 32, tile 256, tiles 3, lanes per thread
/// 2).
fn mutation_cases() -> Vec<Case> {
    cases().into_iter().filter(|c| c.vocab == 600 && c.dtype != Dtype::Bf16).collect()
}

/// Whether `mutate(kir)` is told apart from the hand kernel for some
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

/// `ptx` with the `n`th line equal to `line` (trimmed) removed.
fn without_nth(ptx: &str, line: &str, n: usize) -> String {
    let mut seen = 0;
    let mut out = String::new();
    for l in ptx.lines() {
        if l.trim() == line {
            seen += 1;
            if seen == n + 1 {
                continue;
            }
        }
        out.push_str(l);
        out.push('\n');
    }
    assert!(seen > n, "fewer than {} `{line}` lines", n + 1);
    out
}

/// `ptx` with every `mov.<ty>` line ending in `from` rewritten to end in
/// `to`, after checking there are `expect` of them.
fn remat(ptx: &str, ty: &str, from: &str, to: &str, expect: usize) -> String {
    let mov = format!("mov.{ty} ");
    let hit = |l: &str| l.trim_start().starts_with(&mov) && l.ends_with(from);
    assert_eq!(ptx.lines().filter(|l| hit(l)).count(), expect, "`{from}` materialisations");
    ptx.lines().map(|l| if hit(l) { l.replace(from, to) } else { l.to_string() }).collect::<Vec<_>>().join("\n")
}

#[test]
fn the_unmutated_kernel_is_not_caught() {
    // The baseline for every mutation below: without it, `caught` could be
    // reporting a difference the mutation did not cause.
    assert!(!caught(|p| p.to_string()));
}

/// The barriers, in emission order: after thread 0's `-inf` store to the
/// target slot (0); after the tile fill (1); after thread 0's reduction
/// (2). Each orders something:
///
/// * 0: a thread whose column is the target writes the slot in the first
///   tile; without the barrier, thread 0's `-inf` store can land after it.
/// * 1: thread 0's reduction reads every lane's logit.
/// * 2: the next tile's fill overwrites logits thread 0 is still reading.
#[test]
fn deleting_any_barrier_is_caught() {
    let c = mutation_cases().remove(0);
    let barriers = kir_ptx(&c).lines().filter(|l| l.trim() == "bar.sync 0;").count();
    assert_eq!(barriers, 3);
    for n in 0..barriers {
        assert!(caught(|p| without_nth(p, "bar.sync 0;", n)), "barrier {n} deleted went unnoticed");
    }
}

#[test]
fn nudging_a_baked_constant_is_caught() {
    for (ty, from, to, expect) in [
        // The vocab bound (the fill guard and both scans share it).
        ("u32", ", 600;", ", 599;", 1),
        // Hidden, as the dot's trip count and as the row stride.
        ("u32", ", 32;", ", 31;", 1),
        ("u64", ", 32;", ", 33;", 1),
        // The tile width is both the tile's stride and the scans' bound;
        // one less re-tiles the vocab with neither gap nor double count (an
        // equivalent mutant), one more leaves a gap and scans a lane past
        // the tile.
        ("u32", ", 256;", ", 257;", 1),
        // Tiles, and lanes per thread.
        ("u32", ", 3;", ", 2;", 1),
        ("u32", ", 2;", ", 1;", 1),
        ("s64", ", -100;", ", -99;", 1),
    ] {
        assert!(caught(|p| remat(p, ty, from, to, expect)), "`{ty} {from}` -> `{to}` went unnoticed");
    }
}

/// The kernel's `setp.lt.u32` comparisons, in emission order: the fill's
/// vocab guard (0); thread 0's max scan, its vocab guard (1) and tile bound
/// (2); its sum scan, the same two (3, 4).
const SCAN_MAX_BOUNDS: [usize; 2] = [1, 2];

#[test]
fn relaxing_a_vocab_or_tile_bound_is_caught() {
    // `<` -> `<=`: the fill then reads W and bias one column past the
    // vocab; a scan folds a lane of the tile past the vocab, or the lane
    // past the tile.
    let c = mutation_cases().remove(0);
    let bounds = lt_bounds(&kir_ptx(&c));
    assert_eq!(bounds.len(), 5, "{bounds:?}");
    for (n, g) in bounds.iter().enumerate().filter(|(n, _)| !SCAN_MAX_BOUNDS.contains(n)) {
        let relaxed = g.replace("setp.lt.u32", "setp.le.u32");
        assert!(caught(|p| p.replacen(g, &relaxed, 1)), "bound {n} `{g}` relaxed went unnoticed");
    }
}

/// Relaxing either of the max scan's bounds is an equivalent mutant, in the
/// hand kernels as in this one. The one extra lane each lets the scan read
/// holds nothing the running max does not already cover, and `max` is
/// idempotent, so the new running max, and everything after it, is
/// unchanged:
///
/// * past the vocab (in the ragged last tile), the lane holds the logit an
///   earlier tile of the same row wrote there;
/// * past the tile (in a full one), it is the logit-at-target slot: `-inf`
///   until the target's tile, then the target's logit, which that tile's
///   scan read.
///
/// The sum scan's bounds are not equivalent: an extra term changes the sum.
#[test]
fn the_max_scans_bounds_are_named_equivalent_mutants() {
    let c = mutation_cases().remove(0);
    let bounds = lt_bounds(&kir_ptx(&c));
    for n in SCAN_MAX_BOUNDS {
        let relaxed = bounds[n].replace("setp.lt.u32", "setp.le.u32");
        assert!(!caught(|p| p.replacen(&bounds[n], &relaxed, 1)), "max-scan bound {n} matters after all");
    }
}

fn lt_bounds(ptx: &str) -> Vec<String> {
    ptx.lines().filter(|l| l.trim_start().starts_with("setp.lt.u32 ")).map(str::to_string).collect()
}

#[test]
fn the_interpreter_knows_the_new_forms() {
    // Dynamic shared memory, i64 targets and their negative ignore index,
    // lg2 and the bf16 conversions are all modelled rather than skipped.
    for dtype in [Dtype::F32, Dtype::Bf16] {
        let c = Case { dtype, ..cases().remove(0) };
        let kir = kir_ptx(&c);
        for form in [".extern .shared ", "ld.global.s64 ", "setp.eq.s64 ", "lg2.approx.f32 ", "cvt.s64.u32 "] {
            assert!(kir.contains(form), "{form}");
        }
        if dtype == Dtype::Bf16 {
            assert!(kir.contains("cvt.rn.bf16.f32 ") && kir.contains("cvt.f32.bf16 "));
        }
        parse(&hand_ptx(&c));
        parse(&kir);
    }
}
