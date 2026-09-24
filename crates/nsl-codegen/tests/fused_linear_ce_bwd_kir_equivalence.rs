//! The differential equivalence gate for the fused linear-CE backward
//! (`nsl_fused_linear_ce_bwd_*`, roadmap A2 step 10, third and last slice).
//!
//! `fused_linear_ce` emitted the backward by hand, one kernel per dtype; it
//! now builds all three from one KIR builder (`build_backward`). This file
//! runs the frozen hand emitters (`tests/fixtures/fused_linear_ce_hand.rs`)
//! and the KIR one side by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), one CTA per row as the host
//! launches them.
//!
//! 1. **Agreement**: for every dtype and geometry below, under two thread
//!    schedules, the hand module and the KIR module leave *the same bytes*
//!    in all of global memory — `dx`, `dW` and `dbias`, including ignored
//!    rows (whose `dx` row is zeroed) and a target past the vocab. The
//!    scatters are `red.global.add.f32`, whose rounding depends on the order
//!    threads add in; the interpreter runs threads one at a time in the
//!    schedule's order, and both kernels add the same values in the same
//!    per-thread order, so the sums agree bit for bit.
//! 2. **Correctness**: those bytes are the gradient of the cross-entropy of
//!    the linear head, against an f64 reference over the same inputs.
//! 3. **The gate bites**: deleting any scatter, running any loop one trip
//!    long, relaxing a guard, nudging a baked constant, the ignore index or
//!    the target's `1` is caught, and the one equivalent mutant is named.

use std::collections::HashMap;

use half::{bf16, f16};
use nsl_codegen::fused_linear_ce::{synthesize_fused_linear_ce_backward_ptx, Dtype, FusedLinearCEConfig};

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
const LSE_BASE: u64 = 0x5000_0000;
const DX_BASE: u64 = 0x6000_0000;
const DW_BASE: u64 = 0x7000_0000;
const DBIAS_BASE: u64 = 0x8000_0000;
const BLOCK: u32 = 128;
const IGNORE: i64 = -100;
const GRAD_OUTPUT: f32 = 1.5;
/// What `dx` holds before the launch: a live row accumulates onto it, an
/// ignored row's is zeroed.
const DX_INIT: f32 = 0.25;
/// Indices of the output buffers in [`run`]'s memory.
const DX: usize = 5;
const DW: usize = 6;
const DBIAS: usize = 7;

#[derive(Debug, Clone)]
struct Case {
    dtype: Dtype,
    vocab: u32,
    hidden: u32,
    vocab_tile: u32,
    /// One per row; `IGNORE` rows skip, a target past the vocab is never
    /// matched (the backward reads no memory at the target).
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

    fn num_valid(&self) -> u32 {
        self.targets.iter().filter(|&&t| t != IGNORE).count() as u32
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
    /// `[x, w, bias, targets, lse]`.
    bytes: [Vec<u8>; 5],
    x: Vec<f32>,
    w: Vec<f32>,
    bias: Vec<f32>,
    lse: Vec<f32>,
}

/// Row `r`'s logit at column `j`, in f64 from the widened inputs.
fn logit(c: &Case, x: &[f32], w: &[f32], bias: &[f32], r: usize, j: usize) -> f64 {
    let h = c.hidden as usize;
    (0..h).map(|k| x[r * h + k] as f64 * w[j * h + k] as f64).sum::<f64>() + bias[j] as f64
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
    // The forward's saved log-sum-exp, from the same inputs.
    let lse: Vec<f32> = (0..c.rows())
        .map(|r| {
            let logits: Vec<f64> = (0..v).map(|j| logit(c, &x, &w, &bias, r, j)).collect();
            let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            (m + logits.iter().map(|l| (l - m).exp()).sum::<f64>().ln()) as f32
        })
        .collect();
    let lb = lse.iter().flat_map(|l| l.to_le_bytes()).collect();
    Inputs { bytes: [xb, wb, bb, tb, lb], x, w, bias, lse }
}

/// Global memory after launching `ptx` over every row: `[x, w, bias,
/// targets, lse, dx, dW, dbias]`.
fn run(ptx: &str, c: &Case, input: &Inputs, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let (v, h) = (c.vocab as usize, c.hidden as usize);
    let f32s_of = |n: usize, value: f32| -> Vec<u8> { (0..n).flat_map(|_| value.to_le_bytes()).collect() };
    let [xb, wb, bb, tb, lb] = input.bytes.clone();
    let mut global = vec![
        Segment { base: X_BASE, bytes: xb },
        Segment { base: W_BASE, bytes: wb },
        Segment { base: BIAS_BASE, bytes: bb },
        Segment { base: TARGETS_BASE, bytes: tb },
        Segment { base: LSE_BASE, bytes: lb },
        Segment { base: DX_BASE, bytes: f32s_of(c.rows() * h, DX_INIT) },
        Segment { base: DW_BASE, bytes: f32s_of(v * h, 0.0) },
        Segment { base: DBIAS_BASE, bytes: f32s_of(v, 0.0) },
    ];
    let args: HashMap<String, u64> = [
        ("grad_output", GRAD_OUTPUT.to_bits() as u64),
        ("x", X_BASE),
        ("w", W_BASE),
        ("bias", BIAS_BASE),
        ("targets", TARGETS_BASE),
        ("lse", LSE_BASE),
        ("dx_out", DX_BASE),
        ("dw_out", DW_BASE),
        ("dbias_out", DBIAS_BASE),
        ("B", 1),
        ("S", c.rows() as u64),
        ("V", c.vocab as u64),
        ("H", c.hidden as u64),
        ("num_valid", c.num_valid() as u64),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    // The launcher passes `shared_mem_bytes()`; the backward reads none.
    let shared_len = prog.shared_bytes + prog.dynamic_shared.map_or(0, |_| c.cfg().shared_mem_bytes() as usize);
    for row in 0..c.rows() as u32 {
        let mut launch = Launch {
            prog: &prog,
            args: &args,
            global: &mut global,
            shared: vec![0x7F; shared_len],
            ctaid: row,
            ctaid_y: 0,
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
    text(synthesize_fused_linear_ce_backward_ptx(&c.cfg()))
}

fn hand_ptx(c: &Case) -> String {
    text(hand::synthesize_fused_linear_ce_backward_ptx(&c.hand_cfg()))
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// `(dx, dW, dbias)` in f64: `g_v = (exp(logit_v - lse) - [v == target]) *
/// grad_output / num_valid` per live row, `dx[row] = DX_INIT + sum_v g_v
/// W[v]`, `dW[v] = sum_rows g_v x[row]`, `dbias[v] = sum_rows g_v`; an
/// ignored row's `dx` is zero.
fn reference(c: &Case, input: &Inputs) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let (v, h) = (c.vocab as usize, c.hidden as usize);
    let mut dx = vec![DX_INIT as f64; c.rows() * h];
    let mut dw = vec![0.0; v * h];
    let mut dbias = vec![0.0; v];
    let scale = GRAD_OUTPUT as f64 / c.num_valid() as f64;
    for (r, &t) in c.targets.iter().enumerate() {
        if t == IGNORE {
            dx[r * h..(r + 1) * h].iter_mut().for_each(|d| *d = 0.0);
            continue;
        }
        for j in 0..v {
            let p = (logit(c, &input.x, &input.w, &input.bias, r, j) - input.lse[r] as f64).exp();
            let g = (p - if j as i64 == t { 1.0 } else { 0.0 }) * scale;
            for k in 0..h {
                dx[r * h + k] += g * input.w[j * h + k] as f64;
                dw[j * h + k] += g * input.x[r * h + k] as f64;
            }
            dbias[j] += g;
        }
    }
    (dx, dw, dbias)
}

/// The dtypes, geometries and target patterns the agreement and correctness
/// gates sweep.
fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for dtype in [Dtype::F32, Dtype::F16, Dtype::Bf16] {
        // Three tiles with a ragged last one (600 = 2 * 256 + 88), two lanes
        // per thread; targets in every tile, an ignored row, a target past
        // the vocab, the last vocab entry.
        out.push(Case {
            dtype,
            vocab: 600,
            hidden: 24,
            vocab_tile: 256,
            targets: vec![5, IGNORE, 599, 300, 612, 0, 256],
        });
        // One tile holding the whole vocab; a hidden wider than the block,
        // so an ignored row's zeroing takes two trips.
        out.push(Case { dtype, vocab: 128, hidden: 136, vocab_tile: 128, targets: vec![127, IGNORE, 3] });
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
                "{c:?} {order:?}: global memory differs\nhand dbias: {:?}\nkir dbias:  {:?}",
                &f32s(&expect[DBIAS])[..8],
                &f32s(&got[DBIAS])[..8]
            );
        }
    }
}

#[test]
fn the_shared_answer_is_the_gradient() {
    for c in cases() {
        let input = inputs(&c, 0x5eed ^ c.vocab as u64);
        let mem = run(&kir_ptx(&c), &c, &input, Order::Ascending);
        let (dx, dw, dbias) = reference(&c, &input);
        // f32 accumulation (the dots, the scatters) against f64.
        let tol = 2e-5;
        for (name, got, want) in [("dx", f32s(&mem[DX]), dx), ("dW", f32s(&mem[DW]), dw), ("dbias", f32s(&mem[DBIAS]), dbias)] {
            assert_eq!(got.len(), want.len());
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((*g as f64 - w).abs() <= tol * (1.0 + w.abs()), "{c:?}: {name}[{i}] {g} vs {w}");
            }
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
        let module = kir_ptx(&c);
        assert!(module.contains(&format!(".visible .entry {}(", c.cfg().bwd_kernel_name())));
        // No shared memory, no barrier: every scatter is a reduction.
        assert!(!module.contains(".shared") && !module.contains("bar.sync"), "{module}");
        assert_eq!(module.matches("red.global.add.f32 ").count(), 3);
        assert!(!module.contains("atom."));
        let version = if c.dtype == Dtype::Bf16 { ".version 7.8\n.target sm_80" } else { ".version 7.0\n.target sm_70" };
        assert!(module.starts_with(version), "{c:?}");
    }
}

// ---------------------------------------------------------------------------
// The gate bites
// ---------------------------------------------------------------------------

/// Every mutation is judged on the first case's shape at f32 and at f16:
/// three ragged tiles, targets in every tile, an ignored row and a target
/// past the vocab. Its baked constants are distinct wherever a test nudges
/// one (vocab 600, hidden 24, tile 256, tiles 3, lanes per thread 2).
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

/// The lines of `ptx` that start with `op`, by line number.
fn lines_starting(ptx: &str, op: &str) -> Vec<(usize, String)> {
    ptx.lines().enumerate().filter(|(_, l)| l.trim_start().starts_with(op)).map(|(n, l)| (n, l.to_string())).collect()
}

/// `ptx` with the `i`th line starting with `op` rewritten from `op` to
/// `to`. Lines are found in the text being mutated: the f32 and f16
/// modules number their lines differently, and two lines may read the same.
fn at_nth(ptx: &str, op: &str, i: usize, to: &str) -> String {
    let n = lines_starting(ptx, op)[i].0;
    let mut out: Vec<String> = ptx.lines().map(str::to_string).collect();
    out[n] = out[n].replacen(op, to, 1);
    out.join("\n") + "\n"
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

#[test]
fn the_unmutated_kernel_is_not_caught() {
    // The baseline for every mutation below: without it, `caught` could be
    // reporting a difference the mutation did not cause.
    assert!(!caught(|p| p.to_string()));
}

#[test]
fn deleting_any_scatter_is_caught() {
    // dx, dW and dbias, in emission order.
    let c = mutation_cases().remove(0);
    let reds = lines_starting(&kir_ptx(&c), "red.global.add.f32 ");
    assert_eq!(reds.len(), 3);
    for (i, (_, red)) in reds.iter().enumerate() {
        assert!(
            caught(|p| {
                let n = lines_starting(p, "red.global.add.f32 ")[i].0;
                let mut lines: Vec<&str> = p.lines().collect();
                lines.remove(n);
                lines.join("\n") + "\n"
            }),
            "`{red}` deleted went unnoticed"
        );
    }
}

/// The kernel's `setp.lt.u32` comparisons, in emission order: the vocab
/// guard; the bottom tests of the tile loop, the sub-tile loop, the dot and
/// the scatter over `h`; the zeroing of an ignored row's `dx` (a guard at
/// the top of its loop).
const TILE_LOOP: usize = 1;

#[test]
fn running_any_loop_one_trip_long_or_relaxing_a_guard_is_caught() {
    // `<` -> `<=`. The vocab guard reads `W` one column past the vocab; one
    // more sub-tile repeats a column of the next tile; one more dot or
    // scatter trip reads the next row of `x` or `W`; the zeroing writes the
    // next row's first `dx`.
    let c = mutation_cases().remove(0);
    let lts = lines_starting(&kir_ptx(&c), "setp.lt.u32 ");
    assert_eq!(lts.len(), 6, "{lts:?}");
    for (i, (_, line)) in lts.iter().enumerate().filter(|(i, _)| *i != TILE_LOOP) {
        assert!(caught(|p| at_nth(p, "setp.lt.u32", i, "setp.le.u32")), "bound {i} `{line}` relaxed went unnoticed");
    }
}

/// One more trip of the tile loop is an equivalent mutant, in the hand
/// kernel as in this one: that tile starts at `num_tiles * vtile >= V`, so
/// the vocab guard turns every lane of it away and nothing is read or
/// written. One fewer tile is caught (below).
#[test]
fn the_tile_loops_extra_trip_is_a_named_equivalent_mutant() {
    assert!(
        !caught(|p| at_nth(p, "setp.lt.u32", TILE_LOOP, "setp.le.u32")),
        "the tile loop's extra trip matters after all"
    );
}

#[test]
fn nudging_a_baked_constant_is_caught() {
    for (ty, from, to) in [
        // The vocab bound; hidden as the loops' trip count and as the row
        // stride; the tile width; tiles (one fewer); lanes per thread; the
        // ignore index; the 1 taken off the target's probability.
        ("u32", ", 600;", ", 599;"),
        ("u32", ", 24;", ", 23;"),
        ("u64", ", 24;", ", 25;"),
        ("u32", ", 256;", ", 257;"),
        ("u32", ", 3;", ", 2;"),
        ("u32", ", 2;", ", 1;"),
        ("s64", ", -100;", ", -99;"),
        ("f32", ", 0f3F800000;", ", 0f3F800001;"),
    ] {
        let c = mutation_cases().remove(0);
        let mov = format!("mov.{ty} ");
        let expect = kir_ptx(&c).lines().filter(|l| l.trim_start().starts_with(&mov) && l.ends_with(from)).count();
        assert!(expect >= 1, "`{ty} {from}` is not materialised");
        assert!(caught(|p| remat(p, ty, from, to, expect)), "`{ty} {from}` -> `{to}` went unnoticed");
    }
}

#[test]
fn the_interpreter_knows_the_new_forms() {
    // The reductions and the f32 parameter are modelled rather than skipped.
    let c = cases().remove(0);
    let (hand, kir) = (hand_ptx(&c), kir_ptx(&c));
    assert!(hand.contains("red.global.add.f32 ") && kir.contains("red.global.add.f32 "));
    assert!(hand.contains("ld.param.f32 ") && kir.contains("ld.param.f32 "));
    parse(&hand);
    parse(&kir);
}
