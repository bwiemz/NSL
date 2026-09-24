//! The differential equivalence gate for the CPKD fused KL-CE distillation
//! kernels (`nsl_fused_kl_ce_*` and `nsl_fused_kl_ce_backward_*`, roadmap
//! A2 step 10).
//!
//! `cpkd_fused_loss` emitted both kernels by hand; it now builds them as
//! KIR (`build_forward`, `build_backward`). This file runs the frozen hand
//! emitters (`tests/fixtures/cpkd_fused_loss_hand.rs`) and the KIR ones side
//! by side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), one CTA per row as the host launches
//! them:
//!
//! 1. **Agreement**: under two thread schedules, the hand and KIR modules
//!    leave *the same bytes* in all of global memory — the forward's loss
//!    and three LSEs, the backward's `dx_s`, `dW_s` and `dbias_s`
//!    (`red.global.add.f32`, added in the schedule's thread order by both),
//!    including ignored rows.
//! 2. **Correctness**: those bytes are the distillation loss and its
//!    gradient: the crate's f64 references (`reference_forward_f64`,
//!    `reference_backward_f64`) over the same inputs.
//! 3. **The gate bites**: deleting any barrier or scatter, running any loop
//!    one trip long, relaxing a guard, nudging a baked constant or the ignore
//!    index is caught, and the equivalent mutants are named.
//!
//! The interpreter models `ex2.approx`, `lg2.approx` and `rcp.approx` by
//! their exact counterparts; the device suite (`cpkd_fused_kl_ce_numerical.rs`)
//! and the ptxas gate carry fidelity to the machine.

use std::collections::HashMap;

use nsl_codegen::cpkd_fused_loss::{
    reference_backward_f64, reference_forward_f64, synthesize_fused_kl_ce_backward_ptx,
    synthesize_fused_kl_ce_ptx, FusedKlCeConfig,
};

#[allow(dead_code)]
#[path = "fixtures/cpkd_fused_loss_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const XS: u64 = 0x1000_0000;
const WS: u64 = 0x2000_0000;
const BS: u64 = 0x3000_0000;
const XT: u64 = 0x4000_0000;
const WT: u64 = 0x5000_0000;
const BT: u64 = 0x6000_0000;
const TARGETS: u64 = 0x7000_0000;
/// The forward's outputs, and the backward's saved-LSE inputs.
const OUT0: u64 = 0x8000_0000;
const OUT1: u64 = 0x9000_0000;
const OUT2: u64 = 0xA000_0000;
const OUT3: u64 = 0xB000_0000;
const BLOCK: u32 = 128;
const IGNORE: i64 = -100;
const ALPHA: f32 = 0.3;
const TEMP: f32 = 2.5;
const GRAD_OUTPUT: f32 = 1.5;
/// What `dx_s` holds before the backward: a live row accumulates onto it,
/// an ignored row's is zeroed.
const DX_INIT: f32 = 0.25;

#[derive(Debug, Clone)]
struct Case {
    vocab: u32,
    hs: u32,
    ht: u32,
    vocab_tile: u32,
    /// One per row, each `IGNORE` or in `[0, vocab)`.
    targets: Vec<i64>,
}

impl Case {
    fn cfg(&self) -> FusedKlCeConfig {
        FusedKlCeConfig {
            vocab_size: self.vocab,
            student_hidden: self.hs,
            teacher_hidden: self.ht,
            batch_size: 1,
            seq_len: self.targets.len() as u32,
            vocab_tile: self.vocab_tile,
            gpu_sm: 80,
            ignore_index: IGNORE,
        }
    }

    fn hand_cfg(&self) -> hand::FusedKlCeConfig {
        let c = self.cfg();
        hand::FusedKlCeConfig {
            vocab_size: c.vocab_size,
            student_hidden: c.student_hidden,
            teacher_hidden: c.teacher_hidden,
            batch_size: c.batch_size,
            seq_len: c.seq_len,
            vocab_tile: c.vocab_tile,
            gpu_sm: c.gpu_sm,
            ignore_index: c.ignore_index,
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

struct Inputs {
    xs: Vec<f32>,
    ws: Vec<f32>,
    bs: Vec<f32>,
    xt: Vec<f32>,
    wt: Vec<f32>,
    bt: Vec<f32>,
}

fn inputs(c: &Case, seed: u64) -> Inputs {
    let mut rng = Lcg(seed);
    let (v, hs, ht, rows) = (c.vocab as usize, c.hs as usize, c.ht as usize, c.rows());
    let mut values = |n: usize, scale: f32| -> Vec<f32> { (0..n).map(|_| rng.next() * scale).collect() };
    // Logits of O(1): a dot of `h` terms of size ~1/sqrt(h).
    let xs = values(rows * hs, 1.0);
    let ws = values(v * hs, 2.0 / (hs as f32).sqrt());
    let bs = values(v, 0.5);
    let xt = values(rows * ht, 1.0);
    let wt = values(v * ht, 3.0 / (ht as f32).sqrt());
    let bt = values(v, 0.5);
    Inputs { xs, ws, bs, xt, wt, bt }
}

fn bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

fn f64s(v: &[f32]) -> Vec<f64> {
    v.iter().map(|&x| x as f64).collect()
}

/// The six inputs and the targets as segments, then `outputs`.
fn memory(c: &Case, input: &Inputs, outputs: [Vec<u8>; 4]) -> Vec<Segment> {
    let [o0, o1, o2, o3] = outputs;
    vec![
        Segment { base: XS, bytes: bytes(&input.xs) },
        Segment { base: WS, bytes: bytes(&input.ws) },
        Segment { base: BS, bytes: bytes(&input.bs) },
        Segment { base: XT, bytes: bytes(&input.xt) },
        Segment { base: WT, bytes: bytes(&input.wt) },
        Segment { base: BT, bytes: bytes(&input.bt) },
        Segment { base: TARGETS, bytes: c.targets.iter().flat_map(|t| t.to_le_bytes()).collect() },
        Segment { base: OUT0, bytes: o0 },
        Segment { base: OUT1, bytes: o1 },
        Segment { base: OUT2, bytes: o2 },
        Segment { base: OUT3, bytes: o3 },
    ]
}

fn args(c: &Case, extra: &[(&str, u64)]) -> HashMap<String, u64> {
    let mut a: HashMap<String, u64> = [
        ("xs", XS),
        ("ws", WS),
        ("bs", BS),
        ("xt", XT),
        ("wt", WT),
        ("bt", BT),
        ("targets", TARGETS),
        ("rows", c.rows() as u64),
        ("V", c.vocab as u64),
        ("HS", c.hs as u64),
        ("HT", c.ht as u64),
        ("alpha", ALPHA.to_bits() as u64),
        ("temp", TEMP.to_bits() as u64),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    for (k, v) in extra {
        a.insert(k.to_string(), *v);
    }
    a
}

fn launch_rows(prog: &Program, c: &Case, args: &HashMap<String, u64>, global: &mut [Segment], order: Order) {
    let shared_len = prog.shared_bytes + prog.dynamic_shared.map_or(0, |_| c.cfg().shared_mem_bytes() as usize);
    for row in 0..c.rows() as u32 {
        let mut launch = Launch {
            prog,
            args,
            global,
            // Poisoned with a finite f32 (3.4e38): a read of shared memory
            // no thread wrote this launch shows in a max as well as a sum.
            shared: vec![0x7F; shared_len],
            ctaid: row,
            ctaid_y: 0,
            nctaid_y: 1,
            ntid: BLOCK,
            steps: 0,
        };
        run_cta(&mut launch, order);
    }
}

/// Global memory after the forward: inputs, then `[loss, lse_s1, lse_sT,
/// lse_tT]`.
fn run_forward(ptx: &str, c: &Case, input: &Inputs, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    // A NaN sentinel: an output the kernel fails to write shows.
    let out = || (0..c.rows()).flat_map(|_| 0x7FC0_0001u32.to_le_bytes()).collect::<Vec<u8>>();
    let mut global = memory(c, input, [out(), out(), out(), out()]);
    let a = args(
        c,
        &[("loss_out", OUT0), ("lse_s1_out", OUT1), ("lse_st_out", OUT2), ("lse_tt_out", OUT3)],
    );
    launch_rows(&prog, c, &a, &mut global, order);
    global.into_iter().map(|s| s.bytes).collect()
}

/// The reference LSEs, as the backward's saved inputs.
fn saved_lses(c: &Case, input: &Inputs) -> [Vec<f32>; 3] {
    let (_, s1, st, tt, _) = reference(c, input);
    let f = |v: Vec<f64>| v.into_iter().map(|x| x as f32).collect::<Vec<f32>>();
    [f(s1), f(st), f(tt)]
}

/// Global memory after the backward: inputs, `[lse_s1, lse_sT, lse_tT,
/// dx_s]`, then `[dW_s, dbias_s]`.
fn run_backward(ptx: &str, c: &Case, input: &Inputs, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let (v, hs) = (c.vocab as usize, c.hs as usize);
    let [s1, st, tt] = saved_lses(c, input);
    let dx = vec![DX_INIT; c.rows() * hs];
    let mut global = memory(c, input, [bytes(&s1), bytes(&st), bytes(&tt), bytes(&dx)]);
    const DW: u64 = 0xC000_0000;
    const DB: u64 = 0xD000_0000;
    global.push(Segment { base: DW, bytes: bytes(&vec![0.0; v * hs]) });
    global.push(Segment { base: DB, bytes: bytes(&vec![0.0; v]) });
    let a = args(
        c,
        &[
            ("grad_output", GRAD_OUTPUT.to_bits() as u64),
            ("lse_s1", OUT0),
            ("lse_st", OUT1),
            ("lse_tt", OUT2),
            ("dxs_out", OUT3),
            ("dws_out", DW),
            ("dbs_out", DB),
            ("num_valid", c.num_valid() as u64),
        ],
    );
    launch_rows(&prog, c, &a, &mut global, order);
    global.into_iter().map(|s| s.bytes).collect()
}

fn text(bytes: Vec<u8>) -> String {
    String::from_utf8(bytes.strip_suffix(&[0]).expect("null-terminated").to_vec()).unwrap()
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Kernel {
    Forward,
    Backward,
}

fn kir_ptx(c: &Case, k: Kernel) -> String {
    text(match k {
        Kernel::Forward => synthesize_fused_kl_ce_ptx(&c.cfg()),
        Kernel::Backward => synthesize_fused_kl_ce_backward_ptx(&c.cfg()),
    })
}

fn hand_ptx(c: &Case, k: Kernel) -> String {
    text(match k {
        Kernel::Forward => hand::synthesize_fused_kl_ce_ptx(&c.hand_cfg()),
        Kernel::Backward => hand::synthesize_fused_kl_ce_backward_ptx(&c.hand_cfg()),
    })
}

fn run(ptx: &str, c: &Case, input: &Inputs, order: Order, k: Kernel) -> Vec<Vec<u8>> {
    match k {
        Kernel::Forward => run_forward(ptx, c, input, order),
        Kernel::Backward => run_backward(ptx, c, input, order),
    }
}

#[allow(clippy::type_complexity)]
fn reference(c: &Case, i: &Inputs) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>, usize) {
    reference_forward_f64(
        &f64s(&i.xs),
        &f64s(&i.ws),
        &f64s(&i.bs),
        &f64s(&i.xt),
        &f64s(&i.wt),
        &f64s(&i.bt),
        &c.targets,
        c.rows(),
        c.vocab as usize,
        c.hs as usize,
        c.ht as usize,
        ALPHA as f64,
        TEMP as f64,
        IGNORE,
    )
}

/// The geometries and target patterns the agreement and correctness gates
/// sweep.
fn cases() -> Vec<Case> {
    vec![
        // Three tiles with a ragged last one (600 = 2 * 256 + 88), two lanes
        // per thread; targets in every tile, at lane 0 and past it, ignored
        // rows, the last vocab entry; teacher wider than the student.
        Case { vocab: 600, hs: 32, ht: 64, vocab_tile: 256, targets: vec![5, IGNORE, 599, 300, 0, 256, IGNORE] },
        // One tile holding the whole vocab; a student hidden wider than the
        // block (an ignored row's zeroing takes two trips).
        Case { vocab: 128, hs: 160, ht: 32, vocab_tile: 128, targets: vec![127, IGNORE, 3] },
    ]
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn the_kir_kernels_agree_with_the_hand_written_ones_bit_for_bit() {
    for k in [Kernel::Forward, Kernel::Backward] {
        for c in cases() {
            let (hand, kir) = (hand_ptx(&c, k), kir_ptx(&c, k));
            let input = inputs(&c, 0x5eed ^ c.vocab as u64);
            for order in [Order::Ascending, Order::Descending] {
                let expect = run(&hand, &c, &input, order, k);
                let got = run(&kir, &c, &input, order, k);
                assert!(
                    expect == got,
                    "{k:?} {c:?} {order:?}: global memory differs\nhand out0: {:?}\nkir out0:  {:?}",
                    f32s(&expect[7]),
                    f32s(&got[7])
                );
            }
        }
    }
}

#[test]
fn the_forward_answer_is_the_distillation_loss() {
    for c in cases() {
        let input = inputs(&c, 0x5eed ^ c.vocab as u64);
        let mem = run_forward(&kir_ptx(&c, Kernel::Forward), &c, &input, Order::Ascending);
        let (loss, s1, st, tt, _) = reference(&c, &input);
        for (name, got, want) in [("loss", &mem[7], loss), ("lse_s1", &mem[8], s1), ("lse_sT", &mem[9], st), ("lse_tT", &mem[10], tt)] {
            for (r, (g, w)) in f32s(got).iter().zip(&want).enumerate() {
                assert!((*g as f64 - w).abs() <= 2e-5 * (1.0 + w.abs()), "{c:?}: {name}[{r}] {g} vs {w}");
            }
        }
    }
}

#[test]
fn the_backward_answer_is_the_gradient() {
    for c in cases() {
        let input = inputs(&c, 0x5eed ^ c.vocab as u64);
        let mem = run_backward(&kir_ptx(&c, Kernel::Backward), &c, &input, Order::Ascending);
        let i = &input;
        let (dx, dw, db) = reference_backward_f64(
            &f64s(&i.xs),
            &f64s(&i.ws),
            &f64s(&i.bs),
            &f64s(&i.xt),
            &f64s(&i.wt),
            &f64s(&i.bt),
            &c.targets,
            c.rows(),
            c.vocab as usize,
            c.hs as usize,
            c.ht as usize,
            ALPHA as f64,
            TEMP as f64,
            IGNORE,
            GRAD_OUTPUT as f64,
        );
        let hs = c.hs as usize;
        // A live row's dx accumulates onto DX_INIT; an ignored row's is zero.
        let dx: Vec<f64> = dx
            .iter()
            .enumerate()
            .map(|(k, d)| if c.targets[k / hs] == IGNORE { 0.0 } else { d + DX_INIT as f64 })
            .collect();
        for (name, got, want) in [("dx_s", &mem[10], dx), ("dW_s", &mem[11], dw), ("dbias_s", &mem[12], db)] {
            let got = f32s(got);
            assert_eq!(got.len(), want.len(), "{name}");
            for (k, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((*g as f64 - w).abs() <= 2e-5 * (1.0 + w.abs()), "{c:?}: {name}[{k}] {g} vs {w}");
            }
        }
    }
}

#[test]
fn the_kir_kernels_keep_the_ffi_signatures_and_entry_names() {
    for c in cases() {
        for k in [Kernel::Forward, Kernel::Backward] {
            let (hand, kir) = (parse(&hand_ptx(&c, k)), parse(&kir_ptx(&c, k)));
            let names = |p: &Program| p.params.iter().map(|(ty, n)| format!("{ty} {n}")).collect::<Vec<_>>();
            assert_eq!(names(&hand), names(&kir), "{k:?}: the launcher marshals these positionally");
            let module = kir_ptx(&c, k);
            let name = if k == Kernel::Forward { c.cfg().kernel_name() } else { c.cfg().bwd_kernel_name() };
            assert!(module.contains(&format!(".visible .entry {name}(")));
            assert!(module.starts_with(".version 7.0\n.target sm_70"), "{module}");
        }
        // The forward's tiles fit what the launcher passes; the backward
        // scatters with reductions and reads no shared memory.
        assert!(2 * c.vocab_tile * 4 + 4 <= c.cfg().shared_mem_bytes());
        let bwd = kir_ptx(&c, Kernel::Backward);
        assert!(!bwd.contains(".shared") && !bwd.contains("bar.sync"));
        assert_eq!(bwd.matches("red.global.add.f32 ").count(), 3);
    }
}

#[test]
fn the_backward_emits_no_teacher_gradient() {
    // Invariant I-11 at the ABI: every pointer the backward writes through
    // is one of the student's three gradients.
    let c = cases().remove(0);
    let names: Vec<String> = parse(&kir_ptx(&c, Kernel::Backward)).params.into_iter().map(|(_, n)| n).collect();
    let outputs: Vec<&String> = names.iter().filter(|n| n.ends_with("_out")).collect();
    assert_eq!(outputs, ["dxs_out", "dws_out", "dbs_out"]);
}

#[test]
fn inputs_are_left_untouched() {
    let c = cases().remove(1);
    let input = inputs(&c, 7);
    for k in [Kernel::Forward, Kernel::Backward] {
        let mem = run(&kir_ptx(&c, k), &c, &input, Order::Descending, k);
        let want = memory(&c, &input, [vec![], vec![], vec![], vec![]]);
        for i in 0..7 {
            assert!(mem[i] == want[i].bytes, "{k:?}: the kernel wrote to input {i}");
        }
    }
}

// ---------------------------------------------------------------------------
// The gate bites
// ---------------------------------------------------------------------------

/// Every mutation is judged on the first case: three ragged tiles, targets
/// in every tile, ignored rows. Its baked constants are distinct wherever a
/// test nudges one (vocab 600, student hidden 32, teacher hidden 64, tile
/// 256, tiles 3, lanes per thread 2).
fn mutation_case() -> Case {
    cases().remove(0)
}

/// Whether `mutate(kir)` of kernel `k` is told apart from the hand kernel:
/// under either schedule its memory differs, or the interpreter faults.
fn caught(k: Kernel, mutate: impl Fn(&str) -> String) -> bool {
    let c = mutation_case();
    let input = inputs(&c, 11);
    let hand = hand_ptx(&c, k);
    let mutant = mutate(&kir_ptx(&c, k));
    [Order::Ascending, Order::Descending].into_iter().any(|order| {
        let expect = run(&hand, &c, &input, order, k);
        let (c, input) = (&c, &input);
        match std::panic::catch_unwind(|| run(&mutant, c, input, order, k)) {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
    })
}

/// The lines of `ptx` that start with `op`.
fn lines_starting(ptx: &str, op: &str) -> Vec<String> {
    ptx.lines().filter(|l| l.trim_start().starts_with(op)).map(str::to_string).collect()
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

fn relax(ptx: &str, i: usize) -> String {
    at_nth(ptx, "setp.lt.u32 ", i, |l| Some(l.replacen("setp.lt.u32", "setp.le.u32", 1)))
}

/// `ptx` with every `mov.<ty>` line ending in `from` rewritten to end in
/// `to`, after checking there is at least one.
fn remat(ptx: &str, ty: &str, from: &str, to: &str) -> String {
    let mov = format!("mov.{ty} ");
    let hit = |l: &str| l.trim_start().starts_with(&mov) && l.ends_with(from);
    assert!(ptx.lines().any(hit), "`{ty} {from}` is not materialised");
    ptx.lines().map(|l| if hit(l) { l.replace(from, to) } else { l.to_string() }).collect::<Vec<_>>().join("\n") + "\n"
}

#[test]
fn the_unmutated_kernels_are_not_caught() {
    // The baseline for every mutation below.
    for k in [Kernel::Forward, Kernel::Backward] {
        assert!(!caught(k, |p| p.to_string()), "{k:?}");
    }
}

/// The forward's barriers, in emission order: after thread 0's `-inf` store
/// to the target slot; after the tile fill; after thread 0's reduction.
#[test]
fn deleting_any_forward_barrier_is_caught() {
    let n = lines_starting(&kir_ptx(&mutation_case(), Kernel::Forward), "bar.sync 0;").len();
    assert_eq!(n, 3);
    for i in 0..n {
        assert!(caught(Kernel::Forward, |p| at_nth(p, "bar.sync 0;", i, |_| None)), "barrier {i} deleted went unnoticed");
    }
}

#[test]
fn deleting_any_backward_scatter_is_caught() {
    for i in 0..3 {
        assert!(caught(Kernel::Backward, |p| at_nth(p, "red.global.add.f32 ", i, |_| None)), "scatter {i} deleted went unnoticed");
    }
}

/// The forward's `setp.lt.u32` comparisons, in emission order (KIR prints
/// blocks as they were created): the fill's vocab guard; the bottom tests of
/// the sub-tile loop, the student dot, the teacher dot and the tile loop;
/// thread 0's max scan (its vocab guard, its tile bound) and sum scan (the
/// same two).
const FWD_TILE_LOOP: usize = 4;
const FWD_MAX_SCAN_VOCAB: usize = 5;
const FWD_MAX_SCAN_TILE: usize = 6;

/// The backward's: the vocab guard; the bottom tests of the tile loop, the
/// sub-tile loop, the two dots and the scatter over `h`; the ignored row's
/// zeroing (a guard at the top of its loop).
const BWD_TILE_LOOP: usize = 1;

#[test]
fn running_any_loop_one_trip_long_or_relaxing_a_guard_is_caught() {
    // `<` -> `<=`: an extra dot or scatter trip reads the next row; an extra
    // sub-tile repeats a column of the next tile; the vocab guard reads one
    // column past the vocab; the sum scans add a lane past the vocab or the
    // tile; the zeroing writes the next row's first `dx_s`.
    let c = mutation_case();
    for (k, total, equivalent) in [
        (Kernel::Forward, 9, &[FWD_TILE_LOOP, FWD_MAX_SCAN_VOCAB][..]),
        (Kernel::Backward, 7, &[BWD_TILE_LOOP][..]),
    ] {
        let lts = lines_starting(&kir_ptx(&c, k), "setp.lt.u32 ");
        assert_eq!(lts.len(), total, "{k:?}: {lts:#?}");
        for i in (0..total).filter(|i| !equivalent.contains(i)) {
            assert!(caught(k, |p| relax(p, i)), "{k:?}: bound {i} `{}` relaxed went unnoticed", lts[i]);
        }
    }
}

/// The named equivalent mutants, in the hand kernels as in these:
///
/// * one more trip of either tile loop starts past the vocab, where the
///   vocab guard turns every lane away (and, in the forward, thread 0's
///   scans stop before their first lane, so the rescale sees `-inf` maxima
///   and leaves every family as it was);
/// * the forward max scan's vocab guard relaxed reads, in the ragged last
///   tile, a lane an earlier tile of the row wrote, whose logit the running
///   maxima already cover, and `max` is idempotent.
///
/// The max scan's tile bound is not among them: in a full tile the lane past
/// the student tile is the teacher tile's first.
#[test]
fn the_named_equivalent_mutants_are_equivalent() {
    for (k, i) in [(Kernel::Forward, FWD_TILE_LOOP), (Kernel::Forward, FWD_MAX_SCAN_VOCAB), (Kernel::Backward, BWD_TILE_LOOP)] {
        assert!(!caught(k, |p| relax(p, i)), "{k:?}: bound {i} matters after all");
    }
    assert!(caught(Kernel::Forward, |p| relax(p, FWD_MAX_SCAN_TILE)));
}

#[test]
fn nudging_a_baked_constant_is_caught() {
    for k in [Kernel::Forward, Kernel::Backward] {
        for (ty, from, to) in [
            // The vocab bound; the student and teacher hidden, as trip
            // counts and as row strides; the tile width; tiles (one fewer);
            // lanes per thread; the ignore index; the 1 in `1 - alpha` (and,
            // in the backward, taken off the target's probability).
            ("u32", ", 600;", ", 599;"),
            ("u32", ", 32;", ", 31;"),
            ("u64", ", 32;", ", 33;"),
            ("u32", ", 64;", ", 63;"),
            ("u64", ", 64;", ", 65;"),
            ("u32", ", 256;", ", 257;"),
            ("u32", ", 3;", ", 2;"),
            ("u32", ", 2;", ", 1;"),
            ("s64", ", -100;", ", -99;"),
            ("f32", ", 0f3F800000;", ", 0f3F800001;"),
        ] {
            assert!(caught(k, |p| remat(p, ty, from, to)), "{k:?}: `{ty} {from}` -> `{to}` went unnoticed");
        }
    }
}

/// The forward's `-inf` seeds the target slot and every running and tile
/// maximum. A 0 in its place never wins a `max` while every tile holds a
/// positive logit, as the shared inputs' tiles do, so there it changes
/// nothing; it is judged on inputs whose logits are all negative, where the
/// maxima stay at 0 and every exponent is taken against the wrong shift.
#[test]
fn a_finite_max_seed_is_caught_on_negative_logits() {
    let c = mutation_case();
    let mut input = inputs(&c, 11);
    input.bs.iter_mut().for_each(|b| *b -= 8.0);
    input.bt.iter_mut().for_each(|b| *b -= 8.0);
    let hand = hand_ptx(&c, Kernel::Forward);
    let mutant = remat(&kir_ptx(&c, Kernel::Forward), "f32", ", 0fFF800000;", ", 0f00000000;");
    let expect = run(&hand, &c, &input, Order::Ascending, Kernel::Forward);
    assert!(expect == run(&kir_ptx(&c, Kernel::Forward), &c, &input, Order::Ascending, Kernel::Forward));
    assert!(run(&mutant, &c, &input, Order::Ascending, Kernel::Forward) != expect);
}

#[test]
fn the_interpreter_knows_the_new_forms() {
    // The reciprocal of the temperature is modelled rather than skipped.
    let c = mutation_case();
    for k in [Kernel::Forward, Kernel::Backward] {
        let (hand, kir) = (hand_ptx(&c, k), kir_ptx(&c, k));
        assert!(hand.contains("rcp.approx.f32 ") && kir.contains("rcp.approx.f32 "));
        parse(&hand);
        parse(&kir);
    }
}
