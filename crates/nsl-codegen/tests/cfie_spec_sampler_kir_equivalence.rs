//! The differential equivalence gate for the CFIE draft sampler and verify
//! prob-row writer (roadmap A2 step 9, third slice).
//!
//! `cfie_spec_sampler_ptx` emitted both kernels as hand-assembled PTX; it
//! now builds them as KIR. This file runs the frozen hand emitter
//! (`tests/fixtures/cfie_spec_sampler_hand.rs`) and the KIR one side by
//! side on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), the spec's level 3:
//!
//! 1. **Agreement** — for every geometry below, and under two thread
//!    schedules, the hand module and the KIR module leave *the same bytes*
//!    in all of global memory.
//! 2. **Correctness** — the draft's token is `cpu_reference_draft_sample`'s
//!    and its probability and the verify row are the references' to float
//!    tolerance (the interpreter's `ex2` and `rsqrt` are exact where the
//!    references' are libm's; nothing else differs).
//! 3. **The self-speculation anchor** — with the same weights, the draft's
//!    `p = 1 / sum` equals the verify row at the drafted token bit for bit.
//!    The engine's determinism proof rests on exactly this.
//! 4. **The gate bites** — deleting a barrier that matters, nudging a
//!    baked constant, relaxing the first-max-wins comparison, or dropping
//!    a tail-tile guard is caught. The mutants that are *not* caught are
//!    named, with why they are equivalent.

use std::collections::HashMap;

use half::f16;
use nsl_codegen::cfie_spec_sampler_ptx::{
    cpu_reference_draft_sample, cpu_reference_verify_probs, emit_draft_sample, emit_verify_probs,
    SpecSamplerConfig,
};

/// The one `crate::` path the frozen emitter names, supplied from the
/// library under test (the fixture is included into this test crate).
mod gpu_specs {
    pub use nsl_codegen::gpu_specs::ptx_isa_for_sm;
}

#[allow(dead_code)]
#[path = "fixtures/cfie_spec_sampler_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const HIDDEN_BASE: u64 = 0x1000_0000;
const NORM_BASE: u64 = 0x2000_0000;
const W_BASE: u64 = 0x3000_0000;
const OUT_BASE: u64 = 0x4000_0000;
const PROB_BASE: u64 = 0x5000_0000;
const BLOCK: u32 = 128;
/// A NaN no kernel computes: an output the kernel fails to write shows.
const SENTINEL: u32 = 0x7FC0_0001;

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kernel {
    Draft,
    Verify,
}

#[derive(Debug, Clone, Copy)]
struct Geometry {
    d_model: u32,
    vocab: u32,
}

impl Geometry {
    fn cfg(&self) -> SpecSamplerConfig {
        SpecSamplerConfig { d_model: self.d_model, vocab_size: self.vocab, vocab_tile: 128 }
    }

    fn hand_cfg(&self) -> hand::SpecSamplerConfig {
        hand::SpecSamplerConfig { d_model: self.d_model, vocab_size: self.vocab, vocab_tile: 128, sm_version: 80 }
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
    hidden: Vec<f32>,
    norm_w: Vec<f32>,
    /// f16 bits, `[vocab][d_model]`.
    lm_head: Vec<u16>,
}

impl Inputs {
    fn lm_head_f32(&self) -> Vec<f32> {
        self.lm_head.iter().map(|&b| f16::from_bits(b).to_f32()).collect()
    }
}

/// Random inputs, with one planted tie: the row of the token the draft
/// would pick is copied to a later token, so first-max-wins is what
/// decides between them.
fn inputs(g: Geometry, seed: u64) -> Inputs {
    scaled_inputs(g, seed, 3.0)
}

/// [`inputs`] with the hidden row drawn from `[-scale, scale)`.
fn scaled_inputs(g: Geometry, seed: u64, scale: f32) -> Inputs {
    let mut rng = Lcg(seed);
    let dm = g.d_model as usize;
    let hidden = (0..dm).map(|_| rng.next() * scale).collect();
    let norm_w = (0..dm).map(|_| 0.5 + rng.next().abs()).collect();
    let lm_head = (0..g.vocab as usize * dm).map(|_| f16::from_f32(rng.next()).to_bits()).collect();
    let mut input = Inputs { hidden, norm_w, lm_head };
    if g.vocab >= 2 {
        let (best, _) = cpu_reference_draft_sample(&g.cfg(), &input.hidden, &input.norm_w, &input.lm_head_f32());
        let twin = if (best as usize) + 1 < g.vocab as usize { g.vocab as usize - 1 } else { 0 };
        let row: Vec<u16> = input.lm_head[best as usize * dm..(best as usize + 1) * dm].to_vec();
        input.lm_head[twin * dm..(twin + 1) * dm].copy_from_slice(&row);
    }
    input
}

/// Global memory after running `ptx`: `[hidden, norm_w, lm_head, out,
/// prob]`, where `out` is the token (draft) or the row (verify) and `prob`
/// is the draft's probability (unused by verify).
fn run(ptx: &str, kernel: Kernel, g: Geometry, input: &Inputs, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let out_words = match kernel {
        Kernel::Draft => 1,
        Kernel::Verify => g.vocab as usize,
    };
    let mut global = vec![
        Segment { base: HIDDEN_BASE, bytes: f32s(&input.hidden) },
        Segment { base: NORM_BASE, bytes: f32s(&input.norm_w) },
        Segment { base: W_BASE, bytes: input.lm_head.iter().flat_map(|b| b.to_le_bytes()).collect() },
        Segment { base: OUT_BASE, bytes: (0..out_words).flat_map(|_| SENTINEL.to_le_bytes()).collect() },
        Segment { base: PROB_BASE, bytes: SENTINEL.to_le_bytes().to_vec() },
    ];
    let mut args: HashMap<String, u64> = [
        ("hidden_ptr", HIDDEN_BASE),
        ("norm_w_ptr", NORM_BASE),
        ("lm_head_ptr", W_BASE),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    match kernel {
        Kernel::Draft => {
            args.insert("out_token_ptr".into(), OUT_BASE);
            args.insert("out_prob_ptr".into(), PROB_BASE);
            args.insert("rng_seed".into(), 0x5eed);
        }
        Kernel::Verify => {
            args.insert("out_probs_ptr".into(), OUT_BASE);
        }
    }
    let mut launch = Launch {
        prog: &prog,
        args: &args,
        global: &mut global,
        // Poisoned (0xFF.. is a NaN as f32): a read of shared memory no
        // thread wrote this launch shows in the output.
        shared: vec![0xFF; prog.shared_bytes],
        ctaid: 0,
        ctaid_y: 0,
        nctaid_y: 1,
        ntid: BLOCK,
        steps: 0,
    };
    run_cta(&mut launch, order);
    global.into_iter().map(|s| s.bytes).collect()
}

fn kir_ptx(kernel: Kernel, g: Geometry) -> String {
    match kernel {
        Kernel::Draft => emit_draft_sample(&g.cfg()).0,
        Kernel::Verify => emit_verify_probs(&g.cfg()).0,
    }
}

fn hand_ptx(kernel: Kernel, g: Geometry) -> String {
    match kernel {
        Kernel::Draft => hand::emit_draft_sample(&g.hand_cfg()).0,
        Kernel::Verify => hand::emit_verify_probs(&g.hand_cfg()).0,
    }
}

fn words(bytes: &[u8]) -> Vec<u32> {
    bytes.chunks_exact(4).map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// The geometries the agreement and correctness gates sweep: a single
/// token and a single feature; a vocab one past a tile and a d_model that
/// leaves threads idle in every strided loop; exactly one tile; a
/// d_model past the block, so the strided loops go round twice; a ragged
/// multi-tile vocab.
fn cases() -> Vec<Geometry> {
    vec![
        Geometry { d_model: 1, vocab: 1 },
        Geometry { d_model: 8, vocab: 129 },
        Geometry { d_model: 40, vocab: 128 },
        Geometry { d_model: 130, vocab: 200 },
        Geometry { d_model: 24, vocab: 300 },
    ]
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn the_kir_kernels_agree_with_the_hand_written_ones_bit_for_bit() {
    for g in cases() {
        let input = inputs(g, 0x5eed ^ g.d_model as u64);
        for kernel in [Kernel::Draft, Kernel::Verify] {
            let (hand, kir) = (hand_ptx(kernel, g), kir_ptx(kernel, g));
            for order in [Order::Ascending, Order::Descending] {
                let expect = run(&hand, kernel, g, &input, order);
                let got = run(&kir, kernel, g, &input, order);
                assert!(
                    expect == got,
                    "{kernel:?} {g:?} {order:?}: global memory differs\nhand: {:?} {:?}\nkir:  {:?} {:?}",
                    words(&expect[3]),
                    words(&expect[4]),
                    words(&got[3]),
                    words(&got[4])
                );
            }
        }
    }
}

#[test]
fn the_draft_picks_the_reference_token_and_probability() {
    for g in cases() {
        let input = inputs(g, 0x5eed ^ g.d_model as u64);
        let mem = run(&kir_ptx(Kernel::Draft, g), Kernel::Draft, g, &input, Order::Ascending);
        let token = words(&mem[3])[0];
        let p = f32::from_bits(words(&mem[4])[0]);
        let (want_token, want_p) =
            cpu_reference_draft_sample(&g.cfg(), &input.hidden, &input.norm_w, &input.lm_head_f32());
        assert_eq!(token, want_token, "{g:?}");
        assert!((p - want_p).abs() <= 1e-5 * want_p.abs(), "{g:?}: p = {p}, reference {want_p}");
    }
}

#[test]
fn the_verify_row_is_the_reference_softmax() {
    for g in cases() {
        let input = inputs(g, 0x5eed ^ g.d_model as u64);
        let mem = run(&kir_ptx(Kernel::Verify, g), Kernel::Verify, g, &input, Order::Ascending);
        let got: Vec<f32> = words(&mem[3]).into_iter().map(f32::from_bits).collect();
        let want = cpu_reference_verify_probs(&g.cfg(), &input.hidden, &input.norm_w, &input.lm_head_f32());
        assert_eq!(got.len(), want.len());
        for (i, (a, b)) in got.iter().zip(&want).enumerate() {
            assert!((a - b).abs() <= 1e-5 * (1e-6 + b.abs()), "{g:?}: row[{i}] = {a}, reference {b}");
        }
    }
}

#[test]
fn the_draft_probability_is_the_verify_row_at_the_drafted_token_bit_for_bit() {
    for g in cases() {
        let input = inputs(g, 0x5eed ^ g.d_model as u64);
        let draft = run(&kir_ptx(Kernel::Draft, g), Kernel::Draft, g, &input, Order::Descending);
        let verify = run(&kir_ptx(Kernel::Verify, g), Kernel::Verify, g, &input, Order::Ascending);
        let token = words(&draft[3])[0] as usize;
        assert_eq!(words(&draft[4])[0], words(&verify[3])[token], "{g:?}: the reject ratio would not be 1");
    }
}

#[test]
fn inputs_are_left_untouched() {
    let g = cases()[4];
    let input = inputs(g, 7);
    for kernel in [Kernel::Draft, Kernel::Verify] {
        let mem = run(&kir_ptx(kernel, g), kernel, g, &input, Order::Descending);
        let fresh = run(&kir_ptx(kernel, g), kernel, g, &input, Order::Ascending);
        assert!(mem[..3] == fresh[..3], "{kernel:?} wrote to its inputs");
        let f32s: Vec<u8> = input.hidden.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(mem[0], f32s, "{kernel:?} wrote to hidden");
    }
    // Verify never touches the draft's probability word.
    let mem = run(&kir_ptx(Kernel::Verify, g), Kernel::Verify, g, &input, Order::Ascending);
    assert_eq!(words(&mem[4]), [SENTINEL]);
}

#[test]
fn the_kir_kernels_keep_the_launch_abi_and_entry_names() {
    for g in cases() {
        for (kernel, name, params) in [
            (
                Kernel::Draft,
                "nsl_cfie_draft_sample",
                &["hidden_ptr", "norm_w_ptr", "lm_head_ptr", "out_token_ptr", "out_prob_ptr", "rng_seed"][..],
            ),
            (Kernel::Verify, "nsl_cfie_verify_probs", &["hidden_ptr", "norm_w_ptr", "lm_head_ptr", "out_probs_ptr"][..]),
        ] {
            let (hand, kir) = (parse(&hand_ptx(kernel, g)), parse(&kir_ptx(kernel, g)));
            assert_eq!(hand.params, kir.params, "the launcher marshals these positionally");
            let names: Vec<&str> = kir.params.iter().map(|(_, n)| n.as_str()).collect();
            assert_eq!(names, params);
            assert!(kir_ptx(kernel, g).contains(&format!(".visible .entry {name}(")));
            assert_eq!(hand.shared_bytes, kir.shared_bytes);
            assert_eq!((hand.shared.len(), kir.shared.len()), (1, 1));
        }
    }
}

#[test]
fn the_kir_kernels_keep_the_hand_kernels_header_lines() {
    for g in cases() {
        for kernel in [Kernel::Draft, Kernel::Verify] {
            let header = |ptx: &str| -> Vec<String> {
                ptx.lines().take_while(|l| l.starts_with("//")).map(str::to_string).collect()
            };
            assert_eq!(header(&hand_ptx(kernel, g)), header(&kir_ptx(kernel, g)), "{kernel:?} {g:?}");
        }
    }
}

// ---------------------------------------------------------------------------
// The gate bites
// ---------------------------------------------------------------------------

/// The case every mutation is judged on: three tiles with a ragged last
/// one; a d_model past the block, so every thread's sum-of-squares partial
/// is non-zero and every tree-reduction step moves real data (with fewer
/// features than threads, the first steps add only zeros and their
/// barriers cannot matter); a d_model that is none of the kernels' other
/// constants (the tile width, the tree-reduction offsets, 0 and 1); and a
/// planted tie.
fn mutation_case() -> Geometry {
    Geometry { d_model: 136, vocab: 300 }
}

/// The mutation case's inputs. The hidden row is small, so its mean
/// square (about 1e-5) is the size of the RMSNorm epsilon: with the
/// sweep's larger rows the epsilon is a few-ulp perturbation of the norm,
/// and the draft's probability — dominated by the planted tie — does not
/// move by even one ulp when it doubles.
fn mutation_inputs() -> Inputs {
    scaled_inputs(mutation_case(), 11, 0.005)
}

/// Whether `mutant` is told apart from the hand kernel: under either
/// schedule its memory differs, or the interpreter faults on it.
fn caught(kernel: Kernel, mutant: &str) -> bool {
    let g = mutation_case();
    let input = mutation_inputs();
    let hand = hand_ptx(kernel, g);
    [Order::Ascending, Order::Descending].into_iter().any(|order| {
        let expect = run(&hand, kernel, g, &input, order);
        let input = &input;
        match std::panic::catch_unwind(|| run(mutant, kernel, g, input, order)) {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
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

/// `ptx` with every line that starts with `prefix` and ends with `from`
/// rewritten to end with `to`; there must be exactly one.
fn rewrite_one(ptx: &str, prefix: &str, from: &str, to: &str) -> String {
    let hit = |l: &str| l.trim_start().starts_with(prefix) && l.ends_with(from);
    assert_eq!(ptx.lines().filter(|l| hit(l)).count(), 1, "`{prefix} … {from}` should appear once");
    ptx.lines()
        .map(|l| if hit(l) { format!("{}{to}", &l[..l.len() - from.len()]) } else { l.to_string() })
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn the_mutation_case_is_what_it_claims() {
    let g = mutation_case();
    let input = mutation_inputs();
    let lm = input.lm_head_f32();
    let (best, _) = cpu_reference_draft_sample(&g.cfg(), &input.hidden, &input.norm_w, &lm);
    let dm = g.d_model as usize;
    let twins = (0..g.vocab as usize).filter(|&t| lm[t * dm..(t + 1) * dm] == lm[best as usize * dm..(best as usize + 1) * dm]);
    assert_eq!(twins.count(), 2, "the planted tie is there");
    // The baseline for every mutation below: without it, `caught` could be
    // reporting a difference the mutation did not cause.
    for kernel in [Kernel::Draft, Kernel::Verify] {
        assert!(!caught(kernel, &kir_ptx(kernel, g)), "{kernel:?}");
    }
}

/// Every barrier but two is load-bearing. The two that are not, in both
/// kernels:
///
/// * the one after the cooperative hidden load (index 0): the
///   sum-of-squares loop that follows reads, in each thread, exactly the
///   elements that thread stored — the same strided indices — so no
///   thread reads another's write before it;
/// * the one after the last tree-reduction step (index 8): only thread 0
///   is active at `off = 1`, and the rstd step after it runs in thread 0
///   alone and reads the one element thread 0 just wrote. The barrier
///   after the rstd store is the one the other threads need.
///
/// A barrier the hand kernels carry and the gate cannot see the need for
/// is recorded here rather than dropped: removing either from the KIR
/// would be a change to the kernel, which this slice does not make.
#[test]
fn deleting_a_load_bearing_barrier_is_caught() {
    let g = mutation_case();
    for (kernel, expected) in [(Kernel::Draft, 13), (Kernel::Verify, 14)] {
        let ptx = kir_ptx(kernel, g);
        let barriers = ptx.lines().filter(|l| l.trim() == "bar.sync 0;").count();
        assert_eq!(
            barriers, expected,
            "{kernel:?}: hidden load, sum of squares, 7 reduction steps, rstd, norm, tile scores, tile end{}",
            if kernel == Kernel::Verify { ", (max, sum) publish" } else { "" }
        );
        for n in 0..barriers {
            let equivalent = n == 0 || n == 8;
            assert_eq!(
                caught(kernel, &without_nth(&ptx, "bar.sync 0;", n)),
                !equivalent,
                "{kernel:?}: barrier {n} deleted — expected {}",
                if equivalent { "an equivalent mutant" } else { "to be caught" }
            );
        }
    }
}

#[test]
fn nudging_a_baked_constant_is_caught() {
    let g = mutation_case();
    let f = |v: f32| format!("0f{:08X}", v.to_bits());
    let inv_dm = 1.0 / g.d_model as f32;
    for kernel in [Kernel::Draft, Kernel::Verify] {
        let ptx = kir_ptx(kernel, g);
        for (prefix, from, to) in [
            ("mov.u32", format!(", {};", g.vocab), format!(", {};", g.vocab - 1)),
            ("mov.u32", format!(", {};", g.d_model), format!(", {};", g.d_model - 1)),
            ("mov.u64", format!(", {};", g.d_model), format!(", {};", g.d_model + 1)),
            // 1%, not one ulp: `rsqrt` of the mean rounds a one-ulp nudge
            // of its scale away as often as not.
            ("mov.f32", format!(", {};", f(inv_dm)), format!(", {};", f(inv_dm * 1.01))),
            ("mov.f32", format!(", {};", f(1e-5)), format!(", {};", f(2e-5))),
        ] {
            let mutant = rewrite_one(&ptx, prefix, &from, &to);
            assert!(caught(kernel, &mutant), "{kernel:?}: `{prefix} … {from}` -> `{to}` went unnoticed");
        }
    }
}

#[test]
fn relaxing_first_max_wins_is_caught() {
    // `s > m` -> `s >= m`: the planted twin, later in the vocab, would take
    // the argmax from the first occurrence. Only the draft tracks one; the
    // verify row does not depend on which twin is "the" max.
    let g = mutation_case();
    let ptx = kir_ptx(Kernel::Draft, g);
    let gt: Vec<&str> = ptx.lines().filter(|l| l.trim_start().starts_with("setp.gt.f32 ")).collect();
    assert_eq!(gt.len(), 1, "one running-max comparison");
    let mutant = ptx.replacen(gt[0], &gt[0].replace("setp.gt.f32", "setp.ge.f32"), 1);
    assert!(caught(Kernel::Draft, &mutant), "a later tie taking the argmax went unnoticed");
}

/// The tail-tile *guards* (`tok >= vocab` skips the row dot, and in verify
/// pass 2 the store) are caught: without them the last tile reads LM-head
/// rows past the end, which the interpreter faults on.
///
/// The tail *clamp* (`cnt = min(vocab - tile, TILE)`) is an equivalent
/// mutant: the guard stores -inf for every lane past the vocab, and
/// merging a -inf score leaves the running state exactly as it was
/// (`-inf > m` is false, and `sum + ex2(-inf) == sum + 0`).
#[test]
fn dropping_a_tail_tile_guard_is_caught_and_the_clamp_is_equivalent() {
    let g = mutation_case();
    for kernel in [Kernel::Draft, Kernel::Verify] {
        let ptx = kir_ptx(kernel, g);
        // The guards compare a token against the vocab: `setp.ge.u32 p,
        // tok, vocab_reg`. Find the vocab register from its materialisation.
        let vocab_reg = ptx
            .lines()
            .find_map(|l| {
                let t = l.trim();
                t.strip_prefix("mov.u32 ").and_then(|r| r.strip_suffix(&format!(", {};", g.vocab)))
            })
            .expect("the vocab is materialised")
            .to_string();
        let guards: Vec<usize> = ptx
            .lines()
            .enumerate()
            .filter(|(_, l)| {
                let t = l.trim();
                t.starts_with("setp.ge.u32 ") && t.ends_with(&format!(", {vocab_reg};"))
            })
            .map(|(i, _)| i)
            .collect();
        // Tile loop exits compare the tile cursor against the vocab too;
        // the guards are the ones that are not a loop exit. Each tile loop
        // has one exit, each tile body one guard.
        let loops = if kernel == Kernel::Draft { 1 } else { 2 };
        assert_eq!(guards.len(), 2 * loops, "{kernel:?}: a loop exit and a guard per tile loop");
        let mut guards_caught = 0;
        for &at in &guards {
            // Force the comparison false: never skip.
            let lines: Vec<&str> = ptx.lines().collect();
            let t = lines[at].trim().trim_start_matches("setp.ge.u32 ");
            let pred = t.split(',').next().unwrap();
            let mutant: String = lines
                .iter()
                .enumerate()
                .map(|(i, l)| if i == at { format!("    setp.ne.u32 {pred}, {vocab_reg}, {vocab_reg};") } else { l.to_string() })
                .collect::<Vec<_>>()
                .join("\n");
            // A loop exit that never fires runs past the vocab too, and
            // faults as well; either way the mutant must be caught.
            assert!(caught(kernel, &mutant), "{kernel:?}: `{}` forced false went unnoticed", lines[at].trim());
            guards_caught += 1;
        }
        assert_eq!(guards_caught, 2 * loops);

        let clamps: Vec<&str> = ptx.lines().filter(|l| l.trim_start().starts_with("min.u32 ")).collect();
        assert_eq!(clamps.len(), 1, "{kernel:?}: one tail clamp");
        let (dst, rest) = clamps[0].trim().trim_start_matches("min.u32 ").split_once(',').unwrap();
        let tile = rest.split(',').nth(1).unwrap().trim().trim_end_matches(';');
        let mutant = ptx.replacen(clamps[0], &format!("    mov.u32 {dst}, {tile};"), 1);
        assert!(!caught(kernel, &mutant), "{kernel:?}: the clamp is an equivalent mutant");
    }
}

#[test]
fn the_interpreter_stores_u32_and_models_rsqrt() {
    let ptx = "\
.version 7.0
.target sm_70
.address_size 64
.visible .entry t(.param .u64 param_o) {
    .reg .b32 %r<1>;
    .reg .b64 %rd<1>;
    .reg .f32 %f<2>;
    ld.param.u64 %rd0, [param_o];
    mov.u32 %r0, 305419896;
    st.global.u32 [%rd0], %r0;
    mov.f32 %f0, 0f40800000;
    rsqrt.approx.f32 %f1, %f0;
    st.global.f32 [%rd0+4], %f1;
    ret;
}
";
    let prog = parse(ptx);
    let mut global = vec![Segment { base: 0x100, bytes: vec![0; 8] }];
    let args: HashMap<String, u64> = [("o".to_string(), 0x100)].into_iter().collect();
    let mut launch = Launch { prog: &prog, args: &args, global: &mut global, shared: vec![], ctaid: 0, ctaid_y: 0, nctaid_y: 1, ntid: 1, steps: 0 };
    run_cta(&mut launch, Order::Ascending);
    assert_eq!(words(&global[0].bytes), [0x1234_5678, 0.5f32.to_bits()]);
}

