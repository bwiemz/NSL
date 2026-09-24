//! The differential equivalence gate for the CFIE fused decode-sample
//! kernel, `nsl_cfie_fused_sample` (roadmap A2 step 9, fifth slice).
//!
//! `cfie_sample_ptx` emitted the kernel as hand-assembled PTX; it now
//! builds it as KIR, from the same hidden-load / RMSNorm / row-dot sections
//! as `cfie_spec_sampler_ptx` and the same xorshift64* draw as
//! `cfie_speculative_ptx`. This file runs the frozen hand emitter
//! (`tests/fixtures/cfie_sample_hand.rs`) and the KIR one side by side on
//! the cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`), the
//! spec's level 3:
//!
//! 1. **Agreement**: for every program and geometry below, under two thread
//!    schedules, the hand module and the KIR module leave *the same bytes*
//!    in all of global memory. The programs cover greedy, top-k, top-k +
//!    nucleus and pure multinomial, with and without the fused RMSNorm, the
//!    grammar hook compiled out, compiled in with no mask bound, and bound.
//! 2. **Correctness**: the sampled token is `cpu_reference`'s.
//! 3. **The gate bites**: see the mutation tests at the end, and the
//!    mutants they name as equivalent.
//!
//! As in the sibling gates, the interpreter models `ex2.approx` and
//! `rsqrt.approx` exactly; the device suite (`cfie_fused_sample_gpu_parity.rs`)
//! and the `ptxas` gate carry fidelity to the machine.

use std::collections::HashMap;

use half::f16;
use nsl_codegen::cfie_fused_sample::{emit_program, FusedSampleProgram, LmHeadShape, SamplingParams, SamplingStrategy};
use nsl_codegen::cfie_sample_ptx::{cpu_reference, emit, FusedSampleKernelConfig};

/// The two `crate::` paths the frozen emitter names, supplied from the
/// library under test (the fixture is included into this test crate).
mod cfie_fused_sample {
    pub use nsl_codegen::cfie_fused_sample::{FusedSampleOp, FusedSampleProgram};
}
mod gpu_specs {
    pub use nsl_codegen::gpu_specs::ptx_isa_for_sm;
}

#[allow(dead_code)]
#[path = "fixtures/cfie_sample_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const HIDDEN_BASE: u64 = 0x1000_0000;
const NORM_BASE: u64 = 0x2000_0000;
const W_BASE: u64 = 0x3000_0000;
const OUT_BASE: u64 = 0x4000_0000;
const MASK_BASE: u64 = 0x5000_0000;
const BLOCK: u32 = 128;
/// No vocab this file uses reaches it: a token the kernel fails to write
/// shows.
const SENTINEL: u32 = 0xDEAD_BEEF;

/// The grammar hook's configuration for one case.
#[derive(Debug, Clone)]
enum Grammar {
    /// `grammar_states = 0`: the hook is not compiled in.
    Off,
    /// Compiled in (`states` DFA states), but launched with a null mask
    /// pointer, as serve does until a grammar is live.
    Unbound { states: u32 },
    /// Compiled in and bound: `mask` holds `states` rows of
    /// `ceil(vocab / 8)` bytes, and the launch is in state `state`.
    Bound { states: u32, state: u32, mask: Vec<u8> },
}

#[derive(Debug, Clone)]
struct Case {
    d_model: u32,
    vocab: u32,
    params: SamplingParams,
    grammar: Grammar,
    seed: u64,
}

impl Case {
    fn program(&self) -> FusedSampleProgram {
        let shape = LmHeadShape { d_model: self.d_model, vocab_size: self.vocab, vocab_tile: 128, dtype_bytes: 2 };
        emit_program(self.params.clone(), shape)
    }

    fn states(&self) -> u32 {
        match self.grammar {
            Grammar::Off => 0,
            Grammar::Unbound { states } | Grammar::Bound { states, .. } => states,
        }
    }

    fn cfg(&self) -> FusedSampleKernelConfig {
        FusedSampleKernelConfig {
            d_model: self.d_model,
            vocab_size: self.vocab,
            vocab_tile: 128,
            top_k: self.params.top_k,
            grammar_states: self.states(),
        }
    }

    fn hand_cfg(&self) -> hand::FusedSampleKernelConfig {
        hand::FusedSampleKernelConfig {
            d_model: self.d_model,
            vocab_size: self.vocab,
            vocab_tile: 128,
            top_k: self.params.top_k,
            sm_version: 80,
            grammar_states: self.states(),
        }
    }
}

/// Deterministic values in [-1, 1), and bits.
struct Lcg(u64);

impl Lcg {
    fn step(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn next(&mut self) -> f32 {
        ((self.step() >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
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

/// How the hidden row is drawn.
#[derive(Debug, Clone, Copy)]
enum Hidden {
    /// Uniform in `[-scale, scale)`. A large scale spreads the logits, so
    /// the softmax concentrates on a few candidates; a small one keeps them
    /// close, so every candidate carries mass and the nucleus cutoff and
    /// the CDF walk both have work to do.
    Scale(f32),
    /// Small everywhere but element 127 (when the row reaches it). RMSNorm
    /// scales every logit by one factor, so a slightly wrong sum of squares
    /// only sharpens or flattens the softmax, which rarely moves the
    /// sampled token. Element 127's square reaches thread 0 only through
    /// the last live lane of every level of the tree reduction (127 -> 63
    /// -> ... -> 3 -> 1 -> 0), so a level that runs before the one above it
    /// has finished drops it, the sum collapses to the noise, and the scale
    /// blows up: the soft sample the hand kernel draws becomes an argmax.
    Spiky,
}

const HIDDEN_MODES: [Hidden; 3] = [Hidden::Scale(3.0), Hidden::Scale(0.05), Hidden::Spiky];

/// Random inputs, the hidden row drawn as `mode` says.
fn inputs(c: &Case, mode: Hidden) -> Inputs {
    let mut rng = Lcg(c.seed ^ 0xC0FFEE);
    let dm = c.d_model as usize;
    let hidden = match mode {
        Hidden::Scale(scale) => (0..dm).map(|_| rng.next() * scale).collect(),
        Hidden::Spiky => {
            let mut h: Vec<f32> = (0..dm).map(|_| rng.next() * 0.001).collect();
            h[127.min(dm - 1)] = 2.0 + rng.next();
            h
        }
    };
    let norm_w = (0..dm).map(|_| 0.5 + rng.next().abs()).collect();
    let lm_head = (0..c.vocab as usize * dm).map(|_| f16::from_f32(rng.next()).to_bits()).collect();
    Inputs { hidden, norm_w, lm_head }
}

/// Global memory after running `ptx`: `[hidden, norm_w, lm_head, out]`,
/// plus the mask when one is bound.
fn run(ptx: &str, c: &Case, input: &Inputs, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut global = vec![
        Segment { base: HIDDEN_BASE, bytes: f32s(&input.hidden) },
        Segment { base: NORM_BASE, bytes: f32s(&input.norm_w) },
        Segment { base: W_BASE, bytes: input.lm_head.iter().flat_map(|b| b.to_le_bytes()).collect() },
        Segment { base: OUT_BASE, bytes: SENTINEL.to_le_bytes().to_vec() },
    ];
    let (mask_ptr, state) = match &c.grammar {
        Grammar::Bound { state, mask, .. } => {
            global.push(Segment { base: MASK_BASE, bytes: mask.clone() });
            (MASK_BASE, *state)
        }
        Grammar::Off | Grammar::Unbound { .. } => (0, 0),
    };
    let args: HashMap<String, u64> = [
        ("hidden_ptr", HIDDEN_BASE),
        ("norm_w_ptr", NORM_BASE),
        ("lm_head_ptr", W_BASE),
        ("out_token_ptr", OUT_BASE),
        ("rng_seed", c.seed),
        ("grammar_mask_ptr", mask_ptr),
        ("grammar_state", state as u64),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let mut launch = Launch {
        prog: &prog,
        args: &args,
        global: &mut global,
        // Poisoned (0xFF.. is a NaN as f32): a read of shared memory no
        // thread wrote this launch shows in the output.
        shared: vec![0xFF; prog.shared_bytes],
        ctaid: 0,
        ntid: BLOCK,
        steps: 0,
    };
    run_cta(&mut launch, order);
    global.into_iter().map(|s| s.bytes).collect()
}

fn kir_ptx(c: &Case) -> String {
    emit(&c.program(), &c.cfg()).0
}

fn hand_ptx(c: &Case) -> String {
    hand::emit_fused_sample_ptx(&c.program(), &c.hand_cfg())
}

fn token(mem: &[Vec<u8>]) -> u32 {
    u32::from_le_bytes(mem[3][..4].try_into().unwrap())
}

fn reference(c: &Case, input: &Inputs) -> u32 {
    let mask = match &c.grammar {
        Grammar::Bound { state, mask, .. } => Some((mask.as_slice(), *state)),
        Grammar::Off | Grammar::Unbound { .. } => None,
    };
    cpu_reference(&c.program(), &input.hidden, &input.norm_w, &input.lm_head_f32(), mask, c.seed)
}

fn params(strategy: SamplingStrategy, top_k: u32, top_p: f32, rms_norm: bool) -> SamplingParams {
    SamplingParams {
        strategy,
        temperature: if strategy == SamplingStrategy::Greedy { 0.0 } else { 0.8 },
        top_k,
        top_p,
        grammar_masked: false,
        logits_bias: false,
        rms_norm,
    }
}

/// A mask that allows roughly one token in three in each state, never the
/// empty set.
fn scattered_mask(states: u32, vocab: u32, seed: u64) -> Vec<u8> {
    let row = vocab.div_ceil(8) as usize;
    let mut rng = Lcg(seed);
    let mut mask = vec![0u8; states as usize * row];
    for s in 0..states as usize {
        for t in 0..vocab as usize {
            if rng.step() % 3 == 0 || t == s % vocab as usize {
                mask[s * row + t / 8] |= 1 << (t % 8);
            }
        }
    }
    mask
}

/// The programs and geometries the agreement and correctness gates sweep.
fn cases() -> Vec<Case> {
    use SamplingStrategy::*;
    let mut out = Vec::new();
    let geometries = [
        // A single token; one past a tile with idle threads in every
        // strided loop; exactly one tile; a d_model past the block, so the
        // strided loops go round twice, over a ragged three-tile vocab.
        (1, 1),
        (24, 129),
        (8, 128),
        (136, 300),
    ];
    for (dm, vocab) in geometries {
        for (p, seeds) in [
            (params(Greedy, 5, 1.0, true), &[1u64][..]),
            (params(TopKTopP, 5, 0.9, true), &[0, 1, 0x5eed, 99][..]),
            (params(TopK, 3, 1.0, false), &[7, 8][..]),
            (params(Multinomial, 1, 1.0, true), &[3][..]),
            // k past the vocab: -inf candidates stay in the list.
            (params(TopKTopP, 64, 0.5, true), &[11, 12][..]),
        ] {
            for &seed in seeds {
                out.push(Case { d_model: dm, vocab, params: p.clone(), grammar: Grammar::Off, seed });
            }
        }
    }
    // The grammar hook: compiled in with no mask, and bound in several
    // states (every tile crosses a byte boundary; 300 is not a multiple
    // of 8).
    let mut g = params(TopKTopP, 5, 0.9, true);
    g.grammar_masked = true;
    out.push(Case { d_model: 24, vocab: 300, params: g.clone(), grammar: Grammar::Unbound { states: 3 }, seed: 5 });
    for state in 0..3 {
        out.push(Case {
            d_model: 24,
            vocab: 300,
            params: g.clone(),
            grammar: Grammar::Bound { states: 3, state, mask: scattered_mask(3, 300, 17) },
            seed: 21 + state as u64,
        });
    }
    let mut gg = params(Greedy, 4, 1.0, false);
    gg.grammar_masked = true;
    out.push(Case {
        d_model: 8,
        vocab: 129,
        params: gg,
        grammar: Grammar::Bound { states: 2, state: 1, mask: scattered_mask(2, 129, 3) },
        seed: 1,
    });
    out
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn the_kir_kernel_agrees_with_the_hand_written_one_bit_for_bit() {
    for c in cases() {
        let (hand, kir) = (hand_ptx(&c), kir_ptx(&c));
        for mode in HIDDEN_MODES {
            let input = inputs(&c, mode);
            for order in [Order::Ascending, Order::Descending] {
                let expect = run(&hand, &c, &input, order);
                let got = run(&kir, &c, &input, order);
                assert!(
                    expect == got,
                    "{c:?} {mode:?} {order:?}: hand token {} kir token {}",
                    token(&expect),
                    token(&got)
                );
            }
        }
    }
}

#[test]
fn the_shared_answer_is_the_reference_sample() {
    let mut distinct = std::collections::BTreeSet::new();
    for c in cases() {
        for mode in HIDDEN_MODES {
            let input = inputs(&c, mode);
            let got = token(&run(&kir_ptx(&c), &c, &input, Order::Ascending));
            assert_eq!(got, reference(&c, &input), "{c:?} {mode:?}");
            assert!(got < c.vocab, "{c:?}: token {got} past the vocab");
            if let Grammar::Bound { state, mask, .. } = &c.grammar {
                let row = c.vocab.div_ceil(8);
                let byte = mask[(state * row + got / 8) as usize];
                assert!(byte >> (got % 8) & 1 == 1, "{c:?}: token {got} is masked out");
            }
            distinct.insert((c.vocab, got));
        }
    }
    // The cases do sample: they do not all collapse onto one token per
    // vocab.
    assert!(distinct.len() > 12, "{distinct:?}");
}

#[test]
fn inputs_are_left_untouched() {
    let c = cases().into_iter().find(|c| matches!(c.grammar, Grammar::Bound { .. })).unwrap();
    let input = inputs(&c, Hidden::Scale(3.0));
    let mem = run(&kir_ptx(&c), &c, &input, Order::Descending);
    let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    assert_eq!(mem[0], f32s(&input.hidden));
    assert_eq!(mem[1], f32s(&input.norm_w));
    assert_eq!(mem[2], input.lm_head.iter().flat_map(|b| b.to_le_bytes()).collect::<Vec<u8>>());
    let Grammar::Bound { mask, .. } = &c.grammar else { unreachable!() };
    assert_eq!(&mem[4], mask);
}

#[test]
fn the_kir_kernel_keeps_the_launch_abi_entry_name_and_footprint() {
    for c in cases() {
        let (hand, kir) = (parse(&hand_ptx(&c)), parse(&kir_ptx(&c)));
        assert_eq!(hand.params, kir.params, "the launcher marshals these positionally");
        assert_eq!(hand.shared_bytes, kir.shared_bytes, "{c:?}");
        assert_eq!((hand.shared.len(), kir.shared.len()), (1, 1));
        assert!(kir_ptx(&c).contains(".visible .entry nsl_cfie_fused_sample("));
        let (_, meta) = emit(&c.program(), &c.cfg());
        let (_, hand_meta) = hand::emit(&c.program(), &c.hand_cfg());
        assert_eq!((meta.smem_bytes, meta.block_dim), (hand_meta.smem_bytes, hand_meta.block_dim));
    }
}

#[test]
fn the_kir_kernel_keeps_the_hand_kernels_header_lines() {
    for c in cases() {
        let header = |ptx: &str| -> Vec<String> {
            ptx.lines().take_while(|l| l.starts_with("//")).map(str::to_string).collect()
        };
        assert_eq!(header(&hand_ptx(&c)), header(&kir_ptx(&c)), "{c:?}");
    }
}

// ---------------------------------------------------------------------------
// The gate bites
// ---------------------------------------------------------------------------

/// The cases every mutation is judged on, all with RMSNorm over the ragged
/// three-tile vocab with a d_model past the block, each under several
/// seeds:
///
/// * top-k + nucleus, the default program's shape;
/// * greedy with a bound grammar mask;
/// * greedy at k = 1 and k = 5 with a planted tie (the argmax row copied to
///   the last token), so first-wins is what decides;
/// * pure multinomial, whose candidate list is walked unsorted, so its
///   order is observable;
/// * top-k at temperature 2, whose distribution is soft enough for the
///   RMSNorm scale to be observable (see [`Hidden::Spiky`]).
fn mutation_cases() -> Vec<Case> {
    use SamplingStrategy::*;
    let mut out = Vec::new();
    // Seeds spread over all 64 bits, as real ones are: a small seed leaves
    // the xorshift state small, and then a low-bit nudge of a PRNG constant
    // cannot reach the draw's top 24 bits. Seed 0 stays 0 (it takes the
    // golden-gamma substitution).
    let spread = |seed: u64| seed.wrapping_mul(0xD1B5_4A32_D192_ED03);
    let mut push = |params: SamplingParams, grammar: Grammar, seeds: &[u64]| {
        for &seed in seeds {
            let seed = spread(seed);
            out.push(Case { d_model: 136, vocab: 300, params: params.clone(), grammar: grammar.clone(), seed });
        }
    };
    push(params(TopKTopP, 5, 0.7, true), Grammar::Off, &[0, 1, 2, 3, 4, 5]);
    let mut g = params(Greedy, 5, 1.0, true);
    g.grammar_masked = true;
    for state in 0..2 {
        push(g.clone(), Grammar::Bound { states: 2, state, mask: scattered_mask(2, 300, 9) }, &[1]);
    }
    push(params(Greedy, 1, 1.0, true), Grammar::Off, &[12, 13]);
    push(params(Greedy, 5, 1.0, true), Grammar::Off, &[14, 15]);
    push(params(Multinomial, 8, 1.0, true), Grammar::Off, &[6, 7]);
    // At temperature 2 the top candidates' logits sit about one unit
    // apart: soft enough that every candidate carries mass, sharp enough
    // that the softmax still moves when the scale does.
    let mut hot = params(TopK, 8, 1.0, true);
    hot.temperature = 2.0;
    // Seed 0 takes the golden-gamma substitution.
    push(hot, Grammar::Off, &[0, 8, 9, 10, 11, 16, 17, 18]);
    out
}

/// [`inputs`], plus the planted tie for an ungrammared greedy case: the
/// row of the token it picks is copied to the last token.
fn mutation_inputs(c: &Case, mode: Hidden) -> Inputs {
    let mut input = inputs(c, mode);
    if c.params.strategy == SamplingStrategy::Greedy && matches!(c.grammar, Grammar::Off) {
        let best = reference(c, &input) as usize;
        let last = c.vocab as usize - 1;
        if best != last {
            let dm = c.d_model as usize;
            let row = input.lm_head[best * dm..(best + 1) * dm].to_vec();
            input.lm_head[last * dm..(last + 1) * dm].copy_from_slice(&row);
        }
    }
    input
}

/// Whether `mutate`, applied to each mutation case's KIR text, is told
/// apart from the hand kernel: for some case, hidden row mode and schedule
/// the memory differs, or the interpreter faults.
fn caught(mutate: impl Fn(&Case, &str) -> String) -> bool {
    mutation_cases().iter().any(|c| {
        let hand = hand_ptx(c);
        let mutant = mutate(c, &kir_ptx(c));
        HIDDEN_MODES.into_iter().any(|mode| {
            let input = mutation_inputs(c, mode);
            [Order::Ascending, Order::Descending].into_iter().any(|order| {
                let expect = run(&hand, c, &input, order);
                let got = std::panic::catch_unwind(|| run(&mutant, c, &input, order));
                match got {
                    Ok(mem) => mem != expect,
                    Err(_) => true,
                }
            })
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

/// `ptx` with `f` applied to every line `pick` selects (given the line and
/// the one after it), after checking `pick` selects at least one.
fn rewrite(ptx: &str, pick: impl Fn(&str, &str) -> bool, f: impl Fn(&str) -> String) -> String {
    let lines: Vec<&str> = ptx.lines().collect();
    let mut hits = 0;
    let out: Vec<String> = lines
        .iter()
        .enumerate()
        .map(|(i, l)| {
            let next = lines.get(i + 1).copied().unwrap_or("");
            if pick(l, next) {
                hits += 1;
                f(l)
            } else {
                l.to_string()
            }
        })
        .collect();
    assert!(hits > 0, "the rewrite selected no line");
    out.join("\n")
}

fn is(l: &str, mnemonic: &str) -> bool {
    l.trim_start().starts_with(mnemonic)
}

#[test]
fn the_unmutated_kernel_is_not_caught() {
    // The baseline for every mutation below.
    assert!(!caught(|_, p| p.to_string()));
}

/// The kernel's 14 barriers (with RMSNorm), in the order the module prints
/// them: the hidden load; the sum-of-squares store; the seven tree levels
/// (64 .. 1); the rstd publish; the normalised row; the candidate init;
/// and per tile, the score store and the merge's end.
const BARRIERS: usize = 14;
/// Four order nothing: see `the_redundant_barriers_are_named_equivalent_mutants`.
const REDUNDANT: [usize; 4] = [0, 8, 10, 11];

#[test]
fn deleting_any_needed_barrier_is_caught() {
    let ptx = kir_ptx(&mutation_cases()[0]);
    assert_eq!(ptx.lines().filter(|l| l.trim() == "bar.sync 0;").count(), BARRIERS);
    for n in (0..BARRIERS).filter(|n| !REDUNDANT.contains(n)) {
        assert!(caught(|_, p| without_nth(p, "bar.sync 0;", n)), "barrier {n} deleted went unnoticed");
    }
}

/// Four barriers order nothing, in the hand kernel as in this one:
///
/// * 0, after the hidden load: the sum-of-squares loop that follows reads
///   exactly the elements its own thread loaded.
/// * 8, after the last tree level: that level's only writer, thread 0,
///   is the only reader of `scores[0]` (the rstd).
/// * 10, after the normalised row is written back: the row is next read
///   by the dot products, after the candidate-init barrier (11).
/// * 11, after the candidate init: the list is next read by thread 0's
///   merge, after the first tile's score barrier; the row writes it would
///   also order are ordered by 10. (10 and 11 are each redundant given the
///   other; deleting both is not equivalent.)
///
/// A test claiming to catch their deletion would be vacuous; this pins
/// that it is not caught, and the kernel keeps them.
#[test]
fn the_redundant_barriers_are_named_equivalent_mutants() {
    for n in REDUNDANT {
        assert!(!caught(|_, p| without_nth(p, "bar.sync 0;", n)), "barrier {n} is not redundant after all");
    }
}

#[test]
fn nudging_a_baked_constant_is_caught() {
    let f = |v: f32| format!("0f{:08X}", v.to_bits());
    // (from, to) immediates, each rewritten wherever a `mov` materialises
    // it. Temperature and top_p are per program, so they are nudged per
    // case.
    let fixed: Vec<(String, String)> = vec![
        // RMSNorm: 1/d_model (x4, halving the scale), and an eps that
        // swamps the small rows. RMSNorm only scales the logits, so the
        // nudges are sized for the hot cases to see.
        (format!(", {};", f(1.0 / 136.0)), format!(", {};", f(4.0 / 136.0))),
        (format!(", {};", f(1e-5)), format!(", {};", f(1e-1))),
        // The PRNG: golden gamma (seed 0; nudged in bit 40, since a
        // low-bit nudge moves this seed's draw from 0.053 to 0.063 only,
        // and both pick the same candidate), multiplier, 2^-24, shifts.
        (", 11400714819323198485;".into(), ", 11400715918834826261;".into()),
        (", 2685821657736338717;".into(), ", 2685821657736338719;".into()),
        (format!(", {};", f(1.0 / 16_777_216.0)), format!(", {};", f(1.0 / 8_388_608.0))),
        (", 12;".into(), ", 13;".into()),
        (", 25;".into(), ", 24;".into()),
        (", 27;".into(), ", 28;".into()),
        (", 40;".into(), ", 41;".into()),
    ];
    for (from, to) in &fixed {
        let mutant = |_: &Case, p: &str| {
            if p.lines().any(|l| is(l, "mov.") && l.ends_with(from.as_str())) {
                rewrite(p, |l, _| is(l, "mov.") && l.ends_with(from.as_str()), |l| l.replace(from.as_str(), to))
            } else {
                p.to_string()
            }
        };
        assert!(caught(mutant), "`{from}` -> `{to}` went unnoticed");
    }
    // 1/temperature 1.5x, and top_p 0.1 lower. Like the RMSNorm nudges,
    // the temperature only rescales the logits, and one token is a coarse
    // view of a rescale: over these cases a 1.1x one moves no sample, 1.5x
    // moves 4 in 48, 2x moves 7 in 48 (measured when this gate was
    // written).
    let temp = |c: &Case, p: &str| {
        if c.params.strategy == SamplingStrategy::Greedy {
            return p.to_string();
        }
        let from = format!(", {};", f(1.0 / c.params.temperature));
        let to = format!(", {};", f(1.5 / c.params.temperature));
        rewrite(p, |l, _| is(l, "mov.f32") && l.ends_with(&from), |l| l.replace(&from, &to))
    };
    assert!(caught(temp), "1/temperature nudged went unnoticed");
    let top_p = |c: &Case, p: &str| {
        if c.params.strategy != SamplingStrategy::TopKTopP {
            return p.to_string();
        }
        let from = format!(", {};", f(c.params.top_p));
        let to = format!(", {};", f(c.params.top_p - 0.1));
        rewrite(p, |l, _| is(l, "mov.f32") && l.ends_with(&from), |l| l.replace(&from, &to))
    };
    assert!(caught(top_p), "top_p nudged went unnoticed");
    // One candidate fewer.
    let k = |c: &Case, p: &str| {
        let from = format!(", {};", c.params.top_k);
        let to = format!(", {};", c.params.top_k.max(2) - 1);
        // The entry block: from its label to the next one.
        let entry: Vec<&str> =
            p.lines().skip_while(|l| l.trim() != "BB0:").skip(1).take_while(|l| !l.ends_with(':')).collect();
        let line = entry.iter().rev().find(|l| is(l, "mov.u32") && l.ends_with(&from)).expect("top_k in the entry");
        p.replacen(line, &line.replace(&from, &to), 1)
    };
    assert!(caught(k), "top_k nudged went unnoticed");
}

#[test]
fn relaxing_a_first_wins_comparison_is_caught() {
    // The min-scan (`<` then the selects): the last minimum wins instead,
    // so the replace-min merge reorders the list, which the unsorted walk
    // of the multinomial and top-k programs observes.
    let min_scan = |_: &Case, p: &str| {
        rewrite(p, |l, next| is(l, "setp.lt.f32") && is(next, "selp.f32"), |l| l.replace("setp.lt.f32", "setp.le.f32"))
    };
    assert!(caught(min_scan), "min-scan `<` -> `<=` went unnoticed");
    // The merge (`>`, the first f32 greater-than): a later candidate equal
    // to the list's minimum displaces it; at k = 1 the minimum is the
    // running max, so the planted twin wins.
    let merge = |_: &Case, p: &str| {
        let line = p.lines().find(|l| is(l, "setp.gt.f32")).unwrap();
        p.replacen(line, &line.replace("setp.gt.f32", "setp.ge.f32"), 1)
    };
    assert!(caught(merge), "merge `>` -> `>=` went unnoticed");
    // The argmax (`>` then the selects): the twin, later in the list, wins.
    let argmax = |c: &Case, p: &str| {
        if c.params.strategy != SamplingStrategy::Greedy {
            return p.to_string();
        }
        rewrite(p, |l, next| is(l, "setp.gt.f32") && is(next, "selp.f32"), |l| l.replace("setp.gt.f32", "setp.ge.f32"))
    };
    assert!(caught(argmax), "argmax `>` -> `>=` went unnoticed");
}

/// The walk's positive-entry test (`p > 0`, the last f32 greater-than) is
/// an equivalent mutant when relaxed to `>=`. The running sum only grows at
/// a positive entry, and the target `r * kept` never exceeds the sum of the
/// same entries in the same order, so the walk always stops at a positive
/// entry; letting a zero entry update the selection changes the result only
/// when the target is 0 and no positive entry precedes it, that is, only
/// for a draw of exactly 0 (one in 2^24). Same argument as the rejection
/// kernel's CDF walk.
#[test]
fn the_walks_positive_entry_test_is_a_named_equivalent_mutant() {
    let walk = |c: &Case, p: &str| {
        if c.params.strategy == SamplingStrategy::Greedy {
            return p.to_string();
        }
        let line = p.lines().filter(|l| is(l, "setp.gt.f32")).last().unwrap();
        p.replacen(line, &line.replace("setp.gt.f32", "setp.ge.f32"), 1)
    };
    assert!(!caught(walk));
}

#[test]
fn dropping_the_vocab_guard_is_caught_and_the_merge_clamp_is_equivalent() {
    // `tok < vocab` forced true: lanes past the vocab read past the LM
    // head, which faults.
    let guard = |c: &Case, p: &str| {
        let decl = format!(", {};", c.vocab);
        let vocab_reg = p
            .lines()
            .find(|l| is(l, "mov.u32") && l.ends_with(&decl))
            .and_then(|l| l.trim().strip_prefix("mov.u32 "))
            .and_then(|l| l.split(',').next())
            .expect("the vocab register")
            .to_string();
        let tail = format!(", {vocab_reg};");
        let line = p.lines().find(|l| is(l, "setp.lt.u32") && l.ends_with(&tail)).expect("the guard");
        let (lhs, _) = line.trim().trim_start_matches("setp.lt.u32 ").split_once(", ").unwrap();
        p.replacen(line, &format!("    setp.eq.u32 {lhs}, {vocab_reg}, {vocab_reg};"), 1)
    };
    assert!(caught(guard), "the vocab guard removed went unnoticed");

    // `cnt = min(vocab - tile, TILE)` -> `cnt = TILE`: the merge then also
    // walks the tail lanes, which hold -inf, and -inf never beats the
    // list's minimum under the strict `>` (the list always holds k
    // candidates by then, the -inf initial ones included, so the minimum is
    // never above -inf's reach from below). An equivalent mutant: the guard
    // above is what keeps the tail out.
    let clamp = |_: &Case, p: &str| {
        let line = p.lines().find(|l| is(l, "min.u32")).unwrap();
        let (dst, rest) = line.trim().trim_start_matches("min.u32 ").split_once(',').unwrap();
        let tile = rest.split(',').nth(1).unwrap().trim().trim_end_matches(';');
        p.replacen(line, &format!("    mov.u32 {dst}, {tile};"), 1)
    };
    assert!(!caught(clamp), "the merge clamp is not equivalent after all");
}

#[test]
fn skipping_the_grammar_mask_is_caught() {
    // The null-mask test forced true: the hook never runs and a masked
    // token can win.
    let null = |c: &Case, p: &str| {
        if !matches!(c.grammar, Grammar::Bound { .. }) {
            return p.to_string();
        }
        let line = p.lines().find(|l| is(l, "setp.eq.u64")).unwrap();
        let (lhs, rest) = line.trim().trim_start_matches("setp.eq.u64 ").split_once(", ").unwrap();
        let reg = rest.split(',').next().unwrap();
        p.replacen(line, &format!("    setp.eq.u64 {lhs}, {reg}, {reg};"), 1)
    };
    assert!(caught(null), "the grammar mask skipped went unnoticed");
}

#[test]
fn the_interpreter_knows_the_new_forms() {
    // The hand kernel's byte load and the KIR one's sign-extending pair,
    // and the 64-bit select of the zero-seed substitution.
    let c = mutation_cases().into_iter().find(|c| matches!(c.grammar, Grammar::Bound { .. })).unwrap();
    assert!(hand_ptx(&c).contains("ld.global.u8 "));
    let kir = kir_ptx(&c);
    assert!(kir.contains("ld.global.s8 ") && kir.contains("cvt.u32.s8 "));
    let sampling = kir_ptx(&mutation_cases()[0]);
    assert!(sampling.contains("selp.b64 "));
    parse(&hand_ptx(&c));
    parse(&kir);
    parse(&sampling);
}



