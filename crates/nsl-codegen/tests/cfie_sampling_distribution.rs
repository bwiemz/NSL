//! The fused sampler draws from the distribution its settings describe.
//!
//! The kernel is run as PTX, on the cooperative-CTA interpreter
//! (`tests/support/cta_ptx_interp.rs`), once per sequence position of one
//! request, each launch seeded as the runtime's decode loop seeds it
//! (`nsl_runtime::cfie::engine::step_seed` of the request's key and the
//! position). The tokens it writes are counted and compared with the exact
//! probabilities, which this file works out in f64 from the logits and the
//! settings — temperature, then top-k, then the nucleus — without calling
//! the kernel's CPU reference.
//!
//! Each count must be within five standard deviations of its binomial mean:
//! a correct sampler fails that about once in a million runs, and the seeds
//! are fixed besides. The defect this guards against is not subtle at that
//! resolution: the decode loop used to pass one seed at every position, so
//! every token of a request was drawn at one quantile (0.0528 for the seed
//! `generate()` passed), and "temperature 0.7, top-p 0.9" emitted the most
//! likely token whenever it held more than 5.3% of the kept mass.

use std::collections::HashMap;

use half::f16;
use nsl_codegen::cfie_fused_sample::{emit_program, LmHeadShape, SamplingParams, SamplingStrategy};
use nsl_codegen::cfie_sample_ptx::{emit, FusedSampleKernelConfig};
use nsl_runtime::cfie::engine::{request_key, step_seed};

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const HIDDEN_BASE: u64 = 0x1000_0000;
const NORM_BASE: u64 = 0x2000_0000;
const W_BASE: u64 = 0x3000_0000;
const OUT_BASE: u64 = 0x4000_0000;
const MASK_BASE: u64 = 0x5000_0000;
const VOCAB: usize = 128;

/// The logits of the eight tokens that matter, all exact in f16; every
/// other token sits at -6, far below the top-8.
const LOGITS: [(usize, f32); 8] =
    [(3, 2.0), (17, 1.5), (40, 1.25), (77, 1.0), (100, 0.5), (5, 0.25), (64, 0.0), (127, -0.5)];

fn logits() -> Vec<f32> {
    let mut l = vec![-6.0f32; VOCAB];
    for (tok, v) in LOGITS {
        l[tok] = v;
    }
    l
}

/// The distribution the settings describe, in f64: divide by the
/// temperature, keep the `top_k` largest, softmax over those, and with a
/// `top_p` keep the smallest prefix of the descending order whose mass
/// exceeds it, renormalised.
fn exact_distribution(logits: &[f32], temperature: f64, top_k: usize, top_p: Option<f64>) -> Vec<f64> {
    let mut order: Vec<usize> = (0..logits.len()).collect();
    order.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let kept = &order[..top_k];
    let weights: Vec<f64> = kept.iter().map(|&t| (logits[t] as f64 / temperature).exp()).collect();
    let total: f64 = weights.iter().sum();
    let mut probs: Vec<f64> = weights.iter().map(|w| w / total).collect();
    if let Some(top_p) = top_p {
        let mut cum = 0.0;
        let mut cut = probs.len();
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if cum > top_p {
                cut = i + 1;
                break;
            }
        }
        for p in &mut probs[cut..] {
            *p = 0.0;
        }
        let kept_mass: f64 = probs.iter().sum();
        for p in &mut probs {
            *p /= kept_mass;
        }
    }
    let mut out = vec![0.0; logits.len()];
    for (&tok, p) in kept.iter().zip(probs) {
        out[tok] = p;
    }
    out
}

struct Sampler {
    prog: Program,
    weights: Vec<u8>,
    /// A grammar mask to bind: one row of `VOCAB / 8` bytes per state.
    mask: Option<Vec<u8>>,
}

impl Sampler {
    /// The kernel for `params` over a one-feature model whose LM-head row
    /// `t` is `logits[t]`, so with hidden = [1.0] the logits are exactly
    /// those.
    fn new(params: SamplingParams) -> Self {
        Self::with_mask(params, None)
    }

    /// [`Self::new`] with the grammar hook compiled in and `mask` bound.
    fn with_mask(params: SamplingParams, mask: Option<Vec<u8>>) -> Self {
        let shape = LmHeadShape { d_model: 1, vocab_size: VOCAB as u32, vocab_tile: 128, dtype_bytes: 2 };
        let cfg = FusedSampleKernelConfig {
            d_model: 1,
            vocab_size: VOCAB as u32,
            vocab_tile: 128,
            top_k: params.top_k,
            grammar_states: mask.as_ref().map_or(0, |m| (m.len() / (VOCAB / 8)) as u32),
            norm_eps: 1e-5,
        };
        let params = SamplingParams { grammar_masked: mask.is_some(), ..params };
        let (ptx, _) = emit(&emit_program(params, shape), &cfg);
        let weights = logits().iter().flat_map(|&l| f16::from_f32(l).to_bits().to_le_bytes()).collect();
        Self { prog: parse(&ptx), weights, mask }
    }

    /// One launch with the kernel seed `seed`: the token it writes.
    fn draw(&self, seed: u64) -> u32 {
        self.launch(1.0, seed, 0)
    }

    /// One launch with the hidden row's one element, the kernel seed and
    /// the grammar state given.
    fn launch(&self, hidden: f32, seed: u64, state: u64) -> u32 {
        let mut global = vec![
            Segment { base: HIDDEN_BASE, bytes: hidden.to_le_bytes().to_vec() },
            Segment { base: NORM_BASE, bytes: 1.0f32.to_le_bytes().to_vec() },
            Segment { base: W_BASE, bytes: self.weights.clone() },
            Segment { base: OUT_BASE, bytes: 0x7FC0_0001u32.to_le_bytes().to_vec() },
        ];
        if let Some(mask) = &self.mask {
            global.push(Segment { base: MASK_BASE, bytes: mask.clone() });
        }
        let args: HashMap<String, u64> = [
            ("hidden_ptr", HIDDEN_BASE),
            ("norm_w_ptr", NORM_BASE),
            ("lm_head_ptr", W_BASE),
            ("out_token_ptr", OUT_BASE),
            ("rng_seed", seed),
            ("grammar_mask_ptr", if self.mask.is_some() { MASK_BASE } else { 0 }),
            ("grammar_state", state),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), v))
        .collect();
        let mut launch = Launch {
            prog: &self.prog,
            args: &args,
            global: &mut global,
            shared: vec![0xFF; self.prog.shared_bytes],
            ctaid: 0,
            ctaid_y: 0,
            nctaid_x: 0,
            nctaid_y: 1,
            ntid: 128,
            steps: 0,
        };
        run_cta(&mut launch, Order::Ascending);
        u32::from_le_bytes(global[3].bytes[..4].try_into().unwrap())
    }

    /// The tokens of positions `0..n` of the request with key `key`.
    fn request(&self, key: u64, n: u64) -> Vec<u32> {
        (0..n).map(|pos| self.draw(step_seed(key, pos))).collect()
    }
}

fn params(strategy: SamplingStrategy, temperature: f32, top_k: u32, top_p: f32) -> SamplingParams {
    SamplingParams {
        strategy,
        temperature,
        top_k,
        top_p,
        grammar_masked: false,
        logits_bias: false,
        rms_norm: false,
    }
}

fn assert_frequencies(tokens: &[u32], exact: &[f64], what: &str) {
    let n = tokens.len() as f64;
    let mut counts = vec![0usize; exact.len()];
    for &t in tokens {
        assert!((t as usize) < exact.len(), "{what}: token {t} is outside the vocabulary");
        counts[t as usize] += 1;
    }
    for (tok, (&c, &p)) in counts.iter().zip(exact).enumerate() {
        if p == 0.0 {
            assert_eq!(c, 0, "{what}: token {tok} has probability 0 and was drawn {c} times");
            continue;
        }
        let mean = n * p;
        let sd = (n * p * (1.0 - p)).sqrt();
        assert!(
            (c as f64 - mean).abs() <= 5.0 * sd,
            "{what}: token {tok} was drawn {c} times of {n}; {mean:.1} expected (p = {p:.4}, sd {sd:.1})"
        );
    }
}

const DRAWS: u64 = 3000;

/// The shipped examples' settings: temperature 0.7, top-k 50 would keep
/// all eight live tokens; here top-k 8 does, and top-p 0.9 then drops the
/// tail. The exact nucleus is tokens 3, 17, 40, 77 and 100.
#[test]
fn temperature_top_k_and_top_p_give_the_nucleus_distribution() {
    let exact = exact_distribution(&logits(), 0.7, 8, Some(0.9));
    // The reference itself, pinned by hand: exp(l / 0.7) for the top
    // eight is 17.412, 8.524, 5.964, 4.173, 2.043, 1.429, 1.000, 0.490
    // (sum 41.034); the cumulative shares 0.424, 0.632, 0.777, 0.879,
    // 0.929 cross 0.9 at the fifth, so five tokens are kept, with mass
    // 38.115 between them (numpy: 0.45682 for the first, 0.05359 for the
    // fifth).
    let kept: Vec<usize> = (0..VOCAB).filter(|&t| exact[t] > 0.0).collect();
    assert_eq!(kept, [3, 17, 40, 77, 100]);
    assert!((exact[3] - 0.456_823_19).abs() < 1e-7, "{}", exact[3]);
    assert!((exact[100] - 0.053_594_12).abs() < 1e-7, "{}", exact[100]);

    let sampler = Sampler::new(params(SamplingStrategy::TopKTopP, 0.7, 8, 0.9));
    let tokens = sampler.request(request_key(0, 0), DRAWS);
    assert_frequencies(&tokens, &exact, "T=0.7, top_k=8, top_p=0.9");
}

#[test]
fn temperature_and_top_k_without_top_p_keep_all_k() {
    let exact = exact_distribution(&logits(), 1.0, 4, None);
    let kept: Vec<usize> = (0..VOCAB).filter(|&t| exact[t] > 0.0).collect();
    assert_eq!(kept, [3, 17, 40, 77]);
    let sampler = Sampler::new(params(SamplingStrategy::TopK, 1.0, 4, 1.0));
    let tokens = sampler.request(request_key(1, 0), DRAWS);
    assert_frequencies(&tokens, &exact, "T=1, top_k=4");

    // A high temperature flattens it; the kernel must follow.
    let exact = exact_distribution(&logits(), 2.5, 8, None);
    let sampler = Sampler::new(params(SamplingStrategy::TopK, 2.5, 8, 1.0));
    let tokens = sampler.request(request_key(2, 0), DRAWS);
    assert_frequencies(&tokens, &exact, "T=2.5, top_k=8");
}

/// What the old seeding did, shown on the same kernel: one seed at every
/// position gives one token at every position.
#[test]
fn one_seed_at_every_position_is_one_token_and_the_decode_loop_no_longer_does_that() {
    let sampler = Sampler::new(params(SamplingStrategy::TopKTopP, 0.7, 8, 0.9));
    let replayed: Vec<u32> = (0..64).map(|_| sampler.draw(0)).collect();
    assert!(replayed.iter().all(|&t| t == 3), "seed 0 draws at quantile 0.0528: the top token, every time");

    let tokens = sampler.request(request_key(0, 0), 64);
    let distinct: std::collections::BTreeSet<u32> = tokens.iter().copied().collect();
    assert!(distinct.len() >= 4, "64 positions of one request drew only {distinct:?}");

    // Two requests with one seed are different sequences; one request
    // twice is the same sequence.
    let again = sampler.request(request_key(0, 0), 64);
    let other = sampler.request(request_key(0, 1), 64);
    assert_eq!(tokens, again, "a (seed, request) pair is reproducible");
    assert_ne!(tokens, other, "the next request does not replay this one");
}

#[test]
fn greedy_ignores_the_draw() {
    let sampler = Sampler::new(params(SamplingStrategy::Greedy, 0.0, 8, 1.0));
    let tokens = sampler.request(request_key(9, 0), 32);
    assert!(tokens.iter().all(|&t| t == 3), "{tokens:?}");
}

/// With nothing to choose from the kernel writes `NO_CANDIDATE`, an id no
/// vocabulary contains, which the decode loop refuses. Two ways to get
/// there: a grammar state whose mask row allows no token, and a model
/// whose output is NaN. The kernel used to write token 0, a real token:
/// every candidate was -inf, the softmax NaN, and the walk fell back to
/// the first slot.
#[test]
fn nothing_to_choose_from_is_not_token_zero() {
    use nsl_codegen::cfie_sample_ptx::NO_CANDIDATE;
    assert!(NO_CANDIDATE as usize >= VOCAB);
    // State 0 allows tokens 17 and 40; state 1 allows nothing.
    let row = VOCAB / 8;
    let mut mask = vec![0u8; 2 * row];
    mask[17 / 8] |= 1 << (17 % 8);
    mask[40 / 8] |= 1 << (40 % 8);
    for p in [
        params(SamplingStrategy::Greedy, 0.0, 8, 1.0),
        params(SamplingStrategy::TopK, 1.0, 8, 1.0),
        params(SamplingStrategy::TopKTopP, 0.7, 8, 0.9),
    ] {
        let masked = Sampler::with_mask(p, Some(mask.clone()));
        let plain = Sampler::new(p);
        for seed in [0u64, 1, 0xDEAD_BEEF, step_seed(request_key(4, 2), 9)] {
            let allowed = masked.launch(1.0, seed, 0);
            assert!(allowed == 17 || allowed == 40, "{:?}: state 0 drew {allowed}", p.strategy);
            assert_eq!(masked.launch(1.0, seed, 1), NO_CANDIDATE, "{:?}: an empty mask row", p.strategy);
            assert_eq!(plain.launch(f32::NAN, seed, 0), NO_CANDIDATE, "{:?}: a NaN model output", p.strategy);
        }
    }
}
