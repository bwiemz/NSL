//! The differential equivalence gate for the CFIE speculative-decoding
//! kernels (roadmap A2 step 9, fourth slice). This PR moves the rejection
//! epilogue, `nsl_cfie_spec_reject`; the tree-mask verify attention is the
//! next one and stays hand-written until then.
//!
//! This file runs the frozen hand emitter
//! (`tests/fixtures/cfie_speculative_hand.rs`) and the KIR one side by side
//! on the cooperative-CTA interpreter (`tests/support/cta_ptx_interp.rs`),
//! the spec's level 3:
//!
//! 1. **Agreement** — for every case below the hand module and the KIR
//!    module leave *the same bytes* in all of global memory.
//! 2. **Correctness** — those bytes are exactly `cpu_reference_reject`'s
//!    `(accepted, correction)`. The kernel's arithmetic is the PRNG's
//!    integer steps, `div.rn`, `max`, `add`, `sub` and `mul`, all exact in
//!    the interpreter, so this is bit equality, not a tolerance.
//! 3. **The gate bites** — nudging a PRNG constant, the vocab stride, the
//!    2^-24 scale or the sentinel; forcing the draft-probability guard; or
//!    dropping a residual subtraction or a clamp is caught. The mutants
//!    that are *not* caught are named, with why they are equivalent.

use std::collections::HashMap;

use nsl_codegen::cfie_speculative_ptx::{cpu_reference_reject, emit_rejection_kernel, RejectionConfig};

/// The two `crate::` paths the frozen emitter names, supplied from the
/// library under test (the fixture is included into this test crate).
mod cfie_speculative {
    pub use nsl_codegen::cfie_speculative::TreeMask;
}
mod gpu_specs {
    pub use nsl_codegen::gpu_specs::ptx_isa_for_sm;
}

#[allow(dead_code)]
#[path = "fixtures/cfie_speculative_hand.rs"]
mod hand;

#[allow(dead_code)]
#[path = "support/cta_ptx_interp.rs"]
mod interp;
use interp::*;

const TARGET_BASE: u64 = 0x1000_0000;
const DRAFT_BASE: u64 = 0x2000_0000;
const TOKENS_BASE: u64 = 0x3000_0000;
const ACCEPTED_BASE: u64 = 0x4000_0000;
const CORRECTION_BASE: u64 = 0x5000_0000;
/// The reject kernel's launch block.
const BLOCK: u32 = 32;
/// Neither output's legal value: a word the kernel fails to write shows.
const SENTINEL: u32 = 0xDEAD_BEEF;

/// One launch's inputs.
#[derive(Debug, Clone)]
struct Case {
    k: u32,
    vocab: u32,
    /// `[k][vocab]`.
    target: Vec<f32>,
    draft: Vec<f32>,
    tokens: Vec<u32>,
    seed: u64,
}

impl Case {
    fn cfg(&self) -> RejectionConfig {
        RejectionConfig { k_tokens: self.k, vocab_size: self.vocab }
    }

    fn hand_cfg(&self) -> hand::RejectionConfig {
        hand::RejectionConfig { k_tokens: self.k, vocab_size: self.vocab, sm_version: 80 }
    }

    fn reference(&self) -> (u32, u32) {
        let (acc, corr) = cpu_reference_reject(&self.cfg(), &self.target, &self.draft, &self.tokens, self.seed);
        (acc as u32, corr)
    }
}

/// Deterministic values in [0, 1).
struct Lcg(u64);

impl Lcg {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Rows where the drafted token holds `p_on_draft` and the rest is spread
/// evenly — the module tests' shape.
fn target_rows(k: usize, vocab: usize, tokens: &[u32], p_on_draft: f32) -> Vec<f32> {
    let rest = (1.0 - p_on_draft) / (vocab as f32 - 1.0);
    (0..k * vocab).map(|i| if i % vocab == tokens[i / vocab] as usize { p_on_draft } else { rest }).collect()
}

/// Random normalised rows, random draft probabilities in (0, 1), random
/// tokens.
fn random_case(k: u32, vocab: u32, seed: u64, rng_seed: u64) -> Case {
    let mut rng = Lcg(seed);
    let mut target = Vec::with_capacity((k * vocab) as usize);
    for _ in 0..k {
        let row: Vec<f32> = (0..vocab).map(|_| rng.unit() + 0.01).collect();
        let total: f32 = row.iter().sum();
        target.extend(row.iter().map(|p| p / total));
    }
    let draft = (0..k).map(|_| 0.05 + 0.9 * rng.unit()).collect();
    let tokens = (0..k).map(|_| (rng.next_u64() % vocab as u64) as u32).collect();
    Case { k, vocab, target, draft, tokens, seed: rng_seed }
}

/// The cases the agreement and correctness gates sweep: every branch of
/// the kernel, each under several seeds including 0 (the golden-gamma
/// substitution).
fn cases() -> Vec<Case> {
    let mut out = Vec::new();
    for seed in [0u64, 1, 42, 0xDEAD_BEEF] {
        // All three accept: the ratio 0.9 / 0.5 beats every draw.
        let tokens = vec![1, 4, 6];
        out.push(Case { k: 3, vocab: 8, target: target_rows(3, 8, &tokens, 0.9), draft: vec![0.5; 3], tokens, seed });

        // Position 1's drafted token has no target mass: rejects there and
        // resamples the residual.
        let tokens = vec![1, 4, 6];
        let mut target = target_rows(3, 8, &tokens, 0.9);
        for v in 0..8 {
            target[8 + v] = if v == 4 { 0.0 } else { 1.0 / 7.0 };
        }
        out.push(Case { k: 3, vocab: 8, target, draft: vec![0.5; 3], tokens, seed });

        // A zero draft probability rejects without the division.
        let tokens = vec![3, 5];
        out.push(Case { k: 2, vocab: 8, target: target_rows(2, 8, &tokens, 0.9), draft: vec![0.0, 0.5], tokens, seed });

        // A negative draft probability rejects, and is clamped to 0 for the
        // residual.
        let tokens = vec![2, 5];
        out.push(Case { k: 2, vocab: 8, target: target_rows(2, 8, &tokens, 0.4), draft: vec![-0.3, 0.5], tokens, seed });
    }
    for seed in 0..12u64 {
        // p_target(draft) 0.2 <= p_draft 0.9: the residual zeroes the drafted
        // token, and rejections resample the rest.
        let tokens = vec![7];
        out.push(Case { k: 1, vocab: 16, target: target_rows(1, 16, &tokens, 0.2), draft: vec![0.9], tokens, seed });

        // All target mass on the drafted token and p_draft above it: an
        // empty residual falls back to the drafted token.
        let mut target = vec![0.0; 4];
        target[2] = 1.0;
        out.push(Case { k: 1, vocab: 4, target, draft: vec![2.0], tokens: vec![2], seed });
    }
    // Random rows: a mid-sized vocab, the K ceiling, a one-entry vocab.
    for (i, seed) in [3u64, 17, 0, 99].into_iter().enumerate() {
        out.push(random_case(5, 37, 0x5eed + i as u64, seed));
    }
    out.push(random_case(32, 3, 0xC0FFEE, 7));
    out.push(random_case(4, 1, 0xBEEF, 11));
    out
}

/// Global memory after running `ptx` over one CTA:
/// `[target, draft, tokens, accepted, correction]`.
fn run(ptx: &str, case: &Case, order: Order) -> Vec<Vec<u8>> {
    let prog = parse(ptx);
    let f32s = |v: &[f32]| -> Vec<u8> { v.iter().flat_map(|x| x.to_le_bytes()).collect() };
    let mut global = vec![
        Segment { base: TARGET_BASE, bytes: f32s(&case.target) },
        Segment { base: DRAFT_BASE, bytes: f32s(&case.draft) },
        Segment { base: TOKENS_BASE, bytes: case.tokens.iter().flat_map(|t| t.to_le_bytes()).collect() },
        Segment { base: ACCEPTED_BASE, bytes: SENTINEL.to_le_bytes().to_vec() },
        Segment { base: CORRECTION_BASE, bytes: SENTINEL.to_le_bytes().to_vec() },
    ];
    let args: HashMap<String, u64> = [
        ("target_probs_ptr", TARGET_BASE),
        ("draft_probs_ptr", DRAFT_BASE),
        ("draft_tokens_ptr", TOKENS_BASE),
        ("rng_seed", case.seed),
        ("out_accepted_ptr", ACCEPTED_BASE),
        ("out_correction_token_ptr", CORRECTION_BASE),
    ]
    .into_iter()
    .map(|(k, v)| (k.to_string(), v))
    .collect();
    let mut launch =
        Launch { prog: &prog, args: &args, global: &mut global, shared: vec![], ctaid: 0, ntid: BLOCK, steps: 0 };
    run_cta(&mut launch, order);
    global.into_iter().map(|s| s.bytes).collect()
}

fn word(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes[..4].try_into().unwrap())
}

fn kir_ptx(case: &Case) -> String {
    emit_rejection_kernel(&case.cfg()).0
}

fn hand_ptx(case: &Case) -> String {
    hand::emit_rejection_kernel(&case.hand_cfg()).0
}

// ---------------------------------------------------------------------------
// The gate
// ---------------------------------------------------------------------------

#[test]
fn the_kir_kernel_agrees_with_the_hand_written_one_bit_for_bit() {
    for case in cases() {
        let (hand, kir) = (hand_ptx(&case), kir_ptx(&case));
        for order in [Order::Ascending, Order::Descending] {
            let expect = run(&hand, &case, order);
            let got = run(&kir, &case, order);
            assert!(
                expect == got,
                "k={} vocab={} seed={} {order:?}: hand ({}, {}) kir ({}, {})",
                case.k,
                case.vocab,
                case.seed,
                word(&expect[3]),
                word(&expect[4]),
                word(&got[3]),
                word(&got[4])
            );
        }
    }
}

#[test]
fn the_answer_is_the_cpu_reference_exactly() {
    let mut rejected = 0;
    let mut fell_back = 0;
    for case in cases() {
        let mem = run(&kir_ptx(&case), &case, Order::Ascending);
        let got = (word(&mem[3]), word(&mem[4]));
        let want = case.reference();
        assert_eq!(got, want, "k={} vocab={} seed={}", case.k, case.vocab, case.seed);
        if want.0 < case.k {
            rejected += 1;
            if case.target[want.0 as usize * case.vocab as usize + case.tokens[want.0 as usize] as usize] == 1.0 {
                fell_back += 1;
            }
        }
    }
    // The sweep is only a gate if it reaches every branch.
    assert!(rejected > 20, "{rejected} rejecting cases");
    assert!(fell_back > 0, "no case reached the empty-residual fallback");
}

#[test]
fn inputs_are_left_untouched() {
    for case in cases().into_iter().take(6) {
        let mem = run(&kir_ptx(&case), &case, Order::Descending);
        let target: Vec<u8> = case.target.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(mem[0], target);
        let draft: Vec<u8> = case.draft.iter().flat_map(|x| x.to_le_bytes()).collect();
        assert_eq!(mem[1], draft);
    }
}

#[test]
fn the_kir_kernel_keeps_the_launch_abi_entry_name_and_header() {
    let case = &cases()[0];
    let (hand_text, kir_text) = (hand_ptx(case), kir_ptx(case));
    let (hand, kir) = (parse(&hand_text), parse(&kir_text));
    assert_eq!(hand.params, kir.params, "the launcher marshals these positionally");
    let names: Vec<&str> = kir.params.iter().map(|(_, n)| n.as_str()).collect();
    assert_eq!(
        names,
        ["target_probs_ptr", "draft_probs_ptr", "draft_tokens_ptr", "rng_seed", "out_accepted_ptr", "out_correction_token_ptr"]
    );
    assert!(kir_text.contains(".visible .entry nsl_cfie_spec_reject("));
    assert_eq!((hand.shared_bytes, kir.shared_bytes), (0, 0), "no shared memory");
    let header = |ptx: &str| -> Vec<String> { ptx.lines().take_while(|l| l.starts_with("//")).map(str::to_string).collect() };
    assert_eq!(header(&hand_text), header(&kir_text));
    assert!(!kir_text.contains("bar.sync"), "a serial kernel needs no barrier");
}

// ---------------------------------------------------------------------------
// The gate bites
// ---------------------------------------------------------------------------

/// The cases every mutation is judged on: the full sweep. A mutant is
/// caught when any case's memory differs from the hand kernel's, or the
/// interpreter faults on it.
fn caught(mutant: &str) -> bool {
    cases().iter().any(|case| {
        let expect = run(&hand_ptx(case), case, Order::Ascending);
        match std::panic::catch_unwind(|| run(mutant, case, Order::Ascending)) {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
    })
}

/// The KIR text for the mutation cases' configuration. Every case uses
/// its own `(k, vocab)`, so a mutant is built per configuration: this
/// returns the mutant for each distinct one, keyed by it.
fn mutate_all(f: impl Fn(&str) -> String) -> HashMap<(u32, u32), String> {
    let mut out = HashMap::new();
    for case in cases() {
        out.entry((case.k, case.vocab)).or_insert_with(|| f(&kir_ptx(&case)));
    }
    out
}

/// Whether the per-configuration mutants are caught on any case.
fn caught_per_config(mutants: &HashMap<(u32, u32), String>) -> bool {
    cases().iter().any(|case| {
        let mutant = &mutants[&(case.k, case.vocab)];
        let expect = run(&hand_ptx(case), case, Order::Ascending);
        match std::panic::catch_unwind(|| run(mutant, case, Order::Ascending)) {
            Ok(mem) => mem != expect,
            Err(_) => true,
        }
    })
}

/// Every line starting with `prefix` and ending with `from`, rewritten to
/// end with `to`; at least `min` such lines must exist.
fn rewrite(ptx: &str, prefix: &str, from: &str, to: &str, min: usize) -> String {
    let hit = |l: &str| l.trim_start().starts_with(prefix) && l.ends_with(from);
    assert!(ptx.lines().filter(|l| hit(l)).count() >= min, "`{prefix} … {from}` should appear at least {min}x");
    ptx.lines()
        .map(|l| if hit(l) { format!("{}{to}", &l[..l.len() - from.len()]) } else { l.to_string() })
        .collect::<Vec<_>>()
        .join("\n")
}

/// The `n`th line starting with `prefix`, replaced by `f(line)`.
fn replace_nth(ptx: &str, prefix: &str, n: usize, f: impl Fn(&str) -> String) -> String {
    let mut seen = 0;
    let out: Vec<String> = ptx
        .lines()
        .map(|l| {
            if l.trim_start().starts_with(prefix) {
                seen += 1;
                if seen == n + 1 {
                    return f(l);
                }
            }
            l.to_string()
        })
        .collect();
    assert!(seen > n, "fewer than {} `{prefix}` lines", n + 1);
    out.join("\n")
}

#[test]
fn the_mutation_baseline_is_clean() {
    // Without it, `caught` could be reporting a difference the mutation
    // did not cause.
    assert!(!caught_per_config(&mutate_all(|p| p.to_string())));
}

#[test]
fn nudging_a_prng_or_baked_constant_is_caught() {
    let f = |v: f32| format!("0f{:08X}", v.to_bits());
    for (prefix, from, to) in [
        // The three xorshift amounts and the output shift, in both draws.
        ("mov.u32", ", 12;".to_string(), ", 13;".to_string()),
        ("mov.u32", ", 25;".to_string(), ", 24;".to_string()),
        ("mov.u32", ", 27;".to_string(), ", 26;".to_string()),
        ("mov.u32", ", 40;".to_string(), ", 41;".to_string()),
        // The multiplier and the zero-seed golden gamma.
        ("mov.u64", format!(", {};", 0x2545_F491_4F6C_DD1Du64), format!(", {};", 0x2545_F491_4F6C_DD1Fu64)),
        ("mov.u64", format!(", {};", 0x9E37_79B9_7F4A_7C15u64), format!(", {};", 0x9E37_79B9_7F4A_7C17u64)),
        // The 2^-24 scale and the all-accept sentinel.
        ("mov.f32", format!(", {};", f(1.0 / 16_777_216.0)), format!(", {};", f(1.0 / 8_388_608.0))),
        ("mov.u32", format!(", {};", u32::MAX), format!(", {};", u32::MAX - 1)),
    ] {
        let mutants = mutate_all(|p| rewrite(p, prefix, &from, &to, 1));
        assert!(caught_per_config(&mutants), "`{prefix} … {from}` -> `{to}` went unnoticed");
    }
    // The vocab stride, on the one configuration whose vocab is distinct
    // from every other constant in the module.
    let case = cases().into_iter().find(|c| c.vocab == 37).unwrap();
    let mutant = rewrite(&kir_ptx(&case), "mov.u32", ", 37;", ", 36;", 1);
    assert!(caught(&mutant), "the vocab stride nudged went unnoticed");
}

/// The three `x > 0` tests, each forced true (`x == x`, true for any
/// non-NaN `x`):
///
/// * the draft-probability guard (index 0) is caught: a zero draft
///   probability then divides and accepts on `r < inf`;
/// * the residual-mass test (index 1) is an **equivalent mutant**: forcing
///   the resample on an empty residual walks a CDF of zeros, which never
///   selects, so it stores the drafted token, as the fallback does;
/// * the positive-entry test in the CDF walk (index 2) is an **equivalent
///   mutant** on every input the sweep can build: the running sum only
///   grows at a positive entry, so the first index where it reaches the
///   target is positive unless the draw is exactly 0.
#[test]
fn forcing_the_draft_guard_is_caught_and_the_other_positivity_tests_are_equivalent() {
    for (n, expect_caught) in [(0, true), (1, false), (2, false)] {
        let mutants = mutate_all(|p| {
            replace_nth(p, "setp.gt.f32 ", n, |l| {
                let (lead, rest) = l.split_once("setp.gt.f32 ").unwrap();
                let mut ops = rest.trim_end_matches(';').split(',').map(str::trim);
                let (pred, a) = (ops.next().unwrap(), ops.next().unwrap());
                format!("{lead}setp.eq.f32 {pred}, {a}, {a};")
            })
        });
        assert_eq!(caught_per_config(&mutants), expect_caught, "positivity test {n}");
    }
}

#[test]
fn dropping_a_residual_subtraction_or_clamp_is_caught() {
    // The two residual subtractions (the total's and the walk's): each
    // replaced by a copy of p_target.
    for n in 0..2 {
        let mutants = mutate_all(|p| {
            replace_nth(p, "sub.f32 ", n, |l| {
                let (lead, rest) = l.split_once("sub.f32 ").unwrap();
                let mut ops = rest.trim_end_matches(';').split(',').map(str::trim);
                let (d, a) = (ops.next().unwrap(), ops.next().unwrap());
                format!("{lead}mov.f32 {d}, {a};")
            })
        });
        assert!(caught_per_config(&mutants), "residual subtraction {n} dropped went unnoticed");
    }
    // The three clamps at 0 (the draft probability's and the two residual
    // entries'): each replaced by a copy of its operand.
    let clamps = kir_ptx(&cases()[0]).lines().filter(|l| l.trim_start().starts_with("max.f32 ")).count();
    assert_eq!(clamps, 3, "the draft clamp and the two residual clamps");
    for n in 0..3 {
        let mutants = mutate_all(|p| {
            replace_nth(p, "max.f32 ", n, |l| {
                let (lead, rest) = l.split_once("max.f32 ").unwrap();
                let mut ops = rest.trim_end_matches(';').split(',').map(str::trim);
                let (d, a) = (ops.next().unwrap(), ops.next().unwrap());
                format!("{lead}mov.f32 {d}, {a};")
            })
        });
        assert!(caught_per_config(&mutants), "clamp {n} dropped went unnoticed");
    }
}

#[test]
fn the_interpreter_models_the_new_integer_forms() {
    // Shifts at or past the width produce 0, as PTX's do; a 64-bit shift
    // takes a 32-bit amount; hex immediates parse; the two new converts.
    let ptx = "\
.version 7.0
.target sm_70
.address_size 64
.visible .entry t(.param .u64 param_o) {
    .reg .b32 %r<4>;
    .reg .b64 %rd<4>;
    .reg .f32 %f<1>;
    ld.param.u64 %rd0, [param_o];
    mov.u64 %rd1, 0x8000000000000001;
    mov.u32 %r0, 64;
    shl.b64 %rd2, %rd1, %r0;
    st.global.u32 [%rd0], %rd2;
    mov.u32 %r1, 63;
    shr.u64 %rd2, %rd1, %r1;
    xor.b64 %rd2, %rd2, %rd1;
    cvt.u32.u64 %r2, %rd2;
    st.global.u32 [%rd0+4], %r2;
    mov.u32 %r3, 16777217;
    cvt.rn.f32.u32 %f0, %r3;
    st.global.f32 [%rd0+8], %f0;
    ret;
}
";
    let prog = parse(ptx);
    let mut global = vec![Segment { base: 0x100, bytes: vec![0; 12] }];
    let args: HashMap<String, u64> = [("o".to_string(), 0x100)].into_iter().collect();
    let mut launch = Launch { prog: &prog, args: &args, global: &mut global, shared: vec![], ctaid: 0, ntid: 1, steps: 0 };
    run_cta(&mut launch, Order::Ascending);
    let w = |i: usize| u32::from_le_bytes(global[0].bytes[4 * i..4 * i + 4].try_into().unwrap());
    // 1 << 64 is 0; (x >> 63) ^ x = 1 ^ 0x8000000000000001, low word 0.
    assert_eq!(w(0), 0);
    assert_eq!(w(1), 0);
    // 2^24 + 1 rounds to nearest even: 2^24.
    assert_eq!(f32::from_bits(w(2)), 16_777_216.0);
}
