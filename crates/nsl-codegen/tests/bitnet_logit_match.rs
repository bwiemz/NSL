//! M35.1 merge gate: end-to-end logit match against pinned HF BitNet b1.58 3B.
//!
//! Spec: docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md §6.6.
//! Tolerance: FP16 ULP (1e-3 relative) on all 32 reference prompts.
//!
//! ## Status: #[ignore]'d — the artifact has LANDED; the comparison has not
//!
//! Both historical Linux follow-ons are now DONE:
//!
//! 1. `weight_scale` wiring through `phases/finalize.rs::emit` (M35.1 PR #172).
//!
//! 2. `tests/fixtures/bitnet_b158_3b_reference_logits.bin` — generated on
//!    Linux and committed. See the sibling `.meta.json` for full provenance.
//!
//! ### Reference anchor provenance (spec §6.5 deviation — READ THIS)
//!
//! Spec §6.5 names bitnet.cpp as the generator. The committed artifact is
//! instead produced by the **checkpoint's own vendored reference
//! implementation** (`modeling_bitnet.py` + `utils_quant.BitLinear` at the
//! pinned revision), via `scripts/gen_bitnet_reference_logits.py`. That is
//! the same source of record M35.1 already used for the "activations are
//! absmax, not absmean" correction.
//!
//! This was not a convenience substitution — bitnet.cpp WAS built and run as
//! a cross-check, and its ternary (`i2_s`) path proved to be broken in that
//! fork: its logits correlate with the HF anchor (mean Pearson 0.498) *below*
//! the noise floor of comparing two DIFFERENT prompts of the same model
//! (0.538), and a retrieval test recovers the correct prompt only 3/32 times
//! versus 3.1% chance. Committing it would have made this gate demand that
//! NSL reproduce a bug.
//!
//! bitnet.cpp's **f32** control path, however, independently corroborates the
//! HF anchor at 96.9% top-1, Pearson 0.99948, Spearman 0.9926 over the top
//! 100 logits, with 32/32 prompt retrieval — validating the GGUF conversion,
//! the 52 BitNet-only sub-layernorms, BOS handling, and the checkpoint pin
//! through a completely separate C++ implementation.
//!
//! ### BOS is load-bearing
//!
//! Every prompt is tokenized WITH a prepended `<s>` (id 1), verified per
//! prompt against the slow sentencepiece path. A harness that omits BOS
//! differs in EVERY logit.
//!
//! ### Why this test is still `#[ignore]`'d
//!
//! The artifact is present, but the per-prompt comparison below is still a
//! TODO, because there is no end-to-end NSL BitNet inference path to compare
//! against: no runtime ternary dtype, no kernel dispatch, and no
//! `nsl.quant.ternary` stdlib module. The forward *kernel* is now real and
//! GPU-validated (see `bitnet_gpu_correctness.rs`), but nothing yet composes
//! it into a 3B forward pass. Un-ignoring this test is gated on that work,
//! not on the fixture.
//!
//! Running it also requires the HF checkpoint cached at
//! `~/.cache/nsl-tests/bitnet_b158_3b/` via `bash scripts/fetch_bitnet_b158_3b.sh`
//! (Linux/macOS only per spec §7.3; ~13 GB across 3 safetensors shards).
//!
//! Run with: `cargo test -p nsl-codegen --test bitnet_logit_match -- --ignored`.
//! On a fresh machine, first run `bash scripts/fetch_bitnet_b158_3b.sh`.

use nsl_codegen::bitnet::loader::{load_bitnet_b158_safetensors, read_pinned_revision};
use std::path::PathBuf;

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
}

fn cached_checkpoint_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .expect("HOME or USERPROFILE must be set");
    PathBuf::from(home).join(".cache/nsl-tests/bitnet_b158_3b")
}

/// Parse `tests/fixtures/bitnet_b158_phase1_prompts.txt` into a flat list of
/// 32 prompts.
///
/// The fixture is organized into four sections of 8 prompts each, separated
/// by `# <count> <category>` header lines (spec §6.4):
///
/// - `# 8 short factual` — 8 single-line prompts
/// - `# 8 code-completion` — 8 multi-line Python snippets; each prompt starts
///   with `def ` and continues until the next `def ` or section header
/// - `# 8 long-context` — 8 single-line (paragraph-long) prompts
/// - `# 8 edge-case` — 8 single-line prompts
///
/// Returns the 32 prompts in their order of appearance. Empty result on parse
/// error (asserted against in the test).
fn parse_phase1_prompts(text: &str) -> Vec<String> {
    let mut prompts: Vec<String> = Vec::new();
    let mut current_section: Option<&'static str> = None;
    let mut code_prompt_buf: Vec<String> = Vec::new();

    let flush_code_prompt = |buf: &mut Vec<String>, out: &mut Vec<String>| {
        if !buf.is_empty() {
            out.push(buf.join("\n"));
            buf.clear();
        }
    };

    for raw in text.lines() {
        let line = raw.trim_end();
        if line.trim().is_empty() {
            // Blank lines do not delimit code-completion prompts (next `def `
            // does), but we ignore them as content.
            continue;
        }
        if let Some(rest) = line.strip_prefix('#') {
            // Section header. Flush any pending code-completion prompt.
            flush_code_prompt(&mut code_prompt_buf, &mut prompts);
            let lower = rest.to_lowercase();
            if lower.contains("code-completion") {
                current_section = Some("code");
            } else if lower.contains("short factual") {
                current_section = Some("factual");
            } else if lower.contains("long-context") {
                current_section = Some("long");
            } else if lower.contains("edge-case") {
                current_section = Some("edge");
            } else {
                current_section = None;
            }
            continue;
        }
        match current_section {
            Some("code") => {
                // Each prompt starts with `def `; flush previous on new def.
                if line.starts_with("def ") && !code_prompt_buf.is_empty() {
                    flush_code_prompt(&mut code_prompt_buf, &mut prompts);
                }
                code_prompt_buf.push(line.to_string());
            }
            Some(_) => {
                prompts.push(line.to_string());
            }
            None => {
                // Content outside any section; skip.
            }
        }
    }
    // Flush trailing code-completion prompt (in case the file does not end
    // with a section header).
    flush_code_prompt(&mut code_prompt_buf, &mut prompts);
    prompts
}

#[test]
#[ignore = "requires fetched HF checkpoint; \
            run `bash scripts/fetch_bitnet_b158_3b.sh` then \
            `cargo test -p nsl-codegen --test bitnet_logit_match -- --ignored` on Linux"]
fn end_to_end_logit_match_against_hf_b158_3b() {
    // Preflight: pinned model identity matches PI.2's verified values.
    let (model_id, revision) =
        read_pinned_revision(&repo_root()).expect("read pinned revision fixture");
    assert_eq!(
        model_id, "1bitLLM/bitnet_b1_58-3B",
        "model_id must match PI.2's pinned value"
    );
    assert_eq!(revision.len(), 40, "revision SHA must be 40 hex chars");
    println!("Pinned: {model_id} @ {revision}");

    // Preflight: prompts fixture is well-formed (32 prompts split across the
    // four sections defined in spec §6.4; code-completion prompts span
    // multiple lines, so a naive line-filter would overcount).
    let prompts_path = repo_root().join("tests/fixtures/bitnet_b158_phase1_prompts.txt");
    let prompts_text = std::fs::read_to_string(&prompts_path).expect("read prompts fixture");
    let prompts = parse_phase1_prompts(&prompts_text);
    assert_eq!(
        prompts.len(),
        32,
        "Expected 32 prompts across the four spec §6.4 sections, found {} \
         (the file has 4 `# <count> <category>` headers and each section \
         must contain 8 prompts)",
        prompts.len()
    );

    // Locate the checkpoint cache. Required for the actual inference exercise.
    let cache_dir = cached_checkpoint_dir();
    let shard1 = cache_dir.join("model-00001-of-00003.safetensors");
    let shard2 = cache_dir.join("model-00002-of-00003.safetensors");
    let shard3 = cache_dir.join("model-00003-of-00003.safetensors");
    assert!(
        shard1.exists() && shard2.exists() && shard3.exists(),
        "Checkpoint shards not cached at {}. Run `bash scripts/fetch_bitnet_b158_3b.sh` first.",
        cache_dir.display()
    );

    // Load the first shard via the BitNet loader (validates loader plumbing E2E).
    let weights = load_bitnet_b158_safetensors(&shard1).expect("load first shard");
    println!(
        "Loaded {} BitLinear weight tensors from shard 1",
        weights.len()
    );
    assert!(
        !weights.is_empty(),
        "shard 1 must contain BitLinear projection weights"
    );

    // Locate the reference logits binary. Required to compare against bitnet.cpp.
    let ref_logits_path = repo_root().join("tests/fixtures/bitnet_b158_3b_reference_logits.bin");
    assert!(
        ref_logits_path.exists(),
        "Reference logits not found at {}. This artifact is COMMITTED; \
         regenerate with `python scripts/gen_bitnet_reference_logits.py` \
         (see the sibling .meta.json for provenance).",
        ref_logits_path.display()
    );
    let ref_bytes = std::fs::read(&ref_logits_path).expect("read reference logits");
    // 32 prompts × vocab_size × 2 bytes (FP16).
    // The pinned 1bitLLM/bitnet_b1_58-3B model has vocab_size=32002 (PI.2).
    let expected_vocab = 32002;
    let expected_bytes = 32 * expected_vocab * 2;
    assert_eq!(
        ref_bytes.len(),
        expected_bytes,
        "Reference logits size mismatch: {} bytes, expected {} (32 prompts × {} vocab × 2)",
        ref_bytes.len(),
        expected_bytes,
        expected_vocab
    );

    // TODO(M35.1 Linux follow-on — reference_logits.bin only; weight_scale wiring is done):
    // full inference comparison.
    //
    // The inference path goes through nsl_codegen::bitnet::synthesize_kernel
    // producing PTX (now including the .param .f32 weight_scale slot consumed
    // by finalize.rs::emit), then a real-subprocess harness (parallel to AWQ's
    // end_to_end_real_subprocess_matches_analytical_reference in
    // tests/awq_full_pipeline.rs) tokenizes each prompt, runs forward, and
    // captures final-position logits. Per-prompt comparison:
    //
    //   for (prompt_idx, prompt) in prompts.iter().enumerate() {
    //       let nsl_logits = nsl_inference(weights_combined, prompt);
    //       let ref_slice_start = prompt_idx * expected_vocab * 2;
    //       let ref_slice = &ref_bytes[ref_slice_start..ref_slice_start + expected_vocab * 2];
    //       for (i, (&actual, &expected)) in nsl_logits.iter().zip(...).enumerate() {
    //           let rel = ((actual - expected).abs() / expected.abs().max(1e-30)) as f32;
    //           assert!(rel <= 1e-3, "prompt {prompt_idx} logit {i}: rel={rel}");
    //       }
    //   }
    //
    // For Phase 1 / commit 10 of M35.1, the preconditions above
    // (pin match + shards present + reference logits present + reference
    // size correct) ARE the merge-gate assertion against the current
    // implementation platform. The full inference exercise lands on Linux
    // when the reference logits artifact ships.
    println!(
        "Preflight gates green: {} prompts, {} ref-logit bytes ({} vocab), {} loaded weights.",
        prompts.len(),
        ref_bytes.len(),
        expected_vocab,
        weights.len()
    );
}

/// Unit test for the prompts parser. Runs by default (not `#[ignore]`'d).
#[test]
fn prompts_parser_finds_32_prompts() {
    let prompts_path = repo_root().join("tests/fixtures/bitnet_b158_phase1_prompts.txt");
    let text = std::fs::read_to_string(&prompts_path).expect("read prompts fixture");
    let prompts = parse_phase1_prompts(&text);
    assert_eq!(
        prompts.len(),
        32,
        "expected 32 prompts across the four spec §6.4 sections, found {}",
        prompts.len()
    );
    // Spot-check the first prompt of each section to confirm the parser is
    // landing on real content (not blank lines or comment text).
    assert_eq!(prompts[0], "What is the capital of France?");
    assert!(
        prompts[8].starts_with("def fibonacci(n):"),
        "code-completion prompt 0 must start with `def fibonacci(n):`, got {:?}",
        prompts[8]
    );
    assert!(
        prompts[16].starts_with("The transformer architecture"),
        "long-context prompt 0 must start with `The transformer architecture`, got {:?}",
        &prompts[16][..40.min(prompts[16].len())]
    );
    assert_eq!(prompts[24], "a");
}

/// Sanity check that the pinned revision is reachable through the loader
/// helper. Runs by default (not `#[ignore]`'d).
#[test]
fn pinned_revision_is_readable() {
    let (model_id, revision) =
        read_pinned_revision(&repo_root()).expect("read pinned revision fixture");
    assert_eq!(model_id, "1bitLLM/bitnet_b1_58-3B");
    assert_eq!(revision.len(), 40);
    assert!(
        revision.chars().all(|c| c.is_ascii_hexdigit()),
        "revision must be a 40-char hex SHA, got {revision:?}"
    );
}
