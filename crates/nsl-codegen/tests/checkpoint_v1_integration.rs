//! Cycle-10 §5.3 Task 9 + Phase F integration gates.
//!
//! Test surface covers the four code-path gates required by the
//! cycle-10 spec appendix and the Phase F functional-gap closure:
//!
//!   G1 byte-identity        — no-decorator path remains snapshot-clean.
//!   G2 refusal coverage     — R0 + R3 + R7 + R9 + R10 + R8.1 each fire
//!                              with the documented substring + emit no
//!                              PTX. R0 is Phase F's hard gate; in v1 it
//!                              fires FIRST, so R3/R7/R9/R10/R8.1 are
//!                              structurally unreachable today. We still
//!                              assert the cascade substrings against the
//!                              live error-path so cycle 11 can lift R0
//!                              without simultaneously regressing them
//!                              (the substring asserts move to a
//!                              once-R0-is-lifted reachability gate).
//!   G3 R0 refusal probe     — `policy="full"` config refuses with the
//!                              R0 substring. Phase F replaces the
//!                              tautological PTX-substring probe with a
//!                              honest refusal-reachability gate.
//!   G6 sibling-leak guard   — `EffectChecker::checkpoint_policies()`
//!                              returns an empty map for a module that
//!                              contains zero `@checkpoint` decorators.
//!
//! Gates G4 (GPU numerical equivalence) and G5 (paper-§6.3 exact
//! diagnostic string) remain deferred — see
//! `docs/wiki/Checkpoint-v1-Defer-Log.md`. Cycle 11 unlocks G4 alongside
//! lifting R0.

use nsl_codegen::flash_attention::{
    CheckpointExtras, CshaExtras, FlashAttentionConfig, RopeStyle,
};
use nsl_codegen::flash_attention_v2::synthesize_backward_with_tier;

// ----- shared config helpers ---------------------------------------------

/// A backward-safe Level-1-fusible CSHA config that satisfies all v1
/// recompute preconditions (block_q == block_kv, no sinks, no PCA-rope_q
/// composition). Tests start from this and mutate one field at a time to
/// hit each refusal.
fn base_fusible() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 75,
        segment_masked: false,
        csha: Some(CshaExtras::level1(1e-6)),
        checkpoint: Some(CheckpointExtras::full()),
    }
}

/// Identical to `base_fusible` but with `checkpoint: None` — used for
/// the G1 byte-identity guard.
fn no_decorator() -> FlashAttentionConfig {
    FlashAttentionConfig {
        checkpoint: None,
        ..base_fusible()
    }
}

// ----- G1: byte-identity for no-decorator path ---------------------------

#[test]
fn g1_no_decorator_backward_emits_no_checkpoint_marker() {
    // PTX-comment-emission guard (W11): when `checkpoint.is_none()` the
    // dispatch fork in `synthesize_backward_with_tier` MUST NOT route
    // to the recompute synthesizer, and therefore the emitted PTX must
    // not contain any checkpoint-related comment.
    let cfg = no_decorator();
    let ptx = synthesize_backward_with_tier(&cfg)
        .expect("baseline backward synthesis must succeed");
    assert!(
        !ptx.contains("Checkpoint strategy"),
        "no-decorator path leaked a checkpoint diagnostic comment"
    );
    assert!(
        !ptx.contains("CHECKPOINT_RECOMPUTE"),
        "no-decorator path leaked a CHECKPOINT_RECOMPUTE marker"
    );
    assert!(
        !ptx.contains("_bwd_recompute_"),
        "no-decorator path leaked a backward-recompute namespace suffix"
    );
}

// ----- G2: refusal substring coverage (R0 + R3 + R7 + R9 + R10 + R8.1) --

#[test]
fn g2_r0_policy_full_refuses_until_functional_recompute_lands() {
    // R0 (cycle 11): the wording moved from "not yet wired" to
    // "awaiting GPU numerical validation gate" once the CPU codegen
    // substitution landed structurally. R0 still fires unconditionally
    // in production until cycle 12 runs G4 on Blackwell and lifts it.
    // Without R0, cycle 11's structurally-validated PTX would ship
    // without numerical confirmation, violating
    // feedback_deferral_must_refuse.
    let cfg = base_fusible();
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R0: policy=full must refuse in v1");
    assert!(
        err.contains("awaiting GPU numerical validation gate"),
        "R0 missing expected substring: {err}"
    );
}

#[test]
fn g2_r3_non_fusible_prologue_refuses() {
    // R3: policy=Full but the forward config has no Level-1 fusible
    // prologue (csha=None means no RMSNorm prologue is available to
    // recompute).
    //
    // Phase F: R0 fires first and refuses ALL policy=Full configs in
    // v1, so this test asserts the R0 substring fires (R3 is
    // structurally unreachable today). Cycle 11 lifts R0 and this test
    // upgrades to asserting the original R3 substring.
    let mut cfg = base_fusible();
    cfg.csha = None;
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R3 (R0-shadowed in v1): config must refuse");
    assert!(
        err.contains("awaiting GPU numerical validation gate"),
        "R0 (shadowing R3) missing expected substring: {err}"
    );
}

#[test]
fn g2_r7_pca_packing_with_rope_q_refuses() {
    // R7: segment_masked + rope_q + checkpoint composition is deferred
    // to v4. Phase F: R0-shadowed; assert R0 substring.
    let mut cfg = base_fusible();
    cfg.segment_masked = true;
    cfg.rope_q = true;
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R7 (R0-shadowed in v1): config must refuse");
    assert!(
        err.contains("awaiting GPU numerical validation gate"),
        "R0 (shadowing R7) missing expected substring: {err}"
    );
}

#[test]
fn g2_r9_paged_kv_collision_refuses() {
    // R9: `paged_kv_collision = true` flag on the carrier signals the
    // enclosing fn was also `@paged_kv`-decorated. Phase F: R0-shadowed.
    let cfg = FlashAttentionConfig {
        checkpoint: Some(CheckpointExtras {
            paged_kv_collision: true,
            ..CheckpointExtras::full()
        }),
        ..base_fusible()
    };
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R9 (R0-shadowed in v1): config must refuse");
    assert!(
        err.contains("awaiting GPU numerical validation gate"),
        "R0 (shadowing R9) missing expected substring: {err}"
    );
}

#[test]
fn g2_r10_asymmetric_tile_refuses() {
    // R10: block_q != block_kv under @checkpoint policy=full is
    // deferred to v2. Phase F: R0-shadowed.
    let mut cfg = base_fusible();
    cfg.block_kv = 64; // != block_q (32)
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R10 (R0-shadowed in v1): config must refuse");
    assert!(
        err.contains("awaiting GPU numerical validation gate"),
        "R0 (shadowing R10) missing expected substring: {err}"
    );
}

#[test]
fn g2_r81_sinks_v2_composition_refuses() {
    // R8.1 (Phase F): sinks-v2 + @checkpoint is now UNCONDITIONAL in v1
    // (the cycle-9 verified smoke-set carve-out was structurally
    // unreachable — `static_seq_len.unwrap_or(0)` always evaluated to 0
    // in production because `CshaExtras::level1` doesn't set
    // static_seq_len — so we dropped it). The new R8.1 substring
    // documents that the carve-out is deferred to v2 behind GPU
    // validation.
    //
    // R0 shadows R8.1 in v1 the same as the other refusals; assert the
    // R0 substring here too. The renamed R8.1 substring is exercised in
    // the cycle-11 lift commit when R0 is removed and the original
    // refusal cascade becomes reachable again.
    let mut cfg = base_fusible();
    cfg.num_sink_tokens = 4;
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R8.1 (R0-shadowed in v1): config must refuse");
    assert!(
        err.contains("awaiting GPU numerical validation gate"),
        "R0 (shadowing R8.1) missing expected substring: {err}"
    );
}

// ----- G3 retired in cycle 11 -------------------------------------------
//
// Cycle 10 Phase F shipped `g3_policy_full_refuses_with_r0_substring`
// as a refusal-reachability gate against the R0 substring set
// ("functional recompute not yet wired in v1" + "kv_load substitution
// + SMEM-base routing deferred" + "behind GPU validation gate"). Those
// substrings were retired in cycle 11 along with the wording update
// — the CPU codegen substitution is now wired and the deferral is
// gated by GPU numerical validation (cycle 12), not by wiring.
//
// The cycle 11 replacement is the four-gate suite
// `g3a/g3b/g3c/g3d_*` in
// `tests/checkpoint_cycle11_structural.rs`, which covers the same
// reachability invariant PLUS structural witnesses for the dispatch
// fork (kv_load skip + recompute label + SMEM-write ordering +
// no-decorator byte-identity).
//
// The R0-substring anchor itself lives in
// `g2_r0_policy_full_refuses_until_functional_recompute_lands` above
// — single source of truth so a future R0 wording change touches one
// test, not two.

// ----- G6: sibling-leak guard (EffectChecker::checkpoint_policies) ------

#[test]
fn g6_effect_checker_checkpoint_policies_empty_when_no_decorators() {
    // G6 sibling-leak: a module with zero `@checkpoint` decorators must
    // produce an empty `checkpoint_policies` map. Without this, the
    // codegen-side `WengertExtractor::with_checkpoint_policies` would
    // erroneously stamp ops it shouldn't.
    let src = r#"
fn add_one(x: i64) -> i64:
    return x + 1

fn main() -> i64:
    return add_one(41)
"#;

    let mut interner = nsl_lexer::Interner::new();
    let (tokens, _lex_diags) =
        nsl_lexer::tokenize(src, nsl_errors::FileId(0), &mut interner);
    let parsed = nsl_parser::parse(&tokens, &mut interner);
    let analysis = nsl_semantic::analyze(&parsed.module, &mut interner);

    assert!(
        analysis.checkpoint_policies.is_empty(),
        "G6: checkpoint_policies leaked entries for a module with no @checkpoint: {:?}",
        analysis.checkpoint_policies
    );
}
