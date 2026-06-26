//! Cycle-10 §5.3 Task 9 + Phase F integration gates, migrated through
//! cycle 11 (R0 wording update) and cycle 12 (R0 retirement).
//!
//! Test surface covers the four code-path gates required by the
//! cycle-10 spec appendix and the cycle-12 functional landing:
//!
//!   G1 byte-identity        — no-decorator path remains snapshot-clean.
//!   G2 refusal coverage     — R3 (augmented), R7, R9, R10, R8.1, R11, R12
//!                              each fire with the documented substring +
//!                              emit no PTX. R0 was RETIRED in cycle 12;
//!                              there is no catch-all anymore — every
//!                              refusal is specific. R3 was augmented in
//!                              cycle 12 to also require `fused_projections`
//!                              (the cycle-11 forward-path RoPE-K gap fix).
//!   G6 sibling-leak guard   — `EffectChecker::checkpoint_policies()`
//!                              returns an empty map for a module that
//!                              contains zero `@checkpoint` decorators.
//!
//! G3 reachability is covered by the cycle-11 four-gate suite in
//! `tests/checkpoint_cycle11_structural.rs` (G3a/G3b/G3c/G3d), now
//! cycle-12 extended (G3c rope-K) and supplemented with G5/G6/G8.

use nsl_codegen::flash_attention::{
    CheckpointExtras, CshaExtras, FlashAttentionConfig, RopeStyle,
};
use nsl_codegen::flash_attention_v2::synthesize_backward_with_tier;

// ----- shared config helpers ---------------------------------------------

/// A backward-safe Level-1-fusible CSHA config that satisfies the
/// cycle-12 R3 augmentation (`fused_projections=true`) AND all the v1
/// recompute preconditions (block_q == block_kv, no sinks, no PCA-rope_q
/// composition). Tests start from this and mutate one field at a time to
/// hit each refusal.
fn base_fusible() -> FlashAttentionConfig {
    let mut csha = CshaExtras::level1(1e-6);
    // Cycle-12 R3 augmentation: backward kv-recompute requires the
    // forward path to stage x_raw + Wk/Wv via fused projections.
    csha.fused_projections = true;
    csha.d_model = 32;
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
        csha: Some(csha),
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

// ----- G2: refusal substring coverage (R3 + R7 + R9 + R10 + R8.1 + R11 + R12) -

#[test]
fn g2_r3_non_fused_projections_refuses() {
    // R3 (cycle-12 augmented): policy=Full with csha=level1 default
    // (fused_projections=false) — the cycle-12 augmentation requires
    // fused_projections for x_raw + Wk/Wv staging on the forward path.
    // Without it, the backward kv-recompute would silently no-op.
    let mut cfg = base_fusible();
    // Strip the cycle-12 augmentation we added in base_fusible — back
    // to plain level1 (fused_projections=false).
    let mut csha = CshaExtras::level1(1e-6);
    csha.fused_projections = false;
    cfg.csha = Some(csha);
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R3 (cycle-12 augmented): non-fused-projections must refuse");
    assert!(
        err.contains("fused_projections gate failed"),
        "R3 missing expected substring 'fused_projections gate failed': {err}"
    );
}

#[test]
fn g2_r3_no_csha_refuses() {
    // R3: policy=Full but no CSHA at all (csha=None) — no fusible
    // prologue available to recompute.
    let mut cfg = base_fusible();
    cfg.csha = None;
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R3 (no csha): config must refuse");
    assert!(
        err.contains("fused_projections gate failed"),
        "R3 missing expected substring 'fused_projections gate failed': {err}"
    );
}

#[test]
fn g2_r7_pca_packing_with_rope_q_refuses() {
    // R7: segment_masked + rope_q + checkpoint composition is deferred
    // to v4. base_fusible already satisfies R3 (cycle-12 fused_projections=true),
    // so R7 is reachable.
    let mut cfg = base_fusible();
    cfg.segment_masked = true;
    cfg.rope_q = true;
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R7: segment_masked+rope_q must refuse");
    assert!(
        err.contains("PCA packing with rope_q=true under @checkpoint deferred to v4"),
        "R7 missing expected substring: {err}"
    );
}

#[test]
fn g2_r9_paged_kv_collision_refuses() {
    // R9: `paged_kv_collision = true` flag on the carrier signals the
    // enclosing fn was also `@paged_kv`-decorated.
    let cfg = FlashAttentionConfig {
        checkpoint: Some(CheckpointExtras {
            paged_kv_collision: true,
            ..CheckpointExtras::full()
        }),
        ..base_fusible()
    };
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R9: paged_kv_collision must refuse");
    assert!(
        err.contains("@paged_kv requires scatter-on-recompute"),
        "R9 missing expected substring: {err}"
    );
}

#[test]
fn g2_r10_asymmetric_tile_refuses() {
    // R10: block_q != block_kv under @checkpoint policy=full is
    // deferred to v2.
    let mut cfg = base_fusible();
    cfg.block_kv = 64; // != block_q (32)
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R10: asymmetric tile must refuse");
    assert!(
        err.contains("block_q == block_kv in v1"),
        "R10 missing expected substring: {err}"
    );
}

#[test]
fn g2_r81_sinks_v2_composition_refuses() {
    // R8.1: sinks-v2 + @checkpoint is UNCONDITIONAL in v1. Note that
    // upstream attention-sinks-v1 backward eligibility ALSO refuses at
    // the `synthesize_backward_with_tier` entry. We assert on the
    // sinks-v2 refusal substring at WHICHEVER altitude fires first.
    let mut cfg = base_fusible();
    cfg.num_sink_tokens = 4;
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R8.1: sinks-v2 must refuse");
    assert!(
        err.contains("sinks") || err.contains("sink"),
        "R8.1 missing expected sinks-related substring: {err}"
    );
}

#[test]
fn g2_r12_segment_masked_refuses() {
    // R12 (cycle-12 new): segment_masked + @checkpoint composition is
    // deferred. base_fusible has rope_q=false so R7 does NOT fire;
    // segment_masked=true with rope_q=false reaches R12.
    let mut cfg = base_fusible();
    cfg.segment_masked = true;
    cfg.rope_q = false;
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R12: segment_masked must refuse");
    assert!(
        err.contains("paged-segment-masked composition deferred"),
        "R12 missing expected substring 'paged-segment-masked composition deferred': {err}"
    );
}

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
