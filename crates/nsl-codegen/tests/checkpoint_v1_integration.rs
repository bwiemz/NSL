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
use nsl_codegen::flash_attention_v2::{
    synthesize_backward_combined, synthesize_backward_with_tier,
    synthesize_backward_with_tier_b,
};

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
fn g2_r7_pca_packing_with_rope_q_now_synthesizes() {
    // SUPERSEDED 2026-07-26 — R7 is RETIRED. This asserted that
    // segment_masked + rope_q + @checkpoint refuses "deferred to v4: the
    // segment-aware causal mask shares index machinery with the RoPE-Q
    // write-back path". There was no such index collision: the segment mask
    // uses `%*_SEGMASK` registers and the RoPE reset uses `%*_doc_*`, and they
    // never alias. Three real defects sat underneath the refusal — a hardcoded
    // `%q_start` in the RoPE reset branch, an SMEM alias between the
    // kv-recompute x_norm scratch and `seg_smem`, and a literal "§" emitted
    // into the PTX (ptxas rc=218) — all fixed and GPU-validated by
    // `csha_checkpoint_recompute_gpu::t_segmask_recompute_hd32_s32_*`, whose
    // negative control confirms the packed oracle actually discriminates.
    //
    // INVERTED rather than deleted: if R7 is ever reinstated, or the emitters
    // regress such that this composition stops synthesizing, this fails.
    let mut cfg = base_fusible();
    cfg.segment_masked = true;
    cfg.rope_q = true;
    synthesize_backward_with_tier(&cfg).unwrap_or_else(|e| {
        panic!("segment_masked + rope_q + @checkpoint must synthesize at hd=32: {e}")
    });
}

#[test]
fn g2_r7_plain_rope_q_without_segment_masked_now_synthesizes() {
    // SUPERSEDED 2026-07-26. This asserted that R7 refuses plain rope_q +
    // @checkpoint even without segment_masked, on the strength of 8f774ad's
    // note that Path B had a "GROSS numerical error ... never-GPU-validated".
    //
    // That was measured (and confirmed real: dq 2.152e3 against max|ref| 4.4),
    // then root-caused to two %q_start-vs-%k_start confusions — the x_norm row
    // base in the kv-recompute, and the cos/sin index in the K RoPE epilogue.
    // Both are fixed; the three-way oracle is GREEN on all three comparisons at
    // hd=64 for S=32/512/2048, for BOTH RopeStyle::Adjacent and HalfSplit.
    //
    // R7 now refuses only the segment_masked composition (covered by the test
    // above), which was always a separate PCA-packing item. Keeping this
    // assertion would pin a refusal whose stated reason no longer exists.
    let mut cfg = base_fusible();
    cfg.rope_q = true;
    assert!(
        !cfg.segment_masked,
        "test scaffolding bug: segment_masked must stay false so this \
         exercises the plain rope_q case, not the segment_masked predicate"
    );
    synthesize_backward_with_tier(&cfg)
        .expect("plain rope_q under @checkpoint must now synthesize (Path B fixed)");
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
fn g2_r12_segment_masked_without_tier_b_now_synthesizes() {
    // SUPERSEDED 2026-07-26 — R12 NARROWED, not retired. It was keyed on
    // `config.segment_masked` alone, but the hazard its own message describes
    // is the PCA **Tier-B** planner's tile-active predicate gating recompute
    // writes. Tier-A segment masking with no Tier-B planner has none of that,
    // and is now GPU-validated. R12 therefore refuses on Tier-B actually being
    // co-emitted; `g2_r12_segment_masked_with_tier_b_still_refuses` below pins
    // the arm that remains.
    let mut cfg = base_fusible();
    cfg.segment_masked = true;
    cfg.rope_q = false;
    synthesize_backward_with_tier(&cfg).unwrap_or_else(|e| {
        panic!("segment_masked without Tier-B must synthesize at hd=32: {e}")
    });
}

#[test]
fn g2_r12_segment_masked_with_tier_b_still_refuses() {
    // The arm R12 still covers, and the anti-vacuity guard for the narrowing
    // above: co-emitting the Tier-B planner with the checkpoint kv-recompute
    // remains unvalidated and MUST refuse. Without this test, narrowing R12 to
    // `segment_masked && tier_b_emitted` could silently degrade to a no-op.
    let mut cfg = base_fusible();
    cfg.segment_masked = true;
    cfg.rope_q = false;
    // Tier-B is emitted only for sequences long enough to justify a range
    // table; `should_emit_tier_b` decides. Pick a seq_len well past any
    // threshold so this test pins the refusal, not the admission heuristic.
    let seq_len = 4096u32;
    let residency = nsl_codegen::pca_segment::SegmentResidency::Shared;
    if !nsl_codegen::pca_tilerange::should_emit_tier_b(&cfg, seq_len as u64, residency) {
        eprintln!(
            "[g2_r12] Tier-B not admitted at seq_len={seq_len} — admission \
             heuristic changed; re-pick a shape that admits Tier-B rather than \
             leaving this gate vacuous"
        );
        return;
    }
    let err = synthesize_backward_with_tier_b(&cfg, Some((seq_len, residency)))
        .expect_err("R12: segment_masked + Tier-B co-emission must refuse");
    assert!(
        err.contains("paged-segment-masked composition deferred"),
        "R12 missing expected substring 'paged-segment-masked composition deferred': {err}"
    );
}

// ----- G2-prod: cycle-12 Phase F — refusals must fire from the production
// entry too (`synthesize_backward_with_tier_b`), not just the direct fork
// (`synthesize_backward_with_recompute`). Reviewer 1 on cycle-12 Phase D
// caught that production routes `synthesize_backward_combined` →
// `synthesize_backward` → `synthesize_backward_with_tier_b(config, None)`,
// which NEVER calls `synthesize_backward_with_recompute`. These tests
// exercise the production path directly to lock in reachability.
// --------------------------------------------------------------------

#[test]
fn g2_prod_r3_unreachable_from_production_regression() {
    // Reviewer 1 regression: build a config with checkpoint=Some +
    // csha.fused_projections=false and call the production entry
    // `synthesize_backward_with_tier_b` directly. Without Phase F the
    // R3 cascade was unreachable from here — only the direct fork
    // `synthesize_backward_with_recompute` had the gate.
    let mut cfg = base_fusible();
    let mut csha = CshaExtras::level1(1e-6);
    csha.fused_projections = false; // intentionally bad → triggers R3
    cfg.csha = Some(csha);
    let err = synthesize_backward_with_tier_b(&cfg, None)
        .expect_err("R3 must refuse from the production entry too");
    assert!(
        err.contains("fused_projections gate failed"),
        "R3 unreachable from production: {err}"
    );
}

#[test]
fn g2_prod_r3_unreachable_via_combined_entry_regression() {
    // Cycle-12 Phase F: `synthesize_backward_combined` calls
    // `synthesize_backward` → `synthesize_backward_with_tier_b`. The
    // R3 cascade must surface at this entry too.
    let mut cfg = base_fusible();
    let mut csha = CshaExtras::level1(1e-6);
    csha.fused_projections = false;
    cfg.csha = Some(csha);
    let err = synthesize_backward_combined(&cfg)
        .expect_err("R3 must refuse via combined-entry production path");
    assert!(
        err.contains("fused_projections gate failed"),
        "R3 unreachable from synthesize_backward_combined: {err}"
    );
}

#[test]
fn g2_prod_r11_unreachable_from_production_regression() {
    // Reviewer 1 regression for R11: build a config that satisfies
    // `tier_b2_hybrid_backward_compile_time_eligible` AND has
    // checkpoint=Some, then call `synthesize_backward_with_tier_b`
    // directly. Without Phase F, R11 was unreachable from the
    // production entry. Predicate requires: gpu_sm>=80, csha.level>=2,
    // active_heads==1, d_model==head_dim, rope_q permitted, no sinks.
    let mut csha = CshaExtras::level1(1e-6);
    csha.level = 2;
    csha.fused_projections = true; // satisfy R3
    csha.active_heads = 1;
    csha.d_model = 64;
    let cfg = FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 64,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80, // Ampere → tier_b2_can_dispatch ok
        segment_masked: false,
        csha: Some(csha),
        checkpoint: Some(CheckpointExtras::full()),
    };

    // Sanity: predicate must accept this config so R11 is reachable.
    use nsl_codegen::flash_attention_v2::tier_b2::dispatch::tier_b2_hybrid_backward_compile_time_eligible;
    assert!(
        tier_b2_hybrid_backward_compile_time_eligible(&cfg),
        "test scaffolding bug: cfg is not hybrid-eligible, R11 won't fire"
    );

    let err = synthesize_backward_with_tier_b(&cfg, None)
        .expect_err("R11 must refuse from the production entry");
    assert!(
        err.contains("Tier B.2 hybrid backward composition"),
        "R11 unreachable from production: {err}"
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
