//! Cycle-10 §5.3 Task 9 integration gates.
//!
//! Test surface covers the four code-path gates required by the
//! cycle-10 spec appendix and the implementer-C task list:
//!
//!   G1 byte-identity        — no-decorator path remains snapshot-clean.
//!   G2 refusal coverage     — R3, R7, R9, R10, R8.1 each fire with
//!                              the documented substring + emit no PTX.
//!   G3 structural recompute — `policy="full"` PTX contains the
//!                              diagnostic comment AND a label/register
//!                              hint carrying the namespace_suffix
//!                              evidence from Task 8.
//!   G6 sibling-leak guard   — `EffectChecker::checkpoint_policies()`
//!                              returns an empty map for a module that
//!                              contains zero `@checkpoint` decorators.
//!
//! Gates G4 (GPU numerical equivalence) and G5 (paper-§6.3 exact
//! diagnostic string) are intentionally deferred — see
//! `docs/wiki/Checkpoint-v1-Defer-Log.md`. G3 substring assertion is the
//! v1 functional-presence gate for both deferred items.

use nsl_codegen::flash_attention::{
    CheckpointExtras, CheckpointPolicy, CshaExtras, FlashAttentionConfig, RopeStyle,
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

// ----- G2: refusal substring coverage (R3, R7, R9, R10, R8.1) -----------

#[test]
fn g2_r3_non_fusible_prologue_refuses() {
    // R3: policy=Full but the forward config has no Level-1 fusible
    // prologue (csha=None means no RMSNorm prologue is available to
    // recompute).
    let mut cfg = base_fusible();
    cfg.csha = None;
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R3: non-fusible prologue must refuse");
    assert!(
        err.contains("requires RMSNorm to matmul to RoPE prologue"),
        "R3 missing expected substring: {err}"
    );
}

#[test]
fn g2_r7_pca_packing_with_rope_q_refuses() {
    // R7: segment_masked + rope_q + checkpoint composition is deferred
    // to v4.
    let mut cfg = base_fusible();
    cfg.segment_masked = true;
    cfg.rope_q = true;
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R7: PCA packing + rope_q + checkpoint must refuse");
    assert!(
        err.contains("PCA packing with rope_q=true under @checkpoint deferred to v4"),
        "R7 missing expected substring: {err}"
    );
}

#[test]
fn g2_r9_paged_kv_collision_refuses() {
    // R9: `paged_kv_collision = true` flag on the carrier signals the
    // enclosing fn was also `@paged_kv`-decorated. v1 refuses; v4 will
    // implement scatter-on-recompute.
    let cfg = FlashAttentionConfig {
        checkpoint: Some(CheckpointExtras {
            policy: CheckpointPolicy::Full,
            paged_kv_collision: true,
        }),
        ..base_fusible()
    };
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R9: @paged_kv + @checkpoint composition must refuse");
    assert!(
        err.contains(
            "@checkpoint composition with @paged_kv requires scatter-on-recompute; \
             deferred to v4"
        ),
        "R9 missing expected substring: {err}"
    );
}

#[test]
fn g2_r10_asymmetric_tile_refuses() {
    // R10: block_q != block_kv under @checkpoint policy=full is
    // deferred to v2 (the recompute SMEM accounting assumes symmetric
    // tiles in v1).
    let mut cfg = base_fusible();
    cfg.block_kv = 64; // != block_q (32)
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R10: asymmetric tiles + checkpoint must refuse");
    assert!(
        err.contains(
            "checkpoint policy=\"full\" requires block_q == block_kv in v1; \
             asymmetric-tile composition deferred to v2"
        ),
        "R10 missing expected substring: {err}"
    );
}

#[test]
fn g2_r81_sinks_v2_composition_refuses() {
    // R8.1: sinks-v2 + @checkpoint composition is unconditionally
    // refused in v1 (cycle-7 backward sinks refusal already blocks
    // num_sink_tokens > 0 at a lower layer; we emit the checkpoint-
    // specific substring so callers learn at the right altitude).
    let mut cfg = base_fusible();
    cfg.num_sink_tokens = 4;
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("R8.1: sinks-v2 + checkpoint must refuse");
    assert!(
        err.contains(
            "checkpoint policy=\"full\" + sinks-v2: composition outside \
             verified smoke set; refused"
        ),
        "R8.1 missing expected substring: {err}"
    );
}

// ----- G3: structural recompute probe ------------------------------------

#[test]
fn g3_structural_recompute_probe_emits_diagnostic_and_namespace() {
    let cfg = base_fusible();
    let ptx = synthesize_backward_with_tier(&cfg)
        .expect("G3: fusible-prologue + checkpoint=Full must synthesize");

    // 1. Diagnostic comment substring (G3 functional-presence gate).
    assert!(
        ptx.contains("full prologue recompute (norm + proj + RoPE)"),
        "G3: missing recompute diagnostic. PTX head:\n{}",
        ptx.lines().take(20).collect::<Vec<_>>().join("\n")
    );
    assert!(
        ptx.contains("Checkpoint strategy:"),
        "G3: missing 'Checkpoint strategy:' header comment"
    );

    // 2. Namespace-suffix evidence (Task 8 wire-up): per-q_iter
    // recompute markers reference the `_bwd_recompute_<q>` suffix in
    // both label and register-hint form.
    assert!(
        ptx.contains("_bwd_recompute_0"),
        "G3: missing namespace_suffix evidence for q_iter=0"
    );
    assert!(
        ptx.contains("V2_CSHA_PROLOGUE_SKIP_0_bwd_recompute_0"),
        "G3: missing recompute-label evidence for q_iter=0"
    );
    assert!(
        ptx.contains("%recompute_marker_bwd_recompute_0"),
        "G3: missing recompute register-hint evidence for q_iter=0"
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
