//! Sprint 2 cycle-4 paper §4.3 v0: attention sinks API surface.
//! Sprint 2 cycle-5: closed the silent-correctness gap left by v0 by
//! refusing all `num_sink_tokens > 0` configs unconditionally.
//! Sprint 1b cycle-7: LIFTS the cycle-5 refusal for the NARROW Tier A
//! forward single-tile config and keeps an axis-specific refusal for
//! every ineligible variant.
//! Sprint 2 cycle-8 (this commit): LIFTS the causal=true AND multi-tile
//! (block_q != block_kv) axes simultaneously — splitting would ship
//! dead PTX (causal-only without kv_iter==0 gate corrupts multi-tile;
//! multi-tile-only without OR-bypass refuses sink rows at s_compute).
//!
//! What these tests pin:
//!
//!   1. `tokens_zero_sentinel_flows_through_cleanly` (unchanged from
//!      cycle-5): omitting `@attention_sink` (the only way to express
//!      "sinks disabled" because semantic rejects `tokens=0`) leaves
//!      `num_sink_tokens=0` and compilation succeeds.
//!
//!   2. `narrow_tier_a_forward_sinks_accepted`: the narrow config
//!      (`causal=false`, no rope/gqa/paged/segment_masked/tree_mask,
//!      no @train, no CSHA fusion) is ACCEPTED at codegen and the
//!      emitted PTX contains a gated sink-only emission probe
//!      (`sink_k_ptr` param declaration).
//!
//!   3. `causal_true_with_sinks_accepted_after_sprint2` (Sprint 2
//!      cycle-8): causal=true + sinks now ACCEPTED with sink_k_ptr.
//!
//!   4. `bq_ne_bkv_with_sinks_accepted_after_sprint2` (Sprint 2 cycle-8):
//!      multi-tile via `@autotune(block_q=[64], block_kv=[128])` +
//!      sinks now ACCEPTED.
//!
//!   5. `causal_mt_combined_with_sinks_accepted_after_sprint2` (Sprint
//!      2 cycle-8): BOTH causal=true AND multi-tile simultaneously
//!      ACCEPTED — the canonical combined cell.
//!
//!   6. `ptx_emits_kv_iter_zero_gate_when_multi_tile_and_sinks` /
//!      `ptx_emits_sink_bypass_or_pred_when_causal_and_sinks` (Sprint
//!      2 cycle-8): cycle-3 false-positive-trap pattern — probe gated
//!      PTX REGISTERS (`%p_skip_sinks`, `%p_sink`) NOT comments.
//!
//!   7. `rope_q_with_sinks_refused_naming_sprint4`,
//!      `paged_with_sinks_refused_naming_sprint6`: still-deferred axes
//!      must keep refusing with a Sprint citation.

#![cfg(feature = "test-helpers")]

use nsl_codegen::test_helpers::{
    flash_attention_sink_context_for_source, try_flash_attention_sink_context_for_source,
    try_flash_attention_v2_forward_ptx_for_source,
};

const ATTENTION_SINK_FIXTURE: &str = include_str!("fixtures/attention_sink_decorator.nsl");
const ATTENTION_SINK_DISABLED_FIXTURE: &str =
    include_str!("fixtures/attention_sink_decorator_disabled.nsl");
const ATTENTION_SINK_NARROW_FIXTURE: &str =
    include_str!("fixtures/attention_sink_decorator_narrow.nsl");
const ATTENTION_SINK_MULTI_TILE_FIXTURE: &str =
    include_str!("fixtures/attention_sink_decorator_multi_tile.nsl");
const ATTENTION_SINK_CAUSAL_MT_FIXTURE: &str =
    include_str!("fixtures/attention_sink_decorator_causal_mt.nsl");

#[test]
fn tokens_zero_sentinel_flows_through_cleanly() {
    // "Sinks disabled" sentinel: the decorator is OMITTED, so the
    // local-default `num_sink_tokens=0` flows through all three
    // FlashAttentionConfig construction sites in `kernel.rs`. This pins
    // that the Sprint 1b cycle-7 eligibility check does NOT regress the
    // disabled path — `num_sink_tokens == 0` short-circuits eligibility
    // to (true, None) regardless of other axes.
    let (ctx_set, num_sink_tokens_flag, ptx_has_sink_k_ptr) =
        flash_attention_sink_context_for_source(ATTENTION_SINK_DISABLED_FIXTURE);

    assert!(
        ctx_set,
        "fixture's @flash_attention must build a compile context — \
         if this fails the decorator scanner regressed independent of \
         the attention_sink wiring"
    );

    assert_eq!(
        num_sink_tokens_flag,
        Some(0),
        "the sentinel 'sinks disabled' path (no @attention_sink decorator) \
         must leave num_sink_tokens at the local-default 0. A non-zero \
         value here would mean the cycle-7 eligibility-check assignment \
         in kernel.rs regressed."
    );

    // Sprint 1b cycle-7 holistic-review fix: pin the SPRINT 1A
    // BYTE-IDENTITY invariant at the integration-test layer. When
    // `num_sink_tokens == 0`, the kernel signature MUST NOT declare
    // `sink_k_ptr` — that param is gated at `prelude.rs:156-159` and
    // its presence at zero-sinks would break the fa_v2_snapshots suite
    // and the Sprint 1a refactor's correctness proof. Probing here
    // gives a SECOND line of defense in case a future implementer
    // accidentally unconditional-emits the param.
    assert!(
        !ptx_has_sink_k_ptr,
        "Sprint 1a byte-identity invariant: at num_sink_tokens=0 the \
         emitted PTX MUST NOT declare `sink_k_ptr` — gated emission \
         lives at prelude.rs:156-159. If this fires, an unconditional \
         declaration regression has broken the fa_v2_snapshots suite."
    );
}

#[test]
fn narrow_tier_a_forward_sinks_accepted() {
    // Sprint 1b cycle-7: the narrow config (causal=false, no rope/gqa/
    // paged/segment_masked/tree_mask, no @train, no CSHA fusion) is
    // ACCEPTED at codegen. The compile context's num_sink_tokens flag
    // round-trips the decorator's tokens=4. Eligibility short-circuited
    // for tokens=0 disabled, here it returns (true, None) for tokens=4.
    let result = try_flash_attention_sink_context_for_source(ATTENTION_SINK_NARROW_FIXTURE);
    let (ctx_set, num_sink_tokens_flag, ptx_has_sink_k_ptr) = match result {
        Ok(t) => t,
        Err(e) => panic!(
            "Sprint 1b cycle-7: narrow @attention_sink(tokens=4) MUST \
             compile cleanly under the lifted refusal. Compilation failed \
             with {:?}. If this assertion starts failing because the \
             refusal regressed, restore the cycle-7 eligibility-check \
             wiring in kernel.rs.",
            e
        ),
    };

    assert!(
        ctx_set,
        "narrow fixture's @flash_attention must build a compile context"
    );
    assert_eq!(
        num_sink_tokens_flag,
        Some(4),
        "narrow @attention_sink(tokens=4) must round-trip the decorator's \
         value into the FlashAttentionConfig. A 0 here means the \
         decorator-extraction arm regressed (Sprint 1b restored the \
         `num_sink_tokens = N` assignment that cycle-5 removed)."
    );

    // Sprint 1b cycle-7 holistic-review fix: deliver the PTX probe the
    // test's docstring promises. Mirrors the cycle-3 review-fix pattern
    // in commit `0a987a73` — probe a GATED emission (param declared
    // ONLY when num_sink_tokens > 0 per prelude.rs:156-159), NOT an
    // unconditional structure. The probe is the proof that the Sprint 1b
    // codegen path actually fired, not just that the config value
    // round-tripped.
    assert!(
        ptx_has_sink_k_ptr,
        "Sprint 1b cycle-7: narrow accepted config MUST emit the \
         `sink_k_ptr` param declaration (prelude.rs:156-159 gates on \
         num_sink_tokens > 0). If this fires, the eligibility check let \
         the config through to codegen but the prelude gate did not fire \
         — a soft-correctness regression where the user thinks sinks are \
         on but the kernel emits no sink-loading PTX."
    );
}

#[test]
fn causal_true_with_sinks_accepted_after_sprint2() {
    // Sprint 2 cycle-8: the cycle-4 fixture uses default `@flash_attention`
    // (causal=true) + `@attention_sink(tokens=4)`. The cycle-7 Sprint 1b
    // refusal "causal=true (deferred to Sprint 2)" is now LIFTED — the
    // s_compute pass emits a gated OR-extension to the causal mask
    // predicate via `%p_sink` so sink rows always attend regardless of
    // query position (paper §4.3). The integration helper round-trips
    // tokens=4 into the FlashAttentionConfig and the synthesized PTX
    // declares `sink_k_ptr` (the cycle-7 Sprint 1b probe).
    let (ctx_set, num_sink_tokens_flag, ptx_has_sink_k_ptr) =
        match try_flash_attention_sink_context_for_source(ATTENTION_SINK_FIXTURE) {
            Ok(t) => t,
            Err(e) => panic!(
                "Sprint 2 cycle-8: causal=true + @attention_sink(tokens=4) \
                 MUST compile cleanly under the Sprint 2 lift. Compilation \
                 failed with {:?}. If the predicate refuses causal=true here \
                 the Sprint 2 lift in sinks.rs regressed.",
                e
            ),
        };

    assert!(
        ctx_set,
        "causal+sinks fixture's @flash_attention must build a compile context"
    );
    assert_eq!(
        num_sink_tokens_flag,
        Some(4),
        "causal+sinks fixture must round-trip @attention_sink(tokens=4)"
    );
    assert!(
        ptx_has_sink_k_ptr,
        "Sprint 2 cycle-8: causal=true + sinks accepted config MUST emit \
         the gated `sink_k_ptr` param declaration (prelude.rs:156-159 gates \
         on num_sink_tokens > 0). Soft-correctness regression if missing."
    );
}

#[test]
fn bq_ne_bkv_with_sinks_accepted_after_sprint2() {
    // Sprint 2 cycle-8: `@autotune(block_q=[64], block_kv=[128])` flows
    // through the autotune middle-value selector into `context.config`
    // with `block_q=64, block_kv=128` — multi-tile (the KV loop runs
    // multiple iterations across a sequence-aligned launch). The
    // cycle-7 Sprint 1b refusal "block_q != block_kv (deferred to
    // Sprint 2)" is now LIFTED. The K/V tile load gates the sink
    // pre-load on `%k_start == 0` so the sinks load once at the first
    // KV iter and persist across subsequent ones.
    let (ctx_set, num_sink_tokens_flag, ptx_has_sink_k_ptr) =
        match try_flash_attention_sink_context_for_source(ATTENTION_SINK_MULTI_TILE_FIXTURE) {
            Ok(t) => t,
            Err(e) => panic!(
                "Sprint 2 cycle-8: multi-tile (block_q != block_kv) + \
                 @attention_sink(tokens=4) MUST compile cleanly. \
                 Compilation failed with {:?}.",
                e
            ),
        };

    assert!(ctx_set, "multi-tile fixture must build a compile context");
    assert_eq!(
        num_sink_tokens_flag,
        Some(4),
        "multi-tile fixture must round-trip @attention_sink(tokens=4)"
    );
    assert!(
        ptx_has_sink_k_ptr,
        "Sprint 2 cycle-8: multi-tile accepted config MUST emit gated \
         `sink_k_ptr` (prelude.rs:156-159)."
    );
}

#[test]
fn causal_mt_combined_with_sinks_accepted_after_sprint2() {
    // Sprint 2 cycle-8: BOTH causal=true AND multi-tile (block_q != block_kv)
    // simultaneously, plus @attention_sink(tokens=4) — the canonical
    // combined-cell sanity. The 2-axis lift MUST land together: splitting
    // ships dead PTX (see sinks.rs Sprint 2 docstring).
    let (ctx_set, num_sink_tokens_flag, ptx_has_sink_k_ptr) =
        match try_flash_attention_sink_context_for_source(ATTENTION_SINK_CAUSAL_MT_FIXTURE) {
            Ok(t) => t,
            Err(e) => panic!(
                "Sprint 2 cycle-8: causal=true + multi-tile + \
                 @attention_sink(tokens=4) MUST compile cleanly. \
                 Compilation failed with {:?}.",
                e
            ),
        };

    assert!(
        ctx_set,
        "combined causal+multi-tile fixture must build a compile context"
    );
    assert_eq!(
        num_sink_tokens_flag,
        Some(4),
        "combined fixture must round-trip @attention_sink(tokens=4)"
    );
    assert!(
        ptx_has_sink_k_ptr,
        "Sprint 2 cycle-8: combined accepted config MUST emit gated \
         `sink_k_ptr` (prelude.rs:156-159)."
    );
}

#[test]
fn ptx_emits_kv_iter_zero_gate_when_multi_tile_and_sinks() {
    // Sprint 2 cycle-8 cycle-3 false-positive trap: probe the GATED PTX
    // REGISTER `%p_skip_sinks` (load-bearing — declared in the forward
    // prelude only when `num_sink_tokens > 0` and consumed by the K/V
    // sink pre-load setp+bra at mod.rs::emit_{k,v}_tile_load) NOT a
    // descriptive comment. The presence proves the kv_iter==0 gating
    // PTX actually emits — without this the sink slab would reload
    // every KV iteration and clobber itself (the multi-tile correctness
    // failure mode the gate prevents).
    let (num_sink_tokens, ptx) =
        try_flash_attention_v2_forward_ptx_for_source(ATTENTION_SINK_MULTI_TILE_FIXTURE)
            .expect("multi-tile fixture must compile cleanly after Sprint 2 cycle-8");
    assert_eq!(num_sink_tokens, 4);
    assert!(
        ptx.contains("%p_skip_sinks"),
        "Sprint 2 cycle-8: multi-tile + sinks PTX MUST declare/use the \
         `%p_skip_sinks` predicate register (forward prelude gates the \
         declaration on num_sink_tokens > 0; mod.rs::emit_{{k,v}}_tile_load \
         emits the setp.ne.u64 %p_skip_sinks, %k_start, 0 + \
         @%p_skip_sinks bra V2_*_SINK_LOAD_SKIP_* gate). If absent, the \
         kv_iter==0 first-load semantics are broken."
    );
}

#[test]
fn ptx_emits_sink_bypass_or_pred_when_causal_and_sinks() {
    // Sprint 2 cycle-8 cycle-3 false-positive trap: probe the GATED PTX
    // REGISTER `%p_sink` (load-bearing — declared in the forward prelude
    // only when `num_sink_tokens > 0` and consumed by both the K/V sink
    // pre-load loops AND now by the s_compute causal sink-bypass
    // `setp.ge.u64 %p_sink, %rd34, %rd_num_sink_tokens_reg; and.pred
    // %p0, %p0, %p_sink;` pair). For the causal-only fixture the
    // s_compute use site is the bypass — without it sink rows get
    // masked out by the causal predicate, breaking the paper §4.3
    // "sinks attend regardless of query position" semantics.
    let (num_sink_tokens, ptx) =
        try_flash_attention_v2_forward_ptx_for_source(ATTENTION_SINK_FIXTURE)
            .expect("causal+sinks fixture must compile cleanly after Sprint 2 cycle-8");
    assert_eq!(num_sink_tokens, 4);
    assert!(
        ptx.contains("%p_sink"),
        "Sprint 2 cycle-8: causal=true + sinks PTX MUST declare/use the \
         `%p_sink` predicate register. The s_compute causal-bypass \
         emission lives at phases/forward/s_compute.rs and gates on \
         (config.causal && config.num_sink_tokens > 0). If absent, sink \
         rows get masked out by the causal predicate at this fixture's \
         block_q=64 single-tile config."
    );
    // Also verify the load-bearing %rd_num_sink_tokens_reg appears — the
    // u64 register holding the num_sink_tokens compile-time literal that
    // s_compute compares k_global against. Without it the OR-bypass
    // can't fire because there's nothing to compare to.
    assert!(
        ptx.contains("%rd_num_sink_tokens_reg"),
        "Sprint 2 cycle-8: causal+sinks PTX MUST declare \
         `%rd_num_sink_tokens_reg` (the u64 holding the compile-time \
         num_sink_tokens literal used by s_compute's setp.ge.u64 bypass \
         comparison). If absent, the prelude initialization regressed."
    );
}

#[test]
fn rope_q_with_sinks_refused_naming_sprint4() {
    // Build the fixture inline so the test does not depend on a separate
    // .nsl file — pure substring assertion on the error message.
    let src = "\
@flash_attention(causal=false)
@rope
@attention_sink(tokens=4)
fn forward():
    pass
";
    let result = try_flash_attention_sink_context_for_source(src);
    let err = match result {
        Err(e) => e,
        Ok(out) => panic!(
            "rope_q=true + sinks MUST refuse. Got {:?}",
            out
        ),
    };
    assert!(
        err.contains("rope_q=true"),
        "refusal must name rope_q=true: {:?}",
        err
    );
    assert!(
        err.contains("Sprint 4"),
        "refusal must cite Sprint 4 (StreamingLLM no-rotation-for-sinks \
         policy): {:?}",
        err
    );
}

#[test]
fn paged_with_sinks_refused_naming_sprint6() {
    let src = "\
@flash_attention(causal=false)
@paged_kv
@attention_sink(tokens=4)
fn forward():
    pass
";
    let result = try_flash_attention_sink_context_for_source(src);
    let err = match result {
        Err(e) => e,
        Ok(out) => panic!(
            "paged=true + sinks MUST refuse. Got {:?}",
            out
        ),
    };
    assert!(
        err.contains("paged=true"),
        "refusal must name paged=true: {:?}",
        err
    );
    assert!(
        err.contains("Sprint 6"),
        "refusal must cite Sprint 6 (paged + sinks design pending): {:?}",
        err
    );
}
