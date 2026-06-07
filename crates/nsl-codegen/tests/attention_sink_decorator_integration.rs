//! Sprint 2 cycle-4 paper §4.3 v0: attention sinks API surface.
//! Sprint 2 cycle-5: closed the silent-correctness gap left by v0 by
//! refusing all `num_sink_tokens > 0` configs unconditionally.
//! Sprint 1b cycle-7 (this commit): LIFTS the cycle-5 refusal for the
//! NARROW Tier A forward single-tile config and keeps an axis-specific
//! refusal for every ineligible variant.
//!
//! What these tests pin:
//!
//!   1. `tokens_zero_sentinel_flows_through_cleanly` (unchanged from
//!      cycle-5): omitting `@attention_sink` (the only way to express
//!      "sinks disabled" because semantic rejects `tokens=0`) leaves
//!      `num_sink_tokens=0` and compilation succeeds — the existing
//!      decorator-extraction loop continues to thread the default
//!      through the three `FlashAttentionConfig` construction sites
//!      unchanged.
//!
//!   2. `narrow_tier_a_forward_sinks_accepted`: the narrow config
//!      (`causal=false`, no rope/gqa/paged/segment_masked/tree_mask,
//!      no @train, no CSHA fusion) is ACCEPTED at codegen and the
//!      emitted PTX contains a gated sink-only emission probe
//!      (`sink_k_ptr` param declaration). This mirrors the cycle-3
//!      Sprint 1 review-fix pattern: probe a SINK-only emission so
//!      the test fails loudly if conditional emission regresses to
//!      always-on (which would break Sprint 1a's byte-identity guarantee
//!      at `num_sink_tokens=0`).
//!
//!   3. `causal_true_with_sinks_refused_naming_sprint2`,
//!      `rope_q_with_sinks_refused_naming_sprint4`,
//!      `paged_with_sinks_refused_naming_sprint6`: per-axis refusal —
//!      each blocked axis must name the future Sprint that lifts it.

#![cfg(feature = "test-helpers")]

use nsl_codegen::test_helpers::{
    flash_attention_sink_context_for_source, try_flash_attention_sink_context_for_source,
};

const ATTENTION_SINK_FIXTURE: &str = include_str!("fixtures/attention_sink_decorator.nsl");
const ATTENTION_SINK_DISABLED_FIXTURE: &str =
    include_str!("fixtures/attention_sink_decorator_disabled.nsl");
const ATTENTION_SINK_NARROW_FIXTURE: &str =
    include_str!("fixtures/attention_sink_decorator_narrow.nsl");

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
fn causal_true_with_sinks_refused_naming_sprint2() {
    // The cycle-4 fixture uses `@flash_attention` without `causal=false`,
    // so it inherits the default `causal=true`. With Sprint 1b's
    // eligibility predicate, causal=true is refused with a Sprint 2
    // citation (multi-tile + causal sink-bypass predicate).
    let result = try_flash_attention_sink_context_for_source(ATTENTION_SINK_FIXTURE);
    let err = match result {
        Err(e) => e,
        Ok(out) => panic!(
            "Sprint 1b cycle-7: causal=true + @attention_sink(tokens=4) \
             MUST be refused with a Sprint 2 citation. Compilation \
             unexpectedly succeeded with output {:?}.",
            out
        ),
    };

    assert!(
        err.contains("@attention_sink"),
        "refusal must echo the user's decorator: {:?}",
        err
    );
    assert!(
        err.contains("causal=true"),
        "refusal must name the failing axis (causal=true): {:?}",
        err
    );
    assert!(
        err.contains("Sprint 2"),
        "refusal must cite the future Sprint that lifts the constraint \
         (Sprint 2 — multi-tile + causal sink-bypass predicate): {:?}",
        err
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
