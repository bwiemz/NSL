//! Sprint 2 cycle-4 paper §4.3 v0: attention sinks API surface.
//! Sprint 2 cycle-5: closes the silent-correctness gap that v0 left open.
//!
//! v0 (cycle-4) wired `@attention_sink(tokens=N)` decorator → config
//! field → semantic validation, but emitted NO sink-specific PTX. A
//! user writing `@attention_sink(tokens=4)` would have compiled
//! cleanly and run as if sinks were disabled — no error, no warning.
//! Cycle-5 closes the gap by refusing configs with `num_sink_tokens > 0`
//! at codegen with an error that names the deferred work. `tokens=0`
//! cannot be written (semantic rejects it); the "sinks disabled"
//! sentinel is to OMIT the decorator entirely.
//!
//! When the SMEM emission lands, lift the refusal in
//! `compiler/kernel.rs::attention_sink` and EXTEND the
//! `nonzero_tokens_refused_with_deferral_message` test with a gated
//! PTX probe (e.g., a sink-tile-specific register or comment) per the
//! cycle-3 Sprint 1 review-fix pattern (commit 0a987a73 — params
//! unconditional, register loads gated).
//!
//! What these tests pin:
//!
//!   1. `tokens_zero_sentinel_flows_through_cleanly`: omitting
//!      `@attention_sink` (the only way to express "sinks disabled"
//!      because semantic rejects `tokens=0`) leaves
//!      `num_sink_tokens=0` and compilation succeeds — the existing
//!      decorator-extraction loop continues to thread the default
//!      through the three `FlashAttentionConfig` construction sites
//!      unchanged.
//!
//!   2. `nonzero_tokens_refused_with_deferral_message`:
//!      `@attention_sink(tokens=4)` is REFUSED at codegen with the
//!      cycle-5 deferral error (substring assertion). Pre-Sprint-2
//!      cycle-5, this same fixture flowed through with
//!      `num_sink_tokens=4` — the silent gap. Once the SMEM emission
//!      sprint lands, this test must be REPLACED with a positive
//!      assertion that PTX contains a sink-tile artifact.

#![cfg(feature = "test-helpers")]

use nsl_codegen::test_helpers::{
    flash_attention_sink_context_for_source, try_flash_attention_sink_context_for_source,
};

const ATTENTION_SINK_FIXTURE: &str = include_str!("fixtures/attention_sink_decorator.nsl");
const ATTENTION_SINK_DISABLED_FIXTURE: &str =
    include_str!("fixtures/attention_sink_decorator_disabled.nsl");

#[test]
fn tokens_zero_sentinel_flows_through_cleanly() {
    // "Sinks disabled" sentinel: the decorator is OMITTED, so the
    // local-default `num_sink_tokens=0` flows through all three
    // FlashAttentionConfig construction sites in `kernel.rs`. This pins
    // that the cycle-5 codegen refusal does NOT regress the disabled
    // path — only `num_sink_tokens > 0` is refused.
    let (ctx_set, num_sink_tokens_flag) =
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
         value here would mean the codegen refusal added in Sprint 2 \
         cycle-5 has been bypassed or the kernel.rs default changed."
    );
}

#[test]
fn nonzero_tokens_refused_with_deferral_message() {
    // Sprint 2 cycle-5 silent-gap closure: `@attention_sink(tokens=4)`
    // (the canonical paper §4.3 value) must be REFUSED at codegen with
    // a clear error that names the deferred work. The error string is
    // load-bearing for the user — when the SMEM emission sprint lands,
    // this assertion will fail and the test must be replaced with a
    // positive PTX-probe.
    let result = try_flash_attention_sink_context_for_source(ATTENTION_SINK_FIXTURE);

    let err = match result {
        Err(e) => e,
        Ok(out) => panic!(
            "Sprint 2 cycle-5 task A: @attention_sink(tokens=4) MUST be \
             refused at codegen until SMEM emission lands. Compilation \
             unexpectedly succeeded with output {:?}. If this assertion \
             starts failing because SMEM emission landed, replace the \
             whole test with a positive PTX-contains-sink-tile assertion \
             per the cycle-3 Sprint 1 review-fix pattern.",
            out
        ),
    };

    // The substring is the load-bearing user-facing message. It must
    // specifically name (a) that the user's config was understood,
    // (b) that the SMEM cache emission is the deferred piece, and
    // (c) how to work around (omit decorator / set tokens=0). Match the
    // distinctive prefix that no other codegen error carries.
    let expected_substring =
        "@attention_sink(tokens=N) configured but SMEM cache emission is deferred";
    assert!(
        err.contains(expected_substring),
        "expected refusal error to contain {:?}, got {:?}",
        expected_substring,
        err
    );

    // Spot-check the deferral pointer + workaround language so a future
    // wording shuffle doesn't accidentally drop the user-actionable bits.
    assert!(
        err.contains("paper §4.3"),
        "refusal error should cite paper §4.3, got {:?}",
        err
    );
    assert!(
        err.contains("omit the decorator"),
        "refusal error should tell the user how to disable sinks \
         (omit the decorator), got {:?}",
        err
    );
}
