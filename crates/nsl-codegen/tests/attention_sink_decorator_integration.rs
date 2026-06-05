//! Sprint 2 cycle-4 paper §4.3 v0: attention sinks API surface.
//!
//! NO PTX assertion is made because v0 does NOT emit any sink-specific
//! PTX — the SMEM cache materialization is deferred to a future sprint.
//! This test pins decorator → config wiring only. When the SMEM
//! emission lands, EXTEND THIS TEST with a gated-PTX probe (e.g., a
//! sink-tile-specific register or comment) per the cycle-3 Sprint 1
//! review-fix pattern (commit 0a987a73 — params unconditional, register
//! loads gated).
//!
//! What this test pins:
//!
//!   1. The compile context's `FlashAttentionConfig::num_sink_tokens`
//!      is `4` (proof the new `@attention_sink` extraction-site arm in
//!      `kernel.rs` parsed the `tokens=4` arg and threaded it through
//!      the config constructor — pre-Sprint-2 cycle-4, this field did
//!      not exist; post-add, it would be hardcoded `0` at the three
//!      construction sites without this sprint's wiring).
//!
//! Together with the 5-rule semantic test
//! (`nsl-semantic/tests/attention_sink_decorator.rs`) and the
//! `FlashAttentionConfig::num_sink_tokens` field, this closes the
//! decorator → semantic → codegen API surface for paper §4.3.

#![cfg(feature = "test-helpers")]

use nsl_codegen::test_helpers::flash_attention_sink_context_for_source;

const ATTENTION_SINK_FIXTURE: &str = include_str!("fixtures/attention_sink_decorator.nsl");

#[test]
fn attention_sink_decorator_reaches_config_num_sink_tokens() {
    let (ctx_set, num_sink_tokens_flag) =
        flash_attention_sink_context_for_source(ATTENTION_SINK_FIXTURE);

    assert!(
        ctx_set,
        "fixture's @flash_attention must build a compile context — \
         if this fails the decorator scanner regressed independent of \
         the attention_sink wiring"
    );

    assert_eq!(
        num_sink_tokens_flag,
        Some(4),
        "Sprint 2 cycle-4 task A+B: @attention_sink(tokens=4) must \
         thread `4` into FlashAttentionConfig::num_sink_tokens (the \
         field defaults to 0 at the three construction sites in \
         compiler/kernel.rs; only the decorator-extraction arm flips it)"
    );
}
