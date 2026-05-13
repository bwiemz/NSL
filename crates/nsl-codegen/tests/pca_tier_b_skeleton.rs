//! Skeleton integration tests for PCA Tier B: lock the public API
//! signatures so refactors that change them break visibly.

use nsl_codegen::flash_attention::FlashAttentionConfig;
use nsl_codegen::pca_segment::SegmentResidency;
use nsl_codegen::pca_tilerange::{
    compute_range_table_bytes, emit_range_table_preamble, emit_skip_decision_writeback,
    emit_skip_predicate, should_emit_tier_b, TIER_B_RANGE_TABLE_BUDGET_BYTES,
};

#[test]
fn public_api_signatures_compile() {
    // Explicit type pins — if any signature drifts in arity OR types,
    // this file stops compiling.
    let _: fn(u64, u64, u64) -> u64 = compute_range_table_bytes;
    let _: u64 = TIER_B_RANGE_TABLE_BUDGET_BYTES;
    let _: fn(&mut String, &FlashAttentionConfig, u32, &str, u32) = emit_range_table_preamble;
    let _: fn(&mut String, &FlashAttentionConfig, u32, &str, &str, u32, &str) = emit_skip_predicate;
    let _: fn(&mut String, &FlashAttentionConfig, &str, &str, &str, &str) =
        emit_skip_decision_writeback;
    let _: fn(&FlashAttentionConfig, u64, SegmentResidency) -> bool = should_emit_tier_b;
}

#[test]
fn budget_constant_is_8kb() {
    assert_eq!(TIER_B_RANGE_TABLE_BUDGET_BYTES, 8192);
}
