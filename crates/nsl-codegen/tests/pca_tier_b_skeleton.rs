//! Skeleton integration tests for PCA Tier B: lock the public API
//! signatures so refactors that change them break visibly.

use nsl_codegen::flash_attention::FlashAttentionConfig;
use nsl_codegen::pca_segment::SegmentResidency;
use nsl_codegen::pca_tilerange::{
    compute_range_table_bytes, emit_range_table_preamble, emit_skip_decision_writeback,
    emit_skip_predicate, should_emit_tier_b, IterationOrder, TIER_B_RANGE_TABLE_BUDGET_BYTES,
};

#[test]
fn public_api_signatures_compile() {
    // Explicit type pins — if any signature drifts in arity OR types,
    // this file stops compiling.
    let _: fn(u64, u64, u64) -> u64 = compute_range_table_bytes;
    let _: u64 = TIER_B_RANGE_TABLE_BUDGET_BYTES;
    let _: fn(&mut String, &FlashAttentionConfig, u32, &str, u32) = emit_range_table_preamble;
    // B.2: emit_skip_predicate gained an `IterationOrder` parameter.
    let _: fn(&mut String, &FlashAttentionConfig, u32, &str, &str, u32, &str, IterationOrder) =
        emit_skip_predicate;
    let _: fn(&mut String, &FlashAttentionConfig, u32, &str, &str, &str, &str, u32) =
        emit_skip_decision_writeback;
    let _: fn(&FlashAttentionConfig, u64, SegmentResidency) -> bool = should_emit_tier_b;
}

#[test]
fn budget_constant_is_8kb() {
    assert_eq!(TIER_B_RANGE_TABLE_BUDGET_BYTES, 8192);
}

#[test]
#[cfg(feature = "debug_kernel_instrumentation")]
fn skip_writeback_emits_when_feature_enabled() {
    use nsl_codegen::flash_attention::RopeStyle;

    let cfg = FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 64,
        causal: true, paged: false, rope_q: true,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 2,
        tree_mask: false, gpu_sm: 120, segment_masked: true, csha: None,
    };
    let mut ptx = String::new();
    emit_skip_decision_writeback(
        &mut ptx, &cfg, 4096,
        "%qt", "%kvt", "%p_skip_TB", "skip_decisions_ptr",
        /* num_warps = */ 4,
    );
    assert!(ptx.contains("st.global.u8"), "writeback should emit st.global.u8");
    assert!(ptx.contains("@%p_writeback_TB"), "writeback should be owner-gated via round-robin predicate");
    assert!(ptx.contains("selp.u16 %dec_val_TB, 1, 0"), "selp decision encoding missing");
}
