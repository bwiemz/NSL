//! Codegen emission-helper tests. Per planner spec §8.6 (test surface #7).
//!
//! Verifies `emit_tier_b_variants_for_config` returns:
//!   - both base PTX + Tier-B-on PTX for segment_masked configs
//!   - only base PTX (Tier-B-on = None) for non-segment_masked configs
//!   - kernel-name distinctness via the `_tier_b_max<N>` suffix
//!
//! The compile-time const assertion on TIER_B_MAX_BAKED_SEQ_LEN is verified at
//! build time by the Rust compiler (see nsl-runtime/src/pca_tier_b_runtime.rs);
//! no runtime test is needed for that.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::pca_tier_b::{
    emit_tier_b_variants_for_config, flash_attention_kernel_name_v2_tier_b_on,
    should_emit_tier_b_at_codegen, TIER_B_MAX_BAKED_SEQ_LEN,
};

/// Build a FlashAttentionConfig with `segment_masked = true`.
/// Mirrors the construction pattern from P-3a's inline test
/// (`crates/nsl-codegen/src/pca_tier_b.rs::tests::segment_masked_cfg`).
fn segment_masked_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
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
        gpu_sm: 80,
        segment_masked: true,
        csha: None,
    }
}

#[test]
fn emission_helper_returns_two_blobs_for_segment_masked() {
    let cfg = segment_masked_config();
    let result = emit_tier_b_variants_for_config(&cfg);
    assert!(!result.base_ptx.is_empty(), "base PTX must be non-empty");
    assert!(
        result.tier_b_on_ptx.is_some(),
        "Tier-B-on PTX must be emitted for segment_masked"
    );
    assert!(
        result.tier_b_on_kernel_name.is_some(),
        "Tier-B-on name must be emitted"
    );
    let on_ptx = result.tier_b_on_ptx.as_ref().unwrap();
    assert!(!on_ptx.is_empty(), "Tier-B-on PTX must be non-empty");
}

#[test]
fn emission_helper_returns_one_blob_for_non_segment_masked() {
    let mut cfg = segment_masked_config();
    cfg.segment_masked = false;
    let result = emit_tier_b_variants_for_config(&cfg);
    assert!(!result.base_ptx.is_empty());
    assert!(
        result.tier_b_on_ptx.is_none(),
        "Tier-B-on must NOT be emitted for non-segment_masked"
    );
    assert!(result.tier_b_on_kernel_name.is_none());
}

#[test]
fn kernel_name_distinctness_with_max_baked_suffix() {
    let cfg = segment_masked_config();
    assert!(
        should_emit_tier_b_at_codegen(&cfg),
        "heuristic must fire for segment_masked"
    );
    let base_name = nsl_codegen::flash_attention_v2::flash_attention_kernel_name_v2(&cfg);
    let on_name = flash_attention_kernel_name_v2_tier_b_on(&cfg);
    assert_ne!(base_name, on_name, "Tier-B-on must differ from base");
    let expected_suffix = format!("_tier_b_max{}", TIER_B_MAX_BAKED_SEQ_LEN);
    assert!(
        on_name.ends_with(&expected_suffix),
        "Tier-B-on kernel name must end with {expected_suffix:?}; got {on_name:?}"
    );
}
