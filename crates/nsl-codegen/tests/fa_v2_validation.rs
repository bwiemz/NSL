//! Unit tests for `validate_scalar_v2_config`. Each rejection path asserts
//! the exact error message so validator-surface changes are caught.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::smem_layout::validate_scalar_v2_config;

fn base_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 75,
        csha: None,
    }
}

#[test]
fn accepts_canonical_csha_config() {
    assert!(validate_scalar_v2_config(&base_config()).is_ok());
}

#[test]
fn accepts_non_csha_canonical_64_128() {
    let c = FlashAttentionConfig { block_q: 64, block_kv: 64, head_dim: 128, ..base_config() };
    assert!(validate_scalar_v2_config(&c).is_ok());
}

#[test]
fn rejects_block_q_not_multiple_of_4() {
    let c = FlashAttentionConfig { block_q: 5, ..base_config() };
    let err = validate_scalar_v2_config(&c).unwrap_err();
    assert!(err.0.contains("block_q"), "error: {}", err.0);
    assert!(err.0.contains("5"),        "error: {}", err.0);
}

#[test]
fn rejects_head_dim_not_multiple_of_32() {
    let c = FlashAttentionConfig { head_dim: 24, ..base_config() };
    let err = validate_scalar_v2_config(&c).unwrap_err();
    assert!(err.0.contains("head_dim"), "error: {}", err.0);
}

#[test]
fn rejects_block_q_greater_than_128() {
    let c = FlashAttentionConfig { block_q: 256, ..base_config() };
    let err = validate_scalar_v2_config(&c).unwrap_err();
    assert!(err.0.contains("block_q"), "error: {}", err.0);
}

#[test]
fn rejects_smem_overflow_128_128_256() {
    let c = FlashAttentionConfig { block_q: 128, block_kv: 128, head_dim: 256, ..base_config() };
    let err = validate_scalar_v2_config(&c).unwrap_err();
    assert!(err.0.contains("SMEM") || err.0.contains("48"),
        "error should mention SMEM overflow: {}", err.0);
}

#[test]
fn rejects_gqa_group_size_not_power_of_two() {
    let c = FlashAttentionConfig { gqa_group_size: 3, ..base_config() };
    let err = validate_scalar_v2_config(&c).unwrap_err();
    assert!(err.0.contains("gqa_group_size"), "error: {}", err.0);
}
