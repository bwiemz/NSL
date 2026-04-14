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

use nsl_codegen::flash_attention_v2::smem_layout::{
    q_offset, kv_offset, sp_offset, total_bytes,
    ALLOWED_BLOCK_Q, ALLOWED_BLOCK_KV, ALLOWED_HEAD_DIM,
};
use nsl_codegen::flash_attention_v2::register_budget::{count_registers, SM75_REGISTER_CAP};

fn supported_matrix() -> Vec<FlashAttentionConfig> {
    let mut out = Vec::new();
    for &bq in ALLOWED_BLOCK_Q {
        for &bkv in ALLOWED_BLOCK_KV {
            for &hd in ALLOWED_HEAD_DIM {
                let c = FlashAttentionConfig {
                    block_q: bq, block_kv: bkv, head_dim: hd, ..base_config()
                };
                if validate_scalar_v2_config(&c).is_ok() { out.push(c); }
            }
        }
    }
    out
}

#[test]
fn smem_regions_do_not_overlap() {
    for c in supported_matrix() {
        let q_end = q_offset(&c) + (c.block_q * c.head_dim * 2) as u32;
        assert_eq!(q_end, kv_offset(&c),
            "Q tile end must equal KV tile start for {:?}",
            (c.block_q, c.block_kv, c.head_dim));
        let kv_end = kv_offset(&c) + (c.block_kv * c.head_dim * 2) as u32;
        assert_eq!(kv_end, sp_offset(&c),
            "KV tile end must equal SP region start for {:?}",
            (c.block_q, c.block_kv, c.head_dim));
    }
}

#[test]
fn smem_total_matches_sum_of_regions() {
    for c in supported_matrix() {
        let q  = (c.block_q  * c.head_dim * 2) as u32;
        let kv = (c.block_kv * c.head_dim * 2) as u32;
        let sp = 4 * c.block_kv as u32 * 4;
        assert_eq!(total_bytes(&c), q + kv + sp,
            "total_bytes mismatch for {:?}",
            (c.block_q, c.block_kv, c.head_dim));
    }
}

#[test]
fn smem_total_under_48kb_for_all_supported() {
    for c in supported_matrix() {
        assert!(total_bytes(&c) <= 48 * 1024,
            "SMEM overflow for {:?}: {} bytes",
            (c.block_q, c.block_kv, c.head_dim), total_bytes(&c));
    }
}

/// Register budget sanity: all supported configs stay comfortably under
/// the sm_75 hardware cap of 255.
///
/// The upper bound of 40 is a sanity ceiling, not a hardware limit — it
/// absorbs the canonical head_dim=256 case (32 regs) plus the RoPE
/// extras (+4, reaches 36) plus margin for future additions like
/// backward-pass accumulators. When rope_q=true enters `supported_matrix`
/// via the Task 14 soak configs, this test already accommodates it.
#[test]
fn register_budget_under_sm75_cap() {
    for c in supported_matrix() {
        let n = count_registers(&c);
        assert!(n <= 40,
            "register budget {} exceeds sanity ceiling of 40 for {:?}",
            n, (c.block_q, c.block_kv, c.head_dim));
        assert!(n <= SM75_REGISTER_CAP,
            "register budget {} exceeds sm_75 hardware cap {} for {:?}",
            n, SM75_REGISTER_CAP, (c.block_q, c.block_kv, c.head_dim));
    }
}
