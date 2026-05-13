//! Unit tests for the Tier B SMEM layout API.
//! Spec §3.4.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::smem_layout::tier_b_range_table_offset;
use nsl_codegen::flash_attention_v2::{shared_mem_bytes_v2, shared_mem_bytes_v2_backward};
use nsl_codegen::pca_tile_config::num_tiles;
use nsl_codegen::pca_tilerange::{range_table_addrs, should_emit_tier_b};
use nsl_codegen::pca_segment::SegmentResidency;

fn fa_base_seg_masked() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 64,
        causal: true, paged: false, rope_q: true,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 2,
        tree_mask: false, gpu_sm: 120, segment_masked: true, csha: None,
    }
}

#[test]
fn tier_b_offset_is_two_byte_aligned() {
    let cfg = fa_base_seg_masked();
    let offset = tier_b_range_table_offset(&cfg);
    assert_eq!(offset % 2, 0, "offset {offset} not 2-byte aligned");
}

#[test]
fn range_table_addrs_monotonic_offsets() {
    let addrs = range_table_addrs(0, 64, 64);
    assert_eq!(addrs.qtile_min,  0);
    assert_eq!(addrs.qtile_max,  64 * 2);
    assert_eq!(addrs.kvtile_min, 2 * 64 * 2);
    assert_eq!(addrs.kvtile_max, (2 * 64 + 64) * 2);
}

#[test]
fn range_table_addrs_asymmetric_blocks() {
    let addrs = range_table_addrs(0x1000, 512, 256);
    assert_eq!(addrs.qtile_min,  0x1000);
    assert_eq!(addrs.qtile_max,  0x1000 + 512 * 2);
    assert_eq!(addrs.kvtile_min, 0x1000 + 2 * 512 * 2);
    assert_eq!(addrs.kvtile_max, 0x1000 + 2 * 512 * 2 + 256 * 2);
}

#[test]
fn range_table_addrs_preserves_base() {
    let addrs = range_table_addrs(0xDEADBEE0, 32, 48);
    assert_eq!(addrs.qtile_min, 0xDEADBEE0);
}

#[test]
fn shared_mem_bytes_v2_no_op_when_not_tier_b() {
    let mut cfg = fa_base_seg_masked();
    cfg.segment_masked = false;
    assert!(!should_emit_tier_b(&cfg, 4096, SegmentResidency::Shared));
    assert!(shared_mem_bytes_v2(&cfg) > 0);
}

#[test]
fn tier_b_bytes_match_formula() {
    // 4K seq, block=64 → 64 q-tiles + 64 kv-tiles → 2*(64+64)*2 = 512 B.
    let cfg = fa_base_seg_masked();
    let bytes = nsl_codegen::pca_tilerange::tier_b_range_table_bytes(&cfg, 4096);
    assert_eq!(bytes, 512);
    let _ = num_tiles(4096, 64);
}
