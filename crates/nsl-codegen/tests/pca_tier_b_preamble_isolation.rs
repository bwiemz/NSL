//! Isolation-level insta snapshot of the PCA Tier B preamble PTX.
//!
//! Sentinel base offset 0xDEAD_BEEF — appears verbatim in emitted PTX
//! as a register-loaded immediate or arithmetic operand; offset-consumption
//! regressions are immediately visible in the snapshot diff.
//!
//! Per spec §5.1 of 2026-05-12-pca-tier-b-revision-design.md.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::pca_tilerange::emit_range_table_preamble;

fn fa_4k_block64_seg_masked() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 64,
        causal: true, paged: false, rope_q: true,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 2,
        tree_mask: false, gpu_sm: 120, segment_masked: true, csha: None,
    }
}

#[test]
fn preamble_4k_block64_isolation_snapshot() {
    let mut ptx = String::new();
    emit_range_table_preamble(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "seg_smem",
        0xDEAD_BEEF,
    );
    insta::assert_snapshot!("preamble_4k_block64", ptx);
}

#[test]
fn preamble_sentinel_visible_in_emitted_ptx() {
    let mut ptx = String::new();
    emit_range_table_preamble(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "seg_smem",
        0xDEAD_BEEF,
    );
    // Sentinel must appear verbatim in the emitted PTX as decimal or hex.
    assert!(
        ptx.contains("DEADBEEF") || ptx.contains("deadbeef") || ptx.contains("3735928559"),
        "sentinel 0xDEADBEEF not visible in emitted PTX — sentinel detection broken:\n{ptx}"
    );
}

#[test]
fn preamble_uses_butterfly_shuffles() {
    let mut ptx = String::new();
    emit_range_table_preamble(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "seg_smem",
        0xDEAD_BEEF,
    );
    // 5-step butterfly per phase, 2 ops (min/max) per step, 2 phases = 20 shuffles.
    assert!(
        ptx.matches("shfl.sync.bfly").count() >= 20,
        "expected >=20 butterfly shuffles, got:\n{ptx}"
    );
}

#[test]
fn preamble_uses_predicated_lane_zero_store() {
    let mut ptx = String::new();
    emit_range_table_preamble(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "seg_smem",
        0xDEAD_BEEF,
    );
    // Predicated execution (`@%p_lane_zero_TILERANGE st.shared.u16`),
    // NOT a divergent branch around the store.
    assert!(
        ptx.contains("@%p_lane_zero_TILERANGE st.shared.u16"),
        "lane-0 store should be predicated, not branched. Emitted:\n{ptx}"
    );
    assert!(
        !ptx.contains("@%p_lane_zero_TILERANGE bra"),
        "lane-0 store must not use thread-divergent branch:\n{ptx}"
    );
}

#[test]
fn preamble_loop_counter_register_class_correct() {
    let mut ptx = String::new();
    emit_range_table_preamble(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "seg_smem",
        0xDEAD_BEEF,
    );
    // %r_tile_* must be .reg .u32 per warp-uniformity discipline.
    assert!(
        ptx.contains(".reg .u32") && ptx.contains("r_tile_q_TILERANGE") && ptx.contains("r_tile_kv_TILERANGE"),
        "r_tile must be .reg .u32; emitted preamble:\n{ptx}"
    );
    // Init via mov.u32 from literal 0 (uniformity-through-load anti-pattern check).
    assert!(
        ptx.contains("mov.u32 %r_tile_q_TILERANGE, 0"),
        "r_tile_q init missing — must be mov.u32 from literal:\n{ptx}"
    );
}
