//! Isolation insta snapshot of the PCA Tier B skip predicate PTX.
//! Sentinel range_table_base = 0xDEAD_BEEF.
//! Per spec §5.1 + §3.2.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::pca_tilerange::{emit_skip_predicate, IterationOrder};

fn fa_4k_block64_seg_masked() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 64,
        causal: true,
        paged: false,
        rope_q: true,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 2,
        tree_mask: false,
        gpu_sm: 120,
        segment_masked: true,
        csha: None,
    }
}

// Snapshot is baselined for the production (non-instrumented) path. The
// `debug_kernel_instrumentation` feature appends the M3 round-robin
// writeback PTX inside the predicate scope (covered by the separate
// `pca_tier_b_writeback_isolation` snapshot), so this snapshot's content
// only matches when the feature is OFF. Gated via `cfg(not(feature = ...))`.
#[test]
#[cfg(not(feature = "debug_kernel_instrumentation"))]
fn predicate_4k_block64_isolation_snapshot() {
    let mut ptx = String::new();
    emit_skip_predicate(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "%qt",
        "%kvt",
        0xDEAD_BEEF,
        "KV_TILE_SKIP",
        IterationOrder::QOuter,
    );
    insta::assert_snapshot!("predicate_4k_block64", ptx);
}

#[test]
fn predicate_emits_four_range_table_loads() {
    let mut ptx = String::new();
    emit_skip_predicate(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "%qt",
        "%kvt",
        0xDEAD_BEEF,
        "KV_TILE_SKIP",
        IterationOrder::QOuter,
    );
    // 4 loads: qmin, qmax, kvmin, kvmax.
    assert_eq!(ptx.matches("ld.shared.u16").count(), 4);
}

#[test]
fn predicate_uses_disjoint_logic() {
    let mut ptx = String::new();
    emit_skip_predicate(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "%qt",
        "%kvt",
        0xDEAD_BEEF,
        "KV_TILE_SKIP",
        IterationOrder::QOuter,
    );
    assert!(ptx.contains("setp.lt.u16 %p_lt_TB, %qmax_TB, %kvmin_TB"));
    assert!(ptx.contains("setp.gt.u16 %p_gt_TB, %qmin_TB, %kvmax_TB"));
    assert!(ptx.contains("or.pred %p_skip_TB, %p_lt_TB, %p_gt_TB"));
}

#[test]
fn predicate_branches_to_provided_label() {
    let mut ptx = String::new();
    emit_skip_predicate(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "%qt",
        "%kvt",
        0xDEAD_BEEF,
        "MY_CUSTOM_LABEL",
        IterationOrder::QOuter,
    );
    assert!(ptx.contains("@%p_skip_TB bra MY_CUSTOM_LABEL"));
}

#[test]
fn predicate_sentinel_visible() {
    let mut ptx = String::new();
    emit_skip_predicate(
        &mut ptx,
        &fa_4k_block64_seg_masked(),
        4096,
        "%qt",
        "%kvt",
        0xDEAD_BEEF,
        "KV_TILE_SKIP",
        IterationOrder::QOuter,
    );
    // Sentinel offsets appear as immediate operands of `add.u64`.
    // qmin offset = 0xDEADBEEF (base + 0), shows verbatim.
    assert!(
        ptx.contains("DEADBEEF") || ptx.contains("deadbeef") || ptx.contains("3735928559"),
        "sentinel not visible:\n{ptx}"
    );
}
