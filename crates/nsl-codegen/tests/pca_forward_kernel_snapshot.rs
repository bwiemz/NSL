//! Snapshot of a full forward FA2 kernel with segment_masked=true.
//!
//! Captures the wiring of the segment-mask helper into s_compute.rs and the
//! PCA prelude block (Task 3B of 4). This complements the unit-level snapshot
//! in pca_segment_mask_snapshot.rs by showing the INTEGRATED form.
//!
//! Diffing this snapshot against the equivalent segment_masked=false kernel
//! (fa_v2_snapshots__kernel_full__32x32x32_nocsha) shows exactly what PCA
//! Tier A contributed to the kernel text.
//!
//! Task 8 adds Tier B-enabled variants that call
//! `synthesize_flash_attention_ptx_v2_with_tier_b`. Diffing
//! `forward_kernel_segment_masked_tier_b_causal_32_32_32` against
//! `forward_kernel_segment_masked_causal_32_32_32` shows exactly what Tier B
//! contributed (preamble + skip predicate + KV_TILE_SKIP_TB label).

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::{
    synthesize_flash_attention_ptx_v2, synthesize_flash_attention_ptx_v2_with_tier_b,
};
use nsl_codegen::pca_segment::SegmentResidency;

fn minimal_segment_masked_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 32,
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
        checkpoint: None,
    }
}

/// A slightly larger config for the second Tier B variant (64x64x64 tiles).
fn segment_masked_config_64x64x64() -> FlashAttentionConfig {
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
        checkpoint: None,
    }
}

// ── Tier A baseline (no-op guarantee reference) ────────────────────────────

#[test]
fn forward_kernel_segment_masked_causal_32_32_32() {
    let ptx_bytes = synthesize_flash_attention_ptx_v2(&minimal_segment_masked_config());
    let ptx = String::from_utf8(ptx_bytes).expect("PTX must be valid UTF-8");
    insta::assert_snapshot!(ptx);
}

// ── Tier B-enabled variants (Task 8) ───────────────────────────────────────
//
// seq_len=4096, Shared residency: should_emit_tier_b returns true for these
// configs because the range table fits within the 8 KB budget.
// Diffing these against the Tier A baselines above isolates the Tier B delta:
//   - Range-table preamble block after bar.sync 0 in the forward prelude.
//   - emit_skip_predicate block at the top of each KV-tile (via s_compute).
//   - KV_TILE_SKIP_TB_{q_iter}: labels at the bottom of each KV-tile loop.

// Snapshots are baselined for the production (no-instrumentation) PTX.
// When `debug_kernel_instrumentation` is on the round-robin M3 writeback
// (B1.5-3) is appended inside the predicate scope and the snapshot
// content changes. The feature-on shape is covered separately by
// `pca_tier_b_writeback_isolation`. Gate these tests off when the
// feature is enabled to avoid spurious snapshot churn.
#[test]
#[cfg(not(feature = "debug_kernel_instrumentation"))]
fn forward_kernel_segment_masked_tier_b_causal_32_32_32() {
    let cfg = minimal_segment_masked_config();
    let seq_len: u32 = 4096;
    let residency = SegmentResidency::Shared;
    let ptx_bytes =
        synthesize_flash_attention_ptx_v2_with_tier_b(&cfg, Some((seq_len, residency)));
    let ptx = String::from_utf8(ptx_bytes).expect("PTX must be valid UTF-8");

    // Structural assertions: Tier B additions must be present.
    assert!(
        ptx.contains("PCA Tier B: range-table preamble"),
        "forward prelude must contain Tier B range-table preamble"
    );
    assert!(
        ptx.contains("KV_TILE_SKIP_TB_0:"),
        "KV_TILE_SKIP_TB_0 label must appear in Tier B kernel"
    );
    assert!(
        ptx.contains("PCA Tier B: skip predicate"),
        "s_compute must emit Tier B skip predicate"
    );

    insta::assert_snapshot!(ptx);
}

#[test]
#[cfg(not(feature = "debug_kernel_instrumentation"))]
fn forward_kernel_segment_masked_tier_b_causal_64_64_64() {
    let cfg = segment_masked_config_64x64x64();
    let seq_len: u32 = 4096;
    let residency = SegmentResidency::Shared;
    let ptx_bytes =
        synthesize_flash_attention_ptx_v2_with_tier_b(&cfg, Some((seq_len, residency)));
    let ptx = String::from_utf8(ptx_bytes).expect("PTX must be valid UTF-8");

    // Structural assertions: Tier B additions must be present.
    assert!(
        ptx.contains("PCA Tier B: range-table preamble"),
        "forward prelude must contain Tier B range-table preamble"
    );
    assert!(
        ptx.contains("KV_TILE_SKIP_TB_0:"),
        "KV_TILE_SKIP_TB_0 label must appear in Tier B kernel"
    );

    insta::assert_snapshot!(ptx);
}

// ── No-op guarantee regression tests ──────────────────────────────────────
//
// Passing tier_b=None must produce byte-identical output to the 1-arg form.

#[test]
fn tier_b_none_is_byte_identical_to_1arg() {
    let cfg = minimal_segment_masked_config();
    let via_1arg = synthesize_flash_attention_ptx_v2(&cfg);
    let via_none = synthesize_flash_attention_ptx_v2_with_tier_b(&cfg, None);
    assert_eq!(
        via_1arg, via_none,
        "synthesize_flash_attention_ptx_v2_with_tier_b(cfg, None) must be byte-identical \
         to synthesize_flash_attention_ptx_v2(cfg) — no-op guarantee violated"
    );
}
