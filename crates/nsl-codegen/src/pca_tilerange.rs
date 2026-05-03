//! PCA Tier B — runtime per-tile segment-range tile-skip predicate.
//!
//! Builds a runtime range table from segment_ids in SMEM, then emits a
//! warp-uniform branch that skips QK^T matmul + softmax update + PV
//! reduction for `(q_tile, kv_tile)` pairs whose segment ranges are
//! strictly disjoint. Composes with Tier A's per-element segment_mask:
//! Tier B filters tile pairs; Tier A masks cells inside surviving tiles.
//!
//! Spec: `docs/superpowers/specs/2026-05-02-pca-tier-b-tile-skip-design.md`.
//!
//! # Emitter functions
//!
//! - [`emit_range_table_preamble`] — once per kernel, in the preamble.
//! - [`emit_skip_predicate`] — once per `(q_iter, kv_iter)` boundary.
//! - [`emit_skip_decision_writeback`] — gated debug-only; M3 parity.
//!
//! # Scratch register convention
//!
//! All scratch registers introduced by this helper use the suffix
//! `_TILERANGE` (e.g. `%rs_qmin_TILERANGE`, `%p_skip_TILERANGE`) so
//! they don't collide with Tier A's `_SEGMASK` suffix or the FA2
//! emitter's seeded scratch space.

use crate::flash_attention::FlashAttentionConfig;
use crate::pca_segment::SegmentResidency;

/// Compute the SMEM bytes the range table consumes for a given
/// (block_q, block_kv, seq_len) configuration.
///
/// Formula per spec §3.3:
///
/// ```text
///   bytes = 2 × (num_q_tiles + num_kv_tiles) × sizeof(seg_id_t)
/// ```
///
/// where `seg_id_t = u16` and tile counts are
/// `ceil(seq_len / block_*)`. Both `min` and `max` tables exist per
/// dimension; the `2 ×` covers that.
pub fn compute_range_table_bytes(seq_len: u64, block_q: u64, block_kv: u64) -> u64 {
    let num_q_tiles = seq_len.div_ceil(block_q);
    let num_kv_tiles = seq_len.div_ceil(block_kv);
    2 * (num_q_tiles + num_kv_tiles) * 2 // 2 bytes per u16
}

/// SMEM size threshold above which Tier B falls back to Tier A only
/// (spec §3.3 v1 bound).
pub const TIER_B_RANGE_TABLE_BUDGET_BYTES: u64 = 8192;

/// Decide whether to emit Tier B for a given FA config. Returns false
/// when the range table would exceed the v1 budget OR when Tier A's
/// segment-ID residency is `Streamed` (spec §3.4 — Tier B requires
/// `Shared`).
pub fn should_emit_tier_b(
    config: &FlashAttentionConfig,
    seq_len: u64,
    residency: SegmentResidency,
) -> bool {
    if !config.segment_masked {
        return false;
    }
    if !matches!(residency, SegmentResidency::Shared) {
        return false;
    }
    let bytes = compute_range_table_bytes(seq_len, config.block_q as u64, config.block_kv as u64);
    bytes <= TIER_B_RANGE_TABLE_BUDGET_BYTES
}

/// Emit PTX that builds the per-tile segment-range table at the top
/// of the kernel preamble.
///
/// Reads segment_ids from SMEM (already populated by Tier A), runs a
/// parallel min/max reduction per tile, and stores results into
/// `qtile_seg_min[num_q_tiles] : u16`, `qtile_seg_max[num_q_tiles] : u16`,
/// `kvtile_seg_min[num_kv_tiles] : u16`, `kvtile_seg_max[num_kv_tiles] : u16`.
///
/// Stores values as warp-uniform classified so subsequent
/// [`emit_skip_predicate`] calls compile to `BRA.U`.
///
/// **Stub in skeleton — real emission in Task 3.**
pub fn emit_range_table_preamble(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    seq_len: u64,
    segment_ids_reg: &str,
) {
    let _ = (config, seq_len, segment_ids_reg);
    ptx.push_str("    // PCA Tier B: range-table preamble (Task 3 stub)\n");
}

/// Emit PTX that branches to `on_skip_label` if the `(qt, kvt)` tile
/// pair has strictly disjoint segment ranges.
///
/// Compiles to one comparison + one warp-uniform branch (`BRA.U`).
///
/// **Stub in skeleton — real emission in Task 4.**
pub fn emit_skip_predicate(
    ptx: &mut String,
    qt_reg: &str,
    kvt_reg: &str,
    on_skip_label: &str,
) {
    let _ = (qt_reg, kvt_reg, on_skip_label);
    ptx.push_str("    // PCA Tier B: skip predicate (Task 4 stub)\n");
}

/// Emit PTX that writes the skip decision for `(b, h, qt, kvt)` to
/// the debug instrumentation buffer at the matching slot.
///
/// Lane 0 of the warp owning the tile pair writes; other threads
/// no-op. Decision: `1` = disjoint (skipped), `0` = kept.
///
/// Gated behind `cfg!(debug_kernel_instrumentation)` at codegen time
/// — production builds never call this function.
///
/// **Stub in skeleton — real emission in Task 5.**
pub fn emit_skip_decision_writeback(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    qt_reg: &str,
    kvt_reg: &str,
    is_skip_pred: &str,
    decisions_buf_reg: &str,
) {
    let _ = (config, qt_reg, kvt_reg, is_skip_pred, decisions_buf_reg);
    ptx.push_str("    // PCA Tier B: writeback (Task 5 stub)\n");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fa_base_for_test() -> FlashAttentionConfig {
        // Mirrors pca_segment.rs::tests::fa_base — same shape so
        // skeleton tests share the precedent's config surface.
        use crate::flash_attention::RopeStyle;
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
            gpu_sm: 90,
            segment_masked: true,
            csha: None,
        }
    }

    #[test]
    fn range_table_bytes_4k_block_64() {
        // 4096 / 64 = 64 q-tiles, 64 kv-tiles.
        // 2 × (64 + 64) × 2 = 512 B.
        assert_eq!(compute_range_table_bytes(4096, 64, 64), 512);
    }

    #[test]
    fn range_table_bytes_16k_block_64() {
        // 16384 / 64 = 256 tiles each.
        // 2 × (256 + 256) × 2 = 2048 B.
        assert_eq!(compute_range_table_bytes(16384, 64, 64), 2048);
    }

    #[test]
    fn range_table_bytes_asymmetric_block() {
        // 16K with block_q=32, block_kv=64, head_dim=256.
        // num_q = 512, num_kv = 256. 2 × (512+256) × 2 = 3072 B.
        assert_eq!(compute_range_table_bytes(16384, 32, 64), 3072);
    }

    #[test]
    fn should_emit_tier_b_inside_envelope() {
        let cfg = fa_base_for_test();
        assert!(should_emit_tier_b(&cfg, 4096, SegmentResidency::Shared));
    }

    #[test]
    fn should_skip_tier_b_when_streamed() {
        let cfg = fa_base_for_test();
        assert!(!should_emit_tier_b(&cfg, 4096, SegmentResidency::Streamed));
    }

    #[test]
    fn should_skip_tier_b_when_segment_masked_off() {
        let mut cfg = fa_base_for_test();
        cfg.segment_masked = false;
        assert!(!should_emit_tier_b(&cfg, 4096, SegmentResidency::Shared));
    }

    #[test]
    fn should_skip_tier_b_when_table_exceeds_budget() {
        // Hypothetical 1 M sequence → 16384 q tiles each side → 128 KB.
        let cfg = fa_base_for_test();
        assert!(!should_emit_tier_b(&cfg, 1_048_576, SegmentResidency::Shared));
    }

    /// Boundary-pair test that anchors directly on `TIER_B_RANGE_TABLE_BUDGET_BYTES`.
    /// If the const drifts, at least one assertion fails immediately —
    /// neither `should_emit_tier_b_inside_envelope` (4 K seq → 512 B, ¹⁄₁₆-budget)
    /// nor `should_skip_tier_b_when_table_exceeds_budget` (1 M seq → 128 KB, 16×-budget)
    /// catches drift to 4 KB or 16 KB.
    #[test]
    fn at_range_table_budget_boundary_decides_correctly() {
        // compute_range_table_bytes(16384, 16, 16) = 2 * (1024 + 1024) * 2 = 8192.
        assert_eq!(compute_range_table_bytes(16_384, 16, 16), 8192);
        // compute_range_table_bytes(16384, 8, 8)   = 2 * (2048 + 2048) * 2 = 16384.
        assert_eq!(compute_range_table_bytes(16_384, 8, 8), 16_384);

        // Exactly at budget → fits (`<=` semantics in should_emit_tier_b).
        let mut cfg_at = fa_base_for_test();
        cfg_at.block_q = 16;
        cfg_at.block_kv = 16;
        assert!(should_emit_tier_b(&cfg_at, 16_384, SegmentResidency::Shared));

        // Just past budget → falls back.
        let mut cfg_over = fa_base_for_test();
        cfg_over.block_q = 8;
        cfg_over.block_kv = 8;
        assert!(!should_emit_tier_b(&cfg_over, 16_384, SegmentResidency::Shared));
    }

    #[test]
    fn skeleton_emitters_emit_marker_comments() {
        let mut s = String::new();
        let cfg = fa_base_for_test();
        emit_range_table_preamble(&mut s, &cfg, 4096, "%seg_base");
        emit_skip_predicate(&mut s, "%qt", "%kvt", "skip_kv_iter");
        emit_skip_decision_writeback(&mut s, &cfg, "%qt", "%kvt", "%p_skip_TILERANGE", "%dec_buf");
        assert!(s.contains("PCA Tier B"), "output missing marker:\n{}", s);
    }
}
