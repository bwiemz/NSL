//! PCA — segment-ID-aware FlashAttention kernel plan.
//!
//! When the packing strategy is [`PcaStrategy::SegmentIdMasked`], PCA
//! replaces the dense `S×S` attention mask with a compact `segment_ids: [u16; S]`
//! tensor.  Inside the FA2 KV-tile loop, scores between query `q` and key
//! `k` are masked iff `segment_ids[q] != segment_ids[k]`.
//!
//! This module does not emit PTX.  It emits a [`SegmentKernelPlan`] that
//! the existing FA PTX generator consumes to pick the right mask-variant
//! and wire up the segment-ID shared-memory layout.

use serde::Serialize;

use crate::flash_attention::FlashAttentionConfig;
use crate::pca_detect::{PcaDetection, PcaStrategy};

/// Where the segment-ID tensor is resident during attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SegmentResidency {
    /// Segment IDs fit in shared memory — loaded once per CTA, reused for
    /// all KV tiles.
    Shared,
    /// Sequence too long for SMEM — segment IDs streamed from HBM per
    /// tile.
    Streamed,
}

/// Kernel plan produced by [`plan_kernel`].
#[derive(Debug, Clone)]
pub struct SegmentKernelPlan {
    pub strategy: PcaStrategy,
    /// FA config the kernel generator must honour.  Inherits from the
    /// base FA2 config.
    pub fa_config: FlashAttentionConfig,
    pub residency: SegmentResidency,
    /// Bytes of segment-ID data needed in SMEM when resident (`Shared`).
    pub smem_segment_bytes: u64,
    /// Whether the kernel must apply the standard causal mask **and** the
    /// segment mask.  False for bidirectional packed sequences.
    pub causal_and_segment: bool,
}

impl SegmentKernelPlan {
    /// Is this plan a no-op (packing disabled)?
    pub fn is_noop(&self) -> bool {
        matches!(self.strategy, PcaStrategy::NoPacking)
    }
}

/// Per-CTA SMEM budget for segment_ids in `Shared` residency.
///
/// Set to 32 KB (was 4 KB) as PCA Tier B prerequisite per spec
/// `2026-05-02-pca-tier-b-tile-skip-design.md` §3.4.1: enables
/// `Shared` residency for seq_len ≤ 16 K with u16 segment IDs,
/// covering the §4.4 fixture matrix. Combined with Q/K/V tiles
/// + softmax stats + Tier B's range table, total per-CTA SMEM
/// is ≈ 58.5 KB at the worst-case fixture (`long_seq_5doc`,
/// 16 K seq, `block_q = block_kv = 64`); see spec §3.4.2.
pub const DEFAULT_SMEM_SEGMENT_BUDGET: u64 = 32768;

/// Build the kernel plan.  `seq_len` and `dtype_bytes` come from the
/// enclosing sublayer shape.
pub fn plan_kernel(
    detection: &PcaDetection,
    base: FlashAttentionConfig,
    seq_len: u64,
    causal: bool,
) -> SegmentKernelPlan {
    if matches!(detection.strategy, PcaStrategy::NoPacking) {
        return SegmentKernelPlan {
            strategy: PcaStrategy::NoPacking,
            fa_config: base,
            residency: SegmentResidency::Shared,
            smem_segment_bytes: 0,
            causal_and_segment: false,
        };
    }
    // Segment IDs are stored as u16 — 2 bytes per position.
    let needed_bytes = seq_len.saturating_mul(2);
    let residency = if needed_bytes <= DEFAULT_SMEM_SEGMENT_BUDGET {
        SegmentResidency::Shared
    } else {
        SegmentResidency::Streamed
    };
    let smem_bytes = if residency == SegmentResidency::Shared {
        needed_bytes
    } else {
        0
    };
    SegmentKernelPlan {
        strategy: detection.strategy,
        fa_config: base,
        residency,
        smem_segment_bytes: smem_bytes,
        causal_and_segment: causal,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{FlashAttentionConfig, RopeStyle};
    use crate::pca_detect::{
        detect, DatasetPackingConfig, PcaDetectConfig,
    };

    fn fa_base() -> FlashAttentionConfig {
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
        segment_masked: false,
        csha: None,
        }
    }

    fn det_with_strategy() -> PcaDetection {
        detect(
            &DatasetPackingConfig {
                enabled: true,
                max_sequence_length: 1024,
                mean_doc_length: Some(512),
                doc_length_stddev: Some(100),
                separator_token_id: Some(2),
            },
            &PcaDetectConfig::default(),
            2,
        )
    }

    #[test]
    fn noop_plan_when_detection_disabled() {
        let det = detect(
            &DatasetPackingConfig::default(),
            &PcaDetectConfig::default(),
            2,
        );
        let plan = plan_kernel(&det, fa_base(), 1024, true);
        assert!(plan.is_noop());
    }

    #[test]
    fn short_seq_uses_shared_residency() {
        // 4096 × 2 = 8192 bytes — fits the new 32 KB SMEM budget.
        let det = det_with_strategy();
        let plan = plan_kernel(&det, fa_base(), 4096, true);
        assert_eq!(plan.residency, SegmentResidency::Shared);
        assert_eq!(plan.smem_segment_bytes, 8192);
    }

    #[test]
    fn long_seq_streams_segment_ids() {
        let det = det_with_strategy();
        // 32 K tokens × 2 bytes = 64 KB — exceeds the 32 KB Tier B budget.
        let plan = plan_kernel(&det, fa_base(), 32768, true);
        assert_eq!(plan.residency, SegmentResidency::Streamed);
        assert_eq!(plan.smem_segment_bytes, 0);
    }

    /// Boundary-pair test: directly probes `DEFAULT_SMEM_SEGMENT_BUDGET`'s
    /// value. If the const drifts (e.g. someone bumps it to 16 KB or 64 KB),
    /// at least one of these two assertions will fail — which neither
    /// `short_seq_uses_shared_residency` (8 KB, ¼-budget) nor
    /// `long_seq_streams_segment_ids` (64 KB, 2×-budget) catches.
    #[test]
    fn at_budget_boundary_seq_lens_pick_correct_residency() {
        let det = det_with_strategy();
        // seq_len = 16 384 → 32 KB — exactly equal to the budget; fits Shared.
        let plan_at = plan_kernel(&det, fa_base(), 16_384, true);
        assert_eq!(plan_at.residency, SegmentResidency::Shared);
        assert_eq!(plan_at.smem_segment_bytes, 32_768);
        // seq_len = 16 385 → 32 770 B — one element past the budget; Streamed.
        let plan_over = plan_kernel(&det, fa_base(), 16_385, true);
        assert_eq!(plan_over.residency, SegmentResidency::Streamed);
        assert_eq!(plan_over.smem_segment_bytes, 0);
    }

    #[test]
    fn causal_and_segment_flags_combine() {
        let det = det_with_strategy();
        let plan = plan_kernel(&det, fa_base(), 1024, true);
        assert!(plan.causal_and_segment);
        let plan_bi = plan_kernel(&det, fa_base(), 1024, false);
        assert!(!plan_bi.causal_and_segment);
    }
}
