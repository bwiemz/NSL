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

/// Budget for segment IDs in SMEM per CTA (bytes).  Kept deliberately
/// conservative — real kernels reserve more SMEM for Q/K/V tiles + softmax
/// stats, so we only use a small fixed slice here.
pub const DEFAULT_SMEM_SEGMENT_BUDGET: u64 = 4096;

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
        // 2048 × 2 = 4096 bytes — exactly fits the default SMEM budget.
        let det = det_with_strategy();
        let plan = plan_kernel(&det, fa_base(), 2048, true);
        assert_eq!(plan.residency, SegmentResidency::Shared);
        assert_eq!(plan.smem_segment_bytes, 4096);
    }

    #[test]
    fn long_seq_streams_segment_ids() {
        let det = det_with_strategy();
        // 4096 tokens × 2 bytes = 8 KB — exceeds budget.
        let plan = plan_kernel(&det, fa_base(), 4096, true);
        assert_eq!(plan.residency, SegmentResidency::Streamed);
        assert_eq!(plan.smem_segment_bytes, 0);
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
