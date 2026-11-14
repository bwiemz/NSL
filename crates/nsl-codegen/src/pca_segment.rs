//! PCA — where the segment-id tensor lives during packed attention.
//!
//! Under packing, the attention kernels replace the dense `S×S` mask with
//! one 16-bit segment id per position: scores between query `q` and key
//! `k` are masked iff `segment_ids[q] != segment_ids[k]`. The forward
//! kernel copies the ids of one row into shared memory once per CTA.

use serde::Serialize;

/// Where the segment-ID tensor is resident during attention.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum SegmentResidency {
    /// Segment IDs fit in shared memory — loaded once per CTA, reused for
    /// all KV tiles.
    Shared,
    /// Sequence too long for SMEM — segment IDs streamed from HBM per
    /// tile. No kernel implements this: the emitters trap on it.
    Streamed,
}

/// Bytes of shared memory the forward kernel reserves for one row's
/// segment ids (`seg_smem`). At two bytes per id this holds
/// [`nsl_abi::wire::pca_tier_b::MAX_SEGMENT_MASKED_SEQ_LEN`] positions; the
/// runtime declines the fused packed forward for a longer sequence, which
/// would otherwise write past the buffer.
pub const DEFAULT_SMEM_SEGMENT_BUDGET: u64 =
    nsl_abi::wire::pca_tier_b::SEGMENT_IDS_SMEM_BYTES as u64;

#[cfg(test)]
mod tests {
    use super::*;

    /// The buffer and the runtime's length bound are one fact stated twice.
    #[test]
    fn the_budget_holds_exactly_the_longest_admitted_sequence() {
        assert_eq!(DEFAULT_SMEM_SEGMENT_BUDGET, 32_768);
        assert_eq!(
            DEFAULT_SMEM_SEGMENT_BUDGET,
            2 * u64::from(nsl_abi::wire::pca_tier_b::MAX_SEGMENT_MASKED_SEQ_LEN)
        );
    }
}
