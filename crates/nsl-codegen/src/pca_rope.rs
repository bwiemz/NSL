//! PCA — document-aware RoPE position reset.
//!
//! When sequences are packed, positions must *reset* at each document
//! boundary so that each document sees RoPE positions `0..len_i`.  In the
//! standard flow this requires a separate `position_ids: [S]` tensor
//! constructed by the DataLoader.  PCA eliminates this tensor by fusing
//! the position-reset computation into the RoPE kernel's epilogue (the
//! same one CSHA Level 1 already fuses into the Q/K projections).
//!
//! This module computes the per-position offset function that the fused
//! RoPE kernel evaluates.  Rather than emitting PTX directly, we produce
//! a [`RopePositionPlan`] with three representations:
//!
//!   * `doc_starts: [num_docs + 1]` — cumulative start offsets
//!   * `position_offsets: [S]` — resolved `i - doc_start(segment_id(i))`
//!   * `formula` — a textual representation for CLI reports
//!
//! Only `doc_starts` is consumed by the kernel (the other representations
//! are for testing and diagnostics).

use serde::Serialize;

use crate::pca_tileskip::PackingLayout;

/// RoPE reset plan: how each token's position is computed inside the
/// fused RoPE kernel.
#[derive(Debug, Clone, Serialize)]
pub struct RopePositionPlan {
    /// Document starts `[num_docs + 1]`.
    pub doc_starts: Vec<u32>,
    /// Resolved per-position offsets, for testing.
    pub position_offsets: Vec<u32>,
    /// Pseudocode describing the kernel-level computation.
    pub formula: String,
    /// Whether the plan is a true packing plan (true) or a degenerate
    /// identity (false — single document, no reset needed).
    pub needs_reset: bool,
}

impl RopePositionPlan {
    /// Number of documents in the packed sample.
    pub fn num_docs(&self) -> u32 {
        self.doc_starts.len().saturating_sub(1) as u32
    }

    /// Packed sequence length.
    pub fn packed_length(&self) -> u32 {
        self.doc_starts.last().copied().unwrap_or(0)
    }
}

/// Compute the per-position offset `i - doc_start(segment_id(i))` for a
/// packing layout.  Returns a plan the fused RoPE kernel consumes.
pub fn plan(layout: &PackingLayout) -> RopePositionPlan {
    let doc_starts = layout.doc_starts();
    let segments = layout.segment_ids();
    let mut offsets = Vec::with_capacity(segments.len());
    for (i, seg) in segments.iter().enumerate() {
        let start = doc_starts[*seg as usize];
        offsets.push(i as u32 - start);
    }
    let needs_reset = layout.doc_lengths.len() > 1;
    let formula = if needs_reset {
        "position[i] = i - doc_starts[segment_ids[i]]".to_string()
    } else {
        "position[i] = i  (single document — no reset)".to_string()
    };
    RopePositionPlan {
        doc_starts,
        position_offsets: offsets,
        formula,
        needs_reset,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_doc_is_identity() {
        let p = plan(&PackingLayout::from_docs(vec![5]));
        assert!(!p.needs_reset);
        assert_eq!(p.position_offsets, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn three_docs_reset_positions() {
        let p = plan(&PackingLayout::from_docs(vec![3, 2, 4]));
        assert!(p.needs_reset);
        assert_eq!(
            p.position_offsets,
            vec![0, 1, 2, 0, 1, 0, 1, 2, 3]
        );
    }

    #[test]
    fn doc_starts_match_cumulative_sum() {
        let p = plan(&PackingLayout::from_docs(vec![3, 2, 4]));
        assert_eq!(p.doc_starts, vec![0, 3, 5, 9]);
        assert_eq!(p.packed_length(), 9);
        assert_eq!(p.num_docs(), 3);
    }

    #[test]
    fn empty_layout_is_trivial() {
        let p = plan(&PackingLayout::from_docs(Vec::new()));
        assert_eq!(p.num_docs(), 0);
        assert_eq!(p.packed_length(), 0);
        assert!(p.position_offsets.is_empty());
    }

    #[test]
    fn long_single_doc_keeps_linear() {
        let p = plan(&PackingLayout::from_docs(vec![128]));
        for (i, off) in p.position_offsets.iter().enumerate() {
            assert_eq!(*off, i as u32);
        }
    }

    #[test]
    fn formula_contains_reset_text_when_reset_needed() {
        let p = plan(&PackingLayout::from_docs(vec![3, 2]));
        assert!(p.formula.contains("segment_ids"));
        let p2 = plan(&PackingLayout::from_docs(vec![5]));
        assert!(p2.formula.contains("single document"));
    }
}
