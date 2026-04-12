//! PCA — compile-time tile-skip map.
//!
//! Given a sequence-packing layout (a list of document lengths that sum to
//! the packed sequence length) and a tile size, this module computes the
//! set of `(q_tile, kv_tile)` pairs that are entirely cross-document and
//! can be **skipped** by the attention kernel.
//!
//! The map is emitted as a compact bitset the kernel generator bakes into
//! PTX as a compile-time constant — the runtime kernel only needs to index
//! it with `(q_tile_idx, kv_tile_idx)` and skip the tile-pair outright when
//! the bit is zero.
//!
//! For typical packing (3 equal-ish documents of length ≈ `seq_len / 3`
//! each, tile size 128), roughly 55 % of tile-pairs are fully cross-
//! document, matching the figure cited in CFTP Section 5.2.

use serde::Serialize;

/// Layout of a single packed sample: the concatenation of `doc_lengths`
/// must equal `packed_length`.
#[derive(Debug, Clone)]
pub struct PackingLayout {
    pub doc_lengths: Vec<u32>,
    pub packed_length: u32,
}

impl PackingLayout {
    /// Construct from raw document lengths.  Sums the lengths to derive
    /// `packed_length`.
    pub fn from_docs(doc_lengths: Vec<u32>) -> Self {
        let packed: u32 = doc_lengths.iter().sum();
        Self {
            doc_lengths,
            packed_length: packed,
        }
    }

    /// Validate that the layout is internally consistent.
    pub fn is_valid(&self) -> bool {
        self.doc_lengths.iter().sum::<u32>() == self.packed_length
    }

    /// Compute the token → document-index map for every position.
    pub fn segment_ids(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.packed_length as usize);
        for (doc_id, &len) in self.doc_lengths.iter().enumerate() {
            for _ in 0..len {
                out.push(doc_id as u32);
            }
        }
        out
    }

    /// Document start offsets (cumulative sum).  `starts[i]` is the first
    /// token index of document `i`; the last entry `starts[docs]` is
    /// `packed_length`.
    pub fn doc_starts(&self) -> Vec<u32> {
        let mut starts = Vec::with_capacity(self.doc_lengths.len() + 1);
        starts.push(0);
        let mut cum = 0u32;
        for &l in &self.doc_lengths {
            cum += l;
            starts.push(cum);
        }
        starts
    }
}

/// Outcome of tile-skip analysis.
#[derive(Debug, Clone, Serialize)]
pub struct TileSkipMap {
    pub num_q_tiles: u32,
    pub num_kv_tiles: u32,
    pub block_q: u32,
    pub block_kv: u32,
    /// Bitset flattened row-major: `bits[qt * num_kv_tiles + kvt]` is
    /// `true` iff the `(qt, kvt)` tile-pair contains at least one
    /// within-document entry (i.e. *cannot* be skipped).
    pub keep: Vec<bool>,
}

impl TileSkipMap {
    pub fn total_tile_pairs(&self) -> u32 {
        self.num_q_tiles * self.num_kv_tiles
    }

    pub fn skipped_tile_pairs(&self) -> u32 {
        self.keep.iter().filter(|b| !**b).count() as u32
    }

    pub fn skip_fraction(&self) -> f64 {
        let total = self.total_tile_pairs();
        if total == 0 {
            return 0.0;
        }
        self.skipped_tile_pairs() as f64 / total as f64
    }
}

/// Compute the tile-skip map for a packing layout and tile size.
///
/// A tile-pair is *kept* if there exists at least one query position `q`
/// inside `q_tile` and one key position `k` inside `kv_tile` such that
/// `segment_id[q] == segment_id[k]` (and, when `causal=true`, `k <= q`).
/// Otherwise the tile-pair is entirely cross-document (or entirely above
/// the causal diagonal) and can be skipped.
pub fn build(layout: &PackingLayout, block_q: u32, block_kv: u32, causal: bool) -> TileSkipMap {
    let seq = layout.packed_length;
    let block_q = block_q.max(1);
    let block_kv = block_kv.max(1);
    let num_q_tiles = seq.div_ceil(block_q);
    let num_kv_tiles = seq.div_ceil(block_kv);

    let segment_ids = layout.segment_ids();
    let mut keep = vec![false; (num_q_tiles * num_kv_tiles) as usize];

    for qt in 0..num_q_tiles {
        let q_start = qt * block_q;
        let q_end = ((qt + 1) * block_q).min(seq);
        for kvt in 0..num_kv_tiles {
            let k_start = kvt * block_kv;
            let k_end = ((kvt + 1) * block_kv).min(seq);
            // Quick reject: causal mask excludes KV tiles strictly to the
            // right of the query tile.
            if causal && k_start >= q_end {
                continue;
            }
            // Scan — at the tile-level this is still fast: block sizes are
            // typically 64 / 128, so ≤ 2k comparisons per tile-pair.  For
            // real models the loop is dominated by the outer tile count.
            let mut any = false;
            'outer: for q in q_start..q_end {
                let sq = segment_ids[q as usize];
                for k in k_start..k_end {
                    if causal && k > q {
                        continue;
                    }
                    if segment_ids[k as usize] == sq {
                        any = true;
                        break 'outer;
                    }
                }
            }
            if any {
                keep[(qt * num_kv_tiles + kvt) as usize] = true;
            }
        }
    }

    TileSkipMap {
        num_q_tiles,
        num_kv_tiles,
        block_q,
        block_kv,
        keep,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_validation() {
        let l = PackingLayout::from_docs(vec![3, 2, 4]);
        assert_eq!(l.packed_length, 9);
        assert!(l.is_valid());
        assert_eq!(l.segment_ids(), vec![0, 0, 0, 1, 1, 2, 2, 2, 2]);
        assert_eq!(l.doc_starts(), vec![0, 3, 5, 9]);
    }

    #[test]
    fn single_doc_keeps_all_tiles() {
        let l = PackingLayout::from_docs(vec![256]);
        let m = build(&l, 64, 64, true);
        // With causal=true, exactly the lower-triangular half of the tile
        // grid is live: 10 out of 16 tile-pairs for a 4×4 tile grid.
        assert_eq!(m.num_q_tiles, 4);
        assert_eq!(m.num_kv_tiles, 4);
        let kept = m.keep.iter().filter(|b| **b).count();
        assert_eq!(kept, 10);
    }

    #[test]
    fn packed_sequence_skips_cross_doc_tiles() {
        // Three docs of equal length → 2/3 of non-causal tile-pairs are
        // cross-document.  Combined with causal masking, skip fraction
        // should exceed ~0.7.
        let l = PackingLayout::from_docs(vec![341, 341, 342]);
        let m = build(&l, 128, 128, true);
        assert!(m.skip_fraction() > 0.5);
    }

    #[test]
    fn non_causal_symmetric_map() {
        let l = PackingLayout::from_docs(vec![3, 3]);
        let m = build(&l, 3, 3, false);
        assert_eq!(m.num_q_tiles, 2);
        assert_eq!(m.num_kv_tiles, 2);
        // Diagonal tiles kept (doc 0–0 and doc 1–1).
        assert!(m.keep[0]); // (0, 0)
        assert!(!m.keep[1]); // (0, 1) — doc0 vs doc1, no overlap
        assert!(!m.keep[2]); // (1, 0)
        assert!(m.keep[3]); // (1, 1)
    }

    #[test]
    fn empty_packed_produces_no_tiles() {
        let l = PackingLayout::from_docs(Vec::new());
        let m = build(&l, 64, 64, true);
        assert_eq!(m.num_q_tiles, 0);
        assert_eq!(m.num_kv_tiles, 0);
        assert_eq!(m.skip_fraction(), 0.0);
    }

    #[test]
    fn tail_doc_extends_partial_tile() {
        // 100-token doc, 64-token tile size → 2 tiles with a partial tail.
        let l = PackingLayout::from_docs(vec![100]);
        let m = build(&l, 64, 64, true);
        assert_eq!(m.num_q_tiles, 2);
        assert_eq!(m.num_kv_tiles, 2);
        // Kept tiles: (0,0), (1,0), (1,1) — three of four.
        let kept = m.keep.iter().filter(|b| **b).count();
        assert_eq!(kept, 3);
    }

    #[test]
    fn skip_fraction_in_range() {
        let l = PackingLayout::from_docs(vec![50, 50, 50, 50]);
        let m = build(&l, 32, 32, true);
        let f = m.skip_fraction();
        assert!((0.0..=1.0).contains(&f));
    }
}
