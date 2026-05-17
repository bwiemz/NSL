//! Shared tile-count helper for PCA Tier B and FA-2 kernel emission.
//!
//! Single source of truth: both kernel tile-loop emission and
//! `pca_tilerange::range_table_addrs` callers compute tile counts via
//! this helper, so they cannot drift.

/// Compute the number of tiles for a given (seq_len, block_size).
///
/// Ceiling division — the last tile may be partial. Panics in debug
/// builds if `block_size == 0`.
pub fn num_tiles(seq_len: u32, block_size: u32) -> u32 {
    seq_len.div_ceil(block_size)
}
