//! Per-sequence mapping from logical token position to a physical block.
//!
//! A `PageTable` tracks which physical blocks have been allocated for a single
//! sequence and provides the logical-to-physical translation needed by the
//! paged attention kernel.

use crate::paged_kv::BlockId;

/// Per-sequence page table: maps logical token indices to physical blocks.
///
/// Tokens are appended one at a time.  When a block boundary is reached the
/// caller must first call `push_block` with a freshly-allocated `BlockId` before
/// calling `append_token` again.
pub struct PageTable {
    /// Ordered list of physical blocks assigned to this sequence.
    entries: Vec<BlockId>,
    /// Total number of tokens appended so far.
    token_count: usize,
    /// Maximum number of tokens that fit in one block.
    block_size: usize,
}

impl PageTable {
    /// Create an empty page table with the given block size.
    pub fn new(block_size: usize) -> Self {
        assert!(block_size > 0, "block_size must be > 0");
        PageTable {
            entries: Vec::new(),
            token_count: 0,
            block_size,
        }
    }

    // ── Accessors ─────────────────────────────────────────────────────────────

    /// Total number of tokens recorded in this sequence.
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Number of physical blocks currently assigned to this sequence.
    pub fn num_blocks(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` when the next token requires a new block to be allocated.
    ///
    /// This is the case when no blocks have been assigned yet, or all assigned
    /// block capacity has been consumed (every slot in every block is occupied).
    pub fn needs_new_block(&self) -> bool {
        self.entries.len() * self.block_size <= self.token_count
    }

    // ── Mutation ──────────────────────────────────────────────────────────────

    /// Register a newly-allocated block for this sequence.
    ///
    /// Must be called whenever `needs_new_block()` returns `true`, before the
    /// next `append_token` call.
    pub fn push_block(&mut self, block_id: BlockId) {
        self.entries.push(block_id);
    }

    /// Record one token in the current block.
    ///
    /// Returns `(block_id, offset_within_block)` for the position that was
    /// assigned to this token.
    ///
    /// # Panics
    /// Panics if `needs_new_block()` is true (caller forgot to push a block).
    pub fn append_token(&mut self) -> (BlockId, usize) {
        assert!(
            !self.needs_new_block(),
            "append_token called but needs_new_block() is true — call push_block first"
        );
        let offset = self.token_count % self.block_size;
        let block_idx = self.token_count / self.block_size;
        let block_id = self.entries[block_idx];
        self.token_count += 1;
        (block_id, offset)
    }

    // ── Lookup ────────────────────────────────────────────────────────────────

    /// Look up the physical block that contains the token at `logical_index`.
    ///
    /// Returns `None` if `logical_index >= token_count`.
    pub fn get_block(&self, logical_index: usize) -> Option<BlockId> {
        if logical_index >= self.token_count {
            return None;
        }
        let block_idx = logical_index / self.block_size;
        self.entries.get(block_idx).copied()
    }

    // ── Bulk operations ───────────────────────────────────────────────────────

    /// Slice of all physical block IDs, in order.
    ///
    /// Useful when copying the block table to GPU memory for the attention kernel.
    pub fn block_ids(&self) -> &[BlockId] {
        &self.entries
    }

    /// Remove and return all physical block IDs, resetting the sequence to empty.
    ///
    /// The caller is responsible for freeing the returned blocks back to the
    /// `BlockAllocator`.
    pub fn drain_blocks(&mut self) -> Vec<BlockId> {
        self.token_count = 0;
        std::mem::take(&mut self.entries)
    }

    /// Shallow clone for CoW branching — shares block IDs with parent.
    pub fn clone_shallow(&self) -> Self {
        PageTable {
            entries: self.entries.clone(),
            token_count: self.token_count,
            block_size: self.block_size,
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_table_needs_block() {
        let pt = PageTable::new(4);
        assert!(pt.needs_new_block(), "fresh table should need a block");
        assert_eq!(pt.token_count(), 0);
        assert_eq!(pt.num_blocks(), 0);
    }

    #[test]
    fn test_append_within_block() {
        let mut pt = PageTable::new(4);
        pt.push_block(7);
        let (bid, off) = pt.append_token();
        assert_eq!(bid, 7, "expected block_id 7");
        assert_eq!(off, 0, "first token offset should be 0");

        let (bid2, off2) = pt.append_token();
        assert_eq!(bid2, 7);
        assert_eq!(off2, 1);
    }

    #[test]
    fn test_block_boundary() {
        let mut pt = PageTable::new(2);
        pt.push_block(10);
        pt.append_token(); // token 0, offset 0
        pt.append_token(); // token 1, offset 1

        // Block is now full — needs a new one before the third token.
        assert!(pt.needs_new_block(), "after filling block, needs_new_block should be true");

        pt.push_block(20);
        let (bid, off) = pt.append_token();
        assert_eq!(bid, 20, "token 2 should be in second block");
        assert_eq!(off, 0, "first token of second block has offset 0");
    }

    #[test]
    fn test_drain_blocks() {
        let mut pt = PageTable::new(2);
        pt.push_block(1);
        pt.append_token();
        pt.append_token();
        pt.push_block(2);
        pt.append_token();

        let drained = pt.drain_blocks();
        assert_eq!(drained, vec![1, 2]);
        assert_eq!(pt.token_count(), 0, "token_count should reset after drain");
        assert_eq!(pt.num_blocks(), 0, "num_blocks should reset after drain");
        assert!(pt.needs_new_block(), "should need a block after drain");
    }

    #[test]
    fn test_block_ids_for_sync() {
        let mut pt = PageTable::new(3);
        pt.push_block(100);
        pt.append_token();
        pt.append_token();
        pt.append_token();
        pt.push_block(200);
        pt.append_token();

        let ids = pt.block_ids();
        assert_eq!(ids, &[100u32, 200u32], "block_ids() should return [100, 200]");
    }
}
