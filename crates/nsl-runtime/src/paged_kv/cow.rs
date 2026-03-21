//! Copy-on-Write page branching for speculative decoding (M33).

use super::BlockId;
use super::block_alloc::BlockAllocator;
use super::page_table::PageTable;

/// Branch a sequence's page table for speculative decoding.
/// The child shares all physical pages with the parent.
pub fn branch_page_table(parent_table: &PageTable, allocator: &mut BlockAllocator) -> PageTable {
    let child = parent_table.clone_shallow();
    for &block_id in child.block_ids() {
        allocator.incref(block_id);
    }
    child
}

/// Copy-on-write for a specific block within a page table.
///
/// If refcount > 1, copies data to a new block, updates the page table entry
/// at `entry_index` to point to the new block, and decrements the old block's
/// refcount. If refcount == 1, returns the same block (already exclusively owned).
///
/// **SAFETY:** The caller must ensure no concurrent readers are accessing
/// `page_table[entry_index]` during this operation. The page table update
/// and refcount decrement are NOT atomic with respect to each other.
pub fn cow_copy_block(
    block_id: BlockId,
    entry_index: usize,
    page_table: &mut PageTable,
    allocator: &mut BlockAllocator,
) -> Option<BlockId> {
    if allocator.refcount(block_id) <= 1 {
        return Some(block_id);
    }
    let new_id = allocator.alloc()?;
    let block_bytes = allocator.block_stride;
    unsafe {
        let src_k = allocator.k_block_ptr(block_id);
        let dst_k = allocator.k_block_ptr(new_id);
        std::ptr::copy_nonoverlapping(src_k as *const u8, dst_k as *mut u8, block_bytes);
        let src_v = allocator.v_block_ptr(block_id);
        let dst_v = allocator.v_block_ptr(new_id);
        std::ptr::copy_nonoverlapping(src_v as *const u8, dst_v as *mut u8, block_bytes);
    }
    // Atomically update page table entry BEFORE decrementing old refcount.
    // This ensures no reader sees the old block_id with a decremented refcount.
    page_table.replace_block(entry_index, new_id);
    allocator.free(block_id); // decrement shared refcount
    Some(new_id)
}

/// Clean up a speculative branch — free all blocks, respecting refcounts.
pub fn cleanup_branch(table: &mut PageTable, allocator: &mut BlockAllocator) {
    let blocks = table.drain_blocks();
    for block_id in blocks {
        allocator.free(block_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_allocator() -> BlockAllocator {
        BlockAllocator::new_cpu(16, 4, 1, 64)
    }

    #[test]
    fn test_branch_increments_refcount() {
        let mut alloc = make_allocator();
        let mut parent = PageTable::new(4);
        let b0 = alloc.alloc().unwrap();
        parent.push_block(b0);
        assert_eq!(alloc.refcount(b0), 1);

        let _child = branch_page_table(&parent, &mut alloc);
        assert_eq!(alloc.refcount(b0), 2);
    }

    #[test]
    fn test_cow_copy_creates_new_block() {
        let mut alloc = make_allocator();
        let mut parent = PageTable::new(4);
        let b0 = alloc.alloc().unwrap();
        parent.push_block(b0);
        let mut child = branch_page_table(&parent, &mut alloc);
        assert_eq!(alloc.refcount(b0), 2);

        let new_id = cow_copy_block(b0, 0, &mut child, &mut alloc).unwrap();
        assert_ne!(new_id, b0);
        assert_eq!(alloc.refcount(b0), 1);
        assert_eq!(alloc.refcount(new_id), 1);
    }

    #[test]
    fn test_double_branch_cleanup_one() {
        let mut alloc = make_allocator();
        let mut parent = PageTable::new(4);
        let b0 = alloc.alloc().unwrap();
        parent.push_block(b0);

        let _child1 = branch_page_table(&parent, &mut alloc);
        let mut child2 = branch_page_table(&parent, &mut alloc);
        assert_eq!(alloc.refcount(b0), 3);

        cleanup_branch(&mut child2, &mut alloc);
        assert_eq!(alloc.refcount(b0), 2); // parent + child1
    }

    #[test]
    fn test_cleanup_frees_unshared_blocks() {
        let mut alloc = make_allocator();
        let mut parent = PageTable::new(4);
        let b0 = alloc.alloc().unwrap();
        parent.push_block(b0);

        let mut child = branch_page_table(&parent, &mut alloc);
        let avail_before = alloc.available();

        cleanup_branch(&mut child, &mut alloc);
        assert_eq!(alloc.refcount(b0), 1);
        assert_eq!(alloc.available(), avail_before); // no blocks freed (parent still holds)
    }
}
