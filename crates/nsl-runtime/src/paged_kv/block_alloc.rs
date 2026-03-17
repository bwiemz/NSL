//! Fixed-size block pool for paged KV-cache storage.
//!
//! Provides O(1) alloc/free via a free-list over a pre-allocated contiguous pool.
//! Separate K and V pools are allocated once at construction and indexed by BlockId.

use std::ffi::c_void;

use crate::paged_kv::BlockId;

// ── Pool backend ─────────────────────────────────────────────────────────────

enum PoolBackend {
    Cpu,
    #[cfg(feature = "cuda")]
    Gpu,
}

// ── BlockAllocator ────────────────────────────────────────────────────────────

/// A fixed-size block pool with O(1) alloc/free via free-list.
///
/// Each block holds `num_heads * block_size * head_dim` f32 values.
/// Two pools are maintained: one for K (key) tensors, one for V (value) tensors.
pub struct BlockAllocator {
    /// Tokens per block.
    pub block_size: usize,
    /// Total number of blocks in the pool.
    pub num_blocks: usize,
    /// Number of attention heads stored per block.
    pub num_heads: usize,
    /// Dimension of each attention head.
    pub head_dim: usize,
    /// Bytes per block: `num_heads * block_size * head_dim * size_of::<f32>()`.
    pub block_stride: usize,
    free_list: Vec<BlockId>,
    refcounts: Vec<u32>,
    k_pool: *mut c_void,
    v_pool: *mut c_void,
    backend: PoolBackend,
    allocated_count: usize,
}

// SAFETY: BlockAllocator has exclusive ownership of the raw pool pointers.
// No other code holds or shares these pointers, so cross-thread transfer is safe.
unsafe impl Send for BlockAllocator {}

impl BlockAllocator {
    /// Create a new CPU-backed block allocator.
    pub fn new_cpu(num_blocks: usize, block_size: usize, num_heads: usize, head_dim: usize) -> Self {
        let block_stride = num_heads * block_size * head_dim * std::mem::size_of::<f32>();
        let pool_bytes = num_blocks * block_stride;

        let k_pool = if pool_bytes > 0 {
            crate::memory::checked_alloc_zeroed(pool_bytes) as *mut c_void
        } else {
            std::ptr::NonNull::<u64>::dangling().as_ptr() as *mut c_void
        };
        let v_pool = if pool_bytes > 0 {
            crate::memory::checked_alloc_zeroed(pool_bytes) as *mut c_void
        } else {
            std::ptr::NonNull::<u64>::dangling().as_ptr() as *mut c_void
        };

        // Pre-populate free list in reverse so block 0 is at the top (pop order).
        let free_list = (0..num_blocks as BlockId).rev().collect();

        BlockAllocator {
            block_size,
            num_blocks,
            num_heads,
            head_dim,
            block_stride,
            free_list,
            refcounts: vec![0; num_blocks],
            k_pool,
            v_pool,
            backend: PoolBackend::Cpu,
            allocated_count: 0,
        }
    }

    /// Create a new GPU-backed block allocator (requires the `cuda` feature).
    #[cfg(feature = "cuda")]
    pub fn new_gpu(num_blocks: usize, block_size: usize, num_heads: usize, head_dim: usize) -> Self {
        let block_stride = num_heads * block_size * head_dim * std::mem::size_of::<f32>();
        let pool_bytes = num_blocks * block_stride;

        let k_pool = if pool_bytes > 0 {
            crate::cuda::inner::alloc_device(pool_bytes)
        } else {
            std::ptr::null_mut()
        };
        let v_pool = if pool_bytes > 0 {
            crate::cuda::inner::alloc_device(pool_bytes)
        } else {
            std::ptr::null_mut()
        };

        let free_list = (0..num_blocks as BlockId).rev().collect();

        BlockAllocator {
            block_size,
            num_blocks,
            num_heads,
            head_dim,
            block_stride,
            free_list,
            refcounts: vec![0; num_blocks],
            k_pool,
            v_pool,
            backend: PoolBackend::Gpu,
            allocated_count: 0,
        }
    }

    // ── Allocation ────────────────────────────────────────────────────────────

    /// Allocate one block from the pool.  Returns `None` when the pool is full.
    pub fn alloc(&mut self) -> Option<BlockId> {
        let id = self.free_list.pop()?;
        self.allocated_count += 1;
        self.refcounts[id as usize] = 1;
        Some(id)
    }

    /// Return a block to the free list, respecting refcounts.
    ///
    /// If the block is shared (refcount > 1), only decrements the refcount.
    /// If the block is exclusively owned (refcount <= 1), returns it to the pool.
    pub fn free(&mut self, id: BlockId) {
        debug_assert!((id as usize) < self.num_blocks, "BlockId out of range");
        let idx = id as usize;
        if self.refcounts[idx] > 1 {
            self.refcounts[idx] -= 1;
            return; // shared block — just decrement, don't return to pool
        }
        // Original behavior for single-owner blocks
        debug_assert!(
            self.allocated_count > 0,
            "free() called but allocated_count is 0"
        );
        self.refcounts[idx] = 0;
        self.free_list.push(id);
        self.allocated_count -= 1;
    }

    // ── Refcount helpers ───────────────────────────────────────────────────────

    /// Increment refcount for CoW branching.
    pub fn incref(&mut self, id: BlockId) {
        self.refcounts[id as usize] += 1;
    }

    /// Get current refcount.
    pub fn refcount(&self, id: BlockId) -> u32 {
        self.refcounts[id as usize]
    }

    // ── Pointer accessors ─────────────────────────────────────────────────────

    /// Raw pointer to the beginning of the K data for block `id`.
    ///
    /// # Safety
    /// `id` must be a currently-allocated block id from this allocator.
    pub unsafe fn k_block_ptr(&self, id: BlockId) -> *mut f32 {
        let offset = id as usize * self.block_stride;
        // SAFETY: offset is within the allocated pool.
        unsafe { (self.k_pool as *mut u8).add(offset) as *mut f32 }
    }

    /// Raw pointer to the beginning of the V data for block `id`.
    ///
    /// # Safety
    /// `id` must be a currently-allocated block id from this allocator.
    pub unsafe fn v_block_ptr(&self, id: BlockId) -> *mut f32 {
        let offset = id as usize * self.block_stride;
        // SAFETY: offset is within the allocated pool.
        unsafe { (self.v_pool as *mut u8).add(offset) as *mut f32 }
    }

    // ── Statistics ────────────────────────────────────────────────────────────

    /// Number of currently-allocated blocks.
    pub fn allocated(&self) -> usize {
        self.allocated_count
    }

    /// Number of free blocks remaining in the pool.
    pub fn available(&self) -> usize {
        self.num_blocks - self.allocated_count
    }

    /// Fraction of blocks in use, in `[0.0, 1.0]`.
    pub fn utilization(&self) -> f64 {
        if self.num_blocks == 0 {
            return 0.0;
        }
        self.allocated_count as f64 / self.num_blocks as f64
    }
}

impl Drop for BlockAllocator {
    fn drop(&mut self) {
        let pool_bytes = self.num_blocks * self.block_stride;
        if pool_bytes == 0 {
            return;
        }
        match self.backend {
            PoolBackend::Cpu => {
                // SAFETY: k_pool / v_pool were allocated with checked_alloc_zeroed(pool_bytes).
                unsafe {
                    crate::memory::checked_free(self.k_pool as *mut u8, pool_bytes);
                    crate::memory::checked_free(self.v_pool as *mut u8, pool_bytes);
                }
            }
            #[cfg(feature = "cuda")]
            PoolBackend::Gpu => {
                crate::cuda::inner::free_device(self.k_pool);
                crate::cuda::inner::free_device(self.v_pool);
            }
        }
    }
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn small_alloc() -> BlockAllocator {
        // 4 blocks, block_size=2, num_heads=2, head_dim=4
        BlockAllocator::new_cpu(4, 2, 2, 4)
    }

    #[test]
    fn test_alloc_free_cycle() {
        let mut ba = small_alloc();
        let id0 = ba.alloc().expect("first alloc should succeed");
        let id1 = ba.alloc().expect("second alloc should succeed");
        ba.free(id0);
        // Re-alloc should hand back the freed block (LIFO free list).
        let id2 = ba.alloc().expect("re-alloc should succeed");
        assert_eq!(id2, id0, "re-alloc should return the freed block");
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_pool_exhaustion() {
        let mut ba = BlockAllocator::new_cpu(2, 1, 1, 1);
        assert!(ba.alloc().is_some());
        assert!(ba.alloc().is_some());
        assert!(ba.alloc().is_none(), "pool should be exhausted after 2 allocs");
    }

    #[test]
    fn test_block_pointers_distinct() {
        let ba = small_alloc();
        // Pointers are calculated purely from block_stride, no alloc needed.
        let k0 = unsafe { ba.k_block_ptr(0) };
        let k1 = unsafe { ba.k_block_ptr(1) };
        let v0 = unsafe { ba.v_block_ptr(0) };
        assert_ne!(k0, k1, "k[0] and k[1] must be distinct");
        assert_ne!(k0, v0, "k[0] and v[0] must be distinct (different pools)");
    }

    #[test]
    fn test_utilization() {
        let mut ba = BlockAllocator::new_cpu(4, 1, 1, 1);
        assert_eq!(ba.utilization(), 0.0);
        let _a = ba.alloc();
        let _b = ba.alloc();
        assert_eq!(ba.utilization(), 0.5);
        let _c = ba.alloc();
        let _d = ba.alloc();
        assert_eq!(ba.utilization(), 1.0);
    }

    #[test]
    fn test_cpu_block_read_write() {
        let mut ba = BlockAllocator::new_cpu(4, 2, 2, 4);
        let id = ba.alloc().expect("alloc should succeed");
        // block_stride = 2 * 2 * 4 * 4 = 64 bytes = 16 f32 values
        let floats_per_block = ba.num_heads * ba.block_size * ba.head_dim;
        unsafe {
            let ptr = ba.k_block_ptr(id);
            // Write a known pattern.
            for i in 0..floats_per_block {
                ptr.add(i).write(i as f32 * 0.5);
            }
            // Read it back.
            for i in 0..floats_per_block {
                let val = ptr.add(i).read();
                assert!(
                    (val - i as f32 * 0.5).abs() < 1e-6,
                    "mismatch at index {}: expected {}, got {}",
                    i,
                    i as f32 * 0.5,
                    val
                );
            }
        }
    }
}
