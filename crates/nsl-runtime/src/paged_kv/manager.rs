//! KvCacheManager: high-level API combining BlockAllocator + PageTables
//! for multi-layer, multi-sequence paged KV-cache management.
//!
//! This is the entry point that model codegen calls to manage KV-caches
//! during autoregressive generation.

use std::collections::HashMap;
use std::sync::Mutex;

use crate::paged_kv::block_alloc::BlockAllocator;
use crate::paged_kv::page_table::PageTable;
use crate::paged_kv::{BlockId, SeqId};

/// High-level manager for paged KV-caches across multiple sequences.
///
/// Combines a shared [`BlockAllocator`] with per-sequence [`PageTable`]s,
/// providing a simple API for allocating sequences, appending tokens,
/// and accessing KV storage pointers.
pub struct KvCacheManager {
    allocator: BlockAllocator,
    page_tables: HashMap<SeqId, PageTable>,
    num_layers: usize,
    next_seq_id: SeqId,
}

impl KvCacheManager {
    /// Create a new CPU-backed KV-cache manager.
    ///
    /// # Arguments
    /// - `num_blocks` — total number of blocks in the shared pool
    /// - `block_size` — tokens per block
    /// - `num_heads` — number of attention heads stored per block
    /// - `head_dim` — dimension of each attention head
    /// - `num_layers` — number of transformer layers (stored for future per-layer partitioning)
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        num_layers: usize,
    ) -> Self {
        KvCacheManager {
            allocator: BlockAllocator::new_cpu(num_blocks, block_size, num_heads, head_dim),
            page_tables: HashMap::new(),
            num_layers,
            next_seq_id: 0,
        }
    }

    /// Create a new GPU-backed KV-cache manager (requires the `cuda` feature).
    #[cfg(feature = "cuda")]
    pub fn new_gpu(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        num_layers: usize,
    ) -> Self {
        KvCacheManager {
            allocator: BlockAllocator::new_gpu(num_blocks, block_size, num_heads, head_dim),
            page_tables: HashMap::new(),
            num_layers,
            next_seq_id: 0,
        }
    }

    // ── Sequence lifecycle ────────────────────────────────────────────────────

    /// Allocate a new sequence with an empty page table.
    ///
    /// Returns a unique [`SeqId`] for the new sequence.
    pub fn alloc_sequence(&mut self) -> SeqId {
        let seq_id = self.next_seq_id;
        self.next_seq_id += 1;
        let pt = PageTable::new(self.allocator.block_size);
        self.page_tables.insert(seq_id, pt);
        seq_id
    }

    /// Free all blocks for a sequence and remove its page table.
    ///
    /// All physical blocks are returned to the allocator's free list.
    pub fn free_sequence(&mut self, seq_id: SeqId) {
        if let Some(mut pt) = self.page_tables.remove(&seq_id) {
            let blocks = pt.drain_blocks();
            for block_id in blocks {
                crate::profiling::profiler_record_free(block_id, seq_id);
                self.allocator.free(block_id);
            }
        }
    }

    // ── Token operations ──────────────────────────────────────────────────────

    /// Append a token to a sequence, auto-allocating a new block when needed.
    ///
    /// Returns `(block_id, offset_within_block)` on success.
    /// Returns `Err("KV-cache out of blocks")` if the pool is exhausted.
    ///
    /// # Panics
    /// Panics if `seq_id` does not refer to a valid sequence.
    pub fn append_token(&mut self, seq_id: SeqId) -> Result<(BlockId, usize), &'static str> {
        let pt = self
            .page_tables
            .get_mut(&seq_id)
            .expect("append_token: unknown seq_id");

        if pt.needs_new_block() {
            let block_id = self.allocator.alloc().ok_or("KV-cache out of blocks")?;
            pt.push_block(block_id);
            crate::profiling::profiler_record_alloc(block_id, seq_id);
        }

        Ok(pt.append_token())
    }

    // ── Pointer accessors ─────────────────────────────────────────────────────

    /// Returns `(k_ptr, v_ptr)` for a specific block.
    ///
    /// # Safety
    /// The returned pointers are valid only as long as the block remains allocated.
    /// `block_id` must be a currently-allocated block from this manager.
    pub fn get_kv_ptrs(&self, _seq_id: SeqId, block_id: BlockId) -> (*mut f32, *mut f32) {
        unsafe {
            let k = self.allocator.k_block_ptr(block_id);
            let v = self.allocator.v_block_ptr(block_id);
            (k, v)
        }
    }

    // ── Query methods ─────────────────────────────────────────────────────────

    /// Returns the total number of tokens appended to the given sequence.
    ///
    /// # Panics
    /// Panics if `seq_id` does not refer to a valid sequence.
    pub fn seq_token_count(&self, seq_id: SeqId) -> usize {
        self.page_tables
            .get(&seq_id)
            .expect("seq_token_count: unknown seq_id")
            .token_count()
    }

    /// Returns a slice of all physical block IDs for the given sequence.
    ///
    /// Useful for copying the block table to GPU memory for the attention kernel.
    ///
    /// # Panics
    /// Panics if `seq_id` does not refer to a valid sequence.
    pub fn seq_block_ids(&self, seq_id: SeqId) -> &[BlockId] {
        self.page_tables
            .get(&seq_id)
            .expect("seq_block_ids: unknown seq_id")
            .block_ids()
    }

    /// Returns the number of active sequences.
    pub fn num_sequences(&self) -> usize {
        self.page_tables.len()
    }

    /// Returns the fraction of blocks currently in use, in `[0.0, 1.0]`.
    ///
    /// Delegates to the underlying [`BlockAllocator::utilization`].
    pub fn utilization(&self) -> f64 {
        self.allocator.utilization()
    }

    /// Returns the number of transformer layers (stored for future use).
    pub fn num_layers(&self) -> usize {
        self.num_layers
    }
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// Convert handle (i64) back to `&Mutex<KvCacheManager>`.
///
/// # Safety
/// `handle` must be a valid pointer returned by [`nsl_kv_cache_init`] or
/// [`nsl_kv_cache_init_gpu`].
unsafe fn from_handle(handle: i64) -> &'static Mutex<KvCacheManager> {
    &*(handle as *const Mutex<KvCacheManager>)
}

/// Create a new CPU-backed KV-cache manager and return an opaque handle.
///
/// The `compress_scheme`, `compress_window`, and `compress_sinks` parameters carry the
/// M42 KV compression policy extracted from `@kv_compress` decorators at compile time.
/// `compress_scheme == 0` means no compression (pass-through).  Non-zero values are
/// stored on the manager for use by future compression-aware eviction logic.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_init(
    num_blocks: i64,
    block_size: i64,
    num_heads: i64,
    head_dim: i64,
    num_layers: i64,
    compress_scheme: i64,
    compress_window: i64,
    compress_sinks: i64,
) -> i64 {
    let mgr = KvCacheManager::new(
        num_blocks as usize,
        block_size as usize,
        num_heads as usize,
        head_dim as usize,
        num_layers as usize,
    );
    if compress_scheme > 0 {
        eprintln!(
            "[nsl-runtime] KV cache compression: scheme={}, window={}, sinks={}",
            compress_scheme, compress_window, compress_sinks
        );
        // TODO(M42): plumb compress_scheme/window/sinks into KvCacheManager fields
        // when the compression eviction path is implemented.
    }
    let _ = (compress_scheme, compress_window, compress_sinks);
    let boxed = Box::new(Mutex::new(mgr));
    Box::into_raw(boxed) as i64
}

/// Create a new GPU-backed KV-cache manager and return an opaque handle.
/// Falls back to CPU if the `cuda` feature is not enabled.
///
/// See [`nsl_kv_cache_init`] for the meaning of the `compress_*` parameters.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_init_gpu(
    num_blocks: i64,
    block_size: i64,
    num_heads: i64,
    head_dim: i64,
    num_layers: i64,
    compress_scheme: i64,
    compress_window: i64,
    compress_sinks: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let mgr = KvCacheManager::new_gpu(
            num_blocks as usize,
            block_size as usize,
            num_heads as usize,
            head_dim as usize,
            num_layers as usize,
        );
        if compress_scheme > 0 {
            eprintln!(
                "[nsl-runtime] KV cache compression (GPU): scheme={}, window={}, sinks={}",
                compress_scheme, compress_window, compress_sinks
            );
        }
        let _ = (compress_scheme, compress_window, compress_sinks);
        let boxed = Box::new(Mutex::new(mgr));
        Box::into_raw(boxed) as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!(
            "nsl: nsl_kv_cache_init_gpu called but CUDA feature is disabled, falling back to CPU"
        );
        nsl_kv_cache_init(num_blocks, block_size, num_heads, head_dim, num_layers,
                          compress_scheme, compress_window, compress_sinks)
    }
}

/// Allocate a new sequence and return its [`SeqId`].
#[no_mangle]
pub extern "C" fn nsl_kv_cache_alloc_seq(handle: i64) -> i64 {
    let mgr = unsafe { from_handle(handle) };
    let mut guard = mgr.lock().unwrap();
    guard.alloc_sequence() as i64
}

/// Append a token to a sequence. Returns `(block_id << 32) | offset` on
/// success, or `-1` if the block pool is exhausted.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_append(handle: i64, seq_id: i64) -> i64 {
    let mgr = unsafe { from_handle(handle) };
    let mut guard = mgr.lock().unwrap();
    match guard.append_token(seq_id as u64) {
        Ok((block_id, offset)) => ((block_id as i64) << 32) | (offset as i64),
        Err(e) => {
            eprintln!("nsl: kv_cache_append failed: {}", e);
            -1
        }
    }
}

/// Return the K-cache pointer for a given `(seq_id, block_id)`.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_k_ptr(handle: i64, seq_id: i64, block_id: i64) -> i64 {
    let mgr = unsafe { from_handle(handle) };
    let guard = mgr.lock().unwrap();
    let (k, _) = guard.get_kv_ptrs(seq_id as u64, block_id as u32);
    k as i64
}

/// Return the V-cache pointer for a given `(seq_id, block_id)`.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_v_ptr(handle: i64, seq_id: i64, block_id: i64) -> i64 {
    let mgr = unsafe { from_handle(handle) };
    let guard = mgr.lock().unwrap();
    let (_, v) = guard.get_kv_ptrs(seq_id as u64, block_id as u32);
    v as i64
}

/// Free all blocks for a sequence and remove its page table.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_free_seq(handle: i64, seq_id: i64) {
    let mgr = unsafe { from_handle(handle) };
    let mut guard = mgr.lock().unwrap();
    guard.free_sequence(seq_id as u64);
}

/// Return the number of tokens appended to the given sequence.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_seq_len(handle: i64, seq_id: i64) -> i64 {
    let mgr = unsafe { from_handle(handle) };
    let guard = mgr.lock().unwrap();
    guard.seq_token_count(seq_id as u64) as i64
}

/// Return a raw pointer to the block-ID array for the given sequence.
///
/// # Safety
/// The returned pointer points into the internal `Vec<BlockId>`.  The Mutex
/// guard is released when this function returns, so the pointer is only
/// safe to dereference while no concurrent mutations (append_token,
/// free_sequence) occur.  In single-threaded NSL programs this is always
/// satisfied.  The caller must use [`nsl_kv_cache_seq_num_blocks`] to
/// determine the length, and must NOT store this pointer across calls that
/// mutate the same sequence.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_seq_blocks(handle: i64, seq_id: i64) -> i64 {
    let mgr = unsafe { from_handle(handle) };
    let guard = mgr.lock().unwrap();
    guard.seq_block_ids(seq_id as u64).as_ptr() as i64
}

/// Return the number of physical blocks allocated for the given sequence.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_seq_num_blocks(handle: i64, seq_id: i64) -> i64 {
    let mgr = unsafe { from_handle(handle) };
    let guard = mgr.lock().unwrap();
    guard.seq_block_ids(seq_id as u64).len() as i64
}

/// Return the fraction of blocks currently in use, in `[0.0, 1.0]`.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_utilization(handle: i64) -> f64 {
    let mgr = unsafe { from_handle(handle) };
    let guard = mgr.lock().unwrap();
    guard.utilization()
}

/// Destroy a KV-cache manager, freeing all memory.
///
/// # Safety
/// `handle` must be a valid pointer returned by [`nsl_kv_cache_init`] and
/// must not be used after this call.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_destroy(handle: i64) {
    if handle == 0 {
        return;
    }
    // Reconstruct the Box and let it drop, which runs BlockAllocator::drop
    // to free the backing memory pools.
    let _ = unsafe { Box::from_raw(handle as *mut Mutex<KvCacheManager>) };
}

// ── Unit tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: create a small manager with `num_blocks` blocks of size `block_size`.
    fn small_manager(num_blocks: usize, block_size: usize) -> KvCacheManager {
        // num_heads=2, head_dim=4, num_layers=4
        KvCacheManager::new(num_blocks, block_size, 2, 4, 4)
    }

    #[test]
    fn test_alloc_append_free() {
        let mut mgr = small_manager(8, 4);
        assert_eq!(mgr.utilization(), 0.0);

        let seq = mgr.alloc_sequence();

        // Append 6 tokens (block_size=4, so this spans 2 blocks).
        for _ in 0..6 {
            mgr.append_token(seq).expect("append should succeed");
        }
        assert_eq!(mgr.seq_token_count(seq), 6);
        assert!(mgr.utilization() > 0.0, "utilization should be non-zero after appending");

        // Free the sequence.
        mgr.free_sequence(seq);
        assert_eq!(mgr.utilization(), 0.0, "utilization should return to 0 after freeing");
        assert_eq!(mgr.num_sequences(), 0);
    }

    #[test]
    fn test_multi_sequence() {
        let mut mgr = small_manager(16, 4);

        let seq_a = mgr.alloc_sequence();
        let seq_b = mgr.alloc_sequence();
        assert_eq!(mgr.num_sequences(), 2);

        // Append different numbers of tokens to each sequence.
        for _ in 0..3 {
            mgr.append_token(seq_a).expect("append to seq_a");
        }
        for _ in 0..7 {
            mgr.append_token(seq_b).expect("append to seq_b");
        }

        assert_eq!(mgr.seq_token_count(seq_a), 3);
        assert_eq!(mgr.seq_token_count(seq_b), 7);

        // Free one; the other should remain unaffected.
        mgr.free_sequence(seq_a);
        assert_eq!(mgr.num_sequences(), 1);
        assert_eq!(mgr.seq_token_count(seq_b), 7);
    }

    #[test]
    fn test_pool_exhaustion() {
        // 2 blocks of size 2 => 4 tokens max.
        let mut mgr = small_manager(2, 2);
        let seq = mgr.alloc_sequence();

        // Fill all 4 slots across 2 blocks.
        for _ in 0..4 {
            mgr.append_token(seq).expect("append within capacity");
        }

        // Next append should fail — all blocks are allocated.
        let result = mgr.append_token(seq);
        assert!(result.is_err(), "expected Err when pool is exhausted");
        assert_eq!(result.unwrap_err(), "KV-cache out of blocks");
    }

    #[test]
    fn test_kv_ptrs_read_write() {
        let mut mgr = small_manager(4, 4);
        let seq = mgr.alloc_sequence();

        let (block_id, offset) = mgr.append_token(seq).expect("append");
        assert_eq!(offset, 0);

        let (k_ptr, v_ptr) = mgr.get_kv_ptrs(seq, block_id);

        // Write through the raw pointers and verify read-back.
        unsafe {
            k_ptr.write(42.0_f32);
            v_ptr.write(99.5_f32);
            assert!(
                (k_ptr.read() - 42.0).abs() < 1e-6,
                "k_ptr round-trip failed"
            );
            assert!(
                (v_ptr.read() - 99.5).abs() < 1e-6,
                "v_ptr round-trip failed"
            );
        }
    }

    #[test]
    fn test_block_ids_for_sync() {
        // block_size=2 so we need 3 blocks for 5 tokens.
        let mut mgr = small_manager(8, 2);
        let seq = mgr.alloc_sequence();

        for _ in 0..5 {
            mgr.append_token(seq).expect("append");
        }

        let ids = mgr.seq_block_ids(seq);
        assert_eq!(ids.len(), 3, "5 tokens with block_size=2 should span 3 blocks");

        // Block IDs should be distinct.
        let mut sorted = ids.to_vec();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), 3, "all block IDs should be distinct");
    }
}
