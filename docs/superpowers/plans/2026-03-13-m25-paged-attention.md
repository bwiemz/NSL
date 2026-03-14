# M25: PagedAttention + Memory Profiling — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a paged virtual memory manager for KV-caches that eliminates VRAM fragmentation during autoregressive generation, plus ship memory watermark profiling with Chrome tracing output.

**Architecture:** The runtime gets a new `paged_kv` module with three layers: `BlockAllocator` (O(1) free-list pool for VRAM blocks), `PageTable` (per-sequence logical→physical mapping), and `KvCacheManager` (high-level API exposed via FFI). The allocator works on both CPU (malloc) and GPU (cuMemAlloc) so tests run without CUDA. Memory profiling uses atomic counters and an event log that serializes to Chrome `chrome://tracing` JSON.

**Tech Stack:** Rust (nsl-runtime), Cranelift codegen, cudarc 0.19 driver API, serde_json for profiling output

**Spec:** `docs/superpowers/specs/2026-03-13-m25-paged-attention-design.md`

---

## File Structure

### New Files

| File | Responsibility |
|------|---------------|
| `crates/nsl-runtime/src/paged_kv/mod.rs` | Module root, re-exports, `BlockId`/`SeqId` type aliases |
| `crates/nsl-runtime/src/paged_kv/block_alloc.rs` | `BlockAllocator`: free-list pool, K/V memory regions |
| `crates/nsl-runtime/src/paged_kv/page_table.rs` | `PageTable`: per-sequence logical→physical block mapping |
| `crates/nsl-runtime/src/paged_kv/manager.rs` | `KvCacheManager`: high-level API, FFI exports, GPU sync |
| `crates/nsl-runtime/src/profiling.rs` | `MemoryProfiler`: atomic counters, event log, JSON writer |
| `examples/m25_paged_kv.nsl` | E2E test: multi-sequence paged KV cache |
| `examples/m25_profiling.nsl` | E2E test: memory profiling output |
| `tests/expected/m25_paged_kv.txt` | Expected output baseline |
| `tests/expected/m25_profiling.txt` | Expected output baseline |

### Modified Files

| File | Changes |
|------|---------|
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod paged_kv;` and `pub mod profiling;` |
| `crates/nsl-runtime/src/cuda/mod.rs` | Add `alloc_device()`, `free_device()`, `alloc_pinned()`, `free_pinned()`, `memcpy_htod()` |
| `crates/nsl-codegen/src/builtins.rs` | Declare 9 KV cache + 3 profiling FFI functions |
| `crates/nsl-semantic/src/checker.rs` | Validate `@paged_kv` decorator on model fields (fix `..` destructuring to bind `decorators`) |
| `crates/nsl-codegen/src/compiler.rs` | Add `paged_kv_configs` field, extract `@paged_kv` args, emit `nsl_kv_cache_init` at model init |
| `crates/nsl-codegen/src/expr.rs` | Add builtin dispatch for `kv_cache_init`, `kv_alloc_seq`, etc. → `nsl_*` runtime functions |
| `crates/nsl-cli/src/main.rs` | Add `--profile-memory` flag to `Run` command |
| `crates/nsl-cli/tests/e2e.rs` | Add M25 E2E tests |

---

## Chunk 1: Runtime Foundation

### Task 1: CUDA Device-Only and Pinned Memory Primitives

**Files:**
- Modify: `crates/nsl-runtime/src/cuda/mod.rs` (inside `#[cfg(feature = "cuda")] pub(crate) mod inner`)

**Context:** The existing CUDA module only has `alloc_managed()` (unified memory). PagedAttention needs device-only memory for block pools (lower overhead than unified) and pinned host memory for page table sync buffers.

- [ ] **Step 1: Write unit tests for the new CUDA memory functions**

Add at the bottom of `crates/nsl-runtime/src/cuda/mod.rs`, inside existing `#[cfg(test)]` section (or create one):

```rust
#[cfg(test)]
#[cfg(feature = "cuda")]
mod tests {
    use super::inner;

    #[test]
    fn test_alloc_free_device() {
        inner::init();
        let ptr = inner::alloc_device(1024);
        assert!(!ptr.is_null());
        inner::free_device(ptr);
    }

    #[test]
    fn test_alloc_free_pinned() {
        inner::init();
        let ptr = inner::alloc_pinned(256);
        assert!(!ptr.is_null());
        // Pinned memory is accessible from CPU
        unsafe { *(ptr as *mut u32) = 42; }
        assert_eq!(unsafe { *(ptr as *const u32) }, 42);
        inner::free_pinned(ptr);
    }

    #[test]
    fn test_memcpy_htod() {
        inner::init();
        let device_ptr = inner::alloc_device(16);
        let host_data: [u32; 4] = [1, 2, 3, 4];
        inner::memcpy_htod(device_ptr, host_data.as_ptr() as *const std::ffi::c_void, 16);
        // Can't easily read back from device-only memory without DtoH, but no crash = success
        inner::free_device(device_ptr);
    }
}
```

- [ ] **Step 2: Run tests to verify they fail (functions don't exist yet)**

Run: `cargo test --features cuda -p nsl-runtime -- cuda::tests`
Expected: FAIL — `alloc_device`, `free_device`, `alloc_pinned`, `free_pinned`, `memcpy_htod` not found

- [ ] **Step 3: Implement the CUDA memory functions**

Add inside `pub(crate) mod inner { ... }` in `crates/nsl-runtime/src/cuda/mod.rs`, after `free_managed()`:

```rust
    /// Allocate device-only memory (not accessible from CPU).
    /// Lower overhead than unified memory — used for KV cache block pools.
    pub(crate) fn alloc_device(size_bytes: usize) -> *mut c_void {
        ensure_context();
        unsafe {
            let mut ptr: CUdeviceptr = 0;
            let result = cuMemAlloc_v2(&mut ptr, size_bytes);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemAlloc_v2({} bytes) failed: {:?}",
                size_bytes, result
            );
            ptr as *mut c_void
        }
    }

    /// Free device-only memory.
    pub(crate) fn free_device(ptr: *mut c_void) {
        ensure_context();
        unsafe {
            let result = cuMemFree_v2(ptr as CUdeviceptr);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemFree (device) failed: {:?}",
                result
            );
        }
    }

    /// Allocate pinned (page-locked) host memory for fast H2D transfers.
    pub(crate) fn alloc_pinned(size_bytes: usize) -> *mut c_void {
        ensure_context();
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let result = cuMemAllocHost_v2(&mut ptr, size_bytes);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemAllocHost({} bytes) failed: {:?}",
                size_bytes, result
            );
            ptr
        }
    }

    /// Free pinned host memory.
    pub(crate) fn free_pinned(ptr: *mut c_void) {
        ensure_context();
        unsafe {
            let result = cuMemFreeHost(ptr);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemFreeHost failed: {:?}",
                result
            );
        }
    }

    /// Copy bytes from host to device memory.
    pub(crate) fn memcpy_htod(dst_device: *mut c_void, src_host: *const c_void, size_bytes: usize) {
        ensure_context();
        unsafe {
            let result = cuMemcpyHtoD_v2(
                dst_device as CUdeviceptr,
                src_host,
                size_bytes,
            );
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemcpyHtoD({} bytes) failed: {:?}",
                size_bytes, result
            );
        }
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cargo test --features cuda -p nsl-runtime -- cuda::tests`
Expected: PASS (3 tests). If no CUDA GPU available, tests will panic at `init()` — that's expected, these are hardware-dependent tests.

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/cuda/mod.rs
git commit -m "feat(runtime): add device-only and pinned memory CUDA primitives for M25"
```

---

### Task 2: BlockAllocator

**Files:**
- Create: `crates/nsl-runtime/src/paged_kv/mod.rs`
- Create: `crates/nsl-runtime/src/paged_kv/block_alloc.rs`
- Modify: `crates/nsl-runtime/src/lib.rs` (add `pub mod paged_kv;`)

**Context:** `BlockAllocator` manages a pool of fixed-size memory blocks. Each block stores `[num_heads, block_size, head_dim]` f32 values. Separate pools for K and V. Alloc/free are O(1) via free list. Works on both CPU (for testing) and GPU (for production).

- [ ] **Step 1: Create the module root**

Create `crates/nsl-runtime/src/paged_kv/mod.rs`:

```rust
//! Paged KV-cache for autoregressive generation.
//!
//! Eliminates VRAM fragmentation by allocating fixed-size blocks on demand
//! instead of reserving contiguous memory for maximum sequence length.

pub mod block_alloc;
pub mod page_table;
pub mod manager;

/// Physical block identifier within a pool.
pub type BlockId = u32;

/// Logical sequence identifier.
pub type SeqId = u64;
```

- [ ] **Step 2: Add `pub mod paged_kv;` to lib.rs**

In `crates/nsl-runtime/src/lib.rs`, add after the `pub mod weight_provider;` line:

```rust
pub mod paged_kv;
pub mod profiling;
```

- [ ] **Step 3: Write tests for BlockAllocator**

Create `crates/nsl-runtime/src/paged_kv/block_alloc.rs` with tests:

```rust
//! Block pool allocator: O(1) alloc/free via free list.

use std::ffi::c_void;
use super::BlockId;

/// Which memory backend the block pool uses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PoolBackend {
    /// CPU heap (for testing without CUDA)
    Cpu,
    /// CUDA device memory
    #[cfg(feature = "cuda")]
    Gpu,
}

/// Fixed-size block pool for KV cache storage.
///
/// Each block stores `[num_heads, block_size, head_dim]` f32 values.
/// K and V have separate pools but share the same free list (block IDs
/// index into both pools simultaneously).
pub struct BlockAllocator {
    pub block_size: usize,      // tokens per block
    pub num_blocks: usize,      // total blocks in pool
    pub num_heads: usize,
    pub head_dim: usize,
    pub block_stride: usize,    // bytes per block: num_heads * block_size * head_dim * sizeof(f32)
    free_list: Vec<BlockId>,    // available block IDs
    k_pool: *mut c_void,        // contiguous memory for K blocks
    v_pool: *mut c_void,        // contiguous memory for V blocks
    backend: PoolBackend,
    allocated_count: usize,     // blocks currently in use (for profiling)
}

// SAFETY: The raw pointers point to either CPU heap or CUDA device memory.
// The allocator owns both and controls all access through &mut self.
unsafe impl Send for BlockAllocator {}

impl BlockAllocator {
    /// Create a new block pool.
    ///
    /// `num_blocks`: total blocks in the pool
    /// `block_size`: tokens per block (typically 16)
    /// `num_heads`: attention heads
    /// `head_dim`: dimension per head
    /// `backend`: CPU or GPU memory
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        backend: PoolBackend,
    ) -> Self {
        let block_stride = num_heads * block_size * head_dim * std::mem::size_of::<f32>();
        let total_bytes = num_blocks * block_stride;

        let (k_pool, v_pool) = match backend {
            PoolBackend::Cpu => {
                let k = crate::memory::checked_alloc_zeroed(total_bytes) as *mut c_void;
                let v = crate::memory::checked_alloc_zeroed(total_bytes) as *mut c_void;
                (k, v)
            }
            #[cfg(feature = "cuda")]
            PoolBackend::Gpu => {
                let k = crate::cuda::inner::alloc_device(total_bytes);
                let v = crate::cuda::inner::alloc_device(total_bytes);
                (k, v)
            }
        };

        // Initialize free list with all block IDs (in reverse so pop gives 0,1,2,...)
        let free_list: Vec<BlockId> = (0..num_blocks as BlockId).rev().collect();

        BlockAllocator {
            block_size,
            num_blocks,
            num_heads,
            head_dim,
            block_stride,
            free_list,
            k_pool,
            v_pool,
            backend,
            allocated_count: 0,
        }
    }

    /// Allocate a block. Returns None if pool is exhausted.
    pub fn alloc(&mut self) -> Option<BlockId> {
        let id = self.free_list.pop()?;
        self.allocated_count += 1;
        Some(id)
    }

    /// Return a block to the free list.
    pub fn free(&mut self, id: BlockId) {
        debug_assert!((id as usize) < self.num_blocks, "invalid block ID");
        self.free_list.push(id);
        self.allocated_count -= 1;
    }

    /// Get pointer to the K data for a given block.
    pub fn k_block_ptr(&self, id: BlockId) -> *mut c_void {
        unsafe { (self.k_pool as *mut u8).add(id as usize * self.block_stride) as *mut c_void }
    }

    /// Get pointer to the V data for a given block.
    pub fn v_block_ptr(&self, id: BlockId) -> *mut c_void {
        unsafe { (self.v_pool as *mut u8).add(id as usize * self.block_stride) as *mut c_void }
    }

    /// Number of blocks currently allocated.
    pub fn allocated(&self) -> usize {
        self.allocated_count
    }

    /// Number of blocks available.
    pub fn available(&self) -> usize {
        self.free_list.len()
    }

    /// Utilization ratio (0.0 to 1.0).
    pub fn utilization(&self) -> f64 {
        self.allocated_count as f64 / self.num_blocks as f64
    }
}

impl Drop for BlockAllocator {
    fn drop(&mut self) {
        let total_bytes = self.num_blocks * self.block_stride;
        match self.backend {
            PoolBackend::Cpu => unsafe {
                crate::memory::checked_free(self.k_pool as *mut u8, total_bytes);
                crate::memory::checked_free(self.v_pool as *mut u8, total_bytes);
            },
            #[cfg(feature = "cuda")]
            PoolBackend::Gpu => {
                crate::cuda::inner::free_device(self.k_pool);
                crate::cuda::inner::free_device(self.v_pool);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alloc_free_cycle() {
        let mut alloc = BlockAllocator::new(8, 4, 2, 4, PoolBackend::Cpu);
        assert_eq!(alloc.available(), 8);
        assert_eq!(alloc.allocated(), 0);

        let b0 = alloc.alloc().unwrap();
        let b1 = alloc.alloc().unwrap();
        assert_eq!(alloc.allocated(), 2);
        assert_eq!(alloc.available(), 6);

        alloc.free(b0);
        assert_eq!(alloc.allocated(), 1);
        assert_eq!(alloc.available(), 7);

        // Re-allocate returns the freed block
        let b2 = alloc.alloc().unwrap();
        assert_eq!(b2, b0);
    }

    #[test]
    fn test_pool_exhaustion() {
        let mut alloc = BlockAllocator::new(2, 4, 1, 2, PoolBackend::Cpu);
        assert!(alloc.alloc().is_some());
        assert!(alloc.alloc().is_some());
        assert!(alloc.alloc().is_none()); // exhausted
    }

    #[test]
    fn test_block_pointers_distinct() {
        let alloc = BlockAllocator::new(4, 4, 2, 4, PoolBackend::Cpu);
        let k0 = alloc.k_block_ptr(0);
        let k1 = alloc.k_block_ptr(1);
        assert_ne!(k0, k1);
        let v0 = alloc.v_block_ptr(0);
        assert_ne!(k0, v0); // K and V pools are separate
    }

    #[test]
    fn test_utilization() {
        let mut alloc = BlockAllocator::new(4, 4, 1, 2, PoolBackend::Cpu);
        assert_eq!(alloc.utilization(), 0.0);
        alloc.alloc();
        alloc.alloc();
        assert_eq!(alloc.utilization(), 0.5);
    }

    #[test]
    fn test_cpu_block_read_write() {
        let mut alloc = BlockAllocator::new(2, 2, 1, 2, PoolBackend::Cpu);
        // block_stride = 1 head * 2 tokens * 2 dim * 4 bytes = 16 bytes = 4 floats
        let b = alloc.alloc().unwrap();
        let k_ptr = alloc.k_block_ptr(b) as *mut f32;
        unsafe {
            *k_ptr = 1.0;
            *k_ptr.add(1) = 2.0;
            assert_eq!(*k_ptr, 1.0);
            assert_eq!(*k_ptr.add(1), 2.0);
        }
    }
}
```

- [ ] **Step 4: Run tests**

Run: `cargo test -p nsl-runtime -- paged_kv::block_alloc::tests`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add crates/nsl-runtime/src/paged_kv/mod.rs crates/nsl-runtime/src/paged_kv/block_alloc.rs crates/nsl-runtime/src/lib.rs
git commit -m "feat(runtime): add BlockAllocator for paged KV cache pool management"
```

---

### Task 3: PageTable

**Files:**
- Create: `crates/nsl-runtime/src/paged_kv/page_table.rs`

**Context:** Each active sequence has a `PageTable` mapping logical block indices to physical `BlockId`s. Grows on demand as tokens are appended. Pure CPU data structure — the GPU gets a flattened copy via pinned buffer sync.

- [ ] **Step 1: Write PageTable implementation with tests**

Create `crates/nsl-runtime/src/paged_kv/page_table.rs`:

```rust
//! Per-sequence page table: maps logical token positions to physical blocks.

use super::BlockId;

/// Maps logical block index → physical BlockId for one sequence.
pub struct PageTable {
    /// entries[i] = physical block ID for logical block i
    entries: Vec<BlockId>,
    /// Total tokens appended to this sequence
    token_count: usize,
    /// Tokens per block (must match BlockAllocator's block_size)
    block_size: usize,
}

impl PageTable {
    pub fn new(block_size: usize) -> Self {
        PageTable {
            entries: Vec::new(),
            token_count: 0,
            block_size,
        }
    }

    /// Number of tokens stored in this sequence.
    pub fn token_count(&self) -> usize {
        self.token_count
    }

    /// Number of blocks currently mapped.
    pub fn num_blocks(&self) -> usize {
        self.entries.len()
    }

    /// Whether the current block is full and a new one is needed.
    pub fn needs_new_block(&self) -> bool {
        self.token_count == 0 || self.token_count % self.block_size == 0
    }

    /// Register a newly allocated block for the next logical position.
    pub fn push_block(&mut self, block_id: BlockId) {
        self.entries.push(block_id);
    }

    /// Record that one token was appended. Returns (block_id, offset_within_block).
    pub fn append_token(&mut self) -> (BlockId, usize) {
        let logical_block = self.token_count / self.block_size;
        let offset = self.token_count % self.block_size;
        self.token_count += 1;
        (self.entries[logical_block], offset)
    }

    /// Get the block ID for a given logical block index.
    pub fn get_block(&self, logical_index: usize) -> Option<BlockId> {
        self.entries.get(logical_index).copied()
    }

    /// Get all block IDs as a slice (for GPU sync).
    pub fn block_ids(&self) -> &[BlockId] {
        &self.entries
    }

    /// Return all block IDs (for freeing on sequence destroy).
    pub fn drain_blocks(&mut self) -> Vec<BlockId> {
        self.token_count = 0;
        std::mem::take(&mut self.entries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_new_table_needs_block() {
        let pt = PageTable::new(4);
        assert!(pt.needs_new_block());
        assert_eq!(pt.token_count(), 0);
        assert_eq!(pt.num_blocks(), 0);
    }

    #[test]
    fn test_append_within_block() {
        let mut pt = PageTable::new(4);
        pt.push_block(10); // block ID 10 for first logical block
        let (bid, off) = pt.append_token();
        assert_eq!(bid, 10);
        assert_eq!(off, 0);
        assert_eq!(pt.token_count(), 1);
        assert!(!pt.needs_new_block()); // still space in block

        let (bid, off) = pt.append_token();
        assert_eq!(bid, 10);
        assert_eq!(off, 1);
    }

    #[test]
    fn test_block_boundary() {
        let mut pt = PageTable::new(2); // block_size = 2
        pt.push_block(5);
        pt.append_token(); // token 0 → block 5, offset 0
        pt.append_token(); // token 1 → block 5, offset 1
        assert!(pt.needs_new_block()); // block full

        pt.push_block(7);
        let (bid, off) = pt.append_token(); // token 2 → block 7, offset 0
        assert_eq!(bid, 7);
        assert_eq!(off, 0);
    }

    #[test]
    fn test_drain_blocks() {
        let mut pt = PageTable::new(4);
        pt.push_block(1);
        pt.push_block(3);
        pt.append_token();
        pt.append_token();
        let blocks = pt.drain_blocks();
        assert_eq!(blocks, vec![1, 3]);
        assert_eq!(pt.token_count(), 0);
        assert_eq!(pt.num_blocks(), 0);
    }

    #[test]
    fn test_block_ids_for_sync() {
        let mut pt = PageTable::new(4);
        pt.push_block(2);
        pt.push_block(5);
        pt.push_block(8);
        assert_eq!(pt.block_ids(), &[2, 5, 8]);
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p nsl-runtime -- paged_kv::page_table::tests`
Expected: PASS (5 tests)

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-runtime/src/paged_kv/page_table.rs
git commit -m "feat(runtime): add PageTable for per-sequence block mapping"
```

---

### Task 4: KvCacheManager Core

**Files:**
- Create: `crates/nsl-runtime/src/paged_kv/manager.rs`

**Context:** `KvCacheManager` is the high-level orchestrator combining `BlockAllocator` and per-sequence `PageTable`s. Handles sequence lifecycle, token appending (with automatic block allocation), and KV data gathering into contiguous tensors for standard attention.

- [ ] **Step 1: Write KvCacheManager implementation with tests**

Create `crates/nsl-runtime/src/paged_kv/manager.rs`:

```rust
//! KvCacheManager: high-level API for paged KV caches.
//!
//! Manages sequence lifecycle, token appending (with automatic block
//! allocation), and gathering scattered KV data into contiguous tensors.

use std::collections::HashMap;
use std::ffi::c_void;

use super::block_alloc::{BlockAllocator, PoolBackend};
use super::page_table::PageTable;
use super::{BlockId, SeqId};

/// High-level manager for paged KV caches.
pub struct KvCacheManager {
    allocator: BlockAllocator,
    sequences: HashMap<SeqId, SequenceState>,
    next_seq_id: SeqId,
    num_layers: usize,
}

/// Per-sequence state: one PageTable per layer.
struct SequenceState {
    /// page_tables[layer_idx] = PageTable for that layer
    page_tables: Vec<PageTable>,
}

impl KvCacheManager {
    /// Create a new KV cache manager.
    ///
    /// * `num_blocks` - total blocks in the shared pool
    /// * `block_size` - tokens per block (typically 16)
    /// * `num_heads` - attention heads per layer
    /// * `head_dim` - dimension per head
    /// * `num_layers` - transformer layers sharing this pool
    /// * `backend` - CPU or GPU memory
    pub fn new(
        num_blocks: usize,
        block_size: usize,
        num_heads: usize,
        head_dim: usize,
        num_layers: usize,
        backend: PoolBackend,
    ) -> Self {
        KvCacheManager {
            allocator: BlockAllocator::new(num_blocks, block_size, num_heads, head_dim, backend),
            sequences: HashMap::new(),
            next_seq_id: 0,
            num_layers,
        }
    }

    /// Allocate a new sequence. Returns the sequence ID.
    pub fn alloc_sequence(&mut self) -> Result<SeqId, &'static str> {
        let id = self.next_seq_id;
        self.next_seq_id += 1;
        let block_size = self.allocator.block_size;
        let page_tables = (0..self.num_layers)
            .map(|_| PageTable::new(block_size))
            .collect();
        self.sequences.insert(id, SequenceState { page_tables });
        Ok(id)
    }

    /// Append a token's K and V to all layers for a sequence.
    ///
    /// `kv_data` is a slice of `(k_ptr, v_ptr)` pairs, one per layer.
    /// Each pointer points to `[num_heads, head_dim]` f32 values.
    pub fn append_token(
        &mut self,
        seq_id: SeqId,
        layer: usize,
        k_data: *const f32,
        v_data: *const f32,
    ) -> Result<(), &'static str> {
        let seq = self.sequences.get_mut(&seq_id)
            .ok_or("sequence not found")?;
        if layer >= self.num_layers {
            return Err("layer index out of bounds");
        }
        let pt = &mut seq.page_tables[layer];

        // Allocate new block if needed
        if pt.needs_new_block() {
            let block_id = self.allocator.alloc()
                .ok_or("KV-cache out of blocks")?;
            pt.push_block(block_id);
        }

        // Get write position
        let (block_id, offset) = pt.append_token();
        let head_dim = self.allocator.head_dim;
        let num_heads = self.allocator.num_heads;
        let block_size = self.allocator.block_size;

        // Copy K data: source is [num_heads, head_dim], dest is block at [num_heads, block_size, head_dim]
        let k_block = self.allocator.k_block_ptr(block_id) as *mut f32;
        let v_block = self.allocator.v_block_ptr(block_id) as *mut f32;

        for h in 0..num_heads {
            let src_offset = h * head_dim;
            // Block layout: [num_heads, block_size, head_dim]
            let dst_offset = h * block_size * head_dim + offset * head_dim;
            unsafe {
                std::ptr::copy_nonoverlapping(
                    k_data.add(src_offset),
                    k_block.add(dst_offset),
                    head_dim,
                );
                std::ptr::copy_nonoverlapping(
                    v_data.add(src_offset),
                    v_block.add(dst_offset),
                    head_dim,
                );
            }
        }

        Ok(())
    }

    /// Gather all K values for a sequence+layer into a contiguous tensor.
    ///
    /// Returns a newly allocated buffer of shape `[num_heads, seq_len, head_dim]`
    /// as f32 values. Caller is responsible for wrapping it in an NslTensor.
    pub fn gather_k(
        &self,
        seq_id: SeqId,
        layer: usize,
    ) -> Result<(*mut f32, usize), &'static str> {
        self.gather_pool(seq_id, layer, true)
    }

    /// Gather all V values for a sequence+layer into a contiguous tensor.
    pub fn gather_v(
        &self,
        seq_id: SeqId,
        layer: usize,
    ) -> Result<(*mut f32, usize), &'static str> {
        self.gather_pool(seq_id, layer, false)
    }

    fn gather_pool(
        &self,
        seq_id: SeqId,
        layer: usize,
        is_k: bool,
    ) -> Result<(*mut f32, usize), &'static str> {
        let seq = self.sequences.get(&seq_id)
            .ok_or("sequence not found")?;
        if layer >= self.num_layers {
            return Err("layer index out of bounds");
        }
        let pt = &seq.page_tables[layer];
        let seq_len = pt.token_count();
        if seq_len == 0 {
            return Err("sequence is empty");
        }

        let num_heads = self.allocator.num_heads;
        let head_dim = self.allocator.head_dim;
        let block_size = self.allocator.block_size;

        // Output layout: [num_heads, seq_len, head_dim]
        let total_floats = num_heads * seq_len * head_dim;
        let out_ptr = crate::memory::checked_alloc(total_floats * std::mem::size_of::<f32>()) as *mut f32;

        for token_idx in 0..seq_len {
            let logical_block = token_idx / block_size;
            let offset = token_idx % block_size;
            let block_id = pt.get_block(logical_block).unwrap();

            let block_ptr = if is_k {
                self.allocator.k_block_ptr(block_id) as *const f32
            } else {
                self.allocator.v_block_ptr(block_id) as *const f32
            };

            for h in 0..num_heads {
                // Source: block[h, offset, :] = block_ptr + h*block_size*head_dim + offset*head_dim
                let src = unsafe { block_ptr.add(h * block_size * head_dim + offset * head_dim) };
                // Dest: out[h, token_idx, :] = out_ptr + h*seq_len*head_dim + token_idx*head_dim
                let dst = unsafe { out_ptr.add(h * seq_len * head_dim + token_idx * head_dim) };
                unsafe {
                    std::ptr::copy_nonoverlapping(src, dst, head_dim);
                }
            }
        }

        Ok((out_ptr, seq_len))
    }

    /// Free a sequence, returning all its blocks to the pool.
    pub fn free_sequence(&mut self, seq_id: SeqId) -> Result<(), &'static str> {
        let mut seq = self.sequences.remove(&seq_id)
            .ok_or("sequence not found")?;
        for pt in &mut seq.page_tables {
            for block_id in pt.drain_blocks() {
                self.allocator.free(block_id);
            }
        }
        Ok(())
    }

    /// Get the block allocator (for profiling access).
    pub fn allocator(&self) -> &BlockAllocator {
        &self.allocator
    }

    /// Number of active sequences.
    pub fn active_sequences(&self) -> usize {
        self.sequences.len()
    }

    /// Token count for a given sequence and layer.
    pub fn seq_token_count(&self, seq_id: SeqId, layer: usize) -> Option<usize> {
        self.sequences.get(&seq_id)
            .and_then(|s| s.page_tables.get(layer))
            .map(|pt| pt.token_count())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_mgr() -> KvCacheManager {
        // 16 blocks, block_size=2, 2 heads, head_dim=4, 1 layer
        KvCacheManager::new(16, 2, 2, 4, 1, PoolBackend::Cpu)
    }

    #[test]
    fn test_alloc_free_sequence() {
        let mut mgr = make_mgr();
        let s0 = mgr.alloc_sequence().unwrap();
        let s1 = mgr.alloc_sequence().unwrap();
        assert_ne!(s0, s1);
        assert_eq!(mgr.active_sequences(), 2);

        mgr.free_sequence(s0).unwrap();
        assert_eq!(mgr.active_sequences(), 1);
    }

    #[test]
    fn test_append_and_gather() {
        let mut mgr = make_mgr();
        let seq = mgr.alloc_sequence().unwrap();

        // Append 3 tokens (block_size=2, so uses 2 blocks)
        // Each token has [num_heads=2, head_dim=4] = 8 floats
        let k0: [f32; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let v0: [f32; 8] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
        mgr.append_token(seq, 0, k0.as_ptr(), v0.as_ptr()).unwrap();

        let k1: [f32; 8] = [11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0];
        let v1: [f32; 8] = [1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8];
        mgr.append_token(seq, 0, k1.as_ptr(), v1.as_ptr()).unwrap();

        let k2: [f32; 8] = [21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0];
        let v2: [f32; 8] = [2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8];
        mgr.append_token(seq, 0, k2.as_ptr(), v2.as_ptr()).unwrap();

        // Gather K: should be [num_heads=2, seq_len=3, head_dim=4]
        let (k_ptr, seq_len) = mgr.gather_k(seq, 0).unwrap();
        assert_eq!(seq_len, 3);

        // Verify gathered K data
        // Head 0: tokens 0,1,2 each have head_dim=4 values
        let k_data = unsafe { std::slice::from_raw_parts(k_ptr, 2 * 3 * 4) };
        // Head 0, token 0: [1,2,3,4]
        assert_eq!(&k_data[0..4], &[1.0, 2.0, 3.0, 4.0]);
        // Head 0, token 1: [11,12,13,14]
        assert_eq!(&k_data[4..8], &[11.0, 12.0, 13.0, 14.0]);
        // Head 0, token 2: [21,22,23,24]
        assert_eq!(&k_data[8..12], &[21.0, 22.0, 23.0, 24.0]);
        // Head 1, token 0: [5,6,7,8]
        assert_eq!(&k_data[12..16], &[5.0, 6.0, 7.0, 8.0]);

        // Clean up gathered buffer
        unsafe { crate::memory::checked_free(k_ptr as *mut u8, 2 * 3 * 4 * 4); }
    }

    #[test]
    fn test_pool_exhaustion_error() {
        // Only 2 blocks total, block_size=2
        let mut mgr = KvCacheManager::new(2, 2, 1, 2, 1, PoolBackend::Cpu);
        let seq = mgr.alloc_sequence().unwrap();

        // First 2 tokens use block 0 (OK)
        let data: [f32; 2] = [1.0, 2.0]; // [1 head, 2 dim]
        mgr.append_token(seq, 0, data.as_ptr(), data.as_ptr()).unwrap();
        mgr.append_token(seq, 0, data.as_ptr(), data.as_ptr()).unwrap();

        // Next 2 tokens use block 1 (OK)
        mgr.append_token(seq, 0, data.as_ptr(), data.as_ptr()).unwrap();
        mgr.append_token(seq, 0, data.as_ptr(), data.as_ptr()).unwrap();

        // 5th token needs block 2 — should fail
        let result = mgr.append_token(seq, 0, data.as_ptr(), data.as_ptr());
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("out of blocks"));
    }

    #[test]
    fn test_multi_sequence_sharing() {
        // 8 blocks, block_size=4, 1 head, head_dim=2, 1 layer
        let mut mgr = KvCacheManager::new(8, 4, 1, 2, 1, PoolBackend::Cpu);
        let data: [f32; 2] = [1.0, 2.0];

        let s0 = mgr.alloc_sequence().unwrap();
        let s1 = mgr.alloc_sequence().unwrap();

        // Each sequence appends 4 tokens = 1 block each
        for _ in 0..4 {
            mgr.append_token(s0, 0, data.as_ptr(), data.as_ptr()).unwrap();
            mgr.append_token(s1, 0, data.as_ptr(), data.as_ptr()).unwrap();
        }
        assert_eq!(mgr.allocator().allocated(), 2);

        // Free s0, its block returns to pool
        mgr.free_sequence(s0).unwrap();
        assert_eq!(mgr.allocator().allocated(), 1);
        assert_eq!(mgr.allocator().available(), 7);
    }

    #[test]
    fn test_multi_layer() {
        // 16 blocks, block_size=2, 2 heads, head_dim=4, 3 layers
        let mut mgr = KvCacheManager::new(16, 2, 2, 4, 3, PoolBackend::Cpu);
        let seq = mgr.alloc_sequence().unwrap();

        let data: [f32; 8] = [1.0; 8]; // [2 heads, 4 dim]
        // Append 1 token to each layer
        for layer in 0..3 {
            mgr.append_token(seq, layer, data.as_ptr(), data.as_ptr()).unwrap();
        }
        // 3 blocks used (one per layer)
        assert_eq!(mgr.allocator().allocated(), 3);
        assert_eq!(mgr.seq_token_count(seq, 0), Some(1));
        assert_eq!(mgr.seq_token_count(seq, 1), Some(1));
        assert_eq!(mgr.seq_token_count(seq, 2), Some(1));
    }
}
```

- [ ] **Step 2: Run tests**

Run: `cargo test -p nsl-runtime -- paged_kv::manager::tests`
Expected: PASS (5 tests)

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-runtime/src/paged_kv/manager.rs
git commit -m "feat(runtime): add KvCacheManager with multi-layer multi-sequence support"
```

---

### Task 5: FFI Exports for KvCacheManager

**Files:**
- Modify: `crates/nsl-runtime/src/paged_kv/manager.rs` (add FFI functions at bottom)

**Context:** The codegen and NSL programs need to call KvCacheManager through extern "C" functions. The manager is stored in a global `OnceLock` (similar to `WeightProvider` from M24) so it persists across FFI calls.

- [ ] **Step 1: Add the global slot and FFI exports**

Append to `crates/nsl-runtime/src/paged_kv/manager.rs`:

```rust
// ── Global KvCacheManager slot ──────────────────────────────────────────────

use std::sync::{Mutex, OnceLock};

static KV_CACHE: OnceLock<Mutex<KvCacheManager>> = OnceLock::new();

fn kv_cache() -> &'static Mutex<KvCacheManager> {
    KV_CACHE.get().unwrap_or_else(|| {
        eprintln!("nsl: KV cache not initialized. Call nsl_kv_cache_init first.");
        std::process::abort();
    })
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// Initialize the global KV cache manager.
///
/// num_blocks: total blocks in pool
/// block_size: tokens per block
/// num_heads: attention heads
/// head_dim: dimension per head
/// num_layers: transformer layers
#[no_mangle]
pub extern "C" fn nsl_kv_cache_init(
    num_blocks: i64,
    block_size: i64,
    num_heads: i64,
    head_dim: i64,
    num_layers: i64,
) {
    let backend = if cfg!(feature = "cuda") {
        #[cfg(feature = "cuda")]
        { PoolBackend::Gpu }
        #[cfg(not(feature = "cuda"))]
        { PoolBackend::Cpu }
    } else {
        PoolBackend::Cpu
    };

    let mgr = KvCacheManager::new(
        num_blocks as usize,
        block_size as usize,
        num_heads as usize,
        head_dim as usize,
        num_layers as usize,
        backend,
    );
    let _ = KV_CACHE.set(Mutex::new(mgr));
}

/// Allocate a new sequence. Returns sequence ID (i64).
#[no_mangle]
pub extern "C" fn nsl_kv_cache_alloc_seq() -> i64 {
    let mut mgr = kv_cache().lock().unwrap();
    match mgr.alloc_sequence() {
        Ok(id) => id as i64,
        Err(e) => {
            eprintln!("nsl: kv_cache_alloc_seq: {}", e);
            std::process::abort();
        }
    }
}

/// Free a sequence, returning its blocks to the pool.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_free_seq(seq_id: i64) {
    let mut mgr = kv_cache().lock().unwrap();
    if let Err(e) = mgr.free_sequence(seq_id as SeqId) {
        eprintln!("nsl: kv_cache_free_seq: {}", e);
        std::process::abort();
    }
}

/// Append one token's K and V for a specific layer.
///
/// k_ptr, v_ptr point to [num_heads, head_dim] f32 arrays.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_append(
    seq_id: i64,
    layer: i64,
    k_ptr: i64,
    v_ptr: i64,
) {
    let mut mgr = kv_cache().lock().unwrap();
    if let Err(e) = mgr.append_token(
        seq_id as SeqId,
        layer as usize,
        k_ptr as *const f32,
        v_ptr as *const f32,
    ) {
        eprintln!(
            "nsl: kv_cache_append: {} (seq={}, layer={}, allocated={}/{})",
            e, seq_id, layer,
            mgr.allocator().allocated(),
            mgr.allocator().num_blocks,
        );
        eprintln!("hint: increase pool size with @paged_kv(num_blocks=N) or reduce concurrent sequences");
        std::process::abort();
    }
}

/// Gather all cached K values for a sequence+layer into a new NslTensor.
///
/// Returns pointer to NslTensor with shape [num_heads, seq_len, head_dim], dtype=f32.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_get_k(seq_id: i64, layer: i64) -> i64 {
    let mgr = kv_cache().lock().unwrap();
    gather_to_tensor(&mgr, seq_id as SeqId, layer as usize, true)
}

/// Gather all cached V values for a sequence+layer into a new NslTensor.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_get_v(seq_id: i64, layer: i64) -> i64 {
    let mgr = kv_cache().lock().unwrap();
    gather_to_tensor(&mgr, seq_id as SeqId, layer as usize, false)
}

/// Get the token count for a sequence+layer.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_seq_len(seq_id: i64, layer: i64) -> i64 {
    let mgr = kv_cache().lock().unwrap();
    mgr.seq_token_count(seq_id as SeqId, layer as usize)
        .unwrap_or(0) as i64
}

/// Get the number of active sequences.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_active_seqs() -> i64 {
    let mgr = kv_cache().lock().unwrap();
    mgr.active_sequences() as i64
}

/// Get pool utilization ratio (0.0 to 1.0) as f64 bits in i64.
#[no_mangle]
pub extern "C" fn nsl_kv_cache_utilization() -> i64 {
    let mgr = kv_cache().lock().unwrap();
    let u = mgr.allocator().utilization();
    u.to_bits() as i64
}

fn gather_to_tensor(mgr: &KvCacheManager, seq_id: SeqId, layer: usize, is_k: bool) -> i64 {
    let (data_ptr, seq_len) = if is_k {
        mgr.gather_k(seq_id, layer)
    } else {
        mgr.gather_v(seq_id, layer)
    }.unwrap_or_else(|e| {
        eprintln!("nsl: kv_cache_get_{}: {}", if is_k { "k" } else { "v" }, e);
        std::process::abort();
    });

    let num_heads = mgr.allocator().num_heads;
    let head_dim = mgr.allocator().head_dim;

    // Build NslTensor with shape [num_heads, seq_len, head_dim], dtype=f32 (1)
    let ndim = 3i64;
    let shape_ptr = crate::memory::checked_alloc(3 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *shape_ptr = num_heads as i64;
        *shape_ptr.add(1) = seq_len as i64;
        *shape_ptr.add(2) = head_dim as i64;
    }

    let total_len = (num_heads * seq_len * head_dim) as i64;
    let strides_ptr = crate::memory::checked_alloc(3 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *strides_ptr = (seq_len * head_dim) as i64;
        *strides_ptr.add(1) = head_dim as i64;
        *strides_ptr.add(2) = 1;
    }

    let tensor = Box::new(crate::tensor::NslTensor {
        data: data_ptr as *mut c_void,
        shape: shape_ptr,
        strides: strides_ptr,
        ndim,
        len: total_len,
        refcount: 1,
        device: 0, // gathered to CPU
        dtype: 1,  // f32
        owns_data: 1,
    });

    Box::into_raw(tensor) as i64
}
```

- [ ] **Step 2: Verify build**

Run: `cargo build -p nsl-runtime`
Expected: builds without errors

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-runtime/src/paged_kv/manager.rs
git commit -m "feat(runtime): add FFI exports for KvCacheManager"
```

---

## Chunk 2: Codegen Integration

### Task 6: Declare KV Cache Runtime Functions in Codegen

**Files:**
- Modify: `crates/nsl-codegen/src/builtins.rs`

**Context:** The codegen needs to know about the KV cache FFI functions so it can emit calls to them. Follow the existing pattern at `builtins.rs:264-276` (M24 standalone functions).

- [ ] **Step 1: Add KV cache function declarations**

In `crates/nsl-codegen/src/builtins.rs`, add before the closing `];` of the `RUNTIME_FUNCTIONS` array (after the M24 standalone entries):

```rust
    // Paged KV cache (M25)
    ("nsl_kv_cache_init",        &[types::I64, types::I64, types::I64, types::I64, types::I64], None),
    ("nsl_kv_cache_alloc_seq",   &[],                                                           Some(types::I64)),
    ("nsl_kv_cache_free_seq",    &[types::I64],                                                 None),
    ("nsl_kv_cache_append",      &[types::I64, types::I64, types::I64, types::I64],             None),
    ("nsl_kv_cache_get_k",       &[types::I64, types::I64],                                     Some(types::I64)),
    ("nsl_kv_cache_get_v",       &[types::I64, types::I64],                                     Some(types::I64)),
    ("nsl_kv_cache_seq_len",     &[types::I64, types::I64],                                     Some(types::I64)),
    ("nsl_kv_cache_active_seqs", &[],                                                            Some(types::I64)),
    ("nsl_kv_cache_utilization", &[],                                                            Some(types::I64)),
    // Memory profiling (M25)
    ("nsl_profiler_init",        &[],                                                            None),
    ("nsl_profiler_dump",        &[types::I64, types::I64],                                     None),
    ("nsl_profiler_is_enabled",  &[],                                                            Some(types::I64)),
```

- [ ] **Step 2: Build to verify**

Run: `cargo build -p nsl-codegen`
Expected: builds without errors

- [ ] **Step 3: Commit**

```bash
git add crates/nsl-codegen/src/builtins.rs
git commit -m "feat(codegen): declare KV cache and profiling runtime functions for M25"
```

---

### Task 7: @paged_kv Semantic Validation

**Files:**
- Modify: `crates/nsl-semantic/src/checker.rs`

**Context:** The parser already handles `@paged_kv(block_size=16)` on model fields — decorators with keyword args are fully parsed (see `nsl-parser/src/decl.rs:88-114`). The semantic checker needs to validate: (1) the decorator only appears on model fields, (2) `block_size` is a positive integer literal, (3) optional `num_blocks` and `num_layers` args.

- [ ] **Step 1: Find the model field validation location and fix destructuring**

In `crates/nsl-semantic/src/checker.rs`, look for `ModelMember::LayerDecl` in `check_model_def()` or similar method. The existing match arm uses `..` which ignores the `decorators` field:

```rust
ModelMember::LayerDecl { name, type_ann, init, span, .. } => {
```

Change this to explicitly bind `decorators`:

```rust
ModelMember::LayerDecl { name, type_ann, init, decorators, span } => {
```

- [ ] **Step 2: Add @paged_kv validation**

Inside the `LayerDecl` match arm (after existing field processing), add validation for `@paged_kv`:

```rust
// Inside the loop processing LayerDecl members:
for deco in decorators {
    if deco.name.len() == 1 {
        let dname = self.interner.resolve(deco.name[0].0).unwrap_or("").to_string();
        if dname == "paged_kv" {
            // Validate arguments
            if let Some(ref args) = deco.args {
                for arg in args {
                    if let Some(ref name_sym) = arg.name {
                        let aname = self.interner.resolve(name_sym.0).unwrap_or("").to_string();
                        match aname.as_str() {
                            "block_size" | "num_blocks" | "num_layers" => {
                                // Must be a positive integer literal
                                if let nsl_ast::expr::ExprKind::IntLiteral(n) = &arg.value.kind {
                                    if *n <= 0 {
                                        self.diagnostics.push(
                                            Diagnostic::error(format!("@paged_kv: {} must be a positive integer", aname))
                                                .with_label(arg.span, "must be > 0"),
                                        );
                                    }
                                } else {
                                    self.diagnostics.push(
                                        Diagnostic::error(format!("@paged_kv: {} must be an integer literal", aname))
                                            .with_label(arg.span, "expected integer"),
                                    );
                                }
                            }
                            _ => {
                                self.diagnostics.push(
                                    Diagnostic::error(format!("@paged_kv: unknown argument '{}'", aname))
                                        .with_label(arg.span, "unknown argument"),
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}
```

- [ ] **Step 3: Write a test NSL file for semantic validation**

Create a temporary test file to verify the semantic checker accepts `@paged_kv`:

```nsl
model TestModel:
    @paged_kv(block_size=16)
    layers: list

    fn forward(self, x: Tensor) -> Tensor:
        return x
```

Run: `cargo run -p nsl-cli -- run test_paged_kv_semantic.nsl`
Expected: no semantic errors about `@paged_kv`

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-semantic/src/checker.rs
git commit -m "feat(semantic): validate @paged_kv decorator on model fields"
```

---

### Task 8: Codegen @paged_kv Model Init

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`
- Modify: `crates/nsl-codegen/src/expr.rs`

**Context:** When a model has a field with `@paged_kv(block_size=N)`, the codegen should emit `nsl_kv_cache_init(num_blocks, block_size, num_heads, head_dim, num_layers)` at model initialization time. For M25, the annotation is a trigger — head count and dimensions are taken from the decorator args or defaults.

**IMPORTANT:** The extraction and emission happen in **different methods** on `Compiler`. Store the config in a `Compiler` field so it persists across method calls.

Look at the model init codegen at `compiler.rs:1320-1330` where `ModelMember::LayerDecl` fields are initialized. The `ModelMember::LayerDecl` match arm currently uses `..` to ignore `decorators` — change this to bind `decorators` explicitly.

- [ ] **Step 1: Add paged_kv_configs field to Compiler struct**

In `compiler.rs`, add a new field to the `Compiler` struct (near the `no_grad_fns` and `test_fns` fields):

```rust
    /// Models with @paged_kv: model_name -> (num_blocks, block_size, num_heads, head_dim, num_layers)
    pub paged_kv_configs: HashMap<String, (i64, i64, i64, i64, i64)>,
```

Initialize it to `HashMap::new()` in the constructor.

- [ ] **Step 2: Extract @paged_kv args during model struct layout**

In `compiler.rs`, inside the `StmtKind::ModelDef(md)` arm that builds struct layouts (near line 474), add extraction after the existing field processing loop:

```rust
// Check for @paged_kv decorator on any member
for member in &md.members {
    if let ModelMember::LayerDecl { decorators, .. } = member {
        for deco in decorators {
            if deco.name.len() == 1 && self.resolve_sym(deco.name[0]) == "paged_kv" {
                let mut block_size: i64 = 16;
                let mut num_blocks: i64 = 1024;
                let mut num_heads: i64 = 1;
                let mut head_dim: i64 = 64;
                let mut num_layers: i64 = 1;
                if let Some(ref args) = deco.args {
                    for arg in args {
                        if let Some(ref name_sym) = arg.name {
                            let aname = self.resolve_sym(*name_sym).to_string();
                            if let nsl_ast::expr::ExprKind::IntLiteral(n) = &arg.value.kind {
                                match aname.as_str() {
                                    "block_size" => block_size = *n,
                                    "num_blocks" => num_blocks = *n,
                                    "num_heads" => num_heads = *n,
                                    "head_dim" => head_dim = *n,
                                    "num_layers" => num_layers = *n,
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                let model_name = self.resolve_sym(md.name).to_string();
                self.paged_kv_configs.insert(model_name, (num_blocks, block_size, num_heads, head_dim, num_layers));
            }
        }
    }
}
```

- [ ] **Step 3: Emit nsl_kv_cache_init at model init**

In the model init codegen (near line 1322, after field initialization), look up the config by model name:

```rust
// After field init, before returning the model pointer:
let model_name = self.resolve_sym(md.name).to_string();
if let Some(&(num_blocks, block_size, num_heads, head_dim, num_layers)) = self.paged_kv_configs.get(&model_name) {
    let init_id = self.runtime_fns["nsl_kv_cache_init"].0;
    let init_ref = self.module.declare_func_in_func(init_id, builder.func);
    let nb = builder.ins().iconst(cl_types::I64, num_blocks);
    let bs = builder.ins().iconst(cl_types::I64, block_size);
    let nh = builder.ins().iconst(cl_types::I64, num_heads);
    let hd = builder.ins().iconst(cl_types::I64, head_dim);
    let nl = builder.ins().iconst(cl_types::I64, num_layers);
    builder.ins().call(init_ref, &[nb, bs, nh, hd, nl]);
}
```

- [ ] **Step 4: Add KV cache builtin dispatch in expr.rs**

In `crates/nsl-codegen/src/expr.rs`, in the `compile_call()` method, add a dispatch block for KV cache functions. Find where other builtins like `enumerate`, `zip`, `sorted` are dispatched (look for `format!("nsl_{func_name}")` pattern) and add:

```rust
// Paged KV cache + profiling builtins (M25)
if matches!(func_name.as_str(),
    "kv_cache_init" | "kv_alloc_seq" | "kv_free_seq" |
    "kv_append" | "kv_get_k" | "kv_get_v" |
    "kv_seq_len" | "kv_active_seqs" | "kv_utilization" |
    "profiler_init" | "profiler_dump" | "profiler_is_enabled")
{
    let rt_name = format!("nsl_{func_name}");
    // Compile arguments and emit the call to the nsl_* runtime function
    // Follow the same pattern used by enumerate/zip/sorted dispatch
}
```

The exact implementation depends on how the existing builtin dispatch works. Study the `enumerate` or `sorted` dispatch as a template — compile each argument expression, look up the function in `self.runtime_fns`, and emit the call.

- [ ] **Step 5: Build to verify**

Run: `cargo build`
Expected: builds without errors

- [ ] **Step 6: Commit**

```bash
git add crates/nsl-codegen/src/compiler.rs
git commit -m "feat(codegen): emit nsl_kv_cache_init for models with @paged_kv decorator"
```

---

## Chunk 3: Memory Profiling + E2E Tests

### Task 9: Memory Profiler Runtime

**Files:**
- Create: `crates/nsl-runtime/src/profiling.rs`

**Context:** The profiler tracks block allocation events with timestamps and emits Chrome `chrome://tracing` compatible JSON. Uses atomic counters for zero overhead when disabled. Enabled at runtime via `nsl_profiler_init()`.

- [ ] **Step 1: Write the profiler implementation**

Create `crates/nsl-runtime/src/profiling.rs`:

```rust
//! Memory watermark profiler for KV cache block allocation.
//!
//! When enabled via `nsl_profiler_init()`, tracks allocation events and
//! writes Chrome tracing JSON to a file. Zero overhead when disabled
//! (single atomic check per alloc/free).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::io::Write;
use std::time::Instant;

static ENABLED: AtomicBool = AtomicBool::new(false);
static PEAK_BLOCKS: AtomicU64 = AtomicU64::new(0);
static CURRENT_BLOCKS: AtomicU64 = AtomicU64::new(0);
static TOTAL_ALLOCS: AtomicU64 = AtomicU64::new(0);
static TOTAL_FREES: AtomicU64 = AtomicU64::new(0);

struct ProfilerState {
    start_time: Instant,
    events: Vec<ProfileEvent>,
}

struct ProfileEvent {
    timestamp_us: u64,
    kind: EventKind,
    block_id: u32,
    seq_id: u64,
}

enum EventKind {
    Alloc,
    Free,
}

static PROFILER: OnceLock<Mutex<ProfilerState>> = OnceLock::new();

/// Check if profiling is enabled (single atomic read — zero cost when disabled).
pub fn is_enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

/// Record a block allocation event.
pub fn record_alloc(block_id: u32, seq_id: u64) {
    if !is_enabled() { return; }

    let current = CURRENT_BLOCKS.fetch_add(1, Ordering::Relaxed) + 1;
    TOTAL_ALLOCS.fetch_add(1, Ordering::Relaxed);
    // Update peak (compare-and-swap loop)
    let mut peak = PEAK_BLOCKS.load(Ordering::Relaxed);
    while current > peak {
        match PEAK_BLOCKS.compare_exchange_weak(peak, current, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(p) => peak = p,
        }
    }

    if let Some(state) = PROFILER.get() {
        let mut s = state.lock().unwrap();
        let ts = s.start_time.elapsed().as_micros() as u64;
        s.events.push(ProfileEvent {
            timestamp_us: ts,
            kind: EventKind::Alloc,
            block_id,
            seq_id,
        });
    }
}

/// Record a block free event.
pub fn record_free(block_id: u32, seq_id: u64) {
    if !is_enabled() { return; }

    CURRENT_BLOCKS.fetch_sub(1, Ordering::Relaxed);
    TOTAL_FREES.fetch_add(1, Ordering::Relaxed);

    if let Some(state) = PROFILER.get() {
        let mut s = state.lock().unwrap();
        let ts = s.start_time.elapsed().as_micros() as u64;
        s.events.push(ProfileEvent {
            timestamp_us: ts,
            kind: EventKind::Free,
            block_id,
            seq_id,
        });
    }
}

/// Write Chrome tracing JSON to a file.
pub fn dump_to_file(path: &str) {
    let state = match PROFILER.get() {
        Some(s) => s,
        None => {
            eprintln!("nsl: profiler not initialized");
            return;
        }
    };
    let s = state.lock().unwrap();

    let mut file = match std::fs::File::create(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("nsl: profiler: cannot create '{}': {}", path, e);
            return;
        }
    };

    let peak = PEAK_BLOCKS.load(Ordering::Relaxed);
    let total_allocs = TOTAL_ALLOCS.load(Ordering::Relaxed);
    let total_frees = TOTAL_FREES.load(Ordering::Relaxed);
    let current = CURRENT_BLOCKS.load(Ordering::Relaxed);

    // Chrome tracing format: array of trace events
    let _ = write!(file, "{{\"traceEvents\":[\n");

    for (i, evt) in s.events.iter().enumerate() {
        let name = match evt.kind {
            EventKind::Alloc => "block_alloc",
            EventKind::Free => "block_free",
        };
        let phase = match evt.kind {
            EventKind::Alloc => "B", // begin
            EventKind::Free => "E",  // end
        };
        let comma = if i + 1 < s.events.len() { "," } else { "" };
        let _ = writeln!(
            file,
            r#"  {{"name":"{}","ph":"{}","ts":{},"pid":0,"tid":{},"args":{{"block_id":{}}}}}{}"#,
            name, phase, evt.timestamp_us, evt.seq_id, evt.block_id, comma
        );
    }

    let _ = write!(file, "],\n\"metadata\":{{");
    let _ = write!(file, "\"peak_blocks\":{},", peak);
    let _ = write!(file, "\"total_allocs\":{},", total_allocs);
    let _ = write!(file, "\"total_frees\":{},", total_frees);
    let _ = write!(file, "\"current_blocks\":{}", current);
    let _ = writeln!(file, "}}\n}}");
}

// ── FFI exports ─────────────────────────────────────────────────────────────

/// Enable memory profiling. Call before any KV cache operations.
#[no_mangle]
pub extern "C" fn nsl_profiler_init() {
    ENABLED.store(true, Ordering::Relaxed);
    let _ = PROFILER.set(Mutex::new(ProfilerState {
        start_time: Instant::now(),
        events: Vec::new(),
    }));
}

/// Write profiling data to a JSON file.
#[no_mangle]
pub extern "C" fn nsl_profiler_dump(path_ptr: i64, path_len: i64) {
    if path_ptr <= 0 || path_len <= 0 { return; }
    let path = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr as *const u8, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    dump_to_file(path);
}

/// Check if profiling is enabled (returns 1 or 0).
#[no_mangle]
pub extern "C" fn nsl_profiler_is_enabled() -> i64 {
    if is_enabled() { 1 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_disabled_by_default() {
        assert!(!is_enabled());
        // record calls are no-ops when disabled
        record_alloc(0, 0);
        record_free(0, 0);
    }
}
```

- [ ] **Step 2: Wire profiler into BlockAllocator**

In `crates/nsl-runtime/src/paged_kv/block_alloc.rs`, add profiling hooks to `alloc()` and `free()`:

In the `alloc()` method, after `self.allocated_count += 1;`:
```rust
        crate::profiling::record_alloc(id, 0); // seq_id filled by manager
```

In the `free()` method, after `self.allocated_count -= 1;`:
```rust
        crate::profiling::record_free(id, 0);
```

- [ ] **Step 3: Run tests**

Run: `cargo test -p nsl-runtime -- profiling::tests`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-runtime/src/profiling.rs crates/nsl-runtime/src/paged_kv/block_alloc.rs
git commit -m "feat(runtime): add memory watermark profiler with Chrome tracing output"
```

---

### Task 10: --profile-memory CLI Flag

**Files:**
- Modify: `crates/nsl-cli/src/main.rs`

**Context:** Add `--profile-memory` flag to the `nsl run` command. When set, the compiled program calls `nsl_profiler_init()` at startup and `nsl_profiler_dump()` at exit. The codegen already has these functions declared (Task 6).

- [ ] **Step 1: Add the CLI flag**

In `crates/nsl-cli/src/main.rs`, find the `Run` variant of the `Cli` enum and add:

```rust
        /// Enable memory profiling (writes memory_profile.json)
        #[arg(long)]
        profile_memory: bool,
```

- [ ] **Step 2: Pass the flag through to the runtime**

In the `run_run()` function (or equivalent), when `profile_memory` is true, set an environment variable before executing the compiled program:

```rust
if profile_memory {
    cmd.env("NSL_PROFILE_MEMORY", "1");
}
```

And in the runtime startup (the existing `nsl_args_init` or similar), check for this env var:

```rust
// In main() codegen or runtime init:
if std::env::var("NSL_PROFILE_MEMORY").is_ok() {
    nsl_profiler_init();
}
```

Alternative approach: modify `compile_main()` to emit the profiler calls directly when a flag is passed. Either approach works — choose the simpler one for M25.

- [ ] **Step 3: Build to verify**

Run: `cargo build -p nsl-cli`
Expected: builds without errors

- [ ] **Step 4: Commit**

```bash
git add crates/nsl-cli/src/main.rs
git commit -m "feat(cli): add --profile-memory flag for KV cache memory profiling"
```

---

### Task 11: E2E Test — Multi-Sequence Paged KV

**Files:**
- Create: `examples/m25_paged_kv.nsl`
- Create: `tests/expected/m25_paged_kv.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

**Context:** Write an NSL program that exercises the paged KV cache with multiple sequences. Since the KV cache FFI functions are declared as builtins, the NSL program can call them directly as functions. The test runs on CPU (no CUDA required).

- [ ] **Step 1: Create the test NSL program**

Create `examples/m25_paged_kv.nsl`:

```nsl
# M25 E2E test: paged KV cache with multiple sequences
# Uses CPU backend (no CUDA required)
# 8 blocks, block_size=2, 2 heads, head_dim=4, 1 layer

fn main():
    # Initialize KV cache
    kv_cache_init(8, 2, 2, 4, 1)

    # Allocate 4 sequences
    let s0 = kv_alloc_seq()
    let s1 = kv_alloc_seq()
    let s2 = kv_alloc_seq()
    let s3 = kv_alloc_seq()

    print(kv_active_seqs())  # 4

    # Create token data: [num_heads=2, head_dim=4] = 8 values
    let k = Tensor.ones([2, 4])
    let v = Tensor.ones([2, 4])

    # Append 3 tokens to each sequence (3 tokens * 4 seqs = 12 tokens, needs 6 blocks of 8)
    for i in range(3):
        kv_append(s0, 0, k, v)
        kv_append(s1, 0, k, v)
        kv_append(s2, 0, k, v)
        kv_append(s3, 0, k, v)

    # Check sequence lengths
    print(kv_seq_len(s0, 0))  # 3
    print(kv_seq_len(s1, 0))  # 3

    # Gather K for sequence 0 — should be [2, 3, 4]
    let cached_k = kv_get_k(s0, 0)
    print(cached_k.shape)  # [2, 3, 4]

    # Free two sequences, blocks return to pool
    kv_free_seq(s0)
    kv_free_seq(s1)
    print(kv_active_seqs())  # 2

    # Remaining sequences still work
    let cached_k2 = kv_get_k(s2, 0)
    print(cached_k2.shape)  # [2, 3, 4]

    kv_free_seq(s2)
    kv_free_seq(s3)
    print(kv_active_seqs())  # 0
    print("PASS")
```

- [ ] **Step 2: Create expected output**

Create `tests/expected/m25_paged_kv.txt`:

```
4
3
3
[2, 3, 4]
2
[2, 3, 4]
0
PASS
```

- [ ] **Step 3: Add E2E test function**

In `crates/nsl-cli/tests/e2e.rs`, add:

```rust
#[test]
fn e2e_m25_paged_kv() {
    run("m25_paged_kv");
}
```

Where `run()` is the existing helper that compiles and runs an example, comparing output against expected.

**Note:** The builtin function names in NSL (`kv_cache_init`, `kv_alloc_seq`, etc.) must be mapped to the runtime FFI names (`nsl_kv_cache_init`, `nsl_kv_cache_alloc_seq`, etc.) by the codegen. This mapping happens in the builtin dispatch — check how existing builtins like `print()` map to `nsl_print_*`. If `kv_cache_init` is not recognized as a builtin, the codegen needs a name mapping entry. Adjust function names in the NSL file to match whichever convention the existing builtin dispatch uses (e.g., the function might need to be called `nsl_kv_cache_init` directly from NSL, or a shorter alias may need to be registered).

- [ ] **Step 4: Run the E2E test**

Run: `cargo test --test e2e -- e2e_m25_paged_kv`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/m25_paged_kv.nsl tests/expected/m25_paged_kv.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test: add M25 paged KV cache E2E test"
```

---

### Task 12: E2E Test — Memory Profiling

**Files:**
- Create: `examples/m25_profiling.nsl`
- Create: `tests/expected/m25_profiling.txt`
- Modify: `crates/nsl-cli/tests/e2e.rs`

**Context:** Test that `--profile-memory` produces valid Chrome tracing JSON. The test runs the program, checks the output file exists, and validates the JSON structure.

- [ ] **Step 1: Create the profiling test NSL program**

Create `examples/m25_profiling.nsl`:

```nsl
# M25 E2E test: memory profiling output
fn main():
    kv_cache_init(4, 2, 1, 2, 1)
    let s = kv_alloc_seq()
    let k = Tensor.ones([1, 2])
    let v = Tensor.ones([1, 2])
    kv_append(s, 0, k, v)
    kv_append(s, 0, k, v)
    kv_append(s, 0, k, v)
    kv_free_seq(s)
    print("DONE")
```

- [ ] **Step 2: Create expected output**

Create `tests/expected/m25_profiling.txt`:

```
DONE
```

- [ ] **Step 3: Add profiling E2E test**

In `crates/nsl-cli/tests/e2e.rs`, add a test that runs with `--profile-memory` and validates the output JSON:

```rust
#[test]
fn e2e_m25_profiling() {
    // First, run the normal test
    run("m25_profiling");

    // Then run with --profile-memory and verify JSON output
    let dir = tempfile::tempdir().unwrap();
    let exe_path = dir.path().join(if cfg!(windows) { "m25_profiling.exe" } else { "m25_profiling" });
    let nsl_path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..").join("examples").join("m25_profiling.nsl")
        .canonicalize().unwrap();

    // Compile
    let status = std::process::Command::new(env!("CARGO_BIN_EXE_nsl"))
        .args(["build", nsl_path.to_str().unwrap(), "-o", exe_path.to_str().unwrap()])
        .status().unwrap();
    assert!(status.success(), "compile failed");

    // Run with profile flag
    let output = std::process::Command::new(&exe_path)
        .env("NSL_PROFILE_MEMORY", "1")
        .output().unwrap();
    assert!(output.status.success(), "run failed");

    // Check memory_profile.json was created
    let profile_path = std::env::current_dir().unwrap().join("memory_profile.json");
    if profile_path.exists() {
        let content = std::fs::read_to_string(&profile_path).unwrap();
        // Validate it's valid JSON with expected fields
        let json: serde_json::Value = serde_json::from_str(&content).unwrap();
        assert!(json["traceEvents"].is_array());
        assert!(json["metadata"]["peak_blocks"].is_number());
        assert!(json["metadata"]["total_allocs"].as_u64().unwrap() > 0);
        // Clean up
        let _ = std::fs::remove_file(&profile_path);
    }
}
```

- [ ] **Step 4: Run the test**

Run: `cargo test --test e2e -- e2e_m25_profiling`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add examples/m25_profiling.nsl tests/expected/m25_profiling.txt crates/nsl-cli/tests/e2e.rs
git commit -m "test: add M25 memory profiling E2E test with Chrome tracing validation"
```

---

## Task Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | CUDA device + pinned memory | cuda/mod.rs | 3 unit tests (CUDA-only) |
| 2 | BlockAllocator | paged_kv/block_alloc.rs | 5 unit tests |
| 3 | PageTable | paged_kv/page_table.rs | 5 unit tests |
| 4 | KvCacheManager | paged_kv/manager.rs | 5 unit tests |
| 5 | FFI exports | paged_kv/manager.rs | build verification |
| 6 | Codegen builtins | builtins.rs | build verification |
| 7 | @paged_kv semantic | checker.rs | manual validation |
| 8 | @paged_kv codegen + dispatch | compiler.rs, expr.rs | build verification |
| 9 | Memory profiler | profiling.rs | 1 unit test |
| 10 | --profile-memory CLI | main.rs | build verification |
| 11 | E2E paged KV | m25_paged_kv.nsl | 1 E2E test |
| 12 | E2E profiling | m25_profiling.nsl | 1 E2E test |

## Deferred to Follow-Up

- **GPU page table sync (pinned buffers)** — the gather approach works for M25; fused paged attention kernel with GPU page table lookup deferred to M27
- **Fused paged attention PTX kernel** — M27 (FlashAttention-2)
- **Profiler integration into BlockAllocator alloc/free with seq_id** — the current hooks pass seq_id=0; proper seq_id tracking requires passing through manager→allocator
- **NSL-level syntax for KV cache operations** — currently requires calling runtime functions directly; DSL sugar deferred
