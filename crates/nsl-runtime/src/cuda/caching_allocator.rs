//! GPU caching allocator — PyTorch-style block splitting/coalescing pool.
//!
//! Replaces the old power-of-2 bucketed pool with a segment-based allocator
//! that dramatically reduces `cuMemAlloc` syscalls and internal fragmentation.
//!
//! Design:
//! - Two size classes: small (≤1 MB, 512B-aligned) and large (>1 MB, 2MB-aligned)
//! - Segments are large `cuMemAlloc` regions subdivided into Blocks
//! - Free blocks tracked in BTreeSet for O(log n) best-fit search
//! - Adjacent free blocks coalesce on free (O(1) linked-list merge)
//! - Fully-free segments released to driver on drain

use std::collections::{BTreeSet, HashMap};
use std::ffi::c_void;
use std::sync::{LazyLock, Mutex};

/// Leak-hunt trace: `NSL_DEBUG_MEM_TRACE=1` prints every block handout and
/// free with its pointer and op context. Diffing A/F events isolates which
/// specific allocations strand (the per-context summary only gives counts).
pub(crate) fn mem_trace_on() -> bool {
    static ON: LazyLock<bool> =
        LazyLock::new(|| std::env::var("NSL_DEBUG_MEM_TRACE").ok().as_deref() == Some("1"));
    *ON
}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Allocation pool tag — persistent allocations (model weights, optimizer states)
/// live in their own segments, preventing transient per-step intermediates from
/// fragmenting persistent memory and blocking segment release on drain.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AllocPool {
    /// Model weights, optimizer moment buffers, DataLoader tensors.
    /// These segments are never released by drain_all.
    Persistent,
    /// Forward/backward intermediates, gradients, temporaries.
    /// These segments are released by drain_all when fully free.
    Transient,
}

/// Surface tag — attributes VRAM to a training-loop surface for the
/// peak-VRAM report (P0.1 pretraining memory reduction). Parallel to
/// `AllocPool`: captured from a thread-local at alloc time, purely
/// observational — it never affects placement or reuse decisions.
///
/// Discriminants are the FFI wire values of `nsl_gpu_set_alloc_surface` /
/// `nsl_gpu_get_alloc_surface` and MUST stay in sync with the codegen-side
/// constants in `nsl-codegen/src/stmt.rs` (train-block brackets).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SurfaceTag {
    Other = 0,
    Weights = 1,
    OptimM = 2,
    OptimV = 3,
    MPartial = 4,
    Grads = 5,
    Activations = 6,
    AttnWorkspace = 7,
}

/// Number of `SurfaceTag` variants (array size for per-surface counters).
pub const NUM_SURFACES: usize = 8;

impl SurfaceTag {
    /// All variants in discriminant order (report row order).
    pub const ALL: [SurfaceTag; NUM_SURFACES] = [
        SurfaceTag::Other,
        SurfaceTag::Weights,
        SurfaceTag::OptimM,
        SurfaceTag::OptimV,
        SurfaceTag::MPartial,
        SurfaceTag::Grads,
        SurfaceTag::Activations,
        SurfaceTag::AttnWorkspace,
    ];

    /// Human-readable name used by the memory reports.
    pub fn name(self) -> &'static str {
        match self {
            SurfaceTag::Other => "other",
            SurfaceTag::Weights => "weights",
            SurfaceTag::OptimM => "optim_m",
            SurfaceTag::OptimV => "optim_v",
            SurfaceTag::MPartial => "m_partial",
            SurfaceTag::Grads => "grads",
            SurfaceTag::Activations => "activations",
            SurfaceTag::AttnWorkspace => "attn_workspace",
        }
    }

    /// Decode an FFI byte. Unknown values map to `Other` (never panics —
    /// this sits on the allocation hot path).
    pub fn from_u8(v: u8) -> SurfaceTag {
        match v {
            1 => SurfaceTag::Weights,
            2 => SurfaceTag::OptimM,
            3 => SurfaceTag::OptimV,
            4 => SurfaceTag::MPartial,
            5 => SurfaceTag::Grads,
            6 => SurfaceTag::Activations,
            7 => SurfaceTag::AttnWorkspace,
            _ => SurfaceTag::Other,
        }
    }
}

/// Per-surface byte counters, updated under the allocator lock wherever
/// `allocated_bytes` changes.
#[derive(Clone, Copy, Debug, Default)]
pub struct SurfaceCounters {
    /// Bytes currently allocated to this surface.
    pub current_bytes: usize,
    /// High-water mark of `current_bytes` (this surface's own peak).
    pub peak_bytes: usize,
    /// Bytes attributed to this surface at the moment the GLOBAL allocated
    /// peak was last set — the per-surface decomposition of peak VRAM.
    pub at_global_peak_bytes: usize,
}

/// The mechanism by which a device VRAM allocation was obtained. Every
/// device allocation — pooled block, stream-ordered async, or direct
/// `cuMemAlloc_v2` region — carries one so the memory report and peak
/// decomposition cover the whole process, not just the caching pool (A1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AllocationLifetime {
    /// A splittable/coalescible block from the caching pool (`alloc_managed`).
    Pooled,
    /// A stream-ordered `cuMemAllocAsync` allocation (`NSL_ASYNC_ALLOC=1`).
    Async,
    /// A direct `cuMemAlloc_v2` region outside the pool: the M36 slab, the
    /// paged-KV pools, and flash-attention scratch workspaces.
    DirectDevice,
}

/// Stable identity + attribution for one device allocation — the single
/// accounting record every VRAM allocation flows through (A1). For pooled
/// blocks the surface/pool live on the `Block` itself; async and direct
/// allocations keep an `AllocationMetadata` in the allocator's
/// `external_allocs` side table so their bytes still reach the surface
/// counters, the global peak, and the allocation-count gates.
///
/// `op_id` / `tensor_id` are the stable Wengert-op / tensor identities,
/// captured from a thread-local (`nsl_gpu_set_alloc_identity`) when the
/// allocation originates from a compiled op; runtime-internal library
/// allocations leave them `None`. They are the hook the compile-time arena
/// and CUDA-graph work (Milestone C) consume — populated opportunistically
/// now, load-bearing later.
#[derive(Clone, Copy, Debug)]
pub struct AllocationMetadata {
    pub surface: SurfaceTag,
    pub pool: AllocPool,
    pub bytes: usize,
    pub lifetime: AllocationLifetime,
    pub op_id: Option<u32>,
    pub tensor_id: Option<u64>,
}

/// Aggregate view of the live external (non-pooled) allocations for the
/// memory report — see [`CachingAllocator::external_summary`].
#[derive(Clone, Copy, Debug, Default)]
pub struct ExternalSummary {
    /// Bytes held by stream-ordered `cuMemAllocAsync` allocations.
    pub async_bytes: usize,
    /// Bytes held by direct `cuMemAlloc_v2` regions (slab, paged KV, attn
    /// workspaces).
    pub direct_bytes: usize,
    /// External bytes tagged to the persistent pool.
    pub persistent_bytes: usize,
    /// External allocations carrying a stable op/tensor identity.
    pub identified_count: usize,
    /// Total live external allocations.
    pub total_count: usize,
}

/// Small/large pool boundary (1 MB).
const SMALL_THRESHOLD: usize = 1 << 20;
/// Alignment for small allocations.
const SMALL_ALIGNMENT: usize = 512;
/// Alignment for large allocations.
const LARGE_ALIGNMENT: usize = 2 << 20; // 2 MB
/// Default segment size for small pool growth.
const SMALL_SEGMENT_SIZE: usize = 2 << 20; // 2 MB
/// Default segment size for large pool growth.
const LARGE_SEGMENT_SIZE: usize = 20 << 20; // 20 MB
/// Minimum remainder size to justify splitting a block (small pool).
const SMALL_MIN_SPLIT: usize = 512;
/// Minimum remainder size to justify splitting a block (large pool).
const LARGE_MIN_SPLIT: usize = 2 << 20; // 2 MB

// ---------------------------------------------------------------------------
// Data structures
// ---------------------------------------------------------------------------

/// A contiguous sub-region within a Segment. Blocks form a doubly-linked list
/// ordered by GPU address within their parent segment.
struct Block {
    /// Device pointer to the start of this block.
    ptr: *mut c_void,
    /// Total size of this block in bytes (may exceed requested size).
    size: usize,
    /// Size the user actually requested (for fragmentation stats).
    requested_size: usize,
    /// True if this block is currently handed out to a caller.
    allocated: bool,
    /// Previous block in the same segment (null if first).
    prev: *mut Block,
    /// Next block in the same segment (null if last).
    next: *mut Block,
    /// Index into `CachingAllocator::segments`.
    segment_idx: usize,
    /// CUDA op context captured when the block was handed out.
    context: String,
    /// Pool tag inherited from parent segment.
    pool: AllocPool,
    /// Surface tag captured from the thread-local at hand-out (P0.1
    /// per-surface VRAM accounting). Meaningful only while `allocated`.
    surface: SurfaceTag,
}

/// SAFETY: Block contains raw pointers to GPU memory and sibling Blocks.
/// All access is guarded by the single `CACHING_ALLOCATOR` Mutex.
unsafe impl Send for Block {}

/// A single `cuMemAlloc_v2` region, subdivided into one or more Blocks.
struct Segment {
    /// Base device pointer returned by `cuMemAlloc_v2`.
    base_ptr: *mut c_void,
    /// Total bytes allocated from the driver.
    total_size: usize,
    /// Head of the block linked list.
    blocks_head: *mut Block,
    /// Number of currently-allocated blocks. When 0, the segment is idle.
    allocated_count: usize,
    /// Whether this is a small-pool or large-pool segment.
    is_small: bool,
    /// Pool tag — controls whether drain_all releases this segment.
    pool: AllocPool,
}

unsafe impl Send for Segment {}

/// Key for the free-list BTreeSet. Ordered by (size ASC, ptr ASC) so
/// `range(key..)` yields the smallest block that satisfies the request.
#[derive(Eq, PartialEq, Clone, Copy)]
struct FreeBlockKey {
    size: usize,
    ptr: usize,
}

impl Ord for FreeBlockKey {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.size.cmp(&other.size).then(self.ptr.cmp(&other.ptr))
    }
}

impl PartialOrd for FreeBlockKey {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Allocation statistics.
#[derive(Clone, Debug, Default)]
pub struct AllocStats {
    pub num_allocs: usize,
    pub num_free_blocks: usize,
    pub allocated_bytes: usize,
    pub reserved_bytes: usize,
    pub peak_allocated_bytes: usize,
    pub peak_reserved_bytes: usize,
    pub num_driver_allocs: usize,
    pub num_driver_frees: usize,
    pub num_cache_hits: usize,
    pub num_cache_misses: usize,
    pub num_splits: usize,
    pub num_coalesces: usize,
    pub internal_fragmentation_bytes: usize,
    /// Cumulative count of allocation events since the last
    /// `reset_peak_and_counts` — pooled blocks AND external (async / direct)
    /// allocations. Never decremented on free. This is the allocation-count
    /// signal the A4 regression gates watch: an unexpected jump means the
    /// train loop grew a per-step allocation the arena/liveness passes missed.
    pub cumulative_allocs: u64,
    /// Cumulative count of external (async / direct `cuMemAlloc_v2`)
    /// allocations recorded through `record_external_alloc`.
    pub num_external_allocs: u64,
}

/// Backend trait for driver-level allocation. Allows mock in tests.
pub(crate) trait DriverAlloc {
    fn alloc(&self, size: usize) -> Result<*mut c_void, ()>;
    fn free(&self, ptr: *mut c_void);
    fn memset_zero(&self, ptr: *mut c_void, size: usize);
}

/// Production backend using cuMemAlloc_v2 / cuMemFree_v2.
pub(crate) struct CudaDriverAlloc;

impl DriverAlloc for CudaDriverAlloc {
    fn alloc(&self, size: usize) -> Result<*mut c_void, ()> {
        use cudarc::driver::sys::*;
        unsafe {
            let mut ptr: CUdeviceptr = 0;
            let result = cuMemAlloc_v2(&mut ptr, size);
            if result == CUresult::CUDA_SUCCESS {
                Ok(ptr as *mut c_void)
            } else {
                Err(())
            }
        }
    }

    fn free(&self, ptr: *mut c_void) {
        use cudarc::driver::sys::*;
        unsafe {
            let result = cuMemFree_v2(ptr as CUdeviceptr);
            if result != CUresult::CUDA_SUCCESS {
                eprintln!("nsl: cuMemFree_v2 failed in caching allocator: {:?} for {:p}", result, ptr);
            }
        }
    }

    fn memset_zero(&self, ptr: *mut c_void, size: usize) {
        // Route through the graph-aware wrapper: a zero-fill issued inside a
        // cuda-graph region must be captured/verified like any other GPU op
        // (a raw sync memset would invalidate an in-progress capture).
        super::inner::memset_d8(ptr, size);
    }
}

// ---------------------------------------------------------------------------
// CachingAllocator
// ---------------------------------------------------------------------------

pub(crate) struct CachingAllocator<D: DriverAlloc = CudaDriverAlloc> {
    /// Free blocks for small allocations (≤ 1 MB).
    small_free: BTreeSet<FreeBlockKey>,
    /// Free blocks for large allocations (> 1 MB).
    large_free: BTreeSet<FreeBlockKey>,
    /// Map from device pointer address → Block, for O(1) lookup on free.
    allocated_blocks: HashMap<usize, *mut Block>,
    /// All Block pointers (free and allocated) keyed by device ptr for lookup.
    all_blocks: HashMap<usize, *mut Block>,
    /// All segments owned by this allocator.
    segments: Vec<Segment>,
    /// Statistics.
    stats: AllocStats,
    /// Total bytes reserved from the CUDA driver.
    total_reserved: usize,
    /// Total bytes currently allocated to users.
    total_allocated: usize,
    /// Per-surface current/peak byte counters (P0.1 VRAM accounting).
    /// Indexed by `SurfaceTag as usize`; guarded by the same Mutex.
    surface_counters: [SurfaceCounters; NUM_SURFACES],
    /// Device allocations that are NOT pooled blocks — stream-ordered async
    /// (`cuMemAllocAsync`) and direct `cuMemAlloc_v2` regions (slab, paged
    /// KV, attention workspaces). Keyed by device-ptr address so the free
    /// path can decrement the surface/pool/peak counters (A1 unification).
    external_allocs: HashMap<usize, AllocationMetadata>,
    /// Configurable memory limit (0 = unlimited).
    memory_limit: usize,
    /// The driver backend.
    driver: D,
}

unsafe impl<D: DriverAlloc + Send> Send for CachingAllocator<D> {}

impl CachingAllocator<CudaDriverAlloc> {
    fn new() -> Self {
        let memory_limit = parse_mem_limit();
        CachingAllocator {
            small_free: BTreeSet::new(),
            large_free: BTreeSet::new(),
            allocated_blocks: HashMap::new(),
            all_blocks: HashMap::new(),
            segments: Vec::new(),
            stats: AllocStats::default(),
            total_reserved: 0,
            total_allocated: 0,
            surface_counters: [SurfaceCounters::default(); NUM_SURFACES],
            external_allocs: HashMap::new(),
            memory_limit,
            driver: CudaDriverAlloc,
        }
    }
}

impl<D: DriverAlloc> CachingAllocator<D> {
    #[cfg(test)]
    fn with_driver(driver: D) -> Self {
        CachingAllocator {
            small_free: BTreeSet::new(),
            large_free: BTreeSet::new(),
            allocated_blocks: HashMap::new(),
            all_blocks: HashMap::new(),
            segments: Vec::new(),
            stats: AllocStats::default(),
            total_reserved: 0,
            total_allocated: 0,
            surface_counters: [SurfaceCounters::default(); NUM_SURFACES],
            external_allocs: HashMap::new(),
            memory_limit: 0,
            driver,
        }
    }

    /// Round size up to the appropriate alignment.
    fn round_size(size: usize) -> usize {
        if size == 0 { return 0; }
        if size <= SMALL_THRESHOLD {
            // Round up to next multiple of SMALL_ALIGNMENT, minimum SMALL_ALIGNMENT
            let s = size.max(SMALL_ALIGNMENT);
            (s + SMALL_ALIGNMENT - 1) & !(SMALL_ALIGNMENT - 1)
        } else {
            // Round up to next multiple of LARGE_ALIGNMENT
            (size + LARGE_ALIGNMENT - 1) & !(LARGE_ALIGNMENT - 1)
        }
    }

    /// Select the free-list for a given rounded size.
    fn is_small(rounded_size: usize) -> bool {
        rounded_size <= SMALL_THRESHOLD
    }

    fn free_set(&self, small: bool) -> &BTreeSet<FreeBlockKey> {
        if small { &self.small_free } else { &self.large_free }
    }

    fn free_set_mut(&mut self, small: bool) -> &mut BTreeSet<FreeBlockKey> {
        if small { &mut self.small_free } else { &mut self.large_free }
    }

    fn min_split_size(small: bool) -> usize {
        if small { SMALL_MIN_SPLIT } else { LARGE_MIN_SPLIT }
    }

    fn segment_growth_size(small: bool) -> usize {
        if small { SMALL_SEGMENT_SIZE } else { LARGE_SEGMENT_SIZE }
    }

    /// Try to allocate from the free-list (cache hit). Returns None on miss.
    /// Prefers blocks from the current pool to avoid mixing persistent and
    /// transient allocations in the same segment (prevents fragmentation).
    pub(crate) fn alloc_from_cache(&mut self, size_bytes: usize) -> Option<*mut c_void> {
        if size_bytes == 0 { return None; }
        let rounded = Self::round_size(size_bytes);
        let small = Self::is_small(rounded);
        let current_pool = get_alloc_pool();
        let search_key = FreeBlockKey { size: rounded, ptr: 0 };

        // Best-fit: prefer a free block from the same pool.
        // Fall back to any pool only as a last resort.
        let found_key = self.free_set(small).range(search_key..).find(|k| {
            self.all_blocks.get(&k.ptr)
                .map(|&bptr| unsafe { (*bptr).pool } == current_pool)
                .unwrap_or(false)
        }).copied().or_else(|| {
            // Fallback: any pool (prevents OOM when same-pool is exhausted)
            self.free_set(small).range(search_key..).next().copied()
        });
        let found_key = found_key?;

        // Remove from free-list
        self.free_set_mut(small).remove(&found_key);
        self.stats.num_free_blocks -= 1;

        let block_ptr = *self.all_blocks.get(&found_key.ptr)
            .expect("caching_allocator: free-set entry has no matching block in all_blocks");
        let block = unsafe { &mut *block_ptr };

        // Possibly split
        let remainder = block.size - rounded;
        if remainder >= Self::min_split_size(small) {
            self.split_block(block_ptr, rounded);
        }

        // Mark allocated
        let block = unsafe { &mut *block_ptr };
        block.allocated = true;
        block.requested_size = size_bytes;
        block.context = super::inner::current_oom_context();
        block.surface = get_alloc_surface();

        // Update segment
        let seg = &mut self.segments[block.segment_idx];
        seg.allocated_count += 1;

        // Bookkeeping
        self.allocated_blocks.insert(block.ptr as usize, block_ptr);
        self.total_allocated += block.size;
        self.stats.num_allocs += 1;
        self.stats.allocated_bytes = self.total_allocated;
        self.stats.num_cache_hits += 1;
        self.stats.internal_fragmentation_bytes += block.size - size_bytes;
        self.stats.cumulative_allocs += 1;
        self.note_surface_alloc(block.surface, block.size);

        if mem_trace_on() {
            eprintln!("[mem-trace] A ptr={:p} size={} ctx={}", block.ptr, block.size, block.context);
        }
        Some(block.ptr)
    }

    /// Allocate by growing the cache (new segment from driver). Returns None on driver OOM.
    pub(crate) fn alloc_with_grow(&mut self, size_bytes: usize) -> Option<*mut c_void> {
        if size_bytes == 0 { return None; }
        let rounded = Self::round_size(size_bytes);
        let small = Self::is_small(rounded);
        let segment_size = rounded.max(Self::segment_growth_size(small));

        // Check memory limit
        if self.memory_limit > 0 && self.total_reserved + segment_size > self.memory_limit {
            return None;
        }

        // Allocate from driver
        let base_ptr = self.driver.alloc(segment_size).ok()?;

        // Create segment
        let pool = get_alloc_pool();
        let seg_idx = self.segments.len();
        let block = Box::into_raw(Box::new(Block {
            ptr: base_ptr,
            size: segment_size,
            requested_size: size_bytes,
            allocated: false,
            prev: std::ptr::null_mut(),
            next: std::ptr::null_mut(),
            segment_idx: seg_idx,
            context: String::new(),
            pool,
            surface: SurfaceTag::Other,
        }));

        self.segments.push(Segment {
            base_ptr,
            total_size: segment_size,
            blocks_head: block,
            allocated_count: 0,
            is_small: small,
            pool,
        });

        self.all_blocks.insert(base_ptr as usize, block);
        self.total_reserved += segment_size;
        self.stats.reserved_bytes = self.total_reserved;
        self.stats.num_driver_allocs += 1;
        if self.total_reserved > self.stats.peak_reserved_bytes {
            self.stats.peak_reserved_bytes = self.total_reserved;
        }

        self.stats.num_cache_misses += 1;

        // Allocate directly from the new block (avoid alloc_from_cache to
        // prevent double-counting as both a miss and a hit).
        let rounded = Self::round_size(size_bytes);
        let remainder = segment_size - rounded;
        if remainder >= Self::min_split_size(small) {
            self.split_block(block, rounded);
        }

        let blk = unsafe { &mut *block };
        blk.allocated = true;
        blk.requested_size = size_bytes;
        blk.context = super::inner::current_oom_context();
        blk.surface = get_alloc_surface();

        let seg = &mut self.segments[seg_idx];
        seg.allocated_count += 1;

        self.allocated_blocks.insert(base_ptr as usize, block);
        self.total_allocated += blk.size;
        self.stats.num_allocs += 1;
        self.stats.allocated_bytes = self.total_allocated;
        self.stats.internal_fragmentation_bytes += blk.size - size_bytes;
        self.stats.cumulative_allocs += 1;
        self.note_surface_alloc(blk.surface, blk.size);

        if mem_trace_on() {
            eprintln!("[mem-trace] A ptr={:p} size={} ctx={}", blk.ptr, blk.size, blk.context);
        }
        Some(base_ptr)
    }

    /// Split a block: shrink `block` to `new_size` and create a free remainder.
    fn split_block(&mut self, block_ptr: *mut Block, new_size: usize) {
        let block = unsafe { &mut *block_ptr };
        let remainder_size = block.size - new_size;
        let remainder_ptr = unsafe { (block.ptr as *mut u8).add(new_size) as *mut c_void };
        let segment_idx = block.segment_idx;

        let remainder = Box::into_raw(Box::new(Block {
            ptr: remainder_ptr,
            size: remainder_size,
            requested_size: 0,
            allocated: false,
            prev: block_ptr,
            next: block.next,
            segment_idx: block.segment_idx,
            context: String::new(),
            pool: block.pool,
            surface: SurfaceTag::Other,
        }));

        // Update next block's prev pointer
        if !block.next.is_null() {
            unsafe { (*block.next).prev = remainder; }
        }

        block.next = remainder;
        block.size = new_size;

        // File the remainder in its owning segment's free-list. Choosing by the
        // remainder's own size would strand it: the remainder of a 2 MB small
        // segment exceeds SMALL_THRESHOLD, so it would land in `large_free`,
        // where a small request (which only searches `small_free`) never looks.
        //
        // This makes free-list placement depend on `Block::segment_idx` being
        // accurate. `drain_all` keeps it so by rewriting every block's index
        // after its `swap_remove` calls; any future segment renumbering must do
        // the same or blocks will be filed into the wrong pool.
        let remainder_small = self.segments[segment_idx].is_small;
        self.all_blocks.insert(remainder_ptr as usize, remainder);
        self.free_set_mut(remainder_small).insert(FreeBlockKey {
            size: remainder_size,
            ptr: remainder_ptr as usize,
        });

        self.stats.num_splits += 1;
        self.stats.num_free_blocks += 1;
    }

    /// Free a block and coalesce with neighbors.
    pub(crate) fn free_block(&mut self, ptr: *mut c_void) -> bool {
        if ptr.is_null() { return false; }

        let block_ptr = match self.allocated_blocks.remove(&(ptr as usize)) {
            Some(b) => b,
            None => return false,
        };

        let block = unsafe { &mut *block_ptr };
        debug_assert!(block.allocated, "double free detected");
        debug_assert_eq!(block.ptr, ptr, "block ptr mismatch");

        let old_size = block.size;
        let old_requested = block.requested_size;
        let old_surface = block.surface;
        if mem_trace_on() {
            eprintln!("[mem-trace] F ptr={:p} size={} ctx={}", ptr, old_size, block.context);
        }
        block.allocated = false;
        block.requested_size = 0;
        block.context.clear();

        let seg = &mut self.segments[block.segment_idx];
        seg.allocated_count -= 1;
        // Every block reachable from this one via prev/next lives in the same
        // segment, so one pool flag governs all the free-list edits below.
        let small = seg.is_small;

        self.total_allocated -= old_size;
        self.stats.num_allocs -= 1;
        self.stats.allocated_bytes = self.total_allocated;
        self.stats.internal_fragmentation_bytes = self.stats.internal_fragmentation_bytes
            .saturating_sub(old_size - old_requested);
        self.note_surface_free(old_surface, old_size);

        // Coalesce with next
        let next = block.next;
        if !next.is_null() {
            let next_block = unsafe { &*next };
            if !next_block.allocated {
                // Copy fields from next_block into locals BEFORE reborrowing
                // block mutably. This avoids UB from aliasing &*next while
                // mutating &mut *block_ptr (MIRI-clean).
                let next_size = next_block.size;
                let next_next = next_block.next;
                let next_ptr_addr = next_block.ptr as usize;

                // Remove next from its segment's free-list. Bind the result
                // before asserting: debug_assert! does not evaluate its argument
                // in release builds, which would skip the removal entirely.
                let removed = self.free_set_mut(small).remove(&FreeBlockKey {
                    size: next_size,
                    ptr: next_ptr_addr,
                });
                debug_assert!(removed, "next block absent from its segment's free-list");
                self.stats.num_free_blocks -= 1;

                let block = unsafe { &mut *block_ptr };
                block.size += next_size;
                block.next = next_next;
                if !next_next.is_null() {
                    unsafe { (*next_next).prev = block_ptr; }
                }

                // Remove next from tracking and deallocate
                self.all_blocks.remove(&next_ptr_addr);
                unsafe { drop(Box::from_raw(next)); }
                self.stats.num_coalesces += 1;
            }
        }

        // Coalesce with prev
        let block = unsafe { &*block_ptr };
        let prev = block.prev;
        if !prev.is_null() {
            let prev_block = unsafe { &*prev };
            if !prev_block.allocated {
                // Remove prev from its segment's free-list
                let removed = self.free_set_mut(small).remove(&FreeBlockKey {
                    size: prev_block.size,
                    ptr: prev_block.ptr as usize,
                });
                debug_assert!(removed, "prev block absent from its segment's free-list");
                self.stats.num_free_blocks -= 1;

                // Merge block into prev
                let block = unsafe { &*block_ptr };
                let merged_size = prev_block.size + block.size;
                let block_next = block.next;

                let prev_block = unsafe { &mut *prev };
                prev_block.size = merged_size;
                prev_block.next = block_next;
                if !block_next.is_null() {
                    unsafe { (*block_next).prev = prev; }
                }

                // Remove current block from tracking and deallocate
                self.all_blocks.remove(&(ptr as usize));
                unsafe { drop(Box::from_raw(block_ptr)); }
                self.stats.num_coalesces += 1;

                // Insert prev (the coalesced block) into its segment's free-list
                self.free_set_mut(small).insert(FreeBlockKey {
                    size: merged_size,
                    ptr: prev_block.ptr as usize,
                });
                self.stats.num_free_blocks += 1;

                return true;
            }
        }

        // No prev-coalesce: insert the (possibly next-coalesced) block into free-list
        let block = unsafe { &*block_ptr };
        self.free_set_mut(small).insert(FreeBlockKey {
            size: block.size,
            ptr: block.ptr as usize,
        });
        self.stats.num_free_blocks += 1;

        true
    }

    /// Drain fully-free Transient segments back to the driver. Returns bytes freed.
    /// Persistent segments are never drained (they hold model weights/optimizer states).
    pub(crate) fn drain_all(&mut self) -> usize {
        let mut freed = 0usize;
        let mut to_remove: Vec<usize> = Vec::new();

        for (idx, seg) in self.segments.iter().enumerate() {
            if seg.allocated_count == 0 && seg.pool == AllocPool::Transient {
                to_remove.push(idx);
            }
        }

        // Process in reverse order so swap_remove doesn't invalidate
        // indices we haven't visited yet.
        to_remove.sort_unstable_by(|a, b| b.cmp(a));
        for idx in to_remove {
            // Copy segment fields into locals to avoid borrowing self.segments
            // while we need &mut self for free_set_mut / all_blocks / driver.
            let seg_blocks_head = self.segments[idx].blocks_head;
            let seg_base_ptr = self.segments[idx].base_ptr;
            let seg_total_size = self.segments[idx].total_size;
            let seg_is_small = self.segments[idx].is_small;

            // Walk the block list and remove all blocks from free-set + all_blocks
            let mut bptr = seg_blocks_head;
            while !bptr.is_null() {
                let block = unsafe { &*bptr };
                let next = block.next;
                let blk_size = block.size;
                let blk_ptr = block.ptr as usize;

                // Remove from free-list (all blocks in a fully-free segment must be free)
                let removed = self.free_set_mut(seg_is_small).remove(&FreeBlockKey {
                    size: blk_size,
                    ptr: blk_ptr,
                });
                debug_assert!(removed, "block in a fully-free segment absent from its free-list");
                self.stats.num_free_blocks = self.stats.num_free_blocks.saturating_sub(1);
                self.all_blocks.remove(&blk_ptr);

                // Deallocate the Block node
                unsafe { drop(Box::from_raw(bptr)); }
                bptr = next;
            }

            // Free the GPU memory
            self.driver.free(seg_base_ptr);
            freed += seg_total_size;
            self.total_reserved -= seg_total_size;
            self.stats.num_driver_frees += 1;

            // Remove segment (swap_remove to avoid O(n) shift)
            self.segments.swap_remove(idx);
        }

        // After all removals, do a single O(blocks) sweep to rewrite
        // segment_idx on every block.  Consecutive swap_removes can
        // leave stale indices that the per-removal fixup missed.
        for (idx, seg) in self.segments.iter().enumerate() {
            let mut bptr = seg.blocks_head;
            while !bptr.is_null() {
                let block = unsafe { &mut *bptr };
                block.segment_idx = idx;
                bptr = block.next;
            }
        }

        self.stats.reserved_bytes = self.total_reserved;
        freed
    }

    /// Record `size` freshly allocated bytes against `surface`: bump the
    /// surface's current/peak, and — when the GLOBAL allocated peak moves —
    /// re-snapshot every surface's current bytes as the peak decomposition.
    /// Caller must have already added `size` to `total_allocated`.
    /// Cost: one array add + one compare (plus an 8-slot copy only on a new
    /// global peak) under the already-held lock — acceptable always-on.
    fn note_surface_alloc(&mut self, surface: SurfaceTag, size: usize) {
        let c = &mut self.surface_counters[surface as usize];
        c.current_bytes += size;
        if c.current_bytes > c.peak_bytes {
            c.peak_bytes = c.current_bytes;
        }
        if self.total_allocated > self.stats.peak_allocated_bytes {
            self.stats.peak_allocated_bytes = self.total_allocated;
            for sc in &mut self.surface_counters {
                sc.at_global_peak_bytes = sc.current_bytes;
            }
        }
    }

    /// Record `size` bytes returning from `surface` on free. Saturating: a
    /// counter drift must never poison the allocator (report-only data).
    fn note_surface_free(&mut self, surface: SurfaceTag, size: usize) {
        let c = &mut self.surface_counters[surface as usize];
        c.current_bytes = c.current_bytes.saturating_sub(size);
    }

    /// A1: record a device VRAM allocation that is NOT a pooled block — a
    /// stream-ordered async allocation (`cuMemAllocAsync`) or a direct
    /// `cuMemAlloc_v2` region (slab, paged KV, attention workspace). Routes
    /// it through the SAME surface/pool/peak/count accounting as pooled
    /// blocks so the memory report and peak decomposition cover every device
    /// allocation, not just the caching pool. The caller supplies the
    /// metadata snapshot (surface/pool/identity captured from the
    /// thread-locals at the allocation site via `current_alloc_metadata`).
    pub(crate) fn record_external_alloc(&mut self, ptr: *mut c_void, meta: AllocationMetadata) {
        if ptr.is_null() || meta.bytes == 0 {
            return;
        }
        // Idempotence guard: a re-recorded ptr (should not happen without an
        // intervening free) must not double-count — reconcile via the delta.
        if let Some(old) = self.external_allocs.insert(ptr as usize, meta) {
            self.total_allocated = self.total_allocated.saturating_sub(old.bytes);
            self.note_surface_free(old.surface, old.bytes);
        }
        self.total_allocated += meta.bytes;
        self.stats.allocated_bytes = self.total_allocated;
        self.stats.cumulative_allocs += 1;
        self.stats.num_external_allocs += 1;
        self.note_surface_alloc(meta.surface, meta.bytes);
    }

    /// A1: reverse of `record_external_alloc`. Returns true if `ptr` was a
    /// tracked external allocation. Saturating: counter drift never poisons
    /// the pool (report-only data).
    pub(crate) fn record_external_free(&mut self, ptr: *mut c_void) -> bool {
        match self.external_allocs.remove(&(ptr as usize)) {
            Some(meta) => {
                self.total_allocated = self.total_allocated.saturating_sub(meta.bytes);
                self.stats.allocated_bytes = self.total_allocated;
                self.note_surface_free(meta.surface, meta.bytes);
                true
            }
            None => false,
        }
    }

    /// Global peak allocated device bytes across every surface and every
    /// allocation mechanism (pooled + async + direct). A4 gate input.
    pub(crate) fn peak_allocated_bytes(&self) -> usize {
        self.stats.peak_allocated_bytes
    }

    /// Cumulative allocation-event count since the last
    /// `reset_peak_and_counts` (pooled + external). A4 gate input: an
    /// unexpected jump means the train loop grew a per-step allocation.
    pub(crate) fn cumulative_alloc_count(&self) -> u64 {
        self.stats.cumulative_allocs
    }

    /// This surface's own high-water mark in bytes.
    pub(crate) fn surface_peak(&self, t: SurfaceTag) -> usize {
        self.surface_counters[t as usize].peak_bytes
    }

    /// This surface's bytes at the moment the global allocated peak was set —
    /// the per-surface decomposition of peak VRAM.
    pub(crate) fn surface_at_global_peak(&self, t: SurfaceTag) -> usize {
        self.surface_counters[t as usize].at_global_peak_bytes
    }

    /// Diagnostic over the external (non-pooled) allocations currently live:
    /// async vs direct-device bytes, persistent-pool bytes, and how many carry
    /// a stable op/tensor identity. Reads every `AllocationMetadata` field so
    /// the identity record is observable in the memory report, not merely
    /// stored — and it is the surface the Milestone-C arena/CUDA-graph work
    /// reads back.
    pub(crate) fn external_summary(&self) -> ExternalSummary {
        let mut s = ExternalSummary::default();
        for m in self.external_allocs.values() {
            s.total_count += 1;
            match m.lifetime {
                AllocationLifetime::Async => s.async_bytes += m.bytes,
                AllocationLifetime::DirectDevice => s.direct_bytes += m.bytes,
                AllocationLifetime::Pooled => {}
            }
            if m.pool == AllocPool::Persistent {
                s.persistent_bytes += m.bytes;
            }
            if m.op_id.is_some() || m.tensor_id.is_some() {
                s.identified_count += 1;
            }
        }
        s
    }

    /// Reset the peak high-water marks and cumulative counters so a caller can
    /// measure a fresh region (e.g. one training step). Current live bytes are
    /// preserved; peaks re-seed to the current live level.
    pub(crate) fn reset_peak_and_counts(&mut self) {
        self.stats.peak_allocated_bytes = self.total_allocated;
        self.stats.cumulative_allocs = 0;
        self.stats.num_external_allocs = 0;
        for sc in &mut self.surface_counters {
            sc.peak_bytes = sc.current_bytes;
            sc.at_global_peak_bytes = sc.current_bytes;
        }
    }

    /// Per-surface VRAM accounting snapshot (P0.1), one entry per
    /// `SurfaceTag` in discriminant order.
    /// Format: Vec<(surface_name, current_bytes, peak_bytes)>. u64 so
    /// benchmark consumers get a stable width regardless of platform usize.
    pub fn surface_breakdown(&self) -> Vec<(&'static str, u64, u64)> {
        SurfaceTag::ALL
            .iter()
            .map(|&t| {
                let c = self.surface_counters[t as usize];
                (t.name(), c.current_bytes as u64, c.peak_bytes as u64)
            })
            .collect()
    }

    /// Render the per-surface table for the memory reports, one line per
    /// surface: name / current / surface peak / bytes at the global
    /// allocated peak. Each line is prefixed with `indent`.
    pub fn surface_table_string(&self, indent: &str) -> String {
        let mut out = String::new();
        for t in SurfaceTag::ALL {
            let c = self.surface_counters[t as usize];
            out.push_str(&format!(
                "{indent}{:<14} current {:>10}  peak {:>10}  at-global-peak {:>10}\n",
                t.name(),
                fmt_bytes(c.current_bytes),
                fmt_bytes(c.peak_bytes),
                fmt_bytes(c.at_global_peak_bytes),
            ));
        }
        out
    }

    /// Get a snapshot of current statistics.
    pub(crate) fn stats(&self) -> AllocStats {
        self.stats.clone()
    }

    /// Check if a pointer is managed by this allocator.
    pub(crate) fn is_allocated(&self, ptr: *mut c_void) -> bool {
        self.allocated_blocks.contains_key(&(ptr as usize))
    }

    /// Return per-pool reserved-byte breakdown.
    /// Format: (persistent_bytes, persistent_segs, transient_bytes, transient_segs).
    /// Used by `nsl_debug_gpu_mem` to diagnose the ELTLS leak.
    pub(crate) fn pool_breakdown(&self) -> (usize, usize, usize, usize) {
        let mut p_bytes = 0usize;
        let mut p_segs = 0usize;
        let mut t_bytes = 0usize;
        let mut t_segs = 0usize;
        for seg in &self.segments {
            match seg.pool {
                AllocPool::Persistent => { p_bytes += seg.total_size; p_segs += 1; }
                AllocPool::Transient  => { t_bytes += seg.total_size; t_segs += 1; }
            }
        }
        (p_bytes, p_segs, t_bytes, t_segs)
    }

    /// Return a summary of all currently allocated blocks grouped by context.
    /// Format: Vec<(context, count, total_bytes)> sorted by total_bytes descending.
    pub fn allocated_block_summary(&self) -> Vec<(String, usize, usize)> {
        let mut by_context: HashMap<String, (usize, usize)> = HashMap::new();
        for (_, &block_ptr) in &self.allocated_blocks {
            let block = unsafe { &*block_ptr };
            let ctx = if block.context.is_empty() { "(no context)" } else { &block.context };
            let entry = by_context.entry(ctx.to_string()).or_insert((0, 0));
            entry.0 += 1;
            entry.1 += block.size;
        }
        let mut result: Vec<_> = by_context.into_iter()
            .map(|(ctx, (count, bytes))| (ctx, count, bytes))
            .collect();
        result.sort_by(|a, b| b.2.cmp(&a.2));
        result
    }
}

// ---------------------------------------------------------------------------
// Global static instance
// ---------------------------------------------------------------------------

pub static CACHING_ALLOCATOR: LazyLock<Mutex<CachingAllocator>> =
    LazyLock::new(|| Mutex::new(CachingAllocator::new()));

thread_local! {
    /// Current allocation pool tag. Code that allocates persistent tensors
    /// (model weights, optimizer states) should set this to Persistent before
    /// allocating, then restore to Transient after.
    static CURRENT_POOL: std::cell::Cell<AllocPool> = const { std::cell::Cell::new(AllocPool::Transient) };
}

/// Set the current allocation pool for subsequent GPU allocations.
pub fn set_alloc_pool(pool: AllocPool) {
    CURRENT_POOL.with(|p| p.set(pool));
}

/// Get the current allocation pool.
pub fn get_alloc_pool() -> AllocPool {
    CURRENT_POOL.with(|p| p.get())
}

thread_local! {
    /// Current surface tag for P0.1 per-surface VRAM accounting. Codegen
    /// brackets set this around the train-block allocation regions
    /// (weights / optim m+v / m_partial / grads / activations);
    /// runtime-internal workspaces use `SurfaceGuard`. Parallel to
    /// `CURRENT_POOL` and deliberately independent of it.
    static CURRENT_SURFACE: std::cell::Cell<SurfaceTag> = const { std::cell::Cell::new(SurfaceTag::Other) };
}

/// Set the current allocation surface tag for subsequent GPU allocations.
pub fn set_alloc_surface(surface: SurfaceTag) {
    CURRENT_SURFACE.with(|s| s.set(surface));
}

/// Get the current allocation surface tag.
pub fn get_alloc_surface() -> SurfaceTag {
    CURRENT_SURFACE.with(|s| s.get())
}

thread_local! {
    /// Stable `(op_id, tensor_id)` identity for subsequent allocations, set
    /// by codegen around a compiled op's allocations. `(0, 0)` = unset
    /// (mapped to `None` in the metadata snapshot). The hook the
    /// compile-time arena and CUDA-graph work (Milestone C) consume;
    /// populated opportunistically today, load-bearing later.
    static CURRENT_ALLOC_IDENTITY: std::cell::Cell<(u32, u64)> = const { std::cell::Cell::new((0, 0)) };
}

/// Set the stable `(op_id, tensor_id)` identity for subsequent allocations.
pub fn set_alloc_identity(op_id: u32, tensor_id: u64) {
    CURRENT_ALLOC_IDENTITY.with(|c| c.set((op_id, tensor_id)));
}

/// Clear the allocation identity (back to unset).
pub fn clear_alloc_identity() {
    CURRENT_ALLOC_IDENTITY.with(|c| c.set((0, 0)));
}

/// Read the current allocation identity as optionals (`0` sentinel → `None`).
pub fn get_alloc_identity() -> (Option<u32>, Option<u64>) {
    let (o, t) = CURRENT_ALLOC_IDENTITY.with(|c| c.get());
    (
        if o == 0 { None } else { Some(o) },
        if t == 0 { None } else { Some(t) },
    )
}

/// Snapshot the current surface / pool / identity thread-locals into an
/// [`AllocationMetadata`] for an external (non-pooled) allocation of `bytes`.
/// This is the single capture path both external-allocation sites use, so
/// async and direct allocations are attributed exactly like pooled blocks.
pub fn current_alloc_metadata(bytes: usize, lifetime: AllocationLifetime) -> AllocationMetadata {
    let (op_id, tensor_id) = get_alloc_identity();
    AllocationMetadata {
        surface: get_alloc_surface(),
        pool: get_alloc_pool(),
        bytes,
        lifetime,
        op_id,
        tensor_id,
    }
}

/// RAII guard: tag every allocation in the enclosing scope with `surface`,
/// restoring the previous tag on drop (panic-safe — the restore also runs
/// during unwind).
pub(crate) struct SurfaceGuard {
    prev: SurfaceTag,
}

impl SurfaceGuard {
    pub(crate) fn new(surface: SurfaceTag) -> SurfaceGuard {
        let prev = get_alloc_surface();
        set_alloc_surface(surface);
        SurfaceGuard { prev }
    }
}

impl Drop for SurfaceGuard {
    fn drop(&mut self) {
        set_alloc_surface(self.prev);
    }
}

// ---------------------------------------------------------------------------
// Configuration parsing
// ---------------------------------------------------------------------------

fn parse_mem_limit() -> usize {
    let val = match std::env::var("NSL_GPU_MEM_LIMIT") {
        Ok(v) if !v.is_empty() && v != "0" => v,
        _ => return 0,
    };
    let val = val.trim();
    let (num_str, multiplier) = if let Some(n) = val.strip_suffix('G').or_else(|| val.strip_suffix('g')) {
        (n, 1024 * 1024 * 1024)
    } else if let Some(n) = val.strip_suffix('M').or_else(|| val.strip_suffix('m')) {
        (n, 1024 * 1024)
    } else if let Some(n) = val.strip_suffix('K').or_else(|| val.strip_suffix('k')) {
        (n, 1024)
    } else {
        (val, 1)
    };
    num_str.trim().parse::<usize>().unwrap_or(0) * multiplier
}

/// Human-readable byte count (shared by the surface table and the
/// NSL_MEMSTATS summary).
fn fmt_bytes(b: usize) -> String {
    if b >= 1 << 30 {
        format!("{:.2} GB", b as f64 / (1u64 << 30) as f64)
    } else if b >= 1 << 20 {
        format!("{:.1} MB", b as f64 / (1u64 << 20) as f64)
    } else if b >= 1 << 10 {
        format!("{:.1} KB", b as f64 / (1u64 << 10) as f64)
    } else {
        format!("{} B", b)
    }
}

/// Check if memory stats printing is enabled.
pub(crate) fn memstats_enabled() -> bool {
    static ENABLED: LazyLock<bool> = LazyLock::new(|| {
        std::env::var("NSL_MEMSTATS").map(|v| v == "1").unwrap_or(false)
    });
    *ENABLED
}

/// Print a human-readable summary of GPU memory allocator statistics.
pub(crate) fn print_memory_summary() {
    let alloc = CACHING_ALLOCATOR.lock().unwrap();
    let s = &alloc.stats;
    let fmt = fmt_bytes;
    let mut context_totals: HashMap<String, (usize, usize)> = HashMap::new();
    for &block_ptr in alloc.allocated_blocks.values() {
        let block = unsafe { &*block_ptr };
        let key = if block.context.is_empty() {
            "<unknown>".to_string()
        } else {
            block.context.clone()
        };
        let entry = context_totals.entry(key).or_insert((0, 0));
        entry.0 += block.size;
        entry.1 += 1;
    }
    let mut top_contexts: Vec<_> = context_totals.into_iter().collect();
    top_contexts.sort_by(|a, b| b.1 .0.cmp(&a.1 .0));
    let mut top_context_lines = String::new();
    for (context, (bytes, count)) in top_contexts.into_iter().take(8) {
        top_context_lines.push_str(&format!(
            "         {}: {} ({} blocks)\n",
            context,
            fmt(bytes),
            count,
        ));
    }
    // P0.1 / A1: per-surface VRAM attribution + pool breakdown. Async
    // (cuMemAllocAsync) and direct cuMemAlloc_v2 regions are now routed
    // through the unified surface accounting (external_allocs side table), so
    // the surface table covers them. The pool breakdown below still reflects
    // caching-pool segments only (async/direct allocations own no segment).
    let surface_lines = alloc.surface_table_string("         ");
    let (p_bytes, p_segs, t_bytes, t_segs) = alloc.pool_breakdown();
    let async_note = if super::inner::async_alloc_enabled() {
        "\n         NOTE: NSL_ASYNC_ALLOC=1 — surface bytes include async \
         allocations; the pool (segment) breakdown covers cached blocks only.\n"
    } else {
        ""
    };
    eprintln!(
        "\n[nsl] GPU Memory Summary\n\
         ========================\n\
         Allocated:        {} ({} blocks)\n\
         Reserved:         {} ({} segments)\n\
         Free in cache:    {} ({} blocks)\n\
         Peak allocated:   {}\n\
         Peak reserved:    {}\n\
         Driver allocs:    {}\n\
         Driver frees:     {}\n\
         Cache hits:       {}\n\
         Cache misses:     {}\n\
         Block splits:     {}\n\
         Block coalesces:  {}\n\
         Fragmentation:    {}\n\
         Pools:            persistent {} ({} segs), transient {} ({} segs)\n\
         Surfaces:\n{}{}\
         Top contexts:\n{}",
        fmt(s.allocated_bytes), s.num_allocs,
        fmt(s.reserved_bytes), alloc.segments.len(),
        fmt(s.reserved_bytes.saturating_sub(s.allocated_bytes)), s.num_free_blocks,
        fmt(s.peak_allocated_bytes),
        fmt(s.peak_reserved_bytes),
        s.num_driver_allocs,
        s.num_driver_frees,
        s.num_cache_hits,
        s.num_cache_misses,
        s.num_splits,
        s.num_coalesces,
        fmt(s.internal_fragmentation_bytes),
        fmt(p_bytes), p_segs, fmt(t_bytes), t_segs,
        surface_lines,
        async_note,
        top_context_lines,
    );
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};

    /// Mock driver that hands out sequential addresses from a bump allocator.
    struct MockDriver {
        next_addr: AtomicUsize,
        alloc_count: AtomicUsize,
        free_count: AtomicUsize,
    }

    impl MockDriver {
        fn new() -> Self {
            MockDriver {
                // Start at a page-aligned address to mimic real GPU memory
                next_addr: AtomicUsize::new(0x1000_0000),
                alloc_count: AtomicUsize::new(0),
                free_count: AtomicUsize::new(0),
            }
        }
    }

    impl DriverAlloc for MockDriver {
        fn alloc(&self, size: usize) -> Result<*mut c_void, ()> {
            let addr = self.next_addr.fetch_add(size, Ordering::Relaxed);
            self.alloc_count.fetch_add(1, Ordering::Relaxed);
            Ok(addr as *mut c_void)
        }
        fn free(&self, _ptr: *mut c_void) {
            self.free_count.fetch_add(1, Ordering::Relaxed);
        }
        fn memset_zero(&self, _ptr: *mut c_void, _size: usize) {}
    }

    fn make_alloc() -> CachingAllocator<MockDriver> {
        CachingAllocator::with_driver(MockDriver::new())
    }

    #[test]
    fn test_alloc_returns_non_null() {
        let mut a = make_alloc();
        let ptr = a.alloc_with_grow(1024);
        assert!(ptr.is_some());
        assert!(!ptr.unwrap().is_null());
    }

    #[test]
    fn test_alloc_free_reuse() {
        let mut a = make_alloc();
        let p1 = a.alloc_with_grow(1024).unwrap();
        a.free_block(p1);

        // Second alloc of same size should hit cache (no new driver alloc)
        let driver_allocs_before = a.driver.alloc_count.load(Ordering::Relaxed);
        let p2 = a.alloc_from_cache(1024).unwrap();
        let driver_allocs_after = a.driver.alloc_count.load(Ordering::Relaxed);

        assert_eq!(driver_allocs_before, driver_allocs_after, "should reuse cached block");
        assert_eq!(p1, p2, "should return same pointer");
    }

    #[test]
    fn test_split_creates_remainder() {
        let mut a = make_alloc();
        // Allocate a small amount from a 2MB segment — should split
        let p1 = a.alloc_with_grow(512).unwrap();
        assert_eq!(a.stats.num_splits, 1, "should have split the segment");
        assert_eq!(a.stats.num_free_blocks, 1, "remainder should be in free-list");

        // The remainder should be usable
        let p2 = a.alloc_from_cache(1024).unwrap();
        assert!(p2 != p1);
        assert!(!p2.is_null());
    }

    #[test]
    fn test_coalesce_adjacent() {
        let mut a = make_alloc();
        let p1 = a.alloc_with_grow(512).unwrap();
        let p2 = a.alloc_from_cache(512).unwrap();

        // Free both — they should coalesce
        a.free_block(p1);
        a.free_block(p2);

        assert!(a.stats.num_coalesces > 0, "should have coalesced");
    }

    #[test]
    fn test_coalesce_three_way() {
        let mut a = make_alloc();
        let p1 = a.alloc_with_grow(512).unwrap();
        let p2 = a.alloc_from_cache(512).unwrap();
        let p3 = a.alloc_from_cache(512).unwrap();

        // Free outer two, then middle — all three should coalesce
        a.free_block(p1);
        a.free_block(p3);
        let coalesces_before = a.stats.num_coalesces;
        a.free_block(p2);
        assert!(a.stats.num_coalesces > coalesces_before, "middle free should coalesce with both neighbors");
    }

    #[test]
    fn test_best_fit() {
        let mut a = make_alloc();
        // Carve [guard][victim] pairs out of one 2 MB segment. The guards stay
        // allocated so the victims are never adjacent to each other; without
        // them all three victims coalesce back into a single 2 MB block on free
        // and best-fit selection is unobservable.
        let _g0 = a.alloc_with_grow(512).unwrap();
        let p_small = a.alloc_from_cache(512).unwrap();
        let _g1 = a.alloc_from_cache(512).unwrap();
        let p_med = a.alloc_from_cache(2048).unwrap();
        let _g2 = a.alloc_from_cache(512).unwrap();
        let p_large = a.alloc_from_cache(8192).unwrap();
        let _g3 = a.alloc_from_cache(512).unwrap();

        a.free_block(p_small);
        a.free_block(p_med);
        a.free_block(p_large);

        // Free blocks are now 512, 2048, 8192 (plus the segment's large tail).
        // A 1024 request must pick the 2048 block: 512 is too small, and 8192
        // and the tail are larger than necessary.
        let p = a.alloc_from_cache(1024).unwrap();
        assert_eq!(p, p_med, "best-fit should choose the smallest sufficient block");
    }

    /// Regression: the remainder left over from carving a small allocation out
    /// of a 2 MB small segment is itself larger than SMALL_THRESHOLD. It must be
    /// filed in the small pool (its segment's pool) rather than the large pool
    /// (its own size), or no small request can ever reuse it and every small
    /// alloc grows a fresh segment.
    #[test]
    fn test_small_segment_remainder_stays_in_small_pool() {
        let mut a = make_alloc();
        let _p = a.alloc_with_grow(1024).unwrap();

        let remainder = SMALL_SEGMENT_SIZE - 1024;
        assert!(
            !CachingAllocator::<MockDriver>::is_small(remainder),
            "precondition: the remainder is larger than SMALL_THRESHOLD",
        );
        assert_eq!(a.small_free.len(), 1, "remainder belongs to the small pool");
        assert!(a.large_free.is_empty(), "remainder must not land in the large pool");
    }

    #[test]
    fn test_drain_releases_empty_segments() {
        let mut a = make_alloc();
        let p1 = a.alloc_with_grow(512).unwrap();
        a.free_block(p1);

        assert_eq!(a.segments.len(), 1);
        let freed = a.drain_all();
        assert!(freed > 0);
        assert_eq!(a.segments.len(), 0, "empty segment should be released");
        assert_eq!(a.driver.free_count.load(Ordering::Relaxed), 1);
    }

    #[test]
    fn test_drain_preserves_partial_segments() {
        let mut a = make_alloc();
        let _p1 = a.alloc_with_grow(512).unwrap();
        let p2 = a.alloc_from_cache(512).unwrap();
        a.free_block(p2); // p1 still allocated

        let freed = a.drain_all();
        assert_eq!(freed, 0, "should not free segment with live allocations");
        assert_eq!(a.segments.len(), 1);
    }

    #[test]
    fn test_memory_limit() {
        let mut a = make_alloc();
        a.memory_limit = 4 << 20; // 4 MB limit

        // Each alloc_with_grow reserves a fresh SMALL_SEGMENT_SIZE (2 MB) segment.
        let p1 = a.alloc_with_grow(512);
        assert!(p1.is_some());

        // Second segment: 4 MB reserved, exactly at the limit.
        let p2 = a.alloc_with_grow(512);
        assert!(p2.is_some());

        // Third would reserve 6 MB.
        let p3 = a.alloc_with_grow(512);
        assert!(p3.is_none(), "should fail when memory limit exceeded");
    }

    /// A large request grows a whole LARGE_SEGMENT_SIZE (20 MB) segment even when
    /// the request is far smaller, so a modest memory limit rejects it outright.
    #[test]
    fn test_memory_limit_large_pool_growth_is_coarse() {
        let mut a = make_alloc();
        a.memory_limit = 4 << 20; // 4 MB limit

        // 2 MB is a large request; growth reserves 20 MB, which exceeds the limit.
        assert!(a.alloc_with_grow(2 << 20).is_none());

        // Raising the limit past LARGE_SEGMENT_SIZE lets the same request through.
        a.memory_limit = LARGE_SEGMENT_SIZE;
        assert!(a.alloc_with_grow(2 << 20).is_some());
        assert_eq!(a.total_reserved, LARGE_SEGMENT_SIZE);
    }

    #[test]
    fn test_stats_tracking() {
        let mut a = make_alloc();
        let p1 = a.alloc_with_grow(1024).unwrap();
        assert_eq!(a.stats.num_allocs, 1);
        assert!(a.stats.allocated_bytes > 0);
        assert!(a.stats.reserved_bytes > 0);
        assert_eq!(a.stats.num_driver_allocs, 1);
        assert_eq!(a.stats.num_cache_misses, 1);

        a.free_block(p1);
        assert_eq!(a.stats.num_allocs, 0);

        // Re-alloc should be a cache hit
        let _p2 = a.alloc_from_cache(1024).unwrap();
        assert_eq!(a.stats.num_cache_hits, 1);
    }

    #[test]
    fn test_zero_size() {
        let mut a = make_alloc();
        assert!(a.alloc_from_cache(0).is_none());
        assert!(a.alloc_with_grow(0).is_none());
    }

    #[test]
    fn test_large_pool_boundary() {
        let mut a = make_alloc();
        // Exactly at small threshold should use small pool
        let p1 = a.alloc_with_grow(SMALL_THRESHOLD).unwrap();
        assert!(a.segments[0].is_small);
        a.free_block(p1);
        a.drain_all();

        // Just above threshold should use large pool
        let p2 = a.alloc_with_grow(SMALL_THRESHOLD + 1).unwrap();
        assert!(!a.segments[0].is_small);
        a.free_block(p2);
    }

    #[test]
    fn test_round_size() {
        assert_eq!(CachingAllocator::<MockDriver>::round_size(0), 0);
        assert_eq!(CachingAllocator::<MockDriver>::round_size(1), SMALL_ALIGNMENT);
        assert_eq!(CachingAllocator::<MockDriver>::round_size(512), 512);
        assert_eq!(CachingAllocator::<MockDriver>::round_size(513), 1024);
        // Large
        let large = SMALL_THRESHOLD + 1;
        let rounded = CachingAllocator::<MockDriver>::round_size(large);
        assert_eq!(rounded, LARGE_ALIGNMENT); // 2 MB
        assert_eq!(rounded % LARGE_ALIGNMENT, 0);
    }

    #[test]
    fn test_repeated_alloc_free_no_growth() {
        let mut a = make_alloc();
        // Simulate training loop: same-sized allocs repeated. Cache first, then
        // grow on a miss — the order the runtime uses in cuda::mod::alloc.
        for _ in 0..100 {
            let p = a.alloc_from_cache(4096).or_else(|| a.alloc_with_grow(4096)).unwrap();
            a.free_block(p);
        }
        // Should only have 1 driver allocation (the initial segment)
        assert_eq!(a.driver.alloc_count.load(Ordering::Relaxed), 1,
                   "repeated same-size alloc/free should reuse, not grow");
    }

    // ── P0.1 per-surface VRAM accounting ─────────────────────────────

    /// Restore the thread-local surface on scope exit so a failing assert
    /// can't leak a non-default tag into other tests on this thread.
    fn with_default_surface() -> SurfaceGuard {
        SurfaceGuard::new(SurfaceTag::Other)
    }

    #[test]
    fn test_surface_current_and_peak_tracking() {
        let _restore = with_default_surface();
        let mut a = make_alloc();

        set_alloc_surface(SurfaceTag::Weights);
        let p1 = a.alloc_with_grow(1024).unwrap();
        set_alloc_surface(SurfaceTag::Grads);
        let p2 = a.alloc_from_cache(2048).unwrap();

        let w = a.surface_counters[SurfaceTag::Weights as usize];
        let g = a.surface_counters[SurfaceTag::Grads as usize];
        assert_eq!(w.current_bytes, 1024);
        assert_eq!(g.current_bytes, 2048);

        a.free_block(p1);
        a.free_block(p2);
        let w = a.surface_counters[SurfaceTag::Weights as usize];
        let g = a.surface_counters[SurfaceTag::Grads as usize];
        assert_eq!(w.current_bytes, 0, "weights current returns to 0 on free");
        assert_eq!(g.current_bytes, 0, "grads current returns to 0 on free");
        assert_eq!(w.peak_bytes, 1024, "weights peak survives the free");
        assert_eq!(g.peak_bytes, 2048, "grads peak survives the free");
    }

    /// Untagged allocations land on the `Other` surface (the thread-local
    /// default), and the sum over surfaces matches `allocated_bytes`.
    #[test]
    fn test_surface_default_is_other_and_sums_match() {
        let _restore = with_default_surface();
        let mut a = make_alloc();
        let _p = a.alloc_with_grow(4096).unwrap();
        let o = a.surface_counters[SurfaceTag::Other as usize];
        assert_eq!(o.current_bytes, 4096);
        let total: usize = a.surface_counters.iter().map(|c| c.current_bytes).sum();
        assert_eq!(total, a.stats.allocated_bytes);
    }

    /// The at-global-peak snapshot is the per-surface decomposition of the
    /// moment `peak_allocated_bytes` was last set — later churn below the
    /// peak must not disturb it.
    #[test]
    fn test_surface_snapshot_at_global_peak() {
        let _restore = with_default_surface();
        let mut a = make_alloc();

        set_alloc_surface(SurfaceTag::Weights);
        let _p1 = a.alloc_with_grow(1024).unwrap();
        set_alloc_surface(SurfaceTag::Grads);
        let p2 = a.alloc_from_cache(4096).unwrap();
        // Global peak: 5120 bytes = weights 1024 + grads 4096.

        a.free_block(p2);
        set_alloc_surface(SurfaceTag::Activations);
        let _p3 = a.alloc_from_cache(2048).unwrap();
        // Total now 3072 < peak — the snapshot must still show the peak mix.

        assert_eq!(a.stats.peak_allocated_bytes, 5120);
        let w = a.surface_counters[SurfaceTag::Weights as usize];
        let g = a.surface_counters[SurfaceTag::Grads as usize];
        let act = a.surface_counters[SurfaceTag::Activations as usize];
        assert_eq!(w.at_global_peak_bytes, 1024);
        assert_eq!(g.at_global_peak_bytes, 4096);
        assert_eq!(act.at_global_peak_bytes, 0, "activations were not live at the peak");
    }

    #[test]
    fn test_surface_guard_restores_on_drop_and_nests() {
        let _restore = with_default_surface();
        assert_eq!(get_alloc_surface(), SurfaceTag::Other);
        {
            let _g1 = SurfaceGuard::new(SurfaceTag::AttnWorkspace);
            assert_eq!(get_alloc_surface(), SurfaceTag::AttnWorkspace);
            {
                let _g2 = SurfaceGuard::new(SurfaceTag::Grads);
                assert_eq!(get_alloc_surface(), SurfaceTag::Grads);
            }
            assert_eq!(get_alloc_surface(), SurfaceTag::AttnWorkspace);
        }
        assert_eq!(get_alloc_surface(), SurfaceTag::Other);
    }

    #[test]
    fn test_surface_breakdown_names_and_order() {
        let a = make_alloc();
        let b = a.surface_breakdown();
        assert_eq!(b.len(), NUM_SURFACES);
        let names: Vec<&str> = b.iter().map(|(n, _, _)| *n).collect();
        assert_eq!(
            names,
            vec![
                "other", "weights", "optim_m", "optim_v",
                "m_partial", "grads", "activations", "attn_workspace",
            ],
        );
    }

    #[test]
    fn test_surface_tag_from_u8_roundtrip_and_unknown() {
        for t in SurfaceTag::ALL {
            assert_eq!(SurfaceTag::from_u8(t as u8), t);
        }
        assert_eq!(SurfaceTag::from_u8(200), SurfaceTag::Other);
    }

    // ── A1 unified accounting: external (async / direct-device) allocs ────

    fn ext_meta(surface: SurfaceTag, bytes: usize, lifetime: AllocationLifetime) -> AllocationMetadata {
        AllocationMetadata { surface, pool: AllocPool::Transient, bytes, lifetime, op_id: None, tensor_id: None }
    }

    /// External allocations reach the surface counters, allocated_bytes, the
    /// global peak, and the cumulative-count gate — and reverse cleanly.
    #[test]
    fn test_external_alloc_accounted_and_reversible() {
        let _restore = with_default_surface();
        let mut a = make_alloc();
        // A workspace (attn_workspace) allocated directly via cuMemAlloc_v2.
        let ws = 0x1000usize as *mut c_void;
        a.record_external_alloc(ws, ext_meta(SurfaceTag::AttnWorkspace, 4096, AllocationLifetime::DirectDevice));
        assert_eq!(a.stats.allocated_bytes, 4096);
        assert_eq!(a.surface_counters[SurfaceTag::AttnWorkspace as usize].current_bytes, 4096);
        assert_eq!(a.peak_allocated_bytes(), 4096);
        assert_eq!(a.cumulative_alloc_count(), 1);
        assert_eq!(a.stats.num_external_allocs, 1);

        assert!(a.record_external_free(ws), "known ptr frees");
        assert_eq!(a.stats.allocated_bytes, 0);
        assert_eq!(a.surface_counters[SurfaceTag::AttnWorkspace as usize].current_bytes, 0);
        assert_eq!(a.surface_peak(SurfaceTag::AttnWorkspace), 4096, "peak survives free");
        assert!(!a.record_external_free(ws), "double free is a no-op");
    }

    /// The surface-sum invariant holds across a MIX of pooled blocks and
    /// external allocations — the core A1 unification guarantee.
    #[test]
    fn test_external_and_pooled_sum_invariant() {
        let _restore = with_default_surface();
        let mut a = make_alloc();

        set_alloc_surface(SurfaceTag::Weights);
        let p = a.alloc_with_grow(2048).unwrap(); // pooled weights
        set_alloc_surface(SurfaceTag::Other);
        // Async allocation (e.g. NSL_ASYNC_ALLOC path) tagged Grads.
        a.record_external_alloc(0x2000usize as *mut c_void, ext_meta(SurfaceTag::Grads, 8192, AllocationLifetime::Async));

        let total: usize = a.surface_counters.iter().map(|c| c.current_bytes).sum();
        assert_eq!(total, a.stats.allocated_bytes, "surfaces sum to allocated_bytes across pooled+external");
        assert!(a.stats.allocated_bytes >= 2048 + 8192);

        a.free_block(p);
        assert!(a.record_external_free(0x2000usize as *mut c_void));
        let total: usize = a.surface_counters.iter().map(|c| c.current_bytes).sum();
        assert_eq!(total, 0);
        assert_eq!(a.stats.allocated_bytes, 0);
    }

    /// The global-peak decomposition includes external allocations that were
    /// live at the peak (previously these were invisible → under-reported peak).
    #[test]
    fn test_external_contributes_to_global_peak_decomposition() {
        let _restore = with_default_surface();
        let mut a = make_alloc();
        set_alloc_surface(SurfaceTag::Weights);
        let _p = a.alloc_with_grow(1024).unwrap();
        // Direct workspace live at the peak.
        a.record_external_alloc(0x3000usize as *mut c_void, ext_meta(SurfaceTag::AttnWorkspace, 16384, AllocationLifetime::DirectDevice));
        assert_eq!(a.peak_allocated_bytes(), 1024 + 16384);
        assert_eq!(a.surface_at_global_peak(SurfaceTag::AttnWorkspace), 16384);
        assert_eq!(a.surface_at_global_peak(SurfaceTag::Weights), 1024);
    }

    /// `reset_peak_and_counts` re-seeds peaks to current live bytes and zeroes
    /// the cumulative counters — the per-step measurement primitive.
    #[test]
    fn test_reset_peak_and_counts() {
        let _restore = with_default_surface();
        let mut a = make_alloc();
        set_alloc_surface(SurfaceTag::Weights);
        let p = a.alloc_with_grow(4096).unwrap(); // persistent-ish live bytes
        a.record_external_alloc(0x4000usize as *mut c_void, ext_meta(SurfaceTag::AttnWorkspace, 8192, AllocationLifetime::DirectDevice));
        assert!(a.record_external_free(0x4000usize as *mut c_void));
        // Peak captured 4096+8192; cumulative counted 2 allocs.
        assert_eq!(a.peak_allocated_bytes(), 12288);
        assert_eq!(a.cumulative_alloc_count(), 2);

        a.reset_peak_and_counts();
        assert_eq!(a.cumulative_alloc_count(), 0, "counts zeroed");
        assert_eq!(a.peak_allocated_bytes(), 4096, "peak re-seeds to current live bytes");
        assert_eq!(a.surface_peak(SurfaceTag::Weights), 4096);
        assert_eq!(a.surface_peak(SurfaceTag::AttnWorkspace), 0, "freed surface re-seeds to 0");
        a.free_block(p);
    }

    /// The identity thread-local snapshots into `current_alloc_metadata`.
    #[test]
    fn test_alloc_identity_snapshot() {
        let _restore = with_default_surface();
        let _id_restore = IdentityRestore;
        set_alloc_surface(SurfaceTag::Grads);
        set_alloc_identity(42, 1234);
        let m = current_alloc_metadata(256, AllocationLifetime::Async);
        assert_eq!(m.surface, SurfaceTag::Grads);
        assert_eq!(m.op_id, Some(42));
        assert_eq!(m.tensor_id, Some(1234));
        assert_eq!(m.lifetime, AllocationLifetime::Async);
        clear_alloc_identity();
        let m2 = current_alloc_metadata(256, AllocationLifetime::Async);
        assert_eq!(m2.op_id, None);
        assert_eq!(m2.tensor_id, None);
    }

    /// Restore the identity thread-local on scope exit so a failing assert
    /// can't leak a non-default identity into other tests on this thread.
    struct IdentityRestore;
    impl Drop for IdentityRestore {
        fn drop(&mut self) {
            clear_alloc_identity();
        }
    }
}
