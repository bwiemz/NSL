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

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

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
        use cudarc::driver::sys::*;
        unsafe {
            cuMemsetD8_v2(ptr as CUdeviceptr, 0, size);
        }
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
    pub(crate) fn alloc_from_cache(&mut self, size_bytes: usize) -> Option<*mut c_void> {
        if size_bytes == 0 { return None; }
        let rounded = Self::round_size(size_bytes);
        let small = Self::is_small(rounded);
        let search_key = FreeBlockKey { size: rounded, ptr: 0 };

        // Best-fit: find smallest free block >= rounded size
        let found_key = self.free_set(small).range(search_key..).next().copied();
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
            self.split_block(block_ptr, rounded, small);
        }

        // Mark allocated
        let block = unsafe { &mut *block_ptr };
        block.allocated = true;
        block.requested_size = size_bytes;

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
        if self.total_allocated > self.stats.peak_allocated_bytes {
            self.stats.peak_allocated_bytes = self.total_allocated;
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
        let seg_idx = self.segments.len();
        let block = Box::into_raw(Box::new(Block {
            ptr: base_ptr,
            size: segment_size,
            requested_size: size_bytes,
            allocated: false,
            prev: std::ptr::null_mut(),
            next: std::ptr::null_mut(),
            segment_idx: seg_idx,
        }));

        self.segments.push(Segment {
            base_ptr,
            total_size: segment_size,
            blocks_head: block,
            allocated_count: 0,
            is_small: small,
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
            self.split_block(block, rounded, small);
        }

        let blk = unsafe { &mut *block };
        blk.allocated = true;
        blk.requested_size = size_bytes;

        let seg = &mut self.segments[seg_idx];
        seg.allocated_count += 1;

        self.allocated_blocks.insert(base_ptr as usize, block);
        self.total_allocated += blk.size;
        self.stats.num_allocs += 1;
        self.stats.allocated_bytes = self.total_allocated;
        self.stats.internal_fragmentation_bytes += blk.size - size_bytes;
        if self.total_allocated > self.stats.peak_allocated_bytes {
            self.stats.peak_allocated_bytes = self.total_allocated;
        }

        Some(base_ptr)
    }

    /// Split a block: shrink `block` to `new_size` and create a free remainder.
    fn split_block(&mut self, block_ptr: *mut Block, new_size: usize, small: bool) {
        let block = unsafe { &mut *block_ptr };
        let remainder_size = block.size - new_size;
        let remainder_ptr = unsafe { (block.ptr as *mut u8).add(new_size) as *mut c_void };

        let remainder = Box::into_raw(Box::new(Block {
            ptr: remainder_ptr,
            size: remainder_size,
            requested_size: 0,
            allocated: false,
            prev: block_ptr,
            next: block.next,
            segment_idx: block.segment_idx,
        }));

        // Update next block's prev pointer
        if !block.next.is_null() {
            unsafe { (*block.next).prev = remainder; }
        }

        block.next = remainder;
        block.size = new_size;

        // Track the remainder block — use remainder's actual size to choose pool
        let remainder_small = Self::is_small(remainder_size);
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
        block.allocated = false;
        block.requested_size = 0;

        let seg = &mut self.segments[block.segment_idx];
        seg.allocated_count -= 1;
        let small = seg.is_small;

        self.total_allocated -= old_size;
        self.stats.num_allocs -= 1;
        self.stats.allocated_bytes = self.total_allocated;
        self.stats.internal_fragmentation_bytes = self.stats.internal_fragmentation_bytes
            .saturating_sub(old_size - old_requested);

        // Coalesce with next
        let next = block.next;
        if !next.is_null() {
            let next_block = unsafe { &*next };
            if !next_block.allocated {
                // Remove next from free-list (use next's size to find correct pool)
                let next_small = Self::is_small(next_block.size);
                self.free_set_mut(next_small).remove(&FreeBlockKey {
                    size: next_block.size,
                    ptr: next_block.ptr as usize,
                });
                self.stats.num_free_blocks -= 1;

                let block = unsafe { &mut *block_ptr };
                block.size += next_block.size;
                block.next = next_block.next;
                if !next_block.next.is_null() {
                    unsafe { (*next_block.next).prev = block_ptr; }
                }

                // Remove next from tracking and deallocate
                self.all_blocks.remove(&(next_block.ptr as usize));
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
                // Remove prev from free-list (use prev's size to find correct pool)
                let prev_small = Self::is_small(prev_block.size);
                self.free_set_mut(prev_small).remove(&FreeBlockKey {
                    size: prev_block.size,
                    ptr: prev_block.ptr as usize,
                });
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

                // Insert prev (the coalesced block) into free-list
                let merged_small = Self::is_small(merged_size);
                self.free_set_mut(merged_small).insert(FreeBlockKey {
                    size: merged_size,
                    ptr: prev_block.ptr as usize,
                });
                self.stats.num_free_blocks += 1;

                return true;
            }
        }

        // No prev-coalesce: insert the (possibly next-coalesced) block into free-list
        let block = unsafe { &*block_ptr };
        let block_small = Self::is_small(block.size);
        self.free_set_mut(block_small).insert(FreeBlockKey {
            size: block.size,
            ptr: block.ptr as usize,
        });
        self.stats.num_free_blocks += 1;

        true
    }

    /// Drain all fully-free segments back to the driver. Returns bytes freed.
    pub(crate) fn drain_all(&mut self) -> usize {
        let mut freed = 0usize;
        let mut to_remove: Vec<usize> = Vec::new();

        for (idx, seg) in self.segments.iter().enumerate() {
            if seg.allocated_count == 0 {
                to_remove.push(idx);
            }
        }

        // Process in reverse order to maintain valid indices during removal
        to_remove.reverse();
        for idx in to_remove {
            let seg = &self.segments[idx];

            // Walk the block list and remove all blocks from free-set + all_blocks
            let mut bptr = seg.blocks_head;
            while !bptr.is_null() {
                let block = unsafe { &*bptr };
                let next = block.next;

                // Remove from free-list (all blocks in a fully-free segment must be free)
                let blk_small = Self::is_small(block.size);
                self.free_set_mut(blk_small).remove(&FreeBlockKey {
                    size: block.size,
                    ptr: block.ptr as usize,
                });
                self.stats.num_free_blocks = self.stats.num_free_blocks.saturating_sub(1);
                self.all_blocks.remove(&(block.ptr as usize));

                // Deallocate the Block node
                unsafe { drop(Box::from_raw(bptr)); }
                bptr = next;
            }

            // Free the GPU memory
            self.driver.free(seg.base_ptr);
            freed += seg.total_size;
            self.total_reserved -= seg.total_size;
            self.stats.num_driver_frees += 1;

            // Remove segment (swap_remove to avoid O(n) shift)
            self.segments.swap_remove(idx);

            // Fix segment_idx for the segment that was swapped in
            if idx < self.segments.len() {
                let swapped_seg = &self.segments[idx];
                let mut bptr = swapped_seg.blocks_head;
                while !bptr.is_null() {
                    let block = unsafe { &mut *bptr };
                    block.segment_idx = idx;
                    bptr = block.next;
                }
            }
        }

        self.stats.reserved_bytes = self.total_reserved;
        freed
    }

    /// Get a snapshot of current statistics.
    pub(crate) fn stats(&self) -> AllocStats {
        self.stats.clone()
    }

    /// Check if a pointer is managed by this allocator.
    pub(crate) fn is_allocated(&self, ptr: *mut c_void) -> bool {
        self.allocated_blocks.contains_key(&(ptr as usize))
    }
}

// ---------------------------------------------------------------------------
// Global static instance
// ---------------------------------------------------------------------------

pub(super) static CACHING_ALLOCATOR: LazyLock<Mutex<CachingAllocator>> =
    LazyLock::new(|| Mutex::new(CachingAllocator::new()));

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
    let fmt = |b: usize| -> String {
        if b >= 1 << 30 {
            format!("{:.2} GB", b as f64 / (1u64 << 30) as f64)
        } else if b >= 1 << 20 {
            format!("{:.1} MB", b as f64 / (1u64 << 20) as f64)
        } else if b >= 1 << 10 {
            format!("{:.1} KB", b as f64 / (1u64 << 10) as f64)
        } else {
            format!("{} B", b)
        }
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
         Fragmentation:    {}\n",
        fmt(s.allocated_bytes), s.num_allocs,
        fmt(s.reserved_bytes), alloc.segments.len(),
        fmt(s.reserved_bytes.saturating_sub(s.allocated_bytes)), s.num_free_blocks,
        fmt(s.peak_allocated_bytes),
        fmt(s.peak_reserved_bytes),
        s.num_driver_allocs,
        s.num_driver_frees,
        s.num_cache_hits,
        s.num_cache_misses,
        fmt(s.internal_fragmentation_bytes),
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
        // Allocate and free blocks of different sizes
        let p_small = a.alloc_with_grow(512).unwrap();
        let p_med = a.alloc_from_cache(2048).unwrap();
        let p_large = a.alloc_from_cache(8192).unwrap();

        a.free_block(p_small);
        a.free_block(p_med);
        a.free_block(p_large);

        // Request for 1024 should pick the 2048 block (best fit), not the 8192
        let p = a.alloc_from_cache(1024).unwrap();
        // The returned pointer should be the 2048 block (smallest that fits)
        assert_eq!(p, p_med, "best-fit should choose the smallest sufficient block");
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

        // First alloc: 2MB segment
        let p1 = a.alloc_with_grow(512);
        assert!(p1.is_some());

        // Second alloc: would need another 2MB segment, still within 4MB
        let p2 = a.alloc_with_grow(2 << 20);
        assert!(p2.is_some());

        // Third alloc: would exceed 4MB limit
        let p3 = a.alloc_with_grow(2 << 20);
        assert!(p3.is_none(), "should fail when memory limit exceeded");
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
        // Simulate training loop: same-sized allocs repeated
        for _ in 0..100 {
            let p = a.alloc_with_grow(4096).or_else(|| a.alloc_from_cache(4096)).unwrap();
            a.free_block(p);
        }
        // Should only have 1 driver allocation (the initial segment)
        assert_eq!(a.driver.alloc_count.load(Ordering::Relaxed), 1,
                   "repeated same-size alloc/free should reuse, not grow");
    }
}
