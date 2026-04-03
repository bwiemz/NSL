//! CUDA runtime: context management, kernel launch, module cache.
//! Only compiled when the `cuda` feature is enabled.

#[cfg(feature = "cuda")]
use std::ffi::c_void;
// AtomicI64 and Ordering used by inner module functions when cuda feature is enabled

pub(crate) mod kernels;
pub(crate) mod fused_kernels;
pub(crate) mod kernels_hopper;

#[cfg(feature = "cuda")]
pub(crate) mod caching_allocator;

#[cfg(feature = "cuda")]
pub(crate) mod inner {
    use cudarc::driver::sys::*;
    use std::collections::HashMap;
    use std::ffi::c_void;
    use std::sync::{Mutex, OnceLock};

    struct CudaState {
        device: CUdevice,
        #[allow(dead_code)]
        context: CUcontext,
        module_cache: HashMap<usize, CUmodule>,
    }

    // SAFETY: CUcontext/CUmodule are opaque pointers managed by the CUDA driver.
    // We only access CudaState through the Mutex, ensuring single-threaded access.
    unsafe impl Send for CudaState {}

    static CUDA_STATE: OnceLock<Mutex<CudaState>> = OnceLock::new();

    static CUDA_SYNC_MODE: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    pub fn set_cuda_sync_mode(enabled: bool) {
        CUDA_SYNC_MODE.store(enabled, std::sync::atomic::Ordering::Relaxed);
    }

    pub(crate) fn sync_mode_enabled() -> bool {
        CUDA_SYNC_MODE.load(std::sync::atomic::Ordering::Relaxed)
    }

    #[cfg(test)]
    use std::collections::HashMap as TestHashMap;
    #[cfg(test)]
    static CUDA_SIZE_REGISTRY: std::sync::OnceLock<std::sync::Mutex<TestHashMap<usize, usize>>> = std::sync::OnceLock::new();

    #[cfg(test)]
    fn cuda_size_registry() -> &'static std::sync::Mutex<TestHashMap<usize, usize>> {
        CUDA_SIZE_REGISTRY.get_or_init(|| std::sync::Mutex::new(TestHashMap::new()))
    }

    /// Ensure CUDA is initialized. Called from FFI exports.
    pub(crate) fn init() {
        ensure_context();
    }

    /// Ensure the CUDA context is current on the calling thread.
    /// Must be called before any CUDA driver API call.
    pub(crate) fn ensure_context() {
        let s = state();
        let guard = s.lock().unwrap();
        unsafe {
            cuCtxSetCurrent(guard.context);
        }
    }

    fn state() -> &'static Mutex<CudaState> {
        CUDA_STATE.get_or_init(|| {
            unsafe {
                let result = cuInit(0);
                assert_eq!(
                    result,
                    CUresult::CUDA_SUCCESS,
                    "cuInit failed: {:?}",
                    result
                );
                let mut device: CUdevice = 0;
                let result = cuDeviceGet(&mut device, 0);
                assert_eq!(
                    result,
                    CUresult::CUDA_SUCCESS,
                    "cuDeviceGet failed: {:?}",
                    result
                );
                let mut context: CUcontext = std::ptr::null_mut();
                let result = cuDevicePrimaryCtxRetain(&mut context, device);
                assert_eq!(
                    result,
                    CUresult::CUDA_SUCCESS,
                    "cuDevicePrimaryCtxRetain failed: {:?}",
                    result
                );
                let result = cuCtxSetCurrent(context);
                assert_eq!(
                    result,
                    CUresult::CUDA_SUCCESS,
                    "cuCtxSetCurrent failed: {:?}",
                    result
                );
                if std::env::var("NSL_CUDA_SYNC").map(|v| v == "1").unwrap_or(false) {
                    CUDA_SYNC_MODE.store(true, std::sync::atomic::Ordering::Relaxed);
                    eprintln!("[nsl] CUDA sync mode ENABLED — synchronizing after every kernel launch");
                }
                // Register atexit handler for memory stats if NSL_MEMSTATS=1
                if super::caching_allocator::memstats_enabled() {
                    extern "C" {
                        fn atexit(callback: extern "C" fn()) -> i32;
                    }
                    extern "C" fn memstats_atexit() {
                        super::caching_allocator::print_memory_summary();
                    }
                    atexit(memstats_atexit);
                }
                Mutex::new(CudaState {
                    device,
                    context,
                    module_cache: HashMap::new(),
                })
            }
        })
    }

    /// Detect the SM compute capability of the current GPU.
    /// Returns e.g. 90 for Hopper H100, 89 for Ada RTX 4090, 100 for Blackwell B200.
    pub(crate) fn detect_sm_version() -> u32 {
        let s = state();
        let guard = s.lock().unwrap();
        let mut major: i32 = 0;
        let mut minor: i32 = 0;
        unsafe {
            cuDeviceGetAttribute(
                &mut major,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                guard.device,
            );
            cuDeviceGetAttribute(
                &mut minor,
                CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                guard.device,
            );
        }
        (major * 10 + minor) as u32
    }

    static ALLOC_COUNT_DBG: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    // ------------------------------------------------------------------
    // OOM recovery infrastructure
    // ------------------------------------------------------------------

    /// Query free and total device memory in bytes.
    pub(crate) fn query_vram() -> (usize, usize) {
        ensure_context();
        unsafe {
            let mut free: usize = 0;
            let mut total: usize = 0;
            cuMemGetInfo_v2(&mut free, &mut total);
            (free, total)
        }
    }

    /// Drain all cached allocations from the device memory pool.
    /// Releases fully-free segments back to the CUDA driver.
    /// Returns total bytes freed.
    fn pool_drain() -> usize {
        ensure_context();
        let mut alloc = super::caching_allocator::CACHING_ALLOCATOR.lock().unwrap();
        alloc.drain_all()
    }

    /// Drain all cached allocations from the pinned host memory pool.
    fn pinned_pool_drain() -> usize {
        let mut pool = PINNED_POOL.lock().unwrap();
        let mut freed = 0usize;
        ensure_context();
        for (bucket_size, deque) in pool.buckets.iter_mut() {
            while let Some(ptr) = deque.pop_front() {
                unsafe { cuMemFreeHost(ptr); }
                freed += bucket_size;
            }
        }
        pool.buckets.clear();
        freed
    }

    /// Thread-local context string describing the current GPU operation.
    /// Set before allocations so OOM messages identify which op failed.
    ///
    /// NOTE: This is thread-local, so in multi-threaded GPU dispatch the
    /// context reflects the calling thread only. For single-threaded
    /// inference/training (the common case) this is correct.
    thread_local! {
        static OOM_CONTEXT: std::cell::RefCell<String> = std::cell::RefCell::new(String::new());
    }

    pub(crate) fn set_oom_context(ctx: &str) {
        OOM_CONTEXT.with(|c| {
            let mut s = c.borrow_mut();
            s.clear();
            s.push_str(ctx);
        });
    }

    pub(crate) fn current_oom_context() -> String {
        OOM_CONTEXT.with(|c| c.borrow().clone())
    }

    fn format_bytes(bytes: usize) -> String {
        if bytes >= 1024 * 1024 * 1024 {
            format!("{:.2} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
        } else if bytes >= 1024 * 1024 {
            format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
        } else if bytes >= 1024 {
            format!("{:.1} KB", bytes as f64 / 1024.0)
        } else {
            format!("{} B", bytes)
        }
    }

    fn oom_diagnostic(requested: usize, alloc_num: u64, pool_freed: usize) -> String {
        let (free_vram, total_vram) = query_vram();
        let ctx = current_oom_context();
        let op_line = if ctx.is_empty() {
            String::new()
        } else {
            format!("\n  Operation: {}", ctx)
        };
        format!(
            "[nsl] GPU out of memory\n\
             \n\
               Requested:    {} ({})\n\
               VRAM free:    {} / {}\n\
               Pool drained: {}\n\
               Allocation #: {}{}\n\
             \n\
             Suggestions:\n\
               - Reduce batch size or sequence length\n\
               - Enable gradient checkpointing (@checkpoint)\n\
               - Use lower precision (fp16, fp8, int8)\n\
               - Set NSL_ASYNC_ALLOC=1 for stream-ordered allocation",
            requested, format_bytes(requested),
            format_bytes(free_vram), format_bytes(total_vram),
            format_bytes(pool_freed),
            alloc_num, op_line,
        )
    }

    /// Helper: try to alloc from caching allocator (cache hit or grow).
    /// Handles registration and test stats. Returns None on failure.
    fn caching_alloc(size_bytes: usize) -> Option<*mut c_void> {
        let mut alloc = super::caching_allocator::CACHING_ALLOCATOR.lock().unwrap();
        let ptr = alloc.alloc_from_cache(size_bytes)
            .or_else(|| alloc.alloc_with_grow(size_bytes))?;
        drop(alloc);
        register_cuda_alloc(ptr);
        #[cfg(test)]
        crate::memory::stats::cuda_alloc(size_bytes);
        #[cfg(test)]
        cuda_size_registry().lock().unwrap().insert(ptr as usize, size_bytes);
        Some(ptr)
    }

    /// Try async allocation. Returns null on failure (no fallback/recursion).
    fn alloc_async_inner(size_bytes: usize) -> *mut c_void {
        ensure_context();
        unsafe {
            let mut ptr: CUdeviceptr = 0;
            let result = cuMemAllocAsync(&mut ptr, size_bytes, std::ptr::null_mut());
            if result != CUresult::CUDA_SUCCESS || ptr == 0 {
                return std::ptr::null_mut();
            }
            let cptr = ptr as *mut c_void;
            register_cuda_alloc(cptr);
            ASYNC_ALLOC_SET.lock().unwrap().insert(cptr as usize);
            cptr
        }
    }

    /// Try to allocate GPU memory with OOM recovery. Returns None after all
    /// recovery attempts fail. Use this for ops that can fall back to CPU.
    pub(crate) fn try_alloc_managed(size_bytes: usize) -> Option<*mut c_void> {
        if size_bytes == 0 { return None; }

        // Async alloc path (opt-in, bypasses caching allocator)
        if async_alloc_enabled() {
            let ptr = alloc_async_inner(size_bytes);
            if !ptr.is_null() {
                #[cfg(test)]
                crate::memory::stats::cuda_alloc(size_bytes);
                return Some(ptr);
            }
        }

        ensure_context();

        // Attempt 1: caching allocator (cache hit or grow)
        if let Some(ptr) = caching_alloc(size_bytes) {
            return Some(ptr);
        }

        // Attempt 2: synchronize device (flushes pending async frees) and retry
        unsafe { cuCtxSynchronize(); }
        if let Some(ptr) = caching_alloc(size_bytes) {
            return Some(ptr);
        }

        // Attempt 3: drain caching allocator (free idle segments) and retry
        let _drained = pool_drain();
        if let Some(ptr) = caching_alloc(size_bytes) {
            return Some(ptr);
        }

        // Attempt 4: drain pinned pool (reduces system memory pressure) and retry
        let _pinned_drained = pinned_pool_drain();
        if let Some(ptr) = caching_alloc(size_bytes) {
            return Some(ptr);
        }

        None
    }

    /// Allocate device memory (GPU-only, not accessible from CPU).
    /// Uses a caching allocator with block splitting and coalescing to
    /// recycle freed allocations, reducing cuMemAlloc syscall overhead.
    ///
    /// On OOM, attempts recovery (sync + pool drain) before panicking
    /// with detailed VRAM diagnostics.
    ///
    /// IMPORTANT: The returned pointer is a device pointer. CPU code must NOT
    /// dereference it. Use `memcpy_htod` / `memcpy_dtoh` for data transfer.
    pub(crate) fn alloc_managed(size_bytes: usize) -> *mut c_void {
        if size_bytes == 0 { return std::ptr::null_mut(); }

        let n = ALLOC_COUNT_DBG.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        // Async alloc path: use cuMemAllocAsync when enabled (avoids device-wide sync)
        if async_alloc_enabled() {
            let ptr = alloc_async_inner(size_bytes);
            if !ptr.is_null() {
                #[cfg(test)]
                crate::memory::stats::cuda_alloc(size_bytes);
                return ptr;
            }
        }

        ensure_context();

        // Attempt 1: caching allocator (cache hit or grow new segment)
        if let Some(ptr) = caching_alloc(size_bytes) {
            return ptr;
        }

        // === OOM Recovery ===
        // Step 1: synchronize device — flushes pending async frees
        unsafe { cuCtxSynchronize(); }
        if let Some(ptr) = caching_alloc(size_bytes) {
            return ptr;
        }

        // Step 2: drain caching allocator — release idle segments to driver.
        // Retry unconditionally: even if drain freed 0 bytes, cuCtxSynchronize
        // above may have completed async frees the caching allocator can now use.
        let pool_freed = pool_drain();
        if let Some(ptr) = caching_alloc(size_bytes) {
            return ptr;
        }

        // Step 3: drain pinned pool (reduces system memory pressure) and retry
        let _pinned_freed = pinned_pool_drain();
        if let Some(ptr) = caching_alloc(size_bytes) {
            return ptr;
        }

        // All recovery failed
        panic!("{}", oom_diagnostic(size_bytes, n, pool_freed));
    }

    // Track all CUDA allocations so we can validate frees
    use std::collections::HashSet;

    static CUDA_ALLOC_SET: std::sync::LazyLock<std::sync::Mutex<HashSet<usize>>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(HashSet::new()));

    pub(crate) fn register_cuda_alloc(ptr: *mut c_void) {
        if !ptr.is_null() {
            CUDA_ALLOC_SET.lock().unwrap().insert(ptr as usize);
        }
    }

    pub(crate) fn is_cuda_alloc(ptr: *mut c_void) -> bool {
        if ptr.is_null() { return false; }
        CUDA_ALLOC_SET.lock().unwrap().contains(&(ptr as usize))
    }

    // ------------------------------------------------------------------
    // Device memory pool: PyTorch-style caching allocator with block
    // splitting, coalescing, and best-fit search. See caching_allocator.rs.
    // ------------------------------------------------------------------

    /// Free device memory allocated by alloc_managed.
    /// Returns to caching allocator's free-list (with coalescing).
    /// Routes async-allocated pointers to cuMemFreeAsync.
    pub(crate) fn free_managed(ptr: *mut c_void) {
        if ptr.is_null() { return; }
        // Check if this was async-allocated before removing from general set
        if is_async_alloc(ptr) {
            free_async(ptr);
            return;
        }
        let was_cuda = CUDA_ALLOC_SET.lock().unwrap().remove(&(ptr as usize));
        if !was_cuda { return; }
        // Ensure CUDA context BEFORE acquiring CACHING_ALLOCATOR lock.
        // Lock ordering: CUDA_STATE first, then CACHING_ALLOCATOR.
        // Reversing this order causes ABBA deadlock with alloc_managed.
        ensure_context();
        // Return to caching allocator (coalesces with neighbors)
        let mut alloc = super::caching_allocator::CACHING_ALLOCATOR.lock().unwrap();
        if !alloc.free_block(ptr) {
            // Not tracked by caching allocator — direct free (legacy/fallback)
            drop(alloc);
            unsafe {
                let result = cuMemFree_v2(ptr as CUdeviceptr);
                if result != CUresult::CUDA_SUCCESS {
                    eprintln!("nsl: cuMemFree failed: {:?} for {:p}", result, ptr);
                }
            }
        }
    }

    // ------------------------------------------------------------------
    // Pinned staging pool: caches page-locked host memory buffers for
    // CPU↔GPU transfers. Avoids repeated cuMemAllocHost/cuMemFreeHost
    // syscalls during training loops (~50+ transfers per step).
    // ------------------------------------------------------------------

    /// Round up to next power-of-2 for pinned pool bucket sizing.
    fn pinned_bucket_size(requested: usize) -> usize {
        if requested == 0 { return 0; }
        let s = requested.max(4096);
        s.next_power_of_two()
    }

    const PINNED_POOL_MAX_PER_BUCKET: usize = 4;
    const PINNED_POOL_MAX_ALLOC: usize = 64 * 1024 * 1024; // 64 MB

    struct PinnedPool {
        buckets: HashMap<usize, std::collections::VecDeque<*mut c_void>>,
    }
    unsafe impl Send for PinnedPool {}

    static PINNED_POOL: std::sync::LazyLock<std::sync::Mutex<PinnedPool>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(PinnedPool {
            buckets: HashMap::new(),
        }));

    /// Get a pinned (page-locked) staging buffer. Reuses pooled buffers.
    pub(crate) fn staging_alloc(size_bytes: usize) -> *mut c_void {
        if size_bytes == 0 { return std::ptr::null_mut(); }
        if size_bytes > PINNED_POOL_MAX_ALLOC {
            // Large allocations: use regular heap memory instead of pinned.
            // cuMemcpyHtoD/DtoH work with any host pointer; pinned memory
            // only helps with concurrent stream overlap which we don't use.
            // This avoids exhausting CUDA's pinned memory limit during training
            // when transferring large tensors (e.g., logits [B, S, V]).
            return crate::memory::checked_alloc(size_bytes) as *mut c_void;
        }
        let bsize = pinned_bucket_size(size_bytes);
        {
            let mut pool = PINNED_POOL.lock().unwrap();
            if let Some(deque) = pool.buckets.get_mut(&bsize) {
                if let Some(ptr) = deque.pop_front() {
                    return ptr;
                }
            }
        }
        alloc_pinned(bsize) // allocate at bucket size for reuse
    }

    /// Return a pinned staging buffer to the pool.
    pub(crate) fn staging_free(ptr: *mut c_void, size_bytes: usize) {
        if ptr.is_null() { return; }
        if size_bytes > PINNED_POOL_MAX_ALLOC {
            // Large allocations were heap-allocated, not pinned
            unsafe { crate::memory::checked_free(ptr as *mut u8, size_bytes); }
            return;
        }
        let bsize = pinned_bucket_size(size_bytes);
        let mut pool = PINNED_POOL.lock().unwrap();
        let deque = pool.buckets.entry(bsize).or_insert_with(std::collections::VecDeque::new);
        if deque.len() < PINNED_POOL_MAX_PER_BUCKET {
            deque.push_back(ptr);
        } else {
            drop(pool);
            free_pinned(ptr);
        }
    }

    // ------------------------------------------------------------------
    // Stream-ordered async allocation (cudaMallocAsync fallback)
    // Opt-in via NSL_ASYNC_ALLOC=1. Uses CUDA driver's built-in memory
    // pool to avoid device-wide synchronization on allocation.
    // ------------------------------------------------------------------

    static ASYNC_ALLOC_RESULT: OnceLock<bool> = OnceLock::new();

    /// Check if async allocation is enabled and supported.
    /// Uses OnceLock to avoid TOCTOU race on initialization.
    fn async_alloc_enabled() -> bool {
        *ASYNC_ALLOC_RESULT.get_or_init(|| {
            let env_enabled = std::env::var("NSL_ASYNC_ALLOC")
                .map(|v| v == "1")
                .unwrap_or(false);
            if !env_enabled {
                return false;
            }
            // Probe: try to query the default memory pool
            let s = state();
            let guard = s.lock().unwrap();
            let supported = unsafe {
                let mut pool: CUmemoryPool = std::ptr::null_mut();
                let r = cuDeviceGetDefaultMemPool(&mut pool, guard.device);
                r == CUresult::CUDA_SUCCESS && !pool.is_null()
            };
            if supported {
                eprintln!("[nsl] Async GPU allocation ENABLED (cuMemAllocAsync)");
            } else {
                eprintln!("[nsl] NSL_ASYNC_ALLOC=1 but driver does not support memory pools — using sync alloc");
            }
            supported
        })
    }

    /// Track async-allocated pointers (freed via cuMemFreeAsync, not cuMemFree)
    static ASYNC_ALLOC_SET: std::sync::LazyLock<std::sync::Mutex<HashSet<usize>>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(HashSet::new()));

    /// Allocate device memory asynchronously on the default stream.
    /// Falls back to synchronous allocation (with OOM recovery) on failure.
    pub(crate) fn alloc_async(size_bytes: usize) -> *mut c_void {
        let ptr = alloc_async_inner(size_bytes);
        if !ptr.is_null() {
            return ptr;
        }
        // Fallback to synchronous allocation (with OOM recovery)
        alloc_managed(size_bytes)
    }

    /// Free device memory that was allocated via cuMemAllocAsync.
    pub(crate) fn free_async(ptr: *mut c_void) {
        if ptr.is_null() { return; }
        CUDA_ALLOC_SET.lock().unwrap().remove(&(ptr as usize));
        ASYNC_ALLOC_SET.lock().unwrap().remove(&(ptr as usize));
        ensure_context();
        unsafe {
            let result = cuMemFreeAsync(ptr as CUdeviceptr, std::ptr::null_mut());
            if result != CUresult::CUDA_SUCCESS {
                // Fallback: try synchronous free
                let _ = cuMemFree_v2(ptr as CUdeviceptr);
            }
        }
    }

    /// Check if a pointer was async-allocated.
    pub(crate) fn is_async_alloc(ptr: *mut c_void) -> bool {
        if ptr.is_null() { return false; }
        ASYNC_ALLOC_SET.lock().unwrap().contains(&(ptr as usize))
    }

    /// Allocate device-only memory (not accessible from host without explicit copy).
    /// On OOM, attempts sync + pool drain recovery before panicking.
    pub(crate) fn alloc_device(size_bytes: usize) -> *mut c_void {
        ensure_context();
        unsafe {
            let mut ptr: CUdeviceptr = 0;
            let result = cuMemAlloc_v2(&mut ptr, size_bytes);
            if result == CUresult::CUDA_SUCCESS && ptr != 0 {
                return ptr as *mut c_void;
            }

            // Non-OOM errors: panic immediately
            if result != CUresult::CUDA_SUCCESS
                && !matches!(result, CUresult::CUDA_ERROR_OUT_OF_MEMORY)
            {
                if matches!(result, CUresult::CUDA_ERROR_ILLEGAL_ADDRESS) {
                    panic!(
                        "cuMemAlloc({} bytes) failed with CUDA_ERROR_ILLEGAL_ADDRESS.\n\
                         A prior GPU kernel accessed invalid memory.\n\
                         Re-run with: nsl run --cuda-sync <file>",
                        size_bytes
                    );
                }
                panic!("cuMemAlloc({} bytes) failed: {:?}", size_bytes, result);
            }

            // OOM recovery: sync and retry
            cuCtxSynchronize();
            ptr = 0;
            let result = cuMemAlloc_v2(&mut ptr, size_bytes);
            if result == CUresult::CUDA_SUCCESS && ptr != 0 {
                return ptr as *mut c_void;
            }

            // Drain pool and retry unconditionally: even if drain freed 0 bytes,
            // cuCtxSynchronize above may have completed async frees the caching
            // allocator can now use.
            let pool_freed = pool_drain();
            ptr = 0;
            let result = cuMemAlloc_v2(&mut ptr, size_bytes);
            if result == CUresult::CUDA_SUCCESS && ptr != 0 {
                return ptr as *mut c_void;
            }

            let n = ALLOC_COUNT_DBG.load(std::sync::atomic::Ordering::Relaxed);
            panic!("{}", oom_diagnostic(size_bytes, n, pool_freed));
        }
    }

    /// Free device-only memory allocated with `alloc_device`.
    pub(crate) fn free_device(ptr: *mut c_void) {
        ensure_context();
        unsafe {
            let result = cuMemFree_v2(ptr as CUdeviceptr);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemFree_v2 (device) failed: {:?}",
                result
            );
        }
    }

    /// Allocate pinned (page-locked) host memory for fast DMA transfers.
    pub(crate) fn alloc_pinned(size_bytes: usize) -> *mut c_void {
        ensure_context();
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let result = cuMemAllocHost_v2(&mut ptr, size_bytes);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemAllocHost_v2({} bytes) failed: {:?}",
                size_bytes,
                result
            );
            ptr
        }
    }

    /// Free pinned host memory allocated with `alloc_pinned`.
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

    /// Copy `size_bytes` bytes from host memory to device memory.
    pub(crate) fn memcpy_htod(dst_device: *mut c_void, src_host: *const c_void, size_bytes: usize) {
        ensure_context();
        unsafe {
            let result = cuMemcpyHtoD_v2(dst_device as CUdeviceptr, src_host, size_bytes);
            let ctx = current_oom_context();
            let ctx_suffix = if ctx.is_empty() {
                String::new()
            } else {
                format!(" [context: {}]", ctx)
            };
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemcpyHtoD_v2({} bytes) failed: {:?}{}",
                size_bytes,
                result,
                ctx_suffix
            );
        }
    }

    /// Copy `size_bytes` bytes from device memory to host memory.
    pub(crate) fn memcpy_dtoh(dst_host: *mut c_void, src_device: *const c_void, size_bytes: usize) {
        ensure_context();
        unsafe {
            let result = cuMemcpyDtoH_v2(dst_host, src_device as CUdeviceptr, size_bytes);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemcpyDtoH_v2({} bytes) failed: {:?}",
                size_bytes,
                result
            );
        }
    }

    /// Copy `size_bytes` bytes from one device pointer to another.
    pub(crate) fn memcpy_dtod(dst_device: *mut c_void, src_device: *const c_void, size_bytes: usize) {
        ensure_context();
        unsafe {
            let result = cuMemcpyDtoD_v2(dst_device as CUdeviceptr, src_device as CUdeviceptr, size_bytes);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemcpyDtoD_v2({} bytes) failed: {:?}",
                size_bytes,
                result
            );
        }
    }

    /// Zero-fill device memory.
    pub(crate) fn memset_d8(device_ptr: *mut c_void, size_bytes: usize) {
        ensure_context();
        unsafe {
            let result = cuMemsetD8_v2(device_ptr as CUdeviceptr, 0, size_bytes);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemsetD8_v2({} bytes) failed: {:?}",
                size_bytes,
                result
            );
        }
    }

    /// Prefetch memory to device. Best-effort: silently ignores NOT_SUPPORTED.
    /// NOTE: Only meaningful for unified memory. With device memory this is a no-op.
    pub(crate) fn prefetch_to_device(ptr: *mut c_void, size_bytes: usize, device_id: i32) {
        let state = state();
        let _guard = state.lock().unwrap();
        unsafe {
            let location = CUmemLocation {
                type_: CUmemLocationType::CU_MEM_LOCATION_TYPE_DEVICE,
                __bindgen_anon_1: cudarc::driver::sys::CUmemLocation_st__bindgen_ty_1 { id: device_id },
            };
            let result = cuMemPrefetchAsync_v2(
                ptr as CUdeviceptr,
                size_bytes,
                location,
                0, // flags
                std::ptr::null_mut(), // default stream
            );
            if result != CUresult::CUDA_SUCCESS
                && result != CUresult::CUDA_ERROR_NOT_SUPPORTED
                && result != CUresult::CUDA_ERROR_INVALID_DEVICE
                && result != CUresult::CUDA_ERROR_INVALID_VALUE
            {
                // Don't panic on prefetch failures — device memory (cuMemAlloc_v2)
                // doesn't support prefetch (that's a unified memory API).
                // Log and continue.
                eprintln!("[nsl] cuMemPrefetchAsync warning: {:?} (non-fatal)", result);
            }
        }
    }

    // === cuEvent wrappers for kernel profiler ===

    pub unsafe fn cu_event_create(event: *mut u64) {
        cuEventCreate(event as *mut CUevent, 0); // CU_EVENT_DEFAULT = 0
    }

    pub unsafe fn cu_event_record(event: u64, stream: *mut std::ffi::c_void) {
        cuEventRecord(event as CUevent, stream as CUstream);
    }

    pub unsafe fn cu_event_elapsed_time(ms: *mut f32, start: u64, stop: u64) {
        cuEventElapsedTime_v2(ms, start as CUevent, stop as CUevent);
    }

    pub unsafe fn cu_event_destroy(event: u64) {
        cuEventDestroy_v2(event as CUevent);
    }

    pub unsafe fn cu_ctx_synchronize() {
        cuCtxSynchronize();
    }

    /// Launch a PTX kernel. `ptx_ptr` and `name_ptr` must point to null-terminated C strings.
    /// `args` is a slice of pointers to argument values (as required by `cuLaunchKernel`).
    pub(crate) fn kernel_launch(
        ptx_ptr: *const u8,
        name_ptr: *const u8,
        grid: [i64; 3],
        block: [i64; 3],
        args: &[*mut c_void],
        shared_mem_bytes: u32,
    ) -> CUresult {
        let state = state();
        let func = {
            let mut guard = state.lock().unwrap();
            unsafe { cuCtxSetCurrent(guard.context); }

            // Cache modules by PTX pointer address (stable .rodata addresses)
            let cache_key = ptx_ptr as usize;
            let module = if let Some(m) = guard.module_cache.get(&cache_key) {
                *m
            } else {
                let mut module: CUmodule = std::ptr::null_mut();
                let res = unsafe { cuModuleLoadData(&mut module, ptx_ptr as *const c_void) };
                if res != CUresult::CUDA_SUCCESS { return res; }
                guard.module_cache.insert(cache_key, module);
                module
            };

            let name = unsafe { std::ffi::CStr::from_ptr(name_ptr as *const i8) };
            let mut func: CUfunction = std::ptr::null_mut();
            let res = unsafe { cuModuleGetFunction(&mut func, module, name.as_ptr()) };
            if res != CUresult::CUDA_SUCCESS { return res; }

            func
        }; // guard dropped here — no lock held for CUDA calls

        // Profiler: pop event pair (lock-pop-unlock on profiler mutex)
        let profiler_events = if crate::kernel_profiler::kernel_profiler_enabled() {
            crate::kernel_profiler::kernel_profiler_pop_events()
        } else {
            None
        };

        // Record start event before launch
        if let Some((start, _, _)) = &profiler_events {
            unsafe { cuEventRecord(*start as CUevent, std::ptr::null_mut()); }
        }

        // Validate launch dimensions
        debug_assert!(grid[0] > 0 && grid[1] > 0 && grid[2] > 0,
            "kernel_launch: invalid grid dimensions {:?}", grid);
        debug_assert!(block[0] > 0 && block[1] > 0 && block[2] > 0,
            "kernel_launch: invalid block dimensions {:?}", block);
        debug_assert!(block[0] * block[1] * block[2] <= 1024,
            "kernel_launch: block size {} exceeds max 1024 threads",
            block[0] * block[1] * block[2]);

        // Launch kernel (no lock held)
        let mut kernel_args: Vec<*mut c_void> = args.to_vec();
        let res = unsafe {
            cuLaunchKernel(
                func,
                grid[0] as u32, grid[1] as u32, grid[2] as u32,
                block[0] as u32, block[1] as u32, block[2] as u32,
                shared_mem_bytes, std::ptr::null_mut(),
                kernel_args.as_mut_ptr(), std::ptr::null_mut(),
            )
        };

        // Sync after launch if sync mode is enabled (surfaces async GPU errors)
        if sync_mode_enabled() {
            let sync_result = unsafe { cuCtxSynchronize() };
            if sync_result != CUresult::CUDA_SUCCESS {
                let name_cstr = unsafe { std::ffi::CStr::from_ptr(name_ptr as *const std::ffi::c_char) };
                let name_str = name_cstr.to_string_lossy();
                panic!(
                    "[nsl] CUDA async error after kernel '{}' (grid={:?}, block={:?}, shared={}B): {:?}",
                    name_str, grid, block, shared_mem_bytes, sync_result
                );
            }
        }

        // Record stop event after launch
        if let Some((_, stop, _)) = &profiler_events {
            unsafe { cuEventRecord(*stop as CUevent, std::ptr::null_mut()); }
            // Push trace (lock-push-unlock on profiler mutex)
            let name = unsafe { std::ffi::CStr::from_ptr(name_ptr as *const i8) };
            let name_str = name.to_str().unwrap_or("unknown");
            crate::kernel_profiler::kernel_profiler_push_trace(
                name_str,
                [grid[0] as u32, grid[1] as u32, grid[2] as u32],
                [block[0] as u32, block[1] as u32, block[2] as u32],
            );
        }

        // NOTE: cuCtxSynchronize removed — same-stream kernels are implicitly
        // ordered; profiler flush performs single sync at program end
        res
    }
}

#[cfg(feature = "cuda")]
pub(crate) use inner::{cu_event_create, cu_event_record, cu_event_elapsed_time, cu_event_destroy, cu_ctx_synchronize};

// === GPU op helpers ===

/// CPU fallback for a binary elementwise op when GPU allocation fails.
#[cfg(feature = "cuda")]
fn cpu_fallback_binary(a_ptr: i64, b_ptr: i64, kernel_name: &str) -> i64 {
    let a = unsafe { &*(a_ptr as *const crate::tensor::NslTensor) };
    let a_cpu = if a.device > 0 { crate::tensor::nsl_tensor_to_device(a_ptr, 0) } else { a_ptr };
    let b_cpu = if unsafe { &*(b_ptr as *const crate::tensor::NslTensor) }.device > 0 {
        crate::tensor::nsl_tensor_to_device(b_ptr, 0)
    } else {
        b_ptr
    };
    let op_fn: fn(f64, f64) -> f64 = match kernel_name.trim_end_matches('\0') {
        "nsl_add_f32" => |x, y| x + y,
        "nsl_sub_f32" => |x, y| x - y,
        "nsl_mul_f32" => |x, y| x * y,
        "nsl_div_f32" => |x, y| x / y,
        _ => |x, y| x + y,
    };
    let result_cpu = crate::cpu::tensor_elementwise_op(a_cpu, b_cpu, op_fn);
    // Transfer back to GPU — alloc_managed will retry/drain internally
    let result_gpu = crate::tensor::nsl_tensor_to_device(result_cpu, a.device as i64);
    if a_cpu != a_ptr { crate::tensor::nsl_tensor_free(a_cpu); }
    if b_cpu != b_ptr { crate::tensor::nsl_tensor_free(b_cpu); }
    crate::tensor::nsl_tensor_free(result_cpu);
    result_gpu
}

#[cfg(feature = "cuda")]
fn gpu_broadcast_shape(a: &crate::tensor::NslTensor, b: &crate::tensor::NslTensor) -> Option<Vec<i64>> {
    let a_shape = unsafe { std::slice::from_raw_parts(a.shape, a.ndim as usize) };
    let b_shape = unsafe { std::slice::from_raw_parts(b.shape, b.ndim as usize) };
    let out_ndim = a_shape.len().max(b_shape.len());
    let mut out_shape = vec![1i64; out_ndim];
    for i in 0..out_ndim {
        let a_dim = if i < out_ndim - a_shape.len() { 1 } else { a_shape[i - (out_ndim - a_shape.len())] };
        let b_dim = if i < out_ndim - b_shape.len() { 1 } else { b_shape[i - (out_ndim - b_shape.len())] };
        if a_dim == b_dim || a_dim == 1 || b_dim == 1 {
            out_shape[i] = a_dim.max(b_dim);
        } else {
            return None;
        }
    }
    Some(out_shape)
}

#[cfg(feature = "cuda")]
fn gpu_prepare_binary_operand(ptr: i64, out_shape: &[i64]) -> (i64, bool) {
    let tensor = unsafe { &*(ptr as *const crate::tensor::NslTensor) };
    let current_shape = unsafe { std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize) };
    let needs_expand = current_shape != out_shape;

    if !needs_expand && tensor.is_contiguous() {
        return (ptr, false);
    }

    let contig = if needs_expand {
        let shape_list = crate::list::nsl_list_new();
        for &dim in out_shape {
            crate::list::nsl_list_push(shape_list, dim);
        }
        let expanded = crate::tensor::nsl_tensor_expand(ptr, shape_list);
        crate::list::nsl_list_free(shape_list);
        let contig = crate::tensor::nsl_tensor_contiguous(expanded);
        crate::tensor::nsl_tensor_free(expanded);
        contig
    } else {
        crate::tensor::nsl_tensor_contiguous(ptr)
    };

    (contig, contig != ptr)
}

#[cfg(feature = "cuda")]
fn gpu_prepare_binary_operands(a_ptr: i64, b_ptr: i64) -> Option<(i64, i64, bool, bool)> {
    let a = unsafe { &*(a_ptr as *const crate::tensor::NslTensor) };
    let b = unsafe { &*(b_ptr as *const crate::tensor::NslTensor) };
    let out_shape = gpu_broadcast_shape(a, b)?;
    let (a_prepared, free_a) = gpu_prepare_binary_operand(a_ptr, &out_shape);
    let (b_prepared, free_b) = gpu_prepare_binary_operand(b_ptr, &out_shape);
    Some((a_prepared, b_prepared, free_a, free_b))
}

/// GPU elementwise binary op.
/// Falls back to CPU on GPU OOM (transfers to CPU, computes, transfers back).
#[cfg(feature = "cuda")]
pub(crate) fn gpu_elementwise_binary(a_ptr: i64, b_ptr: i64, ptx: &str, kernel_name: &str) -> i64 {
    use crate::tensor::NslTensor;
    inner::set_oom_context(kernel_name.trim_end_matches('\0'));
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };
    if a.len != b.len || !a.is_contiguous() || !b.is_contiguous() {
        if let Some((prepared_a, prepared_b, free_a, free_b)) = gpu_prepare_binary_operands(a_ptr, b_ptr) {
            if prepared_a != a_ptr || prepared_b != b_ptr {
                let result = gpu_elementwise_binary(prepared_a, prepared_b, ptx, kernel_name);
                if free_a {
                    crate::tensor::nsl_tensor_free(prepared_a);
                }
                if free_b {
                    crate::tensor::nsl_tensor_free(prepared_b);
                }
                return result;
            }
        }
        // Fall back to CPU only when GPU broadcast materialization is impossible.
        if a.len != b.len {
            return cpu_fallback_binary(a_ptr, b_ptr, kernel_name);
        }
    }

    let n = a.len as usize;
    // Try GPU allocation — fall back to CPU on OOM
    let out_data = match inner::try_alloc_managed(n * 4) {
        Some(ptr) => ptr,
        None => {
            eprintln!("[nsl] GPU OOM in {} — falling back to CPU", kernel_name.trim_end_matches('\0'));
            return cpu_fallback_binary(a_ptr, b_ptr, kernel_name);
        }
    };
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);
    let out = Box::new(NslTensor::new(
        out_data,
        shape,
        strides,
        a.ndim,
        a.len,
        a.device,
        1,
        1,
        0,
    ));
    let out_ptr = Box::into_raw(out);
    let out_t = unsafe { &*out_ptr };

    let mut a_data = a.data as u64;
    let mut b_data = b.data as u64;
    let mut c_data = out_t.data as u64;
    let mut n_val = n as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut b_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    if result as u32 != 0 {
        // Free the allocated tensor+data to avoid leak on kernel failure
        eprintln!("GPU kernel '{}' failed: {}", kernel_name.trim_end_matches('\0'), result as u32);
        unsafe { let _ = Box::from_raw(out_ptr); }
        inner::free_managed(out_data);
        return 0;
    }
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    out_ptr as i64
}

/// CPU fallback for a unary elementwise op when GPU allocation fails.
#[cfg(feature = "cuda")]
fn cpu_fallback_unary(a_ptr: i64, kernel_name: &str) -> i64 {
    let a = unsafe { &*(a_ptr as *const crate::tensor::NslTensor) };
    let a_cpu = if a.device > 0 { crate::tensor::nsl_tensor_to_device(a_ptr, 0) } else { a_ptr };
    let op_fn: fn(f64) -> f64 = match kernel_name.trim_end_matches('\0') {
        "nsl_neg_f32" => |x| -x,
        "nsl_relu_f32" => |x| if x > 0.0 { x } else { 0.0 },
        "nsl_exp_f32" => |x| x.exp(),
        "nsl_log_f32" => |x| x.ln(),
        "nsl_sqrt_f32" => |x| x.sqrt(),
        "nsl_abs_f32" => |x| x.abs(),
        "nsl_sign_f32" => |x| if x > 0.0 { 1.0 } else if x < 0.0 { -1.0 } else { 0.0 },
        "nsl_sigmoid_f32" => |x| 1.0 / (1.0 + (-x).exp()),
        "nsl_tanh_f32" => |x| x.tanh(),
        _ => |x| x,
    };
    let a_t = unsafe { &*(a_cpu as *const crate::tensor::NslTensor) };
    let n = a_t.len as usize;
    let src = a_t.data as *const f64;
    let result_ptr = crate::tensor::nsl_tensor_zeros_like(a_cpu);
    let result_t = unsafe { &*(result_ptr as *const crate::tensor::NslTensor) };
    let dst = result_t.data as *mut f64;
    for i in 0..n {
        unsafe { *dst.add(i) = op_fn(*src.add(i)); }
    }
    let result_gpu = crate::tensor::nsl_tensor_to_device(result_ptr, a.device as i64);
    if a_cpu != a_ptr { crate::tensor::nsl_tensor_free(a_cpu); }
    crate::tensor::nsl_tensor_free(result_ptr);
    result_gpu
}

/// GPU elementwise unary op.
/// Falls back to CPU on GPU OOM.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_elementwise_unary(a_ptr: i64, ptx: &str, kernel_name: &str) -> i64 {
    use crate::tensor::NslTensor;
    inner::set_oom_context(kernel_name.trim_end_matches('\0'));
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let n = a.len as usize;
    let out_data = match inner::try_alloc_managed(n * 4) {
        Some(ptr) => ptr,
        None => {
            eprintln!("[nsl] GPU OOM in {} — falling back to CPU", kernel_name.trim_end_matches('\0'));
            return cpu_fallback_unary(a_ptr, kernel_name);
        }
    };
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);
    let out = Box::new(NslTensor::new(
        out_data,
        shape,
        strides,
        a.ndim,
        a.len,
        a.device,
        1,
        1,
        0,
    ));
    let out_ptr = Box::into_raw(out);
    let out_t = unsafe { &*out_ptr };

    let mut a_data = a.data as u64;
    let mut c_data = out_t.data as u64;
    let mut n_val = n as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU kernel '{}' failed: {}", kernel_name.trim_end_matches('\0'), result as u32);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    out_ptr as i64
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_rotate_half_f32(tensor_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;

    fn cpu_fallback_rotate_half(ptr: i64, device: i32) -> i64 {
        let cpu_input = crate::tensor::nsl_tensor_to_device(ptr, 0);
        let cpu_rotated = crate::tensor::shape_ops::nsl_tensor_rotate_half(cpu_input);
        let gpu_rotated = crate::tensor::nsl_tensor_to_device(cpu_rotated, device as i64);
        crate::tensor::nsl_tensor_free(cpu_input);
        crate::tensor::nsl_tensor_free(cpu_rotated);
        gpu_rotated
    }

    const KERNEL_NAME: &str = "nsl_rotate_half_f32\0";

    inner::set_oom_context(KERNEL_NAME.trim_end_matches('\0'));

    let tensor = unsafe { &*(tensor_ptr as *const NslTensor) };
    assert!(tensor.device > 0, "gpu_rotate_half_f32 requires a CUDA tensor");

    let contiguous_ptr = if tensor.is_contiguous() {
        tensor_ptr
    } else {
        crate::tensor::nsl_tensor_contiguous(tensor_ptr)
    };
    let contiguous = unsafe { &*(contiguous_ptr as *const NslTensor) };

    let ndim = contiguous.ndim as usize;
    assert!(ndim > 0, "nsl: rotate_half requires at least 1 dimension");

    let last_dim = unsafe { *contiguous.shape.add(ndim - 1) } as usize;
    assert!(
        last_dim.is_multiple_of(2),
        "nsl: rotate_half requires even last dimension, got {}",
        last_dim
    );

    if contiguous.dtype != 1 {
        let result = cpu_fallback_rotate_half(contiguous_ptr, tensor.device as i32);
        if contiguous_ptr != tensor_ptr {
            crate::tensor::nsl_tensor_free(contiguous_ptr);
        }
        return result;
    }

    let n = contiguous.len as usize;
    let out_data = match inner::try_alloc_managed(n * 4) {
        Some(ptr) => ptr,
        None => {
            let result = cpu_fallback_rotate_half(contiguous_ptr, tensor.device as i32);
            if contiguous_ptr != tensor_ptr {
                crate::tensor::nsl_tensor_free(contiguous_ptr);
            }
            return result;
        }
    };

    let shape = NslTensor::copy_shape(contiguous.shape, contiguous.ndim);
    let strides = NslTensor::compute_strides(shape, contiguous.ndim);
    let out = Box::new(NslTensor::new(
        out_data,
        shape,
        strides,
        contiguous.ndim,
        contiguous.len,
        contiguous.device,
        1,
        1,
        0,
    ));
    let out_ptr = Box::into_raw(out);
    let out_t = unsafe { &*out_ptr };

    let mut a_data = contiguous.data as u64;
    let mut c_data = out_t.data as u64;
    let mut n_val = n as u64;
    let mut last_dim_val = last_dim as u64;
    let mut half_val = (last_dim / 2) as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut last_dim_val as *mut _ as *mut std::ffi::c_void,
        &mut half_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        kernels::ROTATE_HALF_F32_PTX.as_ptr(),
        KERNEL_NAME.as_ptr(),
        [grid, 1, 1],
        [block, 1, 1],
        &args,
        0,
    );
    assert_eq!(
        result as u32,
        0,
        "GPU kernel '{}' failed: {}",
        KERNEL_NAME.trim_end_matches('\0'),
        result as u32
    );
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    if contiguous_ptr != tensor_ptr {
        crate::tensor::nsl_tensor_free(contiguous_ptr);
    }
    out_ptr as i64
}

/// GPU elementwise unary op — in-place (FBIP). Writes output to input buffer.
/// Caller must have verified `can_mutate_inplace_gpu()`.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_elementwise_unary_inplace(a_ptr: i64, ptx: &str, kernel_name: &str) {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let n = a.len as usize;

    let mut a_data = a.data as u64;
    let mut c_data = a.data as u64; // output = input buffer
    let mut n_val = n as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU inplace kernel '{}' failed: {}", kernel_name.trim_end_matches('\0'), result as u32);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
}

/// GPU elementwise binary op — in-place (FBIP). Writes output to left operand's buffer.
/// Caller must have verified `can_mutate_inplace_gpu()` on `a` and shapes match.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_elementwise_binary_inplace(a_ptr: i64, b_ptr: i64, ptx: &str, kernel_name: &str) {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };
    assert_eq!(a.len, b.len, "GPU inplace elementwise: length mismatch");

    let n = a.len as usize;
    let mut a_data = a.data as u64;
    let mut b_data = b.data as u64;
    let mut c_data = a.data as u64; // output = left operand buffer
    let mut n_val = n as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut b_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU inplace kernel '{}' failed: {}", kernel_name.trim_end_matches('\0'), result as u32);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
}

/// GPU scalar op — in-place (FBIP). Writes output to input buffer.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_scalar_op_inplace(a_ptr: i64, scalar: f32, ptx: &str, kernel_name: &str) {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let n = a.len as usize;

    let mut a_data = a.data as u64;
    let mut c_data = a.data as u64; // output = input buffer
    let mut s_val = scalar;
    let mut n_val = n as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut s_val as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU inplace scalar kernel '{}' failed: {}", kernel_name.trim_end_matches('\0'), result as u32);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
}

/// GPU matrix multiplication: C[M,N] = A[M,K] @ B[K,N], f32 inputs.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_matmul_f32(a_ptr: i64, b_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    inner::set_oom_context("matmul_f32");
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };

    assert!(a.ndim >= 2 && b.ndim >= 2, "matmul requires 2D+ tensors");

    let a_shape = unsafe { std::slice::from_raw_parts(a.shape, a.ndim as usize) };
    let b_shape = unsafe { std::slice::from_raw_parts(b.shape, b.ndim as usize) };

    let a_nd = a.ndim as usize;
    let b_nd = b.ndim as usize;

    let m = a_shape[a_nd - 2] as u64;
    let k = a_shape[a_nd - 1] as u64;
    let k2 = b_shape[b_nd - 2] as u64;
    let n = b_shape[b_nd - 1] as u64;
    assert_eq!(k, k2, "matmul inner dimension mismatch: {} vs {}", k, k2);

    // Compute broadcast batch dimensions (all dims before last two)
    let a_batch = &a_shape[..a_nd - 2];
    let b_batch = &b_shape[..b_nd - 2];
    let max_batch_nd = a_batch.len().max(b_batch.len());

    let mut out_batch: Vec<i64> = Vec::with_capacity(max_batch_nd);
    for i in 0..max_batch_nd {
        let a_dim = if i < max_batch_nd - a_batch.len() { 1 } else { a_batch[i - (max_batch_nd - a_batch.len())] };
        let b_dim = if i < max_batch_nd - b_batch.len() { 1 } else { b_batch[i - (max_batch_nd - b_batch.len())] };
        assert!(a_dim == b_dim || a_dim == 1 || b_dim == 1,
            "matmul batch dim mismatch at {}: {} vs {}", i, a_dim, b_dim);
        out_batch.push(a_dim.max(b_dim));
    }

    let total_batch: u64 = out_batch.iter().product::<i64>().max(1) as u64;
    let out_nd = out_batch.len() + 2;

    // Build output shape: batch_dims + [m, n]
    let mut out_shape_vec: Vec<i64> = out_batch.clone();
    out_shape_vec.push(m as i64);
    out_shape_vec.push(n as i64);

    let out_total = (total_batch * m * n) as usize;
    let out_data = inner::alloc_managed(out_total * 4); // f32

    let shape = crate::memory::checked_alloc(out_nd * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &s) in out_shape_vec.iter().enumerate() {
        unsafe { *shape.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape, out_nd as i64);

    let out = Box::new(NslTensor::new(
        out_data,
        shape,
        strides,
        out_nd as i64,
        out_total as i64,
        a.device,
        1,
        1,
        0,
    ));
    let out_ptr = Box::into_raw(out);

    // Compute per-batch strides (elements, not bytes)
    let a_mat_stride = m * k; // elements per batch slice in A
    let b_mat_stride = k * n; // elements per batch slice in B
    let c_mat_stride = m * n; // elements per batch slice in C

    // Broadcast: stride=0 means A or B is shared across all batches
    let a_total_batch: u64 = a_batch.iter().product::<i64>().max(1) as u64;
    let b_total_batch: u64 = b_batch.iter().product::<i64>().max(1) as u64;
    let stride_a = if a_total_batch == 1 { 0u64 } else { a_mat_stride };
    let stride_b = if b_total_batch == 1 { 0u64 } else { b_mat_stride };
    let stride_c = c_mat_stride;

    if total_batch == 1 {
        // Non-batched: use original 2D matmul kernel (simpler, no z-grid overhead)
        let block = 16i64;
        let grid_x = ((n as i64) + block - 1) / block;
        let grid_y = ((m as i64) + block - 1) / block;

        let mut a_data = a.data as u64;
        let mut b_data = b.data as u64;
        let mut c_data = out_data as u64;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;

        let args = [
            &mut a_data as *mut _ as *mut std::ffi::c_void,
            &mut b_data as *mut _ as *mut std::ffi::c_void,
            &mut c_data as *mut _ as *mut std::ffi::c_void,
            &mut m_val as *mut _ as *mut std::ffi::c_void,
            &mut n_val as *mut _ as *mut std::ffi::c_void,
            &mut k_val as *mut _ as *mut std::ffi::c_void,
        ];

        let result = inner::kernel_launch(
            kernels::MATMUL_F32_PTX.as_ptr(),
            "nsl_matmul_f32\0".as_ptr(),
            [grid_x, grid_y, 1],
            [block, block, 1],
            &args, 0,
        );
        assert_eq!(result as u32, 0, "GPU matmul kernel failed: {}", result as u32);
    } else {
        // Batched: single launch with blockIdx.z = batch dimension
        let block = 16i64;
        let grid_x = ((n as i64) + block - 1) / block;
        let grid_y = ((m as i64) + block - 1) / block;
        let grid_z = total_batch as i64;

        let mut a_data = a.data as u64;
        let mut b_data = b.data as u64;
        let mut c_data = out_data as u64;
        let mut m_val = m;
        let mut n_val = n;
        let mut k_val = k;
        let mut batch_val = total_batch;
        let mut sa_val = stride_a;
        let mut sb_val = stride_b;
        let mut sc_val = stride_c;

        let args: [*mut std::ffi::c_void; 10] = [
            &mut a_data as *mut _ as *mut std::ffi::c_void,
            &mut b_data as *mut _ as *mut std::ffi::c_void,
            &mut c_data as *mut _ as *mut std::ffi::c_void,
            &mut m_val as *mut _ as *mut std::ffi::c_void,
            &mut n_val as *mut _ as *mut std::ffi::c_void,
            &mut k_val as *mut _ as *mut std::ffi::c_void,
            &mut batch_val as *mut _ as *mut std::ffi::c_void,
            &mut sa_val as *mut _ as *mut std::ffi::c_void,
            &mut sb_val as *mut _ as *mut std::ffi::c_void,
            &mut sc_val as *mut _ as *mut std::ffi::c_void,
        ];

        let result = inner::kernel_launch(
            fused_kernels::BMM_F32_PTX.as_ptr(),
            b"nsl_bmm_f32\0".as_ptr(),
            [grid_x, grid_y, grid_z],
            [block, block, 1],
            &args, 0,
        );
        assert_eq!(result as u32, 0, "GPU BMM kernel failed: {}", result as u32);
    }
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    out_ptr as i64
}

/// GPU scalar op (tensor op scalar).
#[cfg(feature = "cuda")]
pub(crate) fn gpu_scalar_op(a_ptr: i64, scalar: f32, ptx: &str, kernel_name: &str) -> i64 {
    use crate::tensor::NslTensor;
    inner::set_oom_context(kernel_name.trim_end_matches('\0'));
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let n = a.len as usize;
    let out_data = inner::alloc_managed(n * 4);
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);
    let out = Box::new(NslTensor::new(
        out_data,
        shape,
        strides,
        a.ndim,
        a.len,
        a.device,
        1,
        1,
        0,
    ));
    let out_ptr = Box::into_raw(out);
    let out_t = unsafe { &*out_ptr };

    let mut a_data = a.data as u64;
    let mut c_data = out_t.data as u64;
    let mut s_val = scalar;
    let mut n_val = n as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut s_val as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU kernel '{}' failed: {}", kernel_name.trim_end_matches('\0'), result as u32);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    out_ptr as i64
}

// === GPU backward op helpers ===

/// GPU backward binary op: takes grad tensor and a saved tensor, produces output of same shape as grad.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_backward_binary(a_ptr: i64, b_ptr: i64, ptx: &str, kernel_name: &str) -> i64 {
    use crate::tensor::NslTensor;
    inner::set_oom_context(kernel_name.trim_end_matches('\0'));
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };
    assert_eq!(a.len, b.len, "GPU backward: length mismatch between grad and saved tensors");

    let n = a.len as usize;
    let out_data = inner::alloc_managed(n * 4); // f32 = 4 bytes
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);
    let out = Box::new(NslTensor::new(
        out_data,
        shape,
        strides,
        a.ndim,
        a.len,
        a.device,
        1,
        1,
        0,
    ));
    let out_ptr = Box::into_raw(out);
    let out_t = unsafe { &*out_ptr };

    let mut a_data = a.data as u64;
    let mut b_data = b.data as u64;
    let mut c_data = out_t.data as u64;
    let mut n_val = n as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut b_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU backward kernel '{}' failed: {}", kernel_name.trim_end_matches('\0'), result as u32);
    #[allow(unused_unsafe)]
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    out_ptr as i64
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_relu_backward(grad: i64, input: i64) -> i64 {
    gpu_backward_binary(
        grad, input,
        kernels::RELU_BACKWARD_F32_PTX,
        "nsl_relu_backward_f32\0",
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_sigmoid_backward(grad: i64, saved_out: i64) -> i64 {
    gpu_backward_binary(
        grad, saved_out,
        kernels::SIGMOID_BACKWARD_F32_PTX,
        "nsl_sigmoid_backward_f32\0",
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_tanh_backward(grad: i64, saved_out: i64) -> i64 {
    gpu_backward_binary(
        grad, saved_out,
        kernels::TANH_BACKWARD_F32_PTX,
        "nsl_tanh_backward_f32\0",
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_gelu_backward(grad: i64, input: i64) -> i64 {
    gpu_backward_binary(
        grad, input,
        kernels::GELU_BACKWARD_F32_PTX,
        "nsl_gelu_backward_f32\0",
    )
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_silu_backward(grad: i64, input: i64) -> i64 {
    gpu_backward_binary(
        grad, input,
        kernels::SILU_BACKWARD_F32_PTX,
        "nsl_silu_backward_f32\0",
    )
}

/// GPU clamp forward: out[i] = clamp(in[i], lo, hi)
#[cfg(feature = "cuda")]
pub(crate) fn gpu_clamp_f32(a_ptr: i64, lo: f32, hi: f32) -> i64 {
    use crate::tensor::NslTensor;
    inner::set_oom_context("clamp_f32");
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let n = a.len as usize;
    let out_data = inner::alloc_managed(n * 4);
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);
    let out = Box::new(NslTensor::new(
        out_data,
        shape,
        strides,
        a.ndim,
        a.len,
        a.device,
        1,
        1,
        0,
    ));
    let out_ptr = Box::into_raw(out);
    let out_t = unsafe { &*out_ptr };

    let mut a_data = a.data as u64;
    let mut c_data = out_t.data as u64;
    let mut n_val = n as u64;
    let mut lo_val = lo;
    let mut hi_val = hi;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut lo_val as *mut _ as *mut std::ffi::c_void,
        &mut hi_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        kernels::CLAMP_F32_PTX.as_ptr(),
        "nsl_clamp_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU clamp kernel failed: {}", result as u32);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    out_ptr as i64
}

/// GPU clamp forward in-place: writes clamp result back to input buffer.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_clamp_f32_inplace(a_ptr: i64, lo: f32, hi: f32) {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let n = a.len as usize;

    let mut a_data = a.data as u64;
    let mut c_data = a.data as u64; // output = input buffer
    let mut n_val = n as u64;
    let mut lo_val = lo;
    let mut hi_val = hi;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut lo_val as *mut _ as *mut std::ffi::c_void,
        &mut hi_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        kernels::CLAMP_F32_PTX.as_ptr(),
        "nsl_clamp_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU clamp inplace kernel failed: {}", result as u32);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_clamp_backward(grad: i64, input: i64, min_val: f32, max_val: f32) -> i64 {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(grad as *const NslTensor) };
    let b = unsafe { &*(input as *const NslTensor) };
    assert_eq!(a.len, b.len, "GPU clamp_backward: length mismatch");

    let n = a.len as usize;
    let out_data = inner::alloc_managed(n * 4);
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);
    let out = Box::new(NslTensor::new(
        out_data,
        shape,
        strides,
        a.ndim,
        a.len,
        a.device,
        1,
        1,
        0,
    ));
    let out_ptr = Box::into_raw(out);
    let out_t = unsafe { &*out_ptr };

    let mut a_data = a.data as u64;
    let mut b_data = b.data as u64;
    let mut c_data = out_t.data as u64;
    let mut min_arg = min_val;
    let mut max_arg = max_val;
    let mut n_val = n as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut b_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut min_arg as *mut _ as *mut std::ffi::c_void,
        &mut max_arg as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        kernels::CLAMP_BACKWARD_F32_PTX.as_ptr(),
        "nsl_clamp_backward_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU clamp_backward kernel failed: {}", result as u32);
    #[allow(unused_unsafe)]
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    out_ptr as i64
}

// === FFI exports ===

/// GPU fused log-softmax: (x - max) - log(sum(exp(x - max))) in a single kernel.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_log_softmax_f32(tensor_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    use fused_kernels::LOG_SOFTMAX_F32_PTX;

    let t = NslTensor::from_ptr(tensor_ptr);
    let ndim = t.ndim as usize;
    let shape_slice = unsafe { std::slice::from_raw_parts(t.shape, ndim) };
    let cols = shape_slice[ndim - 1] as u64;
    let rows = (t.len as u64) / cols;
    let total = t.len as usize;
    let out_data = inner::alloc_managed(total * 4);
    let out_shape = NslTensor::copy_shape(t.shape, t.ndim);
    let out_strides = NslTensor::compute_strides(out_shape, t.ndim);

    let mut in_data = t.data as u64;
    let mut out_data_u64 = out_data as u64;
    let mut rows_val = rows;
    let mut cols_val = cols;
    let args: [*mut c_void; 4] = [
        &mut in_data as *mut _ as *mut c_void,
        &mut out_data_u64 as *mut _ as *mut c_void,
        &mut rows_val as *mut _ as *mut c_void,
        &mut cols_val as *mut _ as *mut c_void,
    ];
    let block = 256i64;
    let grid = rows as i64;
    let result = inner::kernel_launch(
        LOG_SOFTMAX_F32_PTX.as_ptr(), b"nsl_log_softmax_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 256 * 4 * 2,
    );
    assert_eq!(result as u32, 0, "GPU log_softmax kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    let out = Box::new(NslTensor::new(out_data, out_shape, out_strides, t.ndim, t.len, t.device, 1, 1, 0));
    NslTensor::publish(out)
}

/// Initialize the CUDA runtime (device 0, primary context).
/// Returns 0 on success. Aborts if CUDA feature is not compiled.
#[no_mangle]
pub extern "C" fn nsl_cuda_init() -> i64 {
    #[cfg(feature = "cuda")]
    {
        inner::init();
        0
    }
    #[cfg(not(feature = "cuda"))]
    {
        eprintln!("CUDA support not compiled. Rebuild with --features cuda");
        std::process::abort();
    }
}

/// Launch a PTX kernel. All params are i64 for Cranelift ABI compatibility.
///
/// - `ptx_ptr`: pointer to null-terminated PTX source string
/// - `name_ptr`: pointer to null-terminated kernel function name
/// - `grid_x/y/z`: grid dimensions
/// - `block_x/y/z`: block dimensions
/// - `args_ptr`: pointer to array of `*mut c_void` (pointers to argument values)
/// - `num_args`: number of arguments
/// - `shared_mem_bytes`: bytes of dynamic shared memory per block
///
/// Returns 0 (CUDA_SUCCESS) on success, non-zero CUDA error code on failure.
#[no_mangle]
pub extern "C" fn nsl_kernel_launch(
    ptx_ptr: i64,
    name_ptr: i64,
    grid_x: i64,
    grid_y: i64,
    grid_z: i64,
    block_x: i64,
    block_y: i64,
    block_z: i64,
    args_ptr: i64,
    num_args: i64,
    shared_mem_bytes: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let args_slice = unsafe {
            std::slice::from_raw_parts(args_ptr as *const *mut c_void, num_args as usize)
        };
        let result = inner::kernel_launch(
            ptx_ptr as *const u8,
            name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            args_slice, shared_mem_bytes as u32,
        );
        result as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ptx_ptr, name_ptr, grid_x, grid_y, grid_z);
        let _ = (block_x, block_y, block_z, args_ptr, num_args, shared_mem_bytes);
        eprintln!("CUDA support not compiled. Rebuild with --features cuda");
        std::process::abort();
    }
}

// ---------------------------------------------------------------------------
// GPU Embedding Lookup
// ---------------------------------------------------------------------------

/// GPU embedding lookup: weight is on GPU (f32), indices may be CPU or GPU.
/// Allocates output via alloc_managed and launches the embedding PTX kernel.
/// Returns a GPU tensor (device = weight.device).
#[cfg(feature = "cuda")]
pub(crate) fn gpu_embedding_lookup(weight_ptr: i64, indices_ptr: i64) -> i64 {
    inner::set_oom_context("embedding_lookup");
    use crate::tensor::NslTensor;

    let weight = unsafe { &*(weight_ptr as *const NslTensor) };
    let indices = unsafe { &*(indices_ptr as *const NslTensor) };

    let vocab_size = unsafe { *weight.shape.add(0) } as u64;
    let embed_dim = unsafe { *weight.shape.add(1) } as u64;
    let seq_len = unsafe { *indices.shape.add(0) } as u64;
    let _ = vocab_size; // bounds checked by CPU fallback before we get here

    let out_elems = (seq_len * embed_dim) as usize;
    let out_data = inner::alloc_managed(out_elems * 4); // f32 = 4 bytes

    // Ensure indices are on GPU (preserving dtype — may be i32 or f32)
    let indices_on_gpu = if indices.device == 0 {
        crate::tensor::nsl_tensor_to_device(indices_ptr, weight.device as i64)
    } else {
        let t = unsafe { &mut *(indices_ptr as *mut NslTensor) };
        t.refcount.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        indices_ptr
    };
    let indices_gpu = unsafe { &*(indices_on_gpu as *const NslTensor) };

    let out_shape = crate::memory::checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = seq_len as i64;
        *out_shape.add(1) = embed_dim as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 2);

    let mut w_data = weight.data as u64;
    let mut i_data = indices_gpu.data as u64;
    let mut o_data = out_data as u64;
    let mut seq_val = seq_len;
    let mut emb_val = embed_dim;

    let args: [*mut std::ffi::c_void; 5] = [
        &mut w_data as *mut _ as *mut std::ffi::c_void,
        &mut i_data as *mut _ as *mut std::ffi::c_void,
        &mut o_data as *mut _ as *mut std::ffi::c_void,
        &mut seq_val as *mut _ as *mut std::ffi::c_void,
        &mut emb_val as *mut _ as *mut std::ffi::c_void,
    ];

    // 2D grid: x=seq, y=embed — each block is 16x16 threads
    let block_x = 16i64;
    let block_y = 16i64;
    let grid_x = ((seq_len as i64) + block_x - 1) / block_x;
    let grid_y = ((embed_dim as i64) + block_y - 1) / block_y;

    // Select kernel based on indices dtype: i32 indices use ld.global.s32,
    // f32 indices use ld.global.f32 + cvt.rzi.u64.f32
    let (ptx, kernel_name): (&str, &[u8]) = if indices_gpu.dtype == 4 {
        (fused_kernels::EMBEDDING_I32IDX_PTX, b"nsl_embedding_i32idx\0")
    } else {
        (fused_kernels::EMBEDDING_F32_PTX, b"nsl_embedding_f32\0")
    };

    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid_x, grid_y, 1], [block_x, block_y, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU embedding kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    crate::tensor::nsl_tensor_free(indices_on_gpu);

    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        2,
        (seq_len * embed_dim) as i64,
        weight.device,
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

// ---------------------------------------------------------------------------
// GPU Bias Add
// ---------------------------------------------------------------------------

/// GPU bias_add: out[i,j] = tensor[i,j] + bias[j].
/// Both tensor and bias must be on GPU (f32). Allocates output via alloc_managed.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_bias_add(tensor_ptr: i64, bias_ptr: i64) -> i64 {
    inner::set_oom_context("bias_add");
    use crate::tensor::NslTensor;
    use fused_kernels::BIAS_ADD_F32_PTX;

    let tensor = unsafe { &*(tensor_ptr as *const NslTensor) };
    let bias_ref = unsafe { &*(bias_ptr as *const NslTensor) };

    let rows = unsafe { *tensor.shape.add(0) } as u64;
    let cols = unsafe { *tensor.shape.add(1) } as u64;
    let total = rows * cols;

    let out_data = inner::alloc_managed((total as usize) * 4);

    // Ensure bias is on GPU
    let bias_on_gpu = if bias_ref.device == 0 {
        crate::tensor::nsl_tensor_to_device(bias_ptr, tensor.device as i64)
    } else {
        let t = unsafe { &mut *(bias_ptr as *mut NslTensor) };
        t.refcount.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        bias_ptr
    };
    let bias_gpu = unsafe { &*(bias_on_gpu as *const NslTensor) };

    let out_shape = crate::memory::checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = rows as i64;
        *out_shape.add(1) = cols as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 2);

    let mut t_data = tensor.data as u64;
    let mut b_data = bias_gpu.data as u64;
    let mut o_data = out_data as u64;
    let mut total_val = total;
    let mut cols_val = cols;

    let args: [*mut std::ffi::c_void; 5] = [
        &mut t_data as *mut _ as *mut std::ffi::c_void,
        &mut b_data as *mut _ as *mut std::ffi::c_void,
        &mut o_data as *mut _ as *mut std::ffi::c_void,
        &mut total_val as *mut _ as *mut std::ffi::c_void,
        &mut cols_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = ((total as i64) + block - 1) / block;

    let result = inner::kernel_launch(
        BIAS_ADD_F32_PTX.as_ptr(), b"nsl_bias_add_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU bias_add kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    crate::tensor::nsl_tensor_free(bias_on_gpu);

    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        2,
        total as i64,
        tensor.device,
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

/// GPU softmax along the last dimension. One thread block per row.
/// Input must be on GPU (f32). Output allocated via alloc_managed.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_softmax_f32(tensor_ptr: i64) -> i64 {
    inner::set_oom_context("softmax_f32");
    use crate::tensor::NslTensor;
    use fused_kernels::SOFTMAX_F32_PTX;

    let t = NslTensor::from_ptr(tensor_ptr);
    let ndim = t.ndim as usize;
    let shape_slice = unsafe { std::slice::from_raw_parts(t.shape, ndim) };

    // Softmax along last dimension
    let cols = shape_slice[ndim - 1] as u64;
    let rows = (t.len as u64) / cols;

    let total = t.len as usize;
    let out_data = inner::alloc_managed(total * 4); // f32
    let out_shape = NslTensor::copy_shape(t.shape, t.ndim);
    let out_strides = NslTensor::compute_strides(out_shape, t.ndim);

    let mut in_data = t.data as u64;
    let mut out_data_u64 = out_data as u64;
    let mut rows_val = rows;
    let mut cols_val = cols;

    let args: [*mut std::ffi::c_void; 4] = [
        &mut in_data as *mut _ as *mut std::ffi::c_void,
        &mut out_data_u64 as *mut _ as *mut std::ffi::c_void,
        &mut rows_val as *mut _ as *mut std::ffi::c_void,
        &mut cols_val as *mut _ as *mut std::ffi::c_void,
    ];

    // One block per row, 256 threads per block
    let block = 256i64;
    let grid = rows as i64;

    let result = inner::kernel_launch(
        SOFTMAX_F32_PTX.as_ptr(), b"nsl_softmax_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 256 * 4 * 2, // shared mem: smax[256] + ssum[256]
    );
    assert_eq!(result as u32, 0, "GPU softmax kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        t.ndim,
        t.len,
        t.device,
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

/// GPU per-dimension sum reduction. Input must be on GPU (f32), contiguous.
/// Returns a new GPU tensor with the reduced dimension removed (or kept as 1 if keepdim).
#[cfg(feature = "cuda")]
pub(crate) fn gpu_sum_dim_f32(tensor_ptr: i64, dim: usize, keepdim: bool) -> i64 {
    inner::set_oom_context("sum_dim_f32");
    use crate::tensor::NslTensor;
    use fused_kernels::SUM_DIM_F32_PTX;

    let t = NslTensor::from_ptr(tensor_ptr);
    let ndim = t.ndim as usize;
    let shape_slice = unsafe { std::slice::from_raw_parts(t.shape, ndim) };

    let reduce_size = shape_slice[dim] as u64;

    // Compute outer = product of dims before dim
    let outer: u64 = shape_slice[..dim].iter().map(|&s| s as u64).product::<u64>().max(1);
    // Compute inner = product of dims after dim
    let inner: u64 = shape_slice[dim + 1..].iter().map(|&s| s as u64).product::<u64>().max(1);

    let out_total = (outer * inner) as usize;
    let out_data = inner::alloc_managed(out_total * 4); // f32

    // Build output shape
    let out_shape_vec: Vec<i64> = if keepdim {
        shape_slice.iter().enumerate()
            .map(|(i, &s)| if i == dim { 1 } else { s })
            .collect()
    } else {
        shape_slice.iter().enumerate()
            .filter(|&(i, _)| i != dim)
            .map(|(_, &s)| s)
            .collect()
    };

    let out_ndim = out_shape_vec.len() as i64;
    let out_shape = crate::memory::checked_alloc(out_shape_vec.len() * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &v) in out_shape_vec.iter().enumerate() {
        unsafe { *out_shape.add(i) = v };
    }
    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);

    let mut in_data = t.data as u64;
    let mut out_data_u64 = out_data as u64;
    let mut outer_val = outer;
    let mut reduce_val = reduce_size;
    let mut inner_val = inner;

    let args: [*mut std::ffi::c_void; 5] = [
        &mut in_data as *mut _ as *mut std::ffi::c_void,
        &mut out_data_u64 as *mut _ as *mut std::ffi::c_void,
        &mut outer_val as *mut _ as *mut std::ffi::c_void,
        &mut reduce_val as *mut _ as *mut std::ffi::c_void,
        &mut inner_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = out_total as i64;

    let result = inner::kernel_launch(
        SUM_DIM_F32_PTX.as_ptr(), b"nsl_sum_dim_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 256 * 4,
    );
    assert_eq!(result as u32, 0, "GPU sum_dim kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        out_ndim,
        out_total as i64,
        t.device,
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

/// GPU global sum reduction (all elements to a single scalar). Input must be on GPU (f32), contiguous.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_global_sum_f32(tensor_ptr: i64) -> i64 {
    inner::set_oom_context("global_sum_f32");
    use crate::tensor::NslTensor;
    use fused_kernels::GLOBAL_SUM_F32_PTX;

    let t = NslTensor::from_ptr(tensor_ptr);
    let n = t.len as u64;

    let out_data = inner::alloc_managed(4); // single f32

    let out_shape = crate::memory::checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *out_shape = 1 };
    let out_strides = NslTensor::compute_strides(out_shape, 1);

    let mut in_data = t.data as u64;
    let mut out_data_u64 = out_data as u64;
    let mut n_val = n;

    let args: [*mut std::ffi::c_void; 3] = [
        &mut in_data as *mut _ as *mut std::ffi::c_void,
        &mut out_data_u64 as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = 1i64;

    let result = inner::kernel_launch(
        GLOBAL_SUM_F32_PTX.as_ptr(), b"nsl_global_sum_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 256 * 4,
    );
    assert_eq!(result as u32, 0, "GPU global_sum kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        1,
        1,
        t.device,
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

// ---------------------------------------------------------------------------
// M45b: GPU tensor statistics — single-block reduction for min/max/sum/sum_sq
// ---------------------------------------------------------------------------

/// Compute min, max, sum, sum_of_squares for a GPU f32 tensor in a single kernel
/// launch. Returns `[min, max, mean, std]` as f32 values on the CPU.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_tensor_stats_f32(tensor_ptr: i64) -> [f32; 4] {
    use crate::tensor::NslTensor;
    use fused_kernels::TENSOR_STATS_F32_PTX;

    let t = NslTensor::from_ptr(tensor_ptr);
    let n = t.len as u64;
    if n == 0 {
        return [0.0; 4];
    }

    // Allocate 4 f32s (16 bytes) on device for kernel output: [min, max, sum, sum_sq]
    let out_data = inner::alloc_managed(16);

    let mut in_data = t.data as u64;
    let mut out_data_u64 = out_data as u64;
    let mut n_val = n;

    let args: [*mut std::ffi::c_void; 3] = [
        &mut in_data as *mut _ as *mut std::ffi::c_void,
        &mut out_data_u64 as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];

    let result = inner::kernel_launch(
        TENSOR_STATS_F32_PTX.as_ptr(), b"nsl_tensor_stats_f32\0".as_ptr(),
        [1, 1, 1], [256, 1, 1], &args, 256 * 4 * 4, // 4 shared arrays of 256 f32
    );
    assert_eq!(result as u32, 0, "GPU tensor_stats kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    // Read back 4 f32s from device
    let mut raw = [0f32; 4];
    inner::memcpy_dtoh(
        raw.as_mut_ptr() as *mut std::ffi::c_void,
        out_data as *const std::ffi::c_void,
        16,
    );
    inner::free_managed(out_data);

    let min_val = raw[0];
    let max_val = raw[1];
    let sum_val = raw[2];
    let sum_sq = raw[3];
    let mean_val = sum_val / n as f32;
    // Population std: sqrt(E[x^2] - (E[x])^2)
    let variance = (sum_sq / n as f32) - (mean_val * mean_val);
    let std_val = if variance > 0.0 { variance.sqrt() } else { 0.0 };

    [min_val, max_val, mean_val, std_val]
}

// ---------------------------------------------------------------------------
// M42b: KV-cache GPU dequantization
// ---------------------------------------------------------------------------

/// Dequantize INT8 per-head data on GPU: output[i] = input_i8[i] * scales[head].
/// `data_ptr`: device pointer to i8 quantized data
/// `out_ptr`: device pointer to f32 output (must be pre-allocated, n elements)
/// `scales_ptr`: device pointer to f32 scales array (num_heads entries)
/// `n`: total number of elements
/// `head_stride`: block_size * head_dim (elements per head)
#[cfg(feature = "cuda")]
pub(crate) fn gpu_dequant_int8_per_head_f32(
    data_ptr: *const std::ffi::c_void,
    out_ptr: *mut std::ffi::c_void,
    scales_ptr: *const std::ffi::c_void,
    n: u64,
    head_stride: u64,
) {
    use fused_kernels::DEQUANT_INT8_PER_HEAD_F32_PTX;

    let mut inp = data_ptr as u64;
    let mut out = out_ptr as u64;
    let mut sc = scales_ptr as u64;
    let mut n_val = n;
    let mut hs = head_stride;

    let args: [*mut std::ffi::c_void; 5] = [
        &mut inp as *mut _ as *mut std::ffi::c_void,
        &mut out as *mut _ as *mut std::ffi::c_void,
        &mut sc as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut hs as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = ((n + 255) / 256) as i64;

    let result = inner::kernel_launch(
        DEQUANT_INT8_PER_HEAD_F32_PTX.as_ptr(), b"nsl_dequant_int8_per_head_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU dequant_int8_per_head kernel failed: {:?}", result);
}

/// Dequantize INT8 per-token data on GPU.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_dequant_int8_per_token_f32(
    data_ptr: *const std::ffi::c_void,
    out_ptr: *mut std::ffi::c_void,
    scales_ptr: *const std::ffi::c_void,
    n: u64,
    head_stride: u64,
    head_dim: u64,
) {
    use fused_kernels::DEQUANT_INT8_PER_TOKEN_F32_PTX;

    let mut inp = data_ptr as u64;
    let mut out = out_ptr as u64;
    let mut sc = scales_ptr as u64;
    let mut n_val = n;
    let mut hs = head_stride;
    let mut hd = head_dim;

    let args: [*mut std::ffi::c_void; 6] = [
        &mut inp as *mut _ as *mut std::ffi::c_void,
        &mut out as *mut _ as *mut std::ffi::c_void,
        &mut sc as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut hs as *mut _ as *mut std::ffi::c_void,
        &mut hd as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = ((n + 255) / 256) as i64;

    let result = inner::kernel_launch(
        DEQUANT_INT8_PER_TOKEN_F32_PTX.as_ptr(), b"nsl_dequant_int8_per_token_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU dequant_int8_per_token kernel failed: {:?}", result);
}

/// Dequantize INT4 per-group data on GPU.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_dequant_int4_per_group_f32(
    data_ptr: *const std::ffi::c_void,
    out_ptr: *mut std::ffi::c_void,
    scales_ptr: *const std::ffi::c_void,
    zero_points_ptr: *const std::ffi::c_void,
    n: u64,
    group_size: u64,
) {
    use fused_kernels::DEQUANT_INT4_PER_GROUP_F32_PTX;

    let mut inp = data_ptr as u64;
    let mut out = out_ptr as u64;
    let mut sc = scales_ptr as u64;
    let mut zp = zero_points_ptr as u64;
    let mut n_val = n;
    let mut gs = group_size;

    let args: [*mut std::ffi::c_void; 6] = [
        &mut inp as *mut _ as *mut std::ffi::c_void,
        &mut out as *mut _ as *mut std::ffi::c_void,
        &mut sc as *mut _ as *mut std::ffi::c_void,
        &mut zp as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut gs as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = ((n + 255) / 256) as i64;

    let result = inner::kernel_launch(
        DEQUANT_INT4_PER_GROUP_F32_PTX.as_ptr(), b"nsl_dequant_int4_per_group_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU dequant_int4_per_group kernel failed: {:?}", result);
}

/// Dequantize FP8 E4M3 data on GPU: bit-manipulation u8 → f32.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_dequant_fp8_e4m3_f32(
    data_ptr: *const std::ffi::c_void,
    out_ptr: *mut std::ffi::c_void,
    n: u64,
) {
    use fused_kernels::DEQUANT_FP8_E4M3_F32_PTX;

    let mut inp = data_ptr as u64;
    let mut out = out_ptr as u64;
    let mut n_val = n;

    let args: [*mut std::ffi::c_void; 3] = [
        &mut inp as *mut _ as *mut std::ffi::c_void,
        &mut out as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = ((n + 255) / 256) as i64;

    let result = inner::kernel_launch(
        DEQUANT_FP8_E4M3_F32_PTX.as_ptr(), b"nsl_dequant_fp8_e4m3_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU dequant_fp8_e4m3 kernel failed: {:?}", result);
}

// ---------------------------------------------------------------------------
// M46b: Deterministic GPU global sum — single-thread sequential accumulation
// ---------------------------------------------------------------------------

/// GPU deterministic global sum reduction. Uses a single-thread kernel that
/// accumulates ALL elements in ascending order (no parallelism = bit-identical).
/// This is slower than the multi-thread tree reduction but guarantees determinism.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_det_global_sum_f32(tensor_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    use fused_kernels::DET_GLOBAL_SUM_F32_PTX;

    let t = NslTensor::from_ptr(tensor_ptr);
    let n = t.len as u64;

    let out_data = inner::alloc_managed(4); // single f32

    let out_shape = crate::memory::checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *out_shape = 1 };
    let out_strides = NslTensor::compute_strides(out_shape, 1);

    let mut in_data = t.data as u64;
    let mut out_data_u64 = out_data as u64;
    let mut n_val = n;

    let args: [*mut std::ffi::c_void; 3] = [
        &mut in_data as *mut _ as *mut std::ffi::c_void,
        &mut out_data_u64 as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];

    // Single block, single thread — guarantees sequential deterministic accumulation
    let result = inner::kernel_launch(
        DET_GLOBAL_SUM_F32_PTX.as_ptr(), b"nsl_det_global_sum_f32\0".as_ptr(),
        [1, 1, 1], [1, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU det_global_sum kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        1,
        1,
        t.device,
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

// ---------------------------------------------------------------------------
// M46b: Deterministic GPU per-dim sum — one thread per output, sequential
// ---------------------------------------------------------------------------

/// GPU deterministic per-dimension sum reduction. Uses one thread per output element,
/// each sequentially accumulating its reduce_size inputs in ascending order.
/// Slower than shared-memory tree reduction but guarantees bit-identical results.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_det_sum_dim_f32(tensor_ptr: i64, dim: usize, keepdim: bool) -> i64 {
    use crate::tensor::NslTensor;
    use fused_kernels::DET_SUM_DIM_F32_PTX;

    let t = NslTensor::from_ptr(tensor_ptr);
    let ndim = t.ndim as usize;
    let shape_slice = unsafe { std::slice::from_raw_parts(t.shape, ndim) };

    let reduce_size = shape_slice[dim] as u64;
    let outer: u64 = shape_slice[..dim].iter().map(|&s| s as u64).product::<u64>().max(1);
    let inner: u64 = shape_slice[dim + 1..].iter().map(|&s| s as u64).product::<u64>().max(1);

    let out_total = (outer * inner) as usize;
    let out_data = inner::alloc_managed(out_total * 4); // f32

    // Build output shape
    let out_shape_vec: Vec<i64> = if keepdim {
        shape_slice.iter().enumerate()
            .map(|(i, &s)| if i == dim { 1 } else { s })
            .collect()
    } else {
        shape_slice.iter().enumerate()
            .filter(|&(i, _)| i != dim)
            .map(|(_, &s)| s)
            .collect()
    };

    let out_ndim = out_shape_vec.len() as i64;
    let out_shape = crate::memory::checked_alloc(out_shape_vec.len() * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &v) in out_shape_vec.iter().enumerate() {
        unsafe { *out_shape.add(i) = v };
    }
    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);

    let mut in_data = t.data as u64;
    let mut out_data_u64 = out_data as u64;
    let mut outer_val = outer;
    let mut reduce_val = reduce_size;
    let mut inner_val = inner;

    let args: [*mut std::ffi::c_void; 5] = [
        &mut in_data as *mut _ as *mut std::ffi::c_void,
        &mut out_data_u64 as *mut _ as *mut std::ffi::c_void,
        &mut outer_val as *mut _ as *mut std::ffi::c_void,
        &mut reduce_val as *mut _ as *mut std::ffi::c_void,
        &mut inner_val as *mut _ as *mut std::ffi::c_void,
    ];

    // One block per output element, single thread per block — sequential accumulation
    let grid = out_total as i64;
    let result = inner::kernel_launch(
        DET_SUM_DIM_F32_PTX.as_ptr(), b"nsl_det_sum_dim_f32\0".as_ptr(),
        [grid, 1, 1], [1, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU det_sum_dim kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        out_ndim,
        out_total as i64,
        t.device,
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

// ---------------------------------------------------------------------------
// M46c: Deterministic GPU scatter_add — output-centric, no atomics
// ---------------------------------------------------------------------------

/// GPU deterministic scatter_add: out[row, col] = input[row, col] + sum(src[i, col] for all i where indices[i] == row).
/// Each output element is computed by exactly one thread that scans all input indices sequentially.
/// Guarantees bit-identical results regardless of GPU scheduling (no atomics).
/// All tensors must be on GPU (f32).
#[cfg(feature = "cuda")]
pub(crate) fn gpu_det_scatter_add_f32(
    input_ptr: i64,
    indices_ptr: i64,
    src_ptr: i64,
) -> i64 {
    use crate::tensor::NslTensor;
    use fused_kernels::DET_SCATTER_ADD_F32_PTX;

    let input = NslTensor::from_ptr(input_ptr);
    let src = unsafe { &*(src_ptr as *const NslTensor) };
    let indices = unsafe { &*(indices_ptr as *const NslTensor) };

    let input_ndim = input.ndim as usize;
    let input_shape = unsafe { std::slice::from_raw_parts(input.shape, input_ndim) };

    // input is [vocab_size, embed_dim] (or [vocab_size] for 1D)
    let vocab_size = input_shape[0] as u64;
    let embed_dim = if input_ndim >= 2 { input_shape[1] as u64 } else { 1u64 };

    let num_indices = indices.len as u64;

    // Allocate output: same shape as input
    let out_elems = (vocab_size * embed_dim) as usize;
    let out_data = inner::alloc_managed(out_elems * 4); // f32

    // Build output shape (same as input)
    let out_ndim = input_ndim;
    let out_shape = crate::memory::checked_alloc(out_ndim * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..out_ndim {
        unsafe { *out_shape.add(i) = input_shape[i] };
    }
    let out_strides = NslTensor::compute_strides(out_shape, out_ndim as i64);

    // Ensure indices are on GPU
    let indices_on_gpu = if indices.device == 0 {
        crate::tensor::nsl_tensor_to_device(indices_ptr, input.device as i64)
    } else {
        let t = unsafe { &mut *(indices_ptr as *mut NslTensor) };
        t.refcount.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        indices_ptr
    };
    let indices_gpu = unsafe { &*(indices_on_gpu as *const NslTensor) };

    let mut s_data = src.data as u64;
    let mut i_data = indices_gpu.data as u64;
    let mut in_data = input.data as u64;
    let mut o_data = out_data as u64;
    let mut n_indices = num_indices;
    let mut emb_dim = embed_dim;
    let mut vocab = vocab_size;

    let args: [*mut std::ffi::c_void; 7] = [
        &mut s_data as *mut _ as *mut std::ffi::c_void,
        &mut i_data as *mut _ as *mut std::ffi::c_void,
        &mut in_data as *mut _ as *mut std::ffi::c_void,
        &mut o_data as *mut _ as *mut std::ffi::c_void,
        &mut n_indices as *mut _ as *mut std::ffi::c_void,
        &mut emb_dim as *mut _ as *mut std::ffi::c_void,
        &mut vocab as *mut _ as *mut std::ffi::c_void,
    ];

    let block_x = 16i64;
    let block_y = 16i64;
    let grid_x = ((vocab_size as i64) + block_x - 1) / block_x;
    let grid_y = ((embed_dim as i64) + block_y - 1) / block_y;

    let result = inner::kernel_launch(
        DET_SCATTER_ADD_F32_PTX.as_ptr(), b"nsl_det_scatter_add_f32\0".as_ptr(),
        [grid_x, grid_y, 1], [block_x, block_y, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU det_scatter_add kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    crate::tensor::nsl_tensor_free(indices_on_gpu);

    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        out_ndim as i64,
        out_elems as i64,
        input.device,
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

/// GPU per-dimension max reduction. Input must be on GPU (f32), contiguous.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_max_dim_f32(tensor_ptr: i64, dim: usize, keepdim: bool) -> i64 {
    inner::set_oom_context("max_dim_f32");
    use crate::tensor::NslTensor;
    use fused_kernels::MAX_DIM_F32_PTX;

    let t = NslTensor::from_ptr(tensor_ptr);
    let ndim = t.ndim as usize;
    let shape_slice = unsafe { std::slice::from_raw_parts(t.shape, ndim) };

    let reduce_size = shape_slice[dim] as u64;
    let outer: u64 = shape_slice[..dim].iter().map(|&s| s as u64).product::<u64>().max(1);
    let inner: u64 = shape_slice[dim + 1..].iter().map(|&s| s as u64).product::<u64>().max(1);

    let out_total = (outer * inner) as usize;
    let out_data = inner::alloc_managed(out_total * 4);

    let out_shape_vec: Vec<i64> = if keepdim {
        shape_slice.iter().enumerate()
            .map(|(i, &s)| if i == dim { 1 } else { s })
            .collect()
    } else {
        shape_slice.iter().enumerate()
            .filter(|&(i, _)| i != dim)
            .map(|(_, &s)| s)
            .collect()
    };

    let out_ndim = out_shape_vec.len() as i64;
    let out_shape = crate::memory::checked_alloc(out_shape_vec.len() * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &v) in out_shape_vec.iter().enumerate() {
        unsafe { *out_shape.add(i) = v };
    }
    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);

    let mut in_data = t.data as u64;
    let mut out_data_u64 = out_data as u64;
    let mut outer_val = outer;
    let mut reduce_val = reduce_size;
    let mut inner_val = inner;

    let args: [*mut std::ffi::c_void; 5] = [
        &mut in_data as *mut _ as *mut std::ffi::c_void,
        &mut out_data_u64 as *mut _ as *mut std::ffi::c_void,
        &mut outer_val as *mut _ as *mut std::ffi::c_void,
        &mut reduce_val as *mut _ as *mut std::ffi::c_void,
        &mut inner_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = out_total as i64;

    let result = inner::kernel_launch(
        MAX_DIM_F32_PTX.as_ptr(), b"nsl_max_dim_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 256 * 4,
    );
    assert_eq!(result as u32, 0, "GPU max_dim kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        out_ndim,
        out_total as i64,
        t.device,
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

/// GPU LayerNorm: fused mean + variance + normalize + scale + shift.
/// Input, gamma, beta must all be on GPU (f32), contiguous.
/// Normalizes along the last dimension.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_layernorm_f32(input_ptr: i64, gamma_ptr: i64, beta_ptr: i64, eps: f32) -> i64 {
    inner::set_oom_context("layernorm_f32");
    use crate::tensor::NslTensor;
    use fused_kernels::LAYERNORM_F32_PTX;

    let t = NslTensor::from_ptr(input_ptr);
    let ndim = t.ndim as usize;
    let shape_slice = unsafe { std::slice::from_raw_parts(t.shape, ndim) };

    let cols = shape_slice[ndim - 1] as u64;
    let rows = (t.len as u64) / cols;

    let total = t.len as usize;
    let out_data = inner::alloc_managed(total * 4);
    let out_shape = NslTensor::copy_shape(t.shape, t.ndim);
    let out_strides = NslTensor::compute_strides(out_shape, t.ndim);

    let g = NslTensor::from_ptr(gamma_ptr);
    let b = NslTensor::from_ptr(beta_ptr);

    let mut in_data = t.data as u64;
    let mut out_data_u64 = out_data as u64;
    let mut g_data = g.data as u64;
    let mut b_data = b.data as u64;
    let mut rows_val = rows;
    let mut cols_val = cols;
    let mut eps_val = eps;

    let args: [*mut std::ffi::c_void; 7] = [
        &mut in_data as *mut _ as *mut std::ffi::c_void,
        &mut out_data_u64 as *mut _ as *mut std::ffi::c_void,
        &mut g_data as *mut _ as *mut std::ffi::c_void,
        &mut b_data as *mut _ as *mut std::ffi::c_void,
        &mut rows_val as *mut _ as *mut std::ffi::c_void,
        &mut cols_val as *mut _ as *mut std::ffi::c_void,
        &mut eps_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = rows as i64;

    let result = inner::kernel_launch(
        LAYERNORM_F32_PTX.as_ptr(), b"nsl_layernorm_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 256 * 4,
    );
    assert_eq!(result as u32, 0, "GPU layernorm kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        t.ndim,
        t.len,
        t.device,
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

/// GPU RMSNorm: fused rms + normalize + scale.
/// Input, gamma must be on GPU (f32), contiguous.
/// Normalizes along the last dimension.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_rmsnorm_f32(input_ptr: i64, gamma_ptr: i64, eps: f32) -> i64 {
    inner::set_oom_context("rmsnorm_f32");
    use crate::tensor::NslTensor;
    use fused_kernels::RMSNORM_F32_PTX;

    let t = NslTensor::from_ptr(input_ptr);
    let ndim = t.ndim as usize;
    let shape_slice = unsafe { std::slice::from_raw_parts(t.shape, ndim) };

    let cols = shape_slice[ndim - 1] as u64;
    let rows = (t.len as u64) / cols;

    let total = t.len as usize;
    let out_data = inner::alloc_managed(total * 4);
    let out_shape = NslTensor::copy_shape(t.shape, t.ndim);
    let out_strides = NslTensor::compute_strides(out_shape, t.ndim);

    let g = NslTensor::from_ptr(gamma_ptr);

    let mut in_data = t.data as u64;
    let mut out_data_u64 = out_data as u64;
    let mut g_data = g.data as u64;
    let mut rows_val = rows;
    let mut cols_val = cols;
    let mut eps_val = eps;

    let args: [*mut std::ffi::c_void; 6] = [
        &mut in_data as *mut _ as *mut std::ffi::c_void,
        &mut out_data_u64 as *mut _ as *mut std::ffi::c_void,
        &mut g_data as *mut _ as *mut std::ffi::c_void,
        &mut rows_val as *mut _ as *mut std::ffi::c_void,
        &mut cols_val as *mut _ as *mut std::ffi::c_void,
        &mut eps_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = rows as i64;

    let result = inner::kernel_launch(
        RMSNORM_F32_PTX.as_ptr(), b"nsl_rmsnorm_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 256 * 4,
    );
    assert_eq!(result as u32, 0, "GPU rmsnorm kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        t.ndim,
        t.len,
        t.device,
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

// ---------------------------------------------------------------------------
// GPU Scatter-Add (embedding backward / index-based gradient accumulation)
// ---------------------------------------------------------------------------

/// GPU scatter_add: out[indices[i], j] += src[i, j] for all (i, j).
/// Uses atomicAdd for thread safety (multiple indices may alias the same row).
/// `out` must be pre-zeroed. Both src and indices must be on GPU.
/// Returns a new GPU tensor of shape [vocab_size, embed_dim].
#[cfg(feature = "cuda")]
pub(crate) fn gpu_scatter_add_f32(
    src_ptr: i64,
    indices_ptr: i64,
    vocab_size: u64,
) -> i64 {
    use crate::tensor::NslTensor;
    use fused_kernels::SCATTER_ADD_F32_PTX;

    let src = unsafe { &*(src_ptr as *const NslTensor) };
    let indices = unsafe { &*(indices_ptr as *const NslTensor) };

    let num_indices = unsafe { *indices.shape.add(0) } as u64;
    let embed_dim = unsafe { *src.shape.add(src.ndim as usize - 1) } as u64;

    // Allocate output: [vocab_size, embed_dim], zeroed
    let out_elems = (vocab_size * embed_dim) as usize;
    let out_data = inner::alloc_managed(out_elems * 4); // f32
    // Zero the output (scatter_add accumulates into it)
    unsafe {
        std::ptr::write_bytes(out_data as *mut u8, 0, out_elems * 4);
    }

    // Ensure indices are on GPU
    let indices_on_gpu = if indices.device == 0 {
        crate::tensor::nsl_tensor_to_device(indices_ptr, src.device as i64)
    } else {
        let t = unsafe { &mut *(indices_ptr as *mut NslTensor) };
        t.refcount.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        indices_ptr
    };
    let indices_gpu = unsafe { &*(indices_on_gpu as *const NslTensor) };

    let out_shape = crate::memory::checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = vocab_size as i64;
        *out_shape.add(1) = embed_dim as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 2);

    let mut s_data = src.data as u64;
    let mut i_data = indices_gpu.data as u64;
    let mut o_data = out_data as u64;
    let mut n_indices = num_indices;
    let mut emb_dim = embed_dim;
    let mut vocab = vocab_size;

    let args: [*mut std::ffi::c_void; 6] = [
        &mut s_data as *mut _ as *mut std::ffi::c_void,
        &mut i_data as *mut _ as *mut std::ffi::c_void,
        &mut o_data as *mut _ as *mut std::ffi::c_void,
        &mut n_indices as *mut _ as *mut std::ffi::c_void,
        &mut emb_dim as *mut _ as *mut std::ffi::c_void,
        &mut vocab as *mut _ as *mut std::ffi::c_void,
    ];

    let block_x = 16i64;
    let block_y = 16i64;
    let grid_x = ((num_indices as i64) + block_x - 1) / block_x;
    let grid_y = ((embed_dim as i64) + block_y - 1) / block_y;

    let result = inner::kernel_launch(
        SCATTER_ADD_F32_PTX.as_ptr(), b"nsl_scatter_add_f32\0".as_ptr(),
        [grid_x, grid_y, 1], [block_x, block_y, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU scatter_add kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    crate::tensor::nsl_tensor_free(indices_on_gpu);

    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        2,
        (vocab_size * embed_dim) as i64,
        src.device,
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

// ---------------------------------------------------------------------------
// GPU Gather (general dim-0 gather)
// ---------------------------------------------------------------------------

/// GPU gather along dim 0: out[i, :] = input[indices[i], :].
/// Works on any 2D+ tensor — flattens trailing dims into `inner_dim`.
/// Both input and indices must be on GPU (f32).
#[cfg(feature = "cuda")]
pub(crate) fn gpu_gather_f32(input_ptr: i64, indices_ptr: i64) -> i64 {
    inner::set_oom_context("gather_f32");
    use crate::tensor::NslTensor;
    use fused_kernels::GATHER_F32_PTX;

    let input = unsafe { &*(input_ptr as *const NslTensor) };
    let indices = unsafe { &*(indices_ptr as *const NslTensor) };

    let input_rows = unsafe { *input.shape.add(0) } as u64;
    let inner_dim: u64 = if input.ndim >= 2 {
        (1..input.ndim as usize).map(|d| unsafe { *input.shape.add(d) } as u64).product()
    } else {
        1
    };
    let num_indices = indices.len as u64;

    // Allocate output: [num_indices, inner_dim]
    let out_elems = (num_indices * inner_dim) as usize;
    let out_data = inner::alloc_managed(out_elems * 4); // f32

    // Ensure indices on GPU
    let indices_on_gpu = if indices.device == 0 {
        crate::tensor::nsl_tensor_to_device(indices_ptr, input.device as i64)
    } else {
        let t = unsafe { &mut *(indices_ptr as *mut NslTensor) };
        t.refcount.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        indices_ptr
    };
    let indices_gpu = unsafe { &*(indices_on_gpu as *const NslTensor) };

    // Build output shape: [num_indices, dim1, dim2, ...]
    let out_ndim = input.ndim;
    let out_shape = crate::memory::checked_alloc(out_ndim as usize * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = num_indices as i64;
        for d in 1..out_ndim as usize {
            *out_shape.add(d) = *input.shape.add(d);
        }
    }
    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);

    let mut i_data = input.data as u64;
    let mut idx_data = indices_gpu.data as u64;
    let mut o_data = out_data as u64;
    let mut n_idx = num_indices;
    let mut inner = inner_dim;
    let mut rows = input_rows;

    let args: [*mut std::ffi::c_void; 6] = [
        &mut i_data as *mut _ as *mut std::ffi::c_void,
        &mut idx_data as *mut _ as *mut std::ffi::c_void,
        &mut o_data as *mut _ as *mut std::ffi::c_void,
        &mut n_idx as *mut _ as *mut std::ffi::c_void,
        &mut inner as *mut _ as *mut std::ffi::c_void,
        &mut rows as *mut _ as *mut std::ffi::c_void,
    ];

    let block_x = 16i64;
    let block_y = 16i64;
    let grid_x = ((num_indices as i64) + block_x - 1) / block_x;
    let grid_y = ((inner_dim as i64) + block_y - 1) / block_y;

    // Select kernel based on indices dtype: i32 uses ld.global.s32
    let (ptx, kernel_name): (&str, &[u8]) = if indices_gpu.dtype == 4 {
        (fused_kernels::GATHER_I32IDX_PTX, b"nsl_gather_i32idx\0")
    } else {
        (GATHER_F32_PTX, b"nsl_gather_f32\0")
    };

    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid_x, grid_y, 1], [block_x, block_y, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU gather kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    crate::tensor::nsl_tensor_free(indices_on_gpu);

    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        out_ndim,
        (num_indices * inner_dim) as i64,
        input.device,
        1, // f32
        1,
        0,
    ));
    NslTensor::publish(out)
}

// ---------------------------------------------------------------------------
// GPU Conv2d (direct convolution, NCHW layout)
// ---------------------------------------------------------------------------

/// GPU conv2d: out[n,co,oh,ow] = sum(input[n,ci,ih,iw] * weight[co,ci,ky,kx]) + bias[co]
/// Input must be 4D NCHW [N, C_in, H, W], weight [C_out, C_in, kH, kW].
#[cfg(feature = "cuda")]
pub(crate) fn gpu_conv2d_f32(
    input_ptr: i64, weight_ptr: i64, bias_ptr: i64,
    stride_h: u64, stride_w: u64, pad_h: u64, pad_w: u64,
) -> i64 {
    inner::set_oom_context("conv2d_f32");
    use crate::tensor::NslTensor;
    use fused_kernels::CONV2D_F32_PTX;

    let input = unsafe { &*(input_ptr as *const NslTensor) };
    let weight = unsafe { &*(weight_ptr as *const NslTensor) };

    let n = unsafe { *input.shape.add(0) } as u64;
    let c_in = unsafe { *input.shape.add(1) } as u64;
    let h = unsafe { *input.shape.add(2) } as u64;
    let w = unsafe { *input.shape.add(3) } as u64;
    let c_out = unsafe { *weight.shape.add(0) } as u64;
    let kh = unsafe { *weight.shape.add(2) } as u64;
    let kw = unsafe { *weight.shape.add(3) } as u64;

    let h_out = (h + 2 * pad_h - kh) / stride_h + 1;
    let w_out = (w + 2 * pad_w - kw) / stride_w + 1;
    let total = n * c_out * h_out * w_out;

    let out_data = inner::alloc_managed(total as usize * 4); // f32

    let out_shape = crate::memory::checked_alloc(4 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = n as i64;
        *out_shape.add(1) = c_out as i64;
        *out_shape.add(2) = h_out as i64;
        *out_shape.add(3) = w_out as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 4);

    // Ensure weight on GPU
    let weight_on_gpu = if weight.device == 0 {
        crate::tensor::nsl_tensor_to_device(weight_ptr, input.device as i64)
    } else {
        let t = unsafe { &mut *(weight_ptr as *mut NslTensor) };
        t.refcount.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        weight_ptr
    };
    let weight_gpu = unsafe { &*(weight_on_gpu as *const NslTensor) };

    // Bias pointer (0 if no bias)
    let bias_data: u64 = if bias_ptr != 0 {
        let bias = unsafe { &*(bias_ptr as *const NslTensor) };
        if bias.device == 0 {
            let bp = crate::tensor::nsl_tensor_to_device(bias_ptr, input.device as i64);
            let b = unsafe { &*(bp as *const NslTensor) };
            let d = b.data as u64;
            // We leak the bias transfer — acceptable for now
            d
        } else {
            bias.data as u64
        }
    } else {
        0u64
    };

    let mut inp_data = input.data as u64;
    let mut wt_data = weight_gpu.data as u64;
    let mut bias_val = bias_data;
    let mut out_val = out_data as u64;
    let mut n_val = n; let mut cin_val = c_in; let mut h_val = h; let mut w_val = w;
    let mut cout_val = c_out; let mut kh_val = kh; let mut kw_val = kw;
    let mut sh_val = stride_h; let mut sw_val = stride_w;
    let mut ph_val = pad_h; let mut pw_val = pad_w;
    let mut hout_val = h_out; let mut wout_val = w_out; let mut total_val = total;

    let args: [*mut std::ffi::c_void; 18] = [
        &mut inp_data as *mut _ as *mut std::ffi::c_void,
        &mut wt_data as *mut _ as *mut std::ffi::c_void,
        &mut bias_val as *mut _ as *mut std::ffi::c_void,
        &mut out_val as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut cin_val as *mut _ as *mut std::ffi::c_void,
        &mut h_val as *mut _ as *mut std::ffi::c_void,
        &mut w_val as *mut _ as *mut std::ffi::c_void,
        &mut cout_val as *mut _ as *mut std::ffi::c_void,
        &mut kh_val as *mut _ as *mut std::ffi::c_void,
        &mut kw_val as *mut _ as *mut std::ffi::c_void,
        &mut sh_val as *mut _ as *mut std::ffi::c_void,
        &mut sw_val as *mut _ as *mut std::ffi::c_void,
        &mut ph_val as *mut _ as *mut std::ffi::c_void,
        &mut pw_val as *mut _ as *mut std::ffi::c_void,
        &mut hout_val as *mut _ as *mut std::ffi::c_void,
        &mut wout_val as *mut _ as *mut std::ffi::c_void,
        &mut total_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = ((total as i64) + block - 1) / block;

    let result = inner::kernel_launch(
        CONV2D_F32_PTX.as_ptr(), b"nsl_conv2d_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU conv2d kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    crate::tensor::nsl_tensor_free(weight_on_gpu);

    let out = Box::new(NslTensor::new(
        out_data, out_shape, out_strides,
        4, total as i64, input.device, 1, 1, 0,
    ));
    NslTensor::publish(out)
}

// ---------------------------------------------------------------------------
// GPU MaxPool2d
// ---------------------------------------------------------------------------

/// GPU maxpool2d: out[n,c,oh,ow] = max over kernel window + argmax indices.
/// Input must be 4D NCHW. Returns output tensor; argmax stored for backward.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_maxpool2d_f32(
    input_ptr: i64, kh: u64, kw: u64, stride: u64, padding: u64,
) -> (i64, Vec<u64>) {
    use crate::tensor::NslTensor;
    use fused_kernels::MAXPOOL2D_F32_PTX;

    let input = unsafe { &*(input_ptr as *const NslTensor) };

    let n = unsafe { *input.shape.add(0) } as u64;
    let c = unsafe { *input.shape.add(1) } as u64;
    let h = unsafe { *input.shape.add(2) } as u64;
    let w = unsafe { *input.shape.add(3) } as u64;

    let h_out = (h + 2 * padding - kh) / stride + 1;
    let w_out = (w + 2 * padding - kw) / stride + 1;
    let total = n * c * h_out * w_out;

    let out_data = inner::alloc_managed(total as usize * 4); // f32
    let argmax_data = inner::alloc_managed(total as usize * 8); // u64 indices

    let out_shape = crate::memory::checked_alloc(4 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = n as i64;
        *out_shape.add(1) = c as i64;
        *out_shape.add(2) = h_out as i64;
        *out_shape.add(3) = w_out as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 4);

    let mut inp_data = input.data as u64;
    let mut out_val = out_data as u64;
    let mut argmax_val = argmax_data as u64;
    let mut n_val = n; let mut c_val = c; let mut h_val = h; let mut w_val = w;
    let mut kh_val = kh; let mut kw_val = kw;
    let mut stride_val = stride; let mut pad_val = padding;
    let mut hout_val = h_out; let mut wout_val = w_out; let mut total_val = total;

    let args: [*mut std::ffi::c_void; 14] = [
        &mut inp_data as *mut _ as *mut std::ffi::c_void,
        &mut out_val as *mut _ as *mut std::ffi::c_void,
        &mut argmax_val as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut c_val as *mut _ as *mut std::ffi::c_void,
        &mut h_val as *mut _ as *mut std::ffi::c_void,
        &mut w_val as *mut _ as *mut std::ffi::c_void,
        &mut kh_val as *mut _ as *mut std::ffi::c_void,
        &mut kw_val as *mut _ as *mut std::ffi::c_void,
        &mut stride_val as *mut _ as *mut std::ffi::c_void,
        &mut pad_val as *mut _ as *mut std::ffi::c_void,
        &mut hout_val as *mut _ as *mut std::ffi::c_void,
        &mut wout_val as *mut _ as *mut std::ffi::c_void,
        &mut total_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = ((total as i64) + block - 1) / block;

    let result = inner::kernel_launch(
        MAXPOOL2D_F32_PTX.as_ptr(), b"nsl_maxpool2d_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU maxpool2d kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    // Read argmax indices back to CPU (needed for backward tape)
    let argmax_vec: Vec<u64> = unsafe {
        std::slice::from_raw_parts(argmax_data as *const u64, total as usize).to_vec()
    };
    // Free GPU argmax buffer
    inner::free_managed(argmax_data);

    let out = Box::new(NslTensor::new(
        out_data, out_shape, out_strides,
        4, total as i64, input.device, 1, 1, 0,
    ));
    (NslTensor::publish(out), argmax_vec)
}

// ---------------------------------------------------------------------------
// GPU Dropout (inverted dropout with per-element PRNG)
// ---------------------------------------------------------------------------

/// GPU dropout: out[i] = keep ? input[i] * scale : 0, mask[i] = keep ? 1 : 0.
/// Uses a hash-based PRNG seeded from a global counter for per-element randomness.
/// Returns (output_ptr, mask_ptr) — mask is f32 on GPU for backward pass.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_dropout_f32(input_ptr: i64, p: f64) -> (i64, i64) {
    inner::set_oom_context("dropout_f32");
    use crate::tensor::NslTensor;
    use fused_kernels::DROPOUT_F32_PTX;
    use std::sync::atomic::{AtomicU64, Ordering};

    // Global seed counter — incremented per dropout call for unique masks
    static DROPOUT_SEED: AtomicU64 = AtomicU64::new(42);

    let input = unsafe { &*(input_ptr as *const NslTensor) };
    let len = input.len as u64;
    let ndim = input.ndim;

    // Allocate output and mask on GPU
    let out_data = inner::alloc_managed(len as usize * 4); // f32
    let mask_data = inner::alloc_managed(len as usize * 4); // f32 mask

    let out_shape = NslTensor::copy_shape(input.shape, ndim);
    let out_strides = NslTensor::compute_strides(out_shape, ndim);
    let mask_shape = NslTensor::copy_shape(input.shape, ndim);
    let mask_strides = NslTensor::compute_strides(mask_shape, ndim);

    // threshold: hash values below this → keep (inverted: keep probability = 1-p)
    // u32::MAX * (1-p) gives the threshold
    let threshold = ((1.0 - p) * u32::MAX as f64) as u32;
    let scale = (1.0 / (1.0 - p)) as f32;
    let seed = DROPOUT_SEED.fetch_add(len, Ordering::SeqCst);

    let mut inp_data = input.data as u64;
    let mut out_val = out_data as u64;
    let mut mask_val = mask_data as u64;
    let mut len_val = len;
    let mut thresh_val = threshold;
    let mut scale_val = scale;
    let mut seed_val = seed;

    let args: [*mut std::ffi::c_void; 7] = [
        &mut inp_data as *mut _ as *mut std::ffi::c_void,
        &mut out_val as *mut _ as *mut std::ffi::c_void,
        &mut mask_val as *mut _ as *mut std::ffi::c_void,
        &mut len_val as *mut _ as *mut std::ffi::c_void,
        &mut thresh_val as *mut _ as *mut std::ffi::c_void,
        &mut scale_val as *mut _ as *mut std::ffi::c_void,
        &mut seed_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = ((len as i64) + block - 1) / block;

    let result = inner::kernel_launch(
        DROPOUT_F32_PTX.as_ptr(), b"nsl_dropout_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU dropout kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    let out = Box::new(NslTensor::new(
        out_data, out_shape, out_strides,
        ndim, len as i64, input.device, 1, 1, 0,
    ));
    let mask = Box::new(NslTensor::new(
        mask_data, mask_shape, mask_strides,
        ndim, len as i64, input.device, 1, 1, 0,
    ));
    (NslTensor::publish(out), NslTensor::publish(mask))
}

// ---------------------------------------------------------------------------
// GPU Slice (native on-device slicing, no CPU round-trip)
// ---------------------------------------------------------------------------

/// GPU slice: extracts a contiguous sub-range along one dimension without
/// transferring to CPU. Replaces the GPU→CPU→slice→CPU→GPU round-trip.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_slice_f32(tensor_ptr: i64, dim: usize, start: usize) -> i64 {
    use crate::tensor::NslTensor;
    use fused_kernels::GPU_SLICE_F32_PTX;

    let t = unsafe { &*(tensor_ptr as *const NslTensor) };
    let ndim = t.ndim as usize;

    // Compute output shape: same as source but with sliced dim size
    let src_shape: Vec<i64> = (0..ndim).map(|i| unsafe { *t.shape.add(i) }).collect();
    let src_strides: Vec<i64> = (0..ndim).map(|i| unsafe { *t.strides.add(i) }).collect();

    // The caller already computed the slice end; we receive start and use
    // the output shape from the caller. For the kernel, we need the OUTPUT shape.
    // Since shape_ops.rs passes us the original tensor + dim + start,
    // we need to determine slice_len from the caller. Instead, let's compute
    // everything here based on the tensor and dimension info.
    // For a simple contiguous slice [start:end], the output dim is (end - start).
    // The caller in shape_ops.rs will pass us the pre-computed output shape.
    // For now, compute based on the source tensor info.

    // Actually, the better approach: the caller passes us the output shape directly.
    // Let me restructure to receive output_dim_size.
    let _ = (tensor_ptr, dim, start, ndim, src_shape, src_strides);
    // This function is called from gpu_slice_f32_with_shape below.
    unreachable!("use gpu_slice_f32_with_shape instead");
}

/// GPU slice with explicit output shape: extracts a sub-range along `dim` starting
/// at `start` with `slice_len` elements along that dimension.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_slice_f32_with_shape(
    tensor_ptr: i64,
    dim: usize,
    start: usize,
    slice_len: usize,
) -> i64 {
    use crate::tensor::NslTensor;
    use fused_kernels::GPU_SLICE_F32_PTX;

    let t = unsafe { &*(tensor_ptr as *const NslTensor) };
    let ndim = t.ndim as usize;

    // Build output shape (same as source except dim has slice_len)
    let out_shape = crate::memory::checked_alloc((ndim) * std::mem::size_of::<i64>()) as *mut i64;
    let mut total: i64 = 1;
    for i in 0..ndim {
        let s = if i == dim { slice_len as i64 } else { unsafe { *t.shape.add(i) } };
        unsafe { *out_shape.add(i) = s };
        total *= s;
    }
    let out_strides = NslTensor::compute_strides(out_shape, ndim as i64);
    let out_data = inner::alloc_managed(total as usize * 4); // f32

    // Upload metadata arrays to device
    let arr_bytes = ndim * std::mem::size_of::<i64>();
    let gpu_shape = inner::alloc_managed(arr_bytes);
    let gpu_src_strides = inner::alloc_managed(arr_bytes);
    let gpu_dst_strides = inner::alloc_managed(arr_bytes);

    inner::memcpy_htod(gpu_shape, out_shape as *const std::ffi::c_void, arr_bytes);
    inner::memcpy_htod(gpu_src_strides, t.strides as *const std::ffi::c_void, arr_bytes);
    inner::memcpy_htod(gpu_dst_strides, out_strides as *const std::ffi::c_void, arr_bytes);

    let mut src_data = t.data as u64;
    let mut dst_data = out_data as u64;
    let mut shape_val = gpu_shape as u64;
    let mut src_str_val = gpu_src_strides as u64;
    let mut dst_str_val = gpu_dst_strides as u64;
    let mut ndim_val = ndim as u64;
    let mut total_val = total as u64;
    let mut dim_val = dim as u64;
    let mut start_val = start as u64;

    let args: [*mut std::ffi::c_void; 9] = [
        &mut src_data as *mut _ as *mut std::ffi::c_void,
        &mut dst_data as *mut _ as *mut std::ffi::c_void,
        &mut shape_val as *mut _ as *mut std::ffi::c_void,
        &mut src_str_val as *mut _ as *mut std::ffi::c_void,
        &mut dst_str_val as *mut _ as *mut std::ffi::c_void,
        &mut ndim_val as *mut _ as *mut std::ffi::c_void,
        &mut total_val as *mut _ as *mut std::ffi::c_void,
        &mut dim_val as *mut _ as *mut std::ffi::c_void,
        &mut start_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = ((total as i64) + block - 1) / block;

    let result = inner::kernel_launch(
        GPU_SLICE_F32_PTX.as_ptr(), b"nsl_slice_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU slice kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    // Free GPU metadata arrays
    inner::free_managed(gpu_shape);
    inner::free_managed(gpu_src_strides);
    inner::free_managed(gpu_dst_strides);

    let out = Box::new(NslTensor::new(
        out_data, out_shape, out_strides,
        ndim as i64, total, t.device, 1, 1, 0,
    ));
    NslTensor::publish(out)
}

// ---------------------------------------------------------------------------
// M50: GPU Sparse SpMM / SpMV dispatch helpers
// These take NslSparseTensor + NslTensor and launch the appropriate kernel.
// ---------------------------------------------------------------------------

/// GPU CSR SpMM from NslSparseTensor: uploads CSR arrays to device, launches kernel.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_csr_spmm_f32_from_sparse(sparse: &crate::sparse::NslSparseTensor, dense_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    let b = unsafe { &*(dense_ptr as *const NslTensor) };
    let n_out = if b.ndim >= 2 { unsafe { *b.shape.add(b.ndim as usize - 1) } } else { 1 } as usize;
    let m = sparse.rows as usize;
    let nnz = sparse.nnz as usize;

    let out_total = m * n_out;
    let out_data = inner::alloc_managed(out_total * 4);
    inner::memset_d8(out_data, out_total * 4);

    // Convert and upload CSR arrays
    let rp_bytes = (m + 1) * 4;
    let ci_bytes = nnz * 4;
    let v_bytes = nnz * 4;

    let gpu_rp = inner::alloc_managed(rp_bytes);
    let gpu_ci = inner::alloc_managed(ci_bytes);
    let gpu_v = inner::alloc_managed(v_bytes);

    // row_ptrs: i64 → u32
    let staging_rp = crate::memory::checked_alloc(rp_bytes) as *mut u32;
    let src_rp = unsafe { std::slice::from_raw_parts(sparse.indices_0, m + 1) };
    for i in 0..m + 1 { unsafe { *staging_rp.add(i) = src_rp[i] as u32; } }
    inner::memcpy_htod(gpu_rp, staging_rp as *const std::ffi::c_void, rp_bytes);
    unsafe { crate::memory::checked_free(staging_rp as *mut u8, rp_bytes); }

    // col_indices: i64 → u32
    let staging_ci = crate::memory::checked_alloc(ci_bytes) as *mut u32;
    let src_ci = unsafe { std::slice::from_raw_parts(sparse.indices_1, nnz) };
    for i in 0..nnz { unsafe { *staging_ci.add(i) = src_ci[i] as u32; } }
    inner::memcpy_htod(gpu_ci, staging_ci as *const std::ffi::c_void, ci_bytes);
    unsafe { crate::memory::checked_free(staging_ci as *mut u8, ci_bytes); }

    // values: f64 → f32
    let staging_v = crate::memory::checked_alloc(v_bytes) as *mut f32;
    let src_v = unsafe { std::slice::from_raw_parts(sparse.data as *const f64, nnz) };
    for i in 0..nnz { unsafe { *staging_v.add(i) = src_v[i] as f32; } }
    inner::memcpy_htod(gpu_v, staging_v as *const std::ffi::c_void, v_bytes);
    unsafe { crate::memory::checked_free(staging_v as *mut u8, v_bytes); }

    let mut rp = gpu_rp as u64;
    let mut ci = gpu_ci as u64;
    let mut v = gpu_v as u64;
    let mut bd = b.data as u64;
    let mut cd = out_data as u64;
    let mut m_val = m as u64;
    let mut n_val = n_out as u64;

    let args: [*mut std::ffi::c_void; 7] = [
        &mut rp as *mut _ as *mut std::ffi::c_void,
        &mut ci as *mut _ as *mut std::ffi::c_void,
        &mut v as *mut _ as *mut std::ffi::c_void,
        &mut bd as *mut _ as *mut std::ffi::c_void,
        &mut cd as *mut _ as *mut std::ffi::c_void,
        &mut m_val as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid_x = m as i64;
    let grid_y = ((n_out as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        fused_kernels::CSR_SPMM_F32_PTX.as_ptr(), b"nsl_csr_spmm_f32\0".as_ptr(),
        [grid_x, grid_y, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU CSR SpMM kernel failed");
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    inner::free_managed(gpu_rp);
    inner::free_managed(gpu_ci);
    inner::free_managed(gpu_v);

    let out_shape = crate::memory::checked_alloc(2 * 8) as *mut i64;
    unsafe { *out_shape = m as i64; *out_shape.add(1) = n_out as i64; }
    let out_strides = NslTensor::compute_strides(out_shape, 2);
    let out = Box::new(NslTensor::new(out_data, out_shape, out_strides, 2, out_total as i64, b.device, 1, 1, 0));
    NslTensor::publish(out)
}

/// GPU COO SpMM: C[M,N] = A_coo @ B[K,N]. Uploads COO arrays to device, launches kernel.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_coo_spmm_f32(sparse: &crate::sparse::NslSparseTensor, dense_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    let b = unsafe { &*(dense_ptr as *const NslTensor) };
    let n_out = if b.ndim >= 2 { unsafe { *b.shape.add(b.ndim as usize - 1) } } else { 1 } as usize;
    let m = sparse.rows as usize;
    let nnz = sparse.nnz as usize;

    let out_total = m * n_out;
    let out_data = inner::alloc_managed(out_total * 4);
    inner::memset_d8(out_data, out_total * 4);

    // Upload COO arrays: row_indices(i64), col_indices(i64), values(f32)
    let ri_bytes = nnz * 8; // i64
    let ci_bytes = nnz * 8;
    let v_bytes = nnz * 4;  // f32

    let gpu_ri = inner::alloc_managed(ri_bytes);
    let gpu_ci = inner::alloc_managed(ci_bytes);
    let gpu_v = inner::alloc_managed(v_bytes);

    // Convert f64 values to f32 for GPU, upload indices as-is (i64)
    inner::memcpy_htod(gpu_ri, sparse.indices_0 as *const std::ffi::c_void, ri_bytes);
    inner::memcpy_htod(gpu_ci, sparse.indices_1 as *const std::ffi::c_void, ci_bytes);
    // Values: sparse stores f64 on CPU, need f32 for GPU kernel
    let staging = crate::memory::checked_alloc(v_bytes) as *mut f32;
    let src_vals = unsafe { std::slice::from_raw_parts(sparse.data as *const f64, nnz) };
    for i in 0..nnz { unsafe { *staging.add(i) = src_vals[i] as f32; } }
    inner::memcpy_htod(gpu_v, staging as *const std::ffi::c_void, v_bytes);
    unsafe { crate::memory::checked_free(staging as *mut u8, v_bytes); }

    let mut ri_val = gpu_ri as u64;
    let mut ci_val = gpu_ci as u64;
    let mut v_val = gpu_v as u64;
    let mut b_data = b.data as u64;
    let mut c_data = out_data as u64;
    let mut n_val = n_out as u64;
    let mut nnz_val = nnz as u64;

    let args: [*mut std::ffi::c_void; 7] = [
        &mut ri_val as *mut _ as *mut std::ffi::c_void,
        &mut ci_val as *mut _ as *mut std::ffi::c_void,
        &mut v_val as *mut _ as *mut std::ffi::c_void,
        &mut b_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut nnz_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = ((nnz as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        fused_kernels::COO_SPMM_F32_PTX.as_ptr(), b"nsl_coo_spmm_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU COO SpMM kernel failed");
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    inner::free_managed(gpu_ri);
    inner::free_managed(gpu_ci);
    inner::free_managed(gpu_v);

    let out_shape = crate::memory::checked_alloc(2 * 8) as *mut i64;
    unsafe { *out_shape = m as i64; *out_shape.add(1) = n_out as i64; }
    let out_strides = NslTensor::compute_strides(out_shape, 2);
    let out = Box::new(NslTensor::new(out_data, out_shape, out_strides, 2, out_total as i64, b.device, 1, 1, 0));
    NslTensor::publish(out)
}

/// GPU CSR SpMV: y[M] = A_csr @ x[K]. One thread per row.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_csr_spmv_f32(sparse: &crate::sparse::NslSparseTensor, vec_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    let x = unsafe { &*(vec_ptr as *const NslTensor) };
    let m = sparse.rows as usize;
    let nnz = sparse.nnz as usize;

    let out_data = inner::alloc_managed(m * 4);
    inner::memset_d8(out_data, m * 4);

    // Upload CSR: row_ptrs as u32, col_indices as u32, values as f32
    let rp_bytes = (m + 1) * 4;
    let ci_bytes = nnz * 4;
    let v_bytes = nnz * 4;

    let gpu_rp = inner::alloc_managed(rp_bytes);
    let gpu_ci = inner::alloc_managed(ci_bytes);
    let gpu_v = inner::alloc_managed(v_bytes);

    // Convert i64 row_ptrs/col_indices to u32 for GPU, f64 values to f32
    let staging_rp = crate::memory::checked_alloc(rp_bytes) as *mut u32;
    let src_rp = unsafe { std::slice::from_raw_parts(sparse.indices_0, m + 1) };
    for i in 0..m + 1 { unsafe { *staging_rp.add(i) = src_rp[i] as u32; } }
    inner::memcpy_htod(gpu_rp, staging_rp as *const std::ffi::c_void, rp_bytes);
    unsafe { crate::memory::checked_free(staging_rp as *mut u8, rp_bytes); }

    let staging_ci = crate::memory::checked_alloc(ci_bytes) as *mut u32;
    let src_ci = unsafe { std::slice::from_raw_parts(sparse.indices_1, nnz) };
    for i in 0..nnz { unsafe { *staging_ci.add(i) = src_ci[i] as u32; } }
    inner::memcpy_htod(gpu_ci, staging_ci as *const std::ffi::c_void, ci_bytes);
    unsafe { crate::memory::checked_free(staging_ci as *mut u8, ci_bytes); }

    let staging_v = crate::memory::checked_alloc(v_bytes) as *mut f32;
    let src_v = unsafe { std::slice::from_raw_parts(sparse.data as *const f64, nnz) };
    for i in 0..nnz { unsafe { *staging_v.add(i) = src_v[i] as f32; } }
    inner::memcpy_htod(gpu_v, staging_v as *const std::ffi::c_void, v_bytes);
    unsafe { crate::memory::checked_free(staging_v as *mut u8, v_bytes); }

    let mut rp = gpu_rp as u64;
    let mut ci = gpu_ci as u64;
    let mut v = gpu_v as u64;
    let mut xd = x.data as u64;
    let mut yd = out_data as u64;
    let mut m_val = m as u64;

    let args: [*mut std::ffi::c_void; 6] = [
        &mut rp as *mut _ as *mut std::ffi::c_void,
        &mut ci as *mut _ as *mut std::ffi::c_void,
        &mut v as *mut _ as *mut std::ffi::c_void,
        &mut xd as *mut _ as *mut std::ffi::c_void,
        &mut yd as *mut _ as *mut std::ffi::c_void,
        &mut m_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = ((m as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        fused_kernels::CSR_SPMV_F32_PTX.as_ptr(), b"nsl_csr_spmv_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU CSR SpMV kernel failed");
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    inner::free_managed(gpu_rp);
    inner::free_managed(gpu_ci);
    inner::free_managed(gpu_v);

    let out_shape = crate::memory::checked_alloc(8) as *mut i64;
    unsafe { *out_shape = m as i64; }
    let out_strides = NslTensor::compute_strides(out_shape, 1);
    let out = Box::new(NslTensor::new(out_data, out_shape, out_strides, 1, m as i64, x.device, 1, 1, 0));
    NslTensor::publish(out)
}

/// GPU BSR SpMM: C[M,N] = A_bsr @ B[K,N]. Uploads BSR arrays, launches kernel.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_bsr_spmm_f32(sparse: &crate::sparse::NslSparseTensor, dense_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    let b = unsafe { &*(dense_ptr as *const NslTensor) };
    let n_out = if b.ndim >= 2 { unsafe { *b.shape.add(b.ndim as usize - 1) } } else { 1 } as usize;
    let m = sparse.rows as usize;
    let br = sparse.block_rows as usize;
    let bc = sparse.block_cols as usize;
    if br == 0 || bc == 0 { return 0; }
    let nblk_rows = (m + br - 1) / br;
    let num_blocks = sparse.nnz as usize;
    let block_size = br * bc;

    let out_total = m * n_out;
    let out_data = inner::alloc_managed(out_total * 4);
    inner::memset_d8(out_data, out_total * 4);

    // Upload BSR arrays: row_ptrs(u32), col_indices(u32), values(f32)
    let rp_bytes = (nblk_rows + 1) * 4;
    let ci_bytes = num_blocks * 4;
    let v_bytes = num_blocks * block_size * 4;

    let gpu_rp = inner::alloc_managed(rp_bytes);
    let gpu_ci = inner::alloc_managed(ci_bytes);
    let gpu_v = inner::alloc_managed(v_bytes);

    // Convert i64 → u32 for row_ptrs
    let staging_rp = crate::memory::checked_alloc(rp_bytes) as *mut u32;
    let src_rp = unsafe { std::slice::from_raw_parts(sparse.indices_0, nblk_rows + 1) };
    for i in 0..nblk_rows + 1 { unsafe { *staging_rp.add(i) = src_rp[i] as u32; } }
    inner::memcpy_htod(gpu_rp, staging_rp as *const std::ffi::c_void, rp_bytes);
    unsafe { crate::memory::checked_free(staging_rp as *mut u8, rp_bytes); }

    // Convert i64 → u32 for col_indices
    let staging_ci = crate::memory::checked_alloc(ci_bytes) as *mut u32;
    let src_ci = unsafe { std::slice::from_raw_parts(sparse.indices_1, num_blocks) };
    for i in 0..num_blocks { unsafe { *staging_ci.add(i) = src_ci[i] as u32; } }
    inner::memcpy_htod(gpu_ci, staging_ci as *const std::ffi::c_void, ci_bytes);
    unsafe { crate::memory::checked_free(staging_ci as *mut u8, ci_bytes); }

    // Convert f64 → f32 for values
    let staging_v = crate::memory::checked_alloc(v_bytes) as *mut f32;
    let src_v = unsafe { std::slice::from_raw_parts(sparse.data as *const f64, num_blocks * block_size) };
    for i in 0..num_blocks * block_size { unsafe { *staging_v.add(i) = src_v[i] as f32; } }
    inner::memcpy_htod(gpu_v, staging_v as *const std::ffi::c_void, v_bytes);
    unsafe { crate::memory::checked_free(staging_v as *mut u8, v_bytes); }

    let mut rp = gpu_rp as u64;
    let mut ci = gpu_ci as u64;
    let mut v = gpu_v as u64;
    let mut bd = b.data as u64;
    let mut cd = out_data as u64;
    let mut n_val = n_out as u64;
    let mut br_val = br as u64;
    let mut bc_val = bc as u64;
    let mut nblk_val = nblk_rows as u64;

    let args: [*mut std::ffi::c_void; 9] = [
        &mut rp as *mut _ as *mut std::ffi::c_void,
        &mut ci as *mut _ as *mut std::ffi::c_void,
        &mut v as *mut _ as *mut std::ffi::c_void,
        &mut bd as *mut _ as *mut std::ffi::c_void,
        &mut cd as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut br_val as *mut _ as *mut std::ffi::c_void,
        &mut bc_val as *mut _ as *mut std::ffi::c_void,
        &mut nblk_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block_x = 256i64.min(n_out as i64).max(1);
    let grid_x = nblk_rows as i64;
    let grid_y = ((n_out as i64) + block_x - 1) / block_x;
    let result = inner::kernel_launch(
        fused_kernels::BSR_SPMM_F32_PTX.as_ptr(), b"nsl_bsr_spmm_f32\0".as_ptr(),
        [grid_x, grid_y, 1], [block_x, br as i64, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU BSR SpMM kernel failed");
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    inner::free_managed(gpu_rp);
    inner::free_managed(gpu_ci);
    inner::free_managed(gpu_v);

    let out_shape = crate::memory::checked_alloc(2 * 8) as *mut i64;
    unsafe { *out_shape = m as i64; *out_shape.add(1) = n_out as i64; }
    let out_strides = NslTensor::compute_strides(out_shape, 2);
    let out = Box::new(NslTensor::new(out_data, out_shape, out_strides, 2, out_total as i64, b.device, 1, 1, 0));
    NslTensor::publish(out)
}

/// GPU COO SpMV: y[M] = A_coo @ x[K]. One thread per nonzero, atomic add.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_coo_spmv_f32(sparse: &crate::sparse::NslSparseTensor, vec_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    let x = unsafe { &*(vec_ptr as *const NslTensor) };
    let m = sparse.rows as usize;
    let nnz = sparse.nnz as usize;

    let out_data = inner::alloc_managed(m * 4);
    inner::memset_d8(out_data, m * 4);

    let ri_bytes = nnz * 8;
    let ci_bytes = nnz * 8;
    let v_bytes = nnz * 4;

    let gpu_ri = inner::alloc_managed(ri_bytes);
    let gpu_ci = inner::alloc_managed(ci_bytes);
    let gpu_v = inner::alloc_managed(v_bytes);

    inner::memcpy_htod(gpu_ri, sparse.indices_0 as *const std::ffi::c_void, ri_bytes);
    inner::memcpy_htod(gpu_ci, sparse.indices_1 as *const std::ffi::c_void, ci_bytes);
    let staging_v = crate::memory::checked_alloc(v_bytes) as *mut f32;
    let src_v = unsafe { std::slice::from_raw_parts(sparse.data as *const f64, nnz) };
    for i in 0..nnz { unsafe { *staging_v.add(i) = src_v[i] as f32; } }
    inner::memcpy_htod(gpu_v, staging_v as *const std::ffi::c_void, v_bytes);
    unsafe { crate::memory::checked_free(staging_v as *mut u8, v_bytes); }

    let mut ri = gpu_ri as u64;
    let mut ci = gpu_ci as u64;
    let mut v = gpu_v as u64;
    let mut xd = x.data as u64;
    let mut yd = out_data as u64;
    let mut nnz_v = nnz as u64;

    let args: [*mut std::ffi::c_void; 6] = [
        &mut ri as *mut _ as *mut std::ffi::c_void,
        &mut ci as *mut _ as *mut std::ffi::c_void,
        &mut v as *mut _ as *mut std::ffi::c_void,
        &mut xd as *mut _ as *mut std::ffi::c_void,
        &mut yd as *mut _ as *mut std::ffi::c_void,
        &mut nnz_v as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = ((nnz as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        fused_kernels::COO_SPMV_F32_PTX.as_ptr(), b"nsl_coo_spmv_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU COO SpMV kernel failed");
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    inner::free_managed(gpu_ri);
    inner::free_managed(gpu_ci);
    inner::free_managed(gpu_v);

    let out_shape = crate::memory::checked_alloc(8) as *mut i64;
    unsafe { *out_shape = m as i64; }
    let out_strides = NslTensor::compute_strides(out_shape, 1);
    let out = Box::new(NslTensor::new(out_data, out_shape, out_strides, 1, m as i64, x.device, 1, 1, 0));
    NslTensor::publish(out)
}

// ---------------------------------------------------------------------------
// GPU Strided Copy (contiguous materialization on-device)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// GPU CSR Sparse MatMul (M52c: weight-aware sparse kernel)
// ---------------------------------------------------------------------------

/// CSR SpMM: C[M,N] = A_sparse[M,K] @ B_dense[K,N]
/// CSR arrays (row_ptrs, col_indices, values) are host pointers — uploaded to device.
/// B is a device tensor pointer. Returns a new dense tensor C.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_sparse_matmul_csr_f32(
    row_ptrs: &[u32],
    col_indices: &[u32],
    values_f32: &[f32],
    b_ptr: i64,
    nrows: usize,
    ncols: usize,
    nnz: usize,
) -> i64 {
    use crate::tensor::NslTensor;
    use fused_kernels::CSR_SPMM_F32_PTX;

    let b = unsafe { &*(b_ptr as *const NslTensor) };
    // B must be 2D: [K, N]
    let last_dim = if b.ndim >= 2 {
        unsafe { *b.shape.add(b.ndim as usize - 1) }
    } else {
        1
    };
    let n_out = last_dim as usize;

    // Allocate output C[nrows, n_out]
    let out_total = nrows * n_out;
    let out_data = inner::alloc_managed(out_total * 4); // f32
    inner::memset_d8(out_data, out_total * 4);

    // Upload CSR arrays to device
    let rp_bytes = (nrows + 1) * std::mem::size_of::<u32>();
    let ci_bytes = nnz * std::mem::size_of::<u32>();
    let val_bytes = nnz * std::mem::size_of::<f32>();

    let gpu_row_ptrs = inner::alloc_managed(rp_bytes);
    let gpu_col_indices = inner::alloc_managed(ci_bytes);
    let gpu_values = inner::alloc_managed(val_bytes);

    inner::memcpy_htod(gpu_row_ptrs, row_ptrs.as_ptr() as *const std::ffi::c_void, rp_bytes);
    inner::memcpy_htod(gpu_col_indices, col_indices.as_ptr() as *const std::ffi::c_void, ci_bytes);
    inner::memcpy_htod(gpu_values, values_f32.as_ptr() as *const std::ffi::c_void, val_bytes);

    let mut rp_val = gpu_row_ptrs as u64;
    let mut ci_val = gpu_col_indices as u64;
    let mut v_val = gpu_values as u64;
    let mut b_data = b.data as u64;
    let mut c_data = out_data as u64;
    let mut m_val = nrows as u64;
    let mut n_val = n_out as u64;

    let args: [*mut std::ffi::c_void; 7] = [
        &mut rp_val as *mut _ as *mut std::ffi::c_void,
        &mut ci_val as *mut _ as *mut std::ffi::c_void,
        &mut v_val as *mut _ as *mut std::ffi::c_void,
        &mut b_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut m_val as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];

    // Grid: [nrows, ceil(N/256), 1], Block: [256, 1, 1]
    let block = 256i64;
    let grid_x = nrows as i64;
    let grid_y = ((n_out as i64) + block - 1) / block;

    let result = inner::kernel_launch(
        CSR_SPMM_F32_PTX.as_ptr(), b"nsl_csr_spmm_f32\0".as_ptr(),
        [grid_x, grid_y, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU CSR SpMM kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    // Free device CSR arrays
    inner::free_managed(gpu_row_ptrs);
    inner::free_managed(gpu_col_indices);
    inner::free_managed(gpu_values);

    // Build output tensor [nrows, n_out]
    let out_shape = crate::memory::checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape = nrows as i64;
        *out_shape.add(1) = n_out as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 2);
    let out = Box::new(NslTensor::new(
        out_data, out_shape, out_strides, 2, out_total as i64, b.device, 1, 1, 0,
    ));
    NslTensor::publish(out)
}

/// GPU strided copy: materializes a non-contiguous view into a contiguous tensor.
/// Replaces the CPU round-trip (GPU→CPU copy→GPU) with a single on-device kernel.
/// The kernel decomposes each flat output index into N-dim coords using dst_strides,
/// then computes the source offset using the source's non-contiguous strides.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_strided_copy_f32(tensor_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    use fused_kernels::STRIDED_COPY_F32_PTX;

    let t = unsafe { &*(tensor_ptr as *const NslTensor) };
    let ndim = t.ndim as usize;
    let total = t.len as u64;

    // Allocate contiguous output on GPU
    let out_data = inner::alloc_managed(total as usize * 4); // f32
    let out_shape = NslTensor::copy_shape(t.shape, t.ndim);
    let out_strides = NslTensor::compute_strides(out_shape, t.ndim);

    // Upload shape, src_strides, and dst_strides to device memory
    let arr_bytes = ndim * std::mem::size_of::<i64>();

    let gpu_shape = inner::alloc_managed(arr_bytes);
    let gpu_src_strides = inner::alloc_managed(arr_bytes);
    let gpu_dst_strides = inner::alloc_managed(arr_bytes);

    // Copy metadata from host to device via explicit memcpy
    inner::memcpy_htod(gpu_shape, t.shape as *const c_void, arr_bytes);
    inner::memcpy_htod(gpu_src_strides, t.strides as *const c_void, arr_bytes);
    inner::memcpy_htod(gpu_dst_strides, out_strides as *const c_void, arr_bytes);

    let mut src_data = t.data as u64;
    let mut dst_data = out_data as u64;
    let mut shape_val = gpu_shape as u64;
    let mut src_str_val = gpu_src_strides as u64;
    let mut dst_str_val = gpu_dst_strides as u64;
    #[allow(unused)]
    let _ = (&shape_val, &src_str_val, &dst_str_val); // used as kernel args below
    let mut ndim_val = ndim as u64;
    let mut total_val = total;

    let args: [*mut std::ffi::c_void; 7] = [
        &mut src_data as *mut _ as *mut std::ffi::c_void,
        &mut dst_data as *mut _ as *mut std::ffi::c_void,
        &mut shape_val as *mut _ as *mut std::ffi::c_void,
        &mut src_str_val as *mut _ as *mut std::ffi::c_void,
        &mut dst_str_val as *mut _ as *mut std::ffi::c_void,
        &mut ndim_val as *mut _ as *mut std::ffi::c_void,
        &mut total_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = ((total as i64) + block - 1) / block;

    let result = inner::kernel_launch(
        STRIDED_COPY_F32_PTX.as_ptr(), b"nsl_strided_copy_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU strided copy kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    // Free GPU metadata arrays
    inner::free_managed(gpu_shape);
    inner::free_managed(gpu_src_strides);
    inner::free_managed(gpu_dst_strides);

    let out = Box::new(NslTensor::new(
        out_data, out_shape, out_strides,
        t.ndim, total as i64, t.device, 1, 1, 0,
    ));
    NslTensor::publish(out)
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;

    const VEC_ADD_PTX: &str = "\
.version 7.0
.target sm_52
.address_size 64

.visible .entry vec_add(
    .param .u64 a_ptr,
    .param .u64 b_ptr,
    .param .u64 c_ptr,
    .param .u64 n
) {
    .reg .u32 %r<4>;
    .reg .u64 %rd<8>;
    .reg .f32 %fs<4>;
    .reg .pred %p1;

    ld.param.u64 %rd1, [a_ptr];
    ld.param.u64 %rd2, [b_ptr];
    ld.param.u64 %rd3, [c_ptr];
    ld.param.u64 %rd4, [n];

    mov.u32 %r1, %ctaid.x;
    mov.u32 %r2, %ntid.x;
    mul.lo.u32 %r3, %r1, %r2;
    mov.u32 %r1, %tid.x;
    add.u32 %r3, %r3, %r1;
    cvt.u64.u32 %rd5, %r3;
    setp.ge.u64 %p1, %rd5, %rd4;
    @%p1 bra DONE;

    shl.b64 %rd6, %rd5, 2;
    add.u64 %rd7, %rd1, %rd6;
    ld.global.f32 %fs1, [%rd7];
    add.u64 %rd7, %rd2, %rd6;
    ld.global.f32 %fs2, [%rd7];
    add.f32 %fs3, %fs1, %fs2;
    add.u64 %rd7, %rd3, %rd6;
    st.global.f32 [%rd7], %fs3;

DONE:
    ret;
}\0";

    #[test]
    fn test_vec_add_kernel_launch() {
        let n: usize = 1024;
        let size_bytes = n * std::mem::size_of::<f32>();

        // Allocate unified memory
        let a = inner::alloc_managed(size_bytes);
        let b = inner::alloc_managed(size_bytes);
        let c = inner::alloc_managed(size_bytes);

        // Fill a and b on CPU (unified memory allows this)
        let a_slice = unsafe { std::slice::from_raw_parts_mut(a as *mut f32, n) };
        let b_slice = unsafe { std::slice::from_raw_parts_mut(b as *mut f32, n) };
        for i in 0..n {
            a_slice[i] = i as f32;
            b_slice[i] = (i * 2) as f32;
        }

        // Launch kernel
        let n_val = n as u64;
        let mut a_arg = a as u64;
        let mut b_arg = b as u64;
        let mut c_arg = c as u64;
        let mut n_arg = n_val;

        let args: [*mut std::ffi::c_void; 4] = [
            &mut a_arg as *mut _ as *mut std::ffi::c_void,
            &mut b_arg as *mut _ as *mut std::ffi::c_void,
            &mut c_arg as *mut _ as *mut std::ffi::c_void,
            &mut n_arg as *mut _ as *mut std::ffi::c_void,
        ];

        let block_size = 256i64;
        let grid_size = ((n as i64) + block_size - 1) / block_size;

        let result = inner::kernel_launch(
            VEC_ADD_PTX.as_ptr(),
            "vec_add\0".as_ptr(),
            [grid_size, 1, 1],
            [block_size, 1, 1],
            &args.map(|p| p), 0,
        );
        assert_eq!(result as u32, 0, "kernel launch failed");

        // Synchronize to ensure kernel completed
        unsafe {
            let sync = cudarc::driver::sys::cuCtxSynchronize();
            assert_eq!(sync as u32, 0, "sync failed");
        }

        // Verify results on CPU (unified memory)
        let c_slice = unsafe { std::slice::from_raw_parts(c as *const f32, n) };
        for i in 0..n {
            let expected = (i + i * 2) as f32;
            assert_eq!(c_slice[i], expected, "mismatch at index {}", i);
        }

        // Cleanup
        inner::free_managed(a);
        inner::free_managed(b);
        inner::free_managed(c);
    }

    #[test]
    fn test_alloc_free_device() {
        // Allocate 1024 bytes of device-only memory and free it — no crash expected.
        let ptr = inner::alloc_device(1024);
        assert!(!ptr.is_null(), "alloc_device returned null");
        inner::free_device(ptr);
    }

    #[test]
    fn test_alloc_free_pinned() {
        // Allocate 256 bytes of pinned host memory, write/read CPU-side, then free.
        let ptr = inner::alloc_pinned(256);
        assert!(!ptr.is_null(), "alloc_pinned returned null");

        // Write and read back on the CPU side (pinned memory is host-accessible).
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr as *mut u8, 256) };
        for (i, byte) in slice.iter_mut().enumerate() {
            *byte = (i % 256) as u8;
        }
        for (i, &byte) in slice.iter().enumerate() {
            assert_eq!(byte, (i % 256) as u8, "pinned memory mismatch at byte {}", i);
        }

        inner::free_pinned(ptr);
    }

    #[test]
    fn test_memcpy_htod() {
        // Copy host data into device memory and free — verifies the copy doesn't crash.
        let host_data: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let size_bytes = host_data.len() * std::mem::size_of::<f32>();

        let dev_ptr = inner::alloc_device(size_bytes);
        assert!(!dev_ptr.is_null(), "alloc_device returned null");

        inner::memcpy_htod(dev_ptr, host_data.as_ptr() as *const std::ffi::c_void, size_bytes);

        // Sync to ensure transfer is complete before freeing.
        unsafe {
            let sync = cudarc::driver::sys::cuCtxSynchronize();
            assert_eq!(sync as u32, 0, "cuCtxSynchronize after memcpy_htod failed");
        }

        inner::free_device(dev_ptr);
    }

    #[test]
    fn test_tensor_to_device_roundtrip() {
        use crate::tensor::{NslTensor, nsl_tensor_to_device};

        // Create a CPU tensor manually: [1.0, 2.0, 3.0, 4.0]
        let data = vec![1.0f64, 2.0, 3.0, 4.0];
        let shape = vec![4i64];
        let strides = vec![1i64];
        let t = Box::new(NslTensor::new(
            data.as_ptr() as *mut std::ffi::c_void,
            shape.as_ptr() as *mut i64,
            strides.as_ptr() as *mut i64,
            1,
            4,
            0,
            0,
            1,
            0,
        ));
        // Leak the vecs so the tensor can use them
        std::mem::forget(data);
        std::mem::forget(shape);
        std::mem::forget(strides);
        let cpu_tensor = Box::into_raw(t) as i64;

        // Transfer CPU → GPU
        let gpu_tensor = nsl_tensor_to_device(cpu_tensor, 1);
        let gpu_t = unsafe { &*(gpu_tensor as *const NslTensor) };
        assert_eq!(gpu_t.device, 1);
        assert_eq!(gpu_t.dtype, 1); // f32

        // Transfer GPU → CPU
        let cpu_back = nsl_tensor_to_device(gpu_tensor, 0);
        let cpu_t = unsafe { &*(cpu_back as *const NslTensor) };
        assert_eq!(cpu_t.device, 0);
        assert_eq!(cpu_t.dtype, 0); // f64

        // Verify values survived the roundtrip (f64 → f32 → f64)
        for i in 0..4 {
            let val = unsafe { *cpu_t.data_f64().add(i) };
            let expected = (i + 1) as f64;
            assert!((val - expected).abs() < 1e-6, "mismatch at {}: {} vs {}", i, val, expected);
        }
    }

    #[test]
    fn test_gpu_matmul() {
        use crate::tensor::{NslTensor, nsl_tensor_to_device, nsl_tensor_matmul};

        // A = [[1,2,3],[4,5,6]] (2x3)
        let a_data = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0];
        let a_shape = vec![2i64, 3];
        let a_strides = vec![3i64, 1];
        let a = Box::new(NslTensor::new(
            a_data.as_ptr() as *mut std::ffi::c_void,
            a_shape.as_ptr() as *mut i64,
            a_strides.as_ptr() as *mut i64,
            2,
            6,
            0,
            0,
            1,
            0,
        ));
        std::mem::forget(a_data); std::mem::forget(a_shape); std::mem::forget(a_strides);
        let a_cpu = Box::into_raw(a) as i64;

        // B = [[7,8],[9,10],[11,12]] (3x2)
        let b_data = vec![7.0f64, 8.0, 9.0, 10.0, 11.0, 12.0];
        let b_shape = vec![3i64, 2];
        let b_strides = vec![2i64, 1];
        let b = Box::new(NslTensor::new(
            b_data.as_ptr() as *mut std::ffi::c_void,
            b_shape.as_ptr() as *mut i64,
            b_strides.as_ptr() as *mut i64,
            2,
            6,
            0,
            0,
            1,
            0,
        ));
        std::mem::forget(b_data); std::mem::forget(b_shape); std::mem::forget(b_strides);
        let b_cpu = Box::into_raw(b) as i64;

        // Transfer to GPU
        let a_gpu = nsl_tensor_to_device(a_cpu, 1);
        let b_gpu = nsl_tensor_to_device(b_cpu, 1);

        // Matmul on GPU
        let c_gpu = nsl_tensor_matmul(a_gpu, b_gpu);

        // Sync and transfer back
        unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
        let c_cpu = nsl_tensor_to_device(c_gpu, 0);
        let c = unsafe { &*(c_cpu as *const NslTensor) };

        // Expected: [[58, 64], [139, 154]]
        // 1*7+2*9+3*11=58, 1*8+2*10+3*12=64
        // 4*7+5*9+6*11=139, 4*8+5*10+6*12=154
        let expected = [58.0, 64.0, 139.0, 154.0];
        for i in 0..4 {
            let val = unsafe { *c.data_f64().add(i) };
            assert!((val - expected[i]).abs() < 0.5, "matmul mismatch at {}: {} vs {}", i, val, expected[i]);
        }
    }

    #[test]
    fn test_gpu_elementwise_add() {
        use crate::tensor::{NslTensor, nsl_tensor_to_device, nsl_tensor_add};

        // Create CPU tensors manually
        let a_data = vec![1.0f64, 2.0, 3.0, 4.0];
        let b_data = vec![10.0f64, 20.0, 30.0, 40.0];
        let shape = vec![4i64];
        let strides = vec![1i64];

        let a = Box::new(NslTensor::new(
            a_data.as_ptr() as *mut std::ffi::c_void,
            shape.as_ptr() as *mut i64,
            strides.as_ptr() as *mut i64,
            1,
            4,
            0,
            0,
            1,
            0,
        ));
        std::mem::forget(a_data); std::mem::forget(shape.clone()); std::mem::forget(strides.clone());
        let a_cpu = Box::into_raw(a) as i64;

        let shape2 = vec![4i64];
        let strides2 = vec![1i64];
        let b = Box::new(NslTensor::new(
            b_data.as_ptr() as *mut std::ffi::c_void,
            shape2.as_ptr() as *mut i64,
            strides2.as_ptr() as *mut i64,
            1,
            4,
            0,
            0,
            1,
            0,
        ));
        std::mem::forget(b_data); std::mem::forget(shape2); std::mem::forget(strides2);
        let b_cpu = Box::into_raw(b) as i64;

        // Transfer to GPU
        let a_gpu = nsl_tensor_to_device(a_cpu, 1);
        let b_gpu = nsl_tensor_to_device(b_cpu, 1);

        // Add on GPU
        let c_gpu = nsl_tensor_add(a_gpu, b_gpu);

        // Sync and transfer back
        unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
        let c_cpu = nsl_tensor_to_device(c_gpu, 0);
        let c = unsafe { &*(c_cpu as *const NslTensor) };

        let expected = [11.0, 22.0, 33.0, 44.0];
        for i in 0..4 {
            let val = unsafe { *c.data_f64().add(i) };
            assert!((val - expected[i]).abs() < 0.1, "mismatch at {}: {} vs {}", i, val, expected[i]);
        }
    }
}
