//! CUDA runtime: context management, kernel launch, module cache.
//! Only compiled when the `cuda` feature is enabled.

#[cfg(feature = "cuda")]
use std::ffi::c_void;
// AtomicI64 and Ordering used by inner module functions when cuda feature is enabled

pub(crate) mod kernels;
pub(crate) mod fused_kernels;
pub(crate) mod fused_ce_kernels;
pub(crate) mod fused_kl_ce_kernels;
pub(crate) mod kernels_hopper;
pub(crate) mod precision_cast_kernels;
pub(crate) mod strided_copy;
pub(crate) mod tier_b1_prepass;

#[cfg(feature = "cuda")]
pub(crate) mod caching_allocator;

pub(crate) mod graph_capture;

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
        // Keyed by FNV-1a hash of PTX content so different PTX Vecs at the
        // same address (heap reuse between sequential test calls) don't
        // produce stale cache hits.  Raw pointer was used previously but
        // caused CUDA_ERROR_NOT_FOUND (rc=500) when a new PTX Vec was
        // allocated at the same address as an old one.
        module_cache: HashMap<u64, CUmodule>,
        // Resolved CUfunction handles, keyed by (module content hash,
        // FNV-1a of the entry name). Content-derived keys inherit the
        // module cache's immunity to heap-address reuse; a CUfunction
        // stays valid as long as its module is loaded (modules are never
        // unloaded here).
        func_cache: HashMap<(u64, u64), CUfunction>,
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

    /// Milestone C·p3: post-kernel synchronization policy.
    ///
    /// Every kernel launches on one per-thread compute stream (self-ordering;
    /// p8 PR-A — NULL under NSL_LEGACY_NULL_STREAM=1), and every host
    /// read goes through the *synchronous* `cuMemcpyDtoH_v2` — itself a
    /// NULL-stream barrier. So a `cuCtxSynchronize` after a pure-GPU kernel is
    /// redundant for correctness: it only forces the CPU to block until the GPU
    /// drains, destroying CPU/GPU overlap and serializing launch overhead with
    /// execution. `kernel_launch` already gates its own post-launch sync on
    /// `sync_mode_enabled()`; this helper lets the op sites that added a
    /// *separate* unconditional sync adopt the same policy in one call.
    ///
    /// Default: no-op (stream ordering carries correctness). With
    /// `NSL_CUDA_SYNC=1` (CLI `--cuda-sync`) it restores an eager
    /// `cuCtxSynchronize` — the bisection kill-switch: if a result changes with
    /// it off but matches with it on, a genuine host-read site was mis-gated and
    /// must keep an explicit sync.
    #[inline]
    pub(crate) fn sync_after_kernel() {
        if sync_mode_enabled() {
            unsafe {
                cuCtxSynchronize();
            }
        }
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

    /// The device this process's CUDA state is bound to.
    ///
    /// For keying caches that hold device pointers: those are per-device, so a
    /// contents-only key would hand a pointer allocated on one device to a
    /// kernel launched on another.
    pub(crate) fn current_device_ordinal() -> i32 {
        let s = state();
        let guard = s.lock().unwrap();
        guard.device as i32
    }

    /// Non-panicking probe: has the process-wide CUDA state already been
    /// initialized (by some prior tensor op)? Diagnostics-only callers
    /// (e.g. the NSL_PHASE_TIMING device sync) must NOT force-initialize
    /// CUDA — `state()`'s lazy init asserts on cuInit failure, which would
    /// abort a pure-CPU run of a cuda-featured binary on a GPU-less
    /// machine from inside an instrumentation path.
    pub(crate) fn context_initialized() -> bool {
        CUDA_STATE.get().is_some()
    }

    /// P4 item 14: pick this process's CUDA device ordinal.
    ///
    /// Priority: explicit `NSL_CUDA_DEVICE=k` override, else — under the
    /// SPMD spawner (`NSL_LOCAL_RANK` set) — `rank % device_count` so an
    /// N-rank run on an M-GPU node stripes ranks across devices, else 0
    /// (the historical single-process behavior, unchanged). `device_count`
    /// comes from the driver so a 2-rank run on a 1-GPU box binds both
    /// ranks to device 0 (useful for CPU-collective + GPU-compute testing;
    /// NCCL itself decides whether it accepts that topology).
    pub(crate) unsafe fn select_device_ordinal() -> i32 {
        if let Some(k) = std::env::var("NSL_CUDA_DEVICE")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
        {
            return k.max(0);
        }
        // Review M5: stripe ONLY under the SPMD spawner protocol (shm path
        // set). The M41 disaggregated-inference spawner also sets
        // NSL_LOCAL_RANK, and its prefill/decode workers must keep the
        // historical device-0 binding (their ranks are per-ROLE, not a
        // device clique).
        if std::env::var("NSL_TP_SHM_PATH").is_err() {
            return 0;
        }
        let Some(rank) = std::env::var("NSL_LOCAL_RANK")
            .ok()
            .and_then(|v| v.parse::<i32>().ok())
        else {
            return 0;
        };
        let mut count: i32 = 0;
        if cuDeviceGetCount(&mut count) != CUresult::CUDA_SUCCESS || count <= 0 {
            return 0;
        }
        let ordinal = rank.max(0) % count;
        if ordinal != 0 {
            eprintln!("[nsl] rank {rank}: binding CUDA device {ordinal} (of {count})");
        }
        ordinal
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
                let ordinal = select_device_ordinal();
                let mut device: CUdevice = 0;
                let result = cuDeviceGet(&mut device, ordinal);
                assert_eq!(
                    result,
                    CUresult::CUDA_SUCCESS,
                    "cuDeviceGet (ordinal {ordinal}) failed: {:?}",
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
                    func_cache: HashMap::new(),
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
        // cuda-graphs: physically unmapping cached blocks while a captured
        // region is mid-replay would leave the pending graph launch pointing
        // at freed memory — taint first (repairs the skipped prefix and
        // downgrades the region to eager, so no graph launch follows).
        super::graph_capture::taint("allocator pool drain");
        // Flush stream-ordered deferred frees first: `pool_drain` is the
        // "reclaim everything reclaimable" path (OOM recovery / explicit cache
        // empty), so raw frees still waiting on a completion event are forced
        // to complete and physically returned to the driver here too. Runs
        // before the allocator lock is taken (it frees via `free_device`,
        // which itself locks the allocator).
        drain_all_deferred_frees();
        ensure_context();
        let mut alloc = super::caching_allocator::CACHING_ALLOCATOR.lock().unwrap();
        alloc.drain_all()
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
        // P0.1: attribute the VRAM that caused this OOM. try_lock — the
        // caller dropped the allocator lock before panicking, but another
        // thread may hold it; a diagnostic must never deadlock or double-
        // panic on a poisoned lock.
        let allocator_report = match super::caching_allocator::CACHING_ALLOCATOR.try_lock() {
            Ok(alloc) => {
                let (p_bytes, p_segs, t_bytes, t_segs) = alloc.pool_breakdown();
                let mut r = format!(
                    "\n\nAllocator surfaces (current / peak / at-global-peak):\n{}\
                     Pools: persistent {} ({} segs), transient {} ({} segs)\n\
                     Top allocation contexts:\n",
                    alloc.surface_table_string("  "),
                    format_bytes(p_bytes), p_segs,
                    format_bytes(t_bytes), t_segs,
                );
                for (context, count, bytes) in
                    alloc.allocated_block_summary().into_iter().take(8)
                {
                    r.push_str(&format!(
                        "  {}: {} ({} blocks)\n",
                        context,
                        format_bytes(bytes),
                        count,
                    ));
                }
                r
            }
            Err(_) => {
                "\n\n(allocator lock unavailable — no surface/context breakdown)\n"
                    .to_string()
            }
        };
        // A1: async and direct-device allocations are now routed through the
        // unified surface/pool accounting, so the tables below cover them —
        // no more "untracked" caveat.
        let async_note = "";
        format!(
            "[nsl] GPU out of memory\n\
             \n\
               Requested:    {} ({})\n\
               VRAM free:    {} / {}\n\
               Pool drained: {}\n\
               Allocation #: {}{}{}{}\n\
             Suggestions:\n\
               - Reduce batch size or sequence length\n\
               - Enable gradient checkpointing (@checkpoint)\n\
               - Use lower precision (fp16, fp8, int8)\n\
               - Set NSL_ASYNC_ALLOC=1 for stream-ordered allocation",
            requested, format_bytes(requested),
            format_bytes(free_vram), format_bytes(total_vram),
            format_bytes(pool_freed),
            alloc_num, op_line,
            allocator_report,
            async_note,
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
            // A1: route async allocations through the unified accounting so
            // their bytes reach the surface counters, the global peak, and
            // the allocation-count gates — they used to be invisible.
            {
                let meta = super::caching_allocator::current_alloc_metadata(
                    size_bytes,
                    super::caching_allocator::AllocationLifetime::Async,
                );
                super::caching_allocator::CACHING_ALLOCATOR
                    .lock()
                    .unwrap()
                    .record_external_alloc(cptr, meta);
            }
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
        let _hp = crate::host_profile::Timer::start(crate::host_profile::Probe::DeviceAlloc);
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
        // The sync above completed every deferred-free event, so their raw
        // buffers can now be physically returned to the driver before retry.
        drain_completed_frees();
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
    // Stream-ordered async allocation (cudaMallocAsync fallback)
    // Opt-in via NSL_ASYNC_ALLOC=1. Uses CUDA driver's built-in memory
    // pool to avoid device-wide synchronization on allocation.
    // ------------------------------------------------------------------

    static ASYNC_ALLOC_RESULT: OnceLock<bool> = OnceLock::new();

    /// Check if async allocation is enabled and supported.
    /// Uses OnceLock to avoid TOCTOU race on initialization.
    /// pub(crate): the memory reports (memstats summary, nsl_debug_gpu_mem)
    /// note that async allocations bypass the caching allocator's
    /// surface/pool accounting.
    pub(crate) fn async_alloc_enabled() -> bool {
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
        // A1: decrement the unified accounting for this async allocation.
        super::caching_allocator::CACHING_ALLOCATOR
            .lock()
            .unwrap()
            .record_external_free(ptr);
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

    /// A1: attribute a direct `cuMemAlloc_v2` region (slab, paged KV,
    /// attention workspace) to the unified accounting, returning `ptr`
    /// unchanged. These allocations bypass the caching pool but are still
    /// device VRAM, so without this they were invisible to the surface
    /// counters and the peak report.
    fn account_direct_device(ptr: *mut c_void, size_bytes: usize) -> *mut c_void {
        if !ptr.is_null() {
            let meta = super::caching_allocator::current_alloc_metadata(
                size_bytes,
                super::caching_allocator::AllocationLifetime::DirectDevice,
            );
            super::caching_allocator::CACHING_ALLOCATOR
                .lock()
                .unwrap()
                .record_external_alloc(ptr, meta);
        }
        ptr
    }

    /// Allocate device-only memory (not accessible from host without explicit copy).
    /// On OOM, attempts sync + pool drain recovery before panicking.
    pub(crate) fn alloc_device(size_bytes: usize) -> *mut c_void {
        ensure_context();
        unsafe {
            let mut ptr: CUdeviceptr = 0;
            let result = cuMemAlloc_v2(&mut ptr, size_bytes);
            if result == CUresult::CUDA_SUCCESS && ptr != 0 {
                return account_direct_device(ptr as *mut c_void, size_bytes);
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
            // Reclaim event-deferred raw frees (now complete after the sync).
            drain_completed_frees();
            ptr = 0;
            let result = cuMemAlloc_v2(&mut ptr, size_bytes);
            if result == CUresult::CUDA_SUCCESS && ptr != 0 {
                return account_direct_device(ptr as *mut c_void, size_bytes);
            }

            // Drain pool and retry unconditionally: even if drain freed 0 bytes,
            // cuCtxSynchronize above may have completed async frees the caching
            // allocator can now use.
            let pool_freed = pool_drain();
            ptr = 0;
            let result = cuMemAlloc_v2(&mut ptr, size_bytes);
            if result == CUresult::CUDA_SUCCESS && ptr != 0 {
                return account_direct_device(ptr as *mut c_void, size_bytes);
            }

            let n = ALLOC_COUNT_DBG.load(std::sync::atomic::Ordering::Relaxed);
            panic!("{}", oom_diagnostic(size_bytes, n, pool_freed));
        }
    }

    /// Free device-only memory allocated with `alloc_device`.
    pub(crate) fn free_device(ptr: *mut c_void) {
        let _hp = crate::host_profile::Timer::start(crate::host_profile::Probe::DeviceFree);
        // A1: decrement the unified accounting for this direct allocation.
        super::caching_allocator::CACHING_ALLOCATOR
            .lock()
            .unwrap()
            .record_external_free(ptr);
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

    // ------------------------------------------------------------------
    // Stream-ordered deferred free (Milestone C · p3-remainder)
    //
    // A raw `alloc_device` buffer (CSHA backward activations, Tier B.1
    // x-scratch) must be physically returned to the driver via
    // `cuMemFree_v2` — routing it through the caching allocator would keep
    // it pooled and defeat the memory-reduction campaign's VRAM profile. But
    // a raw `cuMemFree` is NOT stream-ordered: it must not run until the
    // kernels reading the buffer have finished. The previous code guaranteed
    // that with a blocking `cuCtxSynchronize` before every free — a device-
    // wide host stall on the hot path.
    //
    // This replaces the host barrier with a CUDA event. `defer_free_device`
    // records an event on the NULL stream (the one stream every kernel
    // launches on) AFTER the consuming kernels, enqueues (ptrs, event), and
    // `drain_completed_frees` physically frees once the event is observed
    // complete. NULL-stream ordering guarantees the event cannot complete
    // before those kernels do, so the free stays safe with no CPU stall.
    //
    // `NSL_CUDA_SYNC=1` restores the eager sync-then-free behavior — the same
    // bisection kill-switch used across p3: if a result changes with it off
    // but matches with it on, a genuine use-after-free was exposed.
    // ------------------------------------------------------------------

    struct DeferredFree {
        /// Buffers sharing one lifetime (all consumed by the same preceding
        /// kernels), guarded by a single completion event.
        ptrs: Vec<usize>,
        event: CUevent,
    }
    // SAFETY: `CUevent` is an opaque driver handle (raw pointer) that is never
    // dereferenced on the Rust side; every driver call that touches it first
    // re-establishes the shared primary context via `ensure_context`. Moving
    // the handle between threads (the queue is a global `static`) is therefore
    // sound.
    unsafe impl Send for DeferredFree {}

    static DEFERRED_FREES: std::sync::LazyLock<Mutex<std::collections::VecDeque<DeferredFree>>> =
        std::sync::LazyLock::new(|| Mutex::new(std::collections::VecDeque::new()));
    /// Recycled disable-timing events, to avoid create/destroy churn.
    static FREE_EVENT_POOL: std::sync::LazyLock<Mutex<Vec<usize>>> =
        std::sync::LazyLock::new(|| Mutex::new(Vec::new()));

    fn acquire_free_event() -> CUevent {
        if let Some(ev) = FREE_EVENT_POOL.lock().unwrap().pop() {
            return ev as CUevent;
        }
        let mut ev: CUevent = std::ptr::null_mut();
        // 0x2 = CU_EVENT_DISABLE_TIMING — cheapest event; we only poll completion.
        let r = unsafe { cuEventCreate(&mut ev, 0x2) };
        assert_eq!(
            r,
            CUresult::CUDA_SUCCESS,
            "cuEventCreate (deferred-free) failed: {:?}",
            r
        );
        ev
    }

    fn recycle_free_event(ev: CUevent) {
        FREE_EVENT_POOL.lock().unwrap().push(ev as usize);
    }

    /// Number of buffers currently awaiting a deferred physical free.
    /// Exposed for tests and the memory diagnostics.
    pub(crate) fn deferred_free_pending() -> usize {
        DEFERRED_FREES
            .lock()
            .unwrap()
            .iter()
            .map(|d| d.ptrs.len())
            .sum()
    }

    /// Stream-ordered deferred free of a single raw `alloc_device` buffer.
    pub(crate) fn defer_free_device(ptr: *mut c_void) {
        if ptr.is_null() {
            return;
        }
        defer_free_device_batch(&[ptr]);
    }

    /// Stream-ordered deferred free of one or more raw `alloc_device` buffers
    /// that share a lifetime (all consumed by the same preceding kernels).
    /// A single event guards the whole group.
    pub(crate) fn defer_free_device_batch(ptrs: &[*mut c_void]) {
        // cuda-graphs: inside a region the completion event must NOT be
        // recorded yet — during replay the consuming kernels are not on the
        // stream until the region-end graph launch, so an event recorded now
        // would complete early and free live memory. Queue host-side; the
        // region end records through `defer_free_device_record`.
        if super::graph_capture::in_region() {
            for p in ptrs.iter().filter(|p| !p.is_null()) {
                super::graph_capture::queue_deferred_free(*p);
            }
            return;
        }
        defer_free_device_batch_record(ptrs);
    }

    /// Region-end flush entry: the original (event-recording) deferred-free
    /// path, called once the region's work is actually on the stream.
    pub(crate) fn defer_free_device_record(ptr: *mut c_void) {
        if ptr.is_null() {
            return;
        }
        defer_free_device_batch_record(&[ptr]);
    }

    fn defer_free_device_batch_record(ptrs: &[*mut c_void]) {
        let live: Vec<usize> = ptrs
            .iter()
            .filter(|p| !p.is_null())
            .map(|p| *p as usize)
            .collect();
        if live.is_empty() {
            return;
        }
        ensure_context();
        if sync_mode_enabled() {
            // Eager kill-switch (NSL_CUDA_SYNC=1): the old sync-then-free path.
            unsafe { cuCtxSynchronize(); }
            for p in live {
                free_device(p as *mut c_void);
            }
            return;
        }
        let event = acquire_free_event();
        // Record on the NULL stream: completes only after every previously
        // launched NULL-stream kernel (including this buffer's consumers).
        let rc = unsafe { cuEventRecord(event, current_stream()) };
        if rc != CUresult::CUDA_SUCCESS {
            // Recording failed — fall back to a safe synchronous free rather
            // than risk a use-after-free.
            recycle_free_event(event);
            unsafe { cuCtxSynchronize(); }
            for p in live {
                free_device(p as *mut c_void);
            }
            return;
        }
        DEFERRED_FREES
            .lock()
            .unwrap()
            .push_back(DeferredFree { ptrs: live, event });
        drain_completed_frees();
    }

    /// Poll the deferred-free queue and physically free every entry whose
    /// event has completed. Each event is queried independently, so this is
    /// correct regardless of the order entries were enqueued vs. recorded.
    ///
    /// Precondition: a CUDA context must be current on the calling thread (for
    /// `cuEventQuery`). Every in-tree caller ensures this — `defer_free_device*`
    /// and both OOM-recovery sites call `ensure_context` first — so this hot
    /// path does not re-acquire the CUDA-state lock. Without a current context
    /// `cuEventQuery` returns an error, treated as "not ready" below, which only
    /// defers reclaim (never a use-after-free).
    pub(crate) fn drain_completed_frees() {
        // cuda-graphs: `cuEventQuery` is illegal while the compute stream is
        // capturing, and during replay the polled work may not be issued yet
        // — skip the opportunistic poll inside a region (the region end and
        // every out-of-region defer poll again).
        if super::graph_capture::in_region() {
            return;
        }
        // Collect completed entries under the lock, then free outside it:
        // `free_device` takes the CACHING_ALLOCATOR lock, and holding
        // DEFERRED_FREES across that call would nest two locks.
        let mut ready: Vec<DeferredFree> = Vec::new();
        {
            let mut q = DEFERRED_FREES.lock().unwrap();
            let mut i = 0;
            while i < q.len() {
                // Only CUDA_SUCCESS means "done". CUDA_ERROR_NOT_READY (and, in a
                // faulted context, a sticky error) keeps the entry queued —
                // `drain_all_deferred_frees`'s hard wait reclaims it later.
                if unsafe { cuEventQuery(q[i].event) } == CUresult::CUDA_SUCCESS {
                    ready.push(q.remove(i).unwrap());
                } else {
                    i += 1;
                }
            }
        }
        for entry in ready {
            recycle_free_event(entry.event);
            for p in entry.ptrs {
                free_device(p as *mut c_void);
            }
        }
    }

    /// Force every currently-pending deferred free to complete, then physically
    /// free it. Called from `pool_drain` (OOM recovery / explicit cache empty)
    /// and available for shutdown / test determinism.
    ///
    /// Snapshots the queue FIRST, then waits on each snapshotted entry's own
    /// completion event — deliberately NOT a single global `cuCtxSynchronize`
    /// followed by an unconditional drain. A global-sync-then-sweep has a TOCTOU
    /// hole: another thread could enqueue a deferred free (its event recorded
    /// AFTER our sync) in the window before the drain, and we would then free
    /// its buffer without ever waiting for its consuming kernel — a
    /// use-after-free. Per-entry `cuEventSynchronize` on the snapshot closes that
    /// window; entries enqueued after the snapshot simply stay queued for the
    /// next drain.
    pub(crate) fn drain_all_deferred_frees() {
        let drained: Vec<DeferredFree> = {
            let mut q = DEFERRED_FREES.lock().unwrap();
            if q.is_empty() {
                return;
            }
            q.drain(..).collect()
        };
        ensure_context();
        for entry in &drained {
            // The event was recorded after this buffer's consumers on the NULL
            // stream, so waiting on it guarantees those kernels have retired —
            // the physical free below cannot race them.
            unsafe { cuEventSynchronize(entry.event); }
        }
        for entry in drained {
            recycle_free_event(entry.event);
            for p in entry.ptrs {
                free_device(p as *mut c_void);
            }
        }
    }

    // ------------------------------------------------------------------
    // Pinned (page-locked) host memory — optimizer-state offload (P0.2)
    // ------------------------------------------------------------------

    /// Registry of live pinned host buffers (data pointers). The tensor
    /// free path (`nsl_tensor_free`) consults this set to route a host
    /// tensor's data buffer to `cuMemFreeHost` instead of the Rust heap
    /// free — pinned buffers are driver allocations and MUST NOT go
    /// through `std::alloc::dealloc`.
    static PINNED_HOST_SET: std::sync::LazyLock<std::sync::Mutex<HashSet<usize>>> =
        std::sync::LazyLock::new(|| std::sync::Mutex::new(HashSet::new()));

    /// Live pinned-buffer count — lock-free fast path for `is_pinned`,
    /// which sits on the free path of EVERY host tensor. Runs that never
    /// pin (no offload) skip the registry mutex entirely.
    static PINNED_LIVE_COUNT: std::sync::atomic::AtomicUsize =
        std::sync::atomic::AtomicUsize::new(0);

    /// Is `ptr` a live pinned host buffer allocated by `alloc_pinned`?
    pub(crate) fn is_pinned(ptr: *mut c_void) -> bool {
        if ptr.is_null() { return false; }
        if PINNED_LIVE_COUNT.load(std::sync::atomic::Ordering::Acquire) == 0 {
            return false;
        }
        PINNED_HOST_SET.lock().unwrap().contains(&(ptr as usize))
    }

    /// Try to allocate pinned (page-locked) host memory via
    /// `cuMemAllocHost_v2`; `None` on driver failure (page-lock limits /
    /// fragmented host memory) so callers can degrade to pageable instead
    /// of aborting a multi-GB training run.
    ///
    /// Used by the optimizer-state offload envelope (P0.2): pinned staging
    /// buffers let `cuMemcpyDtoHAsync` run as true async DMA overlap
    /// instead of the pageable double-buffer bounce. The pointer is
    /// tracked in `PINNED_HOST_SET` so the tensor free path can route it
    /// back to `cuMemFreeHost`.
    pub(crate) fn try_alloc_pinned(size_bytes: usize) -> Option<*mut c_void> {
        ensure_context();
        unsafe {
            let mut ptr: *mut c_void = std::ptr::null_mut();
            let result = cuMemAllocHost_v2(&mut ptr, size_bytes);
            if result != CUresult::CUDA_SUCCESS || ptr.is_null() {
                return None;
            }
            PINNED_HOST_SET.lock().unwrap().insert(ptr as usize);
            PINNED_LIVE_COUNT.fetch_add(1, std::sync::atomic::Ordering::Release);
            Some(ptr)
        }
    }

    /// Panicking wrapper over [`try_alloc_pinned`] for callers that treat
    /// pinned allocation failure as a bug (tests, small transfer buffers).
    /// Production state buffers go through `try_alloc_pinned` + pageable
    /// fallback instead — only the unit tests call this today.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn alloc_pinned(size_bytes: usize) -> *mut c_void {
        try_alloc_pinned(size_bytes).unwrap_or_else(|| {
            panic!("cuMemAllocHost_v2({} bytes) failed", size_bytes)
        })
    }

    /// Free pinned host memory allocated with `alloc_pinned`.
    pub(crate) fn free_pinned(ptr: *mut c_void) {
        ensure_context();
        if PINNED_HOST_SET.lock().unwrap().remove(&(ptr as usize)) {
            PINNED_LIVE_COUNT.fetch_sub(1, std::sync::atomic::Ordering::Release);
        }
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
        if size_bytes >= 262144 && crate::host_profile::enabled() {
            eprintln!("[copy] H2D {:>9} KB  ctx={}", size_bytes / 1024, current_oom_context());
        }
        let _hp = crate::host_profile::Timer::start(crate::host_profile::Probe::Memcpy);
        ensure_context();
        // cuda-graphs: uploads inside a region are pseudo-ops — the payload
        // flows through a graph-owned pinned staging buffer (see `on_htod`).
        if !super::graph_capture::on_htod(dst_device, src_host, size_bytes) {
            return;
        }
        unsafe {
            let result = cuMemcpyHtoD_v2(dst_device as CUdeviceptr, src_host, size_bytes);
            // Build the context string only on failure: this ran on every
            // successful copy, and there are ~1300 copies per training step.
            if result != CUresult::CUDA_SUCCESS {
                let ctx = current_oom_context();
                let ctx_suffix = if ctx.is_empty() {
                    String::new()
                } else {
                    format!(" [context: {}]", ctx)
                };
                panic!(
                    "cuMemcpyHtoD_v2({} bytes) failed: {:?}{}",
                    size_bytes, result, ctx_suffix
                );
            }
        }
    }

    /// Copy `size_bytes` bytes from host to device NOW, bypassing cuda-graph
    /// capture.
    ///
    /// `memcpy_htod` hands the payload to `graph_capture::on_htod`, which inside
    /// a capture region stages it into a graph node — the bytes reach the device
    /// at graph *launch*, not at the call. That is right for data the graph
    /// replays, and wrong for a one-shot upload whose device pointer is then
    /// cached and reused: the cache would publish a pointer to memory that was
    /// never written, and skipping the upload on later steps would change the
    /// recorded op sequence between the capture step and the replay steps,
    /// diverging the capture instead of accelerating it.
    ///
    /// Only for immutable, content-addressed uploads (shape and stride vectors),
    /// which are idempotent and must be visible before the kernel that reads
    /// them is recorded.
    pub(crate) fn memcpy_htod_immediate(
        dst_device: *mut c_void,
        src_host: *const c_void,
        size_bytes: usize,
    ) {
        let _hp = crate::host_profile::Timer::start(crate::host_profile::Probe::Memcpy);
        ensure_context();
        unsafe {
            let result = cuMemcpyHtoD_v2(dst_device as CUdeviceptr, src_host, size_bytes);
            if result != CUresult::CUDA_SUCCESS {
                panic!(
                    "cuMemcpyHtoD_v2({} bytes, immediate) failed: {:?}",
                    size_bytes, result
                );
            }
        }
    }

    /// Copy `size_bytes` bytes from device memory to host memory.
    pub(crate) fn memcpy_dtoh(dst_host: *mut c_void, src_device: *const c_void, size_bytes: usize) {
        if size_bytes >= 262144 && crate::host_profile::enabled() {
            eprintln!("[copy] D2H {:>9} KB  ctx={}", size_bytes / 1024, current_oom_context());
        }
        let _hp = crate::host_profile::Timer::start(crate::host_profile::Probe::Memcpy);
        super::graph_capture::taint("sync DtoH readback");
        ensure_context();
        unsafe {
            let result = cuMemcpyDtoH_v2(dst_host, src_device as CUdeviceptr, size_bytes);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemcpyDtoH_v2({} bytes) failed: {:?}",
                size_bytes,
                result,
            );
        }
    }

    /// Copy `size_bytes` bytes from one device pointer to another.
    pub(crate) fn memcpy_dtod(dst_device: *mut c_void, src_device: *const c_void, size_bytes: usize) {
        ensure_context();
        // cuda-graphs: device-to-device copies are pseudo-ops (no host data).
        if !super::graph_capture::on_dtod(dst_device, src_device, size_bytes) {
            return;
        }
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

    // ------------------------------------------------------------------
    // Per-thread transfer stream — optimizer-state offload (P0.2)
    //
    // Kernel launches go to the per-thread compute stream (see
    // `current_stream`; legacy NULL under the kill-switch), so a
    // NON_BLOCKING side stream lets the offload copy-back overlap
    // with the next parameter's update kernels: NULL-stream work never
    // waits on the transfer stream. Correctness ordering (the copy must
    // not start before the update kernels that produced its source have
    // finished) is enforced per copy with a NULL-stream event that the
    // transfer stream waits on.
    //
    // CUDA contexts are THREAD-LOCAL in this runtime, so the stream is
    // thread-local too (mirrors `inspect/stream.rs`). Env kill-switches
    // (documented at their read sites in `tensor/mod.rs`):
    //   NSL_OFFLOAD_SYNC=1     — force synchronous copy-back
    //   NSL_OFFLOAD_PAGEABLE=1 — force pageable host state buffers
    // ------------------------------------------------------------------

    thread_local! {
        // CUstream stored as usize (raw handles are !Send; 0 = not created).
        static TRANSFER_STREAM: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }

    /// Lazily create (once per thread) and return the offload transfer
    /// stream. Must only be called when a CUDA context exists (it calls
    /// `ensure_context`, which force-initializes CUDA).
    pub(crate) fn transfer_stream() -> CUstream {
        TRANSFER_STREAM.with(|s| {
            let cur = s.get();
            if cur != 0 {
                return cur as CUstream;
            }
            ensure_context();
            let mut stream: CUstream = std::ptr::null_mut();
            unsafe {
                // 0x1 = CU_STREAM_NON_BLOCKING: no implicit synchronization
                // with the legacy NULL stream (ordering is via events).
                let result = cuStreamCreate(&mut stream, 0x1);
                assert_eq!(
                    result,
                    CUresult::CUDA_SUCCESS,
                    "cuStreamCreate (offload transfer stream) failed: {:?}",
                    result
                );
            }
            s.set(stream as usize);
            stream
        })
    }

    /// Make the calling thread's transfer stream wait for all previously
    /// launched compute work (record + wait via a throwaway event;
    /// `cuEventDestroy` defers actual destruction until the wait completes,
    /// so destroying immediately is safe).
    ///
    /// p8 PR-A: the event is recorded on `current_stream()` — the compute
    /// stream that now carries every kernel (or the NULL stream under the
    /// legacy kill-switch). Blocking-stream semantics mean work recorded
    /// there is also ordered after any interleaved NULL-stream memcpys, so
    /// this wait covers everything the old NULL-stream record covered.
    unsafe fn transfer_stream_wait_null_stream(stream: CUstream) {
        super::graph_capture::taint("transfer-stream event wait");
        let mut ev: CUevent = std::ptr::null_mut();
        // 0x2 = CU_EVENT_DISABLE_TIMING (cheapest event flavor).
        let r = cuEventCreate(&mut ev, 0x2);
        assert_eq!(r, CUresult::CUDA_SUCCESS, "cuEventCreate (offload) failed: {:?}", r);
        let r = cuEventRecord(ev, current_stream());
        assert_eq!(r, CUresult::CUDA_SUCCESS, "cuEventRecord (offload) failed: {:?}", r);
        let r = cuStreamWaitEvent(stream, ev, 0);
        assert_eq!(r, CUresult::CUDA_SUCCESS, "cuStreamWaitEvent (offload) failed: {:?}", r);
        cuEventDestroy_v2(ev);
    }

    /// Async DtoH on the per-thread transfer stream, ordered AFTER all
    /// previously-launched NULL-stream work (the update kernels that
    /// produced `src_device`). `dst_host` MUST be pinned — the caller
    /// checks `is_pinned` first; an async copy into pageable memory
    /// silently degrades to a staged sync copy inside the driver.
    ///
    /// The copy is NOT complete when this returns: the caller must keep
    /// `src_device` alive until `transfer_stream_synchronize()` (the
    /// offload drain list in `tensor/mod.rs` owns that deferral).
    pub(crate) fn memcpy_dtoh_async(dst_host: *mut c_void, src_device: *const c_void, size_bytes: usize) {
        ensure_context();
        let stream = transfer_stream();
        unsafe {
            transfer_stream_wait_null_stream(stream);
            let result = cuMemcpyDtoHAsync_v2(dst_host, src_device as CUdeviceptr, size_bytes, stream);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemcpyDtoHAsync_v2({} bytes) failed: {:?}",
                size_bytes,
                result
            );
        }
    }

    /// Async HtoD on the per-thread transfer stream (P0.2 item 3; the HtoD
    /// prefetch consumer is a deferred follow-up — no production caller
    /// yet). `src_host` should be pinned for true async DMA. NULL-stream
    /// consumers of `dst_device` are NOT ordered after this copy (the
    /// stream is non-blocking): the caller MUST call
    /// `transfer_stream_synchronize()` before launching kernels that read
    /// `dst_device`.
    ///
    /// ALLOCATOR INVARIANT (load-bearing for the offload envelope's inline
    /// frees, e.g. `nsl_tensor_cast_from_host`'s transient): recycled
    /// caching-allocator blocks may only be RE-WRITTEN by NULL-stream-
    /// ordered work. This function is the one primitive that writes device
    /// memory on the non-blocking transfer stream — a future caller must
    /// only target buffers that were never freed-and-recycled with pending
    /// NULL-stream readers, or must drain before reuse.
    #[allow(dead_code)]
    pub(crate) fn memcpy_htod_async(dst_device: *mut c_void, src_host: *const c_void, size_bytes: usize) {
        ensure_context();
        let stream = transfer_stream();
        unsafe {
            transfer_stream_wait_null_stream(stream);
            let result = cuMemcpyHtoDAsync_v2(dst_device as CUdeviceptr, src_host, size_bytes, stream);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemcpyHtoDAsync_v2({} bytes) failed: {:?}",
                size_bytes,
                result
            );
        }
    }

    /// Item 11: issue an HtoD PREFETCH on the transfer stream that does NOT
    /// wait on prior compute (unlike `memcpy_htod_async`, whose copy is
    /// ordered after the compute that produced its source). A prefetch's
    /// source is a stable pinned host buffer already filled on the host, so
    /// the copy can start immediately and run CONCURRENTLY with compute. The
    /// returned event is recorded on the transfer stream right after the copy;
    /// the consumer discharges it with `compute_stream_wait_event` before any
    /// kernel reads `dst_device`. `src_host` must be pinned for true async DMA.
    ///
    /// CADENCE assume/guarantee: ASSUME (1) `dst_device` has no other pending
    /// writer and (2) `src_host` is not re-written until this copy completes —
    /// both hold because the caller only prefetches into an arena slot freed by
    /// a SYNCHRONOUS writeback evict, whose DtoH drains this HtoD before the
    /// slot's `dst_device`/`src_host` (host_stage) can be reused (see
    /// `weight_stream::arena_acquire`'s LOAD-BEARING note). GUARANTEE the
    /// returned event, waited on the compute stream, orders this copy before
    /// every later compute-stream read of `dst_device`.
    #[must_use]
    pub(crate) fn prefetch_htod_on_transfer(
        dst_device: *mut c_void,
        src_host: *const c_void,
        size_bytes: usize,
    ) -> u64 {
        ensure_context();
        let stream = transfer_stream();
        unsafe {
            let result =
                cuMemcpyHtoDAsync_v2(dst_device as CUdeviceptr, src_host, size_bytes, stream);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemcpyHtoDAsync_v2 (prefetch, {} bytes) failed: {:?}",
                size_bytes,
                result
            );
            let mut ev: CUevent = std::ptr::null_mut();
            // 0x2 = CU_EVENT_DISABLE_TIMING (cheapest).
            let r = cuEventCreate(&mut ev, 0x2);
            assert_eq!(r, CUresult::CUDA_SUCCESS, "cuEventCreate (prefetch) failed: {:?}", r);
            let r = cuEventRecord(ev, stream);
            assert_eq!(r, CUresult::CUDA_SUCCESS, "cuEventRecord (prefetch) failed: {:?}", r);
            ev as u64
        }
    }

    /// Item 11 (writeback half): issue a DtoH WRITEBACK on the transfer
    /// stream, ordered AFTER all previously-launched compute (the per-layer
    /// update kernels that produced `src_device`), and return a completion
    /// event recorded right after the copy. Unlike the synchronous
    /// `memcpy_dtoh`, this returns immediately — the next layer's compute
    /// proceeds while the evicted pack drains to the host.
    ///
    /// CADENCE assume/guarantee: ASSUME (1) `src_device` (the arena slot) is
    /// not re-written until this copy completes and (2) `dst_host` (the
    /// slot's pinned stage) is not read or re-written until then — both hold
    /// because the slot is marked writeback-pending and `arena_acquire`
    /// refuses to reuse it until the caller drains the returned event (see
    /// `weight_stream::drain_pending_writebacks`). GUARANTEE: once the event
    /// is host-synchronized, `dst_host` holds the post-update bytes and the
    /// mirror scatter may proceed on the CPU.
    #[must_use]
    pub(crate) fn writeback_dtoh_on_transfer(
        dst_host: *mut c_void,
        src_device: *const c_void,
        size_bytes: usize,
    ) -> u64 {
        ensure_context();
        let stream = transfer_stream();
        unsafe {
            // Order the copy after the update kernels on the compute stream.
            transfer_stream_wait_null_stream(stream);
            let result =
                cuMemcpyDtoHAsync_v2(dst_host, src_device as CUdeviceptr, size_bytes, stream);
            assert_eq!(
                result,
                CUresult::CUDA_SUCCESS,
                "cuMemcpyDtoHAsync_v2 (writeback, {} bytes) failed: {:?}",
                size_bytes,
                result
            );
            let mut ev: CUevent = std::ptr::null_mut();
            // 0x2 = CU_EVENT_DISABLE_TIMING (cheapest).
            let r = cuEventCreate(&mut ev, 0x2);
            assert_eq!(r, CUresult::CUDA_SUCCESS, "cuEventCreate (writeback) failed: {:?}", r);
            let r = cuEventRecord(ev, stream);
            assert_eq!(r, CUresult::CUDA_SUCCESS, "cuEventRecord (writeback) failed: {:?}", r);
            ev as u64
        }
    }

    /// Host-block until `ev` completes, then destroy it. The drain step of an
    /// async writeback: after this returns the DtoH has landed in the pinned
    /// stage and the CPU may scatter it into the per-param mirrors.
    pub(crate) fn event_synchronize(ev: u64) {
        if ev == 0 {
            return;
        }
        ensure_context();
        unsafe {
            let r = cuEventSynchronize(ev as CUevent);
            assert_eq!(
                r,
                CUresult::CUDA_SUCCESS,
                "cuEventSynchronize (writeback drain) failed: {:?}",
                r
            );
            cuEventDestroy_v2(ev as CUevent);
        }
    }

    /// Item 11: make the COMPUTE stream wait for a prefetch event, then
    /// destroy it. After this returns, every kernel launched on the compute
    /// stream is ordered after the prefetch HtoD — the guarantee half of the
    /// transfer certificate. Destroying immediately is safe: `cuEventDestroy`
    /// defers the actual free until the recorded work and the wait complete.
    pub(crate) fn compute_stream_wait_event(ev: u64) {
        if ev == 0 {
            return;
        }
        ensure_context();
        unsafe {
            let r = cuStreamWaitEvent(current_stream(), ev as CUevent, 0);
            assert_eq!(
                r,
                CUresult::CUDA_SUCCESS,
                "cuStreamWaitEvent (prefetch await) failed: {:?}",
                r
            );
            cuEventDestroy_v2(ev as CUevent);
        }
    }

    /// P4 item 14: watchdog-bounded synchronization of the calling thread's
    /// COMPUTE stream. Polls `cuStreamQuery` until the stream drains or
    /// `timeout_secs` elapses — the per-collective watchdog for NCCL ops
    /// (a dead peer leaves the collective enqueued forever; a bounded wait
    /// turns that hang into a loud symmetric abort). Returns true on drain,
    /// false on timeout. Any error other than NOT_READY aborts (the stream
    /// carries training work — a poisoned stream must not train on).
    pub(crate) fn sync_compute_stream_with_deadline(timeout_secs: u64, what: &str) -> bool {
        ensure_context();
        let stream = current_stream();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
        loop {
            let rc = unsafe { cuStreamQuery(stream) };
            match rc {
                CUresult::CUDA_SUCCESS => return true,
                CUresult::CUDA_ERROR_NOT_READY => {
                    if std::time::Instant::now() >= deadline {
                        return false;
                    }
                    std::thread::sleep(std::time::Duration::from_millis(1));
                }
                other => {
                    eprintln!(
                        "nsl: {what}: cuStreamQuery failed with {other:?} while waiting \
                         on a collective — aborting"
                    );
                    std::process::abort();
                }
            }
        }
    }

    /// Synchronize the calling thread's transfer stream. No-op when the
    /// stream was never created on this thread (does NOT force-initialize
    /// CUDA — safe to call unconditionally from the offload drain).
    pub(crate) fn transfer_stream_synchronize() {
        super::graph_capture::taint("transfer-stream synchronize");
        TRANSFER_STREAM.with(|s| {
            let cur = s.get();
            if cur == 0 {
                return;
            }
            ensure_context();
            unsafe {
                let result = cuStreamSynchronize(cur as CUstream);
                assert_eq!(
                    result,
                    CUresult::CUDA_SUCCESS,
                    "cuStreamSynchronize (offload transfer stream) failed: {:?}",
                    result
                );
            }
        })
    }

    /// Zero-fill device memory. Graph-aware: recorded as a pseudo-op inside
    /// a capture region (async on the compute stream so the driver records a
    /// memset node — ordering-neutral vs the sync form thanks to
    /// blocking-stream semantics), verified/skipped during replay.
    pub(crate) fn memset_d8(device_ptr: *mut c_void, size_bytes: usize) {
        ensure_context();
        match super::graph_capture::on_memset(device_ptr as usize, size_bytes) {
            super::graph_capture::MemsetAction::Skip => return,
            super::graph_capture::MemsetAction::AsyncOnComputeStream => {
                unsafe {
                    let result = cuMemsetD8Async(
                        device_ptr as CUdeviceptr,
                        0,
                        size_bytes,
                        current_stream(),
                    );
                    assert_eq!(
                        result,
                        CUresult::CUDA_SUCCESS,
                        "cuMemsetD8Async({} bytes) failed: {:?}",
                        size_bytes,
                        result
                    );
                }
                return;
            }
            super::graph_capture::MemsetAction::Sync => {}
        }
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

    // ------------------------------------------------------------------
    // Per-thread COMPUTE stream — p8 PR-A stream migration
    //
    // All kernel launches (and cuBLAS, via a per-call cublasSetStream_v2)
    // now issue onto a dedicated per-thread stream created with
    // CU_STREAM_DEFAULT (flags=0, a BLOCKING stream). Blocking-stream
    // semantics make this migration ordering-neutral: the legacy NULL
    // stream is an implicit two-way barrier against every blocking
    // stream, so the runtime's synchronous memcpys (HtoD uploads, the
    // sync-DtoH-as-barrier policy at the top of this file) and any
    // remaining NULL-stream work interleave with compute work in exactly
    // the same total order as before. What changes: the compute stream
    // is a REAL stream, which CUDA graph capture (p8 PR-B) can capture —
    // the legacy NULL stream cannot be captured at all.
    //
    // Kill-switch: NSL_LEGACY_NULL_STREAM=1 restores the old NULL-stream
    // launches (read once, cached). The differential gates prove the two
    // modes bit-identical.
    // ------------------------------------------------------------------

    thread_local! {
        // CUstream stored as usize (raw handles are !Send; 0 = not created).
        static COMPUTE_STREAM: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
    }
    static LEGACY_NULL_STREAM: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

    /// Returns the CUstream that `kernel_launch` issues work onto: the
    /// per-thread blocking compute stream (lazily created), or the legacy
    /// NULL stream under `NSL_LEGACY_NULL_STREAM=1`. Events recorded on this
    /// stream correctly serialize with kernel execution rather than capturing
    /// host-submit latency.
    ///
    /// Also used by the Phase 2 kernel-timing profiler (see
    /// `crate::profiler::cuda_clock::CudaEventClock`) so that begin/end event
    /// recording uses the *same* stream as the launch, and by the deferred-free
    /// machinery for its completion events.
    ///
    /// Only call when CUDA work is (about to be) in flight — creating the
    /// stream force-initializes the context via `ensure_context`.
    ///
    /// SAME-THREAD CONTRACT (p8 PR-A): the compute stream is per-thread and
    /// two blocking streams do NOT synchronize with each other (only with the
    /// legacy NULL stream). Every consumer that records ordering events
    /// against "the work that touched this buffer" (`defer_free_device*`, the
    /// offload transfer-stream wait) must run on the SAME thread that
    /// launched that work. All in-tree launches are single-threaded today; a
    /// future multi-threaded dispatcher must add cross-stream events.
    pub fn current_stream() -> CUstream {
        if *LEGACY_NULL_STREAM
            .get_or_init(|| std::env::var("NSL_LEGACY_NULL_STREAM").ok().as_deref() == Some("1"))
        {
            return std::ptr::null_mut();
        }
        COMPUTE_STREAM.with(|s| {
            let cur = s.get();
            if cur != 0 {
                return cur as CUstream;
            }
            ensure_context();
            let mut stream: CUstream = std::ptr::null_mut();
            unsafe {
                // 0 = CU_STREAM_DEFAULT: a BLOCKING stream — implicit two-way
                // synchronization with the legacy NULL stream (load-bearing;
                // see the module comment above).
                let result = cuStreamCreate(&mut stream, 0);
                assert_eq!(
                    result,
                    CUresult::CUDA_SUCCESS,
                    "cuStreamCreate (compute stream) failed: {:?}",
                    result
                );
            }
            s.set(stream as usize);
            stream
        })
    }

    pub unsafe fn cu_event_synchronize_raw(event: u64) -> CUresult {
        cuEventSynchronize(event as CUevent)
    }

    pub unsafe fn cu_event_elapsed_time_raw(ms: *mut f32, start: u64, stop: u64) -> CUresult {
        cuEventElapsedTime_v2(ms, start as CUevent, stop as CUevent)
    }

    pub unsafe fn cu_event_create_checked() -> Result<u64, CUresult> {
        let mut e: CUevent = std::ptr::null_mut();
        let res = cuEventCreate(&mut e, 0);
        if res != CUresult::CUDA_SUCCESS { return Err(res); }
        Ok(e as u64)
    }

    pub unsafe fn cu_event_record_on_current_stream(event: u64) -> CUresult {
        super::graph_capture::taint("event record on compute stream");
        cuEventRecord(event as CUevent, current_stream())
    }

    /// Elapsed time between two recorded events, in milliseconds.
    ///
    /// Returns the driver status rather than discarding it.  It used to be
    /// discarded, and `cuEventElapsedTime_v2` leaves its out-param UNTOUCHED
    /// on failure — so every failed query silently became a 0.0 ms
    /// measurement.  That is how `--profile-kernels` came to write a
    /// `kernel_profile.json` reporting 412 launches and
    /// `total_kernel_time_ms: 0.0`: not a fast program, a dead API call
    /// nobody was checking.
    #[must_use]
    pub unsafe fn cu_event_elapsed_time(ms: *mut f32, start: u64, stop: u64) -> CUresult {
        cuEventElapsedTime_v2(ms, start as CUevent, stop as CUevent)
    }

    pub unsafe fn cu_event_destroy(event: u64) {
        cuEventDestroy_v2(event as CUevent);
    }

    pub unsafe fn cu_ctx_synchronize() {
        super::graph_capture::taint("explicit ctx synchronize");
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
        let _hp = crate::host_profile::Timer::start(crate::host_profile::Probe::KernelLaunch);
        let state = state();
        let func = {
            let mut guard = state.lock().unwrap();
            unsafe { cuCtxSetCurrent(guard.context); }

            // Cache modules by FNV-1a hash of PTX content.
            // Using pointer address as key (the old approach) caused
            // CUDA_ERROR_NOT_FOUND (rc=500) when a new PTX Vec was allocated
            // at the same heap address as a previously-freed one — the cache
            // returned the stale old module and the new kernel name was not found.
            let cache_key = {
                // Compute length by scanning for the NUL terminator.
                let mut len = 0usize;
                while unsafe { *ptx_ptr.add(len) } != 0 { len += 1; }
                let ptx_bytes = unsafe { std::slice::from_raw_parts(ptx_ptr, len) };
                // FNV-1a 64-bit hash (no external dep, no alloc).
                let mut h: u64 = 14695981039346656037u64;
                for &b in ptx_bytes {
                    h ^= b as u64;
                    h = h.wrapping_mul(1099511628211u64);
                }
                h
            };
            let module = if let Some(m) = guard.module_cache.get(&cache_key) {
                *m
            } else {
                let mut module: CUmodule = std::ptr::null_mut();
                let res = unsafe { cuModuleLoadData(&mut module, ptx_ptr as *const c_void) };
                if res != CUresult::CUDA_SUCCESS { return res; }
                guard.module_cache.insert(cache_key, module);
                module
            };

            // Resolve-once function cache: cuModuleGetFunction used to run
            // on EVERY launch (the CFIE fast path below existed precisely
            // to avoid that on the decode loop; training now gets the same
            // treatment). Keyed by (PTX content hash, name hash) — both
            // content-derived, so heap-address reuse cannot alias entries.
            let name_key = {
                let mut len = 0usize;
                while unsafe { *name_ptr.add(len) } != 0 { len += 1; }
                let name_bytes = unsafe { std::slice::from_raw_parts(name_ptr, len) };
                let mut h: u64 = 14695981039346656037u64;
                for &b in name_bytes {
                    h ^= b as u64;
                    h = h.wrapping_mul(1099511628211u64);
                }
                h
            };
            if let Some(f) = guard.func_cache.get(&(cache_key, name_key)) {
                *f
            } else {
                let name = unsafe { std::ffi::CStr::from_ptr(name_ptr as *const i8) };
                let mut func: CUfunction = std::ptr::null_mut();
                let res = unsafe { cuModuleGetFunction(&mut func, module, name.as_ptr()) };
                if res != CUresult::CUDA_SUCCESS { return res; }
                guard.func_cache.insert((cache_key, name_key), func);
                func
            }
        }; // guard dropped here — no lock held for CUDA calls

        // Profiler: pop event pair (lock-pop-unlock on profiler mutex)
        let profiler_events = if crate::kernel_profiler::kernel_profiler_enabled() {
            crate::kernel_profiler::kernel_profiler_pop_events()
        } else {
            None
        };

        // Record start event before launch
        if let Some((start, _, _)) = &profiler_events {
            unsafe { cuEventRecord(*start as CUevent, current_stream()); }
        }

        // Dynamic SMEM opt-in for kernels using `.extern .shared` PTX declarations.
        // The canonical dynamic-SMEM form `.extern .shared shmem[]` (empty brackets) does NOT
        // bake any size into the PTX — the CUDA driver allocates exactly the bytes we request
        // here via cuFuncSetAttribute.  We must call the attribute for ANY non-zero request:
        //
        //   * shared_mem_bytes <= 48 KB: the driver accepts values within the static cap
        //     without the attribute call, but calling it is harmless and ensures the per-block
        //     budget is explicitly set (guards against driver default being 0 for extern decls).
        //   * shared_mem_bytes > 48 KB: mandatory — without the call the driver silently uses
        //     the static default (48 KB) and the launch fails with CUDA_ERROR_INVALID_VALUE.
        //
        // Guard lowered from `> 48 KB` to `> 0` so that kernels in the 1..=48 KB range (e.g.
        // hd=32, SMEM=37.5 KB) also get the attribute set.  The old 48 KB threshold was correct
        // when kernels used sized externs (`shmem[N]`), which bake the static allocation into
        // the PTX and don't need the attribute for sub-48-KB requests.
        //
        // Callers that do not use dynamic SMEM at all pass shared_mem_bytes=0 — the guard
        // below skips the attribute call for them, preserving existing behaviour.
        //
        // Typical opt-in SMEM limits by architecture:
        //   sm_120 (Blackwell, RTX 5xxx): 99 KB (CU_DEVICE_ATTRIBUTE_…_OPTIN = 101376)
        //   sm_90  (Hopper):              99 KB
        //   sm_89  (Ada Lovelace):        99 KB
        //   sm_86  (Ampere high-end):    ~100 KB
        if shared_mem_bytes > 0 {
            // Query the device's opt-in SMEM limit before attempting the
            // attribute set. Cached: the limit is a device constant, and the
            // old per-launch query took a second CUDA_STATE lock + driver
            // call on every dynamic-SMEM launch (every sum_dim reduction on
            // the training hot path).
            static SMEM_LIMIT: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
            let device_smem_limit = *SMEM_LIMIT.get_or_init(|| {
                let mut guard2 = state.lock().unwrap();
                let mut limit: i32 = 0;
                unsafe {
                    cuDeviceGetAttribute(
                        &mut limit,
                        CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN,
                        guard2.device,
                    );
                }
                let _ = &mut guard2; // keep guard alive to silence lint
                limit as u32
            });
            if shared_mem_bytes > device_smem_limit {
                // Return the same error code the driver would return, but
                // without actually making the doomed cuFuncSetAttribute call.
                return CUresult::CUDA_ERROR_INVALID_VALUE;
            }
            let res = unsafe {
                cuFuncSetAttribute(
                    func,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    shared_mem_bytes as i32,
                )
            };
            if res != CUresult::CUDA_SUCCESS {
                return res;
            }
        }

        // Validate launch dimensions
        debug_assert!(grid[0] > 0 && grid[1] > 0 && grid[2] > 0,
            "kernel_launch: invalid grid dimensions {:?}", grid);
        debug_assert!(block[0] > 0 && block[1] > 0 && block[2] > 0,
            "kernel_launch: invalid block dimensions {:?}", block);
        debug_assert!(block[0] * block[1] * block[2] <= 1024,
            "kernel_launch: block size {} exceeds max 1024 threads",
            block[0] * block[1] * block[2]);

        // cuda-graphs (P5 item 19): record/verify this launch against the
        // active region. `false` = verified against the captured sequence —
        // the region-end graph launch performs it; skip the real launch.
        if !super::graph_capture::on_kernel(
            func as usize,
            [grid[0] as u32, grid[1] as u32, grid[2] as u32],
            [block[0] as u32, block[1] as u32, block[2] as u32],
            shared_mem_bytes,
            args,
            name_ptr,
        ) {
            return CUresult::CUDA_SUCCESS;
        }

        // Launch kernel (no lock held) — on the per-thread compute stream
        // (p8 PR-A; ordering-neutral vs the old NULL-stream launch, see
        // `current_stream`).
        let mut kernel_args: Vec<*mut c_void> = args.to_vec();
        let res = unsafe {
            cuLaunchKernel(
                func,
                grid[0] as u32, grid[1] as u32, grid[2] as u32,
                block[0] as u32, block[1] as u32, block[2] as u32,
                shared_mem_bytes, current_stream(),
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
            unsafe { cuEventRecord(*stop as CUevent, current_stream()); }
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

    // ------------------------------------------------------------------
    // CFIE engine raw-handle helpers (CFIE Cycle 6 decode-loop wiring).
    //
    // `kernel_launch` above re-resolves the CUfunction on every call
    // (cuModuleGetFunction per launch) — acceptable for training-step
    // kernels, far too slow for a per-token decode loop.  The CFIE
    // engine resolves each kernel ONCE at finalize time via
    // `load_module_once` + `get_function`, then launches through
    // `launch_function_raw`, which does nothing but cuLaunchKernel.
    //
    // Handles cross this boundary as `usize`: CUmodule / CUfunction are
    // opaque driver pointers and not `Send` on their own, so the CFIE
    // engine stores the casts behind its own global Mutex — the same
    // pattern CudaState uses for `module_cache`.
    // ------------------------------------------------------------------

    /// Load a PTX module once, reusing the content-hash module cache
    /// shared with `kernel_launch` (loading is one-time, so the FNV-1a
    /// hash cost is fine here; launching must NOT re-hash).  `ptx_nul`
    /// must be NUL-terminated.  Returns the CUmodule as `usize`, or
    /// `Err(positive CUresult code)`.
    pub(crate) fn load_module_once(ptx_nul: &[u8]) -> Result<usize, u32> {
        debug_assert_eq!(ptx_nul.last(), Some(&0u8), "PTX must be NUL-terminated");
        let s = state();
        let mut guard = s.lock().unwrap();
        unsafe { cuCtxSetCurrent(guard.context); }
        // FNV-1a over the PTX bytes excluding the trailing NUL — matches
        // the hash `kernel_launch` computes by scanning to the NUL, so
        // CFIE modules and kernel_launch modules share one cache entry.
        let mut h: u64 = 14695981039346656037u64;
        for &b in &ptx_nul[..ptx_nul.len().saturating_sub(1)] {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211u64);
        }
        if let Some(m) = guard.module_cache.get(&h) {
            return Ok(*m as usize);
        }
        let mut module: CUmodule = std::ptr::null_mut();
        let res = unsafe { cuModuleLoadData(&mut module, ptx_nul.as_ptr() as *const c_void) };
        if res != CUresult::CUDA_SUCCESS {
            return Err(res as u32);
        }
        guard.module_cache.insert(h, module);
        Ok(module as usize)
    }

    /// Resolve a kernel entry point in a module returned by
    /// `load_module_once`.  Returns the CUfunction as `usize`, or
    /// `Err(positive CUresult code)`.
    pub(crate) fn get_function(module: usize, name: &std::ffi::CStr) -> Result<usize, u32> {
        ensure_context();
        let mut func: CUfunction = std::ptr::null_mut();
        let res = unsafe { cuModuleGetFunction(&mut func, module as CUmodule, name.as_ptr()) };
        if res != CUresult::CUDA_SUCCESS {
            return Err(res as u32);
        }
        Ok(func as usize)
    }

    /// Look up a module-scope global (e.g. the CFIE grammar-mask
    /// table).  Returns `(device_address, size_bytes)` on success, or
    /// `Err(positive CUresult code)` — the caller distinguishes
    /// CUDA_ERROR_NOT_FOUND (symbol absent, legal) from real failures.
    pub(crate) fn module_get_global(
        module: usize,
        name: &std::ffi::CStr,
    ) -> Result<(u64, usize), u32> {
        ensure_context();
        let mut dptr: CUdeviceptr = 0;
        let mut bytes: usize = 0;
        let res = unsafe {
            cuModuleGetGlobal_v2(&mut dptr, &mut bytes, module as CUmodule, name.as_ptr())
        };
        if res != CUresult::CUDA_SUCCESS {
            return Err(res as u32);
        }
        Ok((dptr as u64, bytes))
    }

    /// Launch an already-resolved CUfunction on the NULL stream — no
    /// module load, no function lookup, no PTX hashing.  The CFIE
    /// decode loop calls this once per kernel per token.  Honors
    /// NSL_CUDA_SYNC=1 like `kernel_launch`, but returns the async
    /// error code instead of panicking so the engine FFIs can surface
    /// it per the CFIE ABI.  Returns 0 on success or the positive
    /// CUresult code.
    pub(crate) fn launch_function_raw(
        func: usize,
        grid: [u32; 3],
        block: [u32; 3],
        args: &[*mut c_void],
        smem_dyn_bytes: u32,
    ) -> u32 {
        ensure_context();
        if smem_dyn_bytes > 0 {
            // Dynamic-SMEM opt-in for future extern-.shared CFIE kernels
            // (mirrors kernel_launch).  All current CFIE kernels declare
            // static .shared in-module and pass 0 here.
            let res = unsafe {
                cuFuncSetAttribute(
                    func as CUfunction,
                    CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    smem_dyn_bytes as i32,
                )
            };
            if res != CUresult::CUDA_SUCCESS {
                return res as u32;
            }
        }
        let mut kernel_args: Vec<*mut c_void> = args.to_vec();
        let res = unsafe {
            cuLaunchKernel(
                func as CUfunction,
                grid[0], grid[1], grid[2],
                block[0], block[1], block[2],
                smem_dyn_bytes, current_stream(),
                kernel_args.as_mut_ptr(), std::ptr::null_mut(),
            )
        };
        if res != CUresult::CUDA_SUCCESS {
            return res as u32;
        }
        if sync_mode_enabled() {
            let sync = unsafe { cuCtxSynchronize() };
            if sync != CUresult::CUDA_SUCCESS {
                return sync as u32;
            }
        }
        0
    }
}

#[cfg(feature = "cuda")]
pub(crate) use inner::{cu_event_create, cu_event_elapsed_time, cu_event_destroy, cu_ctx_synchronize};

#[cfg(feature = "cuda")]
pub use inner::{current_stream, cu_event_create_checked, cu_event_record_on_current_stream, cu_event_synchronize_raw, cu_event_elapsed_time_raw};

/// Re-exported so callers that now CHECK a driver status (rather than
/// discarding it) can name the success case without importing cudarc.
#[cfg(feature = "cuda")]
pub use cudarc::driver::sys::CUresult;

// === cuBLAS handle + sgemm wrapper (spec 2026-04-21-matmul-cublas-swap-design) ===

/// cuBLAS handle lifecycle + sgemm dispatch for `gpu_matmul_f32`.
///
/// Per spec §2.3: lazy-init a single `cublasHandle_t` per process via `OnceLock`,
/// intentionally leak it at process exit (NO atexit destructor — cuBLAS destruction
/// after the CUDA context has been torn down produces spurious driver errors).
///
/// Per spec §2.1: NSL tensors are row-major; cuBLAS is column-major. We wrap via
/// the operand-swap idiom: `C_row = A_row @ B_row` is submitted as
/// `C^T_col = B^T_col @ A^T_col` with `transa = transb = CUBLAS_OP_N`.
#[cfg(feature = "cuda")]
pub(crate) mod cublas_inner {
    use cudarc::cublas::{result as cublas_result, sys as cublas_sys};
    use std::sync::OnceLock;

    /// Newtype wrapper so we can implement `Send`/`Sync` for the raw
    /// `cublasHandle_t` pointer (opaque and thread-safe per cuBLAS docs
    /// when serialized via NSL's existing single-context model).
    #[derive(Copy, Clone)]
    pub(crate) struct CublasHandle(pub cublas_sys::cublasHandle_t);

    // SAFETY: cublasHandle_t is an opaque driver-managed pointer. NSL serializes
    // access via single-threaded GPU dispatch today; the handle is thread-safe
    // per the cuBLAS API for multi-thread use with external serialization.
    unsafe impl Send for CublasHandle {}
    unsafe impl Sync for CublasHandle {}

    static CUBLAS_HANDLE: OnceLock<CublasHandle> = OnceLock::new();

    /// cuBLAS math-mode selection (spec §9). Resolved ONCE at `OnceLock`
    /// init time and baked into the handle via `cublasSetMathMode`.
    /// Runtime env-var changes after the first matmul call do NOT retune
    /// the handle; a restart is required to switch modes.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub(crate) enum CublasMathMode {
        /// `CUBLAS_DEFAULT_MATH` — f32 throughout, on FP32 CUDA cores.
        ///
        /// This variant used to be called `Default` and was documented as
        /// "TF32 tensor cores on sm_80+". That was **false**, and NSL's
        /// startup banner repeated it. `CUBLAS_DEFAULT_MATH` does not enable
        /// TF32 for `cublasSgemm`; that needs `CUBLAS_TF32_TENSOR_OP_MATH` or
        /// a `..._FAST_TF32` compute type. Measured at 4096^3 on an RTX
        /// 5070 Ti: this mode 32.9 TFLOP/s, `Pedantic` 33.2 — against a banner
        /// promising pedantic would be "~5-10x slower". A comment on
        /// `sgemm_strided_batched_raw` in this same file had said so all
        /// along; only the muon path ever requested TF32.
        Fp32Cores,
        /// `CUBLAS_PEDANTIC_MATH` — strict f32, no algorithmic shortcuts.
        ///
        /// Distinct from `Fp32Cores` in what cuBLAS is permitted to do
        /// internally (split-k reassociation, alternate kernels), not in the
        /// arithmetic unit used. On this hardware the two measure the same.
        Pedantic,
        /// `CUBLAS_TF32_TENSOR_OP_MATH` — f32 in and out, 10-bit-mantissa
        /// multiplies on tensor cores.
        ///
        /// **Opt-in only.** Every product loses roughly 13 bits of mantissa
        /// precision, so this is a numerics change for the entire stack, not
        /// a free speedup — see `tests/matmul_tf32_mode.rs`, which asserts it
        /// is both measurably faster AND measurably less accurate.
        Tf32,
        /// BF16 tensor-core GEMMs: high-intensity products cast their
        /// operands f32 -> bf16 into caching-allocator scratch and run
        /// `cublasGemmEx(16BF, 16BF -> 32F, COMPUTE_32F)`; low-intensity
        /// products stay f32-storage at explicit `FAST_TF32`.
        ///
        /// Two hard-won facts shape this design (both measured on CUDA
        /// 13.3 / RTX 5070 Ti, 2026-07-29; see `gemm_bf16_mode`):
        ///
        /// 1. `cublasMath_t` has no BF16 fast variant, and the obvious
        ///    substitute — per-call `CUBLAS_COMPUTE_32F_FAST_16BF` with f32
        ///    storage — is a silent NO-OP: cuBLAS serves the identical TF32
        ///    kernels, bit-for-bit, at TF32 speed. Only bf16 OPERAND
        ///    STORAGE reaches the fast kernel family (71.3 vs 36.4 TFLOPS
        ///    at N=4096).
        /// 2. The handle must stay `CUBLAS_DEFAULT_MATH` under this mode:
        ///    on CUDA 13.3 a TF32 handle overrides per-call compute types
        ///    (see `sgemm_batched_row_major`), and the low-intensity arm
        ///    relies on its per-call `FAST_TF32` being honoured.
        ///
        /// Why it exists: on GeForce Blackwell (GB203) the TF32 tensor rate
        /// equals the FP32 vector rate (43.9 TFLOPS dense) — TF32's win is
        /// the SIMT path underachieving, not extra FLOPs. BF16-with-f32-accum
        /// is the first mode with real roofline headroom: 87.9 TFLOPS dense.
        /// The cost: GEMM inputs are rounded to 8 mantissa bits (vs TF32's
        /// 10-bit products over exact inputs) — measured ~7x TF32's drift at
        /// N=4096. Accumulation and outputs stay f32; tensors, the tape, and
        /// every non-cuBLAS kernel are untouched.
        ///
        /// **Opt-in only** (`NSL_MATMUL_BF16=1`); `tests/matmul_bf16_mode.rs`
        /// asserts it is measurably faster than the FP32-cores opt-out AND
        /// sits in a distinctly worse accuracy band than TF32.
        Bf16,
    }

    /// Resolve the math mode via env-var > Cargo-feature precedence (spec §9).
    ///
    /// - `NSL_MATMUL_PEDANTIC=1` forces pedantic (beats everything).
    /// - `NSL_MATMUL_BF16=1` forces BF16 tensor-core compute (beats TF32 —
    ///   the more specific, later-added opt-in wins); `=0` is an explicit
    ///   no-op so scripts can pin it off defensively.
    /// - `NSL_MATMUL_TF32=1` forces TF32; `NSL_MATMUL_TF32=0` forces it off.
    /// - Falling through to `cfg!(feature = "strict-matmul")` => pedantic if
    ///   the feature is on, else **TF32**.
    ///
    /// # The default is TF32, and that is a numerics decision
    ///
    /// Measured on a Coder-50M forward (RTX 5070 Ti, sm_120), steady-state
    /// over the second half of a 20-forward loop so the SM clock is up —
    /// a single cold forward swings this by 10x and is worthless:
    ///
    /// | | FP32 cores | TF32 | |
    /// |---|---|---|---|
    /// | sgemm | 33.3 ms | 21.4 ms | **1.55x** |
    /// | all kernels | 76.5 ms | 64.8 ms | **1.18x** |
    ///
    /// The GEMM speedup is real and reproducible (three paired runs, spread
    /// under 5%), but GEMM is only ~44% of kernel time at this model size, so
    /// the end-to-end win is ~15%, not 55%. Quoting the GEMM number as if it
    /// were the model number would be the same overstatement this file's
    /// history is already littered with.
    ///
    /// The cost is ~13 bits of mantissa on every product (f32's 24-bit
    /// significand down to TF32's 11). Anything needing full f32 sets
    /// `NSL_MATMUL_TF32=0`, or builds with `strict-matmul` for pedantic.
    ///
    /// **Known consequence, not a bug:** the naive PTX matmul kernels
    /// (`nsl_bmm_f32` and friends) run on FP32 CUDA cores regardless. A
    /// product that falls off the cuBLAS path — a strided operand, an
    /// inexpressible batch shape — is therefore computed at a DIFFERENT
    /// precision than the same product on the cuBLAS path. That divergence
    /// existed before this change (as `NSL_MATMUL_TF32=1`) but was opt-in;
    /// it is now the default, so `matmul_batch_collapse`'s two-arm parity
    /// gate carries the wider tolerance explicitly rather than by accident.
    pub(crate) fn resolve_math_mode() -> CublasMathMode {
        if std::env::var("NSL_MATMUL_PEDANTIC").ok().as_deref() == Some("1") {
            return CublasMathMode::Pedantic;
        }
        // Same tri-state discipline as NSL_MATMUL_TF32: only the literal "1"
        // engages, only the literal "0" is an (explicit, defensive) opt-out,
        // and anything else falls through so a typo cannot change arithmetic.
        if std::env::var("NSL_MATMUL_BF16").ok().as_deref() == Some("1") {
            return CublasMathMode::Bf16;
        }
        match std::env::var("NSL_MATMUL_TF32").ok().as_deref() {
            Some("1") => return CublasMathMode::Tf32,
            // Explicit opt-out. Anything else (unset, or a value we do not
            // recognise) falls through to the default rather than silently
            // meaning "off" — a typo'd `NSL_MATMUL_TF32=true` must not
            // quietly change the arithmetic.
            Some("0") => return CublasMathMode::Fp32Cores,
            _ => {}
        }
        if cfg!(feature = "strict-matmul") {
            CublasMathMode::Pedantic
        } else {
            CublasMathMode::Tf32
        }
    }

    /// The math mode this PROCESS runs under, resolved exactly once.
    ///
    /// Two consumers exist: the cuBLAS handle (bakes the mode at its lazy
    /// init) and the transpose-views dispatch coupling (resolves at the
    /// first non-contiguous matmul). Before this cache each did its own
    /// `resolve_math_mode()` env read at its own first-use time, so a
    /// process that mutated `NSL_MATMUL_TF32` between those two moments
    /// silently landed in a MIXED cell — e.g. a TF32 handle with copy-arm
    /// dispatch, the configuration measured 1.12x slower — with no signal
    /// (dispatch choice has no numeric signature; review finding on the
    /// coupling commit). First reader wins; both consumers agree forever.
    static RESOLVED_MATH_MODE: std::sync::OnceLock<CublasMathMode> = std::sync::OnceLock::new();

    pub(crate) fn resolved_math_mode() -> CublasMathMode {
        *RESOLVED_MATH_MODE.get_or_init(resolve_math_mode)
    }

    /// Return a reference to the process-global cuBLAS handle, creating it on
    /// first call. Panics on creation failure (catastrophic — no recovery).
    ///
    /// Applies the resolved math mode (spec §9) via raw `cublasSetMathMode`
    /// FFI since cudarc does not expose it through its safe API.  Logs the
    /// active mode once at init for discoverability (spec §9).
    pub(crate) fn cublas_handle() -> cublas_sys::cublasHandle_t {
        CUBLAS_HANDLE
            .get_or_init(|| {
                // Ensure the CUDA primary context is current on this thread
                // before calling any cuBLAS API (cuBLAS piggybacks the current
                // context at handle creation).
                super::inner::ensure_context();
                let handle = cublas_result::create_handle()
                    .expect("cublasCreate_v2 failed during lazy init");
                // cuBLAS defaults to the NULL/default stream, which matches
                // `inner::current_stream()`. No `cublasSetStream` call needed.

                // Apply the resolved math mode via raw FFI.  `cublasSetMathMode`
                // is exported by `cudarc::cublas::sys` but not wrapped by the
                // safe API surface — raw call is the canonical path.
                let mode = resolved_math_mode();
                let raw_mode = match mode {
                    CublasMathMode::Fp32Cores => cublas_sys::cublasMath_t::CUBLAS_DEFAULT_MATH,
                    CublasMathMode::Pedantic => cublas_sys::cublasMath_t::CUBLAS_PEDANTIC_MATH,
                    CublasMathMode::Tf32 => cublas_sys::cublasMath_t::CUBLAS_TF32_TENSOR_OP_MATH,
                    // BF16 deliberately leaves the handle in DEFAULT_MATH:
                    // there is no BF16 cublasMath_t, and on CUDA 13.3 a TF32
                    // handle would OVERRIDE the per-call FAST_16BF compute
                    // type that actually carries this mode (see the
                    // `sgemm_batched_row_major` floor-not-guarantee note).
                    CublasMathMode::Bf16 => cublas_sys::cublasMath_t::CUBLAS_DEFAULT_MATH,
                };
                // SAFETY: `handle` was just returned by `create_handle` and is a
                // valid `cublasHandle_t`; `raw_mode` is a valid enum variant.
                let status = unsafe { cublas_sys::cublasSetMathMode(handle, raw_mode) };
                if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    eprintln!(
                        "[nsl-matmul] cublasSetMathMode({raw_mode:?}) failed: {status:?} \
                         (handle left in cuBLAS default math mode)"
                    );
                }

                match mode {
                    CublasMathMode::Fp32Cores => eprintln!(
                        "[nsl-matmul] cuBLAS math mode: f32 on FP32 CUDA cores \
                         (NSL_MATMUL_TF32=0)"
                    ),
                    CublasMathMode::Pedantic => eprintln!(
                        "[nsl-matmul] cuBLAS math mode: pedantic (strict f32; same arithmetic \
                         units as FP32 cores, fewer internal shortcuts)"
                    ),
                    CublasMathMode::Tf32 => eprintln!(
                        "[nsl-matmul] cuBLAS math mode: TF32 tensor cores (default) — 1.55x on \
                         gemms, 1.18x on total kernel time at Coder-50M, at ~13 bits less \
                         mantissa per product. Set NSL_MATMUL_TF32=0 for full f32."
                    ),
                    CublasMathMode::Bf16 => eprintln!(
                        "[nsl-matmul] cuBLAS math mode: BF16 tensor-core GEMMs \
                         (NSL_MATMUL_BF16=1) — high-intensity products cast operands to \
                         bf16 storage (measured 71.3 vs TF32's 36.4 TFLOPS at N=4096, \
                         ~7x TF32's numeric drift), low-intensity ones stay f32 at \
                         FAST_TF32. Accumulation and outputs remain f32. Handle stays \
                         in DEFAULT_MATH so per-call compute types are authoritative."
                    ),
                }

                CublasHandle(handle)
            })
            .0
    }

    /// The per-call compute type for `cublasGemmStridedBatchedEx` under the
    /// resolved math mode.
    ///
    /// Under `Bf16` this is `FAST_TF32`, NOT `FAST_16BF`: measured on CUDA
    /// 13.3 / RTX 5070 Ti (2026-07-29, N=4096 probe), `FAST_16BF` with f32
    /// operand storage is a NO-OP — cuBLAS serves the same TF32 kernels,
    /// bit-identical output, 36.7 vs 36.4 TFLOPS. Real BF16 rate (71.3
    /// TFLOPS) requires bf16 OPERAND STORAGE (`gemm_bf16_storage` below),
    /// which needs dense casts the strided-batched path cannot express in
    /// general (arbitrary slice strides, stride-0 broadcast). So batched
    /// products under Bf16 run explicit FAST_TF32 — same arithmetic the
    /// TF32 default gives them, requested per call because the Bf16
    /// handle is DEFAULT_MATH.
    fn gemm_ex_compute_type() -> cublas_sys::cublasComputeType_t {
        match resolved_math_mode() {
            CublasMathMode::Bf16 => {
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
            }
            _ => cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
        }
    }

    /// Arithmetic-intensity gate for the bf16-storage path: the cast of both
    /// operands is O(m·k + k·n) bytes of pure bandwidth, the GEMM saving is
    /// O(m·n·k) FLOPs at (1/36.5 - 1/71.3) s/TFLOP. Equating the two on the
    /// measured numbers (896 GB/s, 6 bytes moved per cast element) gives a
    /// break-even near mnk/(a_elems + b_elems) ~ 250; the default of 512
    /// stays a factor of 2 above break-even so marginal shapes do not
    /// thrash. `NSL_MATMUL_BF16_MIN_RATIO` overrides (integer, elements).
    fn bf16_storage_worthwhile(m: i64, n: i64, k: i64, a_elems: usize, b_elems: usize) -> bool {
        static MIN_RATIO: std::sync::OnceLock<f64> = std::sync::OnceLock::new();
        let min_ratio = *MIN_RATIO.get_or_init(|| {
            std::env::var("NSL_MATMUL_BF16_MIN_RATIO")
                .ok()
                .and_then(|v| v.parse::<f64>().ok())
                .unwrap_or(512.0)
        });
        let flops_cells = m as f64 * n as f64 * k as f64;
        let cast_cells = (a_elems + b_elems) as f64;
        cast_cells > 0.0 && flops_cells / cast_cells >= min_ratio
    }

    /// Bf16-cast scratch for one GEMM's two operands, in cuBLAS operand
    /// order (`first16` holds the wrapper's FIRST cuBLAS operand — the
    /// caller's B_row on the row-major path).
    ///
    /// Allocated by `prepare_bf16_operands` and freed by `Drop` on EVERY
    /// path through a wrapper — including a cuda-graphs replay step where
    /// the cast launches and the GemmEx itself are all verified-and-skipped.
    /// Region digest stability across steps depends on that lockstep: the
    /// caching allocator hands out blocks as a function of the alloc/free
    /// sequence, so a scratch alloc skipped only on replay steps would shift
    /// every later same-size allocation in the region and diverge the
    /// recorded pointers.
    struct Bf16Scratch {
        first16: *mut std::ffi::c_void,
        second16: *mut std::ffi::c_void,
    }

    impl Drop for Bf16Scratch {
        fn drop(&mut self) {
            // Freeing immediately after enqueue is safe, but NOT because the
            // free is deferred — `free_managed` returns the block to the
            // caching allocator's free list synchronously, and the very next
            // `alloc_managed` can hand it out while the GemmEx is still
            // pending (review finding, 2026-07-30). Safety rests on two
            // process invariants documented at the launch path: all GPU work
            // is single-threaded on per-thread BLOCKING streams, so any
            // subsequent WRITE into a reused block — kernel or NULL-stream
            // copy — is enqueued after the GemmEx on the same stream. Under
            // cuda-graphs replay the region's work is enqueued later still
            // (the region-end graph launch), but the graph preserves the
            // captured stream order, so the block's write/read timeline
            // inside the graph matches the eager one. One cross-stream
            // writer DOES exist today — `prefetch_htod_on_transfer` on the
            // NON_BLOCKING transfer stream — but it writes only persistent
            // weight-stream arena slots guarded by events, never
            // caching-allocator blocks, and transfer-stream interactions
            // inside regions taint; any future writer without those
            // constraints must either event-defer these frees
            // (`defer_free_device`) or keep the scratch alive until a sync.
            super::inner::free_managed(self.first16);
            super::inner::free_managed(self.second16);
        }
    }

    /// Decide the storage route for one GEMM under the BF16 matmul mode
    /// and, when bf16 storage wins, cast both operands into fresh scratch.
    /// Returns `None` outside BF16 mode and for low-intensity shapes (the
    /// `FAST_TF32` arm of `gemm_bf16_mode`).
    ///
    /// cuda-graphs: this MUST run BEFORE the wrapper's `on_sgemm_full` hook.
    /// The two cast launches are ordinary hooked kernels, so issuing them
    /// first records them AHEAD of the gemm pseudo-op — the order the stream
    /// actually executes. On a replay step each cast then verifies-and-skips
    /// itself at its own hook (its recorded argument bytes pin the scratch
    /// addresses, so scratch that fails to stabilize across steps diverges
    /// the region rather than corrupting it), and the alloc/free bookkeeping
    /// still runs — see `Bf16Scratch`. Recording the casts BEHIND the gemm —
    /// the pre-2026-08 shape, where the decision lived after the hook —
    /// meant a replay's early-return at the gemm hook could never re-issue
    /// them, which is why this path used to taint every region it touched.
    ///
    /// The cast uses the SR-BF16 campaign's round-to-nearest-even
    /// `precision_cast_kernels` (deterministic, not stochastic) and the
    /// caching allocator, so steady-state scratch is a cache hit, not a
    /// cuMemAlloc. Element counts are independent of the transpose flags (a
    /// transposed 2-D view is the same dense buffer read with swapped lda
    /// semantics, so casting the flat buffer is exact).
    fn prepare_bf16_operands(
        first_dev: *const f32,
        first_elems: usize,
        second_dev: *const f32,
        second_elems: usize,
        m: i64,
        n: i64,
        k: i64,
    ) -> Option<Bf16Scratch> {
        if resolved_math_mode() != CublasMathMode::Bf16 {
            return None;
        }
        if !bf16_storage_worthwhile(m, n, k, first_elems, second_elems) {
            return None;
        }
        let first16 = super::inner::alloc_managed(first_elems * 2);
        let second16 = super::inner::alloc_managed(second_elems * 2);
        super::gpu_cast_raw_f32_to_bf16(first_dev as u64, first16 as u64, first_elems);
        super::gpu_cast_raw_f32_to_bf16(second_dev as u64, second16 as u64, second_elems);
        Some(Bf16Scratch { first16, second16 })
    }

    /// The BF16 math-mode GEMM issue path: `cublasGemmEx(16BF, 16BF -> 32F,
    /// COMPUTE_32F)` over the pre-cast scratch when `prepare_bf16_operands`
    /// produced one, or the f32-storage `FAST_TF32` form when it declined
    /// (low-intensity shapes).
    ///
    /// Why storage and not a compute-type hint: measured on CUDA 13.3 /
    /// RTX 5070 Ti (N=4096, 2026-07-29 probe),
    ///
    /// | call | TFLOPS | max_rel_err |
    /// |---|---|---|
    /// | Sgemm, TF32 handle            | 36.4 | 1.1e-3 |
    /// | GemmEx f32 io, FAST_16BF      | 36.7 | 1.1e-3 (bit-identical to TF32) |
    /// | GemmEx bf16 in / f32 out, 32F | 71.3 | 7.5e-3 |
    ///
    /// `FAST_16BF` with f32 storage is silently served by the TF32 kernels;
    /// only bf16 operand storage reaches the 87.9-TFLOPS-peak kernel family.
    ///
    /// Arguments are in cuBLAS column-major order, ALREADY operand-swapped
    /// by the caller (first operand is the caller's B_row); `scratch` was
    /// prepared in the same operand order.
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_bf16_mode(
        handle: cublas_sys::cublasHandle_t,
        transa: cublas_sys::cublasOperation_t,
        transb: cublas_sys::cublasOperation_t,
        m: i32,
        n: i32,
        k: i32,
        alpha: &f32,
        a_dev: *const f32,
        lda: i32,
        b_dev: *const f32,
        ldb: i32,
        beta: &f32,
        c_dev: *mut f32,
        ldc: i32,
        scratch: Option<&Bf16Scratch>,
    ) -> Result<(), cublas_result::CublasError> {
        let f32t = cublas_sys::cudaDataType_t::CUDA_R_32F;
        let (a_ptr, b_ptr, ab_type, compute) = match scratch {
            Some(s) => (
                s.first16 as *const std::ffi::c_void,
                s.second16 as *const std::ffi::c_void,
                cublas_sys::cudaDataType_t::CUDA_R_16BF,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F,
            ),
            None => (
                a_dev as *const std::ffi::c_void,
                b_dev as *const std::ffi::c_void,
                f32t,
                cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32,
            ),
        };
        let status = unsafe {
            cublas_sys::cublasGemmEx(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha as *const f32 as *const std::ffi::c_void,
                a_ptr,
                ab_type,
                lda,
                b_ptr,
                ab_type,
                ldb,
                beta as *const f32 as *const std::ffi::c_void,
                c_dev as *mut std::ffi::c_void,
                f32t,
                ldc,
                compute,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
            )
        };
        // The scratch (when present) is freed by the caller's `Bf16Scratch`
        // Drop right after this returns — see its Drop for the
        // free-after-enqueue safety argument.
        if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublas_result::CublasError(status));
        }
        Ok(())
    }

    /// `sgemm_row_major`, but either operand may be supplied as a TRANSPOSED
    /// 2-D view instead of being materialised first (item 9 phase 2).
    ///
    /// `transa` means: the buffer at `a_dev` is stored as a contiguous
    /// row-major `[k, m]`, and the caller wants its transpose `[m, k]`. Same
    /// for `transb` with `[n, k]` -> `[k, n]`. This is exactly what NSL's
    /// `.transpose(0, 1)` on a 2-D tensor produces — a view with swapped
    /// shape and swapped strides over an untouched buffer.
    ///
    /// ## Deriving the ops and leading dimensions
    ///
    /// The wrapper computes `C_row = A_row @ B_row` by asking cuBLAS (which is
    /// column-major) for `C^T = B^T A^T`, exploiting that a row-major `[p, q]`
    /// buffer IS a column-major `[q, p]`. So cuBLAS's first operand is B and
    /// its second is A. Under that swap:
    ///
    /// * `transb` toggles the op of cuBLAS's FIRST operand, `transa` the
    ///   SECOND. Getting this backwards is the whole risk of this function.
    /// * The leading dimension is always the row length of the operand AS
    ///   STORED, which is what changes: a non-transposed `B_row` is stored
    ///   `[k, n]` so `lda = n`; a transposed one is stored `[n, k]` so
    ///   `lda = k`. Likewise `ldb` is `k` normally and `m` when `transa`.
    ///
    /// Cross those and cuBLAS reads a real, in-bounds, WRONG sub-matrix —
    /// silent garbage, which is precisely the tied-embedding miscompile of
    /// PR #335. `sgemm_wgrad_accum` below is the in-tree precedent for the
    /// crossed form and was used to check this derivation.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn sgemm_row_major_t(
        a_dev: *const f32,
        b_dev: *const f32,
        c_dev: *mut f32,
        m: u64,
        n: u64,
        k: u64,
        transa: bool,
        transb: bool,
        alpha: f32,
        beta: f32,
    ) -> Result<(), cublas_result::CublasError> {
        debug_assert!(m > 0 && n > 0 && k > 0, "sgemm requires m,n,k > 0");
        debug_assert!(
            m <= i32::MAX as u64 && n <= i32::MAX as u64 && k <= i32::MAX as u64,
            "sgemm dims must fit in i32"
        );
        // cuda-graphs: the bf16-storage decision and its two cast launches
        // come BEFORE the gemm hook, so the recorded op order matches the
        // stream order (casts, then gemm) and the scratch bookkeeping runs
        // on replay steps too — see `prepare_bf16_operands`.
        let scratch = prepare_bf16_operands(
            b_dev,
            (k * n) as usize, // cuBLAS's FIRST operand is B_row
            a_dev,
            (m * k) as usize, // ...and its SECOND is A_row
            n as i64,
            m as i64,
            k as i64,
        );
        let precision = if scratch.is_some() {
            super::graph_capture::GemmPrecision::Bf16Storage
        } else {
            super::graph_capture::GemmPrecision::F32
        };
        if !super::graph_capture::on_sgemm_full(
            super::graph_capture::SgemmKind::RowMajor,
            precision,
            transa,
            transb,
            a_dev as usize,
            b_dev as usize,
            c_dev as usize,
            m, n, k, alpha, beta,
            1, m * k, k * n, m * n,
        ) {
            // Verified-and-skipped: the region-end graph launch performs the
            // whole cast+gemm group. `scratch` drops here, keeping the
            // allocator in lockstep with the recorded steps.
            return Ok(());
        }
        let handle = cublas_handle();
        {
            let r = unsafe {
                cublas_sys::cublasSetStream_v2(
                    handle,
                    super::inner::current_stream() as cublas_sys::cudaStream_t,
                )
            };
            if r != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                eprintln!("[nsl] cublasSetStream_v2 failed: {r:?} — gemm stays on its previous stream");
            }
        }
        let op = |t: bool| {
            if t {
                cublas_sys::cublasOperation_t::CUBLAS_OP_T
            } else {
                cublas_sys::cublasOperation_t::CUBLAS_OP_N
            }
        };
        // BF16 cannot be expressed on the handle, so its branch goes through
        // `gemm_bf16_mode` (the pre-cast scratch for high-intensity shapes,
        // explicit FAST_TF32 otherwise). Every other mode keeps the proven
        // cublasSgemm_v2 call, where the handle's math mode governs —
        // converting those too would re-open the measured speed/accuracy
        // gates for no functional gain.
        if resolved_math_mode() == CublasMathMode::Bf16 {
            return unsafe {
                gemm_bf16_mode(
                    handle,
                    op(transb), // cuBLAS's FIRST operand is B_row
                    op(transa), // ...and its SECOND is A_row
                    n as i32,
                    m as i32,
                    k as i32,
                    &alpha,
                    b_dev,
                    if transb { k as i32 } else { n as i32 },
                    a_dev,
                    if transa { m as i32 } else { k as i32 },
                    &beta,
                    c_dev,
                    n as i32,
                    scratch.as_ref(),
                )
            };
        }
        cublas_result::sgemm(
            handle,
            op(transb), // cuBLAS's FIRST operand is B_row
            op(transa), // ...and its SECOND is A_row
            n as i32,
            m as i32,
            k as i32,
            &alpha,
            b_dev,
            if transb { k as i32 } else { n as i32 }, // lda: B_row's stored row length
            a_dev,
            if transa { m as i32 } else { k as i32 }, // ldb: A_row's stored row length
            &beta,
            c_dev, n as i32,
        )
    }

    /// Strided-batched row-major SGEMM: `C[i] = alpha * A[i] @ B[i] + beta * C[i]`
    /// for `batch` slices (item 9).
    ///
    /// All three operands are ROW-MAJOR, using the same operand-swap idiom as
    /// `sgemm_row_major`: cuBLAS is asked for `C^T = B^T A^T` in its
    /// column-major view, which is the same bytes. `stride_*` are ELEMENT
    /// offsets between consecutive slices; **0 broadcasts** that operand
    /// across the batch, which is how `gpu_matmul_f32` expresses
    /// `[b,m,k] @ [k,n]`-style shape broadcasting.
    ///
    /// Distinct from `sgemm_strided_batched_raw` above, which takes RAW
    /// column-major arguments, is muon-only, and deliberately taints
    /// cuda-graph regions. This one is on the general matmul path, so it
    /// records a proper pseudo-op instead — tainting here would disable graph
    /// capture for every model containing a batched product.
    ///
    /// Compute type is `CUBLAS_COMPUTE_32F`, matching `sgemm_row_major`, so
    /// this arm does not itself request tensor cores.
    ///
    /// That is NOT the same as "this arm is always full f32". On CUDA 13.3 the
    /// handle's math mode wins over the per-call compute type: under
    /// `NSL_MATMUL_TF32=1` this call was measured at 8.104e-4 error against
    /// the 2-D path's 8.101e-4 on the same product, i.e. both went to tensor
    /// cores. That consistency is what you want — QK^T and the projections
    /// should not disagree about precision — but it means the compute type
    /// here is a floor, not a guarantee.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn sgemm_batched_row_major(
        a_dev: *const f32,
        b_dev: *const f32,
        c_dev: *mut f32,
        m: u64,
        n: u64,
        k: u64,
        batch: u64,
        stride_a: u64,
        stride_b: u64,
        stride_c: u64,
        alpha: f32,
        beta: f32,
    ) -> Result<(), cublas_result::CublasError> {
        debug_assert!(m > 0 && n > 0 && k > 0 && batch > 0);
        debug_assert!(
            m <= i32::MAX as u64
                && n <= i32::MAX as u64
                && k <= i32::MAX as u64
                && batch <= i32::MAX as u64,
            "batched gemm dims must fit in i32"
        );
        if !super::graph_capture::on_sgemm_batched(
            a_dev as usize, b_dev as usize, c_dev as usize,
            m, n, k, alpha, beta, batch, stride_a, stride_b, stride_c,
        ) {
            return Ok(());
        }
        let handle = cublas_handle();
        {
            let r = unsafe {
                cublas_sys::cublasSetStream_v2(
                    handle,
                    super::inner::current_stream() as cublas_sys::cudaStream_t,
                )
            };
            if r != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                eprintln!(
                    "[nsl] cublasSetStream_v2 failed: {r:?} — batched matmul stays on its previous stream"
                );
            }
        }
        let f32t = cublas_sys::cudaDataType_t::CUDA_R_32F;
        let status = unsafe {
            cublas_sys::cublasGemmStridedBatchedEx(
                handle,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                cublas_sys::cublasOperation_t::CUBLAS_OP_N,
                n as i32, // cublas m = N (cols of row-major C)
                m as i32, // cublas n = M (rows of row-major C)
                k as i32, // contraction dim
                &alpha as *const f32 as *const std::ffi::c_void,
                b_dev as *const std::ffi::c_void, // A^cublas := B_row
                f32t,
                n as i32,        // lda = N
                stride_b as i64, // ...and B_row's stride goes with it
                a_dev as *const std::ffi::c_void, // B^cublas := A_row
                f32t,
                k as i32, // ldb = K
                stride_a as i64,
                &beta as *const f32 as *const std::ffi::c_void,
                c_dev as *mut std::ffi::c_void,
                f32t,
                n as i32, // ldc = N
                stride_c as i64,
                batch as i32,
                // Mode-derived: CUBLAS_COMPUTE_32F except under BF16, where
                // the per-call type is the only carrier of the mode (the
                // handle is DEFAULT_MATH there). Under TF32 the handle
                // overrides this anyway — see the floor-not-guarantee note
                // in the doc comment above.
                gemm_ex_compute_type(),
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
            )
        };
        if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            return Err(cublas_result::CublasError(status));
        }
        Ok(())
    }

    /// Weight-gradient contraction: `C[d, o] = alpha * (X[N, d]^T @ G[N, o]) + beta * C`.
    ///
    /// All three operands are ROW-MAJOR. This is the shape the weight gradient
    /// actually has once the batch dimension is flattened: for a forward
    /// `y = x @ W` with `x: [B, T, d]`, the gradient is
    /// `dW[d, o] = sum_{b,t} x[b,t,d] * dy[b,t,o]`, which is exactly this
    /// contraction with `N = B*T` — the batch sum falls out of the reduction
    /// dimension instead of needing a separate `[B, d, o]` temporary and a
    /// batch-reduce pass.
    ///
    /// With `beta = 1.0` and `C = m_partial`, the whole
    /// "matmul -> reduce_to_shape -> scaled accumulate" chain collapses into
    /// this single call.
    ///
    /// # Column-major derivation (get this wrong and it is silently transposed)
    ///
    /// Row-major `M[r, c]` is column-major `M_cm[c, r]` with `ld = c`. So:
    ///   * `C[d, o]` -> `C_cm[o, d]`, ldc = o
    ///   * `G[N, o]` -> `G_cm[o, N]`, lda = o
    ///   * `X[N, d]` -> `X_cm[d, N]`, ldb = d
    /// and `C[d,o] = sum_n X[n,d] G[n,o]` becomes
    /// `C_cm[o,d] = sum_n G_cm[o,n] X_cm[d,n] = G_cm @ X_cm^T`.
    /// Hence `transa = N` on G, `transb = T` on X, `m = o`, `n = d`, `k = N`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn sgemm_wgrad_accum(
        x_dev: *const f32,
        g_dev: *const f32,
        c_dev: *mut f32,
        n_rows: u64, // N = flattened batch*time (the contraction dim)
        d_in: u64,   // d = C rows      (x's trailing dim)
        d_out: u64,  // o = C cols      (g's trailing dim)
        alpha: f32,
        beta: f32,
    ) -> Result<(), cublas_result::CublasError> {
        debug_assert!(
            n_rows > 0 && d_in > 0 && d_out > 0,
            "wgrad gemm requires N,d,o > 0"
        );
        debug_assert!(
            n_rows <= i32::MAX as u64 && d_in <= i32::MAX as u64 && d_out <= i32::MAX as u64,
            "wgrad gemm dims must fit in i32"
        );
        // cuda-graphs: the bf16-storage decision and its cast launches come
        // BEFORE the gemm hook — see `prepare_bf16_operands` and the
        // row-major wrapper.
        let scratch = prepare_bf16_operands(
            g_dev,
            (n_rows * d_out) as usize, // cuBLAS's FIRST operand is G
            x_dev,
            (n_rows * d_in) as usize, // ...and its SECOND is X
            d_out as i64,
            d_in as i64,
            n_rows as i64,
        );
        let precision = if scratch.is_some() {
            super::graph_capture::GemmPrecision::Bf16Storage
        } else {
            super::graph_capture::GemmPrecision::F32
        };
        // Recorded as a distinct pseudo-op shape. The alpha/beta
        // bits are part of GpuOp::Sgemm's identity, so an accumulating gemm
        // can never be replayed as an overwriting one.
        // Recorded with its OWN kind. `sgemm_row_major` calls the same hook
        // with the same eight values, but performs a different contraction —
        // without the discriminator the two compare equal under capture and
        // eager repair reconstructs whichever it happens to hold as a plain
        // row-major gemm. Silently wrong gradients, no error.
        if !super::graph_capture::on_sgemm_full(
            super::graph_capture::SgemmKind::WgradAccum,
            precision,
            false,
            false,
            x_dev as usize,
            g_dev as usize,
            c_dev as usize,
            n_rows,
            d_in,
            d_out,
            alpha,
            beta,
            1,
            n_rows * d_in,
            n_rows * d_out,
            d_in * d_out,
        ) {
            // Verified-and-skipped; `scratch` drops here (allocator
            // lockstep — see `Bf16Scratch`).
            return Ok(());
        }
        let handle = cublas_handle();
        {
            let r = unsafe {
                cublas_sys::cublasSetStream_v2(
                    handle,
                    super::inner::current_stream() as cublas_sys::cudaStream_t,
                )
            };
            if r != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                eprintln!("[nsl] cublasSetStream_v2 failed: {:?} — wgrad gemm stays on its previous stream", r);
            }
        }
        // BF16 goes per-call through `gemm_bf16_mode` (no handle-level BF16
        // exists). The weight-gradient contraction joins the mode with the
        // forward/dgrad GEMMs deliberately: a mixed cell where wgrad silently
        // ran a different precision than the products it differentiates
        // would be unobservable from dispatch alone. The bf16 cast rounds
        // the INPUTS (x and the incoming grad); the beta=1 accumulation into
        // the f32 master gradient stays f32 (Ctype CUDA_R_32F, COMPUTE_32F).
        if resolved_math_mode() == CublasMathMode::Bf16 {
            return unsafe {
                gemm_bf16_mode(
                    handle,
                    cublas_sys::cublasOperation_t::CUBLAS_OP_N, // G_cm
                    cublas_sys::cublasOperation_t::CUBLAS_OP_T, // X_cm, transposed
                    d_out as i32,
                    d_in as i32,
                    n_rows as i32,
                    &alpha,
                    g_dev,
                    d_out as i32,
                    x_dev,
                    d_in as i32,
                    &beta,
                    c_dev,
                    d_out as i32,
                    scratch.as_ref(),
                )
            };
        }
        cublas_result::sgemm(
            handle,
            cublas_sys::cublasOperation_t::CUBLAS_OP_N, // G_cm, no transpose
            cublas_sys::cublasOperation_t::CUBLAS_OP_T, // X_cm, transposed
            d_out as i32,  // cublas m = rows of C_cm = o
            d_in as i32,   // cublas n = cols of C_cm = d
            n_rows as i32, // contraction dim = N
            &alpha,
            g_dev, d_out as i32,  // A := G_cm, lda = o
            x_dev, d_in as i32,   // B := X_cm, ldb = d
            &beta,
            c_dev, d_out as i32,  // C := C_cm, ldc = o
        )
    }

    /// Strided-batched SGEMM in RAW cuBLAS column-major terms (muon batched
    /// Newton-Schulz). The caller does its own row/column-major mapping —
    /// the muon batch engine's square operands are all symmetric, which is
    /// what makes its mappings transpose-free.
    ///
    /// Not wired into the cuda-graph pseudo-op stream: the optimizer step
    /// runs outside captured regions. If a captured region ever reaches
    /// this, taint it loudly rather than silently diverging the digest.
    #[allow(clippy::too_many_arguments)]
    pub(crate) unsafe fn sgemm_strided_batched_raw(
        transa: bool,
        transb: bool,
        m: i64,
        n: i64,
        k: i64,
        a_dev: *const f32,
        lda: i64,
        stride_a: i64,
        b_dev: *const f32,
        ldb: i64,
        stride_b: i64,
        c_dev: *mut f32,
        ldc: i64,
        stride_c: i64,
        batch: i64,
        tf32: bool,
    ) -> Result<(), cublas_result::CublasError> {
        debug_assert!(m > 0 && n > 0 && k > 0 && batch > 0);
        debug_assert!(
            m <= i32::MAX as i64 && n <= i32::MAX as i64 && k <= i32::MAX as i64
                && batch <= i32::MAX as i64
        );
        if super::graph_capture::in_region() {
            super::graph_capture::taint("batched-sgemm");
        }
        let handle = cublas_handle();
        {
            let r = unsafe {
                cublas_sys::cublasSetStream_v2(
                    handle,
                    super::inner::current_stream() as cublas_sys::cudaStream_t,
                )
            };
            if r != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                eprintln!(
                    "[nsl] cublasSetStream_v2 failed: {r:?} — batched gemm stays on its previous stream"
                );
            }
        }
        let op = |t: bool| {
            if t {
                cublas_sys::cublasOperation_t::CUBLAS_OP_T
            } else {
                cublas_sys::cublasOperation_t::CUBLAS_OP_N
            }
        };
        let alpha: f32 = 1.0;
        let beta: f32 = 0.0;
        // GemmStridedBatchedEx with an explicit compute type. By DEFAULT the
        // process handle sits in CUBLAS_DEFAULT_MATH, which runs f32 gemms on
        // FP32 CUDA cores, so tensor-core TF32 must be requested per call —
        // the muon batch engine opts in, Newton-Schulz being a coarse
        // polynomial approximation whose own gates are tolerance-based.
        //
        // The converse no longer holds. Item 9 made `NSL_MATMUL_TF32=1` put
        // the handle in CUBLAS_TF32_TENSOR_OP_MATH, and on CUDA 13.3 the
        // handle's math mode overrides this per-call compute type: under that
        // flag `tf32 = false` here still runs on tensor cores. Callers passing
        // false are asking for no TF32 *from this call site*, not asserting
        // the process is in full-f32 mode.
        let compute = if tf32 {
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F_FAST_TF32
        } else {
            cublas_sys::cublasComputeType_t::CUBLAS_COMPUTE_32F
        };
        let f32t = cublas_sys::cudaDataType_t::CUDA_R_32F;
        let status = unsafe {
            cublas_sys::cublasGemmStridedBatchedEx(
                handle,
                op(transa),
                op(transb),
                m as i32,
                n as i32,
                k as i32,
                &alpha as *const f32 as *const std::ffi::c_void,
                a_dev as *const std::ffi::c_void,
                f32t,
                lda as i32,
                stride_a,
                b_dev as *const std::ffi::c_void,
                f32t,
                ldb as i32,
                stride_b,
                &beta as *const f32 as *const std::ffi::c_void,
                c_dev as *mut std::ffi::c_void,
                f32t,
                ldc as i32,
                stride_c,
                batch as i32,
                compute,
                cublas_sys::cublasGemmAlgo_t::CUBLAS_GEMM_DFALT,
            )
        };
        if status != cublas_sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            // Surface as the same error type the sgemm path produces.
            return Err(cublas_result::CublasError(status));
        }
        Ok(())
    }

}

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
fn tensor_shape_slice<'a>(tensor: &'a crate::tensor::NslTensor) -> &'a [i64] {
    assert!(tensor.ndim >= 0, "tensor has negative ndim: {}", tensor.ndim);
    let ndim = tensor.ndim as usize;
    if ndim == 0 {
        return &[];
    }
    assert!(
        !tensor.shape.is_null(),
        "tensor shape is null for ndim {} (len={}, device={}, dtype={})",
        tensor.ndim,
        tensor.len,
        tensor.device,
        tensor.dtype,
    );
    unsafe { std::slice::from_raw_parts(tensor.shape, ndim) }
}

#[cfg(feature = "cuda")]
fn gpu_broadcast_shape(a: &crate::tensor::NslTensor, b: &crate::tensor::NslTensor) -> Option<Vec<i64>> {
    let a_shape = tensor_shape_slice(a);
    let b_shape = tensor_shape_slice(b);
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
    let current_shape = tensor_shape_slice(tensor);
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

/// Refuse a non-f32 operand at the head of an f32-only GPU kernel path.
///
/// Most of the kernels below are f32-only in fact and said so nowhere: they
/// size their output with `alloc_managed(n * 4)`, hardcode `1` in the dtype
/// slot of the tensor they publish, and hand `t.data` to a `.f32` PTX kernel.
/// Every dispatcher that reaches them branches on `device` alone. A 2-byte
/// dtype (fp16, bf16) carries the same element COUNT in half the bytes, so
/// such a kernel reads twice its operand's allocation — an out-of-bounds
/// device read that returns plausible numbers instead of faulting, and whose
/// result is then published as f32 and believed. `gpu_matmul_f32` has carried
/// exactly this check since item 9; this generalises it to the other 40-odd
/// sites, which had none.
///
/// `assert!`, never `debug_assert!`: this workspace has no `[profile.release]`
/// section and no `debug-assertions` key anywhere, and CI ships release — a
/// `debug_assert!` here would be a no-op in precisely the builds that matter.
/// That trap was hit twice independently on 2026-07-30 (PRs #453 and #454).
/// Panicking rather than the `eprintln! + process::abort()` idiom used
/// elsewhere in this module is also what makes the refusal observable from a
/// `#[should_panic]` test (`crates/nsl-runtime/tests/gpu_dtype_refusal.rs`);
/// an `abort()` cannot be caught.
///
/// Compares against `DTYPE_F32`, never the literal `1`. The literal is what
/// let the drift go unnoticed: `dtype == 1` reads as a magic number and greps
/// as nothing, so nobody auditing "which paths assume f32" could find them.
///
/// `op` names the kernel or entry point (a trailing NUL from a PTX kernel-name
/// literal is trimmed, so `kernel_name` can be passed through verbatim) and
/// `role` names the operand's position, so a multi-operand kernel says which
/// one is wrong instead of just that something is.
#[cfg(feature = "cuda")]
#[inline]
pub(crate) fn assert_gpu_f32(t: &crate::tensor::NslTensor, op: &str, role: &str) {
    assert!(
        t.dtype == crate::tensor::DTYPE_F32,
        "{}: {role} is f32-only (dtype {}); got dtype {}. This path sizes its \
         buffers at 4 bytes per element and reads the operand as `*const f32`, \
         so a narrower dtype would be read past the end of its allocation and \
         returned as plausible wrong numbers rather than faulting. Widen the \
         operand first (`nsl_tensor_to_f32`, or `.to(f32)` at NSL level); \
         mixed-precision kernel dispatch is roadmap item 9 and is not \
         implemented.",
        op.trim_end_matches('\0'),
        crate::tensor::DTYPE_F32,
        t.dtype,
    );
}

/// GPU elementwise binary op.
/// Falls back to CPU on GPU OOM (transfers to CPU, computes, transfers back).
#[cfg(feature = "cuda")]
pub(crate) fn gpu_elementwise_binary(a_ptr: i64, b_ptr: i64, ptx: &str, kernel_name: &str) -> i64 {
    use crate::tensor::NslTensor;
    inner::set_oom_context(kernel_name.trim_end_matches('\0'));
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };
    assert_gpu_f32(a, kernel_name, "lhs");
    assert_gpu_f32(b, kernel_name, "rhs");
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
    inner::sync_after_kernel();
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
    assert_gpu_f32(a, kernel_name, "input");
    // The kernels index flat row-major; a zero-copy view (transpose/expand)
    // would be read as if contiguous — same stride-blindness class as the
    // tied-embedding matmul bug. Materialize first (on-device strided copy;
    // owned ref, freed after), mirroring gpu_elementwise_binary's guard.
    if !a.is_contiguous() {
        let a_c = crate::tensor::nsl_tensor_contiguous(a_ptr);
        let result = gpu_elementwise_unary(a_c, ptx, kernel_name);
        crate::tensor::nsl_tensor_free(a_c);
        return result;
    }
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
    inner::sync_after_kernel();
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

    // BEFORE the materialization below, not after. This function's contract is
    // to DEGRADE on a non-f32 tensor, not to refuse one — the arm further down
    // hands it to the CPU implementation and returns. But `nsl_tensor_contiguous`
    // now refuses a non-f32 device tensor outright (its GPU arm is
    // `gpu_strided_copy_f32`), so leaving the check where it was would turn a
    // NON-CONTIGUOUS bf16 input into an abort while a contiguous one still
    // degraded — the same input taking two different paths depending on a
    // stride. Checking first makes the degrade unconditional again.
    if tensor.dtype != crate::tensor::DTYPE_F32 {
        return cpu_fallback_rotate_half(tensor_ptr, tensor.device as i32);
    }

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

    // Unreachable now that the dtype is checked above the materialization —
    // kept as a belt, since `contiguous` is a different tensor than `tensor`
    // and nothing else here re-establishes the invariant locally.
    debug_assert_eq!(contiguous.dtype, crate::tensor::DTYPE_F32);

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
    inner::sync_after_kernel();

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
    // The FBIP arms that reach here (`nsl_tensor_relu` and its ~10 siblings in
    // activation.rs) branch on `can_mutate_inplace_gpu()` alone, so the guard
    // in the non-FBIP twin `gpu_elementwise_unary` does not cover them — and
    // the in-place path is the one a freshly produced fp16/bf16 tensor takes,
    // because a fresh tensor is uniquely owned. Every caller passes an
    // `*_f32` kernel, which would overwrite twice the buffer in place.
    assert_gpu_f32(a, kernel_name, "input");
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
    inner::sync_after_kernel();
}

/// GPU elementwise binary op — in-place (FBIP). Writes output to left operand's buffer.
/// Caller must have verified `can_mutate_inplace_gpu()` on `a` and shapes match.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_elementwise_binary_inplace(a_ptr: i64, b_ptr: i64, ptx: &str, kernel_name: &str) {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };
    assert_eq!(a.len, b.len, "GPU inplace elementwise: length mismatch");
    assert_gpu_f32(a, kernel_name, "destination");
    assert_gpu_f32(b, kernel_name, "rhs");

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
    inner::sync_after_kernel();
}

/// GPU scalar op — in-place (FBIP). Writes output to input buffer.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_scalar_op_inplace(a_ptr: i64, scalar: f32, ptx: &str, kernel_name: &str) {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    assert_gpu_f32(a, kernel_name, "destination");
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
    inner::sync_after_kernel();
}

/// In-place scale of a RAW contiguous device f32 buffer (no NslTensor
/// wrapper) — the ZeRO stage-2 scatter averages its staging buffer before
/// unpacking, so byte-range CHUNKS of a tensor scale exactly once. Same
/// kernel as `nsl_tensor_mul_scalar_inplace`'s device path (bit-exact).
///
/// PRECONDITION (unenforceable here): `dev` must address f32 elements.
/// There is no `NslTensor` in this signature, so `assert_gpu_f32` cannot
/// apply — the kernel reads `n` * 4 bytes from `dev` and the caller owns
/// that guarantee.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_scale_raw_f32(dev: *mut std::ffi::c_void, n: usize, scalar: f32) {
    let mut a_data = dev as u64;
    let mut c_data = dev as u64; // output = input buffer
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
        kernels::MUL_SCALAR_F32_PTX.as_ptr(),
        "nsl_mul_scalar_f32\0".as_ptr(),
        [grid, 1, 1],
        [block, 1, 1],
        &args,
        0,
    );
    assert_eq!(
        result as u32, 0,
        "GPU raw f32 scale kernel failed: {}",
        result as u32
    );
    inner::sync_after_kernel();
}

/// Fused weight-gradient accumulate: `m_partial += scale * (x^T @ g)`,
/// contracted over the flattened leading dimensions.
///
/// One cuBLAS call with `alpha = scale, beta = 1.0` writing straight into
/// `m_partial` — no `[B, d, o]` temporary, no batch-reduce pass, no separate
/// elementwise accumulate launch. See `sgemm_wgrad_accum` for the column-major
/// derivation and `nsl_tensor_wgrad_accum` for the (deliberate)
/// non-bit-exactness.
///
/// The caller has already validated device/dtype/contiguity/shape.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_wgrad_accum_f32(
    m_ptr: i64,
    x_ptr: i64,
    g_ptr: i64,
    n_rows: u64,
    d_in: u64,
    d_out: u64,
    scale: f32,
) {
    use crate::tensor::NslTensor;
    let m = unsafe { &*(m_ptr as *const NslTensor) };
    let x = unsafe { &*(x_ptr as *const NslTensor) };
    let g = unsafe { &*(g_ptr as *const NslTensor) };
    debug_assert_eq!(
        m.len as u64,
        d_in * d_out,
        "wgrad accum: m_partial length must be d_in * d_out"
    );
    debug_assert_eq!(
        x.len as u64,
        n_rows * d_in,
        "wgrad accum: x length must be N * d_in"
    );
    debug_assert_eq!(
        g.len as u64,
        n_rows * d_out,
        "wgrad accum: g length must be N * d_out"
    );

    inner::ensure_context();

    // SAFETY: all three are device f32 buffers of the sizes asserted above,
    // validated by `nsl_tensor_wgrad_accum` before dispatch.
    let res = unsafe {
        cublas_inner::sgemm_wgrad_accum(
            x.data as *const f32,
            g.data as *const f32,
            m.data as *mut f32,
            n_rows,
            d_in,
            d_out,
            scale,
            // beta = 1.0: ACCUMULATE into m_partial rather than overwrite it.
            // This is the whole point of the fusion; a 0.0 here would silently
            // discard every earlier micro-batch's contribution.
            1.0,
        )
    };
    if let Err(e) = res {
        panic!(
            "[nsl-wgrad] cuBLAS wgrad accum failed (N={n_rows} d={d_in} o={d_out}): {e:?}. \
             This writes gradients in place, so there is no safe partial result to \
             continue from."
        );
    }

    if inner::sync_mode_enabled() {
        let sync_result = unsafe { cudarc::driver::sys::cuCtxSynchronize() };
        assert_eq!(
            sync_result,
            cudarc::driver::sys::CUresult::CUDA_SUCCESS,
            "[nsl-wgrad] async CUDA error after wgrad accum gemm"
        );
    }
}

/// FASE fused scaled-add (Milestone C · p4): `m[i] = m[i] + g[i] * scale`,
/// in place into `m`. `m` and `g` must be contiguous device f32 buffers of the
/// same length `n`. Bit-exact with `gpu_scalar_op_inplace(g, scale, MUL_SCALAR)`
/// then `gpu_elementwise_binary_inplace(m, g, ADD)` — one launch, no temp.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_scalar_mul_add_inplace_f32(m_ptr: i64, g_ptr: i64, scale: f32) {
    use crate::tensor::NslTensor;
    let m = unsafe { &*(m_ptr as *const NslTensor) };
    let g = unsafe { &*(g_ptr as *const NslTensor) };
    let n = m.len as usize;
    debug_assert_eq!(m.len, g.len, "scalar_mul_add_inplace length mismatch");

    let mut m_data = m.data as u64;
    let mut g_data = g.data as u64;
    let mut s_val = scale;
    let mut n_val = n as u64;
    let args = [
        &mut m_data as *mut _ as *mut std::ffi::c_void,
        &mut g_data as *mut _ as *mut std::ffi::c_void,
        &mut s_val as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        kernels::SCALAR_MUL_ADD_INPLACE_F32_PTX.as_ptr(),
        b"nsl_scalar_mul_add_inplace_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(
        result as u32, 0,
        "GPU scalar_mul_add_inplace kernel failed: {}", result as u32
    );
    inner::sync_after_kernel();
}

/// p9: fused per-parameter FASE-Deferred AdamW step — one launch for the whole
/// m/v/θ update (see `FASE_FUSED_ADAMW_STEP_F32_PTX`). All scalars are already
/// f32 (converted by the FFI with the same `as f32` every scalar op uses).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_fase_fused_adamw_step(
    theta_ptr: i64, m_ptr: i64, v_ptr: i64, mp_ptr: i64, n: usize,
    b1: f32, omb1: f32, b2: f32, omb2: f32, eps: f32,
    neg_lr: f32, neg_lr_wd: f32, bc1: f32, bc2: f32, has_wd: bool,
) {
    use crate::tensor::NslTensor;
    if n == 0 {
        return;
    }
    let th = unsafe { &*(theta_ptr as *const NslTensor) };
    let m = unsafe { &*(m_ptr as *const NslTensor) };
    let v = unsafe { &*(v_ptr as *const NslTensor) };
    let mp = unsafe { &*(mp_ptr as *const NslTensor) };
    let mut th_data = th.data as u64;
    let mut m_data = m.data as u64;
    let mut v_data = v.data as u64;
    let mut mp_data = mp.data as u64;
    let mut n_val = n as u64;
    let (mut b1, mut omb1, mut b2, mut omb2) = (b1, omb1, b2, omb2);
    let (mut eps, mut neg_lr, mut neg_lr_wd, mut bc1, mut bc2) =
        (eps, neg_lr, neg_lr_wd, bc1, bc2);
    let mut has_wd_val: u32 = u32::from(has_wd);
    let args = [
        &mut th_data as *mut _ as *mut std::ffi::c_void,
        &mut m_data as *mut _ as *mut std::ffi::c_void,
        &mut v_data as *mut _ as *mut std::ffi::c_void,
        &mut mp_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut b1 as *mut _ as *mut std::ffi::c_void,
        &mut omb1 as *mut _ as *mut std::ffi::c_void,
        &mut b2 as *mut _ as *mut std::ffi::c_void,
        &mut omb2 as *mut _ as *mut std::ffi::c_void,
        &mut eps as *mut _ as *mut std::ffi::c_void,
        &mut neg_lr as *mut _ as *mut std::ffi::c_void,
        &mut neg_lr_wd as *mut _ as *mut std::ffi::c_void,
        &mut bc1 as *mut _ as *mut std::ffi::c_void,
        &mut bc2 as *mut _ as *mut std::ffi::c_void,
        &mut has_wd_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        kernels::FASE_FUSED_ADAMW_STEP_F32_PTX.as_ptr(),
        b"nsl_fase_fused_adamw_step_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(
        result as u32, 0,
        "GPU fase_fused_adamw_step kernel failed: {}", result as u32
    );
    inner::sync_after_kernel();
}

/// Fusion-queue item 1: MULTI-TENSOR fused AdamW step — one launch for k
/// parameters via device pointer tables. `ptrs` slices carry raw DEVICE
/// DATA pointers (already resolved from NslTensors by the FFI); `lens` the
/// per-param element counts. Tables are staged through a persistent pinned
/// block on the COMPUTE stream (the muon_batch discipline: sync before the
/// host rewrite; enqueue uploads on current_stream so the kernel is
/// stream-ordered after them).
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_fase_fused_adamw_step_multi(
    t_ptrs: &[u64], m_ptrs: &[u64], v_ptrs: &[u64], mp_ptrs: &[u64], lens: &[u32],
    b1: f32, omb1: f32, b2: f32, omb2: f32, eps: f32,
    neg_lr: f32, neg_lr_wd: f32, bc1: f32, bc2: f32, has_wd: bool,
    mp_scale: f32,
) {
    let k = t_ptrs.len();
    if k == 0 {
        return;
    }
    assert!(
        k == m_ptrs.len() && k == v_ptrs.len() && k == mp_ptrs.len() && k == lens.len(),
        "multi adamw: table length mismatch"
    );
    // No grid.y cap any more: the grid is flat (item 8). The old
    // `assert!(k <= 65535)` guarded `grid.y`, which no longer exists.

    struct MultiWs {
        cap: usize,
        stage: u64,   // pinned host: 4*cap u64 + cap u32
        tabs: [u64; 4], // device u64 tables
        ntab: u64,    // device u32 table
        /// Item 8: block -> (param index, element base) tables, and the shape
        /// list they were built from. These depend ONLY on the shapes, which
        /// are static across training steps, so rebuilding them every step
        /// would be a pure per-step H2D cost (1.2 MB at Coder-50M). Cached and
        /// rebuilt only when `lens` changes.
        blk_lens: Vec<u32>,
        blk_param: u64,  // device u32[nblocks]
        blk_base: u64,   // device u32[nblocks]
        blk_count: usize,
    }
    thread_local! {
        static WS: std::cell::Cell<*mut MultiWs> = const { std::cell::Cell::new(std::ptr::null_mut()) };
    }

    inner::set_oom_context("fase_fused_adamw_multi");
    let ws: &mut MultiWs = WS.with(|c| {
        let cur = c.get();
        let need_new = cur.is_null() || unsafe { (*cur).cap } < k;
        if need_new {
            unsafe {
                inner::ensure_context();
                // Quiesce before releasing/rewriting anything a prior step
                // may still be reading.
                let r = cudarc::driver::sys::cuStreamSynchronize(inner::current_stream());
                assert_eq!(r, cudarc::driver::sys::CUresult::CUDA_SUCCESS);
                if !cur.is_null() {
                    let old = Box::from_raw(cur);
                    for t in old.tabs {
                        inner::free_managed(t as *mut c_void);
                    }
                    inner::free_managed(old.ntab as *mut c_void);
                    if old.blk_param != 0 {
                        inner::free_managed(old.blk_param as *mut c_void);
                        inner::free_managed(old.blk_base as *mut c_void);
                    }
                    cudarc::driver::sys::cuMemFreeHost(old.stage as *mut c_void);
                }
                let mut stage: *mut c_void = std::ptr::null_mut();
                let bytes = 4 * k * 8 + k * 4;
                let r = cudarc::driver::sys::cuMemAllocHost_v2(&mut stage, bytes.max(8));
                assert_eq!(
                    r,
                    cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                    "multi adamw: pinned staging alloc failed"
                );
                let tabs = [
                    inner::alloc_managed(k * 8) as u64,
                    inner::alloc_managed(k * 8) as u64,
                    inner::alloc_managed(k * 8) as u64,
                    inner::alloc_managed(k * 8) as u64,
                ];
                let ntab = inner::alloc_managed(k * 4) as u64;
                let fresh = Box::into_raw(Box::new(MultiWs {
                    cap: k,
                    stage: stage as u64,
                    tabs,
                    ntab,
                    blk_lens: Vec::new(),
                    blk_param: 0,
                    blk_base: 0,
                    blk_count: 0,
                }));
                c.set(fresh);
            }
        } else {
            // Same-cap reuse: the previous optimizer step's uploads read this
            // pinned block — quiesce before the host rewrite below.
            unsafe {
                inner::ensure_context();
                let r = cudarc::driver::sys::cuStreamSynchronize(inner::current_stream());
                assert_eq!(r, cudarc::driver::sys::CUresult::CUDA_SUCCESS);
            }
        }
        unsafe { &mut *c.get() }
    });

    unsafe {
        let base = ws.stage as *mut u64;
        std::ptr::copy_nonoverlapping(t_ptrs.as_ptr(), base, k);
        std::ptr::copy_nonoverlapping(m_ptrs.as_ptr(), base.add(ws.cap), k);
        std::ptr::copy_nonoverlapping(v_ptrs.as_ptr(), base.add(2 * ws.cap), k);
        std::ptr::copy_nonoverlapping(mp_ptrs.as_ptr(), base.add(3 * ws.cap), k);
        let nbase = (ws.stage as usize + 4 * ws.cap * 8) as *mut u32;
        std::ptr::copy_nonoverlapping(lens.as_ptr(), nbase, k);
        let up = |dst: u64, src_off: usize, bytes: usize| {
            let r = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dst,
                (ws.stage as usize + src_off) as *const c_void,
                bytes,
                inner::current_stream(),
            );
            assert_eq!(
                r,
                cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "multi adamw: table upload failed"
            );
        };
        for (idx, tab) in ws.tabs.iter().enumerate() {
            up(*tab, idx * ws.cap * 8, k * 8);
        }
        up(ws.ntab, 4 * ws.cap * 8, k * 4);
    }

    // Item 8: (re)build the block -> param / element-base tables when the
    // shape list changes. `build_block_tables` is a pure function of `lens`
    // and is unit-tested on the CPU (`fase_step::item8_tests`).
    let block = 256i64;
    if ws.blk_lens != lens {
        let (bparam, bbase) = crate::fase_step::build_block_tables(lens, block as u32);
        let nblocks = bparam.len();
        unsafe {
            inner::ensure_context();
            // The previous step's launch may still be reading these.
            let r = cudarc::driver::sys::cuStreamSynchronize(inner::current_stream());
            assert_eq!(r, cudarc::driver::sys::CUresult::CUDA_SUCCESS);
            if ws.blk_param != 0 {
                inner::free_managed(ws.blk_param as *mut c_void);
                inner::free_managed(ws.blk_base as *mut c_void);
            }
            ws.blk_param = inner::alloc_managed(nblocks * 4) as u64;
            ws.blk_base = inner::alloc_managed(nblocks * 4) as u64;
            let cp = |dst: u64, src: &[u32]| {
                let r = cudarc::driver::sys::cuMemcpyHtoD_v2(
                    dst,
                    src.as_ptr() as *const c_void,
                    src.len() * 4,
                );
                assert_eq!(
                    r,
                    cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                    "multi adamw: block table upload failed"
                );
            };
            cp(ws.blk_param, &bparam);
            cp(ws.blk_base, &bbase);
        }
        ws.blk_count = nblocks;
        ws.blk_lens = lens.to_vec();
    }

    let mut a0 = ws.tabs[0];
    let mut a1 = ws.tabs[1];
    let mut a2 = ws.tabs[2];
    let mut a3 = ws.tabs[3];
    let mut a4 = ws.ntab;
    let (mut b1, mut omb1, mut b2, mut omb2) = (b1, omb1, b2, omb2);
    let (mut eps, mut neg_lr, mut neg_lr_wd, mut bc1, mut bc2) =
        (eps, neg_lr, neg_lr_wd, bc1, bc2);
    let mut has_wd_val: u32 = u32::from(has_wd);
    let mut a5 = ws.blk_param;
    let mut a6 = ws.blk_base;
    let mut mp_scale_val = mp_scale;
    let args = [
        &mut a0 as *mut _ as *mut c_void,
        &mut a1 as *mut _ as *mut c_void,
        &mut a2 as *mut _ as *mut c_void,
        &mut a3 as *mut _ as *mut c_void,
        &mut a4 as *mut _ as *mut c_void,
        &mut b1 as *mut _ as *mut c_void,
        &mut omb1 as *mut _ as *mut c_void,
        &mut b2 as *mut _ as *mut c_void,
        &mut omb2 as *mut _ as *mut c_void,
        &mut eps as *mut _ as *mut c_void,
        &mut neg_lr as *mut _ as *mut c_void,
        &mut neg_lr_wd as *mut _ as *mut c_void,
        &mut bc1 as *mut _ as *mut c_void,
        &mut bc2 as *mut _ as *mut c_void,
        &mut has_wd_val as *mut _ as *mut c_void,
        &mut a5 as *mut _ as *mut c_void,
        &mut a6 as *mut _ as *mut c_void,
        &mut mp_scale_val as *mut _ as *mut c_void,
    ];
    let grid_x = ws.blk_count as i64;
    let result = inner::kernel_launch(
        kernels::FASE_FUSED_ADAMW_MULTI_F32_PTX.as_ptr(),
        b"nsl_fase_fused_adamw_multi_f32\0".as_ptr(),
        [grid_x, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(
        result as u32, 0,
        "GPU fase_fused_adamw_multi kernel failed: {}", result as u32
    );
    inner::sync_after_kernel();
}

/// Roadmap item 8, bf16-SR arm: the flat-grid MULTI variant of
/// `gpu_fase_fused_adamw_step_bf16sr`. One launch steps every bf16-mirrored
/// parameter in the bucket: `t_ptrs` are RAW bf16 mirror device pointers
/// (2 bytes/elem), `ctr_bases[i]` is param i's stable SR counter base
/// (`param_idx << SR_PARAM_SHIFT`, the SAME value its per-param launch would
/// pass), and `sr_key` is the per-step key — so every (param, element)
/// draws the identical dither the per-param loop draws, and the batched
/// step is bit-identical to the launches it replaces.
///
/// Same workspace discipline as `gpu_fase_fused_adamw_step_multi`: pinned
/// staging + device tables cached at capacity, block tables cached on the
/// shape list. A SEPARATE thread-local workspace — this one stages FIVE u64
/// tables (the extra one is ctrtab), so sharing the f32 workspace would
/// mis-offset every upload after the first.
///
/// No `mp_scale`: the per-param SR entry has no clip fold, and this arm
/// exists to replace exactly that entry. No m_partial zeroing either — the
/// FASE-Deferred lifecycle owns it on the SR path.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_fase_fused_adamw_step_bf16sr_multi(
    t_ptrs: &[u64], m_ptrs: &[u64], v_ptrs: &[u64], mp_ptrs: &[u64],
    lens: &[u32], ctr_bases: &[u64],
    b1: f32, omb1: f32, b2: f32, omb2: f32, eps: f32,
    neg_lr: f32, neg_lr_wd: f32, bc1: f32, bc2: f32, has_wd: bool,
    sr_key: u64,
) {
    let k = t_ptrs.len();
    if k == 0 {
        return;
    }
    assert!(
        k == m_ptrs.len() && k == v_ptrs.len() && k == mp_ptrs.len()
            && k == lens.len() && k == ctr_bases.len(),
        "multi bf16sr adamw: table length mismatch"
    );

    struct SrMultiWs {
        cap: usize,
        stage: u64,     // pinned host: 5*cap u64 + cap u32
        tabs: [u64; 5], // device u64 tables: theta, m, v, mp, ctr
        ntab: u64,      // device u32 table
        blk_lens: Vec<u32>,
        blk_param: u64, // device u32[nblocks]
        blk_base: u64,  // device u32[nblocks]
        blk_count: usize,
    }
    thread_local! {
        static SR_WS: std::cell::Cell<*mut SrMultiWs> =
            const { std::cell::Cell::new(std::ptr::null_mut()) };
    }

    inner::set_oom_context("fase_fused_adamw_multi_bf16sr");
    let ws: &mut SrMultiWs = SR_WS.with(|c| {
        let cur = c.get();
        let need_new = cur.is_null() || unsafe { (*cur).cap } < k;
        if need_new {
            unsafe {
                inner::ensure_context();
                // Quiesce before releasing/rewriting anything a prior step
                // may still be reading.
                let r = cudarc::driver::sys::cuStreamSynchronize(inner::current_stream());
                assert_eq!(r, cudarc::driver::sys::CUresult::CUDA_SUCCESS);
                if !cur.is_null() {
                    let old = Box::from_raw(cur);
                    for t in old.tabs {
                        inner::free_managed(t as *mut c_void);
                    }
                    inner::free_managed(old.ntab as *mut c_void);
                    if old.blk_param != 0 {
                        inner::free_managed(old.blk_param as *mut c_void);
                        inner::free_managed(old.blk_base as *mut c_void);
                    }
                    cudarc::driver::sys::cuMemFreeHost(old.stage as *mut c_void);
                }
                let mut stage: *mut c_void = std::ptr::null_mut();
                let bytes = 5 * k * 8 + k * 4;
                let r = cudarc::driver::sys::cuMemAllocHost_v2(&mut stage, bytes.max(8));
                assert_eq!(
                    r,
                    cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                    "multi bf16sr adamw: pinned staging alloc failed"
                );
                let tabs = [
                    inner::alloc_managed(k * 8) as u64,
                    inner::alloc_managed(k * 8) as u64,
                    inner::alloc_managed(k * 8) as u64,
                    inner::alloc_managed(k * 8) as u64,
                    inner::alloc_managed(k * 8) as u64,
                ];
                let ntab = inner::alloc_managed(k * 4) as u64;
                let fresh = Box::into_raw(Box::new(SrMultiWs {
                    cap: k,
                    stage: stage as u64,
                    tabs,
                    ntab,
                    blk_lens: Vec::new(),
                    blk_param: 0,
                    blk_base: 0,
                    blk_count: 0,
                }));
                c.set(fresh);
            }
        } else {
            // Same-cap reuse: the previous optimizer step's uploads read this
            // pinned block — quiesce before the host rewrite below.
            unsafe {
                inner::ensure_context();
                let r = cudarc::driver::sys::cuStreamSynchronize(inner::current_stream());
                assert_eq!(r, cudarc::driver::sys::CUresult::CUDA_SUCCESS);
            }
        }
        unsafe { &mut *c.get() }
    });

    unsafe {
        let base = ws.stage as *mut u64;
        std::ptr::copy_nonoverlapping(t_ptrs.as_ptr(), base, k);
        std::ptr::copy_nonoverlapping(m_ptrs.as_ptr(), base.add(ws.cap), k);
        std::ptr::copy_nonoverlapping(v_ptrs.as_ptr(), base.add(2 * ws.cap), k);
        std::ptr::copy_nonoverlapping(mp_ptrs.as_ptr(), base.add(3 * ws.cap), k);
        std::ptr::copy_nonoverlapping(ctr_bases.as_ptr(), base.add(4 * ws.cap), k);
        let nbase = (ws.stage as usize + 5 * ws.cap * 8) as *mut u32;
        std::ptr::copy_nonoverlapping(lens.as_ptr(), nbase, k);
        let up = |dst: u64, src_off: usize, bytes: usize| {
            let r = cudarc::driver::sys::cuMemcpyHtoDAsync_v2(
                dst,
                (ws.stage as usize + src_off) as *const c_void,
                bytes,
                inner::current_stream(),
            );
            assert_eq!(
                r,
                cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                "multi bf16sr adamw: table upload failed"
            );
        };
        for (idx, tab) in ws.tabs.iter().enumerate() {
            up(*tab, idx * ws.cap * 8, k * 8);
        }
        up(ws.ntab, 5 * ws.cap * 8, k * 4);
    }

    // Block tables: same pure function of `lens`, same cache-on-shape-list
    // policy as the f32 multi. The `blockDim.x == build_block_tables block`
    // contract is shared — one constant feeds both.
    let block = 256i64;
    if ws.blk_lens != lens {
        let (bparam, bbase) = crate::fase_step::build_block_tables(lens, block as u32);
        let nblocks = bparam.len();
        unsafe {
            inner::ensure_context();
            // The previous step's launch may still be reading these.
            let r = cudarc::driver::sys::cuStreamSynchronize(inner::current_stream());
            assert_eq!(r, cudarc::driver::sys::CUresult::CUDA_SUCCESS);
            if ws.blk_param != 0 {
                inner::free_managed(ws.blk_param as *mut c_void);
                inner::free_managed(ws.blk_base as *mut c_void);
            }
            ws.blk_param = inner::alloc_managed(nblocks * 4) as u64;
            ws.blk_base = inner::alloc_managed(nblocks * 4) as u64;
            let cp = |dst: u64, src: &[u32]| {
                let r = cudarc::driver::sys::cuMemcpyHtoD_v2(
                    dst,
                    src.as_ptr() as *const c_void,
                    src.len() * 4,
                );
                assert_eq!(
                    r,
                    cudarc::driver::sys::CUresult::CUDA_SUCCESS,
                    "multi bf16sr adamw: block table upload failed"
                );
            };
            cp(ws.blk_param, &bparam);
            cp(ws.blk_base, &bbase);
        }
        ws.blk_count = nblocks;
        ws.blk_lens = lens.to_vec();
    }

    let mut a0 = ws.tabs[0];
    let mut a1 = ws.tabs[1];
    let mut a2 = ws.tabs[2];
    let mut a3 = ws.tabs[3];
    let mut a4 = ws.ntab;
    let (mut b1, mut omb1, mut b2, mut omb2) = (b1, omb1, b2, omb2);
    let (mut eps, mut neg_lr, mut neg_lr_wd, mut bc1, mut bc2) =
        (eps, neg_lr, neg_lr_wd, bc1, bc2);
    let mut has_wd_val: u32 = u32::from(has_wd);
    let mut sr_key_val = sr_key;
    let mut a5 = ws.tabs[4];
    let mut a6 = ws.blk_param;
    let mut a7 = ws.blk_base;
    let args = [
        &mut a0 as *mut _ as *mut c_void,
        &mut a1 as *mut _ as *mut c_void,
        &mut a2 as *mut _ as *mut c_void,
        &mut a3 as *mut _ as *mut c_void,
        &mut a4 as *mut _ as *mut c_void,
        &mut b1 as *mut _ as *mut c_void,
        &mut omb1 as *mut _ as *mut c_void,
        &mut b2 as *mut _ as *mut c_void,
        &mut omb2 as *mut _ as *mut c_void,
        &mut eps as *mut _ as *mut c_void,
        &mut neg_lr as *mut _ as *mut c_void,
        &mut neg_lr_wd as *mut _ as *mut c_void,
        &mut bc1 as *mut _ as *mut c_void,
        &mut bc2 as *mut _ as *mut c_void,
        &mut has_wd_val as *mut _ as *mut c_void,
        &mut sr_key_val as *mut _ as *mut c_void,
        &mut a5 as *mut _ as *mut c_void,
        &mut a6 as *mut _ as *mut c_void,
        &mut a7 as *mut _ as *mut c_void,
    ];
    let grid_x = ws.blk_count as i64;
    let result = inner::kernel_launch(
        kernels::FASE_FUSED_ADAMW_MULTI_BF16SR_PTX.as_ptr(),
        b"nsl_fase_fused_adamw_multi_bf16sr\0".as_ptr(),
        [grid_x, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(
        result as u32, 0,
        "GPU fase_fused_adamw_multi_bf16sr kernel failed: {}", result as u32
    );
    inner::sync_after_kernel();
}

/// P4 item 17: fused AdamW step against a BF16 AUTHORITATIVE theta with
/// counter-based stochastic rounding. `theta_dev` is the RAW device pointer
/// of the bf16 mirror (2 bytes/elem, not an NslTensor); m/v/mp are the f32
/// moment/accumulator tensors. `sr_key` = seed ^ (step * SR_STEP_SALT),
/// `sr_ctr_base` = stable param index << SR_PARAM_SHIFT — precomputed by the
/// caller so the kernel stays a pure function of its counters.
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)]
pub(crate) fn gpu_fase_fused_adamw_step_bf16sr(
    theta_dev: u64, m_ptr: i64, v_ptr: i64, mp_ptr: i64, n: usize,
    b1: f32, omb1: f32, b2: f32, omb2: f32, eps: f32,
    neg_lr: f32, neg_lr_wd: f32, bc1: f32, bc2: f32, has_wd: bool,
    sr_key: u64, sr_ctr_base: u64,
) {
    use crate::tensor::NslTensor;
    if n == 0 {
        return;
    }
    let m = unsafe { &*(m_ptr as *const NslTensor) };
    let v = unsafe { &*(v_ptr as *const NslTensor) };
    let mp = unsafe { &*(mp_ptr as *const NslTensor) };
    let mut th_data = theta_dev;
    let mut m_data = m.data as u64;
    let mut v_data = v.data as u64;
    let mut mp_data = mp.data as u64;
    let mut n_val = n as u64;
    let (mut b1, mut omb1, mut b2, mut omb2) = (b1, omb1, b2, omb2);
    let (mut eps, mut neg_lr, mut neg_lr_wd, mut bc1, mut bc2) =
        (eps, neg_lr, neg_lr_wd, bc1, bc2);
    let mut has_wd_val: u32 = u32::from(has_wd);
    let (mut key_val, mut ctr_val) = (sr_key, sr_ctr_base);
    let args = [
        &mut th_data as *mut _ as *mut std::ffi::c_void,
        &mut m_data as *mut _ as *mut std::ffi::c_void,
        &mut v_data as *mut _ as *mut std::ffi::c_void,
        &mut mp_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut b1 as *mut _ as *mut std::ffi::c_void,
        &mut omb1 as *mut _ as *mut std::ffi::c_void,
        &mut b2 as *mut _ as *mut std::ffi::c_void,
        &mut omb2 as *mut _ as *mut std::ffi::c_void,
        &mut eps as *mut _ as *mut std::ffi::c_void,
        &mut neg_lr as *mut _ as *mut std::ffi::c_void,
        &mut neg_lr_wd as *mut _ as *mut std::ffi::c_void,
        &mut bc1 as *mut _ as *mut std::ffi::c_void,
        &mut bc2 as *mut _ as *mut std::ffi::c_void,
        &mut has_wd_val as *mut _ as *mut std::ffi::c_void,
        &mut key_val as *mut _ as *mut std::ffi::c_void,
        &mut ctr_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        kernels::FASE_FUSED_ADAMW_STEP_BF16SR_PTX.as_ptr(),
        b"nsl_fase_fused_adamw_step_bf16sr\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(
        result as u32, 0,
        "GPU fase_fused_adamw_step_bf16sr kernel failed: {}", result as u32
    );
    inner::sync_after_kernel();
}

/// P4 item 17: SR-BF16 rounding-tail probe over raw device buffers — parity
/// gate hook only (see `SR_BF16_ROUND_PROBE_PTX`).
#[cfg(feature = "cuda")]
pub(crate) fn gpu_sr_bf16_round_probe(
    src_f32_dev: u64, dst_bf16_dev: u64, n: usize, sr_key: u64, sr_ctr_base: u64,
) {
    if n == 0 {
        return;
    }
    let mut src = src_f32_dev;
    let mut dst = dst_bf16_dev;
    let mut n_val = n as u64;
    let (mut key_val, mut ctr_val) = (sr_key, sr_ctr_base);
    let args = [
        &mut src as *mut _ as *mut std::ffi::c_void,
        &mut dst as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
        &mut key_val as *mut _ as *mut std::ffi::c_void,
        &mut ctr_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        kernels::SR_BF16_ROUND_PROBE_PTX.as_ptr(),
        b"nsl_sr_bf16_round_probe\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(
        result as u32, 0,
        "GPU sr_bf16_round_probe kernel failed: {}", result as u32
    );
    inner::sync_after_kernel();
}

/// P4 item 17: raw-buffer precision casts between a bf16 mirror and an f32
/// working view (grid-stride kernels from `precision_cast_kernels`). Both
/// pointers are raw device allocations; `n` is the element count.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_cast_raw_f32_to_bf16(src_f32_dev: u64, dst_bf16_dev: u64, n: usize) {
    gpu_cast_raw(
        precision_cast_kernels::PTX_F32_TO_BF16.as_ptr(),
        precision_cast_kernels::KNAME_F32_TO_BF16.as_ptr(),
        src_f32_dev, dst_bf16_dev, n,
    );
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_cast_raw_bf16_to_f32(src_bf16_dev: u64, dst_f32_dev: u64, n: usize) {
    gpu_cast_raw(
        precision_cast_kernels::PTX_BF16_TO_F32.as_ptr(),
        precision_cast_kernels::KNAME_BF16_TO_F32.as_ptr(),
        src_bf16_dev, dst_f32_dev, n,
    );
}

#[cfg(feature = "cuda")]
fn gpu_cast_raw(ptx: *const u8, kname: *const u8, src: u64, dst: u64, n: usize) {
    if n == 0 {
        return;
    }
    let mut src_val = src;
    let mut dst_val = dst;
    let mut n_val = n as u64;
    let args = [
        &mut src_val as *mut _ as *mut std::ffi::c_void,
        &mut dst_val as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = (((n as i64) + block - 1) / block).min(4096);
    let result = inner::kernel_launch(ptx, kname, [grid, 1, 1], [block, 1, 1], &args, 0);
    assert_eq!(result as u32, 0, "GPU raw precision cast failed: {}", result as u32);
    inner::sync_after_kernel();
}

/// `NSL_MATMUL_NO_BATCH_COLLAPSE=1` — restore the pre-item-9 dispatch, keeping
/// every batched product on the naive `nsl_bmm_f32` kernel.
///
/// One switch covers BOTH item-9 paths — the collapse of `[..,m,k] @ [k,n]`
/// into a single sgemm and the routing of genuinely batched products to
/// `cublasGemmStridedBatchedEx`. They are separate dispatch decisions but a
/// single escape hatch is what an operator actually wants: "put the matmul
/// back the way it was" while a suspected numerical regression is bisected.
///
/// This exists so the parity gate has a reference arm that is the OLD
/// numerics, and so a suspected cuBLAS-tiling difference in a training run can
/// be bisected without a rebuild. The env var is read once — `gpu_matmul_f32`
/// is on the hot path and a per-call `std::env::var` would show up in the
/// profile — but the resolved answer lives in an atomic rather than a
/// `OnceLock<bool>` so `test_set_batch_collapse_disabled` can flip it. Without
/// that, comparing the two arms would need two test BINARIES (the isolation
/// dance `matmul_cublas_tf32_default_sanity.rs` documents), and a parity gate
/// split across two processes cannot diff the two results against each other.
///
/// 0 = unresolved, 1 = collapse enabled, 2 = collapse disabled.
#[cfg(feature = "cuda")]
static BATCH_COLLAPSE_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(feature = "cuda")]
fn batch_collapse_disabled() -> bool {
    use std::sync::atomic::Ordering;
    match BATCH_COLLAPSE_STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            let disabled = std::env::var("NSL_MATMUL_NO_BATCH_COLLAPSE").as_deref() == Ok("1");
            BATCH_COLLAPSE_STATE.store(u8::from(disabled) + 1, Ordering::Relaxed);
            disabled
        }
    }
}

/// Test hook: force the batch-collapse decision for the rest of the process.
///
/// Gated on `test-hooks` so production builds cannot reach it. Used by
/// `tests/matmul_batch_collapse.rs` to drive both dispatch arms from one
/// process and compare their outputs element-wise.
#[cfg(all(feature = "cuda", feature = "test-hooks"))]
pub fn test_set_batch_collapse_disabled(disabled: bool) {
    BATCH_COLLAPSE_STATE.store(u8::from(disabled) + 1, std::sync::atomic::Ordering::Relaxed);
}

/// Is `t` a 2-D view whose buffer is a contiguous row-major `[cols, rows]`,
/// i.e. exactly what `.transpose(0, 1)` on a 2-D tensor produces?
///
/// Such an operand needs no materialisation: cuBLAS reads it directly with
/// `CUBLAS_OP_T`. `x @ self.embed.transpose(0, 1)` — the weight-tied LM head
/// every model in `models/` uses — otherwise copies a whole `[vocab, d_model]`
/// matrix on every forward (96 MiB for Coder-50M's 49152 x 512).
///
/// Deliberately conservative:
/// * `is_contiguous()` is checked FIRST. A `[k, 1]` contiguous tensor has
///   strides `[1, 1]` and would satisfy the pattern below when `k == 1`;
///   treating a contiguous operand as transposed is a silent transpose of the
///   result.
/// * Zero strides are rejected. `expand` produces them (shape_ops.rs), and a
///   broadcast operand is not a transpose — `OP_T` would read one row as if
///   it were the whole matrix.
/// * `ndim == 2` only. A permuted 4-D attention tensor is not expressible as
///   `OP_T`, and pretending otherwise reads the wrong elements in bounds.
#[cfg(feature = "cuda")]
fn is_transposed_2d_view(t: &crate::tensor::NslTensor) -> bool {
    if t.ndim != 2 || t.is_contiguous() {
        return false;
    }
    let shape = unsafe { std::slice::from_raw_parts(t.shape, 2) };
    let strides = unsafe { std::slice::from_raw_parts(t.strides, 2) };
    shape[0] > 0
        && shape[1] > 0
        && strides[0] == 1
        && strides[1] == shape[0]
}

/// `NSL_MATMUL_TRANSPOSE_VIEWS` — hand a 2-D transposed operand to cuBLAS as
/// `OP_T` instead of materialising it. **The default is coupled to the math
/// mode: ON under TF32 (the shipped default), OFF under FP32 cores and
/// Pedantic.** `=1`/`=0` always win; see the update at the end for why.
///
/// The argument for defaulting this ON is seductive and false. Materialising
/// `embed.transpose(0,1)` for the weight-tied LM head costs a 25,165,824-element
/// strided copy on every forward — 96 MiB, in a kernel doing a software 64-bit
/// div/rem per element — and cuBLAS reads a transposed operand natively, so the
/// copy looks like pure waste. It is not waste: it buys a much faster GEMM.
///
/// Measured on an RTX 5070 Ti / CUDA 13.3, f32, per call:
///
/// | shape | `OP_T` | copy + `OP_N` | verdict |
/// |---|---|---|---|
/// | `[2048,512] @ [512,49152]` (Coder-50M LM head) | 5.92 ms | 1.04 + 2.89 = **3.93 ms** | OP_T **1.51x SLOWER** |
/// | `[2048,512] @ [512,4096]` | 0.344 ms | 0.084 + 0.267 = 0.351 ms | a wash |
/// | `[512,512] @ [512,512]` | **0.023 ms** | 0.024 + 0.019 = 0.043 ms | OP_T 1.9x faster |
///
/// End to end on a Coder-50M forward: 29 ms/forward with the copy, 33 ms
/// without it, peak GPU 2.10 GB vs 2.01 GB. So this trades ~90 MB of peak
/// memory for ~4 ms per forward — worth having when memory is the binding
/// constraint, and a regression otherwise.
///
/// It is NOT defaulted on with a shape heuristic because three data points is
/// not a cost model. This repo already learned that lesson: item 10's autotune
/// database was ranking variants by a roofline ESTIMATE with no measurement
/// behind it, and no consumer at all. A heuristic here would be the same
/// mistake with fewer points.
///
/// UPDATE (2026-07-28, after TF32 became the matmul default): the committed
/// reproducer (`matmul_transposed_operand::the_op_t_tradeoff_is_remeasurable`)
/// shows the table above is a property of FP32-core math, not of the shapes.
/// Under TF32, cuBLAS's tensor-core kernels read a transposed operand well
/// and OP_T measured FASTER across the whole grid — 2.16 vs 3.30 ms on the
/// LM head (0.65x), 0.72x and 0.46x on the other shapes.
///
/// The flip was then made on TWO levels of measurement plus gates, not the
/// grid alone: a Coder-50M 20-forward loop (second-half kernel sums, three
/// paired runs) went 63.4 -> 56.6 ms end-to-end (1.12x) with sgemm at
/// parity and the win exactly the vanished 96 MiB LM-head copy, on top of
/// the ~90 MB peak-memory saving the copy always cost. Correctness under
/// TF32 is gated by `matmul_dispatch_under_tf32::op_t_exemption_is_correct_
/// under_tf32` (with a kernel-launch path witness, because at small K both
/// arms can produce bit-identical values).
///
/// The default is per-math-mode because each arm won only in its own
/// measured cell: FP32 cores keep the copy (OP_T 1.40x SLOWER on the LM
/// head per the committed reproducer — the table above says 1.51x because
/// it is the ORIGINAL 2026-07-27 hand grid; same cell, different runs),
/// Pedantic keeps the copy (unmeasured). That is a decision per measured
/// cell — NOT the shape heuristic the paragraph above warns about, which
/// would interpolate into cells nobody measured.
/// 0 = unresolved, 1 = off, 2 = on.
#[cfg(feature = "cuda")]
static TRANSPOSE_VIEWS_STATE: std::sync::atomic::AtomicU8 = std::sync::atomic::AtomicU8::new(0);

#[cfg(feature = "cuda")]
fn transpose_views_enabled() -> bool {
    use std::sync::atomic::Ordering;
    match TRANSPOSE_VIEWS_STATE.load(Ordering::Relaxed) {
        1 => false,
        2 => true,
        _ => {
            // Tri-state like NSL_MATMUL_TF32: only the literal "1"/"0" are
            // honoured, so a typo'd value cannot silently change dispatch.
            // The DEFAULT is coupled to the math mode, because each arm is
            // measured only in its own cell (2026-07-28 grid + Coder-50M
            // end-to-end): under TF32, OP_T won every shape per-call
            // (0.65x on the LM head) and the 20-forward model loop
            // (63.4 -> 56.6 ms, three paired runs); under FP32 cores the
            // materialising copy won the wide shapes (OP_T 1.40x slower on
            // the LM head), and Pedantic is unmeasured — both keep the
            // copy. This is a decision per measured cell, not a shape
            // heuristic; the env var always wins over the coupling.
            let on = match std::env::var("NSL_MATMUL_TRANSPOSE_VIEWS").ok().as_deref() {
                Some("1") => true,
                Some("0") => false,
                // Bf16 joins Tf32 in the OP_T cell: measured 2026-07-29 on
                // the Coder-50M training step (mfu_bench, batch=1 accum=8)
                // under NSL_MATMUL_BF16=1 — OP_T 61.96 ms/micro vs the
                // materialising copy's 65.46 ms. The bf16-storage cast is
                // layout-agnostic (it casts the dense buffer either way), so
                // the copy arm buys nothing the cast did not already pay.
                // FP32 cores / Pedantic keep the copy, as before, each per
                // its own measured (or unmeasured) cell.
                _ => matches!(
                    cublas_inner::resolved_math_mode(),
                    cublas_inner::CublasMathMode::Tf32 | cublas_inner::CublasMathMode::Bf16
                ),
            };
            TRANSPOSE_VIEWS_STATE.store(u8::from(on) + 1, Ordering::Relaxed);
            on
        }
    }
}

/// Test hook: force the transposed-view decision for the rest of the process.
/// The gates need BOTH arms in one binary — the copy arm must keep
/// materialising correctly, and the OP_T arm must stay numerically correct —
/// regardless of what the math-mode coupling would have resolved.
#[cfg(all(feature = "cuda", feature = "test-hooks"))]
pub fn test_set_transpose_views(enabled: bool) {
    TRANSPOSE_VIEWS_STATE.store(u8::from(enabled) + 1, std::sync::atomic::Ordering::Relaxed);
}

/// Can `gpu_matmul_f32` consume this operand WITHOUT materialising it?
///
/// `is_a` selects which operand is being asked about. THE SAME function
/// answers for the caller (`nsl_tensor_matmul`, deciding whether to skip
/// `nsl_tensor_contiguous`) and for the dispatcher (deciding whether to pass
/// `CUBLAS_OP_T`). Two copies of this predicate that disagreed by one
/// condition would either materialise pointlessly — harmless — or hand a
/// strided buffer to a stride-blind kernel, which is PR #335 again. There is
/// one copy, and `matmul_operand_exemption_agrees_with_dispatch` pins that.
///
/// Returns true ONLY for an operand that is already contiguous (nothing to do)
/// or is a 2-D transposed view that the 2-D/collapse arm will express as
/// `OP_T`. Every other arm requires contiguous operands, so this returns false
/// for them and the caller copies exactly as it always did.
#[cfg(feature = "cuda")]
pub(crate) fn matmul_operand_needs_no_copy(a_ptr: i64, b_ptr: i64, is_a: bool) -> bool {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };
    let me = if is_a { a } else { b };

    if me.is_contiguous() {
        return true; // `nsl_tensor_contiguous` would only bump a refcount.
    }
    // MATH-MODE-COUPLED DEFAULT (2026-07-28): ON under TF32, OFF under FP32
    // cores and Pedantic; the env literal always wins. Each arm is measured
    // in exactly its own cell — see `transpose_views_enabled`. Do NOT reason
    // downstream from "strided operands are always materialised": under the
    // shipped defaults every stdlib Linear's `x @ w.transpose(0,1)` takes
    // OP_T.
    if !transpose_views_enabled() {
        return false;
    }
    // Beyond here the operand IS strided, so it may only be skipped if the
    // dispatch will genuinely express it.
    if a.ndim < 2 || b.ndim < 2 || a.dtype != 1 || b.dtype != 1 {
        return false;
    }
    if !is_transposed_2d_view(me) {
        return false;
    }

    // Re-derive the output batch extent exactly as `gpu_matmul_f32` does; a
    // transposed A is only expressible when there is no batch to flatten.
    let a_shape = unsafe { std::slice::from_raw_parts(a.shape, a.ndim as usize) };
    let b_shape = unsafe { std::slice::from_raw_parts(b.shape, b.ndim as usize) };
    let a_batch = &a_shape[..a.ndim as usize - 2];
    let b_batch = &b_shape[..b.ndim as usize - 2];
    let nd = a_batch.len().max(b_batch.len());
    let mut total_batch: i64 = 1;
    for i in 0..nd {
        let ad = if i < nd - a_batch.len() { 1 } else { a_batch[i - (nd - a_batch.len())] };
        let bd = if i < nd - b_batch.len() { 1 } else { b_batch[i - (nd - b_batch.len())] };
        if ad != bd && ad != 1 && bd != 1 {
            return false; // shape error; let the dispatcher report it
        }
        total_batch = total_batch.saturating_mul(ad.max(bd));
    }
    let total_batch = total_batch.max(1) as u64;

    if is_a {
        // The collapse flattens A's leading dims into the row count, which
        // needs A's slices contiguous and adjacent. A transposed A is neither,
        // so it is only expressible in the plain 2-D case.
        total_batch == 1
    } else {
        // B is 2-D in both the plain and collapse cases. In the collapse case
        // it must still carry no batch extent of its own, and the collapse
        // must actually be enabled.
        let b_total_batch: u64 = b_batch.iter().product::<i64>().max(1) as u64;
        if total_batch == 1 {
            true
        } else {
            b_total_batch == 1
                && a.is_contiguous()
                && total_batch.saturating_mul(a_shape[a.ndim as usize - 2] as u64)
                    <= i32::MAX as u64
                && !batch_collapse_disabled()
        }
    }
}

/// GPU matrix multiplication: C[M,N] = A[M,K] @ B[K,N], f32 inputs.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_matmul_f32(a_ptr: i64, b_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    inner::set_oom_context("matmul_f32");
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };

    assert!(a.ndim >= 2 && b.ndim >= 2, "matmul requires 2D+ tensors");

    // This whole function is f32-only and says so nowhere else: it sizes the
    // output `alloc_managed(out_total * 4)`, hardcodes the output dtype to 1,
    // and casts `a.data as *const f32`. A bf16 or f16 tensor carries the same
    // element COUNT in half the bytes, so it would be read two-for-one past
    // the end of its allocation — a live out-of-bounds read that returns
    // plausible numbers rather than crashing.
    //
    // Refuse instead of silently mis-reading. Item 9's "mixed precision"
    // bullet is exactly the work that would make this dispatch real; until
    // then, saying so is the only honest option.
    assert!(
        a.dtype == crate::tensor::DTYPE_F32 && b.dtype == crate::tensor::DTYPE_F32,
        "GPU matmul is f32-only (dtype 1); got dtype {} @ dtype {}. A non-f32 \
         tensor here would be read as f32 — for a 2-byte dtype that is a \
         read of twice the allocation, returning plausible wrong numbers. \
         Cast the operands with `.to(f32)` first. (Mixed-precision GEMM \
         dispatch is roadmap item 9 and is not implemented.)",
        a.dtype,
        b.dtype
    );

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

    // PARTIAL batch broadcast is not representable by the single stride above,
    // and the GPU path has always got it wrong. `[2,3,m,k] @ [2,1,k,n]` gives
    // b_total_batch = 2, so stride_b = k*n — but the product has 6 slices and
    // B holds 2, so slices 2..6 read past the end of B's allocation. Measured
    // error against a CPU reference: 2.61 relative to the result's own RMS,
    // BIT-IDENTICAL through the naive kernel and through cuBLAS. The CPU path
    // (`tensor/arithmetic.rs`) walks per-dimension broadcast strides and is
    // correct, so this is a live CPU/GPU divergence.
    //
    // Fixing it means per-dimension strides here, which is a real change with
    // its own gates. Until then this refuses rather than returning plausible
    // garbage from an out-of-bounds read — a wrong number that propagates into
    // a training run is worse than a stopped process.
    let partial_broadcast = |total: u64| total != 1 && total != total_batch;
    assert!(
        !partial_broadcast(a_total_batch) && !partial_broadcast(b_total_batch),
        "GPU matmul does not support PARTIAL batch broadcasting: A batch \
         extent {a_total_batch}, B batch extent {b_total_batch}, output batch \
         {total_batch} (shapes {a_shape:?} @ {b_shape:?}). Each operand must \
         be either fully broadcast (extent 1) or fully present (extent \
         {total_batch}). The CPU path handles this correctly; the GPU path \
         would read past the end of the smaller operand. Expand the operand \
         explicitly, or run this product on CPU."
    );

    // Item 9 phase 1: collapse `[.., m, k] @ [k, n]` to ONE 2-D sgemm.
    //
    // This is the shape of every transformer projection (`x @ w` with
    // `x: [batch, seq, d_model]`), and until this existed it took the
    // `total_batch > 1` arm below — the naive 16x16 scalar `nsl_bmm_f32`.
    // Measured on an RTX 5070 Ti at `[8,1024,512] @ [512,512]`:
    // **0.67 TFLOP/s through the batched kernel vs 29.54 through cuBLAS for
    // the identical math** written as `[8192,512] @ [512,512]`. A
    // `GroupedQueryAttention` forward profiled six `nsl_bmm_f32` and zero
    // cuBLAS calls.
    //
    // The collapse is a pure REINTERPRETATION, not a reassociation: when B is
    // shared across the batch and A's slices are contiguous and adjacent, A's
    // buffer already IS a row-major `(total_batch*m) x k` matrix and C's
    // already IS `(total_batch*m) x n`. Every output element sums the same k
    // products in the same index order, so this changes nothing about which
    // reduction happens — only which kernel performs it. (cuBLAS's tiling
    // still orders that reduction differently from the naive kernel, so
    // results DO move; see the parity gate in
    // `crates/nsl-runtime/tests/matmul_batch_collapse.rs`.)
    //
    // Preconditions, each checked rather than assumed:
    //   * B carries no batch extent (`b_total_batch == 1`), so one B serves
    //     every slice. Combined with `total_batch > 1` this forces
    //     `a_total_batch == total_batch`, hence `stride_a == a_mat_stride`.
    //   * A is contiguous. The sole caller (`nsl_tensor_matmul`) materialises
    //     both operands first, but a strided A would make the flattening a
    //     silently-wrong read — the same class of bug as the tied-embedding
    //     miscompile in PR #335.
    //   * The collapsed row count fits the i32 cuBLAS dimension.
    //
    // `NSL_MATMUL_NO_BATCH_COLLAPSE=1` forces the old kernel; the parity gate
    // uses it as its reference arm.
    // Item 9 phase 2: a 2-D TRANSPOSED operand goes to cuBLAS as `OP_T`
    // instead of being materialised.
    //
    // `nsl_tensor_matmul` used to copy both operands contiguous before every
    // GPU dispatch — correct and necessary when the target was a stride-blind
    // PTX kernel (PR #335, where a stride-blind read of
    // `emb @ embed.transpose(0,1)` silently produced the wrong product and the
    // GPU loss climbed to the uniform plateau while CPU descended). Once the
    // target is cuBLAS that copy buys nothing: `CUBLAS_OP_T` reads the
    // original buffer.
    //
    // What it costs today, measured on Coder-50M: the weight-tied LM head at
    // models/coder50m/model.nsl:80 is `x @ self.embed.transpose(0, 1)` with
    // `embed: [49152, 512]`, so every forward runs one 25,165,824-element
    // strided copy — 96 MiB in and 96 MiB out, through a kernel that does a
    // software 64-bit div/rem per element.
    //
    // Eligibility is per-arm, because only the 2-D/collapse arm reaches
    // `sgemm_row_major_t`:
    //   * A may be transposed only when `total_batch == 1`. The collapse
    //     flattens A's leading dims into the row count, which requires A's
    //     slices to be contiguous and adjacent — a transposed A is neither.
    //   * B may be transposed in both cases: it is 2-D either way.
    // The batched and naive arms below stay strictly contiguous-only.
    // Same predicate the caller used to decide not to copy — one function, so
    // the exemption and the dispatch cannot drift apart.
    let a_2d_transposed = !a.is_contiguous() && matmul_operand_needs_no_copy(a_ptr, b_ptr, true);
    let b_2d_transposed = !b.is_contiguous() && matmul_operand_needs_no_copy(a_ptr, b_ptr, false);

    let collapse_rows = total_batch.saturating_mul(m);
    let collapse_to_2d = total_batch > 1
        && b_total_batch == 1
        && a.is_contiguous()
        // B: contiguous, or a 2-D transpose we can express as OP_T. The doc
        // above says every precondition is checked rather than assumed, and
        // for one commit this one was assumed — the strided-batched arm below
        // checked both operands while this arm checked only A.
        && (b.is_contiguous() || b_2d_transposed)
        && collapse_rows <= i32::MAX as u64
        && !batch_collapse_disabled();

    // The 2-D arm is only reachable with a strided operand when that operand
    // is one we can express; anything else must still have been materialised
    // by the caller.
    let two_d_arm = total_batch == 1
        && (a.is_contiguous() || a_2d_transposed)
        && (b.is_contiguous() || b_2d_transposed);

    if two_d_arm || collapse_to_2d {
        // The only thing the collapse changes is the row count handed to
        // cuBLAS: `k`, `n` and all three data pointers are already correct.
        let m = if collapse_to_2d { collapse_rows } else { m };
        // Non-batched f32 matmul: dispatch to cuBLAS sgemm via the row-major
        // operand-swap idiom (spec 2026-04-21 §2.1). Replaces the naive
        // nsl_matmul_f32 PTX kernel (~1-2 TFLOPs/s on a 5070 Ti) with
        // cuBLAS's architecture-specialized kernels (~25-30 TFLOPs/s peak).
        //
        // Profiler-event wrapping mirrors `inner::kernel_launch`: pop a
        // (start, stop) event pair, record start before launch, record stop
        // after launch, push a trace with a cuBLAS-pattern name so the §3
        // presence assertion in matmul_cublas_equivalence.rs observes
        // "sgemm" / "gemm_" substrings.
        inner::ensure_context();

        let profiler_events = if crate::kernel_profiler::kernel_profiler_enabled() {
            crate::kernel_profiler::kernel_profiler_pop_events()
        } else {
            None
        };

        if let Some((start, _, _)) = &profiler_events {
            unsafe {
                cudarc::driver::sys::cuEventRecord(
                    *start as cudarc::driver::sys::CUevent,
                    inner::current_stream(),
                );
            }
        }

        // SAFETY: a.data, b.data, out_data are valid device f32 pointers of
        // sizes m*k, k*n, m*n respectively (verified by the shape derivation
        // above + the caller contract of gpu_matmul_f32). A transposed operand
        // spans the same element count in the same buffer — only the read
        // order differs, which is what OP_T expresses.
        let res = unsafe {
            cublas_inner::sgemm_row_major_t(
                a.data as *const f32,
                b.data as *const f32,
                out_data as *mut f32,
                m, n, k,
                a_2d_transposed,
                b_2d_transposed,
                // Plain matmul: overwrite a freshly-allocated output.
                1.0, 0.0,
            )
        };
        if let Err(e) = res {
            eprintln!("[nsl-matmul] cuBLAS sgemm failed ({}x{}x{}): {:?}", m, n, k, e);
            // Match existing failure convention: null pointer so callers can
            // detect the failure (existing kernel path used assert_eq!; this
            // is a safer non-panicking failure mode).
            return 0;
        }

        // In sync mode, surface async errors the same way PTX kernels do.
        if inner::sync_mode_enabled() {
            let sync_result = unsafe { cudarc::driver::sys::cuCtxSynchronize() };
            if sync_result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                panic!(
                    "[nsl] CUDA async error after cuBLAS sgemm ({}x{}x{}): {:?}",
                    m, n, k, sync_result
                );
            }
        }

        if let Some((_, stop, _)) = &profiler_events {
            unsafe {
                cudarc::driver::sys::cuEventRecord(
                    *stop as cudarc::driver::sys::CUevent,
                    inner::current_stream(),
                );
            }
            // Name picked to satisfy spec §3's `contains("sgemm") || contains("gemm_")`
            // presence assertion. The exact arch-dispatched cuBLAS kernel name
            // (e.g. `ampere_sgemm_128x64_nn`) is not visible from the public
            // cuBLAS API, so we emit a synthetic marker that the profile parser
            // can still match.
            crate::kernel_profiler::kernel_profiler_push_trace(
                "sgemm_cublas",
                [1, 1, 1],
                [1, 1, 1],
            );
        }
    } else if !batch_collapse_disabled()
        && total_batch <= i32::MAX as u64
        && m <= i32::MAX as u64
        && n <= i32::MAX as u64
        && k <= i32::MAX as u64
        && a.is_contiguous()
        && b.is_contiguous()
    {
        // Item 9: a genuinely batched product — both operands carry a batch
        // extent, or A is broadcast — goes to `cublasGemmStridedBatchedEx`.
        // This is what remains after the collapse above: in a transformer it
        // is QK^T and PV, which the collapse cannot touch because B differs
        // per slice.
        //
        // Unlike the collapse this is NOT a reinterpretation — it is a
        // different kernel computing the same contraction, so summation order
        // changes exactly as it did for the 2-D cuBLAS swap.
        //
        // Contiguity is required because the strides handed to cuBLAS are the
        // natural per-slice extents; a strided view would need its real
        // strides, and passing the natural ones would read the wrong memory
        // rather than fail. The caller materialises both operands, so this is
        // defence in depth.
        inner::ensure_context();

        // Profiler-event wrapping mirrors the 2-D arm above, and must: a bare
        // `kernel_profiler_push_trace` indexes the event pool at
        // `pool_cursor - 1`, so without a popped pair it points one before the
        // start of the pool and the trace is dropped at flush.
        let profiler_events = if crate::kernel_profiler::kernel_profiler_enabled() {
            crate::kernel_profiler::kernel_profiler_pop_events()
        } else {
            None
        };
        if let Some((start, _, _)) = &profiler_events {
            unsafe {
                cudarc::driver::sys::cuEventRecord(
                    *start as cudarc::driver::sys::CUevent,
                    inner::current_stream(),
                );
            }
        }

        let res = unsafe {
            cublas_inner::sgemm_batched_row_major(
                a.data as *const f32,
                b.data as *const f32,
                out_data as *mut f32,
                m, n, k,
                total_batch, stride_a, stride_b, stride_c,
                1.0, 0.0,
            )
        };
        if let Err(e) = res {
            eprintln!(
                "[nsl-matmul] cuBLAS batched gemm failed ({total_batch}x{m}x{n}x{k}): {e:?}"
            );
            // Release the output we allocated above before signalling failure.
            // The 2-D arm's `return 0` does NOT do this and leaks the buffer,
            // its shape and its strides on every cuBLAS error; that is
            // pre-existing and left alone here rather than folded into an
            // unrelated change, but a new `return 0` should not add a second
            // instance of it.
            crate::tensor::nsl_tensor_free(out_ptr as i64);
            return 0;
        }
        if inner::sync_mode_enabled() {
            let sync_result = unsafe { cudarc::driver::sys::cuCtxSynchronize() };
            if sync_result != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                panic!(
                    "[nsl] CUDA async error after cuBLAS batched gemm \
                     ({total_batch}x{m}x{n}x{k}): {sync_result:?}"
                );
            }
        }
        if let Some((_, stop, _)) = &profiler_events {
            unsafe {
                cudarc::driver::sys::cuEventRecord(
                    *stop as cudarc::driver::sys::CUevent,
                    inner::current_stream(),
                );
            }
            // Same synthetic-marker convention as the 2-D arm: the real
            // arch-dispatched cuBLAS kernel name is not visible from the
            // public API. Named distinctly so a profile can tell the two
            // cuBLAS paths apart — and so the "did this collapse?" assertion
            // in the gates can be specific rather than matching any gemm.
            crate::kernel_profiler::kernel_profiler_push_trace(
                "sgemm_cublas_batched",
                [1, 1, 1],
                [1, 1, 1],
            );
        }
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
    inner::sync_after_kernel();

    out_ptr as i64
}

/// GPU scalar op (tensor op scalar).
#[cfg(feature = "cuda")]
pub(crate) fn gpu_scalar_op(a_ptr: i64, scalar: f32, ptx: &str, kernel_name: &str) -> i64 {
    use crate::tensor::NslTensor;
    inner::set_oom_context(kernel_name.trim_end_matches('\0'));
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    assert_gpu_f32(a, kernel_name, "input");
    // Flat row-major kernel — materialize strided views first (see
    // gpu_elementwise_unary; same stride-blindness class as the
    // tied-embedding matmul bug).
    if !a.is_contiguous() {
        let a_c = crate::tensor::nsl_tensor_contiguous(a_ptr);
        let result = gpu_scalar_op(a_c, scalar, ptx, kernel_name);
        crate::tensor::nsl_tensor_free(a_c);
        return result;
    }
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
    inner::sync_after_kernel();
    out_ptr as i64
}

// === GPU backward op helpers ===

/// GPU backward binary op: takes grad tensor and a saved tensor, produces output of same shape as grad.
/// P5 item 20 slice B: ternary elementwise backward launch
/// (grad, up, input) -> out, same conventions as `gpu_backward_binary`.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_backward_ternary(
    a_ptr: i64,
    b_ptr: i64,
    c_ptr: i64,
    ptx: &str,
    kernel_name: &str,
) -> i64 {
    use crate::tensor::NslTensor;
    inner::set_oom_context(kernel_name.trim_end_matches('\0'));
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };
    let c = unsafe { &*(c_ptr as *const NslTensor) };
    assert_gpu_f32(a, kernel_name, "operand 0");
    assert_gpu_f32(b, kernel_name, "operand 1");
    assert_gpu_f32(c, kernel_name, "operand 2");
    assert!(
        a.len == b.len && a.len == c.len,
        "GPU ternary backward: length mismatch ({}, {}, {})",
        a.len, b.len, c.len
    );

    let n = a.len as usize;
    let out_data = inner::alloc_managed(n * 4); // f32
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);
    let out = Box::new(NslTensor::new(
        out_data, shape, strides, a.ndim, a.len, a.device, 1, 1, 0,
    ));
    let out_ptr = Box::into_raw(out);
    let out_t = unsafe { &*out_ptr };

    let mut a_data = a.data as u64;
    let mut b_data = b.data as u64;
    let mut c_data = c.data as u64;
    let mut o_data = out_t.data as u64;
    let mut n_val = n as u64;
    let args = [
        &mut a_data as *mut _ as *mut std::ffi::c_void,
        &mut b_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut o_data as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid, 1, 1], [block, 1, 1], &args, 0,
    );
    assert_eq!(
        result as u32, 0,
        "GPU ternary backward kernel '{}' failed: {}",
        kernel_name.trim_end_matches('\0'),
        result as u32
    );
    inner::sync_after_kernel();
    out_ptr as i64
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_backward_binary(a_ptr: i64, b_ptr: i64, ptx: &str, kernel_name: &str) -> i64 {
    use crate::tensor::NslTensor;
    inner::set_oom_context(kernel_name.trim_end_matches('\0'));
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    let b = unsafe { &*(b_ptr as *const NslTensor) };
    assert_gpu_f32(a, kernel_name, "grad");
    assert_gpu_f32(b, kernel_name, "saved tensor");
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
    inner::sync_after_kernel();
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
    assert_gpu_f32(a, "clamp_f32", "input");
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
    inner::sync_after_kernel();
    out_ptr as i64
}

/// GPU clamp forward in-place: writes clamp result back to input buffer.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_clamp_f32_inplace(a_ptr: i64, lo: f32, hi: f32) {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    assert_gpu_f32(a, "clamp_f32_inplace", "input");
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
    inner::sync_after_kernel();
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_clamp_backward(grad: i64, input: i64, min_val: f32, max_val: f32) -> i64 {
    use crate::tensor::NslTensor;
    let a = unsafe { &*(grad as *const NslTensor) };
    let b = unsafe { &*(input as *const NslTensor) };
    assert_gpu_f32(a, "clamp_backward_f32", "grad");
    assert_gpu_f32(b, "clamp_backward_f32", "input");
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
    inner::sync_after_kernel();
    out_ptr as i64
}

// === FFI exports ===

/// GPU fused log-softmax: (x - max) - log(sum(exp(x - max))) in a single kernel.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_log_softmax_f32(tensor_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    use fused_kernels::LOG_SOFTMAX_F32_PTX;

    let t = NslTensor::from_ptr(tensor_ptr);
    assert_gpu_f32(t, "log_softmax_f32", "input");
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
    inner::sync_after_kernel();
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

/// Marketing name of CUDA device 0 with the vendor/brand prefixes stripped
/// (e.g. "RTX 5070 Ti"), for `nsl-codegen`'s GPU-database lookup
/// (`gpu_specs::find_gpu` normalizes spaces to dashes). Non-panicking by
/// design: compiling on a GPU-less machine (or without the cuda feature)
/// returns `None` and the caller falls back to its default spec — this is
/// probed at COMPILE time, where an abort would kill the compiler.
#[cfg(feature = "cuda")]
pub fn cuda_device_name() -> Option<String> {
    // Fully self-contained probe (review finding): `inner::state()` /
    // `device_name_stripped()` route through the asserting lazy init
    // (cuDevicePrimaryCtxRetain can fail even when cuInit succeeded —
    // exclusive-mode devices, ECC-pending, OOM), and a panic here fires
    // at COMPILE time inside the CSHA planner. Every driver call below is
    // rc-checked; no context is created or retained (cuDeviceGetName
    // needs only a device ordinal).
    use cudarc::driver::sys::*;
    unsafe {
        if cuInit(0) != CUresult::CUDA_SUCCESS {
            return None;
        }
        let mut count: i32 = 0;
        if cuDeviceGetCount(&mut count) != CUresult::CUDA_SUCCESS || count == 0 {
            return None;
        }
        // Name the device this process will actually BIND (NSL_CUDA_DEVICE /
        // spawner striping), not unconditionally ordinal 0 — on a mixed-GPU
        // node the planner would otherwise calibrate rank 1 against rank 0's
        // card. A bad override fails rc-checked into the None fallback.
        let ordinal = inner::select_device_ordinal();
        let mut device: CUdevice = 0;
        if cuDeviceGet(&mut device, ordinal) != CUresult::CUDA_SUCCESS {
            return None;
        }
        device_marketing_name(device)
    }
}

/// `cuDeviceGetName` for an already-resolved device, with the vendor/brand
/// prefixes stripped. Shared by `cuda_device_name` and `cuda_device_identity`
/// so the two can never disagree about what the card is called.
///
/// # Safety
/// `device` must be a valid `CUdevice` obtained from `cuDeviceGet`.
#[cfg(feature = "cuda")]
unsafe fn device_marketing_name(device: cudarc::driver::sys::CUdevice) -> Option<String> {
    use cudarc::driver::sys::*;
    let mut buf = [0i8; 128];
    if cuDeviceGetName(buf.as_mut_ptr(), buf.len() as i32, device) != CUresult::CUDA_SUCCESS {
        return None;
    }
    let cstr = std::ffi::CStr::from_ptr(buf.as_ptr());
    let mut name = cstr.to_str().ok()?.trim().to_string();
    for prefix in ["NVIDIA ", "GeForce ", "Tesla "] {
        if let Some(rest) = name.strip_prefix(prefix) {
            name = rest.to_string();
        }
    }
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// Non-cuda build: no device to name.
#[cfg(not(feature = "cuda"))]
pub fn cuda_device_name() -> Option<String> {
    None
}

/// Identity of the local CUDA device, as reported by the DRIVER.
///
/// This is the cache-key identity for `@autotune` (roadmap item 10). It is
/// deliberately independent of `nsl-codegen`'s `GpuSpec` database: a tuning
/// result is only transferable to hardware that reports the same values here,
/// and that has to be true even for a card the database has never heard of.
/// Reading identity out of the database instead would collapse every unknown
/// GPU onto whatever the database default happens to be.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CudaDeviceIdentity {
    /// Marketing name with vendor prefixes stripped, e.g. "RTX 5070 Ti".
    pub name: String,
    /// Compute capability major * 10 + minor, e.g. 120 for sm_120.
    pub sm_version: u32,
    /// Multiprocessor count.
    pub sm_count: u32,
    /// `cuDriverGetVersion`, e.g. 13030 for CUDA 13.3.
    pub driver_version: u32,
}

/// Probe the local CUDA device for its cache-key identity.
///
/// Non-panicking for the same reason `cuda_device_name` is: this runs at
/// COMPILE time, where an abort would kill the compiler. Every driver call is
/// rc-checked and no context is created or retained.
#[cfg(feature = "cuda")]
pub fn cuda_device_identity() -> Option<CudaDeviceIdentity> {
    use cudarc::driver::sys::*;
    unsafe {
        if cuInit(0) != CUresult::CUDA_SUCCESS {
            return None;
        }
        // Resolve the ordinal ONCE and derive both the name and the attributes
        // from it. Calling `cuda_device_name()` here instead would run
        // `select_device_ordinal` a second time, which under the SPMD spawner
        // prints its "binding CUDA device N" line twice per probe and, worse,
        // lets the name and the attributes resolve to different ordinals if the
        // environment changes in between.
        let ordinal = inner::select_device_ordinal();
        let mut device: CUdevice = 0;
        if cuDeviceGet(&mut device, ordinal) != CUresult::CUDA_SUCCESS {
            return None;
        }
        let name = device_marketing_name(device)?;
        let attr = |a| -> Option<u32> {
            let mut v: i32 = 0;
            if cuDeviceGetAttribute(&mut v, a, device) != CUresult::CUDA_SUCCESS || v < 0 {
                return None;
            }
            Some(v as u32)
        };
        let major = attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)?;
        let minor = attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)?;
        let sm_count = attr(CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)?;
        let mut driver: i32 = 0;
        if cuDriverGetVersion(&mut driver) != CUresult::CUDA_SUCCESS || driver < 0 {
            return None;
        }
        Some(CudaDeviceIdentity {
            name,
            sm_version: major * 10 + minor,
            sm_count,
            driver_version: driver as u32,
        })
    }
}

/// Non-cuda build: no device to identify.
#[cfg(not(feature = "cuda"))]
pub fn cuda_device_identity() -> Option<CudaDeviceIdentity> {
    None
}

/// Whether this binary was compiled with CUDA support at all.
///
/// `cuda` is NOT a default feature (`nsl-cli` and `nsl-codegen` both default to
/// `[]`, and the release workflow builds with no features), so the stock `nsl`
/// binary cannot probe a device even on a machine that has one. Callers that
/// record or key on device identity must be able to say *why* they have none —
/// "this machine has no GPU" and "this compiler cannot see GPUs" are different
/// claims, and conflating them would put `no-cuda-device` in a cache record
/// written on a box with a 5070 Ti sitting in it.
pub const CUDA_SUPPORT_COMPILED: bool = cfg!(feature = "cuda");

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

/// Launch a user `kernel` block whose arguments are NslTensor handles.
///
/// This is the codegen entry point for user kernel calls (e.g.
/// `vec_add(ga, gb, gc, grid=4, block=256)`). Cranelift lowers each tensor
/// argument to its NslTensor *handle* (a host pointer to the struct), but
/// `cuLaunchKernel` needs an array of pointers to the device *data* addresses.
/// This wrapper reads the handle array, extracts each tensor's `.data` (the
/// device pointer), and builds the correct kernel-parameter indirection —
/// unlike [`nsl_kernel_launch`], which takes an already-marshaled parameter
/// array and is used by hand-written launchers/tests.
///
/// `args_ptr` points to an array of `num_args` NslTensor handles (each i64).
/// Returns the `CUresult` code (0 on success).
#[no_mangle]
pub extern "C" fn nsl_kernel_launch_tensors(
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
        use crate::tensor::NslTensor;
        let n = num_args as usize;
        // from_raw_parts requires a non-null pointer even for len 0.
        let handles: &[i64] = if n == 0 {
            &[]
        } else {
            unsafe { std::slice::from_raw_parts(args_ptr as *const i64, n) }
        };

        // Extract each tensor's device data pointer into a stable Vec, then
        // build the kernelParams array where each entry points at the u64
        // device address. Both Vecs must outlive the launch call (cuLaunchKernel
        // reads the argument values synchronously at launch time).
        let mut device_ptrs: Vec<u64> = Vec::with_capacity(n);
        for (i, &handle) in handles.iter().enumerate() {
            let tensor = unsafe { &*(handle as *const NslTensor) };
            // Kernel arguments must be GPU-resident. Passing a CPU tensor here
            // would feed a host pointer to the kernel as a device address and
            // crash with an opaque CUDA_ERROR_ILLEGAL_ADDRESS — refuse loudly
            // with an actionable message instead.
            if tensor.device == 0 {
                eprintln!(
                    "nsl: kernel argument {} is a CPU tensor; kernel arguments must be \
                     moved to the GPU first (e.g. `arg.to(cuda)`).",
                    i
                );
                std::process::abort();
            }
            device_ptrs.push(tensor.data as u64);
        }
        let mut kernel_params: Vec<*mut c_void> = Vec::with_capacity(n);
        for slot in &device_ptrs {
            kernel_params.push(slot as *const u64 as *mut c_void);
        }

        let result = inner::kernel_launch(
            ptx_ptr as *const u8,
            name_ptr as *const u8,
            [grid_x, grid_y, grid_z],
            [block_x, block_y, block_z],
            &kernel_params,
            shared_mem_bytes as u32,
        );
        result as i64
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ptx_ptr, name_ptr, grid_x, grid_y, grid_z);
        let _ = (block_x, block_y, block_z, args_ptr, num_args, shared_mem_bytes);
        eprintln!(
            "nsl: this program launches a GPU `kernel` block, but the nsl runtime \
             was built without CUDA support. Rebuild the toolchain with \
             `--features cuda` (e.g. `cargo build -p nsl-cli --features cuda`) to run \
             GPU kernels."
        );
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
    // Weight only. `indices` is dtype-dispatched on its own further down
    // (DTYPE_I32 vs the i64 default), so it is deliberately not routed
    // through the f32 guard.
    assert_gpu_f32(weight, "embedding_lookup", "weight");

    let vocab_size = unsafe { *weight.shape.add(0) } as u64;
    let embed_dim = unsafe { *weight.shape.add(1) } as u64;
    let seq_len = unsafe { *indices.shape.add(0) } as u64;
    // Host-resident indices are bounds-checked by `nsl_tensor_embedding_lookup`
    // before it dispatches here (it does NOT fall through to the CPU loop on
    // this arm — an earlier comment here claimed otherwise and was wrong).
    // Indices that are ALREADY device-resident remain unchecked: validating
    // them needs a D2H copy plus a sync per lookup.
    let _ = vocab_size;

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
    let (ptx, kernel_name): (&str, &[u8]) = if indices_gpu.dtype == crate::tensor::DTYPE_I32 {
        (fused_kernels::EMBEDDING_I32IDX_PTX, b"nsl_embedding_i32idx\0")
    } else {
        (fused_kernels::EMBEDDING_F32_PTX, b"nsl_embedding_f32\0")
    };

    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid_x, grid_y, 1], [block_x, block_y, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU embedding kernel failed: {:?}", result);
    inner::sync_after_kernel();

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

/// GPU embedding backward: scatter-add grad rows into a zeroed
/// `[vocab, embed]` f32 buffer on the device (scaling campaign item 1).
///
/// `grad` must be a GPU f32 contiguous `[seq_len, embed_dim]` tensor;
/// `indices` a GPU tensor of `seq_len` token ids (f32 dtype=1 or i32
/// DTYPE_I32, mirroring `gpu_embedding_lookup`'s kernel pair). Out-of-range
/// and negative ids are skipped in-kernel, matching the CPU reference.
/// Returns a published GPU `[vocab, embed]` NslTensor, or 0 when the
/// index dtype has no kernel (caller falls back to the host scatter).
///
/// The scatter uses f32 atomics, so the per-row accumulation order is
/// nondeterministic across runs (same policy as the flash phase-2
/// backward); the caller exposes NSL_EMBEDDING_BWD_CPU=1 to restore the
/// deterministic host path.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_embedding_backward(
    grad_ptr: i64,
    indices_ptr: i64,
    vocab_size: u64,
    embed_dim: u64,
    seq_len: u64,
) -> i64 {
    inner::set_oom_context("embedding_backward");
    use crate::tensor::NslTensor;

    let grad = unsafe { &*(grad_ptr as *const NslTensor) };
    let indices = unsafe { &*(indices_ptr as *const NslTensor) };
    assert_gpu_f32(grad, "embedding_backward", "grad");

    // A2 / M46: under deterministic mode, use the per-output-row kernel
    // (no atomics -> bit-identical to the CPU reference, run-to-run stable).
    // The atomicAdd variant stays the production default (deterministic
    // kernel is O(vocab*embed*seq)). `--deterministic`/nsl_set_deterministic
    // sets this; NSL_EMBEDDING_BWD_CPU handled by the caller (host scatter).
    let deterministic = crate::deterministic_ops::is_deterministic();
    let (ptx, kernel_name): (&str, &[u8]) = match (deterministic, indices.dtype) {
        (true, crate::tensor::DTYPE_I32) => (
            fused_kernels::EMBEDDING_BWD_DET_I32IDX_PTX,
            b"nsl_embedding_bwd_det_i32idx\0",
        ),
        (true, 1) => (
            fused_kernels::EMBEDDING_BWD_DET_F32_PTX,
            b"nsl_embedding_bwd_det_f32\0",
        ),
        (false, crate::tensor::DTYPE_I32) => (
            fused_kernels::EMBEDDING_BWD_I32IDX_PTX,
            b"nsl_embedding_bwd_i32idx\0",
        ),
        (false, 1) => (
            fused_kernels::EMBEDDING_BWD_F32_PTX,
            b"nsl_embedding_bwd_f32\0",
        ),
        _ => return 0,
    };

    let out_elems = (vocab_size * embed_dim) as usize;
    let out_data = inner::alloc_managed(out_elems * 4);
    inner::memset_d8(out_data, out_elems * 4);

    let mut g_data = grad.data as u64;
    let mut i_data = indices.data as u64;
    let mut o_data = out_data as u64;
    let mut seq_val = seq_len;
    let mut emb_val = embed_dim;
    let mut vocab_val = vocab_size;

    let args: [*mut std::ffi::c_void; 6] = [
        &mut g_data as *mut _ as *mut std::ffi::c_void,
        &mut i_data as *mut _ as *mut std::ffi::c_void,
        &mut o_data as *mut _ as *mut std::ffi::c_void,
        &mut seq_val as *mut _ as *mut std::ffi::c_void,
        &mut emb_val as *mut _ as *mut std::ffi::c_void,
        &mut vocab_val as *mut _ as *mut std::ffi::c_void,
    ];

    let block_x = 16i64;
    let block_y = 16i64;
    // Deterministic kernel maps one thread per OUTPUT element (vocab x embed);
    // the atomicAdd variant maps one thread per INPUT element (seq x embed).
    let dim_x = if deterministic { vocab_size } else { seq_len } as i64;
    let grid_x = (dim_x + block_x - 1) / block_x;
    let grid_y = ((embed_dim as i64) + block_y - 1) / block_y;

    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid_x, grid_y, 1], [block_x, block_y, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU embedding_bwd kernel failed: {:?}", result);
    inner::sync_after_kernel();

    let out_shape = crate::memory::checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = vocab_size as i64;
        *out_shape.add(1) = embed_dim as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 2);
    let out = Box::new(NslTensor::new(
        out_data,
        out_shape,
        out_strides,
        2,
        out_elems as i64,
        grad.device,
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
    assert_gpu_f32(tensor, "bias_add", "input");

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
    // Guarded AFTER the transfer, not before: a host-resident bias is
    // legitimately f64 (CPU model params are f64) and
    // `nsl_tensor_to_device` converts f64 -> f32 on the way to the
    // device. It is the buffer the kernel actually reads that must be
    // f32, and a 16-bit dtype survives that transfer verbatim.
    assert_gpu_f32(bias_gpu, "bias_add", "bias");

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
    inner::sync_after_kernel();

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
    assert_gpu_f32(t, "softmax_f32", "input");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(t, "sum_dim_f32", "input");
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

    // A block per output element only pays off when the reduced axis is long
    // enough to keep 256 threads busy. Below that the block is almost entirely
    // idle: a tied [49152, 512] embedding gradient reduces 2 elements per
    // output and launched 25,165,824 blocks, which profiling showed to be the
    // single largest consumer of GPU time in a training step. Give each output
    // element one thread instead.
    // Default 1: a reduction over a single element is a copy, so the fast path
    // returns the identical value for that length.
    //
    // This is not a corner case. Instrumenting a Coder-50M step showed that
    // almost all `sum_dim` work IS reduce_size == 1 — including 64 launches of
    // `reduce=1 inner=25165824`, i.e. copying the 49152x512 tied embedding
    // gradient by launching 25,165,824 blocks of 256 threads: 6.4 billion
    // threads to move 25 million floats. Kernel profiling attributed 82.5% of
    // ALL GPU time in the step to `nsl_sum_dim_f32`, and the reduce_size == 1
    // shapes are ~89% of that.
    //
    // Above 1 the two kernels genuinely disagree, because this one accumulates
    // sequentially while the general one tree-reduces pairwise in shared
    // memory; keep the default at 1 unless that reassociation is acceptable.
    //
    // Note that even at 1 a run's loss trajectory can differ from the general
    // kernel in the last couple of significant digits. That is NOT this kernel
    // reassociating — it is `nsl_embedding_bwd_f32` accumulating the embedding
    // gradient with `red.global.add.f32`, whose ordering depends on execution
    // timing. Changing the grid from 25M blocks to 98K blocks changes that
    // timing.
    //
    // Do NOT assume `--deterministic` protects a raised threshold. That flag
    // substitutes `gpu_det_sum_dim_f32` in CODEGEN, for `sum` written in NSL
    // source only; four runtime-internal callers reach this function directly
    // and check nothing — `autodiff/grad_utils.rs` and `autodiff/backward.rs`
    // (broadcast reduce-to-shape), `tensor/ad_ops.rs`, and
    // `flash_attention.rs` (GQA dK/dV, where reduce_size is 1 at group size 1).
    // At the default of 1 those paths are bit-exact, so they are safe; above 1
    // the reassociation would leak into determinism-gated runs through them.
    //
    // NSL_SUM_DIM_SHORT_MAX overrides the threshold; 0 disables the fast path.
    static SHORT_AXIS_MAX: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    let short_axis_max = *SHORT_AXIS_MAX.get_or_init(|| {
        std::env::var("NSL_SUM_DIM_SHORT_MAX")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
    });
    static LOG_SHAPES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *LOG_SHAPES.get_or_init(|| {
        std::env::var("NSL_SUM_DIM_LOG").ok().as_deref() == Some("1")
    }) {
        eprintln!(
            "[sum_dim] reduce={reduce_size} outer={outer} inner={inner} out_total={out_total} fast={}",
            reduce_size <= short_axis_max
        );
    }
    // Second admission route: a COALESCED-layout reduction.
    //
    // The general kernel walks the reduced axis with thread t reading
    // `base + t*inner`, so when `inner > 1` consecutive threads land `inner`
    // elements apart and every load pulls its own cache line. The short kernel
    // assigns one thread per OUTPUT element, so consecutive threads take
    // consecutive `inner_idx` and the loads coalesce. Kernel profiling put
    // `nsl_sum_dim_f32` at 18.3% of GPU time on shapes like
    // `reduce=1024 outer=1 inner=512`; routing those to the coalesced kernel
    // measured 100 -> 83 ms per micro-batch, a 20% step-level win.
    //
    // This reassociates: the general kernel tree-reduces 256 partials pairwise
    // while this accumulates sequentially, so unlike the reduce_size <= 1 route
    // it is NOT bit-exact against the general kernel.
    //
    // It therefore has to check determinism itself. The claim that
    // `--deterministic` runs "never reach here" is false, and the block above
    // says why: that flag only substitutes `gpu_det_sum_dim_f32` in CODEGEN for
    // `sum` written in NSL source, while `autodiff/grad_utils.rs`,
    // `autodiff/backward.rs`, `tensor/ad_ops.rs` and `flash_attention.rs` call
    // `nsl_tensor_sum_dim` -> `gpu_sum_dim_f32` directly. A broadcast
    // reduce-to-shape over dim 0 of [B, S, D] has inner = S*D, far above the
    // threshold, so without this check every such gradient silently lost
    // run-to-run reproducibility under M46.
    //
    // NSL_SUM_DIM_COALESCED=0 opts out; NSL_SUM_DIM_SHORT_MAX=0 disables BOTH
    // routes, since it is documented as the kill switch for this kernel and a
    // bisection that only half-worked would exonerate the wrong code.
    const COALESCE_MIN_INNER: u64 = 32;
    static COALESCED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    let coalesced_ok = short_axis_max > 0
        && *COALESCED.get_or_init(|| {
            std::env::var("NSL_SUM_DIM_COALESCED").ok().as_deref() != Some("0")
        })
        && inner >= COALESCE_MIN_INNER
        && !crate::deterministic_ops::is_deterministic();

    let result = if (short_axis_max > 0 && reduce_size <= short_axis_max) || coalesced_ok {
        let grid = (out_total as i64 + block - 1) / block;
        inner::kernel_launch(
            fused_kernels::SUM_DIM_SHORT_F32_PTX.as_ptr(),
            b"nsl_sum_dim_short_f32\0".as_ptr(),
            [grid, 1, 1],
            [block, 1, 1],
            &args,
            0,
        )
    } else {
        inner::kernel_launch(
            SUM_DIM_F32_PTX.as_ptr(),
            b"nsl_sum_dim_f32\0".as_ptr(),
            [out_total as i64, 1, 1],
            [block, 1, 1],
            &args,
            256 * 4,
        )
    };
    assert_eq!(result as u32, 0, "GPU sum_dim kernel failed: {:?}", result);
    inner::sync_after_kernel();

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
    assert_gpu_f32(t, "global_sum_f32", "input");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(t, "tensor_stats_f32", "input");
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

/// Sum of squares for many tensors with a SINGLE synchronization.
///
/// `gpu_tensor_sum_sq_f32` synchronizes and reads back per call. Gradient
/// clipping needs the norm over every gradient, so calling it per tensor forced
/// one full pipeline drain per parameter — 57 for Coder-50M, every step. Here
/// each tensor's stats land in its own slot of one device buffer, and the host
/// synchronizes and reads once.
///
/// Every tensor must be device-resident f32 AND dense (`data[0..len]` is the
/// whole tensor) — this reads raw, so the caller validates layout.
///
/// Returns the total Sum-of-squares across `tensors`.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_sum_sq_many_f32(tensors: &[i64]) -> f64 {
    use crate::tensor::NslTensor;
    use fused_kernels::SUM_SQ_F64_ACC_F32_PTX;

    let live: Vec<i64> = tensors
        .iter()
        .copied()
        .filter(|&p| p != 0 && NslTensor::from_ptr(p).len > 0)
        .collect();
    if live.is_empty() {
        return 0.0;
    }
    for &p in &live {
        assert_gpu_f32(NslTensor::from_ptr(p), "sum_sq_many_f32", "grad tensor");
    }

    // `nsl_sum_sq_f64_acc_f32`, not slot 3 of `nsl_tensor_stats_f32`: the stats
    // kernel accumulates in f32, and the clip decision is a comparison against
    // `max_norm`, so its drift is not confined to the low digits of the result —
    // it can flip whether clipping happens at all.
    //
    // Every tensor gets `MAX_BLOCKS` f64 slots and the whole set costs ONE drain.
    // The kernel is grid-strided with one partial per block because f64 is 1/64
    // rate on GeForce: a single 256-thread block over the 25M-element embedding
    // gradient measured 80 -> 112 ms per micro-batch. Per-block slots rather than
    // atomics — an atomic variant was tried earlier and measured slower (80 ->
    // 83..85 ms) on top of making the summation order nondeterministic.
    //
    // The grid is derived from `len` alone, so a given shape always produces the
    // same number of partials summed in the same order: reproducible run to run.
    // NSL_SUM_SQ_BLOCKS overrides the block cap; 1 reproduces the single-block
    // version, which is the bisection point if a norm ever looks wrong. It is
    // also the measurement: 1 block costs 80 -> 112 ms per micro-batch, 256
    // costs nothing detectable, and both accumulate in f64.
    const BLOCK: i64 = 256;
    static MAX_BLOCKS_ENV: std::sync::OnceLock<i64> = std::sync::OnceLock::new();
    let max_blocks = *MAX_BLOCKS_ENV.get_or_init(|| {
        std::env::var("NSL_SUM_SQ_BLOCKS")
            .ok()
            .and_then(|v| v.parse::<i64>().ok())
            .filter(|v| *v >= 1)
            .unwrap_or(256)
    });
    let grid_for = |len: i64| -> i64 { ((len + BLOCK - 1) / BLOCK).clamp(1, max_blocks) };

    let out_data = inner::alloc_managed(live.len() * max_blocks as usize * 8);

    for (i, &ptr) in live.iter().enumerate() {
        let t = NslTensor::from_ptr(ptr);
        let mut in_data = t.data as u64;
        let mut out_slot = (out_data as u64) + (i * max_blocks as usize * 8) as u64;
        let mut n_val = t.len as u64;
        let args: [*mut std::ffi::c_void; 3] = [
            &mut in_data as *mut _ as *mut std::ffi::c_void,
            &mut out_slot as *mut _ as *mut std::ffi::c_void,
            &mut n_val as *mut _ as *mut std::ffi::c_void,
        ];
        let rc = inner::kernel_launch(
            SUM_SQ_F64_ACC_F32_PTX.as_ptr(),
            b"nsl_sum_sq_f64_acc_f32\0".as_ptr(),
            [grid_for(t.len), 1, 1],
            [BLOCK, 1, 1],
            &args,
            (BLOCK * 8) as u32,
        );
        assert_eq!(rc as u32, 0, "GPU sum_sq kernel failed: {:?}", rc);
    }

    unsafe {
        cudarc::driver::sys::cuCtxSynchronize();
    }
    let mut raw = vec![0f64; live.len() * max_blocks as usize];
    inner::memcpy_dtoh(
        raw.as_mut_ptr() as *mut std::ffi::c_void,
        out_data as *const std::ffi::c_void,
        raw.len() * 8,
    );
    inner::free_managed(out_data);

    // Sum only the slots the launch actually wrote; the rest are uninitialized.
    live.iter()
        .enumerate()
        .map(|(i, &ptr)| {
            let used = grid_for(NslTensor::from_ptr(ptr).len) as usize;
            let base = i * max_blocks as usize;
            raw[base..base + used].iter().sum::<f64>()
        })
        .sum()
}

/// GPU sum of squares (Σx²) for an f32 tensor — the RAW `sum_sq` accumulator
/// from the stats kernel, WITHOUT the mean/std post-processing that
/// `gpu_tensor_stats_f32` applies to its slot 3. Kept separate precisely
/// because that helper's public contract is `[min, max, mean, std]` (the
/// trace debugger depends on the std), so its slot 3 is NOT Σx². Used by
/// `nsl_tensor_sum_sq`'s GPU fast path — reading `gpu_tensor_stats_f32()[3]`
/// there returned the population std instead of Σx² and silently disabled
/// FASE gradient clipping on GPU (the global norm collapsed by ~n).
///
/// Do NOT collapse this into `gpu_sum_sq_many_f32`: that one exists for the
/// batched-drain case and reads slot 3 of the same kernel, so the confusion
/// above is one refactor away from returning.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_tensor_sum_sq_f32(tensor_ptr: i64) -> f64 {
    use crate::tensor::NslTensor;
    use fused_kernels::TENSOR_STATS_F32_PTX;

    let t = NslTensor::from_ptr(tensor_ptr);
    assert_gpu_f32(t, "tensor_sum_sq_f32", "input");
    let n = t.len as u64;
    if n == 0 {
        return 0.0;
    }

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
        [1, 1, 1], [256, 1, 1], &args, 256 * 4 * 4,
    );
    assert_eq!(result as u32, 0, "GPU tensor_stats kernel failed: {:?}", result);
    unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

    let mut raw = [0f32; 4];
    inner::memcpy_dtoh(
        raw.as_mut_ptr() as *mut std::ffi::c_void,
        out_data as *const std::ffi::c_void,
        16,
    );
    inner::free_managed(out_data);

    // raw = [min, max, sum, sum_sq]; slot 3 is the raw Σx² (pre-transform).
    f64::from(raw[3])
}

/// Fusion-queue item 2: GPU-native cross-entropy backward.
/// out = (softmax(logits) - onehot(targets)) * grad_output / num_valid,
/// invalid (target < 0) rows zeroed — the CPU arm's semantics, computed
/// entirely on device (softmax kernel + valid-count reduction + finish
/// pass; grad_output read in-kernel when device-resident). No host
/// readbacks: the loss epilogue stays cuda-graph-capturable.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_cross_entropy_backward_f32(
    logits_ptr: i64,
    targets_ptr: i64,
    grad_out_ptr: i64,
) -> i64 {
    use crate::tensor::NslTensor;
    use fused_kernels::{CE_BWD_COUNT_F32_PTX, CE_BWD_FINISH_F32_PTX};

    inner::set_oom_context("ce_backward_f32");
    // Logits only. `targets_ptr` is an index tensor and `grad_out_ptr` is
    // read through an explicit dtype match below, so both are deliberately
    // outside the f32 guard.
    assert_gpu_f32(
        unsafe { &*(logits_ptr as *const NslTensor) },
        "cross_entropy_backward_f32",
        "logits",
    );
    let sm = gpu_softmax_f32(logits_ptr);
    let smt = unsafe { &*(sm as *const NslTensor) };
    let ndim = smt.ndim as usize;
    let cols = unsafe { *smt.shape.add(ndim - 1) } as u32;
    let total = smt.len as u32;
    let rows = total / cols.max(1);

    thread_local! {
        static CE_SCRATCH: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }
    let scratch = CE_SCRATCH.with(|c| {
        if c.get() == 0 {
            c.set(inner::alloc_managed(16) as u64);
        }
        c.get()
    });

    let tgt = unsafe { &*(targets_ptr as *const NslTensor) };
    let tgt_i32: u32 = u32::from(tgt.dtype == crate::tensor::DTYPE_I32);

    // K1: valid-target count -> scratch[0] (denom, already max(count,1)).
    {
        let mut a0 = tgt.data as u64;
        let mut a1 = rows;
        let mut a2 = tgt_i32;
        let mut a3 = scratch;
        let args: [*mut std::ffi::c_void; 4] = [
            &mut a0 as *mut _ as *mut std::ffi::c_void,
            &mut a1 as *mut _ as *mut std::ffi::c_void,
            &mut a2 as *mut _ as *mut std::ffi::c_void,
            &mut a3 as *mut _ as *mut std::ffi::c_void,
        ];
        let r = inner::kernel_launch(
            CE_BWD_COUNT_F32_PTX.as_ptr(),
            b"nsl_ce_bwd_count_f32\0".as_ptr(),
            [1, 1, 1],
            [256, 1, 1],
            &args,
            0, // smem is STATIC in the kernel (.shared .u32 cnt[256])
        );
        assert_eq!(r as u32, 0, "GPU ce_bwd_count kernel failed: {r:?}");
    }

    // grad_output: device f32 scalar -> read in-kernel; host scalar ->
    // folded immediate (a host read of a CPU tensor is free); non-scalar
    // -> 1.0 (CPU-arm semantics).
    let go = unsafe { &*(grad_out_ptr as *const NslTensor) };
    let (go_mode, gop, go_imm): (u32, u64, f32) = if go.len == 1 {
        if go.device > 0 && go.dtype == 1 {
            (1, go.data as u64, 1.0)
        } else if go.device == 0 {
            let v = match go.dtype {
                1 => unsafe { *(go.data as *const f32) },
                crate::tensor::DTYPE_I32 => unsafe { *(go.data as *const i32) as f32 },
                _ => unsafe { *(go.data as *const f64) as f32 },
            };
            (0, 0, v)
        } else {
            (0, 0, 1.0)
        }
    } else {
        (0, 0, 1.0)
    };

    // K2: finish pass in place over the softmax buffer.
    {
        let mut a0 = smt.data as u64;
        let mut a1 = tgt.data as u64;
        let mut a2 = scratch;
        let mut a3 = gop;
        let mut a4 = go_imm;
        let mut a5 = go_mode;
        let mut a6 = tgt_i32;
        let mut a7 = total;
        let mut a8 = cols;
        let args: [*mut std::ffi::c_void; 9] = [
            &mut a0 as *mut _ as *mut std::ffi::c_void,
            &mut a1 as *mut _ as *mut std::ffi::c_void,
            &mut a2 as *mut _ as *mut std::ffi::c_void,
            &mut a3 as *mut _ as *mut std::ffi::c_void,
            &mut a4 as *mut _ as *mut std::ffi::c_void,
            &mut a5 as *mut _ as *mut std::ffi::c_void,
            &mut a6 as *mut _ as *mut std::ffi::c_void,
            &mut a7 as *mut _ as *mut std::ffi::c_void,
            &mut a8 as *mut _ as *mut std::ffi::c_void,
        ];
        let block = 256i64;
        let grid = ((total as i64) + block - 1) / block;
        let r = inner::kernel_launch(
            CE_BWD_FINISH_F32_PTX.as_ptr(),
            b"nsl_ce_bwd_finish_f32\0".as_ptr(),
            [grid, 1, 1],
            [block, 1, 1],
            &args,
            0,
        );
        assert_eq!(r as u32, 0, "GPU ce_bwd_finish kernel failed: {r:?}");
    }
    inner::sync_after_kernel();
    sm
}

/// P1 Muon items 8+10: Frobenius-normalize an f32 tensor ENTIRELY on-device —
/// out = x / (sqrt(Σx²) + 1e-7) with the Σx² produced by the stats kernel
/// into a persistent 16-byte device scratch buffer and consumed directly by
/// `nsl_muon_scale_inv_frob_f32`. NO cuCtxSynchronize, NO DtoH copy: this is
/// the removal of the per-param `.item()` sync from the Muon Newton-Schulz
/// pre-normalization (the two launches serialize on the compute stream).
/// The scratch buffer is thread-local and process-lifetime (item 10's
/// preallocated scratch): one 16-byte allocation per training thread, ever.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_muon_frobenius_scale_f32(a_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    use fused_kernels::TENSOR_STATS_F32_PTX;
    use kernels::MUON_SCALE_INV_FROB_F32_PTX;

    inner::set_oom_context("muon_frobenius_scale_f32");
    let a = unsafe { &*(a_ptr as *const NslTensor) };
    // Review finding: the kernels index the buffer as f32 — a quantized /
    // custom-dtype tensor (1 byte/elem) reached via the direct builtin
    // would be read len*4 bytes OOB. Refuse anything but f32 loudly (the
    // CPU arm has the equivalent dtype match).
    if a.dtype != crate::tensor::DTYPE_F32 {
        eprintln!(
            "nsl: muon_orthogonalize GPU path requires an f32 tensor (got dtype {})",
            a.dtype
        );
        std::process::abort();
    }
    // Flat row-major kernels — materialize strided views first (same
    // stride-blindness class as the tied-embedding matmul bug).
    if !a.is_contiguous() {
        let a_c = crate::tensor::nsl_tensor_contiguous(a_ptr);
        let result = gpu_muon_frobenius_scale_f32(a_c);
        crate::tensor::nsl_tensor_free(a_c);
        return result;
    }
    let n = a.len as usize;

    thread_local! {
        static MUON_STATS_BUF: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
    }
    let stats_buf = MUON_STATS_BUF.with(|c| {
        if c.get() == 0 {
            c.set(inner::alloc_managed(16) as u64);
        }
        c.get()
    });

    // Launch 1: stats reduction (slot 3 = raw Σx²) into the device scratch.
    let mut in_data = a.data as u64;
    let mut stats_val = stats_buf;
    let mut n_val = n as u64;
    let stats_args: [*mut std::ffi::c_void; 3] = [
        &mut in_data as *mut _ as *mut std::ffi::c_void,
        &mut stats_val as *mut _ as *mut std::ffi::c_void,
        &mut n_val as *mut _ as *mut std::ffi::c_void,
    ];
    let result = inner::kernel_launch(
        TENSOR_STATS_F32_PTX.as_ptr(), b"nsl_tensor_stats_f32\0".as_ptr(),
        [1, 1, 1], [256, 1, 1], &stats_args, 256 * 4 * 4,
    );
    assert_eq!(result as u32, 0, "GPU tensor_stats kernel failed: {:?}", result);

    // Launch 2: elementwise scale reading the norm from device memory.
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

    let mut x_data = a.data as u64;
    let mut c_data = out_t.data as u64;
    let mut stats_val2 = stats_buf;
    let mut n_val2 = n as u64;
    let scale_args: [*mut std::ffi::c_void; 4] = [
        &mut x_data as *mut _ as *mut std::ffi::c_void,
        &mut c_data as *mut _ as *mut std::ffi::c_void,
        &mut stats_val2 as *mut _ as *mut std::ffi::c_void,
        &mut n_val2 as *mut _ as *mut std::ffi::c_void,
    ];
    let block = 256i64;
    let grid = ((n as i64) + block - 1) / block;
    let result = inner::kernel_launch(
        MUON_SCALE_INV_FROB_F32_PTX.as_ptr(),
        b"nsl_muon_scale_inv_frob_f32\0".as_ptr(),
        [grid, 1, 1], [block, 1, 1], &scale_args, 0,
    );
    assert_eq!(
        result as u32, 0,
        "GPU muon_scale_inv_frob kernel failed: {:?}", result
    );
    inner::sync_after_kernel();
    out_ptr as i64
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
/// PRECONDITION (unenforceable here): every pointer argument is a bare
/// device address with no `NslTensor` wrapper, so there is no dtype to
/// check — `out_ptr` must address `n` f32 elements and `scales_ptr` f32
/// scales. The check belongs at the NslTensor-holding caller.
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
///
/// PRECONDITION: as `gpu_dequant_int8_per_head_f32` — raw device pointers,
/// f32 output and scales, no dtype available to assert on.
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
///
/// PRECONDITION: as `gpu_dequant_int8_per_head_f32` — raw device pointers,
/// f32 output, scales and zero-points, no dtype available to assert on.
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
///
/// PRECONDITION: as `gpu_dequant_int8_per_head_f32` — raw device pointers,
/// f32 output, no dtype available to assert on.
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
    assert_gpu_f32(t, "det_global_sum_f32", "input");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(t, "det_sum_dim_f32", "input");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(input, "det_scatter_add_f32", "destination");
    assert_gpu_f32(src, "det_scatter_add_f32", "source");

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
    inner::sync_after_kernel();

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
    assert_gpu_f32(t, "max_dim_f32", "input");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(t, "layernorm_f32", "input");
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
    assert_gpu_f32(g, "layernorm_f32", "gamma");
    assert_gpu_f32(b, "layernorm_f32", "beta");

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
    inner::sync_after_kernel();

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
    assert_gpu_f32(t, "rmsnorm_f32", "input");
    let ndim = t.ndim as usize;
    let shape_slice = unsafe { std::slice::from_raw_parts(t.shape, ndim) };

    let cols = shape_slice[ndim - 1] as u64;
    let rows = (t.len as u64) / cols;

    let total = t.len as usize;
    let out_data = inner::alloc_managed(total * 4);
    let out_shape = NslTensor::copy_shape(t.shape, t.ndim);
    let out_strides = NslTensor::compute_strides(out_shape, t.ndim);

    let g = NslTensor::from_ptr(gamma_ptr);
    assert_gpu_f32(g, "rmsnorm_f32", "gamma");

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
    inner::sync_after_kernel();

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

/// Fused RMSNorm INPUT-gradient (dx), one block per row. All of `dy`, `x`,
/// `gamma` must be contiguous f32 on-device; returns a fresh f32 dx tensor of
/// `x`'s shape. Recomputes rms internally (no saved-rms dependency), matching
/// the correct RMSNorm dx (no mean-subtract).
/// P5 item 20 slice A: fused RMSNorm gamma gradient.
/// dgamma[j] = sum_rows(dy[i,j] * x[i,j] / rms_i), computed as two
/// deterministic launches (per-row 1/rms into a tiny scratch, then a
/// per-column sequential row loop) — replaces the 7-op decomposition and
/// its three [rows, cols] temporaries. Fixed summation order per column,
/// so bit-deterministic run-to-run.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_rmsnorm_dgamma_backward_f32(
    dy_ptr: i64,
    x_ptr: i64,
    gamma_ptr: i64,
    eps: f32,
) -> i64 {
    inner::set_oom_context("rmsnorm_dgamma_bwd_f32");
    use crate::tensor::NslTensor;
    use fused_kernels::{RMSNORM_DGAMMA_F32_PTX, RMSNORM_RINV_ROWS_F32_PTX};

    let x = NslTensor::from_ptr(x_ptr);
    assert_gpu_f32(x, "rmsnorm_dgamma_backward_f32", "x");
    let ndim = x.ndim as usize;
    let shape_slice = unsafe { std::slice::from_raw_parts(x.shape, ndim) };
    let cols = shape_slice[ndim - 1] as u64;
    let rows = (x.len as u64) / cols;

    let g = NslTensor::from_ptr(gamma_ptr);
    assert_gpu_f32(g, "rmsnorm_dgamma_backward_f32", "gamma");
    assert_eq!(
        g.len as u64, cols,
        "rmsnorm dgamma: gamma len {} != last dim {}",
        g.len, cols
    );
    assert_eq!(
        NslTensor::from_ptr(dy_ptr).len,
        x.len,
        "rmsnorm dgamma: dy/x length mismatch"
    );

    // Output rides gamma's shape (rank-1 [cols] in every stdlib norm).
    let out_data = inner::alloc_managed(cols as usize * 4);
    let out_shape = NslTensor::copy_shape(g.shape, g.ndim);
    let out_strides = NslTensor::compute_strides(out_shape, g.ndim);
    // Tiny per-row scratch for 1/rms.
    let rinv = inner::alloc_managed(rows as usize * 4);

    let dy = NslTensor::from_ptr(dy_ptr);
    assert_gpu_f32(dy, "rmsnorm_dgamma_backward_f32", "dy");
    let mut dy_data = dy.data as u64;
    let mut x_data = x.data as u64;
    let mut rinv_u64 = rinv as u64;
    let mut out_u64 = out_data as u64;
    let mut rows_val = rows;
    let mut cols_val = cols;
    let mut eps_val = eps;

    let rinv_args: [*mut std::ffi::c_void; 5] = [
        &mut x_data as *mut _ as *mut std::ffi::c_void,
        &mut rinv_u64 as *mut _ as *mut std::ffi::c_void,
        &mut rows_val as *mut _ as *mut std::ffi::c_void,
        &mut cols_val as *mut _ as *mut std::ffi::c_void,
        &mut eps_val as *mut _ as *mut std::ffi::c_void,
    ];
    let grid_rows = rows.div_ceil(256) as i64;
    let result = inner::kernel_launch(
        RMSNORM_RINV_ROWS_F32_PTX.as_ptr(),
        b"nsl_rmsnorm_rinv_rows_f32\0".as_ptr(),
        [grid_rows.max(1), 1, 1],
        [256, 1, 1],
        &rinv_args,
        0,
    );
    assert_eq!(result as u32, 0, "GPU rmsnorm rinv kernel failed: {:?}", result);

    let dg_args: [*mut std::ffi::c_void; 6] = [
        &mut dy_data as *mut _ as *mut std::ffi::c_void,
        &mut x_data as *mut _ as *mut std::ffi::c_void,
        &mut rinv_u64 as *mut _ as *mut std::ffi::c_void,
        &mut out_u64 as *mut _ as *mut std::ffi::c_void,
        &mut rows_val as *mut _ as *mut std::ffi::c_void,
        &mut cols_val as *mut _ as *mut std::ffi::c_void,
    ];
    let grid_cols = cols.div_ceil(256) as i64;
    let result = inner::kernel_launch(
        RMSNORM_DGAMMA_F32_PTX.as_ptr(),
        b"nsl_rmsnorm_dgamma_f32\0".as_ptr(),
        [grid_cols.max(1), 1, 1],
        [256, 1, 1],
        &dg_args,
        0,
    );
    assert_eq!(result as u32, 0, "GPU rmsnorm dgamma kernel failed: {:?}", result);
    inner::sync_after_kernel();

    // Stream-ordered pool free: the block only re-enters circulation via
    // this thread's in-order allocator, so the kernel that read it has
    // retired before any same-stream reuse (the caching allocator never
    // returns memory to the driver here).
    inner::free_managed(rinv);

    let out = Box::new(NslTensor::new(
        out_data, out_shape, out_strides, g.ndim, g.len, x.device, 1, 1, 0,
    ));
    NslTensor::publish(out)
}

#[cfg(feature = "cuda")]
/// P5 slice C: fused RMSNorm dx + residual-gradient fold (one launch for
/// what was dx-kernel + elementwise Add). `res` must be contiguous f32 of
/// x's shape on the same device.
pub(crate) fn gpu_rmsnorm_dx_backward_add_f32(
    dy_ptr: i64,
    x_ptr: i64,
    gamma_ptr: i64,
    res_ptr: i64,
    eps: f32,
) -> i64 {
    inner::set_oom_context("rmsnorm_dx_bwd_add_f32");
    use crate::tensor::NslTensor;
    use fused_kernels::RMSNORM_DX_BWD_ADD_F32_PTX;

    let x = NslTensor::from_ptr(x_ptr);
    assert_gpu_f32(x, "rmsnorm_dx_backward_add_f32", "x");
    let ndim = x.ndim as usize;
    let shape_slice = unsafe { std::slice::from_raw_parts(x.shape, ndim) };
    let cols = shape_slice[ndim - 1] as u64;
    let rows = (x.len as u64) / cols;
    assert_eq!(
        NslTensor::from_ptr(res_ptr).len,
        x.len,
        "rmsnorm dx+res: residual/x length mismatch"
    );
    assert_eq!(
        NslTensor::from_ptr(dy_ptr).len,
        x.len,
        "rmsnorm dx+res: dy/x length mismatch"
    );

    let total = x.len as usize;
    let out_data = inner::alloc_managed(total * 4);
    let out_shape = NslTensor::copy_shape(x.shape, x.ndim);
    let out_strides = NslTensor::compute_strides(out_shape, x.ndim);

    let dy = NslTensor::from_ptr(dy_ptr);
    let g = NslTensor::from_ptr(gamma_ptr);
    let r = NslTensor::from_ptr(res_ptr);
    assert_gpu_f32(dy, "rmsnorm_dx_backward_add_f32", "dy");
    assert_gpu_f32(g, "rmsnorm_dx_backward_add_f32", "gamma");
    assert_gpu_f32(r, "rmsnorm_dx_backward_add_f32", "residual");

    let mut dy_data = dy.data as u64;
    let mut x_data = x.data as u64;
    let mut g_data = g.data as u64;
    let mut out_u64 = out_data as u64;
    let mut res_data = r.data as u64;
    let mut rows_val = rows;
    let mut cols_val = cols;
    let mut eps_val = eps;

    let args: [*mut std::ffi::c_void; 8] = [
        &mut dy_data as *mut _ as *mut std::ffi::c_void,
        &mut x_data as *mut _ as *mut std::ffi::c_void,
        &mut g_data as *mut _ as *mut std::ffi::c_void,
        &mut out_u64 as *mut _ as *mut std::ffi::c_void,
        &mut res_data as *mut _ as *mut std::ffi::c_void,
        &mut rows_val as *mut _ as *mut std::ffi::c_void,
        &mut cols_val as *mut _ as *mut std::ffi::c_void,
        &mut eps_val as *mut _ as *mut std::ffi::c_void,
    ];

    let result = inner::kernel_launch(
        RMSNORM_DX_BWD_ADD_F32_PTX.as_ptr(),
        b"nsl_rmsnorm_dx_bwd_add_f32\0".as_ptr(),
        [rows as i64, 1, 1],
        [256, 1, 1],
        &args,
        256 * 4,
    );
    assert_eq!(result as u32, 0, "GPU rmsnorm dx+res kernel failed: {:?}", result);
    inner::sync_after_kernel();

    let out = Box::new(NslTensor::new(
        out_data, out_shape, out_strides, x.ndim, x.len, x.device, 1, 1, 0,
    ));
    NslTensor::publish(out)
}

#[cfg(feature = "cuda")]
pub(crate) fn gpu_rmsnorm_dx_backward_f32(
    dy_ptr: i64,
    x_ptr: i64,
    gamma_ptr: i64,
    eps: f32,
) -> i64 {
    inner::set_oom_context("rmsnorm_dx_bwd_f32");
    use crate::tensor::NslTensor;
    use fused_kernels::RMSNORM_DX_BWD_F32_PTX;

    let x = NslTensor::from_ptr(x_ptr);
    assert_gpu_f32(x, "rmsnorm_dx_backward_f32", "x");
    let ndim = x.ndim as usize;
    let shape_slice = unsafe { std::slice::from_raw_parts(x.shape, ndim) };
    let cols = shape_slice[ndim - 1] as u64;
    let rows = (x.len as u64) / cols;

    let total = x.len as usize;
    let out_data = inner::alloc_managed(total * 4);
    let out_shape = NslTensor::copy_shape(x.shape, x.ndim);
    let out_strides = NslTensor::compute_strides(out_shape, x.ndim);

    let dy = NslTensor::from_ptr(dy_ptr);
    let g = NslTensor::from_ptr(gamma_ptr);
    assert_gpu_f32(dy, "rmsnorm_dx_backward_f32", "dy");
    assert_gpu_f32(g, "rmsnorm_dx_backward_f32", "gamma");

    let mut dy_data = dy.data as u64;
    let mut x_data = x.data as u64;
    let mut g_data = g.data as u64;
    let mut out_u64 = out_data as u64;
    let mut rows_val = rows;
    let mut cols_val = cols;
    let mut eps_val = eps;

    let args: [*mut std::ffi::c_void; 7] = [
        &mut dy_data as *mut _ as *mut std::ffi::c_void,
        &mut x_data as *mut _ as *mut std::ffi::c_void,
        &mut g_data as *mut _ as *mut std::ffi::c_void,
        &mut out_u64 as *mut _ as *mut std::ffi::c_void,
        &mut rows_val as *mut _ as *mut std::ffi::c_void,
        &mut cols_val as *mut _ as *mut std::ffi::c_void,
        &mut eps_val as *mut _ as *mut std::ffi::c_void,
    ];

    let result = inner::kernel_launch(
        RMSNORM_DX_BWD_F32_PTX.as_ptr(),
        b"nsl_rmsnorm_dx_bwd_f32\0".as_ptr(),
        [rows as i64, 1, 1],
        [256, 1, 1],
        &args,
        256 * 4,
    );
    assert_eq!(result as u32, 0, "GPU rmsnorm dx-bwd kernel failed: {:?}", result);
    inner::sync_after_kernel();

    let out = Box::new(NslTensor::new(
        out_data, out_shape, out_strides, x.ndim, x.len, x.device, 1, 1, 0,
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
    assert_gpu_f32(src, "scatter_add_f32", "source");

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
    inner::sync_after_kernel();

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

/// Upload a small i64 metadata array (a shape or a stride vector) to the device,
/// reusing a previous upload of the same contents.
///
/// `cuMemcpyHtoD_v2` is the blocking API, so each of these tiny uploads waits on
/// the stream. The strided-copy path issues three per call and runs hundreds of
/// times per training step, which made these the largest remaining host cost
/// once the bulk transfers were gone. The distinct (shape, strides) tuples in a
/// model are few and fixed, so caching them keyed on contents removes almost
/// every upload after the first step.
///
/// Entries are never freed, so they are allocated from the PERSISTENT pool.
/// They are bounded by the number of distinct metadata vectors in the program —
/// tens of tuples at 8 bytes per dimension — but the caching allocator releases
/// segments, not blocks: `drain_all` only reclaims a transient segment whose
/// `allocated_count` is zero, so a never-freed 24-byte block in a transient
/// segment would pin that whole segment's VRAM for the life of the process and
/// make the per-step drain silently stop working. Persistent segments are never
/// drained anyway, so putting them there costs nothing and pins nothing extra.
///
/// The key includes the device ordinal: the contents alone would hand a pointer
/// uploaded on device 0 to a kernel running on device 1.
#[cfg(feature = "cuda")]
pub(crate) fn upload_meta_i64_cached(host: *const i64, ndim: usize) -> *mut std::ffi::c_void {
    use std::collections::HashMap;
    use std::sync::Mutex;

    static CACHE: std::sync::OnceLock<Mutex<HashMap<(i32, Vec<i64>), u64>>> =
        std::sync::OnceLock::new();
    let device = inner::current_device_ordinal();
    let key = (device, (0..ndim).map(|i| unsafe { *host.add(i) }).collect::<Vec<i64>>());
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));

    // Take the lock only to look up / publish; the upload itself happens outside
    // so a slow copy never serializes other threads' lookups.
    if let Some(&dev) = cache.lock().unwrap().get(&key) {
        return dev as *mut std::ffi::c_void;
    }
    let bytes = ndim * std::mem::size_of::<i64>();
    let dev = {
        use crate::cuda::caching_allocator::{AllocPool, PoolGuard};
        let _pool = PoolGuard::new(AllocPool::Persistent);
        inner::alloc_managed(bytes)
    };
    // Immediate, NOT `memcpy_htod`: a capture region would defer this to graph
    // launch and the cache would publish a pointer to unwritten memory. See
    // `memcpy_htod_immediate`.
    inner::memcpy_htod_immediate(dev, host as *const std::ffi::c_void, bytes);
    let mut guard = cache.lock().unwrap();
    match guard.entry(key) {
        std::collections::hash_map::Entry::Occupied(slot) => {
            // Another thread published an identical vector first; drop ours.
            let winner = *slot.get() as *mut std::ffi::c_void;
            inner::free_managed(dev);
            winner
        }
        std::collections::hash_map::Entry::Vacant(slot) => {
            slot.insert(dev as u64);
            dev
        }
    }
}

/// Gather along an arbitrary dimension, on the device.
///
/// `indices_host` must already be validated against `gather_dim_size` by the
/// caller — the CPU path aborts on an out-of-range index and this keeps that
/// contract without a device-side error channel.
///
/// `inner_size`, not `inner`: the latter would shadow this module's `inner`
/// submodule, which every allocation and copy below goes through.
#[cfg(feature = "cuda")]
pub(crate) fn gpu_gather_dim_f32(
    input_ptr: i64,
    indices_host: &[i64],
    outer: u64,
    gather_dim_size: u64,
    inner_size: u64,
) -> *mut std::ffi::c_void {
    inner::set_oom_context("gather_dim_f32");
    use crate::tensor::NslTensor;

    let input = unsafe { &*(input_ptr as *const NslTensor) };
    assert_gpu_f32(input, "gather_dim_f32", "input");
    let out_elems = (outer * inner_size) as usize;
    if out_elems == 0 {
        return std::ptr::null_mut();
    }

    // The index vector is `outer` elements — kilobytes next to the tensor itself —
    // so staging it as f32 to match the kernel's read is not worth avoiding.
    let staged: Vec<f32> = indices_host.iter().map(|&i| i as f32).collect();
    let idx_bytes = staged.len() * 4;
    let idx_device = inner::alloc_managed(idx_bytes);
    inner::memcpy_htod(idx_device, staged.as_ptr() as *const std::ffi::c_void, idx_bytes);

    let out_data = inner::alloc_managed(out_elems * 4);

    let mut i_data = input.data as u64;
    let mut idx_data = idx_device as u64;
    let mut o_data = out_data as u64;
    let mut outer_v = outer;
    let mut dim_v = gather_dim_size;
    let mut inner_v = inner_size;
    let args: [*mut std::ffi::c_void; 6] = [
        &mut i_data as *mut _ as *mut std::ffi::c_void,
        &mut idx_data as *mut _ as *mut std::ffi::c_void,
        &mut o_data as *mut _ as *mut std::ffi::c_void,
        &mut outer_v as *mut _ as *mut std::ffi::c_void,
        &mut dim_v as *mut _ as *mut std::ffi::c_void,
        &mut inner_v as *mut _ as *mut std::ffi::c_void,
    ];

    let block = 256i64;
    let grid = (out_elems as i64 + block - 1) / block;
    let rc = inner::kernel_launch(
        fused_kernels::GATHER_DIM_F32_PTX.as_ptr(),
        b"nsl_gather_dim_f32\0".as_ptr(),
        [grid, 1, 1],
        [block, 1, 1],
        &args,
        0,
    );
    inner::free_managed(idx_device);
    assert_eq!(rc as u32, 0, "GPU gather_dim kernel failed: {:?}", rc);
    inner::sync_after_kernel();
    out_data
}

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
    assert_gpu_f32(input, "gather_f32", "input");

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
    let (ptx, kernel_name): (&str, &[u8]) = if indices_gpu.dtype == crate::tensor::DTYPE_I32 {
        (fused_kernels::GATHER_I32IDX_PTX, b"nsl_gather_i32idx\0")
    } else {
        (GATHER_F32_PTX, b"nsl_gather_f32\0")
    };

    let result = inner::kernel_launch(
        ptx.as_ptr(), kernel_name.as_ptr(),
        [grid_x, grid_y, 1], [block_x, block_y, 1], &args, 0,
    );
    assert_eq!(result as u32, 0, "GPU gather kernel failed: {:?}", result);
    inner::sync_after_kernel();

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
    assert_gpu_f32(input, "conv2d_f32", "input");

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
    // weight/bias are guarded AFTER their transfer for the same reason as
    // `gpu_bias_add`: a host-resident parameter is legitimately f64 and
    // `nsl_tensor_to_device` narrows it to f32 in flight, while a 16-bit
    // dtype survives verbatim and is what would be misread.
    assert_gpu_f32(weight_gpu, "conv2d_f32", "weight");

    // Bias pointer (0 if no bias)
    let bias_data: u64 = if bias_ptr != 0 {
        let bias = unsafe { &*(bias_ptr as *const NslTensor) };
        if bias.device == 0 {
            let bp = crate::tensor::nsl_tensor_to_device(bias_ptr, input.device as i64);
            let b = unsafe { &*(bp as *const NslTensor) };
            assert_gpu_f32(b, "conv2d_f32", "bias");
            let d = b.data as u64;
            // We leak the bias transfer — acceptable for now
            d
        } else {
            assert_gpu_f32(bias, "conv2d_f32", "bias");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(input, "maxpool2d_f32", "input");

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

    // Read argmax indices back to CPU (needed for the backward tape).
    // `argmax_data` is a DEVICE pointer — `alloc_managed` routes through the
    // caching allocator's `cuMemAlloc`, NOT `cuMemAllocManaged`, so it is not
    // host-accessible (see its doc-comment). Dereferencing it on the host with
    // `from_raw_parts` SIGSEGVs on a discrete GPU and aborts the whole process
    // — the exact bug already fixed in `test_vec_add_kernel_launch`. Stage it
    // through `memcpy_dtoh` instead.
    let mut argmax_vec: Vec<u64> = vec![0u64; total as usize];
    inner::memcpy_dtoh(
        argmax_vec.as_mut_ptr() as *mut std::ffi::c_void,
        argmax_data as *const std::ffi::c_void,
        total as usize * std::mem::size_of::<u64>(),
    );
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
    assert_gpu_f32(input, "dropout_f32", "input");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(t, "slice_f32_with_shape", "input");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(b, "csr_spmm_f32", "dense operand");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(b, "coo_spmm_f32", "dense operand");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(x, "csr_spmv_f32", "dense vector");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(b, "bsr_spmm_f32", "dense operand");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(x, "coo_spmv_f32", "dense vector");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(b, "sparse_matmul_csr_f32", "dense operand");
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
    inner::sync_after_kernel();

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
    assert_gpu_f32(t, "strided_copy_f32", "input");
    let ndim = t.ndim as usize;
    let total = t.len as u64;

    // Allocate contiguous output on GPU
    let out_data = inner::alloc_managed(total as usize * 4); // f32
    let out_shape = NslTensor::copy_shape(t.shape, t.ndim);
    let out_strides = NslTensor::compute_strides(out_shape, t.ndim);

    // Fast path: views whose innermost axis is already a contiguous run (every
    // broadcast, and every reordering that keeps the last axis innermost) can
    // skip the generic kernel's per-element coordinate decomposition entirely.
    // See `strided_copy` for the analysis; the generic kernel below stays as the
    // fallback for true transposes.
    {
        let shape: Vec<i64> = (0..ndim).map(|i| unsafe { *t.shape.add(i) }).collect();
        let src_strides: Vec<i64> = (0..ndim).map(|i| unsafe { *t.strides.add(i) }).collect();
        let dst_strides: Vec<i64> = (0..ndim).map(|i| unsafe { *out_strides.add(i) }).collect();
        if let Some(resident) = strided_copy::resident_plan(&shape, &src_strides, &dst_strides) {
            // Vector moves assume 16-byte-aligned bases. The destination is a
            // fresh allocation, but the source is a view that may point into the
            // middle of a parent buffer.
            let aligned = (t.data as usize) % 16 == 0 && (out_data as usize) % 16 == 0;
            if !resident.plan.vec4 || aligned {
                let (grid, block) = resident.plan.geometry();
                let mut src_data = t.data as u64;
                let mut dst_data = out_data as u64;
                let mut offsets = resident.offsets_dev;
                let mut run_len = resident.plan.run_len as u64;
                let mut outer = resident.plan.outer as u64;
                let args: [*mut std::ffi::c_void; 5] = [
                    &mut src_data as *mut _ as *mut std::ffi::c_void,
                    &mut dst_data as *mut _ as *mut std::ffi::c_void,
                    &mut offsets as *mut _ as *mut std::ffi::c_void,
                    &mut run_len as *mut _ as *mut std::ffi::c_void,
                    &mut outer as *mut _ as *mut std::ffi::c_void,
                ];
                let result = inner::kernel_launch(
                    strided_copy::STRIDED_COPY_RUN_PTX.as_ptr(),
                    resident.plan.kernel_name().as_ptr(),
                    grid,
                    block,
                    &args,
                    0,
                );
                assert_eq!(
                    result as u32, 0,
                    "GPU strided run copy kernel failed: {result:?}"
                );
                strided_copy::ARM_LAUNCHES[resident.plan.arm_index()]
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                inner::sync_after_kernel();

                let out = Box::new(NslTensor::new(
                    out_data, out_shape, out_strides,
                    t.ndim, total as i64, t.device, 1, 1, 0,
                ));
                return NslTensor::publish(out);
            }
        }
    }

    // Upload shape, src_strides, and dst_strides to device memory
    let gpu_shape = upload_meta_i64_cached(t.shape, ndim);
    let gpu_src_strides = upload_meta_i64_cached(t.strides, ndim);
    let gpu_dst_strides = upload_meta_i64_cached(out_strides, ndim);

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
    inner::sync_after_kernel();

    // The metadata arrays are cached and shared across calls, so they must
    // NOT be freed here — a later call with the same shape reuses this exact
    // device pointer.


    let out = Box::new(NslTensor::new(
        out_data, out_shape, out_strides,
        t.ndim, total as i64, t.device, 1, 1, 0,
    ));
    NslTensor::publish(out)
}

// ──────────────────────────────────────────────────────────────────────
// Test-only helpers (doc-hidden, feature-gated) — cross-crate integration
// tests need raw device alloc + H2D/D2H without going through the full
// NslTensor publishing machinery. Keeping these as thin pub wrappers over
// `inner::*` lets tests in sibling crates exercise real kernel launches
// against `nsl_flash_attention` / `nsl_flash_attention_csha`.
// ──────────────────────────────────────────────────────────────────────

/// Allocate `bytes` of device memory. Returns raw device pointer as i64
/// (0 on non-CUDA builds). Test-only — production code paths go through
/// the caching allocator.
#[doc(hidden)]
#[no_mangle]
pub extern "C" fn nsl_test_cuda_alloc(bytes: i64) -> i64 {
    #[cfg(feature = "cuda")]
    {
        inner::alloc_device(bytes as usize) as i64
    }
    #[cfg(not(feature = "cuda"))]
    { let _ = bytes; 0 }
}

/// Free device memory previously returned by `nsl_test_cuda_alloc`.
///
/// Routes to `inner::free_device` (raw `cuMemFree_v2`) to match
/// `nsl_test_cuda_alloc` which calls `inner::alloc_device` (raw `cuMemAlloc_v2`).
///
/// Earlier versions called `inner::free_managed`, which early-returns silently
/// when the pointer is not in `CUDA_ALLOC_SET` — but `alloc_device` never
/// registers in that set, so every `nsl_test_cuda_free` was a no-op. That leak
/// (5-9 buffers per `launch_pca_ex`-style helper, accumulating over the suite)
/// was the empirical root cause of the 2026-05-27 in-suite PCA test flakiness:
/// later test allocations hit cuMemAlloc OOM recovery / pool-drain paths whose
/// timing or driver-state perturbation altered bit-exact assertions downstream.
/// See the 2026-06-29 adversarial-verify workflow on the forward per-doc RoPE
/// reset reimplementation for the discovery context.
#[doc(hidden)]
#[no_mangle]
pub extern "C" fn nsl_test_cuda_free(ptr: i64) {
    #[cfg(feature = "cuda")]
    {
        if ptr != 0 { inner::free_device(ptr as *mut std::ffi::c_void); }
    }
    #[cfg(not(feature = "cuda"))]
    { let _ = ptr; }
}

/// Copy `bytes` from host pointer `src` to device pointer `dst`.
#[doc(hidden)]
#[no_mangle]
pub extern "C" fn nsl_test_cuda_h2d(dst: i64, src: i64, bytes: i64) {
    #[cfg(feature = "cuda")]
    {
        inner::memcpy_htod(
            dst as *mut std::ffi::c_void,
            src as *const std::ffi::c_void,
            bytes as usize,
        );
    }
    #[cfg(not(feature = "cuda"))]
    { let _ = (dst, src, bytes); }
}

/// Attempt to JIT-compile a PTX string via `cuModuleLoadDataEx` with an
/// error log buffer, returning the driver's diagnostic message. Returns
/// an empty string on success, or the error log (possibly with a generic
/// prefix) on failure. Test-only.
#[doc(hidden)]
#[no_mangle]
pub extern "C" fn nsl_test_cuda_jit_log(ptx_ptr: i64) -> i64 {
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys::*;
        // Context is assumed already current on this thread (nsl_cuda_init
        // and any prior kernel launch will have set it). We don't re-set
        // it here to avoid having to reach into the private CudaState.
        let mut log_buf = vec![0u8; 4096];
        let mut info_buf = vec![0u8; 4096];
        let log_size: u32 = log_buf.len() as u32;
        let info_size: u32 = info_buf.len() as u32;
        // JIT options: error log buffer + size + info log buffer + size
        let mut opts = [
            CUjit_option::CU_JIT_ERROR_LOG_BUFFER,
            CUjit_option::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
            CUjit_option::CU_JIT_INFO_LOG_BUFFER,
            CUjit_option::CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES,
        ];
        let mut vals: [*mut std::ffi::c_void; 4] = [
            log_buf.as_mut_ptr() as *mut _,
            log_size as usize as *mut _,
            info_buf.as_mut_ptr() as *mut _,
            info_size as usize as *mut _,
        ];
        let mut module: CUmodule = std::ptr::null_mut();
        unsafe {
            let _ = cuModuleLoadDataEx(
                &mut module,
                ptx_ptr as *const std::ffi::c_void,
                4,
                opts.as_mut_ptr(),
                vals.as_mut_ptr(),
            );
        }
        let end = log_buf.iter().position(|&b| b == 0).unwrap_or(log_buf.len());
        // Empty log ⇒ JIT succeeded (or wrote nothing). Return 0 so the
        // caller's fallback ("<no log>") kicks in and we don't leak a
        // dummy CString on every call. When end > 0 we *do* leak — this
        // is a test-only diagnostic helper, so the cost is bounded by the
        // number of launch failures per test run (typically one).
        if end == 0 { return 0; }
        let msg = std::ffi::CString::new(&log_buf[..end]).unwrap_or_else(|_| std::ffi::CString::new("<binary>").unwrap());
        let leaked = Box::leak(msg.into_boxed_c_str());
        leaked.as_ptr() as i64
    }
    #[cfg(not(feature = "cuda"))]
    { let _ = ptx_ptr; 0 }
}

/// Copy `bytes` from device pointer `src` to host pointer `dst`.
#[doc(hidden)]
#[no_mangle]
pub extern "C" fn nsl_test_cuda_d2h(dst: i64, src: i64, bytes: i64) {
    #[cfg(feature = "cuda")]
    {
        unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
        inner::memcpy_dtoh(
            dst as *mut std::ffi::c_void,
            src as *const std::ffi::c_void,
            bytes as usize,
        );
    }
    #[cfg(not(feature = "cuda"))]
    { let _ = (dst, src, bytes); }
}

/// Test hook: block until every kernel queued on this context has finished.
///
/// Any integration test that TIMES GPU work needs this. `sync_after_kernel` is
/// a no-op unless `NSL_CUDA_SYNC=1`, so a bare `Instant::now()` loop around
/// `nsl_tensor_matmul` measures kernel *enqueue* — which is how the first
/// throughput probe for item 9 reported a 4096^3 gemm at 22,000 TFLOP/s, and
/// how the first version of `matmul_batch_collapse.rs`'s ratchet passed with
/// the fast path deleted (0.0037 vs 0.0034 ms, ratio 0.92).
///
/// Preferred over setting `NSL_CUDA_SYNC=1` from a test: that variable is read
/// once during context init, so whether it takes effect depends on which test
/// in the binary touched CUDA first.
#[cfg(all(feature = "test-hooks", feature = "cuda"))]
pub fn test_cuda_device_synchronize() {
    inner::ensure_context();
    let rc = unsafe { cudarc::driver::sys::cuCtxSynchronize() };
    assert_eq!(
        rc,
        cudarc::driver::sys::CUresult::CUDA_SUCCESS,
        "cuCtxSynchronize failed: {rc:?}"
    );
}

// ---------------------------------------------------------------------------
// test-hooks: SM version query exposed to integration tests
// ---------------------------------------------------------------------------

#[cfg(all(feature = "test-hooks", feature = "cuda"))]
#[no_mangle]
pub extern "C" fn test_detect_sm_version() -> u32 {
    inner::detect_sm_version()
}

#[cfg(all(feature = "test-hooks", not(feature = "cuda")))]
#[no_mangle]
pub extern "C" fn test_detect_sm_version() -> u32 {
    0
}

/// Static drift gate for the f32-only assumption in this module.
///
/// Deliberately NOT `#[cfg(feature = "cuda")]`, unlike the `mod tests` below.
/// `mod cuda` itself is compiled unconditionally (`lib.rs:90`) — only its
/// items are feature-gated — so a plain `#[cfg(test)]` module here runs under
/// `cargo test -p nsl-runtime --lib`, i.e. inside `cargo test --workspace`
/// (`.github/workflows/ci.yml:59`), on a machine with no GPU and no CUDA
/// toolkit. That is the whole point: every other check in this area needs
/// hardware, so it only runs in the nightly cert lane, and a regression is
/// invisible for a day. This one reddens the ordinary CPU lane on the commit
/// that introduces it.
///
/// It reads this file as text rather than inspecting behaviour because the
/// sites are enumerable — ~45 functions in one file — and #453's lesson was
/// to prefer a static drift gate over a runtime assert when they are.
#[cfg(test)]
mod dtype_guard_drift {
    /// Functions this gate deliberately does not require a guard in, each
    /// with the reason.
    ///
    /// An entry is a CLAIM that gets re-checked below, not a suppression: if
    /// the function is renamed or deleted, stops looking f32-shaped, or grows
    /// a dtype guard of its own, the entry goes stale and the gate fails
    /// until it is removed. Without that reverse check an allowlist only ever
    /// grows, and a line added to silence one refactor keeps silencing every
    /// later one — which is how an exemption table stops being a record of
    /// decisions and becomes a place bugs hide.
    const NO_DTYPE_TENSOR: &[(&str, &str)] = &[
        (
            "gpu_scale_raw_f32",
            "bare *mut c_void device buffer, no NslTensor and so no dtype to read; \
             the ZeRO scatter caller owns the precondition",
        ),
        (
            "gpu_scalar_mul_add_inplace_f32",
            "operands validated by the caller at arithmetic.rs:906 before the \
             device dispatch",
        ),
        (
            "gpu_fase_fused_adamw_step",
            "theta/m/v/mp validated f32 by the caller at fase_step.rs:137-141",
        ),
        (
            "gpu_fase_fused_adamw_step_multi",
            "bucket members are admitted only when dtype == DTYPE_F32 \
             (fase_step.rs:76 and :471), so a non-f32 parameter never reaches a bucket",
        ),
        (
            "gpu_relu_backward",
            "three-line delegator to gpu_backward_binary, which guards both operands \
             and names this kernel in the message",
        ),
        (
            "gpu_sigmoid_backward",
            "delegator to gpu_backward_binary, guarded there",
        ),
        (
            "gpu_tanh_backward",
            "delegator to gpu_backward_binary, guarded there",
        ),
        (
            "gpu_gelu_backward",
            "delegator to gpu_backward_binary, guarded there",
        ),
        (
            "gpu_silu_backward",
            "delegator to gpu_backward_binary, guarded there",
        ),
        (
            "gpu_dequant_int8_per_head_f32",
            "raw device pointers only; the dtype belongs to the quantized source \
             and is checked at the NslTensor-holding caller",
        ),
        (
            "gpu_dequant_int8_per_token_f32",
            "raw device pointers only, as gpu_dequant_int8_per_head_f32",
        ),
        (
            "gpu_dequant_int4_per_group_f32",
            "raw device pointers only, as gpu_dequant_int8_per_head_f32",
        ),
        (
            "gpu_dequant_fp8_e4m3_f32",
            "raw device pointers only, as gpu_dequant_int8_per_head_f32",
        ),
        (
            "gpu_fase_fused_adamw_step_bf16sr",
            "the BF16 stochastic-rounding step: its parameter buffer is bf16 BY \
             DESIGN and the kernel reads it as such",
        ),
        (
            "gpu_sr_bf16_round_probe",
            "bf16 rounding probe — bf16 input is the subject of the test, not a bug",
        ),
        (
            "gpu_cast_raw",
            "the precision-cast launcher: raw src/dst device addresses whose widths \
             are the cast's two endpoints, so f32 on both sides would be the bug",
        ),
        (
            "nsl_kernel_launch",
            "user-supplied PTX with user-supplied arguments; NSL has no way to know \
             what element width the caller's kernel reads",
        ),
        (
            "nsl_kernel_launch_tensors",
            "as nsl_kernel_launch — the tensors are marshalled as opaque device \
             pointers into a kernel this runtime did not write",
        ),
    ];

    /// Split this file into top-level function items.
    ///
    /// Column-0 anchored, and the body ends at the first column-0 `}` — the
    /// same brace-free heuristic the sibling PTX-registration gates use.
    /// Functions nested inside `mod inner` are therefore NOT scanned, which
    /// is correct: `inner` is the raw driver-API layer and holds no
    /// `NslTensor`, so it has no dtype to assert on in the first place.
    fn top_level_fns(source: &str) -> Vec<(&str, usize, String)> {
        let lines: Vec<&str> = source.lines().collect();
        // Stop before the test modules so this gate does not scan itself.
        let end = lines
            .iter()
            .position(|l| l.starts_with("#[cfg(test)]") || l.starts_with("#[cfg(all(test"))
            .unwrap_or(lines.len());

        let mut out = Vec::new();
        let mut i = 0;
        while i < end {
            let line = lines[i];
            let rest = line
                .strip_prefix("pub(crate) ")
                .or_else(|| line.strip_prefix("pub "))
                .unwrap_or(line);
            let rest = rest.strip_prefix("unsafe ").unwrap_or(rest);
            let rest = rest.strip_prefix("extern \"C\" ").unwrap_or(rest);
            let Some(after) = rest.strip_prefix("fn ") else {
                i += 1;
                continue;
            };
            let name_len = after
                .find(|c: char| !(c.is_ascii_alphanumeric() || c == '_'))
                .unwrap_or(after.len());
            if name_len == 0 {
                i += 1;
                continue;
            }
            let name = &after[..name_len];
            let mut j = i;
            while j < end && lines[j] != "}" {
                j += 1;
            }
            out.push((name, i + 1, lines[i..=j.min(end - 1)].join("\n")));
            i = j + 1;
        }
        out
    }

    /// A function is "f32-shaped" if its body sizes a device allocation at 4
    /// bytes per element, names an `*_f32` PTX kernel, or launches a kernel
    /// at all.
    ///
    /// The third rule is not redundant. `gpu_elementwise_unary_inplace` and
    /// its two siblings take the kernel name as a PARAMETER, so the `_f32`
    /// literal lives at ~25 call sites in activation.rs / arithmetic.rs and
    /// not in the function that reads the tensor. The first draft of this
    /// gate missed exactly those three, which are the FBIP arms — the ones a
    /// freshly produced (and therefore uniquely owned) fp16/bf16 tensor
    /// actually takes.
    ///
    /// Loose on purpose: a false positive costs one allowlist line with a
    /// reason, a false negative costs an out-of-bounds device read that
    /// returns plausible numbers.
    fn is_f32_shaped(body: &str) -> bool {
        let sizes_at_four =
            (body.contains("alloc_managed(") || body.contains("try_alloc_managed("))
                && body.contains("* 4");
        let names_an_f32_kernel = body.contains("_f32\\0");
        let launches_a_kernel = body.contains("kernel_launch(");
        sizes_at_four || names_an_f32_kernel || launches_a_kernel
    }

    /// Accepts the shared helper, or a hand-rolled comparison against
    /// `DTYPE_F32` specifically.
    ///
    /// NOT any `.dtype ==` / `.dtype !=`. That was the first spelling, and it
    /// was too loose to do its job: several functions here dispatch on an
    /// INDEX operand's dtype (`idx.dtype == DTYPE_I32` in
    /// `gpu_embedding_lookup`, `gpu_gather_f32`,
    /// `gpu_cross_entropy_backward_f32`), which says nothing at all about
    /// whether the f32 DATA operand beside it was checked. Under the loose
    /// rule those three would have satisfied the gate with their real guards
    /// deleted — i.e. the gate would not have caught the very bug class it
    /// was added for, in three of the functions this commit fixed.
    ///
    /// Requiring `DTYPE_F32` by name also enforces the convention the helper's
    /// own docs argue for: `dtype == 1` greps as nothing, so a reader auditing
    /// "which paths assume f32" cannot find it.
    /// Path-agnostic on purpose: the constant is spelled both `DTYPE_F32` and
    /// `crate::tensor::DTYPE_F32` in this file, and a matcher that pinned one
    /// spelling would fail the honest sites while still passing the loose
    /// ones.
    fn has_dtype_guard(body: &str) -> bool {
        body.contains("assert_gpu_f32(") || (body.contains(".dtype") && body.contains("DTYPE_F32"))
    }

    /// Every f32-assuming function in this module either checks the dtype it
    /// is about to read as f32, or is listed above with a reason.
    ///
    /// The bug this exists for: `nsl_tensor_contiguous` and `nsl_tensor_slice`
    /// routed EVERY device tensor to an f32 kernel on `device > 0` alone,
    /// while their own CPU arms had been dtype-correct for years. Nothing
    /// caught it because there is no build in CI that both compiles this
    /// module and runs a non-f32 GPU tensor through it — `cargo clippy
    /// --workspace` has no `--features cuda`, and `cargo build --workspace
    /// --features cuda` compiles no test target. A text scan is the only
    /// check that survives that gap.
    #[test]
    fn every_f32_assuming_fn_checks_its_operand_dtype() {
        let source = include_str!("mod.rs");
        let fns = top_level_fns(source);

        assert!(
            fns.len() > 80,
            "the top-level fn parser found only {} functions in cuda/mod.rs; the \
             declaration style must have changed and this gate is now scanning \
             almost nothing",
            fns.len()
        );

        // The reason column is the only thing that makes an exemption
        // reviewable, so it has to be load-bearing rather than a field nobody
        // reads: an entry added with `""` to make this test go green would
        // otherwise be indistinguishable from a considered decision.
        for (name, reason) in NO_DTYPE_TENSOR {
            assert!(
                reason.len() >= 30,
                "NO_DTYPE_TENSOR entry for `{name}` has no real reason \
                 ({reason:?}); say WHY the f32 assumption is safe or where the \
                 check actually lives"
            );
        }

        let allowed: std::collections::HashMap<&str, &str> =
            NO_DTYPE_TENSOR.iter().copied().collect();

        let mut missing: Vec<String> = Vec::new();
        let mut guarded = 0usize;
        let mut exempt_and_still_shaped: std::collections::HashSet<&str> =
            std::collections::HashSet::new();

        for (name, line, body) in &fns {
            if !is_f32_shaped(body) {
                continue;
            }
            if has_dtype_guard(body) {
                guarded += 1;
                continue;
            }
            if allowed.contains_key(name) {
                exempt_and_still_shaped.insert(name);
                continue;
            }
            missing.push(format!(
                "  {name} (cuda/mod.rs:{line}) — add `crate::cuda::assert_gpu_f32(t, \
                 \"{name}\", \"<operand>\")` at the head of the function, or add\n        \
                 (\"{name}\", \"<one-line reason>\"),\n      to NO_DTYPE_TENSOR"
            ));
        }

        assert!(
            missing.is_empty(),
            "{} function(s) in cuda/mod.rs size buffers at 4 bytes per element or launch \
             an *_f32 kernel without ever looking at the operand's dtype. A 2-byte dtype \
             (fp16/bf16) there is read past the end of its allocation and returns \
             plausible wrong numbers:\n{}",
            missing.len(),
            missing.join("\n")
        );

        // Anti-vacuity. A gate whose detector silently stops matching passes
        // just as quietly as one whose subject is correct, and the failure
        // mode of a renamed helper is exactly that. 48 of the 66 f32-shaped
        // functions carried a guard when this was written; the floor is set
        // below that with room for a few to be refactored away, and is meant
        // to be RAISED, never lowered.
        assert!(
            guarded >= 40,
            "only {guarded} f32-assuming functions carry a dtype guard, out of {} \
             that look f32-shaped; the guard helper or one of the detection \
             needles must have been renamed, because 48 were guarded when this \
             gate was written",
            fns.iter().filter(|(_, _, b)| is_f32_shaped(b)).count()
        );

        // Reverse direction: an allowlist entry that no longer describes a
        // real, still-unguarded, still-f32-shaped function is a stale claim.
        let stale: Vec<&str> = NO_DTYPE_TENSOR
            .iter()
            .map(|(name, _)| *name)
            .filter(|name| !exempt_and_still_shaped.contains(name))
            .collect();
        assert!(
            stale.is_empty(),
            "NO_DTYPE_TENSOR entries that no longer apply — the function was renamed \
             or deleted, stopped being f32-shaped, or grew a dtype guard of its own. \
             Delete them: {stale:?}"
        );
    }
}

#[cfg(all(test, feature = "cuda"))]
mod tests {
    use super::*;
    use crate::tensor::{DTYPE_F32, NslTensor};

    /// Locate `ptxas` under `$CUDA_PATH`/`$CUDA_HOME`, else on PATH.
    fn find_ptxas() -> Option<std::path::PathBuf> {
        for root in ["CUDA_PATH", "CUDA_HOME"] {
            if let Ok(p) = std::env::var(root) {
                let cand = std::path::Path::new(&p).join("bin").join("ptxas");
                if cand.is_file() {
                    return Some(cand);
                }
            }
        }
        let path = std::env::var_os("PATH")?;
        std::env::split_paths(&path)
            .map(|d| d.join("ptxas"))
            .find(|c| c.is_file())
    }

    /// Lowest real SASS architecture CUDA 13's `ptxas` still generates code for.
    ///
    /// A module's `.target sm_XY` names a *virtual* ISA floor, not a real GPU.
    /// `ptxas` compiles `.target sm_52` for any newer `--gpu-name` just fine; what
    /// CUDA 13 removed is SASS generation for pre-Turing *real* architectures, so
    /// `--gpu-name sm_52` and `--gpu-name sm_70` are rejected. Assemble each module
    /// for the newest of (its declared target, this floor).
    const MIN_PTXAS_GPU_ARCH: u32 = 75;

    /// Assemble every hand-written PTX module in `kernels.rs` and `fused_kernels.rs`.
    ///
    /// These modules are only ever handed to `cuModuleLoadData`, so a syntax error
    /// in one stays invisible until a kernel launch fails on a real GPU. `ptxas`
    /// needs no GPU, only the CUDA toolkit, so this gate runs anywhere the
    /// toolkit is installed.
    /// Runtime-loaded PTX that lives outside the two `ALL_PTX` tables, with a
    /// flag for whether it is standalone-assemblable.
    ///
    /// Both PTX gates read this one list, so a module registered here is covered
    /// by BOTH. Registering in only one was the hole that let a whole kernel
    /// module ship unassembled: an entry in the ASCII gate's local `extra` array
    /// was invisible to the ptxas gate, which iterated the tables alone.
    ///
    /// `false` = a header fragment concatenated into other kernels rather than
    /// loaded on its own; the ptxas gate skips those, the ASCII gate does not
    /// (a non-ASCII byte in a header breaks every kernel that embeds it).
    const EXTRA_RUNTIME_PTX: &[(&str, &str, bool)] = &[
        (
            "CSHA_TIER_B1_PREPASS_X_PTX",
            super::tier_b1_prepass::CSHA_TIER_B1_PREPASS_X_PTX,
            true,
        ),
        (
            "CSHA_TIER_B1_PREPASS_W_PTX",
            super::tier_b1_prepass::CSHA_TIER_B1_PREPASS_W_PTX,
            true,
        ),
        (
            "PTX_F32_TO_BF16",
            super::precision_cast_kernels::PTX_F32_TO_BF16,
            true,
        ),
        (
            "PTX_BF16_TO_F32",
            super::precision_cast_kernels::PTX_BF16_TO_F32,
            true,
        ),
        (
            "PTX_F32_TO_FP16",
            super::precision_cast_kernels::PTX_F32_TO_FP16,
            true,
        ),
        (
            "PTX_FP16_TO_F32",
            super::precision_cast_kernels::PTX_FP16_TO_F32,
            true,
        ),
        (
            "STRIDED_COPY_RUN_PTX",
            super::strided_copy::STRIDED_COPY_RUN_PTX,
            true,
        ),
        (
            "HOPPER_PTX_HEADER",
            super::kernels_hopper::HOPPER_PTX_HEADER,
            false,
        ),
    ];

    #[test]
    fn all_handwritten_ptx_assembles_with_ptxas() {
        let Some(ptxas) = find_ptxas() else {
            eprintln!("ptxas not found (no CUDA toolkit); skipping PTX assembly gate");
            return;
        };
        let dir = std::env::temp_dir().join(format!("nsl_ptx_gate_{}", std::process::id()));
        std::fs::create_dir_all(&dir).expect("create temp dir");

        let mut failures = Vec::new();
        let modules = super::kernels::ALL_PTX
            .iter()
            .chain(super::fused_kernels::ALL_PTX.iter())
            .map(|(name, ptx)| (*name, *ptx))
            .chain(
                EXTRA_RUNTIME_PTX
                    .iter()
                    .filter(|(_, _, standalone)| *standalone)
                    .map(|(name, ptx, _)| (*name, *ptx)),
            );

        for (name, ptx) in modules {
            // These constants carry a trailing NUL for `cuModuleLoadData`. ptxas
            // reads an embedded NUL as a premature EOF, so strip it before writing.
            let src = ptx.strip_suffix('\0').unwrap_or(ptx);
            assert!(
                !src.contains('\0'),
                "{name}: NUL byte inside the PTX body, not just at the end"
            );

            let target = src
                .lines()
                .find_map(|l| l.trim().strip_prefix(".target "))
                .unwrap_or_else(|| panic!("{name}: PTX has no .target directive"))
                .trim();
            let declared: u32 = target
                .strip_prefix("sm_")
                .and_then(|n| n.trim_end_matches('a').parse().ok())
                .unwrap_or_else(|| panic!("{name}: unparsable .target `{target}`"));
            let gpu_arch = format!("sm_{}", declared.max(MIN_PTXAS_GPU_ARCH));

            let in_path = dir.join(format!("{name}.ptx"));
            std::fs::write(&in_path, src).expect("write ptx");
            let out = std::process::Command::new(&ptxas)
                .args(["--gpu-name", &gpu_arch, "-o"])
                .arg(dir.join(format!("{name}.cubin")))
                .arg(&in_path)
                .output()
                .expect("run ptxas");

            if !out.status.success() {
                failures.push(format!(
                    "{name} (.target {target}, --gpu-name {gpu_arch}):\n{}",
                    String::from_utf8_lossy(&out.stderr).trim()
                ));
            }
        }

        let _ = std::fs::remove_dir_all(&dir);
        assert!(
            failures.is_empty(),
            "ptxas rejected {} hand-written PTX module(s):\n\n{}",
            failures.len(),
            failures.join("\n\n")
        );
    }

    /// Every hand-written PTX string handed to the driver JIT must be pure
    /// ASCII.
    ///
    /// The CUDA driver's JIT (`cuModuleLoadData`, the runtime loader for these
    /// modules) rejects a non-ASCII byte ANYWHERE in the source -- including
    /// inside a comment -- with CUDA_ERROR_INVALID_PTX. Offline `ptxas` under a
    /// UTF-8 locale accepts the same bytes, so the companion assembly gate
    /// (`all_handwritten_ptx_assembles_with_ptxas`) cannot catch this: a stray
    /// em-dash / arrow / times sign in a comment ships a kernel that silently
    /// fails to load on a real GPU (the launch returns an error the call site
    /// may swallow, or asserts). Two such em-dashes -- in MAXPOOL2D_F32_PTX and
    /// COO_SPMM_F32_PTX -- shipped invalid for exactly this reason.
    ///
    /// Coverage: the two registered kernel tables (`kernels::ALL_PTX` +
    /// `fused_kernels::ALL_PTX`), plus the runtime-loaded PTX in the `cuda::`
    /// submodules that those tables omit -- the tier-B1 prepass kernels (which
    /// had NO ASCII guard before this), the precision-cast kernels (also
    /// covered by their own `embedded_ptx_strings_are_ascii`), and the shared
    /// Hopper header assembled via `push_str`. PTX defined outside `cuda::` --
    /// the CSHA kernels in `flash_attention.rs` -- is out of this gate's
    /// module reach and is guarded by that file's own tests
    /// (`multitile_postpass_ptx_is_ascii_and_nul_terminated` and
    /// `csha_cast_ptx_is_ascii_and_nul_terminated`).
    #[test]
    fn all_handwritten_ptx_is_pure_ascii() {
        // Runtime-loaded PTX outside the two ALL_PTX tables comes from the
        // shared `EXTRA_RUNTIME_PTX` registry, which the ptxas gate reads too.
        let modules = super::kernels::ALL_PTX
            .iter()
            .chain(super::fused_kernels::ALL_PTX.iter())
            .map(|(name, ptx)| (*name, *ptx))
            .chain(EXTRA_RUNTIME_PTX.iter().map(|(name, ptx, _)| (*name, *ptx)));
        let mut failures = Vec::new();
        for (name, ptx) in modules {
            for (i, line) in ptx.lines().enumerate() {
                if !line.is_ascii() {
                    let bad: Vec<char> = line.chars().filter(|c| !c.is_ascii()).collect();
                    failures.push(format!(
                        "{name}: non-ASCII on PTX line {} {bad:?} -- driver JIT rejects this \
                         as CUDA_ERROR_INVALID_PTX (offline ptxas does not). Line: {line:?}",
                        i + 1
                    ));
                }
            }
        }
        assert!(
            failures.is_empty(),
            "{} hand-written PTX module(s) contain non-ASCII bytes:\n\n{}",
            failures.len(),
            failures.join("\n")
        );
    }

    /// Every `*_PTX` constant in `fused_kernels` must appear in `ALL_PTX`.
    ///
    /// `ALL_PTX` is the sole input to the ptxas-assembles and pure-ASCII gates,
    /// so a constant missing from it is a kernel those gates never see — its
    /// syntax errors stay invisible until a launch fails on a real GPU. This
    /// happened: `SUM_DIM_SHORT_F32_PTX` was added and became the default path
    /// for the majority of `sum_dim` calls while absent from the table, and
    /// nothing noticed because no gate compared the two.
    #[test]
    fn every_ptx_constant_is_registered_for_certification() {
        let source = include_str!("fused_kernels.rs");

        let mut declared: Vec<&str> = Vec::new();
        for line in source.lines() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed.strip_prefix("pub(crate) const ") else {
                continue;
            };
            let mut parts = rest.splitn(2, ':');
            let Some(name) = parts.next().map(str::trim) else {
                continue;
            };
            // Only `&str` constants are kernels. This also excludes the
            // `ALL_PTX` table itself, whose name ends in _PTX but whose type is
            // a slice of pairs.
            let is_str = parts.next().is_some_and(|ty| ty.trim_start().starts_with("&str"));
            if name.ends_with("_PTX") && is_str {
                declared.push(name);
            }
        }
        assert!(
            declared.len() > 40,
            "parser found only {} PTX constants; the declaration form must have changed",
            declared.len()
        );

        let registered: std::collections::HashSet<&str> = fused_kernels::ALL_PTX
            .iter()
            .map(|(name, _)| *name)
            .chain(EXTRA_RUNTIME_PTX.iter().map(|(name, _, _)| *name))
            .collect();

        // Fragments that are concatenated into other kernels rather than loaded
        // on their own; `EXTRA_RUNTIME_PTX` covers them for the ASCII gate.
        const NOT_STANDALONE: &[&str] = &["HOPPER_PTX_HEADER_PTX"];

        let missing: Vec<&str> = declared
            .iter()
            .copied()
            .filter(|name| !registered.contains(name) && !NOT_STANDALONE.contains(name))
            .collect();
        assert!(
            missing.is_empty(),
            "PTX constants absent from ALL_PTX, so neither the ptxas gate nor the \
             ASCII gate covers them: {missing:?}"
        );
    }

    /// The same drift check, for `cuda::` submodules that define runtime-loaded
    /// PTX outside `fused_kernels`.
    ///
    /// `every_ptx_constant_is_registered_for_certification` only parses
    /// `fused_kernels.rs`, so a `_PTX` constant added to any sibling module was
    /// invisible to it AND to both PTX gates — a kernel could ship never having
    /// been assembled or ASCII-checked. That is exactly the
    /// `SUM_DIM_SHORT_F32_PTX` failure the original gate was written to prevent,
    /// reappearing one module over.
    #[test]
    fn sibling_module_ptx_constants_are_registered() {
        let sources: &[(&str, &str)] = &[
            ("strided_copy.rs", include_str!("strided_copy.rs")),
            ("tier_b1_prepass.rs", include_str!("tier_b1_prepass.rs")),
            (
                "precision_cast_kernels.rs",
                include_str!("precision_cast_kernels.rs"),
            ),
        ];
        let registered: std::collections::HashSet<&str> =
            EXTRA_RUNTIME_PTX.iter().map(|(name, _, _)| *name).collect();

        let mut missing = Vec::new();
        for (file, source) in sources {
            for line in source.lines() {
                let trimmed = line.trim_start();
                let Some(rest) = trimmed
                    .strip_prefix("pub(crate) const ")
                    .or_else(|| trimmed.strip_prefix("pub const "))
                else {
                    continue;
                };
                let mut parts = rest.splitn(2, ':');
                let Some(name) = parts.next().map(str::trim) else {
                    continue;
                };
                let is_str = parts
                    .next()
                    .is_some_and(|ty| ty.trim_start().starts_with("&str"));
                if is_str
                    && (name.ends_with("_PTX") || name.starts_with("PTX_"))
                    && !registered.contains(name)
                {
                    missing.push(format!("{file}::{name}"));
                }
            }
        }
        assert!(
            missing.is_empty(),
            "PTX constants in cuda:: submodules that are not in EXTRA_RUNTIME_PTX, so no \
             gate assembles or ASCII-checks them: {missing:?}"
        );
    }


    const VEC_ADD_PTX: &str = "\
.version 7.0
.target sm_70
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

        // Despite the name, `alloc_managed` does NOT return unified memory: it
        // routes through the caching allocator, which calls `cuMemAlloc_v2`.
        // `cuMemAllocManaged` is never called anywhere in the runtime. These are
        // plain device pointers, so the host must stage through memcpy. An
        // earlier version of this test built a `&mut [f32]` over `a` and wrote
        // through it, which segfaulted and aborted the entire test binary --
        // masking every cuda test ordered after it.
        let a = inner::alloc_managed(size_bytes);
        let b = inner::alloc_managed(size_bytes);
        let c = inner::alloc_managed(size_bytes);

        let a_host: Vec<f32> = (0..n).map(|i| i as f32).collect();
        let b_host: Vec<f32> = (0..n).map(|i| (i * 2) as f32).collect();
        inner::memcpy_htod(a, a_host.as_ptr() as *const c_void, size_bytes);
        inner::memcpy_htod(b, b_host.as_ptr() as *const c_void, size_bytes);

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

        // Copy results back to the host before reading them.
        let mut c_host = vec![0.0f32; n];
        inner::memcpy_dtoh(c_host.as_mut_ptr() as *mut c_void, c, size_bytes);
        for i in 0..n {
            let expected = (i + i * 2) as f32;
            assert_eq!(c_host[i], expected, "mismatch at index {}", i);
        }

        // Cleanup
        inner::free_managed(a);
        inner::free_managed(b);
        inner::free_managed(c);
    }

    #[test]
    fn tensor_shape_slice_handles_zero_rank_null_shape() {
        let scalar = NslTensor::new(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            1,
            1,
            DTYPE_F32,
            0,
            0,
        );

        assert!(tensor_shape_slice(&scalar).is_empty());
    }

    #[test]
    fn gpu_broadcast_shape_accepts_zero_rank_scalar_tensor() {
        let scalar = NslTensor::new(
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            1,
            1,
            DTYPE_F32,
            0,
            0,
        );
        let mut vec_shape = [4_i64];
        let vector = NslTensor::new(
            std::ptr::null_mut(),
            vec_shape.as_mut_ptr(),
            std::ptr::null_mut(),
            1,
            4,
            1,
            DTYPE_F32,
            0,
            0,
        );

        assert_eq!(gpu_broadcast_shape(&vector, &scalar), Some(vec![4]));
        assert_eq!(gpu_broadcast_shape(&scalar, &vector), Some(vec![4]));
    }

    #[test]
    fn test_alloc_free_device() {
        // Allocate 1024 bytes of device-only memory and free it — no crash expected.
        let ptr = inner::alloc_device(1024);
        assert!(!ptr.is_null(), "alloc_device returned null");
        inner::free_device(ptr);
    }

    /// p3-remainder: the stream-ordered deferred free must physically return a
    /// raw `alloc_device` buffer to the driver once drained — it is a latency
    /// shift, never a leak. Mirrors the CSHA save-buffer regression test but
    /// exercises the raw `alloc_device`→`defer_free_device`→drain path directly.
    ///
    /// A broken deferred free (never physically freeing) would leak ~200 MB
    /// over the loop; the 128 MB threshold has generous headroom for
    /// concurrent tests / desktop VRAM noise while still catching a regression.
    #[test]
    fn test_deferred_free_device_reclaims_vram() {
        const ITERS: usize = 50;
        const BUF_BYTES: usize = 4 * 1024 * 1024; // 4 MB → ~200 MB total
        const MAX_ALLOWED_DROP_BYTES: usize = 128 * 1024 * 1024;

        // Establish the context and flush any prior stragglers so the delta is
        // attributable to this loop.
        inner::ensure_context();
        inner::drain_all_deferred_frees();
        let (free_before, _total) = inner::query_vram();
        for _ in 0..ITERS {
            let ptr = inner::alloc_device(BUF_BYTES);
            assert!(!ptr.is_null(), "alloc_device returned null");
            inner::defer_free_device(ptr);
        }
        // Force every deferred free to complete, then measure.
        inner::drain_all_deferred_frees();
        let (free_after, _total) = inner::query_vram();

        let dropped = free_before.saturating_sub(free_after);
        assert!(
            dropped < MAX_ALLOWED_DROP_BYTES,
            "deferred alloc/free cycle leaked device memory: free VRAM dropped \
             by {} bytes over {} iterations (before={}, after={})",
            dropped, ITERS, free_before, free_after,
        );
    }

    /// The deferred-free guards must tolerate null pointers, all-null batches,
    /// and draining an empty queue without touching the driver or panicking.
    #[test]
    fn test_deferred_free_handles_null_and_empty() {
        // Null single free is a no-op.
        inner::defer_free_device(std::ptr::null_mut());
        // An all-null batch enqueues nothing.
        inner::defer_free_device_batch(&[std::ptr::null_mut(), std::ptr::null_mut()]);
        // Draining an empty (or already-drained) queue must not sync or panic.
        inner::drain_completed_frees();
        inner::drain_all_deferred_frees();
        // A batch that mixes null and live buffers frees only the live ones.
        let a = inner::alloc_device(1024);
        let b = inner::alloc_device(1024);
        assert!(!a.is_null() && !b.is_null());
        inner::defer_free_device_batch(&[a, std::ptr::null_mut(), b]);
        inner::drain_all_deferred_frees();
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

    /// P0.2: the pinned registry must track alloc_pinned buffers so the
    /// tensor free path can route them to cuMemFreeHost.
    #[test]
    fn test_pinned_registry_tracks_alloc_and_free() {
        let ptr = inner::alloc_pinned(128);
        assert!(inner::is_pinned(ptr), "alloc_pinned must register the buffer");
        inner::free_pinned(ptr);
        assert!(!inner::is_pinned(ptr), "free_pinned must unregister the buffer");
    }

    /// P0.2: async DtoH on the per-thread transfer stream lands after
    /// transfer_stream_synchronize (round-trip through device memory).
    #[test]
    fn test_transfer_stream_async_dtoh_roundtrip() {
        let vals: Vec<f32> = (0..256).map(|i| i as f32 * 0.5 - 3.0).collect();
        let bytes = vals.len() * std::mem::size_of::<f32>();
        let dev = inner::alloc_device(bytes);
        inner::memcpy_htod(dev, vals.as_ptr() as *const std::ffi::c_void, bytes);
        let pinned = inner::alloc_pinned(bytes);
        inner::memcpy_dtoh_async(pinned, dev, bytes);
        inner::transfer_stream_synchronize();
        let got = unsafe { std::slice::from_raw_parts(pinned as *const f32, vals.len()) };
        assert_eq!(got, vals.as_slice(), "async DtoH must be complete after stream sync");
        inner::free_pinned(pinned);
        inner::free_device(dev);
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
        let c_gpu = nsl_tensor_matmul(a_gpu, b_gpu, 0);

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
        let c_gpu = nsl_tensor_add(a_gpu, b_gpu, 0);

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
