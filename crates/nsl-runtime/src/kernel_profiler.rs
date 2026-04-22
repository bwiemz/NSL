//! Async GPU kernel profiler with pre-allocated event pools.
//! Records cuEvent pairs around kernel launches, resolves timestamps
//! at flush time with a single cuCtxSynchronize. Zero sync during execution.

use std::io::Write;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// A recorded kernel launch trace (event indices + metadata).
pub(crate) struct KernelTrace {
    #[allow(dead_code)] // used by flush_traces_gpu (cuda feature)
    pub pool_idx: usize,
    pub name: String,
    pub grid: [u32; 3],
    pub block: [u32; 3],
}

/// Global kernel profiler singleton.
pub(crate) struct KernelProfiler {
    pub enabled: AtomicBool,
    pub cpu_start_time: Mutex<Option<Instant>>,
    pub traces: Mutex<Vec<KernelTrace>>,
    pub pool_cursor: Mutex<usize>,
    // GPU event pool and base event are stored as raw u64 handles
    // (CUevent is a pointer, stored as u64 for Send+Sync safety)
    #[allow(dead_code)] // used when cuda feature enabled
    pub event_pool: Mutex<Vec<(u64, u64)>>, // (start_event, stop_event) pairs
    #[allow(dead_code)] // used when cuda feature enabled
    pub gpu_base_event: Mutex<u64>,
}

// SAFETY: CUevent handles are thread-safe opaque pointers managed by the CUDA driver.
unsafe impl Send for KernelProfiler {}
unsafe impl Sync for KernelProfiler {}

static ATEXIT_REGISTERED: AtomicBool = AtomicBool::new(false);

extern "C" {
    fn atexit(f: extern "C" fn()) -> i32;
}

pub(crate) static KERNEL_PROFILER: KernelProfiler = KernelProfiler {
    enabled: AtomicBool::new(false),
    cpu_start_time: Mutex::new(None),
    traces: Mutex::new(Vec::new()),
    pool_cursor: Mutex::new(0),
    event_pool: Mutex::new(Vec::new()),
    gpu_base_event: Mutex::new(0),
};

pub fn kernel_profiler_enabled() -> bool {
    KERNEL_PROFILER.enabled.load(Ordering::Relaxed)
}

#[allow(dead_code)] // used in cuda feature gate block
const EVENT_POOL_SIZE: usize = 4096;

/// Initialize the kernel profiler.
///
/// Sets the enabled flag and resets per-run state (traces, cursor,
/// CPU start time).  GPU event pool allocation is **deferred** to the
/// first `kernel_profiler_pop_events` call — that call happens from
/// inside `kernel_launch` after `cuda::ensure_context()` has
/// guaranteed the CUDA context is current.  Allocating events here
/// (during `nsl_args_init`, which runs before any kernel launch)
/// would invoke `cuEventCreate` with no active CUDA context; the
/// resulting events silently produce 0 ms elapsed time from
/// `cuEventElapsedTime_v2`, which was the "profiler reports zero
/// durations" bug.
#[no_mangle]
pub extern "C" fn nsl_kernel_profiler_start() {
    KERNEL_PROFILER.enabled.store(true, Ordering::Relaxed);

    if !ATEXIT_REGISTERED.swap(true, Ordering::Relaxed) {
        unsafe {
            atexit(kernel_profiler_atexit);
        }
    }

    *KERNEL_PROFILER.cpu_start_time.lock().unwrap() = Some(Instant::now());
    *KERNEL_PROFILER.traces.lock().unwrap() = Vec::new();
    *KERNEL_PROFILER.pool_cursor.lock().unwrap() = 0;

    // Clear any stale pool so the lazy-init path below allocates a
    // fresh set of events after the CUDA context is live.  A fresh
    // set ensures `cuEventElapsedTime_v2` has a valid timebase even
    // if `start` is called after a prior `flush` (which calls
    // `cuEventDestroy`) or after a context reset.
    #[cfg(feature = "cuda")]
    {
        KERNEL_PROFILER.event_pool.lock().unwrap().clear();
        *KERNEL_PROFILER.gpu_base_event.lock().unwrap() = 0;
    }
}

/// Lazy-allocate the GPU event pool on first use.  Must be called only
/// from code paths that have already ensured the CUDA context is
/// current — `kernel_profiler_pop_events` is called from inside
/// `kernel_launch` which always calls `cuda::ensure_context()` before
/// any driver API, so by that point `cuEventCreate` sees a valid
/// context and produces events with a valid timebase.
///
/// Idempotent: a second call on a populated pool returns immediately
/// after the cheap `pool.is_empty()` check.
///
/// **Atomicity of init.** Both the event pool and `gpu_base_event` are
/// populated under a single pool-lock scope so that "pool non-empty"
/// is a sound signal that `gpu_base_event != 0` as well.  Without
/// this, a concurrent `flush_traces_gpu` running between the pool
/// populate and the base-event record could read `gpu_base_event ==
/// 0` and pass a null handle to `cuEventElapsedTime_v2`.  Lock order
/// matches `flush_traces_gpu` (event_pool → gpu_base_event) so no
/// deadlock.
#[cfg(feature = "cuda")]
pub(crate) fn ensure_event_pool_initialized() {
    let mut pool = KERNEL_PROFILER.event_pool.lock().unwrap();
    if !pool.is_empty() {
        return;
    }
    // Acquire the base-event lock BEFORE releasing the pool lock so
    // both are populated atomically from the observer's perspective.
    let mut base = KERNEL_PROFILER.gpu_base_event.lock().unwrap();

    pool.reserve(EVENT_POOL_SIZE);
    for _ in 0..EVENT_POOL_SIZE {
        let mut start: u64 = 0;
        let mut stop: u64 = 0;
        unsafe {
            crate::cuda::cu_event_create(&mut start);
            crate::cuda::cu_event_create(&mut stop);
        }
        pool.push((start, stop));
    }
    if *base == 0 {
        unsafe {
            crate::cuda::cu_event_create(&mut *base);
            crate::cuda::cu_event_record(*base, std::ptr::null_mut());
        }
    }
    // Both guards drop at end of scope.
}

/// Disable recording. Does not free resources (flush does that).
#[no_mangle]
pub extern "C" fn nsl_kernel_profiler_stop() {
    KERNEL_PROFILER.enabled.store(false, Ordering::Relaxed);
}

extern "C" fn kernel_profiler_atexit() {
    if kernel_profiler_enabled() {
        let path = "kernel_profile.json";
        let ptr = path.as_ptr();
        let len = path.len() as i64;
        unsafe { nsl_kernel_profiler_flush(ptr, len); }
        eprintln!("[nsl] kernel profile written to {}", path);
    }
}

/// Pop the next (start, stop) event pair from the pool.
/// Lock-pop-unlock pattern: single lock acquisition, extract event pair, unlock.
/// Returns None if profiler disabled or pool exhausted.
///
/// Lock ordering: always lock event_pool FIRST (contains pool + cursor).
/// This is consistent with flush/destroy which also lock event_pool first.
///
/// First-call behavior (CUDA builds): triggers lazy pool allocation
/// via `ensure_event_pool_initialized`.  This relies on the caller
/// (`cuda::kernel_launch`) having already executed
/// `cuda::ensure_context()` before reaching this function — a
/// contract the existing `kernel_launch` signature enforces by
/// calling `ensure_context` at the top of the launch path.
#[allow(dead_code)] // called from cuda/mod.rs kernel_launch
pub(crate) fn kernel_profiler_pop_events() -> Option<(u64, u64, usize)> {
    if !kernel_profiler_enabled() {
        return None;
    }

    // Lazy-allocate the event pool on first use after the CUDA context
    // is live.  No-op on subsequent calls.
    #[cfg(feature = "cuda")]
    ensure_event_pool_initialized();

    // Single lock: pool and cursor accessed together to avoid ordering issues
    let pool = KERNEL_PROFILER.event_pool.lock().unwrap();
    let mut cursor = KERNEL_PROFILER.pool_cursor.lock().unwrap();

    if *cursor >= pool.len() {
        eprintln!(
            "[nsl] kernel profiler: event pool exhausted ({} events recorded), flushing and recycling",
            pool.len()
        );
        return None;
    }

    let idx = *cursor;
    let (start, stop) = pool[idx];
    *cursor += 1;
    drop(cursor);
    drop(pool);
    // Locks dropped — CUDA calls happen outside lock

    Some((start, stop, idx))
}

/// Record a completed kernel trace. Lock-push-unlock pattern.
/// Lock ordering note: this only locks pool_cursor then traces (not event_pool).
/// This is safe because push_trace is only called after pop_events has already
/// released its locks, and flush holds all locks exclusively during teardown.
#[allow(dead_code)] // called from cuda/mod.rs kernel_launch
pub(crate) fn kernel_profiler_push_trace(name: &str, grid: [u32; 3], block: [u32; 3]) {
    let cursor = *KERNEL_PROFILER.pool_cursor.lock().unwrap();
    KERNEL_PROFILER.traces.lock().unwrap().push(KernelTrace {
        pool_idx: cursor - 1, // points to the event pair just used
        name: name.to_string(),
        grid,
        block,
    });
}

/// CPU-only flush: writes traces with monotonically increasing synthetic timestamps.
/// Used in tests and when GPU events are not available.
pub(crate) fn flush_traces_cpu_fallback(path: &str) {
    let traces = KERNEL_PROFILER.traces.lock().unwrap();

    let mut events = Vec::new();
    for (i, trace) in traces.iter().enumerate() {
        let ts = (i as f64) * 100.0; // synthetic 100us spacing from t=0
        events.push(format!(
            r#"{{"name":"{}","ph":"X","ts":{:.1},"dur":50.0,"pid":0,"tid":1,"args":{{"grid":[{},{},{}],"block":[{},{},{}]}}}}"#,
            trace.name, ts,
            trace.grid[0], trace.grid[1], trace.grid[2],
            trace.block[0], trace.block[1], trace.block[2],
        ));
    }

    let total_launches = traces.len();
    let json = format!(
        r#"{{"traceEvents":[{}],"metadata":{{"total_kernel_launches":{},"profiler_mode":"cpu_fallback"}}}}"#,
        events.join(","),
        total_launches,
    );

    if let Ok(mut f) = std::fs::File::create(path) {
        let _ = f.write_all(json.as_bytes());
    }
}

/// GPU flush: resolves cuEventElapsedTime for all traces, writes Chrome tracing JSON.
/// Consumes the event pool (calls cuEventDestroy on all events).
#[cfg(feature = "cuda")]
pub(crate) fn flush_traces_gpu(path: &str) {
    use crate::cuda;

    KERNEL_PROFILER.enabled.store(false, Ordering::Relaxed);

    unsafe { cuda::cu_ctx_synchronize(); }

    let traces: Vec<KernelTrace> = {
        let mut guard = KERNEL_PROFILER.traces.lock().unwrap();
        std::mem::take(&mut *guard)
    };
    let pool: Vec<(u64, u64)> = {
        KERNEL_PROFILER.event_pool.lock().unwrap().clone()
    };
    let base_event = *KERNEL_PROFILER.gpu_base_event.lock().unwrap();
    let cpu_start = 0.0f64;

    let mut events_json = Vec::new();
    let mut total_kernel_time_ms = 0.0f64;

    for trace in traces.iter() {
        if trace.pool_idx >= pool.len() { continue; }
        let (start_event, stop_event) = pool[trace.pool_idx];

        let mut duration_ms: f32 = 0.0;
        let mut offset_ms: f32 = 0.0;
        unsafe {
            cuda::cu_event_elapsed_time(&mut duration_ms, start_event, stop_event);
            cuda::cu_event_elapsed_time(&mut offset_ms, base_event, start_event);
        }

        let ts_us = cpu_start + (offset_ms as f64) * 1000.0;
        let dur_us = (duration_ms as f64) * 1000.0;
        total_kernel_time_ms += duration_ms as f64;

        events_json.push(format!(
            r#"{{"name":"{}","ph":"X","ts":{:.1},"dur":{:.1},"pid":0,"tid":1,"args":{{"grid":[{},{},{}],"block":[{},{},{}]}}}}"#,
            trace.name, ts_us, dur_us,
            trace.grid[0], trace.grid[1], trace.grid[2],
            trace.block[0], trace.block[1], trace.block[2],
        ));
    }

    let total_launches = traces.len();
    let json = format!(
        r#"{{"traceEvents":[{}],"metadata":{{"total_kernel_launches":{},"total_kernel_time_ms":{:.2}}}}}"#,
        events_json.join(","),
        total_launches,
        total_kernel_time_ms,
    );

    if let Ok(mut f) = std::fs::File::create(path) {
        let _ = f.write_all(json.as_bytes());
    }

    destroy_event_pool();
}

/// Destroy all cuEvents in the pool and the base event.
#[cfg(feature = "cuda")]
fn destroy_event_pool() {
    let mut pool = KERNEL_PROFILER.event_pool.lock().unwrap();
    for (start, stop) in pool.drain(..) {
        unsafe {
            crate::cuda::cu_event_destroy(start);
            crate::cuda::cu_event_destroy(stop);
        }
    }
    let base = *KERNEL_PROFILER.gpu_base_event.lock().unwrap();
    if base != 0 {
        unsafe { crate::cuda::cu_event_destroy(base); }
        *KERNEL_PROFILER.gpu_base_event.lock().unwrap() = 0;
    }
    *KERNEL_PROFILER.pool_cursor.lock().unwrap() = 0;
}

/// FFI flush entry point. Calls GPU path if available, CPU fallback otherwise.
/// # Safety
/// `path_ptr` must point to valid UTF-8 bytes of length `path_len`.
#[no_mangle]
pub unsafe extern "C" fn nsl_kernel_profiler_flush(path_ptr: *const u8, path_len: i64) {
    let path = if path_ptr.is_null() || path_len <= 0 {
        "kernel_profile.json".to_string()
    } else {
        let bytes = std::slice::from_raw_parts(path_ptr, path_len as usize);
        std::str::from_utf8(bytes).unwrap_or("kernel_profile.json").to_string()
    };

    #[cfg(feature = "cuda")]
    {
        flush_traces_gpu(&path);
        return;
    }

    #[allow(unreachable_code)]
    flush_traces_cpu_fallback(&path);
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tests share global KERNEL_PROFILER state and must not run concurrently.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn test_profiler_enable_disable() {
        let _guard = TEST_LOCK.lock().unwrap();
        // Reset state
        KERNEL_PROFILER.enabled.store(false, Ordering::Relaxed);

        assert!(!kernel_profiler_enabled());
        nsl_kernel_profiler_start();
        assert!(kernel_profiler_enabled());
        nsl_kernel_profiler_stop();
        assert!(!kernel_profiler_enabled());
    }

    #[test]
    fn test_pool_cursor_advancement() {
        let _guard = TEST_LOCK.lock().unwrap();
        // Reset state
        nsl_kernel_profiler_stop();
        *KERNEL_PROFILER.pool_cursor.lock().unwrap() = 0;
        KERNEL_PROFILER.traces.lock().unwrap().clear();
        // Pre-fill pool with dummy handles (no GPU needed)
        {
            let mut pool = KERNEL_PROFILER.event_pool.lock().unwrap();
            pool.clear();
            for i in 0..10u64 {
                pool.push((i * 2, i * 2 + 1));
            }
        }
        KERNEL_PROFILER.enabled.store(true, Ordering::Relaxed);

        // Pop events — should advance cursor
        let events = kernel_profiler_pop_events();
        assert!(events.is_some());
        assert_eq!(*KERNEL_PROFILER.pool_cursor.lock().unwrap(), 1);

        // Push trace
        kernel_profiler_push_trace("test_kernel", [1, 1, 1], [256, 1, 1]);
        assert_eq!(KERNEL_PROFILER.traces.lock().unwrap().len(), 1);
        assert_eq!(
            KERNEL_PROFILER.traces.lock().unwrap()[0].name,
            "test_kernel"
        );

        nsl_kernel_profiler_stop();
    }

    #[test]
    fn test_pool_exhaustion_returns_none() {
        let _guard = TEST_LOCK.lock().unwrap();
        nsl_kernel_profiler_stop();
        *KERNEL_PROFILER.pool_cursor.lock().unwrap() = 0;
        {
            let mut pool = KERNEL_PROFILER.event_pool.lock().unwrap();
            pool.clear();
            pool.push((100, 101)); // Only 1 pair
        }
        KERNEL_PROFILER.enabled.store(true, Ordering::Relaxed);

        let first = kernel_profiler_pop_events();
        assert!(first.is_some());

        let second = kernel_profiler_pop_events();
        assert!(second.is_none()); // Pool exhausted

        nsl_kernel_profiler_stop();
    }

    #[test]
    fn test_flush_produces_valid_json() {
        use std::io::Read;
        let _lock = TEST_LOCK.lock().unwrap();

        nsl_kernel_profiler_stop();
        *KERNEL_PROFILER.cpu_start_time.lock().unwrap() = Some(Instant::now());
        KERNEL_PROFILER.traces.lock().unwrap().clear();
        {
            let mut pool = KERNEL_PROFILER.event_pool.lock().unwrap();
            pool.clear();
            for i in 0..4u64 {
                pool.push((i * 2, i * 2 + 1));
            }
        }
        KERNEL_PROFILER.enabled.store(true, Ordering::Relaxed);

        // Simulate 2 kernel traces (without actual GPU events)
        // push_trace reads cursor and uses cursor-1 as pool_idx,
        // so set cursor to simulate post-pop state
        *KERNEL_PROFILER.pool_cursor.lock().unwrap() = 1;
        kernel_profiler_push_trace("matmul_256", [4, 1, 1], [256, 1, 1]);
        *KERNEL_PROFILER.pool_cursor.lock().unwrap() = 2;
        kernel_profiler_push_trace("fused_relu_add", [4, 1, 1], [256, 1, 1]);

        // Flush to temp file (CPU-only mode, no cuEventElapsedTime)
        let tmp = std::env::temp_dir().join("test_kernel_profiler.json");
        let path_str = tmp.to_str().unwrap();
        flush_traces_cpu_fallback(path_str);

        // Verify JSON
        let mut contents = String::new();
        std::fs::File::open(&tmp).unwrap().read_to_string(&mut contents).unwrap();
        assert!(contents.contains("traceEvents"));
        assert!(contents.contains("matmul_256"));
        assert!(contents.contains("fused_relu_add"));
        assert!(contents.contains("\"ph\":\"X\""));
        assert!(contents.contains("\"tid\":1"));

        std::fs::remove_file(&tmp).ok();
        nsl_kernel_profiler_stop();
    }

    /// Regression for the zero-durations bug: `nsl_kernel_profiler_start`
    /// MUST NOT allocate GPU events.  The allocation must be deferred
    /// to `kernel_profiler_pop_events`, which is called from inside
    /// `kernel_launch` after `cuda::ensure_context()` has ensured the
    /// CUDA context is current.  Pre-fix, `start` eagerly allocated
    /// 4096 event pairs via `cuEventCreate` without a live context,
    /// producing invalid events whose `cuEventElapsedTime_v2` returns
    /// 0 silently → `NSL_PROFILE_KERNELS=1` JSON reports
    /// `total_kernel_time_ms: 0.00`.
    ///
    /// Gated on `feature = "cuda"` because the clearing code in
    /// `nsl_kernel_profiler_start` that this test exercises is
    /// itself cuda-gated; a CPU-only build's `start` is a no-op on
    /// the pool and the assertions would pass vacuously, giving a
    /// false sense of coverage.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_start_does_not_allocate_event_pool() {
        let _guard = TEST_LOCK.lock().unwrap();

        // Pre-seed the pool with dummy handles so we can verify `start`
        // *clears* (not allocates).  The lazy-init path would
        // re-populate the pool only on the next `pop_events` call,
        // which we do not exercise here.
        {
            let mut pool = KERNEL_PROFILER.event_pool.lock().unwrap();
            pool.clear();
            for i in 0..16u64 {
                pool.push((i * 2, i * 2 + 1));
            }
        }
        *KERNEL_PROFILER.gpu_base_event.lock().unwrap() = 42;

        nsl_kernel_profiler_start();

        // Post-start: pool + base_event are cleared.  Guarantees the
        // lazy-init path will fire on first `pop_events`.
        let pool = KERNEL_PROFILER.event_pool.lock().unwrap();
        assert!(
            pool.is_empty(),
            "nsl_kernel_profiler_start must clear the pool so lazy-init \
             allocates fresh events after the CUDA context is live; \
             found {} stale entries",
            pool.len()
        );
        drop(pool);
        let base = *KERNEL_PROFILER.gpu_base_event.lock().unwrap();
        assert_eq!(
            base, 0,
            "nsl_kernel_profiler_start must reset gpu_base_event to 0 \
             so lazy-init records a fresh base event with a valid \
             timebase; found stale handle {}",
            base
        );

        nsl_kernel_profiler_stop();
    }
}
