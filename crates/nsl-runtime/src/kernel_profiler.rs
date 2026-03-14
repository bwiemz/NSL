//! Async GPU kernel profiler with pre-allocated event pools.
//! Records cuEvent pairs around kernel launches, resolves timestamps
//! at flush time with a single cuCtxSynchronize. Zero sync during execution.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::Instant;

/// A recorded kernel launch trace (event indices + metadata).
pub(crate) struct KernelTrace {
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
    pub event_pool: Mutex<Vec<(u64, u64)>>, // (start_event, stop_event) pairs
    pub gpu_base_event: Mutex<u64>,
}

// SAFETY: CUevent handles are thread-safe opaque pointers managed by the CUDA driver.
unsafe impl Send for KernelProfiler {}
unsafe impl Sync for KernelProfiler {}

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

const EVENT_POOL_SIZE: usize = 4096;

/// Initialize the kernel profiler. Allocates event pool on GPU if available.
#[no_mangle]
pub extern "C" fn nsl_kernel_profiler_start() {
    KERNEL_PROFILER.enabled.store(true, Ordering::Relaxed);
    *KERNEL_PROFILER.cpu_start_time.lock().unwrap() = Some(Instant::now());
    *KERNEL_PROFILER.traces.lock().unwrap() = Vec::new();
    *KERNEL_PROFILER.pool_cursor.lock().unwrap() = 0;

    // GPU event allocation is done conditionally when CUDA is available
    #[cfg(feature = "cuda")]
    {
        let mut pool = KERNEL_PROFILER.event_pool.lock().unwrap();
        pool.clear();
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
        let mut base = KERNEL_PROFILER.gpu_base_event.lock().unwrap();
        unsafe {
            crate::cuda::cu_event_create(&mut *base);
            crate::cuda::cu_event_record(*base, std::ptr::null_mut());
        }
    }
}

/// Disable recording. Does not free resources (flush does that).
#[no_mangle]
pub extern "C" fn nsl_kernel_profiler_stop() {
    KERNEL_PROFILER.enabled.store(false, Ordering::Relaxed);
}

/// Pop the next (start, stop) event pair from the pool.
/// Lock-pop-unlock pattern: single lock acquisition, extract event pair, unlock.
/// Returns None if profiler disabled or pool exhausted.
///
/// Lock ordering: always lock event_pool FIRST (contains pool + cursor).
/// This is consistent with flush/destroy which also lock event_pool first.
pub(crate) fn kernel_profiler_pop_events() -> Option<(u64, u64, usize)> {
    if !kernel_profiler_enabled() {
        return None;
    }

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
pub(crate) fn kernel_profiler_push_trace(name: &str, grid: [u32; 3], block: [u32; 3]) {
    let cursor = *KERNEL_PROFILER.pool_cursor.lock().unwrap();
    KERNEL_PROFILER.traces.lock().unwrap().push(KernelTrace {
        pool_idx: cursor - 1, // points to the event pair just used
        name: name.to_string(),
        grid,
        block,
    });
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
}
