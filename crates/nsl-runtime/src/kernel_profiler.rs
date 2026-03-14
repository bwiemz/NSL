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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_profiler_enable_disable() {
        // Reset state
        KERNEL_PROFILER.enabled.store(false, Ordering::Relaxed);

        assert!(!kernel_profiler_enabled());
        nsl_kernel_profiler_start();
        assert!(kernel_profiler_enabled());
        nsl_kernel_profiler_stop();
        assert!(!kernel_profiler_enabled());
    }
}
