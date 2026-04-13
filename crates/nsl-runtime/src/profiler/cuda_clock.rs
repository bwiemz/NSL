//! Real CUDA-event-backed `ClockSource` for the kernel-timing collector.
//!
//! The clock holds a pool of `CUevent` handles. `nsl_profile_kernel_begin/end`
//! check out an event each, record it on the *same* stream `nsl_kernel_launch`
//! uses (see `crate::cuda::current_stream`), and pass the handle through
//! `Collector::{begin,end}` as a `u64`. On drain, `cuEventSynchronize(end)`
//! blocks until the kernel has actually executed on the GPU, then
//! `cuEventElapsedTime` returns GPU execution time in milliseconds, which we
//! convert to microseconds.
//!
//! Design rule (spec §4.4): both `cuEventRecord` calls MUST target the same
//! stream as the kernel launch, otherwise the measurement collapses to the
//! host-side submit latency rather than the GPU execution time. The runtime's
//! current `kernel_launch` uses the NULL/default stream, which
//! `crate::cuda::current_stream()` returns — events recorded on that stream
//! serialize correctly with the kernel launch.

#![cfg(feature = "cuda")]

use cudarc::driver::sys::CUresult;
use std::sync::Mutex;

use crate::profiler::collector::ClockSource;

pub struct CudaEventClock {
    pool: Mutex<Vec<u64>>,
}

// SAFETY: CUevent handles are opaque thread-safe pointers managed by the CUDA
// driver. We only access them through the Mutex, and CUDA event APIs are
// thread-safe with respect to the driver's internal synchronization.
unsafe impl Send for CudaEventClock {}
unsafe impl Sync for CudaEventClock {}

impl CudaEventClock {
    pub fn new() -> Self {
        Self { pool: Mutex::new(Vec::with_capacity(128)) }
    }

    /// Get a `CUevent` from the pool (or create a fresh one), returned as a
    /// `u64` so it round-trips through the `Collector`'s `(start, end)` tuple.
    /// Returns `0` on allocation failure (caller treats `0` as a no-op handle).
    pub fn checkout_event(&self) -> u64 {
        if let Some(e) = self.pool.lock().unwrap().pop() {
            return e;
        }
        unsafe {
            match crate::cuda::cu_event_create_checked() {
                Ok(e) => e,
                Err(res) => {
                    eprintln!("[nsl-profiler] cuEventCreate failed: {:?}", res);
                    0
                }
            }
        }
    }

    fn return_event(&self, h: u64) {
        if h != 0 {
            self.pool.lock().unwrap().push(h);
        }
    }
}

impl Default for CudaEventClock {
    fn default() -> Self { Self::new() }
}

impl ClockSource for CudaEventClock {
    fn elapsed_us(&self, start: u64, end: u64) -> f64 {
        if start == 0 || end == 0 {
            // Allocation failed for one side — nothing to measure.
            self.return_event(start);
            self.return_event(end);
            return 0.0;
        }
        unsafe {
            let sync_res = crate::cuda::cu_event_synchronize_raw(end);
            if sync_res != CUresult::CUDA_SUCCESS {
                eprintln!(
                    "[nsl-profiler] cuEventSynchronize failed ({:?}); dropping pair",
                    sync_res
                );
                self.return_event(start);
                self.return_event(end);
                return 0.0;
            }
            let mut ms: f32 = 0.0;
            let el_res = crate::cuda::cu_event_elapsed_time_raw(&mut ms, start, end);
            self.return_event(start);
            self.return_event(end);
            if el_res != CUresult::CUDA_SUCCESS {
                eprintln!("[nsl-profiler] cuEventElapsedTime failed ({:?})", el_res);
                return 0.0;
            }
            // ms → μs
            ms as f64 * 1000.0
        }
    }
}
