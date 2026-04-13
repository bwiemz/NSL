//! C-callable FFI hooks invoked by codegen-emitted kernel wrappers (Phase 2).
//!
//! Under the `cuda` feature, `begin`/`end` check out `CUevent` handles from a
//! `CudaEventClock` pool and record them on the launch stream; the resulting
//! handles round-trip through the `Collector`'s `(start, end)` tuple and are
//! resolved to real GPU microseconds at flush time via
//! `cuEventSynchronize`+`cuEventElapsedTime`.
//!
//! Without the `cuda` feature, the hooks fall back to a monotonic host-side
//! `Instant`-derived nanosecond timestamp (`NanoClock`), which yields
//! host-submit latency rather than GPU execution time — useful for CPU-only
//! E2E tests and for building on machines without CUDA.

use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::time::Instant;

use super::collector::{Collector, NanoClock};

#[cfg(feature = "cuda")]
use super::cuda_clock::CudaEventClock;

static COLLECTOR: Lazy<Mutex<Option<Collector>>> = Lazy::new(|| Mutex::new(None));
static START: Lazy<Instant> = Lazy::new(Instant::now);

#[cfg(feature = "cuda")]
static CUDA_CLOCK: Lazy<CudaEventClock> = Lazy::new(CudaEventClock::new);

#[cfg(not(feature = "cuda"))]
fn now_ns() -> u64 {
    START.elapsed().as_nanos() as u64
}

fn ensure_collector<'a>(
    guard: &'a mut std::sync::MutexGuard<'_, Option<Collector>>,
) -> &'a mut Collector {
    // Touch START so the monotonic base is initialized eagerly under either feature config
    // (harmless no-op under cuda feature — Lazy is evaluated the first time it's used).
    let _ = &*START;
    guard.get_or_insert_with(|| {
        #[cfg(feature = "cuda")]
        {
            Collector::new_with_clock(Box::new(CudaEventClock::new()))
        }
        #[cfg(not(feature = "cuda"))]
        {
            Collector::new_with_clock(Box::new(NanoClock))
        }
    })
}

#[cfg(not(feature = "cuda"))]
#[no_mangle]
pub extern "C" fn nsl_profile_kernel_begin(kernel_id: u32) {
    let t = now_ns();
    let mut guard = COLLECTOR.lock().unwrap();
    ensure_collector(&mut guard).begin(kernel_id, t);
}

#[cfg(not(feature = "cuda"))]
#[no_mangle]
pub extern "C" fn nsl_profile_kernel_end(kernel_id: u32) {
    let t = now_ns();
    let mut guard = COLLECTOR.lock().unwrap();
    ensure_collector(&mut guard).end(kernel_id, t);
}

#[cfg(feature = "cuda")]
#[no_mangle]
pub extern "C" fn nsl_profile_kernel_begin(kernel_id: u32) {
    let event = CUDA_CLOCK.checkout_event();
    if event != 0 {
        // Record on the same stream that `nsl_kernel_launch` uses. Using
        // `crate::cuda::current_stream()` (NOT `0`) is load-bearing: if the
        // launch ever switches off the default stream, this stays consistent.
        unsafe {
            let res = crate::cuda::cu_event_record_on_current_stream(event);
            if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                eprintln!("[nsl-profiler] cuEventRecord(begin) failed: {:?}", res);
            }
        }
    }
    let mut guard = COLLECTOR.lock().unwrap();
    ensure_collector(&mut guard).begin(kernel_id, event);
}

#[cfg(feature = "cuda")]
#[no_mangle]
pub extern "C" fn nsl_profile_kernel_end(kernel_id: u32) {
    let event = CUDA_CLOCK.checkout_event();
    if event != 0 {
        unsafe {
            let res = crate::cuda::cu_event_record_on_current_stream(event);
            if res != cudarc::driver::sys::CUresult::CUDA_SUCCESS {
                eprintln!("[nsl-profiler] cuEventRecord(end) failed: {:?}", res);
            }
        }
    }
    let mut guard = COLLECTOR.lock().unwrap();
    ensure_collector(&mut guard).end(kernel_id, event);
}

/// Flush aggregates as pretty-printed JSON to the given path.
/// Returns 0 on success, non-zero on error (bad path / IO failure).
/// `path_ptr`/`path_len` use UTF-8 bytes; caller is responsible for liveness.
///
/// # Safety
/// Caller must ensure `(path_ptr, path_len)` is a valid UTF-8 slice they own
/// for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn nsl_profile_flush(path_ptr: *const u8, path_len: usize) -> i32 {
    if path_ptr.is_null() {
        return 1;
    }
    let bytes = std::slice::from_raw_parts(path_ptr, path_len);
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return 2,
    };
    let path = std::path::Path::new(s);
    let mut guard = COLLECTOR.lock().unwrap();
    let c = ensure_collector(&mut guard);
    match c.flush_to(path) {
        Ok(_) => 0,
        Err(_) => 3,
    }
}

// Silence unused-import warning under cuda feature (NanoClock only used in
// non-cuda branch of ensure_collector).
#[cfg(feature = "cuda")]
#[allow(dead_code)]
fn _nanoclock_typecheck() -> NanoClock { NanoClock }
