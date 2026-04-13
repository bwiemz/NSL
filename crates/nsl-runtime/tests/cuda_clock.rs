//! CUDA-feature-only smoke test for CudaEventClock pool/handle behavior.
//! End-to-end timing verification is in Task 9 (CLI E2E).
#![cfg(feature = "cuda")]

use nsl_runtime::profiler::collector::Collector;
use nsl_runtime::profiler::cuda_clock::CudaEventClock;

#[test]
fn checkout_returns_distinct_nonzero_handles() {
    // Requires a working CUDA driver to create real events. Skipped implicitly
    // on dev boxes without one (the process aborts inside cuEventCreate).
    let clock = CudaEventClock::new();
    let s = clock.checkout_event();
    let e = clock.checkout_event();
    assert_ne!(s, 0);
    assert_ne!(e, 0);
    assert_ne!(s, e);
}

#[test]
fn collector_constructs_with_cuda_event_clock() {
    let mut c = Collector::new_with_clock(Box::new(CudaEventClock::new()));
    assert_eq!(c.snapshot().len(), 0);
}
