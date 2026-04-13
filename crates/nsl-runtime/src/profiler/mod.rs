//! Kernel-timing profile collector used by `nsl run --monitor`.
//!
//! Phase 2: codegen emits `nsl_profile_kernel_begin/end` hooks around kernel
//! launches. Under the `cuda` feature, the FFI layer checks out `CUevent`
//! handles and records them on the launch stream, producing real GPU
//! execution timings. Without the `cuda` feature the FFI falls back to a
//! host-side `Instant` clock (`NanoClock`) so CPU-only builds still produce a
//! coherent JSON report.

pub mod collector;
#[cfg(feature = "cuda")]
pub mod cuda_clock;
pub mod ffi;
