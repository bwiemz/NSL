//! The dedicated CUstream for inspect copies.
//!
//! Sync model: codegen-emitted hook calls cuEventRecord on the compute stream
//! after the producing kernel, then cuStreamWaitEvent on this inspect stream
//! BEFORE issuing the memcpy. That ordering is not enforced here; this module
//! only owns the accessor.
//!
//! Roadmap A4 step 2: the handle itself moved out of an `INSPECT_STREAM`
//! thread-local and onto the current device context's `StreamPool`, which
//! keeps one per (thread, device). The stream is still created lazily with
//! the same flags on first access, and is still the calling thread's alone —
//! it is now also the calling *device*'s alone.

#![cfg(feature = "cuda")]

use cudarc::driver::sys;

/// Returns this (thread, device)'s inspect stream, creating it on first
/// access.
pub fn current_inspect_stream() -> sys::CUstream {
    let ctx = crate::cuda::context::current();
    ctx.streams.inspect(|| ctx.activate())
}
