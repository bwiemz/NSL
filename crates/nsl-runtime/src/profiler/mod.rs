//! Kernel-timing profile collector used by `nsl run --monitor`.
//!
//! Phase 1: the Collector + FFI are complete, but the codegen side does not
//! emit nsl_profile_kernel_begin/end hooks yet (see
//! nsl_codegen::profiling::instrument TODO). So in practice the "actual" JSON
//! is empty in Phase 1 end-to-end runs. The monitor CLI handles that case
//! gracefully (shows predictions only).

pub mod collector;
pub mod ffi;
