//! Item #5: FASE Deferred peak-memory regression test.
//!
//! Only compiles with --features test-hooks.  Run via:
//!     cargo test -p nsl-codegen --features test-hooks --test fase_peak_memory

#[cfg(feature = "test-hooks")]
#[path = "fase_peak_memory_impl.rs"]
mod test_impl;
