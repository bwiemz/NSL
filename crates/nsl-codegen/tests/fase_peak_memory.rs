//! Peak host memory of the source-AD accumulation hook against the tape
//! path (see `fase_peak_memory_impl.rs` for what that does and does not
//! show).
//!
//! Only compiles with --features test-hooks.  Run via:
//!     cargo test -p nsl-codegen --features test-hooks --test fase_peak_memory

#[cfg(feature = "test-hooks")]
#[path = "fase_peak_memory_impl.rs"]
mod test_impl;
