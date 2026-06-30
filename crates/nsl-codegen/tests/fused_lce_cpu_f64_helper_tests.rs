//! Unit-test driver for the shared CPU f64 fused-linear-CE reference.
//!
//! The actual `#[cfg(test)]` module lives inside `common/fused_lce_cpu_f64.rs`
//! so that production-flavored helper functions can sit alongside the tests
//! that prove they're correct. This file's job is just to drag the module
//! into a discoverable integration-test target.

mod common;

// Re-export so callers writing `fused_lce_cpu_f64_helper_tests::cpu_lce_forward_f64`
// can find the helpers — not strictly required, but makes the surface obvious
// when grepping.
#[allow(unused_imports)]
pub use common::fused_lce_cpu_f64::{
    CpuLceBackward, CpuLceForward, IGNORE_INDEX, cpu_lce_backward_f64, cpu_lce_forward_f64,
};
