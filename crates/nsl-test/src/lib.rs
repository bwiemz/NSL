//! Test infrastructure crate.
//!
//! - `diagnostic_mode` — CSHA backward-kernel localizability (CPU-component swap for bisect)

pub mod b1_adapter;
pub mod cpu_naive_backward;
pub mod cpu_naive_forward;
pub mod cpu_naive_prologue;
pub mod diagnostic_mode;
