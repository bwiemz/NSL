//! Test infrastructure crate.
//!
//! - `diagnostic_mode` — CSHA backward-kernel localizability (CPU-component swap for bisect)
//! - `cpu_reference` — bit-exact CPU implementations matching the M57 v1 MLP arithmetic
//! - `fixture` — M57 binary fixture format reader + hash verification
//! - `fpga_harness` — M57 Verilator external-process invocation
//! - `stimuli` — M57 deterministic ChaCha20Rng-based test stimuli

pub mod cpu_naive_backward;
pub mod cpu_naive_forward;
pub mod diagnostic_mode;

pub mod cpu_reference;
pub mod fixture;
pub mod fpga_harness;
pub mod stimuli;
