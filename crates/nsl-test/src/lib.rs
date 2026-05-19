//! Test infrastructure for M57 v1 FPGA backend.
//!
//! - `fixture` — binary format reader + hash verification
//! - `cpu_reference` — bit-exact CPU implementations matching the v1 MLP arithmetic
//! - `stimuli` — deterministic ChaCha20Rng-based test stimuli
//! - `fpga_harness` — Verilator external-process invocation

pub mod cpu_reference;
pub mod fixture;
pub mod fpga_harness;
pub mod stimuli;
