//! BitNet b1.58 ternary quantization kernel family.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md`
//!
//! ## Subsystem layout (Phase 1)
//!
//! - `config.rs`: `BitNetKernelConfig` — tile shapes, dtypes, fused-RMSNorm flag.
//! - `mod.rs`: orchestrator (`synthesize_kernel`); composes phase emitters into a
//!   standalone BitNet GEMM kernel.
//! - `pack.rs`: packed/unpacked conversion ops (host-side, pure Rust). Added in Task 2.
//! - `reference.rs`: CPU reference impl (`#[cfg(test)]`-gated). Added in Task 3.
//! - `phases/`:
//!   - `packed_load.rs` — PUBLIC; HBM→SMEM load + on-the-fly unpack.
//!   - `absmean_quant.rs` — `pub(super)`; activation quant prologue (BitNet b1.58
//!     uses absmax per-row for activations; file name is historical).
//!   - `ternary_gemm.rs` — `pub(super)`; GEMM body (input invariant: activations
//!     must be quantized; enforced by visibility per IR-001).
//!   - `quantized_ternary_gemm.rs` — PUBLIC; fused activation-quant + ternary GEMM.
//!   - `finalize.rs` — PUBLIC; dequant + epilogue.
//!
//! See `phases/README.md` for the Phase 2 (M35.2) planned additions.

pub mod config;
pub mod pack;
pub mod phases;

pub use config::BitNetKernelConfig;

/// Synthesize a complete BitNet GEMM kernel as PTX bytes.
///
/// Composes the four public phase emitters: `packed_load`,
/// `quantized_ternary_gemm` (which itself fuses activation-quant + ternary GEMM),
/// `finalize`. The bare `absmean_quant` and `ternary_gemm` phases are
/// subsystem-internal and not exposed here per IR-001.
///
/// Phase 1: returns a stub PTX module. Real emitter wiring lands in Task 7.
pub fn synthesize_kernel(config: &BitNetKernelConfig) -> Vec<u8> {
    todo!(
        "BitNet kernel synthesis lands in M35.1 Task 7 (orchestrator wiring). \
         Requested kernel: {}",
        config.kernel_name()
    )
}
