//! BitNet b1.58 ternary quantization kernel family.
//!
//! Spec: `docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md`
//!
//! ## Subsystem layout (Phase 1)
//!
//! - `config.rs`: `BitNetKernelConfig` — tile shapes, dtypes, fused-RMSNorm /
//!   fused-bias / fused-residual flags.
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
pub mod loader;
pub mod pack;
pub mod phases;

// M35.2a training-mode wrapper — gated on V-P1-D pass per BLOCKED_ON_V_P1_D.md.
// Real implementation lands in M35.2a implementation Stage D.5.
#[doc(hidden)]
pub mod orchestrator_train;

pub use config::BitNetKernelConfig;

/// Synthesize a complete BitNet GEMM kernel as PTX bytes.
///
/// Composes the four public phase emitters: `packed_load`,
/// `quantized_ternary_gemm` (which itself fuses activation-quant + ternary GEMM),
/// `finalize`. The bare `absmean_quant` and `ternary_gemm` phases are
/// subsystem-internal and not exposed here per IR-001.
///
/// The emitted PTX module is structurally complete (preamble + signature +
/// phase composition + ret) but is NOT runtime-loadable yet — the body uses
/// raw register names (`%r_row_id`, `%rd_y_out`, etc.) without `.reg`
/// declarations. Task 10's merge gate adds the runtime-loadable wiring.
pub fn synthesize_kernel(config: &BitNetKernelConfig) -> Vec<u8> {
    let mut ptx = String::new();

    // PTX module preamble: target sm_80+ (Ampere baseline; matches NSL
    // architecture floor for cp.async-based packed_load). Convention
    // matches backend_ptx.rs / epilogue_fusion.rs preamble shape.
    ptx.push_str(".version 7.0\n");
    ptx.push_str(".target sm_80\n");
    ptx.push_str(".address_size 64\n");
    ptx.push_str(&format!("// BitNet kernel: {}\n", config.kernel_name()));
    ptx.push_str(&format!(".visible .entry {} (\n", config.kernel_name()));
    ptx.push_str("  .param .u64 act_in,\n");
    ptx.push_str("  .param .u64 weights_packed,\n");
    ptx.push_str("  .param .u64 y_out");
    if config.fused_bias_add {
        ptx.push_str(",\n  .param .u64 bias");
    }
    if config.fused_residual_add {
        ptx.push_str(",\n  .param .u64 residual");
    }
    ptx.push_str("\n) {\n");

    ptx.push_str(
        "// --- Body skeleton (Phase 1; runtime register definitions added in Task 10) ---\n",
    );

    // Compose the phases.
    crate::bitnet::phases::packed_load::emit(&mut ptx, config);
    crate::bitnet::phases::quantized_ternary_gemm::emit(&mut ptx, config);
    crate::bitnet::phases::finalize::emit(&mut ptx, config);

    ptx.push_str("ret;\n");
    ptx.push_str("}\n");

    ptx.into_bytes()
}
