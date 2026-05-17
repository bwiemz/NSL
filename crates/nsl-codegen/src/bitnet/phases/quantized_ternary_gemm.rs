//! BitNet fused activation-quant + ternary GEMM — PUBLIC phase emitter.
//!
//! Spec: docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md §4.3.
//! Public path per IR-001: external callers compose THIS, not the bare
//! `absmean_quant` or `ternary_gemm` emitters. Fusion enforces the
//! "activations must be quantized first" precondition by API shape — the
//! prologue is structurally inseparable from the GEMM body.

use crate::bitnet::config::BitNetKernelConfig;
use crate::bitnet::phases::{absmean_quant, ternary_gemm};

/// Emit the fused absmax-quant prologue + ternary GEMM body in one PTX
/// sequence. The two sub-phases share the kernel context (registers,
/// SMEM allocations) and execute back-to-back.
pub fn emit(ptx: &mut String, config: &BitNetKernelConfig) {
    ptx.push_str("// === BitNet fused quantized_ternary_gemm ===\n");
    ptx.push_str("// Phase 1/2: absmax activation quantization prologue.\n");
    absmean_quant::emit(ptx, config);
    ptx.push_str("// Phase 2/2: ternary GEMM body (consumes quantized activations + scale).\n");
    ternary_gemm::emit(ptx, config);
    ptx.push_str("// === end BitNet fused quantized_ternary_gemm ===\n");
}
