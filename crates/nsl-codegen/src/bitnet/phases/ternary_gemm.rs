//! BitNet ternary GEMM body — INTERNAL phase emitter.
//!
//! Spec: docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md §4.3.
//!
//! **Internal phase emitter** — do NOT call from outside `bitnet/`. Use
//! `quantized_ternary_gemm::emit` (the fused public path) which establishes
//! the "activations must be absmax-quantized first" precondition that this
//! emitter requires. Per IR-001, this is intended to be subsystem-internal;
//! actual visibility is `pub` only to enable structural snapshot tests from
//! integration tests in `tests/` (same rationale as `absmean_quant.rs`).
//!
//! Math (verified against Task 3 reference + 1bitLLM/bitnet_b1_58-3B):
//!   acc[r, c]  = sum_k q_act[r, k] * w_ternary[k, c]    (i32 accumulator)
//!   y[r, c]    = (scale[r] / 127.0) * acc[r, c]         (FP32, then cast to F16)
//!
//! Inputs in registers / SMEM:
//! - `%rd_qact_smem`: int8 quantized activations (1 byte per element).
//! - `%rd_w_smem`: int8 ternary weights (one trit per byte, already unpacked).
//! - `%f_scale`: FP32 absmax scale for the row.
//! - `%r_row_id`, `%r_col_id`: this thread's output coordinates.
//! - `%r_hidden_dim`: hidden dim (compile-time constant baked in).
//! - `%r_out_dim`: out dim (used for the weight row stride).
//!
//! Output registers:
//! - `%f_y_out`: FP32 dequantized output for this (row, col).
//!
//! ## PTX discipline
//!
//! Per project memory (NSL/MEMORY.md "GPU invariants"): PTX ISA 7.0 disallows
//! `mad.lo.u32`; the documented workaround is `mul.lo.u32 + add.u32`. We apply
//! the same pattern to the signed-i32 MAC at the heart of the GEMM
//! (`mul.lo.s32 + add.s32`), even though `mad.lo.s32` is theoretically
//! available on newer ISAs. There is no `mad.lo.s32` precedent in this crate,
//! so we mirror the established mul-then-add idiom for safety under cudarc
//! JIT.

use crate::bitnet::config::BitNetKernelConfig;

/// Emit ternary GEMM body PTX.
///
/// Precondition (enforced by the public fused emitter, NOT by this function):
/// activations at `%rd_qact_smem` MUST already be absmax-quantized by an
/// earlier `absmean_quant::emit` call in the same PTX stream, and `%f_scale`
/// MUST hold that prologue's per-row absmax scale.
pub fn emit(ptx: &mut String, config: &BitNetKernelConfig) {
    let hidden_dim = config.hidden_dim;

    ptx.push_str(&format!(
        "// === BitNet ternary_gemm (hidden_dim={hidden_dim}) ===\n"
    ));

    // Initialize i32 accumulator to zero.
    ptx.push_str("// Init i32 accumulator for this thread's (row, col) output.\n");
    ptx.push_str("mov.s32 %r_acc, 0;\n");

    // Inner reduction loop over hidden_dim.
    ptx.push_str("// Inner loop: acc += q_act[row, k] * w_ternary[k, col] over k.\n");
    ptx.push_str("mov.u32 %r_k, 0;\n");
    ptx.push_str("GEMM_LOOP:\n");
    ptx.push_str("setp.ge.u32 %p_gdone, %r_k, %r_hidden_dim;\n");
    ptx.push_str("@%p_gdone bra GEMM_END;\n");

    // Load q_act[row_id, k] from SMEM (i8 element).
    ptx.push_str("// Load q_act[row, k] = qact_smem[row * hidden_dim + k].\n");
    ptx.push_str("mul.lo.u32 %r_qoff, %r_row_id, %r_hidden_dim;\n");
    ptx.push_str("add.u32 %r_qoff, %r_qoff, %r_k;\n");
    ptx.push_str("cvt.u64.u32 %rd_qoff, %r_qoff;\n");
    ptx.push_str("add.s64 %rd_qaddr, %rd_qact_smem, %rd_qoff;\n");
    ptx.push_str("ld.shared.s8 %rs_qact, [%rd_qaddr];\n");
    ptx.push_str("cvt.s32.s8 %r_qact_i32, %rs_qact;\n");

    // Load w_ternary[k, col_id] from SMEM (i8 element, value in {-1, 0, +1}).
    ptx.push_str("// Load w_ternary[k, col] = w_smem[k * out_dim + col].\n");
    ptx.push_str("mul.lo.u32 %r_woff, %r_k, %r_out_dim;\n");
    ptx.push_str("add.u32 %r_woff, %r_woff, %r_col_id;\n");
    ptx.push_str("cvt.u64.u32 %rd_woff, %r_woff;\n");
    ptx.push_str("add.s64 %rd_waddr, %rd_w_smem, %rd_woff;\n");
    ptx.push_str("ld.shared.s8 %rs_w, [%rd_waddr];\n");
    ptx.push_str("cvt.s32.s8 %r_w_i32, %rs_w;\n");

    // Multiply-accumulate.
    // Use mul.lo.s32 + add.s32 (NOT mad.lo.s32) to mirror the documented
    // PTX ISA 7.0 idiom that NSL applies to mad.lo.u32 — see project memory.
    ptx.push_str("// acc += qact * w  (mul.lo.s32 + add.s32 to avoid mad.lo.s32 risk).\n");
    ptx.push_str("mul.lo.s32 %r_prod, %r_qact_i32, %r_w_i32;\n");
    ptx.push_str("add.s32 %r_acc, %r_acc, %r_prod;\n");

    ptx.push_str("add.u32 %r_k, %r_k, 1;\n");
    ptx.push_str("bra GEMM_LOOP;\n");
    ptx.push_str("GEMM_END:\n");

    // Dequant: y = (scale / 127.0) * acc.
    ptx.push_str("// Dequant: y = (scale / 127.0) * acc.\n");
    ptx.push_str("cvt.rn.f32.s32 %f_acc, %r_acc;\n");
    ptx.push_str("mov.f32 %f_127, 0f42FE0000;\n"); // 127.0
    ptx.push_str("div.rn.f32 %f_inv127_scale, %f_scale, %f_127;\n");
    ptx.push_str("mul.f32 %f_y_out, %f_inv127_scale, %f_acc;\n");

    ptx.push_str("// === end BitNet ternary_gemm ===\n");
}
