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
//!
//! ```text
//! acc[r, c] = sum_k q_act[r, k] * w_ternary[c, k]     (i32 accumulator)
//! y[r, c]   = (scale[r] / 127.0) * acc[r, c]          (FP32; finalize casts)
//! ```
//!
//! Note the weight subscript order: `w_ternary[c, k]`, i.e. `[out_dim,
//! hidden_dim]` — see `packed_load.rs` for why that layout is used. The CPU
//! reference `forward_reference` indexes the transpose (`weights[k][c]`); it is
//! a math oracle, not a layout specification.
//!
//! Reads:
//! - SMEM `bitnet_qact[0..hidden_dim]` — int8 quantized activations.
//! - `%rd_w_packed` — global packed ternary weights (unpacked inline via
//!   `packed_load::emit`, four trits per `ld.global.u8`).
//! - `%f_scale` — FP32 absmax scale for the row.
//! - `%r_row_id`, `%r_col_id` — this thread's output coordinates.
//! - `%r_hidden_dim`, `%r_out_dim`.
//!
//! Writes:
//! - `%f_y_out` — FP32 dequantized output for this (row, col).
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
use crate::bitnet::phases::packed_load;

/// Emit the ternary GEMM body.
///
/// Emits the out-of-range column guard first. That guard MUST come after the
/// last `bar.sync` in `absmean_quant::emit`: a thread that branches to
/// `BITNET_EXIT` before a barrier its CTA peers still wait on leaves the
/// barrier permanently unsatisfied.
pub fn emit(ptx: &mut String, config: &BitNetKernelConfig) {
    let hidden_dim = config.hidden_dim;

    ptx.push_str(&format!(
        "// === BitNet ternary_gemm (hidden_dim={hidden_dim}) ===\n"
    ));

    // Column guard. Safe here and only here: every bar.sync in the kernel is
    // behind us, so exiting early cannot deadlock the CTA.
    ptx.push_str("// Columns past out_dim have no work. Guarded AFTER the last bar.sync so\n");
    ptx.push_str("// early-exiting threads cannot strand their CTA peers at a barrier.\n");
    ptx.push_str("setp.ge.u32 %p_cguard, %r_col_id, %r_out_dim;\n");
    ptx.push_str("@%p_cguard bra BITNET_EXIT;\n");

    // Init i32 accumulator for this thread's (row, col) output.
    ptx.push_str("// Init i32 accumulator for this thread's (row, col) output.\n");
    ptx.push_str("mov.s32 %r_acc, 0;\n");
    // Base flat trit index of this output column's weight row.
    ptx.push_str("// Base flat trit index of this column's weight row: col_id * hidden_dim.\n");
    ptx.push_str("mul.lo.u32 %r_wbase, %r_col_id, %r_hidden_dim;\n");
    ptx.push_str("mov.u32 %r_k, 0;\n");

    // Inner loop: four trits (one packed byte) per iteration.
    ptx.push_str("// Inner loop: acc += sum over 4 consecutive k per packed weight byte.\n");
    ptx.push_str("GEMM_LOOP:\n");
    ptx.push_str("setp.ge.u32 %p_gdone, %r_k, %r_hidden_dim;\n");
    ptx.push_str("@%p_gdone bra GEMM_END;\n");
    ptx.push_str("add.u32 %r_trit_index, %r_wbase, %r_k;\n");

    // Unpack the four trits for k+0..k+3 from a single packed byte.
    packed_load::emit(ptx, config);

    // Multiply-accumulate each of the four (activation, trit) pairs.
    ptx.push_str("add.u32 %r_addr32, %r_qact_base, %r_k;\n");
    for (lane, trit_reg) in ["%r_t0_val", "%r_t1_val", "%r_t2_val", "%r_t3_val"]
        .iter()
        .enumerate()
    {
        if lane == 0 {
            ptx.push_str("ld.shared.s8 %rs_qact, [%r_addr32];\n");
        } else {
            ptx.push_str(&format!("ld.shared.s8 %rs_qact, [%r_addr32+{lane}];\n"));
        }
        ptx.push_str("cvt.s32.s16 %r_qact_i32, %rs_qact;\n");
        ptx.push_str(&format!(
            "mul.lo.s32 %r_prod, %r_qact_i32, {trit_reg};\n"
        ));
        ptx.push_str("add.s32 %r_acc, %r_acc, %r_prod;\n");
    }

    ptx.push_str("add.u32 %r_k, %r_k, 4;\n");
    ptx.push_str("bra GEMM_LOOP;\n");
    ptx.push_str("GEMM_END:\n");

    // Dequant: y = (scale / 127.0) * acc.
    ptx.push_str("// Dequant: y = (scale / 127.0) * acc.\n");
    ptx.push_str("cvt.rn.f32.s32 %f_acc, %r_acc;\n");
    ptx.push_str("mov.f32 %f_127, 0f42FE0000;\n");
    ptx.push_str("div.rn.f32 %f_inv127_scale, %f_scale, %f_127;\n");
    ptx.push_str("mul.f32 %f_y_out, %f_inv127_scale, %f_acc;\n");

    ptx.push_str("// === end BitNet ternary_gemm ===\n");
}
