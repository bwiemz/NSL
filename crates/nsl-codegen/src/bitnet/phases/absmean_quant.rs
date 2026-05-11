//! BitNet 8-bit ABSMAX activation quantization prologue.
//!
//! Spec: docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md §4.2.
//! (Per PI.2 spec correction: BitNet b1.58 uses per-row absmax for activations,
//! not absmean as spec §4.2 originally stated. File name retained for spec
//! traceability; math is absmax. Reference: tests/bitnet_reference_impl.rs.)
//!
//! **Internal phase emitter** — do NOT call from outside `bitnet/`. Use
//! `quantized_ternary_gemm::emit` (the fused public path) instead. This
//! visibility is `pub(crate)` rather than `pub(super)` only to enable
//! structural snapshot tests; future Rust visibility refinements (e.g., a
//! `__test_only` feature gate) can tighten this if external misuse appears.
//!
//! Performs per-row reduction over `hidden_dim` elements via warp-shuffle
//! butterfly reduction (parallel), then per-element quantize:
//!   scale = max_lane(|x[r, k]|)
//!   q[r, k] = round(clip(x[r, k] * 127.0 / scale, -127, 127))
//! Zero-magnitude row writes all zeros to the quantized output.

use crate::bitnet::config::BitNetKernelConfig;

/// Emit absmax-quant prologue PTX.
///
/// Inputs (per PTX register conventions):
/// - `%rd_act_in`: global pointer to activation tile in HBM.
/// - `%r_row_id`: row index within the tile (per-CTA row).
/// - `%rd_qact_smem`: SMEM destination for int8 quantized activations.
///
/// Output:
/// - `%f_scale`: FP32 absmax scale for this row.
/// - SMEM at `%rd_qact_smem`: int8 quantized activations.
pub fn emit(ptx: &mut String, config: &BitNetKernelConfig) {
    let hidden_dim = config.hidden_dim;

    ptx.push_str(&format!(
        "// === BitNet absmax_quant (hidden_dim={hidden_dim}) ===\n"
    ));

    // Step 1: per-thread |x| max accumulation over hidden_dim via lane stride.
    ptx.push_str("// Per-thread: track max(|x[r, k]|) over assigned lane stride.\n");
    ptx.push_str("mov.f32 %f_max, 0f00000000;\n"); // 0.0
    ptx.push_str("mov.u32 %r_k, %tid.x;\n");
    ptx.push_str("ABSMAX_LOOP:\n");
    ptx.push_str("setp.ge.u32 %p_amdone, %r_k, %r_hidden_dim;\n");
    ptx.push_str("@%p_amdone bra ABSMAX_END;\n");
    ptx.push_str("mul.lo.u32 %r_off, %r_row_id, %r_hidden_dim;\n");
    ptx.push_str("add.u32 %r_off, %r_off, %r_k;\n");
    ptx.push_str("shl.b32 %r_off, %r_off, 1;\n"); // F16 element = 2 bytes
    ptx.push_str("cvt.u64.u32 %rd_off, %r_off;\n");
    ptx.push_str("add.s64 %rd_addr, %rd_act_in, %rd_off;\n");
    ptx.push_str("ld.global.b16 %h_x, [%rd_addr];\n");
    ptx.push_str("cvt.f32.f16 %f_x, %h_x;\n");
    ptx.push_str("abs.f32 %f_absx, %f_x;\n");
    ptx.push_str("max.f32 %f_max, %f_max, %f_absx;\n");
    ptx.push_str("add.u32 %r_k, %r_k, 32;\n"); // warp size stride
    ptx.push_str("bra ABSMAX_LOOP;\n");
    ptx.push_str("ABSMAX_END:\n");

    // Step 2: warp-shuffle butterfly reduction of the max across lanes.
    ptx.push_str("// Warp-shuffle butterfly reduction (parallel, not serialized).\n");
    ptx.push_str("shfl.sync.bfly.b32 %f_partial, %f_max, 16, 0x1f, 0xffffffff;\n");
    ptx.push_str("max.f32 %f_max, %f_max, %f_partial;\n");
    ptx.push_str("shfl.sync.bfly.b32 %f_partial, %f_max, 8, 0x1f, 0xffffffff;\n");
    ptx.push_str("max.f32 %f_max, %f_max, %f_partial;\n");
    ptx.push_str("shfl.sync.bfly.b32 %f_partial, %f_max, 4, 0x1f, 0xffffffff;\n");
    ptx.push_str("max.f32 %f_max, %f_max, %f_partial;\n");
    ptx.push_str("shfl.sync.bfly.b32 %f_partial, %f_max, 2, 0x1f, 0xffffffff;\n");
    ptx.push_str("max.f32 %f_max, %f_max, %f_partial;\n");
    ptx.push_str("shfl.sync.bfly.b32 %f_partial, %f_max, 1, 0x1f, 0xffffffff;\n");
    ptx.push_str("max.f32 %f_max, %f_max, %f_partial;\n");
    ptx.push_str("// %f_max now holds row-wide absmax across all lanes.\n");

    // %f_scale = %f_max. (Renamed for clarity in downstream phases.)
    ptx.push_str("mov.f32 %f_scale, %f_max;\n");

    // Step 3: zero-row guard.
    ptx.push_str("// Zero-magnitude row guard: scale == 0 -> write all zeros to SMEM, scale = 0.\n");
    ptx.push_str("setp.eq.f32 %p_zero, %f_scale, 0f00000000;\n");
    ptx.push_str("@%p_zero bra ABSMAX_ZERO_ROW;\n");

    // Step 4: per-element quantize. q = round(clip(x * 127.0 / scale, -127, 127)).
    ptx.push_str("// Per-thread quantize: q = round(clip(x * 127.0 / scale, -127, 127)).\n");
    ptx.push_str("mov.f32 %f_inv_scale, 0f42FE0000;\n"); // 127.0
    ptx.push_str("div.rn.f32 %f_inv_scale, %f_inv_scale, %f_scale;\n");
    ptx.push_str("mov.u32 %r_k, %tid.x;\n");
    ptx.push_str("QUANT_LOOP:\n");
    ptx.push_str("setp.ge.u32 %p_qdone, %r_k, %r_hidden_dim;\n");
    ptx.push_str("@%p_qdone bra QUANT_END;\n");
    ptx.push_str("mul.lo.u32 %r_qoff, %r_row_id, %r_hidden_dim;\n");
    ptx.push_str("add.u32 %r_qoff, %r_qoff, %r_k;\n");
    ptx.push_str("shl.b32 %r_qoff_bytes, %r_qoff, 1;\n");
    ptx.push_str("cvt.u64.u32 %rd_qoff, %r_qoff_bytes;\n");
    ptx.push_str("add.s64 %rd_qload, %rd_act_in, %rd_qoff;\n");
    ptx.push_str("ld.global.b16 %h_xq, [%rd_qload];\n");
    ptx.push_str("cvt.f32.f16 %f_xq, %h_xq;\n");
    ptx.push_str("mul.f32 %f_scaled, %f_xq, %f_inv_scale;\n");
    ptx.push_str("max.f32 %f_clip, %f_scaled, 0fC2FE0000;\n"); // -127.0
    ptx.push_str("min.f32 %f_clip, %f_clip, 0f42FE0000;\n"); // +127.0
    ptx.push_str("cvt.rni.s32.f32 %r_qi, %f_clip;\n");
    ptx.push_str("cvt.s8.s32 %rs_qi8, %r_qi;\n");
    ptx.push_str("cvt.u64.u32 %rd_smemoff, %r_qoff;\n"); // 1 byte per int8
    ptx.push_str("add.s64 %rd_qsmem_addr, %rd_qact_smem, %rd_smemoff;\n");
    ptx.push_str("st.shared.s8 [%rd_qsmem_addr], %rs_qi8;\n");
    ptx.push_str("add.u32 %r_k, %r_k, 32;\n");
    ptx.push_str("bra QUANT_LOOP;\n");
    ptx.push_str("QUANT_END:\n");
    ptx.push_str("bra ABSMAX_DONE;\n");

    ptx.push_str("ABSMAX_ZERO_ROW:\n");
    ptx.push_str("// Zero-magnitude row: write zeros to all quantized slots.\n");
    ptx.push_str("mov.u32 %r_k, %tid.x;\n");
    ptx.push_str("ZERO_LOOP:\n");
    ptx.push_str("setp.ge.u32 %p_zdone, %r_k, %r_hidden_dim;\n");
    ptx.push_str("@%p_zdone bra ABSMAX_DONE;\n");
    ptx.push_str("mul.lo.u32 %r_zoff, %r_row_id, %r_hidden_dim;\n");
    ptx.push_str("add.u32 %r_zoff, %r_zoff, %r_k;\n");
    ptx.push_str("cvt.u64.u32 %rd_zoff, %r_zoff;\n");
    ptx.push_str("add.s64 %rd_zsmem, %rd_qact_smem, %rd_zoff;\n");
    ptx.push_str("st.shared.s8 [%rd_zsmem], 0;\n");
    ptx.push_str("add.u32 %r_k, %r_k, 32;\n");
    ptx.push_str("bra ZERO_LOOP;\n");

    ptx.push_str("ABSMAX_DONE:\n");
    ptx.push_str("bar.sync 0;\n");
    ptx.push_str("// === end BitNet absmax_quant ===\n");
}
