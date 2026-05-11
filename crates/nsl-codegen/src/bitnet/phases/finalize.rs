//! BitNet finalize epilogue — PUBLIC phase emitter.
//!
//! Spec: docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md §4.4.
//!
//! Operates on the FP32 dequantized output from `ternary_gemm` (in
//! `%f_y_out`), optionally adds bias and/or residual, casts to the
//! configured output dtype (F16 or BF16), and stores to HBM.
//!
//! Deferred to Phase 1.5: RMSNorm fold across layer boundaries.

use crate::bitnet::config::BitNetKernelConfig;
use crate::kernel_ir::KirType;

/// Emit finalize epilogue PTX.
///
/// Input registers (set by ternary_gemm.rs):
/// - `%f_y_out`: FP32 dequantized GEMM output for this (row, col).
/// - `%r_row_id`, `%r_col_id`: this thread's output coordinates.
///
/// Output:
/// - HBM at `%rd_y_global + (row_id * out_dim + col_id) * sizeof(dtype)`.
pub fn emit(ptx: &mut String, config: &BitNetKernelConfig) {
    ptx.push_str("// === BitNet finalize ===\n");

    if config.fused_bias_add {
        ptx.push_str("// Bias add: load bias[col_id] from global and add to accumulator.\n");
        ptx.push_str("cvt.u64.u32 %rd_bias_off, %r_col_id;\n");
        // Bias is FP32 in this draft. (BitNet b1.58 Llama-style layers don't
        // use bias on attention/FFN projections, so this path is unused in
        // standard models; included for the optional spec-§4.4 capability.)
        ptx.push_str("shl.b64 %rd_bias_off, %rd_bias_off, 2;\n"); // 4 bytes / FP32
        ptx.push_str("add.s64 %rd_bias_addr, %rd_bias, %rd_bias_off;\n");
        ptx.push_str("ld.global.f32 %f_bias, [%rd_bias_addr];\n");
        ptx.push_str("add.f32 %f_y_out, %f_y_out, %f_bias;\n");
    }

    if config.fused_residual_add {
        ptx.push_str("// Residual add: load residual[row, col] (FP32) and add.\n");
        ptx.push_str("mul.lo.u32 %r_res_off, %r_row_id, %r_out_dim;\n");
        ptx.push_str("add.u32 %r_res_off, %r_res_off, %r_col_id;\n");
        ptx.push_str("shl.b32 %r_res_off, %r_res_off, 2;\n"); // FP32
        ptx.push_str("cvt.u64.u32 %rd_res_off, %r_res_off;\n");
        ptx.push_str("add.s64 %rd_res_addr, %rd_residual, %rd_res_off;\n");
        ptx.push_str("ld.global.f32 %f_res, [%rd_res_addr];\n");
        ptx.push_str("add.f32 %f_y_out, %f_y_out, %f_res;\n");
    }

    // Cast to output dtype and store.
    let (cvt_op, store_ty, elem_bytes) = match config.output_dtype {
        KirType::F16 => ("cvt.rn.f16.f32", "b16", 2u32),
        KirType::Bf16 => ("cvt.rn.bf16.f32", "b16", 2u32),
        _ => {
            // Catch any non-F16/BF16 output_dtype at PTX-emission time.
            // kernel_name() also fail-fast panics on invalid dtype for the
            // dispatch key; this is the analogous guard for the body.
            panic!(
                "finalize: output_dtype must be F16 or Bf16, got {:?}",
                config.output_dtype
            );
        }
    };

    ptx.push_str(&format!(
        "// Cast FP32 -> {} and store to HBM.\n",
        if elem_bytes == 2 { "F16/BF16" } else { "FP32" }
    ));
    ptx.push_str(&format!("{cvt_op} %h_y, %f_y_out;\n"));
    ptx.push_str("mul.lo.u32 %r_yoff, %r_row_id, %r_out_dim;\n");
    ptx.push_str("add.u32 %r_yoff, %r_yoff, %r_col_id;\n");
    ptx.push_str(&format!(
        "shl.b32 %r_yoff, %r_yoff, {};\n",
        elem_bytes.trailing_zeros()
    ));
    ptx.push_str("cvt.u64.u32 %rd_yoff, %r_yoff;\n");
    ptx.push_str("cvta.to.global.u64 %rd_y_global, %rd_y_out;\n");
    ptx.push_str("add.s64 %rd_y_addr, %rd_y_global, %rd_yoff;\n");
    ptx.push_str(&format!("st.global.{store_ty} [%rd_y_addr], %h_y;\n"));

    ptx.push_str("// === end BitNet finalize ===\n");
}
