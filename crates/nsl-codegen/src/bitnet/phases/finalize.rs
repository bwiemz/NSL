//! BitNet finalize epilogue — PUBLIC phase emitter.
//!
//! Spec: docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md §4.4.
//!
//! Operates on the FP32 dequantized output from `ternary_gemm` (in
//! `%f_y_out`), multiplies by the per-tensor `weight_scale` to complete the
//! BitLinear b1.58 forward, optionally adds bias and/or residual, casts to
//! the configured output dtype (F16 or BF16), and stores to HBM.
//!
//! Full BitLinear forward math:
//!
//! ```text
//! %f_y_out (in)  = (act_scale / 127.0) * acc        // produced by ternary_gemm
//! %f_y_out      *= weight_scale                      // this phase, first step
//! %f_y_out      += bias[col]                         // if fused_bias_add
//! %f_y_out      += residual[row, col]                // if fused_residual_add
//! out[row, col]  = cast_to_output_dtype(%f_y_out)
//! ```
//!
//! `weight_scale` is read from the kernel `.param .f32 weight_scale` slot
//! (see `mod.rs::synthesize_kernel`); the host passes the per-layer absmean
//! scale produced by `loader.rs::LoadedTernaryWeight::scale` at
//! quantize-at-load time. The multiplication MUST precede bias and residual
//! because those operate in output space and are not scaled by weight_scale.
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

    // BitLinear weight_scale: per-tensor absmean scale (one f32 per layer).
    // Multiply the FP32 dequant accumulator to complete
    //   y = weight_scale * (act_scale / 127) * acc
    // BEFORE bias/residual (which operate in output space and are not scaled).
    ptx.push_str("// weight_scale: %f_y_out *= ld.param.f32 weight_scale.\n");
    ptx.push_str("ld.param.f32 %f_w_scale, [weight_scale];\n");
    // `mul.rn.f32` (not bare `mul.f32`): with an optional bias/residual `add.f32`
    // following, ptxas contracts the pair into a single FFMA, which rounds once
    // instead of twice and so does not compute the arithmetic documented above.
    // The explicit rounding modifier blocks the contraction. Verified at SASS
    // level: bare mul.f32 emits FFMA, mul.rn.f32 emits FMUL + FADD.
    ptx.push_str("mul.rn.f32 %f_y_out, %f_y_out, %f_w_scale;\n");

    if config.fused_bias_add {
        ptx.push_str("// Bias add: load bias[col_id] from global and add to accumulator.\n");
        ptx.push_str("cvt.u64.u32 %rd_bias_off, %r_col_id;\n");
        // %rd_bias is already a .global address (cvta'd in mod.rs::emit_prologue).
        // Bias is FP32 in this draft. (BitNet b1.58 Llama-style layers don't
        // use bias on attention/FFN projections, so this path is unused in
        // standard models; included for the optional spec-§4.4 capability.)
        ptx.push_str("shl.b64 %rd_bias_off, %rd_bias_off, 2;\n"); // 4 bytes / FP32
        ptx.push_str("add.s64 %rd_bias_addr, %rd_bias, %rd_bias_off;\n");
        ptx.push_str("ld.global.f32 %f_bias, [%rd_bias_addr];\n");
        ptx.push_str("add.f32 %f_y_out, %f_y_out, %f_bias;\n");
    }

    if config.fused_residual_add {
        // NOTE: residual is FP32 here while `y_out` is F16/BF16 — see the
        // module docstring's buffer-layout contract. 64-bit offset for the same
        // reason as the output store below.
        ptx.push_str("// Residual add: load residual[row, col] (FP32) and add.\n");
        ptx.push_str("mul.wide.u32 %rd_res_off, %r_row_id, %r_out_dim;\n");
        ptx.push_str("cvt.u64.u32 %rd_tmp, %r_col_id;\n");
        ptx.push_str("add.s64 %rd_res_off, %rd_res_off, %rd_tmp;\n");
        ptx.push_str("shl.b64 %rd_res_off, %rd_res_off, 2;\n"); // FP32
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
    // 64-bit element index. A 32-bit `row_id * out_dim + col_id` (and a 32-bit
    // byte shift) wraps once the output exceeds 2^32 elements / 4 GiB, and the
    // wrapped address lands back inside the same buffer, so a later row silently
    // overwrites an earlier one with no fault. Reproduced at out_dim=2048,
    // rows=1048577. An lm_head with out_dim=128256 wraps at only ~16.7k rows.
    ptx.push_str("mul.wide.u32 %rd_yoff, %r_row_id, %r_out_dim;\n");
    ptx.push_str("cvt.u64.u32 %rd_tmp, %r_col_id;\n");
    ptx.push_str("add.s64 %rd_yoff, %rd_yoff, %rd_tmp;\n");
    ptx.push_str(&format!(
        "shl.b64 %rd_yoff, %rd_yoff, {};\n",
        elem_bytes.trailing_zeros()
    ));
    // %rd_y_out is already a .global address (cvta'd in mod.rs::emit_prologue);
    // re-applying cvta.to.global here would be a second translation of an
    // already-translated pointer.
    ptx.push_str("add.s64 %rd_y_addr, %rd_y_out, %rd_yoff;\n");
    ptx.push_str(&format!("st.global.{store_ty} [%rd_y_addr], %h_y;\n"));

    ptx.push_str("// === end BitNet finalize ===\n");
}
