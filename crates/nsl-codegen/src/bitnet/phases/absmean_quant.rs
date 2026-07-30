//! BitNet 8-bit ABSMAX activation quantization prologue.
//!
//! Spec: docs/superpowers/specs/2026-05-11-m35-1-bitnet-ternary-design.md §4.2.
//! (Per PI.2 spec correction: BitNet b1.58 uses per-row absmax for activations,
//! not absmean as spec §4.2 originally stated. File name retained for spec
//! traceability; math is absmax. Reference: tests/bitnet_reference_impl.rs.)
//!
//! **Internal phase emitter** — do NOT call from outside `bitnet/`. Use
//! `quantized_ternary_gemm::emit` (the fused public path) instead. This
//! visibility is `pub` (rather than the originally-intended `pub(super)`)
//! only to enable structural snapshot tests from integration tests in
//! `tests/`, which compile as separate crates and cannot reach
//! `pub(super)`/`pub(crate)` items. IR-001 ("only fused
//! `quantized_ternary_gemm` is publicly callable") is therefore
//! documentation-enforced here, not API-shape-enforced; future Rust
//! visibility refinements (e.g., a `__test_only` feature gate) can
//! tighten this if external misuse appears.
//!
//! Performs a per-row reduction over `hidden_dim` elements — warp-shuffle
//! butterfly, then a cross-warp pass through SMEM — followed by a per-element
//! quantize:
//!
//! ```text
//! scale   = max_k(|x[r, k]|)
//! q[r, k] = round(clip(x[r, k] * 127.0 / scale, -127, 127))
//! ```
//!
//! A zero-magnitude row writes all zeros (and leaves `%f_scale` at 0, which
//! makes the downstream dequant produce an all-zero output row).
//!
//! ## Rounding
//!
//! The CPU reference uses Rust's `f32::round`, which is round-half-**away
//! -from-zero**. PTX `cvt.rni` is round-half-to-**even**, so this emitter
//! instead computes `trunc(x + copysign(0.5, x))` via `copysign.f32` +
//! `cvt.rzi.s32.f32`. Without that the two disagree by one int8 step on exact
//! `.5` values, which is visible in the bit-exact fixture comparison.

use crate::bitnet::config::BitNetKernelConfig;

/// Emit absmax-quant prologue PTX.
///
/// Reads:
/// - `%rd_act_in` — global base pointer to the activation matrix.
/// - `%r_row_id`, `%r_tid`, `%r_nthreads` — CTA row and thread geometry.
/// - `%r_hidden_dim` — hidden dimension.
/// - `%r_qact_base`, `%r_red_base`, `%r_scale_base` — SMEM base addresses.
///
/// Writes:
/// - `%f_scale` — the FP32 absmax scale for this row, broadcast to all threads.
/// - SMEM `bitnet_qact[0..hidden_dim]` — the int8 quantized activation row.
///
/// Ends with a `bar.sync` so the quantized row is visible to every thread
/// before the GEMM consumes it. **Every thread in the CTA must reach this
/// emitter** — the out-of-range column guard is emitted afterwards, by
/// `ternary_gemm::emit`.
pub fn emit(ptx: &mut String, config: &BitNetKernelConfig) {
    let hidden_dim = config.hidden_dim;

    ptx.push_str(&format!(
        "// === BitNet absmax_quant (hidden_dim={hidden_dim}) ===\n"
    ));

    // ---- pass 1: per-thread partial absmax over a strided slice of the row ----
    ptx.push_str("// Per-thread: track max(|x[r, k]|) over a grid-stride slice of the row.\n");
    ptx.push_str("mov.f32 %f_max, 0f00000000;\n");
    ptx.push_str("mov.u32 %r_k, %r_tid;\n");
    ptx.push_str("ABSMAX_LOOP:\n");
    ptx.push_str("setp.ge.u32 %p_amdone, %r_k, %r_hidden_dim;\n");
    ptx.push_str("@%p_amdone bra ABSMAX_END;\n");
    ptx.push_str("mul.lo.u32 %r_off, %r_row_id, %r_hidden_dim;\n");
    ptx.push_str("add.u32 %r_off, %r_off, %r_k;\n");
    ptx.push_str("shl.b32 %r_off, %r_off, 1;\n");
    ptx.push_str("cvt.u64.u32 %rd_off, %r_off;\n");
    ptx.push_str("add.s64 %rd_addr, %rd_act_in, %rd_off;\n");
    ptx.push_str("ld.global.b16 %h_x, [%rd_addr];\n");
    ptx.push_str(&format!("{} %f_x, %h_x;\n", cvt_to_f32(config)));
    ptx.push_str("abs.f32 %f_absx, %f_x;\n");
    ptx.push_str("max.f32 %f_max, %f_max, %f_absx;\n");
    ptx.push_str("add.u32 %r_k, %r_k, %r_nthreads;\n");
    ptx.push_str("bra ABSMAX_LOOP;\n");
    ptx.push_str("ABSMAX_END:\n");

    // ---- pass 2: warp-shuffle butterfly reduction ----
    // shfl.sync operates on .b32 lanes; move through an integer register rather
    // than shuffling an .f32 directly so the operand types are unambiguous.
    ptx.push_str("// Warp-shuffle butterfly reduction (parallel, not serialized).\n");
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str("mov.b32 %r_shfl_a, %f_max;\n");
        ptx.push_str(&format!(
            "shfl.sync.bfly.b32 %r_shfl_b, %r_shfl_a, {offset}, 0x1f, 0xffffffff;\n"
        ));
        ptx.push_str("mov.b32 %f_partial, %r_shfl_b;\n");
        ptx.push_str("max.f32 %f_max, %f_max, %f_partial;\n");
    }

    // ---- pass 3: cross-warp reduction through SMEM ----
    // A butterfly only reduces within a warp; CTAs wider than 32 threads need a
    // second stage or every warp keeps its own (too small) maximum.
    ptx.push_str("// Cross-warp: lane 0 of each warp publishes its partial to SMEM.\n");
    ptx.push_str("and.b32 %r_lane, %r_tid, 31;\n");
    ptx.push_str("shr.u32 %r_warp, %r_tid, 5;\n");
    ptx.push_str("setp.ne.u32 %p_lane0, %r_lane, 0;\n");
    ptx.push_str("@%p_lane0 bra ABSMAX_SKIP_PUBLISH;\n");
    ptx.push_str("shl.b32 %r_addr32, %r_warp, 2;\n");
    ptx.push_str("add.u32 %r_addr32, %r_red_base, %r_addr32;\n");
    ptx.push_str("st.shared.f32 [%r_addr32], %f_max;\n");
    ptx.push_str("ABSMAX_SKIP_PUBLISH:\n");
    ptx.push_str("bar.sync 0;\n");

    ptx.push_str("// Thread 0 reduces the per-warp partials and publishes the row scale.\n");
    ptx.push_str("setp.ne.u32 %p_t0, %r_tid, 0;\n");
    ptx.push_str("@%p_t0 bra ABSMAX_SKIP_FINAL;\n");
    ptx.push_str("add.u32 %r_nwarps, %r_nthreads, 31;\n");
    ptx.push_str("shr.u32 %r_nwarps, %r_nwarps, 5;\n");
    ptx.push_str("mov.f32 %f_max, 0f00000000;\n");
    ptx.push_str("mov.u32 %r_i, 0;\n");
    ptx.push_str("ABSMAX_RED_LOOP:\n");
    ptx.push_str("setp.ge.u32 %p_rdone, %r_i, %r_nwarps;\n");
    ptx.push_str("@%p_rdone bra ABSMAX_RED_END;\n");
    ptx.push_str("shl.b32 %r_addr32, %r_i, 2;\n");
    ptx.push_str("add.u32 %r_addr32, %r_red_base, %r_addr32;\n");
    ptx.push_str("ld.shared.f32 %f_partial, [%r_addr32];\n");
    ptx.push_str("max.f32 %f_max, %f_max, %f_partial;\n");
    ptx.push_str("add.u32 %r_i, %r_i, 1;\n");
    ptx.push_str("bra ABSMAX_RED_LOOP;\n");
    ptx.push_str("ABSMAX_RED_END:\n");
    ptx.push_str("st.shared.f32 [%r_scale_base], %f_max;\n");
    ptx.push_str("ABSMAX_SKIP_FINAL:\n");
    ptx.push_str("bar.sync 0;\n");
    ptx.push_str("ld.shared.f32 %f_scale, [%r_scale_base];\n");
    ptx.push_str("// %f_scale now holds the row-wide absmax on every thread.\n");

    // ---- pass 4: quantize the row into SMEM ----
    // The zero-magnitude row is handled with a predicated select rather than a
    // separate branch target: div by zero yields inf/nan, so the select must
    // come after the arithmetic and overwrite it.
    ptx.push_str("// Zero-magnitude row guard: scale == 0 -> q = 0 for every element.\n");
    ptx.push_str("setp.eq.f32 %p_zero, %f_scale, 0f00000000;\n");
    ptx.push_str("// Per-thread quantize: q = round(clip(x * 127.0 / scale, -127, 127)).\n");
    ptx.push_str("mov.f32 %f_inv_scale, 0f42FE0000;\n");
    ptx.push_str("div.rn.f32 %f_inv_scale, %f_inv_scale, %f_scale;\n");
    ptx.push_str("mov.f32 %f_half, 0f3F000000;\n");
    ptx.push_str("mov.u32 %r_k, %r_tid;\n");
    ptx.push_str("QUANT_LOOP:\n");
    ptx.push_str("setp.ge.u32 %p_qdone, %r_k, %r_hidden_dim;\n");
    ptx.push_str("@%p_qdone bra QUANT_END;\n");
    ptx.push_str("mul.lo.u32 %r_qoff, %r_row_id, %r_hidden_dim;\n");
    ptx.push_str("add.u32 %r_qoff, %r_qoff, %r_k;\n");
    ptx.push_str("shl.b32 %r_qoff, %r_qoff, 1;\n");
    ptx.push_str("cvt.u64.u32 %rd_off, %r_qoff;\n");
    ptx.push_str("add.s64 %rd_addr, %rd_act_in, %rd_off;\n");
    ptx.push_str("ld.global.b16 %h_xq, [%rd_addr];\n");
    ptx.push_str(&format!("{} %f_x, %h_xq;\n", cvt_to_f32(config)));
    ptx.push_str("mul.f32 %f_scaled, %f_x, %f_inv_scale;\n");
    ptx.push_str("max.f32 %f_clip, %f_scaled, 0fC2FE0000;\n");
    ptx.push_str("min.f32 %f_clip, %f_clip, 0f42FE0000;\n");
    ptx.push_str(
        "// Round half AWAY FROM ZERO to match the CPU reference's f32::round\n\
         // (cvt.rni would round half to even and disagree by one int8 step).\n",
    );
    ptx.push_str("copysign.f32 %f_rndbias, %f_clip, %f_half;\n");
    ptx.push_str("add.f32 %f_rnd, %f_clip, %f_rndbias;\n");
    ptx.push_str("cvt.rzi.s32.f32 %r_qi, %f_rnd;\n");
    ptx.push_str("selp.b32 %r_qi, 0, %r_qi, %p_zero;\n");
    ptx.push_str("cvt.u16.u32 %rs_qi8, %r_qi;\n");
    ptx.push_str("add.u32 %r_addr32, %r_qact_base, %r_k;\n");
    ptx.push_str("st.shared.s8 [%r_addr32], %rs_qi8;\n");
    ptx.push_str("add.u32 %r_k, %r_k, %r_nthreads;\n");
    ptx.push_str("bra QUANT_LOOP;\n");
    ptx.push_str("QUANT_END:\n");
    ptx.push_str("bar.sync 0;\n");

    ptx.push_str("// === end BitNet absmax_quant ===\n");
}

/// FP16/BF16 -> FP32 conversion opcode for the configured activation dtype.
fn cvt_to_f32(config: &BitNetKernelConfig) -> &'static str {
    use crate::kernel_ir::KirType;
    match config.activation_dtype {
        KirType::F16 => "cvt.f32.f16",
        KirType::Bf16 => "cvt.f32.bf16",
        ref other => panic!(
            "absmax_quant: activation_dtype must be F16 or Bf16, got {other:?}"
        ),
    }
}
