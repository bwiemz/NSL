//! CSHA Tier A extras - prologue (RMSNorm), matmul projection (Q/K/V/O),
//! RoPE epilogue, active_heads guard. Each hook is null-guarded: if the
//! respective CSHA pointer is 0 (e.g. `csha: None`), the kernel skips
//! the phase and falls through to the classic Q-from-HBM path.
//!
//! All hooks obey the warp-per-row contract. Labels are parameterised
//! on `q_tile_iter` so the orchestrator (Task 11) can call them multiple
//! times for block_q > 4 configs without duplicate-label errors.

use crate::flash_attention::FlashAttentionConfig;

/// Emit the §A.4 active_heads guard. When `csha_active_heads` param is
/// non-zero and `head_idx >= csha_active_heads`, the kernel returns
/// immediately (dead-head pruning). Null guard: param=0 means "no
/// pruning, run all heads".
pub fn emit_active_heads_guard(ptx: &mut String, config: &FlashAttentionConfig) {
    if config.csha.is_none() {
        ptx.push_str("    // CSHA A.4 active_heads guard: csha=None, no emission\n");
        return;
    }
    ptx.push_str("    // CSHA A.4: active_heads guard\n");
    ptx.push_str("    ld.param.u32 %r10, [csha_active_heads];\n");
    ptx.push_str("    setp.eq.u32 %p0, %r10, 0;\n");
    ptx.push_str("    @%p0 bra V2_CSHA_ACTIVE_HEADS_SKIP;\n");
    // If head_idx >= active_heads, early-exit.
    ptx.push_str("    cvt.u32.u64 %r11, %head_idx;\n");
    ptx.push_str("    setp.ge.u32 %p0, %r11, %r10;\n");
    ptx.push_str("    @%p0 ret;\n");
    ptx.push_str("V2_CSHA_ACTIVE_HEADS_SKIP:\n");
}

/// Emit the §A.2.2 RMSNorm prologue. Computes
///     x_normed = x / sqrt(mean(x^2) + eps) * norm_weight
/// for the warp's query row and writes the result back into the x
/// buffer in-place. Null-guarded on `csha_x_ptr`.
pub fn emit_prologue(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    if config.csha.is_none() {
        ptx.push_str("    // CSHA A.2.2 prologue: csha=None, no emission\n");
        return;
    }
    let head_dim = config.head_dim as u32;
    let slices = head_dim / 32;

    ptx.push_str(&format!(
        "    // CSHA A.2.2: RMSNorm prologue (q_tile_iter = {})\n",
        q_tile_iter
    ));
    // Null-guard on x_ptr.
    ptx.push_str("    ld.param.u64 %rd52, [csha_x_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p0, %rd52, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_CSHA_PROLOGUE_SKIP_{};\n",
        q_tile_iter
    ));

    // Each warp normalizes its own x_row. Lane-strided sumsq across
    // head_dim slices, warp butterfly reduce, divide, multiply by
    // per-dim norm_weight.
    ptx.push_str("    mov.f32 %f0, 0f00000000;             // sumsq = 0\n");
    for i in 0..slices {
        ptx.push_str(&format!("    // x slice {}: load, square, accumulate\n", i));
        // Compute x row global offset.
        // x layout: [batch, heads, seq, head_dim] row-major, f32.
        ptx.push_str("    cvt.u64.u32 %rd53, %lane;\n");
        if i > 0 {
            ptx.push_str(&format!("    add.u64 %rd53, %rd53, {};\n", i * 32));
        }
        ptx.push_str("    mul.lo.u64 %rd54, %batch_idx, %rd5;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %head_idx;\n");
        ptx.push_str("    mul.lo.u64 %rd54, %rd54, %rd6;\n");
        ptx.push_str(&format!(
            "    add.u32 %r12, %warp_id, {};\n",
            q_tile_iter * 4
        ));
        ptx.push_str("    cvt.u64.u32 %rd55, %r12;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %q_start;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %rd55;\n");
        ptx.push_str("    mul.lo.u64 %rd54, %rd54, %rd7;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %rd53;\n");
        ptx.push_str("    shl.b64 %rd54, %rd54, 2;\n");
        ptx.push_str("    add.u64 %rd54, %rd52, %rd54;\n");
        ptx.push_str("    ld.global.f32 %f1, [%rd54];\n");
        ptx.push_str("    fma.rn.f32 %f0, %f1, %f1, %f0;            // sumsq += x*x\n");
    }
    // 5-step butterfly sum.
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %shfl_tmp, %f0, {}, 31, 0xFFFFFFFF;\n",
            offset
        ));
        ptx.push_str("    add.f32 %f0, %f0, %shfl_tmp;\n");
    }
    // mean = sumsq / head_dim; rms = sqrt(mean + eps); norm = 1/rms
    ptx.push_str(&format!(
        "    mov.f32 %f1, 0f{:08X};       // 1.0 / head_dim\n",
        (1.0f32 / head_dim as f32).to_bits()
    ));
    ptx.push_str("    mul.f32 %f0, %f0, %f1;\n");
    ptx.push_str("    ld.param.f32 %f1, [csha_eps];\n");
    ptx.push_str("    add.f32 %f0, %f0, %f1;\n");
    ptx.push_str("    sqrt.approx.f32 %f0, %f0;\n");
    ptx.push_str("    rcp.approx.f32 %f0, %f0;                  // 1/rms\n");

    // Second pass: x_normed[d] = x[d] * (1/rms) * norm_weight[d], writeback.
    for i in 0..slices {
        ptx.push_str(&format!("    // x slice {}: normalize + scale, writeback\n", i));
        ptx.push_str("    cvt.u64.u32 %rd53, %lane;\n");
        if i > 0 {
            ptx.push_str(&format!("    add.u64 %rd53, %rd53, {};\n", i * 32));
        }
        ptx.push_str("    mul.lo.u64 %rd54, %batch_idx, %rd5;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %head_idx;\n");
        ptx.push_str("    mul.lo.u64 %rd54, %rd54, %rd6;\n");
        ptx.push_str(&format!(
            "    add.u32 %r12, %warp_id, {};\n",
            q_tile_iter * 4
        ));
        ptx.push_str("    cvt.u64.u32 %rd55, %r12;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %q_start;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %rd55;\n");
        ptx.push_str("    mul.lo.u64 %rd54, %rd54, %rd7;\n");
        ptx.push_str("    add.u64 %rd54, %rd54, %rd53;\n");
        ptx.push_str("    shl.b64 %rd54, %rd54, 2;\n");
        ptx.push_str("    add.u64 %rd54, %rd52, %rd54;\n");
        ptx.push_str("    ld.global.f32 %f2, [%rd54];\n");
        ptx.push_str("    mul.f32 %f2, %f2, %f0;                    // x * 1/rms\n");
        // norm_weight[d] load
        ptx.push_str("    ld.param.u64 %rd56, [csha_norm_weight_ptr];\n");
        ptx.push_str("    shl.b64 %rd57, %rd53, 2;\n");
        ptx.push_str("    add.u64 %rd56, %rd56, %rd57;\n");
        ptx.push_str("    ld.global.f32 %f3, [%rd56];\n");
        ptx.push_str("    mul.f32 %f2, %f2, %f3;\n");
        ptx.push_str("    st.global.f32 [%rd54], %f2;\n");
    }

    ptx.push_str(&format!("V2_CSHA_PROLOGUE_SKIP_{}:\n", q_tile_iter));
    ptx.push_str("    bar.sync 0;  // FENCE: all prologue writes complete\n");
}

/// Emit the §A.2.3 matmul projection (Q/K/V/O fused projection).
///
/// **FRAMING SKELETON ONLY.** Null-guard + placeholder-body + bar.sync.
/// The full projection body (per-output-element dot-product across
/// head_dim, lane-coherent scatter for A.2.3.2) is substantial work
/// deserving its own dedicated task. The skeleton ensures the
/// orchestrator can call this hook today, null-guard path works, and
/// future authors can iterate the body without disturbing the framing.
///
/// TODO(fa-v2-projection): implement the full Q/K/V/O matmul body per
/// v1's `emit_csha_matmul_projection` (see `crates/nsl-codegen/src/
/// flash_attention.rs`), adapted to v2's warp-per-row contract:
///   * Each warp owns one output row; lanes distribute the output's
///     d dimension in slices of head_dim/32.
///   * Inner dot product uses 5-step warp butterfly sum (same pattern
///     as Phase 2 S compute).
///   * A.2.3.2 lane-coherent scatter becomes a per-lane direct write
///     within a single row (no inter-row scatter needed because each
///     warp owns its row completely).
pub fn emit_matmul_projection(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    if config.csha.is_none() {
        ptx.push_str("    // CSHA A.2.3 projection: csha=None, no emission\n");
        return;
    }
    ptx.push_str(&format!(
        "    // CSHA A.2.3: Q/K/V matmul projection (q_tile_iter = {})\n",
        q_tile_iter
    ));
    // Null-guard on wq_ptr (if any projection weight is null, skip all).
    ptx.push_str("    ld.param.u64 %rd60, [csha_wq_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p0, %rd60, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_CSHA_PROJECTION_SKIP_{};\n",
        q_tile_iter
    ));

    // FRAMING SKELETON -- real body deferred to TODO(fa-v2-projection).
    ptx.push_str("    // TODO(fa-v2-projection): full Q/K/V/O matmul body\n");
    ptx.push_str("    // Port v1's emit_csha_matmul_projection to warp-per-row:\n");
    ptx.push_str("    //   for each output element (q_row, d) owned by this lane:\n");
    ptx.push_str("    //     acc = 0\n");
    ptx.push_str("    //     for in_dim in 0..d_model:\n");
    ptx.push_str("    //       acc += x_normed[q_row, in_dim] * W[in_dim, d]\n");
    ptx.push_str("    //     write acc to Q/K/V/O tile (A.2.3.2 lane-coherent scatter)\n");
    ptx.push_str("    // See plan Task 10 step 4 + spec Section 1 for full algorithm.\n");

    ptx.push_str(&format!("V2_CSHA_PROJECTION_SKIP_{}:\n", q_tile_iter));
    ptx.push_str("    bar.sync 0;  // FENCE: all projection writes complete (skeleton)\n");
}

/// Emit the §A.2.4 RoPE Q-rotation epilogue. Applied to the post-
/// attention Q tile (%f{O_BASE+i} on each lane) using the same cos/sin
/// tables as Q-load's Phase 1 RoPE. Null-guarded on `cos_ptr` AND only
/// emits when `rope_q=true` (otherwise no rotation to apply).
pub fn emit_rope_epilogue(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    if config.csha.is_none() || !config.rope_q {
        ptx.push_str("    // CSHA A.2.4 RoPE epilogue: csha=None or rope_q=false, no emission\n");
        return;
    }
    ptx.push_str(&format!(
        "    // CSHA A.2.4: RoPE epilogue (q_tile_iter = {})\n",
        q_tile_iter
    ));
    ptx.push_str("    ld.param.u64 %rd62, [cos_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p0, %rd62, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_CSHA_EPILOGUE_SKIP_{};\n",
        q_tile_iter
    ));

    // TODO(fa-v2-rope-epilogue): apply the same HalfSplit/Adjacent
    // rotation as q_load's Phase 1 RoPE but on the post-attention output
    // registers %f{O_BASE+i}. Shares the sign-flip correctness gap
    // documented in q_load.rs (currently deferred to a rope_q=true test
    // expansion). For now this is a structural skeleton.
    ptx.push_str("    // TODO(fa-v2-rope-epilogue): real rotation body\n");
    ptx.push_str("    // Same shape as q_load's emit_rope_rotation_inline but\n");
    ptx.push_str("    // operates on %f{O_BASE+i} instead of %f{Q_BASE+i}.\n");

    ptx.push_str(&format!("V2_CSHA_EPILOGUE_SKIP_{}:\n", q_tile_iter));
}
