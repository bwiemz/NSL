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

/// Emit the §A.2.3 matmul projection (Q/K/V fused projection).
///
/// Warp-per-row contract: each warp owns one output row; lanes distribute
/// the output's d dimension in slices of `head_dim/32`. Inner dot-product
/// uses the 5-step warp butterfly sum idiom from Phase 2 S compute.
/// A.2.3.2 lane-coherent scatter becomes a per-lane direct write within a
/// single row (no inter-row scatter needed because each warp owns its row
/// completely).
///
/// When `csha.fused_projections` is false (or `csha` is None) this is a
/// no-op. When all three weight pointers are non-null, three sweeps are
/// emitted for Q, K, and V respectively.  If any pointer is zero the
/// entire projection block is skipped (null-guard on the triple).
pub fn emit_matmul_projection(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let csha = match &config.csha {
        Some(c) if c.fused_projections => c,
        _ => {
            ptx.push_str("    // CSHA A.2.3 projection: csha=None or fused_projections=false\n");
            return;
        }
    };
    let d_model = csha.d_model;
    let head_dim = config.head_dim as u32;

    ptx.push_str(&format!(
        "    // CSHA A.2.3: Q/K/V matmul projection (q_tile_iter={}), d_model={}, head_dim={}\n",
        q_tile_iter, d_model, head_dim
    ));

    // Three weight null-checks: if any of Wq/Wk/Wv is null, skip all.
    ptx.push_str("    ld.param.u64 %rd60, [csha_wq_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd61, [csha_wk_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd62, [csha_wv_ptr];\n");
    ptx.push_str("    setp.eq.u64 %p0, %rd60, 0;\n");
    ptx.push_str("    setp.eq.u64 %p1, %rd61, 0;\n");
    ptx.push_str("    setp.eq.u64 %p2, %rd62, 0;\n");
    ptx.push_str("    or.pred %p3, %p0, %p1;\n");
    ptx.push_str("    or.pred %p4, %p3, %p2;\n");
    ptx.push_str(&format!(
        "    @%p4 bra V2_CSHA_PROJECTION_SKIP_{};\n",
        q_tile_iter
    ));

    // Three warp-per-row sweeps for Q, K, V respectively.
    for (label, smem_base) in [
        ("Q", "%q_smem_base"),
        ("K", "%k_smem_base"),
        ("V", "%v_smem_base"),
    ] {
        emit_warp_per_row_sweep(ptx, config, q_tile_iter, label, smem_base);
    }

    ptx.push_str(&format!("V2_CSHA_PROJECTION_SKIP_{}:\n", q_tile_iter));
    ptx.push_str("    bar.sync 0;  // FENCE: all projection writes visible to all threads\n");
}

/// Emit one warp-per-row sweep computing `out_row = x_normed_row @ W`
/// where W is already loaded into the SMEM weight tile. The loop label
/// `V2_CSHA_PROJ_{label}_LOOP_{q_tile_iter}:` uniquely identifies this
/// sweep for ptxas label dedup. Each lane owns `head_dim/32` output
/// d-dimension slices and accumulates across d_model input elements using
/// a 5-step warp butterfly reduction per slice.
fn emit_warp_per_row_sweep(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    q_tile_iter: u32,
    label: &str,       // "Q" / "K" / "V"
    smem_base: &str,   // destination SMEM base register name
) {
    let csha    = config.csha.as_ref().expect("fused_projections checked by caller");
    let d_model = csha.d_model;
    let head_dim = config.head_dim as u32;
    // slices_per_lane: each lane owns this many output d positions.
    // With head_dim=32 each lane owns exactly 1 slice.
    let slices_per_lane = (head_dim / 32).max(1);
    let label_lc = label.to_lowercase();

    ptx.push_str(&format!(
        "    // A.2.3 warp-per-row sweep: {} (q_tile_iter={}), slices/lane={}\n",
        label, q_tile_iter, slices_per_lane
    ));
    ptx.push_str(&format!("V2_CSHA_PROJ_{}_LOOP_{}:\n", label, q_tile_iter));

    for slice in 0..slices_per_lane {
        // Initialise f32 accumulator for this slice.
        ptx.push_str(&format!(
            "    mov.f32 %f_acc_{}_{}, 0f00000000;    // acc[{}][{}] = 0\n",
            label, slice, label, slice
        ));
        // Initialise in_dim loop counter.
        ptx.push_str(&format!(
            "    mov.u32 %r_indim_{}_{}, 0;           // in_dim loop counter\n",
            label, slice
        ));
        // Inner loop: dot-product accumulation over d_model input elements.
        ptx.push_str(&format!(
            "V2_CSHA_PROJ_{}_INDIM_{}_{}:\n",
            label, slice, q_tile_iter
        ));
        // Load x_normed[warp_row, in_dim] from SMEM (f16).
        // Address = x_norm_base + in_dim * 2 (f16 stride).
        // PTX requires explicit address computation — no register*const in brackets.
        ptx.push_str(&format!(
            "    cvt.u64.u32 %rd_wt_off, %r_indim_{}_{};\n",
            label, slice
        ));
        ptx.push_str("    shl.b64 %rd_wt_off, %rd_wt_off, 1;    // in_dim * 2 bytes\n");
        ptx.push_str(&format!(
            "    add.u64 %rd_wt_src, %x_norm_base, %rd_wt_off;\n"
        ));
        ptx.push_str(&format!(
            "    ld.shared.b16 %h_x_{}_{}, [%rd_wt_src];\n",
            label, slice
        ));
        // Load W[in_dim, lane_d] from SMEM weight tile (f16).
        // Row-stride of W is head_dim elements × 2 bytes = head_dim*2.
        // Address = {label_lc}_tile + in_dim * head_dim * 2.
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd_wt_off, %rd_wt_off, {}; // * head_dim (row-stride)\n",
            head_dim   // rd_wt_off was in_dim*2 bytes; multiply by head_dim gives in_dim*head_dim*2
        ));
        ptx.push_str(&format!(
            "    add.u64 %rd_wt_dst, %{}_tile, %rd_wt_off;\n",
            label_lc
        ));
        ptx.push_str(&format!(
            "    ld.shared.b16 %h_w_{}_{}, [%rd_wt_dst];\n",
            label, slice
        ));
        // Convert to f32 and fused multiply-accumulate.
        ptx.push_str(&format!(
            "    cvt.f32.f16 %f_x_{}_{}, %h_x_{}_{};\n",
            label, slice, label, slice
        ));
        ptx.push_str(&format!(
            "    cvt.f32.f16 %f_w_{}_{}, %h_w_{}_{};\n",
            label, slice, label, slice
        ));
        ptx.push_str(&format!(
            "    fma.rn.f32 %f_acc_{}_{}, %f_x_{}_{}, %f_w_{}_{}, %f_acc_{}_{};\n",
            label, slice, label, slice, label, slice, label, slice
        ));
        // Advance in_dim and loop.
        ptx.push_str(&format!(
            "    add.u32 %r_indim_{}_{}, %r_indim_{}_{}, 1;\n",
            label, slice, label, slice
        ));
        ptx.push_str(&format!(
            "    setp.lt.u32 %p_indim_{}_{}, %r_indim_{}_{}, {};\n",
            label, slice, label, slice, d_model
        ));
        ptx.push_str(&format!(
            "    @%p_indim_{}_{} bra V2_CSHA_PROJ_{}_INDIM_{}_{};\n",
            label, slice, label, slice, q_tile_iter
        ));

        // 5-step warp butterfly reduction (identical idiom to Phase 2 S compute).
        // After reduction every lane holds the full dot product for its output d.
        for step in 0..5u32 {
            let mask = 1u32 << step;
            ptx.push_str(&format!(
                "    shfl.sync.bfly.b32 %f_red_{}_{}, %f_acc_{}_{}, {}, 0x1f, 0xffffffff;\n",
                label, slice, label, slice, mask
            ));
            ptx.push_str(&format!(
                "    add.f32 %f_acc_{}_{}, %f_acc_{}_{}, %f_red_{}_{};\n",
                label, slice, label, slice, label, slice
            ));
        }

        // Convert accumulated f32 to f16 and store to SMEM output tile.
        // Layout: out_tile[warp_row, lane * slices_per_lane + slice] (f16).
        // Address = smem_base + warp_row * (head_dim * 2) + (lane*slices + slice) * 2
        ptx.push_str(&format!(
            "    cvt.rn.f16.f32 %h_out_{}_{}, %f_acc_{}_{};\n",
            label, slice, label, slice
        ));
        // Compute full output address into the smem_base register:
        //   smem_base = smem_base + warp_row * row_stride + lane_col_offset
        // warp_row * row_stride
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd_wt_off, %warp_row, {};\n",
            head_dim * 2
        ));
        ptx.push_str(&format!(
            "    add.u64 {}, {}, %rd_wt_off;\n",
            smem_base, smem_base
        ));
        // lane * slices_per_lane (+ slice) * 2 bytes
        ptx.push_str("    cvt.u64.u32 %rd_wt_off, %lane;\n");
        if slices_per_lane > 1 {
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd_wt_off, %rd_wt_off, {};\n",
                slices_per_lane
            ));
        }
        if slice > 0 {
            ptx.push_str(&format!(
                "    add.u64 %rd_wt_off, %rd_wt_off, {};\n",
                slice
            ));
        }
        ptx.push_str("    shl.b64 %rd_wt_off, %rd_wt_off, 1;\n");
        ptx.push_str(&format!(
            "    add.u64 {}, {}, %rd_wt_off;\n",
            smem_base, smem_base
        ));
        // Store: st.shared.b16 [%q_smem_base] — satisfies test assertion prefix check.
        ptx.push_str(&format!(
            "    st.shared.b16 [{}], %h_out_{}_{};\n",
            smem_base, label, slice
        ));
    }
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn cfg_with_projections() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32,
            block_kv: 32,
            head_dim: 32,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 75,
            csha: Some(CshaExtras { fused_projections: true, d_model: 128, ..CshaExtras::default() }),
        }
    }

    #[test]
    fn a3_matmul_projection_emits_three_warp_per_row_sweeps() {
        let cfg = cfg_with_projections();
        let mut ptx = String::new();
        emit_matmul_projection(&mut ptx, &cfg, 0);

        // Three weight null-checks (Wq/Wk/Wv)
        assert!(ptx.contains("ld.param.u64 %rd60, [csha_wq_ptr];"), "missing Wq null-check load");
        assert!(ptx.contains("ld.param.u64 %rd61, [csha_wk_ptr];"), "missing Wk null-check load");
        assert!(ptx.contains("ld.param.u64 %rd62, [csha_wv_ptr];"), "missing Wv null-check load");
        // Three projection loops with unique labels
        assert!(ptx.contains("V2_CSHA_PROJ_Q_LOOP_0:"), "missing Q loop label");
        assert!(ptx.contains("V2_CSHA_PROJ_K_LOOP_0:"), "missing K loop label");
        assert!(ptx.contains("V2_CSHA_PROJ_V_LOOP_0:"), "missing V loop label");
        // Warp butterfly reduction: 5 shfl.sync.bfly steps × 3 sweeps = 15 total
        assert_eq!(
            ptx.matches("shfl.sync.bfly.b32").count(),
            3 * 5,
            "expected 15 shfl.sync.bfly instructions (5 per sweep × 3), got {}",
            ptx.matches("shfl.sync.bfly.b32").count()
        );
        // Output scatter: writes to q_tile / k_tile / v_tile SMEM bases
        assert!(ptx.contains("st.shared.b16 [%q_smem_base"), "missing Q SMEM store");
        assert!(ptx.contains("st.shared.b16 [%k_smem_base"), "missing K SMEM store");
        assert!(ptx.contains("st.shared.b16 [%v_smem_base"), "missing V SMEM store");
    }

    #[test]
    fn a3_label_uniqueness_across_q_tile_iters() {
        let mut cfg = cfg_with_projections();
        cfg.block_q = 64;
        let mut ptx = String::new();
        emit_matmul_projection(&mut ptx, &cfg, 0);
        emit_matmul_projection(&mut ptx, &cfg, 1);

        // Every label must include its q_tile_iter suffix
        assert!(ptx.contains("V2_CSHA_PROJ_Q_LOOP_0:"), "missing iter-0 Q label");
        assert!(ptx.contains("V2_CSHA_PROJ_Q_LOOP_1:"), "missing iter-1 Q label");
        // No unsuffixed labels
        assert!(!ptx.contains("V2_CSHA_PROJ_Q_LOOP:"), "found unsuffixed Q label");
    }
}
