//! Tier C backward dQ and dK accumulation.
//!
//! Sources:
//!   dS from backward_ds_offset SMEM tile (ds_compute wrote it).
//!   Q  from %q_smem_base (backward q_load populated).
//!   K  from %k_smem_base (backward kv_load populated).
//!
//! Formulas (chain-rule with scale = 1/sqrt(head_dim)):
//!   dQ[row, d] += sum_col dS[row, col] * K[col, d] * scale
//!   dK[col, d] += sum_row dS[row, col] * Q[row, d] * scale
//!
//! Lane ownership (block_kv == 32):
//!   dQ sweep: warp owns Q row (warp_row), lane owns d-slice.
//!             Inner loop iterates cols.
//!   dK sweep: lane owns KV row (lane_id), each lane owns d-slice.
//!             Inner loop iterates the 4 Q rows in this tile.
//!
//! Accumulators (%f_dq_*, %f_dk_*) are register-held; reset lives in
//! the orchestrator.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::backward_ds_offset;

pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    assert_eq!(
        config.block_kv, 32,
        "T3.5 requires block_kv=32 (got {}); wider tiles land in T3.6+",
        config.block_kv
    );
    let head_dim = config.head_dim as u32;
    let block_kv = config.block_kv as u32;
    let row_stride_bytes = head_dim * 2;
    let slices_per_lane = (head_dim / 32).max(1);
    let ds_offset = backward_ds_offset(config);

    ptx.push_str(&format!(
        "    // Tier C backward dQ+dK accumulation (q_tile_iter={q_tile_iter})\n"
    ));

    // ── dQ sweep: dQ[warp_row, d_slice] += sum_col dS[warp_row, col] * K[col, d_slice] * scale
    ptx.push_str(&format!("V2_BWD_DQ_ACCUM_{q_tile_iter}:\n"));
    // warp_row = warp_id + iter*4
    ptx.push_str(&format!(
        "    add.u32 %r0, %warp_id, {}; // warp_row\n",
        q_tile_iter * 4
    ));
    ptx.push_str("    cvt.u64.u32 %warp_row, %r0;\n");

    ptx.push_str("    mov.u32 %r1, 0;   // col counter\n");
    ptx.push_str(&format!("V2_BWD_DQ_INNER_{q_tile_iter}:\n"));

    // Load dS[warp_row, col] from SMEM at
    //   backward_ds_offset + (warp_row * block_kv + col) * 4
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd30, %warp_row, {};\n",
        block_kv
    ));
    ptx.push_str("    cvt.u64.u32 %rd31, %r1;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %rd31;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 2;\n");
    ptx.push_str(&format!("    add.u64 %rd30, %rd30, {ds_offset};\n"));
    ptx.push_str("    add.u64 %rd30, %shmem_base, %rd30;\n");
    ptx.push_str("    ld.shared.f32 %f_dS, [%rd30];\n");
    // dS_scaled = dS * scale
    ptx.push_str("    mul.f32 %f_dS, %f_dS, %scale;\n");

    // K[col, :] base in SMEM: %k_smem_base + col * row_stride_bytes
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd32, %rd31, {row_stride_bytes};\n"
    ));
    ptx.push_str("    add.u64 %rd32, %k_smem_base, %rd32;  // &K[col, 0]\n");

    for slice in 0..slices_per_lane {
        // d_slice = lane * slices + slice; byte_off = d_slice * 2
        ptx.push_str("    cvt.u64.u32 %rd33, %lane;\n");
        if slices_per_lane > 1 {
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd33, %rd33, {slices_per_lane};\n"
            ));
        }
        if slice > 0 {
            ptx.push_str(&format!("    add.u64 %rd33, %rd33, {slice};\n"));
        }
        ptx.push_str("    shl.b64 %rd33, %rd33, 1;\n");
        ptx.push_str("    add.u64 %rd34, %rd32, %rd33;\n");
        ptx.push_str("    ld.shared.b16 %h0, [%rd34];\n");
        ptx.push_str("    cvt.f32.f16 %f0, %h0;\n");
        ptx.push_str(&format!(
            "    fma.rn.f32 %f_dq_{slice}, %f_dS, %f0, %f_dq_{slice};\n"
        ));
    }

    ptx.push_str("    add.u32 %r1, %r1, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %p0, %r1, {};\n", block_kv
    ));
    ptx.push_str(&format!("    @%p0 bra V2_BWD_DQ_INNER_{q_tile_iter};\n"));

    // ── dK sweep: dK[lane, d_slice] += sum_row dS[row, lane] * Q[row, d_slice] * scale
    ptx.push_str(&format!("V2_BWD_DK_ACCUM_{q_tile_iter}:\n"));
    // Iterate ALL block_q rows (dK accumulates across all queries, not
    // just this iter's 4). For block_q=block_kv=32 single-iter this is
    // rows 0..block_q.
    let block_q = config.block_q as u32;
    ptx.push_str("    mov.u32 %r1, 0;   // row counter (0..block_q)\n");
    ptx.push_str(&format!("V2_BWD_DK_INNER_{q_tile_iter}:\n"));

    // dS[row, lane] from backward_ds_offset + (row * block_kv + lane) * 4
    ptx.push_str("    cvt.u64.u32 %rd30, %r1;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd30, %rd30, {};\n", block_kv
    ));
    ptx.push_str("    cvt.u64.u32 %rd31, %lane;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %rd31;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 2;\n");
    ptx.push_str(&format!("    add.u64 %rd30, %rd30, {ds_offset};\n"));
    ptx.push_str("    add.u64 %rd30, %shmem_base, %rd30;\n");
    ptx.push_str("    ld.shared.f32 %f_dS, [%rd30];\n");
    ptx.push_str("    mul.f32 %f_dS, %f_dS, %scale;\n");

    // Q[row, :] base in SMEM: %q_smem_base + row * row_stride_bytes
    ptx.push_str("    cvt.u64.u32 %rd32, %r1;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd32, %rd32, {row_stride_bytes};\n"
    ));
    ptx.push_str("    add.u64 %rd32, %q_smem_base, %rd32;  // &Q[row, 0]\n");

    for slice in 0..slices_per_lane {
        ptx.push_str("    cvt.u64.u32 %rd33, %lane;\n");
        if slices_per_lane > 1 {
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd33, %rd33, {slices_per_lane};\n"
            ));
        }
        if slice > 0 {
            ptx.push_str(&format!("    add.u64 %rd33, %rd33, {slice};\n"));
        }
        ptx.push_str("    shl.b64 %rd33, %rd33, 1;\n");
        ptx.push_str("    add.u64 %rd34, %rd32, %rd33;\n");
        ptx.push_str("    ld.shared.b16 %h0, [%rd34];\n");
        ptx.push_str("    cvt.f32.f16 %f0, %h0;\n");
        ptx.push_str(&format!(
            "    fma.rn.f32 %f_dk_{slice}, %f_dS, %f0, %f_dk_{slice};\n"
        ));
    }

    ptx.push_str("    add.u32 %r1, %r1, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %p0, %r1, {block_q};\n"
    ));
    ptx.push_str(&format!("    @%p0 bra V2_BWD_DK_INNER_{q_tile_iter};\n"));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn base_cfg_fused_backward(
        block_q: i64, block_kv: i64, head_dim: i64, heads: u32, d_model: u32,
    ) -> FlashAttentionConfig {
        let _ = heads;
        FlashAttentionConfig {
            block_q, block_kv, head_dim,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, gpu_sm: 75,
            csha: Some(CshaExtras {
                fused_projections: true,
                save_activations_for_backward: true,
                d_model,
                ..CshaExtras::default()
            }),
        }
    }

    #[test]
    fn backward_dqdk_accum_both_directions() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert!(ptx.contains("V2_BWD_DQ_ACCUM_0:"));
        assert!(ptx.contains("V2_BWD_DK_ACCUM_0:"));
        assert!(ptx.contains("%f_dq"));
        assert!(ptx.contains("%f_dk"));
        assert!(ptx.contains("%f_dS"));
        // Real SMEM addressing:
        assert!(ptx.contains("%k_smem_base"));
        assert!(ptx.contains("%q_smem_base"));
    }

    #[test]
    fn backward_dqdk_accum_applies_scale_factor() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert!(
            ptx.contains("mul.f32 %f_dS, %f_dS, %scale"),
            "chain-rule scale factor on dS missing"
        );
    }

    #[test]
    fn backward_dqdk_accum_no_inline_reset() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert!(!ptx.contains("mov.f32 %f_dq_0, 0f00000000"));
        assert!(!ptx.contains("mov.f32 %f_dk_0, 0f00000000"));
    }

    #[test]
    fn backward_dqdk_accum_label_uniqueness_across_iters() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        emit(&mut ptx, &cfg, 1);
        for label in [
            "V2_BWD_DQ_ACCUM_0:", "V2_BWD_DQ_ACCUM_1:",
            "V2_BWD_DK_ACCUM_0:", "V2_BWD_DK_ACCUM_1:",
            "V2_BWD_DQ_INNER_0:", "V2_BWD_DQ_INNER_1:",
            "V2_BWD_DK_INNER_0:", "V2_BWD_DK_INNER_1:",
        ] {
            assert!(ptx.contains(label), "missing: {label}");
        }
    }

    #[test]
    fn backward_dqdk_accum_per_slice_fma_for_head_dim_64() {
        let cfg = base_cfg_fused_backward(32, 32, 64, 4, 64);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        // head_dim=64 → 2 slices × 2 sub-sweeps = 4 fmas.
        assert_eq!(
            ptx.matches("fma.rn.f32").count(),
            4,
            "head_dim=64 expects 4 fmas (2 slices × dQ + dK)"
        );
    }
}
