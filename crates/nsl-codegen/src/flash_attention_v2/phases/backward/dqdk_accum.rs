//! Tier C backward dQ and dK accumulation.
//!
//! Two lane-owns-column sub-sweeps consuming the dS tile produced by
//! T3.3:
//!
//!   dQ[row, d] += sum_col dS[row, col] * K[col, d]
//!   dK[col, d] += sum_row dS[row, col] * Q[row, d]
//!
//! Lane ownership (block_kv == 32):
//!   - dQ sweep: warp owns Q row; lane owns one d-slice of that row.
//!     Inner loop iterates columns (dS[row, col]) and K[col, d].
//!   - dK sweep: lane owns one KV row's d-slice. Inner loop iterates
//!     Q rows (dS[row, col] and Q[row, d]).
//!
//! Accumulator lifecycle matches T3.4 dv_accum: resets live in T4.1's
//! orchestrator, not here. Cross-tile accumulation requires that the
//! emit function only *adds* to `%f_dq_{slice}` / `%f_dk_{slice}`.
//!
//! Both sub-sweeps must multiply their partial sum by `%scale`
//! (= `1/sqrt(head_dim)`) because the forward S = Q @ K^T / sqrt(d) —
//! the chain-rule factor propagates to dQ and dK. Applied once per
//! accumulator update to keep the inner fma tight.

use crate::flash_attention::FlashAttentionConfig;

/// Emit one KV-tile worth of dQ + dK accumulation.
pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    assert_eq!(
        config.block_kv, 32,
        "T3.5 requires block_kv=32 (got {}); wider tiles land in T3.6+",
        config.block_kv
    );
    let head_dim = config.head_dim as u32;
    let slices_per_lane = (head_dim / 32).max(1);

    ptx.push_str(&format!(
        "    // Tier C backward dQ+dK accumulation (q_tile_iter={q_tile_iter})\n"
    ));

    // ── dQ sweep: each lane owns a d-slice of its warp's Q row ──────────
    ptx.push_str(&format!("V2_BWD_DQ_ACCUM_{q_tile_iter}:\n"));
    // Inner loop over KV columns (col ∈ 0..block_kv).
    ptx.push_str("    mov.u32 %r1, 0;   // col counter\n");
    ptx.push_str(&format!("V2_BWD_DQ_INNER_{q_tile_iter}:\n"));

    // Load dS[warp_row, col] from SMEM. T3.3 wrote dS at
    // %shmem_base + warp_id*32*4 + col*4.
    ptx.push_str("    cvt.u64.u32 %rd30, %warp_id;\n");
    ptx.push_str(&format!("    mul.lo.u64 %rd30, %rd30, {};\n", 32 * 4));
    ptx.push_str("    cvt.u64.u32 %rd31, %r1;\n");
    ptx.push_str("    shl.b64 %rd31, %rd31, 2;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %rd31;\n");
    ptx.push_str("    add.u64 %rd30, %shmem_base, %rd30;\n");
    ptx.push_str("    ld.shared.f32 %f_dS, [%rd30];\n");

    for slice in 0..slices_per_lane {
        // Placeholder K[col, d_slice]; real K addressing lives in T3.6.
        ptx.push_str(&format!(
            "    mov.f32 %f0, 0f3F800000;   // placeholder K[col, slice={slice}]\n"
        ));
        // dq_slice += dS[row, col] * K[col, d] * scale
        ptx.push_str("    mul.f32 %f1, %f_dS, %scale;\n");
        ptx.push_str(&format!(
            "    fma.rn.f32 %f_dq_{slice}, %f1, %f0, %f_dq_{slice};\n"
        ));
    }

    ptx.push_str("    add.u32 %r1, %r1, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %p0, %r1, {};\n",
        config.block_kv
    ));
    ptx.push_str(&format!("    @%p0 bra V2_BWD_DQ_INNER_{q_tile_iter};\n"));

    // ── dK sweep: each lane owns a d-slice of a KV row ──────────────────
    // KV row index for this lane: lane_id (block_kv=32 scope). A full
    // block has 32 KV rows, one per lane across the warp; each warp
    // handles its q_tile row indirectly via the dS[warp_row, lane_col]
    // readback. For block_kv=32 we use lane as the KV-row index so dK
    // is accumulated per-lane into that lane's %f_dk_{slice}.
    ptx.push_str(&format!("V2_BWD_DK_ACCUM_{q_tile_iter}:\n"));
    // Inner loop over the 4 Q rows in this tile.
    ptx.push_str("    mov.u32 %r1, 0;   // Q-row counter (0..4)\n");
    ptx.push_str(&format!("V2_BWD_DK_INNER_{q_tile_iter}:\n"));

    // Load dS[row, lane] from SMEM. Here `row` is the iter's q-row index
    // (0..4); the stored dS layout puts warp_id in the row slot, so we
    // read from the %r1-indexed row entry. For the skeleton, reuse the
    // same SMEM address pattern as T3.3/T3.4 keyed by %r1 and %lane.
    ptx.push_str("    cvt.u64.u32 %rd30, %r1;\n");
    ptx.push_str(&format!("    mul.lo.u64 %rd30, %rd30, {};\n", 32 * 4));
    ptx.push_str("    cvt.u64.u32 %rd31, %lane;\n");
    ptx.push_str("    shl.b64 %rd31, %rd31, 2;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %rd31;\n");
    ptx.push_str("    add.u64 %rd30, %shmem_base, %rd30;\n");
    ptx.push_str("    ld.shared.f32 %f_dS, [%rd30];\n");

    for slice in 0..slices_per_lane {
        // Placeholder Q[row, d_slice]; real Q addressing lives in T3.6.
        ptx.push_str(&format!(
            "    mov.f32 %f0, 0f3F800000;   // placeholder Q[row, slice={slice}]\n"
        ));
        ptx.push_str("    mul.f32 %f1, %f_dS, %scale;\n");
        ptx.push_str(&format!(
            "    fma.rn.f32 %f_dk_{slice}, %f1, %f0, %f_dk_{slice};\n"
        ));
    }

    ptx.push_str("    add.u32 %r1, %r1, 1;\n");
    ptx.push_str("    setp.lt.u32 %p0, %r1, 4;\n");
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
    }

    #[test]
    fn backward_dqdk_accum_applies_scale_factor() {
        // dS should be multiplied by %scale (1/sqrt(d)) before fma.
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert!(
            ptx.contains("mul.f32 %f1, %f_dS, %scale"),
            "chain-rule scale factor on dS missing — dQ/dK would be off by sqrt(d)"
        );
    }

    #[test]
    fn backward_dqdk_accum_no_inline_reset() {
        // Reset of %f_dq_*/%f_dk_* belongs in the orchestrator.
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
        // 2 slices × 2 sub-sweeps (dQ, dK) = 4 fmas total per iter.
        assert_eq!(
            ptx.matches("fma.rn.f32").count(),
            4,
            "head_dim=64 expects 4 fmas (2 slices × dQ + dK)"
        );
        for reg in ["%f_dq_0", "%f_dq_1", "%f_dk_0", "%f_dk_1"] {
            assert!(ptx.contains(reg), "missing accumulator reg {reg}");
        }
    }
}
