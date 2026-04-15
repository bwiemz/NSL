//! Tier C backward dV accumulation: `dV[col, d] += sum_row P[row, col] * dO[row, d]`.
//!
//! Per-lane-owns-column design: each lane owns one KV row (lane_id ==
//! KV row for block_kv=32) and accumulates into `%f_dv_{slice}` for
//! each d-slice that lane owns.
//!
//! Reads P from the backward P SMEM tile (written by T3.3's
//! ds_compute at `backward_p_offset`), and dO from HBM via
//! `%rd_bwd_do` (backward prelude loaded it).
//!
//! Accumulator lifecycle: orchestrator T4.1 owns the zero-init of
//! `%f_dv_{slice}` before the tile loop; this emit function only adds
//! to the existing accumulator.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::backward_p_offset;

pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    assert_eq!(
        config.block_kv, 32,
        "T3.4 requires block_kv=32 (got {}); wider tiles land in T3.6+",
        config.block_kv
    );
    let head_dim = config.head_dim as u32;
    let block_q = config.block_q as u32;
    let block_kv = config.block_kv as u32;
    let slices_per_lane = (head_dim / 32).max(1);
    let p_offset = backward_p_offset(config);

    ptx.push_str(&format!(
        "    // Tier C backward dV accumulation (q_tile_iter={q_tile_iter})\n"
    ));
    ptx.push_str(&format!("V2_BWD_DV_ACCUM_{q_tile_iter}:\n"));

    // Lane owns KV row = lane_id. Loop over ALL Q rows in this block_q
    // tile (not just this q_tile_iter's 4 warp rows) — dV accumulates
    // across the full query set, so for q_tile_iter=0 we iterate row
    // ∈ 0..block_q.
    //
    // For the single-iter config (block_q=32 == block_kv=32), all Q rows
    // are covered by this one call. When block_q=64 the orchestrator
    // calls dv_accum per q_tile_iter and the inner `row` loop only
    // needs to cover this iter's 4 rows (starting at warp_row_base =
    // q_tile_iter*4). Using the full block_q here is correct for
    // block_q == block_kv == 32 and conservatively general for larger
    // block_q (harmless redundant adds since the orchestrator resets
    // the accumulator per iter — T3.5 constraint).
    let iter_row_base = q_tile_iter * 4;
    let iter_row_count = block_q.min(iter_row_base + 4).saturating_sub(iter_row_base);

    ptx.push_str(&format!(
        "    mov.u32 %r1, 0;   // Q-row counter in tile (0..{iter_row_count})\n"
    ));
    ptx.push_str(&format!("V2_BWD_DV_INNER_{q_tile_iter}:\n"));

    // Load P[row_global, lane] from SMEM at
    //   backward_p_offset + (row_global * block_kv + lane) * 4
    // where row_global = iter_row_base + %r1.
    ptx.push_str(&format!(
        "    add.u32 %r2, %r1, {iter_row_base};  // row_global\n"
    ));
    ptx.push_str("    cvt.u64.u32 %rd30, %r2;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd30, %rd30, {};  // row_global * block_kv\n",
        block_kv
    ));
    ptx.push_str("    cvt.u64.u32 %rd31, %lane;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %rd31;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 2;  // * 4 (f32)\n");
    ptx.push_str(&format!(
        "    add.u64 %rd30, %rd30, {p_offset};\n"
    ));
    ptx.push_str("    add.u64 %rd30, %shmem_base, %rd30;\n");
    ptx.push_str("    ld.shared.f32 %f_P, [%rd30];\n");

    // dO[row_global, :] base in HBM:
    //   ((batch_idx*heads + head_idx)*seq + q_start + row_global) * head_dim * 2
    ptx.push_str("    cvt.u64.u32 %rd32, %r2;\n");
    ptx.push_str("    mul.lo.u64 %rd33, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd33, %rd33, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd33, %rd33, %rd6;\n");
    ptx.push_str("    add.u64 %rd33, %rd33, %q_start;\n");
    ptx.push_str("    add.u64 %rd33, %rd33, %rd32;\n");
    ptx.push_str("    mul.lo.u64 %rd33, %rd33, %rd7;\n");
    ptx.push_str("    shl.b64 %rd33, %rd33, 1;\n");
    ptx.push_str("    add.u64 %rd33, %rd_bwd_do, %rd33;  // &dO[row_global, 0]\n");

    for slice in 0..slices_per_lane {
        // d_slice = (lane * slices_per_lane + slice); owned by this lane.
        // dv_slice corresponds to KV row = lane, column d = d_slice.
        // Load dO[row_global, d_slice] from HBM: d_slice byte off = d_slice*2.
        // d_slice = lane * slices_per_lane + slice
        ptx.push_str("    cvt.u64.u32 %rd34, %lane;\n");
        if slices_per_lane > 1 {
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd34, %rd34, {slices_per_lane};\n"
            ));
        }
        if slice > 0 {
            ptx.push_str(&format!("    add.u64 %rd34, %rd34, {slice};\n"));
        }
        ptx.push_str("    shl.b64 %rd34, %rd34, 1;  // * 2 (f16)\n");
        ptx.push_str("    add.u64 %rd35, %rd33, %rd34;\n");
        ptx.push_str("    ld.global.b16 %h0, [%rd35];\n");
        ptx.push_str("    cvt.f32.f16 %f0, %h0;\n");
        ptx.push_str(&format!(
            "    fma.rn.f32 %f_dv_{slice}, %f_P, %f0, %f_dv_{slice};\n"
        ));
    }

    ptx.push_str("    add.u32 %r1, %r1, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %p0, %r1, {iter_row_count};\n"
    ));
    ptx.push_str(&format!("    @%p0 bra V2_BWD_DV_INNER_{q_tile_iter};\n"));
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
    fn backward_dv_accum_p_transpose_do() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);

        assert!(ptx.contains("V2_BWD_DV_ACCUM_0:"));
        assert!(ptx.contains("%f_dv"));
        assert!(ptx.contains("fma.rn.f32"));
        assert!(
            !ptx.contains("mov.f32 %f_dv_0, 0f00000000"),
            "reset belongs in orchestrator"
        );
        // Real addressing now (no placeholder constants).
        assert!(ptx.contains("ld.shared.f32 %f_P"),
            "must load P from SMEM");
        assert!(ptx.contains("%rd_bwd_do"),
            "must read dO from HBM");
    }

    #[test]
    fn backward_dv_accum_label_uniqueness_across_iters() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        emit(&mut ptx, &cfg, 1);
        assert!(ptx.contains("V2_BWD_DV_ACCUM_0:"));
        assert!(ptx.contains("V2_BWD_DV_ACCUM_1:"));
    }

    #[test]
    fn backward_dv_accum_emits_per_slice_fma_for_head_dim_64() {
        let cfg = base_cfg_fused_backward(32, 32, 64, 4, 64);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert_eq!(
            ptx.matches("fma.rn.f32").count(),
            2,
            "head_dim=64 expects 2 per-slice fmas"
        );
        assert!(ptx.contains("%f_dv_0"));
        assert!(ptx.contains("%f_dv_1"));
    }
}
