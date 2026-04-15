//! Tier C backward dV accumulation: `dV[col, d] += sum_row P[row, col] * dO[row, d]`.
//!
//! Per-lane-owns-column design (mirrors forward Tier A projection):
//! each lane holds `head_dim/32` output d-slots for one KV row in the
//! `%f_dv_{slice}` register pool; the inner loop iterates Q rows.
//!
//! ## Accumulator lifecycle
//!
//! Reset of `%f_dv_{slice}` to zero lives in the ORCHESTRATOR
//! (T4.1), not here. This emit function only adds to the existing
//! accumulator — the orchestrator calls `emit(..)` once per KV tile
//! and writes the final `%f_dv_{slice}` to the SMEM dV region after
//! the last tile.
//!
//! That boundary keeps the per-phase emit stateless and lets the
//! orchestrator drive the tile loop without per-phase resets
//! corrupting cross-tile accumulation.

use crate::flash_attention::FlashAttentionConfig;

/// Emit one KV-tile worth of `dV += P^T @ dO` accumulation. Expects
/// the T3.3 dS phase to have written P values to SMEM already; the
/// inner loop reads P[row, col] for this lane's column and `dO[row, d]`
/// from the backward dO pointer.
///
/// Scope (matches T3.3): `block_kv == 32` — lane owns column = lane_id.
pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    assert_eq!(
        config.block_kv, 32,
        "T3.4 requires block_kv=32 (got {}); wider tiles land in T3.6+",
        config.block_kv
    );
    let head_dim = config.head_dim as u32;
    let slices_per_lane = (head_dim / 32).max(1);

    ptx.push_str(&format!(
        "    // Tier C backward dV accumulation (q_tile_iter={q_tile_iter})\n"
    ));
    ptx.push_str(&format!("V2_BWD_DV_ACCUM_{q_tile_iter}:\n"));

    // warp_row = warp_id + iter*4 — this tile covers Q rows
    // [q_start + iter*4, q_start + iter*4 + 4).
    ptx.push_str(&format!(
        "    add.u32 %r0, %warp_id, {}; // warp_row\n",
        q_tile_iter * 4
    ));
    ptx.push_str("    cvt.u64.u32 %warp_row, %r0;\n");

    // Inner loop over the 4 Q rows in this tile (warps cooperate: each
    // warp owns one row, but dV accumulation is across all Q rows, so
    // we iterate the 4-row block here and read P/dO per row from SMEM/
    // HBM respectively. For the skeleton, use a placeholder scalar
    // inner loop; real Q-row iteration lives in T3.6's orchestrator.)
    ptx.push_str("    mov.u32 %r1, 0;   // Q-row counter in tile (0..4)\n");
    ptx.push_str(&format!("V2_BWD_DV_INNER_{q_tile_iter}:\n"));

    // Load P[row, lane] from SMEM. T3.3 stored dS there; in the real
    // orchestrator a distinct P-tile slot will be used. For the
    // skeleton we read the same slot so ptxas validates the access
    // pattern.
    ptx.push_str("    cvt.u64.u32 %rd30, %warp_id;\n");
    ptx.push_str(&format!("    mul.lo.u64 %rd30, %rd30, {};\n", 32 * 4));
    ptx.push_str("    cvt.u64.u32 %rd31, %lane;\n");
    ptx.push_str("    shl.b64 %rd31, %rd31, 2;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %rd31;\n");
    ptx.push_str("    add.u64 %rd30, %shmem_base, %rd30;\n");
    ptx.push_str("    ld.shared.f32 %f_P, [%rd30];\n");

    // For each slice owned by this lane, accumulate
    //   dv_slice += P[row, lane] * dO[row, d_slice]
    for slice in 0..slices_per_lane {
        // Placeholder dO[row, d_slice] = 1.0 so ptxas can type-check the
        // fma. Full dO addressing lives in T3.6's orchestrator.
        ptx.push_str(&format!(
            "    mov.f32 %f0, 0f3F800000;   // placeholder dO[row, slice={slice}]\n"
        ));
        ptx.push_str(&format!(
            "    fma.rn.f32 %f_dv_{slice}, %f_P, %f0, %f_dv_{slice};\n"
        ));
    }

    // Advance Q-row counter.
    ptx.push_str("    add.u32 %r1, %r1, 1;\n");
    ptx.push_str("    setp.lt.u32 %p0, %r1, 4;\n");
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
            "dv accumulator reset belongs in orchestrator, not inside accum emit"
        );
    }

    #[test]
    fn backward_dv_accum_label_uniqueness_across_iters() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        emit(&mut ptx, &cfg, 1);
        assert!(ptx.contains("V2_BWD_DV_ACCUM_0:"));
        assert!(ptx.contains("V2_BWD_DV_ACCUM_1:"));
        assert!(ptx.contains("V2_BWD_DV_INNER_0:"));
        assert!(ptx.contains("V2_BWD_DV_INNER_1:"));
    }

    #[test]
    fn backward_dv_accum_emits_per_slice_fma_for_head_dim_64() {
        // head_dim=64 → slices_per_lane=2 → should emit 2 fmas per iter.
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
