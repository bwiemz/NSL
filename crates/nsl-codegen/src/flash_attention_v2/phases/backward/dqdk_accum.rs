//! Tier C backward dQ and dK accumulation.
//!
//! Different storage strategies per gradient based on output coverage:
//!
//! - **dQ (register-held)**: warp owns Q row, lane owns d-slice.
//!   Each thread holds `%f_dq_{slice}` for its unique
//!   (warp_row, lane*slices + slice) cell. Coverage is 4_warps × 32_lanes
//!   × slices × iters = block_q × head_dim ✓ for block_q=32, head_dim=32,
//!   slices=1, iters=8 (or iters=1 for block_q=32).
//!   Formula:
//!     dQ[warp_row, d_slice] += sum_col dS[warp_row, col] * K[col, d_slice] * scale
//!
//! - **dK (SMEM-tile)**: lane owns KV col, warp owns a d-slice of
//!   `head_dim/4` d-values. Each thread accumulates into
//!   `head_dim/4` distinct (col, d) cells in the backward_dk_offset
//!   SMEM tile. No atomics — each cell has exactly one writer thread.
//!   Matches the dv_accum design (Option 1 from the Tier C close-out
//!   memory).
//!   Formula:
//!     dK[col, d] += sum_row dS[row, col] * Q[row, d] * scale

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{
    backward_dk_offset, backward_ds_offset,
};

pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    assert_eq!(
        config.block_kv, 32,
        "T3.5 requires block_kv=32 (got {})",
        config.block_kv
    );
    let head_dim = config.head_dim as u32;
    let block_q = config.block_q as u32;
    let block_kv = config.block_kv as u32;
    assert!(
        head_dim.is_multiple_of(4),
        "dqdk_accum warp-d partition requires head_dim % 4 == 0 (got {head_dim})"
    );
    let d_per_warp = head_dim / 4;
    let row_stride_bytes = head_dim * 2;
    let slices_per_lane = (head_dim / 32).max(1);
    let ds_offset = backward_ds_offset(config);
    let dk_offset = backward_dk_offset(config);

    ptx.push_str(&format!(
        "    // Tier C backward dQ+dK accumulation (q_tile_iter={q_tile_iter})\n"
    ));

    // ─────────────────────────────────────────────────────────────────────
    // dQ sweep — register-held accumulator per thread (unique cell).
    // ─────────────────────────────────────────────────────────────────────
    ptx.push_str(&format!("V2_BWD_DQ_ACCUM_{q_tile_iter}:\n"));
    ptx.push_str(&format!(
        "    add.u32 %r0, %warp_id, {}; // warp_row\n",
        q_tile_iter * 4
    ));
    ptx.push_str("    cvt.u64.u32 %warp_row, %r0;\n");

    ptx.push_str("    mov.u32 %r1, 0;   // col counter\n");
    ptx.push_str(&format!("V2_BWD_DQ_INNER_{q_tile_iter}:\n"));

    // dS[warp_row, col] → %f_dS (scaled).
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd30, %warp_row, {};\n", block_kv
    ));
    ptx.push_str("    cvt.u64.u32 %rd31, %r1;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %rd31;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 2;\n");
    ptx.push_str(&format!("    add.u64 %rd30, %rd30, {ds_offset};\n"));
    ptx.push_str("    add.u64 %rd30, %shmem_base, %rd30;\n");
    ptx.push_str("    ld.shared.f32 %f_dS, [%rd30];\n");
    ptx.push_str("    mul.f32 %f_dS, %f_dS, %scale;\n");

    // K[col, :] base
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd32, %rd31, {row_stride_bytes};\n"
    ));
    ptx.push_str("    add.u64 %rd32, %k_smem_base, %rd32;\n");

    for slice in 0..slices_per_lane {
        // d_slice = lane * slices + slice
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

    // ─────────────────────────────────────────────────────────────────────
    // dK sweep — SMEM-tile accumulator (warp-d partition, lane owns col).
    // ─────────────────────────────────────────────────────────────────────
    ptx.push_str(&format!("V2_BWD_DK_ACCUM_{q_tile_iter}:\n"));

    // Iterate the 4 Q rows in this tile-iter (not all block_q — the
    // orchestrator invokes dqdk_accum per iter). Cross-iter accumulation
    // lands naturally because each iter's inner loop adds to the same
    // SMEM cells.
    let iter_row_base = q_tile_iter * 4;
    let iter_row_count = block_q.min(iter_row_base + 4).saturating_sub(iter_row_base);

    ptx.push_str(&format!(
        "    mov.u32 %r1, 0;   // row counter (0..{iter_row_count})\n"
    ));
    ptx.push_str(&format!("V2_BWD_DK_INNER_{q_tile_iter}:\n"));
    ptx.push_str(&format!(
        "    add.u32 %r2, %r1, {iter_row_base};  // row_global\n"
    ));

    // dS[row_global, lane] (scaled) → %f_dS.
    ptx.push_str("    cvt.u64.u32 %rd30, %r2;\n");
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

    // Q[row_global, :] base in SMEM.
    ptx.push_str("    cvt.u64.u32 %rd32, %r2;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd32, %rd32, {row_stride_bytes};\n"
    ));
    ptx.push_str("    add.u64 %rd32, %q_smem_base, %rd32;\n");

    // dK tile: lane owns the row; warp owns d ∈ [warp_id*d_per_warp,
    // warp_id*d_per_warp + d_per_warp). Row stride is head_dim*4 bytes.
    ptx.push_str("    cvt.u64.u32 %rd34, %lane;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd34, %rd34, {};  // lane * head_dim * 4\n",
        head_dim * 4
    ));
    ptx.push_str(&format!(
        "    add.u64 %rd34, %rd34, {dk_offset};\n"
    ));
    ptx.push_str("    add.u64 %rd34, %shmem_base, %rd34;  // &dK[lane, 0]\n");

    // Warp d-base byte offset in the dK row.
    ptx.push_str("    cvt.u64.u32 %rd35, %warp_id;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd35, %rd35, {};\n", d_per_warp * 4
    ));
    // Warp d-base byte offset into the Q SMEM row (f16, stride 2).
    ptx.push_str("    cvt.u64.u32 %rd36, %warp_id;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd36, %rd36, {};\n", d_per_warp * 2
    ));

    for k in 0..d_per_warp {
        // dK SMEM addr = %rd34 + %rd35 + k*4
        ptx.push_str("    add.u64 %rd37, %rd34, %rd35;\n");
        if k > 0 {
            ptx.push_str(&format!("    add.u64 %rd37, %rd37, {};\n", k * 4));
        }
        ptx.push_str("    ld.shared.f32 %f0, [%rd37];\n");
        // Q[row_global, d]
        ptx.push_str("    add.u64 %rd38, %rd32, %rd36;\n");
        if k > 0 {
            ptx.push_str(&format!("    add.u64 %rd38, %rd38, {};\n", k * 2));
        }
        ptx.push_str("    ld.shared.b16 %h0, [%rd38];\n");
        ptx.push_str("    cvt.f32.f16 %f1, %h0;\n");
        ptx.push_str("    fma.rn.f32 %f0, %f_dS, %f1, %f0;\n");
        ptx.push_str("    st.shared.f32 [%rd37], %f0;\n");
    }

    ptx.push_str("    add.u32 %r1, %r1, 1;\n");
    ptx.push_str(&format!(
        "    setp.lt.u32 %p0, %r1, {iter_row_count};\n"
    ));
    ptx.push_str(&format!("    @%p0 bra V2_BWD_DK_INNER_{q_tile_iter};\n"));

    ptx.push_str("    bar.sync 0;  // dK updates visible across warps\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn base_cfg(hd: i64, dm: u32) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: hd,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0, gpu_sm: 75,
            segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: true,
                save_activations_for_backward: true,
                d_model: dm,
                ..CshaExtras::default()
            }),
        }
    }

    #[test]
    fn backward_dqdk_accum_both_directions() {
        let cfg = base_cfg(32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert!(ptx.contains("V2_BWD_DQ_ACCUM_0:"));
        assert!(ptx.contains("V2_BWD_DK_ACCUM_0:"));
        assert!(ptx.contains("%f_dq"));
        assert!(ptx.contains("%f_dS"));
        assert!(ptx.contains("%k_smem_base"));
        assert!(ptx.contains("%q_smem_base"));
        // dK uses SMEM-tile accumulation now:
        assert!(ptx.contains("ld.shared.f32 %f0"),
            "dK sweep must load current dK from SMEM");
        assert!(ptx.contains("st.shared.f32 [%rd37]"),
            "dK sweep must store updated dK back to SMEM");
        // Register-based dK accumulator removed:
        assert!(!ptx.contains("%f_dk_"),
            "dk register accumulator replaced by SMEM tile");
    }

    #[test]
    fn backward_dqdk_accum_applies_scale_factor() {
        let cfg = base_cfg(32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        // Scale applied in both dQ and dK sweeps (2 occurrences total).
        assert_eq!(
            ptx.matches("mul.f32 %f_dS, %f_dS, %scale").count(),
            2,
            "chain-rule scale must fire in both dQ and dK sweeps"
        );
    }

    #[test]
    fn backward_dqdk_accum_no_inline_reset() {
        let cfg = base_cfg(32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert!(!ptx.contains("mov.f32 %f_dq_0, 0f00000000"));
    }

    #[test]
    fn backward_dqdk_accum_label_uniqueness_across_iters() {
        let cfg = base_cfg(32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        emit(&mut ptx, &cfg, 1);
        for label in [
            "V2_BWD_DQ_ACCUM_0:", "V2_BWD_DQ_ACCUM_1:",
            "V2_BWD_DK_ACCUM_0:", "V2_BWD_DK_ACCUM_1:",
        ] {
            assert!(ptx.contains(label), "missing: {label}");
        }
    }

    #[test]
    fn backward_dqdk_accum_warp_partitions_d_for_dk() {
        // head_dim=32 → d_per_warp=8 → 8 fmas in the dK unrolled inner,
        // plus 1 fma in the dQ sweep (slices_per_lane=1) = 9 total.
        let cfg = base_cfg(32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert_eq!(
            ptx.matches("fma.rn.f32").count(),
            9,
            "head_dim=32: 1 fma (dQ) + 8 fma (dK warp-d)"
        );
    }
}
