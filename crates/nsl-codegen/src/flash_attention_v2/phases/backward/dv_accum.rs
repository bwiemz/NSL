//! Tier C backward dV accumulation: `dV[col, d] += sum_row P[row, col] * dO[row, d]`.
//!
//! # Warp-partitioned-d layout (Option 1 from the close-out memory)
//!
//! Cells: `[block_kv × head_dim]` dV accumulator lives in SMEM at
//! `backward_dv_offset` (f32, row-major). The orchestrator cooperatively
//! zeros the tile before the KV loop; this emit function only does
//! `+=` accumulation. No atomics: each (col, d) cell is written by
//! exactly one thread per emit call.
//!
//! Thread ↔ cell partition (assumes head_dim % 4 == 0, block_kv == 32):
//!   lane      → KV col (lane_id 0..31 covers block_kv=32 cols)
//!   warp      → owns a contiguous d-slice of `head_dim/4` values,
//!               specifically d ∈ [warp_id*d_per_warp, warp_id*d_per_warp + d_per_warp)
//!
//! Each thread therefore accumulates into `d_per_warp` cells, iterating
//! an outer loop over Q rows. For the smoke config (head_dim=32,
//! d_per_warp=8), 128 threads cover all 32×32 = 1024 output cells.
//!
//! Reads:
//!   P   — `backward_p_offset` SMEM tile (ds_compute wrote it).
//!   dO  — HBM via `%rd_bwd_do` (backward prelude loaded the pointer).

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{backward_dv_offset, backward_p_offset};

pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    assert_eq!(
        config.block_kv, 32,
        "T3.4 requires block_kv=32 (got {}); wider tiles follow once the \
         warp-d-partition generalises",
        config.block_kv
    );
    let head_dim = config.head_dim as u32;
    let block_q = config.block_q as u32;
    let block_kv = config.block_kv as u32;
    assert!(
        head_dim.is_multiple_of(4),
        "dv_accum warp-d partition requires head_dim % 4 == 0 (got {head_dim})"
    );
    let d_per_warp = head_dim / 4;
    let p_offset = backward_p_offset(config);
    let dv_offset = backward_dv_offset(config);

    ptx.push_str(&format!(
        "    // Tier C backward dV accumulation (q_tile_iter={q_tile_iter}, \
         d_per_warp={d_per_warp})\n"
    ));
    ptx.push_str(&format!("V2_BWD_DV_ACCUM_{q_tile_iter}:\n"));

    // This iter covers Q rows [q_tile_iter*4, q_tile_iter*4 + 4). Each
    // tile-iter call accumulates those 4 rows' contribution into the
    // SMEM dV tile. Across all iters the full block_q is covered.
    let iter_row_base = q_tile_iter * 4;
    let iter_row_count = block_q.min(iter_row_base + 4).saturating_sub(iter_row_base);

    ptx.push_str(&format!(
        "    mov.u32 %r1, 0;   // Q-row counter (0..{iter_row_count})\n"
    ));
    ptx.push_str(&format!("V2_BWD_DV_INNER_{q_tile_iter}:\n"));

    // row_global = iter_row_base + %r1
    ptx.push_str(&format!(
        "    add.u32 %r2, %r1, {iter_row_base};  // row_global\n"
    ));
    ptx.push_str("    cvt.u64.u32 %rd39, %r2;\n");
    ptx.push_str("    add.u64 %rd39, %rd39, %q_start;\n");
    ptx.push_str("    setp.ge.u64 %p0, %rd39, %rd6;\n");
    ptx.push_str(&format!("    @%p0 bra V2_BWD_DV_SKIP_ROW_{q_tile_iter};\n"));

    // Load P[row_global, lane] from SMEM.
    ptx.push_str("    cvt.u64.u32 %rd30, %r2;\n");
    ptx.push_str(&format!("    mul.lo.u64 %rd30, %rd30, {};\n", block_kv));
    ptx.push_str("    cvt.u64.u32 %rd31, %lane;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %rd31;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 2;\n");
    ptx.push_str(&format!("    add.u64 %rd30, %rd30, {p_offset};\n"));
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

    // dV_tile base for this lane's col: dv_offset + lane * head_dim * 4
    // (row stride = head_dim * 4 f32).
    ptx.push_str("    cvt.u64.u32 %rd34, %lane;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd34, %rd34, {};  // lane * head_dim * 4\n",
        head_dim * 4
    ));
    ptx.push_str(&format!("    add.u64 %rd34, %rd34, {dv_offset};\n"));
    ptx.push_str("    add.u64 %rd34, %shmem_base, %rd34;  // &dV[lane, 0]\n");

    // Inner loop over this warp's d-slice: d = warp_id*d_per_warp + k, k ∈ 0..d_per_warp.
    // k unrolled to keep addressing cheap; each thread does d_per_warp iters.
    // d_base_byte_off (in the dV tile row) = warp_id * d_per_warp * 4.
    ptx.push_str("    cvt.u64.u32 %rd35, %warp_id;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd35, %rd35, {};  // warp d-base * 4\n",
        d_per_warp * 4
    ));
    // dO byte offset for this warp's d-base = warp_id * d_per_warp * 2 (f16).
    ptx.push_str("    cvt.u64.u32 %rd36, %warp_id;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd36, %rd36, {};\n",
        d_per_warp * 2
    ));

    for k in 0..d_per_warp {
        // dV SMEM addr = %rd34 + %rd35 + k*4
        ptx.push_str("    add.u64 %rd37, %rd34, %rd35;\n");
        if k > 0 {
            ptx.push_str(&format!("    add.u64 %rd37, %rd37, {};\n", k * 4));
        }
        // Load current dV, accumulate, store back.
        ptx.push_str("    ld.shared.f32 %f0, [%rd37];\n");
        // dO[row, d=warp_id*d_per_warp + k]
        ptx.push_str("    add.u64 %rd38, %rd33, %rd36;\n");
        if k > 0 {
            ptx.push_str(&format!("    add.u64 %rd38, %rd38, {};\n", k * 2));
        }
        ptx.push_str("    ld.global.b16 %h0, [%rd38];\n");
        ptx.push_str("    cvt.f32.f16 %f1, %h0;\n");
        ptx.push_str("    fma.rn.f32 %f0, %f_P, %f1, %f0;\n");
        ptx.push_str("    st.shared.f32 [%rd37], %f0;\n");
    }

    ptx.push_str("    add.u32 %r1, %r1, 1;\n");
    ptx.push_str(&format!("    setp.lt.u32 %p0, %r1, {iter_row_count};\n"));
    ptx.push_str(&format!("    @%p0 bra V2_BWD_DV_INNER_{q_tile_iter};\n"));
    ptx.push_str(&format!("    bra V2_BWD_DV_DONE_ROW_{q_tile_iter};\n"));
    ptx.push_str(&format!("V2_BWD_DV_SKIP_ROW_{q_tile_iter}:\n"));
    ptx.push_str("    add.u32 %r1, %r1, 1;\n");
    ptx.push_str(&format!("    setp.lt.u32 %p0, %r1, {iter_row_count};\n"));
    ptx.push_str(&format!("    @%p0 bra V2_BWD_DV_INNER_{q_tile_iter};\n"));
    ptx.push_str(&format!("V2_BWD_DV_DONE_ROW_{q_tile_iter}:\n"));

    ptx.push_str("    bar.sync 0;  // dV updates visible across warps\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn base_cfg(hd: i64, dm: u32) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32,
            block_kv: 32,
            head_dim: hd,
            causal: false,
            paged: false,
            rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1,
            tree_mask: false,
            gpu_sm: 75,
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
    fn backward_dv_accum_p_transpose_do() {
        let cfg = base_cfg(32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);

        assert!(ptx.contains("V2_BWD_DV_ACCUM_0:"));
        assert!(ptx.contains("fma.rn.f32"));
        // SMEM-based accumulation pattern:
        assert!(
            ptx.contains("ld.shared.f32 %f0"),
            "must load current dV from SMEM"
        );
        assert!(
            ptx.contains("st.shared.f32 [%rd37]"),
            "must store updated dV back to SMEM"
        );
        assert!(ptx.contains("%rd_bwd_do"), "must read dO from HBM");
        // Register-based accumulator path removed — no more %f_dv_*.
        assert!(
            !ptx.contains("%f_dv_"),
            "register-based accumulator replaced by SMEM tile"
        );
    }

    #[test]
    fn backward_dv_accum_warp_partitions_d_for_head_dim_32() {
        // head_dim=32, d_per_warp = head_dim/4 = 8, so 8 fma unrolled per thread.
        let cfg = base_cfg(32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert_eq!(
            ptx.matches("fma.rn.f32").count(),
            8,
            "head_dim=32 → d_per_warp=8 → 8 fmas per thread per Q-row iter"
        );
    }

    #[test]
    fn backward_dv_accum_label_uniqueness_across_iters() {
        let cfg = base_cfg(32, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        emit(&mut ptx, &cfg, 1);
        assert!(ptx.contains("V2_BWD_DV_ACCUM_0:"));
        assert!(ptx.contains("V2_BWD_DV_ACCUM_1:"));
    }

    #[test]
    fn backward_dv_accum_head_dim_64_doubles_fma_count() {
        let cfg = base_cfg(64, 64);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert_eq!(
            ptx.matches("fma.rn.f32").count(),
            16,
            "head_dim=64 → d_per_warp=16 → 16 fmas per thread"
        );
    }
}
