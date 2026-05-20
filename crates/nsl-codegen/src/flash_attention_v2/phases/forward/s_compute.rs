//! Phase 2 - S = Q.K^T (warp-per-row, lane-distributed d, sequential
//! over k).
//!
//! Each warp computes its q_row's S values. For each k in 0..block_kv:
//! lanes cooperate on a warp-wide dot product (lane-local multiply +
//! 5-step shfl.sync.bfly add reduction), then lane 0 stores the scaled
//! (and optionally causally-masked) S value into shmem_S[warp_id, k].
//!
//! Pre-conditions (set by prelude + q_load + k_load):
//!   %f{Q_BASE..Q_BASE + head_dim/32 - 1} on lane L hold Q[q_row, L+32i]
//!   %scale, %shmem_base, %q_start, %head_idx, %batch_idx: set
//!   %warp_id, %lane: set
//!   shmem K tile at kv_offset(config) populated as f16
//!   %k_start, %k_max: current tile's [k_start, k_max)
//!
//! Post-condition: shmem_S[warp_id, k] holds the final scaled (and
//! masked) S[q_row, k] for k in 0..block_kv.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::phases::q_load::Q_BASE;
use crate::flash_attention_v2::phases::softmax::emit_direct_hbm_store_of_reg;
use crate::flash_attention_v2::smem_layout::{kv_offset, sp_offset};
use crate::pca_segment::SegmentResidency;

/// Emit the S = Q·Kᵀ computation for one kv-tile.
///
/// `tier_b` — when `Some((seq_len, residency))`, emits the PCA Tier B
/// skip predicate at the top of the tile body (before QK^T) if the
/// range-table budget check passes.  `None` disables Tier B emission
/// (used by the existing `synthesize_flash_attention_ptx_v2` path).
pub fn emit(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    q_tile_iter: u32,
    tier_b: Option<(u32, SegmentResidency)>,
) {
    let head_dim = config.head_dim as u32;
    let slices = head_dim / 32;
    let block_kv = config.block_kv as u32;
    let fused = config.csha.as_ref().is_some_and(|c| c.fused_projections);
    let capture = config
        .csha
        .as_ref()
        .is_some_and(|c| c.save_activations_for_backward);
    // J-A5 s_compute-internal captures. Env-gated at codegen time.
    //   - direct_q0        : lane-0 snapshot of %f{Q_BASE} before k loop
    //                        (should be 32.0 for ones input; tells us if
    //                        q_load populated Q registers correctly).
    //   - direct_k00       : lane-0 snapshot of %f1 (= K[k=0, d=lane=0])
    //                        after the SMEM load and cvt in slice-0 of the
    //                        k=0 iteration; tells us if k_tile_load wrote
    //                        the SMEM kv_offset region correctly.
    //   - direct_qk_raw    : lane-0 snapshot of %f0 after the fma slices
    //                        loop completes for k=0, before butterfly sum;
    //                        the per-lane partial Q*K contribution. Lane 0
    //                        for slice-0 of head_dim=32 is q[0]*k[0,0].
    let diag_mode: String = std::env::var("NSL_CSHA_DUMP_SAVE_STATE")
        .ok()
        .unwrap_or_default();
    // SP slice base offset for this q_tile_iter (same logic as softmax.rs).
    let sp_iter_offset = if fused {
        sp_offset(config) + q_tile_iter * 4 * block_kv * 4
    } else {
        sp_offset(config)
    };

    ptx.push_str(&format!(
        "    // Phase 2: S = Q*K^T (q_tile_iter = {})\n",
        q_tile_iter
    ));

    // PCA Tier B: tile-level skip predicate — fires once per kv-tile BEFORE
    // the QK^T inner loop.  When ranges are disjoint the entire tile is
    // skipped (including softmax + PV accumulation) by branching to
    // KV_TILE_SKIP_TB_{q_tile_iter}, which is placed in the orchestrator
    // (mod.rs) just before the k_start increment.
    //
    // Implementation notes:
    //   - %k_start (u64) is the absolute token position of the kv-tile
    //     start, set by the orchestrator BEFORE emit() is called.  We derive
    //     the kv-tile ordinal with a right-shift by log2(block_kv) (block_kv
    //     is always a power of 2 in the supported tile matrix).
    //   - The q-tile ordinal is the per-CTA global index `%bid_x` (each CTA
    //     handles ONE q-tile per the launch contract `grid_x = num_q_tiles`).
    //     `q_tile_iter` is the WITHIN-CTA inner iter 0..(block_q/4 - 1) and
    //     does NOT index the range table. Using `q_tile_iter` (as the v1
    //     emission did) reads only range-table slots 0..15 regardless of
    //     which q-tile this CTA is processing, so the predicate always
    //     loaded entries for the first 16 global q-tiles — broken for any
    //     CTA with bid_x >= 16 and incorrect even for bid_x ∈ 0..15.
    //     Codified in `docs/superpowers/specs/2026-05-13-tier-b-b15-3-skip-ratio-investigation.md`.
    if let Some((seq_len, residency)) = tier_b {
        if crate::pca_tilerange::should_emit_tier_b(config, seq_len as u64, residency) {
            let log2_bkv = block_kv.trailing_zeros();
            let range_table_base =
                crate::flash_attention_v2::smem_layout::tier_b_range_table_offset(
                    config,
                    crate::flash_attention_v2::smem_layout::Direction::Forward,
                );
            let skip_label = format!("KV_TILE_SKIP_TB_{q_tile_iter}");
            // Wrap in a PTX lexical scope so the kvt-ordinal registers
            // are local to this q_tile_iter (s_compute::emit is called
            // once per q_iter; without the scope, ptxas would reject
            // duplicate %rd_kvt_ord_TB / %r_kvt_ord_TB declarations).
            ptx.push_str("    { // PCA Tier B per-q_iter scope\n");
            ptx.push_str("    // PCA Tier B: derive kv-tile ordinal from %k_start\n");
            ptx.push_str("    .reg .u64 %rd_kvt_ord_TB;\n");
            ptx.push_str("    .reg .u32 %r_kvt_ord_TB;\n");
            ptx.push_str(&format!(
                "    shr.b64 %rd_kvt_ord_TB, %k_start, {log2_bkv};\n"
            ));
            ptx.push_str("    cvt.u32.u64 %r_kvt_ord_TB, %rd_kvt_ord_TB;\n");
            crate::pca_tilerange::emit_skip_predicate(
                ptx,
                config,
                seq_len,
                "%bid_x", // global q-tile index for THIS CTA (grid_x = num_q_tiles)
                "%r_kvt_ord_TB",
                range_table_base,
                &skip_label,
                // Forward FA-2: per-CTA q-tile fixed by %bid_x, q-iter
                // Rust-unrolled, kv-tile is the PTX-runtime inner loop.
                crate::pca_tilerange::IterationOrder::QOuter,
            );
            ptx.push_str("    } // end PCA Tier B per-q_iter scope\n");
        }
    }

    // J-A5 direct_scale: snapshot %scale (the 1/sqrt(d_k) param loaded at
    // kernel prelude) once per warp before the k loop. Ones-input expected
    // value: 1/sqrt(32) ≈ 0.176777. If this reads back ~1e-34, the scale
    // parameter isn't reaching the kernel correctly (or ld.param.f32 /
    // the FFI is mis-threading the bits).
    if capture && diag_mode == "direct_scale" {
        emit_direct_hbm_store_of_reg(ptx, q_tile_iter, "lane-0 %scale param", "%scale");
    }

    // J-A5 direct_q0: snapshot lane-0's Q slice-0 register before the k
    // loop. Stored to row_max HBM so the dump infrastructure can read it.
    if capture && diag_mode == "direct_q0" {
        let q0_reg = format!("%f{}", Q_BASE);
        emit_direct_hbm_store_of_reg(ptx, q_tile_iter, "lane-0 Q_BASE (Q slice 0)", &q0_reg);
    }

    // Loop over k in 0..block_kv.
    ptx.push_str("    mov.u32 %r1, 0;                           // k = 0\n");
    ptx.push_str(&format!(
        "    mov.u32 %r2, {};                           // block_kv\n",
        block_kv
    ));
    ptx.push_str(&format!("V2_LOOP_S_OVER_K_{}:\n", q_tile_iter));

    // K row base in shmem: kv_offset + k*head_dim*2 (f16 bytes).
    ptx.push_str("    cvt.u64.u32 %rd32, %r1;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd32, %rd32, {};              // k * head_dim\n",
        head_dim
    ));
    ptx.push_str("    shl.b64 %rd32, %rd32, 1;                  // * 2 bytes f16\n");
    ptx.push_str(&format!(
        "    add.u64 %rd32, %rd32, {};                 // + kv_offset\n",
        kv_offset(config)
    ));
    ptx.push_str("    add.u64 %rd32, %rd32, %shmem_base;\n");

    // Per-lane partial dot product over all slices.
    ptx.push_str("    mov.f32 %f0, 0f00000000;                  // partial = 0\n");
    for i in 0..slices {
        ptx.push_str("    cvt.u64.u32 %rd33, %lane;\n");
        if i > 0 {
            ptx.push_str(&format!("    add.u64 %rd33, %rd33, {};\n", i * 32));
        }
        ptx.push_str("    shl.b64 %rd33, %rd33, 1;                  // * 2 bytes f16\n");
        ptx.push_str("    add.u64 %rd33, %rd33, %rd32;              // + K row base\n");
        ptx.push_str("    ld.shared.b16 %h0, [%rd33];\n");
        ptx.push_str("    cvt.f32.f16 %f1, %h0;                     // K[k, d]\n");
        // J-A5 direct_k00 capture: fires only on slice 0 AND only when the
        // k-loop iterator %r1 == 0. %f1 here is K[k=0, d=lane] in f32,
        // just after cvt.f32.f16 from the SMEM b16 tile.
        if i == 0 && capture && diag_mode == "direct_k00" {
            ptx.push_str("    setp.ne.u32 %p0, %r1, 0;\n");
            ptx.push_str(&format!("    @%p0 bra V2_DIAG_K00_SKIP_{};\n", q_tile_iter));
            emit_direct_hbm_store_of_reg(
                ptx,
                q_tile_iter,
                "K[k=0, d=lane] from SMEM (lane-0 stores K[0,0])",
                "%f1",
            );
            ptx.push_str(&format!("V2_DIAG_K00_SKIP_{}:\n", q_tile_iter));
        }
        ptx.push_str(&format!(
            "    fma.rn.f32 %f0, %f{}, %f1, %f0;           // partial += Q*K\n",
            Q_BASE + i
        ));
    }

    // J-A5 direct_qk_raw capture: after the per-slice fma accumulation,
    // BEFORE butterfly sum. Fires only when k-loop iterator %r1 == 0.
    // %f0 on lane 0 is q[0] * k[0, 0..slice-1] summed — the per-lane
    // partial Q*K for lane 0, before the butterfly folds in other lanes'
    // contributions. For head_dim=32 (slices=1), this equals q[0] * k[0,0].
    if capture && diag_mode == "direct_qk_raw" {
        ptx.push_str("    setp.ne.u32 %p0, %r1, 0;\n");
        ptx.push_str(&format!(
            "    @%p0 bra V2_DIAG_QK_RAW_SKIP_{};\n",
            q_tile_iter
        ));
        emit_direct_hbm_store_of_reg(
            ptx,
            q_tile_iter,
            "per-lane partial Q*K for k=0, pre-butterfly",
            "%f0",
        );
        ptx.push_str(&format!("V2_DIAG_QK_RAW_SKIP_{}:\n", q_tile_iter));
    }

    // 5-step warp butterfly: every lane ends with the full dot product.
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %shfl_tmp, %f0, {}, 31, 0xFFFFFFFF;\n",
            offset
        ));
        ptx.push_str("    add.f32 %f0, %f0, %shfl_tmp;\n");
    }
    // J-A5 direct_post_bfly capture: post-butterfly, pre-scale. Lane-0 %f0
    // should be the sum of 32 per-lane partials. For ones, 32 * 1024 = 32768.
    if capture && diag_mode == "direct_post_bfly" {
        ptx.push_str("    setp.ne.u32 %p0, %r1, 0;\n");
        ptx.push_str(&format!(
            "    @%p0 bra V2_DIAG_POST_BFLY_SKIP_{};\n",
            q_tile_iter
        ));
        emit_direct_hbm_store_of_reg(
            ptx,
            q_tile_iter,
            "lane-0 %f0 post-butterfly-sum for k=0 (pre-scale)",
            "%f0",
        );
        ptx.push_str(&format!("V2_DIAG_POST_BFLY_SKIP_{}:\n", q_tile_iter));
    }
    ptx.push_str("    mul.f32 %f0, %f0, %scale;                 // S *= 1/sqrt(d_k)\n");
    // J-A5 direct_post_scale capture: post-scale, pre-causal. Should be
    // 32768 / sqrt(32) ≈ 5792.6 for ones input.
    if capture && diag_mode == "direct_post_scale" {
        ptx.push_str("    setp.ne.u32 %p0, %r1, 0;\n");
        ptx.push_str(&format!(
            "    @%p0 bra V2_DIAG_POST_SCALE_SKIP_{};\n",
            q_tile_iter
        ));
        emit_direct_hbm_store_of_reg(ptx, q_tile_iter, "lane-0 %f0 post-scale for k=0", "%f0");
        ptx.push_str(&format!("V2_DIAG_POST_SCALE_SKIP_{}:\n", q_tile_iter));
    }

    // Causal mask: if k_global > q_row_global, S = -inf.
    // When both config.causal and config.segment_masked are true, also
    // mask if segment_ids[q_row_global] != segment_ids[k_global] (spec §5.1).
    if config.causal {
        ptx.push_str("    // causal: if k_global > q_row_global -> S = -inf\n");
        ptx.push_str("    cvt.u64.u32 %rd34, %r1;                   // k\n");
        ptx.push_str("    add.u64 %rd34, %rd34, %k_start;           // k_global\n");
        ptx.push_str(&format!(
            "    add.u32 %r3, %warp_id, {};                // q_row_local\n",
            q_tile_iter * 4
        ));
        ptx.push_str("    cvt.u64.u32 %rd35, %r3;\n");
        ptx.push_str("    add.u64 %rd35, %q_start, %rd35;            // q_row_global\n");
        ptx.push_str("    setp.gt.u64 %p0, %rd34, %rd35;\n");
        // PCA Tier A: extend %p0 with cross-segment disjunction (spec §5.1).
        // Guard: only when BOTH causal (so %p0 is already set) AND segment_masked.
        // If !causal && segment_masked, %p0 is uninitialized; we skip rather than
        // extend a garbage predicate (spec §3.2 rejects that config in practice).
        if config.segment_masked {
            crate::flash_attention_v2::phases::segment_mask::emit_segment_mask_predicate(
                ptx,
                "%rd35",     // q_pos_reg (q_row_global, u64)
                "%rd34",     // k_pos_reg  (k_global, u64)
                "%seg_base", // SMEM base declared in prelude (u64 generic address)
                crate::pca_segment::SegmentResidency::Shared,
                "%p0", // extend caller's mask predicate in place
            );
        }
        ptx.push_str("    @%p0 mov.f32 %f0, 0fFF800000;             // -inf\n");
    }

    // J-A5 direct_s_store_pre capture: lane-0 %f0 immediately before the
    // SMEM store. This is the value s_compute COMMITS to shmem_S[warp, k].
    // Only fires for k=0. If this shows 5792.6 (scaled Q*K for ones input)
    // but softmax's direct_s0 reads back 1e-30, something overwrites SMEM
    // between s_compute's write and softmax's read.
    if capture && diag_mode == "direct_s_store_pre" {
        ptx.push_str("    setp.ne.u32 %p0, %r1, 0;\n");
        ptx.push_str(&format!(
            "    @%p0 bra V2_DIAG_S_STORE_PRE_SKIP_{};\n",
            q_tile_iter
        ));
        emit_direct_hbm_store_of_reg(
            ptx,
            q_tile_iter,
            "lane-0 %f0 pre-SMEM-store for k=0 (scaled post-causal S)",
            "%f0",
        );
        ptx.push_str(&format!("V2_DIAG_S_STORE_PRE_SKIP_{}:\n", q_tile_iter));
    }

    // Lane 0 stores full S to shmem_S[warp_id, k].
    ptx.push_str("    setp.eq.u32 %p1, %lane, 0;\n");
    ptx.push_str("    cvt.u64.u32 %rd36, %warp_id;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd36, %rd36, {};              // warp_id * block_kv\n",
        block_kv
    ));
    ptx.push_str("    cvt.u64.u32 %rd37, %r1;\n");
    ptx.push_str("    add.u64 %rd36, %rd36, %rd37;              // + k\n");
    ptx.push_str("    shl.b64 %rd36, %rd36, 2;                  // * 4 bytes f32\n");
    ptx.push_str(&format!(
        "    add.u64 %rd36, %rd36, {};                 // + sp_iter_offset\n",
        sp_iter_offset
    ));
    ptx.push_str("    add.u64 %smem_addr, %rd36, %shmem_base;\n");
    ptx.push_str("    @%p1 st.shared.f32 [%smem_addr], %f0;\n");

    ptx.push_str("    add.u32 %r1, %r1, 1;\n");
    ptx.push_str("    setp.lt.u32 %p0, %r1, %r2;\n");
    ptx.push_str(&format!("    @%p0 bra V2_LOOP_S_OVER_K_{};\n", q_tile_iter));
    ptx.push_str("    bar.sync 0;  // FENCE: all warps finished S writes\n");
}
