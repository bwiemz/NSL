//! Tier C backward dS compute: recompute P from saved stats + real
//! Q/K SMEM addressing, compute dP = dO @ V^T with real HBM/SMEM
//! addressing, then dS via the softmax Jacobian.
//!
//! Pipeline per warp-row (one Q row per warp, block_kv == 32 so
//! lane_id == column):
//!   1. HBM load row_max, row_sum for this (batch, head, q_start+warp_row).
//!   2. Recompute S[row, lane] = Q[row,:] · K[lane,:] / sqrt(head_dim)
//!      by iterating d over head_dim, reading Q from SMEM (loaded by
//!      backward q_load) and K from SMEM (loaded by backward kv_load).
//!   3. Apply causal mask when enabled: S = -INF for lane > warp_row.
//!   4. P[row, lane] = ex2((S - row_max) * log2e) / row_sum.
//!      Store P to its backward SMEM tile at backward_p_offset +
//!      (warp_row * block_kv + lane) * 4 so dv_accum can read it.
//!   5. Compute dP[row, lane] = dO[row, :] · V[lane, :] by iterating
//!      d, reading dO from HBM and V from SMEM.
//!   6. Accumulate rowsum_dP_P += dP * P (lane-local).
//!   7. Warp-butterfly reduce rowsum_dP_P.
//!   8. dS[row, lane] = P * (dP - rowsum_dP_P); store to dS SMEM tile
//!      at backward_ds_offset for dqdk_accum to consume.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{
    backward_ds_offset, backward_p_offset, backward_rms_strip_offset,
};
use crate::flash_attention_v2::phases::backward::probe::maybe_emit_probe_store;

/// Compute per-row softmax-backward correction `D[i] = sum_d dO[i, d] * O[i, d]`
/// and store to `strip[warp_row]` in SMEM (overwrite).
///
/// Mathematically equivalent to the `sum_c P[i, c] * dP[i, c]` formulation
/// previously built by `emit_rowsum_prepass`, but computed as a single
/// row-wise dot product over `head_dim` — no cross-KV-tile accumulation,
/// no per-tile RMW, no risk of stale/over-accumulated strip values.
///
/// This is the same formula the CPU reference uses
/// (`flash_attention_backward_cpu_gqa` builds `d_corr[i] = dO[i] . O[i]`).
///
/// Invoked ONCE per q_iter (8x for block_q=32) BEFORE the MAIN KV loop —
/// NOT inside a KV loop. Each emission covers 4 rows (one per warp).
/// Out-of-bounds rows (q_global >= seq_len) write 0.
pub fn emit_d_correction(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let head_dim = config.head_dim as u32;
    let corr_off = backward_rms_strip_offset(config);
    assert!(
        head_dim.is_multiple_of(32),
        "emit_d_correction requires head_dim % 32 == 0 (got {head_dim})"
    );
    let d_per_lane = head_dim / 32;

    ptx.push_str(&format!(
        "    // Tier C backward D correction: D[i] = dO[i] . O[i] (q_tile_iter={q_tile_iter}, \
         d_per_lane={d_per_lane})\n"
    ));
    ptx.push_str(&format!(
        "    add.u32 %r0, %warp_id, {}; // warp_row\n", q_tile_iter * 4
    ));
    ptx.push_str("    cvt.u64.u32 %warp_row, %r0;\n");
    ptx.push_str("    add.u64 %rd39, %q_start, %warp_row;\n");
    ptx.push_str("    setp.ge.u64 %p0, %rd39, %rd6;\n");
    ptx.push_str(&format!("    @%p0 bra V2_BWD_D_ZERO_{q_tile_iter};\n"));

    // Base HBM row offset (shared by dO and O, both [B, H, S, D] f16):
    //   (batch_idx*rd5 + head_idx) * seq_len + q_global, * head_dim, * 2 bytes
    ptx.push_str("    mul.lo.u64 %rd30, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd30, %rd30, %rd6;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %rd39;\n");
    ptx.push_str("    mul.lo.u64 %rd30, %rd30, %rd7;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 1;  // * 2 bytes (f16)\n");
    ptx.push_str("    add.u64 %rd31, %rd_bwd_do, %rd30;  // &dO[row, 0]\n");
    ptx.push_str("    add.u64 %rd32, %rd3, %rd30;         // &O[row, 0]\n");

    // Each lane owns d_per_lane contiguous d values starting at lane*d_per_lane.
    ptx.push_str("    mov.f32 %f_rowsum_dP_P, 0f00000000;\n");
    for slice in 0..d_per_lane {
        ptx.push_str("    cvt.u64.u32 %rd33, %lane;\n");
        if d_per_lane > 1 {
            ptx.push_str(&format!("    mul.lo.u64 %rd33, %rd33, {d_per_lane};\n"));
        }
        if slice > 0 {
            ptx.push_str(&format!("    add.u64 %rd33, %rd33, {slice};\n"));
        }
        ptx.push_str("    shl.b64 %rd33, %rd33, 1;  // d * 2 bytes\n");
        // dO[row, d]
        ptx.push_str("    add.u64 %rd34, %rd31, %rd33;\n");
        ptx.push_str("    ld.global.b16 %h0, [%rd34];\n");
        ptx.push_str("    cvt.f32.f16 %f0, %h0;\n");
        // O[row, d]
        ptx.push_str("    add.u64 %rd34, %rd32, %rd33;\n");
        ptx.push_str("    ld.global.b16 %h0, [%rd34];\n");
        ptx.push_str("    cvt.f32.f16 %f1, %h0;\n");
        ptx.push_str("    fma.rn.f32 %f_rowsum_dP_P, %f0, %f1, %f_rowsum_dP_P;\n");
    }

    // Warp-butterfly reduce — every lane ends with the full row sum.
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %f0, %f_rowsum_dP_P, {offset}, 31, 0xFFFFFFFF;\n"
        ));
        ptx.push_str("    add.f32 %f_rowsum_dP_P, %f_rowsum_dP_P, %f0;\n");
    }

    // Lane 0 writes strip[warp_row] with OVERWRITE (not RMW — one-shot).
    ptx.push_str("    setp.eq.u32 %p0, %lane, 0;\n");
    ptx.push_str("    shl.b64 %rd35, %warp_row, 2;\n");
    ptx.push_str(&format!("    add.u64 %rd35, %rd35, {corr_off};\n"));
    ptx.push_str("    add.u64 %rd35, %shmem_base, %rd35;\n");
    ptx.push_str("    @%p0 st.shared.f32 [%rd35], %f_rowsum_dP_P;\n");
    ptx.push_str(&format!("    bra V2_BWD_D_DONE_{q_tile_iter};\n"));

    // OOB rows: write 0.
    ptx.push_str(&format!("V2_BWD_D_ZERO_{q_tile_iter}:\n"));
    ptx.push_str("    setp.eq.u32 %p0, %lane, 0;\n");
    ptx.push_str("    shl.b64 %rd35, %warp_row, 2;\n");
    ptx.push_str(&format!("    add.u64 %rd35, %rd35, {corr_off};\n"));
    ptx.push_str("    add.u64 %rd35, %shmem_base, %rd35;\n");
    ptx.push_str("    mov.f32 %f_rowsum_dP_P, 0f00000000;\n");
    ptx.push_str("    @%p0 st.shared.f32 [%rd35], %f_rowsum_dP_P;\n");
    ptx.push_str(&format!("V2_BWD_D_DONE_{q_tile_iter}:\n"));
}

pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    assert_eq!(
        config.block_kv, 32,
        "T3.3 requires block_kv=32 (got {}); wider tiles land in T3.6+",
        config.block_kv
    );
    let head_dim = config.head_dim as u32;
    let block_kv = config.block_kv as u32;
    let row_stride_bytes = head_dim * 2;  // f16 row stride in Q/K/V SMEM tiles
    let p_offset = backward_p_offset(config);
    let ds_offset = backward_ds_offset(config);

    ptx.push_str(&format!(
        "    // Tier C backward dS compute (q_tile_iter={q_tile_iter})\n"
    ));

    // ── Load row_max / row_sum for this warp's Q row from HBM ─────────────
    ptx.push_str(&format!(
        "    add.u32 %r0, %warp_id, {}; // warp_row\n",
        q_tile_iter * 4
    ));
    ptx.push_str("    cvt.u64.u32 %warp_row, %r0;\n");
    ptx.push_str("    add.u64 %rd39, %q_start, %warp_row;\n");
    ptx.push_str("    setp.ge.u64 %p0, %rd39, %rd6;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_BWD_ROW_STATS_ZERO_{q_tile_iter};\n"
    ));
    ptx.push_str("    mul.lo.u64 %rd30, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd30, %rd30, %rd6;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %q_start;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %warp_row;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 2;  // * 4 bytes (f32)\n");
    ptx.push_str("    add.u64 %rd31, %rd_bwd_row_max, %rd30;\n");
    ptx.push_str("    ld.global.f32 %row_max, [%rd31];\n");
    ptx.push_str("    add.u64 %rd31, %rd_bwd_row_sum, %rd30;\n");
    ptx.push_str("    ld.global.f32 %row_sum, [%rd31];\n");
    ptx.push_str(&format!("    bra V2_BWD_ROW_STATS_DONE_{q_tile_iter};\n"));
    ptx.push_str(&format!("V2_BWD_ROW_STATS_ZERO_{q_tile_iter}:\n"));
    ptx.push_str("    mov.f32 %row_max, 0f7F800000;\n");
    ptx.push_str("    mov.f32 %row_sum, 0f3F800000;\n");
    ptx.push_str(&format!("V2_BWD_ROW_STATS_DONE_{q_tile_iter}:\n"));

    // ── cycle-20 T1 probe slots 0+1: row_max, row_sum ──
    maybe_emit_probe_store(ptx, config, 0, "%row_max", q_tile_iter);
    maybe_emit_probe_store(ptx, config, 1, "%row_sum", q_tile_iter);

    // ── Pass 1: compute S[row, lane], P[row, lane], dP[row, lane] ─────────
    ptx.push_str(&format!("V2_BWD_DP_LOOP_{q_tile_iter}:\n"));

    // S accumulator and inner-d loop.
    ptx.push_str("    mov.f32 %f_P, 0f00000000;  // S accumulator\n");
    // Q row base in SMEM: %q_smem_base + warp_row * row_stride_bytes
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd40, %warp_row, {row_stride_bytes}; // q row byte off\n"
    ));
    ptx.push_str("    add.u64 %rd40, %q_smem_base, %rd40;     // %rd40 = &Q[warp_row, 0]\n");
    // K row base in SMEM: %k_smem_base + lane * row_stride_bytes
    ptx.push_str("    cvt.u64.u32 %rd41, %lane;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd41, %rd41, {row_stride_bytes};\n"
    ));
    ptx.push_str("    add.u64 %rd41, %k_smem_base, %rd41;     // %rd41 = &K[lane, 0]\n");

    ptx.push_str("    mov.u32 %r0, 0;\n");
    ptx.push_str(&format!("V2_BWD_S_INNER_{q_tile_iter}:\n"));
    // Byte offset for this d: d * 2
    ptx.push_str("    cvt.u64.u32 %rd42, %r0;\n");
    ptx.push_str("    shl.b64 %rd42, %rd42, 1;\n");
    // Load Q[warp_row, d] f16 → f32 into %f0
    ptx.push_str("    add.u64 %rd43, %rd40, %rd42;\n");
    ptx.push_str("    ld.shared.b16 %h0, [%rd43];\n");
    ptx.push_str("    cvt.f32.f16 %f0, %h0;\n");
    // Load K[lane, d] f16 → f32 into %f1
    ptx.push_str("    add.u64 %rd43, %rd41, %rd42;\n");
    ptx.push_str("    ld.shared.b16 %h0, [%rd43];\n");
    ptx.push_str("    cvt.f32.f16 %f1, %h0;\n");
    ptx.push_str("    fma.rn.f32 %f_P, %f0, %f1, %f_P;\n");
    ptx.push_str("    add.u32 %r0, %r0, 1;\n");
    ptx.push_str(&format!("    setp.lt.u32 %p0, %r0, {head_dim};\n"));
    ptx.push_str(&format!("    @%p0 bra V2_BWD_S_INNER_{q_tile_iter};\n"));

    // S *= scale  (= 1/sqrt(head_dim))
    ptx.push_str("    mul.f32 %f_P, %f_P, %scale;\n");

    // ── cycle-20 T1 probe slot 2: S_pre_mask (S scaled, pre causal/segment) ──
    maybe_emit_probe_store(ptx, config, 2, "%f_P", q_tile_iter);

    // Mask invalid KV columns first: k_global >= seq_len must behave like
    // a masked score regardless of causal / segment settings.
    ptx.push_str("    cvt.u64.u32 %rd42, %lane;\n");
    ptx.push_str("    add.u64 %rd42, %rd42, %k_start;\n");
    ptx.push_str("    setp.ge.u64 %p1, %rd42, %rd6;\n");

    // Causal mask: if lane > warp_row → S = -INF. When config.segment_masked,
    // also mask if segment_ids[q_global] != segment_ids[k_global]. The helper
    // extends %p1 with a cross-segment OR (mask-convention, matching spec §5.1).
    if config.causal {
        ptx.push_str("    setp.gt.u64 %p0, %rd42, %rd39;\n");
        ptx.push_str("    or.pred %p1, %p1, %p0;\n");
        if config.segment_masked {
            // Synthesize u64 global positions for the helper.
            // %warp_row is u64 (set by ds_compute line above), %q_start is u64.
            // q_global = q_start + warp_row; k_global = k_start + lane
            ptx.push_str("    add.u64 %rd_bw_q_global, %q_start, %warp_row;\n");
            ptx.push_str("    cvt.u64.u32 %rd_bw_k_global, %lane;\n");
            ptx.push_str("    add.u64 %rd_bw_k_global, %rd_bw_k_global, %k_start;\n");
            crate::flash_attention_v2::phases::segment_mask::emit_segment_mask_predicate(
                ptx,
                "%rd_bw_q_global",
                "%rd_bw_k_global",
                "%seg_base",
                crate::pca_segment::SegmentResidency::Shared,
                "%p1",
            );
        }
        ptx.push_str("    mov.f32 %f2, 0fFF800000;    // -INF\n");
        ptx.push_str("    @%p1 mov.f32 %f_P, %f2;\n");
    }

    // P = ex2((S - row_max) * log2e) / row_sum
    ptx.push_str("    sub.f32 %f_P, %f_P, %row_max;\n");
    ptx.push_str("    mul.f32 %f_P, %f_P, %log2e;\n");
    ptx.push_str("    ex2.approx.f32 %f_P, %f_P;\n");
    ptx.push_str("    div.approx.f32 %f_P, %f_P, %row_sum;\n");

    // ── cycle-20 T1 probe slot 3: P (post normalization) ──
    maybe_emit_probe_store(ptx, config, 3, "%f_P", q_tile_iter);

    // Store P[warp_row, lane] to SMEM at backward_p_offset + (warp_row*block_kv + lane)*4.
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd44, %warp_row, {};  // warp_row * block_kv\n",
        block_kv
    ));
    ptx.push_str("    cvt.u64.u32 %rd45, %lane;\n");
    ptx.push_str("    add.u64 %rd44, %rd44, %rd45;\n");
    ptx.push_str("    shl.b64 %rd44, %rd44, 2;  // * 4 bytes (f32)\n");
    ptx.push_str(&format!(
        "    add.u64 %rd44, %rd44, {p_offset};  // + backward_p_offset\n"
    ));
    ptx.push_str("    add.u64 %rd44, %shmem_base, %rd44;\n");
    ptx.push_str("    st.shared.f32 [%rd44], %f_P;\n");

    // ── dP[row, lane] = dO[row, :] · V[lane, :] ────────────────────────────
    ptx.push_str("    setp.ge.u64 %p0, %rd39, %rd6;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_BWD_DP_ZERO_{q_tile_iter};\n"
    ));
    ptx.push_str("    mov.f32 %f_dP, 0f00000000;\n");
    // dO[warp_row, d] base in HBM:
    //   row_idx_bytes (computed into %rd46) =
    //     ((batch_idx*heads + head_idx)*seq + q_start + warp_row) * head_dim * 2
    ptx.push_str("    mul.lo.u64 %rd46, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd46, %rd46, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd46, %rd46, %rd6;\n");
    ptx.push_str("    add.u64 %rd46, %rd46, %q_start;\n");
    ptx.push_str("    add.u64 %rd46, %rd46, %warp_row;\n");
    ptx.push_str("    mul.lo.u64 %rd46, %rd46, %rd7;\n");
    ptx.push_str("    shl.b64 %rd46, %rd46, 1;  // * 2 (f16)\n");
    ptx.push_str("    add.u64 %rd46, %rd_bwd_do, %rd46;  // &dO[warp_row, 0]\n");
    // V[lane, d] base in SMEM: %v_smem_base + lane * row_stride_bytes
    ptx.push_str("    cvt.u64.u32 %rd47, %lane;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd47, %rd47, {row_stride_bytes};\n"
    ));
    ptx.push_str("    add.u64 %rd47, %v_smem_base, %rd47;  // &V[lane, 0]\n");

    ptx.push_str("    mov.u32 %r0, 0;\n");
    ptx.push_str(&format!("V2_BWD_DP_INNER_{q_tile_iter}:\n"));
    ptx.push_str("    cvt.u64.u32 %rd42, %r0;\n");
    ptx.push_str("    shl.b64 %rd42, %rd42, 1;\n");
    // dO[warp_row, d]
    ptx.push_str("    add.u64 %rd43, %rd46, %rd42;\n");
    ptx.push_str("    ld.global.b16 %h0, [%rd43];\n");
    ptx.push_str("    cvt.f32.f16 %f0, %h0;\n");
    // V[lane, d]
    ptx.push_str("    add.u64 %rd43, %rd47, %rd42;\n");
    ptx.push_str("    ld.shared.b16 %h0, [%rd43];\n");
    ptx.push_str("    cvt.f32.f16 %f1, %h0;\n");
    ptx.push_str("    fma.rn.f32 %f_dP, %f0, %f1, %f_dP;\n");
    ptx.push_str("    add.u32 %r0, %r0, 1;\n");
    ptx.push_str(&format!("    setp.lt.u32 %p0, %r0, {head_dim};\n"));
    ptx.push_str(&format!("    @%p0 bra V2_BWD_DP_INNER_{q_tile_iter};\n"));

    ptx.push_str(&format!("    bra V2_BWD_DP_DONE_{q_tile_iter};\n"));
    ptx.push_str(&format!("V2_BWD_DP_ZERO_{q_tile_iter}:\n"));
    ptx.push_str("    mov.f32 %f_dP, 0f00000000;\n");
    ptx.push_str(&format!("V2_BWD_DP_DONE_{q_tile_iter}:\n"));

    // ── cycle-20 T1 probe slot 4: dP (row_max, lane) ──
    maybe_emit_probe_store(ptx, config, 4, "%f_dP", q_tile_iter);

    ptx.push_str("    shl.b64 %rd45, %warp_row, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd45, %rd45, {};\n", backward_rms_strip_offset(config)
    ));
    ptx.push_str("    add.u64 %rd45, %shmem_base, %rd45;\n");
    ptx.push_str("    ld.shared.f32 %f_rowsum_dP_P, [%rd45];\n");

    // ── cycle-20 T1 probe slot 5: rowsum_dP_P (D correction from SMEM strip) ──
    maybe_emit_probe_store(ptx, config, 5, "%f_rowsum_dP_P", q_tile_iter);

    // Ensure P stores are visible to dv_accum (and to anyone reading P later).
    ptx.push_str("    bar.sync 0;  // P tile visible to dv_accum\n");

    // ── Pass 2: dS = P * (dP - rowsum_dP_P), store to backward_ds_offset ──
    ptx.push_str(&format!("V2_BWD_DS_{q_tile_iter}:\n"));
    ptx.push_str("    sub.f32 %f0, %f_dP, %f_rowsum_dP_P;\n");
    ptx.push_str("    mul.f32 %f_dS, %f_P, %f0;\n");

    // ── cycle-20 T1 probe slot 6: raw dS (pre scale multiplication) ──
    maybe_emit_probe_store(ptx, config, 6, "%f_dS", q_tile_iter);

    // SMEM slot: backward_ds_offset + (warp_row * block_kv + lane) * 4
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd44, %warp_row, {};\n",
        block_kv
    ));
    ptx.push_str("    cvt.u64.u32 %rd45, %lane;\n");
    ptx.push_str("    add.u64 %rd44, %rd44, %rd45;\n");
    ptx.push_str("    shl.b64 %rd44, %rd44, 2;\n");
    ptx.push_str(&format!(
        "    add.u64 %rd44, %rd44, {ds_offset};  // + backward_ds_offset\n"
    ));
    ptx.push_str("    add.u64 %rd44, %shmem_base, %rd44;\n");
    ptx.push_str("    st.shared.f32 [%rd44], %f_dS;\n");

    ptx.push_str("    bar.sync 0;  // dS visible to dqdk_accum\n");
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
            gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0, gpu_sm: 75,
            segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: true,
                save_activations_for_backward: true,
                d_model,
                ..CshaExtras::default()
            }),
            checkpoint: None,
        }
    }

    #[test]
    #[allow(non_snake_case)]
    fn backward_ds_compute_recomputes_P_and_computes_dS() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);

        assert!(ptx.contains("%rd_bwd_row_max"));
        assert!(ptx.contains("%rd_bwd_row_sum"));
        assert!(ptx.contains("sub.f32"));
        assert!(ptx.contains("ex2.approx.f32"));
        assert!(ptx.contains("div.approx.f32"));
        assert!(ptx.contains("V2_BWD_DP_LOOP_0:"));
        assert!(ptx.contains("V2_BWD_DS_0:"));
        // Rowsum butterfly reduction lives in `emit_rowsum_prepass` now
        // (pre-KV-loop correction strip); `emit` reads the correction
        // from the SMEM strip rather than re-reducing.
        assert!(ptx.contains("%f_dS"));
        assert!(ptx.contains("%f_rowsum_dP_P"),
            "dS must consume the pre-reduced correction strip");
        // Real SMEM addressing (not placeholder constants):
        assert!(ptx.contains("ld.shared.b16 %h0"),
            "must read Q/K/V from SMEM");
        assert!(ptx.contains("%rd_bwd_do"),
            "dP must read dO from HBM (not placeholder)");
    }

    #[test]
    fn backward_d_correction_applies_butterfly_reduction() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit_d_correction(&mut ptx, &cfg, 0);
        assert!(
            ptx.matches("shfl.sync.bfly.b32").count() >= 5,
            "5-step butterfly reduction over head_dim for D[i] = dO.O"
        );
        // Uses dO (HBM) and O (HBM via %rd3), not K/V SMEM. Must not
        // touch %k_smem_base or %v_smem_base.
        assert!(ptx.contains("%rd_bwd_do"), "must read dO from HBM");
        assert!(ptx.contains("%rd3"), "must read forward O from HBM");
        // OOB branch guard:
        assert!(ptx.contains("V2_BWD_D_ZERO_0:"));
        assert!(ptx.contains("V2_BWD_D_DONE_0:"));
    }

    #[test]
    fn backward_ds_compute_applies_causal_mask_on_recompute() {
        let mut cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        cfg.causal = true;
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert!(ptx.contains("setp.gt"));
        assert!(ptx.contains("0xff800000") || ptx.contains("0fFF800000"));
    }

    #[test]
    fn backward_ds_compute_label_uniqueness_across_iters() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        emit(&mut ptx, &cfg, 1);
        assert!(ptx.contains("V2_BWD_DP_LOOP_0:"));
        assert!(ptx.contains("V2_BWD_DP_LOOP_1:"));
        assert!(ptx.contains("V2_BWD_DS_0:"));
        assert!(ptx.contains("V2_BWD_DS_1:"));
    }
}
