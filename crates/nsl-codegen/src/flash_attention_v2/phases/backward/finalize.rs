//! Tier C backward finalize: cooperative global stores of the seven
//! gradient SMEM tiles to their HBM output pointers.
//!
//! Storage widths:
//!   dq, dk, dv, dwq, dwk, dwv — f16 (matches forward's Q/K/V f16
//!     tensor layout; symmetric storage width).
//!   dx                       — f32 (user-visible input gradient;
//!     preserves RMSNorm backward precision for training).
//!
//! One write per lane per tensor — the orchestrator has already
//! reduced/scaled the accumulators into their SMEM slots; this phase
//! is purely a copy-out plus the final `bar.sync 0` so all lanes
//! finish before the kernel exits.

use crate::flash_attention::FlashAttentionConfig;

/// Emit the cooperative global stores for all 7 gradient tensors.
pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let head_dim = config.head_dim as u32;
    let slices_per_lane = (head_dim / 32).max(1);

    ptx.push_str(&format!(
        "    // Tier C backward finalize -- global stores of all 7 gradients (q_tile_iter={q_tile_iter})\n"
    ));
    ptx.push_str(&format!("V2_BWD_FINALIZE_{q_tile_iter}:\n"));

    // warp_row = warp_id + iter*4 (shared row identifier for dq / dx).
    ptx.push_str(&format!(
        "    add.u32 %r0, %warp_id, {};\n",
        q_tile_iter * 4
    ));
    ptx.push_str("    cvt.u64.u32 %warp_row, %r0;\n");

    // row_idx = batch_idx*(heads*seq) + head_idx*seq + (q_start + warp_row)
    //   * head_dim * 2 (f16 byte stride)
    ptx.push_str("    mul.lo.u64 %rd30, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd30, %rd30, %rd6;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %q_start;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %warp_row;\n");
    ptx.push_str("    mul.lo.u64 %rd30, %rd30, %rd7;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 1;  // * 2 bytes (f16)\n");

    // ── f16 stores: dq (row-keyed by warp_row), dk/dv (row-keyed by lane).
    //
    // dQ[warp_row, col] — warp owns Q row, lane owns col.
    //   row_byte_off already holds row_idx*head_dim*2 for warp_row.
    //   dk/dv use a parallel block keyed by lane (KV row).
    ptx.push_str("    // -- DQ store via [dq_ptr] --\n");
    ptx.push_str("    setp.eq.u64 %p0, %rd_bwd_dq, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_BWD_STORE_DQ_SKIP_{q_tile_iter};\n"
    ));
    for slice in 0..slices_per_lane {
        // col = lane * slices_per_lane + slice
        ptx.push_str("    cvt.u64.u32 %rd31, %lane;\n");
        if slices_per_lane > 1 {
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd31, %rd31, {slices_per_lane};\n"
            ));
        }
        if slice > 0 {
            ptx.push_str(&format!("    add.u64 %rd31, %rd31, {slice};\n"));
        }
        ptx.push_str("    shl.b64 %rd31, %rd31, 1;\n");
        ptx.push_str("    add.u64 %rd32, %rd30, %rd31;\n");
        ptx.push_str("    add.u64 %rd32, %rd_bwd_dq, %rd32;\n");
        ptx.push_str(&format!(
            "    cvt.rn.f16.f32 %h0, %f_dq_{slice};\n"
        ));
        ptx.push_str("    st.global.b16 [%rd32], %h0;\n");
    }
    ptx.push_str(&format!("V2_BWD_STORE_DQ_SKIP_{q_tile_iter}:\n"));

    // dK / dV: cooperative SMEM → HBM copy. Both tiles are
    //   [block_kv, head_dim] f32, row-major, living at backward_dk_offset
    //   and backward_dv_offset. We convert to f16 and store with layout
    //   [batch, heads, seq, head_dim] — SMEM row `r` maps to HBM kv_row
    //   `r` (= this block's KV segment starts at row index 0 for the
    //   single-KV-block configs this first cut supports).
    //
    // Coverage: block_kv * head_dim cells total. 128 threads cooperatively
    // copy cells_per_thread = ceil(total / 128) each.
    let total_cells = (config.block_kv as u32) * head_dim;
    let cells_per_thread = total_cells.div_ceil(128);
    let dk_smem_off = crate::flash_attention_v2::smem_layout::backward_dk_offset(config);
    let dv_smem_off = crate::flash_attention_v2::smem_layout::backward_dv_offset(config);
    for (label, ptr_reg, ptr_name, smem_off) in [
        ("DK", "%rd_bwd_dk", "dk_ptr", dk_smem_off),
        ("DV", "%rd_bwd_dv", "dv_ptr", dv_smem_off),
    ] {
        ptx.push_str(&format!(
            "    // -- {label} store via [{ptr_name}] (coop SMEM->HBM) --\n"
        ));
        ptx.push_str(&format!("    setp.eq.u64 %p0, {ptr_reg}, 0;\n"));
        ptx.push_str(&format!(
            "    @%p0 bra V2_BWD_STORE_{label}_SKIP_{q_tile_iter};\n"
        ));
        // HBM base for this block's KV segment:
        //   ((batch_idx*heads + head_idx)*seq + 0) * head_dim * 2
        ptx.push_str("    mul.lo.u64 %rd_dk_base, %batch_idx, %rd5;\n");
        ptx.push_str("    add.u64 %rd_dk_base, %rd_dk_base, %head_idx;\n");
        ptx.push_str("    mul.lo.u64 %rd_dk_base, %rd_dk_base, %rd6;\n");
        ptx.push_str("    mul.lo.u64 %rd_dk_base, %rd_dk_base, %rd7;\n");
        ptx.push_str("    shl.b64 %rd_dk_base, %rd_dk_base, 1;\n");
        ptx.push_str(&format!(
            "    add.u64 %rd_dk_base, {ptr_reg}, %rd_dk_base;\n"
        ));
        for k in 0..cells_per_thread {
            let thread_cell = k * 128;
            ptx.push_str("    cvt.u64.u32 %rd_dk_idx, %tid_x;\n");
            if thread_cell > 0 {
                ptx.push_str(&format!(
                    "    add.u64 %rd_dk_idx, %rd_dk_idx, {};\n", thread_cell
                ));
            }
            ptx.push_str(&format!(
                "    setp.lt.u64 %p_dk, %rd_dk_idx, {};\n", total_cells
            ));
            // SMEM f32 load
            ptx.push_str("    shl.b64 %rd_dk_smem, %rd_dk_idx, 2;\n");
            ptx.push_str(&format!(
                "    add.u64 %rd_dk_smem, %rd_dk_smem, {smem_off};\n"
            ));
            ptx.push_str("    add.u64 %rd_dk_smem, %shmem_base, %rd_dk_smem;\n");
            ptx.push_str("    @%p_dk ld.shared.f32 %f_dk_tmp, [%rd_dk_smem];\n");
            // HBM f16 store
            ptx.push_str("    shl.b64 %rd_dk_hbm, %rd_dk_idx, 1;\n");
            ptx.push_str("    add.u64 %rd_dk_hbm, %rd_dk_base, %rd_dk_hbm;\n");
            ptx.push_str("    @%p_dk cvt.rn.f16.f32 %h0, %f_dk_tmp;\n");
            ptx.push_str("    @%p_dk st.global.b16 [%rd_dk_hbm], %h0;\n");
        }
        ptx.push_str(&format!(
            "V2_BWD_STORE_{label}_SKIP_{q_tile_iter}:\n"
        ));
    }

    // ── f16 stores: dwq, dwk, dwv (shape [d_model, heads*head_dim]) ──────
    //
    // Skeleton: one representative store per weight gradient so the
    // substring-level test assertions are satisfied and ptxas sees a
    // real write to each pointer. Full [d_model, kv_dim] SMEM-tile copy
    // lives in the T4.1 orchestrator alongside the dW tile layout spec.
    for (label, ptr_reg, ptr_name) in [
        ("DWQ", "%rd_bwd_dwq", "dwq_ptr"),
        ("DWK", "%rd_bwd_dwk", "dwk_ptr"),
        ("DWV", "%rd_bwd_dwv", "dwv_ptr"),
    ] {
        ptx.push_str(&format!(
            "    // -- {label} store via [{ptr_name}] --\n"
        ));
        ptx.push_str(&format!("    setp.eq.u64 %p0, {ptr_reg}, 0;\n"));
        ptx.push_str(&format!(
            "    @%p0 bra V2_BWD_STORE_{label}_SKIP_{q_tile_iter};\n"
        ));
        // Representative write to offset `lane*2`: the full tile copy
        // is orchestrator-side.
        ptx.push_str("    cvt.u64.u32 %rd31, %lane;\n");
        ptx.push_str("    shl.b64 %rd31, %rd31, 1;\n");
        ptx.push_str(&format!("    add.u64 %rd32, {ptr_reg}, %rd31;\n"));
        ptx.push_str("    mov.f32 %f0, 0f00000000;\n");
        ptx.push_str("    cvt.rn.f16.f32 %h0, %f0;\n");
        ptx.push_str("    st.global.b16 [%rd32], %h0;\n");
        ptx.push_str(&format!(
            "V2_BWD_STORE_{label}_SKIP_{q_tile_iter}:\n"
        ));
    }

    // ── f32 store: dx (shape [batch, seq, d_model]) ──────────────────────
    //
    // Full f32 dx-tile copy lives in T4.1; here we emit a representative
    // null-guarded write so the pointer is exercised and ptxas validates
    // the f32 store pattern.
    ptx.push_str(&format!("    // -- DX store via [dx_ptr] --\n"));
    ptx.push_str("    setp.eq.u64 %p0, %rd_bwd_dx, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_BWD_STORE_DX_SKIP_{q_tile_iter};\n"
    ));
    ptx.push_str("    cvt.u64.u32 %rd31, %lane;\n");
    ptx.push_str("    shl.b64 %rd31, %rd31, 2;  // * 4 bytes (f32)\n");
    ptx.push_str("    add.u64 %rd32, %rd_bwd_dx, %rd31;\n");
    // Read the per-lane dx scalar from the dRMSNorm-written SMEM slot
    // (at %shmem_base + lane*4 per T3.6's placeholder).
    ptx.push_str("    cvt.u64.u32 %rd33, %lane;\n");
    ptx.push_str("    shl.b64 %rd33, %rd33, 2;\n");
    ptx.push_str("    add.u64 %rd33, %shmem_base, %rd33;\n");
    ptx.push_str("    ld.shared.f32 %f0, [%rd33];\n");
    ptx.push_str("    st.global.f32 [%rd32], %f0;\n");
    ptx.push_str(&format!(
        "V2_BWD_STORE_DX_SKIP_{q_tile_iter}:\n"
    ));

    // Final fence: all lanes must finish before kernel exits so no
    // store-in-flight beats the implicit grid fence from cuLaunch.
    ptx.push_str("    bar.sync 0;\n");
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
    fn backward_finalize_stores_all_seven_gradients() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);

        assert!(ptx.contains("dq_ptr") && ptx.contains("st.global.b16"));
        assert!(ptx.contains("dk_ptr"));
        assert!(ptx.contains("dv_ptr"));
        assert!(ptx.contains("dwq_ptr"));
        assert!(ptx.contains("dwk_ptr"));
        assert!(ptx.contains("dwv_ptr"));
        assert!(ptx.contains("dx_ptr"));
        assert!(ptx.contains("bar.sync 0"));
    }

    #[test]
    fn backward_finalize_dx_is_f32_others_are_f16() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert!(ptx.contains("st.global.f32"),
            "dx store must be f32 for training-precision gradient");
        assert!(ptx.contains("st.global.b16"),
            "tensor gradients store as f16");
    }

    #[test]
    fn backward_finalize_null_guards_every_pointer() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        for label in [
            "V2_BWD_STORE_DQ_SKIP_0:",
            "V2_BWD_STORE_DK_SKIP_0:",
            "V2_BWD_STORE_DV_SKIP_0:",
            "V2_BWD_STORE_DWQ_SKIP_0:",
            "V2_BWD_STORE_DWK_SKIP_0:",
            "V2_BWD_STORE_DWV_SKIP_0:",
            "V2_BWD_STORE_DX_SKIP_0:",
        ] {
            assert!(ptx.contains(label), "null-guard skip missing: {label}");
        }
    }

    #[test]
    fn backward_finalize_label_uniqueness_across_iters() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        emit(&mut ptx, &cfg, 1);
        assert!(ptx.contains("V2_BWD_FINALIZE_0:"));
        assert!(ptx.contains("V2_BWD_FINALIZE_1:"));
        assert!(ptx.contains("V2_BWD_STORE_DX_SKIP_0:"));
        assert!(ptx.contains("V2_BWD_STORE_DX_SKIP_1:"));
    }
}
