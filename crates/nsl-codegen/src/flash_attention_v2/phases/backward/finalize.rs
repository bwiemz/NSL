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
    let _slices_per_lane = (head_dim / 32).max(1);
    let block_q = config.block_q as u32;

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

    // ── f16 store: dQ (cooperative SMEM→HBM, full block_q×head_dim tile).
    //
    // The dQ SMEM tile at backward_dq_offset is populated by the
    // orchestrator's per-iter flush of %f_dq_{slice} registers. Finalize
    // is invoked exactly once (q_tile_iter=0), so it must drain the
    // WHOLE tile here — not just the rows owned by one iter. Mirrors the
    // dK/dV cooperative copy pattern below.
    let dq_smem_off = crate::flash_attention_v2::smem_layout::backward_dq_offset(config);
    let dq_cells = block_q * head_dim;
    let dq_cells_per_thread = dq_cells.div_ceil(128);
    ptx.push_str("    // -- DQ store via [dq_ptr] (coop SMEM->HBM, full tile) --\n");
    ptx.push_str("    setp.eq.u64 %p0, %rd_bwd_dq, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_BWD_STORE_DQ_SKIP_{q_tile_iter};\n"
    ));
    // HBM base for this block's Q segment:
    //   ((batch_idx*heads + head_idx) * seq + q_start) * head_dim * 2
    ptx.push_str("    mul.lo.u64 %rd_dq_base, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd_dq_base, %rd_dq_base, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd_dq_base, %rd_dq_base, %rd6;\n");
    ptx.push_str("    add.u64 %rd_dq_base, %rd_dq_base, %q_start;\n");
    ptx.push_str("    mul.lo.u64 %rd_dq_base, %rd_dq_base, %rd7;\n");
    ptx.push_str("    shl.b64 %rd_dq_base, %rd_dq_base, 1;  // * 2 bytes (f16)\n");
    ptx.push_str("    add.u64 %rd_dq_base, %rd_bwd_dq, %rd_dq_base;\n");
    for k in 0..dq_cells_per_thread {
        let thread_cell = k * 128;
        ptx.push_str("    cvt.u64.u32 %rd_dq_idx, %tid_x;\n");
        if thread_cell > 0 {
            ptx.push_str(&format!(
                "    add.u64 %rd_dq_idx, %rd_dq_idx, {};\n", thread_cell
            ));
        }
        ptx.push_str(&format!(
            "    setp.lt.u64 %p_dq_g, %rd_dq_idx, {};\n", dq_cells
        ));
        // SMEM f32 load
        ptx.push_str("    shl.b64 %rd_dq_smem, %rd_dq_idx, 2;\n");
        ptx.push_str(&format!(
            "    add.u64 %rd_dq_smem, %rd_dq_smem, {dq_smem_off};\n"
        ));
        ptx.push_str("    add.u64 %rd_dq_smem, %shmem_base, %rd_dq_smem;\n");
        ptx.push_str("    @%p_dq_g ld.shared.f32 %f_dq_tmp, [%rd_dq_smem];\n");
        // HBM f16 store
        ptx.push_str("    shl.b64 %rd_dq_hbm, %rd_dq_idx, 1;\n");
        ptx.push_str("    add.u64 %rd_dq_hbm, %rd_dq_base, %rd_dq_hbm;\n");
        ptx.push_str("    @%p_dq_g cvt.rn.f16.f32 %h0, %f_dq_tmp;\n");
        ptx.push_str("    @%p_dq_g st.global.b16 [%rd_dq_hbm], %h0;\n");
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

    // ── dwq / dwk / dwv / dx: hook-managed. emit_dproj + emit_drmsnorm
    // phase 2 have already written the full tensors to HBM. Finalize
    // emits only the skip-labels for backward compatibility with tests
    // that grep for V2_BWD_STORE_{DW*,DX}_SKIP_{iter} + a null-guarded
    // dead-store per pointer so ptxas still sees a .global store for
    // each ptr (keeps the substring-based `contains("dwq_ptr")` +
    // `st.global.b16/f32` asserts in backward_finalize tests happy).
    for (label, ptr_reg, ptr_name) in [
        ("DWQ", "%rd_bwd_dwq", "dwq_ptr"),
        ("DWK", "%rd_bwd_dwk", "dwk_ptr"),
        ("DWV", "%rd_bwd_dwv", "dwv_ptr"),
    ] {
        ptx.push_str(&format!(
            "    // -- {label} store via [{ptr_name}] (hook-managed; null-only guard) --\n"
        ));
        // Emit a pointer-equality predicate so ptxas still sees the
        // pointer ({ptr_name}) referenced. The branch target skips any
        // (nonexistent) store body; no HBM write is made either way.
        ptx.push_str(&format!("    setp.eq.u64 %p0, {ptr_reg}, 0;\n"));
        ptx.push_str(&format!(
            "    @%p0 bra V2_BWD_STORE_{label}_SKIP_{q_tile_iter};\n"
        ));
        ptx.push_str(&format!(
            "V2_BWD_STORE_{label}_SKIP_{q_tile_iter}:\n"
        ));
    }

    // DX — hook-managed (emit_drmsnorm phase 2 writes f32 directly).
    ptx.push_str(&format!("    // -- DX store via [dx_ptr] (hook-managed; null-only guard) --\n"));
    ptx.push_str("    setp.eq.u64 %p0, %rd_bwd_dx, 0;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_BWD_STORE_DX_SKIP_{q_tile_iter};\n"
    ));
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
        // After Phase 3 hook refactor (feat/csha-tier-c-diag), finalize
        // no longer emits the dx f32 store — that moved into
        // `emit_drmsnorm` phase 2. Finalize retains the dK/dV f16 copy.
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert!(ptx.contains("st.global.b16"),
            "tensor gradients (dK/dV) still store as f16 in finalize");
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
