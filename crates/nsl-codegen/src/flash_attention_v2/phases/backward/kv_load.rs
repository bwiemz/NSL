//! Tier C backward K and V loads: cooperative HBM→SMEM of the saved
//! post-RoPE K_proj and V_proj activations (forward T1.3 wrote them).
//!
//! Mirrors `phases/backward/q_load.rs` but reads from `k_proj_ptr`
//! into `%k_smem_base` and from `v_proj_ptr` into `%v_smem_base`.
//! K and V are NOT rotated here — saved activations are already
//! post-RoPE. No null-guard bypass: if the saves are null, these
//! functions short-circuit to a labelled skip (backward is only
//! meaningful when saves are present).
//!
//! Unlike the forward-side K pre-pass which covers only block_q rows,
//! the backward path needs the full block_kv KV tile. Each warp still
//! owns one row, so with 4 warps per block we iterate
//! `ceil(block_kv / 4)` KV-rows per warp.

use crate::flash_attention::FlashAttentionConfig;

/// Emit a cooperative HBM→SMEM load for one of K_proj or V_proj.
///
/// `tag` is "K" or "V" (label namespace); `ptr_reg` is the backward
/// prelude's pointer register (`%rd_bwd_k_proj` or `%rd_bwd_v_proj`);
/// `smem_base` is `%k_smem_base` or `%v_smem_base`.
fn emit_one(
    ptx: &mut String,
    config: &FlashAttentionConfig,
    tag: &str,
    ptr_reg: &str,
    smem_base: &str,
) {
    let head_dim = config.head_dim as u32;
    let block_kv = config.block_kv as u32;
    let slices_per_lane = (head_dim / 32).max(1);
    // Each warp owns ceil(block_kv / 4) rows of this tile.
    let rows_per_warp = block_kv.div_ceil(4);

    ptx.push_str(&format!(
        "    // Tier C backward {tag}-load (block_kv={block_kv}, \
         slices/lane={slices_per_lane}, rows/warp={rows_per_warp})\n"
    ));
    ptx.push_str(&format!("V2_BWD_{tag}_LOAD:\n"));
    // Null-guard.
    ptx.push_str(&format!("    setp.eq.u64 %p0, {ptr_reg}, 0;\n"));
    ptx.push_str(&format!("    @%p0 bra V2_BWD_{tag}_LOAD_SKIP;\n"));

    for r in 0..rows_per_warp {
        // kv_row = warp_id + r*4
        ptx.push_str(&format!(
            "    add.u32 %r0, %warp_id, {}; // kv_row\n",
            r * 4
        ));
        // row_idx (global HBM) = batch_idx*(heads*seq) + head_idx*seq + kv_row
        ptx.push_str("    cvt.u64.u32 %rd30, %r0;\n");
        ptx.push_str("    mul.lo.u64 %rd31, %batch_idx, %rd5;\n");
        ptx.push_str("    add.u64 %rd31, %rd31, %head_idx;\n");
        ptx.push_str("    mul.lo.u64 %rd31, %rd31, %rd6;\n");
        ptx.push_str("    add.u64 %rd31, %rd31, %rd30;\n");
        // * head_dim * 2 bytes (f16)
        ptx.push_str("    mul.lo.u64 %rd31, %rd31, %rd7;\n");
        ptx.push_str("    shl.b64 %rd31, %rd31, 1;\n");

        for slice in 0..slices_per_lane {
            // col = lane * slices + slice
            ptx.push_str("    cvt.u64.u32 %rd32, %lane;\n");
            if slices_per_lane > 1 {
                ptx.push_str(&format!(
                    "    mul.lo.u64 %rd32, %rd32, {slices_per_lane};\n"
                ));
            }
            if slice > 0 {
                ptx.push_str(&format!("    add.u64 %rd32, %rd32, {slice};\n"));
            }
            ptx.push_str("    shl.b64 %rd32, %rd32, 1;  // col * 2\n");
            // HBM addr = ptr_base + row_byte_off + col*2
            ptx.push_str("    add.u64 %rd33, %rd31, %rd32;\n");
            ptx.push_str(&format!("    add.u64 %rd33, {ptr_reg}, %rd33;\n"));
            ptx.push_str("    ld.global.b16 %h0, [%rd33];\n");
            // SMEM addr = smem_base + kv_row*(head_dim*2) + col*2
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd34, %rd30, {};\n",
                head_dim * 2
            ));
            ptx.push_str("    add.u64 %rd34, %rd34, %rd32;\n");
            ptx.push_str(&format!("    add.u64 %rd34, {smem_base}, %rd34;\n"));
            ptx.push_str("    st.shared.b16 [%rd34], %h0;\n");
        }
    }

    ptx.push_str(&format!("V2_BWD_{tag}_LOAD_SKIP:\n"));
    ptx.push_str("    bar.sync 0;  // tile visible to all threads\n");
}

/// Load saved K_proj from HBM into the K SMEM tile.
pub fn emit_k(ptx: &mut String, config: &FlashAttentionConfig) {
    emit_one(ptx, config, "K", "%rd_bwd_k_proj", "%k_smem_base");
}

/// Load saved V_proj from HBM into the V SMEM tile.
pub fn emit_v(ptx: &mut String, config: &FlashAttentionConfig) {
    emit_one(ptx, config, "V", "%rd_bwd_v_proj", "%v_smem_base");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn base_cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false, gpu_sm: 75,
            csha: Some(CshaExtras {
                fused_projections: true,
                save_activations_for_backward: true,
                d_model: 32,
                ..CshaExtras::default()
            }),
        }
    }

    #[test]
    fn backward_k_load_reads_from_k_proj_ptr() {
        let cfg = base_cfg();
        let mut ptx = String::new();
        emit_k(&mut ptx, &cfg);
        assert!(ptx.contains("%rd_bwd_k_proj"));
        assert!(ptx.contains("ld.global.b16"));
        assert!(ptx.contains("st.shared.b16"));
        assert!(ptx.contains("V2_BWD_K_LOAD:"));
        assert!(ptx.contains("V2_BWD_K_LOAD_SKIP:"));
        assert!(!ptx.contains("cos_ptr"),
            "backward k_load must use saved post-RoPE K");
    }

    #[test]
    fn backward_v_load_reads_from_v_proj_ptr() {
        let cfg = base_cfg();
        let mut ptx = String::new();
        emit_v(&mut ptx, &cfg);
        assert!(ptx.contains("%rd_bwd_v_proj"));
        assert!(ptx.contains("ld.global.b16"));
        assert!(ptx.contains("st.shared.b16"));
        assert!(ptx.contains("V2_BWD_V_LOAD:"));
    }
}
