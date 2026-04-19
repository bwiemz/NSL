//! Tier C backward q_load: cooperative HBM→SMEM load of the saved
//! post-RoPE Q activation tile.
//!
//! Reads from `q_proj_ptr` (populated by the forward kernel's T1.3
//! save path), NOT from `q_ptr` and NOT by recomputing the fused
//! projection. The saved tensor is already post-RoPE, so this phase
//! never touches `cos_ptr` / `sin_ptr`.
//!
//! Layout: `[batch, heads, seq, head_dim]` f16 row-major. Each warp
//! owns one row; each lane owns `head_dim/32` consecutive columns —
//! same warp-per-row contract as forward.
//!
//! SMEM destination: `%q_smem_base` (Tier A's Q-tile region at byte 0).
//! Backward-specific tiles (P recompute, dQ/dK/dV accumulators) live
//! AFTER the forward layout per T2.1's `backward_extra_bytes`, so no
//! collision.

use crate::flash_attention::FlashAttentionConfig;

/// Emit the cooperative load of `Q_proj[batch, head, q_start+warp_row, :]`
/// from HBM into the Q SMEM tile. Null-guards on `q_proj_ptr` so a
/// caller passing a null pointer (e.g. a diagnostic launch with
/// forward-only saves disabled) doesn't dereference garbage.
pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let head_dim = config.head_dim as u32;
    let slices_per_lane = (head_dim / 32).max(1);

    ptx.push_str(&format!(
        "    // Tier C backward Q-load (q_tile_iter={q_tile_iter}, slices/lane={slices_per_lane})\n"
    ));
    ptx.push_str(&format!("V2_BWD_Q_LOAD_{q_tile_iter}:\n"));

    // Initialise %q_smem_base = %shmem_base (Q tile occupies byte 0).
    // Idempotent: subsequent phases reinit as needed.
    ptx.push_str("    mov.u64 %q_smem_base, %shmem_base;\n");

    // Null-guard on q_proj_ptr (register loaded in prelude as %rd_bwd_q_proj).
    ptx.push_str("    setp.eq.u64 %p0, %rd_bwd_q_proj, 0;\n");
    ptx.push_str(&format!("    @%p0 bra V2_BWD_Q_LOAD_SKIP_{q_tile_iter};\n"));

    // warp_row = warp_id + iter*4 (u64 for address arithmetic).
    ptx.push_str(&format!(
        "    add.u32 %r0, %warp_id, {}; // warp_row = warp_id + iter*4\n",
        q_tile_iter * 4
    ));
    ptx.push_str("    cvt.u64.u32 %warp_row, %r0;\n");

    // row_idx = batch_idx*(heads*seq) + head_idx*seq + (q_start + warp_row)
    //   %rd5 = heads, %rd6 = seq_len, %rd7 = head_dim (from prelude).
    ptx.push_str("    mul.lo.u64 %rd30, %batch_idx, %rd5;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %head_idx;\n");
    ptx.push_str("    mul.lo.u64 %rd30, %rd30, %rd6;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %q_start;\n");
    ptx.push_str("    add.u64 %rd30, %rd30, %warp_row;\n");
    // row byte-offset base: row_idx * head_dim * 2  (f16 = 2 bytes)
    ptx.push_str("    mul.lo.u64 %rd30, %rd30, %rd7;\n");
    ptx.push_str("    shl.b64 %rd30, %rd30, 1;\n");

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
        // col byte-offset: col * 2
        ptx.push_str("    shl.b64 %rd31, %rd31, 1;\n");

        // HBM address = q_proj_base + row_byte_off + col_byte_off
        ptx.push_str("    add.u64 %rd32, %rd30, %rd31;\n");
        ptx.push_str("    add.u64 %rd32, %rd_bwd_q_proj, %rd32;\n");
        ptx.push_str("    ld.global.b16 %h0, [%rd32];\n");

        // SMEM address = q_smem_base + (warp_row*head_dim + col) * 2
        // Compute fresh without reusing %rd30 (row byte-off is HBM-scaled
        // by head_dim*2 already; SMEM row stride is head_dim*2, same
        // value, so re-use the row portion WITHOUT the batch offset).
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd33, %warp_row, {};\n",
            head_dim * 2
        ));
        ptx.push_str("    add.u64 %rd33, %rd33, %rd31;\n");
        ptx.push_str("    add.u64 %rd33, %q_smem_base, %rd33;\n");
        ptx.push_str("    st.shared.b16 [%rd33], %h0;\n");
    }

    ptx.push_str(&format!("V2_BWD_Q_LOAD_SKIP_{q_tile_iter}:\n"));
    ptx.push_str("    bar.sync 0;  // Q tile visible to all threads\n");
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
            segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: true,
                save_activations_for_backward: true,
                d_model,
                ..CshaExtras::default()
            }),
        }
    }

    #[test]
    fn backward_q_load_reads_from_q_proj_ptr() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);

        assert!(ptx.contains("%rd_bwd_q_proj"),
            "must reference prelude-loaded q_proj_ptr register");
        assert!(ptx.contains("ld.global.b16"),
            "f16 HBM load missing");
        assert!(ptx.contains("st.shared.b16"),
            "SMEM write missing");
        assert!(ptx.contains("V2_BWD_Q_LOAD_0:"),
            "per-iter label missing");
        assert!(!ptx.contains("cos_ptr"),
            "backward q_load must use saved post-RoPE Q (no RoPE recompute)");
        assert!(!ptx.contains("sin_ptr"),
            "backward q_load must use saved post-RoPE Q (no RoPE recompute)");
    }

    #[test]
    fn backward_q_load_label_uniqueness_across_iters() {
        let mut cfg = base_cfg_fused_backward(64, 64, 32, 4, 32);
        cfg.csha = Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 32,
            ..CshaExtras::default()
        });
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        emit(&mut ptx, &cfg, 1);
        assert!(ptx.contains("V2_BWD_Q_LOAD_0:"));
        assert!(ptx.contains("V2_BWD_Q_LOAD_1:"));
        assert!(ptx.contains("V2_BWD_Q_LOAD_SKIP_0:"));
        assert!(ptx.contains("V2_BWD_Q_LOAD_SKIP_1:"));
    }

    #[test]
    fn backward_q_load_null_guard_on_q_proj_ptr() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert!(ptx.contains("setp.eq.u64") && ptx.contains("%rd_bwd_q_proj, 0"),
            "null-guard on q_proj_ptr missing");
    }
}
