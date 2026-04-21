//! Phase 3 - online softmax + in-place P writeback.
//!
//! Each warp operates on its own row at shmem_S[warp_id, :]. All 32
//! lanes cooperate via shfl butterfly reductions. After this phase, the
//! same shmem region holds P = exp(S - row_max) values (un-normalised;
//! final divide by row_sum is deferred to Phase 6 finalize). row_max,
//! row_sum, correction are warp-local per-lane state registers
//! (identical value on every lane after the reductions).

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::sp_offset;

pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let block_kv = config.block_kv as u32;
    let fused = config.csha.as_ref().map_or(false, |c| c.fused_projections);
    // J-A3 softmax-internal capture: when save_activations is on, snapshot
    // the three decisive softmax-state registers to dedicated scratch regs
    // (`%f_sdx_fmax` post-butterfly-max, `%f_sdx_nmax` post-online-update,
    // `%f_sdx_fsum` post-butterfly-sum). The save path reads these under
    // `NSL_CSHA_DUMP_SAVE_STATE=fmax|newmax|fsum` to distinguish "softmax
    // computed wrong value" from "ptxas corrupted %row_max between softmax
    // and save." For multi-tile KV loops the final tile's values win — fine
    // for the toy.nsl (seq=32, block_kv=32, one-tile) case we're debugging.
    let capture = config
        .csha
        .as_ref()
        .map_or(false, |c| c.save_activations_for_backward);
    // SP slice base offset for this q_tile_iter.  In the fused split-loop
    // design, all S-passes run before all PV-accums, so each q_tile_iter's P
    // values must live in a distinct SP slice.  In the standard interleaved
    // path q_tile_iter is always 0 here (one pass per iter), so the extra
    // offset is always 0 for the non-fused path.
    let sp_iter_offset = if fused {
        // SP layout: [iters, 4_warps, block_kv] f32.
        // Slice for q_tile_iter starts at: sp_offset + q_tile_iter * 4 * block_kv * 4.
        sp_offset(config) + q_tile_iter * 4 * block_kv * 4
    } else {
        sp_offset(config)
    };
    // block_kv is always a multiple of 32 in the supported matrix (16, 32, 64, 128).
    // For block_kv=16 we have a partial chunk; the predicated load handles it.
    let chunks = (block_kv + 31) / 32;

    ptx.push_str("    // Phase 3: online softmax + P writeback\n");

    // === Step 1: compute local max across lane-strided chunks ===
    ptx.push_str("    mov.f32 %f0, 0fFF800000;                  // local_max = -inf\n");
    for chunk in 0..chunks {
        ptx.push_str(&format!(
            "    // row_max chunk {}: lane sees k = lane + 32*{}\n",
            chunk, chunk
        ));
        ptx.push_str("    cvt.u64.u32 %rd40, %warp_id;\n");
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd40, %rd40, {};              // warp_id * block_kv\n",
            block_kv
        ));
        ptx.push_str("    cvt.u64.u32 %rd41, %lane;\n");
        if chunk > 0 {
            ptx.push_str(&format!("    add.u64 %rd41, %rd41, {};\n", chunk * 32));
        }
        ptx.push_str(&format!(
            "    setp.lt.u64 %p0, %rd41, {};                 // k < block_kv?\n",
            block_kv
        ));
        ptx.push_str("    add.u64 %rd41, %rd41, %rd40;              // warp_base + k\n");
        ptx.push_str("    shl.b64 %rd41, %rd41, 2;                  // * 4 bytes\n");
        ptx.push_str(&format!(
            "    add.u64 %rd41, %rd41, {};                 // + sp_iter_offset\n",
            sp_iter_offset
        ));
        ptx.push_str("    add.u64 %smem_addr, %rd41, %shmem_base;\n");
        ptx.push_str("    mov.f32 %f1, 0fFF800000;\n");
        ptx.push_str("    @%p0 ld.shared.f32 %f1, [%smem_addr];     // S[k] or -inf\n");
        ptx.push_str("    max.f32 %f0, %f0, %f1;\n");
    }

    // Warp butterfly max.
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %shfl_tmp, %f0, {}, 31, 0xFFFFFFFF;\n",
            offset
        ));
        ptx.push_str("    max.f32 %f0, %f0, %shfl_tmp;\n");
    }
    // J-A3 capture: post-butterfly reduced max, before online update.
    if capture {
        ptx.push_str("    mov.f32 %f_sdx_fmax, %f0;  // J-A3 post-butterfly max\n");
    }

    // === Step 2: online update row_max, compute correction ===
    ptx.push_str("    mov.f32 %old_max, %row_max;\n");
    ptx.push_str("    max.f32 %new_max, %row_max, %f0;\n");
    ptx.push_str("    mov.f32 %row_max, %new_max;\n");
    // J-A3 capture: post-online-update, the value that becomes %row_max.
    if capture {
        ptx.push_str("    mov.f32 %f_sdx_nmax, %new_max;  // J-A3 post-online-update max\n");
    }
    ptx.push_str("    sub.f32 %f0, %old_max, %new_max;\n");
    ptx.push_str("    mul.f32 %f0, %f0, %log2e;\n");
    ptx.push_str("    ex2.approx.f32 %correction, %f0;          // exp(old-new), <=1\n");
    ptx.push_str("    mul.f32 %row_sum, %row_sum, %correction;\n");

    // === Step 3: compute P[k] = exp(S[k] - new_max), writeback, sum ===
    ptx.push_str("    mov.f32 %f2, 0f00000000;                  // partial_sum = 0\n");
    for chunk in 0..chunks {
        ptx.push_str(&format!(
            "    // P writeback chunk {}\n",
            chunk
        ));
        ptx.push_str("    cvt.u64.u32 %rd40, %warp_id;\n");
        ptx.push_str(&format!(
            "    mul.lo.u64 %rd40, %rd40, {};\n",
            block_kv
        ));
        ptx.push_str("    cvt.u64.u32 %rd41, %lane;\n");
        if chunk > 0 {
            ptx.push_str(&format!("    add.u64 %rd41, %rd41, {};\n", chunk * 32));
        }
        ptx.push_str(&format!(
            "    setp.lt.u64 %p0, %rd41, {};\n",
            block_kv
        ));
        ptx.push_str("    add.u64 %rd41, %rd41, %rd40;\n");
        ptx.push_str("    shl.b64 %rd41, %rd41, 2;\n");
        ptx.push_str(&format!("    add.u64 %rd41, %rd41, {};\n", sp_iter_offset));
        ptx.push_str("    add.u64 %smem_addr, %rd41, %shmem_base;\n");
        ptx.push_str("    mov.f32 %f1, 0fFF800000;\n");
        ptx.push_str("    @%p0 ld.shared.f32 %f1, [%smem_addr];\n");
        ptx.push_str("    sub.f32 %f1, %f1, %new_max;\n");
        ptx.push_str("    mul.f32 %f1, %f1, %log2e;\n");
        ptx.push_str("    ex2.approx.f32 %f1, %f1;                  // P = exp(S-new_max)\n");
        // Zero P for out-of-range k so it does not pollute sum or later P*V.
        ptx.push_str("    @!%p0 mov.f32 %f1, 0f00000000;\n");
        // Writeback (in-range only, to avoid wild stores).
        ptx.push_str("    @%p0 st.shared.f32 [%smem_addr], %f1;     // in-place P writeback\n");
        ptx.push_str("    add.f32 %f2, %f2, %f1;\n");
    }

    // Warp butterfly sum.
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %shfl_tmp, %f2, {}, 31, 0xFFFFFFFF;\n",
            offset
        ));
        ptx.push_str("    add.f32 %f2, %f2, %shfl_tmp;\n");
    }
    // J-A3 capture: post-butterfly sum, before online row_sum update.
    if capture {
        ptx.push_str("    mov.f32 %f_sdx_fsum, %f2;  // J-A3 post-butterfly sum\n");
    }
    ptx.push_str("    add.f32 %row_sum, %row_sum, %f2;\n");

    ptx.push_str("    bar.sync 0;  // FENCE: all warps done writing P in-place\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, RopeStyle};

    fn cfg_with_save(save: bool) -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 32, block_kv: 32, head_dim: 32,
            causal: false, paged: false, rope_q: false,
            rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
            tree_mask: false, gpu_sm: 75, segment_masked: false,
            csha: Some(CshaExtras {
                fused_projections: false,
                save_activations_for_backward: save,
                d_model: 128,
                ..CshaExtras::default()
            }),
        }
    }

    /// J-A3: when save_activations is on, softmax emits three capture `mov`s
    /// at the three decisive points (post-butterfly-max, post-online-update,
    /// post-butterfly-sum) into dedicated scratch regs.
    #[test]
    fn softmax_emits_j_a3_captures_when_save_activations_on() {
        let cfg = cfg_with_save(true);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert!(
            ptx.contains("mov.f32 %f_sdx_fmax, %f0;"),
            "post-butterfly-max capture missing: {ptx}"
        );
        assert!(
            ptx.contains("mov.f32 %f_sdx_nmax, %new_max;"),
            "post-online-update capture missing: {ptx}"
        );
        assert!(
            ptx.contains("mov.f32 %f_sdx_fsum, %f2;"),
            "post-butterfly-sum capture missing: {ptx}"
        );
    }

    /// J-A3: when save_activations is off, no captures emitted — the scratch
    /// regs aren't declared in prelude for non-save builds.
    #[test]
    fn softmax_skips_j_a3_captures_when_save_activations_off() {
        let cfg = cfg_with_save(false);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        assert!(
            !ptx.contains("%f_sdx_fmax"),
            "fmax capture must not fire without save flag"
        );
        assert!(
            !ptx.contains("%f_sdx_nmax"),
            "nmax capture must not fire without save flag"
        );
        assert!(
            !ptx.contains("%f_sdx_fsum"),
            "fsum capture must not fire without save flag"
        );
    }

    /// J-A3: captures appear in the correct ORDER — fmax before nmax before
    /// fsum. Anchors each to its nearest expected predecessor instruction.
    #[test]
    fn softmax_j_a3_captures_in_correct_order() {
        let cfg = cfg_with_save(true);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg, 0);
        let fmax_pos = ptx.find("mov.f32 %f_sdx_fmax, %f0;").unwrap();
        let nmax_pos = ptx.find("mov.f32 %f_sdx_nmax, %new_max;").unwrap();
        let fsum_pos = ptx.find("mov.f32 %f_sdx_fsum, %f2;").unwrap();
        assert!(fmax_pos < nmax_pos, "fmax must capture before nmax");
        assert!(nmax_pos < fsum_pos, "nmax must capture before fsum");

        // fmax follows the butterfly-max sequence (last `max.f32 %f0, %f0, %shfl_tmp`).
        let before_fmax = &ptx[..fmax_pos];
        assert!(
            before_fmax.contains("max.f32 %f0, %f0, %shfl_tmp;"),
            "fmax capture must come after butterfly max"
        );

        // nmax follows the `mov.f32 %row_max, %new_max` commit.
        let between = &ptx[fmax_pos..nmax_pos];
        assert!(
            between.contains("mov.f32 %row_max, %new_max;"),
            "nmax capture must come after online update commits %row_max"
        );

        // fsum follows the butterfly-sum sequence (last `add.f32 %f2, %f2, %shfl_tmp`).
        let between2 = &ptx[nmax_pos..fsum_pos];
        assert!(
            between2.contains("add.f32 %f2, %f2, %shfl_tmp;"),
            "fsum capture must come after butterfly sum"
        );
    }
}
