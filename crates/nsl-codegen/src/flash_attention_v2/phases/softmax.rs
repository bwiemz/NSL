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

pub fn emit(ptx: &mut String, config: &FlashAttentionConfig) {
    let block_kv = config.block_kv as u32;
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
            "    add.u64 %rd41, %rd41, {};                 // + sp_offset\n",
            sp_offset(config)
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

    // === Step 2: online update row_max, compute correction ===
    ptx.push_str("    mov.f32 %old_max, %row_max;\n");
    ptx.push_str("    max.f32 %new_max, %row_max, %f0;\n");
    ptx.push_str("    mov.f32 %row_max, %new_max;\n");
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
        ptx.push_str(&format!("    add.u64 %rd41, %rd41, {};\n", sp_offset(config)));
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
    ptx.push_str("    add.f32 %row_sum, %row_sum, %f2;\n");

    ptx.push_str("    bar.sync 0;  // FENCE: all warps done writing P in-place\n");
}
