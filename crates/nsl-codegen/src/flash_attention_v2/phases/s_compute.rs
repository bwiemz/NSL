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
use crate::flash_attention_v2::smem_layout::{kv_offset, sp_offset};

pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let head_dim = config.head_dim as u32;
    let slices   = head_dim / 32;
    let block_kv = config.block_kv as u32;
    let fused = config.csha.as_ref().map_or(false, |c| c.fused_projections);
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
        ptx.push_str(&format!(
            "    fma.rn.f32 %f0, %f{}, %f1, %f0;           // partial += Q*K\n",
            Q_BASE + i
        ));
    }

    // 5-step warp butterfly: every lane ends with the full dot product.
    for offset in [16u32, 8, 4, 2, 1] {
        ptx.push_str(&format!(
            "    shfl.sync.bfly.b32 %shfl_tmp, %f0, {}, 31, 0xFFFFFFFF;\n",
            offset
        ));
        ptx.push_str("    add.f32 %f0, %f0, %shfl_tmp;\n");
    }
    ptx.push_str("    mul.f32 %f0, %f0, %scale;                 // S *= 1/sqrt(d_k)\n");

    // Causal mask: if k_global > q_row_global, S = -inf.
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
        ptx.push_str("    @%p0 mov.f32 %f0, 0fFF800000;             // -inf\n");
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
