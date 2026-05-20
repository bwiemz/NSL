//! Phase 5 - O_acc += P . V (lane-local d, scalar P broadcast per k).
//!
//! V tile shares shmem with K tile (same region at kv_offset, loaded
//! after Phase 3's bar.sync by the orchestrator). For each k: broadcast
//! P[q_row, k] from shmem_S[warp_id, k] (scalar -- all 32 lanes see the
//! same value), each lane reads V[k, d=L+32*i] for its d-slices, does
//! fma into O_acc[i]. No cross-lane reduction.
//!
//! Preconditions:
//!   %f{O_BASE..O_BASE + head_dim/32 - 1}: current O_acc (reset to 0 by
//!       orchestrator at the start of each q_tile_iter)
//!   %correction: softmax correction factor from Phase 3, <= 1
//!   shmem KV region at kv_offset(config) loaded with V tile (f16)
//!   shmem_S[warp_id, :] contains P (f32) values from Phase 3
//!   %k_start, %k_max: current tile
//!
//! Postcondition: %f{O_BASE+i} holds the running O_acc contribution for
//! d = lane + 32*i. Finalize phase divides by %row_sum and stores out.

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{kv_offset, sp_offset};

/// O_acc register base - lane-held O slice starts at `%f{O_BASE}`.
/// Sized so Q (Q_BASE=32) and O (O_BASE=48) don't overlap even for
/// head_dim=256 (Q uses %f32..%f39, O uses %f48..%f55).
pub const O_BASE: u32 = 48;

pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let head_dim = config.head_dim as u32;
    let slices = head_dim / 32;
    let block_kv = config.block_kv as u32;
    let fused = config.csha.as_ref().is_some_and(|c| c.fused_projections);
    // SP slice base offset for this q_tile_iter (same logic as softmax.rs).
    let sp_iter_offset = if fused {
        sp_offset(config) + q_tile_iter * 4 * block_kv * 4
    } else {
        sp_offset(config)
    };

    ptx.push_str(&format!(
        "    // Phase 5: O_acc += P * V (q_tile_iter = {})\n",
        q_tile_iter
    ));

    // Rescale O_acc by the softmax correction factor from Phase 3.
    // This is the online-softmax rebase step: scale the accumulated O
    // contributions from previous tiles to the new row_max baseline.
    for i in 0..slices {
        ptx.push_str(&format!(
            "    mul.f32 %f{}, %f{}, %correction;\n",
            O_BASE + i,
            O_BASE + i
        ));
    }

    // Loop over k in 0..block_kv.
    ptx.push_str("    mov.u32 %r5, 0;                           // k = 0\n");
    ptx.push_str(&format!(
        "    mov.u32 %r6, {};                           // block_kv\n",
        block_kv
    ));
    ptx.push_str(&format!("V2_LOOP_PV_OVER_K_{}:\n", q_tile_iter));

    // Load P[k] from shmem_S[warp_id, k] - scalar (all lanes read same address).
    ptx.push_str("    cvt.u64.u32 %rd42, %warp_id;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd42, %rd42, {};              // warp_id * block_kv\n",
        block_kv
    ));
    ptx.push_str("    cvt.u64.u32 %rd43, %r5;\n");
    ptx.push_str("    add.u64 %rd42, %rd42, %rd43;              // + k\n");
    ptx.push_str("    shl.b64 %rd42, %rd42, 2;                  // * 4 bytes\n");
    ptx.push_str(&format!(
        "    add.u64 %rd42, %rd42, {};                 // + sp_iter_offset\n",
        sp_iter_offset
    ));
    ptx.push_str("    add.u64 %smem_addr, %rd42, %shmem_base;\n");
    ptx.push_str("    ld.shared.f32 %f0, [%smem_addr];          // P[k] scalar\n");

    // V row base in shmem: kv_offset + k*head_dim*2 (f16 bytes).
    ptx.push_str("    cvt.u64.u32 %rd44, %r5;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd44, %rd44, {};              // k * head_dim\n",
        head_dim
    ));
    ptx.push_str("    shl.b64 %rd44, %rd44, 1;                  // * 2 bytes f16\n");
    ptx.push_str(&format!(
        "    add.u64 %rd44, %rd44, {};\n",
        kv_offset(config)
    ));
    ptx.push_str("    add.u64 %rd44, %rd44, %shmem_base;\n");

    // Per-slice V load + fma into O_acc.
    for i in 0..slices {
        ptx.push_str("    cvt.u64.u32 %rd45, %lane;\n");
        if i > 0 {
            ptx.push_str(&format!("    add.u64 %rd45, %rd45, {};\n", i * 32));
        }
        ptx.push_str("    shl.b64 %rd45, %rd45, 1;                  // * 2 bytes f16\n");
        ptx.push_str("    add.u64 %smem_addr, %rd44, %rd45;\n");
        ptx.push_str("    ld.shared.b16 %h0, [%smem_addr];\n");
        ptx.push_str("    cvt.f32.f16 %f1, %h0;                     // V[k, d]\n");
        ptx.push_str(&format!(
            "    fma.rn.f32 %f{}, %f0, %f1, %f{};          // O_acc[i] += P*V\n",
            O_BASE + i,
            O_BASE + i
        ));
    }

    ptx.push_str("    add.u32 %r5, %r5, 1;\n");
    ptx.push_str("    setp.lt.u32 %p0, %r5, %r6;\n");
    ptx.push_str(&format!(
        "    @%p0 bra V2_LOOP_PV_OVER_K_{};\n",
        q_tile_iter
    ));
}
