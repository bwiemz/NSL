//! Phase 1 — Q load (warp-per-row). Each warp owns one query row per
//! `q_tile_iter`; lanes distribute the `head_dim` slice across 32
//! threads.
//!
//! After this runs, for the warp owning q_row:
//!   %f{Q_BASE + i} on lane L holds Q[q_row, d = L + 32*i]
//!   for i in 0..head_dim/32
//!
//! Q row is ALSO mirrored into shmem[q_offset + q_row_local*head_dim ..
//! +head_dim] as f16, so later phases that need the full row can read
//! from shmem instead of reconstructing via shfl.
//!
//! rope_q: if configured, rotation is applied on the fly before the
//! shmem store. See the TODO below — current sign logic is a known gap
//! tracked for the rope_q test expansion.

use crate::flash_attention::{FlashAttentionConfig, RopeStyle};
use crate::flash_attention_v2::smem_layout::q_offset;

/// Q register base — lane-held Q slice starts at `%f{Q_BASE}`.
pub const Q_BASE: u32 = 32;

pub fn emit(ptx: &mut String, config: &FlashAttentionConfig, q_tile_iter: u32) {
    let head_dim = config.head_dim as u32;
    let slices = head_dim / 32;
    ptx.push_str(&format!(
        "    // Phase 1: Q load, q_tile_iter = {}\n",
        q_tile_iter
    ));

    // When CSHA fused_projections is active, Q has already been projected
    // into Q-SMEM by emit_matmul_projection (and optionally RoPE-rotated
    // by emit_rope_epilogue).  In this path we load Q from SMEM into the
    // Q registers (%f{Q_BASE..}) so s_compute can proceed -- we do NOT
    // reload from the HBM q_ptr, which would overwrite the projected Q.
    let csha_fused_q = config
        .csha
        .as_ref()
        .is_some_and(|c| c.fused_projections);

    // q_row_local = q_tile_iter * 4 + warp_id
    ptx.push_str(&format!(
        "    add.u32 %r0, %warp_id, {};            // q_row_local = warp_id + q_tile_iter*4\n",
        q_tile_iter * 4
    ));
    ptx.push_str("    cvt.u64.u32 %rd20, %r0;                 // q_row_local as u64\n");

    // Q-shmem row base for this warp (used in both paths for storing/loading).
    ptx.push_str(&format!(
        "    mov.u64 %rd23, {};                    // q_offset\n",
        q_offset(config)
    ));
    ptx.push_str(&format!(
        "    mul.lo.u64 %rd24, %rd20, {};          // q_row_local * head_dim\n",
        head_dim
    ));
    ptx.push_str("    shl.b64 %rd24, %rd24, 1;                  // * 2 bytes (f16)\n");
    ptx.push_str("    add.u64 %rd23, %rd23, %rd24;              // shmem row base offset\n");

    if csha_fused_q {
        // Fused-projection path: load Q from SMEM (already projected + RoPE-rotated).
        // Q-SMEM tile starts at shmem_base + q_offset (offset 0 for v2).
        // Layout: [block_q rows × head_dim f16].  This warp's row is at offset
        //   (warp_id + q_tile_iter*4) * head_dim * 2.
        // Lane l owns column l (for slices_per_lane=1, head_dim=32).
        ptx.push_str("    // fused_projections: Q already in SMEM; load registers from SMEM\n");
        for i in 0..slices {
            ptx.push_str(&format!("    // slice {} from SMEM\n", i));
            ptx.push_str("    cvt.u64.u32 %rd28, %lane;\n");
            if i > 0 {
                ptx.push_str(&format!("    add.u64 %rd28, %rd28, {};\n", i * 32));
            }
            // SMEM address = shmem_base + q_offset + warp_row * head_dim * 2 + lane * 2
            ptx.push_str("    shl.b64 %rd29, %rd28, 1;  // d * 2 (f16)\n");
            ptx.push_str("    add.u64 %smem_addr, %rd23, %rd29;\n");
            ptx.push_str("    add.u64 %smem_addr, %smem_addr, %shmem_base;\n");
            ptx.push_str("    ld.shared.b16 %h0, [%smem_addr];\n");
            ptx.push_str(&format!(
                "    cvt.f32.f16 %f{}, %h0;  // Q[warp_row, d] f16 to f32\n",
                Q_BASE + i
            ));
        }
    } else {
        // Classic path: load Q from HBM q_ptr.
        ptx.push_str("    add.u64 %rd21, %q_start, %rd20;          // q_row_global\n");

        // Q-base global address: q_ptr + (batch*heads*seq_len*head_dim
        //                                 + head_idx*seq_len*head_dim
        //                                 + q_row_global*head_dim) * 4 bytes
        ptx.push_str("    mul.lo.u64 %rd22, %batch_idx, %rd5;      // batch*heads\n");
        ptx.push_str("    add.u64 %rd22, %rd22, %head_idx;         // + head_idx\n");
        ptx.push_str("    mul.lo.u64 %rd22, %rd22, %rd6;            // * seq_len\n");
        ptx.push_str("    add.u64 %rd22, %rd22, %rd21;              // + q_row_global\n");
        ptx.push_str("    mul.lo.u64 %rd22, %rd22, %rd7;            // * head_dim\n");
        ptx.push_str("    shl.b64 %rd22, %rd22, 2;                  // * 4 bytes (f32 source)\n");
        ptx.push_str("    add.u64 %rd22, %rd0, %rd22;               // q_base global\n");

        // When CSHA fused_projections is active, emit_rope_epilogue (called
        // immediately after emit_matmul_projection in mod.rs) already rotates
        // the Q/K SMEM tiles before this q_load path runs.  Skip the inline
        // RoPE branch here to prevent double-rotation.
        let csha_rope_active = config.csha.as_ref().is_some_and(|c| c.fused_projections);

        // Optional RoPE cos/sin bases (position-indexed by q_row_global).
        // Only needed when the non-CSHA inline path will fire.
        if config.rope_q && !csha_rope_active {
            ptx.push_str("    // RoPE: cos/sin bases for q_row_global\n");
            ptx.push_str("    ld.param.u64 %rd25, [cos_ptr];\n");
            ptx.push_str("    ld.param.u64 %rd26, [sin_ptr];\n");
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd27, %rd21, {};          // q_row_global * head_dim\n",
                head_dim
            ));
            ptx.push_str("    shl.b64 %rd27, %rd27, 2;                  // * 4 bytes\n");
            ptx.push_str("    add.u64 %rd25, %rd25, %rd27;              // cos row base\n");
            ptx.push_str("    add.u64 %rd26, %rd26, %rd27;              // sin row base\n");
        }

        // Per-slice load + optional rotate + f16 shmem store.
        for i in 0..slices {
            ptx.push_str(&format!(
                "    // slice {}: d = lane + 32*{} = lane + {}\n",
                i, i, i * 32
            ));
            ptx.push_str("    cvt.u64.u32 %rd28, %lane;\n");
            ptx.push_str(&format!("    add.u64 %rd28, %rd28, {};\n", i * 32));
            ptx.push_str("    shl.b64 %rd29, %rd28, 2;                  // * 4 bytes f32\n");
            ptx.push_str("    add.u64 %rd29, %rd22, %rd29;              // q_base + d*4\n");
            ptx.push_str(&format!(
                "    ld.global.f32 %f{}, [%rd29];\n",
                Q_BASE + i
            ));

            // Inline RoPE rotation: only when rope_q=true AND CSHA has not
            // already rotated Q via emit_rope_epilogue (prevents double-rotation).
            if config.rope_q && !csha_rope_active {
                emit_rope_rotation_inline(ptx, Q_BASE + i, i, config.rope_style);
            }

            // Store into shmem as f16.
            ptx.push_str(&format!("    cvt.rn.f16.f32 %h0, %f{};\n", Q_BASE + i));
            ptx.push_str("    shl.b64 %rd30, %rd28, 1;                  // d * 2 bytes (f16)\n");
            ptx.push_str("    add.u64 %smem_addr, %rd23, %rd30;         // shmem dest\n");
            ptx.push_str("    add.u64 %smem_addr, %smem_addr, %shmem_base;\n");
            ptx.push_str("    st.shared.b16 [%smem_addr], %h0;\n");
        }
    }

    ptx.push_str("    bar.sync 0;  // FENCE: all warps finish Q shmem store\n");
}

// TODO(fa-v2): RoPE register aliasing (%rd30/%rd31 overlap with the
// q_load store-address math) and sign-flip correctness for HalfSplit
// style both need to be revisited when a rope_q=true config enters the
// test matrix. Current implementation is placeholder-correct for
// rope_q=false (not called) and will be exercised once Task 13's sweep
// or a dedicated rope test lands.
#[allow(dead_code)]
fn emit_rope_rotation_inline(
    ptx: &mut String,
    reg: u32,
    slice_idx: u32,
    style: RopeStyle,
) {
    match style {
        RopeStyle::HalfSplit => {
            ptx.push_str(&format!(
                "    // rope halfsplit slice {}: pair across (lane ^ 16)\n",
                slice_idx
            ));
            ptx.push_str("    shl.b64 %rd31, %rd28, 2;  // d*4 for f32 cos/sin row\n");
            ptx.push_str("    add.u64 %rd31, %rd25, %rd31;  ld.global.f32 %f0, [%rd31];  // cos\n");
            ptx.push_str("    add.u64 %rd31, %rd26, %rd31;  ld.global.f32 %f1, [%rd31];  // sin\n");
            ptx.push_str(&format!(
                "    shfl.sync.bfly.b32 %f2, %f{}, 16, 31, 0xFFFFFFFF;  // partner Q\n",
                reg
            ));
            ptx.push_str("    setp.lt.u32 %p0, %lane, 16;\n");
            ptx.push_str(&format!("    @%p0  fma.rn.f32 %f{}, %f{}, %f0, %f1;\n", reg, reg));
            ptx.push_str(&format!("    @!%p0 fma.rn.f32 %f{}, %f{}, %f0, %f1;\n", reg, reg));
        }
        RopeStyle::Adjacent => {
            ptx.push_str(&format!(
                "    // rope adjacent slice {}: partner = lane^1\n",
                slice_idx
            ));
            ptx.push_str("    shl.b64 %rd31, %rd28, 2;\n");
            ptx.push_str("    add.u64 %rd31, %rd25, %rd31;  ld.global.f32 %f0, [%rd31];\n");
            ptx.push_str("    add.u64 %rd31, %rd26, %rd31;  ld.global.f32 %f1, [%rd31];\n");
            ptx.push_str(&format!(
                "    shfl.sync.bfly.b32 %f2, %f{}, 1, 31, 0xFFFFFFFF;\n",
                reg
            ));
            ptx.push_str(&format!("    fma.rn.f32 %f{}, %f{}, %f0, %f1;\n", reg, reg));
        }
    }
}
