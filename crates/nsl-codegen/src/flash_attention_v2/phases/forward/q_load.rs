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

        // Optional RoPE cos/sin bases (position-indexed by q_row_global or effective_pos).
        // Only needed when the non-CSHA inline path will fire.
        //
        // PCA sec.4.3 site 1: when segment_masked && rope_q, route the cos/sin
        // row through effective_pos = q_row_global - smem_doc_starts[seg_smem[q_row_global]]
        // instead of the raw q_row_global, implementing per-document RoPE position
        // reset. Bit-identical to the backward T9 sites 3+4 in csha_hooks_backward.rs
        // so dQ flows back correctly through reset positions. Registers declared
        // by the prelude under the same gate (segment_masked && rope_q).
        //
        // KNOWN-LIMITATION (non-CSHA inline path only): when `csha=None` and
        // `rope_q=true`, only Q is rotated here — K reaches s_compute UNROTATED
        // (emit_k_tile_load reads K from HBM and stores f16 to SMEM without any
        // RoPE rotation site). Attention scores Q_rot * K_unrot^T are therefore
        // semantically WRONG on this branch regardless of PCA reset.
        //
        // PRODUCTION IS UNAFFECTED. The CSHA-fused-projections path
        // (csha_hooks.rs::emit_rope_pair_sweep) is the production PCA RoPE path
        // and rotates both Q AND K correctly under the same effective_pos. The
        // production CSHA training-PTX synthesis site
        // (maybe_synthesize_csha_training_ptx) ALWAYS sets
        // csha=Some(CshaExtras{level=1, ...}), and the @flash_attention
        // decorator's inference path uses the same CSHA-fused PTX when RoPE is
        // active. The non-CSHA + rope_q=true branch is reachable only from
        // direct PTX synthesis tests that exercise the fallback emitters
        // (pca_tier_a_forward_correctness::rope_q_forward_*), and those tests
        // pin only no-crash + per-doc reset semantics, NOT full RoPE
        // correctness — see the comment on cpu_reference_rope_then_attention,
        // which explicitly does NOT rotate K so the CPU side matches the
        // kernel side's structural gap byte-for-byte under this branch.
        //
        // Closing this gap requires K-side rotation in emit_k_tile_load: a
        // partner-shuffle of K halves after the cooperative HBM->SMEM load
        // (the 128-threads-per-tile-element load pattern does NOT give a
        // single thread access to BOTH partners of a HalfSplit pair, so the
        // rotation must be a second sync-rotate-sync pass on the SMEM tile,
        // mirroring emit_rope_pair_sweep's design). This is tracked as a
        // separate follow-on PR.
        if config.rope_q && !csha_rope_active {
            ptx.push_str("    // RoPE: cos/sin bases (position = effective_pos when PCA reset active)\n");
            ptx.push_str("    ld.param.u64 %rd25, [cos_ptr];\n");
            ptx.push_str("    ld.param.u64 %rd26, [sin_ptr];\n");

            if config.segment_masked {
                ptx.push_str("    // PCA sec.4.3 site 1: forward Q effective_pos = q_row_global - doc_starts[seg]\n");
                // abs_pos = q_row_global (narrowed to u32; seq_len < 2^31 per pca_segment ceiling).
                ptx.push_str("    cvt.u32.u64 %r_abs_pos, %rd21;\n");
                // sid = seg_smem[abs_pos] (u16 entries; byte offset abs_pos*2 from
                // %seg_base, the cvta.shared.u64 generic address set in the prelude).
                ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_abs_pos, 2;\n");
                ptx.push_str("    cvt.u64.u32 %rd_doc_smem_addr, %r_doc_starts_byte_off;\n");
                ptx.push_str("    add.u64 %rd_doc_smem_addr, %seg_base, %rd_doc_smem_addr;\n");
                ptx.push_str("    ld.shared.u16 %rs_doc_seg, [%rd_doc_smem_addr];\n");
                ptx.push_str("    cvt.u32.u16 %r_doc_starts_idx, %rs_doc_seg;\n");
                // doc_start = smem_doc_starts[sid] (i32, 4 bytes per entry).
                ptx.push_str("    mul.lo.u32 %r_doc_starts_byte_off, %r_doc_starts_idx, 4;\n");
                ptx.push_str("    cvta.shared.u64 %r_doc_smem_base, smem_doc_starts;\n");
                ptx.push_str("    cvt.u64.u32 %rd_doc_smem_addr, %r_doc_starts_byte_off;\n");
                ptx.push_str("    add.u64 %rd_doc_smem_addr, %r_doc_smem_base, %rd_doc_smem_addr;\n");
                ptx.push_str("    ld.shared.s32 %r_doc_start, [%rd_doc_smem_addr];\n");
                // effective_pos = abs_pos - doc_start (s32; non-negative under packing
                // invariants from pca_rope::plan, so the result fits in u32).
                ptx.push_str("    sub.s32 %r_effective_pos_q, %r_abs_pos, %r_doc_start;\n");
                // Zero-extend the (non-negative) effective_pos into the u64 cos/sin row
                // scratch. Use cvt.u64.u32 — NOT cvt.u64.s32 — to match the backward T9
                // sites bit-for-bit (csha_hooks_backward.rs uses u32). If a packing
                // invariant violation ever produced a "negative" doc_start, the s32 form
                // would propagate sign bits while the backward zero-extended; using u32
                // here keeps the forward/backward address computation byte-identical.
                ptx.push_str("    cvt.u64.u32 %rd27, %r_effective_pos_q;\n");
                ptx.push_str(&format!(
                    "    mul.lo.u64 %rd27, %rd27, {};          // effective_pos * head_dim\n",
                    head_dim
                ));
            } else {
                ptx.push_str(&format!(
                    "    mul.lo.u64 %rd27, %rd21, {};          // q_row_global * head_dim\n",
                    head_dim
                ));
            }
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
            // %rd28 = d (lane + slice_idx*32), set by the caller before this call.
            // Compute d*4 once for cos, then recompute for sin (%rd31 was
            // overwritten by the cos add.u64 above, so we must shift again).
            ptx.push_str("    shl.b64 %rd31, %rd28, 2;  // d*4 for cos row\n");
            ptx.push_str("    add.u64 %rd31, %rd25, %rd31;  ld.global.f32 %f0, [%rd31];  // cos\n");
            // Recompute d*4 from %rd28 (still valid) so sin addr = sin_base + d*4.
            ptx.push_str("    shl.b64 %rd31, %rd28, 2;  // d*4 for sin row (recompute)\n");
            ptx.push_str("    add.u64 %rd31, %rd26, %rd31;  ld.global.f32 %f1, [%rd31];  // sin\n");
            ptx.push_str(&format!(
                "    shfl.sync.bfly.b32 %f2, %f{}, 16, 31, 0xFFFFFFFF;  // partner Q\n",
                reg
            ));
            ptx.push_str("    setp.lt.u32 %p0, %lane, 16;\n");
            // Reference math (csha_hooks.rs:1584-1595):
            //   new_x0 = x0*cos - x1*sin
            //   new_x1 = x0*sin + x1*cos
            // HalfSplit: lanes  <16 hold x0 (self=x0, partner=x1);
            //            lanes >=16 hold x1 (self=x1, partner=x0).
            // -- @%p0  branch: new_x0 = self*cos - partner*sin
            //      encode as: t = self*cos + 0 ; new = (-partner)*sin + t
            ptx.push_str("    @%p0  neg.f32 %f3, %f2;                          // -partner (x1)\n");
            ptx.push_str(&format!(
                "    @%p0  fma.rn.f32 %f{r}, %f{r}, %f0, 0f00000000;   // self*cos\n",
                r = reg
            ));
            ptx.push_str(&format!(
                "    @%p0  fma.rn.f32 %f{r}, %f3, %f1, %f{r};          // + (-partner)*sin\n",
                r = reg
            ));
            // -- @!%p0 branch: new_x1 = partner*sin + self*cos
            //      encode as: t = self*cos + 0 ; new = partner*sin + t
            ptx.push_str(&format!(
                "    @!%p0 fma.rn.f32 %f{r}, %f{r}, %f0, 0f00000000;   // self*cos\n",
                r = reg
            ));
            ptx.push_str(&format!(
                "    @!%p0 fma.rn.f32 %f{r}, %f2, %f1, %f{r};          // + partner*sin\n",
                r = reg
            ));
        }
        RopeStyle::Adjacent => {
            ptx.push_str(&format!(
                "    // rope adjacent slice {}: partner = lane^1\n",
                slice_idx
            ));
            // Same fix: recompute d*4 before sin addr.
            ptx.push_str("    shl.b64 %rd31, %rd28, 2;  // d*4 for cos row\n");
            ptx.push_str("    add.u64 %rd31, %rd25, %rd31;  ld.global.f32 %f0, [%rd31];\n");
            ptx.push_str("    shl.b64 %rd31, %rd28, 2;  // d*4 for sin row (recompute)\n");
            ptx.push_str("    add.u64 %rd31, %rd26, %rd31;  ld.global.f32 %f1, [%rd31];\n");
            ptx.push_str(&format!(
                "    shfl.sync.bfly.b32 %f2, %f{}, 1, 31, 0xFFFFFFFF;  // partner Q\n",
                reg
            ));
            // Reference math (csha_hooks.rs:1584-1595):
            //   new_x0 = x0*cos - x1*sin
            //   new_x1 = x0*sin + x1*cos
            // Adjacent: even lanes hold x0 (self=x0, partner=x1);
            //           odd  lanes hold x1 (self=x1, partner=x0).
            // Split via `lane & 1`:
            //   even (p1=true) : new = self*cos - partner*sin
            //   odd  (p1=false): new = partner*sin + self*cos
            ptx.push_str("    and.b32 %r1, %lane, 1;                           // parity bit\n");
            ptx.push_str("    setp.eq.u32 %p1, %r1, 0;                        // p1 <=> even lane\n");
            // Even lane (holds x0): new_x0 = self*cos + (-partner)*sin
            ptx.push_str("    @%p1  neg.f32 %f3, %f2;                          // -partner (x1)\n");
            ptx.push_str(&format!(
                "    @%p1  fma.rn.f32 %f{r}, %f{r}, %f0, 0f00000000;  // self*cos\n",
                r = reg
            ));
            ptx.push_str(&format!(
                "    @%p1  fma.rn.f32 %f{r}, %f3, %f1, %f{r};         // + (-partner)*sin\n",
                r = reg
            ));
            // Odd lane (holds x1): new_x1 = self*cos + partner*sin
            ptx.push_str(&format!(
                "    @!%p1 fma.rn.f32 %f{r}, %f{r}, %f0, 0f00000000;  // self*cos\n",
                r = reg
            ));
            ptx.push_str(&format!(
                "    @!%p1 fma.rn.f32 %f{r}, %f2, %f1, %f{r};         // + partner*sin\n",
                r = reg
            ));
        }
    }
}
