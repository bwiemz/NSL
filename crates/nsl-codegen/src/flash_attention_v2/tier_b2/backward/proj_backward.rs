//! Tier B.2 projection-backward kernel (`tier_b2_proj_backward`) building block.
//!
//! The projection-backward kernel reuses the scalar `emit_dproj` /
//! `emit_drmsnorm` emitters (in `phases/backward/csha_hooks_backward.rs`).
//! Those emitters READ dQ/dK/dV from SMEM as row-major `[block_q, head_dim]`
//! f32 tiles anchored at `backward_d{q,k,v}_offset(config)` off `%shmem_base`.
//!
//! The Tier B.2 dQ/dK/dV kernels, however, write their gradients to HBM as
//! f32 (see `tier_b2/backward/dq.rs` / `dkdv.rs` finalize stores, all
//! `st.global.f32`). This module bridges the two: `emit_dqkv_hbm_to_smem_load`
//! cooperatively loads each HBM gradient buffer into the EXACT SMEM slot that
//! `emit_dproj` subsequently reads, so the scalar dproj/dRMSNorm math runs
//! unchanged against the Tier B.2 gradients.
//!
//! Dtype: f32 throughout. The HBM dQ/dK/dV buffers are f32 (Tier B.2 dQ/dK/dV
//! kernels emit `st.global.f32`) and `emit_dproj` reads f32 from SMEM, so this
//! emitter does a straight `ld.global.f32` -> `st.shared.f32` with no narrowing.
//! It is READ-ONLY on the HBM buffers (loads only; never stores back).

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{
    backward_dk_offset, backward_dq_offset, backward_dv_offset,
};

/// Cooperatively load the three HBM dQ/dK/dV gradient buffers into the SMEM
/// slots that `emit_dproj` reads (`backward_d{q,k,v}_offset`).
///
/// SMEM layout produced (mirrors `emit_dproj`'s read addressing):
///   SMEM[backward_dq_offset + (s*head_dim + j)*4] = dQ[q_start+s, j]   (f32)
///   SMEM[backward_dk_offset + (s*head_dim + j)*4] = dK[q_start+s, j]   (f32)
///   SMEM[backward_dv_offset + (s*head_dim + j)*4] = dV[q_start+s, j]   (f32)
///
/// HBM source layout (mirrors `emit_xnorm_recompute`'s row addressing for the
/// `[head, seq, head_dim]` f32 buffers under the smoke scope):
///   row_global = q_start + s
///   flat       = head_idx*seq + row_global        (seq in %rd6)
///   byte_off   = flat * head_dim * 4 + j*4        (head_dim in %rd7)
///   src        = base_ptr + byte_off
///
/// Work partition (mirrors `emit_dproj`): 128 threads over `block_q*head_dim`
/// cells. Thread owns `cell = tid_x + k*128`, guarded by `cell < block_q*head_dim`;
/// `s = cell / head_dim`, `j = cell % head_dim`.
///
/// ASSUMES the kernel prelude (orchestrator task) has already declared the
/// shared register pool: `%shmem_base`, `%tid_x`, `%q_start`, `%head_idx`,
/// `%rd6` (seq_len), `%rd7` (head_dim), scratch `%rd_c*`, `%f_dy`, predicate
/// `%p_c0`. This emitter does NOT redeclare `.reg`.
pub fn emit_dqkv_hbm_to_smem_load(ptx: &mut String, config: &FlashAttentionConfig) {
    let csha = match config.csha.as_ref() {
        Some(c) => c,
        None => {
            ptx.push_str("    // Tier B.2 proj-backward load: csha=None, no emission\n");
            return;
        }
    };
    // d_model gates the broader proj-backward path; the load itself is keyed on
    // head_dim, but keep the same guard the sibling emitters use for parity.
    if csha.d_model == 0 {
        ptx.push_str("    // Tier B.2 proj-backward load: d_model=0, no emission\n");
        return;
    }

    let head_dim = config.head_dim as u32;
    let block_q = config.block_q as u32;
    if head_dim == 0 || block_q == 0 {
        ptx.push_str("    // Tier B.2 proj-backward load: head_dim/block_q=0, no emission\n");
        return;
    }
    let total_cells = block_q * head_dim;
    let cells_per_thread = total_cells.div_ceil(128).max(1);

    ptx.push_str(
        "    // Tier B.2 proj-backward: stage HBM dQ/dK/dV into emit_dproj SMEM slots (f32).\n",
    );

    for (label, src_reg, dst_off) in [
        ("DQ", "%rd_bwd_dq", backward_dq_offset(config)),
        ("DK", "%rd_bwd_dk", backward_dk_offset(config)),
        ("DV", "%rd_bwd_dv", backward_dv_offset(config)),
    ] {
        ptx.push_str(&format!("V2_PROJBWD_LOAD_{label}:\n"));
        // Null-guard the HBM source pointer; skip the whole buffer if absent.
        ptx.push_str(&format!("    setp.eq.u64 %p_c0, {src_reg}, 0;\n"));
        ptx.push_str(&format!("    @%p_c0 bra V2_PROJBWD_LOAD_{label}_SKIP;\n"));

        for k in 0..cells_per_thread {
            let thread_cell = k * 128;
            // cell = tid_x + k*128
            ptx.push_str("    cvt.u64.u32 %rd_c0, %tid_x;\n");
            if thread_cell > 0 {
                ptx.push_str(&format!("    add.u64 %rd_c0, %rd_c0, {thread_cell};\n"));
            }
            // guard cell < total_cells
            ptx.push_str(&format!("    setp.lt.u64 %p_c0, %rd_c0, {total_cells};\n"));
            ptx.push_str(&format!(
                "    @!%p_c0 bra V2_PROJBWD_LOAD_{label}_CELL_{k}_SKIP;\n"
            ));

            // s = cell / head_dim; j = cell % head_dim
            ptx.push_str(&format!("    div.u64 %rd_c1, %rd_c0, {head_dim};  // s\n"));
            ptx.push_str(&format!("    rem.u64 %rd_c2, %rd_c0, {head_dim};  // j\n"));

            // --- HBM source address (emit_xnorm_recompute convention) ---
            // row_global = q_start + s
            ptx.push_str("    add.u64 %rd_c3, %rd_c1, %q_start;  // row_global = q_start + s\n");
            // flat = head_idx*seq + row_global  (seq in %rd6)
            ptx.push_str("    mul.lo.u64 %rd_c4, %head_idx, %rd6;  // head_idx*seq\n");
            ptx.push_str("    add.u64 %rd_c4, %rd_c4, %rd_c3;  // + row_global\n");
            // elem = flat*head_dim + j   (head_dim in %rd7)
            ptx.push_str("    mul.lo.u64 %rd_c4, %rd_c4, %rd7;  // flat*head_dim\n");
            ptx.push_str("    add.u64 %rd_c4, %rd_c4, %rd_c2;  // + j\n");
            // byte_off = elem*4 (f32)
            ptx.push_str("    shl.b64 %rd_c4, %rd_c4, 2;  // *4 bytes (f32)\n");
            ptx.push_str(&format!(
                "    add.u64 %rd_c4, {src_reg}, %rd_c4;  // HBM src\n"
            ));
            ptx.push_str("    ld.global.f32 %f_dy, [%rd_c4];\n");

            // --- SMEM dest address (emit_dproj layout) ---
            //   dst = %shmem_base + dst_off + (s*head_dim + j)*4
            // smem_elem = s*head_dim + j = (cell), so reuse %rd_c0 (the cell idx)
            // but recompute to keep the layout explicit and independent of cell.
            ptx.push_str(&format!(
                "    mul.lo.u64 %rd_c3, %rd_c1, {head_dim};  // s*head_dim\n"
            ));
            ptx.push_str("    add.u64 %rd_c3, %rd_c3, %rd_c2;  // + j\n");
            ptx.push_str("    shl.b64 %rd_c3, %rd_c3, 2;  // *4 bytes (f32)\n");
            ptx.push_str(&format!(
                "    add.u64 %rd_c3, %rd_c3, {dst_off};  // + dst offset\n"
            ));
            ptx.push_str("    add.u64 %rd_c3, %shmem_base, %rd_c3;  // SMEM dest\n");
            ptx.push_str("    st.shared.f32 [%rd_c3], %f_dy;\n");

            ptx.push_str(&format!("V2_PROJBWD_LOAD_{label}_CELL_{k}_SKIP:\n"));
        }

        ptx.push_str(&format!("V2_PROJBWD_LOAD_{label}_SKIP:\n"));
    }
    ptx.push_str("    bar.sync 0;  // dQ/dK/dV SMEM tiles visible before dproj reads\n");
}
