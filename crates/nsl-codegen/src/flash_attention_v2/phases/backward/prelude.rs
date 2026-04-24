//! Tier C backward-pass prelude.
//!
//! Emits the PTX file header, `.visible .entry`, shared-memory
//! declaration, register pool, and index-compute block for the fused
//! source-AD backward kernel. Subsequent backward phases (q_load,
//! s_recompute, ds_compute, dqk_accum, dv_accum, dwq_dwk_dwv,
//! dx_finalize) are added in T3.2..T3.8.
//!
//! Param block layout: identical forward-facing 35 args so tooling
//! that treats a CSHA kernel as a drop-in stays stable, plus 8
//! backward-specific append pointers:
//!   dO_ptr             — upstream gradient (input)
//!   dq_ptr / dk_ptr / dv_ptr             — per-token gradient outputs
//!   dwq_ptr / dwk_ptr / dwv_ptr          — weight gradient outputs
//!   dx_ptr                               — input-tensor gradient
//!
//! `q_proj_ptr` / `k_proj_ptr` / `v_proj_ptr` / `row_max_ptr` /
//! `row_sum_ptr` are inherited from the forward-facing param list (the
//! forward kernel uses them as save-write targets; the backward kernel
//! reads them as activation inputs).

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{
    backward_extra_bytes, total_bytes, Direction, SMEM_BUDGET_BYTES,
};
use crate::kernel_skeleton::indexing::{
    emit_thread_lane_warp_register_decl, emit_thread_lane_warp_register_init,
};

/// Stable kernel-name prefix for backward variants.
pub fn kernel_name(config: &FlashAttentionConfig) -> String {
    let fw = crate::flash_attention_v2::flash_attention_kernel_name_v2(config);
    // `flash_attn_v2_...` → `flash_attn_backward_v2_...`
    match fw.strip_prefix("flash_attn_") {
        Some(rest) => format!("flash_attn_backward_{}", rest),
        None => format!("flash_attn_backward_{fw}"),
    }
}

/// Total SMEM bytes required by the backward kernel.
///
/// `total_bytes` covers the forward layout (Q/KV tiles + SP/weight/
/// softmax-save regions); `backward_extra_bytes` appends the recomputed
/// P tile plus three f32 gradient accumulators (dQ, dK, dV).
pub fn backward_total_bytes(config: &FlashAttentionConfig) -> u32 {
    total_bytes(config) + backward_extra_bytes(config)
}

fn backward_needs_dynamic_smem(config: &FlashAttentionConfig) -> bool {
    // PCA Tier A: when segment_masked the prelude also emits seg_smem[4096]
    // as a separate static SMEM array.  4096 bytes is enough to push some
    // configs over the 48 KB static cap.  Force dynamic SMEM in that case
    // so the main shmem[] uses the `extern .shared` form and ptxas sees
    // only the 4096-byte seg_smem in the static budget.
    let seg_overhead = if config.segment_masked { 4096 } else { 0 };
    backward_total_bytes(config) + seg_overhead > SMEM_BUDGET_BYTES
}

/// Emit the backward prelude: header, entry, SMEM, registers, indices.
pub fn emit(ptx: &mut String, config: &FlashAttentionConfig) {
    // Caller should have validated `Direction::Backward`; assert here so
    // a misrouted caller can't produce an invalid kernel.
    let _ = Direction::Backward;

    use crate::kernel_skeleton::header::{emit_ptx_header, PtxVersion, TargetSm};
    emit_ptx_header(ptx, PtxVersion::V8_7, TargetSm::Sm75);

    let dyn_smem = backward_needs_dynamic_smem(config);
    if dyn_smem {
        use crate::kernel_skeleton::smem::emit_dynamic_smem_extern;
        emit_dynamic_smem_extern(ptx);
    }

    let name = kernel_name(config);
    ptx.push_str(&format!(".visible .entry {} (\n", name));
    let mut params: Vec<(&str, &str)> = vec![
        // Forward-compatible block (35 args, identical to forward prelude
        // so tooling can treat the backward kernel as a drop-in CSHA
        // variant for name lookup / launch-list validation).
        (".param .u64", "q_ptr"), (".param .u64", "k_ptr"), (".param .u64", "v_ptr"),
        (".param .u64", "out_ptr"), (".param .f32", "scale"),
        (".param .u64", "batch"), (".param .u64", "heads"), (".param .u64", "seq_len"),
        (".param .u64", "head_dim"), (".param .u64", "block_table_ptr"),
        (".param .u64", "k_pool_ptr"), (".param .u64", "v_pool_ptr"),
        (".param .u64", "block_size"), (".param .u64", "cos_ptr"),
        (".param .u64", "sin_ptr"), (".param .u64", "seq_ids_ptr"),
        (".param .u64", "seq_lens_ptr"), (".param .u64", "dfs_enter_ptr"),
        (".param .u64", "dfs_exit_ptr"), (".param .u64", "num_tree_nodes"),
        (".param .u64", "param_logsumexp"),
        (".param .u64", "csha_x_ptr"), (".param .u64", "csha_norm_weight_ptr"),
        (".param .u64", "csha_wq_ptr"), (".param .u64", "csha_wk_ptr"),
        (".param .u64", "csha_wv_ptr"), (".param .u64", "csha_wo_ptr"),
        (".param .f32", "csha_eps"), (".param .u32", "csha_active_heads"),
        (".param .u32", "csha_d_model"),
        (".param .u64", "q_proj_ptr"), (".param .u64", "k_proj_ptr"),
        (".param .u64", "v_proj_ptr"),
        (".param .u64", "row_max_ptr"), (".param .u64", "row_sum_ptr"),
        // Tier C: pre-RMSNorm raw-x save (forward staged it; backward reads it).
        (".param .u64", "x_raw_ptr"),
        // Tier C backward-specific append.
        (".param .u64", "dO_ptr"),
        (".param .u64", "dq_ptr"), (".param .u64", "dk_ptr"), (".param .u64", "dv_ptr"),
        (".param .u64", "dwq_ptr"), (".param .u64", "dwk_ptr"), (".param .u64", "dwv_ptr"),
        (".param .u64", "dx_ptr"),
        // 8th gradient output (Option A of the Gap I.5 fix): `dx_norm_ptr`
        // receives the SMEM `dx_norm` tile (gradient w.r.t. the RMSNorm
        // OUTPUT — i.e. dy_norm). This is what `RmsNormGammaBackward`
        // semantically needs as its `grad` input; previously the AD-side
        // fed `dx_ptr` (which is dx_raw, post-dRMSNorm) and produced
        // incorrect dgamma values when the CSHA dispatcher claim fired.
        (".param .u64", "dx_norm_ptr"),
        // Option A dK/dV f32 scratch pointers (saturation fix). Each
        // q-block launch does serialized f32 ld/add/st RMW into these
        // scratch buffers; a final host-side conversion kernel writes
        // f32 scratch → f16 dk_ptr / dv_ptr so accumulation never
        // saturates at f16 max (65504) like the prior f16-HBM design.
        (".param .u64", "dk_scratch_ptr"),
        (".param .u64", "dv_scratch_ptr"),
    ];
    // PCA Tier A: segment_ids pointer — trailing, only when segment_masked.
    // Kept at the END so the rest of the param layout stays byte-stable
    // for segment_masked=false backward kernels.
    if config.segment_masked {
        params.push((".param .u64", "segment_ids_ptr"));
    }
    for (i, (ty, pname)) in params.iter().enumerate() {
        let comma = if i + 1 < params.len() { "," } else { "" };
        ptx.push_str(&format!("    {} {}{}\n", ty, pname, comma));
    }
    ptx.push_str(")\n{\n");

    if !dyn_smem {
        use crate::kernel_skeleton::smem::emit_static_smem_decl;
        // PCA Tier A: when segment_masked, reserve an extra 4096 bytes at the
        // tail of shmem for the embedded seg_smem (see register-decl comment).
        let seg_overhead: u32 = if config.segment_masked { 4096 } else { 0 };
        emit_static_smem_decl(ptx, (backward_total_bytes(config) + seg_overhead) as usize);
    }

    // Base register pool. f32 pool: 48 scratch + head_dim/32 Q slice +
    // head_dim/32 O-ish placeholder (retained for parity with forward).
    let slices_per_lane = ((config.head_dim as u32) / 32).max(1);
    let f32_pool = 48 + slices_per_lane;
    emit_thread_lane_warp_register_decl(ptx);
    ptx.push_str("    .reg .u64 %rd<64>;\n");
    ptx.push_str(&format!("    .reg .f32 %f<{}>;\n", f32_pool));
    ptx.push_str("    .reg .b16 %h<32>;\n");
    ptx.push_str("    .reg .pred %p<8>;\n");
    ptx.push_str("    .reg .u32 %r<16>;\n");
    ptx.push_str("    .reg .f32 %scale, %log2e;\n");
    ptx.push_str("    .reg .f32 %row_max, %row_sum;\n");
    ptx.push_str("    .reg .u64 %q_start, %q_launch_base, %head_idx, %batch_idx, %k_start, %k_max;\n");
    ptx.push_str("    .reg .u64 %shmem_base, %smem_addr;\n");

    // Tier C backward-specific registers. Per-slice f32 accumulators for
    // gradient columns and softmax-recompute state. Naming mirrors
    // forward's `%f_acc_{label}_{slice}` convention.
    for slice in 0..slices_per_lane {
        ptx.push_str(&format!(
            "    .reg .f32 %f_dq_{slice}, %f_dk_{slice}, %f_dv_{slice};\n"
        ));
        ptx.push_str(&format!(
            "    .reg .f32 %f_q_{slice}, %f_k_{slice}, %f_v_{slice};\n"
        ));
    }
    // Softmax-recompute state registers (scalar per thread per KV tile
    // column; ds/dp/p share the same reduction shape).
    ptx.push_str("    .reg .f32 %f_P, %f_dP, %f_dS;\n");
    ptx.push_str("    .reg .f32 %f_rowsum_dP_P;      // sum_k P[i,k]*dP[i,k] for softmax Jacobian\n");
    // CSHA hook registers (T3.6): dRMSNorm per-dim gradient scratch
    // (g_d = dx_norm * norm_weight).
    ptx.push_str("    .reg .f32 %f_g;                // dRMSNorm per-dim gradient term\n");

    // Orchestrator scratch for cooperative dK + dV SMEM zero-init.
    ptx.push_str("    .reg .u64 %rd_zero_idx;\n");
    ptx.push_str("    .reg .f32 %f_zero_val;\n");
    ptx.push_str("    .reg .pred %p_zero;\n");
    // Finalize scratch for SMEM->HBM cooperative dK + dV copy.
    ptx.push_str("    .reg .u64 %rd_dk_base, %rd_dk_idx, %rd_dk_smem, %rd_dk_hbm;\n");
    ptx.push_str("    .reg .f32 %f_dk_tmp;\n");
    ptx.push_str("    .reg .pred %p_dk;\n");
    // Finalize scratch for SMEM->HBM cooperative dQ copy + per-iter
    // register-to-SMEM flush.
    ptx.push_str("    .reg .u64 %rd_dq_base, %rd_dq_idx, %rd_dq_smem, %rd_dq_hbm;\n");
    ptx.push_str("    .reg .f32 %f_dq_tmp;\n");
    ptx.push_str("    .reg .pred %p_dq_g;\n");
    ptx.push_str("    .reg .u64 %rd_dqs_row, %rd_dqs_col, %rd_dqs_addr;\n");
    // Tier C dproj / dRMSNorm scratch registers.
    ptx.push_str("    .reg .u64 %rd_c0, %rd_c1, %rd_c2, %rd_c3, %rd_c4, %rd_c5;\n");
    ptx.push_str("    .reg .u32 %r_c0, %r_c1, %r_c2, %r_c3;\n");
    ptx.push_str("    .reg .f32 %f_xn, %f_dy, %f_dw, %f_rms_v, %f_rms_inv, %f_ms;\n");
    ptx.push_str("    .reg .f32 %f_nw, %f_xraw, %f_sgrad, %f_dxn, %f_gd, %f_dxv;\n");
    ptx.push_str("    .reg .pred %p_c0, %p_c1;\n");
    ptx.push_str("    .reg .b16 %h_tmp;\n");

    // Backward HBM pointer registers (saved-activation inputs +
    // gradient outputs). Separate name so an unsuspecting forward
    // phase doesn't clobber them.
    ptx.push_str("    .reg .u64 %rd_bwd_q_proj, %rd_bwd_k_proj, %rd_bwd_v_proj;\n");
    ptx.push_str("    .reg .u64 %rd_bwd_row_max, %rd_bwd_row_sum;\n");
    ptx.push_str("    .reg .u64 %rd_bwd_do, %rd_bwd_dq, %rd_bwd_dk, %rd_bwd_dv;\n");
    ptx.push_str("    .reg .u64 %rd_bwd_dwq, %rd_bwd_dwk, %rd_bwd_dwv, %rd_bwd_dx;\n");
    // Gap I.5 fix (Option A): HBM pointer for the 8th gradient output
    // `dx_norm` (gradient w.r.t. the RMSNorm output). Loaded once in
    // prelude so the dRMSNorm Phase 1 SMEM->HBM copy can reference it
    // via a stable register name.
    ptx.push_str("    .reg .u64 %rd_bwd_dxn;\n");
    // Option A dK/dV f32 scratch pointers. Finalize emits
    // `ld.global.f32 / add.f32 / st.global.f32` RMW into these
    // scratch buffers; a host-side conversion kernel writes
    // f32 scratch → f16 dk_ptr / dv_ptr once all q-block launches
    // complete. Avoids the f16-HBM saturation-to-inf that broke the
    // prior cvt.rn.f16.f32 ld-add-st design.
    ptx.push_str("    .reg .u64 %rd_bwd_dk_scratch, %rd_bwd_dv_scratch;\n");

    // SMEM-base pointer + warp_row (shared with forward contract so
    // backward phase emitters can use the same addressing helpers).
    ptx.push_str("    .reg .u64 %q_smem_base, %k_smem_base, %v_smem_base;\n");
    ptx.push_str("    .reg .u64 %warp_row;\n");

    // PCA Tier A backward: segment-mask helper scratch registers + SMEM buffer.
    // Only emitted when segment_masked; segment_masked=false backward kernels
    // remain byte-identical to pre-Task-4A. Mirrors forward prelude declarations
    // (Task 3B), but with two additional backward-specific u64 scratch registers
    // for synthesizing global q/k positions from within-tile indices.
    if config.segment_masked {
        // u64 pair: global segment_ids pointer + SMEM generic-space pointer.
        // %seg_base is u64 (full generic address) to avoid the cvt.u32.u64
        // truncation bug on Blackwell (sm_120+): at shared offset 0 the low 32
        // bits of the generic address are NOT zero, so the u32 truncation gives
        // a wrong shared-space address.  Use the full u64 generic address with
        // ld/st.shared instead.
        ptx.push_str("    .reg .u64 %rd_seg_global, %seg_base;\n");
        // Scratch registers used by segment_mask::emit_segment_mask_predicate.
        // Now u64 (matching seg_base) to avoid mixing address widths.
        ptx.push_str("    .reg .u64 %rd_q_SEGMASK, %rd_k_SEGMASK;\n");
        ptx.push_str("    .reg .b16 %rs_q_SEGMASK, %rs_k_SEGMASK;\n");
        ptx.push_str("    .reg .pred %p_seg_SEGMASK;\n");
        // Cooperative-load scratch for the warp-0 prelude (seq_len <= 2048).
        ptx.push_str("    .reg .u32 %r_pca_i, %r_pca_seq;\n");
        ptx.push_str("    .reg .u64 %rd_pca_off;\n");
        ptx.push_str("    .reg .b16 %rs_pca;\n");
        ptx.push_str("    .reg .pred %p_pca_load, %p_pca_done;\n");
        // Backward-specific: scratch u64 for synthesizing q_global and k_global
        // from within-tile indices for the helper call in ds_compute.
        // %warp_row (u64) and %k_start (u64) are live at ds_compute call sites;
        // these scratch registers receive q_start + warp_row and k_start + lane.
        ptx.push_str("    .reg .u64 %rd_bw_q_global, %rd_bw_k_global;\n");
        // PCA Tier A — seg_smem location: embedded at the TAIL of the extern
        // shmem[] region (`shared_mem_bytes` in cuLaunchKernel includes an
        // extra 4096 bytes past backward_total_bytes so the last 4096 bytes
        // hold segment_ids). Rationale: a separate `.shared .align 4 .b8
        // seg_smem[4096]` alongside `.extern .shared shmem[]` compiles
        // cleanly on ptxas but fires CUDA_ERROR_ILLEGAL_ADDRESS on Blackwell
        // (sm_120, RTX 5070 Ti) at runtime when both coexist — the mixed
        // static+extern layout exposes a driver / hardware addressing issue.
        // Embedding seg_smem inside the single extern shmem allocation
        // sidesteps the mix. No separate `.shared` decl is emitted.
    }

    crate::kernel_skeleton::smem::emit_shmem_base_cvta(ptx);
    ptx.push_str("    mov.f32 %log2e, 0f3FB8AA3B;  // log2(e)\n");

    // Initialise SMEM tile bases. Q tile at byte 0 (Tier A layout); K
    // at kv_offset (Tier A layout). V gets its OWN tile in the backward
    // extras region — if V aliased K's slot (as in forward), the
    // backward's kv_load::emit_v would clobber K before ds_compute can
    // read it, corrupting every downstream gradient. See bug #2 fix.
    let kv_off = crate::flash_attention_v2::smem_layout::kv_offset(config);
    let v_in_off = crate::flash_attention_v2::smem_layout::backward_v_input_offset(config);
    ptx.push_str("    mov.u64 %q_smem_base, %shmem_base;\n");
    ptx.push_str(&format!(
        "    add.u64 %k_smem_base, %shmem_base, {};\n", kv_off
    ));
    ptx.push_str(&format!(
        "    add.u64 %v_smem_base, %shmem_base, {};\n", v_in_off
    ));

    // Scalar param loads.
    ptx.push_str("    ld.param.f32 %scale, [scale];\n");
    ptx.push_str("    ld.param.u64 %rd0, [q_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd1, [k_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd2, [v_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd3, [out_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd4, [batch];\n");
    ptx.push_str("    ld.param.u64 %rd5, [heads];\n");
    ptx.push_str("    ld.param.u64 %rd6, [seq_len];\n");
    ptx.push_str("    ld.param.u64 %rd7, [head_dim];\n");
    // Backward-specific pointer loads (issued eagerly so subsequent
    // phases can reference the registers without re-loading).
    ptx.push_str("    ld.param.u64 %rd_bwd_q_proj, [q_proj_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_k_proj, [k_proj_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_v_proj, [v_proj_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_row_max, [row_max_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_row_sum, [row_sum_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_do,  [dO_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_dq,  [dq_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_dk,  [dk_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_dv,  [dv_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_dwq, [dwq_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_dwk, [dwk_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_dwv, [dwv_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_dx,  [dx_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_dxn, [dx_norm_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_dk_scratch, [dk_scratch_ptr];\n");
    ptx.push_str("    ld.param.u64 %rd_bwd_dv_scratch, [dv_scratch_ptr];\n");
    ptx.push_str("    ld.param.u64 %q_launch_base, [seq_lens_ptr];\n");

    // Thread/block indices — identical to forward.
    emit_thread_lane_warp_register_init(ptx);

    ptx.push_str("    cvt.u64.u32 %q_start, %bid_x;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %q_start, %q_start, {};\n",
        config.block_q
    ));
    ptx.push_str("    add.u64 %q_start, %q_start, %q_launch_base;\n");
    ptx.push_str("    cvt.u64.u32 %rd16, %bid_y;\n");
    ptx.push_str("    rem.u64 %head_idx,  %rd16, %rd5;\n");
    ptx.push_str("    div.u64 %batch_idx, %rd16, %rd5;\n");
    // Backward loads only the first KV tile into SMEM today, so the live
    // tile origin is k_start=0. Segment-masked ds_compute synthesizes
    // k_global = k_start + lane; leaving k_start uninitialized turns the
    // helper's seg_smem address into garbage and can fault on ld.shared.
    ptx.push_str("    mov.u64 %k_start, 0;\n");
    ptx.push_str("    mov.u64 %k_max, %rd6;\n");

    // PCA Tier A: cooperative warp-0 global→shared load of segment_ids.
    // Mirrors forward prelude's PCA_LOAD_LOOP / PCA_LOAD_DONE (Task 3B),
    // with labels renamed BW_PCA_LOAD_* to avoid any risk of label
    // collision (PTX labels are function-scoped so collision is technically
    // impossible when forward+backward live in separate .entry blocks, but
    // explicit prefixing is cheaper than reasoning about it).
    if config.segment_masked {
        ptx.push_str("\n    // --- PCA Tier A: load segment_ids from global to shared ---\n");
        ptx.push_str("    ld.param.u64 %rd_seg_global, [segment_ids_ptr];\n");
        // seg_base = shmem_base + backward_total_bytes — embed seg_smem at
        // the tail of the extern shmem region (see prelude register-decl
        // comment for the Blackwell illegal-address rationale). The launcher
        // passes shared_mem_bytes = backward_total_bytes + 4096, so the last
        // 4096 bytes past bwd_total are reserved for segment_ids.
        ptx.push_str(&format!(
            "    add.u64 %seg_base, %shmem_base, {};  // bwd_total - seg_smem starts here\n",
            backward_total_bytes(config)
        ));
        ptx.push_str("    setp.lt.u32 %p_pca_load, %tid_x, 32;\n");
        ptx.push_str("    @!%p_pca_load bra BW_PCA_LOAD_DONE;\n");
        // seq_len is in %rd6 (u64); narrow to u32 for loop counter.
        ptx.push_str("    cvt.u32.u64 %r_pca_seq, %rd6;\n");
        ptx.push_str("    mov.u32 %r_pca_i, %tid_x;\n");
        ptx.push_str("BW_PCA_LOAD_LOOP:\n");
        ptx.push_str("    setp.ge.u32 %p_pca_done, %r_pca_i, %r_pca_seq;\n");
        ptx.push_str("    @%p_pca_done bra BW_PCA_LOAD_DONE;\n");
        // Global address = rd_seg_global + i * 2  (u16 = 2 bytes)
        ptx.push_str("    cvt.u64.u32 %rd_pca_off, %r_pca_i;\n");
        ptx.push_str("    shl.b64 %rd_pca_off, %rd_pca_off, 1;\n");
        ptx.push_str("    add.u64 %rd_pca_off, %rd_pca_off, %rd_seg_global;\n");
        ptx.push_str("    ld.global.u16 %rs_pca, [%rd_pca_off];\n");
        // Shared address = seg_base + i * 2  (u64 throughout — no truncation)
        ptx.push_str("    cvt.u64.u32 %rd_pca_off, %r_pca_i;\n");
        ptx.push_str("    shl.b64 %rd_pca_off, %rd_pca_off, 1;\n");
        ptx.push_str("    add.u64 %rd_pca_off, %rd_pca_off, %seg_base;\n");
        ptx.push_str("    st.shared.u16 [%rd_pca_off], %rs_pca;\n");
        // Advance by warp size (32) for next stride.
        ptx.push_str("    add.u32 %r_pca_i, %r_pca_i, 32;\n");
        ptx.push_str("    bra BW_PCA_LOAD_LOOP;\n");
        ptx.push_str("BW_PCA_LOAD_DONE:\n");
        // Fence: all threads (including warps 1+) see segment_ids before use.
        ptx.push_str("    bar.sync 0;\n");
        ptx.push_str("    // --- end PCA Tier A segment_ids load ---\n");
    }
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
    fn backward_prelude_declares_gradient_registers() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);

        assert!(ptx.contains(".reg .f32 %f_dq") || ptx.contains(".reg .f32 %f_dq_"));
        assert!(ptx.contains("%f_dk"));
        assert!(ptx.contains("%f_dv"));
        assert!(ptx.contains("%f_P"));
        assert!(ptx.contains("%f_dP"));
        assert!(ptx.contains("%f_dS"));
        assert!(ptx.contains(".visible .entry"));
        assert!(ptx.contains("dO_ptr"));
        assert!(ptx.contains("q_proj_ptr"));
        assert!(ptx.contains("row_max_ptr"));
    }

    #[test]
    fn backward_prelude_kernel_name_has_backward_prefix() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let name = kernel_name(&cfg);
        assert!(
            name.starts_with("flash_attn_backward_"),
            "expected flash_attn_backward_... got {name}"
        );
    }

    #[test]
    fn backward_prelude_emits_rowsum_reduction_register() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let mut ptx = String::new();
        emit(&mut ptx, &cfg);
        assert!(
            ptx.contains("%f_rowsum_dP_P"),
            "softmax-Jacobian reduction register missing"
        );
    }

    #[test]
    fn backward_prelude_smem_includes_backward_extra() {
        let cfg = base_cfg_fused_backward(32, 32, 32, 4, 32);
        let fwd = total_bytes(&cfg);
        let bwd = backward_total_bytes(&cfg);
        assert!(
            bwd > fwd,
            "backward SMEM ({bwd}) must exceed forward ({fwd}) by P+dQ+dK+dV"
        );
        assert_eq!(bwd - fwd, backward_extra_bytes(&cfg));
    }
}
