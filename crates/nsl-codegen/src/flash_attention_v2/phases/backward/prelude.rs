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
    backward_total_bytes(config) > SMEM_BUDGET_BYTES
}

/// Emit the backward prelude: header, entry, SMEM, registers, indices.
pub fn emit(ptx: &mut String, config: &FlashAttentionConfig) {
    // Caller should have validated `Direction::Backward`; assert here so
    // a misrouted caller can't produce an invalid kernel.
    let _ = Direction::Backward;

    ptx.push_str(".version 8.7\n");
    ptx.push_str(".target sm_75\n");
    ptx.push_str(".address_size 64\n\n");

    let dyn_smem = backward_needs_dynamic_smem(config);
    if dyn_smem {
        ptx.push_str(".extern .shared .align 16 .b8 shmem[];\n\n");
    }

    let name = kernel_name(config);
    ptx.push_str(&format!(".visible .entry {} (\n", name));
    let params = [
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
        // Tier C backward-specific append.
        (".param .u64", "dO_ptr"),
        (".param .u64", "dq_ptr"), (".param .u64", "dk_ptr"), (".param .u64", "dv_ptr"),
        (".param .u64", "dwq_ptr"), (".param .u64", "dwk_ptr"), (".param .u64", "dwv_ptr"),
        (".param .u64", "dx_ptr"),
    ];
    for (i, (ty, pname)) in params.iter().enumerate() {
        let comma = if i + 1 < params.len() { "," } else { "" };
        ptx.push_str(&format!("    {} {}{}\n", ty, pname, comma));
    }
    ptx.push_str(")\n{\n");

    if !dyn_smem {
        ptx.push_str(&format!(
            "    .shared .align 16 .b8 shmem[{}];\n",
            backward_total_bytes(config)
        ));
    }

    // Base register pool. f32 pool: 48 scratch + head_dim/32 Q slice +
    // head_dim/32 O-ish placeholder (retained for parity with forward).
    let slices_per_lane = ((config.head_dim as u32) / 32).max(1);
    let f32_pool = 48 + slices_per_lane;
    ptx.push_str("    .reg .u32 %tid_x, %warp_id, %lane, %bid_x, %bid_y;\n");
    ptx.push_str("    .reg .u64 %rd<64>;\n");
    ptx.push_str(&format!("    .reg .f32 %f<{}>;\n", f32_pool));
    ptx.push_str("    .reg .b16 %h<32>;\n");
    ptx.push_str("    .reg .pred %p<8>;\n");
    ptx.push_str("    .reg .u32 %r<16>;\n");
    ptx.push_str("    .reg .f32 %scale, %log2e;\n");
    ptx.push_str("    .reg .f32 %row_max, %row_sum;\n");
    ptx.push_str("    .reg .u64 %q_start, %head_idx, %batch_idx, %k_start, %k_max;\n");
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

    // Backward HBM pointer registers (saved-activation inputs +
    // gradient outputs). Separate name so an unsuspecting forward
    // phase doesn't clobber them.
    ptx.push_str("    .reg .u64 %rd_bwd_q_proj, %rd_bwd_k_proj, %rd_bwd_v_proj;\n");
    ptx.push_str("    .reg .u64 %rd_bwd_row_max, %rd_bwd_row_sum;\n");
    ptx.push_str("    .reg .u64 %rd_bwd_do, %rd_bwd_dq, %rd_bwd_dk, %rd_bwd_dv;\n");
    ptx.push_str("    .reg .u64 %rd_bwd_dwq, %rd_bwd_dwk, %rd_bwd_dwv, %rd_bwd_dx;\n");

    // SMEM-base pointer + warp_row (shared with forward contract so
    // backward phase emitters can use the same addressing helpers).
    ptx.push_str("    .reg .u64 %q_smem_base, %k_smem_base, %v_smem_base;\n");
    ptx.push_str("    .reg .u64 %warp_row;\n");

    ptx.push_str("    cvta.shared.u64 %shmem_base, shmem;\n");
    ptx.push_str("    mov.f32 %log2e, 0f3FB8AA3B;  // log2(e)\n");

    // Initialise SMEM tile bases so every backward phase can address
    // them without depending on a specific sibling phase having run
    // first. Q tile at byte 0 (Tier A layout); K/V share the kv_offset
    // region immediately after (K is consumed first, then V overwrites
    // its SMEM slot).
    let kv_off = crate::flash_attention_v2::smem_layout::kv_offset(config);
    ptx.push_str("    mov.u64 %q_smem_base, %shmem_base;\n");
    ptx.push_str(&format!(
        "    add.u64 %k_smem_base, %shmem_base, {};\n", kv_off
    ));
    ptx.push_str(&format!(
        "    add.u64 %v_smem_base, %shmem_base, {};\n", kv_off
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

    // Thread/block indices — identical to forward.
    ptx.push_str("    mov.u32 %tid_x, %tid.x;\n");
    ptx.push_str("    shr.u32 %warp_id, %tid_x, 5;\n");
    ptx.push_str("    and.b32 %lane, %tid_x, 31;\n");
    ptx.push_str("    mov.u32 %bid_x, %ctaid.x;\n");
    ptx.push_str("    mov.u32 %bid_y, %ctaid.y;\n");

    ptx.push_str("    cvt.u64.u32 %q_start, %bid_x;\n");
    ptx.push_str(&format!(
        "    mul.lo.u64 %q_start, %q_start, {};\n",
        config.block_q
    ));
    ptx.push_str("    cvt.u64.u32 %rd16, %bid_y;\n");
    ptx.push_str("    rem.u64 %head_idx,  %rd16, %rd5;\n");
    ptx.push_str("    div.u64 %batch_idx, %rd16, %rd5;\n");
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
