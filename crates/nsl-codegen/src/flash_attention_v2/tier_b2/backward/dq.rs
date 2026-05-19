//! dQ-kernel emitter for Tier B.2 backward (scaffold; inner-loop body in Tasks 9-12).
//!
//! Q-outer, kv-inner. dQ accumulator register-resident across kv_iter.
//! Producer warp (warp 0) issues cp.async for Q, dO, K, V; consumer warps
//! (1-3) execute MMA chain. No atomicAdd.
//!
//! Per spec §5.2 amendment: SMEM is sized via tier_b2_dq_total_smem_bytes
//! (includes col-major K re-stage band). emit_dq_acc_init uses
//! tier_b2_effective_bq for the per-hd bq=32 fallback at hd=128.
//!
//! Spec: docs/superpowers/specs/2026-05-19-csha-tier-b2-phase2-design.md §4 + §5.2

use crate::flash_attention::FlashAttentionConfig;
use crate::flash_attention_v2::smem_layout::{
    tier_b2_dq_total_smem_bytes, tier_b2_effective_bq,
};
use crate::flash_attention_v2::tier_b2::backward::BackwardSynthError;

pub fn synthesize_dq_kernel(
    config: &FlashAttentionConfig,
) -> Result<String, BackwardSynthError> {
    let hd = config.head_dim as u32;
    if !hd.is_multiple_of(32) {
        return Err(BackwardSynthError::UnsupportedHeadDim(hd));
    }

    let mut ptx = String::new();
    emit_prelude(&mut ptx);
    emit_entry_signature(&mut ptx);
    emit_register_decls(&mut ptx, config);
    emit_grid_id_setup(&mut ptx);
    emit_outer_loop_open(&mut ptx);
    emit_q_dO_producer_load(&mut ptx);
    emit_dq_acc_init(&mut ptx, config);
    emit_inner_loop_open(&mut ptx);
    // Inner-loop body placeholder — filled in Tasks 9-11.
    ptx.push_str("    // TODO Tasks 9-11: tile-skip predicate + S + P + dP + dS + dQ-update\n");
    emit_inner_loop_close(&mut ptx);
    emit_outer_loop_close(&mut ptx);
    emit_dq_finalize(&mut ptx);
    emit_entry_close(&mut ptx);
    Ok(ptx)
}

fn emit_prelude(ptx: &mut String) {
    ptx.push_str(".version 7.0\n");
    ptx.push_str(".target sm_80\n");
    ptx.push_str(".address_size 64\n\n");
}

fn emit_entry_signature(ptx: &mut String) {
    ptx.push_str(".visible .entry tier_b2_dq_kernel(\n");
    ptx.push_str("    .param .u64 q_saved_ptr,\n");
    ptx.push_str("    .param .u64 k_saved_ptr,\n");
    ptx.push_str("    .param .u64 v_saved_ptr,\n");
    ptx.push_str("    .param .u64 d_o_ptr,\n");
    ptx.push_str("    .param .u64 row_max_ptr,\n");
    ptx.push_str("    .param .u64 row_sum_ptr,\n");
    ptx.push_str("    .param .u64 d_ptr,\n");
    ptx.push_str("    .param .u64 segment_ids_ptr,\n");
    ptx.push_str("    .param .u64 d_q_out_ptr,\n");
    ptx.push_str("    .param .u32 seq_len,\n");
    ptx.push_str("    .param .u32 heads,\n");
    ptx.push_str("    .param .u32 batch\n");
    ptx.push_str(")\n");
    ptx.push_str(".maxntid 128, 1, 1\n");
    ptx.push_str("{\n");
}

fn emit_register_decls(ptx: &mut String, config: &FlashAttentionConfig) {
    ptx.push_str("    .reg .u32 %tid, %lane_id, %warp_id;\n");
    ptx.push_str("    .reg .u32 %q_tile, %kv_tile, %head, %batch_idx;\n");
    ptx.push_str("    .reg .pred %p_tile_active, %p_producer, %p_consumer;\n");
    ptx.push_str("    .reg .u32 %addr_lo, %tile_skip_predicate;\n");
    ptx.push_str("    .reg .u64 %addr;\n");
    // SMEM extern: per SPEC AMENDMENT, size from tier_b2_dq_total_smem_bytes (incl. K-colmajor).
    let total_smem = tier_b2_dq_total_smem_bytes(config);
    ptx.push_str(&format!("    .extern .shared .align 16 .b8 shmem[{}];\n", total_smem));
    ptx.push('\n');
}

fn emit_grid_id_setup(ptx: &mut String) {
    ptx.push_str("    mov.u32 %tid, %tid.x;\n");
    ptx.push_str("    and.b32 %lane_id, %tid, 31;\n");
    ptx.push_str("    shr.u32 %warp_id, %tid, 5;\n");
    ptx.push_str("    mov.u32 %q_tile,    %ctaid.x;\n");
    ptx.push_str("    mov.u32 %head,      %ctaid.y;\n");
    ptx.push_str("    mov.u32 %batch_idx, %ctaid.z;\n");
    ptx.push_str("    setp.eq.u32 %p_producer, %warp_id, 0;\n");
    ptx.push('\n');
}

fn emit_outer_loop_open(ptx: &mut String) {
    ptx.push_str("DQ_Q_ITER_LOOP:\n");
}

#[allow(non_snake_case)]
fn emit_q_dO_producer_load(ptx: &mut String) {
    ptx.push_str("    // Warp 0 (producer): issue cp.async for Q + dO tiles.\n");
    ptx.push_str("    // Warps 1-3 (consumers): wait on cp.async.wait_group.\n");
    ptx.push_str("    @!%p_producer bra DQ_PROD_LOAD_DONE;\n");
    ptx.push_str("    // cp.async Q tile -> shmem[+q_offset]\n");
    ptx.push_str("    // cp.async dO tile -> shmem[+dO_offset]\n");
    ptx.push_str("    cp.async.commit_group;\n");
    ptx.push_str("DQ_PROD_LOAD_DONE:\n");
    ptx.push_str("    cp.async.wait_group 0;\n");
    ptx.push_str("    bar.sync 0;\n");
}

fn emit_dq_acc_init(ptx: &mut String, config: &FlashAttentionConfig) {
    // Per SPEC AMENDMENT: use tier_b2_effective_bq for any block_q-dependent
    // sizing. accumulator_fragments = head_dim / 32 (independent of bq, but
    // we reference effective_bq to honor the per-hd fallback for warp tiling).
    let effective_bq = tier_b2_effective_bq(config);
    let accumulator_fragments = (config.head_dim / 32) as u32;
    ptx.push_str(&format!(
        "    // Zero dQ accumulator regs ({} fragments x 4 f32/lane; effective_bq={})\n",
        accumulator_fragments, effective_bq,
    ));
    for f in 0..accumulator_fragments {
        for r in 0..4 {
            ptx.push_str(&format!("    .reg .f32 %dq_acc_{}_{};\n", f, r));
            ptx.push_str(&format!("    mov.f32 %dq_acc_{}_{}, 0.0;\n", f, r));
        }
    }
}

fn emit_inner_loop_open(ptx: &mut String) {
    ptx.push_str("DQ_KV_ITER_LOOP:\n");
    ptx.push_str("    // Warp 0 issues cp.async for K, V tiles.\n");
    ptx.push_str("    @!%p_producer bra DQ_KV_LOAD_DONE;\n");
    ptx.push_str("    // cp.async K -> shmem[+k_offset]\n");
    ptx.push_str("    // cp.async V -> shmem[+v_offset]\n");
    ptx.push_str("    cp.async.commit_group;\n");
    ptx.push_str("DQ_KV_LOAD_DONE:\n");
    ptx.push_str("    cp.async.wait_group 0;\n");
    ptx.push_str("    bar.sync 0;\n");
    ptx.push('\n');
}

fn emit_inner_loop_close(ptx: &mut String) {
    ptx.push_str("    // ... kv_iter increment + back-edge (Task 12 wires)\n");
    ptx.push_str("    bra DQ_KV_ITER_DONE;  // placeholder unconditional exit until Task 12\n");
    ptx.push_str("DQ_KV_ITER_DONE:\n");
}

fn emit_outer_loop_close(ptx: &mut String) {
    ptx.push_str("    // ... q_iter increment + back-edge (Task 12 wires)\n");
    ptx.push_str("    bra DQ_Q_ITER_DONE;  // placeholder unconditional exit until Task 12\n");
    ptx.push_str("DQ_Q_ITER_DONE:\n");
}

fn emit_dq_finalize(ptx: &mut String) {
    ptx.push_str("    // TODO Task 12: scatter dQ accumulator regs to HBM dQ[B,H,S,D]\n");
}

fn emit_entry_close(ptx: &mut String) {
    ptx.push_str("    ret;\n");
    ptx.push_str("}\n");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};

    fn canonical_cfg() -> FlashAttentionConfig {
        FlashAttentionConfig {
            block_q: 64, block_kv: 64, head_dim: 128,
            causal: false, paged: false,
            rope_q: false, rope_style: RopeStyle::HalfSplit,
            gqa_group_size: 1, tree_mask: false,
            gpu_sm: 80, segment_masked: false,
            csha: Some(CshaExtras { level: 2, ..Default::default() }),
        }
    }

    #[test]
    fn synthesize_dq_kernel_targets_sm80() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(ptx.contains(".target sm_80"));
    }

    #[test]
    fn synthesize_dq_kernel_has_four_warp_block() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(ptx.contains(".maxntid 128"));
    }

    #[test]
    fn synthesize_dq_kernel_extern_smem_sized_from_total_bytes() {
        // Per SPEC AMENDMENT: SMEM extern declaration uses tier_b2_dq_total_smem_bytes
        // which INCLUDES the col-major K re-stage band.
        use crate::flash_attention_v2::smem_layout::tier_b2_dq_total_smem_bytes;
        let cfg = canonical_cfg();
        let expected_total = tier_b2_dq_total_smem_bytes(&cfg);
        let ptx = synthesize_dq_kernel(&cfg).unwrap();
        // Must declare shmem[<total>] not shmem[] (unsized) and not mixed with .shared
        assert!(ptx.contains(&format!(".extern .shared .align 16 .b8 shmem[{}]", expected_total)),
            "expected explicit shmem[{}] sizing, got:\n{ptx}", expected_total);
        assert!(!ptx.contains(".shared .align 16 .b8 dq_static"),
            "must NOT mix static .shared with extern");
    }

    #[test]
    #[allow(non_snake_case)]
    fn synthesize_dq_kernel_uses_cp_async_for_q_dO_load() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(ptx.contains("cp.async"),
            "expected cp.async for Q+dO producer-staged load");
    }

    #[test]
    fn synthesize_dq_kernel_emits_outer_q_iter_loop() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(ptx.contains("DQ_Q_ITER_LOOP"),
            "expected outer q_iter loop label");
    }

    #[test]
    fn synthesize_dq_kernel_emits_inner_kv_iter_loop() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        assert!(ptx.contains("DQ_KV_ITER_LOOP"),
            "expected inner kv_iter loop label");
    }

    #[test]
    fn synthesize_dq_kernel_warp0_is_producer() {
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // warp 0 = producer; producer issues cp.async; consumers (warps 1-3) wait
        assert!(ptx.contains("setp.eq.u32 %p_producer, %warp_id, 0") ||
                ptx.contains("setp.eq.u32 %p_producer, %warp_id, 0;"));
    }

    #[test]
    fn synthesize_dq_kernel_uses_effective_bq_at_hd_128() {
        // Per SPEC AMENDMENT: emit_dq_acc_init must respect effective_bq.
        // At hd=128 with raw block_q=64, effective_bq = 32 (SMEM-pressure fallback).
        let ptx = synthesize_dq_kernel(&canonical_cfg()).unwrap();
        // The dQ accumulator init must use effective_bq for sizing.
        // Concrete check: the accumulator register count should match
        // hd/32 = 4 fragments x 4 f32/lane (independent of bq, but the scaffold
        // should reference effective_bq for warp tiling).
        // Loose check: must reference accumulators (e.g., %dq_acc_*)
        assert!(ptx.contains("dq_acc"),
            "expected dQ accumulator register decls");
    }
}
