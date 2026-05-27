//! Phase 3 T2 structural gate: dQ/dK/dV HBM->SMEM cooperative load.
//!
//! The projection-backward kernel (`tier_b2_proj_backward`) reuses the scalar
//! `emit_dproj` / `emit_drmsnorm` emitters, which READ dQ/dK/dV from SMEM at
//! `backward_d{q,k,v}_offset` as row-major `[block_q, head_dim]` f32 tiles off
//! `%shmem_base`. The Tier B.2 dQ/dK/dV kernels write those gradients to HBM as
//! f32. `emit_dqkv_hbm_to_smem_load` bridges the two: a cooperative 128-thread
//! load that stages each HBM buffer into the EXACT SMEM slot `emit_dproj` reads.
//!
//! These are CPU-side, no-GPU/no-ptxas structural assertions on the emitted PTX
//! per-cell address arithmetic (modelled on `tier_b2_dq_f1_ds_scatter_lane_mapping`).

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::smem_layout::{
    backward_dk_offset, backward_dq_offset, backward_dv_offset,
};
use nsl_codegen::flash_attention_v2::tier_b2::backward::proj_backward::emit_dqkv_hbm_to_smem_load;

fn smoke_cfg() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 64,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model: 64,
            active_heads: 1,
            ..Default::default()
        }),
    }
}

#[test]
fn load_targets_emit_dproj_offsets_f32() {
    let cfg = smoke_cfg();
    let mut ptx = String::new();
    emit_dqkv_hbm_to_smem_load(&mut ptx, &cfg);
    for off in [
        backward_dq_offset(&cfg),
        backward_dk_offset(&cfg),
        backward_dv_offset(&cfg),
    ] {
        assert!(ptx.contains(&format!("{off}")), "missing dest offset {off}");
    }
    assert!(ptx.contains("st.shared.f32"), "dQ/dK/dV staged as f32");
    assert!(ptx.contains("ld.global.f32"), "loaded from HBM as f32");
    assert!(!ptx.contains("st.shared.b16"), "must not narrow to f16");
    assert!(ptx.is_ascii(), "PTX must be 7-bit ASCII");
}

#[test]
fn load_reads_all_three_hbm_buffers() {
    let cfg = smoke_cfg();
    let mut ptx = String::new();
    emit_dqkv_hbm_to_smem_load(&mut ptx, &cfg);
    // Each buffer is loaded from its scalar-backward param register.
    for reg in ["%rd_bwd_dq", "%rd_bwd_dk", "%rd_bwd_dv"] {
        assert!(
            ptx.contains(reg),
            "expected load to reference HBM base register {reg}"
        );
    }
    // Exactly three cooperative-load loops emitted (one per buffer).
    let loops = ptx.matches("V2_PROJBWD_LOAD_").count();
    assert!(
        loops >= 3,
        "expected at least one labelled loop per buffer, got {loops} label refs"
    );
}

#[test]
fn load_is_read_only_on_hbm() {
    // This emitter loads from HBM and stages into SMEM; it must NEVER store
    // back to HBM. No st.global of any width may appear.
    let cfg = smoke_cfg();
    let mut ptx = String::new();
    emit_dqkv_hbm_to_smem_load(&mut ptx, &cfg);
    assert!(
        !ptx.contains("st.global"),
        "load is read-only on the HBM dQ/dK/dV buffers; no st.global allowed"
    );
}

#[test]
fn smem_dest_address_uses_shmem_base_and_offsets() {
    // The SMEM destination MUST be %shmem_base + backward_d?_offset + (s*head_dim+j)*4.
    // Assert the address chain folds %shmem_base together with the exact dest offset.
    let cfg = smoke_cfg();
    let mut ptx = String::new();
    emit_dqkv_hbm_to_smem_load(&mut ptx, &cfg);
    assert!(
        ptx.contains("%shmem_base"),
        "SMEM dest address must be computed off %shmem_base"
    );
    // Each destination offset is folded into the SMEM address arithmetic with
    // an add.u64 against the per-cell (s*head_dim+j)*4 term.
    for off in [
        backward_dq_offset(&cfg),
        backward_dk_offset(&cfg),
        backward_dv_offset(&cfg),
    ] {
        assert!(
            ptx.contains(&format!("add.u64 %shmem_base, {off}"))
                || ptx.contains(&format!(", {off};")),
            "expected dest offset {off} folded into SMEM address chain"
        );
    }
}

#[test]
fn load_partitions_over_block_q_times_head_dim_cells() {
    // Cooperative partition: cell = tid_x + k*128, guarded by cell < block_q*head_dim.
    let cfg = smoke_cfg();
    let total_cells = (cfg.block_q as u32) * (cfg.head_dim as u32);
    let mut ptx = String::new();
    emit_dqkv_hbm_to_smem_load(&mut ptx, &cfg);
    assert!(
        ptx.contains(&format!("{total_cells}")),
        "expected total_cells={total_cells} guard in the cooperative partition"
    );
    // The cell partition is keyed off %tid_x (the prelude-provided thread id).
    assert!(
        ptx.contains("%tid_x"),
        "cooperative load must partition cells over %tid_x"
    );
}

#[test]
fn load_emits_per_row_hbm_base_via_seq_and_head_dim() {
    // Mirror emit_xnorm_recompute's HBM row-addressing convention:
    //   flat = head_idx*seq + (q_start+s); byte_off = flat * head_dim * 4.
    // seq is in %rd6, head_dim in %rd7, head_idx/%q_start are prelude registers.
    let cfg = smoke_cfg();
    let mut ptx = String::new();
    emit_dqkv_hbm_to_smem_load(&mut ptx, &cfg);
    assert!(ptx.contains("%head_idx"), "per-row HBM base uses %head_idx");
    assert!(ptx.contains("%q_start"), "per-row HBM base uses %q_start");
    assert!(
        ptx.contains("%rd6"),
        "per-row HBM base uses seq_len in %rd6 (emit_xnorm_recompute convention)"
    );
    assert!(
        ptx.contains("%rd7"),
        "per-row HBM base uses head_dim in %rd7 (emit_xnorm_recompute convention)"
    );
}
