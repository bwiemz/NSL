//! Cycle-16 G16-1 structural witnesses: 3-defect coordinated fix for Tier B
//! scalar backward dK correctness (Option A).
//!
//! G16-1a: K-tile inverse-RoPE cs_row now adds k_start (defect-2 fix).
//! G16-1b: Phase 4 dK cooperative store present in synthesized PTX (defect-1+3).
//! G16-1c: emit_store_kv_only no longer stores dK to f32 scratch in the KV loop.

use nsl_codegen::flash_attention::{
    CheckpointExtras, CshaExtras, FlashAttentionConfig, RopeStyle,
};
use nsl_codegen::flash_attention_v2::synthesize_backward_with_tier_b;

fn build_cycle16_g16_1_cfg() -> FlashAttentionConfig {
    // Matches the cycle-15 smoke config: head_dim=64, block_q=block_kv=32,
    // rope_q=true to exercise the dRoPE K-tile path (defect-2 site).
    // segment_masked=false (sentinel-disabled) so the non-PCA branch fires,
    // which is the defect-2 site (line 315 of csha_hooks_backward.rs).
    let head_dim: i64 = 64;
    let d_model: u32 = 64;
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim,
        causal: true,
        paged: false,
        rope_q: true,
        rope_style: RopeStyle::Adjacent,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras {
            d_model,
            ..CshaExtras::level1_with_fused_proj(1e-6)
        }),
        checkpoint: Some(CheckpointExtras::full().bypass_r0_for_testing()),
    }
}

/// G16-1a: K-tile dRoPE cs_row uses `add.u64 %rd35, %rd33, %k_start`
/// (cycle-16 defect-2 fix). The prior `mov.u64 %rd35, %rd33` applied the
/// wrong cos/sin slice to K tiles starting past position 0.
#[test]
fn t_cycle16_g16_1a_dk_k_start_offset_in_drope() {
    let cfg = build_cycle16_g16_1_cfg();
    let ptx = synthesize_backward_with_tier_b(&cfg, None).unwrap();
    assert!(
        ptx.contains("add.u64 %rd35, %rd33, %k_start"),
        "G16-1a: K-tile dRoPE must use add.u64 %rd35, %rd33, %k_start \
         (cycle-16 defect-2 fix; prior: mov.u64 %rd35, %rd33)"
    );
    // Verify the Q-tile counterpart is also present (pre-existing, unchanged).
    assert!(
        ptx.contains("add.u64 %rd35, %rd33, %q_start"),
        "G16-1a: Q-tile dRoPE must still use add.u64 %rd35, %rd33, %q_start"
    );
}

/// G16-1b: Phase 4 dK store is present in synthesized PTX.
/// emit_store_dk_only is wired in mod.rs after emit_drmsnorm. Its PTX comment
/// "Phase 4 dK store" serves as the structural witness.
#[test]
fn t_cycle16_g16_1b_phase4_dk_store_present() {
    let cfg = build_cycle16_g16_1_cfg();
    let ptx = synthesize_backward_with_tier_b(&cfg, None).unwrap();
    assert!(
        ptx.contains("Phase 4 dK store"),
        "G16-1b: Phase 4 dK store comment must appear in PTX (emit_store_dk_only \
         wired in mod.rs after emit_drmsnorm)"
    );
    // Also verify the skip label (null-guard) is emitted.
    assert!(
        ptx.contains("V2_BWD_STORE_DK_SKIP_0:"),
        "G16-1b: V2_BWD_STORE_DK_SKIP_0 label must be present (emit_store_dk_only)"
    );
}

/// G16-1c: emit_store_kv_only no longer stores dK to f32 scratch in the KV loop.
/// The DV scratch path remains; DK scratch path is gone.
#[test]
fn t_cycle16_g16_1c_kv_only_no_dk_in_loop() {
    use nsl_codegen::flash_attention_v2::phases::backward::finalize::emit_store_kv_only;
    let cfg = build_cycle16_g16_1_cfg();
    let mut ptx = String::new();
    emit_store_kv_only(&mut ptx, &cfg, 0);
    // dV scratch path still present (V-only emit).
    assert!(
        ptx.contains("dv_scratch_ptr"),
        "G16-1c: emit_store_kv_only must still reference dv_scratch_ptr"
    );
    // dK scratch path must be gone.
    assert!(
        !ptx.contains("dk_scratch_ptr"),
        "G16-1c: emit_store_kv_only must NOT reference dk_scratch_ptr \
         (cycle-16 G16-1 removed per-KV-tile dK f32-scratch RMW)"
    );
    // Confirm V2_BWD_STORE_DV_SKIP is emitted.
    assert!(
        ptx.contains("V2_BWD_STORE_DV_SKIP_0:"),
        "G16-1c: DV skip label must be present in emit_store_kv_only output"
    );
    // Confirm the old DK skip label is gone from this function's output.
    assert!(
        !ptx.contains("V2_BWD_STORE_DK_SKIP_"),
        "G16-1c: DK skip label must NOT be emitted by emit_store_kv_only \
         (that label now belongs to emit_store_dk_only)"
    );
}
