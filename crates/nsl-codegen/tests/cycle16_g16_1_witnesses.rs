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

/// G16-1a/1b formerly asserted on PTX text reachable only by synthesizing
/// checkpoint + rope_q (Path B) end-to-end. That composition is now
/// refused unconditionally by `validate_checkpoint_eligibility`'s R7 (see
/// checkpoint_v1_integration.rs / cycle15_backward_prelude_rope.rs for the
/// full rationale: Path B's kv-recompute math has a documented "GROSS
/// numerical error" that was never fixed, and 8f774ad only unblocked its
/// *compilation*, not its correctness). Both now lock in the refusal
/// instead of asserting on unreachable PTX text; re-derive the original
/// register/label assertions once Path B is fixed and R7 is narrowed.
#[test]
fn t_cycle16_g16_1a_checkpoint_rope_q_now_refused_pending_path_b_fix() {
    let cfg = build_cycle16_g16_1_cfg();
    let err = synthesize_backward_with_tier_b(&cfg, None)
        .expect_err("G16-1a: checkpoint+rope_q (Path B) must be refused, not synthesized");
    assert!(
        err.contains("rope_q=true") && err.contains("checkpoint"),
        "G16-1a: refusal message missing expected substrings: {err}"
    );
}

#[test]
fn t_cycle16_g16_1b_checkpoint_rope_q_now_refused_pending_path_b_fix() {
    let cfg = build_cycle16_g16_1_cfg();
    let err = synthesize_backward_with_tier_b(&cfg, None)
        .expect_err("G16-1b: checkpoint+rope_q (Path B) must be refused, not synthesized");
    assert!(
        err.contains("rope_q=true") && err.contains("checkpoint"),
        "G16-1b: refusal message missing expected substrings: {err}"
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
