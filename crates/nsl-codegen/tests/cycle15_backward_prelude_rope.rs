//! Cycle-15 structural witnesses: backward prelude must declare the
//! RoPE pair-sweep register block when (rope_q && csha.is_some()).
//! Closes cross-prelude register gap surfaced by cycle-14 ptxas rc=218.
//!
//! 2a was superseded when R7 (`validate_checkpoint_eligibility`) was
//! generalized to refuse checkpoint + rope_q unconditionally (Path B's
//! kv-recompute math is known-broken, not just under segment_masked) —
//! see the test's own doc comment for detail.

use nsl_codegen::flash_attention::{
    CheckpointExtras, CshaExtras, FlashAttentionConfig, RopeStyle,
};
use nsl_codegen::flash_attention_v2::synthesize_backward_with_tier_b;

fn build_cycle15_cfg(rope_q: bool, with_csha: bool) -> FlashAttentionConfig {
    // Inlined from csha_checkpoint_recompute_gpu.rs build_cycle14_config
    // (head_dim=64, seq_len=512 footprint). seq_len is not carried by
    // FlashAttentionConfig; block_q/block_kv=32 per cycle-14 size-down.
    let head_dim: i64 = 64;
    let d_model: u32 = 64; // 1 head, dm == hd for shape alignment
    let csha = if with_csha {
        Some(CshaExtras {
            d_model,
            ..CshaExtras::level1_with_fused_proj(1e-6)
        })
    } else {
        None
    };
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim,
        causal: true,
        paged: false,
        rope_q,
        rope_style: RopeStyle::Adjacent,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha,
        checkpoint: Some(CheckpointExtras::full().bypass_r0_for_testing()),
    }
}

#[test]
fn t_cycle15_2a_checkpoint_rope_q_now_refused_pending_path_b_fix() {
    // Formerly `t_cycle15_2a_backward_prelude_emits_rope_registers`: a
    // cycle-15 structural witness that the backward prelude declares the
    // RoPE pair-sweep registers (%rd_rope_cos/%rd_rope_sin/%f_rope_cos/
    // %p_rope_cos_null) when checkpoint + rope_q reaches PTX emission.
    //
    // Commit 8f774ad (Phase 1.3 pt3) added that register declaration to
    // *unblock compilation* of checkpoint + rope_q, but its own message
    // documents Path B's "remaining GROSS numerical error ...
    // never-GPU-validated ... tracked for follow-up". Nothing refused
    // this composition afterward, so `validate_checkpoint_eligibility`'s
    // R7 was generalized (this audit) to refuse rope_q=true under
    // @checkpoint unconditionally, not just under segment_masked. This
    // config — checkpoint + rope_q, no segment_masked — is exactly what
    // R7 now catches, so PTX emission (and the register it used to
    // assert on) is no longer reachable through the public API.
    //
    // This test now locks in that refusal. Re-derive the original
    // register-presence assertions once Path B's kv-recompute math is
    // fixed and GPU-validated and R7 is narrowed back down.
    let cfg = build_cycle15_cfg(/*rope_q=*/ true, /*with_csha=*/ true);
    let err = synthesize_backward_with_tier_b(&cfg, None)
        .expect_err("G15-2a: checkpoint+rope_q (Path B) must be refused, not synthesized");
    assert!(
        err.contains("rope_q=true") && err.contains("checkpoint"),
        "G15-2a: refusal message missing expected substrings: {err}"
    );
}

#[test]
fn t_cycle15_2b_backward_prelude_skips_rope_when_rope_q_false() {
    let cfg = build_cycle15_cfg(/*rope_q=*/ false, /*with_csha=*/ true);
    let ptx = synthesize_backward_with_tier_b(&cfg, None).unwrap();
    let prelude_window = &ptx[..ptx.len().min(8192)];
    assert!(
        !prelude_window.contains("%rd_rope_cos"),
        "G15-2b: backward prelude emits RoPE registers when rope_q=false (gate broken)"
    );
}
