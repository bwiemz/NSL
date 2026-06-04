use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::dispatch::tier_b2_hybrid_backward_eligible;

fn cfg(hd: i64, heads: u32, d_model: u32, rope: bool) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: hd, causal: true, paged: false,
        rope_q: rope, rope_style: RopeStyle::HalfSplit, gqa_group_size: 1, tree_mask: false,
        gpu_sm: 80, segment_masked: false,
        csha: Some(CshaExtras { level: 2, d_model, active_heads: heads, ..Default::default() }),
    }
}

#[test] fn smoke_intersection_is_eligible() {
    assert!(tier_b2_hybrid_backward_eligible(&cfg(64, 1, 64, false), 64));
}
#[test] fn heads_gt_1_rejected() {
    assert!(!tier_b2_hybrid_backward_eligible(&cfg(64, 2, 128, false), 64));
}
#[test] fn d_model_ne_head_dim_rejected() {
    assert!(!tier_b2_hybrid_backward_eligible(&cfg(64, 1, 128, false), 64));
}
#[test] fn seq_gt_block_q_rejected() {
    assert!(!tier_b2_hybrid_backward_eligible(&cfg(64, 1, 64, false), 128));
}
/// Sprint 10 (codegen + runtime) prepared the hybrid for rope_q=true via
/// emit_drope in proj_backward + cos/sin threading in
/// csha_tier_b2_backward_launch. HOWEVER the production wengert lowering
/// at `wengert_lower.rs:1958` still passes null cos/sin, so eligibility
/// must gate on `!rope_q` to keep production safe (the null-guard inside
/// emit_drope would short-circuit silently, leaving dQ/dK post-RoPE
/// when emit_dproj reads them). Reverts the dispatch widening pending
/// the wengert source-side cos/sin wiring follow-on.
#[test] fn rope_q_rejected_pending_wengert_cos_sin_wiring() {
    assert!(!tier_b2_hybrid_backward_eligible(&cfg(64, 1, 64, true), 64));
}
#[test] fn non_csha_rejected() {
    let mut c = cfg(64, 1, 64, false); c.csha = None;
    assert!(!tier_b2_hybrid_backward_eligible(&c, 64));
}
#[test] fn active_heads_zero_rejected() {
    // active_heads=0 ("all heads", possibly many) is ambiguous -> must NOT be eligible.
    assert!(!tier_b2_hybrid_backward_eligible(&cfg(64, 0, 64, false), 64));
}
