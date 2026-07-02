use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b2::dispatch::tier_b2_hybrid_backward_eligible;

fn cfg(hd: i64, heads: u32, d_model: u32, rope: bool) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: hd, causal: true, paged: false,
        rope_q: rope, rope_style: RopeStyle::HalfSplit, gqa_group_size: 1, tree_mask: false, num_sink_tokens: 0,
        gpu_sm: 80, segment_masked: false,
        csha: Some(CshaExtras { level: 2, d_model, active_heads: heads, ..Default::default() }),
        checkpoint: None,
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
/// Sprint 1 cycle-2: rope_q=true is now ACCEPTED. The CshaSavePointers
/// cos/sin channel structurally guarantees forward and backward see
/// identical Cranelift Values for cos/sin (today both null → both skip
/// rotation via the in-kernel null-guard, self-consistent). When future
/// work threads non-null cos/sin into the forward call site at
/// `wengert_lower.rs:564` / `expr/advanced.rs:1692`, backward picks them
/// up automatically via `saves.cos`/`saves.sin` at `wengert_lower.rs:1958`
/// — the H1 divergence risk is structurally closed.
#[test] fn rope_q_accepted_via_saves_channel() {
    assert!(tier_b2_hybrid_backward_eligible(&cfg(64, 1, 64, true), 64));
}
#[test] fn non_csha_rejected() {
    let mut c = cfg(64, 1, 64, false); c.csha = None;
    assert!(!tier_b2_hybrid_backward_eligible(&c, 64));
}
#[test] fn active_heads_zero_rejected() {
    // active_heads=0 ("all heads", possibly many) is ambiguous -> must NOT be eligible.
    assert!(!tier_b2_hybrid_backward_eligible(&cfg(64, 0, 64, false), 64));
}
