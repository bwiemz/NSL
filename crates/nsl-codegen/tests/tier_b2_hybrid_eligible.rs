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
#[test] fn rope_q_rejected() {
    assert!(!tier_b2_hybrid_backward_eligible(&cfg(64, 1, 64, true), 64));
}
#[test] fn non_csha_rejected() {
    let mut c = cfg(64, 1, 64, false); c.csha = None;
    assert!(!tier_b2_hybrid_backward_eligible(&c, 64));
}
