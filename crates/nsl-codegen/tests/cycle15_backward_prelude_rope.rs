//! Cycle-15 structural witnesses: backward prelude must declare the
//! RoPE pair-sweep register block when (rope_q && csha.is_some()).
//! Closes cross-prelude register gap surfaced by cycle-14 ptxas rc=218.

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
fn t_cycle15_2a_backward_prelude_emits_rope_registers() {
    let cfg = build_cycle15_cfg(/*rope_q=*/ true, /*with_csha=*/ true);
    let ptx = synthesize_backward_with_tier_b(&cfg, None).unwrap();
    // Window the first 8KB so the assertion exercises the prelude, not the body.
    let prelude_window = &ptx[..ptx.len().min(8192)];
    assert!(
        prelude_window.contains("%rd_rope_cos"),
        "G15-2a: backward prelude missing %rd_rope_cos under rope_q=true"
    );
    assert!(
        prelude_window.contains("%rd_rope_sin"),
        "G15-2a: backward prelude missing %rd_rope_sin under rope_q=true"
    );
    assert!(
        prelude_window.contains("%f_rope_cos"),
        "G15-2a: backward prelude missing %f_rope_cos under rope_q=true"
    );
    assert!(
        prelude_window.contains("%p_rope_cos_null"),
        "G15-2a: backward prelude missing %p_rope_cos_null under rope_q=true"
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
