use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;

#[test]
fn forward_synthesis_includes_row_max_hbm_save_under_save_activations_without_fused() {
    let cfg = FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: 32,
        causal: false, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, segment_masked: false, gpu_sm: 89,
        csha: Some(CshaExtras {
            level: 1,
            fused_projections: false,
            save_activations_for_backward: true,
            d_model: 32,
            ..CshaExtras::default()
        }),
    };
    let ptx_bytes = synthesize_flash_attention_ptx_v2(&cfg);
    let ptx = String::from_utf8_lossy(&ptx_bytes);
    assert!(
        ptx.contains("row_max_ptr"),
        "Forward PTX under save_activations=true + fused_projections=false must emit row_max_ptr HBM write"
    );
    assert!(
        ptx.contains("row_sum_ptr"),
        "Forward PTX under same config must emit row_sum_ptr HBM write"
    );
}
