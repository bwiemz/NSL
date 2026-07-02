use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::phases::csha_hooks;

#[test]
fn emit_save_softmax_state_fires_hbm_save_under_save_activations_without_fused() {
    let cfg = FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: 32,
        causal: false, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, num_sink_tokens: 0, segment_masked: false, gpu_sm: 89,
        csha: Some(CshaExtras {
            level: 1,
            fused_projections: false,
            save_activations_for_backward: true,
            d_model: 32,
            ..CshaExtras::default()
        }),
        checkpoint: None,
    };
    let mut ptx = String::new();
    csha_hooks::emit_save_softmax_state(&mut ptx, &cfg, 0);
    assert!(ptx.contains("row_max_ptr"), "HBM row_max save must fire under save_activations=true + fused_projections=false");
    assert!(ptx.contains("row_sum_ptr"), "HBM row_sum save must fire same config");
    // SMEM save portion should NOT fire (that's only for fused_projections=true S+PV split design)
    assert!(!ptx.contains("st.shared.f32 [%rd_wt_dst], %row_max"),
        "SMEM-side save should remain gated on fused_projections=true");
}
