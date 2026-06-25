//! Snapshot the cp.async helper sequences in isolation. Each helper
//! has a fixed PTX shape; this test locks down regressions.

use insta::assert_snapshot;
use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b1::pipeline::{
    emit_prologue_kicks, emit_main_loop_phase_a_load, emit_main_loop_phase_c_swap,
};

#[test]
fn prologue_kicks_snapshot() {
    let config = canonical_test_config();
    let mut ptx = String::new();
    emit_prologue_kicks(&mut ptx, &config);
    assert_snapshot!(ptx);
}

#[test]
fn phase_a_load_snapshot() {
    let config = canonical_test_config();
    let mut ptx = String::new();
    emit_main_loop_phase_a_load(&mut ptx, &config, /* kv_iter = */ 0);
    assert_snapshot!(ptx);
}

#[test]
fn phase_c_swap_snapshot() {
    let config = canonical_test_config();
    let mut ptx = String::new();
    emit_main_loop_phase_c_swap(&mut ptx, &config, /* kv_iter = */ 0);
    assert_snapshot!(ptx);
}

fn canonical_test_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 32,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 120,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model: 2048,
            ..CshaExtras::default()
        }),
        checkpoint: None,
    }
}
