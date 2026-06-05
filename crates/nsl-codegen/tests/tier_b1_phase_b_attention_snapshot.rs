//! Snapshot the Phase B (QK^T + softmax + PV) emitter output. The cp.async
//! cadence is in pipeline.rs; this snapshot locks the attention math
//! structure (MMA tile counts, softmax bfly placeholder, O_acc register
//! declarations).

use insta::assert_snapshot;
use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b1::attention_mma::emit_phase_b_attention;

#[test]
fn phase_b_canonical_snapshot() {
    let config = canonical_test_config();
    let mut ptx = String::new();
    emit_phase_b_attention(&mut ptx, &config, /* kv_iter = */ 0, /* slot = */ 0);
    insta::with_settings!({ snapshot_suffix => "kv0_slot0" }, {
        assert_snapshot!(ptx);
    });
}

#[test]
fn phase_b_kv_iter_1_snapshot() {
    let config = canonical_test_config();
    let mut ptx = String::new();
    emit_phase_b_attention(&mut ptx, &config, /* kv_iter = */ 1, /* slot = */ 1);
    insta::with_settings!({ snapshot_suffix => "kv1_slot1" }, {
        assert_snapshot!(ptx);
    });
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
        csha: Some(CshaExtras { level: 2, d_model: 2048, ..CshaExtras::default() }),
    }
}
