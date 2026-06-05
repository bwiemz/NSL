//! Full-kernel PTX byte-snapshot for the B1.5 single-iteration scaffold.
//! The cp.async / commit_group / wait_group sequence within this
//! snapshot is the load-bearing FSM correctness gate per spec section 6.5.
//! A reviewer of the snapshot diff can verify FSM correctness without
//! running the kernel.

use insta::assert_snapshot;
use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b1::synthesize;

#[test]
fn tier_b1_full_kernel_canonical_snapshot() {
    let config = canonical_test_config();
    let chunk = 128; // chunk_config::select picks 128 for this 32x32x32 cfg.
    let ptx = synthesize(&config, chunk);
    let ptx_str = String::from_utf8_lossy(&ptx);
    assert_snapshot!(ptx_str.into_owned());
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
