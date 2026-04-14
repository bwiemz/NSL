//! Per-phase snapshot tests. Each test emits a single phase against a
//! fixed config and diffs the generated PTX string against a stored
//! snapshot. Use `cargo insta review` to accept snapshot changes.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::phases::{prelude, q_load};

fn csha_canonical() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: 32,
        causal: true, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 75, csha: None,
    }
}

fn non_csha_canonical() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 128,
        causal: true, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 75, csha: None,
    }
}

#[test]
fn phase_prelude__32x32x32_snapshot() {
    let mut ptx = String::new();
    prelude::emit(&mut ptx, &csha_canonical());
    insta::assert_snapshot!("phase_prelude__32x32x32", ptx);
}

#[test]
fn phase_prelude__64x64x128_snapshot() {
    let mut ptx = String::new();
    prelude::emit(&mut ptx, &non_csha_canonical());
    insta::assert_snapshot!("phase_prelude__64x64x128", ptx);
}

#[test]
fn phase_q_load__32x32x32_snapshot() {
    let mut ptx = String::new();
    q_load::emit(&mut ptx, &csha_canonical(), 0);
    insta::assert_snapshot!("phase_q_load__32x32x32_iter0", ptx);
}

#[test]
fn phase_q_load__64x64x128_snapshot() {
    let mut ptx = String::new();
    q_load::emit(&mut ptx, &non_csha_canonical(), 0);
    insta::assert_snapshot!("phase_q_load__64x64x128_iter0", ptx);
}
