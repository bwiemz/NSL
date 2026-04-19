//! Snapshot of a full forward FA2 kernel with segment_masked=true.
//!
//! Captures the wiring of the segment-mask helper into s_compute.rs and the
//! PCA prelude block (Task 3B of 4). This complements the unit-level snapshot
//! in pca_segment_mask_snapshot.rs by showing the INTEGRATED form.
//!
//! Diffing this snapshot against the equivalent segment_masked=false kernel
//! (fa_v2_snapshots__kernel_full__32x32x32_nocsha) shows exactly what PCA
//! Tier A contributed to the kernel text.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;

fn minimal_segment_masked_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 32,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        gpu_sm: 80,
        segment_masked: true,
        csha: None,
    }
}

#[test]
fn forward_kernel_segment_masked_causal_32_32_32() {
    let ptx_bytes = synthesize_flash_attention_ptx_v2(&minimal_segment_masked_config());
    let ptx = String::from_utf8(ptx_bytes).expect("PTX must be valid UTF-8");
    insta::assert_snapshot!(ptx);
}
