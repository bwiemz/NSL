//! Snapshot of the full Tier C backward FA2 kernel with segment_masked=true.
//!
//! Paired with pca_forward_kernel_snapshot (Task 3B) to pin the end-to-end
//! emission for PCA's backward path. The snapshot captures:
//!   - `param .u64 segment_ids_ptr` at the end of the entry param list
//!   - BW_PCA_LOAD_LOOP / BW_PCA_LOAD_DONE cooperative load block
//!   - bar.sync 0 after the SMEM load
//!   - In ds_compute: setp.gt.u32 %p1 followed by the segment_mask
//!     helper body, then @%p1 mov.f32 %f_P, ...
//!   - .shared .align 4 .b8 seg_smem[4096] inside the function body

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_backward;

fn minimal_segment_masked_backward_config() -> FlashAttentionConfig {
    // Minimal config that passes the backward validator:
    //   block_q=block_kv=32 (fused_projections requires equality),
    //   head_dim=32 (smallest allowed), causal=true, segment_masked=true.
    // CshaExtras: fused_projections=true (required by validator),
    //             save_activations_for_backward=true (required by backward path).
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
        gpu_sm: 80,
        segment_masked: true,
        csha: Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 32,
            ..CshaExtras::default()
        }),
    }
}

#[test]
fn backward_kernel_segment_masked_causal_32_32_32() {
    let cfg = minimal_segment_masked_backward_config();
    let ptx = synthesize_backward(&cfg).expect("synthesize_backward must succeed");

    // Structural assertions that must hold regardless of snapshot acceptance.
    assert!(
        ptx.contains("segment_ids_ptr"),
        "param block must declare segment_ids_ptr"
    );
    assert!(
        ptx.contains("seg_smem[4096]"),
        "must declare 4096-byte seg_smem buffer"
    );
    assert!(
        ptx.contains("BW_PCA_LOAD_LOOP"),
        "backward load loop label must be BW_PCA_LOAD_LOOP (not forward's PCA_LOAD_LOOP)"
    );
    assert!(
        ptx.contains("BW_PCA_LOAD_DONE"),
        "backward load done label must be BW_PCA_LOAD_DONE"
    );
    assert!(
        ptx.contains("%rd_bw_q_global"),
        "backward-specific q_global scratch register must appear in ds_compute"
    );
    assert!(
        ptx.contains("%rd_bw_k_global"),
        "backward-specific k_global scratch register must appear in ds_compute"
    );
    // segment_mask helper body: the setp.ne that detects cross-segment cells.
    assert!(
        ptx.contains("setp.ne.u16"),
        "segment_mask helper must emit setp.ne.u16 for cross-segment detection"
    );
    // Mask predicate OR-extension: or.pred applied to %p1 (causal register).
    assert!(
        ptx.contains("or.pred        %p1, %p1, %p_seg_SEGMASK"),
        "segment mask must OR-extend causal predicate %p1"
    );

    insta::assert_snapshot!(ptx);
}

#[test]
fn write_ptx_to_file() {
    let cfg = minimal_segment_masked_backward_config();
    let ptx = synthesize_backward(&cfg).expect("synthesize_backward must succeed");
    let path = std::env::temp_dir().join("bwd_full.ptx");
    std::fs::write(&path, &ptx).expect("write failed");
    println!("PTX written to {} ({} bytes)", path.display(), ptx.len());
}
