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
use nsl_codegen::flash_attention_v2::{synthesize_backward, synthesize_backward_with_tier_b};
use nsl_codegen::pca_segment::SegmentResidency;

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
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: true,
        csha: Some(CshaExtras {
            fused_projections: true,
            save_activations_for_backward: true,
            d_model: 32,
            ..CshaExtras::default()
        }),
        checkpoint: None,
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
    // Post-Blackwell-fix: seg_smem is embedded at the TAIL of the extern
    // shmem[] allocation (the launcher passes `backward_total_bytes + 4096`
    // as shared_mem_bytes; the last 4096 bytes hold segment_ids).  A
    // separate `.shared .align 4 .b8 seg_smem[4096]` declaration
    // alongside `.extern .shared shmem[]` compiled cleanly on ptxas but
    // fired CUDA_ERROR_ILLEGAL_ADDRESS at runtime on sm_120 (RTX 5070
    // Ti).  The structural evidence that the 4096-byte region is
    // reserved is the `%seg_base = %shmem_base + bwd_total` setup,
    // which only appears when segment_masked is active.  The comment
    // tail `seg_smem starts here` is also pinned so a future refactor
    // that silently drops the setup is caught by test failure.
    assert!(
        ptx.contains("%seg_base"),
        "backward must set up %seg_base (tail-of-shmem seg_smem region)"
    );
    assert!(
        ptx.contains("seg_smem starts here"),
        "backward %seg_base setup must be annotated so the seg_smem \
         tail-embedding convention is self-documenting in emitted PTX"
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

// ── Tier B.2 no-op guarantee (Task B2-2) ──────────────────────────────────
//
// `synthesize_backward(config)` (1-arg form) must remain byte-identical to
// the pre-B.2 baseline. Verified by checking it has no Tier B markers AND
// equals the explicit `synthesize_backward_with_tier_b(config, None)` form.

#[test]
fn backward_kernel_tier_b_none_is_byte_identical_to_1arg() {
    let cfg = minimal_segment_masked_backward_config();
    let via_1arg = synthesize_backward(&cfg).expect("synthesize_backward must succeed");
    let via_none = synthesize_backward_with_tier_b(&cfg, None)
        .expect("synthesize_backward_with_tier_b(None) must succeed");
    assert_eq!(
        via_1arg, via_none,
        "synthesize_backward_with_tier_b(cfg, None) must be byte-identical to \
         synthesize_backward(cfg) — Tier B.2 no-op guarantee violated"
    );
}

#[test]
fn backward_kernel_tier_b_off_has_no_tier_b_markers() {
    let cfg = minimal_segment_masked_backward_config();
    let ptx = synthesize_backward(&cfg).expect("synthesize_backward must succeed");
    assert!(
        !ptx.contains("PCA Tier B: range-table preamble"),
        "backward kernel must not contain Tier B preamble when tier_b=None"
    );
    // `BWD_KV_TILE_SKIP_TB_` literally contains `KV_TILE_SKIP_TB_` as a
    // substring, so use newline-anchored / explicit-prefix checks here.
    assert!(
        !ptx.contains("\nKV_TILE_SKIP_TB_"),
        "backward kernel must not contain forward's KV_TILE_SKIP_TB labels"
    );
    assert!(
        !ptx.contains("BWD_KV_TILE_SKIP_TB_"),
        "backward kernel must not contain BWD_KV_TILE_SKIP_TB labels when tier_b=None"
    );
}

// ── Tier B.2 ON: backward kernel snapshot + structural assertions ─────────

// Snapshot is baselined for the production (no-instrumentation) PTX. With
// `debug_kernel_instrumentation` the round-robin M3 writeback is appended
// inside the predicate scope; that path is covered separately by the
// forward-predicate-isolation snapshot. Gate this test off when the
// feature is enabled to avoid spurious churn.
#[test]
#[cfg(not(feature = "debug_kernel_instrumentation"))]
fn backward_kernel_segment_masked_tier_b_on_causal_32_32_32() {
    let cfg = minimal_segment_masked_backward_config();
    let seq_len: u32 = 4096;
    let residency = SegmentResidency::Shared;
    let ptx = synthesize_backward_with_tier_b(&cfg, Some((seq_len, residency)))
        .expect("synthesize_backward_with_tier_b must succeed");

    // Structural assertions — Tier B.2 additions must be present.
    assert!(
        ptx.contains("PCA Tier B: range-table preamble"),
        "backward prelude must contain Tier B range-table preamble"
    );
    assert!(
        ptx.contains("BWD_KV_TILE_SKIP_TB_0:"),
        "BWD_KV_TILE_SKIP_TB_0 label must appear in backward Tier-B-on kernel"
    );
    assert!(
        ptx.contains("kv-outer"),
        "predicate body must advertise its KV-outer direction in the comment"
    );
    // Forward namespace must NOT leak into backward. Use newline-anchored
    // search because `BWD_KV_TILE_SKIP_TB_0:` contains the literal substring
    // `KV_TILE_SKIP_TB_0:`.
    assert!(
        !ptx.contains("\nKV_TILE_SKIP_TB_0:"),
        "backward kernel must not carry forward's KV_TILE_SKIP_TB_0 label"
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
