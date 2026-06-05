//! Snapshot the emitted Q-projection PTX for canonical Tier B.1 configs.
//!
//! The snapshot locks down (per spec §6.5):
//!   * The m16n8k16 mma.sync sequence (1 per `(chunk_idx, tile_t)` pair).
//!   * The chunk loop structure (n_chunks = d_model / chunk).
//!   * The null-guard placement (csha_x_ptr + csha_wq_ptr -> SKIP label).
//!   * The lane-coherent scatter pattern (cvt.rn.f16.f32 + st.shared.b16).
//!
//! Reviewer can verify §3.4 + §5.5 + §6.5 by inspection of the diff —
//! the snapshot is the load-bearing FSM gate, not an opaque blob.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b1::projection_mma::emit_q_projection;

fn canonical_config(bq: i64, bkv: i64, hd: i64, dm: u32) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: bq,
        block_kv: bkv,
        head_dim: hd,
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
            d_model: dm,
            ..CshaExtras::default()
        }),
    }
}

#[test]
fn q_projection_canonical_32x32x32_dm2048_chunk128() {
    // V3 supported-matrix CSV: 32,32,32,2048,true -> admitted at chunk=128.
    // Expected: n_chunks=16, tiles_per_warp=1, MMA count=16.
    let cfg = canonical_config(32, 32, 32, 2048);
    let mut ptx = String::new();
    emit_q_projection(&mut ptx, &cfg, 128);
    insta::assert_snapshot!(ptx);
}

#[test]
fn q_projection_canonical_64x64x64_dm2048_chunk64() {
    // V3 CSV: 64,64,64,2048,true -> admitted at chunk=64.
    // Expected: n_chunks=32, tiles_per_warp=4, MMA count=128.
    let cfg = canonical_config(64, 64, 64, 2048);
    let mut ptx = String::new();
    emit_q_projection(&mut ptx, &cfg, 64);
    insta::assert_snapshot!(ptx);
}
