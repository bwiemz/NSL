//! Snapshot the emitted K/V-projection PTX for canonical Tier B.1 configs.
//!
//! The snapshot locks down (per spec §6.5):
//!   * The m16n8k16 mma.sync sequence (2 per `(chunk_idx, tile_t)` pair --
//!     one for K MMA, one for V MMA).
//!   * The chunk loop structure (n_chunks = d_model / chunk; 3 cp.async per chunk).
//!   * The null-guard placement (csha_x_ptr + csha_wk_ptr + csha_wv_ptr -> SKIP label).
//!   * The ping/pong slot selection (slot parameter controls K/V SMEM target offsets).
//!   * The lane-coherent scatter pattern (cvt.rn.f16.f32 + st.shared.b16).
//!
//! Reviewer can verify §3.4 + §5.5 + §6.5 by inspection of the diff --
//! the snapshot is the load-bearing FSM gate, not an opaque blob.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::tier_b1::projection_mma::emit_kv_projection_chunk_loop;

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
fn kv_projection_canonical_32x32x32_dm2048_chunk128_slot0() {
    // V3 supported-matrix CSV: 32,32,32,2048,true -> admitted at chunk=128.
    // Expected: n_chunks=16, tiles_per_warp_kv=1, MMA count=32 (2*1*16).
    // slot=0: K -> k_offset_ping, V -> v_offset_ping.
    let cfg = canonical_config(32, 32, 32, 2048);
    let mut ptx = String::new();
    emit_kv_projection_chunk_loop(&mut ptx, &cfg, 128, 0, 0);
    insta::assert_snapshot!(ptx);
}

#[test]
fn kv_projection_canonical_32x32x32_dm2048_chunk128_slot1() {
    // Same config as above but slot=1: K -> k_offset_pong, V -> v_offset_pong.
    // The skip label changes from V2_TIER_B1_KV_PROJ_SKIP_0 to V2_TIER_B1_KV_PROJ_SKIP_1
    // and SMEM target offsets differ from slot=0. Snapshot locks both invariants.
    let cfg = canonical_config(32, 32, 32, 2048);
    let mut ptx = String::new();
    emit_kv_projection_chunk_loop(&mut ptx, &cfg, 128, 0, 1);
    insta::assert_snapshot!(ptx);
}
