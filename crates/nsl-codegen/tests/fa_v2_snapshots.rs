//! Per-phase snapshot tests. Each test emits a single phase against a
//! fixed config and diffs the generated PTX string against a stored
//! snapshot. Use `cargo insta review` to accept snapshot changes.

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::phases::{prelude, q_load, s_compute, softmax, pv_accum, finalize, csha_hooks};

fn csha_canonical() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: 32,
        causal: true, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 75, segment_masked: false, csha: None,
    }
}

fn non_csha_canonical() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 128,
        causal: true, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, gpu_sm: 75, segment_masked: false, csha: None,
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

#[test]
fn phase_s_compute__32x32x32_causal_snapshot() {
    let mut ptx = String::new();
    s_compute::emit(&mut ptx, &csha_canonical(), 0);
    insta::assert_snapshot!("phase_s_compute__32x32x32_causal_iter0", ptx);
}

#[test]
fn phase_s_compute__64x64x128_causal_snapshot() {
    let mut ptx = String::new();
    s_compute::emit(&mut ptx, &non_csha_canonical(), 0);
    insta::assert_snapshot!("phase_s_compute__64x64x128_causal_iter0", ptx);
}

/// Regression test: the k-loop label must be parameterised on
/// `q_tile_iter` so the orchestrator (Task 11) can call emit() multiple
/// times for `block_q > 4` without producing duplicate labels that
/// ptxas would reject.
#[test]
fn phase_s_compute__label_uniqueness_across_iters() {
    let mut ptx0 = String::new();
    let mut ptx1 = String::new();
    s_compute::emit(&mut ptx0, &csha_canonical(), 0);
    s_compute::emit(&mut ptx1, &csha_canonical(), 1);
    assert!(ptx0.contains("V2_LOOP_S_OVER_K_0:"), "iter 0 label missing");
    assert!(ptx1.contains("V2_LOOP_S_OVER_K_1:"), "iter 1 label missing");
    assert!(!ptx0.contains("V2_LOOP_S_OVER_K_1"), "iter 0 leaks iter 1 label");
    assert!(!ptx1.contains("V2_LOOP_S_OVER_K_0"), "iter 1 leaks iter 0 label");
}

#[test]
fn phase_softmax__32x32x32_snapshot() {
    let mut ptx = String::new();
    softmax::emit(&mut ptx, &csha_canonical(), 0);
    insta::assert_snapshot!("phase_softmax__32x32x32", ptx);
}

#[test]
fn phase_softmax__64x64x128_snapshot() {
    let mut ptx = String::new();
    softmax::emit(&mut ptx, &non_csha_canonical(), 0);
    insta::assert_snapshot!("phase_softmax__64x64x128", ptx);
}

#[test]
fn phase_pv_accum__32x32x32_snapshot() {
    let mut ptx = String::new();
    pv_accum::emit(&mut ptx, &csha_canonical(), 0);
    insta::assert_snapshot!("phase_pv_accum__32x32x32_iter0", ptx);
}

#[test]
fn phase_pv_accum__64x64x128_snapshot() {
    let mut ptx = String::new();
    pv_accum::emit(&mut ptx, &non_csha_canonical(), 0);
    insta::assert_snapshot!("phase_pv_accum__64x64x128_iter0", ptx);
}

#[test]
fn phase_finalize__32x32x32_snapshot() {
    let mut ptx = String::new();
    finalize::emit(&mut ptx, &csha_canonical(), 0);
    insta::assert_snapshot!("phase_finalize__32x32x32_iter0", ptx);
}

#[test]
fn phase_finalize__64x64x128_snapshot() {
    let mut ptx = String::new();
    finalize::emit(&mut ptx, &non_csha_canonical(), 0);
    insta::assert_snapshot!("phase_finalize__64x64x128_iter0", ptx);
}

/// Regression test: the k-loop label must be parameterised on
/// `q_tile_iter` so the orchestrator can call emit() multiple times
/// without producing duplicate labels ptxas would reject.
/// (Same pattern as phase_s_compute__label_uniqueness_across_iters.)
#[test]
fn phase_pv_accum__label_uniqueness_across_iters() {
    let mut ptx0 = String::new();
    let mut ptx1 = String::new();
    pv_accum::emit(&mut ptx0, &csha_canonical(), 0);
    pv_accum::emit(&mut ptx1, &csha_canonical(), 1);
    assert!(ptx0.contains("V2_LOOP_PV_OVER_K_0:"), "iter 0 label missing");
    assert!(ptx1.contains("V2_LOOP_PV_OVER_K_1:"), "iter 1 label missing");
    assert!(!ptx0.contains("V2_LOOP_PV_OVER_K_1"), "iter 0 leaks iter 1 label");
    assert!(!ptx1.contains("V2_LOOP_PV_OVER_K_0"), "iter 1 leaks iter 0 label");
}

// ---------------------------------------------------------------------------
// CSHA Tier A hook tests (Task 10)
// ---------------------------------------------------------------------------

fn csha_l2_rope_config() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32, block_kv: 32, head_dim: 32,
        causal: true, paged: false, rope_q: true,
        rope_style: RopeStyle::Adjacent, gqa_group_size: 1,  // emit_rope_pair_sweep implements Adjacent
        tree_mask: false, gpu_sm: 75, segment_masked: false, csha: Some(CshaExtras::level2(1e-5, 32)),
    }
}

#[test]
fn phase_csha_hooks__prologue_null_config() {
    // With csha: None, emit_prologue should produce empty / comment-only output.
    let mut ptx = String::new();
    csha_hooks::emit_prologue(&mut ptx, &csha_canonical(), 0);
    insta::assert_snapshot!("phase_csha_prologue__null", ptx);
}

#[test]
fn phase_csha_hooks__prologue_active_l2_rope() {
    let mut ptx = String::new();
    csha_hooks::emit_prologue(&mut ptx, &csha_l2_rope_config(), 0);
    insta::assert_snapshot!("phase_csha_prologue__l2_rope", ptx);
}

#[test]
fn phase_csha_hooks__projection_skeleton_l2_rope() {
    let mut ptx = String::new();
    csha_hooks::emit_matmul_projection(&mut ptx, &csha_l2_rope_config(), 0);
    insta::assert_snapshot!("phase_csha_projection__l2_rope_skeleton", ptx);
}

#[test]
fn phase_csha_hooks__epilogue_l2_rope() {
    let mut ptx = String::new();
    csha_hooks::emit_rope_epilogue(&mut ptx, &csha_l2_rope_config(), 0);
    insta::assert_snapshot!("phase_csha_epilogue__l2_rope", ptx);
}

#[test]
fn phase_csha_hooks__active_heads_guard() {
    let mut ptx = String::new();
    csha_hooks::emit_active_heads_guard(&mut ptx, &csha_l2_rope_config());
    insta::assert_snapshot!("phase_csha_active_heads_guard", ptx);
}

/// Label uniqueness regression: each CSHA hook's skip-label must be
/// parameterised on `q_tile_iter` so the orchestrator can call each
/// hook multiple times for block_q > 4 configs.
#[test]
fn phase_csha_hooks__label_uniqueness_across_iters() {
    let cfg = csha_l2_rope_config();
    let mut prologue0 = String::new();
    let mut prologue1 = String::new();
    csha_hooks::emit_prologue(&mut prologue0, &cfg, 0);
    csha_hooks::emit_prologue(&mut prologue1, &cfg, 1);
    assert!(prologue0.contains("V2_CSHA_PROLOGUE_SKIP_0"), "prologue iter 0 label missing");
    assert!(prologue1.contains("V2_CSHA_PROLOGUE_SKIP_1"), "prologue iter 1 label missing");
    assert!(!prologue0.contains("V2_CSHA_PROLOGUE_SKIP_1"), "prologue iter 0 leaks iter 1");

    let mut proj0 = String::new();
    let mut proj1 = String::new();
    csha_hooks::emit_matmul_projection(&mut proj0, &cfg, 0);
    csha_hooks::emit_matmul_projection(&mut proj1, &cfg, 1);
    assert!(proj0.contains("V2_CSHA_PROJECTION_SKIP_0"), "projection iter 0 label missing");
    assert!(proj1.contains("V2_CSHA_PROJECTION_SKIP_1"), "projection iter 1 label missing");
    assert!(!proj0.contains("V2_CSHA_PROJECTION_SKIP_1"), "projection iter 0 leaks iter 1");

    let mut epi0 = String::new();
    let mut epi1 = String::new();
    csha_hooks::emit_rope_epilogue(&mut epi0, &cfg, 0);
    csha_hooks::emit_rope_epilogue(&mut epi1, &cfg, 1);
    // emit_rope_epilogue uses V2_CSHA_ROPE_SKIP_* (not V2_CSHA_EPILOGUE_SKIP_*).
    assert!(epi0.contains("V2_CSHA_ROPE_SKIP_0"), "epilogue iter 0 label missing");
    assert!(epi1.contains("V2_CSHA_ROPE_SKIP_1"), "epilogue iter 1 label missing");
    assert!(!epi0.contains("V2_CSHA_ROPE_SKIP_1"), "epilogue iter 0 leaks iter 1");
}

// ---------------------------------------------------------------------------
// Full-kernel orchestrator snapshot tests (Task 11)
// ---------------------------------------------------------------------------

use nsl_codegen::flash_attention_v2::synthesize_flash_attention_ptx_v2;

#[test]
fn kernel_full__32x32x32_nocsha() {
    let ptx = synthesize_flash_attention_ptx_v2(&csha_canonical());
    let s = String::from_utf8(ptx).expect("PTX must be valid UTF-8");
    insta::assert_snapshot!("kernel_full__32x32x32_nocsha", s);
}

#[test]
fn kernel_full__64x64x128_nocsha() {
    let ptx = synthesize_flash_attention_ptx_v2(&non_csha_canonical());
    let s = String::from_utf8(ptx).expect("PTX must be valid UTF-8");
    insta::assert_snapshot!("kernel_full__64x64x128_nocsha", s);
}

#[test]
fn kernel_full__32x32x32_csha_l2_rope() {
    let ptx = synthesize_flash_attention_ptx_v2(&csha_l2_rope_config());
    let s = String::from_utf8(ptx).expect("PTX must be valid UTF-8");
    insta::assert_snapshot!("kernel_full__32x32x32_csha_l2_rope", s);
}
