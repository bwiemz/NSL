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
        tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false, csha: None,
        checkpoint: None,
    }
}

fn non_csha_canonical() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64, block_kv: 64, head_dim: 128,
        causal: true, paged: false, rope_q: false,
        rope_style: RopeStyle::HalfSplit, gqa_group_size: 1,
        tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false, csha: None,
        checkpoint: None,
    }
}

#[test]
fn phase_prelude__32x32x32_snapshot() {
    let mut ptx = String::new();
    prelude::emit(&mut ptx, &csha_canonical(), None);
    insta::assert_snapshot!("phase_prelude__32x32x32", ptx);
}

#[test]
fn phase_prelude__64x64x128_snapshot() {
    let mut ptx = String::new();
    prelude::emit(&mut ptx, &non_csha_canonical(), None);
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
    s_compute::emit(&mut ptx, &csha_canonical(), 0, None);
    insta::assert_snapshot!("phase_s_compute__32x32x32_causal_iter0", ptx);
}

#[test]
fn phase_s_compute__64x64x128_causal_snapshot() {
    let mut ptx = String::new();
    s_compute::emit(&mut ptx, &non_csha_canonical(), 0, None);
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
    s_compute::emit(&mut ptx0, &csha_canonical(), 0, None);
    s_compute::emit(&mut ptx1, &csha_canonical(), 1, None);
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
        tree_mask: false, num_sink_tokens: 0, gpu_sm: 75, segment_masked: false, csha: Some(CshaExtras::level2(1e-5, 32)),
        checkpoint: None,
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

// ---------------------------------------------------------------------------
// CSHA paper §5.2 v1 audit pins (Sprint 2 cycle-2)
//
// The two tests below codify the v1 dead-head-elimination envelope per the
// in-source doc on `csha_hooks::emit_active_heads_guard`:
//   * `csha = None`        -> comment-only, no PTX instructions / labels.
//   * `csha = Some(...)`   -> sentinel-aware runtime predicate against
//                              the `csha_active_heads` kernel param.
//
// They are structural string asserts (not insta snapshots) so they trip
// independently of any future PTX whitespace / comment churn that the
// existing snapshot would absorb on `cargo insta accept`. Together with
// the `a4_grid_y_*` unit tests in `nsl-runtime/src/flash_attention.rs`
// (~line 4527), they pin both tiers — launcher truncation AND in-kernel
// guard — against silent regression.
// ---------------------------------------------------------------------------

#[test]
fn phase_csha_active_heads_guard__none_emits_comment_only() {
    // CSHA paper §5.2 v1: when no CSHA dispatch is in play, the guard
    // emitter is a no-op (single comment, zero instructions). Pinning
    // this prevents a future refactor from accidentally emitting the
    // guard prelude unconditionally and breaking non-CSHA kernels with
    // an `ld.param.u32 [csha_active_heads]` against a missing param.
    let mut ptx = String::new();
    csha_hooks::emit_active_heads_guard(&mut ptx, &csha_canonical());

    // Must contain the no-emission comment exactly once.
    assert!(
        ptx.contains("CSHA A.4 active_heads guard: csha=None, no emission"),
        "csha=None must emit the no-emission comment; got:\n{ptx}"
    );
    // Must NOT emit any of the active-path PTX instructions or label.
    assert!(
        !ptx.contains("ld.param.u32"),
        "csha=None must not emit param load; got:\n{ptx}"
    );
    assert!(
        !ptx.contains("setp."),
        "csha=None must not emit any setp; got:\n{ptx}"
    );
    assert!(
        !ptx.contains("@%p0 ret"),
        "csha=None must not emit conditional ret; got:\n{ptx}"
    );
    assert!(
        !ptx.contains("V2_CSHA_ACTIVE_HEADS_SKIP"),
        "csha=None must not emit the skip label; got:\n{ptx}"
    );
}

#[test]
fn phase_csha_active_heads_guard__some_emits_param_based_predicate() {
    // CSHA paper §5.2 v1: in the v2-hooks variant the guard predicate
    // lives in the kernel PARAM `csha_active_heads`, NOT in a compile-
    // time literal. This is what lets a single emitted kernel handle
    // any pruning count the launcher supplies, including the
    // active_heads=0 "all heads live" sentinel (handled via the
    // sentinel-zero skip branch).
    let mut ptx = String::new();
    csha_hooks::emit_active_heads_guard(&mut ptx, &csha_l2_rope_config());

    // Param load: this is the v2 contract — predicate is runtime-driven.
    assert!(
        ptx.contains("ld.param.u32 %r10, [csha_active_heads];"),
        "guard must load the csha_active_heads kernel param; got:\n{ptx}"
    );
    // Sentinel-zero skip: param==0 means "all heads live", bypass the
    // head-index check. Without this, the launcher contract for
    // active_heads=0 would mis-eject every head.
    assert!(
        ptx.contains("setp.eq.u32 %p0, %r10, 0;"),
        "guard must emit the sentinel-zero predicate; got:\n{ptx}"
    );
    assert!(
        ptx.contains("@%p0 bra V2_CSHA_ACTIVE_HEADS_SKIP;"),
        "guard must branch over the head-index check on sentinel-zero; got:\n{ptx}"
    );
    // Head-index >= active_heads -> early-exit.
    assert!(
        ptx.contains("cvt.u32.u64 %r11, %head_idx;"),
        "guard must materialise head_idx as u32 for compare; got:\n{ptx}"
    );
    assert!(
        ptx.contains("setp.ge.u32 %p0, %r11, %r10;"),
        "guard must compare head_idx >= csha_active_heads; got:\n{ptx}"
    );
    assert!(
        ptx.contains("@%p0 ret;"),
        "guard must conditionally ret on the head-index check; got:\n{ptx}"
    );
    // Skip label closes the block.
    assert!(
        ptx.contains("V2_CSHA_ACTIVE_HEADS_SKIP:"),
        "guard must emit the skip label; got:\n{ptx}"
    );
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
