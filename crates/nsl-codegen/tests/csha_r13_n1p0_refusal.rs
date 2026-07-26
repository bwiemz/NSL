//! R13 refusal pin: `fused_rmsnorm=true` + `fused_projections=false` (the
//! kernel-name suffix `n1_p0`) must be REFUSED by the fused CSHA backward.
//!
//! Measured 2026-07-25 on RTX 5070 Ti (sm_120), one process per cell, in
//! `csha_cycle15_bug1_ablations::d1..d5`:
//!
//!                    rope_q=true        rope_q=false
//!     S=512          ILLEGAL_ADDRESS    zero gradients
//!     S=32           ILLEGAL_ADDRESS    zero gradients
//!     n1_p1 control  PASS               --
//!
//! RoPE-Q reads cos/sin staged into SMEM by the fused-projection prologue;
//! without that prologue the read lands at the shared-window base and the
//! FORWARD kernel faults. With rope_q off there is no fault but the gradients
//! come back all-zero — the silent arm, which is the one worth a gate.
//!
//! Like R7, this refusal is load-bearing and invisible: nothing else fails if
//! it is deleted. It is defense-in-depth for hand-built configs that call
//! `synthesize_backward_with_tier_b` directly, since production is already
//! protected one layer up by `tier_b2_can_dispatch` rejecting `level < 2`.
//! That means a regression here would be silent in exactly the way the
//! original bug was.
//!
//! Costs nothing to run: pure synthesis, no device required.

#![cfg(feature = "cuda")]

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::synthesize_backward_with_tier_b;

fn cfg(fused_projections: bool, rope_q: bool, level: u8) -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 32,
        block_kv: 32,
        head_dim: 64,
        causal: true,
        paged: false,
        rope_q,
        rope_style: RopeStyle::Adjacent,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some({
            let mut e = CshaExtras::level1_with_fused_proj(1e-6);
            e.level = level;
            e.fused_rmsnorm = true;
            e.fused_projections = fused_projections;
            e.d_model = 64;
            e
        }),
        checkpoint: None,
    }
}

#[test]
fn n1p0_with_rope_q_is_refused() {
    let err = synthesize_backward_with_tier_b(&cfg(false, true, 1), None)
        .expect_err("n1_p0 + rope_q must be refused, not synthesized");
    assert!(
        err.contains("n1_p0") && err.contains("fused_projections"),
        "refusal must name the composition; got: {err}"
    );
}

#[test]
fn n1p0_without_rope_q_is_also_refused() {
    // The arm that matters most. With rope_q off there is no crash to notice —
    // the kernel simply returns zeros — so this is the case a future edit is
    // most likely to "fix" by relaxing the refusal.
    let err = synthesize_backward_with_tier_b(&cfg(false, false, 1), None)
        .expect_err("n1_p0 must be refused even without rope_q (zero gradients)");
    assert!(
        err.contains("n1_p0"),
        "refusal must name the composition; got: {err}"
    );
}

#[test]
fn n1p1_still_synthesizes() {
    // ANTI-VACUITY: the refusal must key on the FLAG PAIR, not on level. A
    // level==1 test would pass the two assertions above while silently
    // refusing `level1_with_fused_proj`, which has passing GPU gates
    // (csha_cycle15_bug1_ablations::a0/a1/a2).
    synthesize_backward_with_tier_b(&cfg(true, true, 1), None)
        .expect("level=1 WITH fused_projections (n1_p1) must still synthesize");
}

#[test]
fn level2_with_both_flags_still_synthesizes() {
    // ANTI-VACUITY, other axis: the refusal must not catch the levels that
    // production actually dispatches to (tier_b2_can_dispatch needs level >= 2).
    synthesize_backward_with_tier_b(&cfg(true, true, 2), None)
        .expect("level=2 with both fusion flags must still synthesize");
}
