//! R7 refusal pin: `rope_q = true` + `@checkpoint` must be REFUSED.
//!
//! Path B (the kv-recompute backward) has a known GROSS numerical error at
//! multi-tile under `RopeStyle::Adjacent`, and `RopeStyle::HalfSplit` is not
//! implemented there at all — `emit_rope_k_epilogue` panics. Rather than emit
//! silently-wrong gradients for what is a perfectly ordinary training config
//! (RoPE + gradient checkpointing), `validate_checkpoint_eligibility` refuses
//! the composition outright.
//!
//! That refusal is load-bearing and invisible: nothing else fails if it is
//! removed. The four `csha_checkpoint_recompute_gpu` gates that used to cover
//! this ground now cannot run *because* of the refusal, so deleting it would
//! turn them from blocked into silently-wrong rather than into failures. This
//! file exists so the refusal itself has a gate.
//!
//! MEASURED 2026-07-26 (RTX 5070 Ti sm_120), R7 temporarily bypassed in a
//! local experiment, `t_recompute_hd64_s512_bq32` three-way oracle:
//!
//!     path A vs cpu_ref   dq 7.695e-3   dx 8.755e-2   (max|ref| ~4.4 / ~47)
//!     path B vs cpu_ref   dq 2.152e3    dx 1.485e4    max_rel up to 3.26e6
//!     path B vs path A    dq 2.152e3    dx 1.485e4
//!
//! Path B is 2-3 orders of magnitude LARGER than the reference values and
//! diverges from the working Path A by the same amount. Two other "gross
//! numerical error" claims in this subsystem turned out to be harness
//! artifacts with the `max_abs == max|ref|` (gpu==0) signature; this one is
//! not that, and is real. The refusal stays until Path B's math is fixed.
//!
//! Costs nothing to run: pure PTX synthesis, no device required.

#![cfg(feature = "cuda")]

use nsl_codegen::flash_attention::{
    CheckpointExtras, CshaExtras, FlashAttentionConfig, RopeStyle,
};
use nsl_codegen::flash_attention_v2::synthesize_backward_with_tier_b;

fn cfg(rope_q: bool, checkpoint: bool, segment_masked: bool) -> FlashAttentionConfig {
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
        segment_masked,
        csha: Some({
            let mut e = CshaExtras::level1_with_fused_proj(1e-6);
            e.d_model = 64;
            e
        }),
        checkpoint: checkpoint.then(CheckpointExtras::full),
    }
}

#[test]
fn rope_q_with_checkpoint_adjacent_now_synthesizes() {
    // R7 NARROWED 2026-07-26. This test previously asserted a refusal. Path
    // B's Adjacent kv-recompute numerics were two %q_start-vs-%k_start
    // confusions (x_norm row base, and the cos/sin index in the K RoPE
    // epilogue); both are fixed and the three-way oracle is GREEN at hd=64
    // for S=32/512/2048. Keeping the old assertion would have pinned a
    // refusal that no longer has a reason to exist.
    synthesize_backward_with_tier_b(&cfg(true, true, false), None)
        .expect("rope_q + @checkpoint with RopeStyle::Adjacent must now synthesize");
}

#[test]
fn rope_q_with_checkpoint_halfsplit_now_synthesizes() {
    // Was refused, because emit_rope_k_epilogue asserted Adjacent-only.
    // Both the forward sweep and the backward emit_drope now select the
    // element pairing from config.rope_style, and the CPU oracle became
    // style-aware in the same change, so HalfSplit is compared against a
    // HalfSplit reference. Three-way oracle GREEN at hd=64, S=32 and S=512.
    let mut c = cfg(true, true, false);
    c.rope_style = RopeStyle::HalfSplit;
    synthesize_backward_with_tier_b(&c, None)
        .expect("rope_q + @checkpoint + HalfSplit must now synthesize");
}

#[test]
fn rope_q_with_checkpoint_and_segment_masking_is_refused() {
    // The segment_masked arm has its own message (PCA packing / RoPE-Q
    // write-back index collision); both arms must refuse.
    let err = synthesize_backward_with_tier_b(&cfg(true, true, true), None)
        .expect_err("segment_masked + rope_q + @checkpoint must be refused");
    assert!(
        err.contains("PCA packing") || err.contains("rope_q=true"),
        "refusal must name the segment_masked composition; got: {err}"
    );
}

#[test]
fn checkpoint_without_rope_q_still_synthesizes() {
    // ANTI-VACUITY: the refusal must be specific to rope_q. If this ever
    // starts failing, R7 has widened into refusing all of @checkpoint and the
    // two assertions above would still pass while covering nothing.
    synthesize_backward_with_tier_b(&cfg(false, true, false), None)
        .expect("@checkpoint without rope_q must still synthesize");
}

#[test]
fn rope_q_without_checkpoint_still_synthesizes() {
    // ANTI-VACUITY, other axis: rope_q alone is fine — it is only the
    // composition with @checkpoint that is unsound today.
    synthesize_backward_with_tier_b(&cfg(true, false, false), None)
        .expect("rope_q without @checkpoint must still synthesize");
}
