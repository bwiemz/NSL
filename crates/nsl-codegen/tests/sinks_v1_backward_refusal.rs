//! Sprint 2 cycle-7 paper §4.3: defense-in-depth backward refusals for
//! attention sinks v1.
//!
//! Sprint 1b lifted the cycle-5 unconditional `num_sink_tokens > 0`
//! refusal for the narrow Tier A forward single-tile config. The
//! `attention_sinks_v1_eligible` predicate refuses
//! `csha.save_activations_for_backward = true` at the
//! `compiler/kernel.rs::attention_sink` decorator-extraction site —
//! the FORWARD front door.
//!
//! The cycle-5 `feedback_deferral_must_refuse` invariant says refusals
//! must be at EVERY entry point — not just the decorator. Sprint 2
//! covers the BACKWARD entry points:
//!
//!   1. `flash_attention_v2::synthesize_backward_with_tier_b`
//!      (transitively reached by `synthesize_backward`).
//!   2. `flash_attention_v2::synthesize_backward_combined`
//!      (hybrid 4-kernel module assembler).
//!   3. `flash_attention_v2::synthesize_backward_with_tier`
//!      (tier-dispatch wrapper).
//!   4. `flash_attention_v2::tier_b2::backward::synthesize_tier_b2_backward`
//!      (direct hybrid synthesis entry).
//!   5. `flash_attention_v2::tier_b2::dispatch::tier_b2_hybrid_backward_eligible`
//!      / `tier_b2_hybrid_backward_compile_time_eligible` (dispatch
//!      predicates that gate wengert lowering's runtime branch flag).
//!
//! Each test below constructs a `FlashAttentionConfig` with
//! `num_sink_tokens > 0` AND `csha.save_activations_for_backward = true`
//! (bypassing the kernel.rs front-door refusal) and asserts the
//! backward entry point refuses with a Sprint-2-naming message.
//!
//! Also pins the kernel.rs front-door cross-check (Task F): the
//! decorator-extraction site must refuse `save_activations_for_backward
//! = true` + `num_sink_tokens > 0` via the
//! `attention_sinks_v1_eligible` axis check, naming Sprint 2.

#![cfg(feature = "test-helpers")]

use nsl_codegen::flash_attention::{CshaExtras, FlashAttentionConfig, RopeStyle};
use nsl_codegen::flash_attention_v2::sinks::attention_sinks_v1_eligible;
use nsl_codegen::flash_attention_v2::tier_b2::backward::{
    synthesize_tier_b2_backward, BackwardSynthError,
};
use nsl_codegen::flash_attention_v2::tier_b2::dispatch::{
    tier_b2_hybrid_backward_compile_time_eligible, tier_b2_hybrid_backward_eligible,
};
use nsl_codegen::flash_attention_v2::{
    synthesize_backward, synthesize_backward_combined, synthesize_backward_with_tier,
    synthesize_backward_with_tier_b,
};

/// Build a config that bypasses the kernel.rs front-door refusal:
/// `num_sink_tokens > 0` AND `csha.save_activations_for_backward = true`.
/// This is the canonical "caller bypassed the front door" config — any
/// backward entry point reached with this config MUST refuse.
///
/// The base config is the Tier B.2 hybrid smoke intersection
/// (`head_dim=64, sm=80, level=2, heads=1, d_model=hd`) so the
/// REFUSAL is exercised against an otherwise-eligible config (proves
/// the refusal axis is sinks, not some other rejection layered on top).
fn sinks_enabled_bypass_cfg() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 64,
        causal: false,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 4,
        gpu_sm: 80,
        segment_masked: false,
        csha: Some(CshaExtras {
            level: 2,
            d_model: 64,
            active_heads: 1,
            save_activations_for_backward: true,
            ..CshaExtras::default()
        }),
    }
}

/// Same base config with `num_sink_tokens = 0` — the "sinks disabled
/// sentinel" path. Used to prove the refusal axis is sinks, not some
/// other regression on the base hybrid config.
fn zero_sinks_bypass_cfg() -> FlashAttentionConfig {
    FlashAttentionConfig {
        num_sink_tokens: 0,
        ..sinks_enabled_bypass_cfg()
    }
}

// -----------------------------------------------------------------
// Task B / C / E.1 — synthesize_backward_with_tier_b refusal
// -----------------------------------------------------------------

#[test]
fn synthesize_backward_with_tier_b_refuses_sinks_enabled() {
    let cfg = sinks_enabled_bypass_cfg();
    let err = synthesize_backward_with_tier_b(&cfg, None)
        .expect_err("must refuse sinks-enabled backward");
    assert!(
        err.contains("Sprint 2 cycle-7"),
        "refusal must name Sprint 2 cycle-7: '{err}'"
    );
    assert!(
        err.contains("v2") || err.contains("forward-only"),
        "refusal must cite v2 sprint or 'forward-only': '{err}'"
    );
}

/// `synthesize_backward` is a thin wrapper around
/// `synthesize_backward_with_tier_b(config, None)` — the refusal
/// propagates transitively.
#[test]
fn synthesize_backward_refuses_sinks_enabled_via_with_tier_b() {
    let cfg = sinks_enabled_bypass_cfg();
    let err = synthesize_backward(&cfg).expect_err("must refuse sinks-enabled backward");
    assert!(
        err.contains("Sprint 2 cycle-7"),
        "refusal must name Sprint 2 cycle-7: '{err}'"
    );
}

// -----------------------------------------------------------------
// Task B / E.2 — synthesize_backward_combined refusal
// -----------------------------------------------------------------

#[test]
fn synthesize_backward_combined_refuses_sinks_enabled() {
    let cfg = sinks_enabled_bypass_cfg();
    let err = synthesize_backward_combined(&cfg)
        .expect_err("must refuse sinks-enabled hybrid combined entry");
    assert!(
        err.contains("Sprint 2 cycle-7"),
        "refusal must name Sprint 2 cycle-7: '{err}'"
    );
    // The combined entry's refusal message names the entry point so
    // the user can tell which code path caught the bypass.
    assert!(
        err.contains("combined") || err.contains("v2") || err.contains("forward-only"),
        "refusal must name combined entry or v2/forward-only: '{err}'"
    );
}

// -----------------------------------------------------------------
// Task E.3 — synthesize_backward_with_tier refusal
// -----------------------------------------------------------------

#[test]
fn synthesize_backward_with_tier_refuses_sinks_enabled() {
    let cfg = sinks_enabled_bypass_cfg();
    let err = synthesize_backward_with_tier(&cfg)
        .expect_err("must refuse sinks-enabled at tier-dispatch entry");
    assert!(
        err.contains("Sprint 2 cycle-7"),
        "refusal must name Sprint 2 cycle-7: '{err}'"
    );
}

// -----------------------------------------------------------------
// Task C / E.4 — synthesize_tier_b2_backward refusal (direct hybrid)
// -----------------------------------------------------------------

#[test]
fn synthesize_tier_b2_backward_refuses_sinks_enabled() {
    let cfg = sinks_enabled_bypass_cfg();
    let err = synthesize_tier_b2_backward(&cfg)
        .expect_err("must refuse sinks-enabled Tier B.2 hybrid synthesis");
    match err {
        BackwardSynthError::UnsupportedConfig(msg) => {
            assert!(
                msg.contains("Sprint 2 cycle-7"),
                "refusal must name Sprint 2 cycle-7: '{msg}'"
            );
            assert!(
                msg.contains("v2") || msg.contains("forward-only"),
                "refusal must cite v2 sprint or 'forward-only': '{msg}'"
            );
        }
        other => panic!(
            "expected UnsupportedConfig refusal naming Sprint 2 cycle-7, got: {other:?}"
        ),
    }
}

// -----------------------------------------------------------------
// Task D / E.5 — dispatch predicate refusals (compile-time + runtime)
// -----------------------------------------------------------------

#[test]
fn tier_b2_hybrid_backward_eligible_returns_false_for_sinks() {
    let cfg = sinks_enabled_bypass_cfg();
    assert!(
        !tier_b2_hybrid_backward_eligible(&cfg, cfg.block_q as u32),
        "runtime hybrid eligibility must refuse sinks-enabled config"
    );
}

#[test]
fn tier_b2_hybrid_backward_compile_time_eligible_returns_false_for_sinks() {
    let cfg = sinks_enabled_bypass_cfg();
    assert!(
        !tier_b2_hybrid_backward_compile_time_eligible(&cfg),
        "compile-time hybrid eligibility must refuse sinks-enabled config"
    );
}

/// Cross-check: the SAME bypass config with `num_sink_tokens = 0` must
/// be ACCEPTED by the hybrid Tier B.2 dispatch predicates AND by the
/// hybrid synthesizer. This pins that the refusal axis is sinks, not
/// some other regression layered on top of the hybrid smoke base
/// config.
///
/// We deliberately do NOT check `synthesize_backward` here: at
/// head_dim=64, scalar v2's backward extra-bytes (P+dQ+dK+dV+V_in+drop)
/// exceeds the 99 KB SMEM cap (~140 KB), so scalar v2 is structurally
/// incapable of handling this config — which is why Tier B.2 hybrid
/// exists for production-scale configs. The scalar refusal proof at
/// `synthesize_backward_refuses_sinks_enabled_via_with_tier_b` already
/// uses this same config and reaches the sinks refusal BEFORE the SMEM
/// validator runs, so the refusal ordering is verified.
#[test]
fn zero_sinks_baseline_accepted_by_hybrid_entry_points() {
    let cfg = zero_sinks_bypass_cfg();
    assert!(
        tier_b2_hybrid_backward_compile_time_eligible(&cfg),
        "baseline compile-time predicate must accept zero-sinks config"
    );
    assert!(
        tier_b2_hybrid_backward_eligible(&cfg, cfg.block_q as u32),
        "baseline runtime predicate must accept zero-sinks config"
    );
    let _ = synthesize_tier_b2_backward(&cfg)
        .expect("baseline hybrid backward must accept zero-sinks");
}

// -----------------------------------------------------------------
// Task F — kernel.rs front-door cross-check
// -----------------------------------------------------------------

/// The decorator-extraction site at `compiler/kernel.rs::attention_sink`
/// calls `attention_sinks_v1_eligible` BEFORE any backward synthesis is
/// reached. That predicate's `save_activations_for_backward` axis must
/// refuse the combination `num_sink_tokens > 0` AND
/// `csha.save_activations_for_backward = true` — closing the gap that
/// Sprint 2's defense-in-depth then back-stops at the codegen layer.
///
/// This test pins the message contains both "save_activations_for_backward"
/// (the axis name) and "Sprint 2" (the future-sprint citation per
/// cycle-5 `feedback_deferral_must_refuse`).
#[test]
fn kernel_front_door_refuses_save_activations_with_sinks() {
    let cfg = sinks_enabled_bypass_cfg();
    let (eligible, why) = attention_sinks_v1_eligible(&cfg);
    assert!(!eligible, "front door must refuse save_activations + sinks");
    let msg = why.expect("blocking reason must be Some for ineligible");
    assert!(
        msg.contains("save_activations_for_backward"),
        "front door message must name the axis: '{msg}'"
    );
    assert!(
        msg.contains("Sprint 2"),
        "front door message must cite Sprint 2: '{msg}'"
    );
}

/// Cross-check: the front door accepts the SAME config at
/// `num_sink_tokens = 0` (sentinel short-circuit). Pins the axis exercise.
#[test]
fn kernel_front_door_accepts_save_activations_at_zero_sinks() {
    let cfg = zero_sinks_bypass_cfg();
    let (eligible, why) = attention_sinks_v1_eligible(&cfg);
    assert!(
        eligible,
        "zero-sinks sentinel must short-circuit even with save_activations_for_backward=true"
    );
    assert!(why.is_none());
}
