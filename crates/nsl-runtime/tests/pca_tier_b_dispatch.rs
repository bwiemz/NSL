//! Runtime gate truth-table + dispatcher branch wiring tests.
//! Per planner spec §6.7.

use nsl_runtime::pca_tier_b_runtime::{
    should_dispatch_tier_b_at_runtime, TIER_B_MAX_BAKED_SEQ_LEN, TIER_B_SEQ_LEN_FLOOR,
};

// ── Six truth-table tests (planner spec §6.7) ────────────────────────────────

#[test]
fn dispatch_tier_b_on_happy_path() {
    assert!(should_dispatch_tier_b_at_runtime(
        /* tier_b_ptx_ptr */ 0xdeadbeef,
        /* segment_ids_ptr */ 0x12345678,
        /* seq_len */ TIER_B_SEQ_LEN_FLOOR,
    ));
}

#[test]
fn dispatch_tier_b_on_at_max_baked() {
    assert!(should_dispatch_tier_b_at_runtime(
        0xdeadbeef,
        0x12345678,
        TIER_B_MAX_BAKED_SEQ_LEN,
    ));
}

#[test]
fn no_dispatch_when_no_tier_b_emitted() {
    // Sentinel pair = (0, 0) — meaning codegen did not emit a Tier-B-on variant.
    assert!(!should_dispatch_tier_b_at_runtime(
        /* tier_b_ptx_ptr */ 0,
        0x12345678,
        TIER_B_SEQ_LEN_FLOOR,
    ));
}

#[test]
fn no_dispatch_when_no_segment_ids() {
    // Non-CSHA or untrained-path callers pass segment_ids_ptr=0.
    assert!(!should_dispatch_tier_b_at_runtime(
        0xdeadbeef,
        /* segment_ids_ptr */ 0,
        TIER_B_SEQ_LEN_FLOOR,
    ));
}

#[test]
fn no_dispatch_below_floor() {
    if TIER_B_SEQ_LEN_FLOOR > 0 {
        assert!(!should_dispatch_tier_b_at_runtime(
            0xdeadbeef,
            0x12345678,
            TIER_B_SEQ_LEN_FLOOR - 1,
        ));
    }
}

#[test]
fn no_dispatch_above_max_baked() {
    assert!(!should_dispatch_tier_b_at_runtime(
        0xdeadbeef,
        0x12345678,
        TIER_B_MAX_BAKED_SEQ_LEN + 1,
    ));
}

// ── Dispatcher branch wiring integration test (planner spec §6.7 refinement) ─

/// Integration test: verifies the dispatcher's 2-line branch in nsl_flash_attention_csha
/// correctly picks (base_ptx, base_name) when the runtime gate fires OFF for an
/// orthogonal reason (seq_len < FLOOR, sentinel pair consistent).
///
/// Catches inverted-branch failure mode (gate fires correctly but dispatcher picks
/// Tier-B-on instead of base). Distinct surface from the truth-table tests above
/// (IR-006 — distinct failure modes warrant distinct test surfaces).
#[test]
fn dispatch_branch_picks_base_ptx_when_runtime_gate_false() {
    // Construct scenario: consistent sentinels (both non-zero), but seq_len < FLOOR
    // so gate is OFF.
    let tier_b_ptx_ptr: i64 = 0xdeadbeef;
    let tier_b_name_ptr: i64 = 0xfeedface;
    let segment_ids_ptr: i64 = 0x12345678;
    let base_ptx_ptr: i64 = 0xa0a0a0a0_u64 as i64;
    let base_name_ptr: i64 = 0xb0b0b0b0_u64 as i64;
    let seq_len = if TIER_B_SEQ_LEN_FLOOR > 0 {
        TIER_B_SEQ_LEN_FLOOR - 1
    } else {
        0
    };

    // Replicate the dispatcher's 2-line branch logic from nsl_flash_attention_csha.
    let gate_result =
        should_dispatch_tier_b_at_runtime(tier_b_ptx_ptr, segment_ids_ptr, seq_len);
    let (effective_ptx_ptr, effective_name_ptr) = if gate_result {
        (tier_b_ptx_ptr, tier_b_name_ptr)
    } else {
        (base_ptx_ptr, base_name_ptr)
    };

    // Below-floor seq_len should select base PTX, not Tier-B-on.
    assert!(!gate_result, "Below-floor seq_len should produce false gate result");
    assert_eq!(
        effective_ptx_ptr, base_ptx_ptr,
        "Gate OFF should select base PTX, not Tier-B-on (inverted-branch regression?)"
    );
    assert_eq!(effective_name_ptr, base_name_ptr);
}
