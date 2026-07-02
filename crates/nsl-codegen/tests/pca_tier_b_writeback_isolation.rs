//! Isolation snapshot test for the round-robin owning_warp writeback.
//!
//! Spec §6.2 of `docs/superpowers/specs/2026-05-13-pca-tier-b15-and-b2-design.md`:
//! `owning_warp(qt, kvt) = (qt * num_kv_tiles + kvt) % num_warps`.
//!
//! This is a UNIT test for `emit_skip_decision_writeback` (no kernel
//! integration, no PTX synthesis). It pins:
//!   1. The presence of warp-id derivation (`shr.u32` + `%warp_id`) —
//!      i.e. NOT the v1 warp-0-always shape.
//!   2. The presence of `rem.u32` (the modulus) — non-power-of-two
//!      `num_warps` values use rem.
//!   3. The PTX lexical scope wrapper `{ ... }` per IR-007.
//!
//! Diffing this snapshot against the v1 baseline (warp-0-always) makes
//! the round-robin migration visible.

use nsl_codegen::flash_attention::{FlashAttentionConfig, RopeStyle};
use nsl_codegen::pca_tilerange;

fn gate_fixture_cfg() -> FlashAttentionConfig {
    FlashAttentionConfig {
        block_q: 64,
        block_kv: 64,
        head_dim: 64,
        causal: true,
        paged: false,
        rope_q: false,
        rope_style: RopeStyle::HalfSplit,
        gqa_group_size: 1,
        tree_mask: false,
        num_sink_tokens: 0,
        gpu_sm: 120,
        segment_masked: true,
        csha: None,
        checkpoint: None,
    }
}

#[test]
fn round_robin_writeback_pattern_present() {
    let mut ptx = String::new();
    pca_tilerange::emit_skip_decision_writeback(
        &mut ptx,
        &gate_fixture_cfg(),
        /* seq_len = */ 4096,
        /* qt_reg = */ "%qt",
        /* kvt_reg = */ "%kvt",
        /* is_skip_pred = */ "%p_skip_TB",
        /* decisions_buf_param = */ "skip_decisions_ptr",
        /* num_warps = */ 4,
    );

    // Sentinel 1: warp-id derivation present (NOT warp-0-always).
    assert!(
        ptx.contains("shr.u32") && ptx.contains("%warp_id_TB"),
        "round-robin requires warp_id derivation (shr.u32 + %warp_id_TB):\n{ptx}"
    );

    // Sentinel 2: rem.u32 with num_warps argument present.
    assert!(
        ptx.contains("rem.u32"),
        "round-robin uses rem.u32 for the modulus:\n{ptx}"
    );

    // Sentinel 3: PTX lexical scope per IR-007.
    let trimmed_start = ptx.trim_start();
    assert!(
        trimmed_start.starts_with("// ----- PCA Tier B: skip-decision writeback")
            || ptx.contains("\n    {\n"),
        "writeback must be wrapped in PTX lexical scope per IR-007:\n{ptx}"
    );

    insta::assert_snapshot!("writeback_round_robin_gate_fixture", ptx);
}

#[test]
fn round_robin_degrades_to_warp_zero_when_num_warps_is_one() {
    // Per spec §6.4: num_warps=1 makes `(...) % 1 == 0` for every tile,
    // and `warp_id == 0` is always true → every tile is written by warp 0
    // lane 0, identical to the v1 warp-0 behaviour. The PTX still includes
    // the round-robin shape (rem with literal 1); ptxas folds it at
    // compile time.
    let mut ptx = String::new();
    pca_tilerange::emit_skip_decision_writeback(
        &mut ptx,
        &gate_fixture_cfg(),
        4096,
        "%qt",
        "%kvt",
        "%p_skip_TB",
        "skip_decisions_ptr",
        /* num_warps = */ 1,
    );
    assert!(
        ptx.contains("rem.u32 %owning_warp_TB, %owning_warp_TB, 1"),
        "num_warps=1 still emits rem with literal 1 (ptxas folds):\n{ptx}"
    );
}

#[test]
#[cfg(debug_assertions)]
#[should_panic(expected = "num_warps")]
fn num_warps_zero_trips_debug_assert() {
    // Per spec §6.4: num_warps=0 is invalid (UB on most architectures).
    // The Rust-side debug_assert! refuses at codegen time before the PTX
    // is emitted. Only meaningful under debug builds — release builds
    // strip debug_assert!, so we gate this test on `debug_assertions`.
    let mut ptx = String::new();
    pca_tilerange::emit_skip_decision_writeback(
        &mut ptx,
        &gate_fixture_cfg(),
        4096,
        "%qt",
        "%kvt",
        "%p_skip_TB",
        "skip_decisions_ptr",
        /* num_warps = */ 0,
    );
}
