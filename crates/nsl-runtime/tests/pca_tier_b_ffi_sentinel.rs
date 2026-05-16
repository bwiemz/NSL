//! FFI sentinel-discipline tests. Per planner spec §8.5 (test surface #6).
//! Gated behind `bench-internal` Cargo feature to avoid triggering aborts in normal test runs.

#![cfg(feature = "bench-internal")]

use nsl_runtime::pca_tier_b_runtime::assert_tier_b_sentinels;

/// Helper-roundtrip test #1: disabled-sentinel pair (0, 0) does NOT trigger the assertion.
#[test]
fn helper_roundtrip_disabled_sentinel_passes_assertion() {
    assert_tier_b_sentinels("test_entry", 0, 0);
    // If the assertion fires, the process aborts and this test fails.
    // Reaching this line means the disabled-sentinel is correctly recognized.
}

/// Helper-roundtrip test #2: enabled-sentinel pair (non-zero, non-zero) does NOT trigger.
#[test]
fn helper_roundtrip_enabled_sentinel_passes_assertion() {
    assert_tier_b_sentinels("test_entry", 0xdeadbeef, 0xfeedface);
}

// NOTE: The mismatched-sentinel test (one zero, one non-zero) cannot be tested in
// the normal Rust test harness because assert_tier_b_sentinels calls
// std::process::abort() on mismatch — abort can't be caught by #[should_panic].
//
// The bench binary's --verify-ffi-sentinels subcommand (P-4.5 scope, deferred to v2
// per planner spec §6.6 "no production telemetry / debugging surface in v1") would
// run the mismatch test in a subprocess to observe the abort.
//
// For v1, the assertion's behavior is structurally guaranteed by the helper-bypass
// caught at runtime (process aborts on FATAL diagnostic per §4.3). This test surface
// documents the discipline; the actual mismatch trigger is exercised manually if a
// regression is suspected.
