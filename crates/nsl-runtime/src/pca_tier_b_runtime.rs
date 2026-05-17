//! PCA Tier B — runtime-side dispatch gate + constants.
//!
//! Per planner spec §6 and (α) commitment from P-2 (this module is the source of truth
//! for both constants since `nsl-codegen` already depends on `nsl-runtime`).
//!
//! The runtime gate's four-condition logic determines whether the kernel launch
//! dispatches to the Tier-B-on PTX variant (when codegen emitted one for this config)
//! or to the base Tier-B-off PTX (always present in the existing FFI's `ptx_ptr/name_ptr`).

/// Empirical seq_len floor (wall-time win ≥ 10% per dispatch spec §6).
///
/// Derived from `docs/superpowers/specs/2026-05-15-tier-b-floor-derivation-findings.md`
/// (P-1 D-2 sweep; 7 seq_lens × 100 iterations × sparsity=50% on RTX 5070 Ti sm_120).
/// All 7 sweep points cleared the 10% bar; FLOOR is the smallest sweep point at 128
/// (win=39.40%; curve saturates near ~75% for seq>=512 = (1-sparsity) theoretical bound).
pub const TIER_B_SEQ_LEN_FLOOR: u32 = 128;

/// Conservative-max seq_len baked into Tier-B-on PTX SMEM allocation.
///
/// Derived from `docs/superpowers/specs/2026-05-15-tier-b-bii-smem-probe-findings.md`
/// (P-0 V-Bii-SMEM probe; 12-config sweep on RTX 5070 Ti sm_120 / sm_80 JIT-fallback).
/// Sub-variant resolved: B-ii unrestricted. MAX=16384/block=32 fits at 12.12% util sm_120
/// (`<60%` headroom bucket → planner spec §3.5 v2 trigger #1 is "extend baked max to 32768").
pub const TIER_B_MAX_BAKED_SEQ_LEN: u32 = 16384;

// Compile-time assertion: MAX_BAKED is a probe-validated value (planner spec §5.3).
const _: () = assert!(
    TIER_B_MAX_BAKED_SEQ_LEN == 4096
        || TIER_B_MAX_BAKED_SEQ_LEN == 8192
        || TIER_B_MAX_BAKED_SEQ_LEN == 16384,
    "TIER_B_MAX_BAKED_SEQ_LEN must be one of {{4096, 8192, 16384}} per V-Bii-SMEM probe's \
     five-outcome matrix; investigation-row outcomes require resolving the probe anomaly \
     before this constant is set."
);

// Compile-time assertion: joint-outcome validity (planner spec §11 risk #9).
// FLOOR > MAX_BAKED would produce a runtime dispatch gate that's never satisfied
// (seq_len >= FLOOR && seq_len <= MAX_BAKED is empty when FLOOR > MAX_BAKED).
const _: () = assert!(
    TIER_B_SEQ_LEN_FLOOR <= TIER_B_MAX_BAKED_SEQ_LEN,
    "TIER_B_SEQ_LEN_FLOOR must be <= TIER_B_MAX_BAKED_SEQ_LEN; otherwise the runtime \
     dispatch gate is never satisfied (Tier B ships but never activates). \
     V-Bii-SMEM and D-2 produced incompatible values; investigate both findings docs."
);

/// Runtime gate: should this kernel launch dispatch to the Tier-B-on variant?
///
/// Returns `true` iff ALL FOUR conditions hold:
/// 1. Codegen emitted a Tier-B-on variant for this config (`tier_b_ptx_ptr != 0`).
/// 2. The caller passed a non-null segment_ids pointer (`segment_ids_ptr != 0`).
/// 3. seq_len is at or above the empirical profitability floor (`seq_len >= TIER_B_SEQ_LEN_FLOOR`).
/// 4. seq_len fits the conservative-max baked into the Tier-B-on PTX
///    (`seq_len <= TIER_B_MAX_BAKED_SEQ_LEN`).
///
/// Conditions 3 and 4 together gate the seq_len range to `[FLOOR, MAX_BAKED]` —
/// below the floor, skip-check overhead exceeds skip-payoff; above MAX_BAKED, the
/// Tier-B-on PTX's SMEM allocation can't handle the table size.
pub fn should_dispatch_tier_b_at_runtime(
    tier_b_ptx_ptr: i64,
    segment_ids_ptr: i64,
    seq_len: u32,
) -> bool {
    tier_b_ptx_ptr != 0
        && segment_ids_ptr != 0
        && (TIER_B_SEQ_LEN_FLOOR..=TIER_B_MAX_BAKED_SEQ_LEN).contains(&seq_len)
}

/// Asserts the Tier B sentinel pair has both values either zero or both non-zero.
/// Panics (via process abort) if mismatched — catches helper-bypass at call sites
/// per planner spec §4.3.
///
/// Called at the entry of each `nsl_flash_attention*` FFI function that gains the
/// Tier B extension (P-3.4 scope; planner spec §4.6).
///
/// **Implementation note (test seam):** Production behavior is `std::process::abort()`.
/// For testing the mismatch case in `bench-internal` mode, we provide a thin shim
/// (P-4.2 scope) that swaps abort for panic so the test harness can observe the
/// failure without unwinding.
#[inline(always)]
pub fn assert_tier_b_sentinels(
    entry_point: &'static str,
    tier_b_ptx_ptr: i64,
    tier_b_name_ptr: i64,
) {
    if (tier_b_ptx_ptr == 0) != (tier_b_name_ptr == 0) {
        eprintln!(
            "FATAL [{entry_point}]: tier_b_ptx_ptr={tier_b_ptx_ptr:#x} but \
             tier_b_name_ptr={tier_b_name_ptr:#x}; sentinel pair must agree \
             (both zero = disabled, both non-zero = enabled). Call site emitted \
             via inline literals instead of tier_b_disabled_sentinel() / \
             tier_b_enabled()?"
        );
        std::process::abort();
    }
}
