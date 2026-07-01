//! CSHA cycle 20 T1 — dS probe integration test.
//!
//! **STATUS: XFAIL / IGNORED (HONEST PARTIAL after c20 T1).**
//!
//! **What LANDED in c20 T1:**
//!   * PTX-side probe emission — new `phases/backward/probe.rs` module
//!     with `maybe_emit_probe_store`; 8 predicated `st.global.f32` sites
//!     wired at ds_compute (slots 0-6) and dqdk_accum (slot 7).
//!   * Backward prelude widened under `csha_cycle19_probe` feature —
//!     two trailing `.param .u64 probe_{ds,dv}_out_ptr`, plus
//!     `%rd_probe_ds` / `%p_probe_active` register block. Default
//!     feature config remains byte-identical (fa_v2_snapshots 25/25).
//!   * Runtime launcher (`nsl_flash_attention_csha_backward` body) —
//!     args array widened to 51 slots under feature ON with sentinel-0
//!     probe pointers so `%p_probe_active` stays false for non-probe
//!     launches; the 12 pre-c19 callers see byte-identical semantics.
//!   * per_doc_cta backward-synth tail matcher — feature-gated to match
//!     `probe_dv_out_ptr\n)` instead of `dv_scratch_ptr\n)`.
//!
//! **What is STILL DEFERRED:**
//!   * The `nsl_flash_attention_csha_backward_probe` FFI wrapper still
//!     DELEGATES to the 54-param body — the probe pointer arguments to
//!     the FFI are dropped on the floor rather than threaded into the
//!     args array as non-null values. Refactoring the launcher body
//!     into a private `csha_backward_impl(Some((probe_ds, probe_dv)))`
//!     is deferred to a T2 follow-on.
//!   * As a result, END-TO-END the probe slots always read 0 (or
//!     un-initialised — driver-defined) because %p_probe_active is
//!     always false when reached through today's FFI. The 8-slot
//!     scratch buffer never receives a non-zero write.
//!   * This test therefore stays `#[ignore]`d as XFAIL — flipping it
//!     to GREEN requires the T2 launcher refactor.
//!
//! **Probe coordinates (NON-DEGENERATE gate — R11):**
//!   (batch=0, head=0, q_tile_iter=0, warp_row=1, lane=0, causal=true):
//!     P[1,0] ≈ 0.5, P[1,1] ≈ 0.5,
//!     dS = 0.25 · (dP[1,0] − dP[1,1]) — NONZERO under random dO
//!
//!   Cross-validation coord: (row=1, col=1, causal=false, warp_row=1, lane=1).
//!
//! When T2 wires the probe FFI to pass its pointer arguments through
//! `nsl_flash_attention_csha_backward_probe`, remove the `#[ignore]`
//! attribute and this test must go GREEN by matching a CPU reference
//! dS/dV at (row=1, col=0).

#![cfg(feature = "csha_cycle19_probe")]

#[test]
#[ignore = "cycle-20 T1 HONEST PARTIAL: PTX-side probe emission LANDED, \
            but the runtime FFI wrapper still delegates without threading \
            probe_ds_out_ptr / probe_dv_out_ptr through the launcher's args \
            array. Flipping to GREEN requires the T2 launcher-body refactor \
            (private `csha_backward_impl` accepting Option<(probe_ds, probe_dv)>)."]
fn csha_cycle19_ds_probe_matches_cpu_reference_at_row1_col0() {
    // XFAIL placeholder. The full body needs (deferred to T2):
    //   1. Randomized dO + V at (heads=4, seq=32, head_dim=64, causal=true).
    //   2. Allocate two 8-slot f32 device buffers for probe_ds_out /
    //      probe_dv_out.
    //   3. Dispatch `nsl_flash_attention_csha_backward_probe` end-to-end.
    //   4. Copy the 8-slot outputs back to host.
    //   5. Compute the CPU reference {row_max, row_sum, S_pre_mask,
    //      P, dP, rowsum_dP_P, dS, scale*dS} at (row=1, col=0).
    //   6. `assert_relative_eq!(probe_ds[6], cpu_ref_ds, tol=1e-4)`.
    //
    // C20 T1 status: steps 1-2 are trivially implementable now that the
    // PTX side accepts + gates on the probe pointers; step 3 will read
    // 0s from the probe buffer because the FFI wrapper drops the probe
    // pointers on the floor. T2 must land the launcher refactor before
    // this test can be marked GREEN — see module docstring for scope.
    unimplemented!(
        "cycle-20 T2 will refactor the runtime launcher body to thread \
         probe pointers through and unignore this test"
    );
}
