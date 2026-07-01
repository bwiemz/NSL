//! CSHA cycle 19 T1 — dS probe integration test.
//!
//! **STATUS: XFAIL / IGNORED.** T1 lands the FFI + Cranelift extern-decl
//! scaffolding for `nsl_flash_attention_csha_backward_probe`, but the
//! PTX-side probe emission (Step 6 of the T1 spec: predicated
//! `st.global.f32` writes of the 8-slot probe layout {row_max, row_sum,
//! S_pre_mask, P, dP, rowsum_dP_P, dS, scale*dS} at (batch=0, head=0,
//! q_tile_iter=0, warp_row=1, lane=0, causal=true)) has NOT been wired.
//!
//! Per the cycle-18 DEGENERATE-PROBE meta-lesson (see
//! `project_csha_paper_completion_cycle18.md`) this is the honest
//! disposition: land the ABI + tests that will go GREEN once T2 (or the
//! T1 follow-on) extends the backward PTX emitter, and mark the
//! integration probe RED-with-XFAIL rather than shipping a green test
//! whose assertions are trivially satisfied by the delegate-only body.
//!
//! **Probe coordinates (NON-DEGENERATE gate — R11):**
//!   (batch=0, head=0, q_tile_iter=0, warp_row=1, lane=0, causal=true):
//!     P[1,0] ≈ 0.5, P[1,1] ≈ 0.5,
//!     dS = 0.25 · (dP[1,0] − dP[1,1]) — NONZERO under random dO
//!
//!   Cross-validation coord: (row=1, col=1, causal=false, warp_row=1, lane=1).
//!
//! When T2 wires the PTX side, remove the `#[ignore]` attribute and this
//! test must go GREEN by matching a CPU reference dS/dV at (row=1, col=0).

#![cfg(feature = "csha_cycle19_probe")]

#[test]
#[ignore = "cycle-19 T1 scaffolding: PTX-side probe emission deferred to T2; \
            body will be authored once the backward emitter accepts \
            probe_ds_out_ptr / probe_dv_out_ptr trailing pointers"]
fn csha_cycle19_ds_probe_matches_cpu_reference_at_row1_col0() {
    // XFAIL placeholder. The full body needs:
    //   1. Randomized dO + V at (heads=4, seq=32, head_dim=64, causal=true).
    //   2. Allocate two 8-slot f32 device buffers for probe_ds_out /
    //      probe_dv_out.
    //   3. Dispatch `nsl_flash_attention_csha_backward_probe` end-to-end.
    //   4. Copy the 8-slot outputs back to host.
    //   5. Compute the CPU reference {row_max, row_sum, S_pre_mask,
    //      P, dP, rowsum_dP_P, dS, scale*dS} at (row=1, col=0).
    //   6. `assert_relative_eq!(probe_ds[6], cpu_ref_ds, tol=1e-4)`.
    //
    // Until Step 6 of the c19 T1 spec is wired on the PTX side, the probe
    // slots are undefined (device build) or zeroed (host sentinel) — the
    // meaningful comparison here would be trivially satisfied or
    // meaningless. Marking XFAIL is the honest disposition.
    unimplemented!("cycle-19 T2 will land the PTX-side probe emission and unignore this test");
}
