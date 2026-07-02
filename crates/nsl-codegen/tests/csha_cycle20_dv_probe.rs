//! CSHA cycle 20 T2 — dV probe integration test.
//!
//! **STATUS: XFAIL / IGNORED (HONEST PARTIAL after c20 T2).**
//!
//! **What LANDED in c20 T2:**
//!   * PTX-side dV probe emission — new `maybe_emit_dv_probe_store`
//!     helper in `phases/backward/probe.rs` with 5 predicated
//!     `st.global.f32` sites wired at:
//!     - dv_accum (slots 0/1/2/4: `%f_P`, `%f0` pre-FMA, `%f1` dO_f32,
//!       `%f0` post-FMA cross-check)
//!     - finalize::emit_store_kv_only (slot 3: `%f_dk_tmp` SMEM
//!       readback)
//!   * SAME FFI symbol as c19 T1 (`nsl_flash_attention_csha_backward_probe`) —
//!     no launcher change; the `probe_dv_out_ptr` param declared but
//!     unused by T1 is now consumed by the new emission sites.
//!   * Prelude changes: NONE — the `%rd_probe_dv` register and its
//!     `ld.param.u64` were already emitted in T1 as forward-compatible
//!     plumbing. This test relies on that machinery.
//!
//! **Slot layout (5 f32 slots):**
//!   Slot 0: dV_P            — P[row=1, col=0] read at FMA site
//!   Slot 1: dV_accum_pre    — SMEM dV[col=0, d=0] BEFORE the FMA
//!   Slot 2: dV_dO_f32       — dO[row=1, d=0] as f32 (post cvt.f32.f16)
//!   Slot 3: dV_final        — final SMEM dV[col=0, d=0] at finalize
//!   Slot 4: dV_cross_check  — %f0 immediately after fma.rn.f32
//!                             (P * dO + prev_accum, register-side)
//!
//!   **Bisection contract:** slot 3 == slot 4 ⇒ SMEM path intact
//!     (hypothesis (iii) inter-phase corruption REFUTED); if they
//!     differ, corruption sits between the fma store and the finalize
//!     load — pointing at (iv) accumulator write path.
//!
//! **What is STILL DEFERRED to c21:**
//!   * End-to-end GPU dispatch. The c19 T1 launcher still delegates
//!     without threading `probe_dv_out_ptr` through the args array as
//!     a non-null value, so on-device stores fall through
//!     %p_probe_active=false. Same story as T1's dS probe: the PTX
//!     accepts + gates on the pointer, but the FFI wrapper drops it.
//!     Flipping this test to GREEN requires the c21 launcher-refactor
//!     work (private `csha_backward_impl(Some((probe_ds, probe_dv)))`).
//!   * Secondary launch coordinates for j=202 and j=158 (only j=284
//!     is scoped to this task) — the PTX gate is uniform; per-cell
//!     differentiation happens on the CPU side by launching multiple
//!     configurations. Documented for the c21 harness.
//!
//! **Probe coordinates (T2 scope: j=284 worst cell):**
//!   Oracle-side reference:
//!     (row_global=1, r1=0, lane=20, warp of d=37, kv_outer_iter=8)
//!   PTX-side gate (uniform with T1 dS probe):
//!     (batch=0, head=0, warp_id=1, lane=0, q_tile_iter=0)
//!
//!   The oracle vs PTX-side coord asymmetry is intentional — the CPU
//!   reference is parameterised by the CELL COORDINATE the paper cites;
//!   the PTX-side gate exists purely to funnel a SINGLE thread into
//!   the store so cross-CTA races cannot corrupt the readback.

#![cfg(feature = "csha_cycle19_probe")]

#[test]
#[ignore = "cycle-20 T2 HONEST PARTIAL: PTX-side dV probe emission LANDED \
            (helper + 5 store sites + unit tests), but end-to-end GPU dispatch \
            requires the c21 launcher-body refactor to thread \
            probe_dv_out_ptr through as a non-null value. See module docstring \
            for the bisection contract slot 3 == slot 4 must satisfy."]
fn csha_cycle20_dv_probe_matches_cpu_reference_at_j284() {
    // XFAIL placeholder. The full body needs (deferred to c21):
    //   1. Randomized dO + V at (heads=4, seq=32, head_dim=64, causal=true)
    //      matching the T4/T5 harness config used for the j=284 worst cell.
    //   2. Allocate a 5-slot f32 device buffer for probe_dv_out.
    //      (probe_ds_out can share the T1 8-slot buffer or be nullptr.)
    //   3. Dispatch `nsl_flash_attention_csha_backward_probe` end-to-end
    //      with a non-null probe_dv_out pointer.
    //   4. Copy the 5-slot output back to host.
    //   5. Compute the CPU reference {P, accum_pre, dO_f32, final, fma}
    //      at (row=1, col=0, d=0) — trivially derivable from the T1
    //      dS-probe CPU oracle already scaffolded.
    //   6. `assert_relative_eq!(probe_dv[0], cpu_ref_P, tol=1e-4)`.
    //   7. `assert_relative_eq!(probe_dv[3], probe_dv[4], tol=1e-6)` —
    //      the SMEM-vs-register cross-check; failure ⇒ hypothesis (iii)
    //      / (iv) diagnostic.
    //
    // T2 status: PTX-side machinery is in place. The 5-slot buffer will
    // read 0s (or driver-defined) until c21 threads the probe pointer
    // through the FFI wrapper.
    unimplemented!(
        "cycle-21 will land the launcher refactor + end-to-end assertions \
         for the dV probe. This test is XFAIL scaffolding — the PTX-side \
         emission is verified by the phases/backward/probe.rs unit tests."
    );
}
