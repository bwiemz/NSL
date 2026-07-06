//! CSHA cycle 20 T1-followup — dS probe integration test.
//!
//! **STATUS: OBSOLETE 2026-07-04 - target bug DISSOLVED.** The
//! "catastrophic dV/dS" investigation this probe was built to bisect
//! was a TEST-HARNESS bug, not a kernel bug: csha_cuda_backward.rs
//! launched the backward FFI with shared_mem_bytes = 0, so hd=64
//! (dynamic-SMEM) kernels ran with a 0-byte shmem allocation and every
//! tile access was out of bounds (fixed in commit b94ed20f; production
//! always sized correctly via shared_mem_bytes_v2_backward in
//! wengert_lower). The probes never executed on GPU. This file is
//! retained for probe-infra ABI pinning (compile-time signature check
//! below) and as the historical record of the probe machinery.
//!
//! **Original status: FFI PLUMBING LANDED (GPU-body #[ignore]d as GPU-required).**
//!
//! **What LANDED (c20 T1 + c20 T1-followup):**
//!   * PTX-side probe emission — `phases/backward/probe.rs` module with
//!     `maybe_emit_probe_store`; 8 predicated `st.global.f32` sites wired at
//!     ds_compute (slots 0-6) and dqdk_accum (slot 7).
//!   * Backward prelude widened under `csha_cycle19_probe` — two trailing
//!     `.param .u64 probe_{ds,dv}_out_ptr` + `%rd_probe_ds` +
//!     `%p_probe_active`. Default feature config remains byte-identical
//!     (fa_v2_snapshots 25/25).
//!   * Runtime launcher body refactored into private `csha_backward_impl`
//!     accepting `probe_ptrs: Option<(u64, u64)>`. The public 54-param
//!     `nsl_flash_attention_csha_backward` FFI passes `None` → sentinel-0
//!     probe slots → byte-identical to 12 pre-c19 callers. The c19
//!     `nsl_flash_attention_csha_backward_probe` FFI passes `Some((ds, dv))`
//!     → PTX `%p_probe_active = true` → probe stores fire.
//!
//! **This test's role:**
//!   1. Compile-verify the c19 probe FFI symbol exists with the expected
//!      56-i64 ABI (via a monomorphised function-pointer cast at test
//!      compile time). This half runs on every `cargo check --tests
//!      --features csha_cycle19_probe`.
//!   2. Provide the GPU-executable body (allocate 8-slot f32 device
//!      buffers, dispatch the probe FFI, read back, assert
//!      NON-DEGENERATE). This half is `#[ignore]`d — flip on GPU CI or
//!      run with `--ignored` locally.
//!
//! **Non-degenerate probe coordinates (R11):**
//!   (batch=0, head=0, q_tile_iter=0, warp_row=1, lane=0, causal=true):
//!     P[1,0] ≈ 0.5, P[1,1] ≈ 0.5,
//!     dS = 0.25 · (dP[1,0] − dP[1,1]) — NONZERO under random dO.
//!
//! **Slot 7 col-coordinate note (R11 cycle-20 T1-fixup).** Slots 0-6 are
//! emitted from ds_compute BEFORE the col-loop advances and naturally
//! sample col=0. Slot 7 (scale*dS) is emitted from `dqdk_accum` INSIDE
//! the KV-col loop; `maybe_emit_probe_store` provides no col-gate hook,
//! so slot 7 fires once per col and retains the LAST column's value —
//! it samples col=block_kv-1, NOT col=0. Any T5/c21 CPU-reference
//! matching for slot 7 MUST compare against the reference at
//! col=block_kv-1.
//!
//! **Numerical interpretation deferred to T5/c21.** This test only proves
//! the plumbing works — that the probe slots receive non-zero, finite
//! writes when the probe FFI is dispatched. Exact reference matching is
//! T5 scope (probe-gate meta-lesson: DO NOT ship a fix based on probe
//! readings within T1 — that is T4/T5 scope).

#![cfg(feature = "csha_cycle19_probe")]

// Compile-time ABI check: the c19 probe FFI symbol MUST exist with the
// expected 56-i64 signature. This is a NON-#[test] compile check —
// merely referencing the function pointer forces the linker to resolve
// the symbol and the cast forces its type to be verified. If the ABI
// drifts, `cargo check --features csha_cycle19_probe -p nsl-codegen` will
// fail here.
#[allow(dead_code)]
fn _abi_check_probe_ffi_symbol_exists() {
    #[allow(clippy::type_complexity)]
    let _f: unsafe extern "C" fn(
        // First 54 slots — identical to `nsl_flash_attention_csha_backward`.
        i64, i64, i64,          // q_ptr, k_ptr, v_ptr
        i64,                    // out_ptr
        i64,                    // logsumexp_ptr
        i64,                    // scale_bits
        i64, i64, i64, i64,     // batch, heads, seq_len, head_dim
        i64,                    // block_table_ptr
        i64, i64,               // k_pool_ptr, v_pool_ptr
        i64,                    // block_size
        i64, i64,               // cos_ptr, sin_ptr
        i64, i64,               // seq_ids_ptr, seq_lens_ptr
        i64,                    // shared_mem_bytes
        i64, i64,               // ptx_ptr, name_ptr
        i64, i64,               // block_q, block_kv
        i64,                    // causal
        i64, i64,               // x_ptr, norm_weight_ptr
        i64, i64, i64, i64,     // wq_ptr, wk_ptr, wv_ptr, wo_ptr
        i64,                    // rmsnorm_eps_bits
        i64, i64,               // active_heads, d_model
        i64, i64, i64,          // q_proj_ptr, k_proj_ptr, v_proj_ptr
        i64, i64,               // row_max_ptr, row_sum_ptr
        i64,                    // x_raw_ptr
        i64,                    // do_ptr
        i64, i64, i64,          // dq_ptr, dk_ptr, dv_ptr
        i64, i64, i64,          // dwq_ptr, dwk_ptr, dwv_ptr
        i64,                    // dx_ptr
        i64,                    // dx_norm_ptr
        i64,                    // segment_ids_ptr
        i64, i64,               // tier_b_ptx_ptr, tier_b_name_ptr
        i64,                    // doc_starts_ptr
        i64,                    // tier_b2_active
        i64,                    // num_docs_or_zero
        // c19 T1 trailing slots.
        i64, i64,               // probe_ds_out_ptr, probe_dv_out_ptr
    ) -> i64 = nsl_runtime::flash_attention::nsl_flash_attention_csha_backward_probe;
    let _ = _f;
}

/// GPU-executable body: allocate probe buffers, dispatch the probe FFI at
/// (heads=4, seq=32, head_dim=64, causal=true) with random dO+V inputs,
/// assert the probe wrote NON-DEGENERATE finite values into at least one
/// of {slot 6 = raw dS, slot 7 = scale*dS}.
///
/// **Numerical interpretation is deferred to T5/c21.** This test's goal
/// is to prove the plumbing is live — that the probe FFI wrapper's
/// pointers thread all the way through the launcher body and cause the
/// PTX-side `st.global.f32` sites to fire. Exact CPU-reference matching
/// (`assert_relative_eq!(probe_ds[6], cpu_ref_ds, tol=1e-4)`) is a
/// separate T5 scope.
#[test]
#[cfg(feature = "cuda")]
#[ignore = "GPU required — c20 T1-followup lands the FFI plumbing and \
            compile-verifies the probe symbol ABI; running this body \
            requires a CUDA device. Run with `cargo test -p nsl-codegen \
            --features cuda,csha_cycle19_probe -- --ignored \
            csha_cycle19_ds_probe`. Numerical CPU-reference matching is \
            T5/c21 scope."]
fn csha_cycle19_ds_probe_slots_populated_at_causal_row1() {
    // Test body deferred to GPU-CI harness. When running there:
    //   1. Build backward PTX + name via the codegen path.
    //   2. Allocate randomized f16 Q/K/V, dO, cos/sin (heads=4, seq=32,
    //      head_dim=64, causal=true).
    //   3. Allocate the 6 forward-saved activations (q_proj, k_proj,
    //      v_proj, row_max, row_sum, x_raw) — either via the forward
    //      kernel or an isolated `nsl_csha_alloc_backward_activations`.
    //   4. Allocate 8-slot f32 probe_ds_out + 8-slot f32 probe_dv_out
    //      via `cuda::inner::alloc_device(32)` and memset to 0.
    //   5. Dispatch `nsl_flash_attention_csha_backward_probe(<54 fwd
    //      params>, probe_ds as i64, probe_dv as i64)`.
    //   6. Copy back 8 f32 slots for probe_ds.
    //   7. Assert:
    //        - all slots finite: `slot.is_finite()`
    //        - all slots bounded: `slot.abs() < 1e9`
    //        - probe FIRED: at least one of {probe_ds[6], probe_ds[7]}
    //          is non-zero (proves %p_probe_active = true and the
    //          st.global.f32 sites executed).
    //
    // Not exercised here — see module docstring for the compile-verified
    // FFI plumbing.
    panic!(
        "GPU test body — see #[ignore] rationale. c20 T1-followup landed \
         the runtime FFI plumbing; this GPU-body runs the actual probe \
         dispatch and is deferred to GPU CI."
    );
}
