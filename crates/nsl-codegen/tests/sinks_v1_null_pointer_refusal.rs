//! Sprint 1b cycle-7 paper §4.3: runtime FFI null-pointer guard pin.
//!
//! The codegen-side eligibility check (`flash_attention_v2::sinks::
//! attention_sinks_v1_eligible`) catches mis-configured kernels at
//! compile time. The runtime-side guard
//! (`nsl_runtime::flash_attention::validate_sinks_pointers_nonnull`)
//! catches the OTHER class of mis-configurations: a caller that
//! correctly compiled a sinks-enabled kernel but failed to thread the
//! actual `sink_k_ptr` / `sink_v_ptr` through the launch list. Without
//! the host-side guard the kernel would launch with null pointers and
//! the first `ld.global.b16 [sink_k_ptr + ...]` access would fail with
//! `CUDA_ERROR_ILLEGAL_ADDRESS` — a useless device-side diagnostic.
//!
//! This test pins the contract from the codegen side so a future
//! refactor that moves the validator into a different module / drops
//! its diagnostic strings is caught immediately.
//!
//! NOTE: the production `nsl_flash_attention` FFI does not yet thread
//! `sink_k_ptr` / `sink_v_ptr` through its signature (the cascade of
//! Cranelift call-site updates exceeds the Sprint 1b 100-LOC budget per
//! the spec's failure-policy guidance). The validator is exposed so the
//! eventual FFI wiring can call it directly without rewriting the
//! null-check.

#![cfg(feature = "test-helpers")]

use nsl_runtime::flash_attention::validate_sinks_pointers_nonnull;

#[test]
fn disabled_sinks_accepts_null_pointers() {
    // Sentinel: when sinks are disabled, the validator MUST accept null
    // pointers (production callers pass 0 for the disabled path so this
    // case must not regress).
    assert!(validate_sinks_pointers_nonnull(0, 0, 0).is_ok());
}

#[test]
fn enabled_sinks_with_both_nonnull_accepts() {
    assert!(validate_sinks_pointers_nonnull(4, 0xdead, 0xbeef).is_ok());
}

#[test]
fn enabled_sinks_with_both_null_refuses_naming_both() {
    let err = validate_sinks_pointers_nonnull(4, 0, 0)
        .expect_err("enabled sinks with both pointers null MUST refuse");
    assert!(
        err.contains("sink_k_ptr") && err.contains("sink_v_ptr"),
        "diagnostic must name BOTH missing pointers so the user knows \
         to thread both through the launch list: {err}"
    );
    assert!(
        err.contains("num_sink_tokens=4"),
        "diagnostic must echo the configured num_sink_tokens for context: {err}"
    );
}

#[test]
fn enabled_sinks_with_null_k_only_refuses_naming_k() {
    let err = validate_sinks_pointers_nonnull(4, 0, 0xbeef)
        .expect_err("enabled sinks with sink_k_ptr=null MUST refuse");
    assert!(
        err.contains("sink_k_ptr"),
        "diagnostic must name the missing sink_k_ptr: {err}"
    );
    assert!(
        !err.contains("sink_v_ptr"),
        "diagnostic must NOT name sink_v_ptr when only k is null \
         (avoids misleading the user): {err}"
    );
}

#[test]
fn enabled_sinks_with_null_v_only_refuses_naming_v() {
    let err = validate_sinks_pointers_nonnull(4, 0xdead, 0)
        .expect_err("enabled sinks with sink_v_ptr=null MUST refuse");
    assert!(
        err.contains("sink_v_ptr"),
        "diagnostic must name the missing sink_v_ptr: {err}"
    );
    assert!(
        !err.contains("sink_k_ptr"),
        "diagnostic must NOT name sink_k_ptr when only v is null \
         (avoids misleading the user): {err}"
    );
}
