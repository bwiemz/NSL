//! Spec B §5.2 — calling `nsl_model_forward_grad` while a tape is
//! already recording on this thread returns -1 with the
//! `ForwardInProgress` error message.
//!
//! Verification: use the `test_set_recording(true)` hook to put the
//! thread-local tape into the "already recording" state, call
//! `nsl_model_forward_grad`, verify rc == -1 and the error text.

#![cfg(feature = "test-hooks")]

use std::os::raw::c_char;

#[test]
fn second_forward_grad_returns_minus_one_with_error() {
    // Put the tape into "already recording" state.
    nsl_runtime::autodiff::test_set_recording(true);

    let mut ctx_out: i64 = 0;
    let rc = nsl_runtime::grad_context::nsl_model_forward_grad(
        1,                                          // dummy model_ptr (never deref'd on this path)
        0, 0,                                       // inputs
        0, 0,                                       // outputs
        &mut ctx_out as *mut i64 as i64,            // grad_context_out
    );

    // Reset state immediately so the failure path doesn't poison
    // subsequent tests on this thread.
    nsl_runtime::autodiff::test_set_recording(false);

    assert_eq!(rc, -1, "re-entry guard must return -1");
    let err_ptr = nsl_runtime::c_api::nsl_get_last_error() as *const c_char;
    assert!(!err_ptr.is_null(), "error must be set");
    let err = unsafe { std::ffi::CStr::from_ptr(err_ptr) }.to_string_lossy();
    assert!(
        err.contains("already in progress"),
        "expected ForwardInProgress error, got: {}", err,
    );
    assert_eq!(ctx_out, 0, "ctx_out must not be written on failure");
}
