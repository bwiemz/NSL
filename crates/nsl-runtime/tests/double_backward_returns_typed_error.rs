#![cfg(feature = "test-hooks")]
//! Spec B §5.5 — second `nsl_model_backward` call on a consumed
//! `GradContext` returns -1 with `ERR_ALREADY_CONSUMED` rather than
//! UAF-ing on the moved-out ops Vec.
//!
//! Constructed via the same structural test-hooks pattern as
//! `tests/backward_does_not_consult_live_tape.rs` (record on the live
//! tape, drain into a fresh `GradContext`) — the `@export` end-to-end
//! path is gated on a pre-existing `tape_id`-mismatch architectural
//! issue documented at `c_wrapper.rs:95-97`.

use nsl_runtime::autodiff::{nsl_tape_start, test_drain_tape_and_params};
use nsl_runtime::grad_context::{nsl_grad_context_destroy, nsl_model_backward, GradContext};
use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_mul, test_build_tensor_2d_f32};

use std::os::raw::{c_char, c_void};

#[repr(C)]
#[derive(Default)]
struct Desc {
    data: *mut c_void,
    shape: *mut i64,
    strides: *mut i64,
    ndim: i32,
    dtype: i32,
    device_type: i32,
    device_id: i32,
}

#[test]
fn second_backward_returns_already_consumed() {
    // -------------------- Forward: y = x * w (x=2, w=5) --------------------
    let x_ptr = test_build_tensor_2d_f32(1, 1, &[2.0]);
    let w_ptr = test_build_tensor_2d_f32(1, 1, &[5.0]);

    let params = nsl_list_new();
    nsl_list_push(params, w_ptr);
    nsl_tape_start(params);
    nsl_list_free(params);

    let y_ptr = nsl_tensor_mul(x_ptr, w_ptr, 0);

    // Drain into a GradContext (mimics nsl_model_forward_grad's tail).
    let (ops, param_ptrs) = test_drain_tape_and_params();
    let ctx = Box::new(GradContext::new(
        ops,
        vec![x_ptr],
        vec![y_ptr],
        param_ptrs,
    ));
    let ctx_ptr = Box::into_raw(ctx) as i64;

    // Caller-side grad descriptor (backward populates `data` via
    // `nsl_tensor_to_desc`).
    let mut grad_descs = vec![Desc::default()];

    // -------------------- First backward — must succeed --------------------
    let rc1 = nsl_model_backward(
        ctx_ptr,
        0,
        0, // grad_outputs reserved (v1 seeds with ones_like internally)
        grad_descs.as_mut_ptr() as i64,
        1,
    );
    assert_eq!(rc1, 0, "first backward must succeed");
    assert!(
        !grad_descs[0].data.is_null(),
        "first backward must populate grad desc data"
    );

    // -------------------- Second backward — typed error --------------------
    let rc2 = nsl_model_backward(
        ctx_ptr,
        0,
        0,
        grad_descs.as_mut_ptr() as i64,
        1,
    );
    assert_eq!(rc2, -1, "second backward on same ctx must return -1");

    // Confirm the typed error message lands in the thread-local slot.
    let err_ptr = nsl_runtime::c_api::nsl_get_last_error() as *const c_char;
    assert!(!err_ptr.is_null(), "last-error pointer must be non-null");
    let err = unsafe { std::ffi::CStr::from_ptr(err_ptr) }.to_string_lossy();
    assert!(
        err.contains("already consumed"),
        "expected 'already consumed' in error, got: {}",
        err,
    );

    // -------------------- Cleanup --------------------
    nsl_grad_context_destroy(ctx_ptr);
}
