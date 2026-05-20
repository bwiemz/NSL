#![cfg(feature = "test-hooks")]
//! Spec B §5.1 — Backward replays from `GradContext`, NEVER from the
//! live thread-local tape.
//!
//! Architectural background: the `@export` dispatch path wraps input
//! `NslTensorDesc`s into fresh `NslTensor` objects (via
//! `nsl_desc_to_tensor`) before the forward body runs. Those wrappers
//! get fresh `tape_id`s that don't match the model's `weight_ptrs`
//! tape_ids, so end-to-end gradient verification through `@export` is
//! a separate (deferred) integration concern — see the GRAD SCOPE comment
//! in `crates/nsl-codegen/src/c_wrapper.rs` (above `emit_c_abi_wrapper`).
//! Spec B T8 removed the model-level `grad_enabled` slot and the
//! `nsl_model_enable_grad` FFI; backward now reads from a `GradContext`
//! produced by the grad-context bridge.
//!
//! This test therefore validates the HEADLINE INVARIANT (Spec B §2)
//! **structurally** without depending on `@export` produce-correct-grads.
//! It directly:
//!
//!   1. Starts a tape, records `Mul { a: x, b: w }` via the autodiff
//!      op surface (matching the `tests/run_backward_core_matches_tape_backward.rs`
//!      pattern from T1).
//!   2. Moves the recorded ops into a fresh `GradContext` (matching
//!      the move-out step §4.2 does at the end of `nsl_model_forward_grad`).
//!   3. Repeats (1)–(2) with DIFFERENT inputs, producing a second
//!      `GradContext` whose ops reference different tensors. The live
//!      tape is empty after this step.
//!   4. Calls the new `nsl_model_backward(ctx_a, ...)` FFI on the
//!      FIRST context. Since backward reads only from `ctx_a.ops`
//!      (Spec B §2 invariant), the gradient must reflect the FIRST
//!      forward's `x_a = 3.0`, not the second's `x_b = 7.0`.
//!
//! Load-bearing assertion: the gradient tensor returned by backward
//! has value `3.0` — matching the first forward's `x_a`. If backward
//! had consulted the live tape (which was last populated and then
//! emptied by the second forward), the gradient would either be `7.0`
//! or zero / NaN depending on the implementation bug.

use nsl_runtime::autodiff::{
    nsl_tape_start, test_drain_tape_and_params, test_tape_ops_len,
};
use nsl_runtime::grad_context::{nsl_grad_context_destroy, nsl_model_backward, GradContext};
use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_mul, test_build_tensor_2d_f32, test_read_tensor_f32};

use std::os::raw::c_void;

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

/// Mimic the move-out step at the end of `nsl_model_forward_grad`:
/// collect ops + param_ptrs from the live tape, build a GradContext,
/// then clear the tape's recording state (RAII guard equivalent).
///
/// Returns the heap-allocated `*mut GradContext` as i64 (matching the
/// FFI's pointer ABI).
fn drain_tape_into_ctx(output_ptr: i64) -> i64 {
    let (ops, param_ptrs) = test_drain_tape_and_params();
    let ctx = Box::new(GradContext::new(
        ops,
        Vec::new(),          // input_ptrs — not used by backward
        vec![output_ptr],    // output_ptrs[0] is the loss seed for backward
        param_ptrs,
    ));
    Box::into_raw(ctx) as i64
}

#[test]
fn backward_does_not_consult_live_tape() {
    // -------------------- Forward A: x = 3.0, w = 2.0 --------------------
    // Forward: y_a = x_a * w. The tape records Mul { a: x_a, b: w }.
    let x_a = test_build_tensor_2d_f32(1, 1, &[3.0]);
    let w_a = test_build_tensor_2d_f32(1, 1, &[2.0]);

    let params_a = nsl_list_new();
    nsl_list_push(params_a, w_a);
    nsl_tape_start(params_a);
    nsl_list_free(params_a);

    let y_a = nsl_tensor_mul(x_a, w_a, 0);
    let vals_a = test_read_tensor_f32(y_a);
    assert_eq!(vals_a, vec![6.0], "forward A: y = x * w = 3 * 2 = 6");

    // Move the recorded ops out of the live tape into ctx_a. After
    // this, the live tape is empty (the RAII guard equivalent has
    // fired).
    let ctx_a = drain_tape_into_ctx(y_a);
    assert_ne!(ctx_a, 0, "ctx_a allocated");

    assert_eq!(
        test_tape_ops_len(),
        0,
        "live tape must be empty after move-out into ctx_a"
    );

    // -------------------- Forward B: x = 7.0, w_b = 11.0 --------------------
    // CRITICAL: this re-populates the thread-local TAPE with DIFFERENT
    // ops that reference x_b (7.0) and w_b (11.0). Then we move them
    // into ctx_b, leaving the live tape empty again. After this step,
    // there is NO tape state that could conceivably produce
    // backward-on-ctx_a's correct gradient EXCEPT ctx_a.ops itself.
    let x_b = test_build_tensor_2d_f32(1, 1, &[7.0]);
    let w_b = test_build_tensor_2d_f32(1, 1, &[11.0]);

    let params_b = nsl_list_new();
    nsl_list_push(params_b, w_b);
    nsl_tape_start(params_b);
    nsl_list_free(params_b);

    let y_b = nsl_tensor_mul(x_b, w_b, 0);
    let vals_b = test_read_tensor_f32(y_b);
    assert_eq!(vals_b, vec![77.0], "forward B: y = x_b * w_b = 7 * 11 = 77");

    let ctx_b = drain_tape_into_ctx(y_b);
    assert_ne!(ctx_b, 0, "ctx_b allocated");

    assert_eq!(
        test_tape_ops_len(),
        0,
        "live tape must be empty after move-out into ctx_b"
    );

    // -------------------- Backward on ctx_a --------------------
    // Spec B §2 says backward replays from ctx_a.ops, NOT from the
    // live tape. ctx_a.ops references x_a (3.0) and w_a (2.0). dy/dw
    // for Mul = x = 3.0.
    let mut grad_descs = vec![Desc {
        data: std::ptr::null_mut(),
        shape: std::ptr::null_mut(),
        strides: std::ptr::null_mut(),
        ndim: 0,
        dtype: 0,
        device_type: 0,
        device_id: 0,
    }];
    let rc = nsl_model_backward(
        ctx_a,
        0,
        0, // grad_outputs reserved (v1 seeds with ones_like internally)
        grad_descs.as_mut_ptr() as i64,
        1,
    );
    assert_eq!(rc, 0, "backward on ctx_a returned non-zero");

    // Read the gradient through the desc's data pointer. The desc was
    // populated by `nsl_tensor_to_desc`, which overrides desc.data
    // with the gradient tensor's owned buffer.
    assert!(
        !grad_descs[0].data.is_null(),
        "backward did not populate grad desc data"
    );
    let grad_w_val: f32 = unsafe { *(grad_descs[0].data as *const f32) };

    // dy/dw = x_a = 3.0 from ctx_a.ops. If backward had consulted the
    // live tape (which was re-populated by forward B before being
    // drained into ctx_b), the gradient would be 7.0 (x_b). Since the
    // live tape is currently empty, a tape-consulting bug would more
    // likely produce 0.0 / NaN — but the 3.0-vs-7.0 distinction is the
    // load-bearing semantic test.
    assert_eq!(
        grad_w_val, 3.0,
        "grad_w must equal x_a=3.0 (from ctx_a.ops), not x_b=7.0 \
         (would mean backward consulted the live tape)"
    );

    // -------------------- Double-backward returns typed error --------------------
    // §5.5: a second backward on the same context must NOT re-execute.
    // Instead, it returns -1 with "already consumed" error.
    let rc2 = nsl_model_backward(ctx_a, 0, 0, 0, 0);
    assert_eq!(rc2, -1, "second backward on ctx_a must fail");

    // -------------------- Null-context error path --------------------
    let rc3 = nsl_model_backward(0, 0, 0, 0, 0);
    assert_eq!(rc3, -1, "backward(NULL) must fail");

    // -------------------- Cleanup --------------------
    nsl_grad_context_destroy(ctx_a);
    nsl_grad_context_destroy(ctx_b);
}
