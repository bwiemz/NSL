#![cfg(feature = "test-hooks")]
//! Spec B §5.4 — `GradContext` is `Send`; created on thread A's local
//! tape, the `*mut GradContext` pointer can be sent across a channel and
//! `nsl_model_backward` consumed on thread B. The gradient produced on
//! thread B must match the analytical value derived from thread A's
//! forward.
//!
//! Built on the same structural test-hooks pattern as the other Spec B
//! tests (`backward_does_not_consult_live_tape.rs`,
//! `multi_context_in_flight.rs`, `double_backward_returns_typed_error.rs`)
//! rather than the `@export` end-to-end path, because the latter is
//! gated on a pre-existing `tape_id`-mismatch architectural issue
//! documented at `c_wrapper.rs:95-97` (out of Plan B's scope).
//!
//! Why this is sound w.r.t. the thread-local TAPE:
//!   - `nsl_tape_start` / `nsl_tensor_mul` / `test_drain_tape_and_params`
//!     all touch the *thread-local* TAPE on thread A.
//!   - Thread B never calls `nsl_tape_start`; `nsl_model_backward` walks
//!     `ctx.ops` (moved into the GradContext on thread A) without
//!     touching thread B's TAPE.
//!   - That separation is exactly the §5.4 invariant.

use std::sync::mpsc;

use nsl_runtime::autodiff::{nsl_tape_start, test_drain_tape_and_params};
use nsl_runtime::grad_context::{nsl_grad_context_destroy, nsl_model_backward, GradContext};
use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_mul, test_build_tensor_2d_f32};

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

/// Wrapper to allow sending an `i64` GradContext pointer across threads.
///
/// Sound because: `GradContext` is `Send` per Spec B §5.4, and the API
/// contract serializes ownership (thread A `Box::into_raw`s and sends;
/// thread B is the sole consumer afterwards — no concurrent access).
struct SendableI64(i64);
unsafe impl Send for SendableI64 {}

#[test]
fn forward_on_thread_a_backward_on_thread_b() {
    let (tx, rx) = mpsc::channel::<SendableI64>();

    // -------------------- Thread A: build ctx --------------------
    let thread_a = std::thread::spawn(move || {
        // y = x * w with w=1.0 → dw = d(x*w)/dw = x = 4.0
        let w_ptr = test_build_tensor_2d_f32(1, 1, &[1.0]);
        let x_ptr = test_build_tensor_2d_f32(1, 1, &[4.0]);

        let params = nsl_list_new();
        nsl_list_push(params, w_ptr);
        nsl_tape_start(params);
        nsl_list_free(params);

        let y_ptr = nsl_tensor_mul(x_ptr, w_ptr, 0);

        // Drain thread A's TAPE into a fresh GradContext (mimics
        // nsl_model_forward_grad's move-out tail).
        let (ops, param_ptrs) = test_drain_tape_and_params();
        let ctx = Box::new(GradContext::new(
            ops,
            vec![x_ptr],
            vec![y_ptr],
            param_ptrs,
        ));
        let ctx_ptr = Box::into_raw(ctx) as i64;

        tx.send(SendableI64(ctx_ptr)).expect("send ctx ptr");
    });
    thread_a.join().expect("thread A joined");

    // -------------------- Thread B: consume ctx via backward --------------------
    let ctx_send = rx.recv().expect("receive ctx ptr");

    let thread_b = std::thread::spawn(move || {
        let ctx_ptr = ctx_send.0;

        let mut grad_descs = vec![Desc::default()];

        let rc = nsl_model_backward(
            ctx_ptr,
            0,
            0, // grad_outputs reserved (v1 seeds with ones_like internally)
            grad_descs.as_mut_ptr() as i64,
            1,
        );
        assert_eq!(rc, 0, "backward on thread B must succeed");
        assert!(
            !grad_descs[0].data.is_null(),
            "backward on thread B must populate grad desc data",
        );

        // dw = x = 4.0. f32 multiply against w=1.0 is bit-exact in
        // IEEE-754, so demand strict equality.
        let grad_w_val: f32 = unsafe { *(grad_descs[0].data as *const f32) };
        assert_eq!(
            grad_w_val, 4.0,
            "expected grad_w=4.0 (= x), got {}",
            grad_w_val,
        );

        nsl_grad_context_destroy(ctx_ptr);
    });
    thread_b.join().expect("thread B joined");
}
