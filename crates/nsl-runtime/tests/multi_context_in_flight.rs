#![cfg(feature = "test-hooks")]
//! Spec B §6 — three `GradContext`s constructed sequentially over the
//! same shared parameter `w`, then backward'd in reverse order. Each
//! context's gradient must reflect *its own* forward's `x_i`, proving
//! that backward replays from `ctx.ops` rather than any thread-local
//! state mutated by later forwards.
//!
//! Built on the structural test-hooks pattern from
//! `tests/backward_does_not_consult_live_tape.rs` — the `@export`
//! end-to-end path is gated on a pre-existing `tape_id`-mismatch
//! architectural issue documented at `c_wrapper.rs:95-97`.

use nsl_runtime::autodiff::{
    nsl_tape_start, test_drain_tape_and_params, test_tape_ops_len,
};
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
    tape_id: i64,
}

#[test]
fn three_contexts_backward_in_reverse_grads_correct() {
    // Shared parameter w=1.0 — chosen so dw = d(x*w)/dw = x = x_i, making
    // the per-ctx expected gradient trivially identifiable.
    let w_ptr = test_build_tensor_2d_f32(1, 1, &[1.0]);

    // (ctx_ptr, expected_grad_w) pairs, in forward (ascending) order.
    let mut ctx_pairs: Vec<(i64, f32)> = Vec::with_capacity(3);
    let inputs: [f32; 3] = [3.0, 5.0, 7.0];

    for &x_val in &inputs {
        let x_ptr = test_build_tensor_2d_f32(1, 1, &[x_val]);

        // Fresh param list per iteration matches the reference test's
        // pattern; `nsl_tape_start` snapshots into TAPE.param_set, so
        // the list itself is safe to free immediately.
        let params = nsl_list_new();
        nsl_list_push(params, w_ptr);
        nsl_tape_start(params);
        nsl_list_free(params);

        // Live tape was just drained at the end of the previous iter
        // (or it's the first iter, so nothing was on it). Either way,
        // tape_start above didn't add ops yet.
        assert_eq!(
            test_tape_ops_len(),
            0,
            "live tape must be empty at start of forward (x={})",
            x_val,
        );

        // y_i = x_i * w. Records Mul { a: x_i, b: w } on the tape.
        let y_ptr = nsl_tensor_mul(x_ptr, w_ptr, 0);

        // Drain into a GradContext (mimics forward_grad's move-out tail).
        let (ops, param_ptrs) = test_drain_tape_and_params();
        let ctx = Box::new(GradContext::new(
            ops,
            vec![x_ptr],
            vec![y_ptr],
            param_ptrs,
        ));
        let ctx_ptr = Box::into_raw(ctx) as i64;

        ctx_pairs.push((ctx_ptr, x_val));

        // Drain reset `recording=false` + `pause_depth=0`. Live tape
        // is empty, ready for the next forward.
        assert_eq!(
            test_tape_ops_len(),
            0,
            "live tape must be empty after drain (x={})",
            x_val,
        );
    }

    // Backward in REVERSE order — verifies that draining ctx_c's ops did
    // not corrupt ctx_b's or ctx_a's ops.
    for (i, (ctx_ptr, expected_grad)) in ctx_pairs.iter().enumerate().rev() {
        let mut grad_descs = vec![Desc::default()];

        let rc = nsl_model_backward(
            *ctx_ptr,
            0,
            0,
            grad_descs.as_mut_ptr() as i64,
            1,
        );
        assert_eq!(rc, 0, "backward on ctx[{}] must succeed", i);
        assert!(
            !grad_descs[0].data.is_null(),
            "backward on ctx[{}] must populate grad desc data",
            i,
        );

        // dw = x_i. f32 multiply against w=1.0 is bit-exact in IEEE-754,
        // so we can demand strict equality.
        let grad_w_val: f32 = unsafe { *(grad_descs[0].data as *const f32) };
        assert_eq!(
            grad_w_val, *expected_grad,
            "ctx[{}] grad_w = {}, expected {}",
            i, grad_w_val, expected_grad,
        );
    }

    // -------------------- Cleanup --------------------
    for (ctx_ptr, _) in ctx_pairs {
        nsl_grad_context_destroy(ctx_ptr);
    }
}
