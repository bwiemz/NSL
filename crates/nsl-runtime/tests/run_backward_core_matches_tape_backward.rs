#![cfg(feature = "test-hooks")]

//! Regression guard for Spec B T1: extracting `run_backward_core` from
//! `nsl_tape_backward` must be behavior-preserving — the public
//! `nsl_tape_backward` entry point must continue producing the same
//! gradient values for a simple forward graph.
//!
//! Forward: y = x * w  (w is parameter, x is input)
//! Backward: dw = x

use nsl_runtime::autodiff::{nsl_tape_backward, nsl_tape_start, nsl_tape_stop};
use nsl_runtime::list::{nsl_list_free, nsl_list_get, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_mul, test_build_tensor_2d_f32, test_read_tensor_f32};

fn make_f32_scalar(val: f32) -> i64 {
    // 1x1 f32 tensor — simplest "scalar" representation available via the
    // public test-hooks surface.
    test_build_tensor_2d_f32(1, 1, &[val])
}

#[test]
fn core_matches_tape_backward_for_simple_mul() {
    let x_ptr = make_f32_scalar(3.0);
    let w_ptr = make_f32_scalar(5.0);

    let params = nsl_list_new();
    nsl_list_push(params, w_ptr);
    nsl_tape_start(params);

    // Forward: y = x * w. flags=0 → no relinquish.
    let _y_ptr = nsl_tensor_mul(x_ptr, w_ptr, 0);

    // Use the same handle we just produced as the "loss". Backward seeds
    // it with ones_like, so dy/dw = x = 3.0.
    let grads = nsl_tape_backward(_y_ptr, params);
    let grad_w = nsl_list_get(grads, 0);
    let grad_vals = test_read_tensor_f32(grad_w);

    assert_eq!(grad_vals.len(), 1);
    assert_eq!(grad_vals[0], 3.0, "dy/dw = x = 3.0");

    nsl_list_free(grads);
    nsl_tape_stop();
    nsl_list_free(params);
}
