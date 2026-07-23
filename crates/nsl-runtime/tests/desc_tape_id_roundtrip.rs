#![cfg(feature = "test-hooks")]
//! Per-call grad context, `c_wrapper.rs` tape_id round-trip fix.
//!
//! The bug being fixed: `NslTensorDesc` had no `tape_id` field, so
//! `nsl_tensor_to_desc` → `desc_to_nsl_tensor` produced a fresh
//! wrapper with `tape_id == 0`. Backward keyed the loss seed on
//! `if t.tape_id != 0 { t.tape_id } else { loss_ptr }`, so a fresh
//! wrapper's raw pointer never matched any `TapeOp::*.out` tape_id,
//! the seed was dropped, and every parameter grad fell through to the
//! `zeros_like` fallback.
//!
//! This file pins the **mechanism** of the fix: the new
//! `tape_id: i64` field is carried verbatim across a desc round-trip.
//! The end-to-end gradient-correctness proof is in
//! `python/tests/test_m62_grad_context.py::test_real_weight_export_backward_returns_correct_grad`
//! (the gating fixture).

use std::os::raw::c_void;

use nsl_runtime::autodiff::{nsl_tape_start, nsl_tape_stop};
use nsl_runtime::c_api::{nsl_desc_to_tensor, nsl_tensor_to_desc_ffi, NslTensorDesc};
use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_free, test_build_tensor_2d_f32, test_tensor_tape_id};

#[test]
fn desc_carries_tape_id_across_roundtrip() {
    // ── Build a 1x1 f32 tensor; initially `tape_id == 0` ───────────
    let t_ptr = test_build_tensor_2d_f32(1, 1, &[42.0]);
    assert_ne!(t_ptr, 0, "test fixture failed to allocate tensor");
    assert_eq!(
        test_tensor_tape_id(t_ptr),
        0,
        "fresh tensor should have tape_id=0"
    );

    // ── Stamp a non-zero tape_id via tape registration ─────────────
    // `nsl_tape_start` walks the param list and calls
    // `Tape::get_or_assign_id`, which assigns `next_id` and bumps.
    let params = nsl_list_new();
    nsl_list_push(params, t_ptr);
    nsl_tape_start(params);
    nsl_list_free(params);

    let assigned_id = test_tensor_tape_id(t_ptr);
    assert!(
        assigned_id > 0,
        "tape_start should have assigned tape_id > 0, got {assigned_id}"
    );

    // ── Write into a desc via nsl_tensor_to_desc_ffi ───────────────
    let mut desc = NslTensorDesc::default();
    nsl_tensor_to_desc_ffi(t_ptr, &mut desc as *mut NslTensorDesc as i64);
    assert_eq!(
        desc.tape_id, assigned_id,
        "nsl_tensor_to_desc_ffi must copy tensor.tape_id ({assigned_id}) \
         into desc.tape_id (got {})",
        desc.tape_id,
    );

    // ── Read back into a fresh wrapper ─────────────────────────────
    let wrapper_ptr = nsl_desc_to_tensor(&desc as *const NslTensorDesc as i64);
    assert_ne!(wrapper_ptr, 0, "nsl_desc_to_tensor returned null");
    let wrapper_id = test_tensor_tape_id(wrapper_ptr);
    assert_eq!(
        wrapper_id, assigned_id,
        "fresh wrapper from nsl_desc_to_tensor must inherit desc.tape_id \
         ({assigned_id}), got {wrapper_id}. This is the load-bearing \
         assertion for the c_wrapper tape_id round-trip fix — if it fails, \
         backward's loss seed falls back to raw-pointer keying and grads \
         silently zero."
    );

    nsl_tensor_free(wrapper_ptr);
    nsl_tensor_free(t_ptr);

    // Reset the tape so subsequent tests start clean.
    nsl_tape_stop();

    // ── Untracked path: tape_id=0 in desc → tape_id=0 in wrapper ──
    let mut shape_buf = vec![1i64];
    let mut data_buf = vec![0.0f32; 1];
    let untracked_desc = NslTensorDesc {
        data: data_buf.as_mut_ptr() as *mut c_void,
        shape: shape_buf.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        ndim: 1,
        dtype: 1, // canonical f32
        device_type: 0,
        device_id: 0,
        tape_id: 0,
    };
    let untracked = nsl_desc_to_tensor(&untracked_desc as *const NslTensorDesc as i64);
    assert_ne!(untracked, 0);
    assert_eq!(
        test_tensor_tape_id(untracked),
        0,
        "tape_id=0 in desc must produce tape_id=0 in wrapper \
         (legitimate untracked semantics; the 'has-no-history' case)"
    );
    nsl_tensor_free(untracked);
}
