//! GPU regression for `nsl_tensor_mul_scalar_inplace`'s device round-trip.
//!
//! The GPU arm pulls the tensor to CPU (which upcasts f32 -> f64), scales, and
//! copies the result back into the f32 device buffer. A prior version copied
//! `len * 4` bytes straight from the f64 CPU buffer, reinterpreting f64 bit
//! patterns as f32 and corrupting the device tensor. This test builds a GPU
//! f32 tensor, scales it, and confirms every element matches the CPU reference
//! (it fails on the pre-fix code).
//!
//! Requires a GPU:
//!   cargo test -p nsl-runtime --features cuda --test mul_scalar_gpu -- --include-ignored

#![cfg(feature = "cuda")]

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_tensor_free, nsl_tensor_from_static, nsl_tensor_get, nsl_tensor_mul_scalar_inplace,
    nsl_tensor_to_device,
};

/// NslTensor dtype tag for f32 (runtime ABI: 0=f64, 1=f32).
const DTYPE_F32: i64 = 1;

/// Probe for a usable GPU (pattern from inspect_ffi.rs / precision_cast tests).
fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_runtime::nsl_cuda_init() == 0
}

/// Build a rank-1 CPU f32 tensor over a leaked buffer (`owns_data = 0`).
fn f32_tensor(vals: &[f32]) -> i64 {
    let leaked: &'static [f32] = Box::leak(vals.to_vec().into_boxed_slice());
    let shape = nsl_list_new();
    nsl_list_push(shape, vals.len() as i64);
    let t = nsl_tensor_from_static(leaked.as_ptr() as i64, shape, DTYPE_F32);
    nsl_list_free(shape);
    t
}

/// Read element `i` of a rank-1 tensor as f64.
fn get1(t: i64, i: i64) -> f64 {
    let idx = nsl_list_new();
    nsl_list_push(idx, i);
    let v = nsl_tensor_get(t, idx);
    nsl_list_free(idx);
    v
}

#[test]
#[ignore]
fn mul_scalar_inplace_gpu_matches_cpu_reference() {
    if !cuda_available() {
        eprintln!("skipping: no usable CUDA GPU");
        return;
    }
    let vals = [1.0_f32, 2.0, -3.0, 4.5, 0.0, -0.25, 8.0, -16.5];
    let scalar = 2.5_f64;

    let cpu = f32_tensor(&vals);
    let gpu = nsl_tensor_to_device(cpu, 1);
    assert_ne!(gpu, 0, "to_device(_, 1) must produce a GPU tensor");

    nsl_tensor_mul_scalar_inplace(gpu, scalar);

    let gpu_back = nsl_tensor_to_device(gpu, 0);
    for (i, &v) in vals.iter().enumerate() {
        let expected = (v as f64) * scalar;
        let got = get1(gpu_back, i as i64);
        assert!(
            (got - expected).abs() <= 1e-5,
            "elem {i}: expected {expected}, got {got} \
             (GPU round-trip reinterpreted f64 bytes as f32?)"
        );
    }

    nsl_tensor_free(gpu_back);
    nsl_tensor_free(gpu);
    nsl_tensor_free(cpu);
}
