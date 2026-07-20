//! Isolation test for the fused RMSNorm dx-backward kernel (Item 9).
//!
//! `nsl_rmsnorm_dx_backward` has a CPU f64 reference and a native GPU kernel
//! (`nsl_rmsnorm_dx_bwd_f32`, one block per row, one fused Σx²/Σȳγx reduction).
//! This builds random-ish multi-row inputs, computes dx on CPU (the reference)
//! and on GPU, and asserts they agree to an f32 tolerance — the approx
//! rsqrt/div path vs the exact f64 path. Validates the kernel in isolation,
//! before it is wired into source-AD.
//!
//!   cargo test -p nsl-runtime --features cuda --test rmsnorm_dx_backward_gpu -- --include-ignored

#![cfg(feature = "cuda")]

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_rmsnorm_dx_backward, nsl_tensor_free, nsl_tensor_from_static, nsl_tensor_get,
    nsl_tensor_to_device,
};

const DTYPE_F32: i64 = 1;

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_runtime::nsl_cuda_init() == 0
}

/// Rank-2 CPU f32 tensor [rows, cols] over a leaked buffer.
fn f32_2d(rows: i64, cols: i64, vals: &[f32]) -> i64 {
    assert_eq!(vals.len() as i64, rows * cols);
    let leaked: &'static [f32] = Box::leak(vals.to_vec().into_boxed_slice());
    let shape = nsl_list_new();
    nsl_list_push(shape, rows);
    nsl_list_push(shape, cols);
    let t = nsl_tensor_from_static(leaked.as_ptr() as i64, shape, DTYPE_F32);
    nsl_list_free(shape);
    t
}

fn f32_1d(vals: &[f32]) -> i64 {
    let leaked: &'static [f32] = Box::leak(vals.to_vec().into_boxed_slice());
    let shape = nsl_list_new();
    nsl_list_push(shape, vals.len() as i64);
    let t = nsl_tensor_from_static(leaked.as_ptr() as i64, shape, DTYPE_F32);
    nsl_list_free(shape);
    t
}

fn get2(t: i64, r: i64, c: i64) -> f64 {
    let idx = nsl_list_new();
    nsl_list_push(idx, r);
    nsl_list_push(idx, c);
    let v = nsl_tensor_get(t, idx);
    nsl_list_free(idx);
    v
}

#[test]
#[ignore]
fn rmsnorm_dx_gpu_matches_cpu_reference() {
    if !cuda_available() {
        eprintln!("skipping: no usable CUDA GPU");
        return;
    }
    let (rows, cols) = (3i64, 8i64);
    let eps = 1e-5_f64;
    // Deterministic, non-uniform inputs (so the reduction terms are nontrivial).
    let mut x = Vec::new();
    let mut dy = Vec::new();
    for i in 0..(rows * cols) {
        let f = i as f32;
        x.push(0.3 + 0.17 * (f % 5.0) - 0.05 * (f % 3.0));
        dy.push(0.02 * (f % 7.0) - 0.1);
    }
    let gamma: Vec<f32> = (0..cols).map(|j| 0.5 + 0.1 * j as f32).collect();

    // CPU reference.
    let (xc, dyc, gc) = (f32_2d(rows, cols, &x), f32_2d(rows, cols, &dy), f32_1d(&gamma));
    let dx_cpu = nsl_rmsnorm_dx_backward(dyc, xc, gc, eps);

    // GPU: move inputs to device, compute, pull back.
    let (xg, dyg, gg) = (
        nsl_tensor_to_device(xc, 1),
        nsl_tensor_to_device(dyc, 1),
        nsl_tensor_to_device(gc, 1),
    );
    let dx_gpu = nsl_rmsnorm_dx_backward(dyg, xg, gg, eps);
    let dx_gpu_cpu = nsl_tensor_to_device(dx_gpu, 0);

    let mut max_abs = 0.0_f64;
    for r in 0..rows {
        for c in 0..cols {
            let a = get2(dx_cpu, r, c);
            let b = get2(dx_gpu_cpu, r, c);
            max_abs = max_abs.max((a - b).abs());
            assert!(
                (a - b).abs() < 2e-4,
                "dx[{r},{c}] cpu={a} gpu={b} (|Δ|={})",
                (a - b).abs()
            );
        }
    }
    eprintln!("[rmsnorm-dx-gpu] max |cpu-gpu| = {max_abs:.2e}");

    for t in [xc, dyc, gc, dx_cpu, xg, dyg, gg, dx_gpu, dx_gpu_cpu] {
        nsl_tensor_free(t);
    }
}
