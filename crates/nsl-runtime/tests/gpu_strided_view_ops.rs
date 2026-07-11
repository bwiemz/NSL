//! Regression gates for GPU stride-blindness (the tied-embedding pretrain bug).
//!
//! `nsl_tensor_transpose` returns a zero-copy view (strides swapped, shared
//! storage). The GPU matmul kernels index flat row-major and never read
//! strides, so before the `nsl_tensor_matmul` GPU branch grew the same
//! `nsl_tensor_contiguous` guard the CPU path always had, a transposed view
//! as either operand computed a silently-WRONG product. Production impact:
//! the weight-tied LM head (`x @ embed.transpose(0,1)`, every stdlib
//! transformer) was miscomputed on GPU — coder-rl pretraining walked its loss
//! UP to the uniform plateau while CPU descended (PR #335's minimal repro).
//!
//! Same class, same fix, also gated here: `gpu_elementwise_unary` and
//! `gpu_scalar_op` on strided views.
//!
//! Run: cargo test -p nsl-runtime --features cuda --test gpu_strided_view_ops
//!      -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_tensor_data_ptr, nsl_tensor_matmul, nsl_tensor_transpose, nsl_tensor_zeros_on,
};
use nsl_runtime::{nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d};

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_cuda_init() == 0
}

fn gpu_tensor(vals: &[f32], shape: &[i64]) -> i64 {
    let shape_list = nsl_list_new();
    for &dim in shape {
        nsl_list_push(shape_list, dim);
    }
    let t = nsl_tensor_zeros_on(shape_list, 1 /* GPU */);
    nsl_list_free(shape_list);
    nsl_test_cuda_h2d(
        nsl_tensor_data_ptr(t),
        vals.as_ptr() as i64,
        (vals.len() * 4) as i64,
    );
    t
}

fn read_gpu(t: i64, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    nsl_test_cuda_d2h(
        out.as_mut_ptr() as i64,
        nsl_tensor_data_ptr(t),
        (len * 4) as i64,
    );
    out
}

/// Deterministic pseudo-random values (LCG), same generator as the flash GPU tests.
fn det_seq(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as f32 / 65535.0) - 0.5
        })
        .collect()
}

/// A[m,k] @ transpose_view(B[n,k]) must equal the CPU-computed A @ B^T.
/// Before the fix the GPU kernel read the [k,n]-shaped view's storage as if
/// contiguous (i.e. it computed A @ B-reinterpreted, not A @ B^T).
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_matmul_with_transposed_view_matches_reference() {
    if !cuda_available() {
        return;
    }
    let (m, k, n) = (8usize, 16usize, 64usize);
    let a_vals = det_seq(21, m * k);
    let b_vals = det_seq(22, n * k); // B is [n, k]; we multiply by its transpose view [k, n]

    // CPU reference: out[i][j] = sum_d a[i][d] * b[j][d]
    let mut expect = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f32;
            for d in 0..k {
                acc += a_vals[i * k + d] * b_vals[j * k + d];
            }
            expect[i * n + j] = acc;
        }
    }

    let a_t = gpu_tensor(&a_vals, &[m as i64, k as i64]);
    let b_t = gpu_tensor(&b_vals, &[n as i64, k as i64]);
    let b_view = nsl_tensor_transpose(b_t, 0, 1); // zero-copy [k, n] view
    let out = nsl_tensor_matmul(a_t, b_view, 0);
    let got = read_gpu(out, m * n);

    let max_diff = got
        .iter()
        .zip(expect.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "GPU matmul with transposed-view B diverged from reference \
         (max_abs_diff={max_diff:.3e}) — the stride-blindness regression is back"
    );
}

/// transpose_view(A[m,k]) @ B[m,n]: the LEFT operand as a view.
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_matmul_with_transposed_view_left_matches_reference() {
    if !cuda_available() {
        return;
    }
    let (m, k, n) = (16usize, 8usize, 12usize);
    let a_vals = det_seq(31, m * k); // A is [m, k]; view is [k, m]
    let b_vals = det_seq(32, m * n); // B is [m, n]; product view@B is [k, n]

    let mut expect = vec![0.0f32; k * n];
    for i in 0..k {
        for j in 0..n {
            let mut acc = 0.0f32;
            for d in 0..m {
                acc += a_vals[d * k + i] * b_vals[d * n + j];
            }
            expect[i * n + j] = acc;
        }
    }

    let a_t = gpu_tensor(&a_vals, &[m as i64, k as i64]);
    let b_t = gpu_tensor(&b_vals, &[m as i64, n as i64]);
    let a_view = nsl_tensor_transpose(a_t, 0, 1);
    let out = nsl_tensor_matmul(a_view, b_t, 0);
    let got = read_gpu(out, k * n);

    let max_diff = got
        .iter()
        .zip(expect.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "GPU matmul with transposed-view A diverged from reference \
         (max_abs_diff={max_diff:.3e})"
    );
}

use nsl_runtime::tensor::nsl_tensor_exp;
use nsl_runtime::tensor::nsl_tensor_mul_scalar;

/// exp() of a transposed GPU view must equal exp of the logically-transposed
/// values (gpu_elementwise_unary's contiguous guard). Before the guard, the
/// kernel read the view's storage flat, producing values in transposed order
/// under the view's shape.
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_unary_on_transposed_view_matches_reference() {
    if !cuda_available() {
        return;
    }
    let (r, c) = (5usize, 7usize);
    let vals = det_seq(41, r * c);
    let t = gpu_tensor(&vals, &[r as i64, c as i64]);
    let view = nsl_tensor_transpose(t, 0, 1); // [c, r] view
    let out = nsl_tensor_exp(view);
    let got = read_gpu(out, r * c);
    // expected[j][i] = exp(vals[i][j]) in [c, r] row-major
    let mut max_diff = 0.0f32;
    for j in 0..c {
        for i in 0..r {
            let e = vals[i * c + j].exp();
            max_diff = max_diff.max((got[j * r + i] - e).abs());
        }
    }
    assert!(
        max_diff < 1e-5,
        "GPU unary on transposed view diverged (max_abs_diff={max_diff:.3e})"
    );
}

/// scalar-mul of a transposed GPU view (gpu_scalar_op's contiguous guard).
#[test]
#[ignore = "requires CUDA GPU"]
fn gpu_scalar_op_on_transposed_view_matches_reference() {
    if !cuda_available() {
        return;
    }
    let (r, c) = (6usize, 4usize);
    let vals = det_seq(42, r * c);
    let t = gpu_tensor(&vals, &[r as i64, c as i64]);
    let view = nsl_tensor_transpose(t, 0, 1);
    let out = nsl_tensor_mul_scalar(view, 3.0, 0);
    let got = read_gpu(out, r * c);
    let mut max_diff = 0.0f32;
    for j in 0..c {
        for i in 0..r {
            let e = vals[i * c + j] * 3.0;
            max_diff = max_diff.max((got[j * r + i] - e).abs());
        }
    }
    assert!(
        max_diff < 1e-5,
        "GPU scalar op on transposed view diverged (max_abs_diff={max_diff:.3e})"
    );
}
