//! `nsl_sparse_spmm` with a BSR operand and a GPU dense operand, against a
//! product written out in f64 in this file.
//!
//! The host wrapper around the BSR kernel (`gpu_bsr_spmm_f32`) had three
//! defects that only sizes like these reach, and nothing ran it:
//!
//! - more than 64 output columns: it asked for `min(256, N) x 16` threads
//!   per block, over the limit of 1024, and the launch failed;
//! - a row count that is not a multiple of 16: the kernel writes whole
//!   16-row blocks into an output allocated as M x N;
//! - a column count of A that is not a multiple of 16: the kernel reads rows
//!   of B past K.
//!
//! Run: cargo test -p nsl-runtime --features cuda --test sparse_bsr_spmm_gpu
//!      -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::sparse::{nsl_sparse_free, nsl_sparse_from_dense, nsl_sparse_spmm};
use nsl_runtime::tensor::{nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_zeros_on};
use nsl_runtime::{nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d};

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_cuda_init() == 0
}

fn tensor_on(vals: &[f32], shape: &[i64], device: i64) -> i64 {
    let shape_list = nsl_list_new();
    for &dim in shape {
        nsl_list_push(shape_list, dim);
    }
    let t = nsl_tensor_zeros_on(shape_list, device);
    nsl_list_free(shape_list);
    if device == 0 {
        let dst = nsl_tensor_data_ptr(t) as *mut f32;
        for (i, &v) in vals.iter().enumerate() {
            unsafe { *dst.add(i) = v };
        }
    } else {
        nsl_test_cuda_h2d(nsl_tensor_data_ptr(t), vals.as_ptr() as i64, (vals.len() * 4) as i64);
    }
    t
}

/// Small integers, so every product and partial sum is exact in f32 and the
/// comparison below can be exact too.
fn det_ints(seed: u32, n: usize, zero_every: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|i| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            if zero_every > 0 && i % zero_every != 0 {
                0.0
            } else {
                (((s >> 16) % 7) as f32) - 3.0
            }
        })
        .collect()
}

fn check(m: usize, k: usize, n: usize) {
    // A is mostly zeros, with whole 16 x 16 blocks empty in its upper left
    // so that the block structure is not trivially dense.
    let mut a = det_ints(7, m * k, 3);
    for r in 0..m.min(16) {
        for c in 0..k.min(16) {
            a[r * k + c] = 0.0;
        }
    }
    let b = det_ints(11, k * n, 0);

    // Reference: the definition, in f64.
    let mut want = vec![0.0f64; m * n];
    for i in 0..m {
        for p in 0..k {
            let av = f64::from(a[i * k + p]);
            if av != 0.0 {
                for j in 0..n {
                    want[i * n + j] += av * f64::from(b[p * n + j]);
                }
            }
        }
    }

    let a_host = tensor_on(&a, &[m as i64, k as i64], 0);
    let bsr = nsl_sparse_from_dense(a_host, 3 /* BSR */, f64::to_bits(0.5) as i64);
    assert_ne!(bsr, 0);
    let b_gpu = tensor_on(&b, &[k as i64, n as i64], 1);

    let out = nsl_sparse_spmm(bsr, b_gpu);
    assert_ne!(out, 0, "[{m} x {k}] @ [{k} x {n}]: null result");
    let mut got = vec![0.0f32; m * n];
    nsl_test_cuda_d2h(got.as_mut_ptr() as i64, nsl_tensor_data_ptr(out), (m * n * 4) as i64);
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert_eq!(
            f64::from(*g),
            *w,
            "[{m} x {k}] @ [{k} x {n}]: element ({}, {}) is {g}, the product is {w}",
            i / n,
            i % n
        );
    }

    nsl_tensor_free(out);
    nsl_tensor_free(b_gpu);
    nsl_sparse_free(bsr);
    nsl_tensor_free(a_host);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn bsr_spmm_on_gpu_matches_the_product_for_awkward_sizes() {
    if !cuda_available() {
        return;
    }
    // (M, K, N): the aligned case; more than 64 output columns; M not a
    // multiple of 16; K not a multiple of 16; all three at once; and a
    // matrix smaller than one block.
    for (m, k, n) in [(32, 32, 8), (32, 32, 100), (20, 32, 8), (32, 40, 8), (37, 45, 300), (5, 3, 2)] {
        check(m, k, n);
    }
}
