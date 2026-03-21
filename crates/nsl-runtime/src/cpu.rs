//! CPU implementations of tensor operations.
//! These are the original implementations extracted from tensor.rs.

use std::ffi::c_void;
use std::sync::atomic::AtomicI64;

use crate::memory::{checked_alloc, checked_alloc_zeroed};
use crate::tensor::NslTensor;

/// Elementwise binary op with NumPy-style broadcasting (f64 path).
pub(crate) fn tensor_elementwise_op(a_ptr: i64, b_ptr: i64, op: fn(f64, f64) -> f64) -> i64 {
    let a = NslTensor::from_ptr(a_ptr);
    let b = NslTensor::from_ptr(b_ptr);

    // Dispatch to f32 path if either tensor is f32
    if a.dtype == 1 || b.dtype == 1 {
        let op_f32 = {
            // We must convert the f64 op into an f32 op by wrapping
            // Use a closure that promotes to f64, applies op, demotes back
            #[allow(clippy::redundant_closure)]
            move |x: f32, y: f32| op(x as f64, y as f64) as f32
        };
        return tensor_elementwise_op_f32_impl(a_ptr, b_ptr, op_f32);
    }

    let a_ndim = a.ndim as usize;
    let b_ndim = b.ndim as usize;
    let out_ndim = a_ndim.max(b_ndim);

    // Build shapes right-aligned (NumPy broadcasting rules)
    let mut a_shape = vec![1i64; out_ndim];
    let mut b_shape = vec![1i64; out_ndim];
    for i in 0..a_ndim {
        a_shape[out_ndim - a_ndim + i] = unsafe { *a.shape.add(i) };
    }
    for i in 0..b_ndim {
        b_shape[out_ndim - b_ndim + i] = unsafe { *b.shape.add(i) };
    }

    // Compute output shape
    let mut out_shape_vec = vec![0i64; out_ndim];
    for i in 0..out_ndim {
        let da = a_shape[i];
        let db = b_shape[i];
        if da == db {
            out_shape_vec[i] = da;
        } else if da == 1 {
            out_shape_vec[i] = db;
        } else if db == 1 {
            out_shape_vec[i] = da;
        } else {
            eprintln!(
                "nsl: tensor shape mismatch in elementwise op (dim {}: {} vs {})",
                i, da, db
            );
            std::process::abort();
        }
    }

    let mut out_len: i64 = 1;
    for &s in &out_shape_vec {
        out_len *= s;
    }

    let shape = checked_alloc(out_ndim * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &s) in out_shape_vec.iter().enumerate().take(out_ndim) {
        unsafe { *shape.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape, out_ndim as i64);
    let data = checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    // Compute strides for a and b (0 for broadcast dims)
    let mut a_strides = vec![0i64; out_ndim];
    let mut b_strides = vec![0i64; out_ndim];
    {
        let mut s = 1i64;
        for i in (0..out_ndim).rev() {
            if a_shape[i] > 1 {
                a_strides[i] = s;
            }
            s *= a_shape[i];
        }
        s = 1;
        for i in (0..out_ndim).rev() {
            if b_shape[i] > 1 {
                b_strides[i] = s;
            }
            s *= b_shape[i];
        }
    }

    // Iterate over output elements using multi-index
    for flat in 0..out_len as usize {
        let mut rem = flat;
        let mut a_idx: usize = 0;
        let mut b_idx: usize = 0;
        for d in 0..out_ndim {
            let coord = rem / {
                let mut p = 1usize;
                for &sv in out_shape_vec.iter().take(out_ndim).skip(d + 1) {
                    p *= sv as usize;
                }
                p
            };
            rem %= {
                let mut p = 1usize;
                for &sv in out_shape_vec.iter().take(out_ndim).skip(d + 1) {
                    p *= sv as usize;
                }
                p
            };
            a_idx += coord * a_strides[d] as usize;
            b_idx += coord * b_strides[d] as usize;
        }
        unsafe { *data.add(flat) = op(*a.data_f64().add(a_idx), *b.data_f64().add(b_idx)) };
    }

    let result = Box::new(NslTensor {
        data: data as *mut c_void,
        shape,
        strides,
        ndim: out_ndim as i64,
        len: out_len,
        refcount: AtomicI64::new(1),
        device: 0,
        dtype: 0,
        owns_data: 1, data_owner: 0,
    });
    Box::into_raw(result) as i64
}

/// Elementwise binary op with NumPy-style broadcasting (f32 path).
pub(crate) fn tensor_elementwise_op_f32_impl(a_ptr: i64, b_ptr: i64, op: impl Fn(f32, f32) -> f32) -> i64 {
    let a = NslTensor::from_ptr(a_ptr);
    let b = NslTensor::from_ptr(b_ptr);

    let a_ndim = a.ndim as usize;
    let b_ndim = b.ndim as usize;
    let out_ndim = a_ndim.max(b_ndim);

    // Build shapes right-aligned (NumPy broadcasting rules)
    let mut a_shape = vec![1i64; out_ndim];
    let mut b_shape = vec![1i64; out_ndim];
    for i in 0..a_ndim {
        a_shape[out_ndim - a_ndim + i] = unsafe { *a.shape.add(i) };
    }
    for i in 0..b_ndim {
        b_shape[out_ndim - b_ndim + i] = unsafe { *b.shape.add(i) };
    }

    // Compute output shape
    let mut out_shape_vec = vec![0i64; out_ndim];
    for i in 0..out_ndim {
        let da = a_shape[i];
        let db = b_shape[i];
        if da == db {
            out_shape_vec[i] = da;
        } else if da == 1 {
            out_shape_vec[i] = db;
        } else if db == 1 {
            out_shape_vec[i] = da;
        } else {
            eprintln!(
                "nsl: tensor shape mismatch in elementwise op (dim {}: {} vs {})",
                i, da, db
            );
            std::process::abort();
        }
    }

    let mut out_len: i64 = 1;
    for &s in &out_shape_vec {
        out_len *= s;
    }

    let shape = checked_alloc(out_ndim * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &s) in out_shape_vec.iter().enumerate().take(out_ndim) {
        unsafe { *shape.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape, out_ndim as i64);
    let data = checked_alloc((out_len as usize) * std::mem::size_of::<f32>()) as *mut f32;

    // Compute strides for a and b (0 for broadcast dims)
    let mut a_strides = vec![0i64; out_ndim];
    let mut b_strides = vec![0i64; out_ndim];
    {
        let mut s = 1i64;
        for i in (0..out_ndim).rev() {
            if a_shape[i] > 1 {
                a_strides[i] = s;
            }
            s *= a_shape[i];
        }
        s = 1;
        for i in (0..out_ndim).rev() {
            if b_shape[i] > 1 {
                b_strides[i] = s;
            }
            s *= b_shape[i];
        }
    }

    // Helper to read element as f32 regardless of source dtype
    let read_a = |idx: usize| -> f32 {
        if a.dtype == 1 {
            unsafe { *a.data_f32().add(idx) }
        } else {
            unsafe { *a.data_f64().add(idx) as f32 }
        }
    };
    let read_b = |idx: usize| -> f32 {
        if b.dtype == 1 {
            unsafe { *b.data_f32().add(idx) }
        } else {
            unsafe { *b.data_f64().add(idx) as f32 }
        }
    };

    // Iterate over output elements using multi-index
    for flat in 0..out_len as usize {
        let mut rem = flat;
        let mut a_idx: usize = 0;
        let mut b_idx: usize = 0;
        for d in 0..out_ndim {
            let coord = rem / {
                let mut p = 1usize;
                for &sv in out_shape_vec.iter().take(out_ndim).skip(d + 1) {
                    p *= sv as usize;
                }
                p
            };
            rem %= {
                let mut p = 1usize;
                for &sv in out_shape_vec.iter().take(out_ndim).skip(d + 1) {
                    p *= sv as usize;
                }
                p
            };
            a_idx += coord * a_strides[d] as usize;
            b_idx += coord * b_strides[d] as usize;
        }
        unsafe { *data.add(flat) = op(read_a(a_idx), read_b(b_idx)) };
    }

    let result = Box::new(NslTensor {
        data: data as *mut c_void,
        shape,
        strides,
        ndim: out_ndim as i64,
        len: out_len,
        refcount: AtomicI64::new(1),
        device: 0,
        dtype: 1,
        owns_data: 1, data_owner: 0,
    });
    Box::into_raw(result) as i64
}

// ---------------------------------------------------------------------------
// Cache-tiled matrix multiply
// ---------------------------------------------------------------------------

const TILE_M: usize = 64;
const TILE_K: usize = 64;
const TILE_N: usize = 64;

/// Cache-tiled matrix multiply: C[m,n] += A[m,k] @ B[k,n]
/// C must be pre-zeroed. All matrices are row-major contiguous.
///
/// Tile loop order: jj (N tiles) -> kk (K tiles) -> ii (M tiles)
/// Inner loop order: i -> k -> j (maximizes B-tile reuse in registers)
pub fn tiled_matmul_f64(
    a: *const f64,
    b: *const f64,
    c: *mut f64,
    m: usize,
    k: usize,
    n: usize,
) {
    unsafe {
        let mut jj = 0;
        while jj < n {
            let j_end = (jj + TILE_N).min(n);
            let mut kk = 0;
            while kk < k {
                let k_end = (kk + TILE_K).min(k);
                let mut ii = 0;
                while ii < m {
                    let i_end = (ii + TILE_M).min(m);
                    for i in ii..i_end {
                        for ki in kk..k_end {
                            let a_val = *a.add(i * k + ki);
                            for j in jj..j_end {
                                *c.add(i * n + j) += a_val * *b.add(ki * n + j);
                            }
                        }
                    }
                    ii += TILE_M;
                }
                kk += TILE_K;
            }
            jj += TILE_N;
        }
    }
}

/// Cache-tiled matrix multiply for f32: C[m,n] += A[m,k] @ B[k,n]
pub fn tiled_matmul_f32(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) {
    unsafe {
        let mut jj = 0;
        while jj < n {
            let j_end = (jj + TILE_N).min(n);
            let mut kk = 0;
            while kk < k {
                let k_end = (kk + TILE_K).min(k);
                let mut ii = 0;
                while ii < m {
                    let i_end = (ii + TILE_M).min(m);
                    for i in ii..i_end {
                        for ki in kk..k_end {
                            let a_val = *a.add(i * k + ki);
                            for j in jj..j_end {
                                *c.add(i * n + j) += a_val * *b.add(ki * n + j);
                            }
                        }
                    }
                    ii += TILE_M;
                }
                kk += TILE_K;
            }
            jj += TILE_N;
        }
    }
}

/// Helper: get the shape of a tensor as a Vec<i64>.
pub(crate) fn get_shape_vec(tensor: &NslTensor) -> Vec<i64> {
    (0..tensor.ndim as usize)
        .map(|i| unsafe { *tensor.shape.add(i) })
        .collect()
}

/// Helper: get the strides of a tensor as a Vec<usize>.
pub(crate) fn get_strides_vec(tensor: &NslTensor) -> Vec<usize> {
    (0..tensor.ndim as usize)
        .map(|i| unsafe { *tensor.strides.add(i) } as usize)
        .collect()
}

/// Helper: create an f64 tensor with a given shape (Rust slice).
pub(crate) fn create_tensor_with_shape_rs(shape: &[i64]) -> i64 {
    create_tensor_with_shape_rs_dtype(shape, 0)
}

/// Helper: create a tensor with a given shape and dtype (Rust slice).
/// dtype=0 → f64, dtype=1 → f32
pub(crate) fn create_tensor_with_shape_rs_dtype(shape: &[i64], dtype: u16) -> i64 {
    let ndim = shape.len() as i64;
    let mut total: i64 = 1;
    for &s in shape {
        total *= s;
    }

    let shape_ptr =
        checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
    for (i, &s) in shape.iter().enumerate() {
        unsafe { *shape_ptr.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape_ptr, ndim);

    let elem_size = if dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let data = checked_alloc_zeroed((total as usize) * elem_size);

    let tensor = Box::new(NslTensor {
        data: data as *mut c_void,
        shape: shape_ptr,
        strides,
        ndim,
        len: total,
        refcount: AtomicI64::new(1),
        device: 0,
        dtype,
        owns_data: 1, data_owner: 0,
    });
    Box::into_raw(tensor) as i64
}

#[cfg(test)]
mod matmul_tests {
    use super::*;

    #[test]
    fn test_tiled_matmul_f64_small() {
        // 2x3 @ 3x4 = 2x4
        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f64> = vec![
            7.0, 8.0, 9.0, 10.0,
            11.0, 12.0, 13.0, 14.0,
            15.0, 16.0, 17.0, 18.0,
        ];
        let mut c = vec![0.0f64; 8];

        tiled_matmul_f64(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 3, 4);

        assert_eq!(c, vec![74.0, 80.0, 86.0, 92.0, 173.0, 188.0, 203.0, 218.0]);
    }

    #[test]
    fn test_tiled_matmul_f64_identity() {
        let a: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
        let b: Vec<f64> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let mut c = vec![0.0f64; 9];

        tiled_matmul_f64(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 3, 3, 3);

        assert_eq!(c, a);
    }

    #[test]
    fn test_tiled_matmul_f64_large() {
        // 128x256 @ 256x64 — exercises tiling with exact tile boundaries
        let m = 128;
        let k = 256;
        let n = 64;
        let a: Vec<f64> = (0..m * k).map(|i| (i % 7) as f64 * 0.1).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 11) as f64 * 0.1).collect();
        let mut c_tiled = vec![0.0f64; m * n];
        let mut c_naive = vec![0.0f64; m * n];

        tiled_matmul_f64(a.as_ptr(), b.as_ptr(), c_tiled.as_mut_ptr(), m, k, n);

        // Naive reference
        for i in 0..m {
            for j in 0..k {
                let a_val = a[i * k + j];
                for l in 0..n {
                    c_naive[i * n + l] += a_val * b[j * n + l];
                }
            }
        }

        for i in 0..m * n {
            assert!((c_tiled[i] - c_naive[i]).abs() < 1e-6,
                "mismatch at {}: {} vs {}", i, c_tiled[i], c_naive[i]);
        }
    }

    #[test]
    fn test_tiled_matmul_f64_non_tile_aligned() {
        // 17x33 @ 33x5 — not aligned to any power of 2
        let m = 17;
        let k = 33;
        let n = 5;
        let a: Vec<f64> = (0..m * k).map(|i| (i as f64) * 0.01).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i as f64) * 0.01).collect();
        let mut c_tiled = vec![0.0f64; m * n];
        let mut c_naive = vec![0.0f64; m * n];

        tiled_matmul_f64(a.as_ptr(), b.as_ptr(), c_tiled.as_mut_ptr(), m, k, n);

        for i in 0..m {
            for j in 0..k {
                let a_val = a[i * k + j];
                for l in 0..n {
                    c_naive[i * n + l] += a_val * b[j * n + l];
                }
            }
        }

        for i in 0..m * n {
            assert!((c_tiled[i] - c_naive[i]).abs() < 1e-6,
                "mismatch at {}: {} vs {}", i, c_tiled[i], c_naive[i]);
        }
    }

    #[test]
    fn test_tiled_matmul_f32_small() {
        let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let b: Vec<f32> = vec![7.0, 8.0, 11.0, 12.0, 15.0, 16.0];
        let mut c = vec![0.0f32; 4];

        tiled_matmul_f32(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 2, 3, 2);

        assert_eq!(c, vec![74.0, 80.0, 173.0, 188.0]);
    }

    #[test]
    fn test_tiled_matmul_single_element() {
        let a = vec![3.0f64];
        let b = vec![5.0f64];
        let mut c = vec![0.0f64];
        tiled_matmul_f64(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), 1, 1, 1);
        assert_eq!(c[0], 15.0);
    }
}
