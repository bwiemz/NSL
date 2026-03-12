//! CPU implementations of tensor operations.
//! These are the original implementations extracted from tensor.rs.

use std::ffi::c_void;

use crate::memory::{checked_alloc, checked_alloc_zeroed};
use crate::tensor::NslTensor;

/// Elementwise binary op with NumPy-style broadcasting.
pub(crate) fn tensor_elementwise_op(a_ptr: i64, b_ptr: i64, op: fn(f64, f64) -> f64) -> i64 {
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
    for i in 0..out_ndim {
        unsafe { *shape.add(i) = out_shape_vec[i] };
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
            let _dim_size = out_shape_vec[d] as usize;
            let coord = rem / {
                let mut p = 1usize;
                for dd in (d + 1)..out_ndim {
                    p *= out_shape_vec[dd] as usize;
                }
                p
            };
            rem %= {
                let mut p = 1usize;
                for dd in (d + 1)..out_ndim {
                    p *= out_shape_vec[dd] as usize;
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
        refcount: 1,
        device: 0,
        dtype: 0,
        owns_data: 1,
    });
    Box::into_raw(result) as i64
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

/// Helper: create a tensor with a given shape (Rust slice).
pub(crate) fn create_tensor_with_shape_rs(shape: &[i64]) -> i64 {
    let ndim = shape.len() as i64;
    let mut total: i64 = 1;
    for &s in shape {
        total *= s;
    }

    let shape_ptr =
        checked_alloc(shape.len() * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &s) in shape.iter().enumerate() {
        unsafe { *shape_ptr.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape_ptr, ndim);
    let data = checked_alloc_zeroed((total as usize) * std::mem::size_of::<f64>()) as *mut f64;

    let tensor = Box::new(NslTensor {
        data: data as *mut c_void,
        shape: shape_ptr,
        strides,
        ndim,
        len: total,
        refcount: 1,
        device: 0,
        dtype: 0,
        owns_data: 1,
    });
    Box::into_raw(tensor) as i64
}
