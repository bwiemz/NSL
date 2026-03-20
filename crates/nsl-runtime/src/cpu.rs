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
