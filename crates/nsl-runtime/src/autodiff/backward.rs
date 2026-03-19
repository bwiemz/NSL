use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::AtomicI64;

use crate::list::NslList;
use crate::tensor::{
    nsl_tensor_clone as tensor_clone,
    nsl_tensor_div as tensor_div,
    nsl_tensor_free as tensor_free,
    nsl_tensor_item as tensor_item,
    nsl_tensor_matmul as tensor_matmul,
    nsl_tensor_mul as tensor_mul,
    nsl_tensor_mul_scalar as tensor_mul_scalar,
    nsl_tensor_neg as tensor_neg,
    nsl_tensor_select as nsl_tensor_select,
    nsl_tensor_shape as tensor_shape,
    nsl_tensor_sign as tensor_sign,
    nsl_tensor_sum_dim as nsl_tensor_sum_dim,
    nsl_tensor_transpose as tensor_transpose,
    nsl_tensor_zeros as tensor_zeros,
    NslTensor,
};

use super::grad_utils::{
    accumulate_grad, broadcast_grad_along_dim, create_tensor_with_shape_dtype,
    create_tensor_with_shape_dtype_device, ones_like, reduce_grad_for_broadcast, reshape_to_shape,
    scatter_gather_grad, scatter_grad_to_argmax,
};
use super::{ones_from_shape, TapeOp, TAPE};

/// ReLU backward: grad * (input > 0 ? 1 : 0)
fn relu_backward(grad_ptr: i64, input_ptr: i64) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    #[cfg(feature = "cuda")]
    if grad.device > 0 {
        return crate::cuda::gpu_relu_backward(grad_ptr, input_ptr);
    }
    let input = NslTensor::from_ptr(input_ptr);
    let len = input.len as usize;
    let ndim = input.ndim;
    let in_dtype = input.dtype;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let elem_size = if in_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let data_raw = crate::memory::checked_alloc(len * elem_size);
    if in_dtype == 1 {
        let data = data_raw as *mut f32;
        for i in 0..len {
            let x = unsafe { *input.data_f32().add(i) };
            let g = unsafe { *grad.data_f32().add(i) };
            unsafe { *data.add(i) = if x > 0.0 { g } else { 0.0 } };
        }
    } else {
        let data = data_raw as *mut f64;
        for i in 0..len {
            let x = unsafe { *input.data_f64().add(i) };
            let g = unsafe { *grad.data_f64().add(i) };
            unsafe { *data.add(i) = if x > 0.0 { g } else { 0.0 } };
        }
    }
    let t = Box::new(NslTensor { data: data_raw as *mut c_void, shape, strides, ndim, len: len as i64, refcount: AtomicI64::new(1), device: 0, dtype: in_dtype, owns_data: 1 });
    Box::into_raw(t) as i64
}

/// GELU backward: gelu'(x) = 0.5*(1+tanh(c*(x+0.044715*x^3))) + 0.5*x*sech^2(c*(x+0.044715*x^3))*c*(1+3*0.044715*x^2)
fn gelu_backward(grad_ptr: i64, input_ptr: i64) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    #[cfg(feature = "cuda")]
    if grad.device > 0 {
        return crate::cuda::gpu_gelu_backward(grad_ptr, input_ptr);
    }
    let input = NslTensor::from_ptr(input_ptr);
    let len = input.len as usize;
    let ndim = input.ndim;
    let in_dtype = input.dtype;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let elem_size = if in_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let data_raw = crate::memory::checked_alloc(len * elem_size);
    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
    if in_dtype == 1 {
        let data = data_raw as *mut f32;
        let c_f32 = c as f32;
        for i in 0..len {
            let x = unsafe { *input.data_f32().add(i) } as f64;
            let g = unsafe { *grad.data_f32().add(i) } as f64;
            let inner = c * (x + 0.044715 * x * x * x);
            let tanh_inner = inner.tanh();
            let sech2 = 1.0 - tanh_inner * tanh_inner;
            let deriv = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x * x);
            unsafe { *data.add(i) = (g * deriv) as f32 };
            let _ = c_f32; // suppress unused warning
        }
    } else {
        let data = data_raw as *mut f64;
        for i in 0..len {
            let x = unsafe { *input.data_f64().add(i) };
            let g = unsafe { *grad.data_f64().add(i) };
            let inner = c * (x + 0.044715 * x * x * x);
            let tanh_inner = inner.tanh();
            let sech2 = 1.0 - tanh_inner * tanh_inner;
            let deriv = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x * x);
            unsafe { *data.add(i) = g * deriv };
        }
    }
    let t = Box::new(NslTensor { data: data_raw as *mut c_void, shape, strides, ndim, len: len as i64, refcount: AtomicI64::new(1), device: 0, dtype: in_dtype, owns_data: 1 });
    Box::into_raw(t) as i64
}

/// SiLU backward: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
fn silu_backward(grad_ptr: i64, input_ptr: i64) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    #[cfg(feature = "cuda")]
    if grad.device > 0 {
        return crate::cuda::gpu_silu_backward(grad_ptr, input_ptr);
    }
    let input = NslTensor::from_ptr(input_ptr);
    let len = input.len as usize;
    let ndim = input.ndim;
    let in_dtype = input.dtype;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let elem_size = if in_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let data_raw = crate::memory::checked_alloc(len * elem_size);
    if in_dtype == 1 {
        let data = data_raw as *mut f32;
        for i in 0..len {
            let x = unsafe { *input.data_f32().add(i) } as f64;
            let g = unsafe { *grad.data_f32().add(i) } as f64;
            let sig = 1.0 / (1.0 + (-x).exp());
            let deriv = sig * (1.0 + x * (1.0 - sig));
            unsafe { *data.add(i) = (g * deriv) as f32 };
        }
    } else {
        let data = data_raw as *mut f64;
        for i in 0..len {
            let x = unsafe { *input.data_f64().add(i) };
            let g = unsafe { *grad.data_f64().add(i) };
            let sig = 1.0 / (1.0 + (-x).exp());
            let deriv = sig * (1.0 + x * (1.0 - sig));
            unsafe { *data.add(i) = g * deriv };
        }
    }
    let t = Box::new(NslTensor { data: data_raw as *mut c_void, shape, strides, ndim, len: len as i64, refcount: AtomicI64::new(1), device: 0, dtype: in_dtype, owns_data: 1 });
    Box::into_raw(t) as i64
}

/// Sigmoid backward: out * (1 - out)
fn sigmoid_backward(grad_ptr: i64, out_ptr: i64) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    #[cfg(feature = "cuda")]
    if grad.device > 0 {
        return crate::cuda::gpu_sigmoid_backward(grad_ptr, out_ptr);
    }
    let out = NslTensor::from_ptr(out_ptr);
    let len = out.len as usize;
    let ndim = out.ndim;
    let out_dtype = out.dtype;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(out.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let elem_size = if out_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let data_raw = crate::memory::checked_alloc(len * elem_size);
    if out_dtype == 1 {
        let data = data_raw as *mut f32;
        for i in 0..len {
            let o = unsafe { *out.data_f32().add(i) };
            let g = unsafe { *grad.data_f32().add(i) };
            unsafe { *data.add(i) = g * o * (1.0 - o) };
        }
    } else {
        let data = data_raw as *mut f64;
        for i in 0..len {
            let o = unsafe { *out.data_f64().add(i) };
            let g = unsafe { *grad.data_f64().add(i) };
            unsafe { *data.add(i) = g * o * (1.0 - o) };
        }
    }
    let t = Box::new(NslTensor { data: data_raw as *mut c_void, shape, strides, ndim, len: len as i64, refcount: AtomicI64::new(1), device: 0, dtype: out_dtype, owns_data: 1 });
    Box::into_raw(t) as i64
}

/// Tanh backward: 1 - out^2
fn tanh_backward(grad_ptr: i64, out_ptr: i64) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    #[cfg(feature = "cuda")]
    if grad.device > 0 {
        return crate::cuda::gpu_tanh_backward(grad_ptr, out_ptr);
    }
    let out = NslTensor::from_ptr(out_ptr);
    let len = out.len as usize;
    let ndim = out.ndim;
    let out_dtype = out.dtype;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(out.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let elem_size = if out_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let data_raw = crate::memory::checked_alloc(len * elem_size);
    if out_dtype == 1 {
        let data = data_raw as *mut f32;
        for i in 0..len {
            let o = unsafe { *out.data_f32().add(i) };
            let g = unsafe { *grad.data_f32().add(i) };
            unsafe { *data.add(i) = g * (1.0 - o * o) };
        }
    } else {
        let data = data_raw as *mut f64;
        for i in 0..len {
            let o = unsafe { *out.data_f64().add(i) };
            let g = unsafe { *grad.data_f64().add(i) };
            unsafe { *data.add(i) = g * (1.0 - o * o) };
        }
    }
    let t = Box::new(NslTensor { data: data_raw as *mut c_void, shape, strides, ndim, len: len as i64, refcount: AtomicI64::new(1), device: 0, dtype: out_dtype, owns_data: 1 });
    Box::into_raw(t) as i64
}

/// Softmax backward: grad_input_i = output_i * (grad_i - sum(grad * output)) along dim
fn softmax_backward(grad_ptr: i64, out_ptr: i64, dim: i64) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    let out = NslTensor::from_ptr(out_ptr);
    let on_gpu = grad.device > 0;
    #[cfg(feature = "cuda")]
    if on_gpu {
        unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    }
    let len = out.len as usize;
    let ndim = out.ndim;
    let out_dtype = out.dtype;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(out.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let elem_size = if out_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let data_size = len * elem_size;
    let data_raw: *mut c_void = if on_gpu {
        #[cfg(feature = "cuda")]
        { crate::cuda::inner::alloc_managed(data_size) }
        #[cfg(not(feature = "cuda"))]
        { crate::memory::checked_alloc_zeroed(data_size) as *mut c_void }
    } else {
        crate::memory::checked_alloc_zeroed(data_size) as *mut c_void
    };

    let d = if dim < 0 { (ndim + dim) as usize } else { dim as usize };
    let o_shape: Vec<i64> = (0..ndim as usize).map(|i| unsafe { *out.shape.add(i) }).collect();
    let o_strides: Vec<i64> = (0..ndim as usize).map(|i| unsafe { *out.strides.add(i) }).collect();
    let dim_size = o_shape[d] as usize;
    let num_slices = len / dim_size;

    for slice_idx in 0..num_slices {
        let mut remaining = slice_idx;
        let mut base_offset: usize = 0;
        for axis in (0..ndim as usize).rev() {
            if axis == d { continue; }
            let idx = remaining % (o_shape[axis] as usize);
            remaining /= o_shape[axis] as usize;
            base_offset += idx * (o_strides[axis] as usize);
        }

        let mut dot = 0.0_f64;
        for k in 0..dim_size {
            let offset = base_offset + k * (o_strides[d] as usize);
            let g = if out_dtype == 1 { unsafe { *grad.data_f32().add(offset) as f64 } } else { unsafe { *grad.data_f64().add(offset) } };
            let o = if out_dtype == 1 { unsafe { *out.data_f32().add(offset) as f64 } } else { unsafe { *out.data_f64().add(offset) } };
            dot += g * o;
        }

        for k in 0..dim_size {
            let offset = base_offset + k * (o_strides[d] as usize);
            let g = if out_dtype == 1 { unsafe { *grad.data_f32().add(offset) as f64 } } else { unsafe { *grad.data_f64().add(offset) } };
            let o = if out_dtype == 1 { unsafe { *out.data_f32().add(offset) as f64 } } else { unsafe { *out.data_f64().add(offset) } };
            let val = o * (g - dot);
            if out_dtype == 1 {
                unsafe { *(data_raw as *mut f32).add(offset) = val as f32 };
            } else {
                unsafe { *(data_raw as *mut f64).add(offset) = val };
            }
        }
    }

    let t = Box::new(NslTensor { data: data_raw, shape, strides, ndim, len: len as i64, refcount: AtomicI64::new(1), device: grad.device, dtype: out_dtype, owns_data: 1 });
    Box::into_raw(t) as i64
}

/// Slice backward: create zeros with input_shape, copy grad into the [start, start+slice_len) region along dim.
fn slice_backward(grad_ptr: i64, input_shape: &[i64], dim: usize, start: usize) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    let ndim = input_shape.len();

    let grad_dtype = grad.dtype;
    // Create zero output with original input_shape, matching grad dtype
    let out_ptr = create_tensor_with_shape_dtype(input_shape, 0.0, grad_dtype);
    let out = NslTensor::from_ptr(out_ptr);

    let out_strides: Vec<usize> = (0..ndim)
        .map(|i| unsafe { *out.strides.add(i) } as usize)
        .collect();
    let grad_strides: Vec<usize> = (0..ndim)
        .map(|i| unsafe { *grad.strides.add(i) } as usize)
        .collect();

    let grad_total = grad.len as usize;
    for flat in 0..grad_total {
        let mut remaining = flat;
        let mut out_offset: usize = 0;
        for axis in 0..ndim {
            let idx = remaining / grad_strides[axis];
            remaining %= grad_strides[axis];
            if axis == dim {
                out_offset += (idx + start) * out_strides[axis];
            } else {
                out_offset += idx * out_strides[axis];
            }
        }
        if grad_dtype == 1 {
            unsafe { *out.data_f32().add(out_offset) = *grad.data_f32().add(flat) };
        } else {
            unsafe { *out.data_f64().add(out_offset) = *grad.data_f64().add(flat) };
        }
    }

    out_ptr
}

/// Cat backward: split the gradient along the cat dim into pieces matching the original input sizes.
fn cat_backward(grad_ptr: i64, dim: usize, split_sizes: &[i64]) -> Vec<i64> {
    let grad = NslTensor::from_ptr(grad_ptr);
    let ndim = grad.ndim as usize;
    let grad_dtype = grad.dtype;

    let grad_shape: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *grad.shape.add(i) })
        .collect();
    let grad_strides: Vec<usize> = (0..ndim)
        .map(|i| unsafe { *grad.strides.add(i) } as usize)
        .collect();

    let mut results = Vec::with_capacity(split_sizes.len());
    let mut offset: usize = 0;

    for &split_size in split_sizes {
        let mut piece_shape = grad_shape.clone();
        piece_shape[dim] = split_size;

        let piece_ptr = create_tensor_with_shape_dtype(&piece_shape, 0.0, grad_dtype);
        let piece = NslTensor::from_ptr(piece_ptr);
        let piece_strides: Vec<usize> = (0..ndim)
            .map(|i| unsafe { *piece.strides.add(i) } as usize)
            .collect();

        let piece_total = piece.len as usize;
        for flat in 0..piece_total {
            let mut remaining = flat;
            let mut grad_offset: usize = 0;
            for axis in 0..ndim {
                let idx = remaining / piece_strides[axis];
                remaining %= piece_strides[axis];
                if axis == dim {
                    grad_offset += (idx + offset) * grad_strides[axis];
                } else {
                    grad_offset += idx * grad_strides[axis];
                }
            }
            if grad_dtype == 1 {
                unsafe { *piece.data_f32().add(flat) = *grad.data_f32().add(grad_offset) };
            } else {
                unsafe { *piece.data_f64().add(flat) = *grad.data_f64().add(grad_offset) };
            }
        }

        results.push(piece_ptr);
        offset += split_size as usize;
    }

    results
}

/// LayerNorm backward: returns (grad_input, grad_weight, grad_bias)
/// Per row of size N:
///   d_normalized = d_out * weight
///   d_x = (1/N) * inv_std * (N * d_normalized - sum(d_normalized) - normalized * sum(d_normalized * normalized))
///   d_weight = sum(d_out * normalized, across batch)
///   d_bias = sum(d_out, across batch)
fn layernorm_backward(grad_ptr: i64, input_ptr: i64, mean_ptr: i64, inv_std_ptr: i64, weight_ptr: i64) -> (i64, i64, i64) {
    let grad = NslTensor::from_ptr(grad_ptr);
    let input = NslTensor::from_ptr(input_ptr);
    let mean_t = NslTensor::from_ptr(mean_ptr);   // always f64
    let inv_std_t = NslTensor::from_ptr(inv_std_ptr); // always f64
    let weight = NslTensor::from_ptr(weight_ptr);

    let out_device = grad.device;
    #[cfg(feature = "cuda")]
    if out_device > 0 {
        unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    }

    let in_dtype = input.dtype;
    let total = input.len as usize;
    let ndim = input.ndim as usize;
    let n = unsafe { *input.shape.add(ndim - 1) } as usize;
    let num_rows = total / n;
    let nf = n as f64;

    let read_grad = |i: usize| -> f64 {
        if grad.dtype == 1 { unsafe { *grad.data_f32().add(i) as f64 } } else { unsafe { *grad.data_f64().add(i) } }
    };
    let read_weight = |i: usize| -> f64 {
        if weight.dtype == 1 { unsafe { *weight.data_f32().add(i) as f64 } } else { unsafe { *weight.data_f64().add(i) } }
    };
    let read_input = |i: usize| -> f64 {
        if in_dtype == 1 { unsafe { *input.data_f32().add(i) as f64 } } else { unsafe { *input.data_f64().add(i) } }
    };

    // grad_input: same shape as input (matches input dtype)
    let dx_ptr = create_tensor_with_shape_dtype_device(
        &(0..ndim).map(|i| unsafe { *input.shape.add(i) }).collect::<Vec<_>>(),
        0.0, in_dtype, out_device,
    );
    let dx = NslTensor::from_ptr(dx_ptr);

    // grad_weight and grad_bias: shape [N], match weight dtype
    let dw_ptr = create_tensor_with_shape_dtype_device(&[n as i64], 0.0, weight.dtype, out_device);
    let dw = NslTensor::from_ptr(dw_ptr);
    let db_ptr = create_tensor_with_shape_dtype_device(&[n as i64], 0.0, weight.dtype, out_device);
    let db = NslTensor::from_ptr(db_ptr);

    let write_dx = |i: usize, val: f64| {
        if in_dtype == 1 { unsafe { *dx.data_f32().add(i) = val as f32 } }
        else { unsafe { *dx.data_f64().add(i) = val } }
    };
    let write_dw = |i: usize, val: f64| {
        if dw.dtype == 1 { unsafe { *dw.data_f32().add(i) = (*dw.data_f32().add(i) as f64 + val) as f32 } }
        else { unsafe { *dw.data_f64().add(i) += val } }
    };
    let write_db = |i: usize, val: f64| {
        if db.dtype == 1 { unsafe { *db.data_f32().add(i) = (*db.data_f32().add(i) as f64 + val) as f32 } }
        else { unsafe { *db.data_f64().add(i) += val } }
    };

    for row in 0..num_rows {
        let base = row * n;
        let mean_val = unsafe { *mean_t.data_f64().add(row) };
        let inv_std_val = unsafe { *inv_std_t.data_f64().add(row) };

        let mut sum_dnorm = 0.0_f64;
        let mut sum_dnorm_norm = 0.0_f64;

        for j in 0..n {
            let g = read_grad(base + j);
            let w = read_weight(j);
            let x = read_input(base + j);
            let normalized = (x - mean_val) * inv_std_val;
            let d_normalized = g * w;
            sum_dnorm += d_normalized;
            sum_dnorm_norm += d_normalized * normalized;

            write_dw(j, g * normalized);
            write_db(j, g);
        }

        for j in 0..n {
            let g = read_grad(base + j);
            let w = read_weight(j);
            let x = read_input(base + j);
            let normalized = (x - mean_val) * inv_std_val;
            let d_normalized = g * w;
            let d_x = (1.0 / nf) * inv_std_val * (nf * d_normalized - sum_dnorm - normalized * sum_dnorm_norm);
            write_dx(base + j, d_x);
        }
    }

    (dx_ptr, dw_ptr, db_ptr)
}

/// RMSNorm backward: returns (grad_input, grad_weight)
/// Per row of size N:
///   d_x = weight * (d_out / rms - x * sum(d_out * x) / (N * rms^3))
///   d_weight = sum(d_out * (x / rms), across batch)
fn rmsnorm_backward(grad_ptr: i64, input_ptr: i64, rms_ptr: i64, weight_ptr: i64) -> (i64, i64) {
    let grad = NslTensor::from_ptr(grad_ptr);
    let input = NslTensor::from_ptr(input_ptr);
    let rms_t = NslTensor::from_ptr(rms_ptr); // always f64
    let weight = NslTensor::from_ptr(weight_ptr);

    let out_device = grad.device;
    #[cfg(feature = "cuda")]
    if out_device > 0 {
        unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
    }

    let in_dtype = input.dtype;
    let total = input.len as usize;
    let ndim = input.ndim as usize;
    let n = unsafe { *input.shape.add(ndim - 1) } as usize;
    let num_rows = total / n;
    let nf = n as f64;

    let read_grad = |i: usize| -> f64 {
        if grad.dtype == 1 { unsafe { *grad.data_f32().add(i) as f64 } } else { unsafe { *grad.data_f64().add(i) } }
    };
    let read_weight = |i: usize| -> f64 {
        if weight.dtype == 1 { unsafe { *weight.data_f32().add(i) as f64 } } else { unsafe { *weight.data_f64().add(i) } }
    };
    let read_input = |i: usize| -> f64 {
        if in_dtype == 1 { unsafe { *input.data_f32().add(i) as f64 } } else { unsafe { *input.data_f64().add(i) } }
    };

    // grad_input: same shape as input (matches input dtype)
    let dx_ptr = create_tensor_with_shape_dtype_device(
        &(0..ndim).map(|i| unsafe { *input.shape.add(i) }).collect::<Vec<_>>(),
        0.0, in_dtype, out_device,
    );
    let dx = NslTensor::from_ptr(dx_ptr);

    // grad_weight: shape [N], match weight dtype
    let dw_ptr = create_tensor_with_shape_dtype_device(&[n as i64], 0.0, weight.dtype, out_device);
    let dw = NslTensor::from_ptr(dw_ptr);

    let write_dx = |i: usize, val: f64| {
        if in_dtype == 1 { unsafe { *dx.data_f32().add(i) = val as f32 } }
        else { unsafe { *dx.data_f64().add(i) = val } }
    };
    let write_dw = |i: usize, val: f64| {
        if dw.dtype == 1 { unsafe { *dw.data_f32().add(i) = (*dw.data_f32().add(i) as f64 + val) as f32 } }
        else { unsafe { *dw.data_f64().add(i) += val } }
    };

    for row in 0..num_rows {
        let base = row * n;
        let rms_val = unsafe { *rms_t.data_f64().add(row) }.max(1e-12); // IMPORTANT-1: epsilon guard against underflow
        let rms_cubed = rms_val * rms_val * rms_val;

        let mut sum_dout_x = 0.0_f64;
        for j in 0..n {
            let g = read_grad(base + j);
            let w = read_weight(j);
            let x = read_input(base + j);
            sum_dout_x += g * w * x;
        }

        for j in 0..n {
            let g = read_grad(base + j);
            let w = read_weight(j);
            let x = read_input(base + j);

            let d_x = w * (g / rms_val - x * sum_dout_x / (nf * rms_cubed));
            write_dx(base + j, d_x);
            write_dw(j, g * (x / rms_val));
        }
    }

    (dx_ptr, dw_ptr)
}

/// Dropout backward: grad_input = grad_output * mask * scale
fn dropout_backward(grad_ptr: i64, mask_ptr: i64, scale: f64) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    let mask = NslTensor::from_ptr(mask_ptr); // always f64
    let len = grad.len as usize;
    let ndim = grad.ndim;
    let grad_dtype = grad.dtype;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(grad.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let elem_size = if grad_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let data_raw = crate::memory::checked_alloc(len * elem_size);
    if grad_dtype == 1 {
        let data = data_raw as *mut f32;
        for i in 0..len {
            let g = unsafe { *grad.data_f32().add(i) as f64 };
            let m = unsafe { *mask.data_f64().add(i) };
            unsafe { *data.add(i) = (g * m * scale) as f32 };
        }
    } else {
        let data = data_raw as *mut f64;
        for i in 0..len {
            let g = unsafe { *grad.data_f64().add(i) };
            let m = unsafe { *mask.data_f64().add(i) };
            unsafe { *data.add(i) = g * m * scale };
        }
    }
    let t = Box::new(NslTensor { data: data_raw as *mut c_void, shape, strides, ndim, len: len as i64, refcount: AtomicI64::new(1), device: 0, dtype: grad_dtype, owns_data: 1 });
    Box::into_raw(t) as i64
}

/// Conv2d backward: returns (grad_input, grad_weight, grad_bias)
fn conv2d_backward(
    grad_ptr: i64,
    input_ptr: i64,
    weight_ptr: i64,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> (i64, i64, i64) {
    let grad = NslTensor::from_ptr(grad_ptr);
    let input = NslTensor::from_ptr(input_ptr);
    let weight = NslTensor::from_ptr(weight_ptr);

    let n = unsafe { *input.shape.add(0) } as usize;
    let c_in = unsafe { *input.shape.add(1) } as usize;
    let h = unsafe { *input.shape.add(2) } as usize;
    let w = unsafe { *input.shape.add(3) } as usize;

    let c_out = unsafe { *weight.shape.add(0) } as usize;
    let kh = unsafe { *weight.shape.add(2) } as usize;
    let kw = unsafe { *weight.shape.add(3) } as usize;

    let h_out = unsafe { *grad.shape.add(2) } as usize;
    let w_out = unsafe { *grad.shape.add(3) } as usize;

    let in_dtype = input.dtype;
    let grad_dtype = grad.dtype;
    // Output grads match their respective source dtypes
    let dx_dtype = in_dtype;
    let dw_dtype = weight.dtype;

    let read_grad = |i: usize| -> f64 {
        if grad_dtype == 1 { unsafe { *grad.data_f32().add(i) as f64 } } else { unsafe { *grad.data_f64().add(i) } }
    };
    let read_input = |i: usize| -> f64 {
        if in_dtype == 1 { unsafe { *input.data_f32().add(i) as f64 } } else { unsafe { *input.data_f64().add(i) } }
    };
    let read_weight = |i: usize| -> f64 {
        if weight.dtype == 1 { unsafe { *weight.data_f32().add(i) as f64 } } else { unsafe { *weight.data_f64().add(i) } }
    };

    // Allocate grad_input [N, C_in, H, W]
    let dx_ptr = create_tensor_with_shape_dtype(&[n as i64, c_in as i64, h as i64, w as i64], 0.0, dx_dtype);
    let dx = NslTensor::from_ptr(dx_ptr);

    // Allocate grad_weight [C_out, C_in, kH, kW]
    let dw_ptr = create_tensor_with_shape_dtype(&[c_out as i64, c_in as i64, kh as i64, kw as i64], 0.0, dw_dtype);
    let dw = NslTensor::from_ptr(dw_ptr);

    // Allocate grad_bias [C_out] — matches weight dtype
    let db_ptr = create_tensor_with_shape_dtype(&[c_out as i64], 0.0, dw_dtype);
    let db = NslTensor::from_ptr(db_ptr);

    let add_dx = |i: usize, val: f64| {
        if dx_dtype == 1 { unsafe { *dx.data_f32().add(i) = (*dx.data_f32().add(i) as f64 + val) as f32 } }
        else { unsafe { *dx.data_f64().add(i) += val } }
    };
    let add_dw = |i: usize, val: f64| {
        if dw_dtype == 1 { unsafe { *dw.data_f32().add(i) = (*dw.data_f32().add(i) as f64 + val) as f32 } }
        else { unsafe { *dw.data_f64().add(i) += val } }
    };
    let add_db = |i: usize, val: f64| {
        if dw_dtype == 1 { unsafe { *db.data_f32().add(i) = (*db.data_f32().add(i) as f64 + val) as f32 } }
        else { unsafe { *db.data_f64().add(i) += val } }
    };

    for ni in 0..n {
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let g_idx = ni * (c_out * h_out * w_out) + co * (h_out * w_out) + oh * w_out + ow;
                    let g_val = read_grad(g_idx);

                    add_db(co, g_val);

                    for ci in 0..c_in {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let ih = oh * stride_h + ky;
                                let iw = ow * stride_w + kx;
                                if ih >= pad_h && iw >= pad_w && ih - pad_h < h && iw - pad_w < w {
                                    let real_ih = ih - pad_h;
                                    let real_iw = iw - pad_w;
                                    let input_idx = ni * (c_in * h * w) + ci * (h * w) + real_ih * w + real_iw;
                                    let weight_idx = co * (c_in * kh * kw) + ci * (kh * kw) + ky * kw + kx;

                                    add_dw(weight_idx, g_val * read_input(input_idx));
                                    add_dx(input_idx, g_val * read_weight(weight_idx));
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (dx_ptr, dw_ptr, db_ptr)
}

/// MaxPool2d backward: scatter gradient to argmax positions
fn maxpool2d_backward(grad_ptr: i64, input_shape: &[i64], argmax: &[usize]) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    let grad_dtype = grad.dtype;
    let out_ptr = create_tensor_with_shape_dtype(input_shape, 0.0, grad_dtype);
    let out = NslTensor::from_ptr(out_ptr);

    let grad_len = grad.len as usize;
    if grad_dtype == 1 {
        for (i, &idx) in argmax.iter().enumerate().take(grad_len) {
            let g_val = unsafe { *grad.data_f32().add(i) };
            unsafe { *out.data_f32().add(idx) += g_val };
        }
    } else {
        for (i, &idx) in argmax.iter().enumerate().take(grad_len) {
            let g_val = unsafe { *grad.data_f64().add(i) };
            unsafe { *out.data_f64().add(idx) += g_val };
        }
    }

    out_ptr
}

/// Run backward pass. `loss_ptr` is the scalar loss tensor. `param_list` is an NslList of
/// parameter tensor pointers. Returns an NslList of gradient tensors (one per param, same order).
#[no_mangle]
pub extern "C" fn nsl_tape_backward(loss_ptr: i64, param_list: i64) -> i64 {
    let params_nsl = NslList::from_ptr(param_list);
    let param_ptrs: Vec<i64> = (0..params_nsl.len as usize)
        .map(|i| unsafe { *params_nsl.data.add(i) })
        .collect();

    // Pause recording during backward (don't tape gradient computation)
    TAPE.with(|t| t.borrow_mut().pause_depth += 1);

    let ops = TAPE.with(|t| t.borrow().ops.clone());

    // grad_map: tensor_ptr -> gradient tensor ptr
    let mut grad_map: HashMap<i64, i64> = HashMap::new();

    // Seed: gradient of loss w.r.t. itself = ones_like(loss)
    let seed = ones_like(loss_ptr);
    grad_map.insert(loss_ptr, seed);

    // Walk tape in reverse, applying chain rule
    for op in ops.iter().rev() {
        match op {
            TapeOp::Add { a, b, out, a_shape, b_shape } => {
                if let Some(&g) = grad_map.get(out) {
                    let ga = reduce_grad_for_broadcast(g, a_shape);
                    let gb = reduce_grad_for_broadcast(g, b_shape);
                    accumulate_grad(&mut grad_map, *a, ga);
                    accumulate_grad(&mut grad_map, *b, gb);
                }
            }
            TapeOp::Sub { a, b, out, a_shape, b_shape } => {
                if let Some(&g) = grad_map.get(out) {
                    let ga = reduce_grad_for_broadcast(g, a_shape);
                    let g_clone = tensor_clone(g);
                    let gb_full = tensor_neg(g_clone);
                    tensor_free(g_clone); // free the clone; tensor_neg created a new tensor
                    let gb = reduce_grad_for_broadcast(gb_full, b_shape);
                    if gb != gb_full { tensor_free(gb_full); }
                    accumulate_grad(&mut grad_map, *a, ga);
                    accumulate_grad(&mut grad_map, *b, gb);
                }
            }
            TapeOp::Mul { a, b, out, saved_a, saved_b, a_shape, b_shape } => {
                if let Some(&g) = grad_map.get(out) {
                    // d/da(a*b) = g * b, d/db(a*b) = g * a
                    let g_clone1 = tensor_clone(g);
                    let g_clone2 = tensor_clone(g);
                    let grad_a_full = tensor_mul(g_clone1, *saved_b);
                    let grad_b_full = tensor_mul(g_clone2, *saved_a);
                    tensor_free(g_clone1);
                    tensor_free(g_clone2);
                    let grad_a = reduce_grad_for_broadcast(grad_a_full, a_shape);
                    let grad_b = reduce_grad_for_broadcast(grad_b_full, b_shape);
                    if grad_a != grad_a_full { tensor_free(grad_a_full); }
                    if grad_b != grad_b_full { tensor_free(grad_b_full); }
                    accumulate_grad(&mut grad_map, *a, grad_a);
                    accumulate_grad(&mut grad_map, *b, grad_b);
                }
            }
            TapeOp::Div { a, b, out, saved_a, saved_b, a_shape, b_shape } => {
                if let Some(&g) = grad_map.get(out) {
                    // Clone g upfront for both grad_a and grad_b computations.
                    // Must happen before any accumulate_grad call, which may free g
                    // if *a == *out (the gradient for *out is g itself).
                    let g_clone1 = tensor_clone(g);
                    let g_clone2 = tensor_clone(g);

                    // d/da(a/b) = g / b
                    let grad_a_full = tensor_div(g_clone1, *saved_b);
                    tensor_free(g_clone1);
                    let grad_a = reduce_grad_for_broadcast(grad_a_full, a_shape);
                    if grad_a != grad_a_full { tensor_free(grad_a_full); }
                    accumulate_grad(&mut grad_map, *a, grad_a);

                    // d/db(a/b) = -g * a / b^2
                    let neg_g = tensor_neg(g_clone2);
                    tensor_free(g_clone2);
                    let neg_ga = tensor_mul(neg_g, *saved_a);
                    let b_sq = tensor_mul(*saved_b, *saved_b);
                    let grad_b_full = tensor_div(neg_ga, b_sq);
                    tensor_free(neg_g);
                    tensor_free(neg_ga);
                    tensor_free(b_sq);
                    let grad_b = reduce_grad_for_broadcast(grad_b_full, b_shape);
                    if grad_b != grad_b_full { tensor_free(grad_b_full); }
                    accumulate_grad(&mut grad_map, *b, grad_b);
                }
            }
            TapeOp::MatMul { a, b, out, saved_a, saved_b } => {
                if let Some(&g) = grad_map.get(out) {
                    // d/dA(A@B) = G @ B^T, d/dB(A@B) = A^T @ G
                    // Use -2,-1 to transpose the last two dims (correct for batched matmul)
                    let b_t = tensor_transpose(*saved_b, -2, -1);
                    let a_t = tensor_transpose(*saved_a, -2, -1);
                    let g_clone1 = tensor_clone(g);
                    let g_clone2 = tensor_clone(g);
                    let grad_a = tensor_matmul(g_clone1, b_t);
                    let grad_b = tensor_matmul(a_t, g_clone2);
                    tensor_free(g_clone1);
                    tensor_free(g_clone2);
                    tensor_free(b_t);
                    tensor_free(a_t);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                    accumulate_grad(&mut grad_map, *b, grad_b);
                }
            }
            TapeOp::Neg { a, out } => {
                if let Some(&g) = grad_map.get(out) {
                    let g_clone = tensor_clone(g);
                    let neg = tensor_neg(g_clone);
                    tensor_free(g_clone);
                    accumulate_grad(&mut grad_map, *a, neg);
                }
            }
            TapeOp::MulScalar { a, scalar, out } => {
                if let Some(&g) = grad_map.get(out) {
                    let g_clone = tensor_clone(g);
                    let scaled = tensor_mul_scalar(g_clone, *scalar);
                    tensor_free(g_clone);
                    accumulate_grad(&mut grad_map, *a, scaled);
                }
            }
            TapeOp::AddScalar { a, out } => {
                if let Some(&g) = grad_map.get(out) {
                    accumulate_grad(&mut grad_map, *a, tensor_clone(g));
                }
            }
            TapeOp::Transpose { a, out, dim0, dim1 } => {
                if let Some(&g) = grad_map.get(out) {
                    // Backward of transpose(dim0, dim1) is transpose(dim0, dim1) again
                    let g_clone = tensor_clone(g);
                    let grad_a = tensor_transpose(g_clone, *dim0, *dim1);
                    tensor_free(g_clone);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::SumReduce { a, out, dim, input_shape, .. } => {
                if let Some(&g) = grad_map.get(out) {
                    if *dim == -1 {
                        // Global reduction: g is scalar, broadcast to input shape
                        let scalar_val = tensor_item(g);
                        let g_dtype = crate::tensor::NslTensor::from_ptr(g).dtype;
                        let ones = ones_from_shape(input_shape, g_dtype);
                        let grad_a = tensor_mul_scalar(ones, scalar_val);
                        tensor_free(ones);
                        accumulate_grad(&mut grad_map, *a, grad_a);
                    } else {
                        // Dimensional reduction: broadcast g along reduced dim
                        let grad_a = broadcast_grad_along_dim(g, input_shape, *dim as usize);
                        accumulate_grad(&mut grad_map, *a, grad_a);
                    }
                }
            }
            TapeOp::MeanReduce { a, out, dim, num_elements, input_shape, .. } => {
                if let Some(&g) = grad_map.get(out) {
                    if *dim == -1 {
                        // Global reduction
                        let scalar_val = tensor_item(g);
                        let g_dtype = crate::tensor::NslTensor::from_ptr(g).dtype;
                        let ones = ones_from_shape(input_shape, g_dtype);
                        let grad_a = tensor_mul_scalar(ones, scalar_val / (*num_elements as f64));
                        tensor_free(ones);
                        accumulate_grad(&mut grad_map, *a, grad_a);
                    } else {
                        // Dimensional reduction: broadcast then scale
                        let expanded = broadcast_grad_along_dim(g, input_shape, *dim as usize);
                        let grad_a = tensor_mul_scalar(expanded, 1.0 / (*num_elements as f64));
                        tensor_free(expanded);
                        accumulate_grad(&mut grad_map, *a, grad_a);
                    }
                }
            }
            TapeOp::ReduceMax { a, out, dim, saved_argmax, input_shape, .. } => {
                if let Some(&g) = grad_map.get(out) {
                    // Scatter grad to argmax positions, zero elsewhere
                    let grad_a = scatter_grad_to_argmax(g, input_shape, *dim as usize, saved_argmax);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Gather { a, out, dim, indices_ptr, input_shape } => {
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = scatter_gather_grad(g, input_shape, *dim as usize, *indices_ptr);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Exp { a, out, saved_out } => {
                // d/da(exp(a)) = exp(a) = saved_out
                if let Some(&g) = grad_map.get(out) {
                    let g_clone = tensor_clone(g);
                    let grad_a = tensor_mul(g_clone, *saved_out);
                    tensor_free(g_clone);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Log { a, out, saved_a } => {
                // d/da(log(a)) = 1/a
                if let Some(&g) = grad_map.get(out) {
                    let g_clone = tensor_clone(g);
                    let grad_a = tensor_div(g_clone, *saved_a);
                    tensor_free(g_clone);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Sqrt { a, out, saved_out } => {
                // d/da(sqrt(a)) = 1 / (2 * sqrt(a)) = 1 / (2 * saved_out)
                if let Some(&g) = grad_map.get(out) {
                    let g_clone = tensor_clone(g);
                    let two_sqrt = tensor_mul_scalar(*saved_out, 2.0);
                    let grad_a = tensor_div(g_clone, two_sqrt);
                    tensor_free(g_clone);
                    tensor_free(two_sqrt);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Abs { a, out, saved_a } => {
                // d/da(abs(a)) = sign(a)
                if let Some(&g) = grad_map.get(out) {
                    let g_clone = tensor_clone(g);
                    let sign = tensor_sign(*saved_a);
                    let grad_a = tensor_mul(g_clone, sign);
                    tensor_free(g_clone);
                    tensor_free(sign);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Clamp { a, out, saved_a, min_val, max_val } => {
                // Gradient passes through where unclamped, zero where clamped
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = crate::tensor::nsl_tensor_clamp_backward(
                        g, *saved_a, *min_val, *max_val,
                    );
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::ReLU { a, out, saved_a } => {
                // d/da(relu(a)) = (a > 0) ? 1 : 0
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = relu_backward(g, *saved_a);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::GELU { a, out, saved_a } => {
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = gelu_backward(g, *saved_a);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::SiLU { a, out, saved_a } => {
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = silu_backward(g, *saved_a);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Sigmoid { a, out, saved_out } => {
                // d/da(sigmoid(a)) = sigmoid(a) * (1 - sigmoid(a)) = out * (1 - out)
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = sigmoid_backward(g, *saved_out);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Tanh { a, out, saved_out } => {
                // d/da(tanh(a)) = 1 - tanh(a)^2 = 1 - out^2
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = tanh_backward(g, *saved_out);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Softmax { a, out, saved_out, dim } => {
                // grad_input_i = output_i * (grad_i - sum(grad * output)) along softmax dim
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = softmax_backward(g, *saved_out, *dim);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Slice { a, out, dim, start, input_shape } => {
                // Backward: create zeros with original input shape, copy grad into [start, start+slice_len)
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = slice_backward(g, input_shape, *dim as usize, *start as usize);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::EmbeddingLookup { weight, indices: _, out, saved_weight, saved_indices } => {
                // Backward: scatter-add output gradient rows into weight gradient
                if let Some(&g) = grad_map.get(out) {
                    let w = crate::tensor::NslTensor::from_ptr(*saved_weight);
                    let idx_t = crate::tensor::NslTensor::from_ptr(*saved_indices);
                    let vocab_size = unsafe { *w.shape.add(0) } as usize;
                    let embed_dim = unsafe { *w.shape.add(1) } as usize;
                    // Use total element count so 2D indices [batch, seq_len] are handled correctly
                    let total_indices = idx_t.len as usize;

                    // Create zero gradient with same shape as weight [vocab_size, embed_dim]
                    let w_shape_list = tensor_shape(*saved_weight);
                    let grad_w = tensor_zeros(w_shape_list);
                    crate::list::nsl_list_free(w_shape_list);

                    let grad_w_t = crate::tensor::NslTensor::from_ptr(grad_w);
                    let g_t = crate::tensor::NslTensor::from_ptr(g);
                    let grad_w_dtype = grad_w_t.dtype;

                    // Scatter-add: for each position i, add grad_out[i, :] to grad_w[idx[i], :]
                    for i in 0..total_indices {
                        let idx = if idx_t.dtype == 1 { (unsafe { *idx_t.data_f32().add(i) }) as usize }
                                  else { (unsafe { *idx_t.data_f64().add(i) }) as usize };
                        if idx < vocab_size {
                            for j in 0..embed_dim {
                                if grad_w_dtype == 1 {
                                    let g_val = unsafe { *g_t.data_f32().add(i * embed_dim + j) };
                                    unsafe { *grad_w_t.data_f32().add(idx * embed_dim + j) += g_val };
                                } else {
                                    let g_val = if g_t.dtype == 1 { unsafe { *g_t.data_f32().add(i * embed_dim + j) as f64 } }
                                                else { unsafe { *g_t.data_f64().add(i * embed_dim + j) } };
                                    unsafe { *grad_w_t.data_f64().add(idx * embed_dim + j) += g_val };
                                }
                            }
                        }
                    }

                    accumulate_grad(&mut grad_map, *weight, grad_w);
                }
            }
            TapeOp::Cat { inputs, out, dim, split_sizes } => {
                // Backward: split the gradient along the cat dim into pieces
                if let Some(&g) = grad_map.get(out) {
                    let grads = cat_backward(g, *dim as usize, split_sizes);
                    for (i, grad_piece) in grads.into_iter().enumerate() {
                        accumulate_grad(&mut grad_map, inputs[i], grad_piece);
                    }
                }
            }
            TapeOp::LayerNorm { input, weight, bias, out, saved_input, saved_mean, saved_inv_std, saved_weight } => {
                if let Some(&g) = grad_map.get(out) {
                    let grad_input = layernorm_backward(g, *saved_input, *saved_mean, *saved_inv_std, *saved_weight);
                    accumulate_grad(&mut grad_map, *input, grad_input.0);
                    accumulate_grad(&mut grad_map, *weight, grad_input.1);
                    accumulate_grad(&mut grad_map, *bias, grad_input.2);
                }
            }
            TapeOp::RMSNorm { input, weight, out, saved_input, saved_rms, saved_weight } => {
                if let Some(&g) = grad_map.get(out) {
                    let grad_result = rmsnorm_backward(g, *saved_input, *saved_rms, *saved_weight);
                    accumulate_grad(&mut grad_map, *input, grad_result.0);
                    accumulate_grad(&mut grad_map, *weight, grad_result.1);
                }
            }
            TapeOp::Dropout { a, out, saved_mask, scale } => {
                // grad_input = grad_output * mask * scale
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = dropout_backward(g, *saved_mask, *scale);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Conv2d { input, weight, bias, out, saved_input, saved_weight, stride_h, stride_w, pad_h, pad_w } => {
                if let Some(&g) = grad_map.get(out) {
                    let (grad_input, grad_weight, grad_bias) = conv2d_backward(
                        g, *saved_input, *saved_weight, *stride_h, *stride_w, *pad_h, *pad_w,
                    );
                    accumulate_grad(&mut grad_map, *input, grad_input);
                    accumulate_grad(&mut grad_map, *weight, grad_weight);
                    if *bias != 0 {
                        accumulate_grad(&mut grad_map, *bias, grad_bias);
                    } else {
                        tensor_free(grad_bias);
                    }
                }
            }
            TapeOp::MaxPool2d { a, out, saved_argmax, input_shape } => {
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = maxpool2d_backward(g, input_shape, saved_argmax);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Unsqueeze { input, out, input_shape } => {
                // Unsqueeze backward: reshape gradient back to original shape
                if let Some(&g) = grad_map.get(out) {
                    let grad_input = reshape_to_shape(g, input_shape);
                    accumulate_grad(&mut grad_map, *input, grad_input);
                }
            }
            TapeOp::Expand { input, out, original_shape } => {
                // Expand backward: sum along each broadcast dimension, then reshape to original shape
                if let Some(&g) = grad_map.get(out) {
                    let grad_t = crate::tensor::NslTensor::from_ptr(g);
                    let grad_shape: Vec<i64> = unsafe {
                        std::slice::from_raw_parts(grad_t.shape, grad_t.ndim as usize)
                    }.to_vec();

                    // Right-align original shape with grad shape
                    let offset = grad_shape.len() - original_shape.len();
                    let mut result = tensor_clone(g);

                    // Sum along broadcast dims (iterate in reverse to keep dim indices stable)
                    for d in (0..grad_shape.len()).rev() {
                        let orig_d = if d >= offset { original_shape[d - offset] } else { 1 };
                        if orig_d == 1 && grad_shape[d] > 1 {
                            let old = result;
                            result = nsl_tensor_sum_dim(result, d as i64, 1); // keepdim=1
                            tensor_free(old);
                        }
                    }
                    // Reshape to original shape
                    let old = result;
                    result = reshape_to_shape(result, original_shape);
                    if old != result { tensor_free(old); }

                    accumulate_grad(&mut grad_map, *input, result);
                }
            }
            TapeOp::Stack { inputs, out, dim } => {
                // Stack backward: select gradient slice for each input
                if let Some(&g) = grad_map.get(out) {
                    for (i, input_id) in inputs.iter().enumerate() {
                        let grad_piece = nsl_tensor_select(g, *dim, i as i64);
                        accumulate_grad(&mut grad_map, *input_id, grad_piece);
                    }
                }
            }
            TapeOp::BiasAdd { tensor, bias, out } => {
                // grad_tensor = upstream grad (same shape as tensor [M, N])
                // grad_bias = sum(upstream grad, dim=0) -> [N]
                if let Some(&g) = grad_map.get(out) {
                    // grad for tensor is just the upstream gradient
                    let grad_tensor = tensor_clone(g);
                    accumulate_grad(&mut grad_map, *tensor, grad_tensor);

                    // grad for bias = sum over batch dim (rows)
                    let g_t = crate::tensor::NslTensor::from_ptr(g);
                    if g_t.ndim < 2 {
                        eprintln!("nsl: BiasAdd backward expects 2D+ gradient, got {}D", g_t.ndim);
                        std::process::abort();
                    }
                    let rows = unsafe { *g_t.shape.add(0) } as usize;
                    let cols = unsafe { *g_t.shape.add(1) } as usize;

                    let g_dtype = g_t.dtype;
                    let grad_bias = crate::cpu::create_tensor_with_shape_rs_dtype(&[cols as i64], g_dtype);
                    let grad_bias_t = crate::tensor::NslTensor::from_ptr(grad_bias);
                    for i in 0..rows {
                        for j in 0..cols {
                            if g_dtype == 1 {
                                let gv = unsafe { *g_t.data_f32().add(i * cols + j) };
                                unsafe { *grad_bias_t.data_f32().add(j) += gv };
                            } else {
                                let gv = unsafe { *g_t.data_f64().add(i * cols + j) };
                                unsafe { *grad_bias_t.data_f64().add(j) += gv };
                            }
                        }
                    }

                    accumulate_grad(&mut grad_map, *bias, grad_bias);
                }
            }
        }
    }

    // Resume recording
    TAPE.with(|t| t.borrow_mut().pause_depth -= 1);

    // Build result list: one gradient per param (in same order as param_list)
    let result_list = crate::list::nsl_list_new();
    for ptr in &param_ptrs {
        if let Some(grad) = grad_map.remove(ptr) {
            crate::list::nsl_list_push(result_list, grad);
        } else {
            // No gradient computed for this param — return zeros_like
            let shape_list = tensor_shape(*ptr);
            let zeros = tensor_zeros(shape_list);
            crate::list::nsl_list_free(shape_list);
            crate::list::nsl_list_push(result_list, zeros);
        }
    }

    // Free all remaining intermediate gradient tensors (seed, activation grads, etc.)
    for (_, grad_tensor) in grad_map {
        tensor_free(grad_tensor);
    }

    result_list
}
