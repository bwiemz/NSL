use std::sync::atomic::AtomicI64;
use std::collections::HashMap;
use std::ffi::c_void;

use crate::memory::checked_alloc;
use crate::tensor::{
    nsl_tensor_add as tensor_add,
    nsl_tensor_clone as tensor_clone,
    nsl_tensor_free as tensor_free,
    nsl_tensor_ones_like as nsl_tensor_ones_like,
    nsl_tensor_sum_dim,
    NslTensor,
};

use super::TAPE;

/// Create a tensor of ones with the same shape and device as the given tensor.
pub(crate) fn ones_like(tensor_ptr: i64) -> i64 {
    nsl_tensor_ones_like(tensor_ptr)
}

/// Accumulate gradient: grads[key] += grad_tensor.
/// If key doesn't exist yet, set it directly. If it does, add and free the old/input tensors.
pub(crate) fn accumulate_grad(grads: &mut HashMap<i64, i64>, key: i64, grad_tensor: i64) {
    if let Some(existing) = grads.get(&key) {
        let old = *existing;
        // Pause recording to avoid taping gradient computation ops
        TAPE.with(|t| t.borrow_mut().pause_depth += 1);
        let summed = tensor_add(old, grad_tensor);
        TAPE.with(|t| t.borrow_mut().pause_depth -= 1);
        tensor_free(old);
        tensor_free(grad_tensor);
        grads.insert(key, summed);
    } else {
        grads.insert(key, grad_tensor);
    }
}

/// Create a tensor with the given shape, filled with zeros (f64, dtype=0).
#[allow(dead_code)]
pub(crate) fn create_tensor_with_shape(shape: &[i64], fill: f64) -> i64 {
    create_tensor_with_shape_dtype(shape, fill, 0)
}

/// Create a tensor with the given shape, filled with `fill`, and a specified dtype.
pub(crate) fn create_tensor_with_shape_dtype(shape: &[i64], fill: f64, dtype: u16) -> i64 {
    use crate::memory::checked_alloc_zeroed;

    let ndim = shape.len() as i64;
    let mut total: i64 = 1;
    for &s in shape {
        total *= s;
    }

    let shape_ptr = crate::memory::checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
    for (i, &s) in shape.iter().enumerate() {
        unsafe { *shape_ptr.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape_ptr, ndim);

    let elem_size = if dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let data_size = (total as usize) * elem_size;
    let data_raw: *mut c_void = if fill == 0.0 {
        checked_alloc_zeroed(data_size) as *mut c_void
    } else if dtype == 1 {
        let data = crate::memory::checked_alloc(data_size) as *mut f32;
        for i in 0..total as usize {
            unsafe { *data.add(i) = fill as f32 };
        }
        data as *mut c_void
    } else {
        let data = crate::memory::checked_alloc(data_size) as *mut f64;
        for i in 0..total as usize {
            unsafe { *data.add(i) = fill };
        }
        data as *mut c_void
    };

    let tensor = Box::new(NslTensor {
        data: data_raw,
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

/// Like `create_tensor_with_shape_dtype` but allocates data in unified CUDA memory when device > 0.
/// This allows CPU code to populate the tensor while the result is accessible on GPU.
pub(crate) fn create_tensor_with_shape_dtype_device(shape: &[i64], fill: f64, dtype: u16, device: u8) -> i64 {
    use crate::memory::checked_alloc_zeroed;

    if device == 0 {
        return create_tensor_with_shape_dtype(shape, fill, dtype);
    }

    let ndim = shape.len() as i64;
    let mut total: i64 = 1;
    for &s in shape {
        total *= s;
    }

    let shape_ptr = crate::memory::checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
    for (i, &s) in shape.iter().enumerate() {
        unsafe { *shape_ptr.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape_ptr, ndim);

    let elem_size = if dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let data_size = (total as usize) * elem_size;

    let data_raw: *mut c_void = {
        #[cfg(feature = "cuda")]
        {
            // Allocate in unified memory (zero-initialized via CUDA managed memory semantics).
            // cuMemAllocManaged does not guarantee zero-init, so we zero it explicitly.
            let ptr = crate::cuda::inner::alloc_managed(data_size);
            unsafe { std::ptr::write_bytes(ptr as *mut u8, 0, data_size) };
            if fill != 0.0 {
                if dtype == 1 {
                    let fptr = ptr as *mut f32;
                    for i in 0..total as usize {
                        unsafe { *fptr.add(i) = fill as f32 };
                    }
                } else {
                    let fptr = ptr as *mut f64;
                    for i in 0..total as usize {
                        unsafe { *fptr.add(i) = fill };
                    }
                }
            }
            ptr
        }
        #[cfg(not(feature = "cuda"))]
        {
            if fill == 0.0 {
                checked_alloc_zeroed(data_size) as *mut c_void
            } else if dtype == 1 {
                let data = crate::memory::checked_alloc(data_size) as *mut f32;
                for i in 0..total as usize {
                    unsafe { *data.add(i) = fill as f32 };
                }
                data as *mut c_void
            } else {
                let data = crate::memory::checked_alloc(data_size) as *mut f64;
                for i in 0..total as usize {
                    unsafe { *data.add(i) = fill };
                }
                data as *mut c_void
            }
        }
    };

    let tensor = Box::new(NslTensor {
        data: data_raw,
        shape: shape_ptr,
        strides,
        ndim,
        len: total,
        refcount: AtomicI64::new(1),
        device,
        dtype,
        owns_data: 1, data_owner: 0,
    });
    Box::into_raw(tensor) as i64
}

/// Reshape a tensor to the given shape by deep-copying its data with new shape metadata.
/// The element count must match (panics otherwise).
pub(crate) fn reshape_to_shape(tensor_ptr: i64, shape: &[i64]) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = shape.len() as i64;
    let total: i64 = shape.iter().product();
    assert_eq!(total, tensor.len, "reshape_to_shape: size mismatch {} vs {}", total, tensor.len);

    let shape_ptr = crate::memory::checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
    for (i, &s) in shape.iter().enumerate() {
        unsafe { *shape_ptr.add(i) = s; }
    }
    let strides = NslTensor::compute_strides(shape_ptr, ndim);

    let data_size = (total as usize) * tensor.element_size();
    let data = crate::memory::checked_alloc(data_size);
    unsafe { std::ptr::copy_nonoverlapping(tensor.data as *const u8, data, data_size); }

    let new_tensor = Box::new(NslTensor {
        data: data as *mut std::ffi::c_void,
        shape: shape_ptr,
        strides,
        ndim,
        len: total,
        refcount: AtomicI64::new(1),
        device: tensor.device,
        dtype: tensor.dtype,
        owns_data: 1, data_owner: 0,
    });
    Box::into_raw(new_tensor) as i64
}

/// Reduce a gradient to match the original input shape after broadcasting.
/// If grad has shape [10, 256] but original input was [10, 1], sum along dim 1 with keepdim.
/// If grad has ndim 2 but original was ndim 1, reduce and squeeze appropriately.
pub(crate) fn reduce_grad_for_broadcast(grad_ptr: i64, orig_shape: &[i64]) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    let grad_ndim = grad.ndim as usize;
    let orig_ndim = orig_shape.len();

    // Build the right-aligned original shape padded with 1s
    let mut orig_padded = vec![1i64; grad_ndim];
    for i in 0..orig_ndim {
        orig_padded[grad_ndim - orig_ndim + i] = orig_shape[i];
    }

    // Check which dims were broadcast (orig was 1 but grad is > 1)
    let grad_shape: Vec<i64> = (0..grad_ndim)
        .map(|i| unsafe { *grad.shape.add(i) })
        .collect();

    // If shapes already match, return clone
    if grad_shape == orig_shape {
        return tensor_clone(grad_ptr);
    }

    let mut result = tensor_clone(grad_ptr);

    // Sum along dimensions that were broadcast (where orig was 1 but grad > 1)
    // We also need to handle leading dims that don't exist in orig (padded with 1)
    for d in 0..grad_ndim {
        if orig_padded[d] == 1 && grad_shape[d] > 1 {
            let old = result;
            // Sum along dim d, keepdim=true to maintain ndim
            result = nsl_tensor_sum_dim(result, d as i64, 1);
            if old != grad_ptr {
                tensor_free(old);
            }
        }
    }

    // If orig had fewer dims, squeeze the leading dims
    if orig_ndim < grad_ndim {
        let res = NslTensor::from_ptr(result);
        // Reshape to orig_shape
        let new_shape = checked_alloc(std::mem::size_of_val(orig_shape)) as *mut i64;
        for (i, &s) in orig_shape.iter().enumerate().take(orig_ndim) {
            unsafe { *new_shape.add(i) = s };
        }
        let new_strides = NslTensor::compute_strides(new_shape, orig_ndim as i64);
        let mut total: i64 = 1;
        for &s in orig_shape {
            total *= s;
        }
        // Copy data (byte-level copy, preserves dtype)
        let elem_size = res.element_size();
        let new_data_raw = checked_alloc((total as usize) * elem_size);
        unsafe { std::ptr::copy_nonoverlapping(res.data as *const u8, new_data_raw, (total as usize) * elem_size) };
        let out = Box::new(NslTensor {
            data: new_data_raw as *mut c_void,
            shape: new_shape,
            strides: new_strides,
            ndim: orig_ndim as i64,
            len: total,
            refcount: AtomicI64::new(1),
        device: 0,
        dtype: res.dtype,
        owns_data: 1, data_owner: 0,
    });
        let out_ptr = Box::into_raw(out) as i64;
        if result != grad_ptr {
            tensor_free(result);
        }
        return out_ptr;
    }

    result
}

/// Broadcast a gradient tensor along a reduced dimension back to input_shape.
/// grad has the reduced shape; we need to expand it along `dim` to match input_shape.
pub(crate) fn broadcast_grad_along_dim(grad_ptr: i64, input_shape: &[i64], dim: usize) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    let ndim = input_shape.len();
    let grad_dtype = grad.dtype;

    // Create output with input_shape, matching grad dtype
    let out_ptr = create_tensor_with_shape_dtype(input_shape, 0.0, grad_dtype);
    let out = NslTensor::from_ptr(out_ptr);

    // Compute contiguous strides from shapes (safe for non-contiguous tensors)
    let out_strides: Vec<usize> = {
        let mut s = vec![1usize; ndim];
        for d in (0..ndim.saturating_sub(1)).rev() {
            s[d] = s[d + 1] * input_shape[d + 1] as usize;
        }
        s
    };

    let grad_ndim = grad.ndim as usize;
    let grad_shape: Vec<usize> = (0..grad_ndim)
        .map(|i| unsafe { *grad.shape.add(i) } as usize)
        .collect();
    let grad_strides: Vec<usize> = {
        let mut s = vec![1usize; grad_ndim];
        for d in (0..grad_ndim.saturating_sub(1)).rev() {
            s[d] = s[d + 1] * grad_shape[d + 1];
        }
        s
    };

    let total = out.len as usize;
    for flat_idx in 0..total {
        let mut remaining = flat_idx;
        let mut indices = vec![0usize; ndim];
        for d in 0..ndim {
            indices[d] = remaining / out_strides[d];
            remaining %= out_strides[d];
        }

        let mut grad_idx = 0usize;
        let mut gi = 0usize;
        for (d, &idx) in indices.iter().enumerate().take(ndim) {
            if d == dim {
                continue;
            }
            grad_idx += idx * grad_strides[gi];
            gi += 1;
        }

        if grad_dtype == 1 {
            unsafe { *out.data_f32().add(flat_idx) = *grad.data_f32().add(grad_idx); }
        } else {
            unsafe { *out.data_f64().add(flat_idx) = *grad.data_f64().add(grad_idx); }
        }
    }

    out_ptr
}

/// Scatter gradient to argmax positions for ReduceMax backward.
pub(crate) fn scatter_grad_to_argmax(
    grad_ptr: i64,
    input_shape: &[i64],
    dim: usize,
    argmax: &[usize],
) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    let ndim = input_shape.len();
    let grad_dtype = grad.dtype;
    // Create zero output with input_shape, matching grad dtype
    let out_ptr = create_tensor_with_shape_dtype(input_shape, 0.0, grad_dtype);
    let out = NslTensor::from_ptr(out_ptr);

    // Compute contiguous strides from shapes (safe for non-contiguous tensors)
    let out_strides: Vec<usize> = {
        let mut s = vec![1usize; ndim];
        for d in (0..ndim.saturating_sub(1)).rev() {
            s[d] = s[d + 1] * input_shape[d + 1] as usize;
        }
        s
    };

    let grad_ndim = grad.ndim as usize;
    let grad_shape: Vec<usize> = (0..grad_ndim)
        .map(|i| unsafe { *grad.shape.add(i) } as usize)
        .collect();
    let grad_strides: Vec<usize> = {
        let mut s = vec![1usize; grad_ndim];
        for d in (0..grad_ndim.saturating_sub(1)).rev() {
            s[d] = s[d + 1] * grad_shape[d + 1];
        }
        s
    };

    let grad_total = grad.len as usize;
    for (grad_flat, &max_idx) in argmax.iter().enumerate().take(grad_total) {
        let mut remaining = grad_flat;
        let mut grad_indices = vec![0usize; grad_ndim];
        for d in 0..grad_ndim {
            grad_indices[d] = remaining / grad_strides[d];
            remaining %= grad_strides[d];
        }

        let mut out_offset = 0usize;
        let mut gi = 0usize;
        for (d, &os) in out_strides.iter().enumerate().take(ndim) {
            if d == dim {
                out_offset += max_idx * os;
            } else {
                out_offset += grad_indices[gi] * os;
                gi += 1;
            }
        }

        if grad_dtype == 1 {
            let g_val = unsafe { *grad.data_f32().add(grad_flat) };
            unsafe { *out.data_f32().add(out_offset) += g_val };
        } else {
            let g_val = unsafe { *grad.data_f64().add(grad_flat) };
            unsafe { *out.data_f64().add(out_offset) += g_val };
        }
    }

    out_ptr
}

/// Scatter gather gradient back to input shape.
pub(crate) fn scatter_gather_grad(
    grad_ptr: i64,
    input_shape: &[i64],
    dim: usize,
    indices_ptr: i64,
) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    let indices = NslTensor::from_ptr(indices_ptr);
    let grad_dtype = grad.dtype;

    // Create zero gradient with input_shape, matching grad dtype
    let out_ptr = create_tensor_with_shape_dtype(input_shape, 0.0, grad_dtype);
    let out = NslTensor::from_ptr(out_ptr);

    let out_strides: Vec<usize> = (0..input_shape.len())
        .map(|i| unsafe { *out.strides.add(i) } as usize)
        .collect();
    // For cross_entropy: input=[batch, classes], dim=1, indices=[batch]
    // grad=[batch], output[b, indices[b]] += grad[b]
    let batch = indices.len as usize;
    for b in 0..batch {
        let idx = if indices.dtype == 1 { (unsafe { *indices.data_f32().add(b) }) as usize }
                  else { (unsafe { *indices.data_f64().add(b) }) as usize };
        let mut out_offset = 0usize;
        if input_shape.len() == 2 && dim == 1 {
            out_offset = b * out_strides[0] + idx * out_strides[1];
        } else if dim == 0 {
            out_offset = idx * out_strides[0] + b * out_strides[1];
        }
        if grad_dtype == 1 {
            let g_val = unsafe { *grad.data_f32().add(b) };
            unsafe { *out.data_f32().add(out_offset) += g_val };
        } else {
            let g_val = unsafe { *grad.data_f64().add(b) };
            unsafe { *out.data_f64().add(out_offset) += g_val };
        }
    }

    out_ptr
}
