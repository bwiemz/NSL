//! Tensor shape operations: reshape, transpose, unsqueeze, select, stack, expand, slice, cat,
//! causal_mask, squeeze.

use std::ffi::c_void;
use std::sync::atomic::Ordering;

use crate::autodiff;
use crate::list::NslList;
use crate::memory::checked_alloc;

use super::{NslTensor, nsl_tensor_free};

// === Shape query operations ===

#[no_mangle]
pub extern "C" fn nsl_tensor_shape(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let list = crate::list::nsl_list_new();
    for i in 0..tensor.ndim as usize {
        crate::list::nsl_list_push(list, unsafe { *tensor.shape.add(i) });
    }
    list
}

/// Return the size of a specific dimension of a tensor.
#[no_mangle]
pub extern "C" fn nsl_tensor_shape_dim(tensor_ptr: i64, dim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;
    let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };
    if d >= ndim {
        eprintln!("nsl: shape_dim dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }
    unsafe { *tensor.shape.add(d) }
}

/// Assert that tensor dimension `dim_index` equals `expected_value`.
#[no_mangle]
pub extern "C" fn nsl_tensor_assert_dim(tensor_ptr: i64, dim_index: i64, expected_value: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;
    let dim_idx = if dim_index < 0 {
        (ndim as i64 + dim_index) as usize
    } else {
        dim_index as usize
    };
    if dim_idx >= ndim {
        eprintln!(
            "nsl: assert_dim: dimension index {} out of range for rank-{} tensor",
            dim_index, ndim
        );
        std::process::abort();
    }
    let actual = unsafe { *tensor.shape.add(dim_idx) };
    if expected_value == -1 {
        return actual;
    }
    if actual != expected_value {
        eprintln!(
            "nsl: dimension mismatch: expected dim[{}] = {}, got {}",
            dim_index, expected_value, actual
        );
        std::process::abort();
    }
    actual
}

/// Assert that a tensor dimension does not exceed an upper bound.
#[no_mangle]
pub extern "C" fn nsl_tensor_assert_dim_bound(tensor_ptr: i64, dim_index: i64, upper_bound: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;
    let dim_idx = if dim_index < 0 {
        (ndim as i64 + dim_index) as usize
    } else {
        dim_index as usize
    };
    if dim_idx >= ndim {
        eprintln!(
            "nsl: assert_dim_bound: dimension index {} out of range for rank-{} tensor",
            dim_index, ndim
        );
        std::process::abort();
    }
    let actual = unsafe { *tensor.shape.add(dim_idx) };
    if actual > upper_bound {
        eprintln!(
            "nsl: dimension bound exceeded: dim[{}] = {} exceeds upper bound {}",
            dim_index, actual, upper_bound
        );
        std::process::abort();
    }
    actual
}

#[no_mangle]
pub extern "C" fn nsl_tensor_ndim(tensor_ptr: i64) -> i64 {
    NslTensor::from_ptr(tensor_ptr).ndim
}

#[no_mangle]
pub extern "C" fn nsl_tensor_get_dtype(tensor_ptr: i64) -> i64 {
    NslTensor::from_ptr(tensor_ptr).dtype as i64
}

#[no_mangle]
pub extern "C" fn nsl_tensor_reshape(tensor_ptr: i64, new_shape_list: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let new_shape_nsl = NslList::from_ptr(new_shape_list);
    let new_ndim = new_shape_nsl.len;

    // Extract shape values
    let mut new_shape_vec = Vec::with_capacity(new_ndim as usize);
    let mut new_len: i64 = 1;
    for i in 0..new_ndim as usize {
        let dim = unsafe { *new_shape_nsl.data.add(i) };
        new_shape_vec.push(dim);
        new_len *= dim;
    }

    if new_len != tensor.len {
        eprintln!(
            "nsl: cannot reshape tensor of size {} into shape of size {}",
            tensor.len, new_len
        );
        std::process::abort();
    }

    // Compute row-major strides for new shape
    let new_strides_vec = {
        let mut s = vec![1i64; new_ndim as usize];
        if new_ndim > 0 {
            for i in (0..(new_ndim as usize) - 1).rev() {
                s[i] = s[i + 1] * new_shape_vec[i + 1];
            }
        }
        s
    };

    // Check contiguity directly to avoid refcount undo-redo pattern.
    let is_contig = tensor.is_contiguous();

    let result = if is_contig {
        // Contiguous — return a zero-copy view directly
        NslTensor::new_view_i64(tensor_ptr, &new_shape_vec, &new_strides_vec, new_ndim, new_len)
    } else {
        // Non-contiguous — materialize to contiguous first, then create view of result
        let contig_ptr = nsl_tensor_contiguous(tensor_ptr);
        let view = NslTensor::new_view_i64(contig_ptr, &new_shape_vec, &new_strides_vec, new_ndim, new_len);
        // Free the intermediate contiguous tensor (view holds the ref via data_owner)
        nsl_tensor_free(contig_ptr);
        view
    };

    // Tracing (unchanged)
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Reshape, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }

    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_transpose(tensor_ptr: i64, dim0: i64, dim1: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);

    let d0 = if dim0 < 0 { dim0 + tensor.ndim } else { dim0 };
    let d1 = if dim1 < 0 { dim1 + tensor.ndim } else { dim1 };

    if d0 < 0 || d0 >= tensor.ndim || d1 < 0 || d1 >= tensor.ndim {
        eprintln!(
            "nsl: transpose dimensions out of range ({}, {} for ndim {})",
            dim0, dim1, tensor.ndim
        );
        std::process::abort();
    }
    let dim0 = d0;
    let dim1 = d1;

    let ndim = tensor.ndim as usize;

    // Build new shape and strides by swapping dim0/dim1
    let mut new_shape_vec = Vec::with_capacity(ndim);
    let mut new_strides_vec = Vec::with_capacity(ndim);
    for i in 0..ndim {
        unsafe {
            new_shape_vec.push(*tensor.shape.add(i));
            new_strides_vec.push(*tensor.strides.add(i));
        }
    }
    new_shape_vec.swap(dim0 as usize, dim1 as usize);
    new_strides_vec.swap(dim0 as usize, dim1 as usize);

    // Create zero-copy view
    let out_ptr = NslTensor::new_view_i64(
        tensor_ptr,
        &new_shape_vec,
        &new_strides_vec,
        tensor.ndim,
        tensor.len,
    );

    // Record on tape for autodiff
    if crate::autodiff::is_recording() {
        crate::autodiff::maybe_record(crate::autodiff::TapeOp::Transpose {
            a: tensor_ptr,
            out: out_ptr,
            dim0,
            dim1,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(out_ptr);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Transpose, vec![tensor_ptr], out_ptr, shape, rt.dtype, vec![]);
    }

    out_ptr
}

/// Insert a dimension of size 1 at position `dim`.
#[no_mangle]
pub extern "C" fn nsl_tensor_unsqueeze(tensor_ptr: i64, dim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;
    let new_ndim = ndim + 1;

    let insert_pos = if dim < 0 {
        (dim + new_ndim as i64) as usize
    } else {
        dim as usize
    };

    assert!(insert_pos <= ndim, "unsqueeze dim {} out of range for ndim {}", dim, ndim);

    let mut new_shape = Vec::with_capacity(new_ndim);
    let mut new_strides = Vec::with_capacity(new_ndim);

    let mut j = 0usize;
    for i in 0..new_ndim {
        if i == insert_pos {
            new_shape.push(1);
            // Stride for size-1 dim doesn't matter for indexing (only 1 element),
            // but for contiguity checks it should be product of dims to its right
            let stride_val = if j < ndim {
                unsafe { *tensor.strides.add(j) * *tensor.shape.add(j) }
            } else {
                1
            };
            new_strides.push(stride_val);
        } else {
            unsafe {
                new_shape.push(*tensor.shape.add(j));
                new_strides.push(*tensor.strides.add(j));
            }
            j += 1;
        }
    }

    let out_ptr = NslTensor::new_view_i64(
        tensor_ptr,
        &new_shape,
        &new_strides,
        new_ndim as i64,
        tensor.len,
    );

    // Autodiff tape (preserve existing behavior)
    if crate::autodiff::is_recording() {
        let input_shape = unsafe {
            std::slice::from_raw_parts(tensor.shape, ndim)
        }.to_vec();
        autodiff::maybe_record(autodiff::TapeOp::Unsqueeze {
            input: tensor_ptr,
            out: out_ptr,
            input_shape,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(out_ptr);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Unsqueeze, vec![tensor_ptr], out_ptr, shape, rt.dtype, vec![]);
    }

    out_ptr
}

/// Extract a hyperplane at `index` along `dim`, removing that dimension.
#[no_mangle]
pub extern "C" fn nsl_tensor_select(tensor_ptr: i64, dim: i64, index: i64) -> i64 {
    // GPU redirect: transfer to CPU, operate, transfer result back.
    let t = NslTensor::from_ptr(tensor_ptr);
    if t.device > 0 {
        let cpu_t = super::nsl_tensor_to_device(tensor_ptr, 0);
        let result = nsl_tensor_select(cpu_t, dim, index);
        let gpu_result = super::nsl_tensor_to_device(result, t.device as i64);
        super::nsl_tensor_free(cpu_t);
        super::nsl_tensor_free(result);
        return gpu_result;
    }

    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;

    // Normalize dim
    let d = if dim < 0 { (tensor.ndim + dim) as usize } else { dim as usize };
    if d >= ndim {
        eprintln!("nsl: select dim {} out of range for ndim {}", dim, ndim);
        std::process::abort();
    }

    let dim_size = unsafe { *tensor.shape.add(d) };

    // Normalize index
    let idx = if index < 0 { index + dim_size } else { index };
    if idx < 0 || idx >= dim_size {
        eprintln!(
            "nsl: select index {} out of range for dim {} size {}",
            index, dim, dim_size
        );
        std::process::abort();
    }
    let idx = idx as usize;

    // Build output shape: old shape minus the selected dimension
    let out_ndim = (ndim - 1) as i64;
    let out_shape = checked_alloc((ndim - 1) * std::mem::size_of::<i64>()) as *mut i64;
    let mut out_axis = 0;
    for i in 0..ndim {
        if i != d {
            unsafe { *out_shape.add(out_axis) = *tensor.shape.add(i) };
            out_axis += 1;
        }
    }

    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);
    let out_len = NslTensor::total_elements(out_shape, out_ndim);

    let in_strides: Vec<i64> = (0..ndim).map(|i| unsafe { *tensor.strides.add(i) }).collect();
    let base_offset = idx * in_strides[d] as usize;

    let out_stride_vec: Vec<i64> = (0..ndim - 1).map(|i| unsafe { *out_strides.add(i) }).collect();
    let axis_map: Vec<usize> = (0..ndim).filter(|&i| i != d).collect();

    let data: *mut c_void = if tensor.dtype == 1 {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for flat in 0..out_len as usize {
            let mut remaining = flat;
            let mut in_offset = base_offset;
            for (oa, &ia) in axis_map.iter().enumerate() {
                let i = remaining / out_stride_vec[oa] as usize;
                remaining %= out_stride_vec[oa] as usize;
                in_offset += i * in_strides[ia] as usize;
            }
            unsafe { *buf.add(flat) = *tensor.data_f32().add(in_offset) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for flat in 0..out_len as usize {
            let mut remaining = flat;
            let mut in_offset = base_offset;
            for (oa, &ia) in axis_map.iter().enumerate() {
                let i = remaining / out_stride_vec[oa] as usize;
                remaining %= out_stride_vec[oa] as usize;
                in_offset += i * in_strides[ia] as usize;
            }
            unsafe { *buf.add(flat) = *tensor.data_f64().add(in_offset) };
        }
        buf as *mut c_void
    };

    let out = Box::new(NslTensor::new(
        data,
        out_shape,
        out_strides,
        out_ndim,
        out_len,
        tensor.device,
        tensor.dtype,
        1,
        0,
    ));
    // NO tape recording -- select is used internally for stack backward
    NslTensor::publish(out)
}

/// Stack a list of same-shape tensors along a NEW dimension at position `dim`.
#[no_mangle]
pub extern "C" fn nsl_tensor_stack(list_ptr: i64, dim: i64) -> i64 {
    let list = NslList::from_ptr(list_ptr);
    let num_tensors = list.len as usize;
    assert!(num_tensors > 0, "nsl_tensor_stack: empty tensor list");

    // GPU redirect: if any tensor is on GPU, move all to CPU, stack, move result to GPU.
    let source_device = {
        let first = NslTensor::from_ptr(unsafe { *list.data });
        first.device
    };
    if source_device > 0 {
        // Transfer all inputs to CPU
        let cpu_ptrs: Vec<i64> = (0..num_tensors)
            .map(|i| super::nsl_tensor_to_device(unsafe { *list.data.add(i) }, 0))
            .collect();
        // Build a temporary NslList on the stack via the runtime list API
        let tmp_list = crate::list::nsl_list_new();
        for &cp in &cpu_ptrs {
            crate::list::nsl_list_push(tmp_list, cp);
        }
        let cpu_result = nsl_tensor_stack(tmp_list, dim);
        let gpu_result = super::nsl_tensor_to_device(cpu_result, source_device as i64);
        // Free temporaries
        super::nsl_tensor_free(cpu_result);
        for &cp in &cpu_ptrs {
            super::nsl_tensor_free(cp);
        }
        // Free temp list (NslList is just a heap allocation)
        crate::list::nsl_list_free(tmp_list);
        return gpu_result;
    }

    // Ensure all input tensors are contiguous for correct flat indexing
    let contiguous_ptrs: Vec<i64> = (0..num_tensors)
        .map(|i| nsl_tensor_contiguous(unsafe { *list.data.add(i) }))
        .collect();

    let first = NslTensor::from_ptr(contiguous_ptrs[0]);
    let in_ndim = first.ndim as usize;
    let out_ndim = (in_ndim + 1) as i64;

    let insert_pos = if dim < 0 {
        (dim + out_ndim) as usize
    } else {
        dim as usize
    };
    assert!(
        insert_pos <= in_ndim,
        "nsl_tensor_stack: dim {} out of range for ndim {}",
        dim, in_ndim
    );

    // Validate all tensors have the same shape
    for &cp in &contiguous_ptrs {
        let t = NslTensor::from_ptr(cp);
        assert_eq!(t.ndim as usize, in_ndim, "nsl_tensor_stack: ndim mismatch");
        for axis in 0..in_ndim {
            let s1 = unsafe { *first.shape.add(axis) };
            let s2 = unsafe { *t.shape.add(axis) };
            assert_eq!(s1, s2, "nsl_tensor_stack: shape mismatch at axis {}: {} vs {}", axis, s1, s2);
        }
    }

    // Build output shape: insert num_tensors at insert_pos
    let out_shape = checked_alloc((out_ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..insert_pos {
        unsafe { *out_shape.add(i) = *first.shape.add(i) };
    }
    unsafe { *out_shape.add(insert_pos) = num_tensors as i64 };
    for i in insert_pos..in_ndim {
        unsafe { *out_shape.add(i + 1) = *first.shape.add(i) };
    }

    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);
    let out_len = NslTensor::total_elements(out_shape, out_ndim);
    let per_tensor = first.len as usize;

    let data: *mut c_void = if first.dtype == 1 {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        let out_stride_vec: Vec<i64> = (0..out_ndim as usize)
            .map(|i| unsafe { *out_strides.add(i) })
            .collect();
        for (t_idx, &cp) in contiguous_ptrs.iter().enumerate() {
            let t = NslTensor::from_ptr(cp);
            let t_strides: Vec<i64> = (0..in_ndim).map(|i| unsafe { *t.strides.add(i) }).collect();
            for flat in 0..per_tensor {
                let mut remaining = flat;
                let mut in_multi: Vec<usize> = vec![0; in_ndim];
                for axis in 0..in_ndim {
                    in_multi[axis] = remaining / t_strides[axis] as usize;
                    remaining %= t_strides[axis] as usize;
                }
                let mut out_offset = t_idx * out_stride_vec[insert_pos] as usize;
                for (ia, &iv) in in_multi.iter().enumerate() {
                    let oa = if ia < insert_pos { ia } else { ia + 1 };
                    out_offset += iv * out_stride_vec[oa] as usize;
                }
                unsafe { *buf.add(out_offset) = *t.data_f32().add(flat) };
            }
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        let out_stride_vec: Vec<i64> = (0..out_ndim as usize)
            .map(|i| unsafe { *out_strides.add(i) })
            .collect();
        for (t_idx, &cp) in contiguous_ptrs.iter().enumerate() {
            let t = NslTensor::from_ptr(cp);
            let t_strides: Vec<i64> = (0..in_ndim).map(|i| unsafe { *t.strides.add(i) }).collect();
            for flat in 0..per_tensor {
                let mut remaining = flat;
                let mut in_multi: Vec<usize> = vec![0; in_ndim];
                for axis in 0..in_ndim {
                    in_multi[axis] = remaining / t_strides[axis] as usize;
                    remaining %= t_strides[axis] as usize;
                }
                let mut out_offset = t_idx * out_stride_vec[insert_pos] as usize;
                for (ia, &iv) in in_multi.iter().enumerate() {
                    let oa = if ia < insert_pos { ia } else { ia + 1 };
                    out_offset += iv * out_stride_vec[oa] as usize;
                }
                unsafe { *buf.add(out_offset) = *t.data_f64().add(flat) };
            }
        }
        buf as *mut c_void
    };

    let out = Box::new(NslTensor::new(
        data,
        out_shape,
        out_strides,
        out_ndim,
        out_len,
        first.device,
        first.dtype,
        1,
        0,
    ));
    let out_ptr = NslTensor::publish(out);

    // Free contiguous copies
    for &cp in &contiguous_ptrs {
        nsl_tensor_free(cp);
    }

    if autodiff::is_recording() {
        let ptrs: Vec<i64> = (0..num_tensors)
            .map(|i| unsafe { *list.data.add(i) })
            .collect();
        for &tp in &ptrs {
            let t = unsafe { &mut *(tp as *mut NslTensor) };
            t.refcount.fetch_add(1, Ordering::SeqCst);
        }
        autodiff::maybe_record(autodiff::TapeOp::Stack {
            inputs: ptrs,
            out: out_ptr,
            dim,
        });
    }

    out_ptr
}

/// Broadcast tensor to target shape (zero-copy stride view).
///
/// Expanded dimensions get stride=0 so they read the same physical data.
/// The result tensor does NOT own its data — it shares the source's buffer.
/// The source tensor's refcount is bumped to keep it alive while the view exists.
#[no_mangle]
pub extern "C" fn nsl_tensor_expand(tensor_ptr: i64, shape_list: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let list = NslList::from_ptr(shape_list);
    let target_ndim = list.len as usize;

    let mut target_shape: Vec<i64> = Vec::with_capacity(target_ndim);
    for i in 0..target_ndim {
        target_shape.push(unsafe { *list.data.add(i) });
    }

    let src_ndim = tensor.ndim as usize;
    if src_ndim > target_ndim {
        eprintln!(
            "nsl: expand: source ndim {} > target ndim {}",
            src_ndim, target_ndim
        );
        std::process::abort();
    }
    let pad = target_ndim.saturating_sub(src_ndim);

    // Validate: source dim must be 1 or match target
    for (i, &t) in target_shape.iter().enumerate() {
        let s = if i < pad { 1 } else { unsafe { *tensor.shape.add(i - pad) } };
        if s != 1 && s != t {
            eprintln!(
                "nsl: expand: cannot expand dim {} from {} to {}",
                i, s, t
            );
            std::process::abort();
        }
    }

    // Build output shape and strides (zero-copy: broadcast dims get stride=0)
    let out_shape = checked_alloc(target_ndim * std::mem::size_of::<i64>()) as *mut i64;
    let out_strides = checked_alloc(target_ndim * std::mem::size_of::<i64>()) as *mut i64;

    for (i, &t) in target_shape.iter().enumerate() {
        unsafe { *out_shape.add(i) = t };
        let src_dim_idx = i as isize - pad as isize;
        if src_dim_idx < 0 {
            // Padded leading dimension — no source data, stride=0
            unsafe { *out_strides.add(i) = 0 };
        } else {
            let s = unsafe { *tensor.shape.add(src_dim_idx as usize) };
            if s == 1 && t != 1 {
                // Broadcast: source dim is 1, target > 1 — stride=0 (zero-copy)
                unsafe { *out_strides.add(i) = 0 };
            } else {
                // Keep original stride from source tensor
                unsafe { *out_strides.add(i) = *tensor.strides.add(src_dim_idx as usize) };
            }
        }
    }

    let out_len = NslTensor::total_elements(out_shape, target_ndim as i64);

    // ZERO-COPY: share data pointer, bump the root owner's refcount to keep it alive.
    // Root-flattening: if the source is itself a view, follow data_owner to the true root
    // to avoid use-after-free if the intermediate view is freed first.
    let true_owner = if tensor.data_owner != 0 {
        let root = NslTensor::from_ptr(tensor.data_owner);
        root.refcount.fetch_add(1, Ordering::SeqCst);
        tensor.data_owner
    } else {
        tensor.refcount.fetch_add(1, Ordering::SeqCst);
        tensor_ptr
    };

    let out = Box::new(NslTensor::new(
        tensor.data,
        out_shape,
        out_strides,
        target_ndim as i64,
        out_len,
        tensor.device,
        tensor.dtype,
        0,
        true_owner,
    ));
    let out_ptr = NslTensor::publish(out);

    if autodiff::is_recording() {
        let original_shape = unsafe {
            std::slice::from_raw_parts(tensor.shape, tensor.ndim as usize)
        }.to_vec();
        autodiff::maybe_record(autodiff::TapeOp::Expand {
            input: tensor_ptr,
            out: out_ptr,
            original_shape,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(out_ptr);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Expand, vec![tensor_ptr], out_ptr, shape, rt.dtype, vec![]);
    }

    out_ptr
}

/// Create a [seq_len, seq_len] causal attention mask.
#[no_mangle]
pub extern "C" fn nsl_tensor_causal_mask(seq_len: i64) -> i64 {
    let n = seq_len as usize;
    let len = seq_len * seq_len;

    let out_shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = seq_len;
        *out_shape.add(1) = seq_len;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 2);
    // Use f32 (dtype=1) to match attention scores dtype and avoid promotion overhead
    let data = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;

    for i in 0..n {
        for j in 0..n {
            let val = if j <= i { 0.0_f32 } else { -1e9_f32 };
            unsafe { *data.add(i * n + j) = val };
        }
    }

    let out = Box::new(NslTensor::new(
        data as *mut c_void,
        out_shape,
        out_strides,
        2,
        len,
        0,
        1,
        1,
        0,
    ));
    NslTensor::publish(out)
}

/// Slice a tensor along a dimension: extract elements [start, end) along dim.
#[no_mangle]
pub extern "C" fn nsl_tensor_slice(tensor_ptr: i64, dim: i64, start: i64, end: i64) -> i64 {
    // GPU: native on-device slice kernel (no CPU round-trip)
    let t = NslTensor::from_ptr(tensor_ptr);
    if t.device > 0 {
        #[cfg(feature = "cuda")]
        {
            let ndim = t.ndim as usize;
            let d = if dim < 0 { (t.ndim + dim) as usize } else { dim as usize };
            let s = if start < 0 { (unsafe { *t.shape.add(d) } + start) as usize } else { start as usize };
            let dim_size = unsafe { *t.shape.add(d) };
            let e = if end < 0 { (dim_size + end) as usize }
                    else if end > dim_size { dim_size as usize }
                    else { end as usize };
            let slice_len = e.saturating_sub(s);
            if slice_len == 0 {
                // Empty slice: fall back to CPU path for correct empty tensor handling
                let cpu_t = super::nsl_tensor_to_device(tensor_ptr, 0);
                let result = nsl_tensor_slice(cpu_t, dim, start, end);
                let gpu_result = super::nsl_tensor_to_device(result, t.device as i64);
                super::nsl_tensor_free(cpu_t);
                super::nsl_tensor_free(result);
                return gpu_result;
            }
            return crate::cuda::gpu_slice_f32_with_shape(tensor_ptr, d, s, slice_len);
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }

    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;

    let d = if dim < 0 { (tensor.ndim + dim) as usize } else { dim as usize };
    assert!(d < ndim, "nsl_tensor_slice: dim {dim} out of range for ndim {ndim}");

    let dim_size = unsafe { *tensor.shape.add(d) };

    let s = if start < 0 { (dim_size + start).max(0) } else { start.min(dim_size) };
    let e = if end < 0 { (dim_size + end).max(0) } else { end.min(dim_size) };
    let slice_len = (e - s).max(0);

    let out_shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim {
        if i == d {
            unsafe { *out_shape.add(i) = slice_len };
        } else {
            unsafe { *out_shape.add(i) = *tensor.shape.add(i) };
        }
    }
    let out_strides = NslTensor::compute_strides(out_shape, tensor.ndim);
    let out_len = NslTensor::total_elements(out_shape, tensor.ndim);

    let in_strides: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *tensor.strides.add(i) })
        .collect();
    let o_strides: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *out_strides.add(i) })
        .collect();

    let data: *mut c_void = if tensor.dtype == 1 {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for flat in 0..out_len as usize {
            let mut remaining = flat;
            let mut in_offset: usize = 0;
            for axis in 0..ndim {
                let idx = remaining / o_strides[axis] as usize;
                remaining %= o_strides[axis] as usize;
                if axis == d {
                    in_offset += (idx + s as usize) * in_strides[axis] as usize;
                } else {
                    in_offset += idx * in_strides[axis] as usize;
                }
            }
            unsafe { *buf.add(flat) = *tensor.data_f32().add(in_offset) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for flat in 0..out_len as usize {
            let mut remaining = flat;
            let mut in_offset: usize = 0;
            for axis in 0..ndim {
                let idx = remaining / o_strides[axis] as usize;
                remaining %= o_strides[axis] as usize;
                if axis == d {
                    in_offset += (idx + s as usize) * in_strides[axis] as usize;
                } else {
                    in_offset += idx * in_strides[axis] as usize;
                }
            }
            unsafe { *buf.add(flat) = *tensor.data_f64().add(in_offset) };
        }
        buf as *mut c_void
    };

    let input_shape: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *tensor.shape.add(i) })
        .collect();

    let out = Box::new(NslTensor::new(
        data,
        out_shape,
        out_strides,
        tensor.ndim,
        out_len,
        tensor.device,
        tensor.dtype,
        1,
        0,
    ));
    let out_ptr = NslTensor::publish(out);

    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Slice {
            a: tensor_ptr,
            out: out_ptr,
            dim,
            start: s,
            input_shape,
        });
    }

    out_ptr
}

/// Concatenate a list of tensors along a dimension.
#[no_mangle]
pub extern "C" fn nsl_tensor_cat(tensor_list: i64, dim: i64) -> i64 {
    let list = NslList::from_ptr(tensor_list);
    let num_tensors = list.len as usize;
    assert!(num_tensors > 0, "nsl_tensor_cat: empty tensor list");

    // GPU redirect: if any tensor is on GPU, move all to CPU, cat, move result to GPU.
    let source_device = {
        let first = NslTensor::from_ptr(unsafe { *list.data });
        first.device
    };
    if source_device > 0 {
        let cpu_ptrs: Vec<i64> = (0..num_tensors)
            .map(|i| super::nsl_tensor_to_device(unsafe { *list.data.add(i) }, 0))
            .collect();
        let tmp_list = crate::list::nsl_list_new();
        for &cp in &cpu_ptrs {
            crate::list::nsl_list_push(tmp_list, cp);
        }
        let cpu_result = nsl_tensor_cat(tmp_list, dim);
        let gpu_result = super::nsl_tensor_to_device(cpu_result, source_device as i64);
        super::nsl_tensor_free(cpu_result);
        for &cp in &cpu_ptrs {
            super::nsl_tensor_free(cp);
        }
        crate::list::nsl_list_free(tmp_list);
        return gpu_result;
    }

    // Ensure all inputs are contiguous for flat-indexed copy
    let contiguous_ptrs: Vec<i64> = (0..num_tensors)
        .map(|i| nsl_tensor_contiguous(unsafe { *list.data.add(i) }))
        .collect();

    let first = NslTensor::from_ptr(contiguous_ptrs[0]);
    let ndim = first.ndim as usize;
    let d = if dim < 0 { (first.ndim + dim) as usize } else { dim as usize };
    assert!(d < ndim, "nsl_tensor_cat: dim {dim} out of range for ndim {ndim}");

    let mut split_sizes: Vec<i64> = Vec::with_capacity(num_tensors);
    let mut total_cat_dim: i64 = 0;

    for &cp in &contiguous_ptrs {
        let t = NslTensor::from_ptr(cp);
        assert_eq!(t.ndim as usize, ndim, "nsl_tensor_cat: ndim mismatch");
        let cat_size = unsafe { *t.shape.add(d) };
        split_sizes.push(cat_size);
        total_cat_dim += cat_size;
        for axis in 0..ndim {
            if axis != d {
                let s1 = unsafe { *first.shape.add(axis) };
                let s2 = unsafe { *t.shape.add(axis) };
                assert_eq!(s1, s2, "nsl_tensor_cat: shape mismatch at dim {axis}: {s1} vs {s2}");
            }
        }
    }

    let out_shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim {
        if i == d {
            unsafe { *out_shape.add(i) = total_cat_dim };
        } else {
            unsafe { *out_shape.add(i) = *first.shape.add(i) };
        }
    }
    let out_ndim = first.ndim;
    let out_dtype = first.dtype;
    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);
    let out_len = NslTensor::total_elements(out_shape, out_ndim);

    let o_strides: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *out_strides.add(i) })
        .collect();

    let data: *mut c_void = if out_dtype == 1 {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        let mut cat_offset: usize = 0;
        for (t_idx, &sz) in split_sizes.iter().enumerate().take(num_tensors) {
            let t = NslTensor::from_ptr(contiguous_ptrs[t_idx]);
            let t_strides: Vec<i64> = (0..ndim).map(|i| unsafe { *t.strides.add(i) }).collect();
            for flat in 0..t.len as usize {
                let mut remaining = flat;
                let mut out_offset: usize = 0;
                for axis in 0..ndim {
                    let idx = remaining / t_strides[axis] as usize;
                    remaining %= t_strides[axis] as usize;
                    if axis == d {
                        out_offset += (idx + cat_offset) * o_strides[axis] as usize;
                    } else {
                        out_offset += idx * o_strides[axis] as usize;
                    }
                }
                unsafe { *buf.add(out_offset) = *t.data_f32().add(flat) };
            }
            cat_offset += sz as usize;
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        let mut cat_offset: usize = 0;
        for (t_idx, &sz) in split_sizes.iter().enumerate().take(num_tensors) {
            let t = NslTensor::from_ptr(contiguous_ptrs[t_idx]);
            let t_strides: Vec<i64> = (0..ndim).map(|i| unsafe { *t.strides.add(i) }).collect();
            for flat in 0..t.len as usize {
                let mut remaining = flat;
                let mut out_offset: usize = 0;
                for axis in 0..ndim {
                    let idx = remaining / t_strides[axis] as usize;
                    remaining %= t_strides[axis] as usize;
                    if axis == d {
                        out_offset += (idx + cat_offset) * o_strides[axis] as usize;
                    } else {
                        out_offset += idx * o_strides[axis] as usize;
                    }
                }
                unsafe { *buf.add(out_offset) = *t.data_f64().add(flat) };
            }
            cat_offset += sz as usize;
        }
        buf as *mut c_void
    };

    let out = Box::new(NslTensor::new(
        data,
        out_shape,
        out_strides,
        out_ndim,
        out_len,
        first.device,
        out_dtype,
        1,
        0,
    ));
    let out_ptr = NslTensor::publish(out);

    // Free contiguous copies
    for &cp in &contiguous_ptrs {
        nsl_tensor_free(cp);
    }

    let input_ptrs: Vec<i64> = (0..num_tensors)
        .map(|i| unsafe { *list.data.add(i) })
        .collect();

    if autodiff::is_recording() {
        for &tp in &input_ptrs {
            let t = unsafe { &mut *(tp as *mut NslTensor) };
            t.refcount.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[cfg(feature = "interop")]
    let trace_input_ptrs = input_ptrs.clone();

    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Cat {
            inputs: input_ptrs,
            out: out_ptr,
            dim,
            split_sizes,
        });
    }

    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(out_ptr);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Concat, trace_input_ptrs, out_ptr, shape, rt.dtype, vec![]);
    }

    out_ptr
}

/// Fused rotate_half for RoPE (LLaMA-2 half-split variant):
/// rotate_half(x) = cat(-x[..., D/2:], x[..., :D/2], dim=-1)
///
/// For each contiguous chunk of `last_dim` elements in the flattened data:
///   output[0..half] = -input[half..last_dim]   (negate second half)
///   output[half..last_dim] = input[0..half]     (copy first half)
#[no_mangle]
pub extern "C" fn nsl_tensor_rotate_half(tensor_ptr: i64) -> i64 {
    // GPU redirect: transfer to CPU, rotate, transfer result back.
    let t_check = NslTensor::from_ptr(tensor_ptr);
    if t_check.device > 0 {
        let cpu_t = super::nsl_tensor_to_device(tensor_ptr, 0);
        let result = nsl_tensor_rotate_half(cpu_t);
        let gpu_result = super::nsl_tensor_to_device(result, t_check.device as i64);
        super::nsl_tensor_free(cpu_t);
        super::nsl_tensor_free(result);
        return gpu_result;
    }

    let t_c = nsl_tensor_contiguous(tensor_ptr);
    let tensor = NslTensor::from_ptr(t_c);
    let ndim = tensor.ndim as usize;

    if ndim == 0 {
        eprintln!("nsl: rotate_half requires at least 1 dimension");
        std::process::abort();
    }

    let last_dim = unsafe { *tensor.shape.add(ndim - 1) } as usize;
    if !last_dim.is_multiple_of(2) {
        eprintln!(
            "nsl: rotate_half requires even last dimension, got {}",
            last_dim
        );
        std::process::abort();
    }
    let half = last_dim / 2;
    let total = tensor.len as usize;
    let num_chunks = total / last_dim;

    let shape = NslTensor::copy_shape(tensor.shape, tensor.ndim);
    let strides = NslTensor::compute_strides(shape, tensor.ndim);

    let data: *mut c_void = if tensor.dtype == 1 {
        let buf = checked_alloc(total * std::mem::size_of::<f32>()) as *mut f32;
        let src = tensor.data_f32();
        for chunk in 0..num_chunks {
            let base = chunk * last_dim;
            for i in 0..half {
                // output[0..half] = -input[half..last_dim]
                unsafe { *buf.add(base + i) = -(*src.add(base + half + i)) };
                // output[half..last_dim] = input[0..half]
                unsafe { *buf.add(base + half + i) = *src.add(base + i) };
            }
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc(total * std::mem::size_of::<f64>()) as *mut f64;
        let src = tensor.data_f64();
        for chunk in 0..num_chunks {
            let base = chunk * last_dim;
            for i in 0..half {
                unsafe { *buf.add(base + i) = -(*src.add(base + half + i)) };
                unsafe { *buf.add(base + half + i) = *src.add(base + i) };
            }
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor::new(
        data,
        shape,
        strides,
        tensor.ndim,
        tensor.len,
        tensor.device,
        tensor.dtype,
        1,
        0,
    ));
    let result = NslTensor::publish(result);
    nsl_tensor_free(t_c);
    result
}

/// Materialize a non-contiguous tensor (e.g. from `expand`) into a contiguous copy.
///
/// If the tensor is already contiguous (strides match row-major layout), the same
/// pointer is returned with its refcount bumped (zero-copy fast path).
/// Otherwise a new tensor is allocated with data copied element-by-element
/// using multi-dimensional coordinate decomposition over the source strides.
#[no_mangle]
pub extern "C" fn nsl_tensor_contiguous(tensor_ptr: i64) -> i64 {
    let t = NslTensor::from_ptr(tensor_ptr);
    let ndim = t.ndim as usize;

    // Check if already contiguous: compare actual strides to expected row-major strides
    let expected = NslTensor::compute_strides(t.shape, t.ndim);
    let mut is_contiguous = true;
    for i in 0..ndim {
        if unsafe { *t.strides.add(i) != *expected.add(i) } {
            is_contiguous = false;
            break;
        }
    }
    // Free the temp expected strides
    unsafe { crate::memory::checked_free(expected as *mut u8, ndim * std::mem::size_of::<i64>()) };

    if is_contiguous {
        // Already contiguous -- bump refcount, return same pointer
        t.refcount.fetch_add(1, Ordering::SeqCst);
        return tensor_ptr;
    }

    // GPU path: native on-device strided copy kernel (no CPU round-trip)
    if t.device > 0 {
        #[cfg(feature = "cuda")]
        {
            return crate::cuda::gpu_strided_copy_f32(tensor_ptr);
        }
        #[cfg(not(feature = "cuda"))]
        {
            // Fallback: CPU round-trip
            let cpu_t = super::nsl_tensor_to_device(tensor_ptr, 0);
            let gpu_contig = super::nsl_tensor_to_device(cpu_t, t.device as i64);
            super::nsl_tensor_free(cpu_t);
            return gpu_contig;
        }
    }

    // Materialize: walk through all elements using multi-dim coords and source strides
    let len = t.len;
    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let out_strides = NslTensor::compute_strides(shape, t.ndim);

    // For each flat output index, compute the n-dim coordinates,
    // then compute the source offset using source strides.
    if t.dtype == 1 {
        // f32
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for flat in 0..len as usize {
            let mut remaining = flat;
            let mut src_offset: usize = 0;
            for d in 0..ndim {
                let stride = unsafe { *out_strides.add(d) } as usize;
                let coord = if stride > 0 { remaining / stride } else { 0 };
                if stride > 0 {
                    remaining %= stride;
                }
                let src_stride = unsafe { *t.strides.add(d) } as usize;
                src_offset += coord * src_stride;
            }
            unsafe { *buf.add(flat) = *t.data_f32().add(src_offset) };
        }
        let result = Box::new(NslTensor::new(
            buf as *mut c_void,
            shape,
            out_strides,
            t.ndim,
            len,
            t.device,
            t.dtype,
            1,
            0,
        ));
        NslTensor::publish(result)
    } else {
        // f64
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for flat in 0..len as usize {
            let mut remaining = flat;
            let mut src_offset: usize = 0;
            for d in 0..ndim {
                let stride = unsafe { *out_strides.add(d) } as usize;
                let coord = if stride > 0 { remaining / stride } else { 0 };
                if stride > 0 {
                    remaining %= stride;
                }
                let src_stride = unsafe { *t.strides.add(d) } as usize;
                src_offset += coord * src_stride;
            }
            unsafe { *buf.add(flat) = *t.data_f64().add(src_offset) };
        }
        let result = Box::new(NslTensor::new(
            buf as *mut c_void,
            shape,
            out_strides,
            t.ndim,
            len,
            t.device,
            t.dtype,
            1,
            0,
        ));
        NslTensor::publish(result)
    }
}
