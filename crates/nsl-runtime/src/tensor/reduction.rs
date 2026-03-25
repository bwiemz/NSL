//! Tensor reduction operations: sum, mean, max, gather, softmax.

use std::ffi::c_void;
use std::sync::atomic::Ordering;

use crate::autodiff;
use crate::memory::checked_alloc_zeroed;

use super::creation::create_scalar_tensor_dtype;
use super::{get_shape_vec, get_strides_vec, nsl_tensor_contiguous, nsl_tensor_free, NslTensor};

/// Global sum reduction (backward compatible wrapper).
#[no_mangle]
pub extern "C" fn nsl_tensor_sum(tensor_ptr: i64) -> i64 {
    nsl_tensor_sum_dim(tensor_ptr, -1, 0)
}

/// Global mean reduction (backward compatible wrapper).
#[no_mangle]
pub extern "C" fn nsl_tensor_mean(tensor_ptr: i64) -> i64 {
    nsl_tensor_mean_dim(tensor_ptr, -1, 0)
}

/// Sum reduction along a dimension (dim=-1 means global).
#[no_mangle]
pub extern "C" fn nsl_tensor_sum_dim(tensor_ptr: i64, dim: i64, keepdim: i64) -> i64 {
    // GPU dispatch: native GPU reduction kernels.
    {
        let tensor = NslTensor::from_ptr(tensor_ptr);
        if tensor.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let keepdim_bool = keepdim != 0;
                let c_ptr = super::nsl_tensor_contiguous(tensor_ptr);
                let result = if dim == -1 {
                    // Global sum
                    crate::cuda::gpu_global_sum_f32(c_ptr)
                } else {
                    let ndim = tensor.ndim as usize;
                    let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };
                    crate::cuda::gpu_sum_dim_f32(c_ptr, d, keepdim_bool)
                };
                if c_ptr != tensor_ptr { super::nsl_tensor_free(c_ptr); }
                let input_shape = get_shape_vec(tensor);
                if autodiff::is_recording() {
                    autodiff::maybe_record(autodiff::TapeOp::SumReduce {
                        a: tensor_ptr,
                        out: result,
                        dim,
                        keepdim: keepdim_bool,
                        input_shape,
                    });
                }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let cpu_t = super::nsl_tensor_to_device(tensor_ptr, 0);
                let result = nsl_tensor_sum_dim(cpu_t, dim, keepdim);
                let gpu_result = super::nsl_tensor_to_device(result, tensor.device as i64);
                super::nsl_tensor_free(cpu_t);
                super::nsl_tensor_free(result);
                return gpu_result;
            }
        }
    }
    let t_c = nsl_tensor_contiguous(tensor_ptr);
    let tensor = NslTensor::from_ptr(t_c);
    let input_shape = get_shape_vec(tensor);
    let keepdim_bool = keepdim != 0;

    if dim == -1 {
        // Global reduction
        let total = if tensor.dtype == 1 {
            let mut s = 0.0_f32;
            for i in 0..tensor.len as usize {
                s += unsafe { *tensor.data_f32().add(i) };
            }
            s as f64
        } else {
            let mut s = 0.0_f64;
            for i in 0..tensor.len as usize {
                s += unsafe { *tensor.data_f64().add(i) };
            }
            s
        };
        let result = create_scalar_tensor_dtype(total, tensor.dtype);
        nsl_tensor_free(t_c);
        if autodiff::is_recording() {
            autodiff::maybe_record(autodiff::TapeOp::SumReduce {
                a: tensor_ptr,
                out: result,
                dim: -1,
                keepdim: false,
                input_shape,
            });
        }
        return result;
    }

    let ndim = tensor.ndim as usize;
    let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };
    if d >= ndim {
        eprintln!("nsl: sum_dim dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let in_strides = get_strides_vec(tensor);

    // Compute output shape
    let out_shape: Vec<i64> = if keepdim_bool {
        input_shape.iter().enumerate()
            .map(|(i, &s)| if i == d { 1 } else { s })
            .collect()
    } else {
        input_shape.iter().enumerate()
            .filter(|&(i, _)| i != d)
            .map(|(_, &s)| s)
            .collect()
    };

    let result_ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&out_shape, tensor.dtype);
    let result = NslTensor::from_ptr(result_ptr);
    let out_strides = get_strides_vec(result);

    // Iterate all elements of input, accumulate into output
    let total_in = tensor.len as usize;
    for flat_in in 0..total_in {
        // Decompose flat_in into multi-index using input strides
        let mut remaining = flat_in;
        let mut indices = vec![0usize; ndim];
        for dd in 0..ndim {
            indices[dd] = remaining / in_strides[dd];
            remaining %= in_strides[dd];
        }

        // Compute output flat index (skip or collapse the reduced dim)
        let mut out_flat = 0usize;
        let mut oi = 0usize;
        for (dd, &idx) in indices.iter().enumerate().take(ndim) {
            if dd == d {
                if keepdim_bool {
                    // dim is kept as size 1, index=0
                    oi += 1;
                }
                continue;
            }
            out_flat += idx * out_strides[oi];
            oi += 1;
        }

        if tensor.dtype == 1 {
            let val = unsafe { *tensor.data_f32().add(flat_in) };
            unsafe { *result.data_f32().add(out_flat) += val };
        } else {
            let val = unsafe { *tensor.data_f64().add(flat_in) };
            unsafe { *result.data_f64().add(out_flat) += val };
        }
    }

    nsl_tensor_free(t_c);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::SumReduce {
            a: tensor_ptr,
            out: result_ptr,
            dim,
            keepdim: keepdim_bool,
            input_shape,
        });
    }
    result_ptr
}

/// Mean reduction along a dimension (dim=-1 means global).
#[no_mangle]
pub extern "C" fn nsl_tensor_mean_dim(tensor_ptr: i64, dim: i64, keepdim: i64) -> i64 {
    // GPU dispatch: use native GPU sum then divide by count.
    {
        let tensor = NslTensor::from_ptr(tensor_ptr);
        if tensor.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let keepdim_bool = keepdim != 0;
                let input_shape = get_shape_vec(tensor);
                let c_ptr = super::nsl_tensor_contiguous(tensor_ptr);

                if dim == -1 {
                    // Global mean: gpu_global_sum / total_elements
                    let num_elements = tensor.len;
                    let sum_ptr = crate::cuda::gpu_global_sum_f32(c_ptr);
                    if c_ptr != tensor_ptr { super::nsl_tensor_free(c_ptr); }
                    // Divide by num_elements using scalar op
                    let inv = 1.0_f32 / num_elements as f32;
                    crate::cuda::gpu_scalar_op_inplace(sum_ptr, inv, crate::cuda::kernels::MUL_SCALAR_F32_PTX, "nsl_mul_scalar_f32\0");
                    if autodiff::is_recording() {
                        autodiff::maybe_record(autodiff::TapeOp::MeanReduce {
                            a: tensor_ptr, out: sum_ptr, dim: -1, keepdim: false,
                            num_elements, input_shape,
                        });
                    }
                    return sum_ptr;
                }

                let ndim = tensor.ndim as usize;
                let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };
                let dim_size = input_shape[d];
                let sum_ptr = crate::cuda::gpu_sum_dim_f32(c_ptr, d, keepdim_bool);
                if c_ptr != tensor_ptr { super::nsl_tensor_free(c_ptr); }
                // Divide by dim_size using scalar op
                let inv = 1.0_f32 / dim_size as f32;
                crate::cuda::gpu_scalar_op_inplace(sum_ptr, inv, crate::cuda::kernels::MUL_SCALAR_F32_PTX, "nsl_mul_scalar_f32\0");
                if autodiff::is_recording() {
                    autodiff::maybe_record(autodiff::TapeOp::MeanReduce {
                        a: tensor_ptr, out: sum_ptr, dim, keepdim: keepdim_bool,
                        num_elements: dim_size, input_shape,
                    });
                }
                return sum_ptr;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let cpu_t = super::nsl_tensor_to_device(tensor_ptr, 0);
                let result = nsl_tensor_mean_dim(cpu_t, dim, keepdim);
                let gpu_result = super::nsl_tensor_to_device(result, tensor.device as i64);
                super::nsl_tensor_free(cpu_t);
                super::nsl_tensor_free(result);
                return gpu_result;
            }
        }
    }
    let t_c = nsl_tensor_contiguous(tensor_ptr);
    let tensor = NslTensor::from_ptr(t_c);
    let input_shape = get_shape_vec(tensor);
    let keepdim_bool = keepdim != 0;

    if dim == -1 {
        // Global reduction
        if tensor.len == 0 {
            nsl_tensor_free(t_c);
            return create_scalar_tensor_dtype(0.0, tensor.dtype);
        }
        let num_elements = tensor.len;
        let total = if tensor.dtype == 1 {
            let mut s = 0.0_f32;
            for i in 0..num_elements as usize {
                s += unsafe { *tensor.data_f32().add(i) };
            }
            (s / num_elements as f32) as f64
        } else {
            let mut s = 0.0_f64;
            for i in 0..num_elements as usize {
                s += unsafe { *tensor.data_f64().add(i) };
            }
            s / num_elements as f64
        };
        let result = create_scalar_tensor_dtype(total, tensor.dtype);
        nsl_tensor_free(t_c);
        if autodiff::is_recording() {
            autodiff::maybe_record(autodiff::TapeOp::MeanReduce {
                a: tensor_ptr,
                out: result,
                dim: -1,
                keepdim: false,
                num_elements,
                input_shape,
            });
        }
        return result;
    }

    let ndim = tensor.ndim as usize;
    let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };
    if d >= ndim {
        eprintln!("nsl: mean_dim dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let dim_size = input_shape[d];

    // Sum first, then divide
    let sum_ptr = nsl_tensor_sum_dim(tensor_ptr, dim, keepdim);

    // Remove the SumReduce tape entry that sum_dim just recorded (we want MeanReduce instead)
    if autodiff::is_recording() {
        crate::autodiff::pop_last_op();
    }

    // Divide by dim_size
    let result = NslTensor::from_ptr(sum_ptr);
    if result.dtype == 1 {
        for i in 0..result.len as usize {
            unsafe { *result.data_f32().add(i) /= dim_size as f32 };
        }
    } else {
        for i in 0..result.len as usize {
            unsafe { *result.data_f64().add(i) /= dim_size as f64 };
        }
    }

    nsl_tensor_free(t_c);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::MeanReduce {
            a: tensor_ptr,
            out: sum_ptr,
            dim,
            keepdim: keepdim_bool,
            num_elements: dim_size,
            input_shape,
        });
    }
    sum_ptr
}

/// Reduce max along a dimension.
#[no_mangle]
pub extern "C" fn nsl_tensor_reduce_max(tensor_ptr: i64, dim: i64, keepdim: i64) -> i64 {
    // GPU dispatch: native GPU max reduction kernel.
    {
        let tensor = NslTensor::from_ptr(tensor_ptr);
        if tensor.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let keepdim_bool = keepdim != 0;
                let ndim = tensor.ndim as usize;
                let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };
                let input_shape = get_shape_vec(tensor);
                let c_ptr = super::nsl_tensor_contiguous(tensor_ptr);
                let result = crate::cuda::gpu_max_dim_f32(c_ptr, d, keepdim_bool);
                if c_ptr != tensor_ptr { super::nsl_tensor_free(c_ptr); }
                // Note: argmax is not computed on GPU — autodiff backward for
                // reduce_max on GPU tensors will use zeros (no gradient flows through max indices).
                if autodiff::is_recording() {
                    let ct = NslTensor::from_ptr(result);
                    let out_len = ct.len as usize;
                    autodiff::maybe_record(autodiff::TapeOp::ReduceMax {
                        a: tensor_ptr,
                        out: result,
                        dim,
                        keepdim: keepdim_bool,
                        saved_argmax: vec![0usize; out_len],
                        input_shape,
                    });
                }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            {
                let cpu_t = super::nsl_tensor_to_device(tensor_ptr, 0);
                let result = nsl_tensor_reduce_max(cpu_t, dim, keepdim);
                let gpu_result = super::nsl_tensor_to_device(result, tensor.device as i64);
                super::nsl_tensor_free(cpu_t);
                super::nsl_tensor_free(result);
                return gpu_result;
            }
        }
    }
    let t_c = nsl_tensor_contiguous(tensor_ptr);
    let tensor = NslTensor::from_ptr(t_c);
    let input_shape = get_shape_vec(tensor);
    let keepdim_bool = keepdim != 0;
    let ndim = tensor.ndim as usize;
    let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };

    if d >= ndim {
        eprintln!("nsl: reduce_max dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let in_strides = get_strides_vec(tensor);

    // Compute output shape
    let out_shape: Vec<i64> = if keepdim_bool {
        input_shape.iter().enumerate()
            .map(|(i, &s)| if i == d { 1 } else { s })
            .collect()
    } else {
        input_shape.iter().enumerate()
            .filter(|&(i, _)| i != d)
            .map(|(_, &s)| s)
            .collect()
    };

    let result_ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&out_shape, tensor.dtype);
    let result = NslTensor::from_ptr(result_ptr);
    let out_strides = get_strides_vec(result);
    let out_total = result.len as usize;

    // Initialize with -inf
    if tensor.dtype == 1 {
        for i in 0..out_total {
            unsafe { *result.data_f32().add(i) = f32::NEG_INFINITY };
        }
    } else {
        for i in 0..out_total {
            unsafe { *result.data_f64().add(i) = f64::NEG_INFINITY };
        }
    }

    // Track argmax per output position
    let mut argmax = vec![0usize; out_total];

    // Iterate all elements of input
    let total_in = tensor.len as usize;
    for flat_in in 0..total_in {
        let mut remaining = flat_in;
        let mut indices = vec![0usize; ndim];
        for dd in 0..ndim {
            indices[dd] = remaining / in_strides[dd];
            remaining %= in_strides[dd];
        }

        // Compute output flat index
        let mut out_flat = 0usize;
        let mut oi = 0usize;
        for (dd, &idx) in indices.iter().enumerate().take(ndim) {
            if dd == d {
                if keepdim_bool {
                    oi += 1;
                }
                continue;
            }
            out_flat += idx * out_strides[oi];
            oi += 1;
        }

        if tensor.dtype == 1 {
            let val = unsafe { *tensor.data_f32().add(flat_in) };
            let cur = unsafe { *result.data_f32().add(out_flat) };
            if val > cur {
                unsafe { *result.data_f32().add(out_flat) = val };
                argmax[out_flat] = indices[d];
            }
        } else {
            let val = unsafe { *tensor.data_f64().add(flat_in) };
            let cur = unsafe { *result.data_f64().add(out_flat) };
            if val > cur {
                unsafe { *result.data_f64().add(out_flat) = val };
                argmax[out_flat] = indices[d];
            }
        }
    }

    nsl_tensor_free(t_c);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::ReduceMax {
            a: tensor_ptr,
            out: result_ptr,
            dim,
            keepdim: keepdim_bool,
            saved_argmax: argmax,
            input_shape,
        });
    }
    result_ptr
}

/// Gather along a dimension: output[b] = tensor[b, indices[b]] (for dim=1, 2D input).
#[no_mangle]
pub extern "C" fn nsl_tensor_gather(tensor_ptr: i64, dim: i64, indices_ptr: i64) -> i64 {
    // GPU dispatch: native GPU gather kernel (dim 0) or CPU-redirect (other dims).
    {
        let tensor = NslTensor::from_ptr(tensor_ptr);
        if tensor.device > 0 {
            let ndim = tensor.ndim;
            let d = if dim < 0 { ndim + dim } else { dim };
            #[cfg(feature = "cuda")]
            if d == 0 {
                // Native GPU gather along dim 0
                let t_c = nsl_tensor_contiguous(tensor_ptr);
                let result = crate::cuda::gpu_gather_f32(t_c, indices_ptr);
                if t_c != tensor_ptr { super::nsl_tensor_free(t_c); }
                return result;
            }
            // Fallback for non-zero dims: CPU redirect
            let cpu_t = super::nsl_tensor_to_device(tensor_ptr, 0);
            let cpu_i = super::nsl_tensor_to_device(indices_ptr, 0);
            let result = nsl_tensor_gather(cpu_t, dim, cpu_i);
            let gpu_result = super::nsl_tensor_to_device(result, tensor.device as i64);
            super::nsl_tensor_free(cpu_t);
            super::nsl_tensor_free(cpu_i);
            super::nsl_tensor_free(result);
            return gpu_result;
        }
    }
    let t_c = nsl_tensor_contiguous(tensor_ptr);
    let i_c = nsl_tensor_contiguous(indices_ptr);
    let tensor = NslTensor::from_ptr(t_c);
    let indices = NslTensor::from_ptr(i_c);
    let input_shape = get_shape_vec(tensor);
    let ndim = tensor.ndim as usize;
    let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };

    if d >= ndim {
        eprintln!("nsl: gather dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let num_indices = indices.len as usize;
    let gather_dim_size = input_shape[d] as usize;

    // Output shape: input shape with d_dim replaced by num_indices
    let out_shape: Vec<i64> = input_shape.iter().enumerate()
        .filter(|&(i, _)| i != d)
        .map(|(_, &s)| s)
        .collect();

    // Compute the number of "outer" elements (product of dims before d)
    // and "inner" elements (product of dims after d)
    let outer: usize = input_shape[..d].iter().map(|&s| s as usize).product::<usize>().max(1);
    let inner: usize = input_shape[d+1..].iter().map(|&s| s as usize).product::<usize>().max(1);

    // indices must match outer dimension count
    if num_indices != outer {
        eprintln!(
            "nsl: gather dim={} requires indices length ({}) == outer size ({})",
            d, num_indices, outer
        );
        std::process::abort();
    }

    let result_ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&out_shape, tensor.dtype);
    let result = NslTensor::from_ptr(result_ptr);

    // General N-D gather along dimension d:
    for o in 0..outer {
        let idx = indices.read_index(o) as usize;
        if idx >= gather_dim_size {
            eprintln!(
                "nsl: gather index {} out of bounds for dim {} with size {}",
                idx, d, gather_dim_size
            );
            std::process::abort();
        }
        let in_base = o * gather_dim_size * inner + idx * inner;
        let out_base = o * inner;
        for k in 0..inner {
            if tensor.dtype == 1 {
                let val = unsafe { *tensor.data_f32().add(in_base + k) };
                unsafe { *result.data_f32().add(out_base + k) = val };
            } else {
                let val = unsafe { *tensor.data_f64().add(in_base + k) };
                unsafe { *result.data_f64().add(out_base + k) = val };
            }
        }
    }

    nsl_tensor_free(t_c);
    nsl_tensor_free(i_c);
    if autodiff::is_recording() {
        // Save indices for backward
        NslTensor::from_ptr(indices_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::Gather {
            a: tensor_ptr,
            out: result_ptr,
            dim,
            indices_ptr,
            input_shape,
        });
    }
    result_ptr
}

/// Softmax along a dimension.
#[no_mangle]
pub extern "C" fn nsl_tensor_softmax(tensor_ptr: i64, dim: i64) -> i64 {
    // GPU dispatch: native GPU softmax kernel (last dim) or CPU-redirect (other dims)
    {
        let tensor = NslTensor::from_ptr(tensor_ptr);
        if tensor.device > 0 {
            // Normalize dim
            let ndim = tensor.ndim;
            let d = if dim < 0 { ndim + dim } else { dim };
            // Native GPU kernel for last-dim softmax (the common case)
            if d == ndim - 1 {
                #[cfg(feature = "cuda")]
                {
                    // Ensure contiguous for GPU kernel
                    let c_ptr = super::nsl_tensor_contiguous(tensor_ptr);
                    let result = crate::cuda::gpu_softmax_f32(c_ptr);
                    if c_ptr != tensor_ptr {
                        super::nsl_tensor_free(c_ptr);
                    }
                    return result;
                }
            }
            // Non-last-dim: CPU redirect
            let cpu_t = super::nsl_tensor_to_device(tensor_ptr, 0);
            let result = nsl_tensor_softmax(cpu_t, dim);
            let gpu_result = super::nsl_tensor_to_device(result, tensor.device as i64);
            super::nsl_tensor_free(cpu_t);
            super::nsl_tensor_free(result);
            return gpu_result;
        }
    }
    let a_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    // Normalize dim
    let d = if dim < 0 { (ndim + dim) as usize } else { dim as usize };

    let a_shape: Vec<i64> = (0..ndim as usize).map(|i| unsafe { *a.shape.add(i) }).collect();
    let a_strides: Vec<i64> = (0..ndim as usize).map(|i| unsafe { *a.strides.add(i) }).collect();
    let dim_size = a_shape[d] as usize;

    let num_slices = (len as usize) / dim_size;

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc_zeroed((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for slice_idx in 0..num_slices {
            let mut remaining = slice_idx;
            let mut base_offset: usize = 0;
            for axis in (0..ndim as usize).rev() {
                if axis == d { continue; }
                let idx = remaining % (a_shape[axis] as usize);
                remaining /= a_shape[axis] as usize;
                base_offset += idx * (a_strides[axis] as usize);
            }
            let mut max_val = f32::NEG_INFINITY;
            for k in 0..dim_size {
                let offset = base_offset + k * (a_strides[d] as usize);
                let val = unsafe { *a.data_f32().add(offset) };
                if val > max_val { max_val = val; }
            }
            let mut sum = 0.0_f32;
            for k in 0..dim_size {
                let offset = base_offset + k * (a_strides[d] as usize);
                let e = (unsafe { *a.data_f32().add(offset) } - max_val).exp();
                unsafe { *buf.add(offset) = e };
                sum += e;
            }
            for k in 0..dim_size {
                let offset = base_offset + k * (a_strides[d] as usize);
                unsafe { *buf.add(offset) /= sum };
            }
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc_zeroed((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for slice_idx in 0..num_slices {
            let mut remaining = slice_idx;
            let mut base_offset: usize = 0;
            for axis in (0..ndim as usize).rev() {
                if axis == d { continue; }
                let idx = remaining % (a_shape[axis] as usize);
                remaining /= a_shape[axis] as usize;
                base_offset += idx * (a_strides[axis] as usize);
            }
            let mut max_val = f64::NEG_INFINITY;
            for k in 0..dim_size {
                let offset = base_offset + k * (a_strides[d] as usize);
                let val = unsafe { *a.data_f64().add(offset) };
                if val > max_val { max_val = val; }
            }
            let mut sum = 0.0_f64;
            for k in 0..dim_size {
                let offset = base_offset + k * (a_strides[d] as usize);
                let e = (unsafe { *a.data_f64().add(offset) } - max_val).exp();
                unsafe { *buf.add(offset) = e };
                sum += e;
            }
            for k in 0..dim_size {
                let offset = base_offset + k * (a_strides[d] as usize);
                unsafe { *buf.add(offset) /= sum };
            }
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor::new(
        data,
        shape,
        strides,
        ndim,
        len,
        a.device,
        a.dtype,
        1,
        0,
    ));
    let result = NslTensor::publish(result);
    nsl_tensor_free(a_c);
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::Softmax {
            a: tensor_ptr,
            out: result,
            saved_out: result,
            dim,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(
            crate::trace::OpType::Softmax,
            vec![tensor_ptr],
            result,
            shape,
            rt.dtype,
            vec![("axis".to_string(), crate::trace::AttrValue::Int(dim))],
        );
    }
    result
}
