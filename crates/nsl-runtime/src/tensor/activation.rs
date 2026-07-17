//! Tensor activation and elementwise math operations: relu, sigmoid, tanh, gelu, silu,
//! exp, log, sqrt, abs, sign, clamp.

use std::ffi::c_void;
use std::sync::atomic::Ordering;

use crate::autodiff;
use crate::memory::checked_alloc;

use super::{nsl_tensor_contiguous, nsl_tensor_free, NslTensor};

// === Element-wise math ops ===

#[no_mangle]
pub extern "C" fn nsl_tensor_exp(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_elementwise_unary_inplace(tensor_ptr, crate::cuda::kernels::EXP_F32_PTX, "nsl_exp_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return tensor_ptr;
                }
                return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::EXP_F32_PTX, "nsl_exp_f32\0");
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned (CPU)
    {
        let t = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        if t.can_mutate_inplace() {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                for i in 0..len { unsafe { *d.add(i) = (*d.add(i)).exp() }; }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len { unsafe { *d.add(i) = (*d.add(i)).exp() }; }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return tensor_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f32().add(i)).exp() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f64().add(i)).exp() };
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
        autodiff::maybe_record(autodiff::TapeOp::Exp {
            a: tensor_ptr, out: result, saved_out: result,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Exp, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_log(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_elementwise_unary_inplace(tensor_ptr, crate::cuda::kernels::LOG_F32_PTX, "nsl_log_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return tensor_ptr;
                }
                return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::LOG_F32_PTX, "nsl_log_f32\0");
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned (CPU)
    {
        let t = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        if t.can_mutate_inplace() {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                for i in 0..len { unsafe { *d.add(i) = (*d.add(i)).ln() }; }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len { unsafe { *d.add(i) = (*d.add(i)).ln() }; }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return tensor_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f32().add(i)).ln() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f64().add(i)).ln() };
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
        NslTensor::from_ptr(tensor_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::Log {
            a: tensor_ptr, out: result, saved_a: tensor_ptr,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Log, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sqrt(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_elementwise_unary_inplace(tensor_ptr, crate::cuda::kernels::SQRT_F32_PTX, "nsl_sqrt_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return tensor_ptr;
                }
                return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::SQRT_F32_PTX, "nsl_sqrt_f32\0");
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned (CPU)
    {
        let t = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        if t.can_mutate_inplace() {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                for i in 0..len { unsafe { *d.add(i) = (*d.add(i)).sqrt() }; }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len { unsafe { *d.add(i) = (*d.add(i)).sqrt() }; }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return tensor_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f32().add(i)).sqrt() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f64().add(i)).sqrt() };
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
        autodiff::maybe_record(autodiff::TapeOp::Sqrt {
            a: tensor_ptr, out: result, saved_out: result,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Sqrt, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_abs(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_elementwise_unary_inplace(tensor_ptr, crate::cuda::kernels::ABS_F32_PTX, "nsl_abs_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return tensor_ptr;
                }
                return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::ABS_F32_PTX, "nsl_abs_f32\0");
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned (CPU)
    {
        let t = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        if t.can_mutate_inplace() {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                for i in 0..len { unsafe { *d.add(i) = (*d.add(i)).abs() }; }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len { unsafe { *d.add(i) = (*d.add(i)).abs() }; }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return tensor_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f32().add(i)).abs() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f64().add(i)).abs() };
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
        NslTensor::from_ptr(tensor_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::Abs {
            a: tensor_ptr, out: result, saved_a: tensor_ptr,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sign(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_elementwise_unary_inplace(tensor_ptr, crate::cuda::kernels::SIGN_F32_PTX, "nsl_sign_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return tensor_ptr;
                }
                return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::SIGN_F32_PTX, "nsl_sign_f32\0");
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned (CPU)
    {
        let t = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        if t.can_mutate_inplace() {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                for i in 0..len {
                    let v = unsafe { *d.add(i) };
                    unsafe { *d.add(i) = if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 } };
                }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len {
                    let v = unsafe { *d.add(i) };
                    unsafe { *d.add(i) = if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 } };
                }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return tensor_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f32().add(i) };
            unsafe {
                *buf.add(i) = if val > 0.0 { 1.0f32 } else if val < 0.0 { -1.0f32 } else { 0.0f32 };
            }
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f64().add(i) };
            unsafe {
                *buf.add(i) = if val > 0.0 { 1.0 } else if val < 0.0 { -1.0 } else { 0.0 };
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
    // sign is non-differentiable -- no tape recording
    let result = NslTensor::publish(result);
    nsl_tensor_free(a_c);
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_clamp(tensor_ptr: i64, min_val: f64, max_val: f64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_clamp_f32_inplace(tensor_ptr, min_val as f32, max_val as f32);
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return tensor_ptr;
                }
                return crate::cuda::gpu_clamp_f32(tensor_ptr, min_val as f32, max_val as f32);
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned (skip for i32 — needs dtype conversion)
    {
        let t = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        if t.dtype != 4 && t.can_mutate_inplace() {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                let (mn, mx) = (min_val as f32, max_val as f32);
                for i in 0..len { unsafe { *d.add(i) = (*d.add(i)).clamp(mn, mx) }; }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len { unsafe { *d.add(i) = (*d.add(i)).clamp(min_val, max_val) }; }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return tensor_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    // Output dtype: i32 inputs produce f32 output (clamp converts int→float)
    let out_dtype = if a.dtype == 4 { 1u16 } else { a.dtype };
    let data: *mut c_void = if a.dtype == 4 {
        // i32 → f32 with clamping (used for token ID masking in cross_entropy)
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        let (mn, mx) = (min_val as f32, max_val as f32);
        let src = a.data as *const i32;
        for i in 0..len as usize {
            let val = unsafe { *src.add(i) } as f32;
            unsafe { *buf.add(i) = val.clamp(mn, mx) };
        }
        buf as *mut c_void
    } else if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        let (mn, mx) = (min_val as f32, max_val as f32);
        for i in 0..len as usize {
            let val = unsafe { *a.data_f32().add(i) };
            unsafe { *buf.add(i) = val.clamp(mn, mx) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f64().add(i) };
            unsafe { *buf.add(i) = val.clamp(min_val, max_val) };
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
        out_dtype,
        1,
        0,
    ));
    let result = NslTensor::publish(result);
    nsl_tensor_free(a_c);
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::Clamp {
            a: tensor_ptr, out: result, saved_a: tensor_ptr, min_val, max_val,
        });
    }
    result
}

/// Helper for clamp backward.
pub(crate) fn nsl_tensor_clamp_backward(
    grad_ptr: i64,
    input_ptr: i64,
    min_val: f64,
    max_val: f64,
) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    #[cfg(feature = "cuda")]
    if grad.device > 0 {
        return crate::cuda::gpu_clamp_backward(grad_ptr, input_ptr, min_val as f32, max_val as f32);
    }
    let input = NslTensor::from_ptr(input_ptr);
    let len = input.len;
    let ndim = input.ndim;
    let shape = NslTensor::copy_shape(input.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if input.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        let (mn, mx) = (min_val as f32, max_val as f32);
        for i in 0..len as usize {
            let val = unsafe { *input.data_f32().add(i) };
            let g_val = unsafe { *grad.data_f32().add(i) };
            unsafe { *buf.add(i) = if val > mn && val < mx { g_val } else { 0.0 } };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let val = unsafe { *input.data_f64().add(i) };
            let g_val = unsafe { *grad.data_f64().add(i) };
            unsafe { *buf.add(i) = if val > min_val && val < max_val { g_val } else { 0.0 } };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor::new(
        data,
        shape,
        strides,
        ndim,
        len,
        input.device,
        input.dtype,
        1,
        0,
    ));
    NslTensor::publish(result)
}

// === Activation functions ===

#[no_mangle]
pub extern "C" fn nsl_tensor_relu(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_elementwise_unary_inplace(tensor_ptr, crate::cuda::kernels::RELU_F32_PTX, "nsl_relu_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return tensor_ptr;
                }
                return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::RELU_F32_PTX, "nsl_relu_f32\0");
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned (CPU)
    {
        let t = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        if t.can_mutate_inplace() {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                for i in 0..len {
                    let v = unsafe { *d.add(i) };
                    if v < 0.0 { unsafe { *d.add(i) = 0.0 }; }
                }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len {
                    let v = unsafe { *d.add(i) };
                    if v < 0.0 { unsafe { *d.add(i) = 0.0 }; }
                }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return tensor_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f32().add(i) };
            unsafe { *buf.add(i) = if val > 0.0 { val } else { 0.0 } };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let val = unsafe { *a.data_f64().add(i) };
            unsafe { *buf.add(i) = if val > 0.0 { val } else { 0.0 } };
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
        NslTensor::from_ptr(tensor_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::ReLU {
            a: tensor_ptr, out: result, saved_a: tensor_ptr,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Relu, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_gelu(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_elementwise_unary_inplace(tensor_ptr, crate::cuda::kernels::GELU_F32_PTX, "nsl_gelu_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return tensor_ptr;
                }
                return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::GELU_F32_PTX, "nsl_gelu_f32\0");
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned
    {
        let t = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        if t.can_mutate_inplace() {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                let c = (2.0_f32 / std::f32::consts::PI).sqrt();
                for i in 0..len {
                    let x = unsafe { *d.add(i) };
                    let inner = c * (x + 0.044715_f32 * x * x * x);
                    unsafe { *d.add(i) = 0.5_f32 * x * (1.0_f32 + inner.tanh()) };
                }
            } else {
                let d = t.data as *mut f64;
                let c = (2.0_f64 / std::f64::consts::PI).sqrt();
                for i in 0..len {
                    let x = unsafe { *d.add(i) };
                    let inner = c * (x + 0.044715 * x * x * x);
                    unsafe { *d.add(i) = 0.5 * x * (1.0 + inner.tanh()) };
                }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return tensor_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        for i in 0..len as usize {
            let x = unsafe { *a.data_f32().add(i) };
            let inner = c * (x + 0.044715_f32 * x * x * x);
            unsafe { *buf.add(i) = 0.5_f32 * x * (1.0_f32 + inner.tanh()) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        let c = (2.0_f64 / std::f64::consts::PI).sqrt();
        for i in 0..len as usize {
            let x = unsafe { *a.data_f64().add(i) };
            let inner = c * (x + 0.044715 * x * x * x);
            unsafe { *buf.add(i) = 0.5 * x * (1.0 + inner.tanh()) };
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
        NslTensor::from_ptr(tensor_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::GELU {
            a: tensor_ptr, out: result, saved_a: tensor_ptr,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_silu(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_elementwise_unary_inplace(tensor_ptr, crate::cuda::kernels::SILU_F32_PTX, "nsl_silu_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return tensor_ptr;
                }
                return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::SILU_F32_PTX, "nsl_silu_f32\0");
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned
    {
        let t = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        if t.can_mutate_inplace() {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                for i in 0..len {
                    let x = unsafe { *d.add(i) };
                    let sig = 1.0_f32 / (1.0_f32 + (-x).exp());
                    unsafe { *d.add(i) = x * sig };
                }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len {
                    let x = unsafe { *d.add(i) };
                    let sig = 1.0 / (1.0 + (-x).exp());
                    unsafe { *d.add(i) = x * sig };
                }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return tensor_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let x = unsafe { *a.data_f32().add(i) };
            let sig = 1.0_f32 / (1.0_f32 + (-x).exp());
            unsafe { *buf.add(i) = x * sig };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let x = unsafe { *a.data_f64().add(i) };
            let sig = 1.0 / (1.0 + (-x).exp());
            unsafe { *buf.add(i) = x * sig };
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
        NslTensor::from_ptr(tensor_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::SiLU {
            a: tensor_ptr, out: result, saved_a: tensor_ptr,
        });
    }
    result
}

/// Source-AD SiLU backward, fused (Milestone C · p4 slice 2):
///   out = grad * silu'(x),  silu'(x) = σ(x) * (1 + x*(1 - σ(x))),  σ(x)=1/(1+e^-x)
///
/// One call replaces the six adjoint ops source-AD emits for `SiluBackward`
/// (Sigmoid, Sub, Mul, Add, Mul, Mul) — see `AdjointExpr::SiluBackward`. It is
/// BIT-EXACT with that decomposed path: the GPU kernel
/// (`SILU_BACKWARD_SRCAD_F32_PTX`) reproduces the exact operation ORDER and the
/// SIGMOID_F32_PTX instructions, with `.rn` on the derivative ops to block ptxas
/// fma-contraction; the CPU path computes the same order in f64/f32 (Rust emits
/// no FMA for separate `*`/`+`). `grad` and `x` are made contiguous first, as the
/// decomposed path's per-op kernels do.
#[no_mangle]
pub extern "C" fn nsl_tensor_silu_backward(grad_ptr: i64, x_ptr: i64) -> i64 {
    let xt = unsafe { &*(x_ptr as *const NslTensor) };
    let gt = unsafe { &*(grad_ptr as *const NslTensor) };
    // Only f64/f32 are handled (both the fused kernel and the decomposed
    // fallback branch f32-vs-f64). Reject f16/bf16 loudly rather than let a
    // `data_f64()`/`data_f32()` accessor over-read a narrower buffer.
    assert!(
        (xt.dtype == 0 || xt.dtype == 1) && (gt.dtype == 0 || gt.dtype == 1),
        "silu_backward: unsupported dtype (x={}, grad={}); f16/bf16 activation backward is not implemented",
        xt.dtype, gt.dtype
    );
    // Broadcast/mismatched-grad fallback: when the incoming adjoint does not
    // share `x`'s shape/device/dtype — e.g. a scalar sum/mean seed [1]
    // broadcasting against x[n], as in `sum(silu(x))` — reproduce the six-op
    // sequence the compiler emitted before fusion (whose final `nsl_tensor_mul`
    // broadcasts grad and whose sub/mul/add promote across dtype), numerically
    // equivalent to the pre-fusion path. The common mid-network case (grad
    // shape/device/dtype == x) takes the single fused launch below. Mirrors the
    // sigmoid/tanh backward fallbacks (p4 slice 3).
    if !gt.shape_eq(xt) || gt.device != xt.device || gt.dtype != xt.dtype {
        // Dtype-matched `1.0` so the chain stays single-dtype (no f64→f32
        // downcast of the gradient). CPU f64 → f64, everything else → f32.
        let one = crate::tensor::nsl_tensor_scalar(1.0, if xt.dtype == 0 { 0 } else { 1 });
        // sigmoid(x) without consuming x (bump so it can't take the FBIP path).
        NslTensor::from_ptr(x_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        let s = nsl_tensor_sigmoid(x_ptr);
        NslTensor::from_ptr(x_ptr).refcount.fetch_sub(1, Ordering::SeqCst);
        let t1 = crate::tensor::nsl_tensor_sub(one, s, 0); // 1 - s
        let t2 = crate::tensor::nsl_tensor_mul(x_ptr, t1, 0); // x*(1-s)
        let t3 = crate::tensor::nsl_tensor_add(one, t2, 0); // 1 + t2
        let t4 = crate::tensor::nsl_tensor_mul(s, t3, 0); // s*t3
        let out = crate::tensor::nsl_tensor_mul(grad_ptr, t4, 0); // grad*t4 (broadcast)
        for p in [one, s, t1, t2, t3, t4] { nsl_tensor_free(p); }
        return out;
    }
    let x_dev = xt.device;
    if x_dev > 0 {
        #[cfg(feature = "cuda")]
        {
            // Contiguous inputs (the decomposed Sigmoid/mul/etc kernels are
            // flat-indexed too). `nsl_tensor_contiguous` always returns an owned
            // ref, so both are freed unconditionally.
            let grad_c = nsl_tensor_contiguous(grad_ptr);
            let x_c = nsl_tensor_contiguous(x_ptr);
            let out = crate::cuda::gpu_backward_binary(
                grad_c,
                x_c,
                crate::cuda::kernels::SILU_BACKWARD_SRCAD_F32_PTX,
                "nsl_silu_backward_srcad_f32\0",
            );
            nsl_tensor_free(grad_c);
            nsl_tensor_free(x_c);
            return out;
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }
    // CPU path: same order as the decomposed sigmoid + sub/mul/add/mul/mul,
    // each a round-to-nearest op (no FMA), so byte-identical.
    let grad_c = nsl_tensor_contiguous(grad_ptr);
    let x_c = nsl_tensor_contiguous(x_ptr);
    let g = NslTensor::from_ptr(grad_c);
    let a = NslTensor::from_ptr(x_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);
    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let av = unsafe { *a.data_f32().add(i) };
            let gv = unsafe { *g.data_f32().add(i) };
            let s = 1.0_f32 / (1.0_f32 + (-av).exp());
            let t1 = 1.0_f32 - s;
            let t2 = av * t1;
            let t3 = 1.0_f32 + t2;
            let t4 = s * t3;
            unsafe { *buf.add(i) = gv * t4 };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let av = unsafe { *a.data_f64().add(i) };
            let gv = unsafe { *g.data_f64().add(i) };
            let s = 1.0 / (1.0 + (-av).exp());
            let t1 = 1.0 - s;
            let t2 = av * t1;
            let t3 = 1.0 + t2;
            let t4 = s * t3;
            unsafe { *buf.add(i) = gv * t4 };
        }
        buf as *mut c_void
    };
    let result = Box::new(NslTensor::new(
        data, shape, strides, ndim, len, a.device, a.dtype, 1, 0,
    ));
    let result = NslTensor::publish(result);
    nsl_tensor_free(grad_c);
    nsl_tensor_free(x_c);
    result
}

/// Source-AD SIGMOID backward, fused (Milestone C · p4 slice 3):
///   out = grad * σ'(x),  σ'(x) = y*(1 - y),  where y = σ(x) is the saved output.
///
/// One call replaces the three adjoint ops source-AD emits for `SigmoidBackward`
/// (Sub, Mul, Mul) — see `AdjointExpr::SigmoidBackward`, whose second operand is
/// the sigmoid OUTPUT `y`. It is BIT-EXACT with that decomposed path: the GPU
/// kernel (`SIGMOID_BACKWARD_SRCAD_F32_PTX`) reproduces the exact operation
/// ORDER with `.rn` on every op; the CPU path computes the same order in f64/f32
/// (Rust emits no FMA for separate `*`/`-`). `grad` and `y` are made contiguous
/// first, as the decomposed path's per-op kernels do.
#[no_mangle]
pub extern "C" fn nsl_tensor_sigmoid_backward(grad_ptr: i64, y_ptr: i64) -> i64 {
    let yt = unsafe { &*(y_ptr as *const NslTensor) };
    let gt = unsafe { &*(grad_ptr as *const NslTensor) };
    assert!(
        (yt.dtype == 0 || yt.dtype == 1) && (gt.dtype == 0 || gt.dtype == 1),
        "sigmoid_backward: unsupported dtype (y={}, grad={}); f16/bf16 activation backward is not implemented",
        yt.dtype, gt.dtype
    );
    // Broadcast/mismatched-grad fallback (scalar sum/mean seed vs y[n],
    // cross-device, or mixed dtype): reproduce the three-op sequence the compiler
    // emitted pre-fusion (dtype-matched `1.0` constant), whose final `nsl_tensor_mul`
    // broadcasts grad and whose sub/mul promote across dtype — numerically
    // equivalent to the pre-fusion path. The dtype guard also keeps the fast
    // path's flat, single-dtype CPU loop from reading `grad` with `y`'s accessor.
    if !gt.shape_eq(yt) || gt.device != yt.device || gt.dtype != yt.dtype {
        let one = crate::tensor::nsl_tensor_scalar(1.0, if yt.dtype == 0 { 0 } else { 1 });
        let t1 = crate::tensor::nsl_tensor_sub(one, y_ptr, 0); // 1 - y
        let t2 = crate::tensor::nsl_tensor_mul(y_ptr, t1, 0); // y*(1-y)
        let out = crate::tensor::nsl_tensor_mul(grad_ptr, t2, 0); // grad*t2 (broadcast)
        for p in [one, t1, t2] { nsl_tensor_free(p); }
        return out;
    }
    if yt.device > 0 {
        #[cfg(feature = "cuda")]
        {
            let grad_c = nsl_tensor_contiguous(grad_ptr);
            let y_c = nsl_tensor_contiguous(y_ptr);
            let out = crate::cuda::gpu_backward_binary(
                grad_c,
                y_c,
                crate::cuda::kernels::SIGMOID_BACKWARD_SRCAD_F32_PTX,
                "nsl_sigmoid_backward_srcad_f32\0",
            );
            nsl_tensor_free(grad_c);
            nsl_tensor_free(y_c);
            return out;
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }
    // CPU path: same order as the decomposed sub/mul/mul, each round-to-nearest.
    let grad_c = nsl_tensor_contiguous(grad_ptr);
    let y_c = nsl_tensor_contiguous(y_ptr);
    let g = NslTensor::from_ptr(grad_c);
    let a = NslTensor::from_ptr(y_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);
    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let yv = unsafe { *a.data_f32().add(i) };
            let gv = unsafe { *g.data_f32().add(i) };
            let t1 = 1.0_f32 - yv;
            let t2 = yv * t1;
            unsafe { *buf.add(i) = gv * t2 };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let yv = unsafe { *a.data_f64().add(i) };
            let gv = unsafe { *g.data_f64().add(i) };
            let t1 = 1.0 - yv;
            let t2 = yv * t1;
            unsafe { *buf.add(i) = gv * t2 };
        }
        buf as *mut c_void
    };
    let result = Box::new(NslTensor::new(
        data, shape, strides, ndim, len, a.device, a.dtype, 1, 0,
    ));
    let result = NslTensor::publish(result);
    nsl_tensor_free(grad_c);
    nsl_tensor_free(y_c);
    result
}

/// Source-AD TANH backward, fused (Milestone C · p4 slice 3):
///   out = grad * tanh'(x),  tanh'(x) = 1 - y*y,  where y = tanh(x) is the output.
///
/// One call replaces the three adjoint ops source-AD emits for `TanhBackward`
/// (Mul, Sub, Mul) — see `AdjointExpr::TanhBackward`, whose second operand is the
/// tanh OUTPUT `y`. It is BIT-EXACT with that decomposed path: the GPU kernel
/// (`TANH_BACKWARD_SRCAD_F32_PTX`) reproduces the exact operation ORDER with
/// LOAD-BEARING `.rn` blocking the `y*y`→`1-y*y` fma-contraction; the CPU path
/// computes the same order in f64/f32. `grad` and `y` are made contiguous first.
#[no_mangle]
pub extern "C" fn nsl_tensor_tanh_backward(grad_ptr: i64, y_ptr: i64) -> i64 {
    let yt = unsafe { &*(y_ptr as *const NslTensor) };
    let gt = unsafe { &*(grad_ptr as *const NslTensor) };
    assert!(
        (yt.dtype == 0 || yt.dtype == 1) && (gt.dtype == 0 || gt.dtype == 1),
        "tanh_backward: unsupported dtype (y={}, grad={}); f16/bf16 activation backward is not implemented",
        yt.dtype, gt.dtype
    );
    // Broadcast/mismatched-grad fallback (scalar sum/mean seed vs y[n],
    // cross-device, or mixed dtype): reproduce the three-op sequence the compiler
    // emitted pre-fusion (dtype-matched `1.0` constant), whose final `nsl_tensor_mul`
    // broadcasts grad and whose sub/mul promote across dtype — numerically
    // equivalent to the pre-fusion path. The dtype guard also keeps the fast
    // path's flat, single-dtype CPU loop from reading `grad` with `y`'s accessor.
    if !gt.shape_eq(yt) || gt.device != yt.device || gt.dtype != yt.dtype {
        let one = crate::tensor::nsl_tensor_scalar(1.0, if yt.dtype == 0 { 0 } else { 1 });
        let y_sq = crate::tensor::nsl_tensor_mul(y_ptr, y_ptr, 0); // y*y
        let t = crate::tensor::nsl_tensor_sub(one, y_sq, 0); // 1 - y*y
        let out = crate::tensor::nsl_tensor_mul(grad_ptr, t, 0); // grad*t (broadcast)
        for p in [one, y_sq, t] { nsl_tensor_free(p); }
        return out;
    }
    if yt.device > 0 {
        #[cfg(feature = "cuda")]
        {
            let grad_c = nsl_tensor_contiguous(grad_ptr);
            let y_c = nsl_tensor_contiguous(y_ptr);
            let out = crate::cuda::gpu_backward_binary(
                grad_c,
                y_c,
                crate::cuda::kernels::TANH_BACKWARD_SRCAD_F32_PTX,
                "nsl_tanh_backward_srcad_f32\0",
            );
            nsl_tensor_free(grad_c);
            nsl_tensor_free(y_c);
            return out;
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }
    // CPU path: same order as the decomposed mul/sub/mul, each round-to-nearest.
    let grad_c = nsl_tensor_contiguous(grad_ptr);
    let y_c = nsl_tensor_contiguous(y_ptr);
    let g = NslTensor::from_ptr(grad_c);
    let a = NslTensor::from_ptr(y_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);
    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let yv = unsafe { *a.data_f32().add(i) };
            let gv = unsafe { *g.data_f32().add(i) };
            let t1 = yv * yv;
            let t2 = 1.0_f32 - t1;
            unsafe { *buf.add(i) = gv * t2 };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let yv = unsafe { *a.data_f64().add(i) };
            let gv = unsafe { *g.data_f64().add(i) };
            let t1 = yv * yv;
            let t2 = 1.0 - t1;
            unsafe { *buf.add(i) = gv * t2 };
        }
        buf as *mut c_void
    };
    let result = Box::new(NslTensor::new(
        data, shape, strides, ndim, len, a.device, a.dtype, 1, 0,
    ));
    let result = NslTensor::publish(result);
    nsl_tensor_free(grad_c);
    nsl_tensor_free(y_c);
    result
}

/// Derivative of the CPU forward gelu's TANH approximation
/// (`0.5·x·(1+tanh(k))`, `k = √(2/π)·(x + 0.044715x³)`):
///   gelu'(x) = 0.5·(1+tanh(k)) + 0.5·x·sech²(k)·√(2/π)·(1 + 3·0.044715·x²)
#[inline]
fn gelu_tanh_deriv_f32(x: f32) -> f32 {
    let c = (2.0_f32 / std::f32::consts::PI).sqrt();
    let k = c * (x + 0.044715_f32 * x * x * x);
    let t = k.tanh();
    let sech2 = 1.0_f32 - t * t;
    0.5_f32 * (1.0_f32 + t) + 0.5_f32 * x * sech2 * c * (1.0_f32 + 3.0_f32 * 0.044715_f32 * x * x)
}
#[inline]
fn gelu_tanh_deriv_f64(x: f64) -> f64 {
    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
    let k = c * (x + 0.044715 * x * x * x);
    let t = k.tanh();
    let sech2 = 1.0 - t * t;
    0.5 * (1.0 + t) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x * x)
}

/// Materialize `gelu'(x)` (CPU tanh-approx derivative) as a fresh tensor of
/// `x`'s shape/dtype. Used by the broadcast-grad fallback.
fn gelu_deriv_cpu(x_ptr: i64) -> i64 {
    let x_c = nsl_tensor_contiguous(x_ptr);
    let a = NslTensor::from_ptr(x_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);
    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = gelu_tanh_deriv_f32(*a.data_f32().add(i)) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = gelu_tanh_deriv_f64(*a.data_f64().add(i)) };
        }
        buf as *mut c_void
    };
    let result = Box::new(NslTensor::new(
        data, shape, strides, ndim, len, a.device, a.dtype, 1, 0,
    ));
    let result = NslTensor::publish(result);
    nsl_tensor_free(x_c);
    result
}

/// Source-AD GELU backward, fused (Milestone C · p4 GELU fix): one call computing
/// `grad * gelu'(x)`, where `gelu'` is the derivative of the forward THIS DEVICE
/// actually ran:
/// - **GPU**: sigmoid approximation `x·σ(1.702x)` (`GELU_F32_PTX`) →
///   `GELU_BACKWARD_SRCAD_F32_PTX` computes `σ(1.702x)·(1+1.702x·(1−σ(1.702x)))`.
/// - **CPU**: tanh approximation (the `nsl_tensor_gelu` CPU loop) →
///   `gelu_tanh_deriv_*` above (same formula as the tape backward).
///
/// Replaces the source-AD 7-op expansion of `AdjointExpr::GeluBackward`, which
/// was numerically WRONG: its internal temp `kx = 1.702·x` (refcount 1) was
/// FBIP-mutated in place by the expansion's own `Sigmoid(kx)` during the adjoint
/// pass (guard deliberately clear there), so the later `Mul(kx, 1−s)` read
/// `σ(kx)` and the whole expression became `s·(1+s·(1−s))` — 0.625 instead of
/// 0.5 at x=0. Fusing eliminates the temp; no aliasing exists inside one kernel.
#[no_mangle]
pub extern "C" fn nsl_tensor_gelu_backward(grad_ptr: i64, x_ptr: i64) -> i64 {
    let xt = unsafe { &*(x_ptr as *const NslTensor) };
    let gt = unsafe { &*(grad_ptr as *const NslTensor) };
    assert!(
        (xt.dtype == 0 || xt.dtype == 1) && (gt.dtype == 0 || gt.dtype == 1),
        "gelu_backward: unsupported dtype (x={}, grad={}); f16/bf16 activation backward is not implemented",
        xt.dtype, gt.dtype
    );
    // Broadcast/mismatched-grad fallback (scalar sum/mean seed vs x[n],
    // cross-device, mixed dtype): materialize `deriv = gelu'(x)` as a fresh
    // tensor, then `grad * deriv` via the broadcasting `nsl_tensor_mul` (grad
    // first, matching the pre-fusion final-op order so the output follows
    // grad's placement).
    if !gt.shape_eq(xt) || gt.device != xt.device || gt.dtype != xt.dtype {
        let deriv = if xt.device > 0 {
            #[cfg(feature = "cuda")]
            {
                // GPU tensors are f32, so ones_like yields a matching GPU ones;
                // deriv = fused kernel evaluated with grad = 1.
                let ones = crate::tensor::nsl_tensor_ones_like(x_ptr);
                let x_c = nsl_tensor_contiguous(x_ptr);
                let d = crate::cuda::gpu_backward_binary(
                    ones,
                    x_c,
                    crate::cuda::kernels::GELU_BACKWARD_SRCAD_F32_PTX,
                    "nsl_gelu_backward_srcad_f32\0",
                );
                nsl_tensor_free(ones);
                nsl_tensor_free(x_c);
                d
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        } else {
            gelu_deriv_cpu(x_ptr)
        };
        let out = crate::tensor::nsl_tensor_mul(grad_ptr, deriv, 0);
        nsl_tensor_free(deriv);
        return out;
    }
    if xt.device > 0 {
        #[cfg(feature = "cuda")]
        {
            let grad_c = nsl_tensor_contiguous(grad_ptr);
            let x_c = nsl_tensor_contiguous(x_ptr);
            let out = crate::cuda::gpu_backward_binary(
                grad_c,
                x_c,
                crate::cuda::kernels::GELU_BACKWARD_SRCAD_F32_PTX,
                "nsl_gelu_backward_srcad_f32\0",
            );
            nsl_tensor_free(grad_c);
            nsl_tensor_free(x_c);
            return out;
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }
    // CPU: tanh-approx derivative (the CPU forward's formula), elementwise.
    let grad_c = nsl_tensor_contiguous(grad_ptr);
    let x_c = nsl_tensor_contiguous(x_ptr);
    let g = NslTensor::from_ptr(grad_c);
    let a = NslTensor::from_ptr(x_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);
    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let xv = unsafe { *a.data_f32().add(i) };
            let gv = unsafe { *g.data_f32().add(i) };
            unsafe { *buf.add(i) = gv * gelu_tanh_deriv_f32(xv) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let xv = unsafe { *a.data_f64().add(i) };
            let gv = unsafe { *g.data_f64().add(i) };
            unsafe { *buf.add(i) = gv * gelu_tanh_deriv_f64(xv) };
        }
        buf as *mut c_void
    };
    let result = Box::new(NslTensor::new(
        data, shape, strides, ndim, len, a.device, a.dtype, 1, 0,
    ));
    let result = NslTensor::publish(result);
    nsl_tensor_free(grad_c);
    nsl_tensor_free(x_c);
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sigmoid(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_elementwise_unary_inplace(tensor_ptr, crate::cuda::kernels::SIGMOID_F32_PTX, "nsl_sigmoid_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return tensor_ptr;
                }
                return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::SIGMOID_F32_PTX, "nsl_sigmoid_f32\0");
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned (CPU)
    {
        let t = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        if t.can_mutate_inplace() {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                for i in 0..len {
                    let x = unsafe { *d.add(i) };
                    unsafe { *d.add(i) = 1.0_f32 / (1.0_f32 + (-x).exp()) };
                }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len {
                    let x = unsafe { *d.add(i) };
                    unsafe { *d.add(i) = 1.0 / (1.0 + (-x).exp()) };
                }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return tensor_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let x = unsafe { *a.data_f32().add(i) };
            unsafe { *buf.add(i) = 1.0_f32 / (1.0_f32 + (-x).exp()) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let x = unsafe { *a.data_f64().add(i) };
            unsafe { *buf.add(i) = 1.0 / (1.0 + (-x).exp()) };
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
        autodiff::maybe_record(autodiff::TapeOp::Sigmoid {
            a: tensor_ptr, out: result, saved_out: result,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Sigmoid, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_tanh_act(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_elementwise_unary_inplace(tensor_ptr, crate::cuda::kernels::TANH_F32_PTX, "nsl_tanh_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return tensor_ptr;
                }
                return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::TANH_F32_PTX, "nsl_tanh_f32\0");
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned (CPU)
    {
        let t = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        if t.can_mutate_inplace() {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                for i in 0..len { unsafe { *d.add(i) = (*d.add(i)).tanh() }; }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len { unsafe { *d.add(i) = (*d.add(i)).tanh() }; }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return tensor_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            let x = unsafe { *a.data_f32().add(i) };
            unsafe { *buf.add(i) = x.tanh() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            let x = unsafe { *a.data_f64().add(i) };
            unsafe { *buf.add(i) = x.tanh() };
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
        autodiff::maybe_record(autodiff::TapeOp::Tanh {
            a: tensor_ptr, out: result, saved_out: result,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Tanh, vec![tensor_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

// === FBIP Phase 2: Unconditional in-place variants ===
// These skip the refcount check — the compiler guarantees the tensor is uniquely
// owned via use-count analysis. No allocation, no refcount bump, no branch.

macro_rules! define_inplace_unary {
    ($name:ident, $op_f32:expr, $op_f64:expr, $ptx:expr, $kernel_name:literal) => {
        #[no_mangle]
        pub extern "C" fn $name(ptr: i64) -> i64 {
            let t = unsafe { &mut *(ptr as *mut NslTensor) };
            // GPU path: dispatch to GPU kernel (device memory not CPU-accessible)
            if t.device > 0 {
                #[cfg(feature = "cuda")]
                {
                    crate::cuda::gpu_elementwise_unary_inplace(ptr, $ptx, $kernel_name);
                    super::fbip_record_reuse();
                    return ptr;
                }
                #[cfg(not(feature = "cuda"))]
                { panic!("CUDA support not compiled"); }
            }
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                for i in 0..len {
                    unsafe {
                        let v = *d.add(i);
                        *d.add(i) = $op_f32(v);
                    }
                }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len {
                    unsafe {
                        let v = *d.add(i);
                        *d.add(i) = $op_f64(v);
                    }
                }
            }
            super::fbip_record_reuse();
            ptr
        }
    };
}

define_inplace_unary!(nsl_tensor_relu_inplace, |v: f32| if v > 0.0 { v } else { 0.0_f32 }, |v: f64| if v > 0.0 { v } else { 0.0 }, crate::cuda::kernels::RELU_F32_PTX, "nsl_relu_f32\0");
define_inplace_unary!(nsl_tensor_exp_inplace, |v: f32| v.exp(), |v: f64| v.exp(), crate::cuda::kernels::EXP_F32_PTX, "nsl_exp_f32\0");
define_inplace_unary!(nsl_tensor_log_inplace, |v: f32| v.ln(), |v: f64| v.ln(), crate::cuda::kernels::LOG_F32_PTX, "nsl_log_f32\0");
define_inplace_unary!(nsl_tensor_sqrt_inplace, |v: f32| v.sqrt(), |v: f64| v.sqrt(), crate::cuda::kernels::SQRT_F32_PTX, "nsl_sqrt_f32\0");
define_inplace_unary!(nsl_tensor_abs_inplace, |v: f32| v.abs(), |v: f64| v.abs(), crate::cuda::kernels::ABS_F32_PTX, "nsl_abs_f32\0");
define_inplace_unary!(nsl_tensor_sigmoid_inplace, |v: f32| 1.0_f32 / (1.0_f32 + (-v).exp()), |v: f64| 1.0 / (1.0 + (-v).exp()), crate::cuda::kernels::SIGMOID_F32_PTX, "nsl_sigmoid_f32\0");
define_inplace_unary!(nsl_tensor_tanh_inplace, |v: f32| v.tanh(), |v: f64| v.tanh(), crate::cuda::kernels::TANH_F32_PTX, "nsl_tanh_f32\0");
define_inplace_unary!(nsl_tensor_neg_inplace, |v: f32| -v, |v: f64| -v, crate::cuda::kernels::NEG_F32_PTX, "nsl_neg_f32\0");
define_inplace_unary!(nsl_tensor_sign_inplace, |v: f32| if v > 0.0 { 1.0_f32 } else if v < 0.0 { -1.0_f32 } else { 0.0_f32 }, |v: f64| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 }, crate::cuda::kernels::SIGN_F32_PTX, "nsl_sign_f32\0");

/// GELU in-place.
#[no_mangle]
pub extern "C" fn nsl_tensor_gelu_inplace(ptr: i64) -> i64 {
    let t = unsafe { &mut *(ptr as *mut NslTensor) };
    // GPU path: dispatch to GPU kernel
    if t.device > 0 {
        #[cfg(feature = "cuda")]
        {
            crate::cuda::gpu_elementwise_unary_inplace(ptr, crate::cuda::kernels::GELU_F32_PTX, "nsl_gelu_f32\0");
            super::fbip_record_reuse();
            return ptr;
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }
    let len = t.len as usize;
    if t.dtype == 1 {
        let d = t.data as *mut f32;
        let c = (2.0_f32 / std::f32::consts::PI).sqrt();
        for i in 0..len {
            let x = unsafe { *d.add(i) };
            let inner = c * (x + 0.044715_f32 * x * x * x);
            unsafe { *d.add(i) = 0.5_f32 * x * (1.0_f32 + inner.tanh()) };
        }
    } else {
        let d = t.data as *mut f64;
        let c = (2.0_f64 / std::f64::consts::PI).sqrt();
        for i in 0..len {
            let x = unsafe { *d.add(i) };
            let inner = c * (x + 0.044715 * x * x * x);
            unsafe { *d.add(i) = 0.5 * x * (1.0 + inner.tanh()) };
        }
    }
    super::fbip_record_reuse();
    ptr
}

/// SiLU in-place.
#[no_mangle]
pub extern "C" fn nsl_tensor_silu_inplace(ptr: i64) -> i64 {
    let t = unsafe { &mut *(ptr as *mut NslTensor) };
    // GPU path: dispatch to GPU kernel
    if t.device > 0 {
        #[cfg(feature = "cuda")]
        {
            crate::cuda::gpu_elementwise_unary_inplace(ptr, crate::cuda::kernels::SILU_F32_PTX, "nsl_silu_f32\0");
            super::fbip_record_reuse();
            return ptr;
        }
        #[cfg(not(feature = "cuda"))]
        { panic!("CUDA support not compiled"); }
    }
    let len = t.len as usize;
    if t.dtype == 1 {
        let d = t.data as *mut f32;
        for i in 0..len {
            let x = unsafe { *d.add(i) };
            let sig = 1.0_f32 / (1.0_f32 + (-x).exp());
            unsafe { *d.add(i) = x * sig };
        }
    } else {
        let d = t.data as *mut f64;
        for i in 0..len {
            let x = unsafe { *d.add(i) };
            let sig = 1.0 / (1.0 + (-x).exp());
            unsafe { *d.add(i) = x * sig };
        }
    }
    super::fbip_record_reuse();
    ptr
}

// ---------------------------------------------------------------------------
// Static reuse: unconditional in-place binary variants (Phase 2, Item 3)
//
// The 11 unary inplace variants were added in Phase 1 (lines 921-1008 above).
// These binary variants are new — they skip all runtime checks and operate
// directly on the left operand's buffer. Only safe when M38a proves left is
// uniquely owned and shapes match.
// ---------------------------------------------------------------------------

/// Unconditional in-place add: left += right.
#[no_mangle]
pub extern "C" fn nsl_tensor_add_inplace_fbip(left_ptr: i64, right_ptr: i64) -> i64 {
    let left = unsafe { &*(left_ptr as *const NslTensor) };
    let right = unsafe { &*(right_ptr as *const NslTensor) };
    let len = left.len.min(right.len) as usize;
    if left.dtype == 1 && right.dtype == 1 {
        let ld = left.data as *mut f32;
        let rd = right.data as *const f32;
        for i in 0..len { unsafe { *ld.add(i) += *rd.add(i) }; }
    } else {
        let ld = left.data as *mut f64;
        let rd = right.data as *const f64;
        for i in 0..len { unsafe { *ld.add(i) += *rd.add(i) }; }
    }
    super::fbip_record_reuse();
    left_ptr
}

/// Unconditional in-place sub: left -= right.
#[no_mangle]
pub extern "C" fn nsl_tensor_sub_inplace_fbip(left_ptr: i64, right_ptr: i64) -> i64 {
    let left = unsafe { &*(left_ptr as *const NslTensor) };
    let right = unsafe { &*(right_ptr as *const NslTensor) };
    let len = left.len.min(right.len) as usize;
    if left.dtype == 1 && right.dtype == 1 {
        let ld = left.data as *mut f32;
        let rd = right.data as *const f32;
        for i in 0..len { unsafe { *ld.add(i) -= *rd.add(i) }; }
    } else {
        let ld = left.data as *mut f64;
        let rd = right.data as *const f64;
        for i in 0..len { unsafe { *ld.add(i) -= *rd.add(i) }; }
    }
    super::fbip_record_reuse();
    left_ptr
}

/// Unconditional in-place mul: left *= right.
#[no_mangle]
pub extern "C" fn nsl_tensor_mul_inplace_fbip(left_ptr: i64, right_ptr: i64) -> i64 {
    let left = unsafe { &*(left_ptr as *const NslTensor) };
    let right = unsafe { &*(right_ptr as *const NslTensor) };
    let len = left.len.min(right.len) as usize;
    if left.dtype == 1 && right.dtype == 1 {
        let ld = left.data as *mut f32;
        let rd = right.data as *const f32;
        for i in 0..len { unsafe { *ld.add(i) *= *rd.add(i) }; }
    } else {
        let ld = left.data as *mut f64;
        let rd = right.data as *const f64;
        for i in 0..len { unsafe { *ld.add(i) *= *rd.add(i) }; }
    }
    super::fbip_record_reuse();
    left_ptr
}

/// Unconditional in-place div: left /= right.
#[no_mangle]
pub extern "C" fn nsl_tensor_div_inplace_fbip(left_ptr: i64, right_ptr: i64) -> i64 {
    let left = unsafe { &*(left_ptr as *const NslTensor) };
    let right = unsafe { &*(right_ptr as *const NslTensor) };
    let len = left.len.min(right.len) as usize;
    if left.dtype == 1 && right.dtype == 1 {
        let ld = left.data as *mut f32;
        let rd = right.data as *const f32;
        for i in 0..len { unsafe { *ld.add(i) /= *rd.add(i) }; }
    } else {
        let ld = left.data as *mut f64;
        let rd = right.data as *const f64;
        for i in 0..len { unsafe { *ld.add(i) /= *rd.add(i) }; }
    }
    super::fbip_record_reuse();
    left_ptr
}

#[cfg(test)]
mod silu_backward_tests {
    use super::*;
    use crate::tensor::NslTensor;
    use std::sync::atomic::Ordering;

    fn make_f32(data: &[f32]) -> i64 {
        let ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&[data.len() as i64], 1);
        let t = NslTensor::from_ptr(ptr);
        for (i, v) in data.iter().enumerate() { unsafe { *t.data_f32().add(i) = *v }; }
        ptr
    }
    fn make_f64(data: &[f64]) -> i64 {
        let ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&[data.len() as i64], 0);
        let t = NslTensor::from_ptr(ptr);
        for (i, v) in data.iter().enumerate() { unsafe { *t.data_f64().add(i) = *v }; }
        ptr
    }

    /// The decomposed source-AD `SiluBackward`: Sigmoid, Sub, Mul, Add, Mul, Mul
    /// (flags=0 everywhere so `grad` and `x` are preserved). Returns a new tensor.
    /// This is exactly what `nsl_tensor_silu_backward` must reproduce bit-for-bit.
    fn silu_bwd_6op(grad: i64, x: i64) -> i64 {
        // s = sigmoid(x), without mutating x (bump so it can't take the FBIP path).
        NslTensor::from_ptr(x).refcount.fetch_add(1, Ordering::SeqCst);
        let s = nsl_tensor_sigmoid(x);
        NslTensor::from_ptr(x).refcount.fetch_sub(1, Ordering::SeqCst);
        // Same-dtype ones (nsl_tensor_ones_like returns f32 even for an f64
        // input on CPU, which would force a dtype reconcile in the reference).
        let xt = NslTensor::from_ptr(x);
        let one = if xt.device == 0 && xt.dtype == 0 {
            make_f64(&vec![1.0f64; xt.len as usize])
        } else {
            crate::tensor::nsl_tensor_ones_like(x)
        };
        let t1 = crate::tensor::nsl_tensor_sub(one, s, 0);    // 1 - s
        let t2 = crate::tensor::nsl_tensor_mul(x, t1, 0);     // x * (1 - s)
        let t3 = crate::tensor::nsl_tensor_add(one, t2, 0);   // 1 + t2
        let t4 = crate::tensor::nsl_tensor_mul(s, t3, 0);     // s * t3
        let out = crate::tensor::nsl_tensor_mul(grad, t4, 0); // grad * t4
        for p in [s, one, t1, t2, t3, t4] { nsl_tensor_free(p); }
        out
    }

    #[test]
    fn silu_backward_cpu_f64_matches_6op() {
        let xs = [-3.0, -0.5, 0.0, 0.5, 2.0, 7.0, -1.25, 12.0, -9.0];
        let gs = [0.3, -1.1, 2.0, 0.75, -0.5, 1.0, 4.2, -0.01, 3.3];
        let (xr, gr) = (make_f64(&xs), make_f64(&gs));
        let ref_out = silu_bwd_6op(gr, xr);
        let (xf, gf) = (make_f64(&xs), make_f64(&gs));
        let fused = nsl_tensor_silu_backward(gf, xf);
        let (ro, fo) = (NslTensor::from_ptr(ref_out), NslTensor::from_ptr(fused));
        for i in 0..xs.len() {
            let (r, f) = unsafe { (*ro.data_f64().add(i), *fo.data_f64().add(i)) };
            assert_eq!(r.to_bits(), f.to_bits(), "cpu f64 mismatch at {i}: {r} vs {f}");
        }
        for p in [xr, gr, ref_out, xf, gf, fused] { nsl_tensor_free(p); }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn silu_backward_gpu_f32_matches_6op() {
        let n = 257usize; // not a multiple of 256 — exercises the block tail guard
        let xs: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 6.0).collect(); // ~[-6, 6.8]
        let gs: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 2.0).collect();
        let to_gpu = |d: &[f32]| crate::tensor::nsl_tensor_to_device(make_f32(d), 1);
        let (xr, gr) = (to_gpu(&xs), to_gpu(&gs));
        let ref_out = silu_bwd_6op(gr, xr);
        let (xf, gf) = (to_gpu(&xs), to_gpu(&gs));
        let fused = nsl_tensor_silu_backward(gf, xf);
        // GPU f32 -> CPU f64 upcast is lossless, so equal upcast == equal f32 bits.
        let up = |p: i64| -> Vec<f64> {
            let c = crate::tensor::nsl_tensor_to_device(p, 0);
            let t = NslTensor::from_ptr(c);
            (0..n).map(|i| unsafe { *t.data_f64().add(i) }).collect()
        };
        let (r, f) = (up(ref_out), up(fused));
        for i in 0..n {
            assert_eq!(r[i].to_bits(), f[i].to_bits(), "gpu f32 mismatch at {i}: {} vs {}", r[i], f[i]);
        }
        for p in [xr, gr, ref_out, xf, gf, fused] { nsl_tensor_free(p); }
    }

    /// A scalar `grad` (a `sum`/`mean` seed) must broadcast against x[n] via the
    /// decomposed fallback rather than panic (the crash reverted in slice 3,
    /// restored in slice 4). Output is x-shaped and each element equals
    /// grad0 * silu'(x[i]).
    #[test]
    fn silu_backward_scalar_grad_broadcasts() {
        let xs = [-2.0, -0.5, 0.0, 1.0, 3.0];
        let c = 1.5_f64;
        let (x, grad) = (make_f64(&xs), make_f64(&[c]));
        let out = nsl_tensor_silu_backward(grad, x);
        assert_eq!(NslTensor::from_ptr(out).len, xs.len() as i64, "output must be x-shaped");
        let ot = NslTensor::from_ptr(out);
        for (i, &xv) in xs.iter().enumerate() {
            let s = 1.0 / (1.0 + (-xv).exp());
            let expected = c * s * (1.0 + xv * (1.0 - s));
            let got = unsafe { *ot.data_f64().add(i) };
            assert!((got - expected).abs() < 1e-5, "at {i}: {got} vs {expected}");
        }
        for p in [x, grad, out] { nsl_tensor_free(p); }
    }

    /// The source-AD in-place-suppression guard: with it raised, a forward
    /// activation on a uniquely-owned input must allocate a fresh output and
    /// leave the input intact (so the adjoint's saved primal survives) — exactly
    /// what tape-AD gets for free from `is_recording()`. With it clear, the
    /// unique-owner FBIP path mutates in place (returns the same pointer).
    #[test]
    fn inplace_suppress_preserves_forward_input() {
        use crate::tensor::{inplace_suppressed, nsl_set_inplace_suppressed};
        let xs = [-1.0f32, 0.0, 1.0, 2.0];

        // Suppressed: silu must NOT reuse x; x stays == its original values.
        let x = make_f32(&xs);
        assert!(!inplace_suppressed());
        nsl_set_inplace_suppressed(1);
        assert!(inplace_suppressed());
        let y = nsl_tensor_silu(x);
        nsl_set_inplace_suppressed(0);
        assert!(!inplace_suppressed(), "guard must pop back to clear");
        assert_ne!(x, y, "suppressed: silu must allocate a fresh output, not reuse x");
        let xt = NslTensor::from_ptr(x);
        for (i, &v) in xs.iter().enumerate() {
            assert_eq!(unsafe { *xt.data_f32().add(i) }, v, "x mutated at {i} despite suppression");
        }
        // y really is silu(x).
        let yt = NslTensor::from_ptr(y);
        for (i, &v) in xs.iter().enumerate() {
            let want = v * (1.0_f32 / (1.0_f32 + (-v).exp()));
            assert!((unsafe { *yt.data_f32().add(i) } - want).abs() < 1e-6);
        }
        for p in [x, y] { nsl_tensor_free(p); }

        // Not suppressed + uniquely owned: FBIP mutates in place (same pointer).
        let x2 = make_f32(&xs);
        let y2 = nsl_tensor_silu(x2);
        assert_eq!(x2, y2, "unsuppressed unique-owner silu should reuse the buffer in place");
        // x2 == y2 is ONE allocation whose refcount FBIP bumped to 2 (two logical
        // owners: the input handle and the result handle) — free both to reach 0.
        nsl_tensor_free(x2);
        nsl_tensor_free(y2);
    }
}

#[cfg(test)]
mod sigmoid_tanh_backward_tests {
    use super::*;
    use crate::tensor::NslTensor;

    // Only the cuda-gated GPU tests build f32 inputs on the host.
    #[cfg(feature = "cuda")]
    fn make_f32(data: &[f32]) -> i64 {
        let ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&[data.len() as i64], 1);
        let t = NslTensor::from_ptr(ptr);
        for (i, v) in data.iter().enumerate() { unsafe { *t.data_f32().add(i) = *v }; }
        ptr
    }
    fn make_f64(data: &[f64]) -> i64 {
        let ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&[data.len() as i64], 0);
        let t = NslTensor::from_ptr(ptr);
        for (i, v) in data.iter().enumerate() { unsafe { *t.data_f64().add(i) = *v }; }
        ptr
    }
    // Same-dtype ones (nsl_tensor_ones_like returns f32 even for an f64 input on
    // CPU, which would force a dtype reconcile the compiler never performs).
    fn ones_like_same_dtype(y: i64) -> i64 {
        let yt = NslTensor::from_ptr(y);
        if yt.device == 0 && yt.dtype == 0 {
            make_f64(&vec![1.0f64; yt.len as usize])
        } else {
            crate::tensor::nsl_tensor_ones_like(y)
        }
    }

    /// Decomposed source-AD `SigmoidBackward`: Sub, Mul, Mul (flags=0 so `grad`
    /// and `y` are preserved). What `nsl_tensor_sigmoid_backward` must match.
    fn sigmoid_bwd_3op(grad: i64, y: i64) -> i64 {
        let one = ones_like_same_dtype(y);
        let t1 = crate::tensor::nsl_tensor_sub(one, y, 0);   // 1 - y
        let t2 = crate::tensor::nsl_tensor_mul(y, t1, 0);    // y * (1 - y)
        let out = crate::tensor::nsl_tensor_mul(grad, t2, 0); // grad * t2
        for p in [one, t1, t2] { nsl_tensor_free(p); }
        out
    }

    /// Decomposed source-AD `TanhBackward`: Mul, Sub, Mul.
    fn tanh_bwd_3op(grad: i64, y: i64) -> i64 {
        let y_sq = crate::tensor::nsl_tensor_mul(y, y, 0);   // y * y
        let one = ones_like_same_dtype(y);
        let t = crate::tensor::nsl_tensor_sub(one, y_sq, 0); // 1 - y*y
        let out = crate::tensor::nsl_tensor_mul(grad, t, 0); // grad * t
        for p in [y_sq, one, t] { nsl_tensor_free(p); }
        out
    }

    // Sigmoid outputs live in (0,1); tanh outputs in (-1,1). The formulas are
    // bit-exact for any input, but realistic ranges keep the test honest.
    #[test]
    fn sigmoid_backward_cpu_f64_matches_3op() {
        let ys = [0.01, 0.25, 0.5, 0.73, 0.99, 0.5, 0.12, 0.88, 0.331];
        let gs = [0.3, -1.1, 2.0, 0.75, -0.5, 1.0, 4.2, -0.01, 3.3];
        let (yr, gr) = (make_f64(&ys), make_f64(&gs));
        let ref_out = sigmoid_bwd_3op(gr, yr);
        let (yf, gf) = (make_f64(&ys), make_f64(&gs));
        let fused = nsl_tensor_sigmoid_backward(gf, yf);
        let (ro, fo) = (NslTensor::from_ptr(ref_out), NslTensor::from_ptr(fused));
        for i in 0..ys.len() {
            let (r, f) = unsafe { (*ro.data_f64().add(i), *fo.data_f64().add(i)) };
            assert_eq!(r.to_bits(), f.to_bits(), "cpu f64 mismatch at {i}: {r} vs {f}");
        }
        for p in [yr, gr, ref_out, yf, gf, fused] { nsl_tensor_free(p); }
    }

    #[test]
    fn tanh_backward_cpu_f64_matches_3op() {
        let ys = [-0.99, -0.5, 0.0, 0.31, 0.75, 0.999, -0.12, 0.88, -0.331];
        let gs = [0.3, -1.1, 2.0, 0.75, -0.5, 1.0, 4.2, -0.01, 3.3];
        let (yr, gr) = (make_f64(&ys), make_f64(&gs));
        let ref_out = tanh_bwd_3op(gr, yr);
        let (yf, gf) = (make_f64(&ys), make_f64(&gs));
        let fused = nsl_tensor_tanh_backward(gf, yf);
        let (ro, fo) = (NslTensor::from_ptr(ref_out), NslTensor::from_ptr(fused));
        for i in 0..ys.len() {
            let (r, f) = unsafe { (*ro.data_f64().add(i), *fo.data_f64().add(i)) };
            assert_eq!(r.to_bits(), f.to_bits(), "cpu f64 mismatch at {i}: {r} vs {f}");
        }
        for p in [yr, gr, ref_out, yf, gf, fused] { nsl_tensor_free(p); }
    }

    // Read a tensor's elements as f64 regardless of its stored dtype.
    fn as_f64(p: i64) -> Vec<f64> {
        let t = NslTensor::from_ptr(p);
        let n = t.len as usize;
        if t.dtype == 1 {
            (0..n).map(|i| unsafe { *t.data_f32().add(i) as f64 }).collect()
        } else {
            (0..n).map(|i| unsafe { *t.data_f64().add(i) }).collect()
        }
    }

    /// A scalar `grad` (len 1, e.g. a `sum`/`mean` seed) must broadcast against
    /// y[n] via the decomposed fallback — NOT panic (the pre-fusion path did this
    /// with its final broadcasting `nsl_tensor_mul`). Output is y-shaped and each
    /// element equals grad0 * activation'(y[i]).
    #[test]
    fn sigmoid_backward_scalar_grad_broadcasts() {
        let ys = [0.1, 0.5, 0.9, 0.3, 0.72];
        let c = 2.5_f64;
        let (y, grad) = (make_f64(&ys), make_f64(&[c]));
        let out = nsl_tensor_sigmoid_backward(grad, y);
        assert_eq!(NslTensor::from_ptr(out).len, ys.len() as i64, "output must be y-shaped");
        let got = as_f64(out);
        for (i, &yv) in ys.iter().enumerate() {
            let expected = c * yv * (1.0 - yv);
            assert!((got[i] - expected).abs() < 1e-5, "at {i}: {} vs {expected}", got[i]);
        }
        for p in [y, grad, out] { nsl_tensor_free(p); }
    }

    #[test]
    fn tanh_backward_scalar_grad_broadcasts() {
        let ys = [0.0, 0.5, -0.5, 0.76, -0.99];
        let c = -1.75_f64;
        let (y, grad) = (make_f64(&ys), make_f64(&[c]));
        let out = nsl_tensor_tanh_backward(grad, y);
        assert_eq!(NslTensor::from_ptr(out).len, ys.len() as i64, "output must be y-shaped");
        let got = as_f64(out);
        for (i, &yv) in ys.iter().enumerate() {
            let expected = c * (1.0 - yv * yv);
            assert!((got[i] - expected).abs() < 1e-5, "at {i}: {} vs {expected}", got[i]);
        }
        for p in [y, grad, out] { nsl_tensor_free(p); }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn sigmoid_backward_gpu_f32_matches_3op() {
        let n = 257usize; // not a multiple of 256 — exercises the block tail guard
        let ys: Vec<f32> = (0..n).map(|i| 1.0 / (1.0 + (-((i as f32) * 0.05 - 6.0)).exp())).collect();
        let gs: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 2.0).collect();
        let to_gpu = |d: &[f32]| crate::tensor::nsl_tensor_to_device(make_f32(d), 1);
        let (yr, gr) = (to_gpu(&ys), to_gpu(&gs));
        let ref_out = sigmoid_bwd_3op(gr, yr);
        let (yf, gf) = (to_gpu(&ys), to_gpu(&gs));
        let fused = nsl_tensor_sigmoid_backward(gf, yf);
        let up = |p: i64| -> Vec<f64> {
            let c = crate::tensor::nsl_tensor_to_device(p, 0);
            let t = NslTensor::from_ptr(c);
            (0..n).map(|i| unsafe { *t.data_f64().add(i) }).collect()
        };
        let (r, f) = (up(ref_out), up(fused));
        for i in 0..n {
            assert_eq!(r[i].to_bits(), f[i].to_bits(), "gpu f32 mismatch at {i}: {} vs {}", r[i], f[i]);
        }
        for p in [yr, gr, ref_out, yf, gf, fused] { nsl_tensor_free(p); }
    }

    /// GPU scalar-grad broadcast: the fallback materialises the f32 `1.0`
    /// constant on CPU and lets the decomposed ops reconcile it onto the GPU y,
    /// with the final `nsl_tensor_mul` broadcasting the scalar grad.
    #[test]
    #[cfg(feature = "cuda")]
    fn sigmoid_backward_gpu_scalar_grad_broadcasts() {
        let ys: Vec<f32> = (0..300).map(|i| 1.0 / (1.0 + (-((i as f32) * 0.03 - 4.5)).exp())).collect();
        let c = 2.5_f32;
        let y = crate::tensor::nsl_tensor_to_device(make_f32(&ys), 1);
        let grad = crate::tensor::nsl_tensor_to_device(make_f32(&[c]), 1);
        let out = nsl_tensor_sigmoid_backward(grad, y);
        assert_eq!(NslTensor::from_ptr(out).len, ys.len() as i64, "output must be y-shaped");
        let cpu = crate::tensor::nsl_tensor_to_device(out, 0);
        let ot = NslTensor::from_ptr(cpu);
        for (i, &yv) in ys.iter().enumerate() {
            let expected = (c * yv * (1.0 - yv)) as f64;
            let got = unsafe { *ot.data_f64().add(i) };
            assert!((got - expected).abs() < 1e-5, "at {i}: {got} vs {expected}");
        }
        for p in [y, grad, out, cpu] { nsl_tensor_free(p); }
    }

    #[test]
    #[cfg(feature = "cuda")]
    fn tanh_backward_gpu_f32_matches_3op() {
        let n = 257usize;
        let ys: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.05 - 6.0).tanh()).collect();
        let gs: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).cos() * 2.0).collect();
        let to_gpu = |d: &[f32]| crate::tensor::nsl_tensor_to_device(make_f32(d), 1);
        let (yr, gr) = (to_gpu(&ys), to_gpu(&gs));
        let ref_out = tanh_bwd_3op(gr, yr);
        let (yf, gf) = (to_gpu(&ys), to_gpu(&gs));
        let fused = nsl_tensor_tanh_backward(gf, yf);
        let up = |p: i64| -> Vec<f64> {
            let c = crate::tensor::nsl_tensor_to_device(p, 0);
            let t = NslTensor::from_ptr(c);
            (0..n).map(|i| unsafe { *t.data_f64().add(i) }).collect()
        };
        let (r, f) = (up(ref_out), up(fused));
        for i in 0..n {
            assert_eq!(r[i].to_bits(), f[i].to_bits(), "gpu f32 mismatch at {i}: {} vs {}", r[i], f[i]);
        }
        for p in [yr, gr, ref_out, yf, gf, fused] { nsl_tensor_free(p); }
    }
}

#[cfg(test)]
mod gelu_backward_tests {
    use super::*;
    use crate::tensor::NslTensor;

    #[cfg(feature = "cuda")]
    fn make_f32(data: &[f32]) -> i64 {
        let ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&[data.len() as i64], 1);
        let t = NslTensor::from_ptr(ptr);
        for (i, v) in data.iter().enumerate() { unsafe { *t.data_f32().add(i) = *v }; }
        ptr
    }
    fn make_f64(data: &[f64]) -> i64 {
        let ptr = crate::cpu::create_tensor_with_shape_rs_dtype(&[data.len() as i64], 0);
        let t = NslTensor::from_ptr(ptr);
        for (i, v) in data.iter().enumerate() { unsafe { *t.data_f64().add(i) = *v }; }
        ptr
    }

    /// CPU f64: fused output equals grad · gelu_tanh_deriv (same formula, same
    /// order → bit-equal), and at x=0 the derivative is exactly 0.5 (true for
    /// EVERY gelu approximation — the old expansion returned 0.625 there).
    #[test]
    fn gelu_backward_cpu_f64_matches_tanh_deriv() {
        let xs = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0];
        let gs = [0.3, -1.1, 2.0, 1.0, 0.75, -0.5, 1.0, 4.2];
        let (x, g) = (make_f64(&xs), make_f64(&gs));
        let out = nsl_tensor_gelu_backward(g, x);
        let ot = NslTensor::from_ptr(out);
        for (i, (&xv, &gv)) in xs.iter().zip(&gs).enumerate() {
            let expected = gv * gelu_tanh_deriv_f64(xv);
            let got = unsafe { *ot.data_f64().add(i) };
            assert_eq!(got.to_bits(), expected.to_bits(), "at {i}: {got} vs {expected}");
        }
        // x=0 (index 3, grad 1.0): every gelu' is exactly 0.5 there.
        assert_eq!(unsafe { *ot.data_f64().add(3) }, 0.5);
        for p in [x, g, out] { nsl_tensor_free(p); }
    }

    /// CPU f64 finite differences against the ACTUAL forward `nsl_tensor_gelu`:
    /// gelu'(x) ≈ (gelu(x+h) − gelu(x−h)) / 2h. The gold-standard check that the
    /// backward is the derivative of the forward this device runs.
    #[test]
    fn gelu_backward_cpu_f64_finite_differences() {
        let xs = [-2.5, -1.0, -0.3, 0.0, 0.4, 1.0, 1.7, 3.0];
        let h = 1e-6_f64;
        let gelu_at = |vals: &Vec<f64>| -> Vec<f64> {
            let t = make_f64(vals);
            // Bump so the forward can't FBIP-consume our handle mid-test.
            NslTensor::from_ptr(t).refcount.fetch_add(1, Ordering::SeqCst);
            let y = nsl_tensor_gelu(t);
            NslTensor::from_ptr(t).refcount.fetch_sub(1, Ordering::SeqCst);
            let yt = NslTensor::from_ptr(y);
            let out = (0..vals.len()).map(|i| unsafe { *yt.data_f64().add(i) }).collect();
            nsl_tensor_free(y);
            nsl_tensor_free(t);
            out
        };
        let up = gelu_at(&xs.iter().map(|v| v + h).collect());
        let dn = gelu_at(&xs.iter().map(|v| v - h).collect());
        let (x, ones) = (make_f64(&xs), make_f64(&vec![1.0; xs.len()]));
        let out = nsl_tensor_gelu_backward(ones, x);
        let ot = NslTensor::from_ptr(out);
        for i in 0..xs.len() {
            let fd = (up[i] - dn[i]) / (2.0 * h);
            let got = unsafe { *ot.data_f64().add(i) };
            assert!((got - fd).abs() < 1e-7, "at {i} (x={}): fused {got} vs FD {fd}", xs[i]);
        }
        for p in [x, ones, out] { nsl_tensor_free(p); }
    }

    /// Scalar `sum`-seed grad broadcasts via the fallback (deriv materialized,
    /// then broadcast mul) instead of panicking.
    #[test]
    fn gelu_backward_scalar_grad_broadcasts() {
        let xs = [-2.0, -0.5, 0.0, 1.0, 3.0];
        let c = 1.5_f64;
        let (x, grad) = (make_f64(&xs), make_f64(&[c]));
        let out = nsl_tensor_gelu_backward(grad, x);
        assert_eq!(NslTensor::from_ptr(out).len, xs.len() as i64, "output must be x-shaped");
        let ot = NslTensor::from_ptr(out);
        for (i, &xv) in xs.iter().enumerate() {
            let expected = c * gelu_tanh_deriv_f64(xv);
            let got = unsafe { *ot.data_f64().add(i) };
            assert!((got - expected).abs() < 1e-12, "at {i}: {got} vs {expected}");
        }
        for p in [x, grad, out] { nsl_tensor_free(p); }
    }

    /// GPU f32: fused kernel vs the host-computed sigmoid-approx derivative
    /// σ(1.702x)·(1+1.702x·(1−σ)) — the derivative of GELU_F32_PTX's forward.
    /// Tolerance covers ex2.approx/rcp.approx (~1-2 ulp each).
    #[test]
    #[cfg(feature = "cuda")]
    fn gelu_backward_gpu_f32_matches_sigmoid_deriv() {
        let n = 257usize; // 256-block tail guard
        let xs: Vec<f32> = (0..n).map(|i| (i as f32) * 0.05 - 6.0).collect();
        let gs: Vec<f32> = (0..n).map(|i| ((i as f32) * 0.013).sin() * 2.0).collect();
        let to_gpu = |d: &[f32]| crate::tensor::nsl_tensor_to_device(make_f32(d), 1);
        let (x, g) = (to_gpu(&xs), to_gpu(&gs));
        let out = nsl_tensor_gelu_backward(g, x);
        let cpu = crate::tensor::nsl_tensor_to_device(out, 0);
        let ot = NslTensor::from_ptr(cpu);
        for i in 0..n {
            let (xv, gv) = (xs[i] as f64, gs[i] as f64);
            let kx = 1.702_f32 as f64 * xv;
            let s = 1.0 / (1.0 + (-kx).exp());
            let expected = gv * (s * (1.0 + kx * (1.0 - s)));
            let got = unsafe { *ot.data_f64().add(i) };
            assert!(
                (got - expected).abs() < 1e-5 * (1.0 + expected.abs()),
                "at {i} (x={xv}): fused {got} vs analytic {expected}"
            );
        }
        for p in [x, g, out, cpu] { nsl_tensor_free(p); }
    }

    /// GPU f32 finite differences against the GPU forward (GELU_F32_PTX):
    /// central difference with h=1e-2; tolerance dominated by f32 rounding noise.
    #[test]
    #[cfg(feature = "cuda")]
    fn gelu_backward_gpu_f32_finite_differences() {
        let xs: Vec<f32> = vec![-2.5, -1.0, -0.3, 0.0, 0.4, 1.0, 1.7, 3.0];
        let n = xs.len();
        let h = 1e-2_f32;
        let gelu_gpu = |vals: &Vec<f32>| -> Vec<f64> {
            let t = crate::tensor::nsl_tensor_to_device(make_f32(vals), 1);
            NslTensor::from_ptr(t).refcount.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
            let y = nsl_tensor_gelu(t);
            NslTensor::from_ptr(t).refcount.fetch_sub(1, std::sync::atomic::Ordering::SeqCst);
            let c = crate::tensor::nsl_tensor_to_device(y, 0);
            let ct = NslTensor::from_ptr(c);
            let out = (0..n).map(|i| unsafe { *ct.data_f64().add(i) }).collect();
            for p in [t, y, c] { nsl_tensor_free(p); }
            out
        };
        let up = gelu_gpu(&xs.iter().map(|v| v + h).collect());
        let dn = gelu_gpu(&xs.iter().map(|v| v - h).collect());
        let to_gpu = |d: &[f32]| crate::tensor::nsl_tensor_to_device(make_f32(d), 1);
        let (x, ones) = (to_gpu(&xs), to_gpu(&vec![1.0_f32; n]));
        let out = nsl_tensor_gelu_backward(ones, x);
        let cpu = crate::tensor::nsl_tensor_to_device(out, 0);
        let ot = NslTensor::from_ptr(cpu);
        for i in 0..n {
            let fd = (up[i] - dn[i]) / (2.0 * h as f64);
            let got = unsafe { *ot.data_f64().add(i) };
            assert!((got - fd).abs() < 5e-3, "at {i} (x={}): fused {got} vs FD {fd}", xs[i]);
        }
        for p in [x, ones, out, cpu] { nsl_tensor_free(p); }
    }

    /// GPU scalar-grad broadcast: exercises the deriv-materialization fallback
    /// (ones_like → fused kernel → broadcasting mul → 3 frees) on device.
    #[test]
    #[cfg(feature = "cuda")]
    fn gelu_backward_gpu_scalar_grad_broadcasts() {
        let xs: Vec<f32> = (0..300).map(|i| (i as f32) * 0.03 - 4.5).collect();
        let c = 2.5_f32;
        let x = crate::tensor::nsl_tensor_to_device(make_f32(&xs), 1);
        let grad = crate::tensor::nsl_tensor_to_device(make_f32(&[c]), 1);
        let out = nsl_tensor_gelu_backward(grad, x);
        assert_eq!(NslTensor::from_ptr(out).len, xs.len() as i64, "output must be x-shaped");
        let cpu = crate::tensor::nsl_tensor_to_device(out, 0);
        let ot = NslTensor::from_ptr(cpu);
        for (i, &xv) in xs.iter().enumerate() {
            let kx = 1.702_f32 as f64 * xv as f64;
            let s = 1.0 / (1.0 + (-kx).exp());
            let expected = c as f64 * (s * (1.0 + kx * (1.0 - s)));
            let got = unsafe { *ot.data_f64().add(i) };
            assert!(
                (got - expected).abs() < 1e-5 * (1.0 + expected.abs()),
                "at {i} (x={xv}): {got} vs {expected}"
            );
        }
        for p in [x, grad, out, cpu] { nsl_tensor_free(p); }
    }
}
