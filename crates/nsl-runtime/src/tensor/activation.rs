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
    // Elementwise backward: the incoming adjoint always shares the saved input's
    // shape and placement. Assert both (parity with `gpu_backward_binary`'s
    // length check; the CPU path is flat-indexed so a shorter grad would OOB).
    assert_eq!(
        gt.len, xt.len,
        "silu_backward: grad len {} != x len {}",
        gt.len, xt.len
    );
    debug_assert_eq!(
        gt.device, xt.device,
        "silu_backward: grad/x device mismatch ({} vs {})",
        gt.device, xt.device
    );
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
}
