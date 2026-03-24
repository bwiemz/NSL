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
    // FBIP: mutate in-place when uniquely owned
    {
        let t = unsafe { &mut *(tensor_ptr as *mut NslTensor) };
        if t.can_mutate_inplace() {
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

    let data: *mut c_void = if a.dtype == 1 {
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
        a.dtype,
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
    ($name:ident, $op_f32:expr, $op_f64:expr) => {
        #[no_mangle]
        pub extern "C" fn $name(ptr: i64) -> i64 {
            let t = unsafe { &mut *(ptr as *mut NslTensor) };
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

define_inplace_unary!(nsl_tensor_relu_inplace, |v: f32| if v > 0.0 { v } else { 0.0_f32 }, |v: f64| if v > 0.0 { v } else { 0.0 });
define_inplace_unary!(nsl_tensor_exp_inplace, |v: f32| v.exp(), |v: f64| v.exp());
define_inplace_unary!(nsl_tensor_log_inplace, |v: f32| v.ln(), |v: f64| v.ln());
define_inplace_unary!(nsl_tensor_sqrt_inplace, |v: f32| v.sqrt(), |v: f64| v.sqrt());
define_inplace_unary!(nsl_tensor_abs_inplace, |v: f32| v.abs(), |v: f64| v.abs());
define_inplace_unary!(nsl_tensor_sigmoid_inplace, |v: f32| 1.0_f32 / (1.0_f32 + (-v).exp()), |v: f64| 1.0 / (1.0 + (-v).exp()));
define_inplace_unary!(nsl_tensor_tanh_inplace, |v: f32| v.tanh(), |v: f64| v.tanh());
define_inplace_unary!(nsl_tensor_neg_inplace, |v: f32| -v, |v: f64| -v);
define_inplace_unary!(nsl_tensor_sign_inplace, |v: f32| if v > 0.0 { 1.0_f32 } else if v < 0.0 { -1.0_f32 } else { 0.0_f32 }, |v: f64| if v > 0.0 { 1.0 } else if v < 0.0 { -1.0 } else { 0.0 });

/// GELU in-place.
#[no_mangle]
pub extern "C" fn nsl_tensor_gelu_inplace(ptr: i64) -> i64 {
    let t = unsafe { &mut *(ptr as *mut NslTensor) };
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
