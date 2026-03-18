//! Tensor activation and elementwise math operations: relu, sigmoid, tanh, gelu, silu,
//! exp, log, sqrt, abs, sign, clamp.

use std::ffi::c_void;

use crate::autodiff;
use crate::memory::checked_alloc;

use super::NslTensor;

// === Element-wise math ops ===

#[no_mangle]
pub extern "C" fn nsl_tensor_exp(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::EXP_F32_PTX, "nsl_exp_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
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
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::LOG_F32_PTX, "nsl_log_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
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
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::SQRT_F32_PTX, "nsl_sqrt_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
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
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::ABS_F32_PTX, "nsl_abs_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
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
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::SIGN_F32_PTX, "nsl_sign_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    // sign is non-differentiable -- no tape recording
    Box::into_raw(result) as i64
}

#[no_mangle]
pub extern "C" fn nsl_tensor_clamp(tensor_ptr: i64, min_val: f64, max_val: f64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
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
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, shape, ndim as usize) };
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: input.device, dtype: input.dtype, owns_data: 1,
    });
    Box::into_raw(result) as i64
}

// === Activation functions ===

#[no_mangle]
pub extern "C" fn nsl_tensor_relu(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::RELU_F32_PTX, "nsl_relu_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
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
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::GELU {
            a: tensor_ptr, out: result, saved_a: tensor_ptr,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_silu(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
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
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::SIGMOID_F32_PTX, "nsl_sigmoid_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
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
            { return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::TANH_F32_PTX, "nsl_tanh_f32\0"); }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
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
