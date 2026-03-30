//! Tensor trigonometric operations: elementwise sin, cos.
//! Used for RoPE (Rotary Position Embeddings) precomputation.

use std::ffi::c_void;
use std::sync::atomic::Ordering;

use crate::autodiff;
use crate::memory::checked_alloc;

use super::{NslTensor, nsl_tensor_contiguous, nsl_tensor_free};

#[no_mangle]
pub extern "C" fn nsl_tensor_sin(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::SIN_F32_PTX, "nsl_sin_f32\0");
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let t_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(t_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f32().add(i)).sin() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f64().add(i)).sin() };
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
    nsl_tensor_free(t_c);
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::Sin {
            a: tensor_ptr, out: result, saved_a: tensor_ptr,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_cos(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                return crate::cuda::gpu_elementwise_unary(tensor_ptr, crate::cuda::kernels::COS_F32_PTX, "nsl_cos_f32\0");
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    let t_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(t_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f32().add(i)).cos() };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*a.data_f64().add(i)).cos() };
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
    nsl_tensor_free(t_c);
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::Cos {
            a: tensor_ptr, out: result, saved_a: tensor_ptr,
        });
    }
    result
}
