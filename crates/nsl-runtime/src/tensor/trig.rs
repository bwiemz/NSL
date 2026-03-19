//! Tensor trigonometric operations: elementwise sin, cos.
//! Used for RoPE (Rotary Position Embeddings) precomputation.

use std::ffi::c_void;
use std::sync::atomic::AtomicI64;

use crate::memory::checked_alloc;

use super::NslTensor;

#[no_mangle]
pub extern "C" fn nsl_tensor_sin(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
            #[cfg(feature = "cuda")]
            { panic!("GPU tensor_sin not yet implemented"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: AtomicI64::new(1),
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    Box::into_raw(result) as i64
}

#[no_mangle]
pub extern "C" fn nsl_tensor_cos(tensor_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(tensor_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
            #[cfg(feature = "cuda")]
            { panic!("GPU tensor_cos not yet implemented"); }
        }
    }
    let a = NslTensor::from_ptr(tensor_ptr);
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

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: AtomicI64::new(1),
        device: a.device, dtype: a.dtype, owns_data: 1,
    });
    Box::into_raw(result) as i64
}
