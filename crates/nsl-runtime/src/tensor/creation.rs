//! Tensor creation functions: zeros, ones, full, rand, randn, arange, scalar creation.

use std::ffi::c_void;

use crate::list::NslList;
use crate::memory::{checked_alloc, checked_alloc_zeroed};

use super::NslTensor;

/// Helper: create a tensor from a shape list, filling data with a given value (f32, dtype=1).
pub(crate) fn tensor_from_shape_list(shape_list: i64, fill: f64) -> i64 {
    let list = NslList::from_ptr(shape_list);
    let ndim = list.len;

    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim as usize {
        unsafe { *shape.add(i) = *list.data.add(i) };
    }

    let len = NslTensor::total_elements(shape, ndim);
    let fill_f32 = fill as f32;
    let data_size = (len as usize) * std::mem::size_of::<f32>();
    let data = if fill == 0.0 {
        checked_alloc_zeroed(data_size) as *mut f32
    } else {
        let data = checked_alloc(data_size) as *mut f32;
        for i in 0..len as usize {
            unsafe { *data.add(i) = fill_f32 };
        }
        data
    };

    let strides = NslTensor::compute_strides(shape, ndim);

    let tensor = Box::new(NslTensor::new(
        data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        0,
        1,
        1,
        0,
    ));
    NslTensor::publish(tensor)
}

/// Helper: create a tensor from a shape list, filling data with a given value (f64, dtype=0).
/// Used for operations that explicitly require double precision.
#[allow(dead_code)]
pub(crate) fn tensor_from_shape_list_f64(shape_list: i64, fill: f64) -> i64 {
    let list = NslList::from_ptr(shape_list);
    let ndim = list.len;

    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim as usize {
        unsafe { *shape.add(i) = *list.data.add(i) };
    }

    let len = NslTensor::total_elements(shape, ndim);
    let data_size = (len as usize) * std::mem::size_of::<f64>();
    let data = if fill == 0.0 {
        checked_alloc_zeroed(data_size) as *mut f64
    } else {
        let data = checked_alloc(data_size) as *mut f64;
        for i in 0..len as usize {
            unsafe { *data.add(i) = fill };
        }
        data
    };

    let strides = NslTensor::compute_strides(shape, ndim);

    let tensor = Box::new(NslTensor::new(
        data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        0,
        0,
        1,
        0,
    ));
    NslTensor::publish(tensor)
}

/// Create a 0-d scalar tensor containing a single f32 value (dtype=1).
pub(crate) fn create_scalar_tensor(value: f64) -> i64 {
    let data = checked_alloc(std::mem::size_of::<f32>()) as *mut f32;
    unsafe { *data = value as f32 };
    let tensor = Box::new(NslTensor::new(
        data as *mut c_void,
        std::ptr::null_mut(),
        std::ptr::null_mut(),
        0,
        1,
        0,
        1,
        1,
        0,
    ));
    NslTensor::publish(tensor)
}

/// Create a 0-d scalar tensor with dtype-aware storage (dtype=0 -> f64, dtype=1 -> f32).
pub(crate) fn create_scalar_tensor_dtype(value: f64, dtype: u16) -> i64 {
    if dtype == 1 {
        create_scalar_tensor(value)
    } else {
        let data = checked_alloc(std::mem::size_of::<f64>()) as *mut f64;
        unsafe { *data = value };
        let tensor = Box::new(NslTensor::new(
            data as *mut c_void,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
            1,
            0,
            0,
            1,
            0,
        ));
        NslTensor::publish(tensor)
    }
}

// === Creation ===

#[no_mangle]
pub extern "C" fn nsl_tensor_zeros(shape_list: i64) -> i64 {
    tensor_from_shape_list(shape_list, 0.0)
}

#[no_mangle]
pub extern "C" fn nsl_tensor_ones(shape_list: i64) -> i64 {
    tensor_from_shape_list(shape_list, 1.0)
}

#[no_mangle]
pub extern "C" fn nsl_tensor_full(shape_list: i64, value: f64) -> i64 {
    tensor_from_shape_list(shape_list, value)
}

#[no_mangle]
pub extern "C" fn nsl_tensor_rand(shape_list: i64) -> i64 {
    let ptr = tensor_from_shape_list(shape_list, 0.0);
    let tensor = NslTensor::from_ptr(ptr);
    for i in 0..tensor.len as usize {
        let val = crate::sampling::rng_f64() as f32;
        unsafe { *tensor.data_f32().add(i) = val };
    }
    ptr
}

#[no_mangle]
pub extern "C" fn nsl_tensor_randn(shape_list: i64) -> i64 {
    let ptr = tensor_from_shape_list(shape_list, 0.0);
    let tensor = NslTensor::from_ptr(ptr);
    // Box-Muller transform: generate N(0,1) from uniform samples using seeded RNG
    let len = tensor.len as usize;
    let mut i = 0;
    while i + 1 < len {
        let u1 = crate::sampling::rng_f64().max(1e-15); // avoid log(0)
        let u2 = crate::sampling::rng_f64();

        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = (mag * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
        let z1 = (mag * (2.0 * std::f64::consts::PI * u2).sin()) as f32;
        unsafe {
            *tensor.data_f32().add(i) = z0;
            *tensor.data_f32().add(i + 1) = z1;
        }
        i += 2;
    }
    // If odd number of elements, generate one more pair and use first
    if i < len {
        let u1 = crate::sampling::rng_f64().max(1e-15);
        let u2 = crate::sampling::rng_f64();

        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = (mag * (2.0 * std::f64::consts::PI * u2).cos()) as f32;
        unsafe { *tensor.data_f32().add(i) = z0 };
    }
    ptr
}

#[no_mangle]
pub extern "C" fn nsl_tensor_arange(start: f64, stop: f64, step: f64) -> i64 {
    if step == 0.0 {
        eprintln!("nsl: tensor arange step cannot be zero");
        std::process::abort();
    }
    let len = ((stop - start) / step).ceil().max(0.0) as i64;

    // Create 1D tensor
    let ndim: i64 = 1;
    let shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *shape = len };

    let strides = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *strides = 1 };

    let data = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
    for i in 0..len as usize {
        unsafe { *data.add(i) = (start + (i as f64) * step) as f32 };
    }

    let tensor = Box::new(NslTensor::new(
        data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        0,
        1,
        1,
        0,
    ));
    NslTensor::publish(tensor)
}

/// Create a tensor from a raw f64 slice and shape array.
/// Used by sparse → dense conversion, SpMM output, and other internal APIs.
/// Returns pointer to NslTensor as i64, or 0 on empty data.
pub(crate) fn create_tensor_from_f64_data(data_slice: &[f64], shape_slice: &[i64]) -> i64 {
    let ndim = shape_slice.len() as i64;
    let len: i64 = shape_slice.iter().product();
    if len == 0 { return 0; }

    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &s) in shape_slice.iter().enumerate() {
        unsafe { *shape.add(i) = s };
    }

    let strides = NslTensor::compute_strides(shape, ndim);

    let data_size = (len as usize) * std::mem::size_of::<f64>();
    let data = checked_alloc(data_size) as *mut f64;
    unsafe {
        std::ptr::copy_nonoverlapping(data_slice.as_ptr(), data, len as usize);
    }

    let tensor = Box::new(NslTensor::new(
        data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        0,
        0,
        1,
        0,
    ));
    NslTensor::publish(tensor)
}
