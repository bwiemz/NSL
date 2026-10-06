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

/// Helper: create a tensor from a shape list, zero-filled, in f16 storage
/// (dtype=2, 2 bytes/element).
///
/// Used by the CSHA Tier C fused backward kernel's gradient-output
/// allocations (dq/dk/dv/dwq/dwk/dwv), which the PTX writes via
/// `st.global.u16`.  The matching f32 (`nsl_tensor_zeros*`) helper
/// over-allocates for these by 2×, and the kernel's half-width stores
/// leave the upper bytes uninitialised — a subsequent f32 read then
/// interprets the raw f16 bits as f32 and produces garbage.  This
/// helper gives the backward a correctly-sized f16 buffer.
///
/// `fill == 0.0` uses zero-init; any other fill value panics (no
/// non-zero caller exists today, and implementing a correct f16
/// conversion would pull in half-crate wiring we don't need).
pub(crate) fn tensor_from_shape_list_f16(shape_list: i64, fill: f64) -> i64 {
    assert_eq!(
        fill, 0.0,
        "tensor_from_shape_list_f16: only fill=0.0 is supported today"
    );

    let list = NslList::from_ptr(shape_list);
    let ndim = list.len;

    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim as usize {
        unsafe { *shape.add(i) = *list.data.add(i) };
    }

    let len = NslTensor::total_elements(shape, ndim);
    // f16 = 2 bytes/element.
    let data_size = (len as usize) * 2;
    let data = checked_alloc_zeroed(data_size) as *mut u16;

    let strides = NslTensor::compute_strides(shape, ndim);

    let tensor = Box::new(NslTensor::new(
        data as *mut c_void,
        shape,
        strides,
        ndim,
        len,
        0,
        super::DTYPE_FP16,
        1,
        0,
    ));
    NslTensor::publish(tensor)
}

/// Helper: create a tensor from a shape list, filling data with a given value (f64, dtype=0).
/// Used for operations that explicitly require double precision.
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

#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_zeros(shape_list: i64) -> i64 {
    tensor_from_shape_list(shape_list, 0.0)
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_ones(shape_list: i64) -> i64 {
    tensor_from_shape_list(shape_list, 1.0)
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_full(shape_list: i64, value: f64) -> i64 {
    tensor_from_shape_list(shape_list, value)
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_rand(shape_list: i64) -> i64 {
    let ptr = tensor_from_shape_list(shape_list, 0.0);
    let tensor = NslTensor::from_ptr(ptr);
    for i in 0..tensor.len as usize {
        let val = crate::sampling::rng_f64() as f32;
        unsafe { *tensor.data_f32().add(i) = val };
    }
    ptr
}

#[unsafe(no_mangle)]
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

#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_arange(start: f64, stop: f64, step: f64) -> i64 {
    if step == 0.0 {
        crate::nsl_log!(ERROR, "nsl", "nsl: tensor arange step cannot be zero");
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

// === Creation in a chosen dtype (C5 step 3) ===
//
// The checker types a creation builtin f32 -- the default float dtype -- or
// f64 when an annotation chooses it (`let x: Tensor<[4], f64> = zeros([4])`),
// and codegen calls these `_dtype` variants for the f64 case only, so every
// f32 creation emits the same call it always did. An f64 tensor is computed
// in f64 (`full(.., 0.1)` holds f64 0.1, not a widened f32), and the random
// variants draw the same RNG stream without rounding it. Any other dtype is
// refused: creation makes f32 or f64.

fn creation_dtype(dtype: i64, what: &str) -> bool {
    match dtype {
        0 => true,
        1 => false,
        other => crate::fatal::die(
            crate::fatal::Fatal::UnsupportedDtype,
            &format!("{what}: a creation builtin makes f32 or f64, not dtype {other}"),
        ),
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_zeros_dtype(shape_list: i64, dtype: i64) -> i64 {
    if creation_dtype(dtype, "nsl_tensor_zeros_dtype") {
        tensor_from_shape_list_f64(shape_list, 0.0)
    } else {
        nsl_tensor_zeros(shape_list)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_ones_dtype(shape_list: i64, dtype: i64) -> i64 {
    if creation_dtype(dtype, "nsl_tensor_ones_dtype") {
        tensor_from_shape_list_f64(shape_list, 1.0)
    } else {
        nsl_tensor_ones(shape_list)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_full_dtype(shape_list: i64, value: f64, dtype: i64) -> i64 {
    if creation_dtype(dtype, "nsl_tensor_full_dtype") {
        tensor_from_shape_list_f64(shape_list, value)
    } else {
        nsl_tensor_full(shape_list, value)
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_rand_dtype(shape_list: i64, dtype: i64) -> i64 {
    if !creation_dtype(dtype, "nsl_tensor_rand_dtype") {
        return nsl_tensor_rand(shape_list);
    }
    let ptr = tensor_from_shape_list_f64(shape_list, 0.0);
    let tensor = NslTensor::from_ptr(ptr);
    for i in 0..tensor.len as usize {
        unsafe { *tensor.data_f64().add(i) = crate::sampling::rng_f64() };
    }
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_randn_dtype(shape_list: i64, dtype: i64) -> i64 {
    if !creation_dtype(dtype, "nsl_tensor_randn_dtype") {
        return nsl_tensor_randn(shape_list);
    }
    let ptr = tensor_from_shape_list_f64(shape_list, 0.0);
    let tensor = NslTensor::from_ptr(ptr);
    // The same Box-Muller pairs `nsl_tensor_randn` draws, kept in f64.
    let len = tensor.len as usize;
    let mut i = 0;
    while i < len {
        let u1 = crate::sampling::rng_f64().max(1e-15);
        let u2 = crate::sampling::rng_f64();
        let mag = (-2.0 * u1.ln()).sqrt();
        unsafe { *tensor.data_f64().add(i) = mag * (2.0 * std::f64::consts::PI * u2).cos() };
        if i + 1 < len {
            unsafe {
                *tensor.data_f64().add(i + 1) = mag * (2.0 * std::f64::consts::PI * u2).sin();
            }
        }
        i += 2;
    }
    ptr
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_arange_dtype(start: f64, stop: f64, step: f64, dtype: i64) -> i64 {
    if !creation_dtype(dtype, "nsl_tensor_arange_dtype") {
        return nsl_tensor_arange(start, stop, step);
    }
    // f64: convert the f32 arange's 1-D layout, but compute every element in f64.
    let f32_ptr = nsl_tensor_arange(start, stop, step);
    let len = NslTensor::from_ptr_ref(f32_ptr).len as usize;
    let out = crate::tensor::precision_cast::convert_untaped(f32_ptr, 0);
    crate::tensor::nsl_tensor_free(f32_ptr);
    let t = NslTensor::from_ptr(out);
    for i in 0..len {
        unsafe { *t.data_f64().add(i) = start + (i as f64) * step };
    }
    out
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

#[cfg(test)]
mod dtype_creation_tests {
    use super::*;
    use crate::list::{nsl_list_free, nsl_list_new, nsl_list_push};

    fn shape(dims: &[i64]) -> i64 {
        let l = nsl_list_new();
        for &d in dims {
            nsl_list_push(l, d);
        }
        l
    }

    /// C5 step 3: the `_dtype` creators make f64 for tag 0 -- computed in f64,
    /// not widened from f32 -- and exactly the plain f32 creators for tag 1;
    /// `zeros_like`/`ones_like` follow an f64 template.
    #[test]
    fn dtype_creators_and_like_follow_the_requested_dtype() {
        let s = shape(&[3]);
        let full64 = nsl_tensor_full_dtype(s, 0.1, 0);
        let t = NslTensor::from_ptr_ref(full64);
        assert_eq!(t.dtype, 0);
        assert_eq!(unsafe { *t.data_f64() }, 0.1, "exact f64 0.1, not a widened f32");
        let full32 = nsl_tensor_full_dtype(s, 0.1, 1);
        assert_eq!(NslTensor::from_ptr_ref(full32).dtype, 1);

        let ar = nsl_tensor_arange_dtype(0.0, 0.4, 0.1, 0);
        let a = NslTensor::from_ptr_ref(ar);
        assert_eq!((a.dtype, a.len), (0, 4));
        assert_eq!(unsafe { *a.data_f64().add(3) }, 0.30000000000000004);

        crate::sampling::nsl_manual_seed(7);
        let r64 = nsl_tensor_randn_dtype(s, 0);
        crate::sampling::nsl_manual_seed(7);
        let r32 = nsl_tensor_randn(s);
        let (x64, x32) = (NslTensor::from_ptr_ref(r64), NslTensor::from_ptr_ref(r32));
        for i in 0..3 {
            let v64 = unsafe { *x64.data_f64().add(i) };
            let v32 = unsafe { *x32.data_f32().add(i) };
            assert_eq!(v64 as f32, v32, "element {i}: the same draw, unrounded");
        }

        let z = crate::tensor::nsl_tensor_zeros_like(full64);
        assert_eq!(NslTensor::from_ptr_ref(z).dtype, 0, "zeros_like of f64 is f64");
        let o = crate::tensor::nsl_tensor_ones_like(full64);
        assert_eq!(NslTensor::from_ptr_ref(o).dtype, 0, "ones_like of f64 is f64");
        let z32 = crate::tensor::nsl_tensor_zeros_like(full32);
        assert_eq!(NslTensor::from_ptr_ref(z32).dtype, 1, "zeros_like of f32 stays f32");

        for p in [full64, full32, ar, r64, r32, z, o, z32] {
            crate::tensor::nsl_tensor_free(p);
        }
        nsl_list_free(s);
    }
}
