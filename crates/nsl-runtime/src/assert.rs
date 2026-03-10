use std::ffi::CStr;
use std::os::raw::c_char;

use crate::tensor::NslTensor;

/// Extract a UTF-8 string from a raw (ptr, len) pair.
///
/// # Safety
/// `ptr` must point to `len` valid bytes.
fn extract_msg(ptr: i64, len: i64) -> &'static str {
    if ptr == 0 || len <= 0 {
        return "<no message>";
    }
    unsafe {
        let slice = std::slice::from_raw_parts(ptr as *const u8, len as usize);
        std::str::from_utf8_unchecked(slice)
    }
}

#[no_mangle]
pub extern "C" fn nsl_assert(condition: i8, message: i64) {
    if condition == 0 {
        let msg = if message != 0 {
            unsafe { CStr::from_ptr(message as *const c_char) }
                .to_str()
                .unwrap_or("assertion failed")
        } else {
            "assertion failed"
        };
        eprintln!("nsl: assertion failed: {}", msg);
        std::process::abort();
    }
}

#[no_mangle]
pub extern "C" fn nsl_assert_eq_int(a: i64, b: i64, msg_ptr: i64, msg_len: i64) {
    if a != b {
        let msg = extract_msg(msg_ptr, msg_len);
        eprintln!("ASSERTION FAILED: {} (expected {} == {})", msg, a, b);
        std::process::abort();
    }
}

#[no_mangle]
pub extern "C" fn nsl_assert_eq_float(a: f64, b: f64, msg_ptr: i64, msg_len: i64) {
    if a != b {
        let msg = extract_msg(msg_ptr, msg_len);
        eprintln!("ASSERTION FAILED: {} (expected {} == {})", msg, a, b);
        std::process::abort();
    }
}

#[no_mangle]
pub extern "C" fn nsl_assert_close(
    a_ptr: i64,
    b_ptr: i64,
    rtol: f64,
    atol: f64,
    msg_ptr: i64,
    msg_len: i64,
) {
    let a = NslTensor::from_ptr(a_ptr);
    let b = NslTensor::from_ptr(b_ptr);
    let msg = extract_msg(msg_ptr, msg_len);

    // Check ndim
    if a.ndim != b.ndim {
        eprintln!(
            "ASSERTION FAILED: {} (ndim mismatch: {} vs {})",
            msg, a.ndim, b.ndim
        );
        std::process::abort();
    }

    // Check each dimension
    for i in 0..a.ndim as usize {
        let da = unsafe { *a.shape.add(i) };
        let db = unsafe { *b.shape.add(i) };
        if da != db {
            eprintln!(
                "ASSERTION FAILED: {} (shape mismatch at dim {}: {} vs {})",
                msg, i, da, db
            );
            std::process::abort();
        }
    }

    // Element-wise closeness check: |a - b| <= atol + rtol * |b|
    for i in 0..a.len as usize {
        let va = unsafe { *a.data.add(i) };
        let vb = unsafe { *b.data.add(i) };
        let diff = (va - vb).abs();
        let tol = atol + rtol * vb.abs();
        if diff > tol {
            eprintln!(
                "ASSERTION FAILED: {} (element {} not close: {} vs {}, diff={}, tol={})",
                msg, i, va, vb, diff, tol
            );
            std::process::abort();
        }
    }
}

#[no_mangle]
pub extern "C" fn nsl_exit(code: i64) {
    std::process::exit(code as i32);
}
