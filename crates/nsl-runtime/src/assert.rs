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

#[unsafe(no_mangle)]
pub extern "C" fn nsl_assert(condition: i8, message: i64) {
    if condition == 0 {
        let msg = if message != 0 {
            unsafe { CStr::from_ptr(message as *const c_char) }
                .to_str()
                .unwrap_or("assertion failed")
        } else {
            "assertion failed"
        };
        crate::nsl_log!(ERROR, "nsl", "nsl: assertion failed: {}", msg);
        std::process::abort();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_assert_eq_int(a: i64, b: i64, msg_ptr: i64, msg_len: i64) {
    if a != b {
        let msg = extract_msg(msg_ptr, msg_len);
        crate::nsl_log!(ERROR, "assert", "ASSERTION FAILED: {} (expected {} == {})", msg, a, b);
        std::process::abort();
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_assert_eq_float(a: f64, b: f64, msg_ptr: i64, msg_len: i64) {
    if a != b {
        let msg = extract_msg(msg_ptr, msg_len);
        crate::nsl_log!(ERROR, "assert", "ASSERTION FAILED: {} (expected {} == {})", msg, a, b);
        std::process::abort();
    }
}

/// `|a - b| <= atol + rtol * |b|`, refusing what that formula lets through
/// as written with `>`: a NaN on either side makes `diff` (or `tol`) NaN, and
/// `NaN > tol` is false, so a NaN compared as close to anything. An infinity
/// is close only to the same infinity — the formula alone would accept
/// `1.0` against `+inf` (`tol` is `inf` too) and `+inf` against `-inf`.
/// NaN is never close, not even to NaN (numpy / torch `equal_nan=False`).
fn is_close(a: f64, b: f64, rtol: f64, atol: f64) -> bool {
    if a.is_nan() || b.is_nan() {
        return false;
    }
    if a.is_infinite() || b.is_infinite() {
        return a == b;
    }
    let diff = (a - b).abs();
    // `<=`, not `!(diff > tol)`: a NaN tolerance (a NaN rtol/atol) fails too.
    diff <= atol + rtol * b.abs()
}

#[unsafe(no_mangle)]
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
        crate::nsl_log!(ERROR, "assert", 
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
            crate::nsl_log!(ERROR, "assert", 
                "ASSERTION FAILED: {} (shape mismatch at dim {}: {} vs {})",
                msg, i, da, db
            );
            std::process::abort();
        }
    }

    // Element-wise closeness check: |a - b| <= atol + rtol * |b|.
    // Read each operand through its own dtype rather than assuming both share
    // one: the two sides frequently differ (e.g. `zeros`/`full`/`ones` create
    // f32 CPU tensors, while a `.to(cpu)` transfer upcasts GPU f32 -> CPU f64),
    // and comparing values must not depend on the dtypes matching. `read_scalar
    // _as_f64` handles f32/f64/f16/bf16/i32, matching the existing contiguous
    // flat-index assumption.
    for i in 0..a.len as usize {
        let va = a.read_scalar_as_f64(i);
        let vb = b.read_scalar_as_f64(i);
        if !is_close(va, vb, rtol, atol) {
            let diff = (va - vb).abs();
            let tol = atol + rtol * vb.abs();
            crate::nsl_log!(ERROR, "assert", 
                "ASSERTION FAILED: {} (element {} not close: {} vs {}, diff={}, tol={})",
                msg, i, va, vb, diff, tol
            );
            std::process::abort();
        }
    }
}

#[unsafe(no_mangle)]
pub extern "C" fn nsl_exit(code: i64) {
    std::process::exit(code as i32);
}

#[cfg(test)]
mod tests {
    use super::is_close;

    #[test]
    fn a_nan_is_never_close() {
        // The `diff > tol` form this replaced passed every one of these.
        assert!(!is_close(f64::NAN, 1.0, 1e-5, 1e-8));
        assert!(!is_close(1.0, f64::NAN, 1e-5, 1e-8));
        assert!(!is_close(f64::NAN, f64::NAN, 1e-5, 1e-8));
        assert!(!is_close(1.0, 1.0, f64::NAN, 1e-8));
    }

    #[test]
    fn an_infinity_is_close_only_to_the_same_infinity() {
        assert!(is_close(f64::INFINITY, f64::INFINITY, 1e-5, 1e-8));
        assert!(is_close(f64::NEG_INFINITY, f64::NEG_INFINITY, 1e-5, 1e-8));
        assert!(!is_close(f64::INFINITY, f64::NEG_INFINITY, 1e-5, 1e-8));
        assert!(!is_close(1.0, f64::INFINITY, 1e-5, 1e-8));
        assert!(!is_close(f64::INFINITY, 1.0, 1e-5, 1e-8));
    }

    #[test]
    fn finite_values_keep_the_atol_rtol_contract() {
        assert!(is_close(1.0, 1.0, 0.0, 0.0));
        assert!(is_close(1.0 + 1e-9, 1.0, 0.0, 1e-8));
        assert!(is_close(100.5, 100.0, 1e-2, 0.0));
        assert!(!is_close(1.1, 1.0, 1e-5, 1e-8));
    }
}
