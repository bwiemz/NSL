//! M62a: Stable C API stubs for NSL shared library interop.
//!
//! These are the foundation functions for the `nsl build --shared-lib` C ABI.
//! M62a provides stubs for the lifecycle and error handling; actual model loading
//! and forward pass logic will be wired in M62b when shared library emission lands.

use std::cell::RefCell;
use std::ffi::{c_void, CStr};
use std::os::raw::c_char;

// DLManagedTensor will be used in M62b for nsl_model_forward_dlpack implementation.
#[allow(unused_imports)]
use crate::dlpack::DLManagedTensor;

// ---------------------------------------------------------------------------
// NslTensorDesc — C API tensor descriptor
// ---------------------------------------------------------------------------

/// Tensor descriptor matching the C header from the M62 spec.
///
/// Note: The C API uses dtype convention 0=f32, 1=f64 — this is **different**
/// from NSL's internal convention (0=f64, 1=f32). The conversion functions
/// `capi_dtype_to_nsl` and `nsl_dtype_to_capi` handle the mapping.
#[repr(C)]
pub struct NslTensorDesc {
    pub data: *mut c_void,
    pub shape: *mut i64,
    pub strides: *mut i64,
    pub ndim: i32,
    /// C API dtype: 0=f32, 1=f64, 2=f16, 3=bf16, 4=int32, 5=int64, 6=int8, 7=uint8
    pub dtype: i32,
    /// 0=CPU, 1=CUDA
    pub device_type: i32,
    /// GPU index (0 for CPU)
    pub device_id: i32,
}

// ---------------------------------------------------------------------------
// Dtype mapping between C API and NSL internal
// ---------------------------------------------------------------------------

/// Convert C API dtype (0=f32, 1=f64) to NSL internal dtype (0=f64, 1=f32).
#[allow(dead_code)]
pub fn capi_dtype_to_nsl(capi_dtype: i32) -> u16 {
    match capi_dtype {
        0 => 1, // C API f32 -> NSL 1 (f32)
        1 => 0, // C API f64 -> NSL 0 (f64)
        2 => 2, // f16
        3 => 3, // bf16
        6 => 4, // int8
        _ => 0, // fallback to f64
    }
}

/// Convert NSL internal dtype (0=f64, 1=f32) to C API dtype (0=f32, 1=f64).
#[allow(dead_code)]
pub fn nsl_dtype_to_capi(nsl_dtype: u16) -> i32 {
    match nsl_dtype {
        0 => 1, // NSL f64 -> C API 1 (f64)
        1 => 0, // NSL f32 -> C API 0 (f32)
        2 => 2, // f16
        3 => 3, // bf16
        4 => 6, // int8
        _ => 1, // fallback to f64
    }
}

// ---------------------------------------------------------------------------
// Opaque model handle (stub)
// ---------------------------------------------------------------------------

/// Opaque model handle. In M62a this is a placeholder; M62b will wire it
/// to actual compiled model code loaded from a shared library.
pub struct NslModel {
    /// Placeholder to prevent zero-sized type issues.
    _version: u32,
}

// ---------------------------------------------------------------------------
// Thread-local error string
// ---------------------------------------------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
}

fn set_error(msg: String) {
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(msg));
}

// ---------------------------------------------------------------------------
// FFI: Error handling
// ---------------------------------------------------------------------------

/// Get the last error message. Returns a null-terminated C string pointer,
/// or a pointer to an empty string if no error is set.
///
/// The returned pointer is valid until the next call to any C API function
/// on the same thread.
#[no_mangle]
pub extern "C" fn nsl_get_last_error() -> i64 {
    LAST_ERROR.with(|e| {
        let borrow = e.borrow();
        match borrow.as_deref() {
            Some(msg) => msg.as_ptr() as i64,
            None => c"".as_ptr() as i64,
        }
    })
}

/// Clear the last error message.
#[no_mangle]
pub extern "C" fn nsl_clear_error() -> i64 {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
    0
}

// ---------------------------------------------------------------------------
// FFI: Model lifecycle (stubs)
// ---------------------------------------------------------------------------

/// Create a model instance from a weights file path.
///
/// M62a stub: allocates a placeholder NslModel. Returns a pointer (as i64)
/// to the model handle, or 0 on error.
#[no_mangle]
pub extern "C" fn nsl_model_create(weights_path_ptr: i64) -> i64 {
    if weights_path_ptr == 0 {
        set_error("nsl_model_create: null weights path\0".to_string());
        return 0;
    }

    // Validate that the path is a valid C string (stub — we don't load anything yet).
    let _path = unsafe { CStr::from_ptr(weights_path_ptr as *const c_char) };

    let model = Box::new(NslModel { _version: 1 });
    Box::into_raw(model) as i64
}

/// Destroy a model instance and free its resources.
#[no_mangle]
pub extern "C" fn nsl_model_destroy(model_ptr: i64) -> i64 {
    if model_ptr == 0 {
        return 0;
    }
    unsafe { drop(Box::from_raw(model_ptr as *mut NslModel)); }
    0
}

// ---------------------------------------------------------------------------
// FFI: Forward pass (stubs)
// ---------------------------------------------------------------------------

/// Run the model's forward pass with NslTensorDesc inputs/outputs.
///
/// M62a stub: always returns 0 (success) without doing any computation.
/// M62b will wire this to the actual compiled forward function.
#[no_mangle]
pub extern "C" fn nsl_model_forward(
    model_ptr: i64,
    _inputs_ptr: i64,
    _num_inputs: i64,
    _outputs_ptr: i64,
    _num_outputs: i64,
) -> i64 {
    if model_ptr == 0 {
        set_error("nsl_model_forward: null model pointer\0".to_string());
        return -1;
    }
    // Stub: success, no computation.
    0
}

/// Run the model's forward pass with DLPack tensors (zero-copy).
///
/// M62a stub: always returns 0 (success) without doing any computation.
#[no_mangle]
pub extern "C" fn nsl_model_forward_dlpack(
    model_ptr: i64,
    _inputs_ptr: i64,
    _num_inputs: i64,
    _outputs_ptr: i64,
    _num_outputs_ptr: i64,
) -> i64 {
    if model_ptr == 0 {
        set_error("nsl_model_forward_dlpack: null model pointer\0".to_string());
        return -1;
    }
    // Stub: success, no computation.
    0
}

// ---------------------------------------------------------------------------
// FFI: Metadata
// ---------------------------------------------------------------------------

/// Get the NSL compiler/runtime version string.
///
/// Returns a pointer to a null-terminated static string.
#[no_mangle]
pub extern "C" fn nsl_model_get_version() -> i64 {
    static VERSION: &[u8] = b"NSL 0.2.0\0";
    VERSION.as_ptr() as i64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_lifecycle() {
        // Create with a valid C string
        let path = b"weights.safetensors\0";
        let model_ptr = nsl_model_create(path.as_ptr() as i64);
        assert_ne!(model_ptr, 0);

        // Destroy
        let rc = nsl_model_destroy(model_ptr);
        assert_eq!(rc, 0);

        // Destroy null is safe
        assert_eq!(nsl_model_destroy(0), 0);
    }

    #[test]
    fn test_model_create_null_returns_zero() {
        let model_ptr = nsl_model_create(0);
        assert_eq!(model_ptr, 0);
    }

    #[test]
    fn test_forward_stub() {
        let path = b"test.safetensors\0";
        let model_ptr = nsl_model_create(path.as_ptr() as i64);
        assert_ne!(model_ptr, 0);

        let rc = nsl_model_forward(model_ptr, 0, 0, 0, 0);
        assert_eq!(rc, 0);

        let rc = nsl_model_forward_dlpack(model_ptr, 0, 0, 0, 0);
        assert_eq!(rc, 0);

        nsl_model_destroy(model_ptr);
    }

    #[test]
    fn test_forward_null_model_returns_error() {
        assert_eq!(nsl_model_forward(0, 0, 0, 0, 0), -1);
        assert_eq!(nsl_model_forward_dlpack(0, 0, 0, 0, 0), -1);
    }

    #[test]
    fn test_error_handling() {
        // Clear first
        nsl_clear_error();

        // No error initially — returns empty string pointer
        let err_ptr = nsl_get_last_error();
        assert_ne!(err_ptr, 0);

        // Trigger an error via null model forward
        nsl_model_forward(0, 0, 0, 0, 0);
        let err_ptr = nsl_get_last_error();
        assert_ne!(err_ptr, 0);

        // Clear and verify
        nsl_clear_error();
    }

    #[test]
    fn test_version_string() {
        let version_ptr = nsl_model_get_version();
        assert_ne!(version_ptr, 0);

        let version = unsafe { CStr::from_ptr(version_ptr as *const c_char) };
        let version_str = version.to_str().unwrap();
        assert!(version_str.starts_with("NSL"), "Version should start with 'NSL', got: {version_str}");
    }

    #[test]
    fn test_dtype_mapping() {
        // C API: 0=f32 -> NSL: 1 (f32)
        assert_eq!(capi_dtype_to_nsl(0), 1);
        // C API: 1=f64 -> NSL: 0 (f64)
        assert_eq!(capi_dtype_to_nsl(1), 0);

        // NSL: 0 (f64) -> C API: 1
        assert_eq!(nsl_dtype_to_capi(0), 1);
        // NSL: 1 (f32) -> C API: 0
        assert_eq!(nsl_dtype_to_capi(1), 0);

        // Roundtrip
        for capi_d in [0, 1, 2, 3, 6] {
            let nsl_d = capi_dtype_to_nsl(capi_d);
            let back = nsl_dtype_to_capi(nsl_d);
            assert_eq!(back, capi_d, "Roundtrip failed for C API dtype {capi_d}");
        }
    }
}
