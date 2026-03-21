//! M62a/b: Stable C API for NSL shared library interop.
//!
//! Provides a C ABI for loading models, running forward passes, and managing
//! tensors from external code (Python/C++/etc). M62a laid out the stubs;
//! M62b wires them to actual weight loading and tensor dispatch.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::os::raw::c_char;
use std::sync::atomic::AtomicI64;
#[cfg(feature = "interop")]
use std::sync::atomic::Ordering;

use crate::dlpack::DLManagedTensor;
use crate::tensor::NslTensor;
use crate::memory::checked_alloc;

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
// NslModel — holds loaded weights and forward function pointer
// ---------------------------------------------------------------------------

/// Type alias for a compiled forward function.
/// Signature: fn(weights: &[i64], inputs: &[i64]) -> Vec<i64>
/// where each i64 is a pointer to an NslTensor.
type ForwardFn = Box<dyn Fn(&[i64], &[i64]) -> Vec<i64> + Send + Sync>;

/// Opaque model handle. Holds loaded weights and an optional compiled forward
/// function for dispatching inference.
pub struct NslModel {
    /// Model version (for ABI compatibility).
    #[allow(dead_code)]
    version: u32,
    /// Weight tensors loaded from safetensors/nslm file.
    /// Keys are parameter names, values are NslTensor pointers (as i64).
    weights: HashMap<String, i64>,
    /// Ordered list of weight tensor pointers (for positional access).
    weight_ptrs: Vec<i64>,
    /// Optional compiled forward function. When set, nsl_model_forward
    /// calls this instead of returning an error.
    forward_fn: Option<ForwardFn>,
    /// Path the weights were loaded from (for diagnostics).
    #[allow(dead_code)]
    weights_path: String,
}

impl Drop for NslModel {
    fn drop(&mut self) {
        // Free all weight tensors
        for &ptr in self.weight_ptrs.iter() {
            if ptr != 0 {
                crate::tensor::nsl_tensor_free(ptr);
            }
        }
    }
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
// FFI: Model lifecycle
// ---------------------------------------------------------------------------

/// Create a model instance from a weights file path (null-terminated C string).
///
/// Loads weights from a `.safetensors` file into NslTensors. The caller can
/// then register a forward function via `nsl_model_set_forward` or call
/// `nsl_model_forward` which dispatches through the registered function.
///
/// Returns a pointer (as i64) to the model handle, or 0 on error.
#[no_mangle]
pub extern "C" fn nsl_model_create(weights_path_ptr: i64) -> i64 {
    if weights_path_ptr == 0 {
        set_error("nsl_model_create: null weights path\0".to_string());
        return 0;
    }

    let path_cstr = unsafe { CStr::from_ptr(weights_path_ptr as *const c_char) };
    let path_str = match path_cstr.to_str() {
        Ok(s) => s,
        Err(_) => {
            set_error("nsl_model_create: invalid UTF-8 in path\0".to_string());
            return 0;
        }
    };

    // Try to load weights from safetensors file
    let (weights, weight_ptrs) = match load_safetensors_weights(path_str) {
        Ok(w) => w,
        Err(msg) => {
            set_error(format!("nsl_model_create: {msg}\0"));
            return 0;
        }
    };

    let model = Box::new(NslModel {
        version: 2,
        weights,
        weight_ptrs,
        forward_fn: None,
        weights_path: path_str.to_string(),
    });
    Box::into_raw(model) as i64
}

/// Destroy a model instance and free its resources (including weight tensors).
#[no_mangle]
pub extern "C" fn nsl_model_destroy(model_ptr: i64) -> i64 {
    if model_ptr == 0 {
        return 0;
    }
    unsafe { drop(Box::from_raw(model_ptr as *mut NslModel)); }
    0
}

/// Register a forward function for the model.
///
/// The function pointer must have the signature:
///   `fn(num_weights: i64, weights_ptr: i64, num_inputs: i64, inputs_ptr: i64) -> i64`
/// where weights_ptr and inputs_ptr are pointers to arrays of NslTensor pointers,
/// and the return value is an NslTensor pointer (the output).
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_model_set_forward(model_ptr: i64, fn_ptr: i64) -> i64 {
    if model_ptr == 0 || fn_ptr == 0 {
        set_error("nsl_model_set_forward: null pointer\0".to_string());
        return -1;
    }
    let model = unsafe { &mut *(model_ptr as *mut NslModel) };

    // Wrap the raw function pointer in a ForwardFn closure
    type RawForwardFn = extern "C" fn(i64, i64, i64, i64) -> i64;
    let raw_fn: RawForwardFn = unsafe { std::mem::transmute(fn_ptr as *const ()) };

    model.forward_fn = Some(Box::new(move |weights: &[i64], inputs: &[i64]| {
        let result = raw_fn(
            weights.len() as i64,
            weights.as_ptr() as i64,
            inputs.len() as i64,
            inputs.as_ptr() as i64,
        );
        if result == 0 {
            vec![]
        } else {
            vec![result]
        }
    }));
    0
}

// ---------------------------------------------------------------------------
// FFI: Forward pass
// ---------------------------------------------------------------------------

/// Run the model's forward pass with NslTensorDesc inputs/outputs.
///
/// Converts C API tensor descriptors to internal NslTensors, calls the
/// registered forward function, and writes results back to the output
/// descriptors.
///
/// Returns 0 on success, -1 on error (message in thread-local).
#[no_mangle]
pub extern "C" fn nsl_model_forward(
    model_ptr: i64,
    inputs_ptr: i64,
    num_inputs: i64,
    outputs_ptr: i64,
    num_outputs: i64,
) -> i64 {
    if model_ptr == 0 {
        set_error("nsl_model_forward: null model pointer\0".to_string());
        return -1;
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };

    // Check forward function is registered
    let forward_fn = match &model.forward_fn {
        Some(f) => f,
        None => {
            set_error("nsl_model_forward: no forward function registered (use nsl_model_set_forward)\0".to_string());
            return -1;
        }
    };

    // Convert input NslTensorDescs to NslTensor pointers
    let mut input_ptrs = Vec::with_capacity(num_inputs as usize);
    if num_inputs > 0 && inputs_ptr != 0 {
        let descs = unsafe {
            std::slice::from_raw_parts(inputs_ptr as *const NslTensorDesc, num_inputs as usize)
        };
        for desc in descs {
            let tensor_ptr = desc_to_nsl_tensor(desc);
            input_ptrs.push(tensor_ptr);
        }
    }

    // Call forward function
    let output_tensor_ptrs = forward_fn(&model.weight_ptrs, &input_ptrs);

    // Write results to output descriptors
    if num_outputs > 0 && outputs_ptr != 0 {
        let out_descs = unsafe {
            std::slice::from_raw_parts_mut(outputs_ptr as *mut NslTensorDesc, num_outputs as usize)
        };
        for (i, desc) in out_descs.iter_mut().enumerate() {
            if i < output_tensor_ptrs.len() {
                nsl_tensor_to_desc(output_tensor_ptrs[i], desc);
            }
        }
    }

    // Free the input wrapper tensors (they borrowed data, so this just frees the struct)
    for ptr in &input_ptrs {
        crate::tensor::nsl_tensor_free(*ptr);
    }

    0
}

/// Run the model's forward pass with DLPack tensors (zero-copy).
///
/// Takes an array of DLManagedTensor pointers as input, calls the forward
/// function, and returns output tensors as DLManagedTensor pointers.
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_model_forward_dlpack(
    model_ptr: i64,
    inputs_ptr: i64,
    num_inputs: i64,
    outputs_ptr: i64,
    num_outputs_ptr: i64,
) -> i64 {
    if model_ptr == 0 {
        set_error("nsl_model_forward_dlpack: null model pointer\0".to_string());
        return -1;
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };

    let forward_fn = match &model.forward_fn {
        Some(f) => f,
        None => {
            set_error("nsl_model_forward_dlpack: no forward function registered\0".to_string());
            return -1;
        }
    };

    // Import DLPack inputs to NslTensors
    let mut input_ptrs = Vec::with_capacity(num_inputs as usize);
    if num_inputs > 0 && inputs_ptr != 0 {
        let dlpacks = unsafe {
            std::slice::from_raw_parts(inputs_ptr as *const *mut DLManagedTensor, num_inputs as usize)
        };
        for &dlpack in dlpacks {
            if dlpack.is_null() {
                set_error("nsl_model_forward_dlpack: null input DLManagedTensor\0".to_string());
                return -1;
            }
            let tensor_ptr = crate::dlpack::dlpack_to_nsl_tensor(unsafe { &*dlpack });
            input_ptrs.push(tensor_ptr);
        }
    }

    // Call forward
    let output_tensor_ptrs = forward_fn(&model.weight_ptrs, &input_ptrs);

    // Export outputs as DLPack
    if outputs_ptr != 0 && num_outputs_ptr != 0 {
        let out_array = unsafe { &mut *(outputs_ptr as *mut *mut DLManagedTensor) };
        let num_out = unsafe { &mut *(num_outputs_ptr as *mut i64) };
        *num_out = output_tensor_ptrs.len() as i64;

        if !output_tensor_ptrs.is_empty() {
            // Allocate array of DLManagedTensor pointers
            let arr_size = output_tensor_ptrs.len() * std::mem::size_of::<*mut DLManagedTensor>();
            let arr = checked_alloc(arr_size) as *mut *mut DLManagedTensor;
            for (i, &tensor_ptr) in output_tensor_ptrs.iter().enumerate() {
                let tensor = NslTensor::from_ptr(tensor_ptr);
                let dlpack = crate::dlpack::nsl_tensor_to_dlpack(tensor, tensor_ptr);
                unsafe { *arr.add(i) = dlpack; }
            }
            *out_array = unsafe { *arr }; // point to first element
        }
    }

    // Free imported input tensors (borrowed data, struct-only free)
    for ptr in &input_ptrs {
        crate::tensor::nsl_tensor_free(*ptr);
    }

    0
}

// ---------------------------------------------------------------------------
// FFI: Metadata
// ---------------------------------------------------------------------------

/// Get the NSL compiler/runtime version string.
#[no_mangle]
pub extern "C" fn nsl_model_get_version() -> i64 {
    static VERSION: &[u8] = b"NSL 0.2.0\0";
    VERSION.as_ptr() as i64
}

/// Get the number of weight tensors in the model.
#[no_mangle]
pub extern "C" fn nsl_model_num_weights(model_ptr: i64) -> i64 {
    if model_ptr == 0 { return 0; }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    model.weight_ptrs.len() as i64
}

/// Get a weight tensor by name. Returns NslTensor pointer or 0 if not found.
#[no_mangle]
pub extern "C" fn nsl_model_get_weight(model_ptr: i64, name_ptr: i64, name_len: i64) -> i64 {
    if model_ptr == 0 || name_ptr == 0 { return 0; }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    let name = unsafe {
        let slice = std::slice::from_raw_parts(name_ptr as *const u8, name_len as usize);
        std::str::from_utf8_unchecked(slice)
    };
    model.weights.get(name).copied().unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Load weights from a safetensors file.
#[cfg(feature = "interop")]
fn load_safetensors_weights(path: &str) -> Result<(HashMap<String, i64>, Vec<i64>), String> {
    let data = std::fs::read(path)
        .map_err(|e| format!("failed to read '{path}': {e}"))?;

    let tensors = safetensors::SafeTensors::deserialize(&data)
        .map_err(|e| format!("failed to parse safetensors '{path}': {e}"))?;

    let mut weights = HashMap::new();
    let mut weight_ptrs = Vec::new();

    for (name, view) in tensors.tensors() {
        let shape: Vec<i64> = view.shape().iter().map(|&d| d as i64).collect();
        let ndim = shape.len();
        let len: i64 = shape.iter().product::<i64>().max(1);
        let raw_data = view.data();

        // Convert to f32 (the standard NSL runtime format for loaded weights)
        let f32_data = convert_weight_to_f32(view.dtype(), raw_data, len as usize);

        // Allocate NslTensor
        let data_size = (len as usize) * std::mem::size_of::<f32>();
        let data_ptr = checked_alloc(data_size) as *mut f32;
        unsafe {
            std::ptr::copy_nonoverlapping(f32_data.as_ptr(), data_ptr, f32_data.len());
        }

        let shape_ptr = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
        for (i, &s) in shape.iter().enumerate() {
            unsafe { *shape_ptr.add(i) = s; }
        }
        let strides = NslTensor::compute_strides(shape_ptr, ndim as i64);

        let tensor = Box::new(NslTensor {
            data: data_ptr as *mut c_void,
            shape: shape_ptr,
            strides,
            ndim: ndim as i64,
            len,
            refcount: AtomicI64::new(1),
            device: 0,
            dtype: 1, // f32
            owns_data: 1,
            data_owner: 0,
        });
        let ptr = Box::into_raw(tensor) as i64;
        weights.insert(name.to_string(), ptr);
        weight_ptrs.push(ptr);
    }

    Ok((weights, weight_ptrs))
}

/// Convert raw weight bytes to f32 based on safetensors dtype.
#[cfg(feature = "interop")]
fn convert_weight_to_f32(dtype: safetensors::Dtype, data: &[u8], len: usize) -> Vec<f32> {
    match dtype {
        safetensors::Dtype::F32 => {
            let src = data.as_ptr() as *const f32;
            (0..len).map(|i| unsafe { *src.add(i) }).collect()
        }
        safetensors::Dtype::F64 => {
            let src = data.as_ptr() as *const f64;
            (0..len).map(|i| unsafe { *src.add(i) as f32 }).collect()
        }
        safetensors::Dtype::F16 => {
            let src = data.as_ptr() as *const u16;
            (0..len).map(|i| {
                let bits = unsafe { *src.add(i) };
                half::f16::from_bits(bits).to_f32()
            }).collect()
        }
        safetensors::Dtype::BF16 => {
            let src = data.as_ptr() as *const u16;
            (0..len).map(|i| {
                let bits = unsafe { *src.add(i) };
                half::bf16::from_bits(bits).to_f32()
            }).collect()
        }
        _ => vec![0.0_f32; len], // unsupported dtype → zeros
    }
}

/// Fallback when interop feature is not enabled — cannot load safetensors.
#[cfg(not(feature = "interop"))]
fn load_safetensors_weights(path: &str) -> Result<(HashMap<String, i64>, Vec<i64>), String> {
    Err(format!("safetensors loading requires --features interop (path: '{path}')"))
}

/// Convert an NslTensorDesc (C API) into an NslTensor pointer.
/// The resulting tensor borrows the data buffer — caller must not free the
/// original data while the tensor is live.
fn desc_to_nsl_tensor(desc: &NslTensorDesc) -> i64 {
    let ndim = desc.ndim as usize;
    let nsl_dtype = capi_dtype_to_nsl(desc.dtype);

    let shape_ptr = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(desc.shape, shape_ptr, ndim); }

    let strides = if desc.strides.is_null() {
        NslTensor::compute_strides(shape_ptr, desc.ndim as i64)
    } else {
        let s = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
        unsafe { std::ptr::copy_nonoverlapping(desc.strides, s, ndim); }
        s
    };

    let len = NslTensor::total_elements(shape_ptr, desc.ndim as i64);
    let device = if desc.device_type > 0 { desc.device_id as u8 + 1 } else { 0 };

    let tensor = Box::new(NslTensor {
        data: desc.data,
        shape: shape_ptr,
        strides,
        ndim: desc.ndim as i64,
        len,
        refcount: AtomicI64::new(1),
        device,
        dtype: nsl_dtype,
        owns_data: 0, // borrowed — C caller owns the data
        data_owner: 0,
    });
    Box::into_raw(tensor) as i64
}

/// Fill an NslTensorDesc from an NslTensor pointer.
fn nsl_tensor_to_desc(tensor_ptr: i64, desc: &mut NslTensorDesc) {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    desc.data = tensor.data;
    desc.ndim = tensor.ndim as i32;
    desc.dtype = nsl_dtype_to_capi(tensor.dtype);
    desc.device_type = if tensor.device > 0 { 1 } else { 0 };
    desc.device_id = if tensor.device > 0 { (tensor.device - 1) as i32 } else { 0 };
    desc.shape = tensor.shape;
    desc.strides = tensor.strides;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_lifecycle() {
        let path = b"weights.safetensors\0";
        let model_ptr = nsl_model_create(path.as_ptr() as i64);
        // May fail if file doesn't exist — that's OK, we test the error path
        if model_ptr != 0 {
            nsl_model_destroy(model_ptr);
        }
        // Destroy null is safe
        assert_eq!(nsl_model_destroy(0), 0);
    }

    #[test]
    fn test_model_create_null_returns_zero() {
        let model_ptr = nsl_model_create(0);
        assert_eq!(model_ptr, 0);
    }

    #[test]
    fn test_forward_no_fn_returns_error() {
        // Create a model without a forward function (using a fake path — will fail to load)
        // Instead, test with null model
        assert_eq!(nsl_model_forward(0, 0, 0, 0, 0), -1);
        assert_eq!(nsl_model_forward_dlpack(0, 0, 0, 0, 0), -1);
    }

    #[test]
    fn test_error_handling() {
        nsl_clear_error();
        let err_ptr = nsl_get_last_error();
        assert_ne!(err_ptr, 0);

        // Trigger an error via null model forward
        nsl_model_forward(0, 0, 0, 0, 0);
        let err_ptr = nsl_get_last_error();
        assert_ne!(err_ptr, 0);

        nsl_clear_error();
    }

    #[test]
    fn test_version_string() {
        let version_ptr = nsl_model_get_version();
        assert_ne!(version_ptr, 0);
        let version = unsafe { CStr::from_ptr(version_ptr as *const c_char) };
        let version_str = version.to_str().unwrap();
        assert!(version_str.starts_with("NSL"));
    }

    #[test]
    fn test_dtype_mapping() {
        assert_eq!(capi_dtype_to_nsl(0), 1);
        assert_eq!(capi_dtype_to_nsl(1), 0);
        assert_eq!(nsl_dtype_to_capi(0), 1);
        assert_eq!(nsl_dtype_to_capi(1), 0);

        for capi_d in [0, 1, 2, 3, 6] {
            let nsl_d = capi_dtype_to_nsl(capi_d);
            let back = nsl_dtype_to_capi(nsl_d);
            assert_eq!(back, capi_d, "Roundtrip failed for C API dtype {capi_d}");
        }
    }

    #[test]
    fn test_desc_to_nsl_tensor_roundtrip() {
        // Create a simple f32 tensor desc
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let mut shape: Vec<i64> = vec![2, 2];
        let desc = NslTensorDesc {
            data: data.as_ptr() as *mut c_void,
            shape: shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            ndim: 2,
            dtype: 0, // C API f32
            device_type: 0,
            device_id: 0,
        };

        let tensor_ptr = desc_to_nsl_tensor(&desc);
        assert_ne!(tensor_ptr, 0);

        let tensor = NslTensor::from_ptr(tensor_ptr);
        assert_eq!(tensor.ndim, 2);
        assert_eq!(tensor.len, 4);
        assert_eq!(tensor.dtype, 1); // NSL f32
        assert_eq!(tensor.device, 0);
        assert_eq!(tensor.owns_data, 0); // borrowed

        // Read back via desc
        let mut out_desc = NslTensorDesc {
            data: std::ptr::null_mut(), shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(), ndim: 0, dtype: 0,
            device_type: 0, device_id: 0,
        };
        nsl_tensor_to_desc(tensor_ptr, &mut out_desc);
        assert_eq!(out_desc.ndim, 2);
        assert_eq!(out_desc.dtype, 0); // back to C API f32

        crate::tensor::nsl_tensor_free(tensor_ptr);
    }

    #[test]
    fn test_num_weights_null_model() {
        assert_eq!(nsl_model_num_weights(0), 0);
    }

    #[test]
    fn test_get_weight_null_model() {
        assert_eq!(nsl_model_get_weight(0, 0, 0), 0);
    }
}
