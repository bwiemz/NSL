//! M62a/b: Stable C API for NSL shared library interop.
//!
//! Provides a C ABI for loading models, running forward passes, and managing
//! tensors from external code (Python/C++/etc). M62a laid out the stubs;
//! M62b wires them to actual weight loading and tensor dispatch.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::os::raw::c_char;
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
    /// Whether to record the tape during the next forward pass (for backward).
    grad_enabled: bool,
    /// Output tensor pointers saved from the most recent grad-enabled forward pass.
    /// Used to seed the backward pass.
    last_forward_outputs: Vec<i64>,
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
        grad_enabled: false,
        last_forward_outputs: Vec::new(),
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

    // If grad recording is enabled, start the tape over the weight parameters
    // so that the backward pass can replay it.
    let grad_enabled = model.grad_enabled;
    if grad_enabled {
        let param_list = crate::list::nsl_list_new();
        for &wptr in &model.weight_ptrs {
            crate::list::nsl_list_push(param_list, wptr);
        }
        crate::autodiff::nsl_tape_start(param_list);
        crate::list::nsl_list_free(param_list);
    }

    // Call forward function
    let output_tensor_ptrs = forward_fn(&model.weight_ptrs, &input_ptrs);

    // Save forward outputs for the backward pass (before writing to output descs,
    // since nsl_tensor_to_desc only borrows the pointer — the tensor stays alive).
    if grad_enabled {
        let model_mut = unsafe { &mut *(model_ptr as *mut NslModel) };
        model_mut.last_forward_outputs = output_tensor_ptrs.clone();
    }

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

        let tensor = Box::new(NslTensor::new(
            data_ptr as *mut c_void,
            shape_ptr,
            strides,
            ndim as i64,
            len,
            0,
            1,
            1,
            0,
        ));
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

/// Public wrapper for `desc_to_nsl_tensor` — used by disaggregated workers
/// to convert model forward outputs back to NslTensor pointers.
pub fn desc_to_nsl_tensor_pub(desc: &NslTensorDesc) -> i64 {
    desc_to_nsl_tensor(desc)
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

    let tensor = Box::new(NslTensor::new(
        desc.data,
        shape_ptr,
        strides,
        desc.ndim as i64,
        len,
        device,
        nsl_dtype,
        0,
        0,
    ));
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
// M62b: Backward pass FFI
// ---------------------------------------------------------------------------

/// Enable or disable tape recording during subsequent forward passes.
///
/// When `enable` is non-zero, the next call to `nsl_model_forward` will
/// start recording the autodiff tape over the model's weight tensors.
/// Call `nsl_model_backward` afterwards to replay the tape and compute
/// parameter gradients.
///
/// Returns 0 on success, -1 if model_ptr is null.
#[no_mangle]
pub extern "C" fn nsl_model_enable_grad(model_ptr: i64, enable: i64) -> i64 {
    if model_ptr == 0 {
        set_error("nsl_model_enable_grad: null model pointer\0".to_string());
        return -1;
    }
    let model = unsafe { &mut *(model_ptr as *mut NslModel) };
    model.grad_enabled = enable != 0;
    0
}

/// Run the model backward pass, computing gradients w.r.t. all weight parameters.
///
/// This function replays the tape recorded during the most recent grad-enabled
/// forward pass (`nsl_model_enable_grad` + `nsl_model_forward`), computing
/// ∂loss/∂param for every weight in the model.
///
/// Parameters:
///   model_ptr:            NslModel* handle
///   grad_outputs_ptr:     (reserved) upstream gradient tensors — currently unused;
///                         the loss scalar is taken directly from the forward output
///   num_grad_outputs:     number of upstream gradient tensors (reserved)
///   grad_inputs_ptr:      pointer to a caller-allocated array of NslTensorDesc that
///                         will be filled with one gradient descriptor per weight;
///                         pass 0 to query the count only
///   num_grad_inputs_ptr:  output — written with the number of weight gradients
///
/// Returns: 0 on success, negative on error (message in thread-local).
///
/// Typical usage (from Python/C bridge):
///   nsl_model_enable_grad(model, 1);
///   nsl_model_forward(model, inputs, n, outputs, n_out);
///   nsl_model_backward(model, 0, 0, grad_buf, &num_grads);
///   nsl_model_enable_grad(model, 0);  // optional: disable for inference
#[no_mangle]
pub extern "C" fn nsl_model_backward(
    model_ptr: i64,
    _grad_outputs_ptr: i64,
    _num_grad_outputs: i64,
    grad_inputs_ptr: i64,
    num_grad_inputs_ptr: i64,
) -> i64 {
    if model_ptr == 0 {
        set_error("nsl_model_backward: null model pointer\0".to_string());
        return -1;
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };

    if !model.grad_enabled {
        set_error("nsl_model_backward: grad not enabled — call nsl_model_enable_grad(model, 1) before nsl_model_forward\0".to_string());
        return -1;
    }

    if model.last_forward_outputs.is_empty() {
        set_error("nsl_model_backward: no forward pass recorded — call nsl_model_forward with grad enabled first\0".to_string());
        return -1;
    }

    // Build the param list for tape_backward — same weights that were registered at tape_start.
    let param_list = crate::list::nsl_list_new();
    for &wptr in &model.weight_ptrs {
        crate::list::nsl_list_push(param_list, wptr);
    }

    // Use the first forward output as the loss tensor.
    // For standard training this is a scalar loss; for multi-output models the
    // caller should pass grad_outputs to scale/combine, but that is reserved for
    // a future revision. Chain-rule seeding from the scalar loss is sufficient
    // for all current NSL training patterns.
    let loss_ptr = model.last_forward_outputs[0];

    // Run backward — this drains the tape ops and returns one gradient per param.
    let grads_list = crate::autodiff::nsl_tape_backward(loss_ptr, param_list);

    // Reset tape to clean state (recording=false, ops/param_set cleared).
    crate::autodiff::nsl_tape_stop();

    let num_params = model.weight_ptrs.len();

    // Write gradient count to caller's output pointer.
    if num_grad_inputs_ptr != 0 {
        unsafe { *(num_grad_inputs_ptr as *mut i64) = num_params as i64 };
    }

    // Fill gradient descriptors if the caller provided a buffer.
    if grad_inputs_ptr != 0 && num_params > 0 {
        let out_descs = unsafe {
            std::slice::from_raw_parts_mut(grad_inputs_ptr as *mut NslTensorDesc, num_params)
        };
        for (i, desc) in out_descs.iter_mut().enumerate() {
            let grad_ptr = crate::list::nsl_list_get(grads_list, i as i64);
            if grad_ptr != 0 {
                nsl_tensor_to_desc(grad_ptr, desc);
            }
        }
    }

    // Clear the saved forward outputs now that backward is done.
    // SAFETY: we hold a shared reference to model above but no longer read it,
    // and last_forward_outputs is only mutated by this function and nsl_model_forward
    // (never concurrently — the C API is single-threaded per the M62 contract).
    {
        let model_mut = unsafe { &mut *(model_ptr as *mut NslModel) };
        model_mut.last_forward_outputs.clear();
    }

    // The gradient tensors in grads_list are referenced by the descriptors we
    // filled above (the descriptors borrow the tensor data pointers). We free the
    // list wrapper but leave the tensors alive — the caller owns them via the
    // NslTensorDesc.data pointers and is responsible for freeing them.
    crate::list::nsl_list_free(grads_list);
    crate::list::nsl_list_free(param_list);

    0
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

    #[test]
    fn test_enable_grad_null_model() {
        assert_eq!(nsl_model_enable_grad(0, 1), -1);
    }

    #[test]
    fn test_backward_null_model() {
        assert_eq!(nsl_model_backward(0, 0, 0, 0, 0), -1);
    }

    #[test]
    fn test_backward_without_enable_grad() {
        // Build a minimal NslModel directly (no weights file needed).
        let model = Box::new(NslModel {
            version: 2,
            weights: HashMap::new(),
            weight_ptrs: vec![],
            forward_fn: None,
            weights_path: String::new(),
            grad_enabled: false,
            last_forward_outputs: vec![],
        });
        let model_ptr = Box::into_raw(model) as i64;

        // Backward without enabling grad must fail with a meaningful error.
        let ret = nsl_model_backward(model_ptr, 0, 0, 0, 0);
        assert_eq!(ret, -1);

        // Error message should mention enable_grad.
        let err_ptr = nsl_get_last_error() as *const std::os::raw::c_char;
        let err_msg = unsafe { CStr::from_ptr(err_ptr) }.to_str().unwrap_or("");
        assert!(err_msg.contains("grad not enabled"), "unexpected error: {err_msg}");

        unsafe { drop(Box::from_raw(model_ptr as *mut NslModel)); }
        nsl_clear_error();
    }

    #[test]
    fn test_backward_without_forward() {
        // Build a model with grad enabled but no prior forward pass.
        let model = Box::new(NslModel {
            version: 2,
            weights: HashMap::new(),
            weight_ptrs: vec![],
            forward_fn: None,
            weights_path: String::new(),
            grad_enabled: true,
            last_forward_outputs: vec![],
        });
        let model_ptr = Box::into_raw(model) as i64;

        let ret = nsl_model_backward(model_ptr, 0, 0, 0, 0);
        assert_eq!(ret, -1);

        let err_ptr = nsl_get_last_error() as *const std::os::raw::c_char;
        let err_msg = unsafe { CStr::from_ptr(err_ptr) }.to_str().unwrap_or("");
        assert!(err_msg.contains("no forward pass recorded"), "unexpected error: {err_msg}");

        unsafe { drop(Box::from_raw(model_ptr as *mut NslModel)); }
        nsl_clear_error();
    }

    #[test]
    fn test_backward_with_tape_e2e() {
        use std::ffi::c_void;
        use crate::memory::checked_alloc;
        use crate::tensor::NslTensor;

        // Build a single-weight model: weight = [2.0] (f64 scalar).
        // forward(w) = w * 3  → loss = w * 3
        // d(loss)/d(w) = 3
        let weight_data = checked_alloc(std::mem::size_of::<f64>()) as *mut f64;
        unsafe { *weight_data = 2.0_f64 };
        let weight_shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
        unsafe { *weight_shape = 1 };
        let weight_strides = NslTensor::compute_strides(weight_shape, 1);
        let weight_tensor = Box::new(NslTensor::new(
            weight_data as *mut c_void,
            weight_shape,
            weight_strides,
            1,
            1,
            0,  // CPU
            0,  // f64
            1,  // owns_data
            0,
        ));
        let weight_ptr = Box::into_raw(weight_tensor) as i64;

        // Forward fn: loss = weight * 3
        let fwd: ForwardFn = Box::new(move |weights: &[i64], _inputs: &[i64]| {
            let w = weights[0];
            let out = crate::tensor::nsl_tensor_mul_scalar(w, 3.0_f64, 0);
            vec![out]
        });

        let mut weights = HashMap::new();
        weights.insert("w".to_string(), weight_ptr);

        let model = Box::new(NslModel {
            version: 2,
            weights,
            weight_ptrs: vec![weight_ptr],
            forward_fn: Some(fwd),
            weights_path: String::new(),
            grad_enabled: false,
            last_forward_outputs: vec![],
        });
        let model_ptr = Box::into_raw(model) as i64;

        // Enable grad, run forward, run backward.
        assert_eq!(nsl_model_enable_grad(model_ptr, 1), 0);
        assert_eq!(nsl_model_forward(model_ptr, 0, 0, 0, 0), 0);

        let mut num_grads: i64 = 0;
        let ret = nsl_model_backward(model_ptr, 0, 0, 0, &mut num_grads as *mut i64 as i64);
        assert_eq!(ret, 0, "backward returned error");
        assert_eq!(num_grads, 1, "expected 1 gradient (one weight)");

        // Cleanup: destroy the model (frees weight tensor).
        nsl_model_destroy(model_ptr);
    }
}
