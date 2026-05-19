//! M62a/b: Stable C API for NSL shared library interop.
//!
//! Provides a C ABI for loading models, running forward passes, and managing
//! tensors from external code (Python/C++/etc). M62a laid out the stubs;
//! M62b wires them to actual weight loading and tensor dispatch.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::{c_void, CStr};
use std::os::raw::c_char;

use crate::dlpack::DLManagedTensor;
use crate::tensor::NslTensor;
use crate::memory::checked_alloc;

pub mod exports;

// ---------------------------------------------------------------------------
// NslTensorDesc — C API tensor descriptor
// ---------------------------------------------------------------------------

/// Tensor descriptor matching the C header from the M62 spec.
///
/// Note: The C API uses dtype convention 0=f32, 1=f64 — this is **different**
/// from NSL's internal convention (0=f64, 1=f32). The conversion functions
/// `capi_dtype_to_nsl` and `nsl_dtype_to_capi` handle the mapping.
#[repr(C)]
#[derive(Default)]
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
    /// Dispatch table for `@export`ed functions. Populated eagerly at create
    /// time via `nsl_model_create_with_lib`. `None` on the legacy
    /// `nsl_model_create` path that doesn't carry the .so path.
    exports: Option<crate::c_api::exports::ExportRegistry>,
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

// `CString` (not `String`) so the pointer returned by `nsl_get_last_error`
// is guaranteed null-terminated. With plain `String` the C-side read would
// run past the last byte into whatever happened to be next on the heap,
// yielding intermittent garbage like `"hello from wrapper[�JV"` on some
// platforms' allocator layouts.
thread_local! {
    static LAST_ERROR: RefCell<Option<std::ffi::CString>> = const { RefCell::new(None) };
}

fn set_error(msg: String) {
    // `new` rejects interior nulls — strip them instead of panicking in the
    // FFI surface.
    let cleaned: String = msg.chars().filter(|c| *c != '\0').collect();
    let cstr = std::ffi::CString::new(cleaned).expect("CString::new after \\0 strip");
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(cstr));
}

fn capi_trace(msg: impl AsRef<str>) {
    if std::env::var_os("NSL_CAPI_TRACE").is_some() {
        eprintln!("[nsl-capi] {}", msg.as_ref());
    }
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
        match borrow.as_ref() {
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

/// Set the thread-local error string from a null-terminated C string pointer.
/// Used by Cranelift-emitted `@export` wrappers; they can't call the Rust-typed
/// `set_error(String)` directly.
///
/// # Arguments
/// * `msg_ptr` - A pointer (as i64) to a null-terminated C string, or 0 for no-op.
#[no_mangle]
pub extern "C" fn nsl_set_error_cstr(msg_ptr: i64) {
    if msg_ptr == 0 {
        return;
    }
    let msg = unsafe {
        CStr::from_ptr(msg_ptr as *const c_char)
            .to_string_lossy()
            .into_owned()
    };
    set_error(msg);
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
    capi_trace(format!("model_create path={path_str}"));

    // Try to load weights from safetensors file
    let (weights, weight_ptrs) = match load_safetensors_weights(path_str) {
        Ok(w) => w,
        Err(msg) => {
            set_error(format!("nsl_model_create: {msg}\0"));
            return 0;
        }
    };
    capi_trace(format!("model_create loaded_weights={}", weight_ptrs.len()));

    let model = Box::new(NslModel {
        version: 2,
        weights,
        weight_ptrs,
        forward_fn: None,
        weights_path: path_str.to_string(),
        grad_enabled: false,
        last_forward_outputs: Vec::new(),
        exports: None,
    });
    Box::into_raw(model) as i64
}

/// Create a model and eagerly populate the export dispatch table from
/// the model's own shared library.
///
/// `weights_path_ptr` — null-terminated C string, same semantics as
///                      `nsl_model_create`.
/// `lib_path_ptr`     — null-terminated C string pointing at the .so/
///                      .dll/.dylib file the model was built from. The
///                      registry dlopens this and caches export pointers.
///
/// Returns 0 on error (thread-local message set), or a model handle on
/// success.
#[no_mangle]
pub extern "C" fn nsl_model_create_with_lib(
    weights_path_ptr: i64,
    lib_path_ptr: i64,
) -> i64 {
    let model_ptr = nsl_model_create(weights_path_ptr);
    if model_ptr == 0 {
        return 0;
    }
    if lib_path_ptr == 0 {
        // Treat as legacy create — registry stays None.
        return model_ptr;
    }
    let path_cstr = unsafe { CStr::from_ptr(lib_path_ptr as *const c_char) };
    let path = std::path::PathBuf::from(path_cstr.to_string_lossy().into_owned());
    capi_trace(format!("model_create_with_lib lib={}", path.display()));
    match crate::c_api::exports::ExportRegistry::from_library_path(&path) {
        Ok(reg) => {
            let model = unsafe { &mut *(model_ptr as *mut NslModel) };
            model.exports = Some(reg);
            model_ptr
        }
        Err(e) => {
            set_error(format!("nsl_model_create_with_lib: {}\0", e));
            // Tear down the half-built model before returning 0.
            let _ = nsl_model_destroy(model_ptr);
            0
        }
    }
}

/// Number of registered exports. 0 if registry is None (legacy create
/// path) or the library had no @export functions.
#[no_mangle]
pub extern "C" fn nsl_model_export_count(model_ptr: i64) -> i64 {
    if model_ptr == 0 {
        return 0;
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    match &model.exports {
        Some(r) => r.len() as i64,
        None => 0,
    }
}

/// Dispatch an `@export`'d function by string name.
///
/// `name_ptr` must point at a null-terminated C string. Returns -1 and
/// sets the thread-local error if the name is not in the registry;
/// otherwise invokes the cached function pointer with the supplied
/// NslTensorDesc arrays and returns whatever the export returned.
#[no_mangle]
pub extern "C" fn nsl_model_call(
    model_ptr: i64,
    name_ptr: i64,
    inputs_ptr: i64,
    num_inputs: i64,
    outputs_ptr: i64,
    num_outputs: i64,
) -> i64 {
    if model_ptr == 0 || name_ptr == 0 {
        set_error("nsl_model_call: null model or name pointer\0".to_string());
        return -1;
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    let registry = match &model.exports {
        Some(r) => r,
        None => {
            set_error(
                "nsl_model_call: model created without export registry; \
                 use nsl_model_create_with_lib(weights, lib)\0"
                    .to_string(),
            );
            return -1;
        }
    };
    let name = unsafe { CStr::from_ptr(name_ptr as *const c_char) }
        .to_string_lossy()
        .into_owned();
    let fn_ptr = match registry.lookup(&name) {
        Some(p) => p,
        None => {
            let available = registry.available_names();
            set_error(format!(
                "nsl_model_call: export '{}' not in registry. Available: {:?}\0",
                name, available
            ));
            return -1;
        }
    };
    capi_trace(format!(
        "model_call name={} num_inputs={} num_outputs={}",
        name, num_inputs, num_outputs
    ));
    unsafe { fn_ptr(model_ptr, inputs_ptr, num_inputs, outputs_ptr, num_outputs) }
}

/// DLPack variant of `nsl_model_call`. Bridges `DLManagedTensor*`-ABI
/// inputs and outputs to the `NslTensorDesc*`-ABI export entry-point
/// and reuses the same dispatcher.
#[no_mangle]
pub extern "C" fn nsl_model_call_dlpack(
    model_ptr: i64,
    name_ptr: i64,
    dl_inputs_ptr: i64,
    num_inputs: i64,
    dl_outputs_ptr: i64,
    num_outputs: i64,
) -> i64 {
    if model_ptr == 0 || name_ptr == 0 {
        set_error("nsl_model_call_dlpack: null model or name pointer\0".to_string());
        return -1;
    }
    // Import DLPack inputs to NslTensors via the existing bridge.
    let input_tensor_ptrs: Vec<i64> = if num_inputs > 0 && dl_inputs_ptr != 0 {
        let dlpacks = unsafe {
            std::slice::from_raw_parts(
                dl_inputs_ptr as *const *mut DLManagedTensor,
                num_inputs as usize,
            )
        };
        dlpacks
            .iter()
            .map(|&dl| crate::dlpack::dlpack_to_nsl_tensor(unsafe { &*dl }))
            .collect()
    } else {
        Vec::new()
    };
    let mut input_descs: Vec<NslTensorDesc> = input_tensor_ptrs
        .iter()
        .map(|&p| {
            let mut desc = NslTensorDesc::default();
            nsl_tensor_to_desc(p, &mut desc);
            desc
        })
        .collect();
    let mut output_descs: Vec<NslTensorDesc> =
        (0..num_outputs).map(|_| NslTensorDesc::default()).collect();
    let rc = nsl_model_call(
        model_ptr,
        name_ptr,
        input_descs.as_mut_ptr() as i64,
        num_inputs,
        output_descs.as_mut_ptr() as i64,
        num_outputs,
    );
    if rc == 0 && dl_outputs_ptr != 0 {
        let out_slot = dl_outputs_ptr as *mut *mut DLManagedTensor;
        for (i, desc) in output_descs.iter().enumerate() {
            let tensor_ptr = desc_to_nsl_tensor(desc);
            let tensor = NslTensor::from_ptr(tensor_ptr);
            let dl_ptr = crate::dlpack::nsl_tensor_to_dlpack(tensor, tensor_ptr);
            unsafe {
                *out_slot.add(i) = dl_ptr;
            }
        }
    }
    for ptr in &input_tensor_ptrs {
        crate::tensor::nsl_tensor_free(*ptr);
    }
    rc
}

/// Return the cached fn pointer for an `@export` by name. Lets
/// tight-loop callers skip the HashMap probe on every call.
///
/// Returns 0 (NULL) if the name is not in the registry or `model_ptr`
/// is null. The thread-local error is set on null inputs and on the
/// "no registry" path; missing-name on a populated registry returns 0
/// silently (the caller is presumed to be probing).
#[no_mangle]
pub extern "C" fn nsl_model_lookup_function(model_ptr: i64, name_ptr: i64) -> i64 {
    if model_ptr == 0 || name_ptr == 0 {
        set_error("nsl_model_lookup_function: null pointer\0".to_string());
        return 0;
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    let registry = match &model.exports {
        Some(r) => r,
        None => {
            set_error(
                "nsl_model_lookup_function: no registry; use nsl_model_create_with_lib\0"
                    .to_string(),
            );
            return 0;
        }
    };
    let name = unsafe { CStr::from_ptr(name_ptr as *const c_char) }
        .to_string_lossy()
        .into_owned();
    match registry.lookup(&name) {
        Some(p) => p as usize as i64,
        None => 0,
    }
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
    capi_trace("model_set_forward registered");

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
    // Prefer the named-dispatch path when a registry exists.
    let model = unsafe { &*(model_ptr as *const NslModel) };
    if model.exports.is_some() {
        let name = c"forward";
        return nsl_model_call(
            model_ptr,
            name.as_ptr() as i64,
            inputs_ptr,
            num_inputs,
            outputs_ptr,
            num_outputs,
        );
    }
    // Legacy fallback: a model created via the old `nsl_model_create`
    // (no _with_lib path) uses the registered `forward_fn`. This path
    // will be removed once all callers migrate.
    legacy_forward_fn_path(model_ptr, inputs_ptr, num_inputs, outputs_ptr, num_outputs)
}

/// Legacy `nsl_model_forward` body — preserved as a fallback for models
/// built via `nsl_model_create` (no shared-library path, so no export
/// registry). Models created with `nsl_model_create_with_lib` route
/// through `nsl_model_call` instead.
fn legacy_forward_fn_path(
    model_ptr: i64,
    inputs_ptr: i64,
    num_inputs: i64,
    outputs_ptr: i64,
    num_outputs: i64,
) -> i64 {
    let model = unsafe { &*(model_ptr as *const NslModel) };
    capi_trace(format!(
        "model_forward start num_inputs={} num_outputs={} weights={}",
        num_inputs,
        num_outputs,
        model.weight_ptrs.len()
    ));

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
        for (idx, desc) in descs.iter().enumerate() {
            capi_trace(format!(
                "model_forward import input={} ndim={} dtype={} shape_ptr={:?} strides_ptr={:?}",
                idx,
                desc.ndim,
                desc.dtype,
                desc.shape,
                desc.strides
            ));
            let tensor_ptr = desc_to_nsl_tensor(desc);
            input_ptrs.push(tensor_ptr);
        }
    }
    capi_trace(format!("model_forward imported_inputs={}", input_ptrs.len()));

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
    capi_trace("model_forward invoke_forward");
    let output_tensor_ptrs = forward_fn(&model.weight_ptrs, &input_ptrs);
    capi_trace(format!("model_forward outputs={}", output_tensor_ptrs.len()));

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

/// Get the number of weight tensors — canonical name used by @export model-method wrappers.
/// Alias for `nsl_model_num_weights`.
#[no_mangle]
pub extern "C" fn nsl_model_get_num_weights(model_ptr: i64) -> i64 {
    nsl_model_num_weights(model_ptr)
}

/// Return a pointer to the contiguous array of weight tensor pointers (`*const i64`).
///
/// The returned pointer is valid for the lifetime of the model; callers must not
/// free it.  Returns 0 if `model_ptr` is null or the model has no weights.
/// Used by `@export` model-method wrappers to thread weight pointers into the
/// compiled impl function.
#[no_mangle]
pub extern "C" fn nsl_model_get_weight_ptrs(model_ptr: i64) -> i64 {
    if model_ptr == 0 { return 0; }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    if model.weight_ptrs.is_empty() {
        return 0;
    }
    model.weight_ptrs.as_ptr() as i64
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
    let weight = model.weights.get(name).copied().unwrap_or(0);
    capi_trace(format!("model_get_weight name={name} found={}", weight != 0));
    weight
}

// NOTE (Gap I.1/I.2 worktree): the previously-present second pair of
// `nsl_model_get_weight_ptrs` / `nsl_model_get_num_weights` definitions
// has been removed. They were duplicate `#[no_mangle]` exports of the
// functions defined at lines 446/457 above, left over from a conflict
// between PR #45 (@export decorator) and PR #48 (m62 c-wrappers). The
// duplicate block blocked `cargo build` on main at `b1c071f`. Removing
// the second pair here is not part of the Gap I bundle — it is a
// pre-existing compile fix needed to run this PR's tests. File a
// separate cleanup PR if a hygiene pass for main is desired.
#[allow(dead_code)]
fn _gap_i_compile_note(model_ptr: i64) -> i64 {
    let _ = model_ptr;
    0
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
        capi_trace(format!("load_weight name={name} shape={shape:?}"));

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
        safetensors::Dtype::F32 => data
            .chunks_exact(std::mem::size_of::<f32>())
            .take(len)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().expect("f32 chunks are 4 bytes");
                f32::from_le_bytes(bytes)
            })
            .collect(),
        safetensors::Dtype::F64 => data
            .chunks_exact(std::mem::size_of::<f64>())
            .take(len)
            .map(|chunk| {
                let bytes: [u8; 8] = chunk.try_into().expect("f64 chunks are 8 bytes");
                f64::from_le_bytes(bytes) as f32
            })
            .collect(),
        safetensors::Dtype::F16 => data
            .chunks_exact(std::mem::size_of::<u16>())
            .take(len)
            .map(|chunk| {
                let bytes: [u8; 2] = chunk.try_into().expect("f16 chunks are 2 bytes");
                half::f16::from_bits(u16::from_le_bytes(bytes)).to_f32()
            })
            .collect(),
        safetensors::Dtype::BF16 => data
            .chunks_exact(std::mem::size_of::<u16>())
            .take(len)
            .map(|chunk| {
                let bytes: [u8; 2] = chunk.try_into().expect("bf16 chunks are 2 bytes");
                half::bf16::from_bits(u16::from_le_bytes(bytes)).to_f32()
            })
            .collect(),
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

/// C-ABI version of `desc_to_nsl_tensor` for use by Cranelift-emitted
/// `@export` wrapper bodies. Takes an `NslTensorDesc*` as an i64 pointer
/// and returns a newly-allocated `NslTensor*` (also as i64).
/// The caller is responsible for freeing the result via `nsl_tensor_free`.
#[no_mangle]
pub extern "C" fn nsl_desc_to_tensor(desc_ptr: i64) -> i64 {
    if desc_ptr == 0 {
        return 0;
    }
    let desc = unsafe { &*(desc_ptr as *const NslTensorDesc) };
    desc_to_nsl_tensor(desc)
}

/// C-ABI version of `nsl_tensor_to_desc` for use by Cranelift-emitted
/// `@export` wrapper bodies. Writes the NslTensor pointed to by `tensor_ptr`
/// into the `NslTensorDesc` pointed to by `desc_ptr`.
#[no_mangle]
pub extern "C" fn nsl_tensor_to_desc_ffi(tensor_ptr: i64, desc_ptr: i64) {
    if tensor_ptr == 0 || desc_ptr == 0 {
        return;
    }
    let desc = unsafe { &mut *(desc_ptr as *mut NslTensorDesc) };
    nsl_tensor_to_desc(tensor_ptr, desc);
}

/// Runtime helper used by the codegen-emitted packed-array dispatch wrapper.
///
/// The dispatch wrapper passes its own scratch `NslTensorDesc` to the typed
/// wrapper so the typed wrapper's data-pointer overwrite lands in scratch
/// (which the typed wrapper does — it borrows the impl tensor's data
/// pointer rather than memcpy'ing into a caller buffer). This helper then
/// folds the scratch desc back onto the caller's preallocated output desc:
///
///   - memcpy(`dst.data`, `src.data`, element_count * dtype_size_bytes)
///   - dst.shape/strides/ndim/dtype/device_* := src.* (caller's preallocated
///     `dst.data` ptr is preserved so subsequent caller reads see the copy)
///
/// Returns 0 on success, -1 on null inputs / unrecognized dtype.
///
/// `src_desc_ptr`: scratch desc post-typed-wrapper (data ptr is borrowed
///                 from an impl-owned tensor; the metadata may have been
///                 freshly written here).
/// `dst_desc_ptr`: caller-supplied output desc whose `data` field holds a
///                 preallocated buffer.
#[no_mangle]
pub extern "C" fn nsl_dispatch_apply_result(src_desc_ptr: i64, dst_desc_ptr: i64) -> i64 {
    if src_desc_ptr == 0 || dst_desc_ptr == 0 {
        set_error("nsl_dispatch_apply_result: null desc pointer\0".to_string());
        return -1;
    }
    let src = unsafe { &*(src_desc_ptr as *const NslTensorDesc) };
    let dst = unsafe { &mut *(dst_desc_ptr as *mut NslTensorDesc) };

    // Bytes per element, indexed by C-ABI dtype (see NslTensorDesc::dtype).
    let elem_bytes: usize = match src.dtype {
        0 => 4, // f32
        1 => 8, // f64
        2 => 2, // f16
        3 => 2, // bf16
        4 => 4, // int32
        5 => 8, // int64
        6 => 1, // int8
        7 => 1, // uint8
        _ => {
            set_error(format!(
                "nsl_dispatch_apply_result: unrecognized dtype {}\0",
                src.dtype
            ));
            return -1;
        }
    };

    // Element count from shape product.
    let ndim = src.ndim.max(0) as usize;
    let mut n_elem: usize = 1;
    if ndim == 0 {
        n_elem = 1;
    } else if src.shape.is_null() {
        set_error("nsl_dispatch_apply_result: null src shape\0".to_string());
        return -1;
    } else {
        for i in 0..ndim {
            let d = unsafe { std::ptr::read_unaligned(src.shape.add(i)) };
            if d < 0 {
                set_error("nsl_dispatch_apply_result: negative dim in src shape\0".to_string());
                return -1;
            }
            n_elem = n_elem.saturating_mul(d as usize);
        }
    }

    let byte_count = n_elem.saturating_mul(elem_bytes);

    let caller_buf = dst.data;
    if byte_count > 0 {
        if caller_buf.is_null() || src.data.is_null() {
            set_error(
                "nsl_dispatch_apply_result: null data pointer (caller buffer or impl result)\0"
                    .to_string(),
            );
            return -1;
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                src.data as *const u8,
                caller_buf as *mut u8,
                byte_count,
            );
        }
    }

    // Mirror metadata (shape/strides/ndim/dtype/device) from src so that
    // callers reading `dst.ndim`/`dst.shape` after the dispatch see the
    // actual output shape, not whatever they had pre-populated. Preserve
    // `dst.data` since that's the caller's buffer we just filled.
    dst.shape = src.shape;
    dst.strides = src.strides;
    dst.ndim = src.ndim;
    dst.dtype = src.dtype;
    dst.device_type = src.device_type;
    dst.device_id = src.device_id;
    0
}

/// Convert an NslTensorDesc (C API) into an NslTensor pointer.
/// The resulting tensor borrows the data buffer — caller must not free the
/// original data while the tensor is live.
fn desc_to_nsl_tensor(desc: &NslTensorDesc) -> i64 {
    let ndim = desc.ndim as usize;
    let nsl_dtype = capi_dtype_to_nsl(desc.dtype);

    let shape_ptr = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim {
        unsafe {
            *shape_ptr.add(i) = std::ptr::read_unaligned(desc.shape.add(i));
        }
    }

    let strides = if desc.strides.is_null() {
        NslTensor::compute_strides(shape_ptr, desc.ndim as i64)
    } else {
        let s = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
        for i in 0..ndim {
            unsafe {
                *s.add(i) = std::ptr::read_unaligned(desc.strides.add(i));
            }
        }
        s
    };

    let len = NslTensor::total_elements(shape_ptr, desc.ndim as i64);
    let device = if desc.device_type > 0 { desc.device_id as u8 + 1 } else { 0 };
    capi_trace(format!(
        "desc_to_tensor ndim={} len={} device={} dtype={}",
        desc.ndim,
        len,
        device,
        nsl_dtype
    ));

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
            exports: None,
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
            exports: None,
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
    fn nsl_set_error_cstr_sets_thread_local() {
        nsl_clear_error();
        let msg = std::ffi::CString::new("hello from wrapper").unwrap();
        nsl_set_error_cstr(msg.as_ptr() as i64);
        let err_ptr = nsl_get_last_error();
        assert_ne!(err_ptr, 0);
        let got = unsafe {
            CStr::from_ptr(err_ptr as *const c_char)
                .to_string_lossy()
                .into_owned()
        };
        assert_eq!(got, "hello from wrapper");
        nsl_clear_error();
    }

    #[test]
    fn nsl_set_error_cstr_null_is_noop() {
        nsl_clear_error();
        nsl_set_error_cstr(0);
        // Should not panic; clear_error still works after
        let err_ptr = nsl_get_last_error();
        let err_msg = unsafe { CStr::from_ptr(err_ptr as *const c_char).to_str().unwrap_or("") };
        assert_eq!(err_msg, "");
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
            exports: None,
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


    #[test]
    fn nsl_model_get_weight_ptrs_returns_valid_pointer() {
        // Create a model with a single weight pointer.
        // To avoid freeing invalid pointers on drop, we clear weight_ptrs manually.
        let fake_weight_ptr: i64 = 0xDEADBEEF_i64;
        let mut model = Box::new(NslModel {
            version: 2,
            weights: HashMap::new(),
            weight_ptrs: vec![fake_weight_ptr],
            forward_fn: None,
            weights_path: String::new(),
            grad_enabled: false,
            last_forward_outputs: vec![],
            exports: None,
        });
        let model_ptr = &mut *model as *mut NslModel as i64;

        let got = nsl_model_get_weight_ptrs(model_ptr);
        assert_ne!(got, 0);
        let first: i64 = unsafe { *(got as *const i64) };
        assert_eq!(first, fake_weight_ptr);

        // Clear weight_ptrs so drop() won't try to free the fake pointers
        model.weight_ptrs.clear();
    }

    #[test]
    fn nsl_model_get_num_weights_returns_length() {
        let mut model = Box::new(NslModel {
            version: 2,
            weights: HashMap::new(),
            weight_ptrs: vec![0x100, 0x200, 0x300],  // Use larger values to avoid null-like misalignment
            forward_fn: None,
            weights_path: String::new(),
            grad_enabled: false,
            last_forward_outputs: vec![],
            exports: None,
        });
        let model_ptr = &mut *model as *mut NslModel as i64;

        assert_eq!(nsl_model_get_num_weights(model_ptr), 3);

        // Clear weight_ptrs so drop() won't try to free
        model.weight_ptrs.clear();
    }

    #[test]
    fn nsl_model_get_weight_ptrs_null_returns_zero() {
        assert_eq!(nsl_model_get_weight_ptrs(0), 0);
        assert_eq!(nsl_model_get_num_weights(0), 0);
    }
}
