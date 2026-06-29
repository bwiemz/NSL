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
///
/// Layout (48 bytes, 8-byte aligned):
///   offset  0: data         (*mut c_void, 8)
///   offset  8: shape        (*mut i64,    8)
///   offset 16: strides      (*mut i64,    8)
///   offset 24: ndim         (i32,         4)
///   offset 28: dtype        (i32,         4)
///   offset 32: device_type  (i32,         4)
///   offset 36: device_id    (i32,         4)
///   offset 40: tape_id      (i64,         8)
///
/// `tape_id` carries the source tensor's autodiff tape id verbatim so that
/// a desc round-trip (`nsl_tensor_to_desc` → `desc_to_nsl_tensor`) does not
/// strip the id. Required for the per-call grad context (Spec B): the
/// loss seed in `run_backward_core` keys on `t.tape_id`, which would fall
/// through to the raw-pointer fallback if the desc dropped the id.
///
/// Semantics of `tape_id`:
///   - `tape_id == 0`: source tensor was never autodiff-tracked (constants,
///     freshly-allocated wrappers, inputs from non-grad code paths).
///   - `tape_id > 0`: matches the source tensor's `tape_id` as assigned by
///     `Tape::get_or_assign_id`. Thread-local `next_id` is monotonic
///     (never reset — see `autodiff/mod.rs:291, 399`), so on the same
///     thread an imported id is guaranteed `< tape.next_id`.
///
/// `#[derive(Default)]` is preserved for the two scratch-desc allocation
/// sites in `nsl_model_call_dlpack`; new struct-literal sites are
/// compiler-required to specify `tape_id` explicitly (no `..Default::default()`
/// shorthand is currently used).
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
    /// Autodiff tape id of the source tensor, copied verbatim across desc
    /// round-trips. `0` means "untracked" (the legacy semantics that
    /// produced the bug fixed by this commit).
    pub tape_id: i64,
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

/// Opaque model handle. Holds loaded weights and the eagerly-populated
/// export dispatch table for `@export`-decorated functions.
pub struct NslModel {
    /// Model version (for ABI compatibility).
    #[allow(dead_code)]
    version: u32,
    /// Weight tensors loaded from safetensors/nslm file.
    /// Keys are parameter names, values are NslTensor pointers (as i64).
    weights: HashMap<String, i64>,
    /// Ordered list of weight tensor pointers (for positional access).
    weight_ptrs: Vec<i64>,
    /// Path the weights were loaded from (for diagnostics).
    #[allow(dead_code)]
    weights_path: String,
    /// Dispatch table for `@export`ed functions. Populated eagerly at
    /// create time — either via runtime self-discovery in `nsl_model_create`
    /// or explicitly through `nsl_model_create_with_lib`. `None` when the
    /// containing object has no export table (e.g., model handles built
    /// inside cargo test binaries with no `@export` functions).
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

/// Spec B §5.2/§5.5 helper: set the thread-local error from an already-
/// constructed `CString`. Used by `grad_context.rs` to avoid the
/// `String → CString` allocation churn for the small set of static
/// messages used by the per-call grad-context FFIs.
pub(crate) fn set_error_cstring(cstr: std::ffi::CString) {
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(cstr));
}

fn capi_trace(msg: impl AsRef<str>) {
    if std::env::var_os("NSL_CAPI_TRACE").is_some() {
        eprintln!("[nsl-capi] {}", msg.as_ref());
    }
}

// ---------------------------------------------------------------------------
// FFI: ABI version
// ---------------------------------------------------------------------------

/// NSL runtime C-ABI version — **major** component.
///
/// Bump this on any *breaking* change to an exported symbol's signature or
/// semantics, or to the [`NslTensorDesc`] memory layout. A host that links a
/// runtime whose `nsl_abi_version()` major differs from the major baked into
/// the generated header (`NSL_ABI_VERSION_MAJOR`) must refuse to run: the ABI
/// is incompatible.
pub const NSL_ABI_VERSION_MAJOR: u32 = 1;

/// NSL runtime C-ABI version — **minor** component.
///
/// Bump this for backward-compatible additions (new exported symbols, new
/// trailing optional behavior). A host built against minor `m` can safely use a
/// runtime with minor `>= m` and the same major.
pub const NSL_ABI_VERSION_MINOR: u32 = 0;

/// Return the runtime's C-ABI version packed as `(major << 16) | minor`.
///
/// Hosts can call this immediately after loading `libnsl_runtime` and compare
/// against the `NSL_ABI_VERSION_*` macros emitted into the generated C header
/// to detect runtime/header skew before making any other call.
#[no_mangle]
pub extern "C" fn nsl_abi_version() -> i64 {
    ((NSL_ABI_VERSION_MAJOR as i64) << 16) | (NSL_ABI_VERSION_MINOR as i64)
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

/// Given a function pointer (as i64), return a heap-allocated null-
/// terminated C string with the filesystem path of the shared library
/// containing that function. Returns 0 (NULL) on failure.
///
/// The caller frees the returned pointer via `nsl_free_cstr`.
///
/// Implementation: `dladdr` on Unix, `GetModuleHandleEx + GetModuleFileName`
/// on Windows. The function pointer must be a symbol whose address lives
/// inside the target shared object; any code address emitted into the
/// library works.
#[no_mangle]
pub extern "C" fn nsl_dl_path_for_fn_addr(fn_addr: i64) -> i64 {
    if fn_addr == 0 {
        return 0;
    }
    #[cfg(unix)]
    unsafe {
        let mut info: libc::Dl_info = std::mem::zeroed();
        let probe = fn_addr as *const std::ffi::c_void;
        if libc::dladdr(probe, &mut info) == 0 || info.dli_fname.is_null() {
            return 0;
        }
        let s = std::ffi::CStr::from_ptr(info.dli_fname)
            .to_string_lossy()
            .into_owned();
        match std::ffi::CString::new(s) {
            Ok(cs) => cs.into_raw() as i64,
            Err(_) => 0,
        }
    }
    #[cfg(windows)]
    unsafe {
        use std::ffi::OsString;
        use std::os::windows::ffi::OsStringExt;
        #[link(name = "kernel32")]
        extern "system" {
            fn GetModuleHandleExW(
                flags: u32,
                module_name: *const u16,
                handle_out: *mut *mut std::ffi::c_void,
            ) -> i32;
            fn GetModuleFileNameW(
                module: *mut std::ffi::c_void,
                filename: *mut u16,
                size: u32,
            ) -> u32;
        }
        const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x4;
        const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x2;
        let mut module: *mut std::ffi::c_void = std::ptr::null_mut();
        let ok = GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            fn_addr as *const u16,
            &mut module,
        );
        if ok == 0 || module.is_null() {
            return 0;
        }
        let mut buf = vec![0u16; 32768];
        let n = GetModuleFileNameW(module, buf.as_mut_ptr(), buf.len() as u32);
        if n == 0 {
            return 0;
        }
        buf.truncate(n as usize);
        let os: OsString = OsString::from_wide(&buf);
        let s = os.to_string_lossy().into_owned();
        match std::ffi::CString::new(s) {
            Ok(cs) => cs.into_raw() as i64,
            Err(_) => 0,
        }
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = fn_addr;
        0
    }
}

/// Free a C string allocated by NSL runtime FFIs that return ownership
/// (currently only `nsl_dl_path_for_fn_addr`). Safe to call with a NULL
/// pointer.
#[no_mangle]
pub extern "C" fn nsl_free_cstr(ptr: i64) {
    if ptr == 0 {
        return;
    }
    unsafe {
        let _ = std::ffi::CString::from_raw(ptr as *mut std::os::raw::c_char);
    }
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
/// Loads weights from a `.safetensors` file into NslTensors. The runtime
/// statically linked into the model's shared library self-discovers the
/// containing .so/.dll/.dylib path (via `dladdr` on Unix / `GetModuleHandleEx`
/// on Windows) and eagerly populates the `ExportRegistry` so subsequent
/// `nsl_model_call`/`nsl_model_forward` calls route to `@export`-emitted
/// dispatch wrappers without any caller-side knowledge of the .so path.
///
/// If self-discovery fails or the containing object has no export table
/// (e.g., loaded from a unit-test binary), the model is returned with a
/// `None` registry; the model is still usable for weight inspection but
/// `nsl_model_call` will then return -1 with an explanatory error.
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

    // Self-discover the containing shared library and eagerly populate the
    // export registry. dladdr/GetModuleHandleEx on a function pointer that
    // belongs to this very runtime (statically linked into the .so when the
    // caller is a Python ctypes consumer of a `nsl build --shared-lib`
    // artifact) returns the .so path. If self-discovery succeeds and the
    // library exposes the codegen-emitted export-table FFIs, the registry
    // populates. Failures (no export table, non-.so context) fall through
    // to a model with `exports = None` — still valid; `nsl_model_call` will
    // surface a clear error.
    let exports = self_discover_export_registry();

    let model = Box::new(NslModel {
        version: 2,
        weights,
        weight_ptrs,
        weights_path: path_str.to_string(),
        exports,
    });
    Box::into_raw(model) as i64
}

/// Best-effort self-discovery of the .so/.dll containing this runtime's
/// code, followed by a `ExportRegistry::from_library_path` call. Used by
/// `nsl_model_create` to populate the dispatch table without requiring
/// callers to pass a library path explicitly.
///
/// Returns `None` if the platform path-lookup fails OR the discovered
/// path has no codegen-emitted export-table FFIs (e.g., when running
/// inside a cargo test binary that links the runtime statically but has
/// no `@export`'d functions).
fn self_discover_export_registry() -> Option<crate::c_api::exports::ExportRegistry> {
    // `nsl_model_create as i64` is an address that lives in the same
    // object file as the runtime code. dladdr / GetModuleHandleEx on
    // that address returns the path of the containing .so/.dll.
    let probe = nsl_model_create as *const () as i64;
    let path_ptr = nsl_dl_path_for_fn_addr(probe);
    if path_ptr == 0 {
        capi_trace("model_create self_discover: dl path lookup failed");
        return None;
    }
    let path = unsafe { CStr::from_ptr(path_ptr as *const c_char) }
        .to_string_lossy()
        .into_owned();
    nsl_free_cstr(path_ptr);
    let pb = std::path::PathBuf::from(&path);
    capi_trace(format!("model_create self_discover lib={path}"));
    match crate::c_api::exports::ExportRegistry::from_library_path(&pb) {
        Ok(reg) => Some(reg),
        Err(e) => {
            capi_trace(format!(
                "model_create self_discover registry build failed: {e}"
            ));
            None
        }
    }
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

// ---------------------------------------------------------------------------
// FFI: Forward pass
// ---------------------------------------------------------------------------

/// Run the model's forward pass with NslTensorDesc inputs/outputs.
///
/// Thin compatibility shim that delegates to `nsl_model_call(model,
/// "forward", ...)`. Preserves the original C ABI bit-stably for Python
/// ctypes callers while routing through the unified named-export dispatch.
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
    let name = c"forward";
    nsl_model_call(
        model_ptr,
        name.as_ptr() as i64,
        inputs_ptr,
        num_inputs,
        outputs_ptr,
        num_outputs,
    )
}

/// Run the model's forward pass with DLPack tensors (zero-copy).
///
/// Compatibility shim that delegates to
/// `nsl_model_call_dlpack(model, "forward", ...)`. Bridges the historical
/// `num_outputs_ptr` (out-pointer style) ABI to the registry-based dispatch
/// that takes a fixed `num_outputs` count, allocating an output array of
/// `*num_outputs_ptr` slots before the call and writing the actual count
/// back afterwards.
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
    // For backwards ABI compatibility, treat `num_outputs_ptr` as both an
    // in-pointer (caller pre-sets it with the slot count of `outputs_ptr`)
    // and out-pointer. If the caller didn't set it, default to 1 — the
    // overwhelmingly common single-output forward case.
    let mut num_outputs: i64 = if num_outputs_ptr != 0 {
        let n = unsafe { *(num_outputs_ptr as *const i64) };
        if n <= 0 { 1 } else { n }
    } else {
        1
    };

    let name = c"forward";
    let rc = nsl_model_call_dlpack(
        model_ptr,
        name.as_ptr() as i64,
        inputs_ptr,
        num_inputs,
        outputs_ptr,
        num_outputs,
    );
    if num_outputs_ptr != 0 {
        // We don't have a true "actual count" from the dispatch path —
        // `nsl_model_call_dlpack` writes exactly `num_outputs` slots. Mirror
        // that back to the caller.
        if rc != 0 {
            num_outputs = 0;
        }
        unsafe { *(num_outputs_ptr as *mut i64) = num_outputs };
    }
    rc
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

    // Mirror metadata (shape/strides/ndim/dtype/device/tape_id) from src so
    // that callers reading `dst.ndim`/`dst.shape` after the dispatch see the
    // actual output shape, not whatever they had pre-populated. Preserve
    // `dst.data` since that's the caller's buffer we just filled.
    //
    // `tape_id` is the load-bearing line for per-call grad context backward:
    // the typed wrapper writes the impl tensor's tape_id into scratch via
    // `nsl_tensor_to_desc_ffi`, and we forward it onto the caller's desc here
    // so `nsl_model_forward_grad` can reconstruct the loss-seed tape_id when
    // it re-wraps the output desc via `desc_to_nsl_tensor`.
    dst.shape = src.shape;
    dst.strides = src.strides;
    dst.ndim = src.ndim;
    dst.dtype = src.dtype;
    dst.device_type = src.device_type;
    dst.device_id = src.device_id;
    dst.tape_id = src.tape_id;
    0
}

/// Convert an NslTensorDesc (C API) into an NslTensor pointer.
/// The resulting tensor borrows the data buffer — caller must not free the
/// original data while the tensor is live.
///
/// **`tape_id` round-trip:** if `desc.tape_id != 0`, the new wrapper inherits
/// it verbatim. This is the load-bearing line for per-call grad context
/// backward: the loss seed in `run_backward_core` keys on
/// `if t.tape_id != 0 { t.tape_id } else { loss_ptr }`, so a fresh wrapper
/// must carry the same id the impl-emitted output tensor was assigned by
/// `Tape::get_or_assign_id` during forward. See `c_api::NslTensorDesc` for
/// the monotonicity invariant; the debug-assert below catches any future
/// regression that violates it.
pub(crate) fn desc_to_nsl_tensor(desc: &NslTensorDesc) -> i64 {
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
        "desc_to_tensor ndim={} len={} device={} dtype={} tape_id={}",
        desc.ndim,
        len,
        device,
        nsl_dtype,
        desc.tape_id,
    ));

    let mut tensor = Box::new(NslTensor::new(
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

    // Inherit the autodiff tape id from the desc verbatim. The thread-local
    // `next_id` counter is monotonic (`autodiff/mod.rs:291, 399` — explicit
    // "Do NOT reset next_id" comments at both `nsl_tape_start` and
    // `nsl_tape_stop`), so on the same thread an imported id is guaranteed
    // strictly less than `tape.next_id`. The debug-assert below pins this
    // invariant: a violation would mean a future change introduced a
    // `next_id` reset, which would silently misattribute the loss seed in
    // backward.
    if desc.tape_id != 0 {
        debug_assert!(
            desc.tape_id > 0,
            "negative desc.tape_id={} — Tape::get_or_assign_id only emits >= 1",
            desc.tape_id,
        );
        crate::autodiff::debug_assert_tape_id_in_range(desc.tape_id);
        tensor.tape_id = desc.tape_id;
    }
    Box::into_raw(tensor) as i64
}

/// Fill an NslTensorDesc from an NslTensor pointer.
///
/// Copies `tape_id` verbatim so that a subsequent `desc_to_nsl_tensor`
/// reconstructs the autodiff identity (Spec B per-call grad context).
pub(crate) fn nsl_tensor_to_desc(tensor_ptr: i64, desc: &mut NslTensorDesc) {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    desc.data = tensor.data;
    desc.ndim = tensor.ndim as i32;
    desc.dtype = nsl_dtype_to_capi(tensor.dtype);
    desc.device_type = if tensor.device > 0 { 1 } else { 0 };
    desc.device_id = if tensor.device > 0 { (tensor.device - 1) as i32 } else { 0 };
    desc.shape = tensor.shape;
    desc.strides = tensor.strides;
    desc.tape_id = tensor.tape_id;
}

// ---------------------------------------------------------------------------
// M62b: Backward pass FFI
// ---------------------------------------------------------------------------

// Spec B §4.4 — the model-level `nsl_model_backward(model, ...)` FFI is
// replaced by the per-call `nsl_model_backward(ctx, ...)` defined in
// `grad_context.rs`. The two cannot coexist (same exported symbol); the
// switch is structural to enforce the headline invariant from §2 (backward
// reads only from `GradContext`, never from the thread-local tape).
//
// Spec B T8 deleted `nsl_model_enable_grad` and the `grad_enabled` /
// `last_forward_outputs` fields on `NslModel`. The Python autograd bridge
// still references the gone symbol; its migration is T9.

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, offset_of, size_of};

    /// Golden ABI guard: the `NslTensorDesc` layout is part of the C-ABI
    /// contract documented on the struct and emitted into generated headers.
    /// Any change here is a breaking ABI change and must bump
    /// `NSL_ABI_VERSION_MAJOR`. This test fails loudly if the layout drifts.
    #[test]
    fn nsl_tensor_desc_abi_layout_is_pinned() {
        assert_eq!(size_of::<NslTensorDesc>(), 48, "NslTensorDesc must be 48 bytes");
        assert_eq!(align_of::<NslTensorDesc>(), 8, "NslTensorDesc must be 8-byte aligned");
        assert_eq!(offset_of!(NslTensorDesc, data), 0);
        assert_eq!(offset_of!(NslTensorDesc, shape), 8);
        assert_eq!(offset_of!(NslTensorDesc, strides), 16);
        assert_eq!(offset_of!(NslTensorDesc, ndim), 24);
        assert_eq!(offset_of!(NslTensorDesc, dtype), 28);
        assert_eq!(offset_of!(NslTensorDesc, device_type), 32);
        assert_eq!(offset_of!(NslTensorDesc, device_id), 36);
        assert_eq!(offset_of!(NslTensorDesc, tape_id), 40);
    }

    /// The packed `nsl_abi_version()` value must decode to the documented
    /// major/minor constants — the same numbers the generated header pins.
    #[test]
    fn nsl_abi_version_packs_major_minor() {
        let packed = nsl_abi_version();
        assert_eq!((packed >> 16) as u32, NSL_ABI_VERSION_MAJOR);
        assert_eq!((packed & 0xffff) as u32, NSL_ABI_VERSION_MINOR);
    }

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
            tape_id: 0,
        };

        let tensor_ptr = desc_to_nsl_tensor(&desc);
        assert_ne!(tensor_ptr, 0);

        let tensor = NslTensor::from_ptr(tensor_ptr);
        assert_eq!(tensor.ndim, 2);
        assert_eq!(tensor.len, 4);
        assert_eq!(tensor.dtype, 1); // NSL f32
        assert_eq!(tensor.device, 0);
        assert_eq!(tensor.owns_data, 0); // borrowed
        assert_eq!(tensor.tape_id, 0); // untracked input

        // Read back via desc
        let mut out_desc = NslTensorDesc {
            data: std::ptr::null_mut(), shape: std::ptr::null_mut(),
            strides: std::ptr::null_mut(), ndim: 0, dtype: 0,
            device_type: 0, device_id: 0, tape_id: 0,
        };
        nsl_tensor_to_desc(tensor_ptr, &mut out_desc);
        assert_eq!(out_desc.ndim, 2);
        assert_eq!(out_desc.dtype, 0); // back to C API f32
        assert_eq!(out_desc.tape_id, 0); // wrapper has no tape_id assigned

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

    // Spec B T3/T4 — the legacy `test_backward_null_model`,
    // `test_backward_without_enable_grad`, and `test_backward_without_forward`
    // tests exercised the model-level `nsl_model_backward(model, ...)` FFI
    // that has been removed (replaced by per-call `nsl_model_backward(ctx,
    // ...)` in `grad_context.rs`). Equivalent contract tests for the new
    // signature live in `tests/backward_does_not_consult_live_tape.rs` and
    // sibling integration files added by Spec B T5/T6.

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

    // NOTE: the previous `test_backward_with_tape_e2e` exercised the legacy
    // tape-recording path that ran a Rust-closure `forward_fn` over weight
    // pointers and saved outputs to a model-level output slot. That path was
    // removed when `nsl_model_forward` became a thin shim over
    // `nsl_model_call`; `@export` dispatch is inference-only and the
    // tape/grad bridge is now exercised through `GradContext` (Spec B).

    #[test]
    fn nsl_model_get_weight_ptrs_returns_valid_pointer() {
        // Create a model with a single weight pointer.
        // To avoid freeing invalid pointers on drop, we clear weight_ptrs manually.
        let fake_weight_ptr: i64 = 0xDEADBEEF_i64;
        let mut model = Box::new(NslModel {
            version: 2,
            weights: HashMap::new(),
            weight_ptrs: vec![fake_weight_ptr],
            weights_path: String::new(),
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
            weights_path: String::new(),
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
