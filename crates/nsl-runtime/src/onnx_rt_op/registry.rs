//! Spec C §2.4 — per-export `OrtCustomOp` vtable construction.
//!
//! ORT's contract: each registered custom op needs an `OrtCustomOp*` pointing
//! at a vtable struct of function pointers. We construct one heap-allocated
//! `OrtCustomOp` per `@export`, with NSL-side functions filling the slots.
//!
//! ## Memory model
//!
//! Each `OrtCustomOp` (along with the `CString` backing its `GetName`) is
//! stored in a `Box<PerExportVtable>` and pushed into a process-wide
//! `VTABLES` registry. The boxes never move — they live until process exit —
//! so the raw pointers handed to ORT remain valid for the session's lifetime.
//! Bounded leak: at most one entry per `@export` per session.
//!
//! ## Symbol resolution
//!
//! The dispatch symbol for each export is resolved **once** at
//! `RegisterCustomOps` time and cached in `PerExportVtable::cached_dispatch_fn`.
//! This avoids a repeated dlsym on every session creation and is the point at
//! which the RTLD_LOCAL workaround matters most (see `resolve_self_symbol`).
//!
//! Platform-specific handle acquisition:
//! - macOS: `dlsym(RTLD_SELF, name)` — searches only the calling library,
//!   works even under RTLD_LOCAL.
//! - Linux: `dladdr(&resolve_self_symbol, &info)` to get our `.so` path,
//!   then `dlopen(path, RTLD_NOLOAD)` to retrieve the existing handle,
//!   then `dlsym(handle, name)`.
//! - Windows: `GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, &resolve_export_via_self_dlsym, &out)`
//!   then `GetProcAddress(out, name)`.
//!
//! Spec A's `nsl_model_lookup_function` requires a model handle, which the
//! ORT kernel doesn't have — ORT loads the `.so` via
//! `register_custom_ops_library` without ever calling `nsl_model_create`.
//! v1 is therefore restricted to stateless exports; `model_ptr` is set to the
//! sentinel `1` (non-zero, never dereferenced) so the codegen-emitted null
//! check passes. Exports that read weights need M62c's session-config plumbing.

use std::ffi::{c_void, CStr, CString};
use std::os::raw::c_char;
use std::sync::{Mutex, OnceLock};

use super::kernel::{nsl_ort_kernel_compute, NslOrtKernelState};
use super::vendored::*;
use crate::c_api::exports::ExportFnPtr;

/// Per-export vtable storage. The pointers in `OrtCustomOp` must outlive the
/// ORT session, so we heap-allocate and keep ownership in `VTABLES`. The
/// `OrtCustomOp` is the FIRST field so casts between `*const OrtCustomOp`
/// and `*const PerExportVtable` are zero-offset.
#[repr(C)]
struct PerExportVtable {
    /// MUST stay first — `vtable_get_name` casts `*const OrtCustomOp` to
    /// `*const PerExportVtable` and reads `name_cstr` by offset.
    vtable: OrtCustomOp,
    /// Backing storage for `vtable.GetName`'s returned pointer.
    name_cstr: CString,
    /// Dispatch fn resolved once at `RegisterCustomOps` time. `0` means the
    /// symbol was not found; `CreateKernel`/`CreateKernelV2` return null in
    /// that case so ORT rejects the op rather than calling an invalid pointer.
    cached_dispatch_fn: usize,
}

// SAFETY: `PerExportVtable` contains only read-only function pointers, a
// `CString`, and a `usize`. Once constructed and pushed into `VTABLES` it is
// never mutated.
unsafe impl Send for PerExportVtable {}
unsafe impl Sync for PerExportVtable {}

// `Box` is load-bearing: pushing a `PerExportVtable` directly into a Vec
// would move it on every realloc, invalidating the raw pointer we hand
// to ORT. The Box pins the heap allocation.
#[allow(clippy::vec_box)]
static VTABLES: OnceLock<Mutex<Vec<Box<PerExportVtable>>>> = OnceLock::new();

/// Spec C §2.4 — construct an `OrtCustomOp` for the export at index `idx`.
///
/// Returns a pointer to a heap-allocated vtable owned by `VTABLES`. The
/// pointer remains valid until process exit. `idx` is currently unused
/// (each kernel-create call uses the cached fn ptr); kept in the signature
/// so a future revision can index a direct dispatch table.
pub fn make_custom_op_for_export(idx: i64, name: *const c_char) -> *const OrtCustomOp {
    let _ = idx;

    // SAFETY: caller (codegen-emitted `nsl_get_export_name`) hands us a
    // null-terminated C string with static lifetime in the same .so. We
    // copy into an owned CString so this registry doesn't depend on the
    // codegen-emitted storage living forever (in case future codegen
    // changes that contract).
    let name_cstr = unsafe { CStr::from_ptr(name) }.to_owned();

    // Eagerly resolve the dispatch symbol once at registration time.
    // We look up `<name>__nsl_dispatch`, NOT `<name>`: the typed `<name>`
    // symbol takes individual tensor-desc pointers plus an output desc as
    // separate params, while ExportFnPtr expects the packed-array ABI
    // (model_ptr, inputs_ptr, n_inputs, outputs_ptr, n_outputs). Calling the
    // typed wrapper as ExportFnPtr misroutes `n_inputs=1` as the output-desc
    // pointer, writing into address 0x1 → SIGABRT. See exports.rs L145.
    let dispatch_cname = {
        let s = format!("{}__nsl_dispatch", name_cstr.to_string_lossy());
        CString::new(s).unwrap_or_else(|_| name_cstr.clone())
    };
    let cached_dispatch_fn = unsafe { resolve_self_symbol(dispatch_cname.as_ptr()) };

    let entry = Box::new(PerExportVtable {
        vtable: OrtCustomOp {
            version: EXPECTED_ORT_API_VERSION,
            CreateKernel: vtable_create_kernel,
            GetName: vtable_get_name,
            GetExecutionProviderType: vtable_get_ep_type,
            GetInputType: vtable_get_input_type,
            GetInputTypeCount: vtable_get_input_count,
            GetOutputType: vtable_get_output_type,
            GetOutputTypeCount: vtable_get_output_count,
            KernelCompute: nsl_ort_kernel_compute,
            KernelDestroy: vtable_kernel_destroy,
            GetInputCharacteristic: vtable_get_input_char,
            GetOutputCharacteristic: vtable_get_output_char,
            GetInputMemoryType: vtable_get_input_mem_type,
            GetVariadicInputMinArity: vtable_get_variadic_min,
            GetVariadicInputHomogeneity: vtable_get_variadic_hom,
            GetVariadicOutputMinArity: vtable_get_variadic_min,
            GetVariadicOutputHomogeneity: vtable_get_variadic_hom,
            // ORT 1.16+ calls V2 callbacks when their slots are non-null,
            // taking priority over V1. Wire real implementations here so
            // kernel state is created and compute actually runs.
            CreateKernelV2: vtable_create_kernel_v2,
            KernelComputeV2: vtable_kernel_compute_v2,
            InferOutputShapeFn: vtable_infer_output_shape_unused,
            GetStartVersion: vtable_get_start_version,
            GetEndVersion: vtable_get_end_version,
            GetMayInplace: vtable_get_may_inplace,
            ReleaseMayInplace: vtable_release_may_inplace,
            GetAliasMap: vtable_get_alias_map,
            ReleaseAliasMap: vtable_release_alias_map,
        },
        name_cstr,
        cached_dispatch_fn,
    });

    // The Box's heap allocation address is what we return; pushing the Box
    // into the Vec below moves the Box pointer-value but not the heap data
    // it points to, so `raw_vtable` stays valid.
    let raw_vtable: *const OrtCustomOp = &entry.vtable as *const OrtCustomOp;
    VTABLES
        .get_or_init(|| Mutex::new(Vec::new()))
        .lock()
        .expect("VTABLES mutex poisoned")
        .push(entry);
    raw_vtable
}

// ---------------------------------------------------------------------------
// V1 vtable functions
// ---------------------------------------------------------------------------

unsafe extern "C" fn vtable_get_name(op: *const OrtCustomOp) -> *const c_char {
    // SAFETY: `op` comes back from `make_custom_op_for_export`, which set
    // `vtable` as the first field of `PerExportVtable`. Zero-offset cast.
    let entry = op as *const PerExportVtable;
    (*entry).name_cstr.as_ptr()
}

unsafe extern "C" fn vtable_get_ep_type(_op: *const OrtCustomOp) -> *const c_char {
    c"CPUExecutionProvider".as_ptr()
}

unsafe extern "C" fn vtable_create_kernel(
    op: *const OrtCustomOp,
    api: *const OrtApi,
    _info: *const OrtKernelInfo,
) -> *mut c_void {
    let entry = op as *const PerExportVtable;
    let raw_fn = (*entry).cached_dispatch_fn;
    if raw_fn == 0 {
        // Symbol not found at registration time — return null kernel so ORT
        // rejects this op. (No status-returning variant in V1 — M62c migrates
        // to CreateKernelV2 for proper error reporting.)
        return std::ptr::null_mut();
    }
    // SAFETY: `cached_dispatch_fn` was set by `resolve_self_symbol` to a
    // non-null pointer to a symbol codegen emitted with the `ExportFnPtr`
    // signature.
    let fn_ptr: ExportFnPtr = std::mem::transmute::<usize, ExportFnPtr>(raw_fn);
    let state = Box::new(NslOrtKernelState {
        api,
        fn_ptr,
        // Non-zero sentinel for stateless exports: the codegen-emitted typed
        // wrapper null-checks model_ptr and returns -1 when it is zero. For
        // @export functions that are not model methods, model_ptr is never
        // dereferenced — the null check is the only consumer. Use 1 so the
        // check passes without providing a real model handle.
        model_ptr: 1,
    });
    Box::into_raw(state) as *mut c_void
}

unsafe extern "C" fn vtable_kernel_destroy(state: *mut c_void) {
    if !state.is_null() {
        drop(Box::from_raw(state as *mut NslOrtKernelState));
    }
}

unsafe extern "C" fn vtable_get_input_type(
    _op: *const OrtCustomOp,
    _index: usize,
) -> ONNXTensorElementDataType {
    // v1 scope: f32 inputs only.
    ONNXTensorElementDataType::FLOAT
}

unsafe extern "C" fn vtable_get_input_count(_op: *const OrtCustomOp) -> usize {
    // One *formal* input slot, declared VARIADIC via
    // `vtable_get_input_char` below, so ORT accepts any actual input count
    // >= the variadic min arity (1). This lets multi-arg exports such as
    // `add(a, b)` register — the kernel reads the real count at compute
    // time via `KernelContext_GetInputCount` and forwards them all to the
    // NSL export. (Previously hardcoded non-variadic 1, which made ORT
    // reject any node with != 1 input: "input size 2 not in range
    // [min=1, max=1]".)
    1
}

unsafe extern "C" fn vtable_get_output_count(_op: *const OrtCustomOp) -> usize {
    1
}

unsafe extern "C" fn vtable_get_output_type(
    _op: *const OrtCustomOp,
    _index: usize,
) -> ONNXTensorElementDataType {
    ONNXTensorElementDataType::FLOAT
}

unsafe extern "C" fn vtable_get_input_char(
    _op: *const OrtCustomOp,
    _idx: usize,
) -> OrtCustomOpInputOutputCharacteristic {
    // VARIADIC (not REQUIRED) so a single formal input slot accepts N >= 1
    // actual inputs of the same type. Required for multi-arg exports like
    // `add(a, b)`; `GetVariadicInputMinArity`/`Homogeneity` below bound it
    // (min 1, homogeneous — consistent with v1's f32-only input scope).
    OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_VARIADIC
}

unsafe extern "C" fn vtable_get_output_char(
    _op: *const OrtCustomOp,
    _idx: usize,
) -> OrtCustomOpInputOutputCharacteristic {
    OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED
}

unsafe extern "C" fn vtable_get_input_mem_type(
    _op: *const OrtCustomOp,
    _idx: usize,
) -> OrtMemType {
    OrtMemType::DEFAULT
}

unsafe extern "C" fn vtable_get_variadic_min(_op: *const OrtCustomOp) -> i32 {
    1
}

unsafe extern "C" fn vtable_get_variadic_hom(_op: *const OrtCustomOp) -> i32 {
    1
}

// ---------------------------------------------------------------------------
// V2 vtable functions — ORT 1.16+ calls these when they are non-null,
// taking priority over V1 CreateKernel/KernelCompute.
// ---------------------------------------------------------------------------

/// V2 CreateKernel — mirrors V1 but reports errors via the return status
/// and writes the kernel state pointer through `kernel_out`.
unsafe extern "C" fn vtable_create_kernel_v2(
    op: *const OrtCustomOp,
    api: *const OrtApi,
    _info: *const OrtKernelInfo,
    kernel_out: *mut *mut c_void,
) -> *mut OrtStatus {
    let entry = op as *const PerExportVtable;
    let raw_fn = (*entry).cached_dispatch_fn;
    if raw_fn == 0 {
        // Dispatch symbol not found — zero out *kernel_out so ORT sees a
        // null kernel (not uninitialized stack garbage) and return an error
        // status so ORT rejects this op cleanly.
        if !kernel_out.is_null() {
            *kernel_out = std::ptr::null_mut();
        }
        let msg = c"NSL: __nsl_dispatch symbol not found in shared library".as_ptr();
        return ((*api).CreateStatus)(OrtErrorCode::ORT_INVALID_ARGUMENT, msg);
    }
    let fn_ptr: ExportFnPtr = std::mem::transmute::<usize, ExportFnPtr>(raw_fn);
    let state = Box::new(NslOrtKernelState {
        api,
        fn_ptr,
        // Non-zero sentinel for stateless exports: see vtable_create_kernel
        // above for the full rationale. model_ptr=1 bypasses the null check
        // in the codegen-emitted typed wrapper without being a real model ptr.
        model_ptr: 1,
    });
    *kernel_out = Box::into_raw(state) as *mut c_void;
    std::ptr::null_mut()
}

/// V2 KernelCompute — delegates to the shared `nsl_ort_kernel_compute` body.
unsafe extern "C" fn vtable_kernel_compute_v2(
    op_kernel: *mut c_void,
    context: *mut OrtKernelContext,
) -> *mut OrtStatus {
    if !op_kernel.is_null() {
        nsl_ort_kernel_compute(op_kernel, context);
    }
    std::ptr::null_mut()
}

unsafe extern "C" fn vtable_infer_output_shape_unused(
    _op: *const OrtCustomOp,
    _ctx: *mut OrtShapeInferContext,
) -> *mut OrtStatus {
    // Spec C v1 leaves shape inference to ORT's "same as first input"
    // default. Returning null here means "no shape-inference contribution".
    std::ptr::null_mut()
}

unsafe extern "C" fn vtable_get_start_version(_op: *const OrtCustomOp) -> i32 {
    1
}

unsafe extern "C" fn vtable_get_end_version(_op: *const OrtCustomOp) -> i32 {
    i32::MAX
}

unsafe extern "C" fn vtable_get_may_inplace(
    _input_index: *mut *mut i32,
    _output_index: *mut *mut i32,
) -> usize {
    0
}

unsafe extern "C" fn vtable_release_may_inplace(
    _input_index: *mut i32,
    _output_index: *mut i32,
) {
}

unsafe extern "C" fn vtable_get_alias_map(
    _input_index: *mut *mut i32,
    _output_index: *mut *mut i32,
) -> usize {
    0
}

unsafe extern "C" fn vtable_release_alias_map(
    _input_index: *mut i32,
    _output_index: *mut i32,
) {
}

// ---------------------------------------------------------------------------
// Symbol resolution
// ---------------------------------------------------------------------------

/// Resolve a symbol by name from this same `.so`/`.dll`. Returns 0 on
/// failure (symbol not present, dlsym/GetModuleHandle error, ...).
///
/// Used for two things in Spec C:
/// 1. `mod.rs::RegisterCustomOps` looks up the codegen-emitted
///    `nsl_get_num_exports` / `nsl_get_export_name` enumeration FFIs at
///    runtime. They may not exist in test binaries; we handle absence by
///    returning success with no domain.
/// 2. `make_custom_op_for_export` eagerly resolves each export's dispatch
///    fn at registration time and caches it in `cached_dispatch_fn`.
///
/// **macOS:** `dlsym(RTLD_SELF, name)` — searches only the calling library's
/// symbol table. Works even when ORT loaded us with `RTLD_LOCAL` (unlike
/// `RTLD_DEFAULT`, which searches only the global scope).
///
/// **Linux:** `dladdr(&resolve_self_symbol, &info)` to find our `.so` path,
/// `dlopen(path, RTLD_NOLOAD)` to retrieve the existing in-process handle
/// without re-executing init, then `dlsym(handle, name)`. `RTLD_DEFAULT`
/// would miss `RTLD_LOCAL` symbols; the explicit handle bypasses that limit.
///
/// **Windows:** `GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, &this_fn, &out)`
/// finds our DLL, then `GetProcAddress(out, name)`.
pub(crate) unsafe fn resolve_self_symbol(name_ptr: *const c_char) -> usize {
    #[cfg(target_os = "macos")]
    {
        extern "C" {
            fn dlsym(handle: *mut c_void, sym: *const c_char) -> *mut c_void;
        }
        // RTLD_SELF (-3) searches only the calling library — works even when
        // ORT loaded us with RTLD_LOCAL.
        const RTLD_SELF: *mut c_void = (-3isize) as *mut c_void;
        let p = dlsym(RTLD_SELF, name_ptr);
        p as usize
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        extern "C" {
            fn dladdr(addr: *const c_void, info: *mut DlInfo) -> i32;
            fn dlopen(filename: *const c_char, flags: i32) -> *mut c_void;
            fn dlsym(handle: *mut c_void, sym: *const c_char) -> *mut c_void;
        }
        #[repr(C)]
        struct DlInfo {
            dli_fname: *const c_char,
            dli_fbase: *mut c_void,
            dli_sname: *const c_char,
            dli_saddr: *mut c_void,
        }
        // RTLD_DEFAULT misses RTLD_LOCAL symbols on Linux. Workaround:
        // 1. dladdr on our own address → get the path of our .so.
        // 2. dlopen(path, RTLD_NOLOAD) → get the existing in-process handle
        //    without re-executing initializers or changing refcount.
        // 3. dlsym(handle, sym) finds RTLD_LOCAL symbols in our .so.
        let mut info = DlInfo {
            dli_fname: std::ptr::null(),
            dli_fbase: std::ptr::null_mut(),
            dli_sname: std::ptr::null(),
            dli_saddr: std::ptr::null_mut(),
        };
        let ok = dladdr(resolve_self_symbol as *const c_void, &mut info);
        if ok == 0 || info.dli_fname.is_null() {
            return 0;
        }
        // RTLD_LAZY=1, RTLD_NOLOAD=4 (glibc / musl).
        const RTLD_LAZY: i32 = 1;
        const RTLD_NOLOAD: i32 = 4;
        let handle = dlopen(info.dli_fname, RTLD_NOLOAD | RTLD_LAZY);
        if handle.is_null() {
            return 0;
        }
        let p = dlsym(handle, name_ptr);
        p as usize
    }
    #[cfg(windows)]
    {
        extern "system" {
            fn GetModuleHandleExW(
                flags: u32,
                module_name: *const u16,
                handle_out: *mut *mut c_void,
            ) -> i32;
            fn GetProcAddress(
                module: *mut c_void,
                name: *const c_char,
            ) -> *mut c_void;
        }
        const GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS: u32 = 0x4;
        const GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT: u32 = 0x2;

        let mut module: *mut c_void = std::ptr::null_mut();
        // Use this function's own address as the anchor — it's guaranteed
        // to live in our DLL. The `module_name` parameter is repurposed
        // (with the FROM_ADDRESS flag) to mean "find the module containing
        // this address".
        let ok = GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS
                | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            resolve_self_symbol as *const u16,
            &mut module,
        );
        if ok == 0 || module.is_null() {
            return 0;
        }
        GetProcAddress(module, name_ptr) as usize
    }
    #[cfg(not(any(unix, windows)))]
    {
        let _ = name_ptr;
        0
    }
}
