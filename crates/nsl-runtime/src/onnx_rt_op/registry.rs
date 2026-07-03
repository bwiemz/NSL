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
//! Each vtable's `CreateKernel` resolves the export by name *from this same
//! `.so`/`.dll`* via:
//! - Unix: `dlsym(RTLD_DEFAULT, name)`.
//! - Windows: `GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, &resolve_export_via_self_dlsym, &out)`
//!   then `GetProcAddress(out, name)`.
//!
//! Spec A's `nsl_model_lookup_function` requires a model handle, which the
//! ORT kernel doesn't have — ORT loads the `.so` via
//! `register_custom_ops_library` without ever calling `nsl_model_create`.
//! v1 is therefore restricted to stateless exports (`model_ptr = 0`); exports
//! that read weights would need M62c's session-config plumbing to receive a
//! handle.

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
}

// SAFETY: `PerExportVtable` contains only read-only function pointers and a
// `CString`. Once constructed and pushed into `VTABLES` it is never mutated.
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
/// (each kernel-create call dlsym's the export by name); kept in the
/// signature so a future revision can index a direct dispatch table.
pub fn make_custom_op_for_export(idx: i64, name: *const c_char) -> *const OrtCustomOp {
    let _ = idx;

    // SAFETY: caller (codegen-emitted `nsl_get_export_name`) hands us a
    // null-terminated C string with static lifetime in the same .so. We
    // copy into an owned CString so this registry doesn't depend on the
    // codegen-emitted storage living forever (in case future codegen
    // changes that contract).
    let name_cstr = unsafe { CStr::from_ptr(name) }.to_owned();

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
            // Post-V1 callbacks — Spec C registers V1 only. These slots
            // must be populated (ORT 1.16+ reads them unconditionally),
            // but the v2 paths are never invoked because we set the v1
            // KernelCompute slot above.
            CreateKernelV2: vtable_create_kernel_v2_unused,
            KernelComputeV2: vtable_kernel_compute_v2_unused,
            InferOutputShapeFn: vtable_infer_output_shape_unused,
            GetStartVersion: vtable_get_start_version,
            GetEndVersion: vtable_get_end_version,
            GetMayInplace: vtable_get_may_inplace,
            ReleaseMayInplace: vtable_release_may_inplace,
            GetAliasMap: vtable_get_alias_map,
            ReleaseAliasMap: vtable_release_alias_map,
        },
        name_cstr,
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
    let name_ptr = (*entry).name_cstr.as_ptr();
    let raw_fn = resolve_self_symbol(name_ptr);
    if raw_fn == 0 {
        // Symbol not found. Return null kernel; ORT treats this as a
        // create-kernel failure. (No status-returning variant in V1 —
        // M62c migrates to CreateKernelV2 for proper error reporting.)
        return std::ptr::null_mut();
    }
    // SAFETY: dlsym/GetProcAddress returned a non-null pointer to a symbol
    // codegen emitted with the `ExportFnPtr` signature.
    let fn_ptr: ExportFnPtr = std::mem::transmute::<usize, ExportFnPtr>(raw_fn);
    let state = Box::new(NslOrtKernelState {
        api,
        fn_ptr,
        model_ptr: 0, // v1: stateless exports only.
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
// V2 / 1.16+ vtable functions — populated with safe stubs.
//
// Spec C v1 uses the V1 KernelCompute path. ORT 1.22 still reads these slots
// during op registration even if it never invokes them through this op, so
// they must contain valid function pointers.
// ---------------------------------------------------------------------------

unsafe extern "C" fn vtable_create_kernel_v2_unused(
    _op: *const OrtCustomOp,
    _api: *const OrtApi,
    _info: *const OrtKernelInfo,
    _kernel_out: *mut *mut c_void,
) -> *mut OrtStatus {
    // Should never be called — Spec C registers via V1 CreateKernel.
    std::ptr::null_mut()
}

unsafe extern "C" fn vtable_kernel_compute_v2_unused(
    _op_kernel: *mut c_void,
    _context: *mut OrtKernelContext,
) -> *mut OrtStatus {
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
/// 2. Per-export `CreateKernel` looks up the export's NSL function
///    pointer by name. The codegen-emitted symbol lives in the same .so
///    as this code, so the same self-dlsym pattern works.
///
/// **Unix:** `dlsym(RTLD_DEFAULT, name)` — searches the process's loaded
/// symbol table including this `.so`.
///
/// **Windows:** `GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, &this_fn, &out)`
/// finds the module containing this function (i.e., our own DLL), then
/// `GetProcAddress(out, name)`. Mirrors the canonical pattern in
/// `c_api/mod.rs::nsl_dl_path_for_fn_addr`. We pass our own function's
/// address as the lookup anchor so the call works equally well from a
/// `.dll`, `.exe`, or test binary.
pub(crate) unsafe fn resolve_self_symbol(name_ptr: *const c_char) -> usize {
    #[cfg(unix)]
    {
        extern "C" {
            fn dlsym(handle: *mut c_void, sym: *const c_char) -> *mut c_void;
        }
        // RTLD_DEFAULT is a sentinel: NULL on Linux/macOS (POSIX). It tells
        // dlsym to search the whole process symbol space, which includes
        // this .so via the normal global symbol scope.
        const RTLD_DEFAULT: *mut c_void = std::ptr::null_mut();
        let p = dlsym(RTLD_DEFAULT, name_ptr);
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
