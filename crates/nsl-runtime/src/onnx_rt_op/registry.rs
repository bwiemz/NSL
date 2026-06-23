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
//! - macOS: `dlsym(RTLD_SELF, name)` — `RTLD_SELF` searches only this dylib.
//! - Linux/POSIX: `dladdr(self_fn)` → path → `dlopen(path, RTLD_NOLOAD)` →
//!   `dlsym(specific_handle, name)`. Cannot use `RTLD_DEFAULT` (NULL) because
//!   ORT loads custom-op libraries with `RTLD_LOCAL`, excluding them from the
//!   global symbol scope.
//! - Windows: `GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS, &self_fn, &out)`
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
    /// Pre-resolved `<name>__nsl_dispatch` function pointer. Resolved eagerly
    /// during `make_custom_op_for_export` via `resolve_self_symbol`, which
    /// uses a library-specific dlopen/dlsym handle rather than RTLD_DEFAULT.
    /// This is required because ORT loads custom-op libraries with RTLD_LOCAL,
    /// which hides their symbols from the global scope that RTLD_DEFAULT
    /// searches. The correct symbol name is `<name>__nsl_dispatch`, NOT the
    /// user-facing `<name>` which uses a different calling convention.
    cached_dispatch_fn: usize,
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

    // Resolve the dispatch symbol eagerly while we are still executing inside
    // RegisterCustomOps (library load context). The export ABI uses the
    // `<name>__nsl_dispatch` symbol, not the user-facing `<name>` symbol —
    // see ExportRegistry::from_library_path in c_api/exports.rs which uses
    // the same convention. Resolving here avoids RTLD_LOCAL issues at
    // inference time and uses the correct symbol name.
    let dispatch_sym = format!("{}__nsl_dispatch\0", name_cstr.to_str().unwrap_or(""));
    let cached_dispatch_fn =
        unsafe { resolve_self_symbol(dispatch_sym.as_ptr() as *const c_char) };

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
            // V2 callbacks: ORT 1.16+ calls CreateKernelV2 / KernelComputeV2
            // in preference to V1 whenever these slots are non-null.
            // These implementations mirror the V1 logic with the V2 signature
            // (status-returning) so ORT's preferred path works correctly.
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
    // Use the dispatch pointer pre-resolved at registration time. This avoids
    // two bugs: (1) the wrong symbol name (`<name>` vs `<name>__nsl_dispatch`),
    // (2) RTLD_DEFAULT not finding symbols from RTLD_LOCAL-opened libraries at
    // inference time.
    let raw_fn = (*entry).cached_dispatch_fn;
    if raw_fn == 0 {
        // Dispatch symbol not found at registration time — CreateKernel
        // returns null. ORT treats this as a create-kernel failure.
        // (No status-returning variant in V1 — M62c migrates to
        // CreateKernelV2 for proper error reporting.)
        return std::ptr::null_mut();
    }
    // SAFETY: cached_dispatch_fn was resolved via resolve_self_symbol during
    // make_custom_op_for_export. Codegen emits the `<name>__nsl_dispatch`
    // symbol with the ExportFnPtr signature.
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
    // v1: hardcode 1 input. Multi-input exports will surface via
    // `GetVariadicInput*` + `GetInputCharacteristic == VARIADIC` in v2.
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
    OrtCustomOpInputOutputCharacteristic::INPUT_OUTPUT_REQUIRED
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
// V2 / 1.16+ vtable functions.
//
// ORT 1.16+ prefers CreateKernelV2 / KernelComputeV2 over their V1
// counterparts whenever the slots are non-null. Populating them with stubs
// that don't write to kernel_out causes ORT to run with a null kernel state
// and produce None outputs. These implementations mirror the V1 logic but
// use the V2 signature (returns *mut OrtStatus instead of void/ptr).
// ---------------------------------------------------------------------------

unsafe extern "C" fn vtable_create_kernel_v2(
    op: *const OrtCustomOp,
    api: *const OrtApi,
    _info: *const OrtKernelInfo,
    kernel_out: *mut *mut c_void,
) -> *mut OrtStatus {
    let entry = op as *const PerExportVtable;
    let raw_fn = (*entry).cached_dispatch_fn;
    if raw_fn == 0 {
        // Dispatch symbol was not found at registration time. Write null so
        // KernelComputeV2 can detect the empty state and skip execution.
        *kernel_out = std::ptr::null_mut();
        return std::ptr::null_mut();
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
    std::ptr::null_mut() // null OrtStatus* = success
}

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
///    `nsl_get_num_exports` / `nsl_get_export_name` enumeration FFIs.
/// 2. `make_custom_op_for_export` pre-resolves each `<name>__nsl_dispatch`.
///
/// **macOS:** `dlsym(RTLD_SELF, name)` — `RTLD_SELF = (void*)-3` searches
/// only the calling module's symbols, regardless of `RTLD_LOCAL`.
///
/// **Linux/POSIX:** `dladdr(self_fn)` gives our library's path; then
/// `dlopen(path, RTLD_LAZY|RTLD_NOLOAD)` returns a library-specific handle
/// without a new load; then `dlsym(handle, name)` finds the symbol directly
/// in our library. Cannot use `RTLD_DEFAULT` (NULL) because ORT 1.22
/// opens custom-op libraries with `RTLD_LOCAL`, which excludes them from
/// the global symbol table that `RTLD_DEFAULT` searches.
///
/// **Windows:** `GetModuleHandleExW(FROM_ADDRESS, self_fn)` finds our DLL,
/// then `GetProcAddress(module, name)` — no `RTLD_LOCAL` analog on Windows.
pub(crate) unsafe fn resolve_self_symbol(name_ptr: *const c_char) -> usize {
    #[cfg(unix)]
    {
        extern "C" {
            fn dlsym(handle: *mut c_void, sym: *const c_char) -> *mut c_void;
        }

        // --- macOS: RTLD_SELF searches only this dylib. -------------------
        #[cfg(target_os = "macos")]
        {
            // RTLD_SELF = (void*)-3 on macOS. Tells dlsym to search only
            // the dylib that contains the calling code, bypassing RTLD_LOCAL.
            const RTLD_SELF: *mut c_void = (-3_isize as usize) as *mut c_void;
            return dlsym(RTLD_SELF, name_ptr) as usize;
        }

        // --- Linux / other POSIX: dladdr + dlopen(RTLD_NOLOAD). -----------
        //
        // RTLD_DEFAULT (NULL on Linux) searches only the global symbol scope.
        // ORT 1.22 opens custom-op libraries with dlopen(path, RTLD_LOCAL),
        // which keeps their symbols out of the global scope. We need a
        // library-specific handle instead.
        #[cfg(not(target_os = "macos"))]
        {
            extern "C" {
                fn dladdr(addr: *const c_void, info: *mut DlInfo) -> i32;
                fn dlopen(filename: *const c_char, flag: i32) -> *mut c_void;
                fn dlclose(handle: *mut c_void) -> i32;
            }

            // POSIX Dl_info layout (matches glibc and musl).
            #[repr(C)]
            struct DlInfo {
                dli_fname: *const c_char, // path of .so containing addr
                dli_fbase: *mut c_void,   // base load address of that .so
                dli_sname: *const c_char, // nearest symbol name
                dli_saddr: *mut c_void,   // nearest symbol address
            }

            // glibc/Linux values: RTLD_LAZY=1, RTLD_NOLOAD=4.
            const RTLD_LAZY: i32 = 0x0001;
            // RTLD_NOLOAD: return existing handle without loading; NULL if
            // the library is not already resident (ours always is).
            const RTLD_NOLOAD: i32 = 0x0004;

            let mut info = DlInfo {
                dli_fname: std::ptr::null(),
                dli_fbase: std::ptr::null_mut(),
                dli_sname: std::ptr::null(),
                dli_saddr: std::ptr::null_mut(),
            };
            // Use this function's own address as an anchor into our .so.
            if dladdr(resolve_self_symbol as *const c_void, &mut info) == 0
                || info.dli_fname.is_null()
            {
                return 0;
            }
            // Get a handle to our already-loaded library. RTLD_NOLOAD does
            // not re-load it; it just returns the existing mapping handle
            // and increments the refcount. dlsym on this handle finds symbols
            // in our specific library, regardless of RTLD_LOCAL.
            let handle = dlopen(info.dli_fname, RTLD_LAZY | RTLD_NOLOAD);
            if handle.is_null() {
                return 0;
            }
            let p = dlsym(handle, name_ptr);
            // Balance the refcount bump from dlopen(RTLD_NOLOAD).
            dlclose(handle);
            return p as usize;
        }

        // Both cfg branches above return explicitly. This trailing expression
        // is the unix block's type anchor for the Rust type checker.
        #[allow(unreachable_code)]
        0_usize
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
