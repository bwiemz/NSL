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
/// Note: The C API speaks the CANONICAL runtime dtype tag space verbatim
/// (`crate::tensor::DTYPE_*`: 0=f64, 1=f32, 2=fp16, 3=bf16, 4=int8,
/// 5=fp8e4m3, 6=fp8e5m2, 7=u16-token, 8=u16-segment, 9=i32). The historical
/// inverted 0=f32/1=f64 convention was removed in the P4 item-16 dtype/ABI
/// migration; `capi_dtype_to_nsl` and `nsl_dtype_to_capi` survive only as
/// validating identity chokepoints (the `dtype_abi_lock` test pins this).
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
    /// Canonical NSL dtype tag: 0=f64, 1=f32, 2=f16, 3=bf16, 4=int8,
    /// 5=fp8e4m3, 6=fp8e5m2, 7=u16-token, 8=u16-segment, 9=int32.
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
// Dtype validation at the C API boundary
// ---------------------------------------------------------------------------
//
// P4 item 16: the C API uses the canonical runtime tag space verbatim, so
// these two functions are VALIDATING IDENTITY maps, not conversions. They are
// kept (rather than deleted) so the boundary crossing stays an explicit,
// greppable chokepoint — and so an unknown tag fails loudly here instead of
// the historical silent fall-back-to-f64, which mislabeled every unmapped
// dtype (e.g. the old C-API int64/uint8 slots) as f64.
//
// The two directions fail DIFFERENTLY, on purpose:
//
//   * `capi_dtype_to_nsl` reads a HOST-CONTROLLED tag off an `NslTensorDesc`
//     that arrived through the exported `nsl_desc_to_tensor`. Bad input from
//     across an FFI boundary is an error to report, not a reason to kill the
//     embedding process, so it returns `Option` and its callers refuse. Do
//     not reintroduce the `abort()` it used to have.
//   * `nsl_dtype_to_capi` reads an INTERNAL tag. Its abort branch is
//     structurally dead — `tensor/mod.rs` assigns 0..=9 and DTYPE_CUSTOM_START
//     is 256, so no tag in 10..=255 exists — and is kept only as a trip-wire
//     for a future tag added outside that range. Nothing gates it, because
//     nothing can reach it.

/// Human-readable list of the canonical dtype tags, shared by every
/// refusal message that reports an out-of-range tag.
pub(crate) const CAPI_DTYPE_TAGS: &str = "0=f64, 1=f32, 2=f16, 3=bf16, 4=int8, \
     5=fp8e4m3, 6=fp8e5m2, 7=u16-token, 8=u16-segment, 9=int32";

/// Validate a C API dtype tag and return it as the canonical internal tag.
///
/// Returns `None` for a tag outside `0..=9`. This input is HOST-CONTROLLED —
/// it arrives on an `NslTensorDesc` handed to the exported `nsl_desc_to_tensor`
/// — so it must not be able to take the process down. Until this returned
/// `Option` it called `std::process::abort()`, which killed the embedding
/// process (a Python interpreter, under the nslpy bridge) on a mislabeled
/// desc. `nsl_dispatch_apply_result` treats the identical input as a plain
/// `-1` + `set_error` eighty lines below; this now mirrors it.
pub fn capi_dtype_to_nsl(capi_dtype: i32) -> Option<u16> {
    if !(0..=9).contains(&capi_dtype) {
        return None;
    }
    Some(capi_dtype as u16)
}

/// Validate a canonical internal dtype tag for export through the C API.
pub fn nsl_dtype_to_capi(nsl_dtype: u16) -> i32 {
    if nsl_dtype > 9 && nsl_dtype < crate::tensor::DTYPE_CUSTOM_START {
        eprintln!(
            "nsl: internal dtype tag {nsl_dtype} has no C API representation \
             (canonical built-in tags are 0..=9)"
        );
        std::process::abort();
    }
    nsl_dtype as i32
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
// Dispatch ownership mode (item 7)
// ---------------------------------------------------------------------------
//
// The codegen-emitted typed wrapper hands every impl result tensor to the
// runtime via `nsl_tensor_to_desc_ffi(result_tensor, scratch_desc)`, and the
// dispatch wrapper then folds scratch onto the caller's desc via
// `nsl_dispatch_apply_result`. Neither generated symbol can change signature
// (the `ExportRegistry` transmutes any dlsym'd `<name>__nsl_dispatch` to
// `ExportFnPtr` — a silent-UB hazard if the ABI forked), so the two new
// ownership-model entry points communicate with those runtime helpers through
// a THREAD-LOCAL mode instead:
//
//   - `Into`  (`nsl_model_call_into`):  capacity-checked deep-copy into
//     caller-owned buffers; the captured impl tensors are freed by the entry
//     point afterwards (the legacy `nsl_model_call` path leaks them by
//     contract — its callers alias the leaked tensor's shape/strides).
//   - `Alloc` (`nsl_model_call_alloc`): no caller buffers; the captured impl
//     tensors are wrapped into `DLManagedTensor`s whose deleter transfers
//     ownership to the consumer.
//
// Dispatch is synchronous on the calling thread, and NSL impl code never
// re-enters the model C API mid-dispatch, so a thread-local is sound. With
// the mode unarmed (plain `nsl_model_call`) every helper behaves
// bit-identically to the pre-item-7 runtime.
//
// **The thread-local is PER-IMAGE.** A `nsl build --shared-lib` artifact
// statically links its own runtime copy, so its generated wrapper calls the
// .so's `nsl_tensor_to_desc_ffi` / `nsl_dispatch_apply_result` — which read
// the .so's `DISPATCH_MODE`, not a host-linked runtime's. The entry points
// therefore never touch `DISPATCH_MODE` directly: they arm/release/finish
// through the three `nsl_dispatch_ownership_*` FFIs below, resolved via
// dlsym FROM THE SAME IMAGE as the dispatch wrapper (`ExportRegistry` owns
// that dlopen handle). In the single-image production topology (nslpy binds
// the model .so directly) those pointers are simply the image's own
// functions. This also keeps every allocation/free pair same-image (slab
// bookkeeping is per-image state).
enum DispatchMode {
    Into {
        /// Byte capacity of each caller output slot's `data` buffer.
        capacities: Vec<u64>,
        /// Next output slot index to be applied (monotonic per dispatch).
        cursor: usize,
        /// Impl result tensors captured via `nsl_tensor_to_desc_ffi`.
        captured: Vec<i64>,
    },
    Alloc {
        captured: Vec<i64>,
    },
}

impl DispatchMode {
    fn captured_mut(&mut self) -> &mut Vec<i64> {
        match self {
            DispatchMode::Into { captured, .. } => captured,
            DispatchMode::Alloc { captured } => captured,
        }
    }
    fn captured(&self) -> &[i64] {
        match self {
            DispatchMode::Into { captured, .. } => captured,
            DispatchMode::Alloc { captured } => captured,
        }
    }
}

thread_local! {
    static DISPATCH_MODE: RefCell<Option<DispatchMode>> = const { RefCell::new(None) };
}

/// Release the impl result tensors captured during a dispatch — one decref
/// per capture, unconditionally.
///
/// **Every captured pointer carries its own owning reference.** A compiled
/// NSL function's return path hands back an OWNING reference even when the
/// returned value is a parameter or a model weight (verified empirically:
/// 3000 `fn ident(x) -> x` dispatches through `nsl_model_call_into` neither
/// double-free nor grow). So:
///   - a fresh result arrives at refcount 1 and this decref frees it (the
///     legacy alias-and-leak path is what leaks it);
///   - a returned INPUT arrives at 2 (the wrapper's borrow + the return's
///     incref); the typed wrapper's own input free takes one, this takes the
///     other;
///   - a returned WEIGHT arrives at 2 (the model's + the return's); this
///     takes the return's, leaving the model's intact for `nsl_model_destroy`.
///
/// An earlier version SKIPPED registered weights here, on the theory that a
/// returned weight arrives holding only the model's single reference. That
/// stranded the return's reference on every call, so weight storage was
/// never freed even after `nsl_model_destroy`. Do not reintroduce the skip
/// without re-establishing the return-incref contract it depends on.
///
/// Duplicates are still coalesced: the same pointer captured twice (an
/// export returning the same tensor in two output slots) holds ONE returned
/// reference, not two.
fn free_captured(captured: &[i64], _model: &NslModel) {
    let mut seen: Vec<i64> = Vec::with_capacity(captured.len());
    for &ptr in captured {
        if ptr == 0 || seen.contains(&ptr) {
            continue;
        }
        seen.push(ptr);
        crate::tensor::nsl_tensor_free(ptr);
    }
}

/// Arm THIS image's dispatch ownership mode. `mode`: 1 = Into (capacity
/// contract; `caps_ptr`/`n_caps` describe the caller's `uint64_t` capacity
/// array), 2 = Alloc (capture-only). Called by the ownership entry points
/// through a registry-dlsym'd pointer so the armed thread-local lives in the
/// same image as the dispatch wrapper's runtime helpers.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_dispatch_ownership_arm(mode: i64, caps_ptr: i64, n_caps: i64) -> i64 {
    let armed = match mode {
        1 => {
            let capacities: Vec<u64> = if n_caps > 0 && caps_ptr != 0 {
                unsafe { std::slice::from_raw_parts(caps_ptr as *const u64, n_caps as usize) }
                    .to_vec()
            } else {
                Vec::new()
            };
            DispatchMode::Into { capacities, cursor: 0, captured: Vec::new() }
        }
        2 => DispatchMode::Alloc { captured: Vec::new() },
        _ => {
            set_error(format!(
                "nsl_dispatch_ownership_arm: unknown mode {mode} (1=into, 2=alloc)\0"
            ));
            return -1;
        }
    };
    DISPATCH_MODE.with(|m| *m.borrow_mut() = Some(armed));
    0
}

/// Disarm THIS image's dispatch ownership mode and free the captured impl
/// result tensors (weight-guarded via [`free_captured`]). This is both the
/// success epilogue of `nsl_model_call_into` (the payload was deep-copied,
/// the impl tensors are ours to release — the legacy path leaks them) and
/// the error epilogue of every armed dispatch. Idempotent: a second call
/// finds the mode already taken.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_dispatch_ownership_release(model_ptr: i64) -> i64 {
    let mode = DISPATCH_MODE.with(|m| m.borrow_mut().take());
    if let Some(mode) = mode {
        if model_ptr != 0 {
            let model = unsafe { &*(model_ptr as *const NslModel) };
            free_captured(mode.captured(), model);
        } else {
            for &ptr in mode.captured() {
                if ptr != 0 {
                    crate::tensor::nsl_tensor_free(ptr);
                }
            }
        }
    }
    0
}

/// Disarm THIS image's Alloc mode and transfer the captured impl results to
/// the caller as ownership-transferring `DLManagedTensor`s. On any refusal
/// the captured set is released, already-wrapped outputs are unwound via
/// their deleters, every slot is nulled, and the error lands in THIS image's
/// thread-local (a cross-image host reads it off the model library handle,
/// exactly like the generated wrappers' refusals).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_dispatch_ownership_finish_alloc(
    model_ptr: i64,
    name_ptr: i64,
    out_dl_ptr: i64,
    num_outputs: i64,
) -> i64 {
    // Null every slot up front so EVERY refusal below satisfies the
    // documented contract ("on any refusal every slot is NULL and nothing
    // needs freeing") without each branch having to remember. A host that
    // frees non-NULL slots after rc != 0 would otherwise chase whatever
    // garbage its stack held.
    if out_dl_ptr != 0 && num_outputs > 0 {
        let slots = out_dl_ptr as *mut *mut crate::dlpack::DLManagedTensor;
        for i in 0..num_outputs as usize {
            unsafe { *slots.add(i) = std::ptr::null_mut() };
        }
    }
    let captured = match DISPATCH_MODE.with(|m| m.borrow_mut().take()) {
        Some(DispatchMode::Alloc { captured }) => captured,
        Some(other) => {
            // Wrong mode — release what was captured and refuse.
            if model_ptr != 0 {
                let model = unsafe { &*(model_ptr as *const NslModel) };
                free_captured(other.captured(), model);
            }
            set_error(
                "nsl_dispatch_ownership_finish_alloc: dispatch was not armed in \
                 alloc mode\0"
                    .to_string(),
            );
            return -1;
        }
        None => Vec::new(),
    };
    if model_ptr == 0 {
        for &ptr in &captured {
            if ptr != 0 {
                crate::tensor::nsl_tensor_free(ptr);
            }
        }
        set_error("nsl_dispatch_ownership_finish_alloc: null model pointer\0".to_string());
        return -1;
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    let name = if name_ptr != 0 {
        unsafe { CStr::from_ptr(name_ptr as *const c_char) }
            .to_string_lossy()
            .into_owned()
    } else {
        String::from("<unknown>")
    };

    if captured.len() != num_outputs.max(0) as usize {
        free_captured(&captured, model);
        set_error(format!(
            "nsl_model_call_alloc: export '{}' produced {} result tensor(s) but \
             the caller requested {} output slot(s)\0",
            name,
            captured.len(),
            num_outputs
        ));
        return -1;
    }

    let out_slots = out_dl_ptr as *mut *mut crate::dlpack::DLManagedTensor;
    for (i, &tensor_ptr) in captured.iter().enumerate() {
        // NO extra incref for weight-aliased results. The capture already
        // holds the return path's owning reference (see `free_captured`),
        // and that is exactly the reference the consumer's deleter consumes;
        // the model keeps its own. An earlier version added one here on the
        // theory that the return did not incref — which stranded a reference
        // and leaked the weight storage past `nsl_model_destroy`.
        match crate::dlpack::nsl_tensor_to_dlpack_owned(tensor_ptr) {
            Ok(managed) => {
                unsafe { *out_slots.add(i) = managed };
            }
            Err(why) => {
                // Unwind: release the outputs already wrapped (their owned
                // deleters free the tensors), free the not-yet-wrapped
                // remainder, and null every slot so the caller holds nothing.
                for k in 0..i {
                    unsafe {
                        crate::dlpack::nsl_dlpack_free(*out_slots.add(k) as i64);
                        *out_slots.add(k) = std::ptr::null_mut();
                    }
                }
                for j in i..captured.len() {
                    unsafe { *out_slots.add(j) = std::ptr::null_mut() };
                }
                free_captured(&captured[i..], model);
                set_error(format!(
                    "nsl_model_call_alloc: output {i} of export '{}' cannot \
                     transfer ownership via DLPack: {why}\0",
                    name
                ));
                return -1;
            }
        }
    }
    0
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
///
/// **Minor 1 (item 7):** adds `nsl_model_call_into`, `nsl_model_call_alloc`,
/// `nsl_model_get_export_signature`, `nsl_dispatch_apply_scalar_result`, and
/// makes `nsl_model_call_dlpack` / `nsl_model_forward_dlpack` actually return
/// ownership-transferring outputs (they previously refused every
/// tensor-returning export — no working caller existed, and the refusal note
/// was contracted to be deleted with this change, so this is additive, not
/// breaking). `NslTensorDesc` stays 48 bytes.
pub const NSL_ABI_VERSION_MINOR: u32 = 1;

/// Return the runtime's C-ABI version packed as `(major << 16) | minor`.
///
/// Hosts can call this immediately after loading `libnsl_runtime` and compare
/// against the `NSL_ABI_VERSION_*` macros emitted into the generated C header
/// to detect runtime/header skew before making any other call.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_abi_version() -> i64 {
    ((NSL_ABI_VERSION_MAJOR as i64) << 16) | (NSL_ABI_VERSION_MINOR as i64)
}

// ---------------------------------------------------------------------------
// FFI: Error handling
// ---------------------------------------------------------------------------

/// Get the last error message. Returns a null-terminated C string pointer,
/// or a pointer to an empty string if no error is set.
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
        unsafe extern "system" {
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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

/// Resolve an export's dispatch fn-pointer, mirroring `nsl_model_call`'s
/// refusal shapes with the given `caller` prefix. Sets the thread-local
/// error and returns `Err(())` on every refusal.
fn resolve_export_fn(
    caller: &str,
    model_ptr: i64,
    name_ptr: i64,
) -> Result<(String, crate::c_api::exports::ExportFnPtr), ()> {
    if model_ptr == 0 || name_ptr == 0 {
        set_error(format!("{caller}: null model or name pointer\0"));
        return Err(());
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    let registry = match &model.exports {
        Some(r) => r,
        None => {
            set_error(format!(
                "{caller}: model created without export registry; \
                 use nsl_model_create_with_lib(weights, lib)\0"
            ));
            return Err(());
        }
    };
    let name = unsafe { CStr::from_ptr(name_ptr as *const c_char) }
        .to_string_lossy()
        .into_owned();
    match registry.lookup(&name) {
        Some(p) => Ok((name, p)),
        None => {
            let available = registry.available_names();
            set_error(format!(
                "{caller}: export '{}' not in registry. Available: {:?}\0",
                name, available
            ));
            Err(())
        }
    }
}

/// Caller-allocates dispatch with an explicit capacity contract
/// (item 7, ownership model A).
///
/// Same ABI as [`nsl_model_call`] plus `out_capacities_ptr`: an array of
/// `num_outputs` u64s, where `out_capacities[i]` is the allocated byte size
/// of `outputs[i].data`. The output-desc contract differs from
/// `nsl_model_call` in two load-bearing ways:
///
///   - **An undersized buffer REFUSES** (rc = -1) and the error text reports
///     the required byte count — instead of the legacy path's unchecked
///     memcpy, which silently overran the caller's heap whenever the caller
///     guessed the output size wrong. A refusal-and-retry probe is also the
///     supported sizing strategy for exports whose output shape is symbolic
///     (batch-dependent).
///   - **Shape/strides are deep-copied into caller-owned arrays.**
///     `outputs[i].shape` / `.strides` must point at caller arrays, and
///     `outputs[i].ndim` ON ENTRY declares their slot capacity; on success
///     `ndim` is set to the result rank. (The legacy path mirrors pointers
///     into a leaked impl tensor — valid only until someone fixes the leak.)
///
/// The impl result tensor is freed before returning: this path does NOT
/// leak per call. Exports that return a model weight directly are handled
/// (the weight is copied out and NOT freed).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_model_call_into(
    model_ptr: i64,
    name_ptr: i64,
    inputs_ptr: i64,
    num_inputs: i64,
    outputs_ptr: i64,
    num_outputs: i64,
    out_capacities_ptr: i64,
) -> i64 {
    let (name, fn_ptr) = match resolve_export_fn("nsl_model_call_into", model_ptr, name_ptr) {
        Ok(v) => v,
        Err(()) => return -1,
    };
    if num_outputs > 0 && (outputs_ptr == 0 || out_capacities_ptr == 0) {
        set_error(format!(
            "nsl_model_call_into: num_outputs={num_outputs} but the output desc \
             array or the capacity array is null\0"
        ));
        return -1;
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    let ffis = match model.exports.as_ref().and_then(|r| r.ownership_ffis()) {
        Some(f) => f,
        None => {
            set_error(
                "nsl_model_call_into: this artifact predates the item-7 ownership \
                 FFIs (nsl_dispatch_ownership_*) — rebuild it with the current \
                 compiler\0"
                    .to_string(),
            );
            return -1;
        }
    };
    capi_trace(format!(
        "model_call_into name={} num_inputs={} num_outputs={}",
        name, num_inputs, num_outputs
    ));

    // Arm/dispatch/release run through the MODEL IMAGE's FFIs so the mode
    // thread-local, the capacity checks, and every free stay same-image with
    // the generated wrapper's runtime helpers (see the DispatchMode docs).
    if unsafe { (ffis.arm)(1, out_capacities_ptr, num_outputs) } != 0 {
        return -1;
    }
    let rc = unsafe { fn_ptr(model_ptr, inputs_ptr, num_inputs, outputs_ptr, num_outputs) };
    // Success or refusal, the impl result tensors are the runtime's to
    // release: on success their payload was deep-copied into caller buffers;
    // on refusal the caller never saw them. (The legacy path must NOT free —
    // its callers alias the leaked tensor. This path deep-copies, so it must.)
    unsafe { (ffis.release)(model_ptr) };
    rc
}

/// NSL-allocates dispatch: outputs transfer ownership via DLPack
/// (item 7, ownership model B).
///
/// `out_dl_ptr` is an array of `num_outputs` slots of `*mut DLManagedTensor`.
/// On rc == 0 each slot holds a DLManagedTensor whose **deleter releases the
/// underlying NSL tensor exactly once** — consuming the capsule in
/// `torch.utils.dlpack.from_dlpack` hands the memory to torch's GC, which is
/// what makes the resulting torch tensor *owned*. A host that does not
/// consume a slot must release it with `nsl_dlpack_free`, exactly once.
///
/// Refusals (rc = -1, error text says which): GPU-resident results, dtypes
/// with no DLPack representation, scalar-returning exports (no DLPack
/// scalar; use `nsl_model_call_into`), and everything `nsl_model_call`
/// itself refuses. Refusal is leak-free: any tensors already produced are
/// released before returning and every slot is nulled.
///
/// `tape_id` does not survive this path (DLPack has no field for it):
/// alloc outputs are terminal for NSL autodiff. Use the desc-based
/// `nsl_model_forward_grad` for grad-tracked forwards.
///
/// **Image lifetime:** each output's deleter is code inside the model's
/// shared library. The library must stay loaded until every transferred
/// output has been released — unloading it first leaves dangling deleter
/// pointers that the consumer's GC will call. nslpy keeps model libraries
/// mapped for the process lifetime for exactly this reason.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_model_call_alloc(
    model_ptr: i64,
    name_ptr: i64,
    inputs_ptr: i64,
    num_inputs: i64,
    out_dl_ptr: i64,
    num_outputs: i64,
) -> i64 {
    let (name, fn_ptr) = match resolve_export_fn("nsl_model_call_alloc", model_ptr, name_ptr) {
        Ok(v) => v,
        Err(()) => return -1,
    };
    model_call_alloc_core(
        "nsl_model_call_alloc",
        &name,
        name_ptr,
        fn_ptr,
        model_ptr,
        inputs_ptr,
        num_inputs,
        out_dl_ptr,
        num_outputs,
    )
}

/// Shared alloc-dispatch core for `nsl_model_call_alloc` and the DLPack
/// entry points. Expects the export already resolved; owns the whole
/// capture → wrap → publish sequence including leak-free unwinding.
#[allow(clippy::too_many_arguments)]
fn model_call_alloc_core(
    caller: &str,
    name: &str,
    name_ptr: i64,
    fn_ptr: crate::c_api::exports::ExportFnPtr,
    model_ptr: i64,
    inputs_ptr: i64,
    num_inputs: i64,
    out_dl_ptr: i64,
    num_outputs: i64,
) -> i64 {
    if num_outputs > 0 && out_dl_ptr == 0 {
        set_error(format!(
            "{caller}: num_outputs={num_outputs} but the DLPack output slot \
             array is null\0"
        ));
        return -1;
    }
    // Null every output slot BEFORE anything can fail. The contract ("on any
    // refusal every slot is NULL and nothing needs freeing") has to hold for
    // refusals raised inside the dispatch too — e.g. a scalar-returning
    // export refuses in `nsl_dispatch_apply_scalar_result`, which returns
    // rc != 0 below and never reaches `finish_alloc`'s own nulling. A host
    // that frees non-NULL slots after rc = -1 would otherwise chase whatever
    // its stack held.
    if out_dl_ptr != 0 && num_outputs > 0 {
        let slots = out_dl_ptr as *mut *mut crate::dlpack::DLManagedTensor;
        for i in 0..num_outputs as usize {
            unsafe { *slots.add(i) = std::ptr::null_mut() };
        }
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    let ffis = match model.exports.as_ref().and_then(|r| r.ownership_ffis()) {
        Some(f) => f,
        None => {
            set_error(format!(
                "{caller}: this artifact predates the item-7 ownership FFIs \
                 (nsl_dispatch_ownership_*) — rebuild it with the current \
                 compiler\0"
            ));
            return -1;
        }
    };

    // Scratch descs: the typed wrapper writes result metadata here and the
    // Alloc-mode `nsl_dispatch_apply_result` deliberately ignores them (the
    // captured tensor pointer carries everything).
    let mut scratch: Vec<NslTensorDesc> =
        (0..num_outputs.max(0)).map(|_| NslTensorDesc::default()).collect();

    capi_trace(format!(
        "model_call name={} num_inputs={} num_outputs={}",
        name, num_inputs, num_outputs
    ));

    // Same-image discipline: arm/finish/release run inside the model image —
    // see the DispatchMode docs.
    if unsafe { (ffis.arm)(2, 0, 0) } != 0 {
        return -1;
    }
    let rc = unsafe {
        fn_ptr(
            model_ptr,
            inputs_ptr,
            num_inputs,
            scratch.as_mut_ptr() as i64,
            num_outputs,
        )
    };
    if rc != 0 {
        unsafe { (ffis.release)(model_ptr) };
        return rc;
    }
    let rc = unsafe { (ffis.finish_alloc)(model_ptr, name_ptr, out_dl_ptr, num_outputs) };
    drop(scratch);
    rc
}

/// DLPack variant of `nsl_model_call`. Bridges `DLManagedTensor*`-ABI
/// inputs and outputs to the `NslTensorDesc*`-ABI export entry-point
/// and reuses the same dispatcher.
///
/// Outputs use the **ownership-transfer** model (item 7): each output slot
/// receives a `DLManagedTensor*` whose deleter releases the underlying NSL
/// tensor exactly once — see [`nsl_model_call_alloc`], which this function
/// shares its output core with. Inputs remain borrow-imported: the caller's
/// producers keep ownership of input memory and NSL frees only its borrowed
/// wrappers before returning.
#[unsafe(no_mangle)]
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
    //
    // `dlpack_to_nsl_tensor` returns 0 for any DLPack dtype outside
    // {f64, f32, f16, bf16, int8, int32} — and torch's DEFAULT integer dtype
    // is int64, so `NslModel.forward(torch.tensor([1, 2, 3]))` lands here.
    // Before this check the 0 was collected like any other pointer and fed
    // straight to `nsl_tensor_to_desc`, which dereferenced it: the bridge
    // SEGFAULTED the embedding process instead of returning -1. Refuse here,
    // naming the input INDEX and the offending code/bits so the caller can
    // fix the one argument that is wrong.
    //
    // A caller that declares an arity but hands over no array is refused HERE,
    // before anything is built. Merely SKIPPING the import loop (what the
    // `dl_inputs_ptr != 0` conjunct below used to do on its own) left
    // `input_descs` empty while still forwarding the caller's `num_inputs` to
    // `nsl_model_call` — and `Vec::as_mut_ptr()` on an empty `Vec` is
    // `NonNull::dangling()`, i.e. 0x8. The dispatch wrapper then called
    // `nsl_desc_to_tensor(8)`, whose `desc_ptr == 0` check does not catch 8,
    // and `desc_to_nsl_tensor` read `desc.ndim` off address 8: SIGSEGV, from a
    // caller that passed an honest NULL. This is the one path that converted a
    // NULL into a dangling pointer before the emitted null guard could see it.
    if num_inputs > 0 && dl_inputs_ptr == 0 {
        set_error(format!(
            "nsl_model_call_dlpack: num_inputs={num_inputs} but the DLPack input \
             array is null\0"
        ));
        return -1;
    }
    let mut input_tensor_ptrs: Vec<i64> = Vec::new();
    if num_inputs > 0 {
        let dlpacks = unsafe {
            std::slice::from_raw_parts(
                dl_inputs_ptr as *const *mut DLManagedTensor,
                num_inputs as usize,
            )
        };
        for (i, &dl) in dlpacks.iter().enumerate() {
            if dl.is_null() {
                for &p in &input_tensor_ptrs {
                    crate::tensor::nsl_tensor_free(p);
                }
                set_error(format!(
                    "nsl_model_call_dlpack: input {i} is a null DLManagedTensor*\0"
                ));
                return -1;
            }
            let managed = unsafe { &*dl };
            let ptr = crate::dlpack::dlpack_to_nsl_tensor(managed);
            if ptr == 0 {
                let dt = managed.dl_tensor.dtype;
                // Free the inputs imported before the bad one — an early
                // return must not leak the tensors that already succeeded.
                for &p in &input_tensor_ptrs {
                    crate::tensor::nsl_tensor_free(p);
                }
                set_error(format!(
                    "nsl_model_call_dlpack: input {i} has an unsupported DLPack dtype \
                     (code={}, bits={}, lanes={}); supported: float64/float32/float16, \
                     bfloat16, int8, int32 (DLPack code 0=int, 1=uint, 2=float, 4=bfloat)\0",
                    dt.code, dt.bits, dt.lanes
                ));
                return -1;
            }
            input_tensor_ptrs.push(ptr);
        }
    }
    let mut input_descs: Vec<NslTensorDesc> = input_tensor_ptrs
        .iter()
        .map(|&p| {
            let mut desc = NslTensorDesc::default();
            nsl_tensor_to_desc(p, &mut desc);
            desc
        })
        .collect();
    // Resolve AFTER the input-import refusals above: the contract gate proves
    // an unsupported-dtype input is refused BEFORE dispatch by asserting the
    // `model_call name=` trace (emitted inside the alloc core) is absent.
    let (name, fn_ptr) =
        match resolve_export_fn("nsl_model_call_dlpack", model_ptr, name_ptr) {
            Ok(v) => v,
            Err(()) => {
                for &p in &input_tensor_ptrs {
                    crate::tensor::nsl_tensor_free(p);
                }
                return -1;
            }
        };
    let rc = model_call_alloc_core(
        "nsl_model_call_dlpack",
        &name,
        name_ptr,
        fn_ptr,
        model_ptr,
        if input_descs.is_empty() { 0 } else { input_descs.as_mut_ptr() as i64 },
        num_inputs,
        dl_outputs_ptr,
        num_outputs,
    );
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
#[unsafe(no_mangle)]
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

/// Return the JSON signature of an `@export` by name (item 7 introspection).
///
/// The returned pointer is a null-terminated JSON object — the serialized
/// codegen `ExportInfo`: `{"symbol_name", "raw_name", "params": [{"name",
/// "ty"}...], "return_type"}` with tensor types carrying stringified shape
/// dims (symbolic dims stay named, e.g. `"B"`), dtype, and device. Hosts use
/// it to size `nsl_model_call_into` buffers exactly and to learn output
/// arity before calling.
///
/// The pointer aliases static data inside the model's dlopen'd artifact:
/// valid until `nsl_model_destroy`, never freed by the caller.
///
/// Returns NULL and sets the thread-local error when: the model or name is
/// null, the model has no export registry, the artifact predates the
/// signature table (rebuild), or the name is not an export.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_model_get_export_signature(model_ptr: i64, name_ptr: i64) -> i64 {
    if model_ptr == 0 || name_ptr == 0 {
        set_error("nsl_model_get_export_signature: null model or name pointer\0".to_string());
        return 0;
    }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    let registry = match &model.exports {
        Some(r) => r,
        None => {
            set_error(
                "nsl_model_get_export_signature: model created without export \
                 registry; use nsl_model_create_with_lib(weights, lib)\0"
                    .to_string(),
            );
            return 0;
        }
    };
    let name = unsafe { CStr::from_ptr(name_ptr as *const c_char) }
        .to_string_lossy()
        .into_owned();
    match registry.lookup_signature(&name) {
        Ok(ptr) => ptr,
        Err(why) => {
            set_error(format!("nsl_model_get_export_signature: {why}\0"));
            0
        }
    }
}

/// Destroy a model instance and free its resources (including weight tensors).
#[unsafe(no_mangle)]
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
#[unsafe(no_mangle)]
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
/// `nsl_model_call_dlpack(model, "forward", ...)`, bridging the historical
/// `num_outputs_ptr` in/out ABI to the registry dispatch's fixed
/// `num_outputs`.
///
/// `*num_outputs_ptr` on entry is an EXACT arity, not a slot capacity: the
/// dispatch wrapper refuses any count that is not the export's own arity
/// (1 in v1), so declaring "I have room for 8" fails rather than filling one
/// slot. `0`/unset reads as 1, the single-output forward case. On return it
/// carries the number of slots written: the requested count on success, 0 on
/// refusal.
///
/// Returns 0 on success, -1 on error. On success each output slot holds an
/// ownership-transferring `DLManagedTensor*` — see [`nsl_model_call_dlpack`]
/// and [`nsl_model_call_alloc`] for the deleter contract.
#[unsafe(no_mangle)]
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
///
/// Derived from the workspace version at compile time. A hardcoded literal
/// here returned "NSL 0.2.0" for seven releases (item 19, 2026-08-24): the
/// string reached Python users via `NslModel.version` but was outside every
/// version gate — the only test on it asserted `starts_with("NSL")`.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_model_get_version() -> i64 {
    static VERSION: &[u8] = concat!("NSL ", env!("CARGO_PKG_VERSION"), "\0").as_bytes();
    VERSION.as_ptr() as i64
}

/// Get the number of weight tensors in the model.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_model_num_weights(model_ptr: i64) -> i64 {
    if model_ptr == 0 { return 0; }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    model.weight_ptrs.len() as i64
}

/// Get the number of weight tensors — canonical name used by @export model-method wrappers.
/// Alias for `nsl_model_num_weights`.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_model_get_num_weights(model_ptr: i64) -> i64 {
    nsl_model_num_weights(model_ptr)
}

/// Return a pointer to the contiguous array of weight tensor pointers (`*const i64`).
///
/// The returned pointer is valid for the lifetime of the model; callers must not
/// free it.  Returns 0 if `model_ptr` is null or the model has no weights.
/// Used by `@export` model-method wrappers to thread weight pointers into the
/// compiled impl function.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_model_get_weight_ptrs(model_ptr: i64) -> i64 {
    if model_ptr == 0 { return 0; }
    let model = unsafe { &*(model_ptr as *const NslModel) };
    if model.weight_ptrs.is_empty() {
        return 0;
    }
    model.weight_ptrs.as_ptr() as i64
}

/// Get a weight tensor by name. Returns NslTensor pointer or 0 if not found.
#[unsafe(no_mangle)]
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

        // Convert to f32 (the standard NSL runtime format for loaded weights).
        // Unsupported dtypes hard-error naming the tensor — the Err propagates
        // to nsl_model_create, which does set_error + returns 0. Before bailing,
        // free the tensors already allocated this call: they are not yet owned
        // by any NslModel, so nothing else would ever reclaim them.
        let f32_data = match convert_weight_to_f32(view.dtype(), raw_data, len as usize) {
            Ok(d) => d,
            Err(e) => {
                for &ptr in &weight_ptrs {
                    if ptr != 0 {
                        crate::tensor::nsl_tensor_free(ptr);
                    }
                }
                return Err(format!("tensor '{name}' in '{path}': {e}"));
            }
        };

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
///
/// Refuse-loudly: an unsupported dtype returns `Err` naming the dtype and the
/// supported list instead of silently producing a zero-filled tensor (which
/// corrupted model loads — the caller mapped the zeros into model weights).
#[cfg(feature = "interop")]
fn convert_weight_to_f32(
    dtype: safetensors::Dtype,
    data: &[u8],
    len: usize,
) -> Result<Vec<f32>, String> {
    match dtype {
        safetensors::Dtype::F32 => Ok(data
            .chunks_exact(std::mem::size_of::<f32>())
            .take(len)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().expect("f32 chunks are 4 bytes");
                f32::from_le_bytes(bytes)
            })
            .collect()),
        safetensors::Dtype::F64 => Ok(data
            .chunks_exact(std::mem::size_of::<f64>())
            .take(len)
            .map(|chunk| {
                let bytes: [u8; 8] = chunk.try_into().expect("f64 chunks are 8 bytes");
                f64::from_le_bytes(bytes) as f32
            })
            .collect()),
        safetensors::Dtype::F16 => Ok(data
            .chunks_exact(std::mem::size_of::<u16>())
            .take(len)
            .map(|chunk| {
                let bytes: [u8; 2] = chunk.try_into().expect("f16 chunks are 2 bytes");
                half::f16::from_bits(u16::from_le_bytes(bytes)).to_f32()
            })
            .collect()),
        safetensors::Dtype::BF16 => Ok(data
            .chunks_exact(std::mem::size_of::<u16>())
            .take(len)
            .map(|chunk| {
                let bytes: [u8; 2] = chunk.try_into().expect("bf16 chunks are 2 bytes");
                half::bf16::from_bits(u16::from_le_bytes(bytes)).to_f32()
            })
            .collect()),
        // Integer weights convert to f32 (same semantics as the NSL builtin
        // safetensors loader — keeps the two loaders' supported sets in sync).
        safetensors::Dtype::I32 => Ok(data
            .chunks_exact(std::mem::size_of::<i32>())
            .take(len)
            .map(|chunk| {
                let bytes: [u8; 4] = chunk.try_into().expect("i32 chunks are 4 bytes");
                i32::from_le_bytes(bytes) as f32
            })
            .collect()),
        safetensors::Dtype::I64 => Ok(data
            .chunks_exact(std::mem::size_of::<i64>())
            .take(len)
            .map(|chunk| {
                let bytes: [u8; 8] = chunk.try_into().expect("i64 chunks are 8 bytes");
                i64::from_le_bytes(bytes) as f32
            })
            .collect()),
        other => Err(format!(
            "unsupported dtype {other:?}; supported dtypes: F32, F64, F16, BF16, I32, I64"
        )),
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
#[unsafe(no_mangle)]
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
///
/// **Item-7 capture point.** The typed wrapper calls this with the IMPL
/// RESULT tensor pointer — the only place the runtime ever sees it (the desc
/// carries data/shape pointers, not the tensor struct). When a dispatch
/// ownership mode is armed ([`nsl_model_call_into`] / [`nsl_model_call_alloc`]),
/// the pointer is recorded so the entry point can free or transfer the result
/// after dispatch returns. Unarmed (legacy `nsl_model_call`, direct typed-
/// symbol callers, grad-context forwards), behavior is unchanged.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_to_desc_ffi(tensor_ptr: i64, desc_ptr: i64) {
    if tensor_ptr == 0 || desc_ptr == 0 {
        return;
    }
    DISPATCH_MODE.with(|m| {
        if let Some(mode) = m.borrow_mut().as_mut() {
            mode.captured_mut().push(tensor_ptr);
        }
    });
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
#[unsafe(no_mangle)]
pub extern "C" fn nsl_dispatch_apply_result(src_desc_ptr: i64, dst_desc_ptr: i64) -> i64 {
    if src_desc_ptr == 0 || dst_desc_ptr == 0 {
        set_error("nsl_dispatch_apply_result: null desc pointer\0".to_string());
        return -1;
    }
    let src = unsafe { &*(src_desc_ptr as *const NslTensorDesc) };
    let dst = unsafe { &mut *(dst_desc_ptr as *mut NslTensorDesc) };

    // Bytes per element, indexed by canonical dtype tag (see
    // NslTensorDesc::dtype — same space as crate::tensor::DTYPE_*).
    let elem_bytes: usize = match src.dtype {
        0 => 8, // f64
        1 => 4, // f32
        2 => 2, // f16
        3 => 2, // bf16
        4 => 1, // int8
        5 => 1, // fp8e4m3
        6 => 1, // fp8e5m2
        7 => 2, // u16 token
        8 => 2, // u16 segment
        9 => 4, // int32
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

    // ── Item-7 ownership modes ────────────────────────────────────────────
    //
    // `Alloc` (`nsl_model_call_alloc`): no caller buffer exists — the entry
    // point owns the captured impl tensor and wraps it into a DLManagedTensor
    // after dispatch. Nothing to copy here; the scratch dst is never read.
    //
    // `Into` (`nsl_model_call_into`): capacity-checked deep-copy. Refuse an
    // undersized buffer REPORTING THE REQUIRED SIZE (the legacy path below
    // memcpys unchecked into whatever the caller guessed — a silent heap
    // overrun for any output larger than the guess), and deep-copy
    // shape/strides into caller-owned arrays instead of mirroring pointers
    // into the impl tensor (which the entry point frees after dispatch).
    enum ApplyPlan {
        Legacy,
        AllocSkip,
        Into { slot: usize, capacity: u64 },
    }
    let plan = DISPATCH_MODE.with(|m| {
        let mut mode = m.borrow_mut();
        match mode.as_mut() {
            None => ApplyPlan::Legacy,
            Some(DispatchMode::Alloc { .. }) => ApplyPlan::AllocSkip,
            Some(DispatchMode::Into { capacities, cursor, .. }) => {
                let slot = *cursor;
                *cursor += 1;
                let capacity = capacities.get(slot).copied().unwrap_or(0);
                ApplyPlan::Into { slot, capacity }
            }
        }
    });

    match plan {
        ApplyPlan::AllocSkip => return 0,
        ApplyPlan::Into { slot, capacity } => {
            // The deep copy below moves `byte_count` CONTIGUOUS bytes, a size
            // derived from the shape product. A non-contiguous result breaks
            // that equivalence in both directions: a stride-0 broadcast view
            // (`expand`) has a backing buffer SMALLER than its shape product,
            // so the memcpy would read past the allocation (out-of-bounds heap
            // read, segfault at scale); a permuted/sliced view would copy the
            // wrong elements. Refuse rather than copy wrongly — the ownership
            // path (nsl_model_call_alloc) carries strides natively and is the
            // supported way to return a view.
            if src.ndim > 0 && !src.strides.is_null() {
                if src.shape.is_null() {
                    set_error(format!(
                        "nsl_dispatch_apply_result: output {slot} has strides but \
                         a null shape\0"
                    ));
                    return -1;
                }
                let n = src.ndim as usize;
                let mut expect: i64 = 1;
                let mut contiguous = true;
                for i in (0..n).rev() {
                    let stride = unsafe { std::ptr::read_unaligned(src.strides.add(i)) };
                    let dim = unsafe { std::ptr::read_unaligned(src.shape.add(i)) };
                    // A dim of 1 pins nothing: any stride addresses the same
                    // single element, so do not treat it as non-contiguous.
                    if dim != 1 && stride != expect {
                        contiguous = false;
                        break;
                    }
                    expect = expect.saturating_mul(dim.max(0));
                }
                if !contiguous {
                    set_error(format!(
                        "nsl_dispatch_apply_result: output {slot} is a \
                         non-contiguous view (broadcast/permuted/sliced strides). \
                         nsl_model_call_into deep-copies a contiguous block and \
                         cannot express this layout — use nsl_model_call_alloc \
                         (DLPack carries strides), or return a materialized \
                         tensor\0"
                    ));
                    return -1;
                }
            }
            // A device pointer cannot be host-memcpy'd. The Alloc path refuses
            // GPU residency explicitly; this path must too, BEFORE the copy —
            // src.device_type is otherwise only mirrored onto the caller's desc
            // after the fact, i.e. after the fault.
            if src.device_type != 0 && byte_count > 0 {
                set_error(format!(
                    "nsl_dispatch_apply_result: output {slot} is GPU-resident \
                     (device_type={}); nsl_model_call_into copies on the host \
                     and cannot read device memory — move the result to CPU in \
                     the export\0",
                    src.device_type
                ));
                return -1;
            }
            if byte_count as u64 > capacity {
                set_error(format!(
                    "nsl_dispatch_apply_result: output {slot} requires {byte_count} \
                     bytes but the caller buffer capacity is {capacity} bytes — \
                     reallocate and retry with at least {byte_count} bytes\0"
                ));
                return -1;
            }
            let src_ndim = src.ndim.max(0);
            if src_ndim > 0 {
                if dst.shape.is_null() {
                    set_error(format!(
                        "nsl_dispatch_apply_result: output {slot} has rank {src_ndim} \
                         but the caller shape array is null (the call_into contract \
                         deep-copies dims into caller-owned arrays)\0"
                    ));
                    return -1;
                }
                if src_ndim > dst.ndim {
                    set_error(format!(
                        "nsl_dispatch_apply_result: output {slot} has rank {src_ndim} \
                         but the caller shape/strides arrays declare only {} slot(s) \
                         (set dst.ndim to the allocated slot count on entry)\0",
                        dst.ndim
                    ));
                    return -1;
                }
            }
            if byte_count > 0 {
                if dst.data.is_null() || src.data.is_null() {
                    set_error(
                        "nsl_dispatch_apply_result: null data pointer (caller buffer \
                         or impl result)\0"
                            .to_string(),
                    );
                    return -1;
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src.data as *const u8,
                        dst.data as *mut u8,
                        byte_count,
                    );
                }
            }
            // Deep-copy dims/strides into the caller's arrays. If the impl
            // result has no strides (contiguous), synthesize row-major
            // element strides so the caller always sees a complete desc.
            for i in 0..src_ndim as usize {
                unsafe {
                    let d = std::ptr::read_unaligned(src.shape.add(i));
                    *dst.shape.add(i) = d;
                }
            }
            if src_ndim > 0 && !dst.strides.is_null() {
                if !src.strides.is_null() {
                    for i in 0..src_ndim as usize {
                        unsafe {
                            let s = std::ptr::read_unaligned(src.strides.add(i));
                            *dst.strides.add(i) = s;
                        }
                    }
                } else {
                    let mut stride: i64 = 1;
                    for i in (0..src_ndim as usize).rev() {
                        unsafe {
                            *dst.strides.add(i) = stride;
                            stride *= std::ptr::read_unaligned(src.shape.add(i)).max(1);
                        }
                    }
                }
            }
            dst.ndim = src_ndim;
            dst.dtype = src.dtype;
            dst.device_type = src.device_type;
            dst.device_id = src.device_id;
            dst.tape_id = src.tape_id;
            return 0;
        }
        ApplyPlan::Legacy => {}
    }

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
    //
    // These mirrored pointers alias the impl result tensor, which the LEGACY
    // path leaks by contract — callers (nslpy `read_f32_output_desc`, direct
    // ctypes hosts) read them after the call returns. Do not "fix" the leak
    // here: freeing the impl tensor under this mode converts the leak into a
    // use-after-free. The leak-free paths are `nsl_model_call_into` (deep
    // copy above) and `nsl_model_call_alloc` (ownership transfer).
    dst.shape = src.shape;
    dst.strides = src.strides;
    dst.ndim = src.ndim;
    dst.dtype = src.dtype;
    dst.device_type = src.device_type;
    dst.device_id = src.device_id;
    dst.tape_id = src.tape_id;
    0
}

/// Scalar sibling of [`nsl_dispatch_apply_result`], called by the dispatch
/// wrapper for `Scalar(f64)`-returning exports (item 7 scalar fix).
///
/// The typed wrapper stores the raw scalar bits at offset 0 of the scratch
/// desc (the `data` field's storage). The PREVIOUS dispatch flow handed that
/// desc to `nsl_dispatch_apply_result`, which interpreted the scalar VALUE as
/// the source ADDRESS of a memcpy — a wild read reachable from any
/// scalar-returning export via `nsl_model_call`. This helper reads the bits
/// AT the scratch desc instead of THROUGH them and stores the 8-byte f64 into
/// the caller's buffer, setting `ndim = 0` and dtype tag 0 (f64).
///
/// Ownership modes: `Into` capacity-checks (>= 8 bytes) against the armed
/// slot; `Alloc` refuses — a scalar has no DLPack representation (use
/// `nsl_model_call_into` or `nsl_model_call`).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_dispatch_apply_scalar_result(src_desc_ptr: i64, dst_desc_ptr: i64) -> i64 {
    if src_desc_ptr == 0 || dst_desc_ptr == 0 {
        set_error("nsl_dispatch_apply_scalar_result: null desc pointer\0".to_string());
        return -1;
    }
    // The scalar bits occupy the first 8 bytes of the scratch desc.
    let bits = unsafe { *(src_desc_ptr as *const u64) };
    let dst = unsafe { &mut *(dst_desc_ptr as *mut NslTensorDesc) };

    enum ScalarPlan {
        Write,
        AllocRefuse,
        IntoChecked { slot: usize, capacity: u64 },
    }
    let plan = DISPATCH_MODE.with(|m| {
        let mut mode = m.borrow_mut();
        match mode.as_mut() {
            None => ScalarPlan::Write,
            Some(DispatchMode::Alloc { .. }) => ScalarPlan::AllocRefuse,
            Some(DispatchMode::Into { capacities, cursor, .. }) => {
                let slot = *cursor;
                *cursor += 1;
                let capacity = capacities.get(slot).copied().unwrap_or(0);
                ScalarPlan::IntoChecked { slot, capacity }
            }
        }
    });
    match plan {
        ScalarPlan::AllocRefuse => {
            set_error(
                "nsl_dispatch_apply_scalar_result: scalar-returning exports have \
                 no DLPack representation; use nsl_model_call_into or \
                 nsl_model_call with an 8-byte output buffer\0"
                    .to_string(),
            );
            return -1;
        }
        ScalarPlan::IntoChecked { slot, capacity } => {
            if capacity < 8 {
                set_error(format!(
                    "nsl_dispatch_apply_scalar_result: output {slot} requires 8 \
                     bytes (f64 scalar) but the caller buffer capacity is \
                     {capacity} bytes\0"
                ));
                return -1;
            }
        }
        ScalarPlan::Write => {}
    }
    if dst.data.is_null() {
        set_error(
            "nsl_dispatch_apply_scalar_result: null caller data buffer (a scalar \
             return needs an 8-byte output buffer)\0"
                .to_string(),
        );
        return -1;
    }
    unsafe { *(dst.data as *mut u64) = bits };
    dst.ndim = 0;
    dst.dtype = 0; // f64
    dst.device_type = 0;
    dst.device_id = 0;
    dst.tape_id = 0;
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
    // Returns 0 (not abort) on an out-of-range tag — see `capi_dtype_to_nsl`.
    // Every caller must treat 0 as a failure; the codegen-emitted `@export`
    // wrappers do so via the null guard `c_wrapper::emit_null_tensor_guard`
    // emits after each `nsl_desc_to_tensor` call.
    let nsl_dtype = match capi_dtype_to_nsl(desc.dtype) {
        Some(d) => d,
        None => {
            set_error(format!(
                "desc_to_nsl_tensor: unrecognized dtype tag {} (valid: {})\0",
                desc.dtype, CAPI_DTYPE_TAGS
            ));
            return 0;
        }
    };

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
    // Null guard, matching the extern-C sibling `nsl_tensor_to_desc_ffi`.
    // `NslTensor::from_ptr` refuses a null handle in every profile now, but
    // it refuses it by aborting; this entry point answers a null with a
    // null instead, so the test below still needs the guard to be here.
    if tensor_ptr == 0 {
        return;
    }
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
        // Exact equality with the workspace version. The previous
        // `starts_with("NSL")` was structurally incapable of catching the
        // numeric drift this function shipped with for seven releases.
        assert_eq!(version_str, format!("NSL {}", env!("CARGO_PKG_VERSION")));
    }

    #[test]
    fn test_dtype_mapping() {
        // P4 item 16: the C API tag space IS the canonical tag space — both
        // chokepoints are validating identity maps.
        for capi_d in 0..=9 {
            let nsl_d = capi_dtype_to_nsl(capi_d)
                .unwrap_or_else(|| panic!("tag {capi_d} must be accepted"));
            assert_eq!(nsl_d as i32, capi_d, "C API tag must equal canonical tag");
            let back = nsl_dtype_to_capi(nsl_d);
            assert_eq!(back, capi_d, "Roundtrip failed for C API dtype {capi_d}");
        }
        // Out-of-range tags are REFUSED, not aborted on. `-1` and `10` bracket
        // the accepted window; `42` is the tag the interop gate drives.
        for bad in [-1_i32, 10, 42, i32::MAX] {
            assert_eq!(
                capi_dtype_to_nsl(bad),
                None,
                "tag {bad} is outside 0..=9 and must be refused"
            );
        }
    }

    /// Child of `nsl_tensor_to_desc_null_ptr_does_not_deref`.
    ///
    /// `nsl_tensor_to_desc` is `pub(crate)`, so this cannot live in
    /// `tests/` alongside the rest of the M62 dtype gates. It re-execs
    /// through the lib test binary instead. The scenario runs in a CHILD
    /// because the failure mode is fatal either way: without the guard the
    /// null reaches `NslTensor::from_ptr`, which aborts (it used to read
    /// address 0 and SIGSEGV), and no `#[should_panic]` can observe
    /// either.
    #[test]
    #[ignore = "child process — driven by nsl_tensor_to_desc_null_ptr_does_not_deref"]
    fn zz_tensor_to_desc_null_child() {
        if std::env::var("NSL_TENSOR_TO_DESC_NULL_CHILD").is_err() {
            return;
        }
        // Pre-poison every field so "untouched" is observable rather than
        // indistinguishable from a zeroed default.
        let mut sentinel_shape: Vec<i64> = vec![7];
        let mut desc = NslTensorDesc {
            data: 0x5A5A_5A5A_usize as *mut c_void,
            shape: sentinel_shape.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            ndim: 99,
            dtype: 7,
            device_type: 5,
            device_id: 6,
            tape_id: 1234,
        };
        nsl_tensor_to_desc(0, &mut desc);
        assert_eq!(desc.ndim, 99, "a null tensor must leave the desc untouched");
        assert_eq!(desc.dtype, 7, "a null tensor must leave the desc untouched");
        assert_eq!(desc.tape_id, 1234, "a null tensor must leave the desc untouched");
        assert_eq!(desc.data as usize, 0x5A5A_5A5A);
        println!("CHILD-SURVIVED-NULL-DESC");
    }

    #[test]
    fn nsl_tensor_to_desc_null_ptr_does_not_deref() {
        let exe = std::env::current_exe().expect("test binary path");
        let out = std::process::Command::new(exe)
            .args([
                "c_api::tests::zz_tensor_to_desc_null_child",
                "--exact",
                "--nocapture",
                "--test-threads=1",
                "--include-ignored",
            ])
            .env("NSL_TENSOR_TO_DESC_NULL_CHILD", "1")
            .env("RUST_BACKTRACE", "0")
            .output()
            .expect("re-exec the lib test binary");
        let stdout = String::from_utf8_lossy(&out.stdout);
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            out.status.success(),
            "child exited {:?} — nsl_tensor_to_desc dereferenced a null tensor \
             pointer (139 = SIGSEGV).\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
            out.status
        );
        assert!(
            stdout.contains("CHILD-SURVIVED-NULL-DESC"),
            "child exited 0 but never reached the end of the scenario — the \
             filter may have selected no test.\n--- stdout ---\n{stdout}\n\
             --- stderr ---\n{stderr}"
        );
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
            dtype: 1, // canonical f32
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
        assert_eq!(out_desc.dtype, 1); // canonical f32, unchanged
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

    // -----------------------------------------------------------------------
    // Refuse-loudly: unsupported safetensors dtypes must fail the load naming
    // the tensor — never silently zero-fill model weights. Interop-gated
    // because `convert_weight_to_f32` / the real `load_safetensors_weights`
    // only exist with `--features interop`.
    // -----------------------------------------------------------------------

    /// Write a temp .safetensors with one supported F32 tensor and one
    /// unsupported U8 tensor; return the temp file handle (unique path).
    #[cfg(feature = "interop")]
    fn write_temp_safetensors_with_u8_tensor() -> tempfile::NamedTempFile {
        use std::io::Write;

        let good_bytes: Vec<u8> = [1.0f32, 2.0].iter().flat_map(|v| v.to_le_bytes()).collect();
        let good = safetensors::tensor::TensorView::new(
            safetensors::Dtype::F32,
            vec![2],
            &good_bytes,
        )
        .unwrap();
        let bad_bytes = [1u8, 2, 3, 4];
        let bad = safetensors::tensor::TensorView::new(
            safetensors::Dtype::U8,
            vec![4],
            &bad_bytes,
        )
        .unwrap();

        let mut map = HashMap::new();
        map.insert("layer.weight".to_string(), good);
        map.insert("quant.blob".to_string(), bad);
        let serialized = safetensors::tensor::serialize(map, None).unwrap();

        let mut tmp = tempfile::NamedTempFile::new().unwrap();
        tmp.write_all(&serialized).unwrap();
        tmp.flush().unwrap();
        tmp
    }

    #[cfg(feature = "interop")]
    #[test]
    fn convert_weight_to_f32_unsupported_dtype_errors() {
        let err = convert_weight_to_f32(safetensors::Dtype::U8, &[1u8, 2, 3], 3).unwrap_err();
        assert!(err.contains("U8"), "error must name the dtype: {err}");
        for supported in ["F32", "F64", "F16", "BF16"] {
            assert!(
                err.contains(supported),
                "error must list supported dtype {supported}: {err}"
            );
        }
    }

    #[cfg(feature = "interop")]
    #[test]
    fn load_safetensors_weights_unsupported_dtype_names_tensor() {
        let tmp = write_temp_safetensors_with_u8_tensor();
        let path = tmp.path().to_str().unwrap();

        let err = load_safetensors_weights(path).unwrap_err();
        assert!(err.contains("quant.blob"), "error must name the tensor: {err}");
        assert!(err.contains("U8"), "error must name the dtype: {err}");
        for supported in ["F32", "F64", "F16", "BF16"] {
            assert!(
                err.contains(supported),
                "error must list supported dtype {supported}: {err}"
            );
        }
    }

    #[cfg(feature = "interop")]
    #[test]
    fn nsl_model_create_unsupported_dtype_sets_error_and_returns_zero() {
        let tmp = write_temp_safetensors_with_u8_tensor();
        let path_c = std::ffi::CString::new(tmp.path().to_str().unwrap()).unwrap();

        nsl_clear_error();
        let model_ptr = nsl_model_create(path_c.as_ptr() as i64);
        assert_eq!(model_ptr, 0, "model create must fail for unsupported dtype");

        let err_ptr = nsl_get_last_error();
        assert_ne!(err_ptr, 0);
        let err = unsafe {
            CStr::from_ptr(err_ptr as *const c_char)
                .to_string_lossy()
                .into_owned()
        };
        assert!(err.contains("quant.blob"), "error must name the tensor: {err}");
        assert!(err.contains("U8"), "error must name the dtype: {err}");
        nsl_clear_error();
    }

    /// I32/I64 checkpoints load via the C-API path (parity with the NSL builtin
    /// safetensors loader, which already supported them).
    #[cfg(feature = "interop")]
    #[test]
    fn convert_weight_to_f32_supports_i32_i64() {
        let i32_bytes = (-3i32).to_le_bytes();
        let i32_out = convert_weight_to_f32(safetensors::Dtype::I32, &i32_bytes, 1).unwrap();
        assert_eq!(i32_out, vec![-3.0f32]);

        let i64_bytes = 7i64.to_le_bytes();
        let i64_out = convert_weight_to_f32(safetensors::Dtype::I64, &i64_bytes, 1).unwrap();
        assert_eq!(i64_out, vec![7.0f32]);

        // The refusal list now advertises I32/I64 too.
        let err = convert_weight_to_f32(safetensors::Dtype::U8, &[0u8], 1).unwrap_err();
        for supported in ["F32", "F64", "F16", "BF16", "I32", "I64"] {
            assert!(err.contains(supported), "error must list {supported}: {err}");
        }
    }

    /// The mid-loop refusal frees the tensors allocated for earlier entries
    /// (they are not yet owned by any NslModel). Calling twice would abort on a
    /// double-free if the cleanup were wrong, so a clean second failure pins the
    /// no-leak / no-double-free error path.
    #[cfg(feature = "interop")]
    #[test]
    fn load_safetensors_weights_error_path_is_clean_and_repeatable() {
        let tmp = write_temp_safetensors_with_u8_tensor();
        let path = tmp.path().to_str().unwrap();

        for _ in 0..2 {
            let err = load_safetensors_weights(path).unwrap_err();
            assert!(err.contains("quant.blob"), "error must name the tensor: {err}");
        }
    }
}
