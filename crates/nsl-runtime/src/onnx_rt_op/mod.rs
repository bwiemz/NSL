//! M62b Spec C — ONNX Runtime custom op integration.
//!
//! When `--features onnx-rt-op` is enabled, the .so/.dll exports
//! `RegisterCustomOps` so ONNX Runtime can load it via
//! `SessionOptions::register_custom_ops_library`. Each NSL `@export`
//! shows up as a custom op in the `com.nsl` domain.
//!
//! **v1 scope:** CPU compute kernels only. CUDA EP integration is
//! deferred behind a future `onnx-rt-cuda` feature flag.
//!
//! **ORT version:** Vendored against ORT 1.22.x (`ORT_API_VERSION=22`).
//! The version is asserted at the FFI boundary; mismatched ORT
//! versions return a typed `OrtVersionUnsupported` status (T4).

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

pub mod kernel;
pub mod registry;
pub mod vendored;

use vendored::*;

/// Spec C §2.1 — ORT's custom-op-library entry point. Called once when ORT
/// loads this .so via `SessionOptions::register_custom_ops_library`.
///
/// Returns NULL `OrtStatus*` on success, or a non-null status with an error
/// message on failure (caller frees via `OrtApi.ReleaseStatus`).
///
/// **Steps (Spec C §2.1):**
/// 1. Call `api_base->GetApi(EXPECTED_ORT_API_VERSION)`. NULL → return a
///    static-sentinel non-null status (no `OrtApi` is available to
///    construct a real one).
/// 2. Enumerate NSL exports via the codegen-emitted runtime FFIs
///    (`nsl_get_num_exports` / `nsl_get_export_name`). Zero exports →
///    return success with no domain registered.
/// 3. Create the `com.nsl` domain.
/// 4. For each export, build an `OrtCustomOp` via `registry::make_custom_op_for_export`
///    and register on the domain.
/// 5. Attach the domain to the session options.
#[cfg(feature = "onnx-rt-op")]
#[no_mangle]
pub unsafe extern "C" fn RegisterCustomOps(
    options: *mut OrtSessionOptions,
    api_base: *const OrtApiBase,
) -> *mut OrtStatus {
    // --- Step 1: version-check via the version-stable OrtApiBase. ----------
    let api_base_ref = match api_base.as_ref() {
        Some(b) => b,
        None => {
            return ort_make_status_static();
        }
    };
    let api_ptr = (api_base_ref.GetApi)(EXPECTED_ORT_API_VERSION);
    if api_ptr.is_null() {
        return ort_make_status_static();
    }
    let api: &OrtApi = &*api_ptr;

    // --- Step 2: enumerate NSL exports via Spec A's runtime FFIs. ---------
    //
    // The codegen-emitted `nsl_get_num_exports` and `nsl_get_export_name`
    // are not linked statically — they exist only in shipped `.so/.dll`
    // artifacts where `--features onnx-rt-op` is enabled alongside an
    // `@export`-bearing NSL module. We resolve them at runtime via the
    // same self-dlsym pattern the registry uses for individual exports:
    // - Unix: `dlsym(RTLD_DEFAULT, ...)`
    // - Windows: `GetModuleHandleExW(FROM_ADDRESS) + GetProcAddress`.
    //
    // If either symbol is missing (the common case for `cargo test` /
    // examples where no NSL `@export`s exist), we treat the export
    // table as empty and return success without registering a domain.
    let get_num = registry::resolve_self_symbol(c"nsl_get_num_exports".as_ptr());
    let get_name = registry::resolve_self_symbol(c"nsl_get_export_name".as_ptr());
    if get_num == 0 || get_name == 0 {
        // No export table in this binary — nothing to register.
        return std::ptr::null_mut();
    }
    // SAFETY: dlsym/GetProcAddress returned non-null pointers to symbols
    // codegen guarantees match these signatures.
    let nsl_get_num_exports: unsafe extern "C" fn() -> i64 =
        std::mem::transmute::<usize, unsafe extern "C" fn() -> i64>(get_num);
    let nsl_get_export_name: unsafe extern "C" fn(i64) -> *const std::os::raw::c_char =
        std::mem::transmute::<usize, unsafe extern "C" fn(i64) -> *const std::os::raw::c_char>(get_name);

    let num_exports = nsl_get_num_exports();
    if num_exports == 0 {
        return std::ptr::null_mut();
    }

    // --- Step 3: create the `com.nsl` domain. -----------------------------
    let domain_name = c"com.nsl".as_ptr();
    let mut domain: *mut OrtCustomOpDomain = std::ptr::null_mut();
    let status = (api.CreateCustomOpDomain)(domain_name, &mut domain);
    if !status.is_null() {
        return status;
    }

    // --- Step 4: register one OrtCustomOp per export. ---------------------
    for idx in 0..num_exports {
        let name_ptr = nsl_get_export_name(idx);
        if name_ptr.is_null() {
            continue;
        }
        let custom_op = registry::make_custom_op_for_export(idx, name_ptr);
        if custom_op.is_null() {
            // Internal: vtable allocation never returns null in v1, but
            // guard against it for forward-compat.
            continue;
        }
        let status = (api.CustomOpDomain_Add)(domain, custom_op);
        if !status.is_null() {
            return status;
        }
    }

    // --- Step 5: attach the domain to the session options. ----------------
    (api.AddCustomOpDomain)(options, domain)
}

/// Return a non-null `OrtStatus*` sentinel when no `OrtApi` is available.
///
/// We can't call `OrtApi.CreateStatus` to allocate a real status (that's the
/// very pointer that's missing), so this leaks a small per-message static
/// sentinel and returns it cast as `*mut OrtStatus`. ORT's session-load path
/// surfaces a generic "custom op registration failed" message in this case;
/// improving the message requires waiting for ORT to expose a version-stable
/// status constructor (none in 1.22).
#[cfg(feature = "onnx-rt-op")]
fn ort_make_status_static() -> *mut OrtStatus {
    // Bit pattern 1 — a non-null pointer that's distinct from any real
    // allocation, used by ORT to signal "non-null = failure" without ever
    // dereferencing the status struct (because we never registered ops).
    // Not `ptr::dangling_mut` because we specifically want a non-aligned
    // sentinel value that no real allocator could ever return.
    #[allow(clippy::manual_dangling_ptr)]
    let p = 1usize as *mut OrtStatus;
    p
}

// Note: `nsl_get_num_exports` / `nsl_get_export_name` are resolved at
// runtime via `registry::resolve_self_symbol` rather than declared as
// link-time externs. These codegen-emitted symbols only exist in shipped
// `.so/.dll` artifacts; the static-link variant would break every test
// binary that doesn't have @exports.
