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
//! **ORT version:** Vendored against ORT 1.22.x (ORT_API_VERSION=22).
//! The version is asserted at the FFI boundary; mismatched ORT
//! versions return a typed `OrtVersionUnsupported` status.
//!
//! **Task 1 status:** This file currently provides only the skeleton —
//! `RegisterCustomOps` is a stub that returns `null` (success) and the
//! `vendored`/`kernel`/`registry` submodules are placeholders. Real
//! implementations land in Tasks 2 (vendored), 3 (kernel), and 4
//! (registry + full `RegisterCustomOps` body).

#![allow(non_snake_case)]
#![allow(non_camel_case_types)]
#![allow(dead_code)]

pub mod vendored;

pub mod kernel;

pub mod registry {
    //! Placeholder — populated in Task 4 with `make_custom_op_for_export`
    //! and the `OrtCustomOp` vtable population.

    use super::vendored::*;

    pub fn make_custom_op_for_export(
        _idx: i64,
        _name: *const std::os::raw::c_char,
    ) -> *const OrtCustomOp {
        std::ptr::null()
    }
}

use vendored::*;

/// Spec C §2.1 — ORT's custom-op-library entry point. Called once
/// when ORT loads this .so via `register_custom_ops_library`.
///
/// Returns NULL `OrtStatus*` on success, or a non-null status with an
/// error message on failure (caller frees via `OrtApi.ReleaseStatus`).
///
/// **Task 1 stub:** real implementation lands in Task 4. This stub
/// exists so Task 1's skeleton compiles cleanly; the test in Task 4
/// catches missing behavior (no-op registration is detected as a
/// missing custom op domain).
#[cfg(feature = "onnx-rt-op")]
#[no_mangle]
pub unsafe extern "C" fn RegisterCustomOps(
    _options: *mut OrtSessionOptions,
    _api_base: *const OrtApiBase,
) -> *mut OrtStatus {
    std::ptr::null_mut()
}

// Re-exports for the same-.so call into Spec A's enumeration FFIs.
// These resolve at link time to the codegen-emitted symbols (Plan A T2)
// that live in the SAME .so as `RegisterCustomOps`. Declared here so
// Task 4's full `RegisterCustomOps` body can call them without dlopen.
#[cfg(feature = "onnx-rt-op")]
extern "C" {
    fn nsl_get_num_exports() -> i64;
    fn nsl_get_export_name(idx: i64) -> *const std::os::raw::c_char;
}
