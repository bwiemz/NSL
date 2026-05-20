#![cfg(feature = "interop")]
//! nsl_model_call returns -1 and sets a thread-local error naming the
//! missing export plus listing available names when the requested
//! function isn't in the registry.

use std::ffi::{CStr, CString};

fn build_lib_with_one_export() -> (std::path::PathBuf, std::path::PathBuf) {
    use assert_cmd::prelude::*;
    use std::process::Command;
    let nsl = r#"
@export
fn alpha(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
    let tmp = std::env::temp_dir().join(format!("nsl_unkn_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = tmp.join("m.nsl");
    std::fs::write(&src, nsl).unwrap();
    let weights = tmp.join("w.safetensors");
    std::fs::write(&weights, b"\x02\x00\x00\x00\x00\x00\x00\x00{}").unwrap();
    let lib_ext = if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let lib = tmp.join(format!("libm.{lib_ext}"));
    let manifest_dir: std::path::PathBuf = env!("CARGO_MANIFEST_DIR").into();
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let stdlib = workspace_root.join("stdlib");
    let status = Command::cargo_bin("nsl")
        .unwrap()
        .env("NSL_STDLIB_PATH", &stdlib)
        .args([
            "build",
            "--shared-lib",
            src.to_str().unwrap(),
            "-o",
            lib.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success(), "nsl build --shared-lib failed");
    (lib, weights)
}

#[test]
fn unknown_name_returns_minus_one_and_sets_export_missing_error() {
    let (lib, weights) = build_lib_with_one_export();
    let w = CString::new(weights.to_str().unwrap()).unwrap();
    let l = CString::new(lib.to_str().unwrap()).unwrap();
    let model = nsl_runtime::c_api::nsl_model_create_with_lib(
        w.as_ptr() as i64,
        l.as_ptr() as i64,
    );
    assert_ne!(model, 0);

    let unknown = CString::new("does_not_exist").unwrap();
    let rc = nsl_runtime::c_api::nsl_model_call(model, unknown.as_ptr() as i64, 0, 0, 0, 0);
    assert_eq!(rc, -1, "expected -1 for unknown export name");
    let err_ptr = nsl_runtime::c_api::nsl_get_last_error() as *const std::os::raw::c_char;
    let err = unsafe { CStr::from_ptr(err_ptr) }.to_string_lossy();
    assert!(
        err.contains("does_not_exist"),
        "error should name the missing export, got: {err}"
    );
    assert!(
        err.contains("alpha"),
        "error should list available exports, got: {err}"
    );

    nsl_runtime::c_api::nsl_model_destroy(model);
}
