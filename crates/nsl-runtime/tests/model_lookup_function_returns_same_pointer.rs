#![cfg(feature = "interop")]
//! nsl_model_lookup_function returns the cached fn pointer for a known
//! @export (deterministic across calls) and 0 for unknown names.

use std::ffi::CString;

fn build_lib(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
    use assert_cmd::prelude::*;
    use std::process::Command;
    let nsl = r#"
@export
fn identity(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
    let tmp = std::env::temp_dir().join(format!("nsl_lookup_{}_{tag}", std::process::id()));
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
fn lookup_returns_non_null_for_existing_export() {
    let (lib, weights) = build_lib("nonnull");
    let w = CString::new(weights.to_str().unwrap()).unwrap();
    let l = CString::new(lib.to_str().unwrap()).unwrap();
    let model = nsl_runtime::c_api::nsl_model_create_with_lib(
        w.as_ptr() as i64,
        l.as_ptr() as i64,
    );
    let name = CString::new("identity").unwrap();
    let ptr1 = nsl_runtime::c_api::nsl_model_lookup_function(model, name.as_ptr() as i64);
    let ptr2 = nsl_runtime::c_api::nsl_model_lookup_function(model, name.as_ptr() as i64);
    assert_ne!(ptr1, 0);
    assert_eq!(ptr1, ptr2, "lookup must be deterministic (cache hit)");
    nsl_runtime::c_api::nsl_model_destroy(model);
}

#[test]
fn lookup_returns_zero_for_unknown_export() {
    let (lib, weights) = build_lib("zero");
    let w = CString::new(weights.to_str().unwrap()).unwrap();
    let l = CString::new(lib.to_str().unwrap()).unwrap();
    let model = nsl_runtime::c_api::nsl_model_create_with_lib(
        w.as_ptr() as i64,
        l.as_ptr() as i64,
    );
    let unk = CString::new("nope").unwrap();
    let ptr = nsl_runtime::c_api::nsl_model_lookup_function(model, unk.as_ptr() as i64);
    assert_eq!(ptr, 0);
    nsl_runtime::c_api::nsl_model_destroy(model);
}
