//! nsl_model_create_with_lib dlopens the model's own .so/.dll/.dylib,
//! enumerates exports, dlsyms each, and stashes the registry on the
//! NslModel. After return, the registry is read-only.
//!
//! Requires the `interop` feature to load safetensors weights.

#![cfg(feature = "interop")]

fn build_test_lib_with_weights(
    nsl_src: &str,
    weights_bytes: &[u8],
) -> (std::path::PathBuf, std::path::PathBuf) {
    use std::process::Command;
    let tmp = std::env::temp_dir().join(format!("nsl_create_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = tmp.join("model.nsl");
    std::fs::write(&src, nsl_src).unwrap();
    let weights = tmp.join("weights.safetensors");
    std::fs::write(&weights, weights_bytes).unwrap();
    let lib_ext = if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let out = tmp.join(format!("libmodel.{lib_ext}"));

    let manifest_dir: std::path::PathBuf = env!("CARGO_MANIFEST_DIR").into();
    let workspace_root = manifest_dir.parent().unwrap().parent().unwrap();
    let stdlib = workspace_root.join("stdlib");

    let status = Command::new(nsl_bin())
        .env("NSL_STDLIB_PATH", &stdlib)
        .args([
            "build",
            "--shared-lib",
            src.to_str().unwrap(),
            "-o",
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());
    (out, weights)
}

#[test]
fn create_populates_registry_or_returns_null() {
    let nsl = r#"
@export
fn identity(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
    // Minimal safetensors header — just an empty {} json blob to keep
    // weight loading happy. The identity model has no weights.
    let (lib_path, weights_path) =
        build_test_lib_with_weights(nsl, b"\x02\x00\x00\x00\x00\x00\x00\x00{}");

    let lib_cstr = std::ffi::CString::new(lib_path.to_str().unwrap()).unwrap();
    let weights_cstr = std::ffi::CString::new(weights_path.to_str().unwrap()).unwrap();

    let model = nsl_runtime::c_api::nsl_model_create_with_lib(
        weights_cstr.as_ptr() as i64,
        lib_cstr.as_ptr() as i64,
    );
    if model == 0 {
        // Surface the thread-local error in case create fails, so failures
        // are diagnosable rather than just `0 != 0`.
        let err_ptr = nsl_runtime::c_api::nsl_get_last_error();
        let err = unsafe { std::ffi::CStr::from_ptr(err_ptr as *const std::os::raw::c_char) }
            .to_string_lossy()
            .into_owned();
        panic!("nsl_model_create_with_lib returned 0; last_error={err}");
    }

    let count = nsl_runtime::c_api::nsl_model_export_count(model);
    assert_eq!(count, 1);

    nsl_runtime::c_api::nsl_model_destroy(model);
}
/// Path to the `nsl` binary built by `cargo test --workspace`.
///
/// `nsl` lives in the sibling `nsl-cli` crate, so Cargo does not set
/// `CARGO_BIN_EXE_nsl` for this crate's integration tests, and assert_cmd 2.2+
/// no longer falls back to the target directory. Resolve it next to the running
/// test executable instead: `target/<profile>/deps/<test>` -> `target/<profile>/nsl`.
fn nsl_bin() -> std::path::PathBuf {
    let mut dir = std::env::current_exe().expect("locate test executable");
    dir.pop(); // drop the test-binary file name
    if dir.ends_with("deps") {
        dir.pop();
    }
    dir.join(format!("nsl{}", std::env::consts::EXE_SUFFIX))
}
