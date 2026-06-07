#![cfg(feature = "interop")]
//! The public `nsl_model_create(weights_path)` exported by every shared
//! library populates the ExportRegistry internally (via the runtime's
//! self-discovery of its containing .so via dladdr / GetModuleHandleEx,
//! then a call to `nsl_model_create_with_lib`). No caller-side knowledge
//! of the .so path is required.

use std::ffi::CString;

#[test]
fn auto_create_populates_registry_from_dlpath() {
    use std::process::Command;
    let nsl = r#"
@export
fn ping(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
    let tmp = std::env::temp_dir().join(format!("nsl_auto_{}", std::process::id()));
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
    Command::new(nsl_bin())
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

    // Load the .so via libloading and call its own nsl_model_create FFI.
    // The runtime's nsl_model_create — statically linked into the .so —
    // should internally self-discover its containing library and populate
    // the registry without the caller passing a lib path.
    let library = unsafe { libloading::Library::new(&lib) }.unwrap();
    let create: libloading::Symbol<unsafe extern "C" fn(i64) -> i64> =
        unsafe { library.get(b"nsl_model_create").unwrap() };
    let w = CString::new(weights.to_str().unwrap()).unwrap();
    let model = unsafe { create(w.as_ptr() as i64) };
    assert_ne!(model, 0, "auto-create must populate registry from own dl path");

    let count: libloading::Symbol<unsafe extern "C" fn(i64) -> i64> =
        unsafe { library.get(b"nsl_model_export_count").unwrap() };
    assert_eq!(unsafe { count(model) }, 1);

    let destroy: libloading::Symbol<unsafe extern "C" fn(i64) -> i64> =
        unsafe { library.get(b"nsl_model_destroy").unwrap() };
    unsafe { destroy(model) };
    drop(library);
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
