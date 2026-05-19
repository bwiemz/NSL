//! ExportRegistry::from_library_path enumerates and resolves every export
//! from the .so's own runtime-FFI introspection (nsl_get_num_exports +
//! nsl_get_export_name + dlsym per name).

fn build_test_lib(nsl_src: &str) -> std::path::PathBuf {
    use assert_cmd::prelude::*;
    use std::process::Command;
    let tmp = std::env::temp_dir().join(format!("nsl_reg_test_{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = tmp.join("test.nsl");
    std::fs::write(&src, nsl_src).unwrap();
    let lib_ext = if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let out = tmp.join(format!("libtest.{lib_ext}"));

    // NSL_STDLIB_PATH must be set so stdlib imports resolve.
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
            out.to_str().unwrap(),
        ])
        .status()
        .unwrap();
    assert!(status.success());
    out
}

#[test]
fn registry_resolves_every_export() {
    let lib_path = build_test_lib(
        r#"
@export
fn alpha(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x

@export
fn beta(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#,
    );
    let registry = nsl_runtime::c_api::exports::ExportRegistry::from_library_path(&lib_path)
        .expect("registry populates without error");
    assert_eq!(registry.len(), 2);
    assert!(registry.lookup("alpha").is_some());
    assert!(registry.lookup("beta").is_some());
    assert!(registry.lookup("does_not_exist").is_none());
}

#[test]
fn registry_missing_runtime_ffis_returns_typed_error() {
    // A library without the export-table FFIs (or a non-shared-object file)
    // should fail with a specific error variant — not a panic.
    let dummy = std::env::temp_dir().join("dummy_no_ffis.so");
    std::fs::write(&dummy, b"not a real shared object").unwrap();
    let err = nsl_runtime::c_api::exports::ExportRegistry::from_library_path(&dummy)
        .expect_err("should fail to load");
    let s = format!("{:?}", err);
    assert!(
        s.contains("LibraryOpen") || s.contains("ExportTableMissing"),
        "expected LibraryOpen or ExportTableMissing, got: {}",
        s
    );
}
