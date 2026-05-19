//! Codegen test: the `nsl_get_num_exports` and `nsl_get_export_name`
//! FFIs (emitted into every `--shared-lib` artifact) round-trip the
//! list of `@export`-decorated functions known to codegen at build time.
//!
//! Uses `assert_cmd::Command::cargo_bin("nsl")` rather than
//! `env!("CARGO_BIN_EXE_nsl")` because `nsl` lives in the sibling
//! `nsl-cli` crate, not in `nsl-codegen`.

use assert_cmd::prelude::*;
use std::ffi::CStr;
use std::os::raw::c_char;
use std::process::Command;
use tempfile::TempDir;

#[test]
fn export_table_round_trips_two_exports() {
    let nsl_source = r#"
@export
fn alpha(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x

@export
fn beta(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("two_exports.nsl");
    std::fs::write(&src, nsl_source).unwrap();

    // Use the platform-appropriate shared-lib extension. `nsl build
    // --shared-lib` honors `-o` and will write the requested path.
    let lib_ext = if cfg!(target_os = "windows") {
        "dll"
    } else if cfg!(target_os = "macos") {
        "dylib"
    } else {
        "so"
    };
    let out = tmp.path().join(format!("libtwo_exports.{lib_ext}"));

    let root: std::path::PathBuf = env!("CARGO_MANIFEST_DIR").into();
    let workspace_root = root.parent().unwrap().parent().unwrap();
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
        .expect("nsl build");
    assert!(status.success(), "nsl build --shared-lib failed");

    let lib = unsafe { libloading::Library::new(&out) }.expect("load shared lib");
    let get_num: libloading::Symbol<unsafe extern "C" fn() -> i64> =
        unsafe { lib.get(b"nsl_get_num_exports").expect("nsl_get_num_exports") };
    let get_name: libloading::Symbol<unsafe extern "C" fn(i64) -> *const c_char> =
        unsafe { lib.get(b"nsl_get_export_name").expect("nsl_get_export_name") };

    assert_eq!(unsafe { get_num() }, 2, "expected 2 exports");
    let n0 = unsafe { CStr::from_ptr(get_name(0)) }
        .to_str()
        .unwrap()
        .to_string();
    let n1 = unsafe { CStr::from_ptr(get_name(1)) }
        .to_str()
        .unwrap()
        .to_string();
    let mut names = vec![n0, n1];
    names.sort();
    assert_eq!(names, vec!["alpha".to_string(), "beta".to_string()]);

    // Out-of-bounds index returns NULL.
    let oob = unsafe { get_name(7) };
    assert!(oob.is_null(), "out-of-bounds index must return NULL");
}
