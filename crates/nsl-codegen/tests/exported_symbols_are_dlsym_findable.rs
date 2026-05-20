//! Codegen contract test: every `@export` produces a default-visibility,
//! unmangled symbol that dlsym can find on every platform we ship to.
//!
//! Without this test, a future codegen optimization that flips
//! default-visibility off (for binary size) would silently break the
//! `nsl_model_call` dispatcher. The dispatcher is only as good as the
//! symbol table's reachability.
//!
//! Uses `assert_cmd::Command::cargo_bin("nsl")` rather than
//! `env!("CARGO_BIN_EXE_nsl")` because `nsl` lives in the sibling
//! `nsl-cli` crate, not in `nsl-codegen`.

use assert_cmd::prelude::*;
use std::process::Command;
use tempfile::TempDir;

#[test]
fn every_export_symbol_is_findable_via_dlsym() {
    let nsl_source = r#"
@export
fn alpha(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x

@export(name="custom_beta")
fn beta(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("dlsym_test.nsl");
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
    let out = tmp.path().join(format!("libdlsym_test.{lib_ext}"));

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

    // Scope the library handle so it is dropped (and the OS file lock
    // released on Windows) BEFORE `tmp` goes out of scope and tries to
    // delete the .dll. Without this ordering, TempDir's cleanup races
    // the still-mapped DLL on Windows.
    {
        let lib = unsafe { libloading::Library::new(&out) }.expect("load shared lib");

        // `alpha` keeps its NSL name; `beta` is renamed to `custom_beta` per
        // the @export(name=...) override.
        for sym in &["alpha", "custom_beta"] {
            let _: libloading::Symbol<unsafe extern "C" fn()> =
                unsafe { lib.get(sym.as_bytes()) }.unwrap_or_else(|e| {
                    panic!(
                        "export '{}' not found via dlsym (would break nsl_model_call): {}",
                        sym, e
                    )
                });
        }
    }

    drop(tmp);
}
