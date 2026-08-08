#![cfg(feature = "interop")]
//! nsl_model_lookup_function returns the cached fn pointer for a known
//! @export (deterministic across calls) and 0 for unknown names.

use std::ffi::CString;

fn build_lib(tag: &str) -> (std::path::PathBuf, std::path::PathBuf) {
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
    let status = Command::new(nsl_bin())
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
    // A full TMPDIR makes the LINKER fail here, and the failure is
    // indistinguishable from a compile refusal unless you read the child's
    // stderr — which is exactly how this file's two tests presented when /tmp
    // (a 31 GB tmpfs) filled up: as if an @export signature had been
    // rejected.
    assert!(
        status.success(),
        "nsl build --shared-lib failed (exit {:?}). If TMPDIR is full the \
         linker fails here and it looks like a compile refusal — check the \
         child's stderr above and `df -h` on TMPDIR before suspecting the \
         compiler.",
        status.code()
    );
    (lib, weights)
}

/// Remove the ~138 MB scratch dir this test's shared library links into.
///
/// This file is the only one of its four siblings with no cleanup, so its
/// directories accumulated one per suite run. On a machine where TMPDIR is a
/// tmpfs that eventually exhausts it, and the linker then starts failing
/// across the WHOLE workspace — a failure that looks like anything except a
/// disk problem. `NSL_KEEP_TEMP=1` keeps them, matching the siblings.
fn cleanup_scratch(dir: &std::path::Path) {
    if std::env::var("NSL_KEEP_TEMP").as_deref() == Ok("1") {
        return;
    }
    let _ = std::fs::remove_dir_all(dir);
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
    cleanup_scratch(lib.parent().expect("lib sits in its scratch dir"));
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
    cleanup_scratch(lib.parent().expect("lib sits in its scratch dir"));
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
