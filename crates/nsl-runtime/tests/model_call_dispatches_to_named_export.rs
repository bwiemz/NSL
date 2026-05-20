#![cfg(feature = "interop")]
//! nsl_model_call routes by string name to the matching @export and
//! produces the expected output.

use std::ffi::CString;
use std::os::raw::c_void;

#[repr(C)]
#[derive(Default)]
struct NslTensorDesc {
    data: *mut c_void,
    shape: *mut i64,
    strides: *mut i64,
    ndim: i32,
    dtype: i32,
    device_type: i32,
    device_id: i32,
    tape_id: i64,
}

fn build_identity_lib() -> (std::path::PathBuf, std::path::PathBuf) {
    use assert_cmd::prelude::*;
    use std::process::Command;
    let nsl = r#"
@export
fn identity(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
    let tmp = std::env::temp_dir().join(format!("nsl_call_{}", std::process::id()));
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

// Dispatch goes through the packed-array sibling wrapper
// (`<name>__nsl_dispatch`) emitted alongside the typed `<name>` wrapper.
// The dispatch wrapper unpacks the descriptor arrays and forwards into the
// typed wrapper, so the ABI presented to `nsl_model_call` matches
// `ExportFnPtr` while the typed wrapper still exists for direct ctypes
// callers (Spec §2.3 ABI bit-stability).
#[test]
fn call_routes_to_named_export_and_produces_output() {
    let (lib, weights) = build_identity_lib();
    let weights_cstr = CString::new(weights.to_str().unwrap()).unwrap();
    let lib_cstr = CString::new(lib.to_str().unwrap()).unwrap();
    let model = nsl_runtime::c_api::nsl_model_create_with_lib(
        weights_cstr.as_ptr() as i64,
        lib_cstr.as_ptr() as i64,
    );
    assert_ne!(model, 0);

    let mut input_data = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut input_shape = vec![4i64];
    let input_desc = NslTensorDesc {
        data: input_data.as_mut_ptr() as *mut c_void,
        shape: input_shape.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        ndim: 1,
        dtype: 0,
        device_type: 0,
        device_id: 0,
        tape_id: 0,
    };
    let mut output_shape = vec![4i64];
    let mut output_data = vec![0.0f32; 4];
    let mut output_desc = NslTensorDesc {
        data: output_data.as_mut_ptr() as *mut c_void,
        shape: output_shape.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        ndim: 1,
        dtype: 0,
        device_type: 0,
        device_id: 0,
        tape_id: 0,
    };

    let name = CString::new("identity").unwrap();
    let rc = nsl_runtime::c_api::nsl_model_call(
        model,
        name.as_ptr() as i64,
        &input_desc as *const NslTensorDesc as i64,
        1,
        &mut output_desc as *mut NslTensorDesc as i64,
        1,
    );
    assert_eq!(rc, 0, "nsl_model_call returned non-zero");
    assert_eq!(output_data, vec![1.0, 2.0, 3.0, 4.0]);

    nsl_runtime::c_api::nsl_model_destroy(model);
}
