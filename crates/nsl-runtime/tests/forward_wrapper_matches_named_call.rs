#![cfg(feature = "interop")]
//! nsl_model_forward and nsl_model_call(_, "forward", _) must produce
//! bit-identical results once forward becomes a wrapper over named
//! dispatch.

use std::ffi::CString;
use std::os::raw::c_void;

#[repr(C)]
#[derive(Default)]
struct Desc {
    data: *mut c_void,
    shape: *mut i64,
    strides: *mut i64,
    ndim: i32,
    dtype: i32,
    device_type: i32,
    device_id: i32,
}

fn build_forward_lib() -> (std::path::PathBuf, std::path::PathBuf) {
    use assert_cmd::prelude::*;
    use std::process::Command;
    let nsl = r#"
@export
fn forward(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
"#;
    let tmp = std::env::temp_dir().join(format!("nsl_fwd_{}", std::process::id()));
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
fn forward_and_named_call_produce_identical_outputs() {
    let (lib, weights) = build_forward_lib();
    let w = CString::new(weights.to_str().unwrap()).unwrap();
    let l = CString::new(lib.to_str().unwrap()).unwrap();
    let model =
        nsl_runtime::c_api::nsl_model_create_with_lib(w.as_ptr() as i64, l.as_ptr() as i64);
    assert_ne!(model, 0);

    let mut input = vec![1.5f32, 2.5, 3.5, 4.5];
    let mut shape = vec![4i64];
    let in_desc = Desc {
        data: input.as_mut_ptr() as *mut c_void,
        shape: shape.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        ndim: 1,
        dtype: 0,
        device_type: 0,
        device_id: 0,
    };

    let mut shape_a = vec![4i64];
    let mut out_a = vec![0.0f32; 4];
    let out_desc_a = Desc {
        data: out_a.as_mut_ptr() as *mut c_void,
        shape: shape_a.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        ndim: 1,
        dtype: 0,
        device_type: 0,
        device_id: 0,
    };
    let rc_a = nsl_runtime::c_api::nsl_model_forward(
        model,
        &in_desc as *const Desc as i64,
        1,
        &out_desc_a as *const Desc as *mut Desc as i64,
        1,
    );

    let mut shape_b = vec![4i64];
    let mut out_b = vec![0.0f32; 4];
    let out_desc_b = Desc {
        data: out_b.as_mut_ptr() as *mut c_void,
        shape: shape_b.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        ndim: 1,
        dtype: 0,
        device_type: 0,
        device_id: 0,
    };
    let fwd = CString::new("forward").unwrap();
    let rc_b = nsl_runtime::c_api::nsl_model_call(
        model,
        fwd.as_ptr() as i64,
        &in_desc as *const Desc as i64,
        1,
        &out_desc_b as *const Desc as *mut Desc as i64,
        1,
    );

    assert_eq!(rc_a, rc_b, "return codes must match");
    assert_eq!(out_a, out_b, "outputs must be bit-identical");

    nsl_runtime::c_api::nsl_model_destroy(model);
}
