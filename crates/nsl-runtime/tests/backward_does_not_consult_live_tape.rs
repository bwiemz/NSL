#![cfg(feature = "interop")]
//! Spec B §5.1 — Backward replays from `GradContext`, NEVER from the
//! live thread-local tape.
//!
//! The test creates two grad contexts on the same thread by running
//! `nsl_model_forward_grad` twice. After the second forward, the
//! thread-local TAPE has been re-populated (and then emptied into
//! ctx_b). Backward on ctx_a MUST replay from ctx_a's stored ops,
//! NOT from the now-stale thread-local.
//!
//! Fixture: a two-input export `forward(x, w) -> x * w` plus a
//! `w.safetensors` containing a single learnable parameter `w = 2.0`.
//! The safetensors loader populates `model.weight_ptrs = [w_ptr]`, so
//! `nsl_model_forward_grad` will register `w` as a tape parameter.
//! Forward records `Mul { a: x, b: w, .. }`. Backward seeds the
//! adjoint with `ones_like(loss)` and propagates `dloss/dw = x`.
//!
//! Load-bearing assertion: after backward on ctx_a, `grad_w[0] == 3.0`
//! (matching x_a). If backward consulted the live tape, it would see
//! ctx_b's ops (x_b = 7.0) instead.

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

/// Build a `forward(x, w) -> x * w` shared library and a one-parameter
/// safetensors weight file containing `w = w_val`. Returns the
/// `(lib_path, weights_path)` pair, both inside a per-test temp dir.
fn build_mul_lib(w_val: f32) -> (std::path::PathBuf, std::path::PathBuf) {
    use assert_cmd::prelude::*;
    use std::process::Command;
    let nsl = r#"
@export
fn forward(x: Tensor<[1], f32>, w: Tensor<[1], f32>) -> Tensor<[1], f32>:
    return x * w
"#;
    let tmp = std::env::temp_dir().join(format!(
        "nsl_inv_{}_{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .subsec_nanos()
    ));
    std::fs::create_dir_all(&tmp).unwrap();
    let src = tmp.join("m.nsl");
    std::fs::write(&src, nsl).unwrap();

    let weights = tmp.join("w.safetensors");
    write_test_safetensors_w(&weights, w_val);

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
        .expect("nsl build");
    assert!(status.success(), "nsl build --shared-lib failed");
    (lib, weights)
}

/// Minimal safetensors writer for a one-element f32 tensor named `w`.
/// Format: 8-byte LE header_len, then JSON header, then raw f32 data.
fn write_test_safetensors_w(path: &std::path::Path, w_val: f32) {
    use std::io::Write;
    let header = r#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let header_bytes = header.as_bytes();
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(&(header_bytes.len() as u64).to_le_bytes()).unwrap();
    f.write_all(header_bytes).unwrap();
    f.write_all(&w_val.to_le_bytes()).unwrap();
}

#[test]
fn backward_does_not_consult_live_tape() {
    // Build a one-parameter linear-ish model: forward(x, w) -> x * w.
    // w = 2.0 from safetensors.
    let (lib, weights) = build_mul_lib(2.0);
    let w_cstr = CString::new(weights.to_str().unwrap()).unwrap();
    let l_cstr = CString::new(lib.to_str().unwrap()).unwrap();
    let model = nsl_runtime::c_api::nsl_model_create_with_lib(
        w_cstr.as_ptr() as i64,
        l_cstr.as_ptr() as i64,
    );
    assert_ne!(model, 0, "model_create_with_lib failed");

    // The model loaded `w` from safetensors → model.weight_ptrs = [w_ptr].
    // Fetch that pointer so we can pass it as the second input to forward.
    let n_weights = nsl_runtime::c_api::nsl_model_get_num_weights(model);
    assert_eq!(n_weights, 1, "expected 1 weight (w) loaded");
    let weight_ptrs_arr = nsl_runtime::c_api::nsl_model_get_weight_ptrs(model);
    assert_ne!(weight_ptrs_arr, 0);
    let w_tensor_ptr: i64 = unsafe { *(weight_ptrs_arr as *const i64) };
    assert_ne!(w_tensor_ptr, 0);

    // Build a Desc that wraps the loaded `w` tensor (shape=[1], f32, CPU).
    let mut w_shape = vec![1i64];
    // Read the underlying buffer pointer from the NslTensor at w_tensor_ptr.
    // We use nsl_tensor_get_data_ptr via a small unsafe cast through the
    // struct's known layout — but the simpler path is to call the
    // existing public FFI nsl_tensor_to_desc_ffi which writes a Desc-
    // compatible struct.
    let mut w_desc = Desc {
        shape: w_shape.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        ndim: 1,
        dtype: 0,
        device_type: 0,
        device_id: 0,
        data: std::ptr::null_mut(),
    };
    nsl_runtime::c_api::nsl_tensor_to_desc_ffi(
        w_tensor_ptr,
        &mut w_desc as *mut Desc as i64,
    );
    // After the FFI write, w_desc.data points at the underlying f32 buffer
    // and w_desc.shape may point at the tensor's owned shape array. Keep
    // w_shape allocation alive regardless.

    // -------------------- Forward A: x = 3.0 --------------------
    let mut x_a = vec![3.0f32];
    let mut sa = vec![1i64];
    let in_a_x = Desc {
        data: x_a.as_mut_ptr() as *mut c_void,
        shape: sa.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        ndim: 1,
        dtype: 0,
        device_type: 0,
        device_id: 0,
    };
    // Pack the two inputs (x, w) into a contiguous Desc array.
    let inputs_a = [
        Desc {
            data: in_a_x.data,
            shape: in_a_x.shape,
            strides: std::ptr::null_mut(),
            ndim: 1,
            dtype: 0,
            device_type: 0,
            device_id: 0,
        },
        Desc {
            data: w_desc.data,
            shape: w_desc.shape,
            strides: std::ptr::null_mut(),
            ndim: 1,
            dtype: 0,
            device_type: 0,
            device_id: 0,
        },
    ];

    let mut y_a = vec![0.0f32];
    let mut so_a = vec![1i64];
    let out_a = [Desc {
        data: y_a.as_mut_ptr() as *mut c_void,
        shape: so_a.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        ndim: 1,
        dtype: 0,
        device_type: 0,
        device_id: 0,
    }];

    let mut ctx_a: i64 = 0;
    let rc = nsl_runtime::grad_context::nsl_model_forward_grad(
        model,
        inputs_a.as_ptr() as i64,
        2,
        out_a.as_ptr() as i64,
        1,
        &mut ctx_a as *mut i64 as i64,
    );
    assert_eq!(rc, 0, "forward A failed");
    assert_ne!(ctx_a, 0, "ctx_a must be populated on success");
    assert_eq!(y_a[0], 3.0 * 2.0, "forward A produced wrong output");

    // -------------------- Forward B: x = 7.0 --------------------
    // CRITICAL: this re-populates the thread-local TAPE with different
    // ops, then moves them into ctx_b. The TAPE is now empty.
    let mut x_b = vec![7.0f32];
    let mut sb = vec![1i64];
    let inputs_b = [
        Desc {
            data: x_b.as_mut_ptr() as *mut c_void,
            shape: sb.as_mut_ptr(),
            strides: std::ptr::null_mut(),
            ndim: 1,
            dtype: 0,
            device_type: 0,
            device_id: 0,
        },
        Desc {
            data: w_desc.data,
            shape: w_desc.shape,
            strides: std::ptr::null_mut(),
            ndim: 1,
            dtype: 0,
            device_type: 0,
            device_id: 0,
        },
    ];
    let mut y_b = vec![0.0f32];
    let mut so_b = vec![1i64];
    let out_b = [Desc {
        data: y_b.as_mut_ptr() as *mut c_void,
        shape: so_b.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        ndim: 1,
        dtype: 0,
        device_type: 0,
        device_id: 0,
    }];
    let mut ctx_b: i64 = 0;
    let rc = nsl_runtime::grad_context::nsl_model_forward_grad(
        model,
        inputs_b.as_ptr() as i64,
        2,
        out_b.as_ptr() as i64,
        1,
        &mut ctx_b as *mut i64 as i64,
    );
    assert_eq!(rc, 0, "forward B failed");
    assert_ne!(ctx_b, 0);
    assert_eq!(y_b[0], 7.0 * 2.0, "forward B produced wrong output");

    // -------------------- Backward on ctx_a --------------------
    // Backward on ctx_a MUST use ctx_a's ops (recorded with x=3.0), not
    // the live tape (which was last populated by forward B with x=7.0).
    let mut seed = vec![1.0f32];
    let mut s_seed = vec![1i64];
    let grad_out = [Desc {
        data: seed.as_mut_ptr() as *mut c_void,
        shape: s_seed.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        ndim: 1,
        dtype: 0,
        device_type: 0,
        device_id: 0,
    }];
    // Output buffer: 1 desc for grad_w (the only learnable param).
    let mut grad_w = vec![0.0f32];
    let mut sw_g = vec![1i64];
    let grad_descs = [Desc {
        data: grad_w.as_mut_ptr() as *mut c_void,
        shape: sw_g.as_mut_ptr(),
        strides: std::ptr::null_mut(),
        ndim: 1,
        dtype: 0,
        device_type: 0,
        device_id: 0,
    }];
    let rc = nsl_runtime::grad_context::nsl_model_backward(
        ctx_a,
        grad_out.as_ptr() as i64,
        1,
        grad_descs.as_ptr() as i64,
        1,
    );
    assert_eq!(rc, 0, "backward on ctx_a failed");

    // dy/dw = x = 3.0 from forward A's recorded ops. If backward
    // consulted the live tape (re-populated by forward B), grad_w
    // would equal 7.0 instead.
    assert_eq!(
        grad_w[0], 3.0,
        "grad_w must equal x_a=3.0, not x_b=7.0 — backward consulted live tape"
    );

    // -------------------- Cleanup --------------------
    nsl_runtime::grad_context::nsl_grad_context_destroy(ctx_a);
    nsl_runtime::grad_context::nsl_grad_context_destroy(ctx_b);
    nsl_runtime::c_api::nsl_model_destroy(model);
}
