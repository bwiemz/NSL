//! `--checkpoint-compress` stores a saved activation as bf16 and the
//! backward restores it by widening to f32. That round trip returns the
//! original's dtype only when the original was f32. The cast used to accept
//! any float dtype, and the restore then handed the backward an f32 tensor in
//! place of an f64 or half-precision one, with no diagnostic.
//!
//! `nsl_ccr_compress_bf16` accepts f32 and refuses everything else with the
//! unsupported-dtype exit code and a message that names the flag.
//!
//! The refusal is a process exit (`fatal::die`), so each refusing case runs
//! in a child: this test binary re-executed with a scenario in the
//! environment, the pattern `mixed_dtype_refusal.rs` uses.

use std::ffi::c_void;

use nsl_runtime::fatal::NSL_EXIT_UNSUPPORTED_DTYPE;

unsafe extern "C" {
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
    fn nsl_tensor_full_dtype(shape_list: i64, value: f64, dtype: i64) -> i64;
    fn nsl_tensor_to_dtype(src: i64, target_dtype: i64) -> i64;
    fn nsl_tensor_to_f32(src: i64) -> i64;
    fn nsl_ccr_compress_bf16(src: i64) -> i64;
}

const F64: i64 = 0;
const F32: i64 = 1;
const FP16: i64 = 2;
const BF16: i64 = 3;

/// Mirror of the runtime's `#[repr(C)]` `NslTensor` header, for reading a
/// result's dtype and elements (the same mirror `mixed_dtype_refusal.rs`
/// keeps).
#[repr(C)]
struct TensorView {
    _magic: u32,
    data: *mut c_void,
    _shape: *mut i64,
    _strides: *mut i64,
    _ndim: i64,
    len: i64,
    _refcount: std::sync::atomic::AtomicI64,
    _device: u8,
    dtype: u16,
    _owns_data: u8,
    _data_owner: i64,
    _slab_managed: u8,
    _tape_id: i64,
}

fn view(ptr: i64) -> &'static TensorView {
    assert_ne!(ptr, 0, "null tensor");
    unsafe { &*(ptr as *const TensorView) }
}

fn full(dims: &[i64], value: f64, dtype: i64) -> i64 {
    let list = unsafe { nsl_list_new() };
    for &d in dims {
        unsafe { nsl_list_push(list, d) };
    }
    let t = unsafe { nsl_tensor_full_dtype(list, value, dtype) };
    assert_eq!(
        view(t).dtype as i64,
        dtype,
        "the producer must make the dtype asked for"
    );
    t
}

/// An f32 activation compresses to bf16 and restores to f32. The values
/// here are exactly representable in bf16 (8 significant bits), so the
/// round trip must return them unchanged: 2.5 = 1.01b x 2^1, whose bf16 bit
/// pattern is 0x4020 (sign 0, exponent 128, top mantissa bits 0100000).
#[test]
fn an_f32_activation_compresses_to_bf16_and_restores_to_f32() {
    let src = full(&[2, 3], 2.5, F32);
    let half = unsafe { nsl_ccr_compress_bf16(src) };
    let h = view(half);
    assert_eq!(h.dtype as i64, BF16);
    assert_eq!(h.len, 6);
    for i in 0..6 {
        assert_eq!(
            unsafe { *(h.data as *const u16).add(i) },
            0x4020,
            "element {i}"
        );
    }
    let back = unsafe { nsl_tensor_to_f32(half) };
    let b = view(back);
    assert_eq!(b.dtype as i64, F32);
    for i in 0..6 {
        assert_eq!(
            unsafe { *(b.data as *const f32).add(i) },
            2.5,
            "element {i}"
        );
    }
}

const SCENARIO: &str = "CCR_TEST_COMPRESS_SCENARIO";

/// One compress call per scenario. Returns immediately unless the parent set
/// the scenario; is expected to EXIT inside the call.
#[test]
fn zz_ccr_compress_child() {
    let Ok(scenario) = std::env::var(SCENARIO) else {
        return;
    };
    let src = match scenario.as_str() {
        "f64" => full(&[2, 3], 1.5, F64),
        "fp16" => unsafe { nsl_tensor_to_dtype(full(&[2, 3], 1.5, F32), FP16) },
        "bf16" => unsafe { nsl_tensor_to_dtype(full(&[2, 3], 1.5, F32), BF16) },
        other => panic!("unknown {SCENARIO} '{other}'"),
    };
    let _ = unsafe { nsl_ccr_compress_bf16(src) };
    // Only reached if nothing refused.
    println!("COMPRESS-NOT-REFUSED scenario={scenario}");
}

/// Run `scenario` in a fresh process; require exit code 17
/// (`Fatal::UnsupportedDtype`) and a message naming the flag and the dtype.
fn assert_refused(scenario: &str) {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([
            "zz_ccr_compress_child",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(SCENARIO, scenario)
        .env("RUST_BACKTRACE", "0")
        .output()
        .expect("re-exec the test binary");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    let ctx = format!("--- child stdout ---\n{stdout}\n--- child stderr ---\n{stderr}");
    assert!(
        !stdout.contains("COMPRESS-NOT-REFUSED"),
        "scenario '{scenario}': a {scenario} activation was compressed.\n{ctx}"
    );
    assert_eq!(
        out.status.code(),
        Some(NSL_EXIT_UNSUPPORTED_DTYPE),
        "scenario '{scenario}': expected the unsupported-dtype exit code.\n{ctx}"
    );
    let want = format!("--checkpoint-compress: a saved activation has dtype {scenario}");
    assert!(
        stderr.contains(&want),
        "scenario '{scenario}': expected `{want}`.\n{ctx}"
    );
}

#[test]
fn an_activation_that_is_not_f32_is_refused() {
    assert_refused("f64");
    assert_refused("fp16");
    assert_refused("bf16");
}
