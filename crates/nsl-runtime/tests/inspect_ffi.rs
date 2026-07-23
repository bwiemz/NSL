use nsl_runtime::inspect::ffi::{nsl_inspect_record_stats, nsl_inspect_set_dir};
use std::sync::{Mutex, MutexGuard, OnceLock};

// `nsl_inspect_set_dir` mutates a process-global path; tests running in
// parallel race on it (one test's record_stats lands in another test's
// temp dir and the path-exists assertion fails). Serialize the tests that
// touch the global so the suite is safe under `cargo test` without needing
// `--test-threads=1`.
fn serial_lock() -> MutexGuard<'static, ()> {
    static SERIAL_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    let m = SERIAL_GUARD.get_or_init(|| Mutex::new(()));
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn set_tmp_dir() -> tempfile::TempDir {
    let tmp = tempfile::tempdir().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    let bytes = path.as_bytes();
    unsafe {
        nsl_inspect_set_dir(bytes.as_ptr(), bytes.len());
    }
    tmp
}

#[test]
fn record_stats_writes_expected_file_layout() {
    let _g = serial_lock();
    let dir = set_tmp_dir();
    let stats: [f64; 6] = [1.5, 0.3, -2.0, 4.5, 0.0, 0.0];
    let name = "h0";
    let nb = name.as_bytes();
    let rc = unsafe { nsl_inspect_record_stats(stats.as_ptr(), 100, nb.as_ptr(), nb.len()) };
    assert_eq!(rc, 0, "expected 0, got {}", rc);

    let path = dir.path().join("step_100_h0.stats.bin");
    assert!(path.exists(), "expected stats file at {}", path.display());
    let bytes = std::fs::read(&path).unwrap();
    assert_eq!(&bytes[0..4], b"NSLI");
}

#[test]
fn record_stats_null_pointer_returns_error() {
    let _g = serial_lock();
    let _dir = set_tmp_dir();
    let rc = unsafe { nsl_inspect_record_stats(std::ptr::null(), 1, std::ptr::null(), 0) };
    assert_ne!(rc, 0);
}

#[test]
fn record_stats_creates_dir_if_missing() {
    let _g = serial_lock();
    let outer = tempfile::tempdir().unwrap();
    let nested = outer.path().join("a").join("b");
    let pstr = nested.to_str().unwrap();
    let pb = pstr.as_bytes();
    unsafe {
        nsl_inspect_set_dir(pb.as_ptr(), pb.len());
    }
    let stats: [f64; 6] = [0.0; 6];
    let name = "x";
    let nb = name.as_bytes();
    let rc = unsafe { nsl_inspect_record_stats(stats.as_ptr(), 0, nb.as_ptr(), nb.len()) };
    assert_eq!(rc, 0);
    assert!(nested.join("step_0_x.stats.bin").exists());
}

#[test]
fn nsl_tensor_stats_symbol_exists() {
    let _f: extern "C" fn(i64, *mut f64) -> i32 = nsl_runtime::inspect::stats_kernel::nsl_tensor_stats;
}

#[test]
fn nsl_inspect_dump_full_symbol_exists() {
    let _f: unsafe extern "C" fn(i64, u64, *const u8, usize) -> i32 =
        nsl_runtime::inspect::ffi::nsl_inspect_dump_full;
}

// ---------------------------------------------------------------------------
// `nsl_tensor_stats` contract tests (@inspect stats path).
//
// Locks in three behaviors:
// 1. CPU six-slot contract: `[mean, std, min, max, nan_count, inf_count]`
//    with NaN/Inf-skipping semantics and SAMPLE std (n-1 divisor).
// 2. Garbage-record regression: codegen at `@inspect` sites ignores the
//    return code of `nsl_tensor_stats` and persists the buffer verbatim
//    into `.nsl-inspect/step_*.stats.bin`. Every error path must therefore
//    leave the buffer fully zeroed, never uninitialized garbage. The tests
//    pre-poison the buffer with `f64::MAX` sentinels to prove zero-fill.
// 3. GPU parity: stats of a GPU-resident tensor (D2H staging path) match
//    the CPU result exactly. Gated `#[ignore]` + `cuda_available()` probe
//    like `precision_cast_gpu_dispatch.rs`; run with:
//
//    cargo test -p nsl-runtime --features cuda \
//        --test inspect_ffi -- --include-ignored --nocapture
// ---------------------------------------------------------------------------

use nsl_runtime::inspect::stats_kernel::nsl_tensor_stats;
use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{nsl_tensor_free, nsl_tensor_from_static};

/// NslTensor dtype tags (runtime ABI: 0=f64, 1=f32, 9=i32).
const DTYPE_F64: i64 = 0;
const DTYPE_F32: i64 = 1;
const DTYPE_I32: i64 = 9;

/// Build a rank-1 CPU tensor over a leaked buffer (`owns_data = 0`; the
/// per-test leak is tiny and keeps the data alive for the tensor's lifetime).
fn tensor_from_raw<T: Copy + 'static>(vals: &[T], dtype: i64) -> i64 {
    let boxed: Box<[T]> = vals.to_vec().into_boxed_slice();
    let leaked: &'static [T] = Box::leak(boxed);
    let data_ptr = leaked.as_ptr() as i64;
    let shape_list = nsl_list_new();
    nsl_list_push(shape_list, vals.len() as i64);
    let t = nsl_tensor_from_static(data_ptr, shape_list, dtype);
    nsl_list_free(shape_list);
    t
}

/// Run `nsl_tensor_stats` against a pre-poisoned sentinel buffer; return
/// `(rc, buf)` so callers can assert both the code and the slot contents.
fn stats_with_sentinel(t: i64) -> (i32, [f64; 6]) {
    let mut buf = [f64::MAX; 6];
    let rc = nsl_tensor_stats(t, buf.as_mut_ptr());
    (rc, buf)
}

/// Slot order + NaN/Inf semantics on a known CPU f64 tensor:
/// finite values {1, 2, 3, 4, 10}: mean = 4, sample var = 50/4 = 12.5,
/// min = 1, max = 10; one NaN and two Infs are skipped but counted.
#[test]
fn cpu_stats_contract_with_nan_and_inf() {
    let vals = [
        1.0_f64,
        2.0,
        3.0,
        4.0,
        f64::NAN,
        f64::INFINITY,
        f64::NEG_INFINITY,
        10.0,
    ];
    let t = tensor_from_raw(&vals, DTYPE_F64);
    let (rc, buf) = stats_with_sentinel(t);
    assert_eq!(rc, 0, "CPU f64 stats must succeed");
    assert_eq!(buf[0], 4.0, "slot 0 = mean over finite values");
    assert_eq!(buf[1], 12.5_f64.sqrt(), "slot 1 = SAMPLE std (n-1 divisor)");
    assert_eq!(buf[2], 1.0, "slot 2 = min over finite values");
    assert_eq!(buf[3], 10.0, "slot 3 = max over finite values");
    assert_eq!(buf[4], 1.0, "slot 4 = nan_count");
    assert_eq!(buf[5], 2.0, "slot 5 = inf_count (+inf and -inf both count)");
    nsl_tensor_free(t);
}

/// f32 CPU tensors are widened elementwise to f64 before the reduction.
#[test]
fn cpu_stats_f32_widens_to_f64() {
    let vals = [1.5_f32, -2.5, 4.0];
    let t = tensor_from_raw(&vals, DTYPE_F32);
    let (rc, buf) = stats_with_sentinel(t);
    assert_eq!(rc, 0, "CPU f32 stats must succeed");
    assert_eq!(buf[0], 1.0, "mean of {{1.5, -2.5, 4.0}}");
    let var = ((1.5 - 1.0f64).powi(2) + (-2.5 - 1.0f64).powi(2) + (4.0 - 1.0f64).powi(2)) / 2.0;
    assert_eq!(buf[1], var.sqrt(), "sample std");
    assert_eq!(buf[2], -2.5);
    assert_eq!(buf[3], 4.0);
    assert_eq!(buf[4], 0.0);
    assert_eq!(buf[5], 0.0);
    nsl_tensor_free(t);
}

/// Degenerate input: nothing finite. mean/std/min/max collapse to 0.0 (not
/// +/-inf sentinels) and the counts still report what was skipped.
#[test]
fn cpu_stats_all_nonfinite() {
    let vals = [f64::NAN, f64::NAN, f64::INFINITY];
    let t = tensor_from_raw(&vals, DTYPE_F64);
    let (rc, buf) = stats_with_sentinel(t);
    assert_eq!(rc, 0);
    assert_eq!(buf, [0.0, 0.0, 0.0, 0.0, 2.0, 1.0]);
    nsl_tensor_free(t);
}

/// REGRESSION (garbage-record bug): a null tensor handle must return a
/// nonzero rc AND zero the whole buffer — codegen persists the buffer even
/// when the rc is nonzero, so sentinels surviving here would mean garbage
/// stats records on disk.
#[test]
fn error_path_null_tensor_zero_fills_buffer() {
    let (rc, buf) = stats_with_sentinel(0);
    assert_eq!(rc, 2, "null tensor handle must return rc=2");
    assert_eq!(
        buf,
        [0.0; 6],
        "error path must zero-fill the buffer, not leave sentinels/garbage"
    );
}

/// REGRESSION: the compute-error path (rc=3, e.g. unsupported dtype) must
/// also leave a fully zeroed buffer.
#[test]
fn error_path_unsupported_dtype_zero_fills_buffer() {
    let vals = [1_i32, 2, 3];
    let t = tensor_from_raw(&vals, DTYPE_I32);
    let (rc, buf) = stats_with_sentinel(t);
    assert_eq!(rc, 3, "unsupported dtype must return rc=3");
    assert_eq!(buf, [0.0; 6], "rc=3 path must zero-fill the buffer");
    nsl_tensor_free(t);
}

/// Null out-buffer is rejected before anything is written.
#[test]
fn error_path_null_out_buf() {
    let rc = nsl_tensor_stats(0, std::ptr::null_mut());
    assert_eq!(rc, 1, "null out_buf must return rc=1");
}

/// Without the `cuda` feature, a GPU-resident tensor must refuse honestly:
/// nonzero rc + zeroed buffer (never a read of the fake device pointer).
/// Uses `nsl_tensor_from_slab` (slab_managed=1) so `nsl_tensor_free` never
/// tries to release the fake data pointer.
#[cfg(not(feature = "cuda"))]
#[test]
fn gpu_tensor_without_cuda_feature_refuses_with_zeroed_buffer() {
    let fake_device_data = [0.0_f32; 4];
    let shape_list = nsl_list_new();
    nsl_list_push(shape_list, 4);
    let t = nsl_runtime::tensor::nsl_tensor_from_slab(
        fake_device_data.as_ptr() as i64,
        shape_list,
        1, // device = GPU
        DTYPE_F32,
    );
    nsl_list_free(shape_list);

    let (rc, buf) = stats_with_sentinel(t);
    assert_eq!(rc, 3, "GPU tensor without cuda feature must return rc=3");
    assert_eq!(buf, [0.0; 6], "refusal must zero-fill the buffer");
    nsl_tensor_free(t);
}

/// Probe for a usable GPU (pattern from precision_cast_gpu_dispatch.rs).
#[cfg(feature = "cuda")]
fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    let rc = nsl_runtime::nsl_cuda_init();
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        return false;
    }
    true
}

/// GPU parity: identical stats for the same values CPU-side and GPU-side.
/// The GPU path stages contiguous -> D2H (f32 widened to f64), then runs the
/// same f64 reduction, so every slot must match the CPU result bit-exactly.
#[cfg(feature = "cuda")]
#[test]
#[ignore]
fn gpu_stats_parity_with_cpu() {
    if !cuda_available() {
        return;
    }
    let vals = [
        1.5_f32,
        -2.25,
        3.75,
        f32::NAN,
        f32::INFINITY,
        0.5,
        -8.0,
        4.25,
    ];
    let cpu_t = tensor_from_raw(&vals, DTYPE_F32);
    let (cpu_rc, cpu_buf) = stats_with_sentinel(cpu_t);
    assert_eq!(cpu_rc, 0, "CPU stats must succeed");

    let gpu_t = nsl_runtime::tensor::nsl_tensor_to_device(cpu_t, 1);
    assert_ne!(gpu_t, 0, "to_device(_, 1) must produce a GPU tensor");
    let (gpu_rc, gpu_buf) = stats_with_sentinel(gpu_t);
    assert_eq!(gpu_rc, 0, "GPU stats must succeed via D2H staging");

    let names = ["mean", "std", "min", "max", "nan_count", "inf_count"];
    for i in 0..6 {
        eprintln!(
            "parity slot {} ({}): cpu={} gpu={}",
            i, names[i], cpu_buf[i], gpu_buf[i]
        );
        assert_eq!(
            cpu_buf[i].to_bits(),
            gpu_buf[i].to_bits(),
            "slot {} ({}) mismatch: cpu={} gpu={}",
            i,
            names[i],
            cpu_buf[i],
            gpu_buf[i]
        );
    }

    nsl_tensor_free(gpu_t);
    nsl_tensor_free(cpu_t);
}
