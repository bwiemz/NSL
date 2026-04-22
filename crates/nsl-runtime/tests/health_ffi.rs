//! FFI hook tests for the health monitor.
//!
//! These tests share a single global `Mutex<HealthCollector>` (see
//! `health::ffi::COLLECTOR`). Parallel execution races on the collector
//! state — each test records data and then flushes, and a concurrent test
//! can overwrite the record-to-flush sequence under that lock. The
//! `SERIAL_GUARD` below pins one test at a time so the suite is safe under
//! `cargo test` without needing `--test-threads=1` on the CLI.

use nsl_runtime::health::ffi::{
    nsl_health_flush_snapshot, nsl_health_get_grad_norm_total, nsl_health_get_loss_ema,
    nsl_health_get_loss_ema_slope, nsl_health_get_nan_inf_count_window,
    nsl_health_record_grad_norm, nsl_health_record_loss, nsl_health_record_weight_norm,
};
use std::sync::{Mutex, MutexGuard, OnceLock};

fn serial_lock() -> MutexGuard<'static, ()> {
    static SERIAL_GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    let m = SERIAL_GUARD.get_or_init(|| Mutex::new(()));
    // Poisoned-mutex recovery: a prior test's panic shouldn't cascade into
    // every subsequent test aborting on `unwrap()`.
    m.lock().unwrap_or_else(|e| e.into_inner())
}

fn flush_to_string() -> String {
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let path = tmp.path().to_str().unwrap().to_string();
    let bytes = path.as_bytes();
    let rc = unsafe { nsl_health_flush_snapshot(bytes.as_ptr(), bytes.len()) };
    assert_eq!(rc, 0, "flush returned non-zero: {}", rc);
    std::fs::read_to_string(tmp.path()).unwrap()
}

#[test]
fn record_loss_then_flush_writes_json_with_step_and_loss() {
    let _g = serial_lock();
    nsl_health_record_loss(2.5, 7);
    let json = flush_to_string();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["step"], 7);
    assert!((v["loss"].as_f64().unwrap() - 2.5).abs() < 1e-6);
}

#[test]
fn record_grad_norm_emits_layer_entry() {
    let _g = serial_lock();
    let path = "m.transformer.h.4.attn.wq";
    let bytes = path.as_bytes();
    unsafe {
        nsl_health_record_grad_norm(bytes.as_ptr(), bytes.len(), 4, 12.5);
    }
    let json = flush_to_string();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!((v["per_layer_grad_norm"]["4"].as_f64().unwrap() - 12.5).abs() < 1e-6);
}

#[test]
fn flush_to_null_path_returns_zero() {
    let _g = serial_lock();
    let rc = unsafe { nsl_health_flush_snapshot(std::ptr::null(), 0) };
    assert_eq!(rc, 0);
}

#[test]
fn flush_invalid_utf8_returns_2() {
    let _g = serial_lock();
    let bytes = [0xFFu8, 0xFE, 0xFD];
    let rc = unsafe { nsl_health_flush_snapshot(bytes.as_ptr(), bytes.len()) };
    assert_eq!(rc, 2);
}

#[test]
fn tensor_l2_norm_symbol_exists() {
    // Link-only assertion: if this coerces, the `#[no_mangle] pub extern "C"`
    // symbol is exported. Numeric correctness of the underlying
    // `tensor_l2_norm` helper is covered by tensor-module unit tests;
    // constructing a valid NslTensor handle from a unit test requires
    // runtime scope setup that's out of scope here.
    let _f: extern "C" fn(i64) -> f64 = nsl_runtime::tensor::nsl_tensor_l2_norm;
}

#[test]
fn weight_norm_is_init_then_update_round_trips() {
    let _g = serial_lock();
    let path = "m.l0.w";
    let b = path.as_bytes();
    unsafe {
        nsl_health_record_weight_norm(b.as_ptr(), b.len(), 100.0, true);
        nsl_health_record_weight_norm(b.as_ptr(), b.len(), 110.0, false);
    }
    let json = flush_to_string();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let pct = v["per_tensor_weight_pct_delta"]["m.l0.w"].as_f64().unwrap();
    assert!((pct - 10.0).abs() < 1e-6, "expected +10%, got {}", pct);
}

#[test]
fn getters_return_finite_values_after_recording() {
    let _g = serial_lock();
    nsl_health_record_loss(2.0, 1);
    nsl_health_record_loss(2.5, 2);
    let ema = nsl_health_get_loss_ema();
    assert!(ema > 0.0, "ema should be > 0 after recording, got {}", ema);

    // grad_norm_total is sticky from earlier tests; only assert non-negative
    let g = nsl_health_get_grad_norm_total();
    assert!(g >= 0.0, "grad_norm_total should be >= 0, got {}", g);

    let nan = nsl_health_get_nan_inf_count_window();
    assert!(nan >= 0, "nan count should be >= 0, got {}", nan);

    // loss_ema_slope is finite (could be 0.0 with too few samples but never NaN)
    let slope = nsl_health_get_loss_ema_slope();
    assert!(slope.is_finite(), "slope should be finite, got {}", slope);
}
