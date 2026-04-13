//! FFI hook tests for the health monitor.
//!
//! NOTE: These tests share a single global `Mutex<HealthCollector>` (see
//! `health::ffi::COLLECTOR`). They MUST be run with `--test-threads=1` to
//! avoid cross-test races on the shared collector state.

use nsl_runtime::health::ffi::{
    nsl_health_flush_snapshot, nsl_health_record_grad_norm, nsl_health_record_loss,
    nsl_health_record_weight_norm,
};

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
    nsl_health_record_loss(2.5, 7);
    let json = flush_to_string();
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert_eq!(v["step"], 7);
    assert!((v["loss"].as_f64().unwrap() - 2.5).abs() < 1e-6);
}

#[test]
fn record_grad_norm_emits_layer_entry() {
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
    let rc = unsafe { nsl_health_flush_snapshot(std::ptr::null(), 0) };
    assert_eq!(rc, 0);
}

#[test]
fn flush_invalid_utf8_returns_2() {
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
