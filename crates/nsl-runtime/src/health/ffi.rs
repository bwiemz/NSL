//! C-callable hooks emitted by codegen at training-step boundaries.
//!
//! Single global Mutex<HealthCollector> is fine today (NSL backward is
//! single-stream sequential). When CPDT lands and overlaps backward across
//! layers, the future fix is per-layer sharded collectors merged at snapshot
//! time. Don't refactor until profiling shows contention.

use super::collector::HealthCollector;
use once_cell::sync::Lazy;
use std::sync::Mutex;

static COLLECTOR: Lazy<Mutex<HealthCollector>> = Lazy::new(|| Mutex::new(HealthCollector::new()));

#[no_mangle]
pub extern "C" fn nsl_health_record_loss(value: f64, step: u64) {
    COLLECTOR.lock().unwrap().record_loss(step, value);
}

/// # Safety
/// Caller must guarantee (path_ptr, path_len) refers to valid UTF-8 bytes
/// they own for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn nsl_health_record_grad_norm(
    path_ptr: *const u8,
    path_len: usize,
    layer_idx: u32,
    norm: f64,
) {
    if path_ptr.is_null() {
        return;
    }
    let bytes = std::slice::from_raw_parts(path_ptr, path_len);
    if let Ok(path) = std::str::from_utf8(bytes) {
        COLLECTOR
            .lock()
            .unwrap()
            .record_grad_norm(path, layer_idx, norm);
    }
}

/// # Safety
/// Same as `nsl_health_record_grad_norm`.
#[no_mangle]
pub unsafe extern "C" fn nsl_health_record_weight_norm(
    path_ptr: *const u8,
    path_len: usize,
    norm: f64,
    is_init: bool,
) {
    if path_ptr.is_null() {
        return;
    }
    let bytes = std::slice::from_raw_parts(path_ptr, path_len);
    if let Ok(path) = std::str::from_utf8(bytes) {
        COLLECTOR
            .lock()
            .unwrap()
            .record_weight_norm(path, norm, is_init);
    }
}

/// Returns: 0 ok, 1 serde fail, 2 invalid UTF-8 path, 3 io fail.
///
/// # Safety
/// Caller must guarantee (path_ptr, path_len) refers to valid UTF-8 bytes
/// they own for the duration of the call, or path_ptr is null (prints JSON
/// to stdout instead).
#[no_mangle]
pub unsafe extern "C" fn nsl_health_flush_snapshot(
    path_ptr: *const u8,
    path_len: usize,
) -> i32 {
    let snap = COLLECTOR.lock().unwrap().snapshot();
    let json = match serde_json::to_string(&snap) {
        Ok(s) => s,
        Err(_) => return 1,
    };
    if path_ptr.is_null() {
        println!("{}", json);
        return 0;
    }
    let bytes = std::slice::from_raw_parts(path_ptr, path_len);
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return 2,
    };
    match std::fs::write(std::path::Path::new(s), json) {
        Ok(_) => 0,
        Err(_) => 3,
    }
}

#[no_mangle]
pub extern "C" fn nsl_health_set_flush_interval(n: u64) {
    COLLECTOR.lock().unwrap().set_flush_interval(n);
}

/// Most recent finite loss recorded this run (0.0 before the first record).
/// Powers the `loss` identifier in `@inspect` predicates — previously that
/// identifier lowered to a compile-time 0.0 constant, so `condition="loss > x"`
/// silently never fired.
#[no_mangle]
pub extern "C" fn nsl_health_get_last_loss() -> f64 {
    COLLECTOR.lock().unwrap().last_loss().unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn nsl_health_get_loss_ema() -> f64 {
    COLLECTOR.lock().unwrap().snapshot().loss_ema.unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn nsl_health_get_loss_ema_slope() -> f64 {
    COLLECTOR
        .lock()
        .unwrap()
        .snapshot()
        .loss_ema_slope
        .unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn nsl_health_get_grad_norm_total() -> f64 {
    COLLECTOR
        .lock()
        .unwrap()
        .snapshot()
        .grad_norm_total
        .unwrap_or(0.0)
}

#[no_mangle]
pub extern "C" fn nsl_health_get_nan_inf_count_window() -> i64 {
    COLLECTOR.lock().unwrap().snapshot().nan_inf_count_window as i64
}
