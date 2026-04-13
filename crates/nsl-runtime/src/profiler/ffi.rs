//! C-callable FFI hooks invoked by codegen-emitted kernel wrappers (Phase 2).
//! Uses a global Mutex<Option<Collector>> — single-threaded use assumed; the
//! mutex is just for safe lazy init across the FFI boundary.

use once_cell::sync::Lazy;
use std::sync::Mutex;
use std::time::Instant;

use super::collector::{Collector, NanoClock};

static COLLECTOR: Lazy<Mutex<Option<Collector>>> = Lazy::new(|| Mutex::new(None));
static START: Lazy<Instant> = Lazy::new(Instant::now);

fn now_ns() -> u64 {
    START.elapsed().as_nanos() as u64
}

#[no_mangle]
pub extern "C" fn nsl_profile_kernel_begin(kernel_id: u32) {
    let t = now_ns();
    let mut guard = COLLECTOR.lock().unwrap();
    let c = guard.get_or_insert_with(|| Collector::new_with_clock(Box::new(NanoClock)));
    c.begin(kernel_id, t);
}

#[no_mangle]
pub extern "C" fn nsl_profile_kernel_end(kernel_id: u32) {
    let t = now_ns();
    let mut guard = COLLECTOR.lock().unwrap();
    let c = guard.get_or_insert_with(|| Collector::new_with_clock(Box::new(NanoClock)));
    c.end(kernel_id, t);
}

/// Flush aggregates as pretty-printed JSON to the given path.
/// Returns 0 on success, non-zero on error (bad path / IO failure).
/// path_ptr/path_len use UTF-8 bytes; caller is responsible for liveness.
///
/// # Safety
/// Caller must ensure (path_ptr, path_len) is a valid UTF-8 slice they own
/// for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn nsl_profile_flush(path_ptr: *const u8, path_len: usize) -> i32 {
    if path_ptr.is_null() {
        return 1;
    }
    let bytes = std::slice::from_raw_parts(path_ptr, path_len);
    let s = match std::str::from_utf8(bytes) {
        Ok(s) => s,
        Err(_) => return 2,
    };
    let path = std::path::Path::new(s);
    let mut guard = COLLECTOR.lock().unwrap();
    let c = guard.get_or_insert_with(|| Collector::new_with_clock(Box::new(NanoClock)));
    match c.flush_to(path) {
        Ok(_) => 0,
        Err(_) => 3,
    }
}
