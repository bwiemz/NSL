//! Inspector FFI hooks emitted by codegen at `@inspect` sites.

use once_cell::sync::Lazy;
use std::path::PathBuf;
use std::sync::Mutex;

use super::format;

static INSPECT_DIR: Lazy<Mutex<PathBuf>> = Lazy::new(|| Mutex::new(PathBuf::from(".nsl-inspect")));

/// Set the directory into which inspect artifacts are written.
///
/// # Safety
/// Caller guarantees `(path_ptr, path_len)` points to valid UTF-8 bytes.
#[no_mangle]
pub unsafe extern "C" fn nsl_inspect_set_dir(path_ptr: *const u8, path_len: usize) {
    if path_ptr.is_null() {
        return;
    }
    let bytes = std::slice::from_raw_parts(path_ptr, path_len);
    if let Ok(s) = std::str::from_utf8(bytes) {
        *INSPECT_DIR.lock().unwrap() = PathBuf::from(s);
    }
}

/// Record the six-stat summary for a tensor at a given step.
///
/// # Safety
/// - `stats_buf_ptr` must point to 6 readable `f64` values.
/// - `(name_ptr, name_len)` must be valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn nsl_inspect_record_stats(
    stats_buf_ptr: *const f64,
    step: u64,
    name_ptr: *const u8,
    name_len: usize,
) -> i32 {
    if stats_buf_ptr.is_null() || name_ptr.is_null() {
        return 1;
    }
    let stats = std::slice::from_raw_parts(stats_buf_ptr, 6);
    let name = match std::str::from_utf8(std::slice::from_raw_parts(name_ptr, name_len)) {
        Ok(s) => s,
        Err(_) => return 2,
    };
    let dir = INSPECT_DIR.lock().unwrap().clone();
    if std::fs::create_dir_all(&dir).is_err() {
        return 3;
    }
    let path = dir.join(format!("step_{}_{}.stats.bin", step, name));
    match format::write_stats(&path, step, name, stats) {
        Ok(_) => 0,
        Err(_) => 4,
    }
}

/// Dump a full tensor (raw bytes + stats) at a given step.
///
/// # Safety
/// - `tensor_handle` must be a valid `NslTensor` pointer.
/// - `(name_ptr, name_len)` must be valid UTF-8.
#[no_mangle]
pub unsafe extern "C" fn nsl_inspect_dump_full(
    tensor_handle: i64,
    step: u64,
    name_ptr: *const u8,
    name_len: usize,
) -> i32 {
    if tensor_handle == 0 || name_ptr.is_null() {
        return 1;
    }
    let name = match std::str::from_utf8(std::slice::from_raw_parts(name_ptr, name_len)) {
        Ok(s) => s,
        Err(_) => return 2,
    };
    let dir = INSPECT_DIR.lock().unwrap().clone();
    if std::fs::create_dir_all(&dir).is_err() {
        return 3;
    }

    let tensor = crate::tensor::NslTensor::from_ptr(tensor_handle);

    let mut stats_buf = [0.0f64; 6];
    if super::stats_kernel::nsl_tensor_stats(tensor_handle, stats_buf.as_mut_ptr()) != 0 {
        return 4;
    }

    let dtype_str = format!("dtype_{}", tensor.dtype);
    let shape: Vec<i64> = crate::tensor::get_shape_vec(tensor);

    // Raw host bytes: supported for CPU tensors only in Phase 5 ship-fast.
    let host_bytes: Vec<u8> = if tensor.device == 0 && !tensor.data.is_null() {
        let elem = crate::tensor::dtype_element_size(tensor.dtype);
        let byte_len = (tensor.len as usize).saturating_mul(elem);
        let mut v = vec![0u8; byte_len];
        std::ptr::copy_nonoverlapping(tensor.data as *const u8, v.as_mut_ptr(), byte_len);
        v
    } else {
        // GPU path or null data: skip raw bytes, record an empty payload so the
        // header/stats are still preserved. A follow-up PR will wire the
        // D2H copy through the dedicated inspect stream.
        Vec::new()
    };

    let header = format::FullHeader {
        step,
        tensor_name: name.into(),
        kind: "full".into(),
        dtype: dtype_str,
        shape,
        stats: format::StatsHeader {
            step,
            tensor_name: name.into(),
            kind: "full".into(),
            mean: stats_buf[0],
            std: stats_buf[1],
            min: stats_buf[2],
            max: stats_buf[3],
            nan_count: stats_buf[4] as u64,
            inf_count: stats_buf[5] as u64,
        },
    };
    let path = dir.join(format!("step_{}_{}.tensor.bin", step, name));
    match format::write_full(&path, &header, &host_bytes) {
        Ok(_) => 0,
        Err(_) => 6,
    }
}
