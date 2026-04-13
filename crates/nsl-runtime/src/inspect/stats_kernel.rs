//! On-device reduction computing six tensor statistics.
//!
//! Writes `[mean, std, min, max, nan_count, inf_count]` into `out_buf`.
//!
//! Phase 5 ship-fast: host-side loop over CPU-resident tensors. GPU tensors
//! currently return an error (rc=3); a follow-up PR will fuse into a single
//! PTX kernel launched on the inspect stream.

use crate::tensor::{NslTensor, DTYPE_F32, DTYPE_F64};

/// # Safety
/// `out_buf` must point to six writable `f64` slots. `t` must be a valid
/// `NslTensor` handle (live magic marker) or zero.
#[no_mangle]
pub extern "C" fn nsl_tensor_stats(t: i64, out_buf: *mut f64) -> i32 {
    if out_buf.is_null() {
        return 1;
    }
    if t == 0 {
        return 2;
    }
    let tensor = NslTensor::from_ptr(t);

    match compute_stats(tensor) {
        Ok(stats) => {
            unsafe {
                out_buf.add(0).write(stats[0]); // mean
                out_buf.add(1).write(stats[1]); // std
                out_buf.add(2).write(stats[2]); // min
                out_buf.add(3).write(stats[3]); // max
                out_buf.add(4).write(stats[4]); // nan_count
                out_buf.add(5).write(stats[5]); // inf_count
            }
            0
        }
        Err(_) => 3,
    }
}

fn compute_stats(t: &NslTensor) -> Result<[f64; 6], &'static str> {
    let host_data = read_host_f64(t)?;
    if host_data.is_empty() {
        return Ok([0.0; 6]);
    }

    let total = host_data.len() as u64;
    let mut sum = 0.0f64;
    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    let mut nan_count: u64 = 0;
    let mut inf_count: u64 = 0;
    for &v in &host_data {
        if v.is_nan() {
            nan_count += 1;
            continue;
        }
        if v.is_infinite() {
            inf_count += 1;
            continue;
        }
        sum += v;
        if v < min {
            min = v;
        }
        if v > max {
            max = v;
        }
    }
    let finite_count = total - nan_count - inf_count;
    let mean = if finite_count == 0 {
        0.0
    } else {
        sum / finite_count as f64
    };

    let mut var_sum = 0.0f64;
    for &v in &host_data {
        if v.is_finite() {
            let d = v - mean;
            var_sum += d * d;
        }
    }
    let std = if finite_count <= 1 {
        0.0
    } else {
        (var_sum / (finite_count as f64 - 1.0)).sqrt()
    };

    if min == f64::INFINITY {
        min = 0.0;
    }
    if max == f64::NEG_INFINITY {
        max = 0.0;
    }

    Ok([mean, std, min, max, nan_count as f64, inf_count as f64])
}

/// Read all tensor elements as `f64`, converting from the underlying dtype.
///
/// Supports CPU-resident f64 / f32 tensors. GPU tensors and non-float dtypes
/// return `Err` for now.
pub(crate) fn read_host_f64(t: &NslTensor) -> Result<Vec<f64>, &'static str> {
    if t.device > 0 {
        return Err("GPU tensor host-read not implemented (Phase 5 ship-fast)");
    }
    if t.data.is_null() {
        return Err("tensor data pointer is null");
    }
    let n = t.len as usize;
    match t.dtype {
        DTYPE_F64 => {
            let src = t.data as *const f64;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(unsafe { *src.add(i) });
            }
            Ok(out)
        }
        DTYPE_F32 => {
            let src = t.data as *const f32;
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                out.push(unsafe { *src.add(i) as f64 });
            }
            Ok(out)
        }
        _ => Err("unsupported dtype for inspect stats"),
    }
}
