//! On-device reduction computing six tensor statistics.
//!
//! Writes `[mean, std, min, max, nan_count, inf_count]` into `out_buf`.
//!
//! Host-side loop over CPU-resident tensors. GPU-resident tensors are staged
//! device-to-host (`nsl_tensor_contiguous` -> `nsl_tensor_to_device(_, 0)`)
//! and then run through the same CPU reduction, so both devices share one
//! set of NaN/Inf-skipping, sample-std semantics. Without the `cuda` feature
//! the GPU path refuses (rc=3) with a zeroed buffer.

use crate::tensor::{NslTensor, DTYPE_F32, DTYPE_F64};

/// # Safety
/// `out_buf` must point to six writable `f64` slots. `t` must be a valid
/// `NslTensor` handle (live magic marker) or zero.
#[no_mangle]
pub extern "C" fn nsl_tensor_stats(t: i64, out_buf: *mut f64) -> i32 {
    if out_buf.is_null() {
        return 1;
    }
    // Zero-fill every slot up front: codegen at `@inspect` sites ignores the
    // return code, so any error path below must leave zeroed records in the
    // .stats.bin file rather than uninitialized stack garbage.
    unsafe {
        std::ptr::write_bytes(out_buf, 0, 6);
    }
    if t == 0 {
        return 2;
    }
    let tensor = NslTensor::from_ptr(t);

    match compute_stats(t, tensor) {
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

fn compute_stats(handle: i64, t: &NslTensor) -> Result<[f64; 6], &'static str> {
    let host_data = read_host_f64(handle, t)?;
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
/// CPU-resident f64 / f32 tensors are read directly. GPU-resident tensors are
/// staged to the host first (see `read_gpu_f64`); without the `cuda` feature
/// they refuse with `Err`. Non-float dtypes return `Err`.
pub(crate) fn read_host_f64(handle: i64, t: &NslTensor) -> Result<Vec<f64>, &'static str> {
    if t.device > 0 {
        #[cfg(feature = "cuda")]
        {
            return read_gpu_f64(handle);
        }
        #[cfg(not(feature = "cuda"))]
        {
            let _ = handle;
            return Err("GPU tensor stats require the `cuda` feature (refusing rather than reading device memory)");
        }
    }
    read_cpu_f64(t)
}

/// Stage a GPU-resident tensor to the host and read it as `f64`.
///
/// `nsl_tensor_contiguous` and `nsl_tensor_to_device` BOTH return owned refs
/// (project invariant) that must be released here via `nsl_tensor_free`.
/// `nsl_tensor_to_device(_, 0)` context-syncs and widens f32 -> f64 (runtime
/// invariant: CPU=f64, GPU=f32), so `read_cpu_f64` sees a plain CPU tensor.
#[cfg(feature = "cuda")]
fn read_gpu_f64(handle: i64) -> Result<Vec<f64>, &'static str> {
    crate::cuda::inner::ensure_context();
    let contig = crate::tensor::nsl_tensor_contiguous(handle);
    if contig == 0 {
        return Err("nsl_tensor_contiguous failed on GPU tensor for inspect stats");
    }
    let host = crate::tensor::nsl_tensor_to_device(contig, 0);
    if host == 0 {
        crate::tensor::nsl_tensor_free(contig);
        return Err("device-to-host staging failed on GPU tensor for inspect stats");
    }
    let result = read_cpu_f64(NslTensor::from_ptr(host));
    crate::tensor::nsl_tensor_free(host);
    crate::tensor::nsl_tensor_free(contig);
    result
}

/// Read a CPU-resident f64 / f32 tensor as `f64`.
fn read_cpu_f64(t: &NslTensor) -> Result<Vec<f64>, &'static str> {
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
