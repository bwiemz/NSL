//! M35: FP8 scale management, cast operations, and matmul FFI.

use std::cell::RefCell;
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::AtomicI64;

use crate::memory::checked_alloc;
use crate::tensor::NslTensor;

// ---------------------------------------------------------------------------
// FP8 constants
// ---------------------------------------------------------------------------

/// Maximum representable value for E4M3 format.
pub const FP8E4M3_MAX: f32 = 448.0;
/// Maximum representable value for E5M2 format.
pub const FP8E5M2_MAX: f32 = 57344.0;

/// FP8 format identifier for FFI.
pub const FP8_FORMAT_E4M3: i64 = 0;
pub const FP8_FORMAT_E5M2: i64 = 1;

// ---------------------------------------------------------------------------
// Thread-local scale table
// ---------------------------------------------------------------------------

thread_local! {
    /// Per-tensor scale factors, keyed by tensor pointer (as i64).
    static FP8_SCALES: RefCell<HashMap<i64, f32>> = RefCell::new(HashMap::new());
}

/// Register scale for an FP8 tensor.
pub fn set_fp8_scale(tensor_ptr: i64, scale: f32) {
    FP8_SCALES.with(|s| s.borrow_mut().insert(tensor_ptr, scale));
}

/// Retrieve scale (returns 1.0 if unregistered — safe default).
pub fn get_fp8_scale(tensor_ptr: i64) -> f32 {
    FP8_SCALES.with(|s| *s.borrow().get(&tensor_ptr).unwrap_or(&1.0))
}

/// Remove scale entry (on tensor free).
pub fn remove_fp8_scale(tensor_ptr: i64) {
    FP8_SCALES.with(|s| s.borrow_mut().remove(&tensor_ptr));
}

// ---------------------------------------------------------------------------
// FP8 quantize / dequantize helpers
// ---------------------------------------------------------------------------

/// Compute optimal scale factor: max(abs(tensor)) / fp8_max.
pub fn compute_scale(data: &[f64], fp8_format: i64) -> f64 {
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f64, f64::max);
    let fp8_max = match fp8_format {
        FP8_FORMAT_E4M3 => FP8E4M3_MAX as f64,
        FP8_FORMAT_E5M2 => FP8E5M2_MAX as f64,
        _ => FP8E4M3_MAX as f64,
    };
    if max_abs == 0.0 {
        1.0
    } else {
        max_abs / fp8_max
    }
}

fn compute_scale_f32(data: &[f32], fp8_format: i64) -> f64 {
    let max_abs = data.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let fp8_max = match fp8_format {
        FP8_FORMAT_E4M3 => FP8E4M3_MAX,
        FP8_FORMAT_E5M2 => FP8E5M2_MAX,
        _ => FP8E4M3_MAX,
    };
    if max_abs == 0.0 {
        1.0
    } else {
        (max_abs / fp8_max) as f64
    }
}

/// Quantize a single f64 value to FP8 (simulated as clamped+scaled f64).
pub fn quantize_fp8(value: f64, scale: f64, fp8_format: i64) -> f64 {
    if scale == 0.0 {
        return 0.0;
    }
    let fp8_max = match fp8_format {
        FP8_FORMAT_E4M3 => FP8E4M3_MAX as f64,
        FP8_FORMAT_E5M2 => FP8E5M2_MAX as f64,
        _ => FP8E4M3_MAX as f64,
    };
    let scaled = value / scale;
    let clamped = scaled.clamp(-fp8_max, fp8_max);
    let precision = match fp8_format {
        FP8_FORMAT_E4M3 => 0.125,
        FP8_FORMAT_E5M2 => 0.5,
        _ => 0.125,
    };
    (clamped / precision).round() * precision
}

/// Dequantize a simulated FP8 value back to f64.
pub fn dequantize_fp8(fp8_value: f64, scale: f64) -> f64 {
    fp8_value * scale
}

// ---------------------------------------------------------------------------
// FFI: FP8 cast
// ---------------------------------------------------------------------------

/// Cast a tensor to FP8 with given scale. If scale=0.0, auto-compute.
/// Returns a new tensor pointer with FP8-quantized-then-dequantized values.
#[no_mangle]
pub extern "C" fn nsl_fp8_cast(tensor_ptr: i64, target_dtype: i64, scale: f64) -> i64 {
    let t = unsafe { &*(tensor_ptr as *const NslTensor) };
    let len = t.len as usize;

    // Read source data as f32 (default runtime dtype)
    let src: Vec<f64> = if t.dtype == 1 {
        let data = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
        data.iter().map(|&v| v as f64).collect()
    } else {
        let data = unsafe { std::slice::from_raw_parts(t.data as *const f64, len) };
        data.to_vec()
    };

    let actual_scale = if scale == 0.0 {
        compute_scale(&src, target_dtype)
    } else {
        scale
    };

    // Quantize to FP8 and dequantize back — simulates FP8 precision loss
    // Output as f32 to match runtime convention
    let result_data = checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
    for (i, &v) in src.iter().enumerate() {
        let fp8 = quantize_fp8(v, actual_scale, target_dtype);
        let deq = dequantize_fp8(fp8, actual_scale);
        unsafe { *result_data.add(i) = deq as f32 };
    }

    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);

    let out = Box::new(NslTensor {
        data: result_data as *mut c_void,
        shape,
        strides,
        ndim: t.ndim,
        len: t.len,
        refcount: AtomicI64::new(1),
        device: t.device,
        dtype: 1, // f32 on CPU
        owns_data: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    // Register scale for this tensor
    set_fp8_scale(out_ptr, actual_scale as f32);

    out_ptr
}

/// Compute optimal scale factor for FP8 conversion.
#[no_mangle]
pub extern "C" fn nsl_fp8_compute_scale(tensor_ptr: i64, fp8_dtype: i64) -> f64 {
    let t = unsafe { &*(tensor_ptr as *const NslTensor) };
    let len = t.len as usize;
    if t.dtype == 1 {
        let data = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
        compute_scale_f32(data, fp8_dtype)
    } else {
        let data = unsafe { std::slice::from_raw_parts(t.data as *const f64, len) };
        compute_scale(data, fp8_dtype)
    }
}

// ---------------------------------------------------------------------------
// FFI: FP8 matmul (CPU fallback)
// ---------------------------------------------------------------------------

/// CPU fallback for FP8 matmul: A[M,K] @ B[K,N] -> C[M,N]
/// Accumulates in f64 (simulating f32 accumulation on GPU).
pub fn fp8_matmul_cpu(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut c = vec![0.0f64; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += a[i * k + p] * b[p * n + j];
            }
            c[i * n + j] = acc;
        }
    }
    c
}

/// FP8 matmul FFI: both inputs are FP8-cast tensors. Output is f32 on CPU.
/// scale_a and scale_b are passed but on CPU, scale is already applied at cast time.
#[no_mangle]
pub extern "C" fn nsl_fp8_matmul(
    a_ptr: i64,
    b_ptr: i64,
    _scale_a: f64,
    _scale_b: f64,
) -> i64 {
    // On CPU, tensors are already dequantized f32. Delegate to standard matmul.
    crate::tensor::nsl_tensor_matmul(a_ptr, b_ptr)
}

/// Update calibration running max (EMA). Returns updated running_max as f64.
/// Deferred to M35b — stub only.
#[no_mangle]
pub extern "C" fn nsl_fp8_update_calibration(
    _tensor_ptr: i64,
    _running_max_ptr: i64,
    _momentum: f64,
) -> f64 {
    0.0
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale_table_set_get() {
        set_fp8_scale(42, 0.5);
        assert_eq!(get_fp8_scale(42), 0.5);
        assert_eq!(get_fp8_scale(999), 1.0); // default
        remove_fp8_scale(42);
        assert_eq!(get_fp8_scale(42), 1.0); // removed
    }

    #[test]
    fn test_compute_scale_e4m3() {
        let data = vec![100.0, -200.0, 50.0];
        let scale = compute_scale(&data, FP8_FORMAT_E4M3);
        assert!((scale - 200.0 / 448.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_scale_e5m2() {
        let data = vec![100.0, -200.0, 50.0];
        let scale = compute_scale(&data, FP8_FORMAT_E5M2);
        assert!((scale - 200.0 / 57344.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_scale_zero() {
        let data = vec![0.0, 0.0, 0.0];
        let scale = compute_scale(&data, FP8_FORMAT_E4M3);
        assert_eq!(scale, 1.0);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() {
        let value = 1.5;
        let scale = 0.01;
        let fp8 = quantize_fp8(value, scale, FP8_FORMAT_E4M3);
        let recovered = dequantize_fp8(fp8, scale);
        assert!((recovered - value).abs() < scale * 0.125 + 1e-10);
    }

    #[test]
    fn test_clamping() {
        let value = 1000.0;
        let scale = 1.0;
        let fp8 = quantize_fp8(value, scale, FP8_FORMAT_E4M3);
        assert_eq!(fp8, 448.0); // clamped to E4M3 max
    }

    #[test]
    fn test_fp8_cast_ffi_auto_scale() {
        let data = vec![1.0f64, 2.0, -3.0, 4.0];
        let scale = compute_scale(&data, FP8_FORMAT_E4M3);

        // Quantize each element and verify round-trip
        for &v in &data {
            let quant = quantize_fp8(v, scale, FP8_FORMAT_E4M3);
            let recovered = dequantize_fp8(quant, scale);
            let rel_error = if v.abs() > 1e-10 {
                (recovered - v).abs() / v.abs()
            } else {
                0.0
            };
            assert!(
                rel_error < 0.01,
                "FP8 E4M3 relative error {} too high for value {}",
                rel_error,
                v
            );
        }
    }

    #[test]
    fn test_fp8_matmul_cpu() {
        // 2x2 matmul: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![5.0, 6.0, 7.0, 8.0];

        let scale_a = compute_scale(&a, FP8_FORMAT_E4M3);
        let scale_b = compute_scale(&b, FP8_FORMAT_E4M3);

        let a_fp8: Vec<f64> = a
            .iter()
            .map(|&v| dequantize_fp8(quantize_fp8(v, scale_a, FP8_FORMAT_E4M3), scale_a))
            .collect();
        let b_fp8: Vec<f64> = b
            .iter()
            .map(|&v| dequantize_fp8(quantize_fp8(v, scale_b, FP8_FORMAT_E4M3), scale_b))
            .collect();

        let result = fp8_matmul_cpu(&a_fp8, &b_fp8, 2, 2, 2);

        assert!((result[0] - 19.0).abs() < 1.0);
        assert!((result[1] - 22.0).abs() < 1.0);
        assert!((result[2] - 43.0).abs() < 2.0);
        assert!((result[3] - 50.0).abs() < 2.0);
    }
}
