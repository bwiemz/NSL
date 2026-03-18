//! M35: AWQ (Activation-Aware Weight Quantization) 4-bit runtime.

use std::ffi::c_void;

use crate::memory::checked_alloc;
use crate::tensor::NslTensor;

// ---------------------------------------------------------------------------
// 4-bit pack/unpack
// ---------------------------------------------------------------------------

/// Pack two 4-bit values into one byte (low nibble first).
pub fn pack_int4(low: u8, high: u8) -> u8 {
    (low & 0x0F) | ((high & 0x0F) << 4)
}

/// Unpack a byte into two 4-bit values (low, high).
pub fn unpack_int4(packed: u8) -> (u8, u8) {
    (packed & 0x0F, (packed >> 4) & 0x0F)
}

// ---------------------------------------------------------------------------
// AWQ packed weight struct
// ---------------------------------------------------------------------------

/// AWQ packed weight representation.
/// Two 4-bit values packed per byte, low nibble first.
#[repr(C)]
pub struct AwqPackedWeight {
    /// Packed 4-bit data [K/2 * N bytes for K*N weights]
    pub data: *mut u8,
    /// Per-group scale factors (stored as f64 on CPU)
    pub scales: *mut f64,
    /// Per-group zero points (stored as f64 on CPU)
    pub zeros: *mut f64,
    /// Original weight dimensions [K, N]
    pub k: i64,
    pub n: i64,
    /// Group size (typically 128)
    pub group_size: i64,
    /// Number of groups per column: ceil(K / group_size)
    pub num_groups: i64,
    pub refcount: i64,
}

// ---------------------------------------------------------------------------
// Group quantization helpers
// ---------------------------------------------------------------------------

/// Compute per-group scale and zero for asymmetric quantization.
/// Returns (scale, zero_point) where: quantized = round((value - zero) / scale)
pub fn compute_group_params(group: &[f64]) -> (f64, f64) {
    let min_val = group.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_val = group.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    if (max_val - min_val).abs() < 1e-10 {
        return (1.0, min_val);
    }

    let scale = (max_val - min_val) / 15.0; // 4-bit: 0..15
    let zero = min_val;
    (scale, zero)
}

/// Quantize a single value to 4-bit given scale and zero.
pub fn quantize_int4(value: f64, scale: f64, zero: f64) -> u8 {
    let q = ((value - zero) / scale).round() as i32;
    q.clamp(0, 15) as u8
}

/// Dequantize a 4-bit value to f64.
pub fn dequantize_int4(quantized: u8, scale: f64, zero: f64) -> f64 {
    (quantized as f64) * scale + zero
}

// ---------------------------------------------------------------------------
// Nibble read/write helpers
// ---------------------------------------------------------------------------

fn read_nibble(data: &[u8], flat_idx: usize) -> u8 {
    let byte_idx = flat_idx / 2;
    if flat_idx.is_multiple_of(2) {
        data[byte_idx] & 0x0F
    } else {
        (data[byte_idx] >> 4) & 0x0F
    }
}

fn write_nibble(data: &mut [u8], flat_idx: usize, val: u8) {
    let byte_idx = flat_idx / 2;
    if flat_idx.is_multiple_of(2) {
        data[byte_idx] = (data[byte_idx] & 0xF0) | (val & 0x0F);
    } else {
        data[byte_idx] = (data[byte_idx] & 0x0F) | ((val & 0x0F) << 4);
    }
}

// ---------------------------------------------------------------------------
// AWQ quantize / dequantize
// ---------------------------------------------------------------------------

/// Quantize a weight matrix [K, N] to AWQ 4-bit packed format.
pub fn awq_quantize_cpu(weights: &[f64], k: usize, n: usize, group_size: usize) -> AwqPackedWeight {
    let num_groups = k.div_ceil(group_size);
    let packed_bytes = (k * n).div_ceil(2);

    let mut data = vec![0u8; packed_bytes];
    let mut scales = vec![0.0f64; num_groups * n];
    let mut zeros = vec![0.0f64; num_groups * n];

    for col in 0..n {
        for g in 0..num_groups {
            let start = g * group_size;
            let end = (start + group_size).min(k);

            let group: Vec<f64> = (start..end).map(|row| weights[row * n + col]).collect();
            let (scale, zero) = compute_group_params(&group);

            scales[g * n + col] = scale;
            zeros[g * n + col] = zero;

            for row in start..end {
                let q = quantize_int4(weights[row * n + col], scale, zero);
                write_nibble(&mut data, row * n + col, q);
            }
        }
    }

    let data = data.into_boxed_slice();
    let scales = scales.into_boxed_slice();
    let zeros = zeros.into_boxed_slice();

    AwqPackedWeight {
        data: Box::into_raw(data) as *mut u8,
        scales: Box::into_raw(scales) as *mut f64,
        zeros: Box::into_raw(zeros) as *mut f64,
        k: k as i64,
        n: n as i64,
        group_size: group_size as i64,
        num_groups: num_groups as i64,
        refcount: 1,
    }
}

/// Dequantize AWQ packed weights back to f64 for verification.
pub fn awq_dequantize_cpu(packed: &AwqPackedWeight) -> Vec<f64> {
    let k = packed.k as usize;
    let n = packed.n as usize;
    let group_size = packed.group_size as usize;
    let mut result = vec![0.0f64; k * n];

    let data = unsafe { std::slice::from_raw_parts(packed.data, (k * n).div_ceil(2)) };
    let scales = unsafe { std::slice::from_raw_parts(packed.scales, packed.num_groups as usize * n) };
    let zeros = unsafe { std::slice::from_raw_parts(packed.zeros, packed.num_groups as usize * n) };

    for col in 0..n {
        for row in 0..k {
            let g = row / group_size;
            let q = read_nibble(data, row * n + col);
            result[row * n + col] = dequantize_int4(q, scales[g * n + col], zeros[g * n + col]);
        }
    }
    result
}

/// Free an AWQ packed weight's internal buffers.
///
/// # Safety
/// `packed` must point to a valid `AwqPackedWeight` whose buffers were allocated via `Box`.
pub unsafe fn awq_free_packed(packed: &AwqPackedWeight) {
    let k = packed.k as usize;
    let n = packed.n as usize;
    let num_groups = packed.num_groups as usize;
    let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(packed.data, (k * n).div_ceil(2)));
    let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(packed.scales, num_groups * n));
    let _ = Box::from_raw(std::ptr::slice_from_raw_parts_mut(packed.zeros, num_groups * n));
}

// ---------------------------------------------------------------------------
// AWQ matmul (CPU dequant-on-the-fly)
// ---------------------------------------------------------------------------

/// Simple CPU matmul: A[M,K] @ B[K,N] -> C[M,N] (used in tests for reference)
#[cfg(test)]
fn matmul_cpu(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
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

/// AWQ dequantize-matmul: input[M,K] @ packed_weight[K,N] -> output[M,N]
/// On CPU, dequantizes on-the-fly during matmul (simulating in-register dequant).
pub fn awq_matmul_cpu(input: &[f64], packed: &AwqPackedWeight, m: usize) -> Vec<f64> {
    let k = packed.k as usize;
    let n = packed.n as usize;
    let group_size = packed.group_size as usize;

    let data = unsafe { std::slice::from_raw_parts(packed.data, (k * n).div_ceil(2)) };
    let scales = unsafe { std::slice::from_raw_parts(packed.scales, packed.num_groups as usize * n) };
    let zeros = unsafe { std::slice::from_raw_parts(packed.zeros, packed.num_groups as usize * n) };

    let mut output = vec![0.0f64; m * n];

    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                let g = p / group_size;
                let q = read_nibble(data, p * n + j);
                let w = dequantize_int4(q, scales[g * n + j], zeros[g * n + j]);
                acc += input[i * k + p] * w;
            }
            output[i * n + j] = acc;
        }
    }
    output
}

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

/// Quantize weight tensor to AWQ4 packed format.
#[no_mangle]
pub extern "C" fn nsl_awq_quantize(
    weight_ptr: i64,
    group_size: i64,
    _calibration_ptr: i64,
) -> i64 {
    let t = unsafe { &*(weight_ptr as *const NslTensor) };
    assert!(t.ndim >= 2, "nsl_awq_quantize requires 2D weight tensor (got {}D)", t.ndim);
    let len = t.len as usize;

    let data: Vec<f64> = if t.dtype == 1 {
        let raw = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
        raw.iter().map(|&v| v as f64).collect()
    } else {
        let raw = unsafe { std::slice::from_raw_parts(t.data as *const f64, len) };
        raw.to_vec()
    };

    let k = unsafe { *t.shape } as usize;
    let n = unsafe { *t.shape.add(1) } as usize;

    let packed = awq_quantize_cpu(&data, k, n, group_size as usize);
    Box::into_raw(Box::new(packed)) as i64
}

/// AWQ dequantize-in-GEMM matmul.
#[no_mangle]
pub extern "C" fn nsl_awq_matmul(
    input_ptr: i64,
    packed_ptr: i64,
    _group_size: i64,
) -> i64 {
    let input_t = unsafe { &*(input_ptr as *const NslTensor) };
    let packed = unsafe { &*(packed_ptr as *const AwqPackedWeight) };

    let m = unsafe { *input_t.shape } as usize;
    let len = input_t.len as usize;

    let input: Vec<f64> = if input_t.dtype == 1 {
        let raw = unsafe { std::slice::from_raw_parts(input_t.data as *const f32, len) };
        raw.iter().map(|&v| v as f64).collect()
    } else {
        let raw = unsafe { std::slice::from_raw_parts(input_t.data as *const f64, len) };
        raw.to_vec()
    };

    let result = awq_matmul_cpu(&input, packed, m);

    // Create output tensor [M, N] as f32
    let n = packed.n as usize;
    let out_len = m * n;
    let out_data = checked_alloc(out_len * std::mem::size_of::<f32>()) as *mut f32;
    for (i, &v) in result.iter().enumerate() {
        unsafe { *out_data.add(i) = v as f32 };
    }

    let shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *shape = m as i64;
        *shape.add(1) = n as i64;
    }
    let strides = NslTensor::compute_strides(shape, 2);

    let tensor = Box::new(NslTensor {
        data: out_data as *mut c_void,
        shape,
        strides,
        ndim: 2,
        len: out_len as i64,
        refcount: 1,
        device: 0,
        dtype: 1,
        owns_data: 1,
    });
    Box::into_raw(tensor) as i64
}

/// Free an AWQ packed weight.
#[no_mangle]
pub extern "C" fn nsl_awq_free(packed_ptr: i64) {
    if packed_ptr == 0 {
        return;
    }
    let packed = unsafe { Box::from_raw(packed_ptr as *mut AwqPackedWeight) };
    unsafe { awq_free_packed(&packed) };
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_int4() {
        let packed = pack_int4(5, 12);
        let (low, high) = unpack_int4(packed);
        assert_eq!(low, 5);
        assert_eq!(high, 12);
    }

    #[test]
    fn test_pack_unpack_boundary() {
        let packed = pack_int4(0, 15);
        let (low, high) = unpack_int4(packed);
        assert_eq!(low, 0);
        assert_eq!(high, 15);
    }

    #[test]
    fn test_group_params() {
        let group = vec![0.0, 1.0, 2.0, 3.0];
        let (scale, zero) = compute_group_params(&group);
        assert!((scale - 0.2).abs() < 1e-10); // (3-0)/15
        assert_eq!(zero, 0.0);
    }

    #[test]
    fn test_quantize_dequantize_int4() {
        let value = 1.5;
        let (scale, zero) = (0.2, 0.0);
        let q = quantize_int4(value, scale, zero);
        let recovered = dequantize_int4(q, scale, zero);
        assert!((recovered - value).abs() < scale);
    }

    #[test]
    fn test_quantize_clamp() {
        let q = quantize_int4(100.0, 1.0, 0.0);
        assert_eq!(q, 15);
    }

    #[test]
    fn test_awq_quantize_small_matrix() {
        let weights: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let packed = awq_quantize_cpu(&weights, 4, 4, 4);

        assert_eq!(packed.k, 4);
        assert_eq!(packed.n, 4);
        assert_eq!(packed.group_size, 4);
        assert_eq!(packed.num_groups, 1);

        let recovered = awq_dequantize_cpu(&packed);
        for (i, (&orig, &rec)) in weights.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 0.15,
                "AWQ error too high at index {}: orig={}, recovered={}",
                i, orig, rec
            );
        }

        unsafe { awq_free_packed(&packed) };
    }

    #[test]
    fn test_awq_matmul_cpu() {
        let input = vec![1.0f64, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weights = vec![
            0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2,
        ];

        let reference = matmul_cpu(&input, &weights, 2, 4, 3);
        let packed = awq_quantize_cpu(&weights, 4, 3, 4);
        let awq_result = awq_matmul_cpu(&input, &packed, 2);

        for (i, (&r, &a)) in reference.iter().zip(awq_result.iter()).enumerate() {
            let tol = r.abs() * 0.1 + 0.1;
            assert!(
                (r - a).abs() < tol,
                "AWQ matmul error at {}: ref={}, awq={}",
                i, r, a
            );
        }

        unsafe { awq_free_packed(&packed) };
    }
}
