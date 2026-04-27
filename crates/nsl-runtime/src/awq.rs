//! M35: AWQ (Activation-Aware Weight Quantization) 4-bit runtime.

use std::sync::atomic::AtomicI64;
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
    pub refcount: AtomicI64,
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

pub fn read_nibble(data: &[u8], flat_idx: usize) -> u8 {
    let byte_idx = flat_idx / 2;
    if flat_idx.is_multiple_of(2) {
        data[byte_idx] & 0x0F
    } else {
        (data[byte_idx] >> 4) & 0x0F
    }
}

pub fn write_nibble(data: &mut [u8], flat_idx: usize, val: u8) {
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
        refcount: AtomicI64::new(1),
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
    if weight_ptr == 0 { eprintln!("nsl_awq_quantize: null weight tensor"); return 0; }
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
    if input_ptr == 0 || packed_ptr == 0 { eprintln!("nsl_awq_matmul: null pointer"); return 0; }
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

    let tensor = Box::new(NslTensor::new(
        out_data as *mut c_void,
        shape,
        strides,
        2,
        out_len as i64,
        0,
        1,
        1,
        0,
    ));
    Box::into_raw(tensor) as i64
}

/// Apply AWQ calibration per-channel scaling to a weight tensor.
///
/// Used by the final-compile AWQ lowering pass when a calibration sidecar is
/// present.  Multiplies each input-channel column of the weight matrix by
/// `scale[col]^alpha`.  Returns a new NslTensor (f32, CPU, [k, n]) with the
/// scaled values; the caller is responsible for freeing it.
///
/// Parameters:
///   * `weight_ptr`  — pointer to an NslTensor of shape `[k, n]` (f32 or f64)
///   * `scales_ptr`  — pointer to a C array of `scales_len` f32 scale values;
///                     must have length == `n` (number of input channels)
///   * `scales_len`  — number of elements in the scales array (must equal `n`)
///   * `alpha`       — exponent applied to each scale (typically 0.5)
///
/// Returns 0 on error (null weight, shape mismatch).
#[no_mangle]
pub extern "C" fn nsl_awq_pre_scale_weight(
    weight_ptr: i64,
    scales_ptr: i64,
    scales_len: i64,
    alpha: f64,
) -> i64 {
    if weight_ptr == 0 {
        eprintln!("nsl_awq_pre_scale_weight: null weight tensor");
        return 0;
    }
    let t = unsafe { &*(weight_ptr as *const NslTensor) };
    if t.ndim < 2 {
        eprintln!("nsl_awq_pre_scale_weight: weight must be 2D (got {}D)", t.ndim);
        return 0;
    }

    let k = unsafe { *t.shape } as usize;
    let n = unsafe { *t.shape.add(1) } as usize;
    let len = t.len as usize;

    if scales_ptr == 0 || scales_len as usize != n {
        // No scaling — clone the weight as-is.
        let data: Vec<f32> = if t.dtype == 1 {
            let raw = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
            raw.to_vec()
        } else {
            let raw = unsafe { std::slice::from_raw_parts(t.data as *const f64, len) };
            raw.iter().map(|&v| v as f32).collect()
        };
        let shape_arr = [k as i64, n as i64];
        return make_f32_tensor(&data, &shape_arr);
    }

    let scales_raw =
        unsafe { std::slice::from_raw_parts(scales_ptr as *const f32, scales_len as usize) };

    let data: Vec<f32> = if t.dtype == 1 {
        let raw = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
        raw.to_vec()
    } else {
        let raw = unsafe { std::slice::from_raw_parts(t.data as *const f64, len) };
        raw.iter().map(|&v| v as f32).collect()
    };

    let alpha_f32 = alpha as f32;
    let mut scaled = Vec::with_capacity(len);
    for row in 0..k {
        for col in 0..n {
            let s = scales_raw[col].powf(alpha_f32);
            scaled.push(data[row * n + col] * s);
        }
    }

    let shape_arr = [k as i64, n as i64];
    make_f32_tensor(&scaled, &shape_arr)
}

/// Helper: allocate a new f32 NslTensor from `data` with the given `shape`.
fn make_f32_tensor(data: &[f32], shape: &[i64]) -> i64 {
    let ndim = shape.len() as i64;
    let total = data.len();
    let shape_ptr =
        crate::memory::checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
    for (i, &s) in shape.iter().enumerate() {
        unsafe { *shape_ptr.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape_ptr, ndim);
    let data_size = std::mem::size_of_val(data);
    let data_ptr = crate::memory::checked_alloc(data_size) as *mut f32;
    unsafe { std::ptr::copy_nonoverlapping(data.as_ptr(), data_ptr, total) };
    let t = Box::new(NslTensor::new(
        data_ptr as *mut std::ffi::c_void,
        shape_ptr,
        strides,
        ndim,
        total as i64,
        0,  // device: CPU
        1,  // dtype: f32
        1,  // owns_data
        0,  // data_owner
    ));
    Box::into_raw(t) as i64
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

// ---------------------------------------------------------------------------
// AWQ sidecar reader (calibration harness → runtime)
// ---------------------------------------------------------------------------

use std::collections::HashMap;
use std::path::Path;

const AWQ_SIDECAR_KEY: &str = "awq_activation_scales";
const AWQ_SIDECAR_VERSION: u32 = 1;

#[derive(Debug, Clone)]
pub struct AwqScales {
    /// Projection name → per-output-channel max|activation| values.
    pub by_projection: HashMap<String, Vec<f32>>,
}

#[derive(Debug)]
pub enum AwqScalesError {
    Io(std::io::Error),
    BadJson(String),
    MissingAwqKey,
    BadBase64(String),
    BlobTooSmall { need: usize, got: usize },
    UnsupportedVersion { got: u32 },
    BlobTruncated { at: &'static str },
    BadUtf8,
}

impl std::fmt::Display for AwqScalesError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(e) => write!(f, "I/O: {e}"),
            Self::BadJson(e) => write!(f, "bad sidecar JSON: {e}"),
            Self::MissingAwqKey => write!(f, "sidecar has no '{AWQ_SIDECAR_KEY}' key"),
            Self::BadBase64(e) => write!(f, "bad base64 for awq blob: {e}"),
            Self::BlobTooSmall { need, got } => write!(f, "blob too small: need {need}, got {got}"),
            Self::UnsupportedVersion { got } => write!(f, "unsupported AWQ sidecar version {got} (expected {AWQ_SIDECAR_VERSION})"),
            Self::BlobTruncated { at } => write!(f, "blob truncated at {at}"),
            Self::BadUtf8 => write!(f, "invalid UTF-8 in projection name"),
        }
    }
}

impl std::error::Error for AwqScalesError {}

impl From<std::io::Error> for AwqScalesError {
    fn from(e: std::io::Error) -> Self { Self::Io(e) }
}

impl AwqScales {
    /// Parse the raw AWQ-format blob produced by the calibration harness.
    pub fn from_blob(blob: &[u8]) -> Result<Self, AwqScalesError> {
        if blob.len() < 8 {
            return Err(AwqScalesError::BlobTooSmall { need: 8, got: blob.len() });
        }
        let version = u32::from_le_bytes(blob[0..4].try_into().unwrap());
        if version != AWQ_SIDECAR_VERSION {
            return Err(AwqScalesError::UnsupportedVersion { got: version });
        }
        let num_projections = u32::from_le_bytes(blob[4..8].try_into().unwrap()) as usize;
        let mut by_projection = HashMap::with_capacity(num_projections);
        let mut cursor = 8;
        for _ in 0..num_projections {
            if blob.len() < cursor + 4 {
                return Err(AwqScalesError::BlobTruncated { at: "name_len" });
            }
            let name_len = u32::from_le_bytes(blob[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            if blob.len() < cursor + name_len {
                return Err(AwqScalesError::BlobTruncated { at: "name bytes" });
            }
            let name = std::str::from_utf8(&blob[cursor..cursor + name_len])
                .map_err(|_| AwqScalesError::BadUtf8)?
                .to_string();
            cursor += name_len;
            if blob.len() < cursor + 4 {
                return Err(AwqScalesError::BlobTruncated { at: "channel_count" });
            }
            let channel_count = u32::from_le_bytes(blob[cursor..cursor + 4].try_into().unwrap()) as usize;
            cursor += 4;
            let scale_bytes = channel_count.checked_mul(4).ok_or(AwqScalesError::BlobTruncated { at: "scales (channel_count overflow)" })?;
            if blob.len() < cursor + scale_bytes {
                return Err(AwqScalesError::BlobTruncated { at: "scales" });
            }
            let mut scales = Vec::with_capacity(channel_count);
            for i in 0..channel_count {
                let off = cursor + i * 4;
                scales.push(f32::from_le_bytes(blob[off..off + 4].try_into().unwrap()));
            }
            cursor += scale_bytes;
            by_projection.insert(name, scales);
        }
        Ok(Self { by_projection })
    }

    /// Read the sidecar JSON at `path`, base64-decode the
    /// `"awq_activation_scales"` key, and parse into `AwqScales`.
    pub fn from_sidecar_json_path(path: &Path) -> Result<Self, AwqScalesError> {
        use base64::{engine::general_purpose::STANDARD, Engine};
        let json = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&json).map_err(|e| AwqScalesError::BadJson(e.to_string()))?;
        let b64 = v
            .get("hooks")
            .and_then(|h| h.get(AWQ_SIDECAR_KEY))
            .and_then(|s| s.as_str())
            .ok_or(AwqScalesError::MissingAwqKey)?;
        let blob = STANDARD.decode(b64).map_err(|e| AwqScalesError::BadBase64(e.to_string()))?;
        Self::from_blob(&blob)
    }
}

#[cfg(test)]
mod awq_sidecar_reader_tests {
    use super::*;
    use std::io::Write;

    /// Produce a valid v1 AWQ sidecar blob for testing.
    fn valid_sidecar_blob(entries: &[(&str, &[f32])]) -> Vec<u8> {
        let mut blob = Vec::new();
        blob.extend_from_slice(&1u32.to_le_bytes()); // version
        blob.extend_from_slice(&(entries.len() as u32).to_le_bytes());
        for (name, scales) in entries {
            let nb = name.as_bytes();
            blob.extend_from_slice(&(nb.len() as u32).to_le_bytes());
            blob.extend_from_slice(nb);
            blob.extend_from_slice(&(scales.len() as u32).to_le_bytes());
            for v in *scales {
                blob.extend_from_slice(&v.to_le_bytes());
            }
        }
        blob
    }

    #[test]
    fn parse_blob_produces_named_lookup() {
        let blob = valid_sidecar_blob(&[
            ("blocks.0.attn.wq", &[1.0, 2.0]),
            ("blocks.0.attn.wk", &[0.5, 0.25, 0.125]),
        ]);
        let scales = AwqScales::from_blob(&blob).unwrap();
        assert_eq!(scales.by_projection.get("blocks.0.attn.wq").unwrap(), &vec![1.0, 2.0]);
        assert_eq!(scales.by_projection.get("blocks.0.attn.wk").unwrap(), &vec![0.5, 0.25, 0.125]);
        assert!(scales.by_projection.get("missing").is_none());
    }

    #[test]
    fn parse_rejects_bad_version() {
        let mut blob = valid_sidecar_blob(&[("x", &[1.0])]);
        blob[0..4].copy_from_slice(&7u32.to_le_bytes());
        assert!(AwqScales::from_blob(&blob).is_err());
    }

    #[test]
    fn parse_rejects_truncated_blob() {
        let full = valid_sidecar_blob(&[("long_name", &[1.0, 2.0, 3.0])]);
        let truncated = &full[..full.len() / 2];
        assert!(AwqScales::from_blob(truncated).is_err());
    }

    #[test]
    fn from_sidecar_reads_base64_key() {
        use base64::{engine::general_purpose::STANDARD, Engine};
        let blob = valid_sidecar_blob(&[("p", &[0.5])]);
        let b64 = STANDARD.encode(&blob);
        let sidecar_json = format!(
            r#"{{"version":1,"checkpoint_sha256":"","calibration_data_sha256":"","hook_set_sha256":"","cache_key_digest":"","num_samples_used":0,"hooks":{{"awq_activation_scales":"{b64}"}}}}"#
        );
        let tmp = std::env::temp_dir().join(format!("nsl-awq-sidecar-{}.json", std::process::id()));
        std::fs::File::create(&tmp).unwrap().write_all(sidecar_json.as_bytes()).unwrap();
        let scales = AwqScales::from_sidecar_json_path(&tmp).unwrap();
        assert_eq!(scales.by_projection.get("p").unwrap(), &vec![0.5]);
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn from_sidecar_errors_when_awq_key_absent() {
        let sidecar_json = r#"{"version":1,"checkpoint_sha256":"","calibration_data_sha256":"","hook_set_sha256":"","cache_key_digest":"","num_samples_used":0,"hooks":{}}"#;
        let tmp = std::env::temp_dir().join(format!("nsl-awq-sidecar-empty-{}.json", std::process::id()));
        std::fs::write(&tmp, sidecar_json).unwrap();
        assert!(AwqScales::from_sidecar_json_path(&tmp).is_err());
        let _ = std::fs::remove_file(&tmp);
    }
}

/// Lightweight output suitable for round-trip tests without requiring
/// a full `AwqPackedWeight` comparison.  The real packed-weight path
/// can layer on top of this helper once the wider AWQ codegen plumbing
/// is in place (Task 13).
#[derive(Debug, Clone, PartialEq)]
pub struct AwqQuantizedChecked {
    pub dequantized_check_bytes: Vec<u8>,
}

/// Quantize a weight matrix using optional calibrated per-input-channel
/// scales.  When `scales` is `None`, uses a vector of 1.0s (equivalent
/// to uncalibrated AWQ).  Otherwise applies `s[c]^alpha` per input
/// channel to protect salient channels during quantization.
///
/// `weight` is `[out_channels, in_channels]` in row-major layout.
/// `scales`, when provided, has length `in_channels`.
///
/// This helper is the entry point for calibration-driven AWQ scaling.
/// It replaces the `_calibration_ptr` placeholder in the existing AWQ
/// runtime — callers that have calibration data supply it; callers
/// that don't pass `None` and get the uncalibrated baseline.
pub fn awq_quantize_with_scales(
    weight: &[f32],
    in_channels: usize,
    out_channels: usize,
    scales: Option<&[f32]>,
    alpha: f32,
) -> AwqQuantizedChecked {
    assert_eq!(weight.len(), in_channels * out_channels, "weight length mismatch");
    let effective: Vec<f32> = match scales {
        Some(s) => {
            assert_eq!(s.len(), in_channels, "scales length must equal in_channels");
            s.iter().map(|v| v.powf(alpha)).collect()
        }
        None => vec![1.0; in_channels],
    };
    // Apply per-input-channel scaling to the weight matrix.
    let mut scaled = Vec::with_capacity(weight.len());
    for row in 0..out_channels {
        for col in 0..in_channels {
            let v = weight[row * in_channels + col] * effective[col];
            scaled.push(v);
        }
    }
    // Minimal quantization: absmax per row, scale to int8 range.
    // Real AWQ would do groupwise; this is enough to expose scale
    // differences in tests and serves as the integration surface for
    // Task 13's quant-pass wiring.
    let mut out_bytes = Vec::with_capacity(weight.len());
    for row in 0..out_channels {
        let start = row * in_channels;
        let amax = scaled[start..start + in_channels]
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f32, f32::max)
            .max(1e-9);
        for i in 0..in_channels {
            let q = ((scaled[start + i] / amax) * 127.0).round().clamp(-128.0, 127.0);
            out_bytes.push(q as i8 as u8);
        }
    }
    AwqQuantizedChecked {
        dequantized_check_bytes: out_bytes,
    }
}

#[cfg(test)]
mod awq_quantize_with_scales_tests {
    use super::*;

    fn toy_weight(out_channels: usize, in_channels: usize) -> Vec<f32> {
        (0..out_channels * in_channels)
            .map(|i| (i as f32 - (out_channels * in_channels) as f32 / 2.0) * 0.01)
            .collect()
    }

    #[test]
    fn quantize_with_none_scales_matches_all_ones_baseline() {
        let weight = toy_weight(8, 16);
        let out_none = awq_quantize_with_scales(&weight, 16, 8, None, 0.5);
        let ones = vec![1.0_f32; 16];
        let out_ones = awq_quantize_with_scales(&weight, 16, 8, Some(&ones), 0.5);
        assert_eq!(out_none.dequantized_check_bytes, out_ones.dequantized_check_bytes,
            "None scales must equal [1.0; in_channels] scales after s^alpha (1^x = 1)");
    }

    #[test]
    fn quantize_with_nontrivial_scales_differs_from_baseline() {
        let weight = toy_weight(8, 16);
        let ones = vec![1.0_f32; 16];
        let varied: Vec<f32> = (0..16).map(|i| 0.5 + i as f32 * 0.2).collect();
        let out_ones = awq_quantize_with_scales(&weight, 16, 8, Some(&ones), 0.5);
        let out_varied = awq_quantize_with_scales(&weight, 16, 8, Some(&varied), 0.5);
        assert_ne!(out_ones.dequantized_check_bytes, out_varied.dequantized_check_bytes,
            "Non-trivial scales must change quantization output");
    }

    #[test]
    fn quantize_rejects_wrong_length_scales() {
        let weight = toy_weight(8, 16);
        let bad = vec![1.0_f32; 99]; // wrong length
        let result = std::panic::catch_unwind(|| {
            awq_quantize_with_scales(&weight, 16, 8, Some(&bad), 0.5)
        });
        assert!(result.is_err(), "mismatched scales length must panic");
    }
}

// ---------------------------------------------------------------------------
// nsl_awq_write_sidecar — subprocess-side serialization FFI
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct AwqProjectionDescriptor {
    pub path_ptr:     *const u8,
    pub path_len:     usize,
    pub channels:     u32,
    pub _pad:         u32,
    pub running_ptr:  *const f32,
}

/// Write an AWQ activation-scales sidecar JSON file.
///
/// Return codes:
///   0 = success
///   1 = sidecar_path is null or not valid UTF-8
///   2 = a projection name is not valid UTF-8, or JSON serialization failed
///   3 = disk write failed
#[no_mangle]
pub extern "C" fn nsl_awq_write_sidecar(
    sidecar_path_ptr: *const u8,
    sidecar_path_len: usize,
    projections_ptr:  *const AwqProjectionDescriptor,
    projections_len:  usize,
) -> i32 {
    if sidecar_path_ptr.is_null() { return 1; }
    let path_bytes = unsafe {
        std::slice::from_raw_parts(sidecar_path_ptr, sidecar_path_len)
    };
    let path_str = match std::str::from_utf8(path_bytes) {
        Ok(s) => s,
        Err(_) => return 1,
    };

    let descs: &[AwqProjectionDescriptor] = if projections_len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(projections_ptr, projections_len) }
    };

    let mut by_projection = std::collections::BTreeMap::<String, Vec<f32>>::new();
    for d in descs {
        let name_bytes = unsafe { std::slice::from_raw_parts(d.path_ptr, d.path_len) };
        let name = match std::str::from_utf8(name_bytes) {
            Ok(s) => s.to_string(),
            Err(_) => return 2,
        };
        let values = unsafe {
            std::slice::from_raw_parts(d.running_ptr, d.channels as usize).to_vec()
        };
        by_projection.insert(name, values);
    }

    let mut blob = Vec::new();
    blob.extend_from_slice(&AWQ_SIDECAR_VERSION.to_le_bytes());
    blob.extend_from_slice(&(by_projection.len() as u32).to_le_bytes());
    for (name, scales) in &by_projection {
        let name_bytes = name.as_bytes();
        blob.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        blob.extend_from_slice(name_bytes);
        blob.extend_from_slice(&(scales.len() as u32).to_le_bytes());
        for scale in scales {
            blob.extend_from_slice(&scale.to_le_bytes());
        }
    }

    use base64::{engine::general_purpose::STANDARD, Engine};
    let awq_blob_b64 = STANDARD.encode(blob);

    let json = serde_json::json!({
        "version": 2,
        "checkpoint_sha256": "",
        "calibration_data_sha256": "",
        "hook_set_sha256": "",
        "cache_key_digest": "",
        "num_samples_used": 0,
        "hooks": {
            "awq_activation_scales": awq_blob_b64,
        },
    });

    let bytes = match serde_json::to_vec(&json) {
        Ok(b) => b,
        Err(_) => return 2,
    };
    match std::fs::write(path_str, bytes) {
        Ok(_) => 0,
        Err(_) => 3,
    }
}

#[cfg(test)]
mod write_sidecar_tests {
    use super::*;
    use tempfile::NamedTempFile;

    #[test]
    fn writes_sidecar_with_two_projections() {
        let tmp = NamedTempFile::new().unwrap();
        let path_str = tmp.path().to_string_lossy().to_string();

        let up_path = "TinyMLP.up_proj";
        let down_path = "TinyMLP.down_proj";
        let up_buf: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let down_buf: Vec<f32> = (0..128).map(|i| (i as f32) * 0.5).collect();

        let descs = vec![
            AwqProjectionDescriptor {
                path_ptr: up_path.as_ptr(),
                path_len: up_path.len(),
                channels: 64,
                _pad: 0,
                running_ptr: up_buf.as_ptr(),
            },
            AwqProjectionDescriptor {
                path_ptr: down_path.as_ptr(),
                path_len: down_path.len(),
                channels: 128,
                _pad: 0,
                running_ptr: down_buf.as_ptr(),
            },
        ];

        let rc = nsl_awq_write_sidecar(
            path_str.as_ptr(), path_str.len(),
            descs.as_ptr(), descs.len(),
        );
        assert_eq!(rc, 0);

        let scales = AwqScales::from_sidecar_json_path(tmp.path()).unwrap();
        assert_eq!(scales.by_projection["TinyMLP.up_proj"].len(), 64);
        assert_eq!(scales.by_projection["TinyMLP.down_proj"].len(), 128);
    }

    #[test]
    fn returns_1_on_invalid_utf8_path() {
        let bad: [u8; 4] = [0xff, 0xfe, 0xfd, 0xfc];
        let rc = nsl_awq_write_sidecar(
            bad.as_ptr(), bad.len(),
            std::ptr::null(), 0,
        );
        assert_eq!(rc, 1);
    }
}
