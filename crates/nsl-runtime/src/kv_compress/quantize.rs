// crates/nsl-runtime/src/kv_compress/quantize.rs
//! KV-cache quantization: INT8/INT4/FP8 compress and decompress.
//!
//! Quantization is fused into the KV write path: after K = matmul(x, W_k),
//! the runtime quantizes in-place and writes the compressed dtype to the
//! block pool. Dequantization runs on read (CPU) or in FlashAttention
//! tile-loading (GPU, deferred to M42b).

use super::KvQuantScheme;

/// Maximum number of scale/zero entries per block (covers up to 128 heads or 128 tokens).
pub const MAX_SCALES: usize = 128;

/// Per-block quantization metadata stored alongside block data.
///
/// For INT8 per-head: `scales[0..num_heads]` are per-head scale factors.
/// For INT8 per-token: `scales[0..block_size]` are per-token scale factors.
/// For INT4 per-group: `scales[g]` = range/15, `zero_points[g]` = min_val (f32, no overflow).
/// For FP8: no metadata needed (direct cast).
#[derive(Clone)]
#[repr(C)]
pub struct KvBlockQuantMeta {
    pub scheme: u8,
    pub num_scales: u16,
    pub _pad: u8,
    pub scales: [f32; MAX_SCALES],
    pub zero_points: [f32; MAX_SCALES],  // f32 to avoid i8 overflow for large min_val/scale ratios
}

impl Default for KvBlockQuantMeta {
    fn default() -> Self {
        KvBlockQuantMeta {
            scheme: 0,
            num_scales: 0,
            _pad: 0,
            scales: [0.0; MAX_SCALES],
            zero_points: [0.0; MAX_SCALES],
        }
    }
}

/// Compute block data bytes (excluding metadata) for a given scheme.
///
/// Arguments:
/// - `num_heads`: attention heads in this layer
/// - `block_size`: tokens per block
/// - `head_dim`: dimension per head
pub fn block_data_bytes(scheme: KvQuantScheme, num_heads: usize, block_size: usize, head_dim: usize) -> usize {
    let elements = num_heads * block_size * head_dim;
    match scheme {
        KvQuantScheme::None => elements * 2,          // FP16: 2 bytes
        KvQuantScheme::Int8PerHead |
        KvQuantScheme::Int8PerToken => elements,      // INT8: 1 byte
        KvQuantScheme::Int4PerGroup => elements.div_ceil(2), // INT4: 0.5 bytes, round up
        KvQuantScheme::Fp8 => elements,               // FP8: 1 byte
    }
}

/// Total block bytes including metadata (K or V, not both).
pub fn block_total_bytes(scheme: KvQuantScheme, num_heads: usize, block_size: usize, head_dim: usize) -> usize {
    let data = block_data_bytes(scheme, num_heads, block_size, head_dim);
    let meta = match scheme {
        KvQuantScheme::None | KvQuantScheme::Fp8 => 0,
        _ => std::mem::size_of::<KvBlockQuantMeta>(),
    };
    data + meta
}

// ---------------------------------------------------------------------------
// INT8 per-head quantization
// ---------------------------------------------------------------------------

/// Quantize f32 values to INT8 with per-head symmetric scaling.
///
/// For each head h: scale_h = max(|values[h]|) / 127.0
/// Output: quantized[i] = round(values[i] / scale_h), clamped to [-127, 127]
///
/// `values`: [num_heads * block_size * head_dim] f32 input
/// `output`: [num_heads * block_size * head_dim] i8 output
/// `meta`: KvBlockQuantMeta to fill with per-head scales
pub fn quantize_int8_per_head(
    values: &[f32],
    output: &mut [i8],
    meta: &mut KvBlockQuantMeta,
    num_heads: usize,
    block_size: usize,
    head_dim: usize,
) {
    meta.scheme = KvQuantScheme::Int8PerHead as u8;
    meta.num_scales = num_heads as u16;
    let head_stride = block_size * head_dim;

    for h in 0..num_heads {
        let start = h * head_stride;
        let end = start + head_stride;
        let head_vals = &values[start..end];

        // Compute per-head scale
        let abs_max = head_vals.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
        meta.scales[h] = scale;

        // Quantize
        let inv_scale = 1.0 / scale;
        for (i, &v) in head_vals.iter().enumerate() {
            let q = (v * inv_scale).round().clamp(-127.0, 127.0) as i8;
            output[start + i] = q;
        }
    }
}

/// Dequantize INT8 per-head back to f32.
pub fn dequantize_int8_per_head(
    quantized: &[i8],
    output: &mut [f32],
    meta: &KvBlockQuantMeta,
    num_heads: usize,
    block_size: usize,
    head_dim: usize,
) {
    let head_stride = block_size * head_dim;
    for h in 0..num_heads {
        let scale = meta.scales[h];
        let start = h * head_stride;
        for i in 0..head_stride {
            output[start + i] = quantized[start + i] as f32 * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// INT8 per-token quantization
// ---------------------------------------------------------------------------

/// Quantize f32 values to INT8 with per-token symmetric scaling.
///
/// For each token t: scale_t = max(|values[:, t, :]|) / 127.0
pub fn quantize_int8_per_token(
    values: &[f32],
    output: &mut [i8],
    meta: &mut KvBlockQuantMeta,
    num_heads: usize,
    block_size: usize,
    head_dim: usize,
) {
    meta.scheme = KvQuantScheme::Int8PerToken as u8;
    meta.num_scales = block_size as u16;

    for t in 0..block_size {
        // Find max across all heads for this token
        let mut abs_max = 0.0f32;
        for h in 0..num_heads {
            let offset = h * block_size * head_dim + t * head_dim;
            for d in 0..head_dim {
                abs_max = abs_max.max(values[offset + d].abs());
            }
        }
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
        meta.scales[t] = scale;

        // Quantize all heads for this token
        let inv_scale = 1.0 / scale;
        for h in 0..num_heads {
            let offset = h * block_size * head_dim + t * head_dim;
            for d in 0..head_dim {
                let q = (values[offset + d] * inv_scale).round().clamp(-127.0, 127.0) as i8;
                output[offset + d] = q;
            }
        }
    }
}

/// Dequantize INT8 per-token back to f32.
pub fn dequantize_int8_per_token(
    quantized: &[i8],
    output: &mut [f32],
    meta: &KvBlockQuantMeta,
    num_heads: usize,
    block_size: usize,
    head_dim: usize,
) {
    for t in 0..block_size {
        let scale = meta.scales[t];
        for h in 0..num_heads {
            let offset = h * block_size * head_dim + t * head_dim;
            for d in 0..head_dim {
                output[offset + d] = quantized[offset + d] as f32 * scale;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// INT4 per-group quantization
// ---------------------------------------------------------------------------

/// Default group size for INT4 quantization.
pub const INT4_GROUP_SIZE: usize = 64;

/// Quantize f32 values to INT4 with per-group asymmetric scaling.
///
/// Groups of `group_size` elements share a scale and zero-point.
/// q = round((v - zero) / scale), clamped to [0, 15]
/// Two INT4 values packed per byte (low nibble first).
pub fn quantize_int4_per_group(
    values: &[f32],
    output: &mut [u8],
    meta: &mut KvBlockQuantMeta,
    group_size: usize,
) {
    // IMPORTANT-3 fix: zero output buffer to prevent stale high nibbles from buffer reuse
    output.fill(0);
    meta.scheme = KvQuantScheme::Int4PerGroup as u8;
    let num_groups = values.len().div_ceil(group_size);
    meta.num_scales = num_groups as u16;

    for g in 0..num_groups {
        let start = g * group_size;
        let end = (start + group_size).min(values.len());
        let group = &values[start..end];

        let min_val = group.iter().fold(f32::MAX, |m, &v| m.min(v));
        let max_val = group.iter().fold(f32::MIN, |m, &v| m.max(v));
        let range = max_val - min_val;
        let scale = if range == 0.0 { 1.0 } else { range / 15.0 };
        meta.scales[g] = scale;
        meta.zero_points[g] = min_val;  // store min_val directly as f32 (no i8 overflow)

        let inv_scale = 1.0 / scale;
        for (i, &v) in group.iter().enumerate() {
            let q = ((v - min_val) * inv_scale).round().clamp(0.0, 15.0) as u8;
            let byte_idx = (start + i) / 2;
            if (start + i).is_multiple_of(2) {
                output[byte_idx] = q; // low nibble
            } else {
                output[byte_idx] |= q << 4; // high nibble
            }
        }
    }
}

/// Dequantize INT4 per-group back to f32.
pub fn dequantize_int4_per_group(
    packed: &[u8],
    output: &mut [f32],
    meta: &KvBlockQuantMeta,
    num_elements: usize,
    group_size: usize,
) {
    for (i, out) in output.iter_mut().enumerate().take(num_elements) {
        let byte_idx = i / 2;
        let nibble = if i % 2 == 0 {
            packed[byte_idx] & 0x0F
        } else {
            (packed[byte_idx] >> 4) & 0x0F
        };
        let g = i / group_size;
        let scale = meta.scales[g];
        let min_val = meta.zero_points[g]; // f32 min_val stored directly
        *out = nibble as f32 * scale + min_val;
    }
}

// ---------------------------------------------------------------------------
// FP8 E4M3 conversion
// ---------------------------------------------------------------------------

/// Convert f32 to FP8 E4M3 (truncation).
///
/// FP8 E4M3: 1 sign + 4 exponent + 3 mantissa, range ~= [-448, 448].
/// Simple clamp-and-truncate — no scaling metadata needed.
pub fn quantize_fp8(values: &[f32], output: &mut [u8]) {
    for (i, &v) in values.iter().enumerate() {
        output[i] = f32_to_fp8_e4m3(v);
    }
}

/// Convert FP8 E4M3 back to f32.
pub fn dequantize_fp8(packed: &[u8], output: &mut [f32]) {
    for (i, &b) in packed.iter().enumerate() {
        output[i] = fp8_e4m3_to_f32(b);
    }
}

/// Convert a single f32 to FP8 E4M3: round to nearest (ties to even) onto the
/// E4M3 grid, saturating at ±448, then encode the rounded value exactly.
///
/// This used to truncate the mantissa to its top 3 bits (a bias of up to one
/// ulp toward zero — 447 became 416) and flush everything below 2^-6 to zero
/// instead of encoding E4M3's subnormals.
fn f32_to_fp8_e4m3(v: f32) -> u8 {
    let r = crate::fp8::round_to_fp8(v as f64, crate::fp8::FP8_FORMAT_E4M3);
    if r.is_nan() {
        return 0x7F;
    }
    let sign = if r.is_sign_negative() { 0x80u8 } else { 0 };
    let a = r.abs();
    if a < 2f64.powi(-6) {
        // Zero or subnormal: a = m * 2^-9 exactly, m in 0..=7.
        return sign | (a * 512.0) as u8;
    }
    // Normal: a = (1 + m/8) * 2^e exactly, e in -6..=8, m in 0..=7.
    let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let m = ((a / 2f64.powi(e) - 1.0) * 8.0) as u8;
    sign | (((e + 7) as u8) << 3) | m
}

/// Convert a single FP8 E4M3 to f32.
fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0 };
    let exp = ((b >> 3) & 0x0F) as i32;
    let mantissa = (b & 0x07) as f32;
    // E4M3 has no infinities; S.1111.111 is its only NaN.
    if exp == 15 && mantissa == 7.0 {
        return f32::NAN;
    }
    if exp == 0 {
        // Subnormal (and signed zero): m/8 * 2^-6.
        return sign * mantissa / 8.0 * 2f32.powi(-6);
    }
    sign * (1.0 + mantissa / 8.0) * 2f32.powi(exp - 7)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int8_per_head_roundtrip() {
        let num_heads = 4;
        let block_size = 2;
        let head_dim = 8;
        let n = num_heads * block_size * head_dim;

        let values: Vec<f32> = (0..n).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let mut quantized = vec![0i8; n];
        let mut meta = KvBlockQuantMeta::default();

        quantize_int8_per_head(&values, &mut quantized, &mut meta, num_heads, block_size, head_dim);
        assert_eq!(meta.scheme, KvQuantScheme::Int8PerHead as u8);
        assert_eq!(meta.num_scales, 4);

        let mut restored = vec![0.0f32; n];
        dequantize_int8_per_head(&quantized, &mut restored, &meta, num_heads, block_size, head_dim);

        // Max error < 0.5% of range
        let range = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        for (orig, rest) in values.iter().zip(restored.iter()) {
            let err = (orig - rest).abs() / range.max(1e-6);
            assert!(err < 0.01, "INT8 per-head error too large: orig={orig}, restored={rest}, err={err}");
        }
    }

    #[test]
    fn int8_per_token_roundtrip() {
        let num_heads = 2;
        let block_size = 4;
        let head_dim = 4;
        let n = num_heads * block_size * head_dim;

        let values: Vec<f32> = (0..n).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let mut quantized = vec![0i8; n];
        let mut meta = KvBlockQuantMeta::default();

        quantize_int8_per_token(&values, &mut quantized, &mut meta, num_heads, block_size, head_dim);
        assert_eq!(meta.scheme, KvQuantScheme::Int8PerToken as u8);
        assert_eq!(meta.num_scales, 4);

        let mut restored = vec![0.0f32; n];
        dequantize_int8_per_token(&quantized, &mut restored, &meta, num_heads, block_size, head_dim);

        let range = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        for (orig, rest) in values.iter().zip(restored.iter()) {
            let err = (orig - rest).abs() / range.max(1e-6);
            assert!(err < 0.01, "INT8 per-token error too large: orig={orig}, restored={rest}");
        }
    }

    #[test]
    fn int4_per_group_roundtrip() {
        let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let packed_len = values.len().div_ceil(2);
        let mut packed = vec![0u8; packed_len];
        let mut meta = KvBlockQuantMeta::default();

        quantize_int4_per_group(&values, &mut packed, &mut meta, INT4_GROUP_SIZE);
        assert_eq!(meta.scheme, KvQuantScheme::Int4PerGroup as u8);

        let mut restored = vec![0.0f32; values.len()];
        dequantize_int4_per_group(&packed, &mut restored, &meta, values.len(), INT4_GROUP_SIZE);

        // INT4 has wider tolerance (~6% of range)
        let range = values.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        for (orig, rest) in values.iter().zip(restored.iter()) {
            let err = (orig - rest).abs() / range.max(1e-6);
            assert!(err < 0.10, "INT4 error too large: orig={orig}, restored={rest}, err={err}");
        }
    }

    #[test]
    fn fp8_roundtrip() {
        // Each value lands on its nearest E4M3 neighbour: 1, 0.5 and 448 are
        // exact, 100 lies in [64, 128) where the spacing is 8 (100 is a tie
        // between 96 and 104 and goes to the even 96), 447 rounds up to 448,
        // and 0.003 is a subnormal: 0.003 * 2^9 = 1.536 rounds to 2, i.e.
        // 2 * 2^-9.
        let values = vec![0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 448.0, 447.0, 0.003];
        let want = [0.0, 1.0, -1.0, 0.5, -0.5, 96.0, -96.0, 448.0, 448.0, 2.0 / 512.0];
        let mut packed = vec![0u8; values.len()];
        quantize_fp8(&values, &mut packed);

        let mut restored = vec![0.0f32; values.len()];
        dequantize_fp8(&packed, &mut restored);
        assert_eq!(restored, want);
    }

    /// Every E4M3 code survives decode → encode, and encoding agrees with
    /// the runtime's FP8 rounding on the midpoints between neighbours.
    #[test]
    fn fp8_e4m3_codes_round_trip_and_round_to_nearest() {
        for code in 0..=255u8 {
            let v = fp8_e4m3_to_f32(code);
            if v.is_nan() {
                assert_eq!(code & 0x7F, 0x7F, "only S.1111.111 is NaN");
                continue;
            }
            assert_eq!(f32_to_fp8_e4m3(v), code, "code {code:#04x} decodes to {v}");
        }
        let mut grid: Vec<f32> = (0..0x7Fu8).map(fp8_e4m3_to_f32).collect();
        grid.sort_by(|a, b| a.partial_cmp(b).unwrap());
        for w in grid.windows(2) {
            for x in [(w[0] + w[1]) / 2.0, w[0] * 0.7 + w[1] * 0.3, w[0] * 0.3 + w[1] * 0.7] {
                let want = crate::fp8::round_to_fp8(x as f64, crate::fp8::FP8_FORMAT_E4M3) as f32;
                assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(x)), want, "x = {x}");
            }
        }
        assert_eq!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(1e9)), 448.0);
        assert!(fp8_e4m3_to_f32(f32_to_fp8_e4m3(f32::NAN)).is_nan());
    }

    #[test]
    fn block_data_bytes_calculation() {
        let heads = 32;
        let bs = 16;
        let dim = 128;
        let elems = heads * bs * dim; // 65536

        assert_eq!(block_data_bytes(KvQuantScheme::None, heads, bs, dim), elems * 2);
        assert_eq!(block_data_bytes(KvQuantScheme::Int8PerHead, heads, bs, dim), elems);
        assert_eq!(block_data_bytes(KvQuantScheme::Int4PerGroup, heads, bs, dim), elems / 2);
        assert_eq!(block_data_bytes(KvQuantScheme::Fp8, heads, bs, dim), elems);
    }

    #[test]
    fn block_total_bytes_includes_metadata() {
        let heads = 8;
        let bs = 16;
        let dim = 64;
        let data = block_data_bytes(KvQuantScheme::Int8PerHead, heads, bs, dim);
        let total = block_total_bytes(KvQuantScheme::Int8PerHead, heads, bs, dim);
        assert!(total > data);
        assert_eq!(total - data, std::mem::size_of::<KvBlockQuantMeta>());

        // FP8 has no metadata
        let fp8_data = block_data_bytes(KvQuantScheme::Fp8, heads, bs, dim);
        let fp8_total = block_total_bytes(KvQuantScheme::Fp8, heads, bs, dim);
        assert_eq!(fp8_data, fp8_total);
    }

    #[test]
    fn quantize_zeros() {
        let values = vec![0.0f32; 32];
        let mut quantized = vec![0i8; 32];
        let mut meta = KvBlockQuantMeta::default();
        quantize_int8_per_head(&values, &mut quantized, &mut meta, 1, 2, 16);
        assert!(quantized.iter().all(|&q| q == 0));
    }
}
