// crates/nsl-runtime/src/kv_compress/quantize.rs
//! KV-cache quantization: INT8/INT4/FP8 compress and decompress.
//!
//! These are library routines. Nothing calls them from the KV write path:
//! the paged KV cache stores K and V as written, and `@kv_compress` is a
//! refused decorator.

use super::KvQuantScheme;

/// Maximum number of scale/zero entries per block: 128 heads, 128 tokens or
/// 128 INT4 groups. A block that needs more is refused with
/// [`KvQuantError::TooManyScales`]; the quantizers used to index past the
/// arrays and panic (INT4 on 8 heads x 16 tokens x 128 dims needs 256
/// groups; per-token INT8 fails above 128 tokens per block).
pub const MAX_SCALES: usize = 128;

/// Why a block cannot be quantized.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuantError {
    /// The scheme needs `needed` scales and the metadata holds [`MAX_SCALES`].
    TooManyScales { needed: usize },
    /// An input or output slice is shorter than the block's dimensions say.
    BufferTooSmall { needed: usize, got: usize },
    /// A token range that does not fit in the block.
    TokenRange { offset: usize, tokens: usize, block_size: usize },
    /// The group size is zero.
    ZeroGroupSize,
}

impl std::fmt::Display for KvQuantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::TooManyScales { needed } => write!(
                f,
                "the block needs {needed} scale entries and its metadata holds {MAX_SCALES}"
            ),
            Self::BufferTooSmall { needed, got } => {
                write!(f, "a buffer of {got} elements where the block needs {needed}")
            }
            Self::TokenRange { offset, tokens, block_size } => write!(
                f,
                "tokens {offset}..{} do not fit in a block of {block_size}",
                offset + tokens
            ),
            Self::ZeroGroupSize => write!(f, "the group size must be positive"),
        }
    }
}

impl std::error::Error for KvQuantError {}

fn check_scales(needed: usize) -> Result<(), KvQuantError> {
    if needed > MAX_SCALES {
        return Err(KvQuantError::TooManyScales { needed });
    }
    Ok(())
}

fn check_len(needed: usize, got: usize) -> Result<(), KvQuantError> {
    if got < needed {
        return Err(KvQuantError::BufferTooSmall { needed, got });
    }
    Ok(())
}

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
) -> Result<(), KvQuantError> {
    check_scales(num_heads)?;
    let n = num_heads * block_size * head_dim;
    check_len(n, values.len())?;
    check_len(n, output.len())?;
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
    Ok(())
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
///
/// A whole block at once: `values` and `output` are both
/// `[num_heads, block_size, head_dim]`.
pub fn quantize_int8_per_token(
    values: &[f32],
    output: &mut [i8],
    meta: &mut KvBlockQuantMeta,
    num_heads: usize,
    block_size: usize,
    head_dim: usize,
) -> Result<(), KvQuantError> {
    meta.num_scales = 0;
    append_int8_per_token(values, output, meta, num_heads, block_size, head_dim, 0, block_size)
}

/// Quantize `num_tokens` new tokens into a block that already holds
/// `token_offset` tokens, per-token INT8.
///
/// `values` is `[num_heads, num_tokens, head_dim]` (the new tokens only);
/// `block` is the whole block, `[num_heads, block_size, head_dim]`. Each new
/// token gets its own scale, stored at its ABSOLUTE position
/// `token_offset + t`, and its codes go to that position in the block. The
/// scales and codes of tokens already stored are not touched, so they decode
/// with the scale they were written with.
#[allow(clippy::too_many_arguments)]
pub fn append_int8_per_token(
    values: &[f32],
    block: &mut [i8],
    meta: &mut KvBlockQuantMeta,
    num_heads: usize,
    block_size: usize,
    head_dim: usize,
    token_offset: usize,
    num_tokens: usize,
) -> Result<(), KvQuantError> {
    if token_offset + num_tokens > block_size {
        return Err(KvQuantError::TokenRange { offset: token_offset, tokens: num_tokens, block_size });
    }
    check_scales(token_offset + num_tokens)?;
    check_len(num_heads * num_tokens * head_dim, values.len())?;
    check_len(num_heads * block_size * head_dim, block.len())?;
    meta.scheme = KvQuantScheme::Int8PerToken as u8;
    meta.num_scales = meta.num_scales.max((token_offset + num_tokens) as u16);

    for t in 0..num_tokens {
        // Find max across all heads for this token
        let mut abs_max = 0.0f32;
        for h in 0..num_heads {
            let src = h * num_tokens * head_dim + t * head_dim;
            for d in 0..head_dim {
                abs_max = abs_max.max(values[src + d].abs());
            }
        }
        let scale = if abs_max == 0.0 { 1.0 } else { abs_max / 127.0 };
        meta.scales[token_offset + t] = scale;

        // Quantize all heads for this token
        let inv_scale = 1.0 / scale;
        for h in 0..num_heads {
            let src = h * num_tokens * head_dim + t * head_dim;
            let dst = h * block_size * head_dim + (token_offset + t) * head_dim;
            for d in 0..head_dim {
                let q = (values[src + d] * inv_scale).round().clamp(-127.0, 127.0) as i8;
                block[dst + d] = q;
            }
        }
    }
    Ok(())
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
) -> Result<(), KvQuantError> {
    if group_size == 0 {
        return Err(KvQuantError::ZeroGroupSize);
    }
    let num_groups = values.len().div_ceil(group_size);
    check_scales(num_groups)?;
    check_len(values.len().div_ceil(2), output.len())?;
    // IMPORTANT-3 fix: zero output buffer to prevent stale high nibbles from buffer reuse
    output.fill(0);
    meta.scheme = KvQuantScheme::Int4PerGroup as u8;
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
    Ok(())
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

        quantize_int8_per_head(&values, &mut quantized, &mut meta, num_heads, block_size, head_dim).unwrap();
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

        quantize_int8_per_token(&values, &mut quantized, &mut meta, num_heads, block_size, head_dim).unwrap();
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

        quantize_int4_per_group(&values, &mut packed, &mut meta, INT4_GROUP_SIZE).unwrap();
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
        quantize_int8_per_head(&values, &mut quantized, &mut meta, 1, 2, 16).unwrap();
        assert!(quantized.iter().all(|&q| q == 0));
    }

    /// A deterministic block of a realistic size, values in [-1, 1).
    fn block_values(n: usize) -> Vec<f32> {
        let mut s: u32 = 12345;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                ((s >> 16) as f32 / 32768.0) - 1.0
            })
            .collect()
    }

    /// 8 heads x 16 tokens x 128 dims in groups of 64 is 256 groups, twice
    /// what the metadata holds. This used to index past the scale array and
    /// panic; the unit tests ran 16 elements.
    #[test]
    fn int4_on_a_realistic_block_is_refused_not_a_panic() {
        let n = 8 * 16 * 128;
        let values = block_values(n);
        let mut packed = vec![0u8; n / 2];
        let mut meta = KvBlockQuantMeta::default();
        assert_eq!(
            quantize_int4_per_group(&values, &mut packed, &mut meta, INT4_GROUP_SIZE),
            Err(KvQuantError::TooManyScales { needed: 256 })
        );
        // The largest block the metadata does cover round-trips within half
        // a step of each group's own range.
        let n = MAX_SCALES * INT4_GROUP_SIZE;
        let values = block_values(n);
        let mut packed = vec![0u8; n / 2];
        quantize_int4_per_group(&values, &mut packed, &mut meta, INT4_GROUP_SIZE).unwrap();
        assert_eq!(meta.num_scales as usize, MAX_SCALES);
        let mut restored = vec![0.0f32; n];
        dequantize_int4_per_group(&packed, &mut restored, &meta, n, INT4_GROUP_SIZE);
        for g in 0..MAX_SCALES {
            let group = &values[g * INT4_GROUP_SIZE..(g + 1) * INT4_GROUP_SIZE];
            let lo = group.iter().cloned().fold(f32::MAX, f32::min);
            let hi = group.iter().cloned().fold(f32::MIN, f32::max);
            let half_step = 0.5 * (hi - lo) / 15.0;
            for (i, v) in group.iter().enumerate() {
                let r = restored[g * INT4_GROUP_SIZE + i];
                assert!((v - r).abs() <= half_step * 1.001, "group {g}, element {i}: {v} decoded as {r}");
            }
        }
    }

    /// A block of 256 tokens (the serving engine's default block) needs 256
    /// per-token scales.
    #[test]
    fn per_token_int8_above_128_tokens_is_refused_not_a_panic() {
        let (heads, block, dim) = (2, 256, 4);
        let values = block_values(heads * block * dim);
        let mut out = vec![0i8; values.len()];
        let mut meta = KvBlockQuantMeta::default();
        assert_eq!(
            quantize_int8_per_token(&values, &mut out, &mut meta, heads, block, dim),
            Err(KvQuantError::TooManyScales { needed: 256 })
        );
        assert_eq!(
            quantize_int8_per_head(&values[..129 * 4], &mut out[..129 * 4], &mut meta, 129, 1, 4),
            Err(KvQuantError::TooManyScales { needed: 129 })
        );
    }

    /// Appending tokens to a block must leave the tokens already stored
    /// decodable. Two heads, a block of four tokens, two dims; tokens 0 and
    /// 1 are written first, then 2 and 3, which are 100 times larger.
    ///
    /// By hand: token t's scale is its largest magnitude / 127. Token 0 holds
    /// {0.1, -0.2, 0.3, 0.4} -> 0.4/127; token 1 {0.5, 0.6, -0.7, 0.8} ->
    /// 0.8/127; token 2 {10, 20, 30, -40} -> 40/127; token 3 {50, 60, 70, 80}
    /// -> 80/127.
    #[test]
    fn appending_tokens_keeps_the_scales_of_earlier_tokens() {
        let (heads, block, dim) = (2usize, 4usize, 2usize);
        // [heads, 2 tokens, dim] for each call
        let first = [0.1f32, -0.2, 0.5, 0.6, /* head 1 */ 0.3, 0.4, -0.7, 0.8];
        let second = [10.0f32, 20.0, 50.0, 60.0, /* head 1 */ 30.0, -40.0, 70.0, 80.0];
        let mut codes = vec![0i8; heads * block * dim];
        let mut meta = KvBlockQuantMeta::default();

        append_int8_per_token(&first, &mut codes, &mut meta, heads, block, dim, 0, 2).unwrap();
        assert_eq!(meta.num_scales, 2);
        assert_eq!(&meta.scales[..2], &[0.4 / 127.0, 0.8 / 127.0]);

        append_int8_per_token(&second, &mut codes, &mut meta, heads, block, dim, 2, 2).unwrap();
        assert_eq!(meta.num_scales, 4);
        assert_eq!(&meta.scales[..4], &[0.4 / 127.0, 0.8 / 127.0, 40.0 / 127.0, 80.0 / 127.0]);

        // The whole block, [heads, 4 tokens, dim], as it was written.
        let whole = [
            0.1f32, -0.2, 0.5, 0.6, 10.0, 20.0, 50.0, 60.0, // head 0
            0.3, 0.4, -0.7, 0.8, 30.0, -40.0, 70.0, 80.0, // head 1
        ];
        let mut restored = vec![0.0f32; whole.len()];
        dequantize_int8_per_token(&codes, &mut restored, &meta, heads, block, dim);
        for h in 0..heads {
            for t in 0..block {
                let half_step = 0.5 * meta.scales[t];
                for d in 0..dim {
                    let i = h * block * dim + t * dim + d;
                    assert!(
                        (whole[i] - restored[i]).abs() <= half_step * 1.001,
                        "head {h} token {t} dim {d}: {} decoded as {}",
                        whole[i],
                        restored[i]
                    );
                }
            }
        }

        // And it is the block a single call over all four tokens produces.
        let mut once = vec![0i8; whole.len()];
        let mut once_meta = KvBlockQuantMeta::default();
        quantize_int8_per_token(&whole, &mut once, &mut once_meta, heads, block, dim).unwrap();
        assert_eq!(codes, once);
        assert_eq!(&meta.scales[..4], &once_meta.scales[..4]);

        // A range that runs off the block is refused.
        assert_eq!(
            append_int8_per_token(&first, &mut codes, &mut meta, heads, block, dim, 3, 2),
            Err(KvQuantError::TokenRange { offset: 3, tokens: 2, block_size: 4 })
        );
    }
}
