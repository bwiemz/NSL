# M42: KV-Cache Compression & Eviction — Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce KV-cache memory consumption for long-context inference via three strategies: quantized KV storage (INT8/INT4/FP8), sliding window eviction with attention sinks, and Heavy Hitter Oracle (H2O) attention-score-based eviction. All operate at the page level, integrating with M25's `BlockAllocator`.

**Architecture:** Five new runtime modules under `crates/nsl-runtime/src/kv_compress/` (quantization, sliding window, H2O, FFI, module root) + semantic validation for `@kv_compress` decorator + codegen config extraction + builtin FFI registration. Compression is declared per-model via `@kv_compress` and resolved at compile time. The runtime performs quantization on KV write and eviction after each decode step.

**Tech Stack:** Rust (runtime FFI + semantic + codegen)

**Spec:** `docs/superpowers/specs/2026-03-15-m42-kv-compression-design.md`

**Prerequisites:** M25 (PagedAttention — block allocator, page tables), M27 (FlashAttention-2), M29 (Continuous Batching)

---

## Important: Scope of This Plan

**This plan builds the core KV compression/eviction infrastructure.** It delivers:
- `KvQuantScheme` enum and `KvBlockQuantMeta` per-block metadata
- INT8 per-head, INT8 per-token, INT4 per-group, and FP8 quantization/dequantization (CPU-side)
- `SlidingWindowManager` with page-level eviction
- `H2OManager` with cumulative score tracking and budget-based eviction
- `LayerCompressionPolicy` for per-layer config
- FFI functions for quantize-and-store, sliding window, H2O, and eviction
- `@kv_compress` decorator semantic validation (method, bits, window, sinks, budget)
- Codegen: `kv_compress_policies` HashMap, decorator extraction, builtin registration
- Page table eviction support (`remove_block_range`)
- 25+ unit tests across all modules

**Deferred to M42b:** Quantized FlashAttention PTX variants (INT8/INT4/FP8 dequant in tile-loading phase), GPU-side quantization kernels, `SlidingWindowAttentionConfig` integration with FlashAttention's tile-skip logic, H2O score extraction from FlashAttention's online softmax (`nsl_flash_attention_with_scores`), per-layer FlashAttention kernel dispatch based on compression policy, compression-aware M29 memory budget estimation, compression-aware M41 disaggregated KV transfer, E2E numerical quality tests.

---

## File Structure

### New Files

| File | Responsibility | ~Lines |
|---|---|---|
| `crates/nsl-runtime/src/kv_compress/mod.rs` | Module declarations, `KvQuantScheme`, `LayerCompressionPolicy` | 80 |
| `crates/nsl-runtime/src/kv_compress/quantize.rs` | Quantization/dequantization, `KvBlockQuantMeta`, block size calc | 300 |
| `crates/nsl-runtime/src/kv_compress/sliding_window.rs` | `SlidingWindowManager`, page-level eviction | 180 |
| `crates/nsl-runtime/src/kv_compress/h2o.rs` | `H2OManager`, score tracking, budget eviction | 220 |
| `crates/nsl-runtime/src/kv_compress/ffi.rs` | FFI wrappers for all compression functions | 200 |
| `crates/nsl-semantic/src/kv_compress.rs` | `@kv_compress` decorator validation | 120 |

### Modified Files

| File | Change |
|---|---|
| `crates/nsl-runtime/src/lib.rs` | Add `pub mod kv_compress;` |
| `crates/nsl-runtime/src/paged_kv/page_table.rs` | Add `remove_block_range()` for eviction |
| `crates/nsl-codegen/src/compiler.rs` | Add `kv_compress_policies` field, decorator extraction |
| `crates/nsl-codegen/src/builtins.rs` | Register 8 new FFI functions |
| `crates/nsl-semantic/src/lib.rs` | Add `pub mod kv_compress;` |
| `crates/nsl-semantic/src/checker.rs` | Wire `@kv_compress` validation |

---

## Phase 1: Core Types + Quantization

### Task 1: Module Root + KvQuantScheme + LayerCompressionPolicy

**Files:**
- Create: `crates/nsl-runtime/src/kv_compress/mod.rs`
- Modify: `crates/nsl-runtime/src/lib.rs`

- [ ] **Step 1: Create `kv_compress/mod.rs` with core types and wire into lib.rs**

```rust
// crates/nsl-runtime/src/kv_compress/mod.rs
//! M42: KV-cache compression and eviction.
//!
//! Three strategies that compose:
//! - Quantized KV storage (INT8/INT4/FP8) — reduces bytes per element
//! - Sliding window with attention sinks — caps token count
//! - H2O (Heavy Hitter Oracle) — attention-score-based eviction

pub mod quantize;
pub mod sliding_window;
pub mod h2o;
pub mod ffi;

/// Quantization scheme for KV-cache storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum KvQuantScheme {
    /// No compression — native dtype (FP16/FP32).
    None = 0,
    /// INT8 symmetric quantization, one scale per attention head.
    Int8PerHead = 1,
    /// INT8 symmetric quantization, one scale per token position.
    Int8PerToken = 2,
    /// INT4 asymmetric quantization, one scale+zero per group.
    Int4PerGroup = 3,
    /// FP8 E4M3 — direct cast, no scaling metadata.
    Fp8 = 4,
}

impl KvQuantScheme {
    /// Bytes per element for this scheme.
    pub fn bytes_per_element(&self) -> f64 {
        match self {
            KvQuantScheme::None => 2.0,      // FP16
            KvQuantScheme::Int8PerHead => 1.0,
            KvQuantScheme::Int8PerToken => 1.0,
            KvQuantScheme::Int4PerGroup => 0.5,
            KvQuantScheme::Fp8 => 1.0,
        }
    }

    /// From integer discriminant (for FFI).
    pub fn from_i64(v: i64) -> Self {
        match v {
            1 => KvQuantScheme::Int8PerHead,
            2 => KvQuantScheme::Int8PerToken,
            3 => KvQuantScheme::Int4PerGroup,
            4 => KvQuantScheme::Fp8,
            _ => KvQuantScheme::None,
        }
    }
}

impl Default for KvQuantScheme {
    fn default() -> Self {
        KvQuantScheme::None
    }
}

/// Sliding window configuration.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    pub window_size: usize,
    pub num_sinks: usize,
}

/// H2O (Heavy Hitter Oracle) configuration.
#[derive(Debug, Clone)]
pub struct H2OConfig {
    pub budget: usize,
    pub num_sinks: usize,
}

/// Per-layer compression policy, resolved from @kv_compress decorators.
#[derive(Debug, Clone, Default)]
pub struct LayerCompressionPolicy {
    pub quant_scheme: KvQuantScheme,
    pub window: Option<SlidingWindowConfig>,
    pub h2o: Option<H2OConfig>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quant_scheme_bytes_per_element() {
        assert_eq!(KvQuantScheme::None.bytes_per_element(), 2.0);
        assert_eq!(KvQuantScheme::Int8PerHead.bytes_per_element(), 1.0);
        assert_eq!(KvQuantScheme::Int4PerGroup.bytes_per_element(), 0.5);
        assert_eq!(KvQuantScheme::Fp8.bytes_per_element(), 1.0);
    }

    #[test]
    fn quant_scheme_from_i64() {
        assert_eq!(KvQuantScheme::from_i64(0), KvQuantScheme::None);
        assert_eq!(KvQuantScheme::from_i64(1), KvQuantScheme::Int8PerHead);
        assert_eq!(KvQuantScheme::from_i64(2), KvQuantScheme::Int8PerToken);
        assert_eq!(KvQuantScheme::from_i64(3), KvQuantScheme::Int4PerGroup);
        assert_eq!(KvQuantScheme::from_i64(4), KvQuantScheme::Fp8);
        assert_eq!(KvQuantScheme::from_i64(99), KvQuantScheme::None);
    }

    #[test]
    fn layer_policy_default_is_none() {
        let p = LayerCompressionPolicy::default();
        assert_eq!(p.quant_scheme, KvQuantScheme::None);
        assert!(p.window.is_none());
        assert!(p.h2o.is_none());
    }
}
```

Add to `crates/nsl-runtime/src/lib.rs`:
```rust
pub mod kv_compress;
```

### Task 2: Quantization Module

**Files:**
- Create: `crates/nsl-runtime/src/kv_compress/quantize.rs`

- [ ] **Step 2: Create `quantize.rs` with quantize/dequantize algorithms, metadata struct, and block size calculation**

```rust
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
        KvQuantScheme::Int4PerGroup => (elements + 1) / 2, // INT4: 0.5 bytes, round up
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
    meta.scheme = KvQuantScheme::Int4PerGroup as u8;
    let num_groups = (values.len() + group_size - 1) / group_size;
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
            if (start + i) % 2 == 0 {
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
    for i in 0..num_elements {
        let byte_idx = i / 2;
        let nibble = if i % 2 == 0 {
            packed[byte_idx] & 0x0F
        } else {
            (packed[byte_idx] >> 4) & 0x0F
        };
        let g = i / group_size;
        let scale = meta.scales[g];
        let min_val = meta.zero_points[g]; // f32 min_val stored directly
        output[i] = nibble as f32 * scale + min_val;
    }
}

// ---------------------------------------------------------------------------
// FP8 E4M3 conversion
// ---------------------------------------------------------------------------

/// Convert f32 to FP8 E4M3 (truncation).
///
/// FP8 E4M3: 1 sign + 4 exponent + 3 mantissa, range ≈ [-448, 448].
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

/// Convert a single f32 to FP8 E4M3.
fn f32_to_fp8_e4m3(v: f32) -> u8 {
    let clamped = v.clamp(-448.0, 448.0);
    let bits = clamped.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32 - 127; // f32 exponent bias
    let mantissa = bits & 0x7F_FFFF;

    if clamped == 0.0 || exp < -6 {
        return (sign << 7) as u8; // zero or denorm → zero
    }
    // E4M3: bias = 7, exp range [-6, 8]
    let biased_exp = (exp + 7).clamp(0, 15) as u8;
    let m3 = (mantissa >> 20) as u8; // top 3 mantissa bits
    ((sign as u8) << 7) | (biased_exp << 3) | m3
}

/// Convert a single FP8 E4M3 to f32.
fn fp8_e4m3_to_f32(b: u8) -> f32 {
    let sign = (b >> 7) & 1;
    let exp = ((b >> 3) & 0x0F) as i32;
    let mantissa = (b & 0x07) as u32;

    if exp == 0 && mantissa == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }
    let f32_exp = (exp - 7 + 127) as u32; // unbias E4M3, rebias f32
    let f32_mantissa = mantissa << 20;     // position in f32 mantissa
    let bits = ((sign as u32) << 31) | (f32_exp << 23) | f32_mantissa;
    f32::from_bits(bits)
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
        let packed_len = (values.len() + 1) / 2;
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
        let values = vec![0.0, 1.0, -1.0, 0.5, -0.5, 100.0, -100.0, 448.0];
        let mut packed = vec![0u8; values.len()];
        quantize_fp8(&values, &mut packed);

        let mut restored = vec![0.0f32; values.len()];
        dequantize_fp8(&packed, &mut restored);

        // FP8 E4M3 has limited precision — check within ~10% for non-zero
        assert_eq!(restored[0], 0.0);
        assert!((restored[1] - 1.0).abs() < 0.2);
        assert!((restored[2] + 1.0).abs() < 0.2);
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
```

---

## Phase 2: Eviction Managers

### Task 3: Sliding Window Manager

**Files:**
- Create: `crates/nsl-runtime/src/kv_compress/sliding_window.rs`

- [ ] **Step 3: Create `sliding_window.rs` with page-level eviction logic**

```rust
// crates/nsl-runtime/src/kv_compress/sliding_window.rs
//! Sliding window KV-cache eviction with attention sinks.
//!
//! Retains the first `num_sinks` tokens (attention sinks) plus the most recent
//! `window_size` tokens. Everything in between is evicted at page granularity.

/// Manages sliding window eviction for one compression policy.
pub struct SlidingWindowManager {
    pub window_size: usize,
    pub num_sinks: usize,
    pub block_size: usize,
}

impl SlidingWindowManager {
    pub fn new(window_size: usize, num_sinks: usize, block_size: usize) -> Self {
        SlidingWindowManager { window_size, num_sinks, block_size }
    }

    /// Check if eviction is needed. Returns logical block indices to evict.
    ///
    /// A block is evicted only if ALL tokens in it fall in the eviction range
    /// (between sinks and the start of the window).
    pub fn check_eviction(&self, current_len: usize) -> Vec<usize> {
        if current_len <= self.num_sinks + self.window_size {
            return vec![];
        }

        let evict_start = self.num_sinks;
        let evict_end = current_len - self.window_size;

        let mut to_evict = Vec::new();
        // Only evict full blocks where every token is in [evict_start, evict_end)
        let start_block = (evict_start + self.block_size - 1) / self.block_size; // ceil
        let end_block = evict_end / self.block_size; // floor

        for block_idx in start_block..end_block {
            let block_start_token = block_idx * self.block_size;
            let block_end_token = (block_idx + 1) * self.block_size;
            if block_start_token >= evict_start && block_end_token <= evict_end {
                to_evict.push(block_idx);
            }
        }
        to_evict
    }

    /// Active token count (sinks + window), capped at current_len.
    pub fn active_tokens(&self, current_len: usize) -> usize {
        if current_len <= self.num_sinks + self.window_size {
            current_len
        } else {
            self.num_sinks + self.window_size
        }
    }

    /// The attention config ranges for this window state.
    ///
    /// Returns (sink_end, window_start, window_end) for FlashAttention tile-skip.
    pub fn attention_ranges(&self, current_len: usize) -> (usize, usize, usize) {
        let sink_end = self.num_sinks.min(current_len);
        if current_len <= self.num_sinks + self.window_size {
            (sink_end, 0, current_len) // no eviction yet — attend to everything
        } else {
            let window_start = current_len - self.window_size;
            (sink_end, window_start, current_len)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_eviction_when_within_budget() {
        let mgr = SlidingWindowManager::new(32, 4, 8);
        assert!(mgr.check_eviction(36).is_empty()); // exactly at budget
        assert!(mgr.check_eviction(20).is_empty());
        assert!(mgr.check_eviction(0).is_empty());
    }

    #[test]
    fn eviction_correct_blocks() {
        // window=32, sinks=4, block_size=8
        // At 100 tokens: sinks=[0..3], window=[68..99], evict=[4..67]
        // Block 0: tokens 0-7 (contains sinks) — NOT evicted
        // Block 1: tokens 8-15 — evicted (all in [4..67])
        // Block 2: tokens 16-23 — evicted
        // ...
        // Block 8: tokens 64-71 — NOT evicted (overlaps window start 68)
        let mgr = SlidingWindowManager::new(32, 4, 8);
        let evicted = mgr.check_eviction(100);
        // Blocks 1..8 (indices 1,2,3,4,5,6,7) — block 0 has sinks, block 8 straddles
        assert_eq!(evicted, vec![1, 2, 3, 4, 5, 6, 7]);
    }

    #[test]
    fn partial_block_not_evicted() {
        // window=10, sinks=2, block_size=8
        // At 20 tokens: evict=[2..9], which is 8 tokens
        // Block 0: 0-7 (has sinks 0-1, also has evictable 2-7) — NOT evicted (sinks protect it)
        // Block 1: 8-15 (has token 8-9 in evict range, 10-15 in window) — NOT evicted
        let mgr = SlidingWindowManager::new(10, 2, 8);
        let evicted = mgr.check_eviction(20);
        assert!(evicted.is_empty()); // no full block is entirely in the evict range
    }

    #[test]
    fn large_eviction() {
        // window=16, sinks=0, block_size=4
        // At 100 tokens: evict=[0..83]
        // Blocks 0..20 are fully evicted (tokens 0..83 → 84/4 = 21 blocks)
        let mgr = SlidingWindowManager::new(16, 0, 4);
        let evicted = mgr.check_eviction(100);
        assert_eq!(evicted.len(), 21); // blocks 0..20
    }

    #[test]
    fn active_tokens_capped() {
        let mgr = SlidingWindowManager::new(32, 4, 8);
        assert_eq!(mgr.active_tokens(10), 10);
        assert_eq!(mgr.active_tokens(36), 36);
        assert_eq!(mgr.active_tokens(100), 36); // sinks + window
        assert_eq!(mgr.active_tokens(10000), 36);
    }

    #[test]
    fn attention_ranges_before_eviction() {
        let mgr = SlidingWindowManager::new(32, 4, 8);
        let (sink_end, window_start, window_end) = mgr.attention_ranges(20);
        assert_eq!(sink_end, 4);
        assert_eq!(window_start, 0); // no eviction
        assert_eq!(window_end, 20);
    }

    #[test]
    fn attention_ranges_after_eviction() {
        let mgr = SlidingWindowManager::new(32, 4, 8);
        let (sink_end, window_start, window_end) = mgr.attention_ranges(100);
        assert_eq!(sink_end, 4);
        assert_eq!(window_start, 68);
        assert_eq!(window_end, 100);
    }
}
```

### Task 4: H2O Manager

**Files:**
- Create: `crates/nsl-runtime/src/kv_compress/h2o.rs`

- [ ] **Step 4: Create `h2o.rs` with cumulative score tracking and budget-based eviction**

```rust
// crates/nsl-runtime/src/kv_compress/h2o.rs
//! Heavy Hitter Oracle (H2O) — attention-score-based KV eviction.
//!
//! Maintains cumulative attention scores per KV position. When the sequence
//! exceeds the budget, positions with the lowest scores are evicted (except sinks).

use std::collections::HashMap;

/// H2O eviction manager for attention-score-based KV pruning.
pub struct H2OManager {
    pub budget: usize,
    pub num_sinks: usize,
    pub block_size: usize,
    /// Per-sequence cumulative scores: scores[seq_id][token_pos] = cumulative attention weight.
    scores: HashMap<u64, Vec<f32>>,
}

impl H2OManager {
    pub fn new(budget: usize, num_sinks: usize, block_size: usize) -> Self {
        H2OManager {
            budget,
            num_sinks,
            block_size,
            scores: HashMap::new(),
        }
    }

    /// Accumulate attention scores for a sequence after one decode step.
    ///
    /// `scores_for_step`: [seq_len] attention weights from the latest query position,
    /// averaged across all layers and heads.
    pub fn accumulate_scores(&mut self, seq_id: u64, scores_for_step: &[f32]) {
        let cumulative = self.scores.entry(seq_id).or_insert_with(Vec::new);

        // Extend if sequence grew
        if cumulative.len() < scores_for_step.len() {
            cumulative.resize(scores_for_step.len(), 0.0);
        }

        for (pos, &weight) in scores_for_step.iter().enumerate() {
            cumulative[pos] += weight;
        }
    }

    /// Check if eviction is needed. Returns logical block indices to evict.
    ///
    /// Evicts tokens with lowest cumulative scores (except sinks).
    /// Only evicts full blocks where ALL tokens are marked for eviction.
    pub fn check_eviction(&self, seq_id: u64, current_len: usize) -> Vec<usize> {
        if current_len <= self.budget {
            return vec![];
        }

        let scores = match self.scores.get(&seq_id) {
            Some(s) => s,
            None => return vec![],
        };

        // Collect non-sink positions with their scores
        let mut candidates: Vec<(usize, f32)> = scores.iter()
            .enumerate()
            .filter(|(pos, _)| *pos >= self.num_sinks && *pos < current_len)
            .map(|(pos, &score)| (pos, score))
            .collect();

        // Sort ascending by score (lowest first = evict first)
        candidates.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let num_to_evict = current_len - self.budget;
        let evict_set: std::collections::HashSet<usize> = candidates.iter()
            .take(num_to_evict)
            .map(|(pos, _)| *pos)
            .collect();

        // Find blocks where ALL tokens are in the evict set
        let num_blocks = (current_len + self.block_size - 1) / self.block_size;
        let mut blocks_to_evict = Vec::new();
        for block_idx in 0..num_blocks {
            let block_start = block_idx * self.block_size;
            let block_end = ((block_idx + 1) * self.block_size).min(current_len);

            // Skip sink blocks
            if block_start < self.num_sinks {
                continue;
            }

            let all_evicted = (block_start..block_end).all(|pos| evict_set.contains(&pos));
            if all_evicted {
                blocks_to_evict.push(block_idx);
            }
        }

        blocks_to_evict
    }

    /// Remove tracking data for a completed sequence.
    pub fn remove_sequence(&mut self, seq_id: u64) {
        self.scores.remove(&seq_id);
    }

    /// Get cumulative scores for a sequence (for debugging/profiling).
    pub fn get_scores(&self, seq_id: u64) -> Option<&[f32]> {
        self.scores.get(&seq_id).map(|v| v.as_slice())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_eviction_within_budget() {
        let mgr = H2OManager::new(10, 2, 4);
        assert!(mgr.check_eviction(0, 5).is_empty());
        assert!(mgr.check_eviction(0, 10).is_empty());
    }

    #[test]
    fn score_accumulation() {
        let mut mgr = H2OManager::new(100, 0, 4);
        mgr.accumulate_scores(0, &[1.0, 2.0, 3.0]);
        mgr.accumulate_scores(0, &[0.5, 0.5, 0.5]);

        let scores = mgr.get_scores(0).unwrap();
        assert_eq!(scores, &[1.5, 2.5, 3.5]);
    }

    #[test]
    fn score_extends_on_growth() {
        let mut mgr = H2OManager::new(100, 0, 4);
        mgr.accumulate_scores(0, &[1.0, 2.0]);
        mgr.accumulate_scores(0, &[0.5, 0.5, 3.0, 4.0]);

        let scores = mgr.get_scores(0).unwrap();
        assert_eq!(scores, &[1.5, 2.5, 3.0, 4.0]);
    }

    #[test]
    fn eviction_order_lowest_first() {
        // budget=4, sinks=0, block_size=1 (for precise per-token eviction)
        let mut mgr = H2OManager::new(4, 0, 1);

        // 6 tokens with scores: token 2 and 4 have lowest scores
        mgr.accumulate_scores(0, &[5.0, 3.0, 1.0, 4.0, 0.5, 6.0]);

        let evicted = mgr.check_eviction(0, 6);
        // Need to evict 2 tokens (6 - 4 = 2). Lowest: pos 4 (0.5), pos 2 (1.0)
        // With block_size=1, blocks = token positions
        assert_eq!(evicted.len(), 2);
        assert!(evicted.contains(&2));
        assert!(evicted.contains(&4));
    }

    #[test]
    fn sinks_protected_from_eviction() {
        // budget=3, sinks=2, block_size=1
        let mut mgr = H2OManager::new(3, 2, 1);

        // Sinks (pos 0,1) have lowest scores but must not be evicted
        mgr.accumulate_scores(0, &[0.1, 0.2, 5.0, 3.0, 1.0]);

        let evicted = mgr.check_eviction(0, 5);
        // Need to evict 2 (5 - 3 = 2). Candidates (non-sink): pos 2(5.0), 3(3.0), 4(1.0)
        // Lowest non-sink: pos 4(1.0), pos 3(3.0)
        assert_eq!(evicted.len(), 2);
        assert!(evicted.contains(&3));
        assert!(evicted.contains(&4));
        // Sinks must NOT be evicted
        assert!(!evicted.contains(&0));
        assert!(!evicted.contains(&1));
    }

    #[test]
    fn block_level_eviction_only_full_blocks() {
        // budget=8, sinks=0, block_size=4
        let mut mgr = H2OManager::new(8, 0, 4);

        // 16 tokens: tokens 0-3 low scores, 4-7 high, 8-11 low, 12-15 high
        let scores: Vec<f32> = (0..16).map(|i| {
            if i < 4 || (8..12).contains(&i) { 0.1 } else { 10.0 }
        }).collect();
        mgr.accumulate_scores(0, &scores);

        let evicted = mgr.check_eviction(0, 16);
        // Need to evict 8 tokens. Lowest 8: tokens 0-3 and 8-11.
        // Block 0 (tokens 0-3): all evicted → evict block
        // Block 2 (tokens 8-11): all evicted → evict block
        assert_eq!(evicted.len(), 2);
        assert!(evicted.contains(&0));
        assert!(evicted.contains(&2));
    }

    #[test]
    fn remove_sequence_clears_data() {
        let mut mgr = H2OManager::new(100, 0, 4);
        mgr.accumulate_scores(42, &[1.0, 2.0]);
        assert!(mgr.get_scores(42).is_some());
        mgr.remove_sequence(42);
        assert!(mgr.get_scores(42).is_none());
    }
}
```

---

## Phase 3: FFI + Page Table Eviction

### Task 5: FFI Functions

**Files:**
- Create: `crates/nsl-runtime/src/kv_compress/ffi.rs`

- [ ] **Step 5: Create `ffi.rs` with all compression FFI functions**

```rust
// crates/nsl-runtime/src/kv_compress/ffi.rs
//! M42: FFI exports for KV-cache compression.

use std::sync::Mutex;

use super::KvQuantScheme;
use super::quantize::{self, KvBlockQuantMeta};
use super::sliding_window::SlidingWindowManager;
use super::h2o::H2OManager;

// ---------------------------------------------------------------------------
// Quantization FFI
// ---------------------------------------------------------------------------

/// Quantize incoming K/V values and store in compressed format.
///
/// Parameters (all i64 for Cranelift):
/// - raw_k/raw_v: pointers to f32 K/V values for new tokens
/// - block_k/block_v: pointers to target blocks (u8 for quantized, f32 for none)
/// - meta_k/meta_v: pointers to KvBlockQuantMeta (null for Fp8/None)
/// - token_offset: position within block
/// - num_tokens: tokens to quantize
/// - num_heads: attention heads
/// - head_dim: dimension per head
/// - scheme: KvQuantScheme discriminant
///
/// Returns 0 on success, -1 on error.
#[no_mangle]
pub extern "C" fn nsl_kv_quantize_and_store(
    raw_k: i64,
    raw_v: i64,
    block_k: i64,
    block_v: i64,
    meta_k: i64,
    meta_v: i64,
    _token_offset: i64,
    num_tokens: i64,
    num_heads: i64,
    head_dim: i64,
    scheme: i64,
) -> i64 {
    let qs = KvQuantScheme::from_i64(scheme);
    let n = num_heads as usize * num_tokens as usize * head_dim as usize;

    if raw_k == 0 || raw_v == 0 || block_k == 0 || block_v == 0 {
        return -1;
    }

    // Read input values
    let k_values = unsafe { std::slice::from_raw_parts(raw_k as *const f32, n) };
    let v_values = unsafe { std::slice::from_raw_parts(raw_v as *const f32, n) };

    match qs {
        KvQuantScheme::None => {
            // Direct copy (no quantization)
            unsafe {
                std::ptr::copy_nonoverlapping(raw_k as *const u8, block_k as *mut u8, n * 4);
                std::ptr::copy_nonoverlapping(raw_v as *const u8, block_v as *mut u8, n * 4);
            }
        }
        KvQuantScheme::Int8PerHead => {
            let k_out = unsafe { std::slice::from_raw_parts_mut(block_k as *mut i8, n) };
            let v_out = unsafe { std::slice::from_raw_parts_mut(block_v as *mut i8, n) };
            let k_meta = unsafe { &mut *(meta_k as *mut KvBlockQuantMeta) };
            let v_meta = unsafe { &mut *(meta_v as *mut KvBlockQuantMeta) };
            quantize::quantize_int8_per_head(k_values, k_out, k_meta, num_heads as usize, num_tokens as usize, head_dim as usize);
            quantize::quantize_int8_per_head(v_values, v_out, v_meta, num_heads as usize, num_tokens as usize, head_dim as usize);
        }
        KvQuantScheme::Int8PerToken => {
            let k_out = unsafe { std::slice::from_raw_parts_mut(block_k as *mut i8, n) };
            let v_out = unsafe { std::slice::from_raw_parts_mut(block_v as *mut i8, n) };
            let k_meta = unsafe { &mut *(meta_k as *mut KvBlockQuantMeta) };
            let v_meta = unsafe { &mut *(meta_v as *mut KvBlockQuantMeta) };
            quantize::quantize_int8_per_token(k_values, k_out, k_meta, num_heads as usize, num_tokens as usize, head_dim as usize);
            quantize::quantize_int8_per_token(v_values, v_out, v_meta, num_heads as usize, num_tokens as usize, head_dim as usize);
        }
        KvQuantScheme::Int4PerGroup => {
            let packed_len = (n + 1) / 2;
            let k_out = unsafe { std::slice::from_raw_parts_mut(block_k as *mut u8, packed_len) };
            let v_out = unsafe { std::slice::from_raw_parts_mut(block_v as *mut u8, packed_len) };
            let k_meta = unsafe { &mut *(meta_k as *mut KvBlockQuantMeta) };
            let v_meta = unsafe { &mut *(meta_v as *mut KvBlockQuantMeta) };
            quantize::quantize_int4_per_group(k_values, k_out, k_meta, quantize::INT4_GROUP_SIZE);
            quantize::quantize_int4_per_group(v_values, v_out, v_meta, quantize::INT4_GROUP_SIZE);
        }
        KvQuantScheme::Fp8 => {
            let k_out = unsafe { std::slice::from_raw_parts_mut(block_k as *mut u8, n) };
            let v_out = unsafe { std::slice::from_raw_parts_mut(block_v as *mut u8, n) };
            quantize::quantize_fp8(k_values, k_out);
            quantize::quantize_fp8(v_values, v_out);
        }
    }
    0
}

// ---------------------------------------------------------------------------
// Sliding Window FFI
// ---------------------------------------------------------------------------

static SW_CTX: Mutex<Option<SlidingWindowManager>> = Mutex::new(None);

/// Initialize sliding window manager.
/// Returns 0 on success, -1 if already initialized.
#[no_mangle]
pub extern "C" fn nsl_kv_sliding_window_init(
    window: i64,
    sinks: i64,
    block_size: i64,
) -> i64 {
    let mut guard = SW_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    *guard = Some(SlidingWindowManager::new(
        window as usize,
        sinks as usize,
        block_size as usize,
    ));
    0
}

/// Check sliding window eviction. Returns number of blocks to evict.
/// Evicted block indices are written to `evict_out_ptr` if non-null.
#[no_mangle]
pub extern "C" fn nsl_kv_sliding_window_check(
    _seq_id: i64,
    current_len: i64,
    evict_out_ptr: i64,
    max_evict: i64,
) -> i64 {
    let guard = SW_CTX.lock().unwrap();
    let mgr = guard.as_ref().expect("nsl_kv_sliding_window_init not called");

    let evicted = mgr.check_eviction(current_len as usize);
    let count = evicted.len().min(max_evict as usize);

    if evict_out_ptr != 0 && count > 0 {
        let out = unsafe { std::slice::from_raw_parts_mut(evict_out_ptr as *mut u32, count) };
        for (i, &block_idx) in evicted.iter().take(count).enumerate() {
            out[i] = block_idx as u32;
        }
    }
    count as i64
}

/// Destroy sliding window manager.
#[no_mangle]
pub extern "C" fn nsl_kv_sliding_window_destroy() -> i64 {
    let mut guard = SW_CTX.lock().unwrap();
    *guard = None;
    0
}

// ---------------------------------------------------------------------------
// H2O FFI
// ---------------------------------------------------------------------------

static H2O_CTX: Mutex<Option<H2OManager>> = Mutex::new(None);

/// Initialize H2O manager.
/// Returns 0 on success, -1 if already initialized.
#[no_mangle]
pub extern "C" fn nsl_kv_h2o_init(budget: i64, sinks: i64, block_size: i64) -> i64 {
    let mut guard = H2O_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    *guard = Some(H2OManager::new(
        budget as usize,
        sinks as usize,
        block_size as usize,
    ));
    0
}

/// Accumulate attention scores for a sequence.
/// scores_ptr: *const f32, [seq_len] averaged scores from latest decode step.
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_kv_h2o_accumulate(
    seq_id: i64,
    scores_ptr: i64,
    seq_len: i64,
) -> i64 {
    let mut guard = H2O_CTX.lock().unwrap();
    let mgr = guard.as_mut().expect("nsl_kv_h2o_init not called");

    if scores_ptr == 0 || seq_len <= 0 {
        return -1;
    }

    let scores = unsafe { std::slice::from_raw_parts(scores_ptr as *const f32, seq_len as usize) };
    mgr.accumulate_scores(seq_id as u64, scores);
    0
}

/// Check H2O eviction. Returns number of blocks to evict.
#[no_mangle]
pub extern "C" fn nsl_kv_h2o_check(
    seq_id: i64,
    current_len: i64,
    evict_out_ptr: i64,
    max_evict: i64,
) -> i64 {
    let guard = H2O_CTX.lock().unwrap();
    let mgr = guard.as_ref().expect("nsl_kv_h2o_init not called");

    let evicted = mgr.check_eviction(seq_id as u64, current_len as usize);
    let count = evicted.len().min(max_evict as usize);

    if evict_out_ptr != 0 && count > 0 {
        let out = unsafe { std::slice::from_raw_parts_mut(evict_out_ptr as *mut u32, count) };
        for (i, &block_idx) in evicted.iter().take(count).enumerate() {
            out[i] = block_idx as u32;
        }
    }
    count as i64
}

/// Remove sequence tracking from H2O manager.
#[no_mangle]
pub extern "C" fn nsl_kv_h2o_remove_sequence(seq_id: i64) -> i64 {
    let mut guard = H2O_CTX.lock().unwrap();
    let mgr = guard.as_mut().expect("nsl_kv_h2o_init not called");
    mgr.remove_sequence(seq_id as u64);
    0
}

/// Destroy H2O manager.
#[no_mangle]
pub extern "C" fn nsl_kv_h2o_destroy() -> i64 {
    let mut guard = H2O_CTX.lock().unwrap();
    *guard = None;
    0
}

// ---------------------------------------------------------------------------
// Compression statistics
// ---------------------------------------------------------------------------

/// Get compression ratio for the active scheme.
/// Returns the compression ratio as f64 bits (e.g., 2.0 for INT8 vs FP16).
#[no_mangle]
pub extern "C" fn nsl_kv_compress_ratio(scheme: i64) -> i64 {
    let qs = KvQuantScheme::from_i64(scheme);
    let ratio = 2.0 / qs.bytes_per_element(); // relative to FP16 baseline
    f64::to_bits(ratio) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        nsl_kv_sliding_window_destroy();
        nsl_kv_h2o_destroy();
        guard
    }

    #[test]
    fn ffi_quantize_none_copies() {
        let k = vec![1.0f32, 2.0, 3.0, 4.0];
        let v = vec![5.0f32, 6.0, 7.0, 8.0];
        let mut k_out = vec![0.0f32; 4];
        let mut v_out = vec![0.0f32; 4];

        let rc = nsl_kv_quantize_and_store(
            k.as_ptr() as i64, v.as_ptr() as i64,
            k_out.as_mut_ptr() as i64, v_out.as_mut_ptr() as i64,
            0, 0, // no metadata for None
            0, 1, 1, 4, // offset=0, tokens=1, heads=1, dim=4
            0, // scheme=None
        );
        assert_eq!(rc, 0);
        assert_eq!(k_out, k);
        assert_eq!(v_out, v);
    }

    #[test]
    fn ffi_quantize_int8() {
        let n = 2 * 1 * 4; // heads=2, tokens=1, dim=4
        let k: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let v: Vec<f32> = (0..n).map(|i| i as f32 * -0.3).collect();
        let mut k_out = vec![0i8; n];
        let mut v_out = vec![0i8; n];
        let mut k_meta = KvBlockQuantMeta::default();
        let mut v_meta = KvBlockQuantMeta::default();

        let rc = nsl_kv_quantize_and_store(
            k.as_ptr() as i64, v.as_ptr() as i64,
            k_out.as_mut_ptr() as i64, v_out.as_mut_ptr() as i64,
            &mut k_meta as *mut _ as i64, &mut v_meta as *mut _ as i64,
            0, 1, 2, 4, // offset=0, tokens=1, heads=2, dim=4
            1, // scheme=Int8PerHead
        );
        assert_eq!(rc, 0);
        assert_eq!(k_meta.scheme, 1);
        assert_eq!(k_meta.num_scales, 2);
    }

    #[test]
    fn ffi_sliding_window_lifecycle() {
        let _lock = setup();

        assert_eq!(nsl_kv_sliding_window_init(32, 4, 8), 0);
        assert_eq!(nsl_kv_sliding_window_init(32, 4, 8), -1); // double init

        // No eviction at 30 tokens
        let count = nsl_kv_sliding_window_check(0, 30, 0, 0);
        assert_eq!(count, 0);

        // Eviction at 100 tokens
        let mut out = vec![0u32; 16];
        let count = nsl_kv_sliding_window_check(0, 100, out.as_mut_ptr() as i64, 16);
        assert!(count > 0);

        assert_eq!(nsl_kv_sliding_window_destroy(), 0);
    }

    #[test]
    fn ffi_h2o_lifecycle() {
        let _lock = setup();

        assert_eq!(nsl_kv_h2o_init(4, 0, 1), 0);
        assert_eq!(nsl_kv_h2o_init(4, 0, 1), -1); // double init

        // Accumulate scores
        let scores = vec![5.0f32, 3.0, 1.0, 4.0, 0.5, 6.0];
        let rc = nsl_kv_h2o_accumulate(0, scores.as_ptr() as i64, 6);
        assert_eq!(rc, 0);

        // Check eviction (6 tokens, budget=4 → evict 2)
        let mut out = vec![0u32; 4];
        let count = nsl_kv_h2o_check(0, 6, out.as_mut_ptr() as i64, 4);
        assert_eq!(count, 2);

        assert_eq!(nsl_kv_h2o_remove_sequence(0), 0);
        assert_eq!(nsl_kv_h2o_destroy(), 0);
    }

    #[test]
    fn ffi_null_pointer_returns_error() {
        let rc = nsl_kv_quantize_and_store(0, 0, 0, 0, 0, 0, 0, 1, 1, 4, 0);
        assert_eq!(rc, -1);
    }

    #[test]
    fn ffi_compress_ratio() {
        let ratio_none = f64::from_bits(nsl_kv_compress_ratio(0) as u64);
        assert_eq!(ratio_none, 1.0); // FP16/FP16

        let ratio_int8 = f64::from_bits(nsl_kv_compress_ratio(1) as u64);
        assert_eq!(ratio_int8, 2.0); // FP16/INT8

        let ratio_int4 = f64::from_bits(nsl_kv_compress_ratio(3) as u64);
        assert_eq!(ratio_int4, 4.0); // FP16/INT4
    }
}
```

### Task 6: Page Table Eviction Support

**Files:**
- Modify: `crates/nsl-runtime/src/paged_kv/page_table.rs`

- [ ] **Step 6: Add `remove_block_range()` to PageTable**

Add methods to `PageTable` for eviction support. The field is `self.entries` (not `self.blocks`). Use `u32::MAX` as a sentinel for evicted slots:

```rust
/// Sentinel value for evicted block slots.
pub const EVICTED_SENTINEL: BlockId = u32::MAX;

/// Remove blocks in the logical index range [start_idx, end_idx).
/// Returns the physical BlockIds that were removed.
/// Evicted slots are set to EVICTED_SENTINEL.
pub fn remove_block_range(&mut self, start_idx: usize, end_idx: usize) -> Vec<BlockId> {
    let mut removed = Vec::new();
    let end = end_idx.min(self.entries.len());
    for i in start_idx..end {
        if self.entries[i] != EVICTED_SENTINEL {
            removed.push(self.entries[i]);
            self.entries[i] = EVICTED_SENTINEL;
        }
    }
    removed
}

/// Check if a logical block slot has been evicted.
pub fn is_evicted(&self, logical_idx: usize) -> bool {
    logical_idx < self.entries.len() && self.entries[logical_idx] == EVICTED_SENTINEL
}
```

Also update the existing `get_block` method to return `None` for sentinel entries:
```rust
// In the existing get_block method, add a sentinel check:
pub fn get_block(&self, logical_index: usize) -> Option<BlockId> {
    self.entries.get(logical_index)
        .copied()
        .filter(|&id| id != EVICTED_SENTINEL)
}
```

---

## Phase 4: Semantic Validation + Codegen + Builtins

### Task 7: Semantic Validation

**Files:**
- Create: `crates/nsl-semantic/src/kv_compress.rs`
- Modify: `crates/nsl-semantic/src/lib.rs`
- Modify: `crates/nsl-semantic/src/checker.rs`

- [ ] **Step 7: Create `kv_compress.rs` validation module and wire into checker**

```rust
// crates/nsl-semantic/src/kv_compress.rs
//! M42: @kv_compress decorator validation.

use nsl_ast::block::Decorator;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Validated KV compression configuration from a single @kv_compress decorator.
#[derive(Debug, Clone)]
pub struct KvCompressConfig {
    pub method: String,
    pub bits: usize,
    pub dtype: String,
    pub granularity: String,
    pub group_size: usize,
    pub window_size: usize,
    pub num_sinks: usize,
    pub budget: usize,
}

/// Validate a @kv_compress decorator and return the parsed config.
pub fn validate_kv_compress_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<KvCompressConfig> {
    let mut method = String::new();
    let mut bits: usize = 8;
    let mut dtype = String::new();
    let mut granularity = "per_head".to_string();
    let mut group_size: usize = 64;
    let mut window_size: usize = 4096;
    let mut num_sinks: usize = 32;
    let mut budget: usize = 2048;

    // Parse keyword arguments (deco.args is Option<Vec<Arg>>, matches moe.rs pattern)
    if let Some(ref args) = deco.args {
        for arg in args {
            let key = if let Some(ref name_sym) = arg.name {
                resolve_sym(*name_sym)
            } else {
                continue;
            };
            match key.as_str() {
                "method" => {
                    if let nsl_ast::expr::ExprKind::StringLiteral(s) = &arg.value.kind {
                        method = s.clone();
                    }
                }
                "bits" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &arg.value.kind {
                        bits = *v as usize;
                    }
                }
                "dtype" => {
                    if let nsl_ast::expr::ExprKind::StringLiteral(s) = &arg.value.kind {
                        dtype = s.clone();
                    }
                }
                "granularity" => {
                    if let nsl_ast::expr::ExprKind::StringLiteral(s) = &arg.value.kind {
                        granularity = s.clone();
                    }
                }
                "group_size" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &arg.value.kind {
                        group_size = *v as usize;
                    }
                }
                "window" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &arg.value.kind {
                        window_size = *v as usize;
                    }
                }
                "sinks" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &arg.value.kind {
                        num_sinks = *v as usize;
                    }
                }
                "budget" => {
                    if let nsl_ast::expr::ExprKind::IntLiteral(v) = &arg.value.kind {
                        budget = *v as usize;
                    }
                }
                other => {
                    diagnostics.push(
                        Diagnostic::error(format!("unknown @kv_compress parameter '{other}'"))
                            .with_label(arg.value.span, "here"),
                    );
                    return None;
                }
            }
        }
    }

    // Validate method
    if method.is_empty() {
        diagnostics.push(
            Diagnostic::error("@kv_compress requires a 'method' parameter")
                .with_label(deco.span, "missing method"),
        );
        return None;
    }

    match method.as_str() {
        "quantize" => {
            if !dtype.is_empty() {
                if !["int8", "int4", "fp8"].contains(&dtype.as_str()) {
                    diagnostics.push(
                        Diagnostic::error(format!("unsupported kv_compress dtype '{dtype}', expected: int8, int4, fp8"))
                            .with_label(deco.span, "here"),
                    );
                    return None;
                }
            } else if ![4, 8].contains(&bits) {
                diagnostics.push(
                    Diagnostic::error("kv_compress quantize bits must be 4 or 8")
                        .with_label(deco.span, "here"),
                );
                return None;
            }
            if !["per_head", "per_token", "per_group"].contains(&granularity.as_str()) {
                diagnostics.push(
                    Diagnostic::error(format!("unknown granularity '{granularity}', expected: per_head, per_token, per_group"))
                        .with_label(deco.span, "here"),
                );
                return None;
            }
        }
        "sliding_window" => {
            if num_sinks >= window_size {
                diagnostics.push(
                    Diagnostic::error("sinks must be less than window size")
                        .with_label(deco.span, "here"),
                );
                return None;
            }
        }
        "h2o" => {
            if num_sinks >= budget {
                diagnostics.push(
                    Diagnostic::error("sinks must be less than budget")
                        .with_label(deco.span, "here"),
                );
                return None;
            }
        }
        other => {
            diagnostics.push(
                Diagnostic::error(format!("unknown kv_compress method '{other}', expected: quantize, sliding_window, h2o"))
                    .with_label(deco.span, "here"),
            );
            return None;
        }
    }

    Some(KvCompressConfig {
        method,
        bits,
        dtype,
        granularity,
        group_size,
        window_size,
        num_sinks,
        budget,
    })
}
```

Wire into `checker.rs` in **two** locations:

1. **Model field decorator section** (near `@moe`, `@speculative`, `@shard` — around line 1060):
```rust
if dname == "kv_compress" {
    let resolve = |s: nsl_ast::Symbol| -> String { self.resolve_sym(s).to_string() };
    crate::kv_compress::validate_kv_compress_decorator(deco, &resolve, &mut self.diagnostics);
}
```

2. **Statement-level decorator section** (for model-level `@kv_compress` applied to the model statement itself):
```rust
if dname == "kv_compress" {
    let resolve = |s: nsl_ast::Symbol| -> String { self.resolve_sym(s).to_string() };
    crate::kv_compress::validate_kv_compress_decorator(deco, &resolve, &mut self.diagnostics);
}
```

Also add **H2O + sliding_window conflict warning** (spec Section 8 requirement) — after validating all `@kv_compress` decorators on a model field, check for conflicting eviction methods:
```rust
// After processing all @kv_compress decorators on a field:
let mut has_h2o = false;
let mut has_sliding = false;
for deco in field_decorators {
    if dname == "kv_compress" {
        // ... (existing validation) ...
        if method == "h2o" { has_h2o = true; }
        if method == "sliding_window" { has_sliding = true; }
    }
}
if has_h2o && has_sliding {
    diagnostics.push(
        Diagnostic::warning("@kv_compress: h2o and sliding_window on same layer group will conflict — both perform eviction")
            .with_label(field_span, "here"),
    );
}
```

Wire into `lib.rs`:
```rust
pub mod kv_compress;
```

### Task 8: Codegen + Builtins

**Files:**
- Modify: `crates/nsl-codegen/src/compiler.rs`
- Modify: `crates/nsl-codegen/src/builtins.rs`

- [ ] **Step 8: Add `kv_compress_policies` field to Compiler struct**

Add to the Compiler struct:
```rust
/// M42: Per-layer KV compression policies from @kv_compress decorators.
pub kv_compress_policies: HashMap<String, Vec<KvCompressPolicy>>,
```

Where `KvCompressPolicy` is a simple codegen-side struct:
```rust
pub struct KvCompressPolicy {
    pub method: String,     // "quantize", "sliding_window", "h2o"
    pub scheme: u8,         // KvQuantScheme discriminant
    pub window: usize,
    pub sinks: usize,
    pub budget: usize,
}
```

Initialize in constructor: `kv_compress_policies: HashMap::new()`.

- [ ] **Step 9: Register FFI functions in builtins.rs**

Add to the `RUNTIME_FUNCTIONS` array:
```rust
// M42: KV-cache compression
("nsl_kv_quantize_and_store", &[I64, I64, I64, I64, I64, I64, I64, I64, I64, I64, I64], Some(I64)),
("nsl_kv_sliding_window_init", &[I64, I64, I64], Some(I64)),
("nsl_kv_sliding_window_check", &[I64, I64, I64, I64], Some(I64)),
("nsl_kv_sliding_window_destroy", &[], Some(I64)),
("nsl_kv_h2o_init", &[I64, I64, I64], Some(I64)),
("nsl_kv_h2o_accumulate", &[I64, I64, I64], Some(I64)),
("nsl_kv_h2o_check", &[I64, I64, I64, I64], Some(I64)),
("nsl_kv_h2o_remove_sequence", &[I64], Some(I64)),
("nsl_kv_h2o_destroy", &[], Some(I64)),
("nsl_kv_compress_ratio", &[I64], Some(I64)),
```

---

## Phase 5: Build Verification

- [ ] **Step 10: `cargo build` — verify no compile errors**

- [ ] **Step 11: `cargo test` — run all tests, expect 25+ new tests passing**

Expected new tests:
- `kv_compress::tests::*` (3 tests: scheme bytes, from_i64, default policy)
- `kv_compress::quantize::tests::*` (7 tests: INT8 per-head/per-token roundtrip, INT4 roundtrip, FP8 roundtrip, block size calc, metadata size, zeros)
- `kv_compress::sliding_window::tests::*` (7 tests: no eviction, correct blocks, partial block, large eviction, active tokens, attention ranges)
- `kv_compress::h2o::tests::*` (7 tests: no eviction, accumulation, score extend, eviction order, sinks protected, block-level eviction, remove sequence)
- `kv_compress::ffi::tests::*` (6 tests: quantize none/int8, sliding window lifecycle, H2O lifecycle, null pointer, compress ratio)

- [ ] **Step 12: `cargo clippy` — no warnings**

---

## Verification Checklist

After implementation, verify:

1. **Quantization roundtrip**: INT8 per-head max error < 1%, INT4 < 10%, FP8 reasonably close
2. **Block size calculation**: `block_data_bytes` correct for all 5 schemes
3. **Sliding window**: Eviction only when `current_len > sinks + window`, only full blocks evicted, sinks always retained
4. **H2O**: Lowest-scored non-sink entries evicted first, sinks protected, block-level granularity
5. **FFI**: All 10 functions callable with correct return codes
6. **Semantic**: Invalid `@kv_compress` config produces clear error diagnostics
7. **Codegen**: Compiler struct accepts kv_compress_policies, builtins registered
8. **No regressions**: All 499+ existing tests pass
