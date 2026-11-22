// crates/nsl-runtime/src/kv_compress/ffi.rs
//! M42: FFI exports for KV-cache compression.

use super::KvQuantScheme;
use super::quantize::{self, KvBlockQuantMeta};

// ---------------------------------------------------------------------------
// Quantization FFI
// ---------------------------------------------------------------------------

/// Quantize `num_tokens` new K/V tokens and store them in a block at
/// `token_offset`.
///
/// Parameters (all i64 for Cranelift):
/// - raw_k/raw_v: pointers to f32 K/V values for the new tokens,
///   `[num_heads, num_tokens, head_dim]`
/// - block_k/block_v: pointers to the WHOLE target blocks,
///   `[num_heads, block_size, head_dim]` in the scheme's storage type (f32
///   for none, i8 for INT8, u8 for FP8, packed nibbles for INT4)
/// - meta_k/meta_v: pointers to KvBlockQuantMeta (null for Fp8/None)
/// - token_offset: tokens already in the block; the new tokens go after them
/// - num_tokens: tokens to quantize
/// - block_size: the block's capacity in tokens
/// - num_heads: attention heads
/// - head_dim: dimension per head
/// - scheme: KvQuantScheme discriminant
///
/// What "append" means depends on what a scale covers:
/// - none, FP8: every element stands alone, so any token range is written
///   in place.
/// - INT8 per token: each token has its own scale, kept at its absolute
///   position. Earlier tokens are untouched.
/// - INT8 per head, INT4 per group: a scale covers elements of many tokens,
///   so adding tokens would change the scale earlier codes were written
///   with. Only a whole block at once is accepted (`token_offset == 0`,
///   `num_tokens == block_size`).
///
/// The token offset used to be ignored: scales were computed over the new
/// tokens alone and written from index 0, replacing the block's scales, so
/// tokens stored earlier would have been decoded with a scale they were not
/// stored with.
///
/// Returns 0 on success; -1 for a null pointer, a negative size or a token
/// range that does not fit the block; -2 when the scheme cannot append and
/// the call is not a whole block; -3 when the block needs more scale entries
/// than its metadata holds.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub extern "C" fn nsl_kv_quantize_and_store(
    raw_k: i64,
    raw_v: i64,
    block_k: i64,
    block_v: i64,
    meta_k: i64,
    meta_v: i64,
    token_offset: i64,
    num_tokens: i64,
    block_size: i64,
    num_heads: i64,
    head_dim: i64,
    scheme: i64,
) -> i64 {
    let qs = KvQuantScheme::from_i64(scheme);
    if raw_k == 0 || raw_v == 0 || block_k == 0 || block_v == 0 {
        return -1;
    }
    if token_offset < 0 || num_tokens < 0 || block_size <= 0 || num_heads <= 0 || head_dim <= 0 {
        return -1;
    }
    let (offset, tokens, block, heads, dim) = (
        token_offset as usize,
        num_tokens as usize,
        block_size as usize,
        num_heads as usize,
        head_dim as usize,
    );
    if offset + tokens > block {
        return -1;
    }
    let needs_meta = matches!(
        qs,
        KvQuantScheme::Int8PerHead | KvQuantScheme::Int8PerToken | KvQuantScheme::Int4PerGroup
    );
    if needs_meta && (meta_k == 0 || meta_v == 0) {
        return -1;
    }
    let whole_block = offset == 0 && tokens == block;
    if matches!(qs, KvQuantScheme::Int8PerHead | KvQuantScheme::Int4PerGroup) && !whole_block {
        return -2;
    }

    let n_new = heads * tokens * dim;
    let n_block = heads * block * dim;

    // Read input values
    let k_values = unsafe { std::slice::from_raw_parts(raw_k as *const f32, n_new) };
    let v_values = unsafe { std::slice::from_raw_parts(raw_v as *const f32, n_new) };

    // Where head `h`'s new tokens start, in the input and in the block.
    let src_at = |h: usize| h * tokens * dim;
    let dst_at = |h: usize| h * block * dim + offset * dim;
    let run = tokens * dim;

    let status = |r: Result<(), quantize::KvQuantError>| match r {
        Ok(()) => 0,
        Err(quantize::KvQuantError::TooManyScales { .. }) => -3,
        Err(_) => -1,
    };

    match qs {
        KvQuantScheme::None => {
            // Direct copy (no quantization)
            let k_out = unsafe { std::slice::from_raw_parts_mut(block_k as *mut f32, n_block) };
            let v_out = unsafe { std::slice::from_raw_parts_mut(block_v as *mut f32, n_block) };
            for h in 0..heads {
                k_out[dst_at(h)..dst_at(h) + run].copy_from_slice(&k_values[src_at(h)..src_at(h) + run]);
                v_out[dst_at(h)..dst_at(h) + run].copy_from_slice(&v_values[src_at(h)..src_at(h) + run]);
            }
            0
        }
        KvQuantScheme::Int8PerHead => {
            let k_out = unsafe { std::slice::from_raw_parts_mut(block_k as *mut i8, n_block) };
            let v_out = unsafe { std::slice::from_raw_parts_mut(block_v as *mut i8, n_block) };
            let k_meta = unsafe { &mut *(meta_k as *mut KvBlockQuantMeta) };
            let v_meta = unsafe { &mut *(meta_v as *mut KvBlockQuantMeta) };
            let k = status(quantize::quantize_int8_per_head(k_values, k_out, k_meta, heads, block, dim));
            if k != 0 {
                return k;
            }
            status(quantize::quantize_int8_per_head(v_values, v_out, v_meta, heads, block, dim))
        }
        KvQuantScheme::Int8PerToken => {
            let k_out = unsafe { std::slice::from_raw_parts_mut(block_k as *mut i8, n_block) };
            let v_out = unsafe { std::slice::from_raw_parts_mut(block_v as *mut i8, n_block) };
            let k_meta = unsafe { &mut *(meta_k as *mut KvBlockQuantMeta) };
            let v_meta = unsafe { &mut *(meta_v as *mut KvBlockQuantMeta) };
            if offset == 0 {
                // A block being started over forgets the scales it had.
                k_meta.num_scales = 0;
                v_meta.num_scales = 0;
            }
            let k = status(quantize::append_int8_per_token(
                k_values, k_out, k_meta, heads, block, dim, offset, tokens,
            ));
            if k != 0 {
                return k;
            }
            status(quantize::append_int8_per_token(
                v_values, v_out, v_meta, heads, block, dim, offset, tokens,
            ))
        }
        KvQuantScheme::Int4PerGroup => {
            let packed_len = n_block.div_ceil(2);
            let k_out = unsafe { std::slice::from_raw_parts_mut(block_k as *mut u8, packed_len) };
            let v_out = unsafe { std::slice::from_raw_parts_mut(block_v as *mut u8, packed_len) };
            let k_meta = unsafe { &mut *(meta_k as *mut KvBlockQuantMeta) };
            let v_meta = unsafe { &mut *(meta_v as *mut KvBlockQuantMeta) };
            let k = status(quantize::quantize_int4_per_group(k_values, k_out, k_meta, quantize::INT4_GROUP_SIZE));
            if k != 0 {
                return k;
            }
            status(quantize::quantize_int4_per_group(v_values, v_out, v_meta, quantize::INT4_GROUP_SIZE))
        }
        KvQuantScheme::Fp8 => {
            let k_out = unsafe { std::slice::from_raw_parts_mut(block_k as *mut u8, n_block) };
            let v_out = unsafe { std::slice::from_raw_parts_mut(block_v as *mut u8, n_block) };
            for h in 0..heads {
                quantize::quantize_fp8(&k_values[src_at(h)..src_at(h) + run], &mut k_out[dst_at(h)..dst_at(h) + run]);
                quantize::quantize_fp8(&v_values[src_at(h)..src_at(h) + run], &mut v_out[dst_at(h)..dst_at(h) + run]);
            }
            0
        }
    }
}

// ---------------------------------------------------------------------------
// GPU dequantization FFI
// ---------------------------------------------------------------------------

/// Dequantize KV-cache data on GPU, writing f32 output.
///
/// Parameters (all i64 for Cranelift):
/// - data_ptr: device pointer to quantized data (i8/u8)
/// - out_ptr: device pointer to f32 output buffer (pre-allocated)
/// - meta_ptr: host pointer to KvBlockQuantMeta (null for FP8)
/// - n: total number of elements to dequantize
/// - num_heads: attention heads (for INT8 per-head scale indexing)
/// - block_size: tokens per block
/// - head_dim: dimension per head
/// - scheme: KvQuantScheme discriminant (1-4)
///
/// Returns 0 on success, -1 on error.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_kv_dequantize_gpu(
    data_ptr: i64,
    out_ptr: i64,
    meta_ptr: i64,
    n: i64,
    // Per-head scales are indexed by element / (block_size * head_dim); the
    // head count adds nothing. Kept for the C ABI.
    _num_heads: i64,
    block_size: i64,
    head_dim: i64,
    scheme: i64,
) -> i64 {
    if data_ptr == 0 || out_ptr == 0 || n <= 0 {
        return -1;
    }

    let qs = KvQuantScheme::from_i64(scheme);
    #[cfg(feature = "cuda")]
    {
        use std::ffi::c_void;

        let n_val = n as u64;
        let data = data_ptr as *const c_void;
        let out = out_ptr as *mut c_void;
        let head_stride = (block_size * head_dim) as u64;

        match qs {
            KvQuantScheme::None => {
                // No dequantization — copy f32 data
                crate::cuda::inner::memcpy_dtod(out, data, n_val as usize * 4);
            }
            KvQuantScheme::Int8PerHead => {
                if meta_ptr == 0 { return -1; }
                let meta = unsafe { &*(meta_ptr as *const KvBlockQuantMeta) };
                // Upload scales to GPU
                let scales_bytes = meta.num_scales as usize * 4;
                let gpu_scales = crate::cuda::inner::alloc_managed(scales_bytes);
                crate::cuda::inner::memcpy_htod(
                    gpu_scales,
                    meta.scales.as_ptr() as *const c_void,
                    scales_bytes,
                );
                crate::cuda::gpu_dequant_int8_per_head_f32(
                    data, out, gpu_scales as *const c_void, n_val, head_stride,
                );
                unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
                crate::cuda::inner::free_managed(gpu_scales);
            }
            KvQuantScheme::Int8PerToken => {
                if meta_ptr == 0 { return -1; }
                let meta = unsafe { &*(meta_ptr as *const KvBlockQuantMeta) };
                let scales_bytes = meta.num_scales as usize * 4;
                let gpu_scales = crate::cuda::inner::alloc_managed(scales_bytes);
                crate::cuda::inner::memcpy_htod(
                    gpu_scales,
                    meta.scales.as_ptr() as *const c_void,
                    scales_bytes,
                );
                crate::cuda::gpu_dequant_int8_per_token_f32(
                    data, out, gpu_scales as *const c_void, n_val, head_stride, head_dim as u64,
                );
                unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
                crate::cuda::inner::free_managed(gpu_scales);
            }
            KvQuantScheme::Int4PerGroup => {
                if meta_ptr == 0 { return -1; }
                let meta = unsafe { &*(meta_ptr as *const KvBlockQuantMeta) };
                let num_groups = meta.num_scales as usize;
                let bytes = num_groups * 4;
                let gpu_scales = crate::cuda::inner::alloc_managed(bytes);
                let gpu_zp = crate::cuda::inner::alloc_managed(bytes);
                crate::cuda::inner::memcpy_htod(
                    gpu_scales,
                    meta.scales.as_ptr() as *const c_void,
                    bytes,
                );
                crate::cuda::inner::memcpy_htod(
                    gpu_zp,
                    meta.zero_points.as_ptr() as *const c_void,
                    bytes,
                );
                crate::cuda::gpu_dequant_int4_per_group_f32(
                    data, out,
                    gpu_scales as *const c_void,
                    gpu_zp as *const c_void,
                    n_val,
                    quantize::INT4_GROUP_SIZE as u64,
                );
                unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
                crate::cuda::inner::free_managed(gpu_scales);
                crate::cuda::inner::free_managed(gpu_zp);
            }
            KvQuantScheme::Fp8 => {
                crate::cuda::gpu_dequant_fp8_e4m3_f32(data, out, n_val);
                unsafe { cudarc::driver::sys::cuCtxSynchronize(); }
            }
        }
        0
    }

    #[cfg(not(feature = "cuda"))]
    {
        let _ = (data_ptr, out_ptr, meta_ptr, n, _num_heads, block_size, head_dim, qs);
        crate::nsl_log!(WARN, "nsl", "[nsl] nsl_kv_dequantize_gpu requires CUDA");
        -1
    }
}

// ---------------------------------------------------------------------------
// Compression statistics
// ---------------------------------------------------------------------------

/// Compression ratio of one K or V block against FP16 storage, counting the
/// per-block quantization metadata.
///
/// `scheme` is a KvQuantScheme discriminant; the block is
/// `num_heads x block_size x head_dim`. Returns the ratio as f64 bits, or
/// the bits of 0.0 when a dimension is not positive.
///
/// This used to take the scheme alone and return 2.0 for INT8 and 4.0 for
/// INT4: the payload ratio, with the 1,032-byte `KvBlockQuantMeta` each
/// quantized block carries left out. For 4 heads x 16 tokens x 64 dims the
/// block is 8,192 bytes in FP16, 4,096 + 1,032 in INT8 (1.60x) and
/// 2,048 + 1,032 in INT4 (2.66x).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_kv_compress_ratio(scheme: i64, num_heads: i64, block_size: i64, head_dim: i64) -> i64 {
    if num_heads <= 0 || block_size <= 0 || head_dim <= 0 {
        return f64::to_bits(0.0) as i64;
    }
    let qs = KvQuantScheme::from_i64(scheme);
    let (heads, block, dim) = (num_heads as usize, block_size as usize, head_dim as usize);
    let fp16 = quantize::block_total_bytes(KvQuantScheme::None, heads, block, dim);
    let stored = quantize::block_total_bytes(qs, heads, block, dim);
    f64::to_bits(fp16 as f64 / stored as f64) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

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
            0, 1, 1, 1, 4, // offset=0, tokens=1, block=1, heads=1, dim=4
            0, // scheme=None
        );
        assert_eq!(rc, 0);
        assert_eq!(k_out, k);
        assert_eq!(v_out, v);
    }

    #[test]
    fn ffi_quantize_int8() {
        let n = 2 * 4; // heads=2, tokens=1, dim=4
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
            0, 1, 1, 2, 4, // offset=0, tokens=1, block=1, heads=2, dim=4
            1, // scheme=Int8PerHead
        );
        assert_eq!(rc, 0);
        assert_eq!(k_meta.scheme, 1);
        assert_eq!(k_meta.num_scales, 2);
    }

    #[test]
    fn ffi_null_pointer_returns_error() {
        let rc = nsl_kv_quantize_and_store(0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 4, 0);
        assert_eq!(rc, -1);
    }

    /// The ratio counts the metadata each quantized block carries. By
    /// hand, for 4 heads x 16 tokens x 64 dims (4,096 elements): FP16 is
    /// 8,192 bytes; INT8 is 4,096 + 1,032; INT4 is 2,048 + 1,032; FP8 has no
    /// metadata.
    #[test]
    fn ffi_compress_ratio_counts_the_metadata() {
        assert_eq!(std::mem::size_of::<KvBlockQuantMeta>(), 1032);
        let ratio = |scheme: i64| f64::from_bits(nsl_kv_compress_ratio(scheme, 4, 16, 64) as u64);
        assert_eq!(ratio(0), 1.0);
        assert_eq!(ratio(1), 8192.0 / 5128.0);
        assert_eq!(ratio(2), 8192.0 / 5128.0);
        assert_eq!(ratio(3), 8192.0 / 3080.0);
        assert_eq!(ratio(4), 2.0);
        assert!((ratio(1) - 1.60).abs() < 0.005 && (ratio(3) - 2.66).abs() < 0.005);
        // The payload-only figures are approached only for large blocks.
        let big = f64::from_bits(nsl_kv_compress_ratio(1, 32, 128, 128) as u64);
        assert!(big > 1.99 && big < 2.0, "{big}");
        assert_eq!(f64::from_bits(nsl_kv_compress_ratio(1, 0, 16, 64) as u64), 0.0);
    }

    /// Two per-token calls, the second at offset 2, against one call over
    /// the whole block: same codes, same scales, and the first two scales
    /// are still the ones the first call wrote (0.4/127 and 0.8/127, by
    /// hand from the largest magnitude of each token).
    #[test]
    fn ffi_per_token_append_keeps_earlier_scales() {
        let (heads, block, dim) = (2i64, 4i64, 2i64);
        let first = [0.1f32, -0.2, 0.5, 0.6, 0.3, 0.4, -0.7, 0.8];
        let second = [10.0f32, 20.0, 50.0, 60.0, 30.0, -40.0, 70.0, 80.0];
        let mut k_block = vec![0i8; 16];
        let mut v_block = vec![0i8; 16];
        let mut k_meta = KvBlockQuantMeta::default();
        let mut v_meta = KvBlockQuantMeta::default();
        let mut store = |vals: &[f32; 8], offset: i64, k_block: &mut Vec<i8>, v_block: &mut Vec<i8>| {
            nsl_kv_quantize_and_store(
                vals.as_ptr() as i64, vals.as_ptr() as i64,
                k_block.as_mut_ptr() as i64, v_block.as_mut_ptr() as i64,
                &mut k_meta as *mut _ as i64, &mut v_meta as *mut _ as i64,
                offset, 2, block, heads, dim,
                2, // Int8PerToken
            )
        };
        assert_eq!(store(&first, 0, &mut k_block, &mut v_block), 0);
        assert_eq!(store(&second, 2, &mut k_block, &mut v_block), 0);
        assert_eq!(k_meta.num_scales, 4);
        assert_eq!(&k_meta.scales[..4], &[0.4 / 127.0, 0.8 / 127.0, 40.0 / 127.0, 80.0 / 127.0]);
        assert_eq!(k_block, v_block);
        // Token 0 at scale 0.4/127: head 0's 0.1 is 31.75 steps (code 32),
        // head 1's {0.3, 0.4} are 95.25 and 127 steps (codes 95 and 127).
        assert_eq!(k_block[0], 32);
        assert_eq!(&k_block[8..10], &[95, 127]);
        // Token 2 of head 1 is {30, -40} at scale 40/127: codes 95 and -127.
        assert_eq!(&k_block[8 + 4..8 + 6], &[95, -127]);
    }

    /// A scheme whose scale spans tokens cannot append; it says so.
    #[test]
    fn ffi_schemes_that_cannot_append_refuse_a_partial_block() {
        let vals = [0.5f32; 8];
        let mut k_block = vec![0u8; 16];
        let mut v_block = vec![0u8; 16];
        let mut k_meta = KvBlockQuantMeta::default();
        let mut v_meta = KvBlockQuantMeta::default();
        for scheme in [1i64, 3] {
            let rc = nsl_kv_quantize_and_store(
                vals.as_ptr() as i64, vals.as_ptr() as i64,
                k_block.as_mut_ptr() as i64, v_block.as_mut_ptr() as i64,
                &mut k_meta as *mut _ as i64, &mut v_meta as *mut _ as i64,
                2, 2, 4, 2, 2, // offset=2, tokens=2, block=4, heads=2, dim=2
                scheme,
            );
            assert_eq!(rc, -2, "scheme {scheme}");
        }
        // A range that runs off the block, and a quantizing scheme with no
        // metadata, are plain errors.
        let rc = nsl_kv_quantize_and_store(
            vals.as_ptr() as i64, vals.as_ptr() as i64,
            k_block.as_mut_ptr() as i64, v_block.as_mut_ptr() as i64,
            &mut k_meta as *mut _ as i64, &mut v_meta as *mut _ as i64,
            3, 2, 4, 2, 2, 2,
        );
        assert_eq!(rc, -1);
        let rc = nsl_kv_quantize_and_store(
            vals.as_ptr() as i64, vals.as_ptr() as i64,
            k_block.as_mut_ptr() as i64, v_block.as_mut_ptr() as i64,
            0, 0, 0, 2, 4, 2, 2, 2,
        );
        assert_eq!(rc, -1);
    }

    /// The uncompressed scheme writes the new tokens at their offset in each
    /// head's run and leaves the rest of the block alone.
    #[test]
    fn ffi_none_scheme_appends_in_place() {
        // heads=2, block=3 tokens, dim=1; one new token per call.
        let mut k_block = vec![-1.0f32; 6];
        let mut v_block = vec![-1.0f32; 6];
        for (offset, vals) in [(0i64, [1.0f32, 4.0]), (1, [2.0, 5.0]), (2, [3.0, 6.0])] {
            let rc = nsl_kv_quantize_and_store(
                vals.as_ptr() as i64, vals.as_ptr() as i64,
                k_block.as_mut_ptr() as i64, v_block.as_mut_ptr() as i64,
                0, 0, offset, 1, 3, 2, 1, 0,
            );
            assert_eq!(rc, 0);
        }
        assert_eq!(k_block, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        assert_eq!(v_block, k_block);
    }

    /// INT4 on a block that needs more groups than the metadata holds
    /// returns a status; it used to panic inside an `extern "C"` function.
    #[test]
    fn ffi_int4_on_a_realistic_block_returns_a_status() {
        let (heads, block, dim) = (8usize, 16usize, 128usize);
        let n = heads * block * dim;
        let vals = vec![0.25f32; n];
        let mut k_block = vec![0u8; n / 2];
        let mut v_block = vec![0u8; n / 2];
        let mut k_meta = KvBlockQuantMeta::default();
        let mut v_meta = KvBlockQuantMeta::default();
        let rc = nsl_kv_quantize_and_store(
            vals.as_ptr() as i64, vals.as_ptr() as i64,
            k_block.as_mut_ptr() as i64, v_block.as_mut_ptr() as i64,
            &mut k_meta as *mut _ as i64, &mut v_meta as *mut _ as i64,
            0, block as i64, block as i64, heads as i64, dim as i64,
            3,
        );
        assert_eq!(rc, -3);
    }
}
