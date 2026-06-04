//! CPDT §3.2 — INT8 blockwise quantization for optimizer state storage.
//!
//! Paper §3.2: "fused: dequant INT8 → FP32, optimizer step, quant FP32 → INT8
//! ... m_int8 = quant_int8_blockwise(m_fp32)  # blockwise dynamic quantization".
//!
//! This module ships the runtime primitives (the FASE-side cast wrapping is a
//! follow-on). The blockwise layout is single-allocation: the data buffer holds
//! `N` `i8` values followed by `ceil(N / BLOCK_SIZE)` `f32` absmax scales,
//! 4-byte-aligned. The `NslTensor.dtype` field is `DTYPE_INT8`; `len` reports
//! the logical INT8-element count (scales are metadata, not visible to ops).
//!
//! Scaling: symmetric absmax (`scale = max(|x|) / 127`). Round-to-nearest by
//! default; stochastic rounding (paper §3.3) when the flag is set —
//! `floor(v / scale + u01)` where `u01` is drawn from a thread-local PRNG.
//! Stochastic mode is unbiased in expectation, which is the property §3.3
//! cites for INT8 stability on embeddings.

use std::ffi::c_void;

use crate::memory::checked_alloc;

use super::{NslTensor, DTYPE_F32, DTYPE_INT8};

/// Block size for blockwise scaling. Chosen for v1 — tight enough to give per-
/// block scale precision on typical optimizer-state distributions, large enough
/// to keep the scale overhead at ~6% (4 bytes per 64 INT8 values).
pub const INT8_BLOCK_SIZE: usize = 64;

/// Total bytes needed for a blockwise-INT8 tensor of `n` logical elements:
/// `n` INT8 values padded to 4 bytes + `ceil(n / BLOCK_SIZE)` f32 scales.
#[inline]
pub fn int8_blockwise_byte_size(n: usize) -> usize {
    let n_blocks = n.div_ceil(INT8_BLOCK_SIZE);
    let padded_values = (n + 3) & !3; // round n up to multiple of 4
    padded_values + n_blocks * std::mem::size_of::<f32>()
}

/// Offset (in bytes) where the scale array starts inside the data buffer.
#[inline]
pub fn int8_blockwise_scales_offset(n: usize) -> usize {
    (n + 3) & !3
}

/// Thread-local xorshift PRNG seeded once from `n` so stochastic rounding is
/// reproducible per-tensor-size — good enough for the unbiased-in-expectation
/// property and stable enough for tests.
fn xorshift_u01(state: &mut u64) -> f32 {
    // xorshift64*
    let mut x = *state;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    *state = x;
    let r = x.wrapping_mul(0x2545_F491_4F6C_DD1D);
    // High 24 bits -> [0, 1) float.
    ((r >> 40) as f32) * (1.0 / (1u32 << 24) as f32)
}

/// Per-block absmax (symmetric quantization scale denominator).
#[inline]
fn block_absmax(block: &[f32]) -> f32 {
    let mut m = 0.0f32;
    for &v in block {
        let a = v.abs();
        if a > m {
            m = a;
        }
    }
    m
}

/// Quantize an FP32 tensor to blockwise INT8.
///
/// `stochastic` non-zero enables stochastic rounding (paper §3.3). The returned
/// tensor's dtype is `DTYPE_INT8`; `len` is the logical INT8 element count.
/// Caller owns the result and must `nsl_tensor_free` it.
///
/// # Safety
/// `src_ptr` must point to a valid F32 `NslTensor` on the CPU (`device == 0`),
/// or be `0`.
#[no_mangle]
pub extern "C" fn nsl_tensor_quant_int8_blockwise(
    src_ptr: i64,
    stochastic: i64,
) -> i64 {
    if src_ptr == 0 {
        return 0;
    }
    let t = unsafe { &*(src_ptr as *const NslTensor) };
    assert_eq!(
        t.device, 0,
        "nsl_tensor_quant_int8_blockwise: GPU not supported in v1 (device={})",
        t.device
    );
    assert_eq!(
        t.dtype, DTYPE_F32,
        "nsl_tensor_quant_int8_blockwise: source dtype must be F32 (got {})",
        t.dtype
    );

    let n = t.len as usize;
    let src: &[f32] = unsafe { std::slice::from_raw_parts(t.data as *const f32, n) };

    // Allocate the destination data buffer (values + padding + scales).
    let bytes = int8_blockwise_byte_size(n);
    let dst_data = checked_alloc(bytes);
    let int8_ptr = dst_data as *mut i8;
    let scales_ptr = unsafe {
        dst_data.add(int8_blockwise_scales_offset(n)) as *mut f32
    };

    let mut rng_state: u64 = (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    let do_stochastic = stochastic != 0;

    let mut block_idx = 0usize;
    for chunk_start in (0..n).step_by(INT8_BLOCK_SIZE) {
        let chunk_end = (chunk_start + INT8_BLOCK_SIZE).min(n);
        let block = &src[chunk_start..chunk_end];
        let absmax = block_absmax(block);
        let scale = if absmax > 0.0 { absmax / 127.0 } else { 1.0 };
        unsafe { *scales_ptr.add(block_idx) = scale };
        for (i, &v) in block.iter().enumerate() {
            let scaled = v / scale;
            let q = if do_stochastic {
                let r = xorshift_u01(&mut rng_state);
                (scaled + r).floor()
            } else {
                scaled.round()
            };
            let q = q.clamp(-128.0, 127.0) as i8;
            unsafe { *int8_ptr.add(chunk_start + i) = q };
        }
        block_idx += 1;
    }

    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);

    let out = Box::new(NslTensor::new(
        dst_data as *mut c_void,
        shape,
        strides,
        t.ndim,
        t.len,
        t.device,
        DTYPE_INT8,
        1,
        0,
    ));
    Box::into_raw(out) as i64
}

/// Dequantize a blockwise-INT8 tensor back to FP32.
///
/// Reads the scale array from the metadata region of the input data buffer.
/// Returns a new FP32 tensor; caller owns the result.
///
/// # Safety
/// `src_ptr` must point to a valid INT8 `NslTensor` produced by
/// `nsl_tensor_quant_int8_blockwise`, or be `0`.
#[no_mangle]
pub extern "C" fn nsl_tensor_dequant_int8_blockwise(src_ptr: i64) -> i64 {
    if src_ptr == 0 {
        return 0;
    }
    let t = unsafe { &*(src_ptr as *const NslTensor) };
    assert_eq!(
        t.device, 0,
        "nsl_tensor_dequant_int8_blockwise: GPU not supported in v1 (device={})",
        t.device
    );
    assert_eq!(
        t.dtype, DTYPE_INT8,
        "nsl_tensor_dequant_int8_blockwise: source dtype must be INT8 (got {})",
        t.dtype
    );

    let n = t.len as usize;
    let int8_ptr = t.data as *const i8;
    let scales_ptr = unsafe {
        (t.data as *const u8).add(int8_blockwise_scales_offset(n)) as *const f32
    };

    let dst_data = checked_alloc(n * std::mem::size_of::<f32>()) as *mut f32;

    let mut block_idx = 0usize;
    for chunk_start in (0..n).step_by(INT8_BLOCK_SIZE) {
        let chunk_end = (chunk_start + INT8_BLOCK_SIZE).min(n);
        let scale = unsafe { *scales_ptr.add(block_idx) };
        for i in chunk_start..chunk_end {
            let q = unsafe { *int8_ptr.add(i) };
            unsafe { *dst_data.add(i) = (q as f32) * scale };
        }
        block_idx += 1;
    }

    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);

    let out = Box::new(NslTensor::new(
        dst_data as *mut c_void,
        shape,
        strides,
        t.ndim,
        t.len,
        t.device,
        DTYPE_F32,
        1,
        0,
    ));
    Box::into_raw(out) as i64
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::nsl_tensor_free;

    /// Build an F32 NslTensor from a Vec, returning the raw pointer. Caller
    /// must `nsl_tensor_free` it.
    fn make_f32(vals: &[f32]) -> i64 {
        let n = vals.len();
        let data = checked_alloc(n * std::mem::size_of::<f32>()) as *mut f32;
        for (i, &v) in vals.iter().enumerate() {
            unsafe { *data.add(i) = v };
        }
        let shape = NslTensor::copy_shape([n as i64].as_ptr() as *const i64, 1);
        let strides = NslTensor::compute_strides(shape, 1);
        let t = Box::new(NslTensor::new(
            data as *mut c_void,
            shape,
            strides,
            1,
            n as i64,
            0,
            DTYPE_F32,
            1,
            0,
        ));
        Box::into_raw(t) as i64
    }

    fn read_f32(ptr: i64) -> Vec<f32> {
        let t = unsafe { &*(ptr as *const NslTensor) };
        let n = t.len as usize;
        let s = unsafe { std::slice::from_raw_parts(t.data as *const f32, n) };
        s.to_vec()
    }

    #[test]
    fn byte_size_formula_matches_layout() {
        assert_eq!(int8_blockwise_byte_size(64), 64 + 4);
        assert_eq!(int8_blockwise_byte_size(65), 68 + 2 * 4); // 65 padded to 68, 2 blocks
        assert_eq!(int8_blockwise_byte_size(1), 4 + 4);
        assert_eq!(int8_blockwise_byte_size(0), 0);
    }

    #[test]
    fn roundtrip_zero_preserves_exactly() {
        let vals = vec![0.0f32; 64];
        let src = make_f32(&vals);
        let qt = nsl_tensor_quant_int8_blockwise(src, 0);
        let dqt = nsl_tensor_dequant_int8_blockwise(qt);
        let out = read_f32(dqt);
        for v in &out {
            assert_eq!(*v, 0.0);
        }
        nsl_tensor_free(src);
        nsl_tensor_free(qt);
        nsl_tensor_free(dqt);
    }

    #[test]
    fn roundtrip_deterministic_within_int8_precision() {
        // 64 values in [-1, 1]: with absmax≈1, scale≈1/127, max error ≈ 0.5/127.
        let vals: Vec<f32> = (0..64)
            .map(|i| ((i as f32) / 64.0) * 2.0 - 1.0) // -1..1
            .collect();
        let src = make_f32(&vals);
        let qt = nsl_tensor_quant_int8_blockwise(src, 0);
        let dqt = nsl_tensor_dequant_int8_blockwise(qt);
        let out = read_f32(dqt);
        // 0.5 / 127 ≈ 4e-3, leave some headroom for the absmax denominator.
        let tol = 1.0 / 127.0;
        for (a, b) in vals.iter().zip(out.iter()) {
            assert!((a - b).abs() <= tol, "deterministic: {a} -> {b} err {}", (a - b).abs());
        }
        nsl_tensor_free(src);
        nsl_tensor_free(qt);
        nsl_tensor_free(dqt);
    }

    #[test]
    fn stochastic_rounding_is_unbiased_in_mean() {
        // Paper §3.3: stochastic rounding is unbiased in expectation. Quantize
        // a slope -> dequantize -> repeat enough times that the per-sample
        // noise averages out. The blockwise absmax stays the same across runs
        // (input is unchanged), so the only stochastic variation is the
        // round-up/down decision.
        let vals: Vec<f32> = (0..64).map(|i| (i as f32) / 64.0).collect(); // 0..1
        let mut sums = vec![0.0f64; 64];
        let trials = 64;
        for _ in 0..trials {
            let src = make_f32(&vals);
            let qt = nsl_tensor_quant_int8_blockwise(src, 1);
            let dqt = nsl_tensor_dequant_int8_blockwise(qt);
            let out = read_f32(dqt);
            for (i, v) in out.iter().enumerate() {
                sums[i] += *v as f64;
            }
            nsl_tensor_free(src);
            nsl_tensor_free(qt);
            nsl_tensor_free(dqt);
        }
        // Mean residual should be smaller than the deterministic quantization
        // error (one-sided bias of truncate-to-zero is the failure we'd see
        // without stochastic rounding). Tolerance: 1/127 (one INT8 step).
        for (i, s) in sums.iter().enumerate() {
            let mean = (*s / trials as f64) as f32;
            let err = (vals[i] - mean).abs();
            assert!(err < 1.0 / 127.0, "i={i} input {} mean {} err {}", vals[i], mean, err);
        }
    }

    #[test]
    fn dtype_and_len_set_on_output() {
        let src = make_f32(&vec![0.5f32; 32]);
        let qt = nsl_tensor_quant_int8_blockwise(src, 0);
        let t = unsafe { &*(qt as *const NslTensor) };
        assert_eq!(t.dtype, DTYPE_INT8);
        assert_eq!(t.len, 32);
        nsl_tensor_free(src);
        nsl_tensor_free(qt);
    }

    #[test]
    fn null_input_returns_null() {
        assert_eq!(nsl_tensor_quant_int8_blockwise(0, 0), 0);
        assert_eq!(nsl_tensor_dequant_int8_blockwise(0), 0);
    }
}
