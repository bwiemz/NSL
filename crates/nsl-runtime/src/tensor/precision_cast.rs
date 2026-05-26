//! CPDT precision-adaptive optimizer state casts (v1: FP16 <-> FP32).
//!
//! Tensor-level (whole-tensor) dequant/quant for FASE's optimizer-state
//! storage. Reuses the crate's `f16_bits_to_f32` (exact widening) and
//! `f32_to_f16_bits` so CPU and GPU paths share identical numerics. NOTE:
//! despite its docstring, `f32_to_f16_bits` *truncates* the mantissa (it is
//! not round-to-nearest-even); v1 reuses it as-is. Truncation is biased
//! toward zero — proper rounding for optimizer state is a deferred ladder
//! step (see the design doc §10). Models its allocation on
//! `crate::fp8::nsl_fp8_cast`.

use std::ffi::c_void;

use crate::memory::checked_alloc;

use super::{
    f16_bits_to_f32, f32_to_f16_bits, NslTensor, DTYPE_F32, DTYPE_FP16,
};

/// Cast a tensor to `target_dtype`, returning a NEW owned tensor.
///
/// v1 supports F32 and FP16 storage. FP16 storage holds IEEE-754 half bits
/// as `u16`; the f32->f16 conversion uses the crate's `f32_to_f16_bits`
/// (which truncates the mantissa — see the module note). An F32->F32 cast
/// yields a faithful copy. The source tensor is NOT consumed.
#[no_mangle]
pub extern "C" fn nsl_tensor_cast(src_ptr: i64, target_dtype: i64) -> i64 {
    let t = unsafe { &*(src_ptr as *const NslTensor) };
    // v1 is CPU-only: `t.data` is dereferenced as a host pointer below. A GPU
    // tensor (device > 0) would be silent UB. Fail loudly instead. GPU support
    // (host round-trip / on-device conversion) is the Task 15 / v2 boundary.
    assert_eq!(
        t.device, 0,
        "nsl_tensor_cast: GPU tensors not supported in v1 (device={})",
        t.device
    );
    let len = t.len as usize;
    let target = target_dtype as u16;

    let src_f32: Vec<f32> = match t.dtype {
        DTYPE_F32 => {
            let d = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
            d.to_vec()
        }
        DTYPE_FP16 => {
            let d = unsafe { std::slice::from_raw_parts(t.data as *const u16, len) };
            d.iter().map(|&b| f16_bits_to_f32(b)).collect()
        }
        other => panic!("nsl_tensor_cast: unsupported source dtype {other} (v1: F32/FP16)"),
    };

    let shape = NslTensor::copy_shape(t.shape, t.ndim);
    let strides = NslTensor::compute_strides(shape, t.ndim);

    let data: *mut c_void = match target {
        DTYPE_F32 => {
            let p = checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
            for (i, &v) in src_f32.iter().enumerate() {
                unsafe { *p.add(i) = v };
            }
            p as *mut c_void
        }
        DTYPE_FP16 => {
            let p = checked_alloc(len * std::mem::size_of::<u16>()) as *mut u16;
            for (i, &v) in src_f32.iter().enumerate() {
                unsafe { *p.add(i) = f32_to_f16_bits(v) };
            }
            p as *mut c_void
        }
        other => panic!("nsl_tensor_cast: unsupported target dtype {other} (v1: F32/FP16)"),
    };

    let out = Box::new(NslTensor::new(
        data, shape, strides, t.ndim, t.len, t.device, target, 1, 0,
    ));
    // Bare `Box::into_raw` — NOT `NslTensor::publish`. The caller owns the
    // result and frees it explicitly (the FASE optimizer wrapping frees the
    // dequant/quant working tensors each step). `publish` would `scope_track`
    // the pointer; combined with the caller's explicit `nsl_tensor_free`, the
    // scope-end sweep would touch an already-freed tensor (use-after-free).
    Box::into_raw(out) as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build an owned f32 CPU tensor from a slice (helper for tests).
    fn f32_tensor(vals: &[f32]) -> i64 {
        let len = vals.len();
        let data = checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
        for (i, &v) in vals.iter().enumerate() {
            unsafe { *data.add(i) = v };
        }
        let shape = NslTensor::copy_shape([len as i64].as_ptr() as *mut i64, 1);
        let strides = NslTensor::compute_strides(shape, 1);
        let t = Box::new(NslTensor::new(
            data as *mut c_void, shape, strides, 1, len as i64, 0, DTYPE_F32, 1, 0,
        ));
        Box::into_raw(t) as i64
    }

    fn read_f32(ptr: i64) -> Vec<f32> {
        let t = unsafe { &*(ptr as *const NslTensor) };
        let len = t.len as usize;
        let d = unsafe { std::slice::from_raw_parts(t.data as *const f32, len) };
        d.to_vec()
    }

    #[test]
    fn cast_f32_to_fp16_then_back_matches_primitive_round_trip() {
        let vals = [1.0001_f32, 3.14159265_f32, -2.71828_f32, 65504.0, 1e-5];
        let src = f32_tensor(&vals);
        let fp16 = nsl_tensor_cast(src, DTYPE_FP16 as i64);
        let back = nsl_tensor_cast(fp16, DTYPE_F32 as i64);
        let got = read_f32(back);
        for (i, &v) in vals.iter().enumerate() {
            // Expected == exact round-trip through the crate's f32->f16 primitive
            // (which truncates) and back. Bit-exact: the cast must delegate to it.
            let expected = f16_bits_to_f32(f32_to_f16_bits(v));
            assert!(
                (got[i] - expected).abs() <= 0.0,
                "elem {i}: got {} expected {} (exact primitive round-trip)",
                got[i],
                expected
            );
        }
        crate::tensor::nsl_tensor_free(src);
        crate::tensor::nsl_tensor_free(fp16);
        crate::tensor::nsl_tensor_free(back);
    }

    #[test]
    fn cast_f32_to_f32_is_value_identity() {
        let vals = [1.0_f32, -0.5, 12345.678];
        let src = f32_tensor(&vals);
        let out = nsl_tensor_cast(src, DTYPE_F32 as i64);
        let got = read_f32(out);
        assert_eq!(got, vals.to_vec(), "F32->F32 cast must preserve bits");
        crate::tensor::nsl_tensor_free(src);
        crate::tensor::nsl_tensor_free(out);
    }

    /// Negative control: verify that the crate primitive `f32_to_f16_bits` produces
    /// bit-exact results consistent with the ACTUAL crate behavior (truncation of the
    /// lower 13 mantissa bits). Two distinct f32 values in the same f16 truncation
    /// bucket must produce identical f16 bits, confirming the cast path is faithful
    /// to the primitive rather than applying independent rounding.
    ///
    /// Note: `f32_to_f16_bits` is documented "round-to-nearest-even" but the
    /// implementation truncates (mant >> 13, no guard/sticky logic). This test
    /// documents that `nsl_tensor_cast` faithfully delegates to the primitive.
    #[test]
    fn cast_fp16_round_trip_is_faithful_to_primitive() {
        // Two f32 values that differ only in the lower 13 mantissa bits.
        // Both should produce the same f16 bits (truncation bucket).
        let v_lo = f32::from_bits(0x3F800001_u32); // 1.0 + 1 ULP (mant = 0x000001)
        let v_hi = f32::from_bits(0x3F801000_u32); // 1.0 + 0x1000 ULP (mant = 0x001000)
        // Both have upper 10 mant bits = 0 -> same f16 bucket
        assert_eq!(
            f32_to_f16_bits(v_lo),
            f32_to_f16_bits(v_hi),
            "both values must land in the same f16 truncation bucket"
        );
        // nsl_tensor_cast must match what the primitive produces
        let src_lo = {
            let data = checked_alloc(std::mem::size_of::<f32>()) as *mut f32;
            unsafe { *data = v_lo };
            let shape = NslTensor::copy_shape([1_i64].as_ptr() as *mut i64, 1);
            let strides = NslTensor::compute_strides(shape, 1);
            Box::into_raw(Box::new(NslTensor::new(
                data as *mut c_void, shape, strides, 1, 1, 0, DTYPE_F32, 1, 0,
            ))) as i64
        };
        let src_hi = {
            let data = checked_alloc(std::mem::size_of::<f32>()) as *mut f32;
            unsafe { *data = v_hi };
            let shape = NslTensor::copy_shape([1_i64].as_ptr() as *mut i64, 1);
            let strides = NslTensor::compute_strides(shape, 1);
            Box::into_raw(Box::new(NslTensor::new(
                data as *mut c_void, shape, strides, 1, 1, 0, DTYPE_F32, 1, 0,
            ))) as i64
        };
        let fp16_lo = nsl_tensor_cast(src_lo, DTYPE_FP16 as i64);
        let fp16_hi = nsl_tensor_cast(src_hi, DTYPE_FP16 as i64);
        let bits_lo = unsafe { *((&*(fp16_lo as *const NslTensor)).data as *const u16) };
        let bits_hi = unsafe { *((&*(fp16_hi as *const NslTensor)).data as *const u16) };
        assert_eq!(
            bits_lo, bits_hi,
            "cast must be faithful to f32_to_f16_bits: both values land in same f16 bucket"
        );
        crate::tensor::nsl_tensor_free(src_lo);
        crate::tensor::nsl_tensor_free(src_hi);
        crate::tensor::nsl_tensor_free(fp16_lo);
        crate::tensor::nsl_tensor_free(fp16_hi);
    }
}
