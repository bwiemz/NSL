//! M35: GPTQ (Generalized Post-Training Quantization) runtime.
//!
//! Reuses AWQ packed format. The difference is quantization:
//! GPTQ uses Hessian information to minimize reconstruction error.
//! Full OBQ (Optimal Brain Quantizer) deferred to M35b — baseline is RTN.

use crate::awq::{awq_quantize_cpu, AwqPackedWeight};
use crate::tensor::NslTensor;

/// GPTQ quantize: uses Hessian diagonal to weight error compensation.
/// For now, implements simple RTN (Round-To-Nearest) as baseline.
/// Full OBQ (Optimal Brain Quantizer) is deferred to M35b.
pub fn gptq_quantize_cpu(
    weights: &[f64],
    _hessian_diag: &[f64],
    k: usize,
    n: usize,
    group_size: usize,
    _bits: usize,
) -> AwqPackedWeight {
    // Baseline: RTN quantization (same as AWQ without salient channel detection).
    awq_quantize_cpu(weights, k, n, group_size)
}

// ---------------------------------------------------------------------------
// FFI
// ---------------------------------------------------------------------------

/// GPTQ quantize: weight + Hessian -> packed weight.
#[no_mangle]
pub extern "C" fn nsl_gptq_quantize(
    weight_ptr: i64,
    hessian_ptr: i64,
    group_size: i64,
    bits: i64,
) -> i64 {
    let t = unsafe { &*(weight_ptr as *const NslTensor) };
    assert!(t.ndim >= 2, "nsl_gptq_quantize requires 2D weight tensor (got {}D)", t.ndim);
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

    let hessian = if hessian_ptr == 0 {
        vec![1.0f64; k]
    } else {
        let h = unsafe { &*(hessian_ptr as *const NslTensor) };
        let h_len = h.len as usize;
        if h.dtype == 1 {
            let raw = unsafe { std::slice::from_raw_parts(h.data as *const f32, h_len) };
            raw.iter().map(|&v| v as f64).collect()
        } else {
            unsafe { std::slice::from_raw_parts(h.data as *const f64, h_len) }.to_vec()
        }
    };

    let packed = gptq_quantize_cpu(&data, &hessian, k, n, group_size as usize, bits as usize);
    Box::into_raw(Box::new(packed)) as i64
}

/// GPTQ dequant-matmul (same kernel as AWQ — packed format is identical).
#[no_mangle]
pub extern "C" fn nsl_gptq_matmul(
    input_ptr: i64,
    packed_ptr: i64,
    group_size: i64,
    _bits: i64,
) -> i64 {
    crate::awq::nsl_awq_matmul(input_ptr, packed_ptr, group_size)
}

/// Free a GPTQ packed weight.
#[no_mangle]
pub extern "C" fn nsl_gptq_free(packed_ptr: i64) {
    crate::awq::nsl_awq_free(packed_ptr);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::awq::{awq_dequantize_cpu, awq_free_packed};

    #[test]
    fn test_gptq_quantize_rtn_baseline() {
        let weights: Vec<f64> = (0..16).map(|i| i as f64 * 0.1).collect();
        let hessian = vec![1.0f64; 4]; // identity Hessian
        let packed = gptq_quantize_cpu(&weights, &hessian, 4, 4, 4, 4);

        let recovered = awq_dequantize_cpu(&packed);
        for (i, (&orig, &rec)) in weights.iter().zip(recovered.iter()).enumerate() {
            assert!(
                (orig - rec).abs() < 0.15,
                "GPTQ RTN error too high at {}: orig={}, recovered={}",
                i,
                orig,
                rec
            );
        }

        unsafe { awq_free_packed(&packed) };
    }
}
