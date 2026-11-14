//! The quantizer entry points on GPU tensors.
//!
//! `nsl_qtensor_quantize`, `nsl_awq_quantize` and `nsl_gptq_quantize` read
//! `tensor.data` as host memory without looking at the tensor's device; for
//! a tensor on a CUDA device that field is a device address. They now stage
//! the tensor to the host first (and read a strided view through its
//! strides). Each expectation here is the tensor's own values, written out
//! by hand, within half a quantization step.
//!
//! Run: cargo test -p nsl-runtime --features cuda --test quantize_gpu_device
//!      -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use nsl_runtime::awq::{awq_dequantize_cpu, nsl_awq_free, nsl_awq_quantize, AwqPackedWeight};
use nsl_runtime::gptq::{nsl_gptq_free, nsl_gptq_quantize};
use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::quantize::{nsl_qtensor_dequantize, nsl_qtensor_quantize, nsl_qtensor_release};
use nsl_runtime::tensor::{nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_transpose, nsl_tensor_zeros_on};
use nsl_runtime::{nsl_cuda_init, nsl_test_cuda_h2d};

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_cuda_init() == 0
}

fn gpu_tensor(vals: &[f32], shape: &[i64]) -> i64 {
    let shape_list = nsl_list_new();
    for &dim in shape {
        nsl_list_push(shape_list, dim);
    }
    let t = nsl_tensor_zeros_on(shape_list, 1);
    nsl_list_free(shape_list);
    nsl_test_cuda_h2d(nsl_tensor_data_ptr(t), vals.as_ptr() as i64, (vals.len() * 4) as i64);
    t
}

/// The stored matrix, 2x3, and its transpose's rows.
const STORED: [f32; 6] = [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5];
const TRANSPOSED: [f32; 6] = [-2.5, 0.5, -1.5, 1.5, -0.5, 2.5];

fn assert_within(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!((g - w).abs() <= tol, "{what}: element {i} is {g}, the tensor holds {w}");
    }
}

/// int8 over [-2.5, 2.5]: one step is 5/255, so a round trip is within
/// 2.5/255 of the value. The result of the round trip is a host tensor.
#[test]
#[ignore = "requires CUDA GPU"]
fn qtensor_quantize_stages_a_gpu_tensor_and_reads_a_gpu_view_through_its_strides() {
    if !cuda_available() {
        return;
    }
    let gpu = gpu_tensor(&STORED, &[2, 3]);
    let view = nsl_tensor_transpose(gpu, 0, 1);
    for (t, want, what) in [(gpu, &STORED, "gpu tensor"), (view, &TRANSPOSED, "transposed gpu view")] {
        let q = nsl_qtensor_quantize(t, 0, 0, 0, 0);
        let d = nsl_qtensor_dequantize(q);
        let got: Vec<f32> = (0..6).map(|i| unsafe { *(nsl_tensor_data_ptr(d) as *const f32).add(i) }).collect();
        assert_within(&got, want, 2.5 / 255.0 * 1.001, what);
        nsl_qtensor_release(q);
        nsl_tensor_free(d);
    }
    nsl_tensor_free(view);
    nsl_tensor_free(gpu);
}

/// The 4-bit packers on the transposed GPU view, [K=3, N=2], one group of
/// three inputs per output. Output 0 holds (-2.5, -1.5, -0.5) and output 1
/// (0.5, 1.5, 2.5): each spans 2.0, so a step is 2/15 and a round trip is
/// within 1/15. A null Hessian makes GPTQ plain rounding, so both agree.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_4bit_packers_stage_a_gpu_view() {
    if !cuda_available() {
        return;
    }
    let gpu = gpu_tensor(&STORED, &[2, 3]);
    let view = nsl_tensor_transpose(gpu, 0, 1);
    let want: Vec<f64> = TRANSPOSED.iter().map(|v| f64::from(*v)).collect();

    let awq = nsl_awq_quantize(view, 3, 0);
    let gptq = nsl_gptq_quantize(view, 0, 3, 4);
    for (packed_ptr, what) in [(awq, "nsl_awq_quantize"), (gptq, "nsl_gptq_quantize")] {
        assert_ne!(packed_ptr, 0, "{what}");
        let packed = unsafe { &*(packed_ptr as *const AwqPackedWeight) };
        assert_eq!((packed.k, packed.n), (3, 2), "{what}");
        let got = awq_dequantize_cpu(packed);
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1.0 / 15.0 * 1.001, "{what}: element {i} is {g}, the view holds {w}");
        }
    }
    nsl_awq_free(awq);
    nsl_gptq_free(gptq);
    nsl_tensor_free(view);
    nsl_tensor_free(gpu);
}
