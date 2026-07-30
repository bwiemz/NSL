//! Device gates for the fp8 host-path FFIs (the tape-AD `@fp8_compute` crash).
//!
//! Every fp8 FFI walks `t.data` with `from_raw_parts` as host memory. Before
//! the `stage_for_host_read` guard, a GPU-resident tensor here dereferenced a
//! CUDA device pointer on the CPU — a segfault, not a wrong answer. The live
//! path: tape-AD backward for `TapeOp::Fp8MatMul` hands GPU gradients
//! straight into `calibrate_gradient_scale` and `nsl_fp8_cast`
//! (`autodiff/backward.rs`), so `@fp8_compute` + `--tape-ad` + GPU crashed at
//! the first backward step. Source AD refuses `@fp8_compute` and points at
//! tape AD, which made this the only path — and it didn't survive.
//!
//! These gates run every guarded FFI on GPU tensors and require the result
//! to MATCH the CPU run on the same values — staging must be a device
//! round-trip, not a refusal, or fp8 training on GPU would be impossible.
//!
//! Run: cargo test -p nsl-runtime --features cuda --test fp8_gpu_device_guard
//!      -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use nsl_runtime::fp8::{
    fp8_matmul_e5m2_backward, nsl_fp8_cast, nsl_fp8_compute_scale, nsl_fp8_gradient_scale,
    nsl_fp8_matmul_training, nsl_mxfp8_quantize, nsl_nvfp4_quantize, FP8_FORMAT_E4M3,
    FP8_FORMAT_E5M2,
};
use nsl_runtime::list::{nsl_list_free, nsl_list_get, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_transpose, nsl_tensor_zeros_on,
};
use nsl_runtime::{nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d};

/// Pin cuBLAS to full f32 before any GPU work (same rationale and soundness
/// argument as `gpu_strided_view_ops::pin_full_f32`): these gates compare a
/// GPU arm against a CPU arm to catch DEVICE-HANDLING bugs, which err by
/// order 1e0 or crash outright; TF32's ~1e-3 drift on the matmul gates would
/// only spend headroom.
fn pin_full_f32() {
    std::env::set_var("NSL_MATMUL_TF32", "0");
    std::env::set_var("NSL_MATMUL_BF16", "0");
}

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
    let t = nsl_tensor_zeros_on(shape_list, 1 /* GPU */);
    nsl_list_free(shape_list);
    nsl_test_cuda_h2d(
        nsl_tensor_data_ptr(t),
        vals.as_ptr() as i64,
        (vals.len() * 4) as i64,
    );
    t
}

fn cpu_tensor(vals: &[f32], shape: &[i64]) -> i64 {
    let shape_list = nsl_list_new();
    for &dim in shape {
        nsl_list_push(shape_list, dim);
    }
    let t = nsl_tensor_zeros_on(shape_list, 0 /* CPU */);
    nsl_list_free(shape_list);
    let dst = nsl_tensor_data_ptr(t) as *mut f32;
    for (i, &v) in vals.iter().enumerate() {
        unsafe { *dst.add(i) = v };
    }
    t
}

fn read_gpu(t: i64, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    nsl_test_cuda_d2h(
        out.as_mut_ptr() as i64,
        nsl_tensor_data_ptr(t),
        (len * 4) as i64,
    );
    out
}

fn read_cpu(t: i64, len: usize) -> Vec<f32> {
    let src = nsl_tensor_data_ptr(t) as *const f32;
    (0..len).map(|i| unsafe { *src.add(i) }).collect()
}

/// Deterministic pseudo-random values (LCG), same generator as the flash GPU tests.
fn det_seq(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as f32 / 65535.0) - 0.5
        })
        .collect()
}

/// `nsl_fp8_cast` on a GPU tensor: no segfault, result lives on the GPU,
/// and the values are bit-identical to the CPU run (the quantization itself
/// happens on the host in both arms).
#[test]
#[ignore = "requires CUDA GPU"]
fn fp8_cast_of_a_gpu_tensor_matches_the_cpu_run() {
    pin_full_f32();
    if !cuda_available() {
        return;
    }
    let vals = det_seq(51, 24);
    let gpu = gpu_tensor(&vals, &[4, 6]);
    let cpu = cpu_tensor(&vals, &[4, 6]);

    let out_gpu = nsl_fp8_cast(gpu, FP8_FORMAT_E4M3, 0.0);
    let out_cpu = nsl_fp8_cast(cpu, FP8_FORMAT_E4M3, 0.0);

    // If the output were mislabeled (host allocation stamped device=1, the
    // pre-fix behavior), this D2H read would fault or return garbage.
    let got = read_gpu(out_gpu, 24);
    let expect = read_cpu(out_cpu, 24);
    assert_eq!(got, expect, "GPU cast diverged from the CPU run");

    nsl_tensor_free(out_gpu);
    nsl_tensor_free(out_cpu);
    nsl_tensor_free(gpu);
    nsl_tensor_free(cpu);
}

/// The scalar probes (`nsl_fp8_compute_scale`, `nsl_fp8_gradient_scale`)
/// on GPU tensors: no segfault, exact agreement with the CPU run.
#[test]
#[ignore = "requires CUDA GPU"]
fn fp8_scale_probes_of_gpu_tensors_match_the_cpu_run() {
    pin_full_f32();
    if !cuda_available() {
        return;
    }
    let vals = det_seq(52, 33);
    let gpu = gpu_tensor(&vals, &[33]);
    let cpu = cpu_tensor(&vals, &[33]);

    // Not bit-equal by contract: the D2H staging copy follows the runtime
    // dtype convention (CPU=f64), so the staged arm computes amax/fp8_max in
    // f64 while the native-f32 CPU arm rounds the division in f32. Same
    // values in, difference bounded by one ulp of the f32 division.
    let s_gpu = nsl_fp8_compute_scale(gpu, FP8_FORMAT_E5M2);
    let s_cpu = nsl_fp8_compute_scale(cpu, FP8_FORMAT_E5M2);
    assert!(
        (s_gpu - s_cpu).abs() <= s_cpu.abs() * 1e-6,
        "compute_scale diverged between devices beyond f32 rounding: {s_gpu} vs {s_cpu}"
    );
    assert_eq!(
        nsl_fp8_gradient_scale(gpu),
        nsl_fp8_gradient_scale(cpu),
        "gradient_scale diverged between devices"
    );

    nsl_tensor_free(gpu);
    nsl_tensor_free(cpu);
}

/// `fp8_matmul_e5m2_backward` with GPU operands, one of them a zero-copy
/// transposed view — the exact call shape the tape backward produces
/// (`tensor_transpose(saved_b)` fed straight in). Pre-fix this was the
/// segfault site; the view half is the wrong-gradient hazard.
#[test]
#[ignore = "requires CUDA GPU"]
fn fp8_e5m2_backward_with_gpu_view_operands_matches_the_cpu_run() {
    pin_full_f32();
    if !cuda_available() {
        return;
    }
    let (m, k, n) = (3usize, 4usize, 5usize);
    let g_vals = det_seq(53, m * n);
    let b_vals = det_seq(54, k * n); // B is [k,n]; backward multiplies by B^T

    let g_gpu = gpu_tensor(&g_vals, &[m as i64, n as i64]);
    let b_gpu = gpu_tensor(&b_vals, &[k as i64, n as i64]);
    let g_cpu = cpu_tensor(&g_vals, &[m as i64, n as i64]);
    let b_cpu = cpu_tensor(&b_vals, &[k as i64, n as i64]);

    let scale_g = nsl_fp8_gradient_scale(g_cpu) as f32;
    let scale_b = nsl_fp8_gradient_scale(b_cpu) as f32;

    let bt_gpu = nsl_tensor_transpose(b_gpu, 0, 1); // zero-copy [n,k] view
    let bt_cpu = nsl_tensor_transpose(b_cpu, 0, 1);

    let out_gpu = fp8_matmul_e5m2_backward(g_gpu, bt_gpu, scale_g, scale_b);
    let out_cpu = fp8_matmul_e5m2_backward(g_cpu, bt_cpu, scale_g, scale_b);

    let got = read_gpu(out_gpu, m * k);
    let expect = read_cpu(out_cpu, m * k);
    let max_diff = got
        .iter()
        .zip(expect.iter())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    assert!(
        max_diff < 1e-4,
        "E5M2 backward diverged between devices (max_abs_diff={max_diff:.3e})"
    );

    nsl_tensor_free(out_gpu);
    nsl_tensor_free(out_cpu);
    nsl_tensor_free(bt_gpu);
    nsl_tensor_free(bt_cpu);
    nsl_tensor_free(g_gpu);
    nsl_tensor_free(b_gpu);
    nsl_tensor_free(g_cpu);
    nsl_tensor_free(b_cpu);
}

/// The full production path: `nsl_fp8_matmul_training` on GPU tensors under
/// tape recording, then `nsl_tape_backward`. This is the exact sequence that
/// segfaulted (backward reaches `calibrate_gradient_scale(g)` with a GPU
/// gradient). Gradients must match the CPU tape run on the same values.
#[test]
#[ignore = "requires CUDA GPU"]
fn tape_fp8_training_backward_on_gpu_matches_the_cpu_run() {
    pin_full_f32();
    if !cuda_available() {
        return;
    }
    use nsl_runtime::autodiff::{nsl_tape_backward, nsl_tape_start, nsl_tape_stop};

    let (m, k, n) = (2usize, 3usize, 4usize);
    let a_vals = det_seq(55, m * k);
    let b_vals = det_seq(56, k * n);

    let run = |device_gpu: bool| -> (Vec<f32>, Vec<f32>) {
        let (a, b) = if device_gpu {
            (
                gpu_tensor(&a_vals, &[m as i64, k as i64]),
                gpu_tensor(&b_vals, &[k as i64, n as i64]),
            )
        } else {
            (
                cpu_tensor(&a_vals, &[m as i64, k as i64]),
                cpu_tensor(&b_vals, &[k as i64, n as i64]),
            )
        };
        let params = nsl_list_new();
        nsl_list_push(params, a);
        nsl_list_push(params, b);
        nsl_tape_start(params);
        let out = nsl_fp8_matmul_training(a, b, 0);
        let grads = nsl_tape_backward(out, params);
        let ga_ptr = nsl_list_get(grads, 0);
        let gb_ptr = nsl_list_get(grads, 1);
        let (ga, gb) = if device_gpu {
            (read_gpu(ga_ptr, m * k), read_gpu(gb_ptr, k * n))
        } else {
            (read_cpu(ga_ptr, m * k), read_cpu(gb_ptr, k * n))
        };
        // nsl_list_free frees only the list storage, not the elements.
        nsl_tensor_free(ga_ptr);
        nsl_tensor_free(gb_ptr);
        nsl_list_free(grads);
        nsl_tape_stop();
        nsl_list_free(params);
        nsl_tensor_free(out);
        nsl_tensor_free(a);
        nsl_tensor_free(b);
        (ga, gb)
    };

    let (ga_cpu, gb_cpu) = run(false);
    let (ga_gpu, gb_gpu) = run(true);

    for (name, gpu, cpu) in [("grad_A", &ga_gpu, &ga_cpu), ("grad_B", &gb_gpu, &gb_cpu)] {
        let max_diff = gpu
            .iter()
            .zip(cpu.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_diff < 1e-4,
            "{name} diverged between the GPU and CPU tape runs (max_abs_diff={max_diff:.3e})"
        );
    }
}

/// The block quantizers on GPU tensors: no segfault, device round-trip,
/// values bit-identical to the CPU run.
#[test]
#[ignore = "requires CUDA GPU"]
fn block_quantizers_on_gpu_tensors_match_the_cpu_run() {
    pin_full_f32();
    if !cuda_available() {
        return;
    }
    let vals = det_seq(57, 64);
    let gpu = gpu_tensor(&vals, &[64]);
    let cpu = cpu_tensor(&vals, &[64]);

    let mx_gpu = nsl_mxfp8_quantize(gpu, 32, FP8_FORMAT_E4M3);
    let mx_cpu = nsl_mxfp8_quantize(cpu, 32, FP8_FORMAT_E4M3);
    assert_eq!(
        read_gpu(mx_gpu, 64),
        read_cpu(mx_cpu, 64),
        "mxfp8 diverged between devices"
    );

    let nv_gpu = nsl_nvfp4_quantize(gpu, 64, 1);
    let nv_cpu = nsl_nvfp4_quantize(cpu, 64, 1);
    assert_eq!(
        read_gpu(nv_gpu, 64),
        read_cpu(nv_cpu, 64),
        "nvfp4 diverged between devices"
    );

    for t in [mx_gpu, mx_cpu, nv_gpu, nv_cpu, gpu, cpu] {
        nsl_tensor_free(t);
    }
}
