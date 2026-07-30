//! Timing probe: the v1 fused linear-CE kernels at PRODUCTION shape
//! (Coder-50M head: rows=1024, V=49152, H=512).
//!
//! Exists to answer, with a measurement instead of a triage guess, whether
//! the v1 backward — a per-row-CTA full-vocab recompute scattering dx/dW
//! through `red.global.add.f32` (rows·V·H ≈ 2.6e10 atomic adds per output
//! pair at this shape) — is usable as the production training path, or
//! whether wiring the auto-substitution must route large-vocab shapes
//! through a GEMM-based path instead. For scale: the composite path's head
//! GEMM chain is ~51.5 GFLOP, ≈1.4 ms forward / ≈2.9 ms backward at the
//! measured 36 TFLOPS TF32 rate.
//!
//! Run:
//! ```bash
//! cargo test -p nsl-codegen --features cuda --test fused_linear_ce_perf_probe \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

use nsl_codegen::fused_linear_ce::{
    synthesize_fused_linear_ce_backward_ptx, synthesize_fused_linear_ce_ptx, Dtype,
    FusedLinearCEConfig,
};
use nsl_runtime::{
    nsl_cuda_init, nsl_fused_linear_ce_backward, nsl_fused_linear_ce_backward_gemm,
    nsl_fused_linear_ce_forward_gemm, nsl_fused_linear_ce_forward_large,
    nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free, nsl_test_cuda_h2d,
};

const B: usize = 1;
const S: usize = 1024;
const V: usize = 49152;
const H: usize = 512;
const VOCAB_TILE: u32 = 1024;

fn fill_seeded(dst: &mut [f32], seed: u64) {
    let mut s = seed;
    for x in dst.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 33) as u32;
        *x = ((u as f32) / (u32::MAX as f32)) * 0.6 - 0.3;
    }
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    unsafe { nsl_cuda_init() == 0 }
}

fn alloc_upload_f32(host: &[f32]) -> i64 {
    let bytes = host.len() * 4;
    let d = unsafe { nsl_test_cuda_alloc(bytes as i64) };
    assert!(d != 0);
    unsafe { nsl_test_cuda_h2d(d, host.as_ptr() as i64, bytes as i64) };
    d
}

/// 4-byte readback = stream drain; keeps the probe free of test-hooks-only
/// sync helpers.
fn sync_via_readback(dev: i64) {
    let mut one = [0f32; 1];
    unsafe { nsl_test_cuda_d2h(one.as_mut_ptr() as i64, dev, 4) };
}

#[test]
#[ignore = "requires CUDA GPU; perf probe, prints timings"]
fn v1_kernels_at_production_shape() {
    if !cuda_available() {
        return;
    }
    let rows = B * S;
    let cfg = FusedLinearCEConfig {
        vocab_size: V as u32,
        hidden_size: H as u32,
        seq_len: S as u32,
        batch_size: B as u32,
        vocab_tile: VOCAB_TILE,
        dtype: Dtype::F32,
        max_vocab_v1: 262_144, // opt into the large-vocab (cross-CTA) path
        ..Default::default()
    };
    cfg.validate().expect("valid config");
    let num_tiles = (V as u32).div_ceil(VOCAB_TILE) as usize;

    let fwd_ptx = synthesize_fused_linear_ce_ptx(&cfg);
    let bwd_ptx = synthesize_fused_linear_ce_backward_ptx(&cfg);
    let partials_kname =
        std::ffi::CString::new(cfg.large_partials_kernel_name()).unwrap();
    let finalize_kname =
        std::ffi::CString::new(cfg.large_finalize_kernel_name()).unwrap();
    let bwd_kname = std::ffi::CString::new(cfg.bwd_kernel_name()).unwrap();

    let mut x = vec![0f32; rows * H];
    let mut w = vec![0f32; V * H];
    let bias = vec![0f32; V];
    fill_seeded(&mut x, 0xC0DE);
    fill_seeded(&mut w, 0xFEED);
    let targets: Vec<i64> = (0..rows).map(|i| ((i * 977) % V) as i64).collect();

    let x_d = alloc_upload_f32(&x);
    let w_d = alloc_upload_f32(&w);
    let bias_d = alloc_upload_f32(&bias);
    let tgt_bytes = rows * 8;
    let tgt_d = unsafe { nsl_test_cuda_alloc(tgt_bytes as i64) };
    unsafe { nsl_test_cuda_h2d(tgt_d, targets.as_ptr() as i64, tgt_bytes as i64) };
    let loss_d = unsafe { nsl_test_cuda_alloc((rows * 4) as i64) };
    let lse_d = unsafe { nsl_test_cuda_alloc((rows * 4) as i64) };
    let partials_d =
        unsafe { nsl_test_cuda_alloc((rows * num_tiles * 2 * 4) as i64) };
    let dx_d = unsafe { nsl_test_cuda_alloc((rows * H * 4) as i64) };
    let dw_d = unsafe { nsl_test_cuda_alloc((V * H * 4) as i64) };
    let dbias_d = unsafe { nsl_test_cuda_alloc((V * 4) as i64) };

    let smem = cfg.shared_mem_bytes() as i64;
    let run_fwd = || {
        let rc = nsl_fused_linear_ce_forward_large(
            fwd_ptx.as_ptr() as i64,
            partials_kname.as_ptr() as i64,
            finalize_kname.as_ptr() as i64,
            x_d,
            w_d,
            bias_d,
            tgt_d,
            partials_d,
            loss_d,
            lse_d,
            B as i64,
            S as i64,
            V as i64,
            H as i64,
            num_tiles as i64,
            smem,
            0,
        );
        assert_eq!(rc, 0, "forward_large rc={rc}");
    };
    let run_bwd = || {
        let rc = nsl_fused_linear_ce_backward(
            bwd_ptx.as_ptr() as i64,
            bwd_kname.as_ptr() as i64,
            1.0f32.to_bits() as i64,
            x_d,
            w_d,
            bias_d,
            tgt_d,
            lse_d,
            dx_d,
            dw_d,
            dbias_d,
            B as i64,
            S as i64,
            V as i64,
            H as i64,
            rows as i64,
            smem,
            0,
        );
        assert_eq!(rc, 0, "backward rc={rc}");
    };

    // Forward timing.
    run_fwd();
    sync_via_readback(loss_d);
    let t0 = std::time::Instant::now();
    let fwd_iters = 5;
    for _ in 0..fwd_iters {
        run_fwd();
    }
    sync_via_readback(loss_d);
    let fwd_ms = t0.elapsed().as_secs_f64() * 1e3 / fwd_iters as f64;

    // Backward timing (buffers are accumulated into, not zeroed between
    // iterations — fine for timing, the arithmetic cost is identical).
    run_bwd();
    sync_via_readback(dx_d);
    let t1 = std::time::Instant::now();
    let bwd_iters = 3;
    for _ in 0..bwd_iters {
        run_bwd();
    }
    sync_via_readback(dx_d);
    let bwd_ms = t1.elapsed().as_secs_f64() * 1e3 / bwd_iters as f64;

    let gflop = 2.0 * rows as f64 * V as f64 * H as f64 / 1e9;
    println!(
        "PROBE rows={rows} V={V} H={H}: fwd {fwd_ms:.2} ms ({:.1} GFLOP eq, {:.1} TFLOPS-eq), \
         bwd {bwd_ms:.2} ms; composite-GEMM reference ~1.4 ms fwd / ~2.9 ms bwd at 36 TFLOPS",
        gflop,
        gflop / fwd_ms / 1e9 * 1e12 / 1e3,
    );

    for d in [x_d, w_d, bias_d, tgt_d, loss_d, lse_d, partials_d, dx_d, dw_d, dbias_d] {
        unsafe { nsl_test_cuda_free(d) };
    }
}

/// The GEMM-chunked path at the same production shape, biasless (the
/// coder50m head). Compare against `v1_kernels_at_production_shape` above.
#[test]
#[ignore = "requires CUDA GPU; perf probe, prints timings"]
fn gemm_path_at_production_shape() {
    if !cuda_available() {
        return;
    }
    let rows = B * S;
    let mut x = vec![0f32; rows * H];
    let mut w = vec![0f32; V * H];
    fill_seeded(&mut x, 0xC0DE);
    fill_seeded(&mut w, 0xFEED);
    let targets: Vec<i64> = (0..rows).map(|i| ((i * 977) % V) as i64).collect();

    let x_d = alloc_upload_f32(&x);
    let w_d = alloc_upload_f32(&w);
    let tgt_d = unsafe { nsl_test_cuda_alloc((rows * 8) as i64) };
    unsafe { nsl_test_cuda_h2d(tgt_d, targets.as_ptr() as i64, (rows * 8) as i64) };
    let loss_d = unsafe { nsl_test_cuda_alloc((rows * 4) as i64) };
    let lse_d = unsafe { nsl_test_cuda_alloc((rows * 4) as i64) };
    let dx_d = unsafe { nsl_test_cuda_alloc((rows * H * 4) as i64) };
    let dw_d = unsafe { nsl_test_cuda_alloc((V * H * 4) as i64) };

    let run_fwd = || {
        let rc = nsl_fused_linear_ce_forward_gemm(
            x_d, w_d, 0, tgt_d, loss_d, lse_d, B as i64, S as i64, V as i64, H as i64, 0,
        );
        assert_eq!(rc, 0, "forward_gemm rc={rc}");
    };
    let run_bwd = || {
        let rc = nsl_fused_linear_ce_backward_gemm(
            1.0f32.to_bits() as i64,
            x_d,
            w_d,
            0,
            tgt_d,
            lse_d,
            dx_d,
            dw_d,
            0,
            B as i64,
            S as i64,
            V as i64,
            H as i64,
            rows as i64,
            0,
        );
        assert_eq!(rc, 0, "backward_gemm rc={rc}");
    };

    for _ in 0..3 {
        run_fwd();
    }
    sync_via_readback(loss_d);
    let t0 = std::time::Instant::now();
    let iters = 20;
    for _ in 0..iters {
        run_fwd();
    }
    sync_via_readback(loss_d);
    let fwd_ms = t0.elapsed().as_secs_f64() * 1e3 / iters as f64;

    for _ in 0..3 {
        run_bwd();
    }
    sync_via_readback(dx_d);
    let t1 = std::time::Instant::now();
    for _ in 0..iters {
        run_bwd();
    }
    sync_via_readback(dx_d);
    let bwd_ms = t1.elapsed().as_secs_f64() * 1e3 / iters as f64;

    let gflop = 2.0 * rows as f64 * V as f64 * H as f64 / 1e9;
    println!(
        "PROBE-GEMM rows={rows} V={V} H={H}: fwd {fwd_ms:.2} ms ({:.1} TFLOPS-eq), \
         bwd {bwd_ms:.2} ms ({:.1} TFLOPS-eq over 3x FLOPs); v1 measured 486 / 2222 ms",
        gflop / fwd_ms, // GFLOP per ms == TFLOPS
        3.0 * gflop / bwd_ms,
    );

    for d in [x_d, w_d, tgt_d, loss_d, lse_d, dx_d, dw_d] {
        unsafe { nsl_test_cuda_free(d) };
    }
}
