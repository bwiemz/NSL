//! G3 — Fused linear-CE end-to-end numerical validation test.
//!
//! Tests the forward + backward kernels emitted by
//! `nsl_codegen::fused_linear_ce` against a CPU f64 reference.
//!
//! ## Fixture
//! B=2, S=64, V=4096, H=128, dtype=F32, vocab_tile=1024.
//! Targets: 128 entries with -100 at every position where (b*S + s) % 16 == 0
//! (8 ignored, 120 valid).
//!
//! ## Tolerances (documented in design)
//! * Forward loss: |gpu - ref| / max(|ref|, 1.0) < 1e-3
//! * Backward dx:  max_abs |dx_gpu - dx_ref| < 5e-3
//! * Skip identity: dx_gpu[row, :] == 0 exactly for all -100 rows
//!
//! ## Running
//! ```bash
//! cargo test -p nsl-codegen --features cuda --test fused_linear_ce_numerical \
//!     -- --ignored --nocapture --test-threads=1
//! ```
//! `--test-threads=1` is required: the CUDA driver singleton is process-global.
//! `#[ignore]` gates on live CUDA device availability.

#![cfg(feature = "cuda")]

use nsl_codegen::fused_linear_ce::{
    Dtype, FusedLinearCEConfig, synthesize_fused_linear_ce_ptx,
    synthesize_fused_linear_ce_backward_ptx,
};
use nsl_runtime::{
    nsl_cuda_init, nsl_test_cuda_alloc, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_d2h,
    nsl_fused_linear_ce_forward, nsl_fused_linear_ce_backward,
};

// ─── Constants ────────────────────────────────────────────────────────────────

const B: usize = 2;
const S: usize = 64;
const V: usize = 4096;
const H: usize = 128;
const VOCAB_TILE: u32 = 1024;
const IGNORE_INDEX: i64 = -100;

// ─── PRNG ─────────────────────────────────────────────────────────────────────

/// Deterministic LCG PRNG — reproducible across platforms.
fn fill_seeded(dst: &mut [f32], seed: u64) {
    let mut s = seed;
    for x in dst.iter_mut() {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u = (s >> 33) as u32;
        // [-0.3, 0.3) — small magnitudes keep softmax numerics calm
        *x = ((u as f32) / (u32::MAX as f32)) * 0.6 - 0.3;
    }
}

// ─── CUDA guard ───────────────────────────────────────────────────────────────

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        eprintln!("skipping: NSL_SKIP_CUDA_TESTS set");
        return false;
    }
    let rc = unsafe { nsl_cuda_init() };
    if rc != 0 {
        eprintln!("skipping: nsl_cuda_init returned {}", rc);
        false
    } else {
        true
    }
}

// ─── CPU reference (f64) ─────────────────────────────────────────────────────

/// Forward reference: loss = mean over valid positions of
///   -log_softmax(logits)[target]
/// where logits = x @ W^T + bias.
fn cpu_reference_forward(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    targets: &[i64],
) -> (f64, Vec<f64>) {
    let rows = B * S;
    // Compute logits in f64 for each row.
    let mut logits = vec![0f64; rows * V];
    for row in 0..rows {
        for vi in 0..V {
            let mut acc = bias[vi] as f64;
            for hi in 0..H {
                acc += x[row * H + hi] as f64 * w[vi * H + hi] as f64;
            }
            logits[row * V + vi] = acc;
        }
    }

    // Forward loss and per-row lse.
    let mut lse = vec![0f64; rows];
    let mut total_loss = 0f64;
    let mut num_valid = 0usize;

    for row in 0..rows {
        let t = targets[row];
        if t == IGNORE_INDEX {
            // skip
            continue;
        }
        let row_logits = &logits[row * V..(row + 1) * V];
        let max_logit = row_logits.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let sum_exp: f64 = row_logits.iter().map(|&l| (l - max_logit).exp()).sum();
        let lse_val = sum_exp.ln() + max_logit;
        lse[row] = lse_val;
        let tgt = t as usize;
        let loss_row = -(row_logits[tgt] - lse_val);
        total_loss += loss_row;
        num_valid += 1;
    }

    let mean_loss = total_loss / num_valid.max(1) as f64;
    (mean_loss, lse)
}

/// Backward reference: compute dx for a mean CE loss given saved lse.
/// Returns dx[B*S, H] and dlogits[B*S, V] (for debugging).
fn cpu_reference_backward(
    x: &[f32],
    w: &[f32],
    bias: &[f32],
    targets: &[i64],
    lse: &[f64],
    grad_output: f64,
) -> (Vec<f32>, Vec<f32>) {
    let rows = B * S;
    let num_valid = targets.iter().filter(|&&t| t != IGNORE_INDEX).count();
    let inv_nv = grad_output / num_valid.max(1) as f64;

    // Compute logits again.
    let mut logits = vec![0f64; rows * V];
    for row in 0..rows {
        for vi in 0..V {
            let mut acc = bias[vi] as f64;
            for hi in 0..H {
                acc += x[row * H + hi] as f64 * w[vi * H + hi] as f64;
            }
            logits[row * V + vi] = acc;
        }
    }

    let mut dx = vec![0f32; rows * H];
    let mut dlogits = vec![0f32; rows * V];

    for row in 0..rows {
        let t = targets[row];
        if t == IGNORE_INDEX {
            // dx stays 0 (skip)
            continue;
        }
        let tgt = t as usize;
        let row_logits = &logits[row * V..(row + 1) * V];
        let lse_val = lse[row];

        // dlogits_v = (softmax_v - 1{v==tgt}) * inv_nv
        for vi in 0..V {
            let p_v = (row_logits[vi] - lse_val).exp();
            let one_hot = if vi == tgt { 1.0f64 } else { 0.0 };
            let dl = (p_v - one_hot) * inv_nv;
            dlogits[row * V + vi] = dl as f32;
        }

        // dx[row, h] = sum_v dlogits[v] * W[v, h]
        for hi in 0..H {
            let mut acc = 0f64;
            for vi in 0..V {
                acc += dlogits[row * V + vi] as f64 * w[vi * H + hi] as f64;
            }
            dx[row * H + hi] = acc as f32;
        }
    }

    (dx, dlogits)
}

// ─── PTX snapshot tests (no GPU, always-on) ──────────────────────────────────

#[test]
fn ptx_snapshot_starts_with_header() {
    let cfg = FusedLinearCEConfig::default();
    let ptx = synthesize_fused_linear_ce_ptx(&cfg);
    let s = std::str::from_utf8(&ptx).unwrap();
    assert!(s.starts_with(".version 7.0\n.target sm_80"), "got: {}", &s[..40.min(s.len())]);
}

#[test]
fn ptx_snapshot_contains_extern_shared() {
    let cfg = FusedLinearCEConfig::default();
    let ptx = synthesize_fused_linear_ce_ptx(&cfg);
    let s = std::str::from_utf8(&ptx).unwrap();
    assert!(s.contains(".extern .shared"), "missing .extern .shared");
}

#[test]
fn ptx_snapshot_contains_skip_predicate() {
    let cfg = FusedLinearCEConfig::default();
    let ptx = synthesize_fused_linear_ce_ptx(&cfg);
    let s = std::str::from_utf8(&ptx).unwrap();
    // Forward kernel must have both setp.eq.s64 and the -100 literal.
    assert!(s.contains("setp.eq.s64"), "missing setp.eq.s64 in forward PTX");
    assert!(s.contains("-100"), "missing -100 ignore_index in forward PTX");
}

#[test]
fn ptx_snapshot_contains_kernel_name() {
    let cfg = FusedLinearCEConfig::default();
    let ptx = synthesize_fused_linear_ce_ptx(&cfg);
    let s = std::str::from_utf8(&ptx).unwrap();
    assert!(s.contains(&cfg.kernel_name()), "kernel name {} not found", cfg.kernel_name());
}

#[test]
fn ptx_is_ascii_only() {
    let cfg = FusedLinearCEConfig::default();
    let fwd = synthesize_fused_linear_ce_ptx(&cfg);
    let bwd = synthesize_fused_linear_ce_backward_ptx(&cfg);
    for b in fwd.iter().chain(bwd.iter()) {
        assert!(*b < 128, "non-ASCII byte 0x{:02x} in PTX", b);
    }
}

// ─── GPU test ────────────────────────────────────────────────────────────────

#[test]
#[ignore]
fn fused_linear_ce_gpu_forward_and_backward() {
    if !cuda_available() {
        return;
    }

    let cfg = FusedLinearCEConfig {
        vocab_size: V as u32,
        hidden_size: H as u32,
        seq_len: S as u32,
        batch_size: B as u32,
        vocab_tile: VOCAB_TILE,
        gpu_sm: 80,
        dtype: Dtype::F32,
        ignore_index: IGNORE_INDEX,
        max_vocab_v1: 8192,
    };
    cfg.validate().expect("config must be valid");

    // Build targets: -100 at positions where idx % 16 == 0.
    let rows = B * S;
    let mut targets = vec![0i64; rows];
    for (i, t) in targets.iter_mut().enumerate() {
        if i % 16 == 0 {
            *t = IGNORE_INDEX;
        } else {
            // Valid target: random within [0, V)
            *t = ((i * 37 + 13) % V) as i64;
        }
    }
    let num_valid = targets.iter().filter(|&&t| t != IGNORE_INDEX).count() as u32;
    eprintln!("fixture: B={B} S={S} V={V} H={H} num_valid={num_valid}");

    // Host buffers.
    let mut x_host = vec![0f32; rows * H];
    let mut w_host = vec![0f32; V * H];
    let mut bias_host = vec![0f32; V];
    fill_seeded(&mut x_host, 42);
    fill_seeded(&mut w_host, 137);
    fill_seeded(&mut bias_host, 7);

    // ── 1. Synthesise PTX ────────────────────────────────────────────────

    let mut fwd_ptx = synthesize_fused_linear_ce_ptx(&cfg);
    fwd_ptx.push(0u8); // NUL-terminate for CUDA driver
    let mut bwd_ptx = synthesize_fused_linear_ce_backward_ptx(&cfg);
    bwd_ptx.push(0u8);

    let fwd_name = format!("{}\0", cfg.kernel_name());
    let bwd_name = format!("{}\0", cfg.bwd_kernel_name());

    // Write PTX to temp for inspection.
    let tmp = std::env::temp_dir();
    std::fs::write(tmp.join("fused_linear_ce_fwd.ptx"), &fwd_ptx[..fwd_ptx.len()-1])
        .ok();
    std::fs::write(tmp.join("fused_linear_ce_bwd.ptx"), &bwd_ptx[..bwd_ptx.len()-1])
        .ok();
    eprintln!("PTX written to {}", tmp.display());

    // ── 2. Allocate device buffers ────────────────────────────────────────

    let x_bytes      = (rows * H * 4) as i64;
    let w_bytes      = (V * H * 4) as i64;
    let bias_bytes   = (V * 4) as i64;
    let tgt_bytes    = (rows * 8) as i64; // i64 targets
    let loss_bytes   = (rows * 4) as i64;
    let lse_bytes    = (rows * 4) as i64;
    let dx_bytes     = (rows * H * 4) as i64;
    let dw_bytes     = (V * H * 4) as i64;
    let dbias_bytes  = (V * 4) as i64;

    let x_dev      = unsafe { nsl_test_cuda_alloc(x_bytes) };
    let w_dev      = unsafe { nsl_test_cuda_alloc(w_bytes) };
    let bias_dev   = unsafe { nsl_test_cuda_alloc(bias_bytes) };
    let tgt_dev    = unsafe { nsl_test_cuda_alloc(tgt_bytes) };
    let loss_dev   = unsafe { nsl_test_cuda_alloc(loss_bytes) };
    let lse_dev    = unsafe { nsl_test_cuda_alloc(lse_bytes) };
    let dx_dev     = unsafe { nsl_test_cuda_alloc(dx_bytes) };
    let dw_dev     = unsafe { nsl_test_cuda_alloc(dw_bytes) };
    let dbias_dev  = unsafe { nsl_test_cuda_alloc(dbias_bytes) };

    assert!(x_dev != 0 && w_dev != 0 && bias_dev != 0, "device alloc failed");
    assert!(tgt_dev != 0 && loss_dev != 0 && lse_dev != 0, "device alloc failed");
    assert!(dx_dev != 0 && dw_dev != 0 && dbias_dev != 0, "device alloc failed");

    // Upload host data to device.
    unsafe {
        nsl_test_cuda_h2d(x_dev,    x_host.as_ptr()    as i64, x_bytes);
        nsl_test_cuda_h2d(w_dev,    w_host.as_ptr()    as i64, w_bytes);
        nsl_test_cuda_h2d(bias_dev, bias_host.as_ptr() as i64, bias_bytes);
        nsl_test_cuda_h2d(tgt_dev,  targets.as_ptr()   as i64, tgt_bytes);
    }

    // ── 3. CPU reference forward ──────────────────────────────────────────

    let (ref_loss, ref_lse) = cpu_reference_forward(&x_host, &w_host, &bias_host, &targets);
    eprintln!("CPU reference loss = {:.6}", ref_loss);

    // ── 4. GPU forward ────────────────────────────────────────────────────

    let smem = cfg.shared_mem_bytes();
    eprintln!("SMEM = {} bytes per CTA", smem);

    let fwd_rc = unsafe {
        nsl_fused_linear_ce_forward(
            fwd_ptx.as_ptr() as i64,
            fwd_name.as_ptr() as i64,
            x_dev, w_dev, bias_dev, tgt_dev, loss_dev, lse_dev,
            B as i64, S as i64, V as i64, H as i64,
            smem as i64,
        )
    };
    assert_eq!(fwd_rc, 0, "nsl_fused_linear_ce_forward failed rc={}", fwd_rc);

    // Read back per-row losses and compute mean over valid rows.
    let mut loss_host = vec![0f32; rows];
    unsafe {
        nsl_test_cuda_d2h(loss_host.as_mut_ptr() as i64, loss_dev, loss_bytes);
    }

    let mut gpu_loss_sum = 0f64;
    let mut gpu_num_valid = 0usize;
    for (i, &loss_val) in loss_host.iter().enumerate() {
        if targets[i] != IGNORE_INDEX {
            gpu_loss_sum += loss_val as f64;
            gpu_num_valid += 1;
        }
    }
    let gpu_loss_mean = gpu_loss_sum / gpu_num_valid.max(1) as f64;
    eprintln!("GPU loss (mean over valid) = {:.6}", gpu_loss_mean);

    let loss_err = (gpu_loss_mean - ref_loss).abs() / ref_loss.abs().max(1.0);
    eprintln!("forward loss relative error = {:.2e}", loss_err);
    assert!(
        loss_err < 1e-3,
        "forward loss error {:.2e} exceeds 1e-3 tolerance (gpu={:.6}, ref={:.6})",
        loss_err, gpu_loss_mean, ref_loss
    );

    // ── 5. GPU backward ───────────────────────────────────────────────────

    // Zero gradient buffers before backward (atomic adds accumulate).
    {
        let zeros_dx    = vec![0f32; rows * H];
        let zeros_dw    = vec![0f32; V * H];
        let zeros_dbias = vec![0f32; V];
        unsafe {
            nsl_test_cuda_h2d(dx_dev,    zeros_dx.as_ptr()    as i64, dx_bytes);
            nsl_test_cuda_h2d(dw_dev,    zeros_dw.as_ptr()    as i64, dw_bytes);
            nsl_test_cuda_h2d(dbias_dev, zeros_dbias.as_ptr() as i64, dbias_bytes);
        }
    }

    // grad_output = 1.0 as f32 bits.
    let go_bits = 1.0f32.to_bits() as i64;

    let bwd_rc = unsafe {
        nsl_fused_linear_ce_backward(
            bwd_ptx.as_ptr() as i64,
            bwd_name.as_ptr() as i64,
            go_bits,
            x_dev, w_dev, bias_dev, tgt_dev, lse_dev,
            dx_dev, dw_dev, dbias_dev,
            B as i64, S as i64, V as i64, H as i64,
            num_valid as i64,
            smem as i64,
        )
    };
    assert_eq!(bwd_rc, 0, "nsl_fused_linear_ce_backward failed rc={}", bwd_rc);

    // Read back dx.
    let mut dx_gpu = vec![0f32; rows * H];
    unsafe {
        nsl_test_cuda_d2h(dx_gpu.as_mut_ptr() as i64, dx_dev, dx_bytes);
    }

    // ── 6. CPU backward reference ─────────────────────────────────────────

    let (dx_ref, _) = cpu_reference_backward(
        &x_host, &w_host, &bias_host, &targets, &ref_lse, 1.0,
    );

    // ── 7. Compare dx ─────────────────────────────────────────────────────

    let mut max_abs_err = 0f32;
    let mut n_checked = 0usize;
    for row in 0..rows {
        if targets[row] == IGNORE_INDEX {
            // Skip-identity check: GPU must write exactly 0.0.
            for hi in 0..H {
                let val = dx_gpu[row * H + hi];
                assert_eq!(
                    val, 0.0f32,
                    "skip-identity violated: dx_gpu[{row},{hi}] = {val} (expected 0.0)"
                );
            }
        } else {
            for hi in 0..H {
                let err = (dx_gpu[row * H + hi] - dx_ref[row * H + hi]).abs();
                if err > max_abs_err {
                    max_abs_err = err;
                }
                n_checked += 1;
            }
        }
    }

    eprintln!("backward dx max_abs_err = {:.2e} over {} elements", max_abs_err, n_checked);
    assert!(
        max_abs_err < 5e-3,
        "backward dx error {:.2e} exceeds 5e-3 tolerance",
        max_abs_err
    );

    // ── 8. Cleanup ────────────────────────────────────────────────────────

    unsafe {
        nsl_test_cuda_free(x_dev);
        nsl_test_cuda_free(w_dev);
        nsl_test_cuda_free(bias_dev);
        nsl_test_cuda_free(tgt_dev);
        nsl_test_cuda_free(loss_dev);
        nsl_test_cuda_free(lse_dev);
        nsl_test_cuda_free(dx_dev);
        nsl_test_cuda_free(dw_dev);
        nsl_test_cuda_free(dbias_dev);
    }

    eprintln!("PASS: forward loss err {:.2e} < 1e-3, backward dx err {:.2e} < 5e-3", loss_err, max_abs_err);
}
