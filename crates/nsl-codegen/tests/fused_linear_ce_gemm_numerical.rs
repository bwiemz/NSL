//! GEMM-chunked fused linear-CE (Sprint 2.5) — numerical validation vs the
//! shared CPU f64 reference, WITH-bias and BIASLESS, across a partial final
//! chunk (V = 6144 = one full 4096 chunk + one 2048 partial).
//!
//! The GEMM path's heavy math runs through cuBLAS, so this gate pins
//! `NSL_MATMUL_TF32=0` / `NSL_MATMUL_BF16=0` exactly like the other
//! operand-mapping value gates: at TF32 the ~1e-3 logits drift would eat the
//! 1e-5 gradient bounds that catch real wiring bugs (a crossed chunk offset,
//! a dropped partial chunk, a bias applied twice), which err by orders of
//! magnitude. TF32/BF16-mode behaviour is the matmul gates' jurisdiction.
//!
//! Every gradient bound is paired with a reference-magnitude assertion
//! (>= 10x the bound), per the item-6 lesson: a tolerance looser than the
//! signal passes an all-zero kernel.
//!
//! Run:
//! ```bash
//! cargo test -p nsl-codegen --features cuda --test fused_linear_ce_gemm_numerical \
//!     -- --ignored --nocapture --test-threads=1
//! ```

#![cfg(feature = "cuda")]

mod common;

use common::fused_lce_cpu_f64::{cpu_lce_backward_f64, cpu_lce_forward_f64};
use nsl_runtime::{
    nsl_cuda_init, nsl_fused_linear_ce_backward_gemm, nsl_fused_linear_ce_forward_gemm,
    nsl_test_cuda_alloc, nsl_test_cuda_d2h, nsl_test_cuda_free, nsl_test_cuda_h2d,
};

const B: usize = 2;
const S: usize = 64;
const V: usize = 6144; // 4096 + 2048: exercises the partial final chunk
const H: usize = 128;
const IGNORE_INDEX: i64 = -100;

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

fn pin_full_f32() {
    unsafe { std::env::set_var("NSL_MATMUL_TF32", "0") };
    unsafe { std::env::set_var("NSL_MATMUL_BF16", "0") };
}

fn alloc_upload_f32(host: &[f32]) -> i64 {
    let bytes = host.len() * 4;
    let d = unsafe { nsl_test_cuda_alloc(bytes as i64) };
    assert!(d != 0);
    unsafe { nsl_test_cuda_h2d(d, host.as_ptr() as i64, bytes as i64) };
    d
}

fn alloc_zeroed_f32(len: usize) -> i64 {
    let zeros = vec![0f32; len];
    alloc_upload_f32(&zeros)
}

fn download_f32(dev: i64, len: usize) -> Vec<f32> {
    let mut out = vec![0f32; len];
    unsafe { nsl_test_cuda_d2h(out.as_mut_ptr() as i64, dev, (len * 4) as i64) };
    out
}

struct Case {
    has_bias: bool,
}

fn run_case(case: &Case) {
    let rows = B * S;
    let mut x = vec![0f32; rows * H];
    let mut w = vec![0f32; V * H];
    let mut bias = vec![0f32; V];
    fill_seeded(&mut x, 0xA11CE);
    fill_seeded(&mut w, 0xB0B);
    if case.has_bias {
        fill_seeded(&mut bias, 0xB1A5);
    }
    // Targets with -100 at every 16th row (matches the v1 numerical fixture).
    let targets: Vec<i64> = (0..rows)
        .map(|i| {
            if i % 16 == 0 {
                IGNORE_INDEX
            } else {
                ((i * 2654435761) % V) as i64
            }
        })
        .collect();
    let num_valid = targets.iter().filter(|&&t| t != IGNORE_INDEX).count();

    // ── CPU f64 reference (zeros bias == no bias, mathematically) ──
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let w64: Vec<f64> = w.iter().map(|&v| v as f64).collect();
    let b64: Vec<f64> = bias.iter().map(|&v| v as f64).collect();
    let t32: Vec<i32> = targets.iter().map(|&t| t as i32).collect();
    let fwd_ref = cpu_lce_forward_f64(&x64, &w64, &b64, &t32, rows, V, H);
    let bwd_ref = cpu_lce_backward_f64(
        &x64,
        &w64,
        &b64,
        &t32,
        &fwd_ref.lse_per_row,
        1.0,
        rows,
        V,
        H,
    );

    // ── GPU GEMM path ──
    let x_d = alloc_upload_f32(&x);
    let w_d = alloc_upload_f32(&w);
    let bias_d = if case.has_bias { alloc_upload_f32(&bias) } else { 0 };
    let tgt_d = unsafe { nsl_test_cuda_alloc((rows * 8) as i64) };
    unsafe { nsl_test_cuda_h2d(tgt_d, targets.as_ptr() as i64, (rows * 8) as i64) };
    let loss_d = alloc_zeroed_f32(rows);
    let lse_d = alloc_zeroed_f32(rows);

    let rc = nsl_fused_linear_ce_forward_gemm(
        x_d,
        w_d,
        bias_d,
        tgt_d,
        loss_d,
        lse_d,
        B as i64,
        S as i64,
        V as i64,
        H as i64,
        i64::from(case.has_bias),
    );
    assert_eq!(rc, 0, "forward_gemm rc={rc}");
    let loss_gpu = download_f32(loss_d, rows);
    let lse_gpu = download_f32(lse_d, rows);

    let mut max_loss_err = 0f64;
    let mut max_lse_err = 0f64;
    for r in 0..rows {
        if targets[r] == IGNORE_INDEX {
            assert_eq!(
                loss_gpu[r], 0.0,
                "ignored row {r} must have exactly zero loss (has_bias={})",
                case.has_bias
            );
            continue;
        }
        let le = (loss_gpu[r] as f64 - fwd_ref.loss_per_row[r]).abs()
            / fwd_ref.loss_per_row[r].abs().max(1.0);
        let se = (lse_gpu[r] as f64 - fwd_ref.lse_per_row[r]).abs()
            / fwd_ref.lse_per_row[r].abs().max(1.0);
        max_loss_err = max_loss_err.max(le);
        max_lse_err = max_lse_err.max(se);
    }
    assert!(
        max_loss_err < 1e-3,
        "forward loss diverged: max rel err {max_loss_err:e} (has_bias={})",
        case.has_bias
    );
    assert!(
        max_lse_err < 1e-3,
        "forward lse diverged: max rel err {max_lse_err:e} (has_bias={})",
        case.has_bias
    );

    // ── Backward ──
    let dx_d = alloc_zeroed_f32(rows * H);
    let dw_d = alloc_zeroed_f32(V * H);
    let dbias_d = if case.has_bias { alloc_zeroed_f32(V) } else { 0 };
    let rc = nsl_fused_linear_ce_backward_gemm(
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
        num_valid as i64,
        i64::from(case.has_bias),
    );
    assert_eq!(rc, 0, "backward_gemm rc={rc}");
    let dx_gpu = download_f32(dx_d, rows * H);
    let dw_gpu = download_f32(dw_d, V * H);

    let check = |name: &str, got: &[f32], want: &[f64], bound: f64| {
        let ref_mag = want.iter().fold(0f64, |m, &v| m.max(v.abs()));
        assert!(
            ref_mag >= bound * 10.0,
            "{name}: reference magnitude {ref_mag:e} is not >= 10x the bound {bound:e} — \
             the tolerance would not discriminate (item-6 lesson)"
        );
        let max_err = got
            .iter()
            .zip(want)
            .fold(0f64, |m, (&g, &r)| m.max((g as f64 - r).abs()));
        assert!(
            max_err < bound,
            "{name} diverged: max abs err {max_err:e} (bound {bound:e}, ref magnitude \
             {ref_mag:e}, has_bias={})",
            case.has_bias
        );
    };
    check("dx", &dx_gpu, &bwd_ref.dx, 1e-5);
    check("dW", &dw_gpu, &bwd_ref.dw, 1e-5);
    if case.has_bias {
        let dbias_gpu = download_f32(dbias_d, V);
        check("dbias", &dbias_gpu, &bwd_ref.dbias, 1e-5);
    }

    // Ignored rows: dx must be exactly zero (never touched — dlogits is
    // zeroed for invalid rows and beta=1 accumulates onto a zeroed buffer).
    for r in (0..rows).filter(|r| targets[*r] == IGNORE_INDEX) {
        for j in 0..H {
            assert_eq!(
                dx_gpu[r * H + j],
                0.0,
                "ignored row {r} col {j} has nonzero dx (has_bias={})",
                case.has_bias
            );
        }
    }

    for d in [x_d, w_d, tgt_d, loss_d, lse_d, dx_d, dw_d] {
        unsafe { nsl_test_cuda_free(d) };
    }
    if case.has_bias {
        unsafe {
            nsl_test_cuda_free(bias_d);
            nsl_test_cuda_free(dbias_d);
        }
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gemm_path_matches_cpu_f64_with_bias() {
    if !cuda_available() {
        return;
    }
    pin_full_f32();
    run_case(&Case { has_bias: true });
}

#[test]
#[ignore = "requires CUDA GPU"]
fn gemm_path_matches_cpu_f64_biasless() {
    if !cuda_available() {
        return;
    }
    pin_full_f32();
    run_case(&Case { has_bias: false });
}
