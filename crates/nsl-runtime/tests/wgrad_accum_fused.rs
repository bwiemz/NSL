//! Item 7 (`--fuse-wgrad-accum`): `nsl_tensor_wgrad_accum` vs the decomposed chain.
//!
//! The fused form collapses
//!
//! ```text
//!   x_t      = transpose(x)              [B, d, T]
//!   raw_grad = matmul(x_t, g)            [B, d, o]   <- B x P temporary
//!   dW       = reduce_to_shape(raw, W)   [d, o]
//!   m       += scale * dW
//! ```
//!
//! into ONE cuBLAS call with `alpha = scale, beta = 1.0` writing straight into
//! `m_partial`. The batch sum is just part of the contraction, so flattening
//! `[B, T, *]` to `[B*T, *]` is mathematically exact — but the FLOATING-POINT
//! result is deliberately NOT bit-identical (cuBLAS's epilogue uses FMA, and
//! the per-batch partial products are no longer rounded to f32 before being
//! summed). This file pins the agreement to a tolerance and, just as
//! importantly, pins that the comparison is SENSITIVE enough to catch a real
//! error.
//!
//! Run:
//!   cargo test -p nsl-runtime --features cuda --test wgrad_accum_fused -- --include-ignored

#![cfg(feature = "cuda")]

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_tensor_free, nsl_tensor_from_static, nsl_tensor_get, nsl_tensor_to_device,
};
use nsl_runtime::tensor::arithmetic::{

    nsl_tensor_matmul, nsl_tensor_scalar_mul_add_inplace, nsl_tensor_wgrad_accum,
};

const DTYPE_F32: i64 = 1;

/// Pin cuBLAS to full f32 for this process before any GPU work.
///
/// The matmul default is TF32 (see `cuda::resolve_math_mode`), which drifts
/// ~1e-3 relative. The gates in this file compare against an f64 CPU
/// reference at ~1e-5 in order to catch **dispatch** bugs — stride-blindness,
/// a crossed operand mapping, a fused GEMM writing the wrong accumulation.
/// Those err by order 1e0, so full f32 keeps four orders of magnitude of
/// headroom; widening to accommodate TF32 would spend that for nothing.
///
/// Sound because `resolve_math_mode` is read exactly once per process, inside
/// `cublas_handle()`'s `OnceLock`, and every GPU test in this file calls this
/// first — so whichever runs first performs the init with the variable
/// already set. Self-checking: if that stops holding, the test fails with a
/// ~1e-4 drift message rather than passing quietly.
///
/// NOT covered here: whether these paths are correct *under* TF32. The math
/// mode is process-global, so that needs a child process (the pattern
/// `matmul_tf32_mode.rs` uses). The risk is low — the mode applies identically
/// to both arms of every comparison in this file — but it is a real gap.
fn pin_full_f32() {
    std::env::set_var("NSL_MATMUL_TF32", "0");
    std::env::set_var("NSL_MATMUL_BF16", "0");
}

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_runtime::nsl_cuda_init() == 0
}

/// Deterministic, well-conditioned values in [-1, 1).
fn det(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            ((s >> 8) as f32 / (1u32 << 24) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn tensor(vals: &[f32], shape_dims: &[i64]) -> i64 {
    let leaked: &'static [f32] = Box::leak(vals.to_vec().into_boxed_slice());
    let shape = nsl_list_new();
    for d in shape_dims {
        nsl_list_push(shape, *d);
    }
    let t = nsl_tensor_from_static(leaked.as_ptr() as i64, shape, DTYPE_F32);
    nsl_list_free(shape);
    t
}

fn to_gpu(t: i64) -> i64 {
    nsl_tensor_to_device(t, 1)
}

/// Read a rank-2 tensor into a flat Vec<f64>.
///
/// Element reads are host-side, so a device tensor is staged to CPU first —
/// `nsl_tensor_get` aborts on a GPU tensor rather than returning garbage.
fn read2(t: i64, rows: i64, cols: i64) -> Vec<f64> {
    let host = nsl_tensor_to_device(t, 0);
    let mut out = Vec::with_capacity((rows * cols) as usize);
    for r in 0..rows {
        for c in 0..cols {
            let idx = nsl_list_new();
            nsl_list_push(idx, r);
            nsl_list_push(idx, c);
            out.push(nsl_tensor_get(host, idx));
            nsl_list_free(idx);
        }
    }
    if host != t {
        nsl_tensor_free(host);
    }
    out
}

/// CPU ground truth in f64: `m0 + scale * sum_{b,t} x[b,t,i] * g[b,t,j]`.
fn cpu_reference(
    x: &[f32], g: &[f32], m0: &[f32], b: usize, tt: usize, d: usize, o: usize, scale: f64,
) -> Vec<f64> {
    let mut out: Vec<f64> = m0.iter().map(|&v| v as f64).collect();
    for i in 0..d {
        for j in 0..o {
            let mut acc = 0.0f64;
            for bb in 0..b {
                for t in 0..tt {
                    acc += x[(bb * tt + t) * d + i] as f64 * g[(bb * tt + t) * o + j] as f64;
                }
            }
            out[i * o + j] += scale * acc;
        }
    }
    out
}

fn max_abs(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
}

/// Shared driver: returns (fused, unfused, cpu_ref) each as [d*o] f64.
fn run_case(b: usize, tt: usize, d: usize, o: usize, scale: f64)
    -> (Vec<f64>, Vec<f64>, Vec<f64>)
{
    let x_h = det(0x51, b * tt * d);
    let g_h = det(0x52, b * tt * o);
    let m_h = det(0x53, d * o);

    let cpu = cpu_reference(&x_h, &g_h, &m_h, b, tt, d, o, scale);

    let bi = b as i64;
    let ti = tt as i64;
    let di = d as i64;
    let oi = o as i64;

    // ── Fused ────────────────────────────────────────────────────────────
    let xf = to_gpu(tensor(&x_h, &[bi, ti, di]));
    let gf = to_gpu(tensor(&g_h, &[bi, ti, oi]));
    let mf = to_gpu(tensor(&m_h, &[di, oi]));
    nsl_tensor_wgrad_accum(mf, xf, gf, scale);
    let fused = read2(mf, di, oi);
    nsl_tensor_free(mf);
    nsl_tensor_free(gf);
    nsl_tensor_free(xf);

    // ── Unfused: exactly the chain codegen emits today ────────────────────
    let xu = to_gpu(tensor(&x_h, &[bi, ti, di]));
    let gu = to_gpu(tensor(&g_h, &[bi, ti, oi]));
    let mu = to_gpu(tensor(&m_h, &[di, oi]));
    let x_t = nsl_runtime::tensor::nsl_tensor_transpose(xu, -2, -1);
    let raw = nsl_tensor_matmul(x_t, gu, 0);
    let dw = nsl_runtime::tensor::nsl_tensor_reduce_to_shape(raw, mu);
    nsl_tensor_scalar_mul_add_inplace(mu, dw, scale);
    let unfused = read2(mu, di, oi);
    nsl_tensor_free(dw);
    nsl_tensor_free(raw);
    nsl_tensor_free(x_t);
    nsl_tensor_free(mu);
    nsl_tensor_free(gu);
    nsl_tensor_free(xu);

    (fused, unfused, cpu)
}

/// The fused GEMM must agree with the decomposed chain to an f32 tolerance,
/// and BOTH must agree with an f64 CPU reference.
///
/// Comparing against the f64 reference as well as against the chain is what
/// makes this meaningful: two GPU paths agreeing with each other but both
/// drifting from the true sum would otherwise pass.
#[test]
#[ignore = "requires CUDA GPU"]
fn fused_wgrad_matches_decomposed_chain_and_cpu() {
    pin_full_f32();
    if !cuda_available() {
        eprintln!("skipped: no CUDA device");
        return;
    }
    for &(b, t, d, o) in &[(1usize, 8usize, 16usize, 16usize), (4, 16, 32, 48), (2, 64, 64, 32)] {
        let scale = 0.125f64;
        let (fused, unfused, cpu) = run_case(b, t, d, o, scale);

        let f_vs_u = max_abs(&fused, &unfused);
        let f_vs_c = max_abs(&fused, &cpu);
        let u_vs_c = max_abs(&unfused, &cpu);
        let mag = cpu.iter().fold(0.0f64, |a, v| a.max(v.abs()));
        eprintln!(
            "[wgrad B={b} T={t} d={d} o={o}] fused_vs_unfused={f_vs_u:.3e} \
             fused_vs_cpu={f_vs_c:.3e} unfused_vs_cpu={u_vs_c:.3e} max|ref|={mag:.3e}"
        );

        // f32 accumulation over N = B*T terms; the bound scales with the
        // magnitude of the result and sqrt(N) growth of the summation error.
        let n = (b * t) as f64;
        let tol = 1e-5 * mag.max(1.0) * n.sqrt();
        assert!(
            f_vs_u <= tol,
            "fused vs decomposed exceeded tolerance: {f_vs_u:.3e} > {tol:.3e}"
        );
        assert!(
            f_vs_c <= tol,
            "fused vs f64 CPU reference exceeded tolerance: {f_vs_c:.3e} > {tol:.3e}"
        );
        assert!(
            u_vs_c <= tol,
            "decomposed vs f64 CPU reference exceeded tolerance: {u_vs_c:.3e} > {tol:.3e} \
             — the BASELINE is off, so the fused comparison above proves nothing"
        );
    }
}

/// Negative control: the comparison must be able to FAIL.
///
/// Perturb one input element and confirm the fused result moves well outside
/// the tolerance used above. Without this, a fused path that silently wrote
/// nothing (or wrote a stale buffer) would pass the agreement test whenever
/// the accumulator happened to dominate.
#[test]
#[ignore = "requires CUDA GPU"]
fn fused_wgrad_comparison_is_sensitive() {
    pin_full_f32();
    if !cuda_available() {
        eprintln!("skipped: no CUDA device");
        return;
    }
    let (b, t, d, o) = (4usize, 16usize, 32usize, 48usize);
    let scale = 0.125f64;

    let x_h = det(0x51, b * t * d);
    let mut x_pert = x_h.clone();
    x_pert[0] += 1.0; // one element, one unit
    let g_h = det(0x52, b * t * o);
    let m_h = det(0x53, d * o);

    let ref_unperturbed = cpu_reference(&x_h, &g_h, &m_h, b, t, d, o, scale);

    let xf = to_gpu(tensor(&x_pert, &[b as i64, t as i64, d as i64]));
    let gf = to_gpu(tensor(&g_h, &[b as i64, t as i64, o as i64]));
    let mf = to_gpu(tensor(&m_h, &[d as i64, o as i64]));
    nsl_tensor_wgrad_accum(mf, xf, gf, scale);
    let perturbed = read2(mf, d as i64, o as i64);
    nsl_tensor_free(mf);
    nsl_tensor_free(gf);
    nsl_tensor_free(xf);

    let diff = max_abs(&perturbed, &ref_unperturbed);
    let mag = ref_unperturbed.iter().fold(0.0f64, |a, v| a.max(v.abs()));
    let tol = 1e-5 * mag.max(1.0) * ((b * t) as f64).sqrt();
    eprintln!("[wgrad-control] perturbed_vs_unperturbed={diff:.3e} tol={tol:.3e}");
    assert!(
        diff > tol * 10.0,
        "perturbing an input element moved the fused result by only {diff:.3e} \
         (tolerance {tol:.3e}) — the agreement test above is not sensitive to the \
         inputs and proves nothing"
    );
}

/// The CPU / non-contiguous / shape-mismatch fallback must still be correct.
///
/// `nsl_tensor_wgrad_accum` falls back to the decomposed chain whenever the
/// single-GEMM preconditions do not hold, so a shape the compiler's static
/// pre-pass misjudged degrades to slow-but-correct rather than silently
/// producing a wrong gradient. This exercises that path on CPU tensors.
#[test]
fn fallback_path_is_correct_on_cpu_tensors() {
    pin_full_f32();
    let (b, t, d, o) = (2usize, 5usize, 4usize, 3usize);
    let scale = 0.25f64;
    let x_h = det(0x71, b * t * d);
    let g_h = det(0x72, b * t * o);
    let m_h = det(0x73, d * o);
    let cpu = cpu_reference(&x_h, &g_h, &m_h, b, t, d, o, scale);

    // CPU-resident: device == 0, so `ok` is false and the fallback runs.
    let x = tensor(&x_h, &[b as i64, t as i64, d as i64]);
    let g = tensor(&g_h, &[b as i64, t as i64, o as i64]);
    let m = tensor(&m_h, &[d as i64, o as i64]);
    nsl_tensor_wgrad_accum(m, x, g, scale);
    let got = read2(m, d as i64, o as i64);
    nsl_tensor_free(m);
    nsl_tensor_free(g);
    nsl_tensor_free(x);

    let diff = max_abs(&got, &cpu);
    eprintln!("[wgrad-fallback] max_abs={diff:.3e}");
    assert!(
        diff < 1e-4,
        "CPU fallback disagrees with the f64 reference by {diff:.3e}"
    );
}
