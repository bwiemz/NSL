//! Silicon gate for the fused RMSNorm gamma backward
//! (`nsl_rmsnorm_dgamma_backward` on a device tensor): the three KIR
//! launches of `nsl_kir::kernels::rmsnorm_dgamma`, a block per row for
//! `1 / rms`, per-chunk column sums, and a final pass over the chunks.
//!
//! The interpreter gate (`nsl-codegen/tests/rmsnorm_dgamma_kir_equivalence.rs`)
//! holds the kernels' PTX to its documented order; this one runs them on the
//! card, where ptxas schedules, unrolls and may contract what the PTX spells,
//! and where warps race. Every gate here:
//!
//! 1. compares with an f64 reference computed in this file from the inputs,
//!    `dgamma[j] = Σ_i dy[i, j] · x[i, j] / sqrt(Σ_k x[i, k]² / cols + eps)`,
//!    within the bound the kernel module derives for its reductions
//!    (`((1 + ε_rinv)(1 + γ_D) - 1) · Σ_i |dy · x · rinv|`, restated below);
//! 2. uses shapes with tails against every block and chunk (a single
//!    element, widths under a warp and past the 256-thread block, ragged
//!    chunks) and the three recipes' shapes (rows = batch × sequence);
//! 3. feeds gradients at realistic magnitudes, 1e-9 to 1e-6 of either sign;
//! 4. runs each shape twice and requires the same bits (deterministic);
//! 5. arms the launch census and asserts all three kernels launched, so a
//!    fallback path cannot pass the numeric check vacuously.
//!
//!   cargo test -p nsl-runtime --features cuda,test-hooks \
//!       --test rmsnorm_dgamma_backward_gpu -- --include-ignored --test-threads=1
#![cfg(all(feature = "cuda", feature = "test-hooks"))]

use std::sync::Mutex;

use nsl_kir::kernels::rmsnorm_dgamma::{dgamma_plan, RmsNormDgammaOp, RINV_BLOCK, WARPS};
use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_rmsnorm_dgamma_backward, nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_from_static,
    nsl_tensor_get_dtype, nsl_tensor_to_device,
};
use nsl_runtime::{test_kernel_launch_census_arm, test_kernel_launch_count};

const DTYPE_F32: i64 = 1;
const EPS: f64 = 1e-5;
/// f32 unit roundoff.
const U: f64 = 1.0 / (1u64 << 24) as f64;

/// The census is process-global; one lock keeps this file's gates apart.
static SERIAL: Mutex<()> = Mutex::new(());

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_runtime::nsl_cuda_init() == 0
}

/// Upload `vals` as an f32 device tensor of `shape`.
fn upload(shape: &[usize], vals: &[f32]) -> i64 {
    assert_eq!(shape.iter().product::<usize>(), vals.len(), "shape/data mismatch");
    let leaked: &'static [f32] = Box::leak(vals.to_vec().into_boxed_slice());
    let s = nsl_list_new();
    for &d in shape {
        nsl_list_push(s, d as i64);
    }
    let cpu = nsl_tensor_from_static(leaked.as_ptr() as i64, s, DTYPE_F32);
    nsl_list_free(s);
    let dev = nsl_tensor_to_device(cpu, 1);
    nsl_tensor_free(cpu);
    dev
}

/// Download `n` f32 elements.
fn download(t: i64, n: usize) -> Vec<f32> {
    let cpu = nsl_tensor_to_device(t, 0);
    assert_eq!(nsl_tensor_get_dtype(cpu), DTYPE_F32, "dgamma comes back f32");
    let p = nsl_tensor_data_ptr(cpu);
    assert_ne!(p, 0, "downloaded tensor has no data pointer");
    let out = unsafe { std::slice::from_raw_parts(p as *const f32, n) }.to_vec();
    nsl_tensor_free(cpu);
    out
}

/// Deterministic, non-repeating draws in [0, 1): a Weyl sequence.
fn weyl(i: usize, salt: f64) -> f64 {
    ((i as f64) * 0.618_033_988_749_894_9 + salt).fract()
}

/// Activations: a Weyl draw in [-3, 3) scaled per row by 2^(row % 7 - 3),
/// so `rinv` differs from row to row; a zero every 29 elements.
fn activations(rows: usize, cols: usize) -> Vec<f32> {
    (0..rows * cols)
        .map(|k| {
            if k % 29 == 11 {
                return 0.0;
            }
            let scale = 2f64.powi((k / cols % 7) as i32 - 3);
            ((weyl(k, 0.1) * 6.0 - 3.0) * scale) as f32
        })
        .collect()
}

/// Gradients: log-uniform magnitude in [1e-9, 1e-6], either sign.
fn gradients(n: usize) -> Vec<f32> {
    (0..n)
        .map(|k| {
            let mag = 10f64.powf(-9.0 + 3.0 * weyl(k, 0.37));
            let sign = if weyl(k, 0.71) < 0.5 { -1.0 } else { 1.0 };
            (sign * mag) as f32
        })
        .collect()
}

fn gamma(n: u64) -> f64 {
    let nu = n as f64 * U;
    nu / (1.0 - nu)
}

/// The kernel module's bounds, restated from its reductions: `rinv` carries
/// `(1 + u) / ((1 - u) sqrt(1 - γ_{m+10})) - 1`, `m = ceil(cols / 256)`; a
/// term of `dgamma` goes through `D = 2 + chunk_rows + ceil(chunks / 8) + 3`
/// roundings.
fn scale_bound(rows: u64, cols: u64) -> f64 {
    let m = cols.div_ceil(u64::from(RINV_BLOCK));
    let e_rinv = (1.0 + U) / ((1.0 - U) * (1.0 - gamma(m + 10)).sqrt()) - 1.0;
    let plan = dgamma_plan(rows, cols);
    let depth = 2 + plan.chunk_rows + plan.chunks.div_ceil(u64::from(WARPS)) + 3;
    assert_eq!(depth, plan.depth(), "the module's depth is the derivation's");
    // The f64 reference's own sums: n · 2^-53 relative to the magnitudes.
    let slack = (rows + cols + 4) as f64 * f64::EPSILON;
    (1.0 + e_rinv) * (1.0 + gamma(depth)) - 1.0 + slack
}

struct Outcome {
    worst_ratio: f64,
    worst_rel: f64,
}

/// Run `[rows, cols]` twice, check determinism, the census and the bound.
fn check(rows: usize, cols: usize) -> Outcome {
    let x = activations(rows, cols);
    let dy = gradients(rows * cols);
    let (xd, dyd, gd) = (upload(&[rows, cols], &x), upload(&[rows, cols], &dy), upload(&[cols], &vec![1.0; cols]));

    test_kernel_launch_census_arm();
    let first = nsl_rmsnorm_dgamma_backward(dyd, xd, gd, EPS);
    let second = nsl_rmsnorm_dgamma_backward(dyd, xd, gd, EPS);
    for op in RmsNormDgammaOp::ALL {
        assert_eq!(test_kernel_launch_count(op.kernel_name()), 2, "{rows}x{cols}: {} launches", op.kernel_name());
    }
    let (a, b) = (download(first, cols), download(second, cols));
    let a_bits: Vec<u32> = a.iter().map(|v| v.to_bits()).collect();
    let b_bits: Vec<u32> = b.iter().map(|v| v.to_bits()).collect();
    assert_eq!(a_bits, b_bits, "{rows}x{cols}: two runs differ");

    let rinv: Vec<f64> = (0..rows)
        .map(|i| {
            let s: f64 = (0..cols).map(|k| f64::from(x[i * cols + k]).powi(2)).sum();
            1.0 / (s / cols as f64 + f64::from(EPS as f32)).sqrt()
        })
        .collect();
    let scale = scale_bound(rows as u64, cols as u64);
    let mut out = Outcome { worst_ratio: 0.0, worst_rel: 0.0 };
    for j in 0..cols {
        let t = |i: usize| f64::from(dy[i * cols + j]) * f64::from(x[i * cols + j]) * rinv[i];
        let want: f64 = (0..rows).map(t).sum();
        let mags: f64 = (0..rows).map(|i| t(i).abs()).sum();
        let err = (f64::from(a[j]) - want).abs();
        let bound = scale * mags;
        assert!(err <= bound, "{rows}x{cols}: dgamma[{j}] = {:e}, f64 {want:e}, |error| {err:e} > bound {bound:e}", a[j]);
        if bound > 0.0 {
            out.worst_ratio = out.worst_ratio.max(err / bound);
        }
        if mags > 0.0 {
            out.worst_rel = out.worst_rel.max(err / mags);
        }
    }
    for t in [first, second, xd, dyd, gd] {
        nsl_tensor_free(t);
    }
    out
}

/// Tails against the 256-thread blocks, the 32-column final blocks and the
/// chunking: a single element, one column, widths under a warp and one past
/// the block, ragged last chunks, rows not a multiple of anything.
#[test]
#[ignore = "requires CUDA GPU"]
fn dgamma_matches_an_f64_reference_within_the_derived_bound_on_tails() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    if !cuda_available() {
        eprintln!("skipping: no usable CUDA GPU");
        return;
    }
    for (rows, cols) in [(1, 1), (3, 7), (37, 1), (5, 257), (130, 260), (1000, 33), (777, 1031), (9, 2049)] {
        let o = check(rows, cols);
        eprintln!("[rmsnorm-dgamma] {rows}x{cols}: worst |error| / bound {:.3}, worst |error| / Σ|t| {:.2e}", o.worst_ratio, o.worst_rel);
    }
}

/// The recipes' shapes: Coder-50M (1 × 1024 rows of 512), Coder-500M
/// (2 × 1024 of 1280), Coder-1B (2 × 2048 of 2048).
#[test]
#[ignore = "requires CUDA GPU"]
fn dgamma_matches_an_f64_reference_within_the_derived_bound_at_the_recipe_shapes() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    if !cuda_available() {
        eprintln!("skipping: no usable CUDA GPU");
        return;
    }
    for (rows, cols) in [(1024, 512), (2048, 1280), (4096, 2048)] {
        let o = check(rows, cols);
        eprintln!("[rmsnorm-dgamma] {rows}x{cols}: worst |error| / bound {:.3}, worst |error| / Σ|t| {:.2e}", o.worst_ratio, o.worst_rel);
    }
}
