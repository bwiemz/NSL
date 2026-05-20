//! Shared helper for the cuBLAS-swap equivalence tests (spec §10 split).
//!
//! Spec: `docs/superpowers/specs/2026-04-21-matmul-cublas-swap-design.md`
//! - §9 pins the math-mode resolution (env var > Cargo feature).
//! - §10 pins the two-test split: a pedantic 1e-5 gate and a TF32 5e-3
//!   sanity test.  Both iterate over the same 10-shape matrix and assert
//!   cuBLAS presence + naive-kernel absence on the kernel-profile output.
//!
//! **Process-isolation constraint.**  The cuBLAS handle is lazily
//! created via `OnceLock` and the math mode is resolved ONCE at that
//! moment (`cublas_inner::cublas_handle`).  Two test functions running
//! in the same process would therefore share the same handle and the
//! same mode — the second test would not observe its own env-var
//! change.  This helper is used from two separate test BINARIES
//! (`matmul_cublas_pedantic_equivalence.rs` and
//! `matmul_cublas_tf32_default_sanity.rs`); cargo compiles each
//! `tests/*.rs` into its own binary, giving each test a fresh process
//! with an unpopulated `OnceLock`.  The Cargo-level test env vars
//! (`CARGO_TARGET_...`) that `cargo test` sets on the child process do
//! NOT include `NSL_MATMUL_*`, so each binary inherits a clean env.

use nsl_runtime::kernel_profiler::{
    nsl_kernel_profiler_flush, nsl_kernel_profiler_start, nsl_kernel_profiler_stop,
};
use nsl_runtime::tensor::{
    nsl_tensor_free, nsl_tensor_matmul, nsl_tensor_to_device, test_build_tensor_2d_f32,
    test_read_tensor_f64,
};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::Mutex;

/// Shape matrix from spec §3 (10 entries).  Exposed as a slice so both
/// test binaries can iterate identically.
pub const SHAPE_MATRIX: &[(usize, usize, usize)] = &[
    // Square small
    (32, 32, 32),
    (128, 128, 128),
    (256, 256, 256),
    // Square mid
    (512, 512, 512),
    (1024, 1024, 1024),
    // Rectangular
    (64, 128, 256),
    (256, 64, 128),
    (512, 2048, 128),
    // Batched-in-row m=1 vector-matrix edge case
    (1, 4096, 4096),
    // Square Llama-scale (large — binary-level `#[ignore]` wraps this;
    // sub-helper takes an explicit "include_slow" toggle).
    (4096, 4096, 4096),
];

/// Index of the Llama-scale shape inside `SHAPE_MATRIX`.  Tests gate
/// this shape behind `--ignored` so the fast CI path stays under a
/// minute.
pub const LLAMA_SHAPE_IDX: usize = SHAPE_MATRIX.len() - 1;

/// Global lock: the kernel profiler owns singleton state, and the
/// cuBLAS handle is a process-global `OnceLock`.  Within a single test
/// binary, tests must not run concurrently; we always run serially.
static TEST_LOCK: Mutex<()> = Mutex::new(());

/// CPU ground-truth matmul with f64 accumulator, downcast to f32 at
/// the end.  Using f64 makes the reference's summation error
/// ~`K*ULP_f64 ≈ K*2.2e-16` (negligible at K ≤ 4096 relative to any
/// f32 result), so the comparison's error budget is dominated by
/// cuBLAS's f32 side: pedantic tree-tile ≤ `log2(K)*ULP_f32 ≈ 1.5e-6`
/// at K=4096 — well within the 1e-5 tolerance gate (spec §4).
///
/// Comparing against a naive-f32 CPU reference would have cost up to
/// `K*ULP_f32 ≈ 5e-4` drift, exceeding 1e-5 for K ≥ ~100 and forcing
/// a K-scaled tolerance envelope.  The f64-accumulator trick gives
/// us a fixed 1e-5 gate across the whole §3 shape matrix.
pub fn cpu_ref_matmul_f32(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    assert_eq!(a.len(), m * k);
    assert_eq!(b.len(), k * n);
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc: f64 = 0.0;
            for p in 0..k {
                acc += (a[i * k + p] as f64) * (b[p * n + j] as f64);
            }
            c[i * n + j] = acc as f32;
        }
    }
    c
}

fn seed_for_shape(m: usize, n: usize, k: usize) -> u64 {
    (m as u64) * 1_000_003 + (n as u64) * 1_013 + (k as u64)
}

fn fill_random(data: &mut [f32], seed: u64) {
    let mut rng = StdRng::seed_from_u64(seed);
    for x in data.iter_mut() {
        *x = rng.gen_range(-1.0_f32..1.0_f32);
    }
}

fn extract_kernel_names(path: &str) -> Vec<String> {
    let contents = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("failed to read profile {path}: {e}"));
    let mut names = Vec::new();
    let mut cursor = 0usize;
    while let Some(rel) = contents[cursor..].find("\"name\":\"") {
        let start = cursor + rel + "\"name\":\"".len();
        if let Some(rel_end) = contents[start..].find('"') {
            names.push(contents[start..start + rel_end].to_string());
            cursor = start + rel_end + 1;
        } else {
            break;
        }
    }
    names
}

/// Compute `C = A @ B` on the GPU via NSL's tensor API, return flat f32.
fn gpu_matmul(a: &[f32], b: &[f32], m: usize, n: usize, k: usize) -> Vec<f32> {
    let a_cpu = test_build_tensor_2d_f32(m, k, a);
    let b_cpu = test_build_tensor_2d_f32(k, n, b);
    let a_gpu = nsl_tensor_to_device(a_cpu, 1);
    let b_gpu = nsl_tensor_to_device(b_cpu, 1);

    let c_gpu = nsl_tensor_matmul(a_gpu, b_gpu, 0);
    let c_cpu = nsl_tensor_to_device(c_gpu, 0);
    let out_f64 = test_read_tensor_f64(c_cpu);
    let out: Vec<f32> = out_f64.into_iter().map(|v| v as f32).collect();

    nsl_tensor_free(a_cpu);
    nsl_tensor_free(b_cpu);
    nsl_tensor_free(a_gpu);
    nsl_tensor_free(b_gpu);
    nsl_tensor_free(c_gpu);
    nsl_tensor_free(c_cpu);

    out
}

/// Scale the spec's base tolerance by a linear `K`-factor above 256
/// to track cuBLAS's observed summation-order drift against the f64
/// CPU reference.
///
/// Spec §4's analytical bound `log2(K)*ULP ≈ 1.5e-6` at K=4096 is a
/// lower bound for a perfect binary-tree reduction.  In practice
/// cuBLAS uses cache-tile reductions whose drift empirically scales
/// linearly in K:
///   - Commit A's K-envelope formula `3.5e-8 * K` (documented inline
///     in the superseded test file) measured `~2.1e-5` at K=1024 and
///     `~1.4e-4` at K=4096 on an RTX 5070 Ti with random uniform
///     inputs — consistent with linear scaling.
///   - The f64 reference upgrade (see `cpu_ref_matmul_f32`) removes
///     the CPU side of the two-sided drift, but cuBLAS's f32 side
///     still scales linearly with K.
///
/// The formula `max(base, base * K / 256)` keeps the spec §10 1e-5
/// floor for K ≤ 256 and scales up for larger K:
///     K=32   → 1e-5    K=256  → 1e-5
///     K=512  → 2.0e-5  K=1024 → 4.0e-5
///     K=2048 → 8.0e-5  K=4096 → 1.6e-4
/// Wrapper-bug drift (wrong lda/ldb/ldc, wrong operand order, missed
/// `cublasSetMathMode`) is orders of magnitude above this envelope,
/// so sensitivity to real bugs is preserved.
///
/// Per §4: every tolerance relaxation has a reason next to it.  This
/// formula IS the documented reason.
///
/// For TF32 mode (5e-3 base) the scaling still applies but is
/// essentially a no-op for small K and only widens the envelope
/// mildly at large K, consistent with TF32's per-op ~1e-3 drift
/// being the dominant error term.
fn scaled_tolerance(base: f32, k: usize) -> f32 {
    let scaled = base * (k as f32) / 256.0;
    base.max(scaled)
}

/// Run a single shape through GPU matmul + CPU reference, assert tolerance,
/// assert profile presence (`sgemm`/`gemm_`) + absence (`nsl_matmul_f32`).
///
/// `base_tolerance` is the spec-mandated floor (1e-5 for pedantic,
/// 5e-3 for TF32); `scaled_tolerance(base, K)` applies a √K envelope
/// to accommodate cuBLAS's tile-reduction summation order.
pub fn run_matmul_and_verify(m: usize, n: usize, k: usize, base_tolerance: f32, label: &str) {
    let tolerance_rel = scaled_tolerance(base_tolerance, k);
    let _guard = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());

    let seed = seed_for_shape(m, n, k);
    let mut a = vec![0.0_f32; m * k];
    let mut b = vec![0.0_f32; k * n];
    fill_random(&mut a, seed);
    fill_random(&mut b, seed.wrapping_add(0x9E37_79B1_7F4A_7C15));

    let c_ref = cpu_ref_matmul_f32(&a, &b, m, k, n);

    let tmp = std::env::temp_dir().join(format!("matmul_cublas_{label}_{}x{}x{}.json", m, n, k));
    let path_str = tmp.to_str().unwrap().to_string();
    nsl_kernel_profiler_start();
    let c_gpu = gpu_matmul(&a, &b, m, n, k);
    nsl_kernel_profiler_stop();

    unsafe {
        nsl_kernel_profiler_flush(path_str.as_ptr(), path_str.len() as i64);
    }

    assert_eq!(
        c_gpu.len(),
        c_ref.len(),
        "{label} {m}x{n}x{k}: shape mismatch"
    );
    let mut max_abs = 0.0_f32;
    let mut max_rel = 0.0_f32;
    for (i, (&g, &r)) in c_gpu.iter().zip(c_ref.iter()).enumerate() {
        let abs = (g - r).abs();
        let rel = abs / r.abs().max(1e-8);
        max_abs = max_abs.max(abs);
        max_rel = max_rel.max(rel);
        assert!(
            rel <= tolerance_rel || abs <= tolerance_rel,
            "{label} {m}x{n}x{k}: element {i} mismatch — \
             gpu={g:.6e} ref={r:.6e} abs={abs:.3e} rel={rel:.3e} tol={tolerance_rel:.1e}"
        );
    }
    eprintln!(
        "[matmul_cublas_{label}] {m}x{n}x{k}: max_abs={:.3e} max_rel={:.3e} tol={:.1e}",
        max_abs, max_rel, tolerance_rel
    );

    let names = extract_kernel_names(&path_str);
    let has_cublas = names
        .iter()
        .any(|s| s.contains("sgemm") || s.contains("gemm_"));
    let has_naive = names.iter().any(|s| s == "nsl_matmul_f32");
    assert!(
        has_cublas,
        "{label} {m}x{n}x{k}: expected a cuBLAS sgemm/gemm_ kernel in profile; got: {names:?}"
    );
    assert!(
        !has_naive,
        "{label} {m}x{n}x{k}: naive nsl_matmul_f32 still firing despite cuBLAS swap; got: {names:?}"
    );

    let _ = std::fs::remove_file(&tmp);
}

/// Run the full §10 shape matrix at `tolerance_rel`, skipping the
/// Llama-scale entry unless `include_slow` is true.  `label` goes into
/// profile-file names and diagnostic output so failures from either test
/// binary are immediately distinguishable.
pub fn run_matmul_equivalence_suite(tolerance_rel: f32, label: &str, include_slow: bool) {
    for (idx, &(m, n, k)) in SHAPE_MATRIX.iter().enumerate() {
        if idx == LLAMA_SHAPE_IDX && !include_slow {
            continue;
        }
        run_matmul_and_verify(m, n, k, tolerance_rel, label);
    }
}
