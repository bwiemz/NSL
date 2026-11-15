#![cfg(all(feature = "cuda", feature = "test-hooks"))]
//! TF32 GEMMs through cublasLt with an enumerated, timed plan per shape must
//! change WHICH kernel computes a product and nothing about WHAT it computes.
//!
//! Why the path exists: cuBLAS 13.8's `cublasSgemm` heuristic gives sm_120 a
//! 64x64-tile kernel for most transformer projections and their gradients
//! (48-57 TFLOPS at Coder-1B's shapes on random data) where 128x128 / 128x256 /
//! 256x128 kernels of the same library run 67-76 (`lt_matmul.rs`, TF32
//! section; `models/benchmarks/NSL_VS_PYTORCH_2026_11.md`).
//!
//! The anchor is an f64 reference on the host from the f32 operands, sampled at
//! deterministic positions, with a tolerance scaled to each dot product's L1
//! mass: 1.5e-3 x L1. TF32 rounds each operand to 11 significant bits, so a
//! product carries a relative error below ~2^-10 (~1e-3) and the sum's error is
//! below that times L1; a crossed layout or a swapped operand lands at O(1) of
//! L1. Shapes are non-square so a crossed layout cannot read a real, in-bounds
//! sub-matrix of the right size.
//!
//! Checks, in a child process with the default configuration (TF32 math mode,
//! timed plan selection on): the plain forward, a transposed-view second
//! operand (the dgrad form), and the weight-gradient funnel with beta = 1
//! accumulate (where a timing launch that wrote the caller's C instead of the
//! scratch would corrupt live gradients) all issue through cublasLt with no
//! fallback and match the reference; a repeat is bit-identical (cached plan);
//! under `--deterministic` a new shape does NOT take the timed path. A second
//! child with `NSL_MATMUL_BF16_LT_TUNE=0` (an unconfigured process reads the
//! environment; compiled programs get it from `--no-bf16-lt-tune`) never
//! issues through cublasLt and gives the same answers.

use nsl_runtime::tensor::{
    nsl_tensor_free, nsl_tensor_matmul, nsl_tensor_to_device, nsl_tensor_wgrad_accum,
    test_build_tensor_2d_f32, test_read_tensor_f32,
};
use nsl_runtime::{test_lt_matmul_reset, test_lt_matmul_tf32_stats};

const PROBE: &str = "NSL_TF32_LT_PROBE"; // "on" | "off"

/// Forward: [M x K] @ [K x N], all distinct; 2^30.6 multiply-adds, above the
/// path's 2^28 floor.
const M: usize = 2048;
const K: usize = 768;
const N: usize = 1024;

/// wgrad: m[D x O] += s * x[R x D]^T @ g[R x O].
const R: usize = 1536;
const D: usize = 1024;
const O: usize = 640;

fn deterministic(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn gpu_2d(rows: usize, cols: usize, data: &[f32]) -> i64 {
    let t = test_build_tensor_2d_f32(rows, cols, data);
    let g = nsl_tensor_to_device(t, 1);
    if g != t {
        nsl_tensor_free(t);
    }
    g
}

fn read_gpu(t: i64) -> Vec<f64> {
    let cpu = nsl_tensor_to_device(t, 0);
    let out: Vec<f64> = test_read_tensor_f32(cpu).into_iter().map(f64::from).collect();
    if cpu != t {
        nsl_tensor_free(cpu);
    }
    out
}

/// Which way an operand is stored.
#[derive(Clone, Copy)]
enum Store {
    /// `[rows x cols]` row-major, read as is.
    Plain,
    /// `[cols x rows]` row-major, read as its transpose.
    Transposed,
}

/// `got[i*n + j] == scale * sum_k a[i,k] * b[k,j] (+ base[i*n + j])` at
/// `samples` deterministic positions, within `1.5e-3 * |scale| * L1 + 1e-6`.
/// `a` is logically [rows x k], `b` logically [k x n].
#[allow(clippy::too_many_arguments)]
fn assert_sampled_product(
    label: &str,
    got: &[f64],
    a: &[f32],
    a_store: Store,
    b: &[f32],
    b_store: Store,
    rows: usize,
    n: usize,
    k: usize,
    add_into: Option<&[f32]>,
    scale: f64,
) {
    assert_eq!(got.len(), rows * n, "{label}: output element count");
    let at = |i: usize, kk: usize| match a_store {
        Store::Plain => a[i * k + kk],
        Store::Transposed => a[kk * rows + i],
    };
    let bt = |kk: usize, j: usize| match b_store {
        Store::Plain => b[kk * n + j],
        Store::Transposed => b[j * k + kk],
    };
    let mut s: u64 = 0x7F32_1A7E ^ ((rows as u64) << 32) ^ (n as u64);
    for _ in 0..2048 {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let i = (s >> 33) as usize % rows;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (s >> 33) as usize % n;
        let (mut acc, mut l1) = (0.0f64, 0.0f64);
        for kk in 0..k {
            let p = at(i, kk) as f64 * bt(kk, j) as f64;
            acc += p;
            l1 += p.abs();
        }
        let want = scale * acc + add_into.map_or(0.0, |base| base[i * n + j] as f64);
        let have = got[i * n + j];
        let tol = 1.5e-3 * scale.abs() * l1 + 1e-6;
        assert!(
            (have - want).abs() <= tol,
            "{label}: C[{i},{j}] = {have}, reference {want} (tol {tol}) — a wrong \
             kernel or layout, not TF32 rounding, at this magnitude"
        );
    }
}

/// The three products every child checks; returns their device results.
fn products() -> Vec<Vec<f64>> {
    let a_data = deterministic(M * K, 431);
    let w_data = deterministic(K * N, 433);
    let wt_data = deterministic(N * K, 437); // [N x K], used as its transpose
    let a = gpu_2d(M, K, &a_data);
    let w = gpu_2d(K, N, &w_data);
    let wt = gpu_2d(N, K, &wt_data);
    let mut out = Vec::new();

    let c = nsl_tensor_matmul(a, w, 0);
    let c_host = read_gpu(c);
    assert_sampled_product("forward", &c_host, &a_data, Store::Plain, &w_data, Store::Plain, M, N, K, None, 1.0);
    out.push(c_host);
    nsl_tensor_free(c);

    // a @ wt^T through a transposed VIEW (no copy): `sgemm_row_major_t` with
    // the second operand transposed — the dgrad form.
    let wt_view = nsl_runtime::tensor::nsl_tensor_transpose(wt, 0, 1);
    let c = nsl_tensor_matmul(a, wt_view, 0);
    let c_host = read_gpu(c);
    assert_sampled_product(
        "transposed-view", &c_host, &a_data, Store::Plain, &wt_data, Store::Transposed, M, N, K, None, 1.0,
    );
    out.push(c_host);
    nsl_tensor_free(c);

    let m0_data = deterministic(D * O, 439);
    let x_data = deterministic(R * D, 443);
    let g_data = deterministic(R * O, 449);
    let m = gpu_2d(D, O, &m0_data);
    let x = gpu_2d(R, D, &x_data);
    let g = gpu_2d(R, O, &g_data);
    nsl_tensor_wgrad_accum(m, x, g, 0.5);
    let m_host = read_gpu(m);
    // x is [R x D] and the product wants x^T: logically [D x R], stored transposed.
    assert_sampled_product(
        "wgrad", &m_host, &x_data, Store::Transposed, &g_data, Store::Plain, D, O, R, Some(&m0_data), 0.5,
    );
    out.push(m_host);
    for t in [m, x, g, wt_view, wt, w, a] {
        nsl_tensor_free(t);
    }
    out
}

#[test]
#[ignore = "spawned by tf32_lt_gates_gpu"]
fn zz_tf32_lt_on_child() {
    if std::env::var(PROBE).unwrap_or_default() != "on" {
        return;
    }
    test_lt_matmul_reset();
    let first = products();
    let (issued, tuned, fallbacks) = test_lt_matmul_tf32_stats();
    assert_eq!(fallbacks, 0, "a TF32 GEMM declined the cublasLt path (see stderr)");
    assert!(
        issued >= 3,
        "TF32 mode with timed selection on, yet only {issued} GEMMs issued through \
         cublasLt — the wiring is dead and an A/B would measure sgemm against sgemm"
    );
    assert_eq!(tuned, 3, "three distinct shapes, each timed once");

    // A cached plan replays the identical kernel: same bits.
    let second = products();
    assert_eq!(first, second, "same operands, same cached plans, different bits");
    assert_eq!(test_lt_matmul_tf32_stats().1, 3, "no shape was re-timed");

    // --deterministic: a timed winner depends on machine state, so a NEW shape
    // must not take the timed path (cublasSgemm's heuristic is reproducible).
    nsl_runtime::deterministic_ops::nsl_set_deterministic(1);
    let issued_before = test_lt_matmul_tf32_stats().0;
    let a_data = deterministic(1536 * 640, 457);
    let w_data = deterministic(640 * 2048, 461);
    let (a, w) = (gpu_2d(1536, 640, &a_data), gpu_2d(640, 2048, &w_data));
    let c = nsl_tensor_matmul(a, w, 0);
    let c_host = read_gpu(c);
    assert_sampled_product("deterministic", &c_host, &a_data, Store::Plain, &w_data, Store::Plain, 1536, 2048, 640, None, 1.0);
    assert_eq!(
        test_lt_matmul_tf32_stats().0,
        issued_before,
        "under --deterministic a GEMM issued through the timed cublasLt path"
    );
    for t in [c, w, a] {
        nsl_tensor_free(t);
    }
    println!("TF32-LT-ON-OK");
}

#[test]
#[ignore = "spawned by tf32_lt_gates_gpu"]
fn zz_tf32_lt_off_child() {
    if std::env::var(PROBE).unwrap_or_default() != "off" {
        return;
    }
    test_lt_matmul_reset();
    let _ = products();
    assert_eq!(
        test_lt_matmul_tf32_stats(),
        (0, 0, 0),
        "NSL_MATMUL_BF16_LT_TUNE=0, yet the timed TF32 cublasLt path moved"
    );
    println!("TF32-LT-OFF-OK");
}

fn spawn_child(mode: &str, child_test: &str, expect: &str) {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = std::process::Command::new(exe);
    cmd.args(["--test-threads=1", "--nocapture", "--include-ignored", child_test])
        .env(PROBE, mode)
        // Every variable the mode and Lt resolvers consult: the children must
        // run the DEFAULT configuration (TF32, timed selection on) apart from
        // the one switch the off child flips.
        .env_remove("NSL_MATMUL_BF16")
        .env_remove("NSL_MATMUL_BF16_LT")
        .env_remove("NSL_MATMUL_BF16_LT_TUNE")
        .env_remove("NSL_MATMUL_BF16_LT_WORKSPACE_MIB")
        .env_remove("NSL_MATMUL_PEDANTIC")
        .env_remove("NSL_MATMUL_TF32")
        // TF32's default hands a transposed view to cuBLAS as OP_T, which is
        // the form the transposed-view check exists to cover.
        .env_remove("NSL_MATMUL_TRANSPOSE_VIEWS")
        .env_remove("NSL_ASYNC_ALLOC");
    if mode == "off" {
        cmd.env("NSL_MATMUL_BF16_LT_TUNE", "0");
    }
    let out = cmd.output().expect("spawn tf32 lt child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.contains(expect),
        "tf32 lt {mode}-child failed (status {:?})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        out.status
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn tf32_lt_gates_gpu() {
    if std::env::var(PROBE).is_ok() {
        return;
    }
    spawn_child("on", "zz_tf32_lt_on_child", "TF32-LT-ON-OK");
    spawn_child("off", "zz_tf32_lt_off_child", "TF32-LT-OFF-OK");
}
