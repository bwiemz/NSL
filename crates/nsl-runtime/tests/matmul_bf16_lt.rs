#![cfg(all(feature = "cuda", feature = "test-hooks"))]
//! The cublasLt bf16 issue path (`NSL_MATMUL_BF16_LT=1`) must change WHICH
//! kernel computes the product and nothing about WHAT it computes.
//!
//! cublasLt and GemmEx pick different kernels with different f32 reduction
//! orders, so cross-path BIT-comparison is not available (unlike the cast
//! cache, whose gate compares cached vs fresh bytes of the SAME kernel).
//! The correctness anchor here is an f64 reference computed on the host from
//! the SAME bf16-rounded operand values the device consumes
//! (`half::bf16::from_f32` and the device cast kernel are both
//! round-to-nearest-even), sampled at deterministic output positions, with a
//! tolerance scaled to each dot product's L1 mass — ~40-60x above honest f32
//! accumulation error at these k (sqrt-k RMS growth), and orders of
//! magnitude below what any layout/transpose/operand-swap bug produces
//! (those land at O(1) of the L1 mass).
//!
//! What the Lt path CAN promise bitwise is run-to-run determinism (the
//! preference masks reduction schemes to the deterministic
//! `COMPUTE_TYPE`-only split-k, and a cached plan replays the identical
//! kernel), so the same product twice must match bit-for-bit.
//!
//! Shapes are deliberately non-square (m != n != k) in the forward check —
//! m = n = k would let a crossed layout or swapped operand read a REAL,
//! in-bounds, wrong sub-matrix and still land within tolerance-shaped
//! coincidence space (the PR #335 failure class). The wgrad check drives the
//! `nsl_tensor_wgrad_accum` production funnel: a TRANSPOSED first operand
//! (OP_T through `sgemm_wgrad_accum`) accumulating with beta = 1 — the case
//! where an autotune rep leaking into the caller's C (instead of the
//! transient scratch D) would corrupt live gradients and this gate's sums.
//!
//! Process isolation: the math mode and the Lt switch resolve once per
//! process, so the parent re-runs this binary as children — one with
//! `NSL_MATMUL_BF16_LT=1` (engagement + correctness + determinism), one
//! WITHOUT it (the default-off witness: zero Lt launches, same answers).

use nsl_runtime::tensor::{
    nsl_tensor_free, nsl_tensor_matmul, nsl_tensor_to_device, nsl_tensor_wgrad_accum,
    test_build_tensor_2d_f32, test_read_tensor_f64,
};
use nsl_runtime::{test_lt_matmul_reset, test_lt_matmul_stats};

const PROBE: &str = "NSL_BF16_LT_PROBE"; // "on" | "off"

/// Forward shape: [M x K] @ [K x N], all three distinct, intensity
/// mnk/(mk+kn) = 682.7 >= the 512 default gate, each GEMM ~3.2 GFLOP.
const M: usize = 2048;
const K: usize = 768;
const N: usize = 1024;

/// wgrad shape: m[D x O] += s * x[R x D]^T @ g[R x O]; intensity
/// d*o/(d+o) = 512 hits the gate exactly, R distinct from D = O.
const R: usize = 1536;
const D: usize = 1024;
const O: usize = 1024;

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
    // GPU->CPU migration converts f32 -> f64 by the params ABI; the reader
    // requires a CPU f64 tensor.
    let cpu = nsl_tensor_to_device(t, 0);
    let out = test_read_tensor_f64(cpu);
    if cpu != t {
        nsl_tensor_free(cpu);
    }
    out
}

fn bf16_of(v: f32) -> f64 {
    half::bf16::from_f32(v).to_f64()
}

/// Assert `got[i*n + j] == sum_k bf16(a[i,k]) * bf16(b[k,j])` (f64) at
/// `samples` deterministic positions, within `1e-4 * L1(products) + 1e-6`.
/// `a` is [rows x k] row-major unless `a_transposed` (then [k x rows], read
/// as its transpose — the wgrad x^T case).
#[allow(clippy::too_many_arguments)]
fn assert_sampled_product(
    label: &str,
    got: &[f64],
    a: &[f32],
    a_transposed: bool,
    b: &[f32],
    rows: usize,
    n: usize,
    k: usize,
    samples: usize,
    add_into: Option<&[f32]>,
    scale: f64,
) {
    assert_eq!(got.len(), rows * n, "{label}: output element count");
    let mut s: u64 = 0x5EED_C0DE ^ ((rows as u64) << 32) ^ (n as u64);
    for _ in 0..samples {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let i = (s >> 33) as usize % rows;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let j = (s >> 33) as usize % n;
        let mut acc = 0.0f64;
        let mut l1 = 0.0f64;
        for kk in 0..k {
            let av = if a_transposed { a[kk * rows + i] } else { a[i * k + kk] };
            let p = bf16_of(av) * bf16_of(b[kk * n + j]);
            acc += p;
            l1 += p.abs();
        }
        let mut want = scale * acc;
        if let Some(base) = add_into {
            want += base[i * n + j] as f64;
        }
        let have = got[i * n + j];
        let tol = 1e-4 * scale.abs() * l1 + 1e-6;
        assert!(
            (have - want).abs() <= tol,
            "{label}: C[{i},{j}] = {have}, reference {want} (tol {tol}) — a \
             wrong kernel/layout, not accumulation noise, at this magnitude"
        );
    }
}

#[test]
#[ignore = "spawned by lt_matmul_gates_gpu"]
fn zz_lt_probe_child() {
    let mode = std::env::var(PROBE).unwrap_or_default();
    if mode != "on" {
        return; // parent process (or the off-child, below the launcher).
    }

    let a_data = deterministic(M * K, 431);
    let w_data = deterministic(K * N, 433);
    let a = gpu_2d(M, K, &a_data);
    let w = gpu_2d(K, N, &w_data);

    // -- check 1: engagement + correctness on a non-square forward --------
    test_lt_matmul_reset();
    let c = nsl_tensor_matmul(a, w, 0);
    let c_host = read_gpu(c);
    let (issued, tuned, fallbacks) = test_lt_matmul_stats();
    assert!(
        issued >= 1,
        "NSL_MATMUL_BF16_LT=1 and a gate-passing shape, yet no GEMM issued \
         through cublasLt (issued={issued}, fallbacks={fallbacks}) — the \
         wiring is dead and any A/B would measure DFALT against DFALT"
    );
    assert_eq!(
        fallbacks, 0,
        "the Lt path declined on the primary shape (fallbacks={fallbacks}) — \
         look for the once-printed reason on stderr"
    );
    assert!(
        tuned >= 1,
        "autotune never timed a candidate set (tuned={tuned}). If the \
         heuristic genuinely returned a single candidate this assert is \
         wrong; every observation so far returns several — check the \
         preference attributes before relaxing the gate"
    );
    assert_sampled_product(
        "forward", &c_host, &a_data, false, &w_data, M, N, K, 2048, None, 1.0,
    );

    // -- check 2: run-to-run bit determinism ------------------------------
    // Fresh-scratch RNE casts + one cached plan (deterministic reduction
    // scheme) => the second product must be the SAME BITS, not merely close.
    let c2 = nsl_tensor_matmul(a, w, 0);
    let c2_host = read_gpu(c2);
    assert_eq!(
        c_host, c2_host,
        "same operands, same cached cublasLt plan, different bits — a \
         non-deterministic reduction scheme escaped the preference mask"
    );
    nsl_tensor_free(c2);
    nsl_tensor_free(c);

    // -- check 3: wgrad funnel — OP_T operand, beta = 1 accumulate --------
    let m0_data = deterministic(D * O, 439);
    let x_data = deterministic(R * D, 443);
    let g_data = deterministic(R * O, 449);
    let m = gpu_2d(D, O, &m0_data);
    let x = gpu_2d(R, D, &x_data);
    let g = gpu_2d(R, O, &g_data);
    let issued_before = test_lt_matmul_stats().0;
    nsl_tensor_wgrad_accum(m, x, g, 0.5);
    let m_host = read_gpu(m);
    let issued_after = test_lt_matmul_stats().0;
    assert!(
        issued_after > issued_before,
        "the wgrad accumulate did not issue through cublasLt \
         (issued {issued_before} -> {issued_after}) — the OP_T/beta=1 route \
         is falling back while the forward route engages"
    );
    // x is [R x D] and the product wants x^T @ g: pass x with the
    // transposed read. beta=1: the reference adds m0 back per element.
    assert_sampled_product(
        "wgrad", &m_host, &x_data, true, &g_data, D, O, R, 2048, Some(&m0_data), 0.5,
    );

    nsl_tensor_free(m);
    nsl_tensor_free(x);
    nsl_tensor_free(g);
    nsl_tensor_free(w);
    nsl_tensor_free(a);
    println!("LT-PROBE-OK");
}

#[test]
#[ignore = "spawned by lt_matmul_gates_gpu"]
fn zz_lt_off_child() {
    if std::env::var(PROBE).unwrap_or_default() != "off" {
        return;
    }
    // Same bf16 mode, NO NSL_MATMUL_BF16_LT: the Lt path must never arm —
    // its counters are the negative witness — and the GemmEx product must
    // satisfy the same reference (the two children agreeing with one f64
    // reference is this gate's substitute for a direct cross-path compare,
    // which one process cannot do: the switch resolves once).
    let a_data = deterministic(M * K, 431);
    let w_data = deterministic(K * N, 433);
    let a = gpu_2d(M, K, &a_data);
    let w = gpu_2d(K, N, &w_data);
    test_lt_matmul_reset();
    let c = nsl_tensor_matmul(a, w, 0);
    let c_host = read_gpu(c);
    let (issued, _tuned, fallbacks) = test_lt_matmul_stats();
    assert_eq!(
        (issued, fallbacks),
        (0, 0),
        "cublasLt path moved without NSL_MATMUL_BF16_LT=1"
    );
    assert_sampled_product(
        "forward-off", &c_host, &a_data, false, &w_data, M, N, K, 2048, None, 1.0,
    );
    nsl_tensor_free(c);
    nsl_tensor_free(w);
    nsl_tensor_free(a);
    println!("LT-OFF-OK");
}

fn spawn_child(mode: &str, child_test: &str, expect: &str) {
    let exe = std::env::current_exe().expect("current_exe");
    let mut cmd = std::process::Command::new(exe);
    cmd.args(["--test-threads=1", "--nocapture", "--include-ignored", child_test])
        .env(PROBE, mode)
        .env("NSL_MATMUL_BF16", "1")
        // Scrub EVERY mode var the resolvers consult, not just the one this
        // test owns (the cast-cache gate's rule): a leaked PEDANTIC=1
        // outranks BF16=1, a leaked MIN_RATIO>683 declines the forward
        // shape, and a leaked NSL_MATMUL_BF16_LT in the OFF child would
        // invert its whole premise.
        .env_remove("NSL_MATMUL_BF16_LT")
        .env_remove("NSL_MATMUL_BF16_LT_TUNE")
        .env_remove("NSL_MATMUL_BF16_LT_WORKSPACE_MIB")
        .env_remove("NSL_MATMUL_BF16_CAST_CACHE")
        .env_remove("NSL_MATMUL_BF16_ROUND")
        .env_remove("NSL_MATMUL_BF16_MIN_RATIO")
        .env_remove("NSL_MATMUL_PEDANTIC")
        .env_remove("NSL_MATMUL_TF32")
        .env_remove("NSL_ASYNC_ALLOC");
    if mode == "on" {
        cmd.env("NSL_MATMUL_BF16_LT", "1");
    }
    let out = cmd.output().expect("spawn lt probe child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.contains(expect),
        "lt {mode}-child failed (status {:?})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        out.status
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn lt_matmul_gates_gpu() {
    if std::env::var(PROBE).is_ok() {
        return; // child process: only its own probe runs there.
    }
    spawn_child("on", "zz_lt_probe_child", "LT-PROBE-OK");
    spawn_child("off", "zz_lt_off_child", "LT-OFF-OK");
}
