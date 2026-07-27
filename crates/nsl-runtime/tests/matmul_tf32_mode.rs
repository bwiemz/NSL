#![cfg(all(feature = "cuda", feature = "test-hooks"))]
//! Roadmap item 9: `NSL_MATMUL_TF32=1` must actually select TF32.
//!
//! ## What was wrong
//!
//! `CublasMathMode::Default` was documented as "`CUBLAS_DEFAULT_MATH` — TF32
//! tensor cores on sm_80+", the startup banner told users pedantic mode was
//! "~5-10x slower than TF32 default", and `NSL_MATMUL_TF32=1` selected that
//! same variant. All of it was false: `CUBLAS_DEFAULT_MATH` does not enable
//! TF32 for `cublasSgemm`. Measured at 4096^3 on an RTX 5070 Ti:
//!
//! | mode | TFLOP/s |
//! |---|---|
//! | "TF32 default" | 32.94 |
//! | `NSL_MATMUL_PEDANTIC=1` | **33.17** |
//!
//! Pedantic was marginally *faster*. Every f32 GEMM in the project was running
//! on FP32 CUDA cores, and the one flag offered to change that did nothing.
//!
//! ## Why the existing gate could not catch it
//!
//! `matmul_cublas_tf32_default_sanity.rs` checks a 5e-3 tolerance. Real f32 and
//! real TF32 both pass that comfortably, so the test is satisfied by either
//! and cannot distinguish them. A tolerance looser than the signal it is
//! supposed to detect is not a gate.
//!
//! This file asserts TF32 is BOTH measurably faster AND measurably less
//! accurate than the default. One direction alone is defeatable: "faster"
//! passes on noise, and "less accurate" passes on any old bug.
//!
//! ## Process isolation
//!
//! The math mode is resolved once, when the `OnceLock` cuBLAS handle is first
//! created, so a single process cannot exercise two modes. Each test here
//! re-executes THIS binary as a child with the env it wants — the same
//! constraint `tests/common/matmul_equiv.rs` documents, solved by spawning
//! rather than by splitting the file in two (a cross-process comparison is the
//! entire point of the gate).

use nsl_runtime::tensor::{
    nsl_tensor_free, nsl_tensor_matmul, nsl_tensor_to_device, test_build_tensor_2d_f32,
    test_read_tensor_f64,
};

const N: usize = 2048;
/// Env var the child looks for; absent means "run the tests, not the probe".
const PROBE: &str = "NSL_TF32_PROBE";

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

/// f64-accumulated CPU reference for row `i` only.
///
/// One row, not the whole matrix: an N^3 f64 reference at N=2048 takes minutes,
/// and the mantissa question is per-element — a single row of 2048 dot products
/// each of length 2048 is a large enough sample to separate 1e-7 from 1e-3 with
/// no ambiguity.
fn cpu_ref_row(a: &[f32], b: &[f32], i: usize, n: usize) -> Vec<f64> {
    (0..n)
        .map(|j| {
            let mut acc = 0.0f64;
            for p in 0..n {
                acc += a[i * n + p] as f64 * b[p * n + j] as f64;
            }
            acc
        })
        .collect()
}

/// Result of one in-process measurement, printed by the child for the parent.
struct Probe {
    secs_per_call: f64,
    max_rel_err: f64,
}

fn measure() -> Probe {
    let a_data = deterministic(N * N, 101);
    let b_data = deterministic(N * N, 103);
    let a = gpu_2d(N, N, &a_data);
    let b = gpu_2d(N, N, &b_data);

    for _ in 0..3 {
        nsl_tensor_free(nsl_tensor_matmul(a, b, 0));
    }
    let t0 = std::time::Instant::now();
    let iters = 20;
    for _ in 0..iters {
        nsl_tensor_free(nsl_tensor_matmul(a, b, 0));
    }
    let secs_per_call = t0.elapsed().as_secs_f64() / iters as f64;

    let c = nsl_tensor_matmul(a, b, 0);
    let c_cpu = nsl_tensor_to_device(c, 0);
    let got = test_read_tensor_f64(c_cpu);
    let row = 7usize;
    let want = cpu_ref_row(&a_data, &b_data, row, N);
    // Normalise by the row's RMS, not per element: these dot products are
    // random walks through zero and a per-element ratio would be dominated by
    // whichever entry happened to land nearest 0.
    let rms = (want.iter().map(|w| w * w).sum::<f64>() / want.len() as f64).sqrt();
    let max_rel_err = (0..N)
        .map(|j| (got[row * N + j] - want[j]).abs())
        .fold(0.0f64, f64::max)
        / rms;

    if c_cpu != c {
        nsl_tensor_free(c_cpu);
    }
    nsl_tensor_free(c);
    nsl_tensor_free(a);
    nsl_tensor_free(b);
    Probe { secs_per_call, max_rel_err }
}

/// Re-exec this binary with `NSL_TF32_PROBE=1` plus `extra`, and parse the one
/// line the probe prints.
fn probe_with(extra: &[(&str, &str)]) -> Probe {
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = std::process::Command::new(exe);
    // `--nocapture` is required, not cosmetic: libtest swallows a passing
    // test's stdout, so without it the child runs, measures, and the PROBE
    // line never reaches this process.
    cmd.args(["zz_probe_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(PROBE, "1")
        .env_remove("NSL_MATMUL_TF32")
        .env_remove("NSL_MATMUL_PEDANTIC")
        // REQUIRED. `inner::sync_after_kernel` is a no-op unless this is set,
        // so without it the timing loop measures kernel *enqueue* — which is
        // how the first run of this probe reported a 2048^3 gemm in 4.9us.
        // The readback in `measure` orders the stream, but it happens after
        // the clock has already stopped.
        .env("NSL_CUDA_SYNC", "1");
    for (k, v) in extra {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn probe child");
    let text = String::from_utf8_lossy(&out.stdout);
    // Substring, not prefix: libtest writes the test's own name in front of
    // captured output, so the line arrives as
    // `test zz_probe_child ... PROBE <secs> <err>`.
    let line = text
        .lines()
        .find_map(|l| l.find("PROBE ").map(|i| &l[i..]))
        .unwrap_or_else(|| {
            panic!(
                "probe child produced no PROBE line (env {extra:?})\nstdout:\n{text}\nstderr:\n{}",
                String::from_utf8_lossy(&out.stderr)
            )
        });
    let mut it = line.split_whitespace().skip(1);
    Probe {
        secs_per_call: it.next().unwrap().parse().unwrap(),
        max_rel_err: it.next().unwrap().parse().unwrap(),
    }
}

/// The child entry point. Cargo runs every `#[test]` in the binary, so the
/// probe has to be a test too — it returns immediately unless asked.
#[test]
fn zz_probe_child() {
    if std::env::var(PROBE).as_deref() != Ok("1") {
        return;
    }
    let p = measure();
    println!("PROBE {} {}", p.secs_per_call, p.max_rel_err);
}

#[test]
#[ignore = "requires CUDA GPU"]
fn tf32_is_faster_and_less_accurate_than_the_default() {
    let base = probe_with(&[]);
    let tf32 = probe_with(&[("NSL_MATMUL_TF32", "1")]);

    let speedup = base.secs_per_call / tf32.secs_per_call;
    let accuracy_ratio = tf32.max_rel_err / base.max_rel_err;
    eprintln!(
        "default: {:.4} ms, err {:e} | tf32: {:.4} ms, err {:e} | \
         speedup {speedup:.2}x, error x{accuracy_ratio:.1}",
        base.secs_per_call * 1e3,
        base.max_rel_err,
        tf32.secs_per_call * 1e3,
        tf32.max_rel_err
    );

    // Both directions. TF32 trades ~13 bits of mantissa for tensor cores, so a
    // build where the mode never reaches cuBLAS shows neither effect, and a
    // build where it reaches cuBLAS but the pointers are wrong shows only the
    // second.
    // Measured with `CUBLAS_TF32_TENSOR_OP_MATH` on an RTX 5070 Ti / CUDA 13.3
    // — so the deprecated math-mode route is still honoured there, and no
    // `cublasGemmEx` rewrite is needed yet:
    //
    //   N=2048   1.56 -> 0.38 ms   4.12x   error 1.4e-6 -> 9.5e-4  (x698)
    //   N=4096   4.54 -> 3.00 ms   1.51x   error 5.7e-6 -> 1.1e-3  (x190)
    //
    // The floor is set against the WEAKER of the two, not the headline.
    assert!(
        speedup > 1.3,
        "NSL_MATMUL_TF32=1 gave only {speedup:.2}x. Either the mode is not \
         reaching cuBLAS (CUBLAS_TF32_TENSOR_OP_MATH is deprecated — check \
         whether this CUDA version still honours it for cublasSgemm, and move \
         to cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32 if not), or this \
         GPU has no TF32 path."
    );
    assert!(
        accuracy_ratio > 20.0,
        "NSL_MATMUL_TF32=1 was {speedup:.2}x faster but only {accuracy_ratio:.1}x \
         less accurate ({:e} vs {:e}). TF32's 10-bit mantissa should cost \
         orders of magnitude, so this speedup is probably not TF32.",
        tf32.max_rel_err,
        base.max_rel_err
    );
}

/// The default must NOT have moved. Item 9 renamed the mode and corrected its
/// documentation; silently switching every matmul in the project to a 10-bit
/// mantissa would be a different change entirely, and one that needs a
/// tolerance audit across every numerical gate first.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_default_mode_is_still_full_f32() {
    let base = probe_with(&[]);
    let pedantic = probe_with(&[("NSL_MATMUL_PEDANTIC", "1")]);
    eprintln!(
        "default err {:e} | pedantic err {:e}",
        base.max_rel_err, pedantic.max_rel_err
    );
    // Measured 1.1e-7 for both at N=2048; TF32 lands near 1e-4. A factor of 4
    // between the two f32 modes is generous, and still an order of magnitude
    // clear of TF32.
    assert!(
        base.max_rel_err < pedantic.max_rel_err.max(1e-6) * 4.0,
        "the default matmul mode drifted to {:e} against pedantic's {:e} — the \
         default has been changed to something less precise than f32",
        base.max_rel_err,
        pedantic.max_rel_err
    );
}
