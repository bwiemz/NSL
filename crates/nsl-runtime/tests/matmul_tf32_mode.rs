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
#[derive(Clone, Copy)]
struct Probe {
    secs_per_call: f64,
    max_rel_err: f64,
}

fn measure() -> Probe {
    let a_data = deterministic(N * N, 101);
    let b_data = deterministic(N * N, 103);
    let a = gpu_2d(N, N, &a_data);
    let b = gpu_2d(N, N, &b_data);

    // 200 warmup iterations, not 3. Each probe runs in a freshly spawned
    // process, and 3 + 20 iterations is roughly 14 ms of work — nowhere near
    // enough to move this GPU off its 180 MHz idle clock toward the 3090 MHz
    // boost. With the short warmup, eight consecutive runs of this test
    // measured speedups of 0.96, 0.74, 0.97, 1.29, 1.29, 1.30, 0.73 and 0.98:
    // every single one below the 1.3 floor, i.e. the gate failed on every run
    // while the feature underneath it worked. Clock state, not TF32.
    for _ in 0..200 {
        nsl_tensor_free(nsl_tensor_matmul(a, b, 0));
    }
    nsl_runtime::test_cuda_device_synchronize();
    let t0 = std::time::Instant::now();
    let iters = 50;
    for _ in 0..iters {
        nsl_tensor_free(nsl_tensor_matmul(a, b, 0));
    }
    nsl_runtime::test_cuda_device_synchronize();
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
    // NaN check first: `f64::max` drops NaN, so folding over an all-NaN row
    // would yield 0.0 and read as perfect accuracy.
    assert!(
        (0..N).all(|j| got[row * N + j].is_finite()),
        "GPU result row {row} contains a non-finite value"
    );
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
        .env_remove("NSL_MATMUL_BF16")
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

/// How many times each arm is measured. See `tf32_is_faster_...` for why.
const ROUNDS: usize = 3;

#[test]
#[ignore = "requires CUDA GPU"]
fn tf32_is_faster_and_less_accurate_than_full_f32() {
    // NOTE: `base` is the OPT-OUT, not the default. The default became TF32,
    // so comparing `probe_with(&[])` against `NSL_MATMUL_TF32=1` would be
    // TF32 against itself — a ~1.0x speedup and a ~1.0x error ratio, which
    // would fail this gate for entirely the wrong reason.
    //
    // BEST-OF-N, INTERLEAVED. A single sample per arm is not a safe way to
    // measure a ratio in this suite: the certification lane runs this gate
    // ~340 targets deep, and on a GPU that has been saturated for the better
    // part of an hour BOTH arms dilate. Observed on an RTX PRO 4500 mid-lane:
    // 2.79 ms / 2.48 ms against the 0.583 / 0.365 reference — every arm ~5x
    // slower, the ratio squeezed to 1.12x, and the gate red for a reason that
    // has nothing to do with TF32. In isolation the same binary passes.
    //
    // Taking the FASTEST sample per arm measures the throughput ceiling,
    // which is the quantity the assertion is actually about, and a transient
    // stall can only ever make a sample slower — never faster. It does not
    // weaken the check: if the mode never reaches cuBLAS the arms are the
    // same code and no number of rounds produces a fast one.
    //
    // INTERLEAVED, because alternating is what keeps a drifting clock from
    // landing on one arm. The inverse of this mistake is already recorded in
    // this tree: a "40% sgemm speedup" that turned out to be a cold clock,
    // whose tell was that kernels the flag could not touch also got faster.
    let (mut base, mut tf32) = (None::<Probe>, None::<Probe>);
    let faster = |acc: Option<Probe>, p: Probe| -> Option<Probe> {
        match acc {
            Some(a) if a.secs_per_call <= p.secs_per_call => Some(a),
            _ => Some(p),
        }
    };
    for _ in 0..ROUNDS {
        base = faster(base, probe_with(&[("NSL_MATMUL_TF32", "0")]));
        tf32 = faster(tf32, probe_with(&[]));
    }
    let (base, tf32) = (base.unwrap(), tf32.unwrap());

    let speedup = base.secs_per_call / tf32.secs_per_call;
    let accuracy_ratio = tf32.max_rel_err / base.max_rel_err;
    eprintln!(
        "best of {ROUNDS} — f32 (opt-out): {:.4} ms, err {:e} | tf32 (default): {:.4} ms, \
         err {:e} | speedup {speedup:.2}x, error x{accuracy_ratio:.1}",
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
    // — the deprecated math-mode route is still honoured there, so no
    // `cublasGemmEx` rewrite is needed yet. Steady state at N=2048, after the
    // warmup above:
    //
    //   default  0.583 ms  29.5 TFLOP/s   err 1.358e-6
    //   TF32     0.365 ms  47.1 TFLOP/s   err 9.482e-4
    //            -> 1.60x faster, 698x less accurate
    //
    // An earlier version of this comment claimed 4.12x. That number came from
    // an under-warmed run against an already-busy GPU and does not reproduce;
    // 1.60x is what holds when the clock is settled. The accuracy ratio, by
    // contrast, is bit-deterministic across every run.
    assert!(
        speedup > 1.35,
        "the TF32 default gave only {speedup:.2}x over NSL_MATMUL_TF32=0, best of {ROUNDS} \
         interleaved rounds. Check the error ratio printed above FIRST: if it is still ~400-700x \
         then TF32 IS reaching cuBLAS and this is a timing problem, not a dispatch one — compare \
         the absolute milliseconds against the 0.583/0.365 reference, and if BOTH arms are \
         dilated the GPU was contended (re-run the gate alone). If the error ratio is ~1x the \
         mode genuinely is not reaching cuBLAS: CUBLAS_TF32_TENSOR_OP_MATH is deprecated, so \
         check whether this CUDA version still honours it for cublasSgemm and move to \
         cublasGemmEx with CUBLAS_COMPUTE_32F_FAST_TF32 if not."
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
/// The default IS TF32 now, and the opt-out works.
///
/// Both halves matter. Asserting only "the default is fast" would pass
/// against a default that had silently become something else fast and less
/// accurate; asserting only "the opt-out is accurate" would pass against a
/// default that never changed. So this pins the default into TF32's accuracy
/// band — clearly worse than pedantic f32, clearly better than garbage — and
/// pins the opt-out back onto f32.
///
/// Measured at N=2048 on an RTX 5070 Ti: pedantic 1.4e-6, TF32 ~9.5e-4.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_default_mode_is_tf32_and_the_opt_out_restores_f32() {
    let base = probe_with(&[]);
    let pedantic = probe_with(&[("NSL_MATMUL_PEDANTIC", "1")]);
    let opted_out = probe_with(&[("NSL_MATMUL_TF32", "0")]);
    eprintln!(
        "default err {:e} | pedantic err {:e} | NSL_MATMUL_TF32=0 err {:e}",
        base.max_rel_err, pedantic.max_rel_err, opted_out.max_rel_err
    );

    // The default lost mantissa relative to strict f32. A default that
    // matched pedantic would mean the flip silently reverted.
    assert!(
        base.max_rel_err > pedantic.max_rel_err * 10.0,
        "the default matmul mode is as accurate as pedantic f32 ({:e} vs \
         {:e}) — TF32 is supposed to be the default and it is not in effect",
        base.max_rel_err,
        pedantic.max_rel_err
    );
    // ...but only about as much as TF32 should. 1e-2 is ~10x TF32's measured
    // drift: loose enough not to flake, tight enough that a default of f16 or
    // a genuinely broken kernel fails here.
    assert!(
        base.max_rel_err < 1e-2,
        "the default matmul mode drifted {:e}, well past TF32's ~1e-3 — the \
         default is not TF32, it is something worse",
        base.max_rel_err
    );

    // The escape hatch has to actually restore f32, or "opt out for full
    // precision" in the banner is a lie.
    assert!(
        opted_out.max_rel_err < pedantic.max_rel_err.max(1e-6) * 4.0,
        "NSL_MATMUL_TF32=0 drifted {:e} against pedantic's {:e} — the opt-out \
         does not restore full f32",
        opted_out.max_rel_err,
        pedantic.max_rel_err
    );
}
