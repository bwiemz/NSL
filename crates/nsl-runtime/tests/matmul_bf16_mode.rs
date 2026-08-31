#![cfg(all(feature = "cuda", feature = "test-hooks"))]
//! `NSL_MATMUL_BF16=1` must actually select BF16 tensor-core compute.
//!
//! ## Why this gate is shaped like `matmul_tf32_mode.rs`
//!
//! BF16 cannot ride the handle (`cublasMath_t` has no BF16 fast variant),
//! and the obvious per-call substitute was measured to be a LIE on this
//! stack: `cublasGemmEx` with f32 storage and `CUBLAS_COMPUTE_32F_FAST_16BF`
//! returns results bit-identical to the TF32 handle at TF32 speed — cuBLAS
//! 13.3 silently serves the TF32 kernels. The first draft of this mode
//! shipped exactly that no-op, and THIS GATE caught it (bf16 err == tf32
//! err to all 16 digits). The real mode casts operands to bf16 STORAGE and
//! runs `GemmEx(16BF, 16BF -> 32F)`, which measured 71.3 vs 36.4 TFLOPS at
//! N=4096.
//!
//! The silent-failure shapes, and the direction that catches each:
//!
//! 1. The env flag never reaches `resolve_math_mode` — every gemm keeps the
//!    TF32 default: caught by the accuracy direction (bf16 rounding of the
//!    INPUTS drifts ~7x TF32's band).
//! 2. The bf16-storage arm never engages (intensity gate broken, casts
//!    skipped, or the FAST_16BF regression above returns): caught by the
//!    accuracy direction again — f32-storage output is bit-identical to
//!    TF32 — and by the speed direction against the FP32-cores opt-out.
//!
//! ## Process isolation
//!
//! The math mode resolves once per process (`RESOLVED_MATH_MODE`), so each
//! probe re-executes this binary as a child with the env it wants.

use nsl_runtime::tensor::{
    nsl_tensor_free, nsl_tensor_matmul, nsl_tensor_to_device, test_build_tensor_2d_f32,
    test_read_tensor_f64,
};

const N: usize = 2048;
/// Env var the child looks for; absent means "run the tests, not the probe".
const PROBE: &str = "NSL_BF16_PROBE";

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

/// f64-accumulated CPU reference for row `i` only (see the TF32 gate for why
/// one row of 2048 dot products is a sufficient sample).
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

struct Probe {
    secs_per_call: f64,
    max_rel_err: f64,
}

fn measure() -> Probe {
    let a_data = deterministic(N * N, 101);
    let b_data = deterministic(N * N, 103);
    let a = gpu_2d(N, N, &a_data);
    let b = gpu_2d(N, N, &b_data);

    // 200 warmup iterations: a freshly spawned child starts at the idle
    // clock, and an under-warmed probe measured speedups anywhere between
    // 0.73x and 1.30x on a feature that worked (see the TF32 gate).
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
    let rms = (want.iter().map(|w| w * w).sum::<f64>() / want.len() as f64).sqrt();
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

/// Re-exec this binary with `NSL_BF16_PROBE=1` plus `extra`, and parse the
/// one line the probe prints. All three matmul-mode env vars are scrubbed
/// first so the parent's environment cannot leak into a probe cell.
fn probe_with(extra: &[(&str, &str)]) -> Probe {
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = std::process::Command::new(exe);
    cmd.args(["zz_probe_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(PROBE, "1")
        .env_remove("NSL_MATMUL_TF32")
        .env_remove("NSL_MATMUL_PEDANTIC")
        .env_remove("NSL_MATMUL_BF16")
        .env_remove("NSL_MATMUL_BF16_ROUND")
        // Without NSL_CUDA_SYNC the loop times kernel ENQUEUE, not the
        // kernel (the TF32 probe's first run reported a 2048^3 gemm at
        // 4.9us this way).
        .env("NSL_CUDA_SYNC", "1");
    for (k, v) in extra {
        cmd.env(k, v);
    }
    let out = cmd.output().expect("spawn probe child");
    let text = String::from_utf8_lossy(&out.stdout);
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

#[test]
fn zz_probe_child() {
    if std::env::var(PROBE).as_deref() != Ok("1") {
        return;
    }
    let p = measure();
    println!("PROBE {} {}", p.secs_per_call, p.max_rel_err);
}

/// BF16 must be faster than the FP32-cores opt-out AND in a distinctly worse
/// accuracy band than TF32. Failure shapes this catches, by direction:
/// not-faster => the GemmEx/FAST_16BF request never took effect (silent
/// FP32-cores); not-less-accurate-than-TF32 => the flag resolved to the TF32
/// default instead of BF16.
#[test]
#[ignore = "requires CUDA GPU"]
fn bf16_is_faster_than_f32_and_less_accurate_than_tf32() {
    // Min-of-3 with a repetition-spread validity check, the same shape as
    // `sr_rounding_keeps_bf16_speed_and_accuracy_class` below and for the
    // same reason: this box shares its GPU with multi-day training runs,
    // and single-shot arms reported bf16 at 0.38x of f32 while a tenant
    // held the card at 100%. The accuracy directions are deterministic and
    // always assert; only the speed directions are skippable, and only
    // when the arms' own spread says the box was not quiet.
    let best = |extra: &[(&str, &str)]| -> (Probe, f64) {
        let mut runs: Vec<Probe> = (0..3).map(|_| probe_with(extra)).collect();
        runs.sort_by(|a, b| a.secs_per_call.total_cmp(&b.secs_per_call));
        let spread = runs[runs.len() - 1].secs_per_call / runs[0].secs_per_call;
        (runs.swap_remove(0), spread)
    };
    let (f32_cores, f32_spread) = best(&[("NSL_MATMUL_TF32", "0")]);
    let (tf32, tf32_spread) = best(&[]); // the default
    let (bf16, bf16_spread) = best(&[("NSL_MATMUL_BF16", "1")]);
    let quiet = f32_spread < 1.25 && tf32_spread < 1.25 && bf16_spread < 1.25;

    let speedup_vs_f32 = f32_cores.secs_per_call / bf16.secs_per_call;
    let speedup_vs_tf32 = tf32.secs_per_call / bf16.secs_per_call;
    let err_vs_tf32 = bf16.max_rel_err / tf32.max_rel_err;
    eprintln!(
        "f32: {:.4} ms err {:e} | tf32: {:.4} ms err {:e} | bf16: {:.4} ms err {:e} | \
         bf16 speedup vs f32 {speedup_vs_f32:.2}x, vs tf32 {speedup_vs_tf32:.2}x, \
         error x{err_vs_tf32:.1} over tf32",
        f32_cores.secs_per_call * 1e3,
        f32_cores.max_rel_err,
        tf32.secs_per_call * 1e3,
        tf32.max_rel_err,
        bf16.secs_per_call * 1e3,
        bf16.max_rel_err
    );

    if quiet {
        assert!(
            speedup_vs_f32 > 1.35,
            "NSL_MATMUL_BF16=1 gave only {speedup_vs_f32:.2}x over NSL_MATMUL_TF32=0 (arm \
             spreads {f32_spread:.2}/{tf32_spread:.2}/{bf16_spread:.2}, so the box was quiet \
             and this is real). Either the mode is not reaching resolve_math_mode, or the \
             bf16-storage arm never engaged and the gemm ran f32."
        );
    } else {
        eprintln!(
            "  speed direction SKIPPED: arm spreads \
             {f32_spread:.2}/{tf32_spread:.2}/{bf16_spread:.2} exceed 1.25 — another GPU \
             tenant. The accuracy directions below still assert."
        );
    }
    // BF16 storage rounds the INPUTS to 8 mantissa bits (TF32 keeps exact
    // inputs and rounds products to 10) — measured ~7x TF32's drift at
    // N=4096. The 1.5x floor is deliberately far below that: bit-identical
    // (the FAST_16BF no-op failure) must fail, real bf16 must pass with
    // margin.
    assert!(
        err_vs_tf32 > 1.5,
        "BF16's error ({:e}) is not distinguishable from TF32's ({:e}) — either the flag \
         resolved to the TF32 default, or the bf16-storage arm silently fell back to \
         f32 storage (the FAST_16BF-is-a-no-op regression)",
        bf16.max_rel_err,
        tf32.max_rel_err
    );
    assert!(
        bf16.max_rel_err < 3e-2,
        "BF16 drifted {:e}, far past the ~7.5e-3 that bf16-input rounding costs at \
         this size — this is not BF16-with-f32-accumulate, something is broken",
        bf16.max_rel_err
    );
}

/// `NSL_MATMUL_BF16_ROUND=sr` must keep the mode's speed and its accuracy
/// class, and must remain OFF unless asked for.
///
/// SR exists to decorrelate the weight-operand error across optimizer steps —
/// see `bf16_rounding()` — and it is only worth having if it is close to free.
/// The dither is six integer ops per element folded into a cast that was
/// already bandwidth-bound, so the GEMM-level cost should be small; this
/// measures it rather than assuming it.
///
/// Accuracy is asserted as a BAND, not a direction. SR does not round more
/// accurately per cast — it rounds differently, trading a fixed error for a
/// zero-mean one, and its single-cast rms is slightly HIGHER than RNE's. What
/// must hold is that it stays in bf16's accuracy class: an SR arm that came
/// back at TF32's error would mean the bf16-storage path silently declined.
#[test]
#[ignore = "requires CUDA GPU"]
fn sr_rounding_keeps_bf16_speed_and_accuracy_class() {
    // Timing arms take the MIN of three repetitions, and report their own
    // spread so the ratio can be DISCARDED when the box is not quiet.
    //
    // This box shares its GPU with multi-day training runs, and a timing
    // assertion that fires on someone else's `pretrain_prod` is a false
    // alarm, not a guard. Two draft runs of this test reported SR at 906% and
    // 742% of RNE while a 27 GB tenant held the card at 100%; five
    // interleaved rounds on a quiet card put it at 8.4% with under 3% spread.
    // A minimum alone does not rescue that — under sustained contention every
    // repetition is slow — so the SPREAD is what says whether the number
    // means anything.
    let best = |extra: &[(&str, &str)]| -> (Probe, f64) {
        let mut runs: Vec<Probe> = (0..3).map(|_| probe_with(extra)).collect();
        runs.sort_by(|a, b| a.secs_per_call.total_cmp(&b.secs_per_call));
        let spread = runs[runs.len() - 1].secs_per_call / runs[0].secs_per_call;
        (runs.swap_remove(0), spread)
    };

    let (tf32, _) = best(&[]);
    let (rne, rne_spread) = best(&[("NSL_MATMUL_BF16", "1")]);
    let (sr, sr_spread) = best(&[("NSL_MATMUL_BF16", "1"), ("NSL_MATMUL_BF16_ROUND", "sr")]);

    let slowdown = sr.secs_per_call / rne.secs_per_call;
    let quiet = rne_spread < 1.25 && sr_spread < 1.25;
    eprintln!(
        "bf16+rne: {:.4} ms err {:e} | bf16+sr: {:.4} ms err {:e} | \
         sr costs {:.1}% | tf32 err {:e}",
        rne.secs_per_call * 1e3,
        rne.max_rel_err,
        sr.secs_per_call * 1e3,
        sr.max_rel_err,
        100.0 * (slowdown - 1.0),
        tf32.max_rel_err
    );

    if quiet {
        assert!(
            slowdown < 1.20,
            "SR rounding cost {:.1}% over RNE at this shape (arm spreads \
             {rne_spread:.2}/{sr_spread:.2}, so the box was quiet and this \
             number is real). The dither is six integer ops on a \
             bandwidth-bound cast; a cost this large means the cast stopped \
             being bandwidth-bound — check the grid sizing — rather than that \
             SR is expensive. Measured 8-11% on a quiet card.",
            100.0 * (slowdown - 1.0)
        );
    } else {
        eprintln!(
            "  timing arm SKIPPED: repetition spread {rne_spread:.2}/{sr_spread:.2} \
             exceeds 1.25, so this GPU has another tenant and the ratio is \
             not a measurement. The accuracy assertions below still hold — \
             they are deterministic."
        );
    }
    assert!(
        sr.max_rel_err > 1.5 * tf32.max_rel_err,
        "SR arm's error ({:e}) is in TF32's class ({:e}) — the bf16-storage \
         path did not engage under NSL_MATMUL_BF16_ROUND=sr, so this arm is \
         measuring the wrong thing",
        sr.max_rel_err,
        tf32.max_rel_err
    );
    assert!(
        sr.max_rel_err < 3.0 * rne.max_rel_err,
        "SR arm drifted {:e} against RNE's {:e}. SR is expected to be slightly \
         worse per cast, not multiples worse — this looks like a broken \
         dither, not stochastic rounding",
        sr.max_rel_err,
        rne.max_rel_err
    );

    // Opt-in: an unrecognised value must fall through to RNE, not engage SR
    // and not change arithmetic (the tri-state discipline the mode resolvers
    // all follow).
    let typo = probe_with(&[("NSL_MATMUL_BF16", "1"), ("NSL_MATMUL_BF16_ROUND", "SR")]);
    assert_eq!(
        typo.max_rel_err, rne.max_rel_err,
        "NSL_MATMUL_BF16_ROUND=SR (wrong case) changed the arithmetic; only \
         the literal \"sr\" may engage stochastic rounding"
    );
}

/// The mode is opt-in: `NSL_MATMUL_BF16=0` (the defensive pin scripts use)
/// and unset must both leave the TF32 default exactly in place.
#[test]
#[ignore = "requires CUDA GPU"]
fn bf16_zero_is_a_no_op_and_the_default_stays_tf32() {
    let default = probe_with(&[]);
    let pinned_off = probe_with(&[("NSL_MATMUL_BF16", "0")]);
    let pedantic = probe_with(&[("NSL_MATMUL_PEDANTIC", "1")]);
    eprintln!(
        "default err {:e} | NSL_MATMUL_BF16=0 err {:e} | pedantic err {:e}",
        default.max_rel_err, pinned_off.max_rel_err, pedantic.max_rel_err
    );
    for (label, p) in [("default", &default), ("NSL_MATMUL_BF16=0", &pinned_off)] {
        assert!(
            p.max_rel_err > pedantic.max_rel_err * 10.0 && p.max_rel_err < 1e-2,
            "{label} error {:e} is outside the TF32 band (pedantic {:e}) — \
             adding the BF16 flag must not move the default",
            p.max_rel_err,
            pedantic.max_rel_err
        );
    }
}

/// Precedence: PEDANTIC beats BF16; BF16 beats TF32 when both are set.
/// Asserted by accuracy signature, which is bit-deterministic per mode.
#[test]
#[ignore = "requires CUDA GPU"]
fn precedence_pedantic_beats_bf16_and_bf16_beats_tf32() {
    let pedantic = probe_with(&[("NSL_MATMUL_PEDANTIC", "1")]);
    let tf32 = probe_with(&[]);
    let ped_and_bf16 =
        probe_with(&[("NSL_MATMUL_PEDANTIC", "1"), ("NSL_MATMUL_BF16", "1")]);
    let bf16_and_tf32 =
        probe_with(&[("NSL_MATMUL_BF16", "1"), ("NSL_MATMUL_TF32", "1")]);
    eprintln!(
        "pedantic {:e} | pedantic+bf16 {:e} | tf32 {:e} | bf16+tf32 {:e}",
        pedantic.max_rel_err,
        ped_and_bf16.max_rel_err,
        tf32.max_rel_err,
        bf16_and_tf32.max_rel_err
    );
    assert!(
        ped_and_bf16.max_rel_err < pedantic.max_rel_err.max(1e-6) * 4.0,
        "PEDANTIC=1 + BF16=1 drifted {:e} against pedantic's {:e} — pedantic \
         must beat BF16",
        ped_and_bf16.max_rel_err,
        pedantic.max_rel_err
    );
    assert!(
        bf16_and_tf32.max_rel_err > tf32.max_rel_err * 1.5,
        "BF16=1 + TF32=1 landed in TF32's band ({:e} vs {:e}) — BF16 must win \
         when both opt-ins are set",
        bf16_and_tf32.max_rel_err,
        tf32.max_rel_err
    );
}
