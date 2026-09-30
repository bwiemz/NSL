//! Silicon numerics gates for three KIR-migrated kernels that had NO GPU
//! numerics gate: `nsl_softmax_f32`, `nsl_layernorm_f32`, `nsl_rmsnorm_f32`
//! (forward) and `nsl_dropout_f32`
//! (roadmap item 2, the focused hardware-cert bundle).
//!
//! Each was migrated from hand PTX behind an interpreter equivalence gate
//! (`crates/nsl-codegen/tests/*_kir_equivalence.rs`). That proves the KIR
//! kernel MEANS the same PTX as its predecessor. It does not prove the kernel
//! is right on silicon, for two reasons `nsl_kir::kernels::norm` itself
//! records:
//!
//! * ptxas contracts an un-rounded `mul` + `add` into `fma` on hardware. A
//!   PTX-semantics interpreter rounds each instruction as written, so it cannot
//!   see that — the hand norm kernels were contracted, the KIR ones round
//!   twice. Interpreter byte-equivalence is not silicon byte-equivalence.
//! * A missing barrier is a timing race. The hand LayerNorm had one: mean and
//!   variance shared a shared-memory region with no barrier between a read and
//!   the next store, so a lagging warp could normalize with thread 0's
//!   VARIANCE partial in place of the mean. Four interpreter schedules may or
//!   may not expose that; real warps do, intermittently.
//!
//! So every gate here:
//!
//! 1. compares against a reference computed INDEPENDENTLY in this file — f64
//!    maths, or the dropout kernel's documented hash reimplemented — never the
//!    runtime's own CPU path, which would share the design under test;
//! 2. uses widths that straddle the 256-thread block (1, 7, 255, 256, 257,
//!    1000, 4096). A block-per-row kernel strides columns `tid, tid + 256, …`,
//!    and a width-8 test never takes a second trip round that stride;
//! 3. arms the launch census and asserts the named kernel LAUNCHED. Softmax,
//!    for one, redirects every non-last dim to the CPU and returns an equally
//!    correct device tensor, so a numerics check alone passes vacuously.
//!
//! Tolerances are DERIVED from the kernels' summation order (a per-thread fold
//! of `ceil(cols / 256)` terms, then thread 0 folding up to 256 partials),
//! not fitted to a measurement. Each gate prints the error it measured.
//!
//!   cargo test -p nsl-runtime --features cuda,test-hooks \
//!       --test hwcert_kir_numerics_gpu -- --include-ignored --test-threads=1
#![cfg(all(feature = "cuda", feature = "test-hooks"))]

use std::sync::Mutex;

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::{
    nsl_tensor_data_ptr, nsl_tensor_dropout, nsl_tensor_free, nsl_tensor_from_static,
    nsl_tensor_get_dtype, nsl_tensor_layernorm, nsl_tensor_rmsnorm, nsl_tensor_softmax,
    nsl_tensor_to_device,
};
use nsl_runtime::{test_kernel_launch_census_arm, test_kernel_launch_count};

const DTYPE_F32: i64 = 1;
const BLOCK: usize = 256;
/// f32 unit roundoff.
const U: f64 = 1.0 / (1u64 << 24) as f64;
/// Widths straddling the block, including its exact multiple and one past it.
const WIDTHS: [usize; 7] = [1, 7, 255, 256, 257, 1000, 4096];

/// The launch census and the dropout counter are process-global, and libtest
/// runs tests on parallel threads. One lock serializes this file's gates so an
/// `arm()` in one cannot zero another's counts mid-check.
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
    nsl_tensor_to_device(cpu, 1)
}

/// Download `n` elements as f32, branching on the dtype the download actually
/// produced rather than assuming it: reading f64 storage through an f32
/// pointer would be silent garbage. Widening f32 -> f64 -> f32 is exact and
/// keeps the sign of zero, which the dropout gate checks bit for bit.
fn download(t: i64, n: usize) -> Vec<f32> {
    let cpu = nsl_tensor_to_device(t, 0);
    let p = nsl_tensor_data_ptr(cpu);
    assert_ne!(p, 0, "downloaded tensor has no data pointer");
    let out = match nsl_tensor_get_dtype(cpu) {
        1 => unsafe { std::slice::from_raw_parts(p as *const f32, n) }.to_vec(),
        0 => unsafe { std::slice::from_raw_parts(p as *const f64, n) }
            .iter()
            .map(|&v| v as f32)
            .collect(),
        other => panic!("unexpected dtype {other} after download"),
    };
    nsl_tensor_free(cpu);
    out
}

/// Deterministic, order-sensitive data with full-width significands: a
/// Weyl sequence, so no two columns repeat and a mis-strided read shows up.
fn weyl(i: usize, lo: f32, hi: f32) -> f32 {
    let f = ((i as f64) * 0.618_033_988_749_894_9).fract() as f32;
    lo + (hi - lo) * f
}

/// Longest chain of dependent f32 adds in one row's reduction: a thread folds
/// `ceil(cols / 256)` columns, then thread 0 folds the other partials.
fn fold_chain(cols: usize) -> f64 {
    (cols.div_ceil(BLOCK) + cols.min(BLOCK)) as f64
}

// ───────────────────────────────── softmax ─────────────────────────────────

/// `nsl_softmax_f32` on silicon, against an f64 softmax computed here.
///
/// Row 0 is ordinary data in [-30, 30]. Row 1 is the same shape shifted by
/// +80, so `exp(x)` without the max subtraction would overflow f32 (it does
/// above ~88.7) — the stable formulation is load-bearing there. Row 2 masks
/// two columns with `-inf`, which must come back EXACTLY zero.
#[test]
#[ignore = "requires CUDA GPU"]
fn softmax_kir_kernel_matches_an_independent_f64_softmax_on_silicon() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    if !cuda_available() {
        eprintln!("skipping: no usable CUDA GPU");
        return;
    }
    test_kernel_launch_census_arm();
    let rows = 3usize;
    let mut launched = 0u64;
    let mut worst = 0.0f64;

    for &cols in &WIDTHS {
        let mut x = vec![0f32; rows * cols];
        for c in 0..cols {
            let v = weyl(c * 7 + cols, -30.0, 30.0);
            x[c] = v;
            x[cols + c] = v + 80.0;
            x[2 * cols + c] = weyl(c * 13 + 5, -30.0, 30.0);
        }
        if cols > 2 {
            x[2 * cols + 1] = f32::NEG_INFINITY;
            x[2 * cols + cols - 1] = f32::NEG_INFINITY;
        }

        let t = upload(&[rows, cols], &x);
        let y_t = nsl_tensor_softmax(t, -1);
        launched += 1;
        let y = download(y_t, rows * cols);

        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let m = row.iter().map(|&v| v as f64).fold(f64::NEG_INFINITY, f64::max);
            let e: Vec<f64> = row.iter().map(|&v| (v as f64 - m).exp()).collect();
            let s: f64 = e.iter().sum();
            // Every exponential carries ex2.approx's ~2 ulp; the sum's error is
            // bounded by its fold chain; the scale by rcp.approx's ~1 ulp.
            let rel_tol = (8.0 + fold_chain(cols)) * U * 4.0;
            let mut row_sum = 0f64;
            for c in 0..cols {
                let want = e[c] / s;
                let got = y[r * cols + c] as f64;
                row_sum += got;
                if row[c] == f32::NEG_INFINITY {
                    assert_eq!(
                        got.to_bits(),
                        0f64.to_bits(),
                        "softmax cols={cols} row={r} col={c}: a -inf input must give exactly +0, got {got:e}"
                    );
                    continue;
                }
                let err = (got - want).abs() / want.max(1e-30);
                worst = worst.max(err);
                assert!(
                    err <= rel_tol,
                    "softmax cols={cols} row={r} col={c}: got {got:e}, want {want:e} \
                     (rel err {err:.3e} > derived tol {rel_tol:.3e})"
                );
            }
            assert!(
                (row_sum - 1.0).abs() <= rel_tol * 2.0,
                "softmax cols={cols} row={r}: row sums to {row_sum:.9}, not 1"
            );
        }
        nsl_tensor_free(y_t);
        nsl_tensor_free(t);
    }

    let n = test_kernel_launch_count("nsl_softmax_f32");
    assert!(
        n >= launched,
        "nsl_softmax_f32 launched {n} time(s) for {launched} call(s): the op took \
         some other path, so the numerics above certified nothing about the kernel"
    );
    println!("softmax: {launched} launch(es) certified, worst relative error {worst:.3e}");
}

// ──────────────────────────────── LayerNorm ────────────────────────────────

/// f64 LayerNorm of one row over the SAME f32 inputs the kernel sees.
fn layernorm_ref(row: &[f32], gamma: &[f32], beta: &[f32], eps: f64) -> Vec<f64> {
    let n = row.len() as f64;
    let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n;
    let var = row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n;
    let inv = 1.0 / (var + eps).sqrt();
    row.iter()
        .zip(gamma.iter().zip(beta))
        .map(|(&x, (&g, &b))| (x as f64 - mean) * inv * g as f64 + b as f64)
        .collect()
}

/// Absolute tolerance for one LayerNorm row. The mean's error is bounded by
/// its fold chain times Σ|x|/n; divided by the row's std it becomes a relative
/// error in every normalized value — that ratio is the row's CONDITION NUMBER,
/// which is why an offset row needs a looser bound than a centred one. The
/// variance adds a comparable term, rsqrt.approx a few ulp.
fn layernorm_tol(row: &[f32], gamma: &[f32], beta: &[f32]) -> f64 {
    let n = row.len() as f64;
    let mean = row.iter().map(|&v| v as f64).sum::<f64>() / n;
    let std = (row.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / n).sqrt();
    let mean_abs = row.iter().map(|&v| (v as f64).abs()).sum::<f64>() / n;
    let cond = if std > 0.0 { mean_abs / std } else { 0.0 };
    let gmax = gamma.iter().map(|&g| (g as f64).abs()).fold(0.0, f64::max);
    let bmax = beta.iter().map(|&b| (b as f64).abs()).fold(0.0, f64::max);
    let chain = fold_chain(row.len());
    4.0 * U * (chain * (cond + 1.0) * 2.0 + 16.0) * (gmax * (1.0 + cond) + bmax) + 1e-7
}

/// `nsl_layernorm_f32` on silicon: correct against f64, AND bit-stable across
/// repeated launches.
///
/// The repetition is the race probe. The hand kernel's defect was a warp
/// reading a shared slot another thread had already overwritten — correct most
/// of the time, wrong when a warp fell a pass behind. That shows up as
/// launch-to-launch variation in the output bytes, so the gate launches the
/// same input five times over 64 rows (more rows = more CTAs for the scheduler
/// to skew) and requires identical bytes every time.
///
/// Row 3 is offset by +1000: a one-pass `E[x²] − E[x]²` variance would lose
/// every significant digit there, while this kernel's two-pass form keeps them.
/// The derived tolerance scales with the row's condition number, so it stays
/// tight on centred rows and still catches a one-pass regression by orders of
/// magnitude.
#[test]
#[ignore = "requires CUDA GPU"]
fn layernorm_kir_kernel_is_correct_and_race_free_on_silicon() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    if !cuda_available() {
        eprintln!("skipping: no usable CUDA GPU");
        return;
    }
    test_kernel_launch_census_arm();
    let rows = 64usize;
    let reps = 5usize;
    let eps = 1e-5f64;
    let mut launched = 0u64;
    let mut worst = 0.0f64;

    for &cols in &WIDTHS {
        let mut x = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let v = weyl(r * 4099 + c * 3 + cols, -3.0, 3.0);
                x[r * cols + c] = if r == 3 { 1000.0 + v } else { v };
            }
        }
        let gamma: Vec<f32> = (0..cols).map(|c| weyl(c + 11, 0.5, 1.5)).collect();
        let beta: Vec<f32> = (0..cols).map(|c| weyl(c + 29, -0.5, 0.5)).collect();

        let (t, g, b) = (upload(&[rows, cols], &x), upload(&[cols], &gamma), upload(&[cols], &beta));
        let mut first: Option<Vec<f32>> = None;
        for rep in 0..reps {
            let y_t = nsl_tensor_layernorm(t, g, b, eps);
            launched += 1;
            let y = download(y_t, rows * cols);
            nsl_tensor_free(y_t);
            match &first {
                None => first = Some(y),
                Some(f0) => {
                    let diff = f0.iter().zip(&y).position(|(a, b)| a.to_bits() != b.to_bits());
                    assert!(
                        diff.is_none(),
                        "layernorm cols={cols}: launch {rep} differs from launch 0 at element {} \
                         ({:e} vs {:e}) — nondeterministic output is the signature of the shared-\
                         memory race the hand kernel had",
                        diff.unwrap(),
                        f0[diff.unwrap()],
                        y[diff.unwrap()]
                    );
                }
            }
        }
        let y = first.expect("at least one launch");

        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let want = layernorm_ref(row, &gamma, &beta, eps);
            let tol = layernorm_tol(row, &gamma, &beta);
            for c in 0..cols {
                let got = y[r * cols + c] as f64;
                let err = (got - want[c]).abs();
                worst = worst.max(err / tol);
                assert!(
                    err <= tol,
                    "layernorm cols={cols} row={r} col={c}: got {got:e}, want {:e} \
                     (abs err {err:.3e} > derived tol {tol:.3e})",
                    want[c]
                );
            }
        }
        nsl_tensor_free(t);
        nsl_tensor_free(g);
        nsl_tensor_free(b);
    }

    let n = test_kernel_launch_count("nsl_layernorm_f32");
    assert!(
        n >= launched,
        "nsl_layernorm_f32 launched {n} time(s) for {launched} call(s): the op took \
         some other path, so the numerics above certified nothing about the kernel"
    );
    println!(
        "layernorm: {launched} launch(es) certified, bit-stable across {reps} reps, \
         worst error {worst:.3} of its derived tolerance"
    );
}

// ─────────────────────────────── RMSNorm forward ───────────────────────────────

/// `nsl_rmsnorm_f32` (forward) on silicon, against an f64 RMSNorm computed
/// here. Same KIR module as LayerNorm (`nsl_kir::kernels::norm`); until this
/// gate, only the RMSNorm BACKWARD kernels had a GPU numerics check.
///
/// `inv = rsqrt(Σx² / cols + eps)`, `out = (x · inv) · gamma`. Σx² is a sum of
/// non-negatives, so there is no cancellation and the error is RELATIVE and
/// uniform across a row — the tolerance is per element, scaled by the fold
/// chain, with no condition-number term. Row 3 is offset by +1000 anyway: it
/// costs nothing and checks that a large mean does not disturb the kernel.
#[test]
#[ignore = "requires CUDA GPU"]
fn rmsnorm_forward_kir_kernel_matches_an_independent_f64_rmsnorm_on_silicon() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    if !cuda_available() {
        eprintln!("skipping: no usable CUDA GPU");
        return;
    }
    test_kernel_launch_census_arm();
    let rows = 64usize;
    let reps = 3usize;
    let eps = 1e-5f64;
    let mut launched = 0u64;
    let mut worst = 0.0f64;

    for &cols in &WIDTHS {
        let mut x = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                let v = weyl(r * 4099 + c * 5 + cols, -3.0, 3.0);
                x[r * cols + c] = if r == 3 { 1000.0 + v } else { v };
            }
        }
        let gamma: Vec<f32> = (0..cols).map(|c| weyl(c + 17, 0.5, 1.5)).collect();
        let (t, g) = (upload(&[rows, cols], &x), upload(&[cols], &gamma));

        let mut first: Option<Vec<f32>> = None;
        for rep in 0..reps {
            let y_t = nsl_tensor_rmsnorm(t, g, eps);
            launched += 1;
            let y = download(y_t, rows * cols);
            nsl_tensor_free(y_t);
            match &first {
                None => first = Some(y),
                Some(f0) => assert!(
                    f0.iter().zip(&y).all(|(a, b)| a.to_bits() == b.to_bits()),
                    "rmsnorm cols={cols}: launch {rep} differs from launch 0"
                ),
            }
        }
        let y = first.expect("at least one launch");

        let rel_tol = 4.0 * U * (fold_chain(cols) + 16.0);
        for r in 0..rows {
            let row = &x[r * cols..(r + 1) * cols];
            let ms = row.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / cols as f64;
            let inv = 1.0 / (ms + eps).sqrt();
            for c in 0..cols {
                let want = row[c] as f64 * inv * gamma[c] as f64;
                let got = y[r * cols + c] as f64;
                let err = (got - want).abs() / want.abs().max(1e-30);
                worst = worst.max(err);
                assert!(
                    err <= rel_tol,
                    "rmsnorm cols={cols} row={r} col={c}: got {got:e}, want {want:e} \
                     (rel err {err:.3e} > derived tol {rel_tol:.3e})"
                );
            }
        }
        nsl_tensor_free(t);
        nsl_tensor_free(g);
    }

    let n = test_kernel_launch_count("nsl_rmsnorm_f32");
    assert!(
        n >= launched,
        "nsl_rmsnorm_f32 launched {n} time(s) for {launched} call(s): the op took \
         some other path, so the numerics above certified nothing about the kernel"
    );
    println!("rmsnorm forward: {launched} launch(es) certified, worst relative error {worst:.3e}");
}

// ───────────────────────────────── dropout ─────────────────────────────────

/// The hash documented in `nsl_kir::kernels::dropout`, reimplemented here from
/// that doc — NOT imported — so the gate checks the kernel against the
/// contract rather than against itself.
fn dropout_keep(seed: u64, i: u64, threshold: u32) -> bool {
    let mut h = seed.wrapping_add(i) as u32;
    h = h.wrapping_mul(0x9E37_79B9);
    h ^= h >> 16;
    h = h.wrapping_mul(0x85EB_CA6B);
    h ^= h >> 13;
    h = h.wrapping_mul(0xC2B2_AE35);
    h ^= h >> 16;
    h < threshold
}

/// `nsl_dropout_f32` on silicon, BIT-EXACT against the documented algorithm.
///
/// Dropout needs no statistical reference: its hash is specified, so every
/// output element is determined. A kept element is `x · scale` (one f32
/// multiply — nothing to contract), a dropped one is `x · 0`, which is `-0.0`
/// for a negative `x`; the comparison is on bits, so the sign of zero counts.
///
/// The counter base is pinned above 2³² and so that `base + i` wraps its low
/// 32 bits inside the launch, which exercises the `u32(seed + i)` truncation
/// the doc specifies. The gate also checks:
///
/// * the launch claims exactly `len` counter values — the contract that keeps
///   consecutive masks uncorrelated and makes checkpoint resume reproducible;
/// * the kept fraction lands within 6σ of `1 − p` — bit-exactness against a
///   reimplemented hash proves the kernel implements THAT hash, not that the
///   hash drops the right fraction;
/// * `training = 0` launches NOTHING and returns the input, so the census
///   count above is not satisfied by some unrelated launch.
#[test]
#[ignore = "requires CUDA GPU"]
fn dropout_kir_kernel_is_bit_exact_against_its_documented_hash_on_silicon() {
    let _g = SERIAL.lock().unwrap_or_else(|p| p.into_inner());
    if !cuda_available() {
        eprintln!("skipping: no usable CUDA GPU");
        return;
    }
    use nsl_runtime::rng_state::{gpu_dropout_counter, set_gpu_dropout_counter};

    let len = 1usize << 16;
    // Never exactly zero, so a kept element is distinguishable from a dropped one.
    let x: Vec<f32> = (0..len).map(|i| ((i % 97) as f32 - 48.5) * 0.37 + weyl(i, 0.0, 0.01)).collect();
    let saved_counter = gpu_dropout_counter();
    test_kernel_launch_census_arm();

    for &p in &[0.1f64, 0.5] {
        let base: u64 = 0x1_FFFF_FC00;
        set_gpu_dropout_counter(base);
        // The runtime derives both from `p` this way (cuda::gpu_dropout_f32).
        let threshold = ((1.0 - p) * u32::MAX as f64) as u32;
        let scale = (1.0 / (1.0 - p)) as f32;

        let t = upload(&[len], &x);
        let y_t = nsl_tensor_dropout(t, p, 1);
        let y = download(y_t, len);
        assert_eq!(
            gpu_dropout_counter(),
            base + len as u64,
            "p={p}: one launch must claim exactly len={len} counter values"
        );

        let mut kept = 0usize;
        for i in 0..len {
            let keep = dropout_keep(base, i as u64, threshold);
            kept += keep as usize;
            let want = x[i] * if keep { scale } else { 0.0 };
            assert_eq!(
                y[i].to_bits(),
                want.to_bits(),
                "dropout p={p} element {i}: got {:e} ({:#010x}), want {want:e} ({:#010x}); \
                 keep={keep}, x={:e}",
                y[i],
                y[i].to_bits(),
                want.to_bits(),
                x[i]
            );
        }
        let n = len as f64;
        let expect = n * (1.0 - p);
        let sigma = (n * p * (1.0 - p)).sqrt();
        assert!(
            (kept as f64 - expect).abs() <= 6.0 * sigma,
            "dropout p={p}: kept {kept} of {len}, expected {expect:.0} ± {:.0} (6σ)",
            6.0 * sigma
        );
        println!("dropout p={p}: bit-exact over {len} elements, kept {kept} (expected {expect:.0})");
        nsl_tensor_free(y_t);
        nsl_tensor_free(t);
    }
    let n = test_kernel_launch_count("nsl_dropout_f32");
    assert!(n >= 2, "nsl_dropout_f32 launched {n} time(s) for 2 call(s)");

    // Inverse anti-vacuity: with training off nothing launches.
    test_kernel_launch_census_arm();
    let t = upload(&[len], &x);
    let y_t = nsl_tensor_dropout(t, 0.5, 0);
    let y = download(y_t, len);
    assert_eq!(test_kernel_launch_count("nsl_dropout_f32"), 0, "training=0 must not launch");
    assert!(
        y.iter().zip(&x).all(|(a, b)| a.to_bits() == b.to_bits()),
        "training=0 must return the input unchanged"
    );
    nsl_tensor_free(y_t);
    nsl_tensor_free(t);

    set_gpu_dropout_counter(saved_counter);
}
