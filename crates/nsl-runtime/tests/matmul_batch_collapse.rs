#![cfg(all(feature = "cuda", feature = "test-hooks"))]
//! Roadmap item 9, phase 1: `[.., m, k] @ [k, n]` must reach cuBLAS.
//!
//! ## What was wrong
//!
//! `gpu_matmul_f32` sent work to cuBLAS **only when `total_batch == 1`**;
//! everything else launched `nsl_bmm_f32`, a 16x16 scalar-`fma.rn.f32` PTX
//! kernel. `total_batch` counts the OUTPUT batch dims, so `[B,S,C] @ [C,O]` —
//! the shape of every transformer projection — had `total_batch = B` and took
//! the naive arm. Nothing in codegen flattened it first.
//!
//! Measured on an RTX 5070 Ti / CUDA 13.3 before the fix:
//!
//! | | TFLOP/s |
//! |---|---|
//! | `[8,1024,512] @ [512,512]` (naive bmm) | 0.67 |
//! | the identical math as `[8192,512] @ [512,512]` (cuBLAS) | 29.54 |
//!
//! and `nsl run --profile-kernels` over a `GroupedQueryAttention` forward
//! showed **six `nsl_bmm_f32` and zero cuBLAS calls** — all four projections
//! plus QK^T and PV.
//!
//! ## What these gates pin
//!
//! The collapse is a reinterpretation of an existing buffer, so the danger is
//! not arithmetic — it is reading memory that does not have the assumed
//! layout. Three of the five tests below are therefore about the cases that
//! must NOT collapse; only one is about speed.
//!
//! `NSL_MATMUL_NO_BATCH_COLLAPSE=1` (or `test_set_batch_collapse_disabled`)
//! restores the old dispatch, which is what makes the parity and dispatch
//! gates non-vacuous: each asserts a DIFFERENT outcome per arm.

use nsl_runtime::kernel_profiler::{
    nsl_kernel_profiler_flush, nsl_kernel_profiler_start, nsl_kernel_profiler_stop,
};
use nsl_runtime::tensor::{
    nsl_tensor_free, nsl_tensor_matmul, nsl_tensor_to_device, test_build_tensor_2d_f32,
    test_read_tensor_f64, NslTensor,
};
use nsl_runtime::{test_cuda_device_synchronize, test_set_batch_collapse_disabled};

/// The cuBLAS handle and the collapse state are process-global, and the
/// kernel profiler is a singleton. Tests in one binary must not interleave.
static TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Take the serialising lock AND pin cuBLAS to full f32 for this process.
///
/// # Why the pin
///
/// The matmul default is now TF32 (see `cuda::resolve_math_mode`), which
/// drifts ~1e-3 relative. Every gate in this file compares a GPU product
/// against an f64 CPU reference at 1e-5 in order to catch **operand-mapping**
/// bugs — a crossed `lda`/`ldb`, an `OP_T` on the wrong cuBLAS operand, a
/// stride-0 broadcast mistaken for a transpose. Those produce errors of order
/// 1e0 (measured 2.0e1, 4.3e0 and 6.4e0 in the mutation controls), so the
/// tolerance has three orders of magnitude of headroom either way.
///
/// Widening these to ~5e-3 to accommodate TF32 would still catch a crossed
/// lda, but it would throw away the sensitivity for free, and these gates are
/// not the place TF32 belongs under test — `matmul_tf32_mode.rs` is, and it
/// asserts the default IS TF32 from a fresh process.
///
/// # Why it is sound to set this here
///
/// `resolve_math_mode` is read exactly once per process, inside
/// `cublas_handle()`'s `OnceLock` initialiser. Every GPU test in this file
/// goes through this function before touching cuBLAS, so whichever test runs
/// first performs the init with the variable already set — order-independent.
/// If that ever stops holding, the affected test fails with a ~1e-3 drift
/// message rather than passing quietly, so this is self-checking.
fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    unsafe { std::env::set_var("NSL_MATMUL_TF32", "0") };
    unsafe { std::env::set_var("NSL_MATMUL_BF16", "0") };
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}


fn deterministic(n: usize, seed: u64) -> Vec<f32> {
    // A cheap full-period LCG mapped to [-1, 1). Deliberately not `rand`: the
    // two dispatch arms must see BYTE-IDENTICAL inputs, and a shared seeded
    // RNG that either arm advances differently would silently weaken the
    // comparison into "two different problems agreed to 4 decimals".
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / 8_388_608.0) - 1.0
        })
        .collect()
}

fn to_gpu(t: i64) -> i64 {
    let g = nsl_tensor_to_device(t, 1);
    if g != t {
        nsl_tensor_free(t);
    }
    g
}

/// A contiguous GPU tensor of the given shape, filled row-major from `data`.
fn gpu_tensor(shape: &[i64], data: &[f32]) -> i64 {
    let len: i64 = shape.iter().product();
    assert_eq!(len as usize, data.len());
    let rows = shape[..shape.len() - 1].iter().product::<i64>() as usize;
    let cols = shape[shape.len() - 1] as usize;
    let flat = to_gpu(test_build_tensor_2d_f32(rows, cols, data));
    if shape.len() == 2 {
        return flat;
    }
    // Reshape via a view so the underlying buffer stays the same contiguous
    // allocation — which is exactly the layout the collapse depends on.
    let mut strides = vec![1i64; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    let view = NslTensor::new_view_i64(flat, shape, &strides, shape.len() as i64, len);
    nsl_tensor_free(flat);
    view
}

/// GPU->CPU transfer widens f32 to f64 (the CPU tensor dtype), exactly as
/// `tests/common/matmul_equiv.rs` does it; narrow back for comparison.
fn read_back(gpu_ptr: i64) -> Vec<f32> {
    let cpu = nsl_tensor_to_device(gpu_ptr, 0);
    let v: Vec<f32> = test_read_tensor_f64(cpu).into_iter().map(|x| x as f32).collect();
    if cpu != gpu_ptr {
        nsl_tensor_free(cpu);
    }
    v
}

/// `C[b,i,j] = sum_p A[b,i,p] * B[p,j]`, f64 accumulator.
///
/// f64 keeps the reference's own summation error near `k * 2.2e-16`, so the
/// error budget below is dominated entirely by the GPU side — the same trick
/// `tests/common/matmul_equiv.rs` documents.
fn cpu_ref(a: &[f32], b: &[f32], batch: usize, m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += a[bi * m * k + i * k + p] as f64 * b[p * n + j] as f64;
                }
                c[bi * m * n + i * n + j] = acc as f32;
            }
        }
    }
    c
}

/// Max absolute error, normalised by the RMS magnitude of the reference.
///
/// NOT per-element relative error. With uniform `[-1,1)` inputs a length-`k`
/// dot product is a random walk, so individual reference entries pass through
/// zero; dividing by one of those turns an absolute error of 1.6e-7 into a
/// "relative error" of 1.6e-4 and the gate starts reporting drift that is
/// purely an artefact of the denominator. That is exactly what the first run
/// of this file did — and the tell was that the NAIVE arm, which this change
/// does not touch, drifted just as much (1.26e-4).
///
/// Normalising by the RMS keeps the discrimination that matters: reading the
/// wrong memory (the failure mode the collapse could actually introduce)
/// produces errors on the order of the RMS itself, i.e. ~1.0 by this measure —
/// four orders of magnitude above the tolerance below.
fn max_err_vs_rms(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    // NaN check FIRST, and separately. `f64::max` propagates the non-NaN
    // operand, so `.fold(0.0, f64::max)` over an all-NaN difference returns
    // 0.0 and every tolerance below passes. That is not hypothetical for these
    // gates: the failure they exist to catch is reading the wrong device
    // memory, and recycled GPU memory is frequently NaN bit patterns.
    let bad = got.iter().position(|v| !v.is_finite());
    assert!(
        bad.is_none(),
        "GPU result has a non-finite value at index {} ({}) — the kernel read \
         memory it does not own, or wrote none at all",
        bad.unwrap(),
        got[bad.unwrap()]
    );
    let rms =
        (want.iter().map(|&w| (w as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    assert!(rms > 1e-6, "degenerate reference (rms {rms:e}) — the fixture is all zeros");
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g as f64 - w as f64).abs())
        .fold(0.0f64, f64::max)
        / rms
}

/// The metric's own guard. Not a GPU test — it pins the property that every
/// tolerance assertion in this file depends on, and that the metric did not
/// have when it was written.
#[test]
#[should_panic(expected = "non-finite")]
fn the_error_metric_rejects_nan_instead_of_scoring_it_perfect() {
    let want = vec![1.0f32, 2.0, 3.0, 4.0];
    let got = vec![f32::NAN; 4];
    // Without the explicit check this returns 0.0 — `f64::max` propagates the
    // non-NaN operand, so folding NaN differences from a 0.0 seed yields 0.0
    // and an all-garbage GPU result scores as a perfect match.
    let _ = max_err_vs_rms(&got, &want);
}

/// ...and still reports a real disagreement as one.
#[test]
fn the_error_metric_still_reports_finite_disagreement() {
    let want = vec![1.0f32, 2.0, 3.0, 4.0];
    let mut got = want.clone();
    got[2] = 3.5;
    assert!(max_err_vs_rms(&got, &want) > 0.1);
    assert_eq!(max_err_vs_rms(&want, &want), 0.0);
}

/// Run one `[batch, m, k] @ [k, n]` product on the GPU under a forced dispatch
/// arm, returning the result, the CPU reference, and the kernels that ran.
///
/// The kernel list is returned so the parity gate can VERIFY the two arms
/// really took different paths. Without it, deleting the fast path degrades
/// that gate into comparing the naive kernel against itself — which passes.
fn project(
    batch: usize,
    m: usize,
    k: usize,
    n: usize,
    collapse: bool,
) -> (Vec<f32>, Vec<f32>, Vec<String>) {
    test_set_batch_collapse_disabled(!collapse);
    let a_data = deterministic(batch * m * k, 11);
    let b_data = deterministic(k * n, 23);
    let a = gpu_tensor(&[batch as i64, m as i64, k as i64], &a_data);
    let b = gpu_tensor(&[k as i64, n as i64], &b_data);
    let mut got = Vec::new();
    let names = kernels_during(|| {
        let c = nsl_tensor_matmul(a, b, 0);
        assert_ne!(c, 0, "matmul returned a null tensor");
        got = read_back(c);
        nsl_tensor_free(c);
    });
    nsl_tensor_free(a);
    nsl_tensor_free(b);
    (got, cpu_ref(&a_data, &b_data, batch, m, k, n), names)
}

/// Kernel names observed while running `f`.
fn kernels_during(f: impl FnOnce()) -> Vec<String> {
    let dir = std::env::temp_dir().join(format!("nsl_bc_prof_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("kp.json");
    let path_str = path.to_str().unwrap().to_string();
    nsl_kernel_profiler_start();
    f();
    nsl_kernel_profiler_stop();
    // SAFETY: `path_str` is live UTF-8 for the duration of the call.
    unsafe { nsl_kernel_profiler_flush(path_str.as_ptr(), path_str.len() as i64) };
    let text = std::fs::read_to_string(&path).expect("profile written");
    std::fs::remove_dir_all(&dir).ok();
    let mut names = Vec::new();
    let mut cursor = 0usize;
    while let Some(rel) = text[cursor..].find("\"name\":\"") {
        let start = cursor + rel + 8;
        let Some(end) = text[start..].find('"') else { break };
        names.push(text[start..start + end].to_string());
        cursor = start + end + 1;
    }
    names
}

// ---------------------------------------------------------------------------

/// THE dispatch gate: the projection shape must reach cuBLAS, and must not
/// reach it when the collapse is off. Asserting both directions is what stops
/// this from passing on a build where the whole fast path was deleted.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_projection_shape_reaches_cublas() {
    // `into_inner` on poison: one failing test must not cascade into four
    // `PoisonError` failures that hide which gate actually broke.
    let _g = test_guard();
    let run = |collapse: bool| {
        test_set_batch_collapse_disabled(!collapse);
        kernels_during(|| {
            let a = gpu_tensor(&[4, 128, 64], &deterministic(4 * 128 * 64, 5));
            let b = gpu_tensor(&[64, 64], &deterministic(64 * 64, 7));
            let c = nsl_tensor_matmul(a, b, 0);
            nsl_tensor_free(c);
            nsl_tensor_free(a);
            nsl_tensor_free(b);
        })
    };

    let with = run(true);
    // Name the exact marker. "any gemm" is NOT sufficient, and this gate said
    // so for one commit before it was true: once the strided-batched arm
    // landed, `[batch,m,k] @ [k,n]` had a SECOND cuBLAS route (that arm handles
    // it with stride_b = 0), so deleting the collapse entirely left this test —
    // and all five others in this file — green while the shape regressed 1.66x.
    // A loose matcher on a two-route dispatch tests nothing.
    assert!(
        with.iter().any(|k| k == "sgemm_cublas"),
        "[4,128,64] @ [64,64] did not reach the single-sgemm collapse; \
         kernels seen: {with:?}"
    );
    assert!(
        !with.iter().any(|k| k == "sgemm_cublas_batched"),
        "[4,128,64] @ [64,64] went to the strided-batched arm instead of \
         collapsing. That is still cuBLAS and still correct, so nothing else \
         here would notice — but it launches one gemm per slice for a product \
         that is one gemm. Kernels seen: {with:?}"
    );
    assert!(
        !with.iter().any(|k| k == "nsl_bmm_f32"),
        "the naive batched kernel still ran alongside cuBLAS: {with:?}"
    );

    let without = run(false);
    assert!(
        without.iter().any(|k| k == "nsl_bmm_f32"),
        "with the collapse disabled the naive kernel should be back, but it \
         was not launched — this gate can no longer tell the two arms apart. \
         Kernels seen: {without:?}"
    );
}

/// Both dispatch arms must compute the same product. The collapse only
/// reinterprets A's existing buffer as `(batch*m) x k`, so a disagreement
/// beyond cuBLAS-vs-naive summation order means the flattening read the wrong
/// memory.
#[test]
#[ignore = "requires CUDA GPU"]
fn both_arms_agree_with_the_cpu_reference() {
    // `into_inner` on poison: one failing test must not cascade into four
    // `PoisonError` failures that hide which gate actually broke.
    let _g = test_guard();
    // Shapes chosen so the collapsed row count is NOT a multiple of any
    // plausible cuBLAS tile: an aligned-only fixture cannot catch an
    // off-by-one in the flattening (the lesson from item 8's parity gate,
    // whose params were all multiples of 256).
    for &(batch, m, k, n) in &[
        (2usize, 3usize, 5usize, 7usize),
        (3, 17, 33, 65),
        (4, 128, 64, 64),
        (8, 129, 63, 127),
    ] {
        let (collapsed, want, collapsed_kernels) = project(batch, m, k, n, true);
        let (naive, want2, naive_kernels) = project(batch, m, k, n, false);
        assert_eq!(want, want2, "the reference itself is not deterministic");

        // Anti-vacuity: prove the two arms are actually two arms, and that the
        // first one is the collapse specifically — `sgemm_cublas_batched` here
        // would mean this file is comparing the strided-batched arm against the
        // naive kernel and never touching the collapse at all.
        assert!(
            collapsed_kernels.iter().any(|kn| kn == "sgemm_cublas"),
            "[{batch},{m},{k}]: the 'collapsed' arm did not take the collapse \
             ({collapsed_kernels:?})"
        );
        assert!(
            !collapsed_kernels.iter().any(|kn| kn == "nsl_bmm_f32"),
            "[{batch},{m},{k}]: the 'collapsed' arm still ran the naive kernel \
             ({collapsed_kernels:?}) — this comparison is naive-vs-naive and \
             proves nothing about the fast path"
        );
        assert!(
            naive_kernels.iter().any(|kn| kn == "nsl_bmm_f32"),
            "[{batch},{m},{k}]: the 'naive' arm did not run the naive kernel \
             ({naive_kernels:?})"
        );

        let e_collapsed = max_err_vs_rms(&collapsed, &want);
        let e_naive = max_err_vs_rms(&naive, &want);
        // 1e-5 is the pedantic-arm tolerance from the 2026-04-21 cuBLAS-swap
        // spec. It applies unscaled here because every K above is <= 256, below
        // which that spec's K-scaled envelope is flat (see
        // `tests/common/matmul_equiv.rs::scaled_tolerance`). The collapse must
        // not need a looser tolerance than the 2-D path it now shares.
        assert!(
            e_collapsed < 1e-5,
            "[{batch},{m},{k}] @ [{k},{n}]: collapsed path drifted {e_collapsed:e} \
             from the f64 reference (naive path: {e_naive:e})"
        );
        assert!(
            e_naive < 1e-5,
            "[{batch},{m},{k}] @ [{k},{n}]: the NAIVE path drifted {e_naive:e} — \
             the reference or the fixture is wrong, not the new code"
        );
    }
}

/// A transposed 3-D operand must still produce the right product.
///
/// Flattening `[batch, m, k]` into `(batch*m) x k` assumes the batch slices are
/// adjacent and each is row-major; for a transposed or expanded view they are
/// not, and reading one as if they were is the tied-embedding miscompile of
/// PR #335 all over again — wrong numbers, no crash.
///
/// Note precisely what this does and does NOT cover. `nsl_tensor_matmul`
/// materialises both operands with `nsl_tensor_contiguous` before dispatch, so
/// by the time `gpu_matmul_f32` sees A it is already a fresh contiguous
/// buffer — the collapse DOES fire here, on the copy, and that is correct. The
/// `a.is_contiguous()` precondition inside `gpu_matmul_f32` is therefore
/// defence in depth against a future second caller, and this test cannot reach
/// it (the function is `pub(crate)`). What it does catch is the composition
/// breaking: remove the materialisation, or widen the collapse to accept a
/// strided A, and this fails.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_transposed_operand_still_computes_the_right_product() {
    // `into_inner` on poison: one failing test must not cascade into four
    // `PoisonError` failures that hide which gate actually broke.
    let _g = test_guard();
    test_set_batch_collapse_disabled(false);
    let (batch, m, k, n) = (3usize, 8usize, 5usize, 6usize);
    // Build A as the TRANSPOSE of a [batch, k, m] buffer, i.e. a view whose
    // last two dims are swapped and whose strides are non-monotonic.
    let src = deterministic(batch * k * m, 31);
    let base = gpu_tensor(&[batch as i64, k as i64, m as i64], &src);
    let shape = [batch as i64, m as i64, k as i64];
    let strides = [(k * m) as i64, 1, m as i64];
    let a_view = NslTensor::new_view_i64(base, &shape, &strides, 3, (batch * m * k) as i64);

    let b_data = deterministic(k * n, 37);
    let b = gpu_tensor(&[k as i64, n as i64], &b_data);
    let c = nsl_tensor_matmul(a_view, b, 0);
    let got = read_back(c);

    // Reference: materialise the transpose on the CPU, then the same product.
    let mut a_row = vec![0.0f32; batch * m * k];
    for bi in 0..batch {
        for p in 0..k {
            for i in 0..m {
                a_row[bi * m * k + i * k + p] = src[bi * k * m + p * m + i];
            }
        }
    }
    let want = cpu_ref(&a_row, &b_data, batch, m, k, n);
    let err = max_err_vs_rms(&got, &want);
    assert!(
        err < 1e-5,
        "a transposed 3-D operand produced the wrong product (max rel err {err:e}). \
         Either the collapse fired on a non-contiguous A, or the contiguous \
         materialisation in nsl_tensor_matmul stopped running."
    );

    nsl_tensor_free(c);
    nsl_tensor_free(a_view);
    nsl_tensor_free(base);
    nsl_tensor_free(b);
}

/// A genuinely batched B (`[batch, k, n]`, a different matrix per slice) has no
/// single 2-D B to share, so it must NOT be collapsed. It goes to
/// `cublasGemmStridedBatchedEx` instead — a different kernel computing the same
/// contraction, not a reinterpretation — and must still be correct.
///
/// The dispatch assertion names `sgemm_cublas_batched` specifically rather than
/// "any gemm": collapsing this shape would emit the plain `sgemm_cublas` marker
/// and silently multiply every slice by slice 0's B.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_batched_b_takes_the_strided_batched_path_not_the_collapse() {
    // `into_inner` on poison: one failing test must not cascade into four
    // `PoisonError` failures that hide which gate actually broke.
    let _g = test_guard();
    test_set_batch_collapse_disabled(false);
    let (batch, m, k, n) = (3usize, 4usize, 5usize, 6usize);
    let a_data = deterministic(batch * m * k, 41);
    let b_data = deterministic(batch * k * n, 43);
    let a = gpu_tensor(&[batch as i64, m as i64, k as i64], &a_data);
    let b = gpu_tensor(&[batch as i64, k as i64, n as i64], &b_data);

    let names = kernels_during(|| {
        let c = nsl_tensor_matmul(a, b, 0);
        nsl_tensor_free(c);
    });
    assert!(
        names.iter().any(|kname| kname == "sgemm_cublas_batched"),
        "a per-slice B must take the strided-batched cuBLAS path; \
         kernels seen: {names:?}"
    );
    assert!(
        !names.iter().any(|kname| kname == "sgemm_cublas"),
        "a per-slice B was collapsed into a single sgemm — every slice would be \
         multiplied by slice 0's B. Kernels seen: {names:?}"
    );

    // ...and still be correct.
    let c = nsl_tensor_matmul(a, b, 0);
    let got = read_back(c);
    let mut want = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    acc += a_data[bi * m * k + i * k + p] as f64
                        * b_data[bi * k * n + p * n + j] as f64;
                }
                want[bi * m * n + i * n + j] = acc as f32;
            }
        }
    }
    let err = max_err_vs_rms(&got, &want);
    assert!(err < 1e-5, "batched-B product drifted {err:e}");
    nsl_tensor_free(c);
    nsl_tensor_free(a);
    nsl_tensor_free(b);
}

/// `[1,m,k] @ [batch,k,n]` — A is BROADCAST across the batch, expressed to
/// cuBLAS as `strideA = 0`.
///
/// This is the one geometry where the strided-batched call does something the
/// naive kernel encoded differently, and a zero stride is easy to get wrong in
/// a way that produces plausible numbers: swap `stride_a` and `stride_b` at the
/// operand-swap site and slice 0 still comes out right, so a single-slice or
/// symmetric fixture would pass. Hence batch=3 with distinct per-slice data and
/// a full element-wise check.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_broadcast_operand_uses_a_zero_stride_correctly() {
    let _g = test_guard();
    test_set_batch_collapse_disabled(false);
    let (batch, m, k, n) = (3usize, 4usize, 5usize, 6usize);
    let a_data = deterministic(m * k, 61); // ONE slice, reused for all `batch`
    let b_data = deterministic(batch * k * n, 67);
    let a = gpu_tensor(&[1, m as i64, k as i64], &a_data);
    let b = gpu_tensor(&[batch as i64, k as i64, n as i64], &b_data);

    let names = kernels_during(|| {
        let c = nsl_tensor_matmul(a, b, 0);
        nsl_tensor_free(c);
    });
    assert!(
        names.iter().any(|kname| kname == "sgemm_cublas_batched"),
        "broadcast-A must reach the strided-batched path; kernels seen: {names:?}"
    );

    let c = nsl_tensor_matmul(a, b, 0);
    let got = read_back(c);
    let mut want = vec![0.0f32; batch * m * n];
    for bi in 0..batch {
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for p in 0..k {
                    // A carries NO batch index — that is the whole point.
                    acc += a_data[i * k + p] as f64 * b_data[bi * k * n + p * n + j] as f64;
                }
                want[bi * m * n + i * n + j] = acc as f32;
            }
        }
    }
    let err = max_err_vs_rms(&got, &want);
    assert!(
        err < 1e-5,
        "broadcast-A product drifted {err:e} — a zero stride is being applied \
         to the wrong operand, or not at all"
    );
    nsl_tensor_free(c);
    nsl_tensor_free(a);
    nsl_tensor_free(b);
}

/// Partial batch broadcast must refuse, not return an out-of-bounds read.
///
/// `[2,3,m,k] @ [2,1,k,n]` needs a different stride per batch DIMENSION; the
/// GPU path carries one stride per operand, so it walked B for 6 slices when B
/// holds 2. Measured error before the refusal: 2.61 relative to the result's
/// own RMS — and bit-identical through the naive kernel and through cuBLAS, so
/// this predates item 9 by a long way. The CPU path gets it right, which makes
/// it a silent CPU/GPU divergence.
///
/// This pins the refusal, not a fix. If per-dimension strides are implemented
/// later, this test should become a correctness check against the CPU
/// reference — deleting it would leave the OOB read unguarded.
///
/// Runs in a CHILD process. `#[should_panic]` does not work here: the refusal
/// fires inside `gpu_matmul_f32`, which is reached through the `extern "C"`
/// `nsl_tensor_matmul`, and a panic crossing an `extern "C"` boundary is a
/// non-unwinding abort — SIGABRT, not a catchable panic. So the assertion has
/// to be on the child's exit status and stderr.
#[test]
#[ignore = "requires CUDA GPU"]
fn partial_batch_broadcast_refuses_rather_than_reading_out_of_bounds() {
    let _g = test_guard();
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([
            "zz_partial_broadcast_child",
            "--exact",
            "--include-ignored",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("NSL_PARTIAL_BCAST_CHILD", "1")
        .output()
        .expect("spawn child");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "partial batch broadcast was ACCEPTED. The GPU path cannot express a \
         per-dimension broadcast stride, so it read past the end of the \
         smaller operand and returned plausible garbage (measured 2.61 \
         relative to the result's own RMS).\nstdout:\n{}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(
        stderr.contains("PARTIAL batch broadcasting"),
        "the child died, but not on the partial-broadcast refusal — so this \
         gate is passing for the wrong reason.\nstderr:\n{stderr}"
    );
}

/// Child body for the test above. Inert unless explicitly asked.
#[test]
#[ignore = "spawned by partial_batch_broadcast_refuses_rather_than_reading_out_of_bounds"]
fn zz_partial_broadcast_child() {
    if std::env::var("NSL_PARTIAL_BCAST_CHILD").as_deref() != Ok("1") {
        return;
    }
    test_set_batch_collapse_disabled(false);
    let (b0, b1, m, k, n) = (2usize, 3usize, 4usize, 5usize, 6usize);
    let a = gpu_tensor(
        &[b0 as i64, b1 as i64, m as i64, k as i64],
        &deterministic(b0 * b1 * m * k, 71),
    );
    // B's second batch dim is 1 while A's is 3: broadcast on ONE axis only.
    let b = gpu_tensor(&[b0 as i64, 1, k as i64, n as i64], &deterministic(b0 * k * n, 73));
    let c = nsl_tensor_matmul(a, b, 0);
    nsl_tensor_free(c);
    nsl_tensor_free(a);
    nsl_tensor_free(b);
}

/// The point of the change. Expressed as a RATIO against the same math written
/// 2-D, so the floor does not encode this particular GPU: the 3-D form now
/// runs the identical cuBLAS call, and the only difference left is dispatch
/// overhead.
///
/// Pre-fix ratio on an RTX 5070 Ti was **0.023** (0.67 vs 29.54 TFLOP/s).
#[test]
#[ignore = "requires CUDA GPU"]
fn the_projection_shape_is_not_slower_than_its_2d_form() {
    // `into_inner` on poison: one failing test must not cascade into four
    // `PoisonError` failures that hide which gate actually broke.
    let _g = test_guard();
    test_set_batch_collapse_disabled(false);
    let (batch, s, c_in, o) = (8usize, 1024usize, 512usize, 512usize);
    let x = deterministic(batch * s * c_in, 53);
    let w_data = deterministic(c_in * o, 59);
    let w = gpu_tensor(&[c_in as i64, o as i64], &w_data);
    let x3 = gpu_tensor(&[batch as i64, s as i64, c_in as i64], &x);
    let x2 = gpu_tensor(&[(batch * s) as i64, c_in as i64], &x);

    let time = |a: i64| -> f64 {
        // Long warmup, not 3 iterations: a freshly spawned process finds the
        // GPU idling at its 180 MHz floor against a 3090 MHz boost clock, and
        // ~14 ms of work never gets it there. Under-warming makes the
        // measurement swing by 2x run to run.
        for _ in 0..100 {
            nsl_tensor_free(nsl_tensor_matmul(a, w, 0));
        }
        test_cuda_device_synchronize();
        let iters = 50;
        let t0 = std::time::Instant::now();
        for _ in 0..iters {
            nsl_tensor_free(nsl_tensor_matmul(a, w, 0));
        }
        // REQUIRED — `sync_after_kernel` is a no-op unless NSL_CUDA_SYNC=1, so
        // without this the loop measures kernel ENQUEUE. Measured with the
        // collapse deleted: 0.0042 vs 0.0032 ms, ratio 0.78, gate green. The
        // original "11.8x slower" negative control for this test only
        // reproduced because the author had NSL_CUDA_SYNC=1 exported in their
        // shell; from a clean environment it proved nothing.
        test_cuda_device_synchronize();
        t0.elapsed().as_secs_f64() / iters as f64
    };
    let t3 = time(x3);
    let t2 = time(x2);
    let ratio = t2 / t3;
    // Measured with sync: 0.1478 vs 0.1469 ms, ratio 0.99 — the two forms run
    // the identical cuBLAS call, so anything below ~0.9 means they no longer
    // do. The old 0.5 floor was too loose to catch the 3-D form falling back
    // to the strided-batched arm (measured ratio 0.707 there).
    assert!(
        ratio > 0.85,
        "the 3-D projection is {:.2}x slower than the identical 2-D math \
         ({:.4} ms vs {:.4} ms). They should be the same cuBLAS call — check \
         the collapse preconditions in gpu_matmul_f32.",
        1.0 / ratio,
        t3 * 1e3,
        t2 * 1e3
    );
    nsl_tensor_free(x3);
    nsl_tensor_free(x2);
    nsl_tensor_free(w);
}
