#![cfg(all(feature = "cuda", feature = "test-hooks"))]
//! Roadmap item 9 phase 2: a 2-D transposed operand reaches cuBLAS as `OP_T`
//! instead of being copied.
//!
//! **This dispatch is OPT-IN (`NSL_MATMUL_TRANSPOSE_VIEWS=1`) and the default
//! is the fast one.** Read the next two sections before changing that.
//!
//! ## The obvious argument, and why it is wrong
//!
//! `nsl_tensor_matmul` materialises BOTH operands before every GPU dispatch.
//! That was correct while the target was a stride-blind PTX kernel — PR #335
//! found `emb @ embed.transpose(0,1)` silently computing the wrong product,
//! GPU loss climbing to the uniform plateau while CPU descended. Since the
//! target is now cuBLAS, which reads a transposed operand natively, the copy
//! looks like pure waste, and it is expensive: on a Coder-50M forward
//! (`--profile-kernels`, RTX 5070 Ti) `nsl_strided_copy_f32` was the single
//! most-launched kernel at **105 of 308 launches (34%)** moving **453 MB**,
//! and one of those is a 25,165,824-element copy — `49152 x 512` =
//! `VOCAB_SIZE x D_MODEL`, the weight-tied LM head at
//! models/coder50m/model.nsl:80, i.e. **96 MiB copied and freed every
//! forward** through a kernel doing a software 64-bit div/rem per element.
//!
//! That argument is wrong. The copy is not waste — it buys a much faster
//! GEMM. Measured per call, f32:
//!
//! | shape | `OP_T` | copy + `OP_N` | verdict |
//! |---|---|---|---|
//! | `[2048,512] @ [512,49152]` (the LM head) | 5.92 ms | 1.04 + 2.89 = **3.93 ms** | OP_T **1.51x SLOWER** |
//! | `[2048,512] @ [512,4096]` | 0.344 ms | 0.351 ms | a wash |
//! | `[512,512] @ [512,512]` | **0.023 ms** | 0.043 ms | OP_T 1.9x faster |
//!
//! End to end: 29 ms/forward with the copy, 33 ms without; peak GPU 2.10 GB
//! versus 2.01 GB. So this trades ~90 MB of peak memory for ~4 ms per forward.
//! Worth having when memory binds, a regression otherwise — hence opt-in, and
//! hence `the_default_still_materialises_transposed_operands` below.
//!
//! ## Why this is the dangerous direction
//!
//! Under the row-major `C^T = B^T A^T` swap, `transb` toggles the op of
//! cuBLAS's FIRST operand and `transa` its SECOND, and the leading dimensions
//! swap which extent they carry. Cross either and cuBLAS reads a real,
//! in-bounds, WRONG sub-matrix — no fault, no NaN, just a different answer.
//! So every test here checks VALUES against an f64 CPU reference, and the
//! transposed fixtures are deliberately non-square with distinct per-element
//! data: a square or symmetric fixture passes with the ops crossed.

use nsl_runtime::kernel_profiler::{
    nsl_kernel_profiler_flush, nsl_kernel_profiler_start, nsl_kernel_profiler_stop,
};
use nsl_runtime::test_set_transpose_views;
use nsl_runtime::tensor::{
    nsl_tensor_free, nsl_tensor_matmul, nsl_tensor_to_device, test_build_tensor_2d_f32,
    test_read_tensor_f64, NslTensor,
};

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
    std::env::set_var("NSL_MATMUL_TF32", "0");
    TEST_LOCK.lock().unwrap_or_else(|e| e.into_inner())
}


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

/// A `[cols, rows]` view over a contiguous `[rows, cols]` GPU buffer — exactly
/// what `.transpose(0, 1)` produces. Returns (view, base); free both.
fn transposed_view(rows: usize, cols: usize, data: &[f32]) -> (i64, i64) {
    let base = gpu_2d(rows, cols, data);
    let shape = [cols as i64, rows as i64];
    let strides = [1i64, cols as i64];
    let view = NslTensor::new_view_i64(base, &shape, &strides, 2, (rows * cols) as i64);
    (view, base)
}

fn read_back(gpu_ptr: i64) -> Vec<f32> {
    let cpu = nsl_tensor_to_device(gpu_ptr, 0);
    let v: Vec<f32> = test_read_tensor_f64(cpu).into_iter().map(|x| x as f32).collect();
    if cpu != gpu_ptr {
        nsl_tensor_free(cpu);
    }
    v
}

fn max_err_vs_rms(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    let bad = got.iter().position(|v| !v.is_finite());
    assert!(bad.is_none(), "non-finite GPU value at {}", bad.unwrap());
    let rms = (want.iter().map(|&w| (w as f64).powi(2)).sum::<f64>() / want.len() as f64).sqrt();
    assert!(rms > 1e-6, "degenerate reference");
    got.iter()
        .zip(want)
        .map(|(&g, &w)| (g as f64 - w as f64).abs())
        .fold(0.0f64, f64::max)
        / rms
}

/// `C[i,j] = sum_p A[i,p] * B[p,j]`, f64 accumulator, both row-major.
fn cpu_ref(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0f32; m * n];
    for i in 0..m {
        for j in 0..n {
            let mut acc = 0.0f64;
            for p in 0..k {
                acc += a[i * k + p] as f64 * b[p * n + j] as f64;
            }
            c[i * n + j] = acc as f32;
        }
    }
    c
}

fn kernels_during(f: impl FnOnce()) -> Vec<String> {
    let dir = std::env::temp_dir().join(format!("nsl_tview_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("kp.json");
    let path_str = path.to_str().unwrap().to_string();
    nsl_kernel_profiler_start();
    f();
    nsl_kernel_profiler_stop();
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

/// THE gate. The tied-LM-head shape must produce no strided copy at all.
///
/// Non-square on purpose (`m != k != n`), so a crossed `lda`/`ldb` cannot
/// coincidentally still be in bounds and right.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_transposed_right_operand_is_not_copied() {
    let _g = test_guard();
    test_set_transpose_views(true);
    let (m, k, n) = (24usize, 40usize, 56usize);
    let a_data = deterministic(m * k, 11);
    // `w` is stored [n, k]; the view presents it as [k, n] — the LM-head shape
    // `x @ embed.transpose(0, 1)` with embed: [vocab, d_model].
    let w_data = deterministic(n * k, 13);
    let a = gpu_2d(m, k, &a_data);
    let (bt, base) = transposed_view(n, k, &w_data);

    let names = kernels_during(|| {
        let c = nsl_tensor_matmul(a, bt, 0);
        nsl_tensor_free(c);
    });
    assert!(
        !names.iter().any(|kn| kn == "nsl_strided_copy_f32"),
        "the transposed operand was still materialised — that is the 96 MiB \
         per forward this change exists to remove. Kernels seen: {names:?}"
    );
    assert!(
        names.iter().any(|kn| kn == "sgemm_cublas"),
        "expected the 2-D cuBLAS arm; kernels seen: {names:?}"
    );

    // ...and the product must be right. Reference builds B row-major [k, n]
    // by transposing w_data explicitly.
    let mut b_row = vec![0.0f32; k * n];
    for p in 0..k {
        for j in 0..n {
            b_row[p * n + j] = w_data[j * k + p];
        }
    }
    let c = nsl_tensor_matmul(a, bt, 0);
    let err = max_err_vs_rms(&read_back(c), &cpu_ref(&a_data, &b_row, m, k, n));
    assert!(
        err < 1e-5,
        "transposed-B product drifted {err:e} — lda/ldb are crossed, or OP_T \
         was applied to the wrong cuBLAS operand"
    );
    nsl_tensor_free(c);
    nsl_tensor_free(bt);
    nsl_tensor_free(base);
    nsl_tensor_free(a);
}

/// The other direction: a transposed LEFT operand, which is what the
/// `dB = A^T @ dC` adjoint produces when `--fuse-wgrad-accum` is off.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_transposed_left_operand_is_not_copied() {
    let _g = test_guard();
    test_set_transpose_views(true);
    let (m, k, n) = (24usize, 40usize, 56usize);
    // A is stored [k, m]; presented as [m, k].
    let a_data = deterministic(k * m, 17);
    let b_data = deterministic(k * n, 19);
    let (at, base) = transposed_view(k, m, &a_data);
    let b = gpu_2d(k, n, &b_data);

    let names = kernels_during(|| {
        let c = nsl_tensor_matmul(at, b, 0);
        nsl_tensor_free(c);
    });
    assert!(
        !names.iter().any(|kn| kn == "nsl_strided_copy_f32"),
        "the transposed LEFT operand was still materialised: {names:?}"
    );

    let mut a_row = vec![0.0f32; m * k];
    for p in 0..k {
        for i in 0..m {
            a_row[i * k + p] = a_data[p * m + i];
        }
    }
    let c = nsl_tensor_matmul(at, b, 0);
    let err = max_err_vs_rms(&read_back(c), &cpu_ref(&a_row, &b_data, m, k, n));
    assert!(err < 1e-5, "transposed-A product drifted {err:e}");
    nsl_tensor_free(c);
    nsl_tensor_free(at);
    nsl_tensor_free(base);
    nsl_tensor_free(b);
}

/// BOTH operands transposed at once — the case where crossing `transa` with
/// `transb` cannot be caught by either single-sided test above.
#[test]
#[ignore = "requires CUDA GPU"]
fn both_operands_transposed() {
    let _g = test_guard();
    test_set_transpose_views(true);
    let (m, k, n) = (12usize, 20usize, 28usize);
    let a_data = deterministic(k * m, 23);
    let b_data = deterministic(n * k, 29);
    let (at, a_base) = transposed_view(k, m, &a_data);
    let (bt, b_base) = transposed_view(n, k, &b_data);

    let mut a_row = vec![0.0f32; m * k];
    for p in 0..k {
        for i in 0..m {
            a_row[i * k + p] = a_data[p * m + i];
        }
    }
    let mut b_row = vec![0.0f32; k * n];
    for p in 0..k {
        for j in 0..n {
            b_row[p * n + j] = b_data[j * k + p];
        }
    }
    let c = nsl_tensor_matmul(at, bt, 0);
    let err = max_err_vs_rms(&read_back(c), &cpu_ref(&a_row, &b_row, m, k, n));
    assert!(err < 1e-5, "double-transposed product drifted {err:e}");
    nsl_tensor_free(c);
    nsl_tensor_free(at);
    nsl_tensor_free(a_base);
    nsl_tensor_free(bt);
    nsl_tensor_free(b_base);
}

/// The transposed exemption must survive the batch collapse: `[B,S,C]` times a
/// transposed `[C,V]` is the real LM-head call, where the left operand is 3-D.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_tied_lm_head_shape_collapses_and_transposes_together() {
    let _g = test_guard();
    test_set_transpose_views(true);
    let (batch, s, c_in, vocab) = (2usize, 8usize, 16usize, 40usize);
    let x = deterministic(batch * s * c_in, 31);
    let emb = deterministic(vocab * c_in, 37); // [vocab, d_model]

    let x_flat = gpu_2d(batch * s, c_in, &x);
    let shape = [batch as i64, s as i64, c_in as i64];
    let strides = [(s * c_in) as i64, c_in as i64, 1];
    let x3 = NslTensor::new_view_i64(x_flat, &shape, &strides, 3, (batch * s * c_in) as i64);
    let (embt, emb_base) = transposed_view(vocab, c_in, &emb);

    let names = kernels_during(|| {
        let c = nsl_tensor_matmul(x3, embt, 0);
        nsl_tensor_free(c);
    });
    assert!(
        !names.iter().any(|kn| kn == "nsl_strided_copy_f32"),
        "the LM-head transpose was materialised even though the collapse arm \
         can express it: {names:?}"
    );
    assert!(
        names.iter().any(|kn| kn == "sgemm_cublas"),
        "expected the collapsed 2-D cuBLAS arm; kernels seen: {names:?}"
    );

    let mut e_row = vec![0.0f32; c_in * vocab];
    for p in 0..c_in {
        for j in 0..vocab {
            e_row[p * vocab + j] = emb[j * c_in + p];
        }
    }
    let c = nsl_tensor_matmul(x3, embt, 0);
    let err = max_err_vs_rms(&read_back(c), &cpu_ref(&x, &e_row, batch * s, c_in, vocab));
    assert!(err < 1e-5, "tied-LM-head product drifted {err:e}");
    nsl_tensor_free(c);
    nsl_tensor_free(x3);
    nsl_tensor_free(x_flat);
    nsl_tensor_free(embt);
    nsl_tensor_free(emb_base);
}

/// A view that is strided but NOT a plain 2-D transpose must still be copied.
/// `expand` produces a stride of 0, which is a broadcast, not a transpose —
/// treating it as one would read a single row as the whole matrix.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_broadcast_view_is_still_materialised() {
    let _g = test_guard();
    test_set_transpose_views(true);
    let (m, k, n) = (6usize, 5usize, 7usize);
    let a_data = deterministic(m * k, 41);
    let row = deterministic(n, 43);
    let a = gpu_2d(m, k, &a_data);
    // B is one row broadcast down k rows: shape [k, n], strides [0, 1].
    let row_t = gpu_2d(1, n, &row);
    let b = NslTensor::new_view_i64(row_t, &[k as i64, n as i64], &[0, 1], 2, (k * n) as i64);

    let c = nsl_tensor_matmul(a, b, 0);
    let mut b_row = vec![0.0f32; k * n];
    for p in 0..k {
        b_row[p * n..(p + 1) * n].copy_from_slice(&row);
    }
    let err = max_err_vs_rms(&read_back(c), &cpu_ref(&a_data, &b_row, m, k, n));
    assert!(
        err < 1e-5,
        "a stride-0 broadcast operand produced {err:e} — it was treated as a \
         transpose, so cuBLAS read one row as the whole matrix"
    );
    nsl_tensor_free(c);
    nsl_tensor_free(b);
    nsl_tensor_free(row_t);
    nsl_tensor_free(a);
}

/// Degenerate 2-D shapes still produce the right product with the exemption
/// ON.
///
/// **What this does NOT gate**, corrected after review: the original comment
/// claimed it pinned `is_transposed_2d_view`'s internal `is_contiguous()`
/// check — the ordering that stops a contiguous `[k, 1]` (strides `[1, 1]`)
/// being read as a transpose. It cannot. That predicate has exactly one
/// caller, `matmul_operand_needs_no_copy`, which returns `true` on the
/// contiguity fast path before ever reaching it, so the internal check is
/// unreachable defence and deleting it changes nothing observable. Every
/// tensor this test builds is contiguous, so it exercises the fast path, not
/// the predicate.
///
/// It is still worth running: these shapes are where the `m`/`n`/`k`-derived
/// leading dimensions collapse onto each other, and a wrong `lda` at
/// `[1, 5] @ [5, 1]` is arithmetically invisible in a way it is not at
/// `[24, 40]`.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_degenerate_unit_shape_is_not_mistaken_for_a_transpose() {
    let _g = test_guard();
    test_set_transpose_views(true);
    for &(m, k, n) in &[(1usize, 1usize, 1usize), (3, 1, 4), (1, 5, 1), (4, 1, 1)] {
        let a_data = deterministic(m * k, 47);
        let b_data = deterministic(k * n, 53);
        let a = gpu_2d(m, k, &a_data);
        let b = gpu_2d(k, n, &b_data);
        let c = nsl_tensor_matmul(a, b, 0);
        let err = max_err_vs_rms(&read_back(c), &cpu_ref(&a_data, &b_data, m, k, n));
        assert!(err < 1e-5, "[{m},{k}] @ [{k},{n}] drifted {err:e}");
        nsl_tensor_free(c);
        nsl_tensor_free(a);
        nsl_tensor_free(b);
    }
}

/// The caller's copy-exemption and the dispatcher's OP_T decision must be the
/// same predicate. They are literally one function
/// (`cuda::matmul_operand_needs_no_copy`); this pins that no second copy has
/// grown back, because a version that skipped the copy while the dispatcher
/// declined OP_T would hand a strided buffer to a stride-blind kernel.
#[test]
fn matmul_operand_exemption_agrees_with_dispatch() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("crates/nsl-runtime has two ancestors");
    let arith = std::fs::read_to_string(root.join("crates/nsl-runtime/src/tensor/arithmetic.rs"))
        .expect("arithmetic.rs readable");
    let cuda = std::fs::read_to_string(root.join("crates/nsl-runtime/src/cuda/mod.rs"))
        .expect("cuda/mod.rs readable");

    assert_eq!(
        arith.matches("matmul_operand_needs_no_copy").count(),
        2,
        "nsl_tensor_matmul must consult the shared predicate for BOTH operands \
         and nothing else — an inline stride test here is how the caller and \
         the dispatcher drift apart"
    );
    // Once for the definition, twice for the dispatcher's two operands.
    assert!(
        cuda.matches("matmul_operand_needs_no_copy").count() >= 3,
        "gpu_matmul_f32 no longer derives its OP_T flags from the shared \
         predicate; it has its own copy of the eligibility rule"
    );
    let start = cuda
        .find("fn is_transposed_2d_view")
        .expect("the transposed-view predicate still exists");
    let body = &cuda[start..start + 900];
    // The predicate's own contiguity rejection. Kept as a drift assertion but
    // honestly labelled after review: this is defence, not a live guard. The
    // sole caller returns on its contiguity fast path first, so removing the
    // inner check changes nothing observable today. It earns its place only
    // if a second caller ever appears — which is exactly when it would matter
    // and exactly when nobody would think to re-derive it.
    assert!(
        body.contains("t.is_contiguous()"),
        "is_transposed_2d_view no longer rejects contiguous tensors itself. \
         Unreachable behind today's only caller, but it makes the predicate \
         correct standalone: a contiguous [k,1] has strides [1,1] and matches \
         the transposed pattern when k == 1"
    );
    assert!(
        body.contains("strides[0] == 1") && body.contains("strides[1] == shape[0]"),
        "the transposed-view stride pattern has changed; a stride-0 broadcast \
         or a partial permutation may now qualify"
    );
}

/// Env var the child probe below reads to know it is the child.
const DEFAULT_PROBE: &str = "NSL_TRANSPOSE_DEFAULT_PROBE";

/// The child entry point for `the_default_still_materialises_transposed_operands`.
///
/// It runs in a FRESH PROCESS with `NSL_MATMUL_TRANSPOSE_VIEWS` removed from
/// the environment, and it deliberately does NOT call
/// `test_set_transpose_views` — that hook writes `TRANSPOSE_VIEWS_STATE`
/// directly, and `transpose_views_enabled()` short-circuits on an already
/// resolved state, so a test that used the hook would never execute the env
/// lookup at all.  Only a virgin process resolves the real default.
///
/// Prints one line the parent parses. Cargo runs every `#[test]` in the
/// binary, so this has to be a test; it returns immediately unless asked.
#[test]
fn zz_default_probe_child() {
    if std::env::var(DEFAULT_PROBE).as_deref() != Ok("1") {
        return;
    }
    let (m, k, n) = (24usize, 40usize, 56usize);
    let a = gpu_2d(m, k, &deterministic(m * k, 11));
    let (bt, base) = transposed_view(n, k, &deterministic(n * k, 13));
    let names = kernels_during(|| {
        let c = nsl_tensor_matmul(a, bt, 0);
        nsl_tensor_free(c);
    });
    nsl_tensor_free(bt);
    nsl_tensor_free(base);
    nsl_tensor_free(a);
    println!(
        "DEFAULTPROBE copied={} kernels={}",
        names.iter().any(|kn| kn == "nsl_strided_copy_f32"),
        names.join(",")
    );
}

/// The DEFAULT must still materialise. `OP_T` is opt-in because it is slower
/// on wide GEMMs — 5.92 ms versus 1.04 + 2.89 for the Coder-50M LM head — so a
/// build that silently enabled it would be a 1.51x regression on the exact
/// shape this file is named after.
///
/// **Why this spawns a child.** The first version of this gate called
/// `test_set_transpose_views(false)`, which sets the resolved-state atomic
/// directly. `transpose_views_enabled()` returns early on states 1 and 2 and
/// only consults `NSL_MATMUL_TRANSPOSE_VIEWS` in the unresolved arm — so that
/// version never ran the resolution logic, and flipping the default (say
/// `== Ok("1")` to `!= Ok("0")`, or seeding the atomic to "enabled") left the
/// gate green while every real run took the slower path. An independent
/// review caught it. A fresh process with the variable REMOVED is the only
/// way to observe the actual default.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_default_still_materialises_transposed_operands() {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([
            "zz_default_probe_child",
            "--exact",
            // Not cosmetic: libtest swallows a passing test's stdout, so
            // without this the child runs and the line never arrives.
            "--nocapture",
            "--test-threads=1",
        ])
        .env(DEFAULT_PROBE, "1")
        // The whole point of the gate: observe what happens with the opt-in
        // absent, whatever this shell happens to export.
        .env_remove("NSL_MATMUL_TRANSPOSE_VIEWS")
        .output()
        .expect("spawn default probe child");
    let text = String::from_utf8_lossy(&out.stdout);
    // Substring, not prefix: libtest prefixes captured output with the test's
    // own name.
    let line = text
        .lines()
        .find_map(|l| l.find("DEFAULTPROBE ").map(|i| &l[i..]))
        .unwrap_or_else(|| {
            panic!(
                "default-probe child produced no line\nstdout:\n{text}\nstderr:\n{}",
                String::from_utf8_lossy(&out.stderr)
            )
        });
    assert!(
        line.contains("copied=true"),
        "with NSL_MATMUL_TRANSPOSE_VIEWS absent the operand must still be \
         materialised — OP_T is 1.51x SLOWER on the LM-head shape and must \
         not become the default by accident. Child said: {line}"
    );
}

/// The opt-in must also actually be reachable THROUGH THE ENV VAR, not just
/// through the test hook. Without this, `the_default_still_materialises_*`
/// could pass simply because the feature was unreachable by any means.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_env_var_really_enables_the_exemption() {
    let exe = std::env::current_exe().expect("test binary path");
    let out = std::process::Command::new(exe)
        .args([
            "zz_default_probe_child",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(DEFAULT_PROBE, "1")
        .env("NSL_MATMUL_TRANSPOSE_VIEWS", "1")
        .output()
        .expect("spawn default probe child");
    let text = String::from_utf8_lossy(&out.stdout);
    let line = text
        .lines()
        .find_map(|l| l.find("DEFAULTPROBE ").map(|i| &l[i..]))
        .unwrap_or_else(|| {
            panic!(
                "default-probe child produced no line\nstdout:\n{text}\nstderr:\n{}",
                String::from_utf8_lossy(&out.stderr)
            )
        });
    assert!(
        line.contains("copied=false"),
        "NSL_MATMUL_TRANSPOSE_VIEWS=1 must skip the materialising copy — \
         if it does not, the opt-in is unreachable and the default gate \
         proves nothing. Child said: {line}"
    );
}
