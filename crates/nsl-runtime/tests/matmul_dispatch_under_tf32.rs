//! Dispatch-path correctness UNDER TF32 — closing the gap every pinned suite
//! documents.
//!
//! `gpu_strided_view_ops`, `matmul_batch_collapse`, `matmul_transposed_operand`
//! and `wgrad_accum_fused` all pin `NSL_MATMUL_TF32=0`, on the argument that
//! they exist to catch operand-mapping bugs (order-1e0 errors) and full f32
//! keeps free headroom. Each of them carries the same admission: whether those
//! dispatch paths are correct *under* TF32 — the shipped default — was not
//! gated anywhere. TF32 selects DIFFERENT cuBLAS kernels, so "the mode applies
//! identically to both arms" arguments say nothing about a transa/lda mistake
//! that only that kernel family makes. This mattered twice over:
//! `the_op_t_tradeoff_is_remeasurable` showed OP_T is FASTER under TF32, and
//! the named precondition for flipping `NSL_MATMUL_TRANSPOSE_VIEWS` was
//! exactly "OP_T has correctness gates under TF32" — these gates are that
//! precondition, and the flip has since happened (the default now couples
//! to the math mode: OP_T under TF32, copy under FP32 cores/Pedantic).
//!
//! The math mode resolves once per process (`cublas_handle()`'s `OnceLock`),
//! so each configuration runs in a fresh child process — the pattern
//! `matmul_tf32_mode.rs` established. The child walks every 2-D/3-D dispatch
//! shape against an f64 CPU reference at a TF32 tolerance (5e-3 relative,
//! ~5x the measured ~1e-3 drift, three orders below an operand-mapping bug),
//! and first proves TF32 is actually ON by requiring a 256^3 product to drift
//! MORE than full f32 ever would — without that, a resolver regression would
//! make every gate here vacuously green.
//!
//! Run: cargo test -p nsl-runtime --features cuda --test matmul_dispatch_under_tf32
//!      -- --ignored --test-threads=1

#![cfg(feature = "cuda")]

use nsl_runtime::kernel_profiler::{
    nsl_kernel_profiler_flush, nsl_kernel_profiler_start, nsl_kernel_profiler_stop,
};
use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::tensor::arithmetic::nsl_tensor_wgrad_accum;
use nsl_runtime::tensor::{
    nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_matmul, nsl_tensor_transpose,
    nsl_tensor_zeros_on,
};
use nsl_runtime::{nsl_cuda_init, nsl_test_cuda_d2h, nsl_test_cuda_h2d};

/// Kernel names launched during `f` (same harness as
/// `matmul_transposed_operand::kernels_during`). Used to PROVE which dispatch
/// arm ran: at these small K the OP_T and copy+OP_N kernels can reduce in the
/// same order and produce bit-identical values, so a value comparison alone
/// cannot tell whether the exemption was active — the launch list can.
fn kernels_during(f: impl FnOnce()) -> Vec<String> {
    let dir = std::env::temp_dir().join(format!("nsl_tf32d_{}", std::process::id()));
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

/// Env var the child probe reads to know it is the child.
const PROBE: &str = "NSL_TF32_DISPATCH_PROBE";

fn cuda_available() -> bool {
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() {
        return false;
    }
    nsl_cuda_init() == 0
}

fn gpu_tensor(vals: &[f32], shape: &[i64]) -> i64 {
    let shape_list = nsl_list_new();
    for &dim in shape {
        nsl_list_push(shape_list, dim);
    }
    let t = nsl_tensor_zeros_on(shape_list, 1 /* GPU */);
    nsl_list_free(shape_list);
    nsl_test_cuda_h2d(
        nsl_tensor_data_ptr(t),
        vals.as_ptr() as i64,
        (vals.len() * 4) as i64,
    );
    t
}

fn read_gpu(t: i64, len: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; len];
    nsl_test_cuda_d2h(
        out.as_mut_ptr() as i64,
        nsl_tensor_data_ptr(t),
        (len * 4) as i64,
    );
    out
}

/// Deterministic pseudo-random values (LCG), same generator as the flash GPU tests.
fn det_seq(seed: u32, n: usize) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(1_103_515_245).wrapping_add(12_345);
            ((s >> 16) as f32 / 65535.0) - 0.5
        })
        .collect()
}

/// max |got - want| normalised by the RMS of the reference — scale-free, so
/// one tolerance covers K=16 and K=256 products alike.
fn rel_err(got: &[f32], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len());
    let rms = (want.iter().map(|v| v * v).sum::<f64>() / want.len() as f64).sqrt();
    let max = got
        .iter()
        .zip(want.iter())
        .map(|(&g, &w)| (f64::from(g) - w).abs())
        .fold(0.0f64, f64::max);
    max / rms.max(1e-12)
}

/// TF32 tolerance: ~5x the measured ~1e-3 drift, three orders of magnitude
/// below an operand-mapping bug (order 1e0).
const TOL: f64 = 5e-3;

fn check(name: &str, got: &[f32], want: &[f64]) {
    let e = rel_err(got, want);
    assert!(
        e < TOL,
        "{name}: rel err {e:.3e} exceeds the TF32 tolerance {TOL:.0e} — an \
         operand-mapping bug in the TF32 kernel family, not TF32 drift"
    );
    println!("TF32CHECK {name} max_rel={e:.3e}");
}

/// Child entry point. Runs every dispatch shape under whatever math mode /
/// transpose-views setting the parent exported. Prints one `TF32CHECK` line
/// per gate; panics (→ parent sees the failure) on any violation.
#[test]
fn zz_tf32_dispatch_child() {
    if std::env::var(PROBE).as_deref() != Ok("1") {
        return;
    }
    if !cuda_available() {
        return;
    }

    // ── Vacuity probe: TF32 must actually be ON in this process. ─────────
    // At K=256, full f32 drifts ~1e-6 relative; TF32 ~1e-3. Requiring more
    // than 1e-5 means a math-mode resolver regression (TF32 silently off)
    // fails HERE instead of making every gate below vacuously green.
    {
        let n = 256usize;
        let a = det_seq(91, n * n);
        let b = det_seq(92, n * n);
        let mut want = vec![0.0f64; n * n];
        for i in 0..n {
            for j in 0..n {
                let mut acc = 0.0f64;
                for d in 0..n {
                    acc += f64::from(a[i * n + d]) * f64::from(b[d * n + j]);
                }
                want[i * n + j] = acc;
            }
        }
        let at = gpu_tensor(&a, &[n as i64, n as i64]);
        let bt = gpu_tensor(&b, &[n as i64, n as i64]);
        let ct = nsl_tensor_matmul(at, bt, 0);
        let e = rel_err(&read_gpu(ct, n * n), &want);
        assert!(
            e > 1e-5,
            "vacuity probe: a 256^3 product drifted only {e:.3e} — that is \
             full-f32 accuracy, so TF32 is NOT active in this child and every \
             gate below would be meaningless. Did resolve_math_mode change?"
        );
        assert!(e < TOL, "vacuity probe: {e:.3e} exceeds even the TF32 tolerance");
        println!("TF32CHECK vacuity max_rel={e:.3e}");
        for t in [ct, bt, at] {
            nsl_tensor_free(t);
        }
    }

    // ── transposed RIGHT operand: A[8,16] @ T(B[64,16]) ──────────────────
    {
        let (m, k, n) = (8usize, 16usize, 64usize);
        let a = det_seq(93, m * k);
        let b = det_seq(94, n * k);
        let mut want = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f64;
                for d in 0..k {
                    acc += f64::from(a[i * k + d]) * f64::from(b[j * k + d]);
                }
                want[i * n + j] = acc;
            }
        }
        let at = gpu_tensor(&a, &[m as i64, k as i64]);
        let bt = gpu_tensor(&b, &[n as i64, k as i64]);
        let bv = nsl_tensor_transpose(bt, 0, 1);
        let mut ct = 0i64;
        let names = kernels_during(|| {
            ct = nsl_tensor_matmul(at, bv, 0);
        });
        check("transposed_right", &read_gpu(ct, m * n), &want);
        // Path witness: prove which dispatch arm actually ran. At K=16 both
        // arms can produce bit-identical values, so without this the OP_T
        // configuration would be indistinguishable from the copy arm and
        // the gates would be vacuous. The EXPECTATION comes from the parent
        // (NSL_TF32_DISPATCH_EXPECT_COPIED), not from re-deriving the
        // resolution here — the transpose-views default is coupled to the
        // math mode, and a child-side re-derivation would just mirror the
        // code under test.
        let copied = names.iter().any(|kn| kn == "nsl_strided_copy_f32");
        let expect_copied = match std::env::var("NSL_TF32_DISPATCH_EXPECT_COPIED").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            other => panic!("parent must declare the expected arm, got {other:?}"),
        };
        assert_eq!(
            copied, expect_copied,
            "dispatch arm does not match the parent's declared expectation \
             (strided-copy launched={copied}, expected {expect_copied}; \
             kernels: {})",
            names.join(",")
        );
        println!("TF32CHECK path_witness copied={copied}");
        for t in [ct, bv, bt, at] {
            nsl_tensor_free(t);
        }
    }

    // ── transposed LEFT operand: T(A[16,8]) @ B[16,12] ───────────────────
    {
        let (m, k, n) = (16usize, 8usize, 12usize);
        let a = det_seq(95, m * k); // A is [m,k]; view is [k,m]
        let b = det_seq(96, m * n);
        let mut want = vec![0.0f64; k * n];
        for i in 0..k {
            for j in 0..n {
                let mut acc = 0.0f64;
                for d in 0..m {
                    acc += f64::from(a[d * k + i]) * f64::from(b[d * n + j]);
                }
                want[i * n + j] = acc;
            }
        }
        let at = gpu_tensor(&a, &[m as i64, k as i64]);
        let av = nsl_tensor_transpose(at, 0, 1);
        let bt = gpu_tensor(&b, &[m as i64, n as i64]);
        let ct = nsl_tensor_matmul(av, bt, 0);
        check("transposed_left", &read_gpu(ct, k * n), &want);
        for t in [ct, bt, av, at] {
            nsl_tensor_free(t);
        }
    }

    // ── BOTH operands transposed: T(A[16,8]) @ T(B[12,16]) ───────────────
    // The one shape that catches transa/transb crossed, which neither
    // single-sided case can.
    {
        let (m, k, n) = (16usize, 8usize, 12usize);
        let a = det_seq(97, m * k); // view [k,m]=[8,16]
        let b = det_seq(98, n * m); // B [12,16], view [16,12]
        let mut want = vec![0.0f64; k * n];
        for i in 0..k {
            for j in 0..n {
                let mut acc = 0.0f64;
                for d in 0..m {
                    acc += f64::from(a[d * k + i]) * f64::from(b[j * m + d]);
                }
                want[i * n + j] = acc;
            }
        }
        let at = gpu_tensor(&a, &[m as i64, k as i64]);
        let av = nsl_tensor_transpose(at, 0, 1);
        let bt = gpu_tensor(&b, &[n as i64, m as i64]);
        let bv = nsl_tensor_transpose(bt, 0, 1);
        let ct = nsl_tensor_matmul(av, bv, 0);
        check("both_transposed", &read_gpu(ct, k * n), &want);
        for t in [ct, bv, bt, av, at] {
            nsl_tensor_free(t);
        }
    }

    // ── batch collapse: A[3,8,16] @ B[16,24] → strided-batched cuBLAS ────
    {
        let (bsz, m, k, n) = (3usize, 8usize, 16usize, 24usize);
        let a = det_seq(99, bsz * m * k);
        let b = det_seq(100, k * n);
        let mut want = vec![0.0f64; bsz * m * n];
        for bb in 0..bsz {
            for i in 0..m {
                for j in 0..n {
                    let mut acc = 0.0f64;
                    for d in 0..k {
                        acc += f64::from(a[bb * m * k + i * k + d]) * f64::from(b[d * n + j]);
                    }
                    want[bb * m * n + i * n + j] = acc;
                }
            }
        }
        let at = gpu_tensor(&a, &[bsz as i64, m as i64, k as i64]);
        let bt = gpu_tensor(&b, &[k as i64, n as i64]);
        let ct = nsl_tensor_matmul(at, bt, 0);
        check("batch_collapse", &read_gpu(ct, bsz * m * n), &want);
        for t in [ct, bt, at] {
            nsl_tensor_free(t);
        }
    }

    // ── the tied-LM-head composite: A[2,6,16] @ T(W[40,16]) ──────────────
    // Batch collapse and a transposed operand on the same call.
    {
        let (bsz, s, c, v) = (2usize, 6usize, 16usize, 40usize);
        let a = det_seq(101, bsz * s * c);
        let w = det_seq(102, v * c);
        let mut want = vec![0.0f64; bsz * s * v];
        for bb in 0..bsz {
            for i in 0..s {
                for j in 0..v {
                    let mut acc = 0.0f64;
                    for d in 0..c {
                        acc += f64::from(a[bb * s * c + i * c + d]) * f64::from(w[j * c + d]);
                    }
                    want[bb * s * v + i * v + j] = acc;
                }
            }
        }
        let at = gpu_tensor(&a, &[bsz as i64, s as i64, c as i64]);
        let wt = gpu_tensor(&w, &[v as i64, c as i64]);
        let wv = nsl_tensor_transpose(wt, 0, 1);
        let ct = nsl_tensor_matmul(at, wv, 0);
        check("lm_head_composite", &read_gpu(ct, bsz * s * v), &want);
        for t in [ct, wv, wt, at] {
            nsl_tensor_free(t);
        }
    }

    // ── fused wgrad accumulation: M += scale * sum_b X^T G ───────────────
    {
        let (bsz, tt, d, o) = (2usize, 8usize, 16usize, 24usize);
        let x = det_seq(103, bsz * tt * d);
        let g = det_seq(104, bsz * tt * o);
        let m0 = det_seq(105, d * o);
        let scale = 0.5f64;
        let mut want: Vec<f64> = m0.iter().map(|&v| f64::from(v)).collect();
        for bb in 0..bsz {
            for i in 0..d {
                for j in 0..o {
                    let mut acc = 0.0f64;
                    for t in 0..tt {
                        acc += f64::from(x[bb * tt * d + t * d + i])
                            * f64::from(g[bb * tt * o + t * o + j]);
                    }
                    want[i * o + j] += scale * acc;
                }
            }
        }
        let xt = gpu_tensor(&x, &[bsz as i64, tt as i64, d as i64]);
        let gt = gpu_tensor(&g, &[bsz as i64, tt as i64, o as i64]);
        let mt = gpu_tensor(&m0, &[d as i64, o as i64]);
        nsl_tensor_wgrad_accum(mt, xt, gt, scale);
        check("wgrad_accum", &read_gpu(mt, d * o), &want);
        for t in [mt, gt, xt] {
            nsl_tensor_free(t);
        }
    }
}

/// Names the child must have printed for a run to count. Kept in one place so
/// the parents cannot drift from the child's coverage.
const EXPECTED: &[&str] = &[
    "vacuity",
    "transposed_right",
    "path_witness",
    "transposed_left",
    "both_transposed",
    "batch_collapse",
    "lm_head_composite",
    "wgrad_accum",
];

fn spawn_child(configure: impl FnOnce(&mut std::process::Command)) -> String {
    let exe = std::env::current_exe().expect("test binary path");
    let mut cmd = std::process::Command::new(exe);
    cmd.args(["zz_tf32_dispatch_child", "--exact", "--nocapture", "--test-threads=1"])
        .env(PROBE, "1");
    configure(&mut cmd);
    let out = cmd.output().expect("spawn tf32 dispatch child");
    let text = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "tf32 dispatch child failed:\n{text}\nstderr:\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    for name in EXPECTED {
        assert!(
            text.contains(&format!("TF32CHECK {name} ")),
            "child never ran the '{name}' gate — coverage silently shrank:\n{text}"
        );
    }
    text
}

/// Every dispatch path, under the SHIPPED defaults — which now means TF32
/// math AND (through the math-mode coupling) OP_T for transposed views.
///
/// The child gets all three variables REMOVED (same reasoning as
/// `matmul_transposed_operand::probe_dispatch`: only a virgin process
/// resolves the real defaults), the vacuity probe requires the TF32
/// signature, and the path witness must see NO copy — so if either shipped
/// default moves, this fails loudly and gets updated consciously rather
/// than rotting.
#[test]
#[ignore = "requires CUDA GPU"]
fn dispatch_paths_are_correct_under_the_shipped_defaults() {
    if !cuda_available() {
        return;
    }
    spawn_child(|cmd| {
        cmd.env_remove("NSL_MATMUL_TF32")
            .env_remove("NSL_MATMUL_PEDANTIC")
            .env_remove("NSL_MATMUL_TRANSPOSE_VIEWS")
            .env("NSL_TF32_DISPATCH_EXPECT_COPIED", "0");
    });
}

/// Every dispatch path under TF32 with the materialising COPY forced — the
/// arm `NSL_MATMUL_TRANSPOSE_VIEWS=0` selects, which f32-pinned suites used
/// to be the only coverage for.
#[test]
#[ignore = "requires CUDA GPU"]
fn copy_arm_is_correct_under_tf32() {
    if !cuda_available() {
        return;
    }
    spawn_child(|cmd| {
        cmd.env("NSL_MATMUL_TF32", "1")
            .env("NSL_MATMUL_TRANSPOSE_VIEWS", "0")
            .env_remove("NSL_MATMUL_PEDANTIC")
            .env("NSL_TF32_DISPATCH_EXPECT_COPIED", "1");
    });
}

/// Every dispatch path under TF32 with the OP_T exemption forced — the gate
/// whose existence was the named precondition for the default flip
/// (`the_op_t_tradeoff_is_remeasurable` measured OP_T faster under TF32,
/// but nothing had ever checked its VALUES under the TF32 kernel family).
/// Kept as an explicit=1 configuration so it stays meaningful even if the
/// coupling changes.
#[test]
#[ignore = "requires CUDA GPU"]
fn op_t_exemption_is_correct_under_tf32() {
    if !cuda_available() {
        return;
    }
    spawn_child(|cmd| {
        cmd.env("NSL_MATMUL_TF32", "1")
            .env("NSL_MATMUL_TRANSPOSE_VIEWS", "1")
            .env_remove("NSL_MATMUL_PEDANTIC")
            .env("NSL_TF32_DISPATCH_EXPECT_COPIED", "0");
    });
}
