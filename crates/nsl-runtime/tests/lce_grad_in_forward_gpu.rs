#![cfg(all(feature = "cuda", feature = "test-hooks"))]
//! `nsl_fused_linear_ce_forward_gemm_grad`, the fused LM head's forward that
//! also computes its backward, held to an f64 reference on both of its
//! routes, in the BF16 matmul mode:
//!
//! - **vocabulary-chunked** (the head weight has no bf16 image yet, the first
//!   accumulation window of a run): the GEMM-chunked forward, then its
//!   backward, in one call;
//! - **row-chunked** (the weight registered with the cast cache through the
//!   production funnel, a zero-lr fused AdamW step): bf16 logits per chunk of
//!   rows, the per-row softmax kernel, the dx and dW GEMMs. `NSL_LCE_ROW_CHUNK`
//!   = 96 splits the 232 rows into chunks of 96, 96 and 40.
//!
//! Each call writes the per-row losses, the x gradient (`scale` = `grad /
//! num_valid`, with ignored rows) and ADDS `alpha · dlogitsᵀ x` into a
//! non-zero accumulator. The reference is computed in f64 from the bf16-rounded
//! operands; the bounds cover the bf16 logits and gradient roundings and are
//! paired with the reference magnitudes (a factor-2 scale, a lost chunk or a
//! crossed row offset errs by tens of percent). The launch counters prove
//! which route ran.
//!
//! Process isolation: the math mode resolves once per process, so the parent
//! test re-runs this binary as a child with `NSL_MATMUL_BF16=1` (the
//! `matmul_bf16_cast_cache.rs` pattern).

use half::bf16;
use nsl_runtime::fase_step::nsl_fase_fused_adamw_step;
use nsl_runtime::fused_linear_ce::{
    nsl_fused_lce_launch_count, nsl_fused_linear_ce_forward_gemm_grad,
};
use nsl_runtime::tensor::{
    nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_to_device, test_build_tensor_2d_f32,
    test_read_tensor_f32,
};
use nsl_runtime::{nsl_test_cuda_alloc, nsl_test_cuda_free, nsl_test_cuda_h2d};

const ROWS: usize = 232;
const V: usize = 4224;
const H: usize = 256;
const ALPHA: f32 = 0.25;
const GRAD: f32 = 1.0;
const PROBE: &str = "NSL_LCE_GIF_PROBE";

fn data(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (((s >> 40) as f32 / 8_388_608.0) - 1.0) * amp
        })
        .collect()
}

fn gpu_2d(rows: usize, cols: usize, v: &[f32]) -> i64 {
    let t = test_build_tensor_2d_f32(rows, cols, v);
    let g = nsl_tensor_to_device(t, 1);
    if g != t {
        nsl_tensor_free(t);
    }
    g
}

fn read(t: i64) -> Vec<f64> {
    let c = nsl_tensor_to_device(t, 0);
    let out = test_read_tensor_f32(c).into_iter().map(f64::from).collect();
    if c != t {
        nsl_tensor_free(c);
    }
    out
}

fn round_bf16(v: &[f32]) -> Vec<f64> {
    v.iter()
        .map(|&x| f64::from(bf16::from_f32(x).to_f32()))
        .collect()
}

fn targets() -> Vec<i64> {
    let mut s = 99u64;
    (0..ROWS)
        .map(|r| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            if r % 7 == 3 {
                -100
            } else {
                ((s >> 33) % V as u64) as i64
            }
        })
        .collect()
}

struct Reference {
    loss: Vec<f64>,
    dx: Vec<f64>,
    /// `alpha · dlogitsᵀ x`, the accumulator's change.
    ddw: Vec<f64>,
}

fn reference(x: &[f64], w: &[f64], t: &[i64]) -> Reference {
    let valid = t.iter().filter(|&&v| v >= 0).count() as f64;
    let scale = f64::from(GRAD) / valid;
    let mut loss = vec![0.0; ROWS];
    let mut dx = vec![0.0; ROWS * H];
    let mut ddw = vec![0.0; V * H];
    let mut z = vec![0.0; V];
    for r in 0..ROWS {
        let xr = &x[r * H..(r + 1) * H];
        for (j, zj) in z.iter_mut().enumerate() {
            *zj = xr
                .iter()
                .zip(&w[j * H..(j + 1) * H])
                .map(|(a, b)| a * b)
                .sum();
        }
        if t[r] < 0 {
            continue;
        }
        let m = z.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let lse = m + z.iter().map(|&v| (v - m).exp()).sum::<f64>().ln();
        loss[r] = lse - z[t[r] as usize];
        for j in 0..V {
            let g = ((z[j] - lse).exp() - if j as i64 == t[r] { 1.0 } else { 0.0 }) * scale;
            for k in 0..H {
                dx[r * H + k] += g * w[j * H + k];
                ddw[j * H + k] += f64::from(ALPHA) * g * xr[k];
            }
        }
    }
    Reference { loss, dx, ddw }
}

/// `‖got - want‖ / ‖want‖`.
fn rel(got: &[f64], want: &[f64]) -> (f64, f64) {
    let err: f64 = got
        .iter()
        .zip(want)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f64>()
        .sqrt();
    let norm: f64 = want.iter().map(|b| b * b).sum::<f64>().sqrt();
    (err / norm, norm)
}

/// One call of the FFI on fresh outputs and an accumulator holding `dw0`;
/// `with_dx = false` passes no x gradient (its buffer stays zero).
fn call(x: i64, w: i64, tgt: i64, dw0: &[f32], with_dx: bool) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let loss = gpu_2d(1, ROWS, &vec![0.0; ROWS]);
    let dx = gpu_2d(ROWS, H, &vec![0.0; ROWS * H]);
    let acc = gpu_2d(V, H, dw0);
    let rc = nsl_fused_linear_ce_forward_gemm_grad(
        nsl_tensor_data_ptr(x),
        nsl_tensor_data_ptr(w),
        tgt,
        nsl_tensor_data_ptr(loss),
        if with_dx { nsl_tensor_data_ptr(dx) } else { 0 },
        acc,
        1,
        ROWS as i64,
        V as i64,
        H as i64,
        targets().iter().filter(|&&v| v >= 0).count() as i64,
        i64::from(GRAD.to_bits()),
        i64::from(ALPHA.to_bits()),
    );
    assert_eq!(rc, 0);
    let out = (read(loss), read(dx), read(acc));
    for t in [loss, dx, acc] {
        nsl_tensor_free(t);
    }
    out
}

fn check(route: &str, got: (Vec<f64>, Vec<f64>, Vec<f64>), want: &Reference, dw0: &[f32]) {
    let (loss, dx, acc) = got;
    let worst = loss
        .iter()
        .zip(&want.loss)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(worst <= 2e-2, "{route}: a row's loss is off by {worst}");
    for (r, (a, b)) in loss.iter().zip(&want.loss).enumerate() {
        if targets()[r] < 0 {
            assert_eq!(*a, 0.0, "{route}: ignored row {r} has loss {a}");
        } else {
            assert!(
                *b > 1.0,
                "the reference loss of row {r} is {b}: the fixture is too easy"
            );
        }
    }
    let ddw: Vec<f64> = acc
        .iter()
        .zip(dw0)
        .map(|(a, &b)| a - f64::from(b))
        .collect();
    for (what, g, w) in [("dx", &dx, &want.dx), ("dW", &ddw, &want.ddw)] {
        let (e, norm) = rel(g, w);
        eprintln!("{route}: {what} relative error {e:.3e} (reference norm {norm:.3e})");
        assert!(
            e <= 2e-2,
            "{route}: {what} is {e:.3e} away from the reference"
        );
        assert!(norm > 0.0, "{route}: the {what} reference is zero");
    }
}

#[test]
#[ignore = "spawned by both_routes_match_the_f64_reference_gpu"]
fn zz_lce_gif_probe_child() {
    if std::env::var(PROBE).is_err() {
        return;
    }
    let (xh, wh) = (data(ROWS * H, 3, 0.5), data(V * H, 5, 0.25));
    let want = reference(&round_bf16(&xh), &round_bf16(&wh), &targets());
    let (x, w) = (gpu_2d(ROWS, H, &xh), gpu_2d(V, H, &wh));
    let t = targets();
    let tgt = nsl_test_cuda_alloc((ROWS * 8) as i64);
    nsl_test_cuda_h2d(tgt, t.as_ptr() as i64, (ROWS * 8) as i64);
    let dw0 = data(V * H, 7, 0.01);

    // The weight has no bf16 image: the vocabulary-chunked route.
    let (r6, r7) = (nsl_fused_lce_launch_count(6), nsl_fused_lce_launch_count(7));
    let got = call(x, w, tgt, &dw0, true);
    assert_eq!(
        (
            nsl_fused_lce_launch_count(6) - r6,
            nsl_fused_lce_launch_count(7) - r7
        ),
        (0, 1),
        "an unregistered weight must take the vocabulary-chunked route"
    );
    check("vocabulary-chunked", got, &want, &dw0);

    // Registered (a zero-lr fused AdamW step, which moves no weight bit): the
    // row-chunked route.
    let zeros = || gpu_2d(V, H, &vec![0.0; V * H]);
    let (m, v, mp) = (zeros(), zeros(), zeros());
    nsl_fase_fused_adamw_step(
        w, m, v, mp, 0.0, 0.9, 0.1, 0.999, 0.001, 1e-8, 0.0, 1.0, 1.0,
    );
    let got = call(x, w, tgt, &dw0, true);
    assert_eq!(
        (
            nsl_fused_lce_launch_count(6) - r6,
            nsl_fused_lce_launch_count(7) - r7
        ),
        (1, 1),
        "a registered weight in the BF16 mode must take the row-chunked route"
    );
    check("row-chunked", got, &want, &dw0);

    // No x gradient asked for: the dx GEMMs are skipped, the rest is the same.
    let (loss, dx, acc) = call(x, w, tgt, &dw0, false);
    assert!(
        dx.iter().all(|&v| v == 0.0),
        "a call without dx wrote the dx buffer"
    );
    check(
        "row-chunked, no dx",
        (loss, want.dx.clone(), acc),
        &want,
        &dw0,
    );

    for t in [x, w, m, v, mp] {
        nsl_tensor_free(t);
    }
    nsl_test_cuda_free(tgt);
    println!("LCE-GIF-PROBE-OK");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn both_routes_match_the_f64_reference_gpu() {
    if std::env::var(PROBE).is_ok() {
        return;
    }
    let exe = std::env::current_exe().expect("current_exe");
    let out = std::process::Command::new(exe)
        .args([
            "--test-threads=1",
            "--nocapture",
            "--include-ignored",
            "zz_lce_gif_probe_child",
        ])
        .env(PROBE, "1")
        .env("NSL_MATMUL_BF16", "1")
        .env("NSL_LCE_ROW_CHUNK", "96")
        .env_remove("NSL_MATMUL_BF16_CAST_CACHE")
        .env_remove("NSL_MATMUL_BF16_ROUND")
        .env_remove("NSL_MATMUL_BF16_MIN_RATIO")
        .env_remove("NSL_MATMUL_PEDANTIC")
        .env_remove("NSL_MATMUL_TF32")
        .env_remove("NSL_ASYNC_ALLOC")
        .output()
        .expect("spawn probe child");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        out.status.success() && stdout.contains("LCE-GIF-PROBE-OK"),
        "probe child failed (status {:?})\n--- stdout ---\n{stdout}\n--- stderr ---\n{stderr}",
        out.status
    );
    eprintln!("{stderr}");
}
