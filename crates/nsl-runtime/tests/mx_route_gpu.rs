//! Silicon gate for the MXFP8 projection GEMMs (`NSL_MXFP8_GEMMS=1`'s runtime
//! entry points), each against the f64 product of the operands the host
//! reference quantizes (`nsl_runtime::mxfp8::reference`), in the orientation
//! the routing claims:
//!
//! 1. `nsl_tensor_matmul_mx(x, W)`, `W [k, n]` contiguous: the forward
//!    (`W` quantized down its columns, `x` by rows), f32 and bf16 outputs;
//! 2. `nsl_tensor_matmul_mx(g, W^T)`, `W [n, k]` seen through its transpose:
//!    the input gradient (`W` by rows);
//! 3. `nsl_tensor_matmul_acc(g, W^T, c, 4)`: the same plus `c`, out of place;
//! 4. `nsl_tensor_wgrad_accum_mx(m, x, g, s)`: `m += s · x^T g`, both
//!    operands quantized down their token columns, twice (accumulating); onto
//!    a fresh accumulator full of stale values (`beta = 0` overwrites them);
//!    and a width outside the contract, which declines to the bf16 GEMM;
//! 5. a registered parameter (the bf16 matmul mode, as `precision="bf16"`
//!    runs): one MXFP8 image per K direction, quantized once and then reused,
//!    quantized again after an optimizer step, and dropped by a write to the
//!    weight, after which the product follows the new bytes.
//!
//! Every shape is non-square, so an operand taken in the wrong orientation
//! fails to multiply or lands far from the reference. A shape outside the
//! MXFP8 contract (`k` not a multiple of 32) takes the bf16 path and is
//! counted as declined.
//!
//!   cargo test -p nsl-runtime --features cuda,test-hooks --test mx_route_gpu \
//!       -- --include-ignored --test-threads=1
#![cfg(all(feature = "cuda", feature = "test-hooks"))]

use nsl_runtime::list::{nsl_list_free, nsl_list_new, nsl_list_push};
use nsl_runtime::mxfp8::reference;
use nsl_runtime::storage_dt::{nsl_tensor_matmul_acc, nsl_tensor_matmul_mx};
use nsl_runtime::tensor::arithmetic::nsl_tensor_wgrad_accum_mx;
use nsl_runtime::tensor::precision_cast::nsl_tensor_to_dtype;
use nsl_runtime::tensor::{
    nsl_tensor_data_ptr, nsl_tensor_free, nsl_tensor_from_static, nsl_tensor_get_dtype,
    nsl_tensor_to_device, nsl_tensor_transpose,
};

const DTYPE_F32: i64 = 1;
const DTYPE_BF16: i64 = 3;

static SERIAL: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn ready() -> bool {
    // The bf16 matmul mode, as `precision="bf16"` programs run: only in it does
    // the optimizer's step register parameters (and so their weight images).
    // Read once by the runtime; every gate sets it before its first GEMM.
    // SAFETY: tests here run one at a time (SERIAL) and no thread reads the
    // environment concurrently.
    unsafe { std::env::set_var("NSL_MATMUL_BF16", "1") };
    if std::env::var("NSL_SKIP_CUDA_TESTS").is_ok() || nsl_runtime::nsl_cuda_init() != 0 {
        return false;
    }
    if !nsl_runtime::mxfp8::supported() {
        eprintln!("[mx_route_gpu] skipped: MXFP8 needs compute capability 10.0 or newer");
        return false;
    }
    true
}

fn upload(shape: &[usize], vals: &[f32]) -> i64 {
    assert_eq!(shape.iter().product::<usize>(), vals.len());
    let leaked: &'static [f32] = Box::leak(vals.to_vec().into_boxed_slice());
    let s = nsl_list_new();
    for &d in shape {
        nsl_list_push(s, d as i64);
    }
    let cpu = nsl_tensor_from_static(leaked.as_ptr() as i64, s, DTYPE_F32);
    nsl_list_free(s);
    let dev = nsl_tensor_to_device(cpu, 1);
    nsl_tensor_free(cpu);
    dev
}

/// The tensor's values as f32 (a bf16 tensor is widened first).
fn download(t: i64, n: usize) -> Vec<f32> {
    let t32 = if nsl_tensor_get_dtype(t) == DTYPE_F32 {
        t
    } else {
        nsl_tensor_to_dtype(t, DTYPE_F32)
    };
    let cpu = nsl_tensor_to_device(t32, 0);
    let p = nsl_tensor_data_ptr(cpu);
    let out = unsafe { std::slice::from_raw_parts(p as *const f32, n) }.to_vec();
    nsl_tensor_free(cpu);
    if t32 != t {
        nsl_tensor_free(t32);
    }
    out
}

/// Values with a different magnitude in every 32-element run, so a scale
/// read from the wrong block shows.
fn values(n: usize, salt: u64) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let block = (i / 32) as u64;
            let e = ((block * 7 + salt) % 9) as i32 - 4;
            let u = ((i as f64) * 0.618_033_988_749_894_9 + salt as f64 * 0.1).fract() * 2.0 - 1.0;
            (u as f32) * 2f32.powi(e)
        })
        .collect()
}

/// `(sum_k a(i,k)·b(j,k), sum |·|)` over two quantized `[outer, k]` operands.
fn dot(a: &(Vec<u8>, Vec<u8>), b: &(Vec<u8>, Vec<u8>), k: usize, i: usize, j: usize) -> (f64, f64) {
    let (mut s, mut mag) = (0f64, 0f64);
    for kk in 0..k {
        let p = reference::dequant(&a.0, &a.1, k, i, kk) * reference::dequant(&b.0, &b.1, k, j, kk);
        s += p;
        mag += p.abs();
    }
    (s, mag)
}

fn close(tag: &str, got: f32, want: f64, mag: f64, rel: f64) {
    let err = (f64::from(got) - want).abs();
    assert!(
        err <= rel * mag + 1e-30,
        "{tag}: got {got}, want {want} (|err| {err:e}, scale {mag:e})"
    );
}

fn mx_counts() -> [u64; 5] {
    nsl_runtime::mxfp8::route_counts()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_forward_multiplies_x_by_w() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    // x [m, k] @ W [k, n].
    let (m, k, n) = (64, 96, 160);
    let (xv, wv) = (values(m * k, 1), values(k * n, 2));
    let (x, w) = (upload(&[m, k], &xv), upload(&[k, n], &wv));
    let xq = reference::quant_rows(&xv, m, k);
    let wq = reference::quant_cols_t(&wv, k, n);
    for (tag, dt) in [("f32", DTYPE_F32), ("bf16", DTYPE_BF16)] {
        let before = mx_counts();
        let y = nsl_tensor_matmul_mx(x, w, dt);
        assert_eq!(
            mx_counts()[0],
            before[0] + 1,
            "{tag}: the forward did not run MXFP8"
        );
        let got = download(y, m * n);
        let rel = if dt == DTYPE_F32 { 1e-5 } else { 2f64.powi(-8) };
        for i in 0..m {
            for j in 0..n {
                let (want, mag) = dot(&wq, &xq, k, j, i);
                close(
                    &format!("{tag} y[{i},{j}]"),
                    got[i * n + j],
                    want,
                    mag + want.abs(),
                    rel,
                );
            }
        }
        nsl_tensor_free(y);
    }
    for t in [x, w] {
        nsl_tensor_free(t);
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_input_gradient_multiplies_g_by_w_transposed() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    // g [m, k] @ W^T, W [n, k] stored contiguous, seen as a [k, n] view.
    let (m, k, n) = (64, 160, 96);
    let (gv, wv) = (values(m * k, 3), values(n * k, 4));
    let (g, w) = (upload(&[m, k], &gv), upload(&[n, k], &wv));
    let wt = nsl_tensor_transpose(w, 0, 1);
    let gq = reference::quant_rows(&gv, m, k);
    let wq = reference::quant_rows(&wv, n, k);
    let before = mx_counts();
    let dx = nsl_tensor_matmul_mx(g, wt, DTYPE_F32);
    assert_eq!(
        mx_counts()[1],
        before[1] + 1,
        "the input gradient did not run MXFP8"
    );
    let got = download(dx, m * n);
    let mut refs = vec![(0f64, 0f64); m * n];
    for i in 0..m {
        for j in 0..n {
            refs[i * n + j] = dot(&wq, &gq, k, j, i);
            let (want, mag) = refs[i * n + j];
            close(&format!("dx[{i},{j}]"), got[i * n + j], want, mag, 1e-5);
        }
    }
    // Route 4: the same product plus c, into a new tensor.
    let cv = values(m * n, 5);
    let c = upload(&[m, n], &cv);
    let before = mx_counts();
    let sum = nsl_tensor_matmul_acc(g, wt, c, 4);
    assert_eq!(
        mx_counts()[2],
        before[2] + 1,
        "the summed input gradient did not run MXFP8"
    );
    let got = download(sum, m * n);
    for at in 0..m * n {
        let (want, mag) = refs[at];
        close(
            &format!("c + dx[{at}]"),
            got[at],
            want + f64::from(cv[at]),
            mag + f64::from(cv[at]).abs(),
            1e-5,
        );
    }
    // c itself is untouched (out of place).
    assert_eq!(download(c, m * n), cv, "route 4 wrote into c");
    for t in [sum, c, dx, wt, g, w] {
        nsl_tensor_free(t);
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_weight_gradient_accumulates_x_transposed_times_g() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    // m [d_in, d_out] += s · x^T g, x [t, d_in], g [t, d_out]; twice.
    let (t, d_in, d_out) = (96, 64, 160);
    let (xv, gv) = (values(t * d_in, 6), values(t * d_out, 7));
    let (x, g) = (upload(&[t, d_in], &xv), upload(&[t, d_out], &gv));
    let acc = upload(&[d_in, d_out], &vec![0.0; d_in * d_out]);
    let xq = reference::quant_cols_t(&xv, t, d_in);
    let gq = reference::quant_cols_t(&gv, t, d_out);
    let s = 0.25;
    let before = mx_counts();
    nsl_tensor_wgrad_accum_mx(acc, x, g, s);
    nsl_tensor_wgrad_accum_mx(acc, x, g, s);
    assert_eq!(
        mx_counts()[3],
        before[3] + 2,
        "the weight gradient did not run MXFP8"
    );
    let got = download(acc, d_in * d_out);
    for i in 0..d_in {
        for o in 0..d_out {
            let (want, mag) = dot(&gq, &xq, t, o, i);
            close(
                &format!("dW[{i},{o}]"),
                got[i * d_out + o],
                2.0 * s * want,
                2.0 * s * mag,
                1e-5,
            );
        }
    }
    for tt in [acc, x, g] {
        nsl_tensor_free(tt);
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_shape_outside_the_contract_takes_bf16() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    // k = 80 is not a multiple of 32.
    let (m, k, n) = (32, 80, 64);
    let (x, w) = (
        upload(&[m, k], &values(m * k, 8)),
        upload(&[k, n], &values(k * n, 9)),
    );
    let before = mx_counts();
    let y = nsl_tensor_matmul_mx(x, w, DTYPE_F32);
    let after = mx_counts();
    assert_eq!(after[4], before[4] + 1, "the decline was not counted");
    assert_eq!(after[0], before[0], "an out-of-contract shape ran MXFP8");
    assert!(download(y, m * n).iter().all(|v| v.is_finite()));
    for tt in [y, x, w] {
        nsl_tensor_free(tt);
    }
}

/// A buffer of stale values, marked fresh as the AdamW step leaves an
/// accumulator: zero in value, its bytes not.
fn fresh_stale(shape: &[usize]) -> i64 {
    let n = shape.iter().product();
    let t = upload(
        shape,
        &(0..n).map(|i| 7.0 + i as f32 * 1e-3).collect::<Vec<_>>(),
    );
    let l = nsl_list_new();
    nsl_list_push(l, t);
    nsl_runtime::accum_fresh::nsl_accum_mark_fresh(l);
    nsl_list_free(l);
    t
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_fresh_accumulator_is_overwritten() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    let (t, d_in, d_out) = (64, 96, 128);
    let (xv, gv) = (values(t * d_in, 10), values(t * d_out, 11));
    let (x, g) = (upload(&[t, d_in], &xv), upload(&[t, d_out], &gv));
    let acc = fresh_stale(&[d_in, d_out]);
    let (xq, gq) = (
        reference::quant_cols_t(&xv, t, d_in),
        reference::quant_cols_t(&gv, t, d_out),
    );
    let before = mx_counts();
    nsl_tensor_wgrad_accum_mx(acc, x, g, 1.0);
    assert_eq!(
        mx_counts()[3],
        before[3] + 1,
        "the weight gradient did not run MXFP8"
    );
    let got = download(acc, d_in * d_out);
    for i in 0..d_in {
        for o in 0..d_out {
            let (want, mag) = dot(&gq, &xq, t, o, i);
            close(&format!("dW[{i},{o}]"), got[i * d_out + o], want, mag, 1e-5);
        }
    }
    for tt in [acc, x, g] {
        nsl_tensor_free(tt);
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_weight_gradient_width_outside_the_contract_takes_bf16() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    // d_out = 48 is not a multiple of 32: the transposing quantizer cannot tile g.
    let (t, d_in, d_out) = (64, 96, 48);
    let (xv, gv) = (values(t * d_in, 12), values(t * d_out, 13));
    let (x, g) = (upload(&[t, d_in], &xv), upload(&[t, d_out], &gv));
    // Fresh, so the bf16 fallback must take the mark the MXFP8 route left.
    let acc = fresh_stale(&[d_in, d_out]);
    let before = mx_counts();
    nsl_tensor_wgrad_accum_mx(acc, x, g, 1.0);
    let after = mx_counts();
    assert_eq!(after[4], before[4] + 1, "the decline was not counted");
    assert_eq!(after[3], before[3], "an out-of-contract width ran MXFP8");
    let got = download(acc, d_in * d_out);
    for i in 0..d_in {
        for o in 0..d_out {
            let (mut want, mut mag) = (0f64, 0f64);
            for tt in 0..t {
                let p = f64::from(xv[tt * d_in + i]) * f64::from(gv[tt * d_out + o]);
                want += p;
                mag += p.abs();
            }
            // bf16 operands: a relative 2^-8 per factor.
            close(
                &format!("dW[{i},{o}]"),
                got[i * d_out + o],
                want,
                mag,
                2f64.powi(-7),
            );
        }
    }
    for tt in [acc, x, g] {
        nsl_tensor_free(tt);
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn registered_weights_are_quantized_once_per_step_and_follow_every_write() {
    let _s = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
    if !ready() {
        return;
    }
    use nsl_runtime::mxfp8::{note_param_stepped_for_test, weight_image_counts};
    let (m, k, n) = (64, 96, 128);
    let xv = values(m * k, 14);
    let mut wv = values(k * n, 16);
    let (x, w) = (upload(&[m, k], &xv), upload(&[k, n], &wv));
    // The input gradient g [m, n] @ w^T, w [k, n] seen as its [n, k] transpose.
    let gn = values(m * n, 17);
    let g = upload(&[m, n], &gn);
    let wt = nsl_tensor_transpose(w, 0, 1);
    note_param_stepped_for_test(w);

    // The forward: x [m, k] @ w [k, n], against the reference of the current w.
    let check_fwd = |wv: &[f32], tag: &str| {
        let y = nsl_tensor_matmul_mx(x, w, DTYPE_F32);
        let got = download(y, m * n);
        let (xq, wq) = (
            reference::quant_rows(&xv, m, k),
            reference::quant_cols_t(wv, k, n),
        );
        for i in 0..m {
            for j in 0..n {
                let (want, mag) = dot(&wq, &xq, k, j, i);
                close(
                    &format!("{tag} y[{i},{j}]"),
                    got[i * n + j],
                    want,
                    mag,
                    1e-5,
                );
            }
        }
        nsl_tensor_free(y);
    };
    let (h0, q0) = weight_image_counts();
    check_fwd(&wv, "first");
    check_fwd(&wv, "second");
    let (h1, q1) = weight_image_counts();
    assert_eq!(
        (h1 - h0, q1 - q0),
        (1, 1),
        "the forward image: quantized once, then reused"
    );

    // The input gradient reads the same parameter by rows: its own image.
    let dx = nsl_tensor_matmul_mx(g, wt, DTYPE_F32);
    let got = download(dx, m * k);
    let (gq, wr) = (
        reference::quant_rows(&gn, m, n),
        reference::quant_rows(&wv, k, n),
    );
    for i in 0..m {
        for j in 0..k {
            let (want, mag) = dot(&wr, &gq, n, j, i);
            close(&format!("dx[{i},{j}]"), got[i * k + j], want, mag, 1e-5);
        }
    }
    nsl_tensor_free(dx);
    let (h2, q2) = weight_image_counts();
    assert_eq!(
        (h2 - h1, q2 - q1),
        (0, 1),
        "the input-gradient image is the entry's second"
    );

    // An optimizer step: the next forward quantizes again.
    note_param_stepped_for_test(w);
    check_fwd(&wv, "after a step");
    let (h3, q3) = weight_image_counts();
    assert_eq!(
        (h3 - h2, q3 - q2),
        (0, 1),
        "a stepped parameter's image was reused"
    );

    // A write to the weight drops its images; the product follows the new bytes.
    for v in wv.iter_mut() {
        *v = -*v * 1.5;
    }
    nsl_runtime::nsl_test_cuda_h2d(
        nsl_tensor_data_ptr(w),
        wv.as_ptr() as i64,
        (wv.len() * 4) as i64,
    );
    check_fwd(&wv, "after a write");
    let (h4, q4) = weight_image_counts();
    assert_eq!(
        (h4 - h3, q4 - q3),
        (0, 0),
        "an evicted parameter must not be served from an image"
    );
    for t in [wt, g, x, w] {
        nsl_tensor_free(t);
    }
}
