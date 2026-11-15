//! bf16-storage GEMMs (`nsl_tensor_matmul_dt`, and `nsl_tensor_wgrad_accum`
//! with bf16 activations) against an f64 reference, on the GPU.
//!
//! Inputs are bf16 values (an f32 operand is rounded to bf16 by the GEMM path
//! before use, so the reference uses its bf16 rounding too). The bound per
//! output is derived, not tuned: k products, each exact in f32 (8 x 8 bits),
//! accumulated in f32 in an order and rounding mode cuBLAS does not document
//! (published measurements of earlier tensor cores found truncation, not
//! round-to-nearest), so |y - y*| <= k * 2^-23 * sum_l |a_l b_l|: one f32 ulp
//! of the running magnitude per add, either rounding. The output must lie
//! between round(y* - E) and round(y* + E) in its storage dtype. With
//! non-negative operands sum |ab| = y*, so E <= k * 2^-23 * y* (2^-12 of y* at
//! k = 2048), far below bf16's rounding spacing: a dropped k-tile, a wrong
//! transposition or a mis-read operand cannot fit.

#![cfg(feature = "cuda")]

use nsl_runtime::nsl_cuda_init;

unsafe extern "C" {
    fn nsl_tensor_from_static(data_ptr: i64, shape_list: i64, dtype: i64) -> i64;
    fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64;
    fn nsl_tensor_transpose(tensor_ptr: i64, dim0: i64, dim1: i64) -> i64;
    fn nsl_tensor_free(tensor_ptr: i64);
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
    fn nsl_list_free(list_ptr: i64);
    fn nsl_tensor_matmul_dt(a: i64, b: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_wgrad_accum(m_ptr: i64, x_ptr: i64, g_ptr: i64, s: f64);
    fn nsl_wgrad_fused_count() -> i64;
    fn nsl_storage_dt_fallback_count() -> i64;
}

const F32: i64 = 1;
const BF16: i64 = 3;
const U23: f64 = 1.192_092_895_507_812_5e-7; // 2^-23
const U24: f64 = 5.960_464_477_539_063e-8; // 2^-24

#[repr(C)]
struct TensorView {
    _magic: u32,
    data: *mut std::ffi::c_void,
    _shape: *mut i64,
    _strides: *mut i64,
    _ndim: i64,
    len: i64,
    _refcount: std::sync::atomic::AtomicI64,
    device: u8,
    dtype: u16,
    _owns_data: u8,
    _data_owner: i64,
    _slab_managed: u8,
    _tape_id: i64,
}

fn view(p: i64) -> &'static TensorView {
    unsafe { &*(p as *const TensorView) }
}

fn init() {
    assert_eq!(nsl_cuda_init(), 0, "nsl_cuda_init failed: these gates run where a CUDA device is a precondition");
}

fn bf16_rne(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let a = x.abs();
    let e = (((a.to_bits() >> 52) & 0x7FF) as i32 - 1023).max(-126);
    let ulp = 2f64.powi(e - 7);
    let q = a / ulp;
    let r = q.floor();
    let frac = q - r;
    let r = if frac > 0.5 || (frac == 0.5 && (r as u64) % 2 == 1) { r + 1.0 } else { r };
    (r * ulp).copysign(x)
}

fn round_to(dtype: i64, x: f64) -> f64 {
    if dtype == BF16 { bf16_rne(x) } else { x as f32 as f64 }
}

fn upload(vals: &[f64], dims: &[i64], dtype: i64) -> (i64, Vec<f64>) {
    let list = unsafe { nsl_list_new() };
    for &d in dims {
        unsafe { nsl_list_push(list, d) };
    }
    let (ptr, stored): (i64, Vec<f64>) = if dtype == BF16 {
        let bits: Vec<u16> = vals.iter().map(|&v| ((bf16_rne(v) as f32).to_bits() >> 16) as u16).collect();
        let stored = bits.iter().map(|&b| f32::from_bits((b as u32) << 16) as f64).collect();
        (Box::leak(bits.into_boxed_slice()).as_ptr() as i64, stored)
    } else {
        let f: Vec<f32> = vals.iter().map(|&v| v as f32).collect();
        let stored = f.iter().map(|&v| v as f64).collect();
        (Box::leak(f.into_boxed_slice()).as_ptr() as i64, stored)
    };
    let cpu = unsafe { nsl_tensor_from_static(ptr, list, dtype) };
    unsafe { nsl_list_free(list) };
    let gpu = unsafe { nsl_tensor_to_device(cpu, 1) };
    assert_eq!(view(gpu).dtype as i64, dtype);
    (gpu, stored)
}

fn download(t: i64, dtype: i64) -> Vec<f64> {
    assert_eq!(view(t).device, 1, "result must be on the device");
    assert_eq!(view(t).dtype as i64, dtype, "result dtype tag");
    let host = unsafe { nsl_tensor_to_device(t, 0) };
    let n = view(host).len as usize;
    let out = if dtype == BF16 {
        let d = unsafe { std::slice::from_raw_parts(view(host).data as *const u16, n) };
        d.iter().map(|&b| f32::from_bits((b as u32) << 16) as f64).collect()
    } else {
        let d = unsafe { std::slice::from_raw_parts(view(host).data as *const f32, n) };
        d.iter().map(|&v| v as f64).collect()
    };
    unsafe { nsl_tensor_free(host) };
    out
}

/// Deterministic values: magnitudes 10^lo..10^hi, `signed` alternates signs.
fn values(n: usize, seed: u64, lo: f64, hi: f64, signed: bool) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|k| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            let v = 10f64.powf(lo + (hi - lo) * u);
            if signed && k % 3 == 1 { -v } else { v }
        })
        .collect()
}

/// Row-major `a[m, k] @ b[k, n]` checked at a deterministic sample of
/// `(i, j)` (every output when `stride == 1`): returns the failures.
#[allow(clippy::too_many_arguments)]
fn check(
    what: &str,
    out: &[f64],
    a: &[f64],
    b: &[f64],
    m: usize,
    k: usize,
    n: usize,
    out_dtype: i64,
    stride: usize,
) {
    let mut bad = Vec::new();
    let mut checked = 0;
    for i in (0..m).step_by(stride) {
        for j in (0..n).step_by(stride.max(1)) {
            let (mut y, mut mag) = (0.0f64, 0.0f64);
            for l in 0..k {
                let p = a[i * k + l] * b[l * n + j];
                y += p;
                mag += p.abs();
            }
            let e = k as f64 * U23 * mag;
            let (lo, hi) = (round_to(out_dtype, y - e), round_to(out_dtype, y + e));
            let o = out[i * n + j];
            if !(lo <= o && o <= hi) {
                bad.push(format!("[{i},{j}] got {o:e}, exact {y:e}, bracket [{lo:e}, {hi:e}]"));
            }
            checked += 1;
        }
    }
    assert!(checked > 100, "{what}: only {checked} outputs checked");
    assert!(bad.is_empty(), "{what}: {} of {checked} outside the bracket:\n{}", bad.len(), bad[..bad.len().min(10)].join("\n"));
}

fn no_fallback<T>(f: impl FnOnce() -> T) -> T {
    let before = unsafe { nsl_storage_dt_fallback_count() };
    let r = f();
    assert_eq!(unsafe { nsl_storage_dt_fallback_count() }, before, "the GEMM took the widen/compute/narrow fallback");
    r
}

#[test]
#[ignore = "requires CUDA GPU"]
fn bf16_operand_gemms_round_once_to_the_requested_dtype() {
    init();
    // (m, k, n, check stride): a GemmEx-sized shape with every tile edge
    // ragged, and one large enough for the timed cublasLt plan (>= 2^28 MACs).
    for (m, k, n, stride) in [(300usize, 520usize, 700usize, 1usize), (768, 1024, 512, 7)] {
        for signed in [false, true] {
            let a_vals = values(m * k, 1 + m as u64, -1.5, 0.0, signed);
            let b_vals = values(k * n, 2 + n as u64, -1.5, 0.0, signed);
            for out in [BF16, F32] {
                // 2-D, both bf16.
                let (a, av) = upload(&a_vals, &[m as i64, k as i64], BF16);
                let (b, bv) = upload(&b_vals, &[k as i64, n as i64], BF16);
                let c = no_fallback(|| unsafe { nsl_tensor_matmul_dt(a, b, out) });
                check(&format!("2-D {m}x{k}x{n} signed={signed} -> {out}"), &download(c, out), &av, &bv, m, k, n, out, stride);
                unsafe { nsl_tensor_free(c) };

                // Batch-collapsed A [2, m/2, k] @ B.
                let (a3, _) = upload(&a_vals, &[2, (m / 2) as i64, k as i64], BF16);
                let c3 = no_fallback(|| unsafe { nsl_tensor_matmul_dt(a3, b, out) });
                check(&format!("collapsed signed={signed} -> {out}"), &download(c3, out), &av, &bv, m, k, n, out, stride);
                unsafe { nsl_tensor_free(c3) };
                unsafe { nsl_tensor_free(a3) };

                // B as the transposed view of a stored [n, k] (dgrad's W^T).
                let bt_vals: Vec<f64> = (0..n * k).map(|idx| bv[(idx % k) * n + idx / k]).collect();
                let (bs, _) = upload(&bt_vals, &[n as i64, k as i64], BF16);
                let bview = unsafe { nsl_tensor_transpose(bs, 0, 1) };
                let ct = no_fallback(|| unsafe { nsl_tensor_matmul_dt(a, bview, out) });
                check(&format!("transposed-B signed={signed} -> {out}"), &download(ct, out), &av, &bv, m, k, n, out, stride);
                for t in [ct, bview, bs] {
                    unsafe { nsl_tensor_free(t) };
                }

                // Mixed: an f32 B, rounded to bf16 by the GEMM path.
                let (bf, bfv) = upload(&b_vals, &[k as i64, n as i64], F32);
                let b_rounded: Vec<f64> = bfv.iter().map(|&v| bf16_rne(v)).collect();
                let cm = no_fallback(|| unsafe { nsl_tensor_matmul_dt(a, bf, out) });
                check(&format!("bf16 x f32 signed={signed} -> {out}"), &download(cm, out), &av, &b_rounded, m, k, n, out, stride);
                for t in [cm, bf, a, b] {
                    unsafe { nsl_tensor_free(t) };
                }
            }
        }
    }
}

/// `m += s * x^T g` with bf16 x and g and an f32 m, through the fused path.
/// Bound: the GEMM's |s| * N * 2^-23 * sum |x g| plus the epilogue's
/// `alpha * acc + beta * m` in f32 (two roundings of magnitude at most
/// |s * acc| + |m|).
#[test]
#[ignore = "requires CUDA GPU"]
fn the_fused_weight_gradient_accumulates_bf16_activations_in_f32() {
    init();
    let (batch, rows, d_in, d_out) = (2usize, 300usize, 96usize, 80usize);
    let n = batch * rows;
    let s = 0.25f64;
    let x_vals = values(n * d_in, 3, -2.0, 0.0, true);
    let g_vals = values(n * d_out, 4, -7.0, -3.0, true); // gradient magnitudes
    let m_vals = values(d_in * d_out, 5, -4.0, -2.0, true);
    let (x, xv) = upload(&x_vals, &[batch as i64, rows as i64, d_in as i64], BF16);
    let (g, gv) = upload(&g_vals, &[batch as i64, rows as i64, d_out as i64], BF16);
    let (m, mv) = upload(&m_vals, &[d_in as i64, d_out as i64], F32);
    let fused_before = unsafe { nsl_wgrad_fused_count() };
    unsafe { nsl_tensor_wgrad_accum(m, x, g, s) };
    assert_eq!(unsafe { nsl_wgrad_fused_count() }, fused_before + 1, "bf16 activations must take the fused GEMM");
    let out = download(m, F32);
    let mut bad = Vec::new();
    for i in 0..d_in {
        for j in 0..d_out {
            let (mut acc, mut mag) = (0.0f64, 0.0f64);
            for r in 0..n {
                let p = xv[r * d_in + i] * gv[r * d_out + j];
                acc += p;
                mag += p.abs();
            }
            let y = mv[i * d_out + j] + s * acc;
            let e = s.abs() * n as f64 * U23 * mag + 2.0 * U24 * ((s * acc).abs() + mv[i * d_out + j].abs());
            let (lo, hi) = (round_to(F32, y - e), round_to(F32, y + e));
            let o = out[i * d_out + j];
            if !(lo <= o && o <= hi) {
                bad.push(format!("[{i},{j}] got {o:e}, exact {y:e}, bracket [{lo:e}, {hi:e}]"));
            }
        }
    }
    assert!(bad.is_empty(), "{} of {} outside the bracket:\n{}", bad.len(), d_in * d_out, bad[..bad.len().min(10)].join("\n"));
    for t in [x, g, m] {
        unsafe { nsl_tensor_free(t) };
    }
}

/// A batched B is not one 2-D GEMM: the entry point widens, runs the f32
/// matmul and narrows, counted, within the same bound.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_shape_one_gemm_cannot_express_takes_the_counted_fallback() {
    init();
    let (m, k, n) = (64usize, 96usize, 48usize);
    let a_vals = values(m * k, 6, -1.0, 0.0, true);
    let b_vals = values(k * n, 7, -1.0, 0.0, true);
    let (a, av) = upload(&a_vals, &[1, m as i64, k as i64], BF16);
    let (b, bv) = upload(&b_vals, &[1, k as i64, n as i64], BF16);
    let before = unsafe { nsl_storage_dt_fallback_count() };
    let c = unsafe { nsl_tensor_matmul_dt(a, b, BF16) };
    assert_eq!(unsafe { nsl_storage_dt_fallback_count() }, before + 1, "a 3-D B must take the counted fallback");
    check("fallback 3-D B -> bf16", &download(c, BF16), &av, &bv, m, k, n, BF16, 1);
    for t in [a, b, c] {
        unsafe { nsl_tensor_free(t) };
    }
}
