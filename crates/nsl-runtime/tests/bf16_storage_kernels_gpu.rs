//! The storage-generic elementwise kernels (bf16 activation storage) against
//! an f64 reference, on the GPU.
//!
//! **The bracket test.** Inputs are built as values exactly representable in
//! their storage dtype, so the f64 reference `y*` is computed from exactly the
//! numbers the kernel reads. The kernel computes in f32 with an error bounded
//! by `E` (derived below, per kernel, from its instruction sequence and the
//! PTX ISA's stated maxima for the approximate instructions, §9.7.3:
//! `rcp.approx.f32` 1 ulp, `ex2.approx.f32` 2 ulp), then rounds once to the
//! output dtype. So the output must lie between `round(y* - E)` and
//! `round(y* + E)` (rounding is monotone). bf16 keeps 8 significand bits, so
//! its rounding spacing is ~2^-8 of a value while `E` is ~2^-20 or less: for
//! almost every element the interval holds ONE bf16 value, and a kernel that
//! rounds twice, truncates instead of rounding to nearest, or reads an operand
//! at the wrong width fails on a large fraction of elements. No tolerance is
//! chosen by hand.
//!
//! Every fast-path call is also checked to have NOT taken the widen/compute/
//! narrow fallback (`nsl_storage_dt_fallback_count`), and its result's dtype
//! tag is checked, so a silent f32 detour cannot satisfy these tests.

#![cfg(feature = "cuda")]

use nsl_abi::wire::dt_op;
use nsl_runtime::nsl_cuda_init;

unsafe extern "C" {
    fn nsl_tensor_from_static(data_ptr: i64, shape_list: i64, dtype: i64) -> i64;
    fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64;
    fn nsl_tensor_free(tensor_ptr: i64);
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
    fn nsl_list_free(list_ptr: i64);
    fn nsl_tensor_binary_dt(op: i64, a: i64, b: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_unary_dt(op: i64, a: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_swiglu_gate_backward_dt(grad: i64, up: i64, input: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_rotate_half_dt(a: i64, neg: i64, out_dtype: i64) -> i64;
    fn nsl_storage_dt_fallback_count() -> i64;
}

const F32: i64 = 1;
const BF16: i64 = 3;
const U: f64 = 5.960_464_477_539_063e-8; // 2^-24, f32's unit roundoff

/// Mirror of `NslTensor`'s `#[repr(C)]` header (as `gpu_dtype_refusal.rs`).
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

/// `x` rounded to bf16, to nearest, ties to even, computed exactly from the
/// f64 value (no intermediate f32 rounding, which could round twice).
fn bf16_rne(x: f64) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let a = x.abs();
    // bf16: 8 significand bits, smallest normal 2^-126, subnormal step 2^-133.
    // The exponent read from the f64 bits (inputs here are normal f64s).
    let e = ((a.to_bits() >> 52) & 0x7FF) as i32 - 1023;
    let e = e.max(-126);
    let ulp = 2f64.powi(e - 7);
    let q = a / ulp;
    let r = q.floor();
    let frac = q - r;
    let r = if frac > 0.5 || (frac == 0.5 && (r as u64) % 2 == 1) { r + 1.0 } else { r };
    (r * ulp).copysign(x)
}

fn f32_rne(x: f64) -> f64 {
    x as f32 as f64 // f64 -> f32 `as` rounds to nearest, ties to even
}

fn round_to(dtype: i64, x: f64) -> f64 {
    if dtype == BF16 { bf16_rne(x) } else { f32_rne(x) }
}

fn bf16_bits(x: f64) -> u16 {
    let v = bf16_rne(x) as f32; // exact: a bf16 value is an f32 value
    (v.to_bits() >> 16) as u16
}

/// A device tensor of shape `dims` holding `vals` (each made exactly
/// representable in `dtype` first). Returns the handle and the stored values.
fn upload(vals: &[f64], dims: &[i64], dtype: i64) -> (i64, Vec<f64>) {
    let list = unsafe { nsl_list_new() };
    for &d in dims {
        unsafe { nsl_list_push(list, d) };
    }
    let (ptr, stored): (i64, Vec<f64>) = if dtype == BF16 {
        let bits: Vec<u16> = vals.iter().map(|&v| bf16_bits(v)).collect();
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
    assert_eq!(view(gpu).device, 1);
    assert_eq!(view(gpu).dtype as i64, dtype, "upload changed the dtype");
    (gpu, stored)
}

/// The values of a device tensor of `dtype`, exactly.
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

/// Deterministic values spanning `[lo_exp, hi_exp]` decades, both signs,
/// not constant along any axis.
fn values(n: usize, seed: u64, lo_exp: f64, hi_exp: f64) -> Vec<f64> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|k| {
            s = s.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            let mag = 10f64.powf(lo_exp + (hi_exp - lo_exp) * u);
            let mant = 1.0 + ((s >> 3) & 0xFF) as f64 / 256.0;
            if k % 3 == 1 { -mag * mant } else { mag * mant }
        })
        .collect()
}

/// Assert every output lies in `[round(y* - E), round(y* + E)]`.
fn bracket(what: &str, out: &[f64], want: &[f64], err: &[f64], out_dtype: i64) {
    assert_eq!(out.len(), want.len(), "{what}: length");
    let mut bad = Vec::new();
    for (i, ((&o, &y), &e)) in out.iter().zip(want).zip(err).enumerate() {
        let (lo, hi) = (round_to(out_dtype, y - e), round_to(out_dtype, y + e));
        if !(lo <= o && o <= hi) {
            bad.push(format!("[{i}] got {o:e}, exact {y:e}, bracket [{lo:e}, {hi:e}] (E {e:e})"));
        }
    }
    assert!(bad.is_empty(), "{what}: {} of {} outside the bracket:\n{}", bad.len(), out.len(), bad[..bad.len().min(12)].join("\n"));
}

/// A shape with several 256-thread blocks and a partial last one.
const DIMS: [i64; 2] = [3, 1317];
const N: usize = 3 * 1317;

fn no_fallback<T>(f: impl FnOnce() -> T) -> T {
    let before = unsafe { nsl_storage_dt_fallback_count() };
    let r = f();
    let after = unsafe { nsl_storage_dt_fallback_count() };
    assert_eq!(after, before, "a fast-path call took the widen/compute/narrow fallback");
    r
}

#[test]
#[ignore = "requires CUDA GPU"]
fn binary_ops_round_once_to_the_requested_dtype() {
    init();
    let a_vals = values(N, 1, -6.0, 2.0);
    let b_vals = values(N, 2, -6.0, 2.0);
    for (code, name) in [(dt_op::ADD, "add"), (dt_op::SUB, "sub"), (dt_op::MUL, "mul"), (dt_op::DIV, "div")] {
        for (da, db, dout) in [(BF16, BF16, BF16), (F32, BF16, F32), (BF16, F32, BF16), (BF16, BF16, F32), (F32, F32, BF16)] {
            let (a, av) = upload(&a_vals, &DIMS, da);
            let (b, bv) = upload(&b_vals, &DIMS, db);
            let c = no_fallback(|| unsafe { nsl_tensor_binary_dt(code, a, b, dout) });
            let out = download(c, dout);
            let want: Vec<f64> = av
                .iter()
                .zip(&bv)
                .map(|(&x, &y)| match code {
                    dt_op::ADD => x + y,
                    dt_op::SUB => x - y,
                    dt_op::MUL => x * y,
                    _ => x / y,
                })
                .collect();
            // add/sub/mul: one correctly rounded f32 op, half an f32 ulp.
            // div: `div.approx.f32`, 2 ulp (PTX ISA §9.7.3), i.e. 2 * 2^-23.
            let rel = if code == dt_op::DIV { 4.0 * U } else { U };
            let err: Vec<f64> = want.iter().map(|y| rel * y.abs()).collect();
            bracket(&format!("{name} {da}x{db}->{dout}"), &out, &want, &err, dout);
            for t in [a, b, c] {
                unsafe { nsl_tensor_free(t) };
            }
        }
    }
}

/// `σ(x)` as the kernels compute it: `rcp.approx(1 + ex2.approx(-x * log2e))`.
/// Relative error, first order: the `log2e` constant and the product each
/// round (2 * 2^-24 of the exponent, i.e. |x| * 2^-23 of 2^(-x log2e) after
/// `ln 2 * log2 e = 1`), `ex2.approx` 2 ulp (2^-22), the add 2^-24, `rcp`
/// 1 ulp (2^-23): (2|x| + 7) * 2^-24.
fn sigmoid_rel(x: f64) -> f64 {
    (2.0 * x.abs() + 7.0) * U
}

#[test]
#[ignore = "requires CUDA GPU"]
fn unary_ops_round_once_to_the_requested_dtype() {
    init();
    // Activations at realistic magnitudes, both signs, plus near-zero values.
    let mut x_vals = values(N, 3, -6.0, 1.45);
    x_vals[5] = 0.0;
    for (code, name) in [
        (dt_op::NEG, "neg"),
        (dt_op::RELU, "relu"),
        (dt_op::EXP, "exp"),
        (dt_op::SIGMOID, "sigmoid"),
        (dt_op::SILU, "silu"),
    ] {
        for (din, dout) in [(BF16, BF16), (F32, BF16), (BF16, F32)] {
            let (a, xv) = upload(&x_vals, &DIMS, din);
            let c = no_fallback(|| unsafe { nsl_tensor_unary_dt(code, a, dout) });
            let out = download(c, dout);
            let (want, err): (Vec<f64>, Vec<f64>) = xv
                .iter()
                .map(|&x| {
                    let s = 1.0 / (1.0 + (-x).exp());
                    match code {
                        dt_op::NEG => (-x, 0.0),
                        dt_op::RELU => (x.max(0.0), 0.0),
                        // e^x = 2^(x log2e): the constant and the product round
                        // (|x| * 2^-23 relative after the exponent), ex2 2 ulp.
                        dt_op::EXP => (x.exp(), (2.0 * x.abs() + 4.0) * U * x.exp()),
                        dt_op::SIGMOID => (s, sigmoid_rel(x) * s),
                        // x * σ(x): σ's error and one more rounding.
                        _ => (x * s, (sigmoid_rel(x) + U) * (x * s).abs()),
                    }
                })
                .unzip();
            bracket(&format!("{name} {din}->{dout}"), &out, &want, &err, dout);
            unsafe { nsl_tensor_free(a) };
            unsafe { nsl_tensor_free(c) };
        }
    }
}

/// The SwiGLU gate adjoint at realistic gradient magnitudes (1e-9..1e-3,
/// bugs.md 2026-11-12: a gradient is not an activation).
///
/// The kernel: `t = grad *rn up`; `s = σ(x)`; `tail = s *rn (1 +rn x *rn (1 -rn s))`;
/// `out = t *rn tail`. The bound propagates absolute errors to first order
/// from `σ`'s relative error (`sigmoid_rel`) and one 2^-24 rounding per op,
/// evaluated at the exact intermediates.
#[test]
#[ignore = "requires CUDA GPU"]
fn swiglu_gate_backward_rounds_once_at_gradient_magnitudes() {
    init();
    let g_vals = values(N, 4, -9.0, -3.0);
    let up_vals = values(N, 5, -3.0, 1.0);
    let x_vals = values(N, 6, -4.0, 1.3);
    for dout in [BF16, F32] {
        let (g, gv) = upload(&g_vals, &DIMS, BF16);
        let (u, uv) = upload(&up_vals, &DIMS, BF16);
        let (x, xv) = upload(&x_vals, &DIMS, BF16);
        let c = no_fallback(|| unsafe { nsl_tensor_swiglu_gate_backward_dt(g, u, x, dout) });
        let out = download(c, dout);
        let mut want = Vec::with_capacity(N);
        let mut err = Vec::with_capacity(N);
        for i in 0..N {
            let (gi, ui, xi) = (gv[i], uv[i], xv[i]);
            let s = 1.0 / (1.0 + (-xi).exp());
            let t = gi * ui;
            let one_m_s = 1.0 - s;
            let a = xi * one_m_s;
            let bb = 1.0 + a;
            let tail = s * bb;
            let y = t * tail;
            let e_s = sigmoid_rel(xi) * s;
            let e_oms = e_s + U * one_m_s.abs();
            let e_a = xi.abs() * e_oms + U * a.abs();
            let e_b = e_a + U * bb.abs();
            let e_tail = bb.abs() * e_s + s * e_b + U * tail.abs();
            let e_t = U * t.abs();
            let e_y = t.abs() * e_tail + tail.abs() * e_t + U * y.abs();
            want.push(y);
            err.push(e_y);
        }
        bracket(&format!("swiglu_gate_backward bf16^3->{dout}"), &out, &want, &err, dout);
        for t in [g, u, x, c] {
            unsafe { nsl_tensor_free(t) };
        }
    }
}

/// `rotate_half` moves and negates values: exact for bf16 -> bf16, one RNE
/// rounding for f32 -> bf16 (E = 0 in both).
#[test]
#[ignore = "requires CUDA GPU"]
fn rotate_half_is_exact_or_rounds_once() {
    init();
    let dims = [2i64, 3, 64]; // last dim 64, as RoPE's head_dim
    let n = 2 * 3 * 64;
    let vals = values(n, 7, -3.0, 2.0);
    for neg in [0i64, 1] {
        for (din, dout) in [(BF16, BF16), (F32, BF16), (BF16, F32)] {
            let (a, av) = upload(&vals, &dims, din);
            let c = no_fallback(|| unsafe { nsl_tensor_rotate_half_dt(a, neg, dout) });
            let out = download(c, dout);
            let want: Vec<f64> = (0..n)
                .map(|i| {
                    let (col, half) = (i % 64, 32);
                    if col < half {
                        let v = av[i + half];
                        if neg == 0 { -v } else { v }
                    } else {
                        let v = av[i - half];
                        if neg == 0 { v } else { -v }
                    }
                })
                .collect();
            bracket(&format!("rotate_half neg={neg} {din}->{dout}"), &out, &want, &vec![0.0; n], dout);
            unsafe { nsl_tensor_free(a) };
            unsafe { nsl_tensor_free(c) };
        }
    }
}

/// A strided or host operand cannot take the kernel: the entry point widens,
/// computes in f32 and narrows, counted, with the same values.
#[test]
#[ignore = "requires CUDA GPU"]
fn the_fallback_is_counted_and_computes_the_same_values() {
    init();
    let a_vals = values(N, 8, -3.0, 1.0);
    let b_vals = values(N, 9, -3.0, 1.0);
    let (a, av) = upload(&a_vals, &DIMS, BF16);
    let (b, bv) = upload(&b_vals, &DIMS, BF16);
    // A host copy of `b` forces the fallback (operands on different devices).
    let b_host = unsafe { nsl_tensor_to_device(b, 0) };
    let before = unsafe { nsl_storage_dt_fallback_count() };
    let c = unsafe { nsl_tensor_binary_dt(dt_op::MUL, a, b_host, BF16) };
    let after = unsafe { nsl_storage_dt_fallback_count() };
    assert_eq!(after, before + 1, "the host operand must have taken the counted fallback");
    let out = download(c, BF16);
    let want: Vec<f64> = av.iter().zip(&bv).map(|(x, y)| x * y).collect();
    let err: Vec<f64> = want.iter().map(|y| U * y.abs()).collect();
    bracket("mul fallback bf16 x host bf16 -> bf16", &out, &want, &err, BF16);
    for t in [a, b, b_host, c] {
        unsafe { nsl_tensor_free(t) };
    }
}
