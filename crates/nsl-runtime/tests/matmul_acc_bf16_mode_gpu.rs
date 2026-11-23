//! `nsl_tensor_matmul_acc` route 0 (the plain `nsl_tensor_matmul`'s product, which the
//! input gradients of the attention projections take under `precision="bf16"`: their
//! upstream gradient is f32) in the BF16 math mode, against the unfused program's three
//! calls, on the GPU. The math mode resolves once per process, hence its own binary.
//!
//! * **Bit for bit.** Where the eager mode runs the product as the bf16 GEMM over
//!   round-to-nearest operand images (the gate passes), the one-GEMM accumulation equals
//!   `nsl_tensor_matmul` + `nsl_tensor_reduce_to_shape` + `nsl_tensor_add` to the bit, at
//!   the Coder-1B and Coder-500M q/k input-gradient shapes, timed and `--deterministic`.
//! * **Declined.** A low-intensity product (the gate keeps f32 operands at TF32) is the
//!   three calls, counted.

#![cfg(all(feature = "cuda", feature = "test-hooks"))]

use nsl_runtime::nsl_cuda_init;
use nsl_runtime::storage_dt::matmul_acc_counts;

unsafe extern "C" {
    fn nsl_tensor_from_static(data_ptr: i64, shape_list: i64, dtype: i64) -> i64;
    fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64;
    fn nsl_tensor_free(tensor_ptr: i64);
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
    fn nsl_list_free(list_ptr: i64);
    fn nsl_tensor_transpose(t: i64, d0: i64, d1: i64) -> i64;
    fn nsl_tensor_add(a: i64, b: i64, flags: u8) -> i64;
    fn nsl_tensor_matmul(a: i64, b: i64, flags: u8) -> i64;
    fn nsl_tensor_reduce_to_shape(g: i64, like: i64) -> i64;
    fn nsl_tensor_matmul_acc(a: i64, b: i64, c: i64, route: i64) -> i64;
    fn nsl_set_deterministic(mode: i64) -> i64;
    fn nsl_set_matmul_config(
        mode: i64,
        rounding: i64,
        min_ratio: f64,
        cast_cache: i64,
        lt: i64,
        ws: i64,
        tune: i64,
    ) -> i64;
}

const F32: i64 = 1;
const BF16: i64 = 3;

fn init() {
    assert_eq!(
        nsl_cuda_init(),
        0,
        "nsl_cuda_init failed: these gates run where a CUDA device is a precondition"
    );
}

fn bf16_rne_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    ((b + 0x7FFF + lsb) >> 16) as u16
}

fn upload(vals: &[f32], dims: &[i64], dtype: i64) -> i64 {
    let list = unsafe { nsl_list_new() };
    for &d in dims {
        unsafe { nsl_list_push(list, d) };
    }
    let ptr = if dtype == BF16 {
        let bits: Vec<u16> = vals.iter().map(|&v| bf16_rne_bits(v)).collect();
        Box::leak(bits.into_boxed_slice()).as_ptr() as i64
    } else {
        Box::leak(vals.to_vec().into_boxed_slice()).as_ptr() as i64
    };
    let cpu = unsafe { nsl_tensor_from_static(ptr, list, dtype) };
    unsafe { nsl_list_free(list) };
    let dev = unsafe { nsl_tensor_to_device(cpu, 1) };
    unsafe { nsl_tensor_free(cpu) };
    dev
}

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

/// The f32 bits of a device tensor (not freed).
fn bits(t: i64) -> Vec<u32> {
    assert_eq!(view(t).device, 1);
    assert_eq!(view(t).dtype as i64, F32, "a summed gradient is f32");
    let host = unsafe { nsl_tensor_to_device(t, 0) };
    let n = view(host).len as usize;
    let out = unsafe { std::slice::from_raw_parts(view(host).data as *const u32, n) }.to_vec();
    unsafe { nsl_tensor_free(host) };
    out
}

/// Gradient-like values: magnitudes 10^lo .. 10^hi, both signs, some exact zeros.
fn values(n: usize, seed: u64, lo: f64, hi: f64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|k| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            if k % 31 == 0 {
                return 0.0;
            }
            (10f64.powf(lo + (hi - lo) * u) * if (s >> 7) & 1 == 1 { -1.0 } else { 1.0 }) as f32
        })
        .collect()
}

/// `((t0 + t1) + t2)` of f32 terms `G [lead.., f_out] @ T(W)`, unfused and through route
/// 0; returns both sums' bits and how many accumulations took one GEMM.
fn chain_f32(lead: &[i64], f_in: i64, f_outs: &[i64], seed: u64) -> (Vec<u32>, Vec<u32>, u64) {
    let rows: i64 = lead.iter().product();
    unsafe {
        let mk = |fo: i64, s: u64| {
            let mut gd = lead.to_vec();
            gd.push(fo);
            let g = upload(&values((rows * fo) as usize, s, -9.0, -5.0), &gd, F32);
            let w = upload(
                &values((f_in * fo) as usize, s + 7, -3.0, -1.3),
                &[f_in, fo],
                F32,
            );
            let wt = nsl_tensor_transpose(w, 0, 1);
            (g, w, wt)
        };
        let terms: Vec<(i64, i64, i64)> = f_outs
            .iter()
            .enumerate()
            .map(|(i, &fo)| mk(fo, seed + 31 * i as u64))
            .collect();
        let first = nsl_tensor_matmul(terms[0].0, terms[0].2, 0);
        let mut unfused = nsl_tensor_reduce_to_shape(first, first);
        let mut fused = first;
        let before = matmul_acc_counts().0;
        for &(g_in, _, wt) in &terms[1..] {
            let g = nsl_tensor_matmul(g_in, wt, 0);
            let r = nsl_tensor_reduce_to_shape(g, unfused);
            let s = nsl_tensor_add(unfused, r, 0);
            nsl_tensor_free(r);
            nsl_tensor_free(g);
            nsl_tensor_free(unfused);
            unfused = s;
            let s2 = nsl_tensor_matmul_acc(g_in, wt, fused, 0);
            nsl_tensor_free(fused);
            fused = s2;
        }
        let one_gemm = matmul_acc_counts().0 - before;
        let out = (bits(unfused), bits(fused), one_gemm);
        nsl_tensor_free(unfused);
        nsl_tensor_free(fused);
        for (g, w, wt) in terms {
            nsl_tensor_free(wt);
            nsl_tensor_free(w);
            nsl_tensor_free(g);
        }
        out
    }
}

/// `(lead dims, f_in, the terms' f_out)`: Coder-1B's q/k terms, Coder-500M's.
const CHAINS: &[(&[i64], i64, &[i64])] = &[
    (&[2, 2048], 2048, &[512, 2048]),
    (&[2, 1024], 1280, &[640, 1280]),
];

#[test]
#[ignore = "requires CUDA GPU"]
fn route_zero_in_the_bf16_math_mode_equals_the_eager_matmul_and_the_add_bit_for_bit() {
    init();
    // The production `precision="bf16"` configuration: BF16 math mode, round to nearest,
    // the weight-cast cache, timed cublasLt plans. Before any GEMM resolves the mode.
    assert_eq!(
        unsafe { nsl_set_matmul_config(1, 0, 512.0, 1, 0, 64, 1) },
        0
    );
    for det in [0, 1] {
        unsafe { nsl_set_deterministic(det) };
        for (ci, &(lead, f_in, f_outs)) in CHAINS.iter().enumerate() {
            let (want, got, one) = chain_f32(lead, f_in, f_outs, 3000 + 13 * ci as u64);
            let diff = want.iter().zip(&got).filter(|(a, b)| a != b).count();
            assert_eq!(
                diff,
                0,
                "det={det} {lead:?} x {f_in}: {diff} of {} elements differ",
                want.len()
            );
            assert_eq!(
                one,
                (f_outs.len() - 1) as u64,
                "det={det}: the one-GEMM path did not run"
            );
        }
    }
    unsafe { nsl_set_deterministic(0) };
    // A low-intensity product: the eager mode keeps f32 operands (TF32), so route 0
    // decomposes; still the same values.
    let (want, got, one) = chain_f32(&[2, 4], 8, &[8, 8], 77);
    assert_eq!(one, 0, "a product the gate declines must decompose");
    assert_eq!(want, got);
}
