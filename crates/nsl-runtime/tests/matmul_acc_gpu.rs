//! `nsl_tensor_matmul_acc` (a linear layer's input gradient added to the gradient summed
//! so far for the same value by one GEMM with beta = 1; round-2 perf item 6) against the
//! unfused program's three calls, on the GPU.
//!
//! * **Bit for bit.** The bf16-storage route (`nsl_tensor_matmul_dt(.., f32)`,
//!   `nsl_tensor_reduce_to_shape`, `nsl_tensor_add`) and the one-GEMM fast path agree to
//!   the bit at the Coder-1B and Coder-500M input-gradient shapes (q/o, k/v, gate/up
//!   dgrads) and at a small shape, with gradient-sized operands (|G| ~ 1e-9..1e-5, the
//!   magnitudes the f16-flush bug of 2026-11-12 was found at), along a three-term chain
//!   `((v + k) + q)` as the compiler emits it. Under the timed cublasLt plans (checked:
//!   the products were issued through Lt) and under `--deterministic` (untimed GemmEx,
//!   where the accumulation copies C into D first).
//! * **Fallback.** A product whose shape is not the accumulator's takes the three calls,
//!   counted; route 0 (an f32 program's plain matmul) is the three calls by design.
//! * **A bf16 sum** (route 3: the value's gradient is bf16) replays the unfused program's
//!   bf16 GEMM, `_dt` reduce and `_dt` add, one rounding each, bit for bit.

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
    fn nsl_tensor_matmul_dt(a: i64, b: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_reduce_to_shape(g: i64, like: i64) -> i64;
    fn nsl_tensor_matmul_acc(a: i64, b: i64, c: i64, route: i64) -> i64;
    fn nsl_tensor_reduce_to_shape_dt(g: i64, like: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_binary_dt(op: i64, a: i64, b: i64, out_dtype: i64) -> i64;
    fn nsl_set_deterministic(mode: i64) -> i64;
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

/// The deterministic mode and the counters are process globals.
fn serial() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
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

/// One input-gradient term: `G [rows.., f_out]` bf16 times `T(W)`, `W [f_in, f_out]` f32
/// (the layer's parameter), as the backward issues it.
struct Term {
    g: i64,
    w: i64,
    wt: i64,
}

fn term(lead: &[i64], f_in: i64, f_out: i64, seed: u64) -> Term {
    let rows: i64 = lead.iter().product();
    let mut gd = lead.to_vec();
    gd.push(f_out);
    let g = upload(
        &values((rows * f_out) as usize, seed, -9.0, -5.0),
        &gd,
        BF16,
    );
    let w = upload(
        &values((f_in * f_out) as usize, seed + 7, -3.0, -1.3),
        &[f_in, f_out],
        F32,
    );
    let wt = unsafe { nsl_tensor_transpose(w, 0, 1) };
    Term { g, w, wt }
}

fn free_term(t: Term) {
    unsafe {
        nsl_tensor_free(t.wt);
        nsl_tensor_free(t.w);
        nsl_tensor_free(t.g);
    }
}

/// `((v + k) + q)` both ways; returns the two sums' bits.
fn chain(lead: &[i64], f_in: i64, f_outs: &[i64], seed: u64) -> (Vec<u32>, Vec<u32>) {
    let terms: Vec<Term> = f_outs
        .iter()
        .enumerate()
        .map(|(i, &fo)| term(lead, f_in, fo, seed + 31 * i as u64))
        .collect();
    unsafe {
        // The first term: the plain dgrad. Both chains start from it, each holding a
        // reference (the identity reduce returns its operand with one more).
        let first = nsl_tensor_matmul_dt(terms[0].g, terms[0].wt, F32);
        let mut unfused = nsl_tensor_reduce_to_shape(first, first);
        let mut fused = first;
        for t in &terms[1..] {
            let g = nsl_tensor_matmul_dt(t.g, t.wt, F32);
            let r = nsl_tensor_reduce_to_shape(g, unfused);
            let s = nsl_tensor_add(unfused, r, 0);
            nsl_tensor_free(r);
            nsl_tensor_free(g);
            nsl_tensor_free(unfused);
            unfused = s;
            let before = matmul_acc_counts();
            let s2 = nsl_tensor_matmul_acc(t.g, t.wt, fused, 1);
            assert_eq!(
                matmul_acc_counts().0,
                before.0 + 1,
                "{lead:?} x {f_in}: the one-GEMM path did not run"
            );
            nsl_tensor_free(fused);
            fused = s2;
        }
        let out = (bits(unfused), bits(fused));
        nsl_tensor_free(unfused);
        nsl_tensor_free(fused);
        for t in terms {
            free_term(t);
        }
        out
    }
}

/// `(lead dims, f_in, the terms' f_out)`: Coder-1B q/k/v and gate/up, Coder-500M's, and
/// a small product below the timed-plan size (GemmEx).
const CHAINS: &[(&[i64], i64, &[i64])] = &[
    (&[2, 2048], 2048, &[512, 512, 2048]),
    (&[2, 2048], 2048, &[8192, 8192]),
    (&[2, 1024], 1280, &[640, 640, 1280]),
    (&[2, 1024], 1280, &[3520, 3520]),
    (&[2, 16], 96, &[40, 40, 80]),
];

fn run_chains(mode: &str) {
    for (ci, &(lead, f_in, f_outs)) in CHAINS.iter().enumerate() {
        let (want, got) = chain(lead, f_in, f_outs, 1000 + 17 * ci as u64);
        let diff = want.iter().zip(&got).filter(|(a, b)| a != b).count();
        assert_eq!(
            diff,
            0,
            "{mode}: {lead:?} x {f_in} <- {f_outs:?}: {diff} of {} elements differ",
            want.len()
        );
        assert!(
            want.iter().any(|&b| b != 0 && b != 0x8000_0000),
            "{mode}: a vacuous all-zero sum"
        );
    }
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_accumulating_gemm_equals_the_gemm_and_the_add_bit_for_bit() {
    init();
    let _held = serial();
    unsafe { nsl_set_deterministic(0) };
    let (issued0, _, _) = nsl_runtime::test_lt_matmul_stats();
    run_chains("timed cublasLt");
    let (issued1, _, _) = nsl_runtime::test_lt_matmul_stats();
    assert!(
        issued1 > issued0,
        "the production-size products must have run through cublasLt"
    );
    // `--deterministic`: untimed GemmEx for both forms (D starts as a copy of C).
    unsafe { nsl_set_deterministic(1) };
    run_chains("--deterministic (GemmEx)");
    let (issued2, _, _) = nsl_runtime::test_lt_matmul_stats();
    unsafe { nsl_set_deterministic(0) };
    assert_eq!(
        issued2, issued1,
        "under --deterministic the products must not use the timed plans"
    );
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_product_shaped_unlike_the_accumulator_and_route_zero_take_the_three_calls() {
    init();
    let _held = serial();
    let t = term(&[4, 8], 24, 40, 5);
    unsafe {
        // c broadcasts over the rows: [1, 24] against the product's [4, 8, 24].
        let c = upload(&values(24, 77, -6.0, -4.0), &[1, 24], F32);
        let g = nsl_tensor_matmul_dt(t.g, t.wt, F32);
        let r = nsl_tensor_reduce_to_shape(g, c);
        let want = nsl_tensor_add(c, r, 0);
        let before = matmul_acc_counts();
        let got = nsl_tensor_matmul_acc(t.g, t.wt, c, 1);
        let after = matmul_acc_counts();
        assert_eq!(
            (after.0, after.1),
            (before.0, before.1 + 1),
            "the shape mismatch must decompose, counted"
        );
        assert_eq!(bits(got), bits(want));
        for x in [got, want, r, g, c] {
            nsl_tensor_free(x);
        }
        // Route 0: an f32 program's plain matmul, decomposed by design.
        let gf = upload(&values(32 * 40, 9, -9.0, -5.0), &[4, 8, 40], F32);
        let c = upload(&values(32 * 24, 10, -6.0, -4.0), &[4, 8, 24], F32);
        let g = nsl_tensor_matmul(gf, t.wt, 0);
        let r = nsl_tensor_reduce_to_shape(g, c);
        let want = nsl_tensor_add(c, r, 0);
        let got = nsl_tensor_matmul_acc(gf, t.wt, c, 0);
        assert_eq!(bits(got), bits(want));
        for x in [got, want, r, g, c, gf] {
            nsl_tensor_free(x);
        }
    }
    free_term(t);
}

/// The raw bf16 bits of a device tensor (not freed).
fn bits16(t: i64) -> Vec<u16> {
    assert_eq!(view(t).device, 1);
    assert_eq!(view(t).dtype as i64, BF16, "a bf16 sum stays bf16");
    let host = unsafe { nsl_tensor_to_device(t, 0) };
    let n = view(host).len as usize;
    let out = unsafe { std::slice::from_raw_parts(view(host).data as *const u16, n) }.to_vec();
    unsafe { nsl_tensor_free(host) };
    out
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_bf16_sum_replays_the_unfused_bf16_gemm_reduce_and_add() {
    init();
    let _held = serial();
    let t = term(&[2, 64], 96, 80, 21);
    unsafe {
        let c = upload(&values(128 * 96, 22, -6.0, -4.0), &[2, 64, 96], BF16);
        let g = nsl_tensor_matmul_dt(t.g, t.wt, BF16);
        let r = nsl_tensor_reduce_to_shape_dt(g, c, BF16);
        let want = nsl_tensor_binary_dt(nsl_abi::wire::dt_op::ADD, c, r, BF16);
        let before = matmul_acc_counts();
        let got = nsl_tensor_matmul_acc(t.g, t.wt, c, 3);
        assert_eq!(
            matmul_acc_counts().1,
            before.1 + 1,
            "route 3 is the unfused bf16 ops, counted"
        );
        assert_eq!(bits16(got), bits16(want));
        for x in [got, want, r, g, c] {
            nsl_tensor_free(x);
        }
    }
    free_term(t);
}
