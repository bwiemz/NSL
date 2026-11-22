//! The fused SwiGLU kernels (`nsl_kir::kernels::swiglu`, `nsl_runtime::swiglu`)
//! against the launches they replace, on the GPU.
//!
//! * **f32 storage, bit for bit.** `swiglu(g, u)` equals `silu(g) * u`,
//!   `swiglu_up_backward(dy, g)` equals `dy * silu(g)`, and the paired
//!   `swiglu_backward(dy, u, g)` equals `[swiglu_gate_backward(dy, u, g),
//!   dy * silu(g)]`, each through the ordinary f32 entry points, at every
//!   width the launcher picks (sizes chosen so it picks 8, 4 and 1).
//! * **bf16 storage, one rounding.** With bf16 operands, a bf16 result is the
//!   f32 kernel's result on the same (exactly widened) values, rounded once
//!   to nearest even; an f32 result from bf16 operands is that f32 result.
//! * **Against the unfused bf16 chain.** That chain also rounds `silu(g)` to
//!   bf16 when it stores it. The fused result differs from it by at most one
//!   bf16 ulp, and only where that rounding moved the product
//!   (`swiglu_round_silu_gpu.rs` holds the kernels bit for bit to the chain
//!   with the rounding reinstated).
//! * **Fallback.** A broadcast operand takes the decomposed path: the same
//!   values as the unfused ops, counted by `nsl_storage_dt_fallback_count`
//!   when bf16 is involved.

#![cfg(feature = "cuda")]

use nsl_abi::wire::dt_op;
use nsl_runtime::nsl_cuda_init;

unsafe extern "C" {
    fn nsl_tensor_from_static(data_ptr: i64, shape_list: i64, dtype: i64) -> i64;
    fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64;
    fn nsl_tensor_free(tensor_ptr: i64);
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
    fn nsl_list_get(list_ptr: i64, idx: i64) -> i64;
    fn nsl_list_free(list_ptr: i64);
    fn nsl_tensor_silu(a: i64) -> i64;
    fn nsl_tensor_mul(a: i64, b: i64, flags: u8) -> i64;
    fn nsl_tensor_swiglu_gate_backward(grad: i64, up: i64, x: i64) -> i64;
    fn nsl_tensor_unary_dt(op: i64, a: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_binary_dt(op: i64, a: i64, b: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_swiglu(gate: i64, up: i64) -> i64;
    fn nsl_tensor_swiglu_dt(gate: i64, up: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_swiglu_up_backward(grad: i64, gate: i64) -> i64;
    fn nsl_tensor_swiglu_up_backward_dt(grad: i64, gate: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_swiglu_backward(grad: i64, up: i64, gate: i64) -> i64;
    fn nsl_tensor_swiglu_backward_dt(grad: i64, up: i64, gate: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_swiglu_backward_recompute(grad: i64, up: i64, gate: i64) -> i64;
    fn nsl_tensor_swiglu_backward_recompute_dt(
        grad: i64,
        up: i64,
        gate: i64,
        out_dtype: i64,
    ) -> i64;
    fn nsl_storage_dt_fallback_count() -> i64;
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

/// Held by every test that reads `nsl_storage_dt_fallback_count` (a process
/// global) across calls: the harness runs tests on threads, and one test's
/// counted fallback would otherwise land inside another's window.
fn fallback_counter() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(|p| p.into_inner())
}

/// bf16 bits of `x` (an f32), rounded to nearest even (`x` finite).
fn bf16_rne_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let lsb = (b >> 16) & 1;
    ((b + 0x7FFF + lsb) >> 16) as u16
}

/// A device tensor of shape `dims`: `vals` as f32, or (bf16) each rounded to
/// nearest even first.
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
    unsafe { nsl_tensor_to_device(cpu, 1) }
}

/// Mirror of `NslTensor`'s `#[repr(C)]` header (as `bf16_storage_kernels_gpu.rs`).
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

/// The raw element bits of a device tensor (u32 for f32, u16 widened for
/// bf16), its dtype checked. Frees `t`.
fn bits(t: i64, dtype: i64) -> Vec<u32> {
    assert_eq!(view(t).device, 1, "result must be on the device");
    assert_eq!(view(t).dtype as i64, dtype, "result dtype tag");
    let host = unsafe { nsl_tensor_to_device(t, 0) };
    let n = view(host).len as usize;
    let out = if dtype == BF16 {
        let d = unsafe { std::slice::from_raw_parts(view(host).data as *const u16, n) };
        d.iter().map(|&b| u32::from(b)).collect()
    } else {
        let d = unsafe { std::slice::from_raw_parts(view(host).data as *const u32, n) };
        d.to_vec()
    };
    unsafe {
        nsl_tensor_free(host);
        nsl_tensor_free(t);
    }
    out
}

/// Deterministic values over several decades and both signs, with exact
/// zeros and sigmoid-saturating magnitudes mixed in.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|k| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            let v = match k % 17 {
                0 => 0.0,
                1 => 30.0 * (u - 0.5) * 4.0, // |v| up to 60: sigmoid saturates
                _ => 10f64.powf(-3.0 + 4.0 * u) * if k % 3 == 1 { -1.0 } else { 1.0 },
            };
            v as f32
        })
        .collect()
}

fn list2(l: i64) -> (i64, i64) {
    let (a, b) = unsafe { (nsl_list_get(l, 0), nsl_list_get(l, 1)) };
    unsafe { nsl_list_free(l) };
    (a, b)
}

fn list3(l: i64) -> (i64, i64, i64) {
    let (a, b, c) = unsafe { (nsl_list_get(l, 0), nsl_list_get(l, 1), nsl_list_get(l, 2)) };
    unsafe { nsl_list_free(l) };
    (a, b, c)
}

/// Shapes whose element counts make the launcher pick width 8, 4 and 1.
const SHAPES: &[&[i64]] = &[&[64, 96], &[3, 4], &[5, 7], &[1, 8], &[2, 2051]];

#[test]
#[ignore = "requires CUDA GPU"]
fn f32_fused_kernels_equal_the_unfused_launches_bit_for_bit() {
    init();
    for (si, dims) in SHAPES.iter().enumerate() {
        let n: usize = dims.iter().product::<i64>() as usize;
        let (g, u, dy) = (
            upload(&values(n, 1 + si as u64), dims, F32),
            upload(&values(n, 101 + si as u64), dims, F32),
            upload(&values(n, 201 + si as u64), dims, F32),
        );
        let unfused_p = unsafe {
            let s = nsl_tensor_silu(g);
            let p = nsl_tensor_mul(s, u, 0);
            nsl_tensor_free(s);
            bits(p, F32)
        };
        let unfused_up = unsafe {
            let s = nsl_tensor_silu(g);
            let r = nsl_tensor_mul(dy, s, 0);
            nsl_tensor_free(s);
            bits(r, F32)
        };
        let unfused_gate = bits(unsafe { nsl_tensor_swiglu_gate_backward(dy, u, g) }, F32);
        assert_eq!(
            bits(unsafe { nsl_tensor_swiglu(g, u) }, F32),
            unfused_p,
            "{dims:?}: forward"
        );
        assert_eq!(
            bits(unsafe { nsl_tensor_swiglu_dt(g, u, F32) }, F32),
            unfused_p,
            "{dims:?}: forward _dt"
        );
        assert_eq!(
            bits(unsafe { nsl_tensor_swiglu_up_backward(dy, g) }, F32),
            unfused_up,
            "{dims:?}: up"
        );
        assert_eq!(
            bits(unsafe { nsl_tensor_swiglu_up_backward_dt(dy, g, F32) }, F32),
            unfused_up,
            "{dims:?}: up _dt"
        );
        for dt in [None, Some(F32)] {
            let l = unsafe {
                match dt {
                    None => nsl_tensor_swiglu_backward(dy, u, g),
                    Some(d) => nsl_tensor_swiglu_backward_dt(dy, u, g, d),
                }
            };
            let (dg, du) = list2(l);
            assert_eq!(bits(dg, F32), unfused_gate, "{dims:?} {dt:?}: paired dgate");
            assert_eq!(bits(du, F32), unfused_up, "{dims:?} {dt:?}: paired dup");
            // With the forward value recomputed in the same launch.
            let l = unsafe {
                match dt {
                    None => nsl_tensor_swiglu_backward_recompute(dy, u, g),
                    Some(d) => nsl_tensor_swiglu_backward_recompute_dt(dy, u, g, d),
                }
            };
            let (dg, du, p) = list3(l);
            assert_eq!(
                bits(dg, F32),
                unfused_gate,
                "{dims:?} {dt:?}: grouped dgate"
            );
            assert_eq!(bits(du, F32), unfused_up, "{dims:?} {dt:?}: grouped dup");
            assert_eq!(
                bits(p, F32),
                unfused_p,
                "{dims:?} {dt:?}: grouped forward value"
            );
        }
        unsafe {
            nsl_tensor_free(g);
            nsl_tensor_free(u);
            nsl_tensor_free(dy);
        }
    }
}

/// bf16 operands: a bf16 result is the f32 result rounded once; an f32
/// result is the f32 result. The f32 reference runs the fused f32 kernels on
/// the same values (exact widening), which the test above holds to the
/// unfused launches.
#[test]
#[ignore = "requires CUDA GPU"]
fn bf16_outputs_are_the_f32_result_rounded_once() {
    init();
    let _counter = fallback_counter();
    let before = unsafe { nsl_storage_dt_fallback_count() };
    for (si, dims) in SHAPES.iter().enumerate() {
        let n: usize = dims.iter().product::<i64>() as usize;
        // Values representable in bf16, uploaded both ways.
        let rt = |v: Vec<f32>| -> Vec<f32> {
            v.into_iter()
                .map(|x| f32::from_bits(u32::from(bf16_rne_bits(x)) << 16))
                .collect()
        };
        let (gv, uv, dv) = (
            rt(values(n, 7 + si as u64)),
            rt(values(n, 77 + si as u64)),
            rt(values(n, 777 + si as u64)),
        );
        let (g16, u16_, d16) = (
            upload(&gv, dims, BF16),
            upload(&uv, dims, BF16),
            upload(&dv, dims, BF16),
        );
        let (g32, u32_, d32) = (
            upload(&gv, dims, F32),
            upload(&uv, dims, F32),
            upload(&dv, dims, F32),
        );
        let once = |f32_bits: Vec<u32>| -> Vec<u32> {
            f32_bits
                .into_iter()
                .map(|b| u32::from(bf16_rne_bits(f32::from_bits(b))))
                .collect()
        };
        let p32 = bits(unsafe { nsl_tensor_swiglu(g32, u32_) }, F32);
        let up32 = bits(unsafe { nsl_tensor_swiglu_up_backward(d32, g32) }, F32);
        let (dg32, du32) = list2(unsafe { nsl_tensor_swiglu_backward(d32, u32_, g32) });
        let (dg32, du32) = (bits(dg32, F32), bits(du32, F32));
        assert_eq!(
            bits(unsafe { nsl_tensor_swiglu_dt(g16, u16_, BF16) }, BF16),
            once(p32.clone()),
            "{dims:?}"
        );
        assert_eq!(
            bits(unsafe { nsl_tensor_swiglu_dt(g16, u16_, F32) }, F32),
            p32,
            "{dims:?}"
        );
        assert_eq!(
            bits(
                unsafe { nsl_tensor_swiglu_up_backward_dt(d16, g16, BF16) },
                BF16
            ),
            once(up32.clone()),
            "{dims:?}"
        );
        let (dg, du) = list2(unsafe { nsl_tensor_swiglu_backward_dt(d16, u16_, g16, BF16) });
        assert_eq!(bits(dg, BF16), once(dg32.clone()), "{dims:?}: paired dgate");
        assert_eq!(bits(du, BF16), once(du32.clone()), "{dims:?}: paired dup");
        let (dg, du, p) =
            list3(unsafe { nsl_tensor_swiglu_backward_recompute_dt(d16, u16_, g16, BF16) });
        assert_eq!(
            bits(dg, BF16),
            once(dg32.clone()),
            "{dims:?}: grouped dgate"
        );
        assert_eq!(bits(du, BF16), once(du32.clone()), "{dims:?}: grouped dup");
        assert_eq!(
            bits(p, BF16),
            once(p32.clone()),
            "{dims:?}: grouped forward value"
        );
        // Mixed operand storages take the kernel too.
        assert_eq!(
            bits(unsafe { nsl_tensor_swiglu_dt(g16, u32_, BF16) }, BF16),
            once(p32),
            "{dims:?}: mixed"
        );
        for t in [g16, u16_, d16, g32, u32_, d32] {
            unsafe { nsl_tensor_free(t) };
        }
    }
    assert_eq!(
        unsafe { nsl_storage_dt_fallback_count() },
        before,
        "a same-shape bf16 call took the fallback"
    );
}

/// The unfused bf16 chain stores `silu(g)` bf16 (one more rounding); the
/// fused kernels do not. The difference is at most one bf16 ulp per element.
#[test]
#[ignore = "requires CUDA GPU"]
fn bf16_fused_results_are_within_one_ulp_of_the_unfused_chain() {
    init();
    let dims: &[i64] = &[256, 384];
    let n = (256 * 384) as usize;
    let (g, u, dy) = (
        upload(&values(n, 3), dims, BF16),
        upload(&values(n, 33), dims, BF16),
        upload(&values(n, 333), dims, BF16),
    );
    let silu16 = unsafe { nsl_tensor_unary_dt(dt_op::SILU, g, BF16) };
    let chain_p = bits(
        unsafe { nsl_tensor_binary_dt(dt_op::MUL, silu16, u, BF16) },
        BF16,
    );
    let chain_up = bits(
        unsafe { nsl_tensor_binary_dt(dt_op::MUL, dy, silu16, BF16) },
        BF16,
    );
    unsafe { nsl_tensor_free(silu16) };
    let fused_p = bits(unsafe { nsl_tensor_swiglu_dt(g, u, BF16) }, BF16);
    let (_, du) = list2(unsafe { nsl_tensor_swiglu_backward_dt(dy, u, g, BF16) });
    let fused_up = bits(du, BF16);
    for (what, a, b) in [
        ("forward", &fused_p, &chain_p),
        ("up adjoint", &fused_up, &chain_up),
    ] {
        let mut differ = 0usize;
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            if x == y {
                continue;
            }
            differ += 1;
            // Same sign, adjacent magnitudes (bf16 bit patterns order like
            // the values within a sign).
            let ulps = (i64::from(x & 0x7FFF) - i64::from(y & 0x7FFF)).abs();
            assert!(
                x & 0x8000 == y & 0x8000 && ulps <= 1,
                "{what}[{i}]: fused {x:#06x} vs chain {y:#06x}"
            );
        }
        eprintln!(
            "{what}: {differ} of {} elements differ by one bf16 ulp",
            a.len()
        );
        assert!(
            differ > 0,
            "{what}: no element differs; the silu rounding should move some products"
        );
        assert!(
            differ * 4 < a.len(),
            "{what}: {differ} of {} differ",
            a.len()
        );
    }
    for t in [g, u, dy] {
        unsafe { nsl_tensor_free(t) };
    }
}

/// A broadcast operand takes the decomposed path: the unfused ops' values.
#[test]
#[ignore = "requires CUDA GPU"]
fn a_broadcast_operand_takes_the_decomposed_path() {
    init();
    let _counter = fallback_counter();
    let (g, u) = (
        upload(&values(24, 5), &[4, 6], F32),
        upload(&values(6, 55), &[6], F32),
    );
    let want = unsafe {
        let s = nsl_tensor_silu(g);
        let p = nsl_tensor_mul(s, u, 0);
        nsl_tensor_free(s);
        bits(p, F32)
    };
    assert_eq!(bits(unsafe { nsl_tensor_swiglu(g, u) }, F32), want);
    // The grouped backward's decomposed path: its forward value too.
    let dy = upload(&values(24, 555), &[4, 6], F32);
    let (dg, du, p) = list3(unsafe { nsl_tensor_swiglu_backward_recompute(dy, u, g) });
    assert_eq!(
        bits(p, F32),
        want,
        "the decomposed grouped backward's forward value"
    );
    assert_eq!(view(dg).len, 24);
    assert_eq!(
        view(du).len,
        24,
        "the up adjoint at the product's shape (the compiler reduces it)"
    );
    for t in [dg, du, dy] {
        unsafe { nsl_tensor_free(t) };
    }
    let (g16, u16_) = (
        upload(&values(24, 5), &[4, 6], BF16),
        upload(&values(6, 55), &[6], BF16),
    );
    let before = unsafe { nsl_storage_dt_fallback_count() };
    let r = unsafe { nsl_tensor_swiglu_dt(g16, u16_, BF16) };
    assert_eq!(view(r).len, 24, "the broadcast result has the full shape");
    assert_eq!(
        unsafe { nsl_storage_dt_fallback_count() },
        before + 1,
        "the bf16 fallback is counted"
    );
    let _ = bits(r, BF16);
    for t in [g, u, g16, u16_] {
        unsafe { nsl_tensor_free(t) };
    }
}
