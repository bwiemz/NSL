//! The fused residual add + RMSNorm (`nsl_tensor_add_rmsnorm`,
//! `nsl_kir::kernels::norm::build_add_rmsnorm`) against the two launches it replaces,
//! on the GPU.
//!
//! * **Bit for bit, every route.** For each operand storage (f32/bf16 `x` and `o`) and
//!   each route the compiler can name for the add (plain `nsl_tensor_add`, or
//!   `nsl_tensor_binary_dt` to f32/bf16) and for the norm (plain `nsl_tensor_rmsnorm`,
//!   or `nsl_tensor_rmsnorm_dt` to f32/bf16), the fused `[h, n]` equals the pair's
//!   outputs bit for bit, on rows shorter than, equal to and longer than the 256-thread
//!   block, a row length that is no multiple of it, and values over many decades with
//!   zeros, signed zeros and overflowing squares mixed in. And the fused kernel ran
//!   (`residual_norm::counts`), so the comparison is not the pair against itself.
//! * **Fallback.** A broadcast operand (or an f32 route the plain entry points do not
//!   take to the KIR kernels) is the pair of calls itself, counted.

#![cfg(feature = "cuda")]

use nsl_abi::wire::dt_op;
use nsl_runtime::nsl_cuda_init;
use nsl_runtime::residual_norm::counts;

unsafe extern "C" {
    fn nsl_tensor_from_static(data_ptr: i64, shape_list: i64, dtype: i64) -> i64;
    fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64;
    fn nsl_tensor_free(tensor_ptr: i64);
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
    fn nsl_list_get(list_ptr: i64, idx: i64) -> i64;
    fn nsl_list_free(list_ptr: i64);
    fn nsl_tensor_add(a: i64, b: i64, flags: u8) -> i64;
    fn nsl_tensor_rmsnorm(x: i64, w: i64, eps: f64) -> i64;
    fn nsl_tensor_binary_dt(op: i64, a: i64, b: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_rmsnorm_dt(x: i64, gamma: i64, eps: f64, out_dtype: i64) -> i64;
    fn nsl_tensor_add_rmsnorm(
        x: i64,
        o: i64,
        gamma: i64,
        eps: f64,
        h_route: i64,
        n_route: i64,
    ) -> i64;
}

const F32: i64 = 1;
const BF16: i64 = 3;
const EPS: f64 = 1e-5;

fn init() {
    assert_eq!(
        nsl_cuda_init(),
        0,
        "nsl_cuda_init failed: these gates run where a CUDA device is a precondition"
    );
}

/// The fused/fallback counters are process globals: tests that read them across calls
/// hold this.
fn counter() -> std::sync::MutexGuard<'static, ()> {
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
    unsafe { nsl_tensor_to_device(cpu, 1) }
}

/// Mirror of `NslTensor`'s `#[repr(C)]` header (as `swiglu_fused_gpu.rs`).
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

/// `(dtype, raw element bits)` of a device tensor; frees `t`.
fn bits(t: i64) -> (i64, Vec<u32>) {
    assert_eq!(view(t).device, 1, "result must be on the device");
    let dtype = view(t).dtype as i64;
    let host = unsafe { nsl_tensor_to_device(t, 0) };
    let n = view(host).len as usize;
    let out = if dtype == BF16 {
        let d = unsafe { std::slice::from_raw_parts(view(host).data as *const u16, n) };
        d.iter().map(|&b| u32::from(b)).collect()
    } else {
        assert_eq!(dtype, F32, "an f32 or bf16 result");
        let d = unsafe { std::slice::from_raw_parts(view(host).data as *const u32, n) };
        d.to_vec()
    };
    unsafe {
        nsl_tensor_free(host);
        nsl_tensor_free(t);
    }
    (dtype, out)
}

/// Values over several decades and both signs, with zeros, signed zeros and (every
/// 97th) magnitudes whose square overflows f32.
fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|k| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            let v = match k % 29 {
                0 => 0.0,
                1 => -0.0,
                _ if k % 97 == 5 => 3.0e20 * (u - 0.5),
                _ => 10f64.powf(-4.0 + 5.0 * u) * if k % 3 == 1 { -1.0 } else { 1.0 },
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

/// The two calls the unfused lowering makes for these routes.
fn pair(x: i64, o: i64, g: i64, h_route: i64, n_route: i64) -> (i64, i64) {
    unsafe {
        let h = if h_route == 0 {
            nsl_tensor_add(x, o, 0)
        } else {
            nsl_tensor_binary_dt(dt_op::ADD, x, o, h_route)
        };
        let n = if n_route == 0 {
            nsl_tensor_rmsnorm(h, g, EPS)
        } else {
            nsl_tensor_rmsnorm_dt(h, g, EPS, n_route)
        };
        (h, n)
    }
}

/// Row lengths below, at and above the block size, one that is no multiple of it, and
/// the two production widths.
const SHAPES: &[&[i64]] = &[&[3, 64], &[2, 5, 256], &[7, 300], &[4, 1280], &[2, 3, 2048]];

#[test]
#[ignore = "requires CUDA GPU"]
fn the_fused_launch_equals_the_add_and_the_norm_bit_for_bit_on_every_route() {
    init();
    let _held = counter();
    let mut cases = 0;
    for (si, dims) in SHAPES.iter().enumerate() {
        let n: usize = dims.iter().product::<i64>() as usize;
        let cols = *dims.last().expect("ndim >= 1") as usize;
        let g = upload(
            &values(cols, 900 + si as u64)
                .iter()
                .map(|v| 1.0 + v.clamp(-0.5, 0.5))
                .collect::<Vec<_>>(),
            &[cols as i64],
            F32,
        );
        for xd in [F32, BF16] {
            for od in [F32, BF16] {
                let x = upload(&values(n, 1 + si as u64), dims, xd);
                let o = upload(&values(n, 51 + si as u64), dims, od);
                for h_route in [0, F32, BF16] {
                    if h_route == 0 && (xd != F32 || od != F32) {
                        continue; // the plain add is f32 + f32 in a lowered program
                    }
                    let h_dt = if h_route == 0 { F32 } else { h_route };
                    for n_route in [0, F32, BF16] {
                        if n_route == 0 && h_dt != F32 {
                            continue; // the plain norm reads an f32 input
                        }
                        let (want_h, want_n) = pair(x, o, g, h_route, n_route);
                        let before = counts();
                        let (h, nn) = list2(unsafe {
                            nsl_tensor_add_rmsnorm(x, o, g, EPS, h_route, n_route)
                        });
                        let after = counts();
                        assert_eq!(
                            after.0,
                            before.0 + 1,
                            "{dims:?} x{xd} o{od} h{h_route} n{n_route}: the fused kernel did not run"
                        );
                        let tag =
                            format!("{dims:?} x{xd} o{od} h-route {h_route} n-route {n_route}");
                        assert_eq!(bits(h), bits(want_h), "h differs: {tag}");
                        assert_eq!(bits(nn), bits(want_n), "n differs: {tag}");
                        cases += 1;
                    }
                }
                unsafe {
                    nsl_tensor_free(x);
                    nsl_tensor_free(o);
                }
            }
        }
        unsafe { nsl_tensor_free(g) };
    }
    // f32 + f32: 3 add routes (8 norm routes in all); each mixed pair: 2 (5).
    assert_eq!(cases, SHAPES.len() * (8 + 3 * 5), "case count");
}

#[test]
#[ignore = "requires CUDA GPU"]
fn a_broadcast_operand_takes_the_pair_of_calls_and_is_counted() {
    init();
    let _held = counter();
    let dims: &[i64] = &[6, 300];
    let x = upload(&values(1800, 7), dims, F32);
    let o = upload(&values(300, 8), &[300], F32);
    let g = upload(&vec![1.25; 300], &[300], F32);
    for (h_route, n_route) in [(0, 0), (F32, BF16)] {
        let (want_h, want_n) = pair(x, o, g, h_route, n_route);
        let before = counts();
        let (h, n) = list2(unsafe { nsl_tensor_add_rmsnorm(x, o, g, EPS, h_route, n_route) });
        let after = counts();
        assert_eq!(
            (after.0, after.1),
            (before.0, before.1 + 1),
            "a broadcast must fall back, counted"
        );
        assert_eq!(bits(h), bits(want_h));
        assert_eq!(bits(n), bits(want_n));
    }
    unsafe {
        nsl_tensor_free(x);
        nsl_tensor_free(o);
        nsl_tensor_free(g);
    }
}
