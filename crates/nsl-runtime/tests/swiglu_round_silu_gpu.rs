//! `NSL_SWIGLU_ROUND_SILU=1`: the fused SwiGLU kernels round `silu(g)` to
//! bf16 before they multiply, as the unfused bf16 chain does when it stores
//! it. With that rounding reinstated, every fused kernel equals the unfused
//! bf16 launches bit for bit, at every width: so the only difference between
//! the fused and the unfused bf16 programs is that one rounding.
//!
//! Its own test binary (one test): the runtime reads the variable once per
//! process, before the first fused call.

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
    fn nsl_tensor_unary_dt(op: i64, a: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_binary_dt(op: i64, a: i64, b: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_swiglu_gate_backward_dt(grad: i64, up: i64, input: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_swiglu_dt(gate: i64, up: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_swiglu_up_backward_dt(grad: i64, gate: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_swiglu_backward_dt(grad: i64, up: i64, gate: i64, out_dtype: i64) -> i64;
}

const BF16: i64 = 3;

fn bf16_rne_bits(x: f32) -> u16 {
    let b = x.to_bits();
    ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
}

fn upload(vals: &[f32], dims: &[i64]) -> i64 {
    let list = unsafe { nsl_list_new() };
    for &d in dims {
        unsafe { nsl_list_push(list, d) };
    }
    let bits: Vec<u16> = vals.iter().map(|&v| bf16_rne_bits(v)).collect();
    let ptr = Box::leak(bits.into_boxed_slice()).as_ptr() as i64;
    let cpu = unsafe { nsl_tensor_from_static(ptr, list, BF16) };
    unsafe { nsl_list_free(list) };
    unsafe { nsl_tensor_to_device(cpu, 1) }
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
    _device: u8,
    dtype: u16,
    _owns_data: u8,
    _data_owner: i64,
    _slab_managed: u8,
    _tape_id: i64,
}

fn bits(t: i64) -> Vec<u16> {
    let v = unsafe { &*(t as *const TensorView) };
    assert_eq!(v.dtype as i64, BF16);
    let host = unsafe { nsl_tensor_to_device(t, 0) };
    let h = unsafe { &*(host as *const TensorView) };
    let out = unsafe { std::slice::from_raw_parts(h.data as *const u16, h.len as usize) }.to_vec();
    unsafe {
        nsl_tensor_free(host);
        nsl_tensor_free(t);
    }
    out
}

fn values(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;
    (0..n)
        .map(|k| {
            s = s
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let u = (s >> 11) as f64 / (1u64 << 53) as f64;
            (10f64.powf(-3.0 + 4.0 * u) * if k % 3 == 1 { -1.0 } else { 1.0 }) as f32
        })
        .collect()
}

#[test]
#[ignore = "requires CUDA GPU"]
fn with_the_silu_rounding_reinstated_the_fused_kernels_equal_the_unfused_bf16_chain() {
    // SAFETY: the test binary's only test sets it before any runtime call.
    unsafe { std::env::set_var("NSL_SWIGLU_ROUND_SILU", "1") };
    assert_eq!(nsl_cuda_init(), 0);
    for (si, dims) in [&[64i64, 96][..], &[3, 4], &[5, 7], &[2, 2051]]
        .iter()
        .enumerate()
    {
        let n: usize = dims.iter().product::<i64>() as usize;
        let (g, u, dy) = (
            upload(&values(n, 1 + si as u64), dims),
            upload(&values(n, 11 + si as u64), dims),
            upload(&values(n, 111 + si as u64), dims),
        );
        let s = unsafe { nsl_tensor_unary_dt(dt_op::SILU, g, BF16) };
        let chain_p = bits(unsafe { nsl_tensor_binary_dt(dt_op::MUL, s, u, BF16) });
        let chain_up = bits(unsafe { nsl_tensor_binary_dt(dt_op::MUL, dy, s, BF16) });
        unsafe { nsl_tensor_free(s) };
        let chain_gate = bits(unsafe { nsl_tensor_swiglu_gate_backward_dt(dy, u, g, BF16) });
        assert_eq!(
            bits(unsafe { nsl_tensor_swiglu_dt(g, u, BF16) }),
            chain_p,
            "{dims:?}: forward"
        );
        assert_eq!(
            bits(unsafe { nsl_tensor_swiglu_up_backward_dt(dy, g, BF16) }),
            chain_up,
            "{dims:?}: up"
        );
        let l = unsafe { nsl_tensor_swiglu_backward_dt(dy, u, g, BF16) };
        let (dg, du) = unsafe { (nsl_list_get(l, 0), nsl_list_get(l, 1)) };
        unsafe { nsl_list_free(l) };
        assert_eq!(bits(dg), chain_gate, "{dims:?}: paired dgate");
        assert_eq!(bits(du), chain_up, "{dims:?}: paired dup");
        for t in [g, u, dy] {
            unsafe { nsl_tensor_free(t) };
        }
    }
}
