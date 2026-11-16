#![cfg(all(feature = "cuda", feature = "test-hooks"))]
//! `nsl_tensor_contiguous` of bf16 GPU views (bf16 activation storage's
//! transposes, reshapes and GQA expands): bit for bit the view's elements in
//! row-major order. Views with an even, contiguous innermost axis take the
//! paired route (the f32 run copy moving bf16 pairs as words); the others take
//! the counted widen/copy/narrow route. The counter says which, so each case
//! asserts the route it covers.

use nsl_runtime::test_storage_copy_regional_count;

unsafe extern "C" {
    fn nsl_tensor_from_static(data_ptr: i64, shape_list: i64, dtype: i64) -> i64;
    fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64;
    fn nsl_tensor_transpose(tensor_ptr: i64, dim0: i64, dim1: i64) -> i64;
    fn nsl_tensor_expand(tensor_ptr: i64, shape_list: i64) -> i64;
    fn nsl_tensor_contiguous(tensor_ptr: i64) -> i64;
    fn nsl_tensor_free(tensor_ptr: i64);
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
    fn nsl_list_free(list_ptr: i64);
}

const BF16: i64 = 3;

#[repr(C)]
struct Head {
    _magic: u32,
    data: *mut std::ffi::c_void,
    shape: *mut i64,
    strides: *mut i64,
    ndim: i64,
    len: i64,
    _refcount: std::sync::atomic::AtomicI64,
    device: u8,
    dtype: u16,
}

fn head(p: i64) -> &'static Head {
    unsafe { &*(p as *const Head) }
}

fn list(dims: &[i64]) -> i64 {
    let l = unsafe { nsl_list_new() };
    for &d in dims {
        unsafe { nsl_list_push(l, d) };
    }
    l
}

/// A bf16 GPU tensor of `bits`, row-major `dims`.
fn upload(bits: &[u16], dims: &[i64]) -> i64 {
    let leaked = Box::leak(bits.to_vec().into_boxed_slice()).as_ptr() as i64;
    let l = list(dims);
    let cpu = unsafe { nsl_tensor_from_static(leaked, l, BF16) };
    unsafe { nsl_list_free(l) };
    unsafe { nsl_tensor_to_device(cpu, 1) }
}

fn download(t: i64) -> Vec<u16> {
    let h = head(t);
    assert_eq!((h.device, h.dtype), (1, BF16 as u16), "a bf16 GPU result");
    let host = unsafe { nsl_tensor_to_device(t, 0) };
    let hh = head(host);
    let out = unsafe { std::slice::from_raw_parts(hh.data as *const u16, hh.len as usize) }.to_vec();
    unsafe { nsl_tensor_free(host) };
    out
}

/// The view's elements in row-major order, read from the source `bits`
/// through the view's own shape and strides.
fn gather(src: &[u16], view: i64) -> Vec<u16> {
    let h = head(view);
    let nd = h.ndim as usize;
    let shape: Vec<i64> = (0..nd).map(|i| unsafe { *h.shape.add(i) }).collect();
    let strides: Vec<i64> = (0..nd).map(|i| unsafe { *h.strides.add(i) }).collect();
    let n: i64 = shape.iter().product();
    (0..n)
        .map(|mut lin| {
            let mut off = 0i64;
            for d in (0..nd).rev() {
                off += (lin % shape[d]) * strides[d];
                lin /= shape[d];
            }
            src[off as usize]
        })
        .collect()
}

fn bits(n: usize, seed: u32) -> Vec<u16> {
    // Distinct, finite bf16 patterns: any misplaced element shows.
    (0..n as u32).map(|i| (0x3C00 + ((i.wrapping_mul(2_654_435_761) ^ seed) % 0x0FFF)) as u16).collect()
}

fn check(what: &str, src: &[u16], view: i64, paired: bool) {
    let before = test_storage_copy_regional_count();
    let c = unsafe { nsl_tensor_contiguous(view) };
    let after = test_storage_copy_regional_count();
    assert_eq!(after - before, u64::from(!paired), "{what}: took the wrong route (paired expected: {paired})");
    let got = download(c);
    let want = gather(src, view);
    let bad = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert_eq!(got.len(), want.len(), "{what}: length");
    assert_eq!(bad, 0, "{what}: {bad} of {} elements differ", want.len());
    unsafe { nsl_tensor_free(c) };
}

#[test]
#[ignore = "requires CUDA GPU"]
fn bf16_views_copy_to_row_major_bit_for_bit() {
    assert_eq!(nsl_runtime::nsl_cuda_init(), 0);

    // Attention: [b, s, h, d] -> transpose(1, 2) (q/k/v, and the output back).
    let src = bits(2 * 6 * 8 * 64, 7);
    let t = upload(&src, &[2, 6, 8, 64]);
    let v = unsafe { nsl_tensor_transpose(t, 1, 2) };
    check("transpose(1,2) of [2,6,8,64]", &src, v, true);
    unsafe { nsl_tensor_free(v) };
    unsafe { nsl_tensor_free(t) };

    // GQA: [b, kvh, 1, s, d] expanded over the group axis.
    let src = bits(2 * 4 * 6 * 128, 9);
    let t = upload(&src, &[2, 4, 1, 6, 128]);
    let l = list(&[2, 4, 3, 6, 128]);
    let v = unsafe { nsl_tensor_expand(t, l) };
    unsafe { nsl_list_free(l) };
    check("expand [2,4,1,6,128] -> [2,4,3,6,128]", &src, v, true);
    unsafe { nsl_tensor_free(v) };
    unsafe { nsl_tensor_free(t) };

    // The innermost axis transposed: the regional route.
    let src = bits(3 * 5 * 7, 11);
    let t = upload(&src, &[3, 5, 7]);
    let v = unsafe { nsl_tensor_transpose(t, 1, 2) };
    check("transpose(1,2) of [3,5,7]", &src, v, false);
    unsafe { nsl_tensor_free(v) };
    unsafe { nsl_tensor_free(t) };

    // An odd innermost length behind a transposed outer pair: regional too.
    let src = bits(4 * 6 * 9, 13);
    let t = upload(&src, &[4, 6, 9]);
    let v = unsafe { nsl_tensor_transpose(t, 0, 1) };
    check("transpose(0,1) of [4,6,9]", &src, v, false);
    unsafe { nsl_tensor_free(v) };
    unsafe { nsl_tensor_free(t) };
}
