#![cfg(feature = "cuda")]
//! The fused RoPE (`rope` builtin) on the GPU: bit for bit the decomposed
//! graph it replaces, `x * cos4 + rotate_half(x) * sin4` and its adjoint
//! `rotate_half_neg(dy * sin4) + dy * cos4`, read through the strided head
//! view; and with a bf16 output, one round-to-nearest-even of that value.
//! (The kernel's mutation, run once, shows the fused path is what computed
//! these: the decomposition would be unaffected by it.)

unsafe extern "C" {
    fn nsl_tensor_from_static(data_ptr: i64, shape_list: i64, dtype: i64) -> i64;
    fn nsl_tensor_to_device(tensor_ptr: i64, target_device: i64) -> i64;
    fn nsl_tensor_transpose(tensor_ptr: i64, dim0: i64, dim1: i64) -> i64;
    fn nsl_tensor_reshape(tensor_ptr: i64, shape_list: i64) -> i64;
    fn nsl_tensor_mul(a: i64, b: i64, flags: u8) -> i64;
    fn nsl_tensor_add(a: i64, b: i64, flags: u8) -> i64;
    fn nsl_tensor_rotate_half(a: i64) -> i64;
    fn nsl_tensor_rotate_half_neg(a: i64) -> i64;
    fn nsl_tensor_rope(x: i64, cos: i64, sin: i64) -> i64;
    fn nsl_tensor_rope_backward(dy: i64, cos: i64, sin: i64) -> i64;
    fn nsl_tensor_rope_dt(x: i64, cos: i64, sin: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_rope_backward_dt(dy: i64, cos: i64, sin: i64, out_dtype: i64) -> i64;
    fn nsl_tensor_to_dtype(t: i64, dtype: i64) -> i64;
    fn nsl_tensor_free(tensor_ptr: i64);
    fn nsl_list_new() -> i64;
    fn nsl_list_push(list_ptr: i64, value: i64);
    fn nsl_list_free(list_ptr: i64);
}

#[repr(C)]
struct Head {
    _magic: u32,
    data: *mut std::ffi::c_void,
    _shape: *mut i64,
    _strides: *mut i64,
    _ndim: i64,
    len: i64,
}

fn list(dims: &[i64]) -> i64 {
    let l = unsafe { nsl_list_new() };
    for &d in dims {
        unsafe { nsl_list_push(l, d) };
    }
    l
}

fn gpu_f32(vals: &[f32], dims: &[i64]) -> i64 {
    let leaked = Box::leak(vals.to_vec().into_boxed_slice()).as_ptr() as i64;
    let l = list(dims);
    let cpu = unsafe { nsl_tensor_from_static(leaked, l, 1) };
    unsafe { nsl_list_free(l) };
    unsafe { nsl_tensor_to_device(cpu, 1) }
}

fn bits(t: i64, width: usize) -> Vec<u32> {
    let host = unsafe { nsl_tensor_to_device(t, 0) };
    let h = unsafe { &*(host as *const Head) };
    let n = h.len as usize;
    let out = if width == 4 {
        unsafe { std::slice::from_raw_parts(h.data as *const u32, n) }.to_vec()
    } else {
        unsafe { std::slice::from_raw_parts(h.data as *const u16, n) }.iter().map(|&b| b as u32).collect()
    };
    unsafe { nsl_tensor_free(host) };
    out
}

fn vals(n: usize, seed: u32, amp: f32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2_654_435_761) | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            ((s >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * amp
        })
        .collect()
}

fn rne_bf16(x: u32) -> u32 {
    (x + 0x7FFF + ((x >> 16) & 1)) >> 16
}

#[test]
#[ignore = "requires CUDA GPU"]
fn the_fused_rope_is_the_decomposed_graph_bit_for_bit() {
    assert_eq!(nsl_runtime::nsl_cuda_init(), 0);
    let (b, h, s, d) = (2i64, 3i64, 37i64, 64i64);
    // The head view: [b, s, h, d] stored, transposed to [b, h, s, d].
    let x_store = gpu_f32(&vals((b * s * h * d) as usize, 3, 1.5), &[b, s, h, d]);
    let x = unsafe { nsl_tensor_transpose(x_store, 1, 2) };
    let dy = gpu_f32(&vals((b * h * s * d) as usize, 5, 1e-6), &[b, h, s, d]);
    let cos = gpu_f32(&vals((s * d) as usize, 7, 1.0), &[s, d]);
    let sin = gpu_f32(&vals((s * d) as usize, 9, 1.0), &[s, d]);
    let l = list(&[1, 1, s, d]);
    let (c4, s4) = unsafe { (nsl_tensor_reshape(cos, l), nsl_tensor_reshape(sin, l)) };
    unsafe { nsl_list_free(l) };

    // Forward.
    let fused = unsafe { nsl_tensor_rope(x, cos, sin) };
    let want = unsafe {
        let a = nsl_tensor_mul(x, c4, 0);
        let r = nsl_tensor_rotate_half(x);
        let m = nsl_tensor_mul(r, s4, 0);
        nsl_tensor_add(a, m, 0)
    };
    let (got_f, want_f) = (bits(fused, 4), bits(want, 4));
    let bad = got_f.iter().zip(&want_f).filter(|(a, b)| a != b).count();
    assert_eq!(bad, 0, "rope forward: {bad} of {} elements differ from the decomposed graph", want_f.len());

    // Adjoint.
    let fused_b = unsafe { nsl_tensor_rope_backward(dy, cos, sin) };
    let want_b = unsafe {
        let a = nsl_tensor_mul(dy, c4, 0);
        let z = nsl_tensor_mul(dy, s4, 0);
        let rz = nsl_tensor_rotate_half_neg(z);
        nsl_tensor_add(rz, a, 0)
    };
    let (got_b, want_bb) = (bits(fused_b, 4), bits(want_b, 4));
    let bad = got_b.iter().zip(&want_bb).filter(|(a, b)| a != b).count();
    assert_eq!(bad, 0, "rope adjoint: {bad} of {} elements differ from the decomposed graph", want_bb.len());

    // bf16 activation storage: a bf16 head view in, one rounding out.
    let x16 = unsafe { nsl_tensor_to_dtype(x_store, 3) };
    let x16v = unsafe { nsl_tensor_transpose(x16, 1, 2) };
    let x16w = unsafe { nsl_tensor_to_dtype(x16, 1) }; // the same values, f32
    let x16wv = unsafe { nsl_tensor_transpose(x16w, 1, 2) };
    let ref32 = bits(unsafe { nsl_tensor_rope(x16wv, cos, sin) }, 4);
    let got16 = bits(unsafe { nsl_tensor_rope_dt(x16v, cos, sin, 3) }, 2);
    let bad = got16.iter().zip(&ref32).filter(|(g, r)| **g != rne_bf16(**r)).count();
    assert_eq!(bad, 0, "bf16 rope: {bad} of {} values are not one RNE rounding of the f32 result", ref32.len());
    let dy16 = unsafe { nsl_tensor_to_dtype(dy, 3) };
    let dy16w = unsafe { nsl_tensor_to_dtype(dy16, 1) };
    let ref32b = bits(unsafe { nsl_tensor_rope_backward(dy16w, cos, sin) }, 4);
    let got16b = bits(unsafe { nsl_tensor_rope_backward_dt(dy16, cos, sin, 3) }, 2);
    let bad = got16b.iter().zip(&ref32b).filter(|(g, r)| **g != rne_bf16(**r)).count();
    assert_eq!(bad, 0, "bf16 rope adjoint: {bad} of {} values are not one RNE rounding", ref32b.len());
}

/// The strides of a tensor handle.
fn strides(t: i64) -> Vec<i64> {
    let h = unsafe { &*(t as *const Head) };
    (0..h._ndim as usize).map(|i| unsafe { *h._strides.add(i) }).collect()
}

/// RoPE keeps its input's dense layout: the `[b, s, h, d]` head view of a
/// projection comes back as the same kind of view (so the attention reads it
/// in place, and the adjoint's result goes back into the projection's GEMMs
/// without a transpose copy), a contiguous input as a contiguous result. In
/// both layouts, for every head dim the vector kernel serves (a multiple of
/// 16, so each half row is whole 8-element runs: 32, 48, 64, 128) and every
/// storage pair, the values are the decomposed graph's bit for bit (one
/// rounding for a bf16 result), as is the strided kernel's for a head dim it
/// does not serve (24).
#[test]
#[ignore = "requires CUDA GPU"]
fn rope_keeps_the_head_layout_and_the_bits() {
    assert_eq!(nsl_runtime::nsl_cuda_init(), 0);
    let bshd_on = std::env::var("NSL_ATTN_BSHD").ok().as_deref() != Some("0");
    for (b, h, s, d) in [(2i64, 3i64, 37i64, 64i64), (1, 5, 19, 128), (3, 2, 11, 32), (2, 3, 7, 48), (2, 2, 9, 24)] {
        let served = d % 16 == 0;
        let x_store = gpu_f32(&vals((b * s * h * d) as usize, 13, 1.5), &[b, s, h, d]);
        let x = unsafe { nsl_tensor_transpose(x_store, 1, 2) };
        let dy = gpu_f32(&vals((b * h * s * d) as usize, 17, 1e-6), &[b, h, s, d]);
        let cos = gpu_f32(&vals((s * d) as usize, 19, 1.0), &[s, d]);
        let sin = gpu_f32(&vals((s * d) as usize, 23, 1.0), &[s, d]);
        let l = list(&[1, 1, s, d]);
        let (c4, s4) = unsafe { (nsl_tensor_reshape(cos, l), nsl_tensor_reshape(sin, l)) };
        unsafe { nsl_list_free(l) };
        let want_f = bits(unsafe { nsl_tensor_add(nsl_tensor_mul(x, c4, 0), nsl_tensor_mul(nsl_tensor_rotate_half(x), s4, 0), 0) }, 4);
        let want_b = bits(
            unsafe { nsl_tensor_add(nsl_tensor_rotate_half_neg(nsl_tensor_mul(dy, s4, 0)), nsl_tensor_mul(dy, c4, 0), 0) },
            4,
        );
        let bshd = [s * h * d, d, h * d, 1];
        let contiguous = [h * s * d, s * d, d, 1];
        for (in_dt, out_dt) in [(1i64, 1i64), (3, 3), (3, 1), (1, 3)] {
            let tag = format!("[{b}, {h}, {s}, {d}] {in_dt}->{out_dt}");
            let (xi, dyi) = if in_dt == 3 {
                unsafe { (nsl_tensor_transpose(nsl_tensor_to_dtype(x_store, 3), 1, 2), nsl_tensor_to_dtype(dy, 3)) }
            } else {
                (x, dy)
            };
            // The reference for a bf16 input reads the same bf16 values.
            let (ref_f, ref_b) = if in_dt == 3 {
                let xw = unsafe { nsl_tensor_to_dtype(xi, 1) };
                let dyw = unsafe { nsl_tensor_to_dtype(dyi, 1) };
                (bits(unsafe { nsl_tensor_rope(xw, cos, sin) }, 4), bits(unsafe { nsl_tensor_rope_backward(dyw, cos, sin) }, 4))
            } else {
                (want_f.clone(), want_b.clone())
            };
            let got_f = unsafe { nsl_tensor_rope_dt(xi, cos, sin, out_dt) };
            let got_b = unsafe { nsl_tensor_rope_backward_dt(dyi, cos, sin, out_dt) };
            let want_layout_f = if served && bshd_on { bshd } else { contiguous };
            assert_eq!(strides(got_f), want_layout_f, "{tag}: forward result layout");
            assert_eq!(strides(got_b), contiguous, "{tag}: adjoint of a contiguous gradient");
            for (what, got, want) in [("forward", got_f, &ref_f), ("adjoint", got_b, &ref_b)] {
                let g = bits(got, if out_dt == 3 { 2 } else { 4 });
                let bad = g
                    .iter()
                    .zip(want.iter())
                    .filter(|(g, w)| if out_dt == 3 { **g != rne_bf16(**w) } else { g != w })
                    .count();
                assert_eq!(bad, 0, "{tag} {what}: {bad} of {} values differ", want.len());
            }
        }
        // The adjoint of a [b, s, h, d] gradient (the attention backward's
        // dQ/dK) keeps that layout, with the decomposed graph's values.
        let dy_store = gpu_f32(&vals((b * s * h * d) as usize, 29, 1e-6), &[b, s, h, d]);
        let dyv = unsafe { nsl_tensor_transpose(dy_store, 1, 2) };
        let want = bits(
            unsafe { nsl_tensor_add(nsl_tensor_rotate_half_neg(nsl_tensor_mul(dyv, s4, 0)), nsl_tensor_mul(dyv, c4, 0), 0) },
            4,
        );
        let got = unsafe { nsl_tensor_rope_backward_dt(dyv, cos, sin, 1) };
        assert_eq!(strides(got), if served && bshd_on { bshd } else { contiguous }, "[{b}, {h}, {s}, {d}]: adjoint layout");
        let g = bits(got, 4);
        let bad = g.iter().zip(&want).filter(|(g, w)| g != w).count();
        assert_eq!(bad, 0, "[{b}, {h}, {s}, {d}] adjoint of a head view: {bad} of {} values differ", want.len());
    }
}
