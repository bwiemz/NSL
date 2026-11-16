//! The `rope` builtin: the fused rotary position embedding
//! `x * cos + rotate_half(x) * sin` and its adjoint, for `x` `[b, h, s, d]`
//! and the `[s, d]` position tables (rows 0..s of the cached cos/sin).
//!
//! On the GPU, one KIR kernel (`nsl_kir::kernels::rope`) reads `x` through its
//! strides (the head view needs no copy) and the tables without expanding
//! them; its f32 arithmetic is the decomposed graph's, rounding for rounding.
//! Everywhere else (the host, other ranks or dtypes, and whenever the autodiff
//! tape is recording, which needs the ops to record themselves) it is the
//! decomposition itself: the same values, more launches.
//!
//! The tables are constants: no gradient flows into them on any path.

use crate::tensor::{NslTensor, DTYPE_BF16, DTYPE_F32};

/// Whether the fused kernel serves `x` with these tables.
#[cfg(feature = "cuda")]
fn fused(x: &NslTensor, cos: &NslTensor, sin: &NslTensor) -> bool {
    let d4 = |i: usize| unsafe { *x.shape.add(i) };
    let t2 = |t: &NslTensor, i: usize| unsafe { *t.shape.add(i) };
    x.device > 0
        && x.ndim == 4
        && (x.dtype == DTYPE_F32 || x.dtype == DTYPE_BF16)
        && x.len > 0
        && x.len < (1i64 << 31)
        && d4(3) % 2 == 0
        && (0..4).all(|i| unsafe { *x.strides.add(i) } >= 0)
        && [cos, sin].iter().all(|t| {
            t.device == x.device
                && t.dtype == DTYPE_F32
                && t.ndim == 2
                && t.is_contiguous()
                && t2(t, 0) == d4(2)
                && t2(t, 1) == d4(3)
        })
}

/// `[s, d]` -> `[1, 1, s, d]` (a view).
fn as_4d(t: i64) -> i64 {
    let tt = NslTensor::from_ptr_ref(t);
    let l = crate::list::nsl_list_new();
    crate::list::nsl_list_push(l, 1);
    crate::list::nsl_list_push(l, 1);
    for i in 0..tt.ndim as usize {
        crate::list::nsl_list_push(l, unsafe { *tt.shape.add(i) });
    }
    let r = crate::tensor::shape_ops::nsl_tensor_reshape(t, l);
    crate::list::nsl_list_free(l);
    r
}

/// The decomposition: forward `x*cos4 + rotate_half(x)*sin4`, backward
/// `rotate_half_neg(dy*sin4) + dy*cos4`. Each op records itself on a
/// recording tape.
fn decomposed(backward: bool, x: i64, cos: i64, sin: i64) -> i64 {
    use crate::tensor::arithmetic::{nsl_tensor_add, nsl_tensor_mul};
    use crate::tensor::shape_ops::{nsl_tensor_rotate_half, nsl_tensor_rotate_half_neg};
    let (c4, s4) = (as_4d(cos), as_4d(sin));
    let r = if backward {
        let a = nsl_tensor_mul(x, c4, 0);
        let z = nsl_tensor_mul(x, s4, 0);
        let rz = nsl_tensor_rotate_half_neg(z);
        let out = nsl_tensor_add(rz, a, 0);
        for t in [a, z, rz] {
            crate::tensor::nsl_tensor_free(t);
        }
        out
    } else {
        let a = nsl_tensor_mul(x, c4, 0);
        let rx = nsl_tensor_rotate_half(x);
        let b = nsl_tensor_mul(rx, s4, 0);
        let out = nsl_tensor_add(a, b, 0);
        for t in [a, rx, b] {
            crate::tensor::nsl_tensor_free(t);
        }
        out
    };
    crate::tensor::nsl_tensor_free(c4);
    crate::tensor::nsl_tensor_free(s4);
    r
}

fn run(backward: bool, x: i64, cos: i64, sin: i64, out_dtype: Option<u16>) -> i64 {
    let tx = NslTensor::from_ptr_ref(x);
    let out = out_dtype.unwrap_or(tx.dtype);
    #[cfg(feature = "cuda")]
    if !crate::autodiff::is_recording()
        && (out == DTYPE_F32 || out == DTYPE_BF16)
        && fused(tx, NslTensor::from_ptr_ref(cos), NslTensor::from_ptr_ref(sin))
    {
        use nsl_kir::kernels::rope::RopeDir;
        use nsl_kir::kernels::storage::Storage;
        let dir = if backward { RopeDir::Backward } else { RopeDir::Forward };
        let st = if out == DTYPE_BF16 { Storage::Bf16 } else { Storage::F32 };
        return crate::cuda::storage_kernels::rope(dir, x, cos, sin, st);
    }
    // The decomposition, in f32 (a bf16 x is widened, the result narrowed to
    // the requested storage: the same values, one rounding).
    let wide = if tx.dtype == DTYPE_BF16 {
        crate::tensor::precision_cast::convert_untaped(x, DTYPE_F32)
    } else {
        tx.refcount.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        x
    };
    let r = decomposed(backward, wide, cos, sin);
    crate::tensor::nsl_tensor_free(wide);
    if out == DTYPE_BF16 && NslTensor::from_ptr_ref(r).dtype != DTYPE_BF16 {
        let n = crate::tensor::precision_cast::convert_untaped(r, DTYPE_BF16);
        crate::tensor::nsl_tensor_free(r);
        return n;
    }
    r
}

/// `rope(x, cos, sin)`: `x * cos + rotate_half(x) * sin`, stored as `x` is.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_rope(x: i64, cos: i64, sin: i64) -> i64 {
    run(false, x, cos, sin, None)
}

/// The adjoint of `rope` in `x`: `dy * cos + rotate_half_neg(dy * sin)`.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_rope_backward(dy: i64, cos: i64, sin: i64) -> i64 {
    run(true, dy, cos, sin, None)
}

/// [`nsl_tensor_rope`] with the result stored as `out_dtype` (bf16
/// activation storage: f32 = 1, bf16 = 3).
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_rope_dt(x: i64, cos: i64, sin: i64, out_dtype: i64) -> i64 {
    run(false, x, cos, sin, Some(out_dtype as u16))
}

/// [`nsl_tensor_rope_backward`] with the result stored as `out_dtype`.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tensor_rope_backward_dt(dy: i64, cos: i64, sin: i64, out_dtype: i64) -> i64 {
    run(true, dy, cos, sin, Some(out_dtype as u16))
}
