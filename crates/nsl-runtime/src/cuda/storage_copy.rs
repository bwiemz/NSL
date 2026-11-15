//! Contiguous copies of bf16 GPU views (`nsl_tensor_contiguous` on a 2-byte
//! tensor): the transposes, reshapes and GQA `expand`s of bf16 activations.
//!
//! **Paired:** a view whose innermost axis is contiguous with an even length,
//! whose other strides are even (in elements) and whose base is 4-byte aligned
//! is, bit for bit, an f32 view of half the width: every adjacent bf16 pair is
//! one f32 word, and the view's runs are runs of whole words. The existing f32
//! run-copy kernels move it exactly; the result is relabelled bf16. Every bf16
//! view of the attention half is of this kind (head_dim 64 or 128 innermost).
//!
//! **Otherwise** (an odd innermost length, a transposed innermost axis, a
//! 2-byte-aligned base): the source region the view spans is widened to f32
//! (exact), copied by the f32 strided copy, and narrowed (exact again, the
//! values are bf16). Three passes instead of one; counted and announced once.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use crate::tensor::{NslTensor, DTYPE_BF16, DTYPE_F32};

static REGIONAL: AtomicU64 = AtomicU64::new(0);
static ANNOUNCED: AtomicBool = AtomicBool::new(false);

/// How many bf16 contiguous copies took the widen/copy/narrow path (a gate
/// reads it through `test_storage_copy_regional_count`).
#[cfg(feature = "test-hooks")]
pub(crate) fn regional_count() -> u64 {
    REGIONAL.load(Ordering::Relaxed)
}

/// A non-owning view tensor over `data` (freed with `nsl_tensor_free`, which
/// releases the shape and strides arrays but not `data`).
fn view(data: *mut c_void, shape: &[i64], strides: &[i64], len: i64, device: u8, dtype: u16) -> i64 {
    let nd = shape.len();
    let sh = crate::memory::checked_alloc(std::mem::size_of_val(shape)) as *mut i64;
    let st = crate::memory::checked_alloc(std::mem::size_of_val(strides)) as *mut i64;
    for i in 0..nd {
        unsafe {
            *sh.add(i) = shape[i];
            *st.add(i) = strides[i];
        }
    }
    Box::into_raw(Box::new(NslTensor::new(data, sh, st, nd as i64, len, device, dtype, 0, 0))) as i64
}

/// A row-major bf16 copy of the bf16 GPU view `tensor_ptr`.
pub(crate) fn contiguous_bf16(tensor_ptr: i64) -> i64 {
    let t = NslTensor::from_ptr_ref(tensor_ptr);
    assert_eq!(t.dtype, DTYPE_BF16);
    let nd = t.ndim as usize;
    let shape: Vec<i64> = (0..nd).map(|i| unsafe { *t.shape.add(i) }).collect();
    let strides: Vec<i64> = (0..nd).map(|i| unsafe { *t.strides.add(i) }).collect();
    if let Some(out) = paired(t, &shape, &strides) {
        return out;
    }
    regional(t, &shape, &strides)
}

fn paired(t: &NslTensor, shape: &[i64], strides: &[i64]) -> Option<i64> {
    let nd = shape.len();
    if nd == 0 || t.len == 0 {
        return None;
    }
    let last = nd - 1;
    let pairable = strides[last] == 1
        && shape[last] % 2 == 0
        && (t.data as usize).is_multiple_of(4)
        && strides[..last].iter().zip(&shape[..last]).all(|(s, n)| *n == 1 || (*s >= 0 && s % 2 == 0));
    if !pairable {
        return None;
    }
    let mut half_shape = shape.to_vec();
    half_shape[last] /= 2;
    let mut half_strides: Vec<i64> = strides.iter().zip(shape).map(|(s, n)| if *n == 1 { 0 } else { s / 2 }).collect();
    half_strides[last] = 1;
    let v = view(t.data, &half_shape, &half_strides, t.len / 2, t.device, DTYPE_F32);
    let f = super::gpu_strided_copy_f32(v);
    crate::tensor::nsl_tensor_free(v);
    // Relabel the f32 result as the bf16 tensor it is, bit for bit.
    let out = NslTensor::from_ptr(f);
    out.dtype = DTYPE_BF16;
    out.len = t.len;
    let mut stride = 1i64;
    for i in (0..nd).rev() {
        unsafe {
            *out.shape.add(i) = shape[i];
            *out.strides.add(i) = stride;
        }
        stride *= shape[i];
    }
    Some(f)
}

fn regional(t: &NslTensor, shape: &[i64], strides: &[i64]) -> i64 {
    if strides.iter().any(|s| *s < 0) {
        crate::fatal::die(crate::fatal::Fatal::Unsupported, "contiguous copy of a bf16 view with a negative stride");
    }
    REGIONAL.fetch_add(1, Ordering::Relaxed);
    if !ANNOUNCED.swap(true, Ordering::Relaxed) {
        crate::nsl_log!(
            INFO,
            "bf16-storage",
            "a bf16 view without an even contiguous innermost axis is copied by widening, copying \
             and narrowing (exact, three passes); further cases are counted, not printed"
        );
    }
    let region: i64 = 1 + shape.iter().zip(strides).map(|(n, s)| (n - 1) * s).sum::<i64>();
    let wide = super::inner::alloc_managed(region as usize * 4);
    super::gpu_cast_raw_bf16_to_f32(t.data as u64, wide as u64, region as usize);
    let v = view(wide, shape, strides, t.len, t.device, DTYPE_F32);
    let f = super::gpu_strided_copy_f32(v);
    crate::tensor::nsl_tensor_free(v);
    super::inner::free_managed(wide);
    let fd = NslTensor::from_ptr_ref(f);
    let narrow = super::inner::alloc_managed(t.len as usize * 2);
    super::gpu_cast_raw_f32_to_bf16(fd.data as u64, narrow as u64, t.len as usize);
    let nd = shape.len() as i64;
    let sh = NslTensor::copy_shape(shape.as_ptr(), nd);
    let st = NslTensor::compute_strides(sh, nd);
    crate::tensor::nsl_tensor_free(f);
    Box::into_raw(Box::new(NslTensor::new(narrow, sh, st, nd, t.len, t.device, DTYPE_BF16, 1, 0))) as i64
}
