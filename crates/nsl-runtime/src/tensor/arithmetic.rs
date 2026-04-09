//! Tensor arithmetic operations: add, sub, mul, div, neg, scalar ops, matmul.

use std::ffi::c_void;
use std::sync::atomic::Ordering;

use crate::autodiff;
use crate::memory::{checked_alloc, checked_alloc_zeroed};

use super::{get_shape_vec, nsl_tensor_contiguous, nsl_tensor_free, nsl_tensor_to_device, NslTensor};

fn tensor_elementwise_op(a_ptr: i64, b_ptr: i64, op: fn(f64, f64) -> f64) -> i64 {
    crate::cpu::tensor_elementwise_op(a_ptr, b_ptr, op)
}

/// If `b` is on a different device than `a`, transfer `b` to `a`'s device.
/// Returns `(effective_b_ptr, true)` if a transfer was made (caller must free),
/// or `(b, false)` if no transfer was needed.
fn reconcile_device(a: i64, b: i64) -> (i64, bool) {
    let ta = unsafe { &*(a as *const NslTensor) };
    let tb = unsafe { &*(b as *const NslTensor) };
    if tb.device != ta.device {
        (nsl_tensor_to_device(b, ta.device as i64), true)
    } else {
        (b, false)
    }
}

#[no_mangle]
pub extern "C" fn nsl_tensor_add(a: i64, b: i64, flags: u8) -> i64 {
    use crate::tensor::fbip_flags::{relinquish_a, relinquish_b};
    let relinq_a = relinquish_a(flags);
    let relinq_b = relinquish_b(flags);
    // Capture caller's original B pointer BEFORE reconcile_device shadows `b`.
    // When reconcile_device performs a cross-device transfer it returns a
    // fresh runtime-owned allocation; the caller's original `b` still needs
    // to be freed if they relinquished it. See ELTLS commit 1b.1 fix.
    let b_orig = b;
    let (b, b_transferred) = reconcile_device(a, b);
    {
        let ta = unsafe { &*(a as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let tb = unsafe { &*(b as *const NslTensor) };
                if (relinq_a || ta.can_mutate_inplace_gpu()) && ta.shape_eq(tb) {
                    crate::cuda::gpu_elementwise_binary_inplace(a, b, crate::cuda::kernels::ADD_F32_PTX, "nsl_add_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    // FBIP-3: free caller's original B if relinquished.
                    // Free the runtime-owned reconciled copy if we created one.
                    // When b_transferred=false, b == b_orig, so relinq_b already
                    // freed it and the b_transferred branch is a no-op.
                    if relinq_b {
                        nsl_tensor_free(b_orig);
                    }
                    if b_transferred {
                        nsl_tensor_free(b);
                    }
                    return a;
                }
                let result = crate::cuda::gpu_elementwise_binary(a, b, crate::cuda::kernels::ADD_F32_PTX, "nsl_add_f32\0");
                // Out-of-place GPU fallback: honor caller relinquish flags.
                if relinq_a {
                    nsl_tensor_free(a);
                }
                if relinq_b {
                    nsl_tensor_free(b_orig);
                }
                if b_transferred {
                    nsl_tensor_free(b);
                }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: reuse left operand when uniquely owned + same shape (no broadcast, CPU).
    // FBIP-3: the relinquish flag is the caller's promise that no other holder exists,
    // so the in-place path is safe even if the refcount heuristic would normally reject it.
    {
        let ta = unsafe { &mut *(a as *mut NslTensor) };
        let tb = unsafe { &*(b as *const NslTensor) };
        if (relinq_a || ta.can_mutate_inplace()) && ta.shape_eq(tb) && ta.dtype == tb.dtype && tb.is_contiguous() {
            let len = ta.len as usize;
            if ta.dtype == 1 {
                let da = ta.data as *mut f32;
                let db = tb.data as *const f32;
                for i in 0..len { unsafe { *da.add(i) += *db.add(i) }; }
            } else {
                let da = ta.data as *mut f64;
                let db = tb.data as *const f64;
                for i in 0..len { unsafe { *da.add(i) += *db.add(i) }; }
            }
            ta.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            // FBIP-3 double-free-safe: even on the in-place-on-A branch, if B was
            // relinquished the caller's original must still be freed (the critical
            // bug being prevented). Runtime-owned reconciled copy freed separately.
            if relinq_b {
                nsl_tensor_free(b_orig);
            }
            if b_transferred {
                nsl_tensor_free(b);
            }
            return a;
        }
    }
    super::fbip_record_alloc();
    // Ensure contiguous inputs for flat-indexed CPU ops
    let a_c = nsl_tensor_contiguous(a);
    let b_c = nsl_tensor_contiguous(b);
    let a_shape = get_shape_vec(NslTensor::from_ptr(a_c));
    let b_shape = get_shape_vec(NslTensor::from_ptr(b_c));
    let result = tensor_elementwise_op(a_c, b_c, |x, y| x + y);
    nsl_tensor_free(a_c);
    nsl_tensor_free(b_c);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Add { a, b, out: result, a_shape, b_shape });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Add, vec![a, b], result, shape, rt.dtype, vec![]);
    }
    // FBIP-3: out-of-place fallback — honor caller relinquish flags.
    if relinq_a {
        nsl_tensor_free(a);
    }
    if relinq_b {
        nsl_tensor_free(b_orig);
    }
    if b_transferred {
        nsl_tensor_free(b);
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sub(a: i64, b: i64, flags: u8) -> i64 {
    use crate::tensor::fbip_flags::{relinquish_a, relinquish_b};
    let relinq_a = relinquish_a(flags);
    let relinq_b = relinquish_b(flags);
    // Capture caller's original B before reconcile_device shadows `b` (ELTLS 1b.1).
    let b_orig = b;
    let (b, b_transferred) = reconcile_device(a, b);
    {
        let ta = unsafe { &*(a as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let tb = unsafe { &*(b as *const NslTensor) };
                if (relinq_a || ta.can_mutate_inplace_gpu()) && ta.shape_eq(tb) {
                    crate::cuda::gpu_elementwise_binary_inplace(a, b, crate::cuda::kernels::SUB_F32_PTX, "nsl_sub_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    if relinq_b { nsl_tensor_free(b_orig); }
                    if b_transferred { nsl_tensor_free(b); }
                    return a;
                }
                let result = crate::cuda::gpu_elementwise_binary(a, b, crate::cuda::kernels::SUB_F32_PTX, "nsl_sub_f32\0");
                if relinq_a { nsl_tensor_free(a); }
                if relinq_b { nsl_tensor_free(b_orig); }
                if b_transferred { nsl_tensor_free(b); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: reuse left operand when uniquely owned + same shape (CPU).
    // FBIP-3: relinq_a grants in-place reuse even if the refcount heuristic rejects it.
    {
        let ta = unsafe { &mut *(a as *mut NslTensor) };
        let tb = unsafe { &*(b as *const NslTensor) };
        if (relinq_a || ta.can_mutate_inplace()) && ta.shape_eq(tb) && ta.dtype == tb.dtype && tb.is_contiguous() {
            let len = ta.len as usize;
            if ta.dtype == 1 {
                let da = ta.data as *mut f32;
                let db = tb.data as *const f32;
                for i in 0..len { unsafe { *da.add(i) -= *db.add(i) }; }
            } else {
                let da = ta.data as *mut f64;
                let db = tb.data as *const f64;
                for i in 0..len { unsafe { *da.add(i) -= *db.add(i) }; }
            }
            ta.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            if relinq_b { nsl_tensor_free(b_orig); }
            if b_transferred { nsl_tensor_free(b); }
            return a;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(a);
    let b_c = nsl_tensor_contiguous(b);
    let a_shape = get_shape_vec(NslTensor::from_ptr(a_c));
    let b_shape = get_shape_vec(NslTensor::from_ptr(b_c));
    let result = tensor_elementwise_op(a_c, b_c, |x, y| x - y);
    nsl_tensor_free(a_c);
    nsl_tensor_free(b_c);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Sub { a, b, out: result, a_shape, b_shape });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Sub, vec![a, b], result, shape, rt.dtype, vec![]);
    }
    // FBIP-3: out-of-place fallback — honor caller relinquish flags.
    if relinq_a { nsl_tensor_free(a); }
    if relinq_b { nsl_tensor_free(b_orig); }
    if b_transferred { nsl_tensor_free(b); }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_mul(a: i64, b: i64, flags: u8) -> i64 {
    use crate::tensor::fbip_flags::{relinquish_a, relinquish_b};
    let relinq_a = relinquish_a(flags);
    let relinq_b = relinquish_b(flags);
    let b_orig = b;
    let (b, b_transferred) = reconcile_device(a, b);
    {
        let ta = unsafe { &*(a as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let tb = unsafe { &*(b as *const NslTensor) };
                if (relinq_a || ta.can_mutate_inplace_gpu()) && ta.shape_eq(tb) {
                    crate::cuda::gpu_elementwise_binary_inplace(a, b, crate::cuda::kernels::MUL_F32_PTX, "nsl_mul_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    if relinq_b { nsl_tensor_free(b_orig); }
                    if b_transferred { nsl_tensor_free(b); }
                    return a;
                }
                let result = crate::cuda::gpu_elementwise_binary(a, b, crate::cuda::kernels::MUL_F32_PTX, "nsl_mul_f32\0");
                if relinq_a { nsl_tensor_free(a); }
                if relinq_b { nsl_tensor_free(b_orig); }
                if b_transferred { nsl_tensor_free(b); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: reuse left operand when uniquely owned + same shape (CPU).
    // FBIP-3: relinq_a grants in-place reuse even if the refcount heuristic rejects it.
    {
        let ta = unsafe { &mut *(a as *mut NslTensor) };
        let tb = unsafe { &*(b as *const NslTensor) };
        if (relinq_a || ta.can_mutate_inplace()) && ta.shape_eq(tb) && ta.dtype == tb.dtype && tb.is_contiguous() {
            let len = ta.len as usize;
            if ta.dtype == 1 {
                let da = ta.data as *mut f32;
                let db = tb.data as *const f32;
                for i in 0..len { unsafe { *da.add(i) *= *db.add(i) }; }
            } else {
                let da = ta.data as *mut f64;
                let db = tb.data as *const f64;
                for i in 0..len { unsafe { *da.add(i) *= *db.add(i) }; }
            }
            ta.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            if relinq_b { nsl_tensor_free(b_orig); }
            if b_transferred { nsl_tensor_free(b); }
            return a;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(a);
    let b_c = nsl_tensor_contiguous(b);
    let a_shape = get_shape_vec(NslTensor::from_ptr(a_c));
    let b_shape = get_shape_vec(NslTensor::from_ptr(b_c));
    let result = tensor_elementwise_op(a_c, b_c, |x, y| x * y);
    nsl_tensor_free(a_c);
    nsl_tensor_free(b_c);
    if autodiff::is_recording() {
        NslTensor::from_ptr(a).refcount.fetch_add(1, Ordering::SeqCst);
        NslTensor::from_ptr(b).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::Mul {
            a,
            b,
            out: result,
            saved_a: a,
            saved_b: b,
            a_shape,
            b_shape,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Mul, vec![a, b], result, shape, rt.dtype, vec![]);
    }
    // FBIP-3: out-of-place fallback — honor caller relinquish flags.
    if relinq_a { nsl_tensor_free(a); }
    if relinq_b { nsl_tensor_free(b_orig); }
    if b_transferred { nsl_tensor_free(b); }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_div(a: i64, b: i64, flags: u8) -> i64 {
    use crate::tensor::fbip_flags::{relinquish_a, relinquish_b};
    let relinq_a = relinquish_a(flags);
    let relinq_b = relinquish_b(flags);
    let b_orig = b;
    let (b, b_transferred) = reconcile_device(a, b);
    {
        let ta = unsafe { &*(a as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let tb = unsafe { &*(b as *const NslTensor) };
                if (relinq_a || ta.can_mutate_inplace_gpu()) && ta.shape_eq(tb) {
                    crate::cuda::gpu_elementwise_binary_inplace(a, b, crate::cuda::kernels::DIV_F32_PTX, "nsl_div_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    if relinq_b { nsl_tensor_free(b_orig); }
                    if b_transferred { nsl_tensor_free(b); }
                    return a;
                }
                let result = crate::cuda::gpu_elementwise_binary(a, b, crate::cuda::kernels::DIV_F32_PTX, "nsl_div_f32\0");
                if relinq_a { nsl_tensor_free(a); }
                if relinq_b { nsl_tensor_free(b_orig); }
                if b_transferred { nsl_tensor_free(b); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: reuse left operand when uniquely owned + same shape (CPU).
    // FBIP-3: relinq_a grants in-place reuse even if the refcount heuristic rejects it.
    {
        let ta = unsafe { &mut *(a as *mut NslTensor) };
        let tb = unsafe { &*(b as *const NslTensor) };
        if (relinq_a || ta.can_mutate_inplace()) && ta.shape_eq(tb) && ta.dtype == tb.dtype && tb.is_contiguous() {
            let len = ta.len as usize;
            if ta.dtype == 1 {
                let da = ta.data as *mut f32;
                let db = tb.data as *const f32;
                for i in 0..len { unsafe { *da.add(i) /= *db.add(i) }; }
            } else {
                let da = ta.data as *mut f64;
                let db = tb.data as *const f64;
                for i in 0..len { unsafe { *da.add(i) /= *db.add(i) }; }
            }
            ta.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            if relinq_b { nsl_tensor_free(b_orig); }
            if b_transferred { nsl_tensor_free(b); }
            return a;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(a);
    let b_c = nsl_tensor_contiguous(b);
    let a_shape = get_shape_vec(NslTensor::from_ptr(a_c));
    let b_shape = get_shape_vec(NslTensor::from_ptr(b_c));
    let result = tensor_elementwise_op(a_c, b_c, |x, y| x / y);
    nsl_tensor_free(a_c);
    nsl_tensor_free(b_c);
    if autodiff::is_recording() {
        NslTensor::from_ptr(a).refcount.fetch_add(1, Ordering::SeqCst);
        NslTensor::from_ptr(b).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::Div {
            a,
            b,
            out: result,
            saved_a: a,
            saved_b: b,
            a_shape,
            b_shape,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Div, vec![a, b], result, shape, rt.dtype, vec![]);
    }
    // FBIP-3: out-of-place fallback — honor caller relinquish flags.
    if relinq_a { nsl_tensor_free(a); }
    if relinq_b { nsl_tensor_free(b_orig); }
    if b_transferred { nsl_tensor_free(b); }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_neg(a_ptr: i64) -> i64 {
    {
        let ta = unsafe { &*(a_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_elementwise_unary_inplace(a_ptr, crate::cuda::kernels::NEG_F32_PTX, "nsl_neg_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return a_ptr;
                }
                return crate::cuda::gpu_elementwise_unary(a_ptr, crate::cuda::kernels::NEG_F32_PTX, "nsl_neg_f32\0");
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned (CPU)
    {
        let t = unsafe { &mut *(a_ptr as *mut NslTensor) };
        if t.can_mutate_inplace() {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                for i in 0..len { unsafe { *d.add(i) = -(*d.add(i)) }; }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len { unsafe { *d.add(i) = -(*d.add(i)) }; }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return a_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(a_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = -(*a.data_f32().add(i)) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = -(*a.data_f64().add(i)) };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor::new(
        data,
        shape,
        strides,
        ndim,
        len,
        a.device,
        a.dtype,
        1,
        0,
    ));
    let result = NslTensor::publish(result);
    nsl_tensor_free(a_c);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Neg { a: a_ptr, out: result });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::Neg, vec![a_ptr], result, shape, rt.dtype, vec![]);
    }
    result
}

// === Scalar-tensor ops ===

#[no_mangle]
pub extern "C" fn nsl_tensor_add_scalar(a_ptr: i64, s: f64, flags: u8) -> i64 {
    use crate::tensor::fbip_flags::relinquish_a;
    let relinq_a = relinquish_a(flags);
    {
        let ta = unsafe { &*(a_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if relinq_a || ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_scalar_op_inplace(a_ptr, s as f32, crate::cuda::kernels::ADD_SCALAR_F32_PTX, "nsl_add_scalar_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return a_ptr;
                }
                let result = crate::cuda::gpu_scalar_op(a_ptr, s as f32, crate::cuda::kernels::ADD_SCALAR_F32_PTX, "nsl_add_scalar_f32\0");
                if relinq_a { nsl_tensor_free(a_ptr); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned (CPU) or when caller relinquished.
    // Skip for dtype=4 (i32) — needs type conversion to float, can't mutate in-place.
    {
        let t = unsafe { &mut *(a_ptr as *mut NslTensor) };
        if t.dtype != 4 && (relinq_a || t.can_mutate_inplace()) {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                let sf = s as f32;
                for i in 0..len { unsafe { *d.add(i) += sf }; }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len { unsafe { *d.add(i) += s }; }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return a_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(a_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    // Output dtype: i32 inputs produce f32 (used for label arithmetic in cross_entropy)
    let out_dtype: u16 = if a.dtype == 4 { 1 } else { a.dtype };
    let data: *mut c_void = if a.dtype == 4 {
        // i32 → f32 with scalar add
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        let src = a.data as *const i32;
        let sf = s as f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = (*src.add(i) as f32) + sf };
        }
        buf as *mut c_void
    } else if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = *a.data_f32().add(i) + (s as f32) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = *a.data_f64().add(i) + s };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor::new(
        data,
        shape,
        strides,
        ndim,
        len,
        a.device,
        out_dtype,
        1,
        0,
    ));
    let result = NslTensor::publish(result);
    nsl_tensor_free(a_c);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::AddScalar { a: a_ptr, out: result });
    }
    if relinq_a { nsl_tensor_free(a_ptr); }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_mul_scalar(a_ptr: i64, s: f64, flags: u8) -> i64 {
    use crate::tensor::fbip_flags::relinquish_a;
    let relinq_a = relinquish_a(flags);
    {
        let ta = unsafe { &*(a_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if relinq_a || ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_scalar_op_inplace(a_ptr, s as f32, crate::cuda::kernels::MUL_SCALAR_F32_PTX, "nsl_mul_scalar_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return a_ptr;
                }
                let result = crate::cuda::gpu_scalar_op(a_ptr, s as f32, crate::cuda::kernels::MUL_SCALAR_F32_PTX, "nsl_mul_scalar_f32\0");
                if relinq_a { nsl_tensor_free(a_ptr); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: mutate in-place when uniquely owned (CPU) or when caller relinquished.
    {
        let t = unsafe { &mut *(a_ptr as *mut NslTensor) };
        if relinq_a || t.can_mutate_inplace() {
            let len = t.len as usize;
            if t.dtype == 1 {
                let d = t.data as *mut f32;
                let sf = s as f32;
                for i in 0..len { unsafe { *d.add(i) *= sf }; }
            } else {
                let d = t.data as *mut f64;
                for i in 0..len { unsafe { *d.add(i) *= s }; }
            }
            t.refcount.fetch_add(1, Ordering::SeqCst);
            super::fbip_record_reuse();
            return a_ptr;
        }
    }
    super::fbip_record_alloc();
    let a_c = nsl_tensor_contiguous(a_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len;
    let ndim = a.ndim;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if a.dtype == 1 {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = *a.data_f32().add(i) * (s as f32) };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len as usize {
            unsafe { *buf.add(i) = *a.data_f64().add(i) * s };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor::new(
        data,
        shape,
        strides,
        ndim,
        len,
        a.device,
        a.dtype,
        1,
        0,
    ));
    let result = NslTensor::publish(result);
    nsl_tensor_free(a_c);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::MulScalar {
            a: a_ptr,
            scalar: s,
            out: result,
        });
    }
    if relinq_a { nsl_tensor_free(a_ptr); }
    result
}

// === Sparse matrix multiply (M52c: weight-aware CSR SpMM) ===

/// CSR sparse matmul: C = A_sparse @ B_dense
/// CSR arrays are passed as device pointers (uploaded by codegen from .rodata).
/// Returns a new dense tensor C[nrows, N] where N is B's last dim.
#[no_mangle]
pub extern "C" fn nsl_sparse_matmul(
    row_ptrs_ptr: i64,
    col_indices_ptr: i64,
    values_ptr: i64,
    b_ptr: i64,
    nrows: i64,
    _ncols: i64,
    _nnz: i64,
) -> i64 {
    #[cfg(feature = "cuda")]
    {
        let b = NslTensor::from_ptr(b_ptr);
        if b.device > 0 {
            let last_dim = if b.ndim >= 2 {
                unsafe { *b.shape.add(b.ndim as usize - 1) }
            } else { 1 };
            let n_out = last_dim as usize;

            let out_total = nrows as usize * n_out;
            let out_data = crate::cuda::inner::alloc_managed(out_total * 4);
            crate::cuda::inner::memset_d8(out_data, out_total * 4);

            let mut rp = row_ptrs_ptr as u64;
            let mut ci = col_indices_ptr as u64;
            let mut v = values_ptr as u64;
            let mut bd = b.data as u64;
            let mut cd = out_data as u64;
            let mut m = nrows as u64;
            let mut n = n_out as u64;

            let args: [*mut std::ffi::c_void; 7] = [
                &mut rp as *mut _ as *mut std::ffi::c_void,
                &mut ci as *mut _ as *mut std::ffi::c_void,
                &mut v as *mut _ as *mut std::ffi::c_void,
                &mut bd as *mut _ as *mut std::ffi::c_void,
                &mut cd as *mut _ as *mut std::ffi::c_void,
                &mut m as *mut _ as *mut std::ffi::c_void,
                &mut n as *mut _ as *mut std::ffi::c_void,
            ];

            let block = 256i64;
            let grid_x = nrows;
            let grid_y = ((n_out as i64) + block - 1) / block;

            let result = crate::cuda::inner::kernel_launch(
                crate::cuda::fused_kernels::CSR_SPMM_F32_PTX.as_ptr(),
                b"nsl_csr_spmm_f32\0".as_ptr(),
                [grid_x, grid_y, 1], [block, 1, 1], &args, 0,
            );
            assert_eq!(result as u32, 0, "GPU CSR SpMM kernel failed: {:?}", result);
            unsafe { cudarc::driver::sys::cuCtxSynchronize(); }

            let out_shape = crate::memory::checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
            unsafe {
                *out_shape = nrows;
                *out_shape.add(1) = n_out as i64;
            }
            let out_strides = NslTensor::compute_strides(out_shape, 2);
            let out = Box::new(NslTensor::new(
                out_data, out_shape, out_strides, 2, out_total as i64, b.device, 1, 1, 0,
            ));
            return NslTensor::publish(out);
        }
    }
    // CPU fallback: use dense matmul
    eprintln!("[nsl] sparse_matmul: CPU fallback (sparse kernels are GPU-only)");
    nsl_tensor_matmul(values_ptr, b_ptr, 0) // approximate fallback
}

// === Matrix multiply ===

#[no_mangle]
pub extern "C" fn nsl_tensor_matmul(a_ptr: i64, b_ptr: i64, flags: u8) -> i64 {
    use crate::tensor::fbip_flags::{relinquish_a, relinquish_b};
    let relinq_a = relinquish_a(flags);
    let relinq_b = relinquish_b(flags);
    // Capture caller's original B before reconcile_device shadows `b_ptr` (ELTLS 1b.1).
    let b_orig = b_ptr;
    let (b_ptr, b_transferred) = reconcile_device(a_ptr, b_ptr);
    // GPU dispatch
    {
        let a = unsafe { &*(a_ptr as *const NslTensor) };
        if a.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let result = crate::cuda::gpu_matmul_f32(a_ptr, b_ptr);
                // FBIP-3: matmul always allocates fresh output; honor relinquish flags
                // by freeing A/B here. No in-place optimization because matmul output
                // shape differs from input in general ([M,K] @ [K,N] -> [M,N]).
                if relinq_a { nsl_tensor_free(a_ptr); }
                if relinq_b { nsl_tensor_free(b_orig); }
                if b_transferred { nsl_tensor_free(b_ptr); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }

    // Ensure contiguous inputs for flat-indexed matmul
    let a_c = nsl_tensor_contiguous(a_ptr);
    let b_c = nsl_tensor_contiguous(b_ptr);
    let a = NslTensor::from_ptr(a_c);
    let b = NslTensor::from_ptr(b_c);

    if a.ndim < 2 || b.ndim < 2 {
        eprintln!(
            "nsl: matmul requires at least 2D tensors (got {}D and {}D)",
            a.ndim, b.ndim
        );
        std::process::abort();
    }

    let a_shape = get_shape_vec(a);
    let b_shape = get_shape_vec(b);
    let a_nd = a.ndim as usize;
    let b_nd = b.ndim as usize;

    let m = a_shape[a_nd - 2];
    let k = a_shape[a_nd - 1];
    let k2 = b_shape[b_nd - 2];
    let n = b_shape[b_nd - 1];

    if k != k2 {
        eprintln!(
            "nsl: matmul inner dimension mismatch ({}x{} @ {}x{})",
            m, k, k2, n
        );
        std::process::abort();
    }

    // Broadcast batch dimensions (all dims before last two)
    let a_batch: Vec<i64> = a_shape[..a_nd - 2].to_vec();
    let b_batch: Vec<i64> = b_shape[..b_nd - 2].to_vec();

    // Compute broadcast batch shape
    let max_batch_nd = a_batch.len().max(b_batch.len());
    let mut out_batch: Vec<i64> = Vec::with_capacity(max_batch_nd);
    for i in 0..max_batch_nd {
        let a_dim = if i < max_batch_nd - a_batch.len() { 1 } else { a_batch[i - (max_batch_nd - a_batch.len())] };
        let b_dim = if i < max_batch_nd - b_batch.len() { 1 } else { b_batch[i - (max_batch_nd - b_batch.len())] };
        if a_dim != b_dim && a_dim != 1 && b_dim != 1 {
            eprintln!("nsl: matmul batch dimension mismatch at dim {}: {} vs {}", i, a_dim, b_dim);
            std::process::abort();
        }
        out_batch.push(a_dim.max(b_dim));
    }

    // Build output shape: batch_dims + [m, n]
    let out_nd = out_batch.len() + 2;
    let mut out_shape_vec: Vec<i64> = out_batch.clone();
    out_shape_vec.push(m);
    out_shape_vec.push(n);

    let total_batch: i64 = out_batch.iter().product::<i64>().max(1);
    let len = total_batch * m * n;

    let shape = checked_alloc(out_nd * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &s) in out_shape_vec.iter().enumerate() {
        unsafe { *shape.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape, out_nd as i64);

    let a_mat_stride = (m * k) as usize;
    let b_mat_stride = (k2 * n) as usize;
    let out_mat_stride = (m * n) as usize;

    // Precompute per-dimension strides for a and b batch dims (for broadcast mapping)
    let mut a_batch_strides: Vec<usize> = vec![0; max_batch_nd];
    let mut b_batch_strides: Vec<usize> = vec![0; max_batch_nd];
    {
        let mut a_s = 1usize;
        let mut b_s = 1usize;
        for i in (0..max_batch_nd).rev() {
            let a_offset = max_batch_nd - a_batch.len();
            let b_offset = max_batch_nd - b_batch.len();
            let a_d = if i < a_offset { 1 } else { a_batch[i - a_offset] as usize };
            let b_d = if i < b_offset { 1 } else { b_batch[i - b_offset] as usize };
            a_batch_strides[i] = a_s;
            b_batch_strides[i] = b_s;
            a_s *= a_d;
            b_s *= b_d;
        }
    }

    // Dispatch based on dtype (use f32 if either input is f32)
    let out_dtype: u16 = if a.dtype == 1 || b.dtype == 1 { 1 } else { 0 };
    let elem_size = if out_dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let raw_data = checked_alloc_zeroed((len as usize) * elem_size);

    for batch_idx in 0..total_batch as usize {
        // Decompose flat batch_idx into per-dimension coordinates, then map to a/b
        let mut a_batch_idx = 0usize;
        let mut b_batch_idx = 0usize;
        let mut remaining = batch_idx;
        for i in 0..max_batch_nd {
            let next_stride = if i + 1 < max_batch_nd {
                out_batch[i + 1..].iter().product::<i64>() as usize
            } else {
                1
            };
            let coord = remaining / next_stride;
            remaining %= next_stride;

            let a_offset = max_batch_nd - a_batch.len();
            let b_offset = max_batch_nd - b_batch.len();
            let a_d = if i < a_offset { 1 } else { a_batch[i - a_offset] as usize };
            let b_d = if i < b_offset { 1 } else { b_batch[i - b_offset] as usize };
            // Clamp coordinate to broadcast dim (size-1 dims broadcast to 0)
            a_batch_idx += coord.min(a_d - 1) * a_batch_strides[i];
            b_batch_idx += coord.min(b_d - 1) * b_batch_strides[i];
        }

        let a_base = a_batch_idx * a_mat_stride;
        let b_base = b_batch_idx * b_mat_stride;
        let out_base = batch_idx * out_mat_stride;

        // 2D matmul for this batch element — dispatch to cache-tiled kernel
        if out_dtype == 1 {
            let c_ptr = unsafe { (raw_data as *mut f32).add(out_base) };
            if a.dtype == 1 && b.dtype == 1 {
                // Both f32 — call tiled kernel directly
                let a_ptr = unsafe { a.data_f32().add(a_base) };
                let b_ptr = unsafe { b.data_f32().add(b_base) };
                crate::cpu::tiled_matmul_f32(a_ptr, b_ptr, c_ptr, m as usize, k as usize, n as usize);
            } else {
                // Mixed dtype — fall back to element-wise conversion (rare path)
                let read_a = |idx: usize| -> f32 {
                    if a.dtype == 1 { unsafe { *a.data_f32().add(idx) } } else { unsafe { *a.data_f64().add(idx) as f32 } }
                };
                let read_b = |idx: usize| -> f32 {
                    if b.dtype == 1 { unsafe { *b.data_f32().add(idx) } } else { unsafe { *b.data_f64().add(idx) as f32 } }
                };
                for i in 0..m as usize {
                    for j in 0..k as usize {
                        let a_val = read_a(a_base + i * k as usize + j);
                        for l in 0..n as usize {
                            let b_val = read_b(b_base + j * n as usize + l);
                            unsafe { *c_ptr.add(i * n as usize + l) += a_val * b_val; }
                        }
                    }
                }
            }
        } else {
            // Both f64 — call tiled kernel
            let c_ptr = unsafe { (raw_data as *mut f64).add(out_base) };
            let a_ptr = unsafe { a.data_f64().add(a_base) };
            let b_ptr = unsafe { b.data_f64().add(b_base) };
            crate::cpu::tiled_matmul_f64(a_ptr, b_ptr, c_ptr, m as usize, k as usize, n as usize);
        }
    }

    let result = Box::new(NslTensor::new(
        raw_data as *mut c_void,
        shape,
        strides,
        out_nd as i64,
        len,
        0,
        out_dtype,
        1,
        0,
    ));
    let result = NslTensor::publish(result);
    nsl_tensor_free(a_c);
    nsl_tensor_free(b_c);
    if autodiff::is_recording() {
        NslTensor::from_ptr(a_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        NslTensor::from_ptr(b_ptr).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::MatMul {
            a: a_ptr,
            b: b_ptr,
            out: result,
            saved_a: a_ptr,
            saved_b: b_ptr,
        });
    }
    #[cfg(feature = "interop")]
    if crate::trace::is_tracing() {
        let rt = NslTensor::from_ptr(result);
        let shape: Vec<i64> = (0..rt.ndim as usize).map(|d| unsafe { *rt.shape.add(d) }).collect();
        crate::trace::record_op(crate::trace::OpType::MatMul, vec![a_ptr, b_ptr], result, shape, rt.dtype, vec![]);
    }
    // FBIP-3: CPU matmul always allocates fresh output; honor relinquish flags.
    if relinq_a { nsl_tensor_free(a_ptr); }
    if relinq_b { nsl_tensor_free(b_orig); }
    if b_transferred { nsl_tensor_free(b_ptr); }
    result
}

#[cfg(test)]
mod fbip_add_tests {
    use super::*;
    use crate::tensor::fbip_flags::{RELINQUISH_A, RELINQUISH_B};

    /// Helper: create a 1-D f64 tensor (CPU, dtype=0, contiguous, refcount=1).
    fn make_tensor_f64(data: &[f64]) -> i64 {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, data.len() as i64);
        let ptr = crate::tensor::creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let t = NslTensor::from_ptr(ptr);
        for (i, v) in data.iter().enumerate() {
            unsafe { *t.data_f64().add(i) = *v };
        }
        ptr
    }

    fn read_f64(ptr: i64, idx: usize) -> f64 {
        let t = unsafe { &*(ptr as *const NslTensor) };
        unsafe { *(t.data as *const f64).add(idx) }
    }

    /// flags=0 — both operands untouched, fresh output.
    #[test]
    fn add_flags_zero_leaves_inputs_alive() {
        // Bump refcount on A so the legacy heuristic cannot take its in-place path;
        // this guarantees the out-of-place path runs and we can observe that A and B
        // are still valid after the call.
        let a = make_tensor_f64(&[2.0, 3.0, 4.0]);
        let b = make_tensor_f64(&[1.0, 1.0, 1.0]);
        NslTensor::from_ptr(a).refcount.fetch_add(1, Ordering::SeqCst);
        let out = nsl_tensor_add(a, b, 0);
        assert_ne!(out, a, "flags=0 + shared A must allocate fresh output");
        assert_ne!(out, b, "flags=0 must not reuse B");
        assert_eq!(read_f64(a, 0), 2.0);
        assert_eq!(read_f64(b, 0), 1.0);
        assert_eq!(read_f64(out, 0), 3.0);
        NslTensor::from_ptr(a).refcount.fetch_sub(1, Ordering::SeqCst);
        nsl_tensor_free(a);
        nsl_tensor_free(b);
        nsl_tensor_free(out);
    }

    /// flags=0x03 — both relinquished, in-place on A, B must still be freed.
    /// This test verifies that the call returns A (in-place path taken) and that
    /// the process does not crash from a double-free of B when the runtime
    /// relinquishes it.
    #[test]
    fn add_flags_both_relinquish_inplace_on_a_frees_b() {
        let a = make_tensor_f64(&[10.0, 20.0]);
        let b = make_tensor_f64(&[1.0, 2.0]);
        let out = nsl_tensor_add(a, b, RELINQUISH_A | RELINQUISH_B);
        // In-place on A: returned pointer equals A.
        assert_eq!(out, a, "in-place on A should return A's pointer");
        // A now holds the result [11.0, 22.0].
        assert_eq!(read_f64(out, 0), 11.0);
        assert_eq!(read_f64(out, 1), 22.0);
        // B has been freed by the runtime. Do not touch it.
        nsl_tensor_free(out); // out == a
    }

    // === Task 3: sub/mul/div/matmul FBIP-3 flag tests ===

    #[test]
    fn sub_flags_zero_leaves_inputs_alive() {
        let a = make_tensor_f64(&[10.0, 20.0, 30.0]);
        let b = make_tensor_f64(&[1.0, 2.0, 3.0]);
        NslTensor::from_ptr(a).refcount.fetch_add(1, Ordering::SeqCst);
        let out = nsl_tensor_sub(a, b, 0);
        assert_ne!(out, a, "flags=0 + shared A must allocate fresh output");
        assert_ne!(out, b, "flags=0 must not reuse B");
        assert_eq!(read_f64(a, 0), 10.0);
        assert_eq!(read_f64(b, 0), 1.0);
        assert_eq!(read_f64(out, 0), 9.0);
        assert_eq!(read_f64(out, 2), 27.0);
        NslTensor::from_ptr(a).refcount.fetch_sub(1, Ordering::SeqCst);
        nsl_tensor_free(a);
        nsl_tensor_free(b);
        nsl_tensor_free(out);
    }

    #[test]
    fn sub_flags_both_relinquish_inplace_on_a_frees_b() {
        let a = make_tensor_f64(&[10.0, 20.0]);
        let b = make_tensor_f64(&[1.0, 2.0]);
        let out = nsl_tensor_sub(a, b, RELINQUISH_A | RELINQUISH_B);
        assert_eq!(out, a, "in-place on A should return A's pointer");
        assert_eq!(read_f64(out, 0), 9.0);
        assert_eq!(read_f64(out, 1), 18.0);
        nsl_tensor_free(out);
    }

    #[test]
    fn mul_flags_zero_leaves_inputs_alive() {
        let a = make_tensor_f64(&[2.0, 3.0, 4.0]);
        let b = make_tensor_f64(&[5.0, 6.0, 7.0]);
        NslTensor::from_ptr(a).refcount.fetch_add(1, Ordering::SeqCst);
        let out = nsl_tensor_mul(a, b, 0);
        assert_ne!(out, a, "flags=0 + shared A must allocate fresh output");
        assert_ne!(out, b, "flags=0 must not reuse B");
        assert_eq!(read_f64(a, 0), 2.0);
        assert_eq!(read_f64(b, 0), 5.0);
        assert_eq!(read_f64(out, 0), 10.0);
        assert_eq!(read_f64(out, 2), 28.0);
        NslTensor::from_ptr(a).refcount.fetch_sub(1, Ordering::SeqCst);
        nsl_tensor_free(a);
        nsl_tensor_free(b);
        nsl_tensor_free(out);
    }

    #[test]
    fn mul_flags_both_relinquish_inplace_on_a_frees_b() {
        let a = make_tensor_f64(&[2.0, 3.0]);
        let b = make_tensor_f64(&[4.0, 5.0]);
        let out = nsl_tensor_mul(a, b, RELINQUISH_A | RELINQUISH_B);
        assert_eq!(out, a, "in-place on A should return A's pointer");
        assert_eq!(read_f64(out, 0), 8.0);
        assert_eq!(read_f64(out, 1), 15.0);
        nsl_tensor_free(out);
    }

    #[test]
    fn div_flags_zero_leaves_inputs_alive() {
        let a = make_tensor_f64(&[10.0, 20.0, 30.0]);
        let b = make_tensor_f64(&[2.0, 4.0, 5.0]);
        NslTensor::from_ptr(a).refcount.fetch_add(1, Ordering::SeqCst);
        let out = nsl_tensor_div(a, b, 0);
        assert_ne!(out, a, "flags=0 + shared A must allocate fresh output");
        assert_ne!(out, b, "flags=0 must not reuse B");
        assert_eq!(read_f64(a, 0), 10.0);
        assert_eq!(read_f64(b, 0), 2.0);
        assert_eq!(read_f64(out, 0), 5.0);
        assert_eq!(read_f64(out, 2), 6.0);
        NslTensor::from_ptr(a).refcount.fetch_sub(1, Ordering::SeqCst);
        nsl_tensor_free(a);
        nsl_tensor_free(b);
        nsl_tensor_free(out);
    }

    #[test]
    fn div_flags_both_relinquish_inplace_on_a_frees_b() {
        let a = make_tensor_f64(&[10.0, 20.0]);
        let b = make_tensor_f64(&[2.0, 5.0]);
        let out = nsl_tensor_div(a, b, RELINQUISH_A | RELINQUISH_B);
        assert_eq!(out, a, "in-place on A should return A's pointer");
        assert_eq!(read_f64(out, 0), 5.0);
        assert_eq!(read_f64(out, 1), 4.0);
        nsl_tensor_free(out);
    }

    /// Helper: create a 2x2 f64 tensor from a row-major slice.
    fn make_2x2_f64(data: &[f64; 4]) -> i64 {
        let shape_list = crate::list::nsl_list_new();
        crate::list::nsl_list_push(shape_list, 2);
        crate::list::nsl_list_push(shape_list, 2);
        let ptr = crate::tensor::creation::tensor_from_shape_list_f64(shape_list, 0.0);
        let t = NslTensor::from_ptr(ptr);
        for (i, v) in data.iter().enumerate() {
            unsafe { *t.data_f64().add(i) = *v };
        }
        ptr
    }

    #[test]
    fn matmul_flags_zero_leaves_inputs_alive() {
        // [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
        let a = make_2x2_f64(&[1.0, 2.0, 3.0, 4.0]);
        let b = make_2x2_f64(&[5.0, 6.0, 7.0, 8.0]);
        let out = nsl_tensor_matmul(a, b, 0);
        // Matmul always allocates fresh output.
        assert_ne!(out, a);
        assert_ne!(out, b);
        // Inputs unchanged.
        assert_eq!(read_f64(a, 0), 1.0);
        assert_eq!(read_f64(b, 0), 5.0);
        // Output correct.
        assert_eq!(read_f64(out, 0), 19.0);
        assert_eq!(read_f64(out, 1), 22.0);
        assert_eq!(read_f64(out, 2), 43.0);
        assert_eq!(read_f64(out, 3), 50.0);
        nsl_tensor_free(a);
        nsl_tensor_free(b);
        nsl_tensor_free(out);
    }

    #[test]
    fn matmul_flags_both_relinquish_frees_a_and_b() {
        // Matmul does not currently do in-place; flags=0x03 means the runtime
        // owns and frees both A and B. Out is a fresh allocation.
        let a = make_2x2_f64(&[1.0, 2.0, 3.0, 4.0]);
        let b = make_2x2_f64(&[5.0, 6.0, 7.0, 8.0]);
        let out = nsl_tensor_matmul(a, b, RELINQUISH_A | RELINQUISH_B);
        // Result: [[19,22],[43,50]]
        assert_eq!(read_f64(out, 0), 19.0);
        assert_eq!(read_f64(out, 3), 50.0);
        // A and B have been freed by the runtime — do not touch them.
        nsl_tensor_free(out);
    }

    // === Task 4: scalar variants FBIP-3 flag tests ===

    #[test]
    fn add_scalar_flags_zero_leaves_input_alive() {
        let a = make_tensor_f64(&[10.0, 20.0]);
        // Bump refcount to force out-of-place path
        NslTensor::from_ptr(a).refcount.fetch_add(1, Ordering::SeqCst);
        let out = nsl_tensor_add_scalar(a, 5.0, 0);
        assert_ne!(out, a, "flags=0 + shared A must allocate fresh output");
        assert_eq!(read_f64(a, 0), 10.0);
        assert_eq!(read_f64(a, 1), 20.0);
        assert_eq!(read_f64(out, 0), 15.0);
        assert_eq!(read_f64(out, 1), 25.0);
        NslTensor::from_ptr(a).refcount.fetch_sub(1, Ordering::SeqCst);
        nsl_tensor_free(a);
        nsl_tensor_free(out);
    }

    #[test]
    fn add_scalar_flags_relinquish_a_inplace() {
        let a = make_tensor_f64(&[10.0, 20.0]);
        let out = nsl_tensor_add_scalar(a, 5.0, RELINQUISH_A);
        assert_eq!(out, a, "in-place on A should return A");
        assert_eq!(read_f64(out, 0), 15.0);
        assert_eq!(read_f64(out, 1), 25.0);
        nsl_tensor_free(out); // == a
    }

    #[test]
    fn mul_scalar_flags_zero_leaves_input_alive() {
        let a = make_tensor_f64(&[10.0, 20.0]);
        NslTensor::from_ptr(a).refcount.fetch_add(1, Ordering::SeqCst);
        let out = nsl_tensor_mul_scalar(a, 5.0, 0);
        assert_ne!(out, a, "flags=0 + shared A must allocate fresh output");
        assert_eq!(read_f64(a, 0), 10.0);
        assert_eq!(read_f64(a, 1), 20.0);
        assert_eq!(read_f64(out, 0), 50.0);
        assert_eq!(read_f64(out, 1), 100.0);
        NslTensor::from_ptr(a).refcount.fetch_sub(1, Ordering::SeqCst);
        nsl_tensor_free(a);
        nsl_tensor_free(out);
    }

    #[test]
    fn mul_scalar_flags_relinquish_a_inplace() {
        let a = make_tensor_f64(&[10.0, 20.0]);
        let out = nsl_tensor_mul_scalar(a, 5.0, RELINQUISH_A);
        assert_eq!(out, a, "in-place on A should return A");
        assert_eq!(read_f64(out, 0), 50.0);
        assert_eq!(read_f64(out, 1), 100.0);
        nsl_tensor_free(out); // == a
    }

    // TODO(eltls): add a regression test for the cross-device RELINQUISH_B path
    // (caller's CPU B reconciled to GPU alongside a GPU A). The bug fixed in
    // ELTLS commit 1b.1 was that `reconcile_device` shadows `b` with a fresh
    // runtime-owned copy, so the original free logic leaked the caller's
    // original B. A proper test requires either:
    //   (a) a per-tensor free counter in the runtime (analogous to the
    //       existing GPU_TENSOR_LIVE tracker, but for CPU tensors too), OR
    //   (b) the thread-local `memory::stats` CPU counters wired into tensor
    //       Box allocations (currently they only track checked_alloc/free
    //       of data buffers, not the NslTensor struct box itself).
    // Until one of those is available, observing "was the caller's original
    // pointer freed" requires reading the poisoned magic field of freed
    // memory, which is undefined behavior (the Box is dropped right after
    // the magic poison write). The fix is verified by code review and by
    // the fact that the GPU leak surfaces visibly in training workloads.
}
