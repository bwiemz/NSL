//! Tensor arithmetic operations: add, sub, mul, div, neg, scalar ops, matmul.

use std::ffi::c_void;
use std::sync::atomic::{AtomicI64, Ordering};

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
pub extern "C" fn nsl_tensor_add(a: i64, b: i64) -> i64 {
    let (b, b_transferred) = reconcile_device(a, b);
    {
        let ta = unsafe { &*(a as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let tb = unsafe { &*(b as *const NslTensor) };
                if ta.can_mutate_inplace_gpu() && ta.shape_eq(tb) {
                    crate::cuda::gpu_elementwise_binary_inplace(a, b, crate::cuda::kernels::ADD_F32_PTX, "nsl_add_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    if b_transferred { nsl_tensor_free(b); }
                    return a;
                }
                let result = crate::cuda::gpu_elementwise_binary(a, b, crate::cuda::kernels::ADD_F32_PTX, "nsl_add_f32\0");
                if b_transferred { nsl_tensor_free(b); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: reuse left operand when uniquely owned + same shape (no broadcast, CPU)
    {
        let ta = unsafe { &mut *(a as *mut NslTensor) };
        let tb = unsafe { &*(b as *const NslTensor) };
        if ta.can_mutate_inplace() && ta.shape_eq(tb) && ta.dtype == tb.dtype && tb.is_contiguous() {
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
            if b_transferred { nsl_tensor_free(b); }
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
    if b_transferred { nsl_tensor_free(b); }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sub(a: i64, b: i64) -> i64 {
    let (b, b_transferred) = reconcile_device(a, b);
    {
        let ta = unsafe { &*(a as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let tb = unsafe { &*(b as *const NslTensor) };
                if ta.can_mutate_inplace_gpu() && ta.shape_eq(tb) {
                    crate::cuda::gpu_elementwise_binary_inplace(a, b, crate::cuda::kernels::SUB_F32_PTX, "nsl_sub_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    if b_transferred { nsl_tensor_free(b); }
                    return a;
                }
                let result = crate::cuda::gpu_elementwise_binary(a, b, crate::cuda::kernels::SUB_F32_PTX, "nsl_sub_f32\0");
                if b_transferred { nsl_tensor_free(b); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: reuse left operand when uniquely owned + same shape (CPU)
    {
        let ta = unsafe { &mut *(a as *mut NslTensor) };
        let tb = unsafe { &*(b as *const NslTensor) };
        if ta.can_mutate_inplace() && ta.shape_eq(tb) && ta.dtype == tb.dtype && tb.is_contiguous() {
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
    if b_transferred { nsl_tensor_free(b); }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_mul(a: i64, b: i64) -> i64 {
    let (b, b_transferred) = reconcile_device(a, b);
    {
        let ta = unsafe { &*(a as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let tb = unsafe { &*(b as *const NslTensor) };
                if ta.can_mutate_inplace_gpu() && ta.shape_eq(tb) {
                    crate::cuda::gpu_elementwise_binary_inplace(a, b, crate::cuda::kernels::MUL_F32_PTX, "nsl_mul_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    if b_transferred { nsl_tensor_free(b); }
                    return a;
                }
                let result = crate::cuda::gpu_elementwise_binary(a, b, crate::cuda::kernels::MUL_F32_PTX, "nsl_mul_f32\0");
                if b_transferred { nsl_tensor_free(b); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: reuse left operand when uniquely owned + same shape (CPU)
    {
        let ta = unsafe { &mut *(a as *mut NslTensor) };
        let tb = unsafe { &*(b as *const NslTensor) };
        if ta.can_mutate_inplace() && ta.shape_eq(tb) && ta.dtype == tb.dtype && tb.is_contiguous() {
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
    if b_transferred { nsl_tensor_free(b); }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_div(a: i64, b: i64) -> i64 {
    let (b, b_transferred) = reconcile_device(a, b);
    {
        let ta = unsafe { &*(a as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let tb = unsafe { &*(b as *const NslTensor) };
                if ta.can_mutate_inplace_gpu() && ta.shape_eq(tb) {
                    crate::cuda::gpu_elementwise_binary_inplace(a, b, crate::cuda::kernels::DIV_F32_PTX, "nsl_div_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    if b_transferred { nsl_tensor_free(b); }
                    return a;
                }
                let result = crate::cuda::gpu_elementwise_binary(a, b, crate::cuda::kernels::DIV_F32_PTX, "nsl_div_f32\0");
                if b_transferred { nsl_tensor_free(b); }
                return result;
            }
            #[cfg(not(feature = "cuda"))]
            { panic!("CUDA support not compiled"); }
        }
    }
    // FBIP: reuse left operand when uniquely owned + same shape (CPU)
    {
        let ta = unsafe { &mut *(a as *mut NslTensor) };
        let tb = unsafe { &*(b as *const NslTensor) };
        if ta.can_mutate_inplace() && ta.shape_eq(tb) && ta.dtype == tb.dtype && tb.is_contiguous() {
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

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: AtomicI64::new(1),
        device: a.device,
        dtype: a.dtype,
        owns_data: 1, data_owner: 0,
    });
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
pub extern "C" fn nsl_tensor_add_scalar(a_ptr: i64, s: f64) -> i64 {
    {
        let ta = unsafe { &*(a_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_scalar_op_inplace(a_ptr, s as f32, crate::cuda::kernels::ADD_SCALAR_F32_PTX, "nsl_add_scalar_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return a_ptr;
                }
                return crate::cuda::gpu_scalar_op(a_ptr, s as f32, crate::cuda::kernels::ADD_SCALAR_F32_PTX, "nsl_add_scalar_f32\0");
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

    let data: *mut c_void = if a.dtype == 1 {
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

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: AtomicI64::new(1),
        device: a.device,
        dtype: a.dtype,
        owns_data: 1, data_owner: 0,
    });
    let result = NslTensor::publish(result);
    nsl_tensor_free(a_c);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::AddScalar { a: a_ptr, out: result });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_mul_scalar(a_ptr: i64, s: f64) -> i64 {
    {
        let ta = unsafe { &*(a_ptr as *const NslTensor) };
        if ta.device > 0 {
            #[cfg(feature = "cuda")]
            {
                if ta.can_mutate_inplace_gpu() {
                    crate::cuda::gpu_scalar_op_inplace(a_ptr, s as f32, crate::cuda::kernels::MUL_SCALAR_F32_PTX, "nsl_mul_scalar_f32\0");
                    ta.refcount.fetch_add(1, Ordering::SeqCst);
                    super::fbip_record_reuse();
                    return a_ptr;
                }
                return crate::cuda::gpu_scalar_op(a_ptr, s as f32, crate::cuda::kernels::MUL_SCALAR_F32_PTX, "nsl_mul_scalar_f32\0");
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

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: AtomicI64::new(1),
        device: a.device,
        dtype: a.dtype,
        owns_data: 1, data_owner: 0,
    });
    let result = NslTensor::publish(result);
    nsl_tensor_free(a_c);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::MulScalar {
            a: a_ptr,
            scalar: s,
            out: result,
        });
    }
    result
}

// === Matrix multiply ===

#[no_mangle]
pub extern "C" fn nsl_tensor_matmul(a_ptr: i64, b_ptr: i64) -> i64 {
    let (b_ptr, b_transferred) = reconcile_device(a_ptr, b_ptr);
    // GPU dispatch
    {
        let a = unsafe { &*(a_ptr as *const NslTensor) };
        if a.device > 0 {
            #[cfg(feature = "cuda")]
            {
                let result = crate::cuda::gpu_matmul_f32(a_ptr, b_ptr);
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

    let result = Box::new(NslTensor {
        data: raw_data as *mut c_void,
        shape,
        strides,
        ndim: out_nd as i64,
        len,
        refcount: AtomicI64::new(1),
        device: 0,
        dtype: out_dtype,
        owns_data: 1, data_owner: 0,
    });
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
    if b_transferred { nsl_tensor_free(b_ptr); }
    result
}
