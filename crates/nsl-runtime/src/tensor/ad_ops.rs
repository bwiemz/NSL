//! Tensor operations required by the source-AD backward lowering pass.
//!
//! These functions are called from Wengert-to-Cranelift generated code to
//! implement individual backward steps: comparisons, ternary select, scalar
//! creation, zero-padding, scatter-add for embedding, log-softmax, spatial
//! repeat (avgpool backward), and inverse RoPE rotation.

use std::ffi::c_void;
use std::sync::atomic::Ordering;

use crate::autodiff;
use crate::memory::{checked_alloc, checked_alloc_zeroed};

use super::{nsl_tensor_contiguous, nsl_tensor_free, nsl_tensor_to_device, NslTensor};

#[inline]
fn publish_cpu_result_to_device(result: Box<NslTensor>, target_device: u8, context: &str) -> i64 {
    let out = NslTensor::publish(result);
    if target_device > 0 {
        #[cfg(not(feature = "cuda"))]
        let _ = context;
        #[cfg(feature = "cuda")]
        crate::cuda::inner::set_oom_context(context);
        let gpu_out = nsl_tensor_to_device(out, target_device as i64);
        nsl_tensor_free(out);
        gpu_out
    } else {
        out
    }
}

#[inline]
fn prepare_cpu_input(tensor_ptr: i64) -> (i64, i64, u8) {
    let contig_ptr = nsl_tensor_contiguous(tensor_ptr);
    let device = NslTensor::from_ptr(contig_ptr).device;
    let cpu_ptr = if device > 0 {
        nsl_tensor_to_device(contig_ptr, 0)
    } else {
        contig_ptr
    };
    (contig_ptr, cpu_ptr, device)
}

#[inline]
fn release_cpu_input(contig_ptr: i64, cpu_ptr: i64) {
    if cpu_ptr != contig_ptr {
        nsl_tensor_free(cpu_ptr);
    }
    nsl_tensor_free(contig_ptr);
}

// ---------------------------------------------------------------------------
// 1. nsl_tensor_compare — elementwise comparison → 0.0 / 1.0 tensor
// ---------------------------------------------------------------------------

/// Elementwise comparison returning a 0.0/1.0 tensor with the same shape as `a`.
///
/// `cmp_kind`:
///   0 = Gt, 1 = GtEq, 2 = Lt, 3 = LtEq, 4 = Eq, 5 = NotEq
#[no_mangle]
pub extern "C" fn nsl_tensor_compare(a_ptr: i64, b_ptr: i64, cmp_kind: i64) -> i64 {
    let (a_contig, a_cpu, a_device) = prepare_cpu_input(a_ptr);
    let (b_contig, b_cpu, _b_device) = prepare_cpu_input(b_ptr);
    let a = NslTensor::from_ptr(a_cpu);
    let b = NslTensor::from_ptr(b_cpu);
    let len = a.len as usize;
    let b_len = b.len as usize;
    let b_is_scalar = b_len == 1;
    // Allow scalar broadcasting: if b has 1 element, broadcast it to match a's length
    if !b_is_scalar && b_len < len {
        panic!("nsl_tensor_compare: b tensor too short ({} < {})", b_len, len);
    }
    let ndim = a.ndim;
    let dtype = a.dtype;
    let shape = NslTensor::copy_shape(a.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if dtype == 1 {
        let buf = checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len {
            let av = unsafe { *a.data_f32().add(i) };
            let bv = unsafe { *b.data_f32().add(if b_is_scalar { 0 } else { i }) };
            let result = compare_f32(av, bv, cmp_kind);
            unsafe { *buf.add(i) = if result { 1.0_f32 } else { 0.0_f32 } };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc(len * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len {
            let av = unsafe { *a.data_f64().add(i) };
            let bv = unsafe { *b.data_f64().add(if b_is_scalar { 0 } else { i }) };
            let result = compare_f64(av, bv, cmp_kind);
            unsafe { *buf.add(i) = if result { 1.0_f64 } else { 0.0_f64 } };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor::new(data, shape, strides, ndim, a.len, 0, dtype, 1, 0));
    release_cpu_input(a_contig, a_cpu);
    release_cpu_input(b_contig, b_cpu);
    publish_cpu_result_to_device(result, a_device, "ad_compare")
}

#[inline(always)]
fn compare_f32(av: f32, bv: f32, cmp_kind: i64) -> bool {
    match cmp_kind {
        0 => av > bv,
        1 => av >= bv,
        2 => av < bv,
        3 => av <= bv,
        4 => (av - bv).abs() < 1e-7_f32,
        5 => (av - bv).abs() >= 1e-7_f32,
        _ => false,
    }
}

#[inline(always)]
fn compare_f64(av: f64, bv: f64, cmp_kind: i64) -> bool {
    match cmp_kind {
        0 => av > bv,
        1 => av >= bv,
        2 => av < bv,
        3 => av <= bv,
        4 => (av - bv).abs() < 1e-12_f64,
        5 => (av - bv).abs() >= 1e-12_f64,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// 2. nsl_tensor_where — ternary elementwise select
// ---------------------------------------------------------------------------

/// Elementwise ternary: `result[i] = cond[i] != 0 ? true_val[i] : false_val[i]`.
///
/// All three tensors must have the same shape. `cond` is read as the dtype of
/// the condition tensor (0.0 = false, anything else = true).
#[no_mangle]
pub extern "C" fn nsl_tensor_where(cond_ptr: i64, true_ptr: i64, false_ptr: i64) -> i64 {
    let (cond_contig, cond_cpu, _cond_device) = prepare_cpu_input(cond_ptr);
    let (true_contig, true_cpu, true_device) = prepare_cpu_input(true_ptr);
    let (false_contig, false_cpu, _false_device) = prepare_cpu_input(false_ptr);

    let cond = NslTensor::from_ptr(cond_cpu);
    let tv = NslTensor::from_ptr(true_cpu);
    let fv = NslTensor::from_ptr(false_cpu);

    let len = tv.len as usize;
    let cond_scalar = cond.len == 1;
    let fv_scalar = fv.len == 1;
    let tv_scalar = tv.len == 1;
    // Determine output length: max of all inputs (broadcast scalars)
    let out_len = len.max(cond.len as usize).max(fv.len as usize);
    let len = out_len;
    let ndim = tv.ndim;
    let dtype = tv.dtype;
    let shape = NslTensor::copy_shape(tv.shape, ndim);
    let strides = NslTensor::compute_strides(shape, ndim);

    let data: *mut c_void = if dtype == 1 {
        let buf = checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
        for i in 0..len {
            let ci = if cond_scalar { 0 } else { i };
            let c_val = if cond.dtype == 1 {
                unsafe { *cond.data_f32().add(ci) != 0.0_f32 }
            } else {
                unsafe { *cond.data_f64().add(ci) != 0.0_f64 }
            };
            let val = if c_val {
                unsafe { *tv.data_f32().add(if tv_scalar { 0 } else { i }) }
            } else {
                unsafe { *fv.data_f32().add(if fv_scalar { 0 } else { i }) }
            };
            unsafe { *buf.add(i) = val };
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc(len * std::mem::size_of::<f64>()) as *mut f64;
        for i in 0..len {
            let ci = if cond_scalar { 0 } else { i };
            let c_val = if cond.dtype == 1 {
                unsafe { *cond.data_f32().add(ci) != 0.0_f32 }
            } else {
                unsafe { *cond.data_f64().add(ci) != 0.0_f64 }
            };
            let val = if c_val {
                unsafe { *tv.data_f64().add(if tv_scalar { 0 } else { i }) }
            } else {
                unsafe { *fv.data_f64().add(if fv_scalar { 0 } else { i }) }
            };
            unsafe { *buf.add(i) = val };
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor::new(data, shape, strides, ndim, len as i64, 0, dtype, 1, 0));
    release_cpu_input(cond_contig, cond_cpu);
    release_cpu_input(true_contig, true_cpu);
    release_cpu_input(false_contig, false_cpu);
    publish_cpu_result_to_device(result, true_device, "ad_where")
}

// ---------------------------------------------------------------------------
// 3. nsl_tensor_scalar — create a 0-dim scalar tensor
// ---------------------------------------------------------------------------

/// Create a 0-dimensional scalar tensor holding a single value.
/// `dtype`: 0 = f64, 1 = f32. Matches the graph's working precision to avoid
/// silent precision loss in mixed-dtype backward computations.
#[no_mangle]
pub extern "C" fn nsl_tensor_scalar(val: f64, dtype: i64) -> i64 {
    let (data, dt): (*mut c_void, u16) = if dtype == 1 {
        let buf = checked_alloc(std::mem::size_of::<f32>()) as *mut f32;
        unsafe { *buf = val as f32 };
        (buf as *mut c_void, 1)
    } else {
        let buf = checked_alloc(std::mem::size_of::<f64>()) as *mut f64;
        unsafe { *buf = val };
        (buf as *mut c_void, 0)
    };
    let result = Box::new(NslTensor::new(
        data,
        std::ptr::null_mut(), // 0-dim: no shape array
        std::ptr::null_mut(), // 0-dim: no strides array
        0,                    // ndim
        1,                    // len = 1 element
        0,                    // device = CPU
        dt,                   // dtype matches request
        1,                    // owns_data
        0,                    // data_owner
    ));
    NslTensor::publish(result)
}

// ---------------------------------------------------------------------------
// 4. nsl_tensor_pad_zero — zero-pad along a dimension
// ---------------------------------------------------------------------------

/// Zero-pad tensor along `dim` with `pad_before` zeros prepended and
/// `pad_after` zeros appended.  The output shape is identical to the input
/// except `shape[dim] += pad_before + pad_after`.
#[no_mangle]
pub extern "C" fn nsl_tensor_pad_zero(
    tensor_ptr: i64,
    dim: i64,
    pad_before: i64,
    pad_after: i64,
) -> i64 {
    let (t_contig, t_cpu, source_device) = prepare_cpu_input(tensor_ptr);
    let tensor = NslTensor::from_ptr(t_cpu);
    let ndim = tensor.ndim as usize;

    if ndim == 0 {
        eprintln!("nsl: pad_zero requires at least 1 dimension");
        std::process::abort();
    }

    // Normalise dim
    let d = if dim < 0 { (dim + ndim as i64) as usize } else { dim as usize };
    if d >= ndim {
        eprintln!("nsl: pad_zero dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    // Build output shape
    let in_shape: Vec<i64> = (0..ndim).map(|i| unsafe { *tensor.shape.add(i) }).collect();
    let mut out_shape: Vec<i64> = in_shape.clone();
    out_shape[d] += pad_before + pad_after;

    let out_len: usize = out_shape.iter().map(|&s| s as usize).product();
    let dtype = tensor.dtype;
    let elem_size = if dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };

    // Allocate output as all zeros
    let out_data = checked_alloc_zeroed(out_len * elem_size);

    // Compute row-major strides for input and output
    let in_strides: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *tensor.strides.add(i) })
        .collect();
    let out_strides: Vec<usize> = {
        let mut s = vec![1usize; ndim];
        for i in (0..ndim - 1).rev() {
            s[i] = s[i + 1] * out_shape[i + 1] as usize;
        }
        s
    };

    // Iterate every element in the input and copy to padded output position
    let in_len = tensor.len as usize;
    for flat in 0..in_len {
        let mut remaining = flat;
        let mut src_offset = 0usize;
        let mut dst_offset = 0usize;
        for ax in 0..ndim {
            let stride_in_flat: usize = if ax + 1 < ndim {
                in_shape[ax + 1..].iter().map(|&s| s as usize).product()
            } else {
                1
            };
            let idx = remaining / stride_in_flat;
            remaining %= stride_in_flat;
            src_offset += idx * in_strides[ax] as usize;
            let out_idx = if ax == d { idx + pad_before as usize } else { idx };
            dst_offset += out_idx * out_strides[ax];
        }

        if dtype == 1 {
            let src = tensor.data as *const f32;
            let dst = out_data as *mut f32;
            unsafe { *dst.add(dst_offset) = *src.add(src_offset) };
        } else {
            let src = tensor.data as *const f64;
            let dst = out_data as *mut f64;
            unsafe { *dst.add(dst_offset) = *src.add(src_offset) };
        }
    }

    // Build output tensor
    let shape_ptr = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &s) in out_shape.iter().enumerate() {
        unsafe { *shape_ptr.add(i) = s };
    }
    let strides_ptr = NslTensor::compute_strides(shape_ptr, ndim as i64);

    let result = Box::new(NslTensor::new(
        out_data as *mut c_void,
        shape_ptr,
        strides_ptr,
        ndim as i64,
        out_len as i64,
        0,
        dtype,
        1,
        0,
    ));
    release_cpu_input(t_contig, t_cpu);
    publish_cpu_result_to_device(result, source_device, "ad_pad_zero")
}

// ---------------------------------------------------------------------------
// 5. nsl_tensor_scatter_add — scatter-add for embedding backward
// ---------------------------------------------------------------------------

/// Scatter-add: for each row `i` in `src`, add it to `output[indices[i]]`.
///
/// Produces a 2D output of shape `[vocab_size, embed_dim]` (sum over the
/// sequence dimension).  `dim` is currently unused but reserved for future
/// generalisation to arbitrary axes.
///
/// `src` shape: `[seq_len, embed_dim]`
/// `indices` shape: `[seq_len]` (integer values, read with `read_index`)
/// Output shape: `[vocab_size, embed_dim]` where `vocab_size = max_index + 1`
/// computed from the actual indices present.
#[no_mangle]
pub extern "C" fn nsl_tensor_scatter_add(src_ptr: i64, indices_ptr: i64, dim: i64) -> i64 {
    let _ = dim; // reserved for future generalisation
    let (src_contig, src_cpu, source_device) = prepare_cpu_input(src_ptr);
    let src = NslTensor::from_ptr(src_cpu);

    let (idx_contig, idx_cpu, _idx_device) = prepare_cpu_input(indices_ptr);
    let idx = NslTensor::from_ptr(idx_cpu);

    if src.ndim < 2 {
        eprintln!("nsl: scatter_add requires src to be at least 2D");
        std::process::abort();
    }

    let seq_len = unsafe { *src.shape.add(0) } as usize;
    let embed_dim = unsafe { *src.shape.add(1) } as usize;
    let n_indices = idx.len as usize;

    if seq_len != n_indices {
        eprintln!(
            "nsl: scatter_add: src seq_len {} != indices len {}",
            seq_len, n_indices
        );
        std::process::abort();
    }

    // Determine vocab_size = max(indices) + 1
    let mut vocab_size: usize = 0;
    for i in 0..n_indices {
        let tok = idx.read_index(i) as usize;
        if tok + 1 > vocab_size {
            vocab_size = tok + 1;
        }
    }

    let dtype = src.dtype;
    let out_len = vocab_size * embed_dim;
    let elem_size = if dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data = checked_alloc_zeroed(out_len * elem_size);

    if dtype == 1 {
        let src_f = src.data_f32();
        let dst_f = out_data as *mut f32;
        for i in 0..seq_len {
            let tok = idx.read_index(i) as usize;
            for e in 0..embed_dim {
                unsafe {
                    *dst_f.add(tok * embed_dim + e) += *src_f.add(i * embed_dim + e);
                }
            }
        }
    } else {
        let src_d = src.data_f64();
        let dst_d = out_data as *mut f64;
        for i in 0..seq_len {
            let tok = idx.read_index(i) as usize;
            for e in 0..embed_dim {
                unsafe {
                    *dst_d.add(tok * embed_dim + e) += *src_d.add(i * embed_dim + e);
                }
            }
        }
    }

    let shape_ptr = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *shape_ptr = vocab_size as i64;
        *shape_ptr.add(1) = embed_dim as i64;
    }
    let strides_ptr = NslTensor::compute_strides(shape_ptr, 2);

    let result = Box::new(NslTensor::new(
        out_data as *mut c_void,
        shape_ptr,
        strides_ptr,
        2,
        out_len as i64,
        0,
        dtype,
        1,
        0,
    ));
    release_cpu_input(src_contig, src_cpu);
    release_cpu_input(idx_contig, idx_cpu);
    publish_cpu_result_to_device(result, source_device, "ad_scatter_add")
}

// ---------------------------------------------------------------------------
// 5b. nsl_embedding_backward — scatter-add grad rows into weight-shaped zeros
// ---------------------------------------------------------------------------

/// Embedding backward: creates a zeros tensor matching the weight shape, then
/// scatter-adds gradient rows at the given index positions.
///
/// `grad` shape: `[seq_len, embed_dim]`
/// `indices` shape: `[seq_len]` (integer token indices)
/// `weight` shape: `[vocab_size, embed_dim]` (used only for sizing the output)
///
/// Returns: gradient w.r.t. weight, shape `[vocab_size, embed_dim]`
#[no_mangle]
pub extern "C" fn nsl_embedding_backward(
    grad_ptr: i64,
    indices_ptr: i64,
    weight_ptr: i64,
) -> i64 {
    let grad_c = nsl_tensor_contiguous(grad_ptr);
    let idx_c = nsl_tensor_contiguous(indices_ptr);
    let weight_c = nsl_tensor_contiguous(weight_ptr);
    let out_device = NslTensor::from_ptr(grad_c).device;
    let (grad_cpu, grad_needs_free) = if NslTensor::from_ptr(grad_c).device > 0 {
        (nsl_tensor_to_device(grad_c, 0), true)
    } else {
        (grad_c, false)
    };
    let (idx_cpu, idx_needs_free) = if NslTensor::from_ptr(idx_c).device > 0 {
        (nsl_tensor_to_device(idx_c, 0), true)
    } else {
        (idx_c, false)
    };
    let (weight_cpu, weight_needs_free) = if NslTensor::from_ptr(weight_c).device > 0 {
        (nsl_tensor_to_device(weight_c, 0), true)
    } else {
        (weight_c, false)
    };
    let grad = NslTensor::from_ptr(grad_cpu);
    let idx = NslTensor::from_ptr(idx_cpu);
    let weight = NslTensor::from_ptr(weight_cpu);

    if weight.ndim < 2 {
        eprintln!("nsl: embedding_backward requires 2D weight, got {}D", weight.ndim);
        std::process::abort();
    }

    let vocab_size = unsafe { *weight.shape } as usize;
    let embed_dim = unsafe { *weight.shape.add(1) } as usize;
    let seq_len = idx.len as usize;
    let dtype = grad.dtype;

    let elem_size = if dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_len = vocab_size * embed_dim;
    let out_data = checked_alloc_zeroed(out_len * elem_size);

    if dtype == 1 {
        let src = grad.data_f32();
        let dst = out_data as *mut f32;
        for i in 0..seq_len {
            let tok = idx.read_index(i) as usize;
            if tok < vocab_size {
                for e in 0..embed_dim {
                    unsafe { *dst.add(tok * embed_dim + e) += *src.add(i * embed_dim + e); }
                }
            }
        }
    } else {
        let src = grad.data_f64();
        let dst = out_data as *mut f64;
        for i in 0..seq_len {
            let tok = idx.read_index(i) as usize;
            if tok < vocab_size {
                for e in 0..embed_dim {
                    unsafe { *dst.add(tok * embed_dim + e) += *src.add(i * embed_dim + e); }
                }
            }
        }
    }

    let shape_ptr = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *shape_ptr = vocab_size as i64;
        *shape_ptr.add(1) = embed_dim as i64;
    }
    let strides_ptr = NslTensor::compute_strides(shape_ptr, 2);
    let result = Box::new(NslTensor::new(
        out_data as *mut c_void,
        shape_ptr,
        strides_ptr,
        2,
        out_len as i64,
        0,
        dtype,
        1,
        0,
    ));
    if grad_needs_free { nsl_tensor_free(grad_cpu); }
    if idx_needs_free { nsl_tensor_free(idx_cpu); }
    if weight_needs_free { nsl_tensor_free(weight_cpu); }
    nsl_tensor_free(grad_c);
    nsl_tensor_free(idx_c);
    nsl_tensor_free(weight_c);
    publish_cpu_result_to_device(result, out_device, "ad_embedding_backward")
}

// ---------------------------------------------------------------------------
// 5c. nsl_cross_entropy_backward — CE gradient: (softmax(logits) - one_hot(targets)) / N
// ---------------------------------------------------------------------------

/// Compute the cross-entropy loss backward gradient w.r.t. logits.
///
/// `logits` shape: `[N, C]`
/// `targets` shape: `[N]` (integer class indices, with negative values ignored)
/// `grad_output` shape: scalar or `[1]`
///
/// Returns: `grad_output * (softmax(logits) - one_hot(targets)) / num_valid`
/// for valid targets, and zero rows for ignored targets (< 0).
#[no_mangle]
pub extern "C" fn nsl_cross_entropy_backward(
    grad_output_ptr: i64,
    logits_ptr: i64,
    targets_ptr: i64,
) -> i64 {
    let logits_c = nsl_tensor_contiguous(logits_ptr);
    let targets_c = nsl_tensor_contiguous(targets_ptr);
    let grad_out_c = nsl_tensor_contiguous(grad_output_ptr);
    let out_device = NslTensor::from_ptr(logits_c).device;
    let (logits_cpu, logits_needs_free) = if NslTensor::from_ptr(logits_c).device > 0 {
        (nsl_tensor_to_device(logits_c, 0), true)
    } else {
        (logits_c, false)
    };
    let (targets_cpu, targets_needs_free) = if NslTensor::from_ptr(targets_c).device > 0 {
        (nsl_tensor_to_device(targets_c, 0), true)
    } else {
        (targets_c, false)
    };
    let (grad_out_cpu, grad_out_needs_free) = if NslTensor::from_ptr(grad_out_c).device > 0 {
        (nsl_tensor_to_device(grad_out_c, 0), true)
    } else {
        (grad_out_c, false)
    };
    let logits = NslTensor::from_ptr(logits_cpu);
    let targets = NslTensor::from_ptr(targets_cpu);
    let grad_out = NslTensor::from_ptr(grad_out_cpu);

    if logits.ndim < 2 {
        eprintln!("nsl: cross_entropy_backward requires 2D logits, got {}D", logits.ndim);
        std::process::abort();
    }

    let n = unsafe { *logits.shape } as usize; // batch size
    let c = unsafe { *logits.shape.add(1) } as usize; // num classes
    let dtype = logits.dtype;

    let mut num_valid = 0usize;
    for i in 0..n {
        if targets.read_index(i) >= 0 {
            num_valid += 1;
        }
    }
    let denom = num_valid.max(1) as f64;

    // Compute softmax along last dim, then subtract 1.0 at target positions
    let elem_size = if dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data = checked_alloc(n * c * elem_size);

    // Get grad_output scalar
    let go = if grad_out.len == 1 {
        match grad_out.dtype {
            1 => (unsafe { *grad_out.data_f32() }) as f64,
            4 => unsafe { *(grad_out.data as *const i32) as f64 },
            _ => unsafe { *grad_out.data_f64() },
        }
    } else {
        1.0
    };

    if dtype == 1 {
        let src = logits.data_f32();
        let dst = out_data as *mut f32;
        for i in 0..n {
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for j in 0..c {
                let v = unsafe { *src.add(i * c + j) };
                if v > max_val { max_val = v; }
            }
            // Compute softmax
            let mut sum_exp = 0.0f32;
            for j in 0..c {
                let v = unsafe { *src.add(i * c + j) };
                let e = (v - max_val).exp();
                unsafe { *dst.add(i * c + j) = e; }
                sum_exp += e;
            }
            let target_raw = targets.read_index(i);
            let valid = target_raw >= 0;
            let target_idx = target_raw as usize;
            let scale = (go / denom) as f32;
            for j in 0..c {
                let sm = unsafe { *dst.add(i * c + j) } / sum_exp;
                let one_hot = if valid && j == target_idx { 1.0f32 } else { 0.0f32 };
                unsafe {
                    *dst.add(i * c + j) = if valid {
                        (sm - one_hot) * scale
                    } else {
                        0.0
                    };
                }
            }
        }
    } else {
        let src = logits.data_f64();
        let dst = out_data as *mut f64;
        for i in 0..n {
            let mut max_val = f64::NEG_INFINITY;
            for j in 0..c {
                let v = unsafe { *src.add(i * c + j) };
                if v > max_val { max_val = v; }
            }
            let mut sum_exp = 0.0f64;
            for j in 0..c {
                let v = unsafe { *src.add(i * c + j) };
                let e = (v - max_val).exp();
                unsafe { *dst.add(i * c + j) = e; }
                sum_exp += e;
            }
            let target_raw = targets.read_index(i);
            let valid = target_raw >= 0;
            let target_idx = target_raw as usize;
            let scale = go / denom;
            for j in 0..c {
                let sm = unsafe { *dst.add(i * c + j) } / sum_exp;
                let one_hot = if valid && j == target_idx { 1.0 } else { 0.0 };
                unsafe {
                    *dst.add(i * c + j) = if valid {
                        (sm - one_hot) * scale
                    } else {
                        0.0
                    };
                }
            }
        }
    }

    let shape_ptr = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *shape_ptr = n as i64;
        *shape_ptr.add(1) = c as i64;
    }
    let strides_ptr = NslTensor::compute_strides(shape_ptr, 2);
    let result = Box::new(NslTensor::new(
        out_data as *mut c_void,
        shape_ptr,
        strides_ptr,
        2,
        (n * c) as i64,
        0,
        dtype,
        1,
        0,
    ));
    if logits_needs_free { nsl_tensor_free(logits_cpu); }
    if targets_needs_free { nsl_tensor_free(targets_cpu); }
    if grad_out_needs_free { nsl_tensor_free(grad_out_cpu); }
    nsl_tensor_free(logits_c);
    nsl_tensor_free(targets_c);
    nsl_tensor_free(grad_out_c);
    publish_cpu_result_to_device(result, out_device, "ad_cross_entropy_backward")
}

// ---------------------------------------------------------------------------
// 6. nsl_tensor_logsoftmax — numerically-stable log-softmax along `dim`
// ---------------------------------------------------------------------------

/// Compute `log(softmax(x, dim))` using the log-sum-exp trick for stability.
///
/// `result[i] = x[i] - logsumexp(x, dim)`
#[no_mangle]
pub extern "C" fn nsl_tensor_logsoftmax(tensor_ptr: i64, dim: i64) -> i64 {
    // GPU dispatch: native GPU log_softmax kernel (last dim) or CPU-redirect (other dims)
    {
        let tensor = NslTensor::from_ptr(tensor_ptr);
        if tensor.device > 0 {
            let ndim = tensor.ndim;
            let d = if dim < 0 { ndim + dim } else { dim };
            // Native GPU kernel for last-dim log_softmax (the common case)
            if d == ndim - 1 {
                #[cfg(feature = "cuda")]
                {
                    let c_ptr = nsl_tensor_contiguous(tensor_ptr);
                    let result = crate::cuda::gpu_log_softmax_f32(c_ptr);
                    nsl_tensor_free(c_ptr);
                    if autodiff::is_recording() {
                        NslTensor::from_ptr(result).refcount.fetch_add(1, Ordering::SeqCst);
                        autodiff::maybe_record(autodiff::TapeOp::LogSoftmax {
                            a: tensor_ptr, out: result, saved_out: result, dim,
                        });
                    }
                    return result;
                }
            }
            // Non-last-dim: CPU redirect
            let cpu_t = nsl_tensor_to_device(tensor_ptr, 0);
            let result = nsl_tensor_logsoftmax(cpu_t, dim);
            let gpu_result = nsl_tensor_to_device(result, tensor.device as i64);
            nsl_tensor_free(cpu_t);
            nsl_tensor_free(result);
            if autodiff::is_recording() {
                NslTensor::from_ptr(gpu_result).refcount.fetch_add(1, Ordering::SeqCst);
                autodiff::maybe_record(autodiff::TapeOp::LogSoftmax {
                    a: tensor_ptr, out: gpu_result, saved_out: gpu_result, dim,
                });
            }
            return gpu_result;
        }
    }
    let a_c = nsl_tensor_contiguous(tensor_ptr);
    let a = NslTensor::from_ptr(a_c);
    let len = a.len as usize;
    let ndim = a.ndim as usize;

    // Normalise dim
    let d = if dim < 0 { (ndim as i64 + dim) as usize } else { dim as usize };
    if d >= ndim {
        eprintln!("nsl: logsoftmax: dim {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let a_shape: Vec<i64> = (0..ndim).map(|i| unsafe { *a.shape.add(i) }).collect();
    let a_strides: Vec<i64> = (0..ndim).map(|i| unsafe { *a.strides.add(i) }).collect();
    let dim_size = a_shape[d] as usize;
    let num_slices = len / dim_size;

    let dtype = a.dtype;
    let shape = NslTensor::copy_shape(a.shape, a.ndim);
    let strides = NslTensor::compute_strides(shape, a.ndim);

    let data: *mut c_void = if dtype == 1 {
        let buf = checked_alloc(len * std::mem::size_of::<f32>()) as *mut f32;
        for slice_idx in 0..num_slices {
            let base = logsoftmax_base_offset(slice_idx, &a_shape, &a_strides, d);
            // Find max for numerical stability
            let mut max_val = f32::NEG_INFINITY;
            for k in 0..dim_size {
                let v = unsafe { *a.data_f32().add(base + k * a_strides[d] as usize) };
                if v > max_val { max_val = v; }
            }
            // log-sum-exp
            let mut log_sum = 0.0_f32;
            for k in 0..dim_size {
                let v = unsafe { *a.data_f32().add(base + k * a_strides[d] as usize) };
                log_sum += (v - max_val).exp();
            }
            let log_sum = log_sum.ln() + max_val;
            // output[i] = x[i] - log_sum
            for k in 0..dim_size {
                let offset = base + k * a_strides[d] as usize;
                let v = unsafe { *a.data_f32().add(offset) };
                unsafe { *buf.add(offset) = v - log_sum };
            }
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc(len * std::mem::size_of::<f64>()) as *mut f64;
        for slice_idx in 0..num_slices {
            let base = logsoftmax_base_offset(slice_idx, &a_shape, &a_strides, d);
            let mut max_val = f64::NEG_INFINITY;
            for k in 0..dim_size {
                let v = unsafe { *a.data_f64().add(base + k * a_strides[d] as usize) };
                if v > max_val { max_val = v; }
            }
            let mut log_sum = 0.0_f64;
            for k in 0..dim_size {
                let v = unsafe { *a.data_f64().add(base + k * a_strides[d] as usize) };
                log_sum += (v - max_val).exp();
            }
            let log_sum = log_sum.ln() + max_val;
            for k in 0..dim_size {
                let offset = base + k * a_strides[d] as usize;
                let v = unsafe { *a.data_f64().add(offset) };
                unsafe { *buf.add(offset) = v - log_sum };
            }
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor::new(data, shape, strides, a.ndim, a.len, a.device, dtype, 1, 0));
    let out = NslTensor::publish(result);
    nsl_tensor_free(a_c);
    if autodiff::is_recording() {
        NslTensor::from_ptr(out).refcount.fetch_add(1, Ordering::SeqCst);
        autodiff::maybe_record(autodiff::TapeOp::LogSoftmax {
            a: tensor_ptr, out, saved_out: out, dim,
        });
    }
    out
}

/// Compute the base offset (in element units) for a reduction slice, skipping
/// dimension `d`.  This mirrors the base-offset calculation in softmax.
#[inline]
fn logsoftmax_base_offset(
    slice_idx: usize,
    a_shape: &[i64],
    a_strides: &[i64],
    d: usize,
) -> usize {
    let ndim = a_shape.len();
    let mut remaining = slice_idx;
    let mut base = 0usize;
    for axis in (0..ndim).rev() {
        if axis == d { continue; }
        let idx = remaining % a_shape[axis] as usize;
        remaining /= a_shape[axis] as usize;
        base += idx * a_strides[axis] as usize;
    }
    base
}

// ---------------------------------------------------------------------------
// 7. nsl_tensor_repeat — spatial repeat for avgpool backward
// ---------------------------------------------------------------------------

/// Upsample by repeating each spatial element `kernel × kernel` times along
/// the last two dimensions.
///
/// For a `[N, C, H, W]` tensor, output shape is `[N, C, H*kernel, W*kernel]`.
/// For lower-rank tensors the last two dims are used similarly.
#[no_mangle]
pub extern "C" fn nsl_tensor_repeat(tensor_ptr: i64, kernel: i64) -> i64 {
    let (t_contig, t_cpu, source_device) = prepare_cpu_input(tensor_ptr);
    let tensor = NslTensor::from_ptr(t_cpu);
    let ndim = tensor.ndim as usize;
    let k = kernel as usize;

    if ndim < 2 {
        eprintln!("nsl: tensor_repeat requires at least 2 dimensions");
        std::process::abort();
    }

    let in_shape: Vec<i64> = (0..ndim).map(|i| unsafe { *tensor.shape.add(i) }).collect();
    let mut out_shape = in_shape.clone();
    out_shape[ndim - 2] *= kernel;
    out_shape[ndim - 1] *= kernel;

    let out_len: usize = out_shape.iter().map(|&s| s as usize).product();
    let dtype = tensor.dtype;
    let elem_size = if dtype == 1 { std::mem::size_of::<f32>() } else { std::mem::size_of::<f64>() };
    let out_data = checked_alloc(out_len * elem_size);

    // Precompute number of "outer" elements (everything except last 2 dims)
    let outer: usize = in_shape[..ndim - 2].iter().map(|&s| s as usize).product();
    let in_h = in_shape[ndim - 2] as usize;
    let in_w = in_shape[ndim - 1] as usize;
    let out_h = out_shape[ndim - 2] as usize;
    let out_w = out_shape[ndim - 1] as usize;

    if dtype == 1 {
        let src = tensor.data as *const f32;
        let dst = out_data as *mut f32;
        for n in 0..outer {
            for ih in 0..in_h {
                for iw in 0..in_w {
                    let src_val = unsafe { *src.add(n * in_h * in_w + ih * in_w + iw) };
                    for ri in 0..k {
                        for rj in 0..k {
                            let oh = ih * k + ri;
                            let ow = iw * k + rj;
                            unsafe { *dst.add(n * out_h * out_w + oh * out_w + ow) = src_val };
                        }
                    }
                }
            }
        }
    } else {
        let src = tensor.data as *const f64;
        let dst = out_data as *mut f64;
        for n in 0..outer {
            for ih in 0..in_h {
                for iw in 0..in_w {
                    let src_val = unsafe { *src.add(n * in_h * in_w + ih * in_w + iw) };
                    for ri in 0..k {
                        for rj in 0..k {
                            let oh = ih * k + ri;
                            let ow = iw * k + rj;
                            unsafe { *dst.add(n * out_h * out_w + oh * out_w + ow) = src_val };
                        }
                    }
                }
            }
        }
    }

    let shape_ptr = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &s) in out_shape.iter().enumerate() {
        unsafe { *shape_ptr.add(i) = s };
    }
    let strides_ptr = NslTensor::compute_strides(shape_ptr, ndim as i64);

    let result = Box::new(NslTensor::new(
        out_data as *mut c_void,
        shape_ptr,
        strides_ptr,
        ndim as i64,
        out_len as i64,
        0,
        dtype,
        1,
        0,
    ));
    release_cpu_input(t_contig, t_cpu);
    publish_cpu_result_to_device(result, source_device, "ad_repeat")
}

// ---------------------------------------------------------------------------
// 8. nsl_tensor_rope_inverse — inverse RoPE rotation (negate angles)
// ---------------------------------------------------------------------------

/// Apply an inverse RoPE rotation by negating the rotation angles.
///
/// Standard RoPE applies:
///   `y = x * cos(θ) + rotate_half(x) * sin(θ)`
///
/// The inverse simply uses `−sin(θ)`:
///   `x = y * cos(θ) + rotate_half(y) * (−sin(θ))`
///
/// In practice this amounts to applying `rotate_half` with the *negative*
/// of the second half of the last dimension, i.e. swapping and negating the
/// same pair that `rotate_half` operates on — but applied with flipped sign
/// to the original.
///
/// For the source-AD backward pass the simplest correct implementation is to
/// call `rotate_half` on the tensor with the rotation applied in the opposite
/// direction.  Since `rotate_half` permutes `[−x2, x1]`, the inverse of
/// `rotate_half` (= unrotate) is `rotate_half` applied twice (it is its own
/// inverse modulo sign): `inv = rotate_half(−rotate_half(t))`.  However the
/// most numerically sensible implementation for the backward is just to negate
/// the odd-indexed half of the last dim (which reverses the cross-term),
/// exactly mirroring what PyTorch does for `apply_rotary_pos_emb` backward.
///
/// Implementation: for every chunk of `last_dim` elements:
///   `out[0..half]    = t[0..half]`
///   `out[half..last] = −t[half..last]`
///
/// This negates the second half, reversing the rotational encoding effect
/// without re-applying the cos/sin (the caller handles that).
#[no_mangle]
pub extern "C" fn nsl_tensor_rope_inverse(tensor_ptr: i64, dim: i64) -> i64 {
    let _ = dim; // dim is reserved; currently always operates on the last dim
    let t_c = nsl_tensor_contiguous(tensor_ptr);
    let source_device = NslTensor::from_ptr(t_c).device;
    let t_cpu = if source_device > 0 {
        super::nsl_tensor_to_device(t_c, 0)
    } else {
        t_c
    };
    let tensor = NslTensor::from_ptr(t_cpu);
    let ndim = tensor.ndim as usize;

    if ndim == 0 {
        eprintln!("nsl: rope_inverse requires at least 1 dimension");
        std::process::abort();
    }

    let last_dim = unsafe { *tensor.shape.add(ndim - 1) } as usize;
    #[allow(clippy::manual_is_multiple_of)]
    if last_dim % 2 != 0 {
        eprintln!("nsl: rope_inverse requires even last dimension, got {}", last_dim);
        std::process::abort();
    }
    let half = last_dim / 2;
    let total = tensor.len as usize;
    let num_chunks = total / last_dim;
    let dtype = tensor.dtype;

    let shape = NslTensor::copy_shape(tensor.shape, tensor.ndim);
    let strides = NslTensor::compute_strides(shape, tensor.ndim);

    let data: *mut c_void = if dtype == 1 {
        let buf = checked_alloc(total * std::mem::size_of::<f32>()) as *mut f32;
        let src = tensor.data_f32();
        for chunk in 0..num_chunks {
            let base = chunk * last_dim;
            for i in 0..half {
                // first half: unchanged
                unsafe { *buf.add(base + i) = *src.add(base + i) };
                // second half: negated
                unsafe { *buf.add(base + half + i) = -(*src.add(base + half + i)) };
            }
        }
        buf as *mut c_void
    } else {
        let buf = checked_alloc(total * std::mem::size_of::<f64>()) as *mut f64;
        let src = tensor.data_f64();
        for chunk in 0..num_chunks {
            let base = chunk * last_dim;
            for i in 0..half {
                unsafe { *buf.add(base + i) = *src.add(base + i) };
                unsafe { *buf.add(base + half + i) = -(*src.add(base + half + i)) };
            }
        }
        buf as *mut c_void
    };

    let result = Box::new(NslTensor::new(
        data,
        shape,
        strides,
        tensor.ndim,
        tensor.len,
        0,
        dtype,
        1,
        0,
    ));
    if t_cpu != t_c {
        nsl_tensor_free(t_cpu);
    }
    nsl_tensor_free(t_c);
    publish_cpu_result_to_device(result, source_device, "ad_rope_inverse")
}

// ---------------------------------------------------------------------------
// 9. nsl_tensor_batchnorm — batch normalization (normalizes over dim 0)
// ---------------------------------------------------------------------------

/// BatchNorm: normalize over the batch dimension (dim=0).
/// `input` shape [B, C, ...], `gamma` shape [C], `beta` shape [C].
/// For each channel c: out[:, c, ...] = gamma[c] * (x[:, c, ...] - mean_c) / sqrt(var_c + eps) + beta[c]
/// where mean_c and var_c are computed over the batch dimension.
#[no_mangle]
pub extern "C" fn nsl_tensor_batchnorm(
    input_ptr: i64, gamma_ptr: i64, beta_ptr: i64, eps: f64, _training: i64,
) -> i64 {
    let t_c = nsl_tensor_contiguous(input_ptr);
    let inp = NslTensor::from_ptr(t_c);
    let g = NslTensor::from_ptr(gamma_ptr);
    let b = NslTensor::from_ptr(beta_ptr);

    let ndim = inp.ndim as usize;
    let len = inp.len as usize;
    let dtype = inp.dtype;
    let shape_slice = unsafe { std::slice::from_raw_parts(inp.shape, ndim) };

    // batch_size = shape[0], channels = shape[1], spatial = product of shape[2..]
    let batch = if ndim > 0 { shape_slice[0] as usize } else { 1 };
    let channels = if ndim > 1 { shape_slice[1] as usize } else { len / batch.max(1) };
    let spatial: usize = shape_slice[2..].iter().map(|&d| d as usize).product::<usize>().max(1);

    let elem_size = if dtype == 1 { 4 } else { 8 };
    let out_data = checked_alloc(len * elem_size);

    if dtype == 1 {
        let src = inp.data as *const f32;
        let dst = out_data as *mut f32;
        let gamma_data = g.data as *const f32;
        let beta_data = b.data as *const f32;
        let eps_f = eps as f32;

        for c in 0..channels {
            // Compute mean and variance over batch and spatial dims
            let mut sum = 0.0_f32;
            let count = (batch * spatial) as f32;
            for n in 0..batch {
                for s in 0..spatial {
                    let idx = n * channels * spatial + c * spatial + s;
                    sum += unsafe { *src.add(idx) };
                }
            }
            let mean = sum / count;

            let mut var_sum = 0.0_f32;
            for n in 0..batch {
                for s in 0..spatial {
                    let idx = n * channels * spatial + c * spatial + s;
                    let diff = unsafe { *src.add(idx) } - mean;
                    var_sum += diff * diff;
                }
            }
            let var = var_sum / count;
            let rstd = 1.0 / (var + eps_f).sqrt();

            let gc = if c < g.len as usize { unsafe { *gamma_data.add(c) } } else { 1.0 };
            let bc = if c < b.len as usize { unsafe { *beta_data.add(c) } } else { 0.0 };

            for n in 0..batch {
                for s in 0..spatial {
                    let idx = n * channels * spatial + c * spatial + s;
                    let x = unsafe { *src.add(idx) };
                    unsafe { *dst.add(idx) = gc * (x - mean) * rstd + bc };
                }
            }
        }
    } else {
        let src = inp.data as *const f64;
        let dst = out_data as *mut f64;
        let gamma_data = g.data as *const f64;
        let beta_data = b.data as *const f64;

        for c in 0..channels {
            let mut sum = 0.0_f64;
            let count = (batch * spatial) as f64;
            for n in 0..batch {
                for s in 0..spatial {
                    let idx = n * channels * spatial + c * spatial + s;
                    sum += unsafe { *src.add(idx) };
                }
            }
            let mean = sum / count;

            let mut var_sum = 0.0_f64;
            for n in 0..batch {
                for s in 0..spatial {
                    let idx = n * channels * spatial + c * spatial + s;
                    let diff = unsafe { *src.add(idx) } - mean;
                    var_sum += diff * diff;
                }
            }
            let var = var_sum / count;
            let rstd = 1.0 / (var + eps).sqrt();

            let gc = if c < g.len as usize { unsafe { *gamma_data.add(c) } } else { 1.0 };
            let bc = if c < b.len as usize { unsafe { *beta_data.add(c) } } else { 0.0 };

            for n in 0..batch {
                for s in 0..spatial {
                    let idx = n * channels * spatial + c * spatial + s;
                    let x = unsafe { *src.add(idx) };
                    unsafe { *dst.add(idx) = gc * (x - mean) * rstd + bc };
                }
            }
        }
    }

    let shape = NslTensor::copy_shape(inp.shape, inp.ndim);
    let strides = NslTensor::compute_strides(shape, inp.ndim);
    let result = Box::new(NslTensor::new(
        out_data as *mut c_void, shape, strides, inp.ndim, len as i64, 0, dtype, 1, 0,
    ));
    let out = NslTensor::publish(result);
    nsl_tensor_free(t_c);
    out
}

// ---------------------------------------------------------------------------
// 10. nsl_tensor_avgpool2d — average pooling
// ---------------------------------------------------------------------------

/// AvgPool2d: average pooling over the last 2 spatial dimensions.
/// Input shape: [B, C, H, W]. Output shape: [B, C, H/kernel, W/kernel].
#[no_mangle]
pub extern "C" fn nsl_tensor_avgpool2d(
    input_ptr: i64, kernel_h: i64, kernel_w: i64, stride: i64, padding: i64,
) -> i64 {
    let t_c = nsl_tensor_contiguous(input_ptr);
    let inp = NslTensor::from_ptr(t_c);
    let ndim = inp.ndim as usize;
    let dtype = inp.dtype;
    let shape_slice = unsafe { std::slice::from_raw_parts(inp.shape, ndim) };

    let kh = kernel_h as usize;
    let kw = kernel_w as usize;
    let s = stride as usize;
    let p = padding as usize;

    // Assume NCHW layout: [..., H, W]
    let h_in = shape_slice[ndim - 2] as usize;
    let w_in = shape_slice[ndim - 1] as usize;
    let h_out = (h_in + 2 * p - kh) / s + 1;
    let w_out = (w_in + 2 * p - kw) / s + 1;

    // Outer dims = product of all dims except last 2
    let outer: usize = shape_slice[..ndim - 2].iter().map(|&d| d as usize).product::<usize>().max(1);
    let total_out = outer * h_out * w_out;

    let elem_size = if dtype == 1 { 4 } else { 8 };
    let out_data = checked_alloc(total_out * elem_size);

    if dtype == 1 {
        let src = inp.data as *const f32;
        let dst = out_data as *mut f32;
        let pool_area = (kh * kw) as f32;

        for o in 0..outer {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0_f32;
                    for khr in 0..kh {
                        for kwr in 0..kw {
                            let ih = oh * s + khr;
                            let iw = ow * s + kwr;
                            if ih < h_in + 2 * p && iw < w_in + 2 * p {
                                let ih_actual = ih.wrapping_sub(p);
                                let iw_actual = iw.wrapping_sub(p);
                                if ih_actual < h_in && iw_actual < w_in {
                                    sum += unsafe { *src.add(o * h_in * w_in + ih_actual * w_in + iw_actual) };
                                }
                            }
                        }
                    }
                    unsafe { *dst.add(o * h_out * w_out + oh * w_out + ow) = sum / pool_area };
                }
            }
        }
    } else {
        let src = inp.data as *const f64;
        let dst = out_data as *mut f64;
        let pool_area = (kh * kw) as f64;

        for o in 0..outer {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut sum = 0.0_f64;
                    for khr in 0..kh {
                        for kwr in 0..kw {
                            let ih = oh * s + khr;
                            let iw = ow * s + kwr;
                            let ih_actual = ih.wrapping_sub(p);
                            let iw_actual = iw.wrapping_sub(p);
                            if ih_actual < h_in && iw_actual < w_in {
                                sum += unsafe { *src.add(o * h_in * w_in + ih_actual * w_in + iw_actual) };
                            }
                        }
                    }
                    unsafe { *dst.add(o * h_out * w_out + oh * w_out + ow) = sum / pool_area };
                }
            }
        }
    }

    // Build output shape: replace last 2 dims with h_out, w_out
    let out_shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    #[allow(clippy::needless_range_loop)]
    for i in 0..ndim - 2 {
        unsafe { *out_shape.add(i) = shape_slice[i] };
    }
    unsafe { *out_shape.add(ndim - 2) = h_out as i64 };
    unsafe { *out_shape.add(ndim - 1) = w_out as i64 };
    let out_strides = NslTensor::compute_strides(out_shape, ndim as i64);
    let result = Box::new(NslTensor::new(
        out_data as *mut c_void, out_shape, out_strides, ndim as i64, total_out as i64, 0, dtype, 1, 0,
    ));
    let out = NslTensor::publish(result);
    nsl_tensor_free(t_c);
    out
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::creation::tensor_from_shape_list;
    use crate::list::{nsl_list_new, nsl_list_push};
    use crate::cpu::create_tensor_with_shape_rs_dtype;
    use crate::tensor::nsl_tensor_to_device;

    // Helper: create a 1D f32 tensor from a slice of values
    fn make_1d_f32(vals: &[f32]) -> i64 {
        let shape_list = nsl_list_new();
        nsl_list_push(shape_list, vals.len() as i64);
        let ptr = tensor_from_shape_list(shape_list, 0.0);
        let t = NslTensor::from_ptr(ptr);
        for (i, &v) in vals.iter().enumerate() {
            unsafe { *t.data_f32().add(i) = v };
        }
        ptr
    }

    // Helper: read a 1D f32 tensor into a Vec
    fn read_1d_f32(ptr: i64) -> Vec<f32> {
        let t = NslTensor::from_ptr(ptr);
        (0..t.len as usize).map(|i| unsafe { *t.data_f32().add(i) }).collect()
    }

    #[test]
    fn test_tensor_compare_gt() {
        let a = make_1d_f32(&[1.0, 3.0, 2.0, 0.0]);
        let b = make_1d_f32(&[0.0, 3.0, 3.0, -1.0]);
        let out = nsl_tensor_compare(a, b, 0); // Gt
        let vals = read_1d_f32(out);
        // 1>0 yes, 3>3 no, 2>3 no, 0>-1 yes
        assert_eq!(vals, vec![1.0_f32, 0.0, 0.0, 1.0]);
        nsl_tensor_free(a);
        nsl_tensor_free(b);
        nsl_tensor_free(out);
    }

    #[test]
    fn test_tensor_compare_eq() {
        let a = make_1d_f32(&[1.0, 2.0, 3.0]);
        let b = make_1d_f32(&[1.0, 0.0, 3.0]);
        let out = nsl_tensor_compare(a, b, 4); // Eq
        let vals = read_1d_f32(out);
        assert_eq!(vals, vec![1.0_f32, 0.0, 1.0]);
        nsl_tensor_free(a);
        nsl_tensor_free(b);
        nsl_tensor_free(out);
    }

    #[test]
    fn test_tensor_where_select() {
        let cond = make_1d_f32(&[1.0, 0.0, 1.0, 0.0]);
        let tv   = make_1d_f32(&[10.0, 20.0, 30.0, 40.0]);
        let fv   = make_1d_f32(&[1.0,  2.0,  3.0,  4.0]);
        let out = nsl_tensor_where(cond, tv, fv);
        let vals = read_1d_f32(out);
        assert_eq!(vals, vec![10.0_f32, 2.0, 30.0, 4.0]);
        nsl_tensor_free(cond);
        nsl_tensor_free(tv);
        nsl_tensor_free(fv);
        nsl_tensor_free(out);
    }

    #[test]
    fn test_tensor_scalar_creation_f32() {
        let s = nsl_tensor_scalar(3.14, 1); // dtype=1 (f32)
        let t = NslTensor::from_ptr(s);
        assert_eq!(t.ndim, 0);
        assert_eq!(t.len, 1);
        assert_eq!(t.dtype, 1);
        let val = unsafe { *t.data_f32() };
        assert!((val - 3.14_f32).abs() < 1e-5_f32);
        nsl_tensor_free(s);
    }

    #[test]
    fn test_tensor_scalar_creation_f64() {
        let s = nsl_tensor_scalar(3.14159265358979, 0); // dtype=0 (f64)
        let t = NslTensor::from_ptr(s);
        assert_eq!(t.ndim, 0);
        assert_eq!(t.len, 1);
        assert_eq!(t.dtype, 0);
        let val = unsafe { *t.data_f64() };
        assert!((val - 3.14159265358979_f64).abs() < 1e-14_f64);
        nsl_tensor_free(s);
    }

    #[test]
    fn test_tensor_pad_zero() {
        // Tensor [1, 2, 3] padded with 1 zero before and 2 zeros after along dim 0
        // → [0, 1, 2, 3, 0, 0]
        let a = make_1d_f32(&[1.0, 2.0, 3.0]);
        let out = nsl_tensor_pad_zero(a, 0, 1, 2);
        let t = NslTensor::from_ptr(out);
        assert_eq!(t.len, 6);
        let vals = read_1d_f32(out);
        assert_eq!(vals, vec![0.0_f32, 1.0, 2.0, 3.0, 0.0, 0.0]);
        nsl_tensor_free(a);
        nsl_tensor_free(out);
    }

    #[test]
    fn test_tensor_logsoftmax() {
        // log-softmax of [1, 1, 1] should be [-ln3, -ln3, -ln3]
        let a = make_1d_f32(&[1.0, 1.0, 1.0]);
        let out = nsl_tensor_logsoftmax(a, 0);
        let vals = read_1d_f32(out);
        let expected = -(3.0_f32.ln());
        for &v in &vals {
            assert!((v - expected).abs() < 1e-5_f32, "got {v}, expected {expected}");
        }
        // logsumexp([1,1,1]) = sum of logsoftmax = log(1) = 0
        let sum: f32 = vals.iter().map(|v| v.exp()).sum();
        assert!((sum - 1.0_f32).abs() < 1e-5_f32);
        nsl_tensor_free(a);
        nsl_tensor_free(out);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn test_cross_entropy_backward_gpu_f64_seed() {
        let logits = create_tensor_with_shape_rs_dtype(&[2, 3], 1);
        let logits_t = NslTensor::from_ptr(logits);
        let logits_vals = [1.0_f32, 0.0, -1.0, 0.5, 1.5, -0.5];
        for (index, value) in logits_vals.iter().enumerate() {
            unsafe { *logits_t.data_f32().add(index) = *value; }
        }

        let targets = create_tensor_with_shape_rs_dtype(&[2], 4);
        let targets_t = NslTensor::from_ptr(targets);
        unsafe {
            *targets_t.data_i32().add(0) = 0;
            *targets_t.data_i32().add(1) = 1;
        }

        let grad_out = nsl_tensor_scalar(1.0, 0);
        let logits_gpu = nsl_tensor_to_device(logits, 1);
        let grad_gpu = nsl_cross_entropy_backward(grad_out, logits_gpu, targets);
        let grad_cpu = nsl_tensor_to_device(grad_gpu, 0);
        let grad_t = NslTensor::from_ptr(grad_cpu);

        assert_eq!(grad_t.ndim, 2);
        assert_eq!(unsafe { *grad_t.shape.add(0) }, 2);
        assert_eq!(unsafe { *grad_t.shape.add(1) }, 3);
        for i in 0..6 {
            let value = unsafe { *grad_t.data_f64().add(i) };
            assert!(value.is_finite(), "gradient element {i} was not finite: {value}");
        }

        nsl_tensor_free(logits);
        nsl_tensor_free(targets);
        nsl_tensor_free(grad_out);
        nsl_tensor_free(logits_gpu);
        nsl_tensor_free(grad_gpu);
        nsl_tensor_free(grad_cpu);
    }
}

// ---------------------------------------------------------------------------
// nsl_tensor_reduce_to_shape — reduce gradient to match parameter shape
// ---------------------------------------------------------------------------

/// Reduce a gradient tensor to match a target parameter's shape by summing
/// over leading batch dimensions. Used in matmul backward when the input
/// has more dimensions than the weight (broadcasting).
///
/// If grad and target already have the same shape, returns a clone of grad.
/// Otherwise sums over leading dimensions until ndims match, then reshapes.
#[no_mangle]
pub extern "C" fn nsl_tensor_reduce_to_shape(grad_ptr: i64, target_ptr: i64) -> i64 {
    if grad_ptr == 0 || target_ptr == 0 {
        eprintln!("nsl_tensor_reduce_to_shape: null pointer (grad={}, target={})", grad_ptr, target_ptr);
        return grad_ptr;
    }
    let grad = NslTensor::from_ptr(grad_ptr);
    let target = NslTensor::from_ptr(target_ptr);

    let g_ndim = grad.ndim as usize;
    let t_ndim = target.ndim as usize;

    // If same ndim and same shape, just clone
    if g_ndim == t_ndim {
        let same = (0..g_ndim).all(|i| unsafe { *grad.shape.add(i) == *target.shape.add(i) });
        if same {
            return super::nsl_tensor_clone(grad_ptr);
        }
    }

    // If grad has more dims, sum over the leading dims
    if g_ndim > t_ndim {
        let extra = g_ndim - t_ndim;
        let mut result = super::nsl_tensor_clone(grad_ptr);
        // Sum over dim 0, `extra` times (each sum reduces ndim by 1 if keepdim=0)
        for _ in 0..extra {
            let old = result;
            result = super::reduction::nsl_tensor_sum_dim(result, 0, 0); // keepdim=0
            if old != grad_ptr {
                super::nsl_tensor_free(old);
            }
        }
        return result;
    }

    // Same ndim but different shape — sum over broadcast dims
    let target_shape: Vec<i64> = (0..t_ndim).map(|i| unsafe { *target.shape.add(i) }).collect();
    crate::autodiff::grad_utils::reduce_grad_for_broadcast(grad_ptr, &target_shape)
}
