//! Quantized tensor support for NeuralScript.
//!
//! Provides INT8 and INT4 quantization with per-tensor, per-channel, and per-group granularity.
//! Uses asymmetric affine quantization (weight-only RTN).

use std::ffi::c_void;

use crate::memory::{checked_alloc, checked_alloc_zeroed, checked_free};
use crate::tensor::NslTensor;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const DTYPE_INT8: i64 = 0;
const DTYPE_INT4: i64 = 1;

const GRAN_PER_TENSOR: i64 = 0;
const GRAN_PER_CHANNEL: i64 = 1;
const GRAN_PER_GROUP: i64 = 2;

// ---------------------------------------------------------------------------
// QuantizedTensor struct
// ---------------------------------------------------------------------------

#[repr(C)]
pub struct QuantizedTensor {
    /// Packed quantized data (INT8: 1 byte/elem, INT4: 2 elems/byte, low nibble first)
    pub(crate) data: *mut u8,
    /// Per-scale-group scale factors
    pub(crate) scale: *mut f32,
    /// Per-scale-group zero points
    pub(crate) zero_point: *mut u8,
    /// Tensor shape (same as original)
    pub(crate) shape: *mut i64,
    /// Number of dimensions
    pub(crate) ndim: i64,
    /// Quantized dtype: 0 = INT8, 1 = INT4
    pub(crate) dtype: i64,
    /// Granularity: 0 = PerTensor, 1 = PerChannel, 2 = PerGroup
    pub(crate) granularity: i64,
    /// Axis for per-channel / per-group granularity
    pub(crate) gran_axis: i64,
    /// Group size (only meaningful for PerGroup)
    pub(crate) group_size: i64,
    /// Number of scale/zero_point entries
    pub(crate) num_scales: i64,
    /// Reference count
    pub(crate) refcount: i64,
}

impl QuantizedTensor {
    fn from_ptr(ptr: i64) -> &'static mut QuantizedTensor {
        unsafe { &mut *(ptr as *mut QuantizedTensor) }
    }
}

// ---------------------------------------------------------------------------
// Allocation helpers
// ---------------------------------------------------------------------------

/// Compute total number of elements from shape.
fn total_elements(shape: *const i64, ndim: i64) -> i64 {
    let mut total: i64 = 1;
    for i in 0..ndim as usize {
        total *= unsafe { *shape.add(i) };
    }
    total
}

/// Compute number of scale/zero_point entries needed.
fn compute_num_scales(shape: *const i64, ndim: i64, granularity: i64, gran_axis: i64, group_size: i64) -> i64 {
    match granularity {
        GRAN_PER_TENSOR => 1,
        GRAN_PER_CHANNEL => {
            if ndim == 0 { return 1; }
            unsafe { *shape.add(gran_axis as usize) }
        }
        GRAN_PER_GROUP => {
            if ndim == 0 { return 1; }
            let axis_size = unsafe { *shape.add(gran_axis as usize) };
            let num_groups = (axis_size + group_size - 1) / group_size;
            // outer_size = product of all dims except the axis dim
            let mut outer_size: i64 = 1;
            for i in 0..ndim as usize {
                if i != gran_axis as usize {
                    outer_size *= unsafe { *shape.add(i) };
                }
            }
            outer_size * num_groups
        }
        _ => 1,
    }
}

/// Allocate a QuantizedTensor. Data buffer is zeroed for INT4 packing safety.
fn alloc_qtensor(
    shape: *const i64,
    ndim: i64,
    dtype: i64,
    granularity: i64,
    gran_axis: i64,
    group_size: i64,
) -> *mut QuantizedTensor {
    let total = total_elements(shape, ndim) as usize;
    let num_scales = compute_num_scales(shape, ndim, granularity, gran_axis, group_size);

    // Data buffer size: INT8 = 1 byte/elem, INT4 = ceil(total/2) bytes
    let data_bytes = match dtype {
        DTYPE_INT4 => (total + 1) / 2,
        DTYPE_INT8 => total,
        _ => { eprintln!("nsl: unknown quantization dtype {}", dtype); std::process::abort(); }
    };

    // MUST use zeroed alloc — int4_pack uses |= which blends with existing bits
    let data = checked_alloc_zeroed(data_bytes);
    let scale = checked_alloc((num_scales as usize) * std::mem::size_of::<f32>()) as *mut f32;
    let zero_point = checked_alloc_zeroed(num_scales as usize);

    // Copy shape
    let shape_copy = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(shape, shape_copy, ndim as usize) };

    let qt = Box::new(QuantizedTensor {
        data,
        scale,
        zero_point,
        shape: shape_copy,
        ndim,
        dtype,
        granularity,
        gran_axis,
        group_size,
        num_scales,
        refcount: 1,
    });
    Box::into_raw(qt)
}

// ---------------------------------------------------------------------------
// Memory management FFI
// ---------------------------------------------------------------------------

#[no_mangle]
pub extern "C" fn nsl_qtensor_free(ptr: i64) {
    if ptr == 0 { return; }
    let qt = QuantizedTensor::from_ptr(ptr);
    let total = total_elements(qt.shape, qt.ndim) as usize;

    let data_bytes = match qt.dtype {
        DTYPE_INT4 => (total + 1) / 2,
        DTYPE_INT8 => total,
        _ => { eprintln!("nsl: unknown quantization dtype {}", qt.dtype); std::process::abort(); }
    };

    unsafe {
        checked_free(qt.data, data_bytes);
        checked_free(qt.scale as *mut u8, (qt.num_scales as usize) * std::mem::size_of::<f32>());
        checked_free(qt.zero_point, qt.num_scales as usize);
        checked_free(qt.shape as *mut u8, (qt.ndim as usize) * std::mem::size_of::<i64>());
        drop(Box::from_raw(qt as *mut QuantizedTensor));
    }
}

#[no_mangle]
pub extern "C" fn nsl_qtensor_addref(ptr: i64) {
    if ptr == 0 { return; }
    let qt = QuantizedTensor::from_ptr(ptr);
    qt.refcount += 1;
}

#[no_mangle]
pub extern "C" fn nsl_qtensor_release(ptr: i64) {
    if ptr == 0 { return; }
    let qt = QuantizedTensor::from_ptr(ptr);
    qt.refcount -= 1;
    if qt.refcount <= 0 {
        nsl_qtensor_free(ptr);
    }
}

// ---------------------------------------------------------------------------
// INT4 pack/unpack helpers
// ---------------------------------------------------------------------------

/// Pack a value (0..15) into the INT4 packed buffer at the given element index.
/// Low nibble first: even indices go in bits [3:0], odd indices in bits [7:4].
/// Buffer MUST be zeroed — this uses |= to set bits.
#[inline]
fn int4_pack(data: *mut u8, index: usize, value: u8) {
    let byte_idx = index / 2;
    unsafe {
        if index % 2 == 0 {
            // Low nibble
            *data.add(byte_idx) |= value & 0x0F;
        } else {
            // High nibble
            *data.add(byte_idx) |= (value & 0x0F) << 4;
        }
    }
}

/// Unpack a value (0..15) from the INT4 packed buffer at the given element index.
#[inline]
fn int4_unpack(data: *const u8, index: usize) -> u8 {
    let byte_idx = index / 2;
    unsafe {
        let byte = *data.add(byte_idx);
        if index % 2 == 0 {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        }
    }
}

// ---------------------------------------------------------------------------
// Quantization math helpers (asymmetric affine)
// ---------------------------------------------------------------------------

/// Compute scale and zero point for a slice of f64 values.
/// Returns (scale, zero_point).
fn compute_scale_zp(values: &[f64], qmin: f64, qmax: f64) -> (f32, u8) {
    if values.is_empty() {
        return (1.0, 0);
    }

    let mut min_val = f64::MAX;
    let mut max_val = f64::MIN;
    for &v in values {
        if v < min_val { min_val = v; }
        if v > max_val { max_val = v; }
    }

    // Handle zero-variance case: all values identical or nearly so
    let raw_range = max_val - min_val;
    if raw_range < 1e-7 {
        // Map the constant value to the midpoint of the quantized range
        let mid_q = ((qmin + qmax) / 2.0).round();
        // scale chosen so that dequant(mid_q) ≈ the constant value
        // dequant: (q - zp) * scale = val → if zp=0, scale = val/mid_q
        // But we need to handle val=0 too. Use: scale = max(|val|, 1e-7) / mid_q
        let abs_val = min_val.abs().max(1e-7);
        let scale = abs_val / mid_q;
        let zp = (mid_q - min_val / scale).round().clamp(qmin, qmax) as u8;
        return (scale as f32, zp);
    }

    let scale = raw_range / (qmax - qmin);
    let zp = (-min_val / scale).round().clamp(qmin, qmax) as u8;
    (scale as f32, zp)
}

/// Quantize a single f64 value given scale and zero point.
#[inline]
fn quantize_val(x: f64, scale: f32, zp: u8, qmin: f64, qmax: f64) -> u8 {
    let q = (x / scale as f64 + zp as f64).round().clamp(qmin, qmax);
    q as u8
}

/// Dequantize a single u8 value given scale and zero point.
#[inline]
fn dequantize_val(q: u8, scale: f32, zp: u8) -> f64 {
    (q as f64 - zp as f64) * scale as f64
}

// ---------------------------------------------------------------------------
// nsl_qtensor_quantize — weight-only RTN quantization
// ---------------------------------------------------------------------------

/// Quantize an NslTensor into a QuantizedTensor.
///
/// Arguments (all as i64 for FFI):
///   tensor_ptr  - pointer to source NslTensor (f64 data)
///   dtype       - 0 = INT8, 1 = INT4
///   granularity - 0 = PerTensor, 1 = PerChannel, 2 = PerGroup
///   gran_axis   - axis for per-channel/per-group
///   group_size  - group size for per-group (ignored otherwise)
///
/// Returns: pointer to QuantizedTensor as i64
#[no_mangle]
pub extern "C" fn nsl_qtensor_quantize(
    tensor_ptr: i64,
    dtype: i64,
    granularity: i64,
    gran_axis: i64,
    group_size: i64,
) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim;
    let total = tensor.len as usize;

    let qt = alloc_qtensor(tensor.shape, ndim, dtype, granularity, gran_axis, group_size);
    let qt_ref = unsafe { &mut *qt };

    let (qmin, qmax) = match dtype {
        DTYPE_INT4 => (0.0_f64, 15.0_f64),
        DTYPE_INT8 => (0.0_f64, 255.0_f64),
        _ => { eprintln!("nsl: unknown quantization dtype {}", dtype); std::process::abort(); }
    };

    // Read all source data into a Vec for easier slicing
    let src: Vec<f64> = (0..total).map(|i| unsafe { *tensor.data_f64().add(i) }).collect();

    match granularity {
        GRAN_PER_TENSOR => {
            let (scale, zp) = compute_scale_zp(&src, qmin, qmax);
            unsafe {
                *qt_ref.scale = scale;
                *qt_ref.zero_point = zp;
            }
            for i in 0..total {
                let q = quantize_val(src[i], scale, zp, qmin, qmax);
                match dtype {
                    DTYPE_INT4 => int4_pack(qt_ref.data, i, q),
                    DTYPE_INT8 => unsafe { *qt_ref.data.add(i) = q },
                    _ => unreachable!(),
                }
            }
        }
        GRAN_PER_CHANNEL => {
            // Per-channel: one scale/zp per slice along gran_axis
            let axis = gran_axis as usize;
            let axis_size = unsafe { *tensor.shape.add(axis) } as usize;

            // Compute stride products for indexing
            let mut inner_size: usize = 1;
            for d in (axis + 1)..(ndim as usize) {
                inner_size *= unsafe { *tensor.shape.add(d) } as usize;
            }
            let outer_size = total / (axis_size * inner_size);

            for ch in 0..axis_size {
                // Gather all elements belonging to this channel
                let mut channel_vals = Vec::with_capacity(outer_size * inner_size);
                for o in 0..outer_size {
                    for inner in 0..inner_size {
                        let idx = o * axis_size * inner_size + ch * inner_size + inner;
                        channel_vals.push(src[idx]);
                    }
                }

                let (scale, zp) = compute_scale_zp(&channel_vals, qmin, qmax);
                unsafe {
                    *qt_ref.scale.add(ch) = scale;
                    *qt_ref.zero_point.add(ch) = zp;
                }

                // Quantize those elements in-place
                let mut vi = 0;
                for o in 0..outer_size {
                    for inner in 0..inner_size {
                        let idx = o * axis_size * inner_size + ch * inner_size + inner;
                        let q = quantize_val(channel_vals[vi], scale, zp, qmin, qmax);
                        match dtype {
                            DTYPE_INT4 => int4_pack(qt_ref.data, idx, q),
                            DTYPE_INT8 => unsafe { *qt_ref.data.add(idx) = q },
                            _ => unreachable!(),
                        }
                        vi += 1;
                    }
                }
            }
        }
        GRAN_PER_GROUP => {
            // Per-group: divide the axis dimension into groups of group_size
            let axis = gran_axis as usize;
            let axis_size = unsafe { *tensor.shape.add(axis) } as usize;
            let gs = group_size as usize;
            let num_groups = (axis_size + gs - 1) / gs;

            let mut inner_size: usize = 1;
            for d in (axis + 1)..(ndim as usize) {
                inner_size *= unsafe { *tensor.shape.add(d) } as usize;
            }
            let outer_size = total / (axis_size * inner_size);

            let mut scale_idx: usize = 0;
            for o in 0..outer_size {
                for g in 0..num_groups {
                    let start = g * gs;
                    let end = (start + gs).min(axis_size);

                    // Gather group values across inner dims
                    let mut group_vals = Vec::with_capacity((end - start) * inner_size);
                    for ch in start..end {
                        for inner in 0..inner_size {
                            let idx = o * axis_size * inner_size + ch * inner_size + inner;
                            group_vals.push(src[idx]);
                        }
                    }

                    let (scale, zp) = compute_scale_zp(&group_vals, qmin, qmax);
                    unsafe {
                        *qt_ref.scale.add(scale_idx) = scale;
                        *qt_ref.zero_point.add(scale_idx) = zp;
                    }

                    // Quantize
                    let mut vi = 0;
                    for ch in start..end {
                        for inner in 0..inner_size {
                            let idx = o * axis_size * inner_size + ch * inner_size + inner;
                            let q = quantize_val(group_vals[vi], scale, zp, qmin, qmax);
                            match dtype {
                                DTYPE_INT4 => int4_pack(qt_ref.data, idx, q),
                                DTYPE_INT8 => unsafe { *qt_ref.data.add(idx) = q },
                            _ => unreachable!(),
                            }
                            vi += 1;
                        }
                    }

                    scale_idx += 1;
                }
            }
        }
        _ => {
            eprintln!("nsl: unknown quantization granularity {}", granularity);
            std::process::abort();
        }
    }

    qt as i64
}

// ---------------------------------------------------------------------------
// nsl_qtensor_dequantize — reconstruct full NslTensor
// ---------------------------------------------------------------------------

/// Dequantize a QuantizedTensor back to a full-precision NslTensor (f64).
///
/// Returns: pointer to NslTensor as i64
#[no_mangle]
pub extern "C" fn nsl_qtensor_dequantize(qtensor_ptr: i64) -> i64 {
    let qt = QuantizedTensor::from_ptr(qtensor_ptr);
    let ndim = qt.ndim;
    let total = total_elements(qt.shape, ndim) as usize;

    // Allocate output NslTensor
    let data = checked_alloc(total * std::mem::size_of::<f64>()) as *mut f64;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(qt.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);

    match qt.granularity {
        GRAN_PER_TENSOR => {
            let scale = unsafe { *qt.scale };
            let zp = unsafe { *qt.zero_point };
            for i in 0..total {
                let q = match qt.dtype {
                    DTYPE_INT4 => int4_unpack(qt.data, i),
                    DTYPE_INT8 => unsafe { *qt.data.add(i) },
                    _ => unreachable!(),
                };
                unsafe { *data.add(i) = dequantize_val(q, scale, zp) };
            }
        }
        GRAN_PER_CHANNEL => {
            let axis = qt.gran_axis as usize;
            let axis_size = unsafe { *qt.shape.add(axis) } as usize;
            let mut inner_size: usize = 1;
            for d in (axis + 1)..(ndim as usize) {
                inner_size *= unsafe { *qt.shape.add(d) } as usize;
            }
            let outer_size = total / (axis_size * inner_size);

            for o in 0..outer_size {
                for ch in 0..axis_size {
                    let scale = unsafe { *qt.scale.add(ch) };
                    let zp = unsafe { *qt.zero_point.add(ch) };
                    for inner in 0..inner_size {
                        let idx = o * axis_size * inner_size + ch * inner_size + inner;
                        let q = match qt.dtype {
                            DTYPE_INT4 => int4_unpack(qt.data, idx),
                            DTYPE_INT8 => unsafe { *qt.data.add(idx) },
                            _ => unreachable!(),
                        };
                        unsafe { *data.add(idx) = dequantize_val(q, scale, zp) };
                    }
                }
            }
        }
        GRAN_PER_GROUP => {
            let axis = qt.gran_axis as usize;
            let axis_size = unsafe { *qt.shape.add(axis) } as usize;
            let gs = qt.group_size as usize;
            let num_groups = (axis_size + gs - 1) / gs;
            let mut inner_size: usize = 1;
            for d in (axis + 1)..(ndim as usize) {
                inner_size *= unsafe { *qt.shape.add(d) } as usize;
            }
            let outer_size = total / (axis_size * inner_size);

            let mut scale_idx: usize = 0;
            for o in 0..outer_size {
                for g in 0..num_groups {
                    let start = g * gs;
                    let end = (start + gs).min(axis_size);
                    let scale = unsafe { *qt.scale.add(scale_idx) };
                    let zp = unsafe { *qt.zero_point.add(scale_idx) };

                    for ch in start..end {
                        for inner in 0..inner_size {
                            let idx = o * axis_size * inner_size + ch * inner_size + inner;
                            let q = match qt.dtype {
                                DTYPE_INT4 => int4_unpack(qt.data, idx),
                                DTYPE_INT8 => unsafe { *qt.data.add(idx) },
                            _ => unreachable!(),
                            };
                            unsafe { *data.add(idx) = dequantize_val(q, scale, zp) };
                        }
                    }
                    scale_idx += 1;
                }
            }
        }
        _ => {
            eprintln!("nsl: unknown quantization granularity {} in dequantize", qt.granularity);
            std::process::abort();
        }
    }

    let out = Box::new(NslTensor {
        data: data as *mut c_void,
        shape,
        strides,
        ndim,
        len: total as i64,
        refcount: 1,
        device: 0,
        dtype: 0,
        owns_data: 1,
    });
    Box::into_raw(out) as i64
}

// ---------------------------------------------------------------------------
// Mixed-precision matmul
// ---------------------------------------------------------------------------

/// Mixed-precision matmul: NslTensor (f64) @ QuantizedTensor -> NslTensor (f64).
/// Dequantizes the quantized weight on-the-fly during matmul.
/// qw must be 2D [K, N]. x must have last dim = K.
#[no_mangle]
pub extern "C" fn nsl_qtensor_matmul_mixed(x_ptr: i64, qw_ptr: i64) -> i64 {
    // Validate qw is 2D
    let qw = QuantizedTensor::from_ptr(qw_ptr);
    if qw.ndim != 2 {
        eprintln!(
            "nsl: nsl_qtensor_matmul_mixed: quantized weight must be 2D, got {}D",
            qw.ndim
        );
        std::process::abort();
    }

    // Validate x last dim matches qw dim 0
    let x = NslTensor::from_ptr(x_ptr);
    let qw_k = unsafe { *qw.shape.add(0) };
    let x_last = if x.ndim > 0 {
        unsafe { *x.shape.add((x.ndim - 1) as usize) }
    } else {
        eprintln!("nsl: nsl_qtensor_matmul_mixed: x must be at least 1D");
        std::process::abort();
    };
    if x_last != qw_k {
        eprintln!(
            "nsl: nsl_qtensor_matmul_mixed: x last dim ({}) != qw dim 0 ({})",
            x_last, qw_k
        );
        std::process::abort();
    }

    // Dequantize the quantized weight to a temporary NslTensor
    let deq_ptr = nsl_qtensor_dequantize(qw_ptr);

    // Perform matmul: x @ deq
    let result_ptr = crate::tensor::nsl_tensor_matmul(x_ptr, deq_ptr);

    // Free the temporary dequantized tensor
    crate::tensor::nsl_tensor_free(deq_ptr);

    result_ptr
}

// ---------------------------------------------------------------------------
// Query helpers
// ---------------------------------------------------------------------------

/// Return the dtype of a QuantizedTensor (0 = INT8, 1 = INT4).
#[no_mangle]
pub extern "C" fn nsl_qtensor_dtype(qtensor_ptr: i64) -> i64 {
    QuantizedTensor::from_ptr(qtensor_ptr).dtype
}

/// Return the shape of a QuantizedTensor as a 1-D NslTensor of i64 values
/// (stored as f64 for compatibility with NslTensor's f64 data).
#[no_mangle]
pub extern "C" fn nsl_qtensor_shape(qtensor_ptr: i64) -> i64 {
    let qt = QuantizedTensor::from_ptr(qtensor_ptr);
    let ndim = qt.ndim as usize;

    let data = checked_alloc(ndim * std::mem::size_of::<f64>()) as *mut f64;
    for i in 0..ndim {
        unsafe { *data.add(i) = *qt.shape.add(i) as f64 };
    }

    let shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *shape = qt.ndim };
    let strides = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *strides = 1 };

    let out = Box::new(NslTensor {
        data: data as *mut c_void,
        shape,
        strides,
        ndim: 1,
        len: qt.ndim,
        refcount: 1,
        device: 0,
        dtype: 0,
        owns_data: 1,
    });
    Box::into_raw(out) as i64
}
