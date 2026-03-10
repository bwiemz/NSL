use std::cell::Cell;

use crate::autodiff;
use crate::list::{nsl_list_new, nsl_list_push, NslList};
use crate::memory::{checked_alloc, checked_alloc_zeroed, checked_free};

// ---------------------------------------------------------------------------
// Global training mode (thread-local)
// ---------------------------------------------------------------------------
thread_local! {
    static TRAINING_MODE: Cell<bool> = const { Cell::new(false) };
}

#[no_mangle]
pub extern "C" fn nsl_set_training_mode(mode: i8) {
    TRAINING_MODE.with(|t| t.set(mode != 0));
}

#[no_mangle]
pub extern "C" fn nsl_is_training() -> i8 {
    TRAINING_MODE.with(|t| t.get() as i8)
}

#[repr(C)]
pub struct NslTensor {
    pub(crate) data: *mut f64,
    pub(crate) shape: *mut i64,
    pub(crate) strides: *mut i64,
    pub(crate) ndim: i64,
    pub(crate) len: i64,
    pub(crate) refcount: i64,
}

impl NslTensor {
    pub(crate) fn from_ptr(ptr: i64) -> &'static mut NslTensor {
        unsafe { &mut *(ptr as *mut NslTensor) }
    }

    pub(crate) fn compute_strides(shape: *const i64, ndim: i64) -> *mut i64 {
        let n = ndim as usize;
        let strides = checked_alloc(n * std::mem::size_of::<i64>()) as *mut i64;
        if n > 0 {
            unsafe {
                *strides.add(n - 1) = 1;
                for i in (0..n - 1).rev() {
                    *strides.add(i) = *strides.add(i + 1) * *shape.add(i + 1);
                }
            }
        }
        strides
    }

    fn total_elements(shape: *const i64, ndim: i64) -> i64 {
        let mut total: i64 = 1;
        for i in 0..ndim as usize {
            total *= unsafe { *shape.add(i) };
        }
        total
    }
}

/// Helper: create a tensor from a shape list, filling data with a given value.
fn tensor_from_shape_list(shape_list: i64, fill: f64) -> i64 {
    let list = NslList::from_ptr(shape_list);
    let ndim = list.len;

    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim as usize {
        unsafe { *shape.add(i) = *list.data.add(i) };
    }

    let len = NslTensor::total_elements(shape, ndim);
    let data_size = (len as usize) * std::mem::size_of::<f64>();
    let data = if fill == 0.0 {
        checked_alloc_zeroed(data_size) as *mut f64
    } else {
        let data = checked_alloc(data_size) as *mut f64;
        for i in 0..len as usize {
            unsafe { *data.add(i) = fill };
        }
        data
    };

    let strides = NslTensor::compute_strides(shape, ndim);

    let tensor = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    Box::into_raw(tensor) as i64
}

/// Create a 0-d scalar tensor containing a single f64 value.
fn create_scalar_tensor(value: f64) -> i64 {
    let data = checked_alloc(std::mem::size_of::<f64>()) as *mut f64;
    unsafe { *data = value };
    let tensor = Box::new(NslTensor {
        data,
        shape: std::ptr::null_mut(),
        strides: std::ptr::null_mut(),
        ndim: 0,
        len: 1,
        refcount: 1,
    });
    Box::into_raw(tensor) as i64
}

// === Creation ===

#[no_mangle]
pub extern "C" fn nsl_tensor_zeros(shape_list: i64) -> i64 {
    tensor_from_shape_list(shape_list, 0.0)
}

#[no_mangle]
pub extern "C" fn nsl_tensor_ones(shape_list: i64) -> i64 {
    tensor_from_shape_list(shape_list, 1.0)
}

#[no_mangle]
pub extern "C" fn nsl_tensor_full(shape_list: i64, value: f64) -> i64 {
    tensor_from_shape_list(shape_list, value)
}

#[no_mangle]
pub extern "C" fn nsl_tensor_rand(shape_list: i64) -> i64 {
    let ptr = tensor_from_shape_list(shape_list, 0.0);
    let tensor = NslTensor::from_ptr(ptr);
    // Simple LCG random number generator (no external deps)
    let mut seed: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(12345);
    for i in 0..tensor.len as usize {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let val = (seed >> 11) as f64 / (1u64 << 53) as f64;
        unsafe { *tensor.data.add(i) = val };
    }
    ptr
}

#[no_mangle]
pub extern "C" fn nsl_tensor_randn(shape_list: i64) -> i64 {
    let ptr = tensor_from_shape_list(shape_list, 0.0);
    let tensor = NslTensor::from_ptr(ptr);
    // Box-Muller transform: generate N(0,1) from uniform samples
    let mut seed: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(12345);

    let len = tensor.len as usize;
    let mut i = 0;
    while i + 1 < len {
        // Generate two uniform (0,1) samples via LCG
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((seed >> 11) as f64 / (1u64 << 53) as f64).max(1e-15); // avoid log(0)
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (seed >> 11) as f64 / (1u64 << 53) as f64;

        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos();
        let z1 = mag * (2.0 * std::f64::consts::PI * u2).sin();
        unsafe {
            *tensor.data.add(i) = z0;
            *tensor.data.add(i + 1) = z1;
        }
        i += 2;
    }
    // If odd number of elements, generate one more pair and use first
    if i < len {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u1 = ((seed >> 11) as f64 / (1u64 << 53) as f64).max(1e-15);
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let u2 = (seed >> 11) as f64 / (1u64 << 53) as f64;

        let mag = (-2.0 * u1.ln()).sqrt();
        let z0 = mag * (2.0 * std::f64::consts::PI * u2).cos();
        unsafe { *tensor.data.add(i) = z0 };
    }
    let _ = seed; // suppress unused warning
    ptr
}

#[no_mangle]
pub extern "C" fn nsl_tensor_arange(start: f64, stop: f64, step: f64) -> i64 {
    if step == 0.0 {
        eprintln!("nsl: tensor arange step cannot be zero");
        std::process::abort();
    }
    let len = ((stop - start) / step).ceil().max(0.0) as i64;

    // Create 1D tensor
    let ndim: i64 = 1;
    let shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *shape = len };

    let strides = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *strides = 1 };

    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;
    for i in 0..len as usize {
        unsafe { *data.add(i) = start + (i as f64) * step };
    }

    let tensor = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    Box::into_raw(tensor) as i64
}

// === Element access ===

#[no_mangle]
pub extern "C" fn nsl_tensor_get(tensor_ptr: i64, indices_list: i64) -> f64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let indices = NslList::from_ptr(indices_list);

    if indices.len != tensor.ndim {
        eprintln!(
            "nsl: tensor index dimension mismatch (got {}, expected {})",
            indices.len, tensor.ndim
        );
        std::process::abort();
    }

    let mut offset: usize = 0;
    for i in 0..tensor.ndim as usize {
        let idx = unsafe { *indices.data.add(i) };
        let dim_size = unsafe { *tensor.shape.add(i) };
        if idx < 0 || idx >= dim_size {
            eprintln!(
                "nsl: tensor index out of bounds (index {} for dim {} of size {})",
                idx, i, dim_size
            );
            std::process::abort();
        }
        offset += (idx as usize) * (unsafe { *tensor.strides.add(i) } as usize);
    }

    unsafe { *tensor.data.add(offset) }
}

#[no_mangle]
pub extern "C" fn nsl_tensor_set(tensor_ptr: i64, indices_list: i64, value: f64) {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let indices = NslList::from_ptr(indices_list);

    if indices.len != tensor.ndim {
        eprintln!(
            "nsl: tensor index dimension mismatch (got {}, expected {})",
            indices.len, tensor.ndim
        );
        std::process::abort();
    }

    let mut offset: usize = 0;
    for i in 0..tensor.ndim as usize {
        let idx = unsafe { *indices.data.add(i) };
        let dim_size = unsafe { *tensor.shape.add(i) };
        if idx < 0 || idx >= dim_size {
            eprintln!(
                "nsl: tensor index out of bounds (index {} for dim {} of size {})",
                idx, i, dim_size
            );
            std::process::abort();
        }
        offset += (idx as usize) * (unsafe { *tensor.strides.add(i) } as usize);
    }

    unsafe { *tensor.data.add(offset) = value };
}

// === Shape operations ===

#[no_mangle]
pub extern "C" fn nsl_tensor_shape(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let list = crate::list::nsl_list_new();
    for i in 0..tensor.ndim as usize {
        crate::list::nsl_list_push(list, unsafe { *tensor.shape.add(i) });
    }
    list
}

#[no_mangle]
pub extern "C" fn nsl_tensor_ndim(tensor_ptr: i64) -> i64 {
    NslTensor::from_ptr(tensor_ptr).ndim
}

#[no_mangle]
pub extern "C" fn nsl_tensor_reshape(tensor_ptr: i64, new_shape_list: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let new_shape_nsl = NslList::from_ptr(new_shape_list);
    let new_ndim = new_shape_nsl.len;

    let new_shape = checked_alloc((new_ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    let mut new_len: i64 = 1;
    for i in 0..new_ndim as usize {
        let dim = unsafe { *new_shape_nsl.data.add(i) };
        unsafe { *new_shape.add(i) = dim };
        new_len *= dim;
    }

    if new_len != tensor.len {
        eprintln!(
            "nsl: cannot reshape tensor of size {} into shape of size {}",
            tensor.len, new_len
        );
        std::process::abort();
    }

    // Create a new tensor that shares data (increment refcount)
    tensor.refcount += 1;
    let strides = NslTensor::compute_strides(new_shape, new_ndim);

    let new_tensor = Box::new(NslTensor {
        data: tensor.data,
        shape: new_shape,
        strides,
        ndim: new_ndim,
        len: new_len,
        refcount: 1, // new wrapper has its own refcount
    });
    // Note: shared data means we need to be careful with free.
    // For M9, we just deep copy the data to keep it simple.
    let result = Box::into_raw(new_tensor) as i64;

    // Actually, let's deep copy to avoid shared ownership complexity in M9
    tensor.refcount -= 1;
    let result_tensor = NslTensor::from_ptr(result);
    let data_size = (new_len as usize) * std::mem::size_of::<f64>();
    let new_data = checked_alloc(data_size) as *mut f64;
    unsafe {
        std::ptr::copy_nonoverlapping(tensor.data, new_data, new_len as usize);
    }
    result_tensor.data = new_data;

    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_transpose(tensor_ptr: i64, dim0: i64, dim1: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);

    if dim0 < 0 || dim0 >= tensor.ndim || dim1 < 0 || dim1 >= tensor.ndim {
        eprintln!(
            "nsl: transpose dimensions out of range ({}, {} for ndim {})",
            dim0, dim1, tensor.ndim
        );
        std::process::abort();
    }

    // Deep copy with transposed shape
    let ndim = tensor.ndim;
    let new_shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim as usize {
        unsafe { *new_shape.add(i) = *tensor.shape.add(i) };
    }
    // Swap dimensions
    unsafe {
        let tmp = *new_shape.add(dim0 as usize);
        *new_shape.add(dim0 as usize) = *new_shape.add(dim1 as usize);
        *new_shape.add(dim1 as usize) = tmp;
    }

    let strides = NslTensor::compute_strides(new_shape, ndim);
    let len = tensor.len;
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    // Copy data with transposition
    let old_strides_arr: Vec<i64> = (0..ndim as usize)
        .map(|i| unsafe { *tensor.strides.add(i) })
        .collect();
    let new_strides_arr: Vec<i64> = (0..ndim as usize)
        .map(|i| unsafe { *strides.add(i) })
        .collect();

    // Iterate over all elements of the new tensor
    for flat_idx in 0..len as usize {
        // Convert flat_idx to new multi-index
        let mut remaining = flat_idx;
        let mut new_indices = vec![0usize; ndim as usize];
        for d in 0..ndim as usize {
            new_indices[d] = remaining / new_strides_arr[d] as usize;
            remaining %= new_strides_arr[d] as usize;
        }

        // Map new indices back to old indices (swap dim0 and dim1)
        let mut old_indices = new_indices.clone();
        old_indices.swap(dim0 as usize, dim1 as usize);

        // Compute old flat offset
        let mut old_offset = 0usize;
        for d in 0..ndim as usize {
            old_offset += old_indices[d] * old_strides_arr[d] as usize;
        }

        unsafe { *data.add(flat_idx) = *tensor.data.add(old_offset) };
    }

    let result = Box::new(NslTensor {
        data,
        shape: new_shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    let out_ptr = Box::into_raw(result) as i64;

    // Record on tape for autodiff
    if crate::autodiff::is_recording() {
        crate::autodiff::maybe_record(crate::autodiff::TapeOp::Transpose {
            a: tensor_ptr,
            out: out_ptr,
            dim0,
            dim1,
        });
    }

    out_ptr
}

// === Elementwise arithmetic ===

fn tensor_elementwise_op(a_ptr: i64, b_ptr: i64, op: fn(f64, f64) -> f64) -> i64 {
    let a = NslTensor::from_ptr(a_ptr);
    let b = NslTensor::from_ptr(b_ptr);

    if a.ndim != b.ndim {
        eprintln!(
            "nsl: tensor shape mismatch in elementwise op (ndim {} vs {})",
            a.ndim, b.ndim
        );
        std::process::abort();
    }
    for i in 0..a.ndim as usize {
        if unsafe { *a.shape.add(i) != *b.shape.add(i) } {
            eprintln!("nsl: tensor shape mismatch in elementwise op");
            std::process::abort();
        }
    }

    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        unsafe { *data.add(i) = op(*a.data.add(i), *b.data.add(i)) };
    }

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    Box::into_raw(result) as i64
}

#[no_mangle]
pub extern "C" fn nsl_tensor_add(a: i64, b: i64) -> i64 {
    let result = tensor_elementwise_op(a, b, |x, y| x + y);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Add { a, b, out: result });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sub(a: i64, b: i64) -> i64 {
    let result = tensor_elementwise_op(a, b, |x, y| x - y);
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Sub { a, b, out: result });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_mul(a: i64, b: i64) -> i64 {
    let result = tensor_elementwise_op(a, b, |x, y| x * y);
    if autodiff::is_recording() {
        NslTensor::from_ptr(a).refcount += 1;
        NslTensor::from_ptr(b).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Mul {
            a,
            b,
            out: result,
            saved_a: a,
            saved_b: b,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_div(a: i64, b: i64) -> i64 {
    let result = tensor_elementwise_op(a, b, |x, y| x / y);
    if autodiff::is_recording() {
        NslTensor::from_ptr(a).refcount += 1;
        NslTensor::from_ptr(b).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Div {
            a,
            b,
            out: result,
            saved_a: a,
            saved_b: b,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_neg(a_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(a_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        unsafe { *data.add(i) = -(*a.data.add(i)) };
    }

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::Neg { a: a_ptr, out: result });
    }
    result
}

// === Scalar-tensor ops ===

#[no_mangle]
pub extern "C" fn nsl_tensor_add_scalar(a_ptr: i64, s: f64) -> i64 {
    let a = NslTensor::from_ptr(a_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        unsafe { *data.add(i) = *a.data.add(i) + s };
    }

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::AddScalar { a: a_ptr, out: result });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_mul_scalar(a_ptr: i64, s: f64) -> i64 {
    let a = NslTensor::from_ptr(a_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        unsafe { *data.add(i) = *a.data.add(i) * s };
    }

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
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
    let a = NslTensor::from_ptr(a_ptr);
    let b = NslTensor::from_ptr(b_ptr);

    if a.ndim != 2 || b.ndim != 2 {
        eprintln!(
            "nsl: matmul requires 2D tensors (got {}D and {}D)",
            a.ndim, b.ndim
        );
        std::process::abort();
    }

    let m = unsafe { *a.shape.add(0) }; // rows of A
    let k = unsafe { *a.shape.add(1) }; // cols of A
    let k2 = unsafe { *b.shape.add(0) }; // rows of B
    let n = unsafe { *b.shape.add(1) }; // cols of B

    if k != k2 {
        eprintln!(
            "nsl: matmul dimension mismatch ({}x{} @ {}x{})",
            m, k, k2, n
        );
        std::process::abort();
    }

    // Result is m x n
    let ndim: i64 = 2;
    let len = m * n;
    let shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *shape.add(0) = m;
        *shape.add(1) = n;
    }
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc_zeroed((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    // Naive triple-loop matmul
    for i in 0..m as usize {
        for j in 0..k as usize {
            let a_val = unsafe { *a.data.add(i * k as usize + j) };
            for l in 0..n as usize {
                let b_val = unsafe { *b.data.add(j * n as usize + l) };
                unsafe {
                    *data.add(i * n as usize + l) += a_val * b_val;
                }
            }
        }
    }

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(a_ptr).refcount += 1;
        NslTensor::from_ptr(b_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::MatMul {
            a: a_ptr,
            b: b_ptr,
            out: result,
            saved_a: a_ptr,
            saved_b: b_ptr,
        });
    }
    result
}

// === Reductions ===

/// Helper: get the shape of a tensor as a Vec<i64>.
fn get_shape_vec(tensor: &NslTensor) -> Vec<i64> {
    (0..tensor.ndim as usize)
        .map(|i| unsafe { *tensor.shape.add(i) })
        .collect()
}

/// Helper: get the strides of a tensor as a Vec<usize>.
fn get_strides_vec(tensor: &NslTensor) -> Vec<usize> {
    (0..tensor.ndim as usize)
        .map(|i| unsafe { *tensor.strides.add(i) } as usize)
        .collect()
}

/// Helper: create a tensor with a given shape (Rust slice), returning (ptr, &mut NslTensor).
fn create_tensor_with_shape_rs(shape: &[i64]) -> i64 {
    let ndim = shape.len() as i64;
    let mut total: i64 = 1;
    for &s in shape {
        total *= s;
    }

    let shape_ptr =
        checked_alloc(shape.len() * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &s) in shape.iter().enumerate() {
        unsafe { *shape_ptr.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape_ptr, ndim);
    let data = checked_alloc_zeroed((total as usize) * std::mem::size_of::<f64>()) as *mut f64;

    let tensor = Box::new(NslTensor {
        data,
        shape: shape_ptr,
        strides,
        ndim,
        len: total,
        refcount: 1,
    });
    Box::into_raw(tensor) as i64
}

/// Global sum reduction (backward compatible wrapper).
#[no_mangle]
pub extern "C" fn nsl_tensor_sum(tensor_ptr: i64) -> i64 {
    nsl_tensor_sum_dim(tensor_ptr, -1, 0)
}

/// Global mean reduction (backward compatible wrapper).
#[no_mangle]
pub extern "C" fn nsl_tensor_mean(tensor_ptr: i64) -> i64 {
    nsl_tensor_mean_dim(tensor_ptr, -1, 0)
}

/// Sum reduction along a dimension (dim=-1 means global).
#[no_mangle]
pub extern "C" fn nsl_tensor_sum_dim(tensor_ptr: i64, dim: i64, keepdim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let input_shape = get_shape_vec(tensor);
    let keepdim_bool = keepdim != 0;

    if dim == -1 {
        // Global reduction
        let mut total = 0.0;
        for i in 0..tensor.len as usize {
            total += unsafe { *tensor.data.add(i) };
        }
        let result = create_scalar_tensor(total);
        if autodiff::is_recording() {
            autodiff::maybe_record(autodiff::TapeOp::SumReduce {
                a: tensor_ptr,
                out: result,
                dim: -1,
                keepdim: false,
                input_shape,
            });
        }
        return result;
    }

    let ndim = tensor.ndim as usize;
    let d = dim as usize;
    if d >= ndim {
        eprintln!("nsl: sum_dim dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let in_strides = get_strides_vec(tensor);

    // Compute output shape
    let out_shape: Vec<i64> = if keepdim_bool {
        input_shape.iter().enumerate()
            .map(|(i, &s)| if i == d { 1 } else { s })
            .collect()
    } else {
        input_shape.iter().enumerate()
            .filter(|&(i, _)| i != d)
            .map(|(_, &s)| s)
            .collect()
    };

    let result_ptr = create_tensor_with_shape_rs(&out_shape);
    let result = NslTensor::from_ptr(result_ptr);
    let out_strides = get_strides_vec(result);

    // Iterate all elements of input, accumulate into output
    let total_in = tensor.len as usize;
    for flat_in in 0..total_in {
        // Decompose flat_in into multi-index using input strides
        let mut remaining = flat_in;
        let mut indices = vec![0usize; ndim];
        for dd in 0..ndim {
            indices[dd] = remaining / in_strides[dd];
            remaining %= in_strides[dd];
        }

        // Compute output flat index (skip or collapse the reduced dim)
        let mut out_flat = 0usize;
        let mut oi = 0usize;
        for dd in 0..ndim {
            if dd == d {
                if keepdim_bool {
                    // dim is kept as size 1, index=0
                    oi += 1;
                }
                continue;
            }
            out_flat += indices[dd] * out_strides[oi];
            oi += 1;
        }

        let val = unsafe { *tensor.data.add(flat_in) };
        unsafe { *result.data.add(out_flat) += val };
    }

    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::SumReduce {
            a: tensor_ptr,
            out: result_ptr,
            dim,
            keepdim: keepdim_bool,
            input_shape,
        });
    }
    result_ptr
}

/// Mean reduction along a dimension (dim=-1 means global).
#[no_mangle]
pub extern "C" fn nsl_tensor_mean_dim(tensor_ptr: i64, dim: i64, keepdim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let input_shape = get_shape_vec(tensor);
    let keepdim_bool = keepdim != 0;

    if dim == -1 {
        // Global reduction
        if tensor.len == 0 {
            return create_scalar_tensor(0.0);
        }
        let num_elements = tensor.len;
        let mut total = 0.0;
        for i in 0..num_elements as usize {
            total += unsafe { *tensor.data.add(i) };
        }
        total /= num_elements as f64;
        let result = create_scalar_tensor(total);
        if autodiff::is_recording() {
            autodiff::maybe_record(autodiff::TapeOp::MeanReduce {
                a: tensor_ptr,
                out: result,
                dim: -1,
                keepdim: false,
                num_elements,
                input_shape,
            });
        }
        return result;
    }

    let ndim = tensor.ndim as usize;
    let d = dim as usize;
    if d >= ndim {
        eprintln!("nsl: mean_dim dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let dim_size = input_shape[d];

    // Sum first, then divide
    let sum_ptr = nsl_tensor_sum_dim(tensor_ptr, dim, keepdim);

    // Remove the SumReduce tape entry that sum_dim just recorded (we want MeanReduce instead)
    if autodiff::is_recording() {
        crate::autodiff::pop_last_op();
    }

    // Divide by dim_size
    let result = NslTensor::from_ptr(sum_ptr);
    for i in 0..result.len as usize {
        unsafe { *result.data.add(i) /= dim_size as f64 };
    }

    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::MeanReduce {
            a: tensor_ptr,
            out: sum_ptr,
            dim,
            keepdim: keepdim_bool,
            num_elements: dim_size,
            input_shape,
        });
    }
    sum_ptr
}

/// Reduce max along a dimension.
#[no_mangle]
pub extern "C" fn nsl_tensor_reduce_max(tensor_ptr: i64, dim: i64, keepdim: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let input_shape = get_shape_vec(tensor);
    let keepdim_bool = keepdim != 0;
    let ndim = tensor.ndim as usize;
    let d = dim as usize;

    if d >= ndim {
        eprintln!("nsl: reduce_max dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let in_strides = get_strides_vec(tensor);

    // Compute output shape
    let out_shape: Vec<i64> = if keepdim_bool {
        input_shape.iter().enumerate()
            .map(|(i, &s)| if i == d { 1 } else { s })
            .collect()
    } else {
        input_shape.iter().enumerate()
            .filter(|&(i, _)| i != d)
            .map(|(_, &s)| s)
            .collect()
    };

    let result_ptr = create_tensor_with_shape_rs(&out_shape);
    let result = NslTensor::from_ptr(result_ptr);
    let out_strides = get_strides_vec(result);
    let out_total = result.len as usize;

    // Initialize with -inf
    for i in 0..out_total {
        unsafe { *result.data.add(i) = f64::NEG_INFINITY };
    }

    // Track argmax per output position
    let mut argmax = vec![0usize; out_total];

    // Iterate all elements of input
    let total_in = tensor.len as usize;
    for flat_in in 0..total_in {
        let mut remaining = flat_in;
        let mut indices = vec![0usize; ndim];
        for dd in 0..ndim {
            indices[dd] = remaining / in_strides[dd];
            remaining %= in_strides[dd];
        }

        // Compute output flat index
        let mut out_flat = 0usize;
        let mut oi = 0usize;
        for dd in 0..ndim {
            if dd == d {
                if keepdim_bool {
                    oi += 1;
                }
                continue;
            }
            out_flat += indices[dd] * out_strides[oi];
            oi += 1;
        }

        let val = unsafe { *tensor.data.add(flat_in) };
        let cur = unsafe { *result.data.add(out_flat) };
        if val > cur {
            unsafe { *result.data.add(out_flat) = val };
            argmax[out_flat] = indices[d];
        }
    }

    if autodiff::is_recording() {
        autodiff::maybe_record(autodiff::TapeOp::ReduceMax {
            a: tensor_ptr,
            out: result_ptr,
            dim,
            keepdim: keepdim_bool,
            saved_argmax: argmax,
            input_shape,
        });
    }
    result_ptr
}

/// Gather along a dimension: output[b] = tensor[b, indices[b]] (for dim=1, 2D input).
#[no_mangle]
pub extern "C" fn nsl_tensor_gather(tensor_ptr: i64, dim: i64, indices_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let indices = NslTensor::from_ptr(indices_ptr);
    let input_shape = get_shape_vec(tensor);
    let ndim = tensor.ndim as usize;
    let d = dim as usize;

    if d >= ndim {
        eprintln!("nsl: gather dimension {} out of range for {}D tensor", dim, ndim);
        std::process::abort();
    }

    let in_strides = get_strides_vec(tensor);
    let batch = indices.len as usize;

    // Output shape: input shape with dim removed
    // For [batch, classes] with dim=1 and indices [batch] -> output [batch]
    let out_shape: Vec<i64> = input_shape.iter().enumerate()
        .filter(|&(i, _)| i != d)
        .map(|(_, &s)| s)
        .collect();

    let result_ptr = create_tensor_with_shape_rs(&out_shape);
    let result = NslTensor::from_ptr(result_ptr);

    // For 2D dim=1: output[b] = input[b, indices[b]]
    // For 2D dim=0: output[c] = input[indices[c], c]
    if ndim == 2 {
        if d == 1 {
            for b in 0..batch {
                let idx = unsafe { *indices.data.add(b) } as usize;
                let val = unsafe { *tensor.data.add(b * in_strides[0] + idx * in_strides[1]) };
                unsafe { *result.data.add(b) = val };
            }
        } else {
            // dim=0
            let cols = input_shape[1] as usize;
            for c in 0..cols {
                let idx = unsafe { *indices.data.add(c) } as usize;
                let val = unsafe { *tensor.data.add(idx * in_strides[0] + c * in_strides[1]) };
                unsafe { *result.data.add(c) = val };
            }
        }
    } else {
        eprintln!("nsl: gather currently only supports 2D tensors");
        std::process::abort();
    }

    if autodiff::is_recording() {
        // Save indices for backward
        NslTensor::from_ptr(indices_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Gather {
            a: tensor_ptr,
            out: result_ptr,
            dim,
            indices_ptr,
            input_shape,
        });
    }
    result_ptr
}

// === Element-wise math ops ===

#[no_mangle]
pub extern "C" fn nsl_tensor_exp(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        unsafe { *data.add(i) = (*a.data.add(i)).exp() };
    }

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Exp {
            a: tensor_ptr,
            out: result,
            saved_out: result,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_log(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        unsafe { *data.add(i) = (*a.data.add(i)).ln() };
    }

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Log {
            a: tensor_ptr,
            out: result,
            saved_a: tensor_ptr,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sqrt(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        unsafe { *data.add(i) = (*a.data.add(i)).sqrt() };
    }

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Sqrt {
            a: tensor_ptr,
            out: result,
            saved_out: result,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_abs(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        unsafe { *data.add(i) = (*a.data.add(i)).abs() };
    }

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Abs {
            a: tensor_ptr,
            out: result,
            saved_a: tensor_ptr,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sign(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        let val = unsafe { *a.data.add(i) };
        unsafe {
            *data.add(i) = if val > 0.0 {
                1.0
            } else if val < 0.0 {
                -1.0
            } else {
                0.0
            }
        };
    }

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    // sign is non-differentiable — no tape recording
    Box::into_raw(result) as i64
}

#[no_mangle]
pub extern "C" fn nsl_tensor_clamp(tensor_ptr: i64, min_val: f64, max_val: f64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        let val = unsafe { *a.data.add(i) };
        unsafe { *data.add(i) = val.clamp(min_val, max_val) };
    }

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Clamp {
            a: tensor_ptr,
            out: result,
            saved_a: tensor_ptr,
            min_val,
            max_val,
        });
    }
    result
}

/// Helper for clamp backward: produces grad * mask where mask is 1 where input is strictly
/// between min_val and max_val, and 0 otherwise.
pub(crate) fn nsl_tensor_clamp_backward(
    grad_ptr: i64,
    input_ptr: i64,
    min_val: f64,
    max_val: f64,
) -> i64 {
    let grad = NslTensor::from_ptr(grad_ptr);
    let input = NslTensor::from_ptr(input_ptr);
    let len = input.len;
    let ndim = input.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        let val = unsafe { *input.data.add(i) };
        let g_val = unsafe { *grad.data.add(i) };
        unsafe {
            *data.add(i) = if val > min_val && val < max_val {
                g_val
            } else {
                0.0
            };
        }
    }

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    Box::into_raw(result) as i64
}

// === Activation functions ===

#[no_mangle]
pub extern "C" fn nsl_tensor_relu(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        let val = unsafe { *a.data.add(i) };
        unsafe { *data.add(i) = if val > 0.0 { val } else { 0.0 } };
    }

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::ReLU {
            a: tensor_ptr,
            out: result,
            saved_a: tensor_ptr,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_gelu(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
    for i in 0..len as usize {
        let x = unsafe { *a.data.add(i) };
        let inner = c * (x + 0.044715 * x * x * x);
        unsafe { *data.add(i) = 0.5 * x * (1.0 + inner.tanh()) };
    }

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::GELU {
            a: tensor_ptr,
            out: result,
            saved_a: tensor_ptr,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_silu(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        let x = unsafe { *a.data.add(i) };
        let sig = 1.0 / (1.0 + (-x).exp());
        unsafe { *data.add(i) = x * sig };
    }

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::SiLU {
            a: tensor_ptr,
            out: result,
            saved_a: tensor_ptr,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_sigmoid(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        let x = unsafe { *a.data.add(i) };
        unsafe { *data.add(i) = 1.0 / (1.0 + (-x).exp()) };
    }

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Sigmoid {
            a: tensor_ptr,
            out: result,
            saved_out: result,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_tanh_act(tensor_ptr: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..len as usize {
        let x = unsafe { *a.data.add(i) };
        unsafe { *data.add(i) = x.tanh() };
    }

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Tanh {
            a: tensor_ptr,
            out: result,
            saved_out: result,
        });
    }
    result
}

#[no_mangle]
pub extern "C" fn nsl_tensor_softmax(tensor_ptr: i64, dim: i64) -> i64 {
    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = checked_alloc_zeroed((len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    // Normalize dim
    let d = if dim < 0 { (ndim + dim) as usize } else { dim as usize };

    let a_shape: Vec<i64> = (0..ndim as usize).map(|i| unsafe { *a.shape.add(i) }).collect();
    let a_strides: Vec<i64> = (0..ndim as usize).map(|i| unsafe { *a.strides.add(i) }).collect();
    let dim_size = a_shape[d] as usize;

    // Iterate over all positions except along dim d
    // Total number of "slices" = total / dim_size
    let num_slices = (len as usize) / dim_size;

    // For each slice, compute the base index (index with dim d = 0)
    // We iterate linearly and reconstruct multi-dim indices
    for slice_idx in 0..num_slices {
        // Convert slice_idx to multi-dim index (skipping dim d)
        let mut remaining = slice_idx;
        let mut base_offset: usize = 0;
        for axis in (0..ndim as usize).rev() {
            if axis == d {
                continue;
            }
            let idx = remaining % (a_shape[axis] as usize);
            remaining /= a_shape[axis] as usize;
            base_offset += idx * (a_strides[axis] as usize);
        }

        // Find max along dim for numerical stability
        let mut max_val = f64::NEG_INFINITY;
        for k in 0..dim_size {
            let offset = base_offset + k * (a_strides[d] as usize);
            let val = unsafe { *a.data.add(offset) };
            if val > max_val {
                max_val = val;
            }
        }

        // Compute exp(x - max) and sum
        let mut sum = 0.0_f64;
        for k in 0..dim_size {
            let offset = base_offset + k * (a_strides[d] as usize);
            let val = unsafe { *a.data.add(offset) };
            let e = (val - max_val).exp();
            unsafe { *data.add(offset) = e };
            sum += e;
        }

        // Normalize
        for k in 0..dim_size {
            let offset = base_offset + k * (a_strides[d] as usize);
            unsafe { *data.add(offset) /= sum };
        }
    }

    let result = Box::new(NslTensor {
        data, shape, strides, ndim, len, refcount: 1,
    });
    let result = Box::into_raw(result) as i64;
    if autodiff::is_recording() {
        NslTensor::from_ptr(result).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Softmax {
            a: tensor_ptr,
            out: result,
            saved_out: result,
            dim,
        });
    }
    result
}

// === Scalar extraction ===

#[no_mangle]
pub extern "C" fn nsl_tensor_item(tensor_ptr: i64) -> f64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    if tensor.len != 1 {
        eprintln!(
            "nsl: .item() requires a scalar tensor (got {} elements)",
            tensor.len
        );
        std::process::abort();
    }
    unsafe { *tensor.data }
}

// === Display ===

#[no_mangle]
pub extern "C" fn nsl_tensor_print(tensor_ptr: i64) {
    let tensor = NslTensor::from_ptr(tensor_ptr);

    if tensor.ndim == 0 {
        if tensor.len > 0 {
            print_float_value(unsafe { *tensor.data });
            println!();
        } else {
            println!("tensor([])");
        }
        return;
    }

    print!("tensor(");
    print_tensor_recursive(tensor.data, tensor.shape, tensor.strides, tensor.ndim, 0);
    println!(")");
}

fn print_float_value(v: f64) {
    if v == v.floor() && v.abs() < 1e15 {
        // Print as integer-looking: "3.0" style for tensor display
        print!("{:.1}", v);
    } else {
        print!("{}", v);
    }
}

fn print_tensor_recursive(
    data: *mut f64,
    shape: *mut i64,
    strides: *mut i64,
    ndim: i64,
    dim: usize,
) {
    let size = unsafe { *shape.add(dim) } as usize;
    let stride = unsafe { *strides.add(dim) } as usize;

    print!("[");
    if dim as i64 == ndim - 1 {
        // Last dimension: print values
        for i in 0..size {
            if i > 0 { print!(", "); }
            let val = unsafe { *data.add(i * stride) };
            print_float_value(val);
        }
    } else {
        // Recursive: print sub-arrays
        for i in 0..size {
            if i > 0 { print!(", "); }
            let offset_data = unsafe { data.add(i * stride) };
            print_tensor_recursive(offset_data, shape, strides, ndim, dim + 1);
        }
    }
    print!("]");
}

// === Memory ===

#[no_mangle]
pub extern "C" fn nsl_tensor_clone(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim;
    let len = tensor.len;

    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(tensor.shape, shape, ndim as usize) };

    let strides = NslTensor::compute_strides(shape, ndim);

    let data_size = (len as usize) * std::mem::size_of::<f64>();
    let data = checked_alloc(data_size) as *mut f64;
    unsafe { std::ptr::copy_nonoverlapping(tensor.data, data, len as usize) };

    let result = Box::new(NslTensor {
        data,
        shape,
        strides,
        ndim,
        len,
        refcount: 1,
    });
    Box::into_raw(result) as i64
}

#[no_mangle]
pub extern "C" fn nsl_tensor_free(tensor_ptr: i64) {
    if tensor_ptr == 0 {
        return;
    }
    let tensor = NslTensor::from_ptr(tensor_ptr);
    tensor.refcount -= 1;
    if tensor.refcount <= 0 {
        let data_size = (tensor.len as usize) * std::mem::size_of::<f64>();
        let shape_size = (tensor.ndim as usize) * std::mem::size_of::<i64>();
        let strides_size = shape_size;

        unsafe {
            checked_free(tensor.data as *mut u8, data_size);
            if !tensor.shape.is_null() {
                checked_free(tensor.shape as *mut u8, shape_size);
            }
            if !tensor.strides.is_null() {
                checked_free(tensor.strides as *mut u8, strides_size);
            }
            // Free the NslTensor struct itself
            drop(Box::from_raw(tensor as *mut NslTensor));
        }
    }
}

// === In-place mutation ops (NOT taped — used outside grad blocks) ===

#[no_mangle]
pub extern "C" fn nsl_tensor_copy_data(dst_ptr: i64, src_ptr: i64) {
    let dst = NslTensor::from_ptr(dst_ptr);
    let src = NslTensor::from_ptr(src_ptr);
    assert_eq!(
        dst.len, src.len,
        "nsl_tensor_copy_data: dst len {} != src len {}",
        dst.len, src.len
    );
    unsafe {
        std::ptr::copy_nonoverlapping(src.data, dst.data, dst.len as usize);
    }
}

#[no_mangle]
pub extern "C" fn nsl_tensor_add_inplace(dst_ptr: i64, src_ptr: i64) {
    let dst = NslTensor::from_ptr(dst_ptr);
    let src = NslTensor::from_ptr(src_ptr);
    assert_eq!(
        dst.len, src.len,
        "nsl_tensor_add_inplace: dst len {} != src len {}",
        dst.len, src.len
    );
    for i in 0..dst.len as usize {
        unsafe {
            *dst.data.add(i) += *src.data.add(i);
        }
    }
}

#[no_mangle]
pub extern "C" fn nsl_tensor_zero_inplace(tensor_ptr: i64) {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    unsafe {
        std::ptr::write_bytes(tensor.data, 0, tensor.len as usize);
    }
}

#[no_mangle]
pub extern "C" fn nsl_tensor_zeros_like(tensor_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let shape_list = nsl_list_new();
    for i in 0..tensor.ndim as usize {
        unsafe {
            nsl_list_push(shape_list, *tensor.shape.add(i));
        }
    }
    let result = nsl_tensor_zeros(shape_list);
    crate::list::nsl_list_free(shape_list);
    result
}

// === Gradient clipping ===

#[no_mangle]
pub extern "C" fn nsl_clip_grad_norm(grad_list_ptr: i64, max_norm: f64) {
    let list = NslList::from_ptr(grad_list_ptr);
    let num_grads = list.len as usize;

    // Compute global L2 norm: sqrt(sum of squares of all elements across all tensors)
    let mut sum_sq: f64 = 0.0;
    for g in 0..num_grads {
        let tensor_ptr = unsafe { *list.data.add(g) };
        let tensor = NslTensor::from_ptr(tensor_ptr);
        for i in 0..tensor.len as usize {
            let val = unsafe { *tensor.data.add(i) };
            sum_sq += val * val;
        }
    }
    let norm = sum_sq.sqrt();

    // Fast path: skip scaling when gradients are within bounds
    if norm <= max_norm {
        return;
    }

    // norm > max_norm: scale all gradients by (max_norm / (norm + 1e-6))
    {
        let scale = max_norm / (norm + 1e-6);
        for g in 0..num_grads {
            let tensor_ptr = unsafe { *list.data.add(g) };
            let tensor = NslTensor::from_ptr(tensor_ptr);
            for i in 0..tensor.len as usize {
                unsafe {
                    *tensor.data.add(i) *= scale;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Tensor Slice
// ---------------------------------------------------------------------------

/// Slice a tensor along a dimension: extract elements [start, end) along dim.
/// Supports negative indices (e.g., -1 means last element).
/// Input shape [d0, d1, ..., d_dim, ...] -> output shape [d0, d1, ..., (end-start), ...]
#[no_mangle]
pub extern "C" fn nsl_tensor_slice(tensor_ptr: i64, dim: i64, start: i64, end: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let ndim = tensor.ndim as usize;

    // Normalize dim
    let d = if dim < 0 { (tensor.ndim + dim) as usize } else { dim as usize };
    assert!(d < ndim, "nsl_tensor_slice: dim {dim} out of range for ndim {ndim}");

    let dim_size = unsafe { *tensor.shape.add(d) };

    // Normalize start/end with negative index support
    let s = if start < 0 { (dim_size + start).max(0) } else { start.min(dim_size) };
    let e = if end < 0 { (dim_size + end).max(0) } else { end.min(dim_size) };
    let slice_len = (e - s).max(0);

    // Build output shape
    let out_shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim {
        if i == d {
            unsafe { *out_shape.add(i) = slice_len };
        } else {
            unsafe { *out_shape.add(i) = *tensor.shape.add(i) };
        }
    }
    let out_strides = NslTensor::compute_strides(out_shape, tensor.ndim);
    let out_len = NslTensor::total_elements(out_shape, tensor.ndim);

    let data = checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    // Copy data: iterate over all output elements, mapping back to input
    let in_strides: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *tensor.strides.add(i) })
        .collect();
    let o_strides: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *out_strides.add(i) })
        .collect();

    for flat in 0..out_len as usize {
        // Convert flat index to multi-dim in output space
        let mut remaining = flat;
        let mut in_offset: usize = 0;
        for axis in 0..ndim {
            let idx = remaining / o_strides[axis] as usize;
            remaining %= o_strides[axis] as usize;
            if axis == d {
                in_offset += (idx + s as usize) * in_strides[axis] as usize;
            } else {
                in_offset += idx * in_strides[axis] as usize;
            }
        }
        unsafe { *data.add(flat) = *tensor.data.add(in_offset) };
    }

    // Save input shape for backward
    let input_shape: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *tensor.shape.add(i) })
        .collect();

    let out = Box::new(NslTensor {
        data,
        shape: out_shape,
        strides: out_strides,
        ndim: tensor.ndim,
        len: out_len,
        refcount: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    // Record on tape for autodiff
    autodiff::maybe_record(autodiff::TapeOp::Slice {
        a: tensor_ptr,
        out: out_ptr,
        dim: dim,
        start: s,
        input_shape,
    });

    out_ptr
}

// ---------------------------------------------------------------------------
// Tensor Cat
// ---------------------------------------------------------------------------

/// Concatenate a list of tensors along a dimension.
/// All tensors must have same shape except on the cat dimension.
/// Returns a new tensor with the cat dimension being the sum of all input dim sizes.
#[no_mangle]
pub extern "C" fn nsl_tensor_cat(tensor_list: i64, dim: i64) -> i64 {
    let list = NslList::from_ptr(tensor_list);
    let num_tensors = list.len as usize;
    assert!(num_tensors > 0, "nsl_tensor_cat: empty tensor list");

    let first = NslTensor::from_ptr(unsafe { *list.data.add(0) });
    let ndim = first.ndim as usize;
    let d = if dim < 0 { (first.ndim + dim) as usize } else { dim as usize };
    assert!(d < ndim, "nsl_tensor_cat: dim {dim} out of range for ndim {ndim}");

    // Collect split sizes and validate shapes
    let mut split_sizes: Vec<i64> = Vec::with_capacity(num_tensors);
    let mut total_cat_dim: i64 = 0;

    for t_idx in 0..num_tensors {
        let t = NslTensor::from_ptr(unsafe { *list.data.add(t_idx) });
        assert_eq!(t.ndim as usize, ndim, "nsl_tensor_cat: ndim mismatch");
        let cat_size = unsafe { *t.shape.add(d) };
        split_sizes.push(cat_size);
        total_cat_dim += cat_size;
        // Validate non-cat dimensions match
        for axis in 0..ndim {
            if axis != d {
                let s1 = unsafe { *first.shape.add(axis) };
                let s2 = unsafe { *t.shape.add(axis) };
                assert_eq!(s1, s2, "nsl_tensor_cat: shape mismatch at dim {axis}: {s1} vs {s2}");
            }
        }
    }

    // Build output shape
    let out_shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    for i in 0..ndim {
        if i == d {
            unsafe { *out_shape.add(i) = total_cat_dim };
        } else {
            unsafe { *out_shape.add(i) = *first.shape.add(i) };
        }
    }
    let out_ndim = first.ndim;
    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);
    let out_len = NslTensor::total_elements(out_shape, out_ndim);
    let data = checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    let o_strides: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *out_strides.add(i) })
        .collect();

    // Copy data from each tensor
    let mut cat_offset: usize = 0;
    for t_idx in 0..num_tensors {
        let t = NslTensor::from_ptr(unsafe { *list.data.add(t_idx) });
        let t_strides: Vec<i64> = (0..ndim)
            .map(|i| unsafe { *t.strides.add(i) })
            .collect();
        let _t_shape: Vec<i64> = (0..ndim)
            .map(|i| unsafe { *t.shape.add(i) })
            .collect();

        let t_len = t.len as usize;
        for flat in 0..t_len {
            // Convert flat index to multi-dim in tensor space
            let mut remaining = flat;
            let mut out_offset: usize = 0;
            for axis in 0..ndim {
                let idx = remaining / t_strides[axis] as usize;
                remaining %= t_strides[axis] as usize;
                if axis == d {
                    out_offset += (idx + cat_offset) * o_strides[axis] as usize;
                } else {
                    out_offset += idx * o_strides[axis] as usize;
                }
            }
            unsafe { *data.add(out_offset) = *t.data.add(flat) };
        }
        cat_offset += split_sizes[t_idx] as usize;
    }

    let out = Box::new(NslTensor {
        data,
        shape: out_shape,
        strides: out_strides,
        ndim: out_ndim,
        len: out_len,
        refcount: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    // Collect input ptrs for tape
    let input_ptrs: Vec<i64> = (0..num_tensors)
        .map(|i| unsafe { *list.data.add(i) })
        .collect();

    autodiff::maybe_record(autodiff::TapeOp::Cat {
        inputs: input_ptrs,
        out: out_ptr,
        dim: dim,
        split_sizes,
    });

    out_ptr
}

// === Embedding lookup ===

/// Look up rows from an embedding weight matrix by integer indices.
/// weight: [vocab_size, embed_dim], indices: [seq_len] (f64 values cast to integer indices)
/// output: [seq_len, embed_dim]
#[no_mangle]
pub extern "C" fn nsl_tensor_embedding_lookup(weight_ptr: i64, indices_ptr: i64) -> i64 {
    let weight = NslTensor::from_ptr(weight_ptr);
    let indices = NslTensor::from_ptr(indices_ptr);

    if weight.ndim != 2 {
        eprintln!(
            "nsl: embedding_lookup requires 2D weight tensor (got {}D)",
            weight.ndim
        );
        std::process::abort();
    }
    if indices.ndim != 1 {
        eprintln!(
            "nsl: embedding_lookup requires 1D indices tensor (got {}D)",
            indices.ndim
        );
        std::process::abort();
    }

    let vocab_size = unsafe { *weight.shape.add(0) } as usize;
    let embed_dim = unsafe { *weight.shape.add(1) } as usize;
    let seq_len = unsafe { *indices.shape.add(0) } as usize;

    // Output shape: [seq_len, embed_dim]
    let out_ndim: i64 = 2;
    let out_len = (seq_len * embed_dim) as i64;
    let out_shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = seq_len as i64;
        *out_shape.add(1) = embed_dim as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);
    let out_data =
        checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    // For each index, copy the corresponding row from weight
    for i in 0..seq_len {
        let idx = unsafe { *indices.data.add(i) } as usize;
        if idx >= vocab_size {
            eprintln!(
                "nsl: embedding_lookup index {} out of bounds for vocab_size {}",
                idx, vocab_size
            );
            std::process::abort();
        }
        unsafe {
            std::ptr::copy_nonoverlapping(
                weight.data.add(idx * embed_dim),
                out_data.add(i * embed_dim),
                embed_dim,
            );
        }
    }

    let out = Box::new(NslTensor {
        data: out_data,
        shape: out_shape,
        strides: out_strides,
        ndim: out_ndim,
        len: out_len,
        refcount: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(weight_ptr).refcount += 1;
        NslTensor::from_ptr(indices_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::EmbeddingLookup {
            weight: weight_ptr,
            indices: indices_ptr,
            out: out_ptr,
            saved_weight: weight_ptr,
            saved_indices: indices_ptr,
        });
    }

    out_ptr
}

// === LayerNorm & RMSNorm (fused runtime primitives) ===

/// LayerNorm: normalize over last dimension, then scale and shift.
/// input: [*, N], weight: [N], bias: [N]
/// For each "row" (slice along last dim):
///   mean = mean(row)
///   var = mean((row - mean)^2)
///   normalized = (row - mean) / sqrt(var + eps)
///   output = normalized * weight + bias
/// Saves {input, mean, inv_std, weight} for backward.
#[no_mangle]
pub extern "C" fn nsl_tensor_layernorm(
    input_ptr: i64,
    weight_ptr: i64,
    bias_ptr: i64,
    eps: f64,
) -> i64 {
    let input = NslTensor::from_ptr(input_ptr);
    let weight = NslTensor::from_ptr(weight_ptr);
    let bias = NslTensor::from_ptr(bias_ptr);

    let total = input.len as usize;
    let ndim = input.ndim as usize;
    let n = unsafe { *input.shape.add(ndim - 1) } as usize; // last dim size
    let num_rows = total / n;

    // Allocate output with same shape as input
    let out_shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, out_shape, ndim) };
    let out_strides = NslTensor::compute_strides(out_shape, ndim as i64);
    let out_data = checked_alloc(total * std::mem::size_of::<f64>()) as *mut f64;

    // Saved tensors for backward: mean [num_rows] and inv_std [num_rows]
    let mean_shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *mean_shape = num_rows as i64 };
    let mean_strides = NslTensor::compute_strides(mean_shape, 1);
    let mean_data = checked_alloc(num_rows * std::mem::size_of::<f64>()) as *mut f64;

    let inv_std_shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *inv_std_shape = num_rows as i64 };
    let inv_std_strides = NslTensor::compute_strides(inv_std_shape, 1);
    let inv_std_data = checked_alloc(num_rows * std::mem::size_of::<f64>()) as *mut f64;

    for row in 0..num_rows {
        let base = row * n;

        // Compute mean
        let mut sum = 0.0_f64;
        for j in 0..n {
            sum += unsafe { *input.data.add(base + j) };
        }
        let mean_val = sum / n as f64;

        // Compute variance
        let mut var = 0.0_f64;
        for j in 0..n {
            let diff = unsafe { *input.data.add(base + j) } - mean_val;
            var += diff * diff;
        }
        var /= n as f64;

        let inv_std_val = 1.0 / (var + eps).sqrt();

        unsafe {
            *mean_data.add(row) = mean_val;
            *inv_std_data.add(row) = inv_std_val;
        }

        // normalized * weight + bias
        for j in 0..n {
            let x = unsafe { *input.data.add(base + j) };
            let normalized = (x - mean_val) * inv_std_val;
            let w = unsafe { *weight.data.add(j) };
            let b = unsafe { *bias.data.add(j) };
            unsafe { *out_data.add(base + j) = normalized * w + b };
        }
    }

    let out = Box::new(NslTensor {
        data: out_data,
        shape: out_shape,
        strides: out_strides,
        ndim: ndim as i64,
        len: total as i64,
        refcount: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    let mean_tensor = Box::new(NslTensor {
        data: mean_data,
        shape: mean_shape,
        strides: mean_strides,
        ndim: 1,
        len: num_rows as i64,
        refcount: 1,
    });
    let mean_ptr = Box::into_raw(mean_tensor) as i64;

    let inv_std_tensor = Box::new(NslTensor {
        data: inv_std_data,
        shape: inv_std_shape,
        strides: inv_std_strides,
        ndim: 1,
        len: num_rows as i64,
        refcount: 1,
    });
    let inv_std_ptr = Box::into_raw(inv_std_tensor) as i64;

    if autodiff::is_recording() {
        // Bump refcounts for saved tensors
        NslTensor::from_ptr(input_ptr).refcount += 1;
        NslTensor::from_ptr(weight_ptr).refcount += 1;
        // mean and inv_std are freshly created with refcount=1; bump for tape
        NslTensor::from_ptr(mean_ptr).refcount += 1;
        NslTensor::from_ptr(inv_std_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::LayerNorm {
            input: input_ptr,
            weight: weight_ptr,
            bias: bias_ptr,
            out: out_ptr,
            saved_input: input_ptr,
            saved_mean: mean_ptr,
            saved_inv_std: inv_std_ptr,
            saved_weight: weight_ptr,
        });
    } else {
        // Not recording — free the saved tensors since they won't be used
        nsl_tensor_free(mean_ptr);
        nsl_tensor_free(inv_std_ptr);
    }

    out_ptr
}

/// RMSNorm: normalize by root-mean-square, scale by weight (no bias, no mean subtraction).
/// rms = sqrt(mean(x^2) + eps)
/// output = x / rms * weight
/// Saves {input, rms, weight} for backward.
#[no_mangle]
pub extern "C" fn nsl_tensor_rmsnorm(input_ptr: i64, weight_ptr: i64, eps: f64) -> i64 {
    let input = NslTensor::from_ptr(input_ptr);
    let weight = NslTensor::from_ptr(weight_ptr);

    let total = input.len as usize;
    let ndim = input.ndim as usize;
    let n = unsafe { *input.shape.add(ndim - 1) } as usize;
    let num_rows = total / n;

    // Allocate output with same shape as input
    let out_shape = checked_alloc(ndim * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, out_shape, ndim) };
    let out_strides = NslTensor::compute_strides(out_shape, ndim as i64);
    let out_data = checked_alloc(total * std::mem::size_of::<f64>()) as *mut f64;

    // Saved: rms per row [num_rows]
    let rms_shape = checked_alloc(std::mem::size_of::<i64>()) as *mut i64;
    unsafe { *rms_shape = num_rows as i64 };
    let rms_strides = NslTensor::compute_strides(rms_shape, 1);
    let rms_data = checked_alloc(num_rows * std::mem::size_of::<f64>()) as *mut f64;

    for row in 0..num_rows {
        let base = row * n;

        // Compute mean(x^2)
        let mut sum_sq = 0.0_f64;
        for j in 0..n {
            let x = unsafe { *input.data.add(base + j) };
            sum_sq += x * x;
        }
        let rms_val = (sum_sq / n as f64 + eps).sqrt();

        unsafe { *rms_data.add(row) = rms_val };

        // output = x / rms * weight
        for j in 0..n {
            let x = unsafe { *input.data.add(base + j) };
            let w = unsafe { *weight.data.add(j) };
            unsafe { *out_data.add(base + j) = x / rms_val * w };
        }
    }

    let out = Box::new(NslTensor {
        data: out_data,
        shape: out_shape,
        strides: out_strides,
        ndim: ndim as i64,
        len: total as i64,
        refcount: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    let rms_tensor = Box::new(NslTensor {
        data: rms_data,
        shape: rms_shape,
        strides: rms_strides,
        ndim: 1,
        len: num_rows as i64,
        refcount: 1,
    });
    let rms_ptr = Box::into_raw(rms_tensor) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(input_ptr).refcount += 1;
        NslTensor::from_ptr(weight_ptr).refcount += 1;
        NslTensor::from_ptr(rms_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::RMSNorm {
            input: input_ptr,
            weight: weight_ptr,
            out: out_ptr,
            saved_input: input_ptr,
            saved_rms: rms_ptr,
            saved_weight: weight_ptr,
        });
    } else {
        nsl_tensor_free(rms_ptr);
    }

    out_ptr
}

// === Dropout ===

/// Dropout with inverted scaling. During training (training != 0, p > 0), randomly zero elements
/// with probability p and scale surviving elements by 1/(1-p). During eval, returns a clone.
#[no_mangle]
pub extern "C" fn nsl_tensor_dropout(tensor_ptr: i64, p: f64, training: i8) -> i64 {
    // Eval mode or p==0: identity
    if training == 0 || p == 0.0 {
        return nsl_tensor_clone(tensor_ptr);
    }

    let a = NslTensor::from_ptr(tensor_ptr);
    let len = a.len as usize;
    let ndim = a.ndim;
    let shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let out_data = checked_alloc(len * std::mem::size_of::<f64>()) as *mut f64;

    // Generate dropout mask: 1.0 if kept, 0.0 if dropped
    let mask_shape = checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(a.shape, mask_shape, ndim as usize) };
    let mask_strides = NslTensor::compute_strides(mask_shape, ndim);
    let mask_data = checked_alloc(len * std::mem::size_of::<f64>()) as *mut f64;

    let scale = 1.0 / (1.0 - p);

    // Simple LCG random number generator
    let mut seed: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(12345);

    for i in 0..len {
        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let rand_val = (seed >> 11) as f64 / (1u64 << 53) as f64;
        let keep = if rand_val >= p { 1.0 } else { 0.0 };
        unsafe {
            *mask_data.add(i) = keep;
            *out_data.add(i) = *a.data.add(i) * keep * scale;
        }
    }

    let result = Box::new(NslTensor {
        data: out_data, shape, strides, ndim, len: len as i64, refcount: 1,
    });
    let result_ptr = Box::into_raw(result) as i64;

    let mask_tensor = Box::new(NslTensor {
        data: mask_data, shape: mask_shape, strides: mask_strides, ndim, len: len as i64, refcount: 1,
    });
    let mask_ptr = Box::into_raw(mask_tensor) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        NslTensor::from_ptr(mask_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Dropout {
            a: tensor_ptr,
            out: result_ptr,
            saved_mask: mask_ptr,
            scale,
        });
    } else {
        nsl_tensor_free(mask_ptr);
    }

    result_ptr
}

// === Conv2d ===

/// 2D convolution: input [N,C_in,H,W], weight [C_out,C_in,kH,kW], bias [C_out] (0 for no bias).
/// Returns output [N,C_out,H_out,W_out].
#[no_mangle]
pub extern "C" fn nsl_tensor_conv2d(
    input_ptr: i64,
    weight_ptr: i64,
    bias_ptr: i64,
    stride_h: i64,
    stride_w: i64,
    pad_h: i64,
    pad_w: i64,
) -> i64 {
    let input = NslTensor::from_ptr(input_ptr);
    let weight = NslTensor::from_ptr(weight_ptr);

    let n = unsafe { *input.shape.add(0) } as usize;
    let c_in = unsafe { *input.shape.add(1) } as usize;
    let h = unsafe { *input.shape.add(2) } as usize;
    let w = unsafe { *input.shape.add(3) } as usize;

    let c_out = unsafe { *weight.shape.add(0) } as usize;
    let kh = unsafe { *weight.shape.add(2) } as usize;
    let kw = unsafe { *weight.shape.add(3) } as usize;

    let sh = stride_h as usize;
    let sw = stride_w as usize;
    let ph = pad_h as usize;
    let pw = pad_w as usize;

    let h_out = (h + 2 * ph - kh) / sh + 1;
    let w_out = (w + 2 * pw - kw) / sw + 1;

    let out_len = n * c_out * h_out * w_out;
    let out_shape = checked_alloc(4 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = n as i64;
        *out_shape.add(1) = c_out as i64;
        *out_shape.add(2) = h_out as i64;
        *out_shape.add(3) = w_out as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 4);
    let out_data = checked_alloc_zeroed(out_len * std::mem::size_of::<f64>()) as *mut f64;

    // Direct nested-loop convolution
    for ni in 0..n {
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut val = 0.0_f64;
                    for ci in 0..c_in {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let ih = oh * sh + ky;
                                let iw = ow * sw + kx;
                                // Check padding bounds
                                if ih >= ph && iw >= pw && ih - ph < h && iw - pw < w {
                                    let input_idx = ni * (c_in * h * w) + ci * (h * w) + (ih - ph) * w + (iw - pw);
                                    let weight_idx = co * (c_in * kh * kw) + ci * (kh * kw) + ky * kw + kx;
                                    val += unsafe { *input.data.add(input_idx) * *weight.data.add(weight_idx) };
                                }
                            }
                        }
                    }
                    // Add bias if provided
                    if bias_ptr != 0 {
                        let bias = NslTensor::from_ptr(bias_ptr);
                        val += unsafe { *bias.data.add(co) };
                    }
                    let out_idx = ni * (c_out * h_out * w_out) + co * (h_out * w_out) + oh * w_out + ow;
                    unsafe { *out_data.add(out_idx) = val };
                }
            }
        }
    }

    let result = Box::new(NslTensor {
        data: out_data, shape: out_shape, strides: out_strides, ndim: 4, len: out_len as i64, refcount: 1,
    });
    let result_ptr = Box::into_raw(result) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(input_ptr).refcount += 1;
        NslTensor::from_ptr(weight_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::Conv2d {
            input: input_ptr,
            weight: weight_ptr,
            bias: bias_ptr,
            out: result_ptr,
            saved_input: input_ptr,
            saved_weight: weight_ptr,
            stride_h: sh,
            stride_w: sw,
            pad_h: ph,
            pad_w: pw,
        });
    }

    result_ptr
}

// === MaxPool2d ===

/// 2D max pooling: input [N,C,H,W] -> output [N,C,H_out,W_out].
/// Saves argmax indices for backward gradient routing.
#[no_mangle]
pub extern "C" fn nsl_tensor_maxpool2d(
    input_ptr: i64,
    kernel_h: i64,
    kernel_w: i64,
    stride: i64,
    padding: i64,
) -> i64 {
    let input = NslTensor::from_ptr(input_ptr);

    let n = unsafe { *input.shape.add(0) } as usize;
    let c = unsafe { *input.shape.add(1) } as usize;
    let h = unsafe { *input.shape.add(2) } as usize;
    let w = unsafe { *input.shape.add(3) } as usize;

    let kh = kernel_h as usize;
    let kw = kernel_w as usize;
    let s = stride as usize;
    let pad = padding as usize;

    let h_out = (h + 2 * pad - kh) / s + 1;
    let w_out = (w + 2 * pad - kw) / s + 1;

    let out_len = n * c * h_out * w_out;
    let out_shape = checked_alloc(4 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = n as i64;
        *out_shape.add(1) = c as i64;
        *out_shape.add(2) = h_out as i64;
        *out_shape.add(3) = w_out as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, 4);
    let out_data = checked_alloc(out_len * std::mem::size_of::<f64>()) as *mut f64;

    // Save argmax indices for backward
    let mut argmax_indices: Vec<usize> = vec![0; out_len];

    for ni in 0..n {
        for ci in 0..c {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let mut max_val = f64::NEG_INFINITY;
                    let mut max_idx: usize = 0;
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let ih = oh * s + ky;
                            let iw = ow * s + kx;
                            if ih >= pad && iw >= pad && ih - pad < h && iw - pad < w {
                                let input_idx = ni * (c * h * w) + ci * (h * w) + (ih - pad) * w + (iw - pad);
                                let val = unsafe { *input.data.add(input_idx) };
                                if val > max_val {
                                    max_val = val;
                                    max_idx = input_idx;
                                }
                            }
                        }
                    }
                    let out_idx = ni * (c * h_out * w_out) + ci * (h_out * w_out) + oh * w_out + ow;
                    unsafe { *out_data.add(out_idx) = max_val };
                    argmax_indices[out_idx] = max_idx;
                }
            }
        }
    }

    let result = Box::new(NslTensor {
        data: out_data, shape: out_shape, strides: out_strides, ndim: 4, len: out_len as i64, refcount: 1,
    });
    let result_ptr = Box::into_raw(result) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(input_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::MaxPool2d {
            a: input_ptr,
            out: result_ptr,
            saved_argmax: argmax_indices,
            input_shape: vec![n as i64, c as i64, h as i64, w as i64],
        });
    }

    result_ptr
}

/// Add 1D bias [N] to 2D tensor [M, N] — broadcasts bias along rows.
/// output[i, j] = tensor[i, j] + bias[j]
#[no_mangle]
pub extern "C" fn nsl_tensor_bias_add(tensor_ptr: i64, bias_ptr: i64) -> i64 {
    let tensor = NslTensor::from_ptr(tensor_ptr);
    let bias = NslTensor::from_ptr(bias_ptr);

    if tensor.ndim != 2 {
        eprintln!(
            "nsl: bias_add requires 2D tensor (got {}D)",
            tensor.ndim
        );
        std::process::abort();
    }
    if bias.ndim != 1 {
        eprintln!(
            "nsl: bias_add requires 1D bias (got {}D)",
            bias.ndim
        );
        std::process::abort();
    }

    let rows = unsafe { *tensor.shape.add(0) } as usize;
    let cols = unsafe { *tensor.shape.add(1) } as usize;
    let bias_len = unsafe { *bias.shape.add(0) } as usize;

    if cols != bias_len {
        eprintln!(
            "nsl: bias_add shape mismatch — tensor has {} cols but bias has {} elements",
            cols, bias_len
        );
        std::process::abort();
    }

    // Output shape: same as tensor [rows, cols]
    let out_ndim: i64 = 2;
    let out_len = (rows * cols) as i64;
    let out_shape = checked_alloc(2 * std::mem::size_of::<i64>()) as *mut i64;
    unsafe {
        *out_shape.add(0) = rows as i64;
        *out_shape.add(1) = cols as i64;
    }
    let out_strides = NslTensor::compute_strides(out_shape, out_ndim);
    let out_data = checked_alloc((out_len as usize) * std::mem::size_of::<f64>()) as *mut f64;

    for i in 0..rows {
        for j in 0..cols {
            unsafe {
                *out_data.add(i * cols + j) = *tensor.data.add(i * cols + j) + *bias.data.add(j);
            }
        }
    }

    let out = Box::new(NslTensor {
        data: out_data,
        shape: out_shape,
        strides: out_strides,
        ndim: out_ndim,
        len: out_len,
        refcount: 1,
    });
    let out_ptr = Box::into_raw(out) as i64;

    if autodiff::is_recording() {
        NslTensor::from_ptr(tensor_ptr).refcount += 1;
        NslTensor::from_ptr(bias_ptr).refcount += 1;
        autodiff::maybe_record(autodiff::TapeOp::BiasAdd {
            tensor: tensor_ptr,
            bias: bias_ptr,
            out: out_ptr,
        });
    }

    out_ptr
}
