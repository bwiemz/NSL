use std::cell::RefCell;
use std::ffi::c_void;
use std::collections::{HashMap, HashSet};

use crate::list::NslList;
use crate::memory::checked_alloc;
use crate::tensor::{
    nsl_tensor_add as tensor_add,
    nsl_tensor_clone as tensor_clone,
    nsl_tensor_div as tensor_div,
    nsl_tensor_free as tensor_free,
    nsl_tensor_item as tensor_item,
    nsl_tensor_matmul as tensor_matmul,
    nsl_tensor_mul as tensor_mul,
    nsl_tensor_mul_scalar as tensor_mul_scalar,
    nsl_tensor_neg as tensor_neg,
    nsl_tensor_ones as tensor_ones,
    nsl_tensor_shape as tensor_shape,
    nsl_tensor_sign as tensor_sign,
    nsl_tensor_transpose as tensor_transpose,
    nsl_tensor_zeros as tensor_zeros,
};

/// Operations recorded on the tape during forward passes inside `grad` blocks.
#[allow(dead_code)]
#[derive(Clone)]
pub enum TapeOp {
    Add { a: i64, b: i64, out: i64, a_shape: Vec<i64>, b_shape: Vec<i64> },
    Sub { a: i64, b: i64, out: i64, a_shape: Vec<i64>, b_shape: Vec<i64> },
    Mul { a: i64, b: i64, out: i64, saved_a: i64, saved_b: i64, a_shape: Vec<i64>, b_shape: Vec<i64> },
    Div { a: i64, b: i64, out: i64, saved_a: i64, saved_b: i64, a_shape: Vec<i64>, b_shape: Vec<i64> },
    MatMul { a: i64, b: i64, out: i64, saved_a: i64, saved_b: i64 },
    Neg { a: i64, out: i64 },
    MulScalar { a: i64, scalar: f64, out: i64 },
    AddScalar { a: i64, out: i64 },
    Transpose { a: i64, out: i64, dim0: i64, dim1: i64 },
    SumReduce { a: i64, out: i64, dim: i64, keepdim: bool, input_shape: Vec<i64> },
    MeanReduce { a: i64, out: i64, dim: i64, keepdim: bool, num_elements: i64, input_shape: Vec<i64> },
    ReduceMax { a: i64, out: i64, dim: i64, keepdim: bool, saved_argmax: Vec<usize>, input_shape: Vec<i64> },
    Gather { a: i64, out: i64, dim: i64, indices_ptr: i64, input_shape: Vec<i64> },
    Exp { a: i64, out: i64, saved_out: i64 },
    Log { a: i64, out: i64, saved_a: i64 },
    Sqrt { a: i64, out: i64, saved_out: i64 },
    Abs { a: i64, out: i64, saved_a: i64 },
    Clamp { a: i64, out: i64, saved_a: i64, min_val: f64, max_val: f64 },
    ReLU { a: i64, out: i64, saved_a: i64 },
    GELU { a: i64, out: i64, saved_a: i64 },
    SiLU { a: i64, out: i64, saved_a: i64 },
    Sigmoid { a: i64, out: i64, saved_out: i64 },
    Tanh { a: i64, out: i64, saved_out: i64 },
    Softmax { a: i64, out: i64, saved_out: i64, dim: i64 },
    Slice { a: i64, out: i64, dim: i64, start: i64, input_shape: Vec<i64> },
    Cat { inputs: Vec<i64>, out: i64, dim: i64, split_sizes: Vec<i64> },
    EmbeddingLookup { weight: i64, indices: i64, out: i64, saved_weight: i64, saved_indices: i64 },
    LayerNorm { input: i64, weight: i64, bias: i64, out: i64, saved_input: i64, saved_mean: i64, saved_inv_std: i64, saved_weight: i64 },
    RMSNorm { input: i64, weight: i64, out: i64, saved_input: i64, saved_rms: i64, saved_weight: i64 },
    Dropout { a: i64, out: i64, saved_mask: i64, scale: f64 },
    Conv2d { input: i64, weight: i64, bias: i64, out: i64, saved_input: i64, saved_weight: i64, stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize },
    MaxPool2d { a: i64, out: i64, saved_argmax: Vec<usize>, input_shape: Vec<i64> },
    BiasAdd { tensor: i64, bias: i64, out: i64 },
}

struct Tape {
    ops: Vec<TapeOp>,
    param_set: HashSet<i64>,
    recording: bool,
    pause_depth: i32,
}

impl Tape {
    fn new() -> Self {
        Tape {
            ops: Vec::new(),
            param_set: HashSet::new(),
            recording: false,
            pause_depth: 0,
        }
    }
}

thread_local! {
    static TAPE: RefCell<Tape> = RefCell::new(Tape::new());
}

/// Returns true if the tape is actively recording (started and not paused).
pub fn is_recording() -> bool {
    TAPE.with(|t| {
        let tape = t.borrow();
        tape.recording && tape.pause_depth == 0
    })
}

/// Remove the last recorded op from the tape (used when a compound op replaces a sub-op).
pub fn pop_last_op() {
    TAPE.with(|t| {
        t.borrow_mut().ops.pop();
    });
}

/// Records an operation on the tape if recording is active.
pub fn maybe_record(op: TapeOp) {
    if !is_recording() {
        return;
    }
    TAPE.with(|t| {
        t.borrow_mut().ops.push(op);
    });
}

/// Start recording operations on the tape.
/// `param_list` is an i64 pointer to an NslList of tensor pointers that are parameters.
#[no_mangle]
pub extern "C" fn nsl_tape_start(param_list: i64) {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        tape.ops.clear();
        tape.param_set.clear();
        tape.recording = true;
        tape.pause_depth = 0;

        let list = NslList::from_ptr(param_list);
        for i in 0..list.len as usize {
            let tensor_ptr = unsafe { *list.data.add(i) };
            tape.param_set.insert(tensor_ptr);
        }
    });
}

/// Stop recording and clean up saved tensor refcounts.
/// This prevents memory leaks even if backward is never called.
#[no_mangle]
pub extern "C" fn nsl_tape_stop() {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        tape.recording = false;
        tape.pause_depth = 0;

        // Release saved tensor refs (ops that hold extra refs).
        // Use tensor_free so tensors are actually deallocated if refcount hits 0.
        for op in tape.ops.iter() {
            match op {
                TapeOp::Mul { saved_a, saved_b, .. }
                | TapeOp::Div { saved_a, saved_b, .. }
                | TapeOp::MatMul { saved_a, saved_b, .. } => {
                    tensor_free(*saved_a);
                    tensor_free(*saved_b);
                }
                TapeOp::Exp { saved_out, .. }
                | TapeOp::Sqrt { saved_out, .. }
                | TapeOp::Sigmoid { saved_out, .. }
                | TapeOp::Tanh { saved_out, .. }
                | TapeOp::Softmax { saved_out, .. } => {
                    tensor_free(*saved_out);
                }
                TapeOp::Log { saved_a, .. }
                | TapeOp::Abs { saved_a, .. }
                | TapeOp::Clamp { saved_a, .. }
                | TapeOp::ReLU { saved_a, .. }
                | TapeOp::GELU { saved_a, .. }
                | TapeOp::SiLU { saved_a, .. } => {
                    tensor_free(*saved_a);
                }
                TapeOp::Gather { indices_ptr, .. } => {
                    tensor_free(*indices_ptr);
                }
                TapeOp::EmbeddingLookup { saved_weight, saved_indices, .. } => {
                    tensor_free(*saved_weight);
                    tensor_free(*saved_indices);
                }
                TapeOp::LayerNorm { saved_input, saved_mean, saved_inv_std, saved_weight, .. } => {
                    tensor_free(*saved_input);
                    tensor_free(*saved_mean);
                    tensor_free(*saved_inv_std);
                    tensor_free(*saved_weight);
                }
                TapeOp::RMSNorm { saved_input, saved_rms, saved_weight, .. } => {
                    tensor_free(*saved_input);
                    tensor_free(*saved_rms);
                    tensor_free(*saved_weight);
                }
                TapeOp::Dropout { saved_mask, .. } => {
                    tensor_free(*saved_mask);
                }
                TapeOp::Conv2d { saved_input, saved_weight, .. } => {
                    tensor_free(*saved_input);
                    tensor_free(*saved_weight);
                }
                TapeOp::BiasAdd { tensor, bias, .. } => {
                    tensor_free(*tensor);
                    tensor_free(*bias);
                }
                _ => {}
            }
        }

        tape.ops.clear();
        tape.param_set.clear();
    });
}

/// Pause recording (used by @no_grad). Increments pause depth.
#[no_mangle]
pub extern "C" fn nsl_tape_pause() {
    TAPE.with(|t| {
        t.borrow_mut().pause_depth += 1;
    });
}

/// Resume recording after a pause. Decrements pause depth.
#[no_mangle]
pub extern "C" fn nsl_tape_resume() {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        if tape.pause_depth > 0 {
            tape.pause_depth -= 1;
        }
    });
}

/// Create a tensor of ones with the same shape as the given tensor.
fn ones_like(tensor_ptr: i64) -> i64 {
    let shape_list = tensor_shape(tensor_ptr);
    let result = tensor_ones(shape_list);
    crate::list::nsl_list_free(shape_list);
    result
}

/// Accumulate gradient: grads[key] += grad_tensor.
/// If key doesn't exist yet, set it directly. If it does, add and free the old/input tensors.
fn accumulate_grad(grads: &mut HashMap<i64, i64>, key: i64, grad_tensor: i64) {
    if let Some(existing) = grads.get(&key) {
        let old = *existing;
        // Pause recording to avoid taping gradient computation ops
        TAPE.with(|t| t.borrow_mut().pause_depth += 1);
        let summed = tensor_add(old, grad_tensor);
        TAPE.with(|t| t.borrow_mut().pause_depth -= 1);
        tensor_free(old);
        tensor_free(grad_tensor);
        grads.insert(key, summed);
    } else {
        grads.insert(key, grad_tensor);
    }
}

/// Create a tensor with the given shape, filled with zeros.
fn create_tensor_with_shape(shape: &[i64], fill: f64) -> i64 {
    use crate::memory::checked_alloc_zeroed;
    use crate::tensor::NslTensor;

    let ndim = shape.len() as i64;
    let mut total: i64 = 1;
    for &s in shape {
        total *= s;
    }

    let shape_ptr = crate::memory::checked_alloc(shape.len() * std::mem::size_of::<i64>()) as *mut i64;
    for (i, &s) in shape.iter().enumerate() {
        unsafe { *shape_ptr.add(i) = s };
    }
    let strides = NslTensor::compute_strides(shape_ptr, ndim);

    let data_size = (total as usize) * std::mem::size_of::<f64>();
    let data = if fill == 0.0 {
        checked_alloc_zeroed(data_size) as *mut f64
    } else {
        let data = crate::memory::checked_alloc(data_size) as *mut f64;
        for i in 0..total as usize {
            unsafe { *data.add(i) = fill };
        }
        data
    };

    let tensor = Box::new(NslTensor {
        data: data as *mut c_void,
        shape: shape_ptr,
        strides,
        ndim,
        len: total,
        refcount: 1,
        device: 0,
        dtype: 0,
    });
    Box::into_raw(tensor) as i64
}

/// Reduce a gradient to match the original input shape after broadcasting.
/// If grad has shape [10, 256] but original input was [10, 1], sum along dim 1 with keepdim.
/// If grad has ndim 2 but original was ndim 1, reduce and squeeze appropriately.
fn reduce_grad_for_broadcast(grad_ptr: i64, orig_shape: &[i64]) -> i64 {
    use crate::tensor::NslTensor;

    let grad = NslTensor::from_ptr(grad_ptr);
    let grad_ndim = grad.ndim as usize;
    let orig_ndim = orig_shape.len();

    // Build the right-aligned original shape padded with 1s
    let mut orig_padded = vec![1i64; grad_ndim];
    for i in 0..orig_ndim {
        orig_padded[grad_ndim - orig_ndim + i] = orig_shape[i];
    }

    // Check which dims were broadcast (orig was 1 but grad is > 1)
    let grad_shape: Vec<i64> = (0..grad_ndim)
        .map(|i| unsafe { *grad.shape.add(i) })
        .collect();

    // If shapes already match, return clone
    if grad_shape == orig_shape {
        return tensor_clone(grad_ptr);
    }

    let mut result = tensor_clone(grad_ptr);

    // Sum along dimensions that were broadcast (where orig was 1 but grad > 1)
    // We also need to handle leading dims that don't exist in orig (padded with 1)
    for d in 0..grad_ndim {
        if orig_padded[d] == 1 && grad_shape[d] > 1 {
            let old = result;
            // Sum along dim d, keepdim=true to maintain ndim
            result = crate::tensor::nsl_tensor_sum_dim(result, d as i64, 1);
            if old != grad_ptr {
                tensor_free(old);
            }
        }
    }

    // If orig had fewer dims, squeeze the leading dims
    if orig_ndim < grad_ndim {
        let res = NslTensor::from_ptr(result);
        // Reshape to orig_shape
        let new_shape = checked_alloc(orig_ndim * std::mem::size_of::<i64>()) as *mut i64;
        for i in 0..orig_ndim {
            unsafe { *new_shape.add(i) = orig_shape[i] };
        }
        let new_strides = NslTensor::compute_strides(new_shape, orig_ndim as i64);
        let mut total: i64 = 1;
        for &s in orig_shape {
            total *= s;
        }
        // Copy data
        let new_data = checked_alloc((total as usize) * std::mem::size_of::<f64>()) as *mut f64;
        unsafe { std::ptr::copy_nonoverlapping(res.data_f64(), new_data, total as usize) };
        let out = Box::new(NslTensor {
            data: new_data as *mut c_void,
            shape: new_shape,
            strides: new_strides,
            ndim: orig_ndim as i64,
            len: total,
            refcount: 1,
        device: 0,
        dtype: 0,
    });
        let out_ptr = Box::into_raw(out) as i64;
        if result != grad_ptr {
            tensor_free(result);
        }
        return out_ptr;
    }

    result
}

/// Broadcast a gradient tensor along a reduced dimension back to input_shape.
/// grad has the reduced shape; we need to expand it along `dim` to match input_shape.
fn broadcast_grad_along_dim(grad_ptr: i64, input_shape: &[i64], dim: usize) -> i64 {
    use crate::tensor::NslTensor;

    let grad = NslTensor::from_ptr(grad_ptr);
    let ndim = input_shape.len();

    // Create output with input_shape
    let out_ptr = create_tensor_with_shape(input_shape, 0.0);
    let out = NslTensor::from_ptr(out_ptr);

    let out_strides: Vec<usize> = (0..ndim)
        .map(|i| unsafe { *out.strides.add(i) } as usize)
        .collect();

    // Build grad strides for indexing: grad has shape with dim removed (or dim=1 if keepdim)
    // We compute the grad linear index by skipping the reduced dim
    let grad_strides: Vec<usize> = (0..grad.ndim as usize)
        .map(|i| unsafe { *grad.strides.add(i) } as usize)
        .collect();

    let total = out.len as usize;
    for flat_idx in 0..total {
        // Convert flat_idx to multi-dim indices using output strides
        let mut remaining = flat_idx;
        let mut indices = vec![0usize; ndim];
        for d in 0..ndim {
            indices[d] = remaining / out_strides[d];
            remaining %= out_strides[d];
        }

        // Compute grad index by removing the reduced dim
        let mut grad_idx = 0usize;
        let mut gi = 0usize;
        for d in 0..ndim {
            if d == dim {
                continue;
            }
            grad_idx += indices[d] * grad_strides[gi];
            gi += 1;
        }

        unsafe {
            *out.data_f64().add(flat_idx) = *grad.data_f64().add(grad_idx);
        }
    }

    out_ptr
}

/// Scatter gradient to argmax positions for ReduceMax backward.
fn scatter_grad_to_argmax(
    grad_ptr: i64,
    input_shape: &[i64],
    dim: usize,
    argmax: &[usize],
) -> i64 {
    use crate::tensor::NslTensor;

    let grad = NslTensor::from_ptr(grad_ptr);
    let ndim = input_shape.len();
    // Create zero output with input_shape
    let out_ptr = create_tensor_with_shape(input_shape, 0.0);
    let out = NslTensor::from_ptr(out_ptr);

    let out_strides: Vec<usize> = (0..ndim)
        .map(|i| unsafe { *out.strides.add(i) } as usize)
        .collect();

    let grad_strides: Vec<usize> = (0..grad.ndim as usize)
        .map(|i| unsafe { *grad.strides.add(i) } as usize)
        .collect();

    // Iterate over reduced shape positions (grad positions)
    let grad_total = grad.len as usize;
    for grad_flat in 0..grad_total {
        // Convert to multi-index in grad space
        let mut remaining = grad_flat;
        let mut grad_indices = vec![0usize; grad.ndim as usize];
        for d in 0..grad.ndim as usize {
            grad_indices[d] = remaining / grad_strides[d];
            remaining %= grad_strides[d];
        }

        let max_idx = argmax[grad_flat];

        // Build the full output index: insert the argmax at dim
        let mut out_offset = 0usize;
        let mut gi = 0usize;
        for d in 0..ndim {
            if d == dim {
                out_offset += max_idx * out_strides[d];
            } else {
                out_offset += grad_indices[gi] * out_strides[d];
                gi += 1;
            }
        }

        let g_val = unsafe { *grad.data_f64().add(grad_flat) };
        unsafe { *out.data_f64().add(out_offset) += g_val };
    }

    out_ptr
}

/// Scatter gather gradient back to input shape.
fn scatter_gather_grad(
    grad_ptr: i64,
    input_shape: &[i64],
    dim: usize,
    indices_ptr: i64,
) -> i64 {
    use crate::tensor::NslTensor;

    let grad = NslTensor::from_ptr(grad_ptr);
    let indices = NslTensor::from_ptr(indices_ptr);

    // Create zero gradient with input_shape
    let out_ptr = create_tensor_with_shape(input_shape, 0.0);
    let out = NslTensor::from_ptr(out_ptr);

    let out_strides: Vec<usize> = (0..input_shape.len())
        .map(|i| unsafe { *out.strides.add(i) } as usize)
        .collect();

    // For cross_entropy: input=[batch, classes], dim=1, indices=[batch]
    // grad=[batch], output[b, indices[b]] += grad[b]
    let batch = indices.len as usize;
    for b in 0..batch {
        let idx = unsafe { *indices.data_f64().add(b) } as usize;
        let g_val = unsafe { *grad.data_f64().add(b) };
        // out_offset = b * stride[0] + idx * stride[1]
        let mut out_offset = 0usize;
        // Build index: for dimensions before dim, use b; at dim, use idx
        // For general case with dim=1 and 2D input:
        // indices is 1D [batch], grad is 1D [batch]
        if input_shape.len() == 2 && dim == 1 {
            out_offset = b * out_strides[0] + idx * out_strides[1];
        } else {
            // General case: not needed for cross_entropy but let's handle 2D dim=0 too
            if dim == 0 {
                out_offset = idx * out_strides[0] + b * out_strides[1];
            }
        }
        unsafe { *out.data_f64().add(out_offset) += g_val };
    }

    out_ptr
}

/// ReLU backward: grad * (input > 0 ? 1 : 0)
fn relu_backward(grad_ptr: i64, input_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    let grad = NslTensor::from_ptr(grad_ptr);
    let input = NslTensor::from_ptr(input_ptr);
    let len = input.len as usize;
    let ndim = input.ndim;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = crate::memory::checked_alloc(len * std::mem::size_of::<f64>()) as *mut f64;
    for i in 0..len {
        let x = unsafe { *input.data_f64().add(i) };
        let g = unsafe { *grad.data_f64().add(i) };
        unsafe { *data.add(i) = if x > 0.0 { g } else { 0.0 } };
    }
    let t = Box::new(NslTensor { data: data as *mut c_void, shape, strides, ndim, len: len as i64, refcount: 1, device: 0, dtype: 0 });
    Box::into_raw(t) as i64
}

/// GELU backward: gelu'(x) = 0.5*(1+tanh(c*(x+0.044715*x^3))) + 0.5*x*sech^2(c*(x+0.044715*x^3))*c*(1+3*0.044715*x^2)
fn gelu_backward(grad_ptr: i64, input_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    let grad = NslTensor::from_ptr(grad_ptr);
    let input = NslTensor::from_ptr(input_ptr);
    let len = input.len as usize;
    let ndim = input.ndim;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = crate::memory::checked_alloc(len * std::mem::size_of::<f64>()) as *mut f64;
    let c = (2.0_f64 / std::f64::consts::PI).sqrt();
    for i in 0..len {
        let x = unsafe { *input.data_f64().add(i) };
        let g = unsafe { *grad.data_f64().add(i) };
        let inner = c * (x + 0.044715 * x * x * x);
        let tanh_inner = inner.tanh();
        let sech2 = 1.0 - tanh_inner * tanh_inner;
        let deriv = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2 * c * (1.0 + 3.0 * 0.044715 * x * x);
        unsafe { *data.add(i) = g * deriv };
    }
    let t = Box::new(NslTensor { data: data as *mut c_void, shape, strides, ndim, len: len as i64, refcount: 1, device: 0, dtype: 0 });
    Box::into_raw(t) as i64
}

/// SiLU backward: sigmoid(x) * (1 + x * (1 - sigmoid(x)))
fn silu_backward(grad_ptr: i64, input_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    let grad = NslTensor::from_ptr(grad_ptr);
    let input = NslTensor::from_ptr(input_ptr);
    let len = input.len as usize;
    let ndim = input.ndim;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(input.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = crate::memory::checked_alloc(len * std::mem::size_of::<f64>()) as *mut f64;
    for i in 0..len {
        let x = unsafe { *input.data_f64().add(i) };
        let g = unsafe { *grad.data_f64().add(i) };
        let sig = 1.0 / (1.0 + (-x).exp());
        let deriv = sig * (1.0 + x * (1.0 - sig));
        unsafe { *data.add(i) = g * deriv };
    }
    let t = Box::new(NslTensor { data: data as *mut c_void, shape, strides, ndim, len: len as i64, refcount: 1, device: 0, dtype: 0 });
    Box::into_raw(t) as i64
}

/// Sigmoid backward: out * (1 - out)
fn sigmoid_backward(grad_ptr: i64, out_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    let grad = NslTensor::from_ptr(grad_ptr);
    let out = NslTensor::from_ptr(out_ptr);
    let len = out.len as usize;
    let ndim = out.ndim;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(out.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = crate::memory::checked_alloc(len * std::mem::size_of::<f64>()) as *mut f64;
    for i in 0..len {
        let o = unsafe { *out.data_f64().add(i) };
        let g = unsafe { *grad.data_f64().add(i) };
        unsafe { *data.add(i) = g * o * (1.0 - o) };
    }
    let t = Box::new(NslTensor { data: data as *mut c_void, shape, strides, ndim, len: len as i64, refcount: 1, device: 0, dtype: 0 });
    Box::into_raw(t) as i64
}

/// Tanh backward: 1 - out^2
fn tanh_backward(grad_ptr: i64, out_ptr: i64) -> i64 {
    use crate::tensor::NslTensor;
    let grad = NslTensor::from_ptr(grad_ptr);
    let out = NslTensor::from_ptr(out_ptr);
    let len = out.len as usize;
    let ndim = out.ndim;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(out.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = crate::memory::checked_alloc(len * std::mem::size_of::<f64>()) as *mut f64;
    for i in 0..len {
        let o = unsafe { *out.data_f64().add(i) };
        let g = unsafe { *grad.data_f64().add(i) };
        unsafe { *data.add(i) = g * (1.0 - o * o) };
    }
    let t = Box::new(NslTensor { data: data as *mut c_void, shape, strides, ndim, len: len as i64, refcount: 1, device: 0, dtype: 0 });
    Box::into_raw(t) as i64
}

/// Softmax backward: grad_input_i = output_i * (grad_i - sum(grad * output)) along dim
fn softmax_backward(grad_ptr: i64, out_ptr: i64, dim: i64) -> i64 {
    use crate::tensor::NslTensor;
    let grad = NslTensor::from_ptr(grad_ptr);
    let out = NslTensor::from_ptr(out_ptr);
    let len = out.len as usize;
    let ndim = out.ndim;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(out.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = crate::memory::checked_alloc_zeroed(len * std::mem::size_of::<f64>()) as *mut f64;

    let d = if dim < 0 { (ndim + dim) as usize } else { dim as usize };
    let o_shape: Vec<i64> = (0..ndim as usize).map(|i| unsafe { *out.shape.add(i) }).collect();
    let o_strides: Vec<i64> = (0..ndim as usize).map(|i| unsafe { *out.strides.add(i) }).collect();
    let dim_size = o_shape[d] as usize;
    let num_slices = len / dim_size;

    for slice_idx in 0..num_slices {
        let mut remaining = slice_idx;
        let mut base_offset: usize = 0;
        for axis in (0..ndim as usize).rev() {
            if axis == d { continue; }
            let idx = remaining % (o_shape[axis] as usize);
            remaining /= o_shape[axis] as usize;
            base_offset += idx * (o_strides[axis] as usize);
        }

        // Compute dot = sum(grad * output) along this slice
        let mut dot = 0.0_f64;
        for k in 0..dim_size {
            let offset = base_offset + k * (o_strides[d] as usize);
            let g = unsafe { *grad.data_f64().add(offset) };
            let o = unsafe { *out.data_f64().add(offset) };
            dot += g * o;
        }

        // grad_input_i = output_i * (grad_i - dot)
        for k in 0..dim_size {
            let offset = base_offset + k * (o_strides[d] as usize);
            let g = unsafe { *grad.data_f64().add(offset) };
            let o = unsafe { *out.data_f64().add(offset) };
            unsafe { *data.add(offset) = o * (g - dot) };
        }
    }

    let t = Box::new(NslTensor { data: data as *mut c_void, shape, strides, ndim, len: len as i64, refcount: 1, device: 0, dtype: 0 });
    Box::into_raw(t) as i64
}

/// Slice backward: create zeros with input_shape, copy grad into the [start, start+slice_len) region along dim.
fn slice_backward(grad_ptr: i64, input_shape: &[i64], dim: usize, start: usize) -> i64 {
    use crate::tensor::NslTensor;

    let grad = NslTensor::from_ptr(grad_ptr);
    let ndim = input_shape.len();

    // Create zero output with original input_shape
    let out_ptr = create_tensor_with_shape(input_shape, 0.0);
    let out = NslTensor::from_ptr(out_ptr);

    let out_strides: Vec<usize> = (0..ndim)
        .map(|i| unsafe { *out.strides.add(i) } as usize)
        .collect();
    let grad_strides: Vec<usize> = (0..ndim)
        .map(|i| unsafe { *grad.strides.add(i) } as usize)
        .collect();

    let grad_total = grad.len as usize;
    for flat in 0..grad_total {
        // Convert flat index in grad to multi-dim, offset by start on the sliced dim
        let mut remaining = flat;
        let mut out_offset: usize = 0;
        for axis in 0..ndim {
            let idx = remaining / grad_strides[axis];
            remaining %= grad_strides[axis];
            if axis == dim {
                out_offset += (idx + start) * out_strides[axis];
            } else {
                out_offset += idx * out_strides[axis];
            }
        }
        unsafe { *out.data_f64().add(out_offset) = *grad.data_f64().add(flat) };
    }

    out_ptr
}

/// Cat backward: split the gradient along the cat dim into pieces matching the original input sizes.
fn cat_backward(grad_ptr: i64, dim: usize, split_sizes: &[i64]) -> Vec<i64> {
    use crate::tensor::NslTensor;

    let grad = NslTensor::from_ptr(grad_ptr);
    let ndim = grad.ndim as usize;

    let grad_shape: Vec<i64> = (0..ndim)
        .map(|i| unsafe { *grad.shape.add(i) })
        .collect();
    let grad_strides: Vec<usize> = (0..ndim)
        .map(|i| unsafe { *grad.strides.add(i) } as usize)
        .collect();

    let mut results = Vec::with_capacity(split_sizes.len());
    let mut offset: usize = 0;

    for &split_size in split_sizes {
        // Build shape for this piece
        let mut piece_shape = grad_shape.clone();
        piece_shape[dim] = split_size;

        let piece_ptr = create_tensor_with_shape(&piece_shape, 0.0);
        let piece = NslTensor::from_ptr(piece_ptr);
        let piece_strides: Vec<usize> = (0..ndim)
            .map(|i| unsafe { *piece.strides.add(i) } as usize)
            .collect();

        let piece_total = piece.len as usize;
        for flat in 0..piece_total {
            let mut remaining = flat;
            let mut grad_offset: usize = 0;
            for axis in 0..ndim {
                let idx = remaining / piece_strides[axis];
                remaining %= piece_strides[axis];
                if axis == dim {
                    grad_offset += (idx + offset) * grad_strides[axis];
                } else {
                    grad_offset += idx * grad_strides[axis];
                }
            }
            unsafe { *piece.data_f64().add(flat) = *grad.data_f64().add(grad_offset) };
        }

        results.push(piece_ptr);
        offset += split_size as usize;
    }

    results
}

/// LayerNorm backward: returns (grad_input, grad_weight, grad_bias)
/// Per row of size N:
///   d_normalized = d_out * weight
///   d_x = (1/N) * inv_std * (N * d_normalized - sum(d_normalized) - normalized * sum(d_normalized * normalized))
///   d_weight = sum(d_out * normalized, across batch)
///   d_bias = sum(d_out, across batch)
fn layernorm_backward(grad_ptr: i64, input_ptr: i64, mean_ptr: i64, inv_std_ptr: i64, weight_ptr: i64) -> (i64, i64, i64) {
    use crate::tensor::NslTensor;

    let grad = NslTensor::from_ptr(grad_ptr);
    let input = NslTensor::from_ptr(input_ptr);
    let mean_t = NslTensor::from_ptr(mean_ptr);
    let inv_std_t = NslTensor::from_ptr(inv_std_ptr);
    let weight = NslTensor::from_ptr(weight_ptr);

    let total = input.len as usize;
    let ndim = input.ndim as usize;
    let n = unsafe { *input.shape.add(ndim - 1) } as usize;
    let num_rows = total / n;
    let nf = n as f64;

    // grad_input: same shape as input
    let dx_ptr = create_tensor_with_shape(
        &(0..ndim).map(|i| unsafe { *input.shape.add(i) }).collect::<Vec<_>>(),
        0.0,
    );
    let dx = NslTensor::from_ptr(dx_ptr);

    // grad_weight: shape [N]
    let dw_ptr = create_tensor_with_shape(&[n as i64], 0.0);
    let dw = NslTensor::from_ptr(dw_ptr);

    // grad_bias: shape [N]
    let db_ptr = create_tensor_with_shape(&[n as i64], 0.0);
    let db = NslTensor::from_ptr(db_ptr);

    for row in 0..num_rows {
        let base = row * n;
        let mean_val = unsafe { *mean_t.data_f64().add(row) };
        let inv_std_val = unsafe { *inv_std_t.data_f64().add(row) };

        // Compute d_normalized = d_out * weight, and normalized values
        // Also accumulate sum(d_normalized) and sum(d_normalized * normalized)
        let mut sum_dnorm = 0.0_f64;
        let mut sum_dnorm_norm = 0.0_f64;

        for j in 0..n {
            let g = unsafe { *grad.data_f64().add(base + j) };
            let w = unsafe { *weight.data_f64().add(j) };
            let x = unsafe { *input.data_f64().add(base + j) };
            let normalized = (x - mean_val) * inv_std_val;
            let d_normalized = g * w;
            sum_dnorm += d_normalized;
            sum_dnorm_norm += d_normalized * normalized;

            // Accumulate d_weight and d_bias
            unsafe {
                *dw.data_f64().add(j) += g * normalized;
                *db.data_f64().add(j) += g;
            }
        }

        // Compute d_x for each element in this row
        for j in 0..n {
            let g = unsafe { *grad.data_f64().add(base + j) };
            let w = unsafe { *weight.data_f64().add(j) };
            let x = unsafe { *input.data_f64().add(base + j) };
            let normalized = (x - mean_val) * inv_std_val;
            let d_normalized = g * w;
            let d_x = (1.0 / nf) * inv_std_val * (nf * d_normalized - sum_dnorm - normalized * sum_dnorm_norm);
            unsafe { *dx.data_f64().add(base + j) = d_x };
        }
    }

    (dx_ptr, dw_ptr, db_ptr)
}

/// RMSNorm backward: returns (grad_input, grad_weight)
/// Per row of size N:
///   d_x = weight * (d_out / rms - x * sum(d_out * x) / (N * rms^3))
///   d_weight = sum(d_out * (x / rms), across batch)
fn rmsnorm_backward(grad_ptr: i64, input_ptr: i64, rms_ptr: i64, weight_ptr: i64) -> (i64, i64) {
    use crate::tensor::NslTensor;

    let grad = NslTensor::from_ptr(grad_ptr);
    let input = NslTensor::from_ptr(input_ptr);
    let rms_t = NslTensor::from_ptr(rms_ptr);
    let weight = NslTensor::from_ptr(weight_ptr);

    let total = input.len as usize;
    let ndim = input.ndim as usize;
    let n = unsafe { *input.shape.add(ndim - 1) } as usize;
    let num_rows = total / n;
    let nf = n as f64;

    // grad_input: same shape as input
    let dx_ptr = create_tensor_with_shape(
        &(0..ndim).map(|i| unsafe { *input.shape.add(i) }).collect::<Vec<_>>(),
        0.0,
    );
    let dx = NslTensor::from_ptr(dx_ptr);

    // grad_weight: shape [N]
    let dw_ptr = create_tensor_with_shape(&[n as i64], 0.0);
    let dw = NslTensor::from_ptr(dw_ptr);

    for row in 0..num_rows {
        let base = row * n;
        let rms_val = unsafe { *rms_t.data_f64().add(row) };
        let rms_cubed = rms_val * rms_val * rms_val;

        // Compute sum(d_out * weight * x) for this row
        let mut sum_dout_x = 0.0_f64;
        for j in 0..n {
            let g = unsafe { *grad.data_f64().add(base + j) };
            let w = unsafe { *weight.data_f64().add(j) };
            let x = unsafe { *input.data_f64().add(base + j) };
            sum_dout_x += g * w * x;
        }

        for j in 0..n {
            let g = unsafe { *grad.data_f64().add(base + j) };
            let w = unsafe { *weight.data_f64().add(j) };
            let x = unsafe { *input.data_f64().add(base + j) };

            // d_x = w * (d_out / rms - x * sum(d_out * w * x) / (N * rms^3))
            let d_x = w * (g / rms_val - x * sum_dout_x / (nf * rms_cubed));
            unsafe { *dx.data_f64().add(base + j) = d_x };

            // d_weight accumulate: d_out * (x / rms)
            unsafe { *dw.data_f64().add(j) += g * (x / rms_val) };
        }
    }

    (dx_ptr, dw_ptr)
}

/// Dropout backward: grad_input = grad_output * mask * scale
fn dropout_backward(grad_ptr: i64, mask_ptr: i64, scale: f64) -> i64 {
    use crate::tensor::NslTensor;

    let grad = NslTensor::from_ptr(grad_ptr);
    let mask = NslTensor::from_ptr(mask_ptr);
    let len = grad.len as usize;
    let ndim = grad.ndim;
    let shape = crate::memory::checked_alloc((ndim as usize) * std::mem::size_of::<i64>()) as *mut i64;
    unsafe { std::ptr::copy_nonoverlapping(grad.shape, shape, ndim as usize) };
    let strides = NslTensor::compute_strides(shape, ndim);
    let data = crate::memory::checked_alloc(len * std::mem::size_of::<f64>()) as *mut f64;
    for i in 0..len {
        let g = unsafe { *grad.data_f64().add(i) };
        let m = unsafe { *mask.data_f64().add(i) };
        unsafe { *data.add(i) = g * m * scale };
    }
    let t = Box::new(NslTensor { data: data as *mut c_void, shape, strides, ndim, len: len as i64, refcount: 1, device: 0, dtype: 0 });
    Box::into_raw(t) as i64
}

/// Conv2d backward: returns (grad_input, grad_weight, grad_bias)
fn conv2d_backward(
    grad_ptr: i64,
    input_ptr: i64,
    weight_ptr: i64,
    stride_h: usize,
    stride_w: usize,
    pad_h: usize,
    pad_w: usize,
) -> (i64, i64, i64) {
    use crate::tensor::NslTensor;

    let grad = NslTensor::from_ptr(grad_ptr);
    let input = NslTensor::from_ptr(input_ptr);
    let weight = NslTensor::from_ptr(weight_ptr);

    let n = unsafe { *input.shape.add(0) } as usize;
    let c_in = unsafe { *input.shape.add(1) } as usize;
    let h = unsafe { *input.shape.add(2) } as usize;
    let w = unsafe { *input.shape.add(3) } as usize;

    let c_out = unsafe { *weight.shape.add(0) } as usize;
    let kh = unsafe { *weight.shape.add(2) } as usize;
    let kw = unsafe { *weight.shape.add(3) } as usize;

    let h_out = unsafe { *grad.shape.add(2) } as usize;
    let w_out = unsafe { *grad.shape.add(3) } as usize;

    // Allocate grad_input [N, C_in, H, W]
    let dx_ptr = create_tensor_with_shape(&[n as i64, c_in as i64, h as i64, w as i64], 0.0);
    let dx = NslTensor::from_ptr(dx_ptr);

    // Allocate grad_weight [C_out, C_in, kH, kW]
    let dw_ptr = create_tensor_with_shape(&[c_out as i64, c_in as i64, kh as i64, kw as i64], 0.0);
    let dw = NslTensor::from_ptr(dw_ptr);

    // Allocate grad_bias [C_out]
    let db_ptr = create_tensor_with_shape(&[c_out as i64], 0.0);
    let db = NslTensor::from_ptr(db_ptr);

    for ni in 0..n {
        for co in 0..c_out {
            for oh in 0..h_out {
                for ow in 0..w_out {
                    let g_idx = ni * (c_out * h_out * w_out) + co * (h_out * w_out) + oh * w_out + ow;
                    let g_val = unsafe { *grad.data_f64().add(g_idx) };

                    // grad_bias[co] += grad_out[n][co][oh][ow]
                    unsafe { *db.data_f64().add(co) += g_val };

                    for ci in 0..c_in {
                        for ky in 0..kh {
                            for kx in 0..kw {
                                let ih = oh * stride_h + ky;
                                let iw = ow * stride_w + kx;
                                if ih >= pad_h && iw >= pad_w && ih - pad_h < h && iw - pad_w < w {
                                    let real_ih = ih - pad_h;
                                    let real_iw = iw - pad_w;
                                    let input_idx = ni * (c_in * h * w) + ci * (h * w) + real_ih * w + real_iw;
                                    let weight_idx = co * (c_in * kh * kw) + ci * (kh * kw) + ky * kw + kx;

                                    // grad_weight[co][ci][ky][kx] += grad_out * input
                                    unsafe { *dw.data_f64().add(weight_idx) += g_val * *input.data_f64().add(input_idx) };

                                    // grad_input[ni][ci][real_ih][real_iw] += grad_out * weight
                                    unsafe { *dx.data_f64().add(input_idx) += g_val * *weight.data_f64().add(weight_idx) };
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    (dx_ptr, dw_ptr, db_ptr)
}

/// MaxPool2d backward: scatter gradient to argmax positions
fn maxpool2d_backward(grad_ptr: i64, input_shape: &[i64], argmax: &[usize]) -> i64 {
    use crate::tensor::NslTensor;

    let grad = NslTensor::from_ptr(grad_ptr);
    let out_ptr = create_tensor_with_shape(input_shape, 0.0);
    let out = NslTensor::from_ptr(out_ptr);

    let grad_len = grad.len as usize;
    for i in 0..grad_len {
        let g_val = unsafe { *grad.data_f64().add(i) };
        let idx = argmax[i];
        unsafe { *out.data_f64().add(idx) += g_val };
    }

    out_ptr
}

/// Run backward pass. `loss_ptr` is the scalar loss tensor. `param_list` is an NslList of
/// parameter tensor pointers. Returns an NslList of gradient tensors (one per param, same order).
#[no_mangle]
pub extern "C" fn nsl_tape_backward(loss_ptr: i64, param_list: i64) -> i64 {
    let params_nsl = NslList::from_ptr(param_list);
    let param_ptrs: Vec<i64> = (0..params_nsl.len as usize)
        .map(|i| unsafe { *params_nsl.data.add(i) })
        .collect();

    // Pause recording during backward (don't tape gradient computation)
    TAPE.with(|t| t.borrow_mut().pause_depth += 1);

    let ops = TAPE.with(|t| t.borrow().ops.clone());

    // grad_map: tensor_ptr -> gradient tensor ptr
    let mut grad_map: HashMap<i64, i64> = HashMap::new();

    // Seed: gradient of loss w.r.t. itself = ones_like(loss)
    let seed = ones_like(loss_ptr);
    grad_map.insert(loss_ptr, seed);

    // Walk tape in reverse, applying chain rule
    for op in ops.iter().rev() {
        match op {
            TapeOp::Add { a, b, out, a_shape, b_shape } => {
                if let Some(&g) = grad_map.get(out) {
                    let ga = reduce_grad_for_broadcast(g, a_shape);
                    let gb = reduce_grad_for_broadcast(g, b_shape);
                    accumulate_grad(&mut grad_map, *a, ga);
                    accumulate_grad(&mut grad_map, *b, gb);
                }
            }
            TapeOp::Sub { a, b, out, a_shape, b_shape } => {
                if let Some(&g) = grad_map.get(out) {
                    let ga = reduce_grad_for_broadcast(g, a_shape);
                    let gb_full = tensor_neg(tensor_clone(g));
                    let gb = reduce_grad_for_broadcast(gb_full, b_shape);
                    if gb != gb_full { tensor_free(gb_full); }
                    accumulate_grad(&mut grad_map, *a, ga);
                    accumulate_grad(&mut grad_map, *b, gb);
                }
            }
            TapeOp::Mul { a, b, out, saved_a, saved_b, a_shape, b_shape } => {
                if let Some(&g) = grad_map.get(out) {
                    // d/da(a*b) = g * b, d/db(a*b) = g * a
                    let g_clone1 = tensor_clone(g);
                    let g_clone2 = tensor_clone(g);
                    let grad_a_full = tensor_mul(g_clone1, *saved_b);
                    let grad_b_full = tensor_mul(g_clone2, *saved_a);
                    tensor_free(g_clone1);
                    tensor_free(g_clone2);
                    let grad_a = reduce_grad_for_broadcast(grad_a_full, a_shape);
                    let grad_b = reduce_grad_for_broadcast(grad_b_full, b_shape);
                    if grad_a != grad_a_full { tensor_free(grad_a_full); }
                    if grad_b != grad_b_full { tensor_free(grad_b_full); }
                    accumulate_grad(&mut grad_map, *a, grad_a);
                    accumulate_grad(&mut grad_map, *b, grad_b);
                }
            }
            TapeOp::Div { a, b, out, saved_a, saved_b, a_shape, b_shape } => {
                if let Some(&g) = grad_map.get(out) {
                    // d/da(a/b) = g / b
                    let g_clone1 = tensor_clone(g);
                    let grad_a_full = tensor_div(g_clone1, *saved_b);
                    tensor_free(g_clone1);
                    let grad_a = reduce_grad_for_broadcast(grad_a_full, a_shape);
                    if grad_a != grad_a_full { tensor_free(grad_a_full); }
                    accumulate_grad(&mut grad_map, *a, grad_a);

                    // d/db(a/b) = -g * a / b^2
                    let g_clone2 = tensor_clone(g);
                    let neg_g = tensor_neg(g_clone2);
                    tensor_free(g_clone2);
                    let neg_ga = tensor_mul(neg_g, *saved_a);
                    let b_sq = tensor_mul(*saved_b, *saved_b);
                    let grad_b_full = tensor_div(neg_ga, b_sq);
                    tensor_free(neg_g);
                    tensor_free(neg_ga);
                    tensor_free(b_sq);
                    let grad_b = reduce_grad_for_broadcast(grad_b_full, b_shape);
                    if grad_b != grad_b_full { tensor_free(grad_b_full); }
                    accumulate_grad(&mut grad_map, *b, grad_b);
                }
            }
            TapeOp::MatMul { a, b, out, saved_a, saved_b } => {
                if let Some(&g) = grad_map.get(out) {
                    // d/dA(A@B) = G @ B^T, d/dB(A@B) = A^T @ G
                    // Use -2,-1 to transpose the last two dims (correct for batched matmul)
                    let b_t = tensor_transpose(*saved_b, -2, -1);
                    let a_t = tensor_transpose(*saved_a, -2, -1);
                    let g_clone1 = tensor_clone(g);
                    let g_clone2 = tensor_clone(g);
                    let grad_a = tensor_matmul(g_clone1, b_t);
                    let grad_b = tensor_matmul(a_t, g_clone2);
                    tensor_free(g_clone1);
                    tensor_free(g_clone2);
                    tensor_free(b_t);
                    tensor_free(a_t);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                    accumulate_grad(&mut grad_map, *b, grad_b);
                }
            }
            TapeOp::Neg { a, out } => {
                if let Some(&g) = grad_map.get(out) {
                    let g_clone = tensor_clone(g);
                    let neg = tensor_neg(g_clone);
                    tensor_free(g_clone);
                    accumulate_grad(&mut grad_map, *a, neg);
                }
            }
            TapeOp::MulScalar { a, scalar, out } => {
                if let Some(&g) = grad_map.get(out) {
                    let g_clone = tensor_clone(g);
                    let scaled = tensor_mul_scalar(g_clone, *scalar);
                    tensor_free(g_clone);
                    accumulate_grad(&mut grad_map, *a, scaled);
                }
            }
            TapeOp::AddScalar { a, out } => {
                if let Some(&g) = grad_map.get(out) {
                    accumulate_grad(&mut grad_map, *a, tensor_clone(g));
                }
            }
            TapeOp::Transpose { a, out, dim0, dim1 } => {
                if let Some(&g) = grad_map.get(out) {
                    // Backward of transpose(dim0, dim1) is transpose(dim0, dim1) again
                    let g_clone = tensor_clone(g);
                    let grad_a = tensor_transpose(g_clone, *dim0, *dim1);
                    tensor_free(g_clone);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::SumReduce { a, out, dim, input_shape, .. } => {
                if let Some(&g) = grad_map.get(out) {
                    if *dim == -1 {
                        // Global reduction: g is scalar, broadcast to input shape
                        let scalar_val = tensor_item(g);
                        let ones = ones_like(*a);
                        let grad_a = tensor_mul_scalar(ones, scalar_val);
                        tensor_free(ones);
                        accumulate_grad(&mut grad_map, *a, grad_a);
                    } else {
                        // Dimensional reduction: broadcast g along reduced dim
                        let grad_a = broadcast_grad_along_dim(g, input_shape, *dim as usize);
                        accumulate_grad(&mut grad_map, *a, grad_a);
                    }
                }
            }
            TapeOp::MeanReduce { a, out, dim, num_elements, input_shape, .. } => {
                if let Some(&g) = grad_map.get(out) {
                    if *dim == -1 {
                        // Global reduction
                        let scalar_val = tensor_item(g);
                        let ones = ones_like(*a);
                        let grad_a = tensor_mul_scalar(ones, scalar_val / (*num_elements as f64));
                        tensor_free(ones);
                        accumulate_grad(&mut grad_map, *a, grad_a);
                    } else {
                        // Dimensional reduction: broadcast then scale
                        let expanded = broadcast_grad_along_dim(g, input_shape, *dim as usize);
                        let grad_a = tensor_mul_scalar(expanded, 1.0 / (*num_elements as f64));
                        tensor_free(expanded);
                        accumulate_grad(&mut grad_map, *a, grad_a);
                    }
                }
            }
            TapeOp::ReduceMax { a, out, dim, saved_argmax, input_shape, .. } => {
                if let Some(&g) = grad_map.get(out) {
                    // Scatter grad to argmax positions, zero elsewhere
                    let grad_a = scatter_grad_to_argmax(g, input_shape, *dim as usize, saved_argmax);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Gather { a, out, dim, indices_ptr, input_shape } => {
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = scatter_gather_grad(g, input_shape, *dim as usize, *indices_ptr);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Exp { a, out, saved_out } => {
                // d/da(exp(a)) = exp(a) = saved_out
                if let Some(&g) = grad_map.get(out) {
                    let g_clone = tensor_clone(g);
                    let grad_a = tensor_mul(g_clone, *saved_out);
                    tensor_free(g_clone);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Log { a, out, saved_a } => {
                // d/da(log(a)) = 1/a
                if let Some(&g) = grad_map.get(out) {
                    let g_clone = tensor_clone(g);
                    let grad_a = tensor_div(g_clone, *saved_a);
                    tensor_free(g_clone);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Sqrt { a, out, saved_out } => {
                // d/da(sqrt(a)) = 1 / (2 * sqrt(a)) = 1 / (2 * saved_out)
                if let Some(&g) = grad_map.get(out) {
                    let g_clone = tensor_clone(g);
                    let two_sqrt = tensor_mul_scalar(*saved_out, 2.0);
                    let grad_a = tensor_div(g_clone, two_sqrt);
                    tensor_free(g_clone);
                    tensor_free(two_sqrt);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Abs { a, out, saved_a } => {
                // d/da(abs(a)) = sign(a)
                if let Some(&g) = grad_map.get(out) {
                    let g_clone = tensor_clone(g);
                    let sign = tensor_sign(*saved_a);
                    let grad_a = tensor_mul(g_clone, sign);
                    tensor_free(g_clone);
                    tensor_free(sign);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Clamp { a, out, saved_a, min_val, max_val } => {
                // Gradient passes through where unclamped, zero where clamped
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = crate::tensor::nsl_tensor_clamp_backward(
                        g, *saved_a, *min_val, *max_val,
                    );
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::ReLU { a, out, saved_a } => {
                // d/da(relu(a)) = (a > 0) ? 1 : 0
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = relu_backward(g, *saved_a);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::GELU { a, out, saved_a } => {
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = gelu_backward(g, *saved_a);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::SiLU { a, out, saved_a } => {
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = silu_backward(g, *saved_a);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Sigmoid { a, out, saved_out } => {
                // d/da(sigmoid(a)) = sigmoid(a) * (1 - sigmoid(a)) = out * (1 - out)
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = sigmoid_backward(g, *saved_out);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Tanh { a, out, saved_out } => {
                // d/da(tanh(a)) = 1 - tanh(a)^2 = 1 - out^2
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = tanh_backward(g, *saved_out);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Softmax { a, out, saved_out, dim } => {
                // grad_input_i = output_i * (grad_i - sum(grad * output)) along softmax dim
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = softmax_backward(g, *saved_out, *dim);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Slice { a, out, dim, start, input_shape } => {
                // Backward: create zeros with original input shape, copy grad into [start, start+slice_len)
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = slice_backward(g, input_shape, *dim as usize, *start as usize);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::EmbeddingLookup { weight, indices: _, out, saved_weight, saved_indices } => {
                // Backward: scatter-add output gradient rows into weight gradient
                if let Some(&g) = grad_map.get(out) {
                    let w = crate::tensor::NslTensor::from_ptr(*saved_weight);
                    let idx_t = crate::tensor::NslTensor::from_ptr(*saved_indices);
                    let vocab_size = unsafe { *w.shape.add(0) } as usize;
                    let embed_dim = unsafe { *w.shape.add(1) } as usize;
                    let seq_len = unsafe { *idx_t.shape.add(0) } as usize;

                    // Create zero gradient with same shape as weight [vocab_size, embed_dim]
                    let w_shape_list = tensor_shape(*saved_weight);
                    let grad_w = tensor_zeros(w_shape_list);
                    crate::list::nsl_list_free(w_shape_list);

                    let grad_w_t = crate::tensor::NslTensor::from_ptr(grad_w);
                    let g_t = crate::tensor::NslTensor::from_ptr(g);

                    // Scatter-add: for each position i, add grad_out[i, :] to grad_w[idx[i], :]
                    for i in 0..seq_len {
                        let idx = unsafe { *idx_t.data_f64().add(i) } as usize;
                        if idx < vocab_size {
                            for j in 0..embed_dim {
                                unsafe {
                                    *grad_w_t.data_f64().add(idx * embed_dim + j) +=
                                        *g_t.data_f64().add(i * embed_dim + j);
                                }
                            }
                        }
                    }

                    accumulate_grad(&mut grad_map, *weight, grad_w);
                }
            }
            TapeOp::Cat { inputs, out, dim, split_sizes } => {
                // Backward: split the gradient along the cat dim into pieces
                if let Some(&g) = grad_map.get(out) {
                    let grads = cat_backward(g, *dim as usize, split_sizes);
                    for (i, grad_piece) in grads.into_iter().enumerate() {
                        accumulate_grad(&mut grad_map, inputs[i], grad_piece);
                    }
                }
            }
            TapeOp::LayerNorm { input, weight, bias, out, saved_input, saved_mean, saved_inv_std, saved_weight } => {
                if let Some(&g) = grad_map.get(out) {
                    let grad_input = layernorm_backward(g, *saved_input, *saved_mean, *saved_inv_std, *saved_weight);
                    accumulate_grad(&mut grad_map, *input, grad_input.0);
                    accumulate_grad(&mut grad_map, *weight, grad_input.1);
                    accumulate_grad(&mut grad_map, *bias, grad_input.2);
                }
            }
            TapeOp::RMSNorm { input, weight, out, saved_input, saved_rms, saved_weight } => {
                if let Some(&g) = grad_map.get(out) {
                    let grad_result = rmsnorm_backward(g, *saved_input, *saved_rms, *saved_weight);
                    accumulate_grad(&mut grad_map, *input, grad_result.0);
                    accumulate_grad(&mut grad_map, *weight, grad_result.1);
                }
            }
            TapeOp::Dropout { a, out, saved_mask, scale } => {
                // grad_input = grad_output * mask * scale
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = dropout_backward(g, *saved_mask, *scale);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::Conv2d { input, weight, bias, out, saved_input, saved_weight, stride_h, stride_w, pad_h, pad_w } => {
                if let Some(&g) = grad_map.get(out) {
                    let (grad_input, grad_weight, grad_bias) = conv2d_backward(
                        g, *saved_input, *saved_weight, *stride_h, *stride_w, *pad_h, *pad_w,
                    );
                    accumulate_grad(&mut grad_map, *input, grad_input);
                    accumulate_grad(&mut grad_map, *weight, grad_weight);
                    if *bias != 0 {
                        accumulate_grad(&mut grad_map, *bias, grad_bias);
                    } else {
                        tensor_free(grad_bias);
                    }
                }
            }
            TapeOp::MaxPool2d { a, out, saved_argmax, input_shape } => {
                if let Some(&g) = grad_map.get(out) {
                    let grad_a = maxpool2d_backward(g, input_shape, saved_argmax);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                }
            }
            TapeOp::BiasAdd { tensor, bias, out } => {
                // grad_tensor = upstream grad (same shape as tensor [M, N])
                // grad_bias = sum(upstream grad, dim=0) -> [N]
                if let Some(&g) = grad_map.get(out) {
                    // grad for tensor is just the upstream gradient
                    let grad_tensor = tensor_clone(g);
                    accumulate_grad(&mut grad_map, *tensor, grad_tensor);

                    // grad for bias = sum over batch dim (rows)
                    let g_t = crate::tensor::NslTensor::from_ptr(g);
                    let rows = unsafe { *g_t.shape.add(0) } as usize;
                    let cols = unsafe { *g_t.shape.add(1) } as usize;

                    let bias_shape_list = crate::list::nsl_list_new();
                    crate::list::nsl_list_push(bias_shape_list, cols as i64);
                    let grad_bias = tensor_zeros(bias_shape_list);
                    crate::list::nsl_list_free(bias_shape_list);

                    let grad_bias_t = crate::tensor::NslTensor::from_ptr(grad_bias);
                    for i in 0..rows {
                        for j in 0..cols {
                            unsafe {
                                *grad_bias_t.data_f64().add(j) += *g_t.data_f64().add(i * cols + j);
                            }
                        }
                    }

                    accumulate_grad(&mut grad_map, *bias, grad_bias);
                }
            }
        }
    }

    // Resume recording
    TAPE.with(|t| t.borrow_mut().pause_depth -= 1);

    // Build result list: one gradient per param (in same order as param_list)
    let result_list = crate::list::nsl_list_new();
    for ptr in &param_ptrs {
        if let Some(grad) = grad_map.remove(ptr) {
            crate::list::nsl_list_push(result_list, grad);
        } else {
            // No gradient computed for this param — return zeros_like
            let shape_list = tensor_shape(*ptr);
            let zeros = tensor_zeros(shape_list);
            crate::list::nsl_list_free(shape_list);
            crate::list::nsl_list_push(result_list, zeros);
        }
    }

    // Free all remaining intermediate gradient tensors (seed, activation grads, etc.)
    for (_, grad_tensor) in grad_map {
        tensor_free(grad_tensor);
    }

    result_list
}
