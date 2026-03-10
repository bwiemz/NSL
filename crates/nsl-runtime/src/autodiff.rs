use std::cell::RefCell;
use std::collections::{HashMap, HashSet};

use crate::list::NslList;
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
    Add { a: i64, b: i64, out: i64 },
    Sub { a: i64, b: i64, out: i64 },
    Mul { a: i64, b: i64, out: i64, saved_a: i64, saved_b: i64 },
    Div { a: i64, b: i64, out: i64, saved_a: i64, saved_b: i64 },
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
                | TapeOp::Sqrt { saved_out, .. } => {
                    tensor_free(*saved_out);
                }
                TapeOp::Log { saved_a, .. }
                | TapeOp::Abs { saved_a, .. }
                | TapeOp::Clamp { saved_a, .. } => {
                    tensor_free(*saved_a);
                }
                TapeOp::Gather { indices_ptr, .. } => {
                    tensor_free(*indices_ptr);
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
    tensor_ones(shape_list)
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
        data,
        shape: shape_ptr,
        strides,
        ndim,
        len: total,
        refcount: 1,
    });
    Box::into_raw(tensor) as i64
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
            *out.data.add(flat_idx) = *grad.data.add(grad_idx);
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

        let g_val = unsafe { *grad.data.add(grad_flat) };
        unsafe { *out.data.add(out_offset) += g_val };
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
        let idx = unsafe { *indices.data.add(b) } as usize;
        let g_val = unsafe { *grad.data.add(b) };
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
        unsafe { *out.data.add(out_offset) += g_val };
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
            TapeOp::Add { a, b, out } => {
                if let Some(&g) = grad_map.get(out) {
                    accumulate_grad(&mut grad_map, *a, tensor_clone(g));
                    accumulate_grad(&mut grad_map, *b, tensor_clone(g));
                }
            }
            TapeOp::Sub { a, b, out } => {
                if let Some(&g) = grad_map.get(out) {
                    accumulate_grad(&mut grad_map, *a, tensor_clone(g));
                    let g_clone = tensor_clone(g);
                    let neg_g = tensor_neg(g_clone);
                    tensor_free(g_clone);
                    accumulate_grad(&mut grad_map, *b, neg_g);
                }
            }
            TapeOp::Mul { a, b, out, saved_a, saved_b } => {
                if let Some(&g) = grad_map.get(out) {
                    // d/da(a*b) = g * b, d/db(a*b) = g * a
                    let g_clone1 = tensor_clone(g);
                    let g_clone2 = tensor_clone(g);
                    let grad_a = tensor_mul(g_clone1, *saved_b);
                    let grad_b = tensor_mul(g_clone2, *saved_a);
                    tensor_free(g_clone1);
                    tensor_free(g_clone2);
                    accumulate_grad(&mut grad_map, *a, grad_a);
                    accumulate_grad(&mut grad_map, *b, grad_b);
                }
            }
            TapeOp::Div { a, b, out, saved_a, saved_b } => {
                if let Some(&g) = grad_map.get(out) {
                    // d/da(a/b) = g / b
                    let g_clone1 = tensor_clone(g);
                    let grad_a = tensor_div(g_clone1, *saved_b);
                    tensor_free(g_clone1);
                    accumulate_grad(&mut grad_map, *a, grad_a);

                    // d/db(a/b) = -g * a / b^2
                    let g_clone2 = tensor_clone(g);
                    let neg_g = tensor_neg(g_clone2);
                    tensor_free(g_clone2);
                    let neg_ga = tensor_mul(neg_g, *saved_a);
                    let b_sq = tensor_mul(*saved_b, *saved_b);
                    let grad_b = tensor_div(neg_ga, b_sq);
                    tensor_free(neg_g);
                    tensor_free(neg_ga);
                    tensor_free(b_sq);
                    accumulate_grad(&mut grad_map, *b, grad_b);
                }
            }
            TapeOp::MatMul { a, b, out, saved_a, saved_b } => {
                if let Some(&g) = grad_map.get(out) {
                    // d/dA(A@B) = G @ B^T, d/dB(A@B) = A^T @ G
                    let b_t = tensor_transpose(*saved_b, 0, 1);
                    let a_t = tensor_transpose(*saved_a, 0, 1);
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
            crate::list::nsl_list_push(result_list, zeros);
        }
    }

    // Free all remaining intermediate gradient tensors (seed, activation grads, etc.)
    for (_, grad_tensor) in grad_map {
        tensor_free(grad_tensor);
    }

    result_list
}
