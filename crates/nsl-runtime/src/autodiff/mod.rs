use std::cell::RefCell;
use std::collections::HashSet;

use crate::list::NslList;
use crate::tensor::{
    nsl_tensor_free as tensor_free,
};

pub mod backward;
pub mod grad_utils;

pub use backward::*;

/// Create a ones tensor from a shape slice with a given dtype.
/// Avoids dereferencing a raw tensor pointer (safe for freed inputs).
pub(crate) fn ones_from_shape(shape: &[i64], dtype: u16) -> i64 {
    let result = crate::cpu::create_tensor_with_shape_rs_dtype(shape, dtype);
    // Fill with 1.0
    let t = crate::tensor::NslTensor::from_ptr(result);
    if dtype == 1 {
        for i in 0..t.len as usize {
            unsafe { *t.data_f32().add(i) = 1.0 };
        }
    } else {
        for i in 0..t.len as usize {
            unsafe { *t.data_f64().add(i) = 1.0 };
        }
    }
    result
}

/// Operations recorded on the tape during forward passes inside `grad` blocks.
#[allow(dead_code)]
#[derive(Clone)]
pub enum TapeOp {
    Add { a: i64, b: i64, out: i64, a_shape: Vec<i64>, b_shape: Vec<i64> },
    Sub { a: i64, b: i64, out: i64, a_shape: Vec<i64>, b_shape: Vec<i64> },
    Mul { a: i64, b: i64, out: i64, saved_a: i64, saved_b: i64, a_shape: Vec<i64>, b_shape: Vec<i64> },
    Div { a: i64, b: i64, out: i64, saved_a: i64, saved_b: i64, a_shape: Vec<i64>, b_shape: Vec<i64> },
    MatMul { a: i64, b: i64, out: i64, saved_a: i64, saved_b: i64 },
    /// FP8 matmul with scale factors for E5M2 backward dispatch.
    /// Forward uses E4M3; backward re-quantizes to E5M2 and uses GPU MMA.
    Fp8MatMul {
        a: i64, b: i64, out: i64,
        saved_a: i64, saved_b: i64,
        scale_a: f32, scale_b: f32,
        k_dim: i64,
        device: u8,
    },
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
    Unsqueeze { input: i64, out: i64, input_shape: Vec<i64> },
    Expand { input: i64, out: i64, original_shape: Vec<i64> },
    Stack { inputs: Vec<i64>, out: i64, dim: i64 },
    FlashAttention {
        q: i64, k: i64, v: i64,
        out: i64,
        logsumexp: i64,
        scale: f32,
        batch: i64, heads: i64, seq_len: i64, head_dim: i64,
        causal: bool,
        saved_q: i64,    // refcount-bumped for backward
        saved_k: i64,
        saved_v: i64,
    },
}

pub(crate) struct Tape {
    pub(crate) ops: Vec<TapeOp>,
    pub(crate) param_set: HashSet<i64>,
    pub(crate) recording: bool,
    pub(crate) pause_depth: i32,
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
    pub(crate) static TAPE: RefCell<Tape> = RefCell::new(Tape::new());
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
    // Pre-flight CUDA health check — surface any deferred errors before recording ops
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys::{cuCtxSynchronize, CUresult};
        unsafe {
            let result = cuCtxSynchronize();
            if result != CUresult::CUDA_SUCCESS {
                eprintln!(
                    "[nsl] WARNING: CUDA deferred error detected at tape_start: {:?}\n\
                     A prior GPU operation failed. Re-run with --cuda-sync to identify it.",
                    result
                );
            }
        }
    }
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        // Release any saved tensor refs from a previous tape session
        // (prevents refcount leaks if tape_start is called without tape_stop)
        release_tape_op_refs(&tape.ops);
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

/// Release saved tensor refcounts held by tape ops.
/// Must be called before clearing ops to prevent refcount leaks.
pub(crate) fn release_tape_op_refs(ops: &[TapeOp]) {
    for op in ops.iter() {
        match op {
            TapeOp::Mul { saved_a, saved_b, .. }
            | TapeOp::Div { saved_a, saved_b, .. }
            | TapeOp::MatMul { saved_a, saved_b, .. }
            | TapeOp::Fp8MatMul { saved_a, saved_b, .. } => {
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
            TapeOp::Unsqueeze { .. } => {}
            TapeOp::Expand { .. } => {}
            TapeOp::Stack { inputs, .. } | TapeOp::Cat { inputs, .. } => {
                for &inp in inputs {
                    tensor_free(inp);
                }
            }
            TapeOp::FlashAttention { saved_q, saved_k, saved_v, out, logsumexp, .. } => {
                tensor_free(*saved_q);
                tensor_free(*saved_k);
                tensor_free(*saved_v);
                tensor_free(*out);
                tensor_free(*logsumexp);
            }
            _ => {}
        }
    }
}

/// Stop recording and clean up saved tensor refcounts.
/// This prevents memory leaks even if backward is never called.
#[no_mangle]
pub extern "C" fn nsl_tape_stop() {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        tape.recording = false;
        tape.pause_depth = 0;

        release_tape_op_refs(&tape.ops);

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
