use std::cell::RefCell;
use std::collections::HashSet;
use std::sync::atomic::Ordering;

use crate::list::NslList;
use crate::tensor::{
    nsl_tensor_free as tensor_free,
    NslTensor,
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
    Sin { a: i64, out: i64, saved_a: i64 },
    Cos { a: i64, out: i64, saved_a: i64 },
    Sigmoid { a: i64, out: i64, saved_out: i64 },
    Tanh { a: i64, out: i64, saved_out: i64 },
    Softmax { a: i64, out: i64, saved_out: i64, dim: i64 },
    LogSoftmax { a: i64, out: i64, saved_out: i64, dim: i64 },
    Slice { a: i64, out: i64, dim: i64, start: i64, input_shape: Vec<i64> },
    Cat { inputs: Vec<i64>, out: i64, dim: i64, split_sizes: Vec<i64> },
    EmbeddingLookup { weight: i64, indices: i64, out: i64, saved_weight: i64, saved_indices: i64 },
    LayerNorm { input: i64, weight: i64, bias: i64, out: i64, saved_input: i64, saved_mean: i64, saved_inv_std: i64, saved_weight: i64 },
    RMSNorm { input: i64, weight: i64, out: i64, saved_input: i64, saved_rms: i64, saved_weight: i64 },
    Dropout { a: i64, out: i64, saved_mask: i64, scale: f64 },
    Conv2d { input: i64, weight: i64, bias: i64, out: i64, saved_input: i64, saved_weight: i64, stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize },
    MaxPool2d { a: i64, out: i64, saved_argmax: Vec<usize>, input_shape: Vec<i64> },
    RotateHalf { a: i64, out: i64 },
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
    /// Gradient checkpointing: re-run the forward function during backward
    /// instead of saving intermediate activations.
    Checkpoint {
        /// Function pointer to re-execute: extern "C" fn(i64, ...) -> i64
        fn_ptr: usize,
        /// All arguments to the function (tensor ptrs + non-tensor ptrs)
        all_args: Vec<i64>,
        /// Indices into `all_args` that are tensor pointers (need refcount management)
        tensor_arg_indices: Vec<usize>,
        /// Output tensor pointer (key for grad_map lookup in backward)
        output: i64,
    },
}

pub(crate) struct Tape {
    pub(crate) ops: Vec<TapeOp>,
    pub(crate) param_set: HashSet<i64>,
    pub(crate) recording: bool,
    pub(crate) pause_depth: i32,
    pub(crate) next_id: i64,
}

impl Tape {
    fn new() -> Self {
        Tape { ops: Vec::new(), param_set: HashSet::new(), recording: false, pause_depth: 0, next_id: 1 }
    }
    fn get_or_assign_id(&mut self, ptr: i64) -> i64 {
        if ptr == 0 { return 0; }
        let t = crate::tensor::NslTensor::from_ptr(ptr);
        if t.tape_id == 0 { t.tape_id = self.next_id; self.next_id += 1; }
        t.tape_id
    }
}

impl TapeOp {
    fn assign_ids(&mut self, tape: &mut Tape) {
        match self {
            TapeOp::Add { a, b, out, .. } | TapeOp::Sub { a, b, out, .. }
            | TapeOp::Mul { a, b, out, .. } | TapeOp::Div { a, b, out, .. }
            | TapeOp::MatMul { a, b, out, .. } | TapeOp::Fp8MatMul { a, b, out, .. } => {
                *a = tape.get_or_assign_id(*a); *b = tape.get_or_assign_id(*b); *out = tape.get_or_assign_id(*out);
            }
            TapeOp::Neg { a, out } | TapeOp::MulScalar { a, out, .. } | TapeOp::AddScalar { a, out }
            | TapeOp::Transpose { a, out, .. } | TapeOp::SumReduce { a, out, .. }
            | TapeOp::MeanReduce { a, out, .. } | TapeOp::ReduceMax { a, out, .. }
            | TapeOp::Gather { a, out, .. } | TapeOp::Exp { a, out, .. } | TapeOp::Log { a, out, .. }
            | TapeOp::Sqrt { a, out, .. } | TapeOp::Abs { a, out, .. } | TapeOp::Clamp { a, out, .. }
            | TapeOp::ReLU { a, out, .. } | TapeOp::GELU { a, out, .. } | TapeOp::SiLU { a, out, .. }
            | TapeOp::Sin { a, out, .. } | TapeOp::Cos { a, out, .. }
            | TapeOp::Sigmoid { a, out, .. } | TapeOp::Tanh { a, out, .. } | TapeOp::Softmax { a, out, .. }
            | TapeOp::LogSoftmax { a, out, .. } | TapeOp::Slice { a, out, .. } | TapeOp::Dropout { a, out, .. }
            | TapeOp::RotateHalf { a, out, .. } | TapeOp::MaxPool2d { a, out, .. } => {
                *a = tape.get_or_assign_id(*a); *out = tape.get_or_assign_id(*out);
            }
            TapeOp::EmbeddingLookup { weight, indices, out, .. } => {
                *weight = tape.get_or_assign_id(*weight); *indices = tape.get_or_assign_id(*indices); *out = tape.get_or_assign_id(*out);
            }
            TapeOp::LayerNorm { input, weight, bias, out, .. } | TapeOp::Conv2d { input, weight, bias, out, .. } => {
                *input = tape.get_or_assign_id(*input); *weight = tape.get_or_assign_id(*weight);
                *bias = tape.get_or_assign_id(*bias); *out = tape.get_or_assign_id(*out);
            }
            TapeOp::RMSNorm { input, weight, out, .. } => {
                *input = tape.get_or_assign_id(*input); *weight = tape.get_or_assign_id(*weight); *out = tape.get_or_assign_id(*out);
            }
            TapeOp::BiasAdd { tensor, bias, out } => {
                *tensor = tape.get_or_assign_id(*tensor); *bias = tape.get_or_assign_id(*bias); *out = tape.get_or_assign_id(*out);
            }
            TapeOp::Unsqueeze { input, out, .. } | TapeOp::Expand { input, out, .. } => {
                *input = tape.get_or_assign_id(*input); *out = tape.get_or_assign_id(*out);
            }
            TapeOp::Cat { inputs, out, .. } | TapeOp::Stack { inputs, out, .. } => {
                for inp in inputs.iter_mut() { *inp = tape.get_or_assign_id(*inp); }
                *out = tape.get_or_assign_id(*out);
            }
            TapeOp::FlashAttention { q, k, v, .. } => {
                *q = tape.get_or_assign_id(*q); *k = tape.get_or_assign_id(*k); *v = tape.get_or_assign_id(*v);
                // out stays raw pointer — backward reads tape_id from tensor
            }
            TapeOp::Checkpoint { output, .. } => { *output = tape.get_or_assign_id(*output); }
        }
    }
}

thread_local! {
    pub(crate) static TAPE: RefCell<Tape> = RefCell::new(Tape::new());
}

// ---------------------------------------------------------------------------
// Test-only helpers (gated on `test-hooks`)
// ---------------------------------------------------------------------------
//
// Spec B's headline-invariant contract test
// (`tests/backward_does_not_consult_live_tape.rs`) needs to inspect the
// thread-local tape's op count to verify the move-out step left the
// live tape empty. Exposing the `TAPE` thread_local directly would
// require making it `pub`, which leaks the autodiff implementation
// surface to every downstream crate. Instead, expose a narrow
// inspector function under the `test-hooks` feature.

/// Number of ops currently held in the thread-local tape. Used by
/// Spec B's headline-invariant contract test to verify forward-grad's
/// move-out step (§4.2) drains the tape.
#[cfg(feature = "test-hooks")]
pub fn test_tape_ops_len() -> usize {
    TAPE.with(|t| t.borrow().ops.len())
}

/// Drain the thread-local tape's ops into a Vec, also snapshotting the
/// current `param_set`. Mirrors the move-out step that
/// `nsl_model_forward_grad` (§4.2) performs at the end of a successful
/// forward. Used by Spec B's headline-invariant contract test to
/// construct `GradContext` instances without going through the
/// `@export` dispatch path (which currently doesn't thread weight
/// pointers through to the tape — see `c_wrapper.rs:95-97`).
///
/// After this call, `recording` is reset to false and `pause_depth` to
/// 0, matching the RAII drop guard's effect.
#[cfg(feature = "test-hooks")]
pub fn test_drain_tape_and_params() -> (Vec<TapeOp>, Vec<i64>) {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        let params: Vec<i64> = tape.param_set.iter().copied().collect();
        let ops = std::mem::take(&mut tape.ops);
        tape.recording = false;
        tape.pause_depth = 0;
        (ops, params)
    })
}

/// Test hook: directly set the thread-local `TAPE.recording` flag.
/// Spec B T5 uses this to assert the re-entry guard fires without
/// actually starting a forward.
///
/// Pairs with `is_recording()` for read-back. Does NOT touch
/// `pause_depth` or `ops` — call `test_drain_tape_and_params` first
/// (or after) if a fully-clean state is required.
#[cfg(feature = "test-hooks")]
#[no_mangle]
pub extern "C" fn test_set_recording(on: bool) {
    TAPE.with(|t| t.borrow_mut().recording = on);
}

/// Test hook: read the thread-local `TAPE.pause_depth` value
/// directly. Spec B T5's RAII guard cleanup test uses this to
/// verify the guard's Drop body reset pause_depth to 0.
#[cfg(feature = "test-hooks")]
pub fn test_tape_pause_depth() -> i32 {
    TAPE.with(|t| t.borrow().pause_depth)
}

pub fn is_recording() -> bool {
    TAPE.with(|t| { let tape = t.borrow(); tape.recording && tape.pause_depth == 0 })
}

/// Debug-only invariant check for `desc_to_nsl_tensor`'s tape_id import path.
///
/// `Tape::next_id` is monotonic for the lifetime of the thread (initialized
/// to 1 by `Tape::new`, only mutated by `get_or_assign_id`'s `+= 1`, never
/// reset — both `nsl_tape_start` and `nsl_tape_stop` have explicit
/// "Do NOT reset next_id" comments). So any tape_id observed in a desc on
/// this thread was assigned at a moment when `next_id` was at most
/// `tape_id`, and `next_id` has only grown since.
///
/// A violation of `tape_id < next_id` would mean either:
///   - a future change introduced a `next_id` reset, silently re-using ids
///     across forwards (Case A: cross-generation misattribution), or
///   - a desc was constructed on a different thread (cross-thread import
///     is not currently a supported path; `GradContext` is `Send + !Sync`
///     but its output_ptrs are constructed on the producing thread).
///
/// Either case would produce wrong-but-plausible backward grads. The
/// assertion fires loudly at the import boundary instead.
///
/// Threshold note: in the special case that the assert runs on a thread
/// whose tape has never recorded (e.g., a Python helper thread that
/// only consumes ctxs), `next_id == 1`. A `tape_id >= 1` would trigger.
/// Callers are expected to hit this only on the producing thread; the
/// `tape_id != 0` early-return in `desc_to_nsl_tensor` already covers
/// the "I'm a virgin thread reading a Python-supplied desc" case.
#[inline]
pub fn debug_assert_tape_id_in_range(tape_id: i64) {
    debug_assert!(
        tape_id != 0,
        "debug_assert_tape_id_in_range called with tape_id=0; caller must early-return on 0"
    );
    TAPE.with(|t| {
        let tape = t.borrow();
        debug_assert!(
            tape_id < tape.next_id,
            "imported desc.tape_id={} >= tape.next_id={} — Tape::next_id monotonicity invariant violated. \
             Either next_id was reset (forbidden — see autodiff/mod.rs:291,399), or the desc was \
             constructed on a different thread (cross-thread desc import is not a supported path).",
            tape_id,
            tape.next_id,
        );
    });
}

pub fn pop_last_op() {
    TAPE.with(|t| { t.borrow_mut().ops.pop(); });
}

/// Records an operation on the tape if recording is active.
/// Converts identity fields from raw pointers to tape_ids via assign_ids.
pub fn maybe_record(mut op: TapeOp) {
    if !is_recording() { return; }
    TAPE.with(|t| { let mut tape = t.borrow_mut(); op.assign_ids(&mut tape); tape.ops.push(op); });
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
        // Do NOT reset next_id — parameters retain their tape_ids across steps.
        // Resetting would cause new intermediates to collide with parameter IDs.

        let list = NslList::from_ptr(param_list);
        for i in 0..list.len as usize {
            let tensor_ptr = unsafe { *list.data.add(i) };
            tape.param_set.insert(tensor_ptr);
            // Assign tape_ids to parameters (reuses existing if already set)
            let _id = tape.get_or_assign_id(tensor_ptr);
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
            | TapeOp::Softmax { saved_out, .. }
            | TapeOp::LogSoftmax { saved_out, .. } => {
                tensor_free(*saved_out);
            }
            TapeOp::Log { saved_a, .. }
            | TapeOp::Abs { saved_a, .. }
            | TapeOp::Clamp { saved_a, .. }
            | TapeOp::ReLU { saved_a, .. }
            | TapeOp::GELU { saved_a, .. }
            | TapeOp::SiLU { saved_a, .. }
            | TapeOp::Sin { saved_a, .. }
            | TapeOp::Cos { saved_a, .. } => {
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
            TapeOp::BiasAdd { .. } => {
                // tensor/bias are tape_ids after assign_ids — nothing to free
            }
            TapeOp::Unsqueeze { .. } => {}
            TapeOp::Expand { .. } => {}
            TapeOp::Stack { .. } | TapeOp::Cat { .. } => {
                // inputs are tape_ids after assign_ids — nothing to free
            }
            TapeOp::FlashAttention { saved_q, saved_k, saved_v, out, logsumexp, .. } => {
                tensor_free(*saved_q);
                tensor_free(*saved_k);
                tensor_free(*saved_v);
                tensor_free(*out);
                tensor_free(*logsumexp);
            }
            TapeOp::Checkpoint { all_args, tensor_arg_indices, .. } => {
                for &idx in tensor_arg_indices {
                    if idx < all_args.len() {
                        tensor_free(all_args[idx]);
                    }
                }
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
        // Do NOT reset next_id — monotonic across steps to avoid collisions
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

/// Call a function pointer with a variable number of i64 arguments (up to 8).
/// Used by gradient checkpointing to re-execute the forward function during backward.
#[allow(dead_code)]
pub(crate) fn call_with_args(fn_ptr: usize, args: &[i64]) -> i64 {
    debug_assert_eq!(std::mem::size_of::<usize>(), 8, "gradient checkpointing requires 64-bit target");
    type Fn0 = extern "C" fn() -> i64;
    type Fn1 = extern "C" fn(i64) -> i64;
    type Fn2 = extern "C" fn(i64, i64) -> i64;
    type Fn3 = extern "C" fn(i64, i64, i64) -> i64;
    type Fn4 = extern "C" fn(i64, i64, i64, i64) -> i64;
    type Fn5 = extern "C" fn(i64, i64, i64, i64, i64) -> i64;
    type Fn6 = extern "C" fn(i64, i64, i64, i64, i64, i64) -> i64;
    type Fn7 = extern "C" fn(i64, i64, i64, i64, i64, i64, i64) -> i64;
    type Fn8 = extern "C" fn(i64, i64, i64, i64, i64, i64, i64, i64) -> i64;
    unsafe {
        match args.len() {
            0 => std::mem::transmute::<usize, Fn0>(fn_ptr)(),
            1 => std::mem::transmute::<usize, Fn1>(fn_ptr)(args[0]),
            2 => std::mem::transmute::<usize, Fn2>(fn_ptr)(args[0], args[1]),
            3 => std::mem::transmute::<usize, Fn3>(fn_ptr)(args[0], args[1], args[2]),
            4 => std::mem::transmute::<usize, Fn4>(fn_ptr)(args[0], args[1], args[2], args[3]),
            5 => std::mem::transmute::<usize, Fn5>(fn_ptr)(args[0], args[1], args[2], args[3], args[4]),
            6 => std::mem::transmute::<usize, Fn6>(fn_ptr)(args[0], args[1], args[2], args[3], args[4], args[5]),
            7 => std::mem::transmute::<usize, Fn7>(fn_ptr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6]),
            8 => std::mem::transmute::<usize, Fn8>(fn_ptr)(args[0], args[1], args[2], args[3], args[4], args[5], args[6], args[7]),
            n => panic!("@checkpoint: function has {n} arguments (max supported: 8)"),
        }
    }
}

/// Record a gradient checkpoint on the tape.
/// Instead of saving intermediate activations, we save the function pointer and args
/// so the forward can be re-executed during backward.
///
/// `fn_ptr`: raw function pointer (as i64)
/// `args_list`: NslList of all arguments (tensor ptrs and non-tensor i64 values)
/// `tensor_mask`: bitmask where bit i indicates args_list[i] is a tensor pointer
/// `output`: the output tensor pointer from the forward call
#[no_mangle]
pub extern "C" fn nsl_checkpoint_record(fn_ptr: i64, args_list: i64, tensor_mask: i64, output: i64) {
    if !is_recording() { return; }

    let list = NslList::from_ptr(args_list);
    let num_args = list.len as usize;

    let mut all_args = Vec::with_capacity(num_args);
    let mut tensor_arg_indices = Vec::new();

    for i in 0..num_args {
        let arg = unsafe { *list.data.add(i) };
        all_args.push(arg);
        // Check bitmask to see if this arg is a tensor
        if (tensor_mask >> i) & 1 == 1 {
            tensor_arg_indices.push(i);
            // Bump refcount on tensor args so they survive until backward
            if arg != 0 {
                NslTensor::from_ptr(arg).refcount.fetch_add(1, Ordering::SeqCst);
            }
        }
    }

    maybe_record(TapeOp::Checkpoint {
        fn_ptr: fn_ptr as usize,
        all_args,
        tensor_arg_indices,
        output,
    });
}
