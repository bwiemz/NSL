use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::sync::atomic::Ordering;

use crate::list::NslList;
use crate::tensor::{
    nsl_tensor_free as tensor_free,
    NslTensor,
};

pub mod backward;
pub mod cert_status;
pub mod grad_utils;

pub use backward::*;
pub use cert_status::{TapeCertStatus, tape_cert_inventory, tape_cert_status};

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

/// A tensor shape as a tape node stores it. Inline up to four dims — every
/// shape a training forward records ([batch, seq, heads, head_dim] is the
/// widest) — so capturing one costs no heap allocation; a 5-D+ shape spills
/// to the heap exactly as the `Vec` it replaced did. Four keeps the variant
/// two words wider than a `Vec`, which leaves the `TapeOp` enum the same size
/// (the `Fp8MatMul`/`FlashAttention` variants set it); six would grow every
/// node. Derefs to `&[i64]`, which is all the backward consumers take.
pub type TapeShape = smallvec::SmallVec<[i64; 4]>;

/// Capture `t`'s shape for a tape node — a copy of `shape[..ndim]` into
/// inline storage, no heap traffic for ≤ 4 dims.
pub(crate) fn tape_shape(t: &NslTensor) -> TapeShape {
    (0..t.ndim as usize).map(|i| unsafe { *t.shape.add(i) }).collect()
}

/// Operations recorded on the tape during forward passes inside `grad` blocks.
#[derive(Clone)]
pub enum TapeOp {
    Add { a: i64, b: i64, out: i64, a_shape: TapeShape, b_shape: TapeShape },
    Sub { a: i64, b: i64, out: i64, a_shape: TapeShape, b_shape: TapeShape },
    Mul { a: i64, b: i64, out: i64, saved_a: i64, saved_b: i64, a_shape: TapeShape, b_shape: TapeShape },
    Div { a: i64, b: i64, out: i64, saved_a: i64, saved_b: i64, a_shape: TapeShape, b_shape: TapeShape },
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
    /// `out = a.to(dtype)` (`nsl_tensor_to_dtype`). The gradient is `out`'s,
    /// converted back to `a`'s dtype.
    Cast { a: i64, out: i64, src_dtype: u16 },
    MulScalar { a: i64, scalar: f64, out: i64 },
    AddScalar { a: i64, out: i64 },
    Transpose { a: i64, out: i64, dim0: i64, dim1: i64 },
    SumReduce { a: i64, out: i64, dim: i64, keepdim: bool, input_shape: TapeShape },
    MeanReduce { a: i64, out: i64, dim: i64, keepdim: bool, num_elements: i64, input_shape: TapeShape },
    ReduceMax { a: i64, out: i64, dim: i64, keepdim: bool, saved_argmax: Vec<usize>, input_shape: TapeShape },
    Gather { a: i64, out: i64, dim: i64, indices_ptr: i64, input_shape: TapeShape },
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
    Slice { a: i64, out: i64, dim: i64, start: i64, input_shape: TapeShape },
    /// Metadata-only relabel: backward reshapes the gradient back to the
    /// input's shape. Without this op every reshape between two recorded ops
    /// silently DISCONNECTED the tape (views get a fresh tape_id), which is
    /// how the Q/K/V-projection -> attention chain lost all its param grads.
    Reshape { a: i64, out: i64, input_shape: TapeShape },
    Cat { inputs: Vec<i64>, out: i64, dim: i64, split_sizes: Vec<i64> },
    EmbeddingLookup { weight: i64, indices: i64, out: i64, saved_weight: i64, saved_indices: i64 },
    LayerNorm { input: i64, weight: i64, bias: i64, out: i64, saved_input: i64, saved_mean: i64, saved_inv_std: i64, saved_weight: i64 },
    RMSNorm { input: i64, weight: i64, out: i64, saved_input: i64, saved_rms: i64, saved_weight: i64 },
    Dropout { a: i64, out: i64, saved_mask: i64, scale: f64 },
    Conv2d { input: i64, weight: i64, bias: i64, out: i64, saved_input: i64, saved_weight: i64, stride_h: usize, stride_w: usize, pad_h: usize, pad_w: usize },
    MaxPool2d { a: i64, out: i64, saved_argmax: Vec<usize>, input_shape: TapeShape },
    RotateHalf { a: i64, out: i64 },
    BiasAdd { tensor: i64, bias: i64, out: i64 },
    Unsqueeze { input: i64, out: i64, input_shape: TapeShape },
    Expand { input: i64, out: i64, original_shape: TapeShape },
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
    /// An in-place write into a tensor: `y[i, j] = v` and its compound forms
    /// (`nsl_tensor_set`, `nsl_tensor_set_compound`, `nsl_tensor_set_element`)
    /// and `nsl_tensor_slice_assign`. The written tensor keeps its storage
    /// but takes a new tape id, `out`; ops recorded before the write keep the
    /// old one, `a` (see [`record_index_write`]). Backward: the gradient of
    /// `a` is `out`'s with the elements of `region` multiplied by `keep`, the
    /// derivative of the stored value with respect to the element it
    /// replaced (0 for `=`, so the written elements get none; 1 for `+=` and
    /// `-=`; `v` for `*= v`; `1/v` for `/= v`), and `src`, when it is a
    /// tensor on the tape, gets `out`'s gradient over `region`. Fields are
    /// tape ids already: nothing is saved.
    IndexWrite {
        a: i64,
        out: i64,
        /// Per dimension of `a`, the written range `[start, end)`.
        region: Vec<(i64, i64)>,
        keep: f64,
        /// Tape id of the tensor whose elements were written, in row-major
        /// order of `region`; 0 for a scalar value.
        src: i64,
        src_shape: Vec<i64>,
        src_dtype: u16,
        src_device: u8,
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
    /// Transient frees SKIPPED during recording (`nsl_tensor_free_transient`
    /// is a lifetime no-op while the tape holds bare pointers), deferred to
    /// `nsl_tape_stop`. Without this, every skipped transient — one score
    /// matrix and one softmax per attention layer per forward, per the
    /// free_transient doc — leaked PERMANENTLY in tape-mode training:
    /// ~784 MB/micro-batch at Coder-50M, OOM inside 6 micro-batches on a
    /// 32 GB card. Each entry owns exactly the one refcount decrement the
    /// skipped call would have performed; the address cannot be reused
    /// while an entry is pending because the deferral is what keeps the
    /// allocation alive.
    pub(crate) deferred_transients: Vec<i64>,
    /// Storage written in place during this recording, by the tensor that
    /// owns it (a view's `data_owner`, else the tensor itself), mapped to the
    /// tape id the write gave the written tensor. A tensor sharing that
    /// storage whose id is older stands in the graph for the values before
    /// the write; [`Tape::get_or_assign_id`] refuses to record it again.
    pub(crate) written_storage: HashMap<i64, i64>,
}

/// The tensor that owns `ptr`'s data buffer: its `data_owner` when it is a
/// view (`new_view_i64` points every view at the root owner), else itself.
fn storage_owner(ptr: i64) -> i64 {
    let t = crate::tensor::NslTensor::from_ptr_ref(ptr);
    if t.data_owner != 0 { t.data_owner } else { ptr }
}

impl Tape {
    fn new() -> Self {
        Tape {
            ops: Vec::new(),
            param_set: HashSet::new(),
            recording: false,
            pause_depth: 0,
            next_id: 1,
            deferred_transients: Vec::new(),
            written_storage: HashMap::new(),
        }
    }
    fn get_or_assign_id(&mut self, ptr: i64) -> i64 {
        if ptr == 0 { return 0; }
        if !self.written_storage.is_empty() {
            self.refuse_stale_alias(ptr);
        }
        let t = crate::tensor::NslTensor::from_ptr(ptr);
        if t.tape_id == 0 { t.tape_id = self.next_id; self.next_id += 1; }
        t.tape_id
    }

    /// Abort when `ptr` shares storage with a tensor written in place since
    /// `ptr` got its tape id. Its id then names the values before the write,
    /// so an op recorded on it would send the gradient of the written
    /// elements to their old producers: `v = y.reshape(...)`, `y[0, 0] = 0`,
    /// then a loss on `v`. (PyTorch rebases such a view onto the write; the
    /// tape has no view relation to rebase along.) Ids are monotonic, so
    /// "older" is "smaller".
    fn refuse_stale_alias(&self, ptr: i64) {
        let t = crate::tensor::NslTensor::from_ptr_ref(ptr);
        if t.tape_id == 0 {
            return;
        }
        let Some(&written_as) = self.written_storage.get(&storage_owner(ptr)) else {
            return;
        };
        if t.tape_id < written_as {
            crate::nsl_log!(ERROR, "tape-ad",
                "nsl: a tensor that shares its storage with one written in place (`y[i, j] = v`) \
                 earlier in this grad/train block is used after the write. The two are views of \
                 one buffer (reshape, transpose, unsqueeze, expand), and the gradient of the one \
                 not written would ignore the write. Write into a copy instead \
                 (`let y = x.clone()`), or take the view after the write."
            );
            std::process::abort();
        }
    }
}

impl TapeOp {
    /// The variant's name as declared. Exhaustive with no wildcard, so a new
    /// variant does not compile without one; the record trace prints it, the
    /// certification inventory ([`cert_status`]) keys on it, and its unit test
    /// holds the names to the enum's text.
    pub fn variant_name(&self) -> &'static str {
        match self {
            TapeOp::Add { .. } => "Add",
            TapeOp::Sub { .. } => "Sub",
            TapeOp::Mul { .. } => "Mul",
            TapeOp::Div { .. } => "Div",
            TapeOp::MatMul { .. } => "MatMul",
            TapeOp::Fp8MatMul { .. } => "Fp8MatMul",
            TapeOp::Neg { .. } => "Neg",
            TapeOp::Cast { .. } => "Cast",
            TapeOp::MulScalar { .. } => "MulScalar",
            TapeOp::AddScalar { .. } => "AddScalar",
            TapeOp::Transpose { .. } => "Transpose",
            TapeOp::SumReduce { .. } => "SumReduce",
            TapeOp::MeanReduce { .. } => "MeanReduce",
            TapeOp::ReduceMax { .. } => "ReduceMax",
            TapeOp::Gather { .. } => "Gather",
            TapeOp::Exp { .. } => "Exp",
            TapeOp::Log { .. } => "Log",
            TapeOp::Sqrt { .. } => "Sqrt",
            TapeOp::Abs { .. } => "Abs",
            TapeOp::Clamp { .. } => "Clamp",
            TapeOp::ReLU { .. } => "ReLU",
            TapeOp::GELU { .. } => "GELU",
            TapeOp::SiLU { .. } => "SiLU",
            TapeOp::Sin { .. } => "Sin",
            TapeOp::Cos { .. } => "Cos",
            TapeOp::Sigmoid { .. } => "Sigmoid",
            TapeOp::Tanh { .. } => "Tanh",
            TapeOp::Softmax { .. } => "Softmax",
            TapeOp::LogSoftmax { .. } => "LogSoftmax",
            TapeOp::Slice { .. } => "Slice",
            TapeOp::Reshape { .. } => "Reshape",
            TapeOp::Cat { .. } => "Cat",
            TapeOp::EmbeddingLookup { .. } => "EmbeddingLookup",
            TapeOp::LayerNorm { .. } => "LayerNorm",
            TapeOp::RMSNorm { .. } => "RMSNorm",
            TapeOp::Dropout { .. } => "Dropout",
            TapeOp::Conv2d { .. } => "Conv2d",
            TapeOp::MaxPool2d { .. } => "MaxPool2d",
            TapeOp::RotateHalf { .. } => "RotateHalf",
            TapeOp::BiasAdd { .. } => "BiasAdd",
            TapeOp::Unsqueeze { .. } => "Unsqueeze",
            TapeOp::Expand { .. } => "Expand",
            TapeOp::Stack { .. } => "Stack",
            TapeOp::FlashAttention { .. } => "FlashAttention",
            TapeOp::IndexWrite { .. } => "IndexWrite",
            TapeOp::Checkpoint { .. } => "Checkpoint",
        }
    }

    fn assign_ids(&mut self, tape: &mut Tape) {
        match self {
            TapeOp::Add { a, b, out, .. } | TapeOp::Sub { a, b, out, .. }
            | TapeOp::Mul { a, b, out, .. } | TapeOp::Div { a, b, out, .. }
            | TapeOp::MatMul { a, b, out, .. } | TapeOp::Fp8MatMul { a, b, out, .. } => {
                *a = tape.get_or_assign_id(*a); *b = tape.get_or_assign_id(*b); *out = tape.get_or_assign_id(*out);
            }
            TapeOp::Neg { a, out } | TapeOp::Cast { a, out, .. } | TapeOp::MulScalar { a, out, .. } | TapeOp::AddScalar { a, out }
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
            TapeOp::Reshape { a, out, .. } => {
                *a = tape.get_or_assign_id(*a); *out = tape.get_or_assign_id(*out);
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
            // Built with ids by `record_index_write`, never through
            // `maybe_record`: its `a` is the id the tensor had BEFORE the
            // write, which a pointer could not name.
            TapeOp::IndexWrite { .. } => {}
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
#[unsafe(no_mangle)]
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

/// Record a transient free that `nsl_tensor_free_transient` skipped because
/// the tape was recording. The deferred decrement lands in `nsl_tape_stop`
/// (or is swept as stale by the next `nsl_tape_start`).
pub(crate) fn defer_transient_free(ptr: i64) {
    if ptr == 0 {
        return;
    }
    TAPE.with(|t| t.borrow_mut().deferred_transients.push(ptr));
}

/// Debug-only invariant check for `desc_to_nsl_tensor`'s tape_id import path.
///
/// # Same-thread precondition (load-bearing)
///
/// This function is **only sound when called on the same thread that wrote
/// the desc**. Currently enforced by call-site discipline:
///   - `desc_to_nsl_tensor` is invoked from `nsl_model_forward_grad`
///     (grad_context.rs:283, 322) and from the codegen-emitted typed
///     wrappers. Both run on the thread that just executed the forward
///     pass, which is the thread whose `TAPE.next_id` advanced past
///     `desc.tape_id`. `nsl_model_backward(ctx)` never calls
///     `desc_to_nsl_tensor` — it consumes `ctx.output_ptrs` directly.
///   - `GradContext` is `Send + !Sync`, so it can be moved across threads
///     after construction, but the desc round-trip has already happened
///     before the move.
///
/// **If a future caller invokes `desc_to_nsl_tensor` from a different
/// thread than the producer**, this assert will fire (in debug) because
/// the consumer thread's `next_id` starts at 1 and any non-zero
/// `desc.tape_id` would trip `tape_id < next_id`. In release builds the
/// assertion is elided, so the consumer would silently misattribute the
/// loss seed — same failure class as the original bug. Audit any new
/// `desc_to_nsl_tensor` call site for this precondition.
///
/// # Monotonicity invariant
///
/// `Tape::next_id` is monotonic for the lifetime of the thread (initialized
/// to 1 by `Tape::new`, only mutated by `get_or_assign_id`'s `+= 1`, never
/// reset — both `nsl_tape_start` and `nsl_tape_stop` have explicit
/// "Do NOT reset next_id" comments). So any tape_id observed in a desc on
/// the producing thread was assigned at a moment when `next_id` was at most
/// `tape_id`, and `next_id` has only grown since.
///
/// A violation of `tape_id < next_id` (on a same-thread call) would mean
/// a future change introduced a `next_id` reset, silently re-using ids
/// across forwards. The assertion fires loudly at the import boundary
/// instead of producing wrong-but-plausible backward grads.
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

/// Leak-hunt trace (`NSL_DEBUG_MEM_TRACE=1`): one-line summary of a tape op's
/// identity ids and saved (refcount-bumped) pointers. Every line starts with
/// the op's [`TapeOp::variant_name`] (a unit test pins it):
/// `crates/nsl-cli/tests/source_ad_rule_cert.rs` reads
/// `[tape-trace] record <Variant>` to learn which ops a certificate recorded.
fn tape_op_trace(op: &TapeOp) -> String {
    match op {
        TapeOp::Add { a, b, out, .. } => format!("Add a={a} b={b} out={out}"),
        TapeOp::Sub { a, b, out, .. } => format!("Sub a={a} b={b} out={out}"),
        TapeOp::Mul { a, b, out, saved_a, saved_b, .. } => {
            format!("Mul a={a} b={b} out={out} saved_a={saved_a:#x} saved_b={saved_b:#x}")
        }
        TapeOp::Div { a, b, out, saved_a, saved_b, .. } => {
            format!("Div a={a} b={b} out={out} saved_a={saved_a:#x} saved_b={saved_b:#x}")
        }
        TapeOp::MatMul { a, b, out, saved_a, saved_b } => {
            format!("MatMul a={a} b={b} out={out} saved_a={saved_a:#x} saved_b={saved_b:#x}")
        }
        TapeOp::SiLU { a, out, saved_a } => format!("SiLU a={a} out={out} saved_a={saved_a:#x}"),
        TapeOp::MulScalar { a, out, scalar } => format!("MulScalar a={a} out={out} s={scalar}"),
        TapeOp::AddScalar { a, out } => format!("AddScalar a={a} out={out}"),
        TapeOp::Neg { a, out } => format!("Neg a={a} out={out}"),
        TapeOp::Cast { a, out, src_dtype } => format!("Cast a={a} out={out} src_dtype={src_dtype}"),
        TapeOp::SumReduce { a, out, dim, .. } => format!("SumReduce a={a} out={out} dim={dim}"),
        TapeOp::MeanReduce { a, out, dim, .. } => format!("MeanReduce a={a} out={out} dim={dim}"),
        TapeOp::Transpose { a, out, .. } => format!("Transpose a={a} out={out}"),
        TapeOp::Reshape { a, out, .. } => format!("Reshape a={a} out={out}"),
        TapeOp::IndexWrite { a, out, keep, src, .. } => {
            format!("IndexWrite a={a} out={out} keep={keep} src={src}")
        }
        _ => op.variant_name().to_string(),
    }
}

/// RAII guard: pause tape recording for a scope. Used by GPU wrappers that
/// redirect through the CPU implementation of the same op — without the pause
/// the inner CPU call records a node whose ids are short-lived transfer temps
/// (a garbage edge that never connects to the live graph); the wrapper records
/// the real op at its own level against the caller's tensors instead.
pub(crate) struct TapePause;
impl TapePause {
    pub(crate) fn new() -> Self {
        TAPE.with(|t| t.borrow_mut().pause_depth += 1);
        TapePause
    }
}
impl Drop for TapePause {
    fn drop(&mut self) {
        TAPE.with(|t| t.borrow_mut().pause_depth -= 1);
    }
}

/// Records an operation on the tape if recording is active.
/// Converts identity fields from raw pointers to tape_ids via assign_ids.
pub fn maybe_record(mut op: TapeOp) {
    if !is_recording() { return; }
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        op.assign_ids(&mut tape);
        if crate::tensor::tensor_trace_on() {
            crate::nsl_log!(INFO, "tape-trace", "[tape-trace] record {}", tape_op_trace(&op));
        }
        tape.ops.push(op);
    });
}

/// Start recording operations on the tape.
/// `param_list` is an i64 pointer to an NslList of tensor pointers that are parameters.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tape_start(param_list: i64) {
    // Pre-flight CUDA health check — surface any deferred errors before recording ops
    #[cfg(feature = "cuda")]
    {
        use cudarc::driver::sys::{cuCtxSynchronize, CUresult};
        unsafe {
            let result = cuCtxSynchronize();
            // "No context" is not a deferred error: it is a program that
            // has not touched the GPU. A runtime built with `cuda` printed
            // the warning below at every `grad` block of every CPU run.
            let no_context = matches!(
                result,
                CUresult::CUDA_ERROR_NOT_INITIALIZED | CUresult::CUDA_ERROR_INVALID_CONTEXT
            );
            if result != CUresult::CUDA_SUCCESS && !no_context {
                crate::nsl_log!(WARN, "nsl", 
                    "[nsl] WARNING: CUDA deferred error detected at tape_start: {:?}\n\
                     A prior GPU operation failed. Re-run with --cuda-sync to identify it.",
                    result
                );
            }
        }
    }
    let stale_deferred = TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        // Release any saved tensor refs from a previous tape session
        // (prevents refcount leaks if tape_start is called without tape_stop)
        release_tape_op_refs(&tape.ops);
        tape.ops.clear();
        tape.param_set.clear();
        tape.written_storage.clear();
        std::mem::take(&mut tape.deferred_transients)
    });
    // Stale deferred transients from an unstopped session: free them outside
    // the borrow and BEFORE recording turns on, so the frees take the normal
    // path (see nsl_tape_stop for the re-entrancy note).
    for ptr in stale_deferred {
        crate::tensor::nsl_tensor_free_if_valid(ptr);
    }
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
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

/// Call `f` on every tensor reference `op` holds: the values its backward
/// reads, each kept alive by a refcount the op owns. The one list both
/// [`release_tape_op_refs`] (which drops them) and [`record_index_write`]
/// (which refuses to overwrite one) read. A field the backward has already
/// consumed is 0.
fn for_each_saved_ref(op: &TapeOp, mut f: impl FnMut(i64)) {
    match op {
        TapeOp::Mul { saved_a, saved_b, .. }
        | TapeOp::Div { saved_a, saved_b, .. }
        | TapeOp::MatMul { saved_a, saved_b, .. }
        | TapeOp::Fp8MatMul { saved_a, saved_b, .. } => {
            f(*saved_a);
            f(*saved_b);
        }
        TapeOp::Exp { saved_out, .. }
        | TapeOp::Sqrt { saved_out, .. }
        | TapeOp::Sigmoid { saved_out, .. }
        | TapeOp::Tanh { saved_out, .. }
        | TapeOp::Softmax { saved_out, .. }
        | TapeOp::LogSoftmax { saved_out, .. } => {
            f(*saved_out);
        }
        TapeOp::Log { saved_a, .. }
        | TapeOp::Abs { saved_a, .. }
        | TapeOp::Clamp { saved_a, .. }
        | TapeOp::ReLU { saved_a, .. }
        | TapeOp::GELU { saved_a, .. }
        | TapeOp::SiLU { saved_a, .. }
        | TapeOp::Sin { saved_a, .. }
        | TapeOp::Cos { saved_a, .. } => {
            f(*saved_a);
        }
        TapeOp::Gather { indices_ptr, .. } => {
            f(*indices_ptr);
        }
        TapeOp::EmbeddingLookup { saved_weight, saved_indices, .. } => {
            f(*saved_weight);
            f(*saved_indices);
        }
        TapeOp::LayerNorm { saved_input, saved_mean, saved_inv_std, saved_weight, .. } => {
            f(*saved_input);
            f(*saved_mean);
            f(*saved_inv_std);
            f(*saved_weight);
        }
        TapeOp::RMSNorm { saved_input, saved_rms, saved_weight, .. } => {
            f(*saved_input);
            f(*saved_rms);
            f(*saved_weight);
        }
        TapeOp::Dropout { saved_mask, .. } => {
            f(*saved_mask);
        }
        TapeOp::Conv2d { saved_input, saved_weight, .. } => {
            f(*saved_input);
            f(*saved_weight);
        }
        TapeOp::FlashAttention { saved_q, saved_k, saved_v, out, logsumexp, .. } => {
            f(*saved_q);
            f(*saved_k);
            f(*saved_v);
            f(*out);
            f(*logsumexp);
        }
        TapeOp::Checkpoint { all_args, tensor_arg_indices, .. } => {
            for &idx in tensor_arg_indices {
                if idx < all_args.len() {
                    f(all_args[idx]);
                }
            }
        }
        // Identity-only fields (tape ids after assign_ids): nothing saved.
        TapeOp::Add { .. }
        | TapeOp::Sub { .. }
        | TapeOp::Neg { .. }
        | TapeOp::Cast { .. }
        | TapeOp::MulScalar { .. }
        | TapeOp::AddScalar { .. }
        | TapeOp::Transpose { .. }
        | TapeOp::SumReduce { .. }
        | TapeOp::MeanReduce { .. }
        | TapeOp::ReduceMax { .. }
        | TapeOp::Slice { .. }
        | TapeOp::Reshape { .. }
        | TapeOp::Cat { .. }
        | TapeOp::MaxPool2d { .. }
        | TapeOp::RotateHalf { .. }
        | TapeOp::BiasAdd { .. }
        | TapeOp::Unsqueeze { .. }
        | TapeOp::Expand { .. }
        | TapeOp::Stack { .. }
        | TapeOp::IndexWrite { .. } => {}
    }
}

/// Release saved tensor refcounts held by tape ops.
/// Must be called before clearing ops to prevent refcount leaks.
pub(crate) fn release_tape_op_refs(ops: &[TapeOp]) {
    for op in ops.iter() {
        if crate::tensor::tensor_trace_on() {
            crate::nsl_log!(INFO, "tape-trace", "[tape-trace] release {}", tape_op_trace(op));
        }
        for_each_saved_ref(op, |p| tensor_free(p));
    }
}

/// Record an in-place write into `target_ptr` on the tape, BEFORE the
/// caller stores anything: `region` is the written range per dimension of
/// the target, `keep` the derivative of each stored value with respect to
/// the element it replaces (see [`TapeOp::IndexWrite`]), `src_ptr` the
/// tensor whose elements are stored (0 for a scalar). A no-op while the
/// tape is not recording.
///
/// The write mutates storage the tape may already depend on, so it is
/// refused (the process aborts with the reason) when an op recorded earlier
/// holds a reference into the same storage for its backward: that backward
/// would read the written values, as PyTorch refuses ("one of the variables
/// needed for gradient computation has been modified by an inplace
/// operation"). This covers `+=` as well.
///
/// Otherwise, unless the write leaves every gradient unchanged (`keep` 1
/// and a scalar value), the target takes a fresh tape id and an
/// [`TapeOp::IndexWrite`] links it to the old one. Ops recorded before the
/// write keep the old id, so the gradient they send back skips the write,
/// and ops after it use the new one, whose gradient the write's backward
/// passes on with the overwritten elements zeroed. Other views of the same
/// storage keep the old id; using one later is refused by
/// [`Tape::refuse_stale_alias`].
pub(crate) fn record_index_write(target_ptr: i64, region: &[(usize, usize)], keep: f64, src_ptr: i64) {
    if !is_recording() {
        return;
    }
    let owner = storage_owner(target_ptr);
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        for op in &tape.ops {
            let mut hit = false;
            for_each_saved_ref(op, |p| hit |= p != 0 && storage_owner(p) == owner);
            if hit {
                crate::nsl_log!(ERROR, "tape-ad",
                    "nsl: an in-place write (`y[i, j] = v`) inside a grad/train block overwrites a \
                     tensor that an earlier `{}` saved for the backward pass; the gradient would be \
                     computed from the overwritten values. Write into a copy instead \
                     (`let y = x.clone()`).",
                    op.variant_name()
                );
                std::process::abort();
            }
        }
        let src_id = if src_ptr != 0 && NslTensor::from_ptr_ref(src_ptr).tape_id != 0 {
            tape.get_or_assign_id(src_ptr)
        } else {
            0
        };
        // A tensor the tape has never seen has no gradient path into it, and
        // a `+=`/`-=` of a scalar passes every gradient through unchanged:
        // neither needs a new version.
        let seen = NslTensor::from_ptr_ref(target_ptr).tape_id != 0;
        if src_id == 0 && (!seen || keep == 1.0) {
            return;
        }
        let a = tape.get_or_assign_id(target_ptr);
        let out = tape.next_id;
        tape.next_id += 1;
        NslTensor::from_ptr(target_ptr).tape_id = out;
        tape.written_storage.insert(owner, out);
        let (src_shape, src_dtype, src_device) = if src_id != 0 {
            let s = NslTensor::from_ptr_ref(src_ptr);
            (tape_shape(s).to_vec(), s.dtype, s.device)
        } else {
            (Vec::new(), 0, 0)
        };
        let op = TapeOp::IndexWrite {
            a,
            out,
            region: region.iter().map(|&(s, e)| (s as i64, e as i64)).collect(),
            keep,
            src: src_id,
            src_shape,
            src_dtype,
            src_device,
        };
        if crate::tensor::tensor_trace_on() {
            crate::nsl_log!(INFO, "tape-trace", "[tape-trace] record {}", tape_op_trace(&op));
        }
        tape.ops.push(op);
    });
}

/// Stop recording and clean up saved tensor refcounts.
/// This prevents memory leaks even if backward is never called.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tape_stop() {
    // Two phases: release refs/collect the deferred list under the borrow,
    // then free the deferred transients AFTER dropping it — a free can
    // re-enter tape code (is_recording checks in the tensor layer), and
    // recording is already false so re-entrant frees take the normal path.
    let deferred = TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        tape.recording = false;
        tape.pause_depth = 0;

        release_tape_op_refs(&tape.ops);

        tape.ops.clear();
        tape.param_set.clear();
        tape.written_storage.clear();
        // Do NOT reset next_id — monotonic across steps to avoid collisions
        std::mem::take(&mut tape.deferred_transients)
    });
    // The backward has consumed the tape (train epilogues call stop right
    // after it), so the skipped transient frees can now land. Refcounted:
    // any live alias (a view's data_owner, a returned grad) keeps storage.
    for ptr in deferred {
        crate::tensor::nsl_tensor_free_if_valid(ptr);
    }
}

/// Pause recording (used by @no_grad). Increments pause depth.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tape_pause() {
    TAPE.with(|t| {
        t.borrow_mut().pause_depth += 1;
    });
}

/// Resume recording after a pause. Decrements pause depth.
#[unsafe(no_mangle)]
pub extern "C" fn nsl_tape_resume() {
    TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        if tape.pause_depth > 0 {
            tape.pause_depth -= 1;
        }
    });
}

/// Record a gradient checkpoint on the tape.
/// Instead of saving intermediate activations, we save the function pointer and args
/// so the forward can be re-executed during backward.
///
/// `fn_ptr`: raw function pointer (as i64)
/// `args_list`: NslList of all arguments (tensor ptrs and non-tensor i64 values)
/// `tensor_mask`: bitmask where bit i indicates args_list[i] is a tensor pointer
/// `output`: the output tensor pointer from the forward call
#[unsafe(no_mangle)]
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
