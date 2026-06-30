//! Spec B §4 — per-call gradient context.
//!
//! `GradContext` owns the tape ops moved out of the thread-local TAPE
//! by `nsl_model_forward_grad`. `nsl_model_backward(ctx, ...)` runs
//! `run_backward_core(ctx.ops, ...)` — never touching the live tape.
//!
//! Lifetime: created on the stack of a successful `forward_grad`,
//! Box::into_raw'd to a heap pointer returned to C; consumed by
//! `nsl_model_backward` (move-out + mark consumed); freed by
//! `nsl_grad_context_destroy`.

use crate::autodiff::TapeOp;

// ---------------------------------------------------------------------------
// Error message constants (carried over from Spec B T2 deferral)
// ---------------------------------------------------------------------------

pub(crate) const ERR_FORWARD_IN_PROGRESS: &str =
    "nsl_model_forward_grad: forward already in progress on this thread";
pub(crate) const ERR_NULL_CONTEXT: &str =
    "nsl_model_backward: null context pointer";
pub(crate) const ERR_ALREADY_CONSUMED: &str =
    "nsl_model_backward: context already consumed";
pub(crate) const ERR_INVALID_CONTEXT: &str =
    "nsl_model_backward: pointer does not point to a valid GradContext (wrong type or stale handle?)";

/// Sentinel value placed at offset 0 of every live `GradContext` to
/// distinguish a real ctx pointer from a stale handle or a misused
/// `*mut NslModel` (the historical bug the legacy Python autograd
/// path hit by calling `nsl_model_backward(model_handle, ...)`).
///
/// Chosen distinct from `tensor::TENSOR_MAGIC` (`0x4E534C54` = "NSLT").
/// Value: `0x4E534C47` = "NSLG" ("NSL Grad context").
pub(crate) const NSL_GRAD_CONTEXT_MAGIC: u32 = 0x4E534C47;
/// Sentinel written into the magic field on drop so a use-after-free
/// is caught instead of looking like a live ctx.
pub(crate) const NSL_GRAD_CONTEXT_FREED: u32 = 0xDEAD4715;

/// Convenience: route a `&'static str` error message to the thread-local
/// last-error slot using the shared `set_error_cstring` helper. Keeps the
/// FFI stubs free of repeated CString::new boilerplate.
fn set_error_static(msg: &'static str) {
    let cstr = std::ffi::CString::new(msg)
        .expect("static error message has interior NUL");
    crate::c_api::set_error_cstring(cstr);
}

/// Per-call gradient context.
///
/// **Send + !Sync** by Spec B §5.4 contract: a `*mut GradContext` may
/// be passed across threads but at any given time only one thread
/// owns it exclusively. Concurrent access is UB by user contract.
///
/// Populated by `nsl_model_forward_grad` (§4.2), consumed by
/// `nsl_model_backward` (§4.3), freed by `nsl_grad_context_destroy`
/// (§4.1). The `consumed: bool` is intentionally non-atomic per the
/// exclusive-ownership contract (§5.4).
#[repr(C)]
pub struct GradContext {
    /// Type-tag sentinel — `NSL_GRAD_CONTEXT_MAGIC` for a live ctx,
    /// `NSL_GRAD_CONTEXT_FREED` after Drop. MUST be the first field so
    /// callers can validate via a single 4-byte read at `ctx_ptr` before
    /// dereferencing the struct (mirrors the `TENSOR_MAGIC` pattern from
    /// `tensor::nsl_tensor_free_if_valid`).
    magic: u32,
    pub(crate) ops: Vec<TapeOp>,
    /// Raw NslTensor* the forward was called with. ctx-owned — these
    /// are fresh wrappers created by `nsl_model_forward_grad` over the
    /// caller's NslTensorDesc data; freed by `Drop for GradContext`.
    /// Reserved for cross-thread backward (§5.4) and future serialization
    /// (§6) — backward replays from `ops` and writes grads into the
    /// caller's desc array, so this field is currently unread.
    #[allow(dead_code)]
    pub(crate) input_ptrs: Vec<i64>,
    /// Raw NslTensor* the forward returned. ctx-owned wrappers — same
    /// lifetime as input_ptrs.
    pub(crate) output_ptrs: Vec<i64>,
    /// Raw NslTensor* of trainable parameters. Borrowed from the model;
    /// NOT freed by Drop.
    pub(crate) param_ptrs: Vec<i64>,
    consumed: bool,
}

// Send: ops vec + ptr vecs are not tied to thread-local state after the
// move-out. The raw NSL tensor pointers are the user's responsibility.
unsafe impl Send for GradContext {}

impl GradContext {
    pub fn new(
        ops: Vec<TapeOp>,
        input_ptrs: Vec<i64>,
        output_ptrs: Vec<i64>,
        param_ptrs: Vec<i64>,
    ) -> Self {
        Self {
            magic: NSL_GRAD_CONTEXT_MAGIC,
            ops,
            input_ptrs,
            output_ptrs,
            param_ptrs,
            consumed: false,
        }
    }

    pub fn consumed(&self) -> bool {
        self.consumed
    }

    /// Set the consumed flag and return the prior value.
    pub fn mark_consumed(&mut self) -> bool {
        let prior = self.consumed;
        self.consumed = true;
        prior
    }

    /// Move ops out of the context. Subsequent reads of `self.ops` see
    /// an empty Vec. Used by backward to take ownership before replaying.
    pub(crate) fn take_ops(&mut self) -> Vec<TapeOp> {
        std::mem::take(&mut self.ops)
    }
}

/// Free the ctx-owned input/output NslTensor wrappers on drop.
///
/// `input_ptrs` / `output_ptrs` are fresh wrappers created by
/// `nsl_model_forward_grad` via `desc_to_nsl_tensor` over the caller's
/// `NslTensorDesc` data buffers — they have `owns_data: 0`, so freeing
/// the wrapper releases only the wrapper's own shape/strides allocations
/// and the `Box<NslTensor>` itself; the caller's data buffer is left
/// untouched (matching the desc-borrow contract).
///
/// `param_ptrs` are intentionally NOT freed — those are the model's
/// weight tensors and the model still owns them.
impl Drop for GradContext {
    fn drop(&mut self) {
        for ptr in self.input_ptrs.drain(..) {
            if ptr != 0 {
                crate::tensor::nsl_tensor_free(ptr);
            }
        }
        for ptr in self.output_ptrs.drain(..) {
            if ptr != 0 {
                crate::tensor::nsl_tensor_free(ptr);
            }
        }
        // param_ptrs intentionally NOT freed — those are the model's weight
        // tensors, the model still owns them.
        self.magic = NSL_GRAD_CONTEXT_FREED;
    }
}

/// Spec B §4.1 — destroy a context. Idempotent on NULL. Validates the
/// magic-header sentinel before `Box::from_raw` so passing a stale
/// handle or a `*mut NslModel` (the legacy Python autograd-path bug)
/// is a silent no-op rather than heap corruption.
///
/// `Drop for GradContext` frees the ctx-owned `input_ptrs` /
/// `output_ptrs` wrappers. `param_ptrs` are borrowed from the model and
/// are NOT freed here.
#[no_mangle]
pub extern "C" fn nsl_grad_context_destroy(ctx_ptr: i64) {
    if ctx_ptr == 0 {
        return;
    }
    // Cheap sanity gates before the unsafe magic read: reject obviously
    // bogus pointers (matches `nsl_tensor_free_if_valid`'s pattern).
    if (ctx_ptr as u64) < 0x10000 {
        return;
    }
    if !(ctx_ptr as usize).is_multiple_of(std::mem::align_of::<u32>()) {
        return;
    }
    // SAFETY: read 4 bytes at `ctx_ptr`. If `ctx_ptr` is invalid this
    // is UB — but the gates above filter the common misuse paths, and
    // the magic check catches the realistic "stale or wrong-type
    // pointer" failure mode. Same risk class as `nsl_tensor_free_if_valid`.
    let magic = unsafe { *(ctx_ptr as *const u32) };
    if magic != NSL_GRAD_CONTEXT_MAGIC {
        return;
    }
    unsafe { drop(Box::from_raw(ctx_ptr as *mut GradContext)); }
}

// ---------------------------------------------------------------------------
// Spec B §4.2 — `nsl_model_forward_grad`
// ---------------------------------------------------------------------------

/// Record a forward pass into a fresh `GradContext`.
///
/// Workflow:
///   1. Re-entry guard: if thread-local TAPE.recording is already true
///      (§5.2), return -1 with the `ForwardInProgress` error.
///   2. RAII drop guard (§5.3): any exit path (success, error, panic)
///      clears the thread-local tape state.
///   3. Start recording on `model.weight_ptrs` (registers them as
///      parameters in `tape.param_set`).
///   4. Snapshot input tensor pointers from the `NslTensorDesc` array
///      so they can live inside the `GradContext` independently of
///      the caller's desc memory.
///   5. Dispatch the forward via Spec A's
///      `nsl_model_call(model, "forward", ...)`.
///   6. On success, MOVE ops out of the thread-local TAPE into the
///      heap-allocated `GradContext` and return its pointer through
///      `grad_context_out`.
///
/// The headline invariant from Spec B §2 holds because step 6 drains
/// the tape into `ctx.ops`; the RAII guard then clears whatever
/// residual state remained (it sees an empty Vec, so the only effect
/// is `recording = false`). A subsequent `forward_grad` on the same
/// thread starts with a clean tape; `nsl_model_backward(ctx_a, ...)`
/// replays from `ctx_a.ops` regardless of what the live tape holds.
#[no_mangle]
pub extern "C" fn nsl_model_forward_grad(
    model_ptr: i64,
    inputs_ptr: i64,
    num_inputs: i64,
    outputs_ptr: i64,
    num_outputs: i64,
    grad_context_out: i64, // *mut *mut GradContext
) -> i64 {
    if model_ptr == 0 || grad_context_out == 0 {
        set_error_static("nsl_model_forward_grad: null model or out pointer");
        return -1;
    }

    // §5.3 RAII drop guard — fires on success, error, AND panic unwind.
    // Always leaves the thread-local in a clean (recording=false, empty
    // ops) state. Moving ops into ctx (step 6) happens BEFORE this
    // guard drops, so the guard sees an empty Vec on the success path.
    struct TapeGuard;
    impl Drop for TapeGuard {
        fn drop(&mut self) {
            crate::autodiff::TAPE.with(|t| {
                let mut tape = t.borrow_mut();
                crate::autodiff::release_tape_op_refs(&tape.ops);
                tape.ops.clear();
                tape.recording = false;
                tape.pause_depth = 0;
            });
        }
    }

    // §5.2 within-thread re-entry guard.
    let already_recording =
        crate::autodiff::TAPE.with(|t| t.borrow().recording);
    if already_recording {
        set_error_static(ERR_FORWARD_IN_PROGRESS);
        return -1;
    }

    let _guard = TapeGuard;

    // Build the param list for tape_start. nsl_tape_start clones
    // pointers into tape.param_set; the list itself is freed below.
    let param_list = crate::list::nsl_list_new();
    let weight_ptrs = crate::c_api::nsl_model_get_weight_ptrs(model_ptr);
    let n_weights = crate::c_api::nsl_model_get_num_weights(model_ptr);
    if weight_ptrs != 0 && n_weights > 0 {
        let weights_slice = unsafe {
            std::slice::from_raw_parts(weight_ptrs as *const i64, n_weights as usize)
        };
        for &wptr in weights_slice {
            crate::list::nsl_list_push(param_list, wptr);
        }
    }
    crate::autodiff::nsl_tape_start(param_list);
    crate::list::nsl_list_free(param_list);

    // Snapshot raw `NslTensor*` for the input descs (used by backward
    // for grad-input mapping; held by ctx). Each call to
    // `desc_to_nsl_tensor` allocates a new NslTensor that borrows the
    // desc's data buffer — the borrow is safe so long as the caller
    // keeps the data buffer alive for at least the lifetime of the ctx.
    let input_ptrs: Vec<i64> = if num_inputs > 0 && inputs_ptr != 0 {
        let descs = unsafe {
            std::slice::from_raw_parts(
                inputs_ptr as *const crate::c_api::NslTensorDesc,
                num_inputs as usize,
            )
        };
        descs
            .iter()
            .map(crate::c_api::desc_to_nsl_tensor)
            .collect()
    } else {
        Vec::new()
    };

    // Dispatch via Spec A: nsl_model_call(model, "forward", ...).
    // The packed-array dispatch wrapper unpacks input/output descs and
    // invokes the typed `forward` wrapper. Records tape ops as a side
    // effect via the autodiff op surface.
    let name = c"forward";
    let rc = crate::c_api::nsl_model_call(
        model_ptr,
        name.as_ptr() as i64,
        inputs_ptr,
        num_inputs,
        outputs_ptr,
        num_outputs,
    );
    if rc != 0 {
        // Free the input wrapper tensors before the guard fires.
        for &p in &input_ptrs {
            crate::tensor::nsl_tensor_free(p);
        }
        return rc;
    }

    // Snapshot output tensor pointers from the (now-populated) output
    // descs. The first output is treated as the loss seed by backward
    // (scalar-loss convention from `nsl_tape_backward`).
    //
    // Load-bearing ordering: this runs AFTER `nsl_model_call` returns,
    // which transitively invokes the codegen-emitted dispatch wrapper.
    // The wrapper calls `nsl_tensor_to_desc_ffi(impl_tensor, scratch)`
    // (copies the impl tensor's tape_id into the scratch desc), then
    // `nsl_dispatch_apply_result(scratch, caller_output_desc)` mirrors
    // `scratch.tape_id` onto the caller's output desc. So by the time
    // `desc_to_nsl_tensor` reads `outputs_ptr` below, the desc's
    // `tape_id` field carries the impl tensor's autodiff identity, and
    // the resulting wrapper inherits it — which is what makes the loss
    // seed in `run_backward_core` match a `TapeOp::*.out` tape_id
    // instead of falling back to the raw-pointer keying that produced
    // the original empty-grads bug.
    let output_ptrs: Vec<i64> = if num_outputs > 0 && outputs_ptr != 0 {
        let descs = unsafe {
            std::slice::from_raw_parts(
                outputs_ptr as *const crate::c_api::NslTensorDesc,
                num_outputs as usize,
            )
        };
        descs
            .iter()
            .map(crate::c_api::desc_to_nsl_tensor)
            .collect()
    } else {
        Vec::new()
    };

    // Snapshot the tape's param_set into a stable Vec. We collect from
    // the live tape (rather than reusing weight_ptrs) so future hooks
    // that register additional params during recording will surface
    // through ctx.param_ptrs without changing this call site.
    let param_ptrs: Vec<i64> =
        crate::autodiff::TAPE.with(|t| t.borrow().param_set.iter().copied().collect());

    // Move ops out of the thread-local TAPE into ctx.ops. This is the
    // headline-invariant load-bearing line: from here on, ctx owns the
    // recording, and the live tape is empty.
    let ops: Vec<crate::autodiff::TapeOp> = crate::autodiff::TAPE.with(|t| {
        let mut tape = t.borrow_mut();
        std::mem::take(&mut tape.ops)
    });

    let ctx = Box::new(GradContext::new(ops, input_ptrs, output_ptrs, param_ptrs));
    let raw_ctx = Box::into_raw(ctx);
    // SAFETY: caller passed a valid `*mut *mut GradContext`. Write the
    // pointer through the out param.
    unsafe {
        *(grad_context_out as *mut *mut GradContext) = raw_ctx;
    }

    // _guard drops here, cleaning the (now-empty) tape state. The empty
    // ops Vec means release_tape_op_refs is a no-op; recording flips to
    // false; pause_depth resets to 0.
    0
}

// ---------------------------------------------------------------------------
// Spec B §4.3 — `nsl_model_backward(ctx, ...)`
// ---------------------------------------------------------------------------

/// Replay `ctx.ops` in reverse to compute gradients w.r.t. each
/// parameter pointer in `ctx.param_ptrs`.
///
/// `grad_outputs_ptr` / `num_grad_outputs` are reserved for explicit
/// upstream gradient seeds — the v1 implementation seeds with
/// `ones_like(loss)` inside `run_backward_core`, matching the scalar-
/// loss convention from the legacy `nsl_tape_backward`. The first
/// element of `ctx.output_ptrs` is used as the loss tensor.
///
/// `grad_inputs_ptr` is a caller-allocated `NslTensorDesc*` array of
/// length `num_grad_inputs`. The callee fills it with one descriptor
/// per recorded parameter, in `ctx.param_ptrs` order. If the caller's
/// buffer is shorter than `param_ptrs.len()`, only the first
/// `num_grad_inputs` descriptors are written.
///
/// On success returns 0; on null ctx or double-backward returns -1
/// with a thread-local error message (§5.5).
#[no_mangle]
pub extern "C" fn nsl_model_backward(
    ctx_ptr: i64,
    grad_outputs_ptr: i64,
    num_grad_outputs: i64,
    grad_inputs_ptr: i64,
    num_grad_inputs: i64,
) -> i64 {
    if ctx_ptr == 0 {
        set_error_static(ERR_NULL_CONTEXT);
        return -1;
    }
    // Validate the magic-header sentinel BEFORE dereferencing as
    // `&mut GradContext`. This catches the realistic legacy-Python
    // failure mode (`autograd.py` passing a `*mut NslModel` where this
    // FFI expects a `*mut GradContext`) — without it, the misused write
    // through `mark_consumed()` would corrupt whatever byte aligns to
    // `consumed`'s struct offset.
    //
    // Cheap sanity gates first (match `nsl_tensor_free_if_valid`'s
    // pattern), then the 4-byte magic read.
    if (ctx_ptr as u64) < 0x10000
        || !(ctx_ptr as usize).is_multiple_of(std::mem::align_of::<u32>())
    {
        set_error_static(ERR_INVALID_CONTEXT);
        return -1;
    }
    // SAFETY: 4-byte read at `ctx_ptr`. Same risk class as
    // `nsl_tensor_free_if_valid`; the alignment + minimum-address gates
    // above filter the common misuse paths.
    let magic = unsafe { *(ctx_ptr as *const u32) };
    if magic != NSL_GRAD_CONTEXT_MAGIC {
        set_error_static(ERR_INVALID_CONTEXT);
        return -1;
    }
    // SAFETY: ctx_ptr passed both the alignment/min-address gates and
    // the magic-sentinel check, so it points to a live `GradContext`
    // produced by `nsl_model_forward_grad`. §5.4 exclusive-ownership
    // contract guarantees no other reference exists.
    let ctx = unsafe { &mut *(ctx_ptr as *mut GradContext) };
    if ctx.mark_consumed() {
        set_error_static(ERR_ALREADY_CONSUMED);
        return -1;
    }

    // Reserved for v2: explicit upstream gradient seeding via
    // grad_outputs_ptr. v1 uses the scalar-loss seed inside
    // run_backward_core.
    let _ = (grad_outputs_ptr, num_grad_outputs);

    // Move ops out of the ctx (steals ownership before replaying).
    // The headline invariant from Spec B §2 says backward MUST NOT
    // consult the live tape — `run_backward_core` operates entirely
    // on the passed-in ops Vec (verified via the contract test in
    // `tests/run_backward_core_matches_tape_backward.rs`).
    let ops = ctx.take_ops();
    let param_ptrs = ctx.param_ptrs.clone();
    let loss_ptr = if !ctx.output_ptrs.is_empty() {
        ctx.output_ptrs[0]
    } else {
        set_error_static("nsl_model_backward: context has no recorded outputs");
        return -1;
    };

    let grads_list = crate::autodiff::run_backward_core(ops, loss_ptr, &param_ptrs);

    // Write per-param grads into the caller's NslTensorDesc array, in
    // ctx.param_ptrs order. The gradient tensors are referenced by the
    // descriptors (which borrow the underlying data); the caller owns
    // them via the desc's data pointer and is responsible for freeing.
    if grad_inputs_ptr != 0 && num_grad_inputs > 0 {
        let n = (num_grad_inputs as usize).min(param_ptrs.len());
        let out_descs = unsafe {
            std::slice::from_raw_parts_mut(
                grad_inputs_ptr as *mut crate::c_api::NslTensorDesc,
                n,
            )
        };
        for (i, desc) in out_descs.iter_mut().enumerate() {
            let g = crate::list::nsl_list_get(grads_list, i as i64);
            if g != 0 {
                crate::c_api::nsl_tensor_to_desc(g, desc);
            }
        }
    }

    crate::list::nsl_list_free(grads_list);
    0
}

#[cfg(test)]
mod abi_layout_tests {
    use super::GradContext;
    use std::mem::{align_of, offset_of};

    /// `GradContext` crosses the C ABI only as an opaque `*mut GradContext`
    /// (passed as an `i64` handle). The one field C touches directly is
    /// `magic`: callers do a 4-byte read at the pointer to distinguish a live
    /// context (`NSL_GRAD_CONTEXT_MAGIC`) from a freed/garbage one before any
    /// dereference — the same guard pattern as `TENSOR_MAGIC` in
    /// `tensor::nsl_tensor_free_if_valid`. That read is only valid while
    /// `magic` is the FIRST field. This golden test pins the invariant so a
    /// field reorder can't silently turn the validation into a wild read.
    #[test]
    fn magic_is_first_field_for_ffi_validation() {
        assert_eq!(
            offset_of!(GradContext, magic),
            0,
            "GradContext.magic must be at offset 0 so the C-side 4-byte magic \
             read validates the right bytes; moving it is a breaking ABI change",
        );
        assert!(
            align_of::<GradContext>() >= align_of::<u32>(),
            "GradContext must be at least u32-aligned so the magic read is not torn",
        );
    }
}
