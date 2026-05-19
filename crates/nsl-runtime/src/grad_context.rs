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

#[allow(dead_code)]
pub(crate) const ERR_FORWARD_IN_PROGRESS: &str =
    "nsl_model_forward_grad: forward already in progress on this thread";
#[allow(dead_code)]
pub(crate) const ERR_NULL_CONTEXT: &str =
    "nsl_model_backward: null context pointer";
#[allow(dead_code)]
pub(crate) const ERR_ALREADY_CONSUMED: &str =
    "nsl_model_backward: context already consumed";

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
/// Fields are populated here but only read by T4's
/// `nsl_model_forward_grad` / `nsl_model_backward` FFI shims. The
/// allow(dead_code) is removed when T4 lands.
#[allow(dead_code)]
pub struct GradContext {
    pub(crate) ops: Vec<TapeOp>,
    pub(crate) input_ptrs: Vec<i64>,
    pub(crate) output_ptrs: Vec<i64>,
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
        Self { ops, input_ptrs, output_ptrs, param_ptrs, consumed: false }
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

/// Spec B §4.1 — destroy a context. Idempotent on NULL. Frees the
/// heap-allocated shell; the underlying tensor pointers in
/// `input_ptrs` / `output_ptrs` / `param_ptrs` are NOT freed (the
/// caller owns them).
#[no_mangle]
pub extern "C" fn nsl_grad_context_destroy(ctx_ptr: i64) {
    if ctx_ptr == 0 {
        return;
    }
    unsafe { drop(Box::from_raw(ctx_ptr as *mut GradContext)); }
}

// ---------------------------------------------------------------------------
// Spec B T3 — FFI STUBS (replaced with real implementations in T4)
// ---------------------------------------------------------------------------
//
// The stubs return -1 with a "not implemented yet" error so the headline-
// invariant contract test in `tests/backward_does_not_consult_live_tape.rs`
// can compile and FAIL at runtime (the `rc == 0` assertion fires). T4
// replaces these with the real Spec B §4.2 / §4.3 implementations.

const ERR_NOT_IMPLEMENTED_FORWARD: &str =
    "nsl_model_forward_grad: not implemented yet (Spec B T4 stub)";
const ERR_NOT_IMPLEMENTED_BACKWARD: &str =
    "nsl_model_backward: not implemented yet (Spec B T4 stub)";

/// Spec B §4.2 — STUB. Real implementation lands in T4.
#[no_mangle]
pub extern "C" fn nsl_model_forward_grad(
    _model_ptr: i64,
    _inputs_ptr: i64,
    _num_inputs: i64,
    _outputs_ptr: i64,
    _num_outputs: i64,
    _grad_context_out: i64,
) -> i64 {
    set_error_static(ERR_NOT_IMPLEMENTED_FORWARD);
    -1
}

/// Spec B §4.3 — STUB. Real implementation lands in T4.
#[no_mangle]
pub extern "C" fn nsl_model_backward(
    _ctx_ptr: i64,
    _grad_outputs_ptr: i64,
    _num_grad_outputs: i64,
    _grad_inputs_ptr: i64,
    _num_grad_inputs: i64,
) -> i64 {
    set_error_static(ERR_NOT_IMPLEMENTED_BACKWARD);
    -1
}
