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
