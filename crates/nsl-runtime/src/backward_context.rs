// crates/nsl-runtime/src/backward_context.rs
//! M40b: Backward context for source-to-source AD.
//!
//! Provides saved tensor slot management for the compile-time backward pass.
//! Unlike the tape (M12), this context is created per-grad-block with a
//! compile-time-known number of slots.
//!
//! Handle-based design: each `nsl_backward_ctx_new` call returns an opaque i64
//! handle (heap pointer). Multiple contexts can exist concurrently, enabling
//! parallel gradient computation and data-parallel training.

/// Context for storing intermediate tensors needed by the backward pass.
/// Created per `grad` block with a fixed number of slots determined at compile time.
pub struct BackwardContext {
    slots: Vec<i64>,  // i64 tensor pointers (0 = empty)
    num_slots: usize,
}

impl BackwardContext {
    pub fn new(num_slots: usize) -> Self {
        BackwardContext {
            slots: vec![0; num_slots],
            num_slots,
        }
    }

    /// Save a tensor pointer in the given slot.
    pub fn save(&mut self, slot: usize, tensor_ptr: i64) {
        if slot < self.num_slots {
            self.slots[slot] = tensor_ptr;
        }
    }

    /// Load a tensor pointer from the given slot.
    pub fn load(&self, slot: usize) -> i64 {
        if slot < self.num_slots {
            self.slots[slot]
        } else {
            0
        }
    }

    /// Clear all slots (for cleanup after backward pass).
    pub fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// FFI — handle-based (no global state)
// ---------------------------------------------------------------------------

/// Create a new backward context with the given number of saved tensor slots.
/// Returns an opaque handle (i64). The handle must be passed to all subsequent
/// save/load/free calls. Multiple contexts can exist concurrently.
#[no_mangle]
pub extern "C" fn nsl_backward_ctx_new(num_slots: i64) -> i64 {
    let ctx = Box::new(BackwardContext::new(num_slots as usize));
    Box::into_raw(ctx) as i64
}

/// Save a tensor pointer in the given slot of the specified context.
/// Returns 0 on success, -1 if handle is null.
#[no_mangle]
pub extern "C" fn nsl_backward_ctx_save(ctx_handle: i64, slot: i64, tensor_ptr: i64) -> i64 {
    if ctx_handle == 0 { return -1; }
    let ctx = unsafe { &mut *(ctx_handle as *mut BackwardContext) };
    ctx.save(slot as usize, tensor_ptr);
    0
}

/// Load a tensor pointer from the given slot of the specified context.
/// Returns the tensor pointer, or 0 if slot is empty/invalid or handle is null.
#[no_mangle]
pub extern "C" fn nsl_backward_ctx_load(ctx_handle: i64, slot: i64) -> i64 {
    if ctx_handle == 0 { return 0; }
    let ctx = unsafe { &*(ctx_handle as *const BackwardContext) };
    ctx.load(slot as usize)
}

/// Destroy the backward context and free all saved references.
/// Returns 0 on success, -1 if handle is null.
#[no_mangle]
pub extern "C" fn nsl_backward_ctx_free(ctx_handle: i64) -> i64 {
    if ctx_handle == 0 { return -1; }
    unsafe { drop(Box::from_raw(ctx_handle as *mut BackwardContext)); }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn context_save_load() {
        let ctx = nsl_backward_ctx_new(4);
        assert!(ctx != 0);

        assert_eq!(nsl_backward_ctx_save(ctx, 0, 0x1234), 0);
        assert_eq!(nsl_backward_ctx_save(ctx, 2, 0x5678), 0);

        assert_eq!(nsl_backward_ctx_load(ctx, 0), 0x1234);
        assert_eq!(nsl_backward_ctx_load(ctx, 1), 0); // empty
        assert_eq!(nsl_backward_ctx_load(ctx, 2), 0x5678);
        assert_eq!(nsl_backward_ctx_load(ctx, 99), 0); // out of range

        assert_eq!(nsl_backward_ctx_free(ctx), 0);
    }

    #[test]
    fn context_multiple_concurrent() {
        let ctx1 = nsl_backward_ctx_new(4);
        assert!(ctx1 != 0);

        let ctx2 = nsl_backward_ctx_new(2);
        assert!(ctx2 != 0);
        assert_ne!(ctx1, ctx2);

        // Save to ctx1
        nsl_backward_ctx_save(ctx1, 0, 0x1111);
        nsl_backward_ctx_save(ctx1, 1, 0x2222);

        // Save to ctx2
        nsl_backward_ctx_save(ctx2, 0, 0xAAAA);

        // Load from ctx1 — should not see ctx2's data
        assert_eq!(nsl_backward_ctx_load(ctx1, 0), 0x1111);
        assert_eq!(nsl_backward_ctx_load(ctx1, 1), 0x2222);

        // Load from ctx2 — should not see ctx1's data
        assert_eq!(nsl_backward_ctx_load(ctx2, 0), 0xAAAA);
        assert_eq!(nsl_backward_ctx_load(ctx2, 1), 0); // empty

        // Free independently
        nsl_backward_ctx_free(ctx1);
        // ctx2 still works after ctx1 is freed
        assert_eq!(nsl_backward_ctx_load(ctx2, 0), 0xAAAA);
        nsl_backward_ctx_free(ctx2);
    }

    #[test]
    fn context_null_handle_safe() {
        assert_eq!(nsl_backward_ctx_save(0, 0, 42), -1);
        assert_eq!(nsl_backward_ctx_load(0, 0), 0);
        assert_eq!(nsl_backward_ctx_free(0), -1);
    }

    #[test]
    fn context_clear() {
        let mut ctx = BackwardContext::new(3);
        ctx.save(0, 42);
        ctx.save(1, 99);
        ctx.clear();
        assert_eq!(ctx.load(0), 0);
        assert_eq!(ctx.load(1), 0);
    }
}
