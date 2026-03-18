// crates/nsl-runtime/src/backward_context.rs
//! M40b: Backward context for source-to-source AD.
//!
//! Provides saved tensor slot management for the compile-time backward pass.
//! Unlike the tape (M12), this context is created per-grad-block with a
//! compile-time-known number of slots.

use std::sync::Mutex;

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
// FFI
// ---------------------------------------------------------------------------

static BACKWARD_CTX: Mutex<Option<BackwardContext>> = Mutex::new(None);

/// Create a new backward context with the given number of saved tensor slots.
/// Returns 0 on success, -1 if already initialized.
#[no_mangle]
pub extern "C" fn nsl_backward_ctx_new(num_slots: i64) -> i64 {
    let mut guard = BACKWARD_CTX.lock().unwrap();
    if guard.is_some() {
        return -1;
    }
    *guard = Some(BackwardContext::new(num_slots as usize));
    0
}

/// Save a tensor pointer in the given slot.
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_backward_ctx_save(slot: i64, tensor_ptr: i64) -> i64 {
    let mut guard = BACKWARD_CTX.lock().unwrap();
    let ctx = guard.as_mut().expect("nsl_backward_ctx_new not called");
    ctx.save(slot as usize, tensor_ptr);
    0
}

/// Load a tensor pointer from the given slot.
/// Returns the tensor pointer, or 0 if slot is empty/invalid.
#[no_mangle]
pub extern "C" fn nsl_backward_ctx_load(slot: i64) -> i64 {
    let guard = BACKWARD_CTX.lock().unwrap();
    let ctx = guard.as_ref().expect("nsl_backward_ctx_new not called");
    ctx.load(slot as usize)
}

/// Destroy the backward context and free all saved references.
/// Returns 0 on success.
#[no_mangle]
pub extern "C" fn nsl_backward_ctx_free() -> i64 {
    let mut guard = BACKWARD_CTX.lock().unwrap();
    *guard = None;
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    static TEST_LOCK: Mutex<()> = Mutex::new(());

    fn setup() -> std::sync::MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        nsl_backward_ctx_free();
        guard
    }

    #[test]
    fn context_save_load() {
        let _lock = setup();
        assert_eq!(nsl_backward_ctx_new(4), 0);

        nsl_backward_ctx_save(0, 0x1234);
        nsl_backward_ctx_save(2, 0x5678);

        assert_eq!(nsl_backward_ctx_load(0), 0x1234);
        assert_eq!(nsl_backward_ctx_load(1), 0); // empty
        assert_eq!(nsl_backward_ctx_load(2), 0x5678);
        assert_eq!(nsl_backward_ctx_load(99), 0); // out of range

        assert_eq!(nsl_backward_ctx_free(), 0);
    }

    #[test]
    fn context_double_init_fails() {
        let _lock = setup();
        assert_eq!(nsl_backward_ctx_new(2), 0);
        assert_eq!(nsl_backward_ctx_new(2), -1);
        assert_eq!(nsl_backward_ctx_free(), 0);
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
