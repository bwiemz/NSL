//! CFTP §4.3 RoPE-reset + Tier A segment_ids — runtime activation registry.
//!
//! Thread-local registry that the train block sets at each step body
//! (extracts `batch["segment_ids"]` and `batch["doc_starts"]` device
//! pointers) and the Cranelift CSHA call sites read inside the model's
//! compiled `forward` function.
//!
//! Solves a scope-visibility problem: the FA call site lives inside
//! `forward`, whose signature doesn't include the batch dict. The user's
//! source `m.forward(batch.input_ids)` doesn't thread segment_ids /
//! doc_starts through. Adding compiler-injected hidden params to
//! `forward` would be invasive; this registry is the decoupled
//! alternative.
//!
//! Uninitialized state returns `(0, 0)` — the spec-defined sentinel for
//! "identity path." Inference, calibration subprocesses, snapshot
//! tests, and `@flash_attention` usages outside a train block all
//! observe byte-stable behavior.
//!
//! Thread-local because:
//!   - Test isolation: parallel `cargo test` runs don't interfere.
//!   - Multi-worker inference: each thread sees its own state.
//!   - No locking overhead: `Cell::set` / `Cell::get` are non-atomic.

use std::cell::Cell;

thread_local! {
    /// `(segment_ids_ptr, doc_starts_ptr)` — raw device pointers stored
    /// by the train block per step, read by the CSHA call sites.
    static PACKING_METADATA: Cell<(i64, i64)> = const { Cell::new((0, 0)) };
}

/// Set the thread-local packing metadata. Called by the train block at
/// each step body (after `batch["input_ids"]` prefetch).
///
/// Pass `(0, 0)` to clear / disable. Must be called every step to
/// prevent stale state from leaking across steps (the train block
/// codegen handles this by always calling set, both in the has-packing
/// and no-packing branches).
#[no_mangle]
pub extern "C" fn nsl_packing_metadata_set(segment_ids_ptr: i64, doc_starts_ptr: i64) {
    PACKING_METADATA.with(|c| c.set((segment_ids_ptr, doc_starts_ptr)));
}

/// Read the thread-local `segment_ids_ptr`. Returns 0 when uninitialized
/// (no `set` has been called on this thread) — sentinel-0 takes the
/// kernel's identity-mask path.
#[no_mangle]
pub extern "C" fn nsl_packing_metadata_get_segment_ids() -> i64 {
    PACKING_METADATA.with(|c| c.get().0)
}

/// Read the thread-local `doc_starts_ptr`. Returns 0 when uninitialized
/// — sentinel-0 takes the kernel's identity-position path (no RoPE
/// reset).
#[no_mangle]
pub extern "C" fn nsl_packing_metadata_get_doc_starts() -> i64 {
    PACKING_METADATA.with(|c| c.get().1)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uninitialized_state_returns_zero() {
        // Each thread starts with (0, 0).
        std::thread::spawn(|| {
            assert_eq!(nsl_packing_metadata_get_segment_ids(), 0);
            assert_eq!(nsl_packing_metadata_get_doc_starts(), 0);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn set_roundtrips_in_same_thread() {
        std::thread::spawn(|| {
            nsl_packing_metadata_set(0xAAAA, 0xBBBB);
            assert_eq!(nsl_packing_metadata_get_segment_ids(), 0xAAAA);
            assert_eq!(nsl_packing_metadata_get_doc_starts(), 0xBBBB);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn cross_thread_isolation() {
        // Parent sets one value; child thread sees uninitialized state.
        std::thread::spawn(|| {
            nsl_packing_metadata_set(0xCAFE, 0xBABE);
            assert_eq!(nsl_packing_metadata_get_segment_ids(), 0xCAFE);

            std::thread::spawn(|| {
                // Child sees fresh thread-local — (0, 0).
                assert_eq!(nsl_packing_metadata_get_segment_ids(), 0);
                assert_eq!(nsl_packing_metadata_get_doc_starts(), 0);
            })
            .join()
            .unwrap();

            // Parent's value preserved across child join.
            assert_eq!(nsl_packing_metadata_get_segment_ids(), 0xCAFE);
            assert_eq!(nsl_packing_metadata_get_doc_starts(), 0xBABE);
        })
        .join()
        .unwrap();
    }

    #[test]
    fn set_zero_clears_state() {
        std::thread::spawn(|| {
            nsl_packing_metadata_set(0x1234, 0x5678);
            nsl_packing_metadata_set(0, 0);
            assert_eq!(nsl_packing_metadata_get_segment_ids(), 0);
            assert_eq!(nsl_packing_metadata_get_doc_starts(), 0);
        })
        .join()
        .unwrap();
    }
}
