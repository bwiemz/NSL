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
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

thread_local! {
    /// `(segment_ids_ptr, doc_starts_ptr)` — raw device pointers stored
    /// by the train block per step, read by the CSHA call sites.
    static PACKING_METADATA: Cell<(i64, i64)> = const { Cell::new((0, 0)) };
}

/// Set the thread-local packing metadata. Called by the train block at
/// each step body (after `batch["input_ids"]` prefetch).
///
/// Residency (PCA Stage C): batch prep calls
/// `nsl_packed_batch_align_device` BEFORE this stash, so on GPU runs the
/// data pointers extracted from the batch dict are DEVICE pointers (the
/// dict's `attention_mask`/`segment_ids` tensors have already been moved
/// to the GPU); on CPU runs they remain host pointers.
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

static STEPS_SEEN: AtomicU32 = AtomicU32::new(0);
static SAW_SEGMENTS: AtomicBool = AtomicBool::new(false);
static WARNED: AtomicBool = AtomicBool::new(false);
const WARN_AFTER_N_STEPS: u32 = 100;

/// Called once per training step (after the packing registry is set) ONLY when
/// the module was compiled with segment masking (a masked kernel was emitted).
/// `had_segments` != 0 means this step's batch contained `segment_ids`.
///
/// Warns ONCE if NO step in the first `WARN_AFTER_N_STEPS` had segment_ids —
/// the "packing declared but the DataLoader never packs" footgun (spec §6.1).
/// Does NOT warn on a single unpacked step (mixed-batch workloads are valid).
#[no_mangle]
pub extern "C" fn nsl_pca_packing_mismatch_check(had_segments: i64) {
    if had_segments != 0 {
        SAW_SEGMENTS.store(true, Ordering::Relaxed);
    }
    let n = STEPS_SEEN.fetch_add(1, Ordering::Relaxed) + 1;
    if n >= WARN_AFTER_N_STEPS
        && !SAW_SEGMENTS.load(Ordering::Relaxed)
        && !WARNED.swap(true, Ordering::Relaxed)
    {
        eprintln!(
            "[nsl-pca] packing was declared (segment masking active) but no \
             segment_ids appeared in the first {WARN_AFTER_N_STEPS} steps — the \
             masked kernel is running UNMASKED (identity). Check that your \
             DataLoader actually packs (emits batch[\"segment_ids\"])."
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mismatch_check_warns_only_on_never_packs() {
        // Reset the process-global statics so this test is order-independent.
        STEPS_SEEN.store(0, std::sync::atomic::Ordering::Relaxed);
        SAW_SEGMENTS.store(false, std::sync::atomic::Ordering::Relaxed);
        WARNED.store(false, std::sync::atomic::Ordering::Relaxed);

        // 99 unpacked steps, then a packed step at step 100 → segments DID appear,
        // so NO warning (this models a mixed-batch workload, not never-packs).
        for _ in 0..99 {
            nsl_pca_packing_mismatch_check(0);
        }
        nsl_pca_packing_mismatch_check(1); // saw segments at step 100
        assert!(
            !WARNED.load(std::sync::atomic::Ordering::Relaxed),
            "should NOT warn when segments appeared within the first N steps"
        );
    }

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
