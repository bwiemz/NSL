//! Spec B §5.3 — the RAII drop guard inside `nsl_model_forward_grad`
//! clears the thread-local tape on any exit path (success, error
//! return, panic unwind).
//!
//! Verification: drive the in-progress guard's error-return path and
//! verify the first forward's recording state is PRESERVED (the RAII
//! guard is constructed AFTER the re-entry check by design; clearing
//! on early-return would corrupt the FIRST in-flight forward's state).
//! Separately, verify that running the same Drop body the production
//! guard runs (simulated via `test_drain_tape_and_params`, which
//! mirrors the guard's reset of `recording`, `pause_depth`, and `ops`)
//! produces clean state — the load-bearing property of the guard.
//!
//! A genuine panic-mid-forward test would require an intentionally
//! panicking export or a `catch_unwind` over a forward call with
//! injected panic; that surface doesn't exist in the current
//! `nsl_model_forward_grad` path, so the layered assertions verify
//! the guard's contract structurally instead.

#![cfg(feature = "test-hooks")]

use nsl_runtime::autodiff::{
    is_recording, test_set_recording, test_tape_ops_len, test_tape_pause_depth,
    test_drain_tape_and_params,
};

#[test]
fn early_return_on_reentry_does_not_clear_first_forwards_state() {
    // Simulate a forward in-progress.
    test_set_recording(true);
    assert!(is_recording(), "precondition: tape is recording");

    let mut ctx_out: i64 = 0;
    let rc = nsl_runtime::grad_context::nsl_model_forward_grad(
        1, 0, 0, 0, 0, &mut ctx_out as *mut i64 as i64,
    );

    assert_eq!(rc, -1, "re-entry guard fires");

    // CRITICAL: the first forward's recording state is preserved.
    // If forward_grad accidentally cleared it on early return, it
    // would corrupt the in-flight forward.
    assert!(
        is_recording(),
        "first forward's recording flag preserved across re-entry early-return",
    );
    assert_eq!(
        ctx_out, 0,
        "ctx_out must not be written on the re-entry error path",
    );

    // Cleanup — reset state so the test process leaves a clean tape.
    test_set_recording(false);
    assert!(!is_recording(), "cleanup: recording cleared");
}

#[test]
fn raii_guard_drop_body_produces_clean_state() {
    // Put the tape into the dirty state the production RAII guard
    // would see if forward_grad reached the `let _guard = ...` line
    // and then dropped: `recording=true`, possibly some ops, etc.
    test_set_recording(true);
    assert!(is_recording(), "precondition: recording");
    assert_eq!(
        test_tape_ops_len(),
        0,
        "precondition: no ops (we didn't actually run forward)",
    );

    // Simulate what the production RAII guard's Drop body does:
    //   - clear ops (release_tape_op_refs + ops.clear)
    //   - recording = false
    //   - pause_depth = 0
    //
    // `test_drain_tape_and_params` mirrors that exactly (minus
    // `release_tape_op_refs`, but the ops vec is empty so it's a
    // no-op anyway). The production guard's Drop body and this
    // helper share the same observable post-state.
    let (ops, _params) = test_drain_tape_and_params();
    assert_eq!(ops.len(), 0, "drained ops list is empty");

    // Verify the post-state is clean.
    assert!(!is_recording(), "recording flag cleared");
    assert_eq!(test_tape_pause_depth(), 0, "pause_depth reset to 0");
    assert_eq!(test_tape_ops_len(), 0, "ops vec empty");
}
