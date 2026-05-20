//! Spec B T4 follow-up: magic-header validation on `nsl_model_backward`
//! and `nsl_grad_context_destroy`.
//!
//! Code-quality review flagged: the legacy Python autograd path
//! (`python/nslpy/_core.py:505`, `python/nslpy/autograd.py:77`) passes
//! a `*mut NslModel` where the new ABI expects a `*mut GradContext`.
//! Without a magic-header check, the misused write through
//! `mark_consumed()` corrupts whatever byte aligns to `consumed`'s
//! struct offset and `run_backward_core` then walks garbage `ops`.
//!
//! These tests verify that:
//!   1. `nsl_model_backward(bogus_ptr, ...)` returns -1 (ERR_INVALID_CONTEXT)
//!      and does NOT segfault / corrupt memory.
//!   2. `nsl_grad_context_destroy(bogus_ptr)` is a silent no-op
//!      (no panic, no abort, no heap corruption).
//!
//! "Bogus" pointers tested: NULL, a low non-NULL value (unmapped), an
//! aligned non-NULL value pointing to a magic-mismatched buffer
//! (simulates a stale handle or `*mut NslModel`).

use nsl_runtime::grad_context::{nsl_grad_context_destroy, nsl_model_backward};

#[test]
fn backward_rejects_null_pointer() {
    let rc = nsl_model_backward(0, 0, 0, 0, 0);
    assert_eq!(rc, -1, "backward(NULL) must return -1");
}

#[test]
fn backward_rejects_low_bogus_pointer() {
    // 0xDEADBEEF is a classic "garbage pointer" sentinel. It's above
    // the 0x10000 minimum-address gate but isn't a valid mapping in
    // any sane process layout — and even if it were mapped, the magic
    // check would reject it.
    let rc = nsl_model_backward(0xDEADBEEFu64 as i64, 0, 0, 0, 0);
    assert_eq!(
        rc, -1,
        "backward(bogus_ptr=0xDEADBEEF) must return -1, not segfault"
    );
}

#[test]
fn backward_rejects_unaligned_pointer() {
    // 0xDEADBEEF + 1 = 0xDEADBEF0 is 4-byte aligned, but 0xDEADBEEE
    // (the +2 offset relative to a u32-aligned base) is misaligned.
    // The alignment gate must reject it BEFORE the unsafe magic read.
    let unaligned: i64 = 0x10001; // > 0x10000, but not u32-aligned
    let rc = nsl_model_backward(unaligned, 0, 0, 0, 0);
    assert_eq!(rc, -1, "backward(unaligned) must return -1");
}

#[test]
fn backward_rejects_magic_mismatch() {
    // Allocate an aligned buffer with a header that ISN'T the
    // GradContext magic. Simulates the legacy Python bug: passing
    // a `*mut NslModel` (or any other valid heap allocation that
    // happens to have a non-magic byte pattern at offset 0).
    let buf: Box<[u32; 8]> = Box::new([0xCAFEBABE; 8]);
    let raw_ptr = Box::into_raw(buf);
    let rc = nsl_model_backward(raw_ptr as i64, 0, 0, 0, 0);
    assert_eq!(
        rc, -1,
        "backward(magic-mismatched buffer) must return -1, not corrupt the buffer"
    );

    // Verify the magic-mismatched buffer was NOT written to. The
    // pre-fix code would have written through `mark_consumed()` to
    // the byte at GradContext's `consumed` field offset. With the
    // fix, the buffer is untouched.
    unsafe {
        let written = &*raw_ptr;
        for &word in written.iter() {
            assert_eq!(
                word, 0xCAFEBABE,
                "buffer was modified — magic check failed to gate the write"
            );
        }
        // Reclaim the box to avoid leaking.
        drop(Box::from_raw(raw_ptr));
    }
}

#[test]
fn destroy_idempotent_on_null() {
    // Pre-existing contract: nsl_grad_context_destroy(NULL) is a no-op.
    nsl_grad_context_destroy(0);
    // Just reaching this line is the assertion.
}

#[test]
fn destroy_rejects_low_bogus_pointer() {
    // Bogus low pointer is filtered by the minimum-address gate
    // before the unsafe magic read.
    nsl_grad_context_destroy(0xDEADBEEFu64 as i64);
    // No panic / abort / segfault → test passes.
}

#[test]
fn destroy_rejects_unaligned_pointer() {
    nsl_grad_context_destroy(0x10001);
    // No panic / abort / segfault → test passes.
}

#[test]
fn destroy_rejects_magic_mismatch() {
    // Same simulated-stale-handle scenario as the backward test:
    // an aligned heap buffer with a non-magic header. destroy must
    // NOT `Box::from_raw` it (which would corrupt the heap by
    // re-interpreting the buffer as a `Box<GradContext>` and running
    // GradContext's destructor over arbitrary memory).
    let buf: Box<[u32; 8]> = Box::new([0xCAFEBABE; 8]);
    let raw_ptr = Box::into_raw(buf);

    nsl_grad_context_destroy(raw_ptr as i64);

    // If destroy had taken the Box::from_raw branch, this buffer
    // would now be freed (or corrupted) and the read below would be
    // UAF. With the magic check, the buffer is untouched.
    unsafe {
        let still_valid = &*raw_ptr;
        for &word in still_valid.iter() {
            assert_eq!(
                word, 0xCAFEBABE,
                "destroy mutated the buffer — magic check failed to gate Box::from_raw"
            );
        }
        // Reclaim the box.
        drop(Box::from_raw(raw_ptr));
    }
}
