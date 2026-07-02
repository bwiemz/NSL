//! CFIE runtime — the *tiny* host-side surface CFIE needs.
//!
//! The compile-time CFIE passes bake KV layout, sampler control flow,
//! speculative tree structure, grammar DFA transitions, and decode
//! fusion into the compiled kernel.  What remains at runtime is:
//!
//!   * A **lock-free ring buffer** in CPU-pinned memory through which
//!     the host pushes new requests; the persistent decode kernel
//!     dequeues from it at the top of every decode iteration.
//!   * A **KV slot free-list**: the compile-time KV pass reserves one
//!     contiguous device buffer of `slot_count x per_slot_tokens`;
//!     the serve scheduler hands whole sequence slots out of a LIFO
//!     free-list — no block allocator.
//!   * Stubs for the DFA/grammar state-machine read path so the FFI
//!     surface is always linkable even on single-GPU builds.
//!   * A **decode-loop engine** (Cycle 6): a registry of the compiled
//!     CFIE PTX kernels that resolves every CUmodule/CUfunction once
//!     at `nsl_cfie_engine_finalize` and then launches per token
//!     through cached handles (`engine.rs`).

pub mod engine;
pub mod ffi;
pub mod kv_slots;
pub mod ring_buffer;

/// One serialization lock shared by every test module that mutates the
/// process-global CFIE state (GLOBAL_KV_SLOTS, the engine).  Separate
/// per-module locks let `cargo test --lib cfie` interleave slot-allocator
/// mutations across modules (reproduced flake: engine tests vs ffi
/// kv_slot_tests).
#[cfg(test)]
pub(crate) fn test_serial_lock() -> std::sync::MutexGuard<'static, ()> {
    static SERIAL: std::sync::OnceLock<std::sync::Mutex<()>> = std::sync::OnceLock::new();
    SERIAL
        .get_or_init(|| std::sync::Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}
