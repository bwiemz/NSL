//! CFIE runtime — the *tiny* host-side surface CFIE needs.
//!
//! The compile-time CFIE passes bake KV layout, sampler control flow,
//! speculative tree structure, grammar DFA transitions, and decode
//! fusion into the compiled kernel.  What remains at runtime is:
//!
//!   * A **request ring buffer** in host memory through which the host
//!     pushes new-request descriptors.  The `ring_buffer` type itself is
//!     a genuine single-producer/single-consumer atomic ring, but the
//!     FFI surface today wraps ONE process-global instance in a `Mutex`
//!     (`ffi::GLOBAL_RING`) that the host decode loop pops from — the
//!     persistent decode kernel does not yet dequeue from it on-device.
//!     The lock-free, GPU-side-consumer design is the intended endpoint
//!     once the on-device scheduler lands.
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
//!   * A **speculative-decode driver** (Cycle 13, G15): a SECOND,
//!     draft-model binding (`nsl_cfie_bind_draft_model`) + an
//!     engine-held draft KV pool (`nsl_cfie_draft_pool_alloc`) and
//!     `nsl_cfie_speculative_generate` — per round: draft K greedily
//!     (kinds 6+7), target-verify K prob rows (kinds 2+8), ONE kind-4
//!     rejection launch, then roll both KV sides back to the emitted
//!     sequence (`engine.rs`).

pub mod bridge;
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
