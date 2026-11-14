//! CFIE runtime — the host-side surface the compiled inference engine
//! needs.
//!
//! The compile-time CFIE passes bake the KV layout, the sampler's
//! parameters and the decode block into kernels.  What remains at
//! runtime is:
//!
//!   * A **KV slot free-list**: the compile-time KV pass reserves one
//!     contiguous device buffer of `slot_count x per_slot_tokens`; a
//!     request holds one whole sequence slot from a LIFO free-list — no
//!     block allocator.
//!   * A **decode-loop engine**: a registry of the compiled CFIE PTX
//!     kernels that resolves every CUmodule/CUfunction once at
//!     `nsl_cfie_engine_finalize` and then launches per token through
//!     cached handles (`engine.rs`), with the generation drivers
//!     `nsl_cfie_generate` and `nsl_cfie_speculative_generate`.
//!   * A host-side grammar-transition lookup
//!     (`nsl_cfie_grammar_transition`) for tests of the compile-time
//!     grammar tables.  No generation driver steps a grammar; a serve
//!     block's `grammar:` section is refused at compile time.
//!
//! The engine serves one sequence at a time.  A request ring buffer used
//! to live here (`nsl_cfie_ring_*`): serve init sized it and nothing
//! ever pushed to it or popped from it.  It has been removed.

pub mod bridge;
pub mod engine;
pub mod ffi;
pub mod kv_slots;

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
