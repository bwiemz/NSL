//! CFIE runtime — the *tiny* host-side surface CFIE needs.
//!
//! The compile-time CFIE passes bake KV layout, sampler control flow,
//! speculative tree structure, grammar DFA transitions, and decode
//! fusion into the compiled kernel.  What remains at runtime is:
//!
//!   * A **lock-free ring buffer** in CPU-pinned memory through which
//!     the host pushes new requests; the persistent decode kernel
//!     dequeues from it at the top of every decode iteration.
//!   * A small **bump allocator** for KV blocks when the static
//!     envelope can't hold every request.
//!   * Stubs for the DFA/grammar state-machine read path so the FFI
//!     surface is always linkable even on single-GPU builds.

pub mod ffi;
pub mod ring_buffer;
