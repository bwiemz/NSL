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

pub mod ffi;
pub mod kv_slots;
pub mod ring_buffer;
