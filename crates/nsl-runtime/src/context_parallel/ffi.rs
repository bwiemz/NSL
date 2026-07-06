//! M34: Context parallelism FFI.
//!
//! Historical note: prior to M34 v1 (this cycle), this file exposed 6 stub
//! `extern "C"` symbols (`nsl_cp_init`, `nsl_sequence_partition`,
//! `nsl_ring_attention`, `nsl_ring_send_recv`, `nsl_sequence_gather`,
//! `nsl_cp_destroy`) that all returned 0. v2.22 (CPDT Part III) unlinked
//! them from Cranelift-emitted code and fell `@context_parallel` through to
//! the naive attention path. M34 v1 deletes the stubs — they were dead
//! symbols with misleading names, and any future FFI shape will differ
//! (v2.20 review found the previous positional layout was wrong at every
//! slot except position 0).
//!
//! What lives here now: nothing. The correctness-verified single-node ring
//! math is exposed on the Rust side via
//! `crate::context_parallel::attention::run_ring_attention_full` and the
//! helpers in `partition`, `softmax`, `ring`, `types`. When real multi-device
//! distribution lands (NCCL send/recv, IPC handle exchange, etc.), a fresh
//! FFI shape gets designed against the runtime's finalized dispatch layer
//! and wired into `crates/nsl-codegen/src/expr/calls.rs` at the
//! `@context_parallel` branch. The right place to add distribution hooks
//! internally is around `ring::ring_next` / `ring::ring_prev` in
//! `run_ring_attention_full`'s per-pass loop — the current loop treats every
//! K/V chunk as locally resident; distribution replaces that with a real
//! send-forward-recv-backward exchange.
