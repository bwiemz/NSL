//! CPDT runtime support.
//!
//! The CPDT compiler passes bake their communication schedule, per-
//! parameter precision, and expert-placement decisions directly into
//! the compiled Cranelift / PTX artifact.  Most of what the paper
//! describes as "runtime work" is actually compile-time work in NSL.
//!
//! What *does* run at runtime is a handful of C-ABI collective wrappers
//! that the compiled code calls when executing the scheduled
//! allgather / reducescatter / all-to-all ops.  Single-GPU builds
//! provide no-op stubs so the FFI surface is always linkable; multi-
//! GPU builds will swap in real NCCL / SHM implementations.

pub mod ffi;
