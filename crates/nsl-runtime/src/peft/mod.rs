//! Parameter-Efficient Fine-Tuning (PEFT) runtime support for WRGA.
//!
//! The bulk of WRGA's work happens at compile time — placement, rank
//! allocation, fusion rewriting, and memory planning are all static passes in
//! `nsl-codegen::wrga*`.  The runtime's only job is to expose a handful of
//! C-ABI entry points that the fused kernels call into when an epilogue-
//! fused LoRA or IA³ step actually executes.
//!
//! These FFI surfaces are intentionally small and pointer-based: the
//! generated code owns the operand tensors and manages their lifetimes.

pub mod ffi;
