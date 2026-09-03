//! The runtime ABI: the `extern "C"` surface the codegen calls into, grouped
//! by subsystem.
//!
//! Distinct from [`mod@crate::builtins`], which is the surface a program
//! writes directly. Adding an entry here does not change what an NSL program
//! can say, only how it runs. See `builtins/mod.rs` for the full rationale,
//! for why declaration order is free, and for how to add one.

pub(crate) mod diagnostics;
pub(crate) mod distributed;
pub(crate) mod inference;
pub(crate) mod interop;
pub(crate) mod memory;
pub(crate) mod optimizer;
pub(crate) mod quantization;
pub(crate) mod tensor;
pub(crate) mod training;
