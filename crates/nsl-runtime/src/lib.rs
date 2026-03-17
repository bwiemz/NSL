//! NSL Runtime Library
//!
//! This crate compiles to a static library that is linked into every compiled NSL program.
//! All public functions use C ABI (`extern "C"`) so Cranelift-generated code can call them.

pub mod print;
pub mod power;
pub mod memory;
pub mod list;
pub mod string;
pub mod string_ops;
pub mod dict;
pub mod range;
pub mod hof;
pub mod math;
pub mod assert;
pub mod file_io;
pub mod args;
pub mod tensor;
pub(crate) mod cpu;
pub(crate) mod cuda;
pub mod autodiff;
pub mod checkpoint;
pub mod tokenizer;
pub mod quantize;

// M18b: Interop modules (feature-gated)
#[cfg(feature = "interop")]
pub mod safetensors_io;
#[cfg(feature = "interop")]
pub mod huggingface;
#[cfg(feature = "interop")]
pub mod weight_map;
#[cfg(feature = "interop")]
pub mod trace;
#[cfg(feature = "interop")]
pub mod onnx;
#[cfg(feature = "interop")]
pub mod onnx_proto;

// Stubs for interop FFI symbols when feature is disabled.
// The codegen always declares these as external imports, so the linker
// needs them even if the program never calls interop functions.
#[cfg(not(feature = "interop"))]
mod interop_stubs;

pub mod sampling;
pub mod data_source;
pub mod packing;
pub mod dataloader;
pub mod weight_provider;
pub mod paged_kv;
pub mod profiling;
pub mod kernel_profiler;
pub mod flash_attention;
pub mod serving;
pub mod tensor_parallel;
pub mod moe;
pub mod speculative;
pub mod context_parallel;

#[cfg(test)]
mod fuzz;
