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

pub mod sampling;
pub mod data_source;
pub mod packing;
pub mod dataloader;

#[cfg(test)]
mod fuzz;
