//! NSL Runtime Library
//!
//! This crate compiles to a static library that is linked into every compiled NSL program.
//! All public functions use C ABI (`extern "C"`) so Cranelift-generated code can call them.

// Clippy lints we accept as-is at the crate level:
//
// - `not_unsafe_ptr_arg_deref` / `missing_safety_doc`: this crate is the C ABI
//   boundary for Cranelift-emitted code. Every public function takes raw
//   pointers; the safety invariants are documented module-by-module in the
//   compiler/runtime contract, not per-function. Marking each entry point
//   `unsafe` would require touching every Cranelift call site.
// - `doc_overindented_list_items` / `doc_lazy_continuation`: pedantic doc
//   formatting lints new in clippy 1.95 that fire on preexisting ASCII
//   diagrams and continuation lines.
// - `needless_range_loop`: indexed loops with `i` are intentional where the
//   index itself feeds into other buffers in lock-step.
// - `manual_checked_ops` (clippy 1.95+): explicit `if divisor == 0` guards
//   are clearer than `.checked_div(...).unwrap_or(...)` at these call sites.
//   `unknown_lints` is listed first so that compilers older than 1.95 don't
//   error on the unrecognised lint name under `-D warnings`.
#![allow(
    unknown_lints,
    clippy::not_unsafe_ptr_arg_deref,
    clippy::missing_safety_doc,
    clippy::manual_checked_ops,
    clippy::doc_overindented_list_items,
    clippy::doc_lazy_continuation,
    clippy::needless_range_loop
)]

pub mod health;
pub mod profiler;
pub mod print;
pub mod power;
pub mod memory;
pub mod list;
pub mod string;
pub mod string_ops;
pub mod io;
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

// Doc-hidden re-exports for cross-crate integration tests
// (see crates/nsl-codegen/tests/csha_cuda_launch_*). Keeping the cuda
// module pub(crate) otherwise avoids leaking the driver API.
#[doc(hidden)]
pub use cuda::{
    nsl_cuda_init,
    nsl_test_cuda_alloc, nsl_test_cuda_free,
    nsl_test_cuda_h2d, nsl_test_cuda_d2h,
    nsl_test_cuda_jit_log,
};

// CSHA Tier B.1 pre-pass PTX kernels (exposed for the GPU validation
// tests in `tests/tier_b1_prepass_gpu.rs`; the orchestration FFI uses
// them internally via `cuda::tier_b1_prepass::launch_*`).
#[doc(hidden)]
pub use cuda::tier_b1_prepass::{
    CSHA_TIER_B1_PREPASS_W_PTX, CSHA_TIER_B1_PREPASS_X_PTX,
};

// Test-only re-export: SM version query for dispatcher tests.
#[cfg(feature = "test-hooks")]
pub use cuda::test_detect_sm_version;

pub mod autodiff;
pub mod checkpoint;
pub mod inspect;
pub mod tokenizer;
pub mod quantize;
pub mod fp8;
pub mod awq;
pub mod fase_bc;
#[allow(deprecated)]
pub use awq::{
    nsl_awq_write_sidecar,  // deprecated; kept as the rename shim during transition
    nsl_calib_write_sidecar,
    AwqProjectionDescriptor,
    WggoLayerDescriptor,
};
pub mod gptq;

// Calibration-data loader: .bin and .safetensors dispatch (unconditional)
pub mod calibration_data;
pub use calibration_data::{
    nsl_calibration_load, nsl_calibration_batch_at,
    nsl_calibration_count, nsl_calibration_free,
};

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
pub mod pca_tier_b_runtime;
pub mod pca_rope_runtime;
pub mod serving;
pub mod tensor_parallel;
pub mod moe;
pub mod speculative;
pub mod context_parallel;
pub mod slab;
pub mod disaggregated;
pub mod elastic;
pub mod kv_compress;
pub mod grammar;
pub mod token_alignment;
pub mod gpu_backend;
pub mod vmap_runtime;
pub mod backward_context;
pub mod pipeline;
pub mod data;
pub mod zero;
pub mod tensor_trace;
pub mod trace_diff;
pub mod deterministic_ops;
pub mod multimodal;
pub mod cfie;
pub mod cpdt;
pub mod sparse;
pub mod peft;

// WRGA B.3 Task 4: fused LoRA/IA³ adapter FFIs
pub mod fused_adapter;

// M54: Unikernel runtime (bare-metal deployment)
pub mod unikernel;

// M62: Legacy Interop — DLPack bridge + C API
pub mod dlpack;
pub mod c_api;

// M56: Agent runtime — mailboxes, scheduler, pool, FFI.
pub mod agent;

#[cfg(test)]
mod fuzz;
