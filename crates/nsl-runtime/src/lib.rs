//! # nsl-runtime — the NSL static runtime library
//!
//! This crate compiles to a static library that is linked into every compiled
//! NSL program. The Cranelift-generated code calls into it across a C ABI
//! (`extern "C"`) boundary, so a large fraction of the public surface is raw
//! pointer functions rather than safe Rust.
//!
//! ## FFI safety contract
//!
//! The `#![allow(clippy::not_unsafe_ptr_arg_deref, clippy::missing_safety_doc)]`
//! at the bottom of this comment is deliberate: every `extern "C"` entry point
//! takes raw pointers and would otherwise need an individual `unsafe` marker
//! and `# Safety` section. Rather than repeat that boilerplate on hundreds of
//! call sites, the contract that **all** runtime FFI functions uphold is stated
//! once, here, and individual modules document deviations:
//!
//! - **Ownership:** unless a function's name or docs say otherwise, pointers
//!   passed in are *borrowed* for the duration of the call only; the runtime
//!   does not take ownership and does not free them. Functions that allocate
//!   return a pointer the caller must release with the matching `*_free`.
//! - **Nullability:** input pointers must be non-null and properly aligned for
//!   their element type unless explicitly documented as nullable. Passing null
//!   or a misaligned pointer is undefined behavior.
//! - **Lifetimes:** returned pointers are valid until the corresponding free /
//!   reset call, or until the owning arena is dropped. They must not outlive
//!   the runtime objects they reference.
//! - **Alignment & layout:** tensor data pointers must point at buffers laid
//!   out as the compiler expects (contiguous, element-type aligned). The
//!   compiler (`nsl-codegen`) and this crate are the two halves of one ABI;
//!   they must be built from the same source revision.
//! - **Error behavior:** FFI functions report failure through documented
//!   sentinel return values (e.g. null pointers, negative counts) rather than
//!   panicking across the ABI boundary. Panics are treated as aborts.
//!
//! Fuzz / property tests for the boundary live in `src/fuzz.rs` (cfg(test)).
//!
//! ## Module organization
//!
//! Modules are declared at the crate root (keeping `nsl_runtime::foo` paths
//! stable) and are also grouped under subsystem facade namespaces — see the
//! [`builtins`], [`training`], [`quantization`], [`attention`], [`dataio`],
//! [`inference`], [`distributed`], [`finetune`], [`observability`], [`ffi`],
//! and [`experimental`] modules at the end of this file, plus the feature-gated
//! [`interop`] facade. See `ARCHITECTURE.md` for details.

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
#[cfg(feature = "onnx-rt-op")]
pub mod onnx_rt_op;

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
pub mod grad_context;
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

// ===========================================================================
// Subsystem facade namespaces
//
// These re-export the modules above under architecture-oriented namespaces so
// the FFI/runtime surface is navigable. The crate-root paths remain available
// for backward compatibility; new code is encouraged to import via the facades.
// `cpu`/`cuda` are intentionally excluded (kept `pub(crate)` to avoid leaking
// the device driver API).
// ===========================================================================

// Facade names are chosen to avoid colliding with real crate-root modules
// (`autodiff`, `data`, `serving`, `peft`) and with the `core` extern-prelude
// crate (`unikernel` uses bare `core::arch` paths).

/// Language builtins and the base runtime: scalars, strings, containers,
/// tensors, memory, and process I/O.
pub mod builtins {
    pub use crate::{
        args, assert, dict, file_io, hof, io, list, math, memory, power, print,
        range, slab, string, string_ops, tensor,
    };
}

/// GPU device backend selection (the concrete `cpu`/`cuda` drivers stay
/// crate-private).
pub mod gpu {
    pub use crate::gpu_backend;
}

/// Automatic differentiation and training-time runtime support.
pub mod training {
    pub use crate::{
        autodiff, backward_context, checkpoint, grad_context, vmap_runtime, zero,
    };
}

/// Quantization and reduced-precision execution.
pub mod quantization {
    pub use crate::{awq, fase_bc, fp8, gptq, packing, quantize};
}

/// Attention runtime kernels.
pub mod attention {
    pub use crate::{flash_attention, pca_rope_runtime, pca_tier_b_runtime};
}

/// Data loading, tokenization, sampling, and calibration ingest.
pub mod dataio {
    pub use crate::{
        calibration_data, data, data_source, dataloader, grammar, sampling,
        token_alignment, tokenizer,
    };
}

/// Inference serving: paged KV cache, speculative decoding, disaggregation,
/// elasticity, and KV compression.
pub mod inference {
    pub use crate::{
        disaggregated, elastic, kv_compress, paged_kv, serving, speculative,
    };
}

/// Distributed execution and parallelism strategies.
pub mod distributed {
    pub use crate::{context_parallel, moe, pipeline, tensor_parallel};
}

/// Parameter-efficient fine-tuning adapters (LoRA / IA³ / gated-LoRA).
pub mod finetune {
    pub use crate::{fused_adapter, peft};
}

/// Observability: profiling, tensor tracing, health monitoring, and
/// deterministic-op verification.
pub mod observability {
    pub use crate::{
        deterministic_ops, health, inspect, kernel_profiler, profiler, profiling,
        tensor_trace, trace_diff,
    };
}

/// Foreign-function / framework interop that is always compiled in: the DLPack
/// bridge, the legacy C API, and weight provisioning.
pub mod ffi {
    pub use crate::{c_api, dlpack, weight_provider};
}

/// Optional framework interop (enabled with the `interop` feature):
/// safetensors, Hugging Face, ONNX, and weight-map bridges.
#[cfg(feature = "interop")]
pub mod interop {
    pub use crate::{huggingface, onnx, onnx_proto, safetensors_io, trace, weight_map};
}

/// Experimental research subsystems. **APIs here are unstable** and may change
/// or be removed between releases.
pub mod experimental {
    pub use crate::{agent, cfie, cpdt, multimodal, sparse, unikernel};
}
