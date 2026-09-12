// crates/nsl-kir/src/lib.rs
//! NSL's kernel IR (roadmap A2 step 1).
//!
//! `KernelIR` / `KirBuilder` (`kernel_ir`), the verifier (`kir_verify`) and
//! the PTX printer (`backend_ptx`) moved here from `nsl-codegen` so that the
//! runtime — which after A3 cannot depend on the compiler — can build its
//! own kernels on the same IR instead of embedding PTX text. The crate
//! depends on nothing in the workspace; `nsl_codegen::{kernel_ir,
//! kir_verify, backend_ptx}` and `nsl_codegen::gpu_target::FeatureSet`
//! re-export everything at the historical paths. The design and the steps
//! after this one are `docs/superpowers/specs/2026-09-09-a2-kir-v2-design.md`.

pub mod backend_ptx;
mod feature_set;
pub mod fragment_layout;
pub mod kernel_ir;
pub mod kir_verify;
pub mod regalloc;

pub use feature_set::FeatureSet;
