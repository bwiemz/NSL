// Shared helpers for integration tests. Declared as a module by every
// test file via `mod common;`. Cargo treats `tests/common/mod.rs` as a
// shared submodule, not its own test binary.
#![allow(dead_code)]

pub mod fp8_reference;

// Guarded by `cuda + test-hooks` to match `matmul_equiv`'s file-level cfg;
// keeps non-CUDA test builds from pulling the helper into their compile
// units (and the cublas_inner re-exports are only available under `cuda`).
#[cfg(all(feature = "cuda", feature = "test-hooks"))]
pub mod matmul_equiv;
