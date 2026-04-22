#![cfg(all(feature = "cuda", feature = "test-hooks"))]
//! **Test A — pedantic / strict-f32 equivalence gate** (spec §10).
//!
//! Spec: `docs/superpowers/specs/2026-04-21-matmul-cublas-swap-design.md`
//! - §4: 1e-5 relative tolerance is the strict-f32 correctness gate.
//! - §9: `NSL_MATMUL_PEDANTIC=1` forces `CUBLAS_PEDANTIC_MATH`
//!   regardless of Cargo feature.
//! - §10: split tests, both asserting cuBLAS presence + naive-kernel
//!   absence on the kernel-profile output.
//!
//! ## Isolation model
//!
//! This binary FIRST-THING sets `NSL_MATMUL_PEDANTIC=1` via
//! `std::env::set_var` at the top of each `#[test]`.  `cublas_handle()`
//! consults the env var inside its `OnceLock::get_or_init` closure, so
//! the value MUST be set before the first matmul call in the process.
//! `std::env::set_var` is safe here because no other test in this
//! binary reads the var concurrently (cuBLAS init happens serially
//! behind the OnceLock) and the Rust stdlib documents `set_var` as
//! safe on single-threaded init paths.  Cargo runs each `tests/*.rs`
//! as its own binary, so the Test A / Test B env isolation is by
//! construction.
//!
//! ## Failure signature
//!
//! If any element drifts past 1e-5 relative at pedantic mode, the bug
//! is in the cuBLAS wrapper: wrong `lda/ldb/ldc`, wrong operand order
//! (row-major-via-op-swap idiom broken), or `cublasSetMathMode` not
//! actually being applied (cudarc safe API doesn't expose it — we
//! call raw FFI in `cublas_inner`).  A drift of order 1e-3 almost
//! certainly means TF32 leaked through; grep the eprintln! init log
//! for the math-mode banner.

mod common;
use common::matmul_equiv as helper;

fn setup_env() {
    // Clear NSL_MATMUL_TF32 first so it cannot mask our PEDANTIC=1 — env
    // precedence is pedantic > tf32 > Cargo-feature, but under
    // `--features strict-matmul` both NSL_MATMUL vars unset still
    // resolves to Pedantic, which is fine for this test.
    // SAFETY: single-threaded init (see file doc-comment).
    std::env::remove_var("NSL_MATMUL_TF32");
    std::env::set_var("NSL_MATMUL_PEDANTIC", "1");
}

/// The full 10-shape matrix (sans Llama) under strict-f32 at 1e-5 rel.
#[test]
fn matmul_cublas_pedantic_equivalence() {
    setup_env();
    helper::run_matmul_equivalence_suite(1e-5, "pedantic", false);
}

/// Llama-scale (4096,4096,4096) — explicit opt-in (may exceed 60 s).
#[test]
#[ignore]
fn matmul_cublas_pedantic_equivalence_llama() {
    setup_env();
    helper::run_matmul_and_verify(4096, 4096, 4096, 1e-5, "pedantic");
}
