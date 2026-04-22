#![cfg(all(feature = "cuda", feature = "test-hooks"))]
//! **Test B — TF32 default sanity** (spec §10).
//!
//! Spec: `docs/superpowers/specs/2026-04-21-matmul-cublas-swap-design.md`
//! - §9: with NO env override, the runtime inherits the Cargo-feature
//!   default (`CUBLAS_DEFAULT_MATH` / TF32 on sm_80+ under the default
//!   Cargo config; `CUBLAS_PEDANTIC_MATH` under `--features strict-matmul`).
//! - §10: 5e-3 relative tolerance per NVIDIA TF32 spec (10-bit
//!   mantissa).  Looser than §4's strict-f32 gate but still tight
//!   enough to catch a real wrapper bug.
//!
//! ## Isolation model
//!
//! `tests/*.rs` files compile into separate binaries; this binary's
//! `cublas_handle` `OnceLock` is independent of Test A's.  We actively
//! clear `NSL_MATMUL_PEDANTIC` and `NSL_MATMUL_TF32` at test entry so
//! the env inherits the Cargo-feature default.  Under default features
//! that's TF32; under `--features strict-matmul` that's pedantic
//! (which trivially satisfies the 5e-3 tolerance).
//!
//! For a test that unambiguously exercises the TF32 dispatch path
//! regardless of the Cargo-feature matrix, see
//! `matmul_cublas_tf32_forced_equivalence` below.
//!
//! ## Failure signature
//!
//! Drift > 5e-3 under TF32 almost certainly means the mode resolution
//! or the `cublasSetMathMode` application is broken.  Under TF32 the
//! wrapper correctness (lda/ldb/ldc, operand order) is also validated
//! — a wrapper bug would typically produce order-of-magnitude drift
//! well above 5e-3.

mod common;
use common::matmul_equiv as helper;

fn setup_env_default() {
    // Inherit Cargo-feature default — clear any lingering env overrides.
    // SAFETY: single-threaded init (see matmul_cublas_equivalence.rs doc).
    std::env::remove_var("NSL_MATMUL_PEDANTIC");
    std::env::remove_var("NSL_MATMUL_TF32");
}

fn setup_env_force_tf32() {
    // Force TF32 regardless of Cargo feature — the canonical way to
    // exercise the TF32 dispatch path in a CI matrix that may flip
    // `strict-matmul` on and off.
    std::env::remove_var("NSL_MATMUL_PEDANTIC");
    std::env::set_var("NSL_MATMUL_TF32", "1");
}

/// Inherits the Cargo-feature default math mode.  Under default
/// features = TF32; under `--features strict-matmul` = pedantic.
/// 5e-3 tolerance accommodates either regime.
#[test]
fn matmul_cublas_tf32_default_sanity() {
    setup_env_default();
    helper::run_matmul_equivalence_suite(5e-3, "tf32_default", false);
}

/// Explicit `NSL_MATMUL_TF32=1`: unambiguously exercises TF32 even
/// when the binary was built with `--features strict-matmul`.  This is
/// the canonical bug-catcher for "mode-resolution env precedence works
/// as spec §9 documents."
#[test]
fn matmul_cublas_tf32_forced_equivalence() {
    setup_env_force_tf32();
    helper::run_matmul_equivalence_suite(5e-3, "tf32_forced", false);
}

/// Llama-scale — explicit opt-in.
#[test]
#[ignore]
fn matmul_cublas_tf32_default_sanity_llama() {
    setup_env_default();
    helper::run_matmul_and_verify(4096, 4096, 4096, 5e-3, "tf32_default");
}
