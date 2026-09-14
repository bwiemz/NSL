// crates/nsl-kir/src/kernels/mod.rs
//! Kernels the runtime builds for itself (roadmap A2 step 7).
//!
//! `nsl-kir` is a leaf crate, so `nsl-runtime` can depend on it where it
//! cannot depend on `nsl-codegen`. That is the whole point of this module:
//! a kernel the runtime needs at execution time is *described* here once
//! and *built* on demand, instead of being emitted by the compiler and
//! then transcribed into the runtime as PTX text that a parity test has to
//! keep honest.

pub mod cast;
