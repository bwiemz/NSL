// crates/nsl-kir/src/kernels/mod.rs
//! Kernels the runtime builds for itself (roadmap A2 steps 7 and 11).
//!
//! `nsl-kir` is a leaf crate, so `nsl-runtime` can depend on it where it
//! cannot depend on `nsl-codegen`. That is the whole point of this module:
//! a kernel the runtime needs at execution time is *described* here once
//! and *built* on demand, instead of being emitted by the compiler and
//! then transcribed into the runtime as PTX text that a parity test has to
//! keep honest.

pub mod block_reduce;
pub mod cast;
pub mod ce_bwd;
pub mod conv2d;
pub mod data_movement;
pub mod dequant;
pub mod det_scatter;
pub mod det_sum;
pub mod dropout;
pub mod elementwise;
pub mod embedding_bwd;
pub mod lce_finalize;
pub mod lookup;
pub mod maxpool;
pub mod muon_batch;
pub mod norm;
pub mod optim;
pub mod rmsnorm_dgamma;
pub mod rmsnorm_dx;
pub mod softmax;
pub mod spmm;
pub mod spmv;
pub mod strided_copy;
pub mod sum_sq;
pub mod tensor_stats;
pub mod tier_b1_prepass;
