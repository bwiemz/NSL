//! Backward-pass phase emission modules (Tier C).
//!
//! Kernel entry + register pool in `prelude`. Per-phase emission
//! (q_load, s_recompute, ds_compute, dp_compute, softmax_backward,
//! dqk_accum, dv_accum, dwq_dwk_dwv, dx_finalize) will land in
//! T3.2..T3.8 as Phase 3 progresses.
pub mod prelude;
pub mod q_load;
pub mod ds_compute;
