//! Per-phase PTX emitters. Each phase obeys the warp-per-row contract
//! defined in the spec's Section 1. Phase files cap at ~300 LOC.

pub mod prelude;
pub mod q_load;
pub mod s_compute;
pub mod softmax;
pub mod pv_accum;
pub mod finalize;
pub mod csha_hooks;
