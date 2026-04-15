//! FA-2 v2 scalar emitter phase modules.
//!
//! Forward phases live under `forward/`; backward phases (added in
//! Phase 3) live under `backward/`. The re-exports below keep internal
//! callers that used `phases::q_load::emit` working unchanged.
pub mod forward;
// `pub mod backward;` added in Phase 3 (Task T3.1).

pub use forward::{
    prelude, q_load, s_compute, softmax, pv_accum, finalize, csha_hooks,
};
