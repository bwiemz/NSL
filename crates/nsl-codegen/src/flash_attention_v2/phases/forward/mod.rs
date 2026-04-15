//! Forward-pass phase emission modules. See `phases/backward/` for the
//! backward pass.
pub mod prelude;
pub mod q_load;
pub mod s_compute;
pub mod softmax;
pub mod pv_accum;
pub mod finalize;
pub mod csha_hooks;
