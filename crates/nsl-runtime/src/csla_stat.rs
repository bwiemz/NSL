//! CSLA Stage-2 (`--layerwise-accum`): window-backward anti-vacuity counter.
//!
//! The layerwise emitter calls `nsl_csla_window_mark` once per accumulation
//! window right before the buffered backward loop runs. The differential gate
//! asserts the count is > 0 on the `--layerwise-accum` arm and == 0 on the
//! baseline arm, so a silently-inert flag can never pass the parity check.
//! The counter is always live (one relaxed atomic per optimizer step); the
//! `NSL_CSLA_COUNTER=1` env var only gates the atexit report (see `args.rs`).

use std::sync::atomic::{AtomicU64, Ordering};

/// Count of CSLA window-backward phases executed.
pub static CSLA_WINDOW_COUNT: AtomicU64 = AtomicU64::new(0);

/// Emitted at the head of every CSLA window-backward phase.
#[no_mangle]
pub extern "C" fn nsl_csla_window_mark() {
    CSLA_WINDOW_COUNT.fetch_add(1, Ordering::Relaxed);
}

/// In-process numeric getter (same family as `nsl_fase_fused_step_count`):
/// lets gates assert the layerwise path actually fired without stderr scraping.
#[no_mangle]
pub extern "C" fn nsl_csla_window_count() -> i64 {
    CSLA_WINDOW_COUNT.load(Ordering::Relaxed) as i64
}
