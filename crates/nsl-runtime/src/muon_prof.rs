//! Muon optimizer internal profiler (perf-campaign item 2).
//!
//! Opt-in, env-gated timestamp regions over every stage of the Muon update
//! path. The 500M campaign proved the aggregate cost is pathological
//! (66.8 s/micro vs AdamW's 4.3) but not WHERE it goes — per-op sync,
//! allocation churn, physical transposes, GEMM shape utilization, state
//! staging, or the nominal Newton-Schulz arithmetic. This module answers
//! that before any execution-strategy retune.
//!
//! Modes (env `NSL_MUON_PROF`):
//!   unset/0 — disabled: zero overhead beyond one cached env read per call
//!             site (a relaxed atomic load).
//!   1       — synced attribution: a device synchronize at every region
//!             boundary so each region's wall time includes its kernels.
//!             Distorts overlap by design; region SHARES are the signal.
//!   2       — enqueue-only: no added syncs; regions measure host-side
//!             dispatch time. The gap between mode-2 totals and mode-1
//!             totals separates launch/dispatch overhead from execution.
//!
//! Report: `[muon-prof]` table on process exit (atexit) or on demand via
//! `nsl_muon_prof_report()`. Counters are process-global atomics; training
//! is single-threaded per process, and cross-thread interleaving would only
//! blur shares, not corrupt memory.

use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::time::Instant;

/// Region ids. Keep in sync with `NAMES`.
#[derive(Clone, Copy)]
#[repr(usize)]
pub enum Region {
    /// Host->device upload of momentum state (offload stage-in).
    MomentumStageIn = 0,
    /// m = momentum*m + grad (+ Nesterov combine).
    MomentumUpdate = 1,
    /// Frobenius sum-square reduction.
    FrobeniusReduce = 2,
    /// The 1/(sqrt(ss)+eps) normalization scale apply.
    NormalizeScale = 3,
    /// Materialized transpose+contiguous at NS entry (tall inputs).
    EntryTranspose = 4,
    /// Gram GEMM: a = x @ x^T.
    GramGemm = 5,
    /// Gram-square GEMM: aa = a @ a.
    GramSqGemm = 6,
    /// Polynomial combine + poly-times-x GEMM: x = NS_A*x + b @ x.
    PolyGemm = 7,
    /// Materialized transpose+contiguous at NS exit (tall inputs).
    ExitTranspose = 8,
    /// theta = theta*(1-lr*wd) - (lr*scale)*o.
    ParamUpdate = 9,
    /// Device->host writeback of momentum state (offload).
    MomentumWriteback = 10,
    /// Whole nsl_tensor_muon_orthogonalize call (superset of 2..=8).
    NsTotal = 11,
    /// AdamW-routed arm of muon_step (embeddings/head/vectors).
    AdamwArm = 12,
}

const N_REGIONS: usize = 13;
const NAMES: [&str; N_REGIONS] = [
    "momentum-stage-in",
    "momentum-update",
    "frobenius-reduce",
    "normalize-scale",
    "entry-transpose",
    "gram-gemm",
    "gram-sq-gemm",
    "poly-gemm",
    "exit-transpose",
    "param-update",
    "momentum-writeback",
    "ns-total",
    "adamw-arm",
];

static TOTAL_NS: [AtomicU64; N_REGIONS] = [const { AtomicU64::new(0) }; N_REGIONS];
static CALLS: [AtomicU64; N_REGIONS] = [const { AtomicU64::new(0) }; N_REGIONS];

/// -1 = not yet read, 0 = off, 1 = synced, 2 = enqueue-only.
static MODE: AtomicI64 = AtomicI64::new(-1);
static REPORT_REGISTERED: AtomicBool = AtomicBool::new(false);

#[inline]
pub fn mode() -> i64 {
    let m = MODE.load(Ordering::Relaxed);
    if m >= 0 {
        return m;
    }
    let parsed = match std::env::var("NSL_MUON_PROF").ok().as_deref() {
        Some("1") => 1,
        Some("2") => 2,
        _ => 0,
    };
    MODE.store(parsed, Ordering::Relaxed);
    if parsed > 0 && !REPORT_REGISTERED.swap(true, Ordering::Relaxed) {
        // Same piggyback-on-C-atexit pattern as args.rs's NSL_GPU_MEM_REPORT.
        extern "C" {
            fn atexit(cb: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(report_at_exit);
        }
    }
    parsed
}

#[inline]
pub fn enabled() -> bool {
    mode() > 0
}

/// Synchronize the device so the elapsed wall time attributes the kernels
/// launched inside the region to the region. Mode 2 skips this on purpose.
fn maybe_sync() {
    if mode() != 1 {
        return;
    }
    #[cfg(feature = "cuda")]
    unsafe {
        crate::cuda::inner::ensure_context();
        cudarc::driver::sys::cuCtxSynchronize();
    }
}

/// RAII region scope. Construct via [`scope`]; `None` when profiling is off
/// so disabled call sites cost one atomic load.
pub struct Scope {
    region: usize,
    start: Instant,
}

#[inline]
pub fn scope(region: Region) -> Option<Scope> {
    if !enabled() {
        return None;
    }
    maybe_sync();
    Some(Scope {
        region: region as usize,
        start: Instant::now(),
    })
}

impl Drop for Scope {
    fn drop(&mut self) {
        maybe_sync();
        let ns = self.start.elapsed().as_nanos() as u64;
        TOTAL_NS[self.region].fetch_add(ns, Ordering::Relaxed);
        CALLS[self.region].fetch_add(1, Ordering::Relaxed);
    }
}

extern "C" fn report_at_exit() {
    report();
}

pub fn report() {
    if !enabled() {
        return;
    }
    let mode_name = if mode() == 1 { "synced" } else { "enqueue-only" };
    // Exclusive stage total: ns-total double-counts its sub-regions, and the
    // AdamW arm is a different code path — sum only the Muon-route stages.
    let stage_ids = [0usize, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let stage_total: u64 = stage_ids
        .iter()
        .map(|&i| TOTAL_NS[i].load(Ordering::Relaxed))
        .sum();
    if stage_total == 0 && TOTAL_NS[Region::AdamwArm as usize].load(Ordering::Relaxed) == 0 {
        return;
    }
    eprintln!("[muon-prof] mode={mode_name} — per-region totals (muon-route stage share excludes ns-total/adamw-arm)");
    eprintln!(
        "[muon-prof] {:<20} {:>10} {:>12} {:>10} {:>7}",
        "region", "calls", "total_ms", "avg_us", "share"
    );
    for i in 0..N_REGIONS {
        let calls = CALLS[i].load(Ordering::Relaxed);
        if calls == 0 {
            continue;
        }
        let ns = TOTAL_NS[i].load(Ordering::Relaxed);
        let share = if stage_ids.contains(&i) && stage_total > 0 {
            format!("{:5.1}%", ns as f64 * 100.0 / stage_total as f64)
        } else {
            "    —".to_string()
        };
        eprintln!(
            "[muon-prof] {:<20} {:>10} {:>12.2} {:>10.1} {:>7}",
            NAMES[i],
            calls,
            ns as f64 / 1e6,
            (ns / calls.max(1)) as f64 / 1e3,
            share
        );
    }
}

/// On-demand report FFI (callable from compiled code / tests).
#[no_mangle]
pub extern "C" fn nsl_muon_prof_report() {
    report();
}

// ── Explicit begin/end markers for compiled-code call sites ────────────────
//
// Codegen emits these around stages that live in emitted IR rather than in
// one Rust function (momentum stage-in/update/writeback, param update,
// AdamW arm). RAII can't cross the FFI boundary, so a thread-local stack of
// open regions pairs each end with its begin; mismatched pairs are a loud
// error rather than silent misattribution.

thread_local! {
    static OPEN: std::cell::RefCell<Vec<(usize, Instant)>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

#[no_mangle]
pub extern "C" fn nsl_muon_prof_begin(region: i64) {
    if !enabled() {
        return;
    }
    let r = region as usize;
    if r >= N_REGIONS {
        eprintln!("nsl: muon_prof_begin: bad region id {region}");
        return;
    }
    maybe_sync();
    OPEN.with(|s| s.borrow_mut().push((r, Instant::now())));
}

#[no_mangle]
pub extern "C" fn nsl_muon_prof_end(region: i64) {
    if !enabled() {
        return;
    }
    let r = region as usize;
    maybe_sync();
    OPEN.with(|s| {
        let mut open = s.borrow_mut();
        match open.pop() {
            Some((top, start)) if top == r => {
                TOTAL_NS[r].fetch_add(start.elapsed().as_nanos() as u64, Ordering::Relaxed);
                CALLS[r].fetch_add(1, Ordering::Relaxed);
            }
            Some((top, _)) => {
                eprintln!(
                    "nsl: muon_prof_end region {region} does not match open region {top} — dropping sample"
                );
            }
            None => {
                eprintln!("nsl: muon_prof_end with no open region (id {region})");
            }
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn disabled_scope_is_none_without_env() {
        // MODE may already be cached by another test; only assert coherence.
        let s = scope(Region::GramGemm);
        assert_eq!(s.is_some(), enabled());
    }

    #[test]
    fn report_names_cover_all_regions() {
        assert_eq!(NAMES.len(), N_REGIONS);
        assert!(NAMES.iter().all(|n| !n.is_empty()));
    }
}
