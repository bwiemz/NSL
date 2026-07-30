//! Host-side time accounting for the training step.
//!
//! Kernel profiling answers "which kernels are slow", which is the wrong
//! question when the GPU is idle most of the wall clock. On a Coder-50M step the
//! GPU is busy roughly 9% of the time, so the interesting cost is host-side and
//! invisible to `--profile-kernels`. This module times a handful of runtime
//! entry points so the step can be attributed rather than guessed at.
//!
//! Off unless `NSL_HOST_PROFILE=1`, and the guard compiles to an `Option<Instant>`
//! test when off. Enable it and every probe reports cumulatively.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::time::Instant;

/// The entry points worth attributing. Kept small and fixed so a probe costs an
/// array index rather than a hash lookup.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(usize)]
pub enum Probe {
    /// Host time inside `kernel_launch`, i.e. PTX cache lookup, module/function
    /// resolution, and the driver's `cuLaunchKernel` call.
    KernelLaunch = 0,
    /// Device allocation through the caching allocator.
    DeviceAlloc = 1,
    /// Device free through the caching allocator.
    DeviceFree = 2,
    /// Reading a scalar back from the device, which drains the pipeline.
    ScalarRead = 3,
    /// Host-to-device and device-to-host bulk copies.
    Memcpy = 4,
    /// Constructing an `NslTensor` plus its shape/stride host allocations.
    TensorAlloc = 5,
}

const N_PROBES: usize = 6;
const NAMES: [&str; N_PROBES] = [
    "kernel_launch",
    "device_alloc",
    "device_free",
    "scalar_read",
    "memcpy",
    "tensor_alloc",
];

static CALLS: [AtomicU64; N_PROBES] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];
static NANOS: [AtomicU64; N_PROBES] = [
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
    AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
];

pub fn enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("NSL_HOST_PROFILE").ok().as_deref() == Some("1"))
}

/// Times a scope into `probe`. Does nothing when profiling is off.
pub struct Timer(Option<Instant>, Probe);

impl Timer {
    #[inline]
    pub fn start(probe: Probe) -> Self {
        Self(enabled().then(Instant::now), probe)
    }
}

impl Drop for Timer {
    #[inline]
    fn drop(&mut self) {
        if let Some(started) = self.0 {
            let i = self.1 as usize;
            CALLS[i].fetch_add(1, Relaxed);
            NANOS[i].fetch_add(started.elapsed().as_nanos() as u64, Relaxed);
        }
    }
}

/// Print the accumulated attribution and reset, so each report covers one
/// interval rather than the whole run.
pub fn report_and_reset(label: &str) {
    if !enabled() {
        return;
    }
    let total: u64 = NANOS.iter().map(|n| n.load(Relaxed)).sum();
    if total == 0 {
        return;
    }
    eprintln!("[host-profile] {label}: {:.1} ms attributed", total as f64 / 1e6);
    let mut rows: Vec<(usize, u64, u64)> = (0..N_PROBES)
        .map(|i| (i, NANOS[i].load(Relaxed), CALLS[i].load(Relaxed)))
        .filter(|(_, nanos, _)| *nanos > 0)
        .collect();
    rows.sort_by_key(|(_, nanos, _)| std::cmp::Reverse(*nanos));
    for (i, nanos, calls) in rows {
        eprintln!(
            "  {:<14} {:>9.1} ms  {:>7} calls  {:>8.2} us/call  {:>5.1}%",
            NAMES[i],
            nanos as f64 / 1e6,
            calls,
            nanos as f64 / 1e3 / calls.max(1) as f64,
            100.0 * nanos as f64 / total as f64
        );
    }
    for i in 0..N_PROBES {
        CALLS[i].store(0, Relaxed);
        NANOS[i].store(0, Relaxed);
    }
}
