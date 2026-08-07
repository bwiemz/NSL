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

/// Item 6: PCIe bytes moved, host-to-device and device-to-host.
///
/// Separate from the `Memcpy` probe's nanos because the benchmark matrix wants
/// TRAFFIC, and time does not give it: a run that is slow because it moves
/// 40 GB across PCIe and a run that is slow because each of its 1300 copies
/// pays a fixed launch cost look identical in the `memcpy` row, and the fix is
/// different in each case.
///
/// Counted unconditionally rather than under `enabled()`. Two `fetch_add`s on
/// a path that already issues a `cuMemcpy` is unmeasurable, and a counter that
/// only works when a separate env var is set is a counter that reads zero in
/// exactly the run someone cared about.
static H2D_BYTES: AtomicU64 = AtomicU64::new(0);
static D2H_BYTES: AtomicU64 = AtomicU64::new(0);

/// Record `n` bytes copied host-to-device.
#[inline]
pub fn record_h2d(n: usize) {
    H2D_BYTES.fetch_add(n as u64, Relaxed);
}

/// Record `n` bytes copied device-to-host.
#[inline]
pub fn record_d2h(n: usize) {
    D2H_BYTES.fetch_add(n as u64, Relaxed);
}

/// `(h2d_bytes, d2h_bytes)` moved since the last [`report_and_reset`].
pub fn pcie_bytes() -> (u64, u64) {
    (H2D_BYTES.load(Relaxed), D2H_BYTES.load(Relaxed))
}

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
    // Reset the byte counters even when nothing was timed, so an interval
    // that moved bytes but attributed no time does not fold into the next
    // report's total.
    let (h2d, d2h) = pcie_bytes();
    H2D_BYTES.store(0, Relaxed);
    D2H_BYTES.store(0, Relaxed);
    let total: u64 = NANOS.iter().map(|n| n.load(Relaxed)).sum();
    if total == 0 {
        return;
    }
    eprintln!("[host-profile] {label}: {:.1} ms attributed", total as f64 / 1e6);
    eprintln!(
        "  {:<14} h2d={:.1} MiB  d2h={:.1} MiB  total={:.1} MiB",
        "pcie",
        h2d as f64 / 1048576.0,
        d2h as f64 / 1048576.0,
        (h2d + d2h) as f64 / 1048576.0,
    );
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

#[cfg(test)]
mod pcie_counter_drift {
    /// Every PCIe-crossing copy must be counted, or the traffic figure is a
    /// LOWER BOUND presented as a measurement.
    ///
    /// This is the failure mode this gate exists for: the first version of the
    /// counter instrumented only the three synchronous `memcpy_*` helpers and
    /// read plausibly on Coder-50M — while missing the async prefetch and
    /// writeback paths, which is where essentially all of a weight-streamed
    /// 500M/1B run's PCIe traffic lives. The number would have been wrong by
    /// orders of magnitude in exactly the configurations the benchmark matrix
    /// was built to measure, and nothing would have said so.
    ///
    /// The scan is textual and deliberately dumb: it finds every
    /// `cuMemcpyHtoD*` / `cuMemcpyDtoH*` invocation in the files that make
    /// them, and requires a `record_h2d` / `record_d2h` in the preceding few
    /// lines. A new call site fails this until someone counts it.
    const SOURCES: &[(&str, &str)] = &[
        ("crates/nsl-runtime/src/cuda/mod.rs", include_str!("cuda/mod.rs")),
        ("crates/nsl-runtime/src/muon_batch.rs", include_str!("muon_batch.rs")),
    ];

    /// How far back a `record_*` may sit from its call. The helpers put it at
    /// the top of the function, above `ensure_context()` and a stream fetch.
    const LOOKBACK: usize = 12;

    #[test]
    fn every_pcie_copy_site_is_counted() {
        let mut uncounted: Vec<String> = Vec::new();
        for (path, src) in SOURCES {
            let lines: Vec<&str> = src.lines().collect();
            for (i, line) in lines.iter().enumerate() {
                // Skip the declarations and the doc comments that name them;
                // only real invocations have an open paren after the name and
                // are not inside a `///`/`//` comment.
                let t = line.trim_start();
                if t.starts_with("//") || t.starts_with("pub fn cuMemcpy") {
                    continue;
                }
                let (needle, marker) = if line.contains("cuMemcpyHtoD") {
                    ("cuMemcpyHtoD", "record_h2d")
                } else if line.contains("cuMemcpyDtoH") {
                    ("cuMemcpyDtoH", "record_d2h")
                } else {
                    continue;
                };
                let at = line.find(needle).expect("contains() matched");
                // Every failure path formats the driver symbol into its panic
                // message (`"cuMemcpyHtoD_v2({} bytes) failed: {:?}"`), which
                // looks exactly like a call site. An odd number of quotes
                // before the match means we are inside one of those literals.
                if line[..at].matches('"').count() % 2 == 1 {
                    continue;
                }
                if !line[at..].contains('(') {
                    continue;
                }
                let from = i.saturating_sub(LOOKBACK);
                if lines[from..=i].iter().any(|l| l.contains(marker)) {
                    continue;
                }
                uncounted.push(format!("{path}:{} — {}", i + 1, line.trim()));
            }
        }
        assert!(
            uncounted.is_empty(),
            "PCIe copy site(s) with no {}/{} within {LOOKBACK} lines. Each makes \
             the benchmark matrix's PCIe figure a silent undercount:\n  {}",
            "record_h2d",
            "record_d2h",
            uncounted.join("\n  "),
        );
    }

    /// The gate must be able to fail. Without this, a scan that matched
    /// nothing — a renamed driver symbol, a moved file — would read as
    /// "everything is counted" forever.
    #[test]
    fn the_scan_actually_finds_call_sites() {
        let found: usize = SOURCES
            .iter()
            .map(|(_, src)| {
                src.lines()
                    .filter(|l| {
                        let t = l.trim_start();
                        if t.starts_with("//") {
                            return false;
                        }
                        // Same string-literal exclusion as the gate above, or
                        // this counter would stay satisfied by seven panic
                        // messages after every real call site had moved away.
                        for needle in ["cuMemcpyHtoD", "cuMemcpyDtoH"] {
                            if let Some(at) = l.find(needle) {
                                if l[..at].matches('"').count() % 2 == 0 {
                                    return true;
                                }
                            }
                        }
                        false
                    })
                    .count()
            })
            .sum();
        assert!(
            found >= 8,
            "the PCIe scan found only {found} call sites; it used to find 8+. \
             Either the copies moved to a file this gate does not read, or the \
             driver symbols were renamed — both make the gate vacuous."
        );
    }
}
