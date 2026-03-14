//! Memory watermark profiler for paged KV-cache.
//!
//! When enabled via `nsl_profiler_start()`, tracks block alloc/free events
//! and computes peak usage, utilization ratio, and fragmentation score.
//! Outputs Chrome tracing JSON via `nsl_profiler_dump()`.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

// ── Data structures ──────────────────────────────────────────────────────────

/// A single profiling event.
struct ProfileEvent {
    /// Microseconds since profiler start.
    timestamp_us: u64,
    /// Whether this is an alloc or free event.
    event_type: EventType,
    /// Block that was allocated or freed.
    block_id: u32,
    /// Sequence that owns the block.
    seq_id: u64,
}

enum EventType {
    Alloc,
    Free,
}

/// Global profiler state. Only one profiler active at a time.
struct Profiler {
    enabled: AtomicBool,
    start_time: Mutex<Option<Instant>>,
    events: Mutex<Vec<ProfileEvent>>,
    current_blocks: AtomicUsize,
    peak_blocks: AtomicUsize,
    total_allocs: AtomicUsize,
    total_frees: AtomicUsize,
    total_blocks: AtomicUsize,
}

static PROFILER: Profiler = Profiler {
    enabled: AtomicBool::new(false),
    start_time: Mutex::new(None),
    events: Mutex::new(Vec::new()),
    current_blocks: AtomicUsize::new(0),
    peak_blocks: AtomicUsize::new(0),
    total_allocs: AtomicUsize::new(0),
    total_frees: AtomicUsize::new(0),
    total_blocks: AtomicUsize::new(0),
};

// ── Internal helpers ─────────────────────────────────────────────────────────

/// Return elapsed microseconds since profiler start, or 0 if not started.
fn elapsed_us() -> u64 {
    let guard = PROFILER.start_time.lock().unwrap();
    match *guard {
        Some(t) => t.elapsed().as_micros() as u64,
        None => 0,
    }
}

// ── Public Rust API ──────────────────────────────────────────────────────────

/// Check whether the profiler is currently enabled.
pub fn profiler_enabled() -> bool {
    PROFILER.enabled.load(Ordering::Relaxed)
}

/// Record a block allocation event. No-op when profiler is disabled.
pub fn profiler_record_alloc(block_id: u32, seq_id: u64) {
    if !profiler_enabled() {
        return;
    }
    let ts = elapsed_us();
    PROFILER
        .events
        .lock()
        .unwrap()
        .push(ProfileEvent {
            timestamp_us: ts,
            event_type: EventType::Alloc,
            block_id,
            seq_id,
        });
    PROFILER.total_allocs.fetch_add(1, Ordering::Relaxed);

    // Increment current_blocks and update peak via CAS loop.
    let new_val = PROFILER.current_blocks.fetch_add(1, Ordering::Relaxed) + 1;
    loop {
        let old_peak = PROFILER.peak_blocks.load(Ordering::Relaxed);
        if new_val <= old_peak {
            break;
        }
        match PROFILER.peak_blocks.compare_exchange_weak(
            old_peak,
            new_val,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break,
            Err(_) => continue,
        }
    }
}

/// Record a block free event. No-op when profiler is disabled.
pub fn profiler_record_free(block_id: u32, seq_id: u64) {
    if !profiler_enabled() {
        return;
    }
    let ts = elapsed_us();
    PROFILER
        .events
        .lock()
        .unwrap()
        .push(ProfileEvent {
            timestamp_us: ts,
            event_type: EventType::Free,
            block_id,
            seq_id,
        });
    PROFILER.total_frees.fetch_add(1, Ordering::Relaxed);
    let prev = PROFILER.current_blocks.fetch_sub(1, Ordering::Relaxed);
    debug_assert!(prev > 0, "profiler current_blocks underflow");
}

/// Return the peak number of simultaneously-allocated blocks.
pub fn profiler_peak() -> usize {
    PROFILER.peak_blocks.load(Ordering::Relaxed)
}

/// Return the current utilization ratio (current_blocks / total_blocks).
pub fn profiler_utilization() -> f64 {
    let total = PROFILER.total_blocks.load(Ordering::Relaxed);
    if total == 0 {
        return 0.0;
    }
    PROFILER.current_blocks.load(Ordering::Relaxed) as f64 / total as f64
}

// ── FFI exports ──────────────────────────────────────────────────────────────

/// Start the profiler, resetting all counters and events.
///
/// `total_blocks` is the size of the block pool (used for utilization ratio).
#[no_mangle]
pub extern "C" fn nsl_profiler_start(total_blocks: i64) {
    // Reset all counters.
    PROFILER.current_blocks.store(0, Ordering::Relaxed);
    PROFILER.peak_blocks.store(0, Ordering::Relaxed);
    PROFILER.total_allocs.store(0, Ordering::Relaxed);
    PROFILER.total_frees.store(0, Ordering::Relaxed);
    PROFILER
        .total_blocks
        .store(total_blocks as usize, Ordering::Relaxed);

    // Clear event log.
    PROFILER.events.lock().unwrap().clear();

    // Record start time.
    *PROFILER.start_time.lock().unwrap() = Some(Instant::now());

    // Enable last so early events aren't recorded before reset completes.
    PROFILER.enabled.store(true, Ordering::Release);

    // Register at-exit handler to auto-dump the profile on program exit.
    // Uses a one-shot flag so we only register once even if start() is called
    // multiple times.
    static REGISTERED: AtomicBool = AtomicBool::new(false);
    if !REGISTERED.swap(true, Ordering::SeqCst) {
        extern "C" fn dump_on_exit() {
            if PROFILER.enabled.load(Ordering::Relaxed) {
                PROFILER.enabled.store(false, Ordering::Release);

                let path = "memory_profile.json";
                let path_bytes = path.as_bytes();
                nsl_profiler_dump(path_bytes.as_ptr(), path_bytes.len() as i64);

                let peak = PROFILER.peak_blocks.load(Ordering::Relaxed);
                eprintln!(
                    "nsl: memory profile written to {} (peak: {} blocks)",
                    path, peak
                );
            }
        }
        // SAFETY: dump_on_exit is an extern "C" fn with the correct atexit signature.
        extern "C" {
            fn atexit(callback: extern "C" fn()) -> i32;
        }
        unsafe {
            atexit(dump_on_exit);
        }
    }
}

/// Stop the profiler.
#[no_mangle]
pub extern "C" fn nsl_profiler_stop() {
    PROFILER.enabled.store(false, Ordering::Release);
}

/// Dump profiling data as Chrome tracing JSON to the file at `path_ptr`.
#[no_mangle]
pub extern "C" fn nsl_profiler_dump(path_ptr: *const u8, path_len: i64) {
    let path_str = unsafe {
        let slice = std::slice::from_raw_parts(path_ptr, path_len as usize);
        std::str::from_utf8_unchecked(slice)
    };

    let events = PROFILER.events.lock().unwrap();
    let peak = PROFILER.peak_blocks.load(Ordering::Relaxed);
    let total_allocs = PROFILER.total_allocs.load(Ordering::Relaxed);
    let total_frees = PROFILER.total_frees.load(Ordering::Relaxed);
    let total_blocks = PROFILER.total_blocks.load(Ordering::Relaxed);
    let current = PROFILER.current_blocks.load(Ordering::Relaxed);

    let utilization = if total_blocks > 0 {
        current as f64 / total_blocks as f64
    } else {
        0.0
    };

    // Build JSON manually to avoid pulling in serde_json's Serialize derive
    // for these small internal types.
    let mut trace_events = String::from("[");
    for (i, ev) in events.iter().enumerate() {
        if i > 0 {
            trace_events.push(',');
        }
        let name = match ev.event_type {
            EventType::Alloc => "block_alloc",
            EventType::Free => "block_free",
        };
        trace_events.push_str(&format!(
            "{{\"name\":\"{}\",\"cat\":\"memory\",\"ph\":\"i\",\"ts\":{},\"pid\":0,\"tid\":0,\"args\":{{\"block_id\":{},\"seq_id\":{}}}}}",
            name,
            ev.timestamp_us,
            ev.block_id,
            ev.seq_id,
        ));
    }
    trace_events.push(']');

    let json = format!(
        "{{\"traceEvents\":{},\"metadata\":{{\"peak_blocks\":{},\"total_allocs\":{},\"total_frees\":{},\"total_blocks\":{},\"utilization\":{:.6}}}}}",
        trace_events,
        peak,
        total_allocs,
        total_frees,
        total_blocks,
        utilization,
    );

    std::fs::write(path_str, json).unwrap_or_else(|e| {
        eprintln!("nsl_profiler_dump: failed to write {}: {}", path_str, e);
    });
}

/// Return the peak number of simultaneously-allocated blocks.
#[no_mangle]
pub extern "C" fn nsl_profiler_peak() -> i64 {
    profiler_peak() as i64
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::MutexGuard;

    /// Global lock to serialize tests that share the static PROFILER state.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// Acquire the test lock and reset profiler state.  The returned guard
    /// keeps the lock held for the duration of the test.
    fn lock_and_reset() -> MutexGuard<'static, ()> {
        let guard = TEST_LOCK.lock().unwrap();
        PROFILER.enabled.store(false, Ordering::Relaxed);
        PROFILER.current_blocks.store(0, Ordering::Relaxed);
        PROFILER.peak_blocks.store(0, Ordering::Relaxed);
        PROFILER.total_allocs.store(0, Ordering::Relaxed);
        PROFILER.total_frees.store(0, Ordering::Relaxed);
        PROFILER.total_blocks.store(0, Ordering::Relaxed);
        PROFILER.events.lock().unwrap().clear();
        *PROFILER.start_time.lock().unwrap() = None;
        guard
    }

    #[test]
    fn test_profiler_start_stop() {
        let _g = lock_and_reset();
        assert!(!profiler_enabled());
        nsl_profiler_start(10);
        assert!(profiler_enabled());
        nsl_profiler_stop();
        assert!(!profiler_enabled());
    }

    #[test]
    fn test_profiler_record_alloc_free() {
        let _g = lock_and_reset();
        nsl_profiler_start(10);

        profiler_record_alloc(0, 1);
        profiler_record_alloc(1, 1);
        assert_eq!(PROFILER.current_blocks.load(Ordering::Relaxed), 2);
        assert_eq!(PROFILER.total_allocs.load(Ordering::Relaxed), 2);

        profiler_record_free(0, 1);
        assert_eq!(PROFILER.current_blocks.load(Ordering::Relaxed), 1);
        assert_eq!(PROFILER.total_frees.load(Ordering::Relaxed), 1);
        assert_eq!(profiler_peak(), 2);

        nsl_profiler_stop();
    }

    #[test]
    fn test_profiler_peak_tracking() {
        let _g = lock_and_reset();
        nsl_profiler_start(10);

        // Alloc 3 blocks.
        profiler_record_alloc(0, 1);
        profiler_record_alloc(1, 1);
        profiler_record_alloc(2, 1);
        assert_eq!(profiler_peak(), 3);

        // Free 1 block.
        profiler_record_free(1, 1);
        assert_eq!(PROFILER.current_blocks.load(Ordering::Relaxed), 2);
        // Peak should still be 3.
        assert_eq!(profiler_peak(), 3);

        // Alloc 2 more blocks (current becomes 4).
        profiler_record_alloc(3, 2);
        profiler_record_alloc(4, 2);
        assert_eq!(PROFILER.current_blocks.load(Ordering::Relaxed), 4);
        // Peak should now be 4.
        assert_eq!(profiler_peak(), 4);

        nsl_profiler_stop();
    }

    #[test]
    fn test_profiler_dump_json() {
        let _g = lock_and_reset();
        nsl_profiler_start(8);

        profiler_record_alloc(0, 1);
        profiler_record_alloc(1, 1);
        profiler_record_free(0, 1);

        let dir = std::env::temp_dir();
        let path: PathBuf = dir.join("nsl_profiler_test.json");
        let path_str = path.to_str().unwrap();
        let path_bytes = path_str.as_bytes();

        nsl_profiler_dump(path_bytes.as_ptr(), path_bytes.len() as i64);

        let content = std::fs::read_to_string(&path).expect("dump file should exist");
        // Verify it's valid JSON.
        let parsed: serde_json::Value =
            serde_json::from_str(&content).expect("dump should be valid JSON");

        // Check structure.
        let trace_events = parsed["traceEvents"]
            .as_array()
            .expect("traceEvents should be an array");
        assert_eq!(trace_events.len(), 3, "should have 3 events");

        assert_eq!(trace_events[0]["name"], "block_alloc");
        assert_eq!(trace_events[1]["name"], "block_alloc");
        assert_eq!(trace_events[2]["name"], "block_free");

        let metadata = &parsed["metadata"];
        assert_eq!(metadata["peak_blocks"], 2);
        assert_eq!(metadata["total_allocs"], 2);
        assert_eq!(metadata["total_frees"], 1);
        assert_eq!(metadata["total_blocks"], 8);

        // Clean up.
        let _ = std::fs::remove_file(&path);

        nsl_profiler_stop();
    }

    #[test]
    fn test_profiler_disabled_noop() {
        let _g = lock_and_reset();
        // Don't start profiler — calls should be no-ops.
        profiler_record_alloc(0, 1);
        profiler_record_free(0, 1);
        assert_eq!(profiler_peak(), 0);
        assert_eq!(PROFILER.total_allocs.load(Ordering::Relaxed), 0);
        assert_eq!(PROFILER.total_frees.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn test_profiler_utilization() {
        let _g = lock_and_reset();
        nsl_profiler_start(4);

        assert_eq!(profiler_utilization(), 0.0);
        profiler_record_alloc(0, 1);
        assert!((profiler_utilization() - 0.25).abs() < 1e-9);
        profiler_record_alloc(1, 1);
        assert!((profiler_utilization() - 0.50).abs() < 1e-9);

        nsl_profiler_stop();
    }
}
