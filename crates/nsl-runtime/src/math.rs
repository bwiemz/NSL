#[no_mangle]
pub extern "C" fn nsl_sqrt(x: f64) -> f64 { x.sqrt() }

#[no_mangle]
pub extern "C" fn nsl_log(x: f64) -> f64 { x.ln() }

#[no_mangle]
pub extern "C" fn nsl_exp(x: f64) -> f64 { x.exp() }

#[no_mangle]
pub extern "C" fn nsl_sin(x: f64) -> f64 { x.sin() }

#[no_mangle]
pub extern "C" fn nsl_cos(x: f64) -> f64 { x.cos() }

#[no_mangle]
pub extern "C" fn nsl_abs_float(x: f64) -> f64 { x.abs() }

#[no_mangle]
pub extern "C" fn nsl_abs_int(x: i64) -> i64 {
    x.checked_abs().unwrap_or_else(|| {
        eprintln!("nsl: integer overflow in abs({})", x);
        std::process::abort();
    })
}

#[no_mangle]
pub extern "C" fn nsl_min_int(a: i64, b: i64) -> i64 { a.min(b) }

#[no_mangle]
pub extern "C" fn nsl_max_int(a: i64, b: i64) -> i64 { a.max(b) }

#[no_mangle]
pub extern "C" fn nsl_min_float(a: f64, b: f64) -> f64 { a.min(b) }

#[no_mangle]
pub extern "C" fn nsl_max_float(a: f64, b: f64) -> f64 { a.max(b) }

#[no_mangle]
pub extern "C" fn nsl_floor(x: f64) -> f64 { x.floor() }

/// High-resolution wall clock in seconds (monotonic).
#[no_mangle]
pub extern "C" fn nsl_clock() -> f64 {
    use std::time::Instant;
    use std::sync::OnceLock;
    static EPOCH: OnceLock<Instant> = OnceLock::new();
    let epoch = EPOCH.get_or_init(Instant::now);
    epoch.elapsed().as_secs_f64()
}

/// Synchronize the CUDA device (no-op on CPU builds). Emitted by the
/// NSL_PHASE_TIMING train-block instrumentation before each phase-boundary
/// `nsl_clock()` read: most runtime ops already end with a
/// `cuCtxSynchronize`, but a few same-stream kernels deliberately skip it,
/// so an explicit sync pins the wall-clock delta to completed GPU work.
#[no_mangle]
pub extern "C" fn nsl_cuda_device_synchronize() {
    #[cfg(feature = "cuda")]
    {
        crate::cuda::inner::ensure_context();
        unsafe { crate::cuda::inner::cu_ctx_synchronize() };
    }
}

/// NSL_PHASE_TIMING: one line per micro-batch with the forward(+loss) and
/// backward wall-clock seconds. Machine-parseable prefix `[phase]`.
#[no_mangle]
pub extern "C" fn nsl_phase_fwd_bwd_report(fwd_s: f64, bwd_s: f64) {
    println!("[phase] fwd={fwd_s:.6} bwd={bwd_s:.6}");
}

/// NSL_PHASE_TIMING: one line per OPTIMIZER step (accumulation boundaries
/// only) with the optimizer wall-clock seconds.
#[no_mangle]
pub extern "C" fn nsl_phase_optim_report(opt_s: f64) {
    println!("[phase] opt={opt_s:.6}");
}

// ---------------------------------------------------------------------------
// Allocation tracking (for benchmarking memory pressure)
// ---------------------------------------------------------------------------
use std::sync::atomic::{AtomicU64, Ordering};

static ALLOC_COUNTER: AtomicU64 = AtomicU64::new(0);
static ALLOC_BYTES: AtomicU64 = AtomicU64::new(0);

/// Increment the global allocation counter. Called from tensor creation paths.
pub fn track_alloc(bytes: usize) {
    ALLOC_COUNTER.fetch_add(1, Ordering::Relaxed);
    ALLOC_BYTES.fetch_add(bytes as u64, Ordering::Relaxed);
}

/// Reset allocation counters to zero.
#[no_mangle]
pub extern "C" fn nsl_alloc_reset() -> i64 {
    ALLOC_COUNTER.store(0, Ordering::Relaxed);
    ALLOC_BYTES.store(0, Ordering::Relaxed);
    0
}

/// Get the number of tensor allocations since last reset.
#[no_mangle]
pub extern "C" fn nsl_alloc_count() -> i64 {
    ALLOC_COUNTER.load(Ordering::Relaxed) as i64
}

/// Get the total bytes allocated since last reset.
#[no_mangle]
pub extern "C" fn nsl_alloc_bytes() -> i64 {
    ALLOC_BYTES.load(Ordering::Relaxed) as i64
}
