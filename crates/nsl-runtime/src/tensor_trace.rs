//! M45: Time-travel tensor debugger — trace recording infrastructure.
//!
//! Provides a compact binary trace format for recording every tensor operation
//! during execution. Each op captures shape, dtype, device, and per-tensor
//! statistics (min/max/mean/std) in a fixed-size `TraceEntry` for O(1) random
//! access. The `TraceRecorder` is a thread-local singleton gated by an `active`
//! flag and a `suppress_depth` counter for `@no_trace` nesting.
//!
//! All FFI functions use `i64` parameters to match the Cranelift calling convention.

use std::sync::Mutex;
use std::time::Instant;

use crate::tensor::NslTensor;

// ── Binary trace format ─────────────────────────────────────────────────────

/// Magic bytes: "NSLT" = 0x4E534C54
pub const TRACE_MAGIC: u32 = 0x4E53_4C54;
/// Current trace format version.
pub const TRACE_VERSION: u32 = 1;

/// File header written at the start of a `.nsltrace` file.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TraceHeader {
    pub magic: u32,
    pub version: u32,
    pub timestamp: u64,
    pub num_ops: u64,
}

/// Per-tensor statistics snapshot (36 bytes).
///
/// Layout: ndim(u8) + dtype(u8) + device(u8) + _pad(u8) + shape([u32;4]) + min/max/mean/std(f32×4)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct TensorStats {
    pub ndim: u8,
    pub dtype: u8,
    pub device: u8,
    pub _pad: u8,
    pub shape: [u32; 4],
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub std: f32,
}

const _ASSERT_TENSOR_STATS_SIZE: () = {
    // 4 bytes (ndim+dtype+device+pad) + 16 bytes (shape) + 16 bytes (4×f32) = 36
    assert!(std::mem::size_of::<TensorStats>() == 36);
};

/// A single trace entry — fixed-size for O(1) random access.
///
/// 4 + 2 + 2 + 8 + 36×3 = 124 bytes
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct TraceEntry {
    pub op_id: u32,
    pub op_type: u16,
    pub flags: u16,
    pub timestamp_ns: u64,
    pub in0: TensorStats,
    pub in1: TensorStats,
    pub out: TensorStats,
}

// Public accessors for packed fields used by trace_diff (avoids unaligned reads).
impl TraceEntry {
    pub fn get_op_type(&self) -> u16 {
        // SAFETY: reading from a packed struct field — use read_unaligned.
        unsafe { std::ptr::addr_of!(self.op_type).read_unaligned() }
    }
    pub fn get_flags(&self) -> u16 {
        unsafe { std::ptr::addr_of!(self.flags).read_unaligned() }
    }
    pub fn get_timestamp_ns(&self) -> u64 {
        unsafe { std::ptr::addr_of!(self.timestamp_ns).read_unaligned() }
    }
    pub fn get_op_id(&self) -> u32 {
        unsafe { std::ptr::addr_of!(self.op_id).read_unaligned() }
    }
    pub fn get_out_min(&self) -> f32 {
        unsafe { std::ptr::addr_of!(self.out.min).read_unaligned() }
    }
    pub fn get_out_max(&self) -> f32 {
        unsafe { std::ptr::addr_of!(self.out.max).read_unaligned() }
    }
    pub fn get_out_mean(&self) -> f32 {
        unsafe { std::ptr::addr_of!(self.out.mean).read_unaligned() }
    }
    pub fn get_out_std(&self) -> f32 {
        unsafe { std::ptr::addr_of!(self.out.std).read_unaligned() }
    }
}

/// Flag bit 0: output contains NaN or Inf.
pub const FLAG_NAN_INF: u16 = 0x0001;
/// Flag bit 1: user breakpoint hit.
pub const FLAG_BREAKPOINT: u16 = 0x0002;

const _ASSERT_TRACE_ENTRY_SIZE: () = {
    assert!(std::mem::size_of::<TraceEntry>() == 124);
};

// ── TraceRecorder ───────────────────────────────────────────────────────────

/// Thread-local trace recorder.
struct TraceRecorder {
    entries: Vec<TraceEntry>,
    start_time: Instant,
    active: bool,
    suppress_depth: u32,
    next_op_id: u32,
    pending_breakpoint: bool,
}

impl TraceRecorder {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            start_time: Instant::now(),
            active: false,
            suppress_depth: 0,
            next_op_id: 0,
            pending_breakpoint: false,
        }
    }
}

thread_local! {
    static RECORDER: Mutex<TraceRecorder> = Mutex::new(TraceRecorder::new());
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/// Zero-valued stats for null tensor pointers.
fn zero_stats() -> TensorStats {
    TensorStats {
        ndim: 0,
        dtype: 0,
        device: 0,
        _pad: 0,
        shape: [0; 4],
        min: 0.0,
        max: 0.0,
        mean: 0.0,
        std: 0.0,
    }
}

/// Compute stats from an `NslTensor` pointer.
///
/// # Safety
/// `ptr` must point to a valid `NslTensor` on the current device. Currently
/// only CPU tensors (device == 0) are summarized — GPU tensors get zero stats
/// (GPU reduction kernel deferred to M45b).
unsafe fn compute_stats(ptr: *const NslTensor) -> TensorStats {
    if ptr.is_null() {
        return zero_stats();
    }
    let t = &*ptr;
    let ndim = t.ndim as u8;
    let dtype = t.dtype as u8;
    let device = t.device;
    let mut shape = [0u32; 4];
    let dims = std::cmp::min(t.ndim as usize, 4);
    for (i, s) in shape.iter_mut().enumerate().take(dims) {
        *s = *t.shape.add(i) as u32;
    }

    // Only compute stats for CPU f64 tensors (dtype 0, device 0).
    // For other dtypes/devices, return zero stats with metadata.
    let len = t.len as usize;
    if device != 0 || len == 0 || t.data.is_null() {
        return TensorStats {
            ndim,
            dtype,
            device,
            _pad: 0,
            shape,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            std: 0.0,
        };
    }

    // CPU f64 path (dtype 0).
    let data = t.data as *const f64;
    let first = *data;
    let mut min_val = first;
    let mut max_val = first;
    let mut sum = 0.0_f64;
    for i in 0..len {
        let v = *data.add(i);
        if v < min_val {
            min_val = v;
        }
        if v > max_val {
            max_val = v;
        }
        sum += v;
    }
    let mean_val = sum / len as f64;

    // Standard deviation (population).
    let mut var_sum = 0.0_f64;
    for i in 0..len {
        let diff = *data.add(i) - mean_val;
        var_sum += diff * diff;
    }
    let std_val = (var_sum / len as f64).sqrt();

    TensorStats {
        ndim,
        dtype,
        device,
        _pad: 0,
        shape,
        min: min_val as f32,
        max: max_val as f32,
        mean: mean_val as f32,
        std: std_val as f32,
    }
}

/// Check whether a `TensorStats` contains NaN or Inf values.
fn has_nan_inf(stats: &TensorStats) -> bool {
    !stats.min.is_finite() || !stats.max.is_finite() || !stats.mean.is_finite()
}

// ── FFI functions ───────────────────────────────────────────────────────────
//
// All parameters are i64 to match the Cranelift calling convention.

/// Initialize the trace recorder. Call once before any ops.
#[no_mangle]
pub extern "C" fn nsl_trace_init() -> i64 {
    RECORDER.with(|r| {
        let mut rec = r.lock().unwrap();
        rec.entries.clear();
        rec.start_time = Instant::now();
        rec.active = true;
        rec.suppress_depth = 0;
        rec.next_op_id = 0;
        rec.pending_breakpoint = false;
    });
    0
}

/// Record a single tensor operation.
///
/// Returns 0 (continue) or 1 (NaN/Inf detected — break).
#[no_mangle]
pub extern "C" fn nsl_trace_record_op(
    op_type: i64,
    in0_ptr: i64,
    in1_ptr: i64,
    out_ptr: i64,
) -> i64 {
    RECORDER.with(|r| {
        let mut rec = r.lock().unwrap();
        if !rec.active || rec.suppress_depth > 0 {
            return 0;
        }

        let in0 = unsafe { compute_stats(in0_ptr as *const NslTensor) };
        let in1 = unsafe { compute_stats(in1_ptr as *const NslTensor) };
        let out = unsafe { compute_stats(out_ptr as *const NslTensor) };

        let mut flags: u16 = 0;
        if has_nan_inf(&out) {
            flags |= FLAG_NAN_INF;
        }
        if rec.pending_breakpoint {
            flags |= FLAG_BREAKPOINT;
            rec.pending_breakpoint = false;
        }

        let op_id = rec.next_op_id;
        rec.next_op_id += 1;
        let timestamp_ns = rec.start_time.elapsed().as_nanos() as u64;

        rec.entries.push(TraceEntry {
            op_id,
            op_type: op_type as u16,
            flags,
            timestamp_ns,
            in0,
            in1,
            out,
        });

        if flags & FLAG_NAN_INF != 0 {
            1
        } else {
            0
        }
    })
}

/// Increment suppress depth (`@no_trace` scope enter).
#[no_mangle]
pub extern "C" fn nsl_trace_suppress() -> i64 {
    RECORDER.with(|r| {
        let mut rec = r.lock().unwrap();
        rec.suppress_depth += 1;
    });
    0
}

/// Decrement suppress depth (`@no_trace` scope exit).
#[no_mangle]
pub extern "C" fn nsl_trace_unsuppress() -> i64 {
    RECORDER.with(|r| {
        let mut rec = r.lock().unwrap();
        rec.suppress_depth = rec.suppress_depth.saturating_sub(1);
    });
    0
}

/// Mark a breakpoint flag on the next recorded op.
#[no_mangle]
pub extern "C" fn nsl_trace_breakpoint() -> i64 {
    RECORDER.with(|r| {
        let mut rec = r.lock().unwrap();
        rec.pending_breakpoint = true;
    });
    0
}

/// Flush recorded trace to `trace.nsltrace` in the current directory.
#[no_mangle]
pub extern "C" fn nsl_trace_flush() -> i64 {
    use std::io::Write;

    RECORDER.with(|r| {
        let rec = r.lock().unwrap();
        if rec.entries.is_empty() {
            return 0;
        }

        let path = "trace.nsltrace";
        let Ok(mut file) = std::fs::File::create(path) else {
            eprintln!("[nsl-trace] failed to create {path}");
            return -1;
        };

        let header = TraceHeader {
            magic: TRACE_MAGIC,
            version: TRACE_VERSION,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_secs())
                .unwrap_or(0),
            num_ops: rec.entries.len() as u64,
        };

        // Write header.
        let header_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &header as *const TraceHeader as *const u8,
                std::mem::size_of::<TraceHeader>(),
            )
        };
        if file.write_all(header_bytes).is_err() {
            eprintln!("[nsl-trace] write header failed");
            return -1;
        }

        // Write entries.
        let entries_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                rec.entries.as_ptr() as *const u8,
                rec.entries.len() * std::mem::size_of::<TraceEntry>(),
            )
        };
        if file.write_all(entries_bytes).is_err() {
            eprintln!("[nsl-trace] write entries failed");
            return -1;
        }

        0
    })
}

/// Called after each traced op when trace_ops is enabled.
/// Prints a warning to stderr if nan_flag == 1 (NaN/Inf detected in output).
#[no_mangle]
pub extern "C" fn nsl_trace_nan_warning(nan_flag: i64, op_type: i64) -> i64 {
    if nan_flag == 1 {
        let op_name = match op_type {
            0 => "add",
            1 => "sub",
            2 => "mul",
            3 => "div",
            4 => "matmul",
            5 => "relu",
            6 => "sigmoid",
            7 => "softmax",
            8 => "sum",
            9 => "mean",
            _ => "unknown",
        };
        eprintln!(
            "[nsl] WARNING: NaN/Inf detected in output of '{}' (op_type={})",
            op_name, op_type
        );
        eprintln!("[nsl]   Run with: nsl debug <trace_file> --find-nan for details");
    }
    0
}

/// Destroy the trace recorder, freeing all buffered data.
#[no_mangle]
pub extern "C" fn nsl_trace_destroy() -> i64 {
    RECORDER.with(|r| {
        let mut rec = r.lock().unwrap();
        rec.entries.clear();
        rec.entries.shrink_to_fit();
        rec.active = false;
        rec.suppress_depth = 0;
        rec.next_op_id = 0;
        rec.pending_breakpoint = false;
    });
    0
}

// ── Public Rust API (for trace_diff and tests) ──────────────────────────────

/// Return a clone of all recorded entries (test/utility helper).
pub fn get_entries() -> Vec<TraceEntry> {
    RECORDER.with(|r| r.lock().unwrap().entries.clone())
}

/// Return current entry count (test helper).
pub fn entry_count() -> usize {
    RECORDER.with(|r| r.lock().unwrap().entries.len())
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::c_void;
    use std::mem::size_of;
    use std::sync::atomic::AtomicI64;

    #[test]
    fn trace_entry_size() {
        assert_eq!(size_of::<TraceEntry>(), 124);
    }

    #[test]
    fn trace_header_fields() {
        let h = TraceHeader {
            magic: TRACE_MAGIC,
            version: TRACE_VERSION,
            timestamp: 1234567890,
            num_ops: 42,
        };
        assert_eq!(h.magic, 0x4E53_4C54);
        assert_eq!(h.version, 1);
        assert_eq!(h.timestamp, 1234567890);
        assert_eq!(h.num_ops, 42);
    }

    #[test]
    fn record_basic() {
        // Reset recorder state for this test.
        nsl_trace_destroy();
        nsl_trace_init();

        // Record 3 ops with null tensors.
        nsl_trace_record_op(0, 0, 0, 0);
        nsl_trace_record_op(1, 0, 0, 0);
        nsl_trace_record_op(2, 0, 0, 0);

        assert_eq!(entry_count(), 3);
        let entries = get_entries();
        assert_eq!(entries[0].get_op_id(), 0);
        assert_eq!(entries[1].get_op_id(), 1);
        assert_eq!(entries[2].get_op_id(), 2);
        assert_eq!(entries[0].get_op_type(), 0);
        assert_eq!(entries[1].get_op_type(), 1);
        assert_eq!(entries[2].get_op_type(), 2);

        nsl_trace_destroy();
    }

    #[test]
    fn nan_detection() {
        nsl_trace_destroy();
        nsl_trace_init();

        // Create a tensor with NaN in it.
        let mut data = [1.0, f64::NAN, 3.0];
        let mut shape_val: i64 = 3;
        let mut stride_val: i64 = 1;
        let t = NslTensor::new(
            data.as_mut_ptr() as *mut c_void,
            &mut shape_val as *mut i64,
            &mut stride_val as *mut i64,
            1,
            3,
            0,
            0,
            0,
            0,
        );

        let ret = nsl_trace_record_op(0, 0, 0, &t as *const NslTensor as i64);
        assert_eq!(ret, 1, "should return 1 (NaN break)");

        let entries = get_entries();
        assert_eq!(entries.len(), 1);
        assert_ne!(entries[0].get_flags() & FLAG_NAN_INF, 0, "NaN flag should be set");

        nsl_trace_destroy();
    }

    #[test]
    fn suppress_blocks_recording() {
        nsl_trace_destroy();
        nsl_trace_init();

        nsl_trace_record_op(0, 0, 0, 0);
        nsl_trace_suppress();
        nsl_trace_record_op(1, 0, 0, 0); // suppressed
        nsl_trace_unsuppress();
        nsl_trace_record_op(2, 0, 0, 0);

        assert_eq!(entry_count(), 2);

        nsl_trace_destroy();
    }

    #[test]
    fn suppress_nesting() {
        nsl_trace_destroy();
        nsl_trace_init();

        nsl_trace_suppress(); // depth 1
        nsl_trace_suppress(); // depth 2
        nsl_trace_record_op(0, 0, 0, 0); // suppressed
        nsl_trace_unsuppress(); // depth 1 (still suppressed)
        nsl_trace_record_op(1, 0, 0, 0); // still suppressed
        nsl_trace_unsuppress(); // depth 0
        nsl_trace_record_op(2, 0, 0, 0); // recorded

        assert_eq!(entry_count(), 1);

        nsl_trace_destroy();
    }

    #[test]
    fn breakpoint_flag() {
        nsl_trace_destroy();
        nsl_trace_init();

        nsl_trace_record_op(0, 0, 0, 0);
        nsl_trace_breakpoint();
        nsl_trace_record_op(1, 0, 0, 0);
        nsl_trace_record_op(2, 0, 0, 0); // breakpoint consumed

        let entries = get_entries();
        assert_eq!(entries.len(), 3);
        assert_eq!(entries[0].get_flags() & FLAG_BREAKPOINT, 0);
        assert_ne!(entries[1].get_flags() & FLAG_BREAKPOINT, 0);
        assert_eq!(entries[2].get_flags() & FLAG_BREAKPOINT, 0);

        nsl_trace_destroy();
    }

    #[test]
    fn ffi_lifecycle() {
        // Full lifecycle: init → record → flush → destroy.
        nsl_trace_destroy();

        let ret = nsl_trace_init();
        assert_eq!(ret, 0);

        nsl_trace_record_op(0, 0, 0, 0);
        assert_eq!(entry_count(), 1);

        let ret = nsl_trace_flush();
        assert_eq!(ret, 0);

        let ret = nsl_trace_destroy();
        assert_eq!(ret, 0);
        assert_eq!(entry_count(), 0);

        // Clean up the trace file if it was created.
        let _ = std::fs::remove_file("trace.nsltrace");
    }

    #[test]
    fn nan_warning_returns_zero() {
        // nsl_trace_nan_warning should always return 0 (no-op return).
        // When nan_flag == 0, it should not print anything.
        assert_eq!(nsl_trace_nan_warning(0, 0), 0);
        assert_eq!(nsl_trace_nan_warning(0, 255), 0);
        // When nan_flag == 1, it prints a warning but still returns 0.
        assert_eq!(nsl_trace_nan_warning(1, 4), 0); // matmul
        assert_eq!(nsl_trace_nan_warning(1, 255), 0); // unknown op
    }
}
