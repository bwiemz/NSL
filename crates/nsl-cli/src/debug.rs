//! M45b: Tensor Debugger CLI — trace file reader, NaN finder, diff, Chrome export.
//!
//! This module duplicates the binary format structs from `nsl-runtime::tensor_trace`
//! so the CLI can read `.nsltrace` files without depending on `nsl-runtime`. The
//! structs are `#[repr(C, packed)]` and all field access goes through safe accessor
//! methods that use `read_unaligned` to avoid undefined behavior.

use std::io::{self, Read};
use std::path::Path;

// ── Binary format constants ─────────────────────────────────────────────────

/// Magic bytes: "NSLT" = 0x4E534C54
const TRACE_MAGIC: u32 = 0x4E53_4C54;

// ── Binary format structs ───────────────────────────────────────────────────
// These must match the layout in nsl-runtime/src/tensor_trace.rs byte-for-byte.

/// File header written at the start of a `.nsltrace` file.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TraceHeader {
    magic: u32,
    version: u32,
    timestamp: u64,
    num_ops: u64,
}

impl TraceHeader {
    pub fn num_ops(&self) -> u64 {
        self.num_ops
    }

    pub fn version(&self) -> u32 {
        self.version
    }

    pub fn timestamp(&self) -> u64 {
        self.timestamp
    }
}

/// Per-tensor statistics snapshot (36 bytes).
///
/// Layout: ndim(u8) + dtype(u8) + device(u8) + _pad(u8) + shape(\[u32;4\]) + min/max/mean/std(f32x4)
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct TensorStats {
    ndim: u8,
    dtype: u8,
    device: u8,
    _pad: u8,
    shape: [u32; 4],
    min: f32,
    max: f32,
    mean: f32,
    std: f32,
}

const _ASSERT_TENSOR_STATS_SIZE: () = {
    assert!(std::mem::size_of::<TensorStats>() == 36);
};

/// A single trace entry — fixed-size for O(1) random access.
///
/// 4 + 2 + 2 + 8 + 36x3 = 124 bytes
#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct TraceEntry {
    op_id: u32,
    op_type: u16,
    flags: u16,
    timestamp_ns: u64,
    in0: TensorStats,
    in1: TensorStats,
    out: TensorStats,
}

const _ASSERT_TRACE_ENTRY_SIZE: () = {
    assert!(std::mem::size_of::<TraceEntry>() == 124);
};

// ── Safe accessor methods (read_unaligned to avoid UB on packed fields) ─────

impl TraceEntry {
    pub fn op_id(&self) -> u32 {
        unsafe { std::ptr::addr_of!(self.op_id).read_unaligned() }
    }

    pub fn op_type(&self) -> u16 {
        unsafe { std::ptr::addr_of!(self.op_type).read_unaligned() }
    }

    pub fn flags(&self) -> u16 {
        unsafe { std::ptr::addr_of!(self.flags).read_unaligned() }
    }

    pub fn timestamp_ns(&self) -> u64 {
        unsafe { std::ptr::addr_of!(self.timestamp_ns).read_unaligned() }
    }

    pub fn out_min(&self) -> f32 {
        unsafe { std::ptr::addr_of!(self.out.min).read_unaligned() }
    }

    pub fn out_max(&self) -> f32 {
        unsafe { std::ptr::addr_of!(self.out.max).read_unaligned() }
    }

    pub fn out_mean(&self) -> f32 {
        unsafe { std::ptr::addr_of!(self.out.mean).read_unaligned() }
    }

    pub fn out_std(&self) -> f32 {
        unsafe { std::ptr::addr_of!(self.out.std).read_unaligned() }
    }
}

/// NaN/Inf flag (bit 0).
const FLAG_NAN_INF: u16 = 0x0001;

// ── Trace file reader ───────────────────────────────────────────────────────

/// Read a binary `.nsltrace` file into a header and vector of entries.
pub fn read_trace_file(path: &Path) -> io::Result<(TraceHeader, Vec<TraceEntry>)> {
    let mut file = std::fs::File::open(path)?;

    // Read header.
    let mut header_bytes = [0u8; std::mem::size_of::<TraceHeader>()];
    file.read_exact(&mut header_bytes)?;
    let header =
        unsafe { std::ptr::read_unaligned(header_bytes.as_ptr() as *const TraceHeader) };

    if header.magic != TRACE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "not a valid .nsltrace file (bad magic)",
        ));
    }

    // Read entries.
    let entry_size = std::mem::size_of::<TraceEntry>();
    let num_ops = header.num_ops() as usize;
    let mut entries = Vec::with_capacity(num_ops);
    let mut entry_bytes = vec![0u8; entry_size];
    for _ in 0..num_ops {
        file.read_exact(&mut entry_bytes)?;
        let entry =
            unsafe { std::ptr::read_unaligned(entry_bytes.as_ptr() as *const TraceEntry) };
        entries.push(entry);
    }

    Ok((header, entries))
}

// ── Trace diffing ───────────────────────────────────────────────────────────

/// Result of comparing two traces.
#[derive(Debug)]
pub enum DiffResult {
    /// Traces are identical within threshold.
    Identical,
    /// Op type sequence diverges at the given positions.
    OpMismatch { pos_a: usize, pos_b: usize },
    /// Stats diverge beyond threshold at the given op.
    StatsDiverge {
        op_id: usize,
        delta_mean: f32,
        delta_max: f32,
    },
    /// Traces have different lengths.
    LengthMismatch { len_a: usize, len_b: usize },
}

/// Find the first point of divergence between two traces.
///
/// Compares entry-by-entry: first checks op_type match, then checks whether
/// the output stats (mean, max) diverge beyond `threshold`.
pub fn find_first_divergence(
    a: &[TraceEntry],
    b: &[TraceEntry],
    threshold: f32,
) -> DiffResult {
    if a.len() != b.len() {
        return DiffResult::LengthMismatch {
            len_a: a.len(),
            len_b: b.len(),
        };
    }
    for i in 0..a.len() {
        if a[i].op_type() != b[i].op_type() {
            return DiffResult::OpMismatch {
                pos_a: i,
                pos_b: i,
            };
        }
        let delta_mean = (a[i].out_mean() - b[i].out_mean()).abs();
        let delta_max = (a[i].out_max() - b[i].out_max()).abs();
        if delta_mean > threshold || delta_max > threshold {
            return DiffResult::StatsDiverge {
                op_id: i,
                delta_mean,
                delta_max,
            };
        }
    }
    DiffResult::Identical
}

// ── Chrome Tracing Export ───────────────────────────────────────────────────

/// Default op name table mapping op_type indices to human-readable names.
const OP_NAMES: &[&str] = &[
    "add", "sub", "mul", "div", "matmul", "relu", "sigmoid", "softmax", "sum", "mean",
];

/// Export a trace to Chrome Trace Event Format JSON.
///
/// `op_names` maps op_type indices to human-readable names. Op types beyond
/// the array length are labeled "unknown". NaN/Inf stats are clamped to 0.0
/// for valid JSON output (NaN is not valid in JSON).
pub fn export_chrome_json(entries: &[TraceEntry], op_names: &[&str]) -> String {
    let mut events = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        let op_type = entry.op_type() as usize;
        let name = if op_type < op_names.len() {
            op_names[op_type]
        } else {
            "unknown"
        };
        let ts_ns = entry.timestamp_ns();
        let dur_ns = if i + 1 < entries.len() {
            entries[i + 1].timestamp_ns().saturating_sub(ts_ns)
        } else {
            0
        };
        // Guard NaN/Inf -> 0.0 for valid JSON.
        let safe = |v: f32| -> f32 {
            if v.is_finite() {
                v
            } else {
                0.0
            }
        };
        let has_nan = entry.flags() & FLAG_NAN_INF != 0;
        events.push(format!(
            r#"{{"name":"{}","cat":"tensor_op","ph":"X","ts":{},"dur":{},"pid":0,"tid":0,"args":{{"op_id":{},"out_min":{},"out_max":{},"out_mean":{},"out_std":{},"has_nan":{}}}}}"#,
            name,
            ts_ns / 1000,
            dur_ns / 1000,
            entry.op_id(),
            safe(entry.out_min()),
            safe(entry.out_max()),
            safe(entry.out_mean()),
            safe(entry.out_std()),
            has_nan,
        ));
    }
    format!(r#"{{"traceEvents":[{}]}}"#, events.join(","))
}

/// Run the debug command: read trace, print summary, optionally find NaN,
/// diff, or export Chrome JSON.
pub fn run_debug(
    file: &Path,
    find_nan: bool,
    diff: Option<&Path>,
    export_chrome: Option<&Path>,
) {
    let (header, entries) = read_trace_file(file).unwrap_or_else(|e| {
        eprintln!("error: failed to read trace '{}': {e}", file.display());
        std::process::exit(1);
    });

    println!(
        "Trace: {} ({} ops, version {}, timestamp {})",
        file.display(),
        entries.len(),
        header.version(),
        header.timestamp(),
    );

    if find_nan {
        let mut found = false;
        for entry in &entries {
            if entry.flags() & FLAG_NAN_INF != 0 {
                println!(
                    "NaN/Inf detected at op #{}: type={}",
                    entry.op_id(),
                    entry.op_type()
                );
                println!(
                    "  Output: min={}, max={}, mean={}, std={}",
                    entry.out_min(),
                    entry.out_max(),
                    entry.out_mean(),
                    entry.out_std()
                );
                found = true;
                break;
            }
        }
        if !found {
            println!("No NaN/Inf detected in trace.");
        }
    }

    if let Some(other_path) = diff {
        let (_, other_entries) = read_trace_file(other_path).unwrap_or_else(|e| {
            eprintln!(
                "error: failed to read diff trace '{}': {e}",
                other_path.display()
            );
            std::process::exit(1);
        });
        match find_first_divergence(&entries, &other_entries, 1e-5) {
            DiffResult::Identical => println!("Traces are identical."),
            DiffResult::OpMismatch { pos_a, pos_b } => {
                println!("Op mismatch at position {pos_a} vs {pos_b}");
            }
            DiffResult::StatsDiverge {
                op_id,
                delta_mean,
                delta_max,
            } => {
                println!(
                    "Stats diverge at op #{op_id}: delta_mean={delta_mean}, delta_max={delta_max}"
                );
            }
            DiffResult::LengthMismatch { len_a, len_b } => {
                println!("Length mismatch: {len_a} vs {len_b}");
            }
        }
    }

    if let Some(chrome_path) = export_chrome {
        let json = export_chrome_json(&entries, OP_NAMES);
        std::fs::write(chrome_path, json).unwrap_or_else(|e| {
            eprintln!(
                "error: failed to write Chrome JSON '{}': {e}",
                chrome_path.display()
            );
            std::process::exit(1);
        });
        println!("Exported Chrome tracing to {}", chrome_path.display());
    }

    // Default: show summary when no specific action requested.
    if !find_nan && diff.is_none() && export_chrome.is_none() {
        let nan_count = entries
            .iter()
            .filter(|e| e.flags() & FLAG_NAN_INF != 0)
            .count();
        println!("  NaN/Inf ops: {nan_count}");
        if !entries.is_empty() {
            let first = &entries[0];
            let last = entries.last().unwrap();
            println!("  First op: id={}, type={}", first.op_id(), first.op_type());
            println!("  Last op:  id={}, type={}", last.op_id(), last.op_type());
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

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

    fn make_entry(op_id: u32, op_type: u16, flags: u16, out_mean: f32, out_max: f32) -> TraceEntry {
        let mut out = zero_stats();
        out.mean = out_mean;
        out.max = out_max;
        TraceEntry {
            op_id,
            op_type,
            flags,
            timestamp_ns: op_id as u64 * 1000,
            in0: zero_stats(),
            in1: zero_stats(),
            out,
        }
    }

    fn write_trace_file(path: &Path, entries: &[TraceEntry]) {
        let mut file = std::fs::File::create(path).unwrap();
        let header = TraceHeader {
            magic: TRACE_MAGIC,
            version: 1,
            timestamp: 1234567890,
            num_ops: entries.len() as u64,
        };
        let header_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                &header as *const TraceHeader as *const u8,
                std::mem::size_of::<TraceHeader>(),
            )
        };
        file.write_all(header_bytes).unwrap();
        let entries_bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(
                entries.as_ptr() as *const u8,
                std::mem::size_of_val(entries),
            )
        };
        file.write_all(entries_bytes).unwrap();
    }

    #[test]
    fn struct_sizes_match_runtime() {
        assert_eq!(std::mem::size_of::<TraceHeader>(), 24);
        assert_eq!(std::mem::size_of::<TensorStats>(), 36);
        assert_eq!(std::mem::size_of::<TraceEntry>(), 124);
    }

    #[test]
    fn read_trace_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.nsltrace");

        let entries = vec![
            make_entry(0, 1, 0, 1.5, 3.0),
            make_entry(1, 2, 0, 2.5, 4.0),
            make_entry(2, 0, FLAG_NAN_INF, f32::NAN, f32::INFINITY),
        ];
        write_trace_file(&path, &entries);

        let (header, read_entries) = read_trace_file(&path).unwrap();
        assert_eq!(header.num_ops(), 3);
        assert_eq!(read_entries.len(), 3);
        assert_eq!(read_entries[0].op_id(), 0);
        assert_eq!(read_entries[0].op_type(), 1);
        assert_eq!(read_entries[1].op_id(), 1);
        assert_eq!(read_entries[1].op_type(), 2);
        assert_eq!(read_entries[2].flags() & FLAG_NAN_INF, FLAG_NAN_INF);
    }

    #[test]
    fn read_trace_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.nsltrace");

        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&[0u8; 24]).unwrap(); // All zeros — bad magic.

        let result = read_trace_file(&path);
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn diff_identical() {
        let a = vec![make_entry(0, 1, 0, 1.0, 2.0), make_entry(1, 2, 0, 3.0, 4.0)];
        let b = vec![make_entry(0, 1, 0, 1.0, 2.0), make_entry(1, 2, 0, 3.0, 4.0)];
        match find_first_divergence(&a, &b, 0.01) {
            DiffResult::Identical => {}
            other => panic!("expected Identical, got {:?}", other),
        }
    }

    #[test]
    fn diff_op_mismatch() {
        let a = vec![make_entry(0, 1, 0, 1.0, 2.0)];
        let b = vec![make_entry(0, 99, 0, 1.0, 2.0)];
        match find_first_divergence(&a, &b, 0.01) {
            DiffResult::OpMismatch { pos_a, pos_b } => {
                assert_eq!(pos_a, 0);
                assert_eq!(pos_b, 0);
            }
            other => panic!("expected OpMismatch, got {:?}", other),
        }
    }

    #[test]
    fn diff_stats_diverge() {
        let a = vec![make_entry(0, 1, 0, 1.0, 2.0)];
        let b = vec![make_entry(0, 1, 0, 5.0, 2.0)];
        match find_first_divergence(&a, &b, 0.5) {
            DiffResult::StatsDiverge {
                op_id,
                delta_mean,
                ..
            } => {
                assert_eq!(op_id, 0);
                assert!((delta_mean - 4.0).abs() < 0.01);
            }
            other => panic!("expected StatsDiverge, got {:?}", other),
        }
    }

    #[test]
    fn diff_length_mismatch() {
        let a = vec![make_entry(0, 1, 0, 1.0, 2.0)];
        let b = vec![
            make_entry(0, 1, 0, 1.0, 2.0),
            make_entry(1, 2, 0, 3.0, 4.0),
        ];
        match find_first_divergence(&a, &b, 0.01) {
            DiffResult::LengthMismatch { len_a, len_b } => {
                assert_eq!(len_a, 1);
                assert_eq!(len_b, 2);
            }
            other => panic!("expected LengthMismatch, got {:?}", other),
        }
    }

    #[test]
    fn chrome_export_format() {
        let entries = vec![
            make_entry(0, 0, 0, 1.5, 3.0),
            make_entry(1, 1, FLAG_NAN_INF, f32::NAN, f32::INFINITY),
        ];
        let names: &[&str] = &["add", "mul"];
        let json = export_chrome_json(&entries, names);

        assert!(json.starts_with(r#"{"traceEvents":["#));
        assert!(json.ends_with("]}"));
        assert!(json.contains(r#""name":"add""#));
        assert!(json.contains(r#""name":"mul""#));
        // NaN clamped to 0.
        assert!(json.contains(r#""out_mean":0"#));
        // has_nan should be false for the first entry (flags=0).
        assert!(json.contains(r#""has_nan":false"#));
        // has_nan should be true for the second entry (flags=FLAG_NAN_INF).
        assert!(json.contains(r#""has_nan":true"#));
    }

    #[test]
    fn chrome_export_unknown_op() {
        let entries = vec![make_entry(0, 255, 0, 1.0, 2.0)];
        let names: &[&str] = &["add"];
        let json = export_chrome_json(&entries, names);
        assert!(json.contains(r#""name":"unknown""#));
    }
}
