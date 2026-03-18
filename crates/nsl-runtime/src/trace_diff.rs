//! M45: Trace diffing and Chrome tracing export.
//!
//! Provides utilities for comparing two traces entry-by-entry and exporting
//! trace data to Chrome Trace Event Format JSON for visualization in
//! `chrome://tracing`.

use crate::tensor_trace::TraceEntry;

// ── DiffResult ──────────────────────────────────────────────────────────────

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

// ── Diffing ─────────────────────────────────────────────────────────────────

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
        if a[i].get_op_type() != b[i].get_op_type() {
            return DiffResult::OpMismatch {
                pos_a: i,
                pos_b: i,
            };
        }
        let delta_mean = (a[i].get_out_mean() - b[i].get_out_mean()).abs();
        let delta_max = (a[i].get_out_max() - b[i].get_out_max()).abs();
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

/// Export a trace to Chrome Trace Event Format JSON.
///
/// `op_names` maps op_type indices to human-readable names. Op types beyond
/// the array length are labeled "unknown". NaN/Inf stats are clamped to 0.0
/// for valid JSON output.
pub fn export_chrome_json(entries: &[TraceEntry], op_names: &[&str]) -> String {
    let mut events = Vec::with_capacity(entries.len());
    for (i, entry) in entries.iter().enumerate() {
        let op_type = entry.get_op_type() as usize;
        let name = if op_type < op_names.len() {
            op_names[op_type]
        } else {
            "unknown"
        };
        let ts_ns = entry.get_timestamp_ns();
        let dur_ns = if i + 1 < entries.len() {
            entries[i + 1].get_timestamp_ns().saturating_sub(ts_ns)
        } else {
            0
        };
        // Guard NaN/Inf → 0.0 for valid JSON (NaN is not valid in JSON).
        let safe = |v: f32| -> f32 {
            if v.is_finite() {
                v
            } else {
                0.0
            }
        };
        let has_nan = entry.get_flags() & 0x01 != 0;
        events.push(format!(
            r#"{{"name":"{}","cat":"tensor_op","ph":"X","ts":{},"dur":{},"pid":0,"tid":0,"args":{{"op_id":{},"out_min":{},"out_max":{},"out_mean":{},"out_std":{},"has_nan":{}}}}}"#,
            name,
            ts_ns / 1000,
            dur_ns / 1000,
            entry.get_op_id(),
            safe(entry.get_out_min()),
            safe(entry.get_out_max()),
            safe(entry.get_out_mean()),
            safe(entry.get_out_std()),
            has_nan,
        ));
    }
    format!(r#"{{"traceEvents":[{}]}}"#, events.join(","))
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor_trace::TensorStats;

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

    fn make_entry(op_id: u32, op_type: u16, out_mean: f32, out_max: f32) -> TraceEntry {
        let mut out = zero_stats();
        out.mean = out_mean;
        out.max = out_max;
        out.min = 0.0;
        out.std = 0.0;
        TraceEntry {
            op_id,
            op_type,
            flags: 0,
            timestamp_ns: op_id as u64 * 1000,
            in0: zero_stats(),
            in1: zero_stats(),
            out,
        }
    }

    #[test]
    fn diff_identical() {
        let a = vec![make_entry(0, 1, 1.0, 2.0), make_entry(1, 2, 3.0, 4.0)];
        let b = vec![make_entry(0, 1, 1.0, 2.0), make_entry(1, 2, 3.0, 4.0)];
        match find_first_divergence(&a, &b, 0.01) {
            DiffResult::Identical => {}
            other => panic!("expected Identical, got {:?}", other),
        }
    }

    #[test]
    fn diff_op_mismatch() {
        let a = vec![make_entry(0, 1, 1.0, 2.0), make_entry(1, 2, 3.0, 4.0)];
        let b = vec![make_entry(0, 1, 1.0, 2.0), make_entry(1, 99, 3.0, 4.0)];
        match find_first_divergence(&a, &b, 0.01) {
            DiffResult::OpMismatch { pos_a, pos_b } => {
                assert_eq!(pos_a, 1);
                assert_eq!(pos_b, 1);
            }
            other => panic!("expected OpMismatch, got {:?}", other),
        }
    }

    #[test]
    fn diff_stats_diverge() {
        let a = vec![make_entry(0, 1, 1.0, 2.0)];
        let b = vec![make_entry(0, 1, 5.0, 2.0)]; // mean differs by 4.0
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
    fn chrome_export_format() {
        let entries = vec![
            make_entry(0, 0, 1.5, 3.0),
            make_entry(1, 1, f32::NAN, f32::INFINITY), // NaN/Inf → 0.0 in JSON
        ];
        let names = ["add", "mul"];
        let json = export_chrome_json(&entries, &names);

        // Must be valid JSON-ish.
        assert!(json.starts_with(r#"{"traceEvents":["#));
        assert!(json.ends_with("]}"));
        // Op names.
        assert!(json.contains(r#""name":"add""#));
        assert!(json.contains(r#""name":"mul""#));
        // NaN clamped to 0.
        assert!(json.contains(r#""out_mean":0"#));
        // has_nan should be false for the first entry.
        assert!(json.contains(r#""has_nan":false"#));
    }
}
