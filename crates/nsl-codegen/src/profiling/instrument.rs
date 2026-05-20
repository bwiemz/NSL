//! Manifest (predictions JSON) produced at compile time by the profile
//! pipeline. The runtime emits an "actual" JSON with the same kernel_id space;
//! the monitor CLI (Task 8) reads both and renders the predicted-vs-actual
//! report.
//!
//! NOTE: In Phase 1 the manifest is derived from the walker's ProfileReport —
//! not from the real codegen kernel-emit site. The cost model produces the
//! predictions either way; the only thing we're faking is the kernel_id
//! assignment, which mirrors the walker's op order.
//!
//! TODO(phase2): wire `nsl_profile_kernel_begin/end` hooks into the real PTX
//! emit path (see `crates/nsl-codegen/src/backend_ptx.rs`) and replace our
//! synthesized IDs with the real kernel IDs assigned at emit time.

use crate::profiling::types::ProfileReport;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceSpanJson {
    pub file: String,
    pub start_line: u32,
    pub end_line: u32,
}

impl SourceSpanJson {
    /// Resolve a byte-positioned Span to 1-based line numbers using the source text.
    pub fn from_span(span: nsl_errors::Span, file_name: &str, source_text: &str) -> Self {
        let start_line = line_of_byte(source_text, span.start.0);
        let end_byte = span.end.0.saturating_sub(1).max(span.start.0);
        let end_line = line_of_byte(source_text, end_byte);
        Self {
            file: file_name.into(),
            start_line,
            end_line,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KernelEntry {
    pub kernel_id: u32,
    pub op_name: String,
    pub source_span: SourceSpanJson,
    pub predicted_us: f64,
    pub predicted_flops: u64,
    pub predicted_hbm_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EliminatedOp {
    pub op_name: String,
    pub source_span: SourceSpanJson,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Manifest {
    pub target_gpu: String,
    pub dtype: String,
    pub kernels: Vec<KernelEntry>,
    pub eliminated_ops: Vec<EliminatedOp>,
}

pub struct ManifestBuilder {
    inner: Manifest,
    next_id: u32,
}

impl ManifestBuilder {
    pub fn new(target_gpu: &str, dtype: &str) -> Self {
        Self {
            inner: Manifest {
                target_gpu: target_gpu.into(),
                dtype: dtype.into(),
                kernels: vec![],
                eliminated_ops: vec![],
            },
            next_id: 0,
        }
    }

    /// Reserve the next kernel_id without recording a KernelEntry.
    /// Used by codegen so the same id can be emitted as a Cranelift constant
    /// in begin/end hook calls, and the entry recorded in the same call site.
    pub fn reserve_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Record a KernelEntry at a caller-supplied id.
    pub fn record_kernel_at(
        &mut self,
        id: u32,
        op_name: &str,
        span: SourceSpanJson,
        predicted_us: f64,
        predicted_flops: u64,
        predicted_hbm_bytes: u64,
    ) {
        self.inner.kernels.push(KernelEntry {
            kernel_id: id,
            op_name: op_name.into(),
            source_span: span,
            predicted_us,
            predicted_flops,
            predicted_hbm_bytes,
        });
    }

    pub fn record_kernel(
        &mut self,
        op_name: &str,
        span: SourceSpanJson,
        us: f64,
        flops: u64,
        hbm: u64,
    ) -> u32 {
        let id = self.reserve_id();
        self.record_kernel_at(id, op_name, span, us, flops, hbm);
        id
    }

    pub fn record_eliminated(&mut self, op_name: &str, span: SourceSpanJson, reason: &str) {
        self.inner.eliminated_ops.push(EliminatedOp {
            op_name: op_name.into(),
            source_span: span,
            reason: reason.into(),
        });
    }

    pub fn finish(self) -> Manifest {
        self.inner
    }
}

/// Convert the walker's compile-time report into the manifest consumed by
/// the monitor CLI. Ops with `flops == 0` OR `fused == true` are recorded as
/// eliminated. Others become kernels with dense IDs in walker-order.
pub fn build_manifest_from_report(
    report: &ProfileReport,
    source_file: &str,
    source_text: &str,
) -> Manifest {
    let mut b = ManifestBuilder::new(&report.target_gpu, &report.dtype);
    for op in &report.ops {
        let span = span_from_op_loc(&op.loc, source_file, source_text);
        if op.fused {
            b.record_eliminated(&op.name, span, "fused into prior kernel");
        } else if op.flops == 0 {
            b.record_eliminated(&op.name, span, "unknown op (zero cost)");
        } else {
            b.record_kernel(
                &op.name,
                span,
                op.estimated_time_us,
                op.flops,
                op.bytes_read + op.bytes_written,
            );
        }
    }
    b.finish()
}

pub fn write_manifest(path: &std::path::Path, m: &Manifest) -> std::io::Result<()> {
    let s = serde_json::to_string_pretty(m).map_err(std::io::Error::other)?;
    std::fs::write(path, s)
}

/// Parse a walker `loc` string (format: `"<file_id>:<start_byte>-<end_byte>"`)
/// and compute 1-based line numbers by scanning `source_text`.
/// Non-matching strings fall back to lines 1..1.
fn span_from_op_loc(loc: &str, source_file: &str, source_text: &str) -> SourceSpanJson {
    let (start_byte, end_byte) = match parse_loc_bytes(loc) {
        Some(t) => t,
        None => {
            return SourceSpanJson {
                file: source_file.into(),
                start_line: 1,
                end_line: 1,
            }
        }
    };
    let start_line = line_of_byte(source_text, start_byte);
    let end_line = line_of_byte(source_text, end_byte.saturating_sub(1).max(start_byte));
    SourceSpanJson {
        file: source_file.into(),
        start_line,
        end_line,
    }
}

fn parse_loc_bytes(loc: &str) -> Option<(u32, u32)> {
    let (_file, rest) = loc.split_once(':')?;
    let rest = rest.split_whitespace().next()?;
    let (a, b) = rest.split_once('-')?;
    Some((a.parse().ok()?, b.parse().ok()?))
}

fn line_of_byte(src: &str, byte: u32) -> u32 {
    let mut line: u32 = 1;
    for (i, c) in src.char_indices() {
        if i as u32 >= byte {
            break;
        }
        if c == '\n' {
            line += 1;
        }
    }
    line
}
