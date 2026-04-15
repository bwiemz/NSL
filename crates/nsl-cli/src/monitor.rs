//! `nsl run --monitor` renderer: predicted-vs-actual kernel timing + source
//! view. In Phase 1 the codegen side does not yet emit runtime hooks, so the
//! "actual" JSON will usually be empty and the renderer falls back to a
//! predictions-only view.

use nsl_codegen::profiling::instrument::{KernelEntry, Manifest};
use nsl_runtime::profiler::collector::ActualReport;
use std::path::Path;

pub fn run_monitor(
    source_file: &Path,
    manifest_path: &Path,
    actual_path: &Path,
) -> Result<String, String> {
    let manifest: Manifest = serde_json::from_str(
        &std::fs::read_to_string(manifest_path).map_err(|e| {
            format!(
                "cannot read manifest at {}: {e}",
                manifest_path.display()
            )
        })?,
    )
    .map_err(|e| format!("manifest JSON parse: {e}"))?;
    let actual: ActualReport = match std::fs::read_to_string(actual_path) {
        Ok(s) => serde_json::from_str(&s).map_err(|e| format!("actual JSON parse: {e}"))?,
        Err(_) => {
            // Expected in Phase 1 (no hooks emitted yet).
            ActualReport { aggregates: vec![] }
        }
    };
    let src = std::fs::read_to_string(source_file)
        .map_err(|e| format!("cannot read source: {e}"))?;
    let mut out = String::new();
    if actual.aggregates.is_empty() {
        out.push_str("Note: no actual timings collected (codegen hooks are a Phase 2 feature).\n");
        out.push_str("      Showing predictions only.\n\n");
    }
    out.push_str(&render_comparison(&manifest, &actual));
    out.push_str(&render_source_view(
        &manifest,
        &actual,
        &src,
        source_file.to_str().unwrap_or("<file>"),
    ));
    Ok(out)
}

pub fn render_comparison(manifest: &Manifest, actual: &ActualReport) -> String {
    let mut out = String::from("=== Predicted vs Actual Performance ===\n\n");
    out.push_str(&format!(
        "{:<20} {:>12} {:>12} {:>10}\n",
        "Operation", "Predicted", "Actual", "\u{0394}%"
    ));
    out.push_str(&format!("{}\n", "-".repeat(60)));
    for k in &manifest.kernels {
        let agg = actual
            .aggregates
            .iter()
            .find(|a| a.kernel_id == k.kernel_id);
        let (actual_str, delta_str, flag) = match agg {
            Some(a) if a.count > 0 => {
                let mean = a.sum_us / a.count as f64;
                let d_pct = (mean - k.predicted_us) / k.predicted_us * 100.0;
                let f = if d_pct.abs() > 20.0 {
                    " \u{274C}"
                } else if d_pct.abs() > 5.0 {
                    " \u{26A0}"
                } else {
                    ""
                };
                (
                    format!("{:>10.2}\u{03BC}s", mean),
                    format!("{:>+8.1}%", d_pct),
                    f.to_string(),
                )
            }
            _ => (
                "         n/a".to_string(),
                "    n/a".to_string(),
                String::new(),
            ),
        };
        out.push_str(&format!(
            "{:<20} {:>10.2}\u{03BC}s {} {}{}\n",
            k.op_name, k.predicted_us, actual_str, delta_str, flag
        ));
        if let Some(a) = agg {
            if a.count > 0 {
                let mean = a.sum_us / a.count as f64;
                let d_pct = (mean - k.predicted_us) / k.predicted_us * 100.0;
                if d_pct.abs() > 20.0 {
                    out.push_str(&format!(
                        "   \u{2192} likely cause: {}\n",
                        likely_cause(k)
                    ));
                }
            }
        }
    }
    out
}

fn likely_cause(k: &KernelEntry) -> String {
    match k.op_name.as_str() {
        s if s.contains("matmul") || s.contains("proj") => {
            "tile-size misalignment; check inner dim % 128 == 0".into()
        }
        s if s.contains("attn") => {
            "SMEM pressure / bank conflicts; try smaller CSHA tile".into()
        }
        _ => "cause unknown; rerun with --target-profile=detailed (Phase 2)".into(),
    }
}

pub fn render_source_view(
    m: &Manifest,
    actual: &ActualReport,
    src: &str,
    file: &str,
) -> String {
    let mut out = String::from("\n=== Source-Mapped Kernel View ===\n\n");
    let lines: Vec<&str> = src.lines().collect();

    let mut annots: Vec<(u32, String)> = vec![];
    for k in &m.kernels {
        if !same_file(&k.source_span.file, file) {
            continue;
        }
        let agg = actual
            .aggregates
            .iter()
            .find(|a| a.kernel_id == k.kernel_id);
        let suffix = match agg {
            Some(a) if a.count > 0 => format!(
                "\u{2190} {} ({:.1}\u{03BC}s actual)",
                k.op_name,
                a.sum_us / a.count as f64
            ),
            _ => format!(
                "\u{2190} {} ({:.1}\u{03BC}s pred)",
                k.op_name, k.predicted_us
            ),
        };
        for line in k.source_span.start_line..=k.source_span.end_line {
            annots.push((line, suffix.clone()));
        }
    }
    for elim in &m.eliminated_ops {
        if !same_file(&elim.source_span.file, file) {
            continue;
        }
        for line in elim.source_span.start_line..=elim.source_span.end_line {
            annots.push((line, format!("\u{2190} eliminated ({})", elim.reason)));
        }
    }
    annots.sort_by_key(|(l, _)| *l);
    if annots.is_empty() {
        out.push_str("(no annotations for this file)\n");
        return out;
    }
    let first = annots.first().unwrap().0;
    let last = annots.last().unwrap().0;
    for (i, line) in lines.iter().enumerate() {
        let n = (i + 1) as u32;
        if n < first || n > last {
            continue;
        }
        let ann: String = annots
            .iter()
            .filter(|(ln, _)| *ln == n)
            .map(|(_, a)| a.clone())
            .collect::<Vec<_>>()
            .join("; ");
        out.push_str(&format!("{:>4} | {:<60} {}\n", n, line, ann));
    }
    out
}

fn same_file(a: &str, b: &str) -> bool {
    if a == b {
        return true;
    }
    fn basename(p: &str) -> &str {
        p.rsplit(['/', '\\']).next().unwrap_or(p)
    }
    basename(a) == basename(b)
}

/// One-shot helper used by the `nsl profile` output pipeline: given the
/// ProfileReport already produced during `nsl profile`, build a manifest and
/// write it next to the source file so a later `nsl run --monitor` invocation
/// can consume it.
pub fn write_manifest_beside(
    source_file: &Path,
    report: &nsl_codegen::profiling::types::ProfileReport,
) -> Result<std::path::PathBuf, String> {
    let src = std::fs::read_to_string(source_file).map_err(|e| e.to_string())?;
    let manifest = nsl_codegen::profiling::instrument::build_manifest_from_report(
        report,
        source_file.to_str().unwrap_or("<file>"),
        &src,
    );
    let path = source_file.with_extension("nsl-profile.json");
    nsl_codegen::profiling::instrument::write_manifest(&path, &manifest)
        .map_err(|e| e.to_string())?;
    Ok(path)
}
