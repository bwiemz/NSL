use nsl_cli::monitor::{render_comparison, render_source_view};
use nsl_codegen::profiling::instrument::{EliminatedOp, KernelEntry, Manifest, SourceSpanJson};
use nsl_runtime::profiler::collector::{ActualReport, Aggregate};

fn mk_manifest() -> Manifest {
    Manifest {
        target_gpu: "h100-sxm".into(),
        dtype: "bf16".into(),
        kernels: vec![KernelEntry {
            kernel_id: 0,
            op_name: "fused_attn".into(),
            source_span: SourceSpanJson {
                file: "m.nsl".into(),
                start_line: 3,
                end_line: 5,
            },
            predicted_us: 9.2,
            predicted_flops: 0,
            predicted_hbm_bytes: 0,
        }],
        eliminated_ops: vec![EliminatedOp {
            op_name: "chunk".into(),
            source_span: SourceSpanJson {
                file: "m.nsl".into(),
                start_line: 4,
                end_line: 4,
            },
            reason: "fused into kernel 0".into(),
        }],
    }
}

fn mk_actual(mean_us: f64) -> ActualReport {
    ActualReport {
        aggregates: vec![Aggregate {
            kernel_id: 0,
            count: 100,
            sum_us: mean_us * 100.0,
            min_us: mean_us,
            max_us: mean_us,
            sum_sq_us: mean_us * mean_us * 100.0,
        }],
    }
}

#[test]
fn comparison_flags_large_divergence() {
    let out = render_comparison(&mk_manifest(), &mk_actual(11.5));
    assert!(out.contains("fused_attn"));
    assert!(out.contains("+") && out.contains("%"));
    assert!(out.contains("\u{274C}") || out.contains("\u{26A0}"));
    assert!(out.contains("likely cause"));
}

#[test]
fn comparison_within_5pct_no_flag() {
    let out = render_comparison(&mk_manifest(), &mk_actual(9.4));
    assert!(out.contains("fused_attn"));
    assert!(!out.contains("\u{274C}"));
    assert!(!out.contains("likely cause"));
}

#[test]
fn comparison_missing_actual_shows_na() {
    let out = render_comparison(&mk_manifest(), &ActualReport { aggregates: vec![] });
    assert!(out.contains("n/a"));
}

#[test]
fn source_view_annotates_kernel_and_eliminated_lines() {
    let src = "line1\nline2\nline3 fused_attn start\nline4 chunk\nline5 end\nline6\n";
    let out = render_source_view(&mk_manifest(), &mk_actual(9.2), src, "m.nsl");
    assert!(out.contains("fused_attn"), "{}", out);
    assert!(out.contains("eliminated"), "{}", out);
    assert!(out.contains("   3 |"), "expected line 3 in output: {}", out);
    assert!(out.contains("   4 |"), "expected line 4 in output: {}", out);
}

#[test]
fn source_view_matches_by_basename() {
    let src = "a\nb\nc\nd\ne\nf\n";
    let out = render_source_view(&mk_manifest(), &mk_actual(9.2), src, "/tmp/m.nsl");
    assert!(!out.contains("no annotations"));
}
