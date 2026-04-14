use nsl_codegen::profiling::instrument::{
    build_manifest_from_report, write_manifest, ManifestBuilder, SourceSpanJson,
};
use nsl_codegen::profiling::types::{EntryKind, ProfileReport};

fn span(line: u32) -> SourceSpanJson {
    SourceSpanJson {
        file: "m.nsl".into(),
        start_line: line,
        end_line: line,
    }
}

#[test]
fn builder_assigns_dense_ids() {
    let mut b = ManifestBuilder::new("h100-sxm", "bf16");
    let id0 = b.record_kernel("fused_attn", span(42), 9.2, 83_900_000, 12_582_912);
    let id1 = b.record_kernel("gate_proj", span(50), 3.4, 23_100_000, 12_058_624);
    assert_eq!(id0, 0);
    assert_eq!(id1, 1);
    let m = b.finish();
    assert_eq!(m.kernels.len(), 2);
}

#[test]
fn builder_records_eliminated_ops_separately() {
    let mut b = ManifestBuilder::new("h100-sxm", "bf16");
    b.record_eliminated("chunk", span(45), "fused into kernel 0");
    let m = b.finish();
    assert!(m.kernels.is_empty());
    assert_eq!(m.eliminated_ops.len(), 1);
    assert_eq!(m.eliminated_ops[0].op_name, "chunk");
}

#[test]
fn manifest_serializes_to_expected_json_shape() {
    let mut b = ManifestBuilder::new("h100-sxm", "bf16");
    b.record_kernel("x", span(1), 1.0, 10, 20);
    let s = serde_json::to_string(&b.finish()).unwrap();
    assert!(s.contains("\"target_gpu\":\"h100-sxm\""));
    assert!(s.contains("\"kernels\""));
    assert!(s.contains("\"eliminated_ops\""));
}

#[test]
fn build_from_report_splits_ops_and_eliminated() {
    use nsl_codegen::cost_model::{BoundClassification, OpCost};
    let mut ops: Vec<OpCost> = vec![];
    ops.push(OpCost {
        name: "matmul".into(),
        loc: "0:10-30".into(),
        input_shapes: vec![],
        output_shape: "".into(),
        flops: 1000,
        bytes_read: 10,
        bytes_written: 20,
        arithmetic_intensity: 1.0,
        classification: BoundClassification::MemoryBound,
        fused: false,
        estimated_time_us: 1.5,
        origin_node: None,
    });
    ops.push(OpCost {
        name: "chunk".into(),
        loc: "0:40-50".into(),
        input_shapes: vec![],
        output_shape: "".into(),
        flops: 0,
        bytes_read: 0,
        bytes_written: 0,
        arithmetic_intensity: 0.0,
        classification: BoundClassification::Unknown,
        fused: false,
        estimated_time_us: 0.0,
        origin_node: None,
    });
    ops.push(OpCost {
        name: "reshape".into(),
        loc: "0:60-70".into(),
        input_shapes: vec![],
        output_shape: "".into(),
        flops: 500,
        bytes_read: 10,
        bytes_written: 10,
        arithmetic_intensity: 1.0,
        classification: BoundClassification::MemoryBound,
        fused: true,
        estimated_time_us: 0.5,
        origin_node: None,
    });
    let report = ProfileReport {
        target_gpu: "h100-sxm".into(),
        dtype: "bf16".into(),
        entry: EntryKind::Auto,
        ops,
        total_flops: 1500,
        total_hbm_bytes: 50,
        total_estimated_us: 2.0,
        fusion: None,
        memory_timeline: None,
        recommendations: vec![],
        wggo_explain: None,
    };
    let src = "line1\nline2\nline3\n";
    let m = build_manifest_from_report(&report, "m.nsl", src);
    assert_eq!(m.kernels.len(), 1);
    assert_eq!(m.kernels[0].op_name, "matmul");
    assert_eq!(m.eliminated_ops.len(), 2);
}

#[test]
fn write_manifest_round_trips() {
    use nsl_codegen::profiling::instrument::Manifest;
    let tmp = tempfile::NamedTempFile::new().unwrap();
    let m = Manifest {
        target_gpu: "h100-sxm".into(),
        dtype: "bf16".into(),
        kernels: vec![],
        eliminated_ops: vec![],
    };
    write_manifest(tmp.path(), &m).unwrap();
    let s = std::fs::read_to_string(tmp.path()).unwrap();
    let back: Manifest = serde_json::from_str(&s).unwrap();
    assert_eq!(back.target_gpu, "h100-sxm");
}

#[test]
fn reserve_id_returns_dense_monotonic() {
    let mut b = ManifestBuilder::new("h100-sxm", "bf16");
    assert_eq!(b.reserve_id(), 0);
    assert_eq!(b.reserve_id(), 1);
    assert_eq!(b.reserve_id(), 2);
}

#[test]
fn record_kernel_at_uses_supplied_id() {
    let mut b = ManifestBuilder::new("h100-sxm", "bf16");
    let id = b.reserve_id();
    b.record_kernel_at(
        id, "matmul",
        SourceSpanJson { file: "m.nsl".into(), start_line: 10, end_line: 12 },
        1.5, 1000, 50,
    );
    let m = b.finish();
    assert_eq!(m.kernels.len(), 1);
    assert_eq!(m.kernels[0].kernel_id, id);
    assert_eq!(m.kernels[0].op_name, "matmul");
    assert_eq!(m.kernels[0].source_span.start_line, 10);
}

#[test]
fn reserve_id_and_record_kernel_interoperate() {
    let mut b = ManifestBuilder::new("h100-sxm", "bf16");
    let a = b.reserve_id();
    b.record_kernel_at(a, "x", SourceSpanJson { file: "m".into(), start_line: 1, end_line: 1 }, 0.0, 0, 0);
    let bid = b.record_kernel("y", SourceSpanJson { file: "m".into(), start_line: 2, end_line: 2 }, 0.0, 0, 0);
    assert_eq!(bid, 1);
    let mfst = b.finish();
    assert_eq!(mfst.kernels.len(), 2);
    assert_eq!(mfst.kernels[0].kernel_id, 0);
    assert_eq!(mfst.kernels[1].kernel_id, 1);
}

#[test]
fn source_span_json_from_span_resolves_lines() {
    let src = "line1\nline2\nline3\nline4\n";
    let span = nsl_errors::Span {
        file_id: nsl_errors::FileId(0),
        start: nsl_errors::BytePos(6),
        end: nsl_errors::BytePos(11),
    };
    let s = SourceSpanJson::from_span(span, "m.nsl", src);
    assert_eq!(s.start_line, 2);
    assert_eq!(s.end_line, 2);
    assert_eq!(s.file, "m.nsl");
}
