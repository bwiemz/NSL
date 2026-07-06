use nsl_codegen::profiling::types::{EntryKind, ProfileReport, Recommendation};

#[test]
fn empty_report_round_trips_json() {
    let r = ProfileReport {
        target_gpu: "h100-sxm".to_string(),
        dtype: "bf16".to_string(),
        entry: EntryKind::Auto,
        ops: vec![],
        total_flops: 0,
        total_hbm_bytes: 0,
        total_estimated_us: 0.0,
        fusion: None,
        memory_timeline: None,
        memory_timeline_approximate: None,
        recommendations: vec![],
        wggo_explain: None,
    };
    let j = serde_json::to_string(&r).unwrap();
    let back: ProfileReport = serde_json::from_str(&j).unwrap();
    assert_eq!(back.target_gpu, "h100-sxm");
    assert_eq!(back.entry, EntryKind::Auto);
}

#[test]
fn entry_kind_parses_from_flag() {
    assert_eq!(EntryKind::parse_flag("auto"), Some(EntryKind::Auto));
    assert_eq!(EntryKind::parse_flag("train"), Some(EntryKind::Train));
    assert_eq!(
        EntryKind::parse_flag("fn:forward"),
        Some(EntryKind::Function("forward".into()))
    );
    assert!(EntryKind::parse_flag("bogus").is_none());
}

#[test]
fn recommendation_has_code_and_message() {
    let rec = Recommendation::memory_bound_batch_hint("gate_proj");
    assert!(rec.code.starts_with("R0"));
    assert!(rec.message.contains("gate_proj"));
}
