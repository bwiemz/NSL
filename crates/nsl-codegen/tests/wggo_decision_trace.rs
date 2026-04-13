use nsl_codegen::wggo_ilp::{DecisionKind, DecisionTrace, LayerDecision, LayerIlpSolution};

#[test]
fn decision_trace_is_constructible_and_serializable() {
    let t = DecisionTrace {
        kind: DecisionKind::CshaLevel,
        chosen: "Level 1".into(),
        runner_up: Some("Level 2 (saves 0.3us vs chosen)".into()),
        binding_constraint: Some("SMEM <= 228 KB".into()),
        metric_summary: "With 6 heads, SMEM for L2 = 84KB. Feasible on H100.".into(),
        cross_decision_note: Some(
            "L2 saves only 1.8us vs 2.3us unpruned. Pruning already cut cost 25%.".into(),
        ),
    };
    let s = serde_json::to_string(&t).unwrap();
    let back: DecisionTrace = serde_json::from_str(&s).unwrap();
    assert_eq!(back.chosen, "Level 1");
    assert!(matches!(back.kind, DecisionKind::CshaLevel));
}

#[test]
fn decision_kind_round_trips_all_variants() {
    use DecisionKind::*;
    for k in [CepHeadPrune, CshaLevel, WrgaAdapter, CpdtPrecision] {
        let s = serde_json::to_string(&k).unwrap();
        let back: DecisionKind = serde_json::from_str(&s).unwrap();
        assert_eq!(format!("{back:?}"), format!("{k:?}"));
    }
}

#[test]
fn layer_ilp_solution_default_decision_trace_is_empty() {
    // LayerIlpSolution doesn't impl Default (LayerDecision is non-trivial);
    // construct manually with empty/zero values to verify the new field.
    let sol = LayerIlpSolution {
        decision: LayerDecision {
            keep_head: vec![],
            ffn_width: 0,
            csha_level: 0,
            adapter_rank: 0,
            optim_m_bits: 8,
            optim_v_bits: 8,
        },
        cost_us: 0.0,
        memory_bytes: 0,
        smem_bytes: 0,
        nodes_explored: 0,
        feasible: false,
        decision_trace: vec![],
    };
    assert!(sol.decision_trace.is_empty());
}
