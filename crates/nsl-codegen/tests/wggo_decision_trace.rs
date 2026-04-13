use nsl_codegen::gpu_specs::{find_gpu, GPU_DATABASE};
use nsl_codegen::wggo_cost::{build_lut, LayerShape, LutAxes};
use nsl_codegen::wggo_ilp::{
    solve_layer, DecisionKind, DecisionTrace, LayerDecision, LayerIlpConstraints, LayerIlpSolution,
};

fn h100() -> &'static nsl_codegen::gpu_specs::GpuSpec {
    find_gpu("H100")
        .or_else(|| find_gpu("h100"))
        .unwrap_or(&GPU_DATABASE[0])
}

fn layer_shape() -> LayerShape {
    LayerShape {
        batch: 1,
        seq: 1024,
        d_model: 512,
        head_dim: 64,
        n_kv_heads: 4,
        dtype_bytes: 2,
    }
}

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
            fase_fused: false,
            packing_mode: 0,
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

// --------------------------- Task 2 --------------------------------

#[test]
fn solve_layer_populates_four_decision_traces_per_layer() {
    let lut = build_lut(&layer_shape(), h100(), &LutAxes::default());
    let sol = solve_layer(&lut, &LayerIlpConstraints::default());
    assert!(sol.feasible, "expected feasible solution");

    let trace = &sol.decision_trace;
    assert_eq!(
        trace.len(),
        4,
        "expected 4 decision traces, got {}: {:?}",
        trace.len(),
        trace
            .iter()
            .map(|t| format!("{:?}", t.kind))
            .collect::<Vec<_>>()
    );

    let kinds: Vec<_> = trace.iter().map(|t| format!("{:?}", t.kind)).collect();
    for expected in &["CepHeadPrune", "CshaLevel", "WrgaAdapter", "CpdtPrecision"] {
        assert!(
            kinds.iter().any(|k| k == expected),
            "missing trace for kind {}, got {:?}",
            expected,
            kinds
        );
    }
    for t in trace {
        assert!(!t.chosen.is_empty(), "{:?} missing chosen", t.kind);
        assert!(
            !t.metric_summary.is_empty(),
            "{:?} missing metric_summary",
            t.kind
        );
    }
}

#[test]
fn infeasible_layer_has_empty_decision_trace() {
    let lut = build_lut(&layer_shape(), h100(), &LutAxes::default());
    let mut constraints = LayerIlpConstraints::default();
    constraints.memory_budget = 1_000; // nothing fits
    let sol = solve_layer(&lut, &constraints);
    assert!(!sol.feasible);
    assert!(
        sol.decision_trace.is_empty(),
        "infeasible layers should not emit traces"
    );
}

#[test]
fn decision_trace_reflects_high_sensitivity_tier() {
    let lut = build_lut(&layer_shape(), h100(), &LutAxes::default());
    let mut constraints = LayerIlpConstraints::default();
    constraints.sensitivity = 0.95; // critical tier
    let sol = solve_layer(&lut, &constraints);
    assert!(sol.feasible);
    let cpdt = sol
        .decision_trace
        .iter()
        .find(|t| matches!(t.kind, DecisionKind::CpdtPrecision))
        .expect("cpdt trace present");
    assert!(
        cpdt.chosen.contains("32"),
        "expected 32-bit choice, got {}",
        cpdt.chosen
    );
    assert!(
        cpdt.binding_constraint
            .as_deref()
            .unwrap_or("")
            .contains("critical"),
        "binding constraint should mention critical tier"
    );
}
