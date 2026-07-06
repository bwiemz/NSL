//! Tests for the `wggo_explain` renderer (Phase 3 Task 3).

use nsl_cli::wggo_explain::render_explain;
use nsl_codegen::wggo::{WggoMode, WggoPlan};
use nsl_codegen::wggo_apply::AppliedPlan;
use nsl_codegen::wggo_conflicts::Resolution;
use nsl_codegen::wggo_dp::InterLayerPlan;
use nsl_codegen::wggo_graph::OptGraph;
use nsl_codegen::wggo_ilp::{
    DecisionKind, DecisionTrace, LayerDecision, LayerIlpSolution,
};

fn mk_trace(kind: DecisionKind, chosen: &str, reason: &str) -> DecisionTrace {
    DecisionTrace {
        kind,
        chosen: chosen.into(),
        runner_up: None,
        binding_constraint: None,
        metric_summary: reason.into(),
        cross_decision_note: None,
    }
}

fn empty_layer_solution() -> LayerIlpSolution {
    LayerIlpSolution {
        decision: LayerDecision {
            keep_head: Vec::new(),
            ffn_width: 0,
            csha_level: 0,
            adapter_rank: 0,
            optim_m_bits: 16,
            optim_v_bits: 16,
            fase_fused: false,
            packing_mode: 0,
            adapter_placement: nsl_codegen::wggo_ilp::AdapterPlacement::None,
        },
        cost_us: 0.0,
        memory_bytes: 0,
        smem_bytes: 0,
        nodes_explored: 0,
        feasible: true,
        decision_trace: Vec::new(),
    }
}

fn empty_plan() -> WggoPlan {
    WggoPlan {
        mode: WggoMode::Full,
        target_gpu: "H100".into(),
        graph: OptGraph::default(),
        inter_layer: InterLayerPlan {
            layers: Vec::new(),
            total_us: 0.0,
            peak_memory_bytes: 0,
            pipeline_stages: 1,
        },
        per_layer: Vec::new(),
        resolutions: Vec::new(),
        applied: AppliedPlan::default(),
        schedule: Default::default(),
        template_stats: Default::default(),
        weight_analysis: Default::default(),
        estimated_solve_us: 0,
        warnings: Vec::new(),
        // G20: CFIE inference decisions (advisory sidecar; empty = gate off).
        cfie_inference: Vec::new(),
    }
}

fn mk_plan_with_one_layer() -> WggoPlan {
    let mut layer = empty_layer_solution();
    layer.decision_trace.push(mk_trace(
        DecisionKind::CepHeadPrune,
        "Pruned 2/8 heads (h3, h6)",
        "importance(h3)=0.12, importance(h6)=0.15 — lowest in layer",
    ));
    layer.decision_trace.push(mk_trace(
        DecisionKind::CshaLevel,
        "Level 1",
        "With 6 heads, SMEM for L2 = 84KB. Feasible on H100 (228KB limit).",
    ));
    layer.decision_trace.push(mk_trace(
        DecisionKind::WrgaAdapter,
        "LoRA r=8 on Wq, Wk",
        "Layer 0 has high roofline slack (memory-bound, 32% utilization).",
    ));
    layer.decision_trace.push(mk_trace(
        DecisionKind::CpdtPrecision,
        "INT8 m, FP16 v",
        "sensitivity(0) = 0.31 (low)",
    ));

    let mut plan = empty_plan();
    plan.per_layer.push(layer);
    plan
}

#[test]
fn renders_layer_header_and_all_four_kinds() {
    let plan = mk_plan_with_one_layer();
    let out = render_explain(&plan);
    assert!(out.contains("=== WGGO Decision Explanation ==="));
    assert!(out.contains("Layer 0 decisions:"));
    assert!(out.contains("CEP: Pruned 2/8 heads (h3, h6)"));
    assert!(out.contains("CSHA: Level 1"));
    assert!(out.contains("WRGA: LoRA r=8 on Wq, Wk"));
    assert!(out.contains("CPDT: INT8 m, FP16 v"));
    assert!(out.contains("Reason: importance(h3)=0.12"));
}

#[test]
fn renders_skip_message_for_empty_layer_trace() {
    let mut plan = mk_plan_with_one_layer();
    plan.per_layer[0].decision_trace.clear();
    let out = render_explain(&plan);
    assert!(out.contains("Layer 0 decisions:"));
    assert!(out.contains("(no decisions traced — wggo mode not Full)"));
}

#[test]
fn renders_csha_downgrade_resolution() {
    let mut plan = mk_plan_with_one_layer();
    plan.resolutions
        .push(Resolution::DowngradeCsha { layer: 0, to_level: 1 });
    let out = render_explain(&plan);
    assert!(out.contains("Conflict resolved: CSHA downgrade → Level 1"));
}

#[test]
fn renders_wrga_removed_resolution() {
    let mut plan = mk_plan_with_one_layer();
    plan.resolutions
        .push(Resolution::RemoveWrgaAdapter { layer: 0 });
    let out = render_explain(&plan);
    assert!(out.contains("Conflict resolved: WRGA removed"));
}

#[test]
fn empty_plan_renders_helpful_note() {
    let plan = empty_plan();
    let out = render_explain(&plan);
    assert!(out.contains("(no layers analyzed — was --wggo full active?)"));
}
