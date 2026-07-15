//! CFIE inference decisions in WGGO Level 2 (audit gap G20) — integration
//! tests.
//!
//! Pins the two frozen honesty constraints:
//!   1. OPT-IN, DEFAULT OFF — with `LayerIlpConstraints::cfie_infer = None`
//!      (the production default) the ILP's decisions, costs, serialized
//!      plans, and rendered reports are byte-identical to a build without
//!      the G20 axes.
//!   2. ADVISORY — when the gate is on, the chosen values surface on the
//!      `WggoPlan::cfie_inference` plan sidecar and the report's
//!      "CFIE inference decisions (advisory)" section ONLY; there is
//!      deliberately NO `DecisionKind::CfieInference` trace variant
//!      (nsl-cli's wggo_explain matches `DecisionKind` exhaustively — the
//!      per-layer model constants live on `CfieLayerInference::note`
//!      instead); nothing else changes.
//!
//! Plus hand-computed cost fixtures for each decision axis (arithmetic in
//! comments — every figure is derivable from the stated constants).

use std::collections::HashMap;

use nsl_codegen::gpu_specs::{find_gpu, GpuSpec};
use nsl_codegen::wengert::{PrimalOp, WengertList, WengertOp};
use nsl_codegen::wggo::{run, WggoInput, WggoMode};
use nsl_codegen::wggo_cfie::{
    surface_from_plan, CfieInferConfig, FusionLevel, KvPrecision, LayoutKind,
};
use nsl_codegen::wggo_cost::{build_lut, LayerCostLut, LayerShape, LutAxes};
use nsl_codegen::wggo_dp::{
    ClusterSpec, CoarseDecision, ImportanceScores, InterLayerPlan, LayerPlan,
};
use nsl_codegen::wggo_ilp::{
    recost_decision, solve_all_greedy_cfie, solve_all_templated_cfie, solve_layer,
    solve_layer_cfie, LayerIlpConstraints,
};
use nsl_codegen::wggo_weight_analysis::AnalysisConfig;

// ---------------------------------------------------------------------------
// Fixture: a single-entry LUT whose arithmetic is checkable by hand.
// ---------------------------------------------------------------------------
//
// Shape: batch=1, seq=64 (bs = 64), d_model=256, head_dim=64, n_kv_heads=4,
// dtype=2 B.  Axes pinned to one point: heads=4, ffn=1024, csha=0, rank=0.
//
//   d_out_attn = 4 heads * 64 = 256
//   param_bytes = (4*d_model*d_out_attn + 3*d_model*ffn) * 2
//               = (4*256*256 + 3*256*1024) * 2
//               = (262,144 + 786,432) * 2 = 2,097,152 B
//   activation_bytes = bs*d_model*2*4 = 64*256*8 = 131,072 B
//
// GPU: H100-SXM, peak_bandwidth_gbs = 3350.0 => bw = 3,350,000 bytes/us.

fn h100() -> &'static GpuSpec {
    find_gpu("H100-SXM").expect("H100-SXM present in the GPU database")
}

fn tiny_shape() -> LayerShape {
    LayerShape {
        batch: 1,
        seq: 64,
        d_model: 256,
        head_dim: 64,
        n_kv_heads: 4,
        dtype_bytes: 2,
    }
}

fn tiny_axes() -> LutAxes {
    LutAxes {
        head_counts: vec![4],
        ffn_widths: vec![1024],
        csha_levels: vec![0],
        adapter_ranks: vec![0],
    }
}

fn tiny_lut() -> LayerCostLut {
    build_lut(&tiny_shape(), h100(), &tiny_axes())
}

/// One head config (num_heads=4, gqa_group=4 => a single all-kept group),
/// one packing mode, FASE off — so the CFIE axes are the only thing that
/// varies across the assertions below besides optimizer precision.
fn tiny_constraints() -> LayerIlpConstraints {
    LayerIlpConstraints {
        num_heads: 4,
        gqa_group: 4,
        packing_modes_mask: 0b0001,
        allow_fase: false,
        // The memory arithmetic below assumes the solver's 8-bit Adam
        // moments ("sensitivity = 0"), i.e. an INFORMED zero-sensitivity
        // layer. Uninformed layers are pinned to 32-bit moments since the
        // optim-bit lowering landed.
        sensitivity_informed: true,
        ..Default::default()
    }
}

/// Gate-ON constraints: KV geometry 2*(K+V)*4 heads*64 dim => fp16 = 1024 B
/// and int8 = 512 B per stored token per layer; max_seq=4096 => the KV read
/// is evaluated at kv_seq = 2048.
fn gate_on(cfg: CfieInferConfig) -> LayerIlpConstraints {
    LayerIlpConstraints {
        cfie_infer: Some(cfg),
        ..tiny_constraints()
    }
}

fn fixture_cfg() -> CfieInferConfig {
    CfieInferConfig {
        n_kv_heads: 4,
        head_dim: 64,
        max_seq: 4096,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Wengert harness for full-driver runs (mirrors wggo.rs's test fixture).
// ---------------------------------------------------------------------------

fn op(id: u32, result: u32, o: PrimalOp, inputs: Vec<u32>) -> WengertOp {
    WengertOp {
        id,
        result,
        op: o,
        inputs,
        saved_for_backward: false,
        checkpointed: false,
    }
}

fn two_block_wengert() -> WengertList {
    let ops = vec![
        op(0, 0, PrimalOp::Input("x".into()), vec![]),
        op(1, 1, PrimalOp::Param("blocks.0.attn.wq".into()), vec![]),
        op(2, 2, PrimalOp::Matmul, vec![1, 0]),
        op(3, 3, PrimalOp::Param("blocks.1.attn.wq".into()), vec![]),
        op(4, 4, PrimalOp::Matmul, vec![3, 2]),
    ];
    WengertList {
        ops,
        output: 4,
        var_names: HashMap::new(),
        var_types: HashMap::new(),
    }
}

fn driver_input<'a>(
    w: &'a WengertList,
    ilp_constraints: Vec<LayerIlpConstraints>,
) -> WggoInput<'a> {
    WggoInput {
        mode: WggoMode::Full,
        target: "H100-SXM",
        wengert: w,
        layer_shape: tiny_shape(),
        cluster: ClusterSpec::default(),
        lut_axes: tiny_axes(),
        importance: ImportanceScores::default(),
        ilp_constraints,
        weights: None,
        analysis_config: AnalysisConfig::default(),
        scorer: None,
        cached_analysis: None,
        packing_supported: true,
        memory_budget_bytes: None,
        packing_stats: None,
    }
}

// ---------------------------------------------------------------------------
// (a) Gate OFF: byte-identity with today's solver / plan / report.
// ---------------------------------------------------------------------------

#[test]
fn gate_off_solution_carries_no_cfie_surface() {
    let lut = tiny_lut();
    let c = tiny_constraints(); // cfie_infer: None — the production default
    let (sol, choice) = solve_layer_cfie(&lut, &c);
    assert!(sol.feasible);
    assert!(choice.is_none(), "gate off must never produce a choice");

    // Serialized solution is byte-free of any CFIE key: LayerIlpSolution's
    // layout is untouched by G20 (the choice travels in a sidecar).
    let json = serde_json::to_string(&sol).unwrap();
    assert!(
        !json.to_lowercase().contains("cfie"),
        "gate-off solution JSON must not mention cfie: {json}"
    );

    // Decision trace stays the original six kinds (the G20 surface is the
    // plan sidecar, not the trace — nsl-cli matches DecisionKind
    // exhaustively, so the enum is deliberately untouched).
    assert_eq!(sol.decision_trace.len(), 6);

    // Objective composition unchanged: the re-cost of the chosen decision
    // (forward+backward + adapter-comm + optimizer, NO decode term) equals
    // the solver's reported cost.
    let recost = recost_decision(&lut, &sol.decision, &c);
    assert!(
        (recost - sol.cost_us).abs() < 1e-9,
        "gate-off objective must not contain a G20 term: recost={recost} cost={sol_cost}",
        sol_cost = sol.cost_us
    );

    // And the legacy entry point returns the identical solution.
    let legacy = solve_layer(&lut, &c);
    assert_eq!(legacy.cost_us.to_bits(), sol.cost_us.to_bits());
    assert_eq!(legacy.memory_bytes, sol.memory_bytes);
    assert_eq!(legacy.decision, sol.decision);
}

#[test]
fn gate_off_report_and_plan_have_no_cfie_section_and_are_deterministic() {
    let w = two_block_wengert();
    let plan1 = run(driver_input(&w, Vec::new())); // default constraints: gate off
    let plan2 = run(driver_input(&w, Vec::new()));

    assert!(plan1.cfie_inference.is_empty());
    let rep1 = plan1.render_report();
    assert!(
        !rep1.contains("CFIE inference decisions"),
        "gate-off report must not contain the advisory section"
    );

    // Gate-off plan JSON is free of the (skip-if-empty) cfie_inference key.
    let json = serde_json::to_string(&plan1).unwrap();
    assert!(
        !json.contains("cfie_inference"),
        "gate-off plan JSON must skip the empty advisory vec"
    );

    // Two identical gate-off runs render bit-identical reports (modulo the
    // wall-clock "Solve time" line) — the same determinism pin the wggo
    // lib tests use.
    let strip = |s: &str| -> String {
        s.lines()
            .filter(|l| !l.contains("Solve time"))
            .collect::<Vec<_>>()
            .join("\n")
    };
    assert_eq!(strip(&rep1), strip(&plan2.render_report()));
}

// ---------------------------------------------------------------------------
// (b) Hand-computed cost fixtures: the KV-precision axis.
// ---------------------------------------------------------------------------

#[test]
fn int8_kv_wins_on_bandwidth_when_dequant_is_cheap() {
    // KV read term at kv_seq = 4096/2 = 2048, bw = 3,350,000 B/us:
    //   fp16: 1024 B/token * 2048 = 2,097,152 B => 0.626016... us
    //   int8:  512 B/token * 2048 = 1,048,576 B => 0.313008... us
    // int8 saves 0.313008 us of bandwidth; its dequant surcharge here is
    // only 0.05 us => int8 total 0.363008 < fp16 0.626016.  Every other
    // term (weight read, launches, layout, spec) is identical across the
    // two precisions, so the ILP must pick Int8.
    let cfg = CfieInferConfig {
        int8_dequant_us: 0.05,
        ..fixture_cfg()
    };
    let (sol, choice) = solve_layer_cfie(&tiny_lut(), &gate_on(cfg));
    assert!(sol.feasible);
    let choice = choice.expect("gate on must produce a choice");
    assert_eq!(choice.kv_precision, KvPrecision::Int8);
    // Unconstrained memory: the never-slower static layout and deepest
    // fusion level are chosen alongside.
    assert_eq!(choice.kv_layout, LayoutKind::Static);
    assert_eq!(choice.fusion_level, FusionLevel::Level3);
}

#[test]
fn fp16_kv_wins_when_int8_dequant_dominates() {
    // Same bandwidth arithmetic as above (int8 saves 0.313008 us), but the
    // dequant surcharge is now 1.0 us => int8 total 0.313008 + 1.0 =
    // 1.313008 us > fp16's 0.626016 us.  The ILP must keep Fp16.
    let cfg = CfieInferConfig {
        int8_dequant_us: 1.0,
        ..fixture_cfg()
    };
    let (sol, choice) = solve_layer_cfie(&tiny_lut(), &gate_on(cfg));
    assert!(sol.feasible);
    assert_eq!(choice.unwrap().kv_precision, KvPrecision::Fp16);
}

// ---------------------------------------------------------------------------
// (c) Speculative decoding wins only when the acceptance rate justifies it.
// ---------------------------------------------------------------------------

#[test]
fn speculative_on_only_when_acceptance_justifies_it() {
    // Factor = (1 + draft_frac*K) / (1 + K*p/(1+K)) with K=4, draft=0.15:
    //   numerator = 1 + 0.6 = 1.6 (fixed)
    //   p=0.90: denominator = 1 + 3.6/5 = 1.72 => factor = 0.93023 < 1
    //           — speculative strictly improves every candidate => ON.
    //   p=0.30: denominator = 1 + 1.2/5 = 1.24 => factor = 1.29032 > 1
    //           — speculative strictly worsens every candidate => OFF.
    let hot = CfieInferConfig {
        spec_acceptance: 0.9,
        ..fixture_cfg()
    };
    let (_, choice_hot) = solve_layer_cfie(&tiny_lut(), &gate_on(hot));
    assert!(
        choice_hot.unwrap().speculative,
        "p=0.9 makes the acceptance-adjusted latency strictly better"
    );

    let cold = CfieInferConfig {
        spec_acceptance: 0.3,
        ..fixture_cfg()
    };
    let (_, choice_cold) = solve_layer_cfie(&tiny_lut(), &gate_on(cold));
    assert!(
        !choice_cold.unwrap().speculative,
        "p=0.3 cannot pay for the 4-token draft overhead"
    );
}

// ---------------------------------------------------------------------------
// KV-layout axis: static unless the memory budget forces paged.
// ---------------------------------------------------------------------------

#[test]
fn static_layout_wins_unless_memory_budget_forces_paged() {
    // Training-resident bytes of the single LUT entry at the cheapest
    // (8-bit) Adam moments the solver will pick (sensitivity = 0):
    //   param 2,097,152 + moments 2*(1,048,576 elems * 1 B) = 2,097,152
    //   + activation 131,072 = 4,325,376 B.
    // KV pools at max_seq=4096 (kv_seq 2048):
    //   static fp16 = 1024*4096 = 4,194,304   paged fp16 = 1024*2048 = 2,097,152
    //   static int8 =  512*4096 = 2,097,152   paged int8 =  512*2048 = 1,048,576
    //
    // Unconstrained budget: static is never slower (paged pays the 5 us
    // block-table/sync overhead), and dequant=1.0 makes fp16 cheaper than
    // int8 => (Static, Fp16).
    let cfg = CfieInferConfig {
        int8_dequant_us: 1.0,
        ..fixture_cfg()
    };
    let (_, unconstrained) = solve_layer_cfie(&tiny_lut(), &gate_on(cfg.clone()));
    let u = unconstrained.unwrap();
    assert_eq!(u.kv_layout, LayoutKind::Static);
    assert_eq!(u.kv_precision, KvPrecision::Fp16);

    // Budget 6,000,000 B: resident 4,325,376 +
    //   static int8  => 6,422,528 > budget  (infeasible)
    //   paged  fp16  => 6,422,528 > budget  (infeasible)
    //   paged  int8  => 5,373,952 <= budget (the ONLY fit)
    // so the ILP must fall to (Paged, Int8) even though the dequant term
    // says fp16 and the latency term says static — memory is binding.
    let mut tight = gate_on(cfg);
    tight.memory_budget = 6_000_000;
    let (sol, forced) = solve_layer_cfie(&tiny_lut(), &tight);
    assert!(sol.feasible, "paged-int8 fits: 5,373,952 <= 6,000,000");
    let f = forced.unwrap();
    assert_eq!(f.kv_layout, LayoutKind::Paged);
    assert_eq!(f.kv_precision, KvPrecision::Int8);
    // The reported memory includes the KV pool.
    assert_eq!(sol.memory_bytes, 4_325_376 + 1_048_576);
}

// ---------------------------------------------------------------------------
// Fusion-level axis: cost-driven, capped by the serve planner's verdict.
// ---------------------------------------------------------------------------

#[test]
fn fusion_level_minimizes_launch_cost_under_the_cap() {
    // Launch cost per token per layer: none = 12*5 = 60 us, level2 = 2*5 =
    // 10 us, level3 = 1*5 = 5 us.  Uncapped => Level3.
    let (_, deep) = solve_layer_cfie(&tiny_lut(), &gate_on(fixture_cfg()));
    assert_eq!(deep.unwrap().fusion_level, FusionLevel::Level3);

    // Cap at Level2 (e.g. cfie_persistent::choose_fusion said Level3's
    // SMEM does not fit): domain shrinks to {none, level2}; level2's 10 us
    // beats none's 60 us.
    let capped_cfg = CfieInferConfig {
        max_fusion_level: FusionLevel::Level2,
        ..fixture_cfg()
    };
    let (_, capped) = solve_layer_cfie(&tiny_lut(), &gate_on(capped_cfg));
    assert_eq!(capped.unwrap().fusion_level, FusionLevel::Level2);
}

// ---------------------------------------------------------------------------
// (d) Gate ON surface: plan sidecar entries + report section (the decision
//     trace deliberately stays the original six kinds; see the module doc).
// ---------------------------------------------------------------------------

#[test]
fn gate_on_populates_plan_sidecar_and_report() {
    let w = two_block_wengert();
    let graph = nsl_codegen::wggo_graph::build(&w);
    let n = graph.layers.len();
    let constraints = vec![gate_on(fixture_cfg()); n];
    let plan = run(driver_input(&w, constraints));

    // Plan surface (G20 deliverable): one advisory entry per non-pruned
    // gate-on layer, each carrying the cost model + constants it was
    // priced with (the CfieCostEstimate.assumptions precedent).
    assert!(
        !plan.cfie_inference.is_empty(),
        "gate on must surface advisory decisions on the plan"
    );
    for d in &plan.cfie_inference {
        assert_eq!(d.choice.fusion_level, FusionLevel::Level3);
        assert_eq!(d.choice.kv_layout, LayoutKind::Static);
        assert!(!d.layer_name.is_empty());
        assert!(d.note.contains("decode_us/token/layer"));
        assert!(d.note.contains("K=4"), "constants stated: {}", d.note);
    }

    // Report section present, advisory-worded, with the per-layer choices
    // and the stated cost model.
    let rep = plan.render_report();
    assert!(rep.contains("CFIE inference decisions (advisory):"));
    assert!(rep.contains("Report-only in this cycle"));
    assert!(rep.contains("fusion=level3"));
    assert!(rep.contains("kv_layout=static"));
    assert!(rep.contains("Leviathan 2023"));

    // Serialized plan carries the advisory vec when (and only when) on.
    let json = serde_json::to_string(&plan).unwrap();
    assert!(json.contains("cfie_inference"));
}

#[test]
fn pruned_layers_are_excluded_from_the_advisory_surface() {
    // surface_from_plan mirrors apply()'s pruned-layer forcing: a layer the
    // inter-layer DP pruned must not advertise inference decisions.
    let mk_layer = |i: u32, decision: CoarseDecision| LayerPlan {
        layer_index: i,
        name: format!("blocks.{i}"),
        decision,
        pipeline_stage: 0,
        shard_params: 1,
        shard_grads: 1,
        shard_optim: 1,
        estimated_us: 1.0,
        estimated_bytes: 1,
        param_bytes: 1,
        activation_bytes: 1,
    };
    let inter = InterLayerPlan {
        layers: vec![
            mk_layer(0, CoarseDecision::KeepFull),
            mk_layer(1, CoarseDecision::Prune),
        ],
        total_us: 2.0,
        peak_memory_bytes: 2,
        pipeline_stages: 1,
    };
    let (_, choice) = solve_layer_cfie(&tiny_lut(), &gate_on(fixture_cfg()));
    let choices = vec![choice, choice]; // both layers "chose" something
    let constraints = vec![gate_on(fixture_cfg()), gate_on(fixture_cfg())];
    let surfaced = surface_from_plan(&inter, &choices, &constraints);
    assert_eq!(surfaced.len(), 1);
    assert_eq!(surfaced[0].layer_index, 0);
    assert_eq!(surfaced[0].layer_name, "blocks.0");
    assert!(surfaced[0].note.contains("decode_us/token/layer"));
}

// ---------------------------------------------------------------------------
// Solver-infrastructure pins: template key, greedy behavior.
// ---------------------------------------------------------------------------

#[test]
fn template_cache_never_shares_across_gate_states() {
    // Two layers with identical LUTs whose constraints differ ONLY in the
    // G20 gate must be solved as two distinct templates.
    let luts = vec![tiny_lut(), tiny_lut()];
    let constraints = vec![tiny_constraints(), gate_on(fixture_cfg())];
    let (sols, stats, choices) = solve_all_templated_cfie(&luts, &constraints);
    assert_eq!(stats.templates_solved, 2);
    assert_eq!(stats.template_hits, 0);
    assert!(choices[0].is_none());
    assert!(choices[1].is_some());
    assert!(sols.iter().all(|s| s.feasible));

    // And identical gate-on layers DO share one template, replicating the
    // cached choice.
    let luts2 = vec![tiny_lut(), tiny_lut()];
    let constraints2 = vec![gate_on(fixture_cfg()), gate_on(fixture_cfg())];
    let (_, stats2, choices2) = solve_all_templated_cfie(&luts2, &constraints2);
    assert_eq!(stats2.templates_solved, 1);
    assert_eq!(stats2.template_hits, 1);
    assert_eq!(choices2[0], choices2[1]);
}

#[test]
fn greedy_advises_when_on_refuses_when_pool_cannot_fit() {
    let luts = vec![tiny_lut()];

    // Gate off: no advice.
    let (_, off) = solve_all_greedy_cfie(&luts, &[tiny_constraints()]);
    assert!(off[0].is_none());

    // Gate on, unconstrained: advice present.
    let (sols_on, on) = solve_all_greedy_cfie(&luts, &[gate_on(fixture_cfg())]);
    assert!(sols_on[0].feasible);
    assert!(on[0].is_some());

    // Gate on, budget 5,000,000: training-resident bytes are 4,325,376
    // (see the layout test) leaving 674,624 B of headroom — below even the
    // smallest pool (paged int8 = 1,048,576).  Greedy must keep the layer
    // feasible but REFUSE to advise rather than surface an over-budget
    // pool.
    let mut squeezed = gate_on(fixture_cfg());
    squeezed.memory_budget = 5_000_000;
    let (sols_sq, sq) = solve_all_greedy_cfie(&luts, &[squeezed]);
    assert!(sols_sq[0].feasible, "the layer itself fits without a pool");
    assert!(
        sq[0].is_none(),
        "no pool fits in 674,624 B of headroom — greedy must refuse to advise"
    );
}
