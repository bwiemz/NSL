//! WGGO — Wengert Graph Global Optimization: driver.
//!
//! Orchestrates the eight stages described in §5 of the research paper:
//!
//!   1. Wengert graph extraction          — `wengert.rs`
//!   2. Cost-model annotation             — `wggo_cost::build_lut`
//!   3. Weight-analysis (optional)        — `run_weight_analysis` +
//!                                          `wggo_weight_analysis::apply_to`
//!                                          (feeds head-importance /
//!                                          min-retained into the ILP)
//!   4. Level 1: inter-layer DP           — `wggo_dp::solve`
//!   5. Level 2: per-layer ILP            — `wggo_ilp::solve_all`
//!   6. Level 3: kernel generation        — delegated to backend
//!   7. Memory planning                   — delegated to M36
//!   8. Communication schedule            — `wggo_schedule::build_schedule`
//!
//! The driver is pure data-in / data-out and has no backend side
//! effects.  It produces a [`WggoPlan`] downstream passes consume.

use serde::Serialize;

use crate::gpu_specs::{default_gpu, find_gpu, GpuSpec};
use crate::wengert::WengertList;
use crate::wggo_apply::{apply, AppliedPlan};
use crate::wggo_conflicts::{greedy_resolve, LayerDecisions, Resolution};
use crate::wggo_cost::{build_lut, LayerCostLut, LayerShape, LutAxes};
use crate::wggo_dp::{
    passthrough_plan, solve as dp_solve, ClusterSpec, DpConfig, ImportanceScores, InterLayerPlan,
};
use crate::wggo_graph::{build as build_graph, LayerRole, OptGraph};
use crate::wggo_shape::LayerShapeInfo;
use crate::wggo_cfie::{surface_from_plan, CfieInferenceChoice, CfieLayerInference};
use crate::wggo_cpkd::{
    surface_from_plan as cpkd_surface_from_plan, CpkdChoice, CpkdLayerDistill,
};
use crate::wggo_ilp::{
    recost_decision_cpkd, solve_all_greedy_cpkd as ilp_solve_all_greedy_cpkd,
    solve_all_templated_cpkd as ilp_solve_all_templated_cpkd,
    solve_layer_cpkd as ilp_solve_layer_cpkd, LayerIlpConstraints, LayerIlpSolution,
    TemplateStats,
};
use crate::wggo_schedule::{build_schedule, CommSchedule};
use crate::wggo_gradient_scorer::GradientScorer;
use crate::wggo_weight_analysis::{
    analyze as analyze_weights, score_layer_magnitude, AnalysisConfig, NullWeightProvider,
    WeightAnalysisReport, WeightProvider,
};

/// User-visible optimisation mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum WggoMode {
    /// Full hierarchical DP + ILP + conflict resolution (paper §5).
    Full,
    /// Priority-ordered greedy resolution (paper §5.3).  ~500 ms, within
    /// 5 % of optimal on typical models.
    Greedy,
    /// Bypass WGGO entirely; downstream passes run independently.
    Off,
}

impl WggoMode {
    pub fn as_str(self) -> &'static str {
        match self {
            WggoMode::Full => "full",
            WggoMode::Greedy => "greedy",
            WggoMode::Off => "off",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "full" | "auto" => Some(WggoMode::Full),
            "greedy" => Some(WggoMode::Greedy),
            "off" | "disable" | "disabled" => Some(WggoMode::Off),
            _ => None,
        }
    }
}

/// Inputs to the WGGO driver.
///
/// Not `Clone` because `scorer` may hold a `Box<dyn GradientScorer>` which
/// is not cloneable.  Callers that previously relied on `WggoInput::clone()`
/// must construct a fresh `WggoInput` instead.
#[derive(Debug)]
pub struct WggoInput<'a> {
    pub mode: WggoMode,
    pub target: &'a str,
    pub wengert: &'a WengertList,
    pub layer_shape: LayerShape,
    pub cluster: ClusterSpec,
    pub lut_axes: LutAxes,
    pub importance: ImportanceScores,
    /// Per-layer ILP constraint overrides.  If empty, defaults are used.
    pub ilp_constraints: Vec<LayerIlpConstraints>,
    /// Optional weight source for Stage 3 importance analysis.  When
    /// `None`, `head_importance` falls back to uniform scores
    /// (today's behaviour).
    pub weights: Option<&'a dyn WeightProvider>,
    /// Stage-3 tunables (prune fraction, etc.).  Ignored when
    /// `weights` is `None`.
    pub analysis_config: AnalysisConfig,
    /// Optional gradient-informed scorer for head importance.  When
    /// `Some`, its `score_layer` result overrides the magnitude-based
    /// importance for each layer where it returns `Some(HeadImportance)`.
    /// When `None` (the default), the Phase 1 magnitude path is used
    /// exclusively, preserving backward compatibility.
    pub scorer: Option<Box<dyn GradientScorer>>,
    /// Cached Stage-3 report (sidecar hit). When `Some`, the analyzer is
    /// skipped and THIS report drives `apply_to` BEFORE the ILP solves —
    /// splicing it in after the solve (the old behavior) meant a cache hit
    /// solved with NullWeightProvider constraints: identical consecutive
    /// compiles produced different optimizer-state allocations (informed
    /// on miss, uninformed on hit).
    pub cached_analysis: Option<WeightAnalysisReport>,
}

/// Aggregate plan emitted by the driver.
#[derive(Debug, Clone, Serialize)]
pub struct WggoPlan {
    pub mode: WggoMode,
    pub target_gpu: String,
    pub graph: OptGraph,
    pub inter_layer: InterLayerPlan,
    pub per_layer: Vec<LayerIlpSolution>,
    pub resolutions: Vec<Resolution>,
    pub applied: AppliedPlan,
    /// Cross-device collective schedule (Stage 8).  Produced from the
    /// post-resolution per-layer plan; consumed by CPDT lowering.
    pub schedule: CommSchedule,
    /// Counters for template-layer reuse during the per-layer ILP.
    pub template_stats: TemplateStats,
    /// Stage-3 importance-analysis output (one entry per layer).
    pub weight_analysis: WeightAnalysisReport,
    /// Total solver wall-clock time (μs, self-reported — not measured).
    pub estimated_solve_us: u64,
    /// Graceful-degradation / limitation warnings accumulated during the run
    /// (e.g. "full mode infeasible, degraded to off").  Empty on a clean run.
    pub warnings: Vec<String>,
    /// CFIE inference decisions (audit gap G20) — ADVISORY, report-only.
    /// Populated only when the opt-in `LayerIlpConstraints::cfie_infer`
    /// gate is on for at least one non-pruned layer; empty (and skipped in
    /// serialization) otherwise, keeping gate-off plans byte-identical.
    /// Not yet consumed by the CFIE serve planner — surfacing here + in
    /// the report IS the deliverable this cycle.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cfie_inference: Vec<CfieLayerInference>,
    /// CPKD distillation decisions (v1) — ADVISORY, report-only.
    /// Populated only when the opt-in `LayerIlpConstraints::cpkd` gate is
    /// on for at least one non-pruned layer; empty (and skipped in
    /// serialization) otherwise, keeping gate-off plans byte-identical.
    /// Not yet consumed by the distill lowering — surfacing here + in the
    /// report's "[cpkd]" section IS the deliverable this cycle.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub cpkd_distill: Vec<CpkdLayerDistill>,
}

impl WggoPlan {
    pub fn render_report(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        writeln!(s, "=== WGGO Global Optimization Report ===").unwrap();
        writeln!(s, "Mode: {}", self.mode.as_str()).unwrap();
        writeln!(s, "Target GPU: {}", self.target_gpu).unwrap();
        writeln!(
            s,
            "Layers: {} ({} pruned, {} kept)",
            self.inter_layer.layers.len(),
            self.inter_layer.pruned_layers(),
            self.inter_layer.kept_layers()
        )
        .unwrap();
        writeln!(s).unwrap();
        writeln!(s, "Level 1 (inter-layer DP):").unwrap();
        writeln!(
            s,
            "  Pipeline stages: {}, ZeRO shard: params={}/grads={}/optim={}",
            self.inter_layer.pipeline_stages,
            self.inter_layer.layers.first().map(|l| l.shard_params).unwrap_or(1),
            self.inter_layer.layers.first().map(|l| l.shard_grads).unwrap_or(1),
            self.inter_layer.layers.first().map(|l| l.shard_optim).unwrap_or(1),
        )
        .unwrap();
        writeln!(s, "Level 2 (per-layer ILP):").unwrap();
        for layer in &self.applied.layers {
            writeln!(
                s,
                "  {}: {}/{} heads, FFN={}, CSHA-L{}, LoRA r={} [{}], m={}b v={}b, FASE={}, PCA={}",
                layer.layer_name,
                layer.active_heads,
                layer.active_heads, // number of actually-kept heads (no "of total" info here)
                layer.ffn_width,
                layer.csha_level,
                layer.adapter_rank,
                layer.adapter_placement.as_str(),
                layer.optim_m_bits,
                layer.optim_v_bits,
                if layer.fase_fused { "fused" } else { "deferred" },
                // Shared with the [pca] consumption diagnostics — one naming
                // authority for packing modes (errata E2 wiring).
                crate::wggo_overrides::packing_mode_name(layer.packing_mode)
            )
            .unwrap();
        }
        // CFIE inference decisions (G20) — printed only when the opt-in
        // gate produced choices, so gate-off reports stay byte-identical.
        if !self.cfie_inference.is_empty() {
            writeln!(s).unwrap();
            writeln!(s, "CFIE inference decisions (advisory):").unwrap();
            writeln!(
                s,
                "  Report-only in this cycle: not yet consumed by the CFIE serve planner; \
                 no generated code changes."
            )
            .unwrap();
            for d in &self.cfie_inference {
                writeln!(
                    s,
                    "  {}: fusion={}, kv_layout={}, kv_precision={}, speculative={}",
                    d.layer_name,
                    d.choice.fusion_level.as_str(),
                    d.choice.kv_layout.as_str(),
                    d.choice.kv_precision.as_str(),
                    if d.choice.speculative { "on" } else { "off" }
                )
                .unwrap();
            }
            // Each record carries the model + constants it was priced with
            // (wggo_cfie::model_note); configs are per-layer, so print the
            // first and flag any layer whose note differs.
            if let Some(first) = self.cfie_inference.first() {
                writeln!(s, "  Cost model: {}", first.note).unwrap();
                for d in self.cfie_inference.iter().skip(1) {
                    if d.note != first.note {
                        writeln!(s, "  Cost model ({}): {}", d.layer_name, d.note).unwrap();
                    }
                }
            }
        }
        // CPKD distillation decisions (v1) — printed only when the opt-in
        // gate produced choices, so gate-off reports stay byte-identical.
        if !self.cpkd_distill.is_empty() {
            writeln!(s).unwrap();
            writeln!(s, "[cpkd] Distillation decisions (advisory):").unwrap();
            writeln!(
                s,
                "[cpkd]   Report-only in v1: not consumed by the distill lowering; \
                 attn transfer + teacher-stream overlap are deferred features."
            )
            .unwrap();
            for d in &self.cpkd_distill {
                writeln!(
                    s,
                    "[cpkd]   {}: feature_match={}, attn_transfer={}, teacher_stream={}",
                    d.layer_name,
                    if d.choice.feature_match { "on" } else { "off" },
                    if d.choice.attn_transfer { "on" } else { "off" },
                    if d.choice.teacher_stream { "on" } else { "off" },
                )
                .unwrap();
            }
            // Each record carries the model + constants it was priced with
            // (wggo_cpkd::model_note); configs are per-layer, so print the
            // first and flag any layer whose note differs.
            if let Some(first) = self.cpkd_distill.first() {
                writeln!(s, "[cpkd]   Cost model: {}", first.note).unwrap();
                for d in self.cpkd_distill.iter().skip(1) {
                    if d.note != first.note {
                        writeln!(s, "[cpkd]   Cost model ({}): {}", d.layer_name, d.note)
                            .unwrap();
                    }
                }
            }
        }
        writeln!(s).unwrap();
        writeln!(s, "Conflicts resolved: {}", self.resolutions.len()).unwrap();
        for (i, r) in self.resolutions.iter().enumerate() {
            writeln!(s, "  [{}] {:?}", i + 1, r).unwrap();
        }
        writeln!(s).unwrap();
        writeln!(s, "Global metrics:").unwrap();
        writeln!(
            s,
            "  Step time: {:.2} μs  |  Peak memory: {:.2} MB",
            self.applied.total_us,
            self.applied.peak_memory_bytes as f64 / 1e6
        )
        .unwrap();
        writeln!(s, "  Solve time: {:.2} ms", self.estimated_solve_us as f64 / 1000.0).unwrap();
        writeln!(
            s,
            "  Templates: {} solved, {} layers reused",
            self.template_stats.templates_solved, self.template_stats.template_hits
        )
        .unwrap();
        writeln!(
            s,
            "  Weight analysis: {} layers analysed, {} without weights",
            self.weight_analysis.per_layer.len(),
            self.weight_analysis.layers_without_weights
        )
        .unwrap();
        writeln!(s).unwrap();
        writeln!(s, "Warnings / limitations: {}", self.warnings.len()).unwrap();
        for w in &self.warnings {
            writeln!(s, "  WARNING: {w}").unwrap();
        }
        writeln!(s).unwrap();
        write!(s, "{}", self.schedule.render()).unwrap();
        s
    }

    /// Summary single-line string suitable for debug logs.
    pub fn summary(&self) -> String {
        format!(
            "wggo[{mode}]: {layers} layers, {kept} kept, step={step_us:.1}μs, mem={mem_mb:.1}MB, conflicts={conf}",
            mode = self.mode.as_str(),
            layers = self.inter_layer.layers.len(),
            kept = self.inter_layer.kept_layers(),
            step_us = self.applied.total_us,
            mem_mb = self.applied.peak_memory_bytes as f64 / 1e6,
            conf = self.resolutions.len()
        )
    }
}

/// Heuristic per-layer ceiling on adapter-allreduce "cost units"
/// (`rank · placement_sites · shard_factor`) under ZeRO sharding.  An adapter
/// whose cost exceeds this on a sharded layer trips the `WrgaVsCpdt` conflict,
/// and the resolver drops the adapter (CPDT > WRGA).  Deterministic fallback
/// heuristic — the repo has no measured collective-latency oracle.
const ADAPTER_COMM_SHARD_BUDGET: f64 = 64.0;

/// Greedy mode escalates conflicting layers to the full ILP when conflict
/// resolution worsens the estimated step time by more than this fraction (G3).
const GREEDY_RECOST_THRESHOLD: f64 = 0.05;

/// Build the conflict-detector input table from the inter-layer plan and the
/// per-layer ILP solutions.  Factored out so it can be rebuilt after greedy
/// escalation re-solves some layers.
fn build_layer_decisions(
    inter: &InterLayerPlan,
    per_layer: &[LayerIlpSolution],
) -> Vec<LayerDecisions> {
    inter
        .layers
        .iter()
        .zip(per_layer.iter())
        .map(|(inter_layer, sol)| LayerDecisions {
            layer: inter_layer.layer_index,
            csha_level: sol.decision.csha_level,
            head_count: sol.decision.active_heads() as u32,
            pruned_heads: (sol.decision.keep_head.len() as u32)
                .saturating_sub(sol.decision.active_heads() as u32),
            adapter_rank: sol.decision.adapter_rank,
            shard_factor: inter_layer.shard_params,
            fase_fused: sol.decision.fase_fused,
            // Real (non-neutralized) adapter-allreduce cost under sharding so
            // the WrgaVsCpdt detector is live in production (G8): cost grows
            // with adapter rank, placement sites, and shard degree.
            adapter_comm_cost: (sol.decision.adapter_rank
                * sol.decision.adapter_placement.proj_sites() as u64
                * inter_layer.shard_params as u64) as f64,
            adapter_comm_budget: ADAPTER_COMM_SHARD_BUDGET,
        })
        .collect()
}

/// Feed the resolver's verdicts back into the per-layer ILP solutions so
/// `apply` sees post-resolution decisions (downgraded CSHA, removed adapter,
/// deferred FASE step).
fn splice_resolved(per_layer: &mut [LayerIlpSolution], resolved: &[LayerDecisions]) {
    for (sol, r) in per_layer.iter_mut().zip(resolved.iter()) {
        sol.decision.csha_level = r.csha_level;
        sol.decision.adapter_rank = r.adapter_rank;
        sol.decision.fase_fused = r.fase_fused;
    }
}

/// Total re-costed step time across all layers (G3 cost re-evaluation).
/// `cfie_choices` / `cpkd_choices` keep a gate-on re-cost comparable with
/// the solver's objective (which includes the G20 decode term and the CPKD
/// distill term); with the gates off every entry is `None` and the sum is
/// bit-identical to the pre-G20 re-cost.
fn recost_total(
    luts: &[LayerCostLut],
    per_layer: &[LayerIlpSolution],
    constraints: &[LayerIlpConstraints],
    cfie_choices: &[Option<CfieInferenceChoice>],
    cpkd_choices: &[Option<CpkdChoice>],
) -> f64 {
    per_layer
        .iter()
        .enumerate()
        .map(|(i, sol)| {
            let lut = luts.get(i).or_else(|| luts.first());
            let cons = constraints.get(i).or_else(|| constraints.first());
            match (lut, cons) {
                (Some(lut), Some(cons)) => recost_decision_cpkd(
                    lut,
                    &sol.decision,
                    cons,
                    cfie_choices.get(i).copied().flatten(),
                    cpkd_choices.get(i).copied().flatten(),
                ),
                _ => 0.0,
            }
        })
        .sum()
}

/// Run the WGGO driver.
pub fn run(input: WggoInput) -> WggoPlan {
    let t0 = std::time::Instant::now();
    let gpu = find_gpu(input.target).unwrap_or_else(default_gpu);

    // 1. Build the layer graph.
    let graph = build_graph(input.wengert);

    // §2.4 inter-layer shape-compatibility precondition (a *hard* constraint).
    // Every kept layer's output activation width must match what its successors
    // consume (or a reshape must bridge them).  On a *provable* incompatibility
    // WGGO must emit NO structural transforms — it returns a transform-free plan
    // (empty `AppliedPlan` ⇒ no per-layer overrides ⇒ the type-checked model
    // compiles unchanged), honoring the project invariant that an unmet
    // transformation precondition must refuse, never silently weaken.  This is
    // deliberately stronger than the inter-layer DP's budget degradation: an
    // over-budget *passthrough* is still a structurally valid model, whereas a
    // shape-incompatible plan is *wrong*, so we refuse the transforms outright
    // rather than letting CSHA fusion / adapters / thinning ride on a broken
    // graph.
    //
    // For the uniform residual transformers WGGO targets today the feeder
    // (`inter_layer_dims`) yields a single `d_model` on every edge, so this is
    // satisfied-by-construction and never triggers; it becomes binding once
    // per-layer or dim-changing activation widths are introduced (see the
    // `wggo_shape` module docs).
    let shape_warns = shape_warnings(&graph, input.layer_shape.d_model);
    if !shape_warns.is_empty() {
        let empty_inter = InterLayerPlan {
            layers: Vec::new(),
            total_us: 0.0,
            peak_memory_bytes: 0,
            pipeline_stages: 1,
        };
        let empty_applied = crate::wggo_apply::AppliedPlan {
            layers: Vec::new(),
            total_us: 0.0,
            peak_memory_bytes: 0,
        };
        let schedule = build_schedule(&empty_inter, &empty_applied);
        return WggoPlan {
            mode: WggoMode::Off,
            target_gpu: gpu.name.to_string(),
            graph,
            inter_layer: empty_inter,
            per_layer: Vec::new(),
            resolutions: Vec::new(),
            applied: empty_applied,
            schedule,
            template_stats: TemplateStats::default(),
            weight_analysis: WeightAnalysisReport::default(),
            estimated_solve_us: t0.elapsed().as_micros() as u64,
            warnings: shape_warns,
            // The §2.4 refusal must not advertise decisions of any kind.
            cfie_inference: Vec::new(),
            cpkd_distill: Vec::new(),
        };
    }

    // Off mode: build a trivial plan that passes through decisions.
    if input.mode == WggoMode::Off {
        let n = graph.layers.len();
        let lut = build_lut(&input.layer_shape, gpu, &input.lut_axes);
        let luts: Vec<LayerCostLut> = vec![lut; n.max(1)];
        let dp_cfg = DpConfig {
            cluster: input.cluster.clone(),
            importance: input.importance.clone(),
            ..Default::default()
        };
        // Off mode bypasses optimization, so an over-budget passthrough is the
        // intended behavior rather than a refusal.
        let inter =
            dp_solve(&graph, &luts, &dp_cfg, gpu).unwrap_or_else(|_| passthrough_plan(&graph, &luts, &dp_cfg, gpu));
        let mut ilp_defaults: Vec<LayerIlpConstraints> = if input.ilp_constraints.is_empty() {
            default_constraints_for(&graph)
        } else {
            input.ilp_constraints
        };
        let weight_analysis = match input.cached_analysis {
            Some(ref cached) => cached.clone(),
            None => run_weight_analysis(
                &graph,
                &input.layer_shape,
                &ilp_defaults,
                input.weights,
                &input.analysis_config,
                input.scorer.as_deref(),
            ),
        };
        weight_analysis.apply_to(&mut ilp_defaults);
        let (per_layer, template_stats, cfie_choices, cpkd_choices) =
            ilp_solve_all_templated_cpkd(&luts, &ilp_defaults);
        let cfie_inference = surface_from_plan(&inter, &cfie_choices, &ilp_defaults);
        let cpkd_distill = cpkd_surface_from_plan(&inter, &cpkd_choices, &ilp_defaults);
        let applied = apply(&inter, &per_layer);
        let schedule = build_schedule(&inter, &applied);
        return WggoPlan {
            mode: WggoMode::Off,
            target_gpu: gpu.name.to_string(),
            graph,
            inter_layer: inter,
            per_layer,
            resolutions: Vec::new(),
            applied,
            schedule,
            template_stats,
            weight_analysis,
            estimated_solve_us: t0.elapsed().as_micros() as u64,
            warnings: Vec::new(),
            cfie_inference,
            cpkd_distill,
        };
    }

    // 2. Cost LUT.
    let lut = build_lut(&input.layer_shape, gpu, &input.lut_axes);
    let n = graph.layers.len();
    let luts: Vec<LayerCostLut> = vec![lut; n.max(1)];

    // 4. Level 1 DP.
    let dp_cfg = DpConfig {
        cluster: input.cluster.clone(),
        importance: input.importance.clone(),
        ..Default::default()
    };
    // G4 degradation ladder, rung 0: the inter-layer DP is shared by Full and
    // Greedy, so if it refuses the budget neither optimized mode can proceed —
    // degrade to Off and run the downstream passes independently, with a
    // warning, rather than emitting a silently over-budget optimization.
    let mut warnings: Vec<String> = Vec::new();
    let mut effective_mode = input.mode;
    let inter = match dp_solve(&graph, &luts, &dp_cfg, gpu) {
        Ok(p) => p,
        Err(e) => {
            warnings.push(format!(
                "inter-layer DP infeasible ({e}); degraded {} -> off (downstream passes run independently)",
                effective_mode.as_str()
            ));
            effective_mode = WggoMode::Off;
            passthrough_plan(&graph, &luts, &dp_cfg, gpu)
        }
    };

    // 5. Level 2 ILP (per layer, independent once inter-layer decisions
    //    are fixed — paper §5.2).
    let mut ilp_constraints: Vec<LayerIlpConstraints> = if input.ilp_constraints.is_empty() {
        default_constraints_for(&graph)
    } else {
        input.ilp_constraints
    };
    // Stage 3: weight analysis — overwrites head_importance and (when
    // the caller left it zero) min_retained_importance on each layer's
    // constraints, before the ILP runs.
    let weight_analysis = match input.cached_analysis {
        Some(ref cached) => cached.clone(),
        None => run_weight_analysis(
            &graph,
            &input.layer_shape,
            &ilp_constraints,
            input.weights,
            &input.analysis_config,
            input.scorer.as_deref(),
        ),
    };
    weight_analysis.apply_to(&mut ilp_constraints);
    // NOTE: FASE is intentionally *not* pre-pruned on sharded layers here.
    // Letting the ILP pick a fused optimizer step and then resolving the
    // resulting FaseVsCpdt conflict (DeferFaseStep) is the architecturally
    // correct path: it keeps the conflict detector live in production (G8) and
    // surfaces a visible resolution in the report, instead of silently
    // disabling the option before it can ever conflict.
    let (mut per_layer, mut template_stats, mut cfie_choices, mut cpkd_choices) =
        match effective_mode {
            WggoMode::Greedy => {
                let (sols, cfie, cpkd) = ilp_solve_all_greedy_cpkd(&luts, &ilp_constraints);
                (sols, TemplateStats::default(), cfie, cpkd)
            }
            _ => ilp_solve_all_templated_cpkd(&luts, &ilp_constraints),
        };

    // G4 degradation ladder, rung 1: if the full branch-and-bound ILP could not
    // make a single layer feasible, fall back to the greedy solver before
    // giving up entirely.
    if effective_mode == WggoMode::Full
        && !per_layer.is_empty()
        && per_layer.iter().all(|s| !s.feasible)
    {
        warnings.push("full ILP found no feasible layer; degraded full -> greedy".to_string());
        effective_mode = WggoMode::Greedy;
        let (sols, cfie, cpkd) = ilp_solve_all_greedy_cpkd(&luts, &ilp_constraints);
        per_layer = sols;
        cfie_choices = cfie;
        cpkd_choices = cpkd;
        template_stats = TemplateStats::default();
    }

    // 6-7. Conflict detection + resolution — skipped in Off (or degraded-Off)
    //      mode, which runs the downstream passes independently.
    let resolutions = if effective_mode == WggoMode::Off {
        Vec::new()
    } else {
        let layer_decisions = build_layer_decisions(&inter, &per_layer);
        // Capture the pre-resolution cost so greedy mode can detect a
        // resolution-induced regression below.
        let cost_before: f64 = per_layer
            .iter()
            .filter(|s| s.cost_us.is_finite())
            .map(|s| s.cost_us)
            .sum();
        let (resolved, mut resolutions) = greedy_resolve(layer_decisions);
        splice_resolved(&mut per_layer, &resolved);

        // G3 (greedy mode only): re-cost the resolved configuration and, if the
        // conflict resolution worsened it beyond the threshold, escalate the
        // conflicting layers to the full ILP and re-resolve.  Greedy's fast
        // local heuristics can leave a lot on the table once a conflict forces
        // a change, so spending the full branch-and-bound on just those layers
        // recovers quality without the cost of a full-model ILP.
        if effective_mode == WggoMode::Greedy && !resolutions.is_empty() {
            let cost_after = recost_total(
                &luts,
                &per_layer,
                &ilp_constraints,
                &cfie_choices,
                &cpkd_choices,
            );
            if cost_after > cost_before * (1.0 + GREEDY_RECOST_THRESHOLD) {
                let conflicting: std::collections::HashSet<u32> =
                    resolutions.iter().filter_map(|r| r.layer()).collect();
                for (i, inter_layer) in inter.layers.iter().enumerate() {
                    if conflicting.contains(&inter_layer.layer_index) {
                        // Defensive indexing (mirrors recost_total) — a caller
                        // may pass fewer ilp_constraints than layers.
                        if let (Some(lut), Some(cons)) =
                            (luts.get(i).or_else(|| luts.first()), ilp_constraints.get(i).or_else(|| ilp_constraints.first()))
                        {
                            let (sol, cfie_choice, cpkd_choice) = ilp_solve_layer_cpkd(lut, cons);
                            per_layer[i] = sol;
                            if let Some(slot) = cfie_choices.get_mut(i) {
                                *slot = cfie_choice;
                            }
                            if let Some(slot) = cpkd_choices.get_mut(i) {
                                *slot = cpkd_choice;
                            }
                        }
                    }
                }
                // Re-detect + re-resolve on the escalated decisions.
                let ld2 = build_layer_decisions(&inter, &per_layer);
                let (resolved2, resolutions2) = greedy_resolve(ld2);
                splice_resolved(&mut per_layer, &resolved2);
                resolutions = resolutions2;
            }
        }
        resolutions
    };

    // CEP honesty advisories: structural decisions the plan carries that
    // downstream codegen does not (or only partially) lowers. These fire
    // only when the decision was actually taken (which requires an
    // informed importance signal — evidence-free plans are pinned to
    // keep-all/full-width by `importance_informed`).
    {
        let pruned_layers = per_layer
            .iter()
            .filter(|s| s.decision.keep_head.iter().any(|k| !k))
            .count();
        if pruned_layers > 0 {
            warnings.push(format!(
                "[cep] {pruned_layers} layer(s) plan pruned attention heads; \
                 head pruning is lowered ONLY through the CSHA fused dispatch \
                 (active_heads guard + launch-grid clamp) — on other attention \
                 paths the decision is report-only"
            ));
        }
        let max_ffn = per_layer.iter().map(|s| s.decision.ffn_width).max().unwrap_or(0);
        let thinned_layers = per_layer
            .iter()
            .filter(|s| s.decision.ffn_width < max_ffn)
            .count();
        if thinned_layers > 0 {
            warnings.push(format!(
                "[cep] {thinned_layers} layer(s) plan a thinned FFN width; \
                 ffn_width has NO codegen consumer (dropped at the \
                 PerLayerOverride boundary) — layers run at full width and \
                 the decision is report-only"
            ));
        }
    }

    // 8. Apply + communication schedule.  (The §2.4 shape-compatibility
    // precondition was checked up front; a violation would have forced Off
    // above, so reaching here means the plan is shape-compatible.)
    let applied = apply(&inter, &per_layer);
    let schedule = build_schedule(&inter, &applied);
    // G20 advisory surface: chosen CFIE inference decisions per non-pruned
    // layer.  Empty (gate off) keeps the plan byte-identical to today.
    let cfie_inference = surface_from_plan(&inter, &cfie_choices, &ilp_constraints);
    // CPKD advisory surface (v1): same contract as the CFIE sidecar.
    let cpkd_distill = cpkd_surface_from_plan(&inter, &cpkd_choices, &ilp_constraints);

    WggoPlan {
        mode: effective_mode,
        target_gpu: gpu.name.to_string(),
        graph,
        inter_layer: inter,
        per_layer,
        resolutions,
        applied,
        schedule,
        template_stats,
        weight_analysis,
        estimated_solve_us: t0.elapsed().as_micros() as u64,
        warnings,
        cfie_inference,
        cpkd_distill,
    }
}

/// Inter-layer activation widths for a layer, by transformer role.
///
/// Residual-stream sublayers (`Attention` / `Ffn` / `Block`) carry `d_model`
/// in and out: per-head pruning and FFN thinning are absorbed by the output
/// projection (`W_O` / `W_down`), so the *inter-layer* tensor width is
/// invariant under WGGO's structural decisions.  The boundary roles expose one
/// real width and one unknown — `Embedding` consumes token ids (unknown width)
/// and produces `d_model`; `LmHead` consumes `d_model` and produces vocab
/// logits (not an inter-layer activation).  `Other` is an unclassified op group
/// with no known residual projection, so both widths are unknown.  Unknown
/// (`None`) widths are treated as compatible by [`crate::wggo_shape::classify`]
/// (fail-safe — only provable mismatches are flagged).
/// Conservative role-based numerical-sensitivity floor for the per-layer ILP.
///
/// The ILP now costs optimizer precision ([`crate::wggo_cost::optimizer_us`]),
/// so with no sensitivity signal it would recommend the cheapest (8-bit) Adam
/// moments for *every* layer — including the numerically fragile ones.
/// Production callers pass no explicit `ilp_constraints`, and the weight-
/// analysis stage populates importance but never sensitivity, so without this
/// floor the embeddings / LM-head / norms would be advised down to 8-bit.
/// Assign a coarse floor by [`LayerRole`]: fragile layers are pinned to high
/// precision; the transformer blocks — where low-bit Adam moments are well
/// established — may be lowered.
///
/// The graph builder buckets most non-`blocks.N` params (embeddings, norms, the
/// LM head, the raw input) into the catch-all `Other` role, so `Other` is held
/// at ≥16-bit as the conservative default.  This is intentionally coarse; a
/// gradient/spectral per-layer sensitivity score (cf. `cpdt_sensitivity`) is the
/// natural refinement.  Note the optimizer-precision decision is currently
/// *advisory* — it is surfaced in the plan/report but not yet lowered to
/// optimizer codegen (`PerLayerOverride` does not carry it), so this governs the
/// recommendation, not yet generated code.
fn role_sensitivity_floor(role: LayerRole) -> f64 {
    match role {
        // ≥ critical_prec_threshold (0.9 default) → forces 32-bit moments.
        LayerRole::Embedding | LayerRole::LmHead => 0.95,
        // ≥ high_prec_threshold (0.5 default) → forces ≥16-bit.
        LayerRole::Other => 0.6,
        // The bulk transformer compute may use low-bit Adam moments.
        LayerRole::Attention | LayerRole::Ffn | LayerRole::Block => 0.0,
    }
}

/// Per-layer default ILP constraints with the [`role_sensitivity_floor`]
/// applied (used when the caller supplies no explicit `ilp_constraints`).
fn default_constraints_for(graph: &OptGraph) -> Vec<LayerIlpConstraints> {
    graph
        .layers
        .iter()
        .map(|l| LayerIlpConstraints {
            sensitivity: role_sensitivity_floor(l.role),
            ..Default::default()
        })
        .collect()
}

fn inter_layer_dims(role: LayerRole, d_model: u64) -> (Option<u64>, Option<u64>) {
    // A zero `d_model` is a defensively-initialised / unknown shape, not a real
    // width: report unknown on every edge so the gate stays fail-safe.
    if d_model == 0 {
        return (None, None);
    }
    match role {
        LayerRole::Attention | LayerRole::Ffn | LayerRole::Block => (Some(d_model), Some(d_model)),
        LayerRole::Embedding => (None, Some(d_model)),
        LayerRole::LmHead => (Some(d_model), None),
        LayerRole::Other => (None, None),
    }
}

/// Build the per-layer [`LayerShapeInfo`] records the §2.4 shape gate validates.
///
/// Widths come from [`inter_layer_dims`]; producer edges are the graph's
/// `depends_on` lists.  `d_model` is the uniform residual-stream width the
/// driver sizes the model with.
fn build_shape_infos(graph: &OptGraph, d_model: u64) -> Vec<LayerShapeInfo> {
    graph
        .layers
        .iter()
        .map(|l| {
            let (input_dim, output_dim) = inter_layer_dims(l.role, d_model);
            LayerShapeInfo {
                layer: l.index,
                name: l.name.clone(),
                input_dim,
                output_dim,
                depends_on: l.depends_on.clone(),
            }
        })
        .collect()
}

/// Run the §2.4 shape-compatibility gate over `graph` and return one warning
/// message per provably-incompatible inter-layer edge (empty when the
/// constraint holds).  See the gate's call site in [`run`] for how a non-empty
/// result forces a passthrough refusal.
fn shape_warnings(graph: &OptGraph, d_model: u64) -> Vec<String> {
    crate::wggo_shape::validate(&build_shape_infos(graph, d_model))
        .iter()
        .map(|v| v.message())
        .collect()
}

/// Run Stage 3 over `graph`, using the first layer's `num_heads` as the
/// shape hint (the driver constructs all layers with uniform num_heads
/// today).  When `weights` is `None`, falls back to `NullWeightProvider`
/// so every layer gets a uniform score vector.
///
/// When `scorer` is `Some`, its `score_layer` output is applied
/// **after** the magnitude analysis: for each layer where the scorer
/// returns `Some(HeadImportance)`, the magnitude-derived `head_scores`
/// in the report are replaced by the scorer's per-head values (converted
/// to `f64`).  Layers where the scorer returns `None` retain the
/// magnitude scores unchanged.  An `importance_source` line is emitted
/// to stderr for each layer so that `--wggo-report` consumers can
/// verify which path fired.
fn run_weight_analysis(
    graph: &crate::wggo_graph::OptGraph,
    layer_shape: &LayerShape,
    constraints: &[LayerIlpConstraints],
    weights: Option<&dyn WeightProvider>,
    config: &AnalysisConfig,
    scorer: Option<&dyn GradientScorer>,
) -> WeightAnalysisReport {
    let num_heads = constraints
        .first()
        .map(|c| c.num_heads as usize)
        .unwrap_or(8);
    let null = NullWeightProvider;
    let provider: &dyn WeightProvider = weights.unwrap_or(&null);
    let mut report = analyze_weights(graph, layer_shape, num_heads, provider, config);

    // Scorer override pass: for each layer, attempt to get a gradient-
    // informed score and substitute it for the magnitude baseline.
    if let Some(sc) = scorer {
        for (layer, imp) in graph.layers.iter().zip(report.per_layer.iter_mut()) {
            let has_real_gradient_for_layer = sc.has_gradient_data_for_layer(&layer.name);
            // Calibrated gradient evidence is a REAL signal for the
            // informed gates (the field docs promise 'weight analysis with
            // an actual provider, or calibration') — without this, users
            // with measured gradients but no --wggo-weights stayed
            // uninformed and quantization was silently impossible.
            if has_real_gradient_for_layer {
                imp.has_signal = true;
            }
            let from_scorer = sc.score_layer(&layer.name, layer_shape);
            let (new_scores, source) = match from_scorer {
                Some(hi) => {
                    // Truth-in-logging: a CalibratedGradientScorer may return
                    // Some(...) via its internal magnitude fallback when the
                    // sidecar lacks an entry for this layer.  Only claim
                    // "gradient (calibrated)" when the scorer confirmed it has
                    // real sidecar-derived data for the layer.
                    let label = if has_real_gradient_for_layer {
                        "gradient (calibrated)"
                    } else {
                        "magnitude (fallback within CalibratedGradientScorer)"
                    };
                    (
                        hi.per_head.iter().map(|&v| v as f64).collect::<Vec<f64>>(),
                        label,
                    )
                }
                None => (
                    score_layer_magnitude(&layer.name, layer_shape, provider)
                        .per_head
                        .iter()
                        .map(|&v| v as f64)
                        .collect::<Vec<f64>>(),
                    "magnitude (fallback)",
                ),
            };
            eprintln!("[wggo] layer:{} importance_source={}", layer.name, source);
            imp.head_scores = new_scores;
        }
    }

    report
}

/// Extract `GpuSpec` for the named target or default.
#[allow(dead_code)]
fn resolve_gpu(name: &str) -> &'static GpuSpec {
    find_gpu(name).unwrap_or_else(default_gpu)
}

/// Convenience: run WGGO on a Wengert list with default cluster/importance/
/// LUT axes.  Used by the compile-pipeline integration point so callers
/// don't have to hand-assemble `WggoInput`.
///
/// * `target` — GPU name (e.g. "H100"); falls back to the default when
///              unknown.
/// * `mode_str` — "full" | "greedy" | "off" | "auto" | "disable".
///              Returns `None` if the string is invalid.
/// * `world_size` — number of devices (drives the `ClusterSpec`).
pub fn run_on_wengert(
    wengert: &WengertList,
    target: &str,
    mode_str: &str,
    world_size: usize,
) -> Option<WggoPlan> {
    let mode = WggoMode::parse(mode_str)?;
    let cluster = ClusterSpec {
        num_gpus: world_size.max(1) as u32,
        ..ClusterSpec::default()
    };
    let input = WggoInput {
        mode,
        target,
        wengert,
        // Reasonable defaults for a small transformer layer; used only to
        // size the LUT — the graph drives which layers are present.
        layer_shape: LayerShape {
            batch: 1,
            seq: 1024,
            d_model: 512,
            head_dim: 64,
            n_kv_heads: 4,
            dtype_bytes: 2,
        },
        cluster,
        lut_axes: LutAxes::default(),
        importance: ImportanceScores::default(),
        ilp_constraints: Vec::new(),
        weights: None,
        analysis_config: AnalysisConfig::default(),
        scorer: None,
        cached_analysis: None,
    };
    Some(run(input))
}

/// Extended convenience entry point — accepts an optional weights file
/// and `AnalysisConfig` so Stage 3 can use real importance scoring.
///
/// When `weights_path` fails to load, emits a warning on stderr and
/// falls back to uniform scores rather than failing the compile.
///
/// When `compile_options` is `Some`, [`build_scorer`] is invoked to
/// construct the appropriate [`GradientScorer`] and wire it into
/// `WggoInput`.  Passing `None` preserves the Phase 1 magnitude-only
/// path (used in tests that don't need gradient scoring).
pub fn run_on_wengert_with_weights(
    wengert: &WengertList,
    target: &str,
    mode_str: &str,
    world_size: usize,
    weights_path: Option<&std::path::Path>,
    analysis_config: AnalysisConfig,
    compile_options: Option<&crate::CompileOptions>,
) -> Option<WggoPlan> {
    use crate::wggo_gradient_scorer::build_scorer;
    use std::sync::Arc;

    let mode = WggoMode::parse(mode_str)?;
    let cluster = ClusterSpec {
        num_gpus: world_size.max(1) as u32,
        ..ClusterSpec::default()
    };

    // Check the sidecar cache first — a hit lets us skip the
    // checkpoint load entirely.
    let cached_report = weights_path
        .and_then(crate::wggo_weight_analysis_cache::try_load);

    // Load the weights file up front so its lifetime covers the call.
    // Skip the load when the cache hit already carries the scores.
    // On failure, capture the message so it lands in the plan's report
    // (Phase 9 field 6) — not only on stderr (G11).
    let mut load_warning: Option<String> = None;
    let checkpoint = if cached_report.is_some() {
        None
    } else {
        match weights_path {
            Some(p) => {
                match crate::wggo_weight_analysis_nslweights::NslWeightsCheckpoint::load(p) {
                    Ok(ck) => Some(ck),
                    Err(e) => {
                        let msg = format!(
                            "could not load weights from {}: {} — falling back to uniform importance scores",
                            p.display(),
                            e
                        );
                        eprintln!("[wggo] warning: {msg}");
                        load_warning = Some(msg);
                        None
                    }
                }
            }
            None => None,
        }
    };
    let weights_ref: Option<&dyn WeightProvider> = checkpoint
        .as_ref()
        .map(|ck| ck as &dyn WeightProvider);

    // Build the gradient scorer when compile options are present.
    // `build_scorer` selects Null/Magnitude/Calibrated based on
    // `wggo_importance` and `calibration_sidecar`.  A `Grad` mode
    // without a sidecar is a hard error propagated as `None` (the
    // compile pipeline will have already caught this via its own
    // validation, but we guard here too).
    let scorer: Option<Box<dyn crate::wggo_gradient_scorer::GradientScorer>> =
        if let Some(opts) = compile_options {
            let provider: Arc<dyn WeightProvider + Send + Sync> =
                Arc::new(NullWeightProvider);
            match build_scorer(opts, provider) {
                Ok(s) => Some(s),
                Err(e) => {
                    eprintln!("[wggo] error: {e:?}");
                    return None;
                }
            }
        } else {
            None
        };

    let input = WggoInput {
        mode,
        target,
        wengert,
        layer_shape: LayerShape {
            batch: 1,
            seq: 1024,
            d_model: 512,
            head_dim: 64,
            n_kv_heads: 4,
            dtype_bytes: 2,
        },
        cluster,
        lut_axes: LutAxes::default(),
        importance: ImportanceScores::default(),
        ilp_constraints: Vec::new(),
        weights: weights_ref,
        analysis_config,
        scorer,
        cached_analysis: cached_report.clone(),
    };
    let mut plan = run(input);

    // Surface the weights-load failure (if any) in the report itself, not
    // only on stderr.  Prepended-style: it precedes any degradation warnings
    // run() may have added.
    if let Some(msg) = load_warning {
        plan.warnings.insert(0, msg);
    }

    // Cache-hit reports were passed into run() via `cached_analysis`
    // (pre-solve), so the plan already carries them.
    if cached_report.is_none() {
        if let Some(p) = weights_path {
        // Cache miss — persist the freshly-computed report alongside
        // the checkpoint.  Failures are silent; the cache is advisory.
        let _ = crate::wggo_weight_analysis_cache::store(p, &plan.weight_analysis);
        }
    }

    Some(plan)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wengert::{PrimalOp, WengertOp};
    use std::collections::HashMap;

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

    fn toy_input<'a>(w: &'a WengertList) -> WggoInput<'a> {
        WggoInput {
            mode: WggoMode::Full,
            target: "H100",
            wengert: w,
            layer_shape: LayerShape {
                batch: 1,
                seq: 1024,
                d_model: 512,
                head_dim: 64,
                n_kv_heads: 4,
                dtype_bytes: 2,
            },
            cluster: ClusterSpec::default(),
            lut_axes: LutAxes::default(),
            importance: ImportanceScores::default(),
            ilp_constraints: Vec::new(),
            weights: None,
            analysis_config: AnalysisConfig::default(),
            scorer: None,
            cached_analysis: None,
        }
    }

    #[test]
    fn inter_layer_dims_maps_roles_to_residual_widths() {
        assert_eq!(inter_layer_dims(LayerRole::Attention, 512), (Some(512), Some(512)));
        assert_eq!(inter_layer_dims(LayerRole::Ffn, 512), (Some(512), Some(512)));
        assert_eq!(inter_layer_dims(LayerRole::Block, 512), (Some(512), Some(512)));
        assert_eq!(inter_layer_dims(LayerRole::Embedding, 512), (None, Some(512)));
        assert_eq!(inter_layer_dims(LayerRole::LmHead, 512), (Some(512), None));
        assert_eq!(inter_layer_dims(LayerRole::Other, 512), (None, None));
    }

    #[test]
    fn inter_layer_dims_zero_d_model_is_unknown() {
        // A zero/defaulted d_model must not synthesise Some(0) widths (which
        // would otherwise drive false classifications): report unknown.
        assert_eq!(inter_layer_dims(LayerRole::Block, 0), (None, None));
        assert_eq!(inter_layer_dims(LayerRole::Attention, 0), (None, None));
    }

    #[test]
    fn build_shape_infos_for_residual_blocks_carry_d_model_and_validate_clean() {
        let w = two_block_wengert();
        let graph = build_graph(&w);
        let infos = build_shape_infos(&graph, 512);
        // Every transformer block exposes d_model on both inter-layer edges.
        for info in infos.iter().filter(|i| i.name.starts_with("blocks.")) {
            assert_eq!(info.input_dim, Some(512));
            assert_eq!(info.output_dim, Some(512));
        }
        // The uniform residual chain satisfies §2.4 by construction.
        assert!(crate::wggo_shape::validate(&infos).is_empty());
    }

    #[test]
    fn shape_warnings_empty_for_uniform_residual_graph() {
        let w = two_block_wengert();
        let graph = build_graph(&w);
        assert!(shape_warnings(&graph, 512).is_empty());
    }

    #[test]
    fn run_residual_model_emits_no_shape_warnings() {
        let w = two_block_wengert();
        let plan = run(toy_input(&w));
        assert!(
            !plan.warnings.iter().any(|m| m.contains("shape_incompatible")),
            "uniform residual transformer must not trip the §2.4 shape gate: {:?}",
            plan.warnings
        );
    }

    #[test]
    fn role_sensitivity_floor_pins_fragile_roles_high_and_blocks_low() {
        assert!(role_sensitivity_floor(LayerRole::Embedding) >= 0.9);
        assert!(role_sensitivity_floor(LayerRole::LmHead) >= 0.9);
        assert!(role_sensitivity_floor(LayerRole::Other) >= 0.5);
        assert!(role_sensitivity_floor(LayerRole::Other) < 0.9);
        assert_eq!(role_sensitivity_floor(LayerRole::Block), 0.0);
        assert_eq!(role_sensitivity_floor(LayerRole::Attention), 0.0);
        assert_eq!(role_sensitivity_floor(LayerRole::Ffn), 0.0);
    }

    #[test]
    fn run_without_weight_signal_keeps_full_precision_moments() {
        // End-to-end: `optim_m/v_bits` is now LOWERED to real optimizer-state
        // allocation (PerLayerOverride → stmt.rs dtype lists), so a plan built
        // with NO weight/calibration evidence must not quantize anything —
        // the toy input has no weight provider, every layer is
        // sensitivity-uninformed, and `prec_allowed` forbids sub-32 bits.
        // (Role floors — Other ≥16-bit, embeddings/head 32-bit, blocks
        // low-bit-allowed — still shape decisions once a real signal sets
        // `sensitivity_informed`; pinned at the ILP level by
        // `low_sensitivity_picks_low_optimizer_precision_when_informed`.)
        let w = two_block_wengert();
        let plan = run(toy_input(&w));
        for layer in &plan.applied.layers {
            assert_eq!(
                layer.optim_m_bits, 32,
                "layer {} must keep 32-bit m without evidence, got {}",
                layer.layer_name, layer.optim_m_bits
            );
            assert_eq!(
                layer.optim_v_bits, 32,
                "layer {} must keep 32-bit v without evidence, got {}",
                layer.layer_name, layer.optim_v_bits
            );
        }
    }

    #[test]
    fn full_mode_produces_plan_with_applied_layers() {
        let w = two_block_wengert();
        let plan = run(toy_input(&w));
        assert_eq!(plan.mode, WggoMode::Full);
        assert!(plan.inter_layer.layers.len() >= 2);
        assert!(!plan.applied.layers.is_empty());
        assert!(plan.applied.total_us >= 0.0);
    }

    #[test]
    fn off_mode_still_produces_some_plan() {
        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        inp.mode = WggoMode::Off;
        let plan = run(inp);
        assert_eq!(plan.mode, WggoMode::Off);
        assert!(plan.resolutions.is_empty());
    }

    #[test]
    fn greedy_mode_returns_resolved_plan() {
        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        inp.mode = WggoMode::Greedy;
        let plan = run(inp);
        assert_eq!(plan.mode, WggoMode::Greedy);
        // Resolutions is only non-empty when conflicts fire; either way
        // the field must be present.
        let _ = plan.resolutions.len();
    }

    #[test]
    fn render_report_contains_expected_sections() {
        let w = two_block_wengert();
        let plan = run(toy_input(&w));
        let rep = plan.render_report();
        assert!(rep.contains("WGGO Global Optimization Report"));
        assert!(rep.contains("Level 1"));
        assert!(rep.contains("Level 2"));
        assert!(rep.contains("Global metrics"));
    }

    #[test]
    fn summary_is_single_line_compact() {
        let w = two_block_wengert();
        let plan = run(toy_input(&w));
        let s = plan.summary();
        assert!(!s.is_empty());
        assert!(!s.contains('\n'));
    }

    #[test]
    fn plan_is_deterministic_across_runs() {
        let w = two_block_wengert();
        let plan1 = run(toy_input(&w));
        let plan2 = run(toy_input(&w));
        // Reports (sans solve-time numeric field) must match bit-for-bit.
        let r1 = plan1.render_report();
        let r2 = plan2.render_report();
        // Strip the "Solve time" line because it depends on wall-clock.
        let strip = |s: &str| -> String {
            s.lines()
                .filter(|l| !l.contains("Solve time"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(strip(&r1), strip(&r2));
    }

    #[test]
    fn mode_parse_roundtrip() {
        for m in [WggoMode::Full, WggoMode::Greedy, WggoMode::Off] {
            assert_eq!(WggoMode::parse(m.as_str()), Some(m));
        }
        assert!(WggoMode::parse("nonsense").is_none());
    }

    #[test]
    fn run_on_wengert_accepts_canonical_mode_strings() {
        let w = two_block_wengert();
        for mode in ["full", "greedy", "off", "auto"] {
            let plan = run_on_wengert(&w, "H100", mode, 1);
            assert!(plan.is_some(), "mode '{}' should be accepted", mode);
        }
        assert!(run_on_wengert(&w, "H100", "nonsense", 1).is_none());
    }

    #[test]
    fn fase_disabled_on_sharded_layers() {
        // Multi-GPU run will trigger ZeRO sharding in the inter-layer DP,
        // which must in turn force allow_fase=false on those layers — so
        // the resulting AppliedLayer must not report fase_fused=true.
        let w = two_block_wengert();
        let plan = run_on_wengert(&w, "H100", "full", 8).expect("plan");
        for layer in &plan.applied.layers {
            if layer.shard_factor > 1 {
                assert!(
                    !layer.fase_fused,
                    "sharded layer {} should not have FASE fused",
                    layer.layer_index
                );
            }
        }
    }

    #[test]
    fn fase_appears_in_report() {
        let w = two_block_wengert();
        let plan = run(toy_input(&w));
        let rep = plan.render_report();
        assert!(rep.contains("FASE="));
    }

    #[test]
    fn greedy_mode_uses_greedy_solver() {
        // The greedy solver explores at most a handful of nodes per layer
        // (sum-of-domains, not product).  The full solver explores
        // hundreds of thousands.  Run both modes and confirm Greedy
        // produces drastically lower per-layer node counts.
        let w = two_block_wengert();
        let mut inp_full = toy_input(&w);
        inp_full.mode = WggoMode::Full;
        let mut inp_greedy = toy_input(&w);
        inp_greedy.mode = WggoMode::Greedy;
        let full = run(inp_full);
        let greedy = run(inp_greedy);
        let full_nodes: u64 = full.per_layer.iter().map(|s| s.nodes_explored).sum();
        let greedy_nodes: u64 = greedy.per_layer.iter().map(|s| s.nodes_explored).sum();
        assert!(
            greedy_nodes * 100 < full_nodes,
            "greedy_nodes={greedy_nodes} full_nodes={full_nodes}"
        );
    }

    #[test]
    fn run_on_wengert_with_weights_handles_missing_file() {
        // A nonexistent weights path should warn and fall back to
        // uniform scores, not fail the call.
        let w = two_block_wengert();
        let plan = run_on_wengert_with_weights(
            &w,
            "H100",
            "full",
            1,
            Some(std::path::Path::new("/nonexistent/path.nslweights")),
            crate::wggo_weight_analysis::AnalysisConfig::default(),
            None,
        )
        .expect("plan");
        // Every layer should be counted as without_weights since the
        // checkpoint didn't load.
        assert_eq!(
            plan.weight_analysis.layers_without_weights,
            plan.weight_analysis.per_layer.len() as u32
        );
    }

    #[test]
    fn weight_analysis_runs_with_null_provider() {
        // Default path (weights=None) → uniform scores, but analyze()
        // still populates per_layer and the report field.
        let w = two_block_wengert();
        let plan = run(toy_input(&w));
        assert_eq!(plan.weight_analysis.per_layer.len(), plan.inter_layer.layers.len());
        assert!(plan.weight_analysis.layers_without_weights >= 1);
        let rep = plan.render_report();
        assert!(rep.contains("Weight analysis"));
    }

    #[test]
    fn template_stats_reuse_identical_blocks() {
        // The two-block toy Wengert produces two `blocks.N` layers (role
        // Block) plus the catch-all `other` bucket.  The role-based sensitivity
        // floor gives Block and Other distinct constraints, so Full mode solves
        // two templates (Block, Other) and the second block reuses the first.
        let w = two_block_wengert();
        let plan = run(toy_input(&w));
        assert_eq!(plan.template_stats.templates_solved, 2);
        assert!(plan.template_stats.template_hits >= 1);
        let rep = plan.render_report();
        assert!(rep.contains("Templates:"));
    }

    #[test]
    fn template_stats_zero_in_greedy_mode() {
        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        inp.mode = WggoMode::Greedy;
        let plan = run(inp);
        // Greedy bypasses the templated solver — counters are default.
        assert_eq!(plan.template_stats.templates_solved, 0);
        assert_eq!(plan.template_stats.template_hits, 0);
    }

    #[test]
    fn schedule_present_for_multi_gpu_run() {
        // Multi-GPU *under memory pressure* triggers ZeRO sharding via the
        // inter-layer DP, which in turn forces the schedule to issue
        // collectives.  (Without pressure the DP correctly keeps shard=1 and
        // the schedule stays empty — sharding is a memory/comm trade-off, not
        // an unconditional consequence of having multiple GPUs.)
        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        let gpu = find_gpu("H100").unwrap_or_else(default_gpu);
        let lut = build_lut(&inp.layer_shape, gpu, &inp.lut_axes);
        let one = lut.argmin_feasible().expect("feasible entry").4;
        let sharded = 3 * one.param_bytes / 8 + one.activation_bytes;
        inp.cluster = ClusterSpec {
            num_gpus: 8,
            memory_budget: 2 * sharded + sharded / 2,
            max_stages: 1,
            interconnect_gbs: 300.0,
        };
        let plan = run(inp);
        assert!(plan.schedule.total_collectives >= 1);
        let rep = plan.render_report();
        assert!(rep.contains("Communication schedule"));
    }

    #[test]
    fn schedule_empty_for_single_gpu_run() {
        let w = two_block_wengert();
        let plan = run_on_wengert(&w, "H100", "full", 1).expect("plan");
        assert_eq!(plan.schedule.total_collectives, 0);
    }

    /// Helper: a memory-pressured cluster that forces the DP to shard every
    /// layer (the sharded footprint of all layers fits, the unsharded one does
    /// not) — so the DP stays feasible and conflicts fire via the real path.
    fn sharded_cluster(inp: &WggoInput) -> ClusterSpec {
        let gpu = find_gpu("H100").unwrap_or_else(default_gpu);
        let lut = build_lut(&inp.layer_shape, gpu, &inp.lut_axes);
        let one = lut.argmin_feasible().expect("feasible entry").4;
        let sharded = 3 * one.param_bytes / 8 + one.activation_bytes;
        let n = build_graph(inp.wengert).layers.len() as u64;
        ClusterSpec {
            num_gpus: 8,
            memory_budget: (n + 1) * sharded,
            max_stages: 1,
            interconnect_gbs: 300.0,
        }
    }

    #[test]
    fn fase_vs_cpdt_conflict_fires_and_defers() {
        // FASE is no longer pre-pruned on sharded layers (G8): the ILP picks a
        // fused optimizer step and the now-live FaseVsCpdt conflict defers it.
        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        inp.cluster = sharded_cluster(&inp);
        let plan = run(inp);
        assert!(
            plan.resolutions
                .iter()
                .any(|r| matches!(r, Resolution::DeferFaseStep { .. })),
            "expected a DeferFaseStep resolution, got {:?}",
            plan.resolutions
        );
    }

    #[test]
    fn wrga_vs_cpdt_conflict_fires_when_sharded() {
        // Force a large adapter on every layer (no rank-0 option) on a sharded
        // cluster.  The driver now feeds a real adapter-allreduce cost (G8), so
        // the WrgaVsCpdt detector fires and the resolver drops the adapter.
        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        inp.lut_axes = LutAxes {
            adapter_ranks: vec![16],
            ..LutAxes::default()
        };
        inp.cluster = sharded_cluster(&inp);
        let plan = run(inp);
        assert!(
            plan.resolutions
                .iter()
                .any(|r| matches!(r, Resolution::RemoveWrgaAdapter { .. })),
            "expected a RemoveWrgaAdapter resolution, got {:?}",
            plan.resolutions
        );
    }

    #[test]
    fn greedy_recost_escalates_conflicting_layers_to_full_ilp() {
        // Sharded greedy run: the ILP picks a fused FASE step, the FaseVsCpdt
        // conflict defers it, and (with a large FASE speedup) that deferral is
        // a big cost regression — so the conflicting layers are re-solved by
        // the full branch-and-bound ILP (G3).  The greedy solver explores only
        // a handful of nodes; the full solver explores far more.
        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        inp.mode = WggoMode::Greedy;
        inp.cluster = sharded_cluster(&inp);
        let n = build_graph(&w).layers.len();
        let mut cons = LayerIlpConstraints::default();
        cons.fase_backward_speedup = 0.5;
        inp.ilp_constraints = vec![cons; n];
        let plan = run(inp);
        let nodes: Vec<u64> = plan.per_layer.iter().map(|s| s.nodes_explored).collect();
        assert!(
            nodes.iter().any(|&n| n > 100),
            "expected a conflicting layer escalated to the full ILP, got nodes {nodes:?}"
        );
    }

    #[test]
    fn greedy_without_conflicts_does_not_escalate() {
        // Single-GPU greedy run: no sharding → no conflicts → no escalation, so
        // the cheap greedy solutions are kept (few nodes explored).
        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        inp.mode = WggoMode::Greedy;
        let plan = run(inp);
        assert!(
            plan.resolutions.is_empty(),
            "single-GPU run should produce no conflicts"
        );
        assert!(
            plan.per_layer.iter().all(|s| s.nodes_explored < 100),
            "no conflicts → greedy solutions kept, no escalation"
        );
    }

    #[test]
    fn full_mode_degrades_to_off_on_infeasible_budget() {
        // 1-byte budget, no shard/prune relief → the DP refuses; Full must
        // degrade gracefully to Off with a warning rather than crash or emit an
        // over-budget plan (G4).
        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        inp.cluster = ClusterSpec {
            num_gpus: 1,
            memory_budget: 1,
            max_stages: 1,
            interconnect_gbs: 300.0,
        };
        let plan = run(inp);
        assert_eq!(
            plan.mode,
            WggoMode::Off,
            "Full should degrade to Off when the DP refuses"
        );
        assert!(
            plan.warnings.iter().any(|w| w.contains("infeasible")),
            "degradation must record a warning, got {:?}",
            plan.warnings
        );
        assert!(
            !plan.applied.layers.is_empty(),
            "degraded run still produces a usable plan"
        );
    }

    #[test]
    fn report_contains_warnings_section() {
        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        inp.cluster = ClusterSpec {
            num_gpus: 1,
            memory_budget: 1,
            max_stages: 1,
            interconnect_gbs: 300.0,
        };
        let rep = run(inp).render_report();
        assert!(rep.contains("Warnings / limitations:"));
        assert!(rep.contains("WARNING:"));
    }

    #[test]
    fn clean_run_has_no_warnings() {
        let w = two_block_wengert();
        let plan = run(toy_input(&w));
        assert!(plan.warnings.is_empty());
        assert!(plan.render_report().contains("Warnings / limitations: 0"));
    }

    /// G11: a weights-load failure must surface in the plan's report
    /// (`warnings` -> "Warnings / limitations" section), not only on stderr.
    #[test]
    fn weights_load_failure_surfaces_in_report() {
        let w = two_block_wengert();
        let missing = std::path::Path::new("definitely/does/not/exist.nslweights");
        let plan = run_on_wengert_with_weights(
            &w,
            "H100",
            "full",
            1,
            Some(missing),
            AnalysisConfig::default(),
            None,
        )
        .expect("plan is still produced via uniform-importance fallback");
        assert!(
            plan.warnings.iter().any(|m| m.contains("could not load weights")),
            "load failure missing from plan.warnings: {:?}",
            plan.warnings
        );
        assert!(
            plan.render_report().contains("could not load weights"),
            "load failure missing from rendered report"
        );
    }

    // -----------------------------------------------------------------------
    // GradientScorer integration tests (Task 7)
    // -----------------------------------------------------------------------

    /// When a `CalibratedGradientScorer` is wired into `WggoInput`, the
    /// weight_analysis report for layers present in the sidecar must reflect
    /// the sidecar's gradient scores, not the magnitude-derived ones.
    ///
    /// We use a single dominant head (score=100.0) vs. three weak heads
    /// (score=5.0) and confirm that the report's `head_scores[0]` is
    /// significantly larger than `head_scores[1]` — proving the gradient
    /// path fired rather than the uniform-magnitude fallback.
    #[test]
    fn head_importance_uses_calibrated_scorer_when_available() {
        use crate::wggo_gradient_scorer::{
            CalibratedGradientScorer, GradientScorer, MagnitudeFallbackScorer,
        };
        use crate::wggo_weight_analysis::NullWeightProvider;
        use std::collections::BTreeMap;
        use std::sync::Arc;

        let mut scores = BTreeMap::new();
        // "blocks.0" and "blocks.1" match the toy Wengert layer keys
        // (produced by two_block_wengert via the graph builder).
        // The toy shape has d_model=512, head_dim=64 → 8 heads, so we
        // supply 8 per-head scores to avoid length mismatches.
        scores.insert(
            "blocks.0".to_string(),
            vec![100.0f32, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        );
        scores.insert(
            "blocks.1".to_string(),
            vec![100.0f32, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0],
        );
        let scorer: Box<dyn GradientScorer> = Box::new(CalibratedGradientScorer::new(
            scores,
            MagnitudeFallbackScorer::new(
                Arc::new(NullWeightProvider) as Arc<dyn WeightProvider + Send + Sync>
            ),
        ));

        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        inp.scorer = Some(scorer);
        let plan = run(inp);

        // The weight_analysis must contain per-layer entries.
        assert!(!plan.weight_analysis.per_layer.is_empty());

        // Find the per_layer entry for "blocks.0" (the graph may also have an
        // "other" layer prepended by the graph builder for unpinned ops, so we
        // locate the blocks.0 layer by searching rather than hard-coding index 0).
        let blocks0_idx = plan
            .weight_analysis
            .per_layer
            .iter()
            .enumerate()
            .zip(plan.graph.layers.iter())
            .find(|((_, _), gl)| gl.name == "blocks.0")
            .map(|((i, _), _)| i)
            .expect("blocks.0 must be present in the graph");

        // Head 0 (score=100) must dominate head 1 (score=5) after scorer override.
        let p = &plan.weight_analysis.per_layer[blocks0_idx].head_scores;
        assert!(
            p[0] > 10.0 * p[1],
            "gradient-dominated head 0 must be at least 10x head 1; got {p:?}"
        );
    }

    /// When the scorer is `NullGradientScorer` (returns `None` for every layer),
    /// the magnitude path is used exclusively and `head_scores` remain uniform
    /// (NullWeightProvider → all-zero raw scores → uniform normalised output).
    #[test]
    fn head_importance_falls_back_to_magnitude_when_scorer_returns_none() {
        use crate::wggo_gradient_scorer::{GradientScorer, NullGradientScorer};

        let scorer: Box<dyn GradientScorer> = Box::new(NullGradientScorer);
        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        inp.scorer = Some(scorer);
        let plan = run(inp);

        assert!(!plan.weight_analysis.per_layer.is_empty());
        let p = &plan.weight_analysis.per_layer[0].head_scores;
        // Magnitude path with NullWeightProvider → uniform scores.
        assert!(
            p.iter().all(|&v| (v - p[0]).abs() < 1e-6),
            "magnitude path with null provider must yield uniform scores; got {p:?}"
        );
    }

    // -----------------------------------------------------------------------
    // Task 8: fallback-dominance regression guards
    //
    // These tests prove that gradient scores, when present, actually change
    // ILP/scoring behaviour compared to pure magnitude.  If Phase 2 ever
    // regresses to magnitude-only, the assertions will catch it because
    // NullWeightProvider produces uniform scores and the dominance check
    // would fail.
    // -----------------------------------------------------------------------

    /// When a `CalibratedGradientScorer` with a strongly skewed sidecar is
    /// wired in, the highest-ranked head must dominate (>10x) the lowest-ranked
    /// head in `weight_analysis.per_layer[*].head_scores`.  Regression to
    /// magnitude-only would yield uniform scores (NullWeightProvider) and the
    /// assertion would fail.
    #[test]
    fn calibrated_gradient_scores_dominate_magnitude_when_sidecar_present() {
        use crate::wggo_gradient_scorer::{
            CalibratedGradientScorer, GradientScorer, MagnitudeFallbackScorer,
        };
        use crate::wggo_weight_analysis::NullWeightProvider;
        use std::collections::BTreeMap;
        use std::sync::Arc;

        // Sidecar: "blocks.0" head 0 score=100, heads 1-7 score=1.
        // With NullWeightProvider the magnitude path would yield uniform [x; 8].
        let mut scores = BTreeMap::new();
        scores.insert(
            "blocks.0".to_string(),
            vec![100.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        );
        scores.insert(
            "blocks.1".to_string(),
            vec![100.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        );
        let scorer: Box<dyn GradientScorer> = Box::new(CalibratedGradientScorer::new(
            scores,
            MagnitudeFallbackScorer::new(
                Arc::new(NullWeightProvider) as Arc<dyn WeightProvider + Send + Sync>,
            ),
        ));

        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        inp.scorer = Some(scorer);
        let plan = run(inp);

        // Locate blocks.0 by name (graph builder may prepend an "other" layer).
        let blocks0_idx = plan
            .weight_analysis
            .per_layer
            .iter()
            .enumerate()
            .zip(plan.graph.layers.iter())
            .find(|((_, _), gl)| gl.name == "blocks.0")
            .map(|((i, _), _)| i)
            .expect("blocks.0 must be present in the weight_analysis report");

        let per_head = &plan.weight_analysis.per_layer[blocks0_idx].head_scores;
        // Gradient signal must dominate — head 0 must be at least 10x head 1.
        assert!(
            per_head[0] > per_head[1] * 10.0,
            "gradient signal should dominate magnitude fallback; got {per_head:?}"
        );
        // If Phase 2 regresses to magnitude-only, per_head would be uniform
        // (null provider → all equal), and the assertion above would fail.
    }

    /// When `scorer = NullGradientScorer` (no gradient signal), the run must
    /// produce uniform head scores — proving Phase 1 is preserved.  This is
    /// the symmetric guard: if Phase 2 accidentally always overrides to a
    /// non-null scorer, this would fail.
    #[test]
    fn phase1_preserved_when_no_sidecar() {
        use crate::wggo_gradient_scorer::{GradientScorer, NullGradientScorer};

        let scorer: Box<dyn GradientScorer> = Box::new(NullGradientScorer);
        let w = two_block_wengert();
        let mut inp = toy_input(&w);
        inp.scorer = Some(scorer);
        let plan = run(inp);

        assert!(!plan.weight_analysis.per_layer.is_empty());
        // NullGradientScorer → run_weight_analysis falls back to magnitude for
        // every layer → NullWeightProvider → uniform scores.
        for entry in &plan.weight_analysis.per_layer {
            let per_head = &entry.head_scores;
            if !per_head.is_empty() {
                assert!(
                    per_head.iter().all(|&v| (v - per_head[0]).abs() < 1e-6),
                    "Phase 1 path must yield uniform scores with null provider; got {per_head:?}"
                );
            }
        }
    }

    /// Verify `run_on_wengert_with_weights` wires the scorer when
    /// `compile_options` carrying a `calibration_sidecar` is passed.
    /// Checks that the `importance_source=gradient (calibrated)` log path
    /// fires by inspecting the plan's weight_analysis scores — the gradient
    /// signal must be non-uniform for the layer named in the sidecar.
    #[test]
    fn run_on_wengert_with_weights_wires_scorer_from_compile_options() {
        use crate::calibration::sidecar::{PerLayerGradient, Sidecar, WggoHeadGradients};
        use crate::{CompileOptions, WggoImportance};
        use std::collections::BTreeMap;

        // Construct a sidecar with a strongly-skewed gradient signal for
        // blocks.0 and blocks.1 (the two layers in two_block_wengert).
        let mut by_layer = BTreeMap::new();
        for name in ["blocks.0", "blocks.1"] {
            by_layer.insert(
                name.to_string(),
                PerLayerGradient {
                    per_head_score: vec![100.0f32, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                    batches_observed: 4,
                },
            );
        }
        let sidecar = Sidecar {
            version: crate::calibration::sidecar::SIDECAR_VERSION,
            checkpoint_sha256: String::new(),
            calibration_data_sha256: String::new(),
            hook_set_sha256: String::new(),
            cache_key_digest: String::new(),
            num_samples_used: 4,
            hooks: BTreeMap::new(),
            wggo_head_gradients: Some(WggoHeadGradients { by_layer }),
        };

        let mut opts = CompileOptions::default();
        opts.wggo.importance = WggoImportance::Auto;
        opts.calibration_sidecar = Some(sidecar);

        let w = two_block_wengert();
        let plan = run_on_wengert_with_weights(
            &w,
            "H100",
            "full",
            1,
            None, // no weight file needed; scorer uses sidecar
            crate::wggo_weight_analysis::AnalysisConfig::default(),
            Some(&opts),
        )
        .expect("plan");

        // The calibrated scorer should have replaced magnitude scores for
        // blocks.0 — head 0 (100.0) must dominate head 1 (1.0).
        let blocks0_idx = plan
            .weight_analysis
            .per_layer
            .iter()
            .enumerate()
            .zip(plan.graph.layers.iter())
            .find(|((_, _), gl)| gl.name == "blocks.0")
            .map(|((i, _), _)| i)
            .expect("blocks.0 must be present");

        let per_head = &plan.weight_analysis.per_layer[blocks0_idx].head_scores;
        assert!(
            per_head[0] > per_head[1] * 10.0,
            "build_scorer wiring via compile_options must produce dominant gradient scores; got {per_head:?}"
        );
    }

    #[test]
    fn pca_appears_in_report() {
        let w = two_block_wengert();
        let plan = run(toy_input(&w));
        let rep = plan.render_report();
        assert!(rep.contains("PCA="));
    }

    #[test]
    fn run_on_wengert_propagates_world_size() {
        // ClusterSpec is buried inside InterLayerPlan's source — we can't
        // read it back directly, but we can verify the plan is produced
        // without panicking for a multi-GPU cluster.
        let w = two_block_wengert();
        let plan = run_on_wengert(&w, "H100", "full", 8).expect("plan");
        assert_eq!(plan.mode, WggoMode::Full);
        assert!(!plan.inter_layer.layers.is_empty());
    }

    // -----------------------------------------------------------------------
    // CPKD distillation sidecar (v1) — driver surface.
    // -----------------------------------------------------------------------

    #[test]
    fn cpkd_gate_off_plan_and_report_are_free_of_cpkd() {
        // Production default (no explicit constraints => cpkd: None): the
        // plan sidecar is empty, the report has no "[cpkd]" section, and
        // the serialized plan skips the empty vec entirely.
        let w = two_block_wengert();
        let plan = run(toy_input(&w));
        assert!(plan.cpkd_distill.is_empty());
        let rep = plan.render_report();
        assert!(
            !rep.contains("[cpkd]"),
            "gate-off report must not contain the advisory section"
        );
        let json = serde_json::to_string(&plan).unwrap();
        assert!(
            !json.contains("cpkd_distill"),
            "gate-off plan JSON must skip the empty advisory vec"
        );
    }

    #[test]
    fn cpkd_gate_on_populates_plan_sidecar_and_report() {
        use crate::wggo_cpkd::CpkdConfig;
        let w = two_block_wengert();
        let n = build_graph(&w).layers.len();
        let mut cons = LayerIlpConstraints::default();
        cons.cpkd = Some(CpkdConfig {
            // Favorable net charge so feature matching is advised on.
            feature_cost_us: -15.0,
            ..Default::default()
        });
        let mut inp = toy_input(&w);
        inp.ilp_constraints = vec![cons; n];
        let plan = run(inp);

        assert!(
            !plan.cpkd_distill.is_empty(),
            "gate on must surface advisory decisions on the plan"
        );
        for d in &plan.cpkd_distill {
            assert!(d.choice.feature_match);
            assert!(!d.choice.attn_transfer, "deferred axis stays off");
            assert!(!d.choice.teacher_stream, "deferred axis stays off");
            assert!(!d.layer_name.is_empty());
            assert!(d.note.contains("distill_us/step/layer"));
            assert!(d.note.contains("ADVISORY"));
        }

        let rep = plan.render_report();
        assert!(rep.contains("[cpkd] Distillation decisions (advisory):"));
        assert!(rep.contains("Report-only in v1"));
        assert!(rep.contains("feature_match=on"));
        assert!(rep.contains("attn_transfer=off"));
        assert!(rep.contains("[cpkd]   Cost model: distill_us/step/layer"));

        // Serialized plan carries the advisory vec when (and only when) on.
        let json = serde_json::to_string(&plan).unwrap();
        assert!(json.contains("cpkd_distill"));
    }
}
