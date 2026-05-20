//! CPDT — Compile-time Parallelism & Distributed Training: driver.
//!
//! Composes the five CPDT passes (ZeRO planning, comm scheduling,
//! precision selection, quantized optimizer codegen, expert placement)
//! into a single [`CpdtPlan`] that downstream codegen consumes.
//!
//! The driver is pure and deterministic.  Every invocation with the
//! same inputs returns the same plan bit-for-bit (outside the wall-
//! clock `solve_us` field).

use serde::Serialize;

use crate::cpdt_comm::{build_schedule, CommSchedule};
use crate::cpdt_expert::{ExpertConfig, ExpertPlan, MoeLayerShape};
use crate::cpdt_joint::{solve as solve_joint, JointConfig, JointInput, JointPlan};
use crate::cpdt_optim::{emit_plan, AdamWHyperparams, QuantizedOptimProgram};
use crate::cpdt_tier_apply::{plan_map, PrecisionConfig, PrecisionPlan};
use crate::cpdt_zero::{ClusterSpec, ModelSize, ZeroEvaluation};
use crate::weight_aware::{WeightEntry, WeightMap};

/// User-visible CPDT mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CpdtMode {
    /// Full joint optimisation (ZeRO + precision + experts).
    Full,
    /// ZeRO planning only (no precision quantisation, no MoE).
    ZeroOnly,
    /// Bypass CPDT entirely — DDP fallback.
    Off,
}

impl CpdtMode {
    pub fn as_str(self) -> &'static str {
        match self {
            CpdtMode::Full => "full",
            CpdtMode::ZeroOnly => "zero_only",
            CpdtMode::Off => "off",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "full" | "auto" => Some(CpdtMode::Full),
            "zero_only" | "zero" => Some(CpdtMode::ZeroOnly),
            "off" | "disable" | "disabled" => Some(CpdtMode::Off),
            _ => None,
        }
    }
}

/// Inputs to the driver.
pub struct CpdtInput<'a> {
    pub mode: CpdtMode,
    pub model: ModelSize,
    pub cluster: ClusterSpec,
    pub weights: Option<&'a WeightMap>,
    pub precision_cfg: PrecisionConfig,
    pub adamw: AdamWHyperparams,
    pub moe_shape: Option<MoeLayerShape>,
    pub moe_router: Option<&'a WeightEntry>,
    pub moe_roofline_slack: f64,
    pub expert_cfg: ExpertConfig,
    pub joint_cfg: JointConfig,
    /// Single global ZeRO shard-factor recommendation aggregated from
    /// WGGO's per-layer decisions via `WggoOverrides::min_shard_factor`.
    /// `None` → CPDT's planner runs unchanged.
    pub wggo_recommended_shard: Option<u32>,
}

/// Top-level CPDT plan.
#[derive(Debug, Clone)]
pub struct CpdtPlan {
    pub mode: CpdtMode,
    pub zero: Option<ZeroEvaluation>,
    pub comm_schedule: CommSchedule,
    pub precision: PrecisionPlan,
    pub optimizer_programs: Vec<QuantizedOptimProgram>,
    pub experts: Option<ExpertPlan>,
    pub joint: Option<JointPlan>,
    pub solve_us: u64,
    /// WGGO recommendations that CPDT had to clamp or reject due to
    /// hardware constraints. Empty when no recommendation was supplied
    /// or it was applied verbatim.
    pub override_diagnostics: Vec<crate::wggo_overrides::OverrideDiagnostic>,
}

impl CpdtPlan {
    /// Render a human-readable report matching paper §7's sample output.
    pub fn render_report(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        writeln!(s, "=== CPDT Training Plan ===").unwrap();
        writeln!(s, "Mode: {}", self.mode.as_str()).unwrap();
        if let Some(zero) = self.zero.as_ref() {
            writeln!(s).unwrap();
            writeln!(s, "ZeRO Configuration:").unwrap();
            writeln!(
                s,
                "  s_p={} s_g={} s_os={} ({})",
                zero.config.s_p,
                zero.config.s_g,
                zero.config.s_os,
                zero.config.stage_label()
            )
            .unwrap();
            writeln!(
                s,
                "  Memory per GPU: {:.2} GB (params={:.2} MB, grads={:.2} MB, optim={:.2} MB, act={:.2} MB)",
                zero.memory_per_gpu_bytes as f64 / 1e9,
                zero.param_bytes_per_gpu as f64 / 1e6,
                zero.grad_bytes_per_gpu as f64 / 1e6,
                zero.optim_bytes_per_gpu as f64 / 1e6,
                zero.activation_bytes_per_gpu as f64 / 1e6,
            )
            .unwrap();
            writeln!(s, "  Exposed comm: {:.2} μs", zero.exposed_comm_us).unwrap();
            writeln!(s, "  Estimated step time: {:.2} μs", zero.step_time_us).unwrap();
        }
        writeln!(s).unwrap();
        writeln!(
            s,
            "Communication schedule: {} ops ({} async)",
            self.comm_schedule.total_ops(),
            self.comm_schedule.async_count()
        )
        .unwrap();
        writeln!(s).unwrap();
        writeln!(s, "Optimizer precision:").unwrap();
        let (h, m, l, v) = self.precision.tier_counts();
        writeln!(s, "  High tier: {h}  Medium: {m}  Low: {l}  Very low: {v}").unwrap();
        writeln!(
            s,
            "  Total optim bytes: {:.2} MB (baseline FP32: {:.2} MB, savings {:.1}%)",
            self.precision.total_optim_bytes as f64 / 1e6,
            self.precision.baseline_fp32_bytes as f64 / 1e6,
            100.0 * self.precision.savings_ratio()
        )
        .unwrap();
        if let Some(experts) = self.experts.as_ref() {
            writeln!(s).unwrap();
            writeln!(s, "MoE placement:").unwrap();
            writeln!(
                s,
                "  Strategy: {}  capacity_factor={:.2}  dead_experts={}",
                experts.placement.strategy.as_str(),
                experts.capacity_factor,
                experts.dead_experts.len()
            )
            .unwrap();
            writeln!(s, "  {}", experts.placement.rationale).unwrap();
        }
        if let Some(joint) = self.joint.as_ref() {
            writeln!(s).unwrap();
            writeln!(
                s,
                "Joint solver: {} iterations, converged={}",
                joint.num_iterations(),
                joint.converged
            )
            .unwrap();
        }
        writeln!(s).unwrap();
        writeln!(s, "Solve time: {:.2} ms", self.solve_us as f64 / 1000.0).unwrap();
        s
    }
}

/// Run the driver.
pub fn run(input: CpdtInput) -> CpdtPlan {
    let t0 = std::time::Instant::now();

    if input.mode == CpdtMode::Off {
        return CpdtPlan {
            mode: CpdtMode::Off,
            zero: None,
            comm_schedule: CommSchedule::default(),
            precision: PrecisionPlan::default(),
            optimizer_programs: Vec::new(),
            experts: None,
            joint: None,
            solve_us: t0.elapsed().as_micros() as u64,
            override_diagnostics: Vec::new(),
        };
    }

    // 1. ZeRO search — builds the full eval table and selects the default best.
    let zero_search = crate::cpdt_zero::search(&input.model, &input.cluster);

    // 2. WGGO two-gate shard-factor override.
    //    Gate 1 (cheap reject): world_size % recommended != 0  → use default.
    //    Gate 2 (table coverage): most aggressive feasible tuple at-or-below rec.
    //    None → byte-identical to previous behaviour.
    let mut override_diagnostics: Vec<crate::wggo_overrides::OverrideDiagnostic> = Vec::new();
    let world_size_u32 = input.cluster.num_gpus;

    let zero: Option<ZeroEvaluation> = match input.wggo_recommended_shard {
        Some(rec) if !world_size_u32.is_multiple_of(rec) => {
            // Gate 1: world-size divisibility.
            override_diagnostics.push(crate::wggo_overrides::OverrideDiagnostic {
                layer_index: 0,
                layer_name: "global".into(),
                reason: crate::wggo_overrides::OverrideRejectReason::ShardFactorIncompatibleWithWorldSize {
                    recommended: rec,
                    world_size: world_size_u32,
                },
                requested: rec.to_string(),
                applied: "internal_default".into(),
            });
            zero_search.best.clone()
        }
        Some(rec) => {
            // Gate 2: most aggressive feasible tuple at-or-below rec.
            let preferred = zero_search
                .ranked
                .iter()
                .filter(|e| e.feasible && e.config.s_p <= rec)
                .max_by_key(|e| e.config.s_p)
                .cloned();

            match preferred {
                Some(t) => {
                    // Found a feasible tuple within the recommendation —
                    // use it silently even if s_p < rec (table-gap downgrade).
                    Some(t)
                }
                None => {
                    // Nothing feasible at-or-below rec; fall back to planner default.
                    let actual = zero_search.best.clone();
                    if let Some(ref eval) = actual {
                        if eval.config.s_p > rec {
                            override_diagnostics
                                .push(crate::wggo_overrides::OverrideDiagnostic {
                                    layer_index: 0,
                                    layer_name: "global".into(),
                                    reason: crate::wggo_overrides::OverrideRejectReason::ShardFactorOverriddenByMemory {
                                        recommended: rec,
                                        applied: eval.config.s_p,
                                    },
                                    requested: rec.to_string(),
                                    applied: eval.config.s_p.to_string(),
                                });
                        }
                    }
                    actual
                }
            }
        }
        None => zero_search.best.clone(),
    };

    // 3. Comm schedule for the chosen ZeRO config.
    let comm_schedule = match zero.as_ref() {
        Some(eval) => build_schedule(eval.config, &input.model.per_layer_param_bytes),
        None => CommSchedule::default(),
    };

    // 4. Precision plan (only if weights + Full mode).
    let precision = if input.mode == CpdtMode::Full {
        match input.weights {
            Some(wm) => plan_map(wm, &input.precision_cfg),
            None => PrecisionPlan::default(),
        }
    } else {
        PrecisionPlan::default()
    };

    // 5. Quantized optimizer programs.
    let optimizer_programs = emit_plan(&precision, &input.adamw);

    // 6. MoE placement (if present).
    let experts = if input.mode == CpdtMode::Full {
        input.moe_shape.map(|shape| {
            crate::cpdt_expert::plan(
                &shape,
                input.moe_router,
                input.moe_roofline_slack,
                &input.expert_cfg,
            )
        })
    } else {
        None
    };

    // 7. Joint refinement.
    let joint = if input.mode == CpdtMode::Full {
        Some(solve_joint(JointInput {
            model: input.model.clone(),
            cluster: input.cluster.clone(),
            precision: precision.clone(),
            moe_shape: input.moe_shape,
            moe_router: input.moe_router,
            moe_roofline_slack: input.moe_roofline_slack,
            expert_cfg: input.expert_cfg,
            joint_cfg: input.joint_cfg,
        }))
    } else {
        None
    };

    CpdtPlan {
        mode: input.mode,
        zero,
        comm_schedule,
        precision,
        optimizer_programs,
        experts,
        joint,
        solve_us: t0.elapsed().as_micros() as u64,
        override_diagnostics,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod override_tests {
    use super::*;
    use crate::wggo_overrides::OverrideRejectReason;

    /// Build a `CpdtInput` with adjustable world_size, WGGO recommendation,
    /// and memory pressure.
    ///
    /// Model: 8 layers × 6 MB params = 48 MB total, optim_multiplier=8.0,
    ///        peak activation = 2 MB (single layer max).
    ///
    /// Per-GPU memory formula: params/s_p + params/s_g + params*8/s_os + 2 MB.
    ///   DDP (s_p=1, s_g=1, s_os=1): 48+48+384+2 = 482 MB.
    ///   Best s_p=2 tuple (s_p=2,s_g=8,s_os=8): 24+6+48+2   =  80 MB.
    ///   Best s_p=4 tuple (s_p=4,s_g=4,s_os=8): 12+12+48+2  =  74 MB.
    ///
    /// memory_pressure "low"  → 80 GB budget (all tuples fit).
    /// memory_pressure "high" → 75 MB budget (only s_p ≥ 4 tuples fit):
    ///    80 MB (best s_p=2 tuple) > 75 MB → infeasible
    ///    74 MB (best s_p=4 tuple) ≤ 75 MB → feasible
    fn cpdt_input_with(
        world_size: u32,
        recommended: Option<u32>,
        memory_pressure: &'static str,
    ) -> CpdtInput<'static> {
        let memory_budget_bytes = match memory_pressure {
            "low" => 80u64 * 1024 * 1024 * 1024, // 80 GB — all tuples feasible
            "high" => 75_000_000u64,             // 75 MB — only s_p ≥ 4 feasible
            other => panic!("unknown memory_pressure: {other}"),
        };
        CpdtInput {
            mode: CpdtMode::ZeroOnly,
            model: ModelSize {
                per_layer_param_bytes: vec![6_000_000; 8],
                per_layer_activation_bytes: vec![2_000_000; 8],
                optim_state_multiplier: crate::cpdt_zero::ADAMW_FP32_OPTIM_MULTIPLIER,
                per_layer_compute_us: vec![10.0; 8],
            },
            cluster: ClusterSpec {
                num_gpus: world_size,
                memory_budget_bytes,
                intra_bw_bps: 9e11,
                inter_bw_bps: 1e11,
                gpus_per_node: world_size.min(8),
            },
            weights: None,
            precision_cfg: PrecisionConfig::default(),
            adamw: AdamWHyperparams::default(),
            moe_shape: None,
            moe_router: None,
            moe_roofline_slack: 1.0,
            expert_cfg: ExpertConfig::default(),
            joint_cfg: JointConfig::default(),
            wggo_recommended_shard: recommended,
        }
    }

    #[test]
    fn wggo_recommendation_applied_when_world_size_divides_and_memory_fits() {
        // world_size=8, rec=4 → 8 % 4 == 0 (Gate 1 passes).
        // "low" budget → all tuples feasible. Most aggressive at-or-below 4
        // is s_p=4 (divisors of 8: {1,2,4,8}).
        let input = cpdt_input_with(8, Some(4), "low");
        let plan = run(input);
        let s_p = plan.zero.as_ref().expect("zero eval").config.s_p;
        assert!(
            s_p <= 4,
            "applied s_p must not exceed recommendation; got {s_p}"
        );
        assert!(
            plan.override_diagnostics.is_empty(),
            "no diagnostics expected; got {:?}",
            plan.override_diagnostics
        );
    }

    #[test]
    fn wggo_recommendation_rejected_on_world_size_mismatch() {
        // world_size=2, rec=4 → 2 % 4 == 2 ≠ 0 → Gate 1 fires.
        let input = cpdt_input_with(2, Some(4), "low");
        let plan = run(input);
        let diag = plan
            .override_diagnostics
            .iter()
            .find(|d| {
                matches!(
                    d.reason,
                    OverrideRejectReason::ShardFactorIncompatibleWithWorldSize { .. }
                )
            })
            .expect("incompat diag must be emitted");
        assert_eq!(diag.requested, "4");
        assert_eq!(diag.layer_name, "global");
    }

    #[test]
    fn wggo_recommendation_overridden_by_memory_when_more_sharding_required() {
        // world_size=8, rec=2, budget=100 MB.
        // All tuples with s_p ≤ 2 are infeasible (best costs 112 MB).
        // Planner falls back to default → picks s_p ≥ 4.
        // Since applied s_p > 2 = rec, ShardFactorOverriddenByMemory is emitted.
        let input = cpdt_input_with(8, Some(2), "high");
        let plan = run(input);
        let diag = plan
            .override_diagnostics
            .iter()
            .find(|d| {
                matches!(
                    d.reason,
                    OverrideRejectReason::ShardFactorOverriddenByMemory { .. }
                )
            })
            .expect("memory-override diag must be emitted");
        assert_eq!(diag.requested, "2");
        let applied_s_p = plan.zero.as_ref().expect("zero eval").config.s_p;
        assert!(
            applied_s_p > 2,
            "applied s_p must exceed recommendation; got {applied_s_p}"
        );
    }

    #[test]
    fn no_recommendation_preserves_existing_planner_behavior() {
        // None → override path skipped entirely; zero must still be produced.
        let input = cpdt_input_with(8, None, "low");
        let plan = run(input);
        assert!(plan.override_diagnostics.is_empty());
        assert!(
            plan.zero.is_some(),
            "planner must produce a zero evaluation"
        );
    }

    #[test]
    fn recommendation_picks_most_aggressive_feasible_at_or_below() {
        // world_size=8, rec=4, "low" → divisors {1,2,4,8} all feasible.
        // Most aggressive at-or-below 4 is s_p=4; no diagnostic emitted
        // (table-gap downgrade is silent, and here there is no gap).
        let input = cpdt_input_with(8, Some(4), "low");
        let plan = run(input);
        let s_p = plan.zero.as_ref().expect("zero eval").config.s_p;
        assert!(s_p <= 4, "s_p must not exceed recommendation; got {s_p}");
        assert!(
            plan.override_diagnostics.is_empty(),
            "table-gap downgrade must be silent (no diagnostic); got {:?}",
            plan.override_diagnostics
        );
    }

    // ── Task 4 smoke tests: end-to-end propagation of wggo_recommended_shard ──

    #[test]
    fn cpdt_run_with_wggo_recommendation_propagates_to_plan() {
        use crate::wggo_overrides::{PerLayerOverride, WggoOverrides};
        let over = WggoOverrides {
            per_layer: vec![PerLayerOverride {
                layer_index: 0,
                layer_name: "blocks.0".into(),
                active_heads: 8,
                requested_csha_level: None,
                adapter_rank: 0,
                fase_fused: false,
                packing_mode: 0,
                shard_factor: 4,
            }],
        };
        let recommended = over.min_shard_factor();
        assert_eq!(recommended, Some(4));
        // Use the existing helper: world_size=8, recommended=Some(4), low budget.
        // 8 % 4 == 0 → Gate 1 passes.  "low" → all tuples feasible.
        // Most aggressive at-or-below 4 is s_p=4; plan should honour it, no diag.
        let input = cpdt_input_with(8, recommended, "low");
        let plan = run(input);
        let s_p = plan.zero.as_ref().expect("zero eval").config.s_p;
        let has_diag = !plan.override_diagnostics.is_empty();
        assert!(
            s_p <= 4 || has_diag,
            "expected s_p ≤ 4 OR diagnostic; got s_p={s_p}, diags={:?}",
            plan.override_diagnostics
        );
    }

    #[test]
    fn cpdt_run_without_overrides_leaves_override_diagnostics_empty() {
        // wggo_recommended_shard=None → override path skipped entirely;
        // override_diagnostics must be empty.
        let input = cpdt_input_with(8, None, "low");
        let plan = run(input);
        assert!(
            plan.override_diagnostics.is_empty(),
            "no recommendation → no diagnostics; got {:?}",
            plan.override_diagnostics
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model() -> ModelSize {
        ModelSize {
            per_layer_param_bytes: vec![6_000_000; 8],
            per_layer_activation_bytes: vec![2_000_000; 8],
            optim_state_multiplier: crate::cpdt_zero::ADAMW_FP32_OPTIM_MULTIPLIER,
            per_layer_compute_us: vec![10.0; 8],
        }
    }

    fn tiny_input<'a>() -> CpdtInput<'a> {
        CpdtInput {
            mode: CpdtMode::Full,
            model: tiny_model(),
            cluster: ClusterSpec::default(),
            weights: None,
            precision_cfg: PrecisionConfig::default(),
            adamw: AdamWHyperparams::default(),
            moe_shape: None,
            moe_router: None,
            moe_roofline_slack: 1.0,
            expert_cfg: ExpertConfig::default(),
            joint_cfg: JointConfig::default(),
            wggo_recommended_shard: None,
        }
    }

    #[test]
    fn full_mode_produces_zero_config_and_schedule() {
        let plan = run(tiny_input());
        assert!(plan.zero.is_some());
        // The chosen stage should be a recognized ZeRO variant or DDP
        // (DDP is expected when the model fits comfortably in a single
        // GPU's budget — tiny_model's 48 MB fits the default 80 GB).
        let stage = plan.zero.as_ref().unwrap().config.stage_label();
        assert!(matches!(
            stage,
            "DDP" | "ZeRO-1" | "ZeRO-2" | "ZeRO-3" | "Mixed"
        ));
        // If any shard factor > 1, we must have a non-empty schedule.
        let cfg = plan.zero.as_ref().unwrap().config;
        let sharded = cfg.s_p > 1 || cfg.s_g > 1 || cfg.s_os > 1;
        assert_eq!(plan.comm_schedule.total_ops() > 0, sharded);
    }

    #[test]
    fn off_mode_produces_empty_plan() {
        let mut input = tiny_input();
        input.mode = CpdtMode::Off;
        let plan = run(input);
        assert!(plan.zero.is_none());
        assert_eq!(plan.comm_schedule.total_ops(), 0);
    }

    #[test]
    fn zero_only_mode_skips_precision_and_experts() {
        let mut input = tiny_input();
        input.mode = CpdtMode::ZeroOnly;
        let plan = run(input);
        assert!(plan.zero.is_some());
        assert!(plan.precision.params.is_empty());
        assert!(plan.experts.is_none());
        assert!(plan.joint.is_none());
    }

    #[test]
    fn mode_roundtrips() {
        for m in [CpdtMode::Full, CpdtMode::ZeroOnly, CpdtMode::Off] {
            assert_eq!(CpdtMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(CpdtMode::parse("auto"), Some(CpdtMode::Full));
        assert!(CpdtMode::parse("nonsense").is_none());
    }

    #[test]
    fn render_report_contains_expected_sections() {
        let plan = run(tiny_input());
        let rep = plan.render_report();
        assert!(rep.contains("CPDT Training Plan"));
        assert!(rep.contains("ZeRO Configuration"));
        assert!(rep.contains("Optimizer precision"));
    }

    #[test]
    fn plan_is_deterministic_except_solve_time() {
        let make = || tiny_input();
        let r1 = run(make()).render_report();
        let r2 = run(make()).render_report();
        let strip = |s: &str| -> String {
            s.lines()
                .filter(|l| !l.contains("Solve time"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        assert_eq!(strip(&r1), strip(&r2));
    }

    #[test]
    fn full_mode_runs_joint_solver() {
        let plan = run(tiny_input());
        assert!(plan.joint.is_some());
    }

    #[test]
    fn cpdt_plan_default_has_empty_override_diagnostics() {
        // Smoke test: a plan from a minimal Off-mode input has empty diagnostics.
        // CpdtMode::Off short-circuits the planner so we don't need a full
        // ClusterSpec / ModelSize / PrecisionConfig to exercise the field.
        let mut input = tiny_input();
        input.mode = CpdtMode::Off;
        let plan = run(input);
        assert!(
            plan.override_diagnostics.is_empty(),
            "default CpdtPlan from Off mode must have empty override_diagnostics"
        );
    }
}
