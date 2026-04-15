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
use crate::cpdt_precision::{plan_map, PrecisionConfig, PrecisionPlan};
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

    // 1. ZeRO search.
    let zero_search = crate::cpdt_zero::search(&input.model, &input.cluster);
    let zero = zero_search.best.clone();

    // 2. Comm schedule for the chosen ZeRO config.
    let comm_schedule = match zero.as_ref() {
        Some(eval) => build_schedule(eval.config, &input.model.per_layer_param_bytes),
        None => CommSchedule::default(),
    };

    // 3. Precision plan (only if weights + Full mode).
    let precision = if input.mode == CpdtMode::Full {
        match input.weights {
            Some(wm) => plan_map(wm, &input.precision_cfg),
            None => PrecisionPlan::default(),
        }
    } else {
        PrecisionPlan::default()
    };

    // 4. Quantized optimizer programs.
    let optimizer_programs = emit_plan(&precision, &input.adamw);

    // 5. MoE placement (if present).
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

    // 6. Joint refinement.
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
        override_diagnostics: Vec::new(),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_model() -> ModelSize {
        ModelSize {
            per_layer_param_bytes: vec![6_000_000; 8],
            per_layer_activation_bytes: vec![2_000_000; 8],
            optim_state_multiplier: 8.0,
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
        assert!(matches!(stage, "DDP" | "ZeRO-1" | "ZeRO-2" | "ZeRO-3" | "Mixed"));
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
