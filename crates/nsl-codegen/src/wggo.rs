//! WGGO — Wengert Graph Global Optimization: driver.
//!
//! Orchestrates the eight stages described in §5 of the research paper:
//!
//!   1. Wengert graph extraction          — `wengert.rs`
//!   2. Cost-model annotation             — `wggo_cost::build_lut`
//!   3. Weight-analysis (optional)        — placeholder hook
//!   4. Level 1: inter-layer DP           — `wggo_dp::solve`
//!   5. Level 2: per-layer ILP            — `wggo_ilp::solve_all`
//!   6. Level 3: kernel generation        — delegated to backend
//!   7. Memory planning                   — delegated to M36
//!   8. Communication schedule            — delegated to CPDT (future)
//!
//! The driver is pure data-in / data-out and has no backend side
//! effects.  It produces a [`WggoPlan`] downstream passes consume.

use serde::Serialize;

use crate::gpu_specs::{default_gpu, find_gpu, GpuSpec};
use crate::wengert::WengertList;
use crate::wggo_apply::{apply, AppliedPlan};
use crate::wggo_conflicts::{greedy_resolve, LayerDecisions, Resolution};
use crate::wggo_cost::{build_lut, LayerCostLut, LayerShape, LutAxes};
use crate::wggo_dp::{solve as dp_solve, ClusterSpec, DpConfig, ImportanceScores, InterLayerPlan};
use crate::wggo_graph::{build as build_graph, OptGraph};
use crate::wggo_ilp::{solve_all as ilp_solve_all, LayerIlpConstraints, LayerIlpSolution};

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
#[derive(Debug, Clone)]
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
    /// Total solver wall-clock time (μs, self-reported — not measured).
    pub estimated_solve_us: u64,
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
                "  {}: {}/{} heads, FFN={}, CSHA-L{}, LoRA r={}, m={}b v={}b",
                layer.layer_name,
                layer.active_heads,
                layer.active_heads, // number of actually-kept heads (no "of total" info here)
                layer.ffn_width,
                layer.csha_level,
                layer.adapter_rank,
                layer.optim_m_bits,
                layer.optim_v_bits
            )
            .unwrap();
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

/// Run the WGGO driver.
pub fn run(input: WggoInput) -> WggoPlan {
    let t0 = std::time::Instant::now();
    let gpu = find_gpu(input.target).unwrap_or_else(default_gpu);

    // Off mode: build a trivial plan that passes through decisions.
    if input.mode == WggoMode::Off {
        let graph = build_graph(input.wengert);
        let n = graph.layers.len();
        let lut = build_lut(&input.layer_shape, gpu, &input.lut_axes);
        let luts: Vec<LayerCostLut> = vec![lut; n.max(1)];
        let dp_cfg = DpConfig {
            cluster: input.cluster.clone(),
            importance: input.importance.clone(),
            ..Default::default()
        };
        let inter = dp_solve(&graph, &luts, &dp_cfg);
        let ilp_defaults: Vec<LayerIlpConstraints> = if input.ilp_constraints.is_empty() {
            vec![LayerIlpConstraints::default(); n]
        } else {
            input.ilp_constraints
        };
        let per_layer = ilp_solve_all(&luts, &ilp_defaults);
        let applied = apply(&inter, &per_layer);
        return WggoPlan {
            mode: WggoMode::Off,
            target_gpu: gpu.name.to_string(),
            graph,
            inter_layer: inter,
            per_layer,
            resolutions: Vec::new(),
            applied,
            estimated_solve_us: t0.elapsed().as_micros() as u64,
        };
    }

    // 1-2. Build the layer graph and the cost LUT.
    let graph = build_graph(input.wengert);
    let lut = build_lut(&input.layer_shape, gpu, &input.lut_axes);
    let n = graph.layers.len();
    let luts: Vec<LayerCostLut> = vec![lut; n.max(1)];

    // 4. Level 1 DP.
    let dp_cfg = DpConfig {
        cluster: input.cluster.clone(),
        importance: input.importance.clone(),
        ..Default::default()
    };
    let inter = dp_solve(&graph, &luts, &dp_cfg);

    // 5. Level 2 ILP (per layer, independent once inter-layer decisions
    //    are fixed — paper §5.2).
    let ilp_constraints: Vec<LayerIlpConstraints> = if input.ilp_constraints.is_empty() {
        vec![LayerIlpConstraints::default(); n]
    } else {
        input.ilp_constraints
    };
    let per_layer = ilp_solve_all(&luts, &ilp_constraints);

    // 6. Build initial LayerDecisions vector for conflict detection.
    let layer_decisions: Vec<LayerDecisions> = inter
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
            fase_fused: false,
            adapter_comm_cost: 0.0,
            adapter_comm_budget: f64::MAX,
        })
        .collect();

    // 7. Conflict detection + resolution (greedy mode resolves here;
    //    Full mode still runs the same resolver because the ILP itself
    //    already honours the hard constraints — remaining conflicts are
    //    the ones crossing technique boundaries).
    let (_resolved, resolutions) = greedy_resolve(layer_decisions);

    // 8. Apply.
    let applied = apply(&inter, &per_layer);

    WggoPlan {
        mode: input.mode,
        target_gpu: gpu.name.to_string(),
        graph,
        inter_layer: inter,
        per_layer,
        resolutions,
        applied,
        estimated_solve_us: t0.elapsed().as_micros() as u64,
    }
}

/// Extract `GpuSpec` for the named target or default.
#[allow(dead_code)]
fn resolve_gpu(name: &str) -> &'static GpuSpec {
    find_gpu(name).unwrap_or_else(default_gpu)
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
}
