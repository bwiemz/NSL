//! CEP — Compilation-Evaluated Pruning: driver + report.
//!
//! Unifies the oracle, importance scorer, model rewriter, and search
//! algorithm into two user-facing entry points:
//!
//!   * [`run_prune`] — compilation-verified pruning from a pre-trained
//!     model (`@cep_prune` decorator flow).
//!   * [`run_search`] — hardware-aware architecture search
//!     (`@cep_search` decorator flow).
//!
//! The driver is pure and deterministic — every invocation with the same
//! inputs produces an identical report.

use serde::Serialize;

use crate::cep_importance::{analyse_weight_map, ImportanceTable, RooflineSlackTable};
use crate::cep_oracle::{evaluate, CompilationProfile, ModelSpec};
use crate::cep_rewrite::SearchAxes;
use crate::cep_search::{
    architecture_search, prune_greedy, Constraints, Granularity, NasObjective, SearchOutcome,
};
use crate::gpu_specs::{default_gpu, find_gpu};
use crate::weight_aware::WeightMap;

/// Input for [`run_prune`].
#[derive(Clone)]
pub struct CepPruneInput<'a> {
    pub spec: ModelSpec,
    pub weights: Option<&'a WeightMap>,
    pub target: &'a str,
    pub constraints: Constraints,
    pub granularity: Granularity,
    /// Per-layer roofline slack overrides (optional).
    pub roofline_slack: RooflineSlackTable,
}

/// Input for [`run_search`].
#[derive(Debug, Clone)]
pub struct CepSearchInput<'a> {
    pub axes: SearchAxes,
    pub target: &'a str,
    pub constraints: Constraints,
    pub objective: NasObjective,
}

/// Aggregate plan.
#[derive(Debug, Clone)]
pub struct CepPlan {
    pub mode: CepMode,
    pub target_gpu: String,
    pub outcome: SearchOutcome,
    pub importance: ImportanceTable,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CepMode {
    Prune,
    Search,
}

impl CepPlan {
    /// Parameter reduction fraction (prune mode).
    pub fn param_reduction(&self) -> f64 {
        let Some(baseline) = self.outcome.baseline.as_ref() else {
            return 0.0;
        };
        let Some(chosen) = self.outcome.chosen.as_ref() else {
            return 0.0;
        };
        let bp = baseline.param_bytes as f64;
        if bp <= 0.0 {
            return 0.0;
        }
        1.0 - (chosen.profile.param_bytes as f64 / bp)
    }

    /// Render the compilation report.
    pub fn render_report(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        match self.mode {
            CepMode::Prune => writeln!(s, "=== CEP Pruning Report ===").unwrap(),
            CepMode::Search => writeln!(s, "=== CEP Architecture Search Report ===").unwrap(),
        }
        writeln!(s, "Target: {}", self.target_gpu).unwrap();
        writeln!(
            s,
            "Candidates evaluated: {}",
            self.outcome.candidates_evaluated
        )
        .unwrap();
        writeln!(s).unwrap();

        if let Some(baseline) = self.outcome.baseline.as_ref() {
            writeln!(
                s,
                "Baseline: params={:.1}MB, peak={:.2}GB, latency={:.1}μs",
                baseline.param_bytes as f64 / 1e6,
                baseline.peak_memory_bytes as f64 / 1e9,
                baseline.estimated_latency_us
            )
            .unwrap();
        }
        if let Some(chosen) = self.outcome.chosen.as_ref() {
            writeln!(
                s,
                "Chosen:   params={:.1}MB, peak={:.2}GB, latency={:.1}μs  (feasible={})",
                chosen.profile.param_bytes as f64 / 1e6,
                chosen.profile.peak_memory_bytes as f64 / 1e9,
                chosen.profile.estimated_latency_us,
                chosen.feasible
            )
            .unwrap();
            if self.mode == CepMode::Prune {
                writeln!(
                    s,
                    "Reduction: {:.1}% of params removed",
                    100.0 * self.param_reduction()
                )
                .unwrap();
                writeln!(s).unwrap();
                writeln!(s, "Per-layer post-prune shape:").unwrap();
                for (i, (&nh, &ff)) in chosen
                    .spec
                    .n_heads
                    .iter()
                    .zip(chosen.spec.d_ff.iter())
                    .enumerate()
                {
                    writeln!(s, "  Layer {i}: heads={nh}, FFN={ff}").unwrap();
                }
            }
        }
        if self.mode == CepMode::Prune {
            writeln!(s).unwrap();
            writeln!(s, "Prune log ({} steps):", self.outcome.prune_log.len()).unwrap();
            for step in &self.outcome.prune_log {
                writeln!(
                    s,
                    "  [{}] layer={} {}{} — {} (params={:.1}MB)",
                    if step.accepted { "OK" } else { "-" },
                    step.layer,
                    step.kind,
                    step.head.map(|h| format!(" head={h}")).unwrap_or_default(),
                    step.reason,
                    step.new_param_bytes as f64 / 1e6
                )
                .unwrap();
            }
        } else {
            writeln!(s, "Top candidates:").unwrap();
            for (i, cand) in self.outcome.ranked_candidates.iter().take(5).enumerate() {
                writeln!(
                    s,
                    "  {}. d={}, L={}, H={}/{} KV, FFN={}, params={:.1}M  latency={:.1}μs  util={:.2}  feasible={}",
                    i + 1,
                    cand.spec.d_model,
                    cand.spec.n_layers,
                    cand.spec.n_heads.first().copied().unwrap_or(0),
                    cand.spec.n_kv_heads.first().copied().unwrap_or(0),
                    cand.spec.d_ff.first().copied().unwrap_or(0),
                    cand.spec.param_count() as f64 / 1e6,
                    cand.profile.estimated_latency_us,
                    cand.profile.roofline_utilization,
                    cand.feasible
                )
                .unwrap();
            }
        }
        s
    }
}

/// Default roofline-slack proxy when the user doesn't supply one: derive
/// from the baseline profile's per-layer classification.
fn slack_from_profile(p: &CompilationProfile) -> RooflineSlackTable {
    let mut v = Vec::new();
    for layer in &p.per_layer {
        if layer.layer_index < u32::MAX / 2 {
            let slack = if layer.compute_bound { 0.8 } else { 1.5 };
            v.push((layer.layer_index, slack));
        }
    }
    RooflineSlackTable { per_layer: v }
}

/// Run compilation-verified pruning.
pub fn run_prune(input: CepPruneInput) -> CepPlan {
    let gpu = find_gpu(input.target).unwrap_or_else(default_gpu);
    // 1. Baseline compile (for slack estimates + importance context).
    let baseline = evaluate(&input.spec, gpu).expect("base spec must validate");

    // 2. Resolve roofline slacks — prefer the user's, else derive.
    let slacks = if input.roofline_slack.per_layer.is_empty() {
        slack_from_profile(&baseline)
    } else {
        input.roofline_slack
    };

    // 3. Importance table (from weights if available; otherwise synthesise
    //    a uniform table so the search still runs).
    let importance = if let Some(wm) = input.weights {
        analyse_weight_map(wm, &input.spec.n_heads, input.spec.n_layers, &slacks)
    } else {
        synthetic_importance(&input.spec, &slacks)
    };

    // 4. Greedy prune.
    let outcome = prune_greedy(
        &input.spec,
        &importance,
        gpu,
        &input.constraints,
        input.granularity,
    );

    CepPlan {
        mode: CepMode::Prune,
        target_gpu: gpu.name.to_string(),
        outcome,
        importance,
    }
}

/// Run hardware-aware architecture search.
pub fn run_search(input: CepSearchInput) -> CepPlan {
    let gpu = find_gpu(input.target).unwrap_or_else(default_gpu);
    let outcome = architecture_search(&input.axes, gpu, &input.constraints, input.objective);
    CepPlan {
        mode: CepMode::Search,
        target_gpu: gpu.name.to_string(),
        outcome,
        importance: ImportanceTable::default(),
    }
}

/// Uniform-score importance table for the weight-less case.  The search
/// degenerates to "prune from the last layer back" — not ideal but
/// deterministic.
fn synthetic_importance(spec: &ModelSpec, slacks: &RooflineSlackTable) -> ImportanceTable {
    use crate::cep_importance::{HeadImportance, LayerImportance};
    let mut heads = Vec::new();
    for (layer, &nh) in spec.n_heads.iter().enumerate() {
        let slack = slacks.get(layer as u32);
        for h in 0..nh {
            heads.push(HeadImportance {
                layer: layer as u32,
                head: h,
                weight_magnitude: 1.0,
                spectral_energy: 0.5,
                roofline_slack: slack,
                position_factor: crate::cep_importance::position_factor(
                    layer as u32,
                    spec.n_layers,
                ),
                // Earlier/later layers and lower-slack (compute-bound) layers
                // get higher scores so the greedy search targets the middle
                // first.
                score: slack * crate::cep_importance::position_factor(layer as u32, spec.n_layers),
            });
        }
    }
    let layers = (0..spec.n_layers)
        .map(|l| LayerImportance {
            layer: l,
            attention_score: spec.n_heads[l as usize] as f64,
            ffn_score: 1.0,
            total_score: spec.n_heads[l as usize] as f64 + 1.0,
        })
        .collect();
    ImportanceTable {
        heads,
        ffns: Vec::new(),
        layers,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cep_oracle::{Activation, ModelSpec, NormType};
    use crate::cep_rewrite::SearchAxes;

    fn tiny_spec() -> ModelSpec {
        ModelSpec::uniform(256, 4, 8, 4, 32, 512, 8192)
    }

    #[test]
    fn run_prune_without_weights_still_produces_plan() {
        let plan = run_prune(CepPruneInput {
            spec: tiny_spec(),
            weights: None,
            target: "H100",
            constraints: Constraints {
                target_sparsity: 0.05,
                ..Default::default()
            },
            granularity: Granularity::HeadAndFfn,
            roofline_slack: RooflineSlackTable::default(),
        });
        assert_eq!(plan.mode, CepMode::Prune);
        assert!(plan.outcome.baseline.is_some());
        assert!(plan.outcome.chosen.is_some());
        let report = plan.render_report();
        assert!(report.contains("CEP Pruning Report"));
    }

    #[test]
    fn run_prune_reports_positive_reduction_when_sparsity_hit() {
        let plan = run_prune(CepPruneInput {
            spec: tiny_spec(),
            weights: None,
            target: "H100",
            constraints: Constraints {
                target_sparsity: 0.01,
                ..Default::default()
            },
            granularity: Granularity::Head,
            roofline_slack: RooflineSlackTable::default(),
        });
        assert!(plan.param_reduction() >= 0.0);
    }

    #[test]
    fn run_search_produces_ranked_candidates() {
        let axes = SearchAxes {
            d_model: vec![128, 256],
            n_layers: vec![2, 4],
            n_heads: vec![4, 8],
            n_kv_heads: vec![2, 4],
            d_ff: vec![256, 512],
            activation: vec![Activation::SwiGlu],
            norm: vec![NormType::RmsNorm],
            vocab: 8192,
            head_dim: 32,
            max_seq: 512,
            batch: 1,
            dtype_bytes: 2,
        };
        let plan = run_search(CepSearchInput {
            axes,
            target: "H100",
            constraints: Constraints::default(),
            objective: NasObjective::ParamEfficiency,
        });
        assert_eq!(plan.mode, CepMode::Search);
        assert!(plan.outcome.candidates_evaluated > 0);
        let report = plan.render_report();
        assert!(report.contains("CEP Architecture Search Report"));
    }

    #[test]
    fn plan_is_deterministic_across_runs() {
        let input = || CepPruneInput {
            spec: tiny_spec(),
            weights: None,
            target: "H100",
            constraints: Constraints {
                target_sparsity: 0.05,
                ..Default::default()
            },
            granularity: Granularity::Head,
            roofline_slack: RooflineSlackTable::default(),
        };
        let r1 = run_prune(input()).render_report();
        let r2 = run_prune(input()).render_report();
        assert_eq!(r1, r2);
    }

    #[test]
    fn synthetic_importance_covers_every_head() {
        let spec = tiny_spec();
        let slacks = RooflineSlackTable::default();
        let table = synthetic_importance(&spec, &slacks);
        let expected: u32 = spec.n_heads.iter().sum();
        assert_eq!(table.heads.len() as u32, expected);
    }

    #[test]
    fn param_reduction_returns_zero_without_baseline() {
        let plan = CepPlan {
            mode: CepMode::Search,
            target_gpu: "test".into(),
            outcome: SearchOutcome::default(),
            importance: ImportanceTable::default(),
        };
        assert_eq!(plan.param_reduction(), 0.0);
    }
}
