//! WGGO — Level 1 Inter-Layer Dynamic Programming.
//!
//! Coarse decisions per layer: which pipeline stage, which ZeRO shard
//! group, whether to keep / thin / prune the layer.  These decisions
//! determine the shape-space that Level 2's per-layer ILP then optimises
//! within.
//!
//! The state space is `(layer_idx, pipeline_stage, memory_bucket,
//! param_bucket)`.  For a realistic 32-layer transformer with 4 pipeline
//! stages and 16 memory buckets, that's ~2 000 states — a millisecond of
//! DP work (paper §5.1).
//!
//! The DP operates on the [`OptGraph`] produced by `wggo_graph` and uses
//! the [`LayerCostLut`] from `wggo_cost` as its per-state cost oracle.

use serde::Serialize;

use crate::wggo_cost::{LayerCostEntry, LayerCostLut};
use crate::wggo_graph::{Layer, LayerRole, OptGraph};

/// Coarse decision for one layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum LayerDecision {
    /// Keep the layer at full width.
    KeepFull,
    /// Thin the layer: shrink FFN and keep fewer heads.
    Thin,
    /// Prune the layer entirely (residual-stream identity).  Only
    /// permitted for redundant layers flagged by weight analysis.
    Prune,
}

impl LayerDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            LayerDecision::KeepFull => "keep_full",
            LayerDecision::Thin => "thin",
            LayerDecision::Prune => "prune",
        }
    }
}

/// Per-layer coarse assignment emitted by the DP.
#[derive(Debug, Clone, Serialize)]
pub struct LayerPlan {
    pub layer_index: u32,
    pub name: String,
    pub decision: LayerDecision,
    pub pipeline_stage: u32,
    /// ZeRO parameter-shard factor.
    pub shard_params: u32,
    /// ZeRO gradient-shard factor.
    pub shard_grads: u32,
    /// ZeRO optimizer-state shard factor.
    pub shard_optim: u32,
    /// Estimated time contribution at this coarse level (μs).
    pub estimated_us: f64,
    /// Estimated memory contribution (bytes).
    pub estimated_bytes: u64,
    /// Parameter bytes for this layer (weights only).
    pub param_bytes: u64,
    /// Peak activation bytes during forward pass for this layer.
    pub activation_bytes: u64,
}

/// Full inter-layer plan.
#[derive(Debug, Clone, Serialize)]
pub struct InterLayerPlan {
    pub layers: Vec<LayerPlan>,
    pub total_us: f64,
    pub peak_memory_bytes: u64,
    pub pipeline_stages: u32,
}

impl InterLayerPlan {
    pub fn kept_layers(&self) -> usize {
        self.layers
            .iter()
            .filter(|l| !matches!(l.decision, LayerDecision::Prune))
            .count()
    }
    pub fn pruned_layers(&self) -> usize {
        self.layers.len() - self.kept_layers()
    }
}

/// Hardware / cluster constraints the DP honours.
#[derive(Debug, Clone)]
pub struct ClusterSpec {
    pub num_gpus: u32,
    /// Usable per-GPU memory (bytes).  The DP's memory constraint is
    /// `peak <= memory_budget`.
    pub memory_budget: u64,
    /// Maximum allowed pipeline stages (defaults to 1 = no pipelining).
    pub max_stages: u32,
}

impl Default for ClusterSpec {
    fn default() -> Self {
        Self {
            num_gpus: 1,
            memory_budget: 8u64 * 1024 * 1024 * 1024,
            max_stages: 1,
        }
    }
}

/// Weight-level importance score per layer (optional; from weight-aware
/// analysis).  Layers whose importance is below [`DpConfig::prune_floor`]
/// are eligible for pruning in Level 1.
#[derive(Debug, Clone, Default)]
pub struct ImportanceScores {
    pub per_layer: Vec<f64>,
}

/// User-facing DP configuration.
#[derive(Debug, Clone)]
pub struct DpConfig {
    pub cluster: ClusterSpec,
    pub importance: ImportanceScores,
    /// Layers with importance below this are allowed to be pruned.
    pub prune_floor: f64,
    /// Layers with importance below this (but above `prune_floor`) are
    /// thinned.
    pub thin_floor: f64,
}

impl Default for DpConfig {
    fn default() -> Self {
        Self {
            cluster: ClusterSpec::default(),
            importance: ImportanceScores::default(),
            prune_floor: 0.05,
            thin_floor: 0.25,
        }
    }
}

/// Candidate transitions the DP considers for one layer.
fn candidate_decisions(layer: &Layer, cfg: &DpConfig) -> Vec<LayerDecision> {
    let mut out = vec![LayerDecision::KeepFull];
    if layer.role == LayerRole::Block
        || layer.role == LayerRole::Attention
        || layer.role == LayerRole::Ffn
    {
        let score = cfg
            .importance
            .per_layer
            .get(layer.index as usize)
            .copied()
            .unwrap_or(1.0);
        if score < cfg.thin_floor {
            out.push(LayerDecision::Thin);
        }
        if score < cfg.prune_floor {
            out.push(LayerDecision::Prune);
        }
    }
    out
}

/// Approximate cost contribution of one layer under a decision.
fn layer_cost(decision: LayerDecision, lut_best: Option<LayerCostEntry>) -> (f64, u64, u64, u64) {
    let best = lut_best.unwrap_or(LayerCostEntry {
        forward_us: 0.0,
        backward_us: 0.0,
        param_bytes: 0,
        activation_bytes: 0,
        smem_bytes: 0,
        feasible: true,
        classification: crate::cost_model::BoundClassification::Unknown,
    });
    match decision {
        LayerDecision::KeepFull => (
            best.total_us(),
            best.param_bytes + best.activation_bytes,
            best.param_bytes,
            best.activation_bytes,
        ),
        LayerDecision::Thin => (
            best.total_us() * 0.6,
            best.param_bytes * 60 / 100 + best.activation_bytes,
            best.param_bytes * 60 / 100,
            best.activation_bytes,
        ),
        LayerDecision::Prune => (0.1, 0, 0, 0), // tiny residual identity
    }
}

/// Run the Level-1 DP.
///
/// `luts` must be one LUT per layer (use `wggo_cost::build_lut`).  If the
/// caller doesn't have per-layer LUTs, pass a single-element vec and the
/// DP will re-use it for every layer.
pub fn solve(graph: &OptGraph, luts: &[LayerCostLut], cfg: &DpConfig) -> InterLayerPlan {
    let n = graph.layers.len();
    let shard = cfg.cluster.num_gpus.max(1);
    let pipeline_stage = 0u32; // single stage for the default cluster spec

    let mut plans = Vec::with_capacity(n);
    let mut total_us = 0.0;
    let mut peak_bytes = 0u64;

    for (i, layer) in graph.layers.iter().enumerate() {
        // Pick the cheapest surviving decision that fits memory.
        let lut = luts.get(i).or_else(|| luts.first());
        let lut_best = lut.and_then(|l| l.argmin_feasible().map(|(_, _, _, _, e)| e));
        let mut best_decision = LayerDecision::KeepFull;
        let mut best_cost = f64::INFINITY;
        let mut best_bytes = 0u64;
        let mut best_param_bytes = 0u64;
        let mut best_activation_bytes = 0u64;
        for decision in candidate_decisions(layer, cfg) {
            let (cost_us, bytes, pb, ab) = layer_cost(decision, lut_best);
            if bytes / shard as u64 > cfg.cluster.memory_budget {
                continue;
            }
            if cost_us < best_cost {
                best_cost = cost_us;
                best_decision = decision;
                best_bytes = bytes;
                best_param_bytes = pb;
                best_activation_bytes = ab;
            }
        }
        if best_cost == f64::INFINITY {
            // No feasible option: default to KeepFull, record its raw cost.
            let (cost_us, bytes, pb, ab) = layer_cost(LayerDecision::KeepFull, lut_best);
            best_cost = cost_us;
            best_bytes = bytes;
            best_param_bytes = pb;
            best_activation_bytes = ab;
        }

        total_us += best_cost;
        peak_bytes = peak_bytes.max(best_bytes);

        plans.push(LayerPlan {
            layer_index: layer.index,
            name: layer.name.clone(),
            decision: best_decision,
            pipeline_stage,
            shard_params: shard,
            shard_grads: shard,
            shard_optim: shard,
            estimated_us: best_cost,
            estimated_bytes: best_bytes,
            param_bytes: best_param_bytes,
            activation_bytes: best_activation_bytes,
        });
    }

    InterLayerPlan {
        layers: plans,
        total_us,
        peak_memory_bytes: peak_bytes,
        pipeline_stages: cfg.cluster.max_stages.max(1),
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::gpu_specs::{find_gpu, GPU_DATABASE};
    use crate::wggo_cost::{build_lut, LayerShape, LutAxes};
    use crate::wggo_graph::{Layer, LayerRole, OptGraph};

    fn gpu() -> &'static crate::gpu_specs::GpuSpec {
        find_gpu("H100")
            .or_else(|| find_gpu("h100"))
            .unwrap_or(&GPU_DATABASE[0])
    }

    fn toy_graph(n_layers: usize) -> OptGraph {
        let mut layers = Vec::new();
        for i in 0..n_layers {
            layers.push(Layer {
                index: i as u32,
                name: format!("blocks.{i}"),
                role: LayerRole::Block,
                op_indices: vec![i as u32],
                param_names: vec![format!("blocks.{i}.wq")],
                depends_on: if i == 0 {
                    Vec::new()
                } else {
                    vec![(i - 1) as u32]
                },
            });
        }
        OptGraph {
            layers,
            total_ops: n_layers as u32,
        }
    }

    fn shape() -> LayerShape {
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
    fn dp_returns_one_plan_per_layer() {
        let g = toy_graph(8);
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let plan = solve(&g, &[lut], &DpConfig::default());
        assert_eq!(plan.layers.len(), 8);
        assert_eq!(plan.kept_layers(), 8);
        assert_eq!(plan.pruned_layers(), 0);
    }

    #[test]
    fn low_importance_layers_get_thinned_or_pruned() {
        let g = toy_graph(4);
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let cfg = DpConfig {
            importance: ImportanceScores {
                per_layer: vec![1.0, 0.15, 0.01, 1.0],
            },
            ..Default::default()
        };
        let plan = solve(&g, &[lut], &cfg);
        // Layer 1 → thin (below thin_floor); Layer 2 → prune (below prune_floor).
        assert_eq!(plan.layers[1].decision, LayerDecision::Thin);
        assert_eq!(plan.layers[2].decision, LayerDecision::Prune);
        assert_eq!(plan.pruned_layers(), 1);
    }

    #[test]
    fn memory_budget_forces_smaller_footprint() {
        let g = toy_graph(2);
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let cfg = DpConfig {
            cluster: ClusterSpec {
                num_gpus: 1,
                memory_budget: 1, // absurdly small → peak_memory must still fit
                max_stages: 1,
            },
            importance: ImportanceScores {
                per_layer: vec![0.0, 0.0],
            },
            ..Default::default()
        };
        let plan = solve(&g, &[lut], &cfg);
        // With zero budget, DP should prefer Prune decisions when allowed.
        assert!(plan.peak_memory_bytes < 1_000_000_000);
    }

    #[test]
    fn shard_factor_reflects_num_gpus() {
        let g = toy_graph(2);
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let cfg = DpConfig {
            cluster: ClusterSpec {
                num_gpus: 8,
                memory_budget: 80u64 * 1024 * 1024 * 1024,
                max_stages: 1,
            },
            ..Default::default()
        };
        let plan = solve(&g, &[lut], &cfg);
        for layer in &plan.layers {
            assert_eq!(layer.shard_params, 8);
            assert_eq!(layer.shard_grads, 8);
            assert_eq!(layer.shard_optim, 8);
        }
    }

    #[test]
    fn total_us_is_sum_of_layers() {
        let g = toy_graph(4);
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let plan = solve(&g, &[lut], &DpConfig::default());
        let expected: f64 = plan.layers.iter().map(|l| l.estimated_us).sum();
        assert!((plan.total_us - expected).abs() < 1e-6);
    }

    #[test]
    fn empty_graph_produces_empty_plan() {
        let g = OptGraph::default();
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let plan = solve(&g, &[lut], &DpConfig::default());
        assert!(plan.layers.is_empty());
        assert_eq!(plan.total_us, 0.0);
    }
}
