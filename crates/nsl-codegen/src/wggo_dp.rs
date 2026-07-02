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

use crate::gpu_specs::GpuSpec;
use crate::wggo_cost::{comm_optim_us, LayerCostEntry, LayerCostLut};
use crate::wggo_graph::{Layer, LayerRole, OptGraph};

/// Error returned by [`solve`] when no assignment of per-layer decisions
/// fits the cluster memory budget — even at the smallest feasible
/// configuration for every layer.
///
/// The DP **refuses** rather than silently emitting an over-budget plan
/// (project invariant: a transformation precondition that cannot be met
/// must refuse, never silently weaken).  The driver converts this into a
/// graceful degradation + warning.
#[derive(Debug, Clone, PartialEq)]
pub enum DpError {
    Infeasible { required_bytes: u64, budget: u64 },
}

impl std::fmt::Display for DpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DpError::Infeasible {
                required_bytes,
                budget,
            } => write!(
                f,
                "WGGO inter-layer DP infeasible: smallest feasible config needs \
                 {required_bytes} bytes but the per-GPU budget is {budget} bytes"
            ),
        }
    }
}

impl std::error::Error for DpError {}

/// Level-1 (inter-layer DP) coarse disposition for one layer.
///
/// Distinct from [`crate::wggo_ilp::LayerDecision`], which is the rich
/// Level-2 (per-layer ILP) decision (head keep-mask, FFN width, CSHA level,
/// adapter rank/placement, optimizer precision, …).  This enum only records
/// the DP's keep/thin/prune disposition; renamed from `LayerDecision` to end
/// the name collision (G13).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CoarseDecision {
    /// Keep the layer at full width.
    KeepFull,
    /// Thin the layer: shrink FFN and keep fewer heads.
    Thin,
    /// Prune the layer entirely (residual-stream identity).  Only
    /// permitted for redundant layers flagged by weight analysis.
    Prune,
}

impl CoarseDecision {
    pub fn as_str(self) -> &'static str {
        match self {
            CoarseDecision::KeepFull => "keep_full",
            CoarseDecision::Thin => "thin",
            CoarseDecision::Prune => "prune",
        }
    }
}

/// Per-layer coarse assignment emitted by the DP.
#[derive(Debug, Clone, Serialize)]
pub struct LayerPlan {
    pub layer_index: u32,
    pub name: String,
    pub decision: CoarseDecision,
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
            .filter(|l| !matches!(l.decision, CoarseDecision::Prune))
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
    /// Inter-GPU interconnect bandwidth (GB/s) used to estimate ZeRO
    /// collective latency.  Default `300.0` is an NVLink-class heuristic;
    /// the repo has no measured collective-latency oracle.
    pub interconnect_gbs: f64,
}

impl Default for ClusterSpec {
    fn default() -> Self {
        Self {
            num_gpus: 1,
            memory_budget: 8u64 * 1024 * 1024 * 1024,
            max_stages: 1,
            interconnect_gbs: 300.0,
        }
    }
}

/// ZeRO sharding stage — the single per-layer sharding decision (errata E1 /
/// audit gap #6).  ZeRO is a **nested chain**, not three independent factors:
/// each higher stage shards a superset of the previous stage's tensors, so a
/// single ordered stage captures the real design space without the incoherent
/// cross-product (e.g. sharded params but replicated gradients) that no runtime
/// implements.
///
/// - `Ddp`         (stage 0): replicate everything (plain data-parallel).
/// - `Os`          (stage 1): shard the optimizer state (Adam `m`, `v`).
/// - `OsGrad`      (stage 2): + shard the gradients.
/// - `OsGradParam` (stage 3): + shard the parameters (full FSDP).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum ZeroStage {
    Ddp,
    Os,
    OsGrad,
    OsGradParam,
}

impl ZeroStage {
    /// Every stage, in increasing sharding order.
    pub const ALL: [ZeroStage; 4] = [
        ZeroStage::Ddp,
        ZeroStage::Os,
        ZeroStage::OsGrad,
        ZeroStage::OsGradParam,
    ];

    /// Decode a stage index `0..=3`.
    pub fn from_index(i: u8) -> Option<ZeroStage> {
        match i {
            0 => Some(ZeroStage::Ddp),
            1 => Some(ZeroStage::Os),
            2 => Some(ZeroStage::OsGrad),
            3 => Some(ZeroStage::OsGradParam),
            _ => None,
        }
    }

    /// Per-component shard factors `(param, grad, optim)` for `num_gpus`.
    ///
    /// A single GPU cannot shard, so every stage collapses to `(1, 1, 1)`.
    /// Otherwise the nested chain shards optimizer state at stage ≥ 1,
    /// gradients at stage ≥ 2, and parameters at stage 3.
    pub fn shards(self, num_gpus: u32) -> (u32, u32, u32) {
        let n = num_gpus.max(1);
        if n == 1 {
            return (1, 1, 1);
        }
        match self {
            ZeroStage::Ddp => (1, 1, 1),
            ZeroStage::Os => (1, 1, n),
            ZeroStage::OsGrad => (1, n, n),
            ZeroStage::OsGradParam => (n, n, n),
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
    /// Opt-in ZeRO-stage search (audit gap #6).  When `false` (default) the DP
    /// weighs the legacy binary shard factor `{1, num_gpus}` and behaves exactly
    /// as before.  When `true` and `num_gpus > 1`, it instead enumerates the
    /// full ZeRO-stage chain [`ZeroStage::ALL`] per layer — the partial stages
    /// (optimizer-only, +gradients) trade less memory for less communication.
    /// Off by default because errata E1 lists the stage enum as a future
    /// extension awaiting author sign-off.
    ///
    /// **Activation follow-up:** some downstream consumers still read only
    /// `LayerPlan.shard_params` (e.g. the CPDT `shard_factor` in `wggo.rs`), so a
    /// partial stage (say stage 1 with `shard_params = 1` but `shard_optim = N`)
    /// would under-report its sharding to them. Update those consumers to honour
    /// all three factors before enabling this in production.
    pub zero_stage_search: bool,
}

impl Default for DpConfig {
    fn default() -> Self {
        Self {
            cluster: ClusterSpec::default(),
            importance: ImportanceScores::default(),
            prune_floor: 0.05,
            thin_floor: 0.25,
            zero_stage_search: false,
        }
    }
}

/// Candidate transitions the DP considers for one layer.
fn candidate_decisions(layer: &Layer, cfg: &DpConfig) -> Vec<CoarseDecision> {
    let mut out = vec![CoarseDecision::KeepFull];
    if layer.role == LayerRole::Block || layer.role == LayerRole::Attention || layer.role == LayerRole::Ffn {
        let score = cfg
            .importance
            .per_layer
            .get(layer.index as usize)
            .copied()
            .unwrap_or(1.0);
        if score < cfg.thin_floor {
            out.push(CoarseDecision::Thin);
        }
        if score < cfg.prune_floor {
            out.push(CoarseDecision::Prune);
        }
    }
    out
}

/// Approximate cost contribution of one layer under a decision.
fn layer_cost(decision: CoarseDecision, lut_best: Option<LayerCostEntry>) -> (f64, u64, u64, u64) {
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
        CoarseDecision::KeepFull => (
            best.total_us(),
            best.param_bytes + best.activation_bytes,
            best.param_bytes,
            best.activation_bytes,
        ),
        CoarseDecision::Thin => (
            best.total_us() * 0.6,
            best.param_bytes * 60 / 100 + best.activation_bytes,
            best.param_bytes * 60 / 100,
            best.activation_bytes,
        ),
        CoarseDecision::Prune => (0.1, 0, 0, 0), // tiny residual identity
    }
}

/// Number of discretized memory buckets the knapsack DP tracks.
const NB: usize = 256;

/// One candidate `(decision, sharding)` assignment for a single layer, with its
/// resolved cost and resident-memory footprint.  The three shard factors are
/// equal in the legacy binary path and follow the [`ZeroStage`] nested chain
/// under `zero_stage_search`.
#[derive(Debug, Clone, Copy)]
struct Cand {
    decision: CoarseDecision,
    shard_param: u32,
    shard_grad: u32,
    shard_optim: u32,
    /// forward + backward + comm + optimizer latency (μs).
    cost_us: f64,
    /// Per-GPU resident memory for this layer (bytes).
    resident: u64,
    param_bytes: u64,
    activation_bytes: u64,
}

/// Resident-memory model for a layer: parameters plus the two Adam moment
/// buffers sharded `shard` ways, plus un-sharded activations.
///
/// The Level-1 DP is coarse — it has no optimizer-precision variable — so it
/// sizes the two moments at parameter precision (`optimizer_state = 2·param`,
/// giving the historical `3·param/shard + activation`).  It routes through the
/// one shared formula ([`crate::wggo_cost::resident_training_bytes`]) that the
/// Level-2 ILP also uses, so the two levels can never diverge on the memory
/// model (audit gap #3); only the fidelity of the optimizer-state input differs
/// (the ILP sizes moments at their chosen precision).
fn resident_bytes(param_bytes: u64, activation_bytes: u64, shard: u32) -> u64 {
    let optimizer_state = 2u64.saturating_mul(param_bytes);
    crate::wggo_cost::resident_training_bytes(param_bytes, optimizer_state, activation_bytes, shard)
}

/// Enumerate the `(decision, sharding)` candidates for one layer.
///
/// Default (`zero_stage_search = false`): the legacy binary shard factor —
/// `{1}` on a single GPU and `{1, num_gpus}` on a cluster, applied uniformly to
/// params/grads/optim, with the coarse `3·param/shard` memory model.
///
/// `zero_stage_search = true` (and `num_gpus > 1`): the full ZeRO-stage chain
/// [`ZeroStage::ALL`], where each stage shards params/grads/optim independently
/// (nested chain) and carries the faithful per-component memory model
/// (including gradient memory), so the partial stages are genuinely distinct
/// memory/comm points the DP can choose between.
fn layer_candidates(
    layer: &Layer,
    cfg: &DpConfig,
    lut_best: Option<LayerCostEntry>,
    gpu: &GpuSpec,
) -> Vec<Cand> {
    let n = cfg.cluster.num_gpus;
    let mut out = Vec::new();
    for decision in candidate_decisions(layer, cfg) {
        let (compute_us, _bytes, param_bytes, activation_bytes) = layer_cost(decision, lut_best);
        if cfg.zero_stage_search && n > 1 {
            // Faithful ZeRO-stage search (audit gap #6).
            for stage in ZeroStage::ALL {
                let (sp, sg, so) = stage.shards(n);
                let (comm_us, optim_us) = crate::wggo_cost::zero_stage_costs(
                    param_bytes, sp, sg, so, cfg.cluster.interconnect_gbs, gpu,
                );
                let resident = crate::wggo_cost::resident_training_bytes_sharded(
                    param_bytes,
                    param_bytes, // gradients are the same size as parameters
                    2u64.saturating_mul(param_bytes), // Adam m + v
                    activation_bytes,
                    sp,
                    sg,
                    so,
                );
                out.push(Cand {
                    decision,
                    shard_param: sp,
                    shard_grad: sg,
                    shard_optim: so,
                    cost_us: compute_us + comm_us + optim_us,
                    resident,
                    param_bytes,
                    activation_bytes,
                });
            }
        } else {
            // Legacy binary shard factor — byte-identical to pre-gap-#6.
            let shard_opts: [u32; 2] = [1, n];
            let take = if n > 1 { 2 } else { 1 };
            for &shard in &shard_opts[..take] {
                let (comm_us, optim_us) =
                    comm_optim_us(param_bytes, shard, shard, cfg.cluster.interconnect_gbs, gpu);
                out.push(Cand {
                    decision,
                    shard_param: shard,
                    shard_grad: shard,
                    shard_optim: shard,
                    cost_us: compute_us + comm_us + optim_us,
                    resident: resident_bytes(param_bytes, activation_bytes, shard),
                    param_bytes,
                    activation_bytes,
                });
            }
        }
    }
    out
}

/// Assign each layer to one of `stages` contiguous pipeline stages, minimizing
/// the maximum per-stage compute load (classic min-max contiguous partition
/// DP).  Returns a stage index in `0..stages` for every layer.
fn assign_stages(costs: &[f64], stages: usize) -> Vec<u32> {
    let n = costs.len();
    if n == 0 {
        return Vec::new();
    }
    let stages = stages.max(1).min(n);
    if stages == 1 {
        return vec![0; n];
    }
    let mut pre = vec![0.0f64; n + 1];
    for i in 0..n {
        pre[i + 1] = pre[i] + costs[i];
    }
    let inf = f64::INFINITY;
    // dpp[k][i] = min achievable max-load using k groups over the first i layers.
    let mut dpp = vec![vec![inf; n + 1]; stages + 1];
    let mut cut = vec![vec![0usize; n + 1]; stages + 1];
    dpp[0][0] = 0.0;
    for k in 1..=stages {
        for i in k..=n {
            for p in (k - 1)..i {
                let prev = dpp[k - 1][p];
                if prev.is_finite() {
                    let group_load = pre[i] - pre[p];
                    let m = prev.max(group_load);
                    if m < dpp[k][i] {
                        dpp[k][i] = m;
                        cut[k][i] = p;
                    }
                }
            }
        }
    }
    let mut stage_of = vec![0u32; n];
    let mut i = n;
    for k in (1..=stages).rev() {
        let p = cut[k][i];
        for layer in p..i {
            stage_of[layer] = (k - 1) as u32;
        }
        i = p;
    }
    stage_of
}

/// Run the Level-1 inter-layer DP.
///
/// Solves the joint keep/thin/prune + ZeRO-shard assignment as a memory-
/// bucketed knapsack: minimize total step time `Σ (forward + backward + comm +
/// optimizer)` subject to `Σ resident_bytes ≤ memory_budget`.  Unlike a greedy
/// per-layer pass, the DP can shard or thin one layer to make room for another
/// (it optimizes the global budget, not each layer in isolation), and it
/// **refuses** with [`DpError::Infeasible`] when no assignment fits rather than
/// emitting an over-budget plan.
///
/// `luts` must be one LUT per layer (use `wggo_cost::build_lut`).  If the
/// caller doesn't have per-layer LUTs, pass a single-element vec and the DP
/// will re-use it for every layer.  `gpu` is the target spec used to cost the
/// communication and optimizer terms.
pub fn solve(
    graph: &OptGraph,
    luts: &[LayerCostLut],
    cfg: &DpConfig,
    gpu: &GpuSpec,
) -> Result<InterLayerPlan, DpError> {
    let n = graph.layers.len();
    let budget = cfg.cluster.memory_budget.max(1);
    let bucket_bytes = budget.div_ceil(NB as u64).max(1);

    // Per-layer candidate sets.
    let cands: Vec<Vec<Cand>> = graph
        .layers
        .iter()
        .enumerate()
        .map(|(i, layer)| {
            let lut = luts.get(i).or_else(|| luts.first());
            let lut_best = lut.and_then(|l| l.argmin_feasible().map(|(_, _, _, _, e)| e));
            layer_candidates(layer, cfg, lut_best, gpu)
        })
        .collect();

    // Knapsack DP over memory buckets: dp[j][b] = min cost for the first j
    // layers using ≤ b buckets of resident memory.  choice records the chosen
    // candidate index and the predecessor bucket for backtracking.
    let inf = f64::INFINITY;
    let mut dp = vec![vec![inf; NB + 1]; n + 1];
    let mut choice: Vec<Vec<Option<(usize, usize)>>> = vec![vec![None; NB + 1]; n + 1];
    for b in 0..=NB {
        dp[0][b] = 0.0;
    }
    for j in 1..=n {
        let layer_cands = &cands[j - 1];
        for b in 0..=NB {
            for (ci, c) in layer_cands.iter().enumerate() {
                let bc = c.resident.div_ceil(bucket_bytes) as usize;
                if bc > b {
                    continue;
                }
                let prev = dp[j - 1][b - bc];
                if prev.is_finite() {
                    let total = prev + c.cost_us;
                    if total < dp[j][b] {
                        dp[j][b] = total;
                        choice[j][b] = Some((ci, b - bc));
                    }
                }
            }
        }
    }

    // Best terminal bucket (min cost using ≤ NB buckets = ≤ budget).
    let mut best_b = 0usize;
    let mut best_cost = inf;
    for b in 0..=NB {
        if dp[n][b] < best_cost {
            best_cost = dp[n][b];
            best_b = b;
        }
    }
    if !best_cost.is_finite() {
        // Even the smallest feasible config per layer overflows the budget.
        let required: u64 = cands
            .iter()
            .map(|cs| cs.iter().map(|c| c.resident).min().unwrap_or(0))
            .sum();
        return Err(DpError::Infeasible {
            required_bytes: required,
            budget,
        });
    }

    // Backtrack the chosen candidate for every layer.
    let mut chosen: Vec<Option<Cand>> = vec![None; n];
    let mut b = best_b;
    for j in (1..=n).rev() {
        let (ci, prev_b) = choice[j][b]
            .expect("DP backtrack: every layer on a finite-cost path has a recorded choice");
        chosen[j - 1] = Some(cands[j - 1][ci]);
        b = prev_b;
    }
    let chosen: Vec<Cand> = chosen
        .into_iter()
        .map(|c| c.expect("DP backtrack fills a candidate for every layer"))
        .collect();

    // Pipeline-stage assignment + per-stage peak memory.
    let stages = (cfg.cluster.max_stages.max(1) as usize).min(n.max(1));
    let costs: Vec<f64> = chosen.iter().map(|c| c.cost_us).collect();
    let stage_of = assign_stages(&costs, stages);
    let mut stage_resident = vec![0u64; stages.max(1)];
    for (idx, c) in chosen.iter().enumerate() {
        let s = *stage_of.get(idx).unwrap_or(&0) as usize;
        stage_resident[s] = stage_resident[s].saturating_add(c.resident);
    }
    let peak = stage_resident.iter().copied().max().unwrap_or(0);

    let mut plans = Vec::with_capacity(n);
    let mut total_us = 0.0;
    for (idx, layer) in graph.layers.iter().enumerate() {
        let c = chosen[idx];
        total_us += c.cost_us;
        plans.push(LayerPlan {
            layer_index: layer.index,
            name: layer.name.clone(),
            decision: c.decision,
            pipeline_stage: *stage_of.get(idx).unwrap_or(&0),
            shard_params: c.shard_param,
            shard_grads: c.shard_grad,
            shard_optim: c.shard_optim,
            estimated_us: c.cost_us,
            estimated_bytes: c.resident,
            param_bytes: c.param_bytes,
            activation_bytes: c.activation_bytes,
        });
    }

    Ok(InterLayerPlan {
        layers: plans,
        total_us,
        peak_memory_bytes: peak,
        pipeline_stages: stages.max(1) as u32,
    })
}

/// Budget-agnostic passthrough plan: every layer `KeepFull` at the cluster's
/// default shard factor, single stage.  Used by Off mode (which bypasses
/// optimization) and as the driver's interim fallback when [`solve`] refuses a
/// budget.
pub fn passthrough_plan(
    graph: &OptGraph,
    luts: &[LayerCostLut],
    cfg: &DpConfig,
    gpu: &GpuSpec,
) -> InterLayerPlan {
    let shard = cfg.cluster.num_gpus.max(1);
    let mut plans = Vec::with_capacity(graph.layers.len());
    let mut total_us = 0.0;
    let mut total_bytes = 0u64;
    for (i, layer) in graph.layers.iter().enumerate() {
        let lut = luts.get(i).or_else(|| luts.first());
        let lut_best = lut.and_then(|l| l.argmin_feasible().map(|(_, _, _, _, e)| e));
        let (compute_us, _bytes, param_bytes, activation_bytes) =
            layer_cost(CoarseDecision::KeepFull, lut_best);
        let (comm_us, optim_us) =
            comm_optim_us(param_bytes, shard, shard, cfg.cluster.interconnect_gbs, gpu);
        let resident = resident_bytes(param_bytes, activation_bytes, shard);
        let cost = compute_us + comm_us + optim_us;
        total_us += cost;
        total_bytes = total_bytes.saturating_add(resident);
        plans.push(LayerPlan {
            layer_index: layer.index,
            name: layer.name.clone(),
            decision: CoarseDecision::KeepFull,
            pipeline_stage: 0,
            shard_params: shard,
            shard_grads: shard,
            shard_optim: shard,
            estimated_us: cost,
            estimated_bytes: resident,
            param_bytes,
            activation_bytes,
        });
    }
    InterLayerPlan {
        layers: plans,
        total_us,
        peak_memory_bytes: total_bytes,
        pipeline_stages: 1,
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
                depends_on: if i == 0 { Vec::new() } else { vec![(i - 1) as u32] },
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
        let plan = solve(&g, &[lut], &DpConfig::default(), gpu()).expect("feasible");
        assert_eq!(plan.layers.len(), 8);
        assert_eq!(plan.kept_layers(), 8);
        assert_eq!(plan.pruned_layers(), 0);
    }

    #[test]
    fn zero_stage_shards_follow_the_nested_chain() {
        let n = 8;
        // (param, grad, optim) shard factors for each ZeRO stage.
        assert_eq!(ZeroStage::Ddp.shards(n), (1, 1, 1)); // stage 0: nothing sharded
        assert_eq!(ZeroStage::Os.shards(n), (1, 1, n)); // stage 1: optimizer state
        assert_eq!(ZeroStage::OsGrad.shards(n), (1, n, n)); // stage 2: + gradients
        assert_eq!(ZeroStage::OsGradParam.shards(n), (n, n, n)); // stage 3: + parameters
        // A single GPU collapses every stage to no sharding.
        for st in ZeroStage::ALL {
            assert_eq!(st.shards(1), (1, 1, 1));
        }
        // The chain is monotone: each higher stage shards a superset of tensors.
        for st in ZeroStage::ALL {
            let (p, g, o) = st.shards(n);
            assert!(o >= g && g >= p, "sharding is nested optim ⊇ grad ⊇ param");
        }
        // Index round-trip.
        assert_eq!(ZeroStage::from_index(0), Some(ZeroStage::Ddp));
        assert_eq!(ZeroStage::from_index(3), Some(ZeroStage::OsGradParam));
        assert_eq!(ZeroStage::from_index(4), None);
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
        let plan = solve(&g, &[lut], &cfg, gpu()).expect("feasible");
        // Layer 1 → thin (below thin_floor); Layer 2 → prune (below prune_floor).
        // With an ample budget the DP picks these because they are cheaper, not
        // because memory forces it.
        assert_eq!(plan.layers[1].decision, CoarseDecision::Thin);
        assert_eq!(plan.layers[2].decision, CoarseDecision::Prune);
        assert_eq!(plan.pruned_layers(), 1);
    }

    #[test]
    fn memory_budget_forces_smaller_footprint() {
        let g = toy_graph(2);
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let cfg = DpConfig {
            cluster: ClusterSpec {
                num_gpus: 1,
                memory_budget: 1, // absurdly small → only Prune (0 resident) fits
                max_stages: 1,
                interconnect_gbs: 300.0,
            },
            importance: ImportanceScores {
                per_layer: vec![0.0, 0.0],
            },
            ..Default::default()
        };
        let plan = solve(&g, &[lut], &cfg, gpu()).expect("prune makes it feasible");
        assert!(plan.peak_memory_bytes < 1_000_000_000);
    }

    #[test]
    fn dp_rejects_infeasible_memory() {
        // KeepFull only (importance 1.0), single GPU (no shard relief), budget
        // of 1 byte → no assignment fits → the DP must refuse, not emit an
        // over-budget plan.
        let g = toy_graph(2);
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let cfg = DpConfig {
            cluster: ClusterSpec {
                num_gpus: 1,
                memory_budget: 1,
                max_stages: 1,
                interconnect_gbs: 300.0,
            },
            importance: ImportanceScores {
                per_layer: vec![1.0, 1.0],
            },
            ..Default::default()
        };
        let err = solve(&g, &[lut], &cfg, gpu()).unwrap_err();
        assert!(matches!(err, DpError::Infeasible { .. }));
    }

    #[test]
    fn dp_shards_under_memory_pressure() {
        // 8 GPUs available; budget only large enough for the sharded footprint.
        // A greedy per-layer pass would pick shard=1 (cheaper cost) and overflow
        // the global budget; the DP must shard every layer to fit.
        let g = toy_graph(4);
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let one = lut.argmin_feasible().expect("feasible entry").4;
        let sharded = 3 * one.param_bytes / 8 + one.activation_bytes;
        let budget = 4 * sharded + sharded / 2;
        let cfg = DpConfig {
            cluster: ClusterSpec {
                num_gpus: 8,
                memory_budget: budget,
                max_stages: 1,
                interconnect_gbs: 300.0,
            },
            importance: ImportanceScores {
                per_layer: vec![1.0; 4],
            },
            ..Default::default()
        };
        let plan = solve(&g, &[lut], &cfg, gpu()).expect("feasible under sharding");
        assert!(
            plan.layers.iter().all(|l| l.shard_params == 8),
            "all layers must shard to fit the global budget"
        );
        let total: u64 = plan.layers.iter().map(|l| l.estimated_bytes).sum();
        assert!(total <= budget, "total {total} > budget {budget}");
    }

    #[test]
    fn dp_no_shard_when_budget_ample() {
        // Memory is not binding, so the DP keeps shard=1: on H100-class HBM
        // bandwidth the added ZeRO communication exceeds the optimizer-memory
        // saving, so sharding is a net cost loss.  (The old greedy pass forced
        // shard = num_gpus unconditionally — this is the fixed behavior.)
        let g = toy_graph(4);
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let cfg = DpConfig {
            cluster: ClusterSpec {
                num_gpus: 8,
                memory_budget: 80u64 * 1024 * 1024 * 1024,
                max_stages: 1,
                interconnect_gbs: 300.0,
            },
            importance: ImportanceScores {
                per_layer: vec![1.0; 4],
            },
            ..Default::default()
        };
        let plan = solve(&g, &[lut], &cfg, gpu()).expect("feasible");
        assert!(plan.layers.iter().all(|l| l.shard_params == 1));
    }

    #[test]
    fn zero_stage_search_picks_partial_sharding_the_binary_model_cannot() {
        // Same ample-budget, 8-GPU setup as `dp_no_shard_when_budget_ample`
        // (which keeps shard=1 under the legacy binary model). With ZeRO-stage
        // search ON, sharding only the optimizer state (stage 1) is a strict
        // win — no extra collective, yet less optimizer-step HBM traffic — so
        // the DP picks a PARTIAL stage the binary {1,N} model cannot express.
        let g = toy_graph(4);
        let luts = [build_lut(&shape(), gpu(), &LutAxes::default())];
        let cfg = DpConfig {
            cluster: ClusterSpec {
                num_gpus: 8,
                memory_budget: 80u64 * 1024 * 1024 * 1024,
                max_stages: 1,
                interconnect_gbs: 300.0,
            },
            importance: ImportanceScores {
                per_layer: vec![1.0; 4],
            },
            zero_stage_search: true,
            ..Default::default()
        };
        let plan = solve(&g, &luts, &cfg, gpu()).expect("feasible");
        for l in &plan.layers {
            assert_eq!(l.shard_optim, 8, "optimizer state sharded (stage ≥ 1)");
            assert_eq!(l.shard_params, 1, "parameters replicated (stage < 3)");
            // Every plan is a real ZeRO stage: optim ⊇ grad ⊇ param.
            assert!(l.shard_optim >= l.shard_grads && l.shard_grads >= l.shard_params);
        }
        // Contrast: flag OFF reproduces the legacy all-replicated outcome.
        let legacy = DpConfig {
            zero_stage_search: false,
            ..cfg.clone()
        };
        let legacy_plan = solve(&g, &luts, &legacy, gpu()).expect("feasible");
        assert!(legacy_plan
            .layers
            .iter()
            .all(|l| l.shard_params == 1 && l.shard_optim == 1));
    }

    #[test]
    fn dp_thins_to_fit_budget() {
        // Single GPU (no shard relief); budget only fits the thinned footprint.
        let g = toy_graph(2);
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let one = lut.argmin_feasible().expect("feasible entry").4;
        let thin = 3 * (one.param_bytes * 60 / 100) + one.activation_bytes;
        let budget = 2 * thin + thin / 4;
        let cfg = DpConfig {
            cluster: ClusterSpec {
                num_gpus: 1,
                memory_budget: budget,
                max_stages: 1,
                interconnect_gbs: 300.0,
            },
            importance: ImportanceScores {
                per_layer: vec![0.1, 0.1],
            },
            ..Default::default()
        };
        let plan = solve(&g, &[lut], &cfg, gpu()).expect("feasible by thinning");
        assert!(plan.layers.iter().all(|l| l.decision == CoarseDecision::Thin));
        let total: u64 = plan.layers.iter().map(|l| l.estimated_bytes).sum();
        assert!(total <= budget, "total {total} > budget {budget}");
    }

    #[test]
    fn dp_pipeline_stages_assigned() {
        let g = toy_graph(4);
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let cfg = DpConfig {
            cluster: ClusterSpec {
                num_gpus: 1,
                memory_budget: 80u64 * 1024 * 1024 * 1024,
                max_stages: 2,
                interconnect_gbs: 300.0,
            },
            ..Default::default()
        };
        let plan = solve(&g, &[lut], &cfg, gpu()).expect("feasible");
        let stages: std::collections::BTreeSet<u32> =
            plan.layers.iter().map(|l| l.pipeline_stage).collect();
        assert_eq!(stages, [0, 1].into_iter().collect());
        assert_eq!(plan.pipeline_stages, 2);
    }

    #[test]
    fn total_us_is_sum_of_layers() {
        let g = toy_graph(4);
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let plan = solve(&g, &[lut], &DpConfig::default(), gpu()).expect("feasible");
        let expected: f64 = plan.layers.iter().map(|l| l.estimated_us).sum();
        assert!((plan.total_us - expected).abs() < 1e-6);
    }

    #[test]
    fn empty_graph_produces_empty_plan() {
        let g = OptGraph::default();
        let lut = build_lut(&shape(), gpu(), &LutAxes::default());
        let plan = solve(&g, &[lut], &DpConfig::default(), gpu()).expect("feasible");
        assert!(plan.layers.is_empty());
        assert_eq!(plan.total_us, 0.0);
    }
}
