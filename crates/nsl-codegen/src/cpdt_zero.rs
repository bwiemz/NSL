//! CPDT — analytical ZeRO-config planner.
//!
//! Enumerates the `(s_p, s_g, s_os)` cross-product of shard factors and
//! ranks feasible configurations by analytical step-time.  The whole
//! search runs in ~200 ms because every per-config evaluation is a
//! closed-form cost-model call (paper §3.1).
//!
//! No runtime profiling; no external solver.  The winner is a
//! compile-time constant the downstream comm scheduler consumes.

use serde::Serialize;

/// AdamW optimizer-state multiplier applied to parameter **bytes**
/// (`ModelSize::per_layer_param_bytes` stores bytes, not element counts):
/// m-buffer (fp32) + v-buffer (fp32) = 2 buffers at the same width as the
/// fp32 master parameters → optimizer state = 2 × param bytes.
///
/// (Equivalently 8 bytes per param **element** — the historical value 8.0
/// was that per-element figure mistakenly applied to a bytes quantity,
/// overstating optimizer memory 4×. The Level-1 DP's
/// `wggo_dp::resident_bytes` has always used `2 × param_bytes`; this
/// constant now matches it, per the audit-gap-#3 rule that the levels must
/// not diverge on the memory model.)
pub const ADAMW_FP32_OPTIM_MULTIPLIER: f64 = 2.0;

/// Cluster / interconnect description.
#[derive(Debug, Clone)]
pub struct ClusterSpec {
    pub num_gpus: u32,
    /// Per-GPU memory budget in bytes.
    pub memory_budget_bytes: u64,
    /// Intra-node bandwidth (bytes/s).  NVLink: 9e11.
    pub intra_bw_bps: f64,
    /// Inter-node bandwidth (bytes/s).  InfiniBand: 1e11.
    pub inter_bw_bps: f64,
    /// Number of GPUs per node (used to classify intra- vs inter-node
    /// traffic).
    pub gpus_per_node: u32,
}

impl Default for ClusterSpec {
    fn default() -> Self {
        Self {
            num_gpus: 8,
            memory_budget_bytes: 80u64 * 1024 * 1024 * 1024,
            intra_bw_bps: 9e11,
            inter_bw_bps: 1e11,
            gpus_per_node: 8,
        }
    }
}

/// Compact model-size description the planner consumes.
#[derive(Debug, Clone)]
pub struct ModelSize {
    /// Per-layer parameter bytes.
    pub per_layer_param_bytes: Vec<u64>,
    /// Per-layer activation bytes (peak).
    pub per_layer_activation_bytes: Vec<u64>,
    /// Optimizer-state multiplier applied to parameter BYTES (2.0 for
    /// fp32 AdamW: m+v at the same dtype as the fp32 master params).
    pub optim_state_multiplier: f64,
    /// Per-layer forward / backward compute time in microseconds
    /// (used for overlap efficiency).
    pub per_layer_compute_us: Vec<f64>,
    /// True when `@checkpoint(policy=...)` activation checkpointing is
    /// active for this compile (non-empty `checkpoint_policies`). Switches
    /// [`ModelSize::retained_activation_bytes`] from the full sequential
    /// live-set (sum) to a checkpoint-aware estimate.
    pub activation_checkpointing: bool,
}

impl ModelSize {
    pub fn total_param_bytes(&self) -> u64 {
        self.per_layer_param_bytes.iter().sum()
    }
    pub fn total_activation_bytes(&self) -> u64 {
        self.per_layer_activation_bytes.iter().sum()
    }
    pub fn peak_activation_bytes(&self) -> u64 {
        self.per_layer_activation_bytes.iter().max().copied().unwrap_or(0)
    }
    pub fn n_layers(&self) -> usize {
        self.per_layer_param_bytes.len()
    }

    /// Activation bytes concurrently live per GPU during one training step.
    ///
    /// * **No checkpointing** (the default): a sequential model retains every
    ///   layer's activations from its forward until its backward runs, so at
    ///   the start of the backward pass the live set is the SUM over layers —
    ///   not the single-layer max the evaluator previously charged (which
    ///   understated activation memory by ~n_layers× and let the planner
    ///   declare infeasible configs feasible).
    /// * **With `@checkpoint(policy=...)` active**: sqrt-style estimate.
    ///   Checkpointing retains ~⌈√L⌉ segment-boundary activations and
    ///   rematerializes one segment at a time, so we charge
    ///   `(⌈√L⌉ + 1) × max_layer_bytes`, clamped to the no-checkpoint sum.
    ///   This is an APPROXIMATION: the planner only has per-layer byte
    ///   totals here (no segment map, no per-policy granularity), so both
    ///   the boundary set and the rematerialized segment are bounded by the
    ///   largest layer. Do not read more precision into it than that.
    pub fn retained_activation_bytes(&self) -> u64 {
        let total = self.total_activation_bytes();
        if !self.activation_checkpointing {
            return total;
        }
        // Layer count from the ACTIVATION vector (not n_layers(), which
        // reads the param vector) so a hand-built ModelSize with mismatched
        // vector lengths cannot silently use the wrong L here.
        let n_act_layers = self.per_layer_activation_bytes.len().max(1);
        let boundaries = (n_act_layers as f64).sqrt().ceil() as u64 + 1;
        boundaries
            .saturating_mul(self.peak_activation_bytes())
            .min(total)
    }

    /// Build a `ModelSize` from an `AppliedPlan`. Sums per-layer bytes
    /// directly from `AppliedLayer.param_bytes` / `activation_bytes` and
    /// uses `estimated_us` as the compute-time proxy. `optim_state_multiplier`
    /// defaults to 2.0 (fp32 AdamW m+v = 2 × param BYTES — see
    /// [`ADAMW_FP32_OPTIM_MULTIPLIER`]). `activation_checkpointing` defaults
    /// to `false`; the caller that knows the compile's checkpoint policies
    /// (`invoke_cpdt_if_enabled`) sets it after construction.
    pub fn from_applied_plan(plan: &crate::wggo_apply::AppliedPlan) -> Self {
        Self {
            per_layer_param_bytes: plan.layers.iter().map(|l| l.param_bytes).collect(),
            per_layer_activation_bytes: plan.layers.iter().map(|l| l.activation_bytes).collect(),
            per_layer_compute_us: plan.layers.iter().map(|l| l.estimated_us).collect(),
            optim_state_multiplier: ADAMW_FP32_OPTIM_MULTIPLIER,
            activation_checkpointing: false,
        }
    }
}

/// ZeRO configuration for one training step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct ZeroConfig {
    /// Parameter shard factor.
    pub s_p: u32,
    /// Gradient shard factor.
    pub s_g: u32,
    /// Optimizer-state shard factor.
    pub s_os: u32,
    /// Mesh for params: 0 = intra-node, 1 = inter-node (two-level mesh).
    pub mesh_p_inter: bool,
    pub mesh_g_inter: bool,
    pub mesh_os_inter: bool,
}

impl ZeroConfig {
    /// Standard ZeRO-1 / ZeRO-2 / ZeRO-3 shorthand.
    pub fn zero_1(num_gpus: u32) -> Self {
        Self {
            s_p: 1,
            s_g: 1,
            s_os: num_gpus,
            mesh_p_inter: false,
            mesh_g_inter: false,
            mesh_os_inter: false,
        }
    }
    pub fn zero_2(num_gpus: u32) -> Self {
        Self {
            s_p: 1,
            s_g: num_gpus,
            s_os: num_gpus,
            mesh_p_inter: false,
            mesh_g_inter: false,
            mesh_os_inter: false,
        }
    }
    pub fn zero_3(num_gpus: u32) -> Self {
        Self {
            s_p: num_gpus,
            s_g: num_gpus,
            s_os: num_gpus,
            mesh_p_inter: false,
            mesh_g_inter: false,
            mesh_os_inter: false,
        }
    }

    /// Convenience classifier: "ZeRO-1 equivalent", "ZeRO-2 equivalent",
    /// or "mixed" based on which shards are active.
    pub fn stage_label(self) -> &'static str {
        match (self.s_p > 1, self.s_g > 1, self.s_os > 1) {
            (false, false, true) => "ZeRO-1",
            (false, true, true) => "ZeRO-2",
            (true, true, true) => "ZeRO-3",
            (false, false, false) => "DDP",
            _ => "Mixed",
        }
    }
}

/// Per-candidate evaluation result.
#[derive(Debug, Clone)]
pub struct ZeroEvaluation {
    pub config: ZeroConfig,
    pub memory_per_gpu_bytes: u64,
    pub param_bytes_per_gpu: u64,
    pub grad_bytes_per_gpu: u64,
    pub optim_bytes_per_gpu: u64,
    pub activation_bytes_per_gpu: u64,
    pub comm_volume_bytes: u64,
    pub exposed_comm_us: f64,
    pub step_time_us: f64,
    pub feasible: bool,
}

/// Full search result.
#[derive(Debug, Clone)]
pub struct ZeroSearchResult {
    pub best: Option<ZeroEvaluation>,
    pub ranked: Vec<ZeroEvaluation>,
}

impl ZeroSearchResult {
    pub fn num_feasible(&self) -> usize {
        self.ranked.iter().filter(|e| e.feasible).count()
    }
}

// ---------------------------------------------------------------------------
// Cost model
// ---------------------------------------------------------------------------

/// Effective bandwidth depends on which mesh this shard group lives on.
fn bandwidth_bps(cluster: &ClusterSpec, inter: bool) -> f64 {
    if inter {
        cluster.inter_bw_bps
    } else {
        cluster.intra_bw_bps
    }
}

/// All-gather latency for `n` shards × `bytes_per_shard` on the given mesh.
///
/// Ring-allgather volume: `(n - 1) × bytes_per_shard`.
fn allgather_us(n: u32, bytes_per_shard: u64, bw: f64) -> f64 {
    if n <= 1 || bw <= 0.0 {
        return 0.0;
    }
    let volume = (n as u64 - 1) * bytes_per_shard;
    (volume as f64 / bw) * 1e6
}

/// Reduce-scatter latency: same volume as allgather for ring algorithm.
fn reducescatter_us(n: u32, bytes: u64, bw: f64) -> f64 {
    allgather_us(n, bytes, bw)
}

/// Single-config evaluator.  Pure function of inputs; O(n_layers) work.
pub fn evaluate_config(
    model: &ModelSize,
    cluster: &ClusterSpec,
    config: ZeroConfig,
) -> ZeroEvaluation {
    let dtype_factor = 1.0; // already folded into param_bytes
    let total_params = model.total_param_bytes();

    let param_per_gpu = (total_params / config.s_p as u64).max(1);
    let grad_per_gpu = (total_params / config.s_g as u64).max(1);
    let optim_per_gpu = ((total_params as f64 * model.optim_state_multiplier
        / config.s_os as f64) as u64)
        .max(1);
    // Concurrent live activation set (sum for a non-checkpointed sequential
    // model; checkpoint-aware estimate otherwise) — NOT the single-layer max.
    let activation_per_gpu = model.retained_activation_bytes();
    let memory = param_per_gpu + grad_per_gpu + optim_per_gpu + activation_per_gpu;
    let feasible = memory <= cluster.memory_budget_bytes;

    // Communication volumes.
    let p_bw = bandwidth_bps(cluster, config.mesh_p_inter);
    let g_bw = bandwidth_bps(cluster, config.mesh_g_inter);
    let os_bw = bandwidth_bps(cluster, config.mesh_os_inter);

    let fwd_allgather_us = allgather_us(config.s_p, param_per_gpu, p_bw);
    let bwd_reducescatter_us = reducescatter_us(config.s_g, grad_per_gpu, g_bw);
    // Post-step parameter all-gather: the optimizer step updates only the
    // 1/s_os slice of parameters whose optimizer state this rank owns, so the
    // updated parameters are re-gathered across the s_os-way optimizer-shard
    // group before the next forward (this is exactly the collective
    // `cpdt_comm::build_schedule` emits in its optimizer pass: AllGather,
    // group_size = s_os, per layer). The shard each rank contributes is
    // total/s_os — NOT total/s_p, which the pre-fix code passed and which
    // over-counted the gather s_os/s_p× (8× for ZeRO-1 on 8 GPUs).
    //
    // ZeRO stage notes: ZeRO-1/2 (s_p == 1) — this is the textbook post-step
    // gather back to fully-replicated parameters. ZeRO-3 (s_p == s_os) — the
    // scheduler currently emits this gather too, so the cost model charges it
    // for consistency; arguably it is redundant there (parameters are stored
    // sharded and the forward-prologue all-gather re-collects layers on
    // demand). TODO(cpdt): deciding to skip it at s_p == s_os is a protocol
    // decision (keep parameters sharded between steps) that is not encoded
    // anywhere — change `build_schedule` and this term together if taken.
    let optim_shard_bytes = (total_params / config.s_os as u64).max(1);
    let poststep_param_allgather_us = allgather_us(config.s_os, optim_shard_bytes, os_bw);

    // Overlap efficiency: compute-bound layers mask communication.
    let compute_total_us: f64 = model.per_layer_compute_us.iter().sum();
    let comm_total_us = fwd_allgather_us + bwd_reducescatter_us + poststep_param_allgather_us;
    let exposed = (comm_total_us - compute_total_us).max(0.0);

    let step_time_us = compute_total_us + exposed;

    let comm_volume_bytes =
        (config.s_p as u64 - 1).saturating_mul(param_per_gpu) * dtype_factor as u64
            + (config.s_g as u64 - 1).saturating_mul(grad_per_gpu)
            + (config.s_os as u64 - 1).saturating_mul(optim_shard_bytes);

    ZeroEvaluation {
        config,
        memory_per_gpu_bytes: memory,
        param_bytes_per_gpu: param_per_gpu,
        grad_bytes_per_gpu: grad_per_gpu,
        optim_bytes_per_gpu: optim_per_gpu,
        activation_bytes_per_gpu: activation_per_gpu,
        comm_volume_bytes,
        exposed_comm_us: exposed,
        step_time_us,
        feasible,
    }
}

/// Candidate-generator: cross-product of `(s_p, s_g, s_os)` where each
/// factor divides `num_gpus`, with the invariant `s_p ≤ s_g ≤ s_os`
/// (ZeRO's monotonicity property).
///
/// The mesh flag (intra-vs-inter) defaults to intra-node until the
/// shard factor exceeds `gpus_per_node`, at which point inter-node is
/// required.
fn generate_candidates(cluster: &ClusterSpec) -> Vec<ZeroConfig> {
    let n = cluster.num_gpus.max(1);
    let divisors: Vec<u32> = (1..=n).filter(|d| n.is_multiple_of(*d)).collect();
    let mut out = Vec::new();
    for &s_p in &divisors {
        for &s_g in &divisors {
            if s_g < s_p {
                continue;
            }
            for &s_os in &divisors {
                if s_os < s_g {
                    continue;
                }
                let cfg = ZeroConfig {
                    s_p,
                    s_g,
                    s_os,
                    mesh_p_inter: s_p > cluster.gpus_per_node,
                    mesh_g_inter: s_g > cluster.gpus_per_node,
                    mesh_os_inter: s_os > cluster.gpus_per_node,
                };
                out.push(cfg);
            }
        }
    }
    out
}

/// Deterministic total order for ranking candidate evaluations.
///
/// Step-time ties are COMMON, not exotic: whenever compute fully hides
/// communication, `step_time_us == compute_total_us` for every such config,
/// and the pre-fix sort left the winner to candidate-generation order.
/// Tie-break chain (each rung only consulted when the previous ties):
///
///   1. feasible before infeasible;
///   2. lowest modeled step time;
///   3. lowest per-GPU peak memory;
///   4. lowest communication volume;
///   5. least aggressive sharding — lexicographic `(s_p, s_g, s_os)`
///      ascending — as the final determinism backstop.
fn rank_cmp(a: &ZeroEvaluation, b: &ZeroEvaluation) -> std::cmp::Ordering {
    b.feasible
        .cmp(&a.feasible)
        // total_cmp, not partial_cmp: a NaN step time (conceivable from a
        // NaN estimated_us upstream) must not break the strict weak order
        // sort_by requires — NaN sorts after every finite time.
        .then(a.step_time_us.total_cmp(&b.step_time_us))
        .then(a.memory_per_gpu_bytes.cmp(&b.memory_per_gpu_bytes))
        .then(a.comm_volume_bytes.cmp(&b.comm_volume_bytes))
        .then(
            (a.config.s_p, a.config.s_g, a.config.s_os)
                .cmp(&(b.config.s_p, b.config.s_g, b.config.s_os)),
        )
}

/// Run the full ZeRO-config search.
pub fn search(model: &ModelSize, cluster: &ClusterSpec) -> ZeroSearchResult {
    let mut evaluated: Vec<ZeroEvaluation> = generate_candidates(cluster)
        .into_iter()
        .map(|cfg| evaluate_config(model, cluster, cfg))
        .collect();

    // Rank by the deterministic total order documented on `rank_cmp`.
    evaluated.sort_by(rank_cmp);
    let best = evaluated.iter().find(|e| e.feasible).cloned();
    ZeroSearchResult {
        best,
        ranked: evaluated,
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
            per_layer_param_bytes: vec![6_000_000; 8], // 48 MB total
            per_layer_activation_bytes: vec![2_000_000; 8], // 16 MB retained (sum)
            // Cost-model audit: 2.0 = fp32 AdamW m+v on param BYTES
            // (deliberately updated from the old 8.0 per-element figure).
            optim_state_multiplier: ADAMW_FP32_OPTIM_MULTIPLIER,
            per_layer_compute_us: vec![10.0; 8],
            activation_checkpointing: false,
        }
    }

    #[test]
    fn model_size_from_applied_plan_sums_per_layer_bytes() {
        use crate::wggo_apply::{AppliedLayer, AppliedPlan};
        use crate::wggo_dp::CoarseDecision;

        let plan = AppliedPlan {
            layers: vec![
                AppliedLayer {
                    layer_index: 0,
                    layer_name: "blocks.0".into(),
                    coarse: CoarseDecision::KeepFull,
                    pipeline_stage: 0,
                    shard_factor: 1,
                    shard_grads: 1,
                    shard_optim: 1,
                    active_heads: 8,
                    ffn_width: 4096,
                    csha_level: 0,
                    adapter_rank: 0,
                    adapter_placement: crate::wggo_ilp::AdapterPlacement::None,
                    optim_m_bits: 32,
                    optim_v_bits: 32,
                    fase_fused: false,
                    packing_mode: 0,
                    estimated_us: 12.5,
                    param_bytes: 6_000_000,
                    activation_bytes: 2_000_000,
                },
                AppliedLayer {
                    layer_index: 1,
                    layer_name: "blocks.1".into(),
                    coarse: CoarseDecision::KeepFull,
                    pipeline_stage: 0,
                    shard_factor: 1,
                    shard_grads: 1,
                    shard_optim: 1,
                    active_heads: 8,
                    ffn_width: 4096,
                    csha_level: 0,
                    adapter_rank: 0,
                    adapter_placement: crate::wggo_ilp::AdapterPlacement::None,
                    optim_m_bits: 32,
                    optim_v_bits: 32,
                    fase_fused: false,
                    packing_mode: 0,
                    estimated_us: 14.0,
                    param_bytes: 8_000_000,
                    activation_bytes: 3_000_000,
                },
            ],
            total_us: 26.5,
            peak_memory_bytes: 0,
        };

        let ms = ModelSize::from_applied_plan(&plan);
        assert_eq!(ms.per_layer_param_bytes, vec![6_000_000, 8_000_000]);
        assert_eq!(ms.per_layer_activation_bytes, vec![2_000_000, 3_000_000]);
        assert_eq!(ms.per_layer_compute_us, vec![12.5, 14.0]);
        // Cost-model audit: deliberately updated from 8.0 — the multiplier
        // applies to param BYTES, so fp32 AdamW m+v is 2× (see the constant).
        assert!((ms.optim_state_multiplier - 2.0).abs() < 1e-9);
        assert!(!ms.activation_checkpointing, "from_applied_plan defaults to no checkpointing");
    }

    #[test]
    fn optim_bytes_are_twice_param_bytes_for_fp32_adamw() {
        // 48 MB fp32 params → m+v = 96 MB, NOT the pre-audit 384 MB
        // (8.0 was the per-ELEMENT byte count applied to a bytes quantity).
        let model = tiny_model();
        let cluster = ClusterSpec::default();
        let ddp = evaluate_config(
            &model,
            &cluster,
            ZeroConfig {
                s_p: 1,
                s_g: 1,
                s_os: 1,
                mesh_p_inter: false,
                mesh_g_inter: false,
                mesh_os_inter: false,
            },
        );
        assert_eq!(ddp.optim_bytes_per_gpu, 2 * model.total_param_bytes());
    }

    #[test]
    fn retained_activations_sum_without_checkpointing() {
        // Non-checkpointed sequential model: all 8 layers' activations are
        // live at the start of backward → the evaluator must charge the SUM
        // (16 MB), not the single-layer max (2 MB) it used pre-audit.
        let model = tiny_model();
        let cluster = ClusterSpec::default();
        let eval = evaluate_config(&model, &cluster, ZeroConfig::zero_1(8));
        assert_eq!(eval.activation_bytes_per_gpu, 16_000_000);
    }

    #[test]
    fn retained_activations_checkpoint_estimate_below_sum_and_above_peak() {
        let mut model = tiny_model();
        model.activation_checkpointing = true;
        // 8 layers → ⌈√8⌉ + 1 = 4 boundaries × 2 MB = 8 MB; clamped to the
        // 16 MB sum. Documented approximation, not a precise live-set.
        let retained = model.retained_activation_bytes();
        assert_eq!(retained, 8_000_000);
        assert!(retained < model.total_activation_bytes());
        assert!(retained >= model.peak_activation_bytes());
    }

    #[test]
    fn poststep_allgather_volume_uses_optimizer_shard_size() {
        // ZeRO-1 (s_p=1, s_g=1, s_os=8): the only sharded collective is the
        // post-step parameter all-gather over the s_os group. Ring volume is
        // (s_os - 1) × total/s_os = 7 × 6 MB = 42 MB. The pre-fix code used
        // param_per_gpu = total/s_p = 48 MB per shard → 336 MB (8× over).
        let model = tiny_model();
        let cluster = ClusterSpec::default();
        let eval = evaluate_config(&model, &cluster, ZeroConfig::zero_1(8));
        assert_eq!(eval.comm_volume_bytes, 7 * 6_000_000);
    }

    #[test]
    fn shorthand_configs_match_stage_labels() {
        assert_eq!(ZeroConfig::zero_1(8).stage_label(), "ZeRO-1");
        assert_eq!(ZeroConfig::zero_2(8).stage_label(), "ZeRO-2");
        assert_eq!(ZeroConfig::zero_3(8).stage_label(), "ZeRO-3");
    }

    #[test]
    fn evaluate_config_memory_accounts_for_sharding() {
        let model = tiny_model();
        let cluster = ClusterSpec::default();
        let ddp = evaluate_config(
            &model,
            &cluster,
            ZeroConfig {
                s_p: 1,
                s_g: 1,
                s_os: 1,
                mesh_p_inter: false,
                mesh_g_inter: false,
                mesh_os_inter: false,
            },
        );
        let z3 = evaluate_config(&model, &cluster, ZeroConfig::zero_3(8));
        // ZeRO-3 should use substantially less per-GPU memory.
        assert!(z3.memory_per_gpu_bytes < ddp.memory_per_gpu_bytes);
    }

    #[test]
    fn search_picks_a_feasible_config() {
        let model = tiny_model();
        let cluster = ClusterSpec::default();
        let result = search(&model, &cluster);
        let best = result.best.as_ref().expect("at least one feasible config");
        assert!(best.feasible);
        assert!(result.num_feasible() > 0);
    }

    #[test]
    fn tight_memory_budget_forces_high_sharding() {
        let model = tiny_model();
        let mut cluster = ClusterSpec::default();
        cluster.memory_budget_bytes = 150_000_000; // Well below DDP footprint.
        let result = search(&model, &cluster);
        let best = result.best.expect("should still find a feasible config");
        // The winning config must shard at least one component to fit.
        assert!(best.config.s_p > 1 || best.config.s_g > 1 || best.config.s_os > 1);
    }

    #[test]
    fn comm_volume_is_zero_when_no_sharding() {
        let model = tiny_model();
        let cluster = ClusterSpec::default();
        let cfg = ZeroConfig {
            s_p: 1,
            s_g: 1,
            s_os: 1,
            mesh_p_inter: false,
            mesh_g_inter: false,
            mesh_os_inter: false,
        };
        let eval = evaluate_config(&model, &cluster, cfg);
        assert_eq!(eval.comm_volume_bytes, 0);
        assert_eq!(eval.exposed_comm_us, 0.0);
    }

    #[test]
    fn candidates_respect_monotonicity() {
        let cluster = ClusterSpec::default();
        let candidates = generate_candidates(&cluster);
        for cfg in &candidates {
            assert!(cfg.s_p <= cfg.s_g);
            assert!(cfg.s_g <= cfg.s_os);
        }
    }

    #[test]
    fn inter_node_mesh_flag_flips_on_large_shards() {
        let mut cluster = ClusterSpec::default();
        cluster.num_gpus = 16;
        cluster.gpus_per_node = 8;
        let candidates = generate_candidates(&cluster);
        // At s_p=16 we must be inter-node.
        let some_16 = candidates.iter().find(|c| c.s_p == 16).unwrap();
        assert!(some_16.mesh_p_inter);
        // At s_p=8 we're still intra.
        let some_8 = candidates.iter().find(|c| c.s_p == 8).unwrap();
        assert!(!some_8.mesh_p_inter);
    }

    #[test]
    fn search_ranks_feasible_before_infeasible() {
        let model = tiny_model();
        let mut cluster = ClusterSpec::default();
        cluster.memory_budget_bytes = 500_000_000;
        let result = search(&model, &cluster);
        let mut seen_infeasible = false;
        for eval in &result.ranked {
            if !eval.feasible {
                seen_infeasible = true;
            } else {
                // Once we start seeing infeasible, we should never see
                // feasible again.
                assert!(!seen_infeasible);
            }
        }
    }

    #[test]
    fn exposed_comm_is_zero_when_compute_covers_comm() {
        let mut model = tiny_model();
        // Huge compute budget → comm fully hidden.
        for t in model.per_layer_compute_us.iter_mut() {
            *t = 1_000_000.0;
        }
        let cluster = ClusterSpec::default();
        let eval = evaluate_config(&model, &cluster, ZeroConfig::zero_3(8));
        assert_eq!(eval.exposed_comm_us, 0.0);
    }

    #[test]
    fn step_time_tie_broken_by_lower_memory() {
        // Two feasible candidates with identical step time: the deterministic
        // rank order must prefer the lower-memory one (rung 3 of rank_cmp).
        let mk = |memory: u64, s_os: u32| ZeroEvaluation {
            config: ZeroConfig {
                s_p: 1,
                s_g: 1,
                s_os,
                mesh_p_inter: false,
                mesh_g_inter: false,
                mesh_os_inter: false,
            },
            memory_per_gpu_bytes: memory,
            param_bytes_per_gpu: 48_000_000,
            grad_bytes_per_gpu: 48_000_000,
            optim_bytes_per_gpu: 96_000_000,
            activation_bytes_per_gpu: 16_000_000,
            comm_volume_bytes: 42_000_000,
            exposed_comm_us: 0.0,
            step_time_us: 80.0,
            feasible: true,
        };
        let high_mem = mk(300_000_000, 4);
        let low_mem = mk(200_000_000, 8);
        assert_eq!(rank_cmp(&low_mem, &high_mem), std::cmp::Ordering::Less);
        assert_eq!(rank_cmp(&high_mem, &low_mem), std::cmp::Ordering::Greater);

        // And an infeasible candidate always ranks after a feasible one,
        // regardless of a better step time.
        let mut infeasible_fast = mk(100_000_000, 2);
        infeasible_fast.feasible = false;
        infeasible_fast.step_time_us = 1.0;
        assert_eq!(rank_cmp(&high_mem, &infeasible_fast), std::cmp::Ordering::Less);
    }

    #[test]
    fn search_breaks_hidden_comm_ties_toward_lowest_memory() {
        // Huge per-layer compute hides ALL communication, so every candidate
        // ties on step_time_us == total compute. The winner must then be the
        // minimum-memory config — full (s_p, s_g, s_os) = (8, 8, 8) sharding
        // for this model — and stay stable across runs.
        let mut model = tiny_model();
        for t in model.per_layer_compute_us.iter_mut() {
            *t = 1_000_000.0;
        }
        let cluster = ClusterSpec::default();
        let result = search(&model, &cluster);
        let best = result.best.expect("feasible config expected");
        let min_mem = result
            .ranked
            .iter()
            .filter(|e| e.feasible)
            .map(|e| e.memory_per_gpu_bytes)
            .min()
            .unwrap();
        assert_eq!(best.memory_per_gpu_bytes, min_mem);
        assert_eq!(
            (best.config.s_p, best.config.s_g, best.config.s_os),
            (8, 8, 8),
            "hidden-comm tie must resolve to the lowest-memory candidate"
        );
    }

    #[test]
    fn ddp_has_no_comm() {
        let model = tiny_model();
        let cluster = ClusterSpec::default();
        let cfg = ZeroConfig {
            s_p: 1,
            s_g: 1,
            s_os: 1,
            mesh_p_inter: false,
            mesh_g_inter: false,
            mesh_os_inter: false,
        };
        let e = evaluate_config(&model, &cluster, cfg);
        assert_eq!(e.stage_label_of_config(), "DDP");
    }

    impl ZeroEvaluation {
        fn stage_label_of_config(&self) -> &'static str {
            self.config.stage_label()
        }
    }
}
