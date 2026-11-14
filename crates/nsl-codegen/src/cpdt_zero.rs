//! CPDT — analytical ZeRO-config planner.
//!
//! Enumerates the `(s_p, s_g, s_os)` cross-product of shard factors and
//! ranks feasible configurations by analytical step time. Every per-config
//! evaluation is a closed-form cost-model call (paper §3.1).
//!
//! No runtime profiling; no external solver. The result is advisory: it is
//! printed by `--cpdt-report` and read by nothing else. The stage a
//! compiled program runs comes from `--zero-stage`
//! ([`crate::cpdt::CpdtPlan::render_report`] says so).
//!
//! ## Communication model
//!
//! Volumes are bytes sent per rank per optimizer step, by ring collectives
//! over all `N` data-parallel ranks, for a model of `Ψ` bytes
//! ([`comm_per_step`]):
//!
//! | Configuration | Collectives | Volume |
//! |---|---|---|
//! | plain data-parallel | gradient all-reduce | `2·Ψ·(N−1)/N` |
//! | optimizer state sharded | gradient all-reduce, parameter broadcast | `3·Ψ·(N−1)/N` |
//! | + gradients sharded | gradient reduce-scatter, parameter broadcast | `2·Ψ·(N−1)/N` |
//! | + parameters sharded | two parameter all-gathers and a gradient reduce-scatter, per micro-batch | `3·M·Ψ·(N−1)/N` |
//!
//! The first, third and fourth rows are the ZeRO paper's `2Ψ`, `2Ψ` and
//! `3Ψ`; the second is what the runtime in `nsl-runtime/src/zero.rs` does at
//! stage 1 (an all-reduce, then each owner broadcasts its updated
//! parameters). Each collective also costs a fixed latency
//! ([`ClusterSpec::collective_latency_us`]) per parameter group, and all of
//! it is added to the step: the runtime's collectives are synchronous, so
//! none of the time is hidden under compute.
//!
//! The earlier model charged plain data-parallel training nothing, left out
//! the second all-gather, the micro-batch factor and the latency, and
//! counted communication only where it exceeded the whole step's compute
//! time; with communication thus hidden every configuration tied on time
//! and the tie went to the least memory, the most sharding.

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
    /// Fixed cost of launching one collective, in microseconds, whatever
    /// its size. An assumed order of magnitude
    /// ([`DEFAULT_COLLECTIVE_LATENCY_US`]), not a measurement of any
    /// cluster.
    pub collective_latency_us: f64,
}

/// Per-GPU memory the planner assumes when the user gives none
/// (`--cpdt-memory-budget`): 80 GiB, the capacity of an A100-80G or H100.
pub const DEFAULT_MEMORY_BUDGET_BYTES: u64 = 80u64 * 1024 * 1024 * 1024;

/// Assumed fixed latency of one collective. Collective launch latencies are
/// tens of microseconds on a single node and more across nodes; this is a
/// round figure of that order so that a plan of many small collectives is
/// not costed as free, not a measured value.
pub const DEFAULT_COLLECTIVE_LATENCY_US: f64 = 50.0;

impl Default for ClusterSpec {
    fn default() -> Self {
        Self {
            num_gpus: 8,
            memory_budget_bytes: DEFAULT_MEMORY_BUDGET_BYTES,
            intra_bw_bps: 9e11,
            inter_bw_bps: 1e11,
            gpus_per_node: 8,
            collective_latency_us: DEFAULT_COLLECTIVE_LATENCY_US,
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
    /// Micro-batches per optimizer step (`grad_accumulation`). The compute
    /// times above are per micro-batch, and with sharded parameters every
    /// micro-batch gathers them again.
    pub micro_batches_per_step: u32,
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
    /// to `false` and `micro_batches_per_step` to 1; the caller that knows
    /// the compile's checkpoint policies and the train block's
    /// `grad_accumulation` (`invoke_cpdt_if_enabled`) sets them after
    /// construction.
    ///
    /// The plan's bytes are those of the configuration WGGO chose for each
    /// layer of the model being compiled (`wggo_apply::apply`).
    pub fn from_applied_plan(plan: &crate::wggo_apply::AppliedPlan) -> Self {
        Self {
            per_layer_param_bytes: plan.layers.iter().map(|l| l.param_bytes).collect(),
            per_layer_activation_bytes: plan.layers.iter().map(|l| l.activation_bytes).collect(),
            per_layer_compute_us: plan.layers.iter().map(|l| l.estimated_us).collect(),
            optim_state_multiplier: ADAMW_FP32_OPTIM_MULTIPLIER,
            activation_checkpointing: false,
            micro_batches_per_step: 1,
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
    /// Bytes each rank sends per optimizer step ([`comm_per_step`]).
    pub comm_volume_bytes: u64,
    /// Communication time on the step: transfer plus per-collective
    /// latency. All of it, since the runtime does not overlap collectives
    /// with compute.
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

/// Communication of one optimizer step under a configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StepComm {
    /// Bytes each rank sends, summed over the step's collectives.
    pub volume_bytes: u64,
    /// Collectives launched per parameter group.
    pub collectives: u32,
}

/// Volume and collective count of one optimizer step (see the module
/// header for the table this implements).
///
/// `total_param_bytes` is the model's `Ψ`, `n` the number of data-parallel
/// ranks, `micro_batches` the micro-batches per optimizer step. On one rank
/// nothing is sent.
pub fn comm_per_step(
    total_param_bytes: u64,
    n: u32,
    config: ZeroConfig,
    micro_batches: u32,
) -> StepComm {
    if n <= 1 {
        return StepComm { volume_bytes: 0, collectives: 0 };
    }
    // One ring pass over a tensor of Ψ bytes moves Ψ·(N−1)/N per rank; an
    // all-reduce is two passes (reduce, then distribute).
    let pass = (total_param_bytes as u128 * (n as u128 - 1) / n as u128) as u64;
    let m = micro_batches.max(1);
    let (passes, collectives) = if config.s_p > 1 {
        // Parameters live sharded: gather them for the forward and again
        // for the backward, reduce-scatter the gradients, every micro-batch.
        (3 * m, 3 * m)
    } else {
        // Gradients: reduce-scatter when sharded, else a full all-reduce.
        let grad_passes = if config.s_g > 1 { 1 } else { 2 };
        // Sharded optimizer state: each owner updates its slice and the
        // updated parameters go back out to every rank.
        let sync = u32::from(config.s_os > 1);
        (grad_passes + sync, 1 + sync)
    };
    StepComm {
        volume_bytes: pass.saturating_mul(u64::from(passes)),
        collectives,
    }
}

/// Single-config evaluator.  Pure function of inputs; O(n_layers) work.
pub fn evaluate_config(
    model: &ModelSize,
    cluster: &ClusterSpec,
    config: ZeroConfig,
) -> ZeroEvaluation {
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

    // Every collective spans all the data-parallel ranks, so it crosses
    // nodes as soon as the cluster does.
    let n = cluster.num_gpus.max(1);
    let bw = if n > cluster.gpus_per_node.max(1) {
        cluster.inter_bw_bps
    } else {
        cluster.intra_bw_bps
    };
    let micro_batches = model.micro_batches_per_step.max(1);
    let comm = comm_per_step(total_params, n, config, micro_batches);
    let transfer_us = if bw > 0.0 {
        comm.volume_bytes as f64 / bw * 1e6
    } else {
        0.0
    };
    // One collective per parameter group (layer) per pass kind.
    let groups = model.n_layers().max(1) as f64;
    let latency_us = comm.collectives as f64 * groups * cluster.collective_latency_us.max(0.0);
    let comm_total_us = transfer_us + latency_us;

    // The runtime's collectives are synchronous: all of the communication
    // time is on the step.
    let compute_total_us: f64 =
        model.per_layer_compute_us.iter().sum::<f64>() * micro_batches as f64;
    let step_time_us = compute_total_us + comm_total_us;

    ZeroEvaluation {
        config,
        memory_per_gpu_bytes: memory,
        param_bytes_per_gpu: param_per_gpu,
        grad_bytes_per_gpu: grad_per_gpu,
        optim_bytes_per_gpu: optim_per_gpu,
        activation_bytes_per_gpu: activation_per_gpu,
        comm_volume_bytes: comm.volume_bytes,
        exposed_comm_us: comm_total_us,
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
/// Step-time ties happen between configurations with the same collectives
/// (for example `(1, 2, 8)` and `(1, 4, 8)`), and on a single GPU, where
/// nothing is sent.
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
            micro_batches_per_step: 1,
        }
    }

    fn cfg(s_p: u32, s_g: u32, s_os: u32) -> ZeroConfig {
        ZeroConfig {
            s_p,
            s_g,
            s_os,
            mesh_p_inter: false,
            mesh_g_inter: false,
            mesh_os_inter: false,
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

    /// The communication volumes of the ZeRO paper (arXiv 1910.02054,
    /// section 7), which states them for a model of Ψ bytes with the
    /// `(N−1)/N` ring factor taken as 1: data-parallel 2Ψ, `P_os+g` 2Ψ,
    /// `P_os+g+p` 3Ψ.
    ///
    /// Here Ψ = 48 MB and N = 8, so one ring pass is 48 MB × 7/8 = 42 MB:
    /// 84 MB, 84 MB and 126 MB. Stage 1 as this runtime does it (all-reduce
    /// then a parameter broadcast) is three passes, 126 MB.
    #[test]
    fn comm_volume_matches_the_zero_paper() {
        const PASS: u64 = 42_000_000;
        let psi = tiny_model().total_param_bytes();
        assert_eq!(psi, 48_000_000);
        let volume = |c: ZeroConfig| comm_per_step(psi, 8, c, 1).volume_bytes;
        assert_eq!(volume(cfg(1, 1, 1)), 2 * PASS, "data-parallel: 2 psi");
        assert_eq!(volume(ZeroConfig::zero_2(8)), 2 * PASS, "P_os+g: 2 psi");
        assert_eq!(volume(ZeroConfig::zero_3(8)), 3 * PASS, "P_os+g+p: 3 psi");
        assert_eq!(volume(ZeroConfig::zero_1(8)), 3 * PASS, "all-reduce + broadcast");
    }

    /// Plain data-parallel training all-reduces its gradients. It was
    /// charged nothing, which made every sharded configuration look no
    /// slower than it.
    #[test]
    fn data_parallel_training_pays_for_its_all_reduce() {
        let model = tiny_model();
        let cluster = ClusterSpec::default();
        let eval = evaluate_config(&model, &cluster, cfg(1, 1, 1));
        assert_eq!(eval.comm_volume_bytes, 84_000_000);
        // 84 MB at 9e11 B/s = 93.33 us, plus one collective for each of the
        // 8 parameter groups at 50 us = 400 us; compute is 8 x 10 us.
        let transfer = 84_000_000.0 / 9e11 * 1e6;
        assert!((eval.exposed_comm_us - (transfer + 400.0)).abs() < 1e-6);
        assert!((eval.step_time_us - (80.0 + transfer + 400.0)).abs() < 1e-6);
    }

    /// With sharded parameters every micro-batch gathers them twice and
    /// reduce-scatters once; with replicated parameters the gradients are
    /// reduced once per optimizer step, however many micro-batches fed it.
    #[test]
    fn sharded_parameters_pay_per_micro_batch() {
        let psi = 48_000_000;
        assert_eq!(comm_per_step(psi, 8, ZeroConfig::zero_3(8), 4).volume_bytes, 12 * 42_000_000);
        assert_eq!(comm_per_step(psi, 8, ZeroConfig::zero_2(8), 4).volume_bytes, 2 * 42_000_000);
        assert_eq!(comm_per_step(psi, 8, cfg(1, 1, 1), 4).volume_bytes, 2 * 42_000_000);
    }

    #[test]
    fn one_rank_sends_nothing() {
        let c = comm_per_step(48_000_000, 1, cfg(1, 1, 1), 4);
        assert_eq!(c, StepComm { volume_bytes: 0, collectives: 0 });
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

    /// The runtime's collectives are synchronous, so compute hides none of
    /// the communication: making the layers a million times slower leaves
    /// the communication time on the step unchanged.
    #[test]
    fn compute_time_hides_no_communication() {
        let cluster = ClusterSpec::default();
        let fast = evaluate_config(&tiny_model(), &cluster, ZeroConfig::zero_3(8));
        let mut slow_model = tiny_model();
        for t in slow_model.per_layer_compute_us.iter_mut() {
            *t = 1_000_000.0;
        }
        let slow = evaluate_config(&slow_model, &cluster, ZeroConfig::zero_3(8));
        assert!(fast.exposed_comm_us > 0.0);
        assert_eq!(slow.exposed_comm_us, fast.exposed_comm_us);
        assert!((slow.step_time_us - (8_000_000.0 + slow.exposed_comm_us)).abs() < 1e-3);
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

    /// When everything fits, the planner recommends the configuration with
    /// the least communication, which is plain data-parallel training (one
    /// collective per group; gradient sharding moves the same bytes in two).
    /// It used to recommend full sharding for any model, because hidden
    /// communication made every configuration tie on time and the tie went
    /// to the least memory.
    #[test]
    fn an_unconstrained_search_does_not_shard() {
        let result = search(&tiny_model(), &ClusterSpec::default());
        let best = result.best.expect("feasible config expected");
        assert_eq!((best.config.s_p, best.config.s_g, best.config.s_os), (1, 1, 1));
        assert_eq!(best.config.stage_label(), "DDP");
    }

    /// Under a budget that plain data-parallel training does not fit, the
    /// cheapest configuration that fits wins: with these sizes that is
    /// gradient and optimizer sharding with replicated parameters, not
    /// parameter sharding, which costs a further pass per micro-batch.
    ///
    /// Memory per GPU is 48/s_p + 48/s_g + 96/s_os + 16 MB: 208 MB
    /// unsharded, 82 MB at (1, 8, 8), 40 MB at (8, 8, 8).
    #[test]
    fn a_tight_budget_shards_only_as_far_as_it_must() {
        let cluster = ClusterSpec {
            memory_budget_bytes: 100_000_000,
            ..ClusterSpec::default()
        };
        let best = search(&tiny_model(), &cluster).best.expect("a configuration fits");
        assert_eq!((best.config.s_p, best.config.s_g, best.config.s_os), (1, 8, 8));
        assert_eq!(best.memory_per_gpu_bytes, 82_000_000);
    }
}
