//! CPDT — MoE expert placement + dead-expert detection.
//!
//! Paper §5: three compile-time optimizations for MoE layers:
//!
//!   1. **Roofline-determined capacity factor** — pick the capacity
//!      factor that fills ALU slack.
//!   2. **Dead-expert detection** — prune experts whose router-weight
//!      affinity falls below a threshold.
//!   3. **Expert placement** — enumerate Balanced / ExpertParallel /
//!      Hybrid placements and rank by comm + load-balance + memory.

use serde::Serialize;

use crate::weight_aware::WeightEntry;

/// MoE layer shape descriptor.
#[derive(Debug, Clone, Copy)]
pub struct MoeLayerShape {
    pub n_experts: u32,
    pub top_k: u32,
    pub d_model: u32,
    pub d_expert: u32,
    pub batch: u32,
    pub seq: u32,
    pub dtype_bytes: u32,
}

impl MoeLayerShape {
    /// Parameters per expert in bytes.
    pub fn per_expert_bytes(&self) -> u64 {
        // Two matmuls: gate+up and down.
        2 * (self.d_model as u64) * (self.d_expert as u64) * self.dtype_bytes as u64
    }
}

/// Per-expert affinity score derived from the router matrix.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct ExpertAffinity {
    pub expert: u32,
    pub affinity: f64,
}

/// Placement options the planner considers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum PlacementStrategy {
    /// Each GPU owns `n_experts / num_gpus` experts (paper §5.3-A).
    Balanced,
    /// Expert-parallel: tokens are routed to the GPU that owns the
    /// expert (paper §5.3-B).
    ExpertParallel,
    /// Popular experts replicated, rare experts sharded (paper §5.3-C).
    Hybrid,
}

impl PlacementStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            PlacementStrategy::Balanced => "balanced",
            PlacementStrategy::ExpertParallel => "expert_parallel",
            PlacementStrategy::Hybrid => "hybrid",
        }
    }
}

/// Evaluation of one placement option.
#[derive(Debug, Clone, Serialize)]
pub struct PlacementEvaluation {
    pub strategy: PlacementStrategy,
    pub comm_volume_bytes: u64,
    pub load_balance: f64,
    pub memory_per_gpu_bytes: u64,
    pub step_time_us: f64,
    pub feasible: bool,
    pub rationale: String,
}

/// Aggregate expert plan.
#[derive(Debug, Clone, Serialize)]
pub struct ExpertPlan {
    pub capacity_factor: f64,
    pub dead_experts: Vec<u32>,
    pub affinities: Vec<ExpertAffinity>,
    pub placement: PlacementEvaluation,
    pub alternatives: Vec<PlacementEvaluation>,
}

impl ExpertPlan {
    pub fn active_experts(&self, total: u32) -> u32 {
        total.saturating_sub(self.dead_experts.len() as u32)
    }
}

/// Configuration for the planner.
#[derive(Debug, Clone)]
pub struct ExpertConfig {
    /// Router-affinity threshold below which experts are pruned.
    pub dead_expert_threshold: f64,
    /// Minimum load-balance score a placement must achieve.
    pub min_load_balance: f64,
    pub num_gpus: u32,
    pub intra_bw_bps: f64,
    pub per_gpu_memory_bytes: u64,
}

impl Default for ExpertConfig {
    fn default() -> Self {
        Self {
            dead_expert_threshold: 0.01,
            min_load_balance: 0.5,
            num_gpus: 8,
            intra_bw_bps: 9e11,
            per_gpu_memory_bytes: 80u64 * 1024 * 1024 * 1024,
        }
    }
}

// ---------------------------------------------------------------------------
// Router analysis
// ---------------------------------------------------------------------------

/// Compute per-expert router affinity from the router weight matrix.
///
/// `router_weights` is expected to be `[d_model, n_experts]` row-major.
pub fn router_affinities(router: &WeightEntry, n_experts: u32) -> Vec<ExpertAffinity> {
    if router.shape.len() != 2 {
        return Vec::new();
    }
    let rows = router.shape[0];
    let cols = router.shape[1];
    if cols != n_experts as usize {
        return Vec::new();
    }
    let bw = router.dtype.byte_width();
    let mut out = Vec::with_capacity(n_experts as usize);
    for expert in 0..n_experts as usize {
        let mut sum_sq = 0.0_f64;
        for r in 0..rows {
            let idx = r * cols + expert;
            let off = idx * bw;
            if off + bw > router.data.len() {
                sum_sq = 0.0;
                break;
            }
            let v = router.dtype.to_f64(&router.data[off..off + bw]);
            sum_sq += v * v;
        }
        out.push(ExpertAffinity {
            expert: expert as u32,
            affinity: sum_sq.sqrt(),
        });
    }
    out
}

/// Detect dead / near-dead experts.
pub fn detect_dead_experts(
    affinities: &[ExpertAffinity],
    threshold: f64,
) -> Vec<u32> {
    affinities
        .iter()
        .filter(|a| a.affinity < threshold)
        .map(|a| a.expert)
        .collect()
}

// ---------------------------------------------------------------------------
// Capacity factor
// ---------------------------------------------------------------------------

/// Pick the capacity factor from the roofline slack ratio.
///
/// `roofline_slack` is the ratio of available-to-used compute on the
/// expert FFN: `> 1` means memory-bound (idle ALUs → increase capacity),
/// `≤ 1` means compute-bound (keep capacity near 1).
pub fn capacity_factor(roofline_slack: f64, top_k: u32, n_experts: u32) -> f64 {
    if n_experts == 0 {
        return 1.0;
    }
    let raw = roofline_slack * (top_k as f64) / (n_experts as f64);
    raw.max(1.0)
}

// ---------------------------------------------------------------------------
// Placement evaluation
// ---------------------------------------------------------------------------

/// Bytes of token activations a step dispatches when every one of its
/// `tokens × top_k` assignments is sent to another device: one `d_model`
/// vector each.
fn full_dispatch_bytes(shape: &MoeLayerShape) -> u64 {
    (shape.batch as u64)
        .saturating_mul(shape.seq as u64)
        .saturating_mul(shape.top_k as u64)
        .saturating_mul(shape.d_model as u64)
        .saturating_mul(shape.dtype_bytes as u64)
}

fn dispatch_us(bytes: u64, cfg: &ExpertConfig) -> f64 {
    (bytes as f64 / cfg.intra_bw_bps.max(1.0)) * 1e6
}

/// Each GPU owns `n_experts / num_gpus` experts. A token on one GPU whose
/// expert lives on another has to be sent there, so with tokens and routing
/// spread evenly `(G − 1) / G` of the assignments are dispatched. (This
/// used to be charged the partitioned memory and no dispatch at all, which
/// no placement can deliver: it made Balanced free and therefore always the
/// winner.)
fn balanced_eval(shape: &MoeLayerShape, cfg: &ExpertConfig) -> PlacementEvaluation {
    let gpus = cfg.num_gpus.max(1);
    let experts_per_gpu = shape.n_experts.div_ceil(gpus);
    let memory_per_gpu = (experts_per_gpu as u64) * shape.per_expert_bytes();
    let comm_volume =
        (full_dispatch_bytes(shape) as u128 * (gpus as u128 - 1) / gpus as u128) as u64;
    let load_balance = 1.0; // every GPU owns exactly the same expert count
    PlacementEvaluation {
        strategy: PlacementStrategy::Balanced,
        comm_volume_bytes: comm_volume,
        load_balance,
        memory_per_gpu_bytes: memory_per_gpu,
        step_time_us: dispatch_us(comm_volume, cfg),
        feasible: memory_per_gpu <= cfg.per_gpu_memory_bytes,
        rationale: format!(
            "balanced: {experts_per_gpu} experts/GPU, tokens dispatched to the GPU holding \
             their expert"
        ),
    }
}

fn expert_parallel_eval(shape: &MoeLayerShape, cfg: &ExpertConfig) -> PlacementEvaluation {
    // In pure expert-parallel, each GPU hosts one expert group.  Tokens
    // must be dispatched to the owning GPU via all-to-all.
    let experts_per_gpu = shape.n_experts.div_ceil(cfg.num_gpus.max(1));
    let memory_per_gpu = (experts_per_gpu as u64) * shape.per_expert_bytes();
    // All-to-all volume per step: every token goes to top_k experts.
    let comm_volume = full_dispatch_bytes(shape);
    let load_balance = if cfg.num_gpus == 0 { 0.0 } else { 1.0 / cfg.num_gpus as f64 }; // pessimistic
    PlacementEvaluation {
        strategy: PlacementStrategy::ExpertParallel,
        comm_volume_bytes: comm_volume,
        load_balance,
        memory_per_gpu_bytes: memory_per_gpu,
        step_time_us: dispatch_us(comm_volume, cfg),
        feasible: memory_per_gpu <= cfg.per_gpu_memory_bytes,
        rationale: format!(
            "expert-parallel: all-to-all dispatch, {experts_per_gpu} experts/GPU"
        ),
    }
}

fn hybrid_eval(shape: &MoeLayerShape, cfg: &ExpertConfig) -> PlacementEvaluation {
    // Hybrid: popular half replicated, rest sharded.  Replicating the
    // hot half halves the all-to-all volume but doubles the memory
    // footprint for those experts.
    let replicated_count = (shape.n_experts / 2).max(1);
    let sharded_count = shape.n_experts - replicated_count;
    let memory_per_gpu = replicated_count as u64 * shape.per_expert_bytes()
        + sharded_count.div_ceil(cfg.num_gpus.max(1)) as u64 * shape.per_expert_bytes();
    // Half the assignments go to a replicated (local) expert; the other
    // half are dispatched. Halved on the byte total: halving `top_k` in
    // integer arithmetic, as this did, made the volume 0 for top-1 routing.
    let comm_volume = full_dispatch_bytes(shape) / 2;
    PlacementEvaluation {
        strategy: PlacementStrategy::Hybrid,
        comm_volume_bytes: comm_volume,
        load_balance: 0.75,
        memory_per_gpu_bytes: memory_per_gpu,
        step_time_us: dispatch_us(comm_volume, cfg),
        feasible: memory_per_gpu <= cfg.per_gpu_memory_bytes,
        rationale: format!(
            "hybrid: {replicated_count} replicated + {sharded_count} sharded experts, halved all-to-all"
        ),
    }
}

/// Rank placement strategies by step-time (feasible first).
pub fn plan_placement(shape: &MoeLayerShape, cfg: &ExpertConfig) -> PlacementEvaluation {
    let options = [
        balanced_eval(shape, cfg),
        expert_parallel_eval(shape, cfg),
        hybrid_eval(shape, cfg),
    ];
    options
        .into_iter()
        .filter(|o| o.feasible && o.load_balance >= cfg.min_load_balance)
        .min_by(|a, b| {
            a.step_time_us
                .partial_cmp(&b.step_time_us)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .unwrap_or_else(|| balanced_eval(shape, cfg))
}

/// Build the full expert plan.
pub fn plan(
    shape: &MoeLayerShape,
    router: Option<&WeightEntry>,
    roofline_slack: f64,
    cfg: &ExpertConfig,
) -> ExpertPlan {
    let affinities = router
        .map(|r| router_affinities(r, shape.n_experts))
        .unwrap_or_default();
    let dead = detect_dead_experts(&affinities, cfg.dead_expert_threshold);
    let capacity = capacity_factor(roofline_slack, shape.top_k, shape.n_experts);
    let placement = plan_placement(shape, cfg);

    let mut alternatives = Vec::new();
    for eval in [balanced_eval(shape, cfg), expert_parallel_eval(shape, cfg), hybrid_eval(shape, cfg)] {
        if eval.strategy != placement.strategy {
            alternatives.push(eval);
        }
    }

    ExpertPlan {
        capacity_factor: capacity,
        dead_experts: dead,
        affinities,
        placement,
        alternatives,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weight_aware::{WeightDType, WeightEntry};

    fn shape() -> MoeLayerShape {
        MoeLayerShape {
            n_experts: 8,
            top_k: 2,
            d_model: 512,
            d_expert: 1024,
            batch: 4,
            seq: 256,
            dtype_bytes: 2,
        }
    }

    fn make_router(per_expert_mag: &[f64]) -> WeightEntry {
        let n = per_expert_mag.len();
        let d = 8;
        let mut data = vec![0u8; d * n * 4];
        for (e, target) in per_expert_mag.iter().enumerate() {
            let value = (target / (d as f64).sqrt()) as f32;
            for r in 0..d {
                let idx = r * n + e;
                let off = idx * 4;
                data[off..off + 4].copy_from_slice(&value.to_le_bytes());
            }
        }
        WeightEntry {
            name: "router.weight".to_string(),
            data,
            shape: vec![d, n],
            dtype: WeightDType::F32,
            num_elements: d * n,
            sparsity: None,
            eliminated: false,
        }
    }

    #[test]
    fn capacity_factor_respects_minimum() {
        // Fraction < 1 → should clamp to 1.
        let cf = capacity_factor(0.5, 2, 16);
        assert!(cf >= 1.0);
    }

    #[test]
    fn capacity_factor_scales_with_slack() {
        // Both inputs must exceed the min=1.0 floor for scaling to be
        // visible: slack × top_k / n_experts > 1.
        let lo = capacity_factor(5.0, 2, 8); // 5 × 2 / 8 = 1.25
        let hi = capacity_factor(20.0, 2, 8); // 20 × 2 / 8 = 5.0
        assert!(hi > lo);
    }

    #[test]
    fn dead_experts_detected_by_affinity() {
        let router = make_router(&[1.0, 1.0, 0.001, 1.0, 1.0, 0.0005, 1.0, 1.0]);
        let aff = router_affinities(&router, 8);
        let dead = detect_dead_experts(&aff, 0.1);
        assert_eq!(dead, vec![2, 5]);
    }

    #[test]
    fn router_affinity_scales_with_weight_magnitude() {
        let router = make_router(&[1.0, 2.0, 3.0, 4.0]);
        let aff = router_affinities(&router, 4);
        assert_eq!(aff.len(), 4);
        for (i, a) in aff.iter().enumerate() {
            assert!(a.affinity > 0.0);
            assert_eq!(a.expert, i as u32);
        }
        assert!(aff[3].affinity > aff[0].affinity);
    }

    /// The three placements on one shape, worked by hand.
    ///
    /// 4 x 256 = 1024 tokens, top-2, 512-wide, 2 bytes: dispatching every
    /// assignment moves 1024 x 2 x 512 x 2 = 2,097,152 bytes. On 8 GPUs a
    /// partition keeps 1/8 of the assignments local (1,835,008 dispatched),
    /// expert-parallel dispatches all, and the hybrid dispatches half
    /// (1,048,576). An expert is 2 x 512 x 1024 x 2 = 2,097,152 bytes; a
    /// partition holds 1 per GPU and the hybrid 4 replicated + 1 sharded.
    #[test]
    fn placements_are_charged_for_what_they_dispatch_and_hold() {
        let cfg = ExpertConfig::default();
        let balanced = balanced_eval(&shape(), &cfg);
        let parallel = expert_parallel_eval(&shape(), &cfg);
        let hybrid = hybrid_eval(&shape(), &cfg);
        assert_eq!(balanced.comm_volume_bytes, 1_835_008);
        assert_eq!(parallel.comm_volume_bytes, 2_097_152);
        assert_eq!(hybrid.comm_volume_bytes, 1_048_576);
        assert_eq!(balanced.memory_per_gpu_bytes, 2_097_152);
        assert_eq!(parallel.memory_per_gpu_bytes, 2_097_152);
        assert_eq!(hybrid.memory_per_gpu_bytes, 5 * 2_097_152);
        // 1,835,008 bytes at 9e11 B/s.
        assert!((balanced.step_time_us - 1_835_008.0 / 9e11 * 1e6).abs() < 1e-9);
    }

    /// Halving `top_k` as an integer made the hybrid's dispatch 0 for top-1
    /// routing. Half of 1024 x 1 x 512 x 2 bytes is 524,288.
    #[test]
    fn hybrid_dispatch_is_not_zero_for_top_1() {
        let top1 = MoeLayerShape { top_k: 1, ..shape() };
        assert_eq!(hybrid_eval(&top1, &ExpertConfig::default()).comm_volume_bytes, 524_288);
    }

    /// With room for the replicated experts, the hybrid's smaller dispatch
    /// wins; when memory only fits a partition, the partition does. Balanced
    /// used to win every time, at a cost of zero.
    #[test]
    fn the_cheapest_placement_that_fits_wins() {
        let roomy = plan_placement(&shape(), &ExpertConfig::default());
        assert_eq!(roomy.strategy, PlacementStrategy::Hybrid);
        let tight = ExpertConfig {
            per_gpu_memory_bytes: 3 * 2_097_152,
            ..ExpertConfig::default()
        };
        assert_eq!(plan_placement(&shape(), &tight).strategy, PlacementStrategy::Balanced);
    }

    #[test]
    fn expert_parallel_has_nonzero_comm() {
        let eval = expert_parallel_eval(&shape(), &ExpertConfig::default());
        assert!(eval.comm_volume_bytes > 0);
    }

    #[test]
    fn plan_picks_a_feasible_strategy() {
        let router = make_router(&[1.0; 8]);
        let p = plan(&shape(), Some(&router), 1.5, &ExpertConfig::default());
        assert!(p.placement.feasible);
        assert!(p.capacity_factor >= 1.0);
    }

    #[test]
    fn plan_reports_dead_experts() {
        let router = make_router(&[1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0]);
        let p = plan(&shape(), Some(&router), 1.0, &ExpertConfig::default());
        assert_eq!(p.dead_experts, vec![2, 5]);
        assert_eq!(p.active_experts(shape().n_experts), 6);
    }

    #[test]
    fn plan_still_works_without_router() {
        let p = plan(&shape(), None, 1.0, &ExpertConfig::default());
        assert!(p.affinities.is_empty());
        assert!(p.placement.feasible);
    }

    #[test]
    fn tight_memory_budget_eliminates_expert_parallel() {
        let mut cfg = ExpertConfig::default();
        cfg.per_gpu_memory_bytes = 1024; // too tight for any expert
        let p = plan(&shape(), None, 1.0, &cfg);
        // Should fall through to the Balanced fallback even though
        // everything is infeasible.
        assert_eq!(p.placement.strategy, PlacementStrategy::Balanced);
    }
}
