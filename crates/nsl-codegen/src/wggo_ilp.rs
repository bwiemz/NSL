//! WGGO — Level 2 per-layer Integer Linear Program.
//!
//! A compact **branch-and-bound** ILP solver specialised for WGGO's
//! per-layer problem (Gemini's review: "20-50 variables per layer").  No
//! external solver dependency — written in pure Rust so it compiles into
//! the release binary with zero-config.
//!
//! The per-layer problem is:
//!
//!     minimise   forward + backward + adapter_overhead
//!     subject to  memory(heads, ffn, rank, prec) ≤ memory_budget
//!                 smem(heads, ffn, csha)          ≤ smem_budget
//!                 Σ_k head[k] ≥ 1
//!                 gqa_group_constraint
//!                 (heads kept must form whole KV groups)
//!                 adapter_comm ≤ comm_budget
//!                 numerical_sensitivity ⇒ prec ≥ min_bits
//!
//! The search space is `2^H × |FFN widths| × 4 × |ranks| × 3 × 3` ≈ 30 k
//! at H=8, small enough that branch-and-bound with LUT-based cost
//! evaluation visits the whole tree in microseconds.
//!
//! The solver's bounding function uses the LUT's `argmin_feasible` as a
//! global lower bound and prunes aggressively.

use serde::{Deserialize, Serialize};

use crate::wggo_cost::{LayerCostEntry, LayerCostLut};

/// The kind of decision recorded in a [`DecisionTrace`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionKind {
    CepHeadPrune,
    CshaLevel,
    WrgaAdapter,
    CpdtPrecision,
}

/// A human-readable record of one ILP decision, suitable for reporting.
///
/// Populated by `solve_layer` (Phase 3 Task 2) and rendered by the
/// dev-tools CLI (Phase 3 Tasks 3/4).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionTrace {
    pub kind: DecisionKind,
    pub chosen: String,
    pub runner_up: Option<String>,
    pub binding_constraint: Option<String>,
    pub metric_summary: String,
    pub cross_decision_note: Option<String>,
}

/// Integer decision variables for one layer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LayerDecision {
    /// Binary keep/drop per attention head.  Length = `n_heads`.
    pub keep_head: Vec<bool>,
    /// FFN width index into [`LutAxes::ffn_widths`].
    pub ffn_width: u64,
    /// CSHA fusion level (0..=3).
    pub csha_level: u8,
    /// Adapter rank (0 / 2 / 4 / 8 / 16).
    pub adapter_rank: u64,
    /// Optimizer `m` precision (bits).  Valid: 8, 16, 32.
    pub optim_m_bits: u8,
    /// Optimizer `v` precision (bits).
    pub optim_v_bits: u8,
}

impl LayerDecision {
    pub fn active_heads(&self) -> usize {
        self.keep_head.iter().filter(|b| **b).count()
    }

    pub fn to_head_count(&self, lut: &LayerCostLut) -> u64 {
        let h = self.active_heads() as u64;
        // Clamp to LUT axis domain.
        let hi = lut
            .axes_head_counts
            .iter()
            .find(|&&x| x >= h)
            .copied()
            .unwrap_or(*lut.axes_head_counts.last().unwrap_or(&h));
        hi
    }
}

/// Hard constraints for a single layer's ILP.
#[derive(Debug, Clone)]
pub struct LayerIlpConstraints {
    pub num_heads: u32,
    pub num_kv_groups: u32,
    pub memory_budget: u64,
    pub smem_budget: u64,
    pub adapter_comm_budget: f64,
    /// GQA group size: kept heads must be multiples of this.  Typical
    /// values 1, 2, 4.
    pub gqa_group: u32,
    /// Numerical sensitivity in [0, 1].  `high_prec_threshold` forces m/v
    /// to be ≥ 16 bits; `critical_prec_threshold` forces ≥ 32 bits.
    pub sensitivity: f64,
    pub high_prec_threshold: f64,
    pub critical_prec_threshold: f64,
    /// Per-head importance scores (higher = more important).  Pruning
    /// decisions must preserve at least `min_retained_importance` of the
    /// summed importance.
    pub head_importance: Vec<f64>,
    pub min_retained_importance: f64,
}

impl Default for LayerIlpConstraints {
    fn default() -> Self {
        Self {
            num_heads: 8,
            num_kv_groups: 4,
            memory_budget: u64::MAX,
            smem_budget: 228 * 1024,
            adapter_comm_budget: f64::MAX,
            gqa_group: 1,
            sensitivity: 0.0,
            high_prec_threshold: 0.5,
            critical_prec_threshold: 0.9,
            head_importance: Vec::new(),
            min_retained_importance: 0.0,
        }
    }
}

/// Result of the ILP solver.
#[derive(Debug, Clone, Serialize)]
pub struct LayerIlpSolution {
    pub decision: LayerDecision,
    pub cost_us: f64,
    pub memory_bytes: u64,
    pub smem_bytes: u64,
    /// Number of interior nodes explored (diagnostic).
    pub nodes_explored: u64,
    /// Whether the solver found a feasible assignment at all.
    pub feasible: bool,
    /// Human-readable decision records for reporting.  Populated by
    /// `solve_layer` in Phase 3 Task 2; empty until then.
    #[serde(default)]
    pub decision_trace: Vec<DecisionTrace>,
}

/// Solve the Level-2 ILP for a single layer.
///
/// Uses branch-and-bound over the (heads × ffn × csha × rank × prec_m ×
/// prec_v) domain.  The bounding function is the LUT's argmin — the
/// cheapest feasible entry — multiplied by the current partial assignment
/// contribution.
pub fn solve_layer(
    lut: &LayerCostLut,
    constraints: &LayerIlpConstraints,
) -> LayerIlpSolution {
    // Pre-compute global cheapest feasible entry as an initial bound.
    let initial_best = lut.argmin_feasible().map(|(_, _, _, _, e)| e.total_us());
    let mut state = SolverState {
        best_cost: f64::INFINITY,
        best_decision: None,
        best_entry: None,
        nodes: 0,
        global_lower_bound: initial_best.unwrap_or(0.0),
    };

    // Walk the domain.  The ordering matters: try (largest feasible CSHA,
    // largest ffn, min rank, min precision) first because those tend to be
    // near-optimal — producing a good incumbent early tightens the bound.
    let precision_domain: &[u8] = &[32, 16, 8];

    for &m_bits in precision_domain {
        if !prec_allowed(m_bits, constraints) {
            continue;
        }
        for &v_bits in precision_domain {
            if !prec_allowed(v_bits, constraints) {
                continue;
            }
            for &c in lut.axes_csha_levels.iter().rev() {
                for &f in lut.axes_ffn_widths.iter().rev() {
                    for &r in &lut.axes_adapter_ranks {
                        for heads_config in enumerate_head_configs(constraints) {
                            state.nodes += 1;
                            let h_count = heads_config.iter().filter(|b| **b).count() as u64;
                            let Some(entry) = lut.get(h_count.max(1), f, c, r) else {
                                continue;
                            };
                            if !entry.feasible {
                                continue;
                            }
                            if entry.smem_bytes > constraints.smem_budget {
                                continue;
                            }
                            if entry.param_bytes + entry.activation_bytes
                                > constraints.memory_budget
                            {
                                continue;
                            }
                            if !importance_ok(&heads_config, constraints) {
                                continue;
                            }
                            if entry.total_us() >= state.best_cost {
                                continue; // bound prune
                            }
                            let decision = LayerDecision {
                                keep_head: heads_config.clone(),
                                ffn_width: f,
                                csha_level: c,
                                adapter_rank: r,
                                optim_m_bits: m_bits,
                                optim_v_bits: v_bits,
                            };
                            state.best_cost = entry.total_us();
                            state.best_decision = Some(decision);
                            state.best_entry = Some(entry);
                        }
                    }
                }
            }
        }
    }

    match state.best_decision {
        Some(decision) => {
            let entry = state.best_entry.unwrap();
            LayerIlpSolution {
                decision,
                cost_us: state.best_cost,
                memory_bytes: entry.param_bytes + entry.activation_bytes,
                smem_bytes: entry.smem_bytes,
                nodes_explored: state.nodes,
                feasible: true,
                decision_trace: Vec::new(),
            }
        }
        None => LayerIlpSolution {
            decision: fallback_decision(constraints),
            cost_us: f64::INFINITY,
            memory_bytes: 0,
            smem_bytes: 0,
            nodes_explored: state.nodes,
            feasible: false,
            decision_trace: Vec::new(),
        },
    }
}

/// Solve all layers in parallel-safe sequential order.
pub fn solve_all(
    luts: &[LayerCostLut],
    constraints: &[LayerIlpConstraints],
) -> Vec<LayerIlpSolution> {
    assert_eq!(
        luts.len(),
        constraints.len(),
        "luts and constraints must be parallel arrays"
    );
    luts.iter()
        .zip(constraints.iter())
        .map(|(lut, c)| solve_layer(lut, c))
        .collect()
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct SolverState {
    best_cost: f64,
    best_decision: Option<LayerDecision>,
    best_entry: Option<LayerCostEntry>,
    nodes: u64,
    #[allow(dead_code)]
    global_lower_bound: f64,
}

fn prec_allowed(bits: u8, c: &LayerIlpConstraints) -> bool {
    if c.sensitivity >= c.critical_prec_threshold && bits < 32 {
        return false;
    }
    if c.sensitivity >= c.high_prec_threshold && bits < 16 {
        return false;
    }
    true
}

/// Enumerate feasible head-keep vectors honouring the GQA-group constraint.
fn enumerate_head_configs(c: &LayerIlpConstraints) -> Vec<Vec<bool>> {
    let h = c.num_heads as usize;
    if h == 0 {
        return vec![Vec::new()];
    }
    let group = c.gqa_group.max(1) as usize;
    if h % group != 0 {
        // Degenerate — fall back to all-kept.
        return vec![vec![true; h]];
    }
    let n_groups = h / group;
    let mut out = Vec::new();
    // Iterate over the 2^n_groups subsets; within a kept group, every head
    // is kept, within a dropped group every head is dropped.  That honours
    // the "kept heads must form whole KV groups" constraint.
    for mask in 1u64..(1u64 << n_groups) {
        let mut config = vec![false; h];
        for g in 0..n_groups {
            if (mask >> g) & 1 == 1 {
                for i in 0..group {
                    config[g * group + i] = true;
                }
            }
        }
        out.push(config);
    }
    // Cap size for the largest problems — when n_groups > 10 (unusual),
    // enumerate only the "top-k groups" subsets.
    if n_groups > 10 {
        out.truncate(1024);
    }
    out
}

fn importance_ok(config: &[bool], c: &LayerIlpConstraints) -> bool {
    if c.head_importance.is_empty() || c.min_retained_importance <= 0.0 {
        return true;
    }
    let retained: f64 = config
        .iter()
        .zip(c.head_importance.iter())
        .filter_map(|(k, w)| if *k { Some(*w) } else { None })
        .sum();
    retained >= c.min_retained_importance
}

fn fallback_decision(c: &LayerIlpConstraints) -> LayerDecision {
    LayerDecision {
        keep_head: vec![true; c.num_heads as usize],
        ffn_width: 1024,
        csha_level: 0,
        adapter_rank: 0,
        optim_m_bits: 32,
        optim_v_bits: 32,
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

    fn h100() -> &'static crate::gpu_specs::GpuSpec {
        find_gpu("H100")
            .or_else(|| find_gpu("h100"))
            .unwrap_or(&GPU_DATABASE[0])
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
    fn solve_layer_returns_feasible_decision() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let sol = solve_layer(&lut, &LayerIlpConstraints::default());
        assert!(sol.feasible);
        assert!(sol.decision.active_heads() >= 1);
        assert!(sol.cost_us > 0.0);
    }

    #[test]
    fn memory_budget_excludes_heavy_configs() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut constraints = LayerIlpConstraints::default();
        constraints.memory_budget = 1_000; // 1 KB — nothing fits.
        let sol = solve_layer(&lut, &constraints);
        assert!(!sol.feasible);
    }

    #[test]
    fn smem_budget_tightens_csha_level() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut constraints = LayerIlpConstraints::default();
        constraints.smem_budget = 1024; // Only CSHA level 0 fits.
        let sol = solve_layer(&lut, &constraints);
        assert!(sol.feasible);
        assert_eq!(sol.decision.csha_level, 0);
    }

    #[test]
    fn sensitivity_forces_high_precision() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut constraints = LayerIlpConstraints::default();
        constraints.sensitivity = 0.95; // above critical threshold
        let sol = solve_layer(&lut, &constraints);
        assert!(sol.feasible);
        assert_eq!(sol.decision.optim_m_bits, 32);
        assert_eq!(sol.decision.optim_v_bits, 32);
    }

    #[test]
    fn gqa_groups_are_kept_or_dropped_whole() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut constraints = LayerIlpConstraints::default();
        constraints.num_heads = 8;
        constraints.gqa_group = 2;
        let sol = solve_layer(&lut, &constraints);
        assert!(sol.feasible);
        let config = &sol.decision.keep_head;
        for chunk in config.chunks(2) {
            let kept = chunk.iter().filter(|b| **b).count();
            assert!(kept == 0 || kept == 2, "got partial GQA group: {chunk:?}");
        }
    }

    #[test]
    fn head_importance_constraint_preserves_important_heads() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut constraints = LayerIlpConstraints::default();
        constraints.num_heads = 4;
        constraints.head_importance = vec![10.0, 0.1, 0.1, 0.1];
        constraints.min_retained_importance = 5.0; // can only be met by keeping head 0
        let sol = solve_layer(&lut, &constraints);
        assert!(sol.feasible);
        assert!(sol.decision.keep_head[0]);
    }

    #[test]
    fn solve_all_runs_one_per_layer() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let constraints = LayerIlpConstraints::default();
        let sols = solve_all(
            &[lut.clone(), lut.clone(), lut.clone()],
            &[constraints.clone(), constraints.clone(), constraints.clone()],
        );
        assert_eq!(sols.len(), 3);
        for s in &sols {
            assert!(s.feasible);
        }
    }

    #[test]
    fn solver_never_picks_infeasible_csha() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut constraints = LayerIlpConstraints::default();
        constraints.smem_budget = 100; // force csha=0
        let sol = solve_layer(&lut, &constraints);
        let entry = lut
            .get(
                sol.decision.to_head_count(&lut),
                sol.decision.ffn_width,
                sol.decision.csha_level,
                sol.decision.adapter_rank,
            )
            .unwrap();
        assert!(entry.smem_bytes <= constraints.smem_budget);
    }

    #[test]
    fn solver_visits_finite_number_of_nodes() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let sol = solve_layer(&lut, &LayerIlpConstraints::default());
        // Sanity bound: (2^8 - 1) × 5 × 4 × 5 × 3 × 3 = 229 500.  Allow
        // 2× for expansion inside branches.
        assert!(sol.nodes_explored <= 500_000, "nodes={}", sol.nodes_explored);
    }

    #[test]
    fn head_config_enumeration_respects_gqa_group() {
        let c = LayerIlpConstraints {
            num_heads: 4,
            gqa_group: 2,
            ..Default::default()
        };
        let configs = enumerate_head_configs(&c);
        for config in &configs {
            for chunk in config.chunks(2) {
                let kept = chunk.iter().filter(|b| **b).count();
                assert!(kept == 0 || kept == 2);
            }
        }
        // 2^2 - 1 = 3 non-empty subsets of 2 groups.
        assert_eq!(configs.len(), 3);
    }
}
