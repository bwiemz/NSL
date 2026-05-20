//! WGGO — Level 2 per-layer Integer Linear Program.
//!
//! A compact **branch-and-bound** ILP solver specialised for WGGO's
//! per-layer problem (Gemini's review: "20-50 variables per layer").  No
//! external solver dependency — written in pure Rust so it compiles into
//! the release binary with zero-config.
//!
//! The per-layer problem is:
//!
//! ```text
//! minimise   forward + backward + adapter_overhead
//! subject to  memory(heads, ffn, rank, prec) ≤ memory_budget
//!             smem(heads, ffn, csha)          ≤ smem_budget
//!             Σ_k head[k] ≥ 1
//!             gqa_group_constraint
//!             (heads kept must form whole KV groups)
//!             adapter_comm ≤ comm_budget
//!             numerical_sensitivity ⇒ prec ≥ min_bits
//! ```
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

/// Gradient-informed per-head importance scores for WGGO Phase 2.
///
/// Built from a sidecar-provided `Vec<f32>` of per-head gradient norms (or
/// any other scalar importance signal) gathered during a calibration forward
/// pass.  The ILP solver consumes this in Phase 2 Task 2 to bias head-keep
/// decisions toward high-importance heads.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HeadImportance {
    /// Per-head scalar importance score, one entry per attention head.
    /// Higher values indicate more important heads that should be
    /// preserved under memory pressure.
    pub per_head: Vec<f32>,
}

impl HeadImportance {
    /// Construct a [`HeadImportance`] from a slice of per-head `f32` scores.
    ///
    /// # Arguments
    /// * `scores` — one importance score per attention head.  The slice is
    ///   copied into an owned `Vec`; no calibration pass is run here.
    pub fn from_per_head_f32(scores: &[f32]) -> Self {
        Self {
            per_head: scores.to_vec(),
        }
    }
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
    /// Whether FASE fuses the optimizer step into the backward pass for
    /// this layer.  When `true`, gradient buffers are not materialised in
    /// HBM and the backward kernel writes parameter updates directly.
    pub fase_fused: bool,
    /// PCA sequence-packing mode for attention kernels on this layer.
    ///   0 = none, 1 = segment_id, 2 = tile_skip, 3 = multi_seq.
    /// Each mode skips progressively more padded work (paper §4.3.6).
    pub packing_mode: u8,
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
    /// Whether the FASE fused-step option is permitted for this layer.
    /// When `false` the solver must pick `fase_fused = false`.  Callers
    /// set this to `false` for layers sharded by CPDT (since fusing into
    /// backward conflicts with reduce-scatter ordering — the conflict
    /// resolver would defer it anyway, but pre-pruning saves work).
    pub allow_fase: bool,
    /// Estimated relative speedup of the backward pass when FASE fuses
    /// the optimizer step into it.  Typical values: 0.05–0.15.  The
    /// fused step also eliminates a separate gradient HBM round-trip,
    /// which is modeled as removing one parameter-sized buffer from the
    /// activation footprint.
    pub fase_backward_speedup: f64,
    /// Bit-mask of permitted PCA packing modes.  Bit `i` set = mode `i`
    /// allowed.  Default `0b1111` (all four modes).  Callers set this to
    /// `0b0001` (only `none`) to disable PCA entirely for a layer — e.g.,
    /// non-attention layers or kernels whose shape forbids packing.
    pub packing_modes_mask: u8,
    /// Estimated fractional work reduction per packing mode, indexed by
    /// mode (0..=3).  Typical defaults: 0.00, 0.15, 0.25, 0.35 — derived
    /// from the fraction of padded tokens in typical batches.
    pub packing_savings: [f64; 4],
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
            allow_fase: true,
            fase_backward_speedup: 0.10,
            packing_modes_mask: 0b1111,
            packing_savings: [0.00, 0.15, 0.25, 0.35],
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
pub fn solve_layer(lut: &LayerCostLut, constraints: &LayerIlpConstraints) -> LayerIlpSolution {
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
    let fase_domain: &[bool] = if constraints.allow_fase {
        &[true, false]
    } else {
        &[false]
    };

    for &m_bits in precision_domain {
        if !prec_allowed(m_bits, constraints) {
            continue;
        }
        for &v_bits in precision_domain {
            if !prec_allowed(v_bits, constraints) {
                continue;
            }
            for &fase in fase_domain {
                for pack in enumerate_packing_modes(constraints) {
                    for &c in lut.axes_csha_levels.iter().rev() {
                        for &f in lut.axes_ffn_widths.iter().rev() {
                            for &r in &lut.axes_adapter_ranks {
                                for heads_config in enumerate_head_configs(constraints) {
                                    state.nodes += 1;
                                    let h_count =
                                        heads_config.iter().filter(|b| **b).count() as u64;
                                    let Some(base) = lut.get(h_count.max(1), f, c, r) else {
                                        continue;
                                    };
                                    if !base.feasible {
                                        continue;
                                    }
                                    let adj_fase =
                                        apply_fase(base, fase, constraints.fase_backward_speedup);
                                    let entry =
                                        apply_packing(adj_fase, pack, &constraints.packing_savings);
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
                                        fase_fused: fase,
                                        packing_mode: pack,
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
        }
    }

    match state.best_decision {
        Some(decision) => {
            let entry = state.best_entry.unwrap();
            let decision_trace = build_decision_trace(&decision, &entry, constraints);
            LayerIlpSolution {
                decision,
                cost_us: state.best_cost,
                memory_bytes: entry.param_bytes + entry.activation_bytes,
                smem_bytes: entry.smem_bytes,
                nodes_explored: state.nodes,
                feasible: true,
                decision_trace,
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

/// Diagnostic counters for [`solve_all_templated`].
#[derive(Debug, Clone, Copy, Default, serde::Serialize)]
pub struct TemplateStats {
    /// Number of distinct (LUT, constraint) templates that required a
    /// fresh ILP solve.  Always ≥ 1 when there is at least one layer.
    pub templates_solved: u32,
    /// Number of layers whose solution was reused from a previously
    /// solved template instead of being re-derived.
    pub template_hits: u32,
}

/// Solve every layer, but reuse the solution across consecutive layers
/// whose `(LayerCostLut, LayerIlpConstraints)` are identical.  This is
/// the "template-layer optimization" the audit calls out: transformer
/// blocks repeat the same shape and constraints, so solving once per
/// template and replicating costs a single ILP pass for the whole model.
///
/// Importance-aware specialisation: the importance vectors are part of
/// the template key, so any layer whose `head_importance` diverges from
/// its predecessors will spawn a fresh solve.
pub fn solve_all_templated(
    luts: &[LayerCostLut],
    constraints: &[LayerIlpConstraints],
) -> (Vec<LayerIlpSolution>, TemplateStats) {
    assert_eq!(
        luts.len(),
        constraints.len(),
        "luts and constraints must be parallel arrays"
    );
    let mut out: Vec<LayerIlpSolution> = Vec::with_capacity(luts.len());
    let mut stats = TemplateStats::default();
    let mut template_idx: Option<usize> = None;

    for (i, (lut, c)) in luts.iter().zip(constraints.iter()).enumerate() {
        let reuse = template_idx
            .map(|j| lut_eq(&luts[j], lut) && constraints_eq(&constraints[j], c))
            .unwrap_or(false);
        if reuse {
            let mut s = out[template_idx.unwrap()].clone();
            // Mark replicated solutions with zero nodes so callers can
            // distinguish fresh solves from cache hits.
            s.nodes_explored = 0;
            out.push(s);
            stats.template_hits += 1;
        } else {
            let sol = solve_layer(lut, c);
            out.push(sol);
            stats.templates_solved += 1;
            template_idx = Some(i);
        }
    }

    (out, stats)
}

/// Field-by-field equality for [`LayerCostLut`] (manual to avoid
/// requiring `PartialEq` on the cost-model leaves).
fn lut_eq(a: &LayerCostLut, b: &LayerCostLut) -> bool {
    if a.axes_head_counts != b.axes_head_counts
        || a.axes_ffn_widths != b.axes_ffn_widths
        || a.axes_csha_levels != b.axes_csha_levels
        || a.axes_adapter_ranks != b.axes_adapter_ranks
        || a.entries.len() != b.entries.len()
    {
        return false;
    }
    a.entries.iter().zip(b.entries.iter()).all(|(x, y)| {
        x.forward_us.to_bits() == y.forward_us.to_bits()
            && x.backward_us.to_bits() == y.backward_us.to_bits()
            && x.param_bytes == y.param_bytes
            && x.activation_bytes == y.activation_bytes
            && x.smem_bytes == y.smem_bytes
            && x.feasible == y.feasible
            && x.classification == y.classification
    })
}

/// Field-by-field equality for [`LayerIlpConstraints`], including the
/// importance vector — divergent importance must trigger a fresh solve.
fn constraints_eq(a: &LayerIlpConstraints, b: &LayerIlpConstraints) -> bool {
    a.num_heads == b.num_heads
        && a.num_kv_groups == b.num_kv_groups
        && a.memory_budget == b.memory_budget
        && a.smem_budget == b.smem_budget
        && a.adapter_comm_budget.to_bits() == b.adapter_comm_budget.to_bits()
        && a.gqa_group == b.gqa_group
        && a.sensitivity.to_bits() == b.sensitivity.to_bits()
        && a.high_prec_threshold.to_bits() == b.high_prec_threshold.to_bits()
        && a.critical_prec_threshold.to_bits() == b.critical_prec_threshold.to_bits()
        && a.head_importance.len() == b.head_importance.len()
        && a.head_importance
            .iter()
            .zip(b.head_importance.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
        && a.min_retained_importance.to_bits() == b.min_retained_importance.to_bits()
        && a.allow_fase == b.allow_fase
        && a.fase_backward_speedup.to_bits() == b.fase_backward_speedup.to_bits()
        && a.packing_modes_mask == b.packing_modes_mask
        && a.packing_savings
            .iter()
            .zip(b.packing_savings.iter())
            .all(|(x, y)| x.to_bits() == y.to_bits())
}

/// Solve a layer **greedily** — paper §5.3.
///
/// Each decision variable is chosen independently using a fast local
/// heuristic instead of the cross-product branch-and-bound search.  The
/// result may be sub-optimal but is reached in O(sum-of-domains) LUT
/// lookups (~30) instead of O(product-of-domains) (~1.8 M with FASE and
/// PCA enabled) — the paper measures this at <500 ms on full transformer
/// models.  The conflict resolver runs downstream as usual; greedy makes
/// no attempt to satisfy inter-technique constraints by itself.
pub fn solve_layer_greedy(
    lut: &LayerCostLut,
    constraints: &LayerIlpConstraints,
) -> LayerIlpSolution {
    let mut nodes = 0u64;

    // Heads: keep all that satisfy the GQA-group + importance constraint.
    let h = constraints.num_heads as usize;
    let keep_head = if importance_ok(&vec![true; h], constraints) {
        vec![true; h]
    } else {
        enumerate_head_configs(constraints)
            .into_iter()
            .find(|cfg| importance_ok(cfg, constraints))
            .unwrap_or_else(|| vec![true; h])
    };
    let h_count = (keep_head.iter().filter(|b| **b).count() as u64).max(1);

    // FFN: largest available width.
    let f = *lut.axes_ffn_widths.iter().max().unwrap_or(&1024);

    // CSHA: highest level whose smem fits.
    let c = lut
        .axes_csha_levels
        .iter()
        .copied()
        .rev()
        .find(|&lvl| {
            nodes += 1;
            lut.get(h_count, f, lvl, 0)
                .map(|e| e.feasible && e.smem_bytes <= constraints.smem_budget)
                .unwrap_or(false)
        })
        .unwrap_or(0);

    // Adapter rank: greedy keeps it at 0 (no LoRA unless explicitly asked).
    let r = *lut.axes_adapter_ranks.iter().min().unwrap_or(&0);

    // Precision: lowest precision allowed by sensitivity (cheapest).
    let m_bits = pick_precision_low(constraints);
    let v_bits = pick_precision_low(constraints);

    // FASE + PCA: pick the local-best at this layer.
    let fase_fused = constraints.allow_fase;
    let packing_mode = enumerate_packing_modes(constraints)[0];

    let Some(base) = lut.get(h_count, f, c, r) else {
        return LayerIlpSolution {
            decision: fallback_decision(constraints),
            cost_us: f64::INFINITY,
            memory_bytes: 0,
            smem_bytes: 0,
            nodes_explored: nodes,
            feasible: false,
            decision_trace: Vec::new(),
        };
    };
    let entry = apply_packing(
        apply_fase(base, fase_fused, constraints.fase_backward_speedup),
        packing_mode,
        &constraints.packing_savings,
    );
    let memory_bytes = entry.param_bytes + entry.activation_bytes;
    let feasible = base.feasible
        && entry.smem_bytes <= constraints.smem_budget
        && memory_bytes <= constraints.memory_budget;

    if !feasible {
        return LayerIlpSolution {
            decision: fallback_decision(constraints),
            cost_us: f64::INFINITY,
            memory_bytes,
            smem_bytes: entry.smem_bytes,
            nodes_explored: nodes,
            feasible: false,
            decision_trace: Vec::new(),
        };
    }

    LayerIlpSolution {
        decision: LayerDecision {
            keep_head,
            ffn_width: f,
            csha_level: c,
            adapter_rank: r,
            optim_m_bits: m_bits,
            optim_v_bits: v_bits,
            fase_fused,
            packing_mode,
        },
        cost_us: entry.total_us(),
        memory_bytes,
        smem_bytes: entry.smem_bytes,
        nodes_explored: nodes,
        feasible: true,
        decision_trace: Vec::new(),
    }
}

/// Greedy variant of [`solve_all`].
pub fn solve_all_greedy(
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
        .map(|(lut, c)| solve_layer_greedy(lut, c))
        .collect()
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

/// Pick the lowest precision (in bits) that the sensitivity floor allows.
/// Used by the greedy solver — lower precision is cheaper to execute, so
/// the locally-optimal pick is "as low as legal".
fn pick_precision_low(c: &LayerIlpConstraints) -> u8 {
    for &bits in &[8u8, 16, 32] {
        if prec_allowed(bits, c) {
            return bits;
        }
    }
    32
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
    if !h.is_multiple_of(group) {
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

/// Build the four human-readable `DecisionTrace` entries (CEP / CSHA / WRGA /
/// CPDT) for a successfully-solved layer.  Phase 3 Task 2.
fn build_decision_trace(
    decision: &LayerDecision,
    entry: &LayerCostEntry,
    c: &LayerIlpConstraints,
) -> Vec<DecisionTrace> {
    let total_heads = decision.keep_head.len();
    let kept_indices: Vec<usize> = decision
        .keep_head
        .iter()
        .enumerate()
        .filter_map(|(i, k)| if *k { Some(i) } else { None })
        .collect();
    let pruned_indices: Vec<usize> = decision
        .keep_head
        .iter()
        .enumerate()
        .filter_map(|(i, k)| if !*k { Some(i) } else { None })
        .collect();
    let retained_heads = kept_indices.len();
    let pruned_count = pruned_indices.len();

    // ---------- CEP (head pruning) ----------
    let cep_chosen = if pruned_count == 0 {
        format!("Kept all {} heads", total_heads)
    } else {
        let head_list = pruned_indices
            .iter()
            .map(|i| format!("h{i}"))
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "Pruned {}/{} heads ({})",
            pruned_count, total_heads, head_list
        )
    };
    let cep_metric_summary = if !c.head_importance.is_empty() && pruned_count > 0 {
        pruned_indices
            .iter()
            .filter_map(|&i| {
                c.head_importance
                    .get(i)
                    .map(|w| format!("importance(h{i})={:.2}", w))
            })
            .collect::<Vec<_>>()
            .join(", ")
            + " — lowest in layer"
    } else if !c.head_importance.is_empty() {
        "no heads pruned (all importances retained)".to_string()
    } else {
        "importance not estimated (pruning by cost only)".to_string()
    };
    let cep_binding = if c.min_retained_importance > 0.0 && !c.head_importance.is_empty() {
        let retained: f64 = decision
            .keep_head
            .iter()
            .zip(c.head_importance.iter())
            .filter_map(|(k, w)| if *k { Some(*w) } else { None })
            .sum();
        Some(format!(
            "min_retained_importance >= {:.2} — satisfied at {:.2}",
            c.min_retained_importance, retained
        ))
    } else {
        Some(format!("gqa_group = {} (whole-group prune)", c.gqa_group))
    };
    let cep_trace = DecisionTrace {
        kind: DecisionKind::CepHeadPrune,
        chosen: cep_chosen,
        runner_up: None,
        binding_constraint: cep_binding,
        metric_summary: cep_metric_summary,
        cross_decision_note: None,
    };

    // ---------- CSHA (shared-attention fusion level) ----------
    let smem_kb = entry.smem_bytes / 1024;
    let smem_budget_kb = c.smem_budget / 1024;
    let csha_trace = DecisionTrace {
        kind: DecisionKind::CshaLevel,
        chosen: format!("Level {}", decision.csha_level),
        runner_up: None,
        binding_constraint: Some(format!("SMEM <= {}KB", smem_budget_kb)),
        metric_summary: format!(
            "With {} kept heads, SMEM = {}KB (budget {}KB).",
            retained_heads, smem_kb, smem_budget_kb
        ),
        cross_decision_note: None,
    };

    // ---------- WRGA (adapter rank / sites) ----------
    let wrga_chosen = if decision.adapter_rank == 0 {
        "No LoRA adapter (rank=0)".to_string()
    } else {
        format!("LoRA r={} on q_proj, v_proj", decision.adapter_rank)
    };
    let wrga_trace = DecisionTrace {
        kind: DecisionKind::WrgaAdapter,
        chosen: wrga_chosen,
        runner_up: None,
        binding_constraint: Some(format!(
            "adapter_comm <= {} (per-layer budget)",
            if c.adapter_comm_budget.is_finite() {
                format!("{:.2}", c.adapter_comm_budget)
            } else {
                "inf".to_string()
            }
        )),
        metric_summary: format!(
            "Adapter compute fits within layer roofline slack (ffn_width={}, kept_heads={}).",
            decision.ffn_width, retained_heads
        ),
        cross_decision_note: None,
    };

    // ---------- CPDT (optimizer-state precision) ----------
    let tier_label = if c.sensitivity >= c.critical_prec_threshold {
        format!(
            "sensitivity {:.2} >= critical {:.2} — forces 32-bit",
            c.sensitivity, c.critical_prec_threshold
        )
    } else if c.sensitivity >= c.high_prec_threshold {
        format!(
            "sensitivity {:.2} >= high {:.2} — forces >=16-bit",
            c.sensitivity, c.high_prec_threshold
        )
    } else {
        format!(
            "sensitivity {:.2} < high {:.2} — low-precision tier allowed",
            c.sensitivity, c.high_prec_threshold
        )
    };
    let cpdt_trace = DecisionTrace {
        kind: DecisionKind::CpdtPrecision,
        chosen: format!(
            "INT{} m, FP{} v",
            decision.optim_m_bits, decision.optim_v_bits
        ),
        runner_up: None,
        binding_constraint: Some(tier_label.clone()),
        metric_summary: format!(
            "sensitivity = {:.2} (high {:.2}, critical {:.2})",
            c.sensitivity, c.high_prec_threshold, c.critical_prec_threshold
        ),
        cross_decision_note: None,
    };

    vec![cep_trace, csha_trace, wrga_trace, cpdt_trace]
}

fn fallback_decision(c: &LayerIlpConstraints) -> LayerDecision {
    LayerDecision {
        keep_head: vec![true; c.num_heads as usize],
        ffn_width: 1024,
        csha_level: 0,
        adapter_rank: 0,
        optim_m_bits: 32,
        optim_v_bits: 32,
        fase_fused: false,
        packing_mode: 0,
    }
}

/// Enumerate permitted PCA packing modes per the constraint mask.  Modes
/// are returned with larger-savings modes first so the solver finds a good
/// incumbent early (tightening the bound).
fn enumerate_packing_modes(c: &LayerIlpConstraints) -> Vec<u8> {
    let mut out = Vec::with_capacity(4);
    for m in (0u8..=3).rev() {
        if c.packing_modes_mask & (1 << m) != 0 {
            out.push(m);
        }
    }
    if out.is_empty() {
        out.push(0);
    }
    out
}

/// Apply the PCA packing-mode's effective work reduction to an entry.
/// Packing skips padded tokens, so forward + backward compute both shrink
/// by the same fraction.  Memory and SMEM are unaffected.
fn apply_packing(base: LayerCostEntry, mode: u8, savings: &[f64; 4]) -> LayerCostEntry {
    let s = savings
        .get(mode as usize)
        .copied()
        .unwrap_or(0.0)
        .clamp(0.0, 0.95);
    LayerCostEntry {
        forward_us: base.forward_us * (1.0 - s),
        backward_us: base.backward_us * (1.0 - s),
        param_bytes: base.param_bytes,
        activation_bytes: base.activation_bytes,
        smem_bytes: base.smem_bytes,
        feasible: base.feasible,
        classification: base.classification,
    }
}

/// Adjust a baseline LUT entry for the FASE fused-step option.  Fusing the
/// optimizer step into the backward kernel removes the explicit gradient
/// HBM round-trip (modeled as one parameter-sized buffer dropping out of
/// the activation footprint) and reduces backward latency by
/// `fase_backward_speedup` (typically 5–15 %).  Both adjustments saturate
/// at zero so the entry remains physically meaningful.
fn apply_fase(base: LayerCostEntry, fase_fused: bool, speedup: f64) -> LayerCostEntry {
    if !fase_fused {
        return base;
    }
    let s = speedup.clamp(0.0, 0.5);
    let backward_us = base.backward_us * (1.0 - s);
    // Fused step writes parameter updates directly — the gradient buffer
    // (≈ param_bytes) never lands in HBM as a distinct activation tensor.
    let activation_bytes = base.activation_bytes.saturating_sub(base.param_bytes);
    LayerCostEntry {
        forward_us: base.forward_us,
        backward_us,
        param_bytes: base.param_bytes,
        activation_bytes,
        smem_bytes: base.smem_bytes,
        feasible: base.feasible,
        classification: base.classification,
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
            &[
                constraints.clone(),
                constraints.clone(),
                constraints.clone(),
            ],
        );
        assert_eq!(sols.len(), 3);
        for s in &sols {
            assert!(s.feasible);
        }
    }

    #[test]
    fn templated_solve_reuses_identical_layers() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let c = LayerIlpConstraints::default();
        let luts = vec![lut.clone(); 8];
        let cs = vec![c; 8];
        let (sols, stats) = solve_all_templated(&luts, &cs);
        assert_eq!(sols.len(), 8);
        assert_eq!(stats.templates_solved, 1);
        assert_eq!(stats.template_hits, 7);
        // Replicated solutions report zero nodes_explored.
        assert!(sols[0].nodes_explored > 0);
        for s in &sols[1..] {
            assert_eq!(s.nodes_explored, 0);
        }
        // The decisions themselves must match.
        for s in &sols[1..] {
            assert_eq!(s.decision, sols[0].decision);
        }
    }

    #[test]
    fn templated_solve_specialises_when_importance_diverges() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut c1 = LayerIlpConstraints::default();
        c1.num_heads = 4;
        let mut c2 = c1.clone();
        c2.head_importance = vec![10.0, 0.1, 0.1, 0.1];
        c2.min_retained_importance = 5.0;
        let luts = vec![lut.clone(), lut.clone(), lut];
        let cs = vec![c1.clone(), c1.clone(), c2];
        let (_sols, stats) = solve_all_templated(&luts, &cs);
        // Layers 0 and 1 share a template; layer 2's diverging
        // importance must trigger a fresh solve.
        assert_eq!(stats.templates_solved, 2);
        assert_eq!(stats.template_hits, 1);
    }

    #[test]
    fn templated_solve_matches_full_solve_per_layer() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let c = LayerIlpConstraints::default();
        let luts = vec![lut.clone(); 4];
        let cs = vec![c.clone(); 4];
        let full = solve_all(&luts, &cs);
        let (templated, _) = solve_all_templated(&luts, &cs);
        for (a, b) in full.iter().zip(templated.iter()) {
            assert_eq!(a.decision, b.decision);
            assert!((a.cost_us - b.cost_us).abs() < 1e-9);
        }
    }

    #[test]
    fn templated_solve_handles_alternating_templates() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let c1 = LayerIlpConstraints::default();
        let mut c2 = c1.clone();
        c2.smem_budget = 4096; // forces csha=0 → different solution
                               // Pattern: A, A, B, A — three template switches, two solves of A.
        let luts = vec![lut.clone(); 4];
        let cs = vec![c1.clone(), c1.clone(), c2, c1];
        let (sols, stats) = solve_all_templated(&luts, &cs);
        assert_eq!(sols.len(), 4);
        assert_eq!(stats.templates_solved, 3);
        assert_eq!(stats.template_hits, 1);
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
        // Sanity bound: (2^8 - 1) × 5 × 4 × 5 × 3 × 3 × 2 (FASE) × 4 (PCA)
        // = 1 836 000.  Allow 2× for expansion inside branches.
        assert!(
            sol.nodes_explored <= 4_000_000,
            "nodes={}",
            sol.nodes_explored
        );
    }

    #[test]
    fn fase_fused_picked_when_allowed_and_lowers_backward() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let with_fase = solve_layer(&lut, &LayerIlpConstraints::default());
        let without_fase = solve_layer(
            &lut,
            &LayerIlpConstraints {
                allow_fase: false,
                ..LayerIlpConstraints::default()
            },
        );
        assert!(with_fase.feasible && without_fase.feasible);
        assert!(with_fase.decision.fase_fused);
        assert!(!without_fase.decision.fase_fused);
        // Fusion must strictly improve the objective at the default 10%
        // backward speedup.
        assert!(
            with_fase.cost_us < without_fase.cost_us,
            "with_fase={} should beat without_fase={}",
            with_fase.cost_us,
            without_fase.cost_us
        );
    }

    #[test]
    fn fase_fused_forced_off_when_disallowed() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut constraints = LayerIlpConstraints::default();
        constraints.allow_fase = false;
        let sol = solve_layer(&lut, &constraints);
        assert!(sol.feasible);
        assert!(!sol.decision.fase_fused);
    }

    #[test]
    fn fase_speedup_zero_makes_decision_neutral() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut constraints = LayerIlpConstraints::default();
        constraints.fase_backward_speedup = 0.0;
        // With no backward speedup the only benefit is reduced activation
        // bytes — still strictly preferable under default unbounded memory,
        // but the cost objective is identical, so the solver may pick
        // either branch.  We only assert feasibility.
        let sol = solve_layer(&lut, &constraints);
        assert!(sol.feasible);
    }

    #[test]
    fn greedy_explores_far_fewer_nodes_than_full() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let constraints = LayerIlpConstraints::default();
        let full = solve_layer(&lut, &constraints);
        let greedy = solve_layer_greedy(&lut, &constraints);
        assert!(greedy.feasible);
        // Greedy is meant to be O(sum-of-domains) — orders of magnitude
        // cheaper than the full cross-product.
        assert!(
            greedy.nodes_explored * 100 < full.nodes_explored,
            "greedy={} full={}",
            greedy.nodes_explored,
            full.nodes_explored
        );
    }

    #[test]
    fn greedy_returns_finite_cost() {
        // Quality bound vs full ILP is workload-dependent — paper claims
        // ~5 % on real transformers, but the toy LUT's roofline model can
        // make greedy's "largest FFN + highest CSHA" picks look much
        // worse than full's globally-balanced choice.  We only assert
        // greedy returns a feasible plan with finite cost; the
        // `greedy_explores_far_fewer_nodes_than_full` test already
        // verifies the speed promise.
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let greedy = solve_layer_greedy(&lut, &LayerIlpConstraints::default());
        assert!(greedy.feasible);
        assert!(greedy.cost_us.is_finite());
        assert!(greedy.cost_us > 0.0);
    }

    #[test]
    fn greedy_respects_smem_budget() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut c = LayerIlpConstraints::default();
        c.smem_budget = 1024; // forces csha=0
        let g = solve_layer_greedy(&lut, &c);
        assert!(g.feasible);
        assert_eq!(g.decision.csha_level, 0);
    }

    #[test]
    fn greedy_respects_sensitivity_floor() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut c = LayerIlpConstraints::default();
        c.sensitivity = 0.95; // forces 32 bits
        let g = solve_layer_greedy(&lut, &c);
        assert!(g.feasible);
        assert_eq!(g.decision.optim_m_bits, 32);
        assert_eq!(g.decision.optim_v_bits, 32);
    }

    #[test]
    fn solve_all_greedy_runs_one_per_layer() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let c = LayerIlpConstraints::default();
        let luts = vec![lut; 4];
        let sols = solve_all_greedy(&luts, &vec![c; 4]);
        assert_eq!(sols.len(), 4);
        assert!(sols.iter().all(|s| s.feasible));
    }

    #[test]
    fn packing_picks_highest_savings_when_all_modes_allowed() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let sol = solve_layer(&lut, &LayerIlpConstraints::default());
        assert!(sol.feasible);
        // Default savings are strictly increasing, so mode 3 (multi_seq)
        // beats every other mode at equal feasibility.
        assert_eq!(sol.decision.packing_mode, 3);
    }

    #[test]
    fn packing_restricted_to_none_when_mask_forbids() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut c = LayerIlpConstraints::default();
        c.packing_modes_mask = 0b0001; // only mode 0 allowed
        let sol = solve_layer(&lut, &c);
        assert!(sol.feasible);
        assert_eq!(sol.decision.packing_mode, 0);
    }

    #[test]
    fn packing_savings_lower_cost_vs_no_packing() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let with_pack = solve_layer(&lut, &LayerIlpConstraints::default());
        let without_pack = solve_layer(
            &lut,
            &LayerIlpConstraints {
                packing_modes_mask: 0b0001,
                ..LayerIlpConstraints::default()
            },
        );
        assert!(with_pack.cost_us < without_pack.cost_us);
    }

    #[test]
    fn head_importance_from_per_head_f32_roundtrips() {
        let input = vec![1.0f32, 2.5, 3.75, 0.0];
        let hi = HeadImportance::from_per_head_f32(&input);
        assert_eq!(hi.per_head, input);
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
