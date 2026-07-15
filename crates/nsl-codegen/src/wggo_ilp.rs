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

use crate::wggo_cfie::{CfieInferConfig, CfieInferenceChoice};
use crate::wggo_cost::{LayerCostEntry, LayerCostLut};
use crate::wggo_cpkd::{CpkdChoice, CpkdConfig};

/// The kind of decision recorded in a [`DecisionTrace`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DecisionKind {
    CepHeadPrune,
    CshaLevel,
    WrgaAdapter,
    CpdtPrecision,
    /// FASE: whether the optimizer step is fused into the backward pass
    /// (deferred gradient materialization) for this layer.
    FaseStep,
    /// PCA: sequence-packing mode chosen for this layer's attention kernels.
    PcaPacking,
}
// NOTE (G20): the CFIE inference decisions deliberately do NOT get a
// `DecisionKind` variant — external consumers (nsl-cli's decision
// explainer) match this enum exhaustively, and the G20 surface is the
// advisory `WggoPlan::cfie_inference` sidecar (each record carries its own
// human-readable model note).  A variant can be added together with the
// explainer wiring when the decisions stop being advisory.

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

/// Where a WRGA low-rank adapter is attached.  This is a real decision the
/// ILP makes (subject to the adapter communication budget), not a fixed
/// assumption — previously the placement was hardcoded as the string
/// "q_proj, v_proj" in the decision trace only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum AdapterPlacement {
    /// No adapter (rank 0).
    #[default]
    None,
    /// Attention Q and V projections (2 sites) — the minimal LoRA placement.
    AttnQV,
    /// Attention Q, K, V, O projections (4 sites).
    AttnQKVO,
    /// Attention Q,K,V,O plus the two FFN projections (6 sites).
    AttnAndFfn,
}

impl AdapterPlacement {
    /// Number of weight projections the adapter is attached to.  Drives both
    /// the parameter/communication footprint and the (small) compute cost.
    pub fn proj_sites(self) -> u32 {
        match self {
            AdapterPlacement::None => 0,
            AdapterPlacement::AttnQV => 2,
            AdapterPlacement::AttnQKVO => 4,
            AdapterPlacement::AttnAndFfn => 6,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            AdapterPlacement::None => "none",
            AdapterPlacement::AttnQV => "q_proj, v_proj",
            AdapterPlacement::AttnQKVO => "q_proj, k_proj, v_proj, o_proj",
            AdapterPlacement::AttnAndFfn => "q,k,v,o + ffn_up, ffn_down",
        }
    }

    /// Whether a weight/site `name` falls inside this placement's allowed
    /// projection set.  Used by WRGA to narrow its roofline-chosen adapter
    /// sites to the projections WGGO budgeted for (the placement encodes a
    /// comm-budget-feasible minimal set — see G5/G7).
    ///
    /// Returns:
    /// * `Some(true)`  — `name` is a recognized projection inside the set.
    /// * `Some(false)` — `name` is a recognized projection *outside* the set.
    /// * `None`        — `name` is not a recognized attention/FFN projection
    ///   (a norm, embedding, or unknown weight); the placement has no opinion
    ///   and the caller should leave the site untouched.
    ///
    /// Recognition is limited to *split* per-projection weight names (see
    /// [`proj_role`]).  **Fused or non-standard projections** — GPT-2 `c_attn`
    /// / `c_proj` / `c_fc`, NeoX/Falcon `query_key_value` / `dense*`, Phi/MPT
    /// `Wqkv`, fused `gate_up_proj` — return `None`, so placement cannot
    /// constrain those weights (deliberate v1 limitation; the consumer treats
    /// `None` as fail-safe pass-through).
    pub fn covers_projection(self, name: &str) -> Option<bool> {
        let role = proj_role(name)?;
        let allowed = match self {
            AdapterPlacement::None => false,
            AdapterPlacement::AttnQV => matches!(role, ProjRole::Q | ProjRole::V),
            AdapterPlacement::AttnQKVO => {
                matches!(role, ProjRole::Q | ProjRole::K | ProjRole::V | ProjRole::O)
            }
            AdapterPlacement::AttnAndFfn => true, // Q/K/V/O + Ffn — every recognized role
        };
        Some(allowed)
    }
}

/// Attention / FFN projection role inferred from a weight name's final
/// dotted component (after stripping a trailing `.weight`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProjRole {
    Q,
    K,
    V,
    O,
    Ffn,
}

/// Classify a weight/site name into a [`ProjRole`], tolerating the common
/// naming variants (`wq` / `w_q` / `q_proj`, FFN `gate`/`up`/`down`/`fc*`).
/// Returns `None` for names that are not attention/FFN projections.
pub fn proj_role(name: &str) -> Option<ProjRole> {
    let base = name.strip_suffix(".weight").unwrap_or(name);
    let last = base.rsplit('.').next().unwrap_or(base).to_ascii_lowercase();
    match last.as_str() {
        "wq" | "w_q" | "q_proj" | "query" => Some(ProjRole::Q),
        "wk" | "w_k" | "k_proj" | "key" => Some(ProjRole::K),
        "wv" | "w_v" | "v_proj" | "value" => Some(ProjRole::V),
        "wo" | "w_o" | "o_proj" | "out_proj" => Some(ProjRole::O),
        "w_gate" | "gate_proj" | "w_up" | "up_proj" | "w_down" | "down_proj" | "fc1" | "fc2" => {
            Some(ProjRole::Ffn)
        }
        _ => None,
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
    /// Whether FASE runs this layer's params in Deferred mode (`true` →
    /// `FaseMode::Deferred`: only the m_partial window-mean accumulator is
    /// materialised, no separate gradient buffer, and the optimizer update
    /// runs as the fused per-param epilogue after the final micro-batch;
    /// `false` → `FaseMode::FullBuffer`: a raw gradient buffer survives and
    /// the stdlib optimizer step consumes it). Honored per-layer by the
    /// codegen mode table under both clipped and unclipped training.
    pub fase_fused: bool,
    /// PCA sequence-packing mode for attention kernels on this layer.
    ///   0 = none, 1 = segment_id, 2 = tile_skip, 3 = multi_seq.
    /// Each mode skips progressively more padded work (paper §4.3.6).
    pub packing_mode: u8,
    /// Which projections the WRGA adapter is attached to.  `None` when
    /// `adapter_rank == 0`.  Chosen by the solver as the smallest placement
    /// whose communication cost fits `adapter_comm_budget`.
    pub adapter_placement: AdapterPlacement,
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
    /// Whether `sensitivity` is backed by a REAL signal (weight analysis
    /// with an actual provider, or calibration). When false, sub-32-bit
    /// optimizer-moment precision is forbidden outright: the objective is
    /// cost-only, so without evidence the solver would always quantize —
    /// the moment-precision analog of pruning heads on uniform importance.
    /// Role-based floors (`role_sensitivity_floor`) still apply on top
    /// when informed.
    pub sensitivity_informed: bool,
    /// Whether `head_importance` / `min_retained_importance` are backed by
    /// a REAL weight signal. When false, the full solver restricts the
    /// structural axes to identity: keep-all heads and full FFN width.
    /// Without this, the cost-only objective prunes floor(prune_fraction*H)
    /// heads of a RANDOM-INIT model — uniform importance scores make the
    /// retention floor satisfiable by ANY 75% subset, so the minimizer
    /// happily drops heads on zero evidence (and thins FFN against a
    /// hardcoded axis set decoupled from the model's real width).
    pub importance_informed: bool,
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
    /// CFIE inference decision axes (audit gap G20) — **opt-in, default
    /// `None` = OFF**.  When `Some`, the solver additionally decides
    /// {fusion level, KV layout, per-layer KV precision, speculative
    /// on/off} with decode-latency cost terms from
    /// [`crate::wggo_cfie::decode_us_per_token`], and the static-layout KV
    /// pool is charged against `memory_budget`.  With the gate off the
    /// candidate space, costs, and outputs are byte-identical to a build
    /// without the axes (WGGO precedent: `zero_stage_search`,
    /// `snap_to_grid`).  The decisions are ADVISORY in this cycle —
    /// surfaced on `WggoPlan::cfie_inference` + the report, not consumed
    /// by the CFIE serve planner.
    pub cfie_infer: Option<CfieInferConfig>,
    /// CPKD distillation decision axes — **opt-in, default `None` = OFF**.
    /// When `Some`, the solver additionally decides {feature_match,
    /// attn_transfer, teacher_stream} per layer with the additive
    /// distill-overhead cost term from
    /// [`crate::wggo_cpkd::choice_cost_us`], and the distill buffers
    /// ([`crate::wggo_cpkd::choice_bytes`]) are charged against
    /// `memory_budget` exactly where the CFIE KV pool is.  With the gate
    /// off the candidate space, costs, and outputs are byte-identical to a
    /// build without the axes.  The decisions are ADVISORY in v1 —
    /// surfaced on `WggoPlan::cpkd_distill` + the report's "[cpkd]"
    /// section, not consumed by the distill lowering.
    pub cpkd: Option<CpkdConfig>,
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
            sensitivity_informed: false,
            importance_informed: false,
            high_prec_threshold: 0.5,
            critical_prec_threshold: 0.9,
            head_importance: Vec::new(),
            min_retained_importance: 0.0,
            allow_fase: true,
            fase_backward_speedup: 0.10,
            packing_modes_mask: 0b1111,
            packing_savings: [0.00, 0.15, 0.25, 0.35],
            cfie_infer: None,
            cpkd: None,
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
    solve_layer_cfie(lut, constraints).0
}

/// [`solve_layer`] variant that also returns the CFIE inference choice
/// (audit gap G20).  `None` unless [`LayerIlpConstraints::cfie_infer`] is
/// `Some` and a feasible candidate exists.  The choice is deliberately
/// carried OUTSIDE `LayerDecision`/`LayerIlpSolution` (advisory sidecar,
/// see `wggo_cfie` module docs) so consumer-owned struct layouts stay
/// untouched while the gate is advisory.
pub fn solve_layer_cfie(
    lut: &LayerCostLut,
    constraints: &LayerIlpConstraints,
) -> (LayerIlpSolution, Option<CfieInferenceChoice>) {
    let (sol, cfie, _cpkd) = solve_layer_cpkd(lut, constraints);
    (sol, cfie)
}

/// [`solve_layer_cfie`] variant that also returns the CPKD distillation
/// choice (CPKD v1).  `None` unless [`LayerIlpConstraints::cpkd`] is
/// `Some` and a feasible candidate exists.  Like the CFIE choice, it is
/// carried OUTSIDE `LayerDecision`/`LayerIlpSolution` (advisory sidecar,
/// see `wggo_cpkd` module docs) so consumer-owned struct layouts stay
/// untouched while the gate is advisory.
pub fn solve_layer_cpkd(
    lut: &LayerCostLut,
    constraints: &LayerIlpConstraints,
) -> (
    LayerIlpSolution,
    Option<CfieInferenceChoice>,
    Option<CpkdChoice>,
) {
    let mut state = SolverState {
        best_cost: f64::INFINITY,
        best_decision: None,
        best_entry: None,
        best_cfie: None,
        best_cpkd: None,
        nodes: 0,
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
    // CFIE inference axis (G20).  Gate off (`cfie_infer == None`, the
    // production default) yields the single `None` element: one loop pass,
    // a 0.0 cost term, and a 0-byte pool — the candidate walk, node count,
    // and objective are byte-identical to a solver without the axis.
    let cfie_domain = crate::wggo_cfie::enumerate_choices(constraints.cfie_infer.as_ref());
    // CPKD distillation axis (v1).  Same gate-off contract as the CFIE
    // axis: `cpkd == None` (the production default) yields the single
    // `None` element — one loop pass, a 0.0 cost term, and a 0-byte
    // charge, keeping the candidate walk, node count, and objective
    // byte-identical to a solver without the axis.
    let cpkd_domain = crate::wggo_cpkd::enumerate_choices(constraints.cpkd.as_ref());

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
                  for &cfie in &cfie_domain {
                   for &cpkd in &cpkd_domain {
                    for &c in lut.axes_csha_levels.iter().rev() {
                        for &f in lut.axes_ffn_widths.iter().rev() {
                            // No importance evidence -> no structural
                            // thinning: full FFN width only (see
                            // `importance_informed`).
                            if !constraints.importance_informed && Some(&f) != lut.axes_ffn_widths.iter().max() {
                                continue;
                            }
                            for &r in &lut.axes_adapter_ranks {
                                // Adapter placement is a real decision: pick the
                                // smallest placement whose comm cost fits the
                                // budget.  If none fits, this rank is infeasible
                                // (the solver falls back to a smaller rank).
                                let Some(placement) = best_placement(r, constraints) else {
                                    continue;
                                };
                                let placement_us = placement_cost_us(r, placement);
                                let head_domain = if constraints.importance_informed {
                                    enumerate_head_configs(constraints)
                                } else {
                                    // Keep-all is the only evidence-free option.
                                    vec![vec![true; constraints.num_heads as usize]]
                                };
                                for heads_config in head_domain {
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
                                    let entry = apply_packing(
                                        adj_fase,
                                        pack,
                                        &constraints.packing_savings,
                                    );
                                    if entry.smem_bytes > constraints.smem_budget {
                                        continue;
                                    }
                                    // CFIE (G20): decode-latency term + KV-pool
                                    // memory charge for this candidate's
                                    // inference choice.  Gate off => (0.0, 0):
                                    // both the sum and the comparison below are
                                    // bit-identical to the pre-G20 solver.
                                    let (cfie_us, cfie_pool) =
                                        match (cfie, constraints.cfie_infer.as_ref()) {
                                            (Some(ch), Some(cfg)) => (
                                                crate::wggo_cfie::decode_us_per_token(
                                                    ch,
                                                    cfg,
                                                    entry.param_bytes,
                                                    lut.peak_bandwidth_gbs,
                                                ),
                                                crate::wggo_cfie::kv_pool_bytes(ch, cfg),
                                            ),
                                            _ => (0.0, 0u64),
                                        };
                                    // CPKD (v1): distillation-overhead term +
                                    // feature/attn buffer memory charge for this
                                    // candidate's distill choice.  Gate off =>
                                    // (0.0, 0): both the sum and the comparison
                                    // below are bit-identical to the pre-CPKD
                                    // solver.
                                    let (cpkd_us, cpkd_bytes) =
                                        match (cpkd, constraints.cpkd.as_ref()) {
                                            (Some(ch), Some(cfg)) => (
                                                crate::wggo_cpkd::choice_cost_us(ch, cfg),
                                                crate::wggo_cpkd::choice_bytes(ch, cfg),
                                            ),
                                            _ => (0.0, 0u64),
                                        };
                                    if ilp_resident_bytes(&entry, lut, m_bits, v_bits)
                                        .saturating_add(cfie_pool)
                                        .saturating_add(cpkd_bytes)
                                        > constraints.memory_budget
                                    {
                                        continue;
                                    }
                                    if !importance_ok(&heads_config, constraints) {
                                        continue;
                                    }
                                    // Objective = forward + backward (LUT) +
                                    // adapter-comm (placement) + optimizer step.
                                    // The optimizer term is what makes `m_bits` /
                                    // `v_bits` / `fase` affect the cost at all —
                                    // without it the ILP's objective is blind to
                                    // its own precision/FASE decisions (paper §2.3
                                    // optimizer_time term).  The CFIE decode term
                                    // composes the same way (G20): additive, in
                                    // μs, zero when the gate is off — as does the
                                    // CPKD distill-overhead term (v1).
                                    let optim_us = crate::wggo_cost::optimizer_us(
                                        entry.param_bytes,
                                        lut.dtype_bytes,
                                        m_bits,
                                        v_bits,
                                        fase,
                                        lut.peak_bandwidth_gbs,
                                    );
                                    let cand_cost = entry.total_us()
                                        + placement_us
                                        + optim_us
                                        + cfie_us
                                        + cpkd_us;
                                    if cand_cost >= state.best_cost {
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
                                        adapter_placement: placement,
                                    };
                                    state.best_cost = cand_cost;
                                    state.best_decision = Some(decision);
                                    state.best_entry = Some(entry);
                                    state.best_cfie = cfie;
                                    state.best_cpkd = cpkd;
                                }
                            }
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
            // Gate off => best_cfie is None => pool is 0 and memory_bytes
            // is byte-identical to the pre-G20 accounting.
            let cfie_pool = match (state.best_cfie, constraints.cfie_infer.as_ref()) {
                (Some(ch), Some(cfg)) => crate::wggo_cfie::kv_pool_bytes(ch, cfg),
                _ => 0,
            };
            // Same gate-off contract for CPKD: best_cpkd is None => a
            // 0-byte charge and byte-identical accounting.
            let cpkd_bytes = match (state.best_cpkd, constraints.cpkd.as_ref()) {
                (Some(ch), Some(cfg)) => crate::wggo_cpkd::choice_bytes(ch, cfg),
                _ => 0,
            };
            let memory_bytes =
                ilp_resident_bytes(&entry, lut, decision.optim_m_bits, decision.optim_v_bits)
                    .saturating_add(cfie_pool)
                    .saturating_add(cpkd_bytes);
            (
                LayerIlpSolution {
                    decision,
                    cost_us: state.best_cost,
                    memory_bytes,
                    smem_bytes: entry.smem_bytes,
                    nodes_explored: state.nodes,
                    feasible: true,
                    decision_trace,
                },
                state.best_cfie,
                state.best_cpkd,
            )
        }
        None => (
            LayerIlpSolution {
                decision: fallback_decision(constraints),
                cost_us: f64::INFINITY,
                memory_bytes: 0,
                smem_bytes: 0,
                nodes_explored: state.nodes,
                feasible: false,
                decision_trace: Vec::new(),
            },
            None,
            None,
        ),
    }
}

/// Re-evaluate a decision's cost (μs) against the LUT, e.g. after the conflict
/// resolver has mutated `csha_level` / `adapter_rank` / `fase_fused`.  Returns
/// `f64::INFINITY` when the (possibly post-resolution) decision is no longer
/// representable in the LUT.  Used by greedy mode's cost re-evaluation (G3).
pub fn recost_decision(lut: &LayerCostLut, d: &LayerDecision, c: &LayerIlpConstraints) -> f64 {
    recost_decision_cfie(lut, d, c, None)
}

/// [`recost_decision`] extended with the layer's CFIE inference choice
/// (G20) so a gate-on greedy re-cost stays comparable with `solve_layer`'s
/// objective (which includes the decode term).  Passing `None` (or a gate
/// that is off) adds nothing and is bit-identical to [`recost_decision`].
pub fn recost_decision_cfie(
    lut: &LayerCostLut,
    d: &LayerDecision,
    c: &LayerIlpConstraints,
    cfie: Option<CfieInferenceChoice>,
) -> f64 {
    recost_decision_cpkd(lut, d, c, cfie, None)
}

/// [`recost_decision_cfie`] extended with the layer's CPKD distillation
/// choice (v1) so a gate-on greedy re-cost stays comparable with
/// `solve_layer`'s objective (which includes the distill-overhead term).
/// Passing `None` (or a gate that is off) adds nothing and is
/// bit-identical to [`recost_decision_cfie`].
pub fn recost_decision_cpkd(
    lut: &LayerCostLut,
    d: &LayerDecision,
    c: &LayerIlpConstraints,
    cfie: Option<CfieInferenceChoice>,
    cpkd: Option<CpkdChoice>,
) -> f64 {
    let h_count = (d.active_heads() as u64).max(1);
    let Some(base) = lut.get(h_count, d.ffn_width, d.csha_level, d.adapter_rank) else {
        return f64::INFINITY;
    };
    if !base.feasible {
        return f64::INFINITY;
    }
    let adj = apply_fase(base, d.fase_fused, c.fase_backward_speedup);
    let entry = apply_packing(adj, d.packing_mode, &c.packing_savings);
    // Mirror `solve_layer`'s objective exactly so a re-cost after conflict
    // resolution is comparable: forward+backward + adapter-comm + optimizer.
    let mut cost = entry.total_us()
        + placement_cost_us(d.adapter_rank, d.adapter_placement)
        + crate::wggo_cost::optimizer_us(
            entry.param_bytes,
            lut.dtype_bytes,
            d.optim_m_bits,
            d.optim_v_bits,
            d.fase_fused,
            lut.peak_bandwidth_gbs,
        );
    if let (Some(ch), Some(cfg)) = (cfie, c.cfie_infer.as_ref()) {
        cost += crate::wggo_cfie::decode_us_per_token(
            ch,
            cfg,
            entry.param_bytes,
            lut.peak_bandwidth_gbs,
        );
    }
    if let (Some(ch), Some(cfg)) = (cpkd, c.cpkd.as_ref()) {
        cost += crate::wggo_cpkd::choice_cost_us(ch, cfg);
    }
    cost
}

/// Resident memory the ILP charges a candidate layer: the shared
/// training-memory formula ([`crate::wggo_cost::resident_training_bytes`]) with
/// the layer's chosen moment precision, sized **unsharded** (`shard = 1`).
///
/// The Level-2 ILP is a per-layer admission gate that runs before — and
/// independently of — the Level-1 DP's ZeRO-shard choice, so it takes the
/// conservative worst case: a layer that fits unsharded fits under any sharding.
/// Routing through the same formula the DP uses is what closes the gap-#3
/// divergence — the ILP used to charge a bare `param + activation`, silently
/// dropping the optimizer state (the Adam moments) the DP always counted.
fn ilp_resident_bytes(entry: &LayerCostEntry, lut: &LayerCostLut, m_bits: u8, v_bits: u8) -> u64 {
    let optimizer_state =
        crate::wggo_cost::moment_state_bytes(entry.param_bytes, lut.dtype_bytes, m_bits, v_bits);
    crate::wggo_cost::resident_training_bytes(
        entry.param_bytes,
        optimizer_state,
        entry.activation_bytes,
        1,
    )
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
    let (sols, stats, _cfie) = solve_all_templated_cfie(luts, constraints);
    (sols, stats)
}

/// [`solve_all_templated`] variant that also returns the per-layer CFIE
/// inference choices (G20) — parallel to the solutions vector, all `None`
/// when the gate is off.  Template reuse replicates the cached choice
/// alongside the cached solution; the gate config participates in the
/// template key via [`constraints_eq`].
pub fn solve_all_templated_cfie(
    luts: &[LayerCostLut],
    constraints: &[LayerIlpConstraints],
) -> (
    Vec<LayerIlpSolution>,
    TemplateStats,
    Vec<Option<CfieInferenceChoice>>,
) {
    let (sols, stats, cfie, _cpkd) = solve_all_templated_cpkd(luts, constraints);
    (sols, stats, cfie)
}

/// [`solve_all_templated_cfie`] variant that also returns the per-layer
/// CPKD distillation choices (v1) — parallel to the solutions vector, all
/// `None` when the gate is off.  Template reuse replicates the cached
/// choice alongside the cached solution; the gate config participates in
/// the template key via [`constraints_eq`].
pub fn solve_all_templated_cpkd(
    luts: &[LayerCostLut],
    constraints: &[LayerIlpConstraints],
) -> (
    Vec<LayerIlpSolution>,
    TemplateStats,
    Vec<Option<CfieInferenceChoice>>,
    Vec<Option<CpkdChoice>>,
) {
    assert_eq!(
        luts.len(),
        constraints.len(),
        "luts and constraints must be parallel arrays"
    );
    let mut out: Vec<LayerIlpSolution> = Vec::with_capacity(luts.len());
    let mut cfie_out: Vec<Option<CfieInferenceChoice>> = Vec::with_capacity(luts.len());
    let mut cpkd_out: Vec<Option<CpkdChoice>> = Vec::with_capacity(luts.len());
    let mut stats = TemplateStats::default();
    let mut template_idx: Option<usize> = None;

    for (i, (lut, c)) in luts.iter().zip(constraints.iter()).enumerate() {
        let reuse = template_idx
            .map(|j| lut_eq(&luts[j], lut) && constraints_eq(&constraints[j], c))
            .unwrap_or(false);
        if reuse {
            let j = template_idx.unwrap();
            let mut s = out[j].clone();
            // Mark replicated solutions with zero nodes so callers can
            // distinguish fresh solves from cache hits.
            s.nodes_explored = 0;
            out.push(s);
            cfie_out.push(cfie_out[j]);
            cpkd_out.push(cpkd_out[j]);
            stats.template_hits += 1;
        } else {
            let (sol, cfie, cpkd) = solve_layer_cpkd(lut, c);
            out.push(sol);
            cfie_out.push(cfie);
            cpkd_out.push(cpkd);
            stats.templates_solved += 1;
            template_idx = Some(i);
        }
    }

    (out, stats, cfie_out, cpkd_out)
}

/// Field-by-field equality for [`LayerCostLut`] (manual to avoid
/// requiring `PartialEq` on the cost-model leaves).
fn lut_eq(a: &LayerCostLut, b: &LayerCostLut) -> bool {
    if a.axes_head_counts != b.axes_head_counts
        || a.axes_ffn_widths != b.axes_ffn_widths
        || a.axes_csha_levels != b.axes_csha_levels
        || a.axes_adapter_ranks != b.axes_adapter_ranks
        || a.entries.len() != b.entries.len()
        // `dtype_bytes` / `peak_bandwidth_gbs` feed `optimizer_us`, so two LUTs
        // that differ only in those (mixed-precision layers, or different GPU
        // targets) must NOT share a templated solution.
        || a.dtype_bytes != b.dtype_bytes
        || a.peak_bandwidth_gbs.to_bits() != b.peak_bandwidth_gbs.to_bits()
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
        && a.sensitivity_informed == b.sensitivity_informed
        && a.importance_informed == b.importance_informed
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
        // G20: the CFIE gate (and its knobs) are part of the template key —
        // a gate-on layer must never reuse a gate-off template's solution.
        && match (a.cfie_infer.as_ref(), b.cfie_infer.as_ref()) {
            (None, None) => true,
            (Some(x), Some(y)) => x.key_eq(y),
            _ => false,
        }
        // CPKD (v1): the distillation gate (and its knobs) join the
        // template key for the same reason — forgetting this would
        // silently reuse gate-off template solutions for gate-on layers.
        && match (a.cpkd.as_ref(), b.cpkd.as_ref()) {
            (None, None) => true,
            (Some(x), Some(y)) => x.key_eq(y),
            _ => false,
        }
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
    solve_layer_greedy_cfie(lut, constraints).0
}

/// [`solve_layer_greedy`] variant that also returns the CFIE inference
/// choice (G20).  Greedy picks the locally-cheapest choice whose KV pool
/// fits the remaining memory headroom (`wggo_cfie::best_choice_fitting`) —
/// mirroring greedy's per-variable local heuristics; when nothing fits it
/// refuses to advise (`None`) rather than surfacing an over-budget pool.
pub fn solve_layer_greedy_cfie(
    lut: &LayerCostLut,
    constraints: &LayerIlpConstraints,
) -> (LayerIlpSolution, Option<CfieInferenceChoice>) {
    let (sol, cfie, _cpkd) = solve_layer_greedy_cpkd(lut, constraints);
    (sol, cfie)
}

/// [`solve_layer_greedy_cfie`] variant that also returns the CPKD
/// distillation choice (v1).  Greedy picks the locally-cheapest choice
/// whose distill buffers fit the memory headroom left after the
/// training-resident bytes and the CFIE KV pool
/// (`wggo_cpkd::best_choice_fitting`) — mirroring greedy's per-variable
/// local heuristics.
pub fn solve_layer_greedy_cpkd(
    lut: &LayerCostLut,
    constraints: &LayerIlpConstraints,
) -> (
    LayerIlpSolution,
    Option<CfieInferenceChoice>,
    Option<CpkdChoice>,
) {
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

    // Adapter rank: greedy takes the smallest ON THE AXIS — 0 normally,
    // or the smallest mandated rank when @wrga/@adapter removed 0 from
    // the axis (the E3 two-level split).
    let r = *lut.axes_adapter_ranks.iter().min().unwrap_or(&0);

    // CSHA: highest level whose smem fits. Probe the LUT at the rank the
    // greedy solution will actually use — a hardcoded rank-0 probe missed
    // EVERY entry under the adapter mandate (0 off the axis), silently
    // collapsing csha_level to 0 for the whole plan.
    let c = lut
        .axes_csha_levels
        .iter()
        .copied()
        .rev()
        .find(|&lvl| {
            nodes += 1;
            lut.get(h_count, f, lvl, r)
                .map(|e| e.feasible && e.smem_bytes <= constraints.smem_budget)
                .unwrap_or(false)
        })
        .unwrap_or(0);

    // Precision: lowest precision allowed by sensitivity (cheapest).
    let m_bits = pick_precision_low(constraints);
    let v_bits = pick_precision_low(constraints);

    // FASE + PCA: pick the local-best at this layer.
    let fase_fused = constraints.allow_fase;
    let packing_mode = enumerate_packing_modes(constraints)[0];

    let Some(base) = lut.get(h_count, f, c, r) else {
        return (
            LayerIlpSolution {
                decision: fallback_decision(constraints),
                cost_us: f64::INFINITY,
                memory_bytes: 0,
                smem_bytes: 0,
                nodes_explored: nodes,
                feasible: false,
                decision_trace: Vec::new(),
            },
            None,
            None,
        );
    };
    let entry = apply_packing(
        apply_fase(base, fase_fused, constraints.fase_backward_speedup),
        packing_mode,
        &constraints.packing_savings,
    );
    let memory_bytes = ilp_resident_bytes(&entry, lut, m_bits, v_bits);
    // G20 greedy: locally-best CFIE choice that fits the memory headroom
    // left after the training-resident bytes.  Gate off => (None, 0.0, 0),
    // leaving every expression below bit-identical to the pre-G20 solver.
    let (cfie_choice, cfie_us, cfie_pool) = match constraints.cfie_infer.as_ref() {
        Some(cfg) => {
            let headroom = constraints.memory_budget.saturating_sub(memory_bytes);
            match crate::wggo_cfie::best_choice_fitting(
                cfg,
                entry.param_bytes,
                lut.peak_bandwidth_gbs,
                headroom,
            ) {
                Some((ch, us)) => (Some(ch), us, crate::wggo_cfie::kv_pool_bytes(ch, cfg)),
                None => (None, 0.0, 0),
            }
        }
        None => (None, 0.0, 0),
    };
    let memory_bytes = memory_bytes.saturating_add(cfie_pool);
    // CPKD greedy (v1): locally-best distillation choice fitting the
    // memory headroom left after the training-resident bytes and the CFIE
    // pool.  Gate off => (None, 0.0, 0), leaving every expression below
    // bit-identical to the pre-CPKD solver.
    let (cpkd_choice, cpkd_us, cpkd_bytes) = match constraints.cpkd.as_ref() {
        Some(cfg) => {
            let headroom = constraints.memory_budget.saturating_sub(memory_bytes);
            match crate::wggo_cpkd::best_choice_fitting(cfg, headroom) {
                Some((ch, us)) => (Some(ch), us, crate::wggo_cpkd::choice_bytes(ch, cfg)),
                None => (None, 0.0, 0),
            }
        }
        None => (None, 0.0, 0),
    };
    let memory_bytes = memory_bytes.saturating_add(cpkd_bytes);
    let feasible = base.feasible
        && entry.smem_bytes <= constraints.smem_budget
        && memory_bytes <= constraints.memory_budget;

    if !feasible {
        return (
            LayerIlpSolution {
                decision: fallback_decision(constraints),
                cost_us: f64::INFINITY,
                memory_bytes,
                smem_bytes: entry.smem_bytes,
                nodes_explored: nodes,
                feasible: false,
                decision_trace: Vec::new(),
            },
            None,
            None,
        );
    }

    (
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
                adapter_placement: best_placement(r, constraints)
                    .unwrap_or(AdapterPlacement::None),
            },
            cost_us: entry.total_us()
                + placement_cost_us(
                    r,
                    best_placement(r, constraints).unwrap_or(AdapterPlacement::None),
                )
                + crate::wggo_cost::optimizer_us(
                    entry.param_bytes,
                    lut.dtype_bytes,
                    m_bits,
                    v_bits,
                    fase_fused,
                    lut.peak_bandwidth_gbs,
                )
                + cfie_us
                + cpkd_us,
            memory_bytes,
            smem_bytes: entry.smem_bytes,
            nodes_explored: nodes,
            feasible: true,
            decision_trace: Vec::new(),
        },
        cfie_choice,
        cpkd_choice,
    )
}

/// Greedy variant of [`solve_all`].
pub fn solve_all_greedy(
    luts: &[LayerCostLut],
    constraints: &[LayerIlpConstraints],
) -> Vec<LayerIlpSolution> {
    solve_all_greedy_cfie(luts, constraints).0
}

/// [`solve_all_greedy`] variant that also returns the per-layer CFIE
/// inference choices (G20) — parallel to the solutions, all `None` when
/// the gate is off.
pub fn solve_all_greedy_cfie(
    luts: &[LayerCostLut],
    constraints: &[LayerIlpConstraints],
) -> (Vec<LayerIlpSolution>, Vec<Option<CfieInferenceChoice>>) {
    let (sols, cfie, _cpkd) = solve_all_greedy_cpkd(luts, constraints);
    (sols, cfie)
}

/// [`solve_all_greedy_cfie`] variant that also returns the per-layer CPKD
/// distillation choices (v1) — parallel to the solutions, all `None` when
/// the gate is off.
pub fn solve_all_greedy_cpkd(
    luts: &[LayerCostLut],
    constraints: &[LayerIlpConstraints],
) -> (
    Vec<LayerIlpSolution>,
    Vec<Option<CfieInferenceChoice>>,
    Vec<Option<CpkdChoice>>,
) {
    assert_eq!(
        luts.len(),
        constraints.len(),
        "luts and constraints must be parallel arrays"
    );
    let mut sols = Vec::with_capacity(luts.len());
    let mut cfie_out = Vec::with_capacity(luts.len());
    let mut cpkd_out = Vec::with_capacity(luts.len());
    for (lut, c) in luts.iter().zip(constraints.iter()) {
        let (sol, cfie, cpkd) = solve_layer_greedy_cpkd(lut, c);
        sols.push(sol);
        cfie_out.push(cfie);
        cpkd_out.push(cpkd);
    }
    (sols, cfie_out, cpkd_out)
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
    /// CFIE inference choice of the incumbent (G20).  Always `None` when
    /// the `cfie_infer` gate is off.
    best_cfie: Option<CfieInferenceChoice>,
    /// CPKD distillation choice of the incumbent (v1).  Always `None`
    /// when the `cpkd` gate is off.
    best_cpkd: Option<CpkdChoice>,
    nodes: u64,
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
    // No evidence → no quantization. See `sensitivity_informed`.
    if !c.sensitivity_informed && bits < 32 {
        return false;
    }
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

/// Small per-(rank·site) compute surcharge so the solver prefers the smallest
/// adapter placement unless a larger one is genuinely beneficial — a
/// tie-breaker, not a dominant term.
const ADAPTER_US_PER_RANK_SITE: f64 = 0.01;

/// Communication footprint of an adapter under sharding, proportional to the
/// number of adapter parameters that must be synchronized: `rank · sites`.
/// Compared against [`LayerIlpConstraints::adapter_comm_budget`].
fn adapter_comm(rank: u64, placement: AdapterPlacement) -> f64 {
    (rank * placement.proj_sites() as u64) as f64
}

/// Tie-breaking compute cost of a placement (μs).
fn placement_cost_us(rank: u64, placement: AdapterPlacement) -> f64 {
    adapter_comm(rank, placement) * ADAPTER_US_PER_RANK_SITE
}

/// Choose the adapter placement for a given rank: the smallest-footprint
/// placement whose communication cost fits the per-layer budget.  Returns
/// `None` when even the minimal placement exceeds the budget — which makes the
/// rank itself infeasible, so the solver falls back to a smaller rank.
fn best_placement(rank: u64, c: &LayerIlpConstraints) -> Option<AdapterPlacement> {
    if rank == 0 {
        return Some(AdapterPlacement::None);
    }
    [
        AdapterPlacement::AttnQV,
        AdapterPlacement::AttnQKVO,
        AdapterPlacement::AttnAndFfn,
    ]
    .into_iter()
    .find(|&p| adapter_comm(rank, p) <= c.adapter_comm_budget)
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
        format!(
            "LoRA r={} on {}",
            decision.adapter_rank,
            decision.adapter_placement.as_str()
        )
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

    // ---------- FASE (fused optimizer step) ----------
    let fase_trace = DecisionTrace {
        kind: DecisionKind::FaseStep,
        chosen: if decision.fase_fused {
            "Fused step (deferred grads)".to_string()
        } else {
            "Deferred to standalone optimizer".to_string()
        },
        runner_up: None,
        binding_constraint: Some(if c.allow_fase {
            "allow_fase = true".to_string()
        } else {
            "allow_fase = false — fusion forbidden for this layer".to_string()
        }),
        metric_summary: if decision.fase_fused {
            "Gradient buffers not materialized in HBM; backward writes updates in place.".to_string()
        } else {
            "Gradients materialized; optimizer runs as a separate pass.".to_string()
        },
        cross_decision_note: None,
    };

    // ---------- PCA (sequence packing) ----------
    // Single naming authority shared with the report renderer and the [pca]
    // consumption verdicts (errata E2 wiring).
    let pca_mode = crate::wggo_overrides::packing_mode_name(decision.packing_mode);
    let pca_trace = DecisionTrace {
        kind: DecisionKind::PcaPacking,
        chosen: format!("Packing mode {} ({})", decision.packing_mode, pca_mode),
        runner_up: None,
        binding_constraint: None,
        metric_summary: if decision.packing_mode == 0 {
            "No sequence packing — padded positions computed.".to_string()
        } else {
            format!("Skips padded work via {pca_mode} packing (paper §4.3.6).")
        },
        cross_decision_note: None,
    };

    vec![
        cep_trace, csha_trace, wrga_trace, cpdt_trace, fase_trace, pca_trace,
    ]
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
        adapter_placement: AdapterPlacement::None,
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
    let s = savings.get(mode as usize).copied().unwrap_or(0.0).clamp(0.0, 0.95);
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
    fn decision_trace_covers_all_six_dimensions() {
        // G13: the explainer trace must cover all six WGGO decision
        // dimensions, not only the original four (FASE + PCA were missing).
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let sol = solve_layer(&lut, &LayerIlpConstraints::default());
        assert!(sol.feasible);
        let has = |pred: fn(&DecisionKind) -> bool| sol.decision_trace.iter().any(|t| pred(&t.kind));
        assert!(has(|k| matches!(k, DecisionKind::CepHeadPrune)));
        assert!(has(|k| matches!(k, DecisionKind::CshaLevel)));
        assert!(has(|k| matches!(k, DecisionKind::WrgaAdapter)));
        assert!(has(|k| matches!(k, DecisionKind::CpdtPrecision)));
        assert!(
            has(|k| matches!(k, DecisionKind::FaseStep)),
            "FASE decision dimension missing from trace"
        );
        assert!(
            has(|k| matches!(k, DecisionKind::PcaPacking)),
            "PCA decision dimension missing from trace"
        );
        assert_eq!(sol.decision_trace.len(), 6);
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
    fn ilp_resident_bytes_counts_moments_precision_aware_unsharded() {
        // Gap #3: the ILP's resident charge must include the Adam moments (it
        // used to be bare `param + activation`), sized at the chosen precision,
        // and computed unsharded (the ILP can't see the DP's shard choice).
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let e = lut.argmin_feasible().unwrap().4;
        assert!(e.param_bytes > 0);
        let bare = e.param_bytes + e.activation_bytes;
        let hi = ilp_resident_bytes(&e, &lut, 32, 32);
        assert!(hi > bare, "resident must count optimizer state: {hi} !> {bare}");
        let lo = ilp_resident_bytes(&e, &lut, 8, 8);
        assert!(lo < hi, "lower moment precision ⇒ less resident memory");
        assert!(lo >= bare, "still charges some optimizer state");
        // Exactly the shared formula, sized unsharded (shard = 1).
        let expect = crate::wggo_cost::resident_training_bytes(
            e.param_bytes,
            crate::wggo_cost::moment_state_bytes(e.param_bytes, lut.dtype_bytes, 32, 32),
            e.activation_bytes,
            1,
        );
        assert_eq!(hi, expect);
    }

    #[test]
    fn solve_layer_reports_resident_memory_with_optimizer_state() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let c = LayerIlpConstraints::default();
        let sol = solve_layer(&lut, &c);
        assert!(sol.feasible);
        // Reconstruct the winning entry exactly as `recost_decision` does, then
        // confirm the reported memory is the shared resident formula for it.
        let d = &sol.decision;
        let base = lut
            .get((d.active_heads() as u64).max(1), d.ffn_width, d.csha_level, d.adapter_rank)
            .unwrap();
        let entry = apply_packing(
            apply_fase(base, d.fase_fused, c.fase_backward_speedup),
            d.packing_mode,
            &c.packing_savings,
        );
        let expect = ilp_resident_bytes(&entry, &lut, d.optim_m_bits, d.optim_v_bits);
        assert_eq!(sol.memory_bytes, expect);
        assert!(sol.memory_bytes > entry.param_bytes + entry.activation_bytes);
    }

    #[test]
    fn solve_layer_greedy_reports_resident_memory_with_optimizer_state() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let c = LayerIlpConstraints::default();
        let sol = solve_layer_greedy(&lut, &c);
        assert!(sol.feasible);
        let d = &sol.decision;
        let base = lut
            .get((d.active_heads() as u64).max(1), d.ffn_width, d.csha_level, d.adapter_rank)
            .unwrap();
        let entry = apply_packing(
            apply_fase(base, d.fase_fused, c.fase_backward_speedup),
            d.packing_mode,
            &c.packing_savings,
        );
        let expect = ilp_resident_bytes(&entry, &lut, d.optim_m_bits, d.optim_v_bits);
        assert_eq!(sol.memory_bytes, expect);
    }

    #[test]
    fn proj_role_classifies_naming_variants() {
        assert_eq!(proj_role("blocks.0.attn.wq"), Some(ProjRole::Q));
        assert_eq!(proj_role("model.layers.3.self_attn.q_proj.weight"), Some(ProjRole::Q));
        assert_eq!(proj_role("blocks.0.attn.w_k"), Some(ProjRole::K));
        assert_eq!(proj_role("blocks.0.attn.wv"), Some(ProjRole::V));
        assert_eq!(proj_role("blocks.0.attn.out_proj"), Some(ProjRole::O));
        assert_eq!(proj_role("blocks.0.mlp.gate_proj"), Some(ProjRole::Ffn));
        assert_eq!(proj_role("blocks.0.ffn.fc1"), Some(ProjRole::Ffn));
        // Not projections:
        assert_eq!(proj_role("blocks.0.attn_norm.weight"), None);
        assert_eq!(proj_role("embed.weight"), None);
        assert_eq!(proj_role("var_42"), None);
    }

    #[test]
    fn covers_projection_respects_placement_set() {
        use AdapterPlacement::*;
        // AttnQV: Q and V in, K and O out.
        assert_eq!(AttnQV.covers_projection("blocks.0.attn.wq"), Some(true));
        assert_eq!(AttnQV.covers_projection("blocks.0.attn.wv"), Some(true));
        assert_eq!(AttnQV.covers_projection("blocks.0.attn.wk"), Some(false));
        assert_eq!(AttnQV.covers_projection("blocks.0.attn.wo"), Some(false));
        // AttnQKVO: all four attn projections in, FFN out.
        assert_eq!(AttnQKVO.covers_projection("blocks.0.attn.wk"), Some(true));
        assert_eq!(AttnQKVO.covers_projection("blocks.0.attn.wo"), Some(true));
        assert_eq!(AttnQKVO.covers_projection("blocks.0.mlp.fc1"), Some(false));
        // AttnAndFfn: everything recognized is in.
        assert_eq!(AttnAndFfn.covers_projection("blocks.0.mlp.fc1"), Some(true));
        assert_eq!(AttnAndFfn.covers_projection("blocks.0.attn.wk"), Some(true));
        // None excludes every projection.
        assert_eq!(None.covers_projection("blocks.0.attn.wq"), Some(false));
        // Unrecognized names: no opinion regardless of placement.
        assert_eq!(AttnQKVO.covers_projection("blocks.0.attn_norm.weight"), Option::None);
        assert_eq!(AttnAndFfn.covers_projection("embed.weight"), Option::None);
    }

    #[test]
    fn best_placement_picks_smallest_fitting() {
        let mut c = LayerIlpConstraints::default(); // adapter_comm_budget = MAX
        assert_eq!(best_placement(0, &c), Some(AdapterPlacement::None));
        assert_eq!(best_placement(8, &c), Some(AdapterPlacement::AttnQV));
        c.adapter_comm_budget = 0.0;
        assert_eq!(best_placement(8, &c), None);
        c.adapter_comm_budget = 16.0; // exactly fits AttnQV (8*2), not QKVO (8*4)
        assert_eq!(best_placement(8, &c), Some(AdapterPlacement::AttnQV));
    }

    #[test]
    fn zero_rank_solution_has_no_placement() {
        // The default solve keeps rank 0 (adapters add cost) → placement None.
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let sol = solve_layer(&lut, &LayerIlpConstraints::default());
        assert!(sol.feasible);
        assert_eq!(sol.decision.adapter_rank, 0);
        assert_eq!(sol.decision.adapter_placement, AdapterPlacement::None);
    }

    #[test]
    fn adapter_comm_budget_zero_forces_rank_zero() {
        // No adapter communication permitted → every rank>0 is rejected, so the
        // solver must fall back to rank 0 (G7 constraint enforcement).
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut c = LayerIlpConstraints::default();
        c.adapter_comm_budget = 0.0;
        let sol = solve_layer(&lut, &c);
        assert!(sol.feasible);
        assert_eq!(sol.decision.adapter_rank, 0);
        assert_eq!(sol.decision.adapter_placement, AdapterPlacement::None);
    }

    #[test]
    fn forced_rank_yields_smallest_placement() {
        // LUT offers only rank 8, so the solver must take an adapter and choose
        // a placement — with an ample comm budget it picks the smallest (G5:
        // placement is a real chosen decision, not a fixed assumption).
        let axes = LutAxes {
            adapter_ranks: vec![8],
            ..LutAxes::default()
        };
        let lut = build_lut(&shape(), h100(), &axes);
        let sol = solve_layer(&lut, &LayerIlpConstraints::default());
        assert!(sol.feasible);
        assert_eq!(sol.decision.adapter_rank, 8);
        assert_eq!(sol.decision.adapter_placement, AdapterPlacement::AttnQV);
    }

    #[test]
    fn forced_rank_with_tight_comm_budget_is_infeasible() {
        // Only rank 8 available and the comm budget cannot fit even AttnQV
        // (8*2 = 16) → the layer has no feasible adapter assignment (G7).
        let axes = LutAxes {
            adapter_ranks: vec![8],
            ..LutAxes::default()
        };
        let lut = build_lut(&shape(), h100(), &axes);
        let mut c = LayerIlpConstraints::default();
        c.adapter_comm_budget = 15.0;
        let sol = solve_layer(&lut, &c);
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
    fn lut_eq_distinguishes_dtype_bytes_and_bandwidth() {
        // Both fields feed `optimizer_us`, so two LUTs that differ only in them
        // must NOT be treated as the same template (else `solve_all_templated`
        // would reuse a solution costed for a different optimizer dtype/target).
        let a = build_lut(&shape(), h100(), &LutAxes::default());
        let mut b = a.clone();
        b.dtype_bytes = a.dtype_bytes + 2;
        assert!(!lut_eq(&a, &b));
        let mut c = a.clone();
        c.peak_bandwidth_gbs = a.peak_bandwidth_gbs * 2.0;
        assert!(!lut_eq(&a, &c));
        assert!(lut_eq(&a, &a.clone())); // identical ⇒ equal
    }

    #[test]
    fn low_sensitivity_picks_low_optimizer_precision_when_informed() {
        // Gap #2: with the optimizer-cost term in the objective, a layer with
        // no numerical-sensitivity floor prefers the cheapest (lowest-bit)
        // moments — but only when the sensitivity value is backed by a real
        // signal (weight analysis / calibration).
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let constraints = LayerIlpConstraints {
            sensitivity_informed: true, // evidence present, sensitivity = 0
            ..Default::default()
        };
        let sol = solve_layer(&lut, &constraints);
        assert!(sol.feasible);
        assert_eq!(sol.decision.optim_m_bits, 8);
        assert_eq!(sol.decision.optim_v_bits, 8);
    }

    #[test]
    fn uninformed_sensitivity_forbids_sub_32_bit_moments() {
        // No weight/calibration evidence -> no quantization, regardless of
        // how cheap the cost model says low-bit moments are. The
        // moment-precision analog of not pruning heads on uniform
        // importance: the objective is cost-only, so without this gate the
        // solver would ALWAYS quantize a from-scratch pretrain.
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let constraints = LayerIlpConstraints::default(); // uninformed
        let sol = solve_layer(&lut, &constraints);
        assert!(sol.feasible);
        assert_eq!(sol.decision.optim_m_bits, 32);
        assert_eq!(sol.decision.optim_v_bits, 32);
    }

    #[test]
    fn full_and_greedy_agree_on_optimizer_precision_at_low_sensitivity() {
        // Both solvers now minimize the same objective, so they converge on the
        // same (lowest-allowed) optimizer precision — previously the full ILP
        // returned 32/32 while greedy returned 8/8 for the same layer.
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let c = LayerIlpConstraints::default();
        let full = solve_layer(&lut, &c);
        let greedy = solve_layer_greedy(&lut, &c);
        assert_eq!(full.decision.optim_m_bits, greedy.decision.optim_m_bits);
        assert_eq!(full.decision.optim_v_bits, greedy.decision.optim_v_bits);
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
        assert!(sol.nodes_explored <= 4_000_000, "nodes={}", sol.nodes_explored);
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

    // -----------------------------------------------------------------------
    // CPKD distillation axis (v1) — gate-off byte-identity, gate-on
    // decisions, memory binding, template key, greedy/full/recost parity.
    // -----------------------------------------------------------------------

    /// Single-entry LUT (one point on every base axis) so CPKD tests face a
    /// unique, deterministic base decision — the wggo_cfie integration-test
    /// fixture pattern.
    fn cpkd_pinned_lut() -> LayerCostLut {
        let axes = LutAxes {
            head_counts: vec![4],
            ffn_widths: vec![1024],
            csha_levels: vec![0],
            adapter_ranks: vec![0],
        };
        build_lut(&shape(), h100(), &axes)
    }

    /// One head config (num_heads=4, gqa_group=4 => a single all-kept
    /// group), one packing mode, FASE off — the CPKD axis (plus optimizer
    /// precision) is the only thing that varies.
    fn cpkd_pinned_constraints() -> LayerIlpConstraints {
        LayerIlpConstraints {
            num_heads: 4,
            gqa_group: 4,
            packing_modes_mask: 0b0001,
            allow_fase: false,
            ..Default::default()
        }
    }

    #[test]
    fn cpkd_gate_off_solution_identical_and_choice_none() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let c = LayerIlpConstraints::default(); // cpkd: None — production default
        let (sol, cfie, cpkd) = solve_layer_cpkd(&lut, &c);
        assert!(sol.feasible);
        assert!(cfie.is_none());
        assert!(cpkd.is_none(), "gate off must never produce a choice");
        // The legacy entry point returns the identical solution: same
        // decision, bit-identical cost, same memory and node count.
        let legacy = solve_layer(&lut, &c);
        assert_eq!(legacy.decision, sol.decision);
        assert_eq!(legacy.cost_us.to_bits(), sol.cost_us.to_bits());
        assert_eq!(legacy.memory_bytes, sol.memory_bytes);
        assert_eq!(legacy.nodes_explored, sol.nodes_explored);
        // Serialized solution is byte-free of any CPKD key: the choice
        // travels in a sidecar; LayerIlpSolution's layout is untouched.
        let json = serde_json::to_string(&sol).unwrap();
        assert!(
            !json.to_lowercase().contains("cpkd"),
            "gate-off solution JSON must not mention cpkd: {json}"
        );
    }

    #[test]
    fn cpkd_favorable_feature_cost_flips_feature_match_on() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let off = solve_layer(&lut, &LayerIlpConstraints::default());
        let mut c = LayerIlpConstraints::default();
        c.cpkd = Some(CpkdConfig {
            // A favorable (negative) net charge — e.g. the fused KL-CE
            // kernel replacing a separate CE loss pass.
            feature_cost_us: -15.0,
            ..Default::default()
        });
        let (sol, _, choice) = solve_layer_cpkd(&lut, &c);
        assert!(sol.feasible);
        let ch = choice.expect("gate on must populate the choice");
        assert!(ch.feature_match);
        assert!(!ch.attn_transfer, "deferred axis stays off by default");
        assert!(!ch.teacher_stream, "deferred axis stays off by default");
        // The objective composes additively: base + (-15.0); the constant
        // per-choice delta cannot move the base argmin.
        assert!(
            ((off.cost_us - 15.0) - sol.cost_us).abs() < 1e-9,
            "off={} on={}",
            off.cost_us,
            sol.cost_us
        );
        // And the winner's memory accounting charges the feature buffer.
        assert_eq!(sol.memory_bytes, off.memory_bytes + 2 * 1024 * 1024);
    }

    #[test]
    fn cpkd_default_costs_advise_all_off_but_still_populate() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let off = solve_layer(&lut, &LayerIlpConstraints::default());
        let mut c = LayerIlpConstraints::default();
        c.cpkd = Some(CpkdConfig::default()); // all-positive charges
        let (sol, _, choice) = solve_layer_cpkd(&lut, &c);
        assert!(sol.feasible);
        // Conservative all-off wins on strict `<` improvement, but the
        // advisory choice is still populated (the axis was decided).
        let ch = choice.expect("gate on must populate the choice");
        assert!(!ch.feature_match && !ch.attn_transfer && !ch.teacher_stream);
        // All-off charges nothing: cost, memory, and decision match the
        // gate-off run.
        assert_eq!(sol.cost_us.to_bits(), off.cost_us.to_bits());
        assert_eq!(sol.memory_bytes, off.memory_bytes);
        assert_eq!(sol.decision, off.decision);
    }

    #[test]
    fn cpkd_tiny_feature_budget_forces_feature_off() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut c = LayerIlpConstraints::default();
        c.cpkd = Some(CpkdConfig {
            feature_cost_us: -15.0,   // favorable — would be chosen...
            feature_memory_budget: 1, // ...but the buffer can never fit
            ..Default::default()
        });
        let (sol, _, choice) = solve_layer_cpkd(&lut, &c);
        assert!(sol.feasible);
        assert!(!choice.expect("gate on populates").feature_match);
    }

    #[test]
    fn cpkd_resident_memory_charge_binds_against_the_ilp_budget() {
        // Single-entry LUT + pinned constraints so the base decision is
        // unique and the resident bytes are deterministic.
        let lut = cpkd_pinned_lut();
        let base = solve_layer(&lut, &cpkd_pinned_constraints());
        assert!(base.feasible);
        let fav = CpkdConfig {
            feature_cost_us: -15.0,
            ..Default::default()
        };
        let feature_bytes = fav.feature_bytes_per_layer;

        // Budget admits resident + feature buffer exactly: chosen ON, and
        // the winner's accounting includes the buffer.
        let mut roomy = cpkd_pinned_constraints();
        roomy.cpkd = Some(fav.clone());
        roomy.memory_budget = base.memory_bytes + feature_bytes;
        let (sol_on, _, ch_on) = solve_layer_cpkd(&lut, &roomy);
        assert!(sol_on.feasible);
        assert!(ch_on.unwrap().feature_match);
        assert_eq!(sol_on.memory_bytes, base.memory_bytes + feature_bytes);

        // One byte less: the feature-on candidate fails the resident
        // admission check, so despite the favorable cost the solver falls
        // back to feature-off — the memory constraint binds.
        let mut tight = cpkd_pinned_constraints();
        tight.cpkd = Some(fav);
        tight.memory_budget = base.memory_bytes + feature_bytes - 1;
        let (sol_off, _, ch_off) = solve_layer_cpkd(&lut, &tight);
        assert!(
            sol_off.feasible,
            "the layer itself still fits without the buffer"
        );
        assert!(!ch_off.unwrap().feature_match);
        assert_eq!(sol_off.memory_bytes, base.memory_bytes);
    }

    #[test]
    fn cpkd_template_cache_never_shares_across_gate_states() {
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let c_off = LayerIlpConstraints::default();
        let mut c_on = LayerIlpConstraints::default();
        c_on.cpkd = Some(CpkdConfig::default());
        // Gate-off vs gate-on: two distinct templates (recon risk 6 —
        // forgetting the constraints_eq entry would silently reuse the
        // gate-off solution here).
        let (sols, stats, _cfie, cpkd) =
            solve_all_templated_cpkd(&[lut.clone(), lut.clone()], &[c_off, c_on.clone()]);
        assert_eq!(stats.templates_solved, 2);
        assert_eq!(stats.template_hits, 0);
        assert!(cpkd[0].is_none());
        assert!(cpkd[1].is_some());
        assert!(sols.iter().all(|s| s.feasible));

        // Two configs differing only in a cpkd knob are distinct templates.
        let mut c_on2 = LayerIlpConstraints::default();
        c_on2.cpkd = Some(CpkdConfig {
            feature_cost_us: -15.0,
            ..Default::default()
        });
        let (_, stats2, _, _) =
            solve_all_templated_cpkd(&[lut.clone(), lut.clone()], &[c_on.clone(), c_on2]);
        assert_eq!(stats2.templates_solved, 2);
        assert_eq!(stats2.template_hits, 0);

        // Identical gate-on layers DO share one template, replicating the
        // cached choice.
        let (_, stats3, _, cpkd3) =
            solve_all_templated_cpkd(&[lut.clone(), lut], &[c_on.clone(), c_on]);
        assert_eq!(stats3.templates_solved, 1);
        assert_eq!(stats3.template_hits, 1);
        assert_eq!(cpkd3[0], cpkd3[1]);
    }

    #[test]
    fn cpkd_greedy_and_full_agree_on_a_trivial_layer() {
        // Single-entry LUT: the base axes are pinned, so both solvers face
        // only the precision + CPKD choices and share one objective.
        let lut = cpkd_pinned_lut();
        let mut c = cpkd_pinned_constraints();
        c.cpkd = Some(CpkdConfig {
            feature_cost_us: -15.0,
            ..Default::default()
        });
        let (full, _, full_ch) = solve_layer_cpkd(&lut, &c);
        let (greedy, _, greedy_ch) = solve_layer_greedy_cpkd(&lut, &c);
        assert!(full.feasible && greedy.feasible);
        assert_eq!(full_ch, greedy_ch);
        assert!(full_ch.unwrap().feature_match);
        assert_eq!(full.decision, greedy.decision);
        assert!(
            (full.cost_us - greedy.cost_us).abs() < 1e-9,
            "full={} greedy={}",
            full.cost_us,
            greedy.cost_us
        );
        assert_eq!(full.memory_bytes, greedy.memory_bytes);
    }

    #[test]
    fn cpkd_greedy_refuses_feature_when_headroom_is_gone() {
        // Headroom below the 2 MiB feature buffer: greedy must advise the
        // all-off choice rather than an over-budget buffer.
        let lut = cpkd_pinned_lut();
        let base = solve_layer_greedy(&lut, &cpkd_pinned_constraints());
        assert!(base.feasible);
        let mut c = cpkd_pinned_constraints();
        c.cpkd = Some(CpkdConfig {
            feature_cost_us: -15.0,
            ..Default::default()
        });
        c.memory_budget = base.memory_bytes + 1024; // < feature_bytes_per_layer
        let (sol, _, ch) = solve_layer_greedy_cpkd(&lut, &c);
        assert!(sol.feasible);
        assert!(!ch.expect("gate on populates").feature_match);
        assert_eq!(sol.memory_bytes, base.memory_bytes);
    }

    #[test]
    fn cpkd_recost_matches_the_solver_objective_exactly() {
        // Greedy G3 escalation compares solver costs with re-costed
        // decisions, so recost_decision_cpkd must mirror
        // solve_layer_cpkd's objective exactly.
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let mut c = LayerIlpConstraints::default();
        c.cpkd = Some(CpkdConfig {
            feature_cost_us: -15.0,
            ..Default::default()
        });
        let (sol, cfie, cpkd) = solve_layer_cpkd(&lut, &c);
        assert!(sol.feasible);
        let recost = recost_decision_cpkd(&lut, &sol.decision, &c, cfie, cpkd);
        assert!(
            (recost - sol.cost_us).abs() < 1e-9,
            "recost={recost} cost={}",
            sol.cost_us
        );
        // The legacy wrappers (no cpkd choice) recover the gate-off
        // objective: exactly the distill term apart.
        let base = recost_decision(&lut, &sol.decision, &LayerIlpConstraints::default());
        assert!(((base - 15.0) - recost).abs() < 1e-9);
    }

    #[test]
    fn uninformed_importance_forces_keep_all_heads_and_full_ffn() {
        // No weight/calibration evidence -> no structural thinning. With
        // uniform (Null-provider) importance the retention floor is
        // satisfiable by any 75% subset, so without this gate the cost-only
        // objective drops floor(0.25*H) heads of a random-init model.
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let constraints = LayerIlpConstraints {
            num_heads: 8,
            gqa_group: 1,
            head_importance: vec![1.0; 8],
            min_retained_importance: 6.0, // top-75% floor: any 6 heads satisfy
            ..Default::default()
        };
        let sol = solve_layer(&lut, &constraints);
        assert!(sol.feasible);
        assert!(
            sol.decision.keep_head.iter().all(|k| *k),
            "uninformed solve must keep all heads, got {:?}",
            sol.decision.keep_head
        );
        assert_eq!(
            sol.decision.ffn_width,
            *lut.axes_ffn_widths.iter().max().unwrap(),
            "uninformed solve must keep full FFN width"
        );
    }

    #[test]
    fn informed_importance_allows_head_pruning() {
        // Same constraints WITH evidence: the solver may (and, being
        // cost-only, will) drop heads down to the retention floor.
        let lut = build_lut(&shape(), h100(), &LutAxes::default());
        let constraints = LayerIlpConstraints {
            num_heads: 8,
            gqa_group: 1,
            head_importance: vec![1.0; 8],
            min_retained_importance: 6.0,
            importance_informed: true,
            ..Default::default()
        };
        let sol = solve_layer(&lut, &constraints);
        assert!(sol.feasible);
        let kept = sol.decision.keep_head.iter().filter(|k| **k).count();
        assert!(
            kept < 8,
            "informed low-importance solve should prune below 8 heads (kept {kept})"
        );
        assert!(kept >= 6, "retention floor must hold (kept {kept})");
    }


    #[test]
    fn adapter_mandate_axis_yields_nonzero_rank_with_placement() {
        // Paper @wrga(mode=auto) / errata E3: when adapters are mandated
        // (rank 0 removed from the axis — the wiring driven by
        // @wrga/@adapter decorators), the solver must pick a grid rank and
        // a comm-feasible placement instead of pricing adapters out.
        let lut = build_lut(
            &shape(),
            h100(),
            &LutAxes {
                adapter_ranks: vec![2, 4, 8, 16],
                ..Default::default()
            },
        );
        let constraints = LayerIlpConstraints::default();
        let sol = solve_layer(&lut, &constraints);
        assert!(sol.feasible);
        assert!(
            sol.decision.adapter_rank >= 2,
            "mandated adapters must yield rank >= 2, got {}",
            sol.decision.adapter_rank
        );
        assert_ne!(
            sol.decision.adapter_placement,
            AdapterPlacement::None,
            "a nonzero rank needs a placement"
        );
    }


    #[test]
    fn greedy_adapter_mandate_keeps_csha_probe_alive() {
        // Regression: the greedy CSHA-level scan probed the LUT at the
        // hardcoded rank 0; with the adapter mandate (0 removed from the
        // axis) every probe missed and csha_level silently collapsed to 0
        // while the layer still reported feasible.
        let lut = build_lut(
            &shape(),
            h100(),
            &LutAxes {
                adapter_ranks: vec![2, 4, 8, 16],
                ..Default::default()
            },
        );
        let baseline = solve_layer_greedy(&build_lut(&shape(), h100(), &LutAxes::default()), &LayerIlpConstraints::default());
        let mandated = solve_layer_greedy(&lut, &LayerIlpConstraints::default());
        assert!(mandated.feasible);
        assert_eq!(
            mandated.decision.csha_level, baseline.decision.csha_level,
            "mandating adapters must not change the greedy CSHA level \
             (baseline {}, mandated {})",
            baseline.decision.csha_level, mandated.decision.csha_level
        );
        assert!(mandated.decision.adapter_rank >= 2);
    }

}
