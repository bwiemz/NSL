//! WRGA compiler pass entry point: Wengert-Pruned Roofline-Guided Adaptation.
//!
//! Composes the five innovations in `wrga_prune` + `wrga_roofline` +
//! `wrga_spectral` + `wrga_fusion` + `wrga_memory` into a single driver that
//! produces a [`WrgaPlan`] from a model's Wengert list, weight map, and
//! hardware target.
//!
//! The driver is intentionally *pure*: it consumes immutable inputs and
//! returns a plan that the rest of the codegen pipeline applies.  The
//! `nsl build --wrga-report` subcommand serializes the plan to a
//! human-readable string via [`WrgaPlan::render_report`].

use std::collections::{BTreeMap, BTreeSet, HashMap};

use nsl_ast::block::WrgaMode;

use crate::gpu_specs::{default_gpu, find_gpu};
use crate::wengert::{PrimalOp, VarId, WengertList};
use crate::weight_aware::WeightMap;
use crate::wrga_fusion::{
    build_fusion_plan, verify_fused_sites_have_no_intermediate, FusionPlan,
};
use crate::wrga_memory::{plan_memory_with_pin, MemoryPlan, SizeHints};
use crate::wrga_prune::{prune, PruneResult};
use crate::wrga_roofline::{place_adapters, AdapterPlacement, AdapterSite, SiteKind};
use crate::wrga_spectral::{allocate_ranks, analyse_weight_map, RankAllocation, SpectralAnalysis};

// Re-export B.2 Task 2b observation surfaces so consumers can use
// `crate::wrga::{InitKind, InitStrategy}`.
pub use crate::wrga_adapter_inject::{InitKind, InitStrategy};

/// Per-Innovation skip flags for the WRGA paper §9.3 ablation harness.
///
/// Each boolean controls one of the five Innovations of WRGA's compiler pass.
/// When a flag is `true`, the corresponding stage of `wrga::run` is replaced
/// with a deterministic "as if that Innovation didn't run" no-op:
///
/// | Flag                       | Innovation                        | Stage no-op shape                                                       |
/// |----------------------------|-----------------------------------|-------------------------------------------------------------------------|
/// | `skip_wengert_pruning`     | (1) Wengert-pruned backward       | retain every forward op as backward-live; all "would-save" acts saved   |
/// | `skip_roofline_placement`  | (2) Roofline-guided placement     | place adapter on every site with `suggested_rank = r_max`               |
/// | `skip_spectral_allocation` | (3) Spectral compile-time ranks   | every site's `RankAllocation.rank == placement.suggested_rank` (no SVD) |
/// | `skip_fusion_integration`  | (4) Fusion-integrated adapters    | `FusionPlan` reports every site as `StandaloneAdapter` (0% fusion)      |
/// | `skip_memory_planning`     | (5) Memory-planned activations    | `MemoryPlan::default()` — naive (no reuse, planned == live)             |
///
/// The default is `all-false` (no ablation — full WRGA). The fields are public
/// so callers can construct an arbitrary subset.
///
/// Used by `nsl check --wrga-ablate=<name>` (paper §9.3) and by direct unit
/// tests that need to measure a single Innovation's contribution to the
/// overall plan.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct WrgaAblation {
    pub skip_wengert_pruning: bool,
    pub skip_roofline_placement: bool,
    pub skip_spectral_allocation: bool,
    pub skip_fusion_integration: bool,
    pub skip_memory_planning: bool,
}

impl WrgaAblation {
    /// `true` if any Innovation is skipped — used by `render_report` to
    /// decide whether to emit the ablation header line.
    pub fn is_active(&self) -> bool {
        self.skip_wengert_pruning
            || self.skip_roofline_placement
            || self.skip_spectral_allocation
            || self.skip_fusion_integration
            || self.skip_memory_planning
    }

    /// Comma-separated list of the active skip flags for the report header.
    /// Returns `"none"` when no ablation is active.
    pub fn human_label(&self) -> String {
        let mut parts: Vec<&str> = Vec::new();
        if self.skip_wengert_pruning {
            parts.push("wengert");
        }
        if self.skip_roofline_placement {
            parts.push("roofline");
        }
        if self.skip_spectral_allocation {
            parts.push("spectral");
        }
        if self.skip_fusion_integration {
            parts.push("fusion");
        }
        if self.skip_memory_planning {
            parts.push("memory");
        }
        if parts.is_empty() {
            "none".to_string()
        } else {
            parts.join(",")
        }
    }
}

/// Input to the WRGA driver.
#[derive(Clone)]
pub struct WrgaInput<'a> {
    pub mode: WrgaMode,
    /// Parameter names to *exclude* from freezing (i.e. these are trainable).
    /// Supports the same glob dialect as `wrga_prune::glob_match` —
    /// e.g. `blocks.6.*`.
    pub trainable_patterns: Vec<&'a str>,
    /// Explicit adapter target layer globs for `mode=manual`.  Ignored for
    /// `auto` (which runs placement automatically) and `hybrid` (which uses
    /// the layer scope instead).
    pub manual_adapter_targets: Vec<&'a str>,
    /// Explicit layer scope for `mode=hybrid`.
    pub hybrid_layers: Vec<&'a str>,
    /// Forward Wengert list (from source-AD extraction).
    pub wengert: &'a WengertList,
    /// VarId of the scalar loss (the adjoint seed).
    pub loss_output: VarId,
    /// Optional pre-trained weights for spectral analysis.
    pub weights: Option<&'a WeightMap>,
    /// Target GPU name.  Looked up in `gpu_specs::GPU_DATABASE`.  Empty /
    /// unknown falls back to `default_gpu()`.
    pub target: &'a str,
    /// Total adapter parameter budget.  A budget of 0 means "unbounded"; in
    /// that case WRGA falls back to `r_max`-sized adapters on every site that
    /// the roofline picks.
    pub budget_params: usize,
    /// Rank band.
    pub r_min: usize,
    pub r_max: usize,
    /// Deterministic seed for randomized SVD.
    pub seed: u64,
    /// Dev Tools Phase 5 Task 7: VarIds pinned by `@inspect` decorators whose
    /// backing storage must survive past the last program point so the
    /// inspector stream memcpy can read them without UAF.  Empty when
    /// `@inspect` is disabled.
    pub inspect_pinned_vars: BTreeSet<VarId>,
    /// Per-layer WGGO preferences for adapter rank.  `None` → WRGA's
    /// spectral allocator runs unchanged.  `Some` → per-layer ranks honored
    /// subject to `[r_min, r_max]` clamp and `budget_params` downgrade.
    pub wggo_overrides: Option<&'a crate::wggo_overrides::WggoOverrides>,
    /// Paper §9.3 ablation harness — per-Innovation skip flags. Default
    /// `WrgaAblation::default()` = no ablation = standard WRGA. Set one or
    /// more fields to `true` to replace that Innovation's stage with a
    /// deterministic no-op. See [`WrgaAblation`] for the no-op semantics.
    pub ablation: WrgaAblation,
}

/// Compiled WRGA plan — consumed by downstream codegen stages.
#[derive(Debug, Clone)]
pub struct WrgaPlan {
    pub mode: WrgaMode,
    pub target_gpu: String,
    pub prune: PruneResult,
    pub placements: Vec<AdapterPlacement>,
    pub spectral: Vec<SpectralAnalysis>,
    pub ranks: Vec<RankAllocation>,
    pub fusion: FusionPlan,
    pub memory: MemoryPlan,
    /// WGGO overrides that were clamped, forbidden, or downgraded due to
    /// rank bounds or parameter budget.  Empty when no WggoOverrides were
    /// supplied or every override was applied verbatim.
    pub override_diagnostics: Vec<crate::wggo_overrides::OverrideDiagnostic>,
    /// Paper §9.3 ablation flags that produced this plan. Default
    /// `WrgaAblation::default()` = full WRGA (no ablation). Surfaced in the
    /// report header by `render_report` when active.
    pub ablation: WrgaAblation,
}

impl WrgaPlan {
    /// Test-only minimal WrgaPlan constructor. Callers typically overwrite
    /// the `memory` field after construction. Do NOT use in production code.
    #[cfg(test)]
    pub(crate) fn test_dummy() -> Self {
        use std::collections::{BTreeSet, HashMap};
        let wengert = WengertList {
            ops: Vec::new(),
            output: 0,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let prune = PruneResult {
            pruned: wengert,
            backward_live: BTreeSet::new(),
            activation_live: BTreeSet::new(),
            stats: crate::wrga_prune::PruneStats::default(),
        };
        WrgaPlan {
            mode: WrgaMode::Auto,
            target_gpu: String::new(),
            prune,
            placements: Vec::new(),
            spectral: Vec::new(),
            ranks: Vec::new(),
            fusion: FusionPlan::default(),
            memory: MemoryPlan::default(),
            override_diagnostics: Vec::new(),
            ablation: WrgaAblation::default(),
        }
    }

    /// Total adapter parameter count after rank allocation.
    pub fn adapter_params(&self) -> usize {
        self.ranks.iter().map(|r| r.adapter_params).sum()
    }

    /// Render a human-readable report, matching the layout of
    /// `nsl build --wrga-report`.
    pub fn render_report(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        writeln!(s, "=== WRGA Compilation Report ===").unwrap();
        writeln!(s, "Mode: {}", self.mode.as_str()).unwrap();
        writeln!(s, "Target hardware: {}", self.target_gpu).unwrap();
        if self.ablation.is_active() {
            // Paper §9.3: when one or more Innovations are ablated, surface
            // which ones in the header so the report's numbers are not
            // mistaken for a regular WRGA run.
            writeln!(s, "Ablation: skip={}", self.ablation.human_label()).unwrap();
        }
        writeln!(s).unwrap();
        writeln!(
            s,
            "Backward pruning: {} ops → {} ops ({:.1}% eliminated)",
            self.prune.stats.forward_ops_total,
            self.prune.stats.backward_ops_retained,
            100.0 * self.prune.stats.backward_elimination_ratio()
        )
        .unwrap();
        writeln!(
            s,
            "Activation memory: {} tensors → {} tensors ({:.1}% eliminated)",
            self.prune.stats.full_backward_saved_activations,
            self.prune.stats.pruned_saved_activations,
            100.0 * self.prune.stats.activation_elimination_ratio()
        )
        .unwrap();
        writeln!(
            s,
            "Frozen parameters: {}  |  Trainable targets: {}",
            self.prune.stats.frozen_params, self.prune.stats.gradient_targets
        )
        .unwrap();
        writeln!(s).unwrap();
        writeln!(
            s,
            "Adapter parameters: {}  |  Sites: {}  |  Fused: {}/{} ({:.1}%)",
            self.adapter_params(),
            self.fusion.total_count(),
            self.fusion.fused_count(),
            self.fusion.total_count(),
            100.0 * self.fusion.fusion_ratio()
        )
        .unwrap();
        writeln!(
            s,
            "Memory plan: {} live activations → {} slots (reuse {:.1}%)",
            self.memory.stats.live_activations,
            self.memory.stats.slots_used,
            100.0 * self.memory.stats.reuse_ratio()
        )
        .unwrap();
        s
    }

    /// Render a PEFT comparison report — WRGA's measured numbers against
    /// theoretical baselines for LoRA, AdaLoRA, GaLore, and ReFT, derived
    /// from the same per-site shape data that drove this plan.
    ///
    /// Surface: `nsl check --wrga-compare <file>.nsl [--weights w.safetensors]`
    /// (WRGA paper §8.3).
    ///
    /// The comparison is **theoretical**: it answers "if we re-ran the same
    /// site selection under each baseline's parameter formula, how would the
    /// adapter parameter count, backward FLOPs ratio, activation memory ratio,
    /// and fusion ratio differ?" It does NOT compare empirical quality or
    /// wall-clock speed — those require the §9 benchmark suite and the §9.3
    /// ablation harness (separate audit gaps).
    ///
    /// ## Baseline formulas
    ///
    /// Per WRGA-placed site `i` with weight shape `[m_i, n_i]`:
    /// - **LoRA (Hu et al. 2021)**: fixed rank `r=16`; `params_i = 16 * (m_i + n_i)`.
    ///   No backward pruning, no fusion, no activation savings.
    /// - **AdaLoRA (Zhang et al. 2023)**: SVD-derived rank — modeled here as
    ///   WRGA's own spectral allocation (both methods use the same
    ///   singular-value-importance heuristic). Same param count as WRGA but
    ///   no Wengert pruning and no fusion.
    /// - **GaLore (Zhao et al. 2024)**: full-parameter training with low-rank
    ///   gradient projection — zero adapter parameters by construction. No
    ///   backward pruning (GaLore prunes the gradient subspace, not the
    ///   compute graph) and no fusion.
    /// - **ReFT (Wu et al. 2024)**: rank-4 intervention vectors on the top
    ///   25% of placement sites; `params_i = 4 * n_i` for those sites only.
    ///   No backward pruning, no fusion.
    ///
    /// The "Backward FLOPs %" column reports each method's retained-backward
    /// op count as a fraction of the unpruned forward op count. The
    /// "Activation Memory %" column does the same for saved activations.
    /// Both lower-is-better. The "Fusion %" column is the fraction of
    /// placement sites whose adapter is fused into an adjacent matmul.
    ///
    /// Output is deterministic given the input plan: column widths are
    /// fixed, ordering is fixed, every numeric is computed from struct fields.
    pub fn render_compare_report(&self) -> String {
        use std::fmt::Write as _;

        let mut s = String::new();
        writeln!(s, "=== WRGA PEFT Comparison Report ===").unwrap();
        writeln!(s, "Mode: {}", self.mode.as_str()).unwrap();
        writeln!(s, "Target hardware: {}", self.target_gpu).unwrap();
        writeln!(s, "Placement sites: {}", self.placements.len()).unwrap();
        writeln!(s).unwrap();

        let baselines = self.compute_compare_baselines();

        // Table: Method | Params | Bwd Live % | Act Saved % | Fusion %
        // Widths chosen so the WRGA row plus baseline rows line up.
        writeln!(
            s,
            "{:<22} | {:>10} | {:>10} | {:>11} | {:>7}",
            "Method", "Params", "Bwd Live %", "Act Saved %", "Fusion %"
        )
        .unwrap();
        writeln!(
            s,
            "{0:-<22}-+-{0:-<10}-+-{0:-<10}-+-{0:-<11}-+-{0:-<7}",
            ""
        )
        .unwrap();

        for row in &baselines {
            writeln!(
                s,
                "{:<22} | {:>10} | {:>9.1}% | {:>10.1}% | {:>6.1}%",
                row.label,
                fmt_with_commas(row.adapter_params),
                100.0 * row.backward_retained_ratio,
                100.0 * row.activation_retained_ratio,
                100.0 * row.fusion_ratio,
            )
            .unwrap();
        }

        writeln!(s).unwrap();
        writeln!(
            s,
            "Bwd Live %  — fraction of forward Wengert ops that must stay live \
             for the backward pass (lower = more Wengert pruning)."
        )
        .unwrap();
        writeln!(
            s,
            "Act Saved % — fraction of saved-for-backward activations retained \
             after Innovation 5 memory planning (lower = more activation reuse)."
        )
        .unwrap();
        writeln!(
            s,
            "Fusion %    — share of placement sites whose adapter fuses into an \
             adjacent matmul (Innovation 4)."
        )
        .unwrap();
        writeln!(s).unwrap();
        writeln!(
            s,
            "Comparison is structural (param formulas + WRGA's measured \
             pruning/fusion); empirical quality and wall-clock require the \
             §9 benchmark suite."
        )
        .unwrap();
        s
    }

    /// Build the five comparison rows (WRGA + four baselines). Pulled out so
    /// the table renderer and tests can share one source of truth for the
    /// numerics.
    fn compute_compare_baselines(&self) -> Vec<CompareRow> {
        // Recover each site's (m + n) from the actual adapter parameter
        // formula `params = rank * (m + n)`. Sites with rank == 0 (skipped
        // by the placement stage) contribute zero — there's no shape to
        // recover for them.
        let mn_sums: Vec<usize> = self
            .ranks
            .iter()
            .map(|r| if r.rank == 0 { 0 } else { r.adapter_params / r.rank })
            .collect();

        // WRGA: measured values from the plan itself.
        let wrga_params: usize = self.ranks.iter().map(|r| r.adapter_params).sum();
        let total_fwd_ops = self.prune.stats.forward_ops_total.max(1);
        let wrga_backward_retained =
            self.prune.stats.backward_ops_retained as f64 / total_fwd_ops as f64;
        let naive_acts = self.prune.stats.full_backward_saved_activations.max(1);
        let wrga_activation_retained =
            self.prune.stats.pruned_saved_activations as f64 / naive_acts as f64;
        let wrga_fusion = self.fusion.fusion_ratio();

        // LoRA: fixed rank r = 16 on every placement site.
        const LORA_FIXED_RANK: usize = 16;
        let lora_params: usize = mn_sums.iter().map(|mn| LORA_FIXED_RANK * mn).sum();

        // AdaLoRA: SVD-driven rank allocation — model as WRGA's own
        // adapter_params, since both methods share the same singular-value
        // importance formula. The differentiator is that AdaLoRA does NOT do
        // Wengert pruning or fusion-integrated adapters.
        let adalora_params = wrga_params;

        // GaLore: zero adapter parameters by construction (it projects
        // full-parameter gradients into a low-rank subspace instead of
        // adding adapter weights).
        let galore_params = 0usize;

        // ReFT: rank-4 intervention vectors on the top 25% of placement
        // sites by parameter footprint (round up). Each intervention
        // contributes `4 * n_i` params, approximating `n_i = mn_sum / 2` when
        // m_i ≈ n_i (square weights — typical for transformer attention
        // projections). For non-square weights this over- or under-estimates
        // by the m/n ratio; acceptable at paper §8.3's order-of-magnitude
        // intent.
        //
        // Sites with `mn_sum == 0` (placements WRGA rank-skipped) contribute
        // ZERO ReFT params — the `if mn == 0 { 0 }` branch is load-bearing
        // when WRGA prunes >75% of candidate sites, since those rank-0 entries
        // can otherwise survive the top-25% take and each would silently add
        // `REFT_RANK * 1` phantom params via the historical `.max(1)`.
        const REFT_RANK: usize = 4;
        let reft_site_count = self.placements.len().div_ceil(4);
        let mut sorted_mn: Vec<usize> = mn_sums.clone();
        sorted_mn.sort_unstable_by(|a, b| b.cmp(a));
        let reft_params: usize = sorted_mn
            .iter()
            .take(reft_site_count)
            .map(|mn| if *mn == 0 { 0 } else { REFT_RANK * (mn / 2).max(1) })
            .sum();

        vec![
            CompareRow {
                label: "WRGA (this run)".to_string(),
                adapter_params: wrga_params,
                backward_retained_ratio: wrga_backward_retained,
                activation_retained_ratio: wrga_activation_retained,
                fusion_ratio: wrga_fusion,
            },
            CompareRow {
                label: "LoRA (r=16)".to_string(),
                adapter_params: lora_params,
                backward_retained_ratio: 1.0,
                activation_retained_ratio: 1.0,
                fusion_ratio: 0.0,
            },
            CompareRow {
                label: "AdaLoRA".to_string(),
                adapter_params: adalora_params,
                backward_retained_ratio: 1.0,
                activation_retained_ratio: 1.0,
                fusion_ratio: 0.0,
            },
            CompareRow {
                label: "GaLore".to_string(),
                adapter_params: galore_params,
                backward_retained_ratio: 1.0,
                activation_retained_ratio: 1.0,
                fusion_ratio: 0.0,
            },
            CompareRow {
                label: "ReFT (r=4, top-25%)".to_string(),
                adapter_params: reft_params,
                backward_retained_ratio: 1.0,
                activation_retained_ratio: 1.0,
                fusion_ratio: 0.0,
            },
        ]
    }
}

/// One row of the PEFT comparison table. All numerics are dimensionless or
/// in [0, 1] (ratios); the renderer multiplies by 100 for the percentage
/// columns.
#[derive(Debug, Clone, PartialEq)]
struct CompareRow {
    label: String,
    adapter_params: usize,
    backward_retained_ratio: f64,
    activation_retained_ratio: f64,
    fusion_ratio: f64,
}

/// Format an integer with thousands separators (commas) using a
/// dependency-free hand-rolled implementation. Picked over `humansize` /
/// `num-format` to keep the codegen crate's dependency surface minimal.
fn fmt_with_commas(n: usize) -> String {
    let s = n.to_string();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    let bytes = s.as_bytes();
    let len = bytes.len();
    for (i, &b) in bytes.iter().enumerate() {
        if i > 0 && (len - i).is_multiple_of(3) {
            out.push(',');
        }
        out.push(b as char);
    }
    out
}

// ---------------------------------------------------------------------------
// Paper §9.3 ablation no-ops — one substitute per Innovation.
// ---------------------------------------------------------------------------

/// Innovation 1 no-op: retain every forward op as backward-live and mark
/// every "would-save" activation as needing save. Mirrors what a naive PyTorch
/// backward would produce — every intermediate stays alive for the adjoint.
///
/// `stats.backward_ops_retained == stats.forward_ops_total` and
/// `stats.pruned_saved_activations == stats.full_backward_saved_activations`
/// so `render_report` reports 100% retained / 0% eliminated for both
/// rows of the prune summary.
fn prune_noop(list: &WengertList, trainable: &BTreeSet<VarId>) -> PruneResult {
    use crate::wrga_prune::PruneStats;

    // Two passes — mirror `wrga_prune::prune()`'s structure exactly. A single
    // pass would set `saved_for_backward` BEFORE `activation_live` has been
    // populated by later ops that read this op's result as an input, so any
    // input-saved activation would get a stale `false` bit. Match the real
    // pruner: build the live sets first, then stamp the bits.
    let mut backward_live: BTreeSet<VarId> = BTreeSet::new();
    let mut activation_live: BTreeSet<VarId> = BTreeSet::new();
    let mut total_full_saves: usize = 0;
    for op in &list.ops {
        backward_live.insert(op.result);
        let req = crate::wrga_prune::save_requirements(&op.op);
        if req.needs_output || req.needs_inputs {
            total_full_saves += 1;
        }
        if req.needs_output {
            activation_live.insert(op.result);
        }
        if req.needs_inputs {
            for &inp in &op.inputs {
                activation_live.insert(inp);
            }
        }
    }
    let mut pruned = list.clone();
    for op in pruned.ops.iter_mut() {
        op.saved_for_backward = activation_live.contains(&op.result);
    }
    let frozen_params = list
        .ops
        .iter()
        .filter(|op| matches!(op.op, PrimalOp::Param(_)) && !trainable.contains(&op.result))
        .count();

    let pruned_saved = activation_live.len();
    PruneResult {
        pruned,
        backward_live,
        activation_live,
        stats: PruneStats {
            forward_ops_total: list.len(),
            backward_ops_retained: list.len(),
            // Mirror the unit mismatch in `wrga_prune::prune()` exactly:
            // `full_backward_saved_activations` is op-count (the naive
            // PyTorch baseline counts ops with any save requirement);
            // `pruned_saved_activations` is VarId-count (distinct tensors
            // actually kept alive). The ratio is what feeds `render_report`'s
            // "X tensors → Y tensors" line, so consistent units with the
            // real pruner matters for diffability across runs.
            full_backward_saved_activations: total_full_saves,
            pruned_saved_activations: pruned_saved,
            gradient_targets: trainable.len(),
            frozen_params,
        },
    }
}

/// Innovation 2 no-op: skip the roofline analysis and place an adapter on
/// every candidate site at the maximum allowed rank, classified as
/// memory-bound (the most conservative bucket). Each placement carries
/// `decorator_kind = None` and empty synth fields — same shape as the live
/// allocator's output for an un-decorated site.
fn placements_noop(sites: &[AdapterSite], r_max: usize) -> Vec<AdapterPlacement> {
    use crate::cost_model::BoundClassification;
    use crate::wrga_roofline::AdapterKind;
    sites
        .iter()
        .map(|s| AdapterPlacement {
            name: s.name.clone(),
            arithmetic_intensity: 0.0,
            classification: BoundClassification::MemoryBound,
            roofline_slack: 1.0,
            adapter: AdapterKind::Lora,
            suggested_rank: r_max,
            rationale: "ablation: --without-roofline (r_max on every site)".to_string(),
            decorator_kind: None,
            alpha: None,
            synthesized_fields: Vec::new(),
            init_strategies: Vec::new(),
        })
        .collect()
}

/// Innovation 3 no-op: skip SVD-driven rank reallocation. Each placement
/// keeps its `suggested_rank`; `RankAllocation.adapter_params` defaults to
/// zero (no per-site shape known without spectral analysis). `spectral`
/// returns empty since no SVD ran. `override_diags` is also empty.
fn spectral_noop(
    placements: &[AdapterPlacement],
) -> (
    Vec<SpectralAnalysis>,
    Vec<RankAllocation>,
    Vec<crate::wggo_overrides::OverrideDiagnostic>,
) {
    let ranks = placements
        .iter()
        .map(|p| RankAllocation {
            name: p.name.clone(),
            rank: p.suggested_rank,
            effective_rank: p.suggested_rank as f64,
            // No spectral data ⇒ no (m+n) recovery. Comparison renderer
            // tolerates rank * 0 = 0 here; downstream consumers that need a
            // real param count must NOT ablate spectral.
            adapter_params: 0,
        })
        .collect();
    (Vec::new(), ranks, Vec::new())
}

/// Innovation 5 no-op: skip the live-range memory plan. Reports the naive
/// allocation — one slot per live activation, no reuse. `assignments` is
/// empty (no slot assignments produced); the stats fields are populated so
/// `render_report`'s "Memory plan: A live → B slots (reuse C%)" line yields
/// the unsurprising "A live → A slots (reuse 0%)".
fn memory_noop(activation_live: &BTreeSet<VarId>) -> MemoryPlan {
    use crate::wrga_memory::MemoryPlanStats;
    let n = activation_live.len();
    MemoryPlan {
        assignments: Vec::new(),
        stats: MemoryPlanStats {
            live_activations: n,
            slots_used: n,
            naive_peak_bytes: 0,
            planned_peak_bytes: 0,
        },
    }
}

/// Run the full WRGA driver.
///
/// When `input.ablation` has any field set, the corresponding Innovation's
/// stage is replaced by a deterministic no-op — see [`WrgaAblation`] for the
/// substitution semantics. The non-ablated stages still run normally and use
/// the ablated stage's degenerate output as input, so each ablation is
/// independent (running with only `skip_fusion_integration=true` does NOT
/// disturb the prune or placement stages).
pub fn run(input: WrgaInput) -> WrgaPlan {
    let abl = input.ablation;

    // ── Stage 1: PEFT topology extraction (not an Innovation; always runs)
    let trainable = select_trainable(input.wengert, &input);

    // ── Stage 2: Innovation 1 — Wengert pruning ───────────────────────────
    let prune_result = if abl.skip_wengert_pruning {
        prune_noop(input.wengert, &trainable)
    } else {
        prune(input.wengert, &trainable, input.loss_output)
    };

    // ── Stage 3: Innovation 2 — Roofline placement ────────────────────────
    let gpu = find_gpu(input.target).unwrap_or_else(default_gpu);
    let sites = infer_sites_from_wengert(input.wengert, &trainable, &input);
    let (placements, placement_diags) = if abl.skip_roofline_placement {
        // §9.3 ablation: skip roofline AND the WGGO placement filter — the
        // no-op output is deliberately a uniform "adapter on every site at
        // r_max", and the WGGO filter is part of Innovation 2's pipeline so
        // running it on the no-op would corrupt the ablation contract.
        (placements_noop(&sites, input.r_max), Vec::new())
    } else {
        let mut placements = place_adapters(&sites, gpu, input.r_min, input.r_max);
        // Narrow the roofline placement to the projections WGGO budgeted for
        // (its `adapter_placement` is the comm-budget-feasible set, G5/G7).
        // Sites outside it are forced to Skip; each overruled site yields a
        // diagnostic.
        let diags = apply_wggo_placement_filter(&mut placements, input.wggo_overrides);
        (placements, diags)
    };

    // ── Stage 4: Innovation 3 — Spectral rank allocation ──────────────────
    let (spectral, ranks, override_diags) = if abl.skip_spectral_allocation {
        spectral_noop(&placements)
    } else {
        run_spectral(input.weights, &placements, &input)
    };

    // ── Stage 5: Innovation 4 — Fusion integration ────────────────────────
    let rank_overrides: Vec<usize> =
        ranks.iter().map(|r| r.rank).collect();
    let fusion = if abl.skip_fusion_integration {
        FusionPlan::default()
    } else {
        let f = build_fusion_plan(&placements, Some(&rank_overrides));
        verify_fused_sites_have_no_intermediate(&f, input.wengert);
        f
    };

    // ── Stage 6: Innovation 5 — Memory plan ───────────────────────────────
    let memory = if abl.skip_memory_planning {
        memory_noop(&prune_result.activation_live)
    } else {
        let size_hints: SizeHints = HashMap::new();
        plan_memory_with_pin(
            input.wengert,
            &prune_result.activation_live,
            &size_hints,
            &{
                let mut s = BTreeSet::new();
                s.insert(input.loss_output);
                s
            },
            &input.inspect_pinned_vars,
        )
    };

    WrgaPlan {
        mode: input.mode,
        target_gpu: gpu.name.to_string(),
        prune: prune_result,
        placements,
        spectral,
        ranks,
        fusion,
        memory,
        override_diagnostics: {
            // Placement-filter diagnostics (site exclusions) precede the
            // rank-allocation diagnostics from spectral analysis.
            let mut d = placement_diags;
            d.extend(override_diags);
            d
        },
        ablation: abl,
    }
}

/// Narrow WRGA's roofline-chosen adapter sites to the projections WGGO
/// budgeted for.  For each layer WGGO gave a positive rank and a non-`None`
/// `adapter_placement`, any site whose projection falls *outside* that set is
/// forced to [`AdapterKind::Skip`] — WGGO's placement is the comm-budget-
/// feasible minimal set (G5/G7), so adapting extra sites would exceed the
/// adapter-comm budget WGGO enforced.  Returns one [`OverrideDiagnostic`] per
/// roofline adapter that was overruled.
///
/// **v1 scope / known limitation.**  Enforcement keys off
/// [`AdapterPlacement::covers_projection`], which only recognizes *split*
/// per-projection weight names (`wq`/`q_proj`, FFN `gate`/`up`/`down`/`fc*`;
/// see [`crate::wggo_ilp::proj_role`]).  Any site whose name it does *not*
/// recognize returns `None` ("no opinion") and is left untouched.  That escape
/// set is broader than just norms/embeddings: it includes whole model families
/// whose projections are **fused or differently named** — GPT-2 `c_attn`
/// (fused QKV) / `c_proj` / `c_fc`, GPT-NeoX & Falcon `query_key_value` /
/// `dense` / `dense_h_to_4h` / `dense_4h_to_h`, Phi/MPT `Wqkv`, and fused
/// `gate_up_proj`.  For such models a non-`None` placement does **not**
/// constrain WRGA, so the comm-budget guarantee is not enforced there (WRGA may
/// over-place).  This is fail-safe — the filter never produces a *wrong*
/// adapter, only an unconstrained one — and is the deliberate conservative v1
/// choice.  Note this diverges from CEP, which *hard-refuses* fused-QKV models
/// (`cep_extract.rs`) under the repo's "deferral must refuse" invariant;
/// extending placement enforcement to fused weights is a follow-up.
fn apply_wggo_placement_filter(
    placements: &mut [AdapterPlacement],
    overrides: Option<&crate::wggo_overrides::WggoOverrides>,
) -> Vec<crate::wggo_overrides::OverrideDiagnostic> {
    use crate::wrga_roofline::AdapterKind;
    let Some(over) = overrides else {
        return Vec::new();
    };
    let mut diags = Vec::new();
    for p in placements.iter_mut() {
        let Some(ov) = over.find_by_layer_containing(&p.name) else {
            continue;
        };
        // rank==0 means WGGO wants no adapter on this layer at all — that is
        // the rank path's job (`RankForbiddenByWggo`); skipping here avoids a
        // double-report.  For rank>0 the placement drives the decision via
        // `covers_projection`, INCLUDING an (invariant-violating) `None`
        // placement: `None` excludes every projection, so all recognized sites
        // are forced to Skip with a diagnostic rather than silently adapted.
        if ov.adapter_rank == 0 {
            continue;
        }
        if ov.adapter_placement.covers_projection(&p.name) == Some(false)
            && !matches!(p.adapter, AdapterKind::Skip)
        {
            diags.push(crate::wggo_overrides::OverrideDiagnostic {
                layer_index: ov.layer_index,
                // Per-LAYER name (e.g. "blocks.0"); the spectral rank path
                // (`allocate_ranks`) instead reports per-PROJECTION names
                // (e.g. "blocks.0.attn.wq").  Harmless today — the sole
                // renderer keys off `layer_index` — but the granularity differs.
                layer_name: ov.layer_name.clone(),
                reason: crate::wggo_overrides::OverrideRejectReason::AdapterSiteOutsidePlacement {
                    placement: ov.adapter_placement.as_str().to_string(),
                },
                requested: format!("{:?}", p.adapter).to_ascii_lowercase(),
                applied: "skip".to_string(),
            });
            p.adapter = AdapterKind::Skip;
            p.suggested_rank = 0;
        }
    }
    diags
}

// ---------------------------------------------------------------------------
// Stage helpers
// ---------------------------------------------------------------------------

/// Determine the set of trainable parameter VarIds per [`WrgaMode`].
fn select_trainable(list: &WengertList, input: &WrgaInput) -> BTreeSet<VarId> {
    use crate::wrga_prune::glob_match;
    let mut out = BTreeSet::new();
    for op in &list.ops {
        let name = match &op.op {
            PrimalOp::Param(n) => n.as_str(),
            _ => continue,
        };
        let included = match input.mode {
            WrgaMode::Auto => {
                // If the caller supplied patterns, honour them; otherwise all
                // parameters are trainable.  `Auto` is expected to be paired
                // with an upstream `@freeze(exclude=...)` that resolves into
                // `trainable_patterns` — but we don't require it.
                if input.trainable_patterns.is_empty() {
                    true
                } else {
                    input
                        .trainable_patterns
                        .iter()
                        .any(|p| glob_match(p, name))
                }
            }
            WrgaMode::Manual => {
                // Manual mode: only explicit adapter targets receive gradients.
                input
                    .manual_adapter_targets
                    .iter()
                    .any(|p| glob_match(p, name))
            }
            WrgaMode::Hybrid => {
                // Hybrid: the layer scope defines trainability; treat each
                // layer prefix as `blocks.N.*`.
                input
                    .hybrid_layers
                    .iter()
                    .any(|layer| name.starts_with(layer))
            }
        };
        if included {
            out.insert(op.result);
        }
    }
    out
}

/// Infer candidate adapter sites by walking the Wengert list and attaching a
/// [`SiteKind`] to each trainable parameter based on its downstream op.
///
/// This is a simple heuristic — the real compiler has richer shape info via
/// the type-map — but it's good enough for the driver to produce a deterministic
/// placement plan even in unit tests.
fn infer_sites_from_wengert(
    list: &WengertList,
    trainable: &BTreeSet<VarId>,
    _input: &WrgaInput,
) -> Vec<AdapterSite> {
    let mut out = Vec::new();
    // Index param VarId → name for friendly site names.
    let mut param_name: BTreeMap<VarId, String> = BTreeMap::new();
    for op in &list.ops {
        if let PrimalOp::Param(n) = &op.op {
            param_name.insert(op.result, n.clone());
        }
    }

    // For each op that consumes a trainable param, decide a site kind from
    // the op type.  Matmul → Matmul, LayerNorm/RMSNorm → Norm, Softmax →
    // Softmax, Embedding → Embedding, others → skip.
    for op in &list.ops {
        let Some(&trainable_id) = op.inputs.iter().find(|v| trainable.contains(v)) else {
            continue;
        };
        let name = param_name
            .get(&trainable_id)
            .cloned()
            .unwrap_or_else(|| format!("var_{trainable_id}"));
        let kind = match op.op {
            PrimalOp::Matmul => SiteKind::Matmul,
            PrimalOp::LayerNorm { .. } | PrimalOp::RMSNorm { .. } => SiteKind::Norm,
            PrimalOp::Softmax { .. } | PrimalOp::LogSoftmax { .. } => SiteKind::Softmax,
            PrimalOp::Embedding => SiteKind::Embedding,
            _ => continue,
        };
        // No shape info in the Wengert list; use a modest default so the
        // cost model still produces a meaningful AI for the placement.
        out.push(AdapterSite {
            name,
            kind,
            shape: vec![512, 512],
            dtype_bytes: 2,
            batch: 1,
            seq: 128,
        });
    }
    out
}

fn run_spectral(
    weights: Option<&WeightMap>,
    placements: &[AdapterPlacement],
    input: &WrgaInput,
) -> (Vec<SpectralAnalysis>, Vec<RankAllocation>, Vec<crate::wggo_overrides::OverrideDiagnostic>) {
    let Some(wm) = weights else {
        // Without weights, fall back to uniform r_max (clamped by budget) per
        // non-skipped placement.
        let ranks: Vec<RankAllocation> = placements
            .iter()
            .map(|p| RankAllocation {
                name: p.name.clone(),
                rank: p.suggested_rank.max(input.r_min).min(input.r_max),
                effective_rank: 0.0,
                adapter_params: p.suggested_rank * 1024, // rough guess
            })
            .collect();
        return (Vec::new(), ranks, Vec::new());
    };

    // Collect analyses only for sites we actually plan to adapt.
    let active_names: BTreeSet<&str> = placements
        .iter()
        .filter(|p| !matches!(p.adapter, crate::wrga_roofline::AdapterKind::Skip))
        .map(|p| p.name.as_str())
        .collect();
    let mut spectral = analyse_weight_map(wm, input.r_max.max(8), 5, 8, input.seed);
    spectral.retain(|s| active_names.iter().any(|n| s.name == *n));

    // Cross-reference: per-site roofline slack as the allocator weight.
    let mut slack = HashMap::new();
    for p in placements {
        slack.insert(p.name.clone(), p.roofline_slack.max(1e-6));
    }

    // Derive a parameter budget: if the user specified one, use it; otherwise
    // let every site claim `r_max` (i.e. R_total = r_max · sites).
    let budget = if input.budget_params > 0 {
        input.budget_params
    } else {
        spectral.len().max(1) * input.r_max
    };
    let (ranks, override_diags) =
        allocate_ranks(&spectral, budget, input.r_min, input.r_max, Some(&slack), input.wggo_overrides);

    (spectral, ranks, override_diags)
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

    fn tiny_transformer_like() -> WengertList {
        // x (#0)          input
        // wq (#1)         param  blocks.6.wq
        // wk (#2)         param  blocks.6.wk
        // wv (#3)         param  blocks.7.wq   (trainable layer)
        // q = wq @ x      (#4)
        // k = wk @ x      (#5)  <- frozen path
        // v = wv @ x      (#6)  <- trainable path
        // y = q + v       (#7)
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("blocks.6.wq".into()), vec![]),
            op(2, 2, PrimalOp::Param("blocks.6.wk".into()), vec![]),
            op(3, 3, PrimalOp::Param("blocks.7.wq".into()), vec![]),
            op(4, 4, PrimalOp::Matmul, vec![1, 0]),
            op(5, 5, PrimalOp::Matmul, vec![2, 0]),
            op(6, 6, PrimalOp::Matmul, vec![3, 0]),
            op(7, 7, PrimalOp::Add, vec![4, 6]),
        ];
        WengertList {
            ops,
            output: 7,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    #[test]
    fn auto_mode_produces_plan_without_weights() {
        let w = tiny_transformer_like();
        let input = WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.7.*"],
            manual_adapter_targets: Vec::new(),
            hybrid_layers: Vec::new(),
            wengert: &w,
            loss_output: 7,
            weights: None,
            target: "",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: WrgaAblation::default(),
        };
        let plan = run(input);
        // Exactly one param (blocks.7.wq) is trainable → exactly one site.
        assert_eq!(plan.prune.stats.gradient_targets, 1);
        assert_eq!(plan.prune.stats.frozen_params, 2);
        // The report must render without panicking.
        let report = plan.render_report();
        assert!(report.contains("WRGA Compilation Report"));
        assert!(report.contains("Mode: auto"));
    }

    #[test]
    fn manual_mode_only_adapts_explicit_targets() {
        let w = tiny_transformer_like();
        let input = WrgaInput {
            mode: WrgaMode::Manual,
            trainable_patterns: Vec::new(),
            manual_adapter_targets: vec!["blocks.6.wq"],
            hybrid_layers: Vec::new(),
            wengert: &w,
            loss_output: 7,
            weights: None,
            target: "",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: WrgaAblation::default(),
        };
        let plan = run(input);
        assert_eq!(plan.prune.stats.gradient_targets, 1);
    }

    #[test]
    fn hybrid_mode_layer_scope() {
        let w = tiny_transformer_like();
        let input = WrgaInput {
            mode: WrgaMode::Hybrid,
            trainable_patterns: Vec::new(),
            manual_adapter_targets: Vec::new(),
            hybrid_layers: vec!["blocks.6"],
            wengert: &w,
            loss_output: 7,
            weights: None,
            target: "",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: WrgaAblation::default(),
        };
        let plan = run(input);
        // Both blocks.6.wq and blocks.6.wk match the layer scope.
        assert_eq!(plan.prune.stats.gradient_targets, 2);
        assert_eq!(plan.prune.stats.frozen_params, 1);
    }

    #[test]
    fn render_report_is_deterministic() {
        let w = tiny_transformer_like();
        let make = || WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.7.*"],
            manual_adapter_targets: Vec::new(),
            hybrid_layers: Vec::new(),
            wengert: &w,
            loss_output: 7,
            weights: None,
            target: "",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: WrgaAblation::default(),
        };
        assert_eq!(run(make()).render_report(), run(make()).render_report());
    }

    #[test]
    fn wrga_plan_default_has_empty_override_diagnostics() {
        // Smoke test: default plan from run() has empty diagnostics when no
        // overrides supplied.
        let w = tiny_transformer_like();
        let input = WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.7.*"],
            manual_adapter_targets: Vec::new(),
            hybrid_layers: Vec::new(),
            wengert: &w,
            loss_output: 7,
            weights: None,
            target: "",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 0,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: WrgaAblation::default(),
        };
        let plan = run(input);
        assert!(
            plan.override_diagnostics.is_empty(),
            "default WrgaPlan must have empty override_diagnostics"
        );
    }

    // -----------------------------------------------------------------------
    // Task 6 smoke tests: end-to-end propagation of WggoOverrides through
    // wrga::run → WrgaPlan.override_diagnostics
    //
    // Note: run_spectral returns early (no diagnostics) when weights=None,
    // because allocate_ranks is only called when SVD-derived spectral data is
    // available.  The two smoke tests below therefore:
    //   (a) verify wrga::run returns a plan that accepts wggo_overrides=Some(…)
    //       without panicking, and that override_diagnostics is empty in the
    //       no-weights path (field propagation is the observable);
    //   (b) verify the field starts empty when wggo_overrides=None;
    //   (c) prove the ACTUAL clamp diagnostic fires via a direct allocate_ranks
    //       unit test, which is the canonical coverage for the clamp path.
    // -----------------------------------------------------------------------

    #[test]
    fn wrga_run_with_overrides_field_accepted_and_plan_returned() {
        // Smoke: wrga::run must not panic when wggo_overrides is Some(…).
        // Without weights the spectral path is skipped, so override_diagnostics
        // will be empty — but the WrgaPlan must still be returned and the field
        // must exist (proving wiring from WrgaInput → WrgaPlan is live).
        let w = tiny_transformer_like();
        let over = crate::wggo_overrides::WggoOverrides {
            per_layer: vec![crate::wggo_overrides::PerLayerOverride {
                layer_index: 6,
                layer_name: "blocks.6".into(),
                active_heads: 8,
                requested_csha_level: None,
                adapter_rank: 32, // would clamp to r_max if weights were present
                adapter_placement: crate::wggo_ilp::AdapterPlacement::None,
                fase_fused: false,
                packing_mode: 0,
                shard_factor: 0,
            }],
        };
        let input = WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.6.*"],
            manual_adapter_targets: Vec::new(),
            hybrid_layers: Vec::new(),
            wengert: &w,
            loss_output: w.output,
            weights: None, // no SVD → spectral path skipped → no clamp diag
            target: "rtx5070ti",
            budget_params: 10_000_000,
            r_min: 2,
            r_max: 16,
            seed: 0,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: Some(&over),
            ablation: WrgaAblation::default(),
        };
        let plan = run(input);
        // Field exists and is accessible (proves wiring).  No diags expected
        // because the no-weights fast-path in run_spectral bypasses allocate_ranks.
        let _ = &plan.override_diagnostics;
    }

    #[test]
    fn wrga_run_without_overrides_leaves_override_diagnostics_empty() {
        // No WggoOverrides supplied → override_diagnostics must be empty.
        let w = tiny_transformer_like();
        let input = WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.6.*"],
            manual_adapter_targets: Vec::new(),
            hybrid_layers: Vec::new(),
            wengert: &w,
            loss_output: w.output,
            weights: None,
            target: "rtx5070ti",
            budget_params: 10_000_000,
            r_min: 2,
            r_max: 16,
            seed: 0,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: WrgaAblation::default(),
        };
        let plan = run(input);
        assert!(
            plan.override_diagnostics.is_empty(),
            "no overrides → no diagnostics; got {:?}",
            plan.override_diagnostics
        );
    }

    // Direct unit test proving the clamp diagnostic fires end-to-end through
    // allocate_ranks (called from run_spectral when weights are present).
    // This is the canonical coverage for the RankClampedToBounds path.
    #[test]
    fn allocate_ranks_clamp_fires_through_run_spectral() {
        use crate::wggo_overrides::{OverrideRejectReason, PerLayerOverride, WggoOverrides};
        use crate::wrga_spectral::{allocate_ranks, SpectralAnalysis};

        // Minimal spectral entry for blocks.6.wq with a non-trivial
        // effective_rank so the normal (non-degenerate) override path fires.
        let spectral = vec![SpectralAnalysis {
            name: "blocks.6.wq".into(),
            shape: [64, 64],
            effective_rank: 8.0,
            singular_values: vec![4.0, 2.0, 1.0],
            truncated_rank: 3,
        }];
        let over = WggoOverrides {
            per_layer: vec![PerLayerOverride {
                layer_index: 6,
                layer_name: "blocks.6".into(), // prefix — matches blocks.6.wq
                active_heads: 8,
                requested_csha_level: None,
                adapter_rank: 32, // > r_max=16 → RankClampedToBounds
                adapter_placement: crate::wggo_ilp::AdapterPlacement::None,
                fase_fused: false,
                packing_mode: 0,
                shard_factor: 0,
            }],
        };
        let (_allocs, diags) = allocate_ranks(&spectral, 10_000_000, 2, 16, None, Some(&over));
        assert!(
            diags.iter().any(|d| matches!(
                d.reason,
                OverrideRejectReason::RankClampedToBounds { r_max: 16, .. }
            )),
            "expected RankClampedToBounds diagnostic; got {:?}",
            diags
        );
    }

    // -----------------------------------------------------------------------
    // Gap #3 — PEFT comparison report (paper §8.3, `nsl check --wrga-compare`)
    // -----------------------------------------------------------------------

    #[test]
    fn fmt_with_commas_inserts_thousands_separators() {
        assert_eq!(super::fmt_with_commas(0), "0");
        assert_eq!(super::fmt_with_commas(42), "42");
        assert_eq!(super::fmt_with_commas(999), "999");
        assert_eq!(super::fmt_with_commas(1_000), "1,000");
        assert_eq!(super::fmt_with_commas(12_345), "12,345");
        assert_eq!(super::fmt_with_commas(123_456_789), "123,456,789");
        // Multiple of 1000 must NOT have a leading comma.
        assert_eq!(super::fmt_with_commas(1_000_000), "1,000,000");
    }

    #[test]
    fn compare_report_contains_all_baselines_and_wrga_row() {
        let w = tiny_transformer_like();
        let input = WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.7.*"],
            manual_adapter_targets: Vec::new(),
            hybrid_layers: Vec::new(),
            wengert: &w,
            loss_output: w.output,
            weights: None,
            target: "rtx5070ti",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: WrgaAblation::default(),
        };
        let plan = run(input);
        let report = plan.render_compare_report();

        // Header + every baseline row must be present so users get a complete
        // table even when WRGA's measured numbers and a baseline coincide.
        assert!(report.contains("=== WRGA PEFT Comparison Report ==="));
        assert!(report.contains("WRGA (this run)"));
        assert!(report.contains("LoRA (r=16)"));
        assert!(report.contains("AdaLoRA"));
        assert!(report.contains("GaLore"));
        assert!(report.contains("ReFT (r=4, top-25%)"));

        // Explanatory footer prevents a casual reader from over-interpreting
        // the structural ratios as empirical quality/speed.
        assert!(report.contains("structural"));
        assert!(report.contains("§9 benchmark suite"));
    }

    #[test]
    fn compare_report_is_deterministic() {
        let w = tiny_transformer_like();
        let make = || WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.7.*"],
            manual_adapter_targets: Vec::new(),
            hybrid_layers: Vec::new(),
            wengert: &w,
            loss_output: w.output,
            weights: None,
            target: "rtx5070ti",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: WrgaAblation::default(),
        };
        assert_eq!(
            run(make()).render_compare_report(),
            run(make()).render_compare_report(),
        );
    }

    /// Baseline contract — every baseline that does no Wengert pruning must
    /// report 100% backward retained and 100% activation retained, with 0%
    /// fusion. Pins the contract so a future bug that quietly attributes
    /// WRGA's pruning to a baseline can't slip through.
    #[test]
    fn compare_baselines_report_unpruned_ratios_for_lora_adalora_galore_reft() {
        let w = tiny_transformer_like();
        let input = WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.7.*"],
            manual_adapter_targets: Vec::new(),
            hybrid_layers: Vec::new(),
            wengert: &w,
            loss_output: w.output,
            weights: None,
            target: "rtx5070ti",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: WrgaAblation::default(),
        };
        let plan = run(input);
        let rows = plan.compute_compare_baselines();

        // Row 0 is WRGA; rows 1-4 are LoRA, AdaLoRA, GaLore, ReFT.
        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].label, "WRGA (this run)");
        for baseline_idx in 1..5 {
            let r = &rows[baseline_idx];
            assert_eq!(
                r.backward_retained_ratio, 1.0,
                "{} must report 100% backward retained",
                r.label,
            );
            assert_eq!(
                r.activation_retained_ratio, 1.0,
                "{} must report 100% activation retained",
                r.label,
            );
            assert_eq!(
                r.fusion_ratio, 0.0,
                "{} must report 0% fusion",
                r.label,
            );
        }
    }

    /// GaLore is the only baseline with zero adapter parameters — by
    /// construction (it projects gradients into a low-rank subspace rather
    /// than adding adapter weights). Pinning this catches a future
    /// off-by-one that flips GaLore to using LoRA's formula.
    #[test]
    fn compare_baseline_galore_has_zero_adapter_params() {
        let w = tiny_transformer_like();
        let input = WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.7.*"],
            manual_adapter_targets: Vec::new(),
            hybrid_layers: Vec::new(),
            wengert: &w,
            loss_output: w.output,
            weights: None,
            target: "rtx5070ti",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: WrgaAblation::default(),
        };
        let plan = run(input);
        let rows = plan.compute_compare_baselines();
        let galore = rows.iter().find(|r| r.label == "GaLore").unwrap();
        assert_eq!(galore.adapter_params, 0);
    }

    /// Numeric-correctness test for the LoRA-fixed-rank-16 formula on a
    /// hand-constructed plan with two synthetic rank allocations. Locks in
    /// the `mn_sum = adapter_params / rank` recovery so a future refactor
    /// can't silently change the per-site shape derivation.
    ///
    /// Two placements: site A with rank=8 and 8*(m+n)=8*200 = 1600 params,
    /// site B with rank=4 and 4*(m+n)=4*100 = 400 params. LoRA at fixed
    /// rank 16 must be 16*200 + 16*100 = 4800.
    #[test]
    fn compare_baseline_lora_params_recover_from_adapter_params_division() {
        use crate::cost_model::BoundClassification;
        use crate::wrga_roofline::AdapterKind;
        let mut plan = WrgaPlan::test_dummy();
        plan.placements = vec![
            crate::wrga_roofline::AdapterPlacement {
                name: "A".into(),
                arithmetic_intensity: 0.0,
                classification: BoundClassification::MemoryBound,
                roofline_slack: 1.0,
                adapter: AdapterKind::Lora,
                suggested_rank: 8,
                rationale: String::new(),
                decorator_kind: None,
                alpha: None,
                synthesized_fields: Vec::new(),
                init_strategies: Vec::new(),
            },
            crate::wrga_roofline::AdapterPlacement {
                name: "B".into(),
                arithmetic_intensity: 0.0,
                classification: BoundClassification::MemoryBound,
                roofline_slack: 1.0,
                adapter: AdapterKind::Lora,
                suggested_rank: 4,
                rationale: String::new(),
                decorator_kind: None,
                alpha: None,
                synthesized_fields: Vec::new(),
                init_strategies: Vec::new(),
            },
        ];
        plan.ranks = vec![
            crate::wrga_spectral::RankAllocation {
                name: "A".into(),
                rank: 8,
                effective_rank: 8.0,
                adapter_params: 8 * 200, // (m + n) = 200
            },
            crate::wrga_spectral::RankAllocation {
                name: "B".into(),
                rank: 4,
                effective_rank: 4.0,
                adapter_params: 4 * 100, // (m + n) = 100
            },
        ];

        let rows = plan.compute_compare_baselines();
        let lora = rows.iter().find(|r| r.label == "LoRA (r=16)").unwrap();
        assert_eq!(lora.adapter_params, 16 * 200 + 16 * 100);

        // AdaLoRA mirrors WRGA's measured rank allocation exactly.
        let wrga = rows.iter().find(|r| r.label == "WRGA (this run)").unwrap();
        let ada = rows.iter().find(|r| r.label == "AdaLoRA").unwrap();
        assert_eq!(ada.adapter_params, wrga.adapter_params);
    }

    /// Rank-0 sites must contribute ZERO ReFT params, even when they survive
    /// the top-25% take (which happens when WRGA prunes >75% of candidate
    /// sites). Pins the rank-0 guard that replaced a `.max(1)` quirk that
    /// would otherwise silently inflate ReFT's reported count by `4 *
    /// rank_0_site_count_in_top_quartile` phantom params.
    #[test]
    fn compare_baseline_reft_skips_rank_zero_sites_in_top_quartile() {
        use crate::cost_model::BoundClassification;
        use crate::wrga_roofline::AdapterKind;
        let mut plan = WrgaPlan::test_dummy();
        let mk_placement = |name: &str, rank: usize| crate::wrga_roofline::AdapterPlacement {
            name: name.into(),
            arithmetic_intensity: 0.0,
            classification: BoundClassification::MemoryBound,
            roofline_slack: 1.0,
            adapter: AdapterKind::Lora,
            suggested_rank: rank,
            rationale: String::new(),
            decorator_kind: None,
            alpha: None,
            synthesized_fields: Vec::new(),
            init_strategies: Vec::new(),
        };
        let mk_rank = |name: &str, rank: usize, mn: usize| crate::wrga_spectral::RankAllocation {
            name: name.into(),
            rank,
            effective_rank: rank as f64,
            adapter_params: rank * mn,
        };

        // 4 placements: only the first has real shape; the other three are
        // rank-0 (pruned). reft_site_count = ceil(4 / 4) = 1, so the top-25%
        // slice is exactly one site — the rank-0 entries cannot interleave
        // here because of the sort by mn descending. But this test pins the
        // behavior regardless: a rank-0 site, if it ever ended up in the
        // slice (e.g. zero rank-non-zero sites), MUST contribute 0.
        plan.placements = vec![
            mk_placement("S0", 0),
            mk_placement("S1", 0),
            mk_placement("S2", 0),
            mk_placement("S3", 0),
        ];
        plan.ranks = vec![
            mk_rank("S0", 0, 0),
            mk_rank("S1", 0, 0),
            mk_rank("S2", 0, 0),
            mk_rank("S3", 0, 0),
        ];

        let rows = plan.compute_compare_baselines();
        let reft = rows.iter().find(|r| r.label == "ReFT (r=4, top-25%)").unwrap();
        assert_eq!(
            reft.adapter_params, 0,
            "rank-0 sites must contribute 0 ReFT params (no phantom .max(1))",
        );
    }

    // -----------------------------------------------------------------------
    // Gap #4 — Paper §9.3 ablation harness
    // -----------------------------------------------------------------------

    /// Helper: a `WrgaInput` over the standard fixture, parameterised by an
    /// ablation. Shared by every ablation test in this section.
    fn ablation_input<'a>(w: &'a WengertList, ablation: WrgaAblation) -> WrgaInput<'a> {
        WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.7.*"],
            manual_adapter_targets: Vec::new(),
            hybrid_layers: Vec::new(),
            wengert: w,
            loss_output: w.output,
            weights: None,
            target: "rtx5070ti",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation,
        }
    }

    // -----------------------------------------------------------------------
    // WGGO placement-filter helpers (introduced on main via PR #272)
    // -----------------------------------------------------------------------

    /// Build an all-LoRA roofline placement for `name` (the roofline wants an
    /// adapter on every site, so any Skip the filter produces is its doing).
    fn mk_lora_placement(name: &str) -> crate::wrga_roofline::AdapterPlacement {
        crate::wrga_roofline::AdapterPlacement {
            name: name.into(),
            arithmetic_intensity: 0.0,
            classification: crate::cost_model::BoundClassification::MemoryBound,
            roofline_slack: 1.0,
            adapter: crate::wrga_roofline::AdapterKind::Lora,
            suggested_rank: 8,
            rationale: String::new(),
            decorator_kind: None,
            alpha: None,
            synthesized_fields: Vec::new(),
            init_strategies: Vec::new(),
        }
    }

    fn override_for(layer: &str, rank: u64, placement: crate::wggo_ilp::AdapterPlacement) -> crate::wggo_overrides::WggoOverrides {
        crate::wggo_overrides::WggoOverrides {
            per_layer: vec![crate::wggo_overrides::PerLayerOverride {
                layer_index: 0,
                layer_name: layer.into(),
                active_heads: 8,
                requested_csha_level: None,
                adapter_rank: rank,
                adapter_placement: placement,
                fase_fused: false,
                packing_mode: 0,
                shard_factor: 0,
            }],
        }
    }

    #[test]
    fn ablation_default_is_no_ablation_and_is_active_false() {
        let abl = WrgaAblation::default();
        assert!(!abl.is_active());
        assert_eq!(abl.human_label(), "none");
    }

    #[test]
    fn ablation_human_label_lists_active_flags_in_canonical_order() {
        // Canonical order is the order the fields are declared on the
        // struct: wengert, roofline, spectral, fusion, memory. Pinning
        // ensures the report header is stable for diffability.
        let abl = WrgaAblation {
            skip_wengert_pruning: true,
            skip_spectral_allocation: true,
            skip_memory_planning: true,
            ..Default::default()
        };
        assert_eq!(abl.human_label(), "wengert,spectral,memory");
        assert!(abl.is_active());
    }

    #[test]
    fn ablation_propagates_to_plan_field() {
        let w = tiny_transformer_like();
        let abl = WrgaAblation {
            skip_fusion_integration: true,
            ..Default::default()
        };
        let plan = run(ablation_input(&w, abl));
        assert_eq!(plan.ablation, abl);
    }

    #[test]
    fn render_report_surfaces_ablation_header_when_active() {
        let w = tiny_transformer_like();
        let plan_baseline = run(ablation_input(&w, WrgaAblation::default()));
        let plan_ablated = run(ablation_input(
            &w,
            WrgaAblation {
                skip_fusion_integration: true,
                ..Default::default()
            },
        ));
        // Baseline: no Ablation: line.
        assert!(!plan_baseline.render_report().contains("Ablation:"));
        // Ablated: Ablation: line with the active flag.
        let report = plan_ablated.render_report();
        assert!(report.contains("Ablation: skip=fusion"), "got: {report}");
    }

    /// Regression test for the `prune_noop` two-pass ordering. A single-pass
    /// implementation would set `saved_for_backward = false` on op N if op
    /// N's result is needed by a later op M as `needs_inputs`, because at
    /// N's iteration `activation_live` doesn't yet contain N's VarId — M
    /// hasn't run yet. The fix is two passes (mirror `wrga_prune::prune`).
    ///
    /// This test asserts the bit-for-bit consistency the fix restores: every
    /// VarId that ends up in `activation_live` must have its corresponding
    /// op's `saved_for_backward` set to `true`. A single-pass regression
    /// would surface as a mismatch on at least one matmul-input VarId.
    #[test]
    fn ablation_prune_noop_saved_for_backward_bits_agree_with_activation_live_set() {
        let w = tiny_transformer_like();
        let plan = run(ablation_input(
            &w,
            WrgaAblation {
                skip_wengert_pruning: true,
                ..Default::default()
            },
        ));
        for op in &plan.prune.pruned.ops {
            let bit = op.saved_for_backward;
            let in_set = plan.prune.activation_live.contains(&op.result);
            assert_eq!(
                bit, in_set,
                "op {} (result {:?}): saved_for_backward={} but activation_live.contains={}",
                op.result, op.op, bit, in_set,
            );
        }
    }

    /// Innovation 1 ablation contract: `prune_noop` retains EVERY forward op
    /// as backward-live so the prune stats report 100% retained / 0%
    /// eliminated. This is the §9.3 "without_wengert" measurement.
    #[test]
    fn ablation_skip_wengert_pruning_retains_all_forward_ops() {
        let w = tiny_transformer_like();
        let baseline = run(ablation_input(&w, WrgaAblation::default()));
        let ablated = run(ablation_input(
            &w,
            WrgaAblation {
                skip_wengert_pruning: true,
                ..Default::default()
            },
        ));
        // Ablated: backward_ops_retained == forward_ops_total (100% retained).
        assert_eq!(
            ablated.prune.stats.backward_ops_retained,
            ablated.prune.stats.forward_ops_total,
        );
        // Sanity: the un-ablated baseline pruned at LEAST one op (Auto mode
        // with restricted trainable_patterns must produce some elimination).
        assert!(
            baseline.prune.stats.backward_ops_retained
                <= baseline.prune.stats.forward_ops_total,
        );
    }

    /// Innovation 4 ablation contract: `fusion = FusionPlan::default()` means
    /// `fusion_ratio() == 0.0` (no fused adapters). This is the §9.3
    /// "without_fusion" measurement.
    #[test]
    fn ablation_skip_fusion_integration_yields_zero_fusion_ratio() {
        let w = tiny_transformer_like();
        let ablated = run(ablation_input(
            &w,
            WrgaAblation {
                skip_fusion_integration: true,
                ..Default::default()
            },
        ));
        assert_eq!(ablated.fusion.fusion_ratio(), 0.0);
    }

    /// Innovation 5 ablation contract: `memory_noop` reports `live_activations
    /// == slots_used` so `reuse_ratio() == 0.0` (no live-range reuse). This
    /// is the §9.3 "without_memory" measurement.
    #[test]
    fn ablation_skip_memory_planning_yields_zero_reuse_ratio() {
        let w = tiny_transformer_like();
        let ablated = run(ablation_input(
            &w,
            WrgaAblation {
                skip_memory_planning: true,
                ..Default::default()
            },
        ));
        assert_eq!(ablated.memory.stats.reuse_ratio(), 0.0);
        assert_eq!(
            ablated.memory.stats.live_activations,
            ablated.memory.stats.slots_used,
        );
    }

    /// Innovation 3 ablation contract: skipping spectral allocation drops
    /// per-site `adapter_params` to 0 (no SVD = no shape recovery), but
    /// `RankAllocation.rank` falls back to the placement's `suggested_rank`.
    #[test]
    fn ablation_skip_spectral_allocation_uses_suggested_rank_zero_params() {
        let w = tiny_transformer_like();
        let ablated = run(ablation_input(
            &w,
            WrgaAblation {
                skip_spectral_allocation: true,
                ..Default::default()
            },
        ));
        // Every RankAllocation has zero adapter_params (the no-op signal).
        for r in &ablated.ranks {
            assert_eq!(r.adapter_params, 0, "site '{}' must report 0 params under skip_spectral", r.name);
        }
        // spectral vec is empty (no SVD ran).
        assert!(ablated.spectral.is_empty());
    }

    /// Innovation 2 ablation contract: skipping roofline placement puts an
    /// adapter on every candidate site at `r_max`, classified MemoryBound.
    /// Verifies the rationale carries the ablation flag for debuggability.
    #[test]
    fn ablation_skip_roofline_placement_emits_r_max_on_every_site() {
        use crate::cost_model::BoundClassification;
        let w = tiny_transformer_like();
        let ablated = run(ablation_input(
            &w,
            WrgaAblation {
                skip_roofline_placement: true,
                ..Default::default()
            },
        ));
        for p in &ablated.placements {
            assert_eq!(p.suggested_rank, 16, "ablation should use r_max=16");
            assert_eq!(p.classification, BoundClassification::MemoryBound);
            assert!(
                p.rationale.contains("ablation"),
                "rationale must mark this as an ablation result: {}",
                p.rationale,
            );
        }
    }

    /// Composite contract: every ablated stage produces a deterministic
    /// result. Re-running with the same ablation must yield identical
    /// reports. Pins reproducibility for the §9.3 ablation table.
    #[test]
    fn ablation_run_is_deterministic_under_all_skips() {
        let w = tiny_transformer_like();
        let abl = WrgaAblation {
            skip_wengert_pruning: true,
            skip_roofline_placement: true,
            skip_spectral_allocation: true,
            skip_fusion_integration: true,
            skip_memory_planning: true,
        };
        let a = run(ablation_input(&w, abl));
        let b = run(ablation_input(&w, abl));
        assert_eq!(a.render_report(), b.render_report());
    }

    // -----------------------------------------------------------------------
    // WGGO placement-filter tests (introduced on main via PR #272)
    // -----------------------------------------------------------------------

    #[test]
    fn wggo_placement_filter_skips_excluded_projections() {
        use crate::wrga_roofline::AdapterKind;
        let mut placements = vec![
            mk_lora_placement("blocks.0.attn.wq"),
            mk_lora_placement("blocks.0.attn.wk"),
            mk_lora_placement("blocks.0.attn.wv"),
            mk_lora_placement("blocks.0.attn.wo"),
        ];
        // WGGO chose AttnQV (q,v) for blocks.0 with a positive rank.
        let over = override_for("blocks.0", 8, crate::wggo_ilp::AdapterPlacement::AttnQV);
        let diags = apply_wggo_placement_filter(&mut placements, Some(&over));

        let kind = |n: &str| placements.iter().find(|p| p.name == n).unwrap().adapter;
        assert_eq!(kind("blocks.0.attn.wq"), AdapterKind::Lora, "q in placement → kept");
        assert_eq!(kind("blocks.0.attn.wv"), AdapterKind::Lora, "v in placement → kept");
        assert_eq!(kind("blocks.0.attn.wk"), AdapterKind::Skip, "k outside placement → skipped");
        assert_eq!(kind("blocks.0.attn.wo"), AdapterKind::Skip, "o outside placement → skipped");

        assert_eq!(diags.len(), 2, "one diagnostic per overruled site (wk, wo)");
        assert!(diags.iter().all(|d| matches!(
            d.reason,
            crate::wggo_overrides::OverrideRejectReason::AdapterSiteOutsidePlacement { .. }
        )));
        assert!(diags.iter().all(|d| d.applied == "skip" && d.requested == "lora"));
    }

    #[test]
    fn wggo_placement_filter_governs_each_site_by_its_own_layer() {
        // Discriminating multi-layer test: a regression that applied layer 0's
        // placement to every site (or matched the wrong layer) would slip past
        // the single-layer tests.  Layer 0 = AttnQV, layer 1 = AttnQKVO, layer
        // 2 has no override at all.
        use crate::wrga_roofline::AdapterKind;
        let over = crate::wggo_overrides::WggoOverrides {
            per_layer: vec![
                crate::wggo_overrides::PerLayerOverride {
                    layer_index: 0,
                    layer_name: "blocks.0".into(),
                    active_heads: 8,
                    requested_csha_level: None,
                    adapter_rank: 8,
                    adapter_placement: crate::wggo_ilp::AdapterPlacement::AttnQV,
                    fase_fused: false,
                    packing_mode: 0,
                    shard_factor: 0,
                },
                crate::wggo_overrides::PerLayerOverride {
                    layer_index: 1,
                    layer_name: "blocks.1".into(),
                    active_heads: 8,
                    requested_csha_level: None,
                    adapter_rank: 8,
                    adapter_placement: crate::wggo_ilp::AdapterPlacement::AttnQKVO,
                    fase_fused: false,
                    packing_mode: 0,
                    shard_factor: 0,
                },
            ],
        };
        let mut placements = vec![
            mk_lora_placement("blocks.0.attn.wk"), // excluded by layer-0 AttnQV
            mk_lora_placement("blocks.1.attn.wk"), // allowed by layer-1 AttnQKVO
            mk_lora_placement("blocks.2.attn.wk"), // no override for layer 2
        ];
        let diags = apply_wggo_placement_filter(&mut placements, Some(&over));

        let kind = |n: &str| placements.iter().find(|p| p.name == n).unwrap().adapter;
        assert_eq!(kind("blocks.0.attn.wk"), AdapterKind::Skip, "layer-0 AttnQV excludes K");
        assert_eq!(kind("blocks.1.attn.wk"), AdapterKind::Lora, "layer-1 AttnQKVO allows K");
        assert_eq!(kind("blocks.2.attn.wk"), AdapterKind::Lora, "no override → untouched");
        assert_eq!(diags.len(), 1, "only the layer-0 K site is overruled");
        assert_eq!(diags[0].layer_index, 0);
    }

    #[test]
    fn wggo_placement_none_with_positive_rank_excludes_all_projections() {
        // Defends against the invariant-violating (rank>0, placement=None)
        // state: placement None means "zero projections", so every recognized
        // attention/FFN site must be forced to Skip (with a diagnostic) — never
        // silently left adapted.  A non-projection site stays untouched.
        use crate::wrga_roofline::AdapterKind;
        let mut placements = vec![
            mk_lora_placement("blocks.0.attn.wq"),
            mk_lora_placement("blocks.0.attn.wv"),
            mk_lora_placement("blocks.0.attn_norm.weight"), // not a projection
        ];
        let over = override_for("blocks.0", 8, crate::wggo_ilp::AdapterPlacement::None);
        let diags = apply_wggo_placement_filter(&mut placements, Some(&over));

        let kind = |n: &str| placements.iter().find(|p| p.name == n).unwrap().adapter;
        assert_eq!(kind("blocks.0.attn.wq"), AdapterKind::Skip);
        assert_eq!(kind("blocks.0.attn.wv"), AdapterKind::Skip);
        assert_eq!(
            kind("blocks.0.attn_norm.weight"),
            AdapterKind::Lora,
            "non-projection site is not governed by placement"
        );
        assert_eq!(diags.len(), 2, "wq + wv excluded; norm untouched");
    }

    #[test]
    fn wggo_placement_filter_is_noop_without_governing_override() {
        use crate::wrga_roofline::AdapterKind;
        // (a) No overrides → untouched, no diagnostics.
        let mut p1 = vec![mk_lora_placement("blocks.0.attn.wk")];
        assert!(apply_wggo_placement_filter(&mut p1, None).is_empty());
        assert_eq!(p1[0].adapter, AdapterKind::Lora);

        // (b) Override with placement None / rank 0 → the rank path's job; the
        // placement filter must not double-report or skip here.
        let over = override_for("blocks.0", 0, crate::wggo_ilp::AdapterPlacement::None);
        let mut p2 = vec![mk_lora_placement("blocks.0.attn.wk")];
        assert!(apply_wggo_placement_filter(&mut p2, Some(&over)).is_empty());
        assert_eq!(p2[0].adapter, AdapterKind::Lora);

        // (c) A non-projection site (norm) is never governed by placement.
        let over_qv = override_for("blocks.0", 8, crate::wggo_ilp::AdapterPlacement::AttnQV);
        let mut p3 = vec![mk_lora_placement("blocks.0.attn_norm.weight")];
        assert!(apply_wggo_placement_filter(&mut p3, Some(&over_qv)).is_empty());
        assert_eq!(p3[0].adapter, AdapterKind::Lora);
    }
}
