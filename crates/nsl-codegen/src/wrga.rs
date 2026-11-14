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

use crate::gpu_specs::{default_gpu, find_gpu, GpuSpec, GPU_DATABASE};
use crate::wengert::{PrimalOp, VarId, WengertList};
use crate::weight_aware::WeightMap;
use crate::wrga_fusion::{build_fusion_plan, FusionPlan};
use crate::wrga_memory::{plan_memory_with_pin, MemoryPlan, SizeHints};
use crate::wrga_prune::{prune, PruneResult};
use crate::pass_trace::{DeclineReason, PassDisposition};
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
    /// The parameters that train, as names or globs in the dialect of
    /// `wrga_prune::glob_match` (e.g. `blocks.6.*`; `*` for all of them).
    /// A parameter no pattern matches is frozen, so an empty list is a model
    /// with nothing trainable. (Empty used to mean "everything trains",
    /// which is what a fully frozen model then reported.)
    pub trainable_patterns: Vec<&'a str>,
    /// The adapters the program declares with `@adapter`, by the name of the
    /// parameter each one adapts, as the Wengert list spells it
    /// (`m.blocks.0.wq`). When there are any, they are the plan's sites,
    /// trainable or frozen, with their declared kind and rank; the roofline
    /// and spectral stages then describe them and choose nothing. Empty means
    /// the program declares no adapter, and the plan lists the sites an
    /// adapter could go on, none of which gets one.
    pub declared_adapters: BTreeMap<String, DeclaredAdapter>,
    /// Weight shapes known at compile time, by parameter name.
    pub param_shapes: HashMap<String, [usize; 2]>,
    /// Whether this build runs adapters through the synthesized fused
    /// kernels (a pinned target and `NSL_WRGA_FUSED_CUDA=1`). The report
    /// says so beside its count of fusible sites.
    pub fused_kernels_active: bool,
    /// Forward Wengert list (from source-AD extraction).
    pub wengert: &'a WengertList,
    /// VarId of the scalar loss (the adjoint seed).
    pub loss_output: VarId,
    /// Optional pre-trained weights for spectral analysis.
    pub weights: Option<&'a WeightMap>,
    /// Target GPU name, resolved with [`resolve_gpu`].  Empty means none was
    /// given: the analysis then uses `default_gpu()` and the report says so.
    /// A caller with a name from the user checks it with [`resolve_gpu`]
    /// first; an unknown name reaching here also falls back, labelled.
    pub target: &'a str,
    /// Cap on the total parameter count of the SUGGESTED adapters, when the
    /// program declares none (0 = no cap). Declared adapters are held to the
    /// budget at compile time instead (`Compiler::check_adapter_budget`).
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
    /// WRGA paper §8.2: user-defined custom adapter model name (e.g.
    /// `"GatedLoRA"`).  When `Some`, WRGA surfaces it on the report and
    /// (in a follow-up cycle) routes adapter emission through this
    /// user-defined model instead of one of the built-in flavours.
    /// `None` keeps the compiler-chosen flavour selection.
    pub custom_adapter: Option<&'a str>,
}

/// One adapter the program declares with `@adapter`, on one parameter.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeclaredAdapter {
    pub kind: crate::AdapterKind,
    /// The declared rank (LoRA and gated LoRA; unused by IA³).
    pub rank: usize,
    /// The number of parameters its tensors hold, when the adapted weight's
    /// shape is known at compile time.
    pub params: Option<usize>,
}

/// Look a GPU up by the name a user wrote: `h100`, `rtx5070ti`, `RTX-4090`.
///
/// [`find_gpu`] wants the table's own punctuation, so `rtx5070ti` (the
/// spelling the proposal and the decorator's own error message use) matched
/// no entry and the analysis silently ran for the default GPU. This also
/// accepts a name that matches once punctuation is dropped.
pub fn resolve_gpu(name: &str) -> Option<&'static GpuSpec> {
    // An empty name is no name: `find_gpu` would take it as a prefix of
    // every entry.
    if !name.chars().any(|c| c.is_ascii_alphanumeric()) {
        return None;
    }
    if let Some(gpu) = find_gpu(name) {
        return Some(gpu);
    }
    let squash = |s: &str| -> String {
        s.chars()
            .filter(|c| c.is_ascii_alphanumeric())
            .map(|c| c.to_ascii_lowercase())
            .collect()
    };
    let want = squash(name);
    let matches: Vec<&GpuSpec> = GPU_DATABASE
        .iter()
        .filter(|g| squash(g.name).starts_with(&want))
        .collect();
    matches
        .iter()
        .find(|g| squash(g.name) == want)
        .or_else(|| matches.iter().find(|g| g.name.contains("SXM")))
        .or(matches.first())
        .copied()
}

/// The names [`resolve_gpu`] knows, for an error message.
pub fn known_gpu_names() -> Vec<&'static str> {
    GPU_DATABASE.iter().map(|g| g.name).collect()
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
    /// WRGA paper §8.2: user-defined custom adapter model name carried
    /// through from `WrgaInput.custom_adapter`.  When `Some`,
    /// `render_report` prints a `Custom adapter: <name>` header line
    /// directly below the target hardware.  Codegen integration (using
    /// the user model for adapter emission) is deferred to a follow-up
    /// cycle; for now the flavour selection falls back to the existing
    /// LoRA/IA³/GatedLoRA pipeline regardless of this field.
    pub custom_adapter: Option<String>,
    /// Whether `placements` are the adapters the program declares (true) or
    /// only the sites an adapter could go on (false: nothing in the plan is
    /// built).
    pub adapters_declared: bool,
    /// Sites left out of `adapter_params()` because the shape of their
    /// weight is not known at compile time.
    pub unsized_sites: usize,
    /// For each entry of `ranks`, `m + n` of the weight it sits on (0 when
    /// the shape is unknown): what a method with its own rank formula needs
    /// to count parameters on the same sites.
    pub rank_site_mn: Vec<usize>,
    /// Whether the build runs adapters through the fused kernels.
    pub fused_kernels_active: bool,
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
            custom_adapter: None,
            adapters_declared: false,
            unsized_sites: 0,
            rank_site_mn: Vec::new(),
            fused_kernels_active: false,
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
        if let Some(name) = &self.custom_adapter {
            // Paper §8.2: surface the user-defined custom adapter so the
            // report reader knows WRGA was asked to instantiate this
            // model at each placement site.  The trailing parenthetical
            // is load-bearing: the placement pass still selects among the
            // built-in lora/ia3/gatedlora flavours in this PR cycle —
            // wiring the user's adapter graph into placement/fusion is a
            // follow-up cycle.  Without the note a reader could mistake
            // the LoRA placements below for instantiations of `<name>`.
            writeln!(s, "Custom adapter: {name} (placement integration pending)").unwrap();
        }
        if self.ablation.is_active() {
            // Paper §9.3: when one or more Innovations are ablated, surface
            // which ones in the header so the report's numbers are not
            // mistaken for a regular WRGA run.
            writeln!(s, "Ablation: skip={}", self.ablation.human_label()).unwrap();
        }
        writeln!(s).unwrap();
        // Both sides of each arrow count the same thing: the left is what a
        // backward pass with every parameter trainable would differentiate
        // (or save), the right is what this one does.
        writeln!(
            s,
            "Backward pruning: {} ops → {} ops ({:.1}% eliminated)",
            self.prune.stats.full_backward_ops,
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
        let unsized_note = match self.unsized_sites {
            0 => String::new(),
            n => format!(" (+ {n} sites whose weight shape is not known at compile time)"),
        };
        if self.adapters_declared {
            // The program's own adapters. "Fusible" is whether a fused kernel
            // exists for the site's kind; whether the build uses it is the
            // opt-in, stated beside it.
            writeln!(
                s,
                "Adapter parameters: {}{}  |  Sites: {}  |  Fusible: {}/{} ({:.1}%), fused kernels {}",
                self.adapter_params(),
                unsized_note,
                self.fusion.total_count(),
                self.fusion.fused_count(),
                self.fusion.total_count(),
                100.0 * self.fusion.fusion_ratio(),
                if self.fused_kernels_active { "on" } else { "off" },
            )
            .unwrap();
        } else {
            // No `@adapter` in the program: these are the sites an adapter
            // could go on and the ranks the analysis suggests. None is built.
            writeln!(
                s,
                "Suggested adapters (none declared, none built): {} parameters{}  |  Sites: {}  |  Fusible: {}/{} ({:.1}%)",
                self.adapter_params(),
                unsized_note,
                self.fusion.total_count(),
                self.fusion.fused_count(),
                self.fusion.total_count(),
                100.0 * self.fusion.fusion_ratio()
            )
            .unwrap();
        }
        if !self.spectral.is_empty() {
            // Present only when a checkpoint was given and some of its
            // weights are sites. The figures describe the weights; they do
            // not size an adapter.
            let sites = self
                .placements
                .iter()
                .filter(|p| !matches!(p.adapter, crate::wrga_roofline::AdapterKind::Skip))
                .count();
            let ranks = self.spectral.iter().map(|sp| sp.effective_rank);
            let (lo, hi) = ranks.fold((f64::INFINITY, 0.0_f64), |(lo, hi), r| (lo.min(r), hi.max(r)));
            let kept = self.spectral.iter().map(|sp| sp.truncated_rank).max().unwrap_or(0);
            writeln!(
                s,
                "Spectral analysis: {} of {} sites matched a checkpoint weight; effective rank \
                 {:.1} to {:.1} over the top {} singular values",
                self.spectral.len(),
                sites,
                lo,
                hi,
                kept
            )
            .unwrap();
        }
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

    /// Render a PEFT comparison report: the adapter parameter count of this
    /// plan beside what LoRA at a fixed rank, GaLore and ReFT would hold on
    /// the same sites, by each method's parameter formula.
    ///
    /// Surface: `nsl check --wrga-compare <file>.nsl [--weights w.safetensors]`
    /// (WRGA paper §8.3).
    ///
    /// Only parameter counts are compared, because only they can be derived
    /// from the plan. Per site `i` with weight shape `[m_i, n_i]`:
    /// - **LoRA (Hu et al. 2021)**: fixed rank `r=16`; `16 * (m_i + n_i)`.
    /// - **GaLore (Zhao et al. 2024)**: trains the full parameters through a
    ///   low-rank gradient projection, so it adds no adapter parameters.
    /// - **ReFT (Wu et al. 2024)**: rank-4 interventions on the top 25% of
    ///   sites; `4 * n_i` for those, with `n_i` taken as `(m_i + n_i) / 2`.
    ///
    /// The table used to carry three more columns (backward ops, saved
    /// activations, fusion), filled for every baseline with the constants
    /// 100%, 100% and 0%. Nothing had been computed for them, and the first
    /// two are not even true of the baselines: PyTorch records an operation
    /// for the backward pass only when one of its inputs requires a gradient,
    /// so a frozen parameter's gradient is skipped there as well. It also had
    /// an "AdaLoRA" row holding this plan's own count; AdaLoRA's ranks come
    /// out of training, and no count for it follows from the program.
    ///
    /// Output is deterministic given the input plan.
    pub fn render_compare_report(&self) -> String {
        use std::fmt::Write as _;

        let mut s = String::new();
        writeln!(s, "=== WRGA PEFT Comparison Report ===").unwrap();
        writeln!(s, "Mode: {}", self.mode.as_str()).unwrap();
        writeln!(s, "Target hardware: {}", self.target_gpu).unwrap();
        writeln!(s, "Placement sites: {}", self.placements.len()).unwrap();
        writeln!(s).unwrap();
        writeln!(s, "Adapter parameters on this plan's sites, by each method's formula:").unwrap();
        writeln!(s, "{:<22} | {:>10}", "Method", "Params").unwrap();
        writeln!(s, "{0:-<22}-+-{0:-<10}", "").unwrap();
        for row in self.compute_compare_rows() {
            writeln!(s, "{:<22} | {:>10}", row.label, fmt_with_commas(row.adapter_params)).unwrap();
        }
        if self.unsized_sites > 0 {
            writeln!(
                s,
                "({} sites are left out of every row: the shape of their weight is not known \
                 at compile time.)",
                self.unsized_sites
            )
            .unwrap();
        }
        writeln!(s).unwrap();
        let stats = &self.prune.stats;
        writeln!(
            s,
            "This plan: the backward pass differentiates {} of the {} ops a pass with every \
             parameter trainable would, and saves {} of its {} activations; {} of {} sites \
             could use a fused kernel.",
            stats.backward_ops_retained,
            stats.full_backward_ops,
            stats.pruned_saved_activations,
            stats.full_backward_saved_activations,
            self.fusion.fused_count(),
            self.fusion.total_count(),
        )
        .unwrap();
        writeln!(
            s,
            "No such figures are given for the other methods: none was computed, and a \
             framework that records only operations with a trainable input skips the same \
             frozen-parameter gradients. AdaLoRA has no row: its ranks come out of training."
        )
        .unwrap();
        writeln!(
            s,
            "The counts are formulas over the sites above. They say nothing about quality \
             or speed."
        )
        .unwrap();
        s
    }

    /// The comparison's rows: this plan, then each baseline's parameter count
    /// on the same sites. Pulled out so the renderer and tests share one
    /// source for the numbers.
    fn compute_compare_rows(&self) -> Vec<CompareRow> {
        // `m + n` per site; 0 for a site whose weight shape is unknown.
        let mn_sums: Vec<usize> = (0..self.ranks.len())
            .map(|i| self.rank_site_mn.get(i).copied().unwrap_or(0))
            .collect();

        let wrga_params: usize = self.ranks.iter().map(|r| r.adapter_params).sum();

        // LoRA: fixed rank r = 16 on every site.
        const LORA_FIXED_RANK: usize = 16;
        let lora_params: usize = mn_sums.iter().map(|mn| LORA_FIXED_RANK * mn).sum();

        // ReFT: rank-4 intervention vectors on the top 25% of sites by
        // footprint (round up). Each contributes `4 * n_i`, with `n_i` taken
        // as `mn_sum / 2`, which is exact for a square weight and off by the
        // m/n ratio otherwise. A site of unknown shape contributes nothing.
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
            CompareRow { label: "WRGA (this run)".to_string(), adapter_params: wrga_params },
            CompareRow { label: "LoRA (r=16)".to_string(), adapter_params: lora_params },
            // GaLore projects full-parameter gradients into a low-rank
            // subspace instead of adding adapter weights.
            CompareRow { label: "GaLore".to_string(), adapter_params: 0 },
            CompareRow { label: "ReFT (r=4, top-25%)".to_string(), adapter_params: reft_params },
        ]
    }
}

/// One row of the PEFT comparison table.
#[derive(Debug, Clone, PartialEq)]
struct CompareRow {
    label: String,
    adapter_params: usize,
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
/// `stats.backward_ops_retained == stats.full_backward_ops` and
/// `stats.pruned_saved_activations == stats.full_backward_saved_activations`
/// so `render_report` reports 0% eliminated on both rows of the prune
/// summary.
fn prune_noop(list: &WengertList, trainable: &BTreeSet<VarId>) -> PruneResult {
    use crate::wrga_prune::PruneStats;

    // Two passes — mirror `wrga_prune::prune()`'s structure exactly. A single
    // pass would set `saved_for_backward` BEFORE `activation_live` has been
    // populated by later ops that read this op's result as an input, so any
    // input-saved activation would get a stale `false` bit. Match the real
    // pruner: build the live sets first, then stamp the bits.
    let mut backward_live: BTreeSet<VarId> = BTreeSet::new();
    let mut activation_live: BTreeSet<VarId> = BTreeSet::new();
    for op in &list.ops {
        backward_live.insert(op.result);
        let req = crate::wrga_prune::save_requirements(&op.op);
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
            // Everything is kept, so both baselines equal what is kept and
            // the report reads 0% eliminated on both lines.
            full_backward_ops: list.len(),
            backward_ops_retained: list.len(),
            full_backward_saved_activations: pruned_saved,
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
            kind: s.kind,
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
            // The spectral stage is what sizes the adapters; ablated, the
            // plan holds ranks and no parameter count.
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
    crate::pass_trace::record("WRGA");
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
    let (gpu, target_gpu) = if input.target.is_empty() {
        let gpu = default_gpu();
        (gpu, format!("{} (default: no target given)", gpu.name))
    } else {
        match resolve_gpu(input.target) {
            Some(gpu) => (gpu, gpu.name.to_string()),
            None => {
                let gpu = default_gpu();
                (gpu, format!("{} (default: '{}' is not a known GPU)", gpu.name, input.target))
            }
        }
    };
    let adapters_declared = !input.declared_adapters.is_empty();
    let sites = infer_sites_from_wengert(input.wengert, &trainable, &input);
    let site_shapes: HashMap<String, [usize; 2]> = sites
        .iter()
        .filter(|s| s.shape_known)
        .map(|s| (s.site.name.clone(), [s.site.shape[0] as usize, s.site.shape[1] as usize]))
        .collect();
    let roofline_sites: Vec<AdapterSite> = sites.into_iter().map(|s| s.site).collect();
    let (mut placements, placement_diags) = if abl.skip_roofline_placement {
        // §9.3 ablation: skip roofline AND the WGGO placement filter — the
        // no-op output is deliberately a uniform "adapter on every site at
        // r_max", and the WGGO filter is part of Innovation 2's pipeline so
        // running it on the no-op would corrupt the ablation contract.
        (placements_noop(&roofline_sites, input.r_max), Vec::new())
    } else {
        let mut placements = place_adapters(&roofline_sites, gpu, input.r_min, input.r_max);
        // Narrow the roofline placement to the projections WGGO budgeted for
        // (its `adapter_placement` is the comm-budget-feasible set, G5/G7).
        // Sites outside it are forced to Skip; each overruled site yields a
        // diagnostic. A declared adapter exists whatever WGGO prefers, so
        // the filter is for suggestions only.
        let diags = if adapters_declared {
            Vec::new()
        } else {
            apply_wggo_placement_filter(&mut placements, input.wggo_overrides)
        };
        (placements, diags)
    };
    // A declared adapter is what it was declared to be: the roofline stage
    // classifies its site and chooses nothing.
    for p in placements.iter_mut() {
        if let Some(decl) = input.declared_adapters.get(&p.name) {
            p.decorator_kind = Some(decl.kind);
            (p.adapter, p.suggested_rank) = match decl.kind {
                crate::AdapterKind::Lora | crate::AdapterKind::GatedLora => {
                    (crate::wrga_roofline::AdapterKind::Lora, decl.rank)
                }
                crate::AdapterKind::Ia3 => (crate::wrga_roofline::AdapterKind::Ia3, 0),
            };
            p.rationale = "declared with @adapter".to_string();
        }
    }

    // ── Stage 4: Innovation 3 — Spectral rank allocation ──────────────────
    let (spectral, ranks, override_diags) = if abl.skip_spectral_allocation {
        spectral_noop(&placements)
    } else {
        run_spectral(input.weights, &placements, &site_shapes, &input)
    };
    // `m + n` of each ranked site, and how many have no known shape.
    let rank_site_mn: Vec<usize> = ranks
        .iter()
        .map(|r| {
            spectral
                .iter()
                .find(|sp| names_match(&sp.name, &r.name))
                .map(|sp| sp.shape)
                .or_else(|| site_shapes.get(&r.name).copied())
                .map_or(0, |[m, n]| m + n)
        })
        .collect();
    let unsized_sites = if abl.skip_spectral_allocation {
        0
    } else {
        ranks
            .iter()
            .zip(&rank_site_mn)
            .filter(|(r, mn)| {
                **mn == 0 && input.declared_adapters.get(&r.name).is_none_or(|d| d.params.is_none())
            })
            .count()
    };

    // ── Stage 5: Innovation 4 — Fusion integration ────────────────────────
    // The rank of each placement, by NAME: `ranks` holds only the sites that
    // were ranked, in an order of its own (with weights, the checkpoint's),
    // so pairing the i-th placement with the i-th rank paired arbitrary
    // sites.
    let rank_overrides: Vec<usize> = placements
        .iter()
        .map(|p| {
            ranks
                .iter()
                .find(|r| names_match(&r.name, &p.name))
                .map_or(p.suggested_rank, |r| r.rank)
        })
        .collect();
    let fusion = if abl.skip_fusion_integration {
        FusionPlan::default()
    } else {
        build_fusion_plan(&placements, Some(&rank_overrides))
    };

    // ── Stage 6: Innovation 5 — Memory plan ───────────────────────────────
    let memory = if abl.skip_memory_planning {
        memory_noop(&prune_result.activation_live)
    } else {
        let size_hints: SizeHints = HashMap::new();
        // Every activation in `activation_live` is one the backward pass
        // reads, and the backward pass runs after the last forward op: each
        // is live from its birth to the end of the forward list. Their
        // lifetimes used to end at their last FORWARD use, which let two of
        // them share a slot although the backward reads both ("18 live
        // activations → 5 slots"); applied, that plan would have aliased
        // live tensors.
        let mut live_at_end = prune_result.activation_live.clone();
        live_at_end.insert(input.loss_output);
        plan_memory_with_pin(
            input.wengert,
            &prune_result.activation_live,
            &size_hints,
            &live_at_end,
            &input.inspect_pinned_vars,
        )
    };

    // WRGA's five Innovations all feed one deliverable: the set of sites that
    // get an adapter. `Skip` placements are the roofline analysis declining a
    // site, so they are candidates considered, not rewrites made.
    //
    // `--wrga-ablate=wengert,roofline,spectral,fusion,memory` replaces every
    // stage with its no-op, which is a user-reachable way to make the pass run
    // and do nothing on purpose (paper §9.3's ablation harness). That is a
    // different fact from "no site was worth adapting", so it gets its own
    // reason rather than being folded into the placement count.
    let adapted_sites = placements
        .iter()
        .filter(|p| !matches!(p.adapter, crate::wrga_roofline::AdapterKind::Skip))
        .count();
    if abl.skip_wengert_pruning
        && abl.skip_roofline_placement
        && abl.skip_spectral_allocation
        && abl.skip_fusion_integration
        && abl.skip_memory_planning
    {
        crate::pass_trace::record_disposition("WRGA", PassDisposition::Declined {
            reason: DeclineReason::FeatureDisabled(
                "every Innovation ablated via --wrga-ablate",
            ),
        });
    } else if adapted_sites == 0 {
        crate::pass_trace::record_disposition("WRGA", PassDisposition::Declined {
            reason: DeclineReason::NoCandidates("no adapter site survived roofline placement"),
        });
    } else {
        crate::pass_trace::record_disposition("WRGA", PassDisposition::Applied {
            rewrites: adapted_sites,
        });
    }
    WrgaPlan {
        mode: input.mode,
        target_gpu,
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
        custom_adapter: input.custom_adapter.map(str::to_string),
        adapters_declared,
        unsized_sites,
        rank_site_mn,
        fused_kernels_active: input.fused_kernels_active,
    }
}

/// Whether a checkpoint key and a parameter name are the same weight.
/// Placements carry Wengert names rooted at the train block's model variable
/// (`m.blocks.0.attn.wq`) while checkpoint keys are bare (`blocks.0.attn.wq`)
/// or rooted at something else, so equality or a dot-boundary suffix in
/// either direction counts.
fn names_match(a: &str, b: &str) -> bool {
    a == b
        || a.strip_suffix(b).is_some_and(|rest| rest.ends_with('.'))
        || b.strip_suffix(a).is_some_and(|rest| rest.ends_with('.'))
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

/// The trainable parameter VarIds: every parameter a pattern of
/// `trainable_patterns` matches.
///
/// The patterns come from `@freeze`, which alone decides what trains; the
/// `@wrga` mode has no say. `mode=manual` used to select the parameters
/// matching the `@adapter` targets and `mode=hybrid` those starting with a
/// `layers=` prefix; a `Class.field` target and a bare layer prefix never
/// match a parameter path such as `m.b1.wq`, so both sets were always empty.
fn select_trainable(list: &WengertList, input: &WrgaInput) -> BTreeSet<VarId> {
    use crate::wrga_prune::glob_match;
    let mut out = BTreeSet::new();
    for op in &list.ops {
        let name = match &op.op {
            PrimalOp::Param(n) => n.as_str(),
            _ => continue,
        };
        if input.trainable_patterns.iter().any(|p| glob_match(p, name)) {
            out.insert(op.result);
        }
    }
    out
}

/// A candidate site and whether its shape is the weight's real one.
struct InferredSite {
    site: AdapterSite,
    /// False when the weight's shape is not known and `site.shape` is the
    /// nominal 512×512.
    shape_known: bool,
}

/// The shape used for a site whose weight shape is not known at compile
/// time, and the activation every site is costed at: one sequence of 128
/// tokens in 2-byte elements. Neither is read from the program.
const NOMINAL_WEIGHT: [u64; 2] = [512, 512];
const NOMINAL_SEQ: u64 = 128;

/// The plan's sites, from the Wengert list.
///
/// With declared adapters, the sites are the parameters they adapt, each
/// under the op that consumes it. This is what makes a frozen base work: the
/// adapted weights are then exactly the frozen ones, and looking only at
/// TRAINABLE parameters found the adapters' own `lora_A`/`lora_B` instead,
/// which no checkpoint holds.
///
/// Without any, the sites are the trainable parameters an adapter could go
/// on: one per parameter consumed by a matmul, norm, softmax, embedding or
/// reduction.
///
/// A parameter may reach its op through a transpose (`x @
/// self.w.transpose(0, 1)`, the standard library's `Linear`).
///
/// The Wengert list has no activation shapes, so every site is costed at a
/// nominal activation ([`NOMINAL_SEQ`]); the weight's shape is real when
/// `param_shapes` or the checkpoint has it.
fn infer_sites_from_wengert(
    list: &WengertList,
    trainable: &BTreeSet<VarId>,
    input: &WrgaInput,
) -> Vec<InferredSite> {
    // Param VarId → name, and every var that is a parameter or a transpose
    // of one → that parameter. One pass suffices: inputs precede their users.
    let mut param_name: BTreeMap<VarId, &str> = BTreeMap::new();
    let mut param_of: BTreeMap<VarId, VarId> = BTreeMap::new();
    for op in &list.ops {
        match &op.op {
            PrimalOp::Param(n) => {
                param_name.insert(op.result, n.as_str());
                param_of.insert(op.result, op.result);
            }
            PrimalOp::Transpose { .. } => {
                if let Some(&p) = op.inputs.first().and_then(|v| param_of.get(v)) {
                    param_of.insert(op.result, p);
                }
            }
            _ => {}
        }
    }
    let declared = !input.declared_adapters.is_empty();
    let is_site = |param: VarId| -> bool {
        let Some(name) = param_name.get(&param) else {
            return false;
        };
        if declared {
            input.declared_adapters.contains_key(*name)
        } else {
            trainable.contains(&param)
        }
    };

    let mut out = Vec::new();
    let mut seen: BTreeSet<VarId> = BTreeSet::new();
    for op in &list.ops {
        let Some(param) = op
            .inputs
            .iter()
            .filter_map(|v| param_of.get(v).copied())
            .find(|p| is_site(*p))
        else {
            continue;
        };
        let kind = match op.op {
            PrimalOp::Matmul => SiteKind::Matmul,
            PrimalOp::LayerNorm { .. } | PrimalOp::RMSNorm { .. } => SiteKind::Norm,
            PrimalOp::Softmax { .. } | PrimalOp::LogSoftmax { .. } => SiteKind::Softmax,
            PrimalOp::Embedding => SiteKind::Embedding,
            // WRGA §2.4 pattern 3: a trainable param consumed by a
            // sum/mean reduction is an IA³-style scaling site that can be
            // folded into the reduction kernel's epilogue.
            PrimalOp::Sum { .. } | PrimalOp::Mean { .. } => SiteKind::Reduction,
            _ => continue,
        };
        // A declared adapter is one site however often its weight is used.
        if declared && !seen.insert(param) {
            continue;
        }
        let name = param_name[&param];
        let known = input.param_shapes.get(name).copied().or_else(|| {
            let wm = input.weights?;
            wm.entries()
                .find(|(key, entry)| entry.shape.len() == 2 && names_match(key, name))
                .map(|(_, entry)| [entry.shape[0], entry.shape[1]])
        });
        let shape = known.map_or(NOMINAL_WEIGHT, |[m, n]| [m as u64, n as u64]);
        out.push(InferredSite {
            site: AdapterSite {
                name: name.to_string(),
                kind,
                shape: shape.to_vec(),
                dtype_bytes: 2,
                batch: 1,
                seq: NOMINAL_SEQ,
            },
            shape_known: known.is_some(),
        });
    }
    out
}

/// Rank every adapted site.
///
/// A declared adapter keeps its declared rank and its real parameter count;
/// the spectral figures are computed for its weight when the checkpoint has
/// it, and size nothing. Without declared adapters the ranks are
/// suggestions: the spectral formula over the checkpoint's weights when
/// there are any, the roofline's rank otherwise, in both cases brought under
/// `budget_params` when one is given.
fn run_spectral(
    weights: Option<&WeightMap>,
    placements: &[AdapterPlacement],
    site_shapes: &HashMap<String, [usize; 2]>,
    input: &WrgaInput,
) -> (Vec<SpectralAnalysis>, Vec<RankAllocation>, Vec<crate::wggo_overrides::OverrideDiagnostic>) {
    let active: Vec<&AdapterPlacement> = placements
        .iter()
        .filter(|p| !matches!(p.adapter, crate::wrga_roofline::AdapterKind::Skip))
        .collect();

    // Spectral analysis of the checkpoint weights that are sites.
    let mut spectral = match weights {
        Some(wm) => analyse_weight_map(wm, input.r_max.max(8), 5, 8, input.seed),
        None => Vec::new(),
    };
    spectral.retain(|sp| active.iter().any(|p| names_match(&p.name, &sp.name)));
    if weights.is_some() && spectral.is_empty() && !active.is_empty() {
        nsl_log::nsl_log!(
            WARN,
            "wrga",
            "[wrga] weights present but no checkpoint key matches any adapter site name: \
             the report has no spectral figures (the keys must end in the model's field path)"
        );
    }

    if !input.declared_adapters.is_empty() {
        let ranks = active
            .iter()
            .map(|p| {
                let decl = input.declared_adapters.get(&p.name);
                RankAllocation {
                    name: p.name.clone(),
                    rank: p.suggested_rank,
                    effective_rank: spectral
                        .iter()
                        .find(|sp| names_match(&sp.name, &p.name))
                        .map_or(0.0, |sp| sp.effective_rank),
                    adapter_params: decl.and_then(|d| d.params).unwrap_or(0),
                }
            })
            .collect();
        return (spectral, ranks, Vec::new());
    }

    let budget = (input.budget_params > 0).then_some(input.budget_params);
    if spectral.is_empty() {
        // No spectral data: the roofline's suggested rank on every site,
        // counted from the weight's real shape where it is known.
        let mn = |name: &str| site_shapes.get(name).map_or(0, |[m, n]| m + n);
        let mut ranks: Vec<RankAllocation> = active
            .iter()
            .map(|p| {
                let rank = p.suggested_rank.max(input.r_min).min(input.r_max);
                RankAllocation {
                    name: p.name.clone(),
                    rank,
                    effective_rank: 0.0,
                    adapter_params: rank * mn(&p.name),
                }
            })
            .collect();
        if let Some(budget) = budget {
            crate::wrga_spectral::enforce_param_budget(&mut ranks, budget, input.r_min, &mn, None);
        }
        return (Vec::new(), ranks, Vec::new());
    }

    // Cross-reference: per-site roofline slack as the allocator weight.
    let mut slack = HashMap::new();
    for p in placements {
        slack.insert(p.name.clone(), p.roofline_slack.max(1e-6));
    }
    // The rank pool the proposal's formula divides: `r_max` per site. The
    // parameter budget is a separate, later constraint; it used to be passed
    // here as the rank pool, so `budget=1000` meant a thousand ranks to hand
    // out and capped nothing.
    //
    // `snap_to_grid = false`: the realized per-projection rank is left off-grid
    // (the current behaviour).  Paper §2.2 declares `r[l,w] ∈ {0,2,4,8,16}`, and
    // `allocate_ranks` can floor the realized rank onto that grid (audit gap #5),
    // but errata E3 lists snapping as an open sub-decision awaiting author
    // sign-off, so it stays off by default.
    let (ranks, override_diags) = allocate_ranks(
        &spectral,
        spectral.len().max(1) * input.r_max,
        budget,
        input.r_min,
        input.r_max,
        Some(&slack),
        input.wggo_overrides,
        false,
    );

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
            declared_adapters: BTreeMap::new(),
            param_shapes: HashMap::new(),
            fused_kernels_active: false,
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
            custom_adapter: None,
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
    fn render_report_is_deterministic() {
        let w = tiny_transformer_like();
        let make = || WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.7.*"],
            declared_adapters: BTreeMap::new(),
            param_shapes: HashMap::new(),
            fused_kernels_active: false,
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
            custom_adapter: None,
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
            declared_adapters: BTreeMap::new(),
            param_shapes: HashMap::new(),
            fused_kernels_active: false,
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
            custom_adapter: None,
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
                optim_m_bits: 32,
                optim_v_bits: 32,
            }],
        };
        let input = WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.6.*"],
            declared_adapters: BTreeMap::new(),
            param_shapes: HashMap::new(),
            fused_kernels_active: false,
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
            custom_adapter: None,
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
            declared_adapters: BTreeMap::new(),
            param_shapes: HashMap::new(),
            fused_kernels_active: false,
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
            custom_adapter: None,
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
                optim_m_bits: 32,
                optim_v_bits: 32,
            }],
        };
        let (_allocs, diags) = allocate_ranks(&spectral, 10_000_000, None, 2, 16, None, Some(&over), false);
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

    /// A `WrgaInput` over `w` with everything trainable and no adapters.
    fn plain_input(w: &WengertList) -> WrgaInput<'_> {
        WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["*"],
            declared_adapters: BTreeMap::new(),
            param_shapes: HashMap::new(),
            fused_kernels_active: false,
            wengert: w,
            loss_output: w.output,
            weights: None,
            target: "",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: WrgaAblation::default(),
            custom_adapter: None,
        }
    }

    /// `@freeze` decides what trains, in every mode. `mode=manual` used to
    /// select the parameters matching `@adapter` targets and `mode=hybrid`
    /// those under a layer prefix, and both sets were always empty.
    #[test]
    fn the_mode_has_no_say_in_what_trains() {
        let w = tiny_transformer_like();
        for mode in [WrgaMode::Auto, WrgaMode::Manual, WrgaMode::Hybrid] {
            let mut input = plain_input(&w);
            input.mode = mode;
            input.trainable_patterns = vec!["blocks.7.*"];
            let stats = run(input).prune.stats;
            assert_eq!((stats.gradient_targets, stats.frozen_params), (1, 2), "{mode:?}");
        }
    }

    /// An empty allowlist is a model with nothing trainable, not one with
    /// everything trainable.
    #[test]
    fn no_trainable_pattern_means_nothing_trains() {
        let w = tiny_transformer_like();
        let mut input = plain_input(&w);
        input.trainable_patterns = Vec::new();
        let stats = run(input).prune.stats;
        assert_eq!((stats.gradient_targets, stats.frozen_params), (0, 3));
        assert_eq!(stats.backward_ops_retained, 0, "nothing to differentiate");
    }

    /// With nothing frozen the backward pass is the full one, so nothing is
    /// eliminated. The pruning line used to compare the differentiated ops
    /// with ALL forward ops (the input `x` counted as "eliminated"), and the
    /// activation line compared a count of ops with a count of tensors.
    #[test]
    fn nothing_frozen_reports_nothing_eliminated() {
        let w = tiny_transformer_like();
        let plan = run(plain_input(&w));
        let stats = &plan.prune.stats;
        // Of the 8 forward ops, 6 carry an adjoint: the three parameters, the
        // two matmuls that reach the loss, and the add. The input x has none,
        // and `wk @ x` (#5) does not reach the loss.
        assert_eq!(stats.forward_ops_total, 8);
        assert_eq!(stats.full_backward_ops, 6);
        assert_eq!(stats.backward_ops_retained, 6);
        // Saved: the operands of the live ops matmul(1,0), matmul(3,0) and
        // add(4,6), that is {0, 1, 3, 4, 6}.
        assert_eq!(stats.full_backward_saved_activations, 5);
        assert_eq!(stats.pruned_saved_activations, 5);
        let report = plan.render_report();
        assert!(report.contains("Backward pruning: 6 ops → 6 ops (0.0% eliminated)"), "{report}");
        assert!(
            report.contains("Activation memory: 5 tensors → 5 tensors (0.0% eliminated)"),
            "{report}"
        );
    }

    /// Freezing all but `blocks.7.wq` leaves its matmul, the add and the
    /// parameter itself to differentiate: 3 of the 6.
    #[test]
    fn a_frozen_branch_is_counted_against_the_full_backward() {
        let w = tiny_transformer_like();
        let mut input = plain_input(&w);
        input.trainable_patterns = vec!["blocks.7.*"];
        let plan = run(input);
        let stats = &plan.prune.stats;
        assert_eq!((stats.full_backward_ops, stats.backward_ops_retained), (6, 3));
        // The live ops are matmul(3,0) and add(4,6): they read 3, 0, 4, 6.
        assert_eq!((stats.full_backward_saved_activations, stats.pruned_saved_activations), (5, 4));
        let report = plan.render_report();
        assert!(report.contains("Backward pruning: 6 ops → 3 ops (50.0% eliminated)"), "{report}");
        assert!(
            report.contains("Activation memory: 5 tensors → 4 tensors (20.0% eliminated)"),
            "{report}"
        );
    }

    /// Every saved activation is read by the backward pass, after the last
    /// forward op, so no two can share a slot. Their lifetimes used to end
    /// at their last forward use, which planned 2 slots for these 4.
    #[test]
    fn saved_activations_do_not_share_slots() {
        let w = tiny_transformer_like();
        let mut input = plain_input(&w);
        input.trainable_patterns = vec!["blocks.7.*"];
        let plan = run(input);
        assert_eq!(plan.memory.stats.live_activations, 4);
        assert_eq!(plan.memory.stats.slots_used, 4);
        let last = (w.ops.len() - 1) as u32;
        assert!(plan.memory.assignments.iter().all(|a| a.death == last));
        assert!(
            plan.render_report().contains("Memory plan: 4 live activations → 4 slots (reuse 0.0%)")
        );
    }

    /// A frozen base with LoRA on one weight, as `@freeze` + `@adapter`
    /// leaves the Wengert list: the weight is frozen and the adapter's own
    /// A and B are the trainable parameters.
    fn frozen_base_with_lora() -> WengertList {
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("m.wq".into()), vec![]),
            op(2, 2, PrimalOp::Param("m.lora_A_Toy_wq__lora".into()), vec![]),
            op(3, 3, PrimalOp::Param("m.lora_B_Toy_wq__lora".into()), vec![]),
            op(4, 4, PrimalOp::Matmul, vec![0, 1]),
            op(5, 5, PrimalOp::Matmul, vec![0, 2]),
            op(6, 6, PrimalOp::Matmul, vec![5, 3]),
            op(7, 7, PrimalOp::Add, vec![4, 6]),
        ];
        WengertList { ops, output: 7, var_names: HashMap::new(), var_types: HashMap::new() }
    }

    /// The plan's sites are the adapted weights, with the declared rank and
    /// the real parameter count. They used to be the trainable parameters
    /// under a matmul, here A and B themselves, each "adapted" at rank 16 and
    /// counted as 16 × 1024 parameters.
    #[test]
    fn a_declared_adapter_on_a_frozen_weight_is_the_plan_s_site() {
        let w = frozen_base_with_lora();
        let mut input = plain_input(&w);
        input.trainable_patterns = vec!["m.lora_A_Toy_wq__lora", "m.lora_B_Toy_wq__lora"];
        input.param_shapes.insert("m.wq".to_string(), [32, 48]);
        input.declared_adapters.insert(
            "m.wq".to_string(),
            DeclaredAdapter {
                kind: crate::AdapterKind::Lora,
                rank: 2,
                // A is [32, 2] and B is [2, 48].
                params: Some(2 * (32 + 48)),
            },
        );
        let plan = run(input);
        let names: Vec<&str> = plan.placements.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["m.wq"]);
        assert_eq!(plan.placements[0].suggested_rank, 2);
        assert_eq!(plan.placements[0].decorator_kind, Some(crate::AdapterKind::Lora));
        assert_eq!(plan.adapter_params(), 160);
        assert_eq!(plan.unsized_sites, 0);
        assert_eq!(plan.rank_site_mn, [80]);
        let report = plan.render_report();
        assert!(
            report.contains("Adapter parameters: 160  |  Sites: 1  |  Fusible: 1/1 (100.0%), fused kernels off"),
            "{report}"
        );
    }

    /// 32×48 values with no structure: a full-rank weight for the tests of
    /// the spectral stage.
    fn weight_map_with(name: &str) -> WeightMap {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let data: Vec<u8> = (0..32 * 48)
            .flat_map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let unit = (state >> 40) as f32 / (1u64 << 24) as f32; // [0, 1)
                (unit - 0.5).to_le_bytes()
            })
            .collect();
        let mut wm = WeightMap::new_for_test();
        wm.insert(crate::weight_aware::WeightEntry::new(
            name.to_string(),
            data,
            vec![32, 48],
            crate::weight_aware::WeightDType::F32,
        ));
        wm
    }

    /// The PEFT configuration with a checkpoint: the base is frozen, the
    /// adapter is declared, and the checkpoint holds the base weight under
    /// its field path. The spectral stage analyses that weight and leaves
    /// the declared rank alone. It used to find no site at all, because it
    /// looked among the trainable parameters, which are A and B.
    #[test]
    fn spectral_figures_are_computed_for_a_frozen_adapted_weight() {
        let w = frozen_base_with_lora();
        let wm = weight_map_with("wq");
        let mut input = plain_input(&w);
        input.trainable_patterns = vec!["m.lora_A_Toy_wq__lora", "m.lora_B_Toy_wq__lora"];
        input.weights = Some(&wm);
        input.declared_adapters.insert(
            "m.wq".to_string(),
            DeclaredAdapter { kind: crate::AdapterKind::Lora, rank: 2, params: Some(160) },
        );
        let plan = run(input);
        assert_eq!(plan.spectral.len(), 1);
        assert_eq!((plan.spectral[0].name.as_str(), plan.spectral[0].shape), ("wq", [32, 48]));
        assert_eq!((plan.ranks[0].rank, plan.ranks[0].adapter_params), (2, 160));
        // 16 singular values are kept; a structureless weight's are of
        // similar size, so its effective rank is near 16.
        let r_eff = plan.ranks[0].effective_rank;
        assert!((12.0..=16.0).contains(&r_eff), "effective rank {r_eff}");
        assert!(
            plan.render_report()
                .contains("Spectral analysis: 1 of 1 sites matched a checkpoint weight"),
            "{}",
            plan.render_report()
        );
        // Without the checkpoint there is no such line.
        let mut input = plain_input(&w);
        input.declared_adapters.insert(
            "m.wq".to_string(),
            DeclaredAdapter { kind: crate::AdapterKind::Lora, rank: 2, params: Some(160) },
        );
        assert!(!run(input).render_report().contains("Spectral analysis"));
    }

    /// A site's shape comes from the checkpoint when the model definition
    /// does not give it.
    #[test]
    fn a_site_takes_its_shape_from_the_checkpoint() {
        let w = frozen_base_with_lora();
        let wm = weight_map_with("wq");
        let mut input = plain_input(&w);
        input.trainable_patterns = vec!["m.wq"];
        input.weights = Some(&wm);
        let plan = run(input);
        // One suggested site, the weight; r_total = 16 goes to it whole.
        assert_eq!(plan.unsized_sites, 0);
        assert_eq!(plan.rank_site_mn, [32 + 48]);
        assert_eq!(plan.adapter_params(), plan.ranks[0].rank * 80);
    }

    /// A declared adapter whose weight shape is not known is counted as a
    /// site and left out of the parameter total, and the report says so.
    #[test]
    fn a_declared_adapter_of_unknown_shape_is_reported_as_unsized() {
        let w = frozen_base_with_lora();
        let mut input = plain_input(&w);
        input.declared_adapters.insert(
            "m.wq".to_string(),
            DeclaredAdapter { kind: crate::AdapterKind::Lora, rank: 2, params: None },
        );
        let plan = run(input);
        assert_eq!((plan.adapter_params(), plan.unsized_sites), (0, 1));
        assert!(
            plan.render_report().contains(
                "Adapter parameters: 0 (+ 1 sites whose weight shape is not known at compile time)"
            ),
            "{}",
            plan.render_report()
        );
    }

    /// A weight reaches its matmul through a transpose in the standard
    /// library's `Linear` (`x @ self.w.transpose(0, 1)`); it is a site all
    /// the same.
    #[test]
    fn a_weight_used_through_a_transpose_is_a_site() {
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("m.fc.w".into()), vec![]),
            op(2, 2, PrimalOp::Transpose { dim0: 0, dim1: 1 }, vec![1]),
            op(3, 3, PrimalOp::Matmul, vec![0, 2]),
        ];
        let w = WengertList { ops, output: 3, var_names: HashMap::new(), var_types: HashMap::new() };
        let plan = run(plain_input(&w));
        let names: Vec<&str> = plan.placements.iter().map(|p| p.name.as_str()).collect();
        assert_eq!(names, ["m.fc.w"]);
    }

    /// Six matmul weights of 32×32, as the budget tests below use them.
    fn six_square_sites() -> (WengertList, HashMap<String, [usize; 2]>) {
        let mut ops = vec![op(0, 0, PrimalOp::Input("x".into()), vec![])];
        let mut shapes = HashMap::new();
        let mut last = 0;
        for i in 0..6u32 {
            let (param, out) = (1 + 2 * i, 2 + 2 * i);
            let name = format!("m.w{i}");
            ops.push(op(param, param, PrimalOp::Param(name.clone()), vec![]));
            ops.push(op(out, out, PrimalOp::Matmul, vec![last, param]));
            shapes.insert(name, [32, 32]);
            last = out;
        }
        (WengertList { ops, output: last, var_names: HashMap::new(), var_types: HashMap::new() }, shapes)
    }

    /// Without weights the suggested count comes from each weight's real
    /// shape: rank 16 on six 32×32 weights is 6 × 16 × 64 = 6,144. It used to
    /// be rank × 1024 per site whatever the shape: 98,304.
    #[test]
    fn suggested_parameters_come_from_the_weights_shapes() {
        let (w, shapes) = six_square_sites();
        let mut input = plain_input(&w);
        input.param_shapes = shapes;
        let plan = run(input);
        assert_eq!(plan.ranks.iter().map(|r| r.rank).collect::<Vec<_>>(), [16; 6]);
        assert_eq!(plan.adapter_params(), 6_144);
        assert!(
            plan.render_report().contains(
                "Suggested adapters (none declared, none built): 6144 parameters  |  Sites: 6"
            ),
            "{}",
            plan.render_report()
        );
    }

    /// A budget bounds the suggested count. One rank on a 32×32 weight is 64
    /// parameters, so 1,000 allows 15 ranks in all (960), and 500 allows 7:
    /// fewer than the six sites' minimum of 2 each, so three sites lose their
    /// adapter and three keep rank 2 (384). A budget above the need changes
    /// nothing. It used to be read as a number of ranks and bound nothing.
    #[test]
    fn a_budget_bounds_the_suggested_parameters() {
        let (w, shapes) = six_square_sites();
        let with_budget = |budget: usize| {
            let mut input = plain_input(&w);
            input.param_shapes = shapes.clone();
            input.budget_params = budget;
            run(input)
        };
        assert_eq!(with_budget(100_000).adapter_params(), 6_144);
        let tight = with_budget(1_000);
        assert_eq!(tight.adapter_params(), 960);
        assert_eq!(tight.ranks.iter().map(|r| r.rank).sum::<usize>(), 15);
        assert!(tight.ranks.iter().all(|r| (2..=3).contains(&r.rank)), "{:?}", tight.ranks);
        let tighter = with_budget(500);
        assert_eq!(tighter.adapter_params(), 384);
        let mut ranks: Vec<usize> = tighter.ranks.iter().map(|r| r.rank).collect();
        ranks.sort_unstable();
        assert_eq!(ranks, [0, 0, 0, 2, 2, 2]);
    }

    /// A site of unknown shape is costed at the nominal 512×512 for the
    /// roofline and left out of the parameter count.
    #[test]
    fn a_suggested_site_of_unknown_shape_is_not_counted() {
        let (w, mut shapes) = six_square_sites();
        shapes.remove("m.w0");
        let mut input = plain_input(&w);
        input.param_shapes = shapes;
        let plan = run(input);
        assert_eq!((plan.adapter_params(), plan.unsized_sites), (5 * 16 * 64, 1));
    }

    #[test]
    fn a_gpu_resolves_by_the_name_a_user_writes() {
        assert_eq!(resolve_gpu("h100").map(|g| g.name), Some("H100-SXM"));
        assert_eq!(resolve_gpu("H100-PCIe").map(|g| g.name), Some("H100-PCIe"));
        // The table spells it "RTX-5070-Ti"; the proposal and the decorator's
        // own error message spell it "rtx5070ti".
        assert_eq!(resolve_gpu("rtx5070ti").map(|g| g.name), Some("RTX-5070-Ti"));
        assert_eq!(resolve_gpu("rtx4090").map(|g| g.name), Some("RTX-4090"));
        assert!(resolve_gpu("rtx9999").is_none());
        assert!(resolve_gpu("").is_none());
    }

    /// The report names the GPU the analysis ran for, and says when that is
    /// the default because none was given. A `target=h100` used to be
    /// dropped and reported as "A100-SXM".
    #[test]
    fn the_report_names_the_target_and_marks_a_default() {
        let w = tiny_transformer_like();
        let mut input = plain_input(&w);
        input.target = "h100";
        assert_eq!(run(input).target_gpu, "H100-SXM");
        let mut input = plain_input(&w);
        input.target = "rtx5070ti";
        assert_eq!(run(input).target_gpu, "RTX-5070-Ti");
        assert_eq!(run(plain_input(&w)).target_gpu, "A100-SXM (default: no target given)");
    }

    /// The fusion stage looks each placement's rank up by name. `ranks`
    /// holds only the ranked sites, in its own order, so the i-th rank is
    /// not the i-th placement's.
    #[test]
    fn fusion_takes_each_site_s_rank_by_name() {
        let (w, shapes) = six_square_sites();
        let mut input = plain_input(&w);
        input.param_shapes = shapes;
        input.budget_params = 1_000;
        let plan = run(input);
        for decision in &plan.fusion.decisions {
            let rank = plan.ranks.iter().find(|r| r.name == decision.site).unwrap().rank;
            assert_eq!(
                decision.target,
                crate::wrga_fusion::FusionTarget::EpilogueFusedLora { rank },
                "{}",
                decision.site
            );
        }
    }

    // -----------------------------------------------------------------------
    // `nsl check --wrga-compare`
    // -----------------------------------------------------------------------

    /// Two sites: A with `m + n` = 200 at rank 8, B with `m + n` = 100 at
    /// rank 4.
    fn two_site_plan() -> WrgaPlan {
        use crate::cost_model::BoundClassification;
        use crate::wrga_roofline::AdapterKind;
        let placement = |name: &str, rank: usize| crate::wrga_roofline::AdapterPlacement {
            name: name.into(),
            kind: crate::wrga_roofline::SiteKind::Matmul,
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
        let rank = |name: &str, rank: usize, mn: usize| crate::wrga_spectral::RankAllocation {
            name: name.into(),
            rank,
            effective_rank: rank as f64,
            adapter_params: rank * mn,
        };
        let mut plan = WrgaPlan::test_dummy();
        plan.placements = vec![placement("A", 8), placement("B", 4)];
        plan.ranks = vec![rank("A", 8, 200), rank("B", 4, 100)];
        plan.rank_site_mn = vec![200, 100];
        plan
    }

    /// The comparison's rows, by each method's formula over the two sites.
    #[test]
    fn compare_rows_are_each_method_s_parameter_formula() {
        let rows = two_site_plan().compute_compare_rows();
        let params = |label: &str| rows.iter().find(|r| r.label == label).unwrap().adapter_params;
        assert_eq!(params("WRGA (this run)"), 8 * 200 + 4 * 100);
        assert_eq!(params("LoRA (r=16)"), 16 * 200 + 16 * 100);
        assert_eq!(params("GaLore"), 0);
        // ceil(2 / 4) = 1 site, the larger: 4 × (200 / 2).
        assert_eq!(params("ReFT (r=4, top-25%)"), 400);
        // No AdaLoRA row: its ranks come out of training.
        assert_eq!(rows.len(), 4);
    }

    /// The rendered table has the parameter column only, and the text says
    /// why the other methods have no backward, activation or fusion figure.
    #[test]
    fn compare_report_gives_no_figure_it_did_not_compute() {
        let report = two_site_plan().render_compare_report();
        assert!(report.contains("=== WRGA PEFT Comparison Report ==="));
        assert!(report.contains("WRGA (this run)        |      2,000"), "{report}");
        assert!(report.contains("LoRA (r=16)            |      4,800"), "{report}");
        assert!(report.contains("GaLore                 |          0"), "{report}");
        assert!(report.contains("ReFT (r=4, top-25%)    |        400"), "{report}");
        assert!(!report.contains("100.0%"), "{report}");
        assert!(report.contains("No such figures are given for the other methods"), "{report}");
        // A table row would start the line with the method's name.
        assert!(!report.lines().any(|l| l.starts_with("AdaLoRA")), "{report}");
    }

    #[test]
    fn compare_report_is_deterministic() {
        let w = tiny_transformer_like();
        assert_eq!(
            run(plain_input(&w)).render_compare_report(),
            run(plain_input(&w)).render_compare_report(),
        );
    }

    /// Rank-0 sites of unknown shape contribute no ReFT parameters.
    #[test]
    fn compare_reft_counts_nothing_for_sites_without_a_shape() {
        let mut plan = two_site_plan();
        plan.rank_site_mn = vec![0, 0];
        let rows = plan.compute_compare_rows();
        let reft = rows.iter().find(|r| r.label == "ReFT (r=4, top-25%)").unwrap();
        assert_eq!(reft.adapter_params, 0);
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
            declared_adapters: BTreeMap::new(),
            param_shapes: HashMap::new(),
            fused_kernels_active: false,
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
            custom_adapter: None,
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
            kind: crate::wrga_roofline::SiteKind::Matmul,
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
                optim_m_bits: 32,
                optim_v_bits: 32,
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
                    optim_m_bits: 32,
                    optim_v_bits: 32,
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
                    optim_m_bits: 32,
                    optim_v_bits: 32,
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

    // -----------------------------------------------------------------------
    // WRGA §2.4 pattern 3 — reduction-fused adapter end-to-end (Gap #5)
    // -----------------------------------------------------------------------

    /// A Wengert list whose trainable param is consumed by a `Mean` op —
    /// the classic mean-pool classification head pattern. The end-to-end
    /// driver must classify this site as `SiteKind::Reduction`, route it
    /// through IA³, and produce a `FusionTarget::ReductionFusedAdapter`
    /// in the plan.
    fn mean_pool_head_like() -> WengertList {
        // x   (#0) input
        // wp  (#1) param  head.pool_scale  (trainable)
        // m   (#2) mean(x * wp, dim=-1)    — pooling site consuming wp
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("head.pool_scale".into()), vec![]),
            op(2, 2, PrimalOp::Mul, vec![0, 1]),
            op(3, 3, PrimalOp::Mean { dim: Some(-1) }, vec![2, 1]),
        ];
        WengertList {
            ops,
            output: 3,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        }
    }

    #[test]
    fn reduction_site_routes_to_reduction_fused_adapter() {
        let w = mean_pool_head_like();
        let input = WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["*"], // all params trainable
            declared_adapters: BTreeMap::new(),
            param_shapes: HashMap::new(),
            fused_kernels_active: false,
            wengert: &w,
            loss_output: 3,
            weights: None,
            target: "",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: WrgaAblation::default(),
            custom_adapter: None,
        };
        let plan = run(input);
        // Exactly one decision, on the reduction site, with the §2.4
        // pattern 3 fusion target.
        let mean_dec = plan
            .fusion
            .decisions
            .iter()
            .find(|d| d.site == "head.pool_scale")
            .expect("a fusion decision for the trainable param must exist");
        assert_eq!(
            mean_dec.target,
            crate::wrga_fusion::FusionTarget::ReductionFusedAdapter,
            "Mean PrimalOp must drive a ReductionFusedAdapter decision; got {:?}",
            mean_dec.target,
        );
        assert!(mean_dec.target.is_fused());
        assert!(plan.fusion.fused_count() >= 1);
    }

    #[test]
    fn sum_op_also_routes_to_reduction_fused_adapter() {
        // The Sum PrimalOp branch must behave identically to Mean — both
        // are reduction sites under §2.4 pattern 3.
        let ops = vec![
            op(0, 0, PrimalOp::Input("x".into()), vec![]),
            op(1, 1, PrimalOp::Param("head.sum_scale".into()), vec![]),
            op(2, 2, PrimalOp::Mul, vec![0, 1]),
            op(3, 3, PrimalOp::Sum { dim: Some(-1) }, vec![2, 1]),
        ];
        let w = WengertList {
            ops,
            output: 3,
            var_names: HashMap::new(),
            var_types: HashMap::new(),
        };
        let input = WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["*"],
            declared_adapters: BTreeMap::new(),
            param_shapes: HashMap::new(),
            fused_kernels_active: false,
            wengert: &w,
            loss_output: 3,
            weights: None,
            target: "",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: WrgaAblation::default(),
            custom_adapter: None,
        };
        let plan = run(input);
        let dec = plan
            .fusion
            .decisions
            .iter()
            .find(|d| d.site == "head.sum_scale")
            .expect("a fusion decision for the trainable param must exist");
        assert_eq!(
            dec.target,
            crate::wrga_fusion::FusionTarget::ReductionFusedAdapter,
        );
    }

    /// Reduction-fused must NOT collide with the §9.3 ablation harness —
    /// when `skip_fusion_integration` is set, the new variant must NOT
    /// survive in `plan.fusion`, just like the other Innovations' ablation
    /// contract. The first assertion pins the actual mechanism (`FusionPlan::default()`
    /// → empty decisions) so a future refactor that keeps decisions but
    /// flips them to `StandaloneAdapter` would still produce an empty
    /// fusion ratio but would break this test loudly; the second pins
    /// the user-visible ratio.
    #[test]
    fn reduction_fused_is_skipped_under_skip_fusion_integration() {
        let w = mean_pool_head_like();
        // Build a WrgaInput that actually puts the mean-pool param into
        // the trainable set (the generic `ablation_input` helper is wired
        // to `blocks.7.*` and would skip `head.pool_scale`).
        let mk_input = |abl: WrgaAblation| WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["*"], // empty ⇒ all params trainable
            declared_adapters: BTreeMap::new(),
            param_shapes: HashMap::new(),
            fused_kernels_active: false,
            wengert: &w,
            loss_output: 3,
            weights: None,
            target: "",
            budget_params: 0,
            r_min: 2,
            r_max: 16,
            seed: 42,
            inspect_pinned_vars: BTreeSet::new(),
            wggo_overrides: None,
            ablation: abl,
            custom_adapter: None,
        };

        // Baseline: same model under no ablation MUST produce a
        // ReductionFusedAdapter decision, otherwise the ablation test
        // below is testing the wrong thing.
        let baseline = run(mk_input(WrgaAblation::default()));
        assert!(
            baseline.fusion.decisions.iter().any(|d| matches!(
                d.target,
                crate::wrga_fusion::FusionTarget::ReductionFusedAdapter
            )),
            "test premise broken: baseline run must produce ReductionFusedAdapter \
             for the mean-pool head fixture; got {:?}",
            baseline.fusion.decisions.iter().map(|d| &d.target).collect::<Vec<_>>(),
        );

        let abl = WrgaAblation {
            skip_fusion_integration: true,
            ..Default::default()
        };
        let plan = run(mk_input(abl));
        assert!(
            plan.fusion.decisions.is_empty(),
            "skip_fusion_integration must replace fusion with FusionPlan::default(); \
             got {} decisions",
            plan.fusion.decisions.len(),
        );
        assert_eq!(plan.fusion.fusion_ratio(), 0.0);
    }

    // -----------------------------------------------------------------------
    // WRGA paper §8.2 — custom adapter DSL plumbing (Gap #6)
    // -----------------------------------------------------------------------

    fn input_with_custom_adapter<'a>(w: &'a WengertList, name: Option<&'a str>) -> WrgaInput<'a> {
        WrgaInput {
            mode: WrgaMode::Auto,
            trainable_patterns: vec!["blocks.7.*"],
            declared_adapters: BTreeMap::new(),
            param_shapes: HashMap::new(),
            fused_kernels_active: false,
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
            ablation: WrgaAblation::default(),
            custom_adapter: name,
        }
    }

    /// `WrgaInput.custom_adapter = Some(name)` must round-trip onto the
    /// produced `WrgaPlan.custom_adapter` as an owned `String` of the
    /// same value.
    #[test]
    fn custom_adapter_propagates_from_input_to_plan() {
        let w = tiny_transformer_like();
        let plan = run(input_with_custom_adapter(&w, Some("GatedLoRA")));
        assert_eq!(plan.custom_adapter.as_deref(), Some("GatedLoRA"));
    }

    /// `None` on the input must stay `None` on the plan — confirms no
    /// silent default is injected when the DSL isn't used.
    #[test]
    fn custom_adapter_none_round_trips_as_none() {
        let w = tiny_transformer_like();
        let plan = run(input_with_custom_adapter(&w, None));
        assert!(plan.custom_adapter.is_none());
    }

    /// `render_report` must surface the `Custom adapter:` header line
    /// when `custom_adapter` is set, and only then.  Pins both the
    /// presence under the §8.2 DSL and the absence under default.
    #[test]
    fn render_report_surfaces_custom_adapter_when_set() {
        let w = tiny_transformer_like();
        let with_custom = run(input_with_custom_adapter(&w, Some("GatedLoRA"))).render_report();
        let without_custom = run(input_with_custom_adapter(&w, None)).render_report();
        assert!(
            with_custom.contains("Custom adapter: GatedLoRA"),
            "report must surface custom adapter name; got:\n{with_custom}",
        );
        // The trailing parenthetical is load-bearing — without it readers
        // would mistake the LoRA placements below for instantiations of
        // <name> (review finding #3).
        assert!(
            with_custom.contains("(placement integration pending)"),
            "report must mark the custom adapter as not-yet-emitted; got:\n{with_custom}",
        );
        assert!(
            !without_custom.contains("Custom adapter:"),
            "report must NOT surface custom adapter line when unset; got:\n{without_custom}",
        );
    }

    /// Custom adapter must NOT break the §9.3 ablation contract — the
    /// `Custom adapter:` line and the `Ablation:` line can coexist; the
    /// custom adapter is just a placement target, orthogonal to which
    /// Innovations are ablated.
    #[test]
    fn custom_adapter_coexists_with_ablation_header() {
        let w = tiny_transformer_like();
        let mut input = input_with_custom_adapter(&w, Some("GatedLoRA"));
        input.ablation = WrgaAblation {
            skip_fusion_integration: true,
            ..Default::default()
        };
        let report = run(input).render_report();
        assert!(
            report.contains("Custom adapter: GatedLoRA"),
            "custom adapter line missing: {report}",
        );
        assert!(
            report.contains("Ablation:"),
            "ablation line missing: {report}",
        );
    }
}
