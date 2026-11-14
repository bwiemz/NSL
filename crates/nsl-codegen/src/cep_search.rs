//! CEP — pruning search and architecture search.
//!
//! [`prune_greedy`] and [`joint_prune_search`] share one loop. Each step
//! removes whichever component costs the least importance per parameter
//! saved, among:
//!
//!   * one key/value group of attention heads from a layer (grouped-query
//!     attention shares a key/value head between `n_heads / n_kv_heads` query
//!     heads, so a group is the smallest unit that can go);
//!   * one step of FFN width from a layer;
//!   * a whole layer (joint search only).
//!
//! The loop runs until the sparsity target AND the memory and latency limits
//! are all met, and stops early only when nothing more can be removed. The
//! limits are upper bounds and pruning only lowers them, so they are targets
//! to reach, never a reason to reject a step. A search that stops short says
//! so: [`SearchOutcome::sparsity_met`] and [`SearchOutcome::constraints_met`].
//!
//! [`architecture_search`] enumerates the search axes and ranks candidates.

use serde::Serialize;

use crate::cep_importance::{ImportanceTable, position_factor};
use crate::cep_oracle::{CompilationProfile, ModelSpec, evaluate};
use crate::cep_rewrite::{LayerDelta, PruneDelta, apply_delta};
use crate::gpu_specs::GpuSpec;

/// Limits and targets for the search.
#[derive(Debug, Clone)]
pub struct Constraints {
    /// Upper bound on peak memory (bytes).
    pub peak_memory_bytes: u64,
    /// Upper bound on the estimated latency of one forward pass (μs).
    pub latency_us: f64,
    /// Fraction of the model's parameters to remove (0..1).
    pub target_sparsity: f64,
    /// Layers the search must leave untouched.
    pub preserve_layers: Vec<u32>,
    /// Prune every layer in step, so all of them end at one shape. The
    /// emitted source can express nothing else: it has one block
    /// constructor for all layers.
    pub uniform: bool,
}

impl Default for Constraints {
    fn default() -> Self {
        Self {
            peak_memory_bytes: u64::MAX,
            latency_us: f64::MAX,
            target_sparsity: 0.3,
            preserve_layers: Vec::new(),
            uniform: false,
        }
    }
}

/// Granularity options (§4 of the paper).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum Granularity {
    /// Prune attention heads only.
    Head,
    /// Prune FFN width only.
    Ffn,
    /// Prune both (default).
    HeadAndFfn,
}

/// Ranking objective used for architecture search.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum NasObjective {
    /// `params × roofline_util / latency`, the proposal's metric. Latency
    /// tracks FLOPs and FLOPs track parameters, so it is nearly flat across
    /// candidates and has no task-quality term; see [`Self::caveat`].
    ParamEfficiency,
    /// Minimise latency only.
    MinLatency,
    /// Minimise peak memory only.
    MinMemory,
    /// Minimise parameter count.
    MinParams,
}

impl NasObjective {
    fn score(&self, p: &CompilationProfile, spec: &ModelSpec) -> f64 {
        match self {
            NasObjective::ParamEfficiency => {
                let params = spec.param_count() as f64;
                let util = p.roofline_utilization.max(1e-3);
                let lat = p.estimated_latency_us.max(1e-3);
                params * util / lat
            }
            NasObjective::MinLatency => -p.estimated_latency_us,
            NasObjective::MinMemory => -(p.peak_memory_bytes as f64),
            NasObjective::MinParams => -(spec.param_count() as f64),
        }
    }

    /// What a reader of the ranking must know about this objective, if
    /// anything.
    pub fn caveat(&self) -> Option<&'static str> {
        match self {
            NasObjective::ParamEfficiency => Some(
                "param_efficiency (params x utilization / latency) has no task-quality term and \
                 is nearly constant across architectures; what variation there is favours small \
                 block stacks. Treat the order as a tie-break, not as a recommendation.",
            ),
            _ => None,
        }
    }
}

/// Single evaluated candidate with its profile and objective score.
#[derive(Debug, Clone)]
pub struct EvaluatedCandidate {
    pub spec: ModelSpec,
    pub profile: CompilationProfile,
    pub score: f64,
    /// Prune: the sparsity target and both limits are met. Search: both
    /// limits are met.
    pub feasible: bool,
    /// Architecture search only: other candidates whose every metric equals
    /// this one's (they differ in an axis the estimate does not see).
    pub also: Vec<String>,
}

/// Output of either search entry point.
#[derive(Debug, Clone, Default)]
pub struct SearchOutcome {
    /// Baseline profile (original model, pre-prune).  `None` for NAS.
    pub baseline: Option<CompilationProfile>,
    /// Chosen spec + its profile.
    pub chosen: Option<EvaluatedCandidate>,
    /// All evaluated candidates, ranked: feasible first, then by score.
    pub ranked_candidates: Vec<EvaluatedCandidate>,
    /// The steps taken, in order (prune modes). Every entry is a step that
    /// was applied.
    pub prune_log: Vec<PruneStep>,
    /// Prune modes: what was removed, relative to the baseline spec. This is
    /// what the weight slicer and the source rewriter consume.
    pub delta: Option<PruneDelta>,
    /// Prune modes: the requested fraction of parameters was removed.
    pub sparsity_met: bool,
    /// Prune modes: the memory and latency limits hold for the chosen spec.
    pub constraints_met: bool,
    /// Prune modes: the fraction of parameters removed.
    pub achieved_sparsity: f64,
    /// Prune modes: the fraction of parameters that was asked for.
    pub target_sparsity: f64,
    /// Number of cost estimates computed.
    pub candidates_evaluated: u32,
    /// Search mode: the candidates the axes span that pass the validity
    /// filters (`SearchAxes::enumerate`), before any is estimated or merged
    /// with an identical one. Prune modes: equal to `candidates_evaluated`.
    pub candidates_enumerated: u32,
    /// Wall-clock duration of the search (microseconds).
    pub wall_clock_us: u64,
}

/// One applied pruning step.
#[derive(Debug, Clone, Serialize)]
pub struct PruneStep {
    /// `"head"` (one key/value group), `"ffn"` or `"layer"`.
    pub kind: String,
    /// Baseline layer index.
    pub layer: u32,
    /// The query heads removed (a whole key/value group), for `"head"`.
    pub heads: Vec<u32>,
    /// The layer's FFN width after the step, for `"ffn"`.
    pub new_d_ff: Option<u32>,
    pub reason: String,
    /// Model totals after the step.
    pub new_param_bytes: u64,
    pub new_latency_us: f64,
}

/// FFN width removed by one step: an eighth of the layer's original width,
/// in multiples of 64 where the width allows it.
pub fn ffn_step(baseline_d_ff: u32) -> u32 {
    let eighth = baseline_d_ff / 8;
    if eighth >= 64 {
        (eighth / 64) * 64
    } else {
        eighth.max(1)
    }
}

/// The narrowest FFN the search will leave: three eighths of the original
/// width, rounded down to a multiple of 64 where it is that wide, and never
/// zero.
pub fn ffn_floor(baseline_d_ff: u32) -> u32 {
    let three_eighths = (baseline_d_ff as u64 * 3).div_ceil(8) as u32;
    if three_eighths >= 64 {
        (three_eighths / 64) * 64
    } else {
        three_eighths.max(1)
    }
}

fn satisfies_constraints(profile: &CompilationProfile, c: &Constraints) -> bool {
    profile.peak_memory_bytes <= c.peak_memory_bytes && profile.estimated_latency_us <= c.latency_us
}

fn sparsity_of(baseline_params: u64, current_params: u64) -> f64 {
    if baseline_params == 0 {
        return 0.0;
    }
    1.0 - (current_params as f64 / baseline_params as f64)
}

/// Importance in comparable units. Head scores and FFN neuron magnitudes are
/// products of different numbers of norms, so each is divided by the mean of
/// its own per-layer total: an average attention sublayer and an average FFN
/// sublayer are worth 1.0 each, and a layer with small weights is worth less
/// than one with large weights. When a kind has no usable scores (no weights,
/// or all-zero weights) every component of a layer is worth the same and the
/// layers differ by the position factor alone.
struct Costs {
    /// `[layer][group]`.
    group: Vec<Vec<f64>>,
    /// `[layer]`: running total of per-neuron cost in the order neurons are
    /// removed (least important first), starting at 0.
    ffn_prefix: Vec<Vec<f64>>,
}

impl Costs {
    fn new(spec: &ModelSpec, importance: &ImportanceTable) -> Self {
        let n = spec.n_layers as usize;
        // --- attention groups ---
        let mut group: Vec<Vec<f64>> = (0..n)
            .map(|l| vec![0.0; spec.n_kv_heads[l].max(1) as usize])
            .collect();
        for h in &importance.heads {
            let l = h.layer as usize;
            if l >= n {
                continue;
            }
            let per_group = (spec.n_heads[l] / spec.n_kv_heads[l].max(1)).max(1);
            if let Some(slot) = group[l].get_mut((h.head / per_group) as usize) {
                *slot += h.score;
            }
        }
        normalise(&mut group, spec);

        // --- FFN neurons ---
        let mut neuron: Vec<Vec<f64>> = (0..n).map(|l| vec![0.0; spec.d_ff[l] as usize]).collect();
        for f in &importance.ffns {
            let l = f.layer as usize;
            if l < n && f.neuron_magnitudes.len() == neuron[l].len() {
                let scale = f.roofline_slack * f.position_factor;
                for (slot, m) in neuron[l].iter_mut().zip(&f.neuron_magnitudes) {
                    *slot = m * scale;
                }
            }
        }
        normalise(&mut neuron, spec);
        let mut ffn_prefix = Vec::with_capacity(n);
        for costs in &neuron {
            // Least important first; among equals the higher index goes
            // first, which is the mirror image of `ffn_survivors` (it keeps
            // the lower index), so the slicer keeps exactly the neurons this
            // total was not charged for.
            let mut order: Vec<usize> = (0..costs.len()).collect();
            order.sort_by(|&a, &b| costs[a].total_cmp(&costs[b]).then(b.cmp(&a)));
            let mut prefix = Vec::with_capacity(order.len() + 1);
            let mut total = 0.0;
            prefix.push(0.0);
            for j in order {
                total += costs[j];
                prefix.push(total);
            }
            ffn_prefix.push(prefix);
        }
        Self { group, ffn_prefix }
    }
}

/// Divide every entry by the mean per-layer total, falling back to a uniform
/// split weighted by the position factor when the totals are all zero.
fn normalise(per_layer: &mut [Vec<f64>], spec: &ModelSpec) {
    let n = per_layer.len().max(1) as f64;
    let mean_total = per_layer.iter().map(|v| v.iter().sum::<f64>()).sum::<f64>() / n;
    if mean_total > 0.0 && mean_total.is_finite() {
        for v in per_layer.iter_mut() {
            for x in v.iter_mut() {
                *x /= mean_total;
            }
        }
        return;
    }
    let pos: Vec<f64> = (0..per_layer.len())
        .map(|l| position_factor(l as u32, spec.n_layers))
        .collect();
    let mean_pos = pos.iter().sum::<f64>() / n;
    for (l, v) in per_layer.iter_mut().enumerate() {
        let each = pos[l] / mean_pos / v.len().max(1) as f64;
        for x in v.iter_mut() {
            *x = each;
        }
    }
}

/// What has been removed from each baseline layer so far.
#[derive(Clone)]
struct LayerState {
    dropped: bool,
    /// Key/value groups removed, in removal order.
    groups_removed: Vec<u32>,
    /// FFN neurons removed so far.
    neurons_removed: u32,
}

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum ActionKind {
    Head,
    Ffn,
    Layer,
}

/// One candidate step: the same kind of removal on one or more layers.
struct Action {
    kind: ActionKind,
    /// `(layer, group to remove)` for heads; `(layer, neurons to remove)` for
    /// FFN; `(layer, 0)` for a layer drop.
    parts: Vec<(u32, u32)>,
    cost: f64,
}

struct PruneState<'a> {
    base: &'a ModelSpec,
    costs: Costs,
    layers: Vec<LayerState>,
}

impl PruneState<'_> {
    fn group_size(&self, l: usize) -> u32 {
        (self.base.n_heads[l] / self.base.n_kv_heads[l].max(1)).max(1)
    }

    fn delta(&self) -> PruneDelta {
        delta_of(self.base, &self.layers)
    }

    /// The least important group still in layer `l`, if one may go (at least
    /// one key/value group always stays).
    fn next_group(&self, l: usize) -> Option<(u32, f64)> {
        let st = &self.layers[l];
        let n_groups = self.costs.group[l].len();
        if st.dropped || st.groups_removed.len() + 1 >= n_groups {
            return None;
        }
        (0..n_groups as u32)
            .filter(|g| !st.groups_removed.contains(g))
            .map(|g| (g, self.costs.group[l][g as usize]))
            .min_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)))
    }

    /// The next FFN step for layer `l`: how many neurons, and what they cost.
    fn next_ffn(&self, l: usize) -> Option<(u32, f64)> {
        let st = &self.layers[l];
        if st.dropped {
            return None;
        }
        let base = self.base.d_ff[l];
        let width = base - st.neurons_removed;
        let step = ffn_step(base);
        if width < step || width - step < ffn_floor(base) {
            return None;
        }
        let prefix = &self.costs.ffn_prefix[l];
        let (lo, hi) = (
            st.neurons_removed as usize,
            (st.neurons_removed + step) as usize,
        );
        Some((step, prefix[hi] - prefix[lo]))
    }

    /// Everything layer `l` still holds.
    fn remaining(&self, l: usize) -> f64 {
        let st = &self.layers[l];
        let groups: f64 = self.costs.group[l]
            .iter()
            .enumerate()
            .filter(|(g, _)| !st.groups_removed.contains(&(*g as u32)))
            .map(|(_, c)| *c)
            .sum();
        let prefix = &self.costs.ffn_prefix[l];
        groups + prefix[prefix.len() - 1] - prefix[st.neurons_removed as usize]
    }

    /// How many steps of `kind` layer `l` has already taken.
    fn steps_taken(&self, l: usize, kind: ActionKind) -> u32 {
        let st = &self.layers[l];
        match kind {
            ActionKind::Head => st.groups_removed.len() as u32,
            ActionKind::Ffn => st.neurons_removed / ffn_step(self.base.d_ff[l]),
            ActionKind::Layer => 0,
        }
    }

    fn retained_capacity(&self) -> f64 {
        (0..self.layers.len())
            .filter(|&l| !self.layers[l].dropped)
            .map(|l| self.remaining(l))
            .sum()
    }

    fn apply(&mut self, action: &Action) {
        apply_action(&mut self.layers, action);
    }

    /// Every step that may be taken now.
    fn candidates(
        &self,
        constraints: &Constraints,
        granularity: Granularity,
        allow_layer_drop: bool,
    ) -> Vec<Action> {
        let free: Vec<usize> = (0..self.layers.len())
            .filter(|&l| {
                !self.layers[l].dropped && !constraints.preserve_layers.contains(&(l as u32))
            })
            .collect();
        let mut out = Vec::new();
        let heads = !matches!(granularity, Granularity::Ffn);
        let ffn = !matches!(granularity, Granularity::Head);

        let mut push = |kind: ActionKind, per_layer: Vec<Option<(u32, u32, f64)>>| {
            if constraints.uniform {
                // All free layers move together, or none does.
                if per_layer.is_empty() || per_layer.iter().any(Option::is_none) {
                    return;
                }
                let parts: Vec<(u32, u32, f64)> = per_layer.into_iter().flatten().collect();
                out.push(Action {
                    kind,
                    cost: parts.iter().map(|p| p.2).sum(),
                    parts: parts.iter().map(|p| (p.0, p.1)).collect(),
                });
            } else {
                for (layer, arg, cost) in per_layer.into_iter().flatten() {
                    out.push(Action {
                        kind,
                        parts: vec![(layer, arg)],
                        cost,
                    });
                }
            }
        };
        if heads {
            push(
                ActionKind::Head,
                free.iter()
                    .map(|&l| self.next_group(l).map(|(g, c)| (l as u32, g, c)))
                    .collect(),
            );
        }
        if ffn {
            push(
                ActionKind::Ffn,
                free.iter()
                    .map(|&l| self.next_ffn(l).map(|(n, c)| (l as u32, n, c)))
                    .collect(),
            );
        }
        if allow_layer_drop {
            let surviving = self.layers.iter().filter(|st| !st.dropped).count();
            if surviving > 1 {
                for &l in &free {
                    out.push(Action {
                        kind: ActionKind::Layer,
                        parts: vec![(l as u32, 0)],
                        cost: self.remaining(l),
                    });
                }
            }
        }
        out
    }
}

fn apply_action(layers: &mut [LayerState], action: &Action) {
    for &(layer, arg) in &action.parts {
        let st = &mut layers[layer as usize];
        match action.kind {
            ActionKind::Head => st.groups_removed.push(arg),
            ActionKind::Ffn => st.neurons_removed += arg,
            ActionKind::Layer => st.dropped = true,
        }
    }
}

/// The removals recorded in `layers`, as a delta against `base`.
fn delta_of(base: &ModelSpec, layers: &[LayerState]) -> PruneDelta {
    let per_layer = layers
        .iter()
        .enumerate()
        .map(|(l, st)| {
            if st.dropped {
                return LayerDelta {
                    layer: l as u32,
                    pruned_heads: Vec::new(),
                    new_d_ff: None,
                    drop_layer: true,
                };
            }
            let per_group = (base.n_heads[l] / base.n_kv_heads[l].max(1)).max(1);
            let mut pruned_heads: Vec<u32> = st
                .groups_removed
                .iter()
                .flat_map(|&g| g * per_group..(g + 1) * per_group)
                .collect();
            pruned_heads.sort_unstable();
            let new_d_ff = (st.neurons_removed > 0).then(|| base.d_ff[l] - st.neurons_removed);
            LayerDelta {
                layer: l as u32,
                pruned_heads,
                new_d_ff,
                drop_layer: false,
            }
        })
        .collect();
    PruneDelta { per_layer }
}

/// The shared pruning loop. See the module documentation.
fn prune_search(
    base_spec: &ModelSpec,
    importance: &ImportanceTable,
    gpu: &GpuSpec,
    constraints: &Constraints,
    granularity: Granularity,
    allow_layer_drop: bool,
) -> SearchOutcome {
    let t0 = std::time::Instant::now();
    let baseline_profile = evaluate(base_spec, gpu).expect("base spec must validate");
    let baseline_params = base_spec.param_count();

    let mut state = PruneState {
        base: base_spec,
        costs: Costs::new(base_spec, importance),
        layers: vec![
            LayerState {
                dropped: false,
                groups_removed: Vec::new(),
                neurons_removed: 0
            };
            base_spec.n_layers as usize
        ],
    };
    let mut current_spec = base_spec.clone();
    let mut current_profile = baseline_profile.clone();
    let mut prune_log = Vec::new();
    let mut candidates_evaluated = 1u32;
    let target = constraints.target_sparsity;
    // A hair of slack so a target that is met exactly is not missed by the
    // rounding of the ratio.
    let met = |params: u64| sparsity_of(baseline_params, params) >= target - 1e-12;

    loop {
        let current_params = current_spec.param_count();
        if met(current_params) && satisfies_constraints(&current_profile, constraints) {
            break;
        }
        // Rank by importance lost per parameter saved. Among equals take the
        // larger saving, then the layer that has given up least so far (so
        // layers that score alike are pruned evenly, not one to its floor
        // before the next), then heads before FFN before layers, then the
        // lower layer.
        let mut best: Option<(
            f64,
            (std::cmp::Reverse<u64>, u32, ActionKind, u32),
            Action,
            ModelSpec,
        )> = None;
        for action in state.candidates(constraints, granularity, allow_layer_drop) {
            let mut trial_layers = state.layers.clone();
            apply_action(&mut trial_layers, &action);
            let trial_spec = apply_delta(base_spec, &delta_of(base_spec, &trial_layers));
            let saved = current_params.saturating_sub(trial_spec.param_count());
            if saved == 0 {
                continue;
            }
            let ratio = action.cost / saved as f64;
            let first = action.parts[0].0;
            let key = (
                std::cmp::Reverse(saved),
                state.steps_taken(first as usize, action.kind),
                action.kind,
                first,
            );
            let better = match &best {
                None => true,
                Some((r, k, _, _)) => ratio < *r || (ratio == *r && key < *k),
            };
            if better {
                best = Some((ratio, key, action, trial_spec));
            }
        }
        let Some((_, _, action, trial_spec)) = best else {
            break;
        };
        let Ok(trial_profile) = evaluate(&trial_spec, gpu) else {
            break;
        };
        candidates_evaluated += 1;

        for &(layer, arg) in &action.parts {
            let l = layer as usize;
            let (kind, heads, new_d_ff, reason) = match action.kind {
                ActionKind::Head => {
                    let per_group = state.group_size(l);
                    let heads: Vec<u32> = (arg * per_group..(arg + 1) * per_group).collect();
                    (
                        "head",
                        heads,
                        None,
                        "least important key/value group per parameter saved",
                    )
                }
                ActionKind::Ffn => {
                    let width = base_spec.d_ff[l] - state.layers[l].neurons_removed - arg;
                    (
                        "ffn",
                        Vec::new(),
                        Some(width),
                        "least important neurons per parameter saved",
                    )
                }
                ActionKind::Layer => (
                    "layer",
                    Vec::new(),
                    None,
                    "least important layer per parameter saved",
                ),
            };
            prune_log.push(PruneStep {
                kind: kind.into(),
                layer,
                heads,
                new_d_ff,
                reason: reason.into(),
                new_param_bytes: trial_profile.param_bytes,
                new_latency_us: trial_profile.estimated_latency_us,
            });
        }
        state.apply(&action);
        current_spec = trial_spec;
        current_profile = trial_profile;
    }

    let achieved = sparsity_of(baseline_params, current_spec.param_count());
    let sparsity_met = met(current_spec.param_count());
    let constraints_met = satisfies_constraints(&current_profile, constraints);
    let wall_clock_us = t0.elapsed().as_micros() as u64;
    SearchOutcome {
        baseline: Some(baseline_profile),
        chosen: Some(EvaluatedCandidate {
            spec: current_spec,
            profile: current_profile,
            score: state.retained_capacity(),
            feasible: sparsity_met && constraints_met,
            also: Vec::new(),
        }),
        ranked_candidates: Vec::new(),
        prune_log,
        delta: Some(state.delta()),
        sparsity_met,
        constraints_met,
        achieved_sparsity: achieved,
        target_sparsity: target,
        candidates_evaluated,
        candidates_enumerated: candidates_evaluated,
        wall_clock_us,
    }
}

/// Prune attention heads and FFN width until the sparsity target and the
/// limits are met. The chosen candidate's `score` is the importance the
/// pruned model retains, in units of an average sublayer.
pub fn prune_greedy(
    base_spec: &ModelSpec,
    importance: &ImportanceTable,
    gpu: &GpuSpec,
    constraints: &Constraints,
    granularity: Granularity,
) -> SearchOutcome {
    prune_search(base_spec, importance, gpu, constraints, granularity, false)
}

/// As [`prune_greedy`], with dropping a whole layer as a further kind of
/// step. At least one layer always survives.
pub fn joint_prune_search(
    base_spec: &ModelSpec,
    importance: &ImportanceTable,
    gpu: &GpuSpec,
    constraints: &Constraints,
    granularity: Granularity,
) -> SearchOutcome {
    prune_search(base_spec, importance, gpu, constraints, granularity, true)
}

/// The fields of a profile the ranking can see. Two candidates that agree on
/// all of them are the same candidate as far as the estimate goes.
fn metric_key(c: &EvaluatedCandidate) -> (u64, u64, u64, u64, u64, u64) {
    (
        c.spec.param_count(),
        c.profile.total_flops,
        c.profile.total_hbm_bytes,
        c.profile.peak_memory_bytes,
        c.profile.estimated_latency_us.to_bits(),
        c.score.to_bits(),
    )
}

/// Architecture search: enumerate the cross-product of search axes, estimate
/// each candidate, and rank by the chosen objective (feasible candidates
/// first). Candidates whose metrics are all equal are listed once, with the
/// others named in [`EvaluatedCandidate::also`].
pub fn architecture_search(
    axes: &crate::cep_rewrite::SearchAxes,
    gpu: &GpuSpec,
    constraints: &Constraints,
    objective: NasObjective,
) -> SearchOutcome {
    let t0 = std::time::Instant::now();
    let mut enumerated_total = 0u32;
    let mut evaluated: Vec<EvaluatedCandidate> = Vec::new();
    let mut candidates_evaluated = 0u32;
    for spec in axes.enumerate() {
        enumerated_total += 1;
        let Ok(profile) = evaluate(&spec, gpu) else {
            continue;
        };
        candidates_evaluated += 1;
        let feasible = satisfies_constraints(&profile, constraints);
        let score = objective.score(&profile, &spec);
        let cand = EvaluatedCandidate {
            spec,
            profile,
            score,
            feasible,
            also: Vec::new(),
        };
        let key = metric_key(&cand);
        if let Some(first) = evaluated.iter_mut().find(|c| metric_key(c) == key) {
            let s = &cand.spec;
            let mut label = format!(
                "activation={}, norm={}",
                s.activation.as_str(),
                s.norm.as_str()
            );
            let dims = |m: &ModelSpec| {
                (
                    m.d_model,
                    m.n_layers,
                    m.n_heads.clone(),
                    m.n_kv_heads.clone(),
                    m.d_ff.clone(),
                )
            };
            if dims(s) != dims(&first.spec) {
                label = format!(
                    "d={}, L={}, H={}, KV={}, FFN={}, {label}",
                    s.d_model,
                    s.n_layers,
                    s.n_heads.first().copied().unwrap_or(0),
                    s.n_kv_heads.first().copied().unwrap_or(0),
                    s.d_ff.first().copied().unwrap_or(0)
                );
            }
            first.also.push(label);
        } else {
            evaluated.push(cand);
        }
    }
    // Feasible first, then by score (descending). The sort is stable, so
    // equal scores keep enumeration order.
    evaluated.sort_by(|a, b| {
        b.feasible
            .cmp(&a.feasible)
            .then(b.score.total_cmp(&a.score))
    });
    let wall_clock_us = t0.elapsed().as_micros() as u64;
    let chosen = evaluated.iter().find(|c| c.feasible).cloned();
    SearchOutcome {
        baseline: None,
        chosen,
        ranked_candidates: evaluated,
        candidates_evaluated,
        candidates_enumerated: enumerated_total,
        wall_clock_us,
        ..Default::default()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cep_importance::{FfnImportance, HeadImportance, ImportanceTable, LayerImportance};
    use crate::cep_oracle::{Activation, ModelSpec, NormType};
    use crate::cep_rewrite::SearchAxes;
    use crate::gpu_specs::find_gpu;

    fn gpu() -> &'static GpuSpec {
        find_gpu("H100-SXM").expect("H100-SXM is in the GPU database")
    }

    /// d=256, 4 layers, 8 query heads in 4 key/value groups of 2, head_dim
    /// 32, d_ff 512, vocab 8192, tied head.
    ///
    /// Parameters: embedding 8192·256 = 2,097,152; a block is
    /// 256·256 (Q) + 2·256·128 (K, V) + 256·256 (O) = 196,608, plus
    /// 3·256·512 = 393,216 (FFN), plus 512 (norms) = 590,336; final norm 256.
    /// Total 2,097,152 + 4·590,336 + 256 = 4,458,752.
    ///
    /// One key/value group is 64 columns of Q, 64 rows of O and 32 columns of
    /// K and V: 2·256·64 + 2·256·32 = 49,152 parameters. One FFN step is
    /// 512/8 = 64 neurons: 3·256·64 = 49,152 parameters as well.
    fn spec() -> ModelSpec {
        ModelSpec::uniform(256, 4, 8, 4, 32, 512, 8192)
    }
    const BASE_PARAMS: u64 = 4_458_752;
    const STEP_PARAMS: u64 = 49_152;

    /// Every head scores `head_score(layer, head)`, every neuron
    /// `neuron(layer, j)`, with neutral slack and position factors so the
    /// numbers given are the numbers ranked.
    fn table(
        spec: &ModelSpec,
        head_score: impl Fn(u32, u32) -> f64,
        neuron: impl Fn(u32, u32) -> f64,
    ) -> ImportanceTable {
        let mut t = ImportanceTable::default();
        for l in 0..spec.n_layers {
            let mut attn = 0.0;
            for h in 0..spec.n_heads[l as usize] {
                let score = head_score(l, h);
                attn += score;
                t.heads.push(HeadImportance {
                    layer: l,
                    head: h,
                    weight_magnitude: score,
                    spectral_energy: 1.0,
                    roofline_slack: 1.0,
                    position_factor: 1.0,
                    score,
                });
            }
            let mags: Vec<f64> = (0..spec.d_ff[l as usize]).map(|j| neuron(l, j)).collect();
            let ffn: f64 = mags.iter().sum();
            t.ffns.push(FfnImportance {
                layer: l,
                weight_magnitude: ffn,
                position_factor: 1.0,
                roofline_slack: 1.0,
                score: ffn,
                neuron_magnitudes: mags,
            });
            t.layers.push(LayerImportance {
                layer: l,
                attention_score: attn,
                ffn_score: ffn,
                total_score: attn + ffn,
            });
        }
        t
    }

    fn flat(spec: &ModelSpec) -> ImportanceTable {
        table(spec, |_, _| 1.0, |_, _| 1.0)
    }

    /// Every layer at the floor the search may reach: one key/value group
    /// (2 query heads) and an FFN of 512·3/8 = 192.
    fn floor_spec() -> ModelSpec {
        ModelSpec::uniform(256, 4, 2, 1, 32, 192, 8192)
    }

    #[test]
    fn ffn_steps_and_floor() {
        // An eighth of the width, in multiples of 64 when it is that wide.
        // 1408·3/8 = 528, which rounds down to 512: seven steps of 128.
        assert_eq!((ffn_step(1408), ffn_floor(1408)), (128, 512));
        assert_eq!((ffn_step(512), ffn_floor(512)), (64, 192));
        // A narrow FFN still has more than one width to try: 128 -> 112, 96, 80, 64, 48.
        assert_eq!((ffn_step(128), ffn_floor(128)), (16, 48));
        assert_eq!((ffn_step(6), ffn_floor(6)), (1, 3));
    }

    /// A 2% target needs 89,176 parameters gone; one step removes 49,152, so
    /// exactly two steps are taken.
    #[test]
    fn stops_as_soon_as_the_target_is_met() {
        let s = spec();
        assert_eq!(s.param_count(), BASE_PARAMS);
        let c = Constraints {
            target_sparsity: 0.02,
            ..Default::default()
        };
        let o = prune_greedy(&s, &flat(&s), gpu(), &c, Granularity::HeadAndFfn);
        assert_eq!(o.prune_log.len(), 2);
        let chosen = o.chosen.as_ref().unwrap();
        assert_eq!(chosen.spec.param_count(), BASE_PARAMS - 2 * STEP_PARAMS);
        assert!(o.sparsity_met && o.constraints_met && chosen.feasible);
        assert!(
            (o.achieved_sparsity - 2.0 * STEP_PARAMS as f64 / BASE_PARAMS as f64).abs() < 1e-12
        );
    }

    /// Components whose score is zero are the ones that go, wherever they
    /// are: here key/value group 2 of layer 2 (heads 4 and 5) and the first
    /// 64 neurons of layer 1.
    #[test]
    fn zero_score_components_are_pruned_first() {
        let s = spec();
        let imp = table(
            &s,
            |l, h| {
                if l == 2 && (h == 4 || h == 5) {
                    0.0
                } else {
                    1.0
                }
            },
            |l, j| if l == 1 && j < 64 { 0.0 } else { 1.0 },
        );
        let c = Constraints {
            target_sparsity: 0.02,
            ..Default::default()
        };
        let o = prune_greedy(&s, &imp, gpu(), &c, Granularity::HeadAndFfn);
        let steps: Vec<(&str, u32, Vec<u32>, Option<u32>)> = o
            .prune_log
            .iter()
            .map(|p| (p.kind.as_str(), p.layer, p.heads.clone(), p.new_d_ff))
            .collect();
        assert_eq!(
            steps,
            vec![("head", 2, vec![4, 5], None), ("ffn", 1, vec![], Some(448))]
        );
        let delta = o.delta.as_ref().unwrap();
        assert_eq!(delta.per_layer[2].pruned_heads, vec![4, 5]);
        assert_eq!(delta.per_layer[1].new_d_ff, Some(448));
        assert!(
            delta.per_layer[0].pruned_heads.is_empty() && delta.per_layer[0].new_d_ff.is_none()
        );
    }

    /// A limit the baseline exceeds is a target to prune towards. With no
    /// sparsity asked for, the search must still take steps until the latency
    /// limit holds, and no longer.
    #[test]
    fn a_latency_limit_is_pruned_towards() {
        let s = spec();
        let base = evaluate(&s, gpu()).unwrap().estimated_latency_us;
        let floor = evaluate(&floor_spec(), gpu()).unwrap().estimated_latency_us;
        assert!(floor < base);
        let limit = (base + floor) / 2.0;
        let c = Constraints {
            target_sparsity: 0.0,
            latency_us: limit,
            ..Default::default()
        };
        let o = prune_greedy(&s, &flat(&s), gpu(), &c, Granularity::HeadAndFfn);
        assert!(
            !o.prune_log.is_empty(),
            "the baseline exceeds the limit, so steps are needed"
        );
        assert!(o.constraints_met && o.sparsity_met);
        let chosen = o.chosen.as_ref().unwrap();
        assert!(chosen.feasible && chosen.profile.estimated_latency_us <= limit);
        // It stopped at the first step that met the limit.
        let (last, earlier) = o.prune_log.split_last().unwrap();
        assert!(last.new_latency_us <= limit);
        assert!(earlier.iter().all(|p| p.new_latency_us > limit));
    }

    /// The same for memory.
    #[test]
    fn a_memory_limit_is_pruned_towards() {
        let s = spec();
        let base = evaluate(&s, gpu()).unwrap().peak_memory_bytes;
        let floor = evaluate(&floor_spec(), gpu()).unwrap().peak_memory_bytes;
        assert!(floor < base);
        let limit = (base + floor) / 2;
        let c = Constraints {
            target_sparsity: 0.0,
            peak_memory_bytes: limit,
            ..Default::default()
        };
        let o = joint_prune_search(&s, &flat(&s), gpu(), &c, Granularity::HeadAndFfn);
        assert!(!o.prune_log.is_empty());
        let chosen = o.chosen.as_ref().unwrap();
        assert!(o.constraints_met && chosen.feasible && chosen.profile.peak_memory_bytes <= limit);
    }

    /// A limit below what the most pruned model reaches cannot be met. The
    /// search goes as far as it may (every layer at the floor) and says the
    /// limit was missed.
    #[test]
    fn an_unreachable_limit_is_reported() {
        let s = spec();
        let floor = evaluate(&floor_spec(), gpu()).unwrap().estimated_latency_us;
        let c = Constraints {
            target_sparsity: 0.0,
            latency_us: floor / 2.0,
            ..Default::default()
        };
        let o = prune_greedy(&s, &flat(&s), gpu(), &c, Granularity::HeadAndFfn);
        let chosen = o.chosen.as_ref().unwrap();
        assert_eq!(chosen.spec, floor_spec());
        assert!(!o.constraints_met && !chosen.feasible);
        assert!(o.sparsity_met, "no sparsity was asked for");
        // 3 groups and 5 FFN steps per layer.
        assert_eq!(o.prune_log.len(), 4 * (3 + 5));
    }

    /// Heads alone cannot remove 90% of this model. At the floor each layer
    /// has lost 3 groups: 4·3·49,152 = 589,824 parameters, 13.2%.
    #[test]
    fn a_missed_sparsity_target_is_reported() {
        let s = spec();
        let c = Constraints {
            target_sparsity: 0.9,
            ..Default::default()
        };
        let o = prune_greedy(&s, &flat(&s), gpu(), &c, Granularity::Head);
        let chosen = o.chosen.as_ref().unwrap();
        assert_eq!(chosen.spec.param_count(), BASE_PARAMS - 589_824);
        assert_eq!(chosen.spec.n_heads, vec![2; 4]);
        assert_eq!(
            chosen.spec.d_ff,
            vec![512; 4],
            "FFN is not in this granularity"
        );
        assert!(!o.sparsity_met && !chosen.feasible);
        assert!(o.constraints_met);
        assert!((o.achieved_sparsity - 589_824.0 / BASE_PARAMS as f64).abs() < 1e-12);
        assert_eq!(o.target_sparsity, 0.9);
    }

    /// With one key/value head there is no group that may go, so a head-only
    /// prune can do nothing, and must say so instead of logging steps.
    #[test]
    fn a_single_key_value_head_leaves_nothing_to_prune() {
        let s = ModelSpec::uniform(256, 4, 8, 1, 32, 512, 8192);
        let c = Constraints {
            target_sparsity: 0.3,
            ..Default::default()
        };
        let o = prune_greedy(&s, &flat(&s), gpu(), &c, Granularity::Head);
        assert!(o.prune_log.is_empty());
        assert_eq!(o.chosen.as_ref().unwrap().spec, s);
        assert!(!o.sparsity_met);
        assert_eq!(o.achieved_sparsity, 0.0);
    }

    /// Head and FFN steps are taken as their importance per parameter
    /// dictates, not one kind to exhaustion before the other, and not one
    /// layer to its floor before the next. Groups score 1, 2, 3, 4 and
    /// neurons 1..=512 in every layer; a 20% target is 891,751 parameters,
    /// 19 steps.
    #[test]
    fn heads_and_ffn_are_interleaved_across_layers() {
        let s = spec();
        let imp = table(&s, |_, h| (h / 2 + 1) as f64, |_, j| (j + 1) as f64);
        let c = Constraints {
            target_sparsity: 0.2,
            ..Default::default()
        };
        let o = prune_greedy(&s, &imp, gpu(), &c, Granularity::HeadAndFfn);
        assert!(o.sparsity_met);
        assert_eq!(o.prune_log.len(), 19);
        let heads = o.prune_log.iter().filter(|p| p.kind == "head").count();
        let ffn = o.prune_log.iter().filter(|p| p.kind == "ffn").count();
        assert!(heads > 0 && ffn > 0, "{heads} head steps, {ffn} FFN steps");
        let chosen = &o.chosen.as_ref().unwrap().spec;
        // Every layer has the same scores, so they are pruned evenly: no
        // layer is more than one step of either kind ahead of another.
        let (h_min, h_max) = (
            chosen.n_heads.iter().min().unwrap(),
            chosen.n_heads.iter().max().unwrap(),
        );
        let (f_min, f_max) = (
            chosen.d_ff.iter().min().unwrap(),
            chosen.d_ff.iter().max().unwrap(),
        );
        assert!(h_max - h_min <= 2, "{:?}", chosen.n_heads);
        assert!(f_max - f_min <= 64, "{:?}", chosen.d_ff);
        // The first group to go in each layer is the lowest-scoring one.
        for l in 0..4 {
            let first = o
                .prune_log
                .iter()
                .find(|p| p.kind == "head" && p.layer == l);
            if let Some(p) = first {
                assert_eq!(p.heads, vec![0, 1]);
            }
        }
    }

    #[test]
    fn preserved_layers_are_left_alone() {
        let s = spec();
        let c = Constraints {
            target_sparsity: 0.9, // unreachable: everything that may go, goes
            preserve_layers: vec![0, 3],
            ..Default::default()
        };
        let o = joint_prune_search(&s, &flat(&s), gpu(), &c, Granularity::HeadAndFfn);
        assert!(o.prune_log.iter().all(|p| p.layer == 1 || p.layer == 2));
        let delta = o.delta.as_ref().unwrap();
        for l in [0usize, 3] {
            assert!(delta.per_layer[l].pruned_heads.is_empty());
            assert!(delta.per_layer[l].new_d_ff.is_none() && !delta.per_layer[l].drop_layer);
        }
    }

    /// In uniform mode every step is applied to all layers at once, so the
    /// layers always share one shape. A 10% target is 445,876 parameters; a
    /// uniform step removes 4·49,152 = 196,608, so three are taken.
    #[test]
    fn uniform_mode_keeps_every_layer_at_one_shape() {
        let s = spec();
        // Different scores per layer, which would otherwise prune unevenly.
        let imp = table(&s, |l, h| (1 + l + h) as f64, |l, j| (1 + l * 7 + j) as f64);
        let c = Constraints {
            target_sparsity: 0.1,
            uniform: true,
            ..Default::default()
        };
        let o = prune_greedy(&s, &imp, gpu(), &c, Granularity::HeadAndFfn);
        let chosen = &o.chosen.as_ref().unwrap().spec;
        assert!(o.sparsity_met);
        assert_eq!(chosen.param_count(), BASE_PARAMS - 3 * 4 * STEP_PARAMS);
        assert!(
            chosen.n_heads.iter().all(|&h| h == chosen.n_heads[0]),
            "{:?}",
            chosen.n_heads
        );
        assert!(
            chosen.d_ff.iter().all(|&f| f == chosen.d_ff[0]),
            "{:?}",
            chosen.d_ff
        );
        assert_eq!(o.prune_log.len(), 3 * 4, "one log line per layer per step");
    }

    /// A layer whose every score is zero costs nothing to drop and saves the
    /// most, so the joint search drops it whole before anything else.
    #[test]
    fn joint_search_drops_a_dead_layer_first() {
        let s = spec();
        let imp = table(
            &s,
            |l, _| if l == 1 { 0.0 } else { 1.0 },
            |l, _| if l == 1 { 0.0 } else { 1.0 },
        );
        let c = Constraints {
            target_sparsity: 0.1,
            ..Default::default()
        };
        let o = joint_prune_search(&s, &imp, gpu(), &c, Granularity::HeadAndFfn);
        assert_eq!(o.prune_log[0].kind, "layer");
        assert_eq!(o.prune_log[0].layer, 1);
        // One block is 590,336 parameters, 13.2%: the target is met in one step.
        assert_eq!(o.prune_log.len(), 1);
        let chosen = &o.chosen.as_ref().unwrap().spec;
        assert_eq!(chosen.n_layers, 3);
        assert_eq!(chosen.param_count(), BASE_PARAMS - 590_336);
        let delta = o.delta.as_ref().unwrap();
        assert!(delta.per_layer[1].drop_layer);
        assert!(!delta.per_layer[0].drop_layer && !delta.per_layer[2].drop_layer);
        // The plain prune search never drops a layer.
        let p = prune_greedy(&s, &imp, gpu(), &c, Granularity::HeadAndFfn);
        assert!(p.prune_log.iter().all(|st| st.kind != "layer"));
        assert_eq!(p.chosen.as_ref().unwrap().spec.n_layers, 4);
    }

    #[test]
    fn joint_search_keeps_at_least_one_layer() {
        let s = spec();
        let c = Constraints {
            target_sparsity: 0.999,
            ..Default::default()
        };
        let o = joint_prune_search(&s, &flat(&s), gpu(), &c, Granularity::HeadAndFfn);
        let chosen = o.chosen.as_ref().unwrap();
        assert_eq!(chosen.spec.n_layers, 1);
        assert_eq!(o.prune_log.iter().filter(|p| p.kind == "layer").count(), 3);
        assert!(!o.sparsity_met && !chosen.feasible);
    }

    #[test]
    fn nothing_is_pruned_when_nothing_is_asked() {
        let s = spec();
        let c = Constraints {
            target_sparsity: 0.0,
            ..Default::default()
        };
        let o = joint_prune_search(&s, &flat(&s), gpu(), &c, Granularity::HeadAndFfn);
        assert!(o.prune_log.is_empty());
        assert_eq!(o.chosen.as_ref().unwrap().spec, s);
        assert!(o.sparsity_met && o.constraints_met);
    }

    /// The retained-capacity score: an untouched model retains its
    /// `2·n_layers` sublayers; each step lowers it by what was removed. With
    /// flat scores a group is a quarter of an attention sublayer and an FFN
    /// step 64/512 of an FFN sublayer.
    #[test]
    fn chosen_score_is_the_capacity_retained() {
        let s = spec();
        let none = prune_greedy(
            &s,
            &flat(&s),
            gpu(),
            &Constraints {
                target_sparsity: 0.0,
                ..Default::default()
            },
            Granularity::HeadAndFfn,
        );
        assert!((none.chosen.as_ref().unwrap().score - 8.0).abs() < 1e-9);
        // Head-only, one step: a quarter of one attention sublayer goes.
        let one = prune_greedy(
            &s,
            &flat(&s),
            gpu(),
            &Constraints {
                target_sparsity: 0.01,
                ..Default::default()
            },
            Granularity::Head,
        );
        assert_eq!(one.prune_log.len(), 1);
        assert!((one.chosen.as_ref().unwrap().score - 7.75).abs() < 1e-9);
    }

    fn axes() -> SearchAxes {
        SearchAxes {
            d_model: vec![128, 256],
            n_layers: vec![2, 4],
            n_heads: vec![4, 8],
            n_kv_heads: vec![2, 4],
            d_ff: vec![256, 512],
            activation: vec![Activation::SwiGlu],
            norm: vec![NormType::RmsNorm],
            vocab: 8192,
            head_dim: 32,
            max_seq: 512,
            batch: 1,
            dtype_bytes: 2,
            tied_lm_head: true,
        }
    }

    /// Latency grows with every one of these axes, so under `min_latency` the
    /// smallest of each comes first and the largest last.
    #[test]
    fn min_latency_ranks_the_smallest_architecture_first() {
        let o = architecture_search(
            &axes(),
            gpu(),
            &Constraints::default(),
            NasObjective::MinLatency,
        );
        // 2·2·2·2·2 = 32 combinations, all valid (2 and 4 key/value heads
        // both divide 4 and 8 query heads).
        assert_eq!(o.candidates_enumerated, 32);
        assert_eq!(o.ranked_candidates.len(), 32);
        let first = &o.ranked_candidates[0].spec;
        assert_eq!(
            (
                first.d_model,
                first.n_layers,
                first.n_heads[0],
                first.d_ff[0]
            ),
            (128, 2, 4, 256)
        );
        let last = &o.ranked_candidates[31].spec;
        assert_eq!(
            (last.d_model, last.n_layers, last.n_heads[0], last.d_ff[0]),
            (256, 4, 8, 512)
        );
        for w in o.ranked_candidates.windows(2) {
            assert!(w[0].profile.estimated_latency_us <= w[1].profile.estimated_latency_us);
        }
        assert_eq!(o.chosen.as_ref().unwrap().spec, *first);
    }

    #[test]
    fn feasible_candidates_rank_ahead_of_infeasible_ones() {
        let all = architecture_search(
            &axes(),
            gpu(),
            &Constraints::default(),
            NasObjective::MinParams,
        );
        // A memory limit at the median splits the candidates.
        let mut mems: Vec<u64> = all
            .ranked_candidates
            .iter()
            .map(|c| c.profile.peak_memory_bytes)
            .collect();
        mems.sort_unstable();
        let limit = mems[mems.len() / 2];
        let c = Constraints {
            peak_memory_bytes: limit,
            ..Default::default()
        };
        // Under an objective that prefers BIG models, the best-scoring
        // candidates are the infeasible ones; they must still come last.
        let o = architecture_search(&axes(), gpu(), &c, NasObjective::ParamEfficiency);
        let n_feasible = o.ranked_candidates.iter().filter(|c| c.feasible).count();
        assert!(n_feasible > 0 && n_feasible < o.ranked_candidates.len());
        assert!(o.ranked_candidates[..n_feasible].iter().all(|c| c.feasible));
        assert!(
            o.ranked_candidates[n_feasible..]
                .iter()
                .all(|c| !c.feasible)
        );
        assert!(
            o.ranked_candidates
                .iter()
                .all(|c| c.feasible == (c.profile.peak_memory_bytes <= limit))
        );
        assert!(o.chosen.as_ref().unwrap().feasible);
    }

    #[test]
    fn no_feasible_candidate_means_no_choice() {
        let c = Constraints {
            peak_memory_bytes: 1,
            ..Default::default()
        };
        let o = architecture_search(&axes(), gpu(), &c, NasObjective::MinLatency);
        assert!(o.chosen.is_none());
        assert_eq!(o.ranked_candidates.len(), 32);
        assert!(o.ranked_candidates.iter().all(|c| !c.feasible));
    }

    /// ReLU and GELU FFNs have the same two matrices, so the estimate cannot
    /// tell them apart: one row, with the other named beside it. RMSNorm and
    /// LayerNorm differ in parameters and stay separate.
    #[test]
    fn candidates_the_estimate_cannot_tell_apart_are_listed_once() {
        let mut a = axes();
        a.d_model = vec![128];
        a.n_layers = vec![2];
        a.n_heads = vec![4];
        a.n_kv_heads = vec![2];
        a.d_ff = vec![256];
        a.activation = vec![Activation::Relu, Activation::Gelu];
        a.norm = vec![NormType::RmsNorm, NormType::LayerNorm];
        let o = architecture_search(&a, gpu(), &Constraints::default(), NasObjective::MinParams);
        assert_eq!(o.candidates_enumerated, 4);
        assert_eq!(o.candidates_evaluated, 4);
        assert_eq!(o.ranked_candidates.len(), 2);
        // Fewest parameters first: RMSNorm (one vector per norm).
        assert_eq!(o.ranked_candidates[0].spec.norm, NormType::RmsNorm);
        assert_eq!(o.ranked_candidates[0].spec.activation, Activation::Relu);
        assert_eq!(
            o.ranked_candidates[0].also,
            vec!["activation=gelu, norm=rms_norm".to_string()]
        );
        assert_eq!(
            o.ranked_candidates[1].also,
            vec!["activation=gelu, norm=layer_norm".to_string()]
        );
    }

    #[test]
    fn empty_search_returns_empty_outcome() {
        let mut a = axes();
        a.d_model = Vec::new();
        let o = architecture_search(
            &a,
            gpu(),
            &Constraints::default(),
            NasObjective::ParamEfficiency,
        );
        assert_eq!(o.candidates_evaluated, 0);
        assert!(o.chosen.is_none());
    }
}
