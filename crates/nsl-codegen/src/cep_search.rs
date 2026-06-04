//! CEP — greedy pruning search + architecture search.
//!
//! The paper's §3.4 algorithm (Algorithm 1): starting from a
//! pre-trained model, rank every structural component by importance,
//! iteratively remove the least-important component, recompile, verify
//! constraints, and stop when the sparsity target is reached.
//!
//! This module implements both entry points:
//!
//!   * [`prune_greedy`] — compilation-verified pruning (paper §3.4)
//!   * [`architecture_search`] — enumerate-and-rank NAS (paper §3.5)
//!
//! Both return a [`SearchOutcome`] whose structure is identical so the
//! driver can render a unified report.

use serde::Serialize;

use crate::cep_importance::{HeadImportance, ImportanceTable};
use crate::cep_oracle::{evaluate, CompilationProfile, ModelSpec};
use crate::cep_rewrite::{apply_delta, PruneDelta};
use crate::gpu_specs::GpuSpec;

/// Constraints enforced by the search.
#[derive(Debug, Clone)]
pub struct Constraints {
    /// Upper bound on peak memory (bytes).  Candidates exceeding this
    /// are infeasible.
    pub peak_memory_bytes: u64,
    /// Upper bound on estimated latency (μs).
    pub latency_us: f64,
    /// Fractional sparsity target (0..1).  The prune search stops when
    /// total parameter count falls by at least this fraction.
    pub target_sparsity: f64,
    /// Layer glob patterns to protect from pruning (paper §4.1 example:
    /// `["blocks.0.*", "blocks.7.*"]`).
    pub preserve_layers: Vec<u32>,
}

impl Default for Constraints {
    fn default() -> Self {
        Self {
            peak_memory_bytes: u64::MAX,
            latency_us: f64::MAX,
            target_sparsity: 0.3,
            preserve_layers: Vec::new(),
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

/// Ranking objective used for NAS.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum NasObjective {
    /// `params × roofline_util / latency` (paper's recommended metric).
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
}

/// Single evaluated candidate with its profile and objective score.
#[derive(Debug, Clone)]
pub struct EvaluatedCandidate {
    pub spec: ModelSpec,
    pub profile: CompilationProfile,
    pub score: f64,
    pub feasible: bool,
}

/// Output of either search entry point.
#[derive(Debug, Clone, Default)]
pub struct SearchOutcome {
    /// Baseline profile (original model, pre-prune).  `None` for NAS.
    pub baseline: Option<CompilationProfile>,
    /// Chosen spec + its profile.
    pub chosen: Option<EvaluatedCandidate>,
    /// All evaluated candidates, ranked by score (descending).
    pub ranked_candidates: Vec<EvaluatedCandidate>,
    /// Log of pruning steps (for greedy prune).
    pub prune_log: Vec<PruneStep>,
    /// Number of candidates evaluated.
    pub candidates_evaluated: u32,
}

#[derive(Debug, Clone, Serialize)]
pub struct PruneStep {
    pub kind: String,
    pub layer: u32,
    pub head: Option<u32>,
    pub accepted: bool,
    pub reason: String,
    pub new_param_bytes: u64,
    pub new_latency_us: f64,
}

fn satisfies_constraints(profile: &CompilationProfile, c: &Constraints) -> bool {
    profile.peak_memory_bytes <= c.peak_memory_bytes
        && profile.estimated_latency_us <= c.latency_us
}

fn sparsity_reached(baseline_params: u64, current_params: u64, target: f64) -> bool {
    if baseline_params == 0 {
        return true;
    }
    let reduction = 1.0 - (current_params as f64 / baseline_params as f64);
    reduction >= target
}

/// Order the importance table so the least-important component is
/// returned first.  Preserved layers are filtered out entirely.
fn prune_order(importance: &ImportanceTable, preserve: &[u32]) -> Vec<HeadImportance> {
    let mut heads = importance.heads.clone();
    heads.retain(|h| !preserve.contains(&h.layer));
    heads.sort_by(|a, b| {
        a.score
            .partial_cmp(&b.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    heads
}

/// Greedy compilation-verified pruning (Algorithm 1 from the paper).
pub fn prune_greedy(
    base_spec: &ModelSpec,
    importance: &ImportanceTable,
    gpu: &GpuSpec,
    constraints: &Constraints,
    granularity: Granularity,
) -> SearchOutcome {
    let baseline_profile = evaluate(base_spec, gpu).expect("base spec must validate");
    let baseline_params = base_spec.param_count();

    let mut current_spec = base_spec.clone();
    let mut current_profile = baseline_profile.clone();
    let mut prune_log = Vec::new();
    let mut candidates_evaluated = 1u32;

    let ordered = prune_order(importance, &constraints.preserve_layers);

    for h in ordered {
        // Only prune at head granularity if allowed; if only Ffn-mode,
        // skip heads and fall through to FFN reduction below.
        if matches!(granularity, Granularity::Ffn) {
            continue;
        }

        // Build a trial delta: drop this one head from its layer.
        let mut delta = PruneDelta::for_spec(&current_spec);
        if let Some(d) = delta.per_layer.iter_mut().find(|d| d.layer == h.layer) {
            d.pruned_heads.push(h.head);
        }
        let trial_spec = apply_delta(&current_spec, &delta);
        let trial_profile = match evaluate(&trial_spec, gpu) {
            Ok(p) => p,
            Err(_) => {
                prune_log.push(PruneStep {
                    kind: "head".into(),
                    layer: h.layer,
                    head: Some(h.head),
                    accepted: false,
                    reason: "pruned spec failed validation".into(),
                    new_param_bytes: current_profile.param_bytes,
                    new_latency_us: current_profile.estimated_latency_us,
                });
                continue;
            }
        };
        candidates_evaluated += 1;

        if !satisfies_constraints(&trial_profile, constraints) {
            prune_log.push(PruneStep {
                kind: "head".into(),
                layer: h.layer,
                head: Some(h.head),
                accepted: false,
                reason: "would violate constraints".into(),
                new_param_bytes: current_profile.param_bytes,
                new_latency_us: current_profile.estimated_latency_us,
            });
            continue;
        }

        // Accept.
        current_spec = trial_spec;
        current_profile = trial_profile;
        prune_log.push(PruneStep {
            kind: "head".into(),
            layer: h.layer,
            head: Some(h.head),
            accepted: true,
            reason: "pruned (least important)".into(),
            new_param_bytes: current_profile.param_bytes,
            new_latency_us: current_profile.estimated_latency_us,
        });

        if sparsity_reached(baseline_params, current_spec.param_count(), constraints.target_sparsity) {
            break;
        }
    }

    // FFN-granularity sweep: shrink each layer's FFN if allowed.
    if matches!(granularity, Granularity::Ffn | Granularity::HeadAndFfn)
        && !sparsity_reached(baseline_params, current_spec.param_count(), constraints.target_sparsity)
    {
        let candidate_widths: [u32; 3] = [1024, 768, 512];
        // Iterate from least-important layer (low importance score).
        let mut layer_scores: Vec<(u32, f64)> = importance
            .layers
            .iter()
            .filter(|l| !constraints.preserve_layers.contains(&l.layer))
            .map(|l| (l.layer, l.total_score))
            .collect();
        layer_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (layer, _) in layer_scores {
            for &w in &candidate_widths {
                let mut delta = PruneDelta::for_spec(&current_spec);
                if let Some(d) = delta.per_layer.iter_mut().find(|d| d.layer == layer) {
                    d.new_d_ff = Some(w);
                }
                let trial_spec = apply_delta(&current_spec, &delta);
                if trial_spec.d_ff[layer as usize] >= current_spec.d_ff[layer as usize] {
                    continue;
                }
                let Ok(trial_profile) = evaluate(&trial_spec, gpu) else {
                    continue;
                };
                candidates_evaluated += 1;
                if !satisfies_constraints(&trial_profile, constraints) {
                    prune_log.push(PruneStep {
                        kind: "ffn".into(),
                        layer,
                        head: None,
                        accepted: false,
                        reason: format!("width {w} violates constraints"),
                        new_param_bytes: current_profile.param_bytes,
                        new_latency_us: current_profile.estimated_latency_us,
                    });
                    continue;
                }
                current_spec = trial_spec;
                current_profile = trial_profile;
                prune_log.push(PruneStep {
                    kind: "ffn".into(),
                    layer,
                    head: None,
                    accepted: true,
                    reason: format!("shrunk to {w}"),
                    new_param_bytes: current_profile.param_bytes,
                    new_latency_us: current_profile.estimated_latency_us,
                });
                if sparsity_reached(
                    baseline_params,
                    current_spec.param_count(),
                    constraints.target_sparsity,
                ) {
                    break;
                }
            }
            if sparsity_reached(
                baseline_params,
                current_spec.param_count(),
                constraints.target_sparsity,
            ) {
                break;
            }
        }
    }

    SearchOutcome {
        baseline: Some(baseline_profile.clone()),
        chosen: Some(EvaluatedCandidate {
            spec: current_spec.clone(),
            profile: current_profile.clone(),
            score: (baseline_params.saturating_sub(current_spec.param_count())) as f64,
            feasible: satisfies_constraints(&current_profile, constraints),
        }),
        ranked_candidates: Vec::new(),
        prune_log,
        candidates_evaluated,
    }
}

/// Architecture search: enumerate the cross-product of search axes,
/// evaluate each candidate, and rank by the chosen objective.
pub fn architecture_search(
    axes: &crate::cep_rewrite::SearchAxes,
    gpu: &GpuSpec,
    constraints: &Constraints,
    objective: NasObjective,
) -> SearchOutcome {
    let mut evaluated = Vec::new();
    for spec in axes.enumerate() {
        let Ok(profile) = evaluate(&spec, gpu) else {
            continue;
        };
        let feasible = satisfies_constraints(&profile, constraints);
        let score = objective.score(&profile, &spec);
        evaluated.push(EvaluatedCandidate {
            spec,
            profile,
            score,
            feasible,
        });
    }
    // Rank by score (descending).  Infeasible candidates end up at the
    // bottom because their score is often negative or dominated.
    evaluated.sort_by(|a, b| {
        match (a.feasible, b.feasible) {
            (true, false) => std::cmp::Ordering::Less,
            (false, true) => std::cmp::Ordering::Greater,
            _ => b
                .score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal),
        }
    });
    let candidates_evaluated = evaluated.len() as u32;
    let chosen = evaluated.iter().find(|c| c.feasible).cloned();
    SearchOutcome {
        baseline: None,
        chosen,
        ranked_candidates: evaluated,
        prune_log: Vec::new(),
        candidates_evaluated,
    }
}

/// Paper §2.2 Mode 3: joint prune-search — starting from a pre-trained model with
/// weight importance, greedily prune across the joint action space of attention heads
/// (Mode 1), FFN widths (Mode 1), AND layer drops (Mode 3 addition). Phases run
/// sequentially against a delta-against-baseline model:
///   1. head pruning (least-important head first, GQA-group snap via `apply_delta`)
///   2. FFN width shrinking (candidate widths 75/50/37.5% of current per least-important layer)
///   3. layer drops (least-important layer first; never drops to zero layers)
/// Each phase short-circuits when `target_sparsity` is reached. The result is a
/// SearchOutcome whose `prune_log` records every attempt and `chosen` is the
/// final pruned spec.
pub fn joint_prune_search(
    base_spec: &ModelSpec,
    importance: &ImportanceTable,
    gpu: &GpuSpec,
    constraints: &Constraints,
    granularity: Granularity,
) -> SearchOutcome {
    let baseline_profile = evaluate(base_spec, gpu).expect("base spec must validate");
    let baseline_params = base_spec.param_count();

    let mut delta = PruneDelta::for_spec(base_spec);
    let mut current_profile = baseline_profile.clone();
    let mut current_params = baseline_params;
    let mut prune_log = Vec::new();
    let mut candidates_evaluated = 1u32;

    // Short-circuit when the baseline already meets the sparsity target (e.g. target=0):
    // no pruning needed, no prune_log entries produced.
    if sparsity_reached(baseline_params, current_params, constraints.target_sparsity) {
        return finish_joint(
            baseline_profile,
            base_spec,
            &delta,
            current_profile,
            prune_log,
            candidates_evaluated,
            constraints,
        );
    }

    // ----- Phase 1: head pruning (skip if Granularity::Ffn) -----
    if !matches!(granularity, Granularity::Ffn) {
        let ordered = prune_order(importance, &constraints.preserve_layers);
        for h in ordered {
            // Skip heads already pruned via the accumulated delta.
            let already = delta
                .per_layer
                .iter()
                .find(|d| d.layer == h.layer)
                .map(|d| d.pruned_heads.contains(&h.head))
                .unwrap_or(false);
            if already {
                continue;
            }
            let mut trial_delta = delta.clone();
            if let Some(d) = trial_delta.per_layer.iter_mut().find(|d| d.layer == h.layer) {
                d.pruned_heads.push(h.head);
            }
            let trial_spec = apply_delta(base_spec, &trial_delta);
            let trial_profile = match evaluate(&trial_spec, gpu) {
                Ok(p) => p,
                Err(_) => {
                    prune_log.push(PruneStep {
                        kind: "head".into(),
                        layer: h.layer,
                        head: Some(h.head),
                        accepted: false,
                        reason: "pruned spec failed validation".into(),
                        new_param_bytes: current_profile.param_bytes,
                        new_latency_us: current_profile.estimated_latency_us,
                    });
                    continue;
                }
            };
            candidates_evaluated += 1;
            if !satisfies_constraints(&trial_profile, constraints) {
                prune_log.push(PruneStep {
                    kind: "head".into(),
                    layer: h.layer,
                    head: Some(h.head),
                    accepted: false,
                    reason: "would violate constraints".into(),
                    new_param_bytes: current_profile.param_bytes,
                    new_latency_us: current_profile.estimated_latency_us,
                });
                continue;
            }
            delta = trial_delta;
            current_profile = trial_profile;
            current_params = trial_spec.param_count();
            prune_log.push(PruneStep {
                kind: "head".into(),
                layer: h.layer,
                head: Some(h.head),
                accepted: true,
                reason: "joint: pruned (least important)".into(),
                new_param_bytes: current_profile.param_bytes,
                new_latency_us: current_profile.estimated_latency_us,
            });
            if sparsity_reached(baseline_params, current_params, constraints.target_sparsity) {
                return finish_joint(baseline_profile, base_spec, &delta, current_profile, prune_log, candidates_evaluated, constraints);
            }
        }
    }

    // ----- Phase 2: FFN width sweep -----
    if matches!(granularity, Granularity::Ffn | Granularity::HeadAndFfn) {
        let mut layer_scores: Vec<(u32, f64)> = importance
            .layers
            .iter()
            .filter(|l| !constraints.preserve_layers.contains(&l.layer))
            .map(|l| (l.layer, l.total_score))
            .collect();
        layer_scores
            .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (layer, _) in layer_scores {
            let current_ff = current_ff_for(base_spec, &delta, layer);
            for w in ffn_candidate_widths(current_ff) {
                if w >= current_ff {
                    continue;
                }
                let mut trial_delta = delta.clone();
                if let Some(d) = trial_delta.per_layer.iter_mut().find(|d| d.layer == layer) {
                    d.new_d_ff = Some(w);
                }
                let trial_spec = apply_delta(base_spec, &trial_delta);
                let Ok(trial_profile) = evaluate(&trial_spec, gpu) else { continue };
                candidates_evaluated += 1;
                if !satisfies_constraints(&trial_profile, constraints) {
                    prune_log.push(PruneStep {
                        kind: "ffn".into(),
                        layer,
                        head: None,
                        accepted: false,
                        reason: format!("joint: width {w} violates constraints"),
                        new_param_bytes: current_profile.param_bytes,
                        new_latency_us: current_profile.estimated_latency_us,
                    });
                    continue;
                }
                delta = trial_delta;
                current_profile = trial_profile;
                current_params = trial_spec.param_count();
                prune_log.push(PruneStep {
                    kind: "ffn".into(),
                    layer,
                    head: None,
                    accepted: true,
                    reason: format!("joint: shrunk to {w}"),
                    new_param_bytes: current_profile.param_bytes,
                    new_latency_us: current_profile.estimated_latency_us,
                });
                if sparsity_reached(baseline_params, current_params, constraints.target_sparsity) {
                    return finish_joint(baseline_profile, base_spec, &delta, current_profile, prune_log, candidates_evaluated, constraints);
                }
            }
        }
    }

    // ----- Phase 3 (NEW for Mode 3): layer drops -----
    let mut layer_scores: Vec<(u32, f64)> = importance
        .layers
        .iter()
        .filter(|l| !constraints.preserve_layers.contains(&l.layer))
        .map(|l| (l.layer, l.total_score))
        .collect();
    layer_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    for (layer, _) in layer_scores {
        // Never drop a layer twice.
        if delta.per_layer.iter().any(|d| d.layer == layer && d.drop_layer) {
            continue;
        }
        // Never drop to zero layers (apply_delta would produce n_layers=0 which downstream
        // code does not handle).
        let surviving = delta.per_layer.iter().filter(|d| !d.drop_layer).count();
        if surviving <= 1 {
            break;
        }
        let mut trial_delta = delta.clone();
        if let Some(d) = trial_delta.per_layer.iter_mut().find(|d| d.layer == layer) {
            d.drop_layer = true;
        }
        let trial_spec = apply_delta(base_spec, &trial_delta);
        if trial_spec.n_layers == 0 {
            // Defensive: should be unreachable given the `surviving <= 1` guard, but the
            // apply_delta + delta accounting can race in pathological cases. Skip.
            continue;
        }
        let trial_profile = match evaluate(&trial_spec, gpu) {
            Ok(p) => p,
            Err(_) => {
                prune_log.push(PruneStep {
                    kind: "layer".into(),
                    layer,
                    head: None,
                    accepted: false,
                    reason: "joint: pruned spec failed validation".into(),
                    new_param_bytes: current_profile.param_bytes,
                    new_latency_us: current_profile.estimated_latency_us,
                });
                continue;
            }
        };
        candidates_evaluated += 1;
        if !satisfies_constraints(&trial_profile, constraints) {
            prune_log.push(PruneStep {
                kind: "layer".into(),
                layer,
                head: None,
                accepted: false,
                reason: "joint: layer drop violates constraints".into(),
                new_param_bytes: current_profile.param_bytes,
                new_latency_us: current_profile.estimated_latency_us,
            });
            continue;
        }
        delta = trial_delta;
        current_profile = trial_profile;
        current_params = trial_spec.param_count();
        prune_log.push(PruneStep {
            kind: "layer".into(),
            layer,
            head: None,
            accepted: true,
            reason: "joint: dropped (least important)".into(),
            new_param_bytes: current_profile.param_bytes,
            new_latency_us: current_profile.estimated_latency_us,
        });
        if sparsity_reached(baseline_params, current_params, constraints.target_sparsity) {
            break;
        }
    }

    finish_joint(baseline_profile, base_spec, &delta, current_profile, prune_log, candidates_evaluated, constraints)
}

/// Read the effective FFN width at `layer` given the delta accumulated so far.
fn current_ff_for(base_spec: &ModelSpec, delta: &PruneDelta, layer: u32) -> u32 {
    delta
        .per_layer
        .iter()
        .find(|d| d.layer == layer)
        .and_then(|d| d.new_d_ff)
        .unwrap_or_else(|| base_spec.d_ff[layer as usize])
}

/// Candidate FFN widths to try below `current` (decreasing). Mirrors the policy used by
/// PR #227 — 75/50/37.5% of current, multiples of 64, strictly less than current.
fn ffn_candidate_widths(current: u32) -> Vec<u32> {
    let raw = [current * 3 / 4, current / 2, current * 3 / 8];
    let mut out: Vec<u32> = raw.iter().map(|&w| (w / 64).max(1) * 64).collect();
    out.retain(|&w| w < current);
    out.dedup();
    out
}

fn finish_joint(
    baseline_profile: CompilationProfile,
    base_spec: &ModelSpec,
    delta: &PruneDelta,
    current_profile: CompilationProfile,
    prune_log: Vec<PruneStep>,
    candidates_evaluated: u32,
    constraints: &Constraints,
) -> SearchOutcome {
    let final_spec = apply_delta(base_spec, delta);
    let baseline_params = base_spec.param_count();
    SearchOutcome {
        baseline: Some(baseline_profile),
        chosen: Some(EvaluatedCandidate {
            spec: final_spec.clone(),
            profile: current_profile.clone(),
            score: (baseline_params.saturating_sub(final_spec.param_count())) as f64,
            feasible: satisfies_constraints(&current_profile, constraints),
        }),
        ranked_candidates: Vec::new(),
        prune_log,
        candidates_evaluated,
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cep_importance::{HeadImportance, ImportanceTable, LayerImportance};
    use crate::cep_oracle::{Activation, ModelSpec, NormType};
    use crate::gpu_specs::{find_gpu, GPU_DATABASE};

    fn gpu() -> &'static GpuSpec {
        find_gpu("H100")
            .or_else(|| find_gpu("h100"))
            .unwrap_or(&GPU_DATABASE[0])
    }

    fn spec() -> ModelSpec {
        ModelSpec::uniform(256, 4, 8, 4, 32, 512, 8192)
    }

    fn fake_importance(spec: &ModelSpec) -> ImportanceTable {
        let mut heads = Vec::new();
        for (layer, &nh) in spec.n_heads.iter().enumerate() {
            for h in 0..nh {
                heads.push(HeadImportance {
                    layer: layer as u32,
                    head: h,
                    weight_magnitude: 1.0,
                    spectral_energy: 0.5,
                    roofline_slack: 1.0,
                    position_factor: 1.0,
                    // Give later heads in the last layer very low scores
                    // so the greedy search picks them first.
                    score: if layer as u32 == spec.n_layers - 1 { 0.1 } else { 1.0 },
                });
            }
        }
        let layers: Vec<LayerImportance> = (0..spec.n_layers)
            .map(|l| LayerImportance {
                layer: l,
                attention_score: 8.0,
                ffn_score: 1.0,
                total_score: 9.0,
            })
            .collect();
        ImportanceTable {
            heads,
            ffns: Vec::new(),
            layers,
        }
    }

    #[test]
    fn prune_greedy_reaches_sparsity_target() {
        let s = spec();
        let baseline = s.param_count();
        let constraints = Constraints {
            peak_memory_bytes: u64::MAX,
            latency_us: f64::MAX,
            target_sparsity: 0.02, // tiny target so a few heads suffice
            preserve_layers: Vec::new(),
        };
        let outcome = prune_greedy(
            &s,
            &fake_importance(&s),
            gpu(),
            &constraints,
            Granularity::HeadAndFfn,
        );
        let chosen = outcome.chosen.as_ref().unwrap();
        let saved = baseline as f64 - chosen.spec.param_count() as f64;
        assert!(saved > 0.0);
        assert!(chosen.feasible);
    }

    #[test]
    fn prune_greedy_respects_preserve_list() {
        let s = spec();
        let preserved_layer = 0;
        let constraints = Constraints {
            target_sparsity: 0.5, // aggressive, but layer 0 is protected
            preserve_layers: vec![preserved_layer],
            ..Default::default()
        };
        let outcome = prune_greedy(
            &s,
            &fake_importance(&s),
            gpu(),
            &constraints,
            Granularity::Head,
        );
        // No prune step should touch layer 0.
        for step in &outcome.prune_log {
            assert_ne!(step.layer, preserved_layer, "layer 0 was pruned");
        }
    }

    #[test]
    fn prune_greedy_rejects_constraint_violations() {
        let s = spec();
        let constraints = Constraints {
            latency_us: 0.0000001, // impossibly tight
            target_sparsity: 0.3,
            ..Default::default()
        };
        let outcome = prune_greedy(
            &s,
            &fake_importance(&s),
            gpu(),
            &constraints,
            Granularity::Head,
        );
        // Every prune step should be rejected; the baseline itself
        // violates the constraint but the chosen spec is still the
        // baseline (nothing improved it).
        let accepted: Vec<_> = outcome.prune_log.iter().filter(|s| s.accepted).collect();
        assert!(accepted.is_empty());
    }

    #[test]
    fn architecture_search_ranks_feasible_first() {
        use crate::cep_rewrite::SearchAxes;
        let axes = SearchAxes {
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
        };
        let outcome = architecture_search(
            &axes,
            gpu(),
            &Constraints::default(),
            NasObjective::ParamEfficiency,
        );
        assert!(outcome.candidates_evaluated > 0);
        // The first entry must be feasible (or none are).
        if let Some(first) = outcome.ranked_candidates.first() {
            assert!(
                first.feasible
                    || outcome.ranked_candidates.iter().all(|c| !c.feasible)
            );
        }
    }

    #[test]
    fn nas_min_latency_picks_shortest_latency_candidate() {
        use crate::cep_rewrite::SearchAxes;
        let axes = SearchAxes {
            d_model: vec![128, 256],
            n_layers: vec![2, 4],
            n_heads: vec![4],
            n_kv_heads: vec![2],
            d_ff: vec![256],
            activation: vec![Activation::SwiGlu],
            norm: vec![NormType::RmsNorm],
            vocab: 8192,
            head_dim: 32,
            max_seq: 512,
            batch: 1,
            dtype_bytes: 2,
        };
        let outcome = architecture_search(&axes, gpu(), &Constraints::default(), NasObjective::MinLatency);
        if outcome.ranked_candidates.len() >= 2 {
            let (first, second) = (
                &outcome.ranked_candidates[0],
                &outcome.ranked_candidates[1],
            );
            assert!(first.profile.estimated_latency_us <= second.profile.estimated_latency_us);
        }
    }

    #[test]
    fn empty_search_returns_empty_outcome() {
        use crate::cep_rewrite::SearchAxes;
        let axes = SearchAxes {
            d_model: Vec::new(),
            n_layers: vec![4],
            n_heads: vec![4],
            n_kv_heads: vec![2],
            d_ff: vec![256],
            activation: vec![Activation::SwiGlu],
            norm: vec![NormType::RmsNorm],
            vocab: 8192,
            head_dim: 32,
            max_seq: 512,
            batch: 1,
            dtype_bytes: 2,
        };
        let outcome = architecture_search(&axes, gpu(), &Constraints::default(), NasObjective::ParamEfficiency);
        assert_eq!(outcome.candidates_evaluated, 0);
        assert!(outcome.chosen.is_none());
    }

    /// Mode 3: layer-importance vector helper — layer N has score 100 - N so layer 0 is
    /// "most important" and the highest layer has lowest score (dropped first).
    fn descending_layer_importance(spec: &ModelSpec) -> ImportanceTable {
        let mut imp = fake_importance(spec);
        imp.layers = (0..spec.n_layers)
            .map(|l| LayerImportance {
                layer: l,
                attention_score: 0.0,
                ffn_score: 0.0,
                total_score: (100 - l as i32) as f64,
            })
            .collect();
        imp
    }

    #[test]
    fn joint_prune_search_accepts_layer_drops_after_head_ffn_pruning() {
        // Aggressive target — head + FFN passes alone can't reach it; layer drop must
        // engage. Layer importance is descending so the LAST layer drops first.
        let s = spec();
        let baseline = s.param_count();
        let constraints = Constraints {
            peak_memory_bytes: u64::MAX,
            latency_us: f64::MAX,
            target_sparsity: 0.5, // aggressive
            preserve_layers: Vec::new(),
        };
        let outcome = joint_prune_search(
            &s,
            &descending_layer_importance(&s),
            gpu(),
            &constraints,
            Granularity::HeadAndFfn,
        );
        let chosen = outcome.chosen.as_ref().unwrap();
        // Some layer drops must have been accepted to reach the aggressive target.
        let accepted_drops: Vec<_> = outcome
            .prune_log
            .iter()
            .filter(|s| s.accepted && s.kind == "layer")
            .collect();
        assert!(
            !accepted_drops.is_empty(),
            "expected at least one layer-drop accept, got prune_log: {:?}",
            outcome.prune_log
        );
        // Final spec must have fewer layers than baseline.
        assert!(
            chosen.spec.n_layers < s.n_layers,
            "expected fewer surviving layers; baseline={} chosen={}",
            s.n_layers,
            chosen.spec.n_layers
        );
        // Some parameter reduction must have occurred (the exact ratio depends on
        // embedding-vs-block size split; we only assert it's nonzero and that drops
        // contributed, since heads + FFN + layer drops compose).
        assert!(
            chosen.spec.param_count() < s.param_count(),
            "expected fewer params than baseline"
        );
    }

    #[test]
    fn joint_prune_search_respects_preserve_layers_for_drops() {
        let s = spec();
        let preserved_layer = s.n_layers - 1; // would otherwise be the first drop candidate
        let constraints = Constraints {
            peak_memory_bytes: u64::MAX,
            latency_us: f64::MAX,
            target_sparsity: 0.9, // force full sweep
            preserve_layers: vec![preserved_layer],
        };
        let outcome = joint_prune_search(
            &s,
            &descending_layer_importance(&s),
            gpu(),
            &constraints,
            Granularity::HeadAndFfn,
        );
        // No layer-drop step may target the preserved layer.
        for step in &outcome.prune_log {
            if step.kind == "layer" {
                assert_ne!(step.layer, preserved_layer, "preserved layer was dropped");
            }
        }
    }

    #[test]
    fn joint_prune_search_never_drops_below_one_layer() {
        let s = spec();
        let constraints = Constraints {
            peak_memory_bytes: u64::MAX,
            latency_us: f64::MAX,
            target_sparsity: 0.999, // unreachable
            preserve_layers: Vec::new(),
        };
        let outcome = joint_prune_search(
            &s,
            &descending_layer_importance(&s),
            gpu(),
            &constraints,
            Granularity::HeadAndFfn,
        );
        // At least one layer must survive.
        let chosen = outcome.chosen.as_ref().unwrap();
        assert!(chosen.spec.n_layers >= 1, "all layers dropped");
    }

    #[test]
    fn joint_prune_search_no_drops_when_baseline_already_feasible_and_within_target() {
        let s = spec();
        let constraints = Constraints {
            peak_memory_bytes: u64::MAX,
            latency_us: f64::MAX,
            target_sparsity: 0.0, // already met at baseline
            preserve_layers: Vec::new(),
        };
        let outcome = joint_prune_search(
            &s,
            &fake_importance(&s),
            gpu(),
            &constraints,
            Granularity::HeadAndFfn,
        );
        let chosen = outcome.chosen.as_ref().unwrap();
        // No accepted prune steps (target met at baseline).
        let accepted: Vec<_> = outcome.prune_log.iter().filter(|s| s.accepted).collect();
        assert!(
            accepted.is_empty(),
            "expected no accepts when target=0, got: {:?}",
            accepted
        );
        assert_eq!(chosen.spec.n_layers, s.n_layers);
    }

    #[test]
    fn joint_prune_search_layer_drops_rejected_by_tight_constraints() {
        let s = spec();
        // Tight latency constraint — every layer-drop trial should still pass it (drops
        // REDUCE latency), so this exercises the constraint-evaluation path on layer drops.
        // To exercise the REJECT path, use an *unreachably* low peak memory: the baseline
        // already fits within u64::MAX, but we set a memory cap below the baseline so
        // every candidate (including layer drops) is rejected — exactly what would happen
        // if a downstream consumer required a minimum residency footprint that the pruned
        // spec violates by being too small (the cost model still produces a profile).
        let baseline_profile = evaluate(&s, gpu()).unwrap();
        let constraints = Constraints {
            // Below baseline -> every step is rejected.
            peak_memory_bytes: baseline_profile.peak_memory_bytes / 2,
            latency_us: f64::MAX,
            target_sparsity: 0.5,
            preserve_layers: Vec::new(),
        };
        let outcome = joint_prune_search(
            &s,
            &descending_layer_importance(&s),
            gpu(),
            &constraints,
            Granularity::HeadAndFfn,
        );
        // chosen.feasible must be false (we couldn't satisfy the impossible constraint).
        let chosen = outcome.chosen.as_ref().unwrap();
        assert!(!chosen.feasible, "should be infeasible under impossible memory cap");
    }

    #[test]
    fn joint_prune_search_log_includes_layer_kind() {
        let s = spec();
        let constraints = Constraints {
            peak_memory_bytes: u64::MAX,
            latency_us: f64::MAX,
            target_sparsity: 0.4,
            preserve_layers: Vec::new(),
        };
        let outcome = joint_prune_search(
            &s,
            &descending_layer_importance(&s),
            gpu(),
            &constraints,
            Granularity::HeadAndFfn,
        );
        // "layer" entries must appear in the prune log (either accepted or rejected).
        let layer_steps: Vec<_> = outcome
            .prune_log
            .iter()
            .filter(|s| s.kind == "layer")
            .collect();
        assert!(!layer_steps.is_empty(), "expected layer-drop entries in prune_log");
    }

    #[test]
    fn prune_log_records_every_attempt() {
        let s = spec();
        let importance = fake_importance(&s);
        let total_heads: usize = s.n_heads.iter().map(|&h| h as usize).sum();
        let outcome = prune_greedy(
            &s,
            &importance,
            gpu(),
            &Constraints {
                target_sparsity: 1.0, // never reachable → search exhausts
                ..Default::default()
            },
            Granularity::Head,
        );
        // Every head (minus preserved) should appear in the log as
        // either accepted or rejected.
        assert!(outcome.prune_log.len() >= 1);
        assert!(outcome.prune_log.len() <= total_heads);
    }
}
