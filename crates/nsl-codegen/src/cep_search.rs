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
    /// Number of candidates evaluated (feasibility-filtered for prune; total
    /// enumerated BEFORE feasibility filtering for architecture_search).
    pub candidates_evaluated: u32,
    /// Total candidates enumerated BEFORE feasibility or constraint filtering.
    /// For prune mode this equals candidates_evaluated (every trial is logged).
    /// For search mode this is the full cross-product count before filtering.
    pub candidates_enumerated: u32,
    /// Wall-clock duration of the search inside prune_greedy / architecture_search
    /// (microseconds). Includes all candidate compilations. Set to 0 in Default.
    pub wall_clock_us: u64,
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

/// Candidate FFN widths for a layer's sweep. Derived from the layer's baseline
/// `d_ff` so all real models (d_ff ∈ {1024, 4096, 8192, ...}) get a meaningful
/// sweep. All candidates are multiples of 64 (paper §4.3 table) and strictly
/// less than `baseline_d_ff`. Returned largest-first (try largest reduction first).
fn ffn_candidate_widths(baseline_d_ff: u32) -> Vec<u32> {
    let multipliers = [0.75_f32, 0.5, 0.375];
    let mut widths: Vec<u32> = multipliers
        .iter()
        .map(|m| ((baseline_d_ff as f32 * m) as u32 / 64) * 64)
        .filter(|&w| w > 0 && w < baseline_d_ff)
        .collect();
    widths.sort_unstable();
    widths.dedup();
    widths.into_iter().rev().collect() // try largest-shrink first
}

/// Retained capacity: the sum of importance scores of components that survived
/// pruning. Paper §2.2 Mode 1: "Maximize: retained capacity (sum of importance
/// scores)".
///
/// - Per surviving head h in layer l: add the head's importance score.
/// - Per layer l: add (chosen.d_ff[l] / baseline.d_ff[l]) × ffn.score
///   (linear retention proxy).
/// - Per surviving layer l: add layers[l].total_score.
fn retained_capacity(baseline: &ModelSpec, chosen: &ModelSpec, importance: &ImportanceTable) -> f64 {
    let mut total = 0.0_f64;
    // Head retention: surviving heads have indices [0, chosen.n_heads[l]).
    for h in &importance.heads {
        let l = h.layer as usize;
        if l < chosen.n_layers as usize && h.head < chosen.n_heads.get(l).copied().unwrap_or(0) {
            total += h.score;
        }
    }
    // FFN width retention.
    for f in &importance.ffns {
        let l = f.layer as usize;
        if l < chosen.n_layers as usize {
            let base = baseline.d_ff.get(l).copied().unwrap_or(0) as f64;
            let new = chosen.d_ff.get(l).copied().unwrap_or(0) as f64;
            if base > 0.0 {
                total += f.score * (new / base).clamp(0.0, 1.0);
            }
        }
    }
    // Layer retention: surviving layers contribute their total_score.
    for lscore in &importance.layers {
        let l = lscore.layer as usize;
        if l < chosen.n_layers as usize {
            total += lscore.total_score;
        }
    }
    total
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
    let t0 = std::time::Instant::now();
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
        // Iterate from least-important layer (low importance score).
        let mut layer_scores: Vec<(u32, f64)> = importance
            .layers
            .iter()
            .filter(|l| !constraints.preserve_layers.contains(&l.layer))
            .map(|l| (l.layer, l.total_score))
            .collect();
        layer_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        for (layer, _) in layer_scores {
            let candidate_widths = ffn_candidate_widths(base_spec.d_ff[layer as usize]);
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

    let wall_clock_us = t0.elapsed().as_micros() as u64;
    SearchOutcome {
        baseline: Some(baseline_profile.clone()),
        chosen: Some(EvaluatedCandidate {
            spec: current_spec.clone(),
            profile: current_profile.clone(),
            score: retained_capacity(base_spec, &current_spec, importance),
            feasible: satisfies_constraints(&current_profile, constraints),
        }),
        ranked_candidates: Vec::new(),
        prune_log,
        candidates_evaluated,
        candidates_enumerated: candidates_evaluated,
        wall_clock_us,
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
    let t0 = std::time::Instant::now();
    let mut enumerated_total = 0u32;
    let mut evaluated = Vec::new();
    for spec in axes.enumerate() {
        enumerated_total += 1;
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
    let wall_clock_us = t0.elapsed().as_micros() as u64;
    let candidates_evaluated = evaluated.len() as u32;
    let chosen = evaluated.iter().find(|c| c.feasible).cloned();
    SearchOutcome {
        baseline: None,
        chosen,
        ranked_candidates: evaluated,
        prune_log: Vec::new(),
        candidates_evaluated,
        candidates_enumerated: enumerated_total,
        wall_clock_us,
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

    // G4 — FFN widths must be data-driven, not hardcoded.
    #[test]
    fn ffn_candidate_widths_scale_to_baseline() {
        // baseline d_ff = 4096 → widths {3072, 2048, 1536} (75%, 50%, 37.5% × mult-64)
        let w = super::ffn_candidate_widths(4096);
        assert!(!w.is_empty());
        assert!(
            w.iter().all(|&x| x % 64 == 0),
            "must be multiples of 64: {w:?}"
        );
        assert!(
            w.iter().all(|&x| x < 4096),
            "must be strictly less than baseline"
        );
        assert_eq!(w[0], 3072); // largest-first
        // baseline d_ff = 1024 → widths {768, 512, 384}
        let w2 = super::ffn_candidate_widths(1024);
        assert_eq!(w2, vec![768, 512, 384]);
    }

    // G5 — prune chosen score must be retained capacity, not params-saved.
    #[test]
    fn prune_chosen_score_is_retained_capacity() {
        use crate::cep_importance::FfnImportance;

        let s = spec();
        let baseline_params = s.param_count() as f64;
        // Build an importance table that includes ffns so retained_capacity is non-zero.
        let mut imp = fake_importance(&s);
        for l in 0..s.n_layers {
            imp.ffns.push(FfnImportance {
                layer: l,
                weight_magnitude: 1.0,
                position_factor: 1.0,
                roofline_slack: 1.0,
                score: 10.0,
            });
        }
        let outcome = prune_greedy(
            &s,
            &imp,
            gpu(),
            &Constraints {
                target_sparsity: 0.02,
                ..Default::default()
            },
            Granularity::HeadAndFfn,
        );
        let chosen = outcome.chosen.as_ref().unwrap();
        // Retained capacity sums importance scores (typically small numbers
        // relative to raw param counts).  It must NOT look like "params saved"
        // (which would be in the millions for this spec).
        assert!(
            chosen.score < baseline_params * 0.5,
            "retained_capacity {} should not look like saved-params {}",
            chosen.score,
            baseline_params
        );
        // And strictly positive (some capacity is retained).
        assert!(chosen.score > 0.0, "retained_capacity must be > 0");
    }
}
