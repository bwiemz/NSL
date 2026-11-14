//! CEP — pruning and architecture-search driver, and its report.
//!
//! Ties the cost estimate, the importance scorer, the spec rewriter and the
//! search together behind three entry points:
//!
//!   * [`run_prune`] — prune heads and FFN width (`@cep_prune`, `--cep-prune`).
//!   * [`run_joint`] — the same, with whole-layer drops (`--cep-joint`).
//!   * [`run_search`] — rank architectures (`@cep_search`, `--cep-search`).
//!
//! The driver is deterministic: the same inputs give the same plan.

use std::path::Path;

use serde::Serialize;

use crate::cep_importance::{ImportanceTable, RooflineSlackTable, analyse_weight_map};
use crate::cep_oracle::{CompilationProfile, ModelSpec, evaluate};
use crate::cep_rewrite::{PruneDelta, SearchAxes};
use crate::cep_search::{
    Constraints, Granularity, NasObjective, SearchOutcome, architecture_search, joint_prune_search,
    prune_greedy,
};
use crate::gpu_specs::{GPU_DATABASE, GpuSpec, find_gpu};
use crate::pass_trace::{DeclineReason, PassDisposition};
use crate::weight_aware::WeightMap;

/// Input for [`run_prune`].
#[derive(Clone, Debug)]
pub struct CepPruneInput<'a> {
    pub spec: ModelSpec,
    pub weights: Option<&'a WeightMap>,
    pub target: &'a str,
    pub constraints: Constraints,
    pub granularity: Granularity,
    /// Per-layer roofline slack overrides (optional).
    pub roofline_slack: RooflineSlackTable,
}

/// Input for [`run_search`].
#[derive(Debug, Clone)]
pub struct CepSearchInput<'a> {
    pub axes: SearchAxes,
    pub target: &'a str,
    pub constraints: Constraints,
    pub objective: NasObjective,
}

/// Aggregate plan.
#[derive(Debug, Clone)]
pub struct CepPlan {
    pub mode: CepMode,
    pub target_gpu: String,
    pub outcome: SearchOutcome,
    pub importance: ImportanceTable,
    /// The limits and targets the search ran under.
    pub constraints: Constraints,
    /// Search mode: the ranking objective.
    pub objective: Option<NasObjective>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CepMode {
    Prune,
    Search,
    /// Heads, FFN width and whole-layer drops.
    Joint,
}

impl CepMode {
    /// The spelling written to the delta file.
    pub fn as_str(self) -> &'static str {
        match self {
            CepMode::Prune => "prune",
            CepMode::Search => "search",
            CepMode::Joint => "joint",
        }
    }
}

impl CepPlan {
    /// Fraction of parameters removed (prune modes).
    pub fn param_reduction(&self) -> f64 {
        let Some(baseline) = self.outcome.baseline.as_ref() else {
            return 0.0;
        };
        let Some(chosen) = self.outcome.chosen.as_ref() else {
            return 0.0;
        };
        let bp = baseline.param_bytes as f64;
        if bp <= 0.0 {
            return 0.0;
        }
        1.0 - (chosen.profile.param_bytes as f64 / bp)
    }

    /// Whether the plan did what was asked: prune modes met the sparsity
    /// target and both limits; search found at least one feasible candidate.
    pub fn succeeded(&self) -> bool {
        self.outcome.chosen.as_ref().is_some_and(|c| c.feasible)
    }

    /// One line per target or limit the plan missed; empty when it met all.
    pub fn failures(&self) -> Vec<String> {
        let mut out = Vec::new();
        let o = &self.outcome;
        match self.mode {
            CepMode::Search => {
                if o.chosen.is_none() {
                    out.push(format!(
                        "no candidate meets the constraints ({} evaluated)",
                        o.candidates_evaluated
                    ));
                }
            }
            CepMode::Prune | CepMode::Joint => {
                let Some(chosen) = o.chosen.as_ref() else {
                    out.push("the search produced no result".to_string());
                    return out;
                };
                if !o.sparsity_met {
                    out.push(format!(
                        "sparsity target missed: {:.1}% of parameters requested, {:.1}% removed, \
                         and nothing more can be removed (granularity, preserved layers and the \
                         one-group and FFN-width floors limit what may go)",
                        100.0 * o.target_sparsity,
                        100.0 * o.achieved_sparsity
                    ));
                }
                let c = &self.constraints;
                if chosen.profile.peak_memory_bytes > c.peak_memory_bytes {
                    out.push(format!(
                        "peak memory limit missed: {} bytes estimated, limit {}",
                        chosen.profile.peak_memory_bytes, c.peak_memory_bytes
                    ));
                }
                if chosen.profile.estimated_latency_us > c.latency_us {
                    out.push(format!(
                        "latency limit missed: {:.1}us per forward estimated, limit {:.1}us",
                        chosen.profile.estimated_latency_us, c.latency_us
                    ));
                }
            }
        }
        out
    }

    /// Render the report.
    pub fn render_report(&self) -> String {
        use std::fmt::Write as _;
        let mut s = String::new();
        match self.mode {
            CepMode::Prune => writeln!(s, "=== CEP Pruning Report ===").unwrap(),
            CepMode::Search => writeln!(s, "=== CEP Architecture Search Report ===").unwrap(),
            CepMode::Joint => writeln!(s, "=== CEP Joint Prune-Search Report ===").unwrap(),
        }
        writeln!(s, "Target: {}", self.target_gpu).unwrap();
        writeln!(s).unwrap();

        if matches!(self.mode, CepMode::Prune | CepMode::Joint) {
            let secs = self.outcome.wall_clock_us as f64 / 1e6;
            writeln!(
                s,
                "Search time: {:.2} seconds ({} cost estimates)",
                secs, self.outcome.candidates_evaluated
            )
            .unwrap();

            if let Some(chosen) = self.outcome.chosen.as_ref() {
                writeln!(s, "{}", assumptions_line(&chosen.spec)).unwrap();
            }
            if let Some(baseline) = self.outcome.baseline.as_ref() {
                writeln!(
                    s,
                    "Baseline: params={:.1}MB, peak={:.2}GB, latency={:.1}us/forward",
                    baseline.param_bytes as f64 / 1e6,
                    baseline.peak_memory_bytes as f64 / 1e9,
                    baseline.estimated_latency_us
                )
                .unwrap();
            }
            if let Some(chosen) = self.outcome.chosen.as_ref() {
                writeln!(
                    s,
                    "Chosen:   params={:.1}MB, peak={:.2}GB, latency={:.1}us/forward",
                    chosen.profile.param_bytes as f64 / 1e6,
                    chosen.profile.peak_memory_bytes as f64 / 1e9,
                    chosen.profile.estimated_latency_us,
                )
                .unwrap();
                writeln!(
                    s,
                    "Reduction: {:.1}% of params removed (target {:.1}%)",
                    100.0 * self.outcome.achieved_sparsity,
                    100.0 * self.outcome.target_sparsity
                )
                .unwrap();

                let failures = self.failures();
                if failures.is_empty() {
                    writeln!(s, "Sparsity target and constraints met.").unwrap();
                } else {
                    for f in &failures {
                        writeln!(s, "NOT MET: {f}").unwrap();
                    }
                }
                if self.constraints.uniform {
                    writeln!(
                        s,
                        "Uniform plan: every unpreserved layer is pruned in step."
                    )
                    .unwrap();
                }

                if let Some(baseline) = self.outcome.baseline.as_ref() {
                    writeln!(
                        s,
                        "Weights + nominal code size: {} -> {}",
                        format_bytes_si(baseline.binary_size_bytes),
                        format_bytes_si(chosen.profile.binary_size_bytes)
                    )
                    .unwrap();
                    writeln!(
                        s,
                        "Kernel launches per forward (fusion template): {} -> {}",
                        baseline.kernel_launches, chosen.profile.kernel_launches
                    )
                    .unwrap();
                }

                writeln!(s).unwrap();
                writeln!(s, "Per-layer post-prune shape:").unwrap();
                let dropped: Vec<u32> = self
                    .outcome
                    .delta
                    .as_ref()
                    .map(|d| {
                        d.per_layer
                            .iter()
                            .filter(|l| l.drop_layer)
                            .map(|l| l.layer)
                            .collect()
                    })
                    .unwrap_or_default();
                let mut survivors = chosen.spec.n_heads.iter().zip(chosen.spec.d_ff.iter());
                let n_base = chosen.spec.n_layers as usize + dropped.len();
                for l in 0..n_base as u32 {
                    if dropped.contains(&l) {
                        writeln!(s, "  Layer {l}: dropped").unwrap();
                    } else if let Some((nh, ff)) = survivors.next() {
                        writeln!(s, "  Layer {l}: heads={nh}, FFN={ff}").unwrap();
                    }
                }
            }

            writeln!(s).unwrap();
            writeln!(s, "Prune log ({} steps):", self.outcome.prune_log.len()).unwrap();
            for step in &self.outcome.prune_log {
                let what = match step.kind.as_str() {
                    "head" => format!("heads {:?} removed", step.heads),
                    "ffn" => format!("FFN narrowed to {}", step.new_d_ff.unwrap_or(0)),
                    _ => "layer dropped".to_string(),
                };
                writeln!(
                    s,
                    "  layer={} {what} - {} (params={:.1}MB, latency={:.1}us)",
                    step.layer,
                    step.reason,
                    step.new_param_bytes as f64 / 1e6,
                    step.new_latency_us
                )
                .unwrap();
            }
        } else {
            let ranked = &self.outcome.ranked_candidates;
            let n_feasible = ranked.iter().filter(|c| c.feasible).count();
            writeln!(
                s,
                "Search space: {} candidates, {} distinct",
                self.outcome.candidates_enumerated,
                ranked.len()
            )
            .unwrap();
            writeln!(s, "Feasible (constraints met): {} candidates", n_feasible).unwrap();
            let secs = self.outcome.wall_clock_us as f64 / 1e6;
            writeln!(s, "Search time: {:.2} seconds", secs).unwrap();
            if let Some(first) = ranked.first() {
                writeln!(s, "{}", assumptions_line(&first.spec)).unwrap();
            }
            if let Some(obj) = self.objective {
                writeln!(s, "Objective: {}", objective_str(obj)).unwrap();
                if let Some(caveat) = obj.caveat() {
                    writeln!(s, "Note: {caveat}").unwrap();
                }
            }
            writeln!(s).unwrap();

            if n_feasible == 0 {
                writeln!(s, "NOT MET: no candidate meets the constraints.").unwrap();
                writeln!(s, "Closest candidates (none is feasible):").unwrap();
            } else {
                writeln!(s, "Top {} architectures:", n_feasible.min(3)).unwrap();
            }
            let shown = if n_feasible == 0 {
                3
            } else {
                n_feasible.min(3)
            };
            for (i, cand) in ranked.iter().take(shown).enumerate() {
                writeln!(
                    s,
                    "  {}. d={}, L={}, H={}, KV={}, FFN={}, act={}, norm={} -> {}M params, {:.2}GB, {:.1}us/forward, util={:.2}{}",
                    i + 1,
                    cand.spec.d_model,
                    cand.spec.n_layers,
                    cand.spec.n_heads.first().copied().unwrap_or(0),
                    cand.spec.n_kv_heads.first().copied().unwrap_or(0),
                    cand.spec.d_ff.first().copied().unwrap_or(0),
                    cand.spec.activation.as_str(),
                    cand.spec.norm.as_str(),
                    format_params_si(cand.spec.param_count()),
                    cand.profile.peak_memory_bytes as f64 / 1e9,
                    cand.profile.estimated_latency_us,
                    cand.profile.roofline_utilization,
                    if cand.feasible { "" } else { "  (infeasible)" },
                )
                .unwrap();
                if !cand.also.is_empty() {
                    writeln!(s, "     same estimate: {}", cand.also.join("; ")).unwrap();
                }
            }
        }
        s
    }
}

/// The inputs every estimate in a report rests on.
pub fn assumptions_line(spec: &ModelSpec) -> String {
    format!(
        "Estimates are for one forward pass of batch {} x {} tokens, {}-byte values, {} output head.",
        spec.batch,
        spec.max_seq,
        spec.dtype_bytes,
        if spec.tied_lm_head { "tied" } else { "untied" }
    )
}

fn objective_str(o: NasObjective) -> &'static str {
    match o {
        NasObjective::ParamEfficiency => "param_efficiency",
        NasObjective::MinLatency => "min_latency",
        NasObjective::MinMemory => "min_memory",
        NasObjective::MinParams => "min_params",
    }
}

// ---------------------------------------------------------------------------
// SI formatting helpers
// ---------------------------------------------------------------------------

fn format_bytes_si(bytes: u64) -> String {
    if bytes >= 1_000_000_000 {
        format!("{:.1}GB", bytes as f64 / 1e9)
    } else if bytes >= 1_000_000 {
        format!("{}MB", bytes / 1_000_000)
    } else if bytes >= 1_000 {
        format!("{}KB", bytes / 1_000)
    } else {
        format!("{}B", bytes)
    }
}

/// Render param count as millions to one decimal place.  The render line uses
/// `"{}M params"` with a literal "M", so this MUST return only the magnitude
/// (no suffix).  Sub-1M counts render as "0.5M params" etc.
fn format_params_si(p: u64) -> String {
    format!("{:.1}", p as f64 / 1e6)
}

/// Roofline-slack multiplier when the caller supplies none: 0.8 for a layer
/// the estimate classes as compute-bound, 1.5 otherwise. The two constants
/// are placeholders; in a model whose layers share one shape they are the
/// same for every layer and change no ranking.
fn slack_from_profile(p: &CompilationProfile) -> RooflineSlackTable {
    let mut v = Vec::new();
    for layer in &p.per_layer {
        if layer.layer_index < u32::MAX / 2 {
            let slack = if layer.compute_bound { 0.8 } else { 1.5 };
            v.push((layer.layer_index, slack));
        }
    }
    RooflineSlackTable { per_layer: v }
}

fn resolve_gpu(target: &str) -> Result<&'static GpuSpec, String> {
    find_gpu(target).ok_or_else(|| {
        CepBridgeError::UnknownTarget {
            target: target.to_string(),
            supported: supported_gpus(),
        }
        .to_string()
    })
}

/// Shared body of [`run_prune`] and [`run_joint`].
fn run_prune_mode(input: CepPruneInput, mode: CepMode) -> Result<CepPlan, String> {
    crate::pass_trace::record("CEP");
    let gpu = resolve_gpu(input.target)?;
    let baseline =
        evaluate(&input.spec, gpu).map_err(|e| format!("CEP: invalid model spec: {e}"))?;

    let slacks = if input.roofline_slack.per_layer.is_empty() {
        slack_from_profile(&baseline)
    } else {
        input.roofline_slack
    };

    // Importance from the weights when there are any; otherwise a table in
    // which every component of a layer counts the same.
    let importance = match input.weights {
        Some(wm) => analyse_weight_map(wm, &input.spec, &slacks)?,
        None => synthetic_importance(&input.spec, &slacks),
    };

    let outcome = match mode {
        CepMode::Joint => joint_prune_search(
            &input.spec,
            &importance,
            gpu,
            &input.constraints,
            input.granularity,
        ),
        _ => prune_greedy(
            &input.spec,
            &importance,
            gpu,
            &input.constraints,
            input.granularity,
        ),
    };

    let disposition = prune_disposition(&outcome);
    crate::pass_trace::record_disposition("CEP", disposition);
    Ok(CepPlan {
        mode,
        target_gpu: gpu.name.to_string(),
        outcome,
        importance,
        constraints: input.constraints,
        objective: None,
    })
}

/// What a prune run reports to the pass trace. `rewrites` is the number of
/// pruning steps applied, and every log entry is one. An empty log is either
/// a request that needed nothing or a model with nothing the search is
/// allowed to remove.
fn prune_disposition(outcome: &SearchOutcome) -> PassDisposition {
    if !outcome.prune_log.is_empty() {
        PassDisposition::Applied {
            rewrites: outcome.prune_log.len(),
        }
    } else if outcome.sparsity_met && outcome.constraints_met {
        PassDisposition::Declined {
            reason: DeclineReason::NoCandidates("the model already meets the target and limits"),
        }
    } else {
        PassDisposition::Declined {
            reason: DeclineReason::NoCandidates("nothing may be removed at this granularity"),
        }
    }
}

/// Prune attention heads and FFN width. Errors when the target GPU is
/// unknown, the spec is invalid, or the weights do not match the spec.
pub fn run_prune(input: CepPruneInput) -> Result<CepPlan, String> {
    run_prune_mode(input, CepMode::Prune)
}

/// As [`run_prune`], with whole-layer drops as a further kind of step.
pub fn run_joint(input: CepPruneInput) -> Result<CepPlan, String> {
    run_prune_mode(input, CepMode::Joint)
}

/// Rank the architectures the search axes span.
pub fn run_search(input: CepSearchInput) -> Result<CepPlan, String> {
    crate::pass_trace::record("CEP");
    let gpu = resolve_gpu(input.target)?;
    let outcome = architecture_search(&input.axes, gpu, &input.constraints, input.objective);
    // Architecture search RECOMMENDS a spec; it rewrites nothing. Its flag
    // (`--cep-search`) lives on `nsl check` only, and the result is printed
    // for a human to act on.
    crate::pass_trace::record_disposition("CEP", PassDisposition::AdvisoryOnly);
    Ok(CepPlan {
        mode: CepMode::Search,
        target_gpu: gpu.name.to_string(),
        outcome,
        importance: ImportanceTable::default(),
        constraints: input.constraints,
        objective: Some(input.objective),
    })
}

/// Importance table for the weight-less case: every head of a layer scores
/// the same and no neuron magnitudes are recorded, so layers differ by the
/// position factor and the slack alone.
fn synthetic_importance(spec: &ModelSpec, slacks: &RooflineSlackTable) -> ImportanceTable {
    use crate::cep_importance::{FfnImportance, HeadImportance, LayerImportance};
    let mut heads = Vec::new();
    let mut ffns = Vec::new();
    let mut layers = Vec::new();
    for (l, &nh) in spec.n_heads.iter().enumerate() {
        let layer = l as u32;
        let slack = slacks.get(layer);
        let pos = crate::cep_importance::position_factor(layer, spec.n_layers);
        for h in 0..nh {
            heads.push(HeadImportance {
                layer,
                head: h,
                weight_magnitude: 1.0,
                spectral_energy: 1.0,
                roofline_slack: slack,
                position_factor: pos,
                score: slack * pos,
            });
        }
        ffns.push(FfnImportance {
            layer,
            weight_magnitude: 1.0,
            position_factor: pos,
            roofline_slack: slack,
            score: slack * pos,
            neuron_magnitudes: Vec::new(),
        });
        layers.push(LayerImportance {
            layer,
            attention_score: nh as f64 * slack * pos,
            ffn_score: slack * pos,
            total_score: (nh as f64 + 1.0) * slack * pos,
        });
    }
    ImportanceTable {
        heads,
        ffns,
        layers,
    }
}

// ---------------------------------------------------------------------------
// Delta-JSON writers — serialise a CepPlan to a versioned *.cep.json file.
// These DTOs pin the on-disk schema independently of the core layout.
// ---------------------------------------------------------------------------

const CEP_DELTA_VERSION: u32 = 2;

// NOTE: these emit the delta-JSON contract spelling (lowercase, no underscore),
// intentionally distinct from cep_oracle's `as_str()` ("rms_norm"). Do not merge.
fn activation_str(a: crate::cep_oracle::Activation) -> &'static str {
    use crate::cep_oracle::Activation::*;
    match a {
        Relu => "relu",
        Gelu => "gelu",
        SiLU => "silu",
        SwiGlu => "swiglu",
    }
}

fn norm_str(n: crate::cep_oracle::NormType) -> &'static str {
    use crate::cep_oracle::NormType::*;
    match n {
        LayerNorm => "layernorm",
        RmsNorm => "rmsnorm",
    }
}

#[derive(Serialize)]
struct SpecDto {
    d_model: u32,
    n_layers: u32,
    n_heads: Vec<u32>,
    n_kv_heads: Vec<u32>,
    /// Per-layer head dimensions (replaces the old scalar `head_dim`; schema v2+).
    head_dims: Vec<u32>,
    d_ff: Vec<u32>,
    activation: &'static str,
    norm: &'static str,
    vocab: u32,
}

impl SpecDto {
    fn from_spec(s: &ModelSpec) -> Self {
        SpecDto {
            d_model: s.d_model,
            n_layers: s.n_layers,
            n_heads: s.n_heads.clone(),
            n_kv_heads: s.n_kv_heads.clone(),
            head_dims: s.head_dim.clone(),
            d_ff: s.d_ff.clone(),
            activation: activation_str(s.activation),
            norm: norm_str(s.norm),
            vocab: s.vocab,
        }
    }
}

#[derive(Serialize)]
struct BaselineProfileDto {
    total_flops: u64,
    param_bytes: u64,
    peak_memory_bytes: u64,
    estimated_latency_us: f64,
}

#[derive(Serialize)]
struct ChosenProfileDto {
    param_bytes: u64,
    peak_memory_bytes: u64,
    estimated_latency_us: f64,
    roofline_utilization: f64,
}

#[derive(Serialize)]
struct LayerDeltaDto {
    layer: u32,
    pruned_heads: Vec<u32>,
    new_d_ff: Option<u32>,
    drop_layer: bool,
}

#[derive(Serialize)]
struct ChosenDto {
    spec: SpecDto,
    profile: ChosenProfileDto,
}

#[derive(Serialize)]
struct PruneDeltaBody {
    per_layer: Vec<LayerDeltaDto>,
}

#[derive(Serialize)]
struct SparsityDto {
    target: f64,
    achieved: f64,
    met: bool,
}

#[derive(Serialize)]
struct PruneDeltaDoc {
    cep_version: u32,
    mode: &'static str,
    target: String,
    /// The sparsity target and both limits were met.
    feasible: bool,
    sparsity: SparsityDto,
    constraints_met: bool,
    baseline_profile: BaselineProfileDto,
    chosen: ChosenDto,
    delta: PruneDeltaBody,
}

/// The plan's removals as a [`PruneDelta`] against `baseline`: whole
/// key/value groups of heads, FFN widths, and dropped layers, exactly as the
/// search chose them. The JSON writer, the weight slicer and the source
/// rewriter all consume this one value.
pub fn plan_to_prune_delta(plan: &CepPlan, baseline: &ModelSpec) -> PruneDelta {
    plan.outcome
        .delta
        .clone()
        .unwrap_or_else(|| PruneDelta::for_spec(baseline))
}

/// Serialize the prune delta.
pub fn write_prune_delta(plan: &CepPlan, baseline: &ModelSpec, path: &Path) -> std::io::Result<()> {
    crate::cep_weights::write_atomically(path, prune_delta_json(plan, baseline)?.as_bytes())
}

/// The delta file's contents.
pub fn prune_delta_json(plan: &CepPlan, baseline: &ModelSpec) -> std::io::Result<String> {
    let chosen = plan.outcome.chosen.as_ref();
    let chosen_spec = chosen.map(|c| &c.spec);

    let prune_delta = plan_to_prune_delta(plan, baseline);
    let per_layer: Vec<LayerDeltaDto> = prune_delta
        .per_layer
        .iter()
        .map(|ld| LayerDeltaDto {
            layer: ld.layer,
            pruned_heads: ld.pruned_heads.clone(),
            new_d_ff: ld.new_d_ff,
            drop_layer: ld.drop_layer,
        })
        .collect();

    let bp = plan.outcome.baseline.as_ref();
    let doc = PruneDeltaDoc {
        cep_version: CEP_DELTA_VERSION,
        mode: plan.mode.as_str(),
        target: plan.target_gpu.clone(),
        feasible: plan.succeeded(),
        sparsity: SparsityDto {
            target: plan.outcome.target_sparsity,
            achieved: plan.outcome.achieved_sparsity,
            met: plan.outcome.sparsity_met,
        },
        constraints_met: plan.outcome.constraints_met,
        baseline_profile: BaselineProfileDto {
            total_flops: bp.map(|p| p.total_flops).unwrap_or(0),
            param_bytes: bp.map(|p| p.param_bytes).unwrap_or(0),
            peak_memory_bytes: bp.map(|p| p.peak_memory_bytes).unwrap_or(0),
            estimated_latency_us: bp.map(|p| p.estimated_latency_us).unwrap_or(0.0),
        },
        chosen: ChosenDto {
            spec: SpecDto::from_spec(chosen_spec.unwrap_or(baseline)),
            profile: ChosenProfileDto {
                param_bytes: chosen.map(|c| c.profile.param_bytes).unwrap_or(0),
                peak_memory_bytes: chosen.map(|c| c.profile.peak_memory_bytes).unwrap_or(0),
                estimated_latency_us: chosen
                    .map(|c| c.profile.estimated_latency_us)
                    .unwrap_or(0.0),
                roofline_utilization: chosen
                    .map(|c| c.profile.roofline_utilization)
                    .unwrap_or(0.0),
            },
        },
        delta: PruneDeltaBody { per_layer },
    };
    Ok(serde_json::to_string_pretty(&doc)?)
}

#[derive(Serialize)]
struct CandidateDto {
    spec: SpecDto,
    profile: ChosenProfileDto,
    score: f64,
    feasible: bool,
}

#[derive(Serialize)]
struct SearchDoc {
    cep_version: u32,
    mode: &'static str,
    target: String,
    baseline_profile: Option<BaselineProfileDto>,
    ranked_candidates: Vec<CandidateDto>,
}

/// Serialize the search result (mode "search" with ranked_candidates).
pub fn write_search_delta(plan: &CepPlan, path: &Path) -> std::io::Result<()> {
    let ranked = plan
        .outcome
        .ranked_candidates
        .iter()
        .take(10)
        .map(|c| CandidateDto {
            spec: SpecDto::from_spec(&c.spec),
            profile: ChosenProfileDto {
                param_bytes: c.profile.param_bytes,
                peak_memory_bytes: c.profile.peak_memory_bytes,
                estimated_latency_us: c.profile.estimated_latency_us,
                roofline_utilization: c.profile.roofline_utilization,
            },
            score: c.score,
            feasible: c.feasible,
        })
        .collect();
    let doc = SearchDoc {
        cep_version: CEP_DELTA_VERSION,
        mode: "search",
        target: plan.target_gpu.clone(),
        baseline_profile: plan.outcome.baseline.as_ref().map(|p| BaselineProfileDto {
            total_flops: p.total_flops,
            param_bytes: p.param_bytes,
            peak_memory_bytes: p.peak_memory_bytes,
            estimated_latency_us: p.estimated_latency_us,
        }),
        ranked_candidates: ranked,
    };
    let json = serde_json::to_string_pretty(&doc)?;
    std::fs::write(path, json)
}

// ---------------------------------------------------------------------------
// Config bridge — semantic config + CLI overrides → driver input structs
// ---------------------------------------------------------------------------

/// CLI scalar overrides; decorator-primary, these win when present.
#[derive(Debug, Clone, Default)]
pub struct CliOverrides {
    pub target: Option<String>,
    pub sparsity: Option<f64>,
    pub cep_out: Option<std::path::PathBuf>,
    /// CEP SP1: also emit the pruned weights to this .safetensors path.
    pub cep_emit_weights: Option<std::path::PathBuf>,
    /// CEP SP2: also emit the rewritten NSL source with pruned dims to this path.
    pub cep_emit_source: Option<std::path::PathBuf>,
}

#[derive(Debug)]
pub enum CepBridgeError {
    UnknownTarget { target: String, supported: String },
    BadPreserve(String),
    PreserveLayerOutOfRange { entry: String, n_layers: u32 },
}

impl std::fmt::Display for CepBridgeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CepBridgeError::UnknownTarget { target, supported } => {
                write!(
                    f,
                    "CEP: unknown --cep-target '{target}'. Supported GPUs: {supported}"
                )
            }
            CepBridgeError::BadPreserve(s) => write!(
                f,
                "CEP: preserve entry '{s}' is not recognized; use a layer index (e.g. '7') or 'blocks.<n>' / 'blocks.<n>.*' pattern (e.g. 'blocks.0.*')"
            ),
            CepBridgeError::PreserveLayerOutOfRange { entry, n_layers } => write!(
                f,
                "CEP: preserve entry '{entry}' references layer >= n_layers ({n_layers})"
            ),
        }
    }
}

fn supported_gpus() -> String {
    GPU_DATABASE
        .iter()
        .map(|g| g.name)
        .collect::<Vec<_>>()
        .join(", ")
}

/// Public alias used by the CLI to report supported GPU names.
pub fn supported_gpus_list() -> String {
    supported_gpus()
}

fn parse_preserve_layers(preserve: &[String], n_layers: u32) -> Result<Vec<u32>, CepBridgeError> {
    let mut out: Vec<u32> = Vec::with_capacity(preserve.len());
    for p in preserve {
        // Bare integer: "0", "7", etc.
        if let Ok(n) = p.parse::<u32>() {
            if n >= n_layers {
                return Err(CepBridgeError::PreserveLayerOutOfRange {
                    entry: p.clone(),
                    n_layers,
                });
            }
            out.push(n);
            continue;
        }
        // "blocks.<N>" or "blocks.<N>.<suffix>"
        if let Some(rest) = p.strip_prefix("blocks.") {
            let head = rest.split('.').next().unwrap_or("");
            // Reject "blocks.*" or "blocks.*.X": cannot resolve to a specific layer.
            if head == "*" || head.is_empty() {
                return Err(CepBridgeError::BadPreserve(p.clone()));
            }
            if let Ok(n) = head.parse::<u32>() {
                if n >= n_layers {
                    return Err(CepBridgeError::PreserveLayerOutOfRange {
                        entry: p.clone(),
                        n_layers,
                    });
                }
                out.push(n);
                continue;
            }
        }
        return Err(CepBridgeError::BadPreserve(p.clone()));
    }
    out.sort_unstable();
    out.dedup();
    Ok(out)
}

fn map_granularity(g: nsl_semantic::cep::Granularity) -> Granularity {
    match g {
        nsl_semantic::cep::Granularity::Head => Granularity::Head,
        nsl_semantic::cep::Granularity::Ffn => Granularity::Ffn,
        nsl_semantic::cep::Granularity::HeadAndFfn => Granularity::HeadAndFfn,
    }
}

fn map_objective(o: nsl_semantic::cep::NasObjective) -> NasObjective {
    match o {
        nsl_semantic::cep::NasObjective::ParamEfficiency => NasObjective::ParamEfficiency,
        nsl_semantic::cep::NasObjective::MinLatency => NasObjective::MinLatency,
        nsl_semantic::cep::NasObjective::MinMemory => NasObjective::MinMemory,
        nsl_semantic::cep::NasObjective::MinParams => NasObjective::MinParams,
    }
}

/// Build `CepPruneInput` from semantic config + extracted spec + overrides.
/// `target` is the pre-resolved effective target string, owned by the caller.
pub fn build_prune_input<'a>(
    cfg: &nsl_semantic::cep::CepPruneConfig,
    spec: ModelSpec,
    weights: Option<&'a WeightMap>,
    target: &'a str,
    sparsity_override: Option<f64>,
) -> Result<CepPruneInput<'a>, CepBridgeError> {
    if find_gpu(target).is_none() {
        return Err(CepBridgeError::UnknownTarget {
            target: target.to_string(),
            supported: supported_gpus(),
        });
    }
    let target_sparsity = sparsity_override.unwrap_or(cfg.sparsity);
    let preserve_layers = parse_preserve_layers(&cfg.preserve, spec.n_layers)?;
    let constraints = Constraints {
        peak_memory_bytes: cfg.constraints.peak_memory_bytes.unwrap_or(u64::MAX),
        latency_us: cfg.constraints.latency_us.unwrap_or(f64::INFINITY),
        target_sparsity,
        preserve_layers,
        uniform: false,
    };
    Ok(CepPruneInput {
        spec,
        weights,
        target,
        constraints,
        granularity: map_granularity(cfg.granularity),
        roofline_slack: RooflineSlackTable {
            per_layer: Vec::new(),
        },
    })
}

/// Build `CepSearchInput` from semantic config + extracted axes + target.
pub fn build_search_input<'a>(
    cfg: &nsl_semantic::cep::CepSearchConfig,
    axes: SearchAxes,
    target: &'a str,
) -> Result<CepSearchInput<'a>, CepBridgeError> {
    if find_gpu(target).is_none() {
        return Err(CepBridgeError::UnknownTarget {
            target: target.to_string(),
            supported: supported_gpus(),
        });
    }
    let constraints = Constraints {
        peak_memory_bytes: cfg.constraints.peak_memory_bytes.unwrap_or(u64::MAX),
        latency_us: cfg.constraints.latency_us.unwrap_or(f64::INFINITY),
        target_sparsity: 0.0,
        preserve_layers: Vec::new(),
        uniform: false,
    };
    Ok(CepSearchInput {
        axes,
        target,
        constraints,
        objective: map_objective(cfg.objective),
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cep_oracle::{Activation, ModelSpec, NormType};
    use crate::cep_rewrite::SearchAxes;

    /// d=256, 4 layers, 8 query heads in 4 key/value groups, head_dim 32,
    /// d_ff 512, vocab 8192, tied head: 4,458,752 parameters, of which one
    /// key/value group or one FFN step (64 neurons) is 49,152. The arithmetic
    /// is written out beside `cep_search`'s tests.
    fn tiny_spec() -> ModelSpec {
        ModelSpec::uniform(256, 4, 8, 4, 32, 512, 8192)
    }

    fn prune_input(sparsity: f64, granularity: Granularity) -> CepPruneInput<'static> {
        CepPruneInput {
            spec: tiny_spec(),
            weights: None,
            target: "H100",
            constraints: Constraints {
                target_sparsity: sparsity,
                ..Default::default()
            },
            granularity,
            roofline_slack: RooflineSlackTable::default(),
        }
    }

    /// 5% of 4,458,752 is 222,938 parameters: five steps of 49,152.
    #[test]
    fn run_prune_without_weights_still_produces_plan() {
        let plan = run_prune(prune_input(0.05, Granularity::HeadAndFfn)).unwrap();
        assert_eq!(plan.mode, CepMode::Prune);
        assert_eq!(plan.outcome.prune_log.len(), 5);
        assert!(plan.succeeded());
        let report = plan.render_report();
        assert!(report.contains("CEP Pruning Report"));
        assert!(
            report.contains("Sparsity target and constraints met."),
            "{report}"
        );
        assert!(!report.contains("NOT MET"), "{report}");
    }

    /// 1% is 44,588 parameters: one group, 49,152 / 4,458,752 = 1.1024%.
    #[test]
    fn run_prune_reports_the_reduction_it_made() {
        let plan = run_prune(prune_input(0.01, Granularity::Head)).unwrap();
        assert!((plan.param_reduction() - 49_152.0 / 4_458_752.0).abs() < 1e-12);
        assert!(
            plan.render_report()
                .contains("Reduction: 1.1% of params removed (target 1.0%)")
        );
    }

    /// Heads alone stop at 13.2% of this model (see `cep_search`). A plan that
    /// misses its target must say so and must not count as a success.
    #[test]
    fn a_missed_target_is_not_reported_as_success() {
        let plan = run_prune(prune_input(0.9, Granularity::Head)).unwrap();
        assert!(!plan.succeeded());
        let failures = plan.failures();
        assert_eq!(failures.len(), 1);
        assert!(
            failures[0].contains("90.0% of parameters requested, 13.2% removed"),
            "{failures:?}"
        );
        let report = plan.render_report();
        assert!(
            report.contains("NOT MET: sparsity target missed"),
            "{report}"
        );
        assert!(!report.contains("constraints met"), "{report}");
    }

    #[test]
    fn a_missed_limit_names_the_limit() {
        let mut input = prune_input(0.0, Granularity::HeadAndFfn);
        input.constraints.latency_us = 1.0; // launch overhead alone is far more
        let plan = run_prune(input).unwrap();
        assert!(!plan.succeeded());
        let failures = plan.failures();
        assert_eq!(failures.len(), 1);
        assert!(
            failures[0].starts_with("latency limit missed"),
            "{failures:?}"
        );
        assert!(failures[0].contains("limit 1.0us"), "{failures:?}");
    }

    #[test]
    fn an_unknown_target_is_an_error_not_a_default() {
        let mut input = prune_input(0.05, Granularity::Head);
        input.target = "NoSuchGPU";
        let err = run_prune(input).unwrap_err();
        assert!(err.contains("NoSuchGPU"), "{err}");
    }

    /// The pass record counts the steps that were applied.
    #[test]
    fn pass_record_counts_applied_steps() {
        let plan = run_prune(prune_input(0.05, Granularity::HeadAndFfn)).unwrap();
        assert_eq!(
            prune_disposition(&plan.outcome),
            PassDisposition::Applied { rewrites: 5 }
        );

        let nothing_needed = run_prune(prune_input(0.0, Granularity::Head)).unwrap();
        assert!(matches!(
            prune_disposition(&nothing_needed.outcome),
            PassDisposition::Declined { reason: DeclineReason::NoCandidates(m) } if m.contains("already meets")
        ));

        // One key/value head: head pruning has nothing it may remove.
        let mut stuck = prune_input(0.3, Granularity::Head);
        stuck.spec = ModelSpec::uniform(256, 4, 8, 1, 32, 512, 8192);
        let stuck = run_prune(stuck).unwrap();
        assert!(stuck.outcome.prune_log.is_empty() && !stuck.succeeded());
        assert!(matches!(
            prune_disposition(&stuck.outcome),
            PassDisposition::Declined { reason: DeclineReason::NoCandidates(m) } if m.contains("nothing may be removed")
        ));
    }

    fn search_axes() -> SearchAxes {
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

    #[test]
    fn run_search_produces_ranked_candidates() {
        let plan = run_search(CepSearchInput {
            axes: search_axes(),
            target: "H100",
            constraints: Constraints::default(),
            objective: NasObjective::MinLatency,
        })
        .unwrap();
        assert_eq!(plan.mode, CepMode::Search);
        assert_eq!(plan.outcome.candidates_evaluated, 32);
        assert!(plan.succeeded());
        let report = plan.render_report();
        assert!(report.contains("CEP Architecture Search Report"));
        assert!(
            report.contains("Search space: 32 candidates, 32 distinct"),
            "{report}"
        );
        assert!(report.contains("Top 3 architectures:"), "{report}");
        // The smallest of every axis is the fastest, so it is row 1.
        assert!(
            report.contains("  1. d=128, L=2, H=4, KV=2, FFN=256, act=swiglu, norm=rms_norm"),
            "{report}"
        );
        assert!(!report.contains("infeasible"), "{report}");
        assert!(
            !report.contains("Note:"),
            "min_latency carries no caveat\n{report}"
        );
    }

    /// With a limit nothing meets there is no architecture to recommend. The
    /// report must say so, and every row it shows must be marked.
    #[test]
    fn a_search_with_no_feasible_candidate_says_so() {
        let plan = run_search(CepSearchInput {
            axes: search_axes(),
            target: "H100",
            constraints: Constraints {
                peak_memory_bytes: 1,
                ..Default::default()
            },
            objective: NasObjective::MinLatency,
        })
        .unwrap();
        assert!(!plan.succeeded());
        assert_eq!(plan.failures().len(), 1);
        let report = plan.render_report();
        assert!(
            report.contains("Feasible (constraints met): 0 candidates"),
            "{report}"
        );
        assert!(
            report.contains("NOT MET: no candidate meets the constraints."),
            "{report}"
        );
        assert!(!report.contains("Top 3 architectures"), "{report}");
        let rows: Vec<&str> = report.lines().filter(|l| l.contains(" d=")).collect();
        assert_eq!(rows.len(), 3);
        assert!(rows.iter().all(|r| r.ends_with("(infeasible)")), "{rows:?}");
    }

    #[test]
    fn param_efficiency_is_reported_with_its_caveat() {
        let plan = run_search(CepSearchInput {
            axes: search_axes(),
            target: "H100",
            constraints: Constraints::default(),
            objective: NasObjective::ParamEfficiency,
        })
        .unwrap();
        let report = plan.render_report();
        assert!(report.contains("Objective: param_efficiency"), "{report}");
        assert!(report.contains("Note: param_efficiency"), "{report}");
        assert!(report.contains("no task-quality term"), "{report}");
    }

    #[test]
    fn plan_is_deterministic_across_runs() {
        // The measured wall-clock line depends on the clock, not on the plan.
        let strip = |s: &str| -> String {
            s.lines()
                .filter(|l| !l.starts_with("Search time:"))
                .collect::<Vec<_>>()
                .join("\n")
        };
        let r1 = run_prune(prune_input(0.05, Granularity::Head))
            .unwrap()
            .render_report();
        let r2 = run_prune(prune_input(0.05, Granularity::Head))
            .unwrap()
            .render_report();
        assert_eq!(strip(&r1), strip(&r2));
    }

    #[test]
    fn synthetic_importance_covers_every_head() {
        let spec = tiny_spec();
        let slacks = RooflineSlackTable::default();
        let table = synthetic_importance(&spec, &slacks);
        let expected: u32 = spec.n_heads.iter().sum();
        assert_eq!(table.heads.len() as u32, expected);
        assert_eq!(table.ffns.len() as u32, spec.n_layers);
    }

    // format_params_si must always return millions-only (no suffix), so the render
    // line "{}M params" composes correctly for sub-1M model candidates.
    #[test]
    fn format_params_si_handles_sub_1m() {
        assert_eq!(format_params_si(500_000), "0.5");
        assert_eq!(format_params_si(100), "0.0");
        assert_eq!(format_params_si(48_800_000), "48.8");
        assert_eq!(format_params_si(1_000_000), "1.0");
        assert_eq!(format_params_si(7_000_000_000), "7000.0");
    }

    #[test]
    fn param_reduction_returns_zero_without_baseline() {
        let plan = CepPlan {
            mode: CepMode::Search,
            target_gpu: "test".into(),
            outcome: SearchOutcome::default(),
            importance: ImportanceTable::default(),
            constraints: Constraints::default(),
            objective: None,
        };
        assert_eq!(plan.param_reduction(), 0.0);
    }

    /// The shipped 50M-parameter shape (`models/coder50m/model.nsl`):
    /// 48,767,488 parameters, of which the tied embedding is 25,165,824.
    /// A key/value group (2 of 8 query heads) is 2·512·128 + 2·512·64 =
    /// 196,608 parameters and an FFN step (128 of 1408 neurons) is
    /// 3·512·128 = 196,608.
    ///
    /// A 30% request needs 14,630,247 parameters gone: 75 steps, 30.24%.
    /// The most head and FFN pruning can remove is 3 groups and 7 FFN steps
    /// (1408 -> 512) from each of 8 layers, 80 steps, 32.25%: a 50% request
    /// cannot be met, because half the model is the embedding.
    #[test]
    fn nslcoder_50m_budget_adherence() {
        let spec = ModelSpec::uniform(512, 8, 8, 4, 64, 1408, 49152);
        assert_eq!(spec.param_count(), 48_767_488);
        let run = |sparsity: f64| {
            run_prune(CepPruneInput {
                spec: spec.clone(),
                weights: None,
                target: "H100",
                constraints: Constraints {
                    target_sparsity: sparsity,
                    ..Default::default()
                },
                granularity: Granularity::HeadAndFfn,
                roofline_slack: RooflineSlackTable::default(),
            })
            .unwrap()
        };

        let p30 = run(0.3);
        assert!(p30.succeeded());
        assert_eq!(p30.outcome.prune_log.len(), 75);
        let chosen = p30.outcome.chosen.as_ref().unwrap();
        assert_eq!(chosen.spec.param_count(), 48_767_488 - 75 * 196_608);
        assert!((p30.outcome.achieved_sparsity - 14_745_600.0 / 48_767_488.0).abs() < 1e-12);

        let p50 = run(0.5);
        assert!(!p50.succeeded());
        assert!(!p50.outcome.sparsity_met);
        assert_eq!(p50.outcome.prune_log.len(), 80);
        let chosen = p50.outcome.chosen.as_ref().unwrap();
        assert_eq!(chosen.spec.n_heads, vec![2; 8]);
        assert_eq!(chosen.spec.d_ff, vec![512; 8]);
        assert_eq!(chosen.spec.param_count(), 48_767_488 - 80 * 196_608);
        // Head and FFN pruning change kernel sizes, not how many are launched.
        assert_eq!(
            chosen.profile.kernel_launches,
            p50.outcome.baseline.as_ref().unwrap().kernel_launches
        );
    }
}

#[cfg(test)]
mod bridge_tests {
    use super::*;
    use crate::cep_oracle::{Activation, ModelSpec, NormType};

    fn spec() -> ModelSpec {
        ModelSpec {
            d_model: 384,
            n_layers: 6,
            n_heads: vec![6; 6],
            n_kv_heads: vec![3; 6],
            head_dim: vec![64; 6],
            d_ff: vec![1024; 6],
            vocab: 4096,
            max_seq: 2048,
            batch: 1,
            activation: Activation::SwiGlu,
            norm: NormType::RmsNorm,
            dtype_bytes: 4,
            tied_lm_head: true,
        }
    }

    fn prune_cfg() -> nsl_semantic::cep::CepPruneConfig {
        nsl_semantic::cep::CepPruneConfig {
            target: None,
            sparsity: 0.3,
            granularity: nsl_semantic::cep::Granularity::HeadAndFfn,
            preserve: vec!["0".to_string(), "5".to_string()],
            constraints: nsl_semantic::cep::CepConstraints {
                peak_memory_bytes: Some(6_000_000_000),
                latency_us: Some(3000.0),
            },
            span: nsl_errors::Span::DUMMY,
        }
    }

    #[test]
    fn build_prune_input_maps_constraints_and_overrides() {
        let cfg = prune_cfg();
        let input = build_prune_input(&cfg, spec(), None, "H100-SXM", Some(0.5)).expect("input");
        assert_eq!(input.target, "H100-SXM");
        assert_eq!(input.constraints.peak_memory_bytes, 6_000_000_000);
        assert_eq!(input.constraints.latency_us, 3000.0);
        assert_eq!(input.constraints.target_sparsity, 0.5); // override beats cfg.sparsity (0.3)
        assert_eq!(input.constraints.preserve_layers, vec![0, 5]);
        assert!(matches!(input.granularity, Granularity::HeadAndFfn));
    }

    #[test]
    fn build_prune_input_unknown_target_errors() {
        let cfg = prune_cfg();
        let err = build_prune_input(&cfg, spec(), None, "NoSuchGPU", None).unwrap_err();
        match err {
            CepBridgeError::UnknownTarget { target, supported } => {
                assert_eq!(target, "NoSuchGPU");
                assert!(supported.contains("H100-SXM"));
            }
            other => panic!("expected UnknownTarget, got {other:?}"),
        }
    }

    #[test]
    fn build_prune_input_bad_preserve_errors() {
        let mut cfg = prune_cfg();
        cfg.preserve = vec!["foo".to_string()];
        let err = build_prune_input(&cfg, spec(), None, "H100-SXM", None).unwrap_err();
        assert!(matches!(err, CepBridgeError::BadPreserve(_)));
    }

    #[test]
    fn build_prune_input_accepts_glob_preserve_blocks_n_star() {
        let baseline = spec(); // n_layers = 6
        let mut cfg = prune_cfg();
        // Paper §6.1 syntax:
        cfg.preserve = vec!["blocks.0.*".to_string(), "blocks.5.*".to_string()];
        let input =
            build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", None).expect("input");
        assert_eq!(input.constraints.preserve_layers, vec![0, 5]);
    }

    #[test]
    fn build_prune_input_accepts_blocks_n_without_star() {
        let baseline = spec();
        let mut cfg = prune_cfg();
        cfg.preserve = vec!["blocks.1".to_string(), "blocks.3.attn".to_string()];
        let input =
            build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", None).expect("input");
        assert_eq!(input.constraints.preserve_layers, vec![1, 3]);
    }

    #[test]
    fn build_prune_input_rejects_blocks_star() {
        let baseline = spec();
        let mut cfg = prune_cfg();
        cfg.preserve = vec!["blocks.*".to_string()];
        let err = build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", None).unwrap_err();
        assert!(matches!(err, CepBridgeError::BadPreserve(_)));
    }

    #[test]
    fn build_prune_input_rejects_out_of_range_layer() {
        let baseline = spec(); // n_layers = 6
        let mut cfg = prune_cfg();
        cfg.preserve = vec!["blocks.99".to_string()];
        let err = build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", None).unwrap_err();
        assert!(
            matches!(err, CepBridgeError::PreserveLayerOutOfRange { .. }),
            "expected PreserveLayerOutOfRange, got {err:?}"
        );
    }

    use std::io::Read;

    #[test]
    fn write_prune_delta_emits_versioned_json() {
        let baseline = spec();
        let cfg = prune_cfg();
        let input =
            build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", Some(0.2)).expect("input");
        let plan = run_prune(input).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.cep.json");
        write_prune_delta(&plan, &baseline, &path).expect("write");

        let mut s = String::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_string(&mut s)
            .unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["cep_version"], 2);
        assert_eq!(v["mode"], "prune");
        assert_eq!(v["target"], "H100-SXM");
        assert!(v["baseline_profile"]["param_bytes"].is_number());
        assert!(v["chosen"]["spec"]["d_model"].is_number());
        assert!(v["chosen"]["spec"]["activation"].is_string());
        assert!(v["delta"]["per_layer"].is_array());
        assert_eq!(
            v["delta"]["per_layer"].as_array().unwrap().len(),
            baseline.n_layers as usize
        );
        // The plan met a 20% target; the file says so and by how much.
        assert_eq!(v["feasible"], true);
        assert_eq!(v["sparsity"]["met"], true);
        assert_eq!(v["sparsity"]["target"], 0.2);
        assert!(v["sparsity"]["achieved"].as_f64().unwrap() >= 0.2);
        assert_eq!(v["constraints_met"], true);
    }

    /// A joint plan's file says it is one, and carries the dropped layers.
    #[test]
    fn joint_delta_json_is_labelled_joint() {
        let baseline = spec();
        let mut cfg = prune_cfg();
        cfg.preserve = Vec::new();
        cfg.constraints.peak_memory_bytes = None;
        cfg.constraints.latency_us = None;
        let input =
            build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", Some(0.5)).expect("input");
        let plan = run_joint(input).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&prune_delta_json(&plan, &baseline).unwrap()).unwrap();
        assert_eq!(v["mode"], "joint");
        let dropped = v["delta"]["per_layer"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|l| l["drop_layer"] == true)
            .count();
        assert_eq!(
            dropped as u32,
            baseline.n_layers - plan.outcome.chosen.as_ref().unwrap().spec.n_layers
        );
        assert!(dropped > 0);
    }

    #[test]
    fn plan_to_prune_delta_is_group_aligned_and_consistent() {
        // baseline: 6 heads, 3 kv -> group size 2. Prune to 4 heads / 2 kv (drop 1 group).
        let baseline = spec(); // d_model=384, n_layers=6, n_heads=[6;6], n_kv_heads=[3;6], head_dim=[64;6], d_ff=[1024;6]
        let cfg = prune_cfg();
        // 30% of 11,309,952 is more than the four unpreserved layers' FFNs can
        // give (4 x 5 steps x 147,456 = 2,949,120), so head groups must go too.
        let input =
            build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", Some(0.3)).expect("input");
        let plan = run_prune(input).unwrap();
        let delta = plan_to_prune_delta(&plan, &baseline);

        // Non-vacuous: the chosen plan must actually prune at least one layer, or the
        // group-alignment/contiguity assertions below would pass trivially on empty deltas.
        assert!(
            delta.per_layer.iter().any(|ld| !ld.pruned_heads.is_empty()),
            "expected at least one pruned layer"
        );

        let chosen = plan.outcome.chosen.as_ref().expect("chosen");
        for ld in &delta.per_layer {
            let l = ld.layer as usize;
            let group = baseline.n_heads[l] / baseline.n_kv_heads[l]; // 2
            // pruned_heads must be a whole number of groups
            assert_eq!(
                ld.pruned_heads.len() % group as usize,
                0,
                "layer {l}: not group-aligned"
            );
            // survivor count must equal the chosen spec exactly
            assert_eq!(
                baseline.n_heads[l] - ld.pruned_heads.len() as u32,
                chosen.spec.n_heads[l],
                "layer {l}: survivor count != chosen.n_heads"
            );
            // each pruned group is contiguous head indices [g*group, (g+1)*group)
            let mut sorted = ld.pruned_heads.clone();
            sorted.sort_unstable();
            for chunk in sorted.chunks(group as usize) {
                let g0 = chunk[0];
                assert_eq!(g0 % group, 0, "layer {l}: group start not aligned");
                for (i, &h) in chunk.iter().enumerate() {
                    assert_eq!(h, g0 + i as u32, "layer {l}: group not contiguous");
                }
            }
        }
    }

    /// Joint mode (Mode 3) regression — PR #232's Bug 2.
    ///
    /// `apply_delta` (cep_rewrite.rs::~120-124) compacts dropped layers out of every
    /// per-layer vector in the chosen spec, so `chosen.spec.n_heads[k]` is the k-th
    /// SURVIVING layer's head count.  Before this fix, `plan_to_prune_delta` indexed
    /// `chosen.spec.n_heads[l]` with the raw baseline layer index, producing:
    ///
    ///   - Wrong head/FFN deltas for every baseline layer following a drop
    ///     (off-by-N shift after the first dropped layer).
    ///   - `drop_layer: false` on every entry, even when joint search accepted drops
    ///     — so SP1 (`--cep-emit-weights`) and the delta JSON output
    ///     (`--cep-out`) lost the layer-drop signal entirely.
    ///
    /// This test forces joint search to drop at least one layer (aggressive sparsity
    /// + descending layer importance so the LAST layer drops first), then asserts:
    ///
    ///   1. Every dropped baseline layer surfaces with `drop_layer: true`.
    ///   2. No surviving layer is mis-flagged with `drop_layer: true`.
    ///   3. Head/FFN deltas on surviving baseline layers AFTER a drop resolve to the
    ///      correct compacted-chosen-index head/FFN counts (the off-by-N case).
    #[test]
    fn plan_to_prune_delta_marks_dropped_layers_in_joint_mode() {
        // Aggressive sparsity target — head + FFN pruning alone can't reach 0.5,
        // so joint search must engage layer drops.  Empty preserve list so the
        // last layer (least important under synthetic position-factor scoring)
        // gets dropped first.
        let baseline = spec();
        let mut cfg = prune_cfg();
        cfg.sparsity = 0.5;
        cfg.preserve = Vec::new();
        cfg.granularity = nsl_semantic::cep::Granularity::HeadAndFfn;
        // Relax constraints so joint search can actually reach the target — the
        // default cfg's peak_memory_bytes / latency_us are tight.
        cfg.constraints.peak_memory_bytes = None;
        cfg.constraints.latency_us = None;
        let input =
            build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", Some(0.5)).expect("input");
        let plan = run_joint(input).unwrap();

        // Force the test to fail loudly if the planner doesn't actually drop a layer
        // — otherwise (1)+(2) below would pass trivially on the empty drop set.
        let accepted_drops: std::collections::HashSet<u32> = plan
            .outcome
            .prune_log
            .iter()
            .filter(|s| s.kind == "layer")
            .map(|s| s.layer)
            .collect();
        assert!(
            !accepted_drops.is_empty(),
            "expected at least one accepted layer drop in joint mode at sparsity=0.5; \
             prune_log = {:?}",
            plan.outcome.prune_log
        );

        let delta = plan_to_prune_delta(&plan, &baseline);

        // (1) Every accepted-dropped baseline layer must surface as drop_layer:true.
        let delta_drops: std::collections::HashSet<u32> = delta
            .per_layer
            .iter()
            .filter(|ld| ld.drop_layer)
            .map(|ld| ld.layer)
            .collect();
        assert_eq!(
            delta_drops, accepted_drops,
            "delta drop-set must match the prune_log accepted-drops exactly. \
             delta_drops={delta_drops:?}, accepted_drops={accepted_drops:?}"
        );

        // (2) No surviving layer mis-flagged with drop_layer:true.
        for ld in &delta.per_layer {
            if accepted_drops.contains(&ld.layer) {
                continue;
            }
            assert!(
                !ld.drop_layer,
                "layer {} survives in chosen spec but delta says drop_layer:true",
                ld.layer
            );
        }

        // (3) Surviving-layer head/FFN deltas resolve via the COMPACTED index.
        //     For each surviving baseline layer, walk a parallel compacted-index
        //     counter; the survivor head count from the delta must match
        //     chosen.spec.n_heads[compacted_idx] — the off-by-N case in Bug 2.
        let chosen = plan
            .outcome
            .chosen
            .as_ref()
            .expect("joint produced no chosen spec");
        let mut compacted_idx = 0usize;
        for ld in &delta.per_layer {
            let l = ld.layer as usize;
            if ld.drop_layer {
                continue;
            }
            let surviving_heads = baseline.n_heads[l] - ld.pruned_heads.len() as u32;
            assert_eq!(
                surviving_heads,
                chosen.spec.n_heads[compacted_idx],
                "layer {} (compacted index {}): survivor count from delta = {}, \
                 chosen.spec.n_heads[{}] = {} — off-by-N bug surfacing again",
                l,
                compacted_idx,
                surviving_heads,
                compacted_idx,
                chosen.spec.n_heads[compacted_idx]
            );
            // FFN check: new_d_ff must either be None (no shrink) OR match
            // chosen.spec.d_ff at the compacted index — NOT at the raw
            // baseline index l (which is what Bug 2 produced).
            if let Some(new_d_ff) = ld.new_d_ff {
                assert_eq!(
                    new_d_ff, chosen.spec.d_ff[compacted_idx],
                    "layer {} (compacted index {}): new_d_ff = {} != chosen.spec.d_ff[{}] = {}",
                    l, compacted_idx, new_d_ff, compacted_idx, chosen.spec.d_ff[compacted_idx]
                );
            }
            compacted_idx += 1;
        }
        // Total surviving layers in delta must equal the chosen spec's layer count.
        assert_eq!(
            compacted_idx as u32, chosen.spec.n_layers,
            "delta surviving-layer count ({}) != chosen.spec.n_layers ({})",
            compacted_idx, chosen.spec.n_layers
        );
    }

    #[test]
    fn delta_json_preserves_per_layer_head_dim() {
        // Build a spec with HETEROGENEOUS head_dim per layer and confirm the serialized
        // DTO carries them all as an array (not just the first scalar value).
        let mut baseline = spec(); // n_layers=6
        baseline.head_dim = vec![64, 64, 64, 64, 32, 32]; // mix two values
        let cfg = prune_cfg();
        let input =
            build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", Some(0.2)).expect("input");
        let plan = run_prune(input).unwrap();

        let dir = tempfile::tempdir().unwrap();
        let out = dir.path().join("delta.cep.json");
        write_prune_delta(&plan, &baseline, &out).unwrap();
        let txt = std::fs::read_to_string(&out).unwrap();
        let v: serde_json::Value = serde_json::from_str(&txt).unwrap();

        // Schema v2: field is now `head_dims` (Vec<u32>), not a scalar `head_dim`.
        let dims = v
            .pointer("/chosen/spec/head_dims")
            .expect("chosen.spec.head_dims must exist in schema v2");
        assert!(dims.is_array(), "head_dims must serialize as a JSON array");
        let dims_u32: Vec<u32> = dims
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap() as u32)
            .collect();
        assert_eq!(dims_u32, vec![64, 64, 64, 64, 32, 32]);
    }

    #[test]
    fn write_search_delta_emits_ranked_candidates() {
        use crate::cep_rewrite::SearchAxes;
        let axes = SearchAxes {
            d_model: vec![256, 384],
            n_layers: vec![4],
            n_heads: vec![4, 8],
            n_kv_heads: vec![2],
            d_ff: vec![512, 1024],
            activation: vec![crate::cep_oracle::Activation::SwiGlu],
            norm: vec![crate::cep_oracle::NormType::RmsNorm],
            vocab: 4096,
            head_dim: 64,
            max_seq: 2048,
            batch: 1,
            dtype_bytes: 4,
            tied_lm_head: true,
        };
        let scfg = nsl_semantic::cep::CepSearchConfig {
            target: None,
            objective: nsl_semantic::cep::NasObjective::ParamEfficiency,
            constraints: Default::default(),
            span: nsl_errors::Span::DUMMY,
        };
        let input = build_search_input(&scfg, axes, "H100-SXM").expect("input");
        let plan = run_search(input).unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.cep.json");
        write_search_delta(&plan, &path).expect("write");
        let s = std::fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["mode"], "search");
        assert!(v["ranked_candidates"].is_array());
    }

    #[test]
    fn prune_report_names_what_its_figures_are() {
        let baseline = spec();
        let cfg = prune_cfg();
        let input =
            build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", Some(0.2)).expect("input");
        let plan = run_prune(input).unwrap();
        let rendered = plan.render_report();
        for line in [
            "Search time:",
            "Estimates are for one forward pass of batch 1 x 2048 tokens, 4-byte values, tied output head.",
            "us/forward",
            "Sparsity target and constraints met.",
            "Weights + nominal code size:",
            "Kernel launches per forward (fusion template): 50 -> 50",
        ] {
            assert!(rendered.contains(line), "missing '{line}'\n{rendered}");
        }
        // Layers 0 and 5 are preserved in `prune_cfg`, and say so by their shape.
        assert!(
            rendered.contains("  Layer 0: heads=6, FFN=1024"),
            "{rendered}"
        );
        assert!(
            rendered.contains("  Layer 5: heads=6, FFN=1024"),
            "{rendered}"
        );
    }
}
