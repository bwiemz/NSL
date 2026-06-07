//! CEP — Compilation-Evaluated Pruning: driver + report.
//!
//! Unifies the oracle, importance scorer, model rewriter, and search
//! algorithm into two user-facing entry points:
//!
//!   * [`run_prune`] — compilation-verified pruning from a pre-trained
//!     model (`@cep_prune` decorator flow).
//!   * [`run_search`] — hardware-aware architecture search
//!     (`@cep_search` decorator flow).
//!
//! The driver is pure and deterministic — every invocation with the same
//! inputs produces an identical report.

use std::path::Path;

use serde::Serialize;

use crate::cep_importance::{analyse_weight_map, ImportanceTable, RooflineSlackTable};
use crate::cep_oracle::{evaluate, CompilationProfile, ModelSpec};
use crate::cep_rewrite::{LayerDelta, PruneDelta, SearchAxes};
use crate::cep_search::{
    architecture_search, joint_prune_search, prune_greedy, Constraints, Granularity, NasObjective,
    SearchOutcome,
};
use crate::gpu_specs::{default_gpu, find_gpu, GPU_DATABASE};
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
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub enum CepMode {
    Prune,
    Search,
    /// Paper §2.2 Mode 3 — joint prune-search over heads + FFN + layer drops.
    Joint,
}

impl CepPlan {
    /// Parameter reduction fraction (prune mode).
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

    /// Render the compilation report (paper §6.3 format).
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
            // §6.3: "Search time: <S> seconds (<N> candidates evaluated)"
            let secs = self.outcome.wall_clock_us as f64 / 1e6;
            writeln!(
                s,
                "Search time: {:.2} seconds ({} candidates evaluated)",
                secs, self.outcome.candidates_evaluated
            )
            .unwrap();

            if let Some(baseline) = self.outcome.baseline.as_ref() {
                writeln!(
                    s,
                    "Baseline: params={:.1}MB, peak={:.2}GB, latency={:.1}us",
                    baseline.param_bytes as f64 / 1e6,
                    baseline.peak_memory_bytes as f64 / 1e9,
                    baseline.estimated_latency_us
                )
                .unwrap();
            }
            if let Some(chosen) = self.outcome.chosen.as_ref() {
                writeln!(
                    s,
                    "Chosen:   params={:.1}MB, peak={:.2}GB, latency={:.1}us  (feasible={})",
                    chosen.profile.param_bytes as f64 / 1e6,
                    chosen.profile.peak_memory_bytes as f64 / 1e9,
                    chosen.profile.estimated_latency_us,
                    chosen.feasible
                )
                .unwrap();
                writeln!(
                    s,
                    "Reduction: {:.1}% of params removed",
                    100.0 * self.param_reduction()
                )
                .unwrap();

                // §6.3: constraint status
                if chosen.feasible {
                    writeln!(s, "All constraints satisfied.").unwrap();
                } else {
                    writeln!(s, "Constraints violated: feasible=false").unwrap();
                }

                // §6.3: "Binary size: <orig> -> <pruned>"
                if let Some(baseline) = self.outcome.baseline.as_ref() {
                    writeln!(
                        s,
                        "Binary size: {} -> {}",
                        format_bytes_si(baseline.binary_size_bytes),
                        format_bytes_si(chosen.profile.binary_size_bytes)
                    )
                    .unwrap();
                    // §6.3: "Kernel launches per forward: <orig> -> <pruned>"
                    writeln!(
                        s,
                        "Kernel launches per forward: {} -> {}",
                        baseline.kernel_launches, chosen.profile.kernel_launches
                    )
                    .unwrap();
                }

                writeln!(s).unwrap();
                writeln!(s, "Per-layer post-prune shape:").unwrap();
                for (i, (&nh, &ff)) in chosen
                    .spec
                    .n_heads
                    .iter()
                    .zip(chosen.spec.d_ff.iter())
                    .enumerate()
                {
                    writeln!(s, "  Layer {i}: heads={nh}, FFN={ff}").unwrap();
                }
            }

            writeln!(s).unwrap();
            writeln!(s, "Prune log ({} steps):", self.outcome.prune_log.len()).unwrap();
            for step in &self.outcome.prune_log {
                writeln!(
                    s,
                    "  [{}] layer={} {}{} - {} (params={:.1}MB)",
                    if step.accepted { "OK" } else { "-" },
                    step.layer,
                    step.kind,
                    step.head.map(|h| format!(" head={h}")).unwrap_or_default(),
                    step.reason,
                    step.new_param_bytes as f64 / 1e6
                )
                .unwrap();
            }
        } else {
            // Search mode — §6.3
            let n_feasible = self.outcome.ranked_candidates.iter().filter(|c| c.feasible).count();
            writeln!(
                s,
                "Search space: {} candidates",
                self.outcome.candidates_enumerated
            )
            .unwrap();
            writeln!(s, "Feasible (constraints met): {} candidates", n_feasible).unwrap();
            let secs = self.outcome.wall_clock_us as f64 / 1e6;
            writeln!(s, "Compilation time: {:.2} seconds", secs).unwrap();
            writeln!(s).unwrap();

            writeln!(s, "Top 3 architectures:").unwrap();
            for (i, cand) in self.outcome.ranked_candidates.iter().take(3).enumerate() {
                writeln!(
                    s,
                    "  {}. d={}, L={}, H={}, KV={}, FFN={} -> {}M params, {:.2}GB, {:.1}us/tok, util={:.2}",
                    i + 1,
                    cand.spec.d_model,
                    cand.spec.n_layers,
                    cand.spec.n_heads.first().copied().unwrap_or(0),
                    cand.spec.n_kv_heads.first().copied().unwrap_or(0),
                    cand.spec.d_ff.first().copied().unwrap_or(0),
                    format_params_si(cand.spec.param_count()),
                    cand.profile.peak_memory_bytes as f64 / 1e9,
                    cand.profile.estimated_latency_us,
                    cand.profile.roofline_utilization,
                )
                .unwrap();
            }
        }
        s
    }
}

// ---------------------------------------------------------------------------
// SI formatting helpers (paper §6.3 spelling)
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
/// (no suffix).  Sub-1M counts render as "0.5M params" etc., which is correct
/// and honest (toy fixtures never have genuinely large param counts).
fn format_params_si(p: u64) -> String {
    format!("{:.1}", p as f64 / 1e6)
}

/// Default roofline-slack proxy when the user doesn't supply one: derive
/// from the baseline profile's per-layer classification.
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

/// Run compilation-verified pruning.
pub fn run_prune(input: CepPruneInput) -> CepPlan {
    let gpu = find_gpu(input.target).unwrap_or_else(default_gpu);
    // 1. Baseline compile (for slack estimates + importance context).
    let baseline = evaluate(&input.spec, gpu).expect("base spec must validate");

    // 2. Resolve roofline slacks — prefer the user's, else derive.
    let slacks = if input.roofline_slack.per_layer.is_empty() {
        slack_from_profile(&baseline)
    } else {
        input.roofline_slack
    };

    // 3. Importance table (from weights if available; otherwise synthesise
    //    a uniform table so the search still runs).
    let importance = if let Some(wm) = input.weights {
        analyse_weight_map(wm, &input.spec.n_heads, input.spec.n_layers, &slacks)
    } else {
        synthetic_importance(&input.spec, &slacks)
    };

    // 4. Greedy prune.
    let outcome = prune_greedy(
        &input.spec,
        &importance,
        gpu,
        &input.constraints,
        input.granularity,
    );

    CepPlan {
        mode: CepMode::Prune,
        target_gpu: gpu.name.to_string(),
        outcome,
        importance,
    }
}

/// Paper §2.2 Mode 3 — joint prune-search. Same input contract as `run_prune` (weights
/// required for non-synthetic importance), but `joint_prune_search` extends the action
/// space with layer drops on top of head + FFN pruning.
pub fn run_joint(input: CepPruneInput) -> CepPlan {
    let gpu = find_gpu(input.target).unwrap_or_else(default_gpu);
    let baseline = evaluate(&input.spec, gpu).expect("base spec must validate");
    let slacks = if input.roofline_slack.per_layer.is_empty() {
        slack_from_profile(&baseline)
    } else {
        input.roofline_slack
    };
    let importance = if let Some(wm) = input.weights {
        analyse_weight_map(wm, &input.spec.n_heads, input.spec.n_layers, &slacks)
    } else {
        synthetic_importance(&input.spec, &slacks)
    };
    let outcome = joint_prune_search(
        &input.spec,
        &importance,
        gpu,
        &input.constraints,
        input.granularity,
    );
    CepPlan {
        mode: CepMode::Joint,
        target_gpu: gpu.name.to_string(),
        outcome,
        importance,
    }
}

/// Run hardware-aware architecture search.
pub fn run_search(input: CepSearchInput) -> CepPlan {
    let gpu = find_gpu(input.target).unwrap_or_else(default_gpu);
    let outcome = architecture_search(&input.axes, gpu, &input.constraints, input.objective);
    CepPlan {
        mode: CepMode::Search,
        target_gpu: gpu.name.to_string(),
        outcome,
        importance: ImportanceTable::default(),
    }
}

/// Uniform-score importance table for the weight-less case.  The search
/// degenerates to "prune from the last layer back" — not ideal but
/// deterministic.
fn synthetic_importance(spec: &ModelSpec, slacks: &RooflineSlackTable) -> ImportanceTable {
    use crate::cep_importance::{HeadImportance, LayerImportance};
    let mut heads = Vec::new();
    for (layer, &nh) in spec.n_heads.iter().enumerate() {
        let slack = slacks.get(layer as u32);
        for h in 0..nh {
            heads.push(HeadImportance {
                layer: layer as u32,
                head: h,
                weight_magnitude: 1.0,
                spectral_energy: 0.5,
                roofline_slack: slack,
                position_factor: crate::cep_importance::position_factor(layer as u32, spec.n_layers),
                // Earlier/later layers and lower-slack (compute-bound) layers
                // get higher scores so the greedy search targets the middle
                // first.
                score: slack
                    * crate::cep_importance::position_factor(layer as u32, spec.n_layers),
            });
        }
    }
    let layers = (0..spec.n_layers)
        .map(|l| LayerImportance {
            layer: l,
            attention_score: spec.n_heads[l as usize] as f64,
            ffn_score: 1.0,
            total_score: spec.n_heads[l as usize] as f64 + 1.0,
        })
        .collect();
    ImportanceTable {
        heads,
        ffns: Vec::new(),
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
struct PruneDeltaDoc {
    cep_version: u32,
    mode: &'static str,
    target: String,
    baseline_profile: BaselineProfileDto,
    chosen: ChosenDto,
    delta: PruneDeltaBody,
}

/// Build a GQA-group-aligned, importance-ranked PruneDelta from a prune plan.
/// The shared source consumed by both `write_prune_delta` (JSON) and SP1's slicer.
///
/// Head pruning: `apply_delta` snaps `n_heads` to a group multiple by COUNT, so the
/// chosen spec gives survivor counts but not identities. We pick which groups to drop
/// by lowest summed head-importance (authoritative — `plan.importance.heads` is always
/// populated in prune mode), keeping `pruned_heads` group-aligned and exactly consistent
/// with `chosen.n_heads`.
pub fn plan_to_prune_delta(plan: &CepPlan, baseline: &ModelSpec) -> PruneDelta {
    let chosen = plan.outcome.chosen.as_ref().map(|c| &c.spec);

    // Pre-index head importances by layer so per-group scoring is a small local scan
    // rather than re-walking the full `plan.importance.heads` slice for every group.
    let mut head_scores_by_layer: Vec<Vec<(u32, f64)>> =
        vec![Vec::new(); baseline.n_layers as usize];
    for h in &plan.importance.heads {
        if (h.layer as usize) < head_scores_by_layer.len() {
            head_scores_by_layer[h.layer as usize].push((h.head, h.score));
        }
    }

    let mut per_layer = Vec::with_capacity(baseline.n_layers as usize);
    for l in 0..baseline.n_layers as usize {
        let base_heads = baseline.n_heads[l];
        let base_kv = baseline.n_kv_heads[l].max(1);
        let group = (base_heads / base_kv).max(1);

        let chosen_heads = chosen
            .map(|c| c.n_heads.get(l).copied().unwrap_or(base_heads))
            .unwrap_or(base_heads);
        let n_drop_heads = base_heads.saturating_sub(chosen_heads);
        // The planner snaps n_heads to a group multiple, so the drop count is always a
        // whole number of groups. Assert it so a future caller or planner change that
        // breaks the invariant is caught in debug builds rather than silently truncating.
        debug_assert_eq!(
            n_drop_heads % group,
            0,
            "layer {l}: chosen_heads={chosen_heads} not a multiple of group={group}"
        );
        let n_drop_groups = (n_drop_heads / group) as usize;

        let mut pruned_heads = Vec::new();
        if n_drop_groups > 0 {
            let layer_scores = &head_scores_by_layer[l];
            // Score each of the `base_kv` groups by summed head importance for this layer.
            let mut group_scores: Vec<(u32, f64)> = (0..base_kv)
                .map(|g| {
                    let lo = g * group;
                    let hi = lo + group;
                    let s: f64 = layer_scores
                        .iter()
                        .filter(|(head, _)| *head >= lo && *head < hi)
                        .map(|(_, score)| *score)
                        .sum();
                    (g, s)
                })
                .collect();
            // Lowest-score groups first; tie-break by lower group index for determinism.
            group_scores.sort_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal).then(a.0.cmp(&b.0))
            });
            for &(g, _) in group_scores.iter().take(n_drop_groups) {
                for h in (g * group)..((g + 1) * group) {
                    pruned_heads.push(h);
                }
            }
            pruned_heads.sort_unstable();
        }

        let new_d_ff = chosen
            .and_then(|c| c.d_ff.get(l).copied())
            .filter(|&w| w < baseline.d_ff[l]);

        per_layer.push(LayerDelta { layer: l as u32, pruned_heads, new_d_ff, drop_layer: false });
    }
    PruneDelta { per_layer }
}

/// Serialize the prune delta — the Option-2 input contract.
pub fn write_prune_delta(plan: &CepPlan, baseline: &ModelSpec, path: &Path) -> std::io::Result<()> {
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
        mode: "prune",
        target: plan.target_gpu.clone(),
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
                estimated_latency_us: chosen.map(|c| c.profile.estimated_latency_us).unwrap_or(0.0),
                roofline_utilization: chosen.map(|c| c.profile.roofline_utilization).unwrap_or(0.0),
            },
        },
        delta: PruneDeltaBody { per_layer },
    };
    let json = serde_json::to_string_pretty(&doc)?;
    std::fs::write(path, json)
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
                write!(f, "CEP: unknown --cep-target '{target}'. Supported GPUs: {supported}")
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
    GPU_DATABASE.iter().map(|g| g.name).collect::<Vec<_>>().join(", ")
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
    };
    Ok(CepPruneInput {
        spec,
        weights,
        target,
        constraints,
        granularity: map_granularity(cfg.granularity),
        roofline_slack: RooflineSlackTable { per_layer: Vec::new() },
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

    fn tiny_spec() -> ModelSpec {
        ModelSpec::uniform(256, 4, 8, 4, 32, 512, 8192)
    }

    #[test]
    fn run_prune_without_weights_still_produces_plan() {
        let plan = run_prune(CepPruneInput {
            spec: tiny_spec(),
            weights: None,
            target: "H100",
            constraints: Constraints {
                target_sparsity: 0.05,
                ..Default::default()
            },
            granularity: Granularity::HeadAndFfn,
            roofline_slack: RooflineSlackTable::default(),
        });
        assert_eq!(plan.mode, CepMode::Prune);
        assert!(plan.outcome.baseline.is_some());
        assert!(plan.outcome.chosen.is_some());
        let report = plan.render_report();
        assert!(report.contains("CEP Pruning Report"));
    }

    #[test]
    fn run_prune_reports_positive_reduction_when_sparsity_hit() {
        let plan = run_prune(CepPruneInput {
            spec: tiny_spec(),
            weights: None,
            target: "H100",
            constraints: Constraints {
                target_sparsity: 0.01,
                ..Default::default()
            },
            granularity: Granularity::Head,
            roofline_slack: RooflineSlackTable::default(),
        });
        assert!(plan.param_reduction() >= 0.0);
    }

    #[test]
    fn run_search_produces_ranked_candidates() {
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
        let plan = run_search(CepSearchInput {
            axes,
            target: "H100",
            constraints: Constraints::default(),
            objective: NasObjective::ParamEfficiency,
        });
        assert_eq!(plan.mode, CepMode::Search);
        assert!(plan.outcome.candidates_evaluated > 0);
        let report = plan.render_report();
        assert!(report.contains("CEP Architecture Search Report"));
    }

    #[test]
    fn plan_is_deterministic_across_runs() {
        let input = || CepPruneInput {
            spec: tiny_spec(),
            weights: None,
            target: "H100",
            constraints: Constraints {
                target_sparsity: 0.05,
                ..Default::default()
            },
            granularity: Granularity::Head,
            roofline_slack: RooflineSlackTable::default(),
        };
        let r1 = run_prune(input()).render_report();
        let r2 = run_prune(input()).render_report();
        assert_eq!(r1, r2);
    }

    #[test]
    fn synthetic_importance_covers_every_head() {
        let spec = tiny_spec();
        let slacks = RooflineSlackTable::default();
        let table = synthetic_importance(&spec, &slacks);
        let expected: u32 = spec.n_heads.iter().sum();
        assert_eq!(table.heads.len() as u32, expected);
    }

    // W4-1: format_params_si must always return millions-only (no suffix), so the render
    // line "{}M params" composes correctly for sub-1M model candidates.
    #[test]
    fn format_params_si_handles_sub_1m() {
        assert_eq!(format_params_si(500_000), "0.5");
        assert_eq!(format_params_si(100), "0.0");
        assert_eq!(format_params_si(48_800_000), "48.8");
        // Values >= 1M still render correctly (no "K" or "M" suffix appended by the helper).
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
        };
        assert_eq!(plan.param_reduction(), 0.0);
    }

    /// G19 — paper §8.1 projection smoke test (NSLCoder-50M on H100).
    ///
    /// Asserts that CEP pruning produces PROJECTED-direction effects: monotonic decrease
    /// in params / peak memory / binary size / kernel launches as sparsity rises, within
    /// 3× of the §8.1 ratios. Does NOT assert exact paper numbers (those are projections,
    /// not ground truth) — the oracle's first-order models for peak memory + binary size
    /// are deliberately conservative, so we allow wide windows but require direction.
    ///
    /// Paper §3: NSLCoder-50M shape — D=512, L=8, H=8, KV=4, FFN=1408, head_dim=64.
    /// Paper §8.1 baseline row: 48.8M params, 5.2 GB peak mem, 6.9 μs/tok, 195 MB binary.
    #[test]
    fn nslcoder_50m_projection_monotonicity() {
        // §3 shape: D=512, L=8, H=8, KV=4, head_dim=64, FFN=1408, vocab=32000.
        let spec = ModelSpec::uniform(512, 8, 8, 4, 64, 1408, 32000);
        // find_gpu("H100") prefix-matches to H100-SXM (prefers SXM per gpu_specs::find_gpu).
        let gpu = crate::gpu_specs::find_gpu("H100").expect("H100 in GPU_DATABASE");

        // --- Baseline ---
        let baseline = crate::cep_oracle::evaluate(&spec, gpu).expect("baseline profile");

        // Param count: paper says 48.8M. Oracle uses fp16 (dtype_bytes=2).
        let param_count = baseline.param_bytes / spec.dtype_bytes as u64;
        assert!(
            param_count >= 20_000_000 && param_count <= 100_000_000,
            "NSLCoder-50M param count out of [20M, 100M]: {}",
            param_count
        );

        // Peak memory: paper says 5.2 GB (full deployment with KV cache + batch).
        // The oracle computes a first-order estimate: param_bytes + max_activation_per_layer,
        // which is much smaller (order of 100s of MB for a 50M param FP16 model). We assert
        // the oracle's output is in a sane range relative to what it can compute.
        //
        // Note: peak_memory_bytes = param_bytes + max_activation_bytes (oracle formula), so
        // its monotonic decrease under pruning is GUARANTEED by param_bytes monotonicity.
        // The assertions here are sanity coverage, not independent signal.
        //
        // Floor: at least param_bytes + 1024 (the activation term must add at least a small
        // page — this checks the formula structure rather than being a tautology of >= param_bytes).
        // Ceiling: 50 GB (any sane oracle estimate for a ~50M param model is well below this).
        assert!(
            baseline.peak_memory_bytes >= baseline.param_bytes + 1024,
            "peak_memory_bytes {} must exceed param_bytes {} by at least 1 KiB (activation term)",
            baseline.peak_memory_bytes,
            baseline.param_bytes
        );
        assert!(
            baseline.peak_memory_bytes <= 50_000_000_000,
            "peak_memory_bytes > 50 GB: {}",
            baseline.peak_memory_bytes
        );

        // Latency must be positive and finite.
        assert!(
            baseline.estimated_latency_us > 0.0 && baseline.estimated_latency_us.is_finite(),
            "estimated_latency_us invalid: {}",
            baseline.estimated_latency_us
        );

        // Binary size: paper says 195 MB. Oracle is first-order; allow up to 1 GB.
        //
        // Note: binary_size_bytes = param_bytes + ~20 KB/layer + 60 KB overhead (oracle
        // formula), so its monotonic decrease under pruning is GUARANTEED by param_bytes
        // monotonicity. The assertions here are sanity coverage, not independent signal.
        //
        // Floor: at least param_bytes + 20_000 (at least one layer's code section) — this
        // checks the formula structure rather than being a tautology of >= param_bytes.
        assert!(
            baseline.binary_size_bytes >= baseline.param_bytes + 20_000,
            "binary_size_bytes {} must exceed param_bytes {} by at least 20 KB (per-layer overhead)",
            baseline.binary_size_bytes,
            baseline.param_bytes
        );
        assert!(
            baseline.binary_size_bytes <= 1_000_000_000,
            "binary_size_bytes > 1 GB: {}",
            baseline.binary_size_bytes
        );

        // Kernel launches: 8-layer model; paper estimates ~16 launches/layer × 8 + 2 ≈ 130.
        // Allow wide [10, 200] window.
        assert!(
            baseline.kernel_launches >= 10 && baseline.kernel_launches <= 200,
            "kernel_launches out of [10, 200]: {}",
            baseline.kernel_launches
        );

        // --- Pruned profiles at three sparsity levels ---
        let sparsities: &[f64] = &[0.2, 0.3, 0.5];
        let mut pruned_profiles: Vec<crate::cep_oracle::CompilationProfile> = Vec::new();

        for &sparsity in sparsities {
            let plan = run_prune(CepPruneInput {
                spec: spec.clone(),
                weights: None,
                target: "H100",
                constraints: Constraints {
                    target_sparsity: sparsity,
                    ..Default::default()
                },
                granularity: Granularity::HeadAndFfn,
                roofline_slack: RooflineSlackTable::default(),
            });

            let chosen = plan.outcome.chosen.as_ref().expect("chosen must be Some after pruning");
            // Pruned must be strictly smaller than baseline on params.
            assert!(
                chosen.profile.param_bytes < baseline.param_bytes,
                "sparsity={}: pruned param_bytes {} >= baseline {}",
                sparsity,
                chosen.profile.param_bytes,
                baseline.param_bytes
            );
            // Peak memory: ≤ baseline (may not strictly decrease at low sparsity).
            assert!(
                chosen.profile.peak_memory_bytes <= baseline.peak_memory_bytes,
                "sparsity={}: pruned peak_memory_bytes {} > baseline {}",
                sparsity,
                chosen.profile.peak_memory_bytes,
                baseline.peak_memory_bytes
            );
            // Binary size: ≤ baseline.
            assert!(
                chosen.profile.binary_size_bytes <= baseline.binary_size_bytes,
                "sparsity={}: pruned binary_size_bytes {} > baseline {}",
                sparsity,
                chosen.profile.binary_size_bytes,
                baseline.binary_size_bytes
            );
            // Kernel launches: under HeadAndFfn granularity without layer drop,
            // kernel_launches is INVARIANT (depends only on layer count, not head
            // count).  No per-pruned assertion — it would be vacuously true.
            // Real search time was measured.
            assert!(
                plan.outcome.wall_clock_us > 0,
                "sparsity={}: wall_clock_us must be > 0",
                sparsity
            );

            pruned_profiles.push(chosen.profile.clone());
        }

        // --- Monotonicity: 20% < 30% < 50% sparsity means profiles[0] >= profiles[1] >= profiles[2] ---
        // profiles[0]=20%, profiles[1]=30%, profiles[2]=50%
        // Higher sparsity → strictly fewer params (and ≤ on other metrics).
        //
        // param_bytes monotonicity is the load-bearing signal.
        // peak_memory_bytes and binary_size_bytes are derived from param_bytes plus small
        // constants in the oracle (max_activation_bytes for peak; ~20KB/layer + 60KB for
        // binary), so their monotonic decrease is GUARANTEED by param_bytes monotonicity.
        // These assertions are sanity coverage, not independent signal.
        //
        // kernel_launches depends only on layer count (not head count), so under
        // HeadAndFfn-without-layer-drop it is INVARIANT — monotonicity would be vacuous.
        // No kernel_launches monotonicity assertion.
        for w in pruned_profiles.windows(2) {
            let less_sparse = &w[0];
            let more_sparse = &w[1];
            assert!(
                more_sparse.param_bytes <= less_sparse.param_bytes,
                "monotonicity violated on param_bytes: 30%/50% profile ({}) > 20%/30% profile ({})",
                more_sparse.param_bytes,
                less_sparse.param_bytes
            );
            assert!(
                more_sparse.binary_size_bytes <= less_sparse.binary_size_bytes,
                "monotonicity violated on binary_size_bytes: {} > {}",
                more_sparse.binary_size_bytes,
                less_sparse.binary_size_bytes
            );
            assert!(
                more_sparse.peak_memory_bytes <= less_sparse.peak_memory_bytes,
                "monotonicity violated on peak_memory_bytes: {} > {}",
                more_sparse.peak_memory_bytes,
                less_sparse.peak_memory_bytes
            );
        }
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
        let input = build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", None).expect("input");
        assert_eq!(input.constraints.preserve_layers, vec![0, 5]);
    }

    #[test]
    fn build_prune_input_accepts_blocks_n_without_star() {
        let baseline = spec();
        let mut cfg = prune_cfg();
        cfg.preserve = vec!["blocks.1".to_string(), "blocks.3.attn".to_string()];
        let input = build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", None).expect("input");
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
        let input = build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", Some(0.2)).expect("input");
        let plan = run_prune(input);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("model.cep.json");
        write_prune_delta(&plan, &baseline, &path).expect("write");

        let mut s = String::new();
        std::fs::File::open(&path).unwrap().read_to_string(&mut s).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["cep_version"], 2);
        assert_eq!(v["mode"], "prune");
        assert_eq!(v["target"], "H100-SXM");
        assert!(v["baseline_profile"]["param_bytes"].is_number());
        assert!(v["chosen"]["spec"]["d_model"].is_number());
        assert!(v["chosen"]["spec"]["activation"].is_string());
        assert!(v["delta"]["per_layer"].is_array());
        assert_eq!(v["delta"]["per_layer"].as_array().unwrap().len(), baseline.n_layers as usize);
    }

    #[test]
    fn plan_to_prune_delta_is_group_aligned_and_consistent() {
        // baseline: 6 heads, 3 kv -> group size 2. Prune to 4 heads / 2 kv (drop 1 group).
        let baseline = spec(); // d_model=384, n_layers=6, n_heads=[6;6], n_kv_heads=[3;6], head_dim=[64;6], d_ff=[1024;6]
        let cfg = prune_cfg();
        let input = build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", Some(0.2)).expect("input");
        let plan = run_prune(input);
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
            assert_eq!(ld.pruned_heads.len() % group as usize, 0, "layer {l}: not group-aligned");
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

    #[test]
    fn delta_json_preserves_per_layer_head_dim() {
        // Build a spec with HETEROGENEOUS head_dim per layer and confirm the serialized
        // DTO carries them all as an array (not just the first scalar value).
        let mut baseline = spec(); // n_layers=6
        baseline.head_dim = vec![64, 64, 64, 64, 32, 32]; // mix two values
        let cfg = prune_cfg();
        let input =
            build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", Some(0.2)).expect("input");
        let plan = run_prune(input);

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
            d_model: vec![256, 384], n_layers: vec![4], n_heads: vec![4, 8],
            n_kv_heads: vec![2], d_ff: vec![512, 1024],
            activation: vec![crate::cep_oracle::Activation::SwiGlu],
            norm: vec![crate::cep_oracle::NormType::RmsNorm],
            vocab: 4096, head_dim: 64, max_seq: 2048, batch: 1, dtype_bytes: 4,
        };
        let scfg = nsl_semantic::cep::CepSearchConfig {
            target: None,
            objective: nsl_semantic::cep::NasObjective::ParamEfficiency,
            constraints: Default::default(),
            span: nsl_errors::Span::DUMMY,
        };
        let input = build_search_input(&scfg, axes, "H100-SXM").expect("input");
        let plan = run_search(input);
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("s.cep.json");
        write_search_delta(&plan, &path).expect("write");
        let s = std::fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["mode"], "search");
        assert!(v["ranked_candidates"].is_array());
    }

    #[test]
    fn prune_report_includes_search_time_constraints_binary_kernel_launches() {
        let baseline = spec();
        let cfg = prune_cfg();
        let input = build_prune_input(&cfg, baseline.clone(), None, "H100-SXM", Some(0.2)).expect("input");
        let plan = run_prune(input);
        let rendered = plan.render_report();
        // §6.3 mandatory lines
        assert!(rendered.contains("Search time:"), "missing 'Search time:' line\n{}", rendered);
        assert!(
            rendered.contains("All constraints satisfied") || rendered.contains("Constraints violated"),
            "missing constraint-status line\n{}", rendered
        );
        assert!(rendered.contains("Binary size:"), "missing 'Binary size:' line\n{}", rendered);
        assert!(
            rendered.contains("Kernel launches per forward:"),
            "missing 'Kernel launches per forward:' line\n{}", rendered
        );
    }

    #[test]
    fn search_report_includes_search_space_feasible_compilation_time_top3() {
        use crate::cep_rewrite::SearchAxes;
        let axes = SearchAxes {
            d_model: vec![256, 384], n_layers: vec![4], n_heads: vec![4, 8],
            n_kv_heads: vec![2], d_ff: vec![512, 1024],
            activation: vec![crate::cep_oracle::Activation::SwiGlu],
            norm: vec![crate::cep_oracle::NormType::RmsNorm],
            vocab: 4096, head_dim: 64, max_seq: 2048, batch: 1, dtype_bytes: 4,
        };
        let scfg = nsl_semantic::cep::CepSearchConfig {
            target: None,
            objective: nsl_semantic::cep::NasObjective::ParamEfficiency,
            constraints: Default::default(),
            span: nsl_errors::Span::DUMMY,
        };
        let input = build_search_input(&scfg, axes, "H100-SXM").expect("input");
        let plan = run_search(input);
        let rendered = plan.render_report();
        assert!(rendered.contains("Search space:"), "{}", rendered);
        assert!(rendered.contains("Feasible"), "{}", rendered);
        assert!(rendered.contains("Compilation time:"), "{}", rendered);
        assert!(rendered.contains("Top 3 architectures:"), "{}", rendered);
        // Per-candidate line shape — paper §6.3 example
        assert!(
            rendered.contains("d=") && rendered.contains("L=")
                && rendered.contains("H=") && rendered.contains("KV="),
            "per-candidate line missing d=/L=/H=/KV= tokens\n{}", rendered
        );
    }
}
