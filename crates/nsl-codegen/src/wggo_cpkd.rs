//! WGGO — CPKD per-layer distillation decision axes (CPKD v1 sidecar).
//!
//! CPKD (Compiler-Planned Knowledge Distillation) asks WGGO's Level-2 ILP
//! to co-decide, per layer, three distillation booleans alongside its
//! training-side axes (heads / FFN / CSHA / rank / optimizer precision /
//! FASE / PCA):
//!
//! * `feature_match[l]` — attach a feature-matching loss (student-side
//!   projection vs the teacher's hidden state) on this layer,
//! * `attn_transfer[l]` — attach an attention-map transfer loss.
//!   **Deferred in v1**: attention probabilities are never materialized in
//!   HBM (flash-style kernels), so the distill frontend hard-errors on
//!   `attn_transfer=true`.  The axis exists so budgets/reports are ready,
//!   but the default config keeps it OFF (`attn_memory_budget = 0`),
//! * `teacher_stream[l]` — overlap the teacher forward on a second CUDA
//!   stream.  **Deferred in v1**: the runtime is single-stream today, so
//!   the default config keeps it OFF (`allow_teacher_stream = false`).
//!
//! # Honesty constraints (repo WGGO precedents — non-negotiable)
//!
//! * **Opt-in, default OFF.**  The axes only exist when the caller sets
//!   [`crate::wggo_ilp::LayerIlpConstraints::cpkd`] to `Some(config)`.
//!   With the gate off (the production default: `None`) the ILP candidate
//!   space, costs, plans, and reports are byte-identical to a build without
//!   this module.  Precedents: `cfie_infer` (G20), `zero_stage_search`
//!   (gap #6), WRGA `snap_to_grid` (gap #5).
//! * **Advisory / report-only in v1.**  Chosen values land on
//!   [`crate::wggo::WggoPlan::cpkd_distill`] and the rendered report's
//!   "[cpkd]" section — they are NOT consumed by the distill lowering and
//!   change no generated code.  There is deliberately NO
//!   `DecisionKind::CpkdDistill` trace variant: nsl-cli's decision
//!   explainer matches `DecisionKind` exhaustively (same G20 rationale,
//!   `wggo_ilp.rs` note above `DecisionTrace`); each surfaced record
//!   carries its own human-readable model note instead.
//!
//! # Surface note (sidecar, not `LayerDecision` fields)
//!
//! `LayerDecision` / `AppliedLayer` / `PerLayerOverride` are
//! literal-constructed by non-WGGO consumers (CPDT/CSHA/WRGA/FASE test
//! fixtures), so growing them is a cross-subsystem edit.  The CPKD choices
//! travel in a parallel `Vec<Option<CpkdChoice>>` sidecar and surface on
//! the plan the WGGO driver alone constructs — exactly the G20 CFIE
//! pattern (`wggo_cfie.rs` module docs).  The override carry-through can
//! be added together with the consumer wiring when the decisions stop
//! being advisory.
//!
//! # Cost model (per training step, per layer, microseconds) — ADVISORY
//!
//! ```text
//! overhead = feature_cost_us   if feature_match
//!          + attn_cost_us      if attn_transfer
//! stream   = teacher_stream_speedup * (feature_cost_us + attn_cost_us)
//! cost_us  = overhead - (teacher_stream ? stream : 0)
//! bytes    = feature_bytes_per_layer  if feature_match
//!          + attn_bytes_per_layer     if attn_transfer
//! ```
//!
//! The stream term estimates the teacher-forward time hidden by running
//! the teacher on a second stream.  v1 has no per-layer teacher-forward
//! figure in the LUT, so the configured distill costs serve as the proxy
//! scale, and the additive term MAY be negative (a net win).  This is a
//! coarse, explicitly ADVISORY estimate: nothing is lowered from it, and
//! the report says so.  Note the costs are *charges* the solver minimizes
//! — with all-positive costs the conservative all-off choice wins; a
//! caller models a *favorable* distillation configuration (e.g. the fused
//! KL-CE kernel replacing a separate CE loss pass) by supplying a negative
//! `feature_cost_us`.  v1 deliberately does not price distillation
//! *quality*; that requires a consumed objective, not an advisory one.
//!
//! Memory: `feature_match` keeps the teacher's hidden state plus the
//! student projection buffer resident (`feature_bytes_per_layer`);
//! `attn_transfer` would keep an attention-map dump resident
//! (`attn_bytes_per_layer`).  Both are charged against the layer's
//! resident-memory check when the gate is on, exactly where the CFIE KV
//! pool is charged.  `teacher_stream` charges no extra bytes in v1 (the
//! overlap needs no persistent buffer beyond activations already counted;
//! runtime streams are deferred anyway).

use serde::Serialize;

use crate::wggo_dp::{CoarseDecision, InterLayerPlan};

/// Opt-in configuration for the CPKD distillation axes.  `Some(config)` on
/// [`crate::wggo_ilp::LayerIlpConstraints::cpkd`] turns the axes on;
/// `None` (the default) keeps the ILP byte-identical to today.
#[derive(Debug, Clone)]
pub struct CpkdConfig {
    /// Ceiling on the per-layer feature-matching buffer.  `feature_match =
    /// true` only enters the candidate domain when
    /// `feature_bytes_per_layer` fits this budget.  Default `u64::MAX`
    /// (unbounded — the same "budget defaults to infinity" convention as
    /// `LayerIlpConstraints::memory_budget` / `adapter_comm_budget`).
    pub feature_memory_budget: u64,
    /// Ceiling on the per-layer attention-map buffer AND the enable gate
    /// for the `attn_transfer` axis: the axis is enumerated only when this
    /// is `> 0` (and `attn_bytes_per_layer` fits).  Default `0` — OFF,
    /// because attention transfer is deferred in v1 (the distill frontend
    /// refuses it; attention maps are never materialized in HBM).
    pub attn_memory_budget: u64,
    /// Enable gate for the `teacher_stream` axis.  Default `false` — OFF,
    /// because two-stream overlap execution is deferred in v1 (the runtime
    /// is single-stream); the decision is a pure advisory estimate.
    pub allow_teacher_stream: bool,
    /// Resident bytes a feature-matching loss keeps live on this layer
    /// (teacher hidden state + student projection buffer).  Default
    /// 2 MiB = `batch 1 x seq 1024 x d_model 512 x 4 B` f32 — the driver's
    /// default `LayerShape` (the same sizing convention as
    /// `CfieInferConfig`'s geometry defaults).
    pub feature_bytes_per_layer: u64,
    /// Resident bytes an attention-map transfer would keep live per layer.
    /// Default 32 MiB = `heads 8 x seq 1024^2 x 4 B` f32 at the driver's
    /// default shape.
    pub attn_bytes_per_layer: u64,
    /// Latency charge (us per step per layer) of the feature-matching loss
    /// (projection matmul + MSE + its backward).  Coarse documented
    /// placeholder; override per deployment.  May be NEGATIVE when the
    /// caller models a net saving (see the module cost-model docs).
    pub feature_cost_us: f64,
    /// Latency charge (us per step per layer) of the attention-transfer
    /// loss (map dump + row softmax + KL).  Coarse documented placeholder.
    pub attn_cost_us: f64,
    /// Fraction of the configured distill cost (`feature_cost_us +
    /// attn_cost_us`) the second-stream teacher overlap is estimated to
    /// hide.  Subtracted from the objective when `teacher_stream` is
    /// chosen — the term may go negative.  ADVISORY estimate only.
    pub teacher_stream_speedup: f64,
}

impl Default for CpkdConfig {
    fn default() -> Self {
        Self {
            feature_memory_budget: u64::MAX,
            // Deferred axes stay OFF by default (v1): attention transfer
            // is refused by the distill frontend, and the runtime has no
            // second stream to overlap the teacher on.
            attn_memory_budget: 0,
            allow_teacher_stream: false,
            // batch 1 x seq 1024 x d_model 512 x 4 B (f32) = 2 MiB.
            feature_bytes_per_layer: 2 * 1024 * 1024,
            // heads 8 x seq 1024 x seq 1024 x 4 B (f32) = 32 MiB.
            attn_bytes_per_layer: 32 * 1024 * 1024,
            feature_cost_us: 25.0,
            attn_cost_us: 60.0,
            teacher_stream_speedup: 0.30,
        }
    }
}

impl CpkdConfig {
    /// Field-by-field equality (f64 via bit patterns) for the templated
    /// ILP's cache key — mirrors `CfieInferConfig::key_eq` so two layers
    /// whose gates differ never share a templated solution.
    pub fn key_eq(&self, other: &Self) -> bool {
        self.feature_memory_budget == other.feature_memory_budget
            && self.attn_memory_budget == other.attn_memory_budget
            && self.allow_teacher_stream == other.allow_teacher_stream
            && self.feature_bytes_per_layer == other.feature_bytes_per_layer
            && self.attn_bytes_per_layer == other.attn_bytes_per_layer
            && self.feature_cost_us.to_bits() == other.feature_cost_us.to_bits()
            && self.attn_cost_us.to_bits() == other.attn_cost_us.to_bits()
            && self.teacher_stream_speedup.to_bits() == other.teacher_stream_speedup.to_bits()
    }
}

/// One assignment of the three CPKD distillation decision variables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct CpkdChoice {
    /// Feature-matching loss on this layer's hidden state.
    pub feature_match: bool,
    /// Attention-map transfer loss on this layer (deferred in v1 — the
    /// axis only enters the domain when the caller opens
    /// `attn_memory_budget`).
    pub attn_transfer: bool,
    /// Teacher forward overlapped on a second stream (deferred in v1 —
    /// the axis only enters the domain under `allow_teacher_stream`).
    pub teacher_stream: bool,
}

/// Per-layer advisory record surfaced on `WggoPlan::cpkd_distill`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct CpkdLayerDistill {
    pub layer_index: u32,
    pub layer_name: String,
    pub choice: CpkdChoice,
    /// Human-readable statement of the cost model AND the constants this
    /// layer's decision was priced with (the `CfieLayerInference::note`
    /// precedent) — the report prints it so the advisory numbers are
    /// inspectable without reading code.
    pub note: String,
}

/// The assumptions line for a config: restates the distill cost model with
/// its actual constants.
pub fn model_note(cfg: &CpkdConfig) -> String {
    format!(
        "distill_us/step/layer = feature({f:.2}us, {fb} B) + attn({a:.2}us, {ab} B) \
         - stream_overlap({s:.2} x configured distill cost); \
         ADVISORY v1 estimate — not consumed by lowering",
        f = cfg.feature_cost_us,
        fb = cfg.feature_bytes_per_layer,
        a = cfg.attn_cost_us,
        ab = cfg.attn_bytes_per_layer,
        s = cfg.teacher_stream_speedup,
    )
}

/// Resident bytes a CPKD choice keeps live on this layer.  Added to the
/// ILP's per-layer resident-memory check when the gate is on (exactly
/// where the CFIE KV pool is charged).  `teacher_stream` charges nothing
/// in v1 — see the module docs.
pub fn choice_bytes(choice: CpkdChoice, cfg: &CpkdConfig) -> u64 {
    let mut bytes = 0u64;
    if choice.feature_match {
        bytes = bytes.saturating_add(cfg.feature_bytes_per_layer);
    }
    if choice.attn_transfer {
        bytes = bytes.saturating_add(cfg.attn_bytes_per_layer);
    }
    bytes
}

/// Distillation latency (us per step) this layer contributes under a CPKD
/// choice.  Additive into the ILP objective, commensurate in us with the
/// forward/backward/optimizer terms the ILP already sums.  The term MAY be
/// negative (favorable `feature_cost_us`, or the teacher-stream overlap
/// estimate) — see the module cost-model docs; v1 treats it as an
/// ADVISORY estimate only.
pub fn choice_cost_us(choice: CpkdChoice, cfg: &CpkdConfig) -> f64 {
    let mut cost = 0.0;
    if choice.feature_match {
        cost += cfg.feature_cost_us;
    }
    if choice.attn_transfer {
        cost += cfg.attn_cost_us;
    }
    if choice.teacher_stream {
        // Overlap estimate: a second stream hides `teacher_stream_speedup`
        // of the configured distill cost (the proxy scale for the teacher
        // forward v1 cannot price per-layer).
        cost -= cfg.teacher_stream_speedup * (cfg.feature_cost_us + cfg.attn_cost_us);
    }
    cost
}

/// Candidate domain for the ILP's CPKD axis.
///
/// * Gate off (`cfg == None`): the single `None` element — the solver's
///   loop collapses to one pass with a zero cost term and a 0-byte charge,
///   keeping the gate-off candidate space (and node counts) byte-identical
///   to today.
/// * Gate on: the cross-product of the *enabled* booleans.
///   `feature_match = true` enters only when `feature_bytes_per_layer`
///   fits `feature_memory_budget`; `attn_transfer = true` only when
///   `attn_memory_budget > 0` (the deferred-axis enable gate) AND
///   `attn_bytes_per_layer` fits it; `teacher_stream = true` only under
///   `allow_teacher_stream`.  Ordering is deterministic and conservative-
///   first (`false` before `true` on every axis) — the solver keeps the
///   FIRST candidate on a tie (strict `<` improvement), so a distillation
///   axis must be strictly cheaper to be advised on.
pub fn enumerate_choices(cfg: Option<&CpkdConfig>) -> Vec<Option<CpkdChoice>> {
    let Some(cfg) = cfg else {
        return vec![None];
    };
    let feature_domain: &[bool] = if cfg.feature_bytes_per_layer <= cfg.feature_memory_budget {
        &[false, true]
    } else {
        &[false]
    };
    let attn_domain: &[bool] =
        if cfg.attn_memory_budget > 0 && cfg.attn_bytes_per_layer <= cfg.attn_memory_budget {
            &[false, true]
        } else {
            &[false]
        };
    let stream_domain: &[bool] = if cfg.allow_teacher_stream {
        &[false, true]
    } else {
        &[false]
    };
    let mut out =
        Vec::with_capacity(feature_domain.len() * attn_domain.len() * stream_domain.len());
    for &feature_match in feature_domain {
        for &attn_transfer in attn_domain {
            for &teacher_stream in stream_domain {
                out.push(Some(CpkdChoice {
                    feature_match,
                    attn_transfer,
                    teacher_stream,
                }));
            }
        }
    }
    out
}

/// Greedy pick: the cheapest choice whose resident bytes fit `pool_budget`
/// (remaining memory headroom after the layer's training-resident bytes
/// and the CFIE KV pool).  Mirrors `wggo_cfie::best_choice_fitting`.  The
/// all-off choice is free (0 bytes, 0 us), so a gate-on call always
/// advises *something*; `None` is kept in the signature for parity and
/// against future non-zero baselines.
pub fn best_choice_fitting(cfg: &CpkdConfig, pool_budget: u64) -> Option<(CpkdChoice, f64)> {
    let mut best: Option<(CpkdChoice, f64)> = None;
    for choice in enumerate_choices(Some(cfg)).into_iter().flatten() {
        if choice_bytes(choice, cfg) > pool_budget {
            continue;
        }
        let cost = choice_cost_us(choice, cfg);
        if best.as_ref().is_none_or(|(_, b)| cost < *b) {
            best = Some((choice, cost));
        }
    }
    best
}

/// Project the per-layer choices onto the advisory plan surface, skipping
/// layers the inter-layer DP pruned (mirroring `wggo_apply::apply`'s
/// forced-safe-defaults treatment of pruned layers).  `constraints`
/// supplies each layer's gate config for the record's model note; the
/// defensive first-element fallback mirrors the driver's `recost_total`
/// indexing (a caller may pass fewer constraints than layers).
pub fn surface_from_plan(
    inter: &InterLayerPlan,
    choices: &[Option<CpkdChoice>],
    constraints: &[crate::wggo_ilp::LayerIlpConstraints],
) -> Vec<CpkdLayerDistill> {
    inter
        .layers
        .iter()
        .zip(choices.iter())
        .enumerate()
        .filter(|(_, (l, _))| l.decision != CoarseDecision::Prune)
        .filter_map(|(i, (l, ch))| {
            ch.map(|choice| {
                let note = constraints
                    .get(i)
                    .or_else(|| constraints.first())
                    .and_then(|c| c.cpkd.as_ref())
                    .map(model_note)
                    .unwrap_or_default();
                CpkdLayerDistill {
                    layer_index: l.layer_index,
                    layer_name: l.name.clone(),
                    choice,
                    note,
                }
            })
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wggo_dp::LayerPlan;

    fn choice(feature_match: bool, attn_transfer: bool, teacher_stream: bool) -> CpkdChoice {
        CpkdChoice {
            feature_match,
            attn_transfer,
            teacher_stream,
        }
    }

    #[test]
    fn gate_off_domain_is_the_single_none() {
        assert_eq!(enumerate_choices(None), vec![None]);
    }

    #[test]
    fn default_config_enumerates_only_the_feature_axis() {
        // v1 defaults keep the deferred axes OFF: attn (budget 0) and
        // stream (not allowed) stay pinned false, so the domain is the
        // feature boolean only — conservative all-off FIRST.
        let cfg = CpkdConfig::default();
        let choices = enumerate_choices(Some(&cfg));
        assert_eq!(
            choices,
            vec![
                Some(choice(false, false, false)),
                Some(choice(true, false, false)),
            ]
        );
    }

    #[test]
    fn tiny_feature_budget_pins_feature_axis_false() {
        let cfg = CpkdConfig {
            feature_memory_budget: 1, // below the 2 MiB buffer
            ..Default::default()
        };
        let choices = enumerate_choices(Some(&cfg));
        assert_eq!(choices, vec![Some(choice(false, false, false))]);
    }

    #[test]
    fn attn_axis_requires_positive_budget_that_fits() {
        // Budget 0: axis off (the deferred-axis enable gate).
        let off = CpkdConfig::default();
        assert!(enumerate_choices(Some(&off))
            .iter()
            .flatten()
            .all(|c| !c.attn_transfer));
        // Budget > 0 but below the buffer: still off.
        let too_small = CpkdConfig {
            attn_memory_budget: 1,
            ..Default::default()
        };
        assert!(enumerate_choices(Some(&too_small))
            .iter()
            .flatten()
            .all(|c| !c.attn_transfer));
        // Budget that fits: 2 feature x 2 attn = 4 choices.
        let on = CpkdConfig {
            attn_memory_budget: u64::MAX,
            ..Default::default()
        };
        let choices = enumerate_choices(Some(&on));
        assert_eq!(choices.len(), 4);
        assert!(choices.iter().flatten().any(|c| c.attn_transfer));
    }

    #[test]
    fn stream_axis_requires_allow_flag() {
        let on = CpkdConfig {
            allow_teacher_stream: true,
            ..Default::default()
        };
        let choices = enumerate_choices(Some(&on));
        // 2 feature x 1 attn x 2 stream = 4.
        assert_eq!(choices.len(), 4);
        assert!(choices.iter().flatten().any(|c| c.teacher_stream));
        // All three open: 2 x 2 x 2 = 8, all-off first.
        let all = CpkdConfig {
            attn_memory_budget: u64::MAX,
            allow_teacher_stream: true,
            ..Default::default()
        };
        let choices = enumerate_choices(Some(&all));
        assert_eq!(choices.len(), 8);
        assert_eq!(choices[0], Some(choice(false, false, false)));
    }

    #[test]
    fn choice_bytes_hand_computed() {
        let cfg = CpkdConfig::default();
        assert_eq!(choice_bytes(choice(false, false, false), &cfg), 0);
        assert_eq!(
            choice_bytes(choice(true, false, false), &cfg),
            2 * 1024 * 1024
        );
        assert_eq!(
            choice_bytes(choice(true, true, false), &cfg),
            2 * 1024 * 1024 + 32 * 1024 * 1024
        );
        // teacher_stream charges no extra bytes in v1.
        assert_eq!(
            choice_bytes(choice(true, false, true), &cfg),
            choice_bytes(choice(true, false, false), &cfg)
        );
    }

    #[test]
    fn choice_cost_hand_computed_including_negative_stream_term() {
        let cfg = CpkdConfig {
            allow_teacher_stream: true,
            ..Default::default()
        };
        // Defaults: feature 25.0, attn 60.0, speedup 0.30.
        assert_eq!(choice_cost_us(choice(false, false, false), &cfg), 0.0);
        assert_eq!(choice_cost_us(choice(true, false, false), &cfg), 25.0);
        assert_eq!(choice_cost_us(choice(true, true, false), &cfg), 85.0);
        // Stream subtracts 0.30 * (25 + 60) = 25.5 us:
        //   feature+stream = 25.0 - 25.5 = -0.5 (negative — documented
        //   advisory overlap estimate).
        let got = choice_cost_us(choice(true, false, true), &cfg);
        assert!((got - (-0.5)).abs() < 1e-12, "got {got}");
        // Stream alone is the pure (negative) overlap estimate.
        let alone = choice_cost_us(choice(false, false, true), &cfg);
        assert!((alone - (-25.5)).abs() < 1e-12, "got {alone}");
    }

    #[test]
    fn best_choice_fitting_prefers_cheapest_that_fits() {
        // Default costs are all-positive: the conservative all-off choice
        // (0 us) wins.
        let cfg = CpkdConfig::default();
        let (c, cost) = best_choice_fitting(&cfg, u64::MAX).unwrap();
        assert_eq!(c, choice(false, false, false));
        assert_eq!(cost, 0.0);
        // A favorable (negative) feature cost flips it on...
        let fav = CpkdConfig {
            feature_cost_us: -15.0,
            ..Default::default()
        };
        let (c2, cost2) = best_choice_fitting(&fav, u64::MAX).unwrap();
        assert!(c2.feature_match);
        assert_eq!(cost2, -15.0);
        // ...unless the pool headroom cannot hold the feature buffer.
        let (c3, _) = best_choice_fitting(&fav, 1024).unwrap();
        assert!(!c3.feature_match);
    }

    #[test]
    fn key_eq_detects_drift_on_every_field() {
        let base = CpkdConfig::default();
        assert!(base.key_eq(&CpkdConfig::default()));
        let variants = [
            CpkdConfig { feature_memory_budget: 1, ..Default::default() },
            CpkdConfig { attn_memory_budget: 1, ..Default::default() },
            CpkdConfig { allow_teacher_stream: true, ..Default::default() },
            CpkdConfig { feature_bytes_per_layer: 1, ..Default::default() },
            CpkdConfig { attn_bytes_per_layer: 1, ..Default::default() },
            CpkdConfig { feature_cost_us: 1.0, ..Default::default() },
            CpkdConfig { attn_cost_us: 1.0, ..Default::default() },
            CpkdConfig { teacher_stream_speedup: 0.5, ..Default::default() },
        ];
        for (i, v) in variants.iter().enumerate() {
            assert!(!base.key_eq(v), "variant {i} must not be key-equal");
        }
    }

    #[test]
    fn model_note_states_the_constants() {
        let note = model_note(&CpkdConfig::default());
        assert!(note.contains("distill_us/step/layer"));
        assert!(note.contains("25.00us"));
        assert!(note.contains("60.00us"));
        assert!(note.contains("0.30"));
        assert!(note.contains("ADVISORY"));
    }

    #[test]
    fn surface_from_plan_skips_pruned_layers() {
        let mk_layer = |i: u32, decision: CoarseDecision| LayerPlan {
            layer_index: i,
            name: format!("blocks.{i}"),
            decision,
            pipeline_stage: 0,
            shard_params: 1,
            shard_grads: 1,
            shard_optim: 1,
            estimated_us: 1.0,
            estimated_bytes: 1,
            param_bytes: 1,
            activation_bytes: 1,
        };
        let inter = InterLayerPlan {
            layers: vec![
                mk_layer(0, CoarseDecision::KeepFull),
                mk_layer(1, CoarseDecision::Prune),
            ],
            total_us: 2.0,
            peak_memory_bytes: 2,
            pipeline_stages: 1,
        };
        let ch = Some(choice(true, false, false));
        let cons = vec![
            crate::wggo_ilp::LayerIlpConstraints {
                cpkd: Some(CpkdConfig::default()),
                ..Default::default()
            };
            2
        ];
        let surfaced = surface_from_plan(&inter, &[ch, ch], &cons);
        assert_eq!(surfaced.len(), 1);
        assert_eq!(surfaced[0].layer_index, 0);
        assert_eq!(surfaced[0].layer_name, "blocks.0");
        assert!(surfaced[0].choice.feature_match);
        assert!(surfaced[0].note.contains("distill_us/step/layer"));
        // Gate-off constraints (no cpkd) still surface the choice, with an
        // empty note (defensive fallback).
        let cons_off = vec![crate::wggo_ilp::LayerIlpConstraints::default(); 2];
        let surfaced_off = surface_from_plan(&inter, &[ch, ch], &cons_off);
        assert_eq!(surfaced_off.len(), 1);
        assert!(surfaced_off[0].note.is_empty());
    }
}
