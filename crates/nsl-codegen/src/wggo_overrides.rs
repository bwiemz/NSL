//! Consumer-facing translation of WGGO's AppliedPlan.
//!
//! Every downstream consumer (CSHA, WRGA, FASE, Prune, Sharding) reads the
//! fields it cares about and ignores the rest. WGGO's decisions are
//! "preferences, not mandates" — consumers check hardware feasibility
//! locally and may downgrade, recording diagnostics for the decision
//! explainer.

use crate::cfie_persistent::FusionLevel;
use serde::Serialize;

/// Consumer diagnostic: one override that was rejected or adjusted.
/// Shared across CSHA, WRGA, and future consumers of `WggoOverrides`.
/// `requested` and `applied` are stringified at the producer so the
/// decision explainer has a uniform wire format across consumers.
#[derive(Debug, Clone, Serialize)]
pub struct OverrideDiagnostic {
    pub layer_index: u32,
    pub layer_name: String,
    pub reason: OverrideRejectReason,
    pub requested: String,
    pub applied: String,
}

/// Why a consumer refused or adjusted a WGGO override.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub enum OverrideRejectReason {
    // CSHA:
    SmemBudgetExceeded { actual_kb: u32, limit_kb: u32 },
    // WRGA:
    RankClampedToBounds { r_min: u32, r_max: u32 },
    RankForbiddenByWggo,
    BudgetExceededDowngraded { original_rank: u32, final_rank: u32 },
    /// WRGA's roofline wanted an adapter on a projection that falls outside
    /// WGGO's chosen `adapter_placement` for the layer (the comm-budget-feasible
    /// minimal set, G5/G7).  WRGA forces the site to `Skip` to respect the
    /// placement; `placement` is the WGGO set that excluded it.
    AdapterSiteOutsidePlacement { placement: String },
    // CPDT:
    /// WGGO's recommended shard factor doesn't divide world_size.
    ShardFactorIncompatibleWithWorldSize { recommended: u32, world_size: u32 },
    /// Memory budget required more aggressive sharding than WGGO recommended.
    ShardFactorOverriddenByMemory { recommended: u32, applied: u32 },
    // FASE:
    /// WGGO requested Deferred mode on a layer whose global FASE plan is
    /// FullBuffer because the optimizer does not support deferred accumulation
    /// (Lion, Unknown, or AdamW/Adam with allow_v_approx=false).
    FaseModeInfeasible {
        optimizer: crate::fase::FaseOptimizer,
        global_mode: crate::fase::FaseMode,
    },
    /// Two-phase-clip + mixed mode conflict. When `grad_clip` is set and
    /// accumulation > 1 and global FASE plan is Deferred, Phase A's global
    /// ||m_partial||² norm requires uniform accumulation convention across
    /// params. WGGO requests of `fase_fused=false` for individual layers
    /// are clamped back to Deferred to preserve the norm's validity.
    TwoPhaseClipConflict {
        grad_clip_threshold: f64,
    },
    // Prune:
    /// Whole-block prune (LayerRole::Block) — not yet implemented; see spec §3.6.
    /// Sub-block prune is handled by wggo_prune.rs.
    ///
    /// WGGO decided `CoarseDecision::Prune` for a layer whose weight
    /// analysis score fell below `prune_floor`, but no downstream
    /// codegen consumer implements the layer-to-residual-identity IR
    /// rewrite required to honor the decision.  The layer still executes
    /// at full cost; the `[prune] layer:N wggo-override-rejected` stderr
    /// diagnostic surfaces the gap so users and future wiring work can
    /// see the planner's intent.  Full IR rewrite is tracked as a
    /// separate follow-up on top of this diagnostic stub.
    WholeBlockPruneNotImplemented,
    /// Prune refusal — spec §3.1. Cross-layer parameter consumption.
    PruneCrossLayerParam,
    /// Prune refusal — spec §3.2. Layer lacks a residual `Add`.
    PruneNoResidualAdd,
    /// Prune refusal — spec §3.3. Parallel residual branches detected.
    PruneParallelResidualBranches,
    /// Prune refusal — spec §3.4. Multiple Adds match the residual pattern ambiguously.
    PruneAmbiguousPatternMatch,
    /// Prune refusal — spec §3.5. No parameters match the requested layer prefix.
    PruneEmptyClosure,
    /// Prune refusal — spec §3.6. Whole-block prune (LayerRole::Block) unsupported in v1.
    /// Distinct from `WholeBlockPruneNotImplemented`: this variant is emitted by
    /// `wggo_prune::run()` via the new sub-block flow; `WholeBlockPruneNotImplemented`
    /// is emitted by the legacy `collect_prune_diagnostics` path. Both coexist
    /// during v1; v2 will consolidate.
    PruneWholeBlockUnsupported,
    /// Prune refusal — spec §3.7. Two prune decisions in the same plan conflict.
    PruneConflictingDecisions,
    // PCA packing (errata E2 / audit gap #4):
    /// WGGO requested packing (`packing_mode > 0`) but the module synthesized
    /// no segment-masked attention kernels — the source has no packed
    /// `dataset` (and/or no `@flash_attention` train context), and segment
    /// masking cannot be synthesized from nothing (dataset packing is
    /// authorial, like `@pca` — see the meta-flag refusal doctrine).
    PackingRequiresPackedDataset { mode: u8 },
    /// WGGO requested `packing_mode = 0` (skip packing) but the dataset IS
    /// packed and masked kernels were synthesized. Segment masking is a
    /// correctness requirement for packed data — unmasked attention leaks
    /// across document boundaries — so the plan's preference is refused and
    /// the masked kernels keep dispatching.
    PackingMaskingMandatoryForPackedDataset,
}

#[derive(Debug, Clone, Serialize)]
pub struct WggoOverrides {
    pub per_layer: Vec<PerLayerOverride>,
}

#[derive(Debug, Clone, Serialize)]
pub struct PerLayerOverride {
    pub layer_index: u32,
    pub layer_name: String,

    // CSHA fields (consumed by this plan):
    pub active_heads: u32,
    pub requested_csha_level: Option<FusionLevel>,

    // Future-consumer fields (populated now so the struct shape is stable;
    // their wiring plans land later without signature churn):
    pub adapter_rank: u64,   // WRGA
    /// WRGA: which projections the adapter attaches to (companion to
    /// `adapter_rank`).  Carried for the WRGA-codegen wiring; consumed
    /// together with `adapter_rank` once WRGA reads per-layer overrides.
    pub adapter_placement: crate::wggo_ilp::AdapterPlacement,
    pub fase_fused: bool,    // FASE
    pub packing_mode: u8,    // PCA / fusion
    /// CPDT: per-layer ZeRO shard factor recommendation. Aggregated to a
    /// single global recommendation via `WggoOverrides::min_shard_factor`.
    /// `0` is the uninitialized sentinel — meaning "no recommendation."
    pub shard_factor: u32,
}

impl WggoOverrides {
    /// Translate from WGGO's AppliedPlan.  Pure data transform.
    pub fn from_applied(applied: &crate::wggo_apply::AppliedPlan) -> Self {
        Self {
            per_layer: applied
                .layers
                .iter()
                .map(|l| PerLayerOverride {
                    layer_index: l.layer_index,
                    layer_name: l.layer_name.clone(),
                    active_heads: l.active_heads,
                    requested_csha_level: map_csha_level(l.csha_level),
                    adapter_rank: l.adapter_rank,
                    adapter_placement: l.adapter_placement,
                    fase_fused: l.fase_fused,
                    packing_mode: l.packing_mode,
                    shard_factor: l.shard_factor,
                })
                .collect(),
        }
    }

    /// O(n) lookup by layer_index. `n` is layer count (32-128 in practice),
    /// so linear scan is fine.
    pub fn find(&self, layer_index: u32) -> Option<&PerLayerOverride> {
        self.per_layer.iter().find(|o| o.layer_index == layer_index)
    }

    /// Minimum shard factor across layers with a meaningful recommendation
    /// (`shard_factor > 0`).  Returns `None` when no override carries a
    /// recommendation.
    ///
    /// Uses MIN because the most-sensitive layer constrains the global
    /// decision: ZeRO's collective ops require uniform sharding, so a layer
    /// that only tolerates factor 2 forces every layer to factor 2.
    pub fn min_shard_factor(&self) -> Option<u32> {
        self.per_layer
            .iter()
            .filter_map(|p| if p.shard_factor > 0 { Some(p.shard_factor) } else { None })
            .min()
    }

    /// Look up the override for the layer containing a given projection
    /// name.  WRGA's `SpectralAnalysis.name` is per-projection (e.g.
    /// `blocks.0.attn.wq`); WGGO's `layer_name` is per-layer (e.g. `blocks.0`).
    /// All projections within a layer share its override.
    ///
    /// Match is **prefix + dot boundary**: `blocks.1` matches `blocks.1.attn.wq`
    /// but NOT `blocks.10.attn.wq`.
    pub fn find_by_layer_containing(&self, projection_name: &str) -> Option<&PerLayerOverride> {
        self.per_layer.iter().find(|o| {
            let ln = &o.layer_name;
            projection_name == ln
                || (projection_name.starts_with(ln.as_str())
                    && projection_name[ln.len()..].starts_with('.'))
        })
    }
}

fn map_csha_level(raw: u8) -> Option<FusionLevel> {
    match raw {
        0 => Some(FusionLevel::None),
        1 => Some(FusionLevel::Level1),
        2 => Some(FusionLevel::Level2),
        3 => Some(FusionLevel::Level3),
        _ => None,
    }
}

/// Walk an `AppliedPlan` and collect one `OverrideDiagnostic` per layer
/// whose coarse decision is `Prune`.  The downstream codegen consumer
/// that would honor these decisions (layer-to-residual-identity IR
/// rewrite) is not yet implemented; this helper makes the gap visible
/// via the `[prune] layer:N wggo-override-rejected requested=prune
/// applied=keepfull reason=ir_rewrite_not_implemented` stderr format
/// that matches the existing CSHA/WRGA/CPDT/FASE diagnostic pattern.
///
/// Empty when no layer is planned for pruning (the shipped-binary
/// common case) — caller can iterate and emit without extra gating.
pub fn collect_prune_diagnostics(
    applied: &crate::wggo_apply::AppliedPlan,
) -> Vec<OverrideDiagnostic> {
    use crate::wggo_dp::CoarseDecision;
    applied
        .layers
        .iter()
        // Spec §3.6 + Task 2: this function now surfaces only whole-block prune
        // decisions. Sub-block prune decisions (Attention / Ffn) flow through
        // `wggo_prune::run()`. Role classification is delegated to
        // `wggo_graph::infer_role` so all consumers use the same authority.
        .filter(|l| {
            matches!(l.coarse, CoarseDecision::Prune)
                && matches!(
                    crate::wggo_graph::infer_role(&l.layer_name),
                    crate::wggo_graph::LayerRole::Block
                )
        })
        .map(|l| OverrideDiagnostic {
            layer_index: l.layer_index,
            layer_name: l.layer_name.clone(),
            reason: OverrideRejectReason::WholeBlockPruneNotImplemented,
            requested: "prune".to_string(),
            applied: "keepfull".to_string(),
        })
        .collect()
}

/// Stable reason-string for `OverrideRejectReason::WholeBlockPruneNotImplemented`,
/// used by the `[prune]` stderr diagnostic emitter.  Factored out so
/// tests can assert on the literal string without reaching into private
/// rendering code in `stmt.rs`.
pub fn whole_block_prune_not_implemented_reason() -> &'static str {
    // Stable string preserved from PR #102; see spec §5.5.
    "ir_rewrite_not_implemented"
}

// ─── PCA packing_mode consumption (errata E2 / audit gap #4) ────────────────

/// Human name for a WGGO `packing_mode` value (the joint-ILP PCA decision,
/// `wggo_ilp::LayerDecision::packing_mode`). Single source of truth shared by
/// the `--wggo-report` renderer and the `[pca]` consumption diagnostics.
pub fn packing_mode_name(mode: u8) -> &'static str {
    match mode {
        0 => "none",
        1 => "segment_id",
        2 => "tile_skip",
        3 => "multi_seq",
        _ => "invalid",
    }
}

/// What packed-attention kernels the module-scan emitter actually synthesized
/// for the train block's flash-attention config — the ground truth the WGGO
/// packing decision is validated against. Derived in `stmt.rs` from
/// `FlashAttentionCompileContext`: masked kernels exist iff the training
/// config had `segment_masked=true` (dataset packing detection); when masked,
/// per-doc CTA *replaced* the Tier-B pair iff the Tier-B-on data IDs are None.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PackingKernelState {
    /// No segment-masked attention kernels in the module (dataset not packed,
    /// or no `@flash_attention` train context at all).
    NoMaskedKernels,
    /// Base + Tier-B-on masked variant pair emitted; the runtime gate picks
    /// the masked variant when segment IDs are present.
    TierBMasked,
    /// Per-document CTA kernels admitted (user `@pca(strategy=per_document)`);
    /// they replace the Tier-B pair as the training forward.
    PerDocCta,
}

/// One `[pca]` consumption verdict for a layer's WGGO packing decision.
#[derive(Debug, Clone)]
pub struct PackingDiagnostic {
    pub layer_index: u32,
    pub layer_name: String,
    /// The plan's requested packing mode (see `packing_mode_name`).
    pub mode: u8,
    pub verdict: PackingVerdict,
}

#[derive(Debug, Clone)]
pub enum PackingVerdict {
    /// The plan's packing request and the synthesized kernels agree; `kernel`
    /// names what actually dispatches.
    Consumed { kernel: &'static str },
    /// The plan's request cannot be honored; masked-kernel reality wins.
    Rejected(OverrideRejectReason),
}

/// Validate the plan's per-layer `packing_mode` against the kernels the module
/// actually synthesized, yielding one diagnostic per layer that has anything
/// to say (silent when plan and kernels trivially agree: `mode == 0` with no
/// masked kernels).
///
/// Semantics (v1 — consumption/validation layer):
///   * `mode > 0` with masked kernels present → **consumed**: the decision and
///     the emitted kernels agree (Tier-B masked or per-doc CTA).
///   * `mode > 0` with NO masked kernels → **rejected**
///     [`OverrideRejectReason::PackingRequiresPackedDataset`]: WGGO wants
///     packing but the source has no packed `dataset` (segment masking cannot
///     be synthesized from nothing — the decorator/dataset are authorial).
///   * `mode == 0` with masked kernels present → **rejected**
///     [`OverrideRejectReason::PackingMaskingMandatoryForPackedDataset`]:
///     segment masking is a *correctness* requirement for packed data
///     (unmasked attention leaks across document boundaries), so the plan's
///     preference to skip packing cannot be honored by dispatching unmasked
///     kernels. The layer still pays the (small) masking cost; the ILP's
///     savings estimate for this layer was optimistic.
///
/// The kernel-granularity caveat: attention-kernel synthesis is per train
/// block today, not per layer, so all layers validate against the same
/// `PackingKernelState`. True per-layer packing dispatch needs per-layer
/// kernel variants — that, plus letting a WGGO decision *flip the per-doc
/// admission itself*, requires hoisting WGGO planning ahead of
/// `compile_flash_attention_kernels` (the module-scan emitter runs before
/// `compile_main`, so `wggo_overrides` is `None` at admission time). That
/// ordering restructure is the tracked follow-up; this validation layer is
/// the same first increment the `[prune]` diagnostics established.
pub fn collect_packing_diagnostics(
    applied: &crate::wggo_apply::AppliedPlan,
    state: PackingKernelState,
) -> Vec<PackingDiagnostic> {
    applied
        .layers
        .iter()
        .filter_map(|l| {
            let verdict = match (l.packing_mode, state) {
                // Agreement, nothing to report: no packing wanted, none exists.
                (0, PackingKernelState::NoMaskedKernels) => return None,
                (0, _) => PackingVerdict::Rejected(
                    OverrideRejectReason::PackingMaskingMandatoryForPackedDataset,
                ),
                (m, PackingKernelState::NoMaskedKernels) => PackingVerdict::Rejected(
                    OverrideRejectReason::PackingRequiresPackedDataset { mode: m },
                ),
                (_, PackingKernelState::TierBMasked) => PackingVerdict::Consumed {
                    kernel: "segment-masked Tier-B attention",
                },
                (_, PackingKernelState::PerDocCta) => PackingVerdict::Consumed {
                    kernel: "per-document CTA attention",
                },
            };
            Some(PackingDiagnostic {
                layer_index: l.layer_index,
                layer_name: l.layer_name.clone(),
                mode: l.packing_mode,
                verdict,
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::wggo_apply::{AppliedLayer, AppliedPlan};
    use crate::wggo_dp::CoarseDecision;

    fn sample_applied() -> AppliedPlan {
        AppliedPlan {
            layers: vec![AppliedLayer {
                layer_index: 7,
                layer_name: "blocks.7".to_string(),
                coarse: CoarseDecision::KeepFull,
                pipeline_stage: 0,
                shard_factor: 1,
                shard_grads: 1,
                shard_optim: 1,
                active_heads: 4,
                ffn_width: 2048,
                csha_level: 2,
                adapter_rank: 16,
                adapter_placement: crate::wggo_ilp::AdapterPlacement::None,
                optim_m_bits: 8,
                optim_v_bits: 16,
                fase_fused: true,
                packing_mode: 1,
                estimated_us: 42.0,
                param_bytes: 0,
                activation_bytes: 0,
            }],
            total_us: 42.0,
            peak_memory_bytes: 0,
        }
    }

    #[test]
    fn from_applied_translation_maps_all_fields() {
        let applied = sample_applied();
        let over = WggoOverrides::from_applied(&applied);
        assert_eq!(over.per_layer.len(), 1);
        let p = &over.per_layer[0];
        assert_eq!(p.layer_index, 7);
        assert_eq!(p.layer_name, "blocks.7");
        assert_eq!(p.active_heads, 4);
        assert_eq!(p.requested_csha_level, Some(FusionLevel::Level2));
        assert_eq!(p.adapter_rank, 16);
        assert!(p.fase_fused);
        assert_eq!(p.packing_mode, 1);
    }

    #[test]
    fn find_returns_record_by_layer_index() {
        let over = WggoOverrides::from_applied(&sample_applied());
        assert!(over.find(7).is_some());
        assert!(over.find(99).is_none());
    }

    #[test]
    fn empty_applied_yields_no_overrides() {
        // Load-bearing invariant for WGGO's §2.4 shape-compatibility refusal:
        // `wggo::run` emits an empty AppliedPlan when a graph is provably
        // shape-incompatible, and the consumer must therefore apply *no*
        // transforms (CSHA/WRGA/CPDT all fall back to defaults).  If this ever
        // started synthesising default overrides, the refusal would silently
        // leak transforms onto a broken graph.
        let empty = crate::wggo_apply::AppliedPlan {
            layers: Vec::new(),
            total_us: 0.0,
            peak_memory_bytes: 0,
        };
        let over = WggoOverrides::from_applied(&empty);
        assert!(over.per_layer.is_empty());
        assert!(over.find(0).is_none());
    }

    #[test]
    fn map_csha_level_covers_all_valid_raws() {
        assert_eq!(map_csha_level(0), Some(FusionLevel::None));
        assert_eq!(map_csha_level(1), Some(FusionLevel::Level1));
        assert_eq!(map_csha_level(2), Some(FusionLevel::Level2));
        assert_eq!(map_csha_level(3), Some(FusionLevel::Level3));
        assert_eq!(map_csha_level(99), None);
    }

    #[test]
    fn override_diagnostic_has_string_requested_and_applied() {
        let d = OverrideDiagnostic {
            layer_index: 3,
            layer_name: "blocks.3".into(),
            reason: OverrideRejectReason::RankClampedToBounds { r_min: 2, r_max: 16 },
            requested: "32".to_string(),
            applied: "16".to_string(),
        };
        assert_eq!(d.requested, "32");
        assert_eq!(d.applied, "16");
    }

    #[test]
    fn fase_mode_infeasible_round_trips_debug() {
        let r = OverrideRejectReason::FaseModeInfeasible {
            optimizer: crate::fase::FaseOptimizer::Lion,
            global_mode: crate::fase::FaseMode::FullBuffer,
        };
        let s = format!("{:?}", r);
        assert!(s.contains("FaseModeInfeasible"));
        assert!(s.contains("Lion"));
        assert!(s.contains("FullBuffer"));
    }

    #[test]
    fn two_phase_clip_conflict_round_trips_debug() {
        let r = OverrideRejectReason::TwoPhaseClipConflict {
            grad_clip_threshold: 1.5,
        };
        let s = format!("{:?}", r);
        assert!(s.contains("TwoPhaseClipConflict"));
        assert!(s.contains("1.5"));
    }

    #[test]
    fn override_reject_reason_covers_csha_and_wrga_variants() {
        let _csha = OverrideRejectReason::SmemBudgetExceeded { actual_kb: 52, limit_kb: 48 };
        let _wrga_clamp = OverrideRejectReason::RankClampedToBounds { r_min: 2, r_max: 16 };
        let _wrga_forbid = OverrideRejectReason::RankForbiddenByWggo;
        let _wrga_budget = OverrideRejectReason::BudgetExceededDowngraded {
            original_rank: 16, final_rank: 8,
        };
    }

    fn layer(name: &str, idx: u32, coarse: CoarseDecision) -> AppliedLayer {
        AppliedLayer {
            layer_index: idx,
            layer_name: name.to_string(),
            coarse,
            pipeline_stage: 0,
            shard_factor: 1,
            shard_grads: 1,
            shard_optim: 1,
            active_heads: 4,
            ffn_width: 2048,
            csha_level: 0,
            adapter_rank: 0,
            adapter_placement: crate::wggo_ilp::AdapterPlacement::None,
            optim_m_bits: 8,
            optim_v_bits: 16,
            fase_fused: true,
            packing_mode: 0,
            estimated_us: 10.0,
            param_bytes: 0,
            activation_bytes: 0,
        }
    }

    #[test]
    fn collect_prune_diagnostics_empty_when_no_layer_pruned() {
        let plan = AppliedPlan {
            layers: vec![
                layer("blocks.0", 0, CoarseDecision::KeepFull),
                layer("blocks.1", 1, CoarseDecision::Thin),
            ],
            total_us: 20.0,
            peak_memory_bytes: 0,
        };
        assert!(collect_prune_diagnostics(&plan).is_empty());
    }

    #[test]
    fn collect_prune_diagnostics_emits_one_per_pruned_layer() {
        let plan = AppliedPlan {
            layers: vec![
                layer("blocks.0", 0, CoarseDecision::KeepFull),
                layer("blocks.1", 1, CoarseDecision::Prune),
                layer("blocks.2", 2, CoarseDecision::Thin),
                layer("blocks.3", 3, CoarseDecision::Prune),
            ],
            total_us: 40.0,
            peak_memory_bytes: 0,
        };
        let diags = collect_prune_diagnostics(&plan);
        assert_eq!(diags.len(), 2);

        // Order matches AppliedPlan.layers iteration order.
        assert_eq!(diags[0].layer_index, 1);
        assert_eq!(diags[0].layer_name, "blocks.1");
        assert_eq!(diags[0].requested, "prune");
        assert_eq!(diags[0].applied, "keepfull");
        assert!(matches!(diags[0].reason, OverrideRejectReason::WholeBlockPruneNotImplemented));

        assert_eq!(diags[1].layer_index, 3);
        assert_eq!(diags[1].layer_name, "blocks.3");
    }

    #[test]
    fn collect_prune_diagnostics_excludes_sub_block_layers() {
        // Spec §3.6 + Task 2: sub-block prune decisions (Attention / Ffn role)
        // are routed through wggo_prune::run() and must NOT be surfaced here.
        // Only whole-block prune (LayerRole::Block) appears in this diagnostic.
        let plan = AppliedPlan {
            layers: vec![
                layer("blocks.3.attn", 0, CoarseDecision::Prune),
                layer("blocks.3.ffn",  1, CoarseDecision::Prune),
                layer("blocks.3",      2, CoarseDecision::Prune),
            ],
            total_us: 30.0,
            peak_memory_bytes: 0,
        };
        let diags = collect_prune_diagnostics(&plan);
        // Only the whole-block layer (blocks.3) should appear.
        assert_eq!(diags.len(), 1, "expected 1 diagnostic for whole-block only; got {}", diags.len());
        assert_eq!(diags[0].layer_name, "blocks.3");
    }

    #[test]
    fn whole_block_prune_not_implemented_reason_string_is_stable() {
        // The reason string is part of the externally-observable stderr
        // format `[prune] layer:N name=<N> wggo-override-rejected ...
        // reason=<S>` which future tools (decision explainer, CI log
        // scanners) may match against.  Pin the literal value here so
        // drift is caught as a test failure rather than a quiet parser
        // regression.
        assert_eq!(whole_block_prune_not_implemented_reason(), "ir_rewrite_not_implemented");
    }

    #[test]
    fn prune_diagnostic_carries_layer_name_for_stderr_format() {
        // The `[prune] layer:N name=<N>` stderr format depends on
        // `OverrideDiagnostic.layer_name` being populated (not just
        // `layer_index`).  Match the precedent the CSHA/WRGA/FASE/CPDT
        // emitters set — they all carry both fields.  Guards the
        // emitter at stmt.rs against regression if a future refactor
        // drops `layer_name` from the diagnostic construction.
        let plan = AppliedPlan {
            layers: vec![layer("h.5", 5, CoarseDecision::Prune)],
            total_us: 0.1,
            peak_memory_bytes: 0,
        };
        let diags = collect_prune_diagnostics(&plan);
        assert_eq!(diags.len(), 1);
        assert_eq!(diags[0].layer_index, 5);
        assert_eq!(diags[0].layer_name, "h.5");
    }

    // ─── PCA packing consumption (errata E2 / audit gap #4) ────────────────

    #[test]
    fn packing_mode_names_are_stable() {
        // Part of the externally-observable `--wggo-report` "PCA=" column and
        // the `[pca] ... packing_mode=<S>` stderr format — pin the literals.
        assert_eq!(packing_mode_name(0), "none");
        assert_eq!(packing_mode_name(1), "segment_id");
        assert_eq!(packing_mode_name(2), "tile_skip");
        assert_eq!(packing_mode_name(3), "multi_seq");
        assert_eq!(packing_mode_name(200), "invalid");
    }

    fn plan_with_packing(modes: &[u8]) -> AppliedPlan {
        let mut layers = Vec::new();
        for (i, &m) in modes.iter().enumerate() {
            let mut l = layer(&format!("blocks.{i}"), i as u32, CoarseDecision::KeepFull);
            l.packing_mode = m;
            layers.push(l);
        }
        AppliedPlan { layers, total_us: 1.0, peak_memory_bytes: 0 }
    }

    #[test]
    fn packing_consumed_when_plan_and_masked_kernels_agree() {
        let plan = plan_with_packing(&[1, 3]);
        for (state, kernel) in [
            (PackingKernelState::TierBMasked, "segment-masked Tier-B attention"),
            (PackingKernelState::PerDocCta, "per-document CTA attention"),
        ] {
            let diags = collect_packing_diagnostics(&plan, state);
            assert_eq!(diags.len(), 2, "one verdict per packing layer at {state:?}");
            for d in &diags {
                match &d.verdict {
                    PackingVerdict::Consumed { kernel: k } => assert_eq!(*k, kernel),
                    other => panic!("expected Consumed at {state:?}, got {other:?}"),
                }
            }
        }
    }

    #[test]
    fn packing_rejected_when_source_has_no_packed_dataset() {
        // WGGO wants packing but the module has no masked kernels: authorial
        // inputs (packed dataset / @pca) cannot be synthesized from a plan.
        let plan = plan_with_packing(&[2]);
        let diags =
            collect_packing_diagnostics(&plan, PackingKernelState::NoMaskedKernels);
        assert_eq!(diags.len(), 1);
        match &diags[0].verdict {
            PackingVerdict::Rejected(OverrideRejectReason::PackingRequiresPackedDataset {
                mode,
            }) => assert_eq!(*mode, 2),
            other => panic!("expected PackingRequiresPackedDataset, got {other:?}"),
        }
    }

    #[test]
    fn packing_mode_zero_is_refused_when_dataset_is_packed() {
        // Correctness invariant: masked kernels cannot be turned off by a plan
        // preference — unmasked attention on packed data leaks across document
        // boundaries. The plan's mode=0 is rejected, masking stays.
        let plan = plan_with_packing(&[0]);
        for state in [PackingKernelState::TierBMasked, PackingKernelState::PerDocCta] {
            let diags = collect_packing_diagnostics(&plan, state);
            assert_eq!(diags.len(), 1, "at {state:?}");
            assert!(
                matches!(
                    diags[0].verdict,
                    PackingVerdict::Rejected(
                        OverrideRejectReason::PackingMaskingMandatoryForPackedDataset
                    )
                ),
                "at {state:?}: {:?}",
                diags[0].verdict
            );
        }
    }

    #[test]
    fn packing_silent_when_no_packing_wanted_and_none_exists() {
        // mode=0 + no masked kernels = trivial agreement; no diagnostic noise.
        let plan = plan_with_packing(&[0, 0, 0]);
        assert!(collect_packing_diagnostics(&plan, PackingKernelState::NoMaskedKernels)
            .is_empty());
    }

    fn overrides_with_layers(layers: &[(&str, u32)]) -> WggoOverrides {
        WggoOverrides {
            per_layer: layers
                .iter()
                .enumerate()
                .map(|(i, (name, rank))| PerLayerOverride {
                    layer_index: i as u32,
                    layer_name: name.to_string(),
                    active_heads: 8,
                    requested_csha_level: None,
                    adapter_rank: *rank as u64,
                    adapter_placement: crate::wggo_ilp::AdapterPlacement::None,
                    fase_fused: false,
                    packing_mode: 0,
                    shard_factor: 0,
                })
                .collect(),
        }
    }

    #[test]
    fn find_by_layer_containing_matches_projections_in_layer() {
        let over = overrides_with_layers(&[("blocks.0", 8)]);
        assert_eq!(over.find_by_layer_containing("blocks.0.attn.wq").unwrap().adapter_rank, 8);
        assert_eq!(over.find_by_layer_containing("blocks.0.attn.wk").unwrap().adapter_rank, 8);
        assert_eq!(over.find_by_layer_containing("blocks.0.mlp.fc1").unwrap().adapter_rank, 8);
    }

    #[test]
    fn find_by_layer_containing_respects_dot_boundary() {
        let over = overrides_with_layers(&[("blocks.1", 4)]);
        assert!(over.find_by_layer_containing("blocks.10.attn.wq").is_none(),
            "'blocks.1' must NOT match 'blocks.10.attn.wq' — dot boundary required");
        assert_eq!(over.find_by_layer_containing("blocks.1.attn.wq").unwrap().adapter_rank, 4);
    }

    #[test]
    fn find_by_layer_containing_matches_bare_layer_name() {
        let over = overrides_with_layers(&[("blocks.2", 6)]);
        assert_eq!(over.find_by_layer_containing("blocks.2").unwrap().adapter_rank, 6);
    }

    #[test]
    fn find_by_layer_containing_returns_none_on_no_match() {
        let over = overrides_with_layers(&[("blocks.0", 8)]);
        assert!(over.find_by_layer_containing("blocks.99.attn.wq").is_none());
    }

    fn overrides_with_shards(shards: &[u32]) -> WggoOverrides {
        WggoOverrides {
            per_layer: shards.iter().enumerate().map(|(i, &s)| PerLayerOverride {
                layer_index: i as u32,
                layer_name: format!("blocks.{i}"),
                active_heads: 8,
                requested_csha_level: None,
                adapter_rank: 0,
                adapter_placement: crate::wggo_ilp::AdapterPlacement::None,
                fase_fused: false,
                packing_mode: 0,
                shard_factor: s,
            }).collect(),
        }
    }

    #[test]
    fn min_shard_factor_returns_none_when_all_zero() {
        let over = overrides_with_shards(&[0, 0, 0]);
        assert!(over.min_shard_factor().is_none());
    }

    #[test]
    fn min_shard_factor_returns_minimum_of_nonzero_values() {
        let over = overrides_with_shards(&[8, 4, 2, 8]);
        assert_eq!(over.min_shard_factor(), Some(2));
    }

    #[test]
    fn min_shard_factor_skips_zero_sentinels() {
        let over = overrides_with_shards(&[8, 0, 4, 0]);
        assert_eq!(over.min_shard_factor(), Some(4));
    }

    #[test]
    fn shard_factor_reject_variants_construct() {
        let _incompat = OverrideRejectReason::ShardFactorIncompatibleWithWorldSize {
            recommended: 4, world_size: 2,
        };
        let _mem = OverrideRejectReason::ShardFactorOverriddenByMemory {
            recommended: 2, applied: 8,
        };
    }
}
