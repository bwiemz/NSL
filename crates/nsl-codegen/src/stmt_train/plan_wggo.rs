//! The WGGO planning site of the train block's source-AD arm: reuse of
//! the wrapper's pre-plan when its tape fingerprint still matches (the
//! Muon mode table and a fresh-plan requirement refuse the reuse), the
//! global-optimization planner run under `PassScheduler::schedule`
//! (`wggo::run_on_wengert_with_weights`), the applied-plan derivation
//! (`wggo_apply`) and the `WggoOverrides` publication on the pass bus that
//! the CSHA, WRGA, FASE and CPDT sites downstream consume.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 348 lines,
//! 4 inputs ([`WggoPlanningInputs`]); pure planning — no IR is
//! emitted. Returns a [`WggoPlanning`]: the applied plan and the pre-plan
//! facts (offered / rejected) the CPDT planning site consults. The
//! train-block CLIF snapshots (`tests/train_clif_snapshots.rs`) pin the
//! lowering every plan drives.
//!
//! Named after the driver phase (`plan_wggo`) rather than `wggo_*`: the
//! pass-bus drift gate classifies a `wggo_*` stem as one of the WGGO pass's
//! own modules, while this is the driver's planning SITE — its
//! `wggo_preplans` read is the driver-mediated edge the channel inventory
//! already declares, not a pass-to-pass dependency.

use cranelift_codegen::ir::Value;

use crate::compiler::Compiler;
use crate::error::CodegenError;

/// Every binding of `compile_train_block_inner` the WGGO planning reads;
/// names are the driver's.
pub(crate) struct WggoPlanningInputs<'a> {
    /// The forward extractor (the planner and the pre-plan fingerprint read its Wengert list).
    pub(crate) extractor: &'a crate::source_ad::WengertExtractor<'a>,
    /// The Muon mode-table base, when one was allocated (a pre-plan cannot be reused over it).
    pub(crate) mode_table_base: Option<Value>,
    /// The train block (the planner reads its optimizer and schedule facts).
    pub(crate) train: &'a nsl_ast::block::TrainBlock,
    /// The train block statement id the wrapper keyed its pre-plan by.
    pub(crate) train_block_stmt_id: nsl_ast::NodeId,
}

/// What the WGGO planning hands back to the driver.
pub(crate) struct WggoPlanning {
    /// The applied plan (`None` when WGGO is off or declined).
    pub(crate) applied: Option<crate::wggo_apply::AppliedPlan>,
    /// Whether a pre-plan existed for this block (the wrapper already
    /// offered it to CPDT before the moment consult).
    pub(crate) preplan_offered: bool,
    /// Whether the fingerprint check rejected that pre-plan.
    pub(crate) preplan_was_rejected: bool,
}

impl Compiler<'_> {
    /// Run the WGGO planning site (see the module header).
    pub(crate) fn plan_wggo(
        &mut self,
        inputs: WggoPlanningInputs<'_>,
    ) -> Result<WggoPlanning, CodegenError> {
        let WggoPlanningInputs {
            extractor,
            mode_table_base,
            train,
            train_block_stmt_id,
        } = inputs;

        // WGGO: run the global optimization planner if enabled.  The
        // planner call itself is pure data-in/data-out — it produces a
        // plan of globally-optimal per-layer decisions.  Several of
        // those decisions are now lowered downstream: sub-block prune
        // rewrites the Wengert list below (via `wggo_prune::run`), and
        // the resulting `WggoOverrides` drive CSHA fusion level, WRGA
        // adapter placement, FASE fused-step, and the CPDT shard factor.
        // Some decisions remain advisory/report-only for now (WGGO-side
        // PCA packing_mode, per-layer optimizer precision, whole-block
        // prune, and CFIE inference decisions).
        //
        let mut wggo_applied: Option<crate::wggo_apply::AppliedPlan> = None;
        // Whether a pre-plan existed for this block (=> the wrapper
        // already offered it to CPDT before the moment consult) and
        // whether the fingerprint check rejected it. Hoisted out of
        // the WGGO block for the CPDT planning site below, which
        // must not re-plan what the wrapper planned, and must refuse
        // when the consult consumed dtype lists a rejection made
        // stale.
        let mut wggo_preplan_offered = false;
        let mut wggo_preplan_was_rejected = false;
        // Owned, not `ref`: the scheduled closure below captures
        // `self` mutably (bus publish), so a `ref` pattern holding a
        // shared borrow of `self.compile_options` across it would not
        // borrow-check.
        let wggo_mode = self.compile_options.wggo.mode.clone();
        if let Some(mode_str) = wggo_mode
            && mode_str != "off" && mode_str != "disable" && mode_str != "disabled"
        {
            // Milestone C: SCHEDULED, both arms of the pre-plan
            // match — on the fingerprint-match arm the pass body
            // never runs (recorded at KernelPrepass), but the
            // scheduler still retains the digest of the tape the
            // reused plan's positional indices will PRUNE, which
            // is exactly what the consumption-fork assert below
            // needs (TapeDigest and fingerprint_wengert are
            // different hashes, not interchangeable). The body is
            // widened through the publish: wggo_overrides
            // declares applied_implies_published Enforced and
            // Applied is recorded inside wggo::run, so a finish
            // directly after the planner call would refuse every
            // correct compile. The stale-mode-table refusal
            // returns THROUGH the closure (R is a Result).
            let sched = self.passes.scheduler();
            let scheduled = sched
                .schedule("WGGO", Some(extractor.wengert_list()), || {
            // Build AnalysisConfig from CLI overrides; clamp is
            // also applied in analyze(), but applying it here
            // keeps the --wggo-report line honest.
            let mut analysis_config =
                crate::wggo_weight_analysis::AnalysisConfig::default();
            if let Some(f) = self.compile_options.wggo.prune_fraction {
                analysis_config.default_prune_fraction = f.clamp(0.0, 0.9);
            }
            // Pass the weights path for magnitude-based scoring
            // (NullWeightProvider is used in run_on_wengert_with_weights
            // when weights_path is None, producing uniform scores).
            // compile_options is forwarded so build_scorer can wire the
            // GradientScorer appropriate for --wggo-importance + --calibration-data.
            // calibration.sidecar is populated by compile_and_calibrate's wrapper-
            // level firing BEFORE compile_main runs, ensuring it's available here
            // when build_scorer reads it (see #134 (c-i) and lib.rs's compile_and_
            // calibrate wrapper).
            let weights_path = self.compile_options.wggo.weights.as_deref();
            // WGGO-before-kernels: consume this block's pre-plan
            // when its graph fingerprint matches the extraction
            // we just did (the list codegen actually lowers, and
            // that wggo_prune may rewrite — indices must refer to
            // THIS graph). On mismatch, reject loudly and solve
            // in place; if the fresh plan then disagrees with the
            // pre-plan on any FASE-relevant decision AND a
            // per-param mode table was already emitted from the
            // pre-plan's overrides earlier in this function,
            // HARD-REFUSE (P0 item 1): executing the stale table
            // would train with FASE modes that do not match the
            // final plan every downstream consumer sees.
            let preplan = self
                .bus
                .wggo_preplans()
                .iter()
                .find(|p| p.train_block_stmt_id == train_block_stmt_id);
            let reused_plan = preplan.and_then(|pre| {
                let fp = crate::wggo_prepass::fingerprint_wengert(
                    extractor.wengert_list(),
                );
                if fp == pre.graph_fingerprint {
                    // Additive observability line (tests key off
                    // it); the [wggo] summary itself still prints
                    // below, identically to the in-place path.
                    nsl_log::nsl_log!(INFO, "wggo", 
                        "[wggo] consumed pre-solved plan \
                             (graph fingerprint match)"
                    );
                    Some(pre.plan.clone())
                } else {
                    nsl_log::nsl_log!(WARN, "wggo", 
                        "[wggo] wggo-preplan-rejected \
                             reason=graph_fingerprint_mismatch — replanning in place"
                    );
                    None
                }
            });
            let preplan_was_rejected =
                preplan.is_some() && reused_plan.is_none();
            wggo_preplan_offered = preplan.is_some();
            wggo_preplan_was_rejected = preplan_was_rejected;
            let plan = match reused_plan {
                Some(plan) => Some(plan),
                None => crate::wggo::run_on_wengert_with_weights(
                    extractor.wengert_list(),
                    &self.compile_options.target,
                    &mode_str,
                    self.compile_options.world_size,
                    weights_path,
                    analysis_config,
                    Some(&self.compile_options),
                    self.features.packing_supported_in_module,
                    // Campaign item 6: same doc-length stats the
                    // pre-pass used (resolved in kernel synthesis),
                    // so an in-place replan prices packing from the
                    // real distribution too.
                    self.features.dataset_packing_stats.clone(),
                )
                .map_err(|e| {
                    e.with_span_if_unset(crate::wggo_prepass::model_arg_span(
                        train,
                        self.interner,
                    ))
                })?,
            };
            // Test-only knob: simulate a rejected pre-plan whose
            // in-place replan diverges on FASE decisions, so the
            // refusal wiring is gate-testable without engineering
            // a real graph-fingerprint drift. Strict value match
            // ("1"), same convention as NSL_FASE_FUSED_OVERRIDE.
            let forced_stale = std::env::var("NSL_WGGO_FORCE_STALE_TABLE")
                .map(|v| v == "1")
                .unwrap_or(false);
            if preplan_was_rejected || forced_stale {
                let fase_diverged = match (preplan, plan.as_ref()) {
                    (Some(pre), Some(fresh)) => {
                        crate::wggo_overrides::fase_overrides_diverge(
                            &pre.overrides,
                            &crate::wggo_overrides::WggoOverrides::from_applied(
                                &fresh.applied,
                            ),
                        )
                    }
                    _ => false,
                } || forced_stale;
                if fase_diverged {
                    if mode_table_base.is_some() {
                        return Err(CodegenError::new(
                            "the FASE per-param mode table for this train \
                                 block was emitted from a WGGO pre-plan whose \
                                 graph fingerprint no longer matches, and the \
                                 in-place replan DISAGREES on per-layer \
                                 fase_fused — refusing to execute a stale mode \
                                 table (the accumulation/optimizer dispatch \
                                 would not match the final WGGO plan). \
                                 Recompile so the pre-plan regenerates against \
                                 the current graph, or drop --wggo for this \
                                 block.",
                        ));
                    }
                    // No mode table was emitted (Passthrough /
                    // FullBuffer-global / muon): nothing stale
                    // executes — downstream consumers get the
                    // fresh plan. Note it loudly anyway.
                    nsl_log::nsl_log!(INFO, "wggo", 
                        "[wggo] note: the rejected pre-plan and the \
                             in-place replan disagree on fase_fused, but no \
                             per-param FASE mode table was emitted for this \
                             train block — the fresh plan governs all \
                             downstream consumers"
                    );
                }
            }
            if let Some(plan) = plan {
                if self.compile_options.wggo.report {
                    nsl_log::nsl_log!(INFO, "codegen", "{}", plan.render_report());
                } else {
                    nsl_log::nsl_log!(INFO, "wggo", "[wggo] {}", plan.summary());
                }
                // Prune consumer (diagnostic stub): WGGO's DP can
                // emit `CoarseDecision::Prune` for low-importance
                // layers, but no downstream codegen implements
                // the layer-to-residual-identity IR rewrite.
                // Surface the gap via the `[prune]` stderr
                // diagnostic matching the CSHA/WRGA/CPDT/FASE
                // pattern, so users and the future IR-rewrite
                // session can see the planner's intent instead
                // of the decision silently no-opping.  Empty
                // iterator when no layer is planned for pruning
                // (shipped-binary common case).
                for diag in crate::wggo_overrides::collect_prune_diagnostics(&plan.applied) {
                    // Dispatch through `diag.reason` rather than
                    // calling the reason-string helper directly,
                    // so a future Prune-adjacent reason variant
                    // automatically renders its own string and
                    // this match statement surfaces the missing
                    // case at compile time via a non-exhaustive
                    // warning (or new arm) rather than silently
                    // printing the wrong string.
                    let reason_str: std::borrow::Cow<'static, str> = match &diag.reason {
                        crate::wggo_overrides::OverrideRejectReason::WholeBlockPruneNotImplemented => {
                            std::borrow::Cow::Borrowed(
                                crate::wggo_overrides::whole_block_prune_not_implemented_reason(),
                            )
                        }
                        other => std::borrow::Cow::Owned(format!("{:?}", other)),
                    };
                    nsl_log::nsl_log!(INFO, "prune", 
                        "[prune] layer:{} name={} wggo-override-rejected \
                             requested={} applied={} reason={}",
                        diag.layer_index,
                        diag.layer_name,
                        diag.requested,
                        diag.applied,
                        reason_str,
                    );
                }
                // PCA packing consumption (errata E2 / audit gap #4):
                // validate the plan's per-layer packing_mode against
                // the attention kernels the module-scan emitter
                // actually synthesized (it ran before compile_main,
                // so the plan could not influence admission — the
                // ordering restructure that would let it is the
                // tracked follow-up). One `[pca] layer:N
                // wggo-override-consumed/rejected` line per layer,
                // matching the CSHA/WRGA/CPDT/FASE/prune pattern.
                {
                    use crate::wggo_overrides::{
                        packing_mode_name, PackingKernelState, PackingVerdict,
                    };
                    let state = match self
                        .kernels
                        .flash_attention_context
                        .as_ref()
                        .and_then(|c| c.csha_training_config.as_ref())
                    {
                        Some(cfg) if cfg.segment_masked => {
                            // Per-doc CTA replaces the Tier-B pair at
                            // emission (compiler/kernel.rs fork), so
                            // masked + no Tier-B IDs ⇔ per-doc active.
                            if self
                                .kernels
                                .flash_attention_context
                                .as_ref()
                                .is_some_and(|c| {
                                    c.csha_with_saves_tier_b_on_ptx_id.is_some()
                                })
                            {
                                PackingKernelState::TierBMasked
                            } else {
                                PackingKernelState::PerDocCta
                            }
                        }
                        // No fused masked kernels — but a model
                        // that consumes the packed mask at the
                        // source level (Stage B masked SDPA)
                        // honors the plan's segment_id preference
                        // itself; only report a rejection when
                        // NEITHER channel exists.
                        // PCA Stage C: the packed builtin on a
                        // CUDA target upgrades the consumption
                        // channel to the fused segment-masked
                        // family (decline path = Stage B chain).
                        _ if self.features.packed_sdpa_in_module
                            && (self.compile_options.target == "cuda"
                                || self.compile_options.target.starts_with("sm_")) =>
                        {
                            PackingKernelState::FusedSegmentMasked
                        }
                        _ if self.features.packing_supported_in_module => {
                            PackingKernelState::SourceMasked
                        }
                        _ => PackingKernelState::NoMaskedKernels,
                    };
                    for diag in crate::wggo_overrides::collect_packing_diagnostics(
                        &plan.applied,
                        state,
                    ) {
                        match &diag.verdict {
                            PackingVerdict::Consumed { kernel } => nsl_log::nsl_log!(INFO, "pca", 
                                "[pca] layer:{} name={} wggo-override-consumed \
                                     packing_mode={} -> {}",
                                diag.layer_index,
                                diag.layer_name,
                                packing_mode_name(diag.mode),
                                kernel,
                            ),
                            PackingVerdict::Rejected(reason) => {
                                // Space-free snake_case token, like every
                                // other consumer's reason string (the
                                // decision explainer splits on whitespace;
                                // `{:?}` of a struct variant would inject
                                // `{ mode: N }` tokens). The requested
                                // mode already rides in `requested=`.
                                let reason_token = match reason {
                                    crate::wggo_overrides::OverrideRejectReason::PackingRequiresPackedDataset { .. } =>
                                        "packing_requires_packed_dataset",
                                    crate::wggo_overrides::OverrideRejectReason::PackingMaskingMandatoryForPackedDataset =>
                                        "packing_masking_mandatory_for_packed_dataset",
                                    other => {
                                        debug_assert!(
                                            false,
                                            "non-packing reject reason in packing verdict: {other:?}"
                                        );
                                        "unexpected_packing_reject_reason"
                                    }
                                };
                                nsl_log::nsl_log!(INFO, "pca", 
                                    "[pca] layer:{} name={} wggo-override-rejected \
                                         requested={} applied={} reason={}",
                                    diag.layer_index,
                                    diag.layer_name,
                                    packing_mode_name(diag.mode),
                                    match state {
                                        PackingKernelState::NoMaskedKernels => "unmasked",
                                        PackingKernelState::SourceMasked =>
                                            "source_masked",
                                        PackingKernelState::FusedSegmentMasked =>
                                            "fused_segment_masked",
                                        PackingKernelState::TierBMasked =>
                                            "segment_masked",
                                        PackingKernelState::PerDocCta => "per_doc_cta",
                                    },
                                    reason_token,
                                );
                            }
                        }
                    }
                }
                // Stash for all downstream consumers (CSHA, WRGA, ...).
                self.bus.publish_wggo_overrides(
                    crate::wggo_overrides::WggoOverrides::from_applied(&plan.applied),
                );
                Ok(Some(plan.applied))
            } else {
                Ok(None)
            }
                })
                .map_err(CodegenError::new)?;
            wggo_applied = scheduled
                .finish(&self.bus)
                .map_err(CodegenError::new)??;
        }

        Ok(WggoPlanning {
            applied: wggo_applied,
            preplan_offered: wggo_preplan_offered,
            preplan_was_rejected: wggo_preplan_was_rejected,
        })
    }
}
