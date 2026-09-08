//! The WRGA driver and the CPDT planning site of the train block's
//! source-AD arm: the WRGA driver run under `PassScheduler::schedule`
//! (pruning / rank allocation / fusion over the extractor's list; a no-op
//! without decorator inputs), then the CPDT planning site — the tier
//! agreement with the wrapper's pre-plan, the optimizer-moment precision
//! arbitration against the lists the moment allocation consumed, the
//! stale-plan refusal, and the no-WGGO skip notice.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 294 lines,
//! 8 inputs ([`WrgaCpdtInputs`]); pure planning — no IR is
//! emitted. Returns the WRGA plan (`None` when WRGA is off or has no
//! inputs) that the pre-forward phases fork the effective primal onto. The
//! train-block CLIF snapshots (`tests/train_clif_snapshots.rs`) pin the
//! lowering every plan drives.
//!
//! Named after the driver phase (`plan_wrga_cpdt`) rather than `wrga_*` /
//! `cpdt_*`: the pass-bus drift gate classifies those stems as the passes'
//! own modules, while this is the driver's planning SITE.

use crate::compiler::Compiler;
use crate::error::CodegenError;

/// Every binding of `compile_train_block_inner` the WRGA driver and the
/// CPDT planning read; names are the driver's.
pub(crate) struct WrgaCpdtInputs<'a> {
    /// The optimizer-moment dtype lists the moment allocation consumed (the CPDT stale-plan check compares against them).
    pub(crate) cpdt_moment_lists_consumed: &'a Option<(Vec<u16>, Vec<u16>)>,
    /// The forward extractor (the WRGA driver scans its Wengert list).
    pub(crate) extractor: &'a crate::source_ad::WengertExtractor<'a>,
    /// Whether the FASE-Deferred schedule is active (a CPDT precondition).
    pub(crate) fase_deferred: bool,
    /// The model tensor paths in `param_list` order (the CPDT tier assignment walks them).
    pub(crate) param_paths: &'a [String],
    /// The train block (the CPDT planner reads its optimizer facts).
    pub(crate) train: &'a nsl_ast::block::TrainBlock,
    /// The applied WGGO plan, when one exists (its shard factor feeds CPDT).
    pub(crate) wggo_applied: &'a Option<crate::wggo_apply::AppliedPlan>,
    /// Whether the wrapper already offered a pre-plan to CPDT before the moment consult.
    pub(crate) wggo_preplan_offered: bool,
    /// Whether the fingerprint check rejected that pre-plan (the consult may then be stale).
    pub(crate) wggo_preplan_was_rejected: bool,
}

impl Compiler<'_> {
    /// Run the WRGA driver and the CPDT planning site (see the module
    /// header); returns the WRGA plan.
    pub(crate) fn run_wrga_and_plan_cpdt(
        &mut self,
        inputs: WrgaCpdtInputs<'_>,
    ) -> Result<Option<crate::wrga::WrgaPlan>, CodegenError> {
        let WrgaCpdtInputs {
            cpdt_moment_lists_consumed,
            extractor,
            fase_deferred,
            param_paths,
            train,
            wggo_applied,
            wggo_preplan_offered,
            wggo_preplan_was_rejected,
        } = inputs;

        // Task 4: invoke WRGA driver (pruning / rank allocation /
        // fusion) before primal lowering.  When WRGA is disabled or
        // inputs are empty, this is a no-op and we use the raw
        // extractor list.
        //
        // Item 2 step 7: this invocation is SCHEDULED, not called. The
        // manager checks WRGA's registry-declared phase and its
        // bus-declared InvocationOrdered predecessor before the body
        // runs, and its applied-implies-published channel after —
        // then retains a digest of the tape WRGA scanned, so the
        // `effective_primal` fork below can prove the plan's
        // positional references still refer to this list. WRGA is the
        // first pass routed this way; the rest still call directly.
        let sched = self.passes.scheduler();
        let scheduled = sched
            .schedule("WRGA", Some(extractor.wengert_list()), || {
                crate::stmt::invoke_wrga_if_enabled(self, extractor.wengert_list())
            })
            .map_err(CodegenError::new)?;
        let wrga_plan = scheduled.finish(&self.bus).map_err(CodegenError::new)?;
        // CPDT planning site. On the pre-plan path the wrapper
        // already planned before the moment consult (a plan
        // published only HERE arrives ~2.2k lines after that consult
        // allocated the moments, which kept CPDT-sourced moment
        // precision structurally inert), so this site plans only
        // when the wrapper could not — no pre-plan — or must not be
        // trusted — fingerprint-rejected pre-plan. Either way it
        // then compares the CPDT-sourced dtype lists the consult
        // consumed against the current plan: matching lists proceed,
        // a consumed-but-stale list refuses (the moments are
        // allocated; their dtypes cannot be re-derived), and a plan
        // that arrives too late to lower says so instead of
        // silently training with FP32 moments.
        if let Some(applied) = wggo_applied {
            if !wggo_preplan_offered || wggo_preplan_was_rejected {
                // Milestone C: SCHEDULED, inside the existing guard —
                // wrapping the guard itself would trace a scheduled
                // no-invocation on every pre-plan-path compile.
                sched
                    .schedule("CPDT", None, || {
                        crate::stmt::invoke_cpdt_if_enabled(
                            self,
                            Some(applied),
                            Some(train),
                        )
                    })
                    .map_err(CodegenError::new)?
                    .finish(&self.bus)
                    .map_err(CodegenError::new)??;
            }
            // Re-arbitrate what the moments WOULD be typed as under
            // the CURRENT plan and overrides — the same pipeline the
            // consult ran (WGGO bits, CPDT lists, opt-in), against
            // the fresh bus state. Comparing final-vs-final rather
            // than offer-vs-offer is what keeps this from refusing a
            // correct compile whose arbitration dropped or merged
            // the CPDT offer, and it also covers a replan that
            // changed WGGO's moment-bit decisions — a divergence the
            // FASE mode-table refusal does not look at.
            let fresh_lists: Option<(Vec<u16>, Vec<u16>)> = {
                let fresh_wggo_bits = if fase_deferred {
                    self.bus.wggo_overrides().and_then(|o| {
                        crate::cpdt_precision_exec::build_dtype_lists_from_overrides(
                            o,
                            param_paths,
                        )
                    })
                } else {
                    None
                };
                let fresh_cpdt = self
                    .bus
                    .cpdt_plan()
                    .filter(|p| {
                        crate::cpdt_precision_exec::precision_active(
                            matches!(p.mode, crate::cpdt::CpdtMode::Full),
                            !p.precision.params.is_empty(),
                            true,
                            fase_deferred,
                            true,
                        )
                    })
                    .map(|p| {
                        crate::cpdt_precision_exec::build_dtype_lists(
                            &p.precision,
                            param_paths,
                        )
                    });
                use crate::cpdt_precision_exec::{
                    arbitrate_moment_precision, MomentPrecisionArbitration as MPA,
                };
                // `arbitrate_moment_precision` is pure; the consult's
                // diagnostics print at ITS call site, so re-running
                // it here is silent by construction.
                match arbitrate_moment_precision(
                    fresh_wggo_bits,
                    fresh_cpdt,
                    self.compile_options.wggo.moment_precision,
                ) {
                    MPA::Merged(m, v) | MPA::WggoOnly(m, v) | MPA::CpdtOnly(m, v) => {
                        Some((m, v))
                    }
                    MPA::NotLoweredNoOptIn | MPA::Inactive => None,
                }
            };
            // Test-only knob, same convention as
            // NSL_WGGO_FORCE_STALE_TABLE: force the divergence arm so
            // the refusal is gate-testable without engineering a real
            // fingerprint drift.
            let forced_stale_plan = crate::stmt::cpdt_forced_stale_plan();
            match (cpdt_moment_lists_consumed, &fresh_lists) {
                (Some(consumed), fresh)
                    if fresh.as_ref() != Some(consumed) || forced_stale_plan =>
                {
                    // Two honest causes, one conservative outcome. The
                    // moments are allocated; their dtypes cannot be
                    // re-derived, so a divergent final arbitration can
                    // only refuse. But WHICH offer went stale differs:
                    // a fingerprint-rejected pre-plan, or — on a block
                    // that never had one — the weights-only offer made
                    // before this block's in-place WGGO plan (and its
                    // moment-bit decisions) existed. Naming the wrong
                    // one sends the user at the wrong artifact.
                    return Err(CodegenError::new(if wggo_preplan_offered {
                        "the optimizer moments were allocated from \
                         dtype decisions derived from a WGGO pre-plan \
                         whose graph fingerprint no longer matches, \
                         and re-arbitrating under the fresh plan \
                         DISAGREES with the allocated moment dtypes — \
                         refusing to execute a stale precision plan. \
                         Recompile so the pre-plan regenerates against \
                         the current graph, or drop --cpdt / \
                         --wggo-moment-precision for this block."
                    } else {
                        // Remedy check (review finding 3): the two
                        // advised here are each FOLLOWABLE — dropping
                        // --wggo removes the in-place plan so the
                        // weights-only offer IS the final
                        // arbitration, and dropping --cpdt removes
                        // the offer. Do NOT advise dropping
                        // --wggo-moment-precision: without it a real
                        // divergence arbitrates to NotLoweredNoOptIn
                        // (fresh = None ≠ consumed), so that advice
                        // could never resolve the refusal.
                        "the optimizer moments were typed from the \
                         weights-only CPDT offer made before this \
                         block's in-place WGGO plan existed, and \
                         re-arbitrating under the final plan (which \
                         now sees WGGO's moment-bit decisions) \
                         DISAGREES with the allocated moment dtypes — \
                         refusing to execute a stale precision plan. \
                         This block's precision cannot be settled \
                         before its in-place WGGO plan exists: drop \
                         --wggo so the weights-only offer is the \
                         final arbitration, or drop --cpdt."
                    }));
                }
                (None, Some(_)) => {
                    // ROUTINELY reachable, not hypothetical: distill
                    // blocks and loop-bound train blocks get no
                    // pre-plan (the prepass walks only plain train
                    // blocks). CPDT-sourced precision now reaches them
                    // through the weights-only pre-body offer, but
                    // WGGO-SOURCED moment bits still cannot: the
                    // in-place plan is born at this site, after the
                    // moments were allocated. That residual gap is
                    // what this arm reports.
                    nsl_runtime::nsl_log!(INFO, "cpdt", 
                        "[cpdt] optimizer-moment precision NOT fully \
                         lowered: WGGO's in-place plan (and its \
                         moment-bit decisions) arrived after the \
                         moments were allocated (no usable WGGO \
                         pre-plan for this block). Moments stay FP32."
                    );
                }
                _ => {}
            }
        } else if cpdt_moment_lists_consumed.is_some() && wggo_preplan_offered {
            // The hole a review closed: the pre-plan was offered, the
            // moments were typed from it, the fingerprint rejected it
            // — and the in-place replan itself returned None (scorer
            // build failure, provably-incompatible shapes). The
            // moment dtypes are baked from a plan the compile just
            // declared stale, and there is no fresh plan to
            // re-arbitrate against; printing the "no CPDT decisions
            // apply" notice here would be factually false.
            return Err(CodegenError::new(
                "the optimizer moments were allocated from dtype \
                 decisions derived from a WGGO pre-plan whose graph \
                 fingerprint no longer matches, and the in-place \
                 replan produced no plan at all — refusing to \
                 execute a stale precision plan. Recompile so the \
                 pre-plan regenerates against the current graph, or \
                 drop --cpdt for this block.",
            ));
        } else if let Some(consumed) = cpdt_moment_lists_consumed {
            // The weights-only path: no WGGO plan ever existed for
            // this block (no pre-plan, and the in-place site produced
            // none), and the moments were typed from the pre-body
            // weights-only CPDT offer. Nothing has re-planned since,
            // so this normally matches by construction — but the
            // moments are allocated, so the same final-vs-final
            // discipline applies as on the pre-plan path: re-derive
            // from the current bus state and refuse on divergence
            // rather than assume it.
            let fresh_weights_only: Option<(Vec<u16>, Vec<u16>)> = {
                use crate::cpdt_precision_exec::{
                    arbitrate_moment_precision, MomentPrecisionArbitration as MPA,
                };
                let fresh_cpdt = self
                    .bus
                    .cpdt_plan()
                    .filter(|p| {
                        crate::cpdt_precision_exec::precision_active(
                            matches!(p.mode, crate::cpdt::CpdtMode::Full),
                            !p.precision.params.is_empty(),
                            true,
                            fase_deferred,
                            true,
                        )
                    })
                    .map(|p| {
                        crate::cpdt_precision_exec::build_dtype_lists(
                            &p.precision,
                            param_paths,
                        )
                    });
                match arbitrate_moment_precision(
                    None,
                    fresh_cpdt,
                    self.compile_options.wggo.moment_precision,
                ) {
                    MPA::Merged(m, v) | MPA::WggoOnly(m, v) | MPA::CpdtOnly(m, v) => {
                        Some((m, v))
                    }
                    MPA::NotLoweredNoOptIn | MPA::Inactive => None,
                }
            };
            let forced_stale_plan = crate::stmt::cpdt_forced_stale_plan();
            if fresh_weights_only.as_ref() != Some(consumed) || forced_stale_plan {
                return Err(CodegenError::new(
                    "the optimizer moments were typed from the \
                     weights-only CPDT offer and re-deriving that \
                     offer from the current plan DISAGREES with the \
                     allocated moment dtypes — refusing to execute a \
                     stale precision plan. Recompile, or drop --cpdt \
                     for this block.",
                ));
            }
        } else if self.cpdt_mode != crate::cpdt::CpdtMode::Off
            && self.cpdt_cluster.is_some()
        {
            // CPDT was requested but nothing lowered. Name the real
            // provenance (review findings: the first wording said
            // "planned weights-only" on a path where the plan came
            // from a pre-plan offer, and claimed planning happened
            // on feature-off builds where the bridge no-ops).
            if !cfg!(feature = "experimental-cpdt") {
                nsl_runtime::nsl_log!(INFO, "cpdt", 
                    "[cpdt] requested, but this build omits the \
                     experimental-cpdt feature — no planning ran. \
                     No CPDT decisions apply to this block."
                );
            } else if wggo_preplan_offered {
                // Reachable: pre-plan offered, fingerprint rejected,
                // in-place replan produced no plan, and the consult
                // consumed nothing (e.g. the pre-plan's moment bits
                // arbitrated to NotLoweredNoOptIn).
                nsl_runtime::nsl_log!(INFO, "cpdt", 
                    "[cpdt] optimizer-moment precision not active \
                     for this block (planned from the WGGO pre-plan \
                     offer; the fingerprint rejected it and the \
                     in-place replan produced no plan): arbitration \
                     lowered nothing. No CPDT decisions apply to \
                     this block."
                );
            } else {
                // The weights-only offer ran and arbitration
                // lowered nothing: empty precision plan (zero_only
                // mode, weight-aware opt-out, no sub-32 decisions)
                // or no FASE-Deferred envelope (grad accumulation
                // < 2). The pre-#470 wording claimed CPDT
                // "requires a WGGO plan", which stopped being true
                // when the weights-only offer landed.
                nsl_runtime::nsl_log!(INFO, "cpdt", 
                    "[cpdt] optimizer-moment precision not active \
                     for this block (planned weights-only; no WGGO \
                     plan): arbitration lowered nothing. Check \
                     --cpdt full, --weights, and grad accumulation \
                     >= 2 (FASE-Deferred). No CPDT decisions apply \
                     to this block."
                );
            }
        }

        Ok(wrga_plan)
    }
}
