//! The WRGA fork and the CCR planning of the train block's source-AD arm:
//! the positional-reference guard (the WRGA plan indexes the list state it
//! was scanned from), the fork of the extractor's list onto that plan —
//! `effective_primal`, the forward every later phase reads — and the CCR
//! plan under `PassScheduler::schedule`: the checkpoint blocks, the stride
//! (fixed, `dp` or `auto`), the VRAM budget's save/recompute flips, the
//! compressed saves and the owned-tensor restriction the forward's
//! ownership classification must later confirm.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 390 lines,
//! 5 inputs ([`PreForwardPlanInputs`]); pure planning — no IR is
//! emitted. Returns a [`PreForwardPlans`]: the effective primal, the CCR
//! plan (`None` when checkpointing is off or declined), the next fresh
//! VarId past both tapes and the compressed-save map. The train-block CLIF
//! snapshots (`tests/train_clif_snapshots.rs`) pin the lowering every plan
//! drives, checkpointed fixtures included.
//!
//! Named after the driver phase (`plan_ccr`) rather than `ccr_*`: the
//! pass-bus drift gate classifies a `ccr_*` stem as one of the CCR pass's
//! own modules, while this is the driver's planning SITE.

use crate::compiler::Compiler;
use crate::error::CodegenError;

/// Every binding of `compile_train_block_inner` the WRGA fork and the CCR
/// planning read; names are the driver's.
pub(crate) struct PreForwardPlanInputs<'a> {
    /// Whether the CSLA window schedule is active (it leans on a CCR plan).
    pub(crate) csla_active: bool,
    /// The forward extractor (the WRGA fork reads its list; the CCR plan starts its fresh VarIds after it).
    pub(crate) extractor: &'a crate::source_ad::WengertExtractor<'a>,
    /// The accumulation window (the CCR budget accounts per-step).
    pub(crate) grad_accumulation_steps: i64,
    /// The pass scheduler handle the driver took before the pipeline (the WRGA guard re-digests through it).
    pub(crate) sched: crate::pass_manager::PassScheduler,
    /// The WRGA plan; `Some` forks the effective primal onto it.
    pub(crate) wrga_plan: &'a Option<crate::wrga::WrgaPlan>,
}

/// What the WRGA fork and the CCR planning hand back to the driver.
pub(crate) struct PreForwardPlans {
    /// The forward tape every later phase reads: the extractor's list
    /// forked onto the WRGA plan, or the list itself.
    pub(crate) effective_primal: crate::wengert::WengertList,
    /// The CCR plan (`None` when checkpointing is off or declined).
    pub(crate) ccr_plan: Option<crate::ccr::CcrPlan>,
    /// The next fresh VarId, advanced past the CCR clones.
    pub(crate) ccr_fresh: crate::wengert::VarId,
    /// The compressed-save map (`--checkpoint-compress`).
    pub(crate) ccr_compress_map: std::collections::HashMap<crate::wengert::VarId, crate::wengert::VarId>,
}

impl Compiler<'_> {
    /// Fork the forward onto the WRGA plan and plan CCR (see the module
    /// header).
    pub(crate) fn fork_wrga_and_plan_ccr(
        &mut self,
        inputs: PreForwardPlanInputs<'_>,
    ) -> Result<PreForwardPlans, CodegenError> {
        let PreForwardPlanInputs {
            csla_active,
            extractor,
            grad_accumulation_steps,
            sched,
            wrga_plan,
        } = inputs;

        // Item 2 step 7: WRGA's plan is consumed HERE, several hundred
        // lines after the scan that produced it, and it holds
        // `TapeRef::PositionalIndex` (declared in `pass_registry`).
        // Positional references are valid only against the list state
        // they were captured from, so if anything moved the extractor's
        // list in between, forking onto this plan indexes the wrong
        // ops. Nothing checked that until the pass was scheduled.
        // Only when a plan actually exists. The `None` arm below clones
        // the extractor list directly — no positional references are
        // being consumed, so there is nothing to invalidate, and
        // asserting there would refuse builds with WRGA entirely off
        // (the common case) for a mutation that harmed nobody.
        if wrga_plan.is_some() {
            sched
                .assert_tape_unchanged_since("WRGA", extractor.wengert_list())
                .map_err(CodegenError::new)?;
        }
        let effective_primal: crate::wengert::WengertList = match wrga_plan {
            Some(plan) => plan.prune.pruned.clone(),
            None => extractor.wengert_list().clone(),
        };

        // Dev-tools paper completion: snapshot the REAL train-block
        // artifacts for `nsl profile`'s real path. The slot is
        // installed only by `compile_with_profile_captures` (None in
        // normal builds), and lives OUTSIDE the compiler so the
        // snapshot survives downstream codegen errors (e.g.
        // unresolved optimizer stdlib symbols on minimal compiles).
        if let Some(slot) = self.profile_capture_slot.clone() {
            let size_hints = crate::profiling::captures::size_hints_from_var_nodes(
                extractor.var_nodes(),
                self.type_map,
            );
            *slot.borrow_mut() = Some(crate::profiling::captures::ProfileCaptures {
                train_wengert: Some(effective_primal.clone()),
                var_size_hints: size_hints,
                fusion: wrga_plan.as_ref().map(|p| p.fusion.clone()),
            });
        }

        // WRGA B.1 Task 4: feed MemoryPlan.assignments to the real
        // memory planner as coalescing hints.  Conservative: only
        // merges pairs that pass size + liveness-disjoint checks.
        // B.2 Task 3: when `--wrga-fold-allocations` is set, also
        // run the typed-key `consume_hints` path on a transient
        // allocator so the side-channel counter bumps. The real
        // folding into the production allocator is wired in later
        // tasks; this branch exists so the flag has observable
        // effect today.
        if let Some(plan) = wrga_plan {
            if self.compile_options.wrga.fold_allocations {
                let mut transient = crate::memory_planner::LivenessAnalyzer::new();
                for a in &plan.memory.assignments {
                    transient.record_activation_alloc(a.var, a.size_bytes);
                }
                let _ = crate::memory_planner::consume_hints(&mut transient, plan);
            } else {
                let _ = crate::memory_planner::apply_wrga_hints(plan);
            }
        }

        // CCR P1.a (--checkpoint-blocks): segment the final primal
        // tape and decide the per-block recompute set BEFORE any
        // lowering. Reads the CSHA claim table non-destructively
        // (the generator `take()`s it later) so claimed segments
        // can be exempted — the claim table is keyed by primal
        // OpId, which a recompute clone cannot satisfy.
        // P1.7 --training-reference: ignore @checkpoint decorators so a
        // decorated program still runs the un-checkpointed reference
        // path (the --checkpoint-blocks flag is already forced off).
        let ccr_selective_decorated = !self.compile_options.diagnostics.training_reference
            && self
                .compile_options
                .checkpoint.policies
                .values()
                .any(|p| matches!(p, nsl_semantic::effects::CheckpointPolicy::Selective));
        let mut effective_primal = effective_primal;
        let mut ccr_compress_map: std::collections::HashMap<
            crate::wengert::VarId,
            crate::wengert::VarId,
        > = Default::default();
        // Fresh-id watermark for CCR-created vars: starts above the
        // extractor's range so tail (half) ids can never collide
        // with adjoint-generator ids (the generator start is bumped
        // to this watermark below).
        let mut ccr_fresh: crate::wengert::VarId = extractor.next_var_id();
        let ccr_plan = if self.compile_options.checkpoint.blocks
            || ccr_selective_decorated
        {
            // Milestone C: SCHEDULED as ONE region — the whole
            // search (DP partition, stride candidates, final plan)
            // is one CCR invocation; recording stays at the callee
            // (plan_impl) and last-wins dispositions already model
            // the candidate sweep. tape=None at schedule time
            // because the body itself mutates the scanned list
            // (append_compressed_saves appends the compressed-save
            // tail), so the entry digest would be stale by the
            // pass's own declared mutation — rescan_tape below
            // captures the state the pass LEFT, which is what the
            // consumption forks must see intact.
            let sched = self.passes.scheduler();
            let scheduled = sched
                .schedule("CCR", None, || {
            let claimed_ids: Option<std::collections::HashSet<u32>> =
                self.bus.csha_backward_claims().map(|claims| {
                    claims.op_to_chain.keys().copied().collect()
                });
            let policy = if self.compile_options.checkpoint.selective
                || ccr_selective_decorated
            {
                crate::ccr::CcrPolicy::Selective
            } else {
                crate::ccr::CcrPolicy::Block
            };
            let compress_requested = self.compile_options.checkpoint.compress.is_some();
            // Item 8: resolve the periodic-checkpoint stride. `Fixed(k)`
            // passes straight through (ccr::plan logs the coalescing);
            // `Auto` searches strides against the projected activation
            // peak — with the CSLA accumulation window G applied to the
            // saved boundaries — and the checkpoint byte budget.
            // P5 item 21: `dp` resolves to a NON-UNIFORM kept-anchor
            // set rather than a stride; carried out-of-band to the
            // plan construction below.
            let mut dp_kept_anchors: Option<Vec<usize>> = None;
            let resolved_stride = match self.compile_options.checkpoint.stride {
                crate::CheckpointStride::Fixed(k) => k,
                crate::CheckpointStride::Dp => {
                    let sizes = crate::profiling::captures::size_hints_from_var_nodes(
                        extractor.var_nodes(),
                        self.type_map,
                    );
                    let window = if csla_active {
                        (grad_accumulation_steps.max(1)) as u64
                    } else {
                        1
                    };
                    let budget_bytes = self
                        .compile_options
                        .checkpoint.budget_mib
                        .map(|m| m.saturating_mul(1024 * 1024));
                    let spec = crate::gpu_specs::find_gpu(&self.compile_options.target_gpu)
                        .unwrap_or_else(crate::gpu_specs::default_gpu);
                    let model = crate::ccr::DpCostModel {
                        window,
                        budget_bytes,
                        launch_overhead_ns: spec.kernel_launch_overhead_ns,
                        hbm_gbps: spec.peak_bandwidth_gbs,
                        // #403 measured −37% backward launches from the
                        // dx fusion alone; gamma fusion (P5 slice A)
                        // removes the remaining norm decompositions.
                        bwd_launch_factor: if self.compile_options.fusion.rmsnorm_backward {
                            1.3
                        } else {
                            2.0
                        },
                        // Conservative overlap credit: only the fixed
                        // PCIe latency any in-flight prefetch is
                        // guaranteed to hide (10 us), and only when the
                        // prefetch machinery is actually on.
                        overlap_credit_ns: if self.compile_options.weight_stream.prefetch {
                            10_000
                        } else {
                            0
                        },
                    };
                    let mut fell_back = true;
                    if sizes.is_empty() {
                        nsl_runtime::nsl_log!(WARN, "ccr", 
                            "[ccr] --checkpoint-stride dp: no static tensor sizes \
                             available (symbolic shapes) — falling back to the \
                             uniform-stride search"
                        );
                    } else if let Some(choice) = crate::ccr::select_partition_dp(
                        &effective_primal,
                        claimed_ids.as_ref(),
                        policy,
                        &sizes,
                        &model,
                    ) {
                        // Verify the DP's projection against the TRUE
                        // plan before committing (the DP models
                        // coalesced escapes from stride-1 metrics; the
                        // real segment analysis is the authority).
                        if let Some(p) = crate::ccr::plan_with_kept_anchors(
                            &effective_primal,
                            claimed_ids.as_ref(),
                            policy,
                            false,
                            &choice.keep,
                        ) {
                            let true_peak =
                                crate::ccr::project_activation_peak(&p, &sizes)
                                    .peak_bytes(window);
                            let budget_ok = budget_bytes
                                .is_none_or(|b| true_peak <= b)
                                || !choice.fits_budget;
                            if budget_ok {
                                nsl_runtime::nsl_log!(INFO, "ccr", 
                                    "[ccr] --checkpoint-stride dp: kept {} of {} block \
                                     anchors {:?} (true peak {} MiB, DP projected {} MiB, \
                                     est recompute {:.2} ms/step, window G={window}{})",
                                    choice.keep.len(),
                                    p.segments.len().max(choice.keep.len()),
                                    choice.keep,
                                    true_peak / (1024 * 1024),
                                    choice.projected_peak_bytes / (1024 * 1024),
                                    choice.est_recompute_ns as f64 / 1e6,
                                    if choice.fits_budget {
                                        ""
                                    } else {
                                        ", NO partition fits the budget — min-peak"
                                    },
                                );
                                dp_kept_anchors = Some(choice.keep);
                                fell_back = false;
                            } else {
                                nsl_runtime::nsl_log!(WARN, "ccr", 
                                    "[ccr] --checkpoint-stride dp: true plan peak \
                                     {} MiB contradicts the DP projection ({} MiB) \
                                     over budget — falling back to the uniform search",
                                    true_peak / (1024 * 1024),
                                    choice.projected_peak_bytes / (1024 * 1024),
                                );
                            }
                        }
                    } else {
                        nsl_runtime::nsl_log!(WARN, "ccr", 
                            "[ccr] --checkpoint-stride dp: DP declined (single block \
                             or no plan) — falling back to the uniform-stride search"
                        );
                    }
                    if fell_back {
                        match crate::ccr::select_stride(
                            &effective_primal,
                            claimed_ids.as_ref(),
                            policy,
                            &sizes,
                            window,
                            budget_bytes,
                            crate::ccr::DEFAULT_STRIDE_CANDIDATES,
                        ) {
                            Some(c) => {
                                nsl_runtime::nsl_log!(WARN, "ccr", 
                                    "[ccr] --checkpoint-stride dp fallback: uniform \
                                     stride {} (peak {} MiB)",
                                    c.stride,
                                    c.peak_bytes / (1024 * 1024)
                                );
                                c.stride
                            }
                            None => 1,
                        }
                    } else {
                        1 // unused: dp_kept_anchors drives the plan below
                    }
                }
                crate::CheckpointStride::Auto => {
                    let sizes = crate::profiling::captures::size_hints_from_var_nodes(
                        extractor.var_nodes(),
                        self.type_map,
                    );
                    // Review MEDIUM: with fully symbolic shapes the size
                    // map is empty → every candidate projects to peak 0
                    // and the search silently returns stride 1. Say so,
                    // rather than printing a decision that looks real.
                    if sizes.is_empty() {
                        nsl_runtime::nsl_log!(WARN, "ccr", 
                            "[ccr] --checkpoint-stride auto: no static tensor sizes \
                             available (symbolic shapes) — cannot project the \
                             activation peak; using stride 1. Pass an explicit \
                             --checkpoint-stride N to force periodic checkpointing."
                        );
                    }
                    let window = if csla_active {
                        (grad_accumulation_steps.max(1)) as u64
                    } else {
                        1
                    };
                    let budget_bytes = self
                        .compile_options
                        .checkpoint.budget_mib
                        .map(|m| m.saturating_mul(1024 * 1024));
                    match crate::ccr::select_stride(
                        &effective_primal,
                        claimed_ids.as_ref(),
                        policy,
                        &sizes,
                        window,
                        budget_bytes,
                        crate::ccr::DEFAULT_STRIDE_CANDIDATES,
                    ) {
                        Some(choice) => {
                            let log: Vec<String> = choice
                                .considered
                                .iter()
                                .map(|(k, pb)| {
                                    format!("k={k}:{}MiB", pb / (1024 * 1024))
                                })
                                .collect();
                            nsl_runtime::nsl_log!(INFO, "ccr", 
                                "[ccr] --checkpoint-stride auto: chose stride {} \
                                 (projected activation peak {} MiB{}, window G={window}); \
                                 candidates [{}]",
                                choice.stride,
                                choice.peak_bytes / (1024 * 1024),
                                if choice.fits_budget {
                                    ""
                                } else {
                                    ", NO stride fits the budget — using min-peak"
                                },
                                log.join(", ")
                            );
                            choice.stride
                        }
                        None => {
                            nsl_runtime::nsl_log!(INFO, "ccr", 
                                "[ccr] --checkpoint-stride auto: no candidate produced a \
                                 plan; using stride 1"
                            );
                            1
                        }
                    }
                }
            };
            let plan = if let Some(keep) = &dp_kept_anchors {
                crate::ccr::plan_with_kept_anchors(
                    &effective_primal,
                    claimed_ids.as_ref(),
                    policy,
                    compress_requested,
                    keep,
                )
            } else {
                crate::ccr::plan(
                    &effective_primal,
                    claimed_ids.as_ref(),
                    policy,
                    compress_requested,
                    resolved_stride,
                )
            };
            // Item 8: one coalescing note for the FINAL stride (the Auto
            // search above ran plan() per candidate silently).
            if resolved_stride > 1
                && let Some(p) = &plan
            {
                nsl_runtime::nsl_log!(INFO, "ccr", 
                    "[ccr] periodic checkpointing: stride {resolved_stride} → \
                         {} CCR super-segment(s) (saving every {resolved_stride}th block \
                         boundary, recomputing each span — bit-exact)",
                    p.segments.len()
                );
            }
            if let (Some(p), Some(dtype)) =
                (&plan, self.compile_options.checkpoint.compress.as_deref())
            {
                if !p.compress.is_empty() {
                    ccr_compress_map = crate::ccr::append_compressed_saves(
                        &mut effective_primal,
                        p,
                        dtype,
                        &mut ccr_fresh,
                    );
                    nsl_runtime::nsl_log!(INFO, "ccr", 
                        "[ccr] compressed saves: {} matmul-class tensors -> {dtype}",
                        ccr_compress_map.len()
                    );
                } else if compress_requested {
                    nsl_runtime::nsl_log!(WARN, "ccr", 
                        "[ccr] --checkpoint-compress requested but no \
                         compressible saves exist (policy must be selective \
                         with matmul-class interiors); continuing without"
                    );
                }
            }
            plan
                })
                .map_err(CodegenError::new)?;
            // The body appended the compressed-save tail to
            // `effective_primal` (append-only: existing positions
            // stay valid, but an exact-equality digest from entry
            // would now refuse every --checkpoint-compress build).
            // Re-digest the state the pass left; the
            // apply_to_adjoint / CSLA seg_bounds forks below assert
            // against THIS.
            scheduled.rescan_tape(&effective_primal);
            scheduled.finish(&self.bus).map_err(CodegenError::new)?
        } else {
            None
        };

        Ok(PreForwardPlans {
            effective_primal,
            ccr_plan,
            ccr_fresh,
            ccr_compress_map,
        })
    }
}
