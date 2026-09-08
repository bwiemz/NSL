//! The CSHA planner schedule and the WGGO prune of the train block's
//! source-AD arm, between the WGGO planning site and the WRGA driver:
//! the CSHA planner run under `PassScheduler::schedule` (the body is
//! `stmt.rs::invoke_csha_if_enabled`, which also publishes to the bus;
//! see the schedule's comment for why no tape digest is captured), the
//! ELTLS free of the tape-held tensors and the tape-region flag clear,
//! the `NSL_DEBUG_WENGERT` primal dump, and the spec §4 WGGO prune — the
//! positional consumption fork that applies the plan's indices to the
//! tape (after asserting nothing moved the list since WGGO's digest),
//! refuses the whole plan on any refusal, records the `Applied`
//! disposition and prints the per-rewrite `[prune]` lines.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 152 lines,
//! 4 inputs ([`CshaPruneInputs`]); the scheduler handle the driver
//! goes on to hand the CCR planning is the function's result. The
//! train-block CLIF snapshots
//! (`tests/train_clif_snapshots.rs`) pin the forward every fixture lowers
//! after this point.

use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::pass_manager::PassScheduler;

/// Every binding of `compile_train_block_inner` this phase reads; names
/// are the driver's.
pub(crate) struct CshaPruneInputs<'a, 'e> {
    /// The forward extractor; the WGGO prune rewrites its Wengert list in place.
    pub(crate) extractor: &'a mut crate::source_ad::WengertExtractor<'e>,
    /// The resolved model type name (the CSHA planner is keyed by it).
    pub(crate) model_type_name: &'a String,
    /// The initial VarMap (read by the `NSL_DEBUG_WENGERT` dump only).
    pub(crate) primal_vars: &'a crate::wengert_lower::VarMap,
    /// The applied WGGO plan; `None` means no prune.
    pub(crate) wggo_applied: &'a Option<crate::wggo_apply::AppliedPlan>,
}

impl Compiler<'_> {
    /// Run the CSHA planner schedule and the WGGO prune (see the module
    /// header); returns the scheduler handle the CCR planning reuses.
    pub(crate) fn run_csha_and_wggo_prune(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        inputs: CshaPruneInputs<'_, '_>,
    ) -> Result<PassScheduler, CodegenError> {
        let CshaPruneInputs {
            extractor,
            model_type_name,
            primal_vars,
            wggo_applied,
        } = inputs;

        // CSHA: Compiler-Synthesized Holistic Attention planner.
        // Runs the boundary-fusion scan, SMEM feasibility model,
        // and weight-informed specialization.  Emits either the
        // full paper-§6.3 report or a compact one-line summary
        // gated by the `--csha` / `--csha-report` flags.  The
        // planner is pure data-in/data-out; wiring the kernel
        // decisions back into codegen is a follow-up step.
        //
        // Pass order: Calibration → WGGO → CSHA.
        // CSHA receives WGGO's AppliedPlan (if any) as WggoOverrides
        // (via bus.wggo_overrides) so that per-layer fusion-level
        // decisions from WGGO are honoured (or rejected with a
        // diagnostic) by CSHA.
        //
        // Milestone C: SCHEDULED — the body (hoisted to
        // `invoke_csha_if_enabled`, mirroring WRGA's bridge) contains
        // the planner run AND all three bus publishes, so finish()'s
        // applied⇒published check on `csha_bridge` (Enforced) judges
        // a settled state. tape=None deliberately: NO
        // assert_tape_unchanged_since exists for CSHA anywhere
        // (`pass_scheduler_coverage.rs` records the exemption) —
        // its positional chain fields are converted to OpIds AT the
        // scan boundary inside this window (`collect_claimed_ops` /
        // the dispatch-map build), OpIds are stable across the
        // deletions `wggo_prune` makes right after this, and an
        // assert against the post-prune list would refuse every
        // prune+CSHA composition for a mutation that invalidates
        // nothing the pass retained. A digest nobody can ever read
        // is a full-tape hash per train block buying only a trace
        // token, so none is captured.
        let sched = self.passes.scheduler();
        sched
            .schedule("CSHA", None, || {
                crate::stmt::invoke_csha_if_enabled(
                    self,
                    extractor.wengert_list(),
                    model_type_name,
                );
            })
            .map_err(CodegenError::new)?
            .finish(&self.bus)
            .map_err(CodegenError::new)?;

        // 5. Lower PRIMAL Wengert list to Cranelift IR.
        //    This IS the forward pass — each WengertOp is compiled to
        //    its runtime FFI call, and ALL intermediate VarId → Value
        //    mappings are recorded in full_vars.
        // ELTLS: free tape-held tensors before clearing the tape flag.
        self.free_tape_held_tensors(builder, state);
        state.flags.in_tape_region = false;
        // Debug: dump primal Wengert ops
        if std::env::var("NSL_DEBUG_WENGERT").is_ok() {
            nsl_runtime::nsl_log!(INFO, "wengert", 
                "[wengert] primal_vars: {:?}",
                primal_vars.keys().collect::<Vec<_>>()
            );
            for op in &extractor.wengert_list().ops {
                let name = extractor
                    .wengert_list()
                    .var_names
                    .get(&op.result)
                    .cloned()
                    .unwrap_or_default();
                nsl_runtime::nsl_log!(INFO, "wengert", 
                    "[wengert] VarId {} '{}' = {:?} inputs={:?} in_primal={}",
                    op.result,
                    name,
                    op.op,
                    op.inputs,
                    primal_vars.contains_key(&op.result)
                );
            }
        }
        // --- NEW: spec §4 WGGO Prune, runs BEFORE wrga so WRGA sees reduced forward ---
        // When WGGO produced a plan, run the prune IR rewriter. On any refusal the
        // whole plan is rejected (spec §5.3 dry-run-then-commit contract) and
        // compilation fails with a CodegenError. On success each rewritten layer
        // gets a stderr marker that Task 15 will upgrade to format_refusal output.
        if let Some(applied_plan) = wggo_applied {
            // Milestone C: THE positional consumption fork — the
            // plan's indices are applied to the tape here, and they
            // are valid only against the list state the plan was
            // produced from (planned in place, or fingerprint-matched
            // to the pre-plan's extraction). Prove nothing moved the
            // list since the scheduled WGGO retained its digest
            // (CSHA in between is Reads-only). Assert at ENTRY of the
            // consumption; after the prune the digest is stale BY
            // DESIGN (the prune is WGGO's declared mutation) and
            // nothing may re-assert it — WRGA's own schedule
            // re-digests the post-prune list.
            {
                let sched = self.passes.scheduler();
                sched
                    .assert_tape_unchanged_since("WGGO", extractor.wengert_list())
                    .map_err(CodegenError::new)?;
            }
            let empty_weight_map = crate::weight_aware::WeightMap::default();
            let weight_map_ref = self.features.weight_map.as_ref().unwrap_or(&empty_weight_map);
            let wggo_prune_result = crate::wggo_prune::run(
                extractor.wengert_list_mut(),
                applied_plan,
                weight_map_ref,
            );
            if !wggo_prune_result.refusals.is_empty() {
                // Spec §3 / §6: emit three-part refusal text per variant.
                // diagnostic_code() provides the structured OverrideRejectReason
                // for any future attach-reason API once diagnostic infrastructure
                // exposes it. For now, the stderr text + CodegenError is the
                // diagnostic contract.
                for refusal in &wggo_prune_result.refusals {
                    let text = crate::wggo_prune::format_refusal(refusal);
                    nsl_runtime::nsl_log!(INFO, "codegen", "{text}");
                }
                return Err(crate::error::CodegenError::new(
                    "wggo_prune: one or more prune decisions refused; see [prune] stderr lines",
                ));
            }
            // Item 3: supersede WGGO's own layer-decision count with
            // the number of layers whose ops this prune actually
            // deleted. This is the ONLY place WGGO mutates the tape,
            // so it is the honest answer to "what did WGGO do".
            //
            // Guarded on non-empty on purpose. A plan with no `Prune`
            // decisions reaches here with zero rewrites, and recording
            // that would erase the true statement that N per-layer
            // decisions were applied (which CSHA / WRGA / FASE then
            // read) in exchange for a zero that is already implied by
            // the absence of `[prune]` lines. The refusal path above
            // needs no disposition: it returns a `CodegenError`, so
            // there is no build left to report on.
            if !wggo_prune_result.rewrites.is_empty() {
                crate::pass_trace::record_disposition("WGGO", crate::pass_trace::PassDisposition::Applied {
                    rewrites: wggo_prune_result.rewrites.len(),
                });
            }
            // Success path: spec §6.1 format per rewrite.
            // layer_index is looked up from applied_plan.layers by name match
            // so we report the index the planner assigned, not the Vec position.
            for rewrite in &wggo_prune_result.rewrites {
                let layer_index = applied_plan.layers.iter()
                    .find(|l| l.layer_name == rewrite.layer_name)
                    .map(|l| l.layer_index)
                    .unwrap_or(0);
                let line = crate::wggo_prune::format_success_stderr(
                    rewrite,
                    layer_index,
                    rewrite.ops_deleted,  // per-rewrite, not aggregate
                );
                nsl_runtime::nsl_log!(INFO, "codegen", "{line}");
            }
        }
        // --- END NEW ---

        Ok(sched)
    }
}
