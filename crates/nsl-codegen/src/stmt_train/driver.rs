//! The train block's driver: `compile_train_block` (the callee-side phase
//! scope — installs `CompilePhase::TrainBlock`, refuses the `@pipeline` +
//! `--layerwise-accum` / `--zero-stage` compositions, offers CPDT at the
//! wrapper, consults `PassManager::enforce_dependency_order`, and wraps the
//! body in the fused-CE decorator config prologue / epilogue) and
//! `compile_train_block_inner` (the body: the epoch and batch loops and the
//! bindings that flow between the phases peeled into this directory's
//! modules, in the order `mod.rs` lists them).
//!
//! Moved out of `stmt.rs` whole, byte-for-byte (roadmap A1), once the
//! peels had taken the body from 11.5k lines to ~2.3k; the statement
//! dispatch (`stmt.rs::compile_stmt_dispatch`) and the distill lowering
//! still call `compile_train_block`. The next step is the `TrainPlan` IR
//! (`docs/superpowers/specs/2026-09-08-a1-train-plan-ir-design.md`), which
//! splits this driver into planning and emission.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::InstBuilder;
use cranelift_frontend::{FunctionBuilder, Variable};

use nsl_semantic::types::Type;

use crate::compiler::Compiler;
use crate::stmt::ParamHookEntry;
use crate::stmt::SURFACE_ACTIVATIONS;
use crate::stmt::resolvable_tensor_rank;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::stmt_train::csla_window::{
    CslaPending, CslaSaveInputs, CslaWindowInputs, CslaWindowSave,
};
use crate::stmt_train::model_params::ModelParams;
use crate::stmt_train::adapter_sites::AdapterSitesInputs;
use crate::stmt_train::adjoint_tape_opt::AdjointTapeOptInputs;
use crate::stmt_train::ccr_adjoint_frees::CcrAdjointFreesInputs;
use crate::stmt_train::csla_precompute::CslaPrecomputeInputs;
use crate::stmt_train::fase_hook_lowering::FaseHookLoweringInputs;
use crate::stmt_train::forward_lowering::ForwardLoweringInputs;
use crate::stmt_train::health_hooks::HealthHooksInputs;
use crate::stmt_train::primal_vars::PrimalVarsInputs;
use crate::stmt_train::scheduler_step::SchedulerStepInputs;
use crate::stmt_train::source_ad_grads::SourceAdGradsInputs;
use crate::stmt_train::transient_arena_projection::TransientArenaInputs;
use crate::stmt_train::plan_csha_prune::CshaPruneInputs;
use crate::stmt_train::plan_ccr::{PreForwardPlanInputs, PreForwardPlans};
use crate::stmt_train::plan_wggo::{WggoPlanning, WggoPlanningInputs};
use crate::stmt_train::plan_wrga_cpdt::WrgaCpdtInputs;
use crate::stmt_train::optimizer_state::OptimizerState;
use crate::stmt_train::optimizer_step::OptimizerStepInputs;
use crate::stmt_train::config::TrainConfigSection;
use crate::stmt_train::contract::TrainContract;
use crate::stmt_train::epoch_close::{emit_epoch_close, EpochClose};
use crate::stmt_train::teardown::{emit_train_teardown, TrainTeardown};
use crate::types::is_block_filled;
use cranelift_codegen::ir::Value;

impl Compiler<'_> {
    pub(crate) fn compile_train_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        train: &nsl_ast::block::TrainBlock,
        // CFTP v10 (item 3): AST NodeId of the enclosing `TrainBlock`
        // `Stmt`, used to install the correct `@fused_lm_ce` decorator
        // config into `self.active_fused_ce_config` before source-AD
        // extraction and fused-LCE dtype resolution.
        train_block_stmt_id: nsl_ast::NodeId,
    ) -> Result<(), CodegenError> {
        // Milestone C: the phase scope lives at the CALLEE — this fn, not
        // the drivers. compile_main/-standalone_main already install
        // TrainBlock (same-value RAII nest, harmless), but a train block
        // also arrives through paths that install nothing — user fns
        // (nested train, @test fns), lambdas, model/agent methods, module
        // compiles. Scoping here covers all of them by construction, which
        // is what closes the "production scheduled pass sees phase=None"
        // gap the scheduler's None arm used to document as live.
        let _phase = crate::pass_trace::enter_phase(
            crate::pass_registry::CompilePhase::TrainBlock,
        );
        // M43b: Pipeline parallel detection
        if self.features.pipeline_config.is_some() {
            if self.compile_options.train.layerwise_accum {
                return Err(CodegenError::new(
                    "--layerwise-accum is not supported on the pipelined train \
                     path (@pipeline): the window-buffered schedule was built \
                     for the single-device source-AD emission. Drop one",
                ));
            }
            // D3: @pipeline + --zero-stage is NOT lowered — the pipelined path
            // emits none of ZeRO's init/partition/grad-all-reduce/owner-gated
            // update/param-broadcast, so each rank would train INDEPENDENTLY
            // (silently replicated, no sharding). Refuse loudly rather than
            // appear to support it (deferral-must-refuse); combining pipeline
            // parallel with ZeRO-1 is future work.
            if self.features.zero_stage.filter(|&s| s >= 1).is_some() {
                return Err(CodegenError::new(
                    "--zero-stage is not supported on the pipelined train path \
                     (@pipeline): ZeRO collectives and owner-gated optimizer \
                     sharding are not lowered there, so ranks would train \
                     independently with no gradient reduction. Drop one",
                ));
            }
            // P4 items 17/18: neither precision ladder is lowered on the
            // pipelined path — without these refusals a @pipeline program
            // would compile cleanly and train plain f32 while the user
            // believes they are validating bf16 (deferral-must-refuse).
            if self.features.param_dtype_bf16sr {
                return Err(CodegenError::new(
                    "--param-dtype bf16-sr is not supported on the pipelined                      train path (@pipeline): the bf16 mirror schedule and the                      fused SR step are not lowered there. Drop one",
                ));
            }
            if self.features.muon_state_bf16 {
                return Err(CodegenError::new(
                    "--muon-state-dtype bf16 is not supported on the pipelined                      train path (@pipeline): the CSLA state envelope is not                      lowered there. Drop one",
                ));
            }
            // P5 item 19: the pipelined path lowers its stages outside the
            // region-marker emission — the flag would silently train eager
            // while the user believes graphs are active.
            if self.compile_options.train.cuda_graphs {
                return Err(CodegenError::new(
                    "--cuda-graphs is not supported on the pipelined train \
                     path (@pipeline): capture regions are not emitted \
                     there. Drop one",
                ));
            }
            // Muon perf campaign: the pipelined path never reaches the
            // batch/resident emission — refuse rather than silently no-op
            // (the non-pipeline path refuses non-muon optimizers too).
            if self.compile_options.muon.batch_ns {
                return Err(CodegenError::new(
                    "--muon-batch-ns is not supported on the pipelined train \
                     path (@pipeline). Drop one",
                ));
            }
            if self.compile_options.muon.resident_momentum {
                return Err(CodegenError::new(
                    "--muon-resident-momentum is not supported on the \
                     pipelined train path (@pipeline). Drop one",
                ));
            }
            // Item 7 (`--fuse-wgrad-accum`): the pipelined lowering passes
            // `None` for `on_param_grad` at BOTH of its
            // `compile_wengert_ops` calls, so `wgrad_fusion::plan` is never
            // called AND the `[wgrad-fusion] N chain(s) fused` counter — gated
            // on the same hook in `wengert_lower.rs` — never prints either.
            // This branch returns before the admission in
            // `compile_train_block_inner`, so without this the flag is exactly
            // as silently inert here as it was everywhere before that
            // admission existed: no count, no note, no error.
            //
            // Provenance-split rather than the flat refusal its seven siblings
            // above use, for the same reason the main admission is: the bundle
            // sets this flag on programs that never asked for it.
            if self.compile_options.fusion.wgrad_accum {
                let reason = "the pipelined train path (@pipeline) passes no \
                              on_param_grad hook to its Wengert lowerings, so \
                              there is no FASE accumulate for the fused GEMM \
                              to fold into";
                nsl_runtime::nsl_log!(WARN, "wgrad-fusion", 
                    "[wgrad-fusion] declined: train block #{} — {reason}",
                    self.wgrad_block_ordinal()
                );
                self.wgrad_declines.push((
                    reason.to_string(),
                    "Drop @pipeline, or drop --fuse-wgrad-accum.",
                ));
                if !self.compile_options.fusion.wgrad_accum_from_bundle {
                    return Err(CodegenError::new(format!(
                        "--fuse-wgrad-accum is not supported on the pipelined \
                         train path (@pipeline): {reason}. Drop one"
                    )));
                }
            }
            // Milestone B: the pipelined path never reaches the checkpoint
            // arg parsing/emission below — a train block carrying the args
            // would silently train WITHOUT checkpoints. Refuse loudly.
            for arg in &train.config {
                if let Some(name_sym) = arg.name {
                    let n = self.resolve_sym(name_sym).to_string();
                    if matches!(
                        n.as_str(),
                        "checkpoint_save" | "checkpoint_load" | "checkpoint_every"
                    ) {
                        return Err(CodegenError::new(format!(
                            "train '{n}' is not supported on the pipelined \
                             train path (@pipeline): full-state checkpointing \
                             serializes the monolithic AdamW state lists, \
                             which the pipeline does not build. Drop the \
                             checkpoint args or the @pipeline decorator."
                        )));
                    }
                }
            }
            // The pipelined path returns before the pre-plan/weights-only
            // offer below, so clear the channels it will never install —
            // the enforcement argument on cpdt_plan ("every consult read
            // follows a same-block publish") must not rest on the accident
            // that this path has no consult today; a stale previous block's
            // plan surviving into it would be exactly the leak the wrapper
            // exists to prevent.
            self.bus.clear_wggo_overrides();
            self.bus.clear_cpdt_plan();
            return self.compile_train_block_pipelined(
                builder,
                state,
                train,
                train_block_stmt_id,
            );
        }

        // P5 item 19 (`--cuda-graphs`): opportunistic per-region capture.
        // Regions only exist in Wengert lowerings (source-AD); the tape
        // path would silently ignore the flag. Multi-rank collectives wait
        // on their own streams mid-backward — unvalidated with capture.
        if self.compile_options.train.cuda_graphs {
            if !self.features.source_ad_enabled {
                return Err(CodegenError::new(
                    "--cuda-graphs requires --source-ad: capture regions \
                     bracket the source-AD Wengert lowerings; the tape path \
                     has none and would silently train eager",
                ));
            }
            if self.features.zero_stage.filter(|&s| s >= 1).is_some() {
                return Err(CodegenError::new(
                    "--cuda-graphs does not compose with --zero-stage yet \
                     (collective waits inside the backward are incompatible \
                     with stream capture). Drop one of the flags",
                ));
            }
        }

        // CFTP v10 (item 3): install the fused-CE config for THIS train
        // block; restored on exit unconditionally so an Err bubble does
        // not leave stale state for the next train block.  Doing it here
        // (rather than at the dispatcher) keeps the invariant local to
        // the two `compile_train_block*` entry points, so no future
        // reentry loses the slot.
        let saved_active_fused_ce =
            self.set_active_fused_ce_config_for_train_block(train_block_stmt_id);
        // WGGO-before-kernels: install THIS block's pre-plan overrides — or
        // explicitly `None`, never a previous block's leftovers (the pre-
        // restructure stale-leak) — BEFORE the body compiles. FASE recipe
        // selection and the per-param mode table read `bus.wggo_overrides`
        // well before the in-place planning site in the same function; the
        // pre-pass is what finally lets them see the plan they were written
        // to consume. The offer is validated against the codegen-time
        // extraction at the planning site (graph-fingerprint check) and
        // replaced by an in-place solve on mismatch.
        let preplan_offer = self
            .bus
            .wggo_preplans()
            .iter()
            .find(|p| p.train_block_stmt_id == train_block_stmt_id)
            .map(|p| (p.overrides.clone(), p.plan.applied.clone()));
        match preplan_offer {
            Some((o, applied)) => {
                self.bus.publish_wggo_overrides(o);
                // CPDT-before-moments: offer the same pre-plan to CPDT. The
                // moment-precision consult reads `bus.cpdt_plan` ~2.2k lines
                // before the in-place planning site in the inner function,
                // so a plan published only there arrives after the consult
                // already allocated the moments — CPDT-sourced moment
                // precision was structurally inert on every fresh compile
                // (the bus proved it: `published 1x, read 0x full` plus a
                // DEAD OUTPUT finding, with every activating flag passed).
                // Speculative in exactly the way the overrides install is;
                // the planning site refuses if the fingerprint rejects this
                // pre-plan and the consumed moment dtypes no longer match.
                //
                // Milestone C: SCHEDULED. tape=None — CPDT is
                // TapeAccess::None (the registry is the authority; its
                // OnWengert stage records pipeline position, not access).
                // finish() is live: cpdt_plan declares
                // applied_implies_published Enforced, Applied is only
                // recordable inside cpdt::run, and invoke publishes
                // unconditionally after it returns — so this is a tripwire
                // for any future early-return between the run and the
                // publish.
                let sched = self.passes.scheduler();
                sched
                    .schedule("CPDT", None, || {
                        crate::stmt_pass_bridges::invoke_cpdt_if_enabled(self, Some(&applied), Some(train))
                    })
                    .map_err(CodegenError::new)?
                    .finish(&self.bus)
                    .map_err(CodegenError::new)??;
            }
            // Explicitly cleared, never a previous block's leftovers — the
            // pre-restructure stale-leak this site exists to prevent. The
            // cpdt_plan clear closes the same leak one channel over: block
            // 2's moment consult must not consume block 1's plan.
            //
            // Then offer CPDT weights-only: the precision plan — the only
            // CPDT product the moment consult reads — never reads the
            // applied plan, so a block with no pre-plan (distill's synthetic
            // train block, loop-bound train blocks) still gets its moments
            // typed from the weight map instead of silently staying FP32.
            // The post-body site re-arbitrates against the final bus state
            // and refuses on divergence, exactly as it does for the
            // speculative pre-plan offer above.
            None => {
                self.bus.clear_wggo_overrides();
                self.bus.clear_cpdt_plan();
                // The clear stays BEFORE the scheduled call: finish() must
                // never observe a cleared channel after this invocation's own
                // Applied record.
                let sched = self.passes.scheduler();
                sched
                    .schedule("CPDT", None, || {
                        crate::stmt_pass_bridges::invoke_cpdt_if_enabled(self, None, Some(train))
                    })
                    .map_err(CodegenError::new)?
                    .finish(&self.bus)
                    .map_err(CodegenError::new)??;
            }
        }
        let result =
            self.compile_train_block_inner(builder, state, train, train_block_stmt_id);
        self.restore_active_fused_ce_config(saved_active_fused_ce);
        // Item 5: the placement map is keyed by VarIds from THIS block's
        // extraction, and the VarId counter restarts per extraction — so
        // leaving it installed would apply this block's byte offsets to the
        // next block's unrelated tensors. Cleared unconditionally, including
        // on the error path, for the same reason the fused-CE config is.
        if !self.arena_placements.is_empty() {
            self.arena_placements.clear();
            let _ = self.compile_call_by_name(builder, "nsl_arena_destroy", &[]);
        }
        // Item 17 phase 3a: the packing-metadata variables belong to the
        // train-step function this block just built — a later function
        // (another train block, a grad block, a model method) using them
        // would reference a variable of the WRONG Cranelift function.
        // Cleared unconditionally, including on the error path, exactly
        // like the placement map above.
        self.packing_meta_vars = None;
        result?;
        // Item 2 step 6: the ordering decision, enforced. Every pass on a
        // declared InvocationOrdered edge (CSHA, WRGA — both invoked inside
        // the inner function) has run by now if it is going to, so the
        // per-compile evidence is complete here and the check is decidable.
        // The manager's view is THIS compile's epoch, which is what makes a
        // refusal sound where the process-scoped advisory could not be (a
        // multi-module build interleaves compiles; see pass_manager.rs).
        // This single-exit wrapper is the enforcement point for the same
        // reason it hosts the fused-CE save/restore: every early-return path
        // of the inner function funnels through it.
        self.passes
            .enforce_dependency_order()
            .map_err(CodegenError::new)?;
        Ok(())
    }

    /// CFTP v10 (item 3): pulled out so `compile_train_block` can wrap it
    /// with a `set_active_fused_ce_config_for_train_block` prologue and a
    /// matching restore epilogue — regardless of which early-return path
    /// the body takes.
    pub(crate) fn compile_train_block_inner(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        train: &nsl_ast::block::TrainBlock,
        train_block_stmt_id: nsl_ast::NodeId,
    ) -> Result<(), CodegenError> {

        let saved_variables = state.variables.clone();
        let saved_variable_types = state.variable_types.clone();
        let saved_dataloader_symbols = state.dataloader_symbols.clone();
        let saved_borrowed_batch_symbols = state.borrowed_batch_symbols.clone();

        // ── 1. Extract config from train(...) args ──────────────────────
        // Moved to `stmt_train/config.rs` byte-for-byte (roadmap A1). The
        // driver destructures the section so every binding below keeps its
        // name.
        let TrainConfigSection {
            purpose,
            model_sym,
            epochs,
            grad_accumulation_steps,
            grad_accumulation_decl,
            grad_clip,
            checkpoint_save_path,
            checkpoint_every,
            checkpoint_load_path,
        } = self.extract_train_config(builder, train)?;

        // ── 2. Resolve the optimizer/scheduler/callbacks contract ───────
        // Moved to `stmt_train/contract.rs` byte-for-byte (roadmap A1). The
        // driver destructures the contract so every binding below keeps its
        // name; the struct is the first cut at the plan a `TrainPlan` IR
        // would carry.
        let TrainContract {
            optimizer_name,
            lr_value,
            momentum_value,
            dampening_value,
            weight_decay_value,
            no_decay_scope,
            nesterov_value,
            beta1_value,
            beta2_value,
            eps_value,
            ns_steps_value,
            adamw_lr_value,
            scheduler,
            step_body,
            step_param_sym,
            callbacks,
            fase_plan,
            fase_deferred,
        } = self.resolve_train_contract(
            builder,
            state,
            train,
            grad_accumulation_steps,
            grad_clip,
            purpose,
        )?;

        // ── Item 7 (`--fuse-wgrad-accum`) admission ─────────────────────
        // Moved to `stmt_admission.rs` byte-for-byte (roadmap A1); this
        // block only RECORDS (`wgrad_hook_blocks` / `wgrad_declines`) — the
        // refusal is compile-scoped, in `Compiler::finish_wgrad_admission`.
        self.wgrad_fusion_admission(
            grad_accumulation_steps,
            grad_accumulation_decl,
            &optimizer_name,
            fase_deferred,
            fase_plan.mode,
        );

        // ── CSLA Stage-2 / ZeRO admission ───────────────────────────────
        // Moved to `stmt_admission.rs` byte-for-byte (roadmap A1); the
        // signature there names the five locals the refusals depend on.
        let csla_active = self.csla_and_zero_admission(
            grad_accumulation_steps,
            grad_clip,
            &optimizer_name,
            fase_deferred,
            fase_plan.mode,
        )?;

        // ── 3. Resolve model type and build param list ──────────────────
        // Moved to `stmt_train/model_params.rs` byte-for-byte (roadmap A1);
        // the driver destructures the twelve bindings it needs.
        let ModelParams {
            model_type_name,
            model_var_name,
            layout,
            model_ptr,
            surface_prev,
            param_paths,
            param_list,
            num_params_val,
            checkpoint_names_list,
            cpdt_moment_lists_consumed,
            cpdt_precision_dtypes,
            mode_table_base,
        } = self.emit_model_params(
            builder,
            state,
            model_sym,
            &optimizer_name,
            &checkpoint_save_path,
            csla_active,
            fase_deferred,
            &fase_plan,
        )?;

        // ── 4a. P1 Muon item 6: parameter-ROLE routing flags ────────────
        // Moved to `stmt_train/param_lists.rs` byte-for-byte (roadmap A1).
        let muon_route_list =
            self.muon_route_flags(builder, &optimizer_name, &model_type_name, &param_paths)?;

        // AdamW parameter groups x hoisted weight-decay compositions: moved
        // to `stmt_admission.rs` byte-for-byte (roadmap A1).
        self.no_decay_composition_admission(&no_decay_scope, csla_active)?;

        // ── 4a-bis. AdamW parameter groups: per-param decay-exempt flags ──
        // Moved to `stmt_train/param_lists.rs` byte-for-byte (roadmap A1).
        let decay_exempt_list = self.decay_exempt_flags(
            builder,
            &no_decay_scope,
            weight_decay_value,
            &model_type_name,
            &param_paths,
        )?;

        // ── 4. Create optimizer state buffers ─────────────────────────
        // Moved to `stmt_train/optimizer_state.rs` byte-for-byte (roadmap
        // A1); the driver destructures the five bindings it needs.
        let OptimizerState {
            num_state_buffers,
            muon_state_m_codes,
            state_list_1,
            state_list_2,
            moment_fill_latch,
        } = self.emit_optimizer_state_buffers(
            builder,
            state,
            &optimizer_name,
            &param_paths,
            muon_route_list,
            cpdt_precision_dtypes,
            &checkpoint_save_path,
            &checkpoint_load_path,
            checkpoint_every,
            num_params_val,
            param_list,
            surface_prev,
        )?;

        // ── 5. Initialize lr and step_count variables ───────────────────
        let lr_var = builder.declare_var(cl_types::F64);
        let lr_const = builder.ins().f64const(lr_value);
        builder.def_var(lr_var, lr_const);

        let step_count_var = builder.declare_var(cl_types::I64);
        let zero_i64 = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_count_var, zero_i64);

        // Item 4 (2026-08-25): render + install the resolved
        // train/optimizer/scheduler record for checkpoint identity. Moved
        // to `stmt_train/identity.rs` byte-for-byte (roadmap A1), where the
        // renderer is a pure function with unit tests.
        self.emit_train_config_record(
            builder,
            &crate::stmt_train::identity::TrainConfigRecordInputs {
                optimizer_name: &optimizer_name,
                lr_value,
                grad_accumulation_steps,
                grad_clip,
                weight_decay_value,
                beta1_value,
                beta2_value,
                eps_value,
                momentum_value,
                dampening_value,
                nesterov_value,
                ns_steps_value,
                adamw_lr_value,
                no_decay_scope: &no_decay_scope,
                scheduler: &scheduler,
            },
        )?;

        // Milestone B: full-state resume — moved to `stmt_train/identity.rs`
        // byte-for-byte (roadmap A1); see its header for the ordering and
        // the once-bound loader handle.
        let (has_dataloader, checkpoint_dl_handle) = self.emit_checkpoint_resume(
            builder,
            state,
            &checkpoint_load_path,
            epochs,
            param_list,
            state_list_1,
            state_list_2,
            step_count_var,
        )?;

        // Dev Tools Phase 5 Task 7: publish step-counter variable so
        // `@inspect` emission inside the step body can gate on `step % N`.
        // Cleared at end of compile_train_block.
        self.inspect_train_step_var = Some(step_count_var);

        // ── 5a. Dev Tools Phase 4 Task 4: optional health flush-interval setter ──
        if self.compile_options.dev_tools.health_monitor
            && let Some(n) = self.compile_options.dev_tools.health_flush_interval
        {
            let n_val = builder.ins().iconst(cl_types::I64, n as i64);
            self.compile_call_by_name(builder, "nsl_health_set_flush_interval", &[n_val])?;
        }

        // ── 5b. Allocate gradient accumulation buffers (if grad_accumulation_steps > 1) ──
        // Moved to `stmt_train/param_lists.rs` byte-for-byte (roadmap A1).
        let accum_list = self.alloc_grad_accum_buffers(
            builder,
            state,
            grad_accumulation_steps,
            csla_active,
            num_params_val,
            param_list,
            surface_prev,
        )?;

        // ── 5b2. CSLA (D1b): one-time pointer-tie guard ─────────────────
        // Pointer-tied weights (two fields aliasing one storage) are
        // invisible to the compile-time layerwise analysis; a per-layer
        // in-place θ update through one alias would corrupt the other
        // alias's pending backward. The runtime scan aborts loudly on the
        // first aliased pair (tensor-pointer or data-pointer identity).
        // Param pointers are stable for the whole run (updates are
        // in-place), so once at setup suffices.
        if csla_active {
            self.compile_call_by_name(
                builder,
                "nsl_csla_assert_params_unaliased",
                &[param_list],
            )?;
        }

        // ── 5c. CSLA Stage-2: window buffer lists ───────────────────────
        // One inner NslList per buffered micro-batch (holding the adjoint's
        // primal imports in a fixed compile-time slot order) pushed into
        // `saves_outer`, plus the batch dict pointers in `dicts` so their
        // tensor values survive to the window's deferred backward. Held as
        // Cranelift Variables: the window cleanup frees the shells and
        // re-news them for the next window. The lists are host heap objects —
        // no GPU surface bracket applies.
        let csla_buffers: Option<(Variable, Variable)> = if csla_active {
            let saves_outer_var = builder.declare_var(cl_types::I64);
            let so = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            builder.def_var(saves_outer_var, so);
            let dicts_var = builder.declare_var(cl_types::I64);
            let dl = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            builder.def_var(dicts_var, dl);
            Some((saves_outer_var, dicts_var))
        } else {
            None
        };

        // ── 6. Emit epoch loop ──────────────────────────────────────────
        let epoch_counter_var = builder.declare_var(cl_types::I64);
        // Item 8: a resumed run starts at the epoch the checkpoint recorded,
        // not 0. Without this the loop re-runs every completed epoch while
        // the step counter (and therefore the LR schedule and bias
        // correction) says the run is far past them — the model re-reads old
        // data under a late-training schedule. The runtime publishes the
        // value from the load above; it is 0 for a fresh run and for a v1
        // sidecar (which cannot carry a data position at all).
        let epoch_start = if checkpoint_load_path.is_some() {
            self.compile_call_by_name(builder, "nsl_train_resume_epoch", &[])?
        } else {
            builder.ins().iconst(cl_types::I64, 0)
        };
        builder.def_var(epoch_counter_var, epoch_start);

        let epochs_val = builder.ins().iconst(cl_types::I64, epochs);

        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let increment_block = builder.create_block();
        let exit_block = builder.create_block();

        builder.ins().jump(header_block, &[]);

        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let counter = builder.use_var(epoch_counter_var);
        let cond = builder
            .ins()
            .icmp(IntCC::SignedLessThan, counter, epochs_val);
        builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        // ── 7. Inner batch loop (when DataLoader exists) or single-step (backward compat) ──

        // `has_dataloader` is bound at the checkpoint-load site above.

        // Declare step parameter variable
        let step_param_var = builder.declare_var(cl_types::I64);
        let init_null = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_param_var, init_null);
        state
            .variables
            .insert(step_param_sym, (step_param_var, cl_types::I64));

        // Item 17 phase 3a: packing metadata as explicit dataflow. Declared
        // here (dominating every stash and every fused-AD read) and defined
        // per micro-batch by `emit_packing_registry_stash`; (0, 0) is the
        // same identity sentinel the runtime registry uses. The wrapper
        // clears the field on every exit path — see `packing_meta_vars`'s
        // doc for why it must not outlive this function.
        let seg_meta_var = builder.declare_var(cl_types::I64);
        builder.def_var(seg_meta_var, init_null);
        let doc_meta_var = builder.declare_var(cl_types::I64);
        builder.def_var(doc_meta_var, init_null);
        self.packing_meta_vars = Some((seg_meta_var, doc_meta_var));

        let epoch_loss_var = builder.declare_var(cl_types::I64);
        let epoch_loss_null = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(epoch_loss_var, epoch_loss_null);

        // If DataLoader exists: emit inner batch loop
        // Structure: reset → [batch_header: next_batch → null check → batch_body | batch_exit]
        let batch_header_block;
        let batch_body_block;
        let batch_exit_block = builder.create_block();

        if let Some(dl_handle) = has_dataloader {
            // Reset DataLoader at epoch start
            self.compile_call_by_name(builder, "nsl_dataloader_reset", &[dl_handle])?;

            batch_header_block = builder.create_block();
            batch_body_block = builder.create_block();

            builder.ins().jump(batch_header_block, &[]);

            // Batch header: get next batch, check for null (exhausted)
            // Don't seal yet — back-edge from batch body will be added later
            builder.switch_to_block(batch_header_block);
            let batch_ptr =
                self.compile_call_by_name(builder, "nsl_dataloader_next_batch", &[dl_handle])?;
            let null_check = builder.ins().iconst(cl_types::I64, 0);
            let is_done = builder.ins().icmp(IntCC::Equal, batch_ptr, null_check);
            builder
                .ins()
                .brif(is_done, batch_exit_block, &[], batch_body_block, &[]);

            // Batch body
            builder.switch_to_block(batch_body_block);
            builder.seal_block(batch_body_block);
            state.current_block = Some(batch_body_block);
            builder.def_var(step_param_var, batch_ptr);
            state.cleanup.active_batch_vars.push(step_param_var);
            state.borrowed_batch_symbols.insert(step_param_sym);
        } else {
            // No DataLoader — step body runs once per epoch (backward compat)
            batch_header_block = body_block; // unused, just needs a value
            batch_body_block = body_block; // we're already in it
        }

        // 7a. Prefetch batch tensors to GPU (if on GPU) to overlap
        // page migration with tape setup. This reduces first-access latency
        // from unified memory page faults.
        if has_dataloader.is_some() {
            let batch_val = builder.use_var(step_param_var);
            // PCA Stage C GPU fix: align packed-batch mask/segment tensors to
            // the params' device BEFORE anything consumes them. Without this,
            // a GPU model adding the HOST attention_mask drags the whole
            // attention chain onto the CPU (f64), and FASE later aborts
            // accumulating a CPU-f64 grad into a GPU-f32 m_partial. Runtime
            // no-ops on CPU models / unpacked batches. Runs BEFORE the
            // packing-registry stash below so the registry sees post-move
            // data pointers.
            self.compile_call_by_name(
                builder,
                "nsl_packed_batch_align_device",
                &[batch_val, param_list],
            )?;
            // Prefetch input_ids and labels from the batch dict
            let k_ids = self.compile_string_literal(builder, "input_ids")?;
            let k_lbl = self.compile_string_literal(builder, "labels")?;
            let ids_tensor =
                self.compile_call_by_name(builder, "nsl_dict_get_str", &[batch_val, k_ids])?;
            let lbl_tensor =
                self.compile_call_by_name(builder, "nsl_dict_get_str", &[batch_val, k_lbl])?;
            let device_1 = builder.ins().iconst(cl_types::I64, 1);
            self.compile_call_by_name(builder, "nsl_tensor_prefetch", &[ids_tensor, device_1])?;
            self.compile_call_by_name(builder, "nsl_tensor_prefetch", &[lbl_tensor, device_1])?;

            // CFTP §4.3 / Tier A activation: probe batch for segment_ids +
            // doc_starts and stash their device pointers in the thread-local
            // packing registry (factored into a helper so the CSLA window
            // backward can re-install micro-batch b's metadata before
            // replaying its adjoint — the registry is per-batch state read
            // at @flash_attention LAUNCH time).
            self.emit_packing_registry_stash(builder, state, batch_val)?;
        }

        let prev_batch_scope = state.flags.in_dataloader_batch_scope;
        if has_dataloader.is_some() {
            state.flags.in_dataloader_batch_scope = true;
        }

        // Snapshot variables before step body for cleanup
        let vars_before_step: std::collections::HashSet<nsl_ast::Symbol> =
            state.variables.keys().copied().collect();
        let _on_step_binds_loss = callbacks.iter().any(|cb| {
            self.resolve_sym(cb.name) == "on_step"
                && cb
                    .params
                    .iter()
                    .any(|param| self.resolve_sym(param.name) == "loss")
        });
        let on_epoch_binds_loss = callbacks.iter().any(|cb| {
            matches!(self.resolve_sym(cb.name), "on_epoch" | "on_epoch_end")
                && cb
                    .params
                    .iter()
                    .any(|param| self.resolve_sym(param.name) == "loss")
        });

        // Switch to transient GPU pool for forward/backward intermediates
        self.compile_call_by_name(builder, "nsl_gpu_set_transient_pool", &[])?;
        // P0.1: default surface during fwd/bwd is Activations (restored to
        // the caller's surface at the end-of-step persistent-pool flip).
        let surface_activations = builder.ins().iconst(cl_types::I8, SURFACE_ACTIVATIONS);
        self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_activations])?;

        // Debug: GPU memory at start of step
        {
            let step_val = builder.use_var(step_count_var);
            self.compile_call_by_name(builder, "nsl_debug_gpu_mem", &[step_val])?;
        }

        // ── 7b. Forward pass + backward pass ─────────────────────────
        // When source AD is enabled, attempt compile-time backward graph
        // generation. If extraction fails (dynamic control flow), fall back
        // to the tape-based AD path.
        //
        // fase_hook_active: when true, param gradients are consumed during
        // adjoint lowering (accumulated into m_partial + freed immediately).
        // The downstream grads_list construction + accumulation loops are
        // skipped; grads_list is a null sentinel (i64 0).
        let fase_hook_active = fase_deferred && self.features.source_ad_enabled;

        let mut csla_pending: Option<CslaPending> = None;
        let mut csla_loss_buffered = false;
        // Teardown sweep plan for the trailing partial window: (slot index,
        // free fn) for every owned buffered slot. Outlives csla_pending
        // (which the window-backward emission consumes).
        let mut csla_teardown_slots: Option<Vec<(i64, &'static str)>> = None;

        let (grads_list, loss_val, source_ad_loss_owned, mut wengert_freed_vals) = if self.features.source_ad_enabled {
            // === Source AD path (compile-time backward) ===
            nsl_runtime::nsl_log!(INFO, "nsl", "[nsl] Using source-to-source AD for backward pass");

            // 1. Set training mode
            let true_val = builder.ins().iconst(cl_types::I8, 1);
            self.compile_call_by_name(builder, "nsl_set_training_mode", &[true_val])?;

            // 2. Try to extract Wengert list from step body
            //
            // Cycle-10 §5.3 Task 6 wire-up: route per-fn @checkpoint(policy=...)
            // policies collected by EffectChecker through CompileOptions into
            // the extractor. Empty map = byte-identity preserved.
            //
            // CFTP §4.4 G3 (Sprint 4): plumb the `@fused_lm_ce` decorator into
            // the extractor so `fused_linear_ce(...)` calls inside this train
            // block can be recognised as a single `PrimalOp::FusedLinearCe`
            // when v1's enabled + shape-hint preconditions hold.
            //
            // CFTP v10 (item 3): read `active_fused_ce_config`, which is set
            // in `compile_train_block` from the fused_ce_configs entry whose
            // `train_block_stmt_id` matches THIS train block.  Pre-v10 this
            // read was `fused_ce_configs.first()`, silently binding EVERY
            // train block's substitution to the FIRST decorator's shape/dtype
            // hints — see `feedback_deferral_must_refuse` and the semantic
            // checker's pre-v10 refusal note.
            let fused_ce_cfg = self.active_fused_ce_config.clone();
            // CPKD: thread the @fused_kl_ce config + loss-section constants
            // from the active distill context (None for plain train blocks).
            let (fused_kl_ce_cfg, distill_alpha, distill_temp) = match &self.active_distill_context
            {
                Some(d) => (
                    // P1.7: --training-reference disables the fused KL-CE
                    // substitution so the composite KL-CE baseline runs instead
                    // (mirrors the @fused_lm_ce gate on active_fused_ce_config).
                    // The composite distill loss (alpha/temperature) still runs.
                    if self.compile_options.diagnostics.training_reference {
                        None
                    } else {
                        d.fused_kl_ce.clone()
                    },
                    // Only EXPLICIT loss-section values participate in the
                    // call-site literal cross-check; defaults must not veto.
                    d.loss_alpha_explicit,
                    d.loss_temperature_explicit,
                ),
                None => (None, None, None),
            };
            let mut extractor = crate::source_ad::WengertExtractor::new(self.interner)
                .with_checkpoint_policies(if self.compile_options.diagnostics.training_reference {
                    Default::default() // P1.7: ignore @checkpoint decorators in the reference path
                } else {
                    self.compile_options.checkpoint.policies.clone()
                })
                .with_fused_ce_config(fused_ce_cfg)
                .with_fused_kl_ce_config(fused_kl_ce_cfg, distill_alpha, distill_temp);

            // Wire model method bodies and field types for inline expansion
            extractor.set_model_method_bodies(self.models.model_method_bodies.clone());
            extractor.set_model_field_types(self.models.model_field_types.clone());
            // CFTP v10 (item 5): thread per-model-field rank info so the
            // source-AD extractor can populate `known_ranks` when it
            // registers a model-field weight as a `Param` leaf.  This
            // closes the LATENT 3-D+ RISK on the MoE expert stack
            // scenario (`self.experts.weight: [D, V, H]` accessed via
            // `matmul(x, transpose(self.experts.weight, -2, -1)) + bias`).
            extractor.set_model_field_ranks(self.models.model_field_ranks.clone());
            // Item 4: the same map at full width. Ranks let the matcher REFUSE
            // a bad `W`; dims let inference AFFIRM a good one, which is what
            // reading `[V, H]` off the weight requires.
            extractor.set_model_field_dims(self.models.model_field_dims.clone());
            extractor.set_model_field_scalar_values(
                self.models.model_field_scalar_values.clone(),
            );
            if let Some(ctx) = self.lm_head_inference_for_train_block(train_block_stmt_id) {
                extractor.set_lm_head_inference(ctx);
            }
            // WRGA B.3.2 Option 3: plumb synth overrides so the extractor
            // resolves sentinel-Ident callees/members emitted by the
            // adapter rewrite (fused FFI name + adapter field names).
            extractor.set_synth_call_names(self.synth_call_names.clone());
            extractor.set_synth_member_names(self.synth_member_names.clone());
            // Item 9 phase 2: thread the `@fp8_compute` model-method set so the
            // extractor can record any decorated method it inlines. Source-AD
            // lowering has no FP8 path at all, so an inlined decorated method
            // silently computes in f32; the refusal below turns that into a
            // compile error naming the method.
            extractor.set_fp8_compute_methods(self.features.fp8_compute_methods.clone());

            // Register the model variable as a model instance so method calls get inlined
            extractor.register_model_instance(model_sym, &model_type_name);

            // CPKD: register the frozen teacher instance.  Method calls on
            // it inline exactly like the student's, but every model field it
            // touches registers as a `PrimalOp::Input` leaf (I-11) — no
            // adjoints, no optimizer participation, teacher backward
            // structurally absent from the compiled step.
            if let Some(distill) = self.active_distill_context.clone() {
                let teacher_type_name = self
                    .resolve_source_ad_model_type_name(state, distill.teacher_sym)
                    .ok_or_else(|| {
                        CodegenError::new(format!(
                            "distill teacher '{}' has no resolvable model type \
                             (must be a model instance bound before the distill block)",
                            self.resolve_sym(distill.teacher_sym)
                        ))
                    })?;
                extractor
                    .register_model_instance(distill.teacher_sym, &teacher_type_name);
                let teacher_root = self.resolve_sym(distill.teacher_sym).to_string();
                extractor.set_frozen_model_roots(
                    std::iter::once(teacher_root).collect(),
                );
            }

            // Pre-register outer variables visible in the step body as inputs.
            //
            // CFTP v10 (item 5): thread the rank derived from the semantic
            // `variable_types` table so
            // `try_match_fused_linear_ce_pattern` can refuse rank-3+ `W`
            // operands.  A missing/zero-rank entry stays `None` → matcher
            // preserves its pre-v10 conservative-fire behaviour on
            // unannotated `Tensor` params.  See
            // [`resolvable_tensor_rank`].
            for sym in self.variables_in_name_order(state) {
                let rank = state
                    .variable_types
                    .get(&sym)
                    .and_then(resolvable_tensor_rank);
                extractor.register_input_with_rank(sym, rank);
            }

            let extraction_ok = extractor.extract_stmts(&step_body.stmts);

            // Item 6: refuse a `@fused_lm_ce(enabled = true)` that fused
            // NOTHING.
            //
            // The decorator's whole purpose is to remove the `[N, V]`
            // logits-gradient surface (402 MB per step at V=49152, H=2048).
            // Its substitution declines for twelve distinct reasons and every
            // one used to fall silently through to
            // `PrimalOp::CrossEntropyLoss`, so a
            // user whose LM head is biasless or reshaped before the loss —
            // which is every production coder model in this repo — got a
            // clean compile, no diagnostic, and the full composite path while
            // believing the fused kernel was live. Per
            // `feedback_deferral_must_refuse`, a capability that cannot be
            // delivered must say so.
            //
            // Scoped to "NOTHING fused" rather than "any decline" on purpose:
            // a step body may legitimately contain an auxiliary
            // `cross_entropy` that is not the LM head, and refusing that
            // would make the decorator unusable. Partial declines warn.
            // Item 9 phase 2: `@fp8_compute` does not survive source AD.
            //
            // The decorator's whole job is to route matmuls in the decorated
            // body to `nsl_fp8_matmul_training` so the backward records a
            // `TapeOp::Fp8MatMul` and the weight/activation gradients get the
            // E5M2 round-trip. That routing lives in `expr/advanced.rs`, which
            // source AD does not use: the method body is inlined into a
            // Wengert list and `wengert_lower.rs` lowers `PrimalOp::Matmul` to
            // an unconditional `nsl_tensor_matmul`. Neither `source_ad.rs` nor
            // `wengert_lower.rs` contains a single occurrence of "fp8".
            //
            // So the user gets plain f32 training with no error, no warning,
            // and a decorator in the source claiming otherwise — the failure
            // shape `feedback_deferral_must_refuse` exists to prevent.
            //
            // Gated on `extraction_ok` because a body source AD could not
            // extract falls back to tape AD, where the decorator DOES take
            // effect. Only refuse when source AD actually took the body.
            if extraction_ok {
                let dropped = extractor.inlined_fp8_methods();
                if !dropped.is_empty() {
                    let mut names = String::new();
                    for (i, m) in dropped.iter().enumerate() {
                        names.push_str(&format!("\n  {}. {}", i + 1, m));
                    }
                    return Err(CodegenError::new(format!(
                        "@fp8_compute has no effect under --source-ad \
                         (source-to-source AD), but this train block inlined \
                         {} decorated method(s):{}\
                         \n\nSource AD lowers every matmul to nsl_tensor_matmul \
                         (see wengert_lower.rs, PrimalOp::Matmul) — there is no \
                         FP8 lowering on this path, so the step would train in \
                         plain f32 while the decorator says otherwise.\
                         \n\nEither pass --tape-ad to compile this block on the \
                         tape path, where @fp8_compute routes through \
                         nsl_fp8_matmul_training, or remove the decorator to \
                         request f32 deliberately.",
                        dropped.len(),
                        names,
                    )));
                }
            }

            if extraction_ok {
                // Deduplicate before reporting. `try_unroll_for` re-extracts a
                // loop body once per iteration, so ONE `cross_entropy` inside
                // an unrolled loop yields N identical declines; reporting "N
                // cross_entropy call(s)" for a single source call, N times,
                // would be actively misleading.
                let mut declines: Vec<crate::source_ad::FusedLceDecline> =
                    Vec::new();
                for d in extractor.fused_lce_declines() {
                    if !declines.contains(d) {
                        declines.push(d.clone());
                    }
                }
                if !declines.is_empty() {
                    if extractor.fused_lce_substitution_count() == 0 {
                        let mut reasons = String::new();
                        for (i, d) in declines.iter().enumerate() {
                            reasons.push_str(&format!("\n  {}. {}", i + 1, d.describe()));
                        }
                        return Err(CodegenError::new(format!(
                            "@fused_lm_ce(enabled = true) is active on this train \
                             block, but the fused linear-CE kernel could not be \
                             substituted for ANY of the {} distinct cross_entropy \
                             site(s) in the step body — so the full [batch*seq, \
                             vocab] logits gradient would still be materialized \
                             every step, which is exactly what the decorator \
                             exists to avoid.\
                             \n\nWhy each call declined:{}\n\n\
                             Fix the head so it matches, or set enabled = false (or \
                             drop the decorator) to request the composite path \
                             deliberately.",
                            declines.len(),
                            reasons,
                        )));
                    }
                    // Partial: at least one call fused. Report the rest so a
                    // head that quietly stopped matching is still visible.
                    for d in &declines {
                        nsl_runtime::nsl_log!(INFO, "fused-lm-ce", 
                            "[fused-lm-ce] a cross_entropy call fell back to the \
                             composite path: {}",
                            d.describe()
                        );
                    }
                }
            } else if self.active_fused_ce_config.as_ref().is_some_and(|c| c.enabled) {
                // Item 6: the OTHER way an enabled decorator delivers nothing.
                //
                // The refusal above is gated on `extraction_ok` because a body
                // that source-AD cannot extract never reaches the substitution
                // arm at all — there are no declines to report, and the
                // fallback below is a general mechanism, not a fused-CE
                // decision. But the user's promise is broken just the same: the
                // fused kernel only exists on the source-AD path, so a tape
                // fallback means it definitely did not run.
                //
                // This is not hypothetical. `crates/nsl-codegen/tests/fixtures/
                // fused_lm_ce_e2e_{fp16,bf16}.nsl` both carry a fully-hinted
                // enabled decorator and both land here, because their heads use
                // `bias_add(...)`, which has no source-AD handler. Warn rather
                // than refuse: unlike a decline, this path has a legitimate
                // reading (the body genuinely is not statically extractable)
                // and refusing would break those fixtures.
                nsl_runtime::nsl_log!(ERROR, "fused-lm-ce", 
                    "[fused-lm-ce] @fused_lm_ce(enabled = true) is active, but \
                     source-AD extraction of the step body failed — the fused \
                     linear-CE kernel exists only on the source-AD path, so it \
                     will NOT run and the full [batch*seq, vocab] logits gradient \
                     will be materialized. Restrict the step body to \
                     source-AD-supported operations, or drop the decorator."
                );
            }

            if !extraction_ok {
                // A recorded refusal is not a fallback candidate: the
                // extractor found something that must abort the compile
                // (e.g. an unresolvable dropout probability — the old path
                // silently assumed 0.1). Falling back to tape here would
                // reintroduce exactly the silent-default behavior the
                // refusal exists to prevent.
                if let Some(msg) = extractor.pending_refusal() {
                    return Err(CodegenError::new(format!(
                        "source-AD extraction refused: {msg}"
                    )));
                }

                // CPKD: the tape records EVERY op on the thread-local tape —
                // including the teacher forward — and its backward allocates
                // gradient buffers for each recorded op.  That is the
                // concrete F-06 failure path (teacher grad memory blow-up),
                // so a distill block must never fall back silently.
                if self.active_distill_context.is_some() {
                    return Err(CodegenError::new(
                        "distill step body could not be extracted for source AD; \
                         tape fallback is refused for distillation (I-11/F-06: \
                         the tape would record teacher ops and allocate teacher \
                         gradient buffers). Restrict the step body to \
                         source-AD-supported operations",
                    ));
                }

                // CSLA: the layerwise schedule replays the compile-time
                // adjoint; a silent tape fallback would run the interleaved
                // baseline under a flag claiming the buffered schedule.
                if csla_active {
                    return Err(CodegenError::new(
                        "--layerwise-accum requires source-AD extraction, but the \
                         step body could not be extracted (dynamic control flow?). \
                         Restrict the step body to source-AD-supported operations \
                         or drop --layerwise-accum",
                    ));
                }

                // `--fuse-lm-head require` promises "fused or refuse"; the
                // tape fallback has no fused path, so falling back silently
                // would make `require` vacuous exactly when it matters (a
                // step-body change breaks extraction and the lane keeps
                // passing while paying for the logits surface).
                if self.compile_options.lm_head_fusion
                    == crate::lm_head_inference::LmHeadFusion::Require
                {
                    return Err(CodegenError::new(
                        "--fuse-lm-head require: source-AD extraction failed,                          and the tape fallback cannot fuse an LM head.                          Restrict the step body to source-AD-supported                          operations or drop `require`",
                    ));
                }
                // Source AD extraction failed — fall back to tape
                nsl_runtime::nsl_log!(WARN, "nsl", "[nsl] source AD extraction failed, falling back to tape-based AD");

                // Undo training mode — tape path sets it itself
                let false_val = builder.ins().iconst(cl_types::I8, 0);
                self.compile_call_by_name(builder, "nsl_set_training_mode", &[false_val])?;

                let (grads, loss) =
                    self.compile_tape_backward(builder, state, step_body, param_list)?;
                // Epilogue owns the loss's last ref on the tape fallback too
                // (see the pure-tape arm below for the ref accounting).
                (grads, loss, true, std::collections::HashSet::new())
            } else {
                // 3. Build initial VarMap: map named input/param VarIds to
                //    Cranelift Values already present in state.variables.
                // Moved to `stmt_train/primal_vars.rs` byte-for-byte (roadmap A1):
                // the two name-order passes, the input device guards, the step
                // parameter, the nested model-parameter and frozen teacher
                // loads, and the CPKD report facts. The map is still mutated
                // below (WRGA's adapter tensors), so it stays a driver binding.
                let mut primal_vars = self.emit_primal_vars(
                    builder,
                    state,
                    PrimalVarsInputs {
                        extractor: &extractor,
                        fase_plan: &fase_plan,
                        grad_accumulation_steps,
                        layout: &layout,
                        model_ptr,
                        model_type_name: &model_type_name,
                        param_list,
                        step_param_sym,
                        step_param_var,
                    },
                )?;

                // 4. Find the loss symbol's VarId and set it as the Wengert
                //    list output. VarDecl extraction does not set list.output
                //    (only Return does), so we resolve "loss" by name.
                let loss_var_id = {
                    let mut found = None;
                    for (sym, vid) in extractor.symbol_var_map() {
                        if self.resolve_sym(*sym) == "loss" {
                            found = Some(*vid);
                            break;
                        }
                    }
                    found.ok_or_else(|| {
                        CodegenError::new("train step body must assign to a variable named 'loss'")
                    })?
                };
                extractor.set_output(loss_var_id);

                // Fused-LCE dead-chain prune (review Finding 1 drain point for
                // the train path, which never calls `finalize()`): remove the
                // dead composite `Transpose → Matmul → Add` head chain BEFORE
                // adjoint generation. Leaving it in the tape makes per-op AD
                // emit ghost adjoints for the dead chain that poison the
                // shared accumulation Adds with the live
                // `FusedLinearCeBackwardExtract` results — the lowerer then
                // skips those Adds (unresolved ghost input) and every
                // parameter gradient downstream of the fused loss vanishes.
                // Sprint 2.5: an Err here means a chain op SURVIVED the
                // prune (a live outside consumer such as `logits.shape`) —
                // unrecoverable after substitution, so it is a hard,
                // actionable refusal rather than a ghost-adjoint ICE later.
                extractor
                    .apply_pending_fused_lce_prunes()
                    .map_err(CodegenError::new)?;

            // ── Item 4: compiler-inferred LM head ──────────────────
            //
            // A distinct marker from `[fused-lm-ce]`, which means "the
            // fusion did NOT happen" and is asserted ABSENT by
            // `fused_lm_ce_decline_gate`. Overloading one token for both
            // outcomes would make every one of those negative assertions
            // pass or fail for the wrong reason.
            if let Some(ctx) =
                self.lm_head_inference_for_train_block(train_block_stmt_id)
            {
                // Same dedup as the declines above, for the same reason:
                // `try_unroll_for` re-extracts a loop body per iteration,
                // so one source-level head can report N times.
                let mut heads: Vec<crate::source_ad::InferredHead> = Vec::new();
                for h in extractor.inferred_heads() {
                    if !heads.contains(&h) {
                        heads.push(h);
                    }
                }
                let mut reasons: Vec<String> = Vec::new();
                for r in extractor.inference_declines() {
                    if !reasons.contains(r) {
                        reasons.push(r.clone());
                    }
                }
                for h in &heads {
                    nsl_runtime::nsl_log!(INFO, "lm-head-fusion", 
                        "[lm-head-fusion] inferred: vocab={} hidden={} \
                         rows={}x{}={} bias={} (no @fused_lm_ce decorator \
                         needed; --fuse-lm-head {})",
                        h.vocab_size,
                        h.hidden_size,
                        h.batch_size,
                        h.seq_len,
                        h.batch_size as u64 * h.seq_len as u64,
                        h.has_bias,
                        ctx.mode.as_str(),
                    );
                }
                for r in &reasons {
                    nsl_runtime::nsl_log!(WARN, "lm-head-fusion", "[lm-head-fusion] declined: {r}");
                }
                if heads.is_empty()
                    && ctx.mode
                        == crate::lm_head_inference::LmHeadFusion::Require
                {
                    // `require` is for lanes that would rather not find out
                    // at step 40,000 that the run has been materializing
                    // the [rows, vocab] logits all along.
                    let detail = if reasons.is_empty() {
                        "this train block's step body has no \
                         cross_entropy(logits, targets) call for the fused \
                         kernel to replace"
                            .to_string()
                    } else {
                        let mut s = String::new();
                        for (i, r) in reasons.iter().enumerate() {
                            s.push_str(&format!("\n  {}. {}", i + 1, r));
                        }
                        format!("the chain could not be proven:{s}")
                    };
                    return Err(CodegenError::new(format!(
                        "--fuse-lm-head require: no fused LM head could be \
                         inferred for this train block, so the full \
                         [batch*seq, vocab] logits surface would be \
                         materialized every step.\n\n{detail}\n\n\
                         Pass --fuse-lm-head auto to fall back to the \
                         composite path instead of refusing, or write an \
                         explicit @fused_lm_ce(...) decorator if the shapes \
                         are known to you but not to the compiler."
                    )));
                }
            }

                // WGGO: run the global optimization planner if enabled.
                // Moved to `stmt_train/plan_wggo.rs` byte-for-byte (roadmap A1):
                // the pre-plan reuse (fingerprint check), the planner call under
                // `PassScheduler::schedule`, the applied-plan derivation and the
                // `WggoOverrides` publication. Returns the applied plan and the
                // pre-plan facts the CPDT planning site below consults.
                let WggoPlanning {
                    applied: wggo_applied,
                    preplan_offered: wggo_preplan_offered,
                    preplan_was_rejected: wggo_preplan_was_rejected,
                } = self.plan_wggo(WggoPlanningInputs {
                    extractor: &extractor,
                    mode_table_base,
                    train,
                    train_block_stmt_id,
                })?;

                // CSHA planner schedule, the ELTLS tape-held free, the
                // NSL_DEBUG_WENGERT dump and the spec §4 WGGO prune.
                // Moved to `stmt_train/plan_csha_prune.rs` byte-for-byte (roadmap A1).
                let sched = self.run_csha_and_wggo_prune(
                    builder,
                    state,
                    CshaPruneInputs {
                        extractor: &mut extractor,
                        model_type_name: &model_type_name,
                        primal_vars: &primal_vars,
                        wggo_applied: &wggo_applied,
                    },
                )?;

                // Task 4: WRGA driver + CPDT planning.
                // Moved to `stmt_train/plan_wrga_cpdt.rs` byte-for-byte (roadmap A1):
                // the WRGA driver run under `PassScheduler::schedule` (pruning /
                // rank allocation / fusion; a no-op without inputs) and the CPDT
                // planning site (tier agreement, the moment-precision
                // arbitration, the stale-plan refusal, the no-WGGO skip notice).
                // Returns the WRGA plan the pre-forward phases fork onto.
                let wrga_plan = self.run_wrga_and_plan_cpdt(WrgaCpdtInputs {
                    cpdt_moment_lists_consumed: &cpdt_moment_lists_consumed,
                    extractor: &extractor,
                    fase_deferred,
                    param_paths: &param_paths,
                    train,
                    wggo_applied: &wggo_applied,
                    wggo_preplan_offered,
                    wggo_preplan_was_rejected,
                })?;
                // Task 6: WRGA adapter sites.
                // Moved to `stmt_train/adapter_sites.rs` byte-for-byte (roadmap A1):
                // the override-rejected diagnostics, the adapter init side-table
                // and the adapter-tensor loads into the VarMap.
                self.emit_wrga_adapter_sites(
                    builder,
                    state,
                    AdapterSitesInputs {
                        extractor: &extractor,
                        layout: &layout,
                        model_ptr,
                        model_type_name: &model_type_name,
                        primal_vars: &mut primal_vars,
                        wrga_plan: &wrga_plan,
                    },
                )?;
                // WRGA fork + CCR planning.
                // Moved to `stmt_train/plan_ccr.rs` byte-for-byte (roadmap A1):
                // the positional-reference guard, the fork of the extractor's list
                // onto the WRGA plan (`effective_primal`), and the CCR plan —
                // checkpoint blocks / stride (dp, auto), the VRAM budget, the
                // compressed saves and the owned-tensor restriction — under
                // `PassScheduler::schedule`.
                let PreForwardPlans {
                    effective_primal,
                    ccr_plan,
                    mut ccr_fresh,
                    ccr_compress_map,
                } = self.fork_wrga_and_plan_ccr(PreForwardPlanInputs {
                    csla_active,
                    extractor: &extractor,
                    grad_accumulation_steps,
                    sched,
                    wrga_plan: &wrga_plan,
                })?;

                // ── D2b part 2: the pre-forward pure pipeline ───────────
                // Everything from here to the forward lowering is PURE
                // analysis (no IR emission). Historically the adjoint
                // pipeline ran after the forward because
                // `restrict_to_owned` consumed the lowering's own owned
                // classification; the segment-streamed forward needs the
                // FINAL adjoint's layer schedule at emission time, so the
                // restriction now consumes a pure replica of the
                // ownership fold (`infer_primal_owned`) and the lowering
                // asserts it classified identically afterwards.
                let inferred_owned: Option<
                    std::collections::HashMap<
                        crate::wengert::VarId,
                        crate::wengert::WengertType,
                    >,
                > = ccr_plan.as_ref().map(|_| {
                    let seed: std::collections::HashSet<crate::wengert::VarId> =
                        primal_vars.keys().copied().collect();
                    crate::wengert_lower::infer_primal_owned(&effective_primal, &seed)
                });

                // CCR P1.a: restrict the recompute set to what the primal
                // lowering will classify as owned Tensors — the tape's
                // type default over-claims for scalar arithmetic (raw f64
                // SSA values), which must neither be cloned nor freed; the
                // adjoint keeps consuming the original scalar values.
                let mut ccr_plan = ccr_plan;
                if let Some(plan) = &mut ccr_plan {
                    let owned = inferred_owned
                        .as_ref()
                        .expect("inferred_owned computed whenever a plan exists");
                    if !plan.restrict_to_owned(owned) {
                        nsl_runtime::nsl_log!(WARN, "ccr", 
                            "[ccr] nothing recomputable after the owned-tensor \
                             restriction; running without checkpointing"
                        );
                        // Milestone C: correct the ledger at the drop site.
                        // plan_impl recorded Applied{segments} at plan
                        // CONSTRUCTION; dropping the plan here used to leave
                        // that Applied standing for a build that runs
                        // uncheckpointed — a disposition finish() cannot see
                        // (CCR publishes no channel). Last-wins makes this
                        // the build-truthful statement, the FASE re-record
                        // precedent.
                        crate::pass_trace::record_disposition(
                            "CCR",
                            crate::pass_trace::PassDisposition::Declined {
                                reason: crate::pass_trace::DeclineReason::PreconditionViolated(
                                    "the owned-tensor restriction emptied the recompute set",
                                ),
                            },
                        );
                        ccr_plan = None;
                    } else if let Some(budget_mib) = self.compile_options.checkpoint.budget_mib {
                        // P1.c: knapsack arbitration under the byte budget,
                        // with the C-01 FASE-Deferred credit (the gradient
                        // buffer Deferred never allocates) when parameter
                        // sizes are statically known.
                        let sizes = crate::profiling::captures::size_hints_from_var_nodes(
                            extractor.var_nodes(),
                            self.type_map,
                        );
                        let mut budget_bytes = budget_mib.saturating_mul(1024 * 1024);
                        if fase_deferred {
                            let credit: u64 = effective_primal
                                .ops
                                .iter()
                                .filter(|op| matches!(op.op, crate::wengert::PrimalOp::Param(_)))
                                .filter_map(|op| sizes.get(&op.result))
                                .sum();
                            if credit > 0 {
                                budget_bytes = budget_bytes.saturating_add(credit);
                                nsl_runtime::nsl_log!(INFO, "ccr", 
                                    "[ccr] C-01 credit: FASE Deferred frees the gradient \
                                     buffer — activation budget grows by {} MiB",
                                    credit / (1024 * 1024)
                                );
                            }
                        }
                        let flipped = crate::ccr::apply_budget(
                            plan,
                            &effective_primal,
                            &crate::ccr::CcrBudget { sizes, budget_bytes },
                        );
                        nsl_runtime::nsl_log!(INFO, "ccr", 
                            "[ccr] budget {} MiB: {} tensors flipped back to SAVE",
                            budget_bytes / (1024 * 1024),
                            flipped
                        );
                    }
                }

                // CSLA: the window-buffered schedule leans on CCR — without a
                // plan, "buffer what the adjoint reads" degenerates to N full
                // activation sets (strictly worse than the baseline). Refuse
                // instead of silently buffering everything: either the
                // --checkpoint-blocks flag is missing, the tape has no
                // blocks.N structure (ccr::plan declined with its own stderr
                // note), or the owned-tensor restriction emptied the plan.
                if csla_active && ccr_plan.is_none() {
                    return Err(CodegenError::new(
                        "--layerwise-accum requires an active checkpoint plan: pass \
                         --checkpoint-blocks on a model with blocks.N structure \
                         (see the [ccr] stderr note above for why checkpointing \
                         declined)",
                    ));
                }

                // 6. Generate adjoint backward graph from the (possibly
                //    pruned) primal list.
                let start_var = extractor.next_var_id().max(ccr_fresh);
                let mut generator = crate::source_ad::AdjointGenerator::new(start_var);
                // T7.1: thread CSHA backward claims into the generator so
                // the reverse walk can route claimed ops through the fused
                // backward dispatcher instead of per-op AD rules.
                if let Some(claims) = self.bus.take_csha_backward_claims() {
                    generator.set_csha_claims(claims);
                }
                // Item 9: opt-in fused RMSNorm input-gradient lowering.
                generator.set_fuse_rmsnorm_backward(self.compile_options.fusion.rmsnorm_backward);
                let mut adjoint = generator.generate(&effective_primal);
                // Item 9 profiling (`NSL_PROFILE_ADJOINT=1`): a launch-count
                // histogram of the generated backward ops. Norm/activation
                // adjoints decompose into many small bandwidth-bound ops
                // (RMSNorm dgamma alone = mean/Sqrt/Div/Mul/reduce); this shows
                // which op classes dominate the launch count — the fusion
                // targets. Pre-CCR: recompute clones are forward ops, so this is
                // the true backward-op composition.
                if std::env::var("NSL_PROFILE_ADJOINT").is_ok() {
                    nsl_runtime::nsl_log!(INFO, "adjoint-profile", 
                        "[adjoint-profile] {} generated backward ops:",
                        adjoint.ops.len()
                    );
                    for (k, c) in crate::ew_chain_fusion::histogram(&adjoint.ops) {
                        nsl_runtime::nsl_log!(INFO, "adjoint-profile", "[adjoint-profile]   {c:>5}  {k}");
                    }
                    // D2b prevalence: binaries whose LEFT operand is a
                    // Constant run the baseline chain in host f64 (the
                    // recorded reconcile_device pull-down) — the v1 fuser
                    // must skip them, so count what that costs.
                    nsl_runtime::nsl_log!(INFO, "adjoint-profile", 
                        "[adjoint-profile] const-left binary sites: {}",
                        crate::ew_chain_fusion::const_left_binary_sites(&adjoint.ops)
                    );
                }
                // D2b part 2: hand the claims BACK for the forward lowering
                // below (the fused-SDPA claim dispatch reads them); the
                // compiler slot is cleared again right after the forward, so
                // the ADJOINT lowering still never sees claims — the same
                // invariant the old post-forward `take()` enforced.
                self.bus.restore_csha_backward_claims(generator.take_csha_claims());
                // T7.1: surface any CSHA fallback diagnostics.
                for diag in generator.csha_diagnostics() {
                    nsl_runtime::nsl_log!(INFO, "nsl", "[nsl] {diag}");
                }

                // 6a–6b.5. Adjoint tape optimizations.
                // Moved to `stmt_train/adjoint_tape_opt.rs` byte-for-byte (roadmap A1):
                // the WRGA backward-live filter, dead-gradient elimination, the
                // SwiGLU / RMSNorm-residual / elementwise-chain backward folds and
                // the CSLA schedule report. Returns `adjoint_needed` (the
                // trainable parameter-gradient adjoint VarIds) for the P0.2
                // gradient-integrity guard below.
                let adjoint_needed = self.optimize_adjoint_tape(AdjointTapeOptInputs {
                    adjoint: &mut adjoint,
                    extractor: &extractor,
                    generator: &generator,
                    wrga_plan: &wrga_plan,
                });

                // 6c. CCR P1.a: splice recompute clones + FreeTensor markers
                // into the (final, post-eliminate) adjoint and remap its
                // references from the early-freed originals to the clones.
                // Runs AFTER the eliminate passes so the splice positions
                // and last-use frees are computed against exactly the op
                // list that will be lowered.
                if let Some(plan) = &ccr_plan {
                    // Milestone C: THE positional fork — apply_to_adjoint
                    // iterates `seg.start..seg.end` indexing
                    // effective_primal.ops. The plan's segment bounds are
                    // valid only against the list state CCR left (post
                    // compressed-save append — rescan_tape re-digested it);
                    // prove nothing moved it in the ~300 lines between.
                    // Assert at ENTRY of the consumption.
                    {
                        let sched = self.passes.scheduler();
                        sched
                            .assert_tape_unchanged_since("CCR", &effective_primal)
                            .map_err(CodegenError::new)?;
                    }
                    ccr_fresh = ccr_fresh.max(
                        effective_primal
                            .ops
                            .iter()
                            .map(|o| o.result)
                            .chain(adjoint.ops.iter().map(|o| o.result))
                            .max()
                            .unwrap_or(0)
                            + 1,
                    );
                    crate::ccr::apply_to_adjoint(
                        &effective_primal,
                        &mut adjoint,
                        plan,
                        &mut ccr_fresh,
                    )?;
                    if !ccr_compress_map.is_empty() {
                        crate::ccr::splice_decompress(
                            &mut adjoint,
                            &ccr_compress_map,
                            &mut ccr_fresh,
                        )?;
                    }
                }

                // 6d. CCR: adjoint-region last-use freeing.
                // Moved to `stmt_train/ccr_adjoint_frees.rs` byte-for-byte (roadmap A1):
                // the protected param-gradient set, the pre-insertion wgrad
                // fusion plan, the fresh-VarId advance and the FreeTensor
                // insertion after each adjoint var's last use.
                self.insert_ccr_adjoint_frees(CcrAdjointFreesInputs {
                    adjoint: &mut adjoint,
                    ccr_fresh,
                    ccr_plan: &ccr_plan,
                    effective_primal: &effective_primal,
                    extractor: &extractor,
                    fase_hook_active,
                    generator: &generator,
                });

                // 6d.5 P0.2 gradient-integrity guard: compute the LIVE adjoint
                // result-VarId set over the FINAL adjoint (post dead-grad
                // elimination, post-CCR splice + last-use frees). Armed on the
                // compiler around each non-CSLA adjoint-lowering call below so a
                // live gradient op that cannot resolve an input becomes a hard
                // compile error instead of a silently-dropped gradient (#396).
                // `None` when there are no trainable-parameter gradients.
                let grad_live_set: Option<std::collections::HashSet<crate::wengert::VarId>> =
                    if adjoint_needed.is_empty() {
                        None
                    } else {
                        Some(crate::source_ad::reachable_result_vars(
                            &adjoint.ops,
                            &adjoint_needed,
                        ))
                    };

                // 6e. Milestone C·p2: transient-memory arena projection.
                // Moved to `stmt_train/transient_arena_projection.rs` byte-for-byte
                // (roadmap A1): the Stage-2A element hints, the arena report
                // and the Stage-2B placement (`--transient-arena`), whose
                // slot geometry is declared to the runtime here. Returns the
                // element hints the CSLA schedule precompute below shares.
                let elem_hints = self.emit_transient_arena_projection(
                    builder,
                    TransientArenaInputs {
                        adjoint: &adjoint,
                        csla_active,
                        effective_primal: &effective_primal,
                        extractor: &extractor,
                        generator: &generator,
                    },
                )?;

                // ── D2b part 2: CSLA schedule precompute (pre-forward) ──
                // Moved to `stmt_train/csla_precompute.rs` byte-for-byte (roadmap A1):
                // the layerwise plan, per-param facts, replay ranges and update
                // grouping, computed on the final adjoint so the segment-streamed
                // forward below and the window backward consume the same
                // schedule. `CslaPre` flows into the save phase; `WsForwardPlan`
                // drives the sliced forward emission.
                let (csla_pre, ws_fwd_plan) = self.precompute_csla_schedule(CslaPrecomputeInputs {
                    adjoint: &adjoint,
                    ccr_plan: &ccr_plan,
                    csla_active,
                    effective_primal: &effective_primal,
                    elem_hints: &elem_hints,
                    extractor: &extractor,
                    generator: &generator,
                    param_paths: &param_paths,
                })?;

                // NSL_PHASE_TIMING (deferral-closure 2026-07-14): per-micro-batch
                // forward/backward wall-clock split, printed by the runtime as
                // "[phase] fwd=... bwd=..." lines. Env is read at COMPILE time —
                // `nsl run` compiles and executes in one process so this IS the
                // run-time setting; for `nsl build` the instrumentation is baked
                // in iff the env was set at build time. Source-AD path only (the
                // tape path's backward is a single opaque nsl_tape_backward call).
                let phase_timing =
                    std::env::var("NSL_PHASE_TIMING").ok().as_deref() == Some("1");
                let phase_t0 = if phase_timing {
                    self.compile_call_by_name(builder, "nsl_cuda_device_synchronize", &[])?;
                    Some(self.compile_call_by_name(builder, "nsl_clock", &[])?)
                } else {
                    None
                };

                // Preserve primal inputs for the adjoint: block forward FBIP from
                // overwriting a uniquely-owned activation input (e.g. the matmul
                // temp feeding `silu(x@W)`, refcount 1) that an input-reading
                // backward still needs. Dropped before the adjoint lowering below.
                // See `emit_inplace_suppress`.
                // D2b part 2: register the streamed params BEFORE the
                // forward — every iteration, idempotent. Iteration 1
                // mirrors + evicts (the model arrived resident from
                // `.to(cuda)`); later iterations no-op (the params are
                // already evicted — the previous forward/window evicted
                // them). This is what makes window-1 forwards stream too:
                // without it the first window's forward peak would still be
                // the full-residency wall the flag exists to remove.
                if let Some(wsplan) = &ws_fwd_plan {
                    // Item 3: the plan, not the raw flag, decides who gets an
                    // SR counter block. Its entries are indexed by param_list
                    // position, so a schedule index the plan did not mark
                    // streamed cannot silently pick up (or lose) a storage
                    // mode here.
                    let plan = csla_pre
                        .as_ref()
                        .map(|p| &p.schedule.plan)
                        .expect("ws_fwd_plan and csla_pre are built in the same branch");
                    // The backend is armed iff some parameter's plan entry
                    // needs it. `srbf16_register` aborts on a parameter that
                    // reaches it without a prior `note_param`, so "enable is
                    // on" and "this parameter gets a note" must come from the
                    // SAME source — spelling one from the flag and the other
                    // from the plan is exactly the split this item removes.
                    if plan.needs_sr_backend() {
                        self.compile_call_by_name(builder, "nsl_sr_bf16_enable", &[])?;
                    }
                    for &idx in &wsplan.register_idxs {
                        let entry = usize::try_from(idx)
                            .ok()
                            .and_then(|u| plan.entries().get(u))
                            .ok_or_else(|| {
                                CodegenError::new(format!(
                                    "parameter plan: forward streaming schedule \
                                     registers parameter {idx}, which the plan \
                                     does not cover"
                                ))
                            })?;
                        let iv = builder.ins().iconst(cl_types::I64, idx);
                        let pw = self
                            .compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
                        if entry.needs_sr_note() {
                            self.compile_call_by_name(
                                builder,
                                "nsl_sr_bf16_note_param",
                                &[pw, iv],
                            )?;
                        }
                        // Item 11: the mark must precede the register that
                        // carves the slice — same plan-driven ordering
                        // discipline as the SR note above. The rc is asserted
                        // (unlike the void SR note): a swallowed -1 here
                        // would resurface one phase later as the elem step's
                        // "never carved" abort, misattributing the cause.
                        if entry.is_elementwise() {
                            // Item 16x11: the slice's STORAGE travels with
                            // the mark, from this same plan entry - the
                            // carve must not infer it from another
                            // backend's pointer-keyed state.
                            let srv = builder.ins().iconst(
                                cl_types::I64,
                                i64::from(entry.needs_sr_note()),
                            );
                            let mrc = self.compile_call_by_name(
                                builder,
                                "nsl_zero3_mark_elementwise",
                                &[pw, iv, srv],
                            )?;
                            let mz = builder.ins().iconst(cl_types::I64, 0);
                            let mok = builder.ins().icmp(IntCC::Equal, mrc, mz);
                            let mmsg = "nsl: zero3 elementwise mark failed \
                                        (ZeRO context missing?) — aborting";
                            self.intern_string(mmsg)?;
                            let mmp = self.compile_string_literal(builder, mmsg)?;
                            self.compile_call_by_name(builder, "nsl_assert", &[mok, mmp])?;
                        }
                        self.compile_call_by_name(
                            builder,
                            "nsl_weight_stream_register",
                            &[pw],
                        )?;
                    }
                    // Bake the plan in and confirm the runtime realized it:
                    // every declared param must be in the ONE residency table
                    // its plan entry names. `register` dispatches on global
                    // flags, so a mode that failed to activate would otherwise
                    // train the wrong storage silently. Emitted here, after
                    // the first belt, so the check covers the earliest moment
                    // the tables are populated.
                    self.emit_param_plan_check(builder, plan, param_list)?;
                }
                // Forward lowering.
                // Moved to `stmt_train/forward_lowering.rs` byte-for-byte (roadmap A1):
                // the memory-planner tape-unchanged assertion, the Item 11
                // per-segment early-free plan, and the primal lowering — monolithic
                // or segment-streamed under `--weight-stream` — inside the
                // in-place-suppress window. Returns the early-free plan and the
                // lowered forward.
                let (ccr_segment_free, full_lowered) = self.emit_forward_lowering(
                    builder,
                    state,
                    ForwardLoweringInputs {
                        adjoint: &adjoint,
                        ccr_plan: &ccr_plan,
                        effective_primal: &effective_primal,
                        param_list,
                        primal_vars: &primal_vars,
                        ws_fwd_plan: &ws_fwd_plan,
                    },
                )?;
                // D2b part 2: the adjoint generator consumed the claims
                // pre-forward and handed them back for the forward's fused
                // dispatch — clear them NOW so the adjoint/window lowering
                // never sees claims (the old post-forward `take()` contract).
                self.bus.clear_csha_backward_claims();
                let full_vars = &full_lowered.var_map;

                // D2b part 2: the plan restriction above consumed
                // `infer_primal_owned`'s PREDICTION of this lowering's
                // ownership classification — verify the prediction. A
                // mismatch means the pure replica and the real fold
                // diverged: fail the compile loudly rather than run with a
                // mis-restricted recompute set. (u32::MAX entries are
                // compiler-side sdpa extras, not tape results.)
                if let Some(inferred) = &inferred_owned {
                    let actual: std::collections::HashMap<
                        crate::wengert::VarId,
                        crate::wengert::WengertType,
                    > = full_lowered
                        .owned_values
                        .iter()
                        .filter(|(vid, _, _)| *vid != u32::MAX)
                        .map(|(vid, _, ty)| (*vid, *ty))
                        .collect();
                    if &actual != inferred {
                        return Err(CodegenError::new(format!(
                            "internal: infer_primal_owned diverged from the primal \
                             lowering's ownership classification ({} inferred vs {} \
                             actual entries) — the CCR plan restriction ran on wrong \
                             data; this is a compiler bug",
                            inferred.len(),
                            actual.len(),
                        )));
                    }
                }

                let loss_val = *full_vars.get(&loss_var_id).ok_or_else(|| {
                    CodegenError::new("source AD: loss VarId not found in compiled forward graph")
                })?;

                // CCR P1.a: free the checkpointed block interiors NOW —
                // the forward is done and the backward will recompute them.
                // Lowered as a tiny FreeTensor-only list seeded with the
                // primal var_map; `explicit_freed_vars` flows into the
                // bulk-free exclusion below so nothing double-frees.
                // (Refcounted runtime: views holding a reference keep the
                // storage alive, so this is a decrement, not a hard free.)
                let ccr_freed_primal: std::collections::HashSet<crate::wengert::VarId> =
                    if ccr_segment_free.is_some() {
                        // Item 11: already freed during the forward, one
                        // segment at a time. Running the post-forward list too
                        // would emit a SECOND FreeTensor per victim — on a
                        // refcounted runtime that is a double-decrement, i.e.
                        // a use-after-free for any view still holding the
                        // storage. The sliced path's freed set flows into the
                        // bulk-free exclusion exactly as this one did.
                        full_lowered.explicit_freed_vars.clone()
                    } else if let Some(plan) = &ccr_plan {
                        let free_list = crate::ccr::build_early_free_list(plan);
                        let freed_lowered = crate::wengert_lower::compile_wengert_ops(
                            self, builder, state, &free_list, full_vars, None,
                        )?;
                        freed_lowered.explicit_freed_vars
                    } else {
                        Default::default()
                    };

                // NSL_PHASE_TIMING: end of forward+loss (all primal ops are
                // emitted above; adjoint GENERATION below is compile-time only).
                let phase_t1 = if phase_timing {
                    self.compile_call_by_name(builder, "nsl_cuda_device_synchronize", &[])?;
                    Some(self.compile_call_by_name(builder, "nsl_clock", &[])?)
                } else {
                    None
                };

                // (D2b part 2: adjoint generation, the eliminate passes,
                // the CCR splice, and the last-use-free insertion all moved
                // ABOVE the forward lowering — the pre-forward pure
                // pipeline. `gen` and `adjoint` flow down from there.)

                // ── Build FASE consume-per-param hook (Task 3) ──
                //
                // When fase_deferred && source_ad_enabled, we wire a callback
                // into the adjoint lowering that immediately accumulates each
                // parameter gradient into m_partial and frees it.  This keeps
                // only one parameter gradient live at a time instead of N.
                //
                // Ordering note: `param_paths` (built from
                // enumerate_model_tensor_paths) drives both param_list
                // construction (line ~3247) and accum_list construction
                // (line ~3346).  Both iterate param_paths in the same order,
                // so param_paths[i] == accum_list[i] == param_list[i].
                // We index accum_list by looking up the parameter name in
                // a compile-time param_name→idx map built from param_paths.
                // (fase_hook_active is defined at the outer scope above.)
                let mut param_adj_set: std::collections::HashSet<crate::wengert::VarId> =
                    std::collections::HashSet::new();
                // Maps adjoint VarId → (param_name, primal_cranelift_value, accum_idx)
                // The primal_cranelift_value is used for a runtime pointer-scan against
                // param_list so we find the correct accum_list slot even when
                // trainable params are a subset of named_param_var_ids.
                let mut adj_vid_to_hook_entry: std::collections::HashMap<
                    crate::wengert::VarId,
                    ParamHookEntry,
                > = std::collections::HashMap::new();

                if fase_hook_active {
                    // Build a compile-time name→index map from param_paths
                    // (param_paths[i] corresponds to accum_list[i]).
                    let param_name_to_accum_idx: std::collections::HashMap<&str, i64> =
                        param_paths
                            .iter()
                            .enumerate()
                            .map(|(i, p)| (p.as_str(), i as i64))
                            .collect();

                    for (param_name, primal_vid) in extractor.named_param_var_ids() {
                        if !self.is_trainable_param_name(param_name) {
                            continue;
                        }
                        let Some(&accum_idx) = param_name_to_accum_idx.get(param_name.as_str())
                        else {
                            // Not a tensor param — skip (scalar configs, etc.)
                            continue;
                        };
                        let Some(adj_vid) = generator.adjoint_of(*primal_vid) else {
                            continue;
                        };
                        let Some(&primal_val) = full_vars.get(primal_vid) else {
                            continue;
                        };
                        param_adj_set.insert(adj_vid);
                        adj_vid_to_hook_entry.insert(
                            adj_vid,
                            ParamHookEntry {
                                primal_val,
                                accum_idx,
                            },
                        );
                    }
                }

                // 7. Lower ADJOINT Wengert list using full_vars, which now
                //    contains all intermediate VarId → Value mappings from
                //    the forward pass. This is the key fix: the old code only
                //    had named variables in primal_vars, so intermediate
                //    VarIds (unnamed temporaries like `x @ m.w`) were missing.
                let grad_lowered: Option<crate::wengert_lower::LoweredWengert> = if csla_active {
                    // === CSLA Stage-2 (D1a): window-buffered save phase ===
                    // Moved to `stmt_train/csla_window.rs` byte-for-byte (roadmap A1):
                    // the side-channel refusals, the per-micro-batch slot push of
                    // every adjoint-read primal value (plus the LSE / fused-CE
                    // tape-carries), and the pending carrier the window backward
                    // consumes. It fills the three window carriers the driver
                    // declared above the epoch loop.
                    let CslaWindowSave {
                        csla_pending: pending,
                        csla_loss_buffered: loss_buffered,
                        csla_teardown_slots: teardown_slots,
                    } = self.emit_csla_window_save(
                        builder,
                        CslaSaveInputs {
                            adjoint: &adjoint,
                            adj_vid_to_hook_entry: &adj_vid_to_hook_entry,
                            csla_buffers,
                            csla_pre,
                            fase_hook_active,
                            fase_plan: &fase_plan,
                            full_lowered: &full_lowered,
                            full_vars,
                            has_dataloader,
                            loss_var_id,
                            param_adj_set: &param_adj_set,
                            step_param_var,
                        },
                    )?;
                    csla_pending = pending;
                    csla_loss_buffered = loss_buffered;
                    csla_teardown_slots = teardown_slots;
                    None
                } else if fase_hook_active && !param_adj_set.is_empty() {
                    // FASE Deferred: consume each param gradient immediately.
                    // Moved to `stmt_train/fase_hook_lowering.rs` byte-for-byte
                    // (roadmap A1): the per-parameter accumulate callback, the
                    // grad-integrity bracket and the guarded adjoint lowering.
                    self.emit_fase_hook_adjoint_lowering(
                        builder,
                        state,
                        FaseHookLoweringInputs {
                            accum_list,
                            adj_vid_to_hook_entry: &adj_vid_to_hook_entry,
                            adjoint: &adjoint,
                            fase_plan: &fase_plan,
                            full_vars,
                            grad_live_set: &grad_live_set,
                            num_params_val,
                            param_adj_set: &param_adj_set,
                            param_list,
                        },
                    )?
                } else {
                    self.grad_live_results = grad_live_set.clone();
                    let full_lowered = crate::wengert_lower::compile_wengert_ops(
                        self, builder, state, &adjoint, full_vars,
                        None,
                    );
                    self.grad_live_results = None;
                    match full_lowered {
                        Ok(gv) => Some(gv),
                        Err(e) => {
                            nsl_runtime::nsl_log!(WARN, "nsl", 
                                "[nsl] source AD lowering failed ({}), \
                                 cannot fall back to tape AD after forward emit; \
                                 rerun without --source-ad",
                                e
                            );
                            return Err(e);
                        }
                    }
                };
                // NSL_PHASE_TIMING: end of backward (all adjoint ops emitted).
                if let (Some(t0), Some(t1)) = (phase_t0, phase_t1) {
                    self.compile_call_by_name(builder, "nsl_cuda_device_synchronize", &[])?;
                    let t2 = self.compile_call_by_name(builder, "nsl_clock", &[])?;
                    let fwd = builder.ins().fsub(t1, t0);
                    let bwd = builder.ins().fsub(t2, t1);
                    self.compile_call_by_name(builder, "nsl_phase_fwd_bwd_report", &[fwd, bwd])?;
                }

                // CSLA: no adjoint was lowered here (the window backward
                // replays it later), so every grad_lowered consumer below is
                // guarded on its presence.
                let grad_vars = grad_lowered.as_ref().map(|g| &g.var_map);
                // When the FASE hook is active, each parameter gradient was
                // already consumed (accumulated into m_partial) and freed by
                // the per-param callback during adjoint lowering.  Add those
                // VarIds to freed_adjoint_vars so free_wengert_owned_values
                // skips them and does not emit a second nsl_tensor_free.
                let mut freed_adjoint_vars = std::collections::HashSet::new();
                if let Some(gl) = &grad_lowered {
                    if fase_hook_active {
                        // Exactly the param grads the hook actually freed — NOT
                        // the whole param_adj_set. A param whose grad adjoint is
                        // a shared intermediate (bias-grad == d_out) was
                        // accumulated but its free was DEFERRED; it is absent
                        // here on purpose so the end-of-backward bulk free
                        // releases it exactly once (it is in owned_values).
                        freed_adjoint_vars.extend(gl.hook_freed_param_vars.iter().copied());
                        // Also skip raw_grad VarIds that were freed early by the
                        // reduce_to_shape identity path in wengert_lower.  When
                        // shapes match, reduce_to_shape returns the input with a
                        // refcount bump; the hook's nsl_tensor_free drops rc 2→1
                        // and wengert_lower emits a second free that drops rc 1→0.
                        // These VarIds must NOT be freed again by end-of-adjoint cleanup.
                        freed_adjoint_vars.extend(gl.hook_freed_input_vars.iter().copied());
                    }
                    // CCR: recompute clones were freed by the spliced FreeTensor
                    // markers during adjoint lowering — exclude them from the
                    // end-of-backward bulk free.
                    freed_adjoint_vars.extend(gl.explicit_freed_vars.iter().copied());
                }

                if std::env::var("NSL_DEBUG_SOURCE_AD_OWNED").is_ok() {
                    let summarize_owned = |label: &str,
                                           wengert: &crate::wengert::WengertList,
                                           owned: &[(
                        crate::wengert::VarId,
                        Value,
                        crate::wengert::WengertType,
                    )]| {
                        let mut counts: std::collections::HashMap<String, usize> =
                            std::collections::HashMap::new();
                        for (var_id, _, _) in owned {
                            if let Some(op) = wengert.ops.iter().find(|op| op.result == *var_id) {
                                let key = format!("{:?}", op.op);
                                *counts.entry(key).or_insert(0) += 1;
                            }
                        }
                        let mut counts: Vec<_> = counts.into_iter().collect();
                        counts.sort_by_key(|a| std::cmp::Reverse(a.1));
                        nsl_runtime::nsl_log!(INFO, "nsl", "[nsl] source-ad owned {} ops:", label);
                        for (name, count) in counts {
                            nsl_runtime::nsl_log!(INFO, "codegen", "  {} -> {}", name, count);
                        }
                    };

                    summarize_owned(
                        "primal",
                        extractor.wengert_list(),
                        &full_lowered.owned_values,
                    );
                    if let Some(gl) = &grad_lowered {
                        summarize_owned("adjoint", &adjoint, &gl.owned_values);
                    }

                    let mut final_grad_counts: std::collections::HashMap<String, usize> =
                        std::collections::HashMap::new();
                    for (param_name, vid) in extractor.named_param_var_ids() {
                        if !self.is_trainable_param_name(param_name) {
                            continue;
                        }
                        let Some(adj_vid) = generator.adjoint_of(*vid) else {
                            continue;
                        };
                        if let Some(op) = adjoint.ops.iter().find(|op| op.result == adj_vid) {
                            let key = format!("{:?}", op.op);
                            *final_grad_counts.entry(key).or_insert(0) += 1;
                        }
                    }
                    let mut final_grad_counts: Vec<_> = final_grad_counts.into_iter().collect();
                    final_grad_counts.sort_by_key(|a| std::cmp::Reverse(a.1));
                    nsl_runtime::nsl_log!(INFO, "nsl", "[nsl] source-ad final grad ops:");
                    for (name, count) in final_grad_counts {
                        nsl_runtime::nsl_log!(INFO, "codegen", "  {} -> {}", name, count);
                    }
                }

                // 8. Collect parameter gradients into grads_list (NslList)
                // Moved to `stmt_train/source_ad_grads.rs` byte-for-byte (roadmap A1):
                // the null sentinel under the FASE hook, otherwise the zero-filled
                // list with the lowered adjoints swapped in, then the ownership
                // sweep of the primal/adjoint intermediates. Its value is the
                // source-AD arm's `(grads, loss, source_ad, wengert_freed)`.
                self.emit_source_ad_grads(
                    builder,
                    state,
                    SourceAdGradsInputs {
                        ccr_freed_primal: &ccr_freed_primal,
                        csla_pending: &csla_pending,
                        fase_hook_active,
                        freed_adjoint_vars: &freed_adjoint_vars,
                        full_lowered: &full_lowered,
                        full_vars,
                        generator: &generator,
                        extractor: &extractor,
                        grad_lowered: &grad_lowered,
                        grad_vars,
                        loss_val,
                        loss_var_id,
                        model_type_name: &model_type_name,
                        model_var_name: &model_var_name,
                        num_params_val,
                        param_list,
                    },
                )?
            }
        } else {
            // === Tape AD path (runtime backward) ===
            let (grads, loss) =
                self.compile_tape_backward(builder, state, step_body, param_list)?;
            // The epilogue owns the loss's last reference on this path too:
            // the tape region's promote-to-tape-held retain is cancelled by
            // free_tape_held_tensors, leaving exactly the born reference —
            // and the step-var sweep skips `loss` (pre-declared for the
            // callback plumbing, so it sits in vars_before_step). Without
            // this the tape path stranded one loss tensor per step.
            (grads, loss, true, std::collections::HashSet::new())
        };

        // 7e1b–7e1c. Per-step diagnostics: the debug gradient checksum, the
        // P0.3 grad-integrity scan and the health-monitor hooks.
        // Moved to `stmt_train/health_hooks.rs` byte-for-byte (roadmap A1).
        self.emit_train_health_hooks(
            builder,
            state,
            HealthHooksInputs {
                fase_hook_active,
                grads_list,
                loss_val,
                num_params_val,
                param_list,
                param_paths: &param_paths,
                step_count_var,
            },
        )?;

        // 7e2. Gradient clipping (only if grad_clip was specified).
        // Skip when FASE hook is active — clip is applied via two_phase_clip
        // on m_partial (Phase A/B in the optimizer block below), not on grads_list.
        if !fase_hook_active && grad_clip < f64::MAX {
            let max_norm_val = builder.ins().f64const(grad_clip);
            self.compile_call_by_name(builder, "nsl_clip_grad_norm", &[grads_list, max_norm_val])?;
        }

        // 7e3. Gradient accumulation: accumulate this batch's grads into
        // persistent buffers, then free the per-batch grads immediately.
        // When not accumulating (steps == 1), grads_list is used directly
        // by the optimizer and freed after the step.

        // Compute should_step ONCE here (for grad_accumulation_steps > 1 paths)
        // so it can be reused both to gate the accumulation loop (when two_phase_clip
        // is active) and at the optimizer-gate branch below.
        // For steps == 1 the value is unused; define it anyway to keep the Variable
        // live (its value is never read in that branch).
        let should_step_var = builder.declare_var(cl_types::I8);
        if grad_accumulation_steps > 1 {
            let sc_early = builder.use_var(step_count_var);
            let one_early = builder.ins().iconst(cl_types::I64, 1);
            let sc_p1_early = builder.ins().iadd(sc_early, one_early);
            let accum_const_early = builder.ins().iconst(cl_types::I64, grad_accumulation_steps);
            let rem_early = builder.ins().srem(sc_p1_early, accum_const_early);
            let zero_early = builder.ins().iconst(cl_types::I64, 0);
            let ss_val = builder.ins().icmp(IntCC::Equal, rem_early, zero_early);
            builder.def_var(should_step_var, ss_val);
        } else {
            // steps == 1 → always step; store a constant true (1i8)
            let true_val = builder.ins().iconst(cl_types::I8, 1);
            builder.def_var(should_step_var, true_val);
        }

        if !fase_hook_active && let Some(accum) = accum_list {
            // When two_phase_clip is active, Phase A (in the optimizer block) handles
            // the final micro-batch's accumulation as part of the fused accumulate+sum_sq
            // pass.  Skip the standard accumulation loop on the final micro-batch to
            // avoid double-accumulating.
            let run_standard_accum = if fase_plan.two_phase_clip {
                // Emit a runtime conditional: skip the loop when should_step == true.
                let pre_accum_skip = builder.create_block();
                let pre_accum_run = builder.create_block();
                let pre_accum_join = builder.create_block();
                let ss_check = builder.use_var(should_step_var);
                builder.ins().brif(ss_check, pre_accum_skip, &[], pre_accum_run, &[]);

                builder.switch_to_block(pre_accum_run);
                builder.seal_block(pre_accum_run);
                // Standard accumulation loop lives here; we fall into the shared
                // emission below via the `run_standard_accum` flag.
                // (We use a sentinel to tell the loop-emission code which block to
                // land in after the loop; the join block is pre_accum_join.)
                Some((pre_accum_skip, pre_accum_join))
            } else {
                None
            };

            // Runtime loop: accum[i] += grads[i], then free grads[i]
            let ga_i_var = builder.declare_var(cl_types::I64);
            let ga_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(ga_i_var, ga_zero);
            let ga_hdr = builder.create_block();
            let ga_body = builder.create_block();
            let ga_exit = builder.create_block();
            builder.ins().jump(ga_hdr, &[]);
            builder.switch_to_block(ga_hdr);
            let gai = builder.use_var(ga_i_var);
            let gac = builder
                .ins()
                .icmp(IntCC::SignedLessThan, gai, num_params_val);
            builder.ins().brif(gac, ga_body, &[], ga_exit, &[]);
            builder.switch_to_block(ga_body);
            builder.seal_block(ga_body);
            let accum_buf = self.compile_call_by_name(builder, "nsl_list_get", &[accum, gai])?;
            let grad = self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, gai])?;
            if let Some(mtb) = mode_table_base {
                // FASE Codegen Phase 2: per-param dispatch via runtime byte load.
                let ga_deferred = builder.create_block();
                let ga_fullbuf = builder.create_block();
                let ga_join = builder.create_block();

                self.emit_fase_mode_branch(builder, mtb, gai, ga_deferred, ga_fullbuf);

                // Deferred path
                builder.switch_to_block(ga_deferred);
                builder.seal_block(ga_deferred);
                let off = self.compile_options.train.optim_state_offload;
                self.fase_emit_accumulate(
                    builder,
                    accum_buf,
                    grad,
                    fase_plan.recipe.accum_scale,
                    off,
                )?;
                builder.ins().jump(ga_join, &[]);

                // FullBuffer path: historical raw-sum convention. Note the
                // two-phase-clip case never reaches this loop: two_phase_clip
                // implies a Deferred-global plan, which under source AD (the
                // only path where a mode table exists) activates the FASE
                // hook — and the hook skips this whole loop, accumulating
                // every param (both modes) with the scaled window-mean
                // convention in fase_cb. That uniform convention is what
                // makes the dispatch's Phase A norm valid for mixed tables;
                // the dispatch refuses two_phase_clip without the hook.
                builder.switch_to_block(ga_fullbuf);
                builder.seal_block(ga_fullbuf);
                let n_elems =
                    self.compile_call_by_name(builder, "nsl_tensor_len", &[accum_buf])?;
                self.compile_call_by_name(
                    builder,
                    "nsl_grad_accumulate_add",
                    &[accum_buf, grad, n_elems],
                )?;
                builder.ins().jump(ga_join, &[]);

                // Join — single tensor_free regardless of path
                builder.switch_to_block(ga_join);
                builder.seal_block(ga_join);
            } else if fase_deferred {
                // Pre-Phase-2 monolithic Deferred path (byte-identical when no overrides).
                // FASE Deferred: m_partial += (1/N) * grad  (scaled accumulation)
                let off = self.compile_options.train.optim_state_offload;
                self.fase_emit_accumulate(
                    builder,
                    accum_buf,
                    grad,
                    fase_plan.recipe.accum_scale,
                    off,
                )?;
            } else {
                // Pre-Phase-2 monolithic FullBuffer path (byte-identical).
                // Existing path: raw gradient sum (divided / zeroed after optimizer step)
                let n_elems = self.compile_call_by_name(builder, "nsl_tensor_len", &[accum_buf])?;
                self.compile_call_by_name(
                    builder,
                    "nsl_grad_accumulate_add",
                    &[accum_buf, grad, n_elems],
                )?;
            }
            self.compile_call_by_name(builder, "nsl_tensor_free", &[grad])?;
            let ga_one = builder.ins().iconst(cl_types::I64, 1);
            let ga_next = builder.ins().iadd(gai, ga_one);
            builder.def_var(ga_i_var, ga_next);
            builder.ins().jump(ga_hdr, &[]);
            builder.seal_block(ga_hdr);
            builder.switch_to_block(ga_exit);
            builder.seal_block(ga_exit);
            state.current_block = Some(ga_exit);
            self.compile_call_by_name(builder, "nsl_list_free", &[grads_list])?;

            // If two_phase_clip gated the loop: jump to join, then wire the skip
            // path in, and continue from join.
            if let Some((skip_block, join_block)) = run_standard_accum {
                builder.ins().jump(join_block, &[]);
                builder.switch_to_block(skip_block);
                builder.seal_block(skip_block);
                builder.ins().jump(join_block, &[]);
                builder.switch_to_block(join_block);
                builder.seal_block(join_block);
                state.current_block = Some(join_block);
            }
        } // end if !fase_hook_active (per-micro-batch accumulation guard)

        // ── 7e3b. CSLA Stage-2: window backward phase ───────────────────
        // Moved to `stmt_train/csla_window.rs` byte-for-byte (roadmap A1):
        // the layer-major replay of the buffered window (schedule, per-b
        // seeding, per-range lowering with the fused per-layer update, the
        // weight-stream prefetch belt and the window cleanup). It consumes
        // `csla_pending`; nothing it binds is read afterwards.
        self.emit_csla_window_backward(
            builder,
            state,
            CslaWindowInputs {
                accum_list,
                adamw_lr_value,
                lr_value,
                beta1_value,
                beta2_value,
                dampening_value,
                eps_value,
                momentum_value,
                weight_decay_value,
                ns_steps_value,
                nesterov_value,
                cpdt_precision_dtypes,
                muon_state_m_codes,
                csla_buffers,
                csla_pending,
                fase_plan: &fase_plan,
                grad_accumulation_steps,
                has_dataloader,
                lr_var,
                should_step_var,
                step_count_var,
                moment_fill_latch,
                muon_route_list,
                num_params_val,
                param_list,
                state_list_1,
                state_list_2,
                num_state_buffers,
                optimizer_name: &optimizer_name,
                param_paths: &param_paths,
            },
        )?;

        // ── 7e4–7g. Optimizer step ──────────────────────────────────────
        // Moved to `stmt_train/optimizer_step.rs` byte-for-byte (roadmap A1):
        // the accumulation gate, the mode-table / FASE-deferred / stdlib step
        // arms, the ZeRO reduce + param sync and the post-optimizer cleanup.
        // Nothing it binds is read afterwards.
        self.emit_optimizer_step(
            builder,
            state,
            OptimizerStepInputs {
                accum_list,
                adamw_lr_value,
                lr_value,
                beta1_value,
                beta2_value,
                dampening_value,
                eps_value,
                momentum_value,
                weight_decay_value,
                ns_steps_value,
                nesterov_value,
                grad_clip,
                cpdt_precision_dtypes,
                csla_active,
                fase_deferred,
                fase_hook_active,
                decay_exempt_list,
                fase_plan,
                grad_accumulation_steps,
                grads_list,
                lr_var,
                should_step_var,
                step_count_var,
                mode_table_base,
                muon_route_list,
                no_decay_scope,
                num_params_val,
                param_list,
                state_list_1,
                state_list_2,
                num_state_buffers,
                optimizer_name,
            },
        )?;

        // 7g2–7h. Scheduler, step-count increment and the periodic
        // full-train-state checkpoint.
        // Moved to `stmt_train/scheduler_step.rs` byte-for-byte (roadmap A1).
        self.emit_scheduler_step(
            builder,
            state,
            SchedulerStepInputs {
                checkpoint_dl_handle,
                checkpoint_every,
                checkpoint_names_list,
                checkpoint_save_path: &checkpoint_save_path,
                epoch_counter_var,
                grad_accumulation_steps,
                has_dataloader,
                lr_value,
                lr_var,
                param_list,
                scheduler: &scheduler,
                state_list_1,
                state_list_2,
                step_count_var,
            },
        )?;

        // 7i. Callbacks: compile on_step body with step_count and loss bound
        for cb in &callbacks {
            let cb_name = self.resolve_sym(cb.name).to_string();
            if cb_name == "on_step" {
                // Bind callback params: on_step(step, loss)
                for param in &cb.params {
                    let pname = self.resolve_sym(param.name).to_string();
                    match pname.as_str() {
                        "step" => {
                            let var = builder.declare_var(cl_types::I64);
                            let step_val = builder.use_var(step_count_var);
                            builder.def_var(var, step_val);
                            state.variables.insert(param.name, (var, cl_types::I64));
                            state.param_symbols.insert(param.name);
                            // RECORD THE TYPE. Without it the step-body sweep
                            // below sees no `variable_types` entry, calls the
                            // slot "indeterminate", and hands the COUNTER
                            // VALUE to nsl_tensor_free_if_valid, which probes
                            // it as a pointer. That probe returns early for
                            // anything under 0x10000, so it is a silent no-op
                            // for the first 65,535 micro-steps and a SIGSEGV
                            // at exactly 65,536 — a hard ceiling every
                            // training run with an `on_step` callback hit
                            // (2026-08-29: the 1B chain, all three
                            // matched-pair arms, and a 30-line CPU fixture
                            // all faulted at fault address 0x10000).
                            state.variable_types.insert(param.name, Type::Int);
                        }
                        "loss" => {
                            // loss is already in state.variables, but rebind for callback scope
                            let var = builder.declare_var(cl_types::I64);
                            builder.def_var(var, loss_val);
                            state.variables.insert(param.name, (var, cl_types::I64));
                            state.param_symbols.insert(param.name);
                        }
                        _ => {
                            // Unknown callback param — bind to zero
                            let var = builder.declare_var(cl_types::I64);
                            let z = builder.ins().iconst(cl_types::I64, 0);
                            builder.def_var(var, z);
                            state.variables.insert(param.name, (var, cl_types::I64));
                            state.param_symbols.insert(param.name);
                            // Typed for the same reason as `step`. Zero is a
                            // no-op in the probe, so this arm never faulted —
                            // but that safety came from the VALUE, not from a
                            // rule, and the next binding to reach here would
                            // not be so lucky.
                            state.variable_types.insert(param.name, Type::Int);
                        }
                    }
                }
                // Item 12: open a scoped residency window if this callback
                // touches model θ under weight streaming (else the reads
                // launch on evicted, null data).
                let ws_guard =
                    self.emit_callback_residency_open(builder, &cb.body, model_sym, &cb_name)?;
                // Compile callback body
                for stmt in &cb.body.stmts {
                    self.compile_stmt(builder, state, stmt)?;
                }
                self.emit_callback_residency_close(builder, ws_guard)?;
            }
        }

        state.flags.in_dataloader_batch_scope = prev_batch_scope;

        if on_epoch_binds_loss {
            let saved_loss = builder.use_var(epoch_loss_var);
            self.compile_call_by_name(builder, "nsl_tensor_free_if_valid", &[saved_loss])?;
            let loss_clone = self.compile_call_by_name(builder, "nsl_tensor_clone", &[loss_val])?;
            builder.def_var(epoch_loss_var, loss_clone);
        }

        // Free the loss tensor after callbacks have used it. The on_epoch path
        // already cloned it into epoch_loss_var, and on_step has finished reading.
        // Previously guarded by !on_step_binds_loss which leaked one loss per step.
        if source_ad_loss_owned {
            if csla_loss_buffered {
                // CSLA with an adjoint-read loss: on non-step iterations the
                // loss must survive in the window buffer (its replay reads
                // it); the window backward frees the older entries and this
                // conditional frees the CURRENT iteration's loss only on the
                // step iteration, after its replay is done.
                let lf_block = builder.create_block();
                let lf_join = builder.create_block();
                let ss_here = builder.use_var(should_step_var);
                builder.ins().brif(ss_here, lf_block, &[], lf_join, &[]);
                builder.switch_to_block(lf_block);
                builder.seal_block(lf_block);
                self.compile_call_by_name(builder, "nsl_tensor_free", &[loss_val])?;
                builder.ins().jump(lf_join, &[]);
                builder.switch_to_block(lf_join);
                builder.seal_block(lf_join);
                state.current_block = Some(lf_join);
            } else {
                self.compile_call_by_name(builder, "nsl_tensor_free", &[loss_val])?;
            }
            wengert_freed_vals.insert(loss_val);
        }

        // Free step-body tensor variables to prevent GPU memory accumulation.
        // In source AD mode, Wengert lowering already frees its owned intermediates,
        // so this sweep must skip values that were released earlier in the step.
        let current_blk = state.current_block.unwrap_or(batch_body_block);
        if !is_block_filled(builder, current_blk) {
            let zero = builder.ins().iconst(cl_types::I64, 0);
            // Name order, not `HashMap` order: each `use_var` below numbers
            // a value, and this sweep is the last thing in `main` that
            // reads the step's variables (see `variables_in_name_order`).
            let step_tensor_vars: Vec<_> = self
                .variables_in_name_order(state)
                .into_iter()
                .filter(|sym| !vars_before_step.contains(sym))
                // Borrow aliases (e.g. `let alias = m.w`, DataLoader handles)
                // don't own their tensor — freeing them here would free the
                // model weight itself and use-after-free the next step.
                .filter(|sym| !state.non_owning_symbols.contains(sym))
                .filter(|sym| !state.borrowed_batch_symbols.contains(sym))
                .filter_map(|sym| {
                    let (var, _) = state.variables[&sym];
                    let sem_ty = state.variable_types.get(&sym);
                    let is_tensor = sem_ty.map(|t| t.is_tensor()).unwrap_or(false);
                    let is_unknown = sem_ty.map(|t| t.is_indeterminate()).unwrap_or(true);
                    if is_tensor || is_unknown {
                        Some((var, is_tensor))
                    } else {
                        None
                    }
                })
                .collect();
            for (var, is_tensor) in step_tensor_vars {
                let val = builder.use_var(var);
                if !wengert_freed_vals.contains(&val) {
                    if is_tensor {
                        let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[val]);
                    } else {
                        let _ =
                            self.compile_call_by_name(builder, "nsl_tensor_free_if_valid", &[val]);
                    }
                }
                builder.def_var(var, zero);
            }
        }

        if has_dataloader.is_some() {
            let current_blk = state.current_block.unwrap_or(batch_body_block);
            if !is_block_filled(builder, current_blk) {
                // CSLA: the batch dict was pushed into the window buffer —
                // its tensor values (input_ids for the embedding scatter,
                // labels for the loss adjoint) must survive to the window
                // backward, which frees each dict's values after its
                // micro-batch's replay (partial-tail dicts are swept at
                // teardown).
                if !csla_active {
                    let batch_to_free = builder.use_var(step_param_var);
                    self.compile_call_by_name(
                        builder,
                        "nsl_dict_free_tensor_values",
                        &[batch_to_free],
                    )?;
                }
                let zero = builder.ins().iconst(cl_types::I64, 0);
                builder.def_var(step_param_var, zero);
            }
            state.cleanup.active_batch_vars.pop();
            state.borrowed_batch_symbols.remove(&step_param_sym);
        }

        // Switch back to persistent pool (for epoch callbacks, optimizer state updates)
        self.compile_call_by_name(builder, "nsl_gpu_set_persistent_pool", &[])?;
        // P0.1: end of the Activations bracket — restore the caller's surface.
        self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_prev])?;

        // Drain GPU caching allocator — only releases Transient segments,
        // which are now fully free since forward/backward intermediates were freed.
        //
        // P5 item 19: the runtime skips this drain while cuda-graph capture is
        // ARMED (per-step segment release would churn every transient address
        // and unmap memory captured graphs reference) — checked at runtime,
        // not compile time, so a declined enable() keeps the drain (review L5).
        self.compile_call_by_name(builder, "nsl_gpu_drain_cache", &[])?;

        // Debug: GPU memory after step cleanup
        {
            let step_val = builder.use_var(step_count_var);
            self.compile_call_by_name(builder, "nsl_debug_gpu_mem", &[step_val])?;
            self.compile_call_by_name(builder, "nsl_debug_gpu_alloc_summary", &[step_val])?;
        }

        // Stage-2C canary: verify every red zone after the step's kernels
        // have all run. Runtime-gated by NSL_ARENA_CHECK=1, so one binary
        // serves both the validation runs and production.
        if self.compile_options.memory.transient_arena {
            let step_val = builder.use_var(step_count_var);
            self.compile_call_by_name(builder, "nsl_arena_check_step", &[step_val])?;
        }

        // ── 8. Close batch loop (if DataLoader) and increment epoch ──────
        // Moved to `stmt_train/epoch_close.rs` byte-for-byte (roadmap A1):
        // the epoch callbacks, the epoch increment and the loop seal.
        emit_epoch_close(
            self,
            builder,
            state,
            &callbacks,
            EpochClose {
                batch_header_block,
                batch_body_block,
                batch_exit_block,
                has_dataloader,
                header_block,
                increment_block,
                exit_block,
                epoch_counter_var,
                epoch_loss_var,
                on_epoch_binds_loss,
                model_sym,
            },
        )?;

        // Free the block's lists, sweep the trailing CSLA window, restore
        // streamed weights, print the graphs banner (stmt_train/teardown.rs).
        emit_train_teardown(
            self,
            builder,
            state,
            TrainTeardown {
                param_list,
                num_params_val,
                state_list_1,
                state_list_2,
                num_state_buffers,
                moment_fill_latch,
                accum_list,
                csla_active,
                csla_buffers,
                csla_teardown_slots,
            },
        )?;

        state.variables = saved_variables;
        state.variable_types = saved_variable_types;
        state.dataloader_symbols = saved_dataloader_symbols;
        state.borrowed_batch_symbols = saved_borrowed_batch_symbols;

        // Phase 5 Task 7: clear train-scope @inspect context on exit.
        self.inspect_train_step_var = None;

        // Gap I.B: drop stale CSHA per-function cache entries so a
        // subsequent train/grad block in the same module gets a clean
        // slate (Cranelift `Value` IDs reset per function and would
        // otherwise alias against leftover keys).
        self.clear_csha_per_function_caches();

        Ok(())
    }
}
