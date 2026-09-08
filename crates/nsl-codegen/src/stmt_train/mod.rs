//! The train block's lowering, one phase per submodule.
//!
//! `stmt.rs::compile_train_block_inner` is the driver: it still owns the
//! epoch + batch loops and the bindings that flow between the phases. Each submodule here is one phase peeled off that function
//! (roadmap A1), in the order the driver runs them:
//!
//!   - [`config`] — section 1: the `train(...)` header resolved through
//!     `nsl-semantic`'s ONE resolver, the CUDA-graphs arming and the
//!     distill overrides, returned as a [`config::TrainConfigSection`].
//!   - [`model_params`] — section 3: the model's layout, its tensor
//!     parameters as a runtime list, the CPDT dtype-code lists and Muon's
//!     mode table, returned as a [`model_params::ModelParams`].
//!   - [`optimizer_state`] — section 4: the moment lists, allocated per
//!     parameter by a runtime loop (device / host / owner-gated / null /
//!     route-conditional), returned as an [`optimizer_state::OptimizerState`].
//!   - [`contract`] — section 2: the resolved optimizer / scheduler /
//!     callbacks contract, the `data:` section, the Muon perf-flag
//!     refusals and the FASE plan, returned as a [`contract::TrainContract`].
//!   - [`csla_window`] — the `--layerwise-accum` window: the save phase
//!     (the `csla_active` arm of the adjoint-lowering site, fed by a
//!     [`csla_window::CslaSaveInputs`], returning the window carriers as a
//!     [`csla_window::CslaWindowSave`]) and section 7e3b's window backward
//!     (schedule replay, per-layer fused updates, prefetch belt, window
//!     cleanup, fed by a [`csla_window::CslaWindowInputs`]); the
//!     `CslaPre` / `CslaPending` / `CslaSchedule` / `CslaParam` /
//!     `CslaSlotKind` carriers live here too.
//!   - [`identity`] — the checkpoint-identity emission at setup: the
//!     resolved train/optimizer/scheduler record (item 4) and the
//!     full-state resume load (Milestone B).
//!   - [`epoch_close`] — the batch-loop seal, the `on_epoch` callbacks,
//!     the epoch increment and the jump back to the epoch header.
//!   - [`optimizer_step`] — sections 7e4–7g: the accumulation gate, the
//!     mode-table / FASE-deferred / stdlib step arms, the ZeRO reduce and
//!     sync, and the post-optimizer cleanup, fed by an
//!     [`optimizer_step::OptimizerStepInputs`].
//!   - [`scheduler_step`] — sections 7g2–7h, after the optimizer step: the
//!     scheduler call that redefines the learning rate, the step-count
//!     increment and the periodic full-train-state checkpoint, fed by a
//!     [`scheduler_step::SchedulerStepInputs`].
//!   - [`adjoint_tape_opt`] — sections 6a–6b.5 of the source-AD arm: the
//!     WRGA backward-live filter, dead-gradient elimination, the bit-exact
//!     backward folds and the CSLA schedule report, fed by an
//!     [`adjoint_tape_opt::AdjointTapeOptInputs`] and returning the
//!     parameter-gradient adjoint set; a pure tape rewrite.
//!   - [`ccr_adjoint_frees`] — section 6d of the source-AD arm: CCR's
//!     adjoint-region last-use freeing (the protected gradient set, the
//!     pre-insertion wgrad fusion plan, the in-place `FreeTensor`
//!     insertion), fed by a [`ccr_adjoint_frees::CcrAdjointFreesInputs`];
//!     a pure tape rewrite.
//!   - [`transient_arena_projection`] — section 6e of the source-AD arm:
//!     the Stage-2A element hints, the arena report and the Stage-2B
//!     `--transient-arena` placement with its runtime slot declarations,
//!     fed by a [`transient_arena_projection::TransientArenaInputs`] and
//!     returning the element hints.
//!   - [`csla_precompute`] — the D2b part 2 CSLA schedule precompute of
//!     the source-AD arm: the layerwise plan, per-param facts, replay
//!     ranges and update grouping, and the `--weight-stream` sliced-forward
//!     plan, fed by a [`csla_precompute::CslaPrecomputeInputs`] and
//!     returning the `CslaPre` / `WsForwardPlan` pair; pure analysis.
//!   - [`forward_lowering`] — the forward lowering of the source-AD arm:
//!     the memory-planner tape-unchanged assertion, the Item 11
//!     per-segment early-free plan and the monolithic or segment-streamed
//!     primal lowering, fed by a [`forward_lowering::ForwardLoweringInputs`]
//!     and returning the early-free plan and the lowered forward.
//!   - [`plan_ccr`] — the WRGA fork and the CCR planning of the source-AD
//!     arm: the positional-reference guard, the effective primal, and the
//!     CCR plan (blocks, stride, budget, compression, the owned-tensor
//!     restriction), fed by a [`plan_ccr::PreForwardPlanInputs`] and
//!     returning a [`plan_ccr::PreForwardPlans`]; pure planning.
//!   - [`adapter_sites`] — the WRGA adapter sites of the source-AD arm:
//!     the override-rejected diagnostics, the adapter init side-table and
//!     the adapter-tensor loads into the VarMap, fed by an
//!     [`adapter_sites::AdapterSitesInputs`].
//!   - [`plan_wrga_cpdt`] — the WRGA driver run and the CPDT planning
//!     site of the source-AD arm (tier agreement, moment-precision
//!     arbitration, the stale-plan refusal), fed by a
//!     [`plan_wrga_cpdt::WrgaCpdtInputs`] and returning the WRGA plan;
//!     pure planning.
//!   - [`plan_wggo`] — the WGGO planning site of the source-AD arm:
//!     pre-plan reuse, the planner run, the applied-plan derivation and the
//!     `WggoOverrides` publication, fed by a
//!     [`plan_wggo::WggoPlanningInputs`] and returning a
//!     [`plan_wggo::WggoPlanning`]; pure planning.
//!   - [`plan_csha_prune`] — the CSHA planner schedule and the WGGO prune
//!     of the source-AD arm (with the ELTLS tape-held free and the
//!     `NSL_DEBUG_WENGERT` dump between them), fed by a
//!     [`plan_csha_prune::CshaPruneInputs`]; the prune is the one place
//!     WGGO mutates the tape.
//!   - [`fase_hook_lowering`] — the FASE-hook arm of section 7 of the
//!     source-AD arm: the adjoint lowering with the per-parameter
//!     accumulate callback and the grad-integrity bracket, fed by a
//!     [`fase_hook_lowering::FaseHookLoweringInputs`] and returning the
//!     lowered adjoint.
//!   - [`primal_vars`] — section 3 of the source-AD arm: the initial
//!     `VarMap` (named inputs / parameters to their Cranelift values, the
//!     input device guards, the nested parameter and frozen teacher loads,
//!     the CPKD report facts), fed by a [`primal_vars::PrimalVarsInputs`]
//!     and returning the map.
//!   - [`source_ad_grads`] — section 8 of the source-AD arm: the
//!     parameter-gradient list (or the FASE-hook null sentinel) and the
//!     ownership sweep of the lowering's intermediates, fed by a
//!     [`source_ad_grads::SourceAdGradsInputs`] and returning the arm's
//!     value.
//!   - [`pipelined`] — the `pipeline(...)` sibling of the driver, whole:
//!     `compile_train_block_pipelined` (the dispatch entry point, which
//!     installs the fused-CE decorator config around the lowering) and
//!     `compile_train_block_pipelined_inner` (the stage loop with logical
//!     stage-to-stage communication in one process).
//!   - [`health_hooks`] — sections 7e1b–7e1c, after the backward: the
//!     `--debug-training` gradient checksum, the P0.3 grad-integrity scan
//!     and the health-monitor hooks (loss record, per-parameter gradient
//!     and weight norms, snapshot flush), fed by a
//!     [`health_hooks::HealthHooksInputs`].
//!   - [`param_lists`] — the per-parameter runtime lists built at setup:
//!     the Muon/AdamW route flags, the weight-decay exemption flags and
//!     the gradient-accumulation buffers.
//!   - [`teardown`] — every emission after the epoch loop's exit block: free
//!     the lists, sweep the trailing CSLA window, restore streamed
//!     weights, print the CUDA-graphs banner.
//!
//! Every peel is a byte-for-byte move of the emission under the
//! train-block CLIF snapshots (`tests/train_clif_snapshots.rs`): the
//! instruction stream a fixture lowers to must not change. The CSLA
//! window helpers live beside this module in `stmt_csla.rs`; the FASE
//! optimizer-step emitters in `stmt_fase.rs`.

pub(crate) mod adapter_sites;
pub(crate) mod adjoint_tape_opt;
pub(crate) mod ccr_adjoint_frees;
pub(crate) mod config;
pub(crate) mod model_params;
pub(crate) mod optimizer_state;
pub(crate) mod contract;
pub(crate) mod csla_precompute;
pub(crate) mod csla_window;
pub(crate) mod epoch_close;
pub(crate) mod fase_hook_lowering;
pub(crate) mod forward_lowering;
pub(crate) mod health_hooks;
pub(crate) mod identity;
pub(crate) mod optimizer_step;
pub(crate) mod param_lists;
pub(crate) mod pipelined;
pub(crate) mod primal_vars;
pub(crate) mod scheduler_step;
pub(crate) mod source_ad_grads;
pub(crate) mod teardown;
pub(crate) mod transient_arena_projection;
pub(crate) mod plan_ccr;
pub(crate) mod plan_csha_prune;
pub(crate) mod plan_wggo;
pub(crate) mod plan_wrga_cpdt;
