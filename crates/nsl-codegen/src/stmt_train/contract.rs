//! Section 2 of the train block: resolve the optimizer / scheduler /
//! callbacks contract, compile the `data:` section and the bare
//! statements, refuse the compositions the Muon perf flags do not lower,
//! and plan FASE.
//!
//! Moved out of `compile_train_block_inner` byte-for-byte (roadmap A1).
//! This is the block the roadmap calls the natural `TrainPlan` seed: it
//! reads the train block and the resolved configuration and produces the
//! eighteen bindings everything after it lowers from — which is what
//! [`TrainContract`] is. The driver destructures it, so every binding
//! keeps the name it had as a local, and the rest of the function is
//! untouched.
//!
//! The accept path is pinned by `tests/train_clif_snapshots.rs` (28
//! snapshots across the optimizer / scheduler / callback fixtures); the
//! refusal paths by the CLI composition gates, which pin each message to
//! this file (`feature_rules.rs`, `STMT_CONTRACT`); the `[fase]` planner
//! lines by `exec_markers.rs` and `fase_decorator_activation_gate.rs`.

use cranelift_frontend::FunctionBuilder;
use nsl_ast::block::TrainSection;
use nsl_semantic::train_config::TrainConfigPurpose;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::param_roles::NoDecayScope;
use crate::stmt::is_data_section_config_pair;

/// What section 2 resolves: the optimizer's scalars, the scheduler, the
/// step body and callbacks, the pass-scheduler handle and the FASE plan.
/// Field names are the driver's binding names.
pub(crate) struct TrainContract<'ast> {
    pub(crate) optimizer_name: String,
    pub(crate) lr_value: f64,
    pub(crate) momentum_value: f64,
    pub(crate) dampening_value: f64,
    pub(crate) weight_decay_value: f64,
    pub(crate) no_decay_scope: NoDecayScope,
    pub(crate) nesterov_value: bool,
    pub(crate) beta1_value: f64,
    pub(crate) beta2_value: f64,
    pub(crate) eps_value: f64,
    pub(crate) ns_steps_value: f64,
    pub(crate) adamw_lr_value: Option<f64>,
    pub(crate) scheduler: Option<nsl_semantic::optim_config::ResolvedScheduler>,
    /// The `step(param):` body and its parameter symbol (a train block
    /// without a step section is refused inside).
    pub(crate) step_body: &'ast nsl_ast::stmt::Block,
    pub(crate) step_param_sym: nsl_ast::Symbol,
    pub(crate) callbacks: Vec<&'ast nsl_ast::block::CallbackDef>,
    pub(crate) fase_plan: crate::fase::FasePlan,
    pub(crate) fase_deferred: bool,
}

impl Compiler<'_> {
    /// Resolve the train block's contract (see the module header).
    pub(crate) fn resolve_train_contract<'ast>(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        train: &'ast nsl_ast::block::TrainBlock,
        grad_accumulation_steps: i64,
        grad_clip: f64,
        purpose: TrainConfigPurpose,
    ) -> Result<TrainContract<'ast>, CodegenError> {
        // Same ONE-resolver pattern as the header above: nsl-semantic owns
        // the closed per-optimizer kwarg tables, the scheduler name/kwarg
        // tables WITH their defaults, the Muon spec-default backfill, and
        // the no_decay role validation. The old inline parse here matched
        // known kwargs with `_ => {}` (a typo'd `lrr=` silently trained at
        // the default lr), dropped known kwargs whose literal shape didn't
        // match (`momentum=1` — an int — kept 0.0), skipped positional
        // args, resolved duplicates last-wins, and let an unknown
        // scheduler name fall through to "no change" (constant lr).
        let optim_cfg = nsl_semantic::optim_config::resolve_optim_config(
            &train.sections,
            train.span,
            &|sym| self.resolve_sym(sym).to_string(),
            purpose,
        )
        .map_err(|diags| {
            let msgs: Vec<String> = diags.into_iter().map(|d| d.message).collect();
            CodegenError::new(format!(
                "optimizer config refused: {}",
                msgs.join("; ")
            ))
        })?;

        let optimizer_name = optim_cfg.optimizer.kind.as_str().to_string();
        let lr_value: f64 = optim_cfg.optimizer.lr;
        let momentum_value: f64 = optim_cfg.optimizer.momentum;
        let dampening_value: f64 = optim_cfg.optimizer.dampening;
        let weight_decay_value: f64 = optim_cfg.optimizer.weight_decay;
        // AdamW parameter groups (`no_decay=[...]`). Empty = every param
        // decays, bit-identical to before this existed.
        let no_decay_scope = crate::param_roles::NoDecayScope {
            static_roles: optim_cfg.optimizer.no_decay_static_roles.clone(),
            exempt_non_rank2: optim_cfg.optimizer.no_decay_exempt_non_rank2,
        };
        let nesterov_value: bool = optim_cfg.optimizer.nesterov;
        let beta1_value: f64 = optim_cfg.optimizer.beta1;
        let beta2_value: f64 = optim_cfg.optimizer.beta2;
        let eps_value: f64 = optim_cfg.optimizer.eps;
        // P5 Muon: Newton-Schulz iteration depth (spec default 5).
        let ns_steps_value: f64 = optim_cfg.optimizer.ns_steps;
        // P1 Muon item 5: separate learning rate for the AdamW arm of the
        // mixed Muon/AdamW step (embeddings/head/vectors). None → the
        // AdamW arm follows `lr` exactly. Threaded as a RATIO of lr so a
        // scheduler modulates both arms coherently.
        let adamw_lr_value: Option<f64> = optim_cfg.optimizer.adamw_lr;
        // Fully validated, defaults applied (including one_cycle's
        // max_lr = 10x lr) — the lowering at 7g2 just emits constants.
        let scheduler: Option<nsl_semantic::optim_config::ResolvedScheduler> =
            optim_cfg.scheduler;
        let mut step_body: Option<(&nsl_ast::stmt::Block, nsl_ast::Symbol)> = None;
        let mut callbacks: Vec<&nsl_ast::block::CallbackDef> = Vec::new();

        for section in &train.sections {
            match section {
                TrainSection::Optimizer(_) => {
                    // Fully consumed by resolve_optim_config above —
                    // constructor name, per-optimizer kwarg namespace,
                    // literal/range validation, no_decay roles, and the
                    // Muon spec-default backfill all live in the resolver.
                }
                TrainSection::Step { param, body } => {
                    step_body = Some((body, *param));
                }
                TrainSection::Callbacks(cbs) => {
                    callbacks.extend(cbs.iter());
                }
                TrainSection::Scheduler(_) => {
                    // Fully consumed by resolve_optim_config above — name
                    // canonicalization, per-scheduler kwarg namespace, and
                    // the defaults all live in the resolver.
                }
                TrainSection::Data(stmts) => {
                    // Compile data section stmts — typically creates a DataLoader.
                    // `key = expr` config pairs (e.g. `source = PretrainCorpus`)
                    // are PCA-detection metadata consumed via the AST walker in
                    // `pca_activation.rs`; they MUST NOT flow through
                    // `compile_assign`, which would try to look up `key` as a
                    // variable. Skip the allowlisted keys here and let
                    // anything else (statements, future loader builders) go
                    // through the standard compile path. Keep this list in
                    // sync with `nsl-semantic/src/checker/block.rs`
                    // (`DATA_SECTION_KEYS`).
                    for stmt in stmts {
                        if is_data_section_config_pair(stmt, self.interner) {
                            continue;
                        }
                        self.compile_stmt(builder, state, stmt)?;
                    }
                }
                // Bare statements in a train block execute once, pre-training,
                // in source position — same treatment as Data-section
                // statements (the semantic checker already declared their
                // symbols). These were previously silently dropped.
                TrainSection::Stmt(s) => {
                    self.compile_stmt(builder, state, s)?;
                }
                // Deferral-must-refuse: these sections parse and type-check but
                // are not executed — silently dropping them meant documented
                // eval/best-checkpoint logic never ran with no diagnostic.
                TrainSection::Eval { .. } => {
                    return Err(CodegenError::new(
                        "train block `eval:` sections are not yet executed; move \
                         evaluation logic into an `on_epoch` callback (which \
                         receives the epoch and loss) so it actually runs",
                    ));
                }
                TrainSection::Distribute(_) => {
                    return Err(CodegenError::new(
                        "train block `distribute:` sections are not supported; \
                         configure distribution via the @pipeline decorator / \
                         CLI options instead",
                    ));
                }
            }
        }

        // (Missing-optimizer refusal, the Muon spec-default backfill, and
        // the adamw_lr x lr guard all moved into resolve_optim_config —
        // they now fire at `nsl check` time too, and identically on the
        // pipelined path.)

        // Muon perf campaign (`--muon-batch-ns`): the batched engine is
        // wired into the FullBuffer optimizer loop only. Every path where
        // it would silently not batch (or corrupt state) refuses loudly.
        if self.compile_options.muon.batch_ns {
            if optimizer_name != "muon" {
                return Err(CodegenError::new(format!(
                    "--muon-batch-ns requires the muon optimizer (train block \
                     uses '{optimizer_name}'). Drop the flag"
                )));
            }
            if self.compile_options.layerwise_accum {
                return Err(CodegenError::new(
                    "--muon-batch-ns does not compose with --layerwise-accum \
                     yet: CSLA fires per-layer group updates at window \
                     boundaries, and cross-layer batching there would change \
                     the accumulation discipline. Drop one of the flags",
                ));
            }
            if self.compile_options.optim_state_offload {
                return Err(CodegenError::new(
                    "--muon-batch-ns does not compose with \
                     --optim-state-offload: the batched kernels update the \
                     momentum in place on the DEVICE, but offload keeps it \
                     host-resident. Use --muon-resident-momentum (device m) \
                     or drop one of the flags",
                ));
            }
            if self.compile_options.muon.state_bf16 {
                return Err(CodegenError::new(
                    "--muon-batch-ns does not compose with --muon-state-dtype \
                     bf16: the batched kernels read/write f32 momentum \
                     directly. Drop one of the flags",
                ));
            }
            if self.compile_options.param_dtype_bf16sr {
                return Err(CodegenError::new(
                    "--muon-batch-ns does not compose with --param-dtype \
                     bf16-sr: the batched kernels read/write f32 params \
                     directly. Drop one of the flags",
                ));
            }
            if self.features.zero_stage.filter(|&s| s >= 1).is_some() {
                return Err(CodegenError::new(
                    "--muon-batch-ns does not compose with --zero-stage: the \
                     batch call updates every listed param, but ZeRO shards \
                     ownership per rank. Drop one of the flags",
                ));
            }
        }

        // Muon perf campaign (`--muon-resident-momentum`): only meaningful
        // under offload, and only for the muon optimizer's routed params.
        if self.compile_options.muon.resident_momentum {
            if optimizer_name != "muon" {
                return Err(CodegenError::new(format!(
                    "--muon-resident-momentum requires the muon optimizer \
                     (train block uses '{optimizer_name}'). Drop the flag"
                )));
            }
            if !self.compile_options.optim_state_offload {
                return Err(CodegenError::new(
                    "--muon-resident-momentum is only meaningful with \
                     --optim-state-offload (without offload the momentum is \
                     already device-resident). Drop the flag",
                ));
            }
            if self.compile_options.muon.state_bf16 {
                return Err(CodegenError::new(
                    "--muon-resident-momentum does not compose with \
                     --muon-state-dtype bf16 (the bf16 envelope owns the \
                     momentum layout). Drop one of the flags",
                ));
            }
            if self.features.zero_stage.filter(|&s| s >= 1).is_some() {
                return Err(CodegenError::new(
                    "--muon-resident-momentum does not compose with \
                     --zero-stage (owner-gated moment allocation). Drop one \
                     of the flags",
                ));
            }
        }

        let (step_body, step_param_sym) =
            step_body.ok_or_else(|| CodegenError::new("train block requires a step section"))?;

        // FASE: plan the backward rewrite.  Passthrough (N=1) and FullBuffer
        // (Lion, Unknown) fall through to the existing accum-buffer path
        // below.  Deferred routes through stmt_fase.
        //
        // Milestone A: the `@fase(...)` decorator captured by the Decorated
        // arm configures the plan here — previously its validated config was
        // discarded at the checker and the two knobs below were hard-coded.
        // `take()` so a later train block never inherits this one's config.
        let fase_decorator = self.fase_decorator.take();
        let fase_forced_mode = fase_decorator.as_ref().and_then(|d| match d.mode {
            nsl_semantic::cftp::FaseMode::Auto => None,
            nsl_semantic::cftp::FaseMode::Off => Some(crate::fase::FaseForce::Off),
            nsl_semantic::cftp::FaseMode::FullBuffer => {
                Some(crate::fase::FaseForce::FullBuffer)
            }
            nsl_semantic::cftp::FaseMode::Deferred => Some(crate::fase::FaseForce::Deferred),
        });
        let fase_cfg = crate::fase::FaseConfig {
            accumulation: grad_accumulation_steps.max(1) as u32,
            optimizer: crate::fase::FaseOptimizer::parse(&optimizer_name),
            grad_clip: if grad_clip < f64::MAX { Some(grad_clip) } else { None },
            lr: lr_value,
            beta1: beta1_value,
            beta2: beta2_value,
            eps: eps_value,
            weight_decay: weight_decay_value,
            momentum: momentum_value,
            allow_v_approx: fase_decorator.as_ref().map(|d| d.allow_v_approx).unwrap_or(true),
            forced_mode: fase_forced_mode,
        };
        // Milestone C: SCHEDULED, with the body widened through the driver's
        // muon x --layerwise-accum rewrite and the disposition re-record —
        // the scheduler prints `<- FASE disposition=` right after the body
        // returns, and a body that ended at `fase::plan` would print the
        // pass's self-report which the driver overwrites just below
        // (Declined flipped to Applied on the muon arm), leaving the trace
        // line disagreeing with the report. tape=None: TapeAccess::None, and
        // no WengertList exists yet (the extractor is built ~1,900 lines
        // down). finish() is vacuous (FASE publishes no channel) but keeps
        // the template uniform. FASE's real staleness guard is value-level —
        // the fase_fused divergence refusal at the WGGO replan site — and
        // stays where it is.
        //
        // The body is fallible (the Milestone A `@fase(...)` decorator
        // refusals below), so it returns a Result that the site unwraps
        // AFTER finish() — the same `??` shape as the WGGO site. The
        // postconditions still run on the refusal path; keeping the checks
        // inside the window is what makes the decorator's verdict part of
        // the scheduled pass rather than a driver-side afterthought.
        let sched = self.passes.scheduler();
        let fase_plan = sched.schedule("FASE", None, || {
        let plan = match self.bus.wggo_overrides() {
            Some(o) => {
                let mut fused: Vec<bool> = o.per_layer.iter().map(|p| p.fase_fused).collect();
                // Diagnostic knob: NSL_FASE_FUSED_OVERRIDE="1,0,..." replaces
                // the plan's per-layer fase_fused pattern (layer order = the
                // WGGO override order). Used by the mixed-mode differential
                // tests to pin a deterministic mode table; the layer registry
                // (names → params) still comes from the real WGGO overrides.
                if let Ok(spec) = std::env::var("NSL_FASE_FUSED_OVERRIDE") {
                    // Strict parse: only 0/1/true/false are meaningful. Any
                    // other token (including an empty-but-set variable, or
                    // "True"/"yes") rejects the WHOLE spec with a warning —
                    // silently mapping unrecognized tokens to FullBuffer
                    // would flip production mode tables on a typo'd or
                    // stray exported variable.
                    let forced: Option<Vec<bool>> = spec
                        .split(',')
                        .map(|t| match t.trim() {
                            "1" | "true" => Some(true),
                            "0" | "false" => Some(false),
                            _ => None,
                        })
                        .collect();
                    match forced {
                        Some(forced) if forced.len() == fused.len() => {
                            nsl_runtime::nsl_log!(INFO, "fase", 
                                "[fase] NSL_FASE_FUSED_OVERRIDE applied: {spec} \
                                 (replacing plan fase_fused for {} layers)",
                                fused.len()
                            );
                            fused = forced;
                        }
                        Some(forced) => {
                            nsl_runtime::nsl_log!(WARN, "fase", 
                                "[fase] NSL_FASE_FUSED_OVERRIDE ignored: {} entries \
                                 for {} WGGO layers",
                                forced.len(),
                                fused.len()
                            );
                        }
                        None => {
                            nsl_runtime::nsl_log!(WARN, "fase", 
                                "[fase] NSL_FASE_FUSED_OVERRIDE ignored: \
                                 unrecognized token in '{spec}' (only 0/1/true/false)"
                            );
                        }
                    }
                }
                crate::fase::plan_with_overrides(&fase_cfg, &fused)
            }
            None => crate::fase::plan(&fase_cfg),
        };

        // Render FaseModeInfeasible diagnostics to stderr in the same format
        // as CSHA / WRGA / CPDT so the Phase 3 decision explainer parses
        // uniformly.
        for diag in &plan.override_diagnostics {
            let reason_str = match &diag.reason {
                crate::wggo_overrides::OverrideRejectReason::FaseModeInfeasible {
                    optimizer,
                    global_mode,
                } => format!("{:?}_optimizer_global_mode_{:?}", optimizer, global_mode)
                    .to_lowercase(),
                other => format!("{:?}", other),
            };
            nsl_runtime::nsl_log!(INFO, "fase", 
                "[fase] layer:{} wggo-override-rejected requested={} applied={} reason={}",
                diag.layer_index, diag.requested, diag.applied, reason_str
            );
        }
        // FASE Codegen Phase 2+3 shipped: the accumulation loop `ga_body`
        // below dispatches per-param via a `.rodata` mode table built from
        // WGGO's per-layer decisions (see `mode_table_base` allocation below
        // + `emit_fase_mode_branch` in stmt_fase.rs), and the optimizer step
        // dispatches per-param through `emit_unified_optim_step_dispatch`.
        // Two-phase clip honors mixed tables on the source-AD hook path
        // (the only one where a mode table exists): the hook accumulates
        // every param with the scaled window-mean convention, so Phase A's
        // global norm is uniform across modes, and Phase B applies the
        // shared clip factor in both dispatch arms.
        // See docs/superpowers/specs/2026-04-15-fase-codegen-phase2-design.md.
        //
        // P1 Muon item 11: under the layerwise schedule, muon runs a
        // Deferred-SHAPED "separate-accumulator" plan — the window backward
        // accumulates RAW gradient sums into m_partial (accum_scale forced
        // to 1.0, the FullBuffer convention, so the boundary muon_step
        // consumes bit-identical input to the non-CSLA path) and the
        // per-layer group updates dispatch the stdlib muon_step instead of
        // a fused elementwise recipe (see emit_csla_group_update's muon
        // arm). Off the layerwise path muon keeps its FullBuffer plan —
        // this override is CSLA-scoped on purpose. The exact one-buffer
        // "classical Muon" accumulation (folding the window sum into the
        // momentum buffer itself) is intentionally NOT this mode; it would
        // ship as a separately named mode when it lands.
        let mut plan = plan;
        if optimizer_name == "muon" && self.compile_options.layerwise_accum {
            plan.mode = crate::fase::FaseMode::Deferred;
            plan.recipe.accum_scale = 1.0;
            plan.rationale = format!(
                "{} (muon x layerwise-accum: Deferred-shaped raw-sum window \
                 accumulation, stdlib muon_step at group updates)",
                plan.rationale
            );
        }
        let plan = plan;

        // Milestone A: the decorator checks sit AFTER the muon x
        // --layerwise-accum rewrite above — review caught the first version
        // running them before it, which refused `@fase(mode = deferred)` on
        // exactly the build that delivers Deferred, and let the rewrite
        // silently override `@fase(mode = off)` while the witness printed
        // the stale mode.
        //
        // A forced-off/full_buffer decorator that the muon rewrite would
        // override is a CONFLICT, not a precedence question: the rewrite
        // exists because layerwise muon requires the Deferred window, so
        // honouring the decorator would break the schedule and ignoring it
        // would break the decorator's contract. Refuse with both facts.
        if matches!(
            fase_forced_mode,
            Some(crate::fase::FaseForce::Off | crate::fase::FaseForce::FullBuffer)
        ) && plan.mode == crate::fase::FaseMode::Deferred
        {
            return Err(CodegenError::new(
                "@fase(mode = off/full_buffer) conflicts with the muon x \
                 --layerwise-accum schedule, which requires the Deferred \
                 accumulation window. Remove the decorator, or drop \
                 --layerwise-accum / switch the optimizer"
                    .to_string(),
            ));
        }

        // `@fase(mode = deferred)` is a REQUIREMENT, not a preference —
        // checked against the FINAL mode (post-rewrite). If the build could
        // not produce Deferred, refuse with requested-vs-derived rather than
        // silently downgrading (transformation-precondition-refusal).
        if matches!(fase_forced_mode, Some(crate::fase::FaseForce::Deferred))
            && plan.mode != crate::fase::FaseMode::Deferred
        {
            return Err(CodegenError::new(format!(
                "@fase(mode = deferred) cannot be honoured: requested the \
                 Deferred envelope, but the build derived {:?} — {}. \
                 Use @fase(mode = auto) to accept the derived mode, or \
                 change the optimizer/accumulation so Deferred is feasible",
                plan.mode, plan.rationale,
            )));
        }

        // The @fase activation witness — the FINAL mode, after every
        // rewrite. Emitted only when the decorator is present, so vanilla
        // builds' stderr is unchanged.
        if fase_decorator.is_some() {
            nsl_runtime::nsl_log!(INFO, "fase", 
                "[fase] @fase decorator applied: mode={:?} v_approx={} — {}",
                plan.mode, fase_cfg.allow_v_approx, plan.rationale,
            );
        }
        // Item 3: re-record FASE's disposition AFTER the driver's own rewrite
        // above. `fase::plan` is accurate about the pass and can be wrong
        // about the build: the muon x --layerwise-accum arm overwrites `mode`
        // and `rationale` here, and a `Passthrough` plan flipped to `Deferred`
        // would otherwise leave the report saying "declined, mode off" for a
        // build in which FASE is active. Last-wins in `record_disposition` is
        // what makes the later, build-truthful statement the one that shows.
        //
        // The disposition payload deliberately carries no mode string
        // (`&'static str` only, and `rationale` is a runtime `String`), so the
        // phase count is what is reported; the mode itself stays visible in
        // the existing `[fase]` diagnostics.
        if matches!(fase_forced_mode, Some(crate::fase::FaseForce::Off)) {
            // Milestone A: `@fase(mode = off)` with accumulation > 1 plans
            // the FullBuffer fallback, but reporting that as "applied" would
            // tell a user who turned FASE OFF that it ran — record the
            // decline the request actually was.
            crate::pass_trace::record_disposition("FASE", crate::pass_trace::PassDisposition::Declined {
                reason: crate::pass_trace::DeclineReason::FeatureDisabled("@fase(mode = off)"),
            });
        } else if plan.mode == crate::fase::FaseMode::Passthrough {
            crate::pass_trace::record_disposition("FASE", crate::pass_trace::PassDisposition::Declined {
                reason: crate::pass_trace::DeclineReason::ModeOff,
            });
        } else {
            crate::pass_trace::record_disposition("FASE", crate::pass_trace::PassDisposition::Applied {
                rewrites: plan.backward_phases.len(),
            });
        }
        Ok(plan)
        })
        .map_err(CodegenError::new)?
        .finish(&self.bus)
        .map_err(CodegenError::new)??;
        let fase_deferred = fase_plan.mode == crate::fase::FaseMode::Deferred;

        Ok(TrainContract {
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
        })
    }
}
