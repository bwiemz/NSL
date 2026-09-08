use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{BlockArg, InstBuilder, MemFlagsData};
use cranelift_frontend::{FunctionBuilder, Variable};
use cranelift_module::Module;

use nsl_ast::block::{QuantDtype, QuantGranularity};
use nsl_ast::expr::ExprKind;
use nsl_ast::operator::AssignOp;
use nsl_ast::pattern::PatternKind;
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_semantic::types::Type;

use crate::compiler::Compiler;
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
use crate::types::{is_block_filled, nsl_type_to_cl};
use cranelift_codegen::ir::Value;

// P0.1 per-surface VRAM accounting: the wire values of
// `nsl_gpu_set_alloc_surface` / `nsl_gpu_get_alloc_surface`. Declared once in
// `nsl_abi::wire::surface` (roadmap A3), which is also where the runtime's
// `SurfaceTag` (#[repr(u8)]) takes its discriminants — so the two cannot
// drift. Widened to i64 here because they are emitted as `iconst I64`.
// Each train-block bracket sets a surface for its allocation region and
// restores the caller's surface afterwards (get/set — nesting-safe).
pub(crate) const SURFACE_WEIGHTS: i64 = nsl_abi::wire::surface::SURFACE_WEIGHTS as i64;
pub(crate) const SURFACE_OPTIM_M: i64 = nsl_abi::wire::surface::SURFACE_OPTIM_M as i64;
pub(crate) const SURFACE_OPTIM_V: i64 = nsl_abi::wire::surface::SURFACE_OPTIM_V as i64;
pub(crate) const SURFACE_M_PARTIAL: i64 = nsl_abi::wire::surface::SURFACE_M_PARTIAL as i64;
pub(crate) const SURFACE_GRADS: i64 = nsl_abi::wire::surface::SURFACE_GRADS as i64;
pub(crate) const SURFACE_ACTIVATIONS: i64 = nsl_abi::wire::surface::SURFACE_ACTIVATIONS as i64;

/// FASE hook: one parameter's primal Value and its `accum_list` /
/// `param_list` index, keyed by its adjoint gradient VarId in the driver's
/// `adj_vid_to_hook_entry` (the CSLA save phase copies the indices into
/// its pending carrier; the FASE Deferred arm consumes the entries).
pub(crate) struct ParamHookEntry {
    pub(crate) primal_val: Value,
    // i64 index into accum_list (== param_list index for this param)
    pub(crate) accum_idx: i64,
}

/// Item C: how ONE parameter's optimizer moments are allocated under
/// `--zero-stage 3`, decided from its `ParameterPlan` entry and consumed by
/// `Compiler::emit_deferred_moment_fill`. Section 4 allocates nothing under
/// stage 3 (the plan and the runtime carve both post-date it), so this is
/// the single place the three shapes are spelled.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum MomentFill {
    /// `--zero-elementwise` eligible: a persistent 1/world_size SLICE on
    /// every rank, sized by the runtime from the carved shard.
    Elementwise,
    /// Tensor-granular sharded: the owner allocates the full moment, the
    /// rest keep the null placeholder (the stages-1/2 machinery, reused).
    OwnerGated,
    /// Replicated (tied / view-rooted / epilogue): full m/v on every rank,
    /// because every rank updates it from all-reduced gradients.
    Full,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum SourceAdParamDiagnosticKind {
    Trainable,
    IgnoredConfig,
    IgnoredNonTensor,
}

/// HOW a train block arrived at its `grad_accumulation` window — kept apart
/// from the window VALUE because a diagnostic that says "1" needs to say why.
///
/// A `NonLiteral` variant used to live here: `train(..., grad_accumulation=GA)`
/// with a `const GA = 4` parsed, type-checked, and then lowered a window of
/// **1** with no diagnostic. The Training Configuration Contract
/// (`nsl_semantic::train_config`) now REFUSES a non-literal window — the
/// same contract `distill` always had — so a Literal here means the window
/// is exactly what the source says.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum GradAccumulationDecl {
    /// No `grad_accumulation=` in the train block's config at all.
    Omitted,
    /// Present as an integer literal — the window is what it says.
    Literal,
}

/// Item 12: result of analyzing whether a train-loop callback body touches
/// the streamed model θ (see `Compiler::analyze_callback_model_touch`).
#[derive(Default, Debug)]
struct CallbackModelTouch {
    /// The callback references the model (a field read, a method call, an
    /// `Ident` passed to `model_save`/a helper, …). Requires a scoped upload
    /// under `--weight-stream` or its reads launch on evicted (null) data.
    touches: bool,
    /// The callback may MUTATE θ (an assignment rooted at the model, or a
    /// method call on the model/one of its fields). Drives writeback=1 on the
    /// closing re-evict so the mutation survives the next window's upload.
    may_write: bool,
    /// First model-rooted access path seen, for the compile-time diagnostic.
    first_path: Option<String>,
}

/// Item 11 calibration: fixed DMA issue+completion latency (μs) added to every
/// pack-transfer estimate — small PCIe copies are latency-bound, not
/// bandwidth-bound, so a bytes/BW model alone would price a 4 KiB pack at
/// ~0.1 μs and activate overlap that cannot pay. Combined with the target
/// `GpuSpec`'s `pcie_bandwidth_gbps` / `kernel_launch_overhead_ns` /
/// `peak_bandwidth_gbs`, this closes the deferred WGGO-ILP cost integration:
/// each prefetch edge is priced per-range compute μs vs pack-byte transfer μs.
pub(crate) const WS_PCIE_FIXED_LAT_US: f64 = 10.0;

/// Fallback for UNPRICED packs (a member's shape is not statically concrete —
/// e.g. a bare `Tensor` field annotation): the v1 structural heuristic, which
/// the GPU bit-exactness gates shipped and validated under. A priced edge is
/// calibrated-safe; an unpriced edge is merely heuristic (review M3: never
/// treat a 0-byte pricing as a real transfer estimate).
pub(crate) const WS_PREFETCH_MIN_OPS_PER_RANGE: usize = 4;

/// `pub(crate)` for the drift gate in `source_ad.rs`: constant-folding a
/// config field's `.item()` is sound only for leaves this function keeps OUT
/// of the parameter list, and that implication is asserted against the real
/// function rather than a restatement of it.
pub(crate) fn is_trainable_param_leaf_name(param_name: &str) -> bool {
    let leaf_name = param_name.rsplit('.').next().unwrap_or(param_name);
    !leaf_name.starts_with('_') && leaf_name != "inv_freq"
}

/// CFTP v10 (item 5): return the declared tensor rank of `ty` when it is
/// unambiguously a tensor with a non-empty shape.
///
/// `nsl_semantic::types::Shape::unknown()` and `Shape::scalar()` both
/// produce `Shape { dims: vec![] }`, so we cannot distinguish an
/// unannotated `Tensor` from a genuine rank-0 scalar tensor.  We treat
/// empty shape as UNKNOWN (`None`) so the matcher preserves its
/// conservative-fire behaviour for unannotated code — the load-bearing
/// rank check runs only when the frontend gave us a rank ≥ 1 to check.
/// A `Borrow(Tensor)` is unwrapped so annotated `&Tensor<[V,H]>`
/// parameters (common for `W` in NSL step signatures) participate too.
pub(crate) fn resolvable_tensor_rank(ty: &Type) -> Option<usize> {
    let inner = match ty {
        Type::Borrow(inner) => inner.as_ref(),
        other => other,
    };
    let rank = match inner {
        Type::Tensor { shape, .. }
        | Type::Param { shape, .. }
        | Type::Buffer { shape, .. }
        | Type::Sparse { shape, .. } => shape.rank(),
        _ => return None,
    };
    if rank == 0 {
        None
    } else {
        Some(rank)
    }
}

/// Allowlist of legal `data:` section config keys. Mirrors
/// `nsl-semantic/src/checker/block.rs::DATA_SECTION_KEYS` — both must stay
/// in sync. A key present here but missing in the semantic table will reach
/// `compile_assign` and fail with an undefined-variable error; a key
/// present in the semantic table but missing here will reach
/// `compile_assign` with the same failure mode. v8 ships with a single
/// canonical key (`source`); future keys should land in both places.
const DATA_SECTION_KEYS: &[&str] = &["source"];

/// Returns true iff `stmt` is a `data:` section config pair of the form
/// `<allowlisted-key> = <expr>` (plain `Assign`, plain ident target). These
/// are PCA-detection metadata consumed via the AST walker in
/// `pca_activation.rs`; they must not be lowered as variable assignments.
pub(crate) fn is_data_section_config_pair(stmt: &Stmt, interner: &nsl_lexer::Interner) -> bool {
    let StmtKind::Assign {
        target,
        op: AssignOp::Assign,
        ..
    } = &stmt.kind
    else {
        return false;
    };
    let ExprKind::Ident(name_sym) = target.kind else {
        return false;
    };
    let name = match interner.resolve(name_sym.0) {
        Some(n) => n,
        None => return false,
    };
    DATA_SECTION_KEYS.contains(&name)
}

/// Task 4: WRGA bridge — build a `WrgaInput` from decorator configs stashed on
/// the Compiler and run `wrga::run` against the primal Wengert list.
///
/// Returns `None` when WRGA is disabled (`wrga_inputs == None`) or when all
/// three decorator sets (`wrga`, `freeze`, `adapter`) are empty.  In that case
/// callers fall back to the unpruned primal/adjoint lists — this matches the
/// Task 5 sanity expectation that an empty `WrgaInputs` is a no-op.
///
/// When a plan is produced, it is published to `compiler.bus.wrga_plan` for
/// later observability (`nsl check --wrga-report`).
/// CPDT driver bridge — mirrors `invoke_wrga_if_enabled`. Builds a `CpdtInput`
/// from the WGGO `AppliedPlan` (when one exists), the optional `@train` block
/// (for AdamW hyperparameters), and the cluster topology stashed on the
/// Compiler, then stores the resulting plan on the `cpdt_plan` bus channel.
///
/// `applied_plan: None` is the weights-only path: the precision plan — the
/// only CPDT product the moment consult reads — is a pure function of the
/// WeightMap (`cpdt::run` step 4 never touches the applied plan), so blocks
/// that get no WGGO plan (distill's synthetic train block, loop-bound train
/// blocks) can still type their optimizer moments from CPDT. The plan-derived
/// inputs degrade explicitly: the ZeRO/comm halves see an empty cost model,
/// the shard recommendation is absent, and the weight-map/plan cross-check
/// has no layers to check.
///
/// No-op when `compiler.cpdt_mode == CpdtMode::Off` or when no cluster is
/// configured.
/// The staleness-refusal test knob, parsed in ONE place: both the
/// pre-plan-path re-arbitration and the weights-only re-check honor it,
/// and a drifted parse between the two arms would make one refusal
/// gate-testable and the other silently not.
pub(crate) fn cpdt_forced_stale_plan() -> bool {
    std::env::var("NSL_CPDT_FORCE_STALE_PLAN")
        .map(|v| v == "1")
        .unwrap_or(false)
}

/// Plan CPDT for one train block and publish the plan on the bus. `Err` is
/// a compile failure — today only the weight-map validation refusal, which
/// the CLI renders and exits on; every "CPDT does not apply here" outcome
/// is a typed decline on the pass trace and `Ok(())`.
pub(crate) fn invoke_cpdt_if_enabled(
    compiler: &mut crate::compiler::Compiler,
    applied_plan: Option<&crate::wggo_apply::AppliedPlan>,
    train_block: Option<&nsl_ast::block::TrainBlock>,
) -> Result<(), CodegenError> {
    // Experimental subsystem (CPDT). Compiled in by default; a build that opts
    // out (`--no-default-features` without `experimental-cpdt`) turns CPDT
    // planning into a no-op here. See STATUS.md / docs/architecture/.
    #[cfg(not(feature = "experimental-cpdt"))]
    {
        let _ = (applied_plan, train_block);
        // Milestone A: a build compiled without the feature cannot honour a
        // CPDT request — say so as a typed decline instead of a silent no-op,
        // or the activation reconciler reports the request as silently inert
        // with no way for the user to tell why.
        // Record unconditionally: `@cpdt(mode = off)` maps to CpdtMode::Off,
        // so gating the record on mode != Off left exactly that request
        // unanswered on feature-stripped builds (review finding). ModeOff is
        // the honest reason when off was requested-or-default; FeatureDisabled
        // when the request wanted CPDT active.
        crate::pass_trace::record("CPDT");
        crate::pass_trace::record_disposition(
            "CPDT",
            crate::pass_trace::PassDisposition::Declined {
                reason: if compiler.cpdt_mode == crate::cpdt::CpdtMode::Off {
                    crate::pass_trace::DeclineReason::ModeOff
                } else {
                    crate::pass_trace::DeclineReason::FeatureDisabled(
                        "this nsl was built without the experimental-cpdt feature",
                    )
                },
            },
        );
        return Ok(());
    }
    use crate::cpdt::{CpdtInput, CpdtMode, run as cpdt_run};
    use crate::cpdt_expert::ExpertConfig;
    use crate::cpdt_joint::JointConfig;
    use crate::cpdt_tier_apply::{compute_tier_agreement, plan_map_noweights, PrecisionConfig};
    use crate::cpdt_zero::ModelSize;
    use crate::wggo_overrides::WggoOverrides;

    // Milestone A: these two early returns were SILENT — `cpdt::run` records
    // the pass's dispositions, but neither exit ever reaches it, so a compile
    // that requested CPDT (flag or decorator) and got nothing looked exactly
    // like one that never asked. Record here, at the decision, mirroring
    // FASE's mode-off recording. `record` first: disposition presupposes it.
    if compiler.cpdt_mode == CpdtMode::Off {
        crate::pass_trace::record("CPDT");
        crate::pass_trace::record_disposition(
            "CPDT",
            crate::pass_trace::PassDisposition::Declined {
                reason: crate::pass_trace::DeclineReason::ModeOff,
            },
        );
        return Ok(());
    }
    let Some(cluster) = compiler.cpdt_cluster.clone() else {
        // Reachable from source alone: `@cpdt(mode = full)` with no cluster
        // argument and no CLI cluster flags leaves `cpdt_cluster` None.
        crate::pass_trace::record("CPDT");
        crate::pass_trace::record_disposition(
            "CPDT",
            crate::pass_trace::PassDisposition::Declined {
                reason: crate::pass_trace::DeclineReason::PreconditionViolated(
                    "no cluster specification — pass --cpdt-num-gpus or give \
                     @cpdt a cluster argument",
                ),
            },
        );
        return Ok(());
    };

    // Weights-only path: an absent plan contributes an empty override set and
    // an empty cost model. `ModelSize::from_applied_plan` on the default
    // (layerless) plan yields empty per-layer vectors, which every consumer
    // treats as a zero-size model rather than an error.
    let overrides = applied_plan.map(WggoOverrides::from_applied);
    let mut model = match applied_plan {
        Some(p) => ModelSize::from_applied_plan(p),
        None => ModelSize::from_applied_plan(&crate::wggo_apply::AppliedPlan::default()),
    };
    // Cost-model audit finding 3: tell the ZeRO evaluator whether
    // @checkpoint(policy=...) activation checkpointing is active for this
    // compile (non-empty checkpoint_policies map). Without it the evaluator
    // charges the full sequential live-set (sum of per-layer activations);
    // with it, a documented checkpoint-aware estimate.
    //
    // Granularity caveat: checkpoint_policies is per-FUNCTION while this
    // flag is per-MODEL — a single checkpointed helper flips the whole
    // model to the sqrt-style estimate, under-charging any layers that are
    // NOT actually checkpointed. The estimate is clamped to the
    // no-checkpoint sum so the optimism is bounded; a per-layer coverage
    // blend needs a layer->function map that AppliedPlan does not carry.
    // P1.7 --training-reference: report no checkpointing in the memory estimate,
    // matching codegen (which ignores @checkpoint decorators in that mode).
    model.activation_checkpointing = !compiler.compile_options.diagnostics.training_reference
        && !compiler.compile_options.checkpoint.policies.is_empty();
    let adamw = adamw_from_train_block(train_block, compiler.interner);

    // Phase 1 weight-aware CPDT: the compiler holds a WeightMap loaded from
    // the CLI's --weights flag. Thread it into the CpdtInput so plan_map
    // runs on real weights.
    let weight_map_ref = compiler.features.weight_map.as_ref();

    // Phase 1 opt-out: `@cpdt(weight_aware=false)` suppresses the weight-aware
    // path entirely. Shadowing `weight_map_ref` to `None` here propagates
    // through every downstream guard:
    //   * plan_map in cpdt::run receives None → returns PrecisionPlan::default().
    //   * tier-agreement diagnostic + CPDT_CALIB_K warning gate on
    //     `weights_present` (derived from weight_map_ref.is_some()) → skipped.
    //   * validate (when PR #90 merges) gates on `if let Some(wm) = weight_map_ref`
    //     → skipped.
    // Design: docs/superpowers/specs/2026-04-20-cpdt-weight-aware-opt-out-design.md.
    let weight_map_ref = if compiler.cpdt_weight_aware {
        weight_map_ref
    } else {
        None
    };

    let weights_present = weight_map_ref.is_some();
    let precision_cfg = PrecisionConfig::default();

    // Phase 1 weight-map validation: when weights + CpdtMode::Full, verify
    // every hierarchical AppliedPlan layer has matching tensors in the
    // WeightMap. Fails fast with an aggregated error naming all missing
    // layers plus a WeightMap-prefix summary. Catches "wrong checkpoint
    // entirely" at plan-time rather than letting CPDT produce corrupt tier
    // assignments for downstream consumers. See
    // docs/superpowers/specs/2026-04-20-cpdt-validate-body-design.md.
    // On the weights-only path there is no plan to validate the WeightMap
    // against — `validate` iterates the plan's layers, so an empty plan would
    // vacuously pass; skipping is the same answer stated honestly.
    if let (Some(wm), Some(plan_for_validation)) = (weight_map_ref, applied_plan)
        && compiler.cpdt_mode == CpdtMode::Full
        && let Err(e) = crate::cpdt_sensitivity::validate(wm, plan_for_validation)
    {
        // KNOWN LIMIT: on the pre-plan offer (compile_train_block),
        // `applied_plan` is the pre-pass's — a plan the fingerprint
        // check has not yet accepted. A checkpoint that would
        // validate against the fresh in-place replan can therefore
        // die here on the stale one; say so, because a validation
        // error that names the wrong plan generation sends the user
        // at the wrong artifact.
        let mut err = CodegenError::new(e.to_string()).with_note(
            "checked against the WGGO plan available at this point — \
                     the pre-pass offer when one exists; if the graph changed \
                     since the pre-pass, recompile so it regenerates",
        );
        if let Some(train) = train_block {
            // The header's `model = …`, not the whole block: the
            // refusal is about that model's weights.
            err = err.with_span(crate::wggo_prepass::model_arg_span(
                train,
                compiler.interner,
            ));
        }
        return Err(err);
    }

    let input = CpdtInput {
        mode: compiler.cpdt_mode,
        model,
        cluster,
        weights: weight_map_ref,
        precision_cfg: precision_cfg.clone(),
        adamw,
        moe_shape: None,
        moe_router: None,
        moe_roofline_slack: 0.0,
        expert_cfg: ExpertConfig::default(),
        joint_cfg: JointConfig::default(),
        wggo_recommended_shard: overrides.as_ref().and_then(|o| o.min_shard_factor()),
    };

    let mut plan = cpdt_run(input);
    plan.planned_without_wggo = applied_plan.is_none();

    // Say which capacity CPDT planned in. The precision half, when one was
    // built at all, is fully valid (it never reads the applied plan); the
    // ZeRO/comm halves ran over an empty cost model and their numbers
    // describe a zero-size model. Do not claim a weight-derived precision
    // plan on configurations that build none (zero_only mode, or the
    // @cpdt(weight_aware=false) opt-out) — the review caught the first
    // wording overclaiming exactly that.
    if applied_plan.is_none() {
        if compiler.cpdt_mode == CpdtMode::Full && weights_present {
            nsl_runtime::nsl_log!(WARN, "cpdt", 
                "[cpdt] planned without a WGGO plan for this block \
                 (weights-only): optimizer-moment precision derives from \
                 the weight map; the ZeRO/comm halves saw an empty cost \
                 model."
            );
        } else {
            nsl_runtime::nsl_log!(WARN, "cpdt", 
                "[cpdt] planned without a WGGO plan for this block: the \
                 ZeRO/comm halves saw an empty cost model, and this \
                 configuration builds no per-param precision plan (mode \
                 {} / weight-aware {}).",
                compiler.cpdt_mode.as_str(),
                compiler.cpdt_weight_aware,
            );
        }
    }

    // Tier-agreement diagnostic requires a populated precision plan, which
    // cpdt::run only builds under CpdtMode::Full. Under ZeroOnly the precision
    // field is default-empty regardless of whether weights were supplied, so
    // gating only on `weights_present` would emit a meaningless 100% / 0-of-0
    // line. Tie the diagnostic to the mode that actually exercises the scorer.
    let precision_plan_built = compiler.cpdt_mode == CpdtMode::Full;

    if weights_present && precision_plan_built
        && let Some(wm) = weight_map_ref
    {
        let plan_nw = plan_map_noweights(wm, &precision_cfg);
        let (agree_layers, total_layers, agree_params, total_params) =
            compute_tier_agreement(&plan.precision, &plan_nw);
        let layer_pct = if total_layers == 0 {
            100.0
        } else {
            100.0 * agree_layers as f64 / total_layers as f64
        };
        let param_pct = if total_params == 0 {
            100.0
        } else {
            100.0 * agree_params as f64 / total_params as f64
        };
        nsl_runtime::nsl_log!(INFO, "cpdt", 
            "[cpdt] weight-aware tier agreement: {:.2}% ({}/{} layers, \
                 parameter-weighted {:.2}%)",
            layer_pct, agree_layers, total_layers, param_pct,
        );
        if param_pct < 95.0 {
            nsl_runtime::nsl_log!(WARN, "codegen", 
                "warning: weight-aware tier agreement below 95% (parameter-weighted \
                     {:.2}%). This may indicate that the calibration constants do not fit \
                     this weight distribution well. Phase 2's spectral factor + sidecar \
                     cache narrow this gap; see docs/superpowers/specs/\
                     2026-04-18-cpdt-weight-aware-phase2-stub.md.",
                param_pct
            );
        }

        if let Ok(val) = std::env::var("CPDT_CALIB_K") {
            nsl_runtime::nsl_log!(WARN, "codegen", 
                "warning: CPDT_CALIB_K={val} is set but ignored. Weights are present, \
                     so the computed gradient_magnitude_est is authoritative. If CPDT_CALIB_K \
                     is vestigial in your shell, you can unset it to silence this warning."
            );
        }
    }

    // Publish to the CLI-owned output slot (if any) so `nsl build` can
    // render the plan after compile returns without threading it through
    // every entry-point's return tuple.
    if let Some(slot) = compiler.compile_options.cpdt.plan_out.as_ref()
        && let Ok(mut guard) = slot.lock()
    {
        *guard = Some(plan.clone());
    }
    compiler.bus.publish_cpdt_plan(plan);
    Ok(())
}

pub(crate) fn invoke_wrga_if_enabled(
    compiler: &mut crate::compiler::Compiler,
    list: &crate::wengert::WengertList,
) -> Option<crate::wrga::WrgaPlan> {
    // Experimental subsystem (WRGA). Compiled in by default; a build that opts
    // out (`--no-default-features` without `experimental-wrga`) turns WRGA
    // adapter/freeze codegen into a no-op here. See STATUS.md / docs/architecture/.
    #[cfg(not(feature = "experimental-wrga"))]
    {
        let _ = (compiler, list);
        return None;
    }
    let inputs = compiler.wrga_inputs.as_ref()?;
    if inputs.wrga.is_empty() && inputs.freeze.is_empty() && inputs.adapter.is_empty() {
        return None;
    }

    let mode = inputs
        .wrga
        .first()
        .map(|c| c.mode)
        .unwrap_or(nsl_ast::block::WrgaMode::Auto);

    // Collect param names present in the Wengert list so we can synthesise
    // the "complement" of an `@freeze(include=...)` spec.  `WrgaInput` only
    // takes a "trainable" allowlist, so `include` (= "these are frozen") is
    // translated to "everything else is trainable".
    let mut param_names: Vec<String> = Vec::new();
    for op in &list.ops {
        if let crate::wengert::PrimalOp::Param(name) = &op.op {
            param_names.push(name.clone());
        }
    }

    let mut trainable_owned: Vec<String> = Vec::new();
    for f in &inputs.freeze {
        // `exclude` semantics: these patterns mark params to *keep* trainable.
        for pat in &f.exclude {
            trainable_owned.push(pat.clone());
        }
        // `include` semantics: these patterns mark params to *freeze*; the
        // complement is trainable.  `wrga_prune::glob_match` is the pattern
        // language.
        if !f.include.is_empty() {
            for name in &param_names {
                let frozen = f
                    .include
                    .iter()
                    .any(|pat| crate::wrga_prune::glob_match(pat, name));
                if !frozen {
                    trainable_owned.push(name.clone());
                }
            }
        }
    }
    if trainable_owned.is_empty() && inputs.freeze.is_empty() {
        // No freeze config at all → default to "everything trainable".
        trainable_owned.push("*".to_string());
    }
    // Deduplicate to keep the allowlist compact.
    trainable_owned.sort();
    trainable_owned.dedup();

    let manual_adapter_owned: Vec<String> = inputs
        .adapter
        .iter()
        .flat_map(|a| a.targets.iter().cloned())
        .collect();

    let hybrid_owned: Vec<String> = inputs
        .wrga
        .iter()
        .flat_map(|c| c.layers.iter().cloned())
        .collect();

    let budget_params = inputs
        .wrga
        .iter()
        .find_map(|c| c.budget)
        .unwrap_or(0)
        .max(0) as usize;

    // Borrow the owned strings into the `&'a str` slots `WrgaInput` expects.
    let trainable_patterns: Vec<&str> = trainable_owned.iter().map(|s| s.as_str()).collect();
    let manual_adapter_targets: Vec<&str> =
        manual_adapter_owned.iter().map(|s| s.as_str()).collect();
    let hybrid_layers: Vec<&str> = hybrid_owned.iter().map(|s| s.as_str()).collect();

    // WRGA paper §8.3: surface the first non-empty `target=` set on
    // `WrgaInputs::wrga[*]`. The CLI's `--wrga-target` override pipes through
    // this field via `apply_wrga_target_override` in the CLI's bridge
    // functions. Source-level `@wrga(target="...")` decorators do NOT yet
    // populate this field — both bridge functions still emit `target: None`
    // pending symbol-resolution wiring — so this `find_map` falls back to the
    // historical "rtx5070ti" default for the `nsl build` path. Existing build
    // behaviour is therefore unchanged; only the `nsl check --wrga-analyze`
    // path produces a non-None target today.
    let target_override = inputs
        .wrga
        .iter()
        .find_map(|c| c.target.as_deref().filter(|s| !s.is_empty()))
        .unwrap_or("rtx5070ti");

    let wrga_input = crate::wrga::WrgaInput {
        mode,
        trainable_patterns,
        manual_adapter_targets,
        hybrid_layers,
        wengert: list,
        loss_output: list.output,
        // M52 weights (--weights safetensors) feed WRGA's spectral rank
        // allocator directly: `run_spectral` consumes &WeightMap, and with
        // None it silently fell back to the roofline suggested_rank clamp —
        // `allocate_ranks` (the actual paper allocator) never ran in the
        // build path. Same field wggo_prune dereferences further down.
        weights: compiler.features.weight_map.as_ref(),
        target: target_override,
        budget_params,
        r_min: 2,
        r_max: 16,
        seed: 0xC0DE_FACE,
        inspect_pinned_vars: compiler.inspect_pinned_vars.clone(),
        wggo_overrides: compiler.bus.wggo_overrides(),
        // Paper §9.3 ablation flags — forwarded verbatim from `WrgaInputs`.
        // No-op for normal `nsl build` (default `WrgaAblation::default()`);
        // populated by `apply_wrga_check_overrides` in nsl-cli (the bridge
        // that handles both `--wrga-target` and `--wrga-ablate`).
        ablation: inputs.ablation,
        // Paper §8.2 user-defined custom adapter name (`@wrga(adapter=...)`).
        // First matching `WrgaDecoratorConfig` wins, falling back to None.
        custom_adapter: inputs
            .wrga
            .iter()
            .find_map(|c| c.custom_adapter.as_deref()),
    };
    let mut plan = crate::wrga::run(wrga_input);

    // B.2 Task 2b: thread the user-facing `@adapter(type=..., alpha=...)`
    // decorator config onto each matching placement, then run the adapter
    // inject pass to populate `synthesized_fields` / `init_strategies`.
    for adapter_cfg in &inputs.adapter {
        for placement in plan.placements.iter_mut() {
            let matches = adapter_cfg
                .targets
                .iter()
                .any(|pat| crate::wrga_prune::glob_match(pat, &placement.name));
            if matches {
                placement.decorator_kind = Some(adapter_cfg.kind);
                placement.alpha = adapter_cfg.alpha;
                if let Some(r) = adapter_cfg.rank
                    && r > 0
                {
                    placement.suggested_rank = r as usize;
                }
            }
        }
    }
    let inject = crate::wrga_adapter_inject::run_with_compiler(&mut plan, compiler);
    // B.2.1 Task 5.5: only clobber `adapter_sites` when this invocation
    // actually produced sites. Otherwise we'd wipe the pre-scan result
    // (which runs before user-function compilation) when a target pattern
    // like "Toy.w" doesn't match any placement name emitted by
    // `infer_sites_from_wengert` (which uses bare "w"-style names).
    // `bus.wrga_plan` is always overwritten — the train-block plan is
    // strictly more informative (real Wengert list, real placements).
    if !inject.sites.is_empty() {
        // B.3 Task 4: wire fusion decisions onto each newly-injected site.
        let mut sites = inject.sites;
        for site in sites.iter_mut() {
            for decision in &plan.fusion.decisions {
                if decision.site == site.target_param {
                    site.fusion_decision = Some(decision.target.clone());
                    break;
                }
            }
        }
        compiler.bus.publish_adapter_sites(sites);
    }
    compiler.bus.publish_wrga_plan(plan.clone());
    Some(plan)
}

/// CSHA driver bridge — mirrors `invoke_wrga_if_enabled` (Milestone C).
///
/// Hoisted out of `compile_train_block_inner` so the scheduler has a body to
/// schedule: resolves the per-model `@csha(...)` config, runs the planner
/// against `list`, and performs the pass's three bus publishes
/// (`csha_bridge`, `csha_claimed_ops`, conditionally
/// `csha_backward_claims`). Everything a `finish(&bus)` postcondition judges
/// happens INSIDE this function — `csha_bridge` declares
/// `applied_implies_published: Enforced`, and `Applied` is recorded inside
/// `csha::run`, so calling finish before the publish would refuse every
/// applying compile.
///
/// No-op when `--csha` was not passed, when the flag or the per-model
/// decorator disables it, or when the planner returns no plan.
pub(crate) fn invoke_csha_if_enabled(
    compiler: &mut crate::compiler::Compiler,
    list: &crate::wengert::WengertList,
    model_type_name: &str,
) {
    let Some(ref mode_str) = compiler.compile_options.csha.mode else {
        return;
    };
    // Sprint 2 (paper §6.2 binding fix): consult the per-model `@csha(...)`
    // config captured by the semantic checker, keyed by the model type the
    // current train block is compiling against.  We resolve effective
    // mode_str / target / disable HERE so the rest of the hook stays uniform.
    let per_model_cfg = compiler
        .compile_options
        .analysis
        .csha_configs
        .get(model_type_name)
        .cloned();
    let model_disabled = per_model_cfg
        .as_ref()
        .map(|c| c.disabled)
        .unwrap_or(false);
    // `level=` on the decorator clamps the planner's mode.
    // We prefer the decorator over `--csha` because the
    // decorator is the per-model authorial intent, not the
    // global build switch.  Mapping mirrors `CshaMode::parse`.
    let effective_mode_string: String = per_model_cfg
        .as_ref()
        .and_then(|c| {
            c.level.map(|l| match l {
                nsl_semantic::csha::CshaLevel::Boundary => {
                    "boundary".to_string()
                }
                nsl_semantic::csha::CshaLevel::Pipeline => {
                    "pipeline".to_string()
                }
                nsl_semantic::csha::CshaLevel::Block => {
                    "block".to_string()
                }
            })
        })
        .unwrap_or_else(|| mode_str.clone());
    let effective_mode_str = effective_mode_string.as_str();
    // `target=` on the decorator overrides the global GPU
    // target for CSHA planning only (does NOT affect the
    // rest of codegen — that runs on the global target).
    let effective_target: &str = per_model_cfg
        .as_ref()
        .and_then(|c| c.target.as_deref())
        .unwrap_or(compiler.compile_options.target.as_str());

    let disabled_by_decorator = model_disabled;
    let disabled_by_flag = mode_str == "off"
        || mode_str == "disable"
        || mode_str == "disabled";
    if !disabled_by_decorator && !disabled_by_flag {
        // H.1: when `@flash_attention(head_dim=N)` is on a
        // method, `compile_flash_attention_kernels` has
        // already populated `flash_attention_context.config.head_dim`
        // with the user-specified N. Without threading that
        // value into CSHA's `LayerShape`, the planner's
        // `roofline_tile_config` sees `head_dim=64` (from
        // `run_on_wengert`'s default shape), every downstream
        // `FlashAttentionConfig` carries `head_dim=64`, and
        // the Tier C backward SMEM validator rejects the
        // fused backward ("713472 bytes > 101376 byte cap").
        // The dispatcher then silently falls back to per-op
        // adjoints and the toy pretrain smoke never emits
        // `nsl_flash_attention_csha_backward`.
        let csha_shape_override = compiler
            .kernels
            .flash_attention_context
            .as_ref()
            .map(|ctx| crate::wggo_cost::LayerShape {
                batch: 1,
                seq: 1024,
                d_model: 512,
                head_dim: ctx.config.head_dim as u64,
                n_kv_heads: 4,
                dtype_bytes: 2,
            });
        if let Some(plan) = crate::csha::run_on_wengert(
            list,
            effective_target,
            effective_mode_str,
            None, // weight-aware analysis hooked up via CompileOptions.weight_file in follow-up
            csha_shape_override, // H.1: forward decorator head_dim to the planner
            8,    // default head count; weight-informed path refines this
            compiler.bus.wggo_overrides(),
        ) {
            if compiler.compile_options.csha.report {
                nsl_runtime::nsl_log!(INFO, "codegen", "{}", plan.render_report());
            } else {
                nsl_runtime::nsl_log!(INFO, "csha", "[csha] {}", plan.summary());
            }
            // Emit override-rejection diagnostics after the summary line
            // so CLI readers see summary first, per-layer details after.
            for diag in &plan.override_diagnostics {
                let reason_str = match &diag.reason {
                    crate::wggo_overrides::OverrideRejectReason::SmemBudgetExceeded {
                        actual_kb,
                        limit_kb,
                    } => {
                        format!("smem_{}kb_exceeds_{}kb", actual_kb, limit_kb)
                    }
                    other => format!("{:?}", other),
                };
                nsl_runtime::nsl_log!(INFO, "csha", 
                    "[csha] layer:{} wggo-override-rejected requested={} applied={} reason={}",
                    diag.layer_index,
                    diag.requested,
                    diag.applied,
                    reason_str
                );
            }
            // A.1: persist the bridge result so the FA call
            // site can route CSHA-active layers through the
            // CSHA-aware FFI. `csha::run` already called
            // `csha_apply::bridge`; we reconstruct it here
            // to keep the kernels / marks / configs map
            // available to downstream code.
            let mut diags = Vec::<String>::new();
            let mut bridge_out = crate::csha_apply::bridge(
                &plan,
                plan.per_layer
                    .first()
                    .map(|lp| lp.tiles.head_dim as i64)
                    .unwrap_or(64),
                &mut diags,
            );
            // Gap A: we're inside `compile_train_block`, so
            // `@train` is active — flip the save flag on
            // every per-layer CshaExtras. The forward FA
            // call site (`compile_flash_attention_call`)
            // reads this to decide between the no-saves
            // and with-saves FFI variants.
            for extras in bridge_out.extras.values_mut() {
                extras.save_activations_for_backward = true;
            }
            compiler.bus.publish_csha_bridge(bridge_out);
            for d in diags { nsl_runtime::nsl_log!(WARN, "codegen", "warning: {d}"); }
            // A.2.1d: record the Wengert OpIds CSHA has
            // claimed across all boundary chains so
            // downstream passes (A.2.2 RMSNorm prologue,
            // A.2.3 matmul projection, A.2.4 RoPE
            // epilogue) can ask `is_csha_claimed(op)`
            // before emitting a redundant launch. The list
            // argument is the id-space conversion boundary:
            // chain fields are positions in the scanned
            // list, and positions stop equaling ids the
            // moment any earlier prune deleted an op.
            compiler.bus.publish_csha_claimed_ops(
                crate::csha_apply::collect_claimed_ops(
                    &plan,
                    list,
                ),
            );
            // T7.1 / Gap D.1: build the chain-level dispatch
            // map for the AD reverse walk. Gap D.1 passes the
            // Wengert list so the dispatcher can resolve
            // per-chain VarIds (Q/K/V outputs, weights,
            // RMSNorm-out) and detect the shared SDPA op —
            // which is the correct primary claim site for
            // `EmitFused`.
            // Computed into a local FIRST so the read borrow of
            // `compiler.bus` ends before the publish below takes it
            // mutably. Both values used to be separate public
            // fields, which let the read and the write overlap;
            // they are one struct now, so the sequence has to be
            // written out. Same values, same order.
            let backward_claims = if let Some(bridge) =
                compiler.bus.csha_bridge()
            {
                // Gap I.1: pass the TRAINING config (clamped, no
                // fusion flags) so the dispatcher's backward SMEM
                // validator sees the same geometry the real
                // launch will fire. Falls back to plan-level
                // config when `csha_training_config` is absent
                // (inference-only builds).
                let training_config = compiler
                    .kernels
                    .flash_attention_context
                    .as_ref()
                    .and_then(|c| c.csha_training_config.as_ref());
                let (op_to_chain, chain_marks) =
                    crate::csha_apply::collect_chain_dispatch_map_with_wengert(
                        &plan,
                        bridge,
                        Some(list),
                        training_config,
                    );
                (!chain_marks.is_empty()).then_some(
                    crate::source_ad::CshaBackwardClaims {
                        op_to_chain,
                        chain_marks,
                    },
                )
            } else {
                None
            };
            if let Some(claims) = backward_claims {
                compiler.bus.publish_csha_backward_claims(claims);
            }
        }
    }
}

/// Dev Tools Phase 4 Task 4: extract a layer index from a parameter path.
/// Finds the last numeric segment (e.g. "blocks.3.attn.wq" -> 3).  Returns
/// `u32::MAX` when no numeric segment is present.
pub(crate) fn parse_layer_idx_for_health(path: &str) -> u32 {
    path.split('.')
        .rev()
        .find_map(|seg| seg.parse::<u32>().ok())
        .unwrap_or(u32::MAX)
}

pub(crate) fn classify_source_ad_param_name(
    param_name: &str,
    tensor_param_paths: &std::collections::HashSet<String>,
) -> SourceAdParamDiagnosticKind {
    if tensor_param_paths.contains(param_name) {
        if is_trainable_param_leaf_name(param_name) {
            SourceAdParamDiagnosticKind::Trainable
        } else {
            SourceAdParamDiagnosticKind::IgnoredConfig
        }
    } else {
        SourceAdParamDiagnosticKind::IgnoredNonTensor
    }
}

impl Compiler<'_> {
    /// Compile one statement. An error raised anywhere beneath it — by this
    /// dispatcher, an expression, or a helper that never sees a span — leaves
    /// here pointing at the innermost statement or expression that was being
    /// compiled (`CodegenError::with_span_if_unset`: the first node on the
    /// way out to attach a span wins, so nested statements keep theirs).
    pub fn compile_stmt(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        stmt: &Stmt,
    ) -> Result<(), CodegenError> {
        self.compile_stmt_dispatch(builder, state, stmt)
            .map_err(|e| e.with_span_if_unset(stmt.span))
    }

    fn compile_stmt_dispatch(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        stmt: &Stmt,
    ) -> Result<(), CodegenError> {
        if let Some(block) = state.current_block
            && is_block_filled(builder, block)
        {
            return Ok(());
        }

        // Clear any stale lambda capture count from a previous statement
        // (only VarDecl should consume this; if it leaks past a statement boundary it's a bug)
        self.registry.last_lambda_capture_count = None;

        // ELTLS v2a: dispatch-fresh membership is a PER-STATEMENT fact
        // (see TensorCleanupState::dispatch_fresh) — a value that survives
        // into the next statement is variable-bound, and SSA use_var can
        // hand a later dispatch that very Value; tracking it there would
        // free the live binding.
        state.cleanup.dispatch_fresh.clear();

        match &stmt.kind {
            StmtKind::VarDecl { pattern, value, .. } => {
                match &pattern.kind {
                    PatternKind::Ident(sym) => {
                        let sym = *sym;
                        if let Some(expr) = value
                            && self.expr_is_borrowed_batch_handle(state, expr)
                        {
                            return Err(CodegenError::new(format!(
                                "cannot bind DataLoader batch handle '{}' directly; access batch fields instead",
                                self.resolve_sym(sym)
                            )));
                        }
                        let init_val = if let Some(expr) = value {
                            // M36: Check if this variable is slab-planned for zero-alloc
                            let slab_result =
                                self.try_compile_slab_tensor(builder, state, &sym, expr);
                            match slab_result {
                                Ok(Some(val)) => val,                          // Slab allocation succeeded
                                _ => self.compile_expr(builder, state, expr)?, // Normal path
                            }
                        } else {
                            builder.ins().iconst(cl_types::I64, 0)
                        };

                        let cl_type = if let Some(expr) = value {
                            let nsl_ty = self.node_type(expr.id).clone();
                            // M56 Task 18: when the semantic pass returns Error/Unknown
                            // (e.g. for vars in @pipeline_agent bodies where agent
                            // bindings are synthesised at codegen time), fall back to
                            // the actual Cranelift type of the compiled Value so that
                            // variable declaration never mismatches the init_val type.
                            if matches!(
                                nsl_ty,
                                nsl_semantic::types::Type::Unknown
                                    | nsl_semantic::types::Type::Error
                            ) {
                                builder.func.dfg.value_type(init_val)
                            } else {
                                nsl_type_to_cl(&nsl_ty)
                            }
                        } else {
                            cl_types::I64
                        };

                        if let Some((var, _)) = state.variables.get(&sym).copied() {
                            if state.dataloader_symbols.contains(&sym) {
                                return Err(CodegenError::new(format!(
                                    "redeclaring DataLoader handle '{}' is unsupported; use a fresh symbol instead",
                                    self.resolve_sym(sym)
                                )));
                            }
                            // ELTLS §6.5: unified slot clear before rebind.
                            self.eltls_clear_old_slot(builder, state, sym);
                            // Dict twin of the slot clear: scan-admitted
                            // loop-body dict locals free the previous
                            // iteration's dict (values + structure) here.
                            // The tensor clear above no-ops on them (their
                            // Dict type fails its filter and they are not
                            // in eltls_loop_predeclared); iteration one
                            // frees the predeclared 0, a no-op. Keyed off
                            // dict_loop_predeclared — slots the predeclare
                            // actually created — NOT the plan set: a decl
                            // shadowing a function parameter is in the
                            // plan but keeps the caller's slot (review
                            // HIGH-1 on d114b5d7, reproduced corruption).
                            if state.dict_loop_predeclared.contains(&sym) {
                                let old_val = builder.use_var(var);
                                let _ = self.compile_call_by_name(
                                    builder,
                                    "nsl_dict_free_tensor_values",
                                    &[old_val],
                                );
                            }
                            builder.def_var(var, init_val);
                        } else {
                            let var = builder.declare_var(cl_type);
                            builder.def_var(var, init_val);
                            state.variables.insert(sym, (var, cl_type));
                        }

                        // Record semantic type for step-variable cleanup
                        if let Some(expr) = value {
                            state
                                .variable_types
                                .insert(sym, self.node_type(expr.id).clone());
                            // Function-valued binding (lambda, fn alias,
                            // call returning a fn): register scoped
                            // liveness for the shadow-dispatch guard.
                            if matches!(
                                self.node_type(expr.id),
                                nsl_semantic::types::Type::Function { .. }
                            ) {
                                state.register_fn_binding(sym);
                            }
                            if self.expr_is_dataloader_handle(state, expr) {
                                state.dataloader_symbols.insert(sym);
                            } else {
                                state.dataloader_symbols.remove(&sym);
                            }
                        } else {
                            state.dataloader_symbols.remove(&sym);
                        }
                        self.update_non_owning_binding(state, sym, value.as_ref());

                        // M50: Track sparse tensor variables for end-to-end dispatch.
                        // Check if the RHS is a call to a sparse function or has Type::Sparse.
                        if let Some(expr) = value {
                            let is_sparse_type = matches!(
                                self.node_type(expr.id),
                                nsl_semantic::types::Type::Sparse { .. }
                            );
                            let is_sparse_call = if let ExprKind::Call { callee, .. } = &expr.kind {
                                if let ExprKind::Ident(fn_sym) = &callee.kind {
                                    let fn_name = self.resolve_sym(*fn_sym);
                                    fn_name.contains("sparse") || fn_name == "from_dense"
                                } else {
                                    false
                                }
                            } else {
                                false
                            };
                            if is_sparse_type || is_sparse_call {
                                state.ownership.sparse_vars.insert(sym);
                            }
                        }

                        // Free intermediate tensor temporaries (keep init_val which is now owned by the variable)
                        self.free_tensor_temporaries(builder, state, Some(init_val));
                        // M38b: Free linear tensors consumed during this let-binding's RHS
                        self.free_linear_consumes(builder, state, Some(init_val));

                        // If the value was a closure lambda, record capture count for indirect call dispatch
                        if let Some(count) = self.registry.last_lambda_capture_count.take() {
                            state.set_closure_info(sym, Some(count));
                        } else {
                            // A rebind to anything that is NOT a capturing
                            // lambda must clear the stale entry, or the call
                            // site reads the new bare function pointer as a
                            // closure struct (probed: silent death). The
                            // scoped setter records an undo entry so a
                            // block-local rebind cannot outlive its scope
                            // (review HIGH on cb1bd16f: a dead if-arm's
                            // rebind deleted the outer closure's entry).
                            state.set_closure_info(sym, None);
                        }
                    }
                    PatternKind::Tuple(sub_patterns) | PatternKind::List(sub_patterns) => {
                        let tuple_val = if let Some(expr) = value {
                            self.compile_expr(builder, state, expr)?
                        } else {
                            return Err(CodegenError::new(
                                "tuple/list destructuring requires a value",
                            ));
                        };
                        let tuple_ty = value.as_ref().map(|expr| self.node_type(expr.id).clone());

                        self.compile_destructure_patterns(
                            builder,
                            state,
                            sub_patterns,
                            tuple_val,
                            tuple_ty.as_ref(),
                        )?;
                        // Review F2 on 67b9ba13: destructuring arms had no
                        // statement-end drain — a sub-expression temp (the
                        // inner `t * 2.0` of a tuple element `t * 2.0 + 1.0`;
                        // the element itself transfers into the tuple)
                        // straddled into the train step loop and was freed
                        // once per step. The destructured value is kept
                        // defensively: if an indeterminate-typed RHS ever
                        // lands it in the list, freeing a list pointer via
                        // nsl_tensor_free would abort on the magic probe.
                        self.free_tensor_temporaries(builder, state, Some(tuple_val));
                        self.free_linear_consumes(builder, state, Some(tuple_val));
                    }
                    PatternKind::Struct { fields, .. } => {
                        // Top-level struct destructuring: let { x, y } = expr
                        let struct_val = if let Some(expr) = value {
                            self.compile_expr(builder, state, expr)?
                        } else {
                            return Err(CodegenError::new("struct destructuring requires a value"));
                        };
                        let struct_ty = value.as_ref().map(|expr| self.node_type(expr.id).clone());
                        for field in fields {
                            let field_name = self.resolve_sym(field.name).to_string();
                            if !self.string_pool.contains_key(field_name.as_str()) {
                                self.intern_string(&field_name)?;
                            }
                            let key_str = self.compile_string_literal(builder, &field_name)?;
                            let field_ty =
                                self.destructure_field_type(struct_ty.as_ref(), field.name);
                            let mut field_val = self.compile_call_by_name(
                                builder,
                                "nsl_dict_get_str",
                                &[struct_val, key_str],
                            )?;
                            if field_ty.as_ref().map(|ty| ty.is_tensor()).unwrap_or(false) {
                                field_val = self.compile_call_by_name(
                                    builder,
                                    "nsl_tensor_clone",
                                    &[field_val],
                                )?;
                            }
                            if let Some(ref pat) = field.pattern {
                                match &pat.kind {
                                    PatternKind::Ident(sym) => {
                                        let var = builder.declare_var(cl_types::I64);
                                        builder.def_var(var, field_val);
                                        state.variables.insert(*sym, (var, cl_types::I64));
                                        if let Some(field_ty) = field_ty.clone() {
                                            state.variable_types.insert(*sym, field_ty);
                                        }
                                    }
                                    PatternKind::Tuple(nested) | PatternKind::List(nested) => {
                                        self.compile_destructure_patterns(
                                            builder,
                                            state,
                                            nested,
                                            field_val,
                                            field_ty.as_ref(),
                                        )?;
                                    }
                                    PatternKind::Wildcard => {}
                                    _ => {
                                        return Err(CodegenError::new(format!(
                                            "unsupported pattern in struct field '{}'",
                                            field_name
                                        )));
                                    }
                                }
                            } else {
                                let var = builder.declare_var(cl_types::I64);
                                builder.def_var(var, field_val);
                                state.variables.insert(field.name, (var, cl_types::I64));
                                if let Some(field_ty) = field_ty {
                                    state.variable_types.insert(field.name, field_ty);
                                }
                            }
                        }
                        // Review F2 on 67b9ba13 — same statement-end drain as
                        // the tuple/list destructure arm above; the bound
                        // field values are clones (never listed) and the
                        // struct value itself is kept defensively.
                        self.free_tensor_temporaries(builder, state, Some(struct_val));
                        self.free_linear_consumes(builder, state, Some(struct_val));
                    }
                    _ => {
                        return Err(CodegenError::new(
                            "only ident, tuple, list, and struct patterns supported",
                        ))
                    }
                }
            }

            StmtKind::Assign { target, op, value } => {
                self.compile_assign(builder, state, target, *op, value)?;
            }

            StmtKind::Return(expr) => {
                if let Some(e) = expr {
                    if self.expr_is_dataloader_handle(state, e) {
                        return Err(CodegenError::new(
                            "cannot return a DataLoader handle directly; create and consume loaders within the same function",
                        ));
                    }
                    if self.expr_is_borrowed_batch_handle(state, e) {
                        return Err(CodegenError::new(
                            "cannot return a DataLoader batch handle directly; return batch fields instead",
                        ));
                    }
                    let mut val = self.compile_expr(builder, state, e)?;
                    // ELTLS §6.5: consult return-value ownership and emit the
                    // correct transfer/retain path for tensor-typed returns.
                    // Require Cranelift value to be I64 AND semantic type to be
                    // a real tensor (NOT indeterminate — BYOD dtype method
                    // returns have Unknown semantic type but are scalars).
                    // Also skip inside dtype methods entirely.
                    let ret_ty = self.node_type(e.id).clone();
                    let val_is_ptr = builder.func.dfg.value_type(val) == cl_types::I64;
                    if val_is_ptr && ret_ty.is_tensor() && !state.flags.in_dtype_method {
                        use crate::ownership_expr::Ownership;
                        // Upgrade Unknown to Owned when the return expr is a
                        // call contractually returning an owning ref (see
                        // expr_call_returns_owning_ref) — the conservative
                        // retain below would double-own it and strand one
                        // reference per call.
                        let mut own = self.get_ownership(state, val);
                        if matches!(own, Ownership::Unknown)
                            && self.expr_call_returns_owning_ref(e)
                        {
                            own = Ownership::Owned;
                        }
                        match own {
                            Ownership::Owned => {
                                self.consume_ownership(state, val);
                            }
                            Ownership::BorrowedFromVar(_) | Ownership::BorrowedWeight => {
                                let _ = self.compile_call_by_name(
                                    builder,
                                    "nsl_tensor_retain",
                                    &[val],
                                );
                            }
                            Ownership::TapeHeld => {
                                nsl_runtime::nsl_log!(WARN, "codegen", 
                                    "ELTLS warning: returning TapeHeld tensor — semantic error"
                                );
                                let _ = self.compile_call_by_name(
                                    builder,
                                    "nsl_tensor_retain",
                                    &[val],
                                );
                            }
                            Ownership::Unknown => {
                                let _ = self.compile_call_by_name(
                                    builder,
                                    "nsl_tensor_retain",
                                    &[val],
                                );
                                self.note_unknown_fallback(state, val);
                            }
                        }
                    }
                    // Free intermediate tensor temporaries before returning (keep return value)
                    self.free_tensor_temporaries(builder, state, Some(val));
                    // M38b: Free linear tensors consumed during the return expression
                    self.free_linear_consumes(builder, state, Some(val));
                    // Free non-parameter tensor locals (see emit_return_local_sweep).
                    //
                    // Tensor returns: safe because the retain above compensates
                    // for the returned value aliasing a swept local.
                    //
                    // SCALAR returns: also swept, and previously missed. Note
                    // what the return type does and does not buy us. A
                    // `-> f64` / `-> int` / `-> bool` result is a value, not a
                    // handle, so it cannot alias a swept local and needs no
                    // retain to protect it. The precondition the SWEEP itself
                    // needs is different and orthogonal: no swept local may
                    // have ESCAPED. That is carried by `non_owning_symbols`
                    // (member reads, borrowed batch handles and — since the
                    // 2026-07-24 review — non-Dict subscripts are all marked
                    // non-owning and skipped), exactly as it already is for
                    // the `-> void` and fall-through return paths this merely
                    // brings into line. Skipping the sweep here stranded one
                    // tensor per call for the extremely common
                    // `fn loss(...) -> f64: let d = a - b; return sum(d*d).item()`
                    // shape (measured: live_blocks 5 -> 11 over 3 -> 9 calls,
                    // while the identical `-> void` twin stayed flat at 2).
                    //
                    // Aggregate returns (list/dict/tuple/str/model) stay
                    // excluded: they can alias a local without a retain.
                    let ret_is_scalar = matches!(
                        ret_ty,
                        Type::Int
                            | Type::Float
                            | Type::Bool
                            | Type::F32
                            | Type::F64
                            | Type::Int8
                            | Type::Int16
                            | Type::Int32
                            | Type::Int64
                            | Type::Uint8
                    );
                    if (ret_is_scalar || (val_is_ptr && ret_ty.is_tensor()))
                        && !state.flags.in_dtype_method
                        && !state.flags.in_tape_region
                    {
                        self.emit_return_local_sweep(builder, state);
                    }
                    self.cleanup_active_loop_batches(builder, state);
                    // Stop and free any DataLoaders created in this scope
                    self.teardown_dataloaders(builder, state);
                    // @no_grad: resume tape before explicit return
                    if state.flags.is_no_grad {
                        self.compile_call_by_name(builder, "nsl_tape_resume", &[])?;
                    }
                    // In element-wise unpack methods, bitcast f64→i64 for the return value
                    if state.flags.dtype_unpack_ret_bitcast {
                        let vt = builder.func.dfg.value_type(val);
                        if vt == cranelift_codegen::ir::types::F64 {
                            val = builder.ins().bitcast(
                                cranelift_codegen::ir::types::I64,
                                cranelift_codegen::ir::MemFlagsData::new(),
                                val,
                            );
                        }
                    }
                    builder.ins().return_(&[val]);
                } else {
                    // Bare `return`: same local sweep as the implicit path.
                    if !state.flags.in_dtype_method && !state.flags.in_tape_region {
                        self.emit_return_local_sweep(builder, state);
                    }
                    self.cleanup_active_loop_batches(builder, state);
                    // Stop and free any DataLoaders created in this scope
                    self.teardown_dataloaders(builder, state);
                    // @no_grad: resume tape before explicit return
                    if state.flags.is_no_grad {
                        self.compile_call_by_name(builder, "nsl_tape_resume", &[])?;
                    }
                    builder.ins().return_(&[]);
                }
            }

            StmtKind::Expr(expr) => {
                // compile_nested_expr so a discarded owning result (bare
                // `user_fn(x)`, `y.sum()`) registers as a temporary and the
                // sweep below frees it.
                let _ = self.compile_nested_expr(builder, state, expr)?;
                // Free all tensor temporaries from this expression (none are kept)
                self.free_tensor_temporaries(builder, state, None);
                // M38b: Free linear tensors consumed during this expression statement
                self.free_linear_consumes(builder, state, None);
            }

            StmtKind::If {
                condition,
                then_block,
                elif_clauses,
                else_block,
            } => {
                self.compile_if_stmt(
                    builder,
                    state,
                    condition,
                    then_block,
                    elif_clauses,
                    else_block,
                )?;
            }

            StmtKind::While { condition, body } => {
                self.compile_while(builder, state, condition, body)?;
            }

            StmtKind::For {
                pattern,
                iterable,
                body,
            } => {
                self.compile_for(builder, state, pattern, iterable, body)?;
            }

            StmtKind::Match { subject, arms } => {
                self.compile_match(builder, state, subject, arms)?;
            }

            StmtKind::Break => {
                let exit = state
                    .loop_stack
                    .last()
                    .map(|lc| lc.exit_block)
                    .ok_or_else(|| CodegenError::new("break outside loop"))?;
                // Free tensor temporaries from current loop iteration before jumping out
                self.emit_loop_scope_cleanup(builder, state);
                builder.ins().jump(exit, &[]);
            }

            StmtKind::Continue => {
                let cont = state
                    .loop_stack
                    .last()
                    .map(|lc| lc.continue_block)
                    .ok_or_else(|| CodegenError::new("continue outside loop"))?;
                // Free tensor temporaries from current loop iteration before restarting
                self.emit_loop_scope_cleanup(builder, state);
                builder.ins().jump(cont, &[]);
            }

            StmtKind::FnDef(fn_def) => {
                // Nested function definition: declare, compile, and bind name
                let base_name = self.resolve_sym(fn_def.name).to_string();
                let unique_name = format!("__nsl_nested_{}_{}", base_name, self.next_func_index());
                let sig = self.build_fn_signature(fn_def);
                let func_id = self
                    .module
                    .declare_function(&unique_name, cranelift_module::Linkage::Local, &sig)
                    .map_err(|e| {
                        CodegenError::new(format!("failed to declare nested fn '{base_name}': {e}"))
                    })?;
                // Temporarily insert under base_name for compile_fn_def lookup, then restore
                let prev_entry = self.registry.functions.remove(&base_name);
                self.registry
                    .functions
                    .insert(base_name.clone(), (func_id, sig.clone()));

                // Compile the nested function body. closure_info now lives
                // on FuncState (per-function), so the nested body's own
                // closure metadata cannot touch this function's — the
                // earlier compiler-global map needed a snapshot here
                // (review HIGH on 44c011c1).
                self.compile_fn_def(fn_def)?;

                // Remove temp entry and restore any previous function with the same name
                self.registry.functions.remove(&base_name);
                if let Some(prev) = prev_entry {
                    self.registry.functions.insert(base_name, prev);
                }

                // Bind function name as a variable holding the function pointer
                let func_ref = self.module.declare_func_in_func(func_id, builder.func);
                let addr = builder
                    .ins()
                    .func_addr(crate::types::pointer_type(), func_ref);
                let var = builder.declare_var(cl_types::I64);
                builder.def_var(var, addr);
                state.variables.insert(fn_def.name, (var, cl_types::I64));
                // Scoped liveness for the shadow-dispatch guard — the
                // flat `variables` entry above never unbinds, but the
                // checker scopes this fn to the enclosing block.
                state.register_fn_binding(fn_def.name);
            }

            StmtKind::GradBlock(grad) => {
                self.compile_grad_block(builder, state, grad)?;
            }

            StmtKind::TrainBlock(train) => {
                // CFTP v10 (item 3): thread the enclosing `Stmt.id` so
                // `compile_train_block` can look up its `@fused_lm_ce`
                // config by AST NodeId instead of hitting
                // `fused_ce_configs.first()`.
                self.compile_train_block(builder, state, train, stmt.id)?;
            }

            StmtKind::DistillBlock(distill) => {
                // CPKD: distillation training loop with a structurally
                // frozen teacher (I-11); delegates into the train-block
                // lowering with an `active_distill_context` installed.
                self.compile_distill_block(builder, state, distill, stmt.id)?;
            }

            StmtKind::StructDef(_)
            | StmtKind::ModelDef(_)
            | StmtKind::EnumDef(_)
            | StmtKind::TraitDef(_)
            | StmtKind::Import(_)
            | StmtKind::FromImport(_)
            | StmtKind::DatasetDef(_)
            | StmtKind::TokenizerDef(_)
            // M56 Task 17: agent declarations are compiled by the dedicated
            // collect_agents / declare_agent_methods / compile_agent_methods
            // passes in entry_points.rs — not inline in stmt compilation.
            | StmtKind::AgentDef(_) => {}

            StmtKind::DatatypeDef(_) => {
                // M23: custom datatype codegen — implemented in Task 9
            }

            StmtKind::ServeBlock(serve) => {
                self.compile_serve_block(builder, state, serve)?;
            }

            StmtKind::KernelDef(_) => {
                // Kernels are compiled in the compile_kernels pass (before functions).
            }

            StmtKind::QuantBlock(quant) => {
                self.compile_quant_block(builder, state, quant)?;
            }

            StmtKind::WhileLet {
                pattern,
                expr,
                body,
            } => {
                self.compile_while_let(builder, state, pattern, expr, body)?;
            }

            StmtKind::Decorated { decorators, stmt } => {
                // Module-scoped decorator configs that apply regardless of
                // the inner stmt kind (i.e. not FnDef-specific). `@cpdt`
                // wraps a TrainBlock but the `weight_aware` kwarg is global
                // compiler state: nsl-semantic enforces exactly-one-@cpdt-
                // per-program so the single-writer semantics are safe.
                // See docs/superpowers/specs/2026-04-20-cpdt-weight-aware-opt-out-design.md.
                for d in decorators {
                    if d.name.len() == 1 && self.resolve_sym(d.name[0]) == "cpdt"
                        && let Some(args) = &d.args
                    {
                        for arg in args {
                            if let Some(name_sym) = arg.name
                                && self.resolve_sym(name_sym) == "weight_aware"
                                && let nsl_ast::expr::ExprKind::BoolLiteral(b) =
                                    arg.value.kind
                            {
                                self.cpdt_weight_aware = b;
                            }
                        }
                    }

                    // CFIE Tier-A wiring (audit gap G4): capture
                    // `@cfie(mode=..., target=...)` on a serve block so
                    // `compile_serve_block` consumes it instead of the
                    // config being validated-then-dropped.
                    if d.name.len() == 1
                        && self.resolve_sym(d.name[0]) == "cfie"
                        && matches!(stmt.kind, StmtKind::ServeBlock(_))
                    {
                        // A bare `@cfie` means "enable, full mode".
                        self.cfie_decorator_mode = Some(crate::cfie::CfieMode::Full);
                        if let Some(args) = &d.args {
                            for arg in args {
                                let Some(name_sym) = arg.name else { continue };
                                let aname = self.resolve_sym(name_sym).to_string();
                                match (aname.as_str(), &arg.value.kind) {
                                    ("mode", nsl_ast::expr::ExprKind::Ident(sym)) => {
                                        let m = self.resolve_sym(*sym).to_string();
                                        self.cfie_decorator_mode =
                                            crate::cfie::CfieMode::parse(&m);
                                    }
                                    ("mode", nsl_ast::expr::ExprKind::StringLiteral(s)) => {
                                        self.cfie_decorator_mode =
                                            crate::cfie::CfieMode::parse(s);
                                    }
                                    ("target", nsl_ast::expr::ExprKind::Ident(sym)) => {
                                        self.cfie_decorator_target =
                                            Some(self.resolve_sym(*sym).to_string());
                                    }
                                    ("target", nsl_ast::expr::ExprKind::StringLiteral(s)) => {
                                        self.cfie_decorator_target = Some(s.clone());
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }

                    // Milestone A: capture `@fase(...)` on a train block so
                    // the FASE planner consumes it instead of the config
                    // being validated-then-dropped (the checker's arm at
                    // nsl-semantic/checker/stmt.rs validated and discarded
                    // it; same defect class as the @cfie gap above).
                    if d.name.len() == 1
                        && self.resolve_sym(d.name[0]) == "fase"
                        && matches!(stmt.kind, StmtKind::TrainBlock(_))
                    {
                        let interner = self.interner;
                        let resolve = |s: nsl_ast::Symbol| -> String {
                            interner.resolve(s.0).unwrap_or("").to_string()
                        };
                        let mut diags = Vec::new();
                        if let Some(cfg) = nsl_semantic::cftp::validate_fase_decorator(
                            d, &resolve, &mut diags,
                        ) {
                            // Invalid configs never get here: nsl-semantic
                            // already failed the compile during analysis.
                            if diags.is_empty() {
                                self.fase_decorator = Some(cfg);
                            }
                        }
                    }
                }

                // Check for @no_grad and @fuse on nested function definitions
                if let StmtKind::FnDef(fn_def) = &stmt.kind {
                    for d in decorators {
                        if d.name.len() == 1 {
                            let dname = self.resolve_sym(d.name[0]);
                            if dname == "no_grad" {
                                let fname = self.resolve_sym(fn_def.name).to_string();
                                self.registry.no_grad_fns.insert(fname);
                            } else if dname == "fp8_compute" {
                                let fname = self.resolve_sym(fn_def.name).to_string();
                                self.features.fp8_compute_fns.insert(fname);
                            } else if dname == "fuse" {
                                self.validate_fuse_body(fn_def)?;
                                // Extract the op chain from the function body's return expression
                                // and register it for fused kernel launch at call sites.
                                let fname = self.resolve_sym(fn_def.name).to_string();
                                let num_params = fn_def.params.len();
                                if let Some(ret_expr) =
                                    fn_def.body.stmts.iter().rev().find_map(|s| match &s.kind {
                                        StmtKind::Return(Some(e)) => Some(e),
                                        StmtKind::Expr(e) => Some(e),
                                        _ => None,
                                    })
                                {
                                    let interner = self.interner;
                                    let resolve = |sym: nsl_ast::Symbol| -> Option<String> {
                                        interner.resolve(sym.0).map(|s| s.to_string())
                                    };
                                    if let Some((ops, _inputs)) =
                                        crate::fusion::analyze_fusible_chain(ret_expr, &resolve)
                                        && ops.len() >= 2
                                    {
                                        self.fusion
                                            .fused_fns
                                            .insert(fname.clone(), (ops, num_params));
                                    }
                                }
                                // Still compile the function normally as fallback (CPU or when
                                // fusion is disabled). The fused path is selected at call site.
                            } else if dname == "grammar" {
                                // M44: @grammar decorator on nested function
                                let fname = self.resolve_sym(fn_def.name).to_string();
                                let mut start_rule = String::new();
                                let mut grammar_source = String::new();
                                if let Some(ref dargs) = d.args {
                                    for arg in dargs {
                                        if let Some(name_sym) = arg.name {
                                            let arg_name = self.resolve_sym(name_sym).to_string();
                                            if arg_name == "start_rule"
                                                && let nsl_ast::expr::ExprKind::StringLiteral(s) =
                                                    &arg.value.kind
                                            {
                                                start_rule = s.clone();
                                            }
                                        } else if let nsl_ast::expr::ExprKind::StringLiteral(s) =
                                            &arg.value.kind
                                        {
                                            grammar_source = s.clone();
                                        }
                                    }
                                }
                                self.features.grammar_configs.insert(
                                    fname,
                                    crate::compiler::GrammarInfo {
                                        start_rule,
                                        grammar_source,
                                    },
                                );
                            }
                        }
                    }
                }
                self.compile_stmt(builder, state, stmt)?;

                // Phase 5 Task 7: after the inner VarDecl has bound the
                // target, emit @inspect hooks.  Only active when
                // `compile_options.dev_tools.inspect_enabled` is true and the stmt
                // is a `let x = ...`.
                if self.compile_options.dev_tools.inspect_enabled
                    && let StmtKind::VarDecl { pattern, .. } = &stmt.kind
                    && let PatternKind::Ident(target_sym) = &pattern.kind
                {
                    for d in decorators {
                        if d.name.len() == 1
                            && self.resolve_sym(d.name[0]) == "inspect"
                        {
                            self.emit_inspect_hook(builder, state, d, *target_sym)?;
                        }
                    }
                }
            }

            _ => {
                return Err(CodegenError::new(
                    "unsupported statement in M3 codegen".to_string(),
                ));
            }
        }
        Ok(())
    }

    /// ELTLS Task 16.1: pre-scan a loop body block for top-level VarDecl
    /// statements that have a simple Ident target, returning the list of
    /// symbols. These symbols are candidates for pre-declaration in the
    /// pre-loop scope so that their second-and-later rebinds are detected
    /// as reassignments (firing eltls_clear_old_slot and freeing the
    /// previous iteration's value).
    ///
    /// Conservative: does NOT filter by type here — nsl_tensor_free_if_valid
    /// handles non-tensor values as runtime no-ops. Does NOT descend into
    /// nested blocks (those are separate scopes and won't share state).
    /// Skips Decorated VarDecls by unwrapping one level of Decorated.
    pub(crate) fn eltls_collect_loop_let_idents(
        &self,
        stmts: &[nsl_ast::stmt::Stmt],
    ) -> Vec<nsl_ast::Symbol> {
        let mut out = Vec::new();
        for stmt in stmts {
            let inner = match &stmt.kind {
                nsl_ast::stmt::StmtKind::Decorated { stmt, .. } => &stmt.kind,
                other => other,
            };
            if let nsl_ast::stmt::StmtKind::VarDecl { pattern, value, .. } = inner {
                // Only top-level simple Ident targets
                if let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind {
                    // Skip declarations with no initializer — nothing useful
                    // to free, and they wouldn't trigger eltls anyway.
                    if value.is_some() {
                        out.push(*sym);
                    }
                }
            }
        }
        out
    }

    /// ELTLS Task 16.1: pre-declare the collected loop-body let-ident
    /// symbols in the current (pre-loop) scope. Each slot is initialized
    /// to an i64 zero, which nsl_tensor_free_if_valid treats as a no-op.
    /// Also records the symbol in state.eltls_loop_predeclared so that
    /// eltls_clear_old_slot unlocks the slot-free path without requiring
    /// a variable_types entry (which is only recorded once the VarDecl
    /// has been compiled).
    ///
    /// Skips symbols already present in state.variables (parameter,
    /// outer-scope let, etc.) — those are not loop-local rebinds and
    /// should not be touched here.
    /// Returns the symbols ACTUALLY inserted — callers must remove exactly
    /// these from state.eltls_loop_predeclared after the body compiles.
    /// Returning skipped (already-declared) syms would let an inner
    /// same-name loop's removal strip a sym the OUTER loop armed.
    pub(crate) fn eltls_predeclare_loop_lets(
        &self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        syms: &[nsl_ast::Symbol],
    ) -> Vec<nsl_ast::Symbol> {
        let mut inserted = Vec::new();
        if syms.is_empty() {
            return inserted;
        }
        let zero = builder.ins().iconst(cl_types::I64, 0);
        for &sym in syms {
            if state.variables.contains_key(&sym) {
                continue;
            }
            if state.param_symbols.contains(&sym) {
                continue;
            }
            let var = builder.declare_var(cl_types::I64);
            builder.def_var(var, zero);
            state.variables.insert(sym, (var, cl_types::I64));
            state.eltls_loop_predeclared.insert(sym);
            inserted.push(sym);
        }
        inserted
    }

    /// Collect + ownership-vet + predeclare loop-body let symbols in one
    /// step. This is the only entry point loop lowerings should use.
    ///
    /// Predeclaring a symbol activates a loop-top free of the loop-carried
    /// value at its first rebind (eltls_clear_old_slot). That is only sound
    /// when EVERY binding of the symbol anywhere in the loop body yields an
    /// OWNING reference: ident copies and member reads hand out the
    /// referent's own pointer with no retain, and freeing a loop-carried
    /// borrow frees the referent itself (a model weight, another local's
    /// tensor). Symbols with any unowned binding are left out — they keep
    /// the old declare-in-body behavior (worst case a status-quo strand,
    /// never a double-free).
    ///
    /// Returns the predeclared symbols; callers must remove them from
    /// state.eltls_loop_predeclared after compiling the body.
    pub(crate) fn eltls_predeclare_loop_lets_checked(
        &self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        body: &nsl_ast::stmt::Block,
    ) -> Vec<nsl_ast::Symbol> {
        let syms: Vec<_> = self
            .eltls_collect_loop_let_idents(&body.stmts)
            .into_iter()
            .filter(|s| self.sym_bindings_all_owning_in_block(body, *s))
            .collect();
        let inserted = self.eltls_predeclare_loop_lets(builder, state, &syms);

        // Dict twin (dict_lifetime.rs): zero-predeclare scan-admitted
        // loop-body dict locals whose decl sits at THIS body's direct
        // level, so the rebind clear and the return sweep always see a
        // defined slot (0 on iteration one / when the loop never runs —
        // free_dict_impl no-ops on 0). Deliberately NOT recorded in
        // eltls_loop_predeclared: that set arms the TENSOR clear, whose
        // free_if_valid must keep no-oping on dict handles; the dict
        // rebind clear keys off state.dict_loop_predeclared, written
        // below only when this pass creates the slot. The scan only
        // admits decls in top-level loops, so this emission point
        // dominates every later return.
        let dict_syms: Vec<_> = body
            .stmts
            .iter()
            .filter_map(|s| {
                let nsl_ast::stmt::StmtKind::VarDecl { pattern, .. } = &s.kind else {
                    return None;
                };
                let nsl_ast::pattern::PatternKind::Ident(sym) = &pattern.kind else {
                    return None;
                };
                (state.dict_loop_rebind.contains(sym)
                    && !state.variables.contains_key(sym)
                    && !state.param_symbols.contains(sym))
                .then_some(*sym)
            })
            .collect();
        if !dict_syms.is_empty() {
            let zero = builder.ins().iconst(cl_types::I64, 0);
            for sym in dict_syms {
                let var = builder.declare_var(cl_types::I64);
                builder.def_var(var, zero);
                state.variables.insert(sym, (var, cl_types::I64));
                // The rebind clear keys off this set — ONLY slots this
                // predeclare created may be dict-freed at rebind. A
                // plan sym skipped here (e.g. it shadows a function
                // parameter, whose slot already exists and holds the
                // caller's value) must never reach the clear (review
                // HIGH-1 on d114b5d7).
                state.dict_loop_predeclared.insert(sym);
            }
        }
        inserted
    }

    /// ELTLS (spec §6.5): clear a tensor-typed variable's old value before
    /// reassignment. Emits nsl_tensor_free on the old pointer and purges all
    /// tracking queues (new AND legacy). Skipped for initial let bindings
    /// (symbol not yet in state.variables), parameters, non-owning symbols,
    /// borrowed batch handles, and non-tensor variables.
    ///
    /// Deliberately does NOT touch tape_held — if the old value had an active
    /// tape lease, the nsl_tensor_free here just decrements the variable-slot
    /// refcount and the tape's retained lease keeps the storage alive until
    /// free_tape_held_tensors runs at tape-region exit.
    pub(crate) fn eltls_clear_old_slot(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        sym: nsl_ast::Symbol,
    ) {
        // Dtype method bodies run on scalar-valued slots — do NOT emit
        // tensor frees on them, even if the slot type says indeterminate.
        if state.flags.in_dtype_method {
            return;
        }
        // Only act on reassignments: the symbol must already exist.
        let Some((var, cl_type)) = state.variables.get(&sym).copied() else {
            return; // initial let — slot uninitialized, use_var would be UB
        };
        if cl_type != cl_types::I64 {
            return; // non-tensor variable
        }
        if state.param_symbols.contains(&sym) {
            return; // parameter — caller owns it
        }
        if state.non_owning_symbols.contains(&sym) {
            return; // view or borrow alias
        }
        if state.borrowed_batch_symbols.contains(&sym) {
            return; // DataLoader batch handle — freed by loader teardown
        }
        if state.dataloader_symbols.contains(&sym) {
            return; // DataLoader handle itself
        }
        // Additional semantic filter: only emit free if the variable's type is
        // actually a tensor or indeterminate, OR the slot was pre-declared by
        // the loop-predeclare pass (which records the sym in
        // eltls_loop_predeclared). Values stored in I64 slots that are
        // integers, booleans, lists, or dicts should NOT be freed via
        // nsl_tensor_free — but nsl_tensor_free_if_valid handles them as
        // no-ops by probing the magic field, so loop-predeclared slots are
        // safe to unconditionally attempt the free on.
        let is_loop_predeclared = state.eltls_loop_predeclared.contains(&sym);
        if !is_loop_predeclared {
            match state.variable_types.get(&sym) {
                Some(ty) if ty.is_tensor() || ty.is_indeterminate() => {}
                _ => return,
            }
        }
        // Don't emit frees into a filled block.
        if let Some(block) = state.current_block
            && is_block_filled(builder, block)
        {
            return;
        }
        // Read the old value and free it via the safe variant that handles
        // null pointers, invalid magic, and non-tensor i64 values as no-ops.
        // This is required for loop-pre-declared slots that start at zero on
        // the first iteration, and is a safer default for reassignment
        // paths generally (a stale non-tensor would otherwise crash).
        let old_val = builder.use_var(var);
        let _ = self.compile_call_by_name(builder, "nsl_tensor_free_if_valid", &[old_val]);
        // Purge tracking queues so the statement/function cleanup paths don't
        // try to free this value again.
        state.cleanup.expr_ownership.remove(&old_val);
        state.cleanup.owned_temporaries.retain(|&v| v != old_val);
        state.cleanup.tensor_temporaries.retain(|&v| v != old_val);
        // DO NOT touch state.cleanup.tape_held — see doc comment above.
    }

    /// Free intermediate tensor temporaries accumulated during expression compilation.
    /// `keep` is the final result value that should NOT be freed (it's owned by a variable).
    /// All other temporaries are intermediates from compound expressions (e.g. `a + b` in `a + b + c`).
    /// Free all I64-typed non-parameter local variables (potential tensors)
    /// at a function-return point. Twin of the implicit-return sweep in
    /// func.rs — until 2026-07 only the fall-off-the-end path had it, so
    /// every `let`-bound tensor local in a function ending with an explicit
    /// `return <expr>` leaked its final reference (`mse_loss`'s `diff`, the
    /// residual-block `f`, the loop-carried `h` — the tape-mode per-step
    /// leak AND the @no_grad inference leak were this hole).
    ///
    /// Safe against the returned value aliasing a swept variable: the
    /// Return arm retains BorrowedFromVar/Unknown returns before calling
    /// this, so the variable's own free leaves the returned reference
    /// intact. `nsl_tensor_free_if_valid` skips non-tensor pointers and
    /// already-poisoned boxes, so aliased vars double-swept in sequence
    /// are no-ops on the second hit.
    pub(crate) fn emit_return_local_sweep(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
    ) {
        // Name order: this sweep is a call per variable, and its order
        // is the function's text (see `variables_in_name_order`).
        let locals: Vec<_> = self
            .variables_in_name_order(state)
            .into_iter()
            .filter(|sym| !state.param_symbols.contains(sym))
            .filter(|sym| !state.non_owning_symbols.contains(sym))
            // Semantic-type filter: only tensor (or indeterminate) locals.
            // free_if_valid's pointer probes are NOT sufficient for plain
            // integers — a large 8-aligned int (e.g. a byte count from
            // gpu_peak_bytes()) passes the null/low/alignment checks and the
            // magic probe DEREFERENCES it, segfaulting on unmapped memory.
            .filter(|sym| {
                matches!(state.variable_types.get(sym),
                         Some(ty) if ty.is_tensor() || ty.is_indeterminate())
            })
            .filter_map(|sym| {
                let (var, cl_type) = state.variables[&sym];
                if cl_type == cl_types::I64 {
                    Some(var)
                } else {
                    None
                }
            })
            .collect();
        for var in locals {
            let val = builder.use_var(var);
            let _ = self.compile_call_by_name(builder, "nsl_tensor_free_if_valid", &[val]);
        }

        // Dict-local pass (the aggregate-lifetime gap, 2026-07-28). A
        // tensor-valued dict local owns its stored tensors outright — every
        // read CLONES (compile_subscript's Dict arm), so no borrow of a
        // stored tensor exists anywhere and freeing the dict with its
        // values cannot free anything reachable. Only symbols the
        // conservative usage scan admitted are freed here (single top-level
        // call binding — so the slot dominates every return that can see it
        // in `state.variables` — subscript reads only, never
        // returned/passed/stored-into); see dict_lifetime.rs for the veto
        // rules. An unscanned body has an empty set: status quo, the dict
        // strands (leak, not crash).
        let dict_locals: Vec<_> = self
            .variables_in_name_order(state)
            .into_iter()
            .filter(|sym| state.sweepable_dict_locals.contains(sym))
            .filter(|sym| !state.param_symbols.contains(sym))
            .filter_map(|sym| {
                let (var, cl_type) = state.variables[&sym];
                if cl_type == cl_types::I64 {
                    Some(var)
                } else {
                    None
                }
            })
            .collect();
        for var in dict_locals {
            let val = builder.use_var(var);
            let _ =
                self.compile_call_by_name(builder, "nsl_dict_free_tensor_values", &[val]);
        }
    }

    pub(crate) fn free_tensor_temporaries(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        keep: Option<Value>,
    ) {
        // Drain only the temporaries registered ABOVE the innermost loop-scope
        // mark. A `mem::take` of the whole list here stole entries that a loop
        // CONDITION / for-ITERABLE registered before `temp_scope_stack` pushed
        // its mark: the first body statement drained them (emitting their
        // frees INSIDE the body = once per iteration), left `scope_start`
        // pointing past the now-empty list, and the loop-exit cleanups'
        // `[scope_start..]` slices panicked at compile time — a pre-existing
        // ICE (method-form condition temps hit it on main) made trivially
        // reachable once nested-arg tracking covered every dispatch arm
        // (review HIGH on 3b7f085f, red-proven with a 6-line while loop).
        // Below-mark entries stay listed; the loop statement's own
        // statement-end cleanup frees the final condition evaluation's temp
        // after the loop exits.
        let start = state
            .cleanup
            .temp_scope_stack
            .last()
            .copied()
            .unwrap_or(0)
            .min(state.cleanup.tensor_temporaries.len());
        let temps = state.cleanup.tensor_temporaries.split_off(start);
        // Tape ID identity: intermediates can now be safely freed during tape recording
        // because TapeOps use monotonic tape_ids as identity keys (not raw pointers).
        for temp in &temps {
            if Some(*temp) == keep {
                continue;
            }
            // Emit nsl_tensor_free(temp) — but only if block is not already filled
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[*temp]);
        }
    }

    /// Free the tensor temporaries a loop CONDITION registered during one
    /// evaluation, inside the block that evaluates it (the loop header),
    /// and drain them from `tensor_temporaries` so the loop statement's
    /// end-of-statement cleanup does not free them a second time.
    ///
    /// `base` is the list length snapshotted immediately before the
    /// condition compiled — everything at or above it was registered by
    /// this evaluation. `keep` is the condition's own result value
    /// (while-let binds it into the body; excluded defensively — a
    /// tracked keep would otherwise become freed-then-read).
    ///
    /// Emitting the frees in the HEADER is what makes this per-iteration:
    /// the header re-executes before every body entry AND before the
    /// exit branch, so the final evaluation's temps are freed exactly
    /// once too. Entries BELOW `base` (an outer statement's temps) stay
    /// listed and untouched.
    pub(crate) fn free_condition_temporaries(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        base: usize,
        keep: Value,
    ) {
        let base = base.min(state.cleanup.tensor_temporaries.len());
        let temps = state.cleanup.tensor_temporaries.split_off(base);
        for temp in &temps {
            if *temp == keep {
                continue;
            }
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[*temp]);
        }
    }

    /// ELTLS: free all TapeHeld tensors accumulated during the current tape
    /// region. Called after nsl_tape_backward runs and before the block's
    /// normal scope cleanup. See spec §7.3.
    ///
    /// Tape-held tensors were promoted by set_ownership_from_op or
    /// promote_to_tape_held when a DataRequired op touched them during
    /// forward pass. The tape holds raw pointers to their data for backward;
    /// we must not free until backward completes.
    pub(crate) fn free_tape_held_tensors(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
    ) {
        let held = std::mem::take(&mut state.cleanup.tape_held);
        for val in held {
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[val]);
            state.cleanup.expr_ownership.remove(&val);
        }
    }

    /// M38b: Free linear tensors that were consumed during the current statement.
    /// Called after `free_tensor_temporaries` at each statement boundary.
    /// `keep` is the value being assigned to a variable (should NOT be freed).
    ///
    /// Only active when `state.ownership.lowering.is_some()` — the pending list
    /// is empty otherwise so the loop is a no-op.
    pub(crate) fn free_linear_consumes(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        keep: Option<Value>,
    ) {
        if state.ownership.linear_consume_pending.is_empty() {
            return;
        }
        // Don't free inside tape-recorded regions — backward needs the data alive.
        // Tape ID identity: linear consumes can now be freed during tape recording.
        let pending = std::mem::take(&mut state.ownership.linear_consume_pending);
        for val in &pending {
            if Some(*val) == keep {
                continue;
            }
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[*val]);
        }
    }

    /// Emit nsl_tensor_free calls for all tensor temporaries accumulated since the
    /// current loop scope started. Used at break/continue points.
    /// CRITICAL: Does NOT truncate tensor_temporaries — that only happens at natural scope exit.
    fn emit_loop_scope_cleanup(&mut self, builder: &mut FunctionBuilder, state: &mut FuncState) {
        if let Some(&scope_start) = state.cleanup.temp_scope_stack.last() {
            // Clamp defensively: the statement-end drain above now preserves
            // below-mark entries, but a stale mark must degrade to a no-op,
            // never a slice panic.
            let scope_start = scope_start.min(state.cleanup.tensor_temporaries.len());
            for &temp in &state.cleanup.tensor_temporaries[scope_start..] {
                if let Some(block) = state.current_block
                    && is_block_filled(builder, block)
                {
                    break;
                }
                let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[temp]);
            }
        }
    }

    /// Emit cleanup AND truncate temporaries at natural loop exit.
    pub(crate) fn cleanup_loop_scope(&mut self, builder: &mut FunctionBuilder, state: &mut FuncState) {
        if let Some(scope_start) = state.cleanup.temp_scope_stack.pop() {
            // Clamp defensively — see emit_loop_scope_cleanup.
            let scope_start = scope_start.min(state.cleanup.tensor_temporaries.len());
            for &temp in &state.cleanup.tensor_temporaries[scope_start..] {
                if let Some(block) = state.current_block
                    && is_block_filled(builder, block)
                {
                    break;
                }
                let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[temp]);
            }
            state.cleanup.tensor_temporaries.truncate(scope_start);
        }
    }

    fn cleanup_active_loop_batches(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
    ) {
        for &batch_var in state.cleanup.active_batch_vars.iter().rev() {
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let batch_ptr = builder.use_var(batch_var);
            let _ = self.compile_call_by_name(builder, "nsl_dict_free_tensor_values", &[batch_ptr]);
        }
        for loop_ctx in state.loop_stack.iter().rev() {
            let Some(batch_var) = loop_ctx.batch_var else {
                continue;
            };
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let batch_ptr = builder.use_var(batch_var);
            let _ = self.compile_call_by_name(builder, "nsl_dict_free_tensor_values", &[batch_ptr]);
        }
    }

    /// Emit nsl_dataloader_stop + nsl_dataloader_free for all DataLoaders
    /// created in this scope. Called before function returns to prevent
    /// thread leaks and resource leaks.
    pub(crate) fn teardown_dataloaders(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
    ) {
        let loaders = std::mem::take(&mut state.cleanup.dataloader_vars);
        for dl in &loaders {
            if let Some(block) = state.current_block
                && is_block_filled(builder, block)
            {
                break;
            }
            let _ = self.compile_call_by_name(builder, "nsl_dataloader_stop", &[*dl]);
            let _ = self.compile_call_by_name(builder, "nsl_dataloader_free", &[*dl]);
        }
    }

    pub(crate) fn expr_is_dataloader_handle(&self, state: &FuncState, expr: &nsl_ast::expr::Expr) -> bool {
        match &expr.kind {
            ExprKind::Call { callee, .. } => {
                if let ExprKind::Ident(fn_sym) = &callee.kind {
                    self.resolve_sym(*fn_sym) == "DataLoader"
                } else {
                    false
                }
            }
            ExprKind::Ident(sym) => state.dataloader_symbols.contains(sym),
            ExprKind::Paren(inner) => self.expr_is_dataloader_handle(state, inner),
            ExprKind::BlockExpr(block) => block
                .stmts
                .last()
                .is_some_and(|stmt| matches!(&stmt.kind, StmtKind::Expr(expr) if self.expr_is_dataloader_handle(state, expr))),
            ExprKind::IfExpr { then_expr, else_expr, .. } => {
                self.expr_is_dataloader_handle(state, then_expr)
                    && self.expr_is_dataloader_handle(state, else_expr)
            }
            _ => false,
        }
    }

    pub(crate) fn expr_is_borrowed_batch_handle(&self, state: &FuncState, expr: &nsl_ast::expr::Expr) -> bool {
        match &expr.kind {
            ExprKind::Ident(sym) => state.borrowed_batch_symbols.contains(sym),
            ExprKind::Paren(inner) => self.expr_is_borrowed_batch_handle(state, inner),
            ExprKind::BlockExpr(block) => block
                .stmts
                .last()
                .is_some_and(|stmt| matches!(&stmt.kind, StmtKind::Expr(expr) if self.expr_is_borrowed_batch_handle(state, expr))),
            ExprKind::IfExpr { then_expr, else_expr, .. } => {
                self.expr_is_borrowed_batch_handle(state, then_expr)
                    && self.expr_is_borrowed_batch_handle(state, else_expr)
            }
            _ => false,
        }
    }

    /// Check if an iterable expression is a real DataLoader handle.
    pub(crate) fn is_dataloader_iterable(&self, state: &FuncState, iterable: &nsl_ast::expr::Expr) -> bool {
        self.expr_is_dataloader_handle(state, iterable)
    }

    /// CPKD: lower a `distill(teacher=t, student=s, epochs=N):` block.
    ///
    /// v1 strategy: distillation IS a training loop over the student, so we
    /// delegate to `compile_train_block` with (a) a synthetic `TrainBlock`
    /// carrying the distill sections and (b) an `active_distill_context`
    /// installed on the compiler.  Inside `compile_train_block_inner` the
    /// context: seeds `model_sym = student` / `epochs`, registers the
    /// teacher instance for method inlining with its fields FROZEN on the
    /// Wengert extractor (I-11: teacher fields become Input leaves → no
    /// adjoints → teacher backward structurally absent), forbids the tape
    /// fallback (F-06: a tape would record teacher ops and allocate teacher
    /// grad buffers), and resolves teacher-field Input leaves to Cranelift
    /// values via `load_source_ad_named_param`.
    fn compile_distill_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        distill: &nsl_ast::block::DistillBlock,
        distill_block_stmt_id: nsl_ast::NodeId,
    ) -> Result<(), CodegenError> {
        // Deferred compositions refuse loudly rather than degrade.
        if self.features.pipeline_config.is_some() {
            return Err(CodegenError::new(
                "distill blocks do not support pipeline-parallel training in CPKD v1 \
                 (remove the pipeline configuration or use a train block)",
            ));
        }
        if !self.features.source_ad_enabled {
            return Err(CodegenError::new(
                "distill blocks require source AD (build with --source-ad): the \
                 teacher-freeze guarantee (I-11) is enforced structurally on the \
                 Wengert list; tape AD would record teacher ops and allocate \
                 teacher gradient buffers (F-06)",
            ));
        }

        // ── Present the distill header as a train header ────────────────
        // `cpkd::distill_as_train_block` is the ONE place that says which
        // header keys travel and how, shared with `training_report` — which
        // is the point: a distill block is a training loop, and before this
        // the module-level AST scans matched `StmtKind::TrainBlock` and saw
        // nothing at all, reporting "Training blocks found: 0" for a block
        // that trains and computing `segment_masked = false` for a packed
        // corpus. Keeping the extraction here, private to codegen, is what
        // left those scans nothing to read.
        let presented = crate::cpkd::distill_as_train_block(distill, self.interner)
            .map_err(CodegenError::new)?;
        let epochs = presented.epochs;
        let teacher_sym = presented.teacher_sym.ok_or_else(|| {
            CodegenError::new("distill block requires 'teacher=<model ident>'")
        })?;
        let student_sym = presented.student_sym.ok_or_else(|| {
            CodegenError::new("distill block requires 'student=<model ident>'")
        })?;

        // ── Parse loss: section into DistillLossConfig ──────────────────
        let mut loss_cfg = crate::cpkd::DistillLossConfig::default();
        let mut loss_alpha_explicit: Option<f64> = None;
        let mut loss_temperature_explicit: Option<f64> = None;
        for entry in &distill.loss {
            let Some(name_sym) = entry.name else { continue };
            let key = self.resolve_sym(name_sym).to_string();
            let as_f64 = |e: &nsl_ast::expr::Expr| -> Option<f64> {
                match e.kind {
                    ExprKind::FloatLiteral(v) => Some(v),
                    ExprKind::IntLiteral(v) => Some(v as f64),
                    _ => None,
                }
            };
            match key.as_str() {
                "alpha" => {
                    if let Some(v) = as_f64(&entry.value) {
                        loss_cfg.alpha = v;
                        loss_alpha_explicit = Some(v);
                    }
                }
                "temperature" => {
                    if let Some(v) = as_f64(&entry.value) {
                        loss_cfg.temperature = v;
                        loss_temperature_explicit = Some(v);
                    }
                }
                "feature_weight" => {
                    if let Some(v) = as_f64(&entry.value) {
                        loss_cfg.feature_weight = v;
                    }
                }
                "feature_layers" => match &entry.value.kind {
                    ExprKind::StringLiteral(s) if s == "auto" => {
                        loss_cfg.feature_layers = crate::cpkd::FeatureLayers::Auto;
                    }
                    ExprKind::ListLiteral(items) => {
                        let mut layers = Vec::with_capacity(items.len());
                        for item in items {
                            if let ExprKind::IntLiteral(n) = item.kind {
                                layers.push(n);
                            }
                        }
                        loss_cfg.feature_layers =
                            crate::cpkd::FeatureLayers::Explicit(layers);
                    }
                    _ => {}
                },
                // attn_transfer=true was refused at the semantic layer;
                // false is the only value that reaches codegen and it is
                // the default (no state to record).
                "attn_transfer" => {}
                _ => {}
            }
        }

        // The synthetic TrainBlock built above by the shared presentation.
        // Which keys travel — and why `teacher`/`student`/`epochs` stay out
        // of the config while `grad_accumulation` goes in — is documented on
        // `cpkd::distill_as_train_block`. Every consumer that needs a distill
        // block's HEADER builds it there, this site included, so they cannot
        // drift apart. `pca_activation` is not one of them: it reads only the
        // verbatim `sections`, so it matches `StmtKind::DistillBlock` directly
        // rather than cloning a step body to answer a boolean.
        let synthetic = presented.train;

        // Per-block @fused_kl_ce dispatch (mirrors CFTP v10 item 3's
        // per-train-block @fused_lm_ce lookup by stmt id).
        let fused_kl_ce = self
            .fused_kl_ce_configs
            .iter()
            .find(|c| c.distill_block_stmt_id == distill_block_stmt_id)
            .cloned();

        let saved_context = self.active_distill_context.replace(crate::cpkd::DistillContext {
            teacher_sym,
            student_sym,
            epochs,
            loss: loss_cfg,
            fused_kl_ce,
            loss_alpha_explicit,
            loss_temperature_explicit,
        });
        let result = self.compile_train_block(builder, state, &synthetic, distill_block_stmt_id);
        self.active_distill_context = saved_context;

        // Render the Distillation Build Report (facts collected during the
        // source-AD extraction inside the inner lowering). Stderr, CFIE
        // convention for in-codegen build reports.
        if result.is_ok()
            && let Some(plan) = self.bus.take_cpkd_plan()
        {
            eprint!("{}", plan.render_report());
        }
        result
    }

    /// Emit a `zeros_like` allocation for ONE optimizer-moment buffer (m or
    /// v), honoring the offload / CPDT-precision variants. `precision_list` is
    /// the per-param dtype-code list for THIS moment when a precision plan is
    /// active (`None` = verbatim f32). Factored out of the optimizer-state
    /// init loop so the D3-v2 owner-gated (ZeRO-1 shard) path can reuse the
    /// exact same allocation logic inside its owned branch.
    pub(crate) fn emit_moment_zeros_like(
        &mut self,
        builder: &mut FunctionBuilder,
        param_i: Value,
        idx: Value,
        precision_list: Option<Value>,
        offload: bool,
    ) -> Result<Value, CodegenError> {
        let buf = if offload {
            if let Some(code_list) = precision_list {
                // P0.3 composition: HOST-resident state at the planned
                // reduced-precision dtype.
                let code = self.compile_call_by_name(builder, "nsl_list_get", &[code_list, idx])?;
                self.compile_call_by_name(
                    builder,
                    "nsl_tensor_zeros_like_host_dtype",
                    &[param_i, code],
                )?
            } else {
                // Offload-only: HOST-resident f32 state.
                self.compile_call_by_name(builder, "nsl_tensor_zeros_like_host_f32", &[param_i])?
            }
        } else if let Some(code_list) = precision_list {
            let code = self.compile_call_by_name(builder, "nsl_list_get", &[code_list, idx])?;
            self.compile_call_by_name(builder, "nsl_tensor_zeros_like_dtype", &[param_i, code])?
        } else {
            self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[param_i])?
        };
        Ok(buf)
    }

    /// D3 v2 (ZeRO-1): emit an owner-gated optimizer-moment allocation. When
    /// this rank owns `idx` (owner = idx % world_size, decided at RUNTIME by
    /// nsl_zero_owns_param — the same predicate the update gate uses) it
    /// allocates the real zeros_like buffer and records its element count
    /// (nsl_zero_note_optim_alloc → the G3 memory-shrink gate); otherwise it
    /// allocates NOTHING and yields a null (0) placeholder. Returns the value
    /// to push into the moment list — real or null — keeping the list
    /// length==num_params and global-index-addressable. Non-owned nulls are
    /// never dereferenced (m/v are read only inside the owner-gated update
    /// branch; the free loop is null-safe).
    pub(crate) fn emit_owner_gated_moment(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        param_i: Value,
        idx: Value,
        precision_list: Option<Value>,
        offload: bool,
    ) -> Result<Value, CodegenError> {
        let owns = self.compile_call_by_name(builder, "nsl_zero_owns_param", &[idx])?;
        let one = builder.ins().iconst(cl_types::I64, 1);
        let owned = builder.ins().icmp(IntCC::Equal, owns, one);
        let alloc_b = builder.create_block();
        let skip_b = builder.create_block();
        let merge_b = builder.create_block();
        builder.append_block_param(merge_b, cl_types::I64);
        builder.ins().brif(owned, alloc_b, &[], skip_b, &[]);

        // Owned: allocate the real moment buffer and record its elements.
        builder.switch_to_block(alloc_b);
        builder.seal_block(alloc_b);
        state.current_block = Some(alloc_b);
        let real = self.emit_moment_zeros_like(builder, param_i, idx, precision_list, offload)?;
        self.compile_call_by_name(builder, "nsl_zero_note_optim_alloc", &[real])?;
        builder.ins().jump(merge_b, &[BlockArg::Value(real)]);

        // Non-owner: allocate nothing — push a null (0) placeholder.
        builder.switch_to_block(skip_b);
        builder.seal_block(skip_b);
        state.current_block = Some(skip_b);
        let null = builder.ins().iconst(cl_types::I64, 0);
        builder.ins().jump(merge_b, &[BlockArg::Value(null)]);

        // Merge — both predecessors connected, safe to seal.
        builder.switch_to_block(merge_b);
        builder.seal_block(merge_b);
        state.current_block = Some(merge_b);
        Ok(builder.block_params(merge_b)[0])
    }

    /// Item C: one slot's moment allocation under the ZeRO-3 deferred fill.
    /// Shared by m and v so the two can never drift into different sharding
    /// decisions for the same parameter.
    fn emit_filled_moment(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        mode: MomentFill,
        param_i: Value,
        idx: Value,
        precision_list: Option<Value>,
        offload: bool,
    ) -> Result<Value, CodegenError> {
        Ok(match mode {
            // Elementwise: every rank holds a persistent 1/ws SLICE and
            // steps it, so there is no owner gate — the runtime sizes the
            // buffer from the carved shard and notes its own elements.
            MomentFill::Elementwise => self.compile_call_by_name(
                builder,
                "nsl_zero3_alloc_elem_moment",
                &[param_i, idx],
            )?,
            // Tensor-granular sharded: the SAME owner gate stages 1/2 ship,
            // reused verbatim — one rank allocates the full moment, the
            // rest carry the established null placeholder.
            MomentFill::OwnerGated => {
                self.emit_owner_gated_moment(builder, state, param_i, idx, precision_list, offload)?
            }
            // Replicated (tied / view-rooted / epilogue params): every rank
            // updates these from all-reduced gradients, so every rank needs
            // full m/v. Deliberately NOT noted against `optim_elems` — see
            // that counter's doc for why counting them would break the
            // `r0 + r1 == full` partition identity.
            //
            // It IS noted against the replica counter. Before that counter
            // existed this arm was invisible to every instrument in the tree:
            // the partition assertion the ZeRO-3 gates rest on stayed green
            // for an arbitrarily large replicated remainder, because `Full`
            // dropped out of BOTH sides of the identity. The epilogue set is
            // not a corner case — it is every parameter with no `blocks.N`
            // key, i.e. embedding / final norm / LM head.
            MomentFill::Full => {
                let real =
                    self.emit_moment_zeros_like(builder, param_i, idx, precision_list, offload)?;
                self.compile_call_by_name(
                    builder,
                    "nsl_zero_note_replicated_optim_alloc",
                    &[real],
                )?;
                real
            }
        })
    }

    /// Item C: fill the ZeRO-3 moment lists that section 4 left null.
    ///
    /// Emitted ONCE, inline at the WINDOW register belt — not the
    /// pre-forward belt, which is guarded by `if let Some(ws_fwd_plan)` and
    /// would make this loop silently vacuous when that plan is None (the
    /// PR #482 failure mode). It is a compile-time loop over EVERY parameter
    /// index, not over `wsplan.register_idxs`: replicated and
    /// tensor-granular params are not in the streamed register set but
    /// still need their moments.
    ///
    /// The whole fill is behind ONE runtime latch (`latch[0] == 0`), not a
    /// per-slot `state_list_1[idx] == 0` test: a non-owner's tensor-granular
    /// slot is legitimately null forever, so a per-slot guard would never
    /// latch and would re-run the owner gate for ~(ws-1)/ws of the sharded
    /// set on every optimizer step, inside the hot window region. One pass
    /// covers every slot, so one flag is equally correct and free after the
    /// first window.
    ///
    /// The alloc-surface bracket is re-opened here because setup's bracket
    /// only wraps section 4; without it the moment bytes land on whatever
    /// surface the step body left current (Activations) and
    /// `mem_accounting_gpu_gate`'s attribution is silently wrong. The pool
    /// bracket matters just as much: the step body runs on the TRANSIENT
    /// pool, whose segments are drained at end of step — moments must be
    /// Persistent or they would be handed back under the optimizer.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_deferred_moment_fill(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        modes: &[MomentFill],
        param_list: Value,
        state_list_1: Value,
        state_list_2: Value,
        num_state_buffers: usize,
        m_codes: Option<Value>,
        v_codes: Option<Value>,
        muon_v_gate: Option<Value>,
        offload: bool,
        latch: Value,
    ) -> Result<(), CodegenError> {
        if m_codes.is_some() && modes.contains(&MomentFill::Elementwise) {
            // deferral-must-refuse: the elementwise slice allocator sizes
            // from the carve and hands back θ's own dtype — it has no
            // reduced-precision arm. Today this is unreachable (CPDT and
            // muon-state-bf16 are both refused alongside --zero-elementwise),
            // so refuse loudly instead of silently ignoring the plan.
            return Err(CodegenError::new(
                "--zero-elementwise does not compose with a per-parameter \
                 moment-precision plan: the elementwise slice moment is \
                 allocated at the parameter's own dtype. Drop one",
            ));
        }

        let slot0 = builder.ins().iconst(cl_types::I64, 0);
        let done = self.compile_call_by_name(builder, "nsl_list_get", &[latch, slot0])?;
        let first_window = builder.ins().icmp_imm_s(IntCC::Equal, done, 0);
        let fill_b = builder.create_block();
        let after_b = builder.create_block();
        builder.ins().brif(first_window, fill_b, &[], after_b, &[]);
        builder.switch_to_block(fill_b);
        builder.seal_block(fill_b);
        state.current_block = Some(fill_b);

        // The step body runs on the TRANSIENT pool (set at its top, flipped
        // back to Persistent at its bottom) and the allocator DRAINS
        // transient segments at end of step. Optimizer moments must outlive
        // every step, so flip to Persistent for the fill and back after —
        // there is no get/set for the pool, and this site is only ever
        // reached from inside that transient region.
        self.compile_call_by_name(builder, "nsl_gpu_set_persistent_pool", &[])?;
        let surface_prev = self.compile_call_by_name(builder, "nsl_gpu_get_alloc_surface", &[])?;
        for (i, &mode) in modes.iter().enumerate() {
            let idx = builder.ins().iconst(cl_types::I64, i as i64);
            let param_i = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, idx])?;

            let s_m = builder.ins().iconst(cl_types::I8, SURFACE_OPTIM_M);
            self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[s_m])?;
            let m_buf =
                self.emit_filled_moment(builder, state, mode, param_i, idx, m_codes, offload)?;
            self.compile_call_by_name(builder, "nsl_list_set", &[state_list_1, idx, m_buf])?;

            if num_state_buffers >= 2 {
                let s_v = builder.ins().iconst(cl_types::I8, SURFACE_OPTIM_V);
                self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[s_v])?;
                let v_buf = if let Some(route_list) = muon_v_gate {
                    // P1 Muon item 9, carried onto the deferred path: v (the
                    // AdamW second moment) is UNREAD on the Muon route, so
                    // route-gate it here exactly as section 4 does for the
                    // non-zero3 case. The route gate wraps the sharding gate
                    // — a Muon-routed param allocates no v on ANY rank, and
                    // an AdamW-routed one still allocates only its share.
                    let routed =
                        self.emit_muon_route_predicate(builder, route_list, idx, param_i)?;
                    let needs_v = builder.ins().icmp_imm_s(IntCC::Equal, routed, 0);
                    let alloc_b = builder.create_block();
                    let skip_b = builder.create_block();
                    let merge_b = builder.create_block();
                    builder.append_block_param(merge_b, cl_types::I64);
                    builder.ins().brif(needs_v, alloc_b, &[], skip_b, &[]);

                    builder.switch_to_block(alloc_b);
                    builder.seal_block(alloc_b);
                    state.current_block = Some(alloc_b);
                    let real = self.emit_filled_moment(
                        builder, state, mode, param_i, idx, v_codes, offload,
                    )?;
                    builder.ins().jump(merge_b, &[BlockArg::Value(real)]);

                    builder.switch_to_block(skip_b);
                    builder.seal_block(skip_b);
                    state.current_block = Some(skip_b);
                    let null_v = builder.ins().iconst(cl_types::I64, 0);
                    builder.ins().jump(merge_b, &[BlockArg::Value(null_v)]);

                    builder.switch_to_block(merge_b);
                    builder.seal_block(merge_b);
                    state.current_block = Some(merge_b);
                    builder.block_params(merge_b)[0]
                } else {
                    self.emit_filled_moment(builder, state, mode, param_i, idx, v_codes, offload)?
                };
                self.compile_call_by_name(builder, "nsl_list_set", &[state_list_2, idx, v_buf])?;
            }
        }
        self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_prev])?;
        self.compile_call_by_name(builder, "nsl_gpu_set_transient_pool", &[])?;
        let one = builder.ins().iconst(cl_types::I64, 1);
        self.compile_call_by_name(builder, "nsl_list_set", &[latch, slot0, one])?;
        builder.ins().jump(after_b, &[]);

        builder.switch_to_block(after_b);
        builder.seal_block(after_b);
        state.current_block = Some(after_b);
        Ok(())
    }

    /// THE Muon-route predicate: `route_flag == 0 && runtime_rank == 2` —
    /// the one definition of "this parameter is stepped by Muon's matrix
    /// path". Item 8 (dispatcher unification): this rule was previously
    /// hand-spelled at three emission sites (batch-skip, resident-momentum,
    /// v-allocation — the last as its own De Morgan inverse) plus twice in
    /// the runtime, with a comment at each pleading that they stay equal.
    ///
    /// It must mirror, EXACTLY:
    ///   - `nsl_muon_step_batch`'s filter (muon_batch.rs): `route != 0` →
    ///     skip, `ndim != 2` → skip, and every LATER exit (device / dtype /
    ///     contiguity / empty matrix) is an ABORT, never a skip. That
    ///     asymmetry is load-bearing: a graceful `continue` added past the
    ///     first two tests would silently DROP params the batch-skip site
    ///     already jumped over. `muon_route_contract_drift` pins both sides.
    ///   - the stdlib branch (`stdlib/nsl/optim/muon.nsl`:
    ///     `adamw_route > 0.5 or len(s) != 2`).
    ///
    /// Callers needing the negation (v-allocation) invert the returned
    /// value (`icmp_imm == 0`) rather than re-deriving the inverse.
    pub(crate) fn emit_muon_route_predicate(
        &mut self,
        builder: &mut FunctionBuilder,
        route_list: Value,
        idx: Value,
        param: Value,
    ) -> Result<Value, CodegenError> {
        let flag_i = self.compile_call_by_name(builder, "nsl_list_get", &[route_list, idx])?;
        let ndim = self.compile_call_by_name(builder, "nsl_tensor_ndim", &[param])?;
        let zero_c = builder.ins().iconst(cl_types::I64, 0);
        let two_c = builder.ins().iconst(cl_types::I64, 2);
        let is_muon = builder.ins().icmp(IntCC::Equal, flag_i, zero_c);
        let is_r2 = builder.ins().icmp(IntCC::Equal, ndim, two_c);
        Ok(builder.ins().band(is_muon, is_r2))
    }

    /// Item 8: emit ONE batched fused-AdamW launch over the FullBuffer
    /// parameter list — the whole list normally, this rank's OWNER SUBSET
    /// under `--zero-stage 1/2`.
    ///
    /// Both Phase-B forks (clipped and unclipped) call this so the ZeRO
    /// decision cannot drift between them; they differ only in `mp_scale`
    /// (the clip factor, folded into the kernel's `m_partial` read, vs 1.0).
    ///
    /// Under ZeRO the subset is not an optimization, it is the correctness
    /// condition: a non-owned parameter's moment buffers are NULL
    /// placeholders (`emit_owner_gated_moment` allocates nothing for
    /// non-owners), so the batched launcher must never be handed one — and
    /// `nsl_zero_owned_step_indices` also performs the non-owner
    /// `m_partial` zero that the per-param loop's skip arm used to do.
    /// Handing the launcher only owners leaves its null assert intact as a
    /// belt rather than something to relax.
    ///
    /// **`owner_gated_moments` is the STAGES-1/2 predicate, not "ZeRO is
    /// on".** That is the whole invariant: the subset is valid exactly when
    /// every parameter's moments were owner-gated, which is true for stages
    /// 1/2 and for no other configuration. The old `&& !zero_enabled`
    /// exclusion was safe under `>= 1` only because it disabled batching for
    /// EVERY stage; narrowing the exclusion without narrowing the predicate
    /// is what opened the gap.
    ///
    /// **Stage 3 is not a third value of this boolean — it has no correct
    /// value.** Item C made stage-3 moments per-parameter: the deferred fill
    /// (`emit_deferred_moment_fill`, see the note at the allocation site)
    /// picks `MomentFill::Elementwise` (a 1/ws SLICE), `OwnerGated` (NULL on
    /// non-owners) or `Full` (replicated — tied / view-rooted / epilogue
    /// params every rank must step) per entry. `nsl_zero_owned_step_indices`
    /// describes none of those three, so `true` would skip the replicated
    /// params on non-owner ranks — the very freeze this paragraph used to
    /// warn about — while `false` hands the launcher the null placeholders
    /// its own assert exists to reject. The invariant for stage 3 is
    /// therefore "must not reach this launcher at all", not "must be
    /// correctly parameterised".
    ///
    /// That is structural, not incidental: stage 3 is refused unless
    /// `csla_active` (and `--weight-stream`, itself requiring
    /// `--layerwise-accum`), and both call sites sit in the `!csla_active`
    /// sub-arm — the SAME local, so `stage 3 => csla_active => short-circuit`
    /// is one implication, not a coincidence of two guards. The CSLA
    /// in-window batched arm is independently gated on
    /// `zero3.is_none() && zero3_elem.is_none()`. Verified by construction:
    /// a hard refusal planted at the top of this function never fires under
    /// any shipped stage-3 configuration.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_fused_multi_launch(
        &mut self,
        builder: &mut FunctionBuilder,
        owner_gated_moments: bool,
        num_params_val: Value,
        lists: (Value, Value, Value, Value),
        sc: &crate::stmt_fase::FusedAdamwScalars,
        // Live learning rate — see `fase_emit_final_step`. `sc.lr` is the
        // plan's base rate and folding it here discards the schedule.
        lr_runtime: Value,
        bc: (Value, Value),
        groups: (Value, Value),
        mp_scale: Value,
    ) -> Result<(), CodegenError> {
        let (param_list, state_list_1, state_list_2, accum) = lists;
        let lr_v = lr_runtime;
        let b1_v = builder.ins().f64const(sc.beta1);
        let omb1_v = builder.ins().f64const(sc.one_minus_beta1);
        let b2_v = builder.ins().f64const(sc.beta2);
        let omb2_v = builder.ins().f64const(sc.one_minus_beta2);
        let eps_v = builder.ins().f64const(sc.eps);
        let wd_v = builder.ins().f64const(sc.wd);

        if owner_gated_moments {
            let il = self.compile_call_by_name(
                builder,
                "nsl_zero_owned_step_indices",
                &[accum, num_params_val],
            )?;
            self.compile_call_by_name(
                builder,
                "nsl_fase_fused_adamw_step_multi_idx",
                &[
                    param_list,
                    state_list_1,
                    state_list_2,
                    accum,
                    il,
                    lr_v,
                    b1_v,
                    omb1_v,
                    b2_v,
                    omb2_v,
                    eps_v,
                    wd_v,
                    bc.0,
                    bc.1,
                    // Parameter-group arguments stay ORIGINAL-position
                    // indexed; the subset launcher documents that it resolves
                    // λ by the caller's numbering, which is why the exempt
                    // list is passed unfiltered alongside a filtered index
                    // list.
                    groups.0,
                    groups.1,
                    mp_scale,
                ],
            )?;
            self.compile_call_by_name(builder, "nsl_list_free", &[il])?;
        } else {
            self.compile_call_by_name(
                builder,
                "nsl_fase_fused_adamw_step_multi",
                &[
                    param_list,
                    state_list_1,
                    state_list_2,
                    accum,
                    lr_v,
                    b1_v,
                    omb1_v,
                    b2_v,
                    omb2_v,
                    eps_v,
                    wd_v,
                    bc.0,
                    bc.1,
                    groups.0,
                    groups.1,
                    mp_scale,
                ],
            )?;
        }
        Ok(())
    }

    /// Item 12: how a train-loop callback body touches the streamed model θ.
    /// Under `--weight-stream` params are EVICTED (`t.data == null`) when a
    /// callback runs, so a model-field read launches on a null pointer (the
    /// #395 crash). This drives a scoped `upload_all` / `reevict_all` bracket
    /// around any body that references the model, turning the runtime crash
    /// into a compile-time-inserted residency window.
    fn analyze_callback_model_touch(
        &self,
        block: &nsl_ast::stmt::Block,
        model_sym: nsl_ast::Symbol,
    ) -> CallbackModelTouch {
        let mut acc = CallbackModelTouch::default();
        for stmt in &block.stmts {
            self.walk_stmt_model_touch(stmt, model_sym, &mut acc);
        }
        acc
    }

    /// NSL in-place tensor mutators — the only calls that write a param
    /// through a receiver/dest operand (`copy_data(dest, src)`,
    /// `zero_inplace(t)`, the `nsl_tensor_{op}_inplace` family). Everything
    /// else is functional (returns a fresh tensor).
    fn is_inplace_mutator(name: &str) -> bool {
        matches!(name, "copy_data" | "copy_") || name.ends_with("_inplace")
    }

    /// Root identifier of an lvalue/access chain (`model.enc.w[0]` -> `model`).
    fn expr_root_ident(e: &nsl_ast::expr::Expr) -> Option<nsl_ast::Symbol> {
        use nsl_ast::expr::ExprKind as E;
        match &e.kind {
            E::Ident(s) => Some(*s),
            E::MemberAccess { object, .. } => Self::expr_root_ident(object),
            E::Subscript { object, .. } => Self::expr_root_ident(object),
            E::Paren(inner) => Self::expr_root_ident(inner),
            _ => None,
        }
    }

    /// Dotted path of a model-rooted member chain, for the diagnostic
    /// (`model.encoder.weight`). Best-effort — falls back to the model name.
    fn model_access_path(&self, e: &nsl_ast::expr::Expr) -> Option<String> {
        use nsl_ast::expr::ExprKind as E;
        match &e.kind {
            E::Ident(s) => Some(self.resolve_sym(*s).to_string()),
            E::MemberAccess { object, member } => {
                let base = self.model_access_path(object)?;
                Some(format!("{base}.{}", self.resolve_sym(*member)))
            }
            E::Subscript { object, .. } => self.model_access_path(object),
            E::Paren(inner) => self.model_access_path(inner),
            _ => None,
        }
    }

    fn walk_stmt_model_touch(
        &self,
        stmt: &nsl_ast::stmt::Stmt,
        model_sym: nsl_ast::Symbol,
        acc: &mut CallbackModelTouch,
    ) {
        use nsl_ast::stmt::StmtKind as S;
        match &stmt.kind {
            S::Assign { target, value, .. } => {
                // A write whose lvalue is rooted at the model mutates θ.
                if Self::expr_root_ident(target) == Some(model_sym) {
                    acc.touches = true;
                    acc.may_write = true;
                    if acc.first_path.is_none() {
                        acc.first_path = self.model_access_path(target);
                    }
                }
                self.walk_expr_model_touch(target, model_sym, acc);
                self.walk_expr_model_touch(value, model_sym, acc);
            }
            S::VarDecl { value: Some(v), .. } => {
                // Binding a model-derived value to a local (`let w = m.field`)
                // creates an ALIAS onto the resident streamed buffer; a later
                // `copy_data(w, ..)` would mutate θ through it without the root
                // ever being `model_sym`. We can't cheaply track the alias, so
                // conservatively treat any model-rooted binding as a possible
                // write (writeback=1 is always safe — for an unmutated param
                // device==mirror, so the extra DtoH is byte-identical).
                if Self::expr_root_ident(v) == Some(model_sym) {
                    acc.touches = true;
                    acc.may_write = true;
                    if acc.first_path.is_none() {
                        acc.first_path = self.model_access_path(v);
                    }
                }
                self.walk_expr_model_touch(v, model_sym, acc)
            }
            S::Expr(e) | S::Return(Some(e)) | S::Yield(Some(e)) => {
                self.walk_expr_model_touch(e, model_sym, acc)
            }
            S::If {
                condition,
                then_block,
                elif_clauses,
                else_block,
            } => {
                self.walk_expr_model_touch(condition, model_sym, acc);
                for s in &then_block.stmts {
                    self.walk_stmt_model_touch(s, model_sym, acc);
                }
                for (c, b) in elif_clauses {
                    self.walk_expr_model_touch(c, model_sym, acc);
                    for s in &b.stmts {
                        self.walk_stmt_model_touch(s, model_sym, acc);
                    }
                }
                if let Some(b) = else_block {
                    for s in &b.stmts {
                        self.walk_stmt_model_touch(s, model_sym, acc);
                    }
                }
            }
            S::For { iterable, body, .. } => {
                self.walk_expr_model_touch(iterable, model_sym, acc);
                for s in &body.stmts {
                    self.walk_stmt_model_touch(s, model_sym, acc);
                }
            }
            S::While { condition, body } => {
                self.walk_expr_model_touch(condition, model_sym, acc);
                for s in &body.stmts {
                    self.walk_stmt_model_touch(s, model_sym, acc);
                }
            }
            // A model read inside any of these still dereferences evicted θ —
            // walk them too, or the residency bracket is silently skipped and
            // the #395 crash returns (e.g. `@no_grad: print(m.x.sum())`).
            S::WhileLet { expr, body, .. } => {
                self.walk_expr_model_touch(expr, model_sym, acc);
                for s in &body.stmts {
                    self.walk_stmt_model_touch(s, model_sym, acc);
                }
            }
            S::Match { subject, arms } => {
                self.walk_expr_model_touch(subject, model_sym, acc);
                for arm in arms {
                    for s in &arm.body.stmts {
                        self.walk_stmt_model_touch(s, model_sym, acc);
                    }
                }
            }
            S::Decorated { stmt, .. } => {
                self.walk_stmt_model_touch(stmt, model_sym, acc);
            }
            _ => {}
        }
    }

    fn walk_expr_model_touch(
        &self,
        e: &nsl_ast::expr::Expr,
        model_sym: nsl_ast::Symbol,
        acc: &mut CallbackModelTouch,
    ) {
        use nsl_ast::expr::ExprKind as E;
        match &e.kind {
            E::Ident(s) if *s == model_sym => {
                acc.touches = true;
                if acc.first_path.is_none() {
                    acc.first_path = Some(self.resolve_sym(*s).to_string());
                }
            }
            E::MemberAccess { object, .. } => {
                if Self::expr_root_ident(e) == Some(model_sym) {
                    acc.touches = true;
                    if acc.first_path.is_none() {
                        acc.first_path = self.model_access_path(e);
                    }
                }
                self.walk_expr_model_touch(object, model_sym, acc);
            }
            E::Call { callee, args } => {
                // Only a genuine IN-PLACE mutator (`copy_data(m.x, ..)`,
                // `m.x.add_inplace(..)`, any `*_inplace`) writes θ. Functional
                // methods (`.sum()`, `.transpose()`, `.mean()`) return new
                // tensors and never mutate the receiver, so they are read-only
                // — flagging them would force a needless full-model writeback
                // on every logging callback. The callee name is the free-fn
                // ident or the method member.
                let callee_name = match &callee.kind {
                    E::Ident(s) => Some(self.resolve_sym(*s).to_string()),
                    E::MemberAccess { member, .. } => {
                        Some(self.resolve_sym(*member).to_string())
                    }
                    _ => None,
                };
                let is_mutator = callee_name.as_deref().is_some_and(Self::is_inplace_mutator);
                // A FREE-FN call passing a model-rooted param to a callee that
                // is not a known read-only sink could mutate that param in
                // place (`my_ema(m.field, ..)`), so treat it conservatively as
                // a write. `print`/`model_save` provably don't mutate their
                // tensor args. Method calls are covered by the mutator check
                // above (functional methods return fresh tensors).
                let is_free_fn = matches!(&callee.kind, E::Ident(_));
                let is_readonly_sink = callee_name
                    .as_deref()
                    .is_some_and(|n| matches!(n, "print" | "model_save"));
                let unknown_free_fn_write =
                    is_free_fn && !is_readonly_sink && !Self::is_inplace_mutator(callee_name.as_deref().unwrap_or(""));
                if is_mutator || unknown_free_fn_write {
                    // Dest is the model-rooted operand: the receiver for the
                    // method form (`m.x.add_inplace(..)`), an arg for the
                    // free-fn form (`copy_data(m.x, ..)` / `my_ema(m.x, ..)`).
                    let receiver_model = matches!(&callee.kind, E::MemberAccess { object, .. }
                        if Self::expr_root_ident(object) == Some(model_sym));
                    let arg_model = args
                        .iter()
                        .any(|a| Self::expr_root_ident(&a.value) == Some(model_sym));
                    if receiver_model || arg_model {
                        acc.touches = true;
                        acc.may_write = true;
                        if acc.first_path.is_none() {
                            acc.first_path = args
                                .iter()
                                .find_map(|a| {
                                    (Self::expr_root_ident(&a.value) == Some(model_sym))
                                        .then(|| self.model_access_path(&a.value))
                                        .flatten()
                                })
                                .or_else(|| match &callee.kind {
                                    E::MemberAccess { object, .. } => {
                                        self.model_access_path(object)
                                    }
                                    _ => None,
                                });
                        }
                    }
                }
                self.walk_expr_model_touch(callee, model_sym, acc);
                for a in args {
                    self.walk_expr_model_touch(&a.value, model_sym, acc);
                }
            }
            E::BinaryOp { left, right, .. } => {
                self.walk_expr_model_touch(left, model_sym, acc);
                self.walk_expr_model_touch(right, model_sym, acc);
            }
            E::UnaryOp { operand, .. } | E::Paren(operand) | E::Await(operand) => {
                self.walk_expr_model_touch(operand, model_sym, acc)
            }
            E::Pipe { left, right } => {
                self.walk_expr_model_touch(left, model_sym, acc);
                self.walk_expr_model_touch(right, model_sym, acc);
            }
            E::Subscript { object, .. } => {
                self.walk_expr_model_touch(object, model_sym, acc)
            }
            E::ListLiteral(xs) | E::TupleLiteral(xs) => {
                for x in xs {
                    self.walk_expr_model_touch(x, model_sym, acc);
                }
            }
            E::FString(parts) => {
                for p in parts {
                    if let nsl_ast::expr::FStringPart::Expr(x) = p {
                        self.walk_expr_model_touch(x, model_sym, acc);
                    }
                }
            }
            _ => {}
        }
    }

    /// Item 12 — open bracket: if `--weight-stream` is active and the
    /// callback body references model θ, make every streamed param resident
    /// (`upload_all`) so the body's reads don't launch on evicted (null)
    /// data. Returns `Some(may_write)` when a bracket was opened — the caller
    /// passes it to `emit_callback_residency_close`. Returns `None` (no-op)
    /// when streaming is off or the callback never touches the model, so the
    /// steady-state transfer arithmetic the CSLA gates assert is unchanged.
    pub(crate) fn emit_callback_residency_open(
        &mut self,
        builder: &mut FunctionBuilder,
        body: &nsl_ast::stmt::Block,
        model_sym: nsl_ast::Symbol,
        cb_name: &str,
    ) -> Result<Option<bool>, CodegenError> {
        if !self.compile_options.weight_stream.enabled {
            return Ok(None);
        }
        let touch = self.analyze_callback_model_touch(body, model_sym);
        if !touch.touches {
            return Ok(None);
        }
        nsl_runtime::nsl_log!(WARN, "weight-stream", 
            "[weight-stream] callback '{}' reads model state ({}); inserting a \
             scoped upload/re-evict bracket ({} writeback) so its reads see \
             resident \u{3b8} instead of crashing on evicted (null) data",
            cb_name,
            touch.first_path.as_deref().unwrap_or("model"),
            if touch.may_write { "with" } else { "without" },
        );
        self.compile_call_by_name(builder, "nsl_weight_stream_upload_all", &[])?;
        Ok(Some(touch.may_write))
    }

    /// Item 12 — close bracket: restore the streamed (evicted) invariant after
    /// a guarded callback body. `writeback=1` when the body might have mutated
    /// θ so the change survives the next window's upload; `0` for a read-only
    /// body (logging, `model_save`).
    pub(crate) fn emit_callback_residency_close(
        &mut self,
        builder: &mut FunctionBuilder,
        guard: Option<bool>,
    ) -> Result<(), CodegenError> {
        if let Some(may_write) = guard {
            let wb = builder
                .ins()
                .iconst(cl_types::I64, if may_write { 1 } else { 0 });
            self.compile_call_by_name(builder, "nsl_weight_stream_reevict_all", &[wb])?;
        }
        Ok(())
    }

    /// Item 10: emit a contiguous layer-pack UPLOAD. Builds an `NslList` of
    /// the pack's param tensor pointers (from their `param_list` indices) and
    /// hands it to the runtime, which stages the whole pack into ONE device
    /// arena transfer. The list build is a few cheap CPU calls; the win is the
    /// single HtoD it replaces N of.
    pub(crate) fn emit_ws_pack_upload(
        &mut self,
        builder: &mut FunctionBuilder,
        param_list: Value,
        idxs: &[i64],
    ) -> Result<(), CodegenError> {
        if idxs.is_empty() {
            return Ok(());
        }
        let pwlist = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        for &idx in idxs {
            let iv = builder.ins().iconst(cl_types::I64, idx);
            let pw = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
            self.compile_call_by_name(builder, "nsl_list_push", &[pwlist, pw])?;
        }
        self.compile_call_by_name(builder, "nsl_weight_stream_upload_pack", &[pwlist])?;
        self.compile_call_by_name(builder, "nsl_list_free", &[pwlist])?;
        Ok(())
    }

    /// Item 10: emit a contiguous layer-pack EVICT (one DtoH when writeback).
    pub(crate) fn emit_ws_pack_evict(
        &mut self,
        builder: &mut FunctionBuilder,
        param_list: Value,
        idxs: &[i64],
        writeback: i64,
    ) -> Result<(), CodegenError> {
        if idxs.is_empty() {
            return Ok(());
        }
        let pwlist = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        for &idx in idxs {
            let iv = builder.ins().iconst(cl_types::I64, idx);
            let pw = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
            self.compile_call_by_name(builder, "nsl_list_push", &[pwlist, pw])?;
        }
        let wb = builder.ins().iconst(cl_types::I64, writeback);
        self.compile_call_by_name(builder, "nsl_weight_stream_evict_pack", &[pwlist, wb])?;
        self.compile_call_by_name(builder, "nsl_list_free", &[pwlist])?;
        Ok(())
    }

    /// Item 11: emit an ASYNC pack transfer (`fn_name` is upload_pack's async
    /// sibling `nsl_weight_stream_prefetch_pack`, or the `nsl_weight_stream_
    /// await_pack` consumer). Shares the pw-list build with the sync helpers.
    pub(crate) fn emit_ws_pack_single(
        &mut self,
        builder: &mut FunctionBuilder,
        param_list: Value,
        idxs: &[i64],
        fn_name: &str,
    ) -> Result<(), CodegenError> {
        if idxs.is_empty() {
            return Ok(());
        }
        let pwlist = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        for &idx in idxs {
            let iv = builder.ins().iconst(cl_types::I64, idx);
            let pw = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
            self.compile_call_by_name(builder, "nsl_list_push", &[pwlist, pw])?;
        }
        self.compile_call_by_name(builder, fn_name, &[pwlist])?;
        self.compile_call_by_name(builder, "nsl_list_free", &[pwlist])?;
        Ok(())
    }

    fn compile_train_block(
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
                        crate::stmt::invoke_cpdt_if_enabled(self, Some(&applied), Some(train))
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
                        crate::stmt::invoke_cpdt_if_enabled(self, None, Some(train))
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
    fn compile_train_block_inner(
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

    /// Emit the tape-based AD backward pass: tape_start, compile forward,
    /// find loss, tape_backward, tape_stop. Returns `(grads_list, loss_val)`.
    fn compile_tape_backward(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        step_body: &nsl_ast::stmt::Block,
        param_list: Value,
    ) -> Result<(Value, Value), CodegenError> {
        // Set training mode = true, then start tape recording
        let true_val = builder.ins().iconst(cl_types::I8, 1);
        self.compile_call_by_name(builder, "nsl_set_training_mode", &[true_val])?;
        self.compile_call_by_name(builder, "nsl_tape_start", &[param_list])?;

        // Compile step body stmts
        // Suppress tensor temporary cleanup — tape holds raw pointers to intermediates.
        state.flags.in_tape_region = true;
        for stmt in &step_body.stmts {
            self.compile_stmt(builder, state, stmt)?;
        }
        // ELTLS: free tape-held tensors before clearing the tape flag.
        self.free_tape_held_tensors(builder, state);
        state.flags.in_tape_region = false;

        // Find loss variable — look for "loss" in state.variables by name
        let loss_val = {
            let mut found = None;
            for (sym, (var, _)) in &state.variables {
                if self.resolve_sym(*sym) == "loss" {
                    found = Some(builder.use_var(*var));
                    break;
                }
            }
            found.ok_or_else(|| {
                CodegenError::new("train step body must assign to a variable named 'loss'")
            })?
        };

        // Run backward pass — the TRAIN entry arms the disconnection
        // backstop (all-params-zeros aborts instead of silently training
        // on weight decay alone). Grad blocks keep plain nsl_tape_backward.
        let grads_list = self
            .compile_call_by_name(builder, "nsl_tape_backward_train", &[loss_val, param_list])?;

        // Stop tape and restore eval mode
        self.compile_call_by_name(builder, "nsl_tape_stop", &[])?;
        let false_val = builder.ins().iconst(cl_types::I8, 0);
        self.compile_call_by_name(builder, "nsl_set_training_mode", &[false_val])?;

        Ok((grads_list, loss_val))
    }

    /// CFTP §4.3 / Tier A activation (spec 2026-05-17): probe a batch dict
    /// for segment_ids + doc_starts. When the DataLoader has packing=true,
    /// the packer (packing.rs::packed_batch_to_dict) emits both tensors per
    /// batch. Extract device pointers and stash them in the thread-local
    /// packing registry; the model's compiled @flash_attention call sites
    /// read them per launch.
    ///
    /// Probing at runtime (not codegen time) lets a single train block
    /// tolerate mixed-batch workloads or DataLoader implementations that
    /// conditionally emit segment_ids based on actual document structure.
    /// The probe is one CStr lookup — negligible cost vs kernel launches.
    ///
    /// Called once per micro-batch in the train loop, and again per buffered
    /// micro-batch at the head of the CSLA window-backward body (the registry
    /// holds the LAST batch's pointers otherwise, which would mis-mask every
    /// earlier micro-batch's replayed attention backward).
    pub(crate) fn emit_packing_registry_stash(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        batch_val: Value,
    ) -> Result<(), CodegenError> {
        use cranelift_codegen::ir::condcodes::IntCC;
        let k_seg = self.compile_string_literal(builder, "segment_ids")?;
        let has_seg =
            self.compile_call_by_name(builder, "nsl_dict_contains", &[batch_val, k_seg])?;
        let has_seg_block = builder.create_block();
        let no_seg_block = builder.create_block();
        let after_block = builder.create_block();
        let has_seg_cond = builder.ins().icmp_imm_s(IntCC::NotEqual, has_seg, 0);
        builder
            .ins()
            .brif(has_seg_cond, has_seg_block, &[], no_seg_block, &[]);

        // Packing-enabled batch: extract device pointers and set the
        // registry. Both segment_ids and doc_starts must be present
        // together — the packer emits them as a pair.
        builder.switch_to_block(has_seg_block);
        builder.seal_block(has_seg_block);
        let seg_tensor =
            self.compile_call_by_name(builder, "nsl_dict_get_str", &[batch_val, k_seg])?;
        let k_doc = self.compile_string_literal(builder, "doc_starts")?;
        let doc_tensor =
            self.compile_call_by_name(builder, "nsl_dict_get_str", &[batch_val, k_doc])?;
        let seg_data_ptr =
            self.compile_call_by_name(builder, "nsl_tensor_data_ptr", &[seg_tensor])?;
        let doc_data_ptr =
            self.compile_call_by_name(builder, "nsl_tensor_data_ptr", &[doc_tensor])?;
        // Item 17 phase 3a: the fused-AD reads consume these VARIABLES; the
        // thread-local set below stays for the model-METHOD readers
        // (`expr/advanced.rs`), which live in a different Cranelift function
        // until the phase-3b ABI change.
        if let Some((sv, dv)) = self.packing_meta_vars {
            builder.def_var(sv, seg_data_ptr);
            builder.def_var(dv, doc_data_ptr);
        }
        self.compile_call_by_name(
            builder,
            "nsl_packing_metadata_set",
            &[seg_data_ptr, doc_data_ptr],
        )?;
        builder.ins().jump(after_block, &[]);

        // Packing-disabled batch: clear the registry so stale state
        // from a prior step doesn't leak. Setting to (0, 0) is the
        // spec-defined sentinel for "identity path" at the kernel.
        builder.switch_to_block(no_seg_block);
        builder.seal_block(no_seg_block);
        let zero = builder.ins().iconst(cl_types::I64, 0);
        if let Some((sv, dv)) = self.packing_meta_vars {
            builder.def_var(sv, zero);
            builder.def_var(dv, zero);
        }
        self.compile_call_by_name(builder, "nsl_packing_metadata_set", &[zero, zero])?;
        builder.ins().jump(after_block, &[]);

        builder.switch_to_block(after_block);
        builder.seal_block(after_block);
        // The probe's brif TERMINATED the block the caller was in; leaving
        // `state.current_block` pointing at it makes compile_stmt's
        // filled-block guard silently SKIP every subsequent statement.
        // That was the entire tape×DataLoader failure: the step body
        // compiled to nothing, so the 'loss' binding never landed and the
        // tape path refused with "must assign to a variable named 'loss'"
        // (source AD never noticed — it lowers the extracted Wengert list
        // without consulting current_block).
        state.current_block = Some(after_block);

        // PCA Tier A (spec §6.1): when a segment-masked kernel was
        // synthesized for this module, warn once if no segment_ids ever
        // appear in the first N steps (DataLoader-never-packs footgun).
        // Gated on the ACTUAL synthesized config so non-packed training
        // (the common case) never sees this call. has_seg is the
        // nsl_dict_contains("segment_ids") i64 result from above.
        let module_is_masked = self
            .kernels
            .flash_attention_context
            .as_ref()
            .and_then(|c| c.csha_training_config.as_ref())
            .map(|cfg| cfg.segment_masked)
            .unwrap_or(false);
        if module_is_masked {
            self.compile_call_by_name(builder, "nsl_pca_packing_mismatch_check", &[has_seg])?;
        }
        Ok(())
    }

    pub(crate) fn free_wengert_owned_values(
        &mut self,
        builder: &mut FunctionBuilder,
        owned_values: &[(crate::wengert::VarId, Value, crate::wengert::WengertType)],
        retained: &std::collections::HashSet<crate::wengert::VarId>,
    ) -> Result<(), CodegenError> {
        for (var_id, value, value_type) in owned_values {
            if retained.contains(var_id) {
                continue;
            }
            match value_type {
                crate::wengert::WengertType::Tensor => {
                    self.compile_call_by_name(builder, "nsl_tensor_free", &[*value])?;
                }
                crate::wengert::WengertType::List => {
                    self.compile_call_by_name(builder, "nsl_list_free", &[*value])?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// M43b: Emit pipeline-parallel training loop with gradient serialization.
    ///
    /// When a model carries `@pipeline(stages=N)`, the train block emits:
    ///   1. `nsl_pipeline_init(num_stages, schedule_type, num_micro_batches)`
    ///   2. Extract model param_list, optimizer config, and step body.
    ///   3. Forward pass under tape recording — compile step body to produce
    ///      activations and loss.
    ///   4. Activation send — serialize the loss tensor to the next pipeline
    ///      stage via `nsl_pipeline_send`.
    ///   5. Backward pass — `nsl_tape_backward` computes per-parameter
    ///      gradients from the recorded tape.
    ///   6. Gradient send — serialize each parameter gradient to the previous
    ///      pipeline stage via `nsl_pipeline_send_grad`.
    ///   7. Optimizer step — apply optimizer update using the computed
    ///      gradients (same dispatch as the non-pipelined path).
    ///   8. `nsl_pipeline_barrier()` — synchronize all stages.
    ///   9. Cleanup — free gradient tensors, param_list, optimizer buffers,
    ///      and `nsl_pipeline_destroy()`.
    ///
    pub(crate) fn is_trainable_param_name(&self, param_name: &str) -> bool {
        is_trainable_param_leaf_name(param_name)
    }

    pub(crate) fn enumerate_all_model_tensor_paths(&self, var_name: &str, type_name: &str) -> Vec<String> {
        let mut paths = Vec::new();
        self.enumerate_tensor_paths_recursive(var_name, type_name, &mut paths, 0, true);
        paths
    }

    /// Enumerate all tensor field paths in a model struct via DFS.
    ///
    /// This mirrors the compiler's view of nested models and fixed arrays, so
    /// the emitted param_list and source-AD parameter resolution stay aligned.
    pub(crate) fn enumerate_model_tensor_paths(&self, var_name: &str, type_name: &str) -> Vec<String> {
        let mut paths = Vec::new();
        self.enumerate_tensor_paths_recursive(var_name, type_name, &mut paths, 0, false);
        paths
    }

    fn enumerate_tensor_paths_recursive(
        &self,
        prefix: &str,
        type_name: &str,
        paths: &mut Vec<String>,
        depth: usize,
        include_nontrainable: bool,
    ) {
        if depth > 16 {
            return;
        }

        let layout = match self.types.struct_layouts.get(type_name) {
            Some(layout) => layout.clone(),
            None => return,
        };
        let field_types = self.models.model_field_types.get(type_name).cloned();

        for field in &layout.fields {
            let field_path = format!("{}.{}", prefix, field.name);
            let field_type = field_types
                .as_ref()
                .and_then(|types| types.get(&field.name));

            if let Some(field_type) = field_type {
                if field_type.starts_with('[') && field_type.contains(';') {
                    let inner = field_type.trim_start_matches('[').trim_end_matches(']');
                    let parts: Vec<&str> = inner.split(';').collect();
                    if parts.len() == 2 {
                        let elem_type = parts[0].trim();
                        let count: usize = parts[1].trim().parse().unwrap_or(0);
                        for index in 0..count {
                            let elem_path = format!("{}.{}", field_path, index);
                            self.enumerate_tensor_paths_recursive(
                                &elem_path,
                                elem_type,
                                paths,
                                depth + 1,
                                include_nontrainable,
                            );
                        }
                    }
                    continue;
                }

                self.enumerate_tensor_paths_recursive(
                    &field_path,
                    field_type,
                    paths,
                    depth + 1,
                    include_nontrainable,
                );
                continue;
            }

            if field.cl_type == cl_types::I64
                && (include_nontrainable || self.is_trainable_param_name(&field_path))
            {
                paths.push(field_path);
            }
        }

        // WRGA B.3.2 Option 3: include synthesized adapter-injected fields
        // (lora_A_*, lora_B_*, ia3_scale_*, gate_*) in the ALL-paths
        // enumeration. Source-AD reads this as `trainable_tensor_param_paths`
        // so the gradient-summary diagnostic counts them (B.5 direct probe).
        //
        // Gated on `include_nontrainable` so `enumerate_model_tensor_paths`
        // (used to build the runtime param_list) does NOT return these —
        // runtime load via `load_nested_field` can't traverse the adapter
        // side-table at that point in codegen.
        if include_nontrainable {
            for site in self.bus.adapter_sites() {
                if site.target_model != type_name {
                    continue;
                }
                if site.input_dim == 0 || site.output_dim == 0 {
                    continue;
                }
                for synth in &site.synthesized_fields {
                    let synth_path = format!("{}.{}", prefix, synth);
                    paths.push(synth_path);
                }
            }
        }
    }

    /// Load a nested model field by traversing struct layouts along a compound name path.
    ///
    /// For a compound name like `m.blocks.0.attn.wq`, emits Cranelift IR to:
    /// 1. Start at `base_ptr` (pointer to top-level model struct)
    /// 2. Load `blocks` field from the top-level layout (FixedArray base)
    /// 3. Index element `0` from the array (load pointer at offset 0*8)
    /// 4. Load `attn` field from the TransformerBlock layout (sub-model pointer)
    /// 5. Load `wq` field from the GroupedQueryAttention layout (tensor pointer)
    ///
    /// Returns None if the path cannot be resolved through the struct layouts.
    pub(crate) fn load_nested_field(
        &self,
        builder: &mut FunctionBuilder,
        base_ptr: Value,
        top_layout: &crate::context::StructLayout,
        top_type_name: &str,
        compound_name: &str,
    ) -> Option<Value> {
        let parts: Vec<&str> = compound_name.split('.').collect();
        if parts.len() < 2 {
            return None;
        }

        // State: current struct pointer and current type name (for layout/field_type lookup)
        let mut current_ptr = base_ptr;
        let mut current_type_name = top_type_name.to_string();
        let mut current_layout = top_layout.clone();

        // Skip first component (model variable name like "m")
        let path = &parts[1..];

        let mut i = 0;
        while i < path.len() {
            let part = path[i];
            let is_last = i == path.len() - 1;

            // Check if this is a numeric array index (from FixedArray unrolling)
            if let Ok(array_idx) = part.parse::<usize>() {
                // current_ptr is already pointing to the base of the inline array
                // region (set by the preceding FixedArray field handler).
                // Each element is an i64 pointer at offset array_idx * 8.
                let elem_ptr = builder.ins().load(
                    cl_types::I64,
                    cranelift_codegen::ir::MemFlagsData::trusted(),
                    current_ptr,
                    (array_idx * 8) as i32,
                );
                if is_last {
                    return Some(elem_ptr);
                }
                current_ptr = elem_ptr;
                // current_layout and current_type_name were already set to the
                // element type by the preceding array field handler.
                i += 1;
                continue;
            }

            // Named field: look up in current struct layout
            let field = current_layout.fields.iter().find(|f| f.name == part)?;

            // Check if this field is a FixedArray type
            let field_type = self
                .models
                .model_field_types
                .get(&current_type_name)
                .and_then(|ft| ft.get(part))
                .cloned();

            if let Some(ref ft) = field_type
                && ft.starts_with('[') && ft.contains(';')
            {
                // FixedArray field: slots are stored inline in the parent struct.
                // DON'T load the field value — instead compute the address of the
                // array base region within the parent struct.
                let inner = ft.trim_start_matches('[').trim_end_matches(']');
                let elem_type = inner.split(';').next().unwrap_or("").trim();

                // Set current_ptr to address of array base in parent struct
                current_ptr = builder.ins().iadd_imm_s(current_ptr, field.offset as i64);
                current_type_name = elem_type.to_string();
                current_layout = self.types.struct_layouts.get(elem_type)?.clone();
                // Next component should be a numeric index
                i += 1;
                continue;
            }

            // Regular field: load the value
            let field_val = builder.ins().load(
                field.cl_type,
                cranelift_codegen::ir::MemFlagsData::trusted(),
                current_ptr,
                field.offset as i32,
            );

            if is_last {
                return Some(field_val);
            }

            // Navigate into sub-model struct
            current_ptr = field_val;
            if let Some(ref ft) = field_type {
                current_type_name = ft.clone();
                current_layout = self.types.struct_layouts.get(ft)?.clone();
            } else {
                // No type info — can't continue traversal
                return None;
            }

            i += 1;
        }

        None
    }

    fn compile_grad_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        grad: &nsl_ast::block::GradBlock,
    ) -> Result<(), CodegenError> {
        // 1. Compile targets expression to get param tensor ptr
        let targets_val = self.compile_expr(builder, state, &grad.targets)?;

        let (loss_tensor, grad_tensor) = if self.features.source_ad_enabled {
            match self.compile_source_ad_grad_block(builder, state, grad, targets_val)? {
                Some(source_ad) => source_ad,
                None => self.compile_tape_grad_block(builder, state, grad, targets_val)?,
            }
        } else {
            self.compile_tape_grad_block(builder, state, grad, targets_val)?
        };

        // 8. Bind output variables if pattern exists
        //    loss is bound as scalar tensor ptr (I64) — use .item() for f64
        //    grads is bound as gradient tensor ptr (I64)
        if let Some(ref pattern) = grad.outputs {
            match &pattern.kind {
                PatternKind::Tuple(pats) if pats.len() == 2 => {
                    // Bind loss (scalar tensor ptr)
                    if let PatternKind::Ident(loss_sym) = &pats[0].kind {
                        let var = builder.declare_var(cl_types::I64);
                        builder.def_var(var, loss_tensor);
                        state.variables.insert(*loss_sym, (var, cl_types::I64));
                    }
                    // Bind grads (tensor ptr)
                    if let PatternKind::Ident(grads_sym) = &pats[1].kind {
                        let var = builder.declare_var(cl_types::I64);
                        builder.def_var(var, grad_tensor);
                        state.variables.insert(*grads_sym, (var, cl_types::I64));
                    }
                }
                _ => {
                    return Err(CodegenError::new(
                        "grad block output must be `let (loss, grads) = grad(...):`",
                    ));
                }
            }
        }

        // Gap I.B: drop stale CSHA per-function cache entries so a
        // subsequent train/grad block in the same module gets a clean
        // slate (Cranelift `Value` IDs reset per function and would
        // otherwise alias against leftover keys).
        self.clear_csha_per_function_caches();

        Ok(())
    }

    /// Emit the runtime in-place-suppression guard around a source-AD FORWARD
    /// primal pass. Raise (`on=true`) before lowering the forward `WengertList`
    /// so FBIP does not overwrite a uniquely-owned input the adjoint still reads
    /// (e.g. `silu(x@W)`'s matmul temp, refcount 1, feeding an input-reading
    /// `SiluBackward`); lower (`on=false`) before the adjoint pass so backward
    /// FBIP still reclaims memory. Tape-AD gets this for free from
    /// `is_recording()`; source-AD builds no tape, so every source-AD forward
    /// site — grad blocks, train blocks, and model calibration — must bracket
    /// its primal lowering with this. Paired inc/dec so nested blocks compose.
    pub(crate) fn emit_inplace_suppress(
        &mut self,
        builder: &mut FunctionBuilder,
        on: bool,
    ) -> Result<(), CodegenError> {
        let v = builder.ins().iconst(cl_types::I64, i64::from(on));
        self.compile_call_by_name(builder, "nsl_set_inplace_suppressed", &[v])?;
        Ok(())
    }

    /// Item 3: bake the derived [`crate::parameter_plan::ParameterPlan`] into
    /// the binary and assert the runtime realized it.
    ///
    /// `nsl_weight_stream_register` chooses a parameter's residency backend
    /// from *global* flags (`zero3_active()` > `srbf16_active()` > host
    /// mirrors) while the plan is per-parameter and compile-time. Nothing
    /// otherwise connects the two, and a mismatch is silent: a parameter that
    /// reached `register` before its mode was enabled lands in the host-mirror
    /// table, trains in f32, and the run exits 0 with a plausible loss curve.
    /// One `declare` per parameter plus one `verify` closes that gap. Both
    /// are emitted inside the per-micro-batch registration region, so the
    /// check re-runs every micro-batch (catching drift, not just the first
    /// step).
    ///
    /// Cost, stated precisely because the two populations differ: the belt
    /// above emits `2s` calls per micro-batch (a `nsl_list_get` + a
    /// `register` for each of the `s` STREAMED parameters); this adds
    /// `2p + 1`, where `p` is ALL parameters — `p >= s`, since residents are
    /// declared too (see below). Measured as no wall-clock change on the
    /// CSLA FFN fixture.
    ///
    /// EVERY parameter is declared, not only the streamed ones. A resident
    /// parameter expects "registered with no backend", which is falsifiable
    /// and worth checking: the streaming schedule deliberately excludes
    /// view-rooted parameters (a buffered `transpose(w)` caches a pointer
    /// into θ's storage, so registering θ would free it under the live view —
    /// the #397 corruption hazard). If such a parameter ever leaks back into
    /// a registration belt, this is what says so.
    pub(crate) fn emit_param_plan_check(
        &mut self,
        builder: &mut FunctionBuilder,
        plan: &crate::parameter_plan::ParameterPlan,
        param_list: Value,
    ) -> Result<(), CodegenError> {
        // Nothing is registered anywhere, so there is no cross-check to make
        // and no call is emitted at all.
        if !plan.has_streamed() {
            return Ok(());
        }
        // Collected first so `plan` is not borrowed across the &mut self calls.
        let declares: Vec<(i64, i64)> = plan
            .entries()
            .iter()
            .map(|e| (e.idx, e.runtime_flags()))
            .collect();
        for (idx, flags) in declares {
            let iv = builder.ins().iconst(cl_types::I64, idx);
            let fv = builder.ins().iconst(cl_types::I64, flags);
            let pw = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
            self.compile_call_by_name(builder, "nsl_param_plan_declare", &[pw, iv, fv])?;
        }
        self.compile_call_by_name(builder, "nsl_param_plan_verify", &[])?;
        Ok(())
    }

    fn compile_source_ad_grad_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        grad: &nsl_ast::block::GradBlock,
        targets_val: Value,
    ) -> Result<Option<(Value, Value)>, CodegenError> {
        nsl_runtime::nsl_log!(INFO, "nsl", "[nsl] Using source-to-source AD for grad block");

        // Cycle-10 §5.3 Task 6 wire-up (grad block): route per-fn
        // @checkpoint(policy=...) policies into the extractor. Empty map
        // = byte-identity preserved.
        let mut extractor = crate::source_ad::WengertExtractor::new(self.interner)
            .with_checkpoint_policies(if self.compile_options.diagnostics.training_reference {
                    Default::default() // P1.7: ignore @checkpoint decorators in the reference path
                } else {
                    self.compile_options.checkpoint.policies.clone()
                });
        extractor.set_model_method_bodies(self.models.model_method_bodies.clone());
        extractor.set_model_field_types(self.models.model_field_types.clone());
        // WRGA B.3.2 Option 3: plumb synth overrides so the extractor
        // resolves sentinel-Ident callees/members emitted by the adapter
        // rewrite.
        extractor.set_synth_call_names(self.synth_call_names.clone());
        extractor.set_synth_member_names(self.synth_member_names.clone());
        self.register_source_ad_model_instances(&mut extractor, state);

        for sym in self.variables_in_name_order(state) {
            extractor.register_input(sym);
        }

        if !extractor.extract_stmts(&grad.body.stmts) {
            // Same contract as the train-block site: a recorded refusal
            // (e.g. unresolvable dropout probability) aborts the compile —
            // the tape fallback would silently reintroduce the default the
            // refusal exists to prevent.
            if let Some(msg) = extractor.pending_refusal() {
                return Err(CodegenError::new(format!(
                    "source-AD extraction refused: {msg}"
                )));
            }
            nsl_runtime::nsl_log!(WARN, "nsl", 
                "[nsl] source AD extraction failed in grad block, falling back to tape-based AD"
            );
            return Ok(None);
        }

        let loss_expr = grad
            .body
            .stmts
            .last()
            .and_then(|stmt| match &stmt.kind {
                StmtKind::Expr(expr) => Some(expr),
                _ => None,
            })
            .ok_or_else(|| {
                CodegenError::new("grad block must end with an expression (the loss)")
            })?;

        let Some(loss_var_id) = self.resolve_source_ad_expr_var_id(&extractor, loss_expr, true)
        else {
            nsl_runtime::nsl_log!(WARN, "nsl", 
                "[nsl] source AD could not resolve grad block loss, falling back to tape-based AD"
            );
            return Ok(None);
        };
        extractor.set_output(loss_var_id);

        let target_var_id = match &grad.targets.kind {
            ExprKind::Ident(_) | ExprKind::MemberAccess { .. } => {
                self.resolve_source_ad_expr_var_id(&extractor, &grad.targets, false)
            }
            _ => {
                nsl_runtime::nsl_log!(WARN, "nsl", 
                    "[nsl] source AD does not yet resolve this grad target shape, falling back to tape-based AD"
                );
                return Ok(None);
            }
        };
        let Some(target_var_id) = target_var_id else {
            nsl_runtime::nsl_log!(WARN, "nsl", 
                "[nsl] source AD could not resolve grad target, falling back to tape-based AD"
            );
            return Ok(None);
        };

        // Both walks in name order, as in the train block's source-AD
        // arm: `use_var` numbers a value per call (see
        // `variables_in_name_order`).
        let state_vars_by_name: std::collections::HashMap<String, Value> = self
            .variables_in_name_order(state)
            .into_iter()
            .map(|sym| {
                let (cvar, _) = state.variables[&sym];
                (self.resolve_sym(sym).to_string(), builder.use_var(cvar))
            })
            .collect();
        let mut primal_vars = crate::wengert_lower::VarMap::new();

        let mut symbol_vars: Vec<(nsl_ast::Symbol, crate::wengert::VarId)> = extractor
            .symbol_var_map()
            .iter()
            .map(|(sym, vid)| (*sym, *vid))
            .collect();
        symbol_vars.sort_by_key(|&(sym, vid)| (vid, self.resolve_sym(sym)));
        for (sym, vid) in symbol_vars {
            if primal_vars.contains_key(&vid) {
                continue;
            }
            if let Some(&(cvar, _)) = state.variables.get(&sym) {
                primal_vars.insert(vid, builder.use_var(cvar));
            } else {
                let name = self.resolve_sym(sym).to_string();
                if let Some(&val) = state_vars_by_name.get(&name) {
                    primal_vars.insert(vid, val);
                }
            }
        }

        for op in &extractor.wengert_list().ops {
            if let crate::wengert::PrimalOp::Input(name) = &op.op {
                if primal_vars.contains_key(&op.result) {
                    continue;
                }
                if let Some(&val) = state_vars_by_name.get(name) {
                    primal_vars.insert(op.result, val);
                }
            }
        }

        for (compound_name, vid) in extractor.named_param_var_ids() {
            if primal_vars.contains_key(vid) {
                continue;
            }
            if let Some(val) = self.load_source_ad_named_param(builder, state, compound_name) {
                primal_vars.insert(*vid, val);
            }
        }

        // Preserve primal inputs for the adjoint (see `emit_inplace_suppress`).
        self.emit_inplace_suppress(builder, true)?;
        let full_lowered = crate::wengert_lower::compile_wengert_ops(
            self,
            builder,
            state,
            extractor.wengert_list(),
            &primal_vars,
            None, // FASE on_param_grad hook — wired in Task 3
        )?;
        self.emit_inplace_suppress(builder, false)?;

        let full_vars = &full_lowered.var_map;

        let loss_tensor = *full_vars.get(&loss_var_id).ok_or_else(|| {
            CodegenError::new("source AD: loss VarId not found in compiled grad graph")
        })?;

        let mut retained_full_vars = std::collections::HashSet::new();
        retained_full_vars.insert(loss_var_id);

        let mut grad_tensor =
            self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[targets_val])?;

        let start_var = extractor.next_var_id();
        let mut generator = crate::source_ad::AdjointGenerator::new(start_var);
        let mut adjoint = generator.generate(extractor.wengert_list());

        if let Some(target_adj_var) = generator.adjoint_of(target_var_id) {
            let needed = std::collections::HashSet::from([target_adj_var]);
            adjoint.ops = crate::source_ad::eliminate_dead_gradients(&adjoint.ops, &needed);
            // P5 item 20 slice B (bit-exact SwiGLU gate fusion; also applied
            // on the train path).
            crate::source_ad::fuse_swiglu_gate_backward(&mut adjoint.ops, &needed);
            // P5 slice C (residual fold — see the train path).
            crate::source_ad::fuse_rmsnorm_dx_residual(&mut adjoint.ops, &needed);

            if !adjoint.ops.is_empty() {
                // P0.2: arm the gradient-integrity guard for the `grad` block's
                // adjoint (a live op that cannot resolve an input silently
                // drops the gradient — see #396), then disarm before the match.
                self.grad_live_results =
                    Some(crate::source_ad::reachable_result_vars(&adjoint.ops, &needed));
                let grad_block_lowered = crate::wengert_lower::compile_wengert_ops(
                    self, builder, state, &adjoint, full_vars,
                    None, // FASE on_param_grad hook — wired in Task 3
                );
                self.grad_live_results = None;
                let grad_lowered = match grad_block_lowered {
                    Ok(gv) => gv,
                    Err(e) => {
                        nsl_runtime::nsl_log!(ERROR, "nsl", 
                            "[nsl] source AD lowering failed ({}) in grad block; rerun without --source-ad",
                            e
                        );
                        return Err(e);
                    }
                };

                let mut retained_adjoint_vars = std::collections::HashSet::new();
                if let Some(grad_val) = grad_lowered.var_map.get(&target_adj_var).copied() {
                    self.compile_call_by_name(builder, "nsl_tensor_free", &[grad_tensor])?;
                    grad_tensor = grad_val;
                    retained_adjoint_vars.insert(target_adj_var);
                }
                self.free_wengert_owned_values(
                    builder,
                    &grad_lowered.owned_values,
                    &retained_adjoint_vars,
                )?;
            }
        }

        self.free_wengert_owned_values(builder, &full_lowered.owned_values, &retained_full_vars)?;
        Ok(Some((loss_tensor, grad_tensor)))
    }

    fn compile_tape_grad_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        grad: &nsl_ast::block::GradBlock,
        targets_val: Value,
    ) -> Result<(Value, Value), CodegenError> {
        // 2. Wrap single tensor in a 1-element list for the tape API
        let param_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        self.compile_call_by_name(builder, "nsl_list_push", &[param_list, targets_val])?;

        // 3. Start tape recording
        self.compile_call_by_name(builder, "nsl_tape_start", &[param_list])?;

        // 4. Compile body — all tensor ops auto-record on the global tape.
        //    The last expression is the loss (a scalar tensor).
        state.flags.in_tape_region = true;
        let mut loss_val = None;
        // The grad body is checker-scoped (check_block ScopeKind::Block)
        // but compiles in the SAME FuncState with no variables restore —
        // a nested fn declared here must not stay a live fn-binding
        // after the block (review MEDIUM on 682641ca: a post-block call
        // the checker resolved to the builtin rerouted into the grad
        // body's dead nested fn — misaligned-deref abort).
        state.push_fn_binding_scope();
        for (i, stmt) in grad.body.stmts.iter().enumerate() {
            if i == grad.body.stmts.len() - 1 {
                if let StmtKind::Expr(ref expr) = stmt.kind {
                    loss_val = Some(self.compile_expr(builder, state, expr)?);
                } else {
                    self.compile_stmt(builder, state, stmt)?;
                }
            } else {
                self.compile_stmt(builder, state, stmt)?;
            }
        }
        state.pop_fn_binding_scope();
        // ELTLS: free tape-held tensors before clearing the tape flag.
        self.free_tape_held_tensors(builder, state);
        state.flags.in_tape_region = false;

        let loss_tensor = loss_val.ok_or_else(|| {
            CodegenError::new("grad block must end with an expression (the loss)")
        })?;

        // 5. Run backward pass
        let grads_list =
            self.compile_call_by_name(builder, "nsl_tape_backward", &[loss_tensor, param_list])?;

        // 6. Stop tape (cleans up saved tensor refcounts)
        self.compile_call_by_name(builder, "nsl_tape_stop", &[])?;

        // 7. Get gradient for the single param (index 0)
        let zero = builder.ins().iconst(cl_types::I64, 0);
        let grad_tensor =
            self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, zero])?;

        // 7b. Free the temporary lists (grad_tensor was extracted, still alive)
        self.compile_call_by_name(builder, "nsl_list_free", &[grads_list])?;
        self.compile_call_by_name(builder, "nsl_list_free", &[param_list])?;

        Ok((loss_tensor, grad_tensor))
    }

    fn register_source_ad_model_instances(
        &self,
        extractor: &mut crate::source_ad::WengertExtractor<'_>,
        state: &FuncState,
    ) {
        for sym in self.variables_in_name_order(state) {
            if let Some(model_type_name) = self.resolve_source_ad_model_type_name(state, sym) {
                extractor.register_model_instance(sym, &model_type_name);
            }
        }
    }

    /// The variables in scope, by name — the order for any walk over
    /// `state.variables` that emits as it goes.
    ///
    /// `state.variables` is a `HashMap`; walking it in iteration order
    /// while calling `use_var` (which numbers a value per call) or
    /// emitting a call per entry laid `main` out differently from one
    /// compile of the same program to the next — same instructions, but
    /// the value numbers and the order of the cleanup frees moved, so no
    /// two `--dump-ir` runs could be compared and the CLIF snapshot tests
    /// (`tests/train_clif_snapshots.rs`) could not exist.
    ///
    /// One interner per compile makes the name injective over symbols
    /// today; the symbol index breaks a tie should that ever change, so
    /// the order never falls back to the map's.
    pub(crate) fn variables_in_name_order(&self, state: &FuncState) -> Vec<nsl_ast::Symbol> {
        let mut syms: Vec<nsl_ast::Symbol> = state.variables.keys().copied().collect();
        syms.sort_by_key(|&sym| (self.resolve_sym(sym), string_interner::Symbol::to_usize(sym.0)));
        syms
    }

    fn resolve_source_ad_model_type_name(
        &self,
        state: &FuncState,
        sym: nsl_ast::Symbol,
    ) -> Option<String> {
        self.models
            .model_var_types
            .get(&sym)
            .cloned()
            .or_else(|| {
                state.variable_types.get(&sym).and_then(|ty| match ty {
                    Type::Model { name, .. } | Type::Struct { name, .. } => {
                        Some(self.resolve_sym(*name).to_string())
                    }
                    _ => None,
                })
            })
            .filter(|name| self.types.struct_layouts.contains_key(name))
    }

    fn resolve_source_ad_expr_name(&self, expr: &nsl_ast::expr::Expr) -> Option<String> {
        match &expr.kind {
            ExprKind::Ident(sym) => Some(self.resolve_sym(*sym).to_string()),
            ExprKind::MemberAccess { object, member } => {
                let prefix = self.resolve_source_ad_expr_name(object)?;
                let member_name = self.resolve_sym(*member).to_string();
                Some(format!("{}.{}", prefix, member_name))
            }
            _ => None,
        }
    }

    fn resolve_source_ad_expr_var_id(
        &self,
        extractor: &crate::source_ad::WengertExtractor<'_>,
        expr: &nsl_ast::expr::Expr,
        allow_last_op_fallback: bool,
    ) -> Option<crate::wengert::VarId> {
        match &expr.kind {
            ExprKind::Ident(sym) => extractor.symbol_var_map().get(sym).copied(),
            ExprKind::MemberAccess { .. } => {
                let name = self.resolve_source_ad_expr_name(expr)?;
                extractor
                    .named_param_var_ids()
                    .iter()
                    .find_map(|(compound_name, vid)| (compound_name == &name).then_some(*vid))
                    .or_else(|| {
                        extractor.wengert_list().var_names.iter().find_map(
                            |(vid, existing_name)| (existing_name == &name).then_some(*vid),
                        )
                    })
            }
            _ if allow_last_op_fallback => extractor.wengert_list().ops.last().map(|op| op.result),
            _ => None,
        }
    }

    pub(crate) fn load_source_ad_named_param(
        &self,
        builder: &mut FunctionBuilder,
        state: &FuncState,
        compound_name: &str,
    ) -> Option<Value> {
        let root_name = compound_name.split('.').next()?;
        let (&root_sym, &(root_var, _)) = state
            .variables
            .iter()
            .find(|(sym, _)| self.resolve_sym(**sym) == root_name)?;
        let model_type_name = self.resolve_source_ad_model_type_name(state, root_sym)?;
        let layout = self.types.struct_layouts.get(&model_type_name)?;
        let root_ptr = builder.use_var(root_var);
        self.load_nested_field(builder, root_ptr, layout, &model_type_name, compound_name)
    }

    // ── Quant block codegen ──────────────────────────────────────────

    pub fn compile_quant_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        quant: &nsl_ast::block::QuantBlock,
    ) -> Result<(), CodegenError> {
        // 1. Get source model variable
        let source_sym = quant.source;
        let source_val = {
            let (var, _) = state.variables.get(&source_sym).ok_or_else(|| {
                CodegenError::new(format!(
                    "undefined model variable '{}' in quant block",
                    self.resolve_sym(source_sym)
                ))
            })?;
            builder.use_var(*var)
        };

        // 2. Resolve model type name using the same strategy as train blocks:
        //    scan the type_map for a Model type with a known struct layout.
        let model_type_name = {
            let mut found_name = None;
            for (_node_id, ty) in self.type_map.iter() {
                match ty {
                    nsl_semantic::types::Type::Model { name, .. } => {
                        let n = self.resolve_sym(*name).to_string();
                        if self.types.struct_layouts.contains_key(&n) {
                            found_name = Some(n);
                            break;
                        }
                    }
                    nsl_semantic::types::Type::Struct { name, .. } => {
                        let n = self.resolve_sym(*name).to_string();
                        if self.types.struct_layouts.contains_key(&n) {
                            found_name = Some(n);
                        }
                    }
                    _ => {}
                }
            }
            found_name.unwrap_or_else(|| self.resolve_sym(source_sym).to_string())
        };

        let layout = self
            .types
            .struct_layouts
            .get(&model_type_name)
            .cloned()
            .ok_or_else(|| {
                CodegenError::new(format!(
                    "no struct layout found for model '{}' in quant block",
                    model_type_name
                ))
            })?;

        // 3. Compute dtype/granularity integer codes for the runtime call
        let dtype_code: i64 = match quant.default_dtype {
            Some(QuantDtype::Int4) => 1,
            Some(QuantDtype::Awq4) => 2,
            Some(QuantDtype::Gptq4) => 3,
            Some(QuantDtype::Gptq8) => 4,
            Some(QuantDtype::Int8) | None => 0,
        };
        let (gran_code, axis_val, gs_val): (i64, i64, i64) = match &quant.default_granularity {
            Some(QuantGranularity::PerChannel(a)) => (1, *a, 0),
            Some(QuantGranularity::PerGroup(a, gs)) => (2, *a, *gs),
            Some(QuantGranularity::PerTensor) | None => (0, 0, 0),
        };

        let dtype_v = builder.ins().iconst(cl_types::I64, dtype_code);
        let gran_v = builder.ins().iconst(cl_types::I64, gran_code);
        let axis_v = builder.ins().iconst(cl_types::I64, axis_val);
        let gs_v = builder.ins().iconst(cl_types::I64, gs_val);

        // 4. Allocate a new struct with the same layout as the source model
        let alloc_size = builder
            .ins()
            .iconst(cl_types::I64, layout.total_size.max(8) as i64);
        let new_ptr = self.compile_call_by_name(builder, "nsl_alloc", &[alloc_size])?;

        // 4b. If this is an AWQ quant block and a calibration sidecar is present,
        //     decode the AWQ activation scales once.  We'll use them per-field below.
        //
        //     Key: "awq_activation_scales" in sidecar.hooks (binary blob).
        //     Projection path format: "{model_type_name}.{field_name}" — same
        //     cache key as Task 8's discovery pass.
        //
        //     Hard error when sidecar present but projection missing:
        //     silent fallback to uncalibrated is a correctness trap.
        let is_awq = matches!(quant.default_dtype, Some(QuantDtype::Awq4));
        let awq_scales_opt: Option<nsl_runtime::awq::AwqScales> = if is_awq {
            match self.compile_options.calibration.sidecar.as_ref() {
                None => None,
                Some(sidecar) => {
                    match sidecar.hooks.get("awq_activation_scales") {
                        None => None, // Sidecar present but no AWQ hook blob → treat as uncalibrated.
                        Some(blob) => {
                            match nsl_runtime::awq::AwqScales::from_blob(blob) {
                                Ok(scales) => Some(scales),
                                Err(e) => {
                                    // Blob present but malformed → hard error.
                                    return Err(CodegenError::new(format!(
                                        "AWQ calibration sidecar blob is malformed: {e}"
                                    )));
                                }
                            }
                        }
                    }
                }
            }
        } else {
            None
        };

        // AWQ calibration alpha (matches awq_quantize_with_scales default).
        let awq_alpha: f64 = 0.5;

        // 5. For each field: quantize→dequantize (or clone if excluded)
        for field in &layout.fields {
            let is_excluded = quant.exclude.iter().any(|pat| glob_match(pat, &field.name));
            let src_val = builder.ins().load(
                field.cl_type,
                MemFlagsData::trusted(),
                source_val,
                field.offset as i32,
            );

            if is_excluded {
                // Copy as-is via clone (bumps refcount internally)
                let cloned = self.compile_call_by_name(builder, "nsl_tensor_clone", &[src_val])?;
                builder
                    .ins()
                    .store(MemFlagsData::trusted(), cloned, new_ptr, field.offset as i32);
            } else {
                // For AWQ with a calibration sidecar, pre-scale the weight tensor using
                // the per-input-channel activation statistics before quantizing.
                // This embeds the scale data as compile-time constants in the object file
                // and calls nsl_awq_pre_scale_weight at runtime to apply them.
                let weight_for_quantize: Value = if is_awq {
                    match awq_scales_opt.as_ref() {
                        None => {
                            // No sidecar → uncalibrated, pass weight through unchanged.
                            src_val
                        }
                        Some(scales_map) => {
                            // Sidecar present — projection MUST have scales.
                            let projection_path =
                                format!("{}.{}", model_type_name, field.name);
                            let field_scales = scales_map
                                .by_projection
                                .get(&projection_path)
                                .ok_or_else(|| {
                                    CodegenError::missing_scales(&projection_path)
                                })?;

                            // Embed scale data as a compile-time constant in .rodata.
                            let data_label = format!(
                                "__nsl_awq_scales_{}_{}",
                                model_type_name, field.name
                            );
                            let scale_bytes: Vec<u8> = field_scales
                                .iter()
                                .flat_map(|v: &f32| v.to_le_bytes())
                                .collect();
                            let scale_data_id = self
                                .module
                                .declare_data(
                                    &data_label,
                                    cranelift_module::Linkage::Local,
                                    false,
                                    false,
                                )
                                .map_err(|e| {
                                    CodegenError::new(format!(
                                        "failed to declare AWQ scale data for \
                                         '{projection_path}': {e}"
                                    ))
                                })?;
                            let mut data_desc = cranelift_module::DataDescription::new();
                            data_desc.define(scale_bytes.into_boxed_slice());
                            self.module
                                .define_data(scale_data_id, &data_desc)
                                .map_err(|e| {
                                    CodegenError::new(format!(
                                        "failed to define AWQ scale data for \
                                         '{projection_path}': {e}"
                                    ))
                                })?;

                            // Get a pointer to the scale data in this function.
                            let scale_gv = self
                                .module
                                .declare_data_in_func(scale_data_id, builder.func);
                            let scales_ptr =
                                builder.ins().symbol_value(cl_types::I64, scale_gv);
                            let scales_len = builder
                                .ins()
                                .iconst(cl_types::I64, field_scales.len() as i64);
                            let alpha_v = builder.ins().f64const(awq_alpha);

                            // Apply calibration scaling: returns a new NslTensor.
                            self.compile_call_by_name(
                                builder,
                                "nsl_awq_pre_scale_weight",
                                &[src_val, scales_ptr, scales_len, alpha_v],
                            )?
                        }
                    }
                } else {
                    src_val
                };

                // Quantize then immediately dequantize — validates the roundtrip and
                // shows quantization effects (precision loss) while storing a regular
                // NslTensor that the original forward method can consume directly.
                let qt = self.compile_call_by_name(
                    builder,
                    "nsl_qtensor_quantize",
                    &[weight_for_quantize, dtype_v, gran_v, axis_v, gs_v],
                )?;
                let deq = self.compile_call_by_name(builder, "nsl_qtensor_dequantize", &[qt])?;
                // Release the intermediate QuantizedTensor (refcount-aware)
                self.compile_call_by_name(builder, "nsl_qtensor_release", &[qt])?;
                // If we pre-scaled the weight, release the intermediate scaled tensor too.
                if is_awq && awq_scales_opt.is_some() {
                    self.compile_call_by_name(
                        builder,
                        "nsl_tensor_release",
                        &[weight_for_quantize],
                    )?;
                }
                builder
                    .ins()
                    .store(MemFlagsData::trusted(), deq, new_ptr, field.offset as i32);
            }
        }

        // 6. Register the quantized model with the same struct layout and methods
        //    so that forward dispatch works identically to the source model.
        let quant_name = self.resolve_sym(quant.name).to_string();
        if !self.types.struct_layouts.contains_key(&quant_name) {
            self.types.struct_layouts.insert(quant_name.clone(), layout);
        }
        if let Some(methods) = self.models.model_methods.get(&model_type_name).cloned() {
            self.models.model_methods.insert(quant_name, methods);
        }

        // 7. Bind the new struct pointer as the output variable
        let var = builder.declare_var(cl_types::I64);
        builder.def_var(var, new_ptr);
        state.variables.insert(quant.name, (var, cl_types::I64));

        Ok(())
    }

    /// Walk the compiled model's `quant { ... }` blocks and produce the
    /// list of ProjectionRefs that AWQ needs calibration data for.
    /// Returns `None` when no AWQ quant block is present or discovery
    /// produces no matches.
    ///
    /// Implementation (Task 3): scans `self.features.quant_configs` for
    /// models quantised with `"awq4"`.  For each such model, retrieves the
    /// `forward` method body from `model_method_bodies`, walks its pipe chain
    /// to enumerate linear-projection call sites, and returns the sorted,
    /// deduplicated `Vec<ProjectionRef>`.  Discovery errors (e.g. empty match)
    /// are logged to stderr and treated as `None` so the harness falls back to
    /// its no-op path rather than crashing the compile.
    pub(crate) fn discover_awq_projections(
        &self,
    ) -> Option<Vec<crate::calibration::DiscoveredProjection>> {
        use crate::calibration::discover_awq_projections_from_state;

        // Collect all AWQ-quantised model names.
        let awq_models: Vec<String> = self
            .features
            .quant_configs
            .iter()
            .filter(|(_, cfg)| cfg.dtype == "awq4")
            .map(|(name, _)| name.clone())
            .collect();

        if awq_models.is_empty() {
            return None;
        }

        let mut all_projections: Vec<crate::calibration::DiscoveredProjection> = Vec::new();

        for model_name in &awq_models {
            // Retrieve the forward method body (if stored).
            let forward_body: Option<&nsl_ast::stmt::Block> = self
                .models
                .model_method_bodies
                .get(model_name)
                .and_then(|methods| methods.get("forward"))
                .map(|fn_def| &fn_def.body);

            // Retrieve field-type and shape maps for this model.
            let empty_field_types = std::collections::HashMap::new();
            let field_types = self
                .models
                .model_field_types
                .get(model_name)
                .unwrap_or(&empty_field_types);

            let empty_shapes = std::collections::HashMap::new();
            let tensor_shapes = self
                .models
                .model_tensor_field_shapes
                .get(model_name)
                .unwrap_or(&empty_shapes);

            match discover_awq_projections_from_state(
                model_name,
                forward_body,
                field_types,
                tensor_shapes,
                &[], // no exclusions from the Compiler-level stub; the QuantBlock's
                     // exclude list is stored in the AST which isn't retained here.
                self.interner,
            ) {
                Ok(discovered) => {
                    for dp in discovered {
                        all_projections.push(dp);
                    }
                }
                Err(e) => {
                    nsl_runtime::nsl_log!(INFO, "calibration", "[calibration] AWQ discovery for model '{model_name}': {e}");
                }
            }
        }

        if all_projections.is_empty() {
            None
        } else {
            // Sort + dedup across models (by qualified path for determinism).
            all_projections.sort_by(|a, b| a.projection.0.cmp(&b.projection.0));
            all_projections.dedup_by(|a, b| a.projection.0 == b.projection.0);
            Some(all_projections)
        }
    }

    /// Dev Tools Phase 5 Task 7: emit IR for one `@inspect(target, every=?, condition=?)`
    /// decorator attached to a `let` binding.
    ///
    /// Ship-first scope:
    ///   * Only fires inside a train block (requires `inspect_train_step_var`).
    ///     Outside train scope, emits nothing.
    ///   * `every=N` → step-gated call to `nsl_tensor_stats` +
    ///     `nsl_inspect_record_stats`.
    ///   * `condition="..."` → predicate-gated `nsl_inspect_dump_full`.
    ///     Predicate AST is lowered via `inspect::predicate::lower_predicate`.
    ///   * The `loss` identifier reads the most recent recorded loss at
    ///     runtime via `nsl_health_get_last_loss` (recorded per step whenever
    ///     `--inspect` or the health monitor is on). Because @inspect fires at
    ///     the let-binding site — before the current step's loss compute — the
    ///     predicate sees the previous completed step's loss (0.0 on step 0).
    ///
    /// All emission gated on `compile_options.dev_tools.inspect_enabled`.  When that
    /// flag is off, this method is never called.
    fn emit_inspect_hook(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        decorator: &nsl_ast::decl::Decorator,
        target_sym: nsl_ast::Symbol,
    ) -> Result<(), CodegenError> {
        // Outside a train block we skip entirely for Phase 5 ship-first.
        let step_count_var = match self.inspect_train_step_var {
            Some(v) => v,
            None => return Ok(()),
        };

        // Resolve target tensor: prefer decorator arg[0] (which semantic
        // guarantees is a positional Ident), fall back to the let binding's
        // own LHS symbol when argument extraction fails.
        let (resolved_sym, tensor_name) = {
            let mut s = target_sym;
            if let Some(args) = &decorator.args
                && let Some(first) = args.first()
                && first.name.is_none()
                && let ExprKind::Ident(sym) = &first.value.kind
            {
                s = *sym;
            }
            let name = self.resolve_sym(s).to_string();
            (s, name)
        };
        let tensor_val = match state.variables.get(&resolved_sym) {
            Some((var, _)) => builder.use_var(*var),
            None => return Ok(()),
        };

        // Extract every=N and condition="..." from decorator args.
        let mut every_n: Option<i64> = None;
        let mut cond_str: Option<String> = None;
        if let Some(args) = &decorator.args {
            // args[0] is the positional tensor target — resolved above.
            for arg in args.iter().skip(1) {
                let kw = arg.name.map(|s| self.resolve_sym(s).to_string());
                match kw.as_deref() {
                    Some("every") => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind
                            && *n > 0
                        {
                            every_n = Some(*n);
                        }
                    }
                    Some("condition") => {
                        if let ExprKind::StringLiteral(s) = &arg.value.kind {
                            cond_str = Some(s.clone());
                        }
                    }
                    _ => {}
                }
            }
        }

        // Intern the tensor name once — shared by stats + dump branches.
        let name_data_id = self.intern_string(&tensor_name)?;
        let name_gv = self
            .module
            .declare_data_in_func(name_data_id, builder.func);

        // ── (a) Stats branch: every=N ────────────────────────────────────
        if let Some(n) = every_n {
            let zero_i64 = builder.ins().iconst(cl_types::I64, 0);
            let step_loaded = builder.use_var(step_count_var);
            let n_val = builder.ins().iconst(cl_types::I64, n);
            let rem = builder.ins().srem(step_loaded, n_val);
            let due = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::Equal,
                rem,
                zero_i64,
            );

            let do_block = builder.create_block();
            let after_block = builder.create_block();
            builder.ins().brif(due, do_block, &[], after_block, &[]);

            builder.switch_to_block(do_block);
            builder.seal_block(do_block);
            state.current_block = Some(do_block);

            // Allocate a 48-byte 8-aligned stack slot for the stats struct
            // (matches the runtime's NslTensorStats layout — 6 × f64).
            let slot = builder.create_sized_stack_slot(
                cranelift_codegen::ir::StackSlotData::new(
                    cranelift_codegen::ir::StackSlotKind::ExplicitSlot,
                    48,
                    3,
                ),
            );
            let stats_ptr = builder.ins().stack_addr(cl_types::I64, slot, 0);

            self.compile_call_by_name(
                builder,
                "nsl_tensor_stats",
                &[tensor_val, stats_ptr],
            )?;

            let name_ptr = builder.ins().symbol_value(cl_types::I64, name_gv);
            let name_len = builder
                .ins()
                .iconst(cl_types::I64, tensor_name.len() as i64);
            let step_now = builder.use_var(step_count_var);
            self.compile_call_by_name(
                builder,
                "nsl_inspect_record_stats",
                &[stats_ptr, step_now, name_ptr, name_len],
            )?;

            builder.ins().jump(after_block, &[]);
            builder.switch_to_block(after_block);
            builder.seal_block(after_block);
            state.current_block = Some(after_block);
        }

        // ── (b) Dump branch: condition="..." ──────────────────────────────
        if let Some(cond_src) = cond_str {
            let ast = match crate::inspect::predicate::parse_predicate(&cond_src) {
                Ok(p) => p,
                Err(e) => {
                    nsl_runtime::nsl_log!(ERROR, "codegen", 
                        "[@inspect] predicate parse failed for {:?}: {}",
                        cond_src, e
                    );
                    return Ok(());
                }
            };

            // Resolve FuncRefs for all health getters.  Any missing symbol
            // means builtins.rs / Phase 4+5 runtime didn't register — bail.
            let (lema_id, _) = match self.registry.runtime_fns.get("nsl_health_get_loss_ema") {
                Some(e) => e.clone(),
                None => return Ok(()),
            };
            let get_loss_ema_ref =
                self.module.declare_func_in_func(lema_id, builder.func);
            let (lslope_id, _) = match self
                .registry
                .runtime_fns
                .get("nsl_health_get_loss_ema_slope")
            {
                Some(e) => e.clone(),
                None => return Ok(()),
            };
            let get_loss_ema_slope_ref =
                self.module.declare_func_in_func(lslope_id, builder.func);
            let (gnt_id, _) = match self
                .registry
                .runtime_fns
                .get("nsl_health_get_grad_norm_total")
            {
                Some(e) => e.clone(),
                None => return Ok(()),
            };
            let get_grad_norm_total_ref =
                self.module.declare_func_in_func(gnt_id, builder.func);
            let (nic_id, _) = match self
                .registry
                .runtime_fns
                .get("nsl_health_get_nan_inf_count_window")
            {
                Some(e) => e.clone(),
                None => return Ok(()),
            };
            let get_nan_inf_count_window_ref =
                self.module.declare_func_in_func(nic_id, builder.func);

            let (lloss_id, _) = match self.registry.runtime_fns.get("nsl_health_get_last_loss") {
                Some(e) => e.clone(),
                None => return Ok(()),
            };
            let get_last_loss_ref =
                self.module.declare_func_in_func(lloss_id, builder.func);

            let step_loaded = builder.use_var(step_count_var);

            let ctx = crate::inspect::predicate::PredicateLowerCtx {
                step_val: step_loaded,
                get_last_loss_ref,
                get_loss_ema_ref,
                get_loss_ema_slope_ref,
                get_grad_norm_total_ref,
                get_nan_inf_count_window_ref,
            };
            let pred_val =
                crate::inspect::predicate::lower_predicate(&ast, builder, &ctx);

            let do_block = builder.create_block();
            let after_block = builder.create_block();
            builder.ins().brif(pred_val, do_block, &[], after_block, &[]);

            builder.switch_to_block(do_block);
            builder.seal_block(do_block);
            state.current_block = Some(do_block);

            let name_ptr = builder.ins().symbol_value(cl_types::I64, name_gv);
            let name_len = builder
                .ins()
                .iconst(cl_types::I64, tensor_name.len() as i64);
            let step_now = builder.use_var(step_count_var);
            self.compile_call_by_name(
                builder,
                "nsl_inspect_dump_full",
                &[tensor_val, step_now, name_ptr, name_len],
            )?;

            builder.ins().jump(after_block, &[]);
            builder.switch_to_block(after_block);
            builder.seal_block(after_block);
            state.current_block = Some(after_block);
        }

        Ok(())
    }
}

/// Simple glob matching supporting `*` (any sequence) and `?` (single char) wildcards.
fn glob_match(pattern: &str, text: &str) -> bool {
    let pb = pattern.as_bytes();
    let tb = text.as_bytes();
    let mut pi = 0usize;
    let mut ti = 0usize;
    let mut star_pi = usize::MAX;
    let mut star_ti = 0usize;

    while ti < tb.len() {
        if pi < pb.len() && (pb[pi] == b'?' || pb[pi] == tb[ti]) {
            pi += 1;
            ti += 1;
        } else if pi < pb.len() && pb[pi] == b'*' {
            star_pi = pi;
            star_ti = ti;
            pi += 1;
        } else if star_pi != usize::MAX {
            pi = star_pi + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }
    while pi < pb.len() && pb[pi] == b'*' {
        pi += 1;
    }
    pi == pb.len()
}

/// Derive [`crate::cpdt_optim::AdamWHyperparams`] from a `@train` block's
/// optimizer section — via the SAME resolver the train lowering consumes,
/// so this can no longer drift from the main parse (pre-contract it was an
/// independent third parse: first-section-wins where the lowering was
/// last-wins, int-tolerant where the lowering was float-only, and unknown
/// kwargs silently ignored).
///
/// Returns library defaults when:
/// - `train` is `None`
/// - the sections fail the optimizer contract (the train lowering itself
///   is the refusal site — CPDT output never outlives a refused compile)
/// - the optimizer is not `AdamW`
///
/// `lr` and `weight_decay` are intentionally NOT read here — CPDT's
/// hyperparams cover only the running-moment constants.
///
/// Takes the `Interner` directly so this helper stays free-standing (callable
/// from `invoke_cpdt_if_enabled` via `&compiler.interner`) and is
/// straightforward to unit-test with a local `Interner`. Missing symbols
/// resolve to `"<unknown>"` to mirror `Compiler::resolve_sym`.
pub(crate) fn adamw_from_train_block(
    train: Option<&nsl_ast::block::TrainBlock>,
    interner: &nsl_lexer::Interner,
) -> crate::cpdt_optim::AdamWHyperparams {
    let mut hp = crate::cpdt_optim::AdamWHyperparams::default();

    let Some(train) = train else {
        return hp;
    };

    let Ok(cfg) = nsl_semantic::optim_config::resolve_optim_config(
        &train.sections,
        train.span,
        &|sym| interner.resolve(sym.0).unwrap_or("<unknown>").to_string(),
        nsl_semantic::train_config::TrainConfigPurpose::UserTrainBlock,
    ) else {
        return hp;
    };
    if cfg.optimizer.kind != nsl_semantic::optim_config::OptimizerKind::AdamW {
        return hp;
    }

    hp.beta1 = cfg.optimizer.beta1;
    hp.beta2 = cfg.optimizer.beta2;
    hp.eps = cfg.optimizer.eps;
    hp
}

#[cfg(test)]
mod tests {
    use super::{
        adamw_from_train_block, classify_source_ad_param_name, is_trainable_param_leaf_name,
        SourceAdParamDiagnosticKind,
    };
    use std::collections::HashSet;

    #[test]
    fn source_ad_param_classification_separates_tensor_and_non_tensor_noise() {
        let tensor_paths: HashSet<String> = [
            "m.blocks.0.attn.wq".to_string(),
            "m.blocks.0.attn._dropout_p".to_string(),
            "m.blocks.0.attn.rope.inv_freq".to_string(),
        ]
        .into_iter()
        .collect();

        assert!(is_trainable_param_leaf_name("m.blocks.0.attn.wq"));
        assert!(!is_trainable_param_leaf_name("m.blocks.0.attn._dropout_p"));
        assert!(!is_trainable_param_leaf_name(
            "m.blocks.0.attn.rope.inv_freq"
        ));

        assert_eq!(
            classify_source_ad_param_name("m.blocks.0.attn.wq", &tensor_paths),
            SourceAdParamDiagnosticKind::Trainable,
        );
        assert_eq!(
            classify_source_ad_param_name("m.blocks.0.attn._dropout_p", &tensor_paths),
            SourceAdParamDiagnosticKind::IgnoredConfig,
        );
        assert_eq!(
            classify_source_ad_param_name("m.blocks.0.attn.rope.inv_freq", &tensor_paths),
            SourceAdParamDiagnosticKind::IgnoredConfig,
        );
        assert_eq!(
            classify_source_ad_param_name("m.blocks.0.attn_norm.eps", &tensor_paths),
            SourceAdParamDiagnosticKind::IgnoredNonTensor,
        );
    }

    // ── Task 2: adamw_from_train_block helper ───────────────────────────
    //
    // Build small TrainBlock fixtures directly (simpler than running the
    // parser + semantic passes just to get AST shape we control).
    use nsl_ast::block::{TrainBlock, TrainSection};
    use nsl_ast::expr::{Arg, Expr, ExprKind};
    use nsl_ast::{NodeId, Span, Symbol};
    use nsl_lexer::Interner;

    fn mk_expr(kind: ExprKind) -> Expr {
        Expr {
            kind,
            span: Span::dummy(),
            id: NodeId::next(),
        }
    }

    fn mk_arg(name: Option<Symbol>, value: Expr) -> Arg {
        Arg {
            name,
            value,
            span: Span::dummy(),
        }
    }

    #[test]
    fn adamw_hyperparams_default_when_no_train_block() {
        let interner: Interner = Interner::new();
        let hp = adamw_from_train_block(None, &interner);
        let d = crate::cpdt_optim::AdamWHyperparams::default();
        assert!((hp.beta1 - d.beta1).abs() < 1e-12);
        assert!((hp.beta2 - d.beta2).abs() < 1e-12);
        assert!((hp.eps - d.eps).abs() < 1e-12);
    }

    #[test]
    fn adamw_hyperparams_derived_from_train_block() {
        let mut interner: Interner = Interner::new();
        let adamw_sym = Symbol(interner.get_or_intern("AdamW"));
        let beta1_sym = Symbol(interner.get_or_intern("beta1"));
        let beta2_sym = Symbol(interner.get_or_intern("beta2"));

        // optimizer = AdamW(beta1=0.85, beta2=0.99)
        let callee = Box::new(mk_expr(ExprKind::Ident(adamw_sym)));
        let args = vec![
            mk_arg(Some(beta1_sym), mk_expr(ExprKind::FloatLiteral(0.85))),
            mk_arg(Some(beta2_sym), mk_expr(ExprKind::FloatLiteral(0.99))),
        ];
        let opt_call = mk_expr(ExprKind::Call { callee, args });

        let train = TrainBlock {
            config: vec![],
            sections: vec![TrainSection::Optimizer(opt_call)],
            span: Span::dummy(),
        };

        let hp = adamw_from_train_block(Some(&train), &interner);
        let d = crate::cpdt_optim::AdamWHyperparams::default();
        assert!((hp.beta1 - 0.85).abs() < 1e-12, "beta1 = {}", hp.beta1);
        assert!((hp.beta2 - 0.99).abs() < 1e-12, "beta2 = {}", hp.beta2);
        // eps was not overridden — should stay at library default.
        assert!((hp.eps - d.eps).abs() < 1e-12, "eps = {}", hp.eps);
    }

    #[test]
    fn adamw_hyperparams_default_when_block_fails_the_contract() {
        // AdamW(beta1=0.85, lrr=0.01): the typo'd kwarg fails
        // resolve_optim_config, so CPDT sees library defaults — never the
        // half-parsed beta1. (The train lowering itself refuses the block,
        // so those defaults cannot train anything; pre-contract this
        // helper would have silently returned beta1=0.85 while the typo
        // trained at the default lr.)
        let mut interner: Interner = Interner::new();
        let adamw_sym = Symbol(interner.get_or_intern("AdamW"));
        let beta1_sym = Symbol(interner.get_or_intern("beta1"));
        let lrr_sym = Symbol(interner.get_or_intern("lrr"));

        let callee = Box::new(mk_expr(ExprKind::Ident(adamw_sym)));
        let args = vec![
            mk_arg(Some(beta1_sym), mk_expr(ExprKind::FloatLiteral(0.85))),
            mk_arg(Some(lrr_sym), mk_expr(ExprKind::FloatLiteral(0.01))),
        ];
        let opt_call = mk_expr(ExprKind::Call { callee, args });

        let train = TrainBlock {
            config: vec![],
            sections: vec![TrainSection::Optimizer(opt_call)],
            span: Span::dummy(),
        };

        let hp = adamw_from_train_block(Some(&train), &interner);
        let d = crate::cpdt_optim::AdamWHyperparams::default();
        assert!(
            (hp.beta1 - d.beta1).abs() < 1e-12,
            "contract-refused block must not leak half-parsed values"
        );
    }

    #[test]
    fn adamw_hyperparams_falls_back_for_non_adamw_optimizer() {
        // SGD(momentum=0.9) should yield library defaults — no silent β1 override.
        let mut interner: Interner = Interner::new();
        let sgd_sym = Symbol(interner.get_or_intern("SGD"));
        let momentum_sym = Symbol(interner.get_or_intern("momentum"));

        let callee = Box::new(mk_expr(ExprKind::Ident(sgd_sym)));
        let args = vec![mk_arg(
            Some(momentum_sym),
            mk_expr(ExprKind::FloatLiteral(0.9)),
        )];
        let opt_call = mk_expr(ExprKind::Call { callee, args });

        let train = TrainBlock {
            config: vec![],
            sections: vec![TrainSection::Optimizer(opt_call)],
            span: Span::dummy(),
        };

        let hp = adamw_from_train_block(Some(&train), &interner);
        let d = crate::cpdt_optim::AdamWHyperparams::default();
        assert!((hp.beta1 - d.beta1).abs() < 1e-12);
        assert!((hp.beta2 - d.beta2).abs() < 1e-12);
        assert!((hp.eps - d.eps).abs() < 1e-12);
    }
}
