use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{BlockArg, InstBuilder, MemFlagsData};
use cranelift_frontend::{FunctionBuilder, Variable};
use cranelift_module::Module;

use nsl_ast::block::{QuantDtype, QuantGranularity, TrainSection};
use nsl_ast::expr::{ExprKind, SubscriptKind};
use nsl_ast::operator::AssignOp;
use nsl_ast::pattern::PatternKind;
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_semantic::types::Type;

use crate::compiler::Compiler;
use crate::context::{FuncState, LoopContext};
use crate::error::CodegenError;
use crate::stmt_train::csla_window::{
    CslaParam, CslaPending, CslaPre, CslaSaveInputs, CslaSchedule, CslaWindowInputs,
    CslaWindowSave,
};
use crate::stmt_train::model_params::ModelParams;
use crate::stmt_train::ccr_adjoint_frees::CcrAdjointFreesInputs;
use crate::stmt_train::fase_hook_lowering::FaseHookLoweringInputs;
use crate::stmt_train::primal_vars::PrimalVarsInputs;
use crate::stmt_train::source_ad_grads::SourceAdGradsInputs;
use crate::stmt_train::optimizer_state::OptimizerState;
use crate::stmt_train::optimizer_step::OptimizerStepInputs;
use crate::stmt_train::config::TrainConfigSection;
use crate::stmt_train::contract::TrainContract;
use crate::stmt_train::epoch_close::{emit_epoch_close, EpochClose};
use crate::stmt_train::teardown::{emit_train_teardown, TrainTeardown};
use crate::types::{is_block_filled, is_float_type, nsl_type_to_cl};
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
fn cpdt_forced_stale_plan() -> bool {
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
    model.activation_checkpointing = !compiler.compile_options.training_reference
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
fn parse_layer_idx_for_health(path: &str) -> u32 {
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
    fn collect_pattern_bound_symbols(
        &self,
        pattern: &nsl_ast::pattern::Pattern,
        targets: &mut std::collections::HashSet<nsl_ast::Symbol>,
    ) {
        match &pattern.kind {
            PatternKind::Ident(sym) => {
                targets.insert(*sym);
            }
            PatternKind::Tuple(items)
            | PatternKind::List(items)
            | PatternKind::Or(items)
            | PatternKind::Constructor { args: items, .. } => {
                for item in items {
                    self.collect_pattern_bound_symbols(item, targets);
                }
            }
            PatternKind::Struct { fields, rest } => {
                for field in fields {
                    if let Some(pattern) = &field.pattern {
                        self.collect_pattern_bound_symbols(pattern, targets);
                    } else {
                        targets.insert(field.name);
                    }
                }
                if let Some(rest_sym) = rest {
                    targets.insert(*rest_sym);
                }
            }
            PatternKind::Guarded { pattern, .. } | PatternKind::Typed { pattern, .. } => {
                self.collect_pattern_bound_symbols(pattern, targets);
            }
            PatternKind::Rest(Some(sym)) => {
                targets.insert(*sym);
            }
            PatternKind::Wildcard
            | PatternKind::Literal(_)
            | PatternKind::Rest(None) => {}
        }
    }

    fn collect_assignment_targets_from_block(
        &self,
        block: &nsl_ast::stmt::Block,
        targets: &mut std::collections::HashSet<nsl_ast::Symbol>,
    ) {
        for stmt in &block.stmts {
            self.collect_assignment_targets_from_stmt(stmt, targets);
        }
    }

    fn collect_assignment_targets_from_stmt(
        &self,
        stmt: &Stmt,
        targets: &mut std::collections::HashSet<nsl_ast::Symbol>,
    ) {
        match &stmt.kind {
            StmtKind::VarDecl { pattern, .. } => {
                self.collect_pattern_bound_symbols(pattern, targets);
            }
            StmtKind::Assign { target, .. } => {
                if let ExprKind::Ident(sym) = &target.kind {
                    targets.insert(*sym);
                }
            }
            StmtKind::If {
                then_block,
                elif_clauses,
                else_block,
                ..
            } => {
                self.collect_assignment_targets_from_block(then_block, targets);
                for (_, block) in elif_clauses {
                    self.collect_assignment_targets_from_block(block, targets);
                }
                if let Some(block) = else_block {
                    self.collect_assignment_targets_from_block(block, targets);
                }
            }
            StmtKind::For { pattern, body, .. } => {
                self.collect_pattern_bound_symbols(pattern, targets);
                self.collect_assignment_targets_from_block(body, targets);
            }
            StmtKind::While { body, .. } => {
                self.collect_assignment_targets_from_block(body, targets);
            }
            StmtKind::WhileLet { pattern, body, .. } => {
                self.collect_pattern_bound_symbols(pattern, targets);
                self.collect_assignment_targets_from_block(body, targets);
            }
            StmtKind::Match { arms, .. } => {
                for arm in arms {
                    self.collect_assignment_targets_from_block(&arm.body, targets);
                }
            }
            StmtKind::Decorated { stmt, .. } => {
                self.collect_assignment_targets_from_stmt(stmt, targets);
            }
            _ => {}
        }
    }

    fn materialize_non_owning_aliases_before_if(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        then_block: &nsl_ast::stmt::Block,
        elif_clauses: &[(nsl_ast::expr::Expr, nsl_ast::stmt::Block)],
        else_block: &Option<nsl_ast::stmt::Block>,
    ) -> Result<(), CodegenError> {
        let mut assigned_symbols = std::collections::HashSet::new();
        self.collect_assignment_targets_from_block(then_block, &mut assigned_symbols);
        for (_, block) in elif_clauses {
            self.collect_assignment_targets_from_block(block, &mut assigned_symbols);
        }
        if let Some(block) = else_block {
            self.collect_assignment_targets_from_block(block, &mut assigned_symbols);
        }

        let materialize: Vec<_> = assigned_symbols
            .into_iter()
            .filter(|sym| state.non_owning_symbols.contains(sym))
            .filter_map(|sym| {
                let is_tensor = state
                    .variable_types
                    .get(&sym)
                    .map(|ty| ty.is_tensor())
                    .unwrap_or(false);
                if !is_tensor {
                    return None;
                }
                state.variables.get(&sym).and_then(|(var, cl_type)| {
                    (*cl_type == cl_types::I64).then_some((sym, *var))
                })
            })
            .collect();

        for (sym, var) in materialize {
            let current_val = builder.use_var(var);
            let cloned = self.compile_call_by_name(builder, "nsl_tensor_clone", &[current_val])?;
            builder.def_var(var, cloned);
            state.non_owning_symbols.remove(&sym);
        }

        Ok(())
    }

    /// Loop twin of `materialize_non_owning_aliases_before_if`.
    ///
    /// `state.non_owning_symbols` is flow-INSENSITIVE, but a loop body is
    /// generated exactly once. So the compile-time state seen at the body's
    /// single `eltls_clear_old_slot` site is the FIRST-iteration state. A
    /// local seeded from a borrow — `let h = x` where `x` is a parameter or a
    /// model field — is non-owning at that moment, the rebind free is skipped,
    /// and because the site is only emitted once it is skipped for EVERY
    /// iteration. `h = block.forward(h)` then strands one owned activation per
    /// iteration, forever. The same veto at `emit_return_local_sweep` strands
    /// the last one too.
    ///
    /// Measured on `main` before this fix (Coder-50M, `[2,1024]`, RTX 5070 Ti):
    /// **+1.81 GB retained per forward**, ~40 transient segments per forward,
    /// OOM by the 8th call — and `@no_grad` did not change a single byte,
    /// because the leak is ownership bookkeeping, not tape retention.
    ///
    /// Fix: before entering the loop, give each such alias its own reference
    /// (`nsl_tensor_retain`, O(1) — a refcount bump, NOT a data copy) and drop
    /// it from `non_owning_symbols`. From the loop's point of view the symbol
    /// is now an ordinary owned local: the first rebind's `free_if_valid`
    /// releases the reference we just took — the lender's own reference keeps
    /// the storage alive — and every later rebind frees that iteration's
    /// value. If the loop body never runs, the return sweep releases it.
    ///
    /// Conservative guard: only materialize when EVERY binding of the symbol
    /// inside the body is owning (`sym_bindings_all_owning_in_block`, the same
    /// predicate that arms the loop-let predeclare). A body that sometimes
    /// rebinds the slot to another borrow cannot be handled by a single
    /// statically-placed free, so those are left alone — leaking, but sound.
    fn materialize_non_owning_aliases_before_loop(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        body: &nsl_ast::stmt::Block,
        loop_pattern: Option<&nsl_ast::pattern::Pattern>,
    ) -> Result<(), CodegenError> {
        // The sweep only runs where a release can actually pair with the
        // retain. `emit_return_local_sweep` is skipped inside dtype methods
        // and tape regions, so materializing there would leak the refcount we
        // are about to take whenever the loop body runs zero times.
        if state.flags.in_dtype_method || state.flags.in_tape_region {
            return Ok(());
        }

        let mut assigned_symbols = std::collections::HashSet::new();
        self.collect_assignment_targets_from_block(body, &mut assigned_symbols);

        let materialize: Vec<_> = assigned_symbols
            .into_iter()
            // Only aliases the veto currently disarms.
            .filter(|sym| state.non_owning_symbols.contains(sym))
            // Never a parameter: the caller owns it, `eltls_clear_old_slot`
            // and the return sweep both skip params, so a retain here would
            // never be released.
            .filter(|sym| !state.param_symbols.contains(sym))
            // DataLoader handles are freed by loader teardown, not by us.
            .filter(|sym| !state.borrowed_batch_symbols.contains(sym))
            .filter(|sym| !state.dataloader_symbols.contains(sym))
            // NEVER the loop's OWN induction/pattern binding.
            //
            // Every loop lowering re-declares its pattern symbol and def's it
            // to ZERO before this hook runs, then rebinds it per iteration to
            // a BORROW taken with no retain (`nsl_list_get`, the dataloader's
            // `next_batch`, a model-array slot). If the pattern name shadows a
            // symbol already in `non_owning_symbols` —
            //
            //     let h = x          # x a param/field, so h is non-owning
            //     for h in items:    # h's slot is re-declared and zeroed
            //         h = f(h)       # owning RHS, so the veto below passes
            //
            // — then `use_var` reads the freshly zeroed slot and the retain is
            // a silent no-op (`nsl_tensor_retain(0)` returns immediately),
            // while the `non_owning_symbols.remove` below still lands. From
            // then on `eltls_clear_old_slot` fires once per iteration on a
            // borrowed container element that nothing ever retained: an
            // UNPAIRED free, i.e. a negative net refcount and a box handed
            // back to the allocator while the container still points at it.
            // A later `free_if_valid` magic probe then hits a recycled box and
            // decrements a DIFFERENT live tensor.
            //
            // `sym_bindings_all_owning_in_block` already rejects *nested*
            // pattern binders, but the loop's own pattern is not part of the
            // body it inspects — it has to be excluded here.
            .filter(|sym| {
                loop_pattern.is_none_or(|p| !self.pattern_binds_sym(p, *sym))
            })
            // Every in-body binding must be owning (see doc comment).
            .filter(|sym| self.sym_bindings_all_owning_in_block(body, *sym))
            .filter_map(|sym| {
                let is_tensor = state
                    .variable_types
                    .get(&sym)
                    .map(|ty| ty.is_tensor())
                    .unwrap_or(false);
                if !is_tensor {
                    return None;
                }
                state.variables.get(&sym).and_then(|(var, cl_type)| {
                    (*cl_type == cl_types::I64).then_some((sym, *var))
                })
            })
            .collect();

        for (sym, var) in materialize {
            let current_val = builder.use_var(var);
            let _ = self.compile_call_by_name(builder, "nsl_tensor_retain", &[current_val])?;
            state.non_owning_symbols.remove(&sym);
        }

        Ok(())
    }

    fn update_non_owning_binding(
        &self,
        state: &mut FuncState,
        target_sym: nsl_ast::Symbol,
        value: Option<&nsl_ast::expr::Expr>,
    ) {
        let Some(expr) = value else {
            state.non_owning_symbols.remove(&target_sym);
            return;
        };

        if let ExprKind::Ident(source_sym) = &expr.kind
            && (state.param_symbols.contains(source_sym)
                || state.non_owning_symbols.contains(source_sym))
        {
            state.non_owning_symbols.insert(target_sym);
            return;
        }

        // A non-Dict subscript hands out a BORROWED element: `compile_subscript`
        // lowers lists/tuples through `nsl_list_get`, which returns the stored
        // raw pointer with no retain. `let t = items[0]` therefore aliases an
        // element the container still owns, and treating it as owning let the
        // return sweep free a live element. Dict reads are the exception the
        // rest of this file already carves out (`loop_binding_rhs_is_owning`
        // encodes exactly this rule for the loop-rebind free).
        if let ExprKind::Subscript { object, .. } = &expr.kind {
            let is_dict = matches!(
                self.node_type(object.id),
                nsl_semantic::types::Type::Dict(_, _)
            );
            if !is_dict {
                state.non_owning_symbols.insert(target_sym);
                return;
            }
        }

        // A bare member access on a model instance hands out the model's own
        // field handle (no retain) — e.g. `let alias = m.w`. The binding is a
        // borrow: freeing it would free the weight itself. Marking it
        // non-owning makes the step-end cleanup skip it, makes ELTLS rebind
        // skip the old-value free, and makes assignment-inside-if materialize
        // a clone first (the established clone-on-mutate discipline).
        if let ExprKind::MemberAccess { object, .. } = &expr.kind
            && matches!(
                self.node_type(object.id),
                nsl_semantic::types::Type::Model { .. }
            )
        {
            state.non_owning_symbols.insert(target_sym);
            return;
        }

        state.non_owning_symbols.remove(&target_sym);
    }

    /// M36: Try to compile a tensor creation as a slab-managed allocation.
    /// Returns Ok(Some(value)) if the variable is slab-planned and the RHS is a
    /// tensor creation (zeros, ones, etc.). Returns Ok(None) to fall through to normal codegen.
    fn try_compile_slab_tensor(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        sym: &nsl_ast::Symbol,
        expr: &nsl_ast::expr::Expr,
    ) -> Result<Option<Value>, CodegenError> {
        // Check if slab is active and this variable is planned
        let slab_var = match state.slab_ptr_var {
            Some(v) => v,
            None => return Ok(None),
        };
        let var_name = match self.interner.resolve(sym.0) {
            Some(n) => n.to_string(),
            None => return Ok(None),
        };
        let offset = match self.memory.slab_name_offsets.get(&var_name) {
            Some(&o) => o,
            None => return Ok(None),
        };

        // Check if the RHS is a tensor creation call (zeros, ones, rand, zeros_on)
        let is_tensor_creation = match &expr.kind {
            ExprKind::Call { callee, .. } => match &callee.kind {
                ExprKind::Ident(func_sym) => {
                    let func_name = self.interner.resolve(func_sym.0).unwrap_or("");
                    matches!(
                        func_name,
                        "zeros" | "ones" | "rand" | "randn" | "zeros_like"
                    )
                }
                _ => false,
            },
            // zeros_on is typically a method call: Tensor.zeros_on(shape, device)
            _ => false,
        };

        if !is_tensor_creation {
            return Ok(None);
        }

        // Extract the shape argument from the call
        let shape_val = if let ExprKind::Call { args, .. } = &expr.kind {
            if args.is_empty() {
                return Ok(None);
            }
            self.compile_expr(builder, state, &args[0].value)?
        } else {
            return Ok(None);
        };

        // Compute data pointer: slab_base + offset
        let slab_ptr = builder.use_var(slab_var);
        let offset_val = builder.ins().iconst(cl_types::I64, offset as i64);
        let data_ptr =
            self.compile_call_by_name(builder, "nsl_slab_offset", &[slab_ptr, offset_val])?;

        // Determine device and dtype from the expression type
        let (device, dtype) = if let Some(ty) = self.type_map.get(&expr.id) {
            if let Some((_shape, dt, dev)) = ty.as_tensor_parts() {
                let dev_val = match dev {
                    nsl_semantic::types::Device::Cuda(_) => 1i64,
                    nsl_semantic::types::Device::Cpu => 0i64,
                    _ => 0i64,
                };
                let dt_val = match dt {
                    nsl_semantic::types::DType::F32 => 1i64,
                    nsl_semantic::types::DType::F64 => 0i64,
                    _ => 1i64, // default GPU dtype
                };
                (dev_val, dt_val)
            } else {
                (0, 1) // fallback
            }
        } else {
            (0, 1)
        };

        let device_val = builder.ins().iconst(cl_types::I64, device);
        let dtype_val = builder.ins().iconst(cl_types::I64, dtype);

        let tensor = self.compile_call_by_name(
            builder,
            "nsl_tensor_from_slab",
            &[data_ptr, shape_val, device_val, dtype_val],
        )?;

        Ok(Some(tensor))
    }

    /// Recursively destructure patterns from a list/tuple value.
    /// Each `PatternKind::Ident` binds a variable, `Wildcard` is skipped,
    /// `Tuple`/`List` recurse into nested `nsl_list_get` calls, and
    /// `Struct` destructures by field name via `nsl_dict_get`.
    fn destructure_element_type(
        &self,
        container_ty: Option<&Type>,
        index: usize,
        rest_index: Option<usize>,
        total_patterns: usize,
    ) -> Option<Type> {
        match container_ty? {
            Type::Tuple(items) => {
                let actual_index = match rest_index {
                    Some(rest_pos) if index > rest_pos => {
                        let tail_count = total_patterns.saturating_sub(index);
                        items.len().checked_sub(tail_count)?
                    }
                    _ => index,
                };
                items.get(actual_index).cloned()
            }
            Type::List(elem_ty) => Some((**elem_ty).clone()),
            _ => None,
        }
    }

    fn destructure_rest_type(
        &self,
        container_ty: Option<&Type>,
        rest_index: usize,
        total_patterns: usize,
    ) -> Option<Type> {
        match container_ty? {
            Type::Tuple(items) => {
                let trailing_patterns = total_patterns.saturating_sub(rest_index + 1);
                let end = items.len().saturating_sub(trailing_patterns);
                let start = rest_index.min(end);
                Some(Type::Tuple(items[start..end].to_vec()))
            }
            Type::List(elem_ty) => Some(Type::List(Box::new((**elem_ty).clone()))),
            _ => None,
        }
    }

    fn destructure_field_type(
        &self,
        container_ty: Option<&Type>,
        field: nsl_ast::Symbol,
    ) -> Option<Type> {
        match container_ty? {
            Type::Dict(_, value_ty) => Some((**value_ty).clone()),
            Type::Struct { fields, .. } | Type::Model { fields, .. } => fields
                .iter()
                .find_map(|(name, ty)| (*name == field).then(|| ty.clone())),
            _ => None,
        }
    }

    fn compile_destructure_patterns(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        patterns: &[nsl_ast::pattern::Pattern],
        container_val: cranelift_codegen::ir::Value,
        container_ty: Option<&Type>,
    ) -> Result<(), CodegenError> {
        let get_id = self.registry.runtime_fns["nsl_list_get"].0;
        let get_ref = self.module.declare_func_in_func(get_id, builder.func);
        let rest_positions: Vec<usize> = patterns
            .iter()
            .enumerate()
            .filter_map(|(i, pat)| matches!(pat.kind, PatternKind::Rest(_)).then_some(i))
            .collect();
        if rest_positions.len() > 1 {
            return Err(CodegenError::new(
                "multiple rest patterns in a single destructuring pattern are not supported",
            ));
        }
        let rest_index = rest_positions.first().copied();
        let container_len = if rest_index.is_some() {
            Some(self.compile_call_by_name(builder, "nsl_list_len", &[container_val])?)
        } else {
            None
        };

        for (i, sub_pat) in patterns.iter().enumerate() {
            let idx = match rest_index {
                Some(rest_pos) if i > rest_pos => {
                    let tail_count = builder
                        .ins()
                        .iconst(cl_types::I64, patterns.len().saturating_sub(i) as i64);
                    builder.ins().isub(container_len.unwrap(), tail_count)
                }
                _ => builder.ins().iconst(cl_types::I64, i as i64),
            };
            match &sub_pat.kind {
                PatternKind::Ident(sym) => {
                    let call = builder.ins().call(get_ref, &[container_val, idx]);
                    let elem = builder.inst_results(call)[0];
                    let var = builder.declare_var(cl_types::I64);
                    builder.def_var(var, elem);
                    state.variables.insert(*sym, (var, cl_types::I64));
                    if let Some(elem_ty) =
                        self.destructure_element_type(container_ty, i, rest_index, patterns.len())
                    {
                        state.variable_types.insert(*sym, elem_ty);
                    }
                }
                PatternKind::Wildcard => {}
                PatternKind::Tuple(nested) | PatternKind::List(nested) => {
                    // Extract the i-th element, then recurse into it
                    let call = builder.ins().call(get_ref, &[container_val, idx]);
                    let nested_val = builder.inst_results(call)[0];
                    let nested_ty =
                        self.destructure_element_type(container_ty, i, rest_index, patterns.len());
                    self.compile_destructure_patterns(
                        builder,
                        state,
                        nested,
                        nested_val,
                        nested_ty.as_ref(),
                    )?;
                }
                PatternKind::Struct { fields, .. } => {
                    // Extract the i-th element (the struct/dict), then destructure fields
                    let call = builder.ins().call(get_ref, &[container_val, idx]);
                    let struct_val = builder.inst_results(call)[0];
                    let struct_ty =
                        self.destructure_element_type(container_ty, i, rest_index, patterns.len());
                    for field in fields {
                        let field_name = self.resolve_sym(field.name).to_string();
                        // Ensure string is in pool, then get pointer for dict lookup
                        if !self.string_pool.contains_key(field_name.as_str()) {
                            self.intern_string(&field_name)?;
                        }
                        let key_str = self.compile_string_literal(builder, &field_name)?;
                        let field_ty = self.destructure_field_type(struct_ty.as_ref(), field.name);
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
                            // Nested pattern: { x: (a, b) } → destructure the field value
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
                                        "unsupported nested pattern in struct field '{}'",
                                        field_name
                                    )));
                                }
                            }
                        } else {
                            // Simple field binding: { name } binds `name` to the value
                            let var = builder.declare_var(cl_types::I64);
                            builder.def_var(var, field_val);
                            state.variables.insert(field.name, (var, cl_types::I64));
                            if let Some(field_ty) = field_ty {
                                state.variable_types.insert(field.name, field_ty);
                            }
                        }
                    }
                }
                PatternKind::Typed { pattern, .. } => {
                    // Type annotation is semantic-only — recurse into the inner pattern
                    let call = builder.ins().call(get_ref, &[container_val, idx]);
                    let elem = builder.inst_results(call)[0];
                    let elem_ty =
                        self.destructure_element_type(container_ty, i, rest_index, patterns.len());
                    match &pattern.kind {
                        PatternKind::Ident(sym) => {
                            let var = builder.declare_var(cl_types::I64);
                            builder.def_var(var, elem);
                            state.variables.insert(*sym, (var, cl_types::I64));
                            if let Some(elem_ty) = elem_ty {
                                state.variable_types.insert(*sym, elem_ty);
                            }
                        }
                        PatternKind::Tuple(nested) | PatternKind::List(nested) => {
                            self.compile_destructure_patterns(
                                builder,
                                state,
                                nested,
                                elem,
                                elem_ty.as_ref(),
                            )?;
                        }
                        PatternKind::Wildcard => {}
                        _ => {
                            return Err(CodegenError::new("unsupported typed pattern variant"));
                        }
                    }
                }
                PatternKind::Rest(rest_sym) => {
                    let lo = builder.ins().iconst(cl_types::I64, i as i64);
                    let hi = if i + 1 < patterns.len() {
                        let trailing = builder
                            .ins()
                            .iconst(cl_types::I64, patterns.len().saturating_sub(i + 1) as i64);
                        builder.ins().isub(container_len.unwrap(), trailing)
                    } else {
                        container_len.unwrap()
                    };
                    let step = builder.ins().iconst(cl_types::I64, 1);
                    let rest_val = self.compile_call_by_name(
                        builder,
                        "nsl_list_slice",
                        &[container_val, lo, hi, step],
                    )?;

                    if let Some(sym) = rest_sym {
                        let var = builder.declare_var(cl_types::I64);
                        builder.def_var(var, rest_val);
                        state.variables.insert(*sym, (var, cl_types::I64));
                        if let Some(rest_ty) =
                            self.destructure_rest_type(container_ty, i, patterns.len())
                        {
                            state.variable_types.insert(*sym, rest_ty);
                        }
                    } else {
                        self.compile_call_by_name(builder, "nsl_list_free", &[rest_val])?;
                    }
                }
                _ => {
                    return Err(CodegenError::new(format!(
                        "unsupported pattern kind in destructuring at position {}",
                        i
                    )));
                }
            }
        }
        Ok(())
    }

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

    /// Shared exit for every non-Ident assignment target (dict/list set,
    /// tensor multi-dim set, struct/model field stores, adapter
    /// side-table stores): drain the statement's temporaries with the
    /// just-stored value excluded, mirroring the Ident arm's tail.
    /// These arms previously had NO drain at all (PR #433 review LOW-2),
    /// which broke in two ways:
    ///
    /// - The stored value: an OWNED temp (`d["k"] = t * 2.0`) stayed in
    ///   `tensor_temporaries`, so the NEXT statement's sweep freed it
    ///   while the container still held the raw handle — dict reads
    ///   cloned a freed tensor, list reads handed out the dangling
    ///   pointer itself, and the adapter side-table's free-on-overwrite
    ///   became a double free. `free_tensor_temporaries` DRAINS the
    ///   list (`split_off`) and skips freeing `keep`, so passing the
    ///   stored value as `keep` IS the ownership transfer into the
    ///   container: the handle leaves the sweep's reach unfree'd.
    ///   Nothing is retained for borrowed stores — per the borrow-store
    ///   convention (dict_lifetime.rs) no machinery ever releases a
    ///   container-stored borrow, so a retain would strand one
    ///   reference per store.
    /// - Sub-expression temporaries (`d["k"] = t * 2.0 + 1.0` leaves
    ///   the inner `t * 2.0`) otherwise straddle past the statement;
    ///   a region that frees the temporaries list without draining it —
    ///   the train-block step loop — then freed the straddler once per
    ///   step: double free at step 2.
    fn assign_container_store_tail(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        stored_val: Value,
    ) {
        self.free_tensor_temporaries(builder, state, Some(stored_val));
        self.free_linear_consumes(builder, state, Some(stored_val));
    }

    fn compile_assign(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        target: &nsl_ast::expr::Expr,
        op: AssignOp,
        value: &nsl_ast::expr::Expr,
    ) -> Result<(), CodegenError> {
        if self.expr_is_borrowed_batch_handle(state, value) {
            return Err(CodegenError::new(
                "cannot assign a DataLoader batch handle directly; access batch fields instead",
            ));
        }
        let new_val = self.compile_expr(builder, state, value)?;
        match &target.kind {
            nsl_ast::expr::ExprKind::Ident(sym) => {
                let (var, _) = *state.variables.get(sym).ok_or_else(|| {
                    CodegenError::new(format!(
                        "undefined variable '{}' in assignment",
                        self.resolve_sym(*sym)
                    ))
                })?;

                let target_type = self.node_type(target.id).clone();
                let is_float = is_float_type(&target_type);

                if matches!(op, AssignOp::Assign) && state.dataloader_symbols.contains(sym) {
                    return Err(CodegenError::new(format!(
                        "reassigning DataLoader handle '{}' is unsupported; create a new loader symbol instead",
                        self.resolve_sym(*sym)
                    )));
                }

                let final_val = match op {
                    AssignOp::Assign => new_val,
                    AssignOp::AddAssign => {
                        let old = builder.use_var(var);
                        if is_float {
                            builder.ins().fadd(old, new_val)
                        } else {
                            builder.ins().iadd(old, new_val)
                        }
                    }
                    AssignOp::SubAssign => {
                        let old = builder.use_var(var);
                        if is_float {
                            builder.ins().fsub(old, new_val)
                        } else {
                            builder.ins().isub(old, new_val)
                        }
                    }
                    AssignOp::MulAssign => {
                        let old = builder.use_var(var);
                        if is_float {
                            builder.ins().fmul(old, new_val)
                        } else {
                            builder.ins().imul(old, new_val)
                        }
                    }
                    AssignOp::DivAssign => {
                        let old = builder.use_var(var);
                        if is_float {
                            builder.ins().fdiv(old, new_val)
                        } else {
                            self.compile_divmod_guard(builder, state, new_val)?;
                            builder.ins().sdiv(old, new_val)
                        }
                    }
                };
                // ELTLS §6.5: clear the old tensor slot before overwriting.
                // Only for plain Assign — compound ops consume `old` in-place
                // as an arithmetic input, not as a storage slot.
                if matches!(op, AssignOp::Assign) {
                    self.eltls_clear_old_slot(builder, state, *sym);
                }

                // ELTLS §6.5: consult RHS ownership for Assign and emit the
                // correct transfer/retain path for tensor-typed values.
                // Require the Cranelift value to be I64 AND semantic type to
                // be a real tensor (NOT indeterminate). Also skip inside
                // dtype methods where slot contents may be scalars.
                if matches!(op, AssignOp::Assign) {
                    let rhs_ty = self.node_type(value.id).clone();
                    let val_is_ptr = builder.func.dfg.value_type(final_val) == cl_types::I64;
                    if val_is_ptr && rhs_ty.is_tensor() && !state.flags.in_dtype_method {
                        use crate::ownership_expr::Ownership;
                        // Upgrade Unknown to Owned when the RHS is a call
                        // contractually returning an owning ref — the exact
                        // twin of the Return handler's upgrade above (ELTLS
                        // §6.5). Without it, `x = self.norm.forward(x)`
                        // (model-method results carry no ELTLS registration)
                        // took the conservative retain below and double-owned
                        // the result: the variable's single release left one
                        // reference behind — one stranded block per
                        // assignment, measured as the final-norm strand on
                        // every Coder-50M forward (the `x = ...` twin of the
                        // `return rmsnorm(...)` leak).
                        let mut own = self.get_ownership(state, final_val);
                        if matches!(own, Ownership::Unknown)
                            && self.expr_call_returns_owning_ref(value)
                        {
                            own = Ownership::Owned;
                        }
                        match own {
                            Ownership::Owned => {
                                self.consume_ownership(state, final_val);
                            }
                            Ownership::BorrowedFromVar(_) | Ownership::BorrowedWeight => {
                                let _ = self.compile_call_by_name(
                                    builder,
                                    "nsl_tensor_retain",
                                    &[final_val],
                                );
                            }
                            Ownership::TapeHeld => {
                                let _ = self.compile_call_by_name(
                                    builder,
                                    "nsl_tensor_retain",
                                    &[final_val],
                                );
                            }
                            Ownership::Unknown => {
                                let _ = self.compile_call_by_name(
                                    builder,
                                    "nsl_tensor_retain",
                                    &[final_val],
                                );
                                self.note_unknown_fallback(state, final_val);
                            }
                        }
                    }
                }
                builder.def_var(var, final_val);
                if matches!(op, AssignOp::Assign) {
                    if self.expr_is_dataloader_handle(state, value) {
                        state.dataloader_symbols.insert(*sym);
                    } else {
                        state.dataloader_symbols.remove(sym);
                    }
                    self.update_non_owning_binding(state, *sym, Some(value));
                    // Same capture-count transfer/clear as the VarDecl arm
                    // (review MEDIUM-1 on 44c011c1): a plain `f = <lambda>`
                    // rebind previously left closure_info stale in BOTH
                    // directions — a non-capturing rebind after a capturing
                    // one made the call site read the bare fn pointer as a
                    // closure struct (silent death), and a capturing rebind
                    // was never recorded at all.
                    if let Some(count) = self.registry.last_lambda_capture_count.take() {
                        state.set_closure_info(*sym, Some(count));
                    } else {
                        state.set_closure_info(*sym, None);
                    }
                }
                // Free intermediate tensor temporaries (keep final_val which is now owned by the variable)
                self.free_tensor_temporaries(builder, state, Some(final_val));
                // M38b: Free linear tensors consumed during this assignment's RHS
                self.free_linear_consumes(builder, state, Some(final_val));
            }
            nsl_ast::expr::ExprKind::Subscript { object, index } => {
                if self.expr_is_borrowed_batch_handle(state, object) {
                    return Err(CodegenError::new(
                        "cannot mutate a DataLoader batch dict directly; bind or replace batch fields instead",
                    ));
                }
                let obj_val = self.compile_expr(builder, state, object)?;
                let obj_type = self.node_type(object.id).clone();
                match index.as_ref() {
                    SubscriptKind::Index(idx_expr) => {
                        let idx_val = self.compile_expr(builder, state, idx_expr)?;
                        let is_dict = matches!(obj_type, nsl_semantic::types::Type::Dict { .. });

                        let final_val = if matches!(op, AssignOp::Assign) {
                            new_val
                        } else {
                            // Read-modify-write: get old value, apply op, write back
                            let get_fn = if is_dict {
                                "nsl_dict_get_str"
                            } else {
                                "nsl_list_get"
                            };
                            let get_id = self.registry.runtime_fns[get_fn].0;
                            let get_ref = self.module.declare_func_in_func(get_id, builder.func);
                            let call = builder.ins().call(get_ref, &[obj_val, idx_val]);
                            let old_val = builder.inst_results(call)[0];

                            match op {
                                AssignOp::AddAssign => builder.ins().iadd(old_val, new_val),
                                AssignOp::SubAssign => builder.ins().isub(old_val, new_val),
                                AssignOp::MulAssign => builder.ins().imul(old_val, new_val),
                                AssignOp::DivAssign => {
                                    self.compile_divmod_guard(builder, state, new_val)?;
                                    builder.ins().sdiv(old_val, new_val)
                                }
                                _ => unreachable!(),
                            }
                        };

                        let set_fn = if is_dict {
                            "nsl_dict_set_str"
                        } else {
                            "nsl_list_set"
                        };
                        let set_id = self.registry.runtime_fns[set_fn].0;
                        let set_ref = self.module.declare_func_in_func(set_id, builder.func);
                        builder.ins().call(set_ref, &[obj_val, idx_val, final_val]);
                        self.assign_container_store_tail(builder, state, final_val);
                    }
                    SubscriptKind::MultiDim(dims) => {
                        // Tensor element write: t[i, j, ...] = v (and compound
                        // forms) → nsl_tensor_set(t, [i, j, ...], v as f64).
                        // The runtime validates arity/bounds and applies strides.
                        if !obj_type.is_tensor() && !obj_type.is_indeterminate() {
                            return Err(CodegenError::new(format!(
                                "multi-dim subscript assignment requires a tensor, got {obj_type:?}"
                            )));
                        }
                        let indices_list =
                            self.compile_call_by_name(builder, "nsl_list_new", &[])?;
                        for dim in dims {
                            let SubscriptKind::Index(idx_expr) = dim else {
                                return Err(CodegenError::new(
                                    "mixed index/slice in multi-dim tensor subscript \
                                     assignment is not supported",
                                ));
                            };
                            let idx_raw = self.compile_expr(builder, state, idx_expr)?;
                            let idx_val = if matches!(
                                self.node_type(idx_expr.id),
                                nsl_semantic::types::Type::Float
                            ) {
                                builder.ins().fcvt_to_sint(cl_types::I64, idx_raw)
                            } else {
                                idx_raw
                            };
                            self.compile_call_by_name(
                                builder,
                                "nsl_list_push",
                                &[indices_list, idx_val],
                            )?;
                        }
                        // nsl_tensor_set takes the value as F64; coerce ints.
                        let rhs_f64 = if matches!(
                            self.node_type(value.id),
                            nsl_semantic::types::Type::Int | nsl_semantic::types::Type::Bool
                        ) {
                            builder.ins().fcvt_from_sint(cl_types::F64, new_val)
                        } else {
                            new_val
                        };
                        let final_val = if matches!(op, AssignOp::Assign) {
                            rhs_f64
                        } else {
                            // Read-modify-write on the element in f64.
                            let old_val = self.compile_call_by_name(
                                builder,
                                "nsl_tensor_get",
                                &[obj_val, indices_list],
                            )?;
                            match op {
                                AssignOp::AddAssign => builder.ins().fadd(old_val, rhs_f64),
                                AssignOp::SubAssign => builder.ins().fsub(old_val, rhs_f64),
                                AssignOp::MulAssign => builder.ins().fmul(old_val, rhs_f64),
                                AssignOp::DivAssign => builder.ins().fdiv(old_val, rhs_f64),
                                _ => unreachable!(),
                            }
                        };
                        self.compile_call_by_name(
                            builder,
                            "nsl_tensor_set",
                            &[obj_val, indices_list, final_val],
                        )?;
                        self.compile_call_by_name(builder, "nsl_list_free", &[indices_list])?;
                        // Stored value is a scalar; the drain covers
                        // index-expression and RHS temporaries.
                        self.assign_container_store_tail(builder, state, final_val);
                    }
                    _ => return Err(CodegenError::new("only simple index assignment supported")),
                }
            }
            nsl_ast::expr::ExprKind::MemberAccess { object, member } => {
                let obj_val = self.compile_expr(builder, state, object)?;
                let member_name = self.resolve_sym(*member).to_string();
                let obj_type = self.node_type(object.id).clone();
                if let nsl_semantic::types::Type::Struct { name, .. } = &obj_type {
                    let struct_name = self.resolve_sym(*name).to_string();
                    if let Some(layout) = self.types.struct_layouts.get(&struct_name) {
                        for field in &layout.fields {
                            if field.name == member_name {
                                let final_val = if matches!(op, AssignOp::Assign) {
                                    new_val
                                } else {
                                    let old_val = builder.ins().load(
                                        field.cl_type,
                                        cranelift_codegen::ir::MemFlagsData::trusted(),
                                        obj_val,
                                        field.offset as i32,
                                    );
                                    let is_float = field.cl_type == cl_types::F64
                                        || field.cl_type == cl_types::F32;
                                    match (op, is_float) {
                                        (AssignOp::AddAssign, true) => {
                                            builder.ins().fadd(old_val, new_val)
                                        }
                                        (AssignOp::SubAssign, true) => {
                                            builder.ins().fsub(old_val, new_val)
                                        }
                                        (AssignOp::MulAssign, true) => {
                                            builder.ins().fmul(old_val, new_val)
                                        }
                                        (AssignOp::DivAssign, true) => {
                                            builder.ins().fdiv(old_val, new_val)
                                        }
                                        (AssignOp::AddAssign, false) => {
                                            builder.ins().iadd(old_val, new_val)
                                        }
                                        (AssignOp::SubAssign, false) => {
                                            builder.ins().isub(old_val, new_val)
                                        }
                                        (AssignOp::MulAssign, false) => {
                                            builder.ins().imul(old_val, new_val)
                                        }
                                        (AssignOp::DivAssign, false) => {
                                            // Inline div-by-zero guard (can't call method due to borrow)
                                            let ok_blk = builder.create_block();
                                            let trap_blk = builder.create_block();
                                            let is_zero =
                                                builder.ins().icmp_imm_s(IntCC::Equal, new_val, 0);
                                            builder.ins().brif(is_zero, trap_blk, &[], ok_blk, &[]);
                                            builder.switch_to_block(trap_blk);
                                            builder.seal_block(trap_blk);
                                            builder.ins().trap(
                                                cranelift_codegen::ir::TrapCode::unwrap_user(1),
                                            );
                                            builder.switch_to_block(ok_blk);
                                            builder.seal_block(ok_blk);
                                            state.current_block = Some(ok_blk);
                                            builder.ins().sdiv(old_val, new_val)
                                        }
                                        _ => unreachable!(),
                                    }
                                };
                                builder.ins().store(
                                    cranelift_codegen::ir::MemFlagsData::trusted(),
                                    final_val,
                                    obj_val,
                                    field.offset as i32,
                                );
                                self.assign_container_store_tail(builder, state, final_val);
                                return Ok(());
                            }
                        }
                        return Err(CodegenError::new(format!(
                            "struct '{struct_name}' has no field '{member_name}'"
                        )));
                    }
                }
                if let nsl_semantic::types::Type::Model { name, .. } = &obj_type {
                    let model_name = self.resolve_sym(*name).to_string();
                    // B.2.1 Task 5.5: synthesized adapter field assignment —
                    // store the new tensor pointer into the model's
                    // side-table slot rather than a struct field. Mirrors
                    // the read-through in `expr/access.rs`.
                    if crate::expr::access::is_synthesized_adapter_field_name(&member_name) {
                        if matches!(op, AssignOp::Assign)
                            && let Some(layout) =
                                self.types.struct_layouts.get(&model_name).cloned()
                            && let Some(slot_off) = layout.adapter_sidetable_offset
                        {
                            let index = self
                                .adapter_field_index(&model_name, &member_name)
                                .ok_or_else(|| {
                                    CodegenError::new(format!(
                                        "synthesized adapter field '{member_name}' \
                                                 not found for model '{model_name}' in \
                                                 current WRGA plan"
                                    ))
                                })?;
                            let table_ptr = builder.ins().load(
                                cl_types::I64,
                                cranelift_codegen::ir::MemFlagsData::trusted(),
                                obj_val,
                                slot_off as i32,
                            );
                            let byte_off = (index * 8) as i32;
                            // Free the existing tensor in the slot
                            // before overwriting (side-table owns
                            // the tensors it holds).
                            let old_ptr = builder.ins().load(
                                cl_types::I64,
                                cranelift_codegen::ir::MemFlagsData::trusted(),
                                table_ptr,
                                byte_off,
                            );
                            self.compile_call_by_name(
                                builder,
                                "nsl_tensor_free_if_valid",
                                &[old_ptr],
                            )?;
                            builder.ins().store(
                                cranelift_codegen::ir::MemFlagsData::trusted(),
                                new_val,
                                table_ptr,
                                byte_off,
                            );
                            // The side-table owns its tensors and
                            // frees the old slot on overwrite — a
                            // swept owned temp here became a later
                            // double free.
                            self.assign_container_store_tail(builder, state, new_val);
                            return Ok(());
                        }
                        return Err(CodegenError::new(format!(
                            "compound-assign to synthesized adapter field \
                             '{member_name}' is not supported"
                        )));
                    }
                    if let Some(layout) = self.types.struct_layouts.get(&model_name) {
                        for field in &layout.fields {
                            if field.name == member_name {
                                let final_val = if matches!(op, AssignOp::Assign) {
                                    new_val
                                } else {
                                    let old_val = builder.ins().load(
                                        field.cl_type,
                                        cranelift_codegen::ir::MemFlagsData::trusted(),
                                        obj_val,
                                        field.offset as i32,
                                    );
                                    let is_float = field.cl_type == cl_types::F64
                                        || field.cl_type == cl_types::F32;
                                    match (op, is_float) {
                                        (AssignOp::AddAssign, true) => {
                                            builder.ins().fadd(old_val, new_val)
                                        }
                                        (AssignOp::SubAssign, true) => {
                                            builder.ins().fsub(old_val, new_val)
                                        }
                                        (AssignOp::MulAssign, true) => {
                                            builder.ins().fmul(old_val, new_val)
                                        }
                                        (AssignOp::DivAssign, true) => {
                                            builder.ins().fdiv(old_val, new_val)
                                        }
                                        (AssignOp::AddAssign, false) => {
                                            builder.ins().iadd(old_val, new_val)
                                        }
                                        (AssignOp::SubAssign, false) => {
                                            builder.ins().isub(old_val, new_val)
                                        }
                                        (AssignOp::MulAssign, false) => {
                                            builder.ins().imul(old_val, new_val)
                                        }
                                        (AssignOp::DivAssign, false) => {
                                            // Inline div-by-zero guard (can't call method due to borrow)
                                            let ok_blk = builder.create_block();
                                            let trap_blk = builder.create_block();
                                            let is_zero =
                                                builder.ins().icmp_imm_s(IntCC::Equal, new_val, 0);
                                            builder.ins().brif(is_zero, trap_blk, &[], ok_blk, &[]);
                                            builder.switch_to_block(trap_blk);
                                            builder.seal_block(trap_blk);
                                            builder.ins().trap(
                                                cranelift_codegen::ir::TrapCode::unwrap_user(1),
                                            );
                                            builder.switch_to_block(ok_blk);
                                            builder.seal_block(ok_blk);
                                            state.current_block = Some(ok_blk);
                                            builder.ins().sdiv(old_val, new_val)
                                        }
                                        _ => unreachable!(),
                                    }
                                };
                                builder.ins().store(
                                    cranelift_codegen::ir::MemFlagsData::trusted(),
                                    final_val,
                                    obj_val,
                                    field.offset as i32,
                                );
                                self.assign_container_store_tail(builder, state, final_val);
                                return Ok(());
                            }
                        }
                        return Err(CodegenError::new(format!(
                            "model '{model_name}' has no field '{member_name}'"
                        )));
                    }
                }
                return Err(CodegenError::new(format!(
                    "member assignment not supported for .{member_name}"
                )));
            }
            _ => {
                return Err(CodegenError::new(
                    "only variable/subscript/member assignment supported in M4",
                ))
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

    /// True when every VarDecl/Assign binding of `sym` in the block —
    /// including nested if/loop/match blocks, which rebind the SAME slot
    /// (this lowering has no block-level scoping) — has an owning RHS.
    fn sym_bindings_all_owning_in_block(
        &self,
        block: &nsl_ast::stmt::Block,
        sym: nsl_ast::Symbol,
    ) -> bool {
        block
            .stmts
            .iter()
            .all(|s| self.sym_bindings_all_owning_in_stmt(s, sym))
    }

    fn pattern_binds_sym(&self, pattern: &nsl_ast::pattern::Pattern, sym: nsl_ast::Symbol) -> bool {
        let mut bound = std::collections::HashSet::new();
        self.collect_pattern_bound_symbols(pattern, &mut bound);
        bound.contains(&sym)
    }

    fn sym_bindings_all_owning_in_stmt(
        &self,
        stmt: &nsl_ast::stmt::Stmt,
        sym: nsl_ast::Symbol,
    ) -> bool {
        use nsl_ast::stmt::StmtKind;
        match &stmt.kind {
            StmtKind::VarDecl { pattern, value, .. } => {
                if let nsl_ast::pattern::PatternKind::Ident(s) = &pattern.kind {
                    if *s == sym {
                        // The RHS must be owning AND tensor-typed: an armed
                        // clear on an INT slot hands the integer to
                        // free_if_valid, whose magic probe dereferences any
                        // 8-aligned value >= 0x10000.
                        return value.as_ref().is_none_or(|e| {
                            self.loop_binding_rhs_is_owning(e)
                                && matches!(self.node_type(e.id),
                                            ty if ty.is_tensor() || ty.is_indeterminate())
                        });
                    }
                    return true;
                }
                // Destructuring patterns (`let (y, z) = pair`) bind shared
                // tuple/list members — non-owning.
                !self.pattern_binds_sym(pattern, sym)
            }
            StmtKind::Assign { target, value, .. } => {
                if let ExprKind::Ident(s) = &target.kind
                    && *s == sym
                {
                    return self.loop_binding_rhs_is_owning(value)
                        && matches!(self.node_type(value.id),
                                    ty if ty.is_tensor() || ty.is_indeterminate());
                }
                true
            }
            StmtKind::If {
                then_block,
                elif_clauses,
                else_block,
                ..
            } => {
                self.sym_bindings_all_owning_in_block(then_block, sym)
                    && elif_clauses
                        .iter()
                        .all(|(_, b)| self.sym_bindings_all_owning_in_block(b, sym))
                    && else_block
                        .as_ref()
                        .is_none_or(|b| self.sym_bindings_all_owning_in_block(b, sym))
            }
            // Loop/while-let/match patterns bind borrowed elements (list
            // members via nsl_list_get, match subjects) into the SAME slot
            // when the name shadows an armed sym — that's a non-owning
            // binding of `sym`.
            StmtKind::For { pattern, body, .. } | StmtKind::WhileLet { pattern, body, .. } => {
                !self.pattern_binds_sym(pattern, sym)
                    && self.sym_bindings_all_owning_in_block(body, sym)
            }
            StmtKind::While { body, .. } => self.sym_bindings_all_owning_in_block(body, sym),
            StmtKind::Match { arms, .. } => arms.iter().all(|arm| {
                !self.pattern_binds_sym(&arm.pattern, sym)
                    && self.sym_bindings_all_owning_in_block(&arm.body, sym)
            }),
            StmtKind::Decorated { stmt, .. } => self.sym_bindings_all_owning_in_stmt(stmt, sym),
            // Opaque block constructs compile their bodies against the SAME
            // FuncState but through their own lowering — this walker cannot
            // see their bindings. Presence of one vetoes the sym.
            StmtKind::TrainBlock(_)
            | StmtKind::GradBlock(_)
            | StmtKind::DistillBlock(_)
            | StmtKind::QuantBlock(_)
            | StmtKind::ServeBlock(_) => false,
            _ => true,
        }
    }

    /// Does this RHS hand back a reference the binding OWNS?
    /// False for the raw-pointer-copy forms: ident references (no retain),
    /// member access (model weights / struct fields share the stored
    /// handle), and non-dict subscripts — compile_subscript lowers
    /// EVERYTHING except Dict (lists, tuples, Unknown) through
    /// nsl_list_get, which shares the stored element with no retain. Only
    /// dict subscripts clone tensors (owned). Match/block expressions do
    /// not retain their arm results (unlike if-expressions, which do), so
    /// they are vetoed too. Calls, method calls, and operators produce
    /// fresh results.
    fn loop_binding_rhs_is_owning(&self, expr: &nsl_ast::expr::Expr) -> bool {
        match &expr.kind {
            ExprKind::Ident(_) => false,
            ExprKind::MemberAccess { .. } => false,
            ExprKind::MatchExpr { .. } => false,
            ExprKind::BlockExpr(_) => false,
            ExprKind::Paren(inner) => self.loop_binding_rhs_is_owning(inner),
            ExprKind::IfExpr {
                then_expr,
                else_expr,
                ..
            } => {
                self.loop_binding_rhs_is_owning(then_expr)
                    && self.loop_binding_rhs_is_owning(else_expr)
            }
            ExprKind::Subscript { object, .. } => matches!(
                self.node_type(object.id),
                nsl_semantic::types::Type::Dict(_, _)
            ),
            _ => true,
        }
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
    fn cleanup_loop_scope(&mut self, builder: &mut FunctionBuilder, state: &mut FuncState) {
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

    fn compile_if_stmt(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        condition: &nsl_ast::expr::Expr,
        then_block: &nsl_ast::stmt::Block,
        elif_clauses: &[(nsl_ast::expr::Expr, nsl_ast::stmt::Block)],
        else_block: &Option<nsl_ast::stmt::Block>,
    ) -> Result<(), CodegenError> {
        self.materialize_non_owning_aliases_before_if(
            builder,
            state,
            then_block,
            elif_clauses,
            else_block,
        )?;
        let merge_block = builder.create_block();
        state.flags.conditional_depth += 1;
        let cond_val = self.compile_expr(builder, state, condition);
        state.flags.conditional_depth -= 1;
        let cond_val = cond_val?;
        let incoming_loader_symbols = state.dataloader_symbols.clone();
        let mut reaching_loader_sets: Vec<std::collections::HashSet<nsl_ast::Symbol>> = Vec::new();
        let incoming_loader_vars = state.cleanup.dataloader_vars.clone();
        let mut reaching_loader_var_sets: Vec<Vec<Value>> = Vec::new();

        let then_bb = builder.create_block();
        let next_bb = if !elif_clauses.is_empty() || else_block.is_some() {
            builder.create_block()
        } else {
            merge_block
        };
        builder.ins().brif(cond_val, then_bb, &[], next_bb, &[]);

        builder.switch_to_block(then_bb);
        builder.seal_block(then_bb);
        state.current_block = Some(then_bb);
        state.dataloader_symbols = incoming_loader_symbols.clone();
        state.cleanup.dataloader_vars = incoming_loader_vars.clone();
        state.flags.conditional_depth += 1;
        state.push_fn_binding_scope();
        for s in &then_block.stmts {
            self.compile_stmt(builder, state, s)?;
        }
        state.pop_fn_binding_scope();
        state.flags.conditional_depth -= 1;
        let current = state.current_block.unwrap_or(then_bb);
        if !is_block_filled(builder, current) {
            reaching_loader_sets.push(state.dataloader_symbols.clone());
            reaching_loader_var_sets.push(state.cleanup.dataloader_vars.clone());
            builder.ins().jump(merge_block, &[]);
        }

        let mut current_else = next_bb;
        for (i, (elif_cond, elif_body)) in elif_clauses.iter().enumerate() {
            builder.switch_to_block(current_else);
            builder.seal_block(current_else);
            state.current_block = Some(current_else);
            let elif_cond_val = self.compile_expr(builder, state, elif_cond)?;

            let elif_then = builder.create_block();
            let elif_next = if i + 1 < elif_clauses.len() || else_block.is_some() {
                builder.create_block()
            } else {
                merge_block
            };
            builder
                .ins()
                .brif(elif_cond_val, elif_then, &[], elif_next, &[]);

            builder.switch_to_block(elif_then);
            builder.seal_block(elif_then);
            state.current_block = Some(elif_then);
            state.dataloader_symbols = incoming_loader_symbols.clone();
            state.cleanup.dataloader_vars = incoming_loader_vars.clone();
            state.flags.conditional_depth += 1;
            state.push_fn_binding_scope();
            for s in &elif_body.stmts {
                self.compile_stmt(builder, state, s)?;
            }
            state.pop_fn_binding_scope();
            state.flags.conditional_depth -= 1;
            let current = state.current_block.unwrap_or(elif_then);
            if !is_block_filled(builder, current) {
                reaching_loader_sets.push(state.dataloader_symbols.clone());
                reaching_loader_var_sets.push(state.cleanup.dataloader_vars.clone());
                builder.ins().jump(merge_block, &[]);
            }

            current_else = elif_next;
        }

        if let Some(else_body) = else_block {
            builder.switch_to_block(current_else);
            builder.seal_block(current_else);
            state.current_block = Some(current_else);
            state.dataloader_symbols = incoming_loader_symbols.clone();
            state.cleanup.dataloader_vars = incoming_loader_vars.clone();
            state.flags.conditional_depth += 1;
            state.push_fn_binding_scope();
            for s in &else_body.stmts {
                self.compile_stmt(builder, state, s)?;
            }
            state.pop_fn_binding_scope();
            state.flags.conditional_depth -= 1;
            let current = state.current_block.unwrap_or(current_else);
            if !is_block_filled(builder, current) {
                reaching_loader_sets.push(state.dataloader_symbols.clone());
                reaching_loader_var_sets.push(state.cleanup.dataloader_vars.clone());
                builder.ins().jump(merge_block, &[]);
            }
        } else if current_else != merge_block {
            builder.switch_to_block(current_else);
            builder.seal_block(current_else);
            state.current_block = Some(current_else);
            reaching_loader_sets.push(incoming_loader_symbols.clone());
            reaching_loader_var_sets.push(incoming_loader_vars.clone());
            builder.ins().jump(merge_block, &[]);
        }

        state.dataloader_symbols = if let Some(first) = reaching_loader_sets.first().cloned() {
            reaching_loader_sets
                .into_iter()
                .skip(1)
                .fold(first, |acc, branch_set| {
                    acc.into_iter()
                        .filter(|sym| branch_set.contains(sym))
                        .collect()
                })
        } else {
            incoming_loader_symbols
        };
        state.cleanup.dataloader_vars =
            if let Some(first) = reaching_loader_var_sets.first().cloned() {
                reaching_loader_var_sets
                    .into_iter()
                    .skip(1)
                    .fold(first, |acc, branch_vec| {
                        acc.into_iter()
                            .filter(|value| branch_vec.contains(value))
                            .collect()
                    })
            } else {
                incoming_loader_vars
            };

        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);
        state.current_block = Some(merge_block);
        Ok(())
    }

    fn compile_while(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        condition: &nsl_ast::expr::Expr,
        body: &nsl_ast::stmt::Block,
    ) -> Result<(), CodegenError> {
        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let exit_block = builder.create_block();

        // ELTLS Task 16.1: pre-declare top-level let-ident symbols from the
        // body so the second-and-later rebinds fire eltls_clear_old_slot
        // and free the previous iteration's tensor. MUST be emitted in the
        // pre-loop block: a def inside the body re-zeroes the slot every
        // iteration, so the rebind free only ever sees 0 and the previous
        // iteration's tensor strands.
        // Loop-carried locals seeded from a borrow (`let h = x`) are vetoed
        // by the flow-insensitive non_owning_symbols set, so their single
        // generated rebind-free site never fires. Give them their own
        // reference first — see materialize_non_owning_aliases_before_loop.
        self.materialize_non_owning_aliases_before_loop(builder, state, body, None)?;
        let predecl_syms = self.eltls_predeclare_loop_lets_checked(builder, state, body);

        builder.ins().jump(header_block, &[]);

        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let cond_base = state.cleanup.tensor_temporaries.len();
        let cond_val = self.compile_expr(builder, state, condition)?;
        // Free per-evaluation condition temporaries INSIDE the header
        // block, after the branch scalar is computed and before the brif.
        // The condition re-evaluates every iteration, but until now its
        // temps sat in tensor_temporaries below the loop-scope mark and
        // only the While STATEMENT's end-of-statement cleanup (in the
        // exit block) ever freed them — which frees exactly ONE
        // evaluation's values (the final one, whose SSA results dominate
        // the exit); every earlier iteration's condition temps stranded,
        // one block per tracked temp per evaluation (the deliberate
        // exact-6 pin in nested_arg_temporaries_gate, now retired).
        // Draining here means the header frees each evaluation's temps —
        // including the final one — and the exit-block cleanup no longer
        // sees them, so nothing double-frees. The brif consumes only the
        // extracted scalar, never the freed handles.
        self.free_condition_temporaries(builder, state, cond_base, cond_val);
        builder
            .ins()
            .brif(cond_val, body_block, &[], exit_block, &[]);

        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        state
            .cleanup
            .temp_scope_stack
            .push(state.cleanup.tensor_temporaries.len());
        state.loop_stack.push(LoopContext {
            continue_block: header_block,
            exit_block,
            batch_var: None,
        });
        state.push_fn_binding_scope();
        for s in &body.stmts {
            self.compile_stmt(builder, state, s)?;
        }
        state.pop_fn_binding_scope();
        state.loop_stack.pop();
        for sym in &predecl_syms {
            state.eltls_loop_predeclared.remove(sym);
        }

        let current = state.current_block.unwrap_or(body_block);
        if !is_block_filled(builder, current) {
            self.cleanup_loop_scope(builder, state);
            builder.ins().jump(header_block, &[]);
        } else {
            state.cleanup.temp_scope_stack.pop();
        }

        builder.seal_block(header_block);
        builder.switch_to_block(exit_block);
        builder.seal_block(exit_block);
        state.current_block = Some(exit_block);
        Ok(())
    }

    fn compile_while_let(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        pattern: &nsl_ast::pattern::Pattern,
        expr: &nsl_ast::expr::Expr,
        body: &nsl_ast::stmt::Block,
    ) -> Result<(), CodegenError> {
        // Pre-declare the pattern variable before the loop (once per function, not per iteration)
        let pattern_var = match &pattern.kind {
            PatternKind::Ident(sym) => {
                let var = builder.declare_var(cl_types::I64);
                let zero = builder.ins().iconst(cl_types::I64, 0);
                builder.def_var(var, zero);
                state.variables.insert(*sym, (var, cl_types::I64));
                Some(var)
            }
            PatternKind::Wildcard => None,
            _ => {
                return Err(CodegenError::new(
                    "only ident or wildcard patterns in while-let",
                ))
            }
        };

        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let exit_block = builder.create_block();

        // ELTLS Task 16.1: pre-declare top-level let-ident symbols from
        // the body so rebinds across iterations free the previous value.
        // Pre-loop block on purpose — see compile_while.
        // Loop-carried locals seeded from a borrow (`let h = x`) are vetoed
        // by the flow-insensitive non_owning_symbols set, so their single
        // generated rebind-free site never fires. Give them their own
        // reference first — see materialize_non_owning_aliases_before_loop.
        self.materialize_non_owning_aliases_before_loop(builder, state, body, Some(pattern))?;
        let predecl_syms = self.eltls_predeclare_loop_lets_checked(builder, state, body);

        builder.ins().jump(header_block, &[]);

        // Header: evaluate expression, check truthiness (non-zero = continue)
        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let expr_base = state.cleanup.tensor_temporaries.len();
        let val = self.compile_expr(builder, state, expr)?;
        // Per-evaluation sub-temporaries of the while-let expression free
        // in the header, same as compile_while. `val` itself is excluded
        // by free_condition_temporaries' keep parameter — it is bound to
        // the pattern variable and read throughout the body (top-level
        // compile_expr results are not tracked today, so the exclusion is
        // defensive, but a tracked `val` would otherwise be a
        // freed-then-read bug, not a leak).
        self.free_condition_temporaries(builder, state, expr_base, val);
        let cond = builder.ins().icmp_imm_s(IntCC::NotEqual, val, 0);
        builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        // Body: update pattern variable with current value, execute body
        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        // Update the pattern variable with the value from this iteration
        if let Some(var) = pattern_var {
            builder.def_var(var, val);
        }

        state
            .cleanup
            .temp_scope_stack
            .push(state.cleanup.tensor_temporaries.len());
        state.loop_stack.push(LoopContext {
            continue_block: header_block,
            exit_block,
            batch_var: None,
        });
        state.push_fn_binding_scope();
        for s in &body.stmts {
            self.compile_stmt(builder, state, s)?;
        }
        state.pop_fn_binding_scope();
        state.loop_stack.pop();
        for sym in &predecl_syms {
            state.eltls_loop_predeclared.remove(sym);
        }

        let current = state.current_block.unwrap_or(body_block);
        if !is_block_filled(builder, current) {
            self.cleanup_loop_scope(builder, state);
            builder.ins().jump(header_block, &[]);
        } else {
            state.cleanup.temp_scope_stack.pop();
        }

        builder.seal_block(header_block);
        builder.switch_to_block(exit_block);
        builder.seal_block(exit_block);
        state.current_block = Some(exit_block);
        Ok(())
    }

    fn compile_for(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        pattern: &nsl_ast::pattern::Pattern,
        iterable: &nsl_ast::expr::Expr,
        body: &nsl_ast::stmt::Block,
    ) -> Result<(), CodegenError> {
        // Check if iterating over a fixed model array
        let iter_type = self.node_type(iterable.id).clone();
        if let Type::FixedModelArray {
            element_model,
            size,
        } = &iter_type
        {
            return self.compile_for_model_array(
                builder,
                state,
                pattern,
                iterable,
                body,
                *element_model,
                *size,
            );
        }

        // DataLoader iteration uses an opaque runtime handle, not a real list.
        // Route to the loader protocol only for expressions proven to come from DataLoader(...).
        if self.is_dataloader_iterable(state, iterable) {
            return self.compile_for_dataloader(builder, state, pattern, iterable, body);
        }
        if matches!(iter_type, Type::Unknown) {
            nsl_runtime::nsl_log!(WARN, "nsl-codegen", 
                "[nsl-codegen] warning: for-loop iterable has Unknown type — compiling as list iteration. \
                 If this is a DataLoader, ensure the variable type is inferred correctly."
            );
        }

        let list_val = self.compile_expr(builder, state, iterable)?;

        let len_id = self.registry.runtime_fns["nsl_list_len"].0;
        let len_ref = self.module.declare_func_in_func(len_id, builder.func);
        let call = builder.ins().call(len_ref, &[list_val]);
        let list_len = builder.inst_results(call)[0];

        let counter_var = builder.declare_var(cl_types::I64);
        let zero = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(counter_var, zero);

        // Pre-declare pattern variables before the loop
        match &pattern.kind {
            PatternKind::Ident(sym) => {
                let elem_var = builder.declare_var(cl_types::I64);
                builder.def_var(elem_var, zero);
                state.variables.insert(*sym, (elem_var, cl_types::I64));
            }
            PatternKind::Tuple(sub_patterns) | PatternKind::List(sub_patterns) => {
                let rest_positions: Vec<usize> = sub_patterns
                    .iter()
                    .enumerate()
                    .filter_map(|(i, pat)| matches!(pat.kind, PatternKind::Rest(_)).then_some(i))
                    .collect();
                if rest_positions.len() > 1 {
                    return Err(CodegenError::new(
                        "multiple rest patterns in a single destructuring pattern are not supported",
                    ));
                }
                for sub_pat in sub_patterns {
                    match &sub_pat.kind {
                        PatternKind::Ident(sym) => {
                            let var = builder.declare_var(cl_types::I64);
                            builder.def_var(var, zero);
                            state.variables.insert(*sym, (var, cl_types::I64));
                        }
                        PatternKind::Rest(Some(sym)) => {
                            let var = builder.declare_var(cl_types::I64);
                            builder.def_var(var, zero);
                            state.variables.insert(*sym, (var, cl_types::I64));
                        }
                        PatternKind::Rest(None) | PatternKind::Wildcard => {}
                        _ => {}
                    }
                }
            }
            _ => {
                return Err(CodegenError::new(
                    "only ident, tuple, and list patterns in for loops",
                ))
            }
        }

        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let increment_block = builder.create_block();
        let exit_block = builder.create_block();

        // ELTLS Task 16.1: pre-declare top-level let-ident symbols from the
        // body so the second-and-later rebinds fire eltls_clear_old_slot.
        // Pre-loop block on purpose — see compile_while.
        // Loop-carried locals seeded from a borrow (`let h = x`) are vetoed
        // by the flow-insensitive non_owning_symbols set, so their single
        // generated rebind-free site never fires. Give them their own
        // reference first — see materialize_non_owning_aliases_before_loop.
        self.materialize_non_owning_aliases_before_loop(builder, state, body, Some(pattern))?;
        let predecl_syms = self.eltls_predeclare_loop_lets_checked(builder, state, body);

        builder.ins().jump(header_block, &[]);

        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let counter = builder.use_var(counter_var);
        let cond = builder.ins().icmp(IntCC::SignedLessThan, counter, list_len);
        builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        let get_id = self.registry.runtime_fns["nsl_list_get"].0;
        let get_ref = self.module.declare_func_in_func(get_id, builder.func);
        let counter = builder.use_var(counter_var);
        let call = builder.ins().call(get_ref, &[list_val, counter]);
        let elem = builder.inst_results(call)[0];

        // Bind element to pattern variable(s)
        match &pattern.kind {
            PatternKind::Ident(sym) => {
                let (var, _) = state.variables[sym];
                builder.def_var(var, elem);
            }
            PatternKind::Tuple(sub_patterns) | PatternKind::List(sub_patterns) => {
                // elem is a tuple/list (NslList ptr) — destructure with Rest support
                let rest_positions: Vec<usize> = sub_patterns
                    .iter()
                    .enumerate()
                    .filter_map(|(i, pat)| matches!(pat.kind, PatternKind::Rest(_)).then_some(i))
                    .collect();
                if rest_positions.len() > 1 {
                    return Err(CodegenError::new(
                        "multiple rest patterns in a single destructuring pattern are not supported",
                    ));
                }
                let rest_pos = rest_positions.first().copied();
                let elem_len = if rest_pos.is_some() {
                    Some(self.compile_call_by_name(builder, "nsl_list_len", &[elem])?)
                } else {
                    None
                };

                for (i, sub_pat) in sub_patterns.iter().enumerate() {
                    match &sub_pat.kind {
                        PatternKind::Ident(sym) => {
                            let idx = match rest_pos {
                                Some(rp) if i > rp => {
                                    // After rest: index from end
                                    let trailing = builder
                                        .ins()
                                        .iconst(cl_types::I64, (sub_patterns.len() - i) as i64);
                                    builder.ins().isub(elem_len.unwrap(), trailing)
                                }
                                _ => builder.ins().iconst(cl_types::I64, i as i64),
                            };
                            let inner_get_ref =
                                self.module.declare_func_in_func(get_id, builder.func);
                            let call = builder.ins().call(inner_get_ref, &[elem, idx]);
                            let sub_elem = builder.inst_results(call)[0];
                            let (var, _) = state.variables[sym];
                            builder.def_var(var, sub_elem);
                        }
                        PatternKind::Rest(rest_sym) => {
                            let lo = builder.ins().iconst(cl_types::I64, i as i64);
                            let hi = if i + 1 < sub_patterns.len() {
                                let trailing = builder.ins().iconst(
                                    cl_types::I64,
                                    sub_patterns.len().saturating_sub(i + 1) as i64,
                                );
                                builder.ins().isub(elem_len.unwrap(), trailing)
                            } else {
                                elem_len.unwrap()
                            };
                            let step = builder.ins().iconst(cl_types::I64, 1);
                            let rest_val = self.compile_call_by_name(
                                builder,
                                "nsl_list_slice",
                                &[elem, lo, hi, step],
                            )?;
                            if let Some(sym) = rest_sym {
                                let (var, _) = state.variables[sym];
                                builder.def_var(var, rest_val);
                            } else {
                                self.compile_call_by_name(builder, "nsl_list_free", &[rest_val])?;
                            }
                        }
                        PatternKind::Wildcard => {}
                        _ => {}
                    }
                }
            }
            _ => unreachable!(),
        }

        // continue jumps to increment_block (not header) so counter is incremented
        state
            .cleanup
            .temp_scope_stack
            .push(state.cleanup.tensor_temporaries.len());
        state.loop_stack.push(LoopContext {
            continue_block: increment_block,
            exit_block,
            batch_var: None,
        });
        state.push_fn_binding_scope();
        for s in &body.stmts {
            self.compile_stmt(builder, state, s)?;
        }
        state.pop_fn_binding_scope();
        state.loop_stack.pop();
        for sym in &predecl_syms {
            state.eltls_loop_predeclared.remove(sym);
        }

        let current = state.current_block.unwrap_or(body_block);
        if !is_block_filled(builder, current) {
            self.cleanup_loop_scope(builder, state);
            builder.ins().jump(increment_block, &[]);
        } else {
            state.cleanup.temp_scope_stack.pop();
        }

        // Increment block: counter++ then jump to header
        builder.switch_to_block(increment_block);
        builder.seal_block(increment_block);
        state.current_block = Some(increment_block);
        let counter = builder.use_var(counter_var);
        let one = builder.ins().iconst(cl_types::I64, 1);
        let next = builder.ins().iadd(counter, one);
        builder.def_var(counter_var, next);
        builder.ins().jump(header_block, &[]);

        builder.seal_block(header_block);
        builder.switch_to_block(exit_block);
        builder.seal_block(exit_block);
        state.current_block = Some(exit_block);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn compile_for_model_array(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        pattern: &nsl_ast::pattern::Pattern,
        iterable: &nsl_ast::expr::Expr,
        body: &nsl_ast::stmt::Block,
        element_model: nsl_ast::Symbol,
        size: i64,
    ) -> Result<(), CodegenError> {
        // Compile iterable to get base address of the array
        let base_val = self.compile_expr(builder, state, iterable)?;

        // Declare loop variable
        let loop_var_sym = match &pattern.kind {
            PatternKind::Ident(sym) => *sym,
            _ => {
                return Err(CodegenError::new(
                    "only ident patterns supported in model array for-loops",
                ))
            }
        };
        let zero = builder.ins().iconst(cl_types::I64, 0);
        let elem_var = builder.declare_var(cl_types::I64);
        builder.def_var(elem_var, zero);
        state
            .variables
            .insert(loop_var_sym, (elem_var, cl_types::I64));

        // Register the loop variable's model type for method dispatch
        let model_name = self.resolve_sym(element_model).to_string();
        self.models.model_var_types.insert(loop_var_sym, model_name);

        // Counter variable
        let counter_var = builder.declare_var(cl_types::I64);
        builder.def_var(counter_var, zero);

        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let increment_block = builder.create_block();
        let exit_block = builder.create_block();

        // ELTLS Task 16.1: pre-declare top-level let-ident symbols from body.
        // Pre-loop block on purpose — see compile_while.
        // Loop-carried locals seeded from a borrow (`let h = x`) are vetoed
        // by the flow-insensitive non_owning_symbols set, so their single
        // generated rebind-free site never fires. Give them their own
        // reference first — see materialize_non_owning_aliases_before_loop.
        self.materialize_non_owning_aliases_before_loop(builder, state, body, Some(pattern))?;
        let predecl_syms = self.eltls_predeclare_loop_lets_checked(builder, state, body);

        builder.ins().jump(header_block, &[]);

        // Header: check i < size
        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let counter = builder.use_var(counter_var);
        let limit = builder.ins().iconst(cl_types::I64, size);
        let cond = builder.ins().icmp(IntCC::SignedLessThan, counter, limit);
        builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        // Body: load element pointer from base_val + i*8
        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);
        let counter = builder.use_var(counter_var);
        let eight = builder.ins().iconst(cl_types::I64, 8);
        let elem_offset = builder.ins().imul(counter, eight);
        let addr = builder.ins().iadd(base_val, elem_offset);
        let elem_ptr = builder
            .ins()
            .load(cl_types::I64, MemFlagsData::trusted(), addr, 0);
        builder.def_var(elem_var, elem_ptr);

        // Compile body statements
        state
            .cleanup
            .temp_scope_stack
            .push(state.cleanup.tensor_temporaries.len());
        state.loop_stack.push(LoopContext {
            continue_block: increment_block,
            exit_block,
            batch_var: None,
        });
        state.push_fn_binding_scope();
        for s in &body.stmts {
            self.compile_stmt(builder, state, s)?;
        }
        state.pop_fn_binding_scope();
        state.loop_stack.pop();
        for sym in &predecl_syms {
            state.eltls_loop_predeclared.remove(sym);
        }

        let current = state.current_block.unwrap_or(body_block);
        if !is_block_filled(builder, current) {
            self.cleanup_loop_scope(builder, state);
            builder.ins().jump(increment_block, &[]);
        } else {
            state.cleanup.temp_scope_stack.pop();
        }

        // Increment: counter++ then jump to header
        builder.switch_to_block(increment_block);
        builder.seal_block(increment_block);
        state.current_block = Some(increment_block);
        let counter = builder.use_var(counter_var);
        let one = builder.ins().iconst(cl_types::I64, 1);
        let next = builder.ins().iadd(counter, one);
        builder.def_var(counter_var, next);
        builder.ins().jump(header_block, &[]);

        builder.seal_block(header_block);
        builder.switch_to_block(exit_block);
        builder.seal_block(exit_block);
        state.current_block = Some(exit_block);
        Ok(())
    }

    fn expr_is_dataloader_handle(&self, state: &FuncState, expr: &nsl_ast::expr::Expr) -> bool {
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

    fn expr_is_borrowed_batch_handle(&self, state: &FuncState, expr: &nsl_ast::expr::Expr) -> bool {
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
    fn is_dataloader_iterable(&self, state: &FuncState, iterable: &nsl_ast::expr::Expr) -> bool {
        self.expr_is_dataloader_handle(state, iterable)
    }

    // ── DataLoader for-loop ──────────────────────────────────────────

    fn compile_for_dataloader(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        pattern: &nsl_ast::pattern::Pattern,
        iterable: &nsl_ast::expr::Expr,
        body: &nsl_ast::stmt::Block,
    ) -> Result<(), CodegenError> {
        let dl_val = self.compile_expr(builder, state, iterable)?;

        // Extract loop variable symbol (must be simple ident)
        let loop_var_sym = match &pattern.kind {
            PatternKind::Ident(sym) => *sym,
            _ => {
                return Err(CodegenError::new(
                    "only ident patterns supported in DataLoader for-loops",
                ))
            }
        };

        // Declare cranelift variable for the batch pointer
        let batch_var = builder.declare_var(cl_types::I64);
        let zero = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(batch_var, zero);
        let prev_binding = state
            .variables
            .insert(loop_var_sym, (batch_var, cl_types::I64));
        let prev_var_type = state.variable_types.get(&loop_var_sym).cloned();
        let prev_loader_symbol = state.dataloader_symbols.contains(&loop_var_sym);
        let prev_borrowed_symbol = state.borrowed_batch_symbols.contains(&loop_var_sym);

        // Create blocks: header, body, cleanup, break_exit, exhausted_exit, exit
        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let cleanup_block = builder.create_block();
        let break_exit_block = builder.create_block();
        let exhausted_exit_block = builder.create_block();
        let exit_block = builder.create_block();

        // ELTLS Task 16.1: pre-declare top-level let-ident symbols from body.
        // THIS IS THE PRIMARY FIX FOR THE TRAINING-LOOP LEAK:
        //   for batch in dataloader:
        //       let y = model(batch)     # previously leaked y each iteration
        //       let loss = loss_fn(y, batch["labels"])
        //       ...
        // By pre-declaring y and loss in the pre-loop scope with zero init,
        // each subsequent rebind is detected as a reassignment and
        // eltls_clear_old_slot fires nsl_tensor_free_if_valid on the
        // previous iteration's tensor. The zero-def MUST live in the
        // pre-loop block — inside the body it re-zeroes the slot every
        // iteration and the free only ever sees 0.
        // Loop-carried locals seeded from a borrow (`let h = x`) are vetoed
        // by the flow-insensitive non_owning_symbols set, so their single
        // generated rebind-free site never fires. Give them their own
        // reference first — see materialize_non_owning_aliases_before_loop.
        self.materialize_non_owning_aliases_before_loop(builder, state, body, Some(pattern))?;
        let predecl_syms = self.eltls_predeclare_loop_lets_checked(builder, state, body);

        builder.ins().jump(header_block, &[]);

        // Header: call nsl_dataloader_next_batch, branch on null
        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let batch_ptr =
            self.compile_call_by_name(builder, "nsl_dataloader_next_batch", &[dl_val])?;
        builder.def_var(batch_var, batch_ptr);
        let is_null = builder.ins().icmp_imm_s(IntCC::Equal, batch_ptr, 0);
        builder
            .ins()
            .brif(is_null, exhausted_exit_block, &[], body_block, &[]);

        // Body: compile loop body statements
        // break → break_exit_block (frees batch, then stops DL)
        // continue → cleanup_block (frees batch, loops back)
        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        // Rely on codegen-level tensor_temporaries for per-statement cleanup.
        // Do NOT use scope_begin/scope_end — it double-frees tensors that are
        // already freed by free_tensor_temporaries in called functions.
        state
            .cleanup
            .temp_scope_stack
            .push(state.cleanup.tensor_temporaries.len());
        state.borrowed_batch_symbols.insert(loop_var_sym);
        state.loop_stack.push(LoopContext {
            continue_block: cleanup_block,
            exit_block: break_exit_block,
            batch_var: Some(batch_var),
        });
        state.push_fn_binding_scope();
        for s in &body.stmts {
            self.compile_stmt(builder, state, s)?;
        }
        state.pop_fn_binding_scope();
        state.loop_stack.pop();
        state.borrowed_batch_symbols.remove(&loop_var_sym);
        for sym in &predecl_syms {
            state.eltls_loop_predeclared.remove(sym);
        }

        let current = state.current_block.unwrap_or(body_block);
        if !is_block_filled(builder, current) {
            self.cleanup_loop_scope(builder, state);
            builder.ins().jump(cleanup_block, &[]);
        } else {
            state.cleanup.temp_scope_stack.pop();
        }

        // Cleanup: free batch dict, loop back
        builder.switch_to_block(cleanup_block);
        builder.seal_block(cleanup_block);
        state.current_block = Some(cleanup_block);
        let batch_to_free = builder.use_var(batch_var);
        self.compile_call_by_name(builder, "nsl_dict_free_tensor_values", &[batch_to_free])?;
        builder.ins().jump(header_block, &[]);

        // Break exit: end tensor scope, free the current batch dict, then stop the DataLoader
        builder.switch_to_block(break_exit_block);
        builder.seal_block(break_exit_block);
        state.current_block = Some(break_exit_block);
        let batch_to_free = builder.use_var(batch_var);
        self.compile_call_by_name(builder, "nsl_dict_free_tensor_values", &[batch_to_free])?;
        self.compile_call_by_name(builder, "nsl_dataloader_reset", &[dl_val])?;
        builder.ins().jump(exit_block, &[]);

        // Exit on natural exhaustion: reset the dataloader for potential next epoch
        builder.seal_block(header_block);
        builder.switch_to_block(exhausted_exit_block);
        builder.seal_block(exhausted_exit_block);
        state.current_block = Some(exhausted_exit_block);
        self.compile_call_by_name(builder, "nsl_dataloader_reset", &[dl_val])?;
        builder.ins().jump(exit_block, &[]);

        builder.seal_block(exit_block);
        builder.switch_to_block(exit_block);
        state.current_block = Some(exit_block);

        if let Some(prev_binding) = prev_binding {
            state.variables.insert(loop_var_sym, prev_binding);
        } else {
            state.variables.remove(&loop_var_sym);
        }
        if let Some(prev_var_type) = prev_var_type {
            state.variable_types.insert(loop_var_sym, prev_var_type);
        } else {
            state.variable_types.remove(&loop_var_sym);
        }
        if prev_loader_symbol {
            state.dataloader_symbols.insert(loop_var_sym);
        } else {
            state.dataloader_symbols.remove(&loop_var_sym);
        }
        if prev_borrowed_symbol {
            state.borrowed_batch_symbols.insert(loop_var_sym);
        } else {
            state.borrowed_batch_symbols.remove(&loop_var_sym);
        }

        Ok(())
    }

    // ── Match/case ──────────────────────────────────────────────────

    fn compile_match(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        subject: &nsl_ast::expr::Expr,
        arms: &[nsl_ast::expr::MatchArm],
    ) -> Result<(), CodegenError> {
        let subject_val = self.compile_expr(builder, state, subject)?;
        let merge_block = builder.create_block();

        let mut remaining_arms: Vec<_> = arms.iter().collect();
        while !remaining_arms.is_empty() {
            let arm = remaining_arms.remove(0);
            let is_last = remaining_arms.is_empty();

            match &arm.pattern.kind {
                PatternKind::Wildcard => {
                    // Default arm — always taken
                    state.flags.conditional_depth += 1;
                    state.push_fn_binding_scope();
                    for s in &arm.body.stmts {
                        self.compile_stmt(builder, state, s)?;
                    }
                    state.pop_fn_binding_scope();
                    state.flags.conditional_depth -= 1;
                    if let Some(block) = state.current_block
                        && !is_block_filled(builder, block)
                    {
                        builder.ins().jump(merge_block, &[]);
                    }
                    break;
                }
                PatternKind::Ident(sym) => {
                    // Could be an enum variant or a binding
                    let name = self.resolve_sym(*sym).to_string();
                    if let Some(tag) = self.lookup_enum_variant_tag(&name) {
                        // Enum variant comparison
                        let tag_val = builder.ins().iconst(cl_types::I64, tag);
                        let cmp = builder.ins().icmp(IntCC::Equal, subject_val, tag_val);
                        let arm_block = builder.create_block();
                        let next_block = if is_last {
                            merge_block
                        } else {
                            builder.create_block()
                        };
                        builder.ins().brif(cmp, arm_block, &[], next_block, &[]);

                        builder.switch_to_block(arm_block);
                        builder.seal_block(arm_block);
                        state.current_block = Some(arm_block);
                        state.flags.conditional_depth += 1;
                        state.push_fn_binding_scope();
                        for s in &arm.body.stmts {
                            self.compile_stmt(builder, state, s)?;
                        }
                        state.pop_fn_binding_scope();
                        state.flags.conditional_depth -= 1;
                        let current = state.current_block.unwrap_or(arm_block);
                        if !is_block_filled(builder, current) {
                            builder.ins().jump(merge_block, &[]);
                        }

                        if !is_last {
                            builder.switch_to_block(next_block);
                            builder.seal_block(next_block);
                            state.current_block = Some(next_block);
                        }
                    } else {
                        // Binding — bind subject to variable, always taken
                        let var = builder.declare_var(cl_types::I64);
                        builder.def_var(var, subject_val);
                        state.variables.insert(*sym, (var, cl_types::I64));
                        state.flags.conditional_depth += 1;
                        state.push_fn_binding_scope();
                        for s in &arm.body.stmts {
                            self.compile_stmt(builder, state, s)?;
                        }
                        state.pop_fn_binding_scope();
                        state.flags.conditional_depth -= 1;
                        if let Some(block) = state.current_block
                            && !is_block_filled(builder, block)
                        {
                            builder.ins().jump(merge_block, &[]);
                        }
                        break;
                    }
                }
                PatternKind::Literal(lit_expr) => {
                    let lit_val = self.compile_expr(builder, state, lit_expr)?;
                    let lit_type = self.node_type(lit_expr.id).clone();
                    let cmp = if is_float_type(&lit_type) {
                        builder.ins().fcmp(
                            cranelift_codegen::ir::condcodes::FloatCC::Equal,
                            subject_val,
                            lit_val,
                        )
                    } else {
                        builder.ins().icmp(IntCC::Equal, subject_val, lit_val)
                    };
                    let arm_block = builder.create_block();
                    let next_block = if is_last {
                        merge_block
                    } else {
                        builder.create_block()
                    };
                    builder.ins().brif(cmp, arm_block, &[], next_block, &[]);

                    builder.switch_to_block(arm_block);
                    builder.seal_block(arm_block);
                    state.current_block = Some(arm_block);
                    state.flags.conditional_depth += 1;
                    state.push_fn_binding_scope();
                    for s in &arm.body.stmts {
                        self.compile_stmt(builder, state, s)?;
                    }
                    state.pop_fn_binding_scope();
                    state.flags.conditional_depth -= 1;
                    let current = state.current_block.unwrap_or(arm_block);
                    if !is_block_filled(builder, current) {
                        builder.ins().jump(merge_block, &[]);
                    }

                    if !is_last {
                        builder.switch_to_block(next_block);
                        builder.seal_block(next_block);
                        state.current_block = Some(next_block);
                    }
                }
                PatternKind::Constructor { path, .. } => {
                    // Enum variant via path: e.g., Activation.ReLU → check tag
                    let variant_name = if !path.is_empty() {
                        self.resolve_sym(*path.last().unwrap()).to_string()
                    } else {
                        return Err(CodegenError::new("empty constructor path in match"));
                    };
                    if let Some(tag) = self.lookup_enum_variant_tag(&variant_name) {
                        let tag_val = builder.ins().iconst(cl_types::I64, tag);
                        let cmp = builder.ins().icmp(IntCC::Equal, subject_val, tag_val);
                        let arm_block = builder.create_block();
                        let next_block = if is_last {
                            merge_block
                        } else {
                            builder.create_block()
                        };
                        builder.ins().brif(cmp, arm_block, &[], next_block, &[]);

                        builder.switch_to_block(arm_block);
                        builder.seal_block(arm_block);
                        state.current_block = Some(arm_block);
                        state.flags.conditional_depth += 1;
                        state.push_fn_binding_scope();
                        for s in &arm.body.stmts {
                            self.compile_stmt(builder, state, s)?;
                        }
                        state.pop_fn_binding_scope();
                        state.flags.conditional_depth -= 1;
                        let current = state.current_block.unwrap_or(arm_block);
                        if !is_block_filled(builder, current) {
                            builder.ins().jump(merge_block, &[]);
                        }

                        if !is_last {
                            builder.switch_to_block(next_block);
                            builder.seal_block(next_block);
                            state.current_block = Some(next_block);
                        }
                    } else {
                        return Err(CodegenError::new(format!(
                            "unknown enum variant '{variant_name}' in match"
                        )));
                    }
                }
                _ => return Err(CodegenError::new("unsupported pattern in match arm")),
            }
        }

        // If we didn't break (no wildcard/binding), need to jump to merge from final else
        if let Some(block) = state.current_block
            && block != merge_block && !is_block_filled(builder, block)
        {
            builder.ins().jump(merge_block, &[]);
        }

        builder.switch_to_block(merge_block);
        builder.seal_block(merge_block);
        state.current_block = Some(merge_block);
        Ok(())
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
            if self.compile_options.layerwise_accum {
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
            if self.compile_options.cuda_graphs {
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
        if self.compile_options.cuda_graphs {
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
                    if self.compile_options.training_reference {
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
                .with_checkpoint_policies(if self.compile_options.training_reference {
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
                            nsl_runtime::nsl_log!(INFO, "wggo", 
                                "[wggo] consumed pre-solved plan \
                                     (graph fingerprint match)"
                            );
                            Some(pre.plan.clone())
                        } else {
                            nsl_runtime::nsl_log!(WARN, "wggo", 
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
                            nsl_runtime::nsl_log!(INFO, "wggo", 
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
                            nsl_runtime::nsl_log!(INFO, "codegen", "{}", plan.render_report());
                        } else {
                            nsl_runtime::nsl_log!(INFO, "wggo", "[wggo] {}", plan.summary());
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
                            nsl_runtime::nsl_log!(INFO, "prune", 
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
                                    PackingVerdict::Consumed { kernel } => nsl_runtime::nsl_log!(INFO, "pca", 
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
                                        nsl_runtime::nsl_log!(INFO, "pca", 
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
                            &model_type_name,
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
                if let Some(ref applied_plan) = wggo_applied {
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
                if let Some(ref applied) = wggo_applied {
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
                                    &param_paths,
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
                                    &param_paths,
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
                    match (&cpdt_moment_lists_consumed, &fresh_lists) {
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
                } else if let Some(ref consumed) = cpdt_moment_lists_consumed {
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
                                    &param_paths,
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
                // Task 6: render any override-rejected diagnostics to stderr so
                // the Phase 3 decision explainer and the user can see which
                // WGGO-requested ranks were adjusted.  Format matches the CSHA
                // renderer so both can be parsed uniformly.
                if let Some(ref plan) = wrga_plan {
                    for diag in &plan.override_diagnostics {
                        let reason_str = match &diag.reason {
                            crate::wggo_overrides::OverrideRejectReason::RankClampedToBounds {
                                r_min,
                                r_max,
                            } => format!("rank_out_of_bounds_[{r_min},{r_max}]"),
                            crate::wggo_overrides::OverrideRejectReason::RankForbiddenByWggo => {
                                "rank_forbidden_by_wggo".to_string()
                            }
                            crate::wggo_overrides::OverrideRejectReason::BudgetExceededDowngraded {
                                original_rank,
                                final_rank,
                            } => format!("budget_exceeded_{original_rank}_to_{final_rank}"),
                            crate::wggo_overrides::OverrideRejectReason::AdapterSiteOutsidePlacement {
                                placement,
                            } => format!("site_outside_placement_[{placement}]"),
                            other => format!("{:?}", other),
                        };
                        nsl_runtime::nsl_log!(INFO, "wrga", 
                            "[wrga] layer:{} wggo-override-rejected requested={} applied={} reason={}",
                            diag.layer_index, diag.requested, diag.applied, reason_str
                        );
                    }
                }
                // B.2.1 Task 2.5: materialise adapter tensors into the model
                // struct's side-table slot now that the plan is known. Task 2
                // reserved the slot + zero-initialised it; this call allocates
                // the heap table, fills it with freshly-initialised tensors
                // (LoRA-A randn-scaled, LoRA-B zeros, IA³ ones, gate zeros),
                // and writes the table pointer into the reserved slot. The
                // iteration order here MUST match `adapter_field_index` in
                // `expr/access.rs`.
                // B.2.1 Task 5.5: prefer the train-block plan only when it
                // has decorated placements; otherwise fall back to the
                // prescan plan already stashed on the compiler (which has
                // the @adapter decorator info attached to a single synthetic
                // placement). Without this, build configs like
                // `@adapter(target=["Toy.w"])` would skip init entirely.
                let init_plan = {
                    let train_has_decorated = wrga_plan
                        .as_ref()
                        .map(|p| {
                            p.placements
                                .iter()
                                .any(|pl| pl.decorator_kind.is_some())
                        })
                        .unwrap_or(false);
                    if train_has_decorated {
                        wrga_plan.clone()
                    } else {
                        self.bus.adapter_prescan_plan().cloned()
                    }
                };
                if let Some(plan_ref) = init_plan.as_ref() {
                    crate::wrga_adapter_init::emit_adapter_init_sidetable(
                        self,
                        builder,
                        state,
                        model_ptr,
                        &model_type_name,
                        plan_ref,
                    )?;
                }

                // WRGA B.3.2 Option 3: resolve any named adapter params
                // that the pre-init pass above couldn't load (because the
                // side-table pointer was still zero). The init just
                // populated it, so MemberAccess loads on the synth adapter
                // field names now return real tensor pointers — emit
                // those loads here, after the init instructions in IR
                // order, so they execute with a valid table pointer.
                for (compound_name, vid) in extractor.named_param_var_ids() {
                    if primal_vars.contains_key(vid) {
                        continue;
                    }
                    let parts: Vec<&str> = compound_name.split('.').collect();
                    if parts.len() < 2 {
                        continue;
                    }
                    let last = parts[parts.len() - 1];
                    if !crate::expr::access::is_synthesized_adapter_field_name(last) {
                        continue;
                    }
                    let mut current_ptr = model_ptr;
                    let mut current_type_name = model_type_name.clone();
                    let mut current_layout = layout.clone();
                    let mut ok = true;
                    for part in &parts[1..parts.len() - 1] {
                        if let Ok(array_idx) = part.parse::<usize>() {
                            current_ptr = builder.ins().load(
                                cl_types::I64,
                                cranelift_codegen::ir::MemFlagsData::trusted(),
                                current_ptr,
                                (array_idx * 8) as i32,
                            );
                            continue;
                        }
                        if let Some(field) =
                            current_layout.fields.iter().find(|f| &f.name == part)
                        {
                            let offset = field.offset as i32;
                            let field_val = builder.ins().load(
                                field.cl_type,
                                cranelift_codegen::ir::MemFlagsData::trusted(),
                                current_ptr,
                                offset,
                            );
                            current_ptr = field_val;
                            let field_type = self
                                .models
                                .model_field_types
                                .get(&current_type_name)
                                .and_then(|ft| ft.get(part.to_owned()))
                                .cloned();
                            if let Some(ft) = field_type {
                                if let Some(inner_layout) = self.types.struct_layouts.get(&ft) {
                                    current_layout = inner_layout.clone();
                                    current_type_name = ft;
                                } else {
                                    ok = false;
                                    break;
                                }
                            } else {
                                ok = false;
                                break;
                            }
                        } else {
                            ok = false;
                            break;
                        }
                    }
                    if !ok {
                        continue;
                    }
                    if let Some(slot_off) = current_layout.adapter_sidetable_offset
                        && let Some(index) = self.adapter_field_index(&current_type_name, last)
                    {
                        let table_ptr = builder.ins().load(
                            cl_types::I64,
                            cranelift_codegen::ir::MemFlagsData::trusted(),
                            current_ptr,
                            slot_off as i32,
                        );
                        let byte_off = (index * 8) as i32;
                        let tensor_ptr = builder.ins().load(
                            cl_types::I64,
                            cranelift_codegen::ir::MemFlagsData::trusted(),
                            table_ptr,
                            byte_off,
                        );
                        primal_vars.insert(*vid, tensor_ptr);
                    }
                }
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
                let effective_primal: crate::wengert::WengertList = match &wrga_plan {
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
                if let Some(plan) = &wrga_plan {
                    if self.compile_options.wrga_fold_allocations {
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
                let ccr_selective_decorated = !self.compile_options.training_reference
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

                // 6a. Task 4: WRGA backward-live filter — drop adjoint ops
                // that the WRGA prune pass proved to be on frozen branches.
                if let Some(plan) = &wrga_plan {
                    adjoint.ops = crate::source_ad::eliminate_by_backward_live(
                        &adjoint.ops,
                        &plan.prune.backward_live,
                        generator.adjoint_vars_map(),
                    );
                }

                // 6b. Dead gradient elimination: prune adjoint ops not needed
                // by any parameter gradient. This removes ghost VarId chains
                // from non-differentiable ops (shape, subscript, list) that
                // would cascade skip in the lowerer.
                //
                // `adjoint_needed` (the trainable parameter-gradient adjoint
                // VarIds) is hoisted here so the P0.2 gradient-integrity guard
                // below can reuse it to classify LIVE vs dead/ghost adjoint ops
                // over the FINAL (post-CCR) op list.
                let adjoint_needed: std::collections::HashSet<crate::wengert::VarId> = {
                    let named_params = extractor.named_param_var_ids();
                    named_params
                        .iter()
                        .filter(|(name, _)| self.is_trainable_param_name(name))
                        .filter_map(|(_, vid)| generator.adjoint_of(*vid))
                        .collect()
                };
                if !adjoint_needed.is_empty() {
                    adjoint.ops =
                        crate::source_ad::eliminate_dead_gradients(&adjoint.ops, &adjoint_needed);
                }
                // P5 item 20 slice B: fuse SwiGLU gate-gradient pairs
                // (bit-exact — see fuse_swiglu_gate_backward).
                crate::source_ad::fuse_swiglu_gate_backward(&mut adjoint.ops, &adjoint_needed);
                // P5 slice C: fold residual-gradient accumulates into fused
                // RMSNorm dx ops (bit-exact; no-op unless
                // --fuse-rmsnorm-backward emitted them).
                let norm_res_folds =
                    crate::source_ad::fuse_rmsnorm_dx_residual(&mut adjoint.ops, &adjoint_needed);
                if norm_res_folds > 0 {
                    nsl_runtime::nsl_log!(INFO, "fuse", "[fuse] rmsnorm dx+residual folds: {norm_res_folds}");
                }
                // MFU campaign C2: the RoPE backward fold is generation-time
                // (rotate_half_neg emitted instead of rotate_half + Neg);
                // count the ops here so gates have an anti-vacuity witness.
                let rope_folds = adjoint
                    .ops
                    .iter()
                    .filter(|op| {
                        matches!(&op.op,
                            crate::wengert::PrimalOp::Passthrough(n) if n == "rotate_half_neg")
                    })
                    .count();
                if rope_folds > 0 {
                    nsl_runtime::nsl_log!(INFO, "fuse", "[fuse] rope backward folds: {rope_folds}");
                }
                // MFU campaign C3: generic elementwise-chain fusion + the
                // standalone scalar-immediate sweep. Chain fuser first so
                // chains absorb Constants as immediates; the sweep catches
                // standalone leftovers (reversed order would turn const
                // sites into Passthrough barriers and starve chains). Runs
                // after the specialized folds above so they claim their
                // better patterns first; skipped under --layerwise-accum
                // (the CSLA range partition is positional over this tape —
                // v1 defers, see ew_chain_fusion module docs).
                if !self.compile_options.layerwise_accum {
                    let ew_stats = crate::ew_chain_fusion::run_backward_ew_fusion(
                        &mut adjoint.ops,
                        &adjoint_needed,
                        &adjoint.var_types,
                    );
                    if ew_stats.chains > 0 {
                        nsl_runtime::nsl_log!(INFO, "fuse", 
                            "[fuse] elementwise backward chains: {} ({} device ops elided, \
                             {} reduces absorbed, {} imms baked)",
                            ew_stats.chains,
                            ew_stats.device_ops_elided,
                            ew_stats.reduces_absorbed,
                            ew_stats.imms_baked
                        );
                    }
                    let scalar_imms = crate::ew_chain_fusion::rewrite_scalar_immediates(
                        &mut adjoint.ops,
                        &adjoint_needed,
                        &adjoint.var_types,
                    );
                    if scalar_imms > 0 {
                        nsl_runtime::nsl_log!(INFO, "fuse", "[fuse] scalar immediates: {scalar_imms}");
                    }
                    if (ew_stats.chains > 0 || scalar_imms > 0)
                        && std::env::var("NSL_PROFILE_ADJOINT").is_ok()
                    {
                        nsl_runtime::nsl_log!(INFO, "adjoint-profile", 
                            "[adjoint-profile] post-fusion: {} backward ops:",
                            adjoint.ops.len()
                        );
                        for (k, c) in crate::ew_chain_fusion::histogram(&adjoint.ops) {
                            nsl_runtime::nsl_log!(INFO, "adjoint-profile", "[adjoint-profile]   {c:>5}  {k}");
                        }
                    }
                } else {
                    nsl_runtime::nsl_log!(WARN, "fuse", "[fuse] elementwise backward fusion skipped (--layerwise-accum)");
                }

                // 6b.5 CSLA (Milestone B): report the layerwise-accumulation
                // schedule when NSL_CSLA_REPORT=1. Pure analysis over the final
                // adjoint — no codegen change. Element counts are left
                // unquantified here (Stage-2 wires the memory planner's shapes);
                // the layer grouping + tied/cross-layer classification is the
                // correctness-relevant part.
                if std::env::var("NSL_CSLA_REPORT").ok().as_deref() == Some("1") {
                    let params: Vec<(String, crate::wengert::VarId)> = extractor
                        .named_param_var_ids()
                        .iter()
                        .filter(|(name, _)| self.is_trainable_param_name(name))
                        .map(|(n, v)| (n.clone(), *v))
                        .collect();
                    let plan = crate::layerwise::analyze(&adjoint, &params, &|_| None);
                    nsl_runtime::nsl_log!(INFO, "csla", "[csla]\n{}", plan.render_report("  "));
                }

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
                // Reuses the M36 interference/BFD engine (transient_arena.rs)
                // over the *final* forward+adjoint tape — post-CCR-splice and
                // post-adjoint-last-use-frees, so FreeTensor markers bound each
                // interval exactly. This is the backward+forward transient
                // surface the M36 slab planner (AST, forward-only) never sees.
                // Pure analysis; no codegen change. Gated by --memory-report or
                // NSL_ARENA_REPORT=1.
                // Stage-2A shape hints, shared by the arena report and the
                // CSLA layerwise calibration below. Three SOUND layers, in
                // priority order:
                //   1. semantic-typed shapes (annotated Tensor<[..]> values
                //      — near-empty on today's corpus since model fields are
                //      typed from annotations only);
                //   2. initializer-derived model-field dims for trainable
                //      params (the unique-field-name bridge; covers the
                //      dominant `randn([64, 128]) * 0.15` idiom);
                //   3. each sized primal mirrored onto its adjoint
                //      accumulator (an adjoint has its primal's shape by
                //      construction) — this is what sizes the backward's
                //      gradient transients, the surface the arena exists
                //      to place.
                // Symbolic/computed dims stay unsized; nothing is guessed.
                let arena_place_on = self.compile_options.transient_arena;
                let arena_report_on = self.compile_options.memory_report
                    || arena_place_on
                    || std::env::var("NSL_ARENA_REPORT").ok().as_deref() == Some("1");
                let elem_hints: std::collections::HashMap<crate::wengert::VarId, u64> =
                    if arena_report_on || csla_active {
                        let mut hints = crate::profiling::captures::elem_hints_from_var_nodes(
                            extractor.var_nodes(),
                            self.type_map,
                        );
                        let field_elems = self.models.unique_field_elems();
                        for (name, vid) in extractor.named_param_var_ids() {
                            if hints.contains_key(vid) {
                                continue;
                            }
                            let leaf = name.rsplit('.').next().unwrap_or(name);
                            if let Some(&e) = field_elems.get(leaf) {
                                hints.insert(*vid, e);
                            }
                        }
                        // Mirror ONLY adjoint accumulators with a UNIQUE
                        // primal preimage. Add/Sub rules alias the OUTPUT's
                        // adjoint onto both operands (Identity — no reduce
                        // op), so a shared accumulator carries the OUTPUT's
                        // shape and mirroring it would mis-size a broadcast
                        // operand's grad. Reducing rules emit a dedicated
                        // reduce_to_shape var per operand, which is exactly
                        // what a unique preimage certifies.
                        let mut preimage: std::collections::HashMap<crate::wengert::VarId, u32> =
                            Default::default();
                        for a in generator.adjoint_vars_map().values() {
                            *preimage.entry(*a).or_default() += 1;
                        }
                        let mirrored: Vec<(crate::wengert::VarId, u64)> = generator
                            .adjoint_vars_map()
                            .iter()
                            .filter(|(_, a)| preimage.get(a) == Some(&1))
                            .filter_map(|(p, a)| hints.get(p).map(|&e| (*a, e)))
                            .collect();
                        for (a, e) in mirrored {
                            hints.entry(a).or_insert(e);
                        }
                        hints
                    } else {
                        Default::default()
                    };

                if arena_report_on {
                    // Stage-2A: partially quantified — sized transients get
                    // real bytes (a lower bound on the full arena), the rest
                    // still report as concurrency-only. Param-gradient
                    // accumulators and the loss ESCAPE the tape (read by the
                    // optimizer emission / callbacks, never by a tape op) —
                    // without the escape pin, last-use liveness gives them
                    // point intervals and BFD time-shares every gradient in
                    // one slot, an illegal aliasing that fakes the savings.
                    // Stage-2B assigns offsets.
                    let param_vids: std::collections::HashSet<crate::wengert::VarId> = extractor
                        .named_param_var_ids()
                        .iter()
                        .map(|(_, v)| *v)
                        .collect();
                    let mut tape_escaping: std::collections::HashSet<crate::wengert::VarId> = generator
                        .adjoint_vars_map()
                        .iter()
                        .filter(|(p, _)| param_vids.contains(p))
                        .map(|(_, a)| *a)
                        .collect();
                    tape_escaping.insert(effective_primal.output);
                    // Stage-2B: sizes the hint bridge cannot reach. Shapes
                    // first — dims propagate through matmul, which is where
                    // the numel-only pass stopped (it sized 0 of 1321
                    // transients on coder50m: every backward elementwise
                    // chain sits downstream of a matmul, and a matmul's
                    // output numel is a function of the SHAPES). The numel
                    // pass still runs last, as a fallback for values whose
                    // dims die at a runtime-shaped op but whose count
                    // survives.
                    let mut dim_seeds: std::collections::HashMap<
                        crate::wengert::VarId,
                        Vec<i64>,
                    > = crate::profiling::captures::dim_hints_from_var_nodes(
                        extractor.var_nodes(),
                        self.type_map,
                    );
                    // Model-type-resolved per-var dims FIRST: a field named
                    // `weight` exists in every Linear/Embedding module, so
                    // the bare-leaf-name bridge below drops it as ambiguous
                    // while this map has each var's correct dims.
                    for (vid, d) in extractor.known_param_dims() {
                        dim_seeds.entry(*vid).or_insert_with(|| d.clone());
                    }
                    let field_dims = self.models.unique_field_dims();
                    for (name, vid) in extractor.named_param_var_ids() {
                        if dim_seeds.contains_key(vid) {
                            continue;
                        }
                        let leaf = name.rsplit('.').next().unwrap_or(name);
                        if let Some(d) = field_dims.get(leaf) {
                            dim_seeds.insert(*vid, d.clone());
                        }
                    }
                    // Item 4's DataLoader proof seeds the batch fields: a
                    // Proven scan certifies every batch is exactly
                    // [batch_size, seq_len] (unanimous loaders, drop_last
                    // required, short batches padded by the runtime), and
                    // those fields are the entry point the whole forward
                    // chain hangs off. Only fields the runtime emits at
                    // [B, S], and only reads of a step-input dict (an Input
                    // leaf) — a user-built dict proves nothing.
                    let arena_debug =
                        std::env::var("NSL_ARENA_DEBUG").ok().as_deref() == Some("1");
                    if let Some(facts) = self.lm_head_loader_scan.facts() {
                        let input_leaves: std::collections::HashSet<
                            crate::wengert::VarId,
                        > = effective_primal
                            .ops
                            .iter()
                            .filter(|o| {
                                matches!(o.op, crate::wengert::PrimalOp::Input(_))
                            })
                            .map(|o| o.result)
                            .collect();
                        for op in &effective_primal.ops {
                            let crate::wengert::PrimalOp::Passthrough(n) = &op.op else {
                                continue;
                            };
                            let Some(field) = n.strip_prefix("dict_get:") else {
                                continue;
                            };
                            let eligible = matches!(
                                field,
                                "input_ids" | "labels" | "segment_ids" | "position_ids"
                            ) && op.inputs.len() == 1
                                && input_leaves.contains(&op.inputs[0]);
                            if arena_debug {
                                nsl_runtime::nsl_log!(INFO, "arena-debug", 
                                    "[arena-debug] dict_get:{field} v{} inputs={:?} \
                                     leaf={} -> seed {}",
                                    op.result,
                                    op.inputs,
                                    op.inputs
                                        .first()
                                        .is_some_and(|i| input_leaves.contains(i)),
                                    eligible,
                                );
                            }
                            if eligible {
                                dim_seeds.entry(op.result).or_insert_with(|| {
                                    vec![facts.batch_size as i64, facts.seq_len as i64]
                                });
                            }
                        }
                    } else if arena_debug {
                        nsl_runtime::nsl_log!(INFO, "arena-debug", 
                            "[arena-debug] loader scan unproven: {:?}",
                            self.lm_head_loader_scan.reason()
                        );
                    }
                    // Same unique-preimage rule as the elems mirror above:
                    // an adjoint has its primal's shape by construction, but
                    // only a dedicated accumulator certifies WHICH primal.
                    let mirror_pairs: Vec<(
                        crate::wengert::VarId,
                        crate::wengert::VarId,
                    )> = {
                        let mut preimage: std::collections::HashMap<
                            crate::wengert::VarId,
                            u32,
                        > = Default::default();
                        for a in generator.adjoint_vars_map().values() {
                            *preimage.entry(*a).or_default() += 1;
                        }
                        generator.adjoint_vars_map()
                            .iter()
                            .filter(|(_, a)| preimage.get(a) == Some(&1))
                            .map(|(p, a)| (*p, *a))
                            .collect()
                    };
                    let n_dim_seeds = dim_seeds.len();
                    let scalar_seeds = extractor.known_param_scalar_values();
                    let size_info = crate::transient_arena::propagate_size_info(
                        &effective_primal,
                        &adjoint,
                        &|v| dim_seeds.get(&v).cloned(),
                        &|v| scalar_seeds.get(&v).copied(),
                        &mirror_pairs,
                    );
                    let shape_elems: std::collections::HashMap<
                        crate::wengert::VarId,
                        u64,
                    > = size_info
                        .iter()
                        .filter_map(|(v, si)| si.numel().map(|n| (*v, n)))
                        .collect();
                    let propagated = crate::transient_arena::propagate_elems(
                        &effective_primal,
                        &adjoint,
                        &|v| {
                            shape_elems
                                .get(&v)
                                .copied()
                                .or_else(|| elem_hints.get(&v).copied())
                        },
                    );
                    // Provenance split. "Sized nothing" reads completely
                    // differently when the seeds are empty vs when the
                    // propagation stopped early — and the fixes differ too.
                    nsl_runtime::nsl_log!(INFO, "arena", 
                        "[arena] element counts: {} dim seed(s) + {} numel \
                         hint(s) -> {} shape-propagated -> {} sized, of {} \
                         tape value(s)",
                        n_dim_seeds,
                        elem_hints.len(),
                        shape_elems.len(),
                        propagated.len(),
                        effective_primal.ops.len() + adjoint.ops.len(),
                    );
                    // Milestone C: SCHEDULED. The pass scans BOTH lists
                    // (birth/death liveness over the concatenated
                    // [forward; adjoint] positions) but the scheduler retains
                    // ONE digest per (epoch, pass) — the adjoint is the one
                    // digested, because every admitted placement lives there
                    // (`admit` refuses NotBackward) and it is the list the
                    // arena-consuming lowering walks. A forward-tape splice
                    // between here and that lowering would evade this digest;
                    // none exists today (fuse_swiglu_gate_backward runs
                    // BEFORE this analyze, CCR's splices before that), and
                    // the honest fix for a second scanned list is a
                    // scheduler API that digests both — deliberately not
                    // built for a window with no known mutator.
                    let sched = self.passes.scheduler();
                    let arena = sched
                        .schedule("MemoryPlanner", Some(&adjoint), || {
                            crate::transient_arena::analyze(
                                &effective_primal,
                                &adjoint,
                                &|v| propagated.get(&v).copied(),
                                &tape_escaping,
                                4, // GPU f32 training dtype width
                            )
                        })
                        .map_err(CodegenError::new)?
                        .finish(&self.bus)
                        .map_err(CodegenError::new)?;
                    nsl_runtime::nsl_log!(INFO, "arena", "[arena]\n{}", arena.render_report("  "));

                    // ── Stage-2B: placement ──────────────────────────
                    //
                    // Admission is deliberately narrow (see `admit`), so the
                    // interesting number is usually how much was REFUSED and
                    // to which rule. Printing that is the difference between
                    // "the arena placed nothing because the model has no
                    // eligible temporaries" and "the arena placed nothing
                    // because a rule is broken", which otherwise look
                    // identical from outside.
                    if arena_place_on {
                        let (ok, refused) = crate::transient_arena::admit(
                            &arena,
                            &effective_primal,
                            &adjoint,
                            &tape_escaping,
                            &size_info,
                        );
                        let (placements, payload) =
                            crate::transient_arena::pack(&arena, &ok);
                        let mut by_reason: std::collections::BTreeMap<&str, usize> =
                            Default::default();
                        for (_, r) in &refused {
                            *by_reason.entry(match r {
                                crate::transient_arena::RefusedBecause::NotBackward =>
                                    "forward region",
                                crate::transient_arena::RefusedBecause::Unsized =>
                                    "no static size",
                                crate::transient_arena::RefusedBecause::SavedForBackward =>
                                    "saved for backward",
                                crate::transient_arena::RefusedBecause::EscapesTape =>
                                    "escapes the tape",
                                crate::transient_arena::RefusedBecause::Aliasing =>
                                    "may alias an input",
                                crate::transient_arena::RefusedBecause::NotSingleAllocation =>
                                    "not a proven single allocation",
                                crate::transient_arena::RefusedBecause::InPlaceReuse =>
                                    "in-place reuse (input dies here)",
                                crate::transient_arena::RefusedBecause::RuntimePathVaries =>
                                    "runtime path varies (broadcast/view operand)",
                            }).or_default() += 1;
                        }
                        nsl_runtime::nsl_log!(INFO, "arena", 
                            "[arena] placement: {} of {} transient(s) admitted, \
                             {:.2} MiB payload in {} slot(s)",
                            placements.len(),
                            arena.transients.len(),
                            payload as f64 / 1048576.0,
                            placements.len(),
                        );
                        for (reason, n) in &by_reason {
                            nsl_runtime::nsl_log!(WARN, "arena", "[arena]   refused {n:>5} — {reason}");
                        }
                        // Which op kinds cost the coverage. Unsized = a
                        // propagation rule is missing or a seed never
                        // reached it; NotSingleAllocation = sized but the
                        // allowlist excludes its producer. The two have
                        // completely different fixes, per-op-kind counts
                        // are what tells them apart.
                        if std::env::var("NSL_ARENA_DEBUG").ok().as_deref() == Some("1") {
                            let mut by_kind: std::collections::BTreeMap<String, usize> =
                                Default::default();
                            for (v, r) in &refused {
                                use crate::transient_arena::RefusedBecause as R;
                                if !matches!(r, R::Unsized | R::NotSingleAllocation) {
                                    continue;
                                }
                                let producer = adjoint
                                    .ops
                                    .iter()
                                    .chain(effective_primal.ops.iter())
                                    .find(|o| o.result == *v);
                                let kind = match producer.map(|o| &o.op) {
                                    Some(crate::wengert::PrimalOp::Passthrough(n)) => {
                                        format!(
                                            "Passthrough:{}",
                                            n.split(':').next().unwrap_or(n)
                                        )
                                    }
                                    Some(other) => {
                                        let d = format!("{other:?}");
                                        d.split([' ', '{', '('])
                                            .next()
                                            .unwrap_or("?")
                                            .to_string()
                                    }
                                    None => "<no producing op>".to_string(),
                                };
                                *by_kind
                                    .entry(format!("{kind} [{r:?}]"))
                                    .or_default() += 1;
                            }
                            let mut rows: Vec<_> = by_kind.into_iter().collect();
                            rows.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
                            for (kind, n) in rows {
                                nsl_runtime::nsl_log!(INFO, "arena-debug", "[arena-debug]   {n:>5} x {kind}");
                            }
                            // The forward stall is invisible above (admit
                            // refuses forward transients NotBackward before
                            // Unsized ever fires), but an unsized forward
                            // value starves everything downstream of it in
                            // the backward too.
                            let mut fwd_unsized: std::collections::BTreeMap<String, usize> =
                                Default::default();
                            for t in &arena.transients {
                                if t.region != crate::transient_arena::Region::Forward
                                    || t.elems.is_some()
                                {
                                    continue;
                                }
                                if let Some(o) =
                                    effective_primal.ops.iter().find(|o| o.result == t.var)
                                {
                                    let kind = match &o.op {
                                        crate::wengert::PrimalOp::Passthrough(n) => {
                                            format!(
                                                "Passthrough:{}",
                                                n.split(':').next().unwrap_or(n)
                                            )
                                        }
                                        other => {
                                            let d = format!("{other:?}");
                                            d.split([' ', '{', '('])
                                                .next()
                                                .unwrap_or("?")
                                                .to_string()
                                        }
                                    };
                                    *fwd_unsized.entry(kind).or_default() += 1;
                                }
                            }
                            let mut rows: Vec<_> = fwd_unsized.into_iter().collect();
                            rows.sort_by_key(|(_, n)| std::cmp::Reverse(*n));
                            for (kind, n) in rows {
                                nsl_runtime::nsl_log!(INFO, "arena-debug", "[arena-debug]   fwd unsized {n:>5} x {kind}");
                            }
                            // The FIRST stalls in tape order — everything
                            // after the first is usually just downstream
                            // starvation.
                            let mut shown = 0;
                            for (i, o) in effective_primal.ops.iter().enumerate() {
                                if shown >= 15 {
                                    break;
                                }
                                if shape_elems.contains_key(&o.result)
                                    || matches!(
                                        o.op,
                                        crate::wengert::PrimalOp::Input(_)
                                            | crate::wengert::PrimalOp::Param(_)
                                            | crate::wengert::PrimalOp::Constant(_)
                                            | crate::wengert::PrimalOp::FreeTensor
                                    )
                                {
                                    continue;
                                }
                                // Const-lattice ops (non-tensor results)
                                // are not stalls; their state is invisible
                                // to shape_elems by design.
                                if let crate::wengert::PrimalOp::Passthrough(n) = &o.op
                                    && matches!(
                                        n.as_str(),
                                        "shape" | "subscript" | "int" | "float" | "list"
                                            | "ndim" | "item"
                                    )
                                {
                                    continue;
                                }
                                let ins: Vec<String> = o
                                    .inputs
                                    .iter()
                                    .map(|v| match size_info.get(v) {
                                        Some(si) => format!("v{v}:{si:?}"),
                                        None => format!("v{v}:?"),
                                    })
                                    .collect();
                                let kind = match &o.op {
                                    crate::wengert::PrimalOp::Passthrough(n) => {
                                        format!("Passthrough:{n}")
                                    }
                                    other => format!("{other:?}"),
                                };
                                nsl_runtime::nsl_log!(INFO, "arena-debug", 
                                    "[arena-debug]   stall #{i} v{} {} <- [{}]",
                                    o.result,
                                    kind.chars().take(60).collect::<String>(),
                                    ins.join(", ")
                                );
                                shown += 1;
                            }
                            // Where dims get LOST (result Numel): the
                            // dims-loss point poisons everything downstream
                            // into numel-land even when counts survive.
                            let mut shown = 0;
                            for (i, o) in effective_primal.ops.iter().enumerate() {
                                if shown >= 12 {
                                    break;
                                }
                                if !matches!(
                                    size_info.get(&o.result),
                                    Some(crate::transient_arena::SizeInfo::Numel(_))
                                ) {
                                    continue;
                                }
                                let ins: Vec<String> = o
                                    .inputs
                                    .iter()
                                    .map(|v| match size_info.get(v) {
                                        Some(si) => format!("v{v}:{si:?}"),
                                        None => format!("v{v}:?"),
                                    })
                                    .collect();
                                let kind = match &o.op {
                                    crate::wengert::PrimalOp::Passthrough(n) => {
                                        format!("Passthrough:{n}")
                                    }
                                    other => format!("{other:?}"),
                                };
                                nsl_runtime::nsl_log!(INFO, "arena-debug", 
                                    "[arena-debug]   numel #{i} v{} {} <- [{}]",
                                    o.result,
                                    kind.chars().take(60).collect::<String>(),
                                    ins.join(", ")
                                );
                                shown += 1;
                            }
                        }
                        if std::env::var("NSL_ARENA_DEBUG").ok().as_deref() == Some("1") {
                            for p in &placements {
                                let kind = adjoint
                                    .ops
                                    .iter()
                                    .find(|o| o.result == p.var)
                                    .map(|o| match &o.op {
                                        crate::wengert::PrimalOp::Passthrough(n) => {
                                            format!("Passthrough:{n}")
                                        }
                                        other => format!("{other:?}")
                                            .split([' ', '{', '('])
                                            .next()
                                            .unwrap_or("?")
                                            .to_string(),
                                    })
                                    .unwrap_or_else(|| "<none>".into());
                                let inputs_desc = adjoint
                                    .ops
                                    .iter()
                                    .find(|o| o.result == p.var)
                                    .map(|o| {
                                        o.inputs
                                            .iter()
                                            .map(|v| {
                                                let pk = adjoint
                                                    .ops
                                                    .iter()
                                                    .chain(effective_primal.ops.iter())
                                                    .find(|q| q.result == *v)
                                                    .map(|q| match &q.op {
                                                        crate::wengert::PrimalOp::Passthrough(n) => n.clone(),
                                                        other => format!("{other:?}")
                                                            .split([' ', '{', '('])
                                                            .next()
                                                            .unwrap_or("?")
                                                            .to_string(),
                                                    })
                                                    .unwrap_or_else(|| "<leaf?>".into());
                                                format!("v{v}<{pk}>{:?}", size_info.get(v))
                                            })
                                            .collect::<Vec<_>>()
                                            .join(", ")
                                    })
                                    .unwrap_or_default();
                                nsl_runtime::nsl_log!(INFO, "arena-debug", 
                                    "[arena-debug] slot {} v{} {} B {} <- {}",
                                    p.slot_index, p.var, p.bytes, kind, inputs_desc
                                );
                            }
                        }
                        self.arena_placements =
                            placements.iter().map(|p| (p.var, *p)).collect();
                        if !placements.is_empty() {
                            let total = builder.ins().iconst(cl_types::I64, payload as i64);
                            let nslots =
                                builder.ins().iconst(cl_types::I64, placements.len() as i64);
                            self.compile_call_by_name(
                                builder, "nsl_arena_init", &[total, nslots])?;
                            // Slot geometry, in dense order, so the runtime
                            // can verify the INTERIOR red zones — without it
                            // only the arena's outermost guards are
                            // checkable and a slot-k overrun into slot k+1
                            // goes unseen.
                            for p in &placements {
                                let off =
                                    builder.ins().iconst(cl_types::I64, p.offset as i64);
                                let bytes =
                                    builder.ins().iconst(cl_types::I64, p.bytes as i64);
                                self.compile_call_by_name(
                                    builder,
                                    "nsl_arena_declare_slot",
                                    &[off, bytes],
                                )?;
                            }
                        }
                    }
                }

                // ── D2b part 2: CSLA schedule precompute (pre-forward) ──
                // The layerwise plan, per-param facts, replay ranges, and
                // update grouping — computed HERE (on the final adjoint) so
                // the segment-streamed forward below and the window backward
                // consume the same schedule. `CslaPre` flows into the save
                // phase; `WsForwardPlan` drives the sliced forward emission.
                struct WsForwardPlan {
                    /// Half-open primal-op slices (prologue, per-segment,
                    /// epilogue) — a partition of the tape.
                    slices: Vec<(usize, usize)>,
                    /// Streamed param_list indices registered (= evicted)
                    /// at step-body top, every iteration (idempotent).
                    register_idxs: Vec<i64>,
                    /// Per-slice param_list indices uploaded before /
                    /// evicted after that slice's ops (per-param mode).
                    upload_per_slice: Vec<Vec<i64>>,
                    evict_per_slice: Vec<Vec<i64>>,
                    /// Item 10 (arena mode): one contiguous pack per streamed
                    /// LAYER group as `(first_slice, last_slice, idxs)`. The
                    /// whole group uploads at `first_slice` and evicts at
                    /// `last_slice` — a matched set, so it holds exactly one
                    /// arena slot for its residency (coarser than per-param
                    /// touch, but batched into one HtoD / DtoH each way).
                    arena_packs: Vec<(usize, usize, Vec<i64>)>,
                }
                let (csla_pre, ws_fwd_plan): (Option<CslaPre>, Option<WsForwardPlan>) =
                    if csla_active {
                        let csla_trainable: Vec<(String, crate::wengert::VarId)> = extractor
                            .named_param_var_ids()
                            .iter()
                            .filter(|(name, _)| self.is_trainable_param_name(name))
                            .map(|(n, v)| (n.clone(), *v))
                            .collect();
                        // Item 11 calibration (review M3 follow-through):
                        // REAL element counts for the layerwise plan, now
                        // from the SHARED Stage-2A hint map above — which
                        // adds the initializer-derived field dims the pure
                        // semantic map lacked (model fields are typed from
                        // annotations only, so the old binding here was
                        // empty for every unannotated real model and the
                        // prefetch pack pricing stayed blind). Symbolic
                        // shapes remain None and decline their edges.
                        let vid_by_pname: std::collections::HashMap<&str, crate::wengert::VarId> =
                            csla_trainable
                                .iter()
                                .map(|(n, v)| (n.as_str(), *v))
                                .collect();
                        let plan_lw =
                            crate::layerwise::analyze(&adjoint, &csla_trainable, &|name| {
                                vid_by_pname
                                    .get(name)
                                    .and_then(|v| elem_hints.get(v))
                                    .copied()
                            });
                        let param_name_to_accum_idx: std::collections::HashMap<&str, i64> =
                            param_paths
                                .iter()
                                .enumerate()
                                .map(|(i, p)| (p.as_str(), i as i64))
                                .collect();
                        let csla_params: Vec<CslaParam> = csla_trainable
                            .iter()
                            .filter_map(|(name, primal_vid)| {
                                let &accum_idx = param_name_to_accum_idx.get(name.as_str())?;
                                Some(CslaParam {
                                    name: name.clone(),
                                    primal_vid: *primal_vid,
                                    adj_vid: generator.adjoint_of(*primal_vid),
                                    accum_idx,
                                })
                            })
                            .collect();
                        // PRIMAL-side view chains rooted at trainable params
                        // (tied-head `embed.transpose(0,1)` etc.) — buffered
                        // as slots but aliasing θ; the window site checks
                        // their reads against each param's update range, and
                        // the forward streamer keys eviction off their last
                        // read too.
                        let trainable_vid_set: std::collections::HashSet<
                            crate::wengert::VarId,
                        > = csla_trainable.iter().map(|(_, v)| *v).collect();
                        let mut primal_view_of: std::collections::HashMap<
                            crate::wengert::VarId,
                            crate::wengert::VarId,
                        > = std::collections::HashMap::new();
                        for op in &effective_primal.ops {
                            if crate::wengert::is_view_producing_op(&op.op) {
                                for &input in &op.inputs {
                                    if trainable_vid_set.contains(&input) {
                                        primal_view_of.insert(op.result, input);
                                    } else if let Some(&p) = primal_view_of.get(&input) {
                                        primal_view_of.insert(op.result, p);
                                    }
                                }
                            }
                        }
                        let imports = crate::layerwise::adjoint_primal_imports(
                            &effective_primal,
                            &adjoint,
                        );

                        // ── D1b schedule derivation (moved from the window
                        // site — indices are FINAL-adjoint positions) ──
                        let adjoint_len = adjoint.ops.len();
                        let mut ranges =
                            crate::layerwise::partition_ranges(&plan_lw, adjoint_len);
                        if ranges.is_empty() {
                            // Degenerate (empty adjoint): one empty prologue
                            // so the update groups still fire.
                            ranges.push(crate::layerwise::ReplayRange {
                                start: 0,
                                end: adjoint_len,
                                layer: None,
                            });
                        }
                        let n_ranges = ranges.len();
                        // Adjoint op position by result vid — for grad-op
                        // containment.
                        let adj_pos: std::collections::HashMap<
                            crate::wengert::VarId,
                            usize,
                        > = adjoint
                            .ops
                            .iter()
                            .enumerate()
                            .map(|(i, op)| (op.result, i))
                            .collect();
                        // Update groups. A layer's param updates right after
                        // its range's replay iff its gradient op sits
                        // positionally INSIDE that range (positional
                        // attribution slop demotes it to the epilogue group
                        // — always correct, merely later). Dead params (no
                        // adjoint) update with their layer on a zero
                        // accumulator. Every param_paths slot lands in
                        // exactly one group.
                        let mut layer_group: Vec<Vec<i64>> = vec![Vec::new(); n_ranges];
                        let mut grouped: std::collections::HashSet<i64> = Default::default();
                        // Item 11 calibration: static element count per grouped
                        // param (0 = symbolic shape), keyed by accum_idx — the
                        // prefetch gate's pack-byte source.
                        let mut elems_by_accum: std::collections::HashMap<i64, u64> =
                            Default::default();
                        {
                            let param_by_name: std::collections::HashMap<&str, &CslaParam> =
                                csla_params.iter().map(|p| (p.name.as_str(), p)).collect();
                            for (ri, range) in ranges.iter().enumerate() {
                                let Some(li) = range.layer else { continue };
                                for pinfo in &plan_lw.layers[li].params {
                                    let Some(cp) = param_by_name.get(pinfo.name.as_str())
                                    else {
                                        continue;
                                    };
                                    let in_range =
                                        match cp.adj_vid.and_then(|a| adj_pos.get(&a)) {
                                            Some(&pos) => {
                                                pos >= range.start && pos < range.end
                                            }
                                            None => true,
                                        };
                                    if in_range && grouped.insert(cp.accum_idx) {
                                        layer_group[ri].push(cp.accum_idx);
                                        elems_by_accum
                                            .insert(cp.accum_idx, pinfo.elems.unwrap_or(0));
                                    }
                                }
                                layer_group[ri].sort_unstable();
                            }
                        }
                        let global_group: Vec<i64> = (0..param_paths.len() as i64)
                            .filter(|i| !grouped.contains(i))
                            .collect();
                        // Compile-time schedule line — the gates' anti-vacuity
                        // anchor for the LAYER-MAJOR shape itself (the runtime
                        // window counter can't distinguish a degenerate
                        // all-epilogue schedule from the real k-range one).
                        nsl_runtime::nsl_log!(INFO, "csla", 
                            "[csla] layer-major schedule: {} ranges, {} layer-grouped params, \
                             {} epilogue params",
                            n_ranges,
                            grouped.len(),
                            global_group.len(),
                        );

                        // ── Weight-stream admission (moved from the window
                        // site) + the part-2 forward streaming plan ──
                        let ws_active = self.compile_options.weight_stream.enabled;
                        let mut ws_streamed_sorted: Vec<i64> = Vec::new();
                        let ws_plan = if ws_active {
                            // Review D2b-1 (HIGH): a buffered primal VIEW of a
                            // streamed param (e.g. transpose(w) saved for the
                            // matmul adjoint) caches a data pointer into θ's
                            // storage — eviction frees that storage and the
                            // later upload allocates a NEW buffer, so the view
                            // slot would read recycled memory: silent
                            // corruption. Any param rooting a view chain that
                            // lands in the buffered-import set stays RESIDENT
                            // (always safe, merely unstreamed). The import
                            // list is the slot superset (ghost imports never
                            // become slots but also never root tensor views).
                            // Pure helper — unit-tested in layerwise.rs
                            // (review D2b-2-3: the exclusion never fires on
                            // the gate fixtures, so the logic is pinned at
                            // the unit level).
                            let view_rooted = crate::layerwise::ws_view_rooted_params(
                                &imports,
                                &primal_view_of,
                            );
                            let unstreamable_idxs: std::collections::HashSet<i64> =
                                csla_params
                                    .iter()
                                    .filter(|cp| view_rooted.contains(&cp.primal_vid))
                                    .map(|cp| cp.accum_idx)
                                    .collect();
                            if !unstreamable_idxs.is_empty() {
                                nsl_runtime::nsl_log!(INFO, "weight-stream", 
                                    "[weight-stream] {} param(s) stay resident: a buffered \
                                     view of their storage rides the window slots",
                                    unstreamable_idxs.len()
                                );
                            }
                            let mut ws_all: Vec<i64> = layer_group
                                .iter()
                                .flatten()
                                .copied()
                                .filter(|i| !unstreamable_idxs.contains(i))
                                .collect();
                            ws_all.sort_unstable();
                            // Part 2: slice the forward per CCR segment and
                            // key each streamed param's upload/evict off its
                            // first/last primal touch (view-closure-extended).
                            let plan_ref = ccr_plan
                                .as_ref()
                                .expect("csla refusal above guarantees a plan");
                            // Milestone C: the SECOND positional fork — the
                            // segment bounds are sliced against
                            // effective_primal ~1,200 lines after planning.
                            // Same digest, same rule as the
                            // apply_to_adjoint fork above.
                            {
                                let sched = self.passes.scheduler();
                                sched
                                    .assert_tape_unchanged_since("CCR", &effective_primal)
                                    .map_err(CodegenError::new)?;
                            }
                            let seg_bounds: Vec<(usize, usize)> = plan_ref
                                .segments
                                .iter()
                                .map(|s| (s.start, s.end))
                                .collect();
                            let slices = crate::layerwise::forward_slices(
                                &seg_bounds,
                                effective_primal.ops.len(),
                            )
                            .map_err(|e| {
                                CodegenError::new(format!(
                                    "--weight-stream: cannot slice the forward per \
                                     CCR segment: {e}"
                                ))
                            })?;
                            let ws_idx_set: std::collections::HashSet<i64> =
                                ws_all.iter().copied().collect();
                            let streamed_vids: std::collections::HashSet<
                                crate::wengert::VarId,
                            > = csla_params
                                .iter()
                                .filter(|cp| ws_idx_set.contains(&cp.accum_idx))
                                .map(|cp| cp.primal_vid)
                                .collect();
                            let touch = crate::layerwise::forward_touch_slices(
                                &effective_primal,
                                &slices,
                                &streamed_vids,
                                &primal_view_of,
                            );
                            let vid_to_idx: std::collections::HashMap<
                                crate::wengert::VarId,
                                i64,
                            > = csla_params
                                .iter()
                                .map(|cp| (cp.primal_vid, cp.accum_idx))
                                .collect();
                            let mut upload_per_slice: Vec<Vec<i64>> =
                                vec![Vec::new(); slices.len()];
                            let mut evict_per_slice: Vec<Vec<i64>> =
                                vec![Vec::new(); slices.len()];
                            for (vid, (first, last)) in &touch {
                                let idx = vid_to_idx[vid];
                                upload_per_slice[*first].push(idx);
                                evict_per_slice[*last].push(idx);
                            }
                            for v in upload_per_slice.iter_mut() {
                                v.sort_unstable();
                            }
                            for v in evict_per_slice.iter_mut() {
                                v.sort_unstable();
                            }
                            // Anti-vacuity: gates assert this exact line so a
                            // degenerate no-slice or no-touch plan can't pass
                            // as streaming. The per-slice vectors pin bracket
                            // PLACEMENT, not just cardinality (review D2b-2-2:
                            // a plan widened by touch over-extension — e.g. all
                            // uploads in slice 0, all evicts in the last —
                            // produces the same counts and bit-exact parity
                            // while silently reverting to full forward
                            // residency).
                            let per_slice = |v: &[Vec<i64>]| -> String {
                                v.iter()
                                    .map(|s| s.len().to_string())
                                    .collect::<Vec<_>>()
                                    .join(",")
                            };
                            nsl_runtime::nsl_log!(INFO, "weight-stream", 
                                "[weight-stream] forward streaming: {} slices, \
                                 {} streamed params ({} touched by the primal); \
                                 uploads/slice [{}] evicts/slice [{}]",
                                slices.len(),
                                ws_all.len(),
                                touch.len(),
                                per_slice(&upload_per_slice),
                                per_slice(&evict_per_slice),
                            );
                            // Item 10: coarsen the per-param touch into one
                            // contiguous pack per streamed LAYER group (the
                            // group's forward bracket = [min first-touch, max
                            // last-touch] over its touched members). Matched
                            // upload/evict sets → one arena slot per pack.
                            let idx_touch: std::collections::HashMap<i64, (usize, usize)> = touch
                                .iter()
                                .map(|(vid, fl)| (vid_to_idx[vid], *fl))
                                .collect();
                            let mut arena_packs: Vec<(usize, usize, Vec<i64>)> = Vec::new();
                            for group in &layer_group {
                                let mut members: Vec<i64> = Vec::new();
                                let mut first = usize::MAX;
                                let mut last = 0usize;
                                for &idx in group {
                                    if let Some(&(f, l)) = idx_touch.get(&idx) {
                                        members.push(idx);
                                        first = first.min(f);
                                        last = last.max(l);
                                    }
                                }
                                if !members.is_empty() {
                                    members.sort_unstable();
                                    arena_packs.push((first, last, members));
                                }
                            }
                            if self.compile_options.weight_stream.arena {
                                nsl_runtime::nsl_log!(INFO, "weight-stream", 
                                    "[weight-stream] arena mode: {} contiguous layer packs \
                                     (sizes [{}])",
                                    arena_packs.len(),
                                    arena_packs
                                        .iter()
                                        .map(|(_, _, m)| m.len().to_string())
                                        .collect::<Vec<_>>()
                                        .join(","),
                                );
                            }
                            ws_streamed_sorted = ws_all.clone();
                            Some(WsForwardPlan {
                                slices,
                                register_idxs: ws_all,
                                upload_per_slice,
                                evict_per_slice,
                                arena_packs,
                            })
                        } else {
                            None
                        };
                        // Item 11 calibration: Σ static elems of each range's
                        // STREAMED params (the pack the gate prices). A member
                        // with SYMBOLIC shape (elems recorded as 0) poisons the
                        // whole range to 0 = "unpriceable" — a partial sum
                        // would UNDERSTATE the transfer and wrongly activate
                        // an overlap edge (review M3); the gate declines
                        // unpriceable packs instead.
                        let ws_set: std::collections::HashSet<i64> =
                            ws_streamed_sorted.iter().copied().collect();
                        let range_pack_elems: Vec<u64> = layer_group
                            .iter()
                            .map(|g| {
                                let members: Vec<u64> = g
                                    .iter()
                                    .filter(|i| ws_set.contains(i))
                                    .map(|i| elems_by_accum.get(i).copied().unwrap_or(0))
                                    .collect();
                                if members.contains(&0) {
                                    0
                                } else {
                                    members.iter().sum()
                                }
                            })
                            .collect();
                        // Item 3: derive the ParameterPlan ONCE, here, where
                        // the streaming schedule is final. Everything
                        // downstream (both registration belts, the runtime
                        // cross-check) reads it rather than re-deriving
                        // "streamed && bf16sr" / "zero3 ? streamed : {}" from
                        // the flags — the duplication that let the three
                        // residency tables be populated from three separate
                        // spellings of the same intent.
                        // Item 11: static element counts, aligned with
                        // param_paths (0 = symbolic — derive treats it as
                        // elementwise-ineligible).
                        let plan_elems: Vec<u64> = (0..param_paths.len())
                            .map(|u| {
                                elems_by_accum.get(&(u as i64)).copied().unwrap_or(0)
                            })
                            .collect();
                        let plan = crate::parameter_plan::ParameterPlan::derive(
                            &param_paths,
                            &ws_streamed_sorted,
                            &plan_elems,
                            &crate::parameter_plan::PlanFeatures {
                                weight_stream: self.compile_options.weight_stream.enabled,
                                param_dtype_bf16sr: self.features.param_dtype_bf16sr,
                                zero_stage: self.features.zero_stage,
                                zero_elementwise: self.features.zero_elementwise,
                                world_size: self.features.world_size as u32,
                            },
                        )
                        .map_err(|e| {
                            CodegenError::new(format!("parameter plan: {e}"))
                        })?;
                        if std::env::var("NSL_PARAM_PLAN_REPORT").ok().as_deref()
                            == Some("1")
                        {
                            eprint!("{}", plan.report());
                        }
                        (
                            Some(CslaPre {
                                params: csla_params,
                                primal_view_of,
                                imports,
                                schedule: CslaSchedule {
                                    ranges,
                                    layer_group,
                                    global_group,
                                    ws_streamed: ws_streamed_sorted,
                                    range_pack_elems,
                                    plan,
                                },
                            }),
                            ws_plan,
                        )
                    } else {
                        (None, None)
                    };

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
                // Milestone C: the arena's positional liveness (birth/death
                // over the concatenated [forward; adjoint] timeline) was
                // captured at `transient_arena::analyze`; the placements it
                // burned into `arena_placements` are consumed from here on,
                // during lowering (`wengert_lower.rs` looks slots up per op).
                // Slot-sharing is sound only if the list being lowered is the
                // list that was analyzed — prove the adjoint has not moved
                // since the scan, at the entry to the consumption window
                // (assert at ENTRY, not after a write). Guarded: no
                // placements means nothing downstream consumes the scan.
                if !self.arena_placements.is_empty() {
                    let sched = self.passes.scheduler();
                    sched
                        .assert_tape_unchanged_since("MemoryPlanner", &adjoint)
                        .map_err(CodegenError::new)?;
                }
                // Item 11: plan the per-segment forward early-free. Built
                // here — after the CCR plan's owned-tensor restriction and
                // after the weight-stream plan exists — so the eligibility
                // conditions are all decidable:
                //   * ws path: already sliced, keeps its own (post-forward)
                //     free discipline; composing the two is future work and
                //     silently changing ws lifetimes here is not it.
                //   * CSLA: buffers the adjoint's primal imports across the
                //     accumulation window; its retention contract is the
                //     bulk-free's, not this pass's. Excluded.
                //   * NSL_CCR_SEGMENT_FREE=0: kill switch for A/B under ONE
                //     binary (the runtime-archive lesson: never A/B by
                //     swapping binaries).
                struct CcrSegmentFree {
                    /// (segment index if this slice IS a segment, start, end)
                    slices: Vec<(Option<usize>, usize, usize)>,
                    lists: Vec<crate::wengert::WengertList>,
                }
                let ccr_segment_free: Option<CcrSegmentFree> = match &ccr_plan {
                    Some(plan)
                        if ws_fwd_plan.is_none()
                            && !self.compile_options.layerwise_accum
                            && std::env::var("NSL_CCR_SEGMENT_FREE").as_deref()
                                != Ok("0") =>
                    {
                        let seg_bounds: Vec<(usize, usize)> = plan
                            .segments
                            .iter()
                            .map(|seg| (seg.start, seg.end))
                            .collect();
                        let slices = crate::layerwise::forward_slices(
                            &seg_bounds,
                            effective_primal.ops.len(),
                        )
                        .map_err(CodegenError::new)?;
                        let mut tagged = Vec::with_capacity(slices.len());
                        for &(s_op, e_op) in &slices {
                            let si = seg_bounds
                                .iter()
                                .position(|&(bs, be)| bs == s_op && be == e_op);
                            tagged.push((si, s_op, e_op));
                        }
                        Some(CcrSegmentFree {
                            slices: tagged,
                            lists: crate::ccr::build_segment_free_lists(plan),
                        })
                    }
                    _ => None,
                };
                self.emit_inplace_suppress(builder, true)?;
                let full_lowered = if let Some(wsplan) = &ws_fwd_plan {
                    // Segment-streamed forward: lower the primal per CCR
                    // slice with upload/evict FFI calls between slices. All
                    // slices share one straight-line block chain, so SSA
                    // values flow across boundaries; the fold state
                    // (var_map/var_types/owned/freed) threads through
                    // `compile_wengert_ops_range` so the result is
                    // byte-identical to the monolithic lowering — streaming
                    // only interleaves resident-set changes for θ.
                    let mut var_map = primal_vars.clone();
                    let mut var_types = effective_primal.var_types.clone();
                    let mut owned_values = Vec::new();
                    let mut hook_freed_input_vars = std::collections::HashSet::new();
                    let mut explicit_freed_vars = std::collections::HashSet::new();
                    // No FASE hook on this forward-streaming path — stays empty.
                    let mut hook_freed_param_vars = std::collections::HashSet::new();
                    let arena_mode = self.compile_options.weight_stream.arena;
                    for (si, &(s, e)) in wsplan.slices.iter().enumerate() {
                        // Item 10: in arena mode a whole layer pack uploads at
                        // its bracket-start slice (ONE contiguous transfer);
                        // otherwise the per-param first-touch set.
                        let arena_uploads: Vec<Vec<i64>> = if arena_mode {
                            wsplan
                                .arena_packs
                                .iter()
                                .filter(|(f, _, _)| *f == si)
                                .map(|(_, _, idxs)| idxs.clone())
                                .collect()
                        } else {
                            Vec::new()
                        };
                        let has_upload = if arena_mode {
                            !arena_uploads.is_empty()
                        } else {
                            !wsplan.upload_per_slice[si].is_empty()
                        };
                        if has_upload {
                            // Upload under the Weights surface (allocation
                            // accounting).
                            let prev_surf = self.compile_call_by_name(
                                builder,
                                "nsl_gpu_get_alloc_surface",
                                &[],
                            )?;
                            let wsurf = builder.ins().iconst(cl_types::I8, SURFACE_WEIGHTS);
                            self.compile_call_by_name(
                                builder,
                                "nsl_gpu_set_alloc_surface",
                                &[wsurf],
                            )?;
                            if arena_mode {
                                for idxs in &arena_uploads {
                                    self.emit_ws_pack_upload(builder, param_list, idxs)?;
                                }
                            } else {
                                for &idx in &wsplan.upload_per_slice[si] {
                                    let iv = builder.ins().iconst(cl_types::I64, idx);
                                    let pw = self.compile_call_by_name(
                                        builder,
                                        "nsl_list_get",
                                        &[param_list, iv],
                                    )?;
                                    self.compile_call_by_name(
                                        builder,
                                        "nsl_weight_stream_upload",
                                        &[pw],
                                    )?;
                                }
                            }
                            self.compile_call_by_name(
                                builder,
                                "nsl_gpu_set_alloc_surface",
                                &[prev_surf],
                            )?;
                        }
                        crate::wengert_lower::compile_wengert_ops_range(
                            self,
                            builder,
                            state,
                            &effective_primal,
                            s..e,
                            &mut var_map,
                            &mut var_types,
                            &mut owned_values,
                            &mut hook_freed_input_vars,
                            &mut explicit_freed_vars,
                            &mut hook_freed_param_vars,
                            None,
                        )?;
                        // Evict this slice's last-touch params/packs — read-only
                        // (writeback=0): forwards never mutate θ, the mirror
                        // is current by construction.
                        if arena_mode {
                            let evicts: Vec<Vec<i64>> = wsplan
                                .arena_packs
                                .iter()
                                .filter(|(_, l, _)| *l == si)
                                .map(|(_, _, idxs)| idxs.clone())
                                .collect();
                            for idxs in &evicts {
                                self.emit_ws_pack_evict(builder, param_list, idxs, 0)?;
                            }
                        } else {
                            for &idx in &wsplan.evict_per_slice[si] {
                                let iv = builder.ins().iconst(cl_types::I64, idx);
                                let pw = self.compile_call_by_name(
                                    builder,
                                    "nsl_list_get",
                                    &[param_list, iv],
                                )?;
                                let wb = builder.ins().iconst(cl_types::I64, 0);
                                self.compile_call_by_name(
                                    builder,
                                    "nsl_weight_stream_evict",
                                    &[pw, wb],
                                )?;
                            }
                        }
                    }
                    // Same sdpa-extras adoption the monolithic wrapper does.
                    for v in self.sdpa_extra_owned.drain(..) {
                        owned_values.push((u32::MAX, v, crate::wengert::WengertType::Tensor));
                    }
                    crate::wengert_lower::LoweredWengert {
                        var_map,
                        owned_values,
                        hook_freed_input_vars,
                        explicit_freed_vars,
                        hook_freed_param_vars,
                    }
                } else if let Some(seg_free) = &ccr_segment_free {
                    // Item 11: per-segment forward early-free. Same sliced
                    // emission the weight-streaming branch above uses — one
                    // straight-line block chain, fold state threaded through
                    // `compile_wengert_ops_range`, byte-identical op stream —
                    // but between slices the only thing emitted is each
                    // segment's FreeTensor mini-list. With the single
                    // post-forward free list, EVERY segment's interiors were
                    // still live at end-of-forward, which is where the global
                    // peak sits: `--checkpoint-blocks` was reducing the
                    // backward's activation wall and never the forward's.
                    let mut var_map = primal_vars.clone();
                    let mut var_types = effective_primal.var_types.clone();
                    let mut owned_values = Vec::new();
                    let mut hook_freed_input_vars = std::collections::HashSet::new();
                    let mut explicit_freed_vars = std::collections::HashSet::new();
                    let mut hook_freed_param_vars = std::collections::HashSet::new();
                    let mut freed_count = 0usize;
                    let mut segments_freed = 0usize;
                    for &(si, s_op, e_op) in &seg_free.slices {
                        crate::wengert_lower::compile_wengert_ops_range(
                            self,
                            builder,
                            state,
                            &effective_primal,
                            s_op..e_op,
                            &mut var_map,
                            &mut var_types,
                            &mut owned_values,
                            &mut hook_freed_input_vars,
                            &mut explicit_freed_vars,
                            &mut hook_freed_param_vars,
                            None,
                        )?;
                        if let Some(seg_idx) = si {
                            let free_list = &seg_free.lists[seg_idx];
                            if free_list.ops.is_empty() {
                                continue;
                            }
                            // Through the RANGE fold, not the wrapper: the
                            // wrapper drains `sdpa_extra_owned` into a return
                            // value, and a mid-forward call here would steal
                            // every fused-SDPA LSE pushed by the slices before
                            // it — un-owning them so the end-of-step bulk free
                            // never releases them. Review finding F1
                            // (2026-08-24), confirmed as a +8 MB/micro-step
                            // ramp at 1B (16 layers x [2,32,2048] f32) before
                            // this fix; the drain happens ONCE, after the last
                            // slice, exactly as the ws branch does.
                            let before = explicit_freed_vars.len();
                            let mut free_var_types = free_list.var_types.clone();
                            crate::wengert_lower::compile_wengert_ops_range(
                                self,
                                builder,
                                state,
                                free_list,
                                0..free_list.ops.len(),
                                &mut var_map,
                                &mut free_var_types,
                                &mut owned_values,
                                &mut hook_freed_input_vars,
                                &mut explicit_freed_vars,
                                &mut hook_freed_param_vars,
                                None,
                            )?;
                            let freed_here = explicit_freed_vars.len() - before;
                            freed_count += freed_here;
                            if freed_here > 0 {
                                segments_freed += 1;
                            }
                        }
                    }
                    nsl_runtime::nsl_log!(INFO, "ccr", 
                        "[ccr] per-segment early-free: {} interior value(s) freed \
                         across {} segment(s) during the forward",
                        freed_count, segments_freed,
                    );
                    for v in self.sdpa_extra_owned.drain(..) {
                        owned_values.push((u32::MAX, v, crate::wengert::WengertType::Tensor));
                    }
                    crate::wengert_lower::LoweredWengert {
                        var_map,
                        owned_values,
                        hook_freed_input_vars,
                        explicit_freed_vars,
                        hook_freed_param_vars,
                    }
                } else {
                    crate::wengert_lower::compile_wengert_ops(
                        self,
                        builder,
                        state,
                        &effective_primal,
                        &primal_vars,
                        None, // FASE on_param_grad hook — wired in Task 3
                    )?
                };
                self.emit_inplace_suppress(builder, false)?;
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

        // 7e1b. Debug training: emit gradient checksum to catch silent corruption.
        // Prints sum(abs(grad)) per parameter — detects NaN, zero, and misrouted gradients.
        // Skip when hook active — grads_list is a null sentinel.
        if self.compile_options.debug_training && !fase_hook_active {
            self.compile_call_by_name(
                builder,
                "nsl_debug_grad_checksum",
                &[grads_list, num_params_val],
            )?;
        }

        // 7e1b'. P0.3 gradient-integrity gate (FullBuffer / composite path):
        // scan the materialized grads list once per step. Skipped when the
        // FASE hook is active (grads_list is a null sentinel) — that path is
        // instrumented per-parameter inside the hook (step_begin/note/step_end).
        if self.compile_options.grad_integrity && !fase_hook_active {
            self.compile_call_by_name(
                builder,
                "nsl_grad_integrity_check",
                &[grads_list, num_params_val],
            )?;
        }

        // 7e1c. Dev Tools Phase 4 Task 4: health-monitor hooks.
        // Emits per-step loss, per-parameter gradient norm, per-parameter
        // weight norm (step 0 + every 100 steps), and a snapshot flush every
        // 100 steps.  Gated on `health_monitor`; when neither it nor
        // `inspect_enabled` is on the IR is byte-identical to pre-phase-4.
        //
        // TODO(phase4-fase): splice grad-norm emission into the FASE per-layer
        // loop when FASE is active.  Phase 4 Task 4 ships the standard-
        // backward-only path — grads_list is indexed the same whether the
        // primary backward was tape-AD or source-AD.

        // (a) Record loss: scalarize loss_val and call
        //     nsl_health_record_loss(loss_scalar, step).
        // Also emitted when only `--inspect` is on: `@inspect` predicates read
        // `loss` back through nsl_health_get_last_loss, so without this call
        // the collector would stay empty and `loss` would read 0.0 (the exact
        // silent-wrong the getter replaced).
        if self.compile_options.dev_tools.health_monitor || self.compile_options.dev_tools.inspect_enabled {
            let loss_scalar = self.compile_call_by_name(
                builder,
                "nsl_tensor_item",
                &[loss_val],
            )?;
            let step_now = builder.use_var(step_count_var);
            self.compile_call_by_name(
                builder,
                "nsl_health_record_loss",
                &[loss_scalar, step_now],
            )?;
        }

        if self.compile_options.dev_tools.health_monitor {
            use cranelift_codegen::ir::{types as cl_types, MemFlagsData};
            let _ = MemFlagsData::trusted(); // keep import valid across cfgs

            // Precompute step-gating flags shared by grad/weight/flush hooks.
            let zero_i64_h = builder.ins().iconst(cl_types::I64, 0);
            let step_h = builder.use_var(step_count_var);
            let hundred = builder.ins().iconst(cl_types::I64, 100);
            let step_mod = builder.ins().srem(step_h, hundred);
            let is_flush_due = builder.ins().icmp(
                cranelift_codegen::ir::condcodes::IntCC::Equal,
                step_mod,
                zero_i64_h,
            );
            // Init == step 0.  We reuse `is_flush_due` for the weight-norm
            // periodic check (step 0 also satisfies step % 100 == 0), so a
            // single gate covers both "init" and "every 100".

            // (b) Per-parameter gradient norms (unrolled over compile-time
            //     param_paths; grads_list / param_list are indexed by idx).
            //
            // P0 cert campaign FIX: under the FASE-Deferred source-AD hook
            // (AdamW/Adam + grad_accumulation >= 2) the hook accumulates
            // each gradient into m_partial and FREES it during the
            // backward — grads_list entries DANGLE here, and the l2_norm
            // read segfaulted every `--monitor` run at 500M/1B scale
            // (SIGSEGV in emitted code, silently reported as exit 1). Skip
            // per-micro-batch grad norms on that path and say so loudly;
            // loss, weight-norm and flush recording still run.
            if !fase_hook_active {
                for (i, path) in param_paths.iter().enumerate() {
                    let path_data_id = self.intern_string(path)?;
                    let gv = self.module.declare_data_in_func(path_data_id, builder.func);
                    let path_ptr = builder.ins().symbol_value(cl_types::I64, gv);
                    let path_len = builder
                        .ins()
                        .iconst(cl_types::I64, path.len() as i64);
                    let layer_idx = parse_layer_idx_for_health(path);
                    let layer_idx_val = builder
                        .ins()
                        .iconst(cl_types::I32, layer_idx as i64);

                    let idx_val = builder.ins().iconst(cl_types::I64, i as i64);
                    let grad = self.compile_call_by_name(
                        builder,
                        "nsl_list_get",
                        &[grads_list, idx_val],
                    )?;
                    let gnorm = self.compile_call_by_name(
                        builder,
                        "nsl_tensor_l2_norm",
                        &[grad],
                    )?;
                    self.compile_call_by_name(
                        builder,
                        "nsl_health_record_grad_norm",
                        &[path_ptr, path_len, layer_idx_val, gnorm],
                    )?;
                }
            } else {
                nsl_runtime::nsl_log!(INFO, "health", 
                    "[health] note: per-parameter gradient norms are not \
                     recorded under the FASE-Deferred hook (per-batch grads \
                     are consumed into m_partial during the backward) — the \
                     health snapshot's grad_norm fields will be absent. \
                     Loss and weight-norm recording are unaffected."
                );
            }

            // (c) Per-parameter weight norms — gated by step % 100 == 0
            //     (which includes step 0 as the initial weight snapshot).
            let wnorm_block = builder.create_block();
            let after_wnorm = builder.create_block();
            builder
                .ins()
                .brif(is_flush_due, wnorm_block, &[], after_wnorm, &[]);

            builder.switch_to_block(wnorm_block);
            builder.seal_block(wnorm_block);
            state.current_block = Some(wnorm_block);

            for (i, path) in param_paths.iter().enumerate() {
                let path_data_id = self.intern_string(path)?;
                let gv = self.module.declare_data_in_func(path_data_id, builder.func);
                let path_ptr = builder.ins().symbol_value(cl_types::I64, gv);
                let path_len = builder
                    .ins()
                    .iconst(cl_types::I64, path.len() as i64);
                let idx_val = builder.ins().iconst(cl_types::I64, i as i64);
                let param = self.compile_call_by_name(
                    builder,
                    "nsl_list_get",
                    &[param_list, idx_val],
                )?;
                let wnorm = self.compile_call_by_name(
                    builder,
                    "nsl_tensor_l2_norm",
                    &[param],
                )?;
                // is_init flag: 1 iff step == 0. icmp already yields I8 in
                // current Cranelift (the old b1 type is gone) — a further
                // uextend to I8 is a same-width extend and fails verification,
                // which broke every `nsl run --monitor` on a train program.
                let step_cmp = builder.use_var(step_count_var);
                let zero_cmp = builder.ins().iconst(cl_types::I64, 0);
                let is_init_i8 = builder.ins().icmp(
                    cranelift_codegen::ir::condcodes::IntCC::Equal,
                    step_cmp,
                    zero_cmp,
                );
                self.compile_call_by_name(
                    builder,
                    "nsl_health_record_weight_norm",
                    &[path_ptr, path_len, wnorm, is_init_i8],
                )?;
            }

            // (d) Snapshot flush (reuses is_flush_due — we're still in wnorm_block).
            let snap_path = self
                .compile_options
                .dev_tools.profile_source_file_name
                .as_ref()
                .map(|p| format!("{}.nsl-health.json", p))
                .unwrap_or_else(|| "nsl-health.json".to_string());
            let snap_data_id = self.intern_string(&snap_path)?;
            let snap_gv = self.module.declare_data_in_func(snap_data_id, builder.func);
            let snap_ptr = builder.ins().symbol_value(cl_types::I64, snap_gv);
            let snap_len = builder
                .ins()
                .iconst(cl_types::I64, snap_path.len() as i64);
            self.compile_call_by_name(
                builder,
                "nsl_health_flush_snapshot",
                &[snap_ptr, snap_len],
            )?;

            builder.ins().jump(after_wnorm, &[]);
            builder.switch_to_block(after_wnorm);
            builder.seal_block(after_wnorm);
            state.current_block = Some(after_wnorm);
        }

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
                let off = self.compile_options.optim_state_offload;
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
                let off = self.compile_options.optim_state_offload;
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

        // 7g2. Scheduler: update learning rate if a scheduler is configured.
        // NOTE: step_count is incremented AFTER the scheduler call so that
        // step 0 produces the step-0 learning rate (e.g. warmup starts
        // correctly). All names/kwargs/defaults were resolved by
        // resolve_optim_config — this just emits the constants in the
        // stdlib signature order after the auto-injected (base_lr, step).
        if let Some(sched) = &scheduler {
            use nsl_semantic::optim_config::ResolvedScheduler;

            let mangled = format!("nsl__optim__schedulers__{}", sched.fn_name());

            // Find the actual function name (check functions/runtime_fns
            // with fallback).
            let sched_fn = if self.registry.functions.contains_key(mangled.as_str()) {
                mangled.clone()
            } else {
                let simple = sched.fn_name().to_string();
                if self.registry.functions.contains_key(simple.as_str()) {
                    simple
                } else if self.registry.runtime_fns.contains_key(mangled.as_str()) {
                    mangled.clone()
                } else if self.registry.runtime_fns.contains_key(simple.as_str()) {
                    simple
                } else {
                    mangled.clone()
                }
            };

            let base_lr_val = builder.ins().f64const(lr_value);
            let step_count_val = builder.use_var(step_count_var);
            let step_float = builder.ins().fcvt_from_sint(cl_types::F64, step_count_val);

            let extra: Vec<f64> = match sched {
                ResolvedScheduler::ConstantLr => vec![],
                ResolvedScheduler::StepLr { step_size, gamma } => vec![*step_size, *gamma],
                ResolvedScheduler::ExponentialLr { gamma } => vec![*gamma],
                ResolvedScheduler::LinearDecay {
                    total_steps,
                    end_factor,
                } => vec![*total_steps, *end_factor],
                ResolvedScheduler::CosineAnneal { t_max, eta_min } => vec![*t_max, *eta_min],
                ResolvedScheduler::WarmupCosine {
                    warmup_steps,
                    total_steps,
                    min_lr,
                } => vec![*warmup_steps, *total_steps, *min_lr],
                ResolvedScheduler::OneCycle {
                    max_lr,
                    total_steps,
                    pct_start,
                } => vec![*max_lr, *total_steps, *pct_start],
            };

            let mut call_args = vec![base_lr_val, step_float];
            for v in extra {
                call_args.push(builder.ins().f64const(v));
            }
            let new_lr = self.compile_call_by_name(builder, &sched_fn, &call_args)?;

            builder.def_var(lr_var, new_lr);
        }

        // 7h. Increment step count (after scheduler so step 0 uses the initial LR)
        let sc = builder.use_var(step_count_var);
        let one_i64 = builder.ins().iadd_imm_s(sc, 1);
        builder.def_var(step_count_var, one_i64);

        // Milestone B: periodic full-train-state checkpoint. Post-increment
        // step_count is a multiple of grad_accumulation exactly at optimizer-
        // step boundaries — where 7g just zeroed the accum buffers and the
        // offload drain has completed — so the state on disk is always a
        // clean boundary state. Fires every `checkpoint_every` optimizer
        // steps; the runtime writes tmp files and renames, so a crash mid-
        // save leaves the previous checkpoint intact.
        if let Some(save_path) = checkpoint_save_path.clone() {
            let names_list = checkpoint_names_list
                .expect("names list is built at setup whenever checkpoint_save is set");
            let interval = builder.ins().iconst(
                cl_types::I64,
                checkpoint_every.saturating_mul(grad_accumulation_steps.max(1)),
            );
            let sc_now = builder.use_var(step_count_var);
            let rem = builder.ins().srem(sc_now, interval);
            let z = builder.ins().iconst(cl_types::I64, 0);
            let fire = builder.ins().icmp(IntCC::Equal, rem, z);
            let save_block = builder.create_block();
            let cont_block = builder.create_block();
            builder.ins().brif(fire, save_block, &[], cont_block, &[]);
            builder.switch_to_block(save_block);
            builder.seal_block(save_block);
            state.current_block = Some(save_block);
            self.intern_string(&save_path)?;
            let path_val = self.compile_string_literal(builder, &save_path)?;
            let path_len = builder.ins().iconst(cl_types::I64, save_path.len() as i64);
            // Item 8: the loader handle and the epoch make the saved state a
            // position in the data stream, not just a step number. Read here
            // — inside the fire block, at an optimizer-step boundary — so the
            // recorded slot is the one the next batch would come from.
            //
            // What is recorded is the epoch to RESUME AT, which is not always
            // the epoch in progress. With a DataLoader the epoch is a loop
            // over batches and the checkpoint lands part-way through it, so
            // the resume point is (this epoch, next slot). WITHOUT a loader
            // the epoch body runs exactly once and has just finished, so the
            // resume point is the NEXT epoch — recording the current one
            // would re-run a completed epoch on every restart.
            let epoch_ctr = builder.use_var(epoch_counter_var);
            let epoch_now = if has_dataloader.is_some() {
                epoch_ctr
            } else {
                builder.ins().iadd_imm_s(epoch_ctr, 1)
            };
            self.compile_call_by_name(
                builder,
                "nsl_train_checkpoint_save",
                &[
                    path_val,
                    path_len,
                    names_list,
                    param_list,
                    state_list_1,
                    state_list_2,
                    sc_now,
                    checkpoint_dl_handle,
                    epoch_now,
                ],
            )?;
            builder.ins().jump(cont_block, &[]);
            builder.switch_to_block(cont_block);
            builder.seal_block(cont_block);
            state.current_block = Some(cont_block);
        }

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
        if self.compile_options.transient_arena {
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

    /// Model partitioning (which layers run on which stage) is deferred to
    /// M43c; the initial implementation runs the full model in a single
    /// process with logical stage-to-stage communication.
    fn compile_train_block_pipelined(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        train: &nsl_ast::block::TrainBlock,
        // CFTP v10 (item 3): matches `compile_train_block`; installs the
        // fused-CE decorator config for THIS train block before the
        // pipelined lowering runs and restores it before returning.
        train_block_stmt_id: nsl_ast::NodeId,
    ) -> Result<(), CodegenError> {
        let saved_active_fused_ce =
            self.set_active_fused_ce_config_for_train_block(train_block_stmt_id);
        let result = self.compile_train_block_pipelined_inner(builder, state, train);
        self.restore_active_fused_ce_config(saved_active_fused_ce);
        result
    }

    /// CFTP v10 (item 3): pipelined-body analogue of
    /// [`compile_train_block_inner`] so the `active_fused_ce_config`
    /// prologue/epilogue can wrap the pipelined path uniformly.
    fn compile_train_block_pipelined_inner(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        train: &nsl_ast::block::TrainBlock,
    ) -> Result<(), CodegenError> {
        let saved_variables = state.variables.clone();
        let saved_variable_types = state.variable_types.clone();
        let saved_dataloader_symbols = state.dataloader_symbols.clone();
        let saved_borrowed_batch_symbols = state.borrowed_batch_symbols.clone();

        let config = self.features.pipeline_config.clone().unwrap();
        let num_stages = config.num_stages;

        // ── 1. Pipeline init ────────────────────────────────────────────
        let v_stages = builder.ins().iconst(cl_types::I64, num_stages as i64);
        let v_schedule = builder.ins().iconst(
            cl_types::I64,
            match config.schedule_type {
                crate::pipeline::ScheduleType::OneF1B => 0i64,
                crate::pipeline::ScheduleType::GPipe => 1i64,
            },
        );
        let v_micro = builder.ins().iconst(cl_types::I64, 8); // default micro-batches
        self.compile_call_by_name(
            builder,
            "nsl_pipeline_init",
            &[v_stages, v_schedule, v_micro],
        )?;

        // ── 2. Extract config from train(...) args ──────────────────────
        // Same Training Configuration Contract as the standard path: the
        // old scan here read ONLY `model=` and accepted everything else
        // unvalidated — under @pipeline a typo'd key (or a duplicate)
        // was doubly invisible. NOTE: epochs/grad_accumulation/grad_clip
        // are still accepted-but-inert on this lowering path (no epoch
        // loop exists; micro-batches are hardcoded below) — a known
        // contract gap tracked for the pipelined path's own change; the
        // resolver at least guarantees the keys are well-formed and the
        // namespace closed.
        let pipe_cfg = nsl_semantic::train_config::resolve_train_config(
            train,
            &|sym| self.resolve_sym(sym).to_string(),
            nsl_semantic::train_config::TrainConfigPurpose::UserTrainBlock,
        )
        .map_err(|diags| {
            let msgs: Vec<String> = diags.into_iter().map(|d| d.message).collect();
            CodegenError::new(format!("train config refused: {}", msgs.join("; ")))
        })?;
        let model_sym: Option<nsl_ast::Symbol> = pipe_cfg.model;

        // Item 8 note: checkpointing is not lowered on this path, and does
        // not need a refusal HERE — `compile_train_block` already refuses
        // checkpoint_save/load/every for every program that reaches this
        // dispatch (the Milestone B arm above the pipelined branch). A
        // second copy here would be dead code that reads like the only
        // guard. Pinned by `pipelined_train_path_refuses_checkpoint_config`
        // in crates/nsl-cli/tests/train_resume_dataloader_gate.rs.

        // Same optimizer/scheduler contract as the standard path — ONE
        // resolver. This also closes the pipelined path's own historical
        // gaps: its private kwarg copy lacked adamw_lr/ns_steps arms and
        // had no Muon spec-default backfill, so the same source trained
        // differently under @pipeline.
        let optim_cfg = nsl_semantic::optim_config::resolve_optim_config(
            &train.sections,
            train.span,
            &|sym| self.resolve_sym(sym).to_string(),
            nsl_semantic::train_config::TrainConfigPurpose::UserTrainBlock,
        )
        .map_err(|diags| {
            let msgs: Vec<String> = diags.into_iter().map(|d| d.message).collect();
            CodegenError::new(format!(
                "optimizer config refused: {}",
                msgs.join("; ")
            ))
        })?;

        // Deferral-must-refuse: the pipelined optimizer step passes only
        // the shared hyperparameters (no adamw_route/ns_steps/adamw_lr
        // slots, and its state allocation gives Muon one moment buffer
        // where muon_step needs two) — lowering Muon here would emit a
        // call that cannot match muon_step's signature.
        if optim_cfg.optimizer.kind == nsl_semantic::optim_config::OptimizerKind::Muon {
            return Err(CodegenError::new(
                "the Muon optimizer is not supported on the @pipeline train \
                 path yet: the per-stage optimizer step does not thread the \
                 route/ns_steps/adamw_lr arguments muon_step requires. Use \
                 AdamW here, or drop @pipeline",
            ));
        }

        let optimizer_name = optim_cfg.optimizer.kind.as_str().to_string();
        let lr_value: f64 = optim_cfg.optimizer.lr;
        let momentum_value: f64 = optim_cfg.optimizer.momentum;
        let dampening_value: f64 = optim_cfg.optimizer.dampening;
        let weight_decay_value: f64 = optim_cfg.optimizer.weight_decay;
        // AdamW parameter groups (`no_decay=[...]`). Non-empty is refused
        // below at the optimizer-step emitter, where the comment explains
        // why the role flags have no list to be parallel to.
        let no_decay_scope = crate::param_roles::NoDecayScope {
            static_roles: optim_cfg.optimizer.no_decay_static_roles.clone(),
            exempt_non_rank2: optim_cfg.optimizer.no_decay_exempt_non_rank2,
        };
        let nesterov_value: bool = optim_cfg.optimizer.nesterov;
        let beta1_value: f64 = optim_cfg.optimizer.beta1;
        let beta2_value: f64 = optim_cfg.optimizer.beta2;
        let eps_value: f64 = optim_cfg.optimizer.eps;
        let mut step_body: Option<(&nsl_ast::stmt::Block, nsl_ast::Symbol)> = None;

        for section in &train.sections {
            match section {
                TrainSection::Optimizer(_) => {
                    // Fully consumed by resolve_optim_config above.
                }
                TrainSection::Step { param, body } => {
                    step_body = Some((body, *param));
                }
                TrainSection::Data(stmts) => {
                    // See `compile_train_block::TrainSection::Data` for why
                    // the allowlisted config pairs are skipped.
                    for stmt in stmts {
                        if is_data_section_config_pair(stmt, self.interner) {
                            continue;
                        }
                        self.compile_stmt(builder, state, stmt)?;
                    }
                }
                // Same treatment as the standard train path: bare statements
                // run once pre-training; eval:/distribute: refuse loudly
                // instead of being silently dropped.
                TrainSection::Stmt(s) => {
                    self.compile_stmt(builder, state, s)?;
                }
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
                // Deferral-must-refuse: these previously fell into a
                // `_ => {}` wildcard and were silently dropped — a
                // scheduler: section compiled clean under @pipeline and
                // trained at constant lr; callbacks: never fired.
                TrainSection::Scheduler(_) => {
                    return Err(CodegenError::new(
                        "scheduler: sections are not supported on the \
                         @pipeline train path yet — its per-stage loop \
                         never updates the learning rate, so the schedule \
                         would be silently ignored. Remove the section or \
                         drop @pipeline",
                    ));
                }
                TrainSection::Callbacks(_) => {
                    return Err(CodegenError::new(
                        "callbacks: sections are not supported on the \
                         @pipeline train path yet — the per-stage loop \
                         never invokes them, so on_step/on_epoch logic \
                         would silently not run. Remove the section or \
                         drop @pipeline",
                    ));
                }
            }
        }

        let model_sym = model_sym.ok_or_else(|| {
            CodegenError::new("pipelined train block requires 'model=<ident>' config argument")
        })?;

        // (Missing-optimizer refusal moved into resolve_optim_config.)

        let (step_body, step_param_sym) = step_body
            .ok_or_else(|| CodegenError::new("pipelined train block requires a step section"))?;

        // ── 3. Resolve model and build param_list ───────────────────────
        let (model_var, _) = *state.variables.get(&model_sym).ok_or_else(|| {
            CodegenError::new(format!(
                "undefined model variable '{}' in pipelined train block",
                self.resolve_sym(model_sym)
            ))
        })?;
        let model_ptr = builder.use_var(model_var);

        let model_var_name = self.resolve_sym(model_sym).to_string();
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
            found_name.unwrap_or_else(|| model_var_name.clone())
        };

        let layout = self
            .types
            .struct_layouts
            .get(&model_type_name)
            .cloned()
            .ok_or_else(|| {
                CodegenError::new(format!(
                    "no struct layout found for model '{}' in pipelined train block",
                    model_type_name
                ))
            })?;

        // Build param_list by recursively collecting tensor fields (same as
        // non-pipelined path — handles nested sub-models and FixedArray fields).
        let num_slots = builder
            .ins()
            .iconst(cl_types::I64, (layout.total_size / 8) as i64);
        let param_list = self.compile_call_by_name(
            builder,
            "nsl_collect_model_params",
            &[model_ptr, num_slots],
        )?;
        let num_params_val = self.compile_call_by_name(builder, "nsl_list_len", &[param_list])?;

        // ── 4. Create optimizer state buffers (runtime NslLists) ────────
        // Optimizer-state offload is NOT wired on the pipelined path (its
        // optimizer emission does not run through the shared envelope
        // helpers) — refuse rather than silently keeping state on-device.
        if self.compile_options.optim_state_offload {
            return Err(CodegenError::new(
                "--optim-state-offload is not supported for pipelined train \
                 blocks yet; remove the flag or use the non-pipelined path.",
            ));
        }
        // P5 Muon: the mixed Muon/AdamW step (routing flags + Newton-Schulz
        // args) is wired into the non-pipelined emitter only. Refuse loudly
        // rather than emit a stale-arity call into the upgraded stdlib fn.
        if optimizer_name == "muon" {
            return Err(CodegenError::new(
                "the mixed Muon/AdamW optimizer is not wired into @pipeline \
                 train blocks yet — drop @pipeline or use adamw/sgd/lion/soap",
            ));
        }
        let num_state_buffers = match optimizer_name.as_str() {
            "adam" | "adamw" | "soap" => 2,
            _ => 1,
        };

        let state_list_1 = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        let state_list_2 = if num_state_buffers >= 2 {
            self.compile_call_by_name(builder, "nsl_list_new", &[])?
        } else {
            builder.ins().iconst(cl_types::I64, 0)
        };

        // Runtime loop: for i in 0..num_params, create zeros_like(param_list[i])
        {
            let init_i = builder.declare_var(cl_types::I64);
            let init_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(init_i, init_zero);
            let hdr = builder.create_block();
            let body = builder.create_block();
            let exit = builder.create_block();
            builder.ins().jump(hdr, &[]);
            builder.switch_to_block(hdr);
            builder.seal_block(hdr);
            let i = builder.use_var(init_i);
            let c = builder.ins().icmp(IntCC::SignedLessThan, i, num_params_val);
            builder.ins().brif(c, body, &[], exit, &[]);
            builder.switch_to_block(body);
            builder.seal_block(body);
            state.current_block = Some(body);
            let p = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, i])?;
            let b1 = self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[p])?;
            self.compile_call_by_name(builder, "nsl_list_push", &[state_list_1, b1])?;
            if num_state_buffers >= 2 {
                let b2 = self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[p])?;
                self.compile_call_by_name(builder, "nsl_list_push", &[state_list_2, b2])?;
            }
            let one = builder.ins().iconst(cl_types::I64, 1);
            let next = builder.ins().iadd(i, one);
            builder.def_var(init_i, next);
            builder.ins().jump(hdr, &[]);
            builder.switch_to_block(exit);
            builder.seal_block(exit);
            state.current_block = Some(exit);
        }

        // ── 5. Declare step parameter and step counter ──────────────────
        let step_param_var = builder.declare_var(cl_types::I64);
        let init_null = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_param_var, init_null);
        state
            .variables
            .insert(step_param_sym, (step_param_var, cl_types::I64));

        let step_count_var = builder.declare_var(cl_types::I64);
        let zero_i64 = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_count_var, zero_i64);

        // Phase 5 Task 7: publish step counter for @inspect in pipelined train.
        self.inspect_train_step_var = Some(step_count_var);

        let lr_var = builder.declare_var(cl_types::F64);
        let lr_const = builder.ins().f64const(lr_value);
        builder.def_var(lr_var, lr_const);

        // ── 6. Forward pass under tape recording ────────────────────────
        let true_val = builder.ins().iconst(cl_types::I8, 1);
        self.compile_call_by_name(builder, "nsl_set_training_mode", &[true_val])?;
        self.compile_call_by_name(builder, "nsl_tape_start", &[param_list])?;

        state.flags.in_tape_region = true;
        for stmt in &step_body.stmts {
            self.compile_stmt(builder, state, stmt)?;
        }
        // ELTLS: free tape-held tensors before clearing the tape flag.
        self.free_tape_held_tensors(builder, state);
        state.flags.in_tape_region = false;

        // Find loss variable
        let loss_val = {
            let mut found = None;
            for (sym, (var, _)) in &state.variables {
                if self.resolve_sym(*sym) == "loss" {
                    found = Some(builder.use_var(*var));
                    break;
                }
            }
            found.ok_or_else(|| {
                CodegenError::new(
                    "pipelined train step body must assign to a variable named 'loss'",
                )
            })?
        };

        // ── 7. Activation send — send loss to next stage ────────────────
        // In single-process pipeline, stage 0 sends activations to logical
        // stage 1. The runtime's shared-memory backend serializes the tensor
        // into a mailbox keyed by (dst_rank, tag).
        let zero_tag = builder.ins().iconst(cl_types::I64, 0);
        let zero_stream = builder.ins().iconst(cl_types::I64, 0);
        let next_stage = builder.ins().iconst(cl_types::I64, 1);
        self.compile_call_by_name(
            builder,
            "nsl_pipeline_send",
            &[loss_val, next_stage, zero_tag, zero_stream],
        )?;

        // ── 8. Backward pass — tape backward + stop ─────────────────────
        let grads_list =
            self.compile_call_by_name(builder, "nsl_tape_backward", &[loss_val, param_list])?;
        self.compile_call_by_name(builder, "nsl_tape_stop", &[])?;

        let false_val = builder.ins().iconst(cl_types::I8, 0);
        self.compile_call_by_name(builder, "nsl_set_training_mode", &[false_val])?;

        // ── 9. Gradient send — serialize each param gradient ────────────
        // Send gradients to the previous stage (stage 0 receives gradients
        // from stage 1 in the backward direction). Each gradient is tagged
        // with its parameter index for correct matching.
        let prev_stage = builder.ins().iconst(cl_types::I64, 0);
        {
            let gs_i = builder.declare_var(cl_types::I64);
            let gs_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(gs_i, gs_zero);
            let gs_hdr = builder.create_block();
            let gs_body = builder.create_block();
            let gs_exit = builder.create_block();
            builder.ins().jump(gs_hdr, &[]);
            builder.switch_to_block(gs_hdr);
            builder.seal_block(gs_hdr);
            let gi = builder.use_var(gs_i);
            let gc = builder
                .ins()
                .icmp(IntCC::SignedLessThan, gi, num_params_val);
            builder.ins().brif(gc, gs_body, &[], gs_exit, &[]);
            builder.switch_to_block(gs_body);
            builder.seal_block(gs_body);
            state.current_block = Some(gs_body);
            let grad_val = self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, gi])?;
            self.compile_call_by_name(
                builder,
                "nsl_pipeline_send_grad",
                &[grad_val, prev_stage, gi, zero_stream],
            )?;
            let g_one = builder.ins().iconst(cl_types::I64, 1);
            let g_next = builder.ins().iadd(gi, g_one);
            builder.def_var(gs_i, g_next);
            builder.ins().jump(gs_hdr, &[]);
            builder.switch_to_block(gs_exit);
            builder.seal_block(gs_exit);
            state.current_block = Some(gs_exit);
        }

        // ── 10. Optimizer step ──────────────────────────────────────────
        // H.2: see comment in the non-pipelined emitter — `stdlib_loader`
        // produces `nsl_optim_sgd__sgd_step` (single underscore between
        // path parts), so this site must match that convention.
        let optimizer_fn_name = match optimizer_name.as_str() {
            "sgd" => "nsl_optim_sgd__sgd_step",
            "adam" => "nsl_optim_adam__adam_step",
            "adamw" => "nsl_optim_adamw__adamw_step",
            "lion" => "nsl_optim_lion__lion_step",
            "muon" => "nsl_optim_muon__muon_step",
            "soap" => "nsl_optim_soap__soap_step",
            _ => {
                return Err(CodegenError::new(format!(
                    "unsupported optimizer '{}' in pipelined train block",
                    optimizer_name
                )));
            }
        };

        let opt_fn = if self.registry.functions.contains_key(optimizer_fn_name) {
            optimizer_fn_name.to_string()
        } else {
            let simple = format!("{}_step", optimizer_name);
            if self.registry.functions.contains_key(&simple) {
                simple
            } else if self.registry.runtime_fns.contains_key(optimizer_fn_name) {
                optimizer_fn_name.to_string()
            } else if self.registry.runtime_fns.contains_key(&simple) {
                simple
            } else {
                optimizer_fn_name.to_string()
            }
        };

        // AdamW parameter groups are not wired through the @pipeline path:
        // its optimizer step is emitted per pipeline stage rather than over
        // the model's flat param list, so the role table's positional flags
        // have no list to be parallel to here. Refuse rather than decay the
        // parameters the user asked to exempt.
        if !no_decay_scope.is_empty() {
            return Err(CodegenError::new(
                "no_decay=[...] is not supported on the @pipeline train path \
                 yet: its optimizer step is emitted per stage rather than over \
                 the flat parameter list the role flags index. Drop one",
            ));
        }

        let lr = builder.use_var(lr_var);
        let momentum_const = builder.ins().f64const(momentum_value);
        let dampening_const = builder.ins().f64const(dampening_value);
        let weight_decay_const = builder.ins().f64const(weight_decay_value);
        let nesterov_const = builder
            .ins()
            .iconst(cl_types::I8, if nesterov_value { 1 } else { 0 });
        let beta1_const = builder.ins().f64const(beta1_value);
        let beta2_const = builder.ins().f64const(beta2_value);
        let eps_const = builder.ins().f64const(eps_value);

        {
            let opt_i = builder.declare_var(cl_types::I64);
            let opt_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(opt_i, opt_zero);
            let opt_hdr = builder.create_block();
            let opt_body = builder.create_block();
            let opt_exit = builder.create_block();
            builder.ins().jump(opt_hdr, &[]);
            builder.switch_to_block(opt_hdr);
            builder.seal_block(opt_hdr);
            let idx = builder.use_var(opt_i);
            let oc = builder
                .ins()
                .icmp(IntCC::SignedLessThan, idx, num_params_val);
            builder.ins().brif(oc, opt_body, &[], opt_exit, &[]);
            builder.switch_to_block(opt_body);
            builder.seal_block(opt_body);
            state.current_block = Some(opt_body);

            let param_val =
                self.compile_call_by_name(builder, "nsl_list_get", &[param_list, idx])?;
            let grad_val =
                self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, idx])?;
            let s1 = self.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, idx])?;

            match optimizer_name.as_str() {
                "sgd" => {
                    self.compile_call_by_name(
                        builder,
                        &opt_fn,
                        &[
                            param_val,
                            grad_val,
                            s1,
                            lr,
                            momentum_const,
                            dampening_const,
                            weight_decay_const,
                            nesterov_const,
                        ],
                    )?;
                }
                "adam" | "adamw" => {
                    let s2 =
                        self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, idx])?;
                    let t_val = builder.use_var(step_count_var);
                    let one = builder.ins().iconst(cl_types::I64, 1);
                    let t_plus_one = builder.ins().iadd(t_val, one);
                    let t_float = builder.ins().fcvt_from_sint(cl_types::F64, t_plus_one);
                    self.compile_call_by_name(
                        builder,
                        &opt_fn,
                        &[
                            param_val,
                            grad_val,
                            s1,
                            s2,
                            lr,
                            beta1_const,
                            beta2_const,
                            eps_const,
                            weight_decay_const,
                            t_float,
                        ],
                    )?;
                }
                "lion" => {
                    self.compile_call_by_name(
                        builder,
                        &opt_fn,
                        &[
                            param_val,
                            grad_val,
                            s1,
                            lr,
                            beta1_const,
                            beta2_const,
                            weight_decay_const,
                        ],
                    )?;
                }
                "muon" => {
                    // Muon refuses at the top of this emitter (mixed
                    // Muon/AdamW is not wired for @pipeline). Keep this arm
                    // an ERROR, not a call: the old 7-arg call shape no
                    // longer matches the 14-param mixed stdlib fn, and a
                    // silently re-enabled arm would pass garbage into
                    // adamw_route/betas/t.
                    return Err(CodegenError::new(
                        "internal: muon reached the pipelined optimizer arm \
                         despite the @pipeline refusal — mixed Muon/AdamW is \
                         not wired for pipelined train blocks",
                    ));
                }
                "soap" => {
                    let s2 =
                        self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, idx])?;
                    let t_val_p = builder.use_var(step_count_var);
                    let one_p = builder.ins().iconst(cl_types::I64, 1);
                    let t_plus_p = builder.ins().iadd(t_val_p, one_p);
                    let t_float_p = builder.ins().fcvt_from_sint(cl_types::F64, t_plus_p);
                    self.compile_call_by_name(
                        builder,
                        &opt_fn,
                        &[
                            param_val,
                            grad_val,
                            s1,
                            s2,
                            lr,
                            beta1_const,
                            beta2_const,
                            eps_const,
                            t_float_p,
                        ],
                    )?;
                }
                _ => {
                    return Err(CodegenError::new(format!(
                        "unsupported optimizer '{}' in pipelined train block",
                        optimizer_name
                    )));
                }
            }

            let o_one = builder.ins().iconst(cl_types::I64, 1);
            let o_next = builder.ins().iadd(idx, o_one);
            builder.def_var(opt_i, o_next);
            builder.ins().jump(opt_hdr, &[]);
            builder.switch_to_block(opt_exit);
            builder.seal_block(opt_exit);
            state.current_block = Some(opt_exit);
        }

        // ── 11. Increment step count ────────────────────────────────────
        let sc = builder.use_var(step_count_var);
        let one_i64 = builder.ins().iconst(cl_types::I64, 1);
        let sc_next = builder.ins().iadd(sc, one_i64);
        builder.def_var(step_count_var, sc_next);

        // ── 12. Barrier — synchronize all pipeline stages ───────────────
        self.compile_call_by_name(builder, "nsl_pipeline_barrier", &[])?;

        // ── 13. Cleanup — free gradients, param_list, optimizer buffers ─
        // Runtime loop for gradient + state buffer cleanup
        {
            let cl_i = builder.declare_var(cl_types::I64);
            let cl_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(cl_i, cl_zero);
            let cl_hdr = builder.create_block();
            let cl_body = builder.create_block();
            let cl_exit = builder.create_block();
            builder.ins().jump(cl_hdr, &[]);
            builder.switch_to_block(cl_hdr);
            builder.seal_block(cl_hdr);
            let ci = builder.use_var(cl_i);
            let cc = builder
                .ins()
                .icmp(IntCC::SignedLessThan, ci, num_params_val);
            builder.ins().brif(cc, cl_body, &[], cl_exit, &[]);
            builder.switch_to_block(cl_body);
            builder.seal_block(cl_body);
            state.current_block = Some(cl_body);
            // Free gradient
            let gv = self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, ci])?;
            self.compile_call_by_name(builder, "nsl_tensor_free", &[gv])?;
            // Free state buffers
            let sb1 = self.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, ci])?;
            self.compile_call_by_name(builder, "nsl_tensor_free", &[sb1])?;
            if num_state_buffers >= 2 {
                let sb2 =
                    self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, ci])?;
                self.compile_call_by_name(builder, "nsl_tensor_free", &[sb2])?;
            }
            let cl_one = builder.ins().iconst(cl_types::I64, 1);
            let cl_next = builder.ins().iadd(ci, cl_one);
            builder.def_var(cl_i, cl_next);
            builder.ins().jump(cl_hdr, &[]);
            builder.switch_to_block(cl_exit);
            builder.seal_block(cl_exit);
            state.current_block = Some(cl_exit);
        }
        self.compile_call_by_name(builder, "nsl_list_free", &[grads_list])?;
        self.compile_call_by_name(builder, "nsl_list_free", &[param_list])?;
        self.compile_call_by_name(builder, "nsl_list_free", &[state_list_1])?;
        if num_state_buffers >= 2 {
            self.compile_call_by_name(builder, "nsl_list_free", &[state_list_2])?;
        }

        // ── 14. Pipeline destroy ────────────────────────────────────────
        self.compile_call_by_name(builder, "nsl_pipeline_destroy", &[])?;

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
            .with_checkpoint_policies(if self.compile_options.training_reference {
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
