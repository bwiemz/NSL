//! The pass bridges the train-block driver calls into: `invoke_wrga_if_enabled`,
//! `invoke_cpdt_if_enabled` and `invoke_csha_if_enabled`, plus the CPDT
//! stale-plan test knob they share. Each builds a pass input from the state
//! stashed on the `Compiler` (decorator configs, the WGGO applied plan, the
//! cluster topology), runs the pass, records its disposition on the pass
//! trace and publishes the product on the bus. Moved whole out of `stmt.rs`
//! (roadmap A1); the driver sites are `stmt_train/driver.rs`,
//! `stmt_train/plan_wrga_cpdt.rs` and `stmt_train/plan_csha_prune.rs`.

use crate::error::CodegenError;
use crate::stmt::adamw_from_train_block;

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
