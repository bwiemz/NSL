use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, MemFlags};
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
use crate::types::{is_block_filled, is_float_type, nsl_type_to_cl};
use cranelift_codegen::ir::Value;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum SourceAdParamDiagnosticKind {
    Trainable,
    IgnoredConfig,
    IgnoredNonTensor,
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
const WS_PCIE_FIXED_LAT_US: f64 = 10.0;

/// Fallback for UNPRICED packs (a member's shape is not statically concrete —
/// e.g. a bare `Tensor` field annotation): the v1 structural heuristic, which
/// the GPU bit-exactness gates shipped and validated under. A priced edge is
/// calibrated-safe; an unpriced edge is merely heuristic (review M3: never
/// treat a 0-byte pricing as a real transfer estimate).
const WS_PREFETCH_MIN_OPS_PER_RANGE: usize = 4;

fn is_trainable_param_leaf_name(param_name: &str) -> bool {
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
fn is_data_section_config_pair(stmt: &Stmt, interner: &nsl_lexer::Interner) -> bool {
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
/// When a plan is produced, it is stashed on `compiler.last_wrga_plan` for
/// later observability (`nsl check --wrga-report`).
/// CPDT driver bridge — mirrors `invoke_wrga_if_enabled`. Builds a `CpdtInput`
/// from the WGGO `AppliedPlan`, the optional `@train` block (for AdamW hyper-
/// parameters), and the cluster topology stashed on the Compiler, then stores
/// the resulting plan on `compiler.cpdt_plan`.
///
/// No-op when `compiler.cpdt_mode == CpdtMode::Off` or when no cluster is
/// configured.
pub(crate) fn invoke_cpdt_if_enabled(
    compiler: &mut crate::compiler::Compiler,
    applied_plan: &crate::wggo_apply::AppliedPlan,
    train_block: Option<&nsl_ast::block::TrainBlock>,
) {
    // Experimental subsystem (CPDT). Compiled in by default; a build that opts
    // out (`--no-default-features` without `experimental-cpdt`) turns CPDT
    // planning into a no-op here. See STATUS.md / docs/architecture/.
    #[cfg(not(feature = "experimental-cpdt"))]
    {
        let _ = (compiler, applied_plan, train_block);
        return;
    }
    use crate::cpdt::{CpdtInput, CpdtMode, run as cpdt_run};
    use crate::cpdt_expert::ExpertConfig;
    use crate::cpdt_joint::JointConfig;
    use crate::cpdt_tier_apply::{compute_tier_agreement, plan_map_noweights, PrecisionConfig};
    use crate::cpdt_zero::ModelSize;
    use crate::wggo_overrides::WggoOverrides;

    if compiler.cpdt_mode == CpdtMode::Off {
        return;
    }
    let Some(cluster) = compiler.cpdt_cluster.clone() else {
        return;
    };

    let overrides = WggoOverrides::from_applied(applied_plan);
    let mut model = ModelSize::from_applied_plan(applied_plan);
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
        && !compiler.compile_options.checkpoint_policies.is_empty();
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
    if let Some(wm) = weight_map_ref {
        if compiler.cpdt_mode == CpdtMode::Full {
            if let Err(e) = crate::cpdt_sensitivity::validate(wm, applied_plan) {
                eprintln!("error: {}", e);
                std::process::exit(1);
            }
        }
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
        wggo_recommended_shard: overrides.min_shard_factor(),
    };

    let plan = cpdt_run(input);

    // Tier-agreement diagnostic requires a populated precision plan, which
    // cpdt::run only builds under CpdtMode::Full. Under ZeroOnly the precision
    // field is default-empty regardless of whether weights were supplied, so
    // gating only on `weights_present` would emit a meaningless 100% / 0-of-0
    // line. Tie the diagnostic to the mode that actually exercises the scorer.
    let precision_plan_built = compiler.cpdt_mode == CpdtMode::Full;

    if weights_present && precision_plan_built {
        if let Some(wm) = weight_map_ref {
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
            eprintln!(
                "[cpdt] weight-aware tier agreement: {:.2}% ({}/{} layers, \
                 parameter-weighted {:.2}%)",
                layer_pct, agree_layers, total_layers, param_pct,
            );
            if param_pct < 95.0 {
                eprintln!(
                    "warning: weight-aware tier agreement below 95% (parameter-weighted \
                     {:.2}%). This may indicate that the calibration constants do not fit \
                     this weight distribution well. Phase 2's spectral factor + sidecar \
                     cache narrow this gap; see docs/superpowers/specs/\
                     2026-04-18-cpdt-weight-aware-phase2-stub.md.",
                    param_pct
                );
            }

            if let Ok(val) = std::env::var("CPDT_CALIB_K") {
                eprintln!(
                    "warning: CPDT_CALIB_K={val} is set but ignored. Weights are present, \
                     so the computed gradient_magnitude_est is authoritative. If CPDT_CALIB_K \
                     is vestigial in your shell, you can unset it to silence this warning."
                );
            }
        }
    }

    // Publish to the CLI-owned output slot (if any) so `nsl build` can
    // render the plan after compile returns without threading it through
    // every entry-point's return tuple.
    if let Some(slot) = compiler.compile_options.cpdt.plan_out.as_ref() {
        if let Ok(mut guard) = slot.lock() {
            *guard = Some(plan.clone());
        }
    }
    compiler.cpdt_plan = Some(plan);
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
        wggo_overrides: compiler.wggo_overrides.as_ref(),
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
                if let Some(r) = adapter_cfg.rank {
                    if r > 0 {
                        placement.suggested_rank = r as usize;
                    }
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
    // `last_wrga_plan` is always overwritten — the train-block plan is
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
        compiler.adapter_sites = sites;
    }
    compiler.last_wrga_plan = Some(plan.clone());
    Some(plan)
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

fn classify_source_ad_param_name(
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

        if let ExprKind::Ident(source_sym) = &expr.kind {
            if state.param_symbols.contains(source_sym)
                || state.non_owning_symbols.contains(source_sym)
            {
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
        if let ExprKind::MemberAccess { object, .. } = &expr.kind {
            if matches!(
                self.node_type(object.id),
                nsl_semantic::types::Type::Model { .. }
            ) {
                state.non_owning_symbols.insert(target_sym);
                return;
            }
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
                    let var = state.new_variable();
                    builder.declare_var(var, cl_types::I64);
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
                                    let var = state.new_variable();
                                    builder.declare_var(var, cl_types::I64);
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
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
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
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
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
                        let var = state.new_variable();
                        builder.declare_var(var, cl_types::I64);
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

    pub fn compile_stmt(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        stmt: &Stmt,
    ) -> Result<(), CodegenError> {
        if let Some(block) = state.current_block {
            if is_block_filled(builder, block) {
                return Ok(());
            }
        }

        // Clear any stale lambda capture count from a previous statement
        // (only VarDecl should consume this; if it leaks past a statement boundary it's a bug)
        self.registry.last_lambda_capture_count = None;

        match &stmt.kind {
            StmtKind::VarDecl { pattern, value, .. } => {
                match &pattern.kind {
                    PatternKind::Ident(sym) => {
                        let sym = *sym;
                        if let Some(expr) = value {
                            if self.expr_is_borrowed_batch_handle(state, expr) {
                                return Err(CodegenError::new(format!(
                                    "cannot bind DataLoader batch handle '{}' directly; access batch fields instead",
                                    self.resolve_sym(sym)
                                )));
                            }
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
                            builder.def_var(var, init_val);
                        } else {
                            let var = state.new_variable();
                            builder.declare_var(var, cl_type);
                            builder.def_var(var, init_val);
                            state.variables.insert(sym, (var, cl_type));
                        }

                        // Record semantic type for step-variable cleanup
                        if let Some(expr) = value {
                            state
                                .variable_types
                                .insert(sym, self.node_type(expr.id).clone());
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
                            self.registry.closure_info.insert(sym, count);
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
                                        let var = state.new_variable();
                                        builder.declare_var(var, cl_types::I64);
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
                                let var = state.new_variable();
                                builder.declare_var(var, cl_types::I64);
                                builder.def_var(var, field_val);
                                state.variables.insert(field.name, (var, cl_types::I64));
                                if let Some(field_ty) = field_ty {
                                    state.variable_types.insert(field.name, field_ty);
                                }
                            }
                        }
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
                        match self.get_ownership(state, val) {
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
                                eprintln!(
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
                                cranelift_codegen::ir::MemFlags::new(),
                                val,
                            );
                        }
                    }
                    builder.ins().return_(&[val]);
                } else {
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
                let _ = self.compile_expr(builder, state, expr)?;
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

                // Compile the nested function body
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
                let var = state.new_variable();
                builder.declare_var(var, cl_types::I64);
                builder.def_var(var, addr);
                state.variables.insert(fn_def.name, (var, cl_types::I64));
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

            StmtKind::QuantBlock(ref quant) => {
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
                    if d.name.len() == 1 && self.resolve_sym(d.name[0]) == "cpdt" {
                        if let Some(args) = &d.args {
                            for arg in args {
                                if let Some(name_sym) = arg.name {
                                    if self.resolve_sym(name_sym) == "weight_aware" {
                                        if let nsl_ast::expr::ExprKind::BoolLiteral(b) =
                                            arg.value.kind
                                        {
                                            self.cpdt_weight_aware = b;
                                        }
                                    }
                                }
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
                                    {
                                        if ops.len() >= 2 {
                                            self.fusion
                                                .fused_fns
                                                .insert(fname.clone(), (ops, num_params));
                                        }
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
                                            if arg_name == "start_rule" {
                                                if let nsl_ast::expr::ExprKind::StringLiteral(s) =
                                                    &arg.value.kind
                                                {
                                                    start_rule = s.clone();
                                                }
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
                // `compile_options.inspect_enabled` is true and the stmt
                // is a `let x = ...`.
                if self.compile_options.inspect_enabled {
                    if let StmtKind::VarDecl { pattern, .. } = &stmt.kind {
                        if let PatternKind::Ident(target_sym) = &pattern.kind {
                            for d in decorators {
                                if d.name.len() == 1
                                    && self.resolve_sym(d.name[0]) == "inspect"
                                {
                                    self.emit_inspect_hook(builder, state, d, *target_sym)?;
                                }
                            }
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
                        match self.get_ownership(state, final_val) {
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
                                        cranelift_codegen::ir::MemFlags::trusted(),
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
                                                builder.ins().icmp_imm(IntCC::Equal, new_val, 0);
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
                                    cranelift_codegen::ir::MemFlags::trusted(),
                                    final_val,
                                    obj_val,
                                    field.offset as i32,
                                );
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
                        if matches!(op, AssignOp::Assign) {
                            if let Some(layout) =
                                self.types.struct_layouts.get(&model_name).cloned()
                            {
                                if let Some(slot_off) = layout.adapter_sidetable_offset {
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
                                        cranelift_codegen::ir::MemFlags::trusted(),
                                        obj_val,
                                        slot_off as i32,
                                    );
                                    let byte_off = (index * 8) as i32;
                                    // Free the existing tensor in the slot
                                    // before overwriting (side-table owns
                                    // the tensors it holds).
                                    let old_ptr = builder.ins().load(
                                        cl_types::I64,
                                        cranelift_codegen::ir::MemFlags::trusted(),
                                        table_ptr,
                                        byte_off,
                                    );
                                    self.compile_call_by_name(
                                        builder,
                                        "nsl_tensor_free_if_valid",
                                        &[old_ptr],
                                    )?;
                                    builder.ins().store(
                                        cranelift_codegen::ir::MemFlags::trusted(),
                                        new_val,
                                        table_ptr,
                                        byte_off,
                                    );
                                    return Ok(());
                                }
                            }
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
                                        cranelift_codegen::ir::MemFlags::trusted(),
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
                                                builder.ins().icmp_imm(IntCC::Equal, new_val, 0);
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
                                    cranelift_codegen::ir::MemFlags::trusted(),
                                    final_val,
                                    obj_val,
                                    field.offset as i32,
                                );
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
    pub(crate) fn eltls_predeclare_loop_lets(
        &self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        syms: &[nsl_ast::Symbol],
    ) {
        if syms.is_empty() {
            return;
        }
        let zero = builder.ins().iconst(cl_types::I64, 0);
        for &sym in syms {
            if state.variables.contains_key(&sym) {
                continue;
            }
            if state.param_symbols.contains(&sym) {
                continue;
            }
            let var = state.new_variable();
            builder.declare_var(var, cl_types::I64);
            builder.def_var(var, zero);
            state.variables.insert(sym, (var, cl_types::I64));
            state.eltls_loop_predeclared.insert(sym);
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
        if let Some(block) = state.current_block {
            if is_block_filled(builder, block) {
                return;
            }
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
    pub(crate) fn free_tensor_temporaries(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        keep: Option<Value>,
    ) {
        let temps = std::mem::take(&mut state.cleanup.tensor_temporaries);
        // Tape ID identity: intermediates can now be safely freed during tape recording
        // because TapeOps use monotonic tape_ids as identity keys (not raw pointers).
        for temp in &temps {
            if Some(*temp) == keep {
                continue;
            }
            // Emit nsl_tensor_free(temp) — but only if block is not already filled
            if let Some(block) = state.current_block {
                if is_block_filled(builder, block) {
                    break;
                }
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
            if let Some(block) = state.current_block {
                if is_block_filled(builder, block) {
                    break;
                }
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
            if let Some(block) = state.current_block {
                if is_block_filled(builder, block) {
                    break;
                }
            }
            let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[*val]);
        }
    }

    /// Emit nsl_tensor_free calls for all tensor temporaries accumulated since the
    /// current loop scope started. Used at break/continue points.
    /// CRITICAL: Does NOT truncate tensor_temporaries — that only happens at natural scope exit.
    fn emit_loop_scope_cleanup(&mut self, builder: &mut FunctionBuilder, state: &mut FuncState) {
        if let Some(&scope_start) = state.cleanup.temp_scope_stack.last() {
            for &temp in &state.cleanup.tensor_temporaries[scope_start..] {
                if let Some(block) = state.current_block {
                    if is_block_filled(builder, block) {
                        break;
                    }
                }
                let _ = self.compile_call_by_name(builder, "nsl_tensor_free", &[temp]);
            }
        }
    }

    /// Emit cleanup AND truncate temporaries at natural loop exit.
    fn cleanup_loop_scope(&mut self, builder: &mut FunctionBuilder, state: &mut FuncState) {
        if let Some(scope_start) = state.cleanup.temp_scope_stack.pop() {
            for &temp in &state.cleanup.tensor_temporaries[scope_start..] {
                if let Some(block) = state.current_block {
                    if is_block_filled(builder, block) {
                        break;
                    }
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
            if let Some(block) = state.current_block {
                if is_block_filled(builder, block) {
                    break;
                }
            }
            let batch_ptr = builder.use_var(batch_var);
            let _ = self.compile_call_by_name(builder, "nsl_dict_free_tensor_values", &[batch_ptr]);
        }
        for loop_ctx in state.loop_stack.iter().rev() {
            let Some(batch_var) = loop_ctx.batch_var else {
                continue;
            };
            if let Some(block) = state.current_block {
                if is_block_filled(builder, block) {
                    break;
                }
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
            if let Some(block) = state.current_block {
                if is_block_filled(builder, block) {
                    break;
                }
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
        for s in &then_block.stmts {
            self.compile_stmt(builder, state, s)?;
        }
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
            for s in &elif_body.stmts {
                self.compile_stmt(builder, state, s)?;
            }
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
            for s in &else_body.stmts {
                self.compile_stmt(builder, state, s)?;
            }
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

        builder.ins().jump(header_block, &[]);

        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let cond_val = self.compile_expr(builder, state, condition)?;
        builder
            .ins()
            .brif(cond_val, body_block, &[], exit_block, &[]);

        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        // ELTLS Task 16.1: pre-declare top-level let-ident symbols from the
        // body so the second-and-later rebinds fire eltls_clear_old_slot
        // and free the previous iteration's tensor.
        let predecl_syms = self.eltls_collect_loop_let_idents(&body.stmts);
        self.eltls_predeclare_loop_lets(builder, state, &predecl_syms);

        state
            .cleanup
            .temp_scope_stack
            .push(state.cleanup.tensor_temporaries.len());
        state.loop_stack.push(LoopContext {
            continue_block: header_block,
            exit_block,
            batch_var: None,
        });
        for s in &body.stmts {
            self.compile_stmt(builder, state, s)?;
        }
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
                let var = state.new_variable();
                builder.declare_var(var, cl_types::I64);
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

        builder.ins().jump(header_block, &[]);

        // Header: evaluate expression, check truthiness (non-zero = continue)
        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let val = self.compile_expr(builder, state, expr)?;
        let cond = builder.ins().icmp_imm(IntCC::NotEqual, val, 0);
        builder.ins().brif(cond, body_block, &[], exit_block, &[]);

        // Body: update pattern variable with current value, execute body
        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        // Update the pattern variable with the value from this iteration
        if let Some(var) = pattern_var {
            builder.def_var(var, val);
        }

        // ELTLS Task 16.1: pre-declare top-level let-ident symbols from
        // the body so rebinds across iterations free the previous value.
        let predecl_syms = self.eltls_collect_loop_let_idents(&body.stmts);
        self.eltls_predeclare_loop_lets(builder, state, &predecl_syms);

        state
            .cleanup
            .temp_scope_stack
            .push(state.cleanup.tensor_temporaries.len());
        state.loop_stack.push(LoopContext {
            continue_block: header_block,
            exit_block,
            batch_var: None,
        });
        for s in &body.stmts {
            self.compile_stmt(builder, state, s)?;
        }
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
            eprintln!(
                "[nsl-codegen] warning: for-loop iterable has Unknown type — compiling as list iteration. \
                 If this is a DataLoader, ensure the variable type is inferred correctly."
            );
        }

        let list_val = self.compile_expr(builder, state, iterable)?;

        let len_id = self.registry.runtime_fns["nsl_list_len"].0;
        let len_ref = self.module.declare_func_in_func(len_id, builder.func);
        let call = builder.ins().call(len_ref, &[list_val]);
        let list_len = builder.inst_results(call)[0];

        let counter_var = state.new_variable();
        builder.declare_var(counter_var, cl_types::I64);
        let zero = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(counter_var, zero);

        // Pre-declare pattern variables before the loop
        match &pattern.kind {
            PatternKind::Ident(sym) => {
                let elem_var = state.new_variable();
                builder.declare_var(elem_var, cl_types::I64);
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
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            builder.def_var(var, zero);
                            state.variables.insert(*sym, (var, cl_types::I64));
                        }
                        PatternKind::Rest(Some(sym)) => {
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
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

        // ELTLS Task 16.1: pre-declare top-level let-ident symbols from the
        // body so the second-and-later rebinds fire eltls_clear_old_slot.
        // This is the primary fix for the `for batch in dl: let y = model(batch)`
        // training leak pattern.
        let predecl_syms = self.eltls_collect_loop_let_idents(&body.stmts);
        self.eltls_predeclare_loop_lets(builder, state, &predecl_syms);

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
        for s in &body.stmts {
            self.compile_stmt(builder, state, s)?;
        }
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
        let elem_var = state.new_variable();
        builder.declare_var(elem_var, cl_types::I64);
        builder.def_var(elem_var, zero);
        state
            .variables
            .insert(loop_var_sym, (elem_var, cl_types::I64));

        // Register the loop variable's model type for method dispatch
        let model_name = self.resolve_sym(element_model).to_string();
        self.models.model_var_types.insert(loop_var_sym, model_name);

        // Counter variable
        let counter_var = state.new_variable();
        builder.declare_var(counter_var, cl_types::I64);
        builder.def_var(counter_var, zero);

        let header_block = builder.create_block();
        let body_block = builder.create_block();
        let increment_block = builder.create_block();
        let exit_block = builder.create_block();

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
            .load(cl_types::I64, MemFlags::trusted(), addr, 0);
        builder.def_var(elem_var, elem_ptr);

        // ELTLS Task 16.1: pre-declare top-level let-ident symbols from body.
        let predecl_syms = self.eltls_collect_loop_let_idents(&body.stmts);
        self.eltls_predeclare_loop_lets(builder, state, &predecl_syms);

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
        for s in &body.stmts {
            self.compile_stmt(builder, state, s)?;
        }
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
        let batch_var = state.new_variable();
        builder.declare_var(batch_var, cl_types::I64);
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

        builder.ins().jump(header_block, &[]);

        // Header: call nsl_dataloader_next_batch, branch on null
        builder.switch_to_block(header_block);
        state.current_block = Some(header_block);
        let batch_ptr =
            self.compile_call_by_name(builder, "nsl_dataloader_next_batch", &[dl_val])?;
        builder.def_var(batch_var, batch_ptr);
        let is_null = builder.ins().icmp_imm(IntCC::Equal, batch_ptr, 0);
        builder
            .ins()
            .brif(is_null, exhausted_exit_block, &[], body_block, &[]);

        // Body: compile loop body statements
        // break → break_exit_block (frees batch, then stops DL)
        // continue → cleanup_block (frees batch, loops back)
        builder.switch_to_block(body_block);
        builder.seal_block(body_block);
        state.current_block = Some(body_block);

        // ELTLS Task 16.1: pre-declare top-level let-ident symbols from body.
        // THIS IS THE PRIMARY FIX FOR THE TRAINING-LOOP LEAK:
        //   for batch in dataloader:
        //       let y = model(batch)     # previously leaked y each iteration
        //       let loss = loss_fn(y, batch["labels"])
        //       ...
        // By pre-declaring y and loss in the pre-loop scope with zero init,
        // each subsequent rebind is detected as a reassignment and
        // eltls_clear_old_slot fires nsl_tensor_free_if_valid on the
        // previous iteration's tensor.
        let predecl_syms = self.eltls_collect_loop_let_idents(&body.stmts);
        self.eltls_predeclare_loop_lets(builder, state, &predecl_syms);

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
        for s in &body.stmts {
            self.compile_stmt(builder, state, s)?;
        }
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
                    for s in &arm.body.stmts {
                        self.compile_stmt(builder, state, s)?;
                    }
                    state.flags.conditional_depth -= 1;
                    if let Some(block) = state.current_block {
                        if !is_block_filled(builder, block) {
                            builder.ins().jump(merge_block, &[]);
                        }
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
                        for s in &arm.body.stmts {
                            self.compile_stmt(builder, state, s)?;
                        }
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
                        let var = state.new_variable();
                        builder.declare_var(var, cl_types::I64);
                        builder.def_var(var, subject_val);
                        state.variables.insert(*sym, (var, cl_types::I64));
                        state.flags.conditional_depth += 1;
                        for s in &arm.body.stmts {
                            self.compile_stmt(builder, state, s)?;
                        }
                        state.flags.conditional_depth -= 1;
                        if let Some(block) = state.current_block {
                            if !is_block_filled(builder, block) {
                                builder.ins().jump(merge_block, &[]);
                            }
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
                    for s in &arm.body.stmts {
                        self.compile_stmt(builder, state, s)?;
                    }
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
                        for s in &arm.body.stmts {
                            self.compile_stmt(builder, state, s)?;
                        }
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
        if let Some(block) = state.current_block {
            if block != merge_block && !is_block_filled(builder, block) {
                builder.ins().jump(merge_block, &[]);
            }
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

        // ── Extract distill config (semantic layer already validated) ──
        let mut teacher_sym: Option<nsl_ast::Symbol> = None;
        let mut student_sym: Option<nsl_ast::Symbol> = None;
        let mut epochs: i64 = 1;
        for arg in &distill.config {
            if let Some(name_sym) = arg.name {
                match self.resolve_sym(name_sym) {
                    "teacher" => {
                        if let ExprKind::Ident(sym) = &arg.value.kind {
                            teacher_sym = Some(*sym);
                        }
                    }
                    "student" => {
                        if let ExprKind::Ident(sym) = &arg.value.kind {
                            student_sym = Some(*sym);
                        }
                    }
                    "epochs" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            epochs = *n;
                        }
                    }
                    _ => {}
                }
            }
        }
        let teacher_sym = teacher_sym.ok_or_else(|| {
            CodegenError::new("distill block requires 'teacher=<model ident>'")
        })?;
        let student_sym = student_sym.ok_or_else(|| {
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

        // ── Synthetic TrainBlock: empty config (the context seeds
        //    model/epochs), shared sections. ─────────────────────────────
        let synthetic = nsl_ast::block::TrainBlock {
            config: Vec::new(),
            sections: distill.sections.clone(),
            span: distill.span,
        };

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
        if result.is_ok() {
            if let Some(plan) = self.last_cpkd_plan.take() {
                eprint!("{}", plan.render_report());
            }
        }
        result
    }

    /// Emit a `zeros_like` allocation for ONE optimizer-moment buffer (m or
    /// v), honoring the offload / CPDT-precision variants. `precision_list` is
    /// the per-param dtype-code list for THIS moment when a precision plan is
    /// active (`None` = verbatim f32). Factored out of the optimizer-state
    /// init loop so the D3-v2 owner-gated (ZeRO-1 shard) path can reuse the
    /// exact same allocation logic inside its owned branch.
    fn emit_moment_zeros_like(
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
    fn emit_owner_gated_moment(
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
        builder.ins().jump(merge_b, &[real]);

        // Non-owner: allocate nothing — push a null (0) placeholder.
        builder.switch_to_block(skip_b);
        builder.seal_block(skip_b);
        state.current_block = Some(skip_b);
        let null = builder.ins().iconst(cl_types::I64, 0);
        builder.ins().jump(merge_b, &[null]);

        // Merge — both predecessors connected, safe to seal.
        builder.switch_to_block(merge_b);
        builder.seal_block(merge_b);
        state.current_block = Some(merge_b);
        Ok(builder.block_params(merge_b)[0])
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
    fn emit_callback_residency_open(
        &mut self,
        builder: &mut FunctionBuilder,
        body: &nsl_ast::stmt::Block,
        model_sym: nsl_ast::Symbol,
        cb_name: &str,
    ) -> Result<Option<bool>, CodegenError> {
        if !self.compile_options.weight_stream {
            return Ok(None);
        }
        let touch = self.analyze_callback_model_touch(body, model_sym);
        if !touch.touches {
            return Ok(None);
        }
        eprintln!(
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
    fn emit_callback_residency_close(
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
    fn emit_ws_pack_upload(
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
    fn emit_ws_pack_evict(
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
    fn emit_ws_pack_single(
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
            return self.compile_train_block_pipelined(
                builder,
                state,
                train,
                train_block_stmt_id,
            );
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
        // selection and the per-param mode table read `self.wggo_overrides`
        // well before the in-place planning site in the same function; the
        // pre-pass is what finally lets them see the plan they were written
        // to consume. The offer is validated against the codegen-time
        // extraction at the planning site (graph-fingerprint check) and
        // replaced by an in-place solve on mismatch.
        self.wggo_overrides = self
            .wggo_preplans
            .iter()
            .find(|p| p.train_block_stmt_id == train_block_stmt_id)
            .map(|p| p.overrides.clone());
        let result =
            self.compile_train_block_inner(builder, state, train, train_block_stmt_id);
        self.restore_active_fused_ce_config(saved_active_fused_ce);
        result
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
        let mut model_sym: Option<nsl_ast::Symbol> = None;
        let mut epochs: i64 = 1;
        let mut grad_accumulation_steps: i64 = 1;
        let mut grad_clip: f64 = f64::MAX; // default: no clipping

        for arg in &train.config {
            if let Some(name_sym) = arg.name {
                let name = self.resolve_sym(name_sym).to_string();
                match name.as_str() {
                    "model" => {
                        if let ExprKind::Ident(sym) = &arg.value.kind {
                            model_sym = Some(*sym);
                        } else {
                            return Err(CodegenError::new(
                                "train 'model' arg must be an identifier",
                            ));
                        }
                    }
                    "epochs" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            epochs = *n;
                        } else {
                            return Err(CodegenError::new(
                                "train 'epochs' arg must be an integer literal",
                            ));
                        }
                    }
                    "grad_accumulation" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            grad_accumulation_steps = (*n).max(1);
                        }
                    }
                    "grad_clip" => {
                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                            grad_clip = *f;
                        } else if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            grad_clip = *n as f64;
                        }
                    }
                    _ => {} // ignore unknown config for forward compat
                }
            }
        }

        // CPKD: a distill block delegates here with an empty config — the
        // student plays the `model=` role and epochs come from the distill
        // header. (`compile_distill_block` installed the context.)
        if let Some(distill) = &self.active_distill_context {
            model_sym = Some(distill.student_sym);
            epochs = distill.epochs;
        }

        let model_sym = model_sym.ok_or_else(|| {
            CodegenError::new("train block requires 'model=<ident>' config argument")
        })?;

        // ── 2. Extract optimizer info from sections ─────────────────────
        let mut optimizer_name = String::new();
        let mut lr_value: f64 = 0.01;
        let mut momentum_value: f64 = 0.0;
        let mut dampening_value: f64 = 0.0;
        let mut weight_decay_value: f64 = 0.0;
        let mut nesterov_value: bool = false;
        let mut beta1_value: f64 = 0.9;
        let mut beta2_value: f64 = 0.999;
        let mut eps_value: f64 = 1e-8;
        let mut step_body: Option<(&nsl_ast::stmt::Block, nsl_ast::Symbol)> = None;
        let mut callbacks: Vec<&nsl_ast::block::CallbackDef> = Vec::new();
        let mut scheduler_name = String::new();
        let mut scheduler_args: Vec<(String, f64)> = Vec::new();

        for section in &train.sections {
            match section {
                TrainSection::Optimizer(expr) => {
                    // Parse call like SGD(lr=0.01, momentum=0.9)
                    if let ExprKind::Call { callee, args } = &expr.kind {
                        if let ExprKind::Ident(sym) = &callee.kind {
                            optimizer_name = self.resolve_sym(*sym).to_string().to_lowercase();
                        }
                        for arg in args {
                            if let Some(name_sym) = arg.name {
                                let name = self.resolve_sym(name_sym).to_string();
                                match name.as_str() {
                                    "lr" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            lr_value = *f;
                                        } else if let ExprKind::IntLiteral(n) = &arg.value.kind {
                                            lr_value = *n as f64;
                                        }
                                    }
                                    "momentum" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            momentum_value = *f;
                                        }
                                    }
                                    "dampening" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            dampening_value = *f;
                                        }
                                    }
                                    "weight_decay" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            weight_decay_value = *f;
                                        }
                                    }
                                    "nesterov" => {
                                        if let ExprKind::BoolLiteral(b) = &arg.value.kind {
                                            nesterov_value = *b;
                                        }
                                    }
                                    "beta1" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            beta1_value = *f;
                                        }
                                    }
                                    "beta2" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            beta2_value = *f;
                                        }
                                    }
                                    "eps" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            eps_value = *f;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                TrainSection::Step { param, body } => {
                    step_body = Some((body, *param));
                }
                TrainSection::Callbacks(cbs) => {
                    callbacks.extend(cbs.iter());
                }
                TrainSection::Scheduler(expr) => {
                    if let ExprKind::Call { callee, args } = &expr.kind {
                        if let ExprKind::Ident(sym) = &callee.kind {
                            scheduler_name = self.resolve_sym(*sym).to_string();
                        }
                        for arg in args {
                            if let Some(name_sym) = arg.name {
                                let name = self.resolve_sym(name_sym).to_string();
                                if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                    scheduler_args.push((name, *f));
                                } else if let ExprKind::IntLiteral(n) = &arg.value.kind {
                                    scheduler_args.push((name, *n as f64));
                                }
                            }
                        }
                    }
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

        if optimizer_name.is_empty() {
            return Err(CodegenError::new(
                "train block requires an optimizer section",
            ));
        }

        let (step_body, step_param_sym) =
            step_body.ok_or_else(|| CodegenError::new("train block requires a step section"))?;

        // FASE: plan the backward rewrite.  Passthrough (N=1) and FullBuffer
        // (Lion, Unknown) fall through to the existing accum-buffer path
        // below.  Deferred routes through stmt_fase.
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
            allow_v_approx: true,
        };
        let fase_plan = match self.wggo_overrides.as_ref() {
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
                            eprintln!(
                                "[fase] NSL_FASE_FUSED_OVERRIDE applied: {spec} \
                                 (replacing plan fase_fused for {} layers)",
                                fused.len()
                            );
                            fused = forced;
                        }
                        Some(forced) => {
                            eprintln!(
                                "[fase] NSL_FASE_FUSED_OVERRIDE ignored: {} entries \
                                 for {} WGGO layers",
                                forced.len(),
                                fused.len()
                            );
                        }
                        None => {
                            eprintln!(
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
        for diag in &fase_plan.override_diagnostics {
            let reason_str = match &diag.reason {
                crate::wggo_overrides::OverrideRejectReason::FaseModeInfeasible {
                    optimizer,
                    global_mode,
                } => format!("{:?}_optimizer_global_mode_{:?}", optimizer, global_mode)
                    .to_lowercase(),
                other => format!("{:?}", other),
            };
            eprintln!(
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
        let fase_deferred = fase_plan.mode == crate::fase::FaseMode::Deferred;

        // ── CSLA Stage-2 (`--layerwise-accum`) admission ────────────────
        // Window-buffered scheduling only composes with the exact set of
        // paths it was validated on; everything else refuses loudly (the
        // repo's deferral-must-refuse rule) instead of silently running the
        // interleaved baseline under a flag that claims otherwise.
        let csla_active = self.compile_options.layerwise_accum;
        if self.compile_options.weight_stream && !csla_active {
            return Err(CodegenError::new(
                "--weight-stream requires --layerwise-accum (the window-scoped \
                 eviction cycle is defined by the layer-major schedule)",
            ));
        }
        if csla_active {
            if !self.features.source_ad_enabled {
                return Err(CodegenError::new(
                    "--layerwise-accum requires --source-ad: the window-buffered \
                     schedule replays the compile-time adjoint tape; the runtime \
                     tape-AD backward cannot be partitioned",
                ));
            }
            if grad_accumulation_steps <= 1 {
                return Err(CodegenError::new(
                    "--layerwise-accum requires grad_accumulation >= 2 in the train \
                     block: with a single-micro-batch window there is no \
                     accumulation window to buffer",
                ));
            }
            if !fase_deferred {
                return Err(CodegenError::new(format!(
                    "--layerwise-accum requires a FASE-Deferred plan (AdamW/Adam + \
                     grad_accumulation >= 2); this train block resolved to \
                     {:?}. Use AdamW or Adam, or drop --layerwise-accum",
                    fase_plan.mode
                )));
            }
            if grad_clip < f64::MAX {
                return Err(CodegenError::new(
                    "--layerwise-accum is incompatible with grad_clip: two-phase \
                     clipping needs the GLOBAL L2 norm over every parameter's \
                     completed m_partial before any update, which the layerwise \
                     schedule never materializes. Remove grad_clip or drop \
                     --layerwise-accum",
                ));
            }
            // D2a: --optim-state-offload NOW COMPOSES. The D1a refusal
            // protected against the P3 m_partial-staging half of the flag,
            // which is structurally moot under csla (the accumulator branch
            // checks csla FIRST, so slots stay NULL/device — host m_partial
            // cannot resurrect). What remains is exactly D2a's target: m/v
            // allocate host-pinned (the existing P0.2 path) and stage
            // per LAYER through fase_emit_final_step's wrap_offload
            // envelope at the per-layer update sites, with a drain after
            // each group update. The window's own accumulate hook stays
            // device-resident (wrap_offload=false at the hook site).
            if self.compile_options.checkpoint_compress.is_some() {
                return Err(CodegenError::new(
                    "--layerwise-accum is incompatible with --checkpoint-compress: \
                     the layerwise gate is bit-exact and compressed saves are not",
                ));
            }
            if self.features.zero_stage.filter(|&s| s >= 1).is_some() {
                return Err(CodegenError::new(
                    "--layerwise-accum is incompatible with --zero-stage: the M43 \
                     ZeRO hooks read every accum slot after the backward, but the \
                     layerwise schedule's per-layer accumulators are freed before \
                     the optimizer gate. Drop one",
                ));
            }
        }

        // D3 v1: only ZeRO-1 (optimizer-state sharding: owner-gated updates
        // + post-step parameter broadcast over the CPU-shm SimulatedBackend)
        // is lowered. Grad sharding (stage 2: per-shard reduce-scatter into
        // sharded accumulators) and param sharding (stage 3: JIT all-gather
        // in the forward) need sharded BUFFERS, not just sharded compute —
        // refuse loudly rather than silently running stage-1 semantics.
        if let Some(s) = self.features.zero_stage {
            // P4 item 16: stage 2 (gradient partitioning via owner-segmented
            // reduce_scatter) is lowered — same emission points as stage 1;
            // the runtime dispatches on the baked stage. Stage 3 (parameter
            // partitioning + just-in-time all-gather) still refuses: params
            // freed between steps would break mid-loop model_save/callbacks
            // and eval reads, which need the residency machinery generalized
            // from the weight-stream Item-12 guard first.
            if s >= 3 {
                return Err(CodegenError::new(format!(
                    "--zero-stage {s} is not lowered yet: stages 1 (optimizer \
                     sharding) and 2 (gradient partitioning) are implemented. \
                     Stage 3 parameter partitioning needs the callback/\
                     model_save residency guard generalized to sharded params \
                     — use --zero-stage 2",
                )));
            }
            // D3 v1 (review): --zero-stage x grad_clip is unsafe and unlowered.
            // The all-reduce is emitted BEFORE the Deferred dispatch, but
            // two-phase clip folds the final micro-batch gradient into
            // m_partial AFTER it and derives the clip factor from a
            // RANK-LOCAL norm — so with real (non-replicated) data each rank
            // would clip differently and the owner's update would diverge.
            // It only looks bit-exact today because the loader is rank-blind.
            // Refuse, mirroring the zero x mode-table / zero x layerwise
            // refusals, rather than ship a latent miscompile.
            if s >= 1 && grad_clip != f64::MAX {
                return Err(CodegenError::new(
                    "--zero-stage is incompatible with grad_clip: the gradient \
                     all-reduce precedes the two-phase clip, whose norm is \
                     computed rank-locally, so clipped multi-rank training \
                     would be silently wrong. Drop grad_clip or --zero-stage",
                ));
            }
        }

        // ── 3. Resolve model type and build param list ──────────────────
        // Get the model pointer from state
        let (model_var, _) = *state.variables.get(&model_sym).ok_or_else(|| {
            CodegenError::new(format!(
                "undefined model variable '{}' in train block",
                self.resolve_sym(model_sym)
            ))
        })?;
        let model_ptr = builder.use_var(model_var);

        // Resolve model type name from the variable's semantic type.
        // First try model_var_types (set for for-loop model vars), then
        // state.variable_types (set for let-bound vars), then fall back to
        // scanning the type_map (unreliable — picks the first model type).
        let model_var_name = self.resolve_sym(model_sym).to_string();
        let model_type_name = self
            .models
            .model_var_types
            .get(&model_sym)
            .cloned()
            .or_else(|| {
                // Check semantic type from variable_types
                state
                    .variable_types
                    .get(&model_sym)
                    .and_then(|ty| match ty {
                        nsl_semantic::types::Type::Model { name, .. } => {
                            Some(self.resolve_sym(*name).to_string())
                        }
                        nsl_semantic::types::Type::Struct { name, .. } => {
                            Some(self.resolve_sym(*name).to_string())
                        }
                        _ => None,
                    })
            })
            .unwrap_or_else(|| model_var_name.clone());

        let layout = self
            .types
            .struct_layouts
            .get(&model_type_name)
            .cloned()
            .ok_or_else(|| {
                CodegenError::new(format!(
                    "no struct layout found for model '{}' in train block",
                    model_type_name
                ))
            })?;

        // Build param_list directly from the compiler's struct layouts instead
        // of the runtime pointer-probing collector. This keeps nested models
        // and fixed arrays aligned with the paths source AD already resolves.
        // Persistent pool for param_list and optimizer state allocation
        self.compile_call_by_name(builder, "nsl_gpu_set_persistent_pool", &[])?;

        // P0.1 per-surface VRAM accounting: tag values MUST match
        // `nsl_runtime::cuda::caching_allocator::SurfaceTag` (#[repr(u8)]).
        // Each bracket below sets a surface for its allocation region and
        // restores the caller's surface afterwards (get/set — nesting-safe).
        const SURFACE_WEIGHTS: i64 = 1;
        const SURFACE_OPTIM_M: i64 = 2;
        const SURFACE_OPTIM_V: i64 = 3;
        const SURFACE_M_PARTIAL: i64 = 4;
        const SURFACE_GRADS: i64 = 5;
        const SURFACE_ACTIVATIONS: i64 = 6;
        let surface_prev = self.compile_call_by_name(builder, "nsl_gpu_get_alloc_surface", &[])?;
        let surface_weights = builder.ins().iconst(cl_types::I8, SURFACE_WEIGHTS);
        self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_weights])?;

        let param_paths = self.enumerate_model_tensor_paths(&model_var_name, &model_type_name);
        let param_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        for path in &param_paths {
            let param_ptr = self
                .load_nested_field(builder, model_ptr, &layout, &model_type_name, path)
                .ok_or_else(|| {
                    CodegenError::new(format!(
                        "could not resolve model parameter '{}' in train block",
                        path,
                    ))
                })?;
            self.compile_call_by_name(builder, "nsl_list_push", &[param_list, param_ptr])?;
        }
        // End of the Weights bracket — restore the caller's surface.
        self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_prev])?;

        // D3 (ZeRO-1): initialize the real sharding context ONCE at train
        // setup — the missing M43b emitter. The runtime builds the CPU-shm
        // SimulatedBackend from the `--devices N` spawner's env protocol
        // (rank + shm path); world_size is the compile-time value baked
        // here, and round-robin ownership (idx % ws) is established before
        // any optimizer machinery runs. world_size == 1 degenerates to
        // no-op collectives and rank-0 owning everything — identical to
        // the unsharded baseline by construction.
        if self.features.zero_stage.filter(|&s| s >= 1).is_some() {
            let stage_val = builder
                .ins()
                .iconst(cl_types::I64, self.features.zero_stage.unwrap_or(1) as i64);
            let ws_val = builder
                .ins()
                .iconst(cl_types::I64, self.features.world_size.max(1) as i64);
            // D3 (review): ZeRO init/partition return codes were discarded —
            // a refused init (-2 NSL_SIMULATED_TP=0, -3 missing shm path)
            // left ZERO_CTX None, and every owner gate then read
            // owns_param==-1 and skipped ALL updates: the run trained a
            // frozen model and exited 0. Assert the rc so a refusal aborts
            // loudly instead of silently training nothing. (The runtime
            // FFIs still RETURN the code — unit tests exercise the refusal
            // paths directly; only the emitted call is made fatal.)
            let init_rc =
                self.compile_call_by_name(builder, "nsl_zero_init", &[stage_val, ws_val])?;
            let zero_i = builder.ins().iconst(cl_types::I64, 0);
            let init_ok = builder.ins().icmp(IntCC::Equal, init_rc, zero_i);
            let init_msg = "nsl: --zero-stage init failed (see message above) — \
                            aborting instead of training an unsynchronized model";
            self.intern_string(init_msg)?;
            let init_msg_ptr = self.compile_string_literal(builder, init_msg)?;
            self.compile_call_by_name(builder, "nsl_assert", &[init_ok, init_msg_ptr])?;

            let np_val = builder
                .ins()
                .iconst(cl_types::I64, param_paths.len() as i64);
            // P4 item 13: BYTE-balanced ownership — the runtime reads each
            // param's byte size from param_list (identical on every rank) and
            // partitions by greedy LPT, so per-rank optimizer work and moment
            // memory track ~1/N in bytes rather than tensor count. Replaces
            // the index round-robin `nsl_zero_partition`.
            let part_rc = self.compile_call_by_name(
                builder,
                "nsl_zero_partition_bytes",
                &[param_list, np_val],
            )?;
            let part_ok = builder.ins().icmp(IntCC::SignedGreaterThanOrEqual, part_rc, zero_i);
            let part_msg = "nsl: --zero-stage partition failed — aborting";
            self.intern_string(part_msg)?;
            let part_msg_ptr = self.compile_string_literal(builder, part_msg)?;
            self.compile_call_by_name(builder, "nsl_assert", &[part_ok, part_msg_ptr])?;
        }
        let num_params_val = builder
            .ins()
            .iconst(cl_types::I64, param_paths.len() as i64);

        // P0.3: arm the gradient-integrity exit report once, at train setup
        // (before the epoch/step loop), so --grad-integrity works with no env
        // var. The per-step check/note calls below feed the accumulator.
        if self.compile_options.grad_integrity {
            self.compile_call_by_name(builder, "nsl_grad_integrity_arm", &[])?;
            // v1 does not wire the gate into the CSLA (--layerwise-accum)
            // windowed-replay backward, so it would report checks=0 there —
            // silently. Warn loudly rather than let a gate pass vacuously
            // (deferral-must-refuse). The FullBuffer and FASE-interleaved paths
            // are covered.
            if self.compile_options.layerwise_accum {
                eprintln!(
                    "[grad-integrity] WARNING: --grad-integrity is not wired into the \
                     --layerwise-accum (CSLA) windowed backward yet — the report will \
                     show checks=0. Drop --layerwise-accum to check gradients, or treat \
                     checks=0 as 'not measured', not 'no problems'."
                );
            }
        }

        // CPDT precision-adaptive optimizer execution (v1): build per-param
        // storage dtype lists aligned with param_list, when active. Inactive ->
        // None (the existing FP32 path runs verbatim, zero behavior change).
        //
        // Borrow discipline: extract the owned dtype Vecs (and the activation
        // decision) into a local FIRST, which ends the `self.cpdt_plan` borrow.
        // Only then do the `compile_call_by_name` loop (which borrows `self`
        // mutably) run. No `unsafe`, no tensor clones.
        let cpdt_precision_dtypes: Option<(Value, Value)> = {
            let dtype_data: Option<(Vec<u16>, Vec<u16>)> = {
                // Pre-S2: the FASE cast wrapping was emitted ONLY on the
                // non-unified-dispatch Deferred branch (which runs iff WGGO is
                // inactive); the unified-dispatch arm hardcoded
                // `wrap_precision=false`. Allocating FP16 m/v on the WGGO path
                // would have fed FP16 buffers to an unwrapped FP32 update →
                // silent corruption.
                //
                // - S2 threaded the wrap through `emit_unified_optim_step_dispatch`'s
                //   Deferred sub-arm.
                // - S4 relaxed the gate from `wggo_overrides.is_none()` toward
                //   `true`, with a review-fix mode-table FullBuffer guard kept
                //   as the structural correctness backstop.
                // - S5 threads the wrap through `emit_stdlib_optim_call` so
                //   the unified-dispatch FullBuffer sub-arm ALSO wraps. With
                //   both sub-arms wrapping, the silent-corruption hazard is
                //   closed structurally and the FullBuffer guard can be
                //   lifted — `wrapped_path_active = true` unconditionally.
                //   The parameter is retained on `precision_active` as
                //   defense-in-depth for any future refactor that
                //   reintroduces a non-wrapping optimizer arm.
                let wrapped_path_active = true;

                // WGGO-plan moment bits take precedence when the plan
                // actually decided sub-32 storage for any layer: per the
                // paper, the Level-2 ILP chooses p_m/p_v (gated by an
                // informed sensitivity signal via `prec_allowed` — a plan
                // with no weight/calibration evidence carries 32/32 and
                // lands in the None arm here). Params are joined to layers
                // with the same `layer_prefix` the graph builder used;
                // unmatched params stay F32. Requires the Deferred plan
                // (both dispatch arms wrap m/v in the dequant→step→quant
                // envelope). 8-bit clamps to FP16 storage in v1, the same
                // ladder step as `clamp_int8_to_fp16`.
                let wggo_bits: Option<(Vec<u16>, Vec<u16>)> = if fase_deferred {
                    self.wggo_overrides.as_ref().and_then(|o| {
                        crate::cpdt_precision_exec::build_dtype_lists_from_overrides(
                            o,
                            &param_paths,
                        )
                    })
                } else {
                    None
                };

                let plan = self.cpdt_plan.as_ref();
                let active = plan
                    .map(|p| {
                        crate::cpdt_precision_exec::precision_active(
                            matches!(p.mode, crate::cpdt::CpdtMode::Full),
                            !p.precision.params.is_empty(),
                            true, // weights_present is implied by a non-empty precision plan
                            fase_deferred,
                            wrapped_path_active,
                        )
                    })
                    .unwrap_or(false);
                let cpdt_lists = if active {
                    let plan = plan.unwrap();
                    Some(crate::cpdt_precision_exec::build_dtype_lists(
                        &plan.precision,
                        &param_paths,
                    ))
                } else {
                    None
                };

                // Arbitration:
                // - WGGO's plan bits lower ONLY behind the explicit opt-in
                //   (--wggo-moment-precision): reduced-precision moments
                //   change training numerics. The dequant->step->quant cast
                //   envelope runs on BOTH devices (deferral-closure
                //   2026-07-14: nsl_tensor_zeros_like_dtype / nsl_tensor_cast
                //   / nsl_tensor_cast_into dispatch to the CFTP-v7 PTX cast
                //   kernels for GPU-resident params). Without the opt-in the
                //   decision stays advisory with a not-lowered notice.
                // - When both sources are live, merge CONSERVATIVELY per
                //   param: F32 wins. CPDT's PrecisionPlan carries per-param
                //   tiers (calibrated critical params pinned to F32) that a
                //   layer-uniform WGGO decision must not override; and a
                //   CPDT FP16 tier the WGGO layer kept at 32 bits is
                //   likewise deferred to the more conservative choice.
                // - When WGGO made NO moment-bit decision for this block at
                //   all (it may still be active for unrelated reasons —
                //   structural pruning, CSHA fusion, packing), an
                //   independent CPDT PrecisionPlan is not gated on the
                //   opt-in flag: there is nothing for it to arbitrate
                //   against. See `arbitrate_moment_precision`.
                use crate::cpdt_precision_exec::{
                    arbitrate_moment_precision, MomentPrecisionArbitration as MPA, DTYPE_F32,
                };
                match arbitrate_moment_precision(
                    wggo_bits,
                    cpdt_lists,
                    self.compile_options.wggo.moment_precision,
                ) {
                    MPA::NotLoweredNoOptIn => {
                        eprintln!(
                            "[cpdt] optimizer-moment precision NOT lowered: WGGO's \
                             plan carries reduced-precision m/v decisions but \
                             --wggo-moment-precision was not passed (opt-in: \
                             changes training numerics). Moments stay FP32."
                        );
                        None
                    }
                    MPA::Merged(m, v) => {
                        let sub32 = m.iter().chain(v.iter()).filter(|&&c| c != DTYPE_F32).count();
                        eprintln!(
                            "[cpdt] WGGO optimizer-moment precision active \
                             (merged with the CPDT per-param plan, F32 wins): \
                             {sub32} moment buffer(s) in FP16 storage \
                             (device-resident; GPU runs use the CFTP-v7 PTX \
                             cast kernels for the dequant->step->quant \
                             envelope)."
                        );
                        Some((m, v))
                    }
                    MPA::WggoOnly(m, v) => {
                        let sub32 = m.iter().chain(v.iter()).filter(|&&c| c != DTYPE_F32).count();
                        eprintln!(
                            "[cpdt] WGGO optimizer-moment precision active: {sub32} \
                             moment buffer(s) in FP16 storage (8-bit clamps to FP16 \
                             in v1; device-resident — GPU runs use the CFTP-v7 PTX \
                             cast kernels for the dequant->step->quant envelope)."
                        );
                        Some((m, v))
                    }
                    MPA::CpdtOnly(m, v) => Some((m, v)),
                    MPA::Inactive => None,
                }
            };
            if let Some((m_codes, v_codes)) = dtype_data {
                let m_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
                let v_list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
                for &code in &m_codes {
                    let c = builder.ins().iconst(cl_types::I64, code as i64);
                    self.compile_call_by_name(builder, "nsl_list_push", &[m_list, c])?;
                }
                for &code in &v_codes {
                    let c = builder.ins().iconst(cl_types::I64, code as i64);
                    self.compile_call_by_name(builder, "nsl_list_push", &[v_list, c])?;
                }
                Some((m_list, v_list))
            } else {
                None
            }
        };

        // FASE Codegen Phase 2: build per-parameter mode table from WGGO's
        // per-layer decisions and emit it as a .rodata byte array. The
        // backward loop below loads `modes[gai]` to choose Deferred vs
        // FullBuffer per param. When None (no WGGO active), the loops use
        // today's monolithic `fase_deferred` branch (byte-identical to
        // pre-Phase-2 codegen).
        let mode_table_base: Option<cranelift_codegen::ir::Value> = {
            let modes = crate::fase_codegen_table::build_param_mode_table(
                &param_paths,
                &model_var_name,
                &fase_plan,
                self.wggo_overrides.as_ref(),
            );
            match modes {
                Some(bytes) => {
                    let suffix = self.fase_table_counter;
                    self.fase_table_counter += 1;
                    let func_suffix = format!("t{suffix}");
                    let data_id = self.emit_param_mode_table_rodata(&bytes, &func_suffix)?;
                    let global = self.module.declare_data_in_func(data_id, builder.func);
                    Some(
                        builder
                            .ins()
                            .symbol_value(cranelift_codegen::ir::types::I64, global),
                    )
                }
                None => None,
            }
        };
        if csla_active && mode_table_base.is_some() {
            return Err(CodegenError::new(
                "--layerwise-accum is incompatible with a WGGO per-parameter FASE \
                 mode table (mixed Deferred/FullBuffer accumulation): the \
                 window-buffered backward assumes the uniform Deferred hook. \
                 Drop --wggo overrides or --layerwise-accum",
            ));
        }

        // ── 4. Create optimizer state buffers ─────────────────────────
        // Number of state buffers per param depends on optimizer:
        //   SGD/Lion/Muon: 1 (velocity/momentum)
        //   Adam/AdamW/SOAP: 2 (first moment m, second moment v)
        //
        // State buffers are now NslLists (sized at runtime from param count)
        // instead of compile-time Vec<Value>, because the number of actual
        // tensor parameters is only known at runtime after recursive collection.
        let num_state_buffers = match optimizer_name.as_str() {
            "adam" | "adamw" | "soap" => 2,
            _ => 1,
        };

        // Optimizer-state offload (scaling campaign item 4): m/v allocate
        // HOST-resident (pinned when a GPU is live — P0.2); every optimizer
        // step stages them to the device, runs the unchanged F32 update, and
        // copies back on the transfer stream. P0.3: COMPOSES with
        // reduced-precision moments on this (non-pipelined) path — host m/v
        // are stored at the planned dtype and staged through the combined
        // cross-device cast envelope (nsl_tensor_cast_from_host /
        // nsl_tensor_cast_to_host_into), which replaces the co-resident
        // nsl_tensor_cast_into requant that used to force a hard refusal
        // here. The pipelined train path still refuses offload outright
        // (see compile_train_block_pipelined).
        if self.compile_options.optim_state_offload {
            if cpdt_precision_dtypes.is_some() {
                eprintln!(
                    "[offload] optimizer state (m/v) is HOST-resident at the \
                     planned reduced-precision dtypes (offload x \
                     --wggo-moment-precision/CPDT composition): each optimizer \
                     step dequants host state to device F32, updates, and \
                     quant-casts back (halved PCIe staging traffic vs f32 \
                     offload). VRAM saved: {num_state_buffers}x parameter bytes."
                );
            } else {
                eprintln!(
                    "[offload] optimizer state (m/v) is HOST-resident: each optimizer \
                     step stages state to the device and copies it back (2 PCIe \
                     round-trips of total state per step). VRAM saved: \
                     {num_state_buffers}x parameter bytes."
                );
            }
        }

        let state_list_1 = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
        let state_list_2 = if num_state_buffers >= 2 {
            self.compile_call_by_name(builder, "nsl_list_new", &[])?
        } else {
            builder.ins().iconst(cl_types::I64, 0)
        };

        // Loop: for i in 0..num_params, create zeros_like(param_list[i])
        {
            let init_counter_var = state.new_variable();
            builder.declare_var(init_counter_var, cl_types::I64);
            let init_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(init_counter_var, init_zero);

            let init_header = builder.create_block();
            let init_body = builder.create_block();
            let init_exit = builder.create_block();

            builder.ins().jump(init_header, &[]);
            builder.switch_to_block(init_header);
            // Do NOT seal init_header here — the back-edge from init_body hasn't been added yet
            state.current_block = Some(init_header);

            let idx = builder.use_var(init_counter_var);
            let cond = builder
                .ins()
                .icmp(IntCC::SignedLessThan, idx, num_params_val);
            builder.ins().brif(cond, init_body, &[], init_exit, &[]);

            builder.switch_to_block(init_body);
            builder.seal_block(init_body);
            state.current_block = Some(init_body);

            let param_i = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, idx])?;

            // D3 v2 (ZeRO-1): shard the optimizer-state ALLOCATION. v1
            // owner-gated only the UPDATE, so every rank still allocated FULL
            // m/v (no memory win). Now, when zero is enabled, a non-owner of
            // param `idx` (owner = idx % world_size) allocates NOTHING for its
            // moment buffers and pushes a null (0) placeholder — the lists stay
            // length==num_params and global-index-addressable, and the
            // owner-gated update branch is the ONLY reader of m/v, so nulls are
            // never dereferenced (the end-of-train free loop is null-safe:
            // nsl_tensor_free(0) is a no-op). The per-step collectives iterate
            // the grad/param lists, NEVER m/v, so allocation gating cannot
            // perturb the identical-collective-sequence spin-barrier invariant.
            // `zero_enabled` is recomputed locally: the outer binding is
            // introduced far below (near the optimizer loop), out of scope here.
            let zero_enabled = self.features.zero_stage.filter(|&s| s >= 1).is_some();
            let offload = self.compile_options.optim_state_offload;
            // `cpdt_precision_dtypes` is `Option<(Value, Value)>` (Value: Copy),
            // so projecting each moment's dtype-code list by value is fine.
            let m_list = cpdt_precision_dtypes.map(|(m, _)| m);

            // P0.1: first-moment buffers under the OptimM surface. The offload
            // variant allocates HOST tensors — inert GPU tag (host state is not
            // VRAM).
            let surface_optim_m = builder.ins().iconst(cl_types::I8, SURFACE_OPTIM_M);
            self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_optim_m])?;
            let buf1 = if zero_enabled {
                self.emit_owner_gated_moment(builder, state, param_i, idx, m_list, offload)?
            } else {
                self.emit_moment_zeros_like(builder, param_i, idx, m_list, offload)?
            };
            self.compile_call_by_name(builder, "nsl_list_push", &[state_list_1, buf1])?;
            if num_state_buffers >= 2 {
                // P0.1: second-moment buffers under the OptimV surface.
                let surface_optim_v = builder.ins().iconst(cl_types::I8, SURFACE_OPTIM_V);
                self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_optim_v])?;
                let v_list = cpdt_precision_dtypes.map(|(_, v)| v);
                let buf2 = if zero_enabled {
                    self.emit_owner_gated_moment(builder, state, param_i, idx, v_list, offload)?
                } else {
                    self.emit_moment_zeros_like(builder, param_i, idx, v_list, offload)?
                };
                self.compile_call_by_name(builder, "nsl_list_push", &[state_list_2, buf2])?;
            }

            let one_init = builder.ins().iconst(cl_types::I64, 1);
            let next_idx = builder.ins().iadd(idx, one_init);
            builder.def_var(init_counter_var, next_idx);
            builder.ins().jump(init_header, &[]);
            // Now seal init_header — both predecessors (entry jump + back-edge) are connected
            builder.seal_block(init_header);

            builder.switch_to_block(init_exit);
            builder.seal_block(init_exit);
            state.current_block = Some(init_exit);

            // End of the OptimM/OptimV bracket — restore the caller's surface.
            self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_prev])?;
        }

        // ── 5. Initialize lr and step_count variables ───────────────────
        let lr_var = state.new_variable();
        builder.declare_var(lr_var, cl_types::F64);
        let lr_const = builder.ins().f64const(lr_value);
        builder.def_var(lr_var, lr_const);

        let step_count_var = state.new_variable();
        builder.declare_var(step_count_var, cl_types::I64);
        let zero_i64 = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_count_var, zero_i64);

        // Dev Tools Phase 5 Task 7: publish step-counter variable so
        // `@inspect` emission inside the step body can gate on `step % N`.
        // Cleared at end of compile_train_block.
        self.inspect_train_step_var = Some(step_count_var);

        // ── 5a. Dev Tools Phase 4 Task 4: optional health flush-interval setter ──
        if self.compile_options.health_monitor {
            if let Some(n) = self.compile_options.health_flush_interval {
                let n_val = builder.ins().iconst(cl_types::I64, n as i64);
                self.compile_call_by_name(builder, "nsl_health_set_flush_interval", &[n_val])?;
            }
        }

        // ── 5b. Allocate gradient accumulation buffers (if grad_accumulation_steps > 1) ──
        // These persist across batches within each accumulation window. Each buffer
        // is zeros_like(param) and gets += each batch's grads, then zeroed after
        // the optimizer step every N batches.
        let accum_list = if grad_accumulation_steps > 1 {
            let list = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            // P0.1: grad-accumulation buffers under the MPartial surface.
            let surface_m_partial = builder.ins().iconst(cl_types::I8, SURFACE_M_PARTIAL);
            self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_m_partial])?;
            // Runtime loop over param_list (not layout.fields — which may include sub-models)
            let accum_i_var = state.new_variable();
            builder.declare_var(accum_i_var, cl_types::I64);
            let accum_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(accum_i_var, accum_zero);
            let accum_hdr = builder.create_block();
            let accum_body = builder.create_block();
            let accum_exit = builder.create_block();
            builder.ins().jump(accum_hdr, &[]);
            builder.switch_to_block(accum_hdr);
            // Don't seal accum_hdr yet — back-edge not added
            let ai = builder.use_var(accum_i_var);
            let ac = builder
                .ins()
                .icmp(IntCC::SignedLessThan, ai, num_params_val);
            builder.ins().brif(ac, accum_body, &[], accum_exit, &[]);
            builder.switch_to_block(accum_body);
            builder.seal_block(accum_body);
            let p = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, ai])?;
            // P3: under --optim-state-offload the FASE window accumulator
            // (m_partial) is the last device-resident param-sized f32 surface
            // (~4.15 GB at 1B). Allocate it HOST-resident (pinned) too; the
            // accumulate hook stages it to the grad's device per micro-batch
            // and the final step stages it in for the m/v update. m_partial
            // is ALWAYS f32 (exact-windowed semantics — the reduced-precision
            // moment path never touches it), so f32 host regardless of the
            // CPDT precision plan.
            //
            // CSLA (D1b): the whole point — do NOT allocate the full-model
            // window here. Slots start NULL; the window backward allocates
            // each layer's accumulators just before that layer's replay and
            // frees them right after its per-layer update, so the live
            // accumulator surface is max(one layer) + the epilogue globals
            // instead of 4·P bytes. (All accumulation happens inside the
            // window region under csla — the per-micro-batch ga_body loop is
            // hook-skipped — so a NULL slot is never read between windows.)
            let zeros = if csla_active {
                builder.ins().iconst(cl_types::I64, 0)
            } else if self.compile_options.optim_state_offload {
                self.compile_call_by_name(builder, "nsl_tensor_zeros_like_host_f32", &[p])?
            } else {
                self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[p])?
            };
            self.compile_call_by_name(builder, "nsl_list_push", &[list, zeros])?;
            let a_one = builder.ins().iconst(cl_types::I64, 1);
            let a_next = builder.ins().iadd(ai, a_one);
            builder.def_var(accum_i_var, a_next);
            builder.ins().jump(accum_hdr, &[]);
            builder.seal_block(accum_hdr);
            builder.switch_to_block(accum_exit);
            builder.seal_block(accum_exit);
            state.current_block = Some(accum_exit);
            // End of the MPartial bracket — restore the caller's surface.
            self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_prev])?;
            Some(list)
        } else {
            None
        };

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
            let saves_outer_var = state.new_variable();
            builder.declare_var(saves_outer_var, cl_types::I64);
            let so = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            builder.def_var(saves_outer_var, so);
            let dicts_var = state.new_variable();
            builder.declare_var(dicts_var, cl_types::I64);
            let dl = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            builder.def_var(dicts_var, dl);
            Some((saves_outer_var, dicts_var))
        } else {
            None
        };

        // ── 6. Emit epoch loop ──────────────────────────────────────────
        let epoch_counter_var = state.new_variable();
        builder.declare_var(epoch_counter_var, cl_types::I64);
        let zero = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(epoch_counter_var, zero);

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

        let has_dataloader = state.cleanup.dataloader_vars.last().copied();

        // Declare step parameter variable
        let step_param_var = state.new_variable();
        builder.declare_var(step_param_var, cl_types::I64);
        let init_null = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_param_var, init_null);
        state
            .variables
            .insert(step_param_sym, (step_param_var, cl_types::I64));

        let epoch_loss_var = state.new_variable();
        builder.declare_var(epoch_loss_var, cl_types::I64);
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
            self.emit_packing_registry_stash(builder, batch_val)?;
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

        /// CSLA Stage-2 (D1a): one buffered import slot of a micro-batch's
        /// window save list.
        enum CslaSlotKind {
            /// i64-typed lowered value (tensor/list pointer or integer).
            /// `owned` carries the forward lowering's ownership type: the
            /// window backward frees Tensor/List-owned slots after their
            /// micro-batch's replay (mirroring `free_wengert_owned_values`).
            Raw {
                owned: Option<crate::wengert::WengertType>,
            },
            /// f64 scalar, stored via a same-width bitcast; never freed.
            F64Bits,
        }
        /// One trainable tensor parameter's compile-time facts for the
        /// layer-major update schedule (D1b).
        struct CslaParam {
            name: String,
            /// Primal leaf VarId (for the view-of-θ hazard check).
            primal_vid: crate::wengert::VarId,
            /// Adjoint (gradient) VarId, when the param is read in the
            /// backward; `None` = dead/frozen — no gradient ever
            /// accumulates, but the update still fires (weight decay +
            /// moment decay mutate θ on zero gradients, exactly like the
            /// baseline's unconditional 0..num_params update loop).
            adj_vid: Option<crate::wengert::VarId>,
            /// param_paths index — the universal join key for
            /// param_list/state lists/accum_list.
            accum_idx: i64,
        }
        /// Compile-time context carried from the save phase (inside the
        /// source-AD arm) to the window-backward emission site just before
        /// the optimizer gate.
        struct CslaPending {
            adjoint: crate::wengert::WengertList,
            slots: Vec<(crate::wengert::VarId, CslaSlotKind)>,
            seed_base: crate::wengert_lower::VarMap,
            param_adj_set: std::collections::HashSet<crate::wengert::VarId>,
            /// adjoint grad VarId → compile-time accum_list index.
            hook_accum_idx: std::collections::HashMap<crate::wengert::VarId, i64>,
            accum_scale: f64,
            /// Slot index of the loss import, when the adjoint reads the loss
            /// tensor itself: that slot's per-b free must skip the window's
            /// LAST micro-batch (the current iteration's loss — still read by
            /// on_step/on_epoch after the backward phase; it is freed by the
            /// conditional at the per-iteration loss-free site instead).
            loss_slot: Option<usize>,
            /// D1b: the per-param update facts (the layerwise plan itself
            /// is consumed by the pre-forward schedule derivation and no
            /// longer travels here — `schedule` below is its product).
            params: Vec<CslaParam>,
            /// PRIMAL-side zero-copy views of trainable params (transpose /
            /// reshape chains rooted at a param leaf): view result vid →
            /// param leaf vid. These ride the window buffer as slots but
            /// ALIAS θ's storage — a read after θ's per-layer update would
            /// see half-updated weights (review D1b-2).
            primal_view_of:
                std::collections::HashMap<crate::wengert::VarId, crate::wengert::VarId>,
            /// LSE tape-carry (lifts the D1a fused-SDPA refusal): one extra
            /// inner-list slot per `flash_attn_aux` entry, holding the
            /// forward-saved logsumexp (a real tensor when the fused
            /// dispatch launched, runtime 0 when it declined — the FFI's
            /// existing decline semantics). `(inner slot index, the fwd-out
            /// vids whose seeded Values must key the aux re-insert)`. The
            /// replay re-binds `flash_attn_aux[seed[vid]] = lse[b]` before
            /// each consuming range's lowering, so the emitted SDPA backward
            /// reads micro-batch b's LSE instead of missing the Value-keyed
            /// side-band.
            lse_slots: Vec<(usize, Vec<crate::wengert::VarId>)>,
            /// The lse Values actually pushed as window slots (tape-carry
            /// review F2): the u32::MAX-sentinel bulk-free retention is
            /// per-Value — aux entries whose out mapped to NO buffered slot
            /// were never pushed and their LSEs must still free per
            /// iteration or every fused-fired one leaks for the whole run.
            lse_pushed: std::collections::HashSet<Value>,
            /// Fused-CE tape-carry: one extra inner-list slot per
            /// `fused_ce_fwd_lse` entry — `(slot index, the fwd-result
            /// vids whose SEEDED Values key the replay re-bind)`. The
            /// replay re-inserts `fused_ce_fwd_lse[seed[vid]] = lse[b]`
            /// before each consuming range's lowering; the emitted fused
            /// backward then consumes AND frees the buffered tensor, so
            /// these slots take no per-b free (teardown sweeps the
            /// trailing partial window only).
            fce_slots: Vec<(usize, Vec<crate::wengert::VarId>)>,
            /// D2b part 2: the layer-major schedule, computed ONCE before
            /// the forward lowering so the segment-streamed forward and the
            /// window backward agree on ranges, update grouping, and the
            /// streamed-param set BY CONSTRUCTION (two independent
            /// derivations could disagree on which params the forward must
            /// re-upload — a null-data crash at best).
            schedule: CslaSchedule,
        }
        /// D2b part 2: the shared layer-major schedule (see
        /// `CslaPending.schedule`).
        struct CslaSchedule {
            /// Positional partition of the FINAL adjoint into replay
            /// ranges (prologue, one per layer in backward order; the last
            /// range swallows the embedding-backward epilogue ops).
            ranges: Vec<crate::layerwise::ReplayRange>,
            /// Per-range accum/param_list indices updating right after
            /// that range's replay.
            layer_group: Vec<Vec<i64>>,
            /// Epilogue group: every param_paths slot not claimed by a
            /// range (globals / tied / cross-layer / dead params).
            global_group: Vec<i64>,
            /// Weight-streamed params (`--weight-stream`): layer-grouped
            /// minus view-rooted, sorted. Empty when streaming is off.
            ws_streamed: Vec<i64>,
            /// Per-range Σ elements of its STREAMED params (0 where shapes
            /// were symbolic). Item 11 calibration: pack bytes = elems × 4
            /// (GPU f32) drive the prefetch overlap gate's transfer-time
            /// estimate against the target GpuSpec.
            range_pack_elems: Vec<u64>,
        }
        let mut csla_pending: Option<CslaPending> = None;
        let mut csla_loss_buffered = false;
        // Teardown sweep plan for the trailing partial window: (slot index,
        // free fn) for every owned buffered slot. Outlives csla_pending
        // (which the window-backward emission consumes).
        let mut csla_teardown_slots: Option<Vec<(i64, &'static str)>> = None;

        let (grads_list, loss_val, source_ad_loss_owned, mut wengert_freed_vals) = if self.features.source_ad_enabled {
            // === Source AD path (compile-time backward) ===
            eprintln!("[nsl] Using source-to-source AD for backward pass");

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
                    self.compile_options.checkpoint_policies.clone()
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
            // WRGA B.3.2 Option 3: plumb synth overrides so the extractor
            // resolves sentinel-Ident callees/members emitted by the
            // adapter rewrite (fused FFI name + adapter field names).
            extractor.set_synth_call_names(self.synth_call_names.clone());
            extractor.set_synth_member_names(self.synth_member_names.clone());

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
            for &sym in state.variables.keys() {
                let rank = state
                    .variable_types
                    .get(&sym)
                    .and_then(resolvable_tensor_rank);
                extractor.register_input_with_rank(sym, rank);
            }

            let extraction_ok = extractor.extract_stmts(&step_body.stmts);

            if !extraction_ok {
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

                // Source AD extraction failed — fall back to tape
                eprintln!("[nsl] source AD extraction failed, falling back to tape-based AD");

                // Undo training mode — tape path sets it itself
                let false_val = builder.ins().iconst(cl_types::I8, 0);
                self.compile_call_by_name(builder, "nsl_set_training_mode", &[false_val])?;

                let (grads, loss) =
                    self.compile_tape_backward(builder, state, step_body, param_list)?;
                (grads, loss, false, std::collections::HashSet::new())
            } else {
                // 3. Build initial VarMap: map named input/param VarIds to
                //    Cranelift Values already present in state.variables.
                let mut primal_vars = crate::wengert_lower::VarMap::new();
                // Build primal_vars from state.variables using both Symbol-based
                // and name-based matching to handle cross-module Symbol mismatches.
                let state_vars_by_name: std::collections::HashMap<String, Value> = state
                    .variables
                    .iter()
                    .map(|(sym, (cvar, _))| {
                        (self.resolve_sym(*sym).to_string(), builder.use_var(*cvar))
                    })
                    .collect();

                // First pass: map symbol_var_map entries via Symbol match or name fallback
                for (sym, vid) in extractor.symbol_var_map() {
                    if primal_vars.contains_key(vid) {
                        continue;
                    }
                    if let Some(&(cvar, _)) = state.variables.get(sym) {
                        primal_vars.insert(*vid, builder.use_var(cvar));
                    } else {
                        let name = self.resolve_sym(*sym).to_string();
                        if let Some(&val) = state_vars_by_name.get(&name) {
                            primal_vars.insert(*vid, val);
                        }
                    }
                }
                // Second pass: map Input ops by name (catches inputs not in symbol_var_map)
                for op in &extractor.wengert_list().ops {
                    if let crate::wengert::PrimalOp::Input(name) = &op.op {
                        if let std::collections::hash_map::Entry::Vacant(entry) =
                            primal_vars.entry(op.result)
                        {
                            if let Some(&val) = state_vars_by_name.get(name) {
                                entry.insert(val);
                            }
                        }
                    }
                }

                // Also populate model parameter VarIds from param_list.
                // MemberAccess expressions (e.g., m.w) are registered in the extractor
                // under the member symbol, but those aren't in state.variables — they're
                // loaded from the model struct. Map them to nsl_list_get(param_list, i).
                //
                // Also add the step parameter (batch) which is stored separately
                if let Some(&step_vid) = extractor.symbol_var_map().get(&step_param_sym) {
                    primal_vars
                        .entry(step_vid)
                        .or_insert_with(|| builder.use_var(step_param_var));
                }
                // Resolve model parameter VarIds by traversing nested struct layouts.
                // Compound names like "m.blocks.0.attn.wq" are split into path
                // components and walked through struct layouts + array indices,
                // emitting a chain of Cranelift loads at each level.
                for (compound_name, vid) in extractor.named_param_var_ids() {
                    if primal_vars.contains_key(vid) {
                        continue;
                    }

                    if let Some(val) = self.load_nested_field(
                        builder,
                        model_ptr,
                        &layout,
                        &model_type_name,
                        compound_name,
                    ) {
                        primal_vars.insert(*vid, val);
                    } else {
                        // Param not resolvable through struct layouts — this is expected
                        // for scalar config fields (eps, _d_model, etc.) that are used
                        // in non-differentiable contexts (int(), item()). The wengert
                        // lowerer will use the null placeholder, which is acceptable
                        // for Passthrough ops that don't need the actual tensor value.
                    }
                }

                // CPKD: resolve frozen teacher-field Input leaves.  Their
                // compound names are rooted at the teacher instance variable
                // (e.g. "teacher.blocks.0.attn.wq"), so the generic
                // any-root resolver applies.  Unresolvable teacher fields
                // fail loudly downstream: wengert_lower hard-errors on any
                // unresolved Input leaf (unlike Params, which degrade to a
                // null placeholder).
                for (compound_name, vid) in extractor.frozen_input_var_ids() {
                    if primal_vars.contains_key(vid) {
                        continue;
                    }
                    if let Some(val) =
                        self.load_source_ad_named_param(builder, state, compound_name)
                    {
                        primal_vars.insert(*vid, val);
                    }
                }

                // CPKD: collect the Distillation Build Report facts while the
                // extractor is in scope. Rendered by `compile_distill_block`.
                if let Some(distill) = self.active_distill_context.clone() {
                    let fused_op = extractor.wengert_list().ops.iter().find_map(|op| {
                        if let crate::wengert::PrimalOp::FusedKlCe {
                            vocab_size,
                            student_hidden,
                            teacher_hidden,
                            batch_size,
                            seq_len,
                            ..
                        } = &op.op
                        {
                            Some((
                                *vocab_size,
                                *student_hidden,
                                *teacher_hidden,
                                batch_size * seq_len,
                            ))
                        } else {
                            None
                        }
                    });
                    let logit_bytes_eliminated = fused_op
                        .map(|(v, _, _, rows)| 2 * (v as u64) * (rows as u64) * 4)
                        .unwrap_or(0);
                    self.last_cpkd_plan = Some(crate::cpkd::CpkdPlan {
                        teacher_name: self.resolve_sym(distill.teacher_sym).to_string(),
                        student_name: self.resolve_sym(distill.student_sym).to_string(),
                        epochs: distill.epochs,
                        loss: distill.loss.clone(),
                        trainable_params: extractor.named_param_var_ids().len(),
                        frozen_teacher_inputs: extractor.frozen_input_var_ids().len(),
                        fused_kl_ce_fired: fused_op.is_some(),
                        fused_shape: fused_op,
                        logit_bytes_eliminated,
                    });
                }

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
                extractor.apply_pending_fused_lce_prunes();

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
                if let Some(ref mode_str) = self.compile_options.wggo.mode {
                    if mode_str != "off" && mode_str != "disable" && mode_str != "disabled" {
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
                        // calibration_sidecar is populated by compile_and_calibrate's wrapper-
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
                        // pre-plan on any FASE-relevant decision, warn — the
                        // per-param mode table was already emitted from the
                        // pre-plan's overrides earlier in this function.
                        let preplan = self
                            .wggo_preplans
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
                                eprintln!(
                                    "[wggo] consumed pre-solved plan \
                                     (graph fingerprint match)"
                                );
                                Some(pre.plan.clone())
                            } else {
                                eprintln!(
                                    "[wggo] wggo-preplan-rejected \
                                     reason=graph_fingerprint_mismatch — replanning in place"
                                );
                                None
                            }
                        });
                        let preplan_was_rejected =
                            preplan.is_some() && reused_plan.is_none();
                        let plan = match reused_plan {
                            Some(plan) => Some(plan),
                            None => crate::wggo::run_on_wengert_with_weights(
                                extractor.wengert_list(),
                                &self.compile_options.target,
                                mode_str,
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
                            ),
                        };
                        if preplan_was_rejected {
                            if let (Some(pre), Some(fresh)) = (preplan, plan.as_ref()) {
                                let fresh_overrides =
                                    crate::wggo_overrides::WggoOverrides::from_applied(
                                        &fresh.applied,
                                    );
                                let fase_diverged = pre.overrides.per_layer.len()
                                    != fresh_overrides.per_layer.len()
                                    || pre
                                        .overrides
                                        .per_layer
                                        .iter()
                                        .zip(fresh_overrides.per_layer.iter())
                                        .any(|(a, b)| {
                                            a.layer_name != b.layer_name
                                                || a.fase_fused != b.fase_fused
                                        });
                                if fase_diverged {
                                    eprintln!(
                                        "[wggo] WARNING: the FASE per-param mode table was \
                                         built from the rejected pre-plan and the in-place \
                                         plan disagrees on fase_fused — per-param FASE \
                                         modes for this train block may not match the \
                                         final plan (rerun with --wggo off to rule the \
                                         table out when debugging)"
                                    );
                                }
                            }
                        }
                        if let Some(plan) = plan {
                            if self.compile_options.wggo.report {
                                eprintln!("{}", plan.render_report());
                            } else {
                                eprintln!("[wggo] {}", plan.summary());
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
                                eprintln!(
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
                                        PackingVerdict::Consumed { kernel } => eprintln!(
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
                                            eprintln!(
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
                            self.wggo_overrides = Some(
                                crate::wggo_overrides::WggoOverrides::from_applied(&plan.applied),
                            );
                            wggo_applied = Some(plan.applied);
                        }
                    }
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
                // (via self.wggo_overrides) so that per-layer fusion-level
                // decisions from WGGO are honoured (or rejected with a
                // diagnostic) by CSHA.
                if let Some(ref mode_str) = self.compile_options.csha.mode {
                    // Sprint 2 (paper §6.2 binding fix): consult the
                    // per-model `@csha(...)` config captured by the semantic
                    // checker, keyed by the model type the current train
                    // block is compiling against.  We resolve effective
                    // mode_str / target / disable HERE so the rest of the
                    // hook stays uniform.
                    let per_model_cfg = self
                        .compile_options
                        .csha_configs
                        .get(&model_type_name)
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
                        .unwrap_or(self.compile_options.target.as_str());

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
                        let csha_shape_override = self
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
                            extractor.wengert_list(),
                            effective_target,
                            effective_mode_str,
                            None, // weight-aware analysis hooked up via CompileOptions.weight_file in follow-up
                            csha_shape_override, // H.1: forward decorator head_dim to the planner
                            8,    // default head count; weight-informed path refines this
                            self.wggo_overrides.as_ref(),
                        ) {
                            if self.compile_options.csha.report {
                                eprintln!("{}", plan.render_report());
                            } else {
                                eprintln!("[csha] {}", plan.summary());
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
                                eprintln!(
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
                            self.last_csha_bridge = Some(bridge_out);
                            for d in diags { eprintln!("warning: {d}"); }
                            // A.2.1d: record the Wengert op indices CSHA
                            // has claimed across all boundary chains so
                            // downstream passes (A.2.2 RMSNorm prologue,
                            // A.2.3 matmul projection, A.2.4 RoPE
                            // epilogue) can ask `is_csha_claimed(op)`
                            // before emitting a redundant launch.
                            self.csha_claimed_ops =
                                crate::csha_apply::collect_claimed_ops(&plan);
                            // T7.1 / Gap D.1: build the chain-level dispatch
                            // map for the AD reverse walk. Gap D.1 passes the
                            // Wengert list so the dispatcher can resolve
                            // per-chain VarIds (Q/K/V outputs, weights,
                            // RMSNorm-out) and detect the shared SDPA op —
                            // which is the correct primary claim site for
                            // `EmitFused`.
                            if let Some(ref bridge) = self.last_csha_bridge {
                                // Gap I.1: pass the TRAINING config (clamped, no
                                // fusion flags) so the dispatcher's backward SMEM
                                // validator sees the same geometry the real
                                // launch will fire. Falls back to plan-level
                                // config when `csha_training_config` is absent
                                // (inference-only builds).
                                let training_config = self
                                    .kernels
                                    .flash_attention_context
                                    .as_ref()
                                    .and_then(|c| c.csha_training_config.as_ref());
                                let (op_to_chain, chain_marks) =
                                    crate::csha_apply::collect_chain_dispatch_map_with_wengert(
                                        &plan,
                                        bridge,
                                        Some(extractor.wengert_list()),
                                        training_config,
                                    );
                                if !chain_marks.is_empty() {
                                    self.csha_backward_claims =
                                        Some(crate::source_ad::CshaBackwardClaims {
                                            op_to_chain,
                                            chain_marks,
                                        });
                                }
                            }
                        }
                    }
                }

                // 5. Lower PRIMAL Wengert list to Cranelift IR.
                //    This IS the forward pass — each WengertOp is compiled to
                //    its runtime FFI call, and ALL intermediate VarId → Value
                //    mappings are recorded in full_vars.
                // ELTLS: free tape-held tensors before clearing the tape flag.
                self.free_tape_held_tensors(builder, state);
                state.flags.in_tape_region = false;
                // Debug: dump primal Wengert ops
                if std::env::var("NSL_DEBUG_WENGERT").is_ok() {
                    eprintln!(
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
                        eprintln!(
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
                            eprintln!("{text}");
                        }
                        return Err(crate::error::CodegenError::new(
                            "wggo_prune: one or more prune decisions refused; see [prune] stderr lines",
                        ));
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
                        eprintln!("{line}");
                    }
                }
                // --- END NEW ---

                // Task 4: invoke WRGA driver (pruning / rank allocation /
                // fusion) before primal lowering.  When WRGA is disabled or
                // inputs are empty, this is a no-op and we use the raw
                // extractor list.
                let wrga_plan = crate::stmt::invoke_wrga_if_enabled(self, extractor.wengert_list());
                // CPDT pipeline planner — runs immediately after WRGA so the
                // AppliedPlan produced by WGGO (if any) flows through as the
                // ModelSize source + wggo_recommended_shard. No-op when
                // `cpdt_mode == Off` or no cluster topology was configured.
                // NOTE: CPDT runs only when WGGO produced a plan. If `--cpdt` is enabled
                // without `--wggo`, `cpdt_plan` remains `None` and no diagnostics fire.
                // The CLI post-compile layer should warn when `cpdt_mode != Off` but
                // `cpdt_plan.is_none()` to surface this silent-skip case.
                if let Some(ref applied) = wggo_applied {
                    crate::stmt::invoke_cpdt_if_enabled(self, applied, Some(train));
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
                        eprintln!(
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
                        self.adapter_prescan_plan.clone()
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
                                cranelift_codegen::ir::MemFlags::trusted(),
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
                                cranelift_codegen::ir::MemFlags::trusted(),
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
                    if let Some(slot_off) = current_layout.adapter_sidetable_offset {
                        if let Some(index) = self.adapter_field_index(&current_type_name, last) {
                            let table_ptr = builder.ins().load(
                                cl_types::I64,
                                cranelift_codegen::ir::MemFlags::trusted(),
                                current_ptr,
                                slot_off as i32,
                            );
                            let byte_off = (index * 8) as i32;
                            let tensor_ptr = builder.ins().load(
                                cl_types::I64,
                                cranelift_codegen::ir::MemFlags::trusted(),
                                table_ptr,
                                byte_off,
                            );
                            primal_vars.insert(*vid, tensor_ptr);
                        }
                    }
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
                        .checkpoint_policies
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
                let ccr_plan = if self.compile_options.checkpoint_blocks
                    || ccr_selective_decorated
                {
                    let claimed_ids: Option<std::collections::HashSet<u32>> =
                        self.csha_backward_claims.as_ref().map(|claims| {
                            claims.op_to_chain.keys().copied().collect()
                        });
                    let policy = if self.compile_options.checkpoint_selective
                        || ccr_selective_decorated
                    {
                        crate::ccr::CcrPolicy::Selective
                    } else {
                        crate::ccr::CcrPolicy::Block
                    };
                    let compress_requested = self.compile_options.checkpoint_compress.is_some();
                    // Item 8: resolve the periodic-checkpoint stride. `Fixed(k)`
                    // passes straight through (ccr::plan logs the coalescing);
                    // `Auto` searches strides against the projected activation
                    // peak — with the CSLA accumulation window G applied to the
                    // saved boundaries — and the checkpoint byte budget.
                    let resolved_stride = match self.compile_options.checkpoint_stride {
                        crate::CheckpointStride::Fixed(k) => k,
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
                                eprintln!(
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
                                .checkpoint_budget_mib
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
                                    eprintln!(
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
                                    eprintln!(
                                        "[ccr] --checkpoint-stride auto: no candidate produced a \
                                         plan; using stride 1"
                                    );
                                    1
                                }
                            }
                        }
                    };
                    let plan = crate::ccr::plan(
                        &effective_primal,
                        claimed_ids.as_ref(),
                        policy,
                        compress_requested,
                        resolved_stride,
                    );
                    // Item 8: one coalescing note for the FINAL stride (the Auto
                    // search above ran plan() per candidate silently).
                    if resolved_stride > 1 {
                        if let Some(p) = &plan {
                            eprintln!(
                                "[ccr] periodic checkpointing: stride {resolved_stride} → \
                                 {} CCR super-segment(s) (saving every {resolved_stride}th block \
                                 boundary, recomputing each span — bit-exact)",
                                p.segments.len()
                            );
                        }
                    }
                    if let (Some(p), Some(dtype)) =
                        (&plan, self.compile_options.checkpoint_compress.as_deref())
                    {
                        if !p.compress.is_empty() {
                            ccr_compress_map = crate::ccr::append_compressed_saves(
                                &mut effective_primal,
                                p,
                                dtype,
                                &mut ccr_fresh,
                            );
                            eprintln!(
                                "[ccr] compressed saves: {} matmul-class tensors -> {dtype}",
                                ccr_compress_map.len()
                            );
                        } else if compress_requested {
                            eprintln!(
                                "[ccr] --checkpoint-compress requested but no \
                                 compressible saves exist (policy must be selective \
                                 with matmul-class interiors); continuing without"
                            );
                        }
                    }
                    plan
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
                        eprintln!(
                            "[ccr] nothing recomputable after the owned-tensor \
                             restriction; running without checkpointing"
                        );
                        ccr_plan = None;
                    } else if let Some(budget_mib) = self.compile_options.checkpoint_budget_mib {
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
                                eprintln!(
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
                        eprintln!(
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
                let mut gen = crate::source_ad::AdjointGenerator::new(start_var);
                // T7.1: thread CSHA backward claims into the generator so
                // the reverse walk can route claimed ops through the fused
                // backward dispatcher instead of per-op AD rules.
                if let Some(claims) = self.csha_backward_claims.take() {
                    gen.set_csha_claims(claims);
                }
                // Item 9: opt-in fused RMSNorm input-gradient lowering.
                gen.set_fuse_rmsnorm_backward(self.compile_options.fuse_rmsnorm_backward);
                let mut adjoint = gen.generate(&effective_primal);
                // Item 9 profiling (`NSL_PROFILE_ADJOINT=1`): a launch-count
                // histogram of the generated backward ops. Norm/activation
                // adjoints decompose into many small bandwidth-bound ops
                // (RMSNorm dgamma alone = mean/Sqrt/Div/Mul/reduce); this shows
                // which op classes dominate the launch count — the fusion
                // targets. Pre-CCR: recompute clones are forward ops, so this is
                // the true backward-op composition.
                if std::env::var("NSL_PROFILE_ADJOINT").is_ok() {
                    use std::collections::BTreeMap;
                    let mut hist: BTreeMap<String, usize> = BTreeMap::new();
                    for op in &adjoint.ops {
                        let key = match &op.op {
                            crate::wengert::PrimalOp::Passthrough(n) => {
                                format!("Passthrough({n})")
                            }
                            other => format!("{other:?}")
                                .split(['(', ' ', '{'])
                                .next()
                                .unwrap_or("?")
                                .to_string(),
                        };
                        *hist.entry(key).or_default() += 1;
                    }
                    eprintln!(
                        "[adjoint-profile] {} generated backward ops:",
                        adjoint.ops.len()
                    );
                    let mut rows: Vec<_> = hist.into_iter().collect();
                    rows.sort_by_key(|(_, c)| std::cmp::Reverse(*c));
                    for (k, c) in rows {
                        eprintln!("[adjoint-profile]   {c:>5}  {k}");
                    }
                }
                // D2b part 2: hand the claims BACK for the forward lowering
                // below (the fused-SDPA claim dispatch reads them); the
                // compiler slot is cleared again right after the forward, so
                // the ADJOINT lowering still never sees claims — the same
                // invariant the old post-forward `take()` enforced.
                self.csha_backward_claims = gen.take_csha_claims();
                // T7.1: surface any CSHA fallback diagnostics.
                for diag in gen.csha_diagnostics() {
                    eprintln!("[nsl] {diag}");
                }

                // 6a. Task 4: WRGA backward-live filter — drop adjoint ops
                // that the WRGA prune pass proved to be on frozen branches.
                if let Some(plan) = &wrga_plan {
                    adjoint.ops = crate::source_ad::eliminate_by_backward_live(
                        &adjoint.ops,
                        &plan.prune.backward_live,
                        gen.adjoint_vars_map(),
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
                        .filter_map(|(_, vid)| gen.adjoint_of(*vid))
                        .collect()
                };
                if !adjoint_needed.is_empty() {
                    adjoint.ops =
                        crate::source_ad::eliminate_dead_gradients(&adjoint.ops, &adjoint_needed);
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
                    eprintln!("[csla]\n{}", plan.render_report("  "));
                }

                // 6c. CCR P1.a: splice recompute clones + FreeTensor markers
                // into the (final, post-eliminate) adjoint and remap its
                // references from the early-freed originals to the clones.
                // Runs AFTER the eliminate passes so the splice positions
                // and last-use frees are computed against exactly the op
                // list that will be lowered.
                if let Some(plan) = &ccr_plan {
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

                // 6d. CCR: adjoint-region last-use freeing. The 500M/seq1024
                // per-surface OOM decomposition showed adjoint intermediates
                // (dx-chain temporaries) are the binding activation wall —
                // they all lived to the end-of-backward bulk free. Insert a
                // FreeTensor after each adjoint var's last use. Protected:
                // every param-gradient adjoint (consumed by the FASE hook or
                // by post-lowering grad collection) plus the inputs of the
                // ops producing them (the hook's reduce_to_shape identity
                // path emits its own extra free for those raw grads).
                if ccr_plan.is_some() {
                    let mut ccr_protect: std::collections::HashSet<crate::wengert::VarId> =
                        std::collections::HashSet::new();
                    for (param_name, primal_vid) in extractor.named_param_var_ids() {
                        if !self.is_trainable_param_name(param_name) {
                            continue;
                        }
                        if let Some(adj_vid) = gen.adjoint_of(*primal_vid) {
                            ccr_protect.insert(adj_vid);
                        }
                    }
                    for op in &adjoint.ops {
                        if ccr_protect.contains(&op.result) {
                            for input in &op.inputs {
                                ccr_protect.insert(*input);
                            }
                        }
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
                    let n = crate::ccr::insert_adjoint_last_use_frees(
                        &mut adjoint,
                        &ccr_protect,
                        &mut ccr_fresh,
                    );
                    if std::env::var("NSL_CCR_DEBUG").is_ok() {
                        eprintln!("[ccr] adjoint last-use frees inserted: {n}");
                    }
                }

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
                if self.compile_options.memory_report
                    || std::env::var("NSL_ARENA_REPORT").ok().as_deref() == Some("1")
                {
                    // Transients are runtime-shaped here, so element counts are
                    // unquantified (|_| None) and the headline is peak
                    // concurrency (the arena slot count). Stage-2 wires shapes.
                    let arena = crate::transient_arena::analyze(
                        &effective_primal,
                        &adjoint,
                        &|_| None,
                        4, // GPU f32 training dtype width
                    );
                    eprintln!("[arena]\n{}", arena.render_report("  "));
                }

                // ── D2b part 2: CSLA schedule precompute (pre-forward) ──
                // The layerwise plan, per-param facts, replay ranges, and
                // update grouping — computed HERE (on the final adjoint) so
                // the segment-streamed forward below and the window backward
                // consume the same schedule. `CslaPre` flows into the save
                // phase; `WsForwardPlan` drives the sliced forward emission.
                struct CslaPre {
                    params: Vec<CslaParam>,
                    primal_view_of: std::collections::HashMap<
                        crate::wengert::VarId,
                        crate::wengert::VarId,
                    >,
                    imports: Vec<crate::wengert::VarId>,
                    schedule: CslaSchedule,
                }
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
                        // Item 11 calibration (review M3 follow-through): wire
                        // REAL element counts into the layerwise plan — this
                        // call site passed `|_| None` since D1, leaving every
                        // ParamInfo::elems empty and the prefetch gate's pack
                        // pricing blind. Static param shapes resolve here;
                        // symbolic ones stay None and decline their edges.
                        let elem_hints = crate::profiling::captures::elem_hints_from_var_nodes(
                            extractor.var_nodes(),
                            self.type_map,
                        );
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
                                    adj_vid: gen.adjoint_of(*primal_vid),
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
                        eprintln!(
                            "[csla] layer-major schedule: {} ranges, {} layer-grouped params, \
                             {} epilogue params",
                            n_ranges,
                            grouped.len(),
                            global_group.len(),
                        );

                        // ── Weight-stream admission (moved from the window
                        // site) + the part-2 forward streaming plan ──
                        let ws_active = self.compile_options.weight_stream;
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
                                eprintln!(
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
                            eprintln!(
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
                            if self.compile_options.stream_arena {
                                eprintln!(
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
                    for &idx in &wsplan.register_idxs {
                        let iv = builder.ins().iconst(cl_types::I64, idx);
                        let pw = self
                            .compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
                        self.compile_call_by_name(
                            builder,
                            "nsl_weight_stream_register",
                            &[pw],
                        )?;
                    }
                }
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
                    let arena_mode = self.compile_options.stream_arena;
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
                            let wsurf = builder.ins().iconst(cl_types::I8, 1); // SURFACE_WEIGHTS
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
                self.csha_backward_claims = None;
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
                    if let Some(plan) = &ccr_plan {
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
                struct ParamHookEntry {
                    primal_val: Value,
                    // i64 index into accum_list (== param_list index for this param)
                    accum_idx: i64,
                }
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
                        let Some(adj_vid) = gen.adjoint_of(*primal_vid) else {
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
                    //
                    // Instead of lowering the adjoint here (the interleaved
                    // schedule), push every adjoint-read primal value into
                    // this micro-batch's slot list and defer the whole
                    // window's backward to the should_step region, where a
                    // runtime loop replays the adjoint once per buffered
                    // micro-batch (one tape, N executions — unrolling would
                    // break CCR's anchor segmentation). Param/Constant leaves
                    // are loop-invariant (FASE Deferred updates θ only at
                    // window boundaries, after the replay), so the replay
                    // reuses the current iteration's SSA values for them via
                    // seed_base.
                    debug_assert!(fase_hook_active);

                    // Review findings H1/H2/M1 (D1a adversarial review): three
                    // compile-time SIDE CHANNELS travel by SSA Value instead
                    // of the tape, so the window replay cannot see them —
                    // refuse each loudly rather than replay wrong.
                    //
                    // H1 — CSHA-claimed fused backward: `csha_forward_saves`
                    // is keyed by layer name and holds the STEP iteration's
                    // save-buffer SSA values; the claimed backward consumes
                    // AND frees them once — inside the b-loop that means
                    // stale saves for b<N-1 plus an N-fold double-free.
                    if !self.csha_forward_saves.is_empty()
                        || adjoint.ops.iter().any(|op| {
                            matches!(
                                op.op,
                                crate::wengert::PrimalOp::FusedCshaBackward { .. }
                                    | crate::wengert::PrimalOp::CshaFusedBackwardExtract { .. }
                            )
                        })
                    {
                        return Err(CodegenError::new(
                            "--layerwise-accum is incompatible with CSHA-claimed \
                             fused attention backward: the claimed saves travel \
                             through a compile-time side channel \
                             (csha_forward_saves) that the window replay cannot \
                             re-bind per micro-batch. Drop @csha/@flash_attention \
                             claims or --layerwise-accum",
                        ));
                    }
                    // H2 (RESOLVED by the LSE tape-carry): the fused SDPA
                    // dispatch saves its logsumexp through the Value-keyed
                    // `flash_attn_aux` side-band + a u32::MAX-sentinel owned
                    // entry. The save phase below buffers every aux entry as
                    // an extra window slot, and the replay re-binds the aux
                    // map per micro-batch before each consuming range's
                    // lowering — the emitted backward then consumes the SAME
                    // per-batch LSE the baseline did (or the runtime-0
                    // decline sentinel), keeping the fused phase-2 kernel on
                    // the fused path.
                    // M1, NARROWED by the fused-CE tape-carry: the f32
                    // @fused_lm_ce path is now supported — the only real
                    // side-band is `fused_ce_fwd_lse` (one [B*S] f32 tensor
                    // per micro-batch), buffered as an extra window slot
                    // below and re-bound per replay range exactly like the
                    // flash-attention LSE; the f32 cast-cache MISS path is
                    // already correct at replay (pass-through of the seeded
                    // inputs, zero extra emission). Still refused:
                    //
                    // - distill blocks: the KL-CE carry would need THREE
                    //   lse slots per micro-batch plus the frozen teacher's
                    //   activation buffered per micro-batch — a large new
                    //   surface with no gate; refuse until designed.
                    // - @fused_lm_ce(dtype="f16"/"bf16"): the fp16/bf16
                    //   shadow-cast tensors are step-scoped (freed at
                    //   function-scope exit); a replay-side re-cast would
                    //   pile N cast sets per window with no free (the
                    //   original M1 finding), and buffering w_cast per
                    //   micro-batch (V×H×2 B) would erase the memory win.
                    if self.active_distill_context.is_some() {
                        return Err(CodegenError::new(
                            "--layerwise-accum is incompatible with distill \
                             blocks: the fused KL-CE backward reads Value-keyed \
                             forward saves (three LSE buffers + the teacher \
                             activations) the window replay cannot see. Drop \
                             the distill block or --layerwise-accum",
                        ));
                    }
                    if let Some(cfg) = &self.active_fused_ce_config {
                        let non_f32 = cfg.enabled
                            && !matches!(
                                cfg.dtype,
                                None | Some(crate::FusedCeDtypeHint::F32)
                            );
                        if non_f32 {
                            return Err(CodegenError::new(
                                "--layerwise-accum supports @fused_lm_ce only \
                                 with dtype=\"f32\": the fp16/bf16 forward cast \
                                 tensors are step-scoped and cannot be carried \
                                 through the window replay without either \
                                 leaking N cast sets per window or buffering \
                                 the V*H weight cast per micro-batch. Use \
                                 dtype=\"f32\" or drop --layerwise-accum",
                            ));
                        }
                    }

                    let (saves_outer_var, dicts_var) =
                        csla_buffers.expect("csla_buffers allocated when csla_active");
                    // D2b part 2: the plan / params / view chains / imports /
                    // layer-major schedule were all computed in the
                    // pre-forward pure pipeline (the forward streamer needed
                    // them at emission time) — consume, don't recompute.
                    let pre = csla_pre.expect("csla_pre computed when csla_active");
                    let imports = pre.imports;
                    let owned_map: std::collections::HashMap<
                        crate::wengert::VarId,
                        crate::wengert::WengertType,
                    > = full_lowered
                        .owned_values
                        .iter()
                        .map(|(vid, _, ty)| (*vid, *ty))
                        .collect();
                    let inner = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
                    let mut slots: Vec<(crate::wengert::VarId, CslaSlotKind)> = Vec::new();
                    let mut loss_slot: Option<usize> = None;
                    for v in imports {
                        // Ghost VarIds (never lowered) are skipped here AND at
                        // replay seed time — the adjoint lowering ghost-skips
                        // their consumers identically to the baseline.
                        let Some(&val) = full_vars.get(&v) else { continue };
                        let vty = builder.func.dfg.value_type(val);
                        if vty == cl_types::I64 {
                            self.compile_call_by_name(builder, "nsl_list_push", &[inner, val])?;
                            if v == loss_var_id {
                                loss_slot = Some(slots.len());
                            }
                            slots.push((
                                v,
                                CslaSlotKind::Raw {
                                    owned: owned_map.get(&v).copied(),
                                },
                            ));
                        } else if vty == cl_types::F64 {
                            let bits = builder.ins().bitcast(
                                cl_types::I64,
                                MemFlags::new(),
                                val,
                            );
                            self.compile_call_by_name(builder, "nsl_list_push", &[inner, bits])?;
                            slots.push((v, CslaSlotKind::F64Bits));
                        } else {
                            return Err(CodegenError::new(format!(
                                "--layerwise-accum: adjoint-imported primal VarId {v} \
                                 lowered to unsupported Cranelift type {vty} — the \
                                 window buffer stores i64 pointers/integers and \
                                 bitcast f64 scalars only",
                            )));
                        }
                    }
                    // LSE tape-carry: append one slot per fused-SDPA aux
                    // entry whose forward output the adjoint reads. The
                    // stored value is the aux LSE (join-block param on the
                    // dispatch arm — a real tensor when the fused launch
                    // fired, runtime 0 when it declined; the decomposed
                    // arm's compile-time iconst 0 buffers as a plain 0).
                    // Sorted by min fwd-out vid so slot order is
                    // deterministic.
                    let mut lse_slots: Vec<(usize, Vec<crate::wengert::VarId>)> = Vec::new();
                    let mut csla_lse_pushed: std::collections::HashSet<Value> =
                        std::collections::HashSet::new();
                    if !self.flash_attn_aux.is_empty() {
                        let slot_vid_set: std::collections::HashSet<crate::wengert::VarId> =
                            slots.iter().map(|(v, _)| *v).collect();
                        let mut by_val: std::collections::HashMap<
                            Value,
                            Vec<crate::wengert::VarId>,
                        > = std::collections::HashMap::new();
                        for (vid, val) in full_vars.iter() {
                            by_val.entry(*val).or_default().push(*vid);
                        }
                        let mut entries: Vec<(Vec<crate::wengert::VarId>, Value)> = self
                            .flash_attn_aux
                            .iter()
                            .filter_map(|(out_val, (_, lse_val))| {
                                let mut vids: Vec<crate::wengert::VarId> = by_val
                                    .get(out_val)?
                                    .iter()
                                    .copied()
                                    .filter(|v| slot_vid_set.contains(v))
                                    .collect();
                                if vids.is_empty() {
                                    return None;
                                }
                                vids.sort_unstable();
                                Some((vids, *lse_val))
                            })
                            .collect();
                        entries.sort_by_key(|(vids, _)| vids[0]);
                        for (vids, lse_val) in entries {
                            let idx = slots.len() + lse_slots.len();
                            self.compile_call_by_name(
                                builder,
                                "nsl_list_push",
                                &[inner, lse_val],
                            )?;
                            csla_lse_pushed.insert(lse_val);
                            lse_slots.push((idx, vids));
                        }
                    }
                    // Anti-vacuity marker (tape-carry review F1): under the
                    // Block checkpoint policy every in-block SDPA out is a
                    // recompute victim (the clone RE-LAUNCHES the fused
                    // forward during replay and re-establishes the aux
                    // side-band locally), so lse_slots is 0 and the carry is
                    // inert; the carry engages under --checkpoint-selective
                    // (SDPA outs saved). Gates assert this line's exact slot
                    // count so the tested path is named, not assumed.
                    eprintln!("[csla] lse tape-carry: {} slots", lse_slots.len());

                    // Fused-CE tape-carry: one extra slot per
                    // `fused_ce_fwd_lse` entry — the [B*S] f32 logsumexp the
                    // fused backward consumes, keyed by the fwd-result
                    // (loss-scalar) Value. Unlike the flash-attention carry
                    // this one is LIVE under BOTH checkpoint policies:
                    // FusedLinearCe sits in the CCR epilogue (never a
                    // recompute victim), so a replay clone can never
                    // re-establish the side-band locally. Lifecycle also
                    // differs: the emitted backward CONSUMES AND FREES the
                    // buffered tensor per (range, b) — the per-b slot-free
                    // machinery must NOT touch these slots; only the
                    // trailing-partial-window teardown sweep frees
                    // unreplayed entries. (The f32 cast cache needs no
                    // carry: its replay MISS path re-emits pass-through
                    // inputs — the seeded x/W/bias — with zero extra cost;
                    // non-f32 dtypes are refused above.)
                    let mut fce_slots: Vec<(usize, Vec<crate::wengert::VarId>)> = Vec::new();
                    if !self.fused_ce_fwd_lse.is_empty() {
                        let slot_vid_set: std::collections::HashSet<crate::wengert::VarId> =
                            slots.iter().map(|(v, _)| *v).collect();
                        let mut by_val: std::collections::HashMap<
                            Value,
                            Vec<crate::wengert::VarId>,
                        > = std::collections::HashMap::new();
                        for (vid, val) in full_vars.iter() {
                            by_val.entry(*val).or_default().push(*vid);
                        }
                        let mut entries: Vec<(Vec<crate::wengert::VarId>, Value)> = self
                            .fused_ce_fwd_lse
                            .iter()
                            .filter_map(|(res_val, lse_val)| {
                                let mut vids: Vec<crate::wengert::VarId> = by_val
                                    .get(res_val)?
                                    .iter()
                                    .copied()
                                    .filter(|v| slot_vid_set.contains(v))
                                    .collect();
                                if vids.is_empty() {
                                    return None;
                                }
                                vids.sort_unstable();
                                Some((vids, *lse_val))
                            })
                            .collect();
                        entries.sort_by_key(|(vids, _)| vids[0]);
                        for (vids, lse_val) in entries {
                            let idx = slots.len() + lse_slots.len() + fce_slots.len();
                            self.compile_call_by_name(
                                builder,
                                "nsl_list_push",
                                &[inner, lse_val],
                            )?;
                            fce_slots.push((idx, vids));
                        }
                        // The step body's adjoint is never lowered under
                        // csla, so its map entries would otherwise linger
                        // for the whole compile — clear them; the replay
                        // re-binds fresh entries keyed by SEEDED Values
                        // per consuming range. Same for the pass-through
                        // cast entries (the replay's miss path is the
                        // correct one).
                        self.fused_ce_fwd_lse.clear();
                        self.fused_ce_fwd_casts.clear();
                    }
                    // Anti-vacuity twin of the LSE line: asserted exactly
                    // by the fused-CE gates (1 slot = the carry engaged; a
                    // composite fallback shows 0 and the launch counters
                    // catch it too).
                    eprintln!(
                        "[csla] fused-ce tape-carry: {} slots",
                        fce_slots.len()
                    );

                    let so = builder.use_var(saves_outer_var);
                    self.compile_call_by_name(builder, "nsl_list_push", &[so, inner])?;
                    if has_dataloader.is_some() {
                        let dl = builder.use_var(dicts_var);
                        let batch_now = builder.use_var(step_param_var);
                        self.compile_call_by_name(builder, "nsl_list_push", &[dl, batch_now])?;
                    }
                    let hook_accum_idx: std::collections::HashMap<crate::wengert::VarId, i64> =
                        adj_vid_to_hook_entry
                            .iter()
                            .map(|(vid, e)| (*vid, e.accum_idx))
                            .collect();
                    csla_loss_buffered = loss_slot.is_some();
                    csla_teardown_slots = Some(
                        slots
                            .iter()
                            .enumerate()
                            .filter_map(|(idx, (_, kind))| match kind {
                                CslaSlotKind::Raw {
                                    owned: Some(crate::wengert::WengertType::Tensor),
                                } => Some((idx as i64, "nsl_tensor_free")),
                                CslaSlotKind::Raw {
                                    owned: Some(crate::wengert::WengertType::List),
                                } => Some((idx as i64, "nsl_list_free")),
                                _ => None,
                            })
                            // LSE slots: owned tensors when the fused launch
                            // fired, runtime 0 when it declined — the
                            // null-safe free covers both.
                            .chain(
                                lse_slots
                                    .iter()
                                    .map(|(idx, _)| (*idx as i64, "nsl_tensor_free_if_valid")),
                            )
                            // Fused-CE LSE slots: always real tensors (the
                            // fused forward allocates unconditionally); the
                            // teardown sweep is their ONLY free on the
                            // trailing partial window (replayed entries are
                            // consumed+freed by the emitted backward).
                            .chain(
                                fce_slots
                                    .iter()
                                    .map(|(idx, _)| (*idx as i64, "nsl_tensor_free")),
                            )
                            .collect(),
                    );
                    csla_pending = Some(CslaPending {
                        adjoint: adjoint.clone(),
                        slots,
                        seed_base: full_vars.clone(),
                        param_adj_set: param_adj_set.clone(),
                        hook_accum_idx,
                        accum_scale: fase_plan.recipe.accum_scale,
                        loss_slot,
                        params: pre.params,
                        primal_view_of: pre.primal_view_of,
                        lse_slots,
                        lse_pushed: csla_lse_pushed,
                        fce_slots,
                        schedule: pre.schedule,
                    });
                    None
                } else if fase_hook_active && !param_adj_set.is_empty() {
                    // FASE Deferred: consume each param gradient immediately.
                    // The callback receives &mut Compiler explicitly so no
                    // double-borrow occurs.
                    let accum_val = accum_list.ok_or_else(|| {
                        CodegenError::new(
                            "fase_hook_active requires accum_list to be Some",
                        )
                    })?;
                    let accum_scale = fase_plan.recipe.accum_scale;
                    let num_params = num_params_val;
                    let hook_map = &adj_vid_to_hook_entry;
                    let plist = param_list;
                    let mut fase_cb = |c: &mut Compiler,
                                       var_id: crate::wengert::VarId,
                                       grad_ptr: Value,
                                       still_needed: bool,
                                       b: &mut cranelift_frontend::FunctionBuilder|
                     -> Result<(), CodegenError> {
                        let Some(entry) = hook_map.get(&var_id) else {
                            return Ok(());
                        };
                        // Runtime pointer-scan: find the index in param_list that
                        // matches the primal param pointer, then use the same index
                        // for accum_list.  This is necessary because param_list and
                        // accum_list are both indexed by param_paths order, but a
                        // given primal_val may appear at any runtime slot if the
                        // model has shared/aliased weights.
                        //
                        // Fast path: use the compile-time accum_idx directly.
                        // accum_list[accum_idx] == the m_partial for this param
                        // (both are param_paths-ordered).
                        let _ = entry.primal_val; // present for future alias detection
                        let idx_val = b.ins().iconst(cranelift_codegen::ir::types::I64, entry.accum_idx);
                        let _ = num_params; // captured for guard assertions if needed
                        let _ = plist;
                        let m_partial =
                            c.compile_call_by_name(b, "nsl_list_get", &[accum_val, idx_val])?;
                        let off = c.compile_options.optim_state_offload;
                        // P0.3: note this parameter's gradient BEFORE accumulate
                        // frees/consumes it. accum_idx == the param_paths index.
                        if c.compile_options.grad_integrity {
                            c.compile_call_by_name(
                                b,
                                "nsl_grad_integrity_note",
                                &[grad_ptr, idx_val],
                            )?;
                        }
                        c.fase_emit_accumulate(b, m_partial, grad_ptr, accum_scale, off)?;
                        // Free the raw gradient now ONLY if no later adjoint op
                        // still reads it. When this param's grad adjoint is a
                        // shared intermediate (a bias whose grad == d_out, which
                        // the weight-grad matmul also consumes), the free is
                        // DEFERRED to end-of-backward cleanup — freeing here
                        // would drop the weight gradient (silently, pre-#396).
                        // `fase_emit_accumulate` leaves grad_ptr intact (rc
                        // unchanged), so the later op reads live data.
                        if !still_needed {
                            c.compile_call_by_name(b, "nsl_tensor_free", &[grad_ptr])?;
                        }
                        Ok(())
                    };
                    // P0.3: bracket the FASE backward with a grad-integrity step
                    // (the hook notes each parameter's gradient between these).
                    let gi = self.compile_options.grad_integrity;
                    if gi {
                        self.compile_call_by_name(
                            builder,
                            "nsl_grad_integrity_step_begin",
                            &[num_params_val],
                        )?;
                    }
                    // P0.2: arm the gradient-integrity guard for the FASE
                    // adjoint lowering, then disarm before the match so it
                    // never leaks into a later (forward / free-list) lowering.
                    self.grad_live_results = grad_live_set.clone();
                    let fase_lowered = crate::wengert_lower::compile_wengert_ops(
                        self,
                        builder,
                        state,
                        &adjoint,
                        full_vars,
                        Some((&param_adj_set, &mut fase_cb)),
                    );
                    self.grad_live_results = None;
                    let fase_out = match fase_lowered {
                        Ok(gv) => Some(gv),
                        Err(e) => {
                            eprintln!(
                                "[nsl] source AD lowering (FASE hook) failed ({}), \
                                 rerun without --source-ad",
                                e
                            );
                            return Err(e);
                        }
                    };
                    if gi {
                        self.compile_call_by_name(
                            builder,
                            "nsl_grad_integrity_step_end",
                            &[],
                        )?;
                    }
                    fase_out
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
                            eprintln!(
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
                        eprintln!("[nsl] source-ad owned {} ops:", label);
                        for (name, count) in counts {
                            eprintln!("  {} -> {}", name, count);
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
                        let Some(adj_vid) = gen.adjoint_of(*vid) else {
                            continue;
                        };
                        if let Some(op) = adjoint.ops.iter().find(|op| op.result == adj_vid) {
                            let key = format!("{:?}", op.op);
                            *final_grad_counts.entry(key).or_insert(0) += 1;
                        }
                    }
                    let mut final_grad_counts: Vec<_> = final_grad_counts.into_iter().collect();
                    final_grad_counts.sort_by_key(|a| std::cmp::Reverse(a.1));
                    eprintln!("[nsl] source-ad final grad ops:");
                    for (name, count) in final_grad_counts {
                        eprintln!("  {} -> {}", name, count);
                    }
                }

                // 8. Collect parameter gradients into grads_list (NslList)
                //
                // When FASE hook is active, every parameter gradient was already
                // consumed (accumulated into m_partial + freed) during adjoint
                // lowering.  Skip the grads_list construction entirely and emit
                // a null sentinel so downstream code that is also guarded by
                // `!fase_hook_active` never dereferences it.
                //
                // When hook is inactive: seed grads_list from the runtime
                // param_list so it always has exactly one slot per collected
                // parameter, then overwrite the matching slots using source-AD
                // gradients keyed by parameter pointer identity.  This avoids
                // relying on a compile-time DFS path enumeration matching the
                // runtime collector's traversal.
                let grads = if !fase_hook_active {
                let grads_inner = self.compile_call_by_name(builder, "nsl_list_new", &[])?;

                // P0.1: gradient seed buffers under the Grads surface. The
                // ambient surface here is Activations (set at the transient-
                // pool flip); get/set keeps the bracket nesting-safe.
                let surface_grads_prev =
                    self.compile_call_by_name(builder, "nsl_gpu_get_alloc_surface", &[])?;
                let surface_grads = builder.ins().iconst(cl_types::I8, SURFACE_GRADS);
                self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surface_grads])?;

                // 8a. Initialize one zero gradient per runtime parameter.
                let fill_i_var = state.new_variable();
                builder.declare_var(fill_i_var, cl_types::I64);
                let fill_zero = builder.ins().iconst(cl_types::I64, 0);
                builder.def_var(fill_i_var, fill_zero);
                let fill_hdr = builder.create_block();
                let fill_body = builder.create_block();
                let fill_exit = builder.create_block();
                builder.ins().jump(fill_hdr, &[]);
                builder.switch_to_block(fill_hdr);
                let fi = builder.use_var(fill_i_var);
                let fc = builder
                    .ins()
                    .icmp(IntCC::SignedLessThan, fi, num_params_val);
                builder.ins().brif(fc, fill_body, &[], fill_exit, &[]);
                builder.switch_to_block(fill_body);
                builder.seal_block(fill_body);
                let p = self.compile_call_by_name(builder, "nsl_list_get", &[param_list, fi])?;
                let z = self.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[p])?;
                self.compile_call_by_name(builder, "nsl_list_push", &[grads_inner, z])?;
                let fill_one = builder.ins().iconst(cl_types::I64, 1);
                let fill_next = builder.ins().iadd(fi, fill_one);
                builder.def_var(fill_i_var, fill_next);
                builder.ins().jump(fill_hdr, &[]);
                builder.seal_block(fill_hdr);
                builder.switch_to_block(fill_exit);
                builder.seal_block(fill_exit);
                state.current_block = Some(fill_exit);

                // End of the Grads bracket — restore the ambient surface.
                self.compile_call_by_name(
                    builder,
                    "nsl_gpu_set_alloc_surface",
                    &[surface_grads_prev],
                )?;

                // 8b. Replace zero slots with actual source-AD gradients by
                // scanning the runtime param_list for each resolved parameter
                // leaf from the extracted forward graph.
                let tensor_param_paths: std::collections::HashSet<String> = self
                    .enumerate_all_model_tensor_paths(&model_var_name, &model_type_name)
                    .into_iter()
                    .collect();
                let trainable_tensor_param_paths: std::collections::HashSet<String> =
                    tensor_param_paths
                        .iter()
                        .filter(|path| self.is_trainable_param_name(path))
                        .cloned()
                        .collect();
                let mut seen_trainable_tensor_params: std::collections::HashSet<String> =
                    std::collections::HashSet::new();

                let mut grad_connected = 0usize;
                let mut grad_ignored_config_tensor = 0usize;
                let mut grad_ignored_non_tensor = 0usize;
                let mut grad_skipped_no_primal = 0usize;
                let mut grad_skipped_no_adjoint = 0usize;
                let mut grad_skipped_no_lowered = 0usize;
                for (param_name, vid) in extractor.named_param_var_ids() {
                    match classify_source_ad_param_name(param_name, &tensor_param_paths) {
                        SourceAdParamDiagnosticKind::Trainable => {
                            seen_trainable_tensor_params.insert(param_name.clone());
                        }
                        SourceAdParamDiagnosticKind::IgnoredConfig => {
                            grad_ignored_config_tensor += 1;
                            continue;
                        }
                        SourceAdParamDiagnosticKind::IgnoredNonTensor => {
                            grad_ignored_non_tensor += 1;
                            continue;
                        }
                    }
                    let Some(param_ptr) = full_vars.get(vid).copied() else {
                        eprintln!("[nsl] source AD: param '{}' has no primal value (VarId {:?} not in full_vars)", param_name, vid);
                        grad_skipped_no_primal += 1;
                        continue;
                    };
                    let Some(adj_vid) = gen.adjoint_of(*vid) else {
                        eprintln!("[nsl] source AD: param '{}' has no adjoint (VarId {:?} — no gradient generated)", param_name, vid);
                        grad_skipped_no_adjoint += 1;
                        continue;
                    };
                    let Some(grad_val) = grad_vars
                        .expect("non-hook path always lowers the adjoint inline")
                        .get(&adj_vid)
                        .copied()
                    else {
                        eprintln!("[nsl] source AD: param '{}' adjoint VarId {:?} not in lowered grad vars (cascade skip)", param_name, adj_vid);
                        grad_skipped_no_lowered += 1;
                        continue;
                    };
                    grad_connected += 1;

                    // WRGA B.3.2 Option 3: adapter-injected params (lora_A_*,
                    // lora_B_*, ia3_scale_*, gate_*) are NOT in the runtime
                    // param_list (the runtime list is built from
                    // `enumerate_model_tensor_paths`, which excludes side-table
                    // entries). Skip the align-with-runtime scan for them —
                    // we count them as connected for the gradient-summary
                    // diagnostic but don't emit the `nsl_assert` that would
                    // always fire "missing param" for these paths.
                    let leaf = param_name
                        .rsplit('.')
                        .next()
                        .unwrap_or(param_name.as_str());
                    if crate::expr::access::is_synthesized_adapter_field_name(leaf) {
                        continue;
                    }

                    let scan_i_var = state.new_variable();
                    let scan_match_count_var = state.new_variable();
                    builder.declare_var(scan_i_var, cl_types::I64);
                    builder.declare_var(scan_match_count_var, cl_types::I64);
                    let scan_zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(scan_i_var, scan_zero);
                    builder.def_var(scan_match_count_var, scan_zero);

                    let scan_hdr = builder.create_block();
                    let scan_body = builder.create_block();
                    let scan_match = builder.create_block();
                    let scan_next = builder.create_block();
                    let scan_exit = builder.create_block();

                    builder.ins().jump(scan_hdr, &[]);
                    builder.switch_to_block(scan_hdr);
                    let si = builder.use_var(scan_i_var);
                    let scan_cond = builder
                        .ins()
                        .icmp(IntCC::SignedLessThan, si, num_params_val);
                    builder
                        .ins()
                        .brif(scan_cond, scan_body, &[], scan_exit, &[]);

                    builder.switch_to_block(scan_body);
                    builder.seal_block(scan_body);
                    let runtime_param =
                        self.compile_call_by_name(builder, "nsl_list_get", &[param_list, si])?;
                    let is_match = builder.ins().icmp(IntCC::Equal, runtime_param, param_ptr);
                    builder
                        .ins()
                        .brif(is_match, scan_match, &[], scan_next, &[]);

                    builder.switch_to_block(scan_match);
                    builder.seal_block(scan_match);
                    let old_grad =
                        self.compile_call_by_name(builder, "nsl_list_get", &[grads_inner, si])?;
                    // ELTLS (FBIP-3): nsl_tensor_add takes a flags byte (flags=0 here).
                    let flags_zero = builder.ins().iconst(cl_types::I8, 0);
                    let summed_grad = self.compile_call_by_name(
                        builder,
                        "nsl_tensor_add",
                        &[old_grad, grad_val, flags_zero],
                    )?;
                    let match_count = builder.use_var(scan_match_count_var);
                    let match_one = builder.ins().iconst(cl_types::I64, 1);
                    let next_match_count = builder.ins().iadd(match_count, match_one);
                    builder.def_var(scan_match_count_var, next_match_count);
                    self.compile_call_by_name(builder, "nsl_tensor_free", &[old_grad])?;
                    self.compile_call_by_name(builder, "nsl_list_set", &[grads_inner, si, summed_grad])?;
                    builder.ins().jump(scan_next, &[]);

                    builder.switch_to_block(scan_next);
                    builder.seal_block(scan_next);
                    let scan_one = builder.ins().iconst(cl_types::I64, 1);
                    let scan_inc = builder.ins().iadd(si, scan_one);
                    builder.def_var(scan_i_var, scan_inc);
                    builder.ins().jump(scan_hdr, &[]);

                    builder.seal_block(scan_hdr);
                    builder.switch_to_block(scan_exit);
                    builder.seal_block(scan_exit);
                    let match_count = builder.use_var(scan_match_count_var);
                    let matched = builder.ins().icmp_imm(IntCC::NotEqual, match_count, 0);
                    let missing_msg = format!(
                        "source AD gradient could not be aligned with runtime param list: {}",
                        param_name,
                    );
                    self.intern_string(&missing_msg)?;
                    let missing_msg_ptr = self.compile_string_literal(builder, &missing_msg)?;
                    self.compile_call_by_name(builder, "nsl_assert", &[matched, missing_msg_ptr])?;
                    state.current_block = Some(scan_exit);
                }

                let grad_missing_trainable = trainable_tensor_param_paths
                    .len()
                    .saturating_sub(seen_trainable_tensor_params.len());
                eprintln!(
                    "[nsl] source AD gradient summary: {}/{} trainable tensor params connected, {} missing-from-forward, {} no-primal, {} no-adjoint, {} cascade-skip, {} ignored config-tensor, {} ignored non-tensor",
                    grad_connected,
                    trainable_tensor_param_paths.len(),
                    grad_missing_trainable,
                    grad_skipped_no_primal,
                    grad_skipped_no_adjoint,
                    grad_skipped_no_lowered,
                    grad_ignored_config_tensor,
                    grad_ignored_non_tensor,
                );
                grads_inner // value returned from the `if !fase_hook_active` arm
                } else {
                    // Hook consumed all param grads during adjoint lowering.
                    // Emit a null sentinel — downstream grads_list consumers are
                    // all guarded by `!fase_hook_active`.
                    builder.ins().iconst(cl_types::I64, 0)
                };

                let mut retained_full_vars = std::collections::HashSet::new();
                retained_full_vars.insert(loss_var_id);
                // CCR: block interiors were freed right after the forward
                // (the early-free mini list) — the bulk free must skip them.
                retained_full_vars.extend(ccr_freed_primal.iter().copied());
                // CCR compressed saves: originals were freed by the primal
                // TAIL (recorded on the main primal lowering), and the half
                // tensors were freed by the adjoint's decompress splice
                // (recorded on the adjoint lowering, but owned by the
                // PRIMAL's owned_values since the tail produced them).
                retained_full_vars.extend(full_lowered.explicit_freed_vars.iter().copied());
                if let Some(gl) = &grad_lowered {
                    retained_full_vars.extend(gl.explicit_freed_vars.iter().copied());
                }
                // CSLA: the window-buffered imports must survive this
                // iteration — their frees happen after each buffered
                // micro-batch's replay in the window backward (or at the
                // window/teardown sweeps for shells and the partial tail).
                if let Some(p) = &csla_pending {
                    retained_full_vars.extend(p.slots.iter().map(|(v, _)| *v));
                    // LSE tape-carry: the PUSHED logsumexps ride owned_values
                    // under the u32::MAX sentinel — buffered per micro-batch,
                    // freed by the window backward (or the teardown sweep),
                    // never by this per-iteration bulk free. Retention is
                    // per-VALUE (review F2): sentinel entries whose aux out
                    // mapped to no buffered slot were NOT pushed — free those
                    // right here, per iteration, exactly as the baseline
                    // would have.
                    if !p.lse_slots.is_empty() {
                        retained_full_vars.insert(u32::MAX);
                        for (vid, val, _) in &full_lowered.owned_values {
                            if *vid == u32::MAX && !p.lse_pushed.contains(val) {
                                self.compile_call_by_name(
                                    builder,
                                    "nsl_tensor_free_if_valid",
                                    &[*val],
                                )?;
                            }
                        }
                    }
                }
                if let Some(gl) = &grad_lowered {
                    self.free_wengert_owned_values(
                        builder,
                        &gl.owned_values,
                        &freed_adjoint_vars,
                    )?;
                }
                self.free_wengert_owned_values(
                    builder,
                    &full_lowered.owned_values,
                    &retained_full_vars,
                )?;

                // Collect all Cranelift Values freed by the Wengert cleanup so the
                // step-variable sweep (below) doesn't double-free them.
                let mut wengert_freed: std::collections::HashSet<Value> = std::collections::HashSet::new();
                if let Some(gl) = &grad_lowered {
                    for (vid, val, _) in &gl.owned_values {
                        if !freed_adjoint_vars.contains(vid) {
                            wengert_freed.insert(*val);
                        }
                    }
                }
                for (vid, val, _) in &full_lowered.owned_values {
                    if !retained_full_vars.contains(vid) {
                        wengert_freed.insert(*val);
                    }
                    // CCR early-freed interiors (mini list), compressed
                    // originals (primal tail) and halves (adjoint splice)
                    // were freed outside the bulk cleanup — still "already
                    // freed" as far as the step-variable sweep is concerned.
                    if ccr_freed_primal.contains(vid)
                        || full_lowered.explicit_freed_vars.contains(vid)
                        || grad_lowered
                            .as_ref()
                            .is_some_and(|gl| gl.explicit_freed_vars.contains(vid))
                    {
                        wengert_freed.insert(*val);
                    }
                }
                // CSLA: buffered imports are handled by the window backward's
                // per-b frees — the step-variable sweep must treat them as
                // already handled or it would free a buffered value that a
                // later micro-batch's replay still reads.
                if let Some(p) = &csla_pending {
                    for (vid, _) in &p.slots {
                        if let Some(v) = full_vars.get(vid) {
                            wengert_freed.insert(*v);
                        }
                    }
                    // LSE tape-carry: the sentinel-owned logsumexp values are
                    // buffer-managed too.
                    if !p.lse_slots.is_empty() {
                        for (vid, val, _) in &full_lowered.owned_values {
                            if *vid == u32::MAX {
                                wengert_freed.insert(*val);
                            }
                        }
                    }
                }

                let false_val = builder.ins().iconst(cl_types::I8, 0);
                self.compile_call_by_name(builder, "nsl_set_training_mode", &[false_val])?;

                (grads, loss_val, true, wengert_freed)
            }
        } else {
            // === Tape AD path (runtime backward) ===
            let (grads, loss) =
                self.compile_tape_backward(builder, state, step_body, param_list)?;
            (grads, loss, false, std::collections::HashSet::new())
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
        if self.compile_options.health_monitor || self.compile_options.inspect_enabled {
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

        if self.compile_options.health_monitor {
            use cranelift_codegen::ir::{types as cl_types, MemFlags};
            let _ = MemFlags::trusted(); // keep import valid across cfgs

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
                .profile_source_file_name
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
        let should_step_var = state.new_variable();
        builder.declare_var(should_step_var, cl_types::I8);
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

        if !fase_hook_active {
        if let Some(accum) = accum_list {
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
            let ga_i_var = state.new_variable();
            builder.declare_var(ga_i_var, cl_types::I64);
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
        }
        } // end if !fase_hook_active (per-micro-batch accumulation guard)

        // ── 7e3b. CSLA Stage-2: window backward phase ───────────────────
        // On accumulation boundaries, replay the adjoint once per buffered
        // micro-batch. Each replay seeds a fresh VarMap from seed_base (the
        // step iteration's forward values — loop-invariant Params/Constants
        // resolve there) overridden with that micro-batch's buffered imports,
        // then lowers the SAME adjoint tape inside a runtime b-loop (one
        // tape, N executions). The FASE hook accumulates every parameter
        // gradient into the same m_partial slot in micro-batch-ascending
        // order — exactly the baseline's per-parameter accumulation sequence,
        // so the optimizer region below consumes bit-identical m_partial.
        if let Some(pending) = csla_pending.take() {
            let (saves_outer_var, dicts_var) =
                csla_buffers.expect("csla_buffers allocated when csla_pending set");
            let accum_val = accum_list
                .ok_or_else(|| CodegenError::new("csla requires accum_list"))?;

            // ── D1b schedule (precomputed) ──────────────────────────────
            // D2b part 2: ranges / grouping / streamed set were derived
            // ONCE in the pre-forward pure pipeline (the segment-streamed
            // forward consumed them at emission time) and travel here via
            // `pending.schedule` — one derivation, both sides agree by
            // construction. The `[csla] layer-major schedule:` line prints
            // at the derivation site.
            let ranges = &pending.schedule.ranges;
            let layer_group = &pending.schedule.layer_group;
            let global_group = &pending.schedule.global_group;
            let n_ranges = ranges.len();
            let last_ri = n_ranges - 1;

            // Carry analysis: adjoint values produced in one range and read
            // in a later one (the boundary adjoints d(residual-after-L) plus
            // any straggler temporaries). Slot order sorted-by-vid.
            let mut produced_range: std::collections::HashMap<crate::wengert::VarId, usize> =
                std::collections::HashMap::new();
            for (ri, r) in ranges.iter().enumerate() {
                for op in &pending.adjoint.ops[r.start..r.end] {
                    produced_range.insert(op.result, ri);
                }
            }
            let mut carry_set: std::collections::BTreeSet<crate::wengert::VarId> =
                Default::default();
            let mut carry_last_read: std::collections::HashMap<crate::wengert::VarId, usize> =
                std::collections::HashMap::new();
            let mut carry_freed_by_marker: std::collections::HashSet<crate::wengert::VarId> =
                Default::default();
            for (ri, r) in ranges.iter().enumerate() {
                for op in &pending.adjoint.ops[r.start..r.end] {
                    for &input in &op.inputs {
                        let Some(&pr) = produced_range.get(&input) else {
                            continue;
                        };
                        if pr < ri {
                            carry_set.insert(input);
                            let e = carry_last_read.entry(input).or_insert(ri);
                            if *e < ri {
                                *e = ri;
                            }
                            if matches!(op.op, crate::wengert::PrimalOp::FreeTensor) {
                                carry_freed_by_marker.insert(input);
                            }
                        }
                    }
                }
            }
            let carry_vids: Vec<crate::wengert::VarId> = carry_set.into_iter().collect();
            let carry_slot: std::collections::HashMap<crate::wengert::VarId, usize> =
                carry_vids.iter().enumerate().map(|(i, v)| (*v, i)).collect();
            let n_carry = carry_vids.len();
            let mut exports_per_range: Vec<Vec<crate::wengert::VarId>> =
                vec![Vec::new(); n_ranges];
            for &v in &carry_vids {
                exports_per_range[produced_range[&v]].push(v);
            }
            let mut imports_per_range: Vec<Vec<crate::wengert::VarId>> =
                vec![Vec::new(); n_ranges];
            for (ri, r) in ranges.iter().enumerate() {
                let mut seen = std::collections::HashSet::new();
                for op in &pending.adjoint.ops[r.start..r.end] {
                    for &input in &op.inputs {
                        if carry_slot.contains_key(&input)
                            && produced_range[&input] < ri
                            && seen.insert(input)
                        {
                            imports_per_range[ri].push(input);
                        }
                    }
                }
                imports_per_range[ri].sort_unstable();
            }
            // Fixpoint-extended last-use POSITIONS (review D1b-3): frees must
            // key off list-membership-extended last use — a slot or carry
            // consumed by a list-building op in range R whose LIST is read in
            // range S>R must survive to S (lists hold raw, un-refcounted
            // element pointers; freeing at the direct read would dangle
            // them). Seeding stays direct-read-based (only actual op inputs
            // need values).
            let extended_last_pos: std::collections::HashMap<crate::wengert::VarId, usize> = {
                let mut lu: std::collections::HashMap<crate::wengert::VarId, usize> =
                    std::collections::HashMap::new();
                for (idx, op) in pending.adjoint.ops.iter().enumerate() {
                    for &input in &op.inputs {
                        lu.insert(input, idx);
                    }
                }
                crate::ccr::extend_last_use_through_lists(&pending.adjoint, &mut lu);
                lu
            };
            let range_of_pos = |pos: usize| -> usize {
                ranges
                    .iter()
                    .position(|r| pos >= r.start && pos < r.end)
                    .unwrap_or(last_ri)
            };

            // Carried Tensor-typed values with NO FreeTensor consumer get an
            // explicit free after their (fixpoint-extended) last consuming
            // range's replay — the belt for values the last-use-frees pass
            // protected or never saw. Skips at emission time (below) also
            // exclude hook-freed raw grads (review D1b-1: a range boundary
            // landing ON a reduce_to_shape grad op makes its raw-grad input a
            // carry that the hook's extra free already releases — freeing it
            // again here would be a double free). List-typed carries are
            // never freed (matching free_eligible's List exclusion).
            let mut carry_explicit_free: Vec<Vec<crate::wengert::VarId>> =
                vec![Vec::new(); n_ranges];
            for &v in &carry_vids {
                if carry_freed_by_marker.contains(&v) {
                    continue;
                }
                if !matches!(
                    pending.adjoint.var_types.get(&v),
                    Some(crate::wengert::WengertType::Tensor) | None
                ) {
                    continue;
                }
                let last_pos = extended_last_pos
                    .get(&v)
                    .copied()
                    .unwrap_or(pending.adjoint.ops.len().saturating_sub(1));
                let ri = range_of_pos(last_pos).max(carry_last_read[&v]);
                carry_explicit_free[ri].push(v);
            }
            // Primal-slot schedule: which ranges read each buffered slot
            // (seed only there — direct reads) and which range is a slot's
            // fixpoint-extended LAST reader (free it there, with the
            // loss-slot b<N-1 conditional).
            let mut slot_seed_per_range: Vec<Vec<usize>> = vec![Vec::new(); n_ranges];
            let mut slot_free_per_range: Vec<Vec<usize>> = vec![Vec::new(); n_ranges];
            {
                let slot_of: std::collections::HashMap<crate::wengert::VarId, usize> = pending
                    .slots
                    .iter()
                    .enumerate()
                    .map(|(i, (v, _))| (*v, i))
                    .collect();
                for (ri, r) in ranges.iter().enumerate() {
                    let mut seen = std::collections::HashSet::new();
                    for op in &pending.adjoint.ops[r.start..r.end] {
                        for &input in &op.inputs {
                            if let Some(&si) = slot_of.get(&input) {
                                if seen.insert(si) {
                                    slot_seed_per_range[ri].push(si);
                                }
                            }
                        }
                    }
                    slot_seed_per_range[ri].sort_unstable();
                }
                for (vid, si) in &slot_of {
                    if let Some(&pos) = extended_last_pos.get(vid) {
                        slot_free_per_range[range_of_pos(pos)].push(*si);
                    }
                }
                for v in slot_free_per_range.iter_mut() {
                    v.sort_unstable();
                }
            }

            // Tape-carry review F3: an SDPA backward CLUSTER — the recompute
            // clone of the forward (whose Value keys the aux re-bind under
            // the Block policy) plus its dQ/dK/dV extract ops — must not be
            // split across replay ranges. The aux insert and the
            // flash_attn_bwd_cache are Value-keyed within one range's
            // lowering: a split would silently downgrade the packed backward
            // to the CPU reference (numeric divergence) and double-emit the
            // full backward triplet. Sequential-block models are
            // structurally safe (a layer's extracts sit inside its own
            // range); this guard makes any future violation loud.
            {
                let mut fwdout_range: std::collections::HashMap<
                    crate::wengert::VarId,
                    usize,
                > = std::collections::HashMap::new();
                for (ri, r) in ranges.iter().enumerate() {
                    for op in &pending.adjoint.ops[r.start..r.end] {
                        let is_extract = matches!(
                            op.op,
                            crate::wengert::PrimalOp::FlashAttentionBackwardExtract { .. }
                                | crate::wengert::PrimalOp::FlashAttentionBackwardExtractPacked { .. }
                        );
                        if !is_extract {
                            continue;
                        }
                        let Some(&fo) = op.inputs.get(4) else { continue };
                        if let Some(&pr) = produced_range.get(&fo) {
                            if pr != ri {
                                return Err(CodegenError::new(format!(
                                    "--layerwise-accum: an SDPA backward extract in \
                                     replay range {ri} reads a forward clone from \
                                     range {pr} — the Value-keyed attention \
                                     side-bands cannot cross ranges. This adjoint \
                                     shape is unsupported; drop --layerwise-accum",
                                )));
                            }
                        }
                        if let Some(&prev) = fwdout_range.get(&fo) {
                            if prev != ri {
                                return Err(CodegenError::new(format!(
                                    "--layerwise-accum: sibling SDPA backward \
                                     extracts for one attention op are split \
                                     across replay ranges {prev} and {ri}. This \
                                     adjoint shape is unsupported; drop \
                                     --layerwise-accum",
                                )));
                            }
                        } else {
                            fwdout_range.insert(fo, ri);
                        }
                    }
                }
            }

            // Fused-CE cluster guard (F3 twin): the three
            // FusedLinearCeBackwardExtract components of one op share the
            // Value-keyed `fused_ce_bwd_cache` (component 0 launches and
            // populates; 1/2 read; 2 evicts) and the LSE is consumed
            // exactly once — a split across replay ranges would relaunch
            // the backward kernel per range and hit a consumed-LSE
            // CodegenError. Structurally the extracts sit together in the
            // prologue range (the fused op is the CCR epilogue's loss
            // head); this guard makes any future violation loud.
            {
                let mut fce_range: std::collections::HashMap<
                    crate::wengert::VarId,
                    usize,
                > = std::collections::HashMap::new();
                for (ri, r) in ranges.iter().enumerate() {
                    for op in &pending.adjoint.ops[r.start..r.end] {
                        if !matches!(
                            op.op,
                            crate::wengert::PrimalOp::FusedLinearCeBackwardExtract { .. }
                        ) {
                            continue;
                        }
                        let Some(&fr) = op.inputs.get(5) else { continue };
                        if let Some(&prev) = fce_range.get(&fr) {
                            if prev != ri {
                                return Err(CodegenError::new(format!(
                                    "--layerwise-accum: sibling fused-CE backward \
                                     extracts for one @fused_lm_ce op are split \
                                     across replay ranges {prev} and {ri}. This \
                                     adjoint shape is unsupported; drop \
                                     --layerwise-accum",
                                )));
                            }
                        } else {
                            fce_range.insert(fr, ri);
                        }
                    }
                }
            }

            // LSE tape-carry: which range frees each buffered logsumexp —
            // the last (fixpoint-extended) range reading ANY of its fwd-out
            // vids; that range necessarily seeds the vid, so the loaded LSE
            // value is in scope for the null-safe free.
            let lse_free_range: Vec<usize> = pending
                .lse_slots
                .iter()
                .map(|(_, vids)| {
                    vids.iter()
                        .filter_map(|v| extended_last_pos.get(v))
                        .map(|&p| range_of_pos(p))
                        .max()
                        .unwrap_or(last_ri)
                })
                .collect();

            // Review D1b-2: zero-copy VIEWS OF θ (transpose/reshape chains
            // rooted at a trainable param) alias parameter storage, so a
            // read through one AFTER that param's per-layer update would see
            // half-updated weights — invisible to the classification, which
            // tracks only the param's DIRECT reads. Refuse any read of a
            // view-of-θ in a range strictly after θ's update range. Views of
            // epilogue-group params (update range = MAX — after every
            // range) can never trip this: the tied-embedding LM head is the
            // canonical safe case. Both primal-side view chains (buffered
            // slots) and adjoint-side view chains are tracked transitively.
            {
                let idx_update_range: std::collections::HashMap<i64, usize> = layer_group
                    .iter()
                    .enumerate()
                    .flat_map(|(ri, g)| g.iter().map(move |&i| (i, ri)))
                    .collect();
                let update_range_of_vid: std::collections::HashMap<
                    crate::wengert::VarId,
                    usize,
                > = pending
                    .params
                    .iter()
                    .map(|cp| {
                        (
                            cp.primal_vid,
                            idx_update_range
                                .get(&cp.accum_idx)
                                .copied()
                                .unwrap_or(usize::MAX),
                        )
                    })
                    .collect();
                let mut view_param: std::collections::HashMap<
                    crate::wengert::VarId,
                    crate::wengert::VarId,
                > = pending.primal_view_of.clone();
                for (ri, r) in ranges.iter().enumerate() {
                    for op in &pending.adjoint.ops[r.start..r.end] {
                        for &input in &op.inputs {
                            if let Some(&p) = view_param.get(&input) {
                                let ur = update_range_of_vid
                                    .get(&p)
                                    .copied()
                                    .unwrap_or(usize::MAX);
                                if ri > ur {
                                    return Err(CodegenError::new(format!(
                                        "--layerwise-accum: a view of parameter \
                                         VarId {p} is read in replay range {ri}, \
                                         after the param's per-layer update in \
                                         range {ur} — the view aliases θ's storage \
                                         and would see half-updated weights. This \
                                         adjoint shape is unsupported; drop \
                                         --layerwise-accum",
                                    )));
                                }
                            }
                        }
                        if crate::wengert::is_view_producing_op(&op.op) {
                            for &input in &op.inputs {
                                if update_range_of_vid.contains_key(&input) {
                                    view_param.insert(op.result, input);
                                } else if let Some(&p) = view_param.get(&input) {
                                    view_param.insert(op.result, p);
                                }
                            }
                        }
                    }
                }
            }

            // ── Emission helpers (compile-time unrolled per group) ──────
            fn emit_csla_accum_alloc(
                c: &mut Compiler,
                builder: &mut FunctionBuilder,
                param_list: Value,
                accum_val: Value,
                idxs: &[i64],
            ) -> Result<(), CodegenError> {
                if idxs.is_empty() {
                    return Ok(());
                }
                // Fresh zeros_like per window under the MPartial surface —
                // identical initial bytes to the baseline's zeroed
                // persistent buffers.
                let prev =
                    c.compile_call_by_name(builder, "nsl_gpu_get_alloc_surface", &[])?;
                let surf = builder.ins().iconst(cl_types::I8, SURFACE_M_PARTIAL);
                c.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[surf])?;
                for &i in idxs {
                    let iv = builder.ins().iconst(cl_types::I64, i);
                    let p = c.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
                    let z = c.compile_call_by_name(builder, "nsl_tensor_zeros_like", &[p])?;
                    c.compile_call_by_name(builder, "nsl_list_set", &[accum_val, iv, z])?;
                }
                c.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[prev])?;
                Ok(())
            }
            #[allow(clippy::too_many_arguments)]
            fn emit_csla_group_update(
                c: &mut Compiler,
                builder: &mut FunctionBuilder,
                param_list: Value,
                state_list_1: Value,
                state_list_2: Value,
                two_state: bool,
                accum_val: Value,
                recipe: &crate::fase::UpdateRecipe,
                bc: (Value, Value),
                wrap_precision: bool,
                wrap_offload: bool,
                idxs: &[i64],
            ) -> Result<(), CodegenError> {
                for &i in idxs {
                    let iv = builder.ins().iconst(cl_types::I64, i);
                    let theta =
                        c.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
                    let m =
                        c.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, iv])?;
                    let m_partial =
                        c.compile_call_by_name(builder, "nsl_list_get", &[accum_val, iv])?;
                    let v = if two_state {
                        c.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, iv])?
                    } else {
                        m
                    };
                    // D2a: wrap_offload stages host-pinned m/v to θ's device
                    // for the update and streams them back asynchronously —
                    // the whole point of the layer-major schedule is that
                    // only ONE layer's m/v are staged at a time. The
                    // envelope's m_partial leg degenerates safely here (the
                    // accumulator is device-resident, so its "stage" is a
                    // same-device refcount bump and the tail's zero is
                    // wasted-but-correct before our free below).
                    c.fase_emit_final_step(
                        builder,
                        theta,
                        m,
                        m_partial,
                        v,
                        recipe,
                        Some(bc),
                        wrap_precision,
                        wrap_offload,
                    )?;
                    // The baseline zeroes m_partial for reuse; the layerwise
                    // schedule frees it — next window allocates fresh zeros.
                    c.compile_call_by_name(builder, "nsl_tensor_free", &[m_partial])?;
                    let z = builder.ins().iconst(cl_types::I64, 0);
                    c.compile_call_by_name(builder, "nsl_list_set", &[accum_val, iv, z])?;
                }
                // D2a: the async DtoH stage-outs defer their frees to the
                // drain — one per GROUP update, so at most one layer's
                // staged tensors are ever in flight.
                if wrap_offload && !idxs.is_empty() {
                    c.compile_call_by_name(builder, "nsl_offload_drain", &[])?;
                }
                Ok(())
            }

            // ── Window region ───────────────────────────────────────────
            let bwd_block = builder.create_block();
            let bwd_join = builder.create_block();
            let ss = builder.use_var(should_step_var);
            builder.ins().brif(ss, bwd_block, &[], bwd_join, &[]);
            builder.switch_to_block(bwd_block);
            builder.seal_block(bwd_block);
            state.current_block = Some(bwd_block);

            // Anti-vacuity mark + loud window-size assert: the modulo fires
            // every N micro-batches exactly (the counter is global, windows
            // straddle epochs, the trailing partial window never fires), so
            // the buffer MUST hold exactly N entries here.
            self.compile_call_by_name(builder, "nsl_csla_window_mark", &[])?;
            let so = builder.use_var(saves_outer_var);
            let win_len = self.compile_call_by_name(builder, "nsl_list_len", &[so])?;
            let n_val = builder.ins().iconst(cl_types::I64, grad_accumulation_steps);
            let len_ok = builder.ins().icmp(IntCC::Equal, win_len, n_val);
            let len_msg = format!(
                "csla window backward: buffered micro-batch count != \
                 grad_accumulation ({grad_accumulation_steps})"
            );
            self.intern_string(&len_msg)?;
            let len_msg_ptr = self.compile_string_literal(builder, &len_msg)?;
            self.compile_call_by_name(builder, "nsl_assert", &[len_ok, len_msg_ptr])?;

            // D2b weight eviction, part 2 (whole-loop streaming): the
            // forward already registered + evicted every streamed param at
            // step-body top and re-uploads per segment, so this window
            // opens with them evicted. The register loop stays as an
            // idempotent belt (a register on an evicted registered tensor
            // is a no-op) — it keeps the window arm self-sufficient if the
            // forward emission ever changes. The streamed SET comes from
            // the shared pre-forward schedule (view-rooted params already
            // excluded there — review D2b-1); each layer re-uploads at its
            // range head and evicts+writes-back after its update; epilogue
            // params never stream.
            let ws_active = self.compile_options.weight_stream;
            let ws_streamed: std::collections::HashSet<i64> =
                pending.schedule.ws_streamed.iter().copied().collect();
            if ws_active {
                for &idx in &pending.schedule.ws_streamed {
                    let iv = builder.ins().iconst(cl_types::I64, idx);
                    let pw =
                        self.compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
                    self.compile_call_by_name(builder, "nsl_weight_stream_register", &[pw])?;
                }
            }

            // Bias correction — the same expression the (now-bypassed)
            // optimizer site computes; step_count is untouched between here
            // and there, so every group in this window shares one pair.
            let sc_val = builder.use_var(step_count_var);
            let one_i64 = builder.ins().iconst(cl_types::I64, 1);
            let sc_plus_one = builder.ins().iadd(sc_val, one_i64);
            let ga_const = builder.ins().iconst(cl_types::I64, grad_accumulation_steps);
            let opt_step = builder.ins().sdiv(sc_plus_one, ga_const);
            let b1c = builder.ins().f64const(fase_plan.recipe.beta1);
            let b2c = builder.ins().f64const(fase_plan.recipe.beta2);
            let bc1_inv = self.compile_call_by_name(
                builder,
                "nsl_bias_correction_inv",
                &[b1c, opt_step],
            )?;
            let bc2_inv = self.compile_call_by_name(
                builder,
                "nsl_bias_correction_inv",
                &[b2c, opt_step],
            )?;
            let wrap_precision = cpdt_precision_dtypes.is_some();
            // Review D2a-1: csla x offload x reduced-precision moments (the
            // P0.3 combined cast_from_host arm) became reachable when the
            // offload refusal narrowed. The walk-through found no defect,
            // but zero gates cover the combination — refuse until a parity
            // gate exists (deferral-must-refuse).
            if wrap_precision && self.compile_options.optim_state_offload {
                return Err(CodegenError::new(
                    "--layerwise-accum with --optim-state-offload does not yet                      support a CPDT reduced-precision moment plan (the P0.3                      combined staging arm is ungated under the layerwise                      schedule). Drop the precision plan or --optim-state-offload",
                ));
            }
            let two_state = num_state_buffers >= 2;

            // Global/epilogue accumulators live for the whole window (their
            // grads may come from any range).
            emit_csla_accum_alloc(self, builder, param_list, accum_val, global_group)?;

            // Cross-range adjoint carry: one inner list per micro-batch,
            // slot-indexed, created by the first range's replay loop.
            let carry_outer = if n_carry > 0 {
                Some(self.compile_call_by_name(builder, "nsl_list_new", &[])?)
            } else {
                None
            };

            // Exports that ghost-skipped at their producing range (compile-
            // time knowledge accumulated range by range): later ranges skip
            // seeding them so their consumers ghost-skip identically.
            let mut ghost_carries: std::collections::HashSet<crate::wengert::VarId> =
                Default::default();

            // ── Item 11: double-buffer prefetch calibration + certificate ──
            // Streamed layer groups eligible as a prefetch target.
            let streamed_range_count = (0..ranges.len())
                .filter(|&ri| layer_group[ri].iter().any(|i| ws_streamed.contains(i)))
                .count();
            // WGGO-calibrated activation, PER EDGE (issue during ri, consume
            // at ri+1): overlap pays iff range ri's compute can hide range
            // ri+1's pack transfer. Both sides in μs from the target GpuSpec:
            //   transfer(ri+1) = DMA fixed latency + pack_bytes / PCIe BW
            //   compute_lb(ri) = accum_window × max(launch floor, HBM floor)
            // where the launch floor is ops × kernel_launch_overhead (every
            // adjoint op is ≥ one launch) and the HBM floor is the range's own
            // pack read once per replay. Both compute terms are LOWER bounds
            // (no FLOP term — adjoint shapes are not static), so a discharged
            // edge is calibrated-safe while a declined edge may merely be
            // unproven — the right polarity for a perf heuristic.
            let gpu_spec = crate::gpu_specs::find_gpu(&self.compile_options.target_gpu)
                .unwrap_or_else(crate::gpu_specs::default_gpu);
            let accum_window = grad_accumulation_steps.max(1) as f64;
            let pack_bytes =
                |ri: usize| pending.schedule.range_pack_elems.get(ri).copied().unwrap_or(0) * 4;
            let transfer_us = |ri: usize| {
                WS_PCIE_FIXED_LAT_US
                    + pack_bytes(ri) as f64 / (gpu_spec.pcie_bandwidth_gbps.max(1.0) * 1e3)
            };
            let compute_lb_us = |ri: usize| {
                let launch_floor = (ranges[ri].end - ranges[ri].start) as f64
                    * gpu_spec.kernel_launch_overhead_ns as f64
                    / 1e3;
                let hbm_floor =
                    pack_bytes(ri) as f64 / (gpu_spec.peak_bandwidth_gbs.max(1.0) * 1e3);
                accum_window * launch_floor.max(hbm_floor)
            };
            // Edge ri → ri+1 activates iff ri's compute covers ri+1's
            // transfer (and both ends actually stream). Review M3: a pack
            // containing a symbolic-shape param prices at 0 bytes, which
            // would UNDERSTATE the transfer — the unsafe direction. An
            // unpriced pack instead falls back to the v1 structural
            // heuristic (avg ops/range), the behavior the GPU gates shipped
            // under; a PRICED edge uses the calibrated μs comparison and is
            // calibrated-safe.
            let avg_ops_per_range = pending.adjoint.ops.len() / ranges.len().max(1);
            let edge_on: Vec<bool> = (0..ranges.len())
                .map(|ri| {
                    let next_streams = ri + 1 < ranges.len()
                        && layer_group[ri + 1].iter().any(|i| ws_streamed.contains(i));
                    if !next_streams {
                        return false;
                    }
                    if pack_bytes(ri + 1) > 0 {
                        compute_lb_us(ri) >= transfer_us(ri + 1)
                    } else {
                        avg_ops_per_range >= WS_PREFETCH_MIN_OPS_PER_RANGE
                    }
                })
                .collect();
            let prefetch_active = self.compile_options.stream_prefetch
                && self.compile_options.stream_arena
                && ws_active
                && streamed_range_count >= 2
                && edge_on.iter().any(|&e| e);
            if self.compile_options.stream_prefetch {
                let edges: Vec<String> = (0..ranges.len().saturating_sub(1))
                    .map(|ri| {
                        if pack_bytes(ri + 1) > 0 {
                            format!(
                                "L{ri}->L{}: compute>={:.1}us transfer~{:.1}us {}",
                                ri + 1,
                                compute_lb_us(ri),
                                transfer_us(ri + 1),
                                if edge_on[ri] { "ON" } else { "off" }
                            )
                        } else {
                            format!(
                                "L{ri}->L{}: unpriced pack, ops-heuristic (avg {} vs \
                                 min {}) {}",
                                ri + 1,
                                avg_ops_per_range,
                                WS_PREFETCH_MIN_OPS_PER_RANGE,
                                if edge_on[ri] { "ON" } else { "off" }
                            )
                        }
                    })
                    .collect();
                eprintln!(
                    "[weight-stream] prefetch double-buffer: {} \
                     (streamed_ranges={streamed_range_count}, gpu={}, accum_window={}, \
                     edges [{}])",
                    if prefetch_active {
                        "ACTIVE — prefetch layer L+1 while computing L, event-ordered"
                    } else {
                        "DECLINED (compute too small to hide the transfer; synchronous arena)"
                    },
                    gpu_spec.name,
                    accum_window,
                    edges.join("; "),
                );
            }
            if self.compile_options.stream_async_writeback {
                eprintln!(
                    "[weight-stream] async writeback: {}",
                    if ws_active && streamed_range_count > 0 {
                        "ACTIVE — pack evict DtoH on the transfer stream, mirror \
                         scatter deferred to drain points"
                    } else {
                        "no streamed ranges (no effect)"
                    },
                );
            }
            // CADENCE-style transfer certificate: each entry is a discharged
            // (issue_range, consume_range, pack_size) obligation — a prefetch
            // issued during `issue_range`'s compute and awaited (event) at
            // `consume_range`'s head before any read.
            let mut transfer_cert: Vec<(usize, usize, usize)> = Vec::new();
            // Whether the CURRENT range's weights were prefetched by the
            // previous iteration (→ await instead of a sync upload).
            let mut ri_was_prefetched = false;

            for (ri, range) in ranges.iter().enumerate() {
                // This layer's accumulators exist only from here to its
                // update below — the m_partial surface the schedule shrinks.
                emit_csla_accum_alloc(self, builder, param_list, accum_val, &layer_group[ri])?;
                // D2b: re-upload this layer's weights for its replay range
                // (recompute clones + adjoint reads need them) under the
                // Weights surface bracket. Values dominate the b-loop.
                let ws_range: Vec<i64> = layer_group[ri]
                    .iter()
                    .copied()
                    .filter(|i| ws_streamed.contains(i))
                    .collect();
                if ws_active && !ws_range.is_empty() {
                    let prev_surf =
                        self.compile_call_by_name(builder, "nsl_gpu_get_alloc_surface", &[])?;
                    let wsurf = builder.ins().iconst(cl_types::I8, 1); // SURFACE_WEIGHTS
                    self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[wsurf])?;
                    if self.compile_options.stream_arena {
                        if prefetch_active && ri_was_prefetched {
                            // Item 11: weights already streaming in (prefetched
                            // during the previous range's compute). Just wait
                            // on the transfer event before reading them.
                            self.emit_ws_pack_single(
                                builder,
                                param_list,
                                &ws_range,
                                "nsl_weight_stream_await_pack",
                            )?;
                        } else {
                            // Item 10: this layer group is one contiguous pack.
                            self.emit_ws_pack_upload(builder, param_list, &ws_range)?;
                        }
                    } else {
                        for &idx in &ws_range {
                            let iv = builder.ins().iconst(cl_types::I64, idx);
                            let pw = self
                                .compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
                            self.compile_call_by_name(
                                builder,
                                "nsl_weight_stream_upload",
                                &[pw],
                            )?;
                        }
                    }
                    self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[prev_surf])?;
                }

                // Item 11: prefetch the NEXT streamed layer group so its HtoD
                // overlaps THIS range's compute (the b-loop below). Async on
                // the transfer stream; the next range's head awaits its event.
                // Per-edge: only where the calibration proved THIS range's
                // compute covers the NEXT pack's transfer.
                ri_was_prefetched = false;
                if prefetch_active && ri + 1 < ranges.len() && edge_on[ri] {
                    let ws_next: Vec<i64> = layer_group[ri + 1]
                        .iter()
                        .copied()
                        .filter(|i| ws_streamed.contains(i))
                        .collect();
                    if !ws_next.is_empty() {
                        let prev_surf = self
                            .compile_call_by_name(builder, "nsl_gpu_get_alloc_surface", &[])?;
                        let wsurf = builder.ins().iconst(cl_types::I8, 1); // SURFACE_WEIGHTS
                        self.compile_call_by_name(builder, "nsl_gpu_set_alloc_surface", &[wsurf])?;
                        self.emit_ws_pack_single(
                            builder,
                            param_list,
                            &ws_next,
                            "nsl_weight_stream_prefetch_pack",
                        )?;
                        self.compile_call_by_name(
                            builder,
                            "nsl_gpu_set_alloc_surface",
                            &[prev_surf],
                        )?;
                        transfer_cert.push((ri, ri + 1, ws_next.len()));
                        ri_was_prefetched = true;
                    }
                }

                let slice = crate::wengert::WengertList {
                    ops: pending.adjoint.ops[range.start..range.end].to_vec(),
                    output: pending.adjoint.output,
                    var_names: pending.adjoint.var_names.clone(),
                    var_types: pending.adjoint.var_types.clone(),
                };

                // b-loop over the buffered micro-batches, oldest first.
                let b_var = state.new_variable();
                builder.declare_var(b_var, cl_types::I64);
                let b_zero = builder.ins().iconst(cl_types::I64, 0);
                builder.def_var(b_var, b_zero);
                let b_hdr = builder.create_block();
                let b_body = builder.create_block();
                let b_exit = builder.create_block();
                builder.ins().jump(b_hdr, &[]);
                builder.switch_to_block(b_hdr);
                let b_i = builder.use_var(b_var);
                let b_cont = builder.ins().icmp(IntCC::SignedLessThan, b_i, n_val);
                builder.ins().brif(b_cont, b_body, &[], b_exit, &[]);
                builder.switch_to_block(b_body);
                builder.seal_block(b_body);
                state.current_block = Some(b_body);

                let so_in = builder.use_var(saves_outer_var);
                let b_now = builder.use_var(b_var);
                let inner =
                    self.compile_call_by_name(builder, "nsl_list_get", &[so_in, b_now])?;
                let dict_b = if has_dataloader.is_some() {
                    let dl = builder.use_var(dicts_var);
                    let d = self.compile_call_by_name(builder, "nsl_list_get", &[dl, b_now])?;
                    // Re-install micro-batch b's packing metadata every
                    // range: the registry is thread-local per-batch state
                    // read at @flash_attention launch time, and any range
                    // may contain attention backward ops.
                    self.emit_packing_registry_stash(builder, d)?;
                    Some(d)
                } else {
                    None
                };
                let inner_carry = if let Some(co) = carry_outer {
                    if ri == 0 {
                        // First range: create this micro-batch's carry list,
                        // pre-filled so later nsl_list_set slots are in-bounds.
                        let ic = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
                        let zero = builder.ins().iconst(cl_types::I64, 0);
                        for _ in 0..n_carry {
                            self.compile_call_by_name(builder, "nsl_list_push", &[ic, zero])?;
                        }
                        self.compile_call_by_name(builder, "nsl_list_push", &[co, ic])?;
                        Some(ic)
                    } else {
                        Some(self.compile_call_by_name(builder, "nsl_list_get", &[co, b_now])?)
                    }
                } else {
                    None
                };

                // Seed: forward values (params/constants/loop-invariant SSA)
                // + this range's buffered primal imports + this range's
                // cross-range adjoint imports.
                let mut seed = pending.seed_base.clone();
                for &si in &slot_seed_per_range[ri] {
                    let (vid, kind) = &pending.slots[si];
                    let idx_val = builder.ins().iconst(cl_types::I64, si as i64);
                    let raw =
                        self.compile_call_by_name(builder, "nsl_list_get", &[inner, idx_val])?;
                    let val = match kind {
                        CslaSlotKind::Raw { .. } => raw,
                        CslaSlotKind::F64Bits => {
                            builder.ins().bitcast(cl_types::F64, MemFlags::new(), raw)
                        }
                    };
                    seed.insert(*vid, val);
                }
                for &cv in &imports_per_range[ri] {
                    if ghost_carries.contains(&cv) {
                        continue;
                    }
                    let ic = inner_carry.expect("carry imports imply carry_outer");
                    let slot_val = builder
                        .ins()
                        .iconst(cl_types::I64, carry_slot[&cv] as i64);
                    let val =
                        self.compile_call_by_name(builder, "nsl_list_get", &[ic, slot_val])?;
                    seed.insert(cv, val);
                }

                // LSE tape-carry: re-bind the fused-SDPA aux side-band to
                // this micro-batch's buffered logsumexp for every fwd-out
                // vid seeded in this range — BEFORE the slice lowering emits
                // the SDPA backward that consults the Value-keyed map. The
                // loaded value is the fused forward's real LSE (or its
                // runtime-0 decline sentinel), so the emitted backward takes
                // the same kernel arm the baseline did.
                let seeded_vids_here: std::collections::HashSet<crate::wengert::VarId> =
                    slot_seed_per_range[ri]
                        .iter()
                        .map(|&si| pending.slots[si].0)
                        .collect();
                let mut lse_loaded_here: Vec<(usize, Value)> = Vec::new();
                for (k, (idx, vids)) in pending.lse_slots.iter().enumerate() {
                    let targets: Vec<crate::wengert::VarId> = vids
                        .iter()
                        .copied()
                        .filter(|v| seeded_vids_here.contains(v))
                        .collect();
                    if targets.is_empty() {
                        continue;
                    }
                    let idx_val = builder.ins().iconst(cl_types::I64, *idx as i64);
                    let lse =
                        self.compile_call_by_name(builder, "nsl_list_get", &[inner, idx_val])?;
                    for v in targets {
                        if let Some(&out_seed) = seed.get(&v) {
                            self.flash_attn_aux.insert(out_seed, (out_seed, lse));
                        }
                    }
                    lse_loaded_here.push((k, lse));
                }

                // Fused-CE tape-carry: re-bind `fused_ce_fwd_lse` to this
                // micro-batch's buffered logsumexp, keyed by the SEEDED
                // fwd-result (loss-scalar) Value — BEFORE the slice
                // lowering emits the fused backward whose first extract
                // `.remove()`s the entry (a miss is a hard CodegenError).
                // The emitted backward frees the loaded tensor per (range,
                // b) itself; no per-b slot free exists for these slots.
                // The cast cache is deliberately NOT re-bound: the f32
                // MISS path re-emits pass-through of the extract's own
                // (seeded) inputs — byte-identical, zero extra emission.
                for (idx, vids) in pending.fce_slots.iter() {
                    let targets: Vec<crate::wengert::VarId> = vids
                        .iter()
                        .copied()
                        .filter(|v| seeded_vids_here.contains(v))
                        .collect();
                    if targets.is_empty() {
                        continue;
                    }
                    let idx_val = builder.ins().iconst(cl_types::I64, *idx as i64);
                    let lse =
                        self.compile_call_by_name(builder, "nsl_list_get", &[inner, idx_val])?;
                    for v in targets {
                        if let Some(&res_seed) = seed.get(&v) {
                            self.fused_ce_fwd_lse.insert(res_seed, lse);
                        }
                    }
                }

                // FASE hook: identical to the baseline's fase_cb — accumulate
                // each parameter gradient into its compile-time accum_list
                // slot (allocated at window start or at this range's head)
                // and free it immediately. wrap_offload is const false BY
                // DESIGN (review D2a-2): the per-layer accumulators are
                // device-resident — that is D1's point — and the offload
                // envelope applies only at the update sites, where one
                // layer's m/v stage through at a time.
                let hook_idx_map = &pending.hook_accum_idx;
                let accum_scale = pending.accum_scale;
                let mut fase_cb = |c: &mut Compiler,
                                   var_id: crate::wengert::VarId,
                                   grad_ptr: Value,
                                   still_needed: bool,
                                   b: &mut cranelift_frontend::FunctionBuilder|
                 -> Result<(), CodegenError> {
                    let Some(&accum_idx) = hook_idx_map.get(&var_id) else {
                        return Ok(());
                    };
                    let idx_val =
                        b.ins().iconst(cranelift_codegen::ir::types::I64, accum_idx);
                    let m_partial =
                        c.compile_call_by_name(b, "nsl_list_get", &[accum_val, idx_val])?;
                    c.fase_emit_accumulate(b, m_partial, grad_ptr, accum_scale, false)?;
                    // Defer the free when this param grad is a shared
                    // intermediate a later in-slice op still reads (bias-grad
                    // == d_out feeding the weight matmul). wengert_lower keeps
                    // it in var_map + owned_values; end-of-range cleanup frees
                    // it. Exports are distinct activation adjoints, never param
                    // grads, so this never strands a cross-range carry.
                    if !still_needed {
                        c.compile_call_by_name(b, "nsl_tensor_free", &[grad_ptr])?;
                    }
                    Ok(())
                };
                let grad_lowered = match crate::wengert_lower::compile_wengert_ops(
                    self,
                    builder,
                    state,
                    &slice,
                    &seed,
                    Some((&pending.param_adj_set, &mut fase_cb)),
                ) {
                    Ok(gv) => gv,
                    Err(e) => {
                        eprintln!(
                            "[nsl] csla window backward lowering failed (range {ri}: {}), \
                             rerun without --layerwise-accum",
                            e
                        );
                        return Err(e);
                    }
                };

                // Exports: store this range's boundary adjoints into the
                // carry list for later ranges (ghost-skips recorded so
                // consumers ghost-skip identically to the baseline).
                for &ev in &exports_per_range[ri] {
                    match grad_lowered.var_map.get(&ev) {
                        Some(&val) => {
                            // Review D1b-4: only i64-typed values (tensor /
                            // list pointers, integers) can ride the carry
                            // list; a scalar f64 crossing a range boundary
                            // would otherwise be a Cranelift verifier ICE.
                            // AD rules emit constants adjacent to consumers,
                            // so this is unreachable today — refuse loudly
                            // if that ever changes.
                            let vty = builder.func.dfg.value_type(val);
                            if vty != cl_types::I64 {
                                return Err(CodegenError::new(format!(
                                    "--layerwise-accum: cross-range adjoint value \
                                     VarId {ev} has non-i64 Cranelift type {vty}; \
                                     scalar carries are unsupported — drop \
                                     --layerwise-accum",
                                )));
                            }
                            let ic = inner_carry.expect("exports imply carry_outer");
                            let slot_val = builder
                                .ins()
                                .iconst(cl_types::I64, carry_slot[&ev] as i64);
                            self.compile_call_by_name(
                                builder,
                                "nsl_list_set",
                                &[ic, slot_val, val],
                            )?;
                        }
                        None => {
                            ghost_carries.insert(ev);
                        }
                    }
                }

                // Per-replay cleanup: adjoint-owned intermediates minus
                // hook-consumed gradients, explicit FreeTensor victims, and
                // the exports (they outlive this range; their frees are the
                // consuming ranges' markers or the explicit list below).
                //
                // Seed from exactly the param grads the hook FREED — not the
                // whole param_adj_set. A param grad that is a shared
                // intermediate (bias-grad == d_out, read later in-slice) was
                // accumulated but its free was DEFERRED; it is absent here on
                // purpose so free_wengert_owned_values releases it once.
                let mut freed_adjoint_vars: std::collections::HashSet<crate::wengert::VarId> =
                    grad_lowered.hook_freed_param_vars.iter().copied().collect();
                freed_adjoint_vars.extend(grad_lowered.hook_freed_input_vars.iter().copied());
                freed_adjoint_vars.extend(grad_lowered.explicit_freed_vars.iter().copied());
                freed_adjoint_vars.extend(exports_per_range[ri].iter().copied());
                self.free_wengert_owned_values(
                    builder,
                    &grad_lowered.owned_values,
                    &freed_adjoint_vars,
                )?;

                // Carried values whose last consumer is this range and that
                // no FreeTensor marker covers: free their seeded value now.
                // Review D1b-1 (HIGH): skip anything the replay itself
                // already freed — the hook's reduce_to_shape extra free
                // (hook_freed_input_vars) releases a carried RAW grad when a
                // range boundary lands on the reduce op, and explicit
                // FreeTensor victims are covered by their markers. Freeing
                // either again here would be a double free.
                for &cv in &carry_explicit_free[ri] {
                    if ghost_carries.contains(&cv)
                        || grad_lowered.hook_freed_input_vars.contains(&cv)
                        || grad_lowered.explicit_freed_vars.contains(&cv)
                    {
                        continue;
                    }
                    if let Some(&val) = seed.get(&cv) {
                        self.compile_call_by_name(builder, "nsl_tensor_free", &[val])?;
                    }
                }

                // Free the buffered primal slots whose LAST reader is this
                // range (owned tensors/lists only). The loss slot skips the
                // window's LAST entry: that is the CURRENT iteration's loss,
                // still read by on_step / on_epoch after this phase; the
                // conditional per-iteration loss-free site below owns it.
                let n_minus_1 =
                    builder.ins().iconst(cl_types::I64, grad_accumulation_steps - 1);
                for &si in &slot_free_per_range[ri] {
                    let (_vid, kind) = &pending.slots[si];
                    let CslaSlotKind::Raw { owned: Some(ty) } = kind else {
                        continue;
                    };
                    let free_fn = match ty {
                        crate::wengert::WengertType::Tensor => "nsl_tensor_free",
                        crate::wengert::WengertType::List => "nsl_list_free",
                        _ => continue,
                    };
                    let idx_val = builder.ins().iconst(cl_types::I64, si as i64);
                    let slot_val =
                        self.compile_call_by_name(builder, "nsl_list_get", &[inner, idx_val])?;
                    if Some(si) == pending.loss_slot {
                        let b_cur = builder.use_var(b_var);
                        let is_last = builder.ins().icmp(IntCC::Equal, b_cur, n_minus_1);
                        let loss_free = builder.create_block();
                        let loss_join = builder.create_block();
                        builder.ins().brif(is_last, loss_join, &[], loss_free, &[]);
                        builder.switch_to_block(loss_free);
                        builder.seal_block(loss_free);
                        self.compile_call_by_name(builder, free_fn, &[slot_val])?;
                        builder.ins().jump(loss_join, &[]);
                        builder.switch_to_block(loss_join);
                        builder.seal_block(loss_join);
                        state.current_block = Some(loss_join);
                    } else {
                        self.compile_call_by_name(builder, free_fn, &[slot_val])?;
                    }
                }

                // LSE tape-carry: free this micro-batch's logsumexps whose
                // last consuming range is this one (null-safe — the decline
                // sentinel is a runtime 0). Re-load from the inner list when
                // this range didn't seed any of the slot's vids (review F4:
                // the fixpoint-extended free range can trail the direct
                // reads through list membership — mirror the regular slot
                // frees' unconditional re-load instead of leaking).
                for (k, (idx, _vids)) in pending.lse_slots.iter().enumerate() {
                    if lse_free_range[k] != ri {
                        continue;
                    }
                    let lse = match lse_loaded_here.iter().find(|(lk, _)| *lk == k) {
                        Some((_, v)) => *v,
                        None => {
                            let idx_val = builder.ins().iconst(cl_types::I64, *idx as i64);
                            self.compile_call_by_name(
                                builder,
                                "nsl_list_get",
                                &[inner, idx_val],
                            )?
                        }
                    };
                    self.compile_call_by_name(builder, "nsl_tensor_free_if_valid", &[lse])?;
                }

                if ri == last_ri {
                    // Batch dict values die with their micro-batch's LAST
                    // replay (the per-iteration dict free is suppressed
                    // under csla), then the shells. INVARIANT (review L2):
                    // on step iterations this destroys the CURRENT batch
                    // dict at b==N-1, i.e. BEFORE the optimizer updates and
                    // the on_step callback — nothing after this point may
                    // read step_param_var's dict (on_step receives only
                    // (step, loss)). nsl_dict_free_tensor_values destroys
                    // the WHOLE dict structure, matching the baseline's
                    // per-iteration call — popped dicts are never touched by
                    // the DataLoader teardown.
                    if let Some(d) = dict_b {
                        self.compile_call_by_name(
                            builder,
                            "nsl_dict_free_tensor_values",
                            &[d],
                        )?;
                    }
                    self.compile_call_by_name(builder, "nsl_list_free", &[inner])?;
                    if let Some(ic) = inner_carry {
                        self.compile_call_by_name(builder, "nsl_list_free", &[ic])?;
                    }
                }

                let b_cur = builder.use_var(b_var);
                let b_one = builder.ins().iconst(cl_types::I64, 1);
                let b_next = builder.ins().iadd(b_cur, b_one);
                builder.def_var(b_var, b_next);
                builder.ins().jump(b_hdr, &[]);
                builder.seal_block(b_hdr);
                builder.switch_to_block(b_exit);
                builder.seal_block(b_exit);
                state.current_block = Some(b_exit);

                // Per-layer update: this layer's accumulators are complete
                // (all N micro-batches replayed) and — by the CrossLayer
                // classification + the list-membership fixpoint — no later
                // range reads these params' OLD θ. Update, then free the
                // layer's accumulators.
                emit_csla_group_update(
                    self,
                    builder,
                    param_list,
                    state_list_1,
                    state_list_2,
                    two_state,
                    accum_val,
                    &fase_plan.recipe,
                    (bc1_inv, bc2_inv),
                    wrap_precision,
                    self.compile_options.optim_state_offload,
                    &layer_group[ri],
                )?;
                // D2b: this layer's θ is final for the window — write back
                // to the mirror and drop the device buffer.
                if ws_active {
                    if self.compile_options.stream_async_writeback
                        && self.compile_options.stream_arena
                    {
                        // Item 11 (writeback half): issue the pack's DtoH on
                        // the transfer stream and move on — the next range's
                        // compute overlaps this layer's writeback. The mirror
                        // scatter lands at the runtime's drain points (queue
                        // cap / affected re-upload / teardown).
                        self.emit_ws_pack_single(
                            builder,
                            param_list,
                            &ws_range,
                            "nsl_weight_stream_evict_pack_async",
                        )?;
                    } else if self.compile_options.stream_arena {
                        // Item 10: one DtoH writeback for the whole layer pack.
                        self.emit_ws_pack_evict(builder, param_list, &ws_range, 1)?;
                    } else {
                        for &idx in &ws_range {
                            let iv = builder.ins().iconst(cl_types::I64, idx);
                            let pw = self
                                .compile_call_by_name(builder, "nsl_list_get", &[param_list, iv])?;
                            let wb = builder.ins().iconst(cl_types::I64, 1);
                            self.compile_call_by_name(
                                builder,
                                "nsl_weight_stream_evict",
                                &[pw, wb],
                            )?;
                        }
                    }
                }
            }

            // Item 11: emit the discharged transfer certificate. Each prefetch
            // issued during range Li's compute is provably awaited (a CUDA
            // event on the compute stream) at range Li+1's head before any
            // read — the CADENCE assume/guarantee obligation, discharged.
            if prefetch_active {
                let total: usize = transfer_cert.iter().map(|(_, _, n)| n).sum();
                eprintln!(
                    "[weight-stream] transfer certificate: {} prefetch obligations discharged \
                     ({total} params double-buffered); chain [{}]",
                    transfer_cert.len(),
                    transfer_cert
                        .iter()
                        .map(|(i, c, n)| format!("L{i}->L{c}:{n}"))
                        .collect::<Vec<_>>()
                        .join(" "),
                );
            }

            // Epilogue: globals (embedding / final norm / LM head), tied and
            // cross-layer params, dead layers, and anything the extractor
            // never saw — after the whole backward, like the baseline.
            emit_csla_group_update(
                self,
                builder,
                param_list,
                state_list_1,
                state_list_2,
                two_state,
                accum_val,
                &fase_plan.recipe,
                (bc1_inv, bc2_inv),
                wrap_precision,
                self.compile_options.optim_state_offload,
                global_group,
            )?;
            // D2b part 2: NO post-epilogue restore. The next iterations'
            // forwards re-upload each layer right before its own segment
            // (and evict it after its last primal touch), so streamed
            // params stay off-device between their brackets for the WHOLE
            // training loop — the forward-side residency wall this part
            // removes. Teardown still restores for model_save/eval.

            // Window cleanup: drop the shells and start fresh lists for the
            // next window (carry inner shells died with the last range).
            if let Some(co) = carry_outer {
                self.compile_call_by_name(builder, "nsl_list_free", &[co])?;
            }
            let so_done = builder.use_var(saves_outer_var);
            self.compile_call_by_name(builder, "nsl_list_free", &[so_done])?;
            let so_new = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            builder.def_var(saves_outer_var, so_new);
            let dl_done = builder.use_var(dicts_var);
            self.compile_call_by_name(builder, "nsl_list_free", &[dl_done])?;
            let dl_new = self.compile_call_by_name(builder, "nsl_list_new", &[])?;
            builder.def_var(dicts_var, dl_new);

            builder.ins().jump(bwd_join, &[]);
            builder.switch_to_block(bwd_join);
            builder.seal_block(bwd_join);
            state.current_block = Some(bwd_join);
        }

        // 7e4. Gradient accumulation gate: only step optimizer every N batches
        let optimizer_block = builder.create_block();
        let post_optimizer_block = builder.create_block();

        if grad_accumulation_steps > 1 {
            // Reuse the already-computed should_step value (defined above).
            let should_step = builder.use_var(should_step_var);
            builder
                .ins()
                .brif(should_step, optimizer_block, &[], post_optimizer_block, &[]);

            builder.switch_to_block(optimizer_block);
            builder.seal_block(optimizer_block);
            state.current_block = Some(optimizer_block);
        } else {
            // No accumulation — always step
            builder.ins().jump(optimizer_block, &[]);
            builder.switch_to_block(optimizer_block);
            builder.seal_block(optimizer_block);
            state.current_block = Some(optimizer_block);
        }

        // NSL_PHASE_TIMING: optimizer-phase start (fires only on accumulation
        // boundaries — this block is skipped on non-step micro-batches). The
        // matching report is emitted before each of the three
        // post_optimizer_block jumps below.
        let phase_timing_opt =
            std::env::var("NSL_PHASE_TIMING").ok().as_deref() == Some("1");
        let phase_opt_t0 = if phase_timing_opt {
            self.compile_call_by_name(builder, "nsl_cuda_device_synchronize", &[])?;
            Some(self.compile_call_by_name(builder, "nsl_clock", &[])?)
        } else {
            None
        };

        // 7f. Optimizer step: for each param, call optimizer step function
        // FASE Deferred: emit fused per-parameter step; otherwise use existing path.
        //
        // FASE Codegen Phase 2: Optimizer step dispatch is at outer scope here
        // (the `if fase_deferred { Deferred loops } else { per-optimizer stdlib
        // step }` shape can't be split per-param without hoisting a runtime
        // branch through two structurally different optimizer-step emission
        // paths — the Deferred side uses `fase_emit_final_step` on m_partial,
        // the FullBuffer side dispatches to optimizer-specific stdlib functions
        // (adam_step/sgd_step/...) with different calling conventions.
        //
        // Deferring per-param optimizer dispatch to a follow-up plan. WGGO's
        // per-layer signal still influences accumulation behavior via the
        // mode-table dispatch in ga_body (Task 4b), which is the dominant
        // memory-cost decision. The optimizer step's compute cost is identical
        // between modes — only accumulation buffer shape differs.
        //
        // FASE Optim-Step Dispatch Task 5: shared locals hoisted from the
        // fallback `else` arm so the new `if let Some(mtb)` branch can
        // reference them. Cranelift DCE ensures constants unused by the
        // Deferred arm don't affect emitted code for that path.
        let opt_grads = if let Some(accum) = accum_list {
            accum
        } else {
            grads_list
        };

        // H.2: mangling convention is `{module_prefix}__{fn_name}` where
        // `module_prefix` is the dotted stdlib path with `.` -> `_`
        // (single underscore per path separator). So `nsl.optim.sgd` ->
        // `nsl_optim_sgd`, and the step fn becomes
        // `nsl_optim_sgd__sgd_step`. Pre-H.2 this site emitted
        // `nsl__optim__sgd__sgd_step` (double underscores between every
        // path part) which did not match `stdlib_loader::module_prefix_for`
        // output — `compile_entry` failed with "undefined function".
        let optimizer_fn_name = match optimizer_name.as_str() {
            "sgd" => "nsl_optim_sgd__sgd_step",
            "adam" => "nsl_optim_adam__adam_step",
            "adamw" => "nsl_optim_adamw__adamw_step",
            "lion" => "nsl_optim_lion__lion_step",
            "muon" => "nsl_optim_muon__muon_step",
            "soap" => "nsl_optim_soap__soap_step",
            _ => {
                return Err(CodegenError::new(format!(
                    "unsupported optimizer '{}' in train block",
                    optimizer_name
                )));
            }
        };

        // Check if optimizer function exists, try fallback name patterns
        let opt_fn = if self.registry.functions.contains_key(optimizer_fn_name) {
            optimizer_fn_name.to_string()
        } else {
            // Try simpler name: e.g. "sgd_step"
            let simple = format!("{}_step", optimizer_name);
            if self.registry.functions.contains_key(&simple) {
                simple
            } else if self.registry.runtime_fns.contains_key(optimizer_fn_name) {
                optimizer_fn_name.to_string()
            } else if self.registry.runtime_fns.contains_key(&simple) {
                simple
            } else {
                // Register as runtime function so it can be resolved at link time
                optimizer_fn_name.to_string()
            }
        };

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

        // M43b: ZeRO Stage 1+ — all-reduce gradients before optimizer step.
        // The rc is asserted: a mid-run collective failure (capacity -4,
        // GPU-placement refusal -5, dtype -1) must abort, not continue with
        // un-reduced gradients (review: discarded rc -> silent wrong train).
        let zero_enabled = self.features.zero_stage.filter(|&s| s >= 1).is_some();
        if zero_enabled {
            let rc = self.compile_call_by_name(
                builder,
                "nsl_zero_reduce_grads",
                &[opt_grads, num_params_val],
            )?;
            let z = builder.ins().iconst(cl_types::I64, 0);
            let ok = builder.ins().icmp(IntCC::Equal, rc, z);
            let m = "nsl: ZeRO gradient reduction failed (see message above) — aborting";
            self.intern_string(m)?;
            let mp = self.compile_string_literal(builder, m)?;
            self.compile_call_by_name(builder, "nsl_assert", &[ok, mp])?;
        }

        if let Some(mtb) = mode_table_base {
            // D3 v1: the unified mode-table dispatch has its own per-param
            // update kernel selection — owner-gating inside it is untested
            // territory. Refuse the composition loudly.
            if zero_enabled {
                return Err(CodegenError::new(
                    "--zero-stage is not supported with a WGGO per-param mode \
                     table yet: the sharded update gate is lowered only for \
                     the monolithic Deferred and FullBuffer optimizer arms. \
                     Drop --wggo mode overrides or --zero-stage",
                ));
            }
            self.emit_unified_optim_step_dispatch(
                builder,
                state,
                mtb,
                num_params_val,
                param_list,
                state_list_1,
                state_list_2,
                num_state_buffers,
                accum_list,
                opt_grads,
                fase_hook_active,
                step_count_var,
                &fase_plan,
                optimizer_name.as_str(),
                &opt_fn,
                lr,
                momentum_const,
                dampening_const,
                weight_decay_const,
                nesterov_const,
                beta1_const,
                beta2_const,
                eps_const,
                grad_accumulation_steps,
                grad_clip,
                cpdt_precision_dtypes,
            )?;

            // M43b: ZeRO Stage 1+ — all-gather updated params after optimizer step
            if zero_enabled {
                self.compile_call_by_name(builder, "nsl_zero_step", &[])?;
            }

            // Jump to post-optimizer block (merges optimizer and skip paths)
            if let Some(t0) = phase_opt_t0 {
                self.compile_call_by_name(builder, "nsl_cuda_device_synchronize", &[])?;
                let t3 = self.compile_call_by_name(builder, "nsl_clock", &[])?;
                let opt = builder.ins().fsub(t3, t0);
                self.compile_call_by_name(builder, "nsl_phase_optim_report", &[opt])?;
            }
            builder.ins().jump(post_optimizer_block, &[]);
            builder.switch_to_block(post_optimizer_block);
            builder.seal_block(post_optimizer_block);
            state.current_block = Some(post_optimizer_block);
        } else if fase_deferred {
            // CSLA (D1b): every parameter was already updated inside the
            // window backward region (per-layer groups + the epilogue group,
            // with the same bias-correction pair this site would compute) —
            // the monolithic loop below must not run on the NULL accumulator
            // slots. Emit only the pass-through to post_optimizer_block.
            if csla_active {
                if let Some(t0) = phase_opt_t0 {
                    self.compile_call_by_name(builder, "nsl_cuda_device_synchronize", &[])?;
                    let t3 = self.compile_call_by_name(builder, "nsl_clock", &[])?;
                    let opt = builder.ins().fsub(t3, t0);
                    self.compile_call_by_name(builder, "nsl_phase_optim_report", &[opt])?;
                }
                builder.ins().jump(post_optimizer_block, &[]);
                builder.switch_to_block(post_optimizer_block);
                builder.seal_block(post_optimizer_block);
                state.current_block = Some(post_optimizer_block);
            } else if let Some(accum) = accum_list {
                // ── FASE Deferred: compute bias-correction scalars once per step ──
                // opt_step = (step_count + 1) / grad_accumulation_steps
                // bc_inv = nsl_bias_correction_inv(β, opt_step)
                let sc_val = builder.use_var(step_count_var);
                let one_i64 = builder.ins().iconst(cl_types::I64, 1);
                let sc_plus_one = builder.ins().iadd(sc_val, one_i64);
                let grad_accum_const = builder.ins().iconst(cl_types::I64, grad_accumulation_steps);
                let opt_step = builder.ins().sdiv(sc_plus_one, grad_accum_const);

                let beta1_const = builder.ins().f64const(fase_plan.recipe.beta1);
                let beta2_const = builder.ins().f64const(fase_plan.recipe.beta2);
                let bc1_inv = self.compile_call_by_name(
                    builder,
                    "nsl_bias_correction_inv",
                    &[beta1_const, opt_step],
                )?;
                let bc2_inv = self.compile_call_by_name(
                    builder,
                    "nsl_bias_correction_inv",
                    &[beta2_const, opt_step],
                )?;

                if fase_plan.two_phase_clip {
                    // ── Phase A: fused accumulation + sum_sq loop ──
                    // Accumulate the final micro-batch's gradients into m_partial
                    // (the standard accumulation loop was skipped for this batch),
                    // and simultaneously accumulate sum(||g_i||^2) for the global
                    // L2 norm.
                    let pa_tot_var = state.new_variable();
                    builder.declare_var(pa_tot_var, cl_types::F64);
                    let pa_zero_f = builder.ins().f64const(0.0);
                    builder.def_var(pa_tot_var, pa_zero_f);

                    let pa_i_var = state.new_variable();
                    builder.declare_var(pa_i_var, cl_types::I64);
                    let pa_i_zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(pa_i_var, pa_i_zero);

                    let pa_hdr = builder.create_block();
                    let pa_body = builder.create_block();
                    let pa_exit = builder.create_block();
                    builder.ins().jump(pa_hdr, &[]);
                    builder.switch_to_block(pa_hdr);
                    let pa_i = builder.use_var(pa_i_var);
                    let pa_cont =
                        builder.ins().icmp(IntCC::SignedLessThan, pa_i, num_params_val);
                    builder.ins().brif(pa_cont, pa_body, &[], pa_exit, &[]);
                    builder.switch_to_block(pa_body);
                    builder.seal_block(pa_body);

                    let pa_mpart =
                        self.compile_call_by_name(builder, "nsl_list_get", &[accum, pa_i])?;
                    // Monolithic Phase A: this branch only runs when NO mode
                    // table exists (no WGGO overrides), so every param is
                    // globally Deferred — no mode dispatch needed. Mixed
                    // tables take the emit_unified_optim_step_dispatch path,
                    // whose Phase A mirrors this fused accumulate.
                    //
                    // When FASE hook is active, accumulation already happened
                    // during adjoint lowering — skip the grads_list read + accumulate.
                    if !fase_hook_active {
                        let pa_grad = self.compile_call_by_name(
                            builder,
                            "nsl_list_get",
                            &[grads_list, pa_i],
                        )?;
                        let off = self.compile_options.optim_state_offload;
                        self.fase_emit_accumulate(
                            builder,
                            pa_mpart,
                            pa_grad,
                            fase_plan.recipe.accum_scale,
                            off,
                        )?;
                        self.compile_call_by_name(builder, "nsl_tensor_free", &[pa_grad])?;
                    }
                    let pa_sq = self.compile_call_by_name(
                        builder,
                        "nsl_tensor_sum_sq",
                        &[pa_mpart],
                    )?;
                    let pa_tot_cur = builder.use_var(pa_tot_var);
                    let pa_tot_new = builder.ins().fadd(pa_tot_cur, pa_sq);
                    builder.def_var(pa_tot_var, pa_tot_new);
                    let pa_i_next = builder.ins().iadd_imm(pa_i, 1);
                    builder.def_var(pa_i_var, pa_i_next);
                    builder.ins().jump(pa_hdr, &[]);

                    builder.switch_to_block(pa_exit);
                    builder.seal_block(pa_hdr);
                    builder.seal_block(pa_exit);

                    // Free grads_list wrapper — skip when hook active (null sentinel).
                    if !fase_hook_active {
                        self.compile_call_by_name(builder, "nsl_list_free", &[grads_list])?;
                    }

                    // ── Scalar: clip_factor = min(1, grad_clip / (sqrt(total_sq) + 1e-6)) ──
                    // `grad_clip` is the raw f64 from the train-block config
                    // (f64::MAX when not set, but two_phase_clip is only true when it IS set).
                    let grad_clip_threshold = grad_clip;
                    let total_sq = builder.use_var(pa_tot_var);
                    let norm = builder.ins().sqrt(total_sq);
                    let eps_v = builder.ins().f64const(1e-6_f64);
                    let denom = builder.ins().fadd(norm, eps_v);
                    let tau_v = builder.ins().f64const(grad_clip_threshold);
                    let ratio = builder.ins().fdiv(tau_v, denom);
                    let one_f = builder.ins().f64const(1.0_f64);
                    let clip_factor = builder.ins().fmin(one_f, ratio);

                    // ── Phase B: scale m_partial in place, then fused optimizer step ──
                    let pb_i_var = state.new_variable();
                    builder.declare_var(pb_i_var, cl_types::I64);
                    let pb_i_zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(pb_i_var, pb_i_zero);

                    let pb_hdr = builder.create_block();
                    let pb_body = builder.create_block();
                    let pb_exit = builder.create_block();
                    builder.ins().jump(pb_hdr, &[]);
                    builder.switch_to_block(pb_hdr);
                    let pb_i = builder.use_var(pb_i_var);
                    let pb_cont =
                        builder.ins().icmp(IntCC::SignedLessThan, pb_i, num_params_val);
                    builder.ins().brif(pb_cont, pb_body, &[], pb_exit, &[]);
                    builder.switch_to_block(pb_body);
                    builder.seal_block(pb_body);

                    let pb_mpart =
                        self.compile_call_by_name(builder, "nsl_list_get", &[accum, pb_i])?;
                    // D3 (ZeRO-1): only the owner rank updates this param.
                    // Non-owners must still ZERO m_partial (the update
                    // normally does it) or the next window accumulates onto
                    // stale gradients.
                    let pb_zero_blocks = if zero_enabled {
                        let owns = self
                            .compile_call_by_name(builder, "nsl_zero_owns_param", &[pb_i])?;
                        let one_i = builder.ins().iconst(cl_types::I64, 1);
                        let owned = builder.ins().icmp(IntCC::Equal, owns, one_i);
                        let pb_do = builder.create_block();
                        let pb_skip = builder.create_block();
                        let pb_join = builder.create_block();
                        builder.ins().brif(owned, pb_do, &[], pb_skip, &[]);
                        builder.switch_to_block(pb_skip);
                        builder.seal_block(pb_skip);
                        // L8 hygiene: a true zero-fill, NOT mul-by-0.0 — if
                        // a diverging window left Inf/NaN in m_partial,
                        // x*0.0 is NaN and the buffer never recovers.
                        self.compile_call_by_name(
                            builder,
                            "nsl_tensor_zero_inplace",
                            &[pb_mpart],
                        )?;
                        builder.ins().jump(pb_join, &[]);
                        builder.switch_to_block(pb_do);
                        builder.seal_block(pb_do);
                        Some(pb_join)
                    } else {
                        None
                    };
                    self.compile_call_by_name(
                        builder,
                        "nsl_tensor_mul_scalar_inplace",
                        &[pb_mpart, clip_factor],
                    )?;
                    let pb_theta = self.compile_call_by_name(
                        builder,
                        "nsl_list_get",
                        &[param_list, pb_i],
                    )?;
                    let pb_m = self.compile_call_by_name(
                        builder,
                        "nsl_list_get",
                        &[state_list_1, pb_i],
                    )?;
                    let pb_v = if num_state_buffers >= 2 {
                        self.compile_call_by_name(
                            builder,
                            "nsl_list_get",
                            &[state_list_2, pb_i],
                        )?
                    } else {
                        pb_m
                    };
                    let wrap_precision = cpdt_precision_dtypes.is_some();
                    self.fase_emit_final_step(
                        builder,
                        pb_theta,
                        pb_m,
                        pb_mpart,
                        pb_v,
                        &fase_plan.recipe,
                        Some((bc1_inv, bc2_inv)),
                        wrap_precision,
                        self.compile_options.optim_state_offload,
                    )?;
                    if let Some(pb_join) = pb_zero_blocks {
                        builder.ins().jump(pb_join, &[]);
                        builder.switch_to_block(pb_join);
                        builder.seal_block(pb_join);
                    }
                    let pb_i_next = builder.ins().iadd_imm(pb_i, 1);
                    builder.def_var(pb_i_var, pb_i_next);
                    builder.ins().jump(pb_hdr, &[]);

                    builder.switch_to_block(pb_exit);
                    builder.seal_block(pb_hdr);
                    builder.seal_block(pb_exit);
                    state.current_block = Some(pb_exit);
                    // Offload P0.2: one drain per optimizer step (transfer-
                    // stream sync + deferred frees of the staged tensors).
                    if self.compile_options.optim_state_offload {
                        self.compile_call_by_name(builder, "nsl_offload_drain", &[])?;
                    }
                } else {
                    // ── Non-clip Deferred path: per-parameter fused final step ──
                    // accum_list is m_partial.  state_list_1 = m, state_list_2 = v.
                    let fs_i_var = state.new_variable();
                    builder.declare_var(fs_i_var, cl_types::I64);
                    let fs_zero = builder.ins().iconst(cl_types::I64, 0);
                    builder.def_var(fs_i_var, fs_zero);
                    let fs_hdr = builder.create_block();
                    let fs_body = builder.create_block();
                    let fs_exit = builder.create_block();
                    builder.ins().jump(fs_hdr, &[]);
                    builder.switch_to_block(fs_hdr);
                    let fs_i = builder.use_var(fs_i_var);
                    let fs_cont =
                        builder.ins().icmp(IntCC::SignedLessThan, fs_i, num_params_val);
                    builder.ins().brif(fs_cont, fs_body, &[], fs_exit, &[]);
                    builder.switch_to_block(fs_body);
                    builder.seal_block(fs_body);
                    // D3 (ZeRO-1): owner-gated update; non-owners zero
                    // m_partial (see the clip-path comment).
                    let fs_zero_blocks = if zero_enabled {
                        let mp = self
                            .compile_call_by_name(builder, "nsl_list_get", &[accum, fs_i])?;
                        let owns = self
                            .compile_call_by_name(builder, "nsl_zero_owns_param", &[fs_i])?;
                        let one_i = builder.ins().iconst(cl_types::I64, 1);
                        let owned = builder.ins().icmp(IntCC::Equal, owns, one_i);
                        let fs_do = builder.create_block();
                        let fs_skip = builder.create_block();
                        let fs_join = builder.create_block();
                        builder.ins().brif(owned, fs_do, &[], fs_skip, &[]);
                        builder.switch_to_block(fs_skip);
                        builder.seal_block(fs_skip);
                        // L8 hygiene: true zero-fill (see the clip-path
                        // comment — mul-by-0.0 keeps NaN/Inf alive).
                        self.compile_call_by_name(
                            builder,
                            "nsl_tensor_zero_inplace",
                            &[mp],
                        )?;
                        builder.ins().jump(fs_join, &[]);
                        builder.switch_to_block(fs_do);
                        builder.seal_block(fs_do);
                        Some(fs_join)
                    } else {
                        None
                    };
                    let theta =
                        self.compile_call_by_name(builder, "nsl_list_get", &[param_list, fs_i])?;
                    let m =
                        self.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, fs_i])?;
                    let m_partial =
                        self.compile_call_by_name(builder, "nsl_list_get", &[accum, fs_i])?;
                    let v = if num_state_buffers >= 2 {
                        self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, fs_i])?
                    } else {
                        // SGD has no v state — pass m as a placeholder (not used by SgdUpdate recipe)
                        m
                    };
                    let wrap_precision = cpdt_precision_dtypes.is_some();
                    self.fase_emit_final_step(
                        builder,
                        theta,
                        m,
                        m_partial,
                        v,
                        &fase_plan.recipe,
                        Some((bc1_inv, bc2_inv)),
                        wrap_precision,
                        self.compile_options.optim_state_offload,
                    )?;
                    // fase_emit_final_step zeroed m_partial already — no Site E needed.
                    if let Some(fs_join) = fs_zero_blocks {
                        builder.ins().jump(fs_join, &[]);
                        builder.switch_to_block(fs_join);
                        builder.seal_block(fs_join);
                    }
                    let fs_one = builder.ins().iconst(cl_types::I64, 1);
                    let fs_next = builder.ins().iadd(fs_i, fs_one);
                    builder.def_var(fs_i_var, fs_next);
                    builder.ins().jump(fs_hdr, &[]);
                    builder.seal_block(fs_hdr);
                    builder.switch_to_block(fs_exit);
                    builder.seal_block(fs_exit);
                    state.current_block = Some(fs_exit);
                    // Offload P0.2: one drain per optimizer step (transfer-
                    // stream sync + deferred frees of the staged tensors).
                    if self.compile_options.optim_state_offload {
                        self.compile_call_by_name(builder, "nsl_offload_drain", &[])?;
                    }
                }

                // D3 (ZeRO-1): broadcast every param from its owner so
                // all ranks hold the full updated model — the post-step
                // sync both Deferred sub-arms converge on. Fixes the
                // pre-existing gap where this arm emitted no post-step
                // ZeRO call at all.
                if zero_enabled {
                    let rc = self.compile_call_by_name(
                        builder,
                        "nsl_zero_sync_params",
                        &[param_list, num_params_val],
                    )?;
                    let z = builder.ins().iconst(cl_types::I64, 0);
                    let ok = builder.ins().icmp(IntCC::Equal, rc, z);
                    let m = "nsl: ZeRO param sync failed (see message above) — aborting";
                    self.intern_string(m)?;
                    let mp = self.compile_string_literal(builder, m)?;
                    self.compile_call_by_name(builder, "nsl_assert", &[ok, mp])?;
                }
                // Jump to post-optimizer block (merges optimizer and skip paths)
                if let Some(t0) = phase_opt_t0 {
                self.compile_call_by_name(builder, "nsl_cuda_device_synchronize", &[])?;
                let t3 = self.compile_call_by_name(builder, "nsl_clock", &[])?;
                let opt = builder.ins().fsub(t3, t0);
                self.compile_call_by_name(builder, "nsl_phase_optim_report", &[opt])?;
            }
            builder.ins().jump(post_optimizer_block, &[]);
                builder.switch_to_block(post_optimizer_block);
                builder.seal_block(post_optimizer_block);
                state.current_block = Some(post_optimizer_block);
            }
        } else {

        // 7f. Optimizer step loop: for i in 0..num_params (runtime loop)
        {
            let opt_i_var = state.new_variable();
            builder.declare_var(opt_i_var, cl_types::I64);
            let opt_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(opt_i_var, opt_zero);

            let opt_header = builder.create_block();
            let opt_body = builder.create_block();
            let opt_exit = builder.create_block();

            builder.ins().jump(opt_header, &[]);
            builder.switch_to_block(opt_header);
            state.current_block = Some(opt_header);

            let idx = builder.use_var(opt_i_var);
            let opt_cond = builder
                .ins()
                .icmp(IntCC::SignedLessThan, idx, num_params_val);
            builder.ins().brif(opt_cond, opt_body, &[], opt_exit, &[]);

            builder.switch_to_block(opt_body);
            builder.seal_block(opt_body);
            state.current_block = Some(opt_body);

            // D3 (ZeRO-1): owner-gated update. FullBuffer accum cleanup
            // (7g below) zeroes every accum slot unconditionally, so the
            // skip path needs no manual m_partial zero here.
            let opt_zero_blocks = if zero_enabled {
                let owns =
                    self.compile_call_by_name(builder, "nsl_zero_owns_param", &[idx])?;
                let one_i = builder.ins().iconst(cl_types::I64, 1);
                let owned = builder.ins().icmp(IntCC::Equal, owns, one_i);
                let z_do = builder.create_block();
                let z_join = builder.create_block();
                builder.ins().brif(owned, z_do, &[], z_join, &[]);
                builder.switch_to_block(z_do);
                builder.seal_block(z_do);
                Some(z_join)
            } else {
                None
            };

            // Get param, gradient, state buffers via runtime list indexing
            let param_val =
                self.compile_call_by_name(builder, "nsl_list_get", &[param_list, idx])?;
            let grad_val = self.compile_call_by_name(builder, "nsl_list_get", &[opt_grads, idx])?;
            let s1 = self.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, idx])?;

            let s2 = if num_state_buffers >= 2 {
                self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, idx])?
            } else {
                s1 // placeholder for non-Adam/SOAP optimizers (ignored by helper)
            };
            // FullBuffer-global path (no mode table, no WGGO). The CPDT
            // PrecisionPlan gate (`precision_active`'s 4th condition
            // requires `fase_deferred=true`) suppresses cpdt_precision_dtypes
            // construction here, so wrap_precision is always structurally
            // false at this call site. Threaded through for signature
            // uniformity with the unified-dispatch site.
            self.emit_stdlib_optim_call(
                builder,
                optimizer_name.as_str(),
                &opt_fn,
                param_val,
                grad_val,
                s1,
                s2,
                lr,
                momentum_const,
                dampening_const,
                weight_decay_const,
                nesterov_const,
                beta1_const,
                beta2_const,
                eps_const,
                step_count_var,
                false,
                None,
                self.compile_options.optim_state_offload,
            )?;

            if let Some(z_join) = opt_zero_blocks {
                builder.ins().jump(z_join, &[]);
                builder.switch_to_block(z_join);
                builder.seal_block(z_join);
            }
            let one_opt = builder.ins().iconst(cl_types::I64, 1);
            let next_opt = builder.ins().iadd(idx, one_opt);
            builder.def_var(opt_i_var, next_opt);
            builder.ins().jump(opt_header, &[]);
            builder.seal_block(opt_header);

            builder.switch_to_block(opt_exit);
            builder.seal_block(opt_exit);
            state.current_block = Some(opt_exit);

            // Offload P0.2: one drain per optimizer step (transfer-stream
            // sync + deferred frees of the staged tensors).
            if self.compile_options.optim_state_offload {
                self.compile_call_by_name(builder, "nsl_offload_drain", &[])?;
            }
        }

        // D3 (ZeRO-1): broadcast every param from its owner — the real
        // post-step sync (nsl_zero_step retained as a no-op for ABI compat).
        if zero_enabled {
            self.compile_call_by_name(builder, "nsl_zero_step", &[])?;
            let rc = self.compile_call_by_name(
                builder,
                "nsl_zero_sync_params",
                &[param_list, num_params_val],
            )?;
            let z = builder.ins().iconst(cl_types::I64, 0);
            let ok = builder.ins().icmp(IntCC::Equal, rc, z);
            let m = "nsl: ZeRO param sync failed (see message above) — aborting";
            self.intern_string(m)?;
            let mp = self.compile_string_literal(builder, m)?;
            self.compile_call_by_name(builder, "nsl_assert", &[ok, mp])?;
        }

        // 7g. Post-optimizer cleanup: zero accum buffers or free direct grads
        // Runtime loop over num_params_val
        if let Some(accum) = accum_list {
            let cleanup_i_var = state.new_variable();
            builder.declare_var(cleanup_i_var, cl_types::I64);
            let c_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(cleanup_i_var, c_zero);
            let c_header = builder.create_block();
            let c_body = builder.create_block();
            let c_exit = builder.create_block();
            builder.ins().jump(c_header, &[]);
            builder.switch_to_block(c_header);
            let ci = builder.use_var(cleanup_i_var);
            let cc = builder
                .ins()
                .icmp(IntCC::SignedLessThan, ci, num_params_val);
            builder.ins().brif(cc, c_body, &[], c_exit, &[]);
            builder.switch_to_block(c_body);
            builder.seal_block(c_body);
            let buf = self.compile_call_by_name(builder, "nsl_list_get", &[accum, ci])?;
            let n_elems = self.compile_call_by_name(builder, "nsl_tensor_len", &[buf])?;
            self.compile_call_by_name(builder, "nsl_grad_zero", &[buf, n_elems])?;
            let c_one = builder.ins().iconst(cl_types::I64, 1);
            let c_next = builder.ins().iadd(ci, c_one);
            builder.def_var(cleanup_i_var, c_next);
            builder.ins().jump(c_header, &[]);
            builder.seal_block(c_header);
            builder.switch_to_block(c_exit);
            builder.seal_block(c_exit);
            state.current_block = Some(c_exit);
        } else if !fase_hook_active {
            // No accumulation — free gradient tensors and grads_list every batch.
            // Skip when FASE hook is active: grads are already freed during
            // adjoint lowering, and grads_list is a null sentinel.
            let cleanup_i_var = state.new_variable();
            builder.declare_var(cleanup_i_var, cl_types::I64);
            let c_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(cleanup_i_var, c_zero);
            let c_header = builder.create_block();
            let c_body = builder.create_block();
            let c_exit = builder.create_block();
            builder.ins().jump(c_header, &[]);
            builder.switch_to_block(c_header);
            let ci = builder.use_var(cleanup_i_var);
            let cc = builder
                .ins()
                .icmp(IntCC::SignedLessThan, ci, num_params_val);
            builder.ins().brif(cc, c_body, &[], c_exit, &[]);
            builder.switch_to_block(c_body);
            builder.seal_block(c_body);
            let grad_val = self.compile_call_by_name(builder, "nsl_list_get", &[grads_list, ci])?;
            self.compile_call_by_name(builder, "nsl_tensor_free", &[grad_val])?;
            let c_one = builder.ins().iconst(cl_types::I64, 1);
            let c_next = builder.ins().iadd(ci, c_one);
            builder.def_var(cleanup_i_var, c_next);
            builder.ins().jump(c_header, &[]);
            builder.seal_block(c_header);
            builder.switch_to_block(c_exit);
            builder.seal_block(c_exit);
            state.current_block = Some(c_exit);
            self.compile_call_by_name(builder, "nsl_list_free", &[grads_list])?;
        }

        // Jump to post-optimizer block (merges optimizer and skip paths)
        if let Some(t0) = phase_opt_t0 {
                self.compile_call_by_name(builder, "nsl_cuda_device_synchronize", &[])?;
                let t3 = self.compile_call_by_name(builder, "nsl_clock", &[])?;
                let opt = builder.ins().fsub(t3, t0);
                self.compile_call_by_name(builder, "nsl_phase_optim_report", &[opt])?;
            }
            builder.ins().jump(post_optimizer_block, &[]);
        builder.switch_to_block(post_optimizer_block);
        builder.seal_block(post_optimizer_block);
        state.current_block = Some(post_optimizer_block);
        } // end else (non-FASE-Deferred optimizer path)

        // 7g2. Scheduler: update learning rate if scheduler is configured
        // NOTE: step_count is incremented AFTER the scheduler call so that
        // step 0 produces the step-0 learning rate (e.g. warmup starts correctly).
        if !scheduler_name.is_empty() {
            let sched_fn_name = match scheduler_name.to_lowercase().as_str() {
                "constant_lr" | "constantlr" => "constant_lr",
                "step_lr" | "steplr" => "step_lr",
                "exponential_lr" | "exponentiallr" => "exponential_lr",
                "linear_decay" | "lineardecay" => "linear_decay",
                "cosine_anneal" | "cosineanneal" => "cosine_anneal",
                "warmup_cosine" | "warmupcosine" => "warmup_cosine",
                "one_cycle" | "onecycle" => "one_cycle",
                _ => &scheduler_name,
            };
            let mangled = format!("nsl__optim__schedulers__{}", sched_fn_name);

            // Find the actual function name (check functions/runtime_fns with fallback)
            let sched_fn = if self.registry.functions.contains_key(mangled.as_str()) {
                mangled.clone()
            } else {
                let simple = sched_fn_name.to_string();
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

            let new_lr = match sched_fn_name {
                "constant_lr" => {
                    self.compile_call_by_name(builder, &sched_fn, &[base_lr_val, step_float])?
                }
                "step_lr" => {
                    let step_size = scheduler_args
                        .iter()
                        .find(|(n, _)| n == "step_size")
                        .map(|(_, v)| *v)
                        .unwrap_or(10.0);
                    let gamma = scheduler_args
                        .iter()
                        .find(|(n, _)| n == "gamma")
                        .map(|(_, v)| *v)
                        .unwrap_or(0.1);
                    let ss_val = builder.ins().f64const(step_size);
                    let g_val = builder.ins().f64const(gamma);
                    self.compile_call_by_name(
                        builder,
                        &sched_fn,
                        &[base_lr_val, step_float, ss_val, g_val],
                    )?
                }
                "exponential_lr" => {
                    let gamma = scheduler_args
                        .iter()
                        .find(|(n, _)| n == "gamma")
                        .map(|(_, v)| *v)
                        .unwrap_or(0.95);
                    let g_val = builder.ins().f64const(gamma);
                    self.compile_call_by_name(
                        builder,
                        &sched_fn,
                        &[base_lr_val, step_float, g_val],
                    )?
                }
                "linear_decay" => {
                    let total_steps = scheduler_args
                        .iter()
                        .find(|(n, _)| n == "total_steps")
                        .map(|(_, v)| *v)
                        .unwrap_or(1000.0);
                    let end_factor = scheduler_args
                        .iter()
                        .find(|(n, _)| n == "end_factor")
                        .map(|(_, v)| *v)
                        .unwrap_or(0.0);
                    let ts_val = builder.ins().f64const(total_steps);
                    let ef_val = builder.ins().f64const(end_factor);
                    self.compile_call_by_name(
                        builder,
                        &sched_fn,
                        &[base_lr_val, step_float, ts_val, ef_val],
                    )?
                }
                "cosine_anneal" => {
                    let t_max = scheduler_args
                        .iter()
                        .find(|(n, _)| n == "t_max")
                        .map(|(_, v)| *v)
                        .unwrap_or(1000.0);
                    let eta_min = scheduler_args
                        .iter()
                        .find(|(n, _)| n == "eta_min")
                        .map(|(_, v)| *v)
                        .unwrap_or(0.0);
                    let tm_val = builder.ins().f64const(t_max);
                    let em_val = builder.ins().f64const(eta_min);
                    self.compile_call_by_name(
                        builder,
                        &sched_fn,
                        &[base_lr_val, step_float, tm_val, em_val],
                    )?
                }
                "warmup_cosine" => {
                    let ws = scheduler_args
                        .iter()
                        .find(|(n, _)| n == "warmup_steps")
                        .map(|(_, v)| *v)
                        .unwrap_or(100.0);
                    let ts = scheduler_args
                        .iter()
                        .find(|(n, _)| n == "total_steps")
                        .map(|(_, v)| *v)
                        .unwrap_or(1000.0);
                    let ml = scheduler_args
                        .iter()
                        .find(|(n, _)| n == "min_lr")
                        .map(|(_, v)| *v)
                        .unwrap_or(1e-5);
                    let ws_val = builder.ins().f64const(ws);
                    let ts_val = builder.ins().f64const(ts);
                    let ml_val = builder.ins().f64const(ml);
                    self.compile_call_by_name(
                        builder,
                        &sched_fn,
                        &[base_lr_val, step_float, ws_val, ts_val, ml_val],
                    )?
                }
                "one_cycle" => {
                    let max_lr = scheduler_args
                        .iter()
                        .find(|(n, _)| n == "max_lr")
                        .map(|(_, v)| *v)
                        .unwrap_or(lr_value * 10.0);
                    let total_steps = scheduler_args
                        .iter()
                        .find(|(n, _)| n == "total_steps")
                        .map(|(_, v)| *v)
                        .unwrap_or(1000.0);
                    let pct_start = scheduler_args
                        .iter()
                        .find(|(n, _)| n == "pct_start")
                        .map(|(_, v)| *v)
                        .unwrap_or(0.3);
                    let ml_val = builder.ins().f64const(max_lr);
                    let ts_val = builder.ins().f64const(total_steps);
                    let ps_val = builder.ins().f64const(pct_start);
                    self.compile_call_by_name(
                        builder,
                        &sched_fn,
                        &[base_lr_val, step_float, ml_val, ts_val, ps_val],
                    )?
                }
                _ => base_lr_val, // fallback: no change
            };

            builder.def_var(lr_var, new_lr);
        }

        // 7h. Increment step count (after scheduler so step 0 uses the initial LR)
        let sc = builder.use_var(step_count_var);
        let one_i64 = builder.ins().iconst(cl_types::I64, 1);
        let sc_next = builder.ins().iadd(sc, one_i64);
        builder.def_var(step_count_var, sc_next);

        // 7i. Callbacks: compile on_step body with step_count and loss bound
        for cb in &callbacks {
            let cb_name = self.resolve_sym(cb.name).to_string();
            if cb_name == "on_step" {
                // Bind callback params: on_step(step, loss)
                for param in &cb.params {
                    let pname = self.resolve_sym(param.name).to_string();
                    match pname.as_str() {
                        "step" => {
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            let step_val = builder.use_var(step_count_var);
                            builder.def_var(var, step_val);
                            state.variables.insert(param.name, (var, cl_types::I64));
                        }
                        "loss" => {
                            // loss is already in state.variables, but rebind for callback scope
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            builder.def_var(var, loss_val);
                            state.variables.insert(param.name, (var, cl_types::I64));
                        }
                        _ => {
                            // Unknown callback param — bind to zero
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            let z = builder.ins().iconst(cl_types::I64, 0);
                            builder.def_var(var, z);
                            state.variables.insert(param.name, (var, cl_types::I64));
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
            let step_tensor_vars: Vec<_> = state
                .variables
                .iter()
                .filter(|(sym, _)| !vars_before_step.contains(sym))
                // Borrow aliases (e.g. `let alias = m.w`, DataLoader handles)
                // don't own their tensor — freeing them here would free the
                // model weight itself and use-after-free the next step.
                .filter(|(sym, _)| !state.non_owning_symbols.contains(sym))
                .filter(|(sym, _)| !state.borrowed_batch_symbols.contains(sym))
                .filter_map(|(sym, (var, _))| {
                    let sem_ty = state.variable_types.get(sym);
                    let is_tensor = sem_ty.map(|t| t.is_tensor()).unwrap_or(false);
                    let is_unknown = sem_ty.map(|t| t.is_indeterminate()).unwrap_or(true);
                    if is_tensor || is_unknown {
                        Some((*var, is_tensor))
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
        self.compile_call_by_name(builder, "nsl_gpu_drain_cache", &[])?;

        // Debug: GPU memory after step cleanup
        {
            let step_val = builder.use_var(step_count_var);
            self.compile_call_by_name(builder, "nsl_debug_gpu_mem", &[step_val])?;
            self.compile_call_by_name(builder, "nsl_debug_gpu_alloc_summary", &[step_val])?;
        }

        // ── 8. Close batch loop (if DataLoader) and increment epoch ──────
        let current = state.current_block.unwrap_or(batch_body_block);
        if has_dataloader.is_some() {
            // Jump back to batch_header for next batch
            if !is_block_filled(builder, current) {
                builder.ins().jump(batch_header_block, &[]);
            }
            // Now seal batch_header — both predecessors connected (entry + back-edge)
            builder.seal_block(batch_header_block);
            // batch_exit: all batches done → run epoch callbacks then increment
            builder.switch_to_block(batch_exit_block);
            builder.seal_block(batch_exit_block);
            state.current_block = Some(batch_exit_block);
        } else {
            // No DataLoader — single step per epoch, stay in the current block for epoch callbacks
            state.current_block = Some(current);
        }

        for cb in &callbacks {
            let cb_name = self.resolve_sym(cb.name).to_string();
            if cb_name == "on_epoch" || cb_name == "on_epoch_end" {
                for param in &cb.params {
                    let pname = self.resolve_sym(param.name).to_string();
                    match pname.as_str() {
                        "epoch" => {
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            let epoch_val = builder.use_var(epoch_counter_var);
                            builder.def_var(var, epoch_val);
                            state.variables.insert(param.name, (var, cl_types::I64));
                        }
                        "loss" => {
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            let epoch_loss = builder.use_var(epoch_loss_var);
                            builder.def_var(var, epoch_loss);
                            state.variables.insert(param.name, (var, cl_types::I64));
                        }
                        _ => {
                            let var = state.new_variable();
                            builder.declare_var(var, cl_types::I64);
                            let z = builder.ins().iconst(cl_types::I64, 0);
                            builder.def_var(var, z);
                            state.variables.insert(param.name, (var, cl_types::I64));
                        }
                    }
                }
                // Item 12: same scoped-residency guard as on_step — an
                // on_epoch callback that logs / saves model state runs with
                // every streamed param evicted.
                let ws_guard =
                    self.emit_callback_residency_open(builder, &cb.body, model_sym, &cb_name)?;
                for stmt in &cb.body.stmts {
                    self.compile_stmt(builder, state, stmt)?;
                }
                self.emit_callback_residency_close(builder, ws_guard)?;
            }
        }

        if on_epoch_binds_loss {
            let saved_loss = builder.use_var(epoch_loss_var);
            self.compile_call_by_name(builder, "nsl_tensor_free_if_valid", &[saved_loss])?;
            let epoch_loss_null = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(epoch_loss_var, epoch_loss_null);
        }

        let epoch_callback_block = state.current_block.unwrap_or(current);
        if !is_block_filled(builder, epoch_callback_block) {
            builder.ins().jump(increment_block, &[]);
        }

        builder.switch_to_block(increment_block);
        builder.seal_block(increment_block);
        state.current_block = Some(increment_block);
        let counter = builder.use_var(epoch_counter_var);
        let one = builder.ins().iconst(cl_types::I64, 1);
        let next = builder.ins().iadd(counter, one);
        builder.def_var(epoch_counter_var, next);
        builder.ins().jump(header_block, &[]);

        builder.seal_block(header_block);
        builder.switch_to_block(exit_block);
        builder.seal_block(exit_block);
        state.current_block = Some(exit_block);

        // Free param_list after training loop completes
        self.compile_call_by_name(builder, "nsl_list_free", &[param_list])?;

        // Free optimizer state buffers (runtime lists of momentum/velocity tensors)
        // Runtime loop: free each tensor in state_list_1 and state_list_2
        {
            let free_i_var = state.new_variable();
            builder.declare_var(free_i_var, cl_types::I64);
            let f_zero = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(free_i_var, f_zero);
            let f_header = builder.create_block();
            let f_body = builder.create_block();
            let f_exit = builder.create_block();
            builder.ins().jump(f_header, &[]);
            builder.switch_to_block(f_header);
            let fi = builder.use_var(free_i_var);
            let fc = builder
                .ins()
                .icmp(IntCC::SignedLessThan, fi, num_params_val);
            builder.ins().brif(fc, f_body, &[], f_exit, &[]);
            builder.switch_to_block(f_body);
            builder.seal_block(f_body);
            let buf1 = self.compile_call_by_name(builder, "nsl_list_get", &[state_list_1, fi])?;
            self.compile_call_by_name(builder, "nsl_tensor_free", &[buf1])?;
            if num_state_buffers >= 2 {
                let buf2 =
                    self.compile_call_by_name(builder, "nsl_list_get", &[state_list_2, fi])?;
                self.compile_call_by_name(builder, "nsl_tensor_free", &[buf2])?;
            }
            let f_one = builder.ins().iconst(cl_types::I64, 1);
            let f_next = builder.ins().iadd(fi, f_one);
            builder.def_var(free_i_var, f_next);
            builder.ins().jump(f_header, &[]);
            builder.seal_block(f_header);
            builder.switch_to_block(f_exit);
            builder.seal_block(f_exit);
            state.current_block = Some(f_exit);
        }
        self.compile_call_by_name(builder, "nsl_list_free", &[state_list_1])?;
        if num_state_buffers >= 2 {
            self.compile_call_by_name(builder, "nsl_list_free", &[state_list_2])?;
        }

        // Free gradient accumulation buffers (if allocated) — runtime loop.
        // CSLA (D1b): slots are NULL between windows (each window allocates
        // its accumulators fresh and frees them after its group updates;
        // the partial tail never allocates), so only the shell needs
        // freeing — the per-slot loop would nsl_tensor_free(0).
        if let Some(accum) = accum_list.filter(|_| csla_active) {
            self.compile_call_by_name(builder, "nsl_list_free", &[accum])?;
        } else if let Some(accum) = accum_list {
            let fa_i_var = state.new_variable();
            builder.declare_var(fa_i_var, cl_types::I64);
            let fa_z = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(fa_i_var, fa_z);
            let fa_hdr = builder.create_block();
            let fa_body = builder.create_block();
            let fa_exit = builder.create_block();
            builder.ins().jump(fa_hdr, &[]);
            builder.switch_to_block(fa_hdr);
            let fai = builder.use_var(fa_i_var);
            let fac = builder
                .ins()
                .icmp(IntCC::SignedLessThan, fai, num_params_val);
            builder.ins().brif(fac, fa_body, &[], fa_exit, &[]);
            builder.switch_to_block(fa_body);
            builder.seal_block(fa_body);
            let buf = self.compile_call_by_name(builder, "nsl_list_get", &[accum, fai])?;
            self.compile_call_by_name(builder, "nsl_tensor_free", &[buf])?;
            let fa_one = builder.ins().iconst(cl_types::I64, 1);
            let fa_next = builder.ins().iadd(fai, fa_one);
            builder.def_var(fa_i_var, fa_next);
            builder.ins().jump(fa_hdr, &[]);
            builder.seal_block(fa_hdr);
            builder.switch_to_block(fa_exit);
            builder.seal_block(fa_exit);
            state.current_block = Some(fa_exit);
            self.compile_call_by_name(builder, "nsl_list_free", &[accum])?;
        }

        // CSLA: sweep the trailing partial window. A window that never
        // reached the modulo boundary left its buffered saves + batch dicts
        // alive (the baseline discards the same tail's m_partial content —
        // its gradients never influence θ either way, so parity holds; this
        // sweep is purely against leaks). Every entry here is stale: its
        // iteration's callbacks are long done, so loss slots free
        // unconditionally too.
        if let (Some((saves_outer_var, dicts_var)), Some(sweep)) =
            (csla_buffers, &csla_teardown_slots)
        {
            let so = builder.use_var(saves_outer_var);
            let tail_len = self.compile_call_by_name(builder, "nsl_list_len", &[so])?;
            let sw_i_var = state.new_variable();
            builder.declare_var(sw_i_var, cl_types::I64);
            let sw_z = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(sw_i_var, sw_z);
            let sw_hdr = builder.create_block();
            let sw_body = builder.create_block();
            let sw_exit = builder.create_block();
            builder.ins().jump(sw_hdr, &[]);
            builder.switch_to_block(sw_hdr);
            let swi = builder.use_var(sw_i_var);
            let swc = builder.ins().icmp(IntCC::SignedLessThan, swi, tail_len);
            builder.ins().brif(swc, sw_body, &[], sw_exit, &[]);
            builder.switch_to_block(sw_body);
            builder.seal_block(sw_body);
            let inner = self.compile_call_by_name(builder, "nsl_list_get", &[so, swi])?;
            for (idx, free_fn) in sweep {
                let idx_val = builder.ins().iconst(cl_types::I64, *idx);
                let slot_val =
                    self.compile_call_by_name(builder, "nsl_list_get", &[inner, idx_val])?;
                self.compile_call_by_name(builder, free_fn, &[slot_val])?;
            }
            self.compile_call_by_name(builder, "nsl_list_free", &[inner])?;
            let sw_one = builder.ins().iconst(cl_types::I64, 1);
            let sw_next = builder.ins().iadd(swi, sw_one);
            builder.def_var(sw_i_var, sw_next);
            builder.ins().jump(sw_hdr, &[]);
            builder.seal_block(sw_hdr);
            builder.switch_to_block(sw_exit);
            builder.seal_block(sw_exit);
            state.current_block = Some(sw_exit);
            self.compile_call_by_name(builder, "nsl_list_free", &[so])?;

            // Tail batch dicts: nsl_dict_free_tensor_values destroys the
            // whole dict structure (values + shell, free_dict_impl(_, true))
            // — the same call the baseline makes per iteration; the
            // DataLoader teardown never touches popped dicts.
            let dl = builder.use_var(dicts_var);
            let dl_len = self.compile_call_by_name(builder, "nsl_list_len", &[dl])?;
            let dw_i_var = state.new_variable();
            builder.declare_var(dw_i_var, cl_types::I64);
            let dw_z = builder.ins().iconst(cl_types::I64, 0);
            builder.def_var(dw_i_var, dw_z);
            let dw_hdr = builder.create_block();
            let dw_body = builder.create_block();
            let dw_exit = builder.create_block();
            builder.ins().jump(dw_hdr, &[]);
            builder.switch_to_block(dw_hdr);
            let dwi = builder.use_var(dw_i_var);
            let dwc = builder.ins().icmp(IntCC::SignedLessThan, dwi, dl_len);
            builder.ins().brif(dwc, dw_body, &[], dw_exit, &[]);
            builder.switch_to_block(dw_body);
            builder.seal_block(dw_body);
            let tail_dict = self.compile_call_by_name(builder, "nsl_list_get", &[dl, dwi])?;
            self.compile_call_by_name(builder, "nsl_dict_free_tensor_values", &[tail_dict])?;
            let dw_one = builder.ins().iconst(cl_types::I64, 1);
            let dw_next = builder.ins().iadd(dwi, dw_one);
            builder.def_var(dw_i_var, dw_next);
            builder.ins().jump(dw_hdr, &[]);
            builder.seal_block(dw_hdr);
            builder.switch_to_block(dw_exit);
            builder.seal_block(dw_exit);
            state.current_block = Some(dw_exit);
            self.compile_call_by_name(builder, "nsl_list_free", &[dl])?;
        }

        // D2b part 2: LOAD-BEARING — under whole-loop streaming every
        // streamed param exits the training loop EVICTED (the forward
        // evicts after its last primal touch each micro-batch and there is
        // no post-epilogue restore), so this teardown is the SOLE restore
        // of device residency for model_save/eval, plus the pinned-mirror
        // release. Removing or reordering it after any θ reader crashes on
        // null data pointers.
        if csla_active && self.compile_options.weight_stream {
            self.compile_call_by_name(builder, "nsl_weight_stream_teardown", &[])?;
        }

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

        // Run backward pass
        let grads_list =
            self.compile_call_by_name(builder, "nsl_tape_backward", &[loss_val, param_list])?;

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
    fn emit_packing_registry_stash(
        &mut self,
        builder: &mut FunctionBuilder,
        batch_val: Value,
    ) -> Result<(), CodegenError> {
        use cranelift_codegen::ir::condcodes::IntCC;
        let k_seg = self.compile_string_literal(builder, "segment_ids")?;
        let has_seg =
            self.compile_call_by_name(builder, "nsl_dict_contains", &[batch_val, k_seg])?;
        let has_seg_block = builder.create_block();
        let no_seg_block = builder.create_block();
        let after_block = builder.create_block();
        let has_seg_cond = builder.ins().icmp_imm(IntCC::NotEqual, has_seg, 0);
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
        self.compile_call_by_name(builder, "nsl_packing_metadata_set", &[zero, zero])?;
        builder.ins().jump(after_block, &[]);

        builder.switch_to_block(after_block);
        builder.seal_block(after_block);

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

    fn free_wengert_owned_values(
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
    fn is_trainable_param_name(&self, param_name: &str) -> bool {
        is_trainable_param_leaf_name(param_name)
    }

    fn enumerate_all_model_tensor_paths(&self, var_name: &str, type_name: &str) -> Vec<String> {
        let mut paths = Vec::new();
        self.enumerate_tensor_paths_recursive(var_name, type_name, &mut paths, 0, true);
        paths
    }

    /// Enumerate all tensor field paths in a model struct via DFS.
    ///
    /// This mirrors the compiler's view of nested models and fixed arrays, so
    /// the emitted param_list and source-AD parameter resolution stay aligned.
    fn enumerate_model_tensor_paths(&self, var_name: &str, type_name: &str) -> Vec<String> {
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
            for site in &self.adapter_sites {
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
                    cranelift_codegen::ir::MemFlags::trusted(),
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

            if let Some(ref ft) = field_type {
                if ft.starts_with('[') && ft.contains(';') {
                    // FixedArray field: slots are stored inline in the parent struct.
                    // DON'T load the field value — instead compute the address of the
                    // array base region within the parent struct.
                    let inner = ft.trim_start_matches('[').trim_end_matches(']');
                    let elem_type = inner.split(';').next().unwrap_or("").trim();

                    // Set current_ptr to address of array base in parent struct
                    current_ptr = builder.ins().iadd_imm(current_ptr, field.offset as i64);
                    current_type_name = elem_type.to_string();
                    current_layout = self.types.struct_layouts.get(elem_type)?.clone();
                    // Next component should be a numeric index
                    i += 1;
                    continue;
                }
            }

            // Regular field: load the value
            let field_val = builder.ins().load(
                field.cl_type,
                cranelift_codegen::ir::MemFlags::trusted(),
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
        let mut model_sym: Option<nsl_ast::Symbol> = None;
        let mut optimizer_name = String::new();
        let mut lr_value: f64 = 0.01;
        let mut momentum_value: f64 = 0.0;
        let mut dampening_value: f64 = 0.0;
        let mut weight_decay_value: f64 = 0.0;
        let mut nesterov_value: bool = false;
        let mut beta1_value: f64 = 0.9;
        let mut beta2_value: f64 = 0.999;
        let mut eps_value: f64 = 1e-8;
        let mut step_body: Option<(&nsl_ast::stmt::Block, nsl_ast::Symbol)> = None;

        for arg in &train.config {
            if let Some(name_sym) = arg.name {
                let name = self.resolve_sym(name_sym).to_string();
                if name == "model" {
                    if let ExprKind::Ident(sym) = &arg.value.kind {
                        model_sym = Some(*sym);
                    }
                }
            }
        }

        for section in &train.sections {
            match section {
                TrainSection::Optimizer(expr) => {
                    if let ExprKind::Call { callee, args } = &expr.kind {
                        if let ExprKind::Ident(sym) = &callee.kind {
                            optimizer_name = self.resolve_sym(*sym).to_string().to_lowercase();
                        }
                        for arg in args {
                            if let Some(name_sym) = arg.name {
                                let name = self.resolve_sym(name_sym).to_string();
                                match name.as_str() {
                                    "lr" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            lr_value = *f;
                                        } else if let ExprKind::IntLiteral(n) = &arg.value.kind {
                                            lr_value = *n as f64;
                                        }
                                    }
                                    "momentum" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            momentum_value = *f;
                                        }
                                    }
                                    "dampening" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            dampening_value = *f;
                                        }
                                    }
                                    "weight_decay" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            weight_decay_value = *f;
                                        }
                                    }
                                    "nesterov" => {
                                        if let ExprKind::BoolLiteral(b) = &arg.value.kind {
                                            nesterov_value = *b;
                                        }
                                    }
                                    "beta1" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            beta1_value = *f;
                                        }
                                    }
                                    "beta2" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            beta2_value = *f;
                                        }
                                    }
                                    "eps" => {
                                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                                            eps_value = *f;
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
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
                _ => {}
            }
        }

        let model_sym = model_sym.ok_or_else(|| {
            CodegenError::new("pipelined train block requires 'model=<ident>' config argument")
        })?;

        if optimizer_name.is_empty() {
            return Err(CodegenError::new(
                "pipelined train block requires an optimizer section",
            ));
        }

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
            let init_i = state.new_variable();
            builder.declare_var(init_i, cl_types::I64);
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
        let step_param_var = state.new_variable();
        builder.declare_var(step_param_var, cl_types::I64);
        let init_null = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_param_var, init_null);
        state
            .variables
            .insert(step_param_sym, (step_param_var, cl_types::I64));

        let step_count_var = state.new_variable();
        builder.declare_var(step_count_var, cl_types::I64);
        let zero_i64 = builder.ins().iconst(cl_types::I64, 0);
        builder.def_var(step_count_var, zero_i64);

        // Phase 5 Task 7: publish step counter for @inspect in pipelined train.
        self.inspect_train_step_var = Some(step_count_var);

        let lr_var = state.new_variable();
        builder.declare_var(lr_var, cl_types::F64);
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
            let gs_i = state.new_variable();
            builder.declare_var(gs_i, cl_types::I64);
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
            let opt_i = state.new_variable();
            builder.declare_var(opt_i, cl_types::I64);
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
                    self.compile_call_by_name(
                        builder,
                        &opt_fn,
                        &[
                            param_val,
                            grad_val,
                            s1,
                            lr,
                            momentum_const,
                            weight_decay_const,
                            nesterov_const,
                        ],
                    )?;
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
            let cl_i = state.new_variable();
            builder.declare_var(cl_i, cl_types::I64);
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
                        let var = state.new_variable();
                        builder.declare_var(var, cl_types::I64);
                        builder.def_var(var, loss_tensor);
                        state.variables.insert(*loss_sym, (var, cl_types::I64));
                    }
                    // Bind grads (tensor ptr)
                    if let PatternKind::Ident(grads_sym) = &pats[1].kind {
                        let var = state.new_variable();
                        builder.declare_var(var, cl_types::I64);
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

    fn compile_source_ad_grad_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        grad: &nsl_ast::block::GradBlock,
        targets_val: Value,
    ) -> Result<Option<(Value, Value)>, CodegenError> {
        eprintln!("[nsl] Using source-to-source AD for grad block");

        // Cycle-10 §5.3 Task 6 wire-up (grad block): route per-fn
        // @checkpoint(policy=...) policies into the extractor. Empty map
        // = byte-identity preserved.
        let mut extractor = crate::source_ad::WengertExtractor::new(self.interner)
            .with_checkpoint_policies(if self.compile_options.training_reference {
                    Default::default() // P1.7: ignore @checkpoint decorators in the reference path
                } else {
                    self.compile_options.checkpoint_policies.clone()
                });
        extractor.set_model_method_bodies(self.models.model_method_bodies.clone());
        extractor.set_model_field_types(self.models.model_field_types.clone());
        // WRGA B.3.2 Option 3: plumb synth overrides so the extractor
        // resolves sentinel-Ident callees/members emitted by the adapter
        // rewrite.
        extractor.set_synth_call_names(self.synth_call_names.clone());
        extractor.set_synth_member_names(self.synth_member_names.clone());
        self.register_source_ad_model_instances(&mut extractor, state);

        for &sym in state.variables.keys() {
            extractor.register_input(sym);
        }

        if !extractor.extract_stmts(&grad.body.stmts) {
            eprintln!(
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
            eprintln!(
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
                eprintln!(
                    "[nsl] source AD does not yet resolve this grad target shape, falling back to tape-based AD"
                );
                return Ok(None);
            }
        };
        let Some(target_var_id) = target_var_id else {
            eprintln!(
                "[nsl] source AD could not resolve grad target, falling back to tape-based AD"
            );
            return Ok(None);
        };

        let state_vars_by_name: std::collections::HashMap<String, Value> = state
            .variables
            .iter()
            .map(|(sym, (cvar, _))| (self.resolve_sym(*sym).to_string(), builder.use_var(*cvar)))
            .collect();
        let mut primal_vars = crate::wengert_lower::VarMap::new();

        for (sym, vid) in extractor.symbol_var_map() {
            if primal_vars.contains_key(vid) {
                continue;
            }
            if let Some(&(cvar, _)) = state.variables.get(sym) {
                primal_vars.insert(*vid, builder.use_var(cvar));
            } else {
                let name = self.resolve_sym(*sym).to_string();
                if let Some(&val) = state_vars_by_name.get(&name) {
                    primal_vars.insert(*vid, val);
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
        let mut gen = crate::source_ad::AdjointGenerator::new(start_var);
        let mut adjoint = gen.generate(extractor.wengert_list());

        if let Some(target_adj_var) = gen.adjoint_of(target_var_id) {
            let needed = std::collections::HashSet::from([target_adj_var]);
            adjoint.ops = crate::source_ad::eliminate_dead_gradients(&adjoint.ops, &needed);

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
                        eprintln!(
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
        for &sym in state.variables.keys() {
            if let Some(model_type_name) = self.resolve_source_ad_model_type_name(state, sym) {
                extractor.register_model_instance(sym, &model_type_name);
            }
        }
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

    fn load_source_ad_named_param(
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
            match self.compile_options.calibration_sidecar.as_ref() {
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
                MemFlags::trusted(),
                source_val,
                field.offset as i32,
            );

            if is_excluded {
                // Copy as-is via clone (bumps refcount internally)
                let cloned = self.compile_call_by_name(builder, "nsl_tensor_clone", &[src_val])?;
                builder
                    .ins()
                    .store(MemFlags::trusted(), cloned, new_ptr, field.offset as i32);
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
                    .store(MemFlags::trusted(), deq, new_ptr, field.offset as i32);
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
        let var = state.new_variable();
        builder.declare_var(var, cl_types::I64);
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
                    eprintln!("[calibration] AWQ discovery for model '{model_name}': {e}");
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
    /// All emission gated on `compile_options.inspect_enabled`.  When that
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
            if let Some(args) = &decorator.args {
                if let Some(first) = args.first() {
                    if first.name.is_none() {
                        if let ExprKind::Ident(sym) = &first.value.kind {
                            s = *sym;
                        }
                    }
                }
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
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            if *n > 0 {
                                every_n = Some(*n);
                            }
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
                    eprintln!(
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
/// optimizer section.
///
/// Returns library defaults when:
/// - `train` is `None`
/// - the optimizer section is missing
/// - the optimizer is not `AdamW` (case-insensitive, matches the FASE
///   optimizer-name lookup at the top of `compile_train_block`)
/// - the optimizer expression is not a direct call
///
/// For β1, β2, ε: any field present as a `FloatLiteral` keyword arg overrides
/// the default; missing or non-literal values retain the default. `lr` and
/// `weight_decay` are intentionally NOT read here — CPDT's hyperparams cover
/// only the running-moment constants. Unknown kwargs are silently ignored for
/// forward compatibility.
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

    // Find the Optimizer section (there should be at most one meaningful one).
    let opt_expr = train.sections.iter().find_map(|s| match s {
        TrainSection::Optimizer(e) => Some(e),
        _ => None,
    });
    let Some(opt_expr) = opt_expr else {
        return hp;
    };

    // Pattern-match Call { callee=Ident("AdamW"), args }.
    let ExprKind::Call { callee, args } = &opt_expr.kind else {
        return hp;
    };
    let ExprKind::Ident(name_sym) = &callee.kind else {
        return hp;
    };
    // Mirror the FASE site: lowercase compare ("AdamW" → "adamw").
    if interner
        .resolve(name_sym.0)
        .unwrap_or("<unknown>")
        .to_lowercase()
        != "adamw"
    {
        return hp;
    }

    for arg in args {
        let Some(name_sym) = arg.name else { continue };
        let kw = interner.resolve(name_sym.0).unwrap_or("<unknown>");
        // Accept FloatLiteral; also tolerate IntLiteral for eps (e.g. eps=0).
        let val = match &arg.value.kind {
            ExprKind::FloatLiteral(f) => *f,
            ExprKind::IntLiteral(n) => *n as f64,
            _ => continue,
        };
        match kw {
            "beta1" => hp.beta1 = val,
            "beta2" => hp.beta2 = val,
            "eps" => hp.eps = val,
            _ => {} // unknown kwargs silently ignored
        }
    }

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
