//! WRGA B.2.1 Task 5.5 wiring fix: pre-scan adapter sites from decorator
//! info before `compile_user_functions` runs.
//!
//! Problem this solves: `compile_user_functions` compiles model method bodies
//! (including `forward`, which runs the Task 3 LoRA rewrite pass) BEFORE
//! `compile_main` runs the `@train` block that invokes
//! `invoke_wrga_if_enabled`. So `compiler.adapter_sites` was empty when
//! `forward` compiled, and the rewrite fell through silently.
//!
//! Fix: derive `adapter_sites` (and a minimal `bus.wrga_plan`) purely from
//! the user-facing `@adapter(target=..., type=..., rank=..., alpha=...)`
//! decorator configs (`compiler.wrga_inputs.adapter`) — no Wengert list,
//! no roofline analysis, no spectral pass. Just enough so:
//!
//!   * the Task 3 rewrite pass sees the sites for each target model, and
//!   * `adapter_field_index` can resolve synthesized field names to
//!     side-table slots.
//!
//! `invoke_wrga_if_enabled` still runs later inside the train block; it will
//! overwrite `bus.wrga_plan` / `adapter_sites` with the real plan. The
//! pre-scan is transparent to that path.

use std::collections::BTreeSet;

use crate::compiler::Compiler;
use crate::error::CodegenError;
use crate::cost_model::BoundClassification;
use crate::wengert::{VarId, WengertList};
use crate::wrga::WrgaPlan;
use crate::wrga_adapter_inject;
use crate::wrga_fused_ptx::{
    synthesize_fused_gatedlora_ptx, synthesize_fused_ia3_ptx, synthesize_fused_lora_ptx,
    FusedGatedLoraConfig, FusedIa3Config, FusedLoraConfig, LoraKernelKey,
};
use crate::wrga_fusion::{build_fusion_plan, FusionPlan, FusionTarget};
use crate::wrga_memory::MemoryPlan;
use crate::wrga_prune::{PruneResult, PruneStats};
use crate::wrga_roofline::{AdapterKind as RooflineAdapterKind, AdapterPlacement, SiteKind};

/// Build a minimal, zero-analysis `WrgaPlan` with one placement per
/// `@adapter(...)` target, then run the inject pass (with compiler dim
/// resolution) so `compiler.adapter_sites` is populated and each placement
/// has `synthesized_fields` + `init_strategies` set.
///
/// Must be called AFTER `collect_models` (so `model_field_types` is
/// populated) and BEFORE `compile_user_functions` (so model-method
/// compilation sees the sites).
///
/// An `@adapter` target that does not resolve is an error (see
/// [`wrga_adapter_inject::resolve_sites`]).
pub(crate) fn prescan_adapter_sites_from_decorators(
    compiler: &mut Compiler<'_>,
) -> Result<(), CodegenError> {
    let Some(inputs) = compiler.wrga_inputs.as_ref() else {
        return Ok(());
    };
    if inputs.adapter.is_empty() {
        return Ok(());
    }
    let scope = if inputs.adapters_inherited {
        wrga_adapter_inject::TargetScope::OwnModelsOnly
    } else {
        wrga_adapter_inject::TargetScope::AllTargets
    };

    let mut placements: Vec<AdapterPlacement> = Vec::new();
    for cfg in &inputs.adapter {
        for target in &cfg.targets {
            let rank = cfg.rank.unwrap_or(2).max(1) as usize;
            let roofline_kind = match cfg.kind {
                crate::AdapterKind::Lora => RooflineAdapterKind::Lora,
                crate::AdapterKind::Ia3 => RooflineAdapterKind::Ia3,
                // GatedLoRA has no roofline variant; re-use Lora as a
                // placeholder. `decorator_kind` carries the real kind into
                // the inject pass.
                crate::AdapterKind::GatedLora => RooflineAdapterKind::Lora,
            };
            placements.push(AdapterPlacement {
                name: target.clone(),
                // Decorator-targeted fields are matmul-shaped by
                // construction (LoRA/IA³/GatedLoRA only attach to 2D
                // weights through the prescan path), so set kind so the
                // fusion-decision pass doesn't re-derive from name.
                kind: SiteKind::Matmul,
                // B.3 Task 5: the prescan path has no Wengert graph to
                // measure arithmetic intensity, but `@adapter(type=lora,
                // target=["Class.field"])` syntactically targets a
                // matmul-shaped field by construction (LoRA only makes
                // sense on a 2D weight).  Default AI to 1.0 so the
                // fusion-decision pass classifies the site as Matmul →
                // EpilogueFusedLora, enabling the B.3 fused FFI path.
                arithmetic_intensity: 1.0,
                classification: BoundClassification::Unknown,
                roofline_slack: 1.0,
                adapter: roofline_kind,
                suggested_rank: rank,
                rationale: "wrga prescan (decorator-driven)".to_string(),
                decorator_kind: Some(cfg.kind),
                alpha: cfg.alpha,
                synthesized_fields: Vec::new(),
                init_strategies: Vec::new(),
            });
        }
    }

    if placements.is_empty() {
        return Ok(());
    }

    let empty_wengert = WengertList {
        ops: Vec::new(),
        output: 0 as VarId,
        var_names: Default::default(),
        var_types: Default::default(),
    };
    let mut plan = WrgaPlan {
        mode: nsl_ast::block::WrgaMode::Auto,
        target_gpu: String::new(),
        prune: PruneResult {
            pruned: empty_wengert,
            backward_live: BTreeSet::new(),
            activation_live: BTreeSet::new(),
            stats: PruneStats::default(),
        },
        placements,
        spectral: Vec::new(),
        ranks: Vec::new(),
        fusion: FusionPlan::default(),
        memory: MemoryPlan::default(),
        override_diagnostics: Vec::new(),
        ablation: Default::default(),
        custom_adapter: None,
        adapters_declared: true,
        unsized_sites: 0,
        rank_site_mn: Vec::new(),
        fused_kernels_active: false,
    };

    let inject = wrga_adapter_inject::resolve_sites(&mut plan, compiler, scope)?;
    // B.3 Task 4: run the fusion decision pass and wire the resulting
    // `FusionTarget` onto each `AdapterSite`, then synthesise + dedup
    // the PTX kernel per unique `(m, n, k, rank, target_sm)` key.  For
    // the prescan path we use batch=1 as the "m" (the real batch is
    // only known at runtime; the kernel shape is keyed on `m` only for
    // dedup / autotune — the CUDA launcher at Task 5 will handle batch
    // dispatch).
    let fusion = build_fusion_plan(&plan.placements, None);
    plan.fusion = fusion.clone();

    let mut sites = inject.sites;
    for site in sites.iter_mut() {
        for decision in &fusion.decisions {
            if decision.site == site.target_param {
                site.fusion_decision = Some(decision.target.clone());
                break;
            }
        }
    }

    // PTX synthesis + dedup.  Only runs for EpilogueFusedLora /
    // EpilogueFusedGatedLora sites whose weight shape is a literal
    // (input_dim/output_dim non-zero): a kernel is synthesized per shape.
    // Target sm is read from the compile target; defaults to 80 when absent
    // (the PTX asserts sm >= 80 anyway).
    //
    // Only the module that carries the decorators registers kernels with the
    // runtime (`main_entry`), so only it synthesizes them: a handle minted by
    // an imported module would index a registry that module never fills.
    let target_sm = compiler.target_sm().unwrap_or(80);
    if target_sm >= 80 && scope == wrga_adapter_inject::TargetScope::AllTargets {
        for site in &sites {
            match site.fusion_decision.as_ref() {
                Some(FusionTarget::EpilogueFusedLora { rank }) => {
                    if site.input_dim == 0 || site.output_dim == 0 || *rank > 16 {
                        continue;
                    }
                    let cfg = FusedLoraConfig {
                        site_id: site.site_id.clone(),
                        m: 1,
                        n: site.output_dim,
                        k: site.input_dim,
                        rank: *rank as u32,
                        target_sm,
                    };
                    let key = cfg.kernel_key();
                    compiler
                        .fused_ptx_kernels
                        .entry(key)
                        .or_insert_with(|| synthesize_fused_lora_ptx(&cfg));
                }
                Some(FusionTarget::EpilogueFusedGatedLora { rank }) => {
                    // B.3.1 Task 5.0.c: synthesise GatedLoRA PTX and insert
                    // into the separate `fused_gatedlora_ptx_kernels` map.
                    // Kept separate from `fused_ptx_kernels` to avoid key
                    // collision when LoRA and GatedLoRA share the same
                    // (m, n, k, rank, target_sm) shape.
                    if site.input_dim == 0 || site.output_dim == 0 || *rank > 16 {
                        continue;
                    }
                    let cfg = FusedGatedLoraConfig {
                        site_id: site.site_id.clone(),
                        m: 1,
                        n: site.output_dim,
                        k: site.input_dim,
                        rank: *rank as u32,
                        target_sm,
                    };
                    let key = cfg.kernel_key();
                    compiler
                        .fused_gatedlora_ptx_kernels
                        .entry(key)
                        .or_insert_with(|| synthesize_fused_gatedlora_ptx(&cfg));
                }
                Some(FusionTarget::ActivationFusedIa3) => {
                    // The IA³ kernel was synthesized by nothing: every IA³
                    // call carried handle -1, so the "fused" IA³ site could
                    // only ever run the ordinary operations.
                    if site.input_dim == 0 || site.output_dim == 0 {
                        continue;
                    }
                    let cfg = FusedIa3Config {
                        site_id: site.site_id.clone(),
                        m: 1,
                        n: site.output_dim,
                        k: site.input_dim,
                        target_sm,
                    };
                    let key = LoraKernelKey {
                        m: cfg.m,
                        n: cfg.n,
                        k: cfg.k,
                        rank: 0,
                        target_sm,
                    };
                    compiler
                        .fused_ia3_ptx_kernels
                        .entry(key)
                        .or_insert_with(|| synthesize_fused_ia3_ptx(&cfg));
                }
                _ => continue,
            }
        }
    }

    // The fused kernels are an opt-in whose cost is not visible in the
    // program: say what was opted into. They take 16-bit float fragments, so
    // the whole site, the base product `x @ W` included, is computed in half
    // precision (a result off by 1e-3 where the ordinary operations are good
    // to 1e-6), and they measured slower than the ordinary GEMMs.
    let fused_kernels = compiler.fused_ptx_kernels.len()
        + compiler.fused_gatedlora_ptx_kernels.len()
        + compiler.fused_ia3_ptx_kernels.len();
    if fused_kernels > 0 && crate::wrga_adapter_rewrite::fused_path_selected(compiler.target_sm()) {
        nsl_log::nsl_log!(
            WARN,
            "wrga",
            "[wrga] NSL_WRGA_FUSED_CUDA=1: {fused_kernels} adapter kernel(s) synthesized. On \
             the GPU an adapted matmul then runs in half precision (16-bit floats), the base \
             product `x @ W` included: expect about three significant digits where the \
             ordinary operations give seven. Unset NSL_WRGA_FUSED_CUDA to use the ordinary \
             operations."
        );
    }

    compiler.bus.publish_adapter_sites(sites);
    compiler.bus.publish_wrga_plan(plan);
    Ok(())
}

impl Compiler<'_> {
    /// Hold a train block's model to what its `@wrga(...)` decorators ask
    /// for, in both AD modes: a `target=` that names a GPU the analysis
    /// knows, and a `budget=` the declared adapters fit in.
    ///
    /// `param_paths` is the train block's parameter list; the adapters'
    /// tensors are in it, one set per instance of an adapted model.
    ///
    /// `budget=N` used to reach only the rank suggestions of
    /// `--wrga-report`, and there as a number of ranks: it bounded nothing.
    /// The ranks that exist are the ones `@adapter` declares, so the budget
    /// is checked against them: a model whose adapters hold more than `N`
    /// parameters does not compile.
    pub(crate) fn check_wrga_decorators(
        &self,
        model_var_name: &str,
        model_type_name: &str,
        param_paths: &[String],
    ) -> Result<(), CodegenError> {
        let Some(inputs) = self.wrga_inputs.as_ref() else {
            return Ok(());
        };
        for cfg in &inputs.wrga {
            if let Some(name) = cfg.target.as_deref().filter(|n| !n.is_empty())
                && crate::wrga::resolve_gpu(name).is_none()
            {
                return Err(CodegenError::new(format!(
                    "WRGA target '{name}' (from @wrga(target=...) or --wrga-target) is not a \
                     GPU the analysis knows; known targets: {}",
                    crate::wrga::known_gpu_names().join(", ")
                )));
            }
        }
        let Some(budget) = inputs.wrga.iter().find_map(|c| c.budget).filter(|b| *b > 0) else {
            return Ok(());
        };
        let budget = budget as usize;
        let mut total = 0usize;
        for path in param_paths {
            let leaf = path.rsplit('.').next().unwrap_or(path);
            if !crate::expr::access::is_synthesized_adapter_field_name(leaf) {
                continue;
            }
            let Some((class, _)) = self.resolve_param_owner(model_type_name, path) else {
                continue;
            };
            let Some(site) = self
                .bus
                .adapter_sites()
                .iter()
                .find(|s| s.target_model == class && s.synthesized_fields.iter().any(|f| f == leaf))
            else {
                continue;
            };
            if site.input_dim == 0 || site.output_dim == 0 {
                return Err(CodegenError::new(format!(
                    "@wrga(budget={budget}) cannot be checked: the shape of '{}.{}' is not a \
                     literal in the model definition, so the size of its adapter is not known \
                     at compile time. Remove the budget, or give the weight a literal shape",
                    site.target_model, site.target_field
                )));
            }
            let (n_in, n_out) = (site.input_dim as usize, site.output_dim as usize);
            let rank = site.rank.max(1) as usize;
            total += if leaf.starts_with("lora_A_") {
                n_in * rank
            } else if leaf.starts_with("lora_B_") {
                rank * n_out
            } else {
                // `ia3_scale_*` and `gate_*`: one value per output channel.
                n_out
            };
        }
        if total > budget {
            return Err(CodegenError::new(format!(
                "@wrga(budget={budget}): the adapters on '{model_var_name}' hold {total} \
                 parameters, {} over the budget. Lower a rank, adapt fewer weights, or raise \
                 the budget",
                total - budget
            )));
        }
        Ok(())
    }
}

/// WRGA B.3.2 Option 3 phase 3e: apply the adapter rewrite to
/// `compiler.models.model_method_bodies` so source-AD's inline expansion
/// sees the fused FFI call in the train block.
///
/// Parallel to `compile_user_functions`'s rewrite — both paths use the
/// shared helper `wrga_adapter_rewrite::rewrite_stmts_for_model` to
/// prevent drift. See
/// `docs/superpowers/specs/2026-04-19-wrga-b32-option3-revised-design.md`
/// §1 for the gap this closes: `model_method_bodies` is populated at
/// `declaration.rs:490` with unrewritten `fn_def.clone()`, *before*
/// `prescan_adapter_sites_from_decorators` runs. Without this pass the
/// source-AD path walks stale AST and never sees the fused FFI callee.
///
/// Runs immediately after `prescan_adapter_sites_from_decorators` at each
/// of the 4 entry points in `compiler/entry_points.rs`, and before the
/// first consumer of `model_method_bodies` (which is source-AD inside
/// `compile_main`).
///
/// No-op if `adapter_sites` is empty.
pub(crate) fn rewrite_model_method_bodies_with_adapter_sites(
    compiler: &mut crate::compiler::Compiler<'_>,
) {
    if compiler.bus.adapter_sites().is_empty() {
        return;
    }
    // Collect model names first to avoid mutable-borrow conflict with the
    // helper's `&mut compiler` argument.
    let model_names: Vec<String> = compiler
        .models
        .model_method_bodies
        .keys()
        .cloned()
        .collect();
    for model_name in model_names {
        let method_names: Vec<String> = compiler
            .models
            .model_method_bodies
            .get(&model_name)
            .map(|m| m.keys().cloned().collect())
            .unwrap_or_default();
        for method_name in method_names {
            // Take a clone of the fn_def for input; we only mutate
            // `body.stmts` after the helper returns, so holding the clone
            // for the helper call is safe.
            let fn_def_opt = compiler
                .models
                .model_method_bodies
                .get(&model_name)
                .and_then(|m| m.get(&method_name))
                .cloned();
            let Some(fn_def) = fn_def_opt else {
                continue;
            };
            let rewritten = crate::wrga_adapter_rewrite::rewrite_stmts_for_model(
                compiler,
                &model_name,
                &fn_def,
                &fn_def.body.stmts,
            );
            if let Some(map) = compiler.models.model_method_bodies.get_mut(&model_name)
                && let Some(body) = map.get_mut(&method_name)
            {
                body.body.stmts = rewritten;
            }
        }
    }
}
