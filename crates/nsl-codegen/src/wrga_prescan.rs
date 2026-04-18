//! WRGA B.2.1 Task 5.5 wiring fix: pre-scan adapter sites from decorator
//! info before `compile_user_functions` runs.
//!
//! Problem this solves: `compile_user_functions` compiles model method bodies
//! (including `forward`, which runs the Task 3 LoRA rewrite pass) BEFORE
//! `compile_main` runs the `@train` block that invokes
//! `invoke_wrga_if_enabled`. So `compiler.adapter_sites` was empty when
//! `forward` compiled, and the rewrite fell through silently.
//!
//! Fix: derive `adapter_sites` (and a minimal `last_wrga_plan`) purely from
//! the user-facing `@adapter(target=..., type=..., rank=..., alpha=...)`
//! decorator configs (`compiler.wrga_inputs.adapter`) — no Wengert list,
//! no roofline analysis, no spectral pass. Just enough so:
//!
//!   * the Task 3 rewrite pass sees the sites for each target model, and
//!   * `adapter_field_index` can resolve synthesized field names to
//!     side-table slots.
//!
//! `invoke_wrga_if_enabled` still runs later inside the train block; it will
//! overwrite `last_wrga_plan` / `adapter_sites` with the real plan. The
//! pre-scan is transparent to that path.

use std::collections::BTreeSet;

use crate::compiler::Compiler;
use crate::cost_model::BoundClassification;
use crate::wengert::{VarId, WengertList};
use crate::wrga::WrgaPlan;
use crate::wrga_adapter_inject;
use crate::wrga_fused_ptx::{
    synthesize_fused_gatedlora_ptx, synthesize_fused_lora_ptx, FusedGatedLoraConfig,
    FusedLoraConfig,
};
use crate::wrga_fusion::{build_fusion_plan, FusionPlan, FusionTarget};
use crate::wrga_memory::MemoryPlan;
use crate::wrga_prune::{PruneResult, PruneStats};
use crate::wrga_roofline::{AdapterKind as RooflineAdapterKind, AdapterPlacement};

/// Build a minimal, zero-analysis `WrgaPlan` with one placement per
/// `@adapter(...)` target, then run the inject pass (with compiler dim
/// resolution) so `compiler.adapter_sites` is populated and each placement
/// has `synthesized_fields` + `init_strategies` set.
///
/// Must be called AFTER `collect_models` (so `model_field_types` is
/// populated) and BEFORE `compile_user_functions` (so model-method
/// compilation sees the sites).
pub(crate) fn prescan_adapter_sites_from_decorators(compiler: &mut Compiler<'_>) {
    let Some(inputs) = compiler.wrga_inputs.as_ref() else {
        return;
    };
    if inputs.adapter.is_empty() {
        return;
    }

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
        return;
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
    };

    let inject = wrga_adapter_inject::run_with_compiler(&mut plan, compiler);
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
    // EpilogueFusedGatedLora sites whose site record has resolved dimensions
    // (input_dim/output_dim non-zero).  Target sm is read from the compile
    // target; defaults to 80 when absent (the PTX asserts sm >= 80 anyway).
    let target_sm = compiler.target_sm().unwrap_or(80);
    if target_sm >= 80 {
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
                _ => continue,
            }
        }
    }

    compiler.adapter_sites = sites;
    compiler.last_wrga_plan = Some(plan.clone());
    compiler.adapter_prescan_plan = Some(plan);
}
