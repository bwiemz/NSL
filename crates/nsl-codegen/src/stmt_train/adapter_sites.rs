//! The WRGA adapter sites of the train block's source-AD arm: the
//! override-rejected diagnostics (which WGGO-requested ranks the plan
//! adjusted, rendered like the CSHA explainer).
//!
//! The adapter tensors themselves are no longer handled here. Each model
//! instance's side-table is built by its constructor
//! (`wrga_adapter_init::emit_adapter_init_sidetable`), and an adapter
//! parameter such as `m.blocks.0.lora_A_Blk_w__lora` resolves through
//! `plan_nested_field` like any other parameter, via a
//! `FieldStep::AdapterSlot`. Moved out of `compile_train_block_inner`
//! (roadmap A1).

use crate::compiler::Compiler;

impl Compiler<'_> {
    /// Log the WRGA plan's override-rejected diagnostics. Emits no IR.
    pub(crate) fn report_wrga_override_diagnostics(&self, wrga_plan: &Option<crate::wrga::WrgaPlan>) {
        // Task 6: render any override-rejected diagnostics to stderr so
        // the Phase 3 decision explainer and the user can see which
        // WGGO-requested ranks were adjusted.  Format matches the CSHA
        // renderer so both can be parsed uniformly.
        if let Some(plan) = wrga_plan {
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
                nsl_log::nsl_log!(INFO, "wrga", 
                    "[wrga] layer:{} wggo-override-rejected requested={} applied={} reason={}",
                    diag.layer_index, diag.requested, diag.applied, reason_str
                );
            }
        }
    }
}
