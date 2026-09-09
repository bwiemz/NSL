//! The WRGA adapter sites of the train block's source-AD arm: the
//! override-rejected diagnostics (which WGGO-requested ranks the plan
//! adjusted, rendered like the CSHA explainer), the adapter init
//! side-table (from the pre-scan plan or the plan itself), and the
//! adapter-tensor loads that walk the model struct layout and insert each
//! adapter's tensor into the VarMap so the forward reads it like a
//! parameter.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 158 lines,
//! 6 inputs ([`AdapterSitesInputs`]) plus the function builder
//! and state. A no-op without a WRGA plan. The train-block CLIF snapshots
//! (`tests/train_clif_snapshots.rs`) pin the lowering around it.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, Value};
use cranelift_frontend::FunctionBuilder;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::context::StructLayout;
use crate::wengert_lower::VarMap;

/// Every binding of `compile_train_block_inner` the adapter sites read;
/// names are the driver's.
pub(crate) struct AdapterSitesInputs<'a> {
    /// The forward extractor (its named parameters locate the adapter sites).
    pub(crate) extractor: &'a crate::source_ad::WengertExtractor<'a>,
    /// The model struct layout (the adapter-tensor loads walk it).
    pub(crate) layout: &'a StructLayout,
    /// The model struct pointer.
    pub(crate) model_ptr: Value,
    /// The resolved model type name (the adapter init side-table is keyed by it; the layout walk starts from a clone).
    pub(crate) model_type_name: &'a String,
    /// The initial VarMap; the adapter tensors are inserted into it.
    pub(crate) primal_vars: &'a mut VarMap,
    /// The WRGA plan; `None` means no adapter sites.
    pub(crate) wrga_plan: &'a Option<crate::wrga::WrgaPlan>,
}

impl Compiler<'_> {
    /// Emit the WRGA adapter sites (see the module header).
    pub(crate) fn emit_wrga_adapter_sites(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        inputs: AdapterSitesInputs<'_>,
    ) -> Result<(), CodegenError> {
        let AdapterSitesInputs {
            extractor,
            layout,
            model_ptr,
            model_type_name,
            primal_vars,
            wrga_plan,
        } = inputs;

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
                model_type_name,
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

        Ok(())
    }
}
