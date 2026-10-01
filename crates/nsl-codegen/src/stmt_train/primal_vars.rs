//! Section 3 of the train block's source-AD arm: the initial `VarMap`.
//! Every named input / parameter `VarId` of the extracted Wengert list is
//! mapped to the Cranelift value already present in the function state —
//! two name-order passes (symbol map, then `Input` leaves by name), the
//! runtime device guard on every consumed tensor input, the step
//! parameter, the nested model-parameter loads walked through the struct
//! layout, and the frozen teacher inputs. The CPKD Distillation Build
//! Report facts are collected here too, while the extractor is in scope.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1). TrainPlan step 5
//! splits it in three, planning from emission:
//! - [`Compiler::plan_primal_facts`] reads the extractor and the function
//!   state and returns [`PrimalFacts`]: what the map needs, as data. It takes
//!   no builder. The facts that read the tape (the `Input` leaves and which of
//!   them are guarded) are taken here, on the tape as extracted, so a later
//!   pass that rewrites the tape cannot change what the emitter guards.
//! - [`Compiler::plan_cpkd_report`] runs the scheduled CPKD pass.
//! - [`Compiler::emit_primal_vars`] emits the `use_var`s, guards and loads
//!   from the facts, in the order the combined function did.
//!
//! The map is the emitter's result (the driver keeps it mutable: WRGA's
//! adapter tensors are inserted later). The train-block CLIF snapshots
//! (`tests/train_clif_snapshots.rs`) pin the emitted loads and guards on
//! every source-AD fixture.

use cranelift_codegen::ir::Value;
use cranelift_frontend::{FunctionBuilder, Variable};
use nsl_semantic::types::Type;

use crate::compiler::Compiler;
use crate::context::{FuncState, StructLayout};
use crate::error::CodegenError;

/// What the VarMap build needs from the extractor and the function state,
/// as data: every list in the order the emitter walks it.
pub(crate) struct PrimalFacts {
    /// The extractor's symbol map, sorted by `(VarId, name)`.
    symbol_vars: Vec<(nsl_ast::Symbol, crate::wengert::VarId)>,
    /// The tape's `Input` leaves, in tape order.
    input_leaves: Vec<(crate::wengert::VarId, String)>,
    /// The `Input` leaves that get a device guard, in tape order: those whose
    /// variable is semantically a tensor and that some op consumes.
    guarded_inputs: Vec<crate::wengert::VarId>,
    /// The step parameter's VarId, if the step body reads it.
    step_vid: Option<crate::wengert::VarId>,
    /// Model parameters in the extractor's order, each with its field walk
    /// through the model's layout.
    named_params: Vec<(crate::wengert::VarId, crate::stmt::FieldPlan)>,
    /// CPKD: frozen teacher inputs in the extractor's order, each with its
    /// root variable and field walk; `None` when the root, its model type or
    /// its layout does not resolve (nothing is emitted for those).
    frozen_inputs: Vec<(crate::wengert::VarId, Option<(nsl_ast::Symbol, crate::stmt::FieldPlan)>)>,
    /// The VarIds the emitter will map: the VarMap's key set, known before
    /// anything is emitted (TrainPlan step 5b). Adapter tensors are among
    /// them: `plan_nested_field` reaches each through a
    /// `FieldStep::AdapterSlot`.
    pub(crate) mapped: std::collections::HashSet<crate::wengert::VarId>,
}

/// The emission handles the VarMap build reads; names are the driver's.
pub(crate) struct PrimalVarsHandles {
    pub(crate) model_ptr: Value,
    pub(crate) param_list: Value,
    /// The step parameter's Cranelift variable.
    pub(crate) step_param_var: Variable,
}

impl Compiler<'_> {
    /// The facts the VarMap build needs (see the module header). Reads the
    /// extractor and the function state; emits nothing.
    pub(crate) fn plan_primal_facts(
        &self,
        state: &FuncState,
        extractor: &crate::source_ad::WengertExtractor<'_>,
        step_param_sym: nsl_ast::Symbol,
        layout: &StructLayout,
        model_type_name: &str,
    ) -> PrimalFacts {
        let mut symbol_vars: Vec<(nsl_ast::Symbol, crate::wengert::VarId)> = extractor
            .symbol_var_map()
            .iter()
            .map(|(sym, vid)| (*sym, *vid))
            .collect();
        symbol_vars.sort_by_key(|&(sym, vid)| (vid, self.resolve_sym(sym)));

        let input_leaves: Vec<(crate::wengert::VarId, String)> = extractor
            .wengert_list()
            .ops
            .iter()
            .filter_map(|op| match &op.op {
                crate::wengert::PrimalOp::Input(name) => Some((op.result, name.clone())),
                _ => None,
            })
            .collect();

        // Item 2 (2026-08-25): every SEMANTICALLY-TENSOR Input leaf
        // gets a device guard call — a host-resident dense-float
        // input on a GPU-parameter model REFUSES at runtime instead
        // of silently reconciling every weight down to the host
        // (f64, single-threaded; the defect that made #524's first
        // gate fixture look like a hang). The type filter is
        // load-bearing: EVERY outer variable registers as an Input
        // leaf (DataLoader handles, strings, ints included), and
        // handing a non-tensor i64 to the runtime guard is a wild
        // pointer read — `NslTensor::from_ptr` checks its magic
        // only under debug_assert. A variable whose semantic type
        // is unknown is skipped (best-effort guard; unsound
        // guarding is worse than a miss). The runtime additionally
        // no-ops for CPU models, scalars, and index-dtype tensors.
        let tensor_input_names: std::collections::HashSet<String> = state
            .variables
            .keys()
            .filter(|sym| {
                state.variable_types.get(sym).is_some_and(|ty| {
                    let inner = match ty {
                        Type::Borrow(inner) => inner.as_ref(),
                        other => other,
                    };
                    // NOT Sparse: NslSparseTensor is a different
                    // repr(C) with no magic field — handing it to
                    // the guard is the wild-read class this filter
                    // exists to kill (review S1). Unreachable today
                    // (source AD has no sparse handlers); a skip is
                    // the safe direction if that changes.
                    matches!(
                        inner,
                        Type::Tensor { .. }
                            | Type::Param { .. }
                            | Type::Buffer { .. }
                    )
                })
            })
            .map(|sym| self.resolve_sym(*sym).to_string())
            .collect();
        // Only CONSUMED inputs: registration is eager over every
        // outer variable, so a tensor the step body never reads
        // (e.g. one used only to build the token stream before the
        // train block) still carries an Input leaf — guarding it
        // would refuse programs the defect cannot touch.
        let consumed_vars: std::collections::HashSet<crate::wengert::VarId> = extractor
            .wengert_list()
            .ops
            .iter()
            .flat_map(|op| op.inputs.iter().copied())
            .collect();
        let guarded_inputs: Vec<crate::wengert::VarId> = input_leaves
            .iter()
            .filter(|(vid, name)| tensor_input_names.contains(name) && consumed_vars.contains(vid))
            .map(|(vid, _)| *vid)
            .collect();

        let named_params: Vec<_> = extractor
            .named_param_var_ids()
            .iter()
            .map(|(name, vid)| (*vid, self.plan_nested_field(layout, model_type_name, name)))
            .collect();
        let frozen_inputs: Vec<_> = extractor
            .frozen_input_var_ids()
            .iter()
            .map(|(name, vid)| (*vid, self.plan_source_ad_named_param(state, name)))
            .collect();
        let step_vid = extractor.symbol_var_map().get(&step_param_sym).copied();

        // The key set, by the emitter's own rules in its order: a VarId the
        // map already holds is skipped, so the first rule to claim it wins.
        let state_names: std::collections::HashSet<&str> =
            state.variables.keys().map(|sym| self.resolve_sym(*sym)).collect();
        let mut mapped = std::collections::HashSet::new();
        for &(sym, vid) in &symbol_vars {
            if state.variables.contains_key(&sym) || state_names.contains(self.resolve_sym(sym)) {
                mapped.insert(vid);
            }
        }
        for (vid, name) in &input_leaves {
            if state_names.contains(name.as_str()) {
                mapped.insert(*vid);
            }
        }
        mapped.extend(step_vid);
        for (vid, plan) in &named_params {
            if plan.resolved {
                mapped.insert(*vid);
            }
        }
        for (vid, frozen) in &frozen_inputs {
            if frozen.as_ref().is_some_and(|(_, plan)| plan.resolved) {
                mapped.insert(*vid);
            }
        }

        PrimalFacts { symbol_vars, input_leaves, guarded_inputs, step_vid, named_params, frozen_inputs, mapped }
    }

    /// Build the initial primal `VarMap` from [`PrimalFacts`] and emit the
    /// input device guards; returns the map.
    pub(crate) fn emit_primal_vars(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        facts: &PrimalFacts,
        handles: PrimalVarsHandles,
    ) -> Result<crate::wengert_lower::VarMap, CodegenError> {
        let PrimalVarsHandles { model_ptr, param_list, step_param_var } = handles;

        // 3. Build initial VarMap: map named input/param VarIds to
        //    Cranelift Values already present in state.variables.
        let mut primal_vars = crate::wengert_lower::VarMap::new();
        // Build primal_vars from state.variables using both Symbol-based
        // and name-based matching to handle cross-module Symbol mismatches.
        // Both walks are in name order: `use_var` numbers a value per
        // call, and walking either `HashMap` in its own order numbered
        // `main` differently on every compile (see
        // `variables_in_name_order`).
        let state_vars_by_name: std::collections::HashMap<String, Value> = self
            .variables_in_name_order(state)
            .into_iter()
            .map(|sym| {
                let (cvar, _) = state.variables[&sym];
                (self.resolve_sym(sym).to_string(), builder.use_var(cvar))
            })
            .collect();

        // First pass: map symbol_var_map entries via Symbol match or name fallback
        for &(sym, vid) in &facts.symbol_vars {
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
        // Second pass: map Input ops by name (catches inputs not in symbol_var_map)
        for (vid, name) in &facts.input_leaves {
            if let std::collections::hash_map::Entry::Vacant(entry) = primal_vars.entry(*vid)
                && let Some(&val) = state_vars_by_name.get(name)
            {
                entry.insert(val);
            }
        }

        // The device guards (see `plan_primal_facts` for which inputs).
        for vid in &facts.guarded_inputs {
            if let Some(&val) = primal_vars.get(vid) {
                self.compile_call_by_name(builder, "nsl_train_input_device_guard", &[val, param_list])?;
            }
        }

        // Also populate model parameter VarIds from param_list.
        // MemberAccess expressions (e.g., m.w) are registered in the extractor
        // under the member symbol, but those aren't in state.variables — they're
        // loaded from the model struct. Map them to nsl_list_get(param_list, i).
        //
        // Also add the step parameter (batch) which is stored separately
        if let Some(step_vid) = facts.step_vid {
            primal_vars
                .entry(step_vid)
                .or_insert_with(|| builder.use_var(step_param_var));
        }
        // Resolve model parameter VarIds by traversing nested struct layouts.
        // Compound names like "m.blocks.0.attn.wq" are split into path
        // components and walked through struct layouts + array indices,
        // emitting a chain of Cranelift loads at each level.
        for (vid, plan) in &facts.named_params {
            if primal_vars.contains_key(vid) {
                continue;
            }

            if let Some(val) = crate::stmt::emit_field_plan(builder, model_ptr, plan) {
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
        for (vid, frozen) in &facts.frozen_inputs {
            if primal_vars.contains_key(vid) {
                continue;
            }
            if let Some((root_sym, plan)) = frozen {
                let (root_var, _) = state.variables[root_sym];
                let root_ptr = builder.use_var(root_var);
                if let Some(val) = crate::stmt::emit_field_plan(builder, root_ptr, plan) {
                    primal_vars.insert(*vid, val);
                }
            }
        }

        Ok(primal_vars)
    }

    /// CPKD: collect the Distillation Build Report facts while the extractor
    /// is in scope, under the scheduled CPKD pass. Rendered by
    /// `compile_distill_block`. A no-op outside a `distill` block.
    pub(crate) fn plan_cpkd_report(
        &mut self,
        extractor: &crate::source_ad::WengertExtractor<'_>,
        fase_plan: &crate::fase::FasePlan,
        grad_accumulation_steps: i64,
    ) -> Result<(), CodegenError> {
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
            // Milestone C: CPKD gained a real module entry
            // (`cpkd::build_plan`) — record/disposition moved to the
            // callee with it — and the invocation is SCHEDULED. The
            // FusedKlCe tape scan above stays HERE: the registry
            // declares TapeAccess::None for CPKD and the drift gate
            // enforces zero WengertList mentions in the cpkd* family,
            // so the driver reads the tape and hands the RESULT in.
            // tape=None for the same reason; the plan holds no
            // positional refs and its sole consumer (`render_report`
            // via take_cpkd_plan) runs after the tape is gone.
            let sched = self.passes.scheduler();
            sched
                .schedule("CPKD", None, || {
                    let plan =
                        crate::cpkd::build_plan(crate::cpkd::CpkdPlan {
                teacher_name: self.resolve_sym(distill.teacher_sym).to_string(),
                student_name: self.resolve_sym(distill.student_sym).to_string(),
                epochs: distill.epochs,
                // The window and the FASE mode it selected. The report
                // stated `Epochs` alone, which is not the optimizer-step
                // count once a window exists, and said nothing about the
                // decision the window actually drives: `Passthrough` here
                // means there is no Deferred envelope, and therefore that
                // a CPDT optimizer-moment precision plan will arbitrate
                // to nothing no matter how good the plan is. Read from
                // the SAME locals the lowering used, three thousand lines
                // after they were computed, so the report cannot claim a
                // mode the emitted code does not have.
                grad_accumulation: grad_accumulation_steps.max(1),
                fase_mode: format!("{:?}", fase_plan.mode),
                loss: distill.loss.clone(),
                trainable_params: extractor.named_param_var_ids().len(),
                frozen_teacher_inputs: extractor.frozen_input_var_ids().len(),
                fused_shape: fused_op,
                logit_bytes_eliminated,
                        });
                    self.bus.publish_cpkd_plan(plan);
                })
                .map_err(CodegenError::new)?
                .finish(&self.bus)
                .map_err(CodegenError::new)?;
        }
        Ok(())
    }
}
