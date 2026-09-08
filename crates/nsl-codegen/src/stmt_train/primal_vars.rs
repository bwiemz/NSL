//! Section 3 of the train block's source-AD arm: the initial `VarMap`.
//! Every named input / parameter `VarId` of the extracted Wengert list is
//! mapped to the Cranelift value already present in the function state —
//! two name-order passes (symbol map, then `Input` leaves by name), the
//! runtime device guard on every consumed tensor input, the step
//! parameter, the nested model-parameter loads walked through the struct
//! layout, the frozen teacher inputs, and the CPKD Distillation Build
//! Report facts collected while the extractor is in scope.
//!
//! Moved out of `compile_train_block_inner` (roadmap A1): 239 lines,
//! 9 inputs ([`PrimalVarsInputs`]); the map is the function's
//! result (the driver keeps it mutable: WRGA's adapter tensors are inserted
//! later). The train-block CLIF snapshots (`tests/train_clif_snapshots.rs`)
//! pin the emitted loads and guards on every source-AD fixture.

use cranelift_codegen::ir::Value;
use cranelift_frontend::{FunctionBuilder, Variable};
use nsl_semantic::types::Type;

use crate::compiler::Compiler;
use crate::context::{FuncState, StructLayout};
use crate::error::CodegenError;

/// Every binding of `compile_train_block_inner` the VarMap build reads;
/// names are the driver's.
pub(crate) struct PrimalVarsInputs<'a> {
    /// The forward extractor (symbol map, Wengert list, named parameter and frozen-input VarIds).
    pub(crate) extractor: &'a crate::source_ad::WengertExtractor<'a>,
    pub(crate) fase_plan: &'a crate::fase::FasePlan,
    pub(crate) grad_accumulation_steps: i64,
    /// The model's struct layout (nested parameter loads walk it).
    pub(crate) layout: &'a StructLayout,
    pub(crate) model_ptr: Value,
    pub(crate) model_type_name: &'a str,
    pub(crate) param_list: Value,
    /// The step parameter (the batch) and its Cranelift variable.
    pub(crate) step_param_sym: nsl_ast::Symbol,
    pub(crate) step_param_var: Variable,
}

impl Compiler<'_> {
    /// Build the initial primal `VarMap` (see the module header) and emit
    /// the input device guards; returns the map.
    pub(crate) fn emit_primal_vars(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        inputs: PrimalVarsInputs<'_>,
    ) -> Result<crate::wengert_lower::VarMap, CodegenError> {
        let PrimalVarsInputs {
            extractor,
            fase_plan,
            grad_accumulation_steps,
            layout,
            model_ptr,
            model_type_name,
            param_list,
            step_param_sym,
            step_param_var,
        } = inputs;

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
        // Second pass: map Input ops by name (catches inputs not in symbol_var_map)
        for op in &extractor.wengert_list().ops {
            if let crate::wengert::PrimalOp::Input(name) = &op.op
                && let std::collections::hash_map::Entry::Vacant(entry) =
                    primal_vars.entry(op.result)
                && let Some(&val) = state_vars_by_name.get(name)
            {
                entry.insert(val);
            }
        }

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
        for op in &extractor.wengert_list().ops {
            if let crate::wengert::PrimalOp::Input(name) = &op.op {
                if !tensor_input_names.contains(name)
                    || !consumed_vars.contains(&op.result)
                {
                    continue;
                }
                if let Some(&val) = primal_vars.get(&op.result) {
                    self.compile_call_by_name(
                        builder,
                        "nsl_train_input_device_guard",
                        &[val, param_list],
                    )?;
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
                layout,
                model_type_name,
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

        Ok(primal_vars)
    }
}
