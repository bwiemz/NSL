//! The `grad` block drivers: `compile_grad_block` (the dispatch between the
//! source-AD and tape-AD arms), `compile_source_ad_grad_block` (the
//! compile-time backward: extract the step body's Wengert list, lower the
//! primal, generate and lower the adjoint, build the gradient list) and
//! `compile_tape_grad_block` / `compile_tape_backward` (the runtime tape
//! backward the train block's tape-AD arm shares).
//!
//! Moved out of `stmt.rs` whole, byte-for-byte (roadmap A1), after the
//! control-flow and assignment lowerings; the statement dispatch
//! (`stmt.rs::compile_stmt_dispatch`) still calls `compile_grad_block`, and
//! the train block's tape-AD arm still calls `compile_tape_backward`.

use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::InstBuilder;
use cranelift_frontend::FunctionBuilder;

use nsl_ast::expr::ExprKind;

use nsl_ast::pattern::PatternKind;
use nsl_ast::stmt::StmtKind;

use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;

use cranelift_codegen::ir::Value;
impl Compiler<'_> {
    /// Emit the tape-based AD backward pass: tape_start, compile forward,
    /// find loss, tape_backward, tape_stop. Returns `(grads_list, loss_val)`.
    pub(crate) fn compile_tape_backward(
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

    pub(crate) fn compile_grad_block(
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

    pub(crate) fn compile_source_ad_grad_block(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        grad: &nsl_ast::block::GradBlock,
        targets_val: Value,
    ) -> Result<Option<(Value, Value)>, CodegenError> {
        nsl_log::nsl_log!(INFO, "nsl", "[nsl] Using source-to-source AD for grad block");

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
            nsl_log::nsl_log!(WARN, "nsl", 
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
            nsl_log::nsl_log!(WARN, "nsl", 
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
                nsl_log::nsl_log!(WARN, "nsl", 
                    "[nsl] source AD does not yet resolve this grad target shape, falling back to tape-based AD"
                );
                return Ok(None);
            }
        };
        let Some(target_var_id) = target_var_id else {
            nsl_log::nsl_log!(WARN, "nsl", 
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
                        nsl_log::nsl_log!(ERROR, "nsl", 
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

    pub(crate) fn compile_tape_grad_block(
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
}
