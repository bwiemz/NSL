//! The control-flow statement lowerings: `if`, `while`, `while let`,
//! `for` (over ranges and lists, over model arrays, over a DataLoader) and
//! `match`, with the two helpers that materialize non-owning aliases
//! before a branch or a loop so the ownership sweep sees one owner per
//! tensor on every path.
//!
//! Moved out of `stmt.rs` whole, byte-for-byte (roadmap A1): the file was
//! the dispatch, these lowerings, the assignment lowering and the train /
//! grad / quant / distill block drivers in one 10k-line module; the
//! statement dispatch (`stmt.rs::compile_stmt_dispatch`) still calls
//! every entry point here.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::{InstBuilder, MemFlagsData};
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;
use nsl_ast::pattern::PatternKind;
use nsl_semantic::types::Type;
use crate::compiler::Compiler;
use crate::context::{FuncState, LoopContext};
use crate::error::CodegenError;
use crate::types::{is_block_filled, is_float_type};
use cranelift_codegen::ir::Value;

impl Compiler<'_> {
    pub(crate) fn materialize_non_owning_aliases_before_if(
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
    pub(crate) fn materialize_non_owning_aliases_before_loop(
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

    pub(crate) fn compile_if_stmt(
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

    pub(crate) fn compile_while(
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

    pub(crate) fn compile_while_let(
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

    pub(crate) fn compile_for(
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
    pub(crate) fn compile_for_model_array(
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

    // ── DataLoader for-loop ──────────────────────────────────────────

    pub(crate) fn compile_for_dataloader(
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

    pub(crate) fn compile_match(
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
}
