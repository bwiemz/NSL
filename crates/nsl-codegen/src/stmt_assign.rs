//! The assignment lowering and the binding facts it rests on: `compile_assign`
//! (simple, subscript, field, augmented and destructuring targets, with the
//! slab-tensor fast path), the destructuring-pattern lowering and its
//! element / rest / field typing, the pattern-bound-symbol and
//! assignment-target collectors the control-flow lowerings consult, the
//! non-owning-binding update, and the "are all bindings of this symbol
//! owning" facts (`sym_bindings_all_owning_in_*`, `pattern_binds_sym`,
//! `loop_binding_rhs_is_owning`) the ownership sweep and the alias
//! materialization read.
//!
//! Moved out of `stmt.rs` whole, byte-for-byte (roadmap A1), after the
//! control-flow lowerings went to `stmt_control.rs`; the statement dispatch
//! (`stmt.rs::compile_stmt_dispatch`) still calls `compile_assign`.

use cranelift_codegen::ir::condcodes::IntCC;
use cranelift_codegen::ir::types as cl_types;
use cranelift_codegen::ir::InstBuilder;
use cranelift_frontend::FunctionBuilder;
use cranelift_module::Module;

use nsl_ast::expr::{ExprKind, SubscriptKind};
use nsl_ast::operator::AssignOp;
use nsl_ast::pattern::PatternKind;
use nsl_ast::stmt::{Stmt, StmtKind};
use nsl_semantic::types::Type;
use crate::compiler::Compiler;
use crate::context::FuncState;
use crate::error::CodegenError;
use crate::types::is_float_type;
use cranelift_codegen::ir::Value;
impl Compiler<'_> {
    pub(crate) fn collect_pattern_bound_symbols(
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

    pub(crate) fn collect_assignment_targets_from_block(
        &self,
        block: &nsl_ast::stmt::Block,
        targets: &mut std::collections::HashSet<nsl_ast::Symbol>,
    ) {
        for stmt in &block.stmts {
            self.collect_assignment_targets_from_stmt(stmt, targets);
        }
    }

    pub(crate) fn collect_assignment_targets_from_stmt(
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

    pub(crate) fn update_non_owning_binding(
        &self,
        state: &mut FuncState,
        target_sym: nsl_ast::Symbol,
        value: Option<&nsl_ast::expr::Expr>,
    ) {
        let Some(expr) = value else {
            state.non_owning_symbols.remove(&target_sym);
            return;
        };

        if let ExprKind::Ident(source_sym) = &expr.kind
            && (state.param_symbols.contains(source_sym)
                || state.non_owning_symbols.contains(source_sym))
        {
            state.non_owning_symbols.insert(target_sym);
            return;
        }

        // A non-Dict subscript hands out a BORROWED element: `compile_subscript`
        // lowers lists/tuples through `nsl_list_get`, which returns the stored
        // raw pointer with no retain. `let t = items[0]` therefore aliases an
        // element the container still owns, and treating it as owning let the
        // return sweep free a live element. Dict reads are the exception the
        // rest of this file already carves out (`loop_binding_rhs_is_owning`
        // encodes exactly this rule for the loop-rebind free).
        if let ExprKind::Subscript { object, .. } = &expr.kind {
            let is_dict = matches!(
                self.node_type(object.id),
                nsl_semantic::types::Type::Dict(_, _)
            );
            if !is_dict {
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
        if let ExprKind::MemberAccess { object, .. } = &expr.kind
            && matches!(
                self.node_type(object.id),
                nsl_semantic::types::Type::Model { .. }
            )
        {
            state.non_owning_symbols.insert(target_sym);
            return;
        }

        state.non_owning_symbols.remove(&target_sym);
    }

    /// M36: Try to compile a tensor creation as a slab-managed allocation.
    /// Returns Ok(Some(value)) if the variable is slab-planned and the RHS is a
    /// tensor creation (zeros, ones, etc.). Returns Ok(None) to fall through to normal codegen.
    pub(crate) fn try_compile_slab_tensor(
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
    pub(crate) fn destructure_element_type(
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

    pub(crate) fn destructure_rest_type(
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

    pub(crate) fn destructure_field_type(
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

    pub(crate) fn compile_destructure_patterns(
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
                    let var = builder.declare_var(cl_types::I64);
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
                                    let var = builder.declare_var(cl_types::I64);
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
                            let var = builder.declare_var(cl_types::I64);
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
                            let var = builder.declare_var(cl_types::I64);
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
                        let var = builder.declare_var(cl_types::I64);
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

    /// Shared exit for every non-Ident assignment target (dict/list set,
    /// tensor multi-dim set, struct/model field stores, adapter
    /// side-table stores): drain the statement's temporaries with the
    /// just-stored value excluded, mirroring the Ident arm's tail.
    /// These arms previously had NO drain at all (PR #433 review LOW-2),
    /// which broke in two ways:
    ///
    /// - The stored value: an OWNED temp (`d["k"] = t * 2.0`) stayed in
    ///   `tensor_temporaries`, so the NEXT statement's sweep freed it
    ///   while the container still held the raw handle — dict reads
    ///   cloned a freed tensor, list reads handed out the dangling
    ///   pointer itself, and the adapter side-table's free-on-overwrite
    ///   became a double free. `free_tensor_temporaries` DRAINS the
    ///   list (`split_off`) and skips freeing `keep`, so passing the
    ///   stored value as `keep` IS the ownership transfer into the
    ///   container: the handle leaves the sweep's reach unfree'd.
    ///   Nothing is retained for borrowed stores — per the borrow-store
    ///   convention (dict_lifetime.rs) no machinery ever releases a
    ///   container-stored borrow, so a retain would strand one
    ///   reference per store.
    /// - Sub-expression temporaries (`d["k"] = t * 2.0 + 1.0` leaves
    ///   the inner `t * 2.0`) otherwise straddle past the statement;
    ///   a region that frees the temporaries list without draining it —
    ///   the train-block step loop — then freed the straddler once per
    ///   step: double free at step 2.
    pub(crate) fn assign_container_store_tail(
        &mut self,
        builder: &mut FunctionBuilder,
        state: &mut FuncState,
        stored_val: Value,
    ) {
        self.free_tensor_temporaries(builder, state, Some(stored_val));
        self.free_linear_consumes(builder, state, Some(stored_val));
    }

    pub(crate) fn compile_assign(
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
                        // Upgrade Unknown to Owned when the RHS is a call
                        // contractually returning an owning ref — the exact
                        // twin of the Return handler's upgrade above (ELTLS
                        // §6.5). Without it, `x = self.norm.forward(x)`
                        // (model-method results carry no ELTLS registration)
                        // took the conservative retain below and double-owned
                        // the result: the variable's single release left one
                        // reference behind — one stranded block per
                        // assignment, measured as the final-norm strand on
                        // every Coder-50M forward (the `x = ...` twin of the
                        // `return rmsnorm(...)` leak).
                        let mut own = self.get_ownership(state, final_val);
                        if matches!(own, Ownership::Unknown)
                            && self.expr_call_returns_owning_ref(value)
                        {
                            own = Ownership::Owned;
                        }
                        match own {
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
                    // Same capture-count transfer/clear as the VarDecl arm
                    // (review MEDIUM-1 on 44c011c1): a plain `f = <lambda>`
                    // rebind previously left closure_info stale in BOTH
                    // directions — a non-capturing rebind after a capturing
                    // one made the call site read the bare fn pointer as a
                    // closure struct (silent death), and a capturing rebind
                    // was never recorded at all.
                    if let Some(count) = self.registry.last_lambda_capture_count.take() {
                        state.set_closure_info(*sym, Some(count));
                    } else {
                        state.set_closure_info(*sym, None);
                    }
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
                        self.assign_container_store_tail(builder, state, final_val);
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
                        // Stored value is a scalar; the drain covers
                        // index-expression and RHS temporaries.
                        self.assign_container_store_tail(builder, state, final_val);
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
                                        cranelift_codegen::ir::MemFlagsData::trusted(),
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
                                                builder.ins().icmp_imm_s(IntCC::Equal, new_val, 0);
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
                                    cranelift_codegen::ir::MemFlagsData::trusted(),
                                    final_val,
                                    obj_val,
                                    field.offset as i32,
                                );
                                self.assign_container_store_tail(builder, state, final_val);
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
                        if matches!(op, AssignOp::Assign)
                            && let Some(layout) =
                                self.types.struct_layouts.get(&model_name).cloned()
                            && let Some(slot_off) = layout.adapter_sidetable_offset
                        {
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
                                cranelift_codegen::ir::MemFlagsData::trusted(),
                                obj_val,
                                slot_off as i32,
                            );
                            let byte_off = (index * 8) as i32;
                            // Free the existing tensor in the slot
                            // before overwriting (side-table owns
                            // the tensors it holds).
                            let old_ptr = builder.ins().load(
                                cl_types::I64,
                                cranelift_codegen::ir::MemFlagsData::trusted(),
                                table_ptr,
                                byte_off,
                            );
                            self.compile_call_by_name(
                                builder,
                                "nsl_tensor_free_if_valid",
                                &[old_ptr],
                            )?;
                            builder.ins().store(
                                cranelift_codegen::ir::MemFlagsData::trusted(),
                                new_val,
                                table_ptr,
                                byte_off,
                            );
                            // The side-table owns its tensors and
                            // frees the old slot on overwrite — a
                            // swept owned temp here became a later
                            // double free.
                            self.assign_container_store_tail(builder, state, new_val);
                            return Ok(());
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
                                        cranelift_codegen::ir::MemFlagsData::trusted(),
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
                                                builder.ins().icmp_imm_s(IntCC::Equal, new_val, 0);
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
                                    cranelift_codegen::ir::MemFlagsData::trusted(),
                                    final_val,
                                    obj_val,
                                    field.offset as i32,
                                );
                                self.assign_container_store_tail(builder, state, final_val);
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

    /// True when every VarDecl/Assign binding of `sym` in the block —
    /// including nested if/loop/match blocks, which rebind the SAME slot
    /// (this lowering has no block-level scoping) — has an owning RHS.
    pub(crate) fn sym_bindings_all_owning_in_block(
        &self,
        block: &nsl_ast::stmt::Block,
        sym: nsl_ast::Symbol,
    ) -> bool {
        block
            .stmts
            .iter()
            .all(|s| self.sym_bindings_all_owning_in_stmt(s, sym))
    }

    pub(crate) fn pattern_binds_sym(&self, pattern: &nsl_ast::pattern::Pattern, sym: nsl_ast::Symbol) -> bool {
        let mut bound = std::collections::HashSet::new();
        self.collect_pattern_bound_symbols(pattern, &mut bound);
        bound.contains(&sym)
    }

    pub(crate) fn sym_bindings_all_owning_in_stmt(
        &self,
        stmt: &nsl_ast::stmt::Stmt,
        sym: nsl_ast::Symbol,
    ) -> bool {
        use nsl_ast::stmt::StmtKind;
        match &stmt.kind {
            StmtKind::VarDecl { pattern, value, .. } => {
                if let nsl_ast::pattern::PatternKind::Ident(s) = &pattern.kind {
                    if *s == sym {
                        // The RHS must be owning AND tensor-typed: an armed
                        // clear on an INT slot hands the integer to
                        // free_if_valid, whose magic probe dereferences any
                        // 8-aligned value >= 0x10000.
                        return value.as_ref().is_none_or(|e| {
                            self.loop_binding_rhs_is_owning(e)
                                && matches!(self.node_type(e.id),
                                            ty if ty.is_tensor() || ty.is_indeterminate())
                        });
                    }
                    return true;
                }
                // Destructuring patterns (`let (y, z) = pair`) bind shared
                // tuple/list members — non-owning.
                !self.pattern_binds_sym(pattern, sym)
            }
            StmtKind::Assign { target, value, .. } => {
                if let ExprKind::Ident(s) = &target.kind
                    && *s == sym
                {
                    return self.loop_binding_rhs_is_owning(value)
                        && matches!(self.node_type(value.id),
                                    ty if ty.is_tensor() || ty.is_indeterminate());
                }
                true
            }
            StmtKind::If {
                then_block,
                elif_clauses,
                else_block,
                ..
            } => {
                self.sym_bindings_all_owning_in_block(then_block, sym)
                    && elif_clauses
                        .iter()
                        .all(|(_, b)| self.sym_bindings_all_owning_in_block(b, sym))
                    && else_block
                        .as_ref()
                        .is_none_or(|b| self.sym_bindings_all_owning_in_block(b, sym))
            }
            // Loop/while-let/match patterns bind borrowed elements (list
            // members via nsl_list_get, match subjects) into the SAME slot
            // when the name shadows an armed sym — that's a non-owning
            // binding of `sym`.
            StmtKind::For { pattern, body, .. } | StmtKind::WhileLet { pattern, body, .. } => {
                !self.pattern_binds_sym(pattern, sym)
                    && self.sym_bindings_all_owning_in_block(body, sym)
            }
            StmtKind::While { body, .. } => self.sym_bindings_all_owning_in_block(body, sym),
            StmtKind::Match { arms, .. } => arms.iter().all(|arm| {
                !self.pattern_binds_sym(&arm.pattern, sym)
                    && self.sym_bindings_all_owning_in_block(&arm.body, sym)
            }),
            StmtKind::Decorated { stmt, .. } => self.sym_bindings_all_owning_in_stmt(stmt, sym),
            // Opaque block constructs compile their bodies against the SAME
            // FuncState but through their own lowering — this walker cannot
            // see their bindings. Presence of one vetoes the sym.
            StmtKind::TrainBlock(_)
            | StmtKind::GradBlock(_)
            | StmtKind::DistillBlock(_)
            | StmtKind::QuantBlock(_)
            | StmtKind::ServeBlock(_) => false,
            _ => true,
        }
    }

    /// Does this RHS hand back a reference the binding OWNS?
    /// False for the raw-pointer-copy forms: ident references (no retain),
    /// member access (model weights / struct fields share the stored
    /// handle), and non-dict subscripts — compile_subscript lowers
    /// EVERYTHING except Dict (lists, tuples, Unknown) through
    /// nsl_list_get, which shares the stored element with no retain. Only
    /// dict subscripts clone tensors (owned). Match/block expressions do
    /// not retain their arm results (unlike if-expressions, which do), so
    /// they are vetoed too. Calls, method calls, and operators produce
    /// fresh results.
    pub(crate) fn loop_binding_rhs_is_owning(&self, expr: &nsl_ast::expr::Expr) -> bool {
        match &expr.kind {
            ExprKind::Ident(_) => false,
            ExprKind::MemberAccess { .. } => false,
            ExprKind::MatchExpr { .. } => false,
            ExprKind::BlockExpr(_) => false,
            ExprKind::Paren(inner) => self.loop_binding_rhs_is_owning(inner),
            ExprKind::IfExpr {
                then_expr,
                else_expr,
                ..
            } => {
                self.loop_binding_rhs_is_owning(then_expr)
                    && self.loop_binding_rhs_is_owning(else_expr)
            }
            ExprKind::Subscript { object, .. } => matches!(
                self.node_type(object.id),
                nsl_semantic::types::Type::Dict(_, _)
            ),
            _ => true,
        }
    }
}
