use super::*;

impl<'a> TypeChecker<'a> {
    // ===== Statement checking =====

    pub(crate) fn check_stmt(&mut self, stmt: &Stmt) {
        match &stmt.kind {
            StmtKind::VarDecl {
                is_const,
                pattern,
                type_ann,
                value,
            } => self.check_var_decl(*is_const, pattern, type_ann.as_ref(), value.as_ref(), stmt.span),
            StmtKind::FnDef(fn_def) => self.check_fn_def(fn_def),
            StmtKind::ModelDef(model_def) => self.check_model_def(model_def),
            StmtKind::StructDef(struct_def) => self.check_struct_def(struct_def),
            StmtKind::EnumDef(enum_def) => self.check_enum_def(enum_def),
            StmtKind::TraitDef(_) => {} // Deferred
            StmtKind::If {
                condition,
                then_block,
                elif_clauses,
                else_block,
            } => {
                self.check_expr(condition);
                self.check_block(then_block, ScopeKind::Block);
                for (cond, block) in elif_clauses {
                    self.check_expr(cond);
                    self.check_block(block, ScopeKind::Block);
                }
                if let Some(else_block) = else_block {
                    self.check_block(else_block, ScopeKind::Block);
                }
            }
            StmtKind::For {
                pattern,
                iterable,
                body,
            } => {
                let iter_ty = self.check_expr(iterable);
                let elem_ty = match &iter_ty {
                    Type::List(elem) => *elem.clone(),
                    Type::Dict(key, _val) => *key.clone(),
                    Type::Str => Type::Str,
                    Type::Tuple(elems) => {
                        // Iterating over a tuple: element type is union or first element
                        elems.first().cloned().unwrap_or(Type::Unknown)
                    }
                    Type::FixedModelArray { element_model, .. } => {
                        Type::Model {
                            name: *element_model,
                            fields: Vec::new(),
                            methods: Vec::new(),
                        }
                    }
                    _ => Type::Unknown,
                };
                let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Loop);
                let prev = self.current_scope;
                self.current_scope = scope;
                self.declare_pattern(pattern, &elem_ty);
                for s in &body.stmts {
                    self.check_stmt(s);
                }
                self.current_scope = prev;
            }
            StmtKind::While { condition, body } => {
                self.check_expr(condition);
                self.check_block(body, ScopeKind::Loop);
            }
            StmtKind::WhileLet {
                pattern,
                expr,
                body,
            } => {
                let ty = self.check_expr(expr);
                let inner_ty = match &ty {
                    Type::Optional(inner) => *inner.clone(),
                    _ => ty,
                };
                let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Loop);
                let prev = self.current_scope;
                self.current_scope = scope;
                self.declare_pattern(pattern, &inner_ty);
                for s in &body.stmts {
                    self.check_stmt(s);
                }
                self.current_scope = prev;
            }
            StmtKind::Match { subject, arms } => {
                let subject_ty = self.check_expr(subject);
                for arm in arms {
                    let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Block);
                    let prev = self.current_scope;
                    self.current_scope = scope;
                    self.declare_pattern(&arm.pattern, &subject_ty);
                    if let Some(guard) = &arm.guard {
                        self.check_expr(guard);
                    }
                    for s in &arm.body.stmts {
                        self.check_stmt(s);
                    }
                    self.current_scope = prev;
                }
            }
            StmtKind::Return(expr) => {
                if self.scopes.enclosing_function(self.current_scope).is_none() {
                    self.diagnostics.push(
                        Diagnostic::error("`return` outside of function")
                            .with_label(stmt.span, "not inside a function"),
                    );
                }
                if let Some(expr) = expr {
                    let ty = self.check_expr(expr);
                    if let Some(expected) = &self.current_return_type {
                        if !is_assignable(&ty, expected) {
                            self.diagnostics.push(
                                Diagnostic::error(format!(
                                    "return type mismatch: expected {}, got {}",
                                    display_type(expected), display_type(&ty)
                                ))
                                .with_label(expr.span, "wrong type"),
                            );
                        }
                    }
                }
            }
            StmtKind::Break => {
                if !self.scopes.is_in_loop(self.current_scope) {
                    self.diagnostics.push(
                        Diagnostic::error("`break` outside of loop")
                            .with_label(stmt.span, "not inside a loop"),
                    );
                }
            }
            StmtKind::Continue => {
                if !self.scopes.is_in_loop(self.current_scope) {
                    self.diagnostics.push(
                        Diagnostic::error("`continue` outside of loop")
                            .with_label(stmt.span, "not inside a loop"),
                    );
                }
            }
            StmtKind::Yield(expr) => {
                if let Some(expr) = expr {
                    self.check_expr(expr);
                }
            }
            StmtKind::Assign { target, op, value } => {
                let target_ty = self.check_expr(target);
                let value_ty = self.check_expr(value);

                // Check const violation
                if let ExprKind::Ident(sym) = &target.kind {
                    if let Some((_sid, info)) = self.scopes.lookup(self.current_scope, *sym) {
                        if info.is_const {
                            let name = self.resolve_name(*sym);
                            self.diagnostics.push(
                                Diagnostic::error(format!(
                                    "cannot assign to `{name}`: declared as const"
                                ))
                                .with_label(target.span, "const binding")
                                .with_label(info.def_span, "declared here"),
                            );
                        }
                    }
                }

                // Check type compatibility
                if *op == AssignOp::Assign {
                    if !is_assignable(&value_ty, &target_ty) && !target_ty.is_indeterminate() {
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "type mismatch in assignment: expected {}, got {}",
                                display_type(&target_ty), display_type(&value_ty)
                            ))
                            .with_label(value.span, "wrong type"),
                        );
                    }
                } else {
                    // Compound assignment (+=, -=, *=, /=): both sides must be numeric
                    let is_numeric = |ty: &Type| matches!(ty,
                        Type::Int | Type::Int64 | Type::Int32 | Type::Int16 | Type::Int8
                        | Type::Uint8 | Type::Float | Type::F64 | Type::F32
                        | Type::Tensor { .. } | Type::Unknown
                    );
                    if (!is_numeric(&target_ty) || !is_numeric(&value_ty))
                        && !target_ty.is_indeterminate() && !value_ty.is_indeterminate() {
                            self.diagnostics.push(
                                Diagnostic::error(format!(
                                    "invalid operand types for compound assignment: {} and {}",
                                    display_type(&target_ty), display_type(&value_ty)
                                ))
                                .with_label(value.span, "invalid type for compound assignment"),
                            );
                    }
                }
            }
            StmtKind::Import(_) | StmtKind::FromImport(_) => {
                // Already processed in collect_top_level_decls
            }
            StmtKind::Expr(expr) => {
                self.check_expr(expr);
            }
            StmtKind::Decorated { decorators, stmt } => {
                // Validate @test decorator constraints
                for deco in decorators {
                    if deco.name.len() == 1 {
                        let dname = self.interner.resolve(deco.name[0].0).unwrap_or("").to_string();
                        if dname == "test" {
                            // @test must decorate a function
                            if let StmtKind::FnDef(fn_def) = &stmt.kind {
                                // @test functions must have no parameters
                                if !fn_def.params.is_empty() {
                                    self.diagnostics.push(
                                        Diagnostic::error("@test function must have no parameters")
                                            .with_label(fn_def.span, "has parameters"),
                                    );
                                }
                                // @test functions must have no return type
                                if fn_def.return_type.is_some() {
                                    self.diagnostics.push(
                                        Diagnostic::error("@test function must not have a return type")
                                            .with_label(fn_def.span, "has return type"),
                                    );
                                }
                            } else {
                                self.diagnostics.push(
                                    Diagnostic::error("@test can only decorate a function definition")
                                        .with_label(deco.span, "invalid @test target"),
                                );
                            }
                        }

                        if dname == "fuse" {
                            match &stmt.kind {
                                StmtKind::FnDef(_) => {
                                    // Valid target — further validation in codegen
                                }
                                StmtKind::KernelDef(_) => {
                                    self.diagnostics.push(
                                        Diagnostic::error("@fuse cannot be applied to kernel blocks (kernel blocks are already single PTX kernels)")
                                            .with_label(deco.span, "invalid @fuse target")
                                    );
                                }
                                StmtKind::ModelDef(_) => {
                                    self.diagnostics.push(
                                        Diagnostic::error("@fuse cannot be applied to model methods; extract the fusible logic into a standalone fn")
                                            .with_label(deco.span, "invalid @fuse target")
                                    );
                                }
                                _ => {
                                    self.diagnostics.push(
                                        Diagnostic::error("@fuse can only be applied to fn declarations")
                                            .with_label(deco.span, "invalid @fuse target")
                                    );
                                }
                            }
                        }

                        if dname == "fuse_graph" {
                            match &stmt.kind {
                                StmtKind::FnDef(_) => {
                                    // Valid target — check for conflicting @fuse on the same fn
                                    let has_fuse = decorators.iter().any(|d| {
                                        d.name.len() == 1
                                            && self.interner.resolve(d.name[0].0).unwrap_or("") == "fuse"
                                    });
                                    if has_fuse {
                                        self.diagnostics.push(
                                            Diagnostic::error("@fuse_graph and @fuse cannot both be applied to the same function; use one or the other")
                                                .with_label(deco.span, "cannot combine @fuse_graph with @fuse")
                                        );
                                    }
                                }
                                StmtKind::KernelDef(_) => {
                                    self.diagnostics.push(
                                        Diagnostic::error("@fuse_graph cannot be applied to kernel blocks")
                                            .with_label(deco.span, "invalid @fuse_graph target")
                                    );
                                }
                                StmtKind::ModelDef(_) => {
                                    self.diagnostics.push(
                                        Diagnostic::error("@fuse_graph cannot be applied to model definitions")
                                            .with_label(deco.span, "invalid @fuse_graph target")
                                    );
                                }
                                _ => {
                                    self.diagnostics.push(
                                        Diagnostic::error("@fuse_graph can only be applied to fn declarations")
                                            .with_label(deco.span, "invalid @fuse_graph target")
                                    );
                                }
                            }
                        }

                        if dname == "no_fuse" {
                            match &stmt.kind {
                                StmtKind::VarDecl { .. } => {
                                    // Valid target — marks this let-binding as a fusion barrier
                                }
                                _ => {
                                    self.diagnostics.push(
                                        Diagnostic::error("@no_fuse can only be applied to let-bindings (use it to mark individual operations as fusion barriers)")
                                            .with_label(deco.span, "invalid @no_fuse target")
                                    );
                                }
                            }
                        }

                        // M38a: @shared annotation — valid on let-bindings
                        if dname == "shared" {
                            match &stmt.kind {
                                StmtKind::VarDecl { .. } => {
                                    // Valid — tensor will be marked Shared in ownership pass
                                }
                                _ => {
                                    self.diagnostics.push(
                                        Diagnostic::error("@shared can only be applied to let-bindings")
                                            .with_label(deco.span, "invalid @shared target")
                                    );
                                }
                            }
                        }

                        // M39: @vmap decorator validation
                        if dname == "vmap" {
                            let resolve = |s: nsl_ast::Symbol| -> String {
                                self.interner
                                    .resolve(s.0)
                                    .unwrap_or("")
                                    .to_string()
                            };
                            crate::vmap::validate_vmap_decorator(
                                deco,
                                &resolve,
                                &mut self.diagnostics,
                            );
                        }

                        // M45: @no_trace and @trace_breakpoint — valid on FnDef only
                        if dname == "no_trace" || dname == "trace_breakpoint" {
                            // These decorators are valid on FnDef only — no args needed
                            // No validation logic required (just recognized and passed through)
                        }

                        // M46: @deterministic — recognized here, validation in determinism.rs pass
                        if dname == "deterministic" {
                            // Recognized — validation happens in determinism.rs pass
                        }

                        // M51: @pure — recognized here, effect validation in effects.rs pass
                        if dname == "pure" {
                            // Recognized — effect validation in effects.rs pass
                        }

                        // M51: @checkpoint — recognized here, requires @pure (validated by effects.rs)
                        if dname == "checkpoint" {
                            // Recognized — requires @pure (validated by effects.rs)
                        }

                        // M49: @shape_assert — recognized here, constraints added to solver during semantic analysis
                        if dname == "shape_assert" {
                            // Recognized — constraint added to solver during semantic analysis
                        }

                        if dname == "flash_attention" {
                            match &stmt.kind {
                                StmtKind::FnDef(_) => {
                                    // Valid target — validate optional args
                                    if let Some(ref args) = deco.args {
                                        for arg in args {
                                            if let Some(ref name_sym) = arg.name {
                                                let aname = self.interner.resolve(name_sym.0).unwrap_or("").to_string();
                                                if aname == "causal" {
                                                    if !matches!(arg.value.kind, ExprKind::BoolLiteral(_)) {
                                                        self.diagnostics.push(
                                                            Diagnostic::error("@flash_attention 'causal' argument must be a bool literal")
                                                                .with_label(arg.span, "expected bool")
                                                        );
                                                    }
                                                } else {
                                                    self.diagnostics.push(
                                                        Diagnostic::error(format!("@flash_attention unknown argument '{}'", aname))
                                                            .with_label(arg.span, "unknown argument")
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                                _ => {
                                    self.diagnostics.push(
                                        Diagnostic::error("@flash_attention can only be applied to fn declarations")
                                            .with_label(deco.span, "invalid @flash_attention target")
                                    );
                                }
                            }
                        }

                        if dname == "rope" {
                            // @rope requires @flash_attention on the same function
                            let has_flash = decorators.iter().any(|d| {
                                d.name.len() == 1 && self.interner.resolve(d.name[0].0).unwrap_or("") == "flash_attention"
                            });
                            if !has_flash {
                                self.diagnostics.push(
                                    Diagnostic::error("@rope requires @flash_attention on the same function")
                                        .with_label(deco.span, "missing @flash_attention")
                                );
                            }
                            // Validate optional args
                            if let Some(ref args) = deco.args {
                                for arg in args {
                                    if let Some(ref name_sym) = arg.name {
                                        let aname = self.interner.resolve(name_sym.0).unwrap_or("").to_string();
                                        if aname == "style" {
                                            if let ExprKind::StringLiteral(s) = &arg.value.kind {
                                                if s != "half_split" && s != "adjacent" {
                                                    self.diagnostics.push(
                                                        Diagnostic::error("@rope 'style' must be \"half_split\" or \"adjacent\"")
                                                            .with_label(arg.span, "invalid style")
                                                    );
                                                }
                                            } else {
                                                self.diagnostics.push(
                                                    Diagnostic::error("@rope 'style' argument must be a string literal")
                                                        .with_label(arg.span, "expected string")
                                                );
                                            }
                                        } else {
                                            self.diagnostics.push(
                                                Diagnostic::error(format!("@rope unknown argument '{}'", aname))
                                                    .with_label(arg.span, "unknown argument")
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        if dname == "gqa" {
                            // @gqa requires @flash_attention on the same function
                            let has_flash = decorators.iter().any(|d| {
                                d.name.len() == 1 && self.interner.resolve(d.name[0].0).unwrap_or("") == "flash_attention"
                            });
                            if !has_flash {
                                self.diagnostics.push(
                                    Diagnostic::error("@gqa requires @flash_attention on the same function")
                                        .with_label(deco.span, "missing @flash_attention")
                                );
                            }
                            // Validate required 'groups' arg
                            if let Some(ref args) = deco.args {
                                let mut found_groups = false;
                                for arg in args {
                                    if let Some(ref name_sym) = arg.name {
                                        let aname = self.interner.resolve(name_sym.0).unwrap_or("").to_string();
                                        if aname == "groups" {
                                            found_groups = true;
                                            if let ExprKind::IntLiteral(n) = &arg.value.kind {
                                                if *n <= 0 {
                                                    self.diagnostics.push(
                                                        Diagnostic::error("@gqa 'groups' must be a positive integer")
                                                            .with_label(arg.span, "must be > 0")
                                                    );
                                                }
                                            } else {
                                                self.diagnostics.push(
                                                    Diagnostic::error("@gqa 'groups' argument must be an integer literal")
                                                        .with_label(arg.span, "expected integer")
                                                );
                                            }
                                        } else {
                                            self.diagnostics.push(
                                                Diagnostic::error(format!("@gqa unknown argument '{}'", aname))
                                                    .with_label(arg.span, "unknown argument")
                                            );
                                        }
                                    }
                                }
                                if !found_groups {
                                    self.diagnostics.push(
                                        Diagnostic::error("@gqa requires 'groups' argument")
                                            .with_label(deco.span, "missing 'groups'")
                                    );
                                }
                            } else {
                                self.diagnostics.push(
                                    Diagnostic::error("@gqa requires 'groups' argument")
                                        .with_label(deco.span, "missing 'groups'")
                                );
                            }
                        }

                        if dname == "autotune" {
                            match &stmt.kind {
                                StmtKind::KernelDef(_) => {
                                    // Valid target — validate args are lists of integers
                                    if let Some(ref args) = deco.args {
                                        for arg in args {
                                            if let Some(ref name_sym) = arg.name {
                                                let _aname = self.interner.resolve(name_sym.0).unwrap_or("").to_string();
                                                // Each arg value must be a list literal of integers
                                                match &arg.value.kind {
                                                    ExprKind::ListLiteral(items) => {
                                                        for item in items {
                                                            if !matches!(item.kind, ExprKind::IntLiteral(_)) {
                                                                self.diagnostics.push(
                                                                    Diagnostic::error("@autotune parameter values must be integer literals")
                                                                        .with_label(item.span, "expected integer")
                                                                );
                                                            }
                                                        }
                                                    }
                                                    _ => {
                                                        self.diagnostics.push(
                                                            Diagnostic::error("@autotune parameters must be lists of integers (e.g., [64, 128, 256])")
                                                                .with_label(arg.span, "expected list")
                                                        );
                                                    }
                                                }
                                            }
                                        }
                                    } else {
                                        self.diagnostics.push(
                                            Diagnostic::error("@autotune requires at least one tuning parameter")
                                                .with_label(deco.span, "missing parameters")
                                        );
                                    }
                                }
                                StmtKind::FnDef(_) => {
                                    // Valid if @flash_attention is also present
                                    let has_flash = decorators.iter().any(|d| {
                                        d.name.len() == 1 && self.interner.resolve(d.name[0].0).unwrap_or("") == "flash_attention"
                                    });
                                    if !has_flash {
                                        self.diagnostics.push(
                                            Diagnostic::error("@autotune on fn requires @flash_attention")
                                                .with_label(deco.span, "requires @flash_attention")
                                        );
                                    } else {
                                        // Validate args same as kernel blocks
                                        if let Some(ref args) = deco.args {
                                            for arg in args {
                                                if let Some(ref name_sym) = arg.name {
                                                    let _aname = self.interner.resolve(name_sym.0).unwrap_or("").to_string();
                                                    match &arg.value.kind {
                                                        ExprKind::ListLiteral(items) => {
                                                            for item in items {
                                                                if !matches!(item.kind, ExprKind::IntLiteral(_)) {
                                                                    self.diagnostics.push(
                                                                        Diagnostic::error("@autotune parameter values must be integer literals")
                                                                            .with_label(item.span, "expected integer")
                                                                    );
                                                                }
                                                            }
                                                        }
                                                        _ => {
                                                            self.diagnostics.push(
                                                                Diagnostic::error("@autotune parameters must be lists of integers (e.g., [64, 128, 256])")
                                                                    .with_label(arg.span, "expected list")
                                                            );
                                                        }
                                                    }
                                                }
                                            }
                                        } else {
                                            self.diagnostics.push(
                                                Diagnostic::error("@autotune requires at least one tuning parameter")
                                                    .with_label(deco.span, "missing parameters")
                                            );
                                        }
                                    }
                                }
                                _ => {
                                    self.diagnostics.push(
                                        Diagnostic::error("@autotune can only be applied to kernel blocks or @flash_attention functions")
                                            .with_label(deco.span, "invalid @autotune target")
                                    );
                                }
                            }
                        }
                    }
                }
                self.check_stmt(stmt);
            }
            // ML blocks: walk children for name resolution but defer validation
            StmtKind::TrainBlock(train) => self.check_train_block(train),
            StmtKind::GradBlock(grad) => {
                self.check_expr(&grad.targets);
                self.check_block(&grad.body, ScopeKind::Block);

                // If there are output bindings, declare them in the enclosing scope
                // For `let (loss, grads) = grad(w):`, loss is a scalar tensor, grads is a tensor
                if let Some(ref pattern) = grad.outputs {
                    let unknown_tensor = Type::Tensor {
                        shape: crate::types::Shape::unknown(),
                        dtype: crate::types::DType::Unknown,
                        device: crate::types::Device::Unknown,
                    };
                    match &pattern.kind {
                        PatternKind::Tuple(pats) => {
                            for p in pats {
                                if let PatternKind::Ident(sym) = &p.kind {
                                    self.declare_symbol(*sym, unknown_tensor.clone(), p.span, false, true);
                                }
                            }
                        }
                        PatternKind::Ident(sym) => {
                            self.declare_symbol(*sym, unknown_tensor, pattern.span, false, true);
                        }
                        _ => {}
                    }
                }
            }
            StmtKind::QuantBlock(quant) => {
                // Validate source is a known model variable
                let source_ty = if let Some((_sid, info)) =
                    self.scopes.lookup(self.current_scope, quant.source)
                {
                    info.ty.clone()
                } else {
                    let source_name = self.interner.resolve(quant.source.0).unwrap_or("<unknown>").to_string();
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "quant source '{source_name}' is not defined"
                        ))
                        .with_label(quant.span, "undefined source"),
                    );
                    Type::Error
                };

                // Check source is a Model type
                let model_fields = match &source_ty {
                    Type::Model { fields, .. } => fields.clone(),
                    Type::Error | Type::Unknown => Vec::new(),
                    _ => {
                        let source_name = self.interner.resolve(quant.source.0).unwrap_or("<unknown>").to_string();
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "quant source '{source_name}' is not a model type"
                            ))
                            .with_label(quant.span, "expected model type"),
                        );
                        Vec::new()
                    }
                };

                // Resolve exclude globs against model field names
                let excluded_fields: Vec<String> = if !quant.exclude.is_empty() {
                    let field_names: Vec<String> = model_fields
                        .iter()
                        .map(|(sym, _)| {
                            self.interner.resolve(sym.0).unwrap_or("").to_string()
                        })
                        .collect();
                    field_names
                        .iter()
                        .filter(|name| {
                            quant.exclude.iter().any(|pattern| glob_match(pattern, name))
                        })
                        .cloned()
                        .collect()
                } else {
                    Vec::new()
                };

                // Validate calibration data variable if present
                if let Some(ref cal) = quant.calibration {
                    if self.scopes.lookup(self.current_scope, cal.data).is_none() {
                        let data_name = self.interner.resolve(cal.data.0).unwrap_or("<unknown>").to_string();
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "calibration data variable '{data_name}' is not defined"
                            ))
                            .with_label(quant.span, "undefined calibration data"),
                        );
                    }
                }

                // Build quantized model type: clone fields, replace tensor types
                // for non-excluded fields with QuantizedTensor
                let quantized_fields: Vec<(nsl_ast::Symbol, Type)> = model_fields
                    .iter()
                    .map(|(sym, ty)| {
                        let field_name = self.interner.resolve(sym.0).unwrap_or("").to_string();
                        let is_excluded = excluded_fields.contains(&field_name);
                        if !is_excluded && ty.is_tensor() {
                            (*sym, Type::QuantizedTensor)
                        } else {
                            (*sym, ty.clone())
                        }
                    })
                    .collect();

                // Get the source model's methods
                let model_methods = match &source_ty {
                    Type::Model { methods, .. } => methods.clone(),
                    _ => Vec::new(),
                };

                // Register output variable with quantized Model type
                let quant_model_ty = Type::Model {
                    name: quant.name,
                    fields: quantized_fields,
                    methods: model_methods,
                };
                self.declare_symbol(quant.name, quant_model_ty, quant.span, true, false);
            }
            StmtKind::KernelDef(kernel) => {
                self.declare_symbol(kernel.name, Type::Unknown, kernel.span, true, false);
                // Push kernel scope and register params (like fn_def)
                let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Function);
                let prev_scope = self.current_scope;
                self.current_scope = scope;
                for param in &kernel.params {
                    self.declare_symbol(param.name, Type::Unknown, param.span, false, true);
                }
                for s in &kernel.body.stmts {
                    self.check_stmt(s);
                }
                self.current_scope = prev_scope;
            }
            StmtKind::TokenizerDef(tok) => {
                // Declare the tokenizer name; body validation deferred to M3.
                self.declare_symbol(tok.name, Type::Unknown, tok.span, true, false);
            }
            StmtKind::DatasetDef(ds) => {
                // Declare the dataset name; body validation deferred to M3.
                // Dataset body uses DSL field assignments (source=, packing=, etc.)
                // that aren't general variable assignments.
                self.declare_symbol(ds.name, Type::Unknown, ds.span, true, false);
            }
            StmtKind::DatatypeDef(def) => {
                if def.bits.is_none() {
                    self.diagnostics.push(
                        Diagnostic::error("datatype block must declare 'bits'")
                            .with_label(def.span, "missing 'bits' declaration"),
                    );
                }

                let has_pack = def.methods.iter().any(|m| m.kind == DatatypeMethodKind::Pack);
                let has_unpack = def.methods.iter().any(|m| m.kind == DatatypeMethodKind::Unpack);
                if !has_pack {
                    self.diagnostics.push(
                        Diagnostic::error("datatype block must define @pack method")
                            .with_label(def.span, "missing @pack"),
                    );
                }
                if !has_unpack {
                    self.diagnostics.push(
                        Diagnostic::error("datatype block must define @unpack method")
                            .with_label(def.span, "missing @unpack"),
                    );
                }

                if let Some(bs) = def.block_size {
                    if bs == 0 {
                        self.diagnostics.push(
                            Diagnostic::error("block_size must be > 0")
                                .with_label(def.span, "zero block_size"),
                        );
                    }
                }

                let id = DTYPE_CUSTOM_START + self.custom_datatypes.len() as u16;
                let name = self.interner.resolve(def.name.0).unwrap_or("?").to_string();
                self.custom_datatypes.insert(name, CustomDtypeSemanticInfo {
                    dtype_id: id,
                    bit_width: def.bits.unwrap_or(8),
                    block_size: def.block_size,
                    has_pack,
                    has_unpack,
                    has_pack_ptx: def.ptx_blocks.iter().any(|b| b.kind == DatatypePtxKind::PackPtx),
                    has_unpack_ptx: def.ptx_blocks.iter().any(|b| b.kind == DatatypePtxKind::UnpackPtx),
                });
            }
            StmtKind::ServeBlock(serve) => self.check_serve_block(serve),
        }
    }
}
