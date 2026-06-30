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
            StmtKind::TraitDef(trait_def) => self.check_trait_def(trait_def),
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
                            type_params: Vec::new(),
                            type_args: Vec::new(),
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

                        // Dev Tools Phase 5: @inspect decorator validation
                        if dname == "inspect" {
                            let resolve = |s: nsl_ast::Symbol| -> String {
                                self.interner.resolve(s.0).unwrap_or("").to_string()
                            };
                            crate::inspect::validate_inspect_decorator(
                                deco,
                                &resolve,
                                &mut self.diagnostics,
                            );
                        }

                        // CFIE: @cfie decorator validation
                        if dname == "cfie" {
                            let resolve = |s: nsl_ast::Symbol| -> String {
                                self.interner.resolve(s.0).unwrap_or("").to_string()
                            };
                            crate::cfie::validate_cfie_decorator(
                                deco,
                                &resolve,
                                &mut self.diagnostics,
                            );
                        }

                        // CPDT: @cpdt decorator validation. Phase 1 requires
                        // exactly one @cpdt decorator per program; a second
                        // occurrence emits a diagnostic error referencing
                        // both spans. Only the FIRST decorator runs through
                        // validate_cpdt_decorator — running it on the
                        // duplicate would produce noise alongside the
                        // single-instance error. See
                        // docs/superpowers/specs/2026-04-20-cpdt-weight-aware-opt-out-design.md.
                        if dname == "cpdt" {
                            if let Some(prev_span) = self.cpdt_decorator_span {
                                self.diagnostics.push(
                                    Diagnostic::error(
                                        "@cpdt may appear at most once per program (Phase 1 restriction)".to_string(),
                                    )
                                    .with_label(deco.span, "duplicate @cpdt decorator")
                                    .with_label(prev_span, "previous @cpdt decorator here"),
                                );
                            } else {
                                self.cpdt_decorator_span = Some(deco.span);
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner.resolve(s.0).unwrap_or("").to_string()
                                };
                                crate::cpdt::validate_cpdt_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                            }
                        }

                        // CEP: @cep_prune / @cep_search decorator validation
                        if dname == "cep_prune" {
                            let resolve = |s: nsl_ast::Symbol| -> String {
                                self.interner.resolve(s.0).unwrap_or("").to_string()
                            };
                            crate::cep::validate_cep_prune_decorator(
                                deco,
                                &resolve,
                                &mut self.diagnostics,
                            );
                        }
                        if dname == "cep_search" {
                            let resolve = |s: nsl_ast::Symbol| -> String {
                                self.interner.resolve(s.0).unwrap_or("").to_string()
                            };
                            crate::cep::validate_cep_search_decorator(
                                deco,
                                &resolve,
                                &mut self.diagnostics,
                            );
                        }

                        // CSHA: @csha decorator validation.
                        //
                        // Sprint 2 (paper §6.2 binding fix): the validator
                        // returns a `CshaConfig` describing the per-model
                        // `level=`, `target=`, and `disable=` requests. Before
                        // Sprint 2 the result was dropped on the floor, so
                        // `@csha(disable=true)` parsed cleanly but had zero
                        // codegen effect. Now we capture it keyed by the
                        // decorated model's name (or the LHS binding name for
                        // `@csha let m = SomeModel()`), and the CLI forwards
                        // the side-table into `CompileOptions.csha_configs`
                        // so the CSHA hook in `nsl-codegen/src/stmt.rs` can
                        // look it up by `model_type_name` and:
                        //   * `disabled=true`  -> skip the CSHA pipeline.
                        //   * `level=Some(L)`  -> clamp the planner's mode.
                        //   * `target=Some(T)` -> override the planner target.
                        if dname == "csha" {
                            let resolve = |s: nsl_ast::Symbol| -> String {
                                self.interner.resolve(s.0).unwrap_or("").to_string()
                            };
                            if let Some(cfg) = crate::csha::validate_csha_decorator(
                                deco,
                                &resolve,
                                &mut self.diagnostics,
                            ) {
                                // Resolve the model-or-binding name we want
                                // to key the config by. Two shapes are
                                // supported:
                                //   1. `@csha(...) model Foo: ...`
                                //   2. `@csha(...) let m = SomeModel()`
                                // Case 2 keys by the LHS binding name (e.g.
                                // "m") which is what the codegen hook will
                                // see as `model_type_name` only when the
                                // semantic type maps `m` back to its struct.
                                // To keep this side-table useful in both
                                // cases we record whichever name we can
                                // recover here, and codegen does a HashMap
                                // lookup either way.
                                let model_name: Option<String> = match &stmt.kind {
                                    StmtKind::ModelDef(model_def) => Some(
                                        self.interner
                                            .resolve(model_def.name.0)
                                            .unwrap_or("")
                                            .to_string(),
                                    ),
                                    StmtKind::VarDecl { pattern, .. } => {
                                        if let nsl_ast::pattern::PatternKind::Ident(sym)
                                            = &pattern.kind
                                        {
                                            Some(
                                                self.interner
                                                    .resolve(sym.0)
                                                    .unwrap_or("")
                                                    .to_string(),
                                            )
                                        } else {
                                            None
                                        }
                                    }
                                    _ => None,
                                };
                                if let Some(name) = model_name {
                                    if !name.is_empty() {
                                        self.csha_configs.push((name, cfg));
                                    }
                                }
                                // Silently drop on non-model / non-binding
                                // targets: the validator already accepted the
                                // decorator on a syntactically valid form,
                                // and the existing per-target diagnostics
                                // (e.g. "@csha can only be applied to a
                                // model") would be a follow-up to add.
                            }
                        }

                        // WGGO: @wggo decorator validation
                        if dname == "wggo" {
                            let resolve = |s: nsl_ast::Symbol| -> String {
                                self.interner.resolve(s.0).unwrap_or("").to_string()
                            };
                            crate::wggo::validate_wggo_decorator(
                                deco,
                                &resolve,
                                &mut self.diagnostics,
                            );
                        }

                        // CFTP: @fase and @pca decorator validation
                        if dname == "fase" {
                            let resolve = |s: nsl_ast::Symbol| -> String {
                                self.interner.resolve(s.0).unwrap_or("").to_string()
                            };
                            crate::cftp::validate_fase_decorator(
                                deco,
                                &resolve,
                                &mut self.diagnostics,
                            );
                        }
                        if dname == "pca" {
                            let resolve = |s: nsl_ast::Symbol| -> String {
                                self.interner.resolve(s.0).unwrap_or("").to_string()
                            };
                            crate::cftp::validate_pca_decorator(
                                deco,
                                &resolve,
                                &mut self.diagnostics,
                            );
                        }
                        // CFTP §4.4 G3: @fused_lm_ce decorator validation.
                        // Validated configs are captured so codegen can
                        // later substitute composite cross_entropy with the
                        // fused linear-CE kernel. Sprint 2 (PR #226 follow-on)
                        // wires only the decorator collection + plumbing; the
                        // lowering-site auto-substitution is deferred to
                        // Sprint 2.5 (see docs/superpowers/specs/2026-04-20-cftp-v2-followon.md
                        // and the deferral note in
                        // crates/nsl-codegen/src/wengert_lower.rs near
                        // PrimalOp::CrossEntropyLoss).
                        //
                        // Enforces the design rule that @fused_lm_ce attaches
                        // only to a `train` block — applying it elsewhere
                        // produces a noise-free single error (matches the
                        // pattern used by @freeze / @adapter above).
                        if dname == "fused_lm_ce" {
                            if !matches!(&stmt.kind, StmtKind::TrainBlock(_)) {
                                self.diagnostics.push(
                                    Diagnostic::error(
                                        "@fused_lm_ce may only be applied to a `train` block"
                                            .to_string(),
                                    )
                                    .with_label(deco.span, "invalid @fused_lm_ce target"),
                                );
                            } else if !self.fused_ce_configs.is_empty() {
                                // CFTP v5 follow-on Finding 1 (HIGH): codegen reads
                                // `compiler.fused_ce_configs.first()` for every train-block
                                // lowering via `fused_ce_dtype_for_compiler`.  A second
                                // `@fused_lm_ce` in the same compilation unit therefore
                                // gets SILENTLY ignored — every train block, including
                                // the second one, sees the FIRST decorator's dtype hint.
                                // This is the silent-corruption gap from the adversarial
                                // review: a user who writes
                                //   @fused_lm_ce(dtype="fp16") train(...) ...
                                //   @fused_lm_ce(dtype="bf16") train(...) ...
                                // would get fp16 PTX + fp16 dtype_tag for BOTH train
                                // blocks, while the second block's HBM is bf16-shaped.
                                //
                                // Per the `feedback_deferral_must_refuse` invariant we
                                // refuse rather than silently weaken.  Lifting this
                                // limitation requires either (a) per-train-block
                                // dispatch keyed by AST node id, or (b) explicit
                                // dispatch-index plumbing — both are v6+ ladder steps.
                                self.diagnostics.push(
                                    Diagnostic::error(
                                        "@fused_lm_ce: at most one decorator is allowed per \
                                         compilation unit (v5 codegen uses a single \
                                         compiler-wide dtype hint; a second decorator \
                                         would be silently ignored). Merge train blocks \
                                         or remove the duplicate."
                                            .to_string(),
                                    )
                                    .with_label(deco.span, "duplicate @fused_lm_ce decorator"),
                                );
                            } else {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner.resolve(s.0).unwrap_or("").to_string()
                                };
                                if let Some(cfg) = crate::cftp::validate_fused_ce_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                ) {
                                    self.fused_ce_configs.push(cfg);
                                }
                            }
                        }

                        // WRGA: @wrga / @freeze / @adapter decorator validation.
                        // Validated configs are captured so codegen's
                        // `wrga::run` driver can consume them later.
                        if dname == "wrga" {
                            let resolve = |s: nsl_ast::Symbol| -> String {
                                self.interner.resolve(s.0).unwrap_or("").to_string()
                            };
                            if let Some(cfg) = crate::wrga::validate_wrga_decorator(
                                deco,
                                &resolve,
                                &mut self.diagnostics,
                            ) {
                                self.wrga_configs.push(cfg);
                            }
                        }
                        if dname == "freeze" {
                            if !is_model_target(&stmt.kind) {
                                self.diagnostics.push(
                                    Diagnostic::error("@freeze can only be applied to a model (or a let-binding whose RHS is a model)")
                                        .with_label(deco.span, "invalid @freeze target"),
                                );
                            } else {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner.resolve(s.0).unwrap_or("").to_string()
                                };
                                let cfg = crate::wrga::validate_freeze_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                                self.freeze_configs.push(cfg);
                            }
                        }
                        if dname == "adapter" {
                            if !is_model_target(&stmt.kind) {
                                self.diagnostics.push(
                                    Diagnostic::error("@adapter can only be applied to a model (or a let-binding whose RHS is a model)")
                                        .with_label(deco.span, "invalid @adapter target"),
                                );
                            } else {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner.resolve(s.0).unwrap_or("").to_string()
                                };
                                if let Some(cfg) = crate::wrga::validate_adapter_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                ) {
                                    self.adapter_configs.push(cfg);
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

                        // M46: @deterministic — register with effect checker for validation
                        if dname == "deterministic" {
                            if let StmtKind::FnDef(fn_def) = &stmt.kind {
                                let fn_name = self.interner.resolve(fn_def.name.0).unwrap_or("?").to_string();
                                self.effect_checker.mark_deterministic(&fn_name);
                            }
                        }

                        // M51: @pure — register with effect checker for validation
                        if dname == "pure" {
                            if let StmtKind::FnDef(fn_def) = &stmt.kind {
                                let fn_name = self.interner.resolve(fn_def.name.0).unwrap_or("?").to_string();
                                self.effect_checker.mark_pure(&fn_name);
                            }
                        }

                        // M51 + cycle-10 §5.3 (Task 3): @checkpoint —
                        // register with effect checker (requires @pure)
                        // and parse the optional `policy=...` kwarg.
                        //
                        // - Bare `@checkpoint`            → R0 deprecation
                        //                                   warning (Path B
                        //                                   per T3); treated
                        //                                   as policy="none"
                        //                                   (M14 tape
                        //                                   fallback).
                        // - `policy="full"`               → semantic policy
                        //                                   = Full + tape
                        //                                   fallback marker.
                        // - `policy="none"`               → tape fallback
                        //                                   marker only.
                        // - `policy="selective"`          → ERROR (reserved
                        //   `policy="selective_postnorm"`   for §5.3 v2/v3).
                        //   `policy="custom"`
                        // - any other value               → ERROR (valid-list
                        //                                   diagnostic).
                        if dname == "checkpoint" {
                            if let StmtKind::FnDef(fn_def) = &stmt.kind {
                                let fn_name = self.interner.resolve(fn_def.name.0).unwrap_or("?").to_string();

                                // Walk deco.args for the optional `policy=` kwarg.
                                let mut explicit_policy: Option<&str> = None;
                                let mut policy_arg_span: Option<nsl_ast::Span> = None;
                                let mut had_args = false;
                                if let Some(ref args) = deco.args {
                                    for arg in args {
                                        had_args = true;
                                        let Some(ref name_sym) = arg.name else {
                                            self.diagnostics.push(
                                                Diagnostic::error(
                                                    "@checkpoint: positional arguments are not allowed".to_string(),
                                                )
                                                .with_label(arg.span, "expected `policy = \"...\"`")
                                            );
                                            continue;
                                        };
                                        let aname = self.interner.resolve(name_sym.0).unwrap_or("").to_string();
                                        if aname != "policy" {
                                            self.diagnostics.push(
                                                Diagnostic::error(format!(
                                                    "@checkpoint: unknown argument '{aname}'"
                                                ))
                                                .with_label(arg.span, "unknown argument")
                                            );
                                            continue;
                                        }
                                        match &arg.value.kind {
                                            ExprKind::StringLiteral(s) => {
                                                explicit_policy = Some(match s.as_str() {
                                                    "full" => "full",
                                                    "none" => "none",
                                                    "selective" => "selective",
                                                    "selective_postnorm" => "selective_postnorm",
                                                    "custom" => "custom",
                                                    _ => "__invalid__",
                                                });
                                                // Stash the originally-typed string for diagnostics.
                                                if explicit_policy == Some("__invalid__") {
                                                    self.diagnostics.push(
                                                        Diagnostic::error(format!(
                                                            "@checkpoint policy must be one of: \"full\", \"none\", \"selective\" (refused), \"selective_postnorm\" (refused), \"custom\" (refused); got \"{s}\""
                                                        ))
                                                        .with_label(arg.span, "invalid policy value")
                                                    );
                                                }
                                                policy_arg_span = Some(arg.span);
                                            }
                                            _ => {
                                                self.diagnostics.push(
                                                    Diagnostic::error(
                                                        "@checkpoint policy must be a string literal".to_string(),
                                                    )
                                                    .with_label(arg.span, "expected string literal")
                                                );
                                            }
                                        }
                                    }
                                }

                                match explicit_policy {
                                    Some("full") => {
                                        self.effect_checker.mark_checkpointed(&fn_name);
                                        self.effect_checker.mark_checkpointed_with_policy(
                                            &fn_name,
                                            crate::effects::CheckpointPolicy::Full,
                                        );
                                    }
                                    Some("none") => {
                                        // M14 tape fallback continues; no codegen-side policy.
                                        self.effect_checker.mark_checkpointed(&fn_name);
                                    }
                                    Some("selective") => {
                                        self.diagnostics.push(
                                            Diagnostic::error(
                                                "@checkpoint(policy=\"selective\") reserved for §5.3-v2/v3; not implemented in v1".to_string(),
                                            )
                                            .with_label(policy_arg_span.unwrap_or(deco.span), "reserved policy")
                                        );
                                    }
                                    Some("selective_postnorm") => {
                                        self.diagnostics.push(
                                            Diagnostic::error(
                                                "@checkpoint(policy=\"selective_postnorm\") reserved for §5.3-v2/v3; not implemented in v1".to_string(),
                                            )
                                            .with_label(policy_arg_span.unwrap_or(deco.span), "reserved policy")
                                        );
                                    }
                                    Some("custom") => {
                                        self.diagnostics.push(
                                            Diagnostic::error(
                                                "@checkpoint(policy=\"custom\") reserved for §5.3-v2/v3; not implemented in v1".to_string(),
                                            )
                                            .with_label(policy_arg_span.unwrap_or(deco.span), "reserved policy")
                                        );
                                    }
                                    Some("__invalid__") => {
                                        // Diagnostic already pushed above.
                                    }
                                    Some(_) => unreachable!(),
                                    None => {
                                        // No `policy=` kwarg present (had_args may be true if
                                        // other kwargs were tried — those already errored).
                                        // Bare `@checkpoint` (no args at all) → R0 deprecation
                                        // warning (once per source file) + treat as policy="none".
                                        if !had_args {
                                            let file_id = deco.span.file_id;
                                            if self.effect_checker.emitted_checkpoint_deprecation.insert(file_id) {
                                                self.diagnostics.push(
                                                    Diagnostic::warning(
                                                        "@checkpoint without policy= is deprecated; specify policy=\"full\" (shipped) or policy=\"none\" (no-op). Hard refusal in cycle 11.".to_string(),
                                                    )
                                                    .with_label(deco.span, "bare @checkpoint")
                                                );
                                            }
                                            self.effect_checker.mark_checkpointed(&fn_name);
                                        } else {
                                            // had_args true but no valid policy= seen — caller
                                            // already got an error per-arg. Still mark for R4.
                                            self.effect_checker.mark_checkpointed(&fn_name);
                                        }
                                    }
                                }
                            }
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

                        // WGGO Phase 2 Task 1: `@wggo_target` placement.
                        // At the top level the decorator is invalid on any
                        // non-`fn` form (let-bindings, model defs, etc.).
                        // The forward-method-name rule is enforced inside
                        // `checker/model.rs` where the method's name is in
                        // scope; mirroring the @flash_attention pattern.
                        if dname == "wggo_target" {
                            match &stmt.kind {
                                StmtKind::FnDef(fn_def) => {
                                    let fn_name = self
                                        .interner
                                        .resolve(fn_def.name.0)
                                        .unwrap_or("?")
                                        .to_string();
                                    if fn_name != "forward" {
                                        self.diagnostics.push(
                                            Diagnostic::error(format!(
                                                "@wggo_target must be on the model's 'forward' method; found on '{fn_name}'"
                                            ))
                                            .with_label(deco.span, "invalid @wggo_target target"),
                                        );
                                    }
                                    // WGGO Phase 2 Task 2: required-args check.
                                    // Even when placement is wrong (non-`forward`)
                                    // we still report missing args so users see
                                    // all issues in one pass.
                                    let resolve = |s: nsl_ast::Symbol| -> String {
                                        self.interner.resolve(s.0).unwrap_or("").to_string()
                                    };
                                    crate::wggo::validate_wggo_target_required_args(
                                        deco,
                                        &resolve,
                                        &mut self.diagnostics,
                                    );
                                    // WGGO Phase 2 Task 3: each required
                                    // arg's value must be a `self.<field>`
                                    // reference.
                                    crate::wggo::validate_wggo_target_self_field_args(
                                        deco,
                                        &resolve,
                                        &mut self.diagnostics,
                                    );
                                    // WGGO Phase 2 Task 4 is deliberately
                                    // SKIPPED here. Field-existence and
                                    // field-type validation requires the
                                    // enclosing `ModelDef`'s `LayerDecl`
                                    // member list to resolve field types,
                                    // which is not available at the
                                    // top-level `fn` site. Task 1's
                                    // placement check already errors on
                                    // any non-`forward` standalone fn,
                                    // and a standalone `forward` fn has
                                    // no `self` model context anyway, so
                                    // any `self.<field>` arg here is
                                    // unresolvable by construction. See
                                    // the model-method call site in
                                    // `checker/model.rs` for the actual
                                    // Task 4 call.
                                }
                                _ => {
                                    self.diagnostics.push(
                                        Diagnostic::error(
                                            "@wggo_target can only be applied to fn declarations",
                                        )
                                        .with_label(deco.span, "invalid @wggo_target target"),
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

                        if dname == "tree_mask" {
                            // Paper §4 tree-mask: bare companion decorator on
                            // an @flash_attention fn. v1 is bare (no args);
                            // arg-less form mirrors the PTX-side gate plumbed
                            // through `FlashAttentionConfig::tree_mask`.
                            let has_flash = decorators.iter().any(|d| {
                                d.name.len() == 1 && self.interner.resolve(d.name[0].0).unwrap_or("") == "flash_attention"
                            });
                            if !has_flash {
                                self.diagnostics.push(
                                    Diagnostic::error("@tree_mask requires @flash_attention on the same function")
                                        .with_label(deco.span, "missing @flash_attention")
                                );
                            }
                            if let Some(ref args) = deco.args {
                                if !args.is_empty() {
                                    self.diagnostics.push(
                                        Diagnostic::error("@tree_mask takes no arguments in v1 (bare decorator only)")
                                            .with_label(deco.span, "unexpected arguments")
                                    );
                                }
                            }
                        }

                        if dname == "paged_kv" {
                            // Sprint 2 cycle-3 (paper §3.2 paged KV): companion
                            // decorator on an @flash_attention fn. Mirrors the
                            // @tree_mask / @gqa companion-required pattern so
                            // standalone use (which silently fell through the
                            // function-level decorator scan pre-Sprint-2)
                            // surfaces as a semantic error instead of a no-op.
                            //
                            // Note: the same decorator on a model-block layer
                            // is handled by `checker/model.rs`, which validates
                            // the arg names/types but does NOT require a
                            // companion (model layers don't carry the
                            // @flash_attention decorator — they wire paged KV
                            // through the model graph instead). This arm is
                            // therefore intentionally function-scope only.
                            let has_flash = decorators.iter().any(|d| {
                                d.name.len() == 1 && self.interner.resolve(d.name[0].0).unwrap_or("") == "flash_attention"
                            });
                            if !has_flash {
                                self.diagnostics.push(
                                    Diagnostic::error("@paged_kv requires @flash_attention on the same function")
                                        .with_label(deco.span, "missing @flash_attention")
                                );
                            }
                            // Arg validation: only `block_size` is recognised
                            // at function scope; other args are model-level
                            // (see `checker/model.rs::paged_kv`) and would be
                            // silently ignored by `compiler/kernel.rs:1005-1022`,
                            // so surface them as errors at the function level
                            // to mirror the kernel.rs extraction rules.
                            if let Some(ref args) = deco.args {
                                for arg in args {
                                    if let Some(ref name_sym) = arg.name {
                                        let aname = self.interner.resolve(name_sym.0).unwrap_or("").to_string();
                                        if aname == "block_size" {
                                            if let ExprKind::IntLiteral(n) = &arg.value.kind {
                                                if *n <= 0 {
                                                    self.diagnostics.push(
                                                        Diagnostic::error("@paged_kv 'block_size' must be a positive integer")
                                                            .with_label(arg.span, "must be > 0")
                                                    );
                                                }
                                            } else {
                                                self.diagnostics.push(
                                                    Diagnostic::error("@paged_kv 'block_size' argument must be an integer literal")
                                                        .with_label(arg.span, "expected integer")
                                                );
                                            }
                                        } else {
                                            self.diagnostics.push(
                                                Diagnostic::error(format!("@paged_kv unknown argument '{}' at function scope (only 'block_size' is recognised here)", aname))
                                                    .with_label(arg.span, "unknown argument")
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        if dname == "attention_sink" {
                            // Sprint 2 cycle-4 (paper §4.3 attention sinks):
                            // companion decorator on an @flash_attention fn.
                            // v0 API surface: extracts `tokens=N` (positive
                            // integer literal) into `FlashAttentionConfig::
                            // num_sink_tokens`. The SMEM-layout codegen that
                            // actually pins the first N tokens is DEFERRED to
                            // a future sprint — until then, the field is
                            // wired through the pipeline but the kernel does
                            // not emit any sink-specific PTX.
                            //
                            // Mirrors @paged_kv 'block_size' arg validation:
                            //   - companion-required on @flash_attention,
                            //   - unknown args rejected,
                            //   - 'tokens' must be a positive integer literal.
                            let has_flash = decorators.iter().any(|d| {
                                d.name.len() == 1 && self.interner.resolve(d.name[0].0).unwrap_or("") == "flash_attention"
                            });
                            if !has_flash {
                                self.diagnostics.push(
                                    Diagnostic::error("@attention_sink requires @flash_attention on the same function")
                                        .with_label(deco.span, "missing @flash_attention")
                                );
                            }
                            if let Some(ref args) = deco.args {
                                for arg in args {
                                    if let Some(ref name_sym) = arg.name {
                                        let aname = self.interner.resolve(name_sym.0).unwrap_or("").to_string();
                                        if aname == "tokens" {
                                            if let ExprKind::IntLiteral(n) = &arg.value.kind {
                                                if *n <= 0 {
                                                    self.diagnostics.push(
                                                        Diagnostic::error("@attention_sink 'tokens' must be a positive integer")
                                                            .with_label(arg.span, "must be > 0")
                                                    );
                                                }
                                            } else {
                                                self.diagnostics.push(
                                                    Diagnostic::error("@attention_sink 'tokens' argument must be an integer literal")
                                                        .with_label(arg.span, "expected integer")
                                                );
                                            }
                                        } else {
                                            self.diagnostics.push(
                                                Diagnostic::error(format!("@attention_sink unknown argument '{}' at function scope (only 'tokens' is recognised here)", aname))
                                                    .with_label(arg.span, "unknown argument")
                                            );
                                        }
                                    }
                                }
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
                    type_params: Vec::new(),
                    type_args: Vec::new(),
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
                self.check_tokenizer_def(tok);
            }
            StmtKind::DatasetDef(ds) => {
                self.check_dataset_def(ds);
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
            // M56: Agent body checking is deferred to Tasks 5–12.
            // For now we simply accept the declaration; E0610 is emitted
            // in check_module after collect_top_level_decls returns.
            StmtKind::AgentDef(_) => {}
        }
    }
}

/// True if a `@freeze` / `@adapter` decorator is on a valid target:
/// either a `ModelDef` directly, or a `VarDecl` whose initialiser is plausibly
/// a model (an identifier, call, or member access). Full type resolution
/// happens later in codegen when the WRGA driver runs.
fn is_model_target(kind: &StmtKind) -> bool {
    match kind {
        StmtKind::ModelDef(_) => true,
        StmtKind::VarDecl { value: Some(expr), .. } => {
            matches!(
                expr.kind,
                ExprKind::Call { .. }
                    | ExprKind::Ident(_)
                    | ExprKind::MemberAccess { .. }
            )
        }
        _ => false,
    }
}
