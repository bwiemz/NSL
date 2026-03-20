use super::*;

impl<'a> TypeChecker<'a> {
    pub(crate) fn check_model_def(&mut self, model_def: &ModelDef) {
        // Push model scope
        let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Model);
        let prev = self.current_scope;
        self.current_scope = scope;

        // Declare constructor params
        for param in &model_def.params {
            let param_ty = param
                .type_ann
                .as_ref()
                .map(|t| self.resolve_type(t))
                .unwrap_or(Type::Unknown);
            self.declare_symbol(param.name, param_ty, param.span, false, true);
        }

        // Two-pass approach: first collect ALL fields and method signatures,
        // then check method bodies with a complete `self` type.
        let mut fields: Vec<(Symbol, Type)> = Vec::new();
        let mut methods: Vec<(Symbol, Type)> = Vec::new();

        // Pass 1: Collect fields and method signatures
        for member in &model_def.members {
            match member {
                ModelMember::LayerDecl {
                    name,
                    type_ann,
                    init,
                    decorators,
                    span,
                } => {
                    let ty = self.resolve_type(type_ann);
                    self.declare_symbol(*name, ty.clone(), *span, false, false);
                    fields.push((*name, ty));
                    if let Some(init_expr) = init {
                        self.check_expr(init_expr);
                    }
                    for deco in decorators {
                        if deco.name.len() == 1 {
                            let dname = self
                                .interner
                                .resolve(deco.name[0].0)
                                .unwrap_or("")
                                .to_string();
                            // M30: @shard decorator validation
                            if dname == "shard" {
                                if let Some(ref args) = deco.args {
                                    for arg in args {
                                        if let Some(ref name_sym) = arg.name {
                                            let aname = self
                                                .interner
                                                .resolve(name_sym.0)
                                                .unwrap_or("")
                                                .to_string();
                                            match aname.as_str() {
                                                "dim" => {
                                                    if let ExprKind::IntLiteral(n) =
                                                        &arg.value.kind
                                                    {
                                                        if *n < 0 {
                                                            self.diagnostics.push(
                                                                Diagnostic::error(
                                                                    "@shard: dim must be a non-negative integer"
                                                                        .to_string(),
                                                                )
                                                                .with_label(
                                                                    arg.span,
                                                                    "must be >= 0",
                                                                ),
                                                            );
                                                        }
                                                    } else {
                                                        self.diagnostics.push(
                                                            Diagnostic::error(
                                                                "@shard: dim must be an integer literal"
                                                                    .to_string(),
                                                            )
                                                            .with_label(
                                                                arg.span,
                                                                "expected integer",
                                                            ),
                                                        );
                                                    }
                                                }
                                                _ => {
                                                    self.diagnostics.push(
                                                        Diagnostic::error(format!(
                                                            "@shard: unknown argument '{}'",
                                                            aname
                                                        ))
                                                        .with_label(
                                                            arg.span,
                                                            "unknown argument",
                                                        ),
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // M32: @moe decorator validation
                            if dname == "moe" {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner
                                        .resolve(s.0)
                                        .unwrap_or("")
                                        .to_string()
                                };
                                crate::moe::validate_moe_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                            }
                            // M33: @speculative decorator validation
                            if dname == "speculative" {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner
                                        .resolve(s.0)
                                        .unwrap_or("")
                                        .to_string()
                                };
                                crate::speculative::validate_speculative_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                            }
                            // M33b: @medusa decorator validation
                            if dname == "medusa" {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner
                                        .resolve(s.0)
                                        .unwrap_or("")
                                        .to_string()
                                };
                                crate::speculative::validate_medusa_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                            }
                            // M34: @context_parallel decorator validation
                            if dname == "context_parallel" {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner
                                        .resolve(s.0)
                                        .unwrap_or("")
                                        .to_string()
                                };
                                crate::context_parallel::validate_context_parallel_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                            }
                            // M35: @fp8_compute decorator validation
                            if dname == "fp8_compute" {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner
                                        .resolve(s.0)
                                        .unwrap_or("")
                                        .to_string()
                                };
                                crate::fp8::validate_fp8_compute_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                            }
                            // M50: @sparse decorator validation
                            if dname == "sparse" {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner
                                        .resolve(s.0)
                                        .unwrap_or("")
                                        .to_string()
                                };
                                crate::sparse::validate_sparse_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                            }
                            // M37: @perf_budget decorator validation
                            if dname == "perf_budget" {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner
                                        .resolve(s.0)
                                        .unwrap_or("")
                                        .to_string()
                                };
                                crate::perf_budget::validate_perf_budget_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                            }
                            if dname == "paged_kv" {
                                if let Some(ref args) = deco.args {
                                    for arg in args {
                                        if let Some(ref name_sym) = arg.name {
                                            let aname = self
                                                .interner
                                                .resolve(name_sym.0)
                                                .unwrap_or("")
                                                .to_string();
                                            match aname.as_str() {
                                                "block_size" | "num_blocks" | "num_heads"
                                                | "head_dim" | "num_layers" => {
                                                    if let ExprKind::IntLiteral(n) =
                                                        &arg.value.kind
                                                    {
                                                        if *n <= 0 {
                                                            self.diagnostics.push(
                                                                Diagnostic::error(format!(
                                                                    "@paged_kv: {} must be a positive integer",
                                                                    aname
                                                                ))
                                                                .with_label(
                                                                    arg.span,
                                                                    "must be > 0",
                                                                ),
                                                            );
                                                        }
                                                    } else {
                                                        self.diagnostics.push(
                                                            Diagnostic::error(format!(
                                                                "@paged_kv: {} must be an integer literal",
                                                                aname
                                                            ))
                                                            .with_label(
                                                                arg.span,
                                                                "expected integer",
                                                            ),
                                                        );
                                                    }
                                                }
                                                _ => {
                                                    self.diagnostics.push(
                                                        Diagnostic::error(format!(
                                                            "@paged_kv: unknown argument '{}'",
                                                            aname
                                                        ))
                                                        .with_label(
                                                            arg.span,
                                                            "unknown argument",
                                                        ),
                                                    );
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                            // M42: @kv_compress decorator validation
                            if dname == "kv_compress" {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner
                                        .resolve(s.0)
                                        .unwrap_or("")
                                        .to_string()
                                };
                                crate::kv_compress::validate_kv_compress_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                            }
                            // M43: @pipeline decorator validation
                            if dname == "pipeline" {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner
                                        .resolve(s.0)
                                        .unwrap_or("")
                                        .to_string()
                                };
                                crate::pipeline::validate_pipeline_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                            }
                            // M44: @grammar decorator validation
                            if dname == "grammar" {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner
                                        .resolve(s.0)
                                        .unwrap_or("")
                                        .to_string()
                                };
                                crate::grammar::validate_grammar_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                            }
                            // M48: @multimodal decorator validation
                            if dname == "multimodal" {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner
                                        .resolve(s.0)
                                        .unwrap_or("")
                                        .to_string()
                                };
                                crate::multimodal::validate_multimodal_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                            }
                            // M47: @target decorator validation
                            if dname == "target" {
                                let resolve = |s: nsl_ast::Symbol| -> String {
                                    self.interner
                                        .resolve(s.0)
                                        .unwrap_or("")
                                        .to_string()
                                };
                                crate::target::validate_target_decorator(
                                    deco,
                                    &resolve,
                                    &mut self.diagnostics,
                                );
                            }
                        }
                    }
                }
                ModelMember::Method(fn_def, _decos) => {
                    let method_ty = self.build_fn_type(fn_def);
                    methods.push((fn_def.name, method_ty));
                }
            }
        }

        // Build complete self type with all fields and methods
        let complete_self_type = Type::Model {
            name: model_def.name,
            fields: fields.clone(),
            methods: methods.clone(),
        };

        // Pass 2: Check method bodies with complete self type
        for member in &model_def.members {
            if let ModelMember::Method(fn_def, _decos) = member {
                let method_ty = self.build_fn_type(fn_def);

                let method_scope =
                    self.scopes.push_scope(self.current_scope, ScopeKind::Method);
                let prev_method = self.current_scope;
                let prev_return = self.current_return_type.take();
                self.current_scope = method_scope;

                // Declare self with complete model type
                let self_sym = Symbol(self.interner.get_or_intern_static("self"));
                self.declare_symbol(self_sym, complete_self_type.clone(), fn_def.span, false, true);

                // Set return type
                if let Type::Function { ret, .. } = &method_ty {
                    self.current_return_type = Some(*ret.clone());
                }

                // Declare params (skip self)
                for param in &fn_def.params {
                    let name_str = self.resolve_name(param.name);
                    if name_str == "self" {
                        continue;
                    }
                    let param_ty = param
                        .type_ann
                        .as_ref()
                        .map(|t| self.resolve_type(t))
                        .unwrap_or(Type::Unknown);
                    self.declare_symbol(param.name, param_ty, param.span, false, true);
                    if let Some(default) = &param.default {
                        self.check_expr(default);
                    }
                }

                // Check method body
                for s in &fn_def.body.stmts {
                    self.check_stmt(s);
                }

                self.current_scope = prev_method;
                self.current_return_type = prev_return;
            }
        }

        self.current_scope = prev;

        // Build and register the model type
        let model_ty = Type::Model {
            name: model_def.name,
            fields,
            methods,
        };
        if let Some(info) = self.scopes.lookup_mut(self.current_scope, model_def.name) {
            info.ty = model_ty;
        } else {
            self.declare_symbol(model_def.name, model_ty, model_def.span, true, false);
        }
    }
}
