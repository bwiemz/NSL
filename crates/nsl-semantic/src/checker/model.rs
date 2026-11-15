use super::*;

impl<'a> TypeChecker<'a> {
    pub(crate) fn check_model_def(&mut self, model_def: &ModelDef) {
        // Push model scope
        let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Model);
        let prev = self.current_scope;
        self.current_scope = scope;

        for tp in &model_def.type_params {
            self.declare_symbol(tp.name, Type::TypeVar(tp.name), tp.span, true, false);
        }
        self.check_type_param_bounds(&model_def.type_params);

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
                    fields.push((*name, ty.clone()));
                    if let Some(init_expr) = init {
                        // C5: an annotated field's creation initializer stores
                        // the annotation's f32/f64 dtype, as a `let` does (step
                        // 3); any other dtype the initializer is known to make
                        // differs from what the field claims to hold (step 5's
                        // field check, on the dtype only).
                        let init_ty = self.check_expr(init_expr);
                        let init_ty = self.creation_dtype_from_annotation(init_expr, init_ty, Some(&ty));
                        self.check_field_init_dtype(*name, &ty, &init_ty, init_expr.span);
                        if let Type::FixedModelArray { element_model, size } = ty {
                            self.check_model_array_init(element_model, size, init_expr);
                        }
                    }
                    for deco in decorators {
                        if deco.name.len() == 1 {
                            let dname = self
                                .interner
                                .resolve(deco.name[0].0)
                                .unwrap_or("")
                                .to_string();
                            // M30: @shard decorator validation
                            if dname == "shard"
                                && let Some(ref args) = deco.args
                            {
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
                                // Cycle-10 §5.3 (Task 4 / T6 / R9): record
                                // the enclosing model's name so the
                                // codegen-side dispatch fork at
                                // `flash_attention_v2/mod.rs::synthesize_backward_with_tier`
                                // can refuse the
                                // `@checkpoint(policy="full")` + `@paged_kv`
                                // composition before either side fires. v1
                                // records lexical-scope membership only;
                                // call-graph resolution deferred to v4.
                                let model_name = self
                                    .interner
                                    .resolve(model_def.name.0)
                                    .unwrap_or("")
                                    .to_string();
                                if !model_name.is_empty() {
                                    self.effect_checker.mark_paged_kv_model(&model_name);
                                }
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
                            // WGGO Phase 2 Task 1: `@wggo_target` is only
                            // valid on the model's `forward` method, never
                            // on a layer declaration.
                            if dname == "wggo_target" {
                                self.diagnostics.push(
                                    Diagnostic::error(
                                        "@wggo_target can only be applied to fn declarations"
                                            .to_string(),
                                    )
                                    .with_label(deco.span, "invalid @wggo_target target"),
                                );
                            }
                        }
                    }
                }
                ModelMember::Method(fn_def, decos) => {
                    let method_ty = self.build_fn_type(fn_def);
                    methods.push((fn_def.name, method_ty));

                    // WGGO Phase 2 Task 1: `@wggo_target` placement validation.
                    // The decorator is only valid on the model's `forward`
                    // method — the AWQ-calibration backward pass needs to
                    // observe the canonical forward computation. Mirrors the
                    // existing `@flash_attention` per-target validation in
                    // `checker/stmt.rs` for top-level fn declarations.
                    for deco in decos {
                        if deco.name.len() != 1 {
                            continue;
                        }
                        let dname = self
                            .interner
                            .resolve(deco.name[0].0)
                            .unwrap_or("")
                            .to_string();
                        // `@checkpoint` is read on top-level functions
                        // only (`checker/stmt.rs`). On a method it used to
                        // be accepted with any arguments and do nothing.
                        if dname == "checkpoint" {
                            self.diagnostics.push(
                                Diagnostic::error(
                                    "@checkpoint is not supported on a model method: it applies to a top-level function. For block checkpointing of a train block use --checkpoint-blocks and the other --checkpoint-* flags".to_string(),
                                )
                                .with_label(deco.span, "unsupported @checkpoint target"),
                            );
                        }
                        // `@fp8_compute` on a method is what codegen reads
                        // (compiler/functions.rs); validate it here too.
                        if dname == "fp8_compute" {
                            let resolve = |s: nsl_ast::Symbol| -> String {
                                self.interner.resolve(s.0).unwrap_or("").to_string()
                            };
                            crate::fp8::validate_fp8_compute_decorator(
                                deco,
                                &resolve,
                                &mut self.diagnostics,
                            );
                        }
                        if dname == "wggo_target" {
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
                            // we still report the missing-args error so
                            // users see all the issues in one pass rather
                            // than fixing the name and re-running to
                            // discover the args were missing too.
                            let resolve = |s: nsl_ast::Symbol| -> String {
                                self.interner.resolve(s.0).unwrap_or("").to_string()
                            };
                            crate::wggo::validate_wggo_target_required_args(
                                deco,
                                &resolve,
                                &mut self.diagnostics,
                            );
                            // WGGO Phase 2 Task 3: each required arg's
                            // value must be a `self.<field>` reference.
                            crate::wggo::validate_wggo_target_self_field_args(
                                deco,
                                &resolve,
                                &mut self.diagnostics,
                            );
                            // WGGO Phase 2 Task 4: each `self.<field>`
                            // arg must reference an existing field of
                            // the enclosing model with the right type
                            // (Tensor for w_q/w_k/w_v/w_o; int for
                            // head_dim). Args that didn't pass Task 3
                            // are silently skipped inside the helper.
                            crate::wggo::validate_wggo_target_field_types(
                                deco,
                                model_def,
                                &resolve,
                                &mut self.diagnostics,
                            );
                        }
                    }
                }
            }
        }

        // Build complete self type with all fields and methods
        let complete_self_type = Type::Model {
            name: model_def.name,
            type_params: model_def.type_params.iter().map(|tp| tp.name).collect(),
            type_args: Vec::new(),
            fields: fields.clone(),
            methods: methods.clone(),
        };

        // Pass 2: Check method bodies with complete self type. The method's
        // signature is the one pass 1 resolved (resolving it again
        // reported its errors again).
        let mut method_types = methods.iter().map(|(_, ty)| ty.clone());
        for member in &model_def.members {
            if let ModelMember::Method(fn_def, _decos) = member {
                // Pass 1 pushed one signature per method, in member order.
                let method_ty = method_types.next().unwrap_or(Type::Unknown);

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
                self.declare_params_from_signature(fn_def, &method_ty);

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
            type_params: model_def.type_params.iter().map(|tp| tp.name).collect(),
            type_args: Vec::new(),
            fields,
            methods,
        };
        if let Some(info) = self.scopes.lookup_mut(self.current_scope, model_def.name) {
            info.ty = model_ty;
        } else {
            self.declare_symbol(model_def.name, model_ty, model_def.span, true, false);
        }
    }

    /// The initialiser of a `[M; n]` field is either one expression the
    /// constructor evaluates once per element (`= M()`), or a list of `n`
    /// expressions, element `i` from item `i` (`= [M(1), M(2)]`). Each
    /// element must be an `M`. Codegen stores one pointer per element; a
    /// list or a value of another model stored there was read later as an
    /// `M` ("invalid tensor handle" on `block.w`).
    fn check_model_array_init(&mut self, element_model: Symbol, size: i64, init: &Expr) {
        let model = self.resolve_name(element_model);
        let elements: Vec<&Expr> = match &init.kind {
            ExprKind::ListLiteral(items) => {
                if items.len() as i64 != size {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "a `[{model}; {size}]` field initialised with a list needs {size} \
                             elements, one per array element; this list has {}",
                            items.len()
                        ))
                        .with_label(init.span, format!("expected {size} elements")),
                    );
                    return;
                }
                items.iter().collect()
            }
            _ => vec![init],
        };
        for element in elements {
            let element_ty = self.type_map.get(&element.id).cloned().unwrap_or(Type::Unknown);
            let fits = match &element_ty {
                Type::Model { name, .. } => *name == element_model,
                other => other.is_indeterminate(),
            };
            if !fits {
                let found = match &element_ty {
                    Type::Model { name, .. } => format!("a `{}`", self.resolve_name(*name)),
                    other => display_type(other),
                };
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "an element of a `[{model}; {size}]` field must be a `{model}`, found \
                         {found}; initialise the field with `{model}(...)` (one constructor call \
                         per element) or a list of {size} of them"
                    ))
                    .with_label(element.span, format!("not a `{model}`")),
                );
            }
        }
    }
}
