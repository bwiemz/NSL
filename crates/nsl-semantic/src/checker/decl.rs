use super::*;

impl<'a> TypeChecker<'a> {
    pub(crate) fn check_var_decl(
        &mut self,
        is_const: bool,
        pattern: &Pattern,
        type_ann: Option<&TypeExpr>,
        value: Option<&Expr>,
        _span: Span,
    ) {
        let ann_ty = type_ann.map(|t| self.resolve_type(t));
        let val_ty = value.map(|v| self.check_expr(v));

        let ty = match (&ann_ty, &val_ty) {
            (Some(ann), Some(val)) => {
                if !is_assignable(val, ann) {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "type mismatch: annotation is {}, but value has type {}",
                            display_type(ann), display_type(val)
                        ))
                        .with_label(
                            value.unwrap().span,
                            format!("expected {}", display_type(ann)),
                        ),
                    );
                }
                ann.clone()
            }
            (Some(ann), None) => ann.clone(),
            (None, Some(val)) => val.clone(),
            (None, None) => Type::Unknown,
        };

        self.declare_pattern_with_const(pattern, &ty, is_const);
    }

    pub(crate) fn check_fn_def(&mut self, fn_def: &FnDef) {
        let fn_ty = self.build_fn_type(fn_def);

        // Update the pre-declared symbol with the resolved type
        if let Some(info) = self.scopes.lookup_mut(self.current_scope, fn_def.name) {
            info.ty = fn_ty.clone();
        } else {
            self.declare_symbol(fn_def.name, fn_ty.clone(), fn_def.span, true, false);
        }

        // Push function scope
        let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Function);
        let prev_scope = self.current_scope;
        let prev_return = self.current_return_type.take();
        self.current_scope = scope;

        // Set expected return type
        if let Type::Function { ret, .. } = &fn_ty {
            self.current_return_type = Some(*ret.clone());
        }

        // Declare type params
        for tp in &fn_def.type_params {
            self.declare_symbol(tp.name, Type::TypeVar(tp.name), tp.span, true, false);
        }
        self.check_type_param_bounds(&fn_def.type_params);

        // Declare params
        for param in &fn_def.params {
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

        // Check body (track callees for effect graph)
        let prev_callees = std::mem::take(&mut self.current_callees);
        for s in &fn_def.body.stmts {
            self.check_stmt(s);
        }
        let callees = std::mem::replace(&mut self.current_callees, prev_callees);

        // M51: Register function with effect checker for propagation/validation.
        // Local effects are computed from direct builtin calls in the callee list.
        let fn_name = self.interner.resolve(fn_def.name.0).unwrap_or("?").to_string();
        let mut local_effects = crate::effects::EffectSet::PURE;
        for callee in &callees {
            let ce = crate::effects::classify_builtin_effects(callee);
            local_effects |= ce;
        }
        self.effect_checker.register_function(&fn_name, local_effects, callees);

        self.current_scope = prev_scope;
        self.current_return_type = prev_return;
    }

    pub(crate) fn check_struct_def(&mut self, struct_def: &StructDef) {
        let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Block);
        let prev_scope = self.current_scope;
        self.current_scope = scope;
        for tp in &struct_def.type_params {
            self.declare_symbol(tp.name, Type::TypeVar(tp.name), tp.span, true, false);
        }
        self.check_type_param_bounds(&struct_def.type_params);

        let mut fields: Vec<(Symbol, Type)> = Vec::new();
        for field in &struct_def.fields {
            let ty = self.resolve_type(&field.type_ann);
            fields.push((field.name, ty));
            if let Some(default) = &field.default {
                self.check_expr(default);
            }
        }

        self.current_scope = prev_scope;

        let struct_ty = Type::Struct {
            name: struct_def.name,
            type_params: struct_def.type_params.iter().map(|tp| tp.name).collect(),
            type_args: Vec::new(),
            fields,
        };
        if let Some(info) = self.scopes.lookup_mut(self.current_scope, struct_def.name) {
            info.ty = struct_ty;
        } else {
            self.declare_symbol(struct_def.name, struct_ty, struct_def.span, true, false);
        }
    }

    pub(crate) fn check_enum_def(&mut self, enum_def: &EnumDef) {
        let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Block);
        let prev_scope = self.current_scope;
        self.current_scope = scope;
        for tp in &enum_def.type_params {
            self.declare_symbol(tp.name, Type::TypeVar(tp.name), tp.span, true, false);
        }
        self.check_type_param_bounds(&enum_def.type_params);

        let mut variants: Vec<(Symbol, Vec<Type>)> = Vec::new();
        let mut seen_names: std::collections::HashSet<Symbol> = std::collections::HashSet::new();
        for variant in &enum_def.variants {
            if !seen_names.insert(variant.name) {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "duplicate variant '{}' in enum '{}'",
                        self.resolve_name(variant.name),
                        self.resolve_name(enum_def.name),
                    ))
                    .with_label(variant.span, "duplicate variant"),
                );
            }
            let field_types: Vec<Type> =
                variant.fields.iter().map(|t| self.resolve_type(t)).collect();
            variants.push((variant.name, field_types.clone()));

            let type_params: Vec<Symbol> = enum_def.type_params.iter().map(|tp| tp.name).collect();

            // Declare each variant as a name in scope
            let variant_ty = if field_types.is_empty() {
                Type::Unknown // unit variant — acts like a value
            } else {
                Type::Function {
                    params: field_types,
                    ret: Box::new(Type::Enum {
                        name: enum_def.name,
                        type_params,
                        type_args: Vec::new(),
                        variants: Vec::new(), // filled in later
                    }),
                    effect: Effect::Inferred,
                }
            };
            self.current_scope = prev_scope;
            self.declare_symbol(variant.name, variant_ty, variant.span, true, false);
            self.current_scope = scope;
        }

        self.current_scope = prev_scope;

        let enum_ty = Type::Enum {
            name: enum_def.name,
            type_params: enum_def.type_params.iter().map(|tp| tp.name).collect(),
            type_args: Vec::new(),
            variants,
        };
        if let Some(info) = self.scopes.lookup_mut(self.current_scope, enum_def.name) {
            info.ty = enum_ty;
        } else {
            self.declare_symbol(enum_def.name, enum_ty, enum_def.span, true, false);
        }
    }

    fn check_trait_method_def(&mut self, fn_def: &FnDef) {
        let fn_ty = self.build_fn_type(fn_def);

        if let Some(info) = self.scopes.lookup_mut(self.current_scope, fn_def.name) {
            info.ty = fn_ty.clone();
        } else {
            self.declare_symbol(fn_def.name, fn_ty.clone(), fn_def.span, true, false);
        }

        let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Function);
        let prev_scope = self.current_scope;
        let prev_return = self.current_return_type.take();
        self.current_scope = scope;

        if let Type::Function { ret, .. } = &fn_ty {
            self.current_return_type = Some(*ret.clone());
        }

        for tp in &fn_def.type_params {
            self.declare_symbol(tp.name, Type::TypeVar(tp.name), tp.span, true, false);
        }
        self.check_type_param_bounds(&fn_def.type_params);

        for param in &fn_def.params {
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

        // Trait methods don't participate in the module-level effect graph yet,
        // but their local callee list still needs to stay isolated.
        let prev_callees = std::mem::take(&mut self.current_callees);
        for stmt in &fn_def.body.stmts {
            self.check_stmt(stmt);
        }
        self.current_callees = prev_callees;

        self.current_scope = prev_scope;
        self.current_return_type = prev_return;
    }

    pub(crate) fn check_trait_def(&mut self, trait_def: &TraitDef) {
        let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Block);
        let prev_scope = self.current_scope;
        self.current_scope = scope;

        for tp in &trait_def.type_params {
            self.declare_symbol(tp.name, Type::TypeVar(tp.name), tp.span, true, false);
        }
        self.check_type_param_bounds(&trait_def.type_params);

        let mut methods: Vec<(Symbol, Type)> = Vec::new();
        let mut unique_methods: Vec<&FnDef> = Vec::new();
        let mut seen_names: std::collections::HashSet<Symbol> = std::collections::HashSet::new();
        for method in &trait_def.methods {
            if !seen_names.insert(method.name) {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "duplicate trait method '{}' in trait '{}'",
                        self.resolve_name(method.name),
                        self.resolve_name(trait_def.name),
                    ))
                    .with_label(method.span, "duplicate trait method"),
                );
                continue;
            }

            let method_ty = self.build_fn_type(method);
            self.declare_symbol(method.name, method_ty.clone(), method.span, true, false);
            methods.push((method.name, method_ty));
            unique_methods.push(method);
        }

        for method in unique_methods {
            self.check_trait_method_def(method);
        }

        self.current_scope = prev_scope;

        let trait_ty = Type::Trait {
            name: trait_def.name,
            type_params: trait_def.type_params.iter().map(|tp| tp.name).collect(),
            type_args: Vec::new(),
            methods,
        };
        if let Some(info) = self.scopes.lookup_mut(self.current_scope, trait_def.name) {
            info.ty = trait_ty;
        } else {
            self.declare_symbol(trait_def.name, trait_ty, trait_def.span, true, false);
        }
    }

    pub(crate) fn check_import(&mut self, import: &ImportStmt) {
        // Handle `import nsl.math as math` — alias imports
        if let Some(alias) = import.alias {
            let ty = self.import_types.get(&alias).cloned().unwrap_or(Type::Unknown);
            self.declare_symbol(alias, ty, import.span, true, false);
            return;
        }

        match &import.items {
            ImportItems::Module => {
                // `import nsl.nn` — declare last path segment
                if let Some(last) = import.path.last() {
                    let ty = self.import_types.get(last).cloned().unwrap_or(Type::Unknown);
                    self.declare_symbol(*last, ty, import.span, true, false);
                }
            }
            ImportItems::Named(items) => {
                for item in items {
                    let local_name = item.alias.unwrap_or(item.name);
                    // Look up by original name first (import_types is keyed by export name),
                    // then fall back to alias name for compatibility.
                    let ty = self.import_types.get(&item.name)
                        .or_else(|| self.import_types.get(&local_name))
                        .cloned()
                        .unwrap_or(Type::Unknown);
                    self.declare_symbol(local_name, ty, item.span, true, false);
                }
            }
            ImportItems::Glob => {
                // For glob imports, all import_types have been pre-populated by the loader
                let entries: Vec<_> = self.import_types.iter()
                    .map(|(sym, ty)| (*sym, ty.clone()))
                    .collect();
                for (sym, ty) in entries {
                    self.declare_symbol(sym, ty, import.span, true, false);
                }
            }
        }
    }

    pub(crate) fn check_from_import(&mut self, import: &FromImportStmt) {
        match &import.items {
            ImportItems::Module => {}
            ImportItems::Named(items) => {
                for item in items {
                    let local_name = item.alias.unwrap_or(item.name);
                    let ty = self.import_types.get(&item.name)
                        .or_else(|| self.import_types.get(&local_name))
                        .cloned()
                        .unwrap_or(Type::Unknown);
                    self.declare_symbol(local_name, ty, item.span, true, false);
                }
            }
            ImportItems::Glob => {
                // For glob imports, all import_types have been pre-populated by the loader
                let entries: Vec<_> = self.import_types.iter()
                    .map(|(sym, ty)| (*sym, ty.clone()))
                    .collect();
                for (sym, ty) in entries {
                    self.declare_symbol(sym, ty, import.span, true, false);
                }
            }
        }
    }
}
