use std::collections::HashMap;

use nsl_ast::block::*;
use nsl_ast::decl::*;
use nsl_ast::expr::*;
use nsl_ast::operator::*;
use nsl_ast::pattern::{Pattern, PatternKind};
use nsl_ast::stmt::*;
use nsl_ast::types::TypeExpr;
use nsl_ast::{Module, NodeId, Symbol};
use nsl_errors::{Diagnostic, Span};
use nsl_lexer::Interner;

use crate::resolve::TypeResolver;
use crate::scope::*;
use crate::shapes;
use crate::types::*;

/// Maps each expression NodeId to its resolved Type.
pub type TypeMap = HashMap<NodeId, Type>;

pub struct TypeChecker<'a> {
    pub interner: &'a mut Interner,
    pub scopes: &'a mut ScopeMap,
    pub diagnostics: Vec<Diagnostic>,
    pub type_map: TypeMap,
    current_scope: ScopeId,
    current_return_type: Option<Type>,
    /// Pre-resolved types for imported symbols (from other modules).
    import_types: HashMap<Symbol, Type>,
}

impl<'a> TypeChecker<'a> {
    pub fn new(interner: &'a mut Interner, scopes: &'a mut ScopeMap) -> Self {
        Self {
            interner,
            scopes,
            diagnostics: Vec::new(),
            type_map: HashMap::new(),
            current_scope: ScopeId::ROOT,
            current_return_type: None,
            import_types: HashMap::new(),
        }
    }

    /// Set pre-resolved import types from other modules.
    pub fn set_import_types(&mut self, import_types: &HashMap<Symbol, Type>) {
        self.import_types = import_types.clone();
    }

    pub fn check_module(&mut self, module: &Module) {
        // Two-pass: first collect top-level declarations, then check bodies
        self.collect_top_level_decls(&module.stmts);
        for stmt in &module.stmts {
            self.check_stmt(stmt);
        }
    }

    /// Pre-declare top-level names so forward references work.
    /// Two sub-passes: first imports (so types are available), then declarations.
    fn collect_top_level_decls(&mut self, stmts: &[Stmt]) {
        // Sub-pass 1: Process imports first so imported types are available
        for stmt in stmts {
            match &stmt.kind {
                StmtKind::Import(import) => self.check_import(import),
                StmtKind::FromImport(import) => self.check_from_import(import),
                _ => {}
            }
        }
        // Sub-pass 2: Pre-declare type/function names
        for stmt in stmts {
            match &stmt.kind {
                StmtKind::FnDef(fn_def) => {
                    let ty = self.build_fn_type(fn_def);
                    self.declare_symbol(fn_def.name, ty, stmt.span, true, false);
                }
                StmtKind::ModelDef(model_def) => {
                    // Declare model name as Unknown initially, refined during check
                    self.declare_symbol(model_def.name, Type::Unknown, stmt.span, true, false);
                }
                StmtKind::StructDef(struct_def) => {
                    self.declare_symbol(struct_def.name, Type::Unknown, stmt.span, true, false);
                }
                StmtKind::EnumDef(enum_def) => {
                    self.declare_symbol(enum_def.name, Type::Unknown, stmt.span, true, false);
                }
                StmtKind::TraitDef(trait_def) => {
                    self.declare_symbol(trait_def.name, Type::Unknown, stmt.span, true, false);
                }
                StmtKind::Decorated { stmt, .. } => {
                    // Recurse into the inner stmt for pre-declaration
                    self.collect_top_level_decls(std::slice::from_ref(stmt));
                }
                _ => {}
            }
        }
    }

    // ===== Statement checking =====

    fn check_stmt(&mut self, stmt: &Stmt) {
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
                    if !is_numeric(&target_ty) || !is_numeric(&value_ty) {
                        if !target_ty.is_indeterminate() && !value_ty.is_indeterminate() {
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
        }
    }

    fn check_var_decl(
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

    fn check_fn_def(&mut self, fn_def: &FnDef) {
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

        // Check body
        for s in &fn_def.body.stmts {
            self.check_stmt(s);
        }

        self.current_scope = prev_scope;
        self.current_return_type = prev_return;
    }

    fn check_model_def(&mut self, model_def: &ModelDef) {
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
                    span,
                    ..
                } => {
                    let ty = self.resolve_type(type_ann);
                    self.declare_symbol(*name, ty.clone(), *span, false, false);
                    fields.push((*name, ty));
                    if let Some(init_expr) = init {
                        self.check_expr(init_expr);
                    }
                }
                ModelMember::Method(fn_def) => {
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
            if let ModelMember::Method(fn_def) = member {
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

    fn check_struct_def(&mut self, struct_def: &StructDef) {
        let mut fields: Vec<(Symbol, Type)> = Vec::new();
        for field in &struct_def.fields {
            let ty = self.resolve_type(&field.type_ann);
            fields.push((field.name, ty));
            if let Some(default) = &field.default {
                self.check_expr(default);
            }
        }

        let struct_ty = Type::Struct {
            name: struct_def.name,
            fields,
        };
        if let Some(info) = self.scopes.lookup_mut(self.current_scope, struct_def.name) {
            info.ty = struct_ty;
        } else {
            self.declare_symbol(struct_def.name, struct_ty, struct_def.span, true, false);
        }
    }

    fn check_enum_def(&mut self, enum_def: &EnumDef) {
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

            // Declare each variant as a name in scope
            let variant_ty = if field_types.is_empty() {
                Type::Unknown // unit variant — acts like a value
            } else {
                Type::Function {
                    params: field_types,
                    ret: Box::new(Type::Enum {
                        name: enum_def.name,
                        variants: Vec::new(), // filled in later
                    }),
                }
            };
            self.declare_symbol(variant.name, variant_ty, variant.span, true, false);
        }

        let enum_ty = Type::Enum {
            name: enum_def.name,
            variants,
        };
        if let Some(info) = self.scopes.lookup_mut(self.current_scope, enum_def.name) {
            info.ty = enum_ty;
        } else {
            self.declare_symbol(enum_def.name, enum_ty, enum_def.span, true, false);
        }
    }

    fn check_import(&mut self, import: &ImportStmt) {
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
            ImportItems::Glob => {} // Can't declare specific names
        }
    }

    fn check_from_import(&mut self, import: &FromImportStmt) {
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

    fn check_train_block(&mut self, train: &TrainBlock) {
        // Check config expressions (model=, epochs=, etc.)
        for arg in &train.config {
            self.check_expr(&arg.value);
        }

        let mut has_step = false;

        for section in &train.sections {
            match section {
                TrainSection::Data(stmts) => {
                    for stmt in stmts {
                        self.check_stmt(stmt);
                    }
                }
                TrainSection::Optimizer(expr) => {
                    self.check_expr(expr);
                }
                TrainSection::Scheduler(_expr) => {
                    // Skip type-checking the scheduler call: codegen auto-injects
                    // base_lr and step as the first two arguments, so the user-facing
                    // arg count is intentionally less than the stdlib signature.
                }
                TrainSection::Step { param: _, body } => {
                    has_step = true;
                    self.check_block(body, ScopeKind::Block);
                }
                TrainSection::Eval { param: _, body } => {
                    self.check_block(body, ScopeKind::Block);
                }
                TrainSection::Callbacks(callbacks) => {
                    for cb in callbacks {
                        // Create a new scope and declare callback params
                        let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Block);
                        let prev = self.current_scope;
                        self.current_scope = scope;

                        for param in &cb.params {
                            let param_ty = if let Some(type_ann) = &param.type_ann {
                                self.resolve_type(type_ann)
                            } else {
                                // Infer types for well-known callback params
                                let pname = self.resolve_name(param.name);
                                match pname.as_str() {
                                    "step" | "epoch" => Type::Int,
                                    "loss" => Type::Tensor {
                                        shape: Shape::unknown(),
                                        dtype: DType::F64,
                                        device: Device::Cpu,
                                    },
                                    _ => Type::Unknown,
                                }
                            };
                            self.declare_symbol(param.name, param_ty, param.span, false, true);
                        }

                        for s in &cb.body.stmts {
                            self.check_stmt(s);
                        }

                        self.current_scope = prev;
                    }
                }
                TrainSection::Distribute(expr) => {
                    self.check_expr(expr);
                }
                TrainSection::Stmt(stmt) => {
                    self.check_stmt(stmt);
                }
            }
        }

        if !has_step {
            self.diagnostics.push(
                Diagnostic::warning("train block missing 'step' section")
                    .with_label(train.span, "expected a step(batch): section"),
            );
        }
    }

    // ===== Expression checking =====

    pub fn check_expr(&mut self, expr: &Expr) -> Type {
        let ty = match &expr.kind {
            ExprKind::IntLiteral(_) => Type::Int,
            ExprKind::FloatLiteral(_) => Type::Float,
            ExprKind::StringLiteral(_) => Type::Str,
            ExprKind::BoolLiteral(_) => Type::Bool,
            ExprKind::NoneLiteral => Type::NoneType,
            ExprKind::FString(parts) => {
                for part in parts {
                    if let FStringPart::Expr(e) = part {
                        self.check_expr(e);
                    }
                }
                Type::Str
            }
            ExprKind::Ident(sym) => {
                if let Some((_sid, info)) = self.scopes.lookup(self.current_scope, *sym) {
                    info.ty.clone()
                } else {
                    let name = self.resolve_name(*sym);
                    self.diagnostics.push(
                        Diagnostic::error(format!("undefined variable `{name}`"))
                            .with_label(expr.span, "not found in scope"),
                    );
                    Type::Error
                }
            }
            ExprKind::SelfRef => {
                let self_sym = Symbol(self.interner.get_or_intern_static("self"));
                if let Some((_sid, info)) = self.scopes.lookup(self.current_scope, self_sym) {
                    info.ty.clone()
                } else {
                    self.diagnostics.push(
                        Diagnostic::error("`self` outside of model method")
                            .with_label(expr.span, "not inside a method"),
                    );
                    Type::Error
                }
            }
            ExprKind::BinaryOp { left, op, right } => {
                self.check_binary_op(left, *op, right, expr.span)
            }
            ExprKind::UnaryOp { op, operand } => self.check_unary_op(*op, operand, expr.span),
            ExprKind::Call { callee, args } => self.check_call(callee, args, expr.span),
            ExprKind::MemberAccess { object, member } => {
                self.check_member_access(object, *member, expr.span)
            }
            ExprKind::Subscript { object, index } => {
                let obj_ty = self.check_expr(object);
                // Type-check the index expression(s)
                self.check_subscript_kind(index);
                match &obj_ty {
                    Type::List(elem) => *elem.clone(),
                    Type::Dict(_, val) => *val.clone(),
                    Type::Str => Type::Str,
                    Type::Tuple(elems) => {
                        if elems.is_empty() {
                            Type::Unknown
                        } else if let nsl_ast::expr::SubscriptKind::Index(idx_expr) = index.as_ref() {
                            if let ExprKind::IntLiteral(i) = &idx_expr.kind {
                                let idx = if *i < 0 {
                                    (*i + elems.len() as i64) as usize
                                } else {
                                    *i as usize
                                };
                                if idx < elems.len() {
                                    elems[idx].clone()
                                } else {
                                    Type::Unknown
                                }
                            } else {
                                Type::Unknown
                            }
                        } else {
                            // Slice or multi-dim on tuple
                            Type::Unknown
                        }
                    }
                    _ => Type::Unknown,
                }
            }
            ExprKind::Pipe { left, right } => {
                let left_ty = self.check_expr(left);
                let right_ty = self.check_expr(right);
                match &right_ty {
                    Type::Function { ret, .. } => *ret.clone(),
                    _ => {
                        // Could be a callable model or unknown function
                        let _ = left_ty;
                        Type::Unknown
                    }
                }
            }
            ExprKind::ListLiteral(items) => {
                if items.is_empty() {
                    Type::List(Box::new(Type::Unknown))
                } else {
                    let elem_ty = self.check_expr(&items[0]);
                    for item in &items[1..] {
                        self.check_expr(item);
                    }
                    Type::List(Box::new(elem_ty))
                }
            }
            ExprKind::TupleLiteral(items) => {
                let types: Vec<Type> = items.iter().map(|e| self.check_expr(e)).collect();
                Type::Tuple(types)
            }
            ExprKind::DictLiteral(entries) => {
                if entries.is_empty() {
                    Type::Dict(Box::new(Type::Unknown), Box::new(Type::Unknown))
                } else {
                    // Dict literal keys that are bare identifiers are treated as
                    // string-like keys (e.g. { vocab_size: 50257 }), not variable lookups.
                    let key_ty = if matches!(entries[0].0.kind, ExprKind::Ident(_)) {
                        Type::Str
                    } else {
                        self.check_expr(&entries[0].0)
                    };
                    let mut val_ty = self.check_expr(&entries[0].1);
                    for (k, v) in &entries[1..] {
                        if !matches!(k.kind, ExprKind::Ident(_)) {
                            self.check_expr(k);
                        }
                        let v_ty = self.check_expr(v);
                        // Widen to Unknown if values have mixed types
                        if val_ty != v_ty && val_ty != Type::Unknown {
                            val_ty = Type::Unknown;
                        }
                    }
                    Type::Dict(Box::new(key_ty), Box::new(val_ty))
                }
            }
            ExprKind::Lambda { params, body } => {
                let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Lambda);
                let prev = self.current_scope;
                self.current_scope = scope;
                let param_types: Vec<Type> = params
                    .iter()
                    .map(|p| {
                        let ty = p
                            .type_ann
                            .as_ref()
                            .map(|t| self.resolve_type(t))
                            .unwrap_or(Type::Unknown);
                        self.declare_symbol(p.name, ty.clone(), p.span, false, true);
                        ty
                    })
                    .collect();
                let body_ty = self.check_expr(body);
                self.current_scope = prev;
                Type::Function {
                    params: param_types,
                    ret: Box::new(body_ty),
                }
            }
            ExprKind::ListComp {
                element,
                generators,
            } => {
                let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Block);
                let prev = self.current_scope;
                self.current_scope = scope;
                for gen in generators {
                    let iter_ty = self.check_expr(&gen.iterable);
                    let elem_ty = match &iter_ty {
                        Type::List(e) => *e.clone(),
                        _ => Type::Unknown,
                    };
                    self.declare_pattern(&gen.pattern, &elem_ty);
                    for cond in &gen.conditions {
                        self.check_expr(cond);
                    }
                }
                let elem_ty = self.check_expr(element);
                self.current_scope = prev;
                Type::List(Box::new(elem_ty))
            }
            ExprKind::IfExpr {
                condition,
                then_expr,
                else_expr,
            } => {
                self.check_expr(condition);
                let then_ty = self.check_expr(then_expr);
                let else_ty = self.check_expr(else_expr);
                // Use the more specific type when one branch is Unknown
                if matches!(then_ty, Type::Unknown) {
                    else_ty
                } else {
                    then_ty
                }
            }
            ExprKind::Range { start, end, .. } => {
                if let Some(s) = start {
                    self.check_expr(s);
                }
                if let Some(e) = end {
                    self.check_expr(e);
                }
                Type::List(Box::new(Type::Int))
            }
            ExprKind::Paren(inner) => self.check_expr(inner),
            ExprKind::Await(inner) => self.check_expr(inner),
            ExprKind::MatchExpr { subject, arms } => {
                self.check_expr(subject);
                let mut result_ty = Type::Unknown;
                for arm in arms {
                    let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Block);
                    let prev = self.current_scope;
                    self.current_scope = scope;
                    self.declare_pattern(&arm.pattern, &Type::Unknown);
                    if let Some(guard) = &arm.guard {
                        self.check_expr(guard);
                    }
                    for s in &arm.body.stmts {
                        self.check_stmt(s);
                    }
                    self.current_scope = prev;
                    // Use first arm's last expr type as result
                    if matches!(result_ty, Type::Unknown) {
                        if let Some(last) = arm.body.stmts.last() {
                            if let StmtKind::Expr(e) = &last.kind {
                                result_ty = self.type_map.get(&e.id).cloned().unwrap_or(Type::Unknown);
                            }
                        }
                    }
                }
                result_ty
            }
            ExprKind::Error => Type::Error,
        };

        self.type_map.insert(expr.id, ty.clone());
        ty
    }

    fn check_binary_op(&mut self, left: &Expr, op: BinOp, right: &Expr, span: Span) -> Type {
        let lty = self.check_expr(left);
        let rty = self.check_expr(right);

        if lty.is_indeterminate() || rty.is_indeterminate() {
            return if matches!(lty, Type::Error) || matches!(rty, Type::Error) {
                Type::Error
            } else {
                Type::Unknown
            };
        }

        match op {
            BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div | BinOp::FloorDiv | BinOp::Mod | BinOp::Pow => {
                self.check_arithmetic(&lty, &rty, op, span)
            }
            BinOp::MatMul => self.check_matmul_op(&lty, &rty, span),
            BinOp::Eq | BinOp::NotEq | BinOp::Lt | BinOp::Gt | BinOp::LtEq | BinOp::GtEq => {
                Type::Bool
            }
            BinOp::And | BinOp::Or => Type::Bool,
            BinOp::Is | BinOp::In => Type::Bool,
            BinOp::BitOr | BinOp::BitAnd => lty,
        }
    }

    fn check_arithmetic(&mut self, lty: &Type, rty: &Type, op: BinOp, span: Span) -> Type {
        match (lty, rty) {
            (Type::Int, Type::Int) => {
                if matches!(op, BinOp::Div) {
                    Type::Float
                } else {
                    Type::Int
                }
            }
            (Type::Float, Type::Float)
            | (Type::Int, Type::Float)
            | (Type::Float, Type::Int) => Type::Float,
            // Tensor element-wise ops
            (
                Type::Tensor {
                    shape: ls,
                    dtype: ld,
                    device: ldev,
                },
                Type::Tensor {
                    shape: rs,
                    dtype: rd,
                    device: rdev,
                },
            ) => {
                if ldev != rdev
                    && !matches!(ldev, Device::Unknown)
                    && !matches!(rdev, Device::Unknown)
                {
                    self.diagnostics.push(
                        Diagnostic::error("cannot operate on tensors on different devices")
                            .with_label(span, format!("{} vs {}", display_device(ldev), display_device(rdev))),
                    );
                }
                match shapes::check_elementwise(ls, rs, span) {
                    Ok(result_shape) => Type::Tensor {
                        shape: result_shape,
                        dtype: wider_dtype(*ld, *rd),
                        device: ldev.clone(),
                    },
                    Err(diag) => {
                        self.diagnostics.push(diag);
                        Type::Error
                    }
                }
            }
            // Scalar + tensor
            (Type::Tensor { .. }, Type::Int | Type::Float) => lty.clone(),
            (Type::Int | Type::Float, Type::Tensor { .. }) => rty.clone(),
            // String concatenation
            (Type::Str, Type::Str) if matches!(op, BinOp::Add) => Type::Str,
            // String repeat
            (Type::Str, Type::Int) if matches!(op, BinOp::Mul) => Type::Str,
            (Type::Int, Type::Str) if matches!(op, BinOp::Mul) => Type::Str,
            // Specific numeric types (e.g. f32 + f32, int32 + int32)
            (l, r) if dtype_rank(l).0 > 0 && dtype_rank(r).0 > 0 => {
                let (lf, lr) = dtype_rank(l);
                let (rf, rr) = dtype_rank(r);
                if lf == rf {
                    // Same family: return the wider type
                    if lr >= rr { l.clone() } else { r.clone() }
                } else {
                    // Mixed int/float: promote to float side
                    if lf == 2 { l.clone() } else { r.clone() }
                }
            }
            _ => {
                self.diagnostics.push(
                    Diagnostic::error(format!(
                        "unsupported operand types for {:?}: {} and {}",
                        op, display_type(lty), display_type(rty)
                    ))
                    .with_label(span, "type error"),
                );
                Type::Error
            }
        }
    }

    fn check_matmul_op(&mut self, lty: &Type, rty: &Type, span: Span) -> Type {
        match (lty, rty) {
            (
                Type::Tensor {
                    shape: ls,
                    dtype: ld,
                    device: ldev,
                },
                Type::Tensor {
                    shape: rs,
                    dtype: rd,
                    device: rdev,
                },
            ) => {
                if ldev != rdev
                    && !matches!(ldev, Device::Unknown)
                    && !matches!(rdev, Device::Unknown)
                {
                    self.diagnostics.push(
                        Diagnostic::error("matmul: device mismatch")
                            .with_label(span, format!("{} vs {}", display_device(ldev), display_device(rdev))),
                    );
                }
                match shapes::check_matmul(ls, rs, span) {
                    Ok(result_shape) => Type::Tensor {
                        shape: result_shape,
                        dtype: wider_dtype(*ld, *rd),
                        device: ldev.clone(),
                    },
                    Err(diag) => {
                        self.diagnostics.push(diag);
                        Type::Error
                    }
                }
            }
            _ => {
                self.diagnostics.push(
                    Diagnostic::error("@ (matmul) requires tensor operands")
                        .with_label(span, "not a tensor"),
                );
                Type::Error
            }
        }
    }

    fn check_unary_op(&mut self, op: UnaryOp, operand: &Expr, _span: Span) -> Type {
        let ty = self.check_expr(operand);
        match op {
            UnaryOp::Neg => match &ty {
                Type::Int => Type::Int,
                Type::Float => Type::Float,
                Type::Tensor { .. } => ty,
                _ => ty,
            },
            UnaryOp::Not => Type::Bool,
        }
    }

    fn check_call(&mut self, callee: &Expr, args: &[Arg], span: Span) -> Type {
        let callee_ty = self.check_expr(callee);

        // Check each argument
        let arg_types: Vec<Type> = args.iter().map(|a| self.check_expr(&a.value)).collect();

        // Special type inference for enumerate(list) → List(Tuple(Int, T))
        if let ExprKind::Ident(sym) = &callee.kind {
            let name = self.interner.resolve(sym.0).unwrap_or("").to_string();
            if name == "enumerate" {
                if let Some(Type::List(elem_ty)) = arg_types.first() {
                    return Type::List(Box::new(Type::Tuple(vec![Type::Int, *elem_ty.clone()])));
                }
            }
            if name == "zip" {
                if arg_types.len() >= 2 {
                    let a = match &arg_types[0] {
                        Type::List(t) => *t.clone(),
                        _ => Type::Unknown,
                    };
                    let b = match &arg_types[1] {
                        Type::List(t) => *t.clone(),
                        _ => Type::Unknown,
                    };
                    return Type::List(Box::new(Type::Tuple(vec![a, b])));
                }
            }

            // Tensor creation shape inference
            if matches!(name.as_str(), "zeros" | "ones" | "rand" | "randn" | "empty") {
                let shape = self.extract_shape_from_args(args);
                return Type::Tensor {
                    shape,
                    dtype: DType::F64,
                    device: Device::Cpu,
                };
            }
            if name == "full" {
                let shape = self.extract_shape_from_args(args);
                return Type::Tensor {
                    shape,
                    dtype: DType::F64,
                    device: Device::Cpu,
                };
            }
            if name == "arange" {
                return Type::Tensor {
                    shape: Shape::unknown(),
                    dtype: DType::F64,
                    device: Device::Cpu,
                };
            }

            // Math builtins (exp, log, sqrt, sin, cos, abs) — when called with tensor or
            // unknown args, return the arg type (tensor) instead of Float.
            if matches!(name.as_str(), "exp" | "log" | "sqrt" | "sin" | "cos" | "abs" | "neg" | "floor") {
                if let Some(first_arg_ty) = arg_types.first() {
                    if first_arg_ty.is_tensor() || first_arg_ty.is_indeterminate() {
                        return first_arg_ty.clone();
                    }
                }
            }

            // Tensor reduction / manipulation builtins — always return tensor-like
            if matches!(name.as_str(), "mean" | "sum" | "reduce_max" | "gather" | "clamp" | "neg") {
                if let Some(first_arg_ty) = arg_types.first() {
                    if first_arg_ty.is_tensor() || first_arg_ty.is_indeterminate() {
                        return first_arg_ty.clone();
                    }
                }
            }
        }

        // Tensor method shape inference (reshape, transpose)
        if let ExprKind::MemberAccess { object, member } = &callee.kind {
            let obj_ty = self.type_map.get(&object.id).cloned().unwrap_or(Type::Unknown);
            if obj_ty.is_tensor() {
                let method_name = self.interner.resolve(member.0).unwrap_or("").to_string();
                if let Type::Tensor { dtype, device, .. } = &obj_ty {
                    match method_name.as_str() {
                        "reshape" => {
                            let shape = self.extract_shape_from_args(args);
                            return Type::Tensor {
                                shape,
                                dtype: *dtype,
                                device: device.clone(),
                            };
                        }
                        "transpose" => {
                            if let Type::Tensor { shape, dtype, device } = &obj_ty {
                                if shape.rank() >= 2 && args.len() >= 2 {
                                    let d0 = match &args[0].value.kind {
                                        ExprKind::IntLiteral(n) => Some(*n as usize),
                                        _ => None,
                                    };
                                    let d1 = match &args[1].value.kind {
                                        ExprKind::IntLiteral(n) => Some(*n as usize),
                                        _ => None,
                                    };
                                    if let (Some(d0), Some(d1)) = (d0, d1) {
                                        if d0 < shape.rank() && d1 < shape.rank() {
                                            let mut new_dims = shape.dims.clone();
                                            new_dims.swap(d0, d1);
                                            return Type::Tensor {
                                                shape: Shape { dims: new_dims },
                                                dtype: *dtype,
                                                device: device.clone(),
                                            };
                                        }
                                    }
                                }
                                // Can't determine statically — return unknown shape
                                return Type::Tensor {
                                    shape: Shape::unknown(),
                                    dtype: *dtype,
                                    device: device.clone(),
                                };
                            }
                        }
                        _ => {}
                    }
                }
            }
        }

        match &callee_ty {
            Type::Function { params, ret } => {
                // Check arity (allow flexible arity if params has Unknown — variadic builtins)
                let is_variadic = params.iter().any(|p| matches!(p, Type::Unknown));
                if arg_types.len() > params.len() && !is_variadic {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "too many arguments: expected {}, got {}",
                            params.len(),
                            arg_types.len()
                        ))
                        .with_label(span, "extra arguments"),
                    );
                }
                if arg_types.len() < params.len() && !is_variadic {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "too few arguments: expected {}, got {}",
                            params.len(),
                            arg_types.len()
                        ))
                        .with_label(span, "missing arguments"),
                    );
                }
                // Check each arg type against param type
                for (i, (arg_ty, param_ty)) in
                    arg_types.iter().zip(params.iter()).enumerate()
                {
                    if !is_assignable(arg_ty, param_ty) {
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "argument {}: expected {}, got {}",
                                i + 1,
                                display_type(param_ty),
                                display_type(arg_ty)
                            ))
                            .with_label(args[i].span, "type mismatch"),
                        );
                    }
                }
                *ret.clone()
            }
            Type::Model { .. } => callee_ty.clone(),
            Type::Struct { .. } => callee_ty.clone(),
            Type::Error => Type::Error,
            Type::Unknown => Type::Unknown,
            _ => {
                self.diagnostics.push(
                    Diagnostic::error("expression is not callable")
                        .with_label(span, format!("type {} is not callable", display_type(&callee_ty))),
                );
                Type::Error
            }
        }
    }

    /// Try to extract a concrete tensor shape from a function call's arguments.
    /// Returns Shape::unknown() if the first arg isn't a list literal or contains non-integer elements.
    fn extract_shape_from_args(&self, args: &[Arg]) -> Shape {
        if args.is_empty() {
            return Shape::unknown();
        }
        match &args[0].value.kind {
            ExprKind::ListLiteral(elems) => {
                let mut dims = Vec::new();
                for elem in elems {
                    match &elem.kind {
                        ExprKind::IntLiteral(n) => dims.push(Dim::Concrete(*n)),
                        ExprKind::Ident(sym) => dims.push(Dim::Symbolic(*sym)),
                        _ => return Shape::unknown(),
                    }
                }
                Shape { dims }
            }
            _ => Shape::unknown(),
        }
    }

    fn check_subscript_kind(&mut self, kind: &nsl_ast::expr::SubscriptKind) {
        match kind {
            nsl_ast::expr::SubscriptKind::Index(expr) => {
                self.check_expr(expr);
            }
            nsl_ast::expr::SubscriptKind::Slice { lower, upper, step } => {
                if let Some(e) = lower { self.check_expr(e); }
                if let Some(e) = upper { self.check_expr(e); }
                if let Some(e) = step { self.check_expr(e); }
            }
            nsl_ast::expr::SubscriptKind::MultiDim(dims) => {
                for d in dims {
                    self.check_subscript_kind(d);
                }
            }
        }
    }

    fn check_member_access(&mut self, object: &Expr, member: Symbol, span: Span) -> Type {
        let obj_ty = self.check_expr(object);

        match &obj_ty {
            Type::Module { exports } => {
                if let Some(ty) = exports.get(&member) {
                    return *ty.clone();
                }
                let name = self.resolve_name(member);
                self.diagnostics.push(
                    Diagnostic::error(format!("module has no export `{name}`"))
                        .with_label(span, "not found in module"),
                );
                return Type::Error;
            }
            Type::Struct { fields, .. } => {
                if let Some((_, field_ty)) = fields.iter().find(|(name, _)| *name == member) {
                    field_ty.clone()
                } else {
                    let name = self.resolve_name(member);
                    self.diagnostics.push(
                        Diagnostic::error(format!("no field `{name}` on struct"))
                            .with_label(span, "unknown field"),
                    );
                    Type::Error
                }
            }
            Type::Model { fields, methods, .. } => {
                if let Some((_, field_ty)) = fields.iter().find(|(name, _)| *name == member) {
                    return field_ty.clone();
                }
                if let Some((_, method_ty)) = methods.iter().find(|(name, _)| *name == member) {
                    return method_ty.clone();
                }
                // Models have many dynamic attributes, be lenient
                Type::Unknown
            }
            Type::Tensor { .. } | Type::Param { .. } | Type::Buffer { .. } => {
                // Tensors have built-in attributes
                let name = self.resolve_name(member);
                match name.as_str() {
                    "shape" => Type::List(Box::new(Type::Int)),
                    "ndim" => Type::Int,
                    "dtype" | "device" => Type::Unknown,
                    "T" => obj_ty.clone(), // transpose
                    "sum" => Type::Function {
                        params: vec![],
                        ret: Box::new(Type::Tensor {
                            shape: crate::types::Shape::unknown(),
                            dtype: crate::types::DType::Unknown,
                            device: crate::types::Device::Unknown,
                        }),
                    },
                    "mean" => Type::Function {
                        params: vec![],
                        ret: Box::new(Type::Tensor {
                            shape: crate::types::Shape::unknown(),
                            dtype: crate::types::DType::Unknown,
                            device: crate::types::Device::Unknown,
                        }),
                    },
                    "reshape" => Type::Function {
                        params: vec![Type::List(Box::new(Type::Int))],
                        ret: Box::new(obj_ty.clone()),
                    },
                    "transpose" => Type::Function {
                        params: vec![Type::Int, Type::Int],
                        ret: Box::new(obj_ty.clone()),
                    },
                    "clone" => Type::Function {
                        params: vec![],
                        ret: Box::new(obj_ty.clone()),
                    },
                    "item" => Type::Function {
                        params: vec![],
                        ret: Box::new(Type::Float),
                    },
                    _ => Type::Unknown,    // tensor methods
                }
            }
            Type::Str => {
                let name = self.resolve_name(member);
                match name.as_str() {
                    // String methods that return strings
                    "upper" | "lower" | "strip" | "replace" | "join" => Type::Function {
                        params: vec![Type::Unknown],
                        ret: Box::new(Type::Str),
                    },
                    // String methods that return lists
                    "split" => Type::Function {
                        params: vec![Type::Unknown],
                        ret: Box::new(Type::List(Box::new(Type::Str))),
                    },
                    // String methods that return int
                    "find" => Type::Function {
                        params: vec![Type::Unknown],
                        ret: Box::new(Type::Int),
                    },
                    // String methods that return bool
                    "startswith" | "endswith" | "contains" => Type::Function {
                        params: vec![Type::Unknown],
                        ret: Box::new(Type::Bool),
                    },
                    _ => Type::Unknown,
                }
            }
            Type::List(_) => {
                let name = self.resolve_name(member);
                match name.as_str() {
                    "append" | "push" => Type::Function {
                        params: vec![Type::Unknown],
                        ret: Box::new(Type::Void),
                    },
                    "len" => Type::Function {
                        params: vec![],
                        ret: Box::new(Type::Int),
                    },
                    _ => Type::Unknown,
                }
            }
            Type::Error => Type::Error,
            Type::Unknown => Type::Unknown,
            _ => Type::Unknown,
        }
    }

    // ===== Helpers =====

    fn check_block(&mut self, block: &Block, kind: ScopeKind) {
        let scope = self.scopes.push_scope(self.current_scope, kind);
        let prev = self.current_scope;
        self.current_scope = scope;
        for s in &block.stmts {
            self.check_stmt(s);
        }
        self.current_scope = prev;
    }

    fn build_fn_type(&mut self, fn_def: &FnDef) -> Type {
        let params: Vec<Type> = fn_def
            .params
            .iter()
            .filter_map(|p| {
                let name = self.resolve_name(p.name);
                if name == "self" {
                    return None;
                }
                Some(
                    p.type_ann
                        .as_ref()
                        .map(|t| self.resolve_type(t))
                        .unwrap_or(Type::Unknown),
                )
            })
            .collect();
        let ret = fn_def
            .return_type
            .as_ref()
            .map(|t| self.resolve_type(t))
            .unwrap_or(Type::Void);
        Type::Function {
            params,
            ret: Box::new(ret),
        }
    }

    fn resolve_type(&mut self, type_expr: &TypeExpr) -> Type {
        let mut resolver = TypeResolver {
            interner: self.interner,
            scopes: self.scopes,
            diagnostics: &mut self.diagnostics,
        };
        resolver.resolve(type_expr, self.current_scope)
    }

    fn declare_symbol(
        &mut self,
        name: Symbol,
        ty: Type,
        span: Span,
        is_const: bool,
        is_param: bool,
    ) {
        let info = SymbolInfo {
            ty,
            def_span: span,
            is_const,
            is_param,
            is_used: false,
        };
        if self.scopes.declare(self.current_scope, name, info.clone()).is_err() {
            // NSL allows rebinding (Python-like semantics), so update the existing binding
            if let Some(existing) = self.scopes.lookup_mut(self.current_scope, name) {
                existing.ty = info.ty;
                existing.is_const = info.is_const;
                existing.def_span = info.def_span;
            }
        }
    }

    fn declare_pattern(&mut self, pattern: &Pattern, ty: &Type) {
        self.declare_pattern_with_const(pattern, ty, false);
    }

    fn declare_pattern_with_const(&mut self, pattern: &Pattern, ty: &Type, is_const: bool) {
        match &pattern.kind {
            PatternKind::Ident(sym) => {
                self.declare_symbol(*sym, ty.clone(), pattern.span, is_const, false);
            }
            PatternKind::Tuple(pats) => {
                if let Type::Tuple(types) = ty {
                    for (p, t) in pats.iter().zip(types.iter()) {
                        self.declare_pattern_with_const(p, t, is_const);
                    }
                } else {
                    for p in pats {
                        self.declare_pattern_with_const(p, &Type::Unknown, is_const);
                    }
                }
            }
            PatternKind::Typed { pattern, type_ann } => {
                let ann_ty = self.resolve_type(type_ann);
                self.declare_pattern_with_const(pattern, &ann_ty, is_const);
            }
            PatternKind::Wildcard => {} // Don't bind anything
            PatternKind::Rest(Some(sym)) => {
                self.declare_symbol(*sym, Type::Unknown, pattern.span, is_const, false);
            }
            _ => {} // Literal, Constructor, etc. — no bindings in M2
        }
    }

    fn resolve_name(&self, sym: Symbol) -> String {
        self.interner
            .resolve(sym.0)
            .unwrap_or("<unknown>")
            .to_string()
    }
}

/// Simple glob matching supporting `*` as a wildcard that matches any number
/// of characters (including zero). Uses a two-pointer / DP-lite approach.
fn glob_match(pattern: &str, text: &str) -> bool {
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();
    let (plen, tlen) = (pat.len(), txt.len());

    let mut pi = 0;
    let mut ti = 0;
    let mut star_pi: Option<usize> = None;
    let mut star_ti = 0;

    while ti < tlen {
        if pi < plen && (pat[pi] == txt[ti] || pat[pi] == '?') {
            pi += 1;
            ti += 1;
        } else if pi < plen && pat[pi] == '*' {
            star_pi = Some(pi);
            star_ti = ti;
            pi += 1;
        } else if let Some(sp) = star_pi {
            pi = sp + 1;
            star_ti += 1;
            ti = star_ti;
        } else {
            return false;
        }
    }

    while pi < plen && pat[pi] == '*' {
        pi += 1;
    }

    pi == plen
}
