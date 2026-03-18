use super::*;

impl<'a> TypeChecker<'a> {
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
                let subject_ty = self.check_expr(subject);
                let mut result_ty = Type::Unknown;
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
}
