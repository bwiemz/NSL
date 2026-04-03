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
                    effect: Effect::Inferred,
                }
            }
            ExprKind::BlockExpr(block) => self.check_block_expr(block),
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

                let is_parser_placeholder = |branch: &Expr| {
                    matches!(branch.kind, ExprKind::NoneLiteral) && branch.span.is_empty()
                };

                let branch_missing_value = |branch: &Expr| match &branch.kind {
                    ExprKind::BlockExpr(block) => !Self::block_expr_has_reachable_value(block),
                    _ => false,
                };

                if is_parser_placeholder(then_expr) || branch_missing_value(then_expr) {
                    self.diagnostics.push(
                        Diagnostic::error("if-expression then-branch must end with a value")
                            .with_label(then_expr.span, "this branch does not yield a value"),
                    );
                }

                if is_parser_placeholder(else_expr) {
                    self.diagnostics.push(
                        Diagnostic::error("if-expression requires an else branch that yields a value")
                            .with_label(expr.span, "add an else branch with a value"),
                    );
                }

                if branch_missing_value(else_expr) {
                    self.diagnostics.push(
                        Diagnostic::error("if-expression else-branch must end with a value")
                            .with_label(else_expr.span, "this branch does not yield a value"),
                    );
                }

                if is_parser_placeholder(then_expr)
                    || branch_missing_value(then_expr)
                    || is_parser_placeholder(else_expr)
                    || branch_missing_value(else_expr)
                {
                    Type::Unknown
                } else if matches!(then_ty, Type::Unknown) {
                    else_ty
                } else if matches!(else_ty, Type::Unknown) {
                    then_ty
                } else if is_assignable(&then_ty, &else_ty) {
                    if matches!(else_ty, Type::Function { .. }) {
                        self.diagnostics.push(
                            Diagnostic::error(
                                "if-expression function values are not supported yet"
                            )
                            .with_label(
                                expr.span,
                                "lift the function selection into a statement or bind each branch separately",
                            ),
                        );
                        Type::Unknown
                    } else {
                        else_ty
                    }
                } else if is_assignable(&else_ty, &then_ty) {
                    if matches!(then_ty, Type::Function { .. }) {
                        self.diagnostics.push(
                            Diagnostic::error(
                                "if-expression function values are not supported yet"
                            )
                            .with_label(
                                expr.span,
                                "lift the function selection into a statement or bind each branch separately",
                            ),
                        );
                        Type::Unknown
                    } else {
                        then_ty
                    }
                } else {
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "if-expression branch type mismatch: then branch has type {}, else branch has type {}",
                            display_type(&then_ty),
                            display_type(&else_ty)
                        ))
                        .with_label(
                            then_expr.span,
                            format!("then branch is {}", display_type(&then_ty)),
                        )
                        .with_label(
                            else_expr.span,
                            format!("else branch is {}", display_type(&else_ty)),
                        ),
                    );
                    Type::Unknown
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

impl<'a> TypeChecker<'a> {
    fn check_block_expr(&mut self, block: &Block) -> Type {
        self.check_block_expr_shadowing(block, self.current_scope);

        let scope = self.scopes.push_scope(self.current_scope, ScopeKind::Block);
        let prev = self.current_scope;
        self.current_scope = scope;

        for stmt in &block.stmts {
            self.check_stmt(stmt);
        }

        let result_ty = if Self::block_expr_has_reachable_value(block) {
            let last = block.stmts.last().expect("reachable block expression must have a tail statement");
            if let StmtKind::Expr(expr) = &last.kind {
                self.type_map.get(&expr.id).cloned().unwrap_or(Type::Unknown)
            } else {
                Type::Unknown
            }
        } else {
            Type::Unknown
        };

        self.current_scope = prev;
        result_ty
    }

    fn block_expr_has_reachable_value(block: &Block) -> bool {
        let Some(last) = block.stmts.last() else {
            return false;
        };

        if !matches!(last.kind, StmtKind::Expr(_)) {
            return false;
        }

        block.stmts[..block.stmts.len().saturating_sub(1)]
            .iter()
            .all(|stmt| !Self::stmt_terminates_control_flow(stmt))
    }

    fn stmt_terminates_control_flow(stmt: &Stmt) -> bool {
        match &stmt.kind {
            StmtKind::Return(_) | StmtKind::Yield(_) | StmtKind::Break | StmtKind::Continue => {
                true
            }
            StmtKind::If {
                then_block,
                elif_clauses,
                else_block,
                ..
            } => {
                Self::block_terminates_control_flow(then_block)
                    && elif_clauses
                        .iter()
                        .all(|(_, block)| Self::block_terminates_control_flow(block))
                    && else_block
                        .as_ref()
                        .is_some_and(Self::block_terminates_control_flow)
            }
            StmtKind::Match { arms, .. } => {
                !arms.is_empty()
                    && arms
                        .iter()
                        .all(|arm| Self::block_terminates_control_flow(&arm.body))
            }
            _ => false,
        }
    }

    fn block_terminates_control_flow(block: &Block) -> bool {
        block.stmts
            .last()
            .is_some_and(Self::stmt_terminates_control_flow)
    }

    fn check_block_expr_shadowing(&mut self, block: &Block, outer_scope: ScopeId) {
        let mut visible = std::collections::HashSet::new();
        for stmt in &block.stmts {
            self.check_block_expr_stmt_shadowing(stmt, outer_scope, &mut visible);
        }
    }

    fn check_block_expr_stmt_shadowing(
        &mut self,
        stmt: &Stmt,
        outer_scope: ScopeId,
        visible: &mut std::collections::HashSet<Symbol>,
    ) {
        match &stmt.kind {
            StmtKind::VarDecl { pattern, .. } => {
                self.check_block_expr_pattern_shadowing(pattern, outer_scope, visible);
            }
            StmtKind::If {
                then_block,
                elif_clauses,
                else_block,
                ..
            } => {
                let mut then_visible = visible.clone();
                self.check_block_expr_shadowing_with_visible(then_block, outer_scope, &mut then_visible);
                for (_, block) in elif_clauses {
                    let mut elif_visible = visible.clone();
                    self.check_block_expr_shadowing_with_visible(block, outer_scope, &mut elif_visible);
                }
                if let Some(block) = else_block {
                    let mut else_visible = visible.clone();
                    self.check_block_expr_shadowing_with_visible(block, outer_scope, &mut else_visible);
                }
            }
            StmtKind::For { pattern, body, .. } => {
                self.check_block_expr_pattern_shadowing(pattern, outer_scope, visible);
                let mut body_visible = visible.clone();
                self.check_block_expr_shadowing_with_visible(body, outer_scope, &mut body_visible);
            }
            StmtKind::While { body, .. } => {
                let mut body_visible = visible.clone();
                self.check_block_expr_shadowing_with_visible(body, outer_scope, &mut body_visible);
            }
            StmtKind::WhileLet { pattern, body, .. } => {
                self.check_block_expr_pattern_shadowing(pattern, outer_scope, visible);
                let mut body_visible = visible.clone();
                self.check_block_expr_shadowing_with_visible(body, outer_scope, &mut body_visible);
            }
            StmtKind::Match { arms, .. } => {
                for arm in arms {
                    let mut arm_visible = visible.clone();
                    self.check_block_expr_pattern_shadowing(&arm.pattern, outer_scope, &mut arm_visible);
                    self.check_block_expr_shadowing_with_visible(&arm.body, outer_scope, &mut arm_visible);
                }
            }
            _ => {}
        }
    }

    fn check_block_expr_shadowing_with_visible(
        &mut self,
        block: &Block,
        outer_scope: ScopeId,
        visible: &mut std::collections::HashSet<Symbol>,
    ) {
        for stmt in &block.stmts {
            self.check_block_expr_stmt_shadowing(stmt, outer_scope, visible);
        }
    }

    fn check_block_expr_pattern_shadowing(
        &mut self,
        pattern: &Pattern,
        outer_scope: ScopeId,
        visible: &mut std::collections::HashSet<Symbol>,
    ) {
        match &pattern.kind {
            PatternKind::Ident(sym) => {
                if visible.contains(sym) || self.scopes.lookup(outer_scope, *sym).is_some() {
                    let name = self.resolve_name(*sym);
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "shadowing outer binding `{name}` inside an if-expression block is not supported yet"
                        ))
                        .with_label(pattern.span, "rename this binding or lift it out of the if-expression"),
                    );
                }
                visible.insert(*sym);
            }
            PatternKind::Tuple(items) | PatternKind::List(items) => {
                for item in items {
                    self.check_block_expr_pattern_shadowing(item, outer_scope, visible);
                }
            }
            PatternKind::Struct { fields, rest } => {
                for field in fields {
                    if let Some(pattern) = &field.pattern {
                        self.check_block_expr_pattern_shadowing(pattern, outer_scope, visible);
                    } else {
                        self.check_block_expr_pattern_shadowing(
                            &Pattern {
                                kind: PatternKind::Ident(field.name),
                                span: field.span,
                                id: pattern.id,
                            },
                            outer_scope,
                            visible,
                        );
                    }
                }
                if let Some(rest) = rest {
                    if visible.contains(rest) || self.scopes.lookup(outer_scope, *rest).is_some() {
                        let name = self.resolve_name(*rest);
                        self.diagnostics.push(
                            Diagnostic::error(format!(
                                "shadowing outer binding `{name}` inside an if-expression block is not supported yet"
                            ))
                            .with_label(pattern.span, "rename this binding or lift it out of the if-expression"),
                        );
                    }
                    visible.insert(*rest);
                }
            }
            PatternKind::Constructor { args, .. } | PatternKind::Or(args) => {
                for arg in args {
                    self.check_block_expr_pattern_shadowing(arg, outer_scope, visible);
                }
            }
            PatternKind::Guarded { pattern: inner, .. } => {
                self.check_block_expr_pattern_shadowing(inner, outer_scope, visible);
            }
            PatternKind::Rest(Some(sym)) => {
                if visible.contains(sym) || self.scopes.lookup(outer_scope, *sym).is_some() {
                    let name = self.resolve_name(*sym);
                    self.diagnostics.push(
                        Diagnostic::error(format!(
                            "shadowing outer binding `{name}` inside an if-expression block is not supported yet"
                        ))
                        .with_label(pattern.span, "rename this binding or lift it out of the if-expression"),
                    );
                }
                visible.insert(*sym);
            }
            PatternKind::Typed { pattern: inner, .. } => {
                self.check_block_expr_pattern_shadowing(inner, outer_scope, visible);
            }
            _ => {}
        }
    }
}
