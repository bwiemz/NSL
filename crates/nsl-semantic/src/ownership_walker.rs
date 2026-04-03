//! M38a: AST walker that drives the OwnershipChecker.
//!
//! Walks function bodies after type checking, registering tensor bindings
//! and tracking consumption. Produces diagnostics for use-after-move errors
//! and per-function ownership metadata for codegen.

use std::collections::HashMap;

use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::pattern::PatternKind;
use nsl_ast::stmt::{Stmt, StmtKind, Block};
use nsl_ast::decl::ModelMember;
use nsl_errors::Diagnostic;
use nsl_lexer::Interner;

use crate::checker::TypeMap;
use crate::ownership::OwnershipChecker;
use crate::FunctionOwnershipInfo;

/// Walk a module's function definitions and run ownership analysis on each.
/// Returns ownership diagnostics and per-function metadata.
pub fn analyze_ownership(
    module: &nsl_ast::Module,
    interner: &Interner,
    type_map: &TypeMap,
    _scopes: &crate::ScopeMap,
) -> (Vec<Diagnostic>, HashMap<String, FunctionOwnershipInfo>) {
    let mut all_diagnostics = Vec::new();
    let mut ownership_info = HashMap::new();

    for stmt in &module.stmts {
        walk_top_level_stmt(stmt, interner, type_map, &mut all_diagnostics, &mut ownership_info);
    }

    (all_diagnostics, ownership_info)
}

fn walk_top_level_stmt(
    stmt: &Stmt,
    interner: &Interner,
    type_map: &TypeMap,
    diagnostics: &mut Vec<Diagnostic>,
    ownership_info: &mut HashMap<String, FunctionOwnershipInfo>,
) {
    match &stmt.kind {
        StmtKind::FnDef(fn_def) => {
            let fn_name = interner.resolve(fn_def.name.0)
                .unwrap_or("?")
                .to_string();

            let mut checker = OwnershipChecker::new(interner);
            let mut info = FunctionOwnershipInfo::default();

            // Register function parameters as tensor bindings
            for param in &fn_def.params {
                if is_tensor_param(param) {
                    // All tensor params are linear by default (no @shared decorator on Param)
                    checker.register_binding(param.name, param.span, false);
                    info.linear_params.push(param.name);
                }
            }

            // Walk function body
            walk_block(&fn_def.body, &mut checker, type_map);

            // Finalize: check all owned tensors were consumed
            checker.check_unconsumed();

            diagnostics.append(&mut checker.diagnostics);
            ownership_info.insert(fn_name, info);
        }
        StmtKind::ModelDef(model_def) => {
            // Walk each method in the model
            for member in &model_def.members {
                if let ModelMember::Method(method, _decorators) = member {
                    let model_name = interner.resolve(model_def.name.0)
                        .unwrap_or("?")
                        .to_string();
                    let method_name = interner.resolve(method.name.0)
                        .unwrap_or("?")
                        .to_string();
                    let mangled = format!("__nsl_model_{model_name}_{method_name}");

                    let mut checker = OwnershipChecker::new(interner);
                    let mut info = FunctionOwnershipInfo::default();

                    // Register method parameters (skip self)
                    for param in &method.params {
                        if is_tensor_param(param) {
                            checker.register_binding(param.name, param.span, false);
                            info.linear_params.push(param.name);
                        }
                    }

                    walk_block(&method.body, &mut checker, type_map);
                    checker.check_unconsumed();
                    diagnostics.append(&mut checker.diagnostics);
                    ownership_info.insert(mangled, info);
                }
            }
        }
        StmtKind::Decorated { stmt, .. } => {
            walk_top_level_stmt(stmt, interner, type_map, diagnostics, ownership_info);
        }
        _ => {} // Skip non-function top-level items
    }
}

fn walk_block(block: &Block, checker: &mut OwnershipChecker<'_>, type_map: &TypeMap) {
    for stmt in &block.stmts {
        walk_stmt(stmt, checker, type_map);
    }
}

fn walk_stmt(stmt: &Stmt, checker: &mut OwnershipChecker<'_>, type_map: &TypeMap) {
    match &stmt.kind {
        StmtKind::VarDecl { pattern, value, .. } => {
            if let Some(expr) = value {
                walk_expr(expr, checker, type_map);
            }
            register_pattern_bindings(pattern, checker, type_map, value.as_ref());
        }
        StmtKind::Assign { target, value, .. } => {
            walk_expr(target, checker, type_map);
            walk_expr(value, checker, type_map);
        }
        StmtKind::Expr(expr) => walk_expr(expr, checker, type_map),
        StmtKind::Return(Some(expr)) | StmtKind::Yield(Some(expr)) => {
            walk_expr(expr, checker, type_map);
        }
        StmtKind::Return(None) | StmtKind::Yield(None) => {}
        StmtKind::If { condition, then_block, elif_clauses, else_block } => {
            walk_expr(condition, checker, type_map);

            let before = checker.snapshot_all();
            let mut branch_states = Vec::new();

            walk_block(then_block, checker, type_map);
            let after_then = checker.snapshot_all();
            checker.check_branch_local_unconsumed(&before, &after_then);
            branch_states.push(after_then.states.clone());

            checker.restore_all(before.clone());
            for (cond, block) in elif_clauses {
                walk_expr(cond, checker, type_map);
                walk_block(block, checker, type_map);
                let after_elif = checker.snapshot_all();
                checker.check_branch_local_unconsumed(&before, &after_elif);
                branch_states.push(after_elif.states.clone());
                checker.restore_all(before.clone());
            }
            if let Some(block) = else_block {
                walk_block(block, checker, type_map);
                let after_else = checker.snapshot_all();
                checker.check_branch_local_unconsumed(&before, &after_else);
                branch_states.push(after_else.states.clone());
            } else {
                branch_states.push(before.states.clone());
            }

            checker.restore_all(before.clone());

            checker.check_multi_branch_symmetry(
                &before.states,
                &branch_states,
                condition.span,
            );
        }
        StmtKind::While { condition, body, .. } => {
            walk_expr(condition, checker, type_map);
            checker.enter_loop();
            walk_block(body, checker, type_map);
            checker.exit_loop();
        }
        StmtKind::WhileLet { pattern, expr, body } => {
            walk_expr(expr, checker, type_map);
            // The bound value has the same type as the expression
            register_pattern_bindings(pattern, checker, type_map, Some(expr));
            checker.enter_loop();
            walk_block(body, checker, type_map);
            checker.exit_loop();
        }
        StmtKind::For { pattern, iterable, body } => {
            walk_expr(iterable, checker, type_map);
            let elem_ty = iter_element_type(type_map.get(&iterable.id));
            register_pattern_bindings_with_type(pattern, checker, &elem_ty);
            checker.enter_loop();
            walk_block(body, checker, type_map);
            checker.exit_loop();
        }
        StmtKind::Match { subject, arms } => {
            walk_expr(subject, checker, type_map);
            for arm in arms {
                // Register match arm pattern bindings (subject's type)
                register_pattern_bindings(
                    &arm.pattern, checker, type_map, Some(subject),
                );
                if let Some(guard) = &arm.guard {
                    walk_expr(guard, checker, type_map);
                }
                walk_block(&arm.body, checker, type_map);
            }
        }
        StmtKind::Break | StmtKind::Continue => {}
        StmtKind::Decorated { stmt, .. } => walk_stmt(stmt, checker, type_map),
        _ => {}
    }
}

fn walk_expr(expr: &Expr, checker: &mut OwnershipChecker<'_>, type_map: &TypeMap) {
    match &expr.kind {
        ExprKind::Ident(sym) => {
            checker.use_binding(*sym, expr.span, "expression");
        }
        ExprKind::BinaryOp { left, right, .. } => {
            walk_expr(left, checker, type_map);
            walk_expr(right, checker, type_map);
        }
        ExprKind::UnaryOp { operand, .. } => walk_expr(operand, checker, type_map),
        ExprKind::Pipe { left, right } => {
            walk_expr(left, checker, type_map);
            walk_expr(right, checker, type_map);
        }
        ExprKind::MemberAccess { object, .. } => walk_expr(object, checker, type_map),
        ExprKind::Subscript { object, index } => {
            walk_expr(object, checker, type_map);
            walk_subscript(index, checker, type_map);
        }
        ExprKind::Call { callee, args } => {
            walk_expr(callee, checker, type_map);
            for arg in args { walk_expr(&arg.value, checker, type_map); }
        }
        ExprKind::Lambda { body, .. } => walk_expr(body, checker, type_map),
        ExprKind::BlockExpr(block) => walk_block(block, checker, type_map),
        ExprKind::ListComp { element, generators } => {
            walk_expr(element, checker, type_map);
            for gen in generators {
                walk_expr(&gen.iterable, checker, type_map);
                for cond in &gen.conditions { walk_expr(cond, checker, type_map); }
            }
        }
        ExprKind::IfExpr { condition, then_expr, else_expr } => {
            walk_expr(condition, checker, type_map);

            let before = checker.snapshot_all();
            walk_expr(then_expr, checker, type_map);
            let after_then = checker.snapshot_all();
            checker.check_branch_local_unconsumed(&before, &after_then);

            checker.restore_all(before.clone());
            walk_expr(else_expr, checker, type_map);
            let after_else = checker.snapshot_all();
            checker.check_branch_local_unconsumed(&before, &after_else);

            checker.restore_all(before.clone());

            checker.check_branch_symmetry(
                &before.states,
                &after_then.states,
                &after_else.states,
                expr.span,
            );
        }
        ExprKind::ListLiteral(elems) | ExprKind::TupleLiteral(elems) => {
            for e in elems { walk_expr(e, checker, type_map); }
        }
        ExprKind::DictLiteral(pairs) => {
            for (k, v) in pairs {
                walk_expr(k, checker, type_map);
                walk_expr(v, checker, type_map);
            }
        }
        ExprKind::FString(parts) => {
            for part in parts {
                if let nsl_ast::expr::FStringPart::Expr(e) = part {
                    walk_expr(e, checker, type_map);
                }
            }
        }
        ExprKind::Range { start, end, .. } => {
            if let Some(e) = start { walk_expr(e, checker, type_map); }
            if let Some(e) = end { walk_expr(e, checker, type_map); }
        }
        ExprKind::Paren(e) | ExprKind::Await(e) => walk_expr(e, checker, type_map),
        ExprKind::MatchExpr { subject, arms } => {
            walk_expr(subject, checker, type_map);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    walk_expr(guard, checker, type_map);
                }
                for stmt in &arm.body.stmts { walk_stmt(stmt, checker, type_map); }
            }
        }
        _ => {} // Literals, SelfRef, Error
    }
}

fn walk_subscript(
    index: &nsl_ast::expr::SubscriptKind,
    checker: &mut OwnershipChecker<'_>,
    type_map: &TypeMap,
) {
    match index {
        nsl_ast::expr::SubscriptKind::Index(e) => walk_expr(e, checker, type_map),
        nsl_ast::expr::SubscriptKind::Slice { lower, upper, step } => {
            if let Some(e) = lower { walk_expr(e, checker, type_map); }
            if let Some(e) = upper { walk_expr(e, checker, type_map); }
            if let Some(e) = step { walk_expr(e, checker, type_map); }
        }
        nsl_ast::expr::SubscriptKind::MultiDim(dims) => {
            for d in dims { walk_subscript(d, checker, type_map); }
        }
    }
}

fn register_pattern_bindings(
    pattern: &nsl_ast::pattern::Pattern,
    checker: &mut OwnershipChecker<'_>,
    type_map: &TypeMap,
    value: Option<&Expr>,
) {
    match &pattern.kind {
        PatternKind::Ident(sym) => {
            let is_tensor = value
                .and_then(|expr| type_map.get(&expr.id))
                .map(|ty| ty.is_tensor())
                .unwrap_or(false);
            if is_tensor {
                checker.register_binding(*sym, pattern.span, false);
            }
        }
        PatternKind::Tuple(pats) | PatternKind::List(pats) => {
            for p in pats {
                register_pattern_bindings(p, checker, type_map, value);
            }
        }
        PatternKind::Typed { pattern: inner, .. } => {
            register_pattern_bindings(inner, checker, type_map, value);
        }
        _ => {}
    }
}

/// Extract the element type from an iterable type.
fn iter_element_type(ty: Option<&crate::types::Type>) -> Option<crate::types::Type> {
    use crate::types::Type;
    match ty? {
        Type::List(elem) => Some(*elem.clone()),
        Type::Dict(key, _) => Some(*key.clone()),
        Type::Tuple(elems) => elems.first().cloned(),
        Type::Str => Some(Type::Str),
        _ => None,
    }
}

/// Register pattern bindings using a known element type instead of a value expression.
fn register_pattern_bindings_with_type(
    pattern: &nsl_ast::pattern::Pattern,
    checker: &mut OwnershipChecker<'_>,
    elem_ty: &Option<crate::types::Type>,
) {
    let is_tensor = elem_ty.as_ref().map(|ty| ty.is_tensor()).unwrap_or(false);
    match &pattern.kind {
        PatternKind::Ident(sym) => {
            if is_tensor {
                checker.register_binding(*sym, pattern.span, false);
            }
        }
        PatternKind::Tuple(pats) | PatternKind::List(pats) => {
            // For tuple destructuring in for-loops (e.g., `for (i, t) in enumerate(list):`),
            // each sub-pattern gets the same element type as a conservative approximation.
            for p in pats {
                register_pattern_bindings_with_type(p, checker, elem_ty);
            }
        }
        PatternKind::Typed { pattern: inner, .. } => {
            register_pattern_bindings_with_type(inner, checker, elem_ty);
        }
        _ => {}
    }
}

/// Check if a function parameter is tensor-typed from its type annotation.
fn is_tensor_param(param: &nsl_ast::decl::Param) -> bool {
    if let Some(ref ann) = param.type_ann {
        matches!(ann.kind,
            nsl_ast::types::TypeExprKind::Tensor { .. }
            | nsl_ast::types::TypeExprKind::Param { .. }
            | nsl_ast::types::TypeExprKind::Buffer { .. }
        )
    } else {
        false
    }
}
