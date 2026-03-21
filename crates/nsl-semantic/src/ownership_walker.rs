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
                walk_expr(expr, checker);
            }
            register_pattern_bindings(pattern, checker, type_map, stmt);
        }
        StmtKind::Assign { target, value, .. } => {
            walk_expr(target, checker);
            walk_expr(value, checker);
        }
        StmtKind::Expr(expr) => walk_expr(expr, checker),
        StmtKind::Return(Some(expr)) | StmtKind::Yield(Some(expr)) => {
            walk_expr(expr, checker);
        }
        StmtKind::Return(None) | StmtKind::Yield(None) => {}
        StmtKind::If { condition, then_block, elif_clauses, else_block } => {
            walk_expr(condition, checker);

            let before = checker.snapshot();
            walk_block(then_block, checker, type_map);
            let after_then = checker.snapshot();

            checker.restore(before.clone());
            for (cond, block) in elif_clauses {
                walk_expr(cond, checker);
                walk_block(block, checker, type_map);
            }
            if let Some(block) = else_block {
                walk_block(block, checker, type_map);
            }
            let after_else = checker.snapshot();

            checker.check_branch_symmetry(&before, &after_then, &after_else, condition.span);
        }
        StmtKind::While { condition, body, .. } => {
            walk_expr(condition, checker);
            checker.enter_loop();
            walk_block(body, checker, type_map);
            checker.exit_loop();
        }
        StmtKind::WhileLet { expr, body, .. } => {
            walk_expr(expr, checker);
            checker.enter_loop();
            walk_block(body, checker, type_map);
            checker.exit_loop();
        }
        StmtKind::For { iterable, body, .. } => {
            walk_expr(iterable, checker);
            checker.enter_loop();
            walk_block(body, checker, type_map);
            checker.exit_loop();
        }
        StmtKind::Match { subject, arms } => {
            walk_expr(subject, checker);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    walk_expr(guard, checker);
                }
                walk_block(&arm.body, checker, type_map);
            }
        }
        StmtKind::Break | StmtKind::Continue => {}
        StmtKind::Decorated { stmt, .. } => walk_stmt(stmt, checker, type_map),
        _ => {}
    }
}

fn walk_expr(expr: &Expr, checker: &mut OwnershipChecker<'_>) {
    match &expr.kind {
        ExprKind::Ident(sym) => {
            checker.use_binding(*sym, expr.span, "expression");
        }
        ExprKind::BinaryOp { left, right, .. } => {
            walk_expr(left, checker);
            walk_expr(right, checker);
        }
        ExprKind::UnaryOp { operand, .. } => walk_expr(operand, checker),
        ExprKind::Pipe { left, right } => {
            walk_expr(left, checker);
            walk_expr(right, checker);
        }
        ExprKind::MemberAccess { object, .. } => walk_expr(object, checker),
        ExprKind::Subscript { object, index } => {
            walk_expr(object, checker);
            walk_subscript(index, checker);
        }
        ExprKind::Call { callee, args } => {
            walk_expr(callee, checker);
            for arg in args { walk_expr(&arg.value, checker); }
        }
        ExprKind::Lambda { body, .. } => walk_expr(body, checker),
        ExprKind::ListComp { element, generators } => {
            walk_expr(element, checker);
            for gen in generators {
                walk_expr(&gen.iterable, checker);
                for cond in &gen.conditions { walk_expr(cond, checker); }
            }
        }
        ExprKind::IfExpr { condition, then_expr, else_expr } => {
            walk_expr(condition, checker);
            walk_expr(then_expr, checker);
            walk_expr(else_expr, checker);
        }
        ExprKind::ListLiteral(elems) | ExprKind::TupleLiteral(elems) => {
            for e in elems { walk_expr(e, checker); }
        }
        ExprKind::DictLiteral(pairs) => {
            for (k, v) in pairs { walk_expr(k, checker); walk_expr(v, checker); }
        }
        ExprKind::FString(parts) => {
            for part in parts {
                if let nsl_ast::expr::FStringPart::Expr(e) = part {
                    walk_expr(e, checker);
                }
            }
        }
        ExprKind::Range { start, end, .. } => {
            if let Some(e) = start { walk_expr(e, checker); }
            if let Some(e) = end { walk_expr(e, checker); }
        }
        ExprKind::Paren(e) | ExprKind::Await(e) => walk_expr(e, checker),
        ExprKind::MatchExpr { subject, arms } => {
            walk_expr(subject, checker);
            for arm in arms {
                if let Some(guard) = &arm.guard {
                    walk_expr(guard, checker);
                }
                for stmt in &arm.body.stmts { walk_stmt(stmt, checker, &TypeMap::default()); }
            }
        }
        _ => {} // Literals, SelfRef, Error
    }
}

fn walk_subscript(index: &nsl_ast::expr::SubscriptKind, checker: &mut OwnershipChecker<'_>) {
    match index {
        nsl_ast::expr::SubscriptKind::Index(e) => walk_expr(e, checker),
        nsl_ast::expr::SubscriptKind::Slice { lower, upper, step } => {
            if let Some(e) = lower { walk_expr(e, checker); }
            if let Some(e) = upper { walk_expr(e, checker); }
            if let Some(e) = step { walk_expr(e, checker); }
        }
        nsl_ast::expr::SubscriptKind::MultiDim(dims) => {
            for d in dims { walk_subscript(d, checker); }
        }
    }
}

fn register_pattern_bindings(
    pattern: &nsl_ast::pattern::Pattern,
    checker: &mut OwnershipChecker<'_>,
    type_map: &TypeMap,
    stmt: &Stmt,
) {
    match &pattern.kind {
        PatternKind::Ident(sym) => {
            let is_tensor = type_map.get(&stmt.id)
                .map(|ty| ty.is_tensor())
                .unwrap_or(false);
            if is_tensor {
                checker.register_binding(*sym, pattern.span, false);
            }
        }
        PatternKind::Tuple(pats) | PatternKind::List(pats) => {
            for p in pats { register_pattern_bindings(p, checker, type_map, stmt); }
        }
        PatternKind::Typed { pattern: inner, .. } => {
            register_pattern_bindings(inner, checker, type_map, stmt);
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
