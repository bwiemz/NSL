//! M62 `@export` decorator semantic validation.
//!
//! Enforces the C-ABI-compatible subset of NSL function signatures.
//! Errors here block codegen — `declaration.rs`'s `@export` branch
//! assumes signatures are already validated.

use nsl_ast::decl::{Decorator, FnDef, ModelMember, Param};
use nsl_ast::expr::{Expr, ExprKind};
use nsl_ast::stmt::{Block, Stmt, StmtKind};
use nsl_ast::types::{TypeExpr, TypeExprKind};
use nsl_ast::Module;
use nsl_errors::Diagnostic;
use nsl_lexer::Interner;

/// Run `@export` validation over the top-level statements of a module.
///
/// Returns diagnostics that should be appended to the rest of the
/// analysis diagnostics.  Pure-additive: does not read or modify
/// other analysis state.
pub fn validate_exports(module: &Module, interner: &Interner) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    for stmt in &module.stmts {
        validate_stmt(stmt, interner, &mut diagnostics);
        validate_model_method_exports(stmt, interner, &mut diagnostics);
    }
    diagnostics
}

/// Validate `@export` decorators on methods inside `model` blocks.
///
/// Rule 1 — method has `self` as first param: accepted, no error.
/// Rule 2 — method lacks `self`: error with a pointed diagnostic.
fn validate_model_method_exports(
    stmt: &Stmt,
    interner: &Interner,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let StmtKind::ModelDef(md) = &stmt.kind else {
        return;
    };

    for member in &md.members {
        let ModelMember::Method(fn_def, decos) = member else {
            continue;
        };

        let has_export = decos.iter().any(|d| {
            d.name.len() == 1
                && interner.resolve(d.name[0].0).unwrap_or("") == "export"
        });
        if !has_export {
            continue;
        }

        let method_name = interner.resolve(fn_def.name.0).unwrap_or("<unknown>").to_string();
        let has_self = fn_def
            .params
            .first()
            .map(|p| interner.resolve(p.name.0).unwrap_or("") == "self")
            .unwrap_or(false);

        if !has_self {
            diagnostics.push(
                Diagnostic::error(format!(
                    "@export model method '{method_name}' requires `self` as first parameter; \
                     for a standalone function export use a top-level `@export fn`"
                )),
            );
        }
    }
}

fn validate_stmt(stmt: &Stmt, interner: &Interner, diagnostics: &mut Vec<Diagnostic>) {
    let StmtKind::Decorated { decorators, stmt: inner } = &stmt.kind else {
        return;
    };

    let export_occurrences: Vec<&Decorator> = decorators
        .iter()
        .filter(|d| {
            d.name.len() == 1
                && interner.resolve(d.name[0].0).unwrap_or("") == "export"
        })
        .collect();

    if export_occurrences.is_empty() {
        return;
    }

    // Duplicate @export
    if export_occurrences.len() > 1 {
        diagnostics.push(
            Diagnostic::error(format!(
                "@export decorator appears multiple times on '{}'",
                describe_stmt(inner, interner)
            ))
            .with_label(export_occurrences[1].span, "duplicate @export"),
        );
    }

    // Validate kwargs on the first occurrence
    validate_export_args(export_occurrences[0], interner, diagnostics);

    // Must decorate a FnDef
    let StmtKind::FnDef(fn_def) = &inner.kind else {
        diagnostics.push(
            Diagnostic::error("@export can only be applied to functions".to_string())
                .with_label(export_occurrences[0].span, "not a function"),
        );
        return;
    };

    // Top-level @export fn — never a model method.
    validate_fn_signature(fn_def, interner, export_occurrences[0], false, diagnostics);
}

fn validate_export_args(
    d: &Decorator,
    interner: &Interner,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let Some(ref args) = d.args else {
        return; // bare @export is fine
    };

    for arg in args {
        match arg.name {
            None => {
                diagnostics.push(
                    Diagnostic::error(
                        "@export takes only a 'name=...' keyword argument; no positional arguments"
                            .to_string(),
                    )
                    .with_label(arg.span, "positional argument not allowed"),
                );
            }
            Some(sym) => {
                let kw = interner.resolve(sym.0).unwrap_or("").to_string();
                if kw != "name" {
                    diagnostics.push(
                        Diagnostic::error(format!(
                            "@export takes only a 'name' keyword argument; got '{kw}'"
                        ))
                        .with_label(arg.span, "unknown keyword argument"),
                    );
                    continue;
                }
                match &arg.value.kind {
                    ExprKind::StringLiteral(s) => {
                        if s.is_empty() {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@export(name=\"...\") cannot be empty".to_string(),
                                )
                                .with_label(arg.value.span, "empty name"),
                            );
                        } else if !is_valid_c_identifier(s) {
                            diagnostics.push(
                                Diagnostic::error(format!(
                                    "@export(name=\"{s}\") must be a valid C identifier: \
                                     letters, digits, underscore; cannot start with digit"
                                ))
                                .with_label(arg.value.span, "invalid C identifier"),
                            );
                        }
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(
                                "@export(name=...) must be a string literal".to_string(),
                            )
                            .with_label(arg.value.span, "expected string literal"),
                        );
                    }
                }
            }
        }
    }
}

fn validate_fn_signature(
    fn_def: &FnDef,
    interner: &Interner,
    export_decorator: &Decorator,
    is_model_method_with_self: bool,
    diagnostics: &mut Vec<Diagnostic>,
) {
    // No generic type parameters — C ABI requires monomorphised types.
    if !fn_def.type_params.is_empty() {
        diagnostics.push(
            Diagnostic::error(
                "@export function cannot have generic type parameters — \
                 C ABI requires monomorphized types"
                    .to_string(),
            )
            .with_label(export_decorator.span, "generic @export"),
        );
    }

    for param in &fn_def.params {
        validate_param(param, interner, diagnostics);
    }

    if let Some(ref ret_ty) = fn_def.return_type {
        if !is_c_abi_compatible(ret_ty, interner) {
            diagnostics.push(
                Diagnostic::error(
                    "@export function must return a tensor, scalar, or tuple of those"
                        .to_string(),
                )
                .with_label(ret_ty.span, "non-ABI return type"),
            );
        }
    }

    // Warn if the body references model weights — these are silently absent
    // at C-ABI call time because @export functions use compile-time-baked weights.
    // Model methods with `self` are exempt: the weight-loading PR makes those work.
    check_no_model_weight_access(fn_def, interner, diagnostics, is_model_method_with_self);
}

fn validate_param(param: &Param, interner: &Interner, diagnostics: &mut Vec<Diagnostic>) {
    let Some(ref type_ann) = param.type_ann else {
        diagnostics.push(
            Diagnostic::error(format!(
                "@export function parameter '{}' must have an explicit type",
                interner.resolve(param.name.0).unwrap_or("<unknown>")
            ))
            .with_label(param.span, "missing type annotation"),
        );
        return;
    };

    if is_closure_type(type_ann) {
        diagnostics.push(
            Diagnostic::error(
                "@export function cannot take closure parameters — \
                 closures cannot cross the C ABI"
                    .to_string(),
            )
            .with_label(type_ann.span, "closure type"),
        );
    } else if !is_c_abi_compatible(type_ann, interner) {
        diagnostics.push(
            Diagnostic::error(format!(
                "@export function parameter '{}' must be a tensor, scalar, or tuple of those",
                interner.resolve(param.name.0).unwrap_or("<unknown>")
            ))
            .with_label(type_ann.span, "non-ABI parameter type"),
        );
    }
}

/// Warn when an `@export` function body accesses fields on model-typed parameters.
///
/// C-ABI wrappers do not load weights at runtime — they use compile-time-baked
/// weights instead.  A field access like `net.W` inside an `@export` function
/// silently does the wrong thing at the call site.  This is a syntactic heuristic
/// (§5 CAUTION): a parameter is considered model-typed if it has a `Named` type
/// annotation that does not resolve to a primitive scalar.  `self` (rendered as
/// `ExprKind::SelfRef`) is always treated as model-typed.
fn check_no_model_weight_access(
    fn_def: &FnDef,
    interner: &Interner,
    diagnostics: &mut Vec<Diagnostic>,
    is_model_method_with_self: bool,
) {
    // Model methods with `self` are weight-loading-aware — skip the warning.
    if is_model_method_with_self {
        return;
    }

    let fn_name = interner.resolve(fn_def.name.0).unwrap_or("<unknown>");

    // Build the set of "suspect" parameter names: self + Named-typed params.
    let mut suspect_syms: std::collections::HashSet<nsl_ast::Symbol> =
        std::collections::HashSet::new();
    let mut has_self_param = false;

    for param in &fn_def.params {
        let param_name = interner.resolve(param.name.0).unwrap_or("");
        if param_name == "self" {
            has_self_param = true;
            suspect_syms.insert(param.name);
        } else if let Some(ref ty) = param.type_ann {
            if is_model_typed(ty, interner) {
                suspect_syms.insert(param.name);
            }
        }
    }

    if suspect_syms.is_empty() && !has_self_param {
        return;
    }

    // Walk every expression in the function body looking for MemberAccess on a
    // suspect receiver.
    let found = find_weight_access_in_block(&fn_def.body, &suspect_syms, has_self_param);
    if let Some(span) = found {
        diagnostics.push(
            Diagnostic::warning(format!(
                "@export function '{}' accesses model weights via field access — \
                 C-ABI wrappers do not load weights at runtime; \
                 weight values are baked at compile time",
                fn_name
            ))
            .with_label(span, "model-weight reference"),
        );
    }
}

/// Returns `true` if `ty` is a user-defined named type (not a primitive scalar).
fn is_model_typed(ty: &TypeExpr, interner: &Interner) -> bool {
    if let TypeExprKind::Named(sym) = &ty.kind {
        let name = interner.resolve(sym.0).unwrap_or("");
        !matches!(
            name,
            "f32" | "f64" | "f16" | "bf16" | "fp32" | "fp64" | "fp16"
                | "i8" | "i16" | "i32" | "i64"
                | "u8" | "u16" | "u32" | "u64"
                | "int" | "long" | "bool"
                | "float" | "double"
                | "str" | "string"
        )
    } else {
        false
    }
}

/// Search a `Block` for the first `MemberAccess` whose object is a suspect
/// identifier.  Returns the span of the offending expression.
fn find_weight_access_in_block(
    block: &Block,
    suspects: &std::collections::HashSet<nsl_ast::Symbol>,
    has_self_param: bool,
) -> Option<nsl_ast::Span> {
    for stmt in &block.stmts {
        if let Some(span) = find_weight_access_in_stmt(stmt, suspects, has_self_param) {
            return Some(span);
        }
    }
    None
}

fn find_weight_access_in_stmt(
    stmt: &nsl_ast::stmt::Stmt,
    suspects: &std::collections::HashSet<nsl_ast::Symbol>,
    has_self_param: bool,
) -> Option<nsl_ast::Span> {
    match &stmt.kind {
        StmtKind::Return(Some(expr)) | StmtKind::Yield(Some(expr)) | StmtKind::Expr(expr) => {
            find_weight_access_in_expr(expr, suspects, has_self_param)
        }
        StmtKind::VarDecl { value: Some(expr), .. } => {
            find_weight_access_in_expr(expr, suspects, has_self_param)
        }
        StmtKind::Assign { target, value, .. } => {
            find_weight_access_in_expr(target, suspects, has_self_param)
                .or_else(|| find_weight_access_in_expr(value, suspects, has_self_param))
        }
        StmtKind::If { condition, then_block, elif_clauses, else_block } => {
            find_weight_access_in_expr(condition, suspects, has_self_param)
                .or_else(|| find_weight_access_in_block(then_block, suspects, has_self_param))
                .or_else(|| {
                    for (cond, blk) in elif_clauses {
                        if let Some(s) = find_weight_access_in_expr(cond, suspects, has_self_param)
                            .or_else(|| find_weight_access_in_block(blk, suspects, has_self_param))
                        {
                            return Some(s);
                        }
                    }
                    None
                })
                .or_else(|| {
                    else_block
                        .as_ref()
                        .and_then(|b| find_weight_access_in_block(b, suspects, has_self_param))
                })
        }
        StmtKind::For { iterable, body, .. } => {
            find_weight_access_in_expr(iterable, suspects, has_self_param)
                .or_else(|| find_weight_access_in_block(body, suspects, has_self_param))
        }
        StmtKind::While { condition, body } => {
            find_weight_access_in_expr(condition, suspects, has_self_param)
                .or_else(|| find_weight_access_in_block(body, suspects, has_self_param))
        }
        StmtKind::WhileLet { expr, body, .. } => {
            find_weight_access_in_expr(expr, suspects, has_self_param)
                .or_else(|| find_weight_access_in_block(body, suspects, has_self_param))
        }
        StmtKind::Decorated { stmt: inner, .. } => {
            find_weight_access_in_stmt(inner, suspects, has_self_param)
        }
        _ => None,
    }
}

fn find_weight_access_in_expr(
    expr: &Expr,
    suspects: &std::collections::HashSet<nsl_ast::Symbol>,
    has_self_param: bool,
) -> Option<nsl_ast::Span> {
    match &expr.kind {
        ExprKind::MemberAccess { object, .. } => {
            // Check if receiver is a suspect.
            let receiver_is_suspect = match &object.kind {
                ExprKind::SelfRef => has_self_param,
                ExprKind::Ident(sym) => suspects.contains(sym),
                _ => false,
            };
            if receiver_is_suspect {
                return Some(expr.span);
            }
            // Recurse into the object in case it's a chained access.
            find_weight_access_in_expr(object, suspects, has_self_param)
        }
        ExprKind::BinaryOp { left, right, .. } => {
            find_weight_access_in_expr(left, suspects, has_self_param)
                .or_else(|| find_weight_access_in_expr(right, suspects, has_self_param))
        }
        ExprKind::UnaryOp { operand, .. } | ExprKind::Paren(operand) | ExprKind::Await(operand) => {
            find_weight_access_in_expr(operand, suspects, has_self_param)
        }
        ExprKind::Pipe { left, right } => {
            find_weight_access_in_expr(left, suspects, has_self_param)
                .or_else(|| find_weight_access_in_expr(right, suspects, has_self_param))
        }
        ExprKind::Call { callee, args } => {
            find_weight_access_in_expr(callee, suspects, has_self_param).or_else(|| {
                for arg in args {
                    if let Some(s) =
                        find_weight_access_in_expr(&arg.value, suspects, has_self_param)
                    {
                        return Some(s);
                    }
                }
                None
            })
        }
        ExprKind::Subscript { object, index } => {
            find_weight_access_in_expr(object, suspects, has_self_param).or_else(|| {
                find_weight_access_in_subscript(index, suspects, has_self_param)
            })
        }
        ExprKind::ListLiteral(exprs) | ExprKind::TupleLiteral(exprs) => {
            for e in exprs {
                if let Some(s) = find_weight_access_in_expr(e, suspects, has_self_param) {
                    return Some(s);
                }
            }
            None
        }
        ExprKind::DictLiteral(pairs) => {
            for (k, v) in pairs {
                if let Some(s) = find_weight_access_in_expr(k, suspects, has_self_param)
                    .or_else(|| find_weight_access_in_expr(v, suspects, has_self_param))
                {
                    return Some(s);
                }
            }
            None
        }
        ExprKind::IfExpr { condition, then_expr, else_expr } => {
            find_weight_access_in_expr(condition, suspects, has_self_param)
                .or_else(|| find_weight_access_in_expr(then_expr, suspects, has_self_param))
                .or_else(|| find_weight_access_in_expr(else_expr, suspects, has_self_param))
        }
        ExprKind::BlockExpr(block) => {
            find_weight_access_in_block(block, suspects, has_self_param)
        }
        ExprKind::FString(parts) => {
            for part in parts {
                if let nsl_ast::expr::FStringPart::Expr(e) = part {
                    if let Some(s) = find_weight_access_in_expr(e, suspects, has_self_param) {
                        return Some(s);
                    }
                }
            }
            None
        }
        ExprKind::Lambda { body, .. } => {
            find_weight_access_in_expr(body, suspects, has_self_param)
        }
        ExprKind::ListComp { element, generators } => {
            find_weight_access_in_expr(element, suspects, has_self_param).or_else(|| {
                for gen in generators {
                    if let Some(s) =
                        find_weight_access_in_expr(&gen.iterable, suspects, has_self_param)
                    {
                        return Some(s);
                    }
                    for cond in &gen.conditions {
                        if let Some(s) =
                            find_weight_access_in_expr(cond, suspects, has_self_param)
                        {
                            return Some(s);
                        }
                    }
                }
                None
            })
        }
        ExprKind::MatchExpr { subject, arms } => {
            find_weight_access_in_expr(subject, suspects, has_self_param).or_else(|| {
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        if let Some(s) =
                            find_weight_access_in_expr(guard, suspects, has_self_param)
                        {
                            return Some(s);
                        }
                    }
                    if let Some(s) =
                        find_weight_access_in_block(&arm.body, suspects, has_self_param)
                    {
                        return Some(s);
                    }
                }
                None
            })
        }
        ExprKind::Range { start, end, .. } => {
            start
                .as_ref()
                .and_then(|e| find_weight_access_in_expr(e, suspects, has_self_param))
                .or_else(|| {
                    end.as_ref()
                        .and_then(|e| find_weight_access_in_expr(e, suspects, has_self_param))
                })
        }
        // Leaf nodes — no sub-expressions.
        ExprKind::Ident(_)
        | ExprKind::SelfRef
        | ExprKind::IntLiteral(_)
        | ExprKind::FloatLiteral(_)
        | ExprKind::StringLiteral(_)
        | ExprKind::BoolLiteral(_)
        | ExprKind::NoneLiteral
        | ExprKind::Error => None,
    }
}

fn find_weight_access_in_subscript(
    index: &nsl_ast::expr::SubscriptKind,
    suspects: &std::collections::HashSet<nsl_ast::Symbol>,
    has_self_param: bool,
) -> Option<nsl_ast::Span> {
    match index {
        nsl_ast::expr::SubscriptKind::Index(e) => {
            find_weight_access_in_expr(e, suspects, has_self_param)
        }
        nsl_ast::expr::SubscriptKind::Slice { lower, upper, step } => lower
            .as_ref()
            .and_then(|e| find_weight_access_in_expr(e, suspects, has_self_param))
            .or_else(|| {
                upper
                    .as_ref()
                    .and_then(|e| find_weight_access_in_expr(e, suspects, has_self_param))
            })
            .or_else(|| {
                step.as_ref()
                    .and_then(|e| find_weight_access_in_expr(e, suspects, has_self_param))
            }),
        nsl_ast::expr::SubscriptKind::MultiDim(dims) => {
            for d in dims {
                if let Some(s) = find_weight_access_in_subscript(d, suspects, has_self_param) {
                    return Some(s);
                }
            }
            None
        }
    }
}

fn is_closure_type(ty: &TypeExpr) -> bool {
    matches!(ty.kind, TypeExprKind::Function { .. })
}

fn is_c_abi_compatible(ty: &TypeExpr, interner: &Interner) -> bool {
    match &ty.kind {
        TypeExprKind::Tensor { .. }
        | TypeExprKind::Param { .. }
        | TypeExprKind::Buffer { .. } => true,
        TypeExprKind::Named(sym) => {
            let name = interner.resolve(sym.0).unwrap_or("");
            matches!(
                name,
                "f32" | "f64" | "f16" | "bf16" | "fp32" | "fp64" | "fp16"
                    | "i8" | "i16" | "i32" | "i64"
                    | "u8" | "u16" | "u32" | "u64"
                    | "int" | "long" | "bool"
                    | "float" | "double"
            )
        }
        TypeExprKind::Tuple(elems) => elems.iter().all(|e| is_c_abi_compatible(e, interner)),
        _ => false,
    }
}

fn is_valid_c_identifier(s: &str) -> bool {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) if c.is_ascii_alphabetic() || c == '_' => {}
        _ => return false,
    }
    chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

fn describe_stmt(stmt: &Stmt, interner: &Interner) -> String {
    match &stmt.kind {
        StmtKind::FnDef(fd) => interner
            .resolve(fd.name.0)
            .unwrap_or("<unknown>")
            .to_string(),
        _ => "<non-fn>".to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_errors::FileId;

    fn parse_and_validate(src: &str) -> Vec<Diagnostic> {
        let mut interner: Interner = Interner::new();
        let (tokens, lex_diags) = nsl_lexer::tokenize(src, FileId(0), &mut interner);
        assert!(
            lex_diags.is_empty(),
            "lex errors in fixture: {:?}",
            lex_diags
        );
        let parse_result = nsl_parser::parse(&tokens, &mut interner);
        assert!(
            parse_result.diagnostics.is_empty(),
            "parse errors in fixture: {:?}",
            parse_result.diagnostics
        );
        validate_exports(&parse_result.module, &interner)
    }

    #[test]
    fn valid_simple_export_has_no_errors() {
        let src = "\
@export
fn forward(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
";
        let errs = parse_and_validate(src);
        assert!(errs.is_empty(), "expected no errors, got: {:?}", errs);
    }

    #[test]
    fn export_on_model_errors() {
        let src = "\
@export
model Foo:
    w: Tensor<[4], f32>
";
        let errs = parse_and_validate(src);
        assert!(
            errs.iter()
                .any(|d| d.message.contains("only be applied to functions")),
            "expected 'only be applied to functions' error, got: {:?}",
            errs
        );
    }

    #[test]
    fn export_empty_name_errors() {
        let src = "\
@export(name=\"\")
fn foo(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
";
        let errs = parse_and_validate(src);
        assert!(
            errs.iter().any(|d| d.message.contains("cannot be empty")),
            "expected 'cannot be empty' error, got: {:?}",
            errs
        );
    }

    #[test]
    fn export_invalid_c_identifier_errors() {
        let src = "\
@export(name=\"123bad\")
fn foo(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
";
        let errs = parse_and_validate(src);
        assert!(
            errs.iter().any(|d| d.message.contains("valid C identifier")),
            "expected 'valid C identifier' error, got: {:?}",
            errs
        );
    }

    #[test]
    fn export_unknown_kwarg_errors() {
        let src = "\
@export(other=1)
fn foo(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
";
        let errs = parse_and_validate(src);
        assert!(
            errs.iter().any(|d| d.message.contains("'other'")),
            "expected 'other' error, got: {:?}",
            errs
        );
    }

    #[test]
    fn export_positional_arg_errors() {
        let src = "\
@export(\"positional\")
fn foo(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
";
        let errs = parse_and_validate(src);
        assert!(
            errs.iter().any(|d| d.message.contains("positional")),
            "expected 'positional' error, got: {:?}",
            errs
        );
    }

    #[test]
    fn duplicate_export_errors() {
        let src = "\
@export
@export
fn foo(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return x
";
        let errs = parse_and_validate(src);
        assert!(
            errs.iter().any(|d| d.message.contains("multiple times")),
            "expected 'multiple times' error, got: {:?}",
            errs
        );
    }

    #[test]
    fn export_function_referencing_model_field_produces_warning() {
        // An @export function that accesses a field on a Named-typed parameter
        // (model weight reference) should produce a warning.
        let src = "\
@export
fn predict(net: Net, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return net.W @ x
";
        let diags = parse_and_validate(src);
        let warnings: Vec<_> = diags
            .iter()
            .filter(|d| d.level == nsl_errors::Level::Warning)
            .collect();
        assert!(
            warnings.iter().any(|d| d.message.contains("weight") && d.message.contains("predict")),
            "expected weight-reference warning for 'predict', got: {:?}",
            diags
        );
    }

    #[test]
    fn export_pure_function_has_no_weight_warning() {
        // An @export function with only tensor/scalar params and no field access
        // should not produce any weight warning.
        let src = "\
@export
fn add(a: Tensor<[4], f32>, b: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return a + b
";
        let diags = parse_and_validate(src);
        let weight_warnings: Vec<_> = diags
            .iter()
            .filter(|d| {
                d.level == nsl_errors::Level::Warning && d.message.contains("weight")
            })
            .collect();
        assert!(
            weight_warnings.is_empty(),
            "unexpected weight warning: {:?}",
            diags
        );
    }

    #[test]
    fn export_model_method_without_self_produces_error() {
        let src = r#"
model Net:
    W: Tensor<[4, 4], f32>

    @export
    fn helper(x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return x
"#;
        let diags = parse_and_validate(src);
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.level == nsl_errors::Level::Error)
            .collect();
        assert!(
            errors
                .iter()
                .any(|d| d.message.contains("requires `self`") && d.message.contains("helper")),
            "expected self-required error, got: {:?}",
            diags
        );
    }

    #[test]
    fn export_model_method_with_self_accesses_weights_no_warning() {
        let src = r#"
model Net:
    W: Tensor<[4, 4], f32>

    @export
    fn predict(self, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
        return self.W @ x
"#;
        let diags = parse_and_validate(src);
        let weight_warnings: Vec<_> = diags
            .iter()
            .filter(|d| d.level == nsl_errors::Level::Warning && d.message.contains("weight"))
            .collect();
        assert!(
            weight_warnings.is_empty(),
            "@export model method with self should not warn; got: {:?}",
            diags
        );
    }

    #[test]
    fn export_top_level_with_model_field_still_warns() {
        let src = r#"
model Net:
    W: Tensor<[4, 4], f32>

@export
fn predict_toplevel(net: Net, x: Tensor<[4], f32>) -> Tensor<[4], f32>:
    return net.W @ x
"#;
        let diags = parse_and_validate(src);
        let warnings: Vec<_> = diags
            .iter()
            .filter(|d| d.level == nsl_errors::Level::Warning)
            .collect();
        assert!(
            warnings.iter().any(|d| d.message.contains("weight")),
            "top-level @export with model field should still warn; got: {:?}",
            diags
        );
    }
}
