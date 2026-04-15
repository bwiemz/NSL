//! M62 `@export` decorator semantic validation.
//!
//! Enforces the C-ABI-compatible subset of NSL function signatures.
//! Errors here block codegen — `declaration.rs`'s `@export` branch
//! assumes signatures are already validated.

use nsl_ast::decl::{Decorator, FnDef, Param};
use nsl_ast::expr::ExprKind;
use nsl_ast::stmt::{Stmt, StmtKind};
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
    }
    diagnostics
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

    validate_fn_signature(fn_def, interner, export_occurrences[0], diagnostics);
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
}
