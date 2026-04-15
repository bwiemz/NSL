//! Semantic validation for the `@inspect(...)` decorator.
//!
//! Accepted arguments (Phase 5 spec §4.1):
//!   * First positional argument: the tensor handle to inspect.
//!   * `every = <positive int literal>` — sample cadence.
//!   * `condition = "<string literal>"` — predicate source string.
//!
//! At least one of `every` / `condition` MUST be specified. Both may be
//! specified together. Any other keyword is an error. Any extra positional
//! argument is an error.
//!
//! This pass does NOT verify the first-arg Tensor type — the semantic
//! analyzer does not populate a stable `type_of(expr)` at the decorator
//! check site, so type enforcement is deferred to codegen which errors on
//! non-tensor handles. Literal/positional/keyword shape is enforced here.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

pub fn validate_inspect_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let args = match &deco.args {
        Some(a) => a,
        None => {
            diagnostics.push(
                Diagnostic::error(
                    "@inspect requires arguments: first positional is the tensor"
                        .to_string(),
                )
                .with_label(deco.span, "missing arguments"),
            );
            return;
        }
    };

    if args.is_empty() {
        diagnostics.push(
            Diagnostic::error(
                "@inspect requires at least one argument (the tensor)".to_string(),
            )
            .with_label(deco.span, "empty argument list"),
        );
        return;
    }

    // First argument must be positional (the tensor).
    if args[0].name.is_some() {
        diagnostics.push(
            Diagnostic::error(
                "@inspect: first argument must be positional (the tensor)".to_string(),
            )
            .with_label(args[0].span, "expected positional tensor"),
        );
        return;
    }

    let mut has_every = false;
    let mut has_cond = false;

    for arg in &args[1..] {
        let Some(name_sym) = arg.name else {
            diagnostics.push(
                Diagnostic::error(
                    "@inspect: only the tensor is positional; other arguments must be `every=` / `condition=`"
                        .to_string(),
                )
                .with_label(arg.span, "unexpected positional argument"),
            );
            continue;
        };
        let aname = resolve_sym(name_sym);
        match aname.as_str() {
            "every" => {
                has_every = true;
                let ok = matches!(arg.value.kind, ExprKind::IntLiteral(n) if n > 0);
                if !ok {
                    diagnostics.push(
                        Diagnostic::error(
                            "@inspect every= must be a positive integer literal".to_string(),
                        )
                        .with_label(arg.value.span, "not a positive int literal"),
                    );
                }
            }
            "condition" => {
                has_cond = true;
                if !matches!(arg.value.kind, ExprKind::StringLiteral(_)) {
                    diagnostics.push(
                        Diagnostic::error(
                            "@inspect condition= must be a string literal".to_string(),
                        )
                        .with_label(arg.value.span, "not a string literal"),
                    );
                }
            }
            other => {
                diagnostics.push(
                    Diagnostic::error(format!("@inspect: unknown keyword {}", other))
                        .with_label(arg.span, "unknown keyword"),
                );
            }
        }
    }

    if !has_every && !has_cond {
        diagnostics.push(
            Diagnostic::error(
                "@inspect: must specify every=N or condition=\"...\" (or both)".to_string(),
            )
            .with_label(deco.span, "missing every=/condition="),
        );
    }
}
