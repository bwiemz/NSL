//! M44: @grammar decorator validation.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

use nsl_errors::Diagnostic;

/// Validated grammar configuration from a @grammar decorator.
#[derive(Debug, Clone)]
pub struct GrammarConfig {
    pub start_rule: String,
}

/// Validate a @grammar decorator.
///
/// @grammar("start_rule_name") on a function means the function's docstring
/// contains a BNF grammar definition.
pub fn validate_grammar_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<GrammarConfig> {
    let mut start_rule = "start".to_string();

    if let Some(ref args) = deco.args {
        for arg in args {
            // Positional arg: the start rule name
            if arg.name.is_none() {
                if let ExprKind::StringLiteral(s) = &arg.value.kind {
                    start_rule = s.clone();
                }
            } else if let Some(ref name_sym) = arg.name {
                let key = resolve_sym(*name_sym);
                if key == "start" {
                    if let ExprKind::StringLiteral(s) = &arg.value.kind {
                        start_rule = s.clone();
                    }
                } else {
                    diagnostics.push(
                        Diagnostic::error(format!("unknown @grammar parameter '{key}'"))
                            .with_label(arg.value.span, "here"),
                    );
                    return None;
                }
            }
        }
    }

    if start_rule.is_empty() {
        diagnostics.push(
            Diagnostic::error("@grammar requires a non-empty start rule name")
                .with_label(deco.span, "here"),
        );
        return None;
    }

    Some(GrammarConfig { start_rule })
}
