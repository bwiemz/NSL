//! M47: @target(backend) decorator validation.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Validate `@target(backend)` decorator arguments.
///
/// Returns the list of valid target names found. Emits diagnostics for
/// unknown targets or missing arguments.
pub fn validate_target_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Vec<String> {
    let mut targets = Vec::new();
    if let Some(ref args) = deco.args {
        for arg in args {
            // Positional args: target names
            if arg.name.is_none() {
                if let ExprKind::Ident(sym) = &arg.value.kind {
                    let name = resolve_sym(*sym);
                    let valid = ["cuda", "rocm", "metal", "webgpu"];
                    if !valid.contains(&name.as_str()) {
                        diagnostics.push(
                            Diagnostic::error(format!(
                                "unknown target '{name}', expected: cuda, rocm, metal, webgpu"
                            ))
                            .with_label(arg.value.span, "here"),
                        );
                    } else {
                        targets.push(name);
                    }
                }
            }
        }
    }
    if targets.is_empty() {
        diagnostics.push(
            Diagnostic::error("@target requires at least one backend name")
                .with_label(deco.span, "here"),
        );
    }
    targets
}
