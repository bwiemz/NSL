use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

use nsl_errors::Diagnostic;

/// Validate @context_parallel decorator arguments.
/// Returns ring_size or None on error.
pub fn validate_context_parallel_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<usize> {
    let mut ring_size: Option<usize> = None;

    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = resolve_sym(*name_sym);
                match aname.as_str() {
                    "ring_size" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            if *n < 2 {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@context_parallel: ring_size must be >= 2".to_string(),
                                    )
                                    .with_label(arg.span, "must be >= 2"),
                                );
                            } else {
                                ring_size = Some(*n as usize);
                            }
                        } else {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@context_parallel: ring_size must be an integer literal"
                                        .to_string(),
                                )
                                .with_label(arg.span, "expected integer"),
                            );
                        }
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!(
                                "@context_parallel: unknown argument '{}'",
                                aname
                            ))
                            .with_label(arg.span, "unknown argument"),
                        );
                    }
                }
            }
        }
    }

    if ring_size.is_none() {
        diagnostics.push(
            Diagnostic::error("@context_parallel: ring_size is required".to_string())
                .with_label(deco.span, "missing ring_size"),
        );
    }

    ring_size
}
