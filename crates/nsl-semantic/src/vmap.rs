use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Validate @vmap decorator arguments.
/// Returns batch_dim value (defaults to 0 per spec).
pub fn validate_vmap_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<usize> {
    let mut batch_dim: Option<usize> = None;

    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = resolve_sym(*name_sym);
                match aname.as_str() {
                    "batch_dim" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            if *n < 0 {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@vmap: batch_dim must be non-negative".to_string(),
                                    )
                                    .with_label(arg.span, "negative batch_dim"),
                                );
                            } else {
                                batch_dim = Some(*n as usize);
                            }
                        } else {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@vmap: batch_dim must be an integer literal".to_string(),
                                )
                                .with_label(arg.span, "expected integer"),
                            );
                        }
                    }
                    "batch_size" => {
                        // Optional — symbolic or concrete batch size. Accept for now.
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!("@vmap: unknown argument '{}'", aname))
                                .with_label(arg.span, "unknown argument"),
                        );
                    }
                }
            }
        }
    }

    // Default batch_dim to 0 per spec (Section 1: "Default: 0")
    Some(batch_dim.unwrap_or(0))
}
