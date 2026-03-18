use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

use nsl_errors::Diagnostic;

/// Validate @fp8_compute decorator arguments.
/// Returns calibrate flag (default false).
pub fn validate_fp8_compute_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> bool {
    let mut calibrate = false;

    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = resolve_sym(*name_sym);
                match aname.as_str() {
                    "calibrate" => {
                        if let ExprKind::BoolLiteral(b) = &arg.value.kind {
                            calibrate = *b;
                        } else {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@fp8_compute: calibrate must be a boolean".to_string(),
                                )
                                .with_label(arg.span, "expected true or false"),
                            );
                        }
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!(
                                "@fp8_compute: unknown argument '{}'",
                                aname
                            ))
                            .with_label(arg.span, "unknown argument"),
                        );
                    }
                }
            }
        }
    }

    calibrate
}
