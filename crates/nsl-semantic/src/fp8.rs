use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

use nsl_errors::Diagnostic;

/// Why `@fp8_compute(calibrate = true)` is refused: codegen reads only the
/// decorator's presence, so the flag was accepted and dropped, and
/// `nsl_fp8_update_calibration` has no caller.
pub const FP8_CALIBRATE_REFUSAL: &str = "the flag was never read: FP8 scales are computed per call \
     from the tensor's own absmax, and no calibration pass exists. Remove the argument";

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
                            if *b {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@fp8_compute(calibrate = true) is not implemented".to_string(),
                                    )
                                    .with_label(arg.span, FP8_CALIBRATE_REFUSAL),
                                );
                            }
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
