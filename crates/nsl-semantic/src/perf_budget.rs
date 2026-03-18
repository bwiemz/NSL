use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Validate @perf_budget decorator arguments.
/// Returns max_tflops value or None on error.
pub fn validate_perf_budget_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<f64> {
    let mut max_tflops: Option<f64> = None;

    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = resolve_sym(*name_sym);
                match aname.as_str() {
                    "max_tflops" => match &arg.value.kind {
                        ExprKind::FloatLiteral(f) => {
                            max_tflops = Some(*f);
                        }
                        ExprKind::IntLiteral(n) => {
                            max_tflops = Some(*n as f64);
                        }
                        _ => {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@perf_budget: max_tflops must be a numeric literal"
                                        .to_string(),
                                )
                                .with_label(arg.span, "expected number"),
                            );
                        }
                    },
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!(
                                "@perf_budget: unknown argument '{}'",
                                aname
                            ))
                            .with_label(arg.span, "unknown argument"),
                        );
                    }
                }
            }
        }
    }

    if max_tflops.is_none() {
        diagnostics.push(
            Diagnostic::error("@perf_budget: max_tflops is required".to_string())
                .with_label(deco.span, "missing max_tflops"),
        );
    }

    max_tflops
}
