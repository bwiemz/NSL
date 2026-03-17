use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;

use nsl_errors::Diagnostic;

/// Validate @moe decorator arguments.
/// Returns (num_experts, top_k, capacity_factor, aux_loss_coeff) or None on error.
pub fn validate_moe_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<(usize, usize, f64, f64)> {
    let mut num_experts: Option<usize> = None;
    let mut top_k: usize = 2;
    let mut capacity_factor: f64 = 1.25;
    let mut aux_loss_coeff: f64 = 0.01;

    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = resolve_sym(*name_sym);
                match aname.as_str() {
                    "num_experts" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            if *n <= 0 {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@moe: num_experts must be a positive integer".to_string(),
                                    )
                                    .with_label(arg.span, "must be > 0"),
                                );
                            } else {
                                num_experts = Some(*n as usize);
                            }
                        } else {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@moe: num_experts must be an integer literal".to_string(),
                                )
                                .with_label(arg.span, "expected integer"),
                            );
                        }
                    }
                    "top_k" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            if *n != 1 && *n != 2 {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@moe: top_k must be 1 or 2".to_string(),
                                    )
                                    .with_label(arg.span, "must be 1 or 2"),
                                );
                            } else {
                                top_k = *n as usize;
                            }
                        }
                    }
                    "capacity_factor" => {
                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                            if *f < 1.0 {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@moe: capacity_factor must be >= 1.0".to_string(),
                                    )
                                    .with_label(arg.span, "must be >= 1.0"),
                                );
                            } else {
                                capacity_factor = *f;
                            }
                        }
                    }
                    "aux_loss_coeff" => {
                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                            if *f < 0.0 {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@moe: aux_loss_coeff must be >= 0.0".to_string(),
                                    )
                                    .with_label(arg.span, "must be >= 0.0"),
                                );
                            } else {
                                aux_loss_coeff = *f;
                            }
                        }
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!("@moe: unknown argument '{}'", aname))
                                .with_label(arg.span, "unknown argument"),
                        );
                    }
                }
            }
        }
    }

    if num_experts.is_none() {
        diagnostics.push(
            Diagnostic::error("@moe: num_experts is required".to_string())
                .with_label(deco.span, "missing num_experts"),
        );
        return None;
    }

    Some((num_experts.unwrap(), top_k, capacity_factor, aux_loss_coeff))
}
