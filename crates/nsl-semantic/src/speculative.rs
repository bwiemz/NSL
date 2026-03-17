use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Validate @speculative decorator arguments.
/// Returns (draft_model_name, num_tokens, temperature, tree_width) or None.
pub fn validate_speculative_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<(String, usize, f32, usize)> {
    let mut draft_model: Option<String> = None;
    let mut num_tokens: usize = 5;
    let mut temperature: f32 = 0.0;
    let mut tree_width: usize = 1;
    let mut medusa = false;

    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = resolve_sym(*name_sym);
                match aname.as_str() {
                    "draft_model" => {
                        if let ExprKind::StringLiteral(ref s) = arg.value.kind {
                            draft_model = Some(s.clone());
                        } else {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@speculative: draft_model must be a string".to_string(),
                                )
                                .with_label(arg.span, "expected string"),
                            );
                        }
                    }
                    "num_tokens" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            if *n < 1 || *n > 10 {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@speculative: num_tokens must be 1-10".to_string(),
                                    )
                                    .with_label(arg.span, "must be 1-10"),
                                );
                            } else {
                                num_tokens = *n as usize;
                            }
                        }
                    }
                    "temperature" => {
                        if let ExprKind::FloatLiteral(f) = &arg.value.kind {
                            if *f < 0.0 {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@speculative: temperature must be >= 0.0".to_string(),
                                    )
                                    .with_label(arg.span, "must be >= 0.0"),
                                );
                            } else {
                                temperature = *f as f32;
                            }
                        }
                    }
                    "tree_width" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            if *n < 1 {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@speculative: tree_width must be >= 1".to_string(),
                                    )
                                    .with_label(arg.span, "must be >= 1"),
                                );
                            } else {
                                tree_width = *n as usize;
                            }
                        }
                    }
                    "medusa" => {
                        medusa = true;
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!(
                                "@speculative: unknown argument '{}'",
                                aname
                            ))
                            .with_label(arg.span, "unknown argument"),
                        );
                    }
                }
            }
        }
    }

    if draft_model.is_none() && !medusa {
        diagnostics.push(
            Diagnostic::error(
                "@speculative: requires either draft_model or medusa=true".to_string(),
            )
            .with_label(deco.span, "missing draft_model"),
        );
        return None;
    }
    if draft_model.is_some() && medusa {
        diagnostics.push(
            Diagnostic::error(
                "@speculative: draft_model and medusa=true are mutually exclusive".to_string(),
            )
            .with_label(deco.span, "pick one"),
        );
        return None;
    }

    Some((
        draft_model.unwrap_or_default(),
        num_tokens,
        temperature,
        tree_width,
    ))
}

/// Validate @medusa decorator arguments.
/// Returns (num_heads, tree_width) or None on error.
pub fn validate_medusa_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<(usize, usize)> {
    let mut num_heads: Option<usize> = None;
    let mut tree_width: usize = 1;

    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = resolve_sym(*name_sym);
                match aname.as_str() {
                    "num_heads" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            if *n < 1 {
                                diagnostics.push(
                                    Diagnostic::error("@medusa: num_heads must be >= 1".to_string())
                                        .with_label(arg.span, "must be >= 1"),
                                );
                            } else {
                                num_heads = Some(*n as usize);
                            }
                        }
                    }
                    "tree_width" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            if *n < 1 {
                                diagnostics.push(
                                    Diagnostic::error("@medusa: tree_width must be >= 1".to_string())
                                        .with_label(arg.span, "must be >= 1"),
                                );
                            } else {
                                tree_width = *n as usize;
                            }
                        }
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!("@medusa: unknown argument '{}'", aname))
                                .with_label(arg.span, "unknown argument"),
                        );
                    }
                }
            }
        }
    }

    if num_heads.is_none() {
        diagnostics.push(
            Diagnostic::error("@medusa: num_heads is required".to_string())
                .with_label(deco.span, "missing num_heads"),
        );
        return None;
    }

    Some((num_heads.unwrap(), tree_width))
}
