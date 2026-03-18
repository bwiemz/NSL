//! M43: Semantic validation for @pipeline decorator.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Configuration extracted from a validated `@pipeline` decorator.
pub struct PipelineDecoratorConfig {
    pub num_stages: usize,
    pub schedule: String,
    pub checkpoint_stages: bool,
}

/// Validate `@pipeline(stages=N, schedule="1f1b", checkpoint_stages=true)` decorator.
///
/// - `stages` is required and must be >= 2.
/// - `schedule` defaults to `"1f1b"`, valid values: `"1f1b"`, `"gpipe"`.
/// - `checkpoint_stages` defaults to `true`.
pub fn validate_pipeline_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<PipelineDecoratorConfig> {
    let mut num_stages: Option<usize> = None;
    let mut schedule = "1f1b".to_string();
    let mut checkpoint_stages = true;

    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = resolve_sym(*name_sym);
                match aname.as_str() {
                    "stages" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            if *n < 2 {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@pipeline: stages must be >= 2".to_string(),
                                    )
                                    .with_label(arg.span, "must be >= 2"),
                                );
                            } else {
                                num_stages = Some(*n as usize);
                            }
                        } else {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@pipeline: stages must be an integer literal".to_string(),
                                )
                                .with_label(arg.span, "expected integer"),
                            );
                        }
                    }
                    "schedule" => {
                        if let ExprKind::StringLiteral(ref s) = &arg.value.kind {
                            match s.as_str() {
                                "1f1b" | "gpipe" => {
                                    schedule = s.clone();
                                }
                                _ => {
                                    diagnostics.push(
                                        Diagnostic::error(format!(
                                            "@pipeline: unknown schedule '{}', expected '1f1b' or 'gpipe'",
                                            s
                                        ))
                                        .with_label(arg.span, "unknown schedule"),
                                    );
                                }
                            }
                        } else {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@pipeline: schedule must be a string literal".to_string(),
                                )
                                .with_label(arg.span, "expected string"),
                            );
                        }
                    }
                    "checkpoint_stages" => {
                        if let ExprKind::BoolLiteral(b) = &arg.value.kind {
                            checkpoint_stages = *b;
                        } else {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@pipeline: checkpoint_stages must be a boolean literal"
                                        .to_string(),
                                )
                                .with_label(arg.span, "expected boolean"),
                            );
                        }
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!(
                                "@pipeline: unknown argument '{}'",
                                aname
                            ))
                            .with_label(arg.span, "unknown argument"),
                        );
                    }
                }
            }
        }
    }

    if num_stages.is_none() {
        diagnostics.push(
            Diagnostic::error("@pipeline: stages is required".to_string())
                .with_label(deco.span, "missing stages"),
        );
        return None;
    }

    Some(PipelineDecoratorConfig {
        num_stages: num_stages.unwrap(),
        schedule,
        checkpoint_stages,
    })
}
