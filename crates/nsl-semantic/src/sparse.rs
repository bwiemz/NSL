//! M50: Sparse tensor semantic validation — `@sparse` decorator.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::Symbol;
use nsl_errors::Diagnostic;

/// Parsed configuration from `@sparse(pattern="2:4")` or `@sparse(pattern="block", block_size=64)`.
pub struct SparseConfig {
    pub pattern: String,      // "2:4", "block"
    pub block_size: usize,    // for pattern="block" (default 16)
}

/// Validate `@sparse` decorator arguments.
///
/// Required: `pattern` (string, must be "2:4" or "block").
/// Optional: `block_size` (integer, default 16, only valid when pattern="block").
///
/// Returns `Some(SparseConfig)` on success, `None` if validation fails.
pub fn validate_sparse_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<SparseConfig> {
    let mut pattern: Option<String> = None;
    let mut block_size: usize = 16;

    if let Some(ref args) = deco.args {
        for arg in args {
            if let Some(ref name_sym) = arg.name {
                let aname = resolve_sym(*name_sym);
                match aname.as_str() {
                    "pattern" => {
                        if let ExprKind::StringLiteral(s) = &arg.value.kind {
                            match s.as_str() {
                                "2:4" | "block" => {
                                    pattern = Some(s.clone());
                                }
                                _ => {
                                    diagnostics.push(
                                        Diagnostic::error(format!(
                                            "@sparse: pattern must be \"2:4\" or \"block\", got \"{}\"",
                                            s
                                        ))
                                        .with_label(arg.span, "invalid pattern"),
                                    );
                                }
                            }
                        } else {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@sparse: pattern must be a string literal".to_string(),
                                )
                                .with_label(arg.span, "expected string"),
                            );
                        }
                    }
                    "block_size" => {
                        if let ExprKind::IntLiteral(n) = &arg.value.kind {
                            if *n <= 0 {
                                diagnostics.push(
                                    Diagnostic::error(
                                        "@sparse: block_size must be a positive integer".to_string(),
                                    )
                                    .with_label(arg.span, "must be > 0"),
                                );
                            } else {
                                block_size = *n as usize;
                            }
                        } else {
                            diagnostics.push(
                                Diagnostic::error(
                                    "@sparse: block_size must be an integer literal".to_string(),
                                )
                                .with_label(arg.span, "expected integer"),
                            );
                        }
                    }
                    _ => {
                        diagnostics.push(
                            Diagnostic::error(format!("@sparse: unknown argument '{}'", aname))
                                .with_label(arg.span, "unknown argument"),
                        );
                    }
                }
            }
        }
    }

    if pattern.is_none() {
        diagnostics.push(
            Diagnostic::error("@sparse: pattern is required".to_string())
                .with_label(deco.span, "missing pattern"),
        );
        return None;
    }

    let pat = pattern.unwrap();

    // block_size only valid with pattern="block"
    if pat != "block" && block_size != 16 {
        diagnostics.push(
            Diagnostic::error(
                "@sparse: block_size is only valid with pattern=\"block\"".to_string(),
            )
            .with_label(deco.span, "block_size not applicable"),
        );
    }

    Some(SparseConfig {
        pattern: pat,
        block_size,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use nsl_ast::decl::Decorator;
    use nsl_ast::expr::{Arg, Expr, ExprKind};
    use nsl_ast::Span;
    use nsl_errors::{BytePos, FileId};

    const ZERO_SPAN: Span = Span { file_id: FileId(0), start: BytePos(0), end: BytePos(0) };

    fn make_sym(n: u32) -> Symbol {
        // Safety: we construct test symbols directly; in real code the interner handles this
        Symbol(unsafe { std::mem::transmute::<u32, string_interner::DefaultSymbol>(n) })
    }

    fn make_string_arg(name_id: u32, value: &str) -> Arg {
        Arg {
            name: Some(make_sym(name_id)),
            value: Expr {
                kind: ExprKind::StringLiteral(value.to_string()),
                span: ZERO_SPAN,
                id: nsl_ast::NodeId(0),
            },
            span: ZERO_SPAN,
        }
    }

    fn make_int_arg(name_id: u32, value: i64) -> Arg {
        Arg {
            name: Some(make_sym(name_id)),
            value: Expr {
                kind: ExprKind::IntLiteral(value),
                span: ZERO_SPAN,
                id: nsl_ast::NodeId(0),
            },
            span: ZERO_SPAN,
        }
    }

    fn resolver(sym: Symbol) -> String {
        // Map test symbol IDs to names
        match unsafe { std::mem::transmute::<string_interner::DefaultSymbol, u32>(sym.0) } {
            10 => "pattern".to_string(),
            11 => "block_size".to_string(),
            _ => "unknown".to_string(),
        }
    }

    #[test]
    fn valid_two_four_pattern() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![make_string_arg(10, "2:4")]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let result = validate_sparse_decorator(&deco, &resolver, &mut diags);
        assert!(result.is_some());
        assert!(diags.is_empty());
        let cfg = result.unwrap();
        assert_eq!(cfg.pattern, "2:4");
        assert_eq!(cfg.block_size, 16); // default
    }

    #[test]
    fn valid_block_pattern_with_size() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![
                make_string_arg(10, "block"),
                make_int_arg(11, 64),
            ]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let result = validate_sparse_decorator(&deco, &resolver, &mut diags);
        assert!(result.is_some());
        assert!(diags.is_empty());
        let cfg = result.unwrap();
        assert_eq!(cfg.pattern, "block");
        assert_eq!(cfg.block_size, 64);
    }

    #[test]
    fn missing_pattern_error() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let result = validate_sparse_decorator(&deco, &resolver, &mut diags);
        assert!(result.is_none());
        assert!(!diags.is_empty());
    }

    #[test]
    fn invalid_pattern_value() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![make_string_arg(10, "dense")]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let result = validate_sparse_decorator(&deco, &resolver, &mut diags);
        // pattern is invalid, so pattern stays None -> missing pattern error too
        assert!(result.is_none());
        assert!(diags.len() >= 1);
    }

    #[test]
    fn block_size_on_non_block_warns() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![
                make_string_arg(10, "2:4"),
                make_int_arg(11, 32),
            ]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let result = validate_sparse_decorator(&deco, &resolver, &mut diags);
        assert!(result.is_some()); // config is still returned
        assert!(!diags.is_empty()); // but with a diagnostic
    }

    #[test]
    fn unknown_arg_error() {
        let deco = Decorator {
            name: vec![make_sym(0)],
            args: Some(vec![
                make_string_arg(10, "2:4"),
                make_int_arg(99, 42), // resolver returns "unknown"
            ]),
            span: ZERO_SPAN,
        };
        let mut diags = vec![];
        let result = validate_sparse_decorator(&deco, &resolver, &mut diags);
        assert!(result.is_some());
        assert!(!diags.is_empty());
    }
}
