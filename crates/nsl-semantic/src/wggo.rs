//! Semantic validation for the `@wggo` decorator.
//!
//! Accepted arguments:
//!   * `mode = full | greedy | off`
//!   * `target = <ident|string>`
//!
//! `@wggo` is valid on the `train` block.  When absent, WGGO decides
//! whether to activate based on the presence of other training
//! decorators (WRGA + CPDT + CEP).

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::{Span, Symbol};
use nsl_errors::Diagnostic;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WggoMode {
    Full,
    Greedy,
    Off,
}

impl WggoMode {
    pub fn as_str(self) -> &'static str {
        match self {
            WggoMode::Full => "full",
            WggoMode::Greedy => "greedy",
            WggoMode::Off => "off",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "full" | "auto" => Some(WggoMode::Full),
            "greedy" => Some(WggoMode::Greedy),
            "off" | "disable" | "disabled" => Some(WggoMode::Off),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct WggoConfig {
    pub mode: WggoMode,
    pub target: Option<Symbol>,
    pub span: Span,
}

pub fn validate_wggo_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<WggoConfig> {
    let mut mode = WggoMode::Full;
    let mut target: Option<Symbol> = None;

    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error(
                        "@wggo: positional arguments are not allowed".to_string(),
                    )
                    .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            match aname.as_str() {
                "mode" => {
                    let parsed = match &arg.value.kind {
                        ExprKind::Ident(sym) => WggoMode::parse(&resolve_sym(*sym)),
                        ExprKind::StringLiteral(s) => WggoMode::parse(s),
                        _ => None,
                    };
                    match parsed {
                        Some(m) => mode = m,
                        None => diagnostics.push(
                            Diagnostic::error(
                                "@wggo: mode must be full, greedy, or off".to_string(),
                            )
                            .with_label(arg.span, "invalid mode"),
                        ),
                    }
                }
                "target" => match &arg.value.kind {
                    ExprKind::Ident(sym) => target = Some(*sym),
                    ExprKind::StringLiteral(_) => target = Some(*name_sym),
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@wggo: target must be an identifier (e.g. h100)".to_string(),
                        )
                        .with_label(arg.span, "expected ident"),
                    ),
                },
                _ => diagnostics.push(
                    Diagnostic::error(format!("@wggo: unknown argument '{aname}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }

    Some(WggoConfig {
        mode,
        target,
        span: deco.span,
    })
}

// ---------------------------------------------------------------------------
// WGGO Phase 2 Task 2: `@wggo_target` required-arguments validation
// ---------------------------------------------------------------------------

/// The five named arguments `@wggo_target` requires.
///
/// Order matters for the diagnostic message: it's the canonical ordering
/// the user-facing error lists, and the order of `WGGO_TARGET_REQUIRED_ARGS`
/// is the source of truth for that header.
pub const WGGO_TARGET_REQUIRED_ARGS: &[&str] = &["w_q", "w_k", "w_v", "w_o", "head_dim"];

/// Validate that a `@wggo_target` decorator provides all five required
/// named arguments (`w_q`, `w_k`, `w_v`, `w_o`, `head_dim`).
///
/// This task ONLY checks that the names appear as arguments. It does NOT
/// validate that the values are `self.<field>` references (Task 3) or that
/// those fields exist on the model with appropriate types (Task 4).
///
/// Called from both:
///   * `crates/nsl-semantic/src/checker/model.rs` — `ModelMember::Method`
///   * `crates/nsl-semantic/src/checker/stmt.rs` — top-level `StmtKind::FnDef`
///
/// Both call sites guard on the placement check from Task 1 first; this
/// function assumes placement has already been validated.
pub fn validate_wggo_target_required_args(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let provided: std::collections::HashSet<String> = match deco.args {
        Some(ref args) => args
            .iter()
            .filter_map(|a| a.name.map(resolve_sym))
            .collect(),
        None => std::collections::HashSet::new(),
    };
    let missing: Vec<&str> = WGGO_TARGET_REQUIRED_ARGS
        .iter()
        .filter(|a| !provided.contains(**a))
        .copied()
        .collect();
    if !missing.is_empty() {
        diagnostics.push(
            Diagnostic::error(format!(
                "@wggo_target requires arguments: w_q, w_k, w_v, w_o, head_dim; missing: {missing:?}"
            ))
            .with_label(deco.span, "missing required arguments"),
        );
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mode_parse_roundtrip() {
        for m in [WggoMode::Full, WggoMode::Greedy, WggoMode::Off] {
            assert_eq!(WggoMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(WggoMode::parse("auto"), Some(WggoMode::Full));
        assert!(WggoMode::parse("nonsense").is_none());
    }

    #[test]
    fn wggo_target_required_args_is_canonical_five() {
        // Guard against accidental reordering or additions: the diagnostic
        // message header lists these in a fixed order.
        assert_eq!(
            WGGO_TARGET_REQUIRED_ARGS,
            &["w_q", "w_k", "w_v", "w_o", "head_dim"]
        );
    }
}
