//! Semantic validation for the `@csha` decorator — Compiler-Synthesized
//! Holistic Attention (NSL-CSHA-Research.PDF §6.2).
//!
//! Accepted arguments:
//!   * `level    = 1 | 2 | 3`      — force a specific fusion level
//!   * `target   = <ident|string>` — GPU name (e.g. `h100`, `rtx5070ti`)
//!   * `disable  = true | false`   — opt this model out of CSHA
//!
//! When absent, the compiler picks a level automatically based on the
//! cost model and memory planner (paper §2).
//!
//! `@csha` is valid on `model` declarations.  The semantic layer only
//! checks argument names / value kinds; the heavy lifting (pattern
//! detection, SMEM feasibility, per-layer kernel selection) happens
//! in `nsl-codegen::csha`.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::{Span, Symbol};
use nsl_errors::Diagnostic;

/// The three CSHA fusion levels defined in §2 of the paper.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CshaLevel {
    /// Level 1 — boundary fusion (epilogue/prologue absorption).
    Boundary,
    /// Level 2 — projection-attention pipelining.
    Pipeline,
    /// Level 3 — full transformer-block fusion.
    Block,
}

impl CshaLevel {
    pub fn as_u8(self) -> u8 {
        match self {
            CshaLevel::Boundary => 1,
            CshaLevel::Pipeline => 2,
            CshaLevel::Block => 3,
        }
    }

    pub fn from_u8(n: u8) -> Option<Self> {
        match n {
            1 => Some(CshaLevel::Boundary),
            2 => Some(CshaLevel::Pipeline),
            3 => Some(CshaLevel::Block),
            _ => None,
        }
    }

    pub fn as_str(self) -> &'static str {
        match self {
            CshaLevel::Boundary => "boundary",
            CshaLevel::Pipeline => "pipeline",
            CshaLevel::Block => "block",
        }
    }
}

/// Validated `@csha(...)` configuration.
#[derive(Debug, Clone)]
pub struct CshaConfig {
    /// User-forced level.  `None` = auto.
    pub level: Option<CshaLevel>,
    /// GPU target name as written by the user.  The codegen side looks
    /// this up in `gpu_specs::GPU_DATABASE`.
    pub target: Option<Symbol>,
    /// `@csha(disable=true)` — skip CSHA for this model.
    pub disabled: bool,
    pub span: Span,
}

pub fn validate_csha_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<CshaConfig> {
    let mut level: Option<CshaLevel> = None;
    let mut target: Option<Symbol> = None;
    let mut disabled = false;

    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error("@csha: positional arguments are not allowed".to_string())
                        .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            match aname.as_str() {
                "level" => match &arg.value.kind {
                    ExprKind::IntLiteral(n) => match CshaLevel::from_u8(*n as u8) {
                        Some(l) => level = Some(l),
                        None => diagnostics.push(
                            Diagnostic::error(format!("@csha: level must be 1, 2, or 3 (got {n})"))
                                .with_label(arg.span, "invalid level"),
                        ),
                    },
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@csha: level must be an integer literal (1, 2, or 3)".to_string(),
                        )
                        .with_label(arg.span, "expected integer"),
                    ),
                },
                "target" => match &arg.value.kind {
                    ExprKind::Ident(sym) => target = Some(*sym),
                    ExprKind::StringLiteral(_) => target = Some(*name_sym),
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@csha: target must be an identifier (e.g. h100, rtx5070ti)"
                                .to_string(),
                        )
                        .with_label(arg.span, "expected ident"),
                    ),
                },
                "disable" | "disabled" => match &arg.value.kind {
                    ExprKind::BoolLiteral(b) => disabled = *b,
                    _ => diagnostics.push(
                        Diagnostic::error("@csha: disable must be a boolean literal".to_string())
                            .with_label(arg.span, "expected bool"),
                    ),
                },
                _ => diagnostics.push(
                    Diagnostic::error(format!("@csha: unknown argument '{aname}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }

    // Cross-field validation: `disable=true` with a forced level is
    // inconsistent — warn rather than error to match the tone of the
    // other compiler-native-training decorators.
    if disabled && level.is_some() {
        diagnostics.push(
            Diagnostic::error("@csha: cannot specify both `level` and `disable=true`".to_string())
                .with_label(deco.span, "conflicting arguments"),
        );
    }

    Some(CshaConfig {
        level,
        target,
        disabled,
        span: deco.span,
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn level_roundtrip() {
        for n in 1..=3 {
            let l = CshaLevel::from_u8(n).unwrap();
            assert_eq!(l.as_u8(), n);
        }
        assert!(CshaLevel::from_u8(0).is_none());
        assert!(CshaLevel::from_u8(4).is_none());
    }

    #[test]
    fn level_strings_are_distinct() {
        let names: Vec<&'static str> = [CshaLevel::Boundary, CshaLevel::Pipeline, CshaLevel::Block]
            .into_iter()
            .map(|l| l.as_str())
            .collect();
        let unique: std::collections::HashSet<_> = names.iter().collect();
        assert_eq!(names.len(), unique.len());
    }
}
