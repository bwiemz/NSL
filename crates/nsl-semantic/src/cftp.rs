//! Semantic validation for CFTP training-pipeline optimisations.
//!
//! Two feature surfaces:
//!
//!   * `@fase(...)` — optional decorator on the `train` block that lets the
//!     user pin a specific FASE mode (`auto` / `deferred` / `full_buffer`
//!     / `off`) and explicitly toggle the AdamW v_t approximation.
//!   * `@pca(...)` — optional decorator on the `dataset` or `train` block
//!     that forces a specific PCA strategy (`auto` / `segment_id` /
//!     `per_document` / `off`).
//!
//! When no decorator is present, the compiler applies FASE/PCA
//! automatically based on the other configuration (grad_accumulation,
//! packing, etc.) — this mirrors the design intent in Section 7 of the
//! CFTP paper.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::{Span, Symbol};
use nsl_errors::Diagnostic;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FaseMode {
    Auto,
    Deferred,
    FullBuffer,
    Off,
}

impl FaseMode {
    pub fn as_str(self) -> &'static str {
        match self {
            FaseMode::Auto => "auto",
            FaseMode::Deferred => "deferred",
            FaseMode::FullBuffer => "full_buffer",
            FaseMode::Off => "off",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(FaseMode::Auto),
            "deferred" => Some(FaseMode::Deferred),
            "full_buffer" | "full" => Some(FaseMode::FullBuffer),
            "off" | "disable" | "disabled" => Some(FaseMode::Off),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct FaseConfig {
    pub mode: FaseMode,
    pub allow_v_approx: bool,
    pub span: Span,
}

pub fn validate_fase_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<FaseConfig> {
    let mut mode = FaseMode::Auto;
    let mut allow_v_approx = true;

    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error("@fase: positional arguments are not allowed".to_string())
                        .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            match aname.as_str() {
                "mode" => {
                    let parsed = match &arg.value.kind {
                        ExprKind::Ident(sym) => FaseMode::parse(&resolve_sym(*sym)),
                        ExprKind::StringLiteral(s) => FaseMode::parse(s),
                        _ => None,
                    };
                    match parsed {
                        Some(m) => mode = m,
                        None => diagnostics.push(
                            Diagnostic::error(
                                "@fase: mode must be auto, deferred, full_buffer, or off"
                                    .to_string(),
                            )
                            .with_label(arg.span, "invalid mode"),
                        ),
                    }
                }
                "allow_v_approx" | "v_approx" => match &arg.value.kind {
                    ExprKind::BoolLiteral(b) => allow_v_approx = *b,
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "@fase: allow_v_approx must be a bool literal".to_string(),
                        )
                        .with_label(arg.span, "expected bool"),
                    ),
                },
                _ => diagnostics.push(
                    Diagnostic::error(format!("@fase: unknown argument '{aname}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }

    Some(FaseConfig {
        mode,
        allow_v_approx,
        span: deco.span,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PcaStrategy {
    Auto,
    SegmentId,
    PerDocument,
    Off,
}

impl PcaStrategy {
    pub fn as_str(self) -> &'static str {
        match self {
            PcaStrategy::Auto => "auto",
            PcaStrategy::SegmentId => "segment_id",
            PcaStrategy::PerDocument => "per_document",
            PcaStrategy::Off => "off",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(PcaStrategy::Auto),
            "segment_id" | "segment" => Some(PcaStrategy::SegmentId),
            "per_document" | "per_doc" | "perdoc" => Some(PcaStrategy::PerDocument),
            "off" | "disable" | "disabled" => Some(PcaStrategy::Off),
            _ => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PcaConfig {
    pub strategy: PcaStrategy,
    pub span: Span,
}

pub fn validate_pca_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<PcaConfig> {
    let mut strategy = PcaStrategy::Auto;
    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error("@pca: positional arguments are not allowed".to_string())
                        .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            match aname.as_str() {
                "strategy" | "mode" => {
                    let parsed = match &arg.value.kind {
                        ExprKind::Ident(sym) => PcaStrategy::parse(&resolve_sym(*sym)),
                        ExprKind::StringLiteral(s) => PcaStrategy::parse(s),
                        _ => None,
                    };
                    match parsed {
                        Some(s) => strategy = s,
                        None => diagnostics.push(
                            Diagnostic::error(
                                "@pca: strategy must be auto, segment_id, per_document, or off"
                                    .to_string(),
                            )
                            .with_label(arg.span, "invalid strategy"),
                        ),
                    }
                }
                _ => diagnostics.push(
                    Diagnostic::error(format!("@pca: unknown argument '{aname}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }
    Some(PcaConfig {
        strategy,
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
    fn fase_mode_roundtrip() {
        for m in [
            FaseMode::Auto,
            FaseMode::Deferred,
            FaseMode::FullBuffer,
            FaseMode::Off,
        ] {
            assert_eq!(FaseMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(FaseMode::parse("nonsense"), None);
    }

    #[test]
    fn pca_strategy_roundtrip() {
        for s in [
            PcaStrategy::Auto,
            PcaStrategy::SegmentId,
            PcaStrategy::PerDocument,
            PcaStrategy::Off,
        ] {
            assert_eq!(PcaStrategy::parse(s.as_str()), Some(s));
        }
        assert_eq!(PcaStrategy::parse("nonsense"), None);
    }
}
