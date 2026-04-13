//! Semantic validation for CFIE serve-block options.
//!
//! The CFIE paper uses a `serve` block with the following knobs:
//!
//! ```text
//! serve(model=m, port=8080):
//!     max_seq = 4096
//!     max_batch = 64
//!     kv_layout = static        # or "paged", "auto"
//!     kv_quant = auto           # or "uniform_fp16", "uniform_int8"
//!     speculative:
//!         draft = DraftModel.load("draft.nslm")
//!         tokens = 5
//!         method = tree          # or standard, medusa, lookahead
//!         tree_width = 2
//!     sampling:
//!         temperature = 0.7
//!         top_k = 50
//!         top_p = 0.9
//!         fused = true
//!     grammar:
//!         schema = JsonSchema("schema.json")
//! ```
//!
//! Today the serve block parses these as generic key/value pairs; this
//! module validates the values.  We also accept a `@cfie(mode=...)`
//! decorator on the serve block as a convenience override.

use nsl_ast::decl::Decorator;
use nsl_ast::expr::ExprKind;
use nsl_ast::{Span, Symbol};
use nsl_errors::Diagnostic;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CfieMode {
    Full,
    Sampling,
    Off,
}

impl CfieMode {
    pub fn as_str(self) -> &'static str {
        match self {
            CfieMode::Full => "full",
            CfieMode::Sampling => "sampling",
            CfieMode::Off => "off",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "full" | "auto" => Some(CfieMode::Full),
            "sampling" => Some(CfieMode::Sampling),
            "off" | "disable" | "disabled" => Some(CfieMode::Off),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvLayoutChoice {
    Static,
    Paged,
    Auto,
}

impl KvLayoutChoice {
    pub fn as_str(self) -> &'static str {
        match self {
            KvLayoutChoice::Static => "static",
            KvLayoutChoice::Paged => "paged",
            KvLayoutChoice::Auto => "auto",
        }
    }
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "static" => Some(KvLayoutChoice::Static),
            "paged" => Some(KvLayoutChoice::Paged),
            "auto" => Some(KvLayoutChoice::Auto),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvQuantChoice {
    Auto,
    UniformFp16,
    UniformInt8,
    Off,
}

impl KvQuantChoice {
    pub fn parse(s: &str) -> Option<Self> {
        match s {
            "auto" => Some(KvQuantChoice::Auto),
            "uniform_fp16" | "fp16" => Some(KvQuantChoice::UniformFp16),
            "uniform_int8" | "int8" => Some(KvQuantChoice::UniformInt8),
            "off" | "disable" | "disabled" => Some(KvQuantChoice::Off),
            _ => None,
        }
    }
    pub fn as_str(self) -> &'static str {
        match self {
            KvQuantChoice::Auto => "auto",
            KvQuantChoice::UniformFp16 => "uniform_fp16",
            KvQuantChoice::UniformInt8 => "uniform_int8",
            KvQuantChoice::Off => "off",
        }
    }
}

#[derive(Debug, Clone)]
pub struct CfieConfig {
    pub mode: CfieMode,
    pub target: Option<Symbol>,
    pub span: Span,
}

/// Validate an optional `@cfie(...)` decorator on the serve block.
pub fn validate_cfie_decorator(
    deco: &Decorator,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) -> Option<CfieConfig> {
    let mut mode = CfieMode::Full;
    let mut target: Option<Symbol> = None;
    if let Some(ref args) = deco.args {
        for arg in args {
            let Some(ref name_sym) = arg.name else {
                diagnostics.push(
                    Diagnostic::error(
                        "@cfie: positional arguments are not allowed".to_string(),
                    )
                    .with_label(arg.span, "expected `key = value`"),
                );
                continue;
            };
            let aname = resolve_sym(*name_sym);
            match aname.as_str() {
                "mode" => {
                    let parsed = match &arg.value.kind {
                        ExprKind::Ident(sym) => CfieMode::parse(&resolve_sym(*sym)),
                        ExprKind::StringLiteral(s) => CfieMode::parse(s),
                        _ => None,
                    };
                    match parsed {
                        Some(m) => mode = m,
                        None => diagnostics.push(
                            Diagnostic::error(
                                "@cfie: mode must be full, sampling, or off".to_string(),
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
                            "@cfie: target must be an identifier (e.g. h100)".to_string(),
                        )
                        .with_label(arg.span, "expected ident"),
                    ),
                },
                _ => diagnostics.push(
                    Diagnostic::error(format!("@cfie: unknown argument '{aname}'"))
                        .with_label(arg.span, "unknown argument"),
                ),
            }
        }
    }
    Some(CfieConfig {
        mode,
        target,
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
    fn mode_parse_roundtrip() {
        for m in [CfieMode::Full, CfieMode::Sampling, CfieMode::Off] {
            assert_eq!(CfieMode::parse(m.as_str()), Some(m));
        }
        assert_eq!(CfieMode::parse("auto"), Some(CfieMode::Full));
        assert!(CfieMode::parse("nonsense").is_none());
    }

    #[test]
    fn kv_layout_parse_roundtrip() {
        for c in [
            KvLayoutChoice::Static,
            KvLayoutChoice::Paged,
            KvLayoutChoice::Auto,
        ] {
            assert_eq!(KvLayoutChoice::parse(c.as_str()), Some(c));
        }
        assert!(KvLayoutChoice::parse("nonsense").is_none());
    }

    #[test]
    fn kv_quant_parse_roundtrip() {
        for c in [
            KvQuantChoice::Auto,
            KvQuantChoice::UniformFp16,
            KvQuantChoice::UniformInt8,
            KvQuantChoice::Off,
        ] {
            assert_eq!(KvQuantChoice::parse(c.as_str()), Some(c));
        }
        assert_eq!(KvQuantChoice::parse("fp16"), Some(KvQuantChoice::UniformFp16));
        assert!(KvQuantChoice::parse("nonsense").is_none());
    }
}
