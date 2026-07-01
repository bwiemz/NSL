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
// Serve-block CFIE config validation
// ---------------------------------------------------------------------------

/// Known nested serve sections and the keys each accepts.
const KNOWN_SECTIONS: &[&str] = &["sampling", "speculative", "grammar"];

fn expr_as_name(
    value: &nsl_ast::expr::Expr,
    resolve_sym: &dyn Fn(Symbol) -> String,
) -> Option<String> {
    match &value.kind {
        ExprKind::StringLiteral(s) => Some(s.clone()),
        ExprKind::Ident(sym) => Some(resolve_sym(*sym)),
        _ => None,
    }
}

fn expr_as_f64(value: &nsl_ast::expr::Expr) -> Option<f64> {
    match &value.kind {
        ExprKind::FloatLiteral(f) => Some(*f),
        ExprKind::IntLiteral(n) => Some(*n as f64),
        _ => None,
    }
}

fn expr_as_i64(value: &nsl_ast::expr::Expr) -> Option<i64> {
    match &value.kind {
        ExprKind::IntLiteral(n) => Some(*n),
        _ => None,
    }
}

fn expr_as_bool(value: &nsl_ast::expr::Expr) -> Option<bool> {
    match &value.kind {
        ExprKind::BoolLiteral(b) => Some(*b),
        _ => None,
    }
}

/// Validate CFIE-specific serve-block configuration: the flat
/// `kv_layout` / `kv_quant` / `max_seq` / `target_gpu` keys and the
/// nested `sampling:` / `speculative:` / `grammar:` sections.
///
/// Non-CFIE keys are left alone (they are validated by the M29/M41
/// serve checks); unknown *sections* are hard errors because a nested
/// section that silently does nothing is exactly the failure mode the
/// CFIE audit flagged.
pub fn validate_serve_config(
    serve: &nsl_ast::block::ServeBlock,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for entry in &serve.config {
        let key = resolve_sym(entry.key);
        match key.as_str() {
            "kv_layout" => {
                let parsed = expr_as_name(&entry.value, resolve_sym)
                    .and_then(|s| KvLayoutChoice::parse(&s));
                if parsed.is_none() {
                    diagnostics.push(
                        Diagnostic::error(
                            "kv_layout must be \"static\", \"paged\", or \"auto\"".to_string(),
                        )
                        .with_label(entry.value.span, "invalid kv_layout"),
                    );
                }
            }
            "kv_quant" => {
                let parsed = expr_as_name(&entry.value, resolve_sym)
                    .and_then(|s| KvQuantChoice::parse(&s));
                if parsed.is_none() {
                    diagnostics.push(
                        Diagnostic::error(
                            "kv_quant must be \"auto\", \"uniform_fp16\", \"uniform_int8\", or \"off\""
                                .to_string(),
                        )
                        .with_label(entry.value.span, "invalid kv_quant"),
                    );
                }
            }
            "max_seq" => {
                if !matches!(expr_as_i64(&entry.value), Some(v) if v >= 1) {
                    diagnostics.push(
                        Diagnostic::error("max_seq must be a positive integer".to_string())
                            .with_label(entry.value.span, "invalid max_seq"),
                    );
                }
            }
            "target_gpu" => {
                if expr_as_name(&entry.value, resolve_sym).is_none() {
                    diagnostics.push(
                        Diagnostic::error(
                            "target_gpu must be a GPU name string (e.g. \"h100\")".to_string(),
                        )
                        .with_label(entry.value.span, "invalid target_gpu"),
                    );
                }
            }
            _ => {}
        }
    }

    for sub in &serve.sub_blocks {
        let section = resolve_sym(sub.key);
        match section.as_str() {
            "sampling" => validate_sampling_section(sub, resolve_sym, diagnostics),
            "speculative" => validate_speculative_section(sub, resolve_sym, diagnostics),
            "grammar" => validate_grammar_section(sub, resolve_sym, diagnostics),
            other => {
                diagnostics.push(
                    Diagnostic::error(format!(
                        "unknown serve section '{other}:'; expected one of: {}",
                        KNOWN_SECTIONS.join(", ")
                    ))
                    .with_label(sub.span, "unknown section"),
                );
            }
        }
    }
}

fn validate_sampling_section(
    sub: &nsl_ast::block::ServeSubBlock,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) {
    for entry in &sub.entries {
        let key = resolve_sym(entry.key);
        match key.as_str() {
            "temperature" => {
                if !matches!(expr_as_f64(&entry.value), Some(t) if (0.0..=10.0).contains(&t)) {
                    diagnostics.push(
                        Diagnostic::error(
                            "sampling temperature must be a number in [0, 10]".to_string(),
                        )
                        .with_label(entry.value.span, "invalid temperature"),
                    );
                }
            }
            "top_k" => {
                if !matches!(expr_as_i64(&entry.value), Some(k) if (1..=1024).contains(&k)) {
                    diagnostics.push(
                        Diagnostic::error(
                            "sampling top_k must be an integer in [1, 1024]".to_string(),
                        )
                        .with_label(entry.value.span, "invalid top_k"),
                    );
                }
            }
            "top_p" => {
                if !matches!(expr_as_f64(&entry.value), Some(p) if p > 0.0 && p <= 1.0) {
                    diagnostics.push(
                        Diagnostic::error("sampling top_p must be a number in (0, 1]".to_string())
                            .with_label(entry.value.span, "invalid top_p"),
                    );
                }
            }
            "fused" => {
                if expr_as_bool(&entry.value).is_none() {
                    diagnostics.push(
                        Diagnostic::error("sampling fused must be true or false".to_string())
                            .with_label(entry.value.span, "invalid fused"),
                    );
                }
            }
            other => {
                diagnostics.push(
                    Diagnostic::error(format!(
                        "unknown sampling key '{other}'; expected temperature, top_k, top_p, or fused"
                    ))
                    .with_label(entry.span, "unknown key"),
                );
            }
        }
    }
}

fn validate_speculative_section(
    sub: &nsl_ast::block::ServeSubBlock,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut method: Option<String> = None;
    let mut has_tree_width = false;
    for entry in &sub.entries {
        let key = resolve_sym(entry.key);
        match key.as_str() {
            "draft" => {
                // A path to a compiled draft model (`"draft.nslm"`).
                // Richer forms (DraftModel.load(...)) are a codegen
                // concern once compiled-speculative kernels land.
                if !matches!(&entry.value.kind, ExprKind::StringLiteral(_)) {
                    diagnostics.push(
                        Diagnostic::error(
                            "speculative draft must be a .nslm path string".to_string(),
                        )
                        .with_label(entry.value.span, "invalid draft"),
                    );
                }
            }
            "tokens" => {
                if !matches!(expr_as_i64(&entry.value), Some(k) if (1..=32).contains(&k)) {
                    diagnostics.push(
                        Diagnostic::error(
                            "speculative tokens must be an integer in [1, 32]".to_string(),
                        )
                        .with_label(entry.value.span, "invalid tokens"),
                    );
                }
            }
            "method" => {
                let parsed = expr_as_name(&entry.value, resolve_sym);
                match parsed.as_deref() {
                    Some("standard") | Some("tree") | Some("medusa") | Some("lookahead") => {
                        method = parsed;
                    }
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "speculative method must be \"standard\", \"tree\", \"medusa\", or \"lookahead\""
                                .to_string(),
                        )
                        .with_label(entry.value.span, "invalid method"),
                    ),
                }
            }
            "tree_width" => {
                has_tree_width = true;
                if !matches!(expr_as_i64(&entry.value), Some(w) if (1..=8).contains(&w)) {
                    diagnostics.push(
                        Diagnostic::error(
                            "speculative tree_width must be an integer in [1, 8]".to_string(),
                        )
                        .with_label(entry.value.span, "invalid tree_width"),
                    );
                }
            }
            "temperature" => {
                if !matches!(expr_as_f64(&entry.value), Some(t) if (0.0..=10.0).contains(&t)) {
                    diagnostics.push(
                        Diagnostic::error(
                            "speculative temperature must be a number in [0, 10]".to_string(),
                        )
                        .with_label(entry.value.span, "invalid temperature"),
                    );
                }
            }
            other => {
                diagnostics.push(
                    Diagnostic::error(format!(
                        "unknown speculative key '{other}'; expected draft, tokens, method, tree_width, or temperature"
                    ))
                    .with_label(entry.span, "unknown key"),
                );
            }
        }
    }
    if has_tree_width && method.as_deref() != Some("tree") {
        diagnostics.push(
            Diagnostic::warning(
                "speculative tree_width has no effect unless method = \"tree\"".to_string(),
            )
            .with_label(sub.span, "tree_width without tree method"),
        );
    }
}

fn validate_grammar_section(
    sub: &nsl_ast::block::ServeSubBlock,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let mut has_schema = false;
    for entry in &sub.entries {
        let key = resolve_sym(entry.key);
        match key.as_str() {
            "schema" => {
                has_schema = true;
                if !matches!(&entry.value.kind, ExprKind::StringLiteral(_)) {
                    diagnostics.push(
                        Diagnostic::error(
                            "grammar schema must be a JSON-schema path string".to_string(),
                        )
                        .with_label(entry.value.span, "invalid schema"),
                    );
                }
            }
            other => {
                diagnostics.push(
                    Diagnostic::error(format!("unknown grammar key '{other}'; expected schema"))
                        .with_label(entry.span, "unknown key"),
                );
            }
        }
    }
    if !has_schema {
        diagnostics.push(
            Diagnostic::error("grammar section requires a schema key".to_string())
                .with_label(sub.span, "missing schema"),
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
