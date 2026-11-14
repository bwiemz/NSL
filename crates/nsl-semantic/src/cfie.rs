//! Semantic validation for CFIE serve-block options.
//!
//! A CFIE `serve` block accepts these knobs:
//!
//! ```text
//! serve Inference:
//!     max_seq: 4096
//!     max_batch: 64
//!     kv_layout: "static"       # or "paged", "auto"
//!     kv_quant: "uniform_fp16"  # or "auto" (advisory report), "off"
//!     speculative:
//!         draft_weights: "draft.safetensors"
//!         tokens: 4
//!         method: "standard"
//!     sampling:
//!         temperature: 0.7
//!         top_k: 50
//!         top_p: 0.9
//!         seed: 0
//!         fused: true
//! ```
//!
//! The serve block parses these as generic key/value pairs; this module
//! validates the values, and refuses the requests nothing would honour:
//! a `grammar:` section, INT8 KV, a non-zero sampling temperature with no
//! `top_k`, a `top_k` above 64, and every `speculative:` setting other
//! than a greedy draft model drafting a linear chain. We also accept a
//! `@cfie(mode=...)` decorator on the serve block as a convenience
//! override.

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

/// Known nested serve sections.
const KNOWN_SECTIONS: &[&str] = &["sampling", "speculative"];

/// The fused sampler's candidate-list capacity (`cfie_sample_ptx`).
const MAX_TOP_K: i64 = 64;

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
                if parsed == Some(KvQuantChoice::UniformInt8) {
                    diagnostics.push(
                        Diagnostic::error(
                            "kv_quant: INT8 KV is not supported — no kernel quantizes keys or \
                             values into the cache, so an INT8 pool could never be filled; use \
                             \"uniform_fp16\" or drop the key"
                                .to_string(),
                        )
                        .with_label(entry.value.span, "unsupported kv_quant"),
                    );
                } else if parsed.is_none() {
                    diagnostics.push(
                        Diagnostic::error(
                            "kv_quant must be \"auto\", \"uniform_fp16\", \"uniform_int8\", or \"off\""
                                .to_string(),
                        )
                        .with_label(entry.value.span, "invalid kv_quant"),
                    );
                }
            }
            // Guard-collapsed (clippy::collapsible_match): a valid value falls
            // through to the `_ => {}` no-op arm, identical to the prior
            // arm-body `if`. Behavior-preserving.
            "max_seq" if !matches!(expr_as_i64(&entry.value), Some(v) if v >= 1) => {
                diagnostics.push(
                    Diagnostic::error("max_seq must be a positive integer".to_string())
                        .with_label(entry.value.span, "invalid max_seq"),
                );
            }
            "target_gpu" if expr_as_name(&entry.value, resolve_sym).is_none() => {
                diagnostics.push(
                    Diagnostic::error(
                        "target_gpu must be a GPU name string (e.g. \"h100\")".to_string(),
                    )
                    .with_label(entry.value.span, "invalid target_gpu"),
                );
            }
            _ => {}
        }
    }

    for sub in &serve.sub_blocks {
        let section = resolve_sym(sub.key);
        match section.as_str() {
            "sampling" => validate_sampling_section(sub, resolve_sym, diagnostics),
            "speculative" => validate_speculative_section(sub, resolve_sym, diagnostics),
            "grammar" => {
                // Refused, not ignored: the build used to bake the
                // grammar's token masks into the sampler and then apply
                // the start state's mask to every token.
                diagnostics.push(
                    Diagnostic::error(
                        "serve `grammar:` is not supported: the decode loop does not advance \
                         the grammar's automaton from token to token and has no \
                         end-of-sequence rule, so every token would be held to the set \
                         allowed at the start. Remove the section"
                            .to_string(),
                    )
                    .with_label(sub.span, "unsupported section"),
                );
            }
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
    let mut temperature: Option<f64> = None;
    let mut has_top_k = false;
    for entry in &sub.entries {
        let key = resolve_sym(entry.key);
        match key.as_str() {
            "temperature" => {
                temperature = expr_as_f64(&entry.value);
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
                has_top_k = true;
                // 64 is the fused sampler's candidate-list capacity. A
                // larger top_k used to be accepted here and then skip the
                // sampler kernel with only a note in the build report.
                if !matches!(expr_as_i64(&entry.value), Some(k) if (1..=MAX_TOP_K).contains(&k)) {
                    diagnostics.push(
                        Diagnostic::error(format!(
                            "sampling top_k must be an integer in [1, {MAX_TOP_K}] (the fused \
                             sampler keeps at most {MAX_TOP_K} candidates)"
                        ))
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
            "seed" => {
                if !matches!(expr_as_i64(&entry.value), Some(s) if s >= 0) {
                    diagnostics.push(
                        Diagnostic::error(
                            "sampling seed must be a non-negative integer".to_string(),
                        )
                        .with_label(entry.value.span, "invalid seed"),
                    );
                }
            }
            other => {
                diagnostics.push(
                    Diagnostic::error(format!(
                        "unknown sampling key '{other}'; expected temperature, top_k, top_p, seed, or fused"
                    ))
                    .with_label(entry.span, "unknown key"),
                );
            }
        }
    }
    // A non-zero temperature with no top_k has no meaning the sampler can
    // honour: it cannot draw from the whole vocabulary. It used to get
    // top_k = 50 and top_p = 0.9 without notice.
    if matches!(temperature, Some(t) if t != 0.0) && !has_top_k {
        diagnostics.push(
            Diagnostic::error(format!(
                "sampling: a non-zero temperature needs top_k (1..={MAX_TOP_K}); the fused \
                 sampler draws from a candidate list and will not choose a top_k for you. \
                 Add top_k, or set temperature: 0.0 for greedy decoding"
            ))
            .with_label(sub.span, "temperature without top_k"),
        );
    }
}

fn validate_speculative_section(
    sub: &nsl_ast::block::ServeSubBlock,
    resolve_sym: &dyn Fn(Symbol) -> String,
    diagnostics: &mut Vec<Diagnostic>,
) {
    // One configuration has a driver: a greedy draft model compiled into
    // the binary (`draft_weights`), proposing a linear chain of `tokens`
    // tokens. The other settings this section used to accept were stored
    // and never read; each is now an error that says so.
    let unsupported = |diagnostics: &mut Vec<Diagnostic>, span: Span, what: String| {
        diagnostics.push(Diagnostic::error(what).with_label(span, "not supported"));
    };
    let mut has_draft_weights = false;
    for entry in &sub.entries {
        let key = resolve_sym(entry.key);
        match key.as_str() {
            "draft" => unsupported(
                diagnostics,
                entry.span,
                "speculative `draft` is not supported: nothing loads it. Name the draft \
                 checkpoint with draft_weights: \"<path.safetensors>\" and the draft_* shape \
                 keys"
                    .to_string(),
            ),
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
                    Some("standard") => {}
                    Some(m @ ("tree" | "medusa" | "lookahead")) => unsupported(
                        diagnostics,
                        entry.value.span,
                        format!(
                            "speculative method \"{m}\" is not supported: the driver drafts a \
                             linear chain of tokens (\"standard\"); tree, medusa and lookahead \
                             drafting have no driver"
                        ),
                    ),
                    _ => diagnostics.push(
                        Diagnostic::error(
                            "speculative method must be \"standard\"".to_string(),
                        )
                        .with_label(entry.value.span, "invalid method"),
                    ),
                }
            }
            "tree_width" => unsupported(
                diagnostics,
                entry.span,
                "speculative tree_width is not supported: there is no tree drafting".to_string(),
            ),
            "temperature" => {
                // 0 says what the draft does anyway.
                if !matches!(expr_as_f64(&entry.value), Some(t) if t == 0.0) {
                    unsupported(
                        diagnostics,
                        entry.value.span,
                        "speculative temperature is not supported: the draft model is greedy. \
                         (The target's sampling is set in the sampling: section, and the \
                         output follows it.)"
                            .to_string(),
                    );
                }
            }
            // A draft model compiled into the same binary. `draft_weights`
            // names the draft checkpoint; the draft_* shape keys mirror the
            // top-level architecture keys (vocab is the target's).
            "draft_weights" => {
                has_draft_weights = true;
                if !matches!(&entry.value.kind, ExprKind::StringLiteral(_)) {
                    diagnostics.push(
                        Diagnostic::error(
                            "speculative draft_weights must be a .safetensors path string"
                                .to_string(),
                        )
                        .with_label(entry.value.span, "invalid draft_weights"),
                    );
                }
            }
            "draft_n_layers" | "draft_d_model" | "draft_n_heads" | "draft_n_kv_heads"
            | "draft_head_dim" | "draft_d_ff" => {
                if !matches!(expr_as_i64(&entry.value), Some(v) if v >= 1) {
                    diagnostics.push(
                        Diagnostic::error(format!(
                            "speculative {key} must be a positive integer"
                        ))
                        .with_label(entry.value.span, "invalid draft shape key"),
                    );
                }
            }
            other => {
                diagnostics.push(
                    Diagnostic::error(format!(
                        "unknown speculative key '{other}'; expected tokens, method, \
                         draft_weights, or a draft_* shape key (draft_n_layers, draft_d_model, \
                         draft_n_heads, draft_n_kv_heads, draft_head_dim, draft_d_ff)"
                    ))
                    .with_label(entry.span, "unknown key"),
                );
            }
        }
    }
    if !has_draft_weights {
        diagnostics.push(
            Diagnostic::error(
                "speculative: the section needs draft_weights: \"<path.safetensors>\". Without \
                 a draft model there is nothing to speculate with; generate() would run plain \
                 decoding while the build report described speculation"
                    .to_string(),
            )
            .with_label(sub.span, "no draft model"),
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
